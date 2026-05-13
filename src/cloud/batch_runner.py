"""Submit-and-poll runner for the hmr_world stage in batch mode.

``BatchRunner`` is the only non-local-mode code the stage touches. It
uploads shared per-shot inputs once, writes a per-job ``JobManifest`` to
S3, submits a single Batch array job, polls until the array completes,
then downloads outputs back to ``output/hmr_world/``.

The runner is intentionally easy to test: it accepts injected
``S3Client`` and ``BatchClient`` (a Protocol with the four boto3 methods
it needs). Production code wires real boto3 clients.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.cloud.manifest import ArrayManifest, JobManifest
from src.cloud.s3_io import Boto3S3Client, S3Client


# Statuses a Batch job can move through. Terminal: SUCCEEDED, FAILED.
_TERMINAL = {"SUCCEEDED", "FAILED"}


class BatchClient(Protocol):
    """Minimal boto3 ``batch`` surface the runner uses."""

    def submit_job(self, **kwargs: Any) -> dict[str, Any]: ...

    def describe_jobs(self, **kwargs: Any) -> dict[str, Any]: ...


def _make_run_id() -> str:
    """Sortable, unique-per-run identifier (timestamp + short uuid)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{ts}-{suffix}"


@dataclass(frozen=True)
class _JobSpec:
    """One element of the array job — what the orchestrator submitted."""

    shot_id: str
    player_id: str
    manifest_uri: str


class BatchRunner:
    """Orchestrate the fan-out from the stage's player loop.

    Constructor takes the merged ``hmr_world`` config dict and the
    pipeline output_dir. ``run_tracks`` is the one entry point —
    everything else is helper machinery.
    """

    def __init__(
        self,
        *,
        cfg: dict[str, Any],
        output_dir: Path,
        s3_client: S3Client | None = None,
        batch_client: BatchClient | None = None,
    ) -> None:
        batch_cfg = cfg.get("batch", {}) or {}
        self._cfg = cfg
        self._batch_cfg = batch_cfg
        self._output_dir = Path(output_dir)
        self._region = str(batch_cfg.get("region") or "us-east-1")
        self._bucket = str(batch_cfg.get("s3_bucket") or "")
        self._prefix = str(batch_cfg.get("s3_prefix") or "runs")
        self._job_queue = str(batch_cfg.get("job_queue") or "")
        self._job_definition = str(batch_cfg.get("job_definition") or "")
        self._poll_seconds = max(1, int(batch_cfg.get("poll_seconds", 15)))
        self._failure_policy = str(
            batch_cfg.get("failure_policy", "continue")
        ).lower()
        self._max_concurrent = int(batch_cfg.get("max_concurrent_jobs", 24))
        self._job_timeout_seconds = int(
            batch_cfg.get("job_timeout_seconds", 600)
        )

        self._s3 = s3_client
        self._batch = batch_client

    # --- public --------------------------------------------------------

    def run_tracks(
        self,
        *,
        ordered: list[tuple[tuple[str, str], list[tuple[int, tuple[int, int, int, int]]]]],
        camera_tracks_by_shot: dict[str, Any],
        out_dir: Path,
    ) -> None:
        """Run every ``(shot_id, player_id)`` in ``ordered`` via Batch.

        Outputs land in ``out_dir`` on local disk; subsequent stages
        consume them exactly as today.
        """
        if not ordered:
            print("[hmr_world] batch runner: no tracks to process")
            return

        self._validate_config()
        out_dir.mkdir(parents=True, exist_ok=True)

        s3 = self._s3 or Boto3S3Client(region=self._region)
        batch = self._batch or self._make_boto3_batch()

        run_id = self._ensure_run_id()
        run_root = f"s3://{self._bucket}/{self._prefix}/{run_id}"
        print(f"[hmr_world] batch runner: run_id={run_id} bucket={self._bucket}")

        # Upload one shot.mp4 + one camera_track.json per shot.
        shot_uris, camera_uris = self._upload_shared_inputs(
            ordered=ordered,
            camera_tracks_by_shot=camera_tracks_by_shot,
            run_root=run_root,
            s3=s3,
        )

        # Build per-job manifests, then the array manifest.
        job_specs = self._write_manifests(
            ordered=ordered,
            run_id=run_id,
            run_root=run_root,
            shot_uris=shot_uris,
            camera_uris=camera_uris,
            s3=s3,
        )
        array_manifest_uri = f"{run_root}/jobs/array_index.json"
        s3.upload_bytes(
            ArrayManifest(
                run_id=run_id,
                entries=tuple(spec.manifest_uri for spec in job_specs),
            ).to_json().encode("utf-8"),
            array_manifest_uri,
        )

        # Submit one array job (or a single job if there's only one).
        job_id = self._submit_array_job(
            batch=batch,
            run_id=run_id,
            array_manifest_uri=array_manifest_uri,
            manifest_uri_for_index_0=job_specs[0].manifest_uri,
            n_jobs=len(job_specs),
        )

        # Poll until every child terminates.
        self._poll_until_done(batch=batch, job_id=job_id, n_jobs=len(job_specs))

        # Download outputs; each spec's per-job status.json is the source
        # of truth for ok/error.
        succeeded, failed = self._download_outputs(
            job_specs=job_specs,
            run_root=run_root,
            out_dir=out_dir,
            s3=s3,
        )

        self._print_summary(succeeded, failed, run_root)
        if failed and self._failure_policy == "abort":
            raise RuntimeError(
                f"hmr_world batch run had {len(failed)} failure(s); "
                f"failure_policy=abort"
            )

    # --- internals -----------------------------------------------------

    def _make_boto3_batch(self) -> BatchClient:
        import boto3
        return boto3.client("batch", region_name=self._region)

    def _validate_config(self) -> None:
        missing = [
            k for k, v in {
                "s3_bucket": self._bucket,
                "job_queue": self._job_queue,
                "job_definition": self._job_definition,
            }.items()
            if not v
        ]
        if missing:
            raise RuntimeError(
                f"hmr_world.batch config missing: {', '.join(missing)} — "
                f"set via Terraform outputs or env vars and re-run"
            )

    def _ensure_run_id(self) -> str:
        run_id_path = self._output_dir / ".run_id"
        if run_id_path.exists():
            return run_id_path.read_text().strip()
        run_id = _make_run_id()
        run_id_path.write_text(run_id)
        return run_id

    def _upload_shared_inputs(
        self,
        *,
        ordered: list[tuple[tuple[str, str], list[tuple[int, tuple[int, int, int, int]]]]],
        camera_tracks_by_shot: dict[str, Any],
        run_root: str,
        s3: S3Client,
    ) -> tuple[dict[str, str], dict[str, str]]:
        shots_dir = self._output_dir / "shots"
        camera_dir = self._output_dir / "camera"
        shot_uris: dict[str, str] = {}
        camera_uris: dict[str, str] = {}
        seen_shots: set[str] = set()
        for (shot_id, _player_id), _frames in ordered:
            if shot_id in seen_shots:
                continue
            seen_shots.add(shot_id)
            shot_path = shots_dir / f"{shot_id}.mp4"
            cam_path = camera_dir / f"{shot_id}_camera_track.json"
            if not shot_path.exists():
                raise RuntimeError(
                    f"hmr_world batch: shot clip not found at {shot_path}"
                )
            if not cam_path.exists():
                raise RuntimeError(
                    f"hmr_world batch: camera track not found at {cam_path}"
                )
            shot_uri = f"{run_root}/shots/{shot_id}.mp4"
            cam_uri = f"{run_root}/camera/{shot_id}_camera_track.json"
            s3.upload(shot_path, shot_uri)
            s3.upload(cam_path, cam_uri)
            shot_uris[shot_id] = shot_uri
            camera_uris[shot_id] = cam_uri
        print(
            f"[hmr_world] batch runner: uploaded {len(shot_uris)} shot(s) "
            f"+ camera track(s) to s3"
        )
        return shot_uris, camera_uris

    def _write_manifests(
        self,
        *,
        ordered: list[tuple[tuple[str, str], list[tuple[int, tuple[int, int, int, int]]]]],
        run_id: str,
        run_root: str,
        shot_uris: dict[str, str],
        camera_uris: dict[str, str],
        s3: S3Client,
    ) -> list[_JobSpec]:
        # Strip the bucket-level keys; the handler reads only what it needs.
        hmr_cfg_subset = {
            k: v for k, v in self._cfg.items()
            if k != "batch" and k != "runner"
        }
        specs: list[_JobSpec] = []
        for (shot_id, player_id), frames in ordered:
            output_prefix = f"{run_root}/jobs/{shot_id}__{player_id}"
            manifest = JobManifest(
                run_id=run_id,
                shot_id=shot_id,
                player_id=player_id,
                video_uri=shot_uris[shot_id],
                camera_track_uri=camera_uris[shot_id],
                track_frames=tuple(
                    (int(fi), tuple(int(x) for x in bbox))
                    for fi, bbox in frames
                ),
                hmr_world_cfg=hmr_cfg_subset,
                output_prefix=output_prefix,
            )
            uri = f"{output_prefix}/input.json"
            s3.upload_bytes(manifest.to_json().encode("utf-8"), uri)
            specs.append(_JobSpec(shot_id=shot_id, player_id=player_id, manifest_uri=uri))
        return specs

    def _submit_array_job(
        self,
        *,
        batch: BatchClient,
        run_id: str,
        array_manifest_uri: str,
        manifest_uri_for_index_0: str,
        n_jobs: int,
    ) -> str:
        # Array jobs ≥ 2 read AWS_BATCH_JOB_ARRAY_INDEX, dereference the
        # ArrayManifest at JOB_MANIFEST_S3, and use each per-index
        # JobManifest's own output_prefix. For n=1 we submit a single
        # (non-array) job pointing directly at that one JobManifest URI;
        # Batch rejects arrays of size 1.
        kwargs: dict[str, Any] = {
            "jobName": f"hmr-world-{run_id}",
            "jobQueue": self._job_queue,
            "jobDefinition": self._job_definition,
            "tags": {"RunId": run_id, "Project": "football-perspectives"},
        }
        if n_jobs == 1:
            manifest_uri = manifest_uri_for_index_0
            kwargs["containerOverrides"] = {
                "environment": [
                    {"name": "JOB_MANIFEST_S3", "value": manifest_uri},
                ],
            }
        else:
            kwargs["arrayProperties"] = {"size": n_jobs}
            kwargs["containerOverrides"] = {
                "environment": [
                    {"name": "JOB_MANIFEST_S3", "value": array_manifest_uri},
                ],
            }
        response = batch.submit_job(**kwargs)
        job_id = response["jobId"]
        print(
            f"[hmr_world] batch runner: submitted array job {job_id} "
            f"({n_jobs} children) → queue={self._job_queue}"
        )
        return job_id

    def _poll_until_done(
        self,
        *,
        batch: BatchClient,
        job_id: str,
        n_jobs: int,
    ) -> None:
        """Block until every child has terminated.

        Per-index ok/error is recovered later by reading each spec's
        ``status.json`` from S3 — this loop only needs to detect "all
        done" so we don't have to fan out per-child DescribeJobs.
        """
        while True:
            resp = batch.describe_jobs(jobs=[job_id])
            jobs = resp.get("jobs", [])
            if not jobs:
                raise RuntimeError(f"describe_jobs returned empty for {job_id}")
            parent = jobs[0]
            summary = parent.get("arrayProperties", {}).get("statusSummary", {})
            if summary:
                terminal = int(summary.get("SUCCEEDED", 0)) + int(summary.get("FAILED", 0))
                pending = n_jobs - terminal
                parts = ", ".join(f"{k}={v}" for k, v in sorted(summary.items()))
                print(f"[hmr_world] batch: {pending} pending — {parts}")
                if pending <= 0:
                    return
            else:
                # Non-array (single-job) submission.
                status = parent.get("status")
                print(f"[hmr_world] batch: status = {status}")
                if status in _TERMINAL:
                    return
            time.sleep(self._poll_seconds)

    def _download_outputs(
        self,
        *,
        job_specs: list[_JobSpec],
        run_root: str,
        out_dir: Path,
        s3: S3Client,
    ) -> tuple[list[_JobSpec], list[tuple[_JobSpec, str | None]]]:
        succeeded: list[_JobSpec] = []
        failed: list[tuple[_JobSpec, str | None]] = []
        for idx, spec in enumerate(job_specs):
            output_prefix = f"{run_root}/jobs/{spec.shot_id}__{spec.player_id}"
            status_uri = f"{output_prefix}/status.json"
            # Try to read status.json regardless — even on Batch FAILED,
            # the handler may have written one.
            status_json: dict[str, Any] | None = None
            try:
                status_json = json.loads(s3.read_text(status_uri))
            except Exception:
                status_json = None
            ok = bool(status_json and status_json.get("status") in ("ok", "too_short"))
            if ok:
                # Download .npz + .json side-output.
                npz_local = out_dir / f"{spec.shot_id}__{spec.player_id}_smpl_world.npz"
                kp2d_local = out_dir / f"{spec.shot_id}__{spec.player_id}_kp2d.json"
                npz_uri = f"{output_prefix}/output/{spec.shot_id}__{spec.player_id}_smpl_world.npz"
                kp2d_uri = f"{output_prefix}/output/{spec.shot_id}__{spec.player_id}_kp2d.json"
                if status_json and status_json.get("status") == "ok":
                    try:
                        s3.download(npz_uri, npz_local)
                        s3.download(kp2d_uri, kp2d_local)
                        succeeded.append(spec)
                    except Exception as exc:  # noqa: BLE001
                        failed.append((spec, f"download failed: {exc}"))
                else:
                    # too_short — no output to download, but it's still success.
                    succeeded.append(spec)
            else:
                msg = (
                    status_json.get("error_message")
                    if status_json else "no status.json"
                )
                failed.append((spec, msg))
        return succeeded, failed

    def _print_summary(
        self,
        succeeded: list[_JobSpec],
        failed: list[tuple[_JobSpec, str | None]],
        run_root: str,
    ) -> None:
        total = len(succeeded) + len(failed)
        print(
            f"[hmr_world] {total} jobs submitted: "
            f"{len(succeeded)} succeeded, {len(failed)} failed"
        )
        for spec, message in failed:
            label = f"{spec.shot_id}__{spec.player_id}"
            print(
                f"  FAILED: {label} — {message or 'unknown'}\n"
                f"    log: {run_root}/jobs/{label}/stderr.log"
            )

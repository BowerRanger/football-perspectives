"""Pipeline wrapper around the released PhysPT model (CVPR 2024).

PhysPT refines an existing SMPL motion sequence with a learned physics
prior — it never sees images. The vendored author checkout lives at
``third_party/PhysPT`` (gitignored: re-clone from
https://github.com/zhangy76/PhysPT at 40d8699 and download its released
``assets/`` bundle; see docs/physpt-experiment-results.md). Following
the repo's research-code convention, author imports run under a
context shim (sys.path + cwd redirect) rather than editing the
checkout.

The model is a 20 fps transformer over 16-sample windows; motion is
resampled 30->20 fps and back, and each window's central XY increments
are integrated from one anchor per contiguous run (the author demo's
scheme). Torch is imported lazily so light environments that never
enable the takeover pass keep working without it.
"""
from __future__ import annotations

import contextlib
import inspect
import logging
import os
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from src.utils.physpt_adapter import resample_motion, stitch_windows
from src.utils.pose_temporal import frame_runs

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_AUTHOR_DIR = _REPO / "third_party" / "PhysPT"
_SMPL_NEUTRAL = _REPO / "third_party/gvhmr/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl"


def physpt_available(author_dir: Path = DEFAULT_AUTHOR_DIR) -> bool:
    """True when the author checkout, its released assets, and torch are
    all present — the takeover pass skips cleanly otherwise."""
    needed = [author_dir / "models", author_dir / "assets/checkpoint/PhysPT.pt",
              author_dir / "assets/data/regressor_physpt.npz", _SMPL_NEUTRAL]
    if not all(p.exists() for p in needed):
        return False
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@contextlib.contextmanager
def _author_context(author_dir: Path):
    """sys.path + cwd shim for the author code's relative asset loads."""
    prev_cwd = os.getcwd()
    sys.path.insert(0, str(author_dir))
    os.chdir(author_dir)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
        with contextlib.suppress(ValueError):
            sys.path.remove(str(author_dir))


def prepare_body(author_dir: Path) -> Path:
    """Adapt the existing licensed SMPL asset with the authors' regressors."""
    dest = author_dir / "assets/data/smpl/neutral.pkl"
    if dest.exists():
        return dest
    # Compatibility for this locally installed legacy licensed SMPL pickle.
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    for name, typ in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
                      ("object", object), ("str", str), ("unicode", str)]:
        if name not in np.__dict__:
            setattr(np, name, typ)
    with _SMPL_NEUTRAL.open("rb") as f:
        raw = pickle.load(f, encoding="latin1")
    dense = lambda x: np.asarray(x.toarray() if hasattr(x, "toarray") else x)  # noqa: E731
    params = {k: dense(raw[k]) for k in ("v_template", "f", "weights", "posedirs")}
    params["shapedirs"] = dense(raw["shapedirs"])[..., :10]
    params["V_regressor"] = dense(raw["J_regressor"])
    params["kintree_table"] = dense(raw["kintree_table"])[0].astype(np.int64)
    params["kintree_table"][0] = -1
    with np.load(author_dir / "assets/data/regressor_physpt.npz") as z:
        params.update({k: z[k] for k in z.files})
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        pickle.dump(params, f, protocol=4)
    return dest


class PhysPTRefiner:
    """Loads the released PhysPT network once and refines RefinedPose
    tracks in place of the experiment script's ad-hoc setup."""

    def __init__(self, author_dir: Path = DEFAULT_AUTHOR_DIR, device: str = "auto",
                 batch_size: int = 8):
        import torch
        self.author_dir = Path(author_dir)
        prepare_body(self.author_dir)
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        with _author_context(self.author_dir):
            from models import PhysPT  # type: ignore
            from models.smpl_phys import SMPL  # type: ignore
            torch.set_num_threads(4)
            self.model = PhysPT(device=self.device, seqlen=16, mode="test", f_dim=279,
                                d_model=1024, nhead=8, d_hid=1024, nlayers=6,
                                dropout=.1).to(self.device)
            checkpoint = self.author_dir / "assets/checkpoint/PhysPT.pt"
            self.model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
            self.model.eval()
            self.smpl = SMPL(device=self.device)

    def refine(self, track, model_fps: float = 20.0, input_fps: float = 30.0):
        """PhysPT-refine a RefinedPose track; returns ``(track, runs)``.

        Runs shorter than the 16-sample model window are left unchanged
        (recorded with status ``too_short``). Mirrors the validated
        experiment implementation (scripts/experiment_physpt.py).
        """
        import torch
        with _author_context(self.author_dir), torch.no_grad():
            import constants  # type: ignore
            from assets.utils import rot6d_to_rotmat  # type: ignore
            model, smpl, device, batch_size = self.model, self.smpl, self.device, self.batch_size
            theta = track.thetas.astype(float).copy()
            rr = track.root_R.astype(float).copy()
            rt = track.root_t.astype(float).copy()
            runs = []
            beta = torch.tensor(track.betas, dtype=torch.float32, device=device)
            rest_vertices = torch.tensordot(beta, smpl.shapedirs, dims=([0], [0])) + smpl.v_template
            rest_root = (rest_vertices.T @ smpl.regressor).T[0].cpu().numpy()
            for a, b in frame_runs(track.frames):
                times = (track.frames[a:b] - track.frames[a]) / input_fps
                if len(times) < 2 or times[-1] * model_fps < 15:
                    runs.append({"start": int(track.frames[a]), "end": int(track.frames[b - 1]),
                                 "status": "too_short"})
                    continue
                mt = np.arange(int(np.ceil(times[-1] * model_fps)) + 1) / model_fps
                input_R, input_root = resample_motion(times, theta[a:b], rr[a:b], rt[a:b], mt)
                # SMPL transl is an offset from the shaped rest pelvis,
                # whereas root_t is the world pelvis position.
                input_trans = input_root - rest_root
                starts = np.arange(len(mt) - 16 + 1)
                out_R = []
                out_T = []
                start_time = time.monotonic()
                for pos in range(0, len(starts), batch_size):
                    chunk = starts[pos:pos + batch_size]
                    idx = chunk[:, None] + np.arange(16)
                    betas = beta[None].expand(len(chunk) * 16, -1)
                    matrices = torch.tensor(input_R[idx].reshape(-1, 24, 3, 3),
                                            dtype=torch.float32, device=device)
                    trans = input_trans[idx].copy()
                    trans[:, :, :2] -= trans[:, :1, :2]
                    body = smpl.forward(betas=betas, rotmat=matrices)
                    dynamic = torch.cat((torch.tensor(trans, dtype=torch.float32, device=device),
                        matrices[:, :, :, :2].reshape(len(chunk), 16, 144),
                        body.joints_smpl.reshape(len(chunk), 16, 72),
                        body.joints[:, constants.target].reshape(len(chunk), 16, 60)), dim=-1)
                    q, _ = model(dynamic.transpose(0, 1), dynamic.transpose(0, 1),
                                 beta[None].expand(len(chunk), -1), None, None, None, smpl)
                    rotations = rot6d_to_rotmat(q[:, :, 3:].contiguous()).reshape(len(chunk), 16, 24, 3, 3)
                    out_R.extend(rotations.cpu().numpy())
                    out_T.extend(q[:, :, :3].cpu().numpy())
                r, t, selection = stitch_windows(out_R, out_T, starts, len(mt), input_trans[0, :2])
                t += rest_root
                for j in range(24):
                    interp = Slerp(mt, Rotation.from_matrix(r[:, j]))(times)
                    if j == 0:
                        rr[a:b] = interp.as_matrix()
                    else:
                        theta[a:b, j] = interp.as_rotvec()
                theta[a:b, 0] = 0
                for j in range(3):
                    rt[a:b, j] = np.interp(times, mt, t[:, j])
                runs.append({"start": int(track.frames[a]), "end": int(track.frames[b - 1]),
                             "status": "processed", "model_samples": len(mt),
                             "windows": len(starts),
                             "seconds": time.monotonic() - start_time,
                             "endpoint_xy_shift_m": float(np.linalg.norm(
                                 rt[b - 1, :2] - track.root_t[b - 1, :2])),
                             "window_selection": selection.tolist()})
        return replace(track, thetas=theta.astype(np.float32), root_R=rr.astype(np.float32),
                       root_t=rt.astype(np.float32)), runs

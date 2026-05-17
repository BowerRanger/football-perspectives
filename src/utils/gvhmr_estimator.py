"""GVHMR monocular human mesh recovery wrapper.

GVHMR (https://github.com/zju3dv/GVHMR, SIGGRAPH Asia 2024) is included as
a git submodule under ``third_party/gvhmr``.  This module wraps inference so
the pipeline can call :meth:`GVHMREstimator.estimate_sequence` per tracked
player.

GVHMR outputs SMPL parameters in the 'ay' coordinate frame:
  - Y-axis = gravity (pointing down)
  - X/Z   = horizontal ground plane
  - Ground is snapped to Y = 0

The estimator also returns 3D joint positions (22-joint SMPL topology) and
weak-perspective camera parameters for 2D reprojection.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GVHMR_ROOT = _REPO_ROOT / "third_party" / "gvhmr"
_GVHMR_SHIMS = _REPO_ROOT / "third_party" / "gvhmr_shims"

# Module-level lock guarding every GVHMR inference call. The shim that
# redirects ``.cuda()`` to ``.to(device)`` mutates process-global state
# (``torch.Tensor.cuda``) and is *not* thread-safe — two concurrent
# ``estimate_sequence`` calls race each other's monkey-patch save/
# restore and one of them ends up calling the original C-level
# ``.cuda()``, which on a non-CUDA build raises
# ``AssertionError: Torch not compiled with CUDA enabled``.
#
# Serialising at the call site is the cheapest robust fix: GVHMR is
# CPU/GPU bound and saturates one device anyway, so concurrent calls
# weren't speeding anything up — they were just corrupting state.
_GVHMR_INFERENCE_LOCK = threading.Lock()


@contextlib.contextmanager
def _cwd(target: Path):
    """Temporarily switch working directory."""
    prev = Path.cwd()
    try:
        os.chdir(target)
        yield
    finally:
        os.chdir(prev)


@contextlib.contextmanager
def _redirect_cuda(device: str):
    """Redirect ``.cuda()`` calls on modules/tensors to ``device``.

    GVHMR's preprocessors and several internal modules hardcode ``.cuda()``
    which fails on macOS (no CUDA) even when the user requested ``cpu`` or
    ``mps``.  This context manager patches the relevant methods on
    ``torch.nn.Module`` and ``torch.Tensor`` to call ``.to(device)`` instead,
    then restores the originals on exit.
    """
    import torch

    if device == "cuda" or device.startswith("cuda:"):
        # CUDA is actually available; don't patch.
        yield
        return

    orig_module_cuda = torch.nn.Module.cuda
    orig_tensor_cuda = torch.Tensor.cuda

    def _module_cuda(self, device_arg=None, *args, **kwargs):  # noqa: ARG001
        return self.to(device)

    def _tensor_cuda(self, device_arg=None, *args, **kwargs):  # noqa: ARG001
        return self.to(device)

    torch.nn.Module.cuda = _module_cuda  # type: ignore[method-assign]
    torch.Tensor.cuda = _tensor_cuda  # type: ignore[method-assign]

    # PyTorch Lightning's _DeviceDtypeModuleMixin overrides .cuda() with its
    # own version that bypasses nn.Module's (HMR2 inherits from it).
    patched_mixins = []
    for mod_path, cls_name in [
        ("lightning_fabric.utilities.device_dtype_mixin", "_DeviceDtypeModuleMixin"),
        ("lightning_fabric.utilities.device_dtype_mixin", "DeviceDtypeModuleMixin"),
        ("pytorch_lightning.core.mixins.device_dtype_mixin", "_DeviceDtypeModuleMixin"),
        ("pytorch_lightning.core.mixins.device_dtype_mixin", "DeviceDtypeModuleMixin"),
    ]:
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name, None)
            if cls is None or not hasattr(cls, "cuda"):
                continue
            patched_mixins.append((cls, cls.cuda))
            cls.cuda = _module_cuda  # type: ignore[method-assign]
        except ImportError:
            continue

    try:
        yield
    finally:
        torch.nn.Module.cuda = orig_module_cuda  # type: ignore[method-assign]
        torch.Tensor.cuda = orig_tensor_cuda  # type: ignore[method-assign]
        for cls, orig in patched_mixins:
            cls.cuda = orig  # type: ignore[method-assign]


def _normalize_device(device: str) -> str:
    requested = (device or "auto").strip().lower()
    if requested == "auto":
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda:0"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda":
        return "cuda:0"
    return requested


class GVHMREstimator:
    """Wraps GVHMR inference for per-track video sequences.

    The model is lazy-loaded on the first call to :meth:`estimate_sequence`.
    If GVHMR dependencies are not installed the constructor succeeds but
    :meth:`estimate_sequence` raises :class:`RuntimeError`.

    Parameters
    ----------
    checkpoint:
        Path to the GVHMR release checkpoint (relative to repo root or
        absolute).  Defaults to the standard location inside the GVHMR
        submodule.
    device:
        PyTorch device string (``"auto"`` selects CUDA > MPS > CPU).
    """

    def __init__(
        self,
        checkpoint: str = "third_party/gvhmr/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt",
        device: str = "auto",
        static_cam: bool = False,
    ) -> None:
        self._checkpoint = Path(checkpoint)
        if not self._checkpoint.is_absolute():
            self._checkpoint = _REPO_ROOT / self._checkpoint
        self._device = _normalize_device(device)
        # Broadcast cameras pan, so default to running SimpleVO to estimate
        # camera rotation per frame.  Set static_cam=True only when the
        # camera really doesn't move (e.g. a fixed tripod).
        self._static_cam = bool(static_cam)
        self._model: Any | None = None
        self._body_model: Any | None = None
        self._vitpose: Any | None = None
        self._extractor: Any | None = None
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        """Return True if GVHMR and its dependencies can be imported."""
        if self._available is None:
            try:
                self._ensure_imports()
                self._available = True
            except (ImportError, RuntimeError):
                self._available = False
        return self._available

    def _patch_gvhmr_preprocessors(self) -> None:
        """Replace ``.cuda()`` with ``.to(device)`` in GVHMR's preprocessors.

        The earlier ``_redirect_cuda`` context-manager approach (patching
        ``Module.cuda`` / ``Tensor.cuda`` for the duration of a load)
        ought to work — and does in unit tests — but in the actual web
        worker run the patch is somehow bypassed and the original
        ``Tensor.cuda`` is reached, which on a non-CUDA build raises
        ``AssertionError: Torch not compiled with CUDA enabled``.

        The simpler, more direct fix is to replace the ``__init__`` of
        ``VitPoseExtractor`` and ``Extractor`` so they don't call
        ``.cuda()`` at all, and to wrap their ``extract`` methods so
        inline ``imgs.cuda()`` becomes ``imgs.to(device)``.
        """
        import types
        import torch

        from hmr4d.utils.preproc import vitpose as _vitpose_mod
        from hmr4d.utils.preproc import vitfeat_extractor as _ext_mod

        device = self._device

        # ── VitPoseExtractor ───────────────────────────────────────────
        if not getattr(_vitpose_mod.VitPoseExtractor, "_fp_patched", False):
            from hmr4d.utils.preproc.vitpose_pytorch import build_model

            orig_extract = _vitpose_mod.VitPoseExtractor.extract

            def _patched_init(self, tqdm_leave=True):
                ckpt_path = "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
                self.pose = build_model("ViTPose_huge_coco_256x192", ckpt_path)
                self.pose.to(device).eval()
                self.flip_test = True
                self.tqdm_leave = tqdm_leave
                self._fp_device = device

            def _patched_extract(self, video_path, bbx_xys, img_ds=0.5):
                # Force ``.cuda()`` on intermediate tensors to instead use
                # the configured device.  Easiest: temporarily patch
                # ``Tensor.cuda`` for the duration of the call.
                orig = torch.Tensor.cuda

                def _to_device(t, *_a, **_kw):
                    return t.to(device)

                torch.Tensor.cuda = _to_device  # type: ignore[method-assign]
                try:
                    return orig_extract(self, video_path, bbx_xys, img_ds)
                finally:
                    torch.Tensor.cuda = orig  # type: ignore[method-assign]

            _vitpose_mod.VitPoseExtractor.__init__ = _patched_init
            _vitpose_mod.VitPoseExtractor.extract = _patched_extract
            _vitpose_mod.VitPoseExtractor._fp_patched = True

        # ── Extractor (HMR2 ViT features) ──────────────────────────────
        if not getattr(_ext_mod.Extractor, "_fp_patched", False):
            from hmr4d.network.hmr2 import load_hmr2

            orig_extract_features = _ext_mod.Extractor.extract_video_features

            def _patched_ext_init(self, tqdm_leave=True):
                self.extractor = load_hmr2().to(device).eval()
                self.tqdm_leave = tqdm_leave

            def _patched_ext_extract(self, video_path, bbx_xys, img_ds=0.5):
                orig = torch.Tensor.cuda

                def _to_device(t, *_a, **_kw):
                    return t.to(device)

                torch.Tensor.cuda = _to_device  # type: ignore[method-assign]
                try:
                    return orig_extract_features(self, video_path, bbx_xys, img_ds)
                finally:
                    torch.Tensor.cuda = orig  # type: ignore[method-assign]

            _ext_mod.Extractor.__init__ = _patched_ext_init
            _ext_mod.Extractor.extract_video_features = _patched_ext_extract
            _ext_mod.Extractor._fp_patched = True

    def _ensure_imports(self) -> None:
        """Add GVHMR and pytorch3d shims to sys.path and apply compat shims."""
        if not _GVHMR_ROOT.exists():
            raise RuntimeError(
                f"GVHMR submodule missing at {_GVHMR_ROOT}. "
                "Run `git submodule update --init --recursive`."
            )
        # Drop GVHMR's own colorlog StreamHandler from the root logger so
        # we don't get every line twice (once from our _LogQueueHandler,
        # once from their handler writing ANSI-coloured copy to stderr,
        # which our _QueueWriter then captures again).  Their handler is
        # added at import time of ``hmr4d.utils.pylogger``.
        import logging as _logging
        root = _logging.getLogger()
        for h in list(root.handlers):
            fmt = getattr(h, "formatter", None)
            if fmt is not None and "log_color" in (getattr(fmt, "_fmt", "") or ""):
                root.removeHandler(h)
        # chumpy (SMPL's legacy array library) uses ``inspect.getargspec``,
        # removed in Python 3.11.  Install a shim before any chumpy import.
        import inspect as _inspect
        if not hasattr(_inspect, "getargspec"):
            _inspect.getargspec = _inspect.getfullargspec  # type: ignore[attr-defined]
        # chumpy also expects numpy aliases (np.bool, np.int, np.float, etc.)
        # that were removed in numpy >= 1.20.  Re-add them to Python builtins'
        # equivalents.
        import numpy as _np
        for _alias, _target in (
            ("bool", bool), ("int", int), ("float", float),
            ("complex", complex), ("object", object),
            ("str", str), ("unicode", str),
        ):
            if not hasattr(_np, _alias):
                setattr(_np, _alias, _target)
        # Insert pytorch3d shims BEFORE gvhmr so our pure-PyTorch
        # reimplementation of pytorch3d.transforms is used instead of
        # requiring the full pytorch3d build (which needs CUDA C++ compilation).
        shims_str = str(_GVHMR_SHIMS)
        if shims_str not in sys.path:
            sys.path.insert(0, shims_str)
        gvhmr_str = str(_GVHMR_ROOT)
        if gvhmr_str not in sys.path:
            sys.path.insert(1, gvhmr_str)

    def _load_model(self) -> None:
        """Lazy-load the GVHMR network, preprocessors, and body model.

        Uses GVHMR's own hydra compose path (matching ``tools/demo/demo.py``)
        but via our minimal config registration so the heavy dataset-module
        imports are skipped.  The ``siga24_release.yaml`` file shipped in
        GVHMR is dead — it references a non-existent ``NetworkEncoderRoPEV2``
        class.  The live configs are registered from Python with
        ``populate_full_signature=True``, which keeps them in sync with code.
        """
        if self._model is not None:
            return

        self._ensure_imports()

        import hydra  # type: ignore[import-untyped]
        from hydra import compose, initialize_config_module  # type: ignore[import-untyped]
        from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL  # type: ignore[import-untyped]
        from hmr4d.utils.preproc import Extractor, VitPoseExtractor  # type: ignore[import-untyped]
        from hmr4d.utils.smplx_utils import make_smplx  # type: ignore[import-untyped]

        from src.utils.gvhmr_register import register_minimal_gvhmr

        if not self._checkpoint.exists():
            raise RuntimeError(
                f"GVHMR checkpoint not found at {self._checkpoint}. "
                "Download from https://github.com/zju3dv/GVHMR#getting-started "
                "or run scripts/setup_gvhmr.sh"
            )

        logger.info("Loading GVHMR model from %s on %s", self._checkpoint, self._device)

        # GVHMR's preprocessors and body models use paths relative to the
        # GVHMR repo root (e.g. "inputs/checkpoints/vitpose/..." and
        # "hmr4d/utils/body_model/smplx2smpl_sparse.pt").  Switch cwd for
        # the duration of model/preprocessor loading so those resolve.
        # Also redirect hardcoded ``.cuda()`` calls when CUDA isn't available.
        with _cwd(_GVHMR_ROOT), _redirect_cuda(self._device):
            # Hydra's GlobalHydra is process-wide singleton state; if a
            # previous load failed mid-init, the global is left set and a
            # subsequent ``initialize_config_module`` raises.  Clear it
            # defensively each time.
            from hydra.core.global_hydra import GlobalHydra
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()

            with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
                register_minimal_gvhmr()
                cfg = compose(
                    config_name="demo",
                    overrides=[
                        "video_name=_inference_",
                        "static_cam=True",
                    ],
                )

            model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
            model.load_pretrained_model(str(self._checkpoint))
            model = model.eval().to(self._device)
            self._model = model

            # Patch GVHMR's preprocessors to use .to(device) instead of
            # the hardcoded .cuda() before instantiating them.
            self._patch_gvhmr_preprocessors()

            # Load preprocessors (weights downloaded on first use into their own dirs)
            self._vitpose = VitPoseExtractor()
            self._extractor = Extractor()

            # Load SMPLX body model for FK.  GVHMR outputs 21-body-joint SMPLX
            # params (body_pose is 63D).  The "supermotion" variant is
            # SmplxLite, which accepts exactly those params.
            self._body_model = make_smplx("supermotion").to(self._device)

        logger.info("GVHMR loaded successfully on %s", self._device)

    def estimate_sequence(
        self,
        frames_bgr: list[np.ndarray],
        bboxes: list[list[float]],
        fps: float = 30.0,
        K_per_frame: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        # Serialise across the whole process — see the lock's docstring
        # for the rationale. This blocks rather than failing because the
        # caller is a stage worker that has nothing useful to do until
        # the prior estimate finishes.
        with _GVHMR_INFERENCE_LOCK:
            return self._estimate_sequence_locked(
                frames_bgr, bboxes, fps, K_per_frame
            )

    def _estimate_sequence_locked(
        self,
        frames_bgr: list[np.ndarray],
        bboxes: list[list[float]],
        fps: float = 30.0,
        K_per_frame: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Run GVHMR on a tracked player's video sequence.

        Parameters
        ----------
        frames_bgr:
            List of BGR images (full frames, not crops).
        bboxes:
            Per-frame bounding boxes ``[x1, y1, x2, y2]``.
        fps:
            Video frame rate.

        Returns
        -------
        dict with keys:
            global_orient : (N, 3) axis-angle root in 'ay' coords
            body_pose     : (N, 63) axis-angle for 21 body joints
            betas         : (10,) shape params (averaged across frames)
            transl        : (N, 3) root translation in 'ay' coords
            joints_3d     : (N, 22, 3) joint positions in 'ay' coords
            pred_cam      : (N, 3) weak-perspective [s, tx, ty]
            bbx_xys       : (N, 3) bbox [center_x, center_y, size]
        """
        self._load_model()

        import torch
        from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K  # type: ignore[import-untyped]
        from hmr4d.utils.geo_transform import compute_cam_angvel  # type: ignore[import-untyped]

        n_frames = len(frames_bgr)
        if n_frames == 0:
            return self._empty_result()

        # Convert bboxes [x1,y1,x2,y2] → tensors
        bbx_xyxy = torch.tensor(bboxes, dtype=torch.float32)  # (N, 4)
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (N, 3)

        img_h, img_w = frames_bgr[0].shape[:2]

        # Write frames to a temp video for GVHMR's preprocessors
        # (they expect a video path, not raw frames)
        tmp_video = self._write_temp_video(frames_bgr, fps)

        # GVHMR's preprocessors also call ``.cuda()`` on per-batch tensors
        # during inference, so keep the redirect active here.
        R_w2c: torch.Tensor | None = None
        try:
            with _redirect_cuda(self._device):
                # ViTPose: 2D keypoints
                kp2d = self._vitpose.extract(str(tmp_video), bbx_xys)  # (N, 17, 3)

                # ViT features: HMR2.0 backbone
                f_imgseq = self._extractor.extract_video_features(str(tmp_video), bbx_xys)  # (N, 1024)

                # Camera trajectory via SimpleVO (broadcast cameras pan).
                # Skipped if static_cam was forced on by the caller.
                if not self._static_cam:
                    R_w2c = self._estimate_camera_rotations(tmp_video, n_frames)
        finally:
            tmp_video.unlink(missing_ok=True)

        # Camera intrinsics: prefer the calibrated per-frame K from the
        # pipeline's camera_track when supplied. GVHMR's default
        # ``estimate_K(w, h)`` assumes ~60° FOV (fy ≈ 0.866·max(w, h)),
        # which under-estimates focal length for broadcast telephoto
        # shots and biases the predicted body to lean away from camera.
        if K_per_frame is not None and len(K_per_frame) == n_frames:
            K_fullimg = torch.tensor(
                np.asarray(K_per_frame, dtype=np.float32),
                dtype=torch.float32,
            )  # (N, 3, 3)
        else:
            K_fullimg = estimate_K(img_w, img_h).repeat(n_frames, 1, 1)

        if R_w2c is None:
            # Static-camera assumption: identity rotation, zero ang velocity.
            R_w2c = torch.eye(3).repeat(n_frames, 1, 1)
        cam_angvel = compute_cam_angvel(R_w2c)  # (N, 6)

        # Build data dict
        data = {
            "length": torch.tensor(n_frames),
            "bbx_xys": bbx_xys,
            "kp2d": kp2d,
            "K_fullimg": K_fullimg,
            "cam_angvel": cam_angvel,
            "f_imgseq": f_imgseq,
        }

        # Move all tensors to the configured device
        data = {
            k: v.to(self._device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }

        # Run GVHMR — keep the cuda redirect active because postprocess hops
        # (IK, static-joint refinement) also hardcode .cuda() internally.
        with torch.no_grad(), _redirect_cuda(self._device):
            pred = self._model.predict(data, static_cam=self._static_cam)

        # Extract incam params: global_orient maps SMPL canonical (y-up)
        # straight to OpenCV camera frame (y-down, z-into-scene). The
        # GVHMR demo feeds these into SMPL and renders the output verts
        # directly in camera view (no intermediate 'ay' frame), so this
        # is the convention to bridge into pitch world via R_w2c.T.
        # body_pose is identical between incam and global (they share the
        # same per-joint canonical axis-angles); betas likewise. transl
        # we don't use — hmr_world computes its own foot-anchored root_t.
        incam_params = pred["smpl_params_incam"]
        global_orient = incam_params["global_orient"].cpu().numpy()   # (N, 3)
        body_pose = incam_params["body_pose"].cpu().numpy()           # (N, 63)
        betas = incam_params["betas"].cpu().numpy()                   # (N, 10)
        transl = incam_params["transl"].cpu().numpy()                 # (N, 3)

        # Average betas across frames
        betas_avg = betas.mean(axis=0)  # (10,)

        # 3D joint positions via SMPL FK
        joints_3d = self._forward_kinematics(
            global_orient, body_pose, betas_avg, transl
        )  # (N, 22, 3)

        # pred_cam from network
        net_outputs = pred.get("net_outputs", {})
        model_output = net_outputs.get("model_output", {})
        pred_cam_raw = model_output.get("pred_cam")
        if pred_cam_raw is not None:
            pred_cam = pred_cam_raw.squeeze(0).cpu().numpy()  # (N, 3)
        else:
            pred_cam = np.zeros((n_frames, 3), dtype=np.float32)

        # 2D keypoints from ViTPose-Huge (already in image-pixel coords).
        # Saving these lets the dashboard draw an overlay that's far more
        # reliable than re-projecting world-frame joints through the
        # weak-perspective pred_cam (which was calibrated for in-camera
        # joints, not gravity-view ones).
        kp2d_np = kp2d.detach().cpu().numpy().astype(np.float32)  # (N, 17, 3)

        return {
            "global_orient": global_orient.astype(np.float32),
            "body_pose": body_pose.astype(np.float32),
            "betas": betas_avg.astype(np.float32),
            "transl": transl.astype(np.float32),
            "joints_3d": joints_3d.astype(np.float32),
            "pred_cam": pred_cam.astype(np.float32),
            "bbx_xys": bbx_xys.cpu().numpy().astype(np.float32),
            "kp2d": kp2d_np,
        }

    def _forward_kinematics(
        self,
        global_orient: np.ndarray,
        body_pose: np.ndarray,
        betas: np.ndarray,
        transl: np.ndarray,
    ) -> np.ndarray:
        """Run SMPL forward kinematics to get 3D joint positions."""
        import torch

        n = global_orient.shape[0]
        betas_expanded = np.tile(betas, (n, 1))

        output = self._body_model(
            global_orient=torch.tensor(global_orient, dtype=torch.float32, device=self._device),
            body_pose=torch.tensor(body_pose, dtype=torch.float32, device=self._device),
            betas=torch.tensor(betas_expanded, dtype=torch.float32, device=self._device),
            transl=torch.tensor(transl, dtype=torch.float32, device=self._device),
        )
        joints = output.joints[:, :22, :].cpu().numpy()
        return joints

    @staticmethod
    def _write_temp_video(frames_bgr: list[np.ndarray], fps: float) -> Path:
        """Write frames to a temporary mp4 file for GVHMR's preprocessors."""
        h, w = frames_bgr[0].shape[:2]
        tmp = Path(tempfile.mktemp(suffix=".mp4"))
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame in frames_bgr:
            writer.write(frame)
        writer.release()
        return tmp

    def _estimate_camera_rotations(
        self, video_path: Path, n_frames: int
    ) -> "torch.Tensor | None":
        """Run GVHMR's SimpleVO on the video to estimate per-frame camera
        rotation.  Returns ``(N, 3, 3)`` ``R_w2c`` rotations as a torch
        tensor, or ``None`` if VO fails (caller falls back to identity).

        Cameras in broadcast football pan to follow play, so feeding GVHMR
        the wrong (static) camera assumption causes player-position drift
        in the opposite direction of the pan ("sliding").  SimpleVO uses
        SIFT feature matching between sampled frames + a two-view
        pycolmap solve to estimate per-frame rotations cheaply.
        """
        import torch

        try:
            from hmr4d.utils.preproc.relpose.simple_vo import SimpleVO  # type: ignore[import-untyped]
        except Exception as exc:
            logger.warning("SimpleVO import failed (%s) — falling back to static cam", exc)
            return None

        # Run inside the GVHMR repo root; SimpleVO uses relative paths
        # internally for some intermediate artifacts.
        try:
            with _cwd(_GVHMR_ROOT):
                vo = SimpleVO(str(video_path), scale=0.5, step=8, method="sift")
                traj = vo.compute()  # (N, 4, 4) numpy
        except Exception as exc:
            logger.warning(
                "SimpleVO failed on %s (%s) — falling back to static cam",
                video_path.name, exc,
            )
            return None

        if traj is None or len(traj) == 0:
            return None
        traj = np.asarray(traj)
        if traj.shape[0] != n_frames:
            logger.warning(
                "SimpleVO returned %d frames but expected %d — falling back to static cam",
                traj.shape[0], n_frames,
            )
            return None

        return torch.from_numpy(traj[:, :3, :3]).float()

    @staticmethod
    def _empty_result() -> dict[str, np.ndarray]:
        return {
            "global_orient": np.zeros((0, 3), dtype=np.float32),
            "body_pose": np.zeros((0, 63), dtype=np.float32),
            "betas": np.zeros(10, dtype=np.float32),
            "transl": np.zeros((0, 3), dtype=np.float32),
            "joints_3d": np.zeros((0, 22, 3), dtype=np.float32),
            "pred_cam": np.zeros((0, 3), dtype=np.float32),
            "bbx_xys": np.zeros((0, 3), dtype=np.float32),
            "kp2d": np.zeros((0, 17, 3), dtype=np.float32),
        }


def _axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Convert (N, 3) axis-angle vectors to (N, 3, 3) rotation matrices.

    Implemented in numpy so this module stays importable without torch.
    Uses Rodrigues' formula:  R = I + sin(theta) K + (1 - cos(theta)) K^2
    where K is the skew-symmetric matrix of the unit axis.
    """
    aa = np.asarray(aa, dtype=np.float64).reshape(-1, 3)
    n = aa.shape[0]
    if n == 0:
        return np.zeros((0, 3, 3), dtype=np.float64)

    theta = np.linalg.norm(aa, axis=1)  # (N,)
    out = np.tile(np.eye(3), (n, 1, 1))
    nonzero = theta > 1e-9
    if not np.any(nonzero):
        return out

    axis = aa[nonzero] / theta[nonzero, np.newaxis]  # (M, 3)
    th = theta[nonzero]
    sin_t = np.sin(th)[:, np.newaxis, np.newaxis]
    cos_t = np.cos(th)[:, np.newaxis, np.newaxis]

    M = axis.shape[0]
    K = np.zeros((M, 3, 3), dtype=np.float64)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    K2 = K @ K
    R = np.eye(3) + sin_t * K + (1.0 - cos_t) * K2
    out[nonzero] = R
    return out


def _read_video_frames(
    video_path: Path, frame_indices: list[int]
) -> list[np.ndarray]:
    """Read specific frames from a video file by index. Returns BGR frames.

    Frames missing from the video (eof reached) are returned as black images
    matching the first decoded frame's shape. Raises if the video cannot
    be opened at all.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    try:
        # Sort/dedupe so seeking is monotonic and we only decode each frame once.
        wanted = sorted(set(frame_indices))
        cache: dict[int, np.ndarray] = {}
        last_shape: tuple[int, int, int] | None = None
        for fi in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                if last_shape is None:
                    last_shape = (720, 1280, 3)
                cache[fi] = np.zeros(last_shape, dtype=np.uint8)
            else:
                last_shape = frame.shape
                cache[fi] = frame
        return [cache[fi] for fi in frame_indices]
    finally:
        cap.release()


def run_on_track(
    track_frames: list[tuple[int, tuple[int, int, int, int]]],
    *,
    video_path: Path,
    checkpoint: Path,
    device: str,
    batch_size: int,
    max_sequence_length: int,
    estimator: "GVHMREstimator | None" = None,
    per_frame_K: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run GVHMR over a single player's track.

    Parameters
    ----------
    track_frames:
        Ordered list of ``(frame_index, (x1, y1, x2, y2))``.
    video_path:
        Path to the source clip; frames are decoded by index.
    checkpoint:
        Path to the GVHMR release checkpoint. Must exist on disk.
    device:
        PyTorch device string (``"auto"``, ``"cpu"``, ``"mps"``, ``"cuda:0"``).
    batch_size:
        Reserved for future use; current GVHMR predict() is sequence-level.
    max_sequence_length:
        Maximum number of frames per inference call. Long tracks are
        chunked to avoid the MPS allocation issue noted in the spec.
    estimator:
        Optional pre-constructed estimator. When supplied, its already-loaded
        GVHMR + ViTPose-Huge + HMR2-ViT + SMPLX models are reused, saving
        the 30-60s per-call load. Pass ``None`` to construct a fresh
        estimator (legacy behaviour). The caller is responsible for
        ensuring the estimator's checkpoint/device match.
    per_frame_K:
        Optional ``(N, 3, 3)`` array of per-frame camera intrinsics
        aligned to ``track_frames`` order. When supplied, GVHMR uses
        these instead of its built-in ``estimate_K(w, h)`` default —
        fixing the lean-away bias caused by underestimated focal length
        on broadcast telephoto shots. Missing-camera frames should be
        backfilled (e.g. with the shot's median K) by the caller so the
        array is dense.

    Returns
    -------
    dict with arrays keyed by per-frame index in track order:
        thetas:           (N, 24, 3)  axis-angle pose, root included at idx 0
        betas:            (N, 10)     per-frame shape (caller medians)
        root_R_cam:       (N, 3, 3)   root rotation in camera frame
        root_t_cam:       (N, 3)      root translation in camera frame
        joint_confidence: (N, 24)     per-joint confidence in [0, 1]
        kp2d:             (N, 17, 3)  COCO-17 image-pixel keypoints + conf
                                     from GVHMR's internal ViTPose-Huge.
                                     Consumed by hmr_world for foot-anchoring
                                     and persisted as the dashboard overlay.

    Notes
    -----
    GVHMR returns SMPL pose in 21 body joints + global_orient. SMPL canonical
    is 24 joints (root + 23). The two extra joints (hands_left, hands_right
    under the SMPL-H/SMPL-X convention) are zero-padded.
    """
    n = len(track_frames)
    if n == 0:
        return {
            "thetas": np.zeros((0, 24, 3), dtype=np.float32),
            "betas": np.zeros((0, 10), dtype=np.float32),
            "root_R_cam": np.zeros((0, 3, 3), dtype=np.float32),
            "root_t_cam": np.zeros((0, 3), dtype=np.float32),
            "joint_confidence": np.zeros((0, 24), dtype=np.float32),
            "kp2d": np.zeros((0, 17, 3), dtype=np.float32),
        }

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise RuntimeError(
            f"GVHMR checkpoint not found at {checkpoint}. "
            "Download from https://github.com/zju3dv/GVHMR or run scripts/setup_gvhmr.sh"
        )

    if estimator is None:
        estimator = GVHMREstimator(checkpoint=str(checkpoint), device=device)

    frame_indices = [int(fi) for fi, _ in track_frames]
    bboxes = [list(map(float, bb)) for _, bb in track_frames]
    frames_bgr = _read_video_frames(video_path, frame_indices)

    chunk = max(1, int(max_sequence_length))
    all_thetas: list[np.ndarray] = []
    all_betas: list[np.ndarray] = []
    all_root_R_cam: list[np.ndarray] = []
    all_root_t_cam: list[np.ndarray] = []
    all_joint_conf: list[np.ndarray] = []
    all_kp2d: list[np.ndarray] = []

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sub_frames = frames_bgr[start:end]
        sub_bboxes = bboxes[start:end]
        sub_K = (
            per_frame_K[start:end] if per_frame_K is not None else None
        )
        out = estimator.estimate_sequence(
            sub_frames, sub_bboxes, K_per_frame=sub_K
        )

        global_orient = out["global_orient"]                # (M, 3)
        body_pose = out["body_pose"].reshape(-1, 21, 3)     # (M, 21, 3)
        betas_avg = out["betas"]                            # (10,)
        transl = out["transl"]                              # (M, 3)
        kp2d = out["kp2d"]                                  # (M, 17, 3)

        m = global_orient.shape[0]
        thetas = np.zeros((m, 24, 3), dtype=np.float32)
        thetas[:, 0, :] = global_orient
        thetas[:, 1:22, :] = body_pose
        # joints 22, 23 (hand placeholders) remain zero

        betas_per_frame = np.tile(betas_avg.astype(np.float32), (m, 1))
        root_R_cam = _axis_angle_to_matrix(global_orient).astype(np.float32)
        root_t_cam = transl.astype(np.float32)

        # Joint confidence: GVHMR's ViTPose 2D keypoint confidences cover 17
        # COCO joints. For the 24 SMPL joints, take the mean of the per-frame
        # 2D-keypoint confidence and broadcast — the foot-anchor stage only
        # uses min(joint_confidence) anyway. Joints with no 2D analogue
        # (root, spine) inherit the same scalar. This is intentionally
        # conservative; if needed, downstream can refine.
        if kp2d.size:
            scalar_conf = kp2d[:, :, 2].mean(axis=1, keepdims=True)  # (M, 1)
        else:
            scalar_conf = np.zeros((m, 1), dtype=np.float32)
        joint_conf = np.broadcast_to(scalar_conf, (m, 24)).astype(np.float32)

        all_thetas.append(thetas)
        all_betas.append(betas_per_frame)
        all_root_R_cam.append(root_R_cam)
        all_root_t_cam.append(root_t_cam)
        all_joint_conf.append(joint_conf)
        all_kp2d.append(kp2d.astype(np.float32))

    return {
        "thetas": np.concatenate(all_thetas, axis=0) if all_thetas else np.zeros((0, 24, 3), dtype=np.float32),
        "betas": np.concatenate(all_betas, axis=0) if all_betas else np.zeros((0, 10), dtype=np.float32),
        "root_R_cam": np.concatenate(all_root_R_cam, axis=0) if all_root_R_cam else np.zeros((0, 3, 3), dtype=np.float32),
        "root_t_cam": np.concatenate(all_root_t_cam, axis=0) if all_root_t_cam else np.zeros((0, 3), dtype=np.float32),
        "joint_confidence": np.concatenate(all_joint_conf, axis=0) if all_joint_conf else np.zeros((0, 24), dtype=np.float32),
        "kp2d": np.concatenate(all_kp2d, axis=0) if all_kp2d else np.zeros((0, 17, 3), dtype=np.float32),
    }


class FakeGVHMREstimator:
    """Deterministic test double that produces plausible SMPL parameters.

    Generates a standing pose with slight arm variation per frame,
    translation derived from bbox center, and identity shape.
    """

    available = True

    def estimate_sequence(
        self,
        frames_bgr: list[np.ndarray],
        bboxes: list[list[float]],
        fps: float = 30.0,
    ) -> dict[str, np.ndarray]:
        n = len(frames_bgr)
        if n == 0:
            return GVHMREstimator._empty_result()

        bbx_xys = _bboxes_to_xys(bboxes)

        global_orient = np.zeros((n, 3), dtype=np.float32)
        body_pose = np.zeros((n, 63), dtype=np.float32)
        betas = np.zeros(10, dtype=np.float32)

        transl = np.zeros((n, 3), dtype=np.float32)
        # ay frame is Y-up post-snap: feet at Y=0, pelvis built into offsets
        transl[:, 2] = 2.0   # 2m from camera

        joints_3d = _standing_skeleton(n, transl)

        pred_cam = np.zeros((n, 3), dtype=np.float32)
        pred_cam[:, 0] = 1.0

        # Synthetic 2D keypoints at bbox center, full confidence.
        kp2d = np.zeros((n, 17, 3), dtype=np.float32)
        kp2d[:, :, 0] = bbx_xys[:, 0:1]  # x = bbox center x
        kp2d[:, :, 1] = bbx_xys[:, 1:2]  # y = bbox center y
        kp2d[:, :, 2] = 1.0              # confidence

        return {
            "global_orient": global_orient,
            "body_pose": body_pose,
            "betas": betas,
            "transl": transl,
            "joints_3d": joints_3d,
            "pred_cam": pred_cam,
            "bbx_xys": bbx_xys,
            "kp2d": kp2d,
        }


def _bboxes_to_xys(bboxes: list[list[float]]) -> np.ndarray:
    """Convert [x1,y1,x2,y2] bounding boxes to [center_x, center_y, size]."""
    arr = np.array(bboxes, dtype=np.float32)
    cx = (arr[:, 0] + arr[:, 2]) / 2.0
    cy = (arr[:, 1] + arr[:, 3]) / 2.0
    size = np.maximum(arr[:, 2] - arr[:, 0], arr[:, 3] - arr[:, 1])
    return np.stack([cx, cy, size], axis=1)


def _standing_skeleton(n: int, transl: np.ndarray) -> np.ndarray:
    """Generate a synthetic standing SMPL 22-joint skeleton.

    GVHMR's post-processed 'ay' frame is empirically Y-up: the lowest
    body point sits at Y≈0 and Y increases upward.  Pelvis ≈ 0.9m above
    ground, head ≈ 1.7m.
    """
    offsets = np.array([
        [0.00,  0.90, 0.0],   # 0  pelvis
        [0.10,  0.90, 0.0],   # 1  left hip
        [-0.10, 0.90, 0.0],   # 2  right hip
        [0.00,  1.08, 0.0],   # 3  spine1
        [0.10,  0.45, 0.0],   # 4  left knee
        [-0.10, 0.45, 0.0],   # 5  right knee
        [0.00,  1.26, 0.0],   # 6  spine2
        [0.10,  0.05, 0.0],   # 7  left ankle (just above ground)
        [-0.10, 0.05, 0.0],   # 8  right ankle
        [0.00,  1.40, 0.0],   # 9  spine3
        [0.10,  0.00, 0.0],   # 10 left foot
        [-0.10, 0.00, 0.0],   # 11 right foot
        [0.00,  1.55, 0.0],   # 12 neck
        [0.10,  1.50, 0.0],   # 13 left collar
        [-0.10, 1.50, 0.0],   # 14 right collar
        [0.00,  1.70, 0.0],   # 15 head
        [0.20,  1.50, 0.0],   # 16 left shoulder
        [-0.20, 1.50, 0.0],   # 17 right shoulder
        [0.20,  1.25, 0.0],   # 18 left elbow
        [-0.20, 1.25, 0.0],   # 19 right elbow
        [0.20,  1.00, 0.0],   # 20 left wrist
        [-0.20, 1.00, 0.0],   # 21 right wrist
    ], dtype=np.float32)

    joints = np.tile(offsets, (n, 1, 1))  # (N, 22, 3)
    joints += transl[:, np.newaxis, :]
    return joints

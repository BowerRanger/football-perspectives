"""Stage 8: SMPL Body Model Fitting.

Fits SMPL pose, shape, and translation parameters to triangulated 3D joints.
"""

import logging
from pathlib import Path

import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.smpl_result import SmplResult
from src.schemas.triangulated import TriangulatedPlayer
from src.utils.smpl_fitting import fit_smpl_sequence


class SmplFittingStage(BaseStage):
    name = "smpl_fitting"

    def is_complete(self) -> bool:
        smpl_dir = self.output_dir / "smpl"
        return smpl_dir.exists() and any(smpl_dir.glob("*.npz"))

    def run(self) -> None:
        cfg = self.config.get("smpl_fitting", {})
        model_path = str(cfg.get("model_path", "data/smpl/SMPL_NEUTRAL.pkl"))
        device = str(cfg.get("device", "cpu"))
        lr = float(cfg.get("lr", 0.01))
        n_iterations = int(cfg.get("n_iterations", 100))
        lambda_joint = float(cfg.get("lambda_joint", 1.0))
        lambda_prior = float(cfg.get("lambda_prior", 0.01))
        lambda_shape = float(cfg.get("lambda_shape", 0.1))
        lambda_smooth = float(cfg.get("lambda_smooth", 0.5))
        lambda_ground = float(cfg.get("lambda_ground", 0.1))

        tri_dir = self.output_dir / "triangulated"
        if not tri_dir.exists():
            logging.warning("No triangulated/ directory — skipping SMPL fitting")
            return

        npz_files = sorted(tri_dir.glob("*.npz"))
        if not npz_files:
            logging.warning("No triangulated .npz files — skipping SMPL fitting")
            return

        smpl_dir = self.output_dir / "smpl"
        smpl_dir.mkdir(parents=True, exist_ok=True)

        print(f"  -> fitting SMPL for {len(npz_files)} players")

        for npz_path in npz_files:
            player = TriangulatedPlayer.load(npz_path)
            print(f"     {player.player_id}: {player.positions.shape[0]} frames")

            betas, poses, transl = fit_smpl_sequence(
                positions_3d=player.positions,
                confidences=player.confidences,
                model_path=model_path,
                device=device,
                lr=lr,
                n_iterations=n_iterations,
                lambda_joint=lambda_joint,
                lambda_prior=lambda_prior,
                lambda_shape=lambda_shape,
                lambda_smooth=lambda_smooth,
                lambda_ground=lambda_ground,
            )

            result = SmplResult(
                player_id=player.player_id,
                player_name=player.player_name,
                team=player.team,
                betas=betas,
                poses=poses,
                transl=transl,
                fps=player.fps,
            )
            result.save(smpl_dir / f"{player.player_id}_smpl.npz")

        n_saved = len(list(smpl_dir.glob("*.npz")))
        print(f"  -> saved {n_saved} SMPL fits to smpl/")

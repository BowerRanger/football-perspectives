"""Stage 4: 2D pose estimation (ViTPose via MMPose).

TODO(Phase 1c): Reimplement against the new pose_2d schema and the
``pose_2d`` config block.  See spec section 5.4.
"""

from src.pipeline.base import BaseStage


class Pose2DStage(BaseStage):
    name = "pose_2d"

    def is_complete(self) -> bool:
        return False

    def run(self) -> None:
        raise NotImplementedError(
            "Pose2DStage is not implemented yet — see Phase 1c of "
            "docs/superpowers/plans/2026-05-04-broadcast-mono-pipeline.md"
        )

"""Stage 1: Prepare shots — copy / re-encode source clips into the working dir.

TODO(Phase 1c / 4): Replace with the broadcast-mono prepare-shots stage that
copies the source mp4 into ``shots/`` and writes a flat manifest.  See spec
section 5.1.
"""

from src.pipeline.base import BaseStage


class PrepareShotsStage(BaseStage):
    name = "prepare_shots"

    def is_complete(self) -> bool:
        return (self.output_dir / "shots" / "shots_manifest.json").exists()

    def run(self) -> None:
        raise NotImplementedError(
            "PrepareShotsStage is not implemented yet — see Phase 1c/4 of "
            "docs/superpowers/plans/2026-05-04-broadcast-mono-pipeline.md"
        )

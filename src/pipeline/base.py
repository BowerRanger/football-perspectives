from abc import ABC, abstractmethod
from pathlib import Path


class BaseStage(ABC):
    name: str
    # Optional multi-shot filter set by ``run_pipeline`` when the caller
    # restricts a stage to a single shot. ``None`` means "process all
    # shots in the manifest". Stages that iterate ``manifest.shots``
    # check this and skip non-matching shots; stages that don't use the
    # manifest ignore it.
    shot_filter: str | None = None

    def __init__(self, config: dict, output_dir: Path, **kwargs) -> None:
        self.config = config
        self.output_dir = output_dir

    @abstractmethod
    def run(self) -> None:
        ...

    def is_complete(self) -> bool:
        """Return True if all expected outputs exist (enables cache skip)."""
        return False

from abc import ABC, abstractmethod
from pathlib import Path


class BaseStage(ABC):
    name: str

    def __init__(self, config: dict, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir

    @abstractmethod
    def run(self) -> None:
        ...

    def is_complete(self) -> bool:
        """Return True if all expected outputs exist (enables cache skip)."""
        return False

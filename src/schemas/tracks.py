from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import os
import tempfile


@dataclass
class TrackFrame:
    frame: int
    bbox: list[float]           # [x1, y1, x2, y2] in pixel space
    confidence: float
    pitch_position: list[float] | None  # [x, y] in pitch metres, or None
    interpolated: bool = False  # True when the bbox was linearly filled
                                # in by a post-pass gap interpolator
                                # rather than produced by the detector.


@dataclass
class Track:
    track_id: str
    class_name: str             # "player" | "goalkeeper" | "referee" | "ball"
    team: str                   # "A" | "B" | "referee" | "unknown"
    player_id: str = ""         # global ID across shots (e.g., "P001")
    player_name: str = ""       # user-friendly name (e.g., "Salah")
    frames: list[TrackFrame] = field(default_factory=list)


@dataclass
class TracksResult:
    shot_id: str
    tracks: list[Track] = field(default_factory=list)

    def save(self, path: Path) -> None:
        """Atomically replace ``path`` with the serialised result.

        Concurrent writers used to interleave their bytes when two
        ``write_text`` calls raced on the same file, producing a JSON
        document with extra trailing data the parser rejects. Writing
        to a temp file in the same directory and ``os.replace``-ing
        over the target is atomic on POSIX / macOS, so the worst a
        reader observes mid-flight is the previous valid version.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(self), indent=2)
        fd, tmp = tempfile.mkstemp(
            prefix=path.name + ".",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(payload)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(cls, path: Path) -> "TracksResult":
        data = json.loads(path.read_text())
        tracks = []
        for t in data.pop("tracks"):
            frames = [TrackFrame(**f) for f in t.pop("frames")]
            tracks.append(Track(frames=frames, **t))
        return cls(tracks=tracks, **data)

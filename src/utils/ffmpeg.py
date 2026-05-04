import subprocess
from pathlib import Path


def extract_clip(src: Path, out: Path, start_s: float, end_s: float) -> None:
    """Extract a clip from src between start_s and end_s (seconds)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start_s),
            "-to", str(end_s),
            "-i", str(src),
            "-c", "copy",
            str(out),
        ],
        check=True,
        capture_output=True,
    )


def extract_thumbnail(src: Path, out: Path, time_s: float) -> None:
    """Extract a single frame as JPEG at time_s (seconds)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(time_s),
            "-i", str(src),
            "-vframes", "1",
            str(out),
        ],
        check=True,
        capture_output=True,
    )

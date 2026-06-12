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


def extract_clip_reencode(
    src: Path,
    out: Path,
    start_s: float,
    end_s: float,
    fps: float,
    speed_factor: float = 1.0,
    crf: int = 18,
) -> None:
    """Frame-accurate clip extraction (re-encode, unlike ``extract_clip``
    whose stream-copy snaps to keyframes).

    ``speed_factor`` > 1 means the span is slow-motion; the output is
    retimed to real time (``setpts``) and resampled to ``fps``. Audio is
    tempo-matched so the dashboard's sync editor keeps usable sound.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    retimed = abs(speed_factor - 1.0) > 1e-6
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-to", f"{end_s:.3f}",
        "-i", str(src),
        "-vf", f"setpts=PTS/{speed_factor:.6f}" if retimed else "null",
        "-r", f"{fps:.6f}",
        "-c:v", "libx264", "-crf", str(crf), "-preset", "fast",
        "-pix_fmt", "yuv420p",
    ]
    if retimed:
        # atempo only supports 0.5–100.0 per instance; the broadcast
        # slow-mo range (≤4x) fits a single instance after clamping.
        tempo = min(100.0, max(0.5, speed_factor))
        cmd += ["-af", f"atempo={tempo:.6f}"]
    cmd += ["-c:a", "aac", "-b:a", "96k", str(out)]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        # Some sources trip on the audio graph (exotic codecs, broken
        # streams). The clip video is what the pipeline needs — retry
        # without audio rather than failing the extraction.
        cmd_noaudio = cmd[:-1]
        if "-af" in cmd_noaudio:
            i = cmd_noaudio.index("-af")
            del cmd_noaudio[i:i + 2]
        if "-c:a" in cmd_noaudio:
            i = cmd_noaudio.index("-c:a")
            del cmd_noaudio[i:i + 4]
        cmd_noaudio += ["-an", str(out)]
        subprocess.run(cmd_noaudio, check=True, capture_output=True)


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

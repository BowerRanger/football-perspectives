"""Compose a vertical YouTube Short from rendered 9:16 clips.

The "shorts" last-mile compositor (logic moved here from
``scripts/compose_short.py``, which is now a thin CLI): an
edit-decision list names rendered 9:16 clips by SCENE frame, and
this module trims, speed-ramps, freezes, cuts and captions them into one
1080x1920 H.264 mp4 with a silent AAC track (platform-friendly).

ffmpeg here has no drawtext (no libfreetype), so captions are rasterised
with Pillow to full-frame transparent PNGs and overlaid with ``enable=``.

EDL schema::

    size: [1080, 1920]
    fps: 30
    fonts: {title: <ttf path>, body: <ttf path>}
    segments:
      - src: <mp4>              # a render_experiments 9:16 clip
        first_frame: 300        # scene frame of the mp4's first frame
        stretch: 1              # time_stretch the clip was rendered with
        from: 330               # scene frames (inclusive start, exclusive end)
        to: 372
        speed: 1.0              # extra playback multiplier (<1 = slower)
        hold_s: 0.0             # freeze the last frame this long
        flash: false            # white flash-in at the cut
        label: "GK CAM"         # optional chip shown for the segment
    captions:
      - {text: "WHAT GOAL IS THIS?", style: title, start: 0, end: null}
      - {text: "...", style: sub, start: -3.0}    # negative = from the end

The audio track is silent AAC unless ``compose(..., audio=<wav/m4a>)`` is
given (see ``src.utils.shorts_audio``). ``edl_time_map`` returns the
output-timeline span of every segment so audio events can be placed.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONTS = {
    "title": "/System/Library/Fonts/Supplemental/Impact.ttf",
    "body": "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
}
# Shorts UI safe area (1080x1920): keep text out of the top ~220 px
# (status/search bar) and bottom ~420 px (title, channel, buttons).
SAFE_TOP_PX = 250
SAFE_BOTTOM_PX = 430


@dataclass(frozen=True)
class CaptionStyle:
    font: str
    size: int
    fill: str
    stroke: str
    stroke_px: int
    box: str | None
    y: int            # top of text block (px); negative = from bottom
    pad_px: int = 26


def _styles(fonts: dict, height: int) -> dict[str, CaptionStyle]:
    return {
        "title": CaptionStyle(fonts["title"], 98, "#ffffff", "#000000", 9,
                              None, SAFE_TOP_PX),
        "kicker": CaptionStyle(fonts["title"], 90, "#111111", "#111111", 0,
                               "#ffd400", SAFE_TOP_PX - 10),
        "sub": CaptionStyle(fonts["body"], 82, "#ffffff", "#000000", 6,
                            None, -(SAFE_BOTTOM_PX + 130)),
        "chip": CaptionStyle(fonts["body"], 54, "#ffffff", "#ffffff", 0,
                             "#e10600", SAFE_TOP_PX + 150),
        "chip_dark": CaptionStyle(fonts["body"], 54, "#ffffff", "#ffffff", 0,
                                  "#111111", SAFE_TOP_PX + 150),
    }


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        words, cur = para.split(), ""
        for w in words:
            trial = f"{cur} {w}".strip()
            if draw.textlength(trial, font=font) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def render_caption_png(text: str, style: CaptionStyle, size: tuple[int, int],
                       path: Path) -> None:
    w, h = size
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(style.font, style.size)
    lines = _wrap(draw, text, font, int(w * 0.84))
    line_h = int(style.size * 1.08)
    block_h = line_h * len(lines)
    y0 = style.y if style.y >= 0 else h + style.y - block_h
    for i, line in enumerate(lines):
        tw = draw.textlength(line, font=font)
        x = (w - tw) / 2
        y = y0 + i * line_h
        if style.box:
            bbox = draw.textbbox((x, y), line, font=font)
            draw.rounded_rectangle(
                (bbox[0] - style.pad_px, bbox[1] - style.pad_px * 0.6,
                 bbox[2] + style.pad_px, bbox[3] + style.pad_px * 0.6),
                radius=14, fill=style.box)
        draw.text((x, y), line, font=font, fill=style.fill,
                  stroke_width=style.stroke_px, stroke_fill=style.stroke)
    img.save(path)


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({' '.join(cmd[:6])} ...):\n{proc.stderr[-2000:]}")


def _probe_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    return int(json.loads(out)["streams"][0]["nb_read_frames"])


def build_segment(seg: dict, idx: int, size: tuple[int, int], fps: int,
                  work: Path) -> tuple[Path, float]:
    """Trim/scale/speed/freeze one segment; returns (path, duration_s)."""
    src = Path(seg["src"])
    stretch = int(seg.get("stretch", 1))
    first = int(seg["first_frame"])
    a = (int(seg["from"]) - first) * stretch
    b = (int(seg["to"]) - first) * stretch
    n_src = _probe_frames(src)
    if not 0 <= a < b <= n_src:
        raise ValueError(f"segment {idx}: frames [{a},{b}) outside {src} (0..{n_src})")
    speed = float(seg.get("speed", 1.0))
    hold = float(seg.get("hold_s", 0.0))
    w, h = size
    vf = [f"trim=start_frame={a}:end_frame={b}", "setpts=PTS-STARTPTS"]
    if speed != 1.0:
        vf.append(f"setpts=PTS/{speed}")
    vf += [f"fps={fps}", f"scale={w}:{h}:force_original_aspect_ratio=increase",
           f"crop={w}:{h}", "setsar=1"]
    if hold > 0:
        vf.append(f"tpad=stop_mode=clone:stop_duration={hold}")
    if seg.get("flash"):
        vf.append("fade=t=in:st=0:d=0.18:color=white")
    out = work / f"seg_{idx:02d}.mp4"
    _run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", ",".join(vf),
          "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
          "-pix_fmt", "yuv420p", "-r", str(fps), str(out)])
    duration = (b - a) / fps / speed + hold
    return out, duration


def segment_duration_s(seg: dict, fps: float) -> float:
    """Output-timeline duration of one EDL segment (matches build_segment)."""
    stretch = int(seg.get("stretch", 1))
    first = int(seg["first_frame"])
    a = (int(seg["from"]) - first) * stretch
    b = (int(seg["to"]) - first) * stretch
    return (b - a) / fps / float(seg.get("speed", 1.0)) + float(seg.get("hold_s", 0.0))


def edl_time_map(edl: dict, fps: float | None = None) -> list[dict]:
    """Output-timeline spans for every segment of ``edl``.

    Each entry: ``index, start_s, end_s, scene_from, scene_to, stretch,
    speed, hold_s, src``. ``scene_frame_to_out_s`` below inverts it.
    """
    fps = float(fps if fps is not None else edl.get("fps", 30))
    out, t = [], 0.0
    for i, seg in enumerate(edl["segments"]):
        dur = segment_duration_s(seg, fps)
        out.append({
            "index": i, "start_s": t, "end_s": t + dur,
            "scene_from": int(seg["from"]), "scene_to": int(seg["to"]),
            "stretch": int(seg.get("stretch", 1)),
            "speed": float(seg.get("speed", 1.0)),
            "hold_s": float(seg.get("hold_s", 0.0)),
            "src": str(seg["src"]),
        })
        t += dur
    return out


def scene_frame_to_out_s(edl: dict, scene_frame: float, fps: float | None = None) -> list[float]:
    """Every output time (s) at which ``scene_frame`` is on screen (a frame
    shown in several segments, e.g. a replay, appears once per segment)."""
    fps = float(fps if fps is not None else edl.get("fps", 30))
    hits = []
    for span in edl_time_map(edl, fps):
        if span["scene_from"] <= scene_frame < span["scene_to"]:
            rel = (scene_frame - span["scene_from"]) * span["stretch"]
            hits.append(span["start_s"] + rel / fps / span["speed"])
    return hits


def compose(edl: dict, out_path: Path, audio: Path | None = None) -> dict:
    size = tuple(edl.get("size", [1080, 1920]))
    fps = int(edl.get("fps", 30))
    fonts = {**DEFAULT_FONTS, **(edl.get("fonts") or {})}
    styles = _styles(fonts, size[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        segs, t, timeline = [], 0.0, []
        for i, seg in enumerate(edl["segments"]):
            path, dur = build_segment(seg, i, size, fps, work)
            segs.append(path)
            timeline.append({"start": t, "end": t + dur, "label": seg.get("label"),
                             "label_style": seg.get("label_style", "chip")})
            t += dur
        total = t
        concat_list = work / "concat.txt"
        concat_list.write_text("".join(f"file '{p}'\n" for p in segs))
        body = work / "body.mp4"
        _run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
              "-i", str(concat_list), "-c", "copy", str(body)])

        overlays: list[tuple[Path, float, float]] = []
        for j, cap in enumerate(edl.get("captions", [])):
            start = float(cap.get("start", 0.0))
            start = total + start if start < 0 else start
            end = cap.get("end")
            end = total if end is None else (total + float(end) if float(end) < 0 else float(end))
            png = work / f"cap_{j:02d}.png"
            render_caption_png(cap["text"], styles[cap.get("style", "title")], size, png)
            overlays.append((png, start, end))
        for k, span in enumerate(timeline):
            if span["label"]:
                png = work / f"label_{k:02d}.png"
                render_caption_png(span["label"], styles[span["label_style"]], size, png)
                overlays.append((png, span["start"], span["end"]))

        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(body)]
        for png, _, _ in overlays:
            cmd += ["-i", str(png)]
        if audio is not None:
            cmd += ["-i", str(audio)]
        else:
            cmd += ["-f", "lavfi", "-t", f"{total:.3f}", "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=48000"]
        chain, last = [], "[0:v]"
        for n, (_, s, e) in enumerate(overlays, start=1):
            tag = f"[v{n}]"
            chain.append(f"{last}[{n}:v]overlay=0:0:enable='between(t,{s:.3f},{e:.3f})'{tag}")
            last = tag
        audio_idx = len(overlays) + 1
        if chain:
            cmd += ["-filter_complex", ";".join(chain), "-map", last]
        else:
            cmd += ["-map", "0:v"]
        cmd += ["-map", f"{audio_idx}:a", "-c:v", "libx264", "-preset", "slow",
                "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart", "-shortest", str(out_path)]
        _run(cmd)
    return {"out": str(out_path), "duration_s": round(total, 2),
            "segments": len(segs), "overlays": len(overlays)}

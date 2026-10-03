"""Shorts template DSL: edits expressed relative to moments, never frames.

A template (``config/shorts/templates/<name>.yaml``) lists ordered *slots*;
each slot has ordered ``candidates`` (camera rigs). Every frame reference is
a moment expression -- ``strike-40``, ``impact+5``, ``line_cross`` -- and
every player reference is ``@scorer`` / ``@keeper`` / ``@goal_end``::

    slots:
      - id: keeper_eyes
        label: KEEPER CAM
        candidates:
          - {camera: "eyes:@keeper", from: strike-41, to: strike-9,
             rig: {eyes_fov_deg: 50}, pad: [10, 37], subject: "@keeper"}
          - {camera: "goalline:@goal_end", from: strike-41, to: strike-9}

``resolve_template`` picks, per slot, the first candidate whose framing check
passes, and emits render pass specs (``scripts/render_experiments`` entry
shape) plus a compositor EDL whose segments name passes (the stage fills in
the mp4 paths with ``fill_sources``). Unknown moments or bare frame numbers
are errors. Pure; the stage supplies ``framing_check``.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Callable, Mapping

import yaml

MOMENT_NAMES = ("strike", "line_cross", "impact", "keeper_dive", "buildup_start")
ROLE_REFS = ("scorer", "keeper", "goal_end")
_EXPR_RE = re.compile(r"^\s*([a-z_]+)\s*(?:([+-])\s*(\d+))?\s*$")
_CAMERA_RE = re.compile(
    r"^(broadcast|drone|orbit|chase|dolly"
    r"|(?:pov|ots|eyes):[A-Za-z0-9_-]+"
    r"|(?:goal|goalline):(?:left|right))$")
_REF_RE = re.compile(r"@([a-z_]+)")
DEFAULT_PAD = (6, 6)
TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "config" / "shorts" / "templates"
_CAND_KEYS = {"camera", "from", "to", "rig", "pad", "time_stretch", "freeze_at", "freeze_s",
              "hold_s", "speed", "flash", "label", "label_style", "subject", "framing", "style"}
_SLOT_KEYS = {"id", "candidates", "optional"} | _CAND_KEYS


class ShortsTemplateError(ValueError):
    """Template/DSL problem (unknown moment, frame literal, bad camera ...)."""


# --- expressions ---------------------------------------------------------

def check_moment_expr(expr: object, where: str = "") -> None:
    """Syntax check only (no moments needed): raises on literals / unknown names."""
    if isinstance(expr, bool) or isinstance(expr, (int, float)):
        raise ShortsTemplateError(
            f"{where}: bare frame number {expr!r} is not allowed; use a moment expression "
            f"like 'strike-40' (moments: {', '.join(MOMENT_NAMES)})")
    m = _EXPR_RE.match(str(expr))
    if not m:
        raise ShortsTemplateError(f"{where}: cannot parse moment expression {expr!r}")
    if m.group(1) not in MOMENT_NAMES:
        raise ShortsTemplateError(
            f"{where}: unknown moment {m.group(1)!r} in {expr!r}; known: {', '.join(MOMENT_NAMES)}")


def eval_moment_expr(expr: object, moments: Mapping, where: str = "") -> int:
    check_moment_expr(expr, where)
    m = _EXPR_RE.match(str(expr))
    base = moments.get(m.group(1))
    if base is None:
        raise ShortsTemplateError(
            f"{where}: moment {m.group(1)!r} could not be derived for this shot (expr {expr!r})")
    off = int(m.group(3) or 0) * (-1 if m.group(2) == "-" else 1)
    return int(base) + off


def resolve_refs(value, moments: Mapping, where: str = ""):
    """Substitute ``@scorer``/``@keeper``/``@goal_end`` in strings (recursively)."""
    if isinstance(value, str):
        def sub(m):
            ref = m.group(1)
            if ref not in ROLE_REFS:
                raise ShortsTemplateError(f"{where}: unknown reference @{ref}; known: "
                                          + ", ".join("@" + r for r in ROLE_REFS))
            key = {"scorer": "scorer_pid", "keeper": "keeper_pid", "goal_end": "goal_end"}[ref]
            if moments.get(key) is None:
                raise ShortsTemplateError(f"{where}: @{ref} could not be resolved for this shot")
            return str(moments[key])
        return _REF_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: resolve_refs(v, moments, where) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_refs(v, moments, where) for v in value]
    return value


# --- loading / validation -------------------------------------------------

def _validate_candidate(cand: Mapping, where: str) -> None:
    extra = set(cand) - _CAND_KEYS
    if extra:
        raise ShortsTemplateError(f"{where}: unknown key(s) {sorted(extra)}")
    for key in ("camera", "from", "to"):
        if key not in cand:
            raise ShortsTemplateError(f"{where}: missing '{key}'")
    check_moment_expr(cand["from"], f"{where}.from")
    check_moment_expr(cand["to"], f"{where}.to")
    if "freeze_at" in cand:
        check_moment_expr(cand["freeze_at"], f"{where}.freeze_at")
    for k, v in (cand.get("rig") or {}).items():
        if k.endswith("_frame") and v != -1:
            check_moment_expr(v, f"{where}.rig.{k}")
    ts = cand.get("time_stretch", 1)
    if isinstance(ts, bool) or not isinstance(ts, int) or not 1 <= ts <= 9:
        raise ShortsTemplateError(f"{where}: time_stretch must be an int 1..9, got {ts!r}")
    # '@role' camera ids are validated after substitution; plain ones now.
    if "@" not in str(cand["camera"]) and not _CAMERA_RE.match(str(cand["camera"])):
        raise ShortsTemplateError(f"{where}: invalid camera id {cand['camera']!r}")


def validate_template(raw: object) -> dict:
    """Validate + normalise a template mapping (never mutates ``raw``)."""
    if not isinstance(raw, dict):
        raise ShortsTemplateError("template must be a mapping")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ShortsTemplateError("template needs a non-empty 'name'")
    slots_in = raw.get("slots")
    if not isinstance(slots_in, list) or not slots_in:
        raise ShortsTemplateError(f"template {name!r}: 'slots' must be a non-empty list")
    slots, seen = [], set()
    for i, s in enumerate(slots_in):
        if not isinstance(s, dict):
            raise ShortsTemplateError(f"template {name!r}: slot {i} must be a mapping")
        extra = set(s) - _SLOT_KEYS
        if extra:
            raise ShortsTemplateError(f"template {name!r} slot {i}: unknown key(s) {sorted(extra)}")
        sid = s.get("id")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ShortsTemplateError(f"template {name!r} slot {i}: needs a unique string 'id'")
        seen.add(sid)
        defaults = {k: v for k, v in s.items() if k not in ("id", "candidates", "optional")}
        if "candidates" in s and not s["candidates"]:
            raise ShortsTemplateError(f"template {name!r} slot {sid!r}: empty candidates")
        cands = s.get("candidates") or [{}]
        merged = []
        for j, c in enumerate(cands):
            mc = {**copy.deepcopy(defaults), **copy.deepcopy(c)}
            _validate_candidate(mc, f"template {name!r} slot {sid!r} candidate {j}")
            merged.append(mc)
        slots.append({"id": sid, "optional": bool(s.get("optional", False)), "candidates": merged})
    size = raw.get("size", [1080, 1920])
    return {
        "name": name,
        "size": list(size),
        "fps": int(raw.get("fps", 30)),
        "style_name": raw.get("style_name"),
        "style": copy.deepcopy(raw.get("style")),
        "fonts": copy.deepcopy(raw.get("fonts")),
        "slots": slots,
        "captions": copy.deepcopy(raw.get("captions") or []),
    }


def load_template(name_or_path: str | Path) -> dict:
    p = Path(name_or_path)
    if not p.suffix:
        p = TEMPLATE_DIR / f"{name_or_path}.yaml"
    return validate_template(yaml.safe_load(p.read_text()))


def available_templates() -> list[str]:
    return sorted(p.stem for p in TEMPLATE_DIR.glob("*.yaml"))


# --- resolution -------------------------------------------------------------

FramingCheck = Callable[[dict, int, int, "str | None", "list[str]", dict], object]


def _resolve_candidate(tpl: Mapping, slot_id: str, k: int, cand: Mapping,
                       moments: Mapping) -> dict:
    where = f"template {tpl['name']!r} slot {slot_id!r} candidate {k}"
    c = resolve_refs(copy.deepcopy(dict(cand)), moments, where)
    cut_from = eval_moment_expr(cand["from"], moments, f"{where}.from")
    cut_to = eval_moment_expr(cand["to"], moments, f"{where}.to")
    if cut_from >= cut_to:
        raise ShortsTemplateError(f"{where}: empty cut ({cand['from']} -> {cand['to']} = {cut_from}..{cut_to})")
    if not _CAMERA_RE.match(c["camera"]):
        raise ShortsTemplateError(f"{where}: invalid camera id {c['camera']!r} after resolution")
    rig = dict(c.get("rig") or {})
    for key, v in list(rig.items()):
        if key.endswith("_frame") and v != -1:
            rig[key] = eval_moment_expr(cand["rig"][key], moments, f"{where}.rig.{key}")
    pad_a, pad_b = c.get("pad", DEFAULT_PAD)
    window = [max(0, cut_from - int(pad_a)), cut_to + int(pad_b)]
    freeze = None
    if "freeze_at" in cand:
        freeze = eval_moment_expr(cand["freeze_at"], moments, f"{where}.freeze_at")
        if not cut_from < freeze < cut_to:
            raise ShortsTemplateError(f"{where}: freeze_at {freeze} outside cut {cut_from}..{cut_to}")
    subject = c.get("subject")
    exclude = [c["camera"].split(":", 1)[1]] if c["camera"].startswith("eyes:") else []
    return {"cand": c, "rig": rig, "cut": (cut_from, cut_to), "window": window,
            "framing": dict(c.get("framing") or {}),
            "freeze": freeze, "subject": subject, "exclude": exclude,
            "stretch": int(c.get("time_stretch", 1))}


def _pass_spec(tpl: Mapping, pass_id: str, r: Mapping) -> dict:
    c = r["cand"]
    return {
        "id": pass_id, "camera": c["camera"], "rig": r["rig"],
        "style": copy.deepcopy(c.get("style", tpl.get("style"))),
        "style_name": tpl.get("style_name"),
        "frames": list(r["window"]), "time_stretch": r["stretch"],
        "vertical": True, "speed": None,
    }


def _segments(slot_id: str, pass_id: str, r: Mapping) -> list[dict]:
    c, (a, b) = r["cand"], r["cut"]
    base = {"pass": pass_id, "slot": slot_id, "first_frame": r["window"][0],
            "stretch": r["stretch"]}
    if c.get("speed") is not None:
        base["speed"] = float(c["speed"])
    for k in ("label", "label_style"):
        if c.get(k) is not None:
            base[k] = c[k]
    if r["freeze"] is None:
        seg = {**base, "from": a, "to": b}
        if c.get("hold_s"):
            seg["hold_s"] = float(c["hold_s"])
        if c.get("flash"):
            seg["flash"] = True
        return [seg]
    first = {**base, "from": a, "to": r["freeze"], "hold_s": float(c.get("freeze_s", 1.4)),
             "freeze": True}
    if c.get("flash"):
        first["flash"] = True
    second = {**base, "from": r["freeze"], "to": b}
    if c.get("hold_s"):
        second["hold_s"] = float(c["hold_s"])
    return [first, second]


def _seg_duration(seg: Mapping, fps: float) -> float:
    n = (seg["to"] - seg["from"]) * seg["stretch"]
    return n / fps / float(seg.get("speed", 1.0)) + float(seg.get("hold_s", 0.0))


def _resolve_captions(caps: list, segments: list[dict], fps: float,
                      dropped_slots: set[str]) -> list[dict]:
    spans, t = {}, 0.0
    for seg in segments:
        d = _seg_duration(seg, fps)
        key = seg["slot"]
        s = spans.setdefault(key, {"start": t, "end": t + d, "freeze": None})
        s["end"] = t + d
        if seg.get("freeze"):
            s["freeze"] = (t + d - float(seg["hold_s"]), t + d)
        t += d
    out = []
    for cap in caps:
        cap = dict(cap)
        over = cap.pop("over", None)
        if over:
            slot, _, part = over.partition(".")
            if slot in dropped_slots:
                continue
            if slot not in spans:
                raise ShortsTemplateError(f"caption over {over!r}: no such slot")
            if part == "freeze":
                if spans[slot]["freeze"] is None:
                    raise ShortsTemplateError(f"caption over {over!r}: slot has no freeze")
                cap["start"], cap["end"] = spans[slot]["freeze"]
            else:
                cap["start"], cap["end"] = spans[slot]["start"], spans[slot]["end"]
        out.append(cap)
    return out


def resolve_template(
    template: Mapping,
    moments: Mapping,
    *,
    framing_check: FramingCheck | None = None,
    captions_override: list | None = None,
    candidate_pins: Mapping[str, int] | None = None,
) -> dict:
    """Resolve ``template`` against ``moments``.

    ``framing_check(pass_spec, cut_from, cut_to, subject_pid, exclude_pids,
    limit_overrides)``
    returns a ``FramingResult`` (or ``None`` to skip); candidates failing it
    are recorded in ``slots[i]['rejected']`` and the next one is tried.
    ``candidate_pins`` ({slot_id: candidate index}) is the operator override:
    a pinned candidate skips the framing check.

    Returns ``{template, passes, edl, slots, ok}``; ``ok`` is False when a
    non-optional slot has no passing candidate (its segments are omitted).
    """
    tpl = validate_template(template)      # idempotent on normalised templates
    fps = float(tpl.get("fps", 30))
    passes: list[dict] = []
    segments: list[dict] = []
    slot_report: list[dict] = []
    dropped: set[str] = set()
    ok = True
    for slot in tpl["slots"]:
        rejected, chosen = [], None
        pin = (candidate_pins or {}).get(slot["id"])
        order = [pin] if pin is not None else range(len(slot["candidates"]))
        for k in order:
            r = _resolve_candidate(tpl, slot["id"], k, slot["candidates"][k], moments)
            pass_id = f"{tpl['name']}_{slot['id']}" + (f"_c{k}" if k else "")
            spec = _pass_spec(tpl, pass_id, r)
            result = None
            if framing_check is not None and pin is None:
                subj = r["subject"]
                result = framing_check(spec, r["cut"][0], r["cut"][1], subj, r["exclude"],
                                       r["framing"])
                if result is not None and not result.ok:
                    rejected.append({"candidate": k, "pass": pass_id, "framing": result.to_dict()})
                    continue
            chosen = (k, spec, r, result)
            break
        if chosen is None:
            dropped.add(slot["id"])
            ok = ok and slot["optional"]
            slot_report.append({"id": slot["id"], "chosen": None,
                                "status": "dropped" if slot["optional"] else "unresolved",
                                "rejected": rejected})
            continue
        k, spec, r, result = chosen
        passes.append(spec)
        segments.extend(_segments(slot["id"], spec["id"], r))
        slot_report.append({
            "id": slot["id"], "chosen": k, "pass": spec["id"], "status": "ok",
            "cut": list(r["cut"]), "rejected": rejected,
            "framing": result.to_dict() if result is not None else None})
    caps = captions_override if captions_override is not None else tpl["captions"]
    edl = {
        "size": list(tpl["size"]), "fps": int(fps), "segments": segments,
        "captions": _resolve_captions(caps, segments, fps, dropped),
    }
    if tpl.get("fonts"):
        edl["fonts"] = dict(tpl["fonts"])
    return {"template": tpl["name"], "passes": passes, "edl": edl,
            "slots": slot_report, "ok": ok}


def fill_sources(edl: Mapping, src_for: Callable[[str], str | Path]) -> dict:
    """Compositor-ready EDL: set ``src`` from each segment's ``pass`` id and
    drop the template-only keys (``pass``, ``slot``, ``freeze``)."""
    out = copy.deepcopy(dict(edl))
    segs = []
    for seg in out["segments"]:
        seg = {k: v for k, v in seg.items() if k not in ("pass", "slot", "freeze")} | {
            "src": str(src_for(seg["pass"]))}
        segs.append(seg)
    out["segments"] = segs
    return out

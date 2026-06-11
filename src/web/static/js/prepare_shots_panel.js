// prepare_shots_panel.js — the dashboard's Prepare Shots panel.
//
// Loaded as a plain <script> before index.html's inline code and relies
// on these globals from there: makePanel, makeToolbarBtn, emptyNote,
// fetchJsonOrNull, attachToJob, loadStages, selectStage, currentStage,
// renderMatchInfoForm, renderMultiShotStatus.
//
// Server contract:
//   GET   /api/shots/manifest      full manifest (groups + excluded shots)
//   GET   /api/shots/features      per-shot classifier sidecar
//   GET   /api/sync                v2 group-scoped sync map
//   POST  /api/sync                one group's alignments
//   POST  /api/sync/auto           re-run the auto-aligner for a group
//   PATCH /api/shots/bulk          discard/restore/regroup shots
//   POST  /api/shots/upload-reel   full-reel ingest (split mode job)
//   POST  /api/shots/upload        pre-trimmed clip upload (copy mode)
//   GET   /api/shots/{id}/thumb    thumbnail JPEG
//   GET   /api/video/{id}          clip stream

// The active sync editor owns window-level drag listeners, a rAF loop
// and playing <video> elements. Whoever tears its DOM down (tab switch,
// whole-panel refresh) must run this first or the listeners/rAF leak
// across re-renders.
let _activeSyncEditorCleanup = null;

function _teardownActiveSyncEditor() {
  if (_activeSyncEditorCleanup) {
    try { _activeSyncEditorCleanup(); } catch {}
    _activeSyncEditorCleanup = null;
  }
}

// Drop-target dragleave fires when the cursor enters a child element;
// only reset the highlight when the pointer actually left the target.
function _trulyLeft(el, ev) {
  return !(ev.relatedTarget && el.contains(ev.relatedTarget));
}

async function renderPrepareShots(panel) {
  _teardownActiveSyncEditor();
  // Match-info form intentionally not rendered for now — the panel is
  // focused on shot review; re-add renderMatchInfoForm(panel) here when
  // match metadata matters again.

  const [manifest, features, sync] = await Promise.all([
    fetchJsonOrNull("/api/shots/manifest"),
    fetchJsonOrNull("/api/shots/features"),
    fetchJsonOrNull("/api/sync"),
  ]);

  const refresh = async () => {
    panel.innerHTML = "";
    await renderPrepareShots(panel);
  };

  _renderIngestCard(panel, refresh);

  const shots = (manifest && manifest.shots) || [];
  if (shots.length === 0) {
    emptyNote(panel, "No shots yet — drop a full highlights reel above to auto-split it, use Add Shots for pre-trimmed clips, or pass --input to recon.py.");
    return;
  }

  const model = _buildShotModel(manifest, features || {}, sync || { groups: [] });
  _renderGroupsBoard(panel, model, refresh);
  _renderDroppedTray(panel, model, refresh);
  await renderMultiShotStatus(panel, model.activeIds);
  _renderGroupSyncEditor(panel, model, refresh);
}

// ── Model ────────────────────────────────────────────────────────────
// Joins manifest shots ↔ features sidecar ↔ sync map into the shape the
// board + sync editor render from.
function _buildShotModel(manifest, features, sync) {
  const shotsById = new Map();
  for (const s of manifest.shots) {
    shotsById.set(s.id, { ...s, features: features[s.id] || null });
  }

  const syncByGroup = new Map();
  for (const g of (sync.groups || [])) syncByGroup.set(g.group_id, g);

  // Groups in manifest order; members in the group's stored order,
  // active members only (excluded ones live in the dropped tray).
  const groups = [];
  for (const g of (manifest.groups || [])) {
    const members = g.shot_ids
      .map((sid) => shotsById.get(sid))
      .filter((s) => s && !s.excluded);
    groups.push({ ...g, members, sync: syncByGroup.get(g.id) || null });
  }
  const ungrouped = manifest.shots
    .filter((s) => !s.excluded && !s.group_id)
    .map((s) => shotsById.get(s.id));
  const dropped = manifest.shots
    .filter((s) => s.excluded)
    .map((s) => shotsById.get(s.id));

  return {
    manifest,
    shotsById,
    groups,
    ungrouped,
    ungroupedSync: syncByGroup.get("") || null,
    dropped,
    activeIds: manifest.shots.filter((s) => !s.excluded).map((s) => s.id),
  };
}

// ── Shared plumbing ──────────────────────────────────────────────────

async function _patchShots(updates) {
  const res = await fetch("/api/shots/bulk", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ updates }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

function _toast(msg, ok = true) {
  const el = document.createElement("div");
  el.textContent = msg;
  el.style.cssText = `
    position:fixed;right:18px;bottom:18px;z-index:1000;
    background:${ok ? "#14532d" : "#7f1d1d"};color:#f1f5f9;
    border:1px solid ${ok ? "#22c55e" : "#f87171"};
    border-radius:6px;padding:8px 14px;font-size:12px;
    box-shadow:0 4px 16px rgba(0,0,0,.5);opacity:0;transition:opacity .15s;`;
  document.body.appendChild(el);
  requestAnimationFrame(() => { el.style.opacity = "1"; });
  setTimeout(() => {
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 250);
  }, ok ? 2200 : 4200);
}

async function _mutateShots(updates, okMsg, refresh) {
  try {
    await _patchShots(updates);
    if (okMsg) _toast(okMsg);
    await refresh();
  } catch (err) {
    _toast(`Failed: ${err.message || err}`, false);
  }
}

function _nextFreeGroupId(model) {
  const taken = new Set(model.manifest.groups.map((g) => g.id));
  for (let i = 1; i < 1000; i++) {
    const gid = `g${String(i).padStart(2, "0")}`;
    if (!taken.has(gid)) return gid;
  }
  return `g${Date.now() % 100000}`;
}

function _fmtClock(seconds) {
  if (!(seconds >= 0)) return "?";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function _chip(text, fg, bg, title) {
  const c = document.createElement("span");
  c.textContent = text;
  c.style.cssText = `display:inline-block;padding:1px 7px;border-radius:9px;font-size:10px;font-weight:600;letter-spacing:.03em;color:${fg};background:${bg};white-space:nowrap;`;
  if (title) c.title = title;
  return c;
}

// ── Ingest card (reel upload + Add Shots) ────────────────────────────

function _renderIngestCard(panel, refresh) {
  const { wrap, body } = makePanel("Ingest");
  body.style.cssText += "display:flex;gap:14px;flex-wrap:wrap;padding:14px 16px;";

  // Reel drop-zone.
  const zone = document.createElement("div");
  zone.style.cssText = `
    flex:1 1 320px;min-height:84px;border:2px dashed #334155;border-radius:8px;
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    gap:4px;color:#94a3b8;font-size:12px;cursor:pointer;padding:12px;
    transition:border-color .15s,background .15s;text-align:center;`;
  zone.innerHTML = `<div style="font-size:13px;font-weight:600;color:#e2e8f0;">Drop a full highlights reel here</div>
    <div>auto-splits into shots, drops reactions, groups highlights &amp; aligns replays</div>
    <div style="color:#64748b;">…or click to pick an .mp4</div>`;
  const reelInput = document.createElement("input");
  reelInput.type = "file";
  reelInput.accept = "video/mp4,.mp4";
  reelInput.style.display = "none";
  const reelStatus = document.createElement("div");
  reelStatus.style.cssText = "flex-basis:100%;color:#94a3b8;font-size:12px;min-height:14px;";

  zone.addEventListener("click", () => reelInput.click());
  zone.addEventListener("dragover", (ev) => {
    ev.preventDefault();
    zone.style.borderColor = "#6366f1";
    zone.style.background = "#11142a";
  });
  zone.addEventListener("dragleave", (ev) => {
    if (!_trulyLeft(zone, ev)) return;
    zone.style.borderColor = "#334155";
    zone.style.background = "transparent";
  });
  zone.addEventListener("drop", (ev) => {
    ev.preventDefault();
    zone.style.borderColor = "#334155";
    zone.style.background = "transparent";
    const f = ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0];
    if (f) uploadReel(f);
  });
  reelInput.addEventListener("change", () => {
    if (reelInput.files && reelInput.files[0]) uploadReel(reelInput.files[0]);
    reelInput.value = "";
  });

  async function uploadReel(file) {
    const fd = new FormData();
    fd.append("file", file);
    reelStatus.style.color = "#94a3b8";
    reelStatus.textContent = `Uploading ${file.name} (${(file.size / 1e6).toFixed(0)} MB)…`;
    let body;
    try {
      const res = await fetch("/api/shots/upload-reel", { method: "POST", body: fd });
      body = await res.json().catch(() => ({}));
      if (!res.ok) {
        reelStatus.style.color = "#f87171";
        reelStatus.textContent = `Upload failed: ${body.detail || res.statusText}`;
        return;
      }
    } catch (err) {
      reelStatus.style.color = "#f87171";
      reelStatus.textContent = `Upload failed: ${err.message || err}`;
      return;
    }
    reelStatus.style.color = "#4ade80";
    reelStatus.textContent = `Saved ${body.saved}. Splitting into shots — watch the job log below…`;
    attachToJob(body.job_id, "prepare_shots", async () => {
      await loadStages();
      if (currentStage === "prepare_shots") await refresh();
    });
  }

  // Pre-trimmed clips ("Add Shots", copy mode) — unchanged flow.
  const clipsBox = document.createElement("div");
  clipsBox.style.cssText = "flex:0 0 auto;display:flex;flex-direction:column;gap:8px;justify-content:center;";
  const addBtn = document.createElement("button");
  addBtn.className = "btn btn-primary";
  addBtn.textContent = "Add Shots";
  addBtn.style.fontSize = "12px";
  addBtn.title = "Upload one or more pre-trimmed .mp4 clips and merge them into the shots manifest (no splitting).";
  const clipsInput = document.createElement("input");
  clipsInput.type = "file";
  clipsInput.accept = "video/mp4,.mp4";
  clipsInput.multiple = true;
  clipsInput.style.display = "none";
  const clipsHint = document.createElement("div");
  clipsHint.style.cssText = "color:#64748b;font-size:11px;max-width:180px;";
  clipsHint.textContent = "Pre-trimmed clips, added as ungrouped shots.";
  clipsBox.append(addBtn, clipsInput, clipsHint);

  addBtn.addEventListener("click", () => clipsInput.click());
  clipsInput.addEventListener("change", async () => {
    if (!clipsInput.files || clipsInput.files.length === 0) return;
    const fd = new FormData();
    for (const f of clipsInput.files) fd.append("files", f);
    addBtn.disabled = true;
    reelStatus.style.color = "#94a3b8";
    reelStatus.textContent = `Uploading ${clipsInput.files.length} clip(s)…`;
    let body;
    try {
      const res = await fetch("/api/shots/upload", { method: "POST", body: fd });
      body = await res.json().catch(() => ({}));
      if (!res.ok) {
        reelStatus.style.color = "#f87171";
        reelStatus.textContent = `Upload failed: ${body.detail || res.statusText}`;
        addBtn.disabled = false;
        return;
      }
    } catch (err) {
      reelStatus.style.color = "#f87171";
      reelStatus.textContent = `Upload failed: ${err.message || err}`;
      addBtn.disabled = false;
      return;
    } finally {
      clipsInput.value = "";
    }
    const skippedNote = (body.skipped && body.skipped.length)
      ? ` (skipped ${body.skipped.length}: ${body.skipped.map((s) => `${s.name} — ${s.reason}`).join("; ")})`
      : "";
    if (!body.saved || body.saved.length === 0) {
      reelStatus.style.color = "#f59e0b";
      reelStatus.textContent = `Nothing uploaded${skippedNote}`;
      addBtn.disabled = false;
      return;
    }
    reelStatus.style.color = "#4ade80";
    reelStatus.textContent = `Uploaded ${body.saved.length} shot(s)${skippedNote}. Registering…`;
    if (body.job_id) {
      attachToJob(body.job_id, "prepare_shots", async () => {
        addBtn.disabled = false;
        await loadStages();
        if (currentStage === "prepare_shots") await refresh();
      });
    } else {
      addBtn.disabled = false;
      await refresh();
    }
  });

  body.append(zone, reelInput, clipsBox, reelStatus);
  panel.appendChild(wrap);
}

// ── Shot tiles ───────────────────────────────────────────────────────

function _badgesFor(shot, groupSync) {
  const badges = [];
  const f = shot.features;
  if (f && f.scale) {
    const label = { wide: "WIDE", medium: "MED", tight: "TIGHT" }[f.scale] || f.scale;
    badges.push(_chip(label, "#cbd5e1", "#1e293b",
      `pitch ratio ${f.pitch_ratio_median ?? "?"}`));
  }
  const sf = (f && f.speed_factor) || shot.speed_factor || 1.0;
  if (sf >= 1.25) {
    badges.push(_chip(`REPLAY ×${sf.toFixed(1)}`, "#fde68a", "#451a03",
      "slow-motion replay, retimed to real time at extraction"));
  }
  if (groupSync) {
    const a = (groupSync.alignments || []).find((x) => x.shot_id === shot.id);
    if (a) {
      const isRef = groupSync.reference_shot === shot.id;
      if (isRef) {
        badges.push(_chip("REF", "#c7d2fe", "#312e81", "group sync reference (offset 0)"));
      } else if (a.method === "manual") {
        badges.push(_chip("manual", "#c7d2fe", "#312e81", `offset ${a.frame_offset}f, operator-set`));
      } else {
        const low = a.confidence < 0.5;
        badges.push(_chip(
          `auto ${a.confidence.toFixed(2)}`,
          low ? "#fecaca" : "#bbf7d0",
          low ? "#450a0a" : "#052e16",
          `offset ${a.frame_offset}f via ${a.method}`,
        ));
      }
    }
  }
  return badges;
}

function _shotTile(shot, opts) {
  // opts: {groupSync, draggable, actions: [{icon,title,onClick}], menu: [{label,onClick}]}
  const tile = document.createElement("div");
  tile.style.cssText = `
    flex:0 0 auto;width:172px;background:#0f1117;border:1px solid #1f2937;
    border-radius:8px;overflow:hidden;display:flex;flex-direction:column;
    transition:border-color .12s;`;
  tile.addEventListener("mouseenter", () => { tile.style.borderColor = "#475569"; });
  tile.addEventListener("mouseleave", () => { tile.style.borderColor = "#1f2937"; });

  if (opts.draggable) {
    tile.draggable = true;
    tile.addEventListener("dragstart", (ev) => {
      ev.dataTransfer.setData("text/shot-id", shot.id);
      ev.dataTransfer.effectAllowed = "move";
      tile.style.opacity = "0.5";
    });
    tile.addEventListener("dragend", () => { tile.style.opacity = "1"; });
  }

  // Media: thumbnail poster that plays muted on hover.
  const media = document.createElement("video");
  media.poster = `/api/shots/${encodeURIComponent(shot.id)}/thumb`;
  media.muted = true;
  media.loop = true;
  media.playsInline = true;
  media.preload = "none";
  media.style.cssText = "width:100%;aspect-ratio:16/9;object-fit:cover;background:#000;display:block;cursor:grab;";
  media.title = "Hover to preview";
  media.addEventListener("mouseenter", () => {
    if (!media.src) media.src = `/api/video/${encodeURIComponent(shot.id)}`;
    media.play().catch(() => {});
  });
  media.addEventListener("mouseleave", () => { media.pause(); });
  tile.appendChild(media);

  const info = document.createElement("div");
  info.style.cssText = "padding:6px 8px;display:flex;flex-direction:column;gap:5px;";
  const titleRow = document.createElement("div");
  titleRow.style.cssText = "display:flex;align-items:center;gap:6px;font-size:12px;color:#f1f5f9;font-weight:600;";
  const name = document.createElement("span");
  name.textContent = shot.id;
  name.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
  const dur = document.createElement("span");
  const seconds = shot.end_time - shot.start_time;
  dur.textContent = `${seconds.toFixed(1)}s`;
  dur.style.cssText = "margin-left:auto;color:#64748b;font-weight:400;font-size:11px;";
  if (shot.source_start_s >= 0) {
    dur.title = `source ${_fmtClock(shot.source_start_s)}–${_fmtClock(shot.source_end_s)}`;
  }
  titleRow.append(name, dur);
  info.appendChild(titleRow);

  const badgeRow = document.createElement("div");
  badgeRow.dataset.role = "badges";
  badgeRow.style.cssText = "display:flex;gap:4px;flex-wrap:wrap;min-height:16px;";
  for (const b of _badgesFor(shot, opts.groupSync)) badgeRow.appendChild(b);
  info.appendChild(badgeRow);

  const actionsRow = document.createElement("div");
  actionsRow.style.cssText = "display:flex;gap:4px;align-items:center;";
  for (const a of opts.actions || []) {
    const btn = document.createElement("button");
    btn.textContent = a.icon;
    btn.title = a.title;
    btn.style.cssText = "background:#1e293b;color:#cbd5e1;border:1px solid #334155;border-radius:4px;padding:2px 7px;font-size:11px;cursor:pointer;";
    btn.addEventListener("click", (ev) => { ev.stopPropagation(); a.onClick(); });
    actionsRow.appendChild(btn);
  }
  if (opts.menu && opts.menu.length) {
    const menuBtn = document.createElement("button");
    menuBtn.textContent = "⋮";
    menuBtn.title = "More actions";
    menuBtn.style.cssText = "margin-left:auto;background:#1e293b;color:#cbd5e1;border:1px solid #334155;border-radius:4px;padding:2px 8px;font-size:11px;cursor:pointer;";
    menuBtn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      _openTileMenu(menuBtn, opts.menu);
    });
    actionsRow.appendChild(menuBtn);
  }
  info.appendChild(actionsRow);
  tile.appendChild(info);
  return tile;
}

function _openTileMenu(anchorEl, items) {
  document.querySelectorAll("[data-tile-menu]").forEach((m) => m.remove());
  const menu = document.createElement("div");
  menu.dataset.tileMenu = "1";
  const rect = anchorEl.getBoundingClientRect();
  menu.style.cssText = `
    position:fixed;left:${Math.round(rect.left)}px;top:${Math.round(rect.bottom + 4)}px;
    z-index:1001;background:#16182c;border:1px solid #334155;border-radius:6px;
    box-shadow:0 8px 24px rgba(0,0,0,.6);min-width:170px;padding:4px;`;
  for (const item of items) {
    const row = document.createElement("div");
    row.textContent = item.label;
    row.style.cssText = "padding:6px 10px;font-size:12px;color:#e2e8f0;border-radius:4px;cursor:pointer;";
    row.addEventListener("mouseenter", () => { row.style.background = "#1e2440"; });
    row.addEventListener("mouseleave", () => { row.style.background = "transparent"; });
    row.addEventListener("click", () => { menu.remove(); item.onClick(); });
    menu.appendChild(row);
  }
  const close = (ev) => {
    if (!menu.contains(ev.target)) {
      menu.remove();
      window.removeEventListener("mousedown", close);
    }
  };
  window.addEventListener("mousedown", close);
  document.body.appendChild(menu);
}

// ── Highlight groups board ───────────────────────────────────────────

function _renderGroupsBoard(panel, model, refresh) {
  const activeCount = model.activeIds.length;
  const { wrap, body } = makePanel(
    `Highlight Groups (${model.groups.length} group${model.groups.length === 1 ? "" : "s"} · ${activeCount} active shots)`,
  );
  body.style.cssText += "display:flex;flex-direction:column;gap:12px;padding:14px 16px;";

  const hint = document.createElement("div");
  hint.style.cssText = "font-size:12px;color:#94a3b8;";
  hint.textContent = "Each card is one highlight (live action + its replays). Drag tiles between cards to regroup, ✕ to drop a shot into the tray below, hover a thumbnail to preview.";
  body.appendChild(hint);

  const renderableGroups = model.groups.filter((g) => g.members.length > 0);
  for (const group of renderableGroups) {
    body.appendChild(_groupCard(group, model, refresh));
  }
  if (model.ungrouped.length) {
    body.appendChild(_groupCard({
      id: "",
      label: "Ungrouped",
      boundary_rule: "manual",
      boundary_confidence: 1.0,
      members: model.ungrouped,
      sync: model.ungroupedSync,
    }, model, refresh));
  }

  // "New group" drop target.
  const newZone = document.createElement("div");
  newZone.style.cssText = `
    border:2px dashed #334155;border-radius:8px;padding:10px;text-align:center;
    color:#64748b;font-size:12px;transition:border-color .12s,background .12s;`;
  newZone.textContent = "Drag a shot here to start a new group";
  newZone.addEventListener("dragover", (ev) => {
    ev.preventDefault();
    newZone.style.borderColor = "#6366f1";
    newZone.style.background = "#11142a";
  });
  newZone.addEventListener("dragleave", (ev) => {
    if (!_trulyLeft(newZone, ev)) return;
    newZone.style.borderColor = "#334155";
    newZone.style.background = "transparent";
  });
  newZone.addEventListener("drop", (ev) => {
    ev.preventDefault();
    const sid = ev.dataTransfer.getData("text/shot-id");
    if (!sid) return;
    _mutateShots([{ shot_id: sid, group_id: _nextFreeGroupId(model) }],
      `${sid} → new group`, refresh);
  });
  body.appendChild(newZone);

  panel.appendChild(wrap);
}

function _groupCard(group, model, refresh) {
  const card = document.createElement("div");
  card.style.cssText = `
    background:#0a0e1a;border:1px solid #1f2937;border-radius:8px;
    padding:10px 12px;display:flex;flex-direction:column;gap:8px;
    transition:border-color .12s;`;

  // Drop target for regrouping.
  card.addEventListener("dragover", (ev) => {
    ev.preventDefault();
    card.style.borderColor = "#6366f1";
  });
  card.addEventListener("dragleave", (ev) => {
    if (!_trulyLeft(card, ev)) return;
    card.style.borderColor = "#1f2937";
  });
  card.addEventListener("drop", (ev) => {
    ev.preventDefault();
    card.style.borderColor = "#1f2937";
    const sid = ev.dataTransfer.getData("text/shot-id");
    if (!sid) return;
    const already = group.members.some((m) => m.id === sid);
    if (already) return;
    _mutateShots([{ shot_id: sid, group_id: group.id }],
      `${sid} → ${group.label}`, refresh);
  });

  // Header row.
  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;gap:8px;flex-wrap:wrap;";
  const title = document.createElement("span");
  title.style.cssText = "font-size:13px;font-weight:700;color:#f1f5f9;";
  title.textContent = group.label;
  header.appendChild(title);

  const meta = document.createElement("span");
  meta.style.cssText = "font-size:11px;color:#64748b;";
  const starts = group.members.map((m) => m.source_start_s).filter((v) => v >= 0);
  const ends = group.members.map((m) => m.source_end_s).filter((v) => v >= 0);
  const range = starts.length
    ? ` · ${_fmtClock(Math.min(...starts))}–${_fmtClock(Math.max(...ends))}`
    : "";
  meta.textContent = `${group.id || "—"} · ${group.members.length} shot${group.members.length === 1 ? "" : "s"}${range}`;
  header.appendChild(meta);

  if (group.boundary_confidence < 0.9 && group.boundary_rule !== "manual") {
    header.appendChild(_chip(
      `boundary ${group.boundary_rule} ·${group.boundary_confidence.toFixed(2)}`,
      "#fde68a", "#451a03",
      "Low-confidence auto boundary — check whether this group should merge with its neighbour.",
    ));
  }

  const spacer = document.createElement("span");
  spacer.style.cssText = "margin-left:auto;";
  header.appendChild(spacer);

  if (group.members.length >= 2 && group.id) {
    const alignBtn = makeToolbarBtn("⟳ Re-align", "#334155");
    alignBtn.title = "Re-run the motion-profile auto-aligner for this group (overwrites auto offsets, keeps manual ones unless none exist).";
    alignBtn.addEventListener("click", async () => {
      alignBtn.disabled = true;
      try {
        const res = await fetch("/api/sync/auto", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ group_id: group.id, force: true }),
        });
        const body = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(body.detail || res.statusText);
        _toast(`Re-aligned ${body.aligned} shot(s) in ${group.label}`);
        await refresh();
      } catch (err) {
        _toast(`Re-align failed: ${err.message || err}`, false);
        alignBtn.disabled = false;
      }
    });
    header.appendChild(alignBtn);
  }

  if (group.id) {
    const mergeBtn = makeToolbarBtn("⇤ Merge left", "#334155");
    mergeBtn.title = "Merge this group into the previous one (moves every shot).";
    const idx = model.groups.findIndex((g) => g.id === group.id);
    const prev = idx > 0 ? model.groups[idx - 1] : null;
    if (!prev) {
      mergeBtn.disabled = true;
      mergeBtn.style.opacity = "0.4";
    } else {
      mergeBtn.addEventListener("click", () => {
        _mutateShots(
          group.members.map((m) => ({ shot_id: m.id, group_id: prev.id })),
          `${group.label} merged into ${prev.label}`, refresh,
        );
      });
    }
    header.appendChild(mergeBtn);

    const discardBtn = makeToolbarBtn("✕ Discard group", "#7f1d1d");
    discardBtn.title = "Exclude every shot in this group (restorable from the dropped tray).";
    discardBtn.addEventListener("click", () => {
      _mutateShots(
        group.members.map((m) => ({
          shot_id: m.id, excluded: true, exclude_reason: "manual",
        })),
        `${group.label} discarded (${group.members.length} shots)`, refresh,
      );
    });
    header.appendChild(discardBtn);
  }
  card.appendChild(header);

  // Tile strip.
  const strip = document.createElement("div");
  strip.style.cssText = "display:flex;gap:8px;overflow-x:auto;padding-bottom:4px;";
  group.members.forEach((shot, i) => {
    const actions = [
      {
        icon: "✕", title: "Discard this shot (restorable from the dropped tray)",
        onClick: () => _mutateShots(
          [{ shot_id: shot.id, excluded: true, exclude_reason: "manual" }],
          `${shot.id} dropped`, refresh,
        ),
      },
    ];
    const idx = model.groups.findIndex((g) => g.id === group.id);
    const prevGroup = group.id && idx > 0 ? model.groups[idx - 1] : null;
    const nextGroup = group.id && idx >= 0 && idx < model.groups.length - 1
      ? model.groups[idx + 1] : null;
    if (prevGroup) {
      actions.push({
        icon: "◀", title: `Move to ${prevGroup.label}`,
        onClick: () => _mutateShots(
          [{ shot_id: shot.id, group_id: prevGroup.id }],
          `${shot.id} → ${prevGroup.label}`, refresh,
        ),
      });
    }
    if (nextGroup) {
      actions.push({
        icon: "▶", title: `Move to ${nextGroup.label}`,
        onClick: () => _mutateShots(
          [{ shot_id: shot.id, group_id: nextGroup.id }],
          `${shot.id} → ${nextGroup.label}`, refresh,
        ),
      });
    }
    const menu = [
      {
        label: "Move to new group",
        onClick: () => _mutateShots(
          [{ shot_id: shot.id, group_id: _nextFreeGroupId(model) }],
          `${shot.id} → new group`, refresh,
        ),
      },
    ];
    if (group.id && i > 0) {
      menu.push({
        label: "Split group here",
        onClick: () => {
          const gid = _nextFreeGroupId(model);
          const tail = group.members.slice(i).map((m) => ({
            shot_id: m.id, group_id: gid,
          }));
          _mutateShots(tail, `${group.label} split at ${shot.id}`, refresh);
        },
      });
    }
    strip.appendChild(_shotTile(shot, {
      groupSync: group.sync,
      draggable: true,
      actions,
      menu,
    }));
  });
  card.appendChild(strip);
  return card;
}

// ── Dropped tray ─────────────────────────────────────────────────────

function _renderDroppedTray(panel, model, refresh) {
  if (!model.dropped.length) return;
  const details = document.createElement("details");
  details.style.cssText = "background:#16182c;border:1px solid #2d3148;border-radius:8px;margin-bottom:16px;";
  const summary = document.createElement("summary");
  summary.style.cssText = "cursor:pointer;padding:10px 16px;font-size:13px;font-weight:600;color:#cbd5e1;";
  const reasons = {};
  for (const s of model.dropped) {
    const r = s.exclude_reason || "unknown";
    reasons[r] = (reasons[r] || 0) + 1;
  }
  const reasonNote = Object.entries(reasons)
    .map(([r, n]) => `${n} ${r}`).join(", ");
  summary.textContent = `Dropped shots (${model.dropped.length}: ${reasonNote})`;
  details.appendChild(summary);

  const strip = document.createElement("div");
  strip.style.cssText = "display:flex;gap:8px;overflow-x:auto;padding:0 16px 12px;";
  for (const shot of model.dropped) {
    const tile = _shotTile(shot, {
      groupSync: null,
      draggable: false,
      actions: [{
        icon: "↩ Restore",
        title: shot.group_id
          ? `Restore into ${shot.group_id}`
          : "Restore as ungrouped",
        onClick: () => _mutateShots(
          [{ shot_id: shot.id, excluded: false, exclude_reason: "" }],
          `${shot.id} restored`, refresh,
        ),
      }],
      menu: [],
    });
    tile.style.opacity = "0.75";
    // Reason badge on top of the standard ones.
    const badgeRow = tile.querySelector('[data-role="badges"]');
    if (badgeRow) {
      const reasonTips = {
        reaction: "Classified as a crowd/bench reaction shot",
        transition: "Classified as a fade/graphic transition",
        closeup: "Classified as a player close-up (celebration / tight cut — person dominates the frame)",
      };
      badgeRow.prepend(_chip(
        shot.exclude_reason || "dropped", "#fecaca", "#450a0a",
        reasonTips[shot.kind] || "Manually discarded",
      ));
    }
    strip.appendChild(tile);
  }
  details.appendChild(strip);
  panel.appendChild(details);
}

// ── Group-scoped sync editor ─────────────────────────────────────────
// The two-video scrub + draggable timeline, scoped to one highlight
// group at a time (tabs). Offsets are meaningless across groups, so
// the editor only ever compares members of the same highlight.
//
// Sign convention (matches sync_map.py):
//   frame_offset = matched_frame_in_active - matched_frame_in_reference
function _renderGroupSyncEditor(panel, model, refresh) {
  const editableGroups = [
    ...model.groups.filter((g) => g.members.length >= 2),
    ...(model.ungrouped.length >= 2 ? [{
      id: "",
      label: "Ungrouped",
      members: model.ungrouped,
      sync: model.ungroupedSync,
    }] : []),
  ];
  if (!editableGroups.length) return;

  const { wrap, body } = makePanel("Group Sync — align shots within a highlight");
  body.style.padding = "14px 16px";

  // Group tabs.
  const tabs = document.createElement("div");
  tabs.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px;";
  const editorHost = document.createElement("div");
  let activeGid = editableGroups[0].id;

  function renderTabs() {
    tabs.innerHTML = "";
    for (const g of editableGroups) {
      const tab = document.createElement("button");
      const active = g.id === activeGid;
      tab.textContent = `${g.label} (${g.members.length})`;
      tab.style.cssText = `
        background:${active ? "#4f46e5" : "#1e293b"};color:${active ? "#fff" : "#cbd5e1"};
        border:1px solid ${active ? "#6366f1" : "#334155"};border-radius:6px;
        padding:5px 12px;font-size:12px;cursor:pointer;font-weight:${active ? "700" : "400"};`;
      tab.addEventListener("click", () => {
        activeGid = g.id;
        renderTabs();
        renderEditor();
      });
      tabs.appendChild(tab);
    }
  }

  function renderEditor() {
    _teardownActiveSyncEditor();
    editorHost.innerHTML = "";
    const group = editableGroups.find((g) => g.id === activeGid);
    if (group) editorHost.appendChild(_buildGroupSyncEditor(group));
  }

  body.append(tabs, editorHost);
  renderTabs();
  renderEditor();
  panel.appendChild(wrap);
}

function _buildGroupSyncEditor(group) {
  const root = document.createElement("div");
  const shotIds = group.members.map((m) => m.id);
  const fpsMap = new Map();
  const framesByShot = new Map();
  for (const m of group.members) {
    const frames = Math.max(1, m.end_frame - m.start_frame + 1);
    framesByShot.set(m.id, frames);
    const dur = m.end_time - m.start_time;
    fpsMap.set(m.id, dur > 0 ? frames / dur : 25);
  }

  // Seed offsets + per-shot method/confidence from the group's sync.
  const offsetByShot = new Map();
  const methodByShot = new Map();
  for (const a of (group.sync && group.sync.alignments) || []) {
    offsetByShot.set(a.shot_id, a.frame_offset);
    methodByShot.set(a.shot_id, { method: a.method, confidence: a.confidence });
  }
  for (const id of shotIds) {
    if (!offsetByShot.has(id)) offsetByShot.set(id, 0);
    if (!methodByShot.has(id)) {
      methodByShot.set(id, { method: "manual", confidence: 1.0 });
    }
  }

  const savedRef = group.sync && group.sync.reference_shot;
  const initialRef = savedRef && shotIds.includes(savedRef) ? savedRef : shotIds[0];
  const firstActive = shotIds.find((s) => s !== initialRef) || shotIds[0];

  // Header (hint + save status pinned right).
  const header = document.createElement("div");
  header.style.cssText = "display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px;font-size:12px;color:#94a3b8;";
  header.innerHTML = `<span>Scrub both videos to the same instant, then <b>Lock offset</b> — or drag clips on the timeline. Edits become <code>manual</code> and survive re-alignment. Saves to <code>output/shots/sync_map.json</code>.</span>`;
  const status = document.createElement("span");
  status.style.cssText = "color:#94a3b8;font-size:12px;margin-left:auto;";
  header.appendChild(status);
  root.appendChild(header);

  // Two-column video grid.
  const topRow = document.createElement("div");
  topRow.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:12px;";
  root.appendChild(topRow);

  const state = {
    referenceShot: initialRef,
    activeShot: firstActive,
    offsets: offsetByShot,
    methods: methodByShot,
    playing: false,
    rafId: null,
  };

  const refColumn = _buildSyncVideoColumn("Reference (offset = 0)", shotIds, state.referenceShot, fpsMap);
  const activeColumn = _buildSyncVideoColumn("Clip to sync", shotIds, state.activeShot, fpsMap);
  topRow.append(refColumn.wrap, activeColumn.wrap);

  function refreshSelectExclusion() {
    for (const opt of refColumn.select.options) {
      opt.disabled = opt.value === state.activeShot && shotIds.length > 1;
    }
    for (const opt of activeColumn.select.options) {
      opt.disabled = opt.value === state.referenceShot && shotIds.length > 1;
    }
  }

  refColumn.select.addEventListener("change", () => {
    state.referenceShot = refColumn.select.value;
    refColumn.loadShot(state.referenceShot);
    state.offsets.set(state.referenceShot, 0);
    refreshSelectExclusion();
    rebuildTimeline();
    syncActiveToRef();
  });
  activeColumn.select.addEventListener("change", () => {
    state.activeShot = activeColumn.select.value;
    activeColumn.loadShot(state.activeShot);
    refreshSelectExclusion();
    refreshOffsetInput();
    rebuildTimeline();
    syncActiveToRef();
  });

  // Controls row (play / lock / nudge / save).
  const controls = document.createElement("div");
  controls.style.cssText = "display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:10px 0;font-size:12px;color:#cbd5e1;";

  const playBtn = makeToolbarBtn("▶ Play both", "#4f46e5");
  playBtn.style.minWidth = "92px";
  controls.appendChild(playBtn);

  const lockBtn = makeToolbarBtn("Lock offset to current frames", "#334155");
  lockBtn.title = "Set the active clip's offset to (active frame − reference frame) using both videos' current positions.";
  controls.appendChild(lockBtn);

  controls.appendChild(_label("Active offset:"));
  const offsetInput = document.createElement("input");
  offsetInput.type = "number";
  offsetInput.step = "1";
  offsetInput.style.cssText = "width:90px;background:#252840;color:#e2e8f0;border:1px solid #2d3148;border-radius:4px;padding:3px 6px;font-size:12px;";
  controls.appendChild(offsetInput);

  const nudgeBack = makeToolbarBtn("◀", "#334155");
  nudgeBack.title = "Move active clip 1 frame earlier on the timeline (offset −1).";
  const nudgeFwd = makeToolbarBtn("▶", "#334155");
  nudgeFwd.title = "Move active clip 1 frame later on the timeline (offset +1).";
  controls.appendChild(nudgeBack);
  controls.appendChild(nudgeFwd);

  const methodChipHost = document.createElement("span");
  controls.appendChild(methodChipHost);

  const saveBtn = document.createElement("button");
  saveBtn.className = "btn btn-primary";
  saveBtn.style.cssText = "font-size:12px;padding:5px 12px;margin-left:auto;";
  saveBtn.textContent = "Save group";
  controls.appendChild(saveBtn);

  root.appendChild(controls);

  const timelineHost = document.createElement("div");
  root.appendChild(timelineHost);

  function getFps(id) { return fpsMap.get(id) || 25; }
  function getOffset(id) { return state.offsets.get(id) || 0; }

  function refreshMethodChip() {
    methodChipHost.innerHTML = "";
    const m = state.methods.get(state.activeShot);
    if (!m) return;
    if (m.method === "manual") {
      methodChipHost.appendChild(_chip("manual", "#c7d2fe", "#312e81"));
    } else {
      const low = m.confidence < 0.5;
      methodChipHost.appendChild(_chip(
        `auto ${m.confidence.toFixed(2)}`,
        low ? "#fecaca" : "#bbf7d0",
        low ? "#450a0a" : "#052e16",
        m.method,
      ));
    }
  }

  function refreshOffsetInput() {
    offsetInput.value = String(getOffset(state.activeShot));
    refreshMethodChip();
  }

  function syncActiveToRef() {
    const refFps = getFps(state.referenceShot);
    const actFps = getFps(state.activeShot);
    const offset = getOffset(state.activeShot);
    const refFrame = refColumn.video.currentTime * refFps;
    const targetActFrame = refFrame + offset;
    const targetTime = Math.max(0, targetActFrame / actFps);
    if (Math.abs(activeColumn.video.currentTime - targetTime) > 0.05) {
      try { activeColumn.video.currentTime = targetTime; } catch {}
    }
  }

  refColumn.video.addEventListener("seeked", () => { syncActiveToRef(); drawCursor(); });
  refColumn.video.addEventListener("timeupdate", () => drawCursor());
  refColumn.video.addEventListener("loadedmetadata", () => drawCursor());

  function setOffset(shotId, newOffset) {
    if (shotId === state.referenceShot) return;  // reference pinned to 0
    state.offsets.set(shotId, Math.round(newOffset));
    // Operator touched it → it's a manual alignment now.
    state.methods.set(shotId, { method: "manual", confidence: 1.0 });
    if (shotId === state.activeShot) refreshOffsetInput();
    rebuildTimeline();
    if (shotId === state.activeShot) syncActiveToRef();
  }

  function setActiveShot(shotId) {
    if (shotId === state.activeShot) return;
    if (shotId === state.referenceShot) return;
    state.activeShot = shotId;
    activeColumn.select.value = shotId;
    activeColumn.loadShot(shotId);
    refreshSelectExclusion();
    refreshOffsetInput();
    rebuildTimeline();
    syncActiveToRef();
  }

  let drawCursor = () => {};   // overwritten by rebuildTimeline
  let cleanupTimeline = () => {};
  function rebuildTimeline() {
    cleanupTimeline();
    timelineHost.innerHTML = "";
    const tl = _buildSyncTimeline({
      shotIds, framesByShot, fpsMap,
      referenceShot: state.referenceShot,
      activeShot: state.activeShot,
      getOffset,
      getMethod: (id) => state.methods.get(id),
      commitOffset: setOffset,
      onPickShot: setActiveShot,
      onScrubGlobalFrame: (gf) => {
        const refFps = getFps(state.referenceShot);
        const tRef = Math.max(0, gf / refFps);
        try { refColumn.video.currentTime = tRef; } catch {}
      },
      getCursorFrame: () => refColumn.video.currentTime * getFps(state.referenceShot),
    });
    timelineHost.appendChild(tl.element);
    drawCursor = tl.drawCursor;
    cleanupTimeline = tl.cleanup;
  }

  lockBtn.addEventListener("click", () => {
    const refFps = getFps(state.referenceShot);
    const actFps = getFps(state.activeShot);
    const refFrame = Math.round(refColumn.video.currentTime * refFps);
    const actFrame = Math.round(activeColumn.video.currentTime * actFps);
    setOffset(state.activeShot, actFrame - refFrame);
  });

  nudgeBack.addEventListener("click", () => {
    setOffset(state.activeShot, getOffset(state.activeShot) - 1);
  });
  nudgeFwd.addEventListener("click", () => {
    setOffset(state.activeShot, getOffset(state.activeShot) + 1);
  });

  offsetInput.addEventListener("change", () => {
    const v = Number(offsetInput.value);
    if (!Number.isFinite(v)) return;
    setOffset(state.activeShot, v);
  });

  function stopPlay() {
    state.playing = false;
    playBtn.textContent = "▶ Play both";
    refColumn.video.pause();
    activeColumn.video.pause();
    if (state.rafId) { cancelAnimationFrame(state.rafId); state.rafId = null; }
  }
  function startPlay() {
    syncActiveToRef();
    state.playing = true;
    playBtn.textContent = "⏸ Pause";
    refColumn.video.play().catch(() => {});
    activeColumn.video.play().catch(() => {});
    const tick = () => {
      if (!state.playing) return;
      drawCursor();
      // Soft re-sync: nudge the active video back when drift exceeds
      // 80 ms (setting currentTime mid-playback hiccups visibly).
      const refFps = getFps(state.referenceShot);
      const actFps = getFps(state.activeShot);
      const off = getOffset(state.activeShot);
      const target = (refColumn.video.currentTime * refFps + off) / actFps;
      if (Math.abs(activeColumn.video.currentTime - target) > 0.08) {
        try { activeColumn.video.currentTime = Math.max(0, target); } catch {}
      }
      if (refColumn.video.ended) { stopPlay(); return; }
      state.rafId = requestAnimationFrame(tick);
    };
    state.rafId = requestAnimationFrame(tick);
  }
  playBtn.addEventListener("click", () => {
    if (state.playing) stopPlay(); else startPlay();
  });

  saveBtn.addEventListener("click", async () => {
    const alignments = [];
    for (const sid of shotIds) {
      const m = state.methods.get(sid) || { method: "manual", confidence: 1.0 };
      alignments.push({
        shot_id: sid,
        frame_offset: sid === state.referenceShot ? 0 : getOffset(sid),
        method: sid === state.referenceShot ? "manual" : m.method,
        confidence: sid === state.referenceShot ? 1.0 : m.confidence,
      });
    }
    saveBtn.disabled = true;
    status.style.color = "#94a3b8";
    status.textContent = "Saving…";
    try {
      const res = await fetch("/api/sync", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          group_id: group.id,
          reference_shot: state.referenceShot,
          alignments,
        }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        status.style.color = "#f87171";
        status.textContent = `Save failed: ${body.detail || res.statusText}`;
        return;
      }
      status.style.color = "#4ade80";
      status.textContent = `Saved ${group.label} (${body.count} shots)`;
      setTimeout(() => {
        if (status.textContent.startsWith("Saved")) status.textContent = "";
      }, 2000);
    } catch (err) {
      status.style.color = "#f87171";
      status.textContent = `Save failed: ${err.message || err}`;
    } finally {
      saveBtn.disabled = false;
    }
  });

  refreshSelectExclusion();
  refreshOffsetInput();
  rebuildTimeline();

  _activeSyncEditorCleanup = () => {
    stopPlay();
    cleanupTimeline();
  };
  return root;
}

function _label(text) {
  const s = document.createElement("span");
  s.textContent = text;
  s.style.color = "#94a3b8";
  return s;
}

function _buildSyncVideoColumn(roleLabel, shotIds, initialShotId, fpsMap) {
  const wrap = document.createElement("div");
  wrap.style.cssText = "background:#0f1117;border:1px solid #1f2937;border-radius:6px;padding:10px;display:flex;flex-direction:column;gap:8px;min-width:0;";

  const headerEl = document.createElement("div");
  headerEl.style.cssText = "display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:12px;color:#cbd5e1;";
  const label = document.createElement("span");
  label.textContent = roleLabel;
  label.style.fontWeight = "600";
  label.style.color = "#f1f5f9";
  headerEl.appendChild(label);

  const select = document.createElement("select");
  select.style.cssText = "background:#252840;color:#e2e8f0;border:1px solid #2d3148;border-radius:4px;padding:3px 6px;font-size:12px;margin-left:auto;";
  for (const id of shotIds) {
    const opt = document.createElement("option");
    opt.value = id; opt.textContent = id;
    select.appendChild(opt);
  }
  select.value = initialShotId;
  headerEl.appendChild(select);
  wrap.appendChild(headerEl);

  const video = document.createElement("video");
  video.controls = true;
  video.preload = "metadata";
  // Fills the column at 16:9 (the broadcast clip aspect). max-height
  // keeps the controls + timeline visible on shorter windows.
  video.style.cssText = "width:100%;height:auto;aspect-ratio:16/9;max-height:60vh;border-radius:4px;background:#000;display:block;object-fit:contain;";
  wrap.appendChild(video);

  const meta = document.createElement("div");
  meta.style.cssText = "font-size:11px;color:#94a3b8;font-variant-numeric:tabular-nums;";
  wrap.appendChild(meta);
  let currentShotId = initialShotId;
  function updateMeta() {
    const fps = (fpsMap && fpsMap.get(currentShotId)) || 25;
    const f = Math.round(video.currentTime * fps);
    meta.textContent = `Frame ${f}  ·  ${video.currentTime.toFixed(2)}s`;
  }
  video.addEventListener("timeupdate", updateMeta);
  video.addEventListener("loadedmetadata", updateMeta);

  function loadShot(shotId) {
    currentShotId = shotId;
    video.src = `/api/video/${encodeURIComponent(shotId)}`;
  }
  loadShot(initialShotId);

  return { wrap, select, video, loadShot };
}

// Video-editor-style timeline. Each shot renders as a coloured block at
// its real-time position; the reference is pinned, others drag
// horizontally to edit their offset. Click a block to make it active;
// a red cursor mirrors the reference video's current frame. Blocks
// carry the alignment method/confidence chip text.
function _buildSyncTimeline(opts) {
  const {
    shotIds, framesByShot, fpsMap,
    referenceShot, activeShot,
    getOffset, getMethod, commitOffset,
    onPickShot, onScrubGlobalFrame, getCursorFrame,
  } = opts;

  const wrap = document.createElement("div");
  wrap.style.cssText = "background:#0a0e1a;border:1px solid #1f2937;border-radius:6px;padding:10px;";

  const meta = document.createElement("div");
  meta.style.cssText = "font-size:11px;color:#94a3b8;margin-bottom:6px;";
  meta.textContent = "Each block sits at its real-time position on the group timeline (reference anchored at frame 0). Drag a clip to slide it earlier/later. Click a clip to make it active. Click + drag empty timeline to scrub the reference.";
  wrap.appendChild(meta);

  // Each clip occupies global range [startGlobal, startGlobal + len]
  // with startGlobal = -offset: shot X's local frame f corresponds to
  // reference frame f - offset (sync_map.py), so a clip "ahead" of the
  // reference started earlier in wall-clock time and sits to the LEFT.
  function startGlobalFor(id) {
    return id === referenceShot ? 0 : -getOffset(id);
  }
  let minStart = Infinity, maxEnd = -Infinity;
  for (const id of shotIds) {
    const start = startGlobalFor(id);
    minStart = Math.min(minStart, start);
    maxEnd = Math.max(maxEnd, start + (framesByShot.get(id) || 0));
  }
  if (!isFinite(minStart)) { minStart = 0; maxEnd = 0; }
  const PAD_FRAMES = Math.max(60, Math.round((maxEnd - minStart) * 0.05));
  const spanStart = minStart - PAD_FRAMES;
  const spanEnd = maxEnd + PAD_FRAMES;
  const spanFrames = Math.max(1, spanEnd - spanStart);

  // Auto-fit ~900 px, clamped so tiny timelines stay legible and huge
  // ones scroll.
  const TARGET_WIDTH = 900;
  let pxPerFrame = TARGET_WIDTH / spanFrames;
  pxPerFrame = Math.max(0.4, Math.min(8, pxPerFrame));
  const contentWidth = Math.round(spanFrames * pxPerFrame);
  const ROW_HEIGHT = 36;

  const scroller = document.createElement("div");
  scroller.style.cssText = "overflow-x:auto;overflow-y:hidden;border-radius:4px;background:#080a14;";
  wrap.appendChild(scroller);

  const content = document.createElement("div");
  content.style.cssText = `position:relative;width:${contentWidth}px;height:${ROW_HEIGHT * (shotIds.length + 1) + 8}px;`;
  scroller.appendChild(content);

  // Ruler with frame ticks (every ~120px).
  const ruler = document.createElement("div");
  ruler.style.cssText = `position:relative;width:100%;height:18px;border-bottom:1px solid #1f2937;color:#475569;font-size:10px;font-variant-numeric:tabular-nums;`;
  const tickStride = Math.max(
    1, Math.round(120 / pxPerFrame / 10) * 10,
  );
  for (let f = Math.ceil(spanStart / tickStride) * tickStride; f <= spanEnd; f += tickStride) {
    const x = Math.round((f - spanStart) * pxPerFrame);
    const tick = document.createElement("div");
    tick.style.cssText = `position:absolute;left:${x}px;top:0;bottom:0;border-left:1px solid #1e293b;padding-left:3px;`;
    tick.textContent = String(f);
    ruler.appendChild(tick);
  }
  content.appendChild(ruler);

  // Single window-level mousemove/mouseup pair so listeners don't
  // accumulate as the timeline rebuilds.
  let dragCtx = null;

  // Reference first, then others sorted by real-time start.
  const orderedIds = [
    referenceShot,
    ...shotIds.filter((id) => id !== referenceShot)
                 .sort((a, b) => startGlobalFor(a) - startGlobalFor(b)),
  ];

  function methodNote(sid) {
    if (!getMethod || sid === referenceShot) return "";
    const m = getMethod(sid);
    if (!m) return "";
    return m.method === "manual"
      ? "  ·  manual"
      : `  ·  auto ${m.confidence.toFixed(2)}`;
  }

  function blockLabel(sid, startGlobal, len) {
    if (sid === referenceShot) {
      return `${sid} (ref · frames 0–${len})`;
    }
    return `${sid} · global ${startGlobal}–${startGlobal + len}${methodNote(sid)}`;
  }

  for (let i = 0; i < orderedIds.length; i++) {
    const sid = orderedIds[i];
    const isRef = sid === referenceShot;
    const isActive = sid === activeShot;
    const off = isRef ? 0 : getOffset(sid);
    const startGlobal = startGlobalFor(sid);
    const len = framesByShot.get(sid) || 1;
    const left = Math.round((startGlobal - spanStart) * pxPerFrame);
    const width = Math.max(2, Math.round(len * pxPerFrame));
    const top = 18 + 4 + i * ROW_HEIGHT;

    const block = document.createElement("div");
    const m = getMethod && !isRef ? getMethod(sid) : null;
    const lowConf = m && m.method !== "manual" && m.confidence < 0.5;
    const baseColor = isRef ? "#6366f1" : (isActive ? "#22c55e" : "#475569");
    block.style.cssText = `
      position:absolute;
      left:${left}px;top:${top}px;width:${width}px;height:${ROW_HEIGHT - 6}px;
      background:${baseColor};
      border:1px solid ${lowConf ? "#f59e0b" : (isActive ? "#86efac" : "#0f172a")};
      ${lowConf ? "box-shadow:0 0 0 1px #f59e0b inset;" : ""}
      border-radius:4px;
      color:#fff;font-size:11px;font-weight:600;
      padding:4px 6px;
      cursor:${isRef ? "pointer" : "grab"};
      user-select:none;
      box-sizing:border-box;
      overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
    `;
    const fps = fpsMap.get(sid) || 25;
    const methodTip = m
      ? `\nalignment: ${m.method} (confidence ${m.confidence.toFixed(2)})`
      : "";
    block.title = `${sid}\nframes: ${len}\nfps: ${fps.toFixed(2)}\noffset: ${off}\nglobal range: ${startGlobal}–${startGlobal + len}${methodTip}`;
    block.textContent = blockLabel(sid, startGlobal, len);

    block.addEventListener("click", (ev) => {
      ev.stopPropagation();
      if (!isRef && !block.dataset.dragMoved) onPickShot(sid);
      delete block.dataset.dragMoved;
    });

    // Drag horizontally → live-preview; commit on mouseup. Dragging
    // RIGHT moves the clip later in real time ⇒ offset DECREASES.
    if (!isRef) {
      block.addEventListener("mousedown", (ev) => {
        if (ev.button !== 0) return;
        ev.preventDefault();
        ev.stopPropagation();
        dragCtx = {
          kind: "block",
          shotId: sid,
          startX: ev.clientX,
          startOffset: off,
          startLeft: left,
          block,
          len,
          fps,
        };
        block.style.cursor = "grabbing";
      });
    }

    content.appendChild(block);
  }

  function onMove(ev) {
    if (!dragCtx) return;
    if (dragCtx.kind === "block") {
      const dxPx = ev.clientX - dragCtx.startX;
      const dxFrames = Math.round(dxPx / pxPerFrame);
      const newOffset = dragCtx.startOffset - dxFrames;
      const newStartGlobal = -newOffset;
      const newLeft = Math.round((newStartGlobal - spanStart) * pxPerFrame);
      dragCtx.block.style.left = `${newLeft}px`;
      dragCtx.block.textContent =
        blockLabel(dragCtx.shotId, newStartGlobal, dragCtx.len);
      dragCtx.lastOffset = newOffset;
      if (Math.abs(dxFrames) > 0) dragCtx.block.dataset.dragMoved = "1";
    } else if (dragCtx.kind === "scrub") {
      const rect = content.getBoundingClientRect();
      const x = ev.clientX - rect.left + scroller.scrollLeft;
      const globalFrame = spanStart + x / pxPerFrame;
      onScrubGlobalFrame(globalFrame);
    }
  }
  function onUp() {
    if (!dragCtx) return;
    const ctx = dragCtx;
    dragCtx = null;
    if (ctx.kind === "block") {
      ctx.block.style.cursor = "grab";
      if (ctx.lastOffset != null && ctx.lastOffset !== ctx.startOffset) {
        commitOffset(ctx.shotId, ctx.lastOffset);
      }
    }
  }
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);

  // Cursor — vertical line at the reference's current frame.
  const cursor = document.createElement("div");
  cursor.style.cssText = "position:absolute;top:18px;bottom:0;width:2px;background:#ef4444;pointer-events:none;";
  content.appendChild(cursor);
  const cursorHandle = document.createElement("div");
  cursorHandle.style.cssText = "position:absolute;top:14px;width:12px;height:12px;background:#ef4444;border:2px solid #fff;border-radius:50%;transform:translateX(-5px);cursor:ew-resize;z-index:5;";
  cursorHandle.title = "Drag to scrub the reference video";
  content.appendChild(cursorHandle);

  function drawCursor() {
    const f = getCursorFrame();
    const x = Math.round((f - spanStart) * pxPerFrame);
    cursor.style.left = `${x}px`;
    cursorHandle.style.left = `${x}px`;
  }
  drawCursor();

  cursorHandle.addEventListener("mousedown", (ev) => {
    if (ev.button !== 0) return;
    ev.preventDefault();
    ev.stopPropagation();
    dragCtx = { kind: "scrub" };
    onMove(ev);
  });

  function startScrub(ev) {
    if (ev.button !== 0) return;
    if (ev.target !== content && ev.target !== ruler) return;
    ev.preventDefault();
    dragCtx = { kind: "scrub" };
    onMove(ev);
  }
  content.addEventListener("mousedown", startScrub);
  ruler.addEventListener("mousedown", startScrub);
  content.style.cursor = "ew-resize";
  ruler.style.cursor = "ew-resize";

  function cleanup() {
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
  }

  return { element: wrap, drawCursor, cleanup };
}

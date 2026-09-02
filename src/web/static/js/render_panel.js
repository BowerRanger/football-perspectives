// render_panel.js — the dashboard's Render panel (pipeline stage #8).
//
// Loaded as a plain <script> before index.html's inline code and relies
// on these globals from there: makePanel, makeTable, addRow, emptyNote,
// fetchJsonOrNull, attachToJob, loadStages.
//
// Server contract:
//   GET   /api/render/outputs                per-shot rendered camera inventory
//   GET   /api/render/selection?shot=         operator camera selection (empty default)
//   PUT   /api/render/selection?shot=         save camera selection
//   GET   /api/render/video/{shot}/{camera}   mp4 stream (range-aware)
//   GET   /api/export/available-players?shot= POV/OTS player picker source
//   GET   /api/output/shots                   known shot ids (before the first render)

async function renderRender(panel) {
  const [outputsResp, allShotsResp] = await Promise.all([
    fetchJsonOrNull("/api/render/outputs"),
    fetchJsonOrNull("/api/output/shots"),
  ]);
  const shots = (outputsResp && outputsResp.shots) || {};
  const allShots = (allShotsResp && allShotsResp.shots) || [];
  // Union — shots with cameras already rendered, plus shots that exist
  // but haven't been rendered yet — so the selection editor is usable
  // before the first Render run.
  const shotIds = Array.from(new Set([...Object.keys(shots), ...allShots])).sort();

  if (shotIds.length === 0) {
    emptyNote(panel, "No shots yet — run prepare_shots first.");
    return;
  }

  const toolbar = document.createElement("div");
  toolbar.style.cssText = "padding:14px 16px 6px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;";

  const renderBtn = document.createElement("button");
  renderBtn.className = "btn btn-primary";
  renderBtn.style.fontSize = "12px";
  renderBtn.textContent = "Render";
  toolbar.appendChild(renderBtn);

  const label = document.createElement("span");
  label.textContent = "Shot:";
  label.style.cssText = "color:#cbd5e1;font-size:13px;";
  toolbar.appendChild(label);

  const select = document.createElement("select");
  select.style.cssText = "background:#252840;color:#e2e8f0;border:1px solid #2d3148;border-radius:4px;padding:4px 8px;font-size:13px;";
  for (const id of shotIds) {
    const opt = document.createElement("option");
    opt.value = id; opt.textContent = id;
    select.appendChild(opt);
  }
  toolbar.appendChild(select);

  const status = document.createElement("span");
  status.style.cssText = "font-size:12px;color:#64748b;margin-left:4px;";
  toolbar.appendChild(status);

  panel.appendChild(toolbar);

  const editorWrap = document.createElement("div");
  panel.appendChild(editorWrap);

  let latestShots = shots;

  async function loadShot(shotId) {
    editorWrap.innerHTML = "";
    _renderCameraGrid(editorWrap, shotId, latestShots[shotId]);
    await _renderSelectionEditor(editorWrap, shotId);
  }

  async function refresh() {
    const fresh = await fetchJsonOrNull("/api/render/outputs");
    latestShots = (fresh && fresh.shots) || {};
    await loadShot(select.value);
  }

  select.addEventListener("change", () => loadShot(select.value));

  renderBtn.addEventListener("click", async () => {
    renderBtn.disabled = true;
    status.style.color = "#fbbf24";
    status.textContent = "Dispatching render job…";
    let payload;
    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stages: "render" }),
      });
      payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        status.style.color = "#f87171";
        status.textContent = `Dispatch failed: ${payload.detail || res.statusText}`;
        renderBtn.disabled = false;
        return;
      }
    } catch (err) {
      status.style.color = "#f87171";
      status.textContent = `Failed to dispatch: ${err.message || err}`;
      renderBtn.disabled = false;
      return;
    }
    status.style.color = "#4ade80";
    status.textContent = `Job ${payload.job_id} running — log panel above.`;
    attachToJob(payload.job_id, "render", async () => {
      renderBtn.disabled = false;
      await loadStages();
      await refresh();
    });
  });

  await loadShot(select.value);
}

// ── Camera-card grid ─────────────────────────────────────────────────
function _renderCameraGrid(container, shotId, shotData) {
  const { wrap, body } = makePanel(`Rendered cameras — ${shotId}`);
  const cameras = (shotData && shotData.cameras) || [];
  if (!cameras.length) {
    body.style.padding = "14px 16px";
    emptyNote(body, "No rendered cameras for this shot yet — configure a selection below, then click Render.");
    container.appendChild(wrap);
    return;
  }
  body.style.padding = "0";
  const grid = document.createElement("div");
  grid.className = "render-grid";

  const landscape = cameras.filter((c) => !c.vertical);
  const verticalById = new Map(cameras.filter((c) => c.vertical).map((c) => [c.id, c]));
  const landscapeIds = new Set(landscape.map((c) => c.id));
  for (const cam of landscape) {
    grid.appendChild(_renderCameraCard(shotId, cam, shotData));
    const v = verticalById.get(cam.id);
    if (v) grid.appendChild(_renderCameraCard(shotId, v, shotData));
  }
  // Orphan verticals (no matching landscape entry) still get a card.
  for (const cam of cameras.filter((c) => c.vertical && !landscapeIds.has(c.id))) {
    grid.appendChild(_renderCameraCard(shotId, cam, shotData));
  }

  body.appendChild(grid);
  container.appendChild(wrap);
}

function _renderCameraCard(shotId, cam, shotData) {
  const card = document.createElement("div");
  card.className = "render-card" + (cam.vertical ? " vertical" : "");

  const video = document.createElement("video");
  video.controls = true;
  video.preload = "metadata";
  const stem = cam.file.replace(/\.mp4$/, "");
  video.src = `/api/render/video/${encodeURIComponent(shotId)}/${encodeURIComponent(stem)}`;
  card.appendChild(video);

  const body = document.createElement("div");
  body.className = "render-card-body";

  const title = document.createElement("div");
  title.className = "render-card-title";
  title.textContent = cam.id + (cam.vertical ? " (9:16)" : "");
  body.appendChild(title);

  const badges = document.createElement("div");
  const mb = (cam.size_bytes / (1024 * 1024)).toFixed(1);
  let html = `<span class="chip">${mb} MB</span>`;
  if (shotData && shotData.render_seconds != null) {
    html += `<span class="chip">${Number(shotData.render_seconds).toFixed(1)}s</span>`;
  }
  if (shotData && shotData.aov) {
    html += `<span class="chip chip-aov">AOV</span>`;
  }
  badges.innerHTML = html;
  body.appendChild(badges);

  card.appendChild(body);
  return card;
}

// ── Selection editor ─────────────────────────────────────────────────
async function _renderSelectionEditor(container, shotId) {
  const { wrap, body } = makePanel("Camera selection");
  body.style.padding = "14px 16px";
  container.appendChild(wrap);

  if (!shotId) {
    emptyNote(body, "No shot selected.");
    return;
  }

  const [sel, avail] = await Promise.all([
    fetchJsonOrNull(`/api/render/selection?shot=${encodeURIComponent(shotId)}`),
    fetchJsonOrNull(`/api/export/available-players?shot=${encodeURIComponent(shotId)}`),
  ]);
  const players = (avail && avail.players) || [];
  const chosenCameras = new Set((sel && sel.cameras) || []);
  const initialVertical = sel ? sel.vertical_variant : null;

  const status = document.createElement("p");
  status.style.margin = "10px 0 0";
  status.className = "cell-dim";

  let verticalToggle; // assigned below, referenced by save()

  async function save() {
    const chosen = [];
    body.querySelectorAll("input[data-cam]").forEach((cb) => {
      if (cb.checked) chosen.push(cb.dataset.cam);
    });
    const verticalVariant = verticalToggle.indeterminate ? null : verticalToggle.checked;
    status.className = "cell-dim";
    status.textContent = "Saving…";
    try {
      const res = await fetch(`/api/render/selection?shot=${encodeURIComponent(shotId)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ shot_id: shotId, cameras: chosen, vertical_variant: verticalVariant }),
      });
      if (!res.ok) {
        let detail;
        try { detail = (await res.json()).detail; } catch (e) { detail = await res.text(); }
        status.className = "cell-red";
        status.textContent = "Save failed: " + detail;
        return;
      }
      status.className = chosen.length ? "cell-green" : "cell-dim";
      status.textContent = chosen.length
        ? `Saved ${chosen.length} camera(s) for "${shotId}" → click Render to generate them.`
        : `No cameras selected for "${shotId}".`;
    } catch (e) {
      status.className = "cell-red";
      status.textContent = "Save failed: " + e;
    }
  }

  // broadcast / drone
  const fixedRow = document.createElement("div");
  fixedRow.style.cssText = "display:flex;gap:16px;align-items:center;margin-bottom:10px;";
  for (const camId of ["broadcast", "drone"]) {
    const cbLabel = document.createElement("label");
    cbLabel.style.cssText = "display:flex;align-items:center;gap:6px;font-size:13px;color:#cbd5e1;cursor:pointer;";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.dataset.cam = camId;
    cb.checked = chosenCameras.has(camId);
    cb.addEventListener("change", save);
    cbLabel.appendChild(cb);
    cbLabel.append(camId);
    fixedRow.appendChild(cbLabel);
  }
  body.appendChild(fixedRow);

  // POV / OTS per player
  const playerWrap = document.createElement("div");
  if (players.length) {
    const { t, tbody } = makeTable(["Player", "POV", "OTS"]);
    for (const p of players) {
      const mk = (rig) => {
        const camId = `${rig}:${p.player_id}`;
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.dataset.cam = camId;
        cb.checked = chosenCameras.has(camId);
        cb.addEventListener("change", save);
        const td = document.createElement("td");
        td.appendChild(cb);
        return td;
      };
      const nameTd = document.createElement("td");
      nameTd.textContent = p.display_name || p.player_id;
      const tr = document.createElement("tr");
      tr.appendChild(nameTd);
      tr.appendChild(mk("pov"));
      tr.appendChild(mk("ots"));
      tbody.appendChild(tr);
    }
    playerWrap.appendChild(t);
  } else {
    emptyNote(playerWrap, "No players with SMPL data for this shot yet — run hmr_world first for POV/OTS cameras.");
  }
  body.appendChild(playerWrap);

  // Vertical-variant toggle — tri-state: indeterminate means "unset,
  // follow config.render.vertical_variant".
  const vvRow = document.createElement("div");
  vvRow.style.cssText = "margin-top:12px;";
  const vvLabel = document.createElement("label");
  vvLabel.style.cssText = "font-size:13px;color:#cbd5e1;cursor:pointer;display:flex;align-items:center;gap:8px;";
  verticalToggle = document.createElement("input");
  verticalToggle.type = "checkbox";
  verticalToggle.checked = initialVertical === true;
  verticalToggle.indeterminate = initialVertical === null || initialVertical === undefined;
  verticalToggle.addEventListener("change", () => {
    verticalToggle.indeterminate = false;
    save();
  });
  vvLabel.appendChild(verticalToggle);
  vvLabel.append("Force 9:16 vertical variant (unset = use config default)");
  vvRow.appendChild(vvLabel);
  body.appendChild(vvRow);

  body.appendChild(status);

  const initialCount = chosenCameras.size;
  if (initialCount) {
    status.className = "cell-green";
    status.textContent = `Saved ${initialCount} camera(s) for "${shotId}" → click Render to generate them.`;
  }
}

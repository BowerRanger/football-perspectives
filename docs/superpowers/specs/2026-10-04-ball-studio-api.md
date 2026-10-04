# Ball Studio — HTTP API contract

Date: 2026-10-04 · Branch: `worktree-gberch-shorts` · Companion to
`2026-10-04-ball-studio-design.md` (read that first for the why).
Implementation: `src/web/ball_studio.py` (router), `src/utils/ball_truth_solver.py`
(pure solver), `src/schemas/ball_truth.py` (truth validator). The frontend
codes against THIS document; any deviation in the implementation is a bug in
the implementation, report it.

All routes live under `/api/ball-studio`. JSON in/out. Errors use FastAPI's
`{"detail": ...}`: 400 bad id / bad request, 404 unknown group, 422 truth
failed validation (`detail` is `{"errors": [{"path": "keys[2].frame", "message": "..."}]}`),
409 `PUT truth` optimistic-concurrency conflict (see PUT). The SPA page route is `GET /ball-studio?group=<group_id>`
(serves the SPA shell).

## Conventions

* **Reference timeline.** Every group has a `reference_shot`. Reference frame
  `r` is an integer. A member shot's local frame is
  `shot_frame = r + frame_offset` (equivalently `r = shot_frame - frame_offset`;
  sign matches `/api/sync` and `/refined_poses/preview?shot=`). All keys,
  events, dense-track frames and player frames are on the reference timeline;
  only 2-D observations carry a `(shot_id, shot_frame)`.
* **World frame.** Pitch metres, `x` along the near touchline (0..105), `y`
  across (0..68), `z` up. Ball centre rests at `z = 0.11` (ball radius).
* **Pixels.** Full-resolution video pixels (`image_size`), origin top-left,
  `uv = [u, v]`. Camera lens distortion (k1, k2 radial) is applied by the
  server whenever it projects or back-projects; the client projects with the
  same model when it draws (formula in "Camera model" below).
* **Group id.** The manifest `group_id` when non-empty (e.g. `g04`), otherwise
  the sync reference shot id (e.g. `origi01`, `gberch`, `kroupi01`).
  Pattern `[A-Za-z0-9_-]+`.
* Floats are rounded to 4 decimals (positions/uv) in responses.

## Camera model (for client-side drawing)

Per shot frame: `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]`, world->camera rotation
`R` (3x3), translation `t` (3). `Xc = R·X + t`; `x = Xc.x/Xc.z`,
`y = Xc.y/Xc.z`; radial `s = 1 + k1·r² + k2·r⁴` with `r² = x²+y²`;
`u = fx·x·s + cx`, `v = fy·y·s + cy`. Camera centre `C = -Rᵀ·t`. Ray through
pixel: undistort (invert the radial model), `d = Rᵀ·K⁻¹·[u,v,1]ᵀ`, normalise.
The server also returns pre-projected tracks so a client may avoid this
entirely for the common overlay.

## GET `/api/ball-studio/groups`

```json
{
  "groups": [
    {
      "group_id": "origi01",
      "label": "origi01 + origi02",
      "reference_shot": "origi01",
      "fps": 30.0,
      "ref_frame_range": [0, 205],
      "shots": [
        {"shot_id": "origi01", "frame_offset": 0, "n_frames": 206,
         "fps": 30.0, "image_size": [1920, 1080], "width": 1920, "height": 1080,
         "frame_range": [0, 205], "excluded": false,
         "video_url": "/api/video/origi01",
         "frame_url": "/api/video/origi01/frame"},
        {"shot_id": "origi02", "frame_offset": -142, "n_frames": 340, "...": "..."}
      ],
      "has_truth": false,
      "truth_updated_at": null,
      "n_keys": 0,
      "outcome": "unknown",
      "status": "draft"
    }
  ]
}
```

Members are the sync-map members (or manifest group members) that have a solved
camera track; excluded shots are kept when they have a camera track (e.g.
spidercam `gberch-2`) and flagged `excluded: true`. `ref_frame_range` is the
union of all members' camera frame ranges mapped onto the reference timeline.
Groups without any usable member are omitted. Single-camera shots
(`kroupi01`) are groups of one.

## GET `/api/ball-studio/groups/{g}/scene`

One call that loads everything the page draws. Large but compact (columnar
arrays). Cached server-side on file mtimes; send `Cache-Control: no-store`.

```json
{
  "group_id": "origi01", "reference_shot": "origi01", "fps": 30.0,
  "ref_frame_range": [0, 205],
  "shots": [
    {
      "shot_id": "origi01", "frame_offset": 0, "image_size": [1920, 1080],
      "width": 1920, "height": 1080,
      "frame_range": [0, 205], "n_frames": 206, "fps": 30.0, "excluded": false,
      "distortion": [0.0, 0.0],
      "camera_centre": [52.5, -38.0, 21.0],
      "frames":   [0, 1, 2],
      "K":        [[fx, fy, cx, cy], [fx, fy, cx, cy], [fx, fy, cx, cy]],
      "R":        [[9 floats row-major], "..."],
      "t":        [[3 floats], "..."],
      "confidence": [0.9, 0.9, 0.88],
      "video_url": "/api/video/origi01", "frame_url": "/api/video/origi01/frame"
    }
  ],
  "goals": {
    "goal_line_x_near": 0.0, "goal_line_x_far": 105.0,
    "post_y_left": 30.34, "post_y_right": 37.66,
    "crossbar_z": 2.44, "net_depth": 1.5,
    "pitch": {"length_m": 105.0, "width_m": 68.0},
    "goal_planes": [
      {"id": "goal_line_near", "axis": "x", "value": 0.0,
       "mouth": {"y_range": [30.34, 37.66], "z_range": [0.0, 2.44]}},
      {"id": "goal_line_far", "axis": "x", "value": 105.0,
       "mouth": {"y_range": [30.34, 37.66], "z_range": [0.0, 2.44]}}
    ]
  },
  "bones": ["pelvis","l_foot","r_foot","l_knee","r_knee","chest","head",
            "l_shoulder","r_shoulder","l_hand","r_hand"],
  "players": [
    {"player_id": "P001",
     "frames": [0, 1, 2],
     "root": [[x,y,z], "..."],
     "joints": {"l_foot": [[x,y,z], "..."], "r_foot": ["..."], "head": ["..."]},
     "confidence": [0.9, 0.9, 0.9]}
  ],
  "pipeline_tracks": [
    {"shot_id": "origi01", "frames": [0, 1, 2],
     "shot_frames": [0, 1, 2],
     "xyz": [[x,y,z], null, [x,y,z]],
     "state": ["grounded", "missing", "flight"],
     "confidence": [0.7, 0.0, 0.8]}
  ],
  "pipeline_anchors": [
    {"shot_id": "origi01", "ref_frame": 440, "shot_frame": 440,
     "kind": "player_touch", "uv": [812.0, 604.5], "source": "manual"}
  ]
}
```

* `shots[].frames` are *shot-local* frame indices (index `i` of `K`/`R`/`t` is
  frame `frames[i]`); reference frame = `frame - frame_offset`.
* `players[].frames` are **reference** frames. `joints` holds the SMPL-FK
  world position of each bone in `bones` (the anchor-editor bone vocabulary +
  `pelvis`) for `player` constraints and the 3-D view. Players come from
  `refined_poses`; absent when that stage hasn't run (`players: []`).
* `pipeline_tracks[].frames` are **reference** frames, `shot_frames` the same
  rows in shot-local frames; `xyz` null where the pipeline has no position.
  Pipeline output is reference-only data — the tool never writes it.
* `pipeline_anchors`: the shot's manual + auto ball anchors (pixel + kind) for
  overlay; optional, may be `[]`.

## GET `/api/ball-studio/groups/{g}/truth`

Returns the stored truth document or an empty skeleton (`"exists": false`):

```json
{
  "exists": false,
  "truth": { "...see Truth document..." },
  "dense": null
}
```

When it exists, `dense` is the last `_dense.json` written on save (same shape
as `solve.dense`/`projections`, may be stale if camera tracks changed since;
call `solve` for a fresh one) or `null`.

## Truth document

Stored at `<output>/ball_truth/<group_id>_ball_truth.json`.

```json
{
  "version": 1,
  "group_id": "origi01",
  "reference_shot": "origi01",
  "fps": 30.0,
  "shots": [{"shot_id": "origi01", "frame_offset": 0},
            {"shot_id": "origi02", "frame_offset": -142}],
  "outcome": "goal",
  "keys": [
    {
      "id": "k7", "frame": 440, "xyz": [2.1, 31.0, 0.11],
      "source": "triangulated",
      "constraint": {"height_m": null, "plane": null, "depth_m": null,
                     "player_id": null, "bone": null, "offset": null},
      "observations": [
        {"shot_id": "origi01", "shot_frame": 440, "uv": [812.0, 604.5]},
        {"shot_id": "origi02", "shot_frame": 298, "uv": [1203.1, 455.0]}
      ],
      "residual_px": {"origi01": 0.8, "origi02": 1.4},
      "note": ""
    }
  ],
  "segments": [
    {"from": "k6", "to": "k7", "kind": "roll",
     "params": {"drag": true, "cd": null, "magnus": "auto",
                "player_id": null, "bone": null}}
  ],
  "observations": [
    {"shot_id": "origi01", "shot_frame": 444, "uv": [790.0, 600.0]}
  ],
  "events": [
    {"frame": 440, "kind": "touch", "player_id": "P023", "bone": "r_foot", "note": ""}
  ],
  "meta": {"authored_by": "operator", "updated_at": "2026-10-04T12:00:00Z", "notes": "",
           "status": "draft"}
}
```

`meta.status` is `draft | reviewed` (default `draft`); `meta.updated_at` is
server-owned (overwritten on every PUT, null before the first save).

Field rules (the server validates on `PUT`, returns 422 with all errors):

* `version` must be `1`. `group_id` must equal the URL group. `fps` > 0.
* `shots[].shot_id` must be group members; `frame_offset` int.
* `outcome` in `goal | no_goal | unknown`.
* `keys[].id` unique, non-empty, `[A-Za-z0-9_-]+`; `frame` int, unique per key
  (two keys on one frame are rejected); `xyz` 3 finite floats.
* `keys[].source` in `triangulated | ray_ground | ray_height | ray_plane |
  ray_depth | player | manual`.
  * `triangulated`: >= 2 observations from different shots, **same reference
    instant** as `frame` (`shot_frame - frame_offset == frame`).
  * `ray_ground`: 1 observation, ball on the ground (z = 0.11).
  * `ray_height`: 1 observation + `constraint.height_m`.
  * `ray_plane`: 1 observation + `constraint.plane = {"axis": "x"|"y"|"z", "value": m}`
    (goal line = `{"axis":"x","value":0}` or `105`).
  * `ray_depth`: 1 observation + `constraint.depth_m` (metres along the ray from the camera).
  * `player`: `constraint.player_id` + `constraint.bone` (a bone name). With one
    observation the pixel stays authoritative laterally and depth comes from
    the joint (ray-faithful); with none the key IS the joint position (+
    optional `constraint.offset` `[dx,dy,dz]`).
  * `manual`: `xyz` is used as given, observations optional.
* `observations[].uv` finite floats; `shot_frame` int. Soft observations
  (top-level `observations`) are NOT keys — they are fit targets for segments.
* `segments[].from/to` reference key ids with `from.frame < to.frame`; `kind` in
  `flight | roll | carried | linear | static`. Segments are optional: a missing
  segment between consecutive keys is auto-proposed by the solver (both keys
  on the ground -> `roll`, else `flight`) and reported in `solve.segments`
  with `"auto": true`. `params`: `drag` (bool, default true), `cd` (float|null =
  default 0.25), `magnus` (`"auto"` fit when >= 2 soft observations exist in
  the span | `"off"`), `player_id`/`bone` (for `carried`).
* `events[].kind` in `touch | bounce | post | crossbar | net | line_cross |
  out | keeper_save`; `frame` int; `player_id`/`bone` optional (touch/save).
* Unknown fields anywhere in the document are rejected (422) — keep the
  document to this schema.

## PUT `/api/ball-studio/groups/{g}/truth`

Body (wrapper, because of optimistic concurrency):

```json
{"truth": { "...Truth document..." }, "expected_updated_at": "2026-10-04T11:58:00Z"}
```

* `expected_updated_at` is the `meta.updated_at` the client loaded (`null` =
  "I believe no file exists"). If the key is **omitted** no concurrency check
  is made. When present and different from the stored `meta.updated_at` (or
  `null` while a file exists) the server answers **409** and writes nothing:
  `{"detail": {"message": "truth changed since it was loaded",
  "current_updated_at": "...", "expected_updated_at": "..."}}`.
* Validates the truth (422 with all errors), overwrites `meta.updated_at`,
  writes `ball_truth/<g>_ball_truth.json` atomically (tmp + replace), copies the
  previous file to `ball_truth/.history/<g>_ball_truth.<UTC timestamp>.json`
  (kept, never pruned), solves and writes `<g>_ball_truth_dense.json`.

```json
{"ok": true, "updated_at": "2026-10-04T12:00:00Z",
 "history_file": ".history/origi01_ball_truth.20261004T120000000000Z.json",
 "solve_ok": true, "n_flags": 2}
```

A document that fails validation is rejected (422), nothing written. A
document that validates but cannot be solved (e.g. a key is behind the camera)
is still saved (`solve_ok: false`) - operator data is never dropped.

## POST `/api/ball-studio/groups/{g}/triangulate`

Live while clicking. Body:

```json
{
  "frame": 440,
  "observations": [
    {"shot_id": "origi01", "shot_frame": 440, "uv": [812.0, 604.5]},
    {"shot_id": "origi02", "shot_frame": 298, "uv": [1203.1, 455.0]}
  ],
  "constraint": null,
  "offsets": null
}
```

`offsets` (optional, read-only **sync probe**): `{shot_id: frame_offset}`
overrides. For an overridden shot the held pixel is re-attributed to camera
frame `frame + override` (the response's `observations_used[].shot_frame`
shows the frame actually used) and the epipolar lines use the overridden
offsets too. Nothing is persisted. The UI calls this at stored offset -2..+2 and
compares `max_residual_px` / `skew_gap_cm`.

`constraint` (optional, for single-observation requests) is the same object
as `keys[].constraint` plus `"mode"`: one of `ground | height | plane | depth |
player`. With >= 2 observations the constraint is ignored. `frame` is the
reference frame (needed for the player lookup and to check that all
observations share the instant). Response (always 200 when well-formed; the
verdict is in the body):

```json
{
  "ok": true,
  "xyz": [2.1013, 31.0042, 0.1187],
  "source": "triangulated",
  "residual_px": {"origi01": 0.8, "origi02": 1.4},
  "max_residual_px": 1.4,
  "reprojected_uv": {"origi01": [812.4, 604.2], "origi02": [1203.8, 455.6]},
  "ray_angle_deg": 37.2,
  "skew_gap_cm": 3.1,
  "offsets_used": {"origi01": 0, "origi02": -142},
  "observations_used": [
    {"shot_id": "origi01", "shot_frame": 440, "uv": [812.0, 604.5]},
    {"shot_id": "origi02", "shot_frame": 298, "uv": [1203.1, 455.0]}
  ],
  "rays": [
    {"shot_id": "origi01", "origin": [52.5, -38.0, 21.0], "direction": [0.1, 0.8, -0.5]},
    {"shot_id": "origi02", "origin": [...], "direction": [...]}
  ],
  "epipolar": [
    {"shot_id": "origi02", "shot_frame": 298,
     "polyline_uv": [[0, 0], "... up to 24 points, clipped to the image"],
     "segment_uv": [[u0, v0], [u1, v1]]}
  ],
  "flags": [{"level": "warn", "code": "weak_baseline", "message": "rays differ by 1.2 deg"}]
}
```

* `skew_gap_cm`: the largest closest-approach distance between any two of the
  rays (how far the clicks are from being geometrically consistent, independent
  of pixel scale); also present on `solve.keys[]` for triangulated keys.
* `ok: false` + `reason` (`"behind_camera"`, `"no_camera_frame"`,
  `"residual_exceeds_limit"`, `"parallel_rays"`, ...) when it cannot produce a
  key. For `residual_exceeds_limit` (any view > 15 px) `xyz` and residuals are
  STILL returned so the UI can show the bad view; the UI must not silently save.
* With ONE observation and no constraint the response has `ok: true`,
  `source: "ray"`, `xyz: null`, the `rays` entry, and `epipolar` lines for the
  other group shots: the 3-D ray projected into each other view at the same
  reference instant (clipped to the image; `segment_uv` null if outside) —
  this is what the UI draws as the epipolar line in view B after the click in
  view A.
* Distance range of the epipolar segment: ray points 2 m .. 250 m from the camera.

## POST `/api/ball-studio/groups/{g}/solve`

Stateless; body = the Truth document (validated leniently: structural errors ->
422, solver-level problems become `flags`). Typically 10-200 ms.

```json
{
  "ok": true,
  "keys": [
    {"id": "k7", "frame": 440, "xyz": [2.1013, 31.0042, 0.1187],
     "source": "triangulated", "residual_px": {"origi01": 0.8, "origi02": 1.4},
     "ray_angle_deg": 37.2, "status": "ok", "messages": []}
  ],
  "segments": [
    {"index": 0, "from": "k6", "to": "k7", "kind": "roll", "auto": false,
     "frame_range": [431, 440],
     "params": {"cd": 0.25, "accel_xy": [-0.4, 0.1], "v0": [3.1, 0.2, 0.0], "omega": null},
     "rms_obs_px": 1.9, "n_soft_obs": 3, "max_speed_m_s": 4.1, "status": "ok"}
  ],
  "dense": {
    "frames": [431, 432, 433],
    "xyz": [[x,y,z], "..."],
    "segment": [0, 0, 0],
    "kind": ["roll", "roll", "roll"],
    "speed_m_s": [3.1, 3.1, 3.0]
  },
  "projections": {
    "origi01": {"frames": [431, 432], "shot_frames": [431, 432],
                "uv": [[812.0, 604.5], null], "depth_m": [61.2, null]},
    "origi02": {"frames": [431, 432], "shot_frames": [289, 290],
                "uv": [[1203.1, 455.0], [1210.0, 456.0]], "depth_m": [..]}
  },
  "observations": [
    {"kind": "key", "key_id": "k7", "shot_id": "origi01", "shot_frame": 440,
     "uv": [812.0, 604.5], "projected_uv": [812.3, 604.1], "residual_px": 0.5},
    {"kind": "soft", "index": 0, "shot_id": "origi01", "shot_frame": 444,
     "uv": [790.0, 600.0], "projected_uv": [791.0, 601.0], "residual_px": 1.4}
  ],
  "flags": [
    {"level": "warn", "code": "below_ground", "frame": 441,
     "ref": {"segment": 0}, "message": "z=-0.03 m"}
  ],
  "stats": {"n_keys": 8, "n_segments": 7, "n_dense": 90,
            "max_key_residual_px": 1.4, "max_speed_m_s": 31.2}
}
```

* `dense` covers every reference frame from the first to the last key
  (inclusive); outside that span there is no track. `projections` are keyed by
  shot id and cover each dense frame whose shot frame has a camera;
  `uv`/`depth_m` are `null` where the point is behind the camera.
* `keys[].xyz` is the server-resolved position (re-derived from the
  observations/constraint for non-`manual` keys), which can differ slightly
  from the stored `xyz` — the client should adopt the resolved value for
  display and the next save.
* `status` (key/segment) in `ok | warn | error`.
* Flag `code`s: `residual_exceeds_limit` (error, key view > 15 px),
  `weak_baseline` (warn, ray angle < 3 deg), `behind_camera` (error),
  `no_camera_frame` (error), `below_ground` (warn, z < 0.05 m),
  `speed_exceeds_limit` (warn, > 45 m/s), `discontinuity` (warn, implied
  speed > 45 m/s across consecutive dense frames), `curl_exceeds_limit` (warn,
  fitted Magnus acceleration > 15 m/s²), `segment_infeasible` (error,
  e.g. roll between keys not both near the ground, flight with
  non-positive duration), `observation_outlier` (warn, soft obs > 12 px from
  the solved track), `unknown_player_joint` (error, `player`/`carried` joint
  has no data at that frame), `unsorted_keys`/`duplicate_frame` (error).
  `ok` is `false` iff any `error` flag exists; the dense track is still
  returned best-effort.

## Eval (offline, no HTTP)

`scripts/eval_ball_truth.py --output <dir> --group <g> [--shot <shot>]` compares
`ball/<shot>_ball_track.json` (mapped to the reference timeline) with
`ball_truth/<g>_ball_truth_dense.json` and prints/writes JSON: per-frame 3-D
error p50/p95, % within 0.2 m / 0.5 m, per-view reprojection error, event timing
error, and the goal line-cross point error (outcome `goal` only).

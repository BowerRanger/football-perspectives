# Stadium scene and camera selection

The Blender renderer builds a stylized stadium with two seating tiers, aisles,
access openings, rails, a supported roof, spectators, advertising ribbons,
dugouts, floodlight masts and corner flags. Goal nets and rear supports, penalty
arcs, corner arcs and spots complete the pitch; procedural turf variation adds
surface detail. Stadium meshes are batched by material and spectator placement
uses a fixed seed, keeping rerenders consistent without external assets.

Configure `render.style.stadium` in `config/default.yaml`: `enabled`, `roof`,
`crowd_density` (0–1), `seat_color` and `accent_color`. Floodlights are visual
set dressing for the daytime scene; they do not provide night illumination.
Player geometry, tracking, and kit assignment still come from the reconstruction.

The dashboard Render panel now exposes all stadium and action cameras. Selection
sidecars accept the same IDs as the renderer:

| Camera | View |
| --- | --- |
| `broadcast` | Calibrated source view |
| `drone` | Elevated action-following view |
| `tactical` | Static overhead covering the full pitch with run-off margin |
| `sideline:near`, `sideline:far` | Elevated main and reverse touchline views |
| `corner:left`, `corner:right` | Elevated diagonals from the near corners |
| `goal:left`, `goal:right` | Low behind-goal views through the net |
| `goalline:left`, `goalline:right` | Low views inside the goal mouth |
| `orbit`, `chase`, `dolly` | Orbit around play, ball chase, touchline tracking |
| `pov:<PID>`, `ots:<PID>` | Player perspective and over the shoulder |

Tune camera FOVs/heights under `export.virtual_cameras`, alongside the existing
rig settings. Full-pitch tactical framing is computed for the requested image
aspect ratio. The existing portrait variant reframes the same camera; use a
portrait render resolution when full-pitch coverage in portrait is required.
Moving action rigs do not perform collision avoidance against the stadium.

Local review artifacts are in `output/stadium_review/`, separate from existing
renders. `gberch/scene.blend` contains the animated reconstruction and four new
cameras. The saved scene opens in EEVEE at the requested resolution and frame.
Existing MP4s are reused by the pipeline; to review the new scene without
replacing them, pass an alternate `--render-root` to the Blender script and
stage virtual-camera tracks under that root's `<shot>/cameras/` directory.

Open `output/stadium_review/index.html` for the local four-camera video gallery.
The gberch fixture currently labels all players' teams `unknown`, so its previews
use the configured neutral kit. Team colours require classified tracking labels
or explicit `render.teams.by_player` overrides.

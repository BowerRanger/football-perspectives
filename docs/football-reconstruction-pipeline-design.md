# Football Match Reconstruction Pipeline — Technical Design Document

## 1. Overview

A Python CLI tool that takes broadcast football footage (single file with multiple replay angles, or multiple files), automatically segments it into individual camera shots, synchronises overlapping views via visual event matching, runs 2D pose estimation per view, calibrates cameras against pitch geometry, triangulates 3D joint positions, fits SMPL body models, and exports animation assets compatible with Unreal Engine 5 and a browser-based 3D preview viewer.

### 1.1 Design Goals

- **Single entry point**: one command, feed in video(s), get 3D assets out
- **Modular stages**: each pipeline stage writes intermediate outputs to disk, so any stage can be re-run or inspected independently
- **UE5-native output**: FBX files with skeleton animation that import directly into Unreal Engine 5's retargeting system
- **Browser preview**: a lightweight Three.js viewer that loads pipeline output for quick validation without launching UE5
- **Honest about limitations**: the pipeline should surface confidence metrics at each stage so the user knows where manual cleanup is needed

### 1.2 Non-Goals (for v1)

- Real-time processing
- Crowd or coaching staff reconstruction (players and ball only)
- Facial expression capture
- Finger/hand pose estimation
- Automatic MetaHuman likeness generation

---

## 2. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT                                │
│  Single video file with replays  OR  multiple video files   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 1: Shot Segmentation                                  │
│  Split video into individual camera shots                    │
│  Output: shots/ directory with clip files + metadata JSON    │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 2: Camera Calibration                                 │
│  Detect pitch lines → solve camera intrinsics/extrinsics     │
│  Output: calibration JSON per shot                           │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 3: Player Detection & Tracking                        │
│  Detect + track players and ball per shot                    │
│  Output: tracks JSON per shot (bounding boxes + IDs)         │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 4: 2D Pose Estimation                                 │
│  Run ViTPose on each tracked player crop, per shot           │
│  Output: 2D keypoints JSON per player per shot               │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 5: Temporal Synchronisation                           │
│  Fuse ball and player-motion signals for frame alignment     │
│  Output: sync_map.json with frame offsets between views      │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 6: Cross-View Player Matching                         │
│  Match player identities across synchronised views           │
│  Output: player_matches.json mapping IDs across shots        │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 7: 3D Triangulation                                   │
│  Triangulate 3D joint positions from matched 2D keypoints    │
│  Output: 3D skeleton sequences per player (NumPy arrays)     │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 8: SMPL Fitting                                       │
│  Fit SMPL body model to triangulated 3D joints               │
│  Output: SMPL parameters (pose, shape, translation) per frame│
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 9: Export                                             │
│  Convert SMPL sequences → FBX (UE5) + glTF (browser viewer) │
│  Output: export/ directory with FBX and glTF files           │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Stage Details

### 3.1 Shot Segmentation

**Purpose**: Split a single video containing multiple replay angles into individual clips, each representing a continuous camera shot.

**Approach**: Use PySceneDetect with the ContentDetector algorithm, which identifies shot boundaries by measuring pixel-level changes between consecutive frames. Broadcast football footage has hard cuts between angles during replays, making content-based detection highly reliable.

**Implementation**:
```
Library: PySceneDetect (scenedetect)
Detector: ContentDetector(threshold=30.0)
```

For each detected shot, extract:
- Start/end frame numbers and timestamps
- A representative thumbnail (middle frame)
- Average brightness and dominant colour histogram (used later for camera matching)

**Output schema** (`shots/shots_manifest.json`):
```json
{
  "source_file": "input.mp4",
  "fps": 25.0,
  "total_frames": 5000,
  "shots": [
    {
      "id": "shot_001",
      "start_frame": 0,
      "end_frame": 312,
      "start_time": 0.0,
      "end_time": 12.48,
      "clip_file": "shots/shot_001.mp4",
      "thumbnail": "shots/shot_001_thumb.jpg"
    }
  ]
}
```

Each shot is also extracted as a standalone video file using FFmpeg for downstream processing.

**Edge cases**:
- Slow dissolves or wipes between angles: increase ContentDetector sensitivity or add a secondary AdaptiveDetector pass
- Picture-in-picture overlays: not handled in v1; user should crop input if present
- Broadcast graphics overlays (score bug, replay banner): these persist across cuts and shouldn't trigger false positives, but the calibration stage needs to mask them

---

### 3.2 Camera Calibration

**Purpose**: For each shot, determine the camera's position, orientation, and lens parameters relative to the football pitch.

**Approach**: Detect pitch line segments and known geometric features (centre circle, penalty arcs, goal area lines), match them to the FIFA standard pitch model (105m × 68m), and solve for camera parameters using PnP (Perspective-n-Point).

**Implementation**:

1. **Pitch line detection**: Use a semantic segmentation model trained on football pitch markings. The SoccerNet Camera Calibration challenge has produced several suitable models, including line detection networks that output pitch keypoints (line intersections, arc tangent points, etc.).

2. **2D-3D correspondence**: Map detected 2D pitch keypoints to their known 3D coordinates on the standard pitch model. The pitch is treated as the z=0 ground plane.

3. **Camera solve**: Use OpenCV's `solvePnP` with RANSAC to estimate the camera extrinsic matrix (rotation + translation) from the 2D-3D correspondences. For intrinsics, start with reasonable broadcast camera defaults (focal length ~2000-4000px equivalent, no distortion) and refine using bundle adjustment if multiple pitch features are visible.

4. **Tracking camera motion**: For shots where the camera pans/zooms (the main broadcast camera), solve calibration per-frame or at regular intervals (every 5-10 frames) and interpolate. For static replay cameras (e.g., behind-goal), a single calibration per shot is sufficient.

**Output schema** (`calibration/shot_001_calibration.json`):
```json
{
  "shot_id": "shot_001",
  "camera_type": "static|tracking",
  "frames": [
    {
      "frame": 0,
      "intrinsic_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "rotation_vector": [rx, ry, rz],
      "translation_vector": [tx, ty, tz],
      "reprojection_error": 2.3,
      "num_correspondences": 12,
      "confidence": 0.87
    }
  ]
}
```

**Confidence metric**: reprojection error (average pixel distance between projected 3D pitch points and detected 2D keypoints). Values under 5px are good; over 15px suggests unreliable calibration. Shots with poor calibration are flagged and can be excluded from triangulation.

**Known challenges**:
- Tight close-ups may show very few pitch markings → low confidence or calibration failure
- Zoomed-in shots distort the relationship between focal length and position → degenerate solutions possible
- Rain/snow obscuring pitch lines

---

### 3.5 Temporal Synchronisation (Execution Stage 5)

**Purpose**: Determine the precise frame offset between overlapping shots so that frame N in shot A corresponds to the same real-world moment as frame M in shot B.

**Approach**: Visual event matching using the ball trajectory and player positions. Since audio is often the live feed overlaid on replays (as you noted), we rely entirely on visual cues.

**Implementation — multi-signal approach**:

1. **Ball position matching**: The ball's trajectory through space is the strongest synchronisation signal. Detect the ball in each shot (using a specialised ball detector — YOLO-based, trained on football-specific data), project its 2D position to pitch coordinates using the calibrated camera, and cross-correlate ball trajectories across shots. The moment of foot-to-ball contact during a shot or pass creates a distinctive trajectory inflection that should be identifiable across all angles.

2. **Player formation matching**: As a secondary signal, compute the spatial arrangement of all detected players in pitch coordinates at each frame. Use a distance metric (e.g., Procrustes analysis or Hungarian matching on player positions) to find the frame alignment that minimises the total position discrepancy across players.

3. **Distinctive event detection**: Identify high-confidence anchor events — goal net ripple, goalkeeper dive apex, ball crossing the goal line. These can be detected with simple heuristics (e.g., ball velocity near zero after a shot on goal) and serve as coarse alignment anchors before fine-grained cross-correlation.

**Algorithm**:
```
For each pair of overlapping shots (A, B):
  1. Project ball positions to pitch coordinates in both shots
  2. Compute normalised cross-correlation of ball trajectories
  3. Compute player-motion signal from tracked pitch positions and correlate it
  4. If both offsets agree within tolerance, fuse them; otherwise pick higher confidence
  5. Compute confidence from aligned-signal correlation and overlap quality
```

**Output schema** (`sync/sync_map.json`):
```json
{
  "reference_shot": "shot_001",
  "alignments": [
    {
      "shot_id": "shot_003",
      "frame_offset": -47,
      "confidence": 0.92,
      "method": "ball_trajectory",
      "overlap_frames": [120, 280]
    }
  ]
}
```

**Frame offset convention**: `frame_offset = -47` means shot_003's frame 0 corresponds to shot_001's frame 47. So to find the matching frame: `frame_in_reference = frame_in_shot + offset`.

**Failure modes**:
- Ball occluded in one or more views during the key moment
- Very short shots (< 1 second) with insufficient trajectory data
- Static ball (e.g., before a free kick) provides no temporal signal

For failure cases, fall back to player formation matching, or flag for manual alignment (the viewer tool should support manual frame offset input).

---

### 3.3 Player Detection & Tracking (Execution Stage 3)

**Purpose**: Detect all players (and the ball) in each shot and maintain consistent identity tracking across frames within a shot.

**Approach**: YOLOv8 for detection, ByteTrack for multi-object tracking, SigLIP embeddings for team classification.

**Implementation**:

1. **Detection**: YOLOv8 model fine-tuned on football data (SoccerNet or similar). Detect classes: player, goalkeeper, referee, ball. Ball detection uses a separate specialised model due to its small size in frame.

2. **Tracking**: ByteTrack associates detections across frames using motion prediction (Kalman filter) and appearance similarity. Each tracked entity gets a persistent ID within the shot.

3. **Team classification**: Extract visual embeddings (SigLIP or CLIP) from each player crop, cluster using K-means (k=3: team A, team B, referee), assign team labels. Goalkeepers are assigned to the nearest team by pitch position.

**Output schema** (`tracks/shot_001_tracks.json`):
```json
{
  "shot_id": "shot_001",
  "tracks": [
    {
      "track_id": "T001",
      "class": "player",
      "team": "A",
      "frames": [
        {
          "frame": 0,
          "bbox": [x1, y1, x2, y2],
          "confidence": 0.94,
          "pitch_position": [34.2, 21.5]
        }
      ]
    }
  ]
}
```

`pitch_position` is computed using the camera calibration from Stage 2 — project the bottom-centre of the bounding box (foot position) onto the pitch ground plane.

---

### 3.4 2D Pose Estimation (Execution Stage 4)

**Purpose**: Extract 2D body keypoints for each tracked player in each frame.

**Approach**: ViTPose (top-down via MMPose). For each tracked bounding box, crop the player region with padding, run pose inference on the crop, and normalize the result to 17 COCO-format keypoints.

**Implementation**:
```
Backend: MMPose top-down inference
Input: cropped player images from tracker bounding boxes (padded by 20%)
Output: 17 keypoints in COCO format (x, y, confidence) per player per frame
Notes: keypoints are remapped into original-frame pixel coordinates after crop inference
```

**Keypoint set (COCO 17)**:
nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle

**Output schema** (`poses/shot_001_poses.json`):
```json
{
  "shot_id": "shot_001",
  "players": [
    {
      "track_id": "T001",
      "frames": [
        {
          "frame": 0,
          "keypoints": [
            {"name": "nose", "x": 412.3, "y": 156.7, "conf": 0.91},
            {"name": "left_shoulder", "x": 398.1, "y": 189.2, "conf": 0.88}
          ]
        }
      ]
    }
  ]
}
```

Coordinates are in the original video frame pixel space (not the crop space).

**Quality considerations**:
- Low-confidence keypoints (< 0.3) should be marked as occluded and excluded from triangulation
- Apply temporal smoothing (1D Gaussian filter, σ=2 frames) to reduce jitter before triangulation
- For players occupying < 60px height in frame, flag as "low resolution" — pose estimates will be unreliable

---

### 3.6 Cross-View Player Matching (Execution Stage 6)

**Purpose**: Determine which tracked player in shot A is the same person as which tracked player in shot B, so their 2D keypoints can be triangulated together.

**Approach**: Match players across synchronised views using their projected pitch positions plus visual appearance.

**Implementation**:

1. For each pair of synchronised shots at matched frames, compute each player's pitch position (from Stage 3.3)
2. Use the Hungarian algorithm to find the optimal assignment between players in shot A and shot B, minimising the sum of pitch-coordinate distances
3. Validate assignments using visual appearance (team colour should match, rough height/build should be consistent)
4. For players visible in only one view, mark as "single-view only" — these will fall back to monocular pose estimation

**Output schema** (`matching/player_matches.json`):
```json
{
  "matched_players": [
    {
      "player_id": "P001",
      "team": "A",
      "views": [
        {"shot_id": "shot_001", "track_id": "T003"},
        {"shot_id": "shot_003", "track_id": "T007"},
        {"shot_id": "shot_005", "track_id": "T001"}
      ]
    }
  ]
}
```

---

### 3.7 3D Triangulation

**Purpose**: Combine 2D keypoints from multiple calibrated views to compute 3D joint positions.

**Approach**: For each matched player at each synchronised frame, take the 2D keypoint observations from all available views and triangulate using Direct Linear Transform (DLT), weighted by keypoint confidence.

**Implementation**:

1. For each joint, collect all 2D observations across views (with confidence > 0.3)
2. If ≥ 2 views available: triangulate using weighted DLT
   - Weight each observation by its ViTPose confidence score
   - Use RANSAC to reject outlier observations (a view where the keypoint is clearly misdetected)
3. If only 1 view available: use monocular depth estimation as fallback (or mark joint as low-confidence)
4. Compute reprojection error: project the triangulated 3D point back into each camera and measure pixel distance from the original 2D detection

```python
# Pseudocode for weighted DLT triangulation
def triangulate_joint(observations, cameras, confidences):
    """
    observations: list of (u, v) pixel coordinates
    cameras: list of 3x4 projection matrices
    confidences: list of float confidence scores
    """
    A = []
    for (u, v), P, w in zip(observations, cameras, confidences):
        A.append(w * (u * P[2] - P[0]))
        A.append(w * (v * P[2] - P[1]))
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]  # dehomogenise
```

**Post-processing**:
- Temporal smoothing: apply a Savitzky-Golay filter (window=7, order=3) to each joint trajectory
- Anatomical constraint enforcement: reject frames where bone lengths deviate > 20% from the running median
- Foot-ground contact: snap ankle joints to z=0 when velocity is near zero (prevents floating/underground feet)

**Output**: NumPy arrays saved as `.npz` files per player:
```
triangulated/P001_3d_joints.npz
  - positions: (num_frames, 17, 3) float32 — joint positions in pitch coordinates (metres)
  - confidences: (num_frames, 17) float32 — per-joint confidence
  - reprojection_errors: (num_frames, 17) float32 — per-joint reprojection error in pixels
  - num_views: (num_frames, 17) int8 — number of views contributing to each joint
  - fps: float — frame rate
```

---

### 3.8 SMPL Fitting

**Purpose**: Fit the SMPL parametric body model to the triangulated 3D joints, producing smooth, anatomically valid body poses with a full mesh surface.

**Approach**: Optimisation-based fitting (SMPLify-style). Given known 3D joint targets, optimise SMPL pose (θ), shape (β), and global translation (t) parameters to minimise the distance between the SMPL joint regressor output and the triangulated joints.

**Implementation**:

```python
# Optimisation objective (per frame)
L = λ_joint * ||J(θ,β) - J_target||² +    # joint position loss
    λ_prior * L_pose_prior(θ) +              # penalise unlikely poses (GMM prior)
    λ_shape * ||β||² +                        # shape regularisation
    λ_smooth * ||θ_t - θ_{t-1}||² +          # temporal smoothness
    λ_ground * L_ground_contact               # foot-ground penetration penalty
```

**Key details**:
- Use the SMPL neutral model (10 shape parameters, 72 pose parameters for 24 joints)
- Initialise from a standing pose and optimise per-frame, using the previous frame's result as initialisation
- The joint regressor maps SMPL's 24 joints to the COCO 17 keypoints (a well-established mapping exists)
- Shape parameters (β) should be shared across all frames for a given player (body shape doesn't change)
- For single-view-only frames, apply stronger pose priors to prevent implausible solutions

**Library options**:
- `smplx` (official SMPL-X Python package from MPI) for the body model
- PyTorch optimiser (Adam, lr=0.01, 100 iterations per frame) for the fitting
- VPoser (learned pose prior from MPI) as an alternative to GMM priors

**Output schema** (`smpl/P001_smpl.npz`):
```
  - betas: (10,) float32 — body shape parameters (shared across all frames)
  - poses: (num_frames, 72) float32 — axis-angle rotations for 24 joints per frame
  - transl: (num_frames, 3) float32 — global translation per frame
  - fps: float
```

---

### 3.9 Export

**Purpose**: Convert SMPL parameter sequences into industry-standard animation files.

#### 3.9.1 FBX Export (UE5 target)

**Approach**: Use Blender in headless mode as the conversion bridge. SMPL parameters → Blender armature animation → FBX export.

**Pipeline**:
1. Load the SMPL mesh and skeleton into Blender using the `smplx-blender-addon`
2. Apply pose and shape parameters per frame as keyframed bone rotations
3. Export as FBX with the following settings:
   - Scale: 1.0 (metres)
   - Forward axis: -Y (UE5 convention)
   - Up axis: Z
   - Armature: include, with bone hierarchy
   - Animation: baked, all frames
   - Mesh: include SMPL mesh (optional — useful for preview but will be replaced by MetaHuman in UE5)

**UE5 import workflow** (documented for the user):
1. Import FBX into UE5 — set skeleton to `SK_Meshcapade_fbx` (from Meshcapade plugin) or create a new skeleton
2. Use UE5's IK Retargeter to map from SMPL skeleton → MetaHuman skeleton
3. Right-click animation → "Retarget Animations" → select MetaHuman target

#### 3.9.2 glTF Export (browser viewer target)

**Approach**: Export a simplified skeleton + animation as glTF 2.0 binary (.glb) for loading into Three.js.

**Pipeline**:
1. Build a simplified skeleton (24 joints, matching SMPL topology) as a glTF skin
2. Encode joint rotations as quaternion animation channels (sampled)
3. Optionally include a low-poly mesh (decimated SMPL mesh, ~2000 triangles) for visual preview
4. Include ball trajectory as a separate animated node
5. Include a ground plane mesh representing the pitch (105m × 68m, textured with pitch markings)

**Output files**:
```
export/
  ├── fbx/
  │   ├── P001_animation.fbx
  │   ├── P002_animation.fbx
  │   └── ball_trajectory.fbx
  ├── gltf/
  │   ├── scene.glb          # all players + ball + pitch in one file
  │   └── scene_metadata.json # player IDs, teams, frame range, etc.
  └── pitch_model.fbx        # static pitch mesh for UE5
```

---

## 4. Browser-Based 3D Viewer

A standalone HTML file that loads the glTF output and provides an interactive 3D preview.

### 4.1 Features

- **Load scene**: drag-and-drop or file picker for the `.glb` file
- **Playback controls**: play/pause, scrub timeline, speed control (0.25x–2x), frame-by-frame stepping
- **Camera controls**: orbit (Three.js OrbitControls), preset camera positions (broadcast angle, behind goal, bird's eye, player POV)
- **Player selection**: click a player to highlight them and lock camera follow
- **Skeleton overlay**: toggle wireframe skeleton visualisation over the mesh
- **Joint confidence heatmap**: colour joints by confidence score (green = high, red = low) to quickly identify where the pipeline struggled
- **Grid/pitch toggle**: show/hide pitch markings and coordinate grid
- **Export camera**: copy current virtual camera position/rotation as JSON (for recreating the same angle in UE5)

### 4.2 Technical Stack

- Three.js (r128 or later) for rendering
- GLTFLoader for scene import
- OrbitControls for camera interaction
- AnimationMixer for playback
- Single HTML file, no build step required — can be opened directly from the filesystem

### 4.3 Data Flow

The viewer reads:
1. `scene.glb` — the 3D scene with all animations
2. `scene_metadata.json` — player names/IDs, team assignments, confidence data per joint per frame, frame rate, original video timestamps

Confidence data is stored as a custom glTF extension or as a separate JSON sidecar to avoid bloating the binary.

---

## 5. CLI Interface

### 5.1 Main Command

```bash
python recon.py \
  --input match_replay.mp4 \
  --output ./output/ \
  --stages all \
  --fps 25 \
  --viewer
```

### 5.2 Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Video file(s). Accepts single file or glob pattern | required |
| `--output` | Output directory for all pipeline artifacts | `./output/` |
| `--stages` | Which stages to run: `all`, or comma-separated list like `1,2,3` or `segmentation,calibration,sync` | `all` |
| `--from-stage` | Resume from a specific stage (uses cached outputs from previous stages) | `1` |
| `--fps` | Target frame rate for processing (downsample if source is higher) | source fps |
| `--players` | Filter to specific players by track ID (e.g., `P001,P003`) | all players |
| `--viewer` | Launch browser viewer after export | false |
| `--device` | Compute device: `cuda`, `cpu`, `mps` | auto-detect |
| `--config` | Path to YAML config file for advanced settings | `config/default.yaml` |

### 5.3 Stage-Specific Overrides

```bash
# Re-run only triangulation and export with different smoothing
python recon.py \
  --input match_replay.mp4 \
  --output ./output/ \
  --from-stage triangulation \
  --config config/smooth_heavy.yaml

# Run just the viewer on existing output
python recon.py \
  --output ./output/ \
  --viewer
```

### 5.4 Configuration File

```yaml
# config/default.yaml

shot_segmentation:
  detector: content          # content | adaptive | threshold
  threshold: 30.0
  min_shot_duration_s: 0.5

calibration:
  pitch_model: fifa_standard  # 105m x 68m
  max_reprojection_error: 15.0
  keyframe_interval: 5        # frames between calibration solves for tracking cameras

sync:
  method: ball_trajectory     # ball_trajectory | player_formation | hybrid
  search_window_s: 5.0        # max offset to search
  min_overlap_frames: 25

detection:
  model: yolov8x              # yolov8n | yolov8s | yolov8m | yolov8l | yolov8x
  ball_model: yolov8_ball_custom
  confidence_threshold: 0.5

tracking:
  algorithm: bytetrack
  max_age: 30                  # frames to keep lost track alive

pose_estimation:
  model_config: td-hm_ViTPose-small_8xb64-210e_coco-256x192
  checkpoint: null             # optional explicit checkpoint path
  device: auto                 # auto | cpu | mps | cuda
  min_confidence: 0.3
  temporal_smoothing_sigma: 2.0

triangulation:
  method: weighted_dlt
  ransac_threshold: 15.0       # pixels
  min_views: 2
  temporal_filter: savgol       # savgol | gaussian | none
  temporal_filter_window: 7
  bone_length_tolerance: 0.2

smpl_fitting:
  iterations: 100
  learning_rate: 0.01
  use_vposer: true
  lambda_joint: 1.0
  lambda_prior: 0.01
  lambda_shape: 0.05
  lambda_smooth: 0.5
  lambda_ground: 10.0

export:
  fbx: true
  gltf: true
  include_mesh: true
  gltf_mesh_decimation: 2000   # target face count for preview mesh
  coordinate_system: unreal    # unreal | blender | unity
```

---

## 6. Output Directory Structure

```
output/
├── shots/
│   ├── shots_manifest.json
│   ├── shot_001.mp4
│   ├── shot_001_thumb.jpg
│   ├── shot_002.mp4
│   └── ...
├── calibration/
│   ├── shot_001_calibration.json
│   └── ...
├── sync/
│   └── sync_map.json
├── tracks/
│   ├── shot_001_tracks.json
│   └── ...
├── poses/
│   ├── shot_001_poses.json
│   └── ...
├── matching/
│   └── player_matches.json
├── triangulated/
│   ├── P001_3d_joints.npz
│   ├── P002_3d_joints.npz
│   └── ...
├── smpl/
│   ├── P001_smpl.npz
│   └── ...
├── export/
│   ├── fbx/
│   │   ├── P001_animation.fbx
│   │   └── ...
│   ├── gltf/
│   │   ├── scene.glb
│   │   └── scene_metadata.json
│   └── pitch_model.fbx
├── viewer/
│   └── index.html
├── logs/
│   ├── pipeline.log
│   └── stage_timings.json
└── quality_report.json          # per-stage confidence summary
```

---

## 7. Quality Report

The pipeline generates a summary report highlighting areas that may need manual attention.

```json
{
  "overall_confidence": 0.74,
  "stages": {
    "calibration": {
      "shots_calibrated": 5,
      "shots_failed": 1,
      "avg_reprojection_error": 4.2,
      "worst_shot": {"id": "shot_004", "error": 18.7, "reason": "insufficient pitch lines visible"}
    },
    "sync": {
      "pairs_synced": 4,
      "avg_confidence": 0.88,
      "worst_pair": {"shots": ["shot_001", "shot_004"], "confidence": 0.43}
    },
    "triangulation": {
      "players_reconstructed": 8,
      "avg_views_per_joint": 2.7,
      "low_confidence_joints": [
        {"player": "P003", "joint": "left_wrist", "frames": [45, 46, 47], "reason": "single view only"}
      ]
    },
    "smpl_fitting": {
      "avg_joint_error_mm": 34.2,
      "players_with_ground_penetration": ["P005"],
      "temporal_jitter_score": 0.12
    }
  }
}
```

---

## 8. Dependencies

### 8.1 Python Packages

```
# Core
torch >= 2.0
torchvision
numpy
scipy
opencv-python

# Detection & Tracking
ultralytics                    # YOLOv8
supervision                    # ByteTrack integration

# Pose Estimation
mmpose
mmengine
mmcv

# Runtime note
# This implementation runs pose on tracked player crops, so it does not require
# a separate detector pass inside the pose stage.

# Shot Detection
scenedetect[opencv]

# SMPL
smplx                          # SMPL body model
chumpy                         # SMPL dependency
vposer                         # learned pose prior (optional)

# Export
bpy                            # Blender Python API (for FBX export)
trimesh                        # mesh processing
pygltflib                      # glTF construction

# Utilities
tqdm
pyyaml
click                          # CLI framework
```

### 8.2 External Tools

- **Blender** (≥ 3.6): headless mode for FBX export. Install via `snap install blender --classic` or download from blender.org
- **FFmpeg**: video file manipulation (shot extraction, frame extraction)
- **SMPL model files**: download from https://smpl.is.tue.mpg.de/ (requires registration — academic/non-commercial license)

### 8.3 GPU Requirements

- Minimum: NVIDIA GPU with 8GB VRAM (RTX 3070 or equivalent)
- Recommended: 12GB+ VRAM (RTX 4080 or A5000) for ViTPose-Large + YOLOv8x concurrent processing
- SMPL fitting runs on GPU but is not the bottleneck
- Estimated processing time: ~10-20 minutes per 10 seconds of multi-angle footage (on RTX 4080)

---

## 9. Known Limitations & Future Work

### 9.1 Current Limitations

- **Ball 3D position**: ball tracking in 2D is reliable, but 3D triangulation of the ball is hard because it moves faster than players and is often occluded. v1 may produce noisy ball trajectories.
- **Player re-identification across shots**: if the camera cuts away and returns, re-identifying the same player requires jersey number recognition or appearance matching, which is not robust in v1.
- **Tight close-ups**: shots where a single player fills most of the frame may fail camera calibration (too few pitch lines visible) but will produce excellent 2D pose estimates. These could be used for monocular HMR as a fallback.
- **Rapid motion blur**: sprint phases and shots produce motion blur that degrades both detection and pose estimation.
- **SMPL limitations**: SMPL is a body-only model — no clothing, hair, or equipment. The exported mesh is a naked body shape, intended to be replaced by a MetaHuman in UE5.

### 9.2 Future Enhancements

- **Jersey number recognition** for automatic player identification
- **Motion infilling** using motion diffusion models (MDM) to fill gaps where a player is occluded in all views
- **Crowd reconstruction** using procedural animation driven by detected crowd motion
- **Automatic MetaHuman matching** from player face crops
- **Live processing mode** for near-real-time reconstruction from multi-camera stadium feeds
- **SMPL-X upgrade** for hand and face pose when close-up footage is available

---

## 10. Development Phases

### Phase 1: Foundation (Weeks 1-3)
- Shot segmentation (Stage 1)
- Player detection and tracking (Stage 4)
- 2D pose estimation (Stage 5)
- Basic CLI with stage caching

### Phase 2: Multi-View Core (Weeks 4-7)
- Camera calibration (Stage 2)
- Temporal synchronisation (Stage 3)
- Cross-view player matching (Stage 6)
- 3D triangulation (Stage 7)

### Phase 3: Body Model & Export (Weeks 8-10)
- SMPL fitting (Stage 8)
- FBX and glTF export (Stage 9)
- Browser viewer

### Phase 4: Polish (Weeks 11-12)
- Quality report generation
- Confidence visualisation in viewer
- Documentation and example data
- End-to-end testing on 3-5 iconic football moments

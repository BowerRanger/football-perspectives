// Anchor tag palette. Colours are data colours (drawn on the frame overlay).

export interface AnchorTag {
  id: string
  label: string
  color: string
  range: string
  description: string
  /** Keyboard shortcut that selects this tag. */
  key: string
}

export const TAGS: AnchorTag[] = [
  {
    id: "grounded", label: "Grounded", color: "#34d399", range: "z = 0.11 m", key: "1",
    description: "Ball on the pitch (rolling, stationary, at feet). Hard knot — the parabola fit will pass exactly through the anchor's pixel ray-cast at ground level.",
  },
  {
    id: "airborne_low", label: "Airborne — low", color: "#fbbf24", range: "0 – 2 m", key: "2",
    description: "Ball in the air, ankle-to-waist height. A low pass or rebound. Acts as a soft height hint (bucket midpoint 1 m) — the fit is biased but not pinned exactly, because the bucket is coarse.",
  },
  {
    id: "airborne_mid", label: "Airborne — mid", color: "#fb923c", range: "2 – 10 m", key: "3",
    description: "Ball in the air, head-to-rooftop height. The most common aerial pass. Bucket midpoint 6 m, soft constraint.",
  },
  {
    id: "airborne_high", label: "Airborne — high", color: "#f87171", range: "10 m +", key: "4",
    description: "Ball high in the air — goal kick, long clearance, lobbed shot. Bucket midpoint 15 m, soft constraint.",
  },
  {
    id: "off_screen_flight", label: "Off-screen flight", color: "#94a3b8", range: "no pixel", key: "5",
    description: "Ball is airborne but not visible in this frame (occluded, off-screen, motion-blurred beyond recognition). No pixel position; tells the pipeline 'keep treating this frame as flight even though WASB missed it'. Use the dedicated button under the frame.",
  },
  {
    id: "kick", label: "Kick (foot)", color: "#60a5fa", range: "z = 0.11 m, event", key: "6",
    description: "Ball leaves a player's foot. Hard knot at ground level; splits the flight run — a new flight segment starts at the kick frame. Place the crosshair at the ball, not the foot.",
  },
  {
    id: "catch", label: "Catch", color: "#a78bfa", range: "z = 1.5 m, event", key: "7",
    description: "Ball stops in a player's hands (goalkeeper catch, throw-in catch). Hard knot at chest height; ends the current flight run.",
  },
  {
    id: "bounce", label: "Bounce", color: "#f472b6", range: "z = 0.11 m, event", key: "8",
    description: "Ball touches the ground briefly. Hard knot at ground level; splits the flight run — the segment before and after the bounce are fit separately, with the bounce frame excluded from both.",
  },
  {
    id: "header", label: "Header", color: "#facc15", range: "z ≈ 2.5 m, event", key: "9",
    description: "Ball contacts a player's head mid-flight. Soft constraint at 2.5 m (typical jumping-header height) — the fit isn't pinned exactly because header heights vary. Splits the flight run. For unusually high headers, drop an airborne_mid/high anchor on the same frame for a stronger height hint.",
  },
  {
    id: "player_touch", label: "Player touch", color: "#22d3ee", range: "body-pinned", key: "0",
    description: "A player contacts the ball with a body part. In events mode the ball's 3D position at this frame is pinned to the contacting joint (depth-stable, occlusion-robust). Pick a player + body part, then click the ball — the nearest reconstructed joint under the cursor is auto-suggested. Set type to Shot for a strike at goal.",
  },
  {
    id: "goal_impact", label: "Goal impact", color: "#f59e0b", range: "ray ∩ goal, event", key: "-",
    description: "Ball strikes the goal frame or net (post, crossbar, back net, side net). Hard knot at the ray-geometry intersection. The element is auto-suggested from your click; override it in the panel.",
  },
  {
    id: "pitch_fix", label: "Pitch fix", color: "#a3e635", range: "z = 0.11 m, exact", key: "=",
    description: "Ball visibly coincides with a known pitch feature (penalty spot, line, corner arc). Saves a grounded anchor snapped to the feature's exact FIFA coordinates — an exact hard knot from one click.",
  },
]

export const tagById = (id: string): AnchorTag | undefined => TAGS.find((t) => t.id === id)
export const tagColour = (id: string, fallback = "#94a3b8"): string => tagById(id)?.color ?? fallback

// SMPL contact bones offered for a player touch (mirror BONE_TO_SMPL_INDEX).
export const TOUCH_BONES: { id: string; label: string }[] = [
  { id: "r_foot", label: "Right foot" },
  { id: "l_foot", label: "Left foot" },
  { id: "r_knee", label: "Right knee" },
  { id: "l_knee", label: "Left knee" },
  { id: "head", label: "Head" },
  { id: "chest", label: "Chest" },
  { id: "r_shoulder", label: "Right shoulder" },
  { id: "l_shoulder", label: "Left shoulder" },
  { id: "r_hand", label: "Right hand" },
  { id: "l_hand", label: "Left hand" },
]

export const GOAL_ELEMENTS: { id: string; label: string }[] = [
  { id: "post", label: "Post" },
  { id: "crossbar", label: "Crossbar" },
  { id: "back_net", label: "Back net" },
  { id: "side_net", label: "Side net" },
  { id: "mouth", label: "Goal mouth / line cross" },
]

export const TOUCH_TYPES: { id: string; label: string }[] = [
  { id: "none", label: "None (plain contact)" },
  { id: "shot", label: "Shot (strike at goal)" },
  { id: "volley", label: "Volley" },
]

export const SPIN_OPTIONS: { id: string; label: string }[] = [
  { id: "none", label: "None / unknown" },
  { id: "instep_curl_right", label: "Instep curl (R-foot, R→L)" },
  { id: "instep_curl_left", label: "Instep curl (L-foot, L→R)" },
  { id: "outside_curl_right", label: "Outside curl (R-foot, L→R)" },
  { id: "outside_curl_left", label: "Outside curl (L-foot, R→L)" },
  { id: "topspin", label: "Top-spin (dipping)" },
  { id: "backspin", label: "Back-spin (floating)" },
  { id: "knuckle", label: "Knuckle (no spin)" },
]

export const AUTO = "__auto__"

import numpy as np

PITCH_LENGTH = 105.0  # metres (FIFA standard)
PITCH_WIDTH = 68.0    # metres
# Origin at near-left corner; x along length (0→105), y along width (0→68), z up
# "near" = y=0 (near side, bottom of broadcast view)
# "far"  = y=68 (far side, top of broadcast view)

_GOAL_HALF = 7.32 / 2.0  # 3.66m — half goal width
_GOAL_HEIGHT = 2.44       # metres — crossbar height
_FLAG_HEIGHT = 1.5        # metres — corner flag height

_CIRCLE_R = 9.15  # centre circle radius

# Penalty arc: radius 9.15m from penalty spot, intersects 18-yard line
_LEFT_D_DY = (9.15 ** 2 - (16.5 - 11.0) ** 2) ** 0.5   # ≈ 7.312
_RIGHT_D_DY = _LEFT_D_DY

FIFA_LANDMARKS: dict[str, np.ndarray] = {
    # Corners (near = y=0, far = y=68)
    "corner_near_left":             np.array([0.0,   0.0,  0.0]),
    "corner_near_right":            np.array([105.0, 0.0,  0.0]),
    "corner_far_left":              np.array([0.0,   68.0, 0.0]),
    "corner_far_right":             np.array([105.0, 68.0, 0.0]),
    # Halfway line
    "halfway_near":                 np.array([52.5,  0.0,  0.0]),
    "halfway_far":                  np.array([52.5,  68.0, 0.0]),
    "center_spot":                  np.array([52.5,  34.0, 0.0]),
    # Centre circle intersections with halfway line
    "centre_circle_near":           np.array([52.5,  34.0 - _CIRCLE_R, 0.0]),
    "centre_circle_far":            np.array([52.5,  34.0 + _CIRCLE_R, 0.0]),
    # Penalty spots
    "left_penalty_spot":            np.array([11.0,  34.0, 0.0]),
    "right_penalty_spot":           np.array([94.0,  34.0, 0.0]),
    # Left 6-yard box (near = lower y, far = higher y)
    "left_6yard_near_left":         np.array([0.0,   24.84, 0.0]),
    "left_6yard_near_right":        np.array([5.5,   24.84, 0.0]),
    "left_6yard_far_right":         np.array([5.5,   43.16, 0.0]),
    "left_6yard_far_left":          np.array([0.0,   43.16, 0.0]),
    # Right 6-yard box
    "right_6yard_near_left":        np.array([99.5,  24.84, 0.0]),
    "right_6yard_near_right":       np.array([105.0, 24.84, 0.0]),
    "right_6yard_far_right":        np.array([105.0, 43.16, 0.0]),
    "right_6yard_far_left":         np.array([99.5,  43.16, 0.0]),
    # Left 18-yard box
    "left_18yard_near_left":        np.array([0.0,   13.84, 0.0]),
    "left_18yard_near_right":       np.array([16.5,  13.84, 0.0]),
    "left_18yard_far_right":        np.array([16.5,  54.16, 0.0]),
    "left_18yard_far_left":         np.array([0.0,   54.16, 0.0]),
    # Right 18-yard box
    "right_18yard_near_left":       np.array([88.5,  13.84, 0.0]),
    "right_18yard_near_right":      np.array([105.0, 13.84, 0.0]),
    "right_18yard_far_right":       np.array([105.0, 54.16, 0.0]),
    "right_18yard_far_left":        np.array([88.5,  54.16, 0.0]),
    # Penalty arc / 18-yard line intersections ("D" shape)
    "left_18yard_d_near":           np.array([16.5,  34.0 - _LEFT_D_DY, 0.0]),
    "left_18yard_d_far":            np.array([16.5,  34.0 + _LEFT_D_DY, 0.0]),
    "right_18yard_d_near":          np.array([88.5,  34.0 - _RIGHT_D_DY, 0.0]),
    "right_18yard_d_far":           np.array([88.5,  34.0 + _RIGHT_D_DY, 0.0]),
    # Goal structure — left goal (x=0)
    "left_goal_near_post_base":     np.array([0.0,   34.0 - _GOAL_HALF, 0.0]),
    "left_goal_near_post_top":      np.array([0.0,   34.0 - _GOAL_HALF, _GOAL_HEIGHT]),
    "left_goal_far_post_base":      np.array([0.0,   34.0 + _GOAL_HALF, 0.0]),
    "left_goal_far_post_top":       np.array([0.0,   34.0 + _GOAL_HALF, _GOAL_HEIGHT]),
    # Goal structure — right goal (x=105)
    "right_goal_near_post_base":    np.array([105.0, 34.0 - _GOAL_HALF, 0.0]),
    "right_goal_near_post_top":     np.array([105.0, 34.0 - _GOAL_HALF, _GOAL_HEIGHT]),
    "right_goal_far_post_base":     np.array([105.0, 34.0 + _GOAL_HALF, 0.0]),
    "right_goal_far_post_top":      np.array([105.0, 34.0 + _GOAL_HALF, _GOAL_HEIGHT]),
    # Corner flag tops
    "corner_near_left_flag_top":    np.array([0.0,   0.0,  _FLAG_HEIGHT]),
    "corner_near_right_flag_top":   np.array([105.0, 0.0,  _FLAG_HEIGHT]),
    "corner_far_left_flag_top":     np.array([0.0,   68.0, _FLAG_HEIGHT]),
    "corner_far_right_flag_top":    np.array([105.0, 68.0, _FLAG_HEIGHT]),
}

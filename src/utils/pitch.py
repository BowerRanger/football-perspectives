import numpy as np

PITCH_LENGTH = 105.0  # metres (FIFA standard)
PITCH_WIDTH = 68.0    # metres
# Origin at top-left corner; x along length (0→105), y along width (0→68), z up

FIFA_LANDMARKS: dict[str, np.ndarray] = {
    # Corners
    "top_left_corner":     np.array([0.0,   0.0,  0.0]),
    "top_right_corner":    np.array([105.0, 0.0,  0.0]),
    "bottom_left_corner":  np.array([0.0,   68.0, 0.0]),
    "bottom_right_corner": np.array([105.0, 68.0, 0.0]),
    # Halfway line
    "halfway_top":         np.array([52.5,  0.0,  0.0]),
    "halfway_bottom":      np.array([52.5,  68.0, 0.0]),
    "center_spot":         np.array([52.5,  34.0, 0.0]),
    # Penalty spots
    "left_penalty_spot":   np.array([11.0,  34.0, 0.0]),
    "right_penalty_spot":  np.array([94.0,  34.0, 0.0]),
    # Left goal area (5.5m box)
    "left_goal_area_tl":   np.array([0.0,   24.84, 0.0]),
    "left_goal_area_tr":   np.array([5.5,   24.84, 0.0]),
    "left_goal_area_br":   np.array([5.5,   43.16, 0.0]),
    "left_goal_area_bl":   np.array([0.0,   43.16, 0.0]),
    # Right goal area
    "right_goal_area_tl":  np.array([99.5,  24.84, 0.0]),
    "right_goal_area_tr":  np.array([105.0, 24.84, 0.0]),
    "right_goal_area_br":  np.array([105.0, 43.16, 0.0]),
    "right_goal_area_bl":  np.array([99.5,  43.16, 0.0]),
    # Left penalty box (16.5m box)
    "left_box_tl":         np.array([0.0,   13.84, 0.0]),
    "left_box_tr":         np.array([16.5,  13.84, 0.0]),
    "left_box_br":         np.array([16.5,  54.16, 0.0]),
    "left_box_bl":         np.array([0.0,   54.16, 0.0]),
    # Right penalty box
    "right_box_tl":        np.array([88.5,  13.84, 0.0]),
    "right_box_tr":        np.array([105.0, 13.84, 0.0]),
    "right_box_br":        np.array([105.0, 54.16, 0.0]),
    "right_box_bl":        np.array([88.5,  54.16, 0.0]),
}

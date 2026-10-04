import numpy as np

from src.utils import team_clustering as tc
from src.utils.kit_palette import hex_to_lab


def _player(shirt, shorts=None, rng=None, jitter=1.5):
    j = (lambda: rng.normal(0, jitter, 3)) if rng is not None else (lambda: 0)
    shirt_lab = hex_to_lab(shirt) + j()
    return {"torso": shirt_lab, "shorts": hex_to_lab(shorts or shirt) + j()}


def _world(rng):
    players, px = {}, {}
    for i in range(10):
        players[f"P{i:03d}"] = _player("#c8102e", rng=rng)
        px[f"P{i:03d}"] = 20 + 5 * i
    for i in range(10, 20):
        players[f"P{i:03d}"] = _player("#034694", rng=rng)
        px[f"P{i:03d}"] = 25 + 4 * i % 40
    players["P020"] = _player("#2a9d6f", rng=rng)      # keeper near x=0
    px["P020"] = 3.0
    players["P021"] = _player("#33373b", rng=rng)      # keeper near x=105
    px["P021"] = 101.0
    players["P022"] = _player("#e8df3a", "#151515", rng=rng)   # referee mid
    px["P022"] = 50.0
    return players, px


def test_cluster_teams_peels_keepers_and_referee():
    players, px = _world(np.random.default_rng(0))
    c = tc.cluster_teams(players, px)
    assert set(c.keepers) == {"P020", "P021"}
    assert c.keepers["P020"] == 0 and c.keepers["P021"] == 1
    assert c.referees == ["P022"]
    red = {c.team_of[f"P{i:03d}"] for i in range(10)}
    blue = {c.team_of[f"P{i:03d}"] for i in range(10, 20)}
    assert len(red) == 1 and len(blue) == 1 and red != blue
    assert c.team_of["P000"] == 0  # deterministic: team 0 holds the smallest pid


def test_at_most_one_keeper_per_goal():
    players, px = _world(np.random.default_rng(1))
    players["P023"] = _player("#ff00ff")
    px["P023"] = 12.0                                   # second odd one near x=0
    c = tc.cluster_teams(players, px)
    assert sum(1 for s in c.keepers.values() if s == 0) == 1
    assert "P023" in c.referees or "P023" in c.keepers


def test_outlier_without_pitch_x_is_referee_with_note():
    players, px = _world(np.random.default_rng(2))
    del px["P022"]
    c = tc.cluster_teams(players, px)
    assert "P022" in c.referees
    assert any("P022" in n for n in c.notes)


def test_too_few_players():
    c = tc.cluster_teams({"P1": _player("#ff0000"), "P2": _player("#0000ff")}, {})
    assert c.notes == ["too_few_players_for_clustering"]


def test_keeper_team_from_mean_x():
    c = tc.TeamClustering(team_mean_x=[10.0, 30.0])
    assert tc.keeper_team(0, c) == (0, True)       # team 0 sits nearer x=0
    assert tc.keeper_team(1, c) == (1, True)
    c2 = tc.TeamClustering(team_mean_x=[10.0, 11.0])
    assert tc.keeper_team(0, c2)[1] is False
    assert tc.keeper_team(0, tc.TeamClustering(team_mean_x=[None, None]))[1] is False


def _synthetic_frame():
    img = np.zeros((300, 200, 3), np.uint8)
    img[:] = (40, 150, 40)                              # grass (BGR)
    img[60:120, 70:130] = (46, 16, 200)                 # red torso (BGR of #c8102e)
    img[120:160, 70:130] = (255, 255, 255)              # white shorts
    img[180:230, 70:130] = (30, 30, 30)                 # dark socks
    return img


def _kp():
    kp = np.zeros((17, 3))
    pts = {tc.L_SHO: (75, 62), tc.R_SHO: (125, 62), tc.L_HIP: (78, 118), tc.R_HIP: (122, 118),
           tc.L_KNE: (80, 170), tc.R_KNE: (120, 170), tc.L_ANK: (80, 225), tc.R_ANK: (120, 225),
           tc.L_ELB: (60, 90), tc.R_ELB: (140, 90)}
    for i, (x, y) in pts.items():
        kp[i] = (x, y, 0.9)
    return kp


def test_sample_player_regions_reads_kit_colours():
    s = tc.sample_player_regions(_synthetic_frame(), _kp())
    assert {"torso", "shorts", "socks"} <= set(s)
    assert np.linalg.norm(s["torso"] - hex_to_lab("#c8102e")) < 6
    assert s["shorts"][0] > 90                         # white-ish shorts
    assert s["socks"][0] < 25                          # dark socks


def test_low_confidence_keypoints_drop_regions():
    kp = _kp()
    kp[[tc.L_KNE, tc.R_KNE]] = 0
    s = tc.sample_player_regions(_synthetic_frame(), kp)
    assert "shorts" not in s and "socks" not in s and "torso" in s


def test_grass_only_region_is_none():
    img = np.zeros((300, 200, 3), np.uint8)
    img[:] = (40, 150, 40)
    assert "torso" not in tc.sample_player_regions(img, _kp())


def test_aggregate_samples_median():
    a = {"torso": np.array([50.0, 10, 10])}
    b = {"torso": np.array([60.0, 10, 10])}
    c = {"torso": np.array([70.0, 10, 10]), "socks": np.array([1.0, 0, 0])}
    out = tc.aggregate_samples([a, b, c])
    assert out["torso"][0] == 60.0 and out["socks"][0] == 1.0

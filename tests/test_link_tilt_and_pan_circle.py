import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import measure_link_tilt as S7  # noqa: E402
import measure_pan_circle as S8  # noqa: E402


def test_s7_recovers_a_known_arm():
    # Offsets off by up to 22 deg (the size of the 10/08 home error), scales off by 2-3 %, base tilted.
    err, rms = S7.run_selftest(noise_deg=0.3, seed=0)
    assert rms < 0.5
    assert max(abs(e) for k, e in err.items() if k.startswith("o_")) < 0.6
    assert max(abs(e) for k, e in err.items() if k.startswith("s_")) < 0.01


def test_s7_nominal_constants_predict_zero_residual():
    p0 = S7.nominal_params()
    rows = S7.synthetic_rows(np.random.default_rng(2), p0, noise_deg=0.0)
    p, res, _ = S7.fit(rows, S7.free_params(rows, True), p0, iters=5)
    assert np.abs(res).max() < 1e-6


def test_s7_pitch_does_not_fold_past_vertical():
    # A link tipped 10 deg past straight down must read -100, not fold onto -80 (the 2026-09-29 bug).
    assert S7.to_value(-80.0, "b") == -100.0
    assert S7.to_value(80.0, "b") == 100.0
    assert S7.to_value(-80.0, "f") == -80.0
    st = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0, "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": 50.0}
    p = S7.nominal_params()
    # the forearm face points along link3 +x: at all-zero readings it is FK-horizontal give or take the offsets
    a = S7.predict_deg(S7.FACES["forearm"], st, p)
    expect = -(p["o_shoulder_lift"] + p["o_elbow_flex"])
    assert abs(a - expect) < 1e-6


def test_s7_upper_arm_face_is_the_link_axis_not_the_joint_line():
    # CAD: the upper arm's long faces have normals +-x, so the bar is vertical at joint2 = 0 even though
    # joint2 -> joint3 leans 20.14 deg (S6 4-a). The face reads 90 deg at q2 = 0, not 69.86.
    p = dict(S7.nominal_params(), o_shoulder_lift=0.0, s_shoulder_lift=1.8)
    st = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0, "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": 50.0}
    assert abs(S7.predict_deg(S7.FACES["upper"], st, p) - 90.0) < 1e-6


def test_s8_recovers_a_known_pan_axis():
    rng = np.random.default_rng(1)
    truth = {"cx": 1.4, "cy": -0.8, "s": S8.deg_per_unit_nominal() * 1.01, "o": -3.5}
    rows = S8.synthetic(rng, truth, 0.1)
    alpha = {c: S8.tip_polar_in_arm_plane({j: float(np.mean([r["state"][j] for r in rows if r["circle"] == c]))
                                            for j in S8.JOINTS})[1] for c in ("C0", "C1")}
    P, res = S8.fit(rows, alpha)
    assert abs(P["cx"] - truth["cx"]) < 0.15 and abs(P["cy"] - truth["cy"]) < 0.15
    assert abs(P["s"] - truth["s"]) < 0.01 and abs(P["o"] - truth["o"]) < 0.5
    assert math.sqrt((res ** 2).sum(1).mean()) < 0.2


def test_s8_centre_does_not_depend_on_the_radius():
    # The whole point of S8: a wrong lift/elbow zero only changes the radius. Same axis, two radii, no noise.
    truth = {"cx": -2.0, "cy": 3.0, "s": S8.deg_per_unit_nominal(), "o": 0.0}
    rows = S8.synthetic(np.random.default_rng(0), truth, 0.0)
    for c in ("C0", "C1"):
        sub = [r for r in rows if r["circle"] == c]
        cx, cy, _ = S8.kasa(np.array([[r["x"], r["y"]] for r in sub]))
        assert abs(cx - truth["cx"]) < 0.05 and abs(cy - truth["cy"]) < 0.05

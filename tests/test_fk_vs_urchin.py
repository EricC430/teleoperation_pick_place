"""Cross-check the hand-coded fk.py against an independent URDF FK implementation.

placo (lerobot's FK oracle) has no Windows wheels (D026), so we use urchin, a
pure-Python URDF library, purely as a test oracle. This catches transform-order
and axis-sign mistakes that hand-computed closed forms might not.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from reach_logger import fk

urchin = pytest.importorskip("urchin")

_URDF = Path(__file__).resolve().parents[1] / "assets" / "omx_f" / "omx_f.urdf"
_ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5")


@pytest.fixture(scope="module")
def robot():
    return urchin.URDF.load(str(_URDF), lazy_load_meshes=True)


def _urchin_ee(robot, q):
    cfg = dict(zip(_ARM_JOINTS, q))
    fk_map = robot.link_fk(cfg=cfg)
    ee_link = next(link for link in fk_map if link.name == "end_effector_link")
    return np.asarray(fk_map[ee_link], dtype=float)


@pytest.mark.parametrize("seed", range(25))
def test_ee_transform_matches_urchin_at_random_configs(robot, seed):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-math.pi, math.pi, size=fk.N_JOINTS).tolist()

    mine = fk.ee_transform(q)
    ref = _urchin_ee(robot, q)

    np.testing.assert_allclose(mine, ref, atol=1e-9)


def test_reach_and_azimuth_match_urchin(robot):
    rng = np.random.default_rng(99)
    for _ in range(10):
        q = rng.uniform(-math.pi, math.pi, size=fk.N_JOINTS).tolist()
        x, y, _z = _urchin_ee(robot, q)[:3, 3]
        ref_reach = math.hypot(x + 0.01125, y) * 100.0
        ref_az = math.degrees(math.atan2(y, x + 0.01125))
        assert fk.reach_cm(q) == pytest.approx(ref_reach, abs=1e-7)
        assert fk.azimuth_base_deg(q) == pytest.approx(ref_az, abs=1e-7)

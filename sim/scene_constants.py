"""Scene geometry for the simulated pick-and-place cell.

🔴 READ THIS BEFORE TRUSTING ANY NUMBER HERE.

`docs/experiment_spec.md` §3 (場景常數表) is the authority for every constant in this file, and
most of its rows are still BLANK — table height, camera x/y/z, camera pitch, lighting are all
`___` as of 2026-09-18.

So the values below are of two kinds, and they are labelled:

  MEASURED    — taken from a real measurement already in the repo (S1 tape measure, the S2
                seeded placement list, `configs/record_omx.yaml`).
  PLACEHOLDER — invented so the scene can be built and rendered at all. NOT the real cell.

A scene built from PLACEHOLDER values is good for exactly one thing: checking that the pipeline
runs (S4 §5-5 says so explicitly). It is NOT geometrically aligned with the real cell, and any
dataset recorded from it must say so in its `meta`.

The path to replacing the placeholders is S4 §5-5 T1/T2, not yet done as of 2026-09-18:
  T1 (intrinsics) — front-left (D455): sim/calib_intrinsics_realsense.py reads the device.
                    wrist (Innomaker U20CAM-720P, opencv_uvc, D022 2026-09-13 — NOT a RealSense,
                    no device-readable intrinsics): sim/calib_intrinsics_checkerboard.py instead.
  T2 (extrinsics) — sim/calib_extrinsics_aruco.py, both cameras, from ArUco markers on the mat.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------------------
# Frame convention — the same one the placement mat uses
# --------------------------------------------------------------------------------------
# Origin  = the pan axis (joint1) of the arm, projected onto the table top.
# +X      = straight ahead, away from the operator.   (theta = 0 in experiment_spec §3)
# +Y      = to the operator's left.                   (theta > 0)
# +Z      = up.
# This matches `docs/assets/placement_label_map_*.csv` columns x_pan_cm / y_pan_cm exactly,
# which is the whole point: a sim placement and a real placement can share an id.

# --------------------------------------------------------------------------------------
# MEASURED
# --------------------------------------------------------------------------------------
R_INNER_M = 0.22            # S1 tape measure 2026-08-31 (17 cm + 5 cm d_offset)
R_OUTER_M = 0.41            # S1 tape measure 2026-08-31 (36 cm + 5 cm d_offset)
THETA_MIN_DEG = -90.0       # experiment_spec §3: camera-rig collision, not FOV
THETA_MAX_DEG = 45.0

# MEASURED (2026-09-18, read from configs/record_omx.yaml) — the two cameras do NOT share a
# resolution; a single shared CAM_WIDTH/CAM_HEIGHT here was a bug, not a simplification:
#   wrist       = Innomaker U20CAM-720P, opencv_uvc (D022 2026-09-13) -> 640x480
#                 (848 not supported under DSHOW, see configs/teleoperate_omx.yaml's own comment)
#   front-left  = RealSense D455, intelrealsense_pinned                -> 848x480
CAM_WIDTH_WRIST = 640
CAM_HEIGHT_WRIST = 480
CAM_WIDTH_FRONT_LEFT = 848
CAM_HEIGHT_FRONT_LEFT = 480
CAM_FPS = 15                # configs/record_omx.yaml (dataset fps must match; the UVC sensor sends 30, lerobot's record loop takes the newest frame)

# --------------------------------------------------------------------------------------
# PLACEHOLDER — every one of these replaces a blank row in experiment_spec §3
# --------------------------------------------------------------------------------------
TABLE_TOP_Z = 0.75          # PLACEHOLDER  桌面高度 ___ cm
TABLE_THICKNESS = 0.04           # PLACEHOLDER
# `[Eric說 2026-09-21]` "桌面僅須至少大於 placement ids 的工作範圍即可" -- so the table is DERIVED
# from what has to fit on it, not invented. Extent of campA_136sym's 108 points, pan-axis frame:
#   x 0.101 .. 0.383,  y -0.327 .. +0.343   (t1..t60 alone: x 0.105..0.367, y -0.315..+0.343)
# The table must also carry the riser, which reaches y = -0.385 and x = -0.139.
_PLACEMENT_X = (0.101, 0.383)
_PLACEMENT_Y = (-0.327, 0.343)
_TABLE_MARGIN = 0.10             # [AI推論] breathing room beyond the outermost thing on the table

# --------------------------------------------------------------------------------------
# Riser / bin / cup -- `[Eric說 2026-09-21]`, MEASURED unless marked otherwise
# --------------------------------------------------------------------------------------
# 🔴 THE ARM DOES NOT SIT ON THE TABLE. One platform 15 cm tall carries BOTH the arm base and
# the bin. Until 2026-09-21 the scene put the arm straight on the table, which is why a kinematic
# replay left the gripper ~11 cm above the object (S5 §2-C/§2-D).
ARM_RISER_HEIGHT = 0.15          # MEASURED

# Arm base plate footprint, read off `follower_01_base.stl` (URDF link0, scale 0.001):
# x -0.060..+0.060, y -0.075..+0.075, z 0..0.0575. The pan axis sits at x=-0.01125 in that frame,
# so in the PAN-AXIS frame (the one the placement mat uses) the plate spans:
ARM_BASE_FRONT_X = 0.060 + 0.01125     # +0.0713 m ahead of the pan axis
ARM_BASE_BACK_X = -0.060 + 0.01125     # -0.0488
ARM_BASE_HALF_Y = 0.075                # +/- from the pan axis

# `[Eric說]` the arm base's FRONT edge is flush with the riser's FRONT edge.
RISER_FRONT_X = ARM_BASE_FRONT_X

# Bin: a truncated cone. `[Eric說]` base dia 15 cm, opening dia 20 cm, height 22 cm, standing ON
# the riser, and its LEFT edge is 10 cm to the right of the arm base's RIGHT edge.
# (+Y is the operator's left, so "right" is -Y.)
BIN_BASE_DIA = 0.15              # MEASURED
BIN_OPENING_DIA = 0.20           # MEASURED
BIN_HEIGHT = 0.22                # MEASURED
BIN_GAP_FROM_ARM_BASE = 0.10     # MEASURED
_bin_max_r = BIN_OPENING_DIA / 2.0
BIN_CENTER_Y = -(ARM_BASE_HALF_Y + BIN_GAP_FROM_ARM_BASE + _bin_max_r)   # -0.275
# ⚠️ [AI推論] Eric specified the bin's SIDEWAYS offset only. Its X is assumed flush at the front
#    with the arm base and the riser -- tidy, and consistent with "front edges line up", but not
#    something he said. Move it if the real layout differs.
BIN_CENTER_X = RISER_FRONT_X - _bin_max_r

# the third-person camera also stands ON the riser, so its Y is needed to size the riser below
CAM_FRONT_LEFT_LEFT_OF_ARM = 0.20     # MEASURED: left of the arm base's left edge
CAM_FRONT_LEFT_Y_PRE = ARM_BASE_HALF_Y + CAM_FRONT_LEFT_LEFT_OF_ARM

# Riser footprint DERIVED to contain the arm base, the bin AND the third-person camera.
# `[Eric說 2026-09-21]` only the HEIGHT is measured; the footprint follows from what stands on it.
# 🔴 The FRONT edge takes NO margin -- `[Eric說]` the arm base plate and the bin are FLUSH with it.
#    A first version added margin on all four sides, which pushed the front edge 1 cm proud and
#    left both of them visibly short of it on the scene plan.
_RISER_MARGIN = 0.03            # back and far side only
_riser_back_x = min(ARM_BASE_BACK_X, BIN_CENTER_X - _bin_max_r) - _RISER_MARGIN
_riser_left_y = max(ARM_BASE_HALF_Y, CAM_FRONT_LEFT_Y_PRE) + _RISER_MARGIN
_riser_right_y = BIN_CENTER_Y - _bin_max_r - _RISER_MARGIN
ARM_RISER_SIZE = (RISER_FRONT_X - _riser_back_x, _riser_left_y - _riser_right_y, ARM_RISER_HEIGHT)
ARM_RISER_CENTER = ((RISER_FRONT_X + _riser_back_x) / 2.0, (_riser_left_y + _riser_right_y) / 2.0)

# Cup (the real manipulated object). `[Eric說 2026-09-21]` opening dia 7.5 cm, base dia 5 cm,
# height 9.5 cm, standing UPRIGHT on the TABLE. Replaces the `trash_obj` can, which was lying on
# its side with its centre only 4.1 cm up -- a different object in a different pose.
CUP_OPENING_DIA = 0.075          # MEASURED
CUP_BASE_DIA = 0.05              # MEASURED
CUP_HEIGHT = 0.095               # MEASURED
# ⚠️ Modelled as a CYLINDER at the mean diameter: Isaac Lab's primitives have no truncated cone,
#    and inventing a mesh would be a bigger fiction than a documented approximation. The taper is
#    NOT modelled; dimensions and upright pose are right.
CUP_MEAN_DIA = (CUP_OPENING_DIA + CUP_BASE_DIA) / 2.0

# Table box, derived: must cover every placement AND the whole riser, plus margin.
_t_min_x = min(_PLACEMENT_X[0], ARM_RISER_CENTER[0] - ARM_RISER_SIZE[0] / 2.0) - _TABLE_MARGIN
_t_max_x = max(_PLACEMENT_X[1], ARM_RISER_CENTER[0] + ARM_RISER_SIZE[0] / 2.0) + _TABLE_MARGIN
_t_min_y = min(_PLACEMENT_Y[0], ARM_RISER_CENTER[1] - ARM_RISER_SIZE[1] / 2.0) - _TABLE_MARGIN
_t_max_y = max(_PLACEMENT_Y[1], ARM_RISER_CENTER[1] + ARM_RISER_SIZE[1] / 2.0) + _TABLE_MARGIN
TABLE_SIZE = (_t_max_x - _t_min_x, _t_max_y - _t_min_y, TABLE_THICKNESS)
TABLE_CENTER_XY = ((_t_min_x + _t_max_x) / 2.0, (_t_min_y + _t_max_y) / 2.0)

# --------------------------------------------------------------------------------------
# MEASURED (of the asset files, not the real cell) — `assets/trash_obj/*.usd`
# --------------------------------------------------------------------------------------
# 🔴 Every trash_obj USD is authored with stage metersPerUnit=0.01 (its own coordinates are
#    centimetres), but Isaac Sim's world stage is metersPerUnit=1.0 (metres). USD reference
#    composition does NOT auto-rescale for a metersPerUnit mismatch between stages — the raw
#    numbers are taken as-is. Referencing one of these without this scale makes an 8 cm can
#    render as an 8-METRE object.
#
# [已查證 2026-09-03] checked via `sim/inspect_object_usd.py` (no simulation, just UsdGeom.BBoxCache)
# on all 8 objects in assets/trash_obj/: every one reports metersPerUnit=0.01, and the resulting
# real-world sizes are all physically plausible (banana 15x8x18cm, bottle_1 9x31x9cm, cans_1
# 7.7x16.6x8.1cm, ...). This is a property of the asset family, not a per-object guess.
#
# First-run evidence this matters: with scale=1.0, an object placed 6cm above the table
# (`preview_scene.py`, object USD trash_cans_1) settled at z=357.6cm after 120 physics steps —
# it was never "on the table", it exploded through it.
TRASH_OBJ_SCALE = (0.01, 0.01, 0.01)

# Third-person camera, "front-left" -- the name is from the OPERATOR's seat, see D022.
# `[Eric說 2026-09-21]` MEASURED, in the pan-axis frame (+X ahead, +Y operator-left):
#   * 20 cm to the LEFT of the arm base's left edge          -> y = ARM_BASE_HALF_Y + 0.20
#   * 4.5 cm inward (-X, toward the operator) from the riser's FRONT edge
#   * 11 cm high ABOVE THE RISER, i.e. 15 + 11 = 26 cm above the table top. `[Eric說 2026-09-21]`
#     corrected this: a first version placed it 11 cm above the TABLE, which put it below the
#     riser it actually stands on.
#   * aimed at the centre, 45 deg off the rightward horizontal, i.e. bearing -45 deg from +X
#   * pitched DOWN about 10 deg
CAM_FRONT_LEFT_INSET_FROM_RISER = 0.045   # MEASURED
CAM_FRONT_LEFT_Z = ARM_RISER_HEIGHT + 0.11   # MEASURED: 11 cm above the riser top
CAM_FRONT_LEFT_BEARING_DEG = -45.0    # MEASURED: from the rightward horizontal, turned to centre
CAM_FRONT_LEFT_PITCH_DEG = -10.0      # MEASURED: negative = looking down

# `[Eric說 2026-09-21]` the inset is -X (toward the operator), corrected after a first render put
# the camera a few cm the wrong side of the riser's front edge and inside the placement cloud.
CAM_FRONT_LEFT_POS = (
    RISER_FRONT_X - CAM_FRONT_LEFT_INSET_FROM_RISER,
    CAM_FRONT_LEFT_Y_PRE,
    CAM_FRONT_LEFT_Z,
)
# Look-at point: along the measured bearing, dropping at the measured pitch.
_cam_bearing = math.radians(CAM_FRONT_LEFT_BEARING_DEG)
_cam_pitch = math.radians(CAM_FRONT_LEFT_PITCH_DEG)
_CAM_LOOK_DIST = 0.50
CAM_FRONT_LEFT_LOOKAT = (
    CAM_FRONT_LEFT_POS[0] + _CAM_LOOK_DIST * math.cos(_cam_pitch) * math.cos(_cam_bearing),
    CAM_FRONT_LEFT_POS[1] + _CAM_LOOK_DIST * math.cos(_cam_pitch) * math.sin(_cam_bearing),
    CAM_FRONT_LEFT_Z + _CAM_LOOK_DIST * math.sin(_cam_pitch),
)

CAM_WRIST_PARENT_LINK = "link5"
CAM_WRIST_OFFSET_POS = (0.02, 0.0, 0.03)   # PLACEHOLDER  手腕相機安裝方式 ___
CAM_WRIST_OFFSET_ROT = (0.5, -0.5, 0.5, -0.5)  # PLACEHOLDER (ros convention)

DOME_LIGHT_INTENSITY = 1200.0   # PLACEHOLDER  光照強度 ___ lux

# --------------------------------------------------------------------------------------
# Intrinsics — PROVISIONAL, from datasheets. S4 §5-5 T1 says read them off the DEVICE.
# --------------------------------------------------------------------------------------
# 🔴 A datasheet FOV is the nominal design value. The per-unit intrinsics that pyrealsense2
#    reports are what actually determines where a 3-D point lands in the image. Do not treat
#    these as "the cameras are aligned" — they are "the scene can be rendered".
SENSOR_APERTURE_MM = 20.955     # Isaac Sim's standard 35 mm-equivalent horizontal aperture
HFOV_WRIST_DEG = 87.0           # PLACEHOLDER  Innomaker U20CAM-720P, unmeasured. Real value: sim/calib_intrinsics_checkerboard.py (T1)
HFOV_FRONT_LEFT_DEG = 90.0      # D455 datasheet RGB horizontal FOV        [PROVISIONAL]
CLIP_WRIST = (0.04, 2.0)        # PLACEHOLDER  Innomaker U20CAM-720P render clip range, unmeasured
CLIP_FRONT_LEFT = (0.10, 3.0)


def focal_length_mm(hfov_deg: float, aperture_mm: float = SENSOR_APERTURE_MM) -> float:
    """f = aperture / (2 tan(hfov/2)) — the conversion Isaac Sim's PinholeCameraCfg wants."""
    return aperture_mm / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


# --------------------------------------------------------------------------------------
# Seeded placements — the same frozen list the real campaign uses
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Placement:
    short_id: str
    placement_id: str
    x_m: float
    y_m: float

    @property
    def radius_m(self) -> float:
        return math.hypot(self.x_m, self.y_m)

    @property
    def theta_deg(self) -> float:
        return math.degrees(math.atan2(self.y_m, self.x_m))


def load_placements(csv_path: str | Path) -> list[Placement]:
    """Read `docs/assets/placement_label_map_<camp>.csv` (S2/S3 output).

    Using the SAME file as the printed mat is deliberate: a sim episode and a real episode can
    then carry the same `placement_id`, which is the only way the two datasets are comparable
    at all (D025 premise 2 forbids pooling them, not comparing them).
    """
    out: list[Placement] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(
                Placement(
                    short_id=row["short_id"],
                    placement_id=row["placement_id"],
                    x_m=float(row["x_pan_cm"]) / 100.0,
                    y_m=float(row["y_pan_cm"]) / 100.0,
                )
            )
    return out


def check_placement_reachable(p: Placement) -> list[str]:
    """Complain if a seeded point falls outside the measured workspace."""
    problems = []
    if not (R_INNER_M <= p.radius_m <= R_OUTER_M):
        problems.append(f"radius {p.radius_m*100:.1f} cm outside [{R_INNER_M*100:.0f}, {R_OUTER_M*100:.0f}]")
    if not (THETA_MIN_DEG <= p.theta_deg <= THETA_MAX_DEG):
        problems.append(f"theta {p.theta_deg:.1f} deg outside [{THETA_MIN_DEG:.0f}, {THETA_MAX_DEG:.0f}]")
    return problems


if __name__ == "__main__":
    import sys

    print(f"frame: origin = pan axis on the table top, +X ahead, +Y operator-left, +Z up")
    print(f"workspace: r {R_INNER_M*100:.0f}-{R_OUTER_M*100:.0f} cm, theta {THETA_MIN_DEG:.0f}..{THETA_MAX_DEG:.0f} deg")
    print(f"cameras: wrist {CAM_WIDTH_WRIST}x{CAM_HEIGHT_WRIST}, front-left {CAM_WIDTH_FRONT_LEFT}x{CAM_HEIGHT_FRONT_LEFT} @ {CAM_FPS} fps")
    print(f"  wrist       hfov {HFOV_WRIST_DEG} deg -> focal {focal_length_mm(HFOV_WRIST_DEG):.3f} mm  [PROVISIONAL]")
    print(f"  front-left  hfov {HFOV_FRONT_LEFT_DEG} deg -> focal {focal_length_mm(HFOV_FRONT_LEFT_DEG):.3f} mm  [PROVISIONAL]")
    if len(sys.argv) > 1:
        ps = load_placements(sys.argv[1])
        bad = [(p, check_placement_reachable(p)) for p in ps]
        bad = [(p, w) for p, w in bad if w]
        print(f"\nplacements: {len(ps)} loaded, {len(bad)} outside the measured workspace")
        for p, w in bad[:10]:
            print(f"  {p.short_id:<5}{p.placement_id:<14}r={p.radius_m*100:5.1f}cm theta={p.theta_deg:6.1f}deg  {'; '.join(w)}")

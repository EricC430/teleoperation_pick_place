"""Glue between the keyboard loop and the pure modules - kept out of
scripts/reach_logger.py so it can be tested without hardware or pynput.

Spec: docs/specs/S1_reach_logger.md 7-9.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from reach_logger import fk, plot
from reach_logger.joints import FK_MOTOR_ORDER, JointCalibration, joint_vector
from reach_logger.samples import SampleRow
from reach_logger.summary import FkValidation, ReachSummary, build_summary

KEY_TO_TYPE: dict[str, str] = {
    "o": "outer_topdown",
    "s": "outer_side",
    "i": "inner",
    "a": "azimuth_limit",
    "r": "reference",
}

_ALL_MOTORS = (*FK_MOTOR_ORDER, "gripper")


def make_sample_row(
    key: str,
    ticks: dict[str, float],
    calib: JointCalibration,
    *,
    method: str,
    now: str,
    tape_cm: float | None = None,
    mat_deg: float | None = None,
    note: str = "",
) -> SampleRow:
    sample_type = KEY_TO_TYPE[key]
    jv = joint_vector(ticks, calib)
    ex, ey, ez = (v * 100.0 for v in fk.ee_position_m(jv))
    az_base = fk.azimuth_base_deg(jv)

    if sample_type == "azimuth_limit":
        radius: float | None = None
    elif method == "tape":
        if tape_cm is None:
            raise ValueError("tape method needs a tape_cm reading")
        radius = tape_cm
    else:
        radius = fk.reach_cm(jv)

    return SampleRow(
        ts_iso=now,
        sample_type=sample_type,
        method=method,
        radius_cm=radius,
        ee_x_cm=ex,
        ee_y_cm=ey,
        ee_z_cm=ez,
        azimuth_base_deg=az_base,
        azimuth_mat_deg=mat_deg if sample_type == "reference" else None,
        shoulder_pan_pos=ticks["shoulder_pan"],
        shoulder_lift_pos=ticks["shoulder_lift"],
        elbow_flex_pos=ticks["elbow_flex"],
        wrist_flex_pos=ticks["wrist_flex"],
        wrist_roll_pos=ticks["wrist_roll"],
        gripper_pos=ticks["gripper"],
        note=note,
    )


def plan_text(*, config: str, urdf: str, out: str, mode: str) -> str:
    return "\n".join(
        [
            "reach_logger --dry-run — nothing below is opened or moved:",
            f"  mode        : {mode}",
            f"  robot config: {config}",
            f"  URDF (FK)   : {urdf}",
            f"  CSV out     : {out}",
            "  hardware    : NOT contacted (no serial port opened)",
        ]
    )


@dataclass(frozen=True)
class FinalizeResult:
    summary: ReachSummary
    summary_json_path: Path
    plot_path: Path


def _sibling(csv_path: Path, token: str, suffix: str) -> Path:
    name = csv_path.name.replace("reach_log", token, 1)
    return csv_path.with_name(name).with_suffix(suffix)


def finalize(
    rows: Sequence[SampleRow],
    *,
    csv_path: str | Path,
    fk_validation: FkValidation | None,
    git_commit: str,
    now: str,
) -> FinalizeResult:
    csv_path = Path(csv_path)
    summary = build_summary(
        rows,
        fk_validation=fk_validation,
        source_csv=str(csv_path),
        git_commit=git_commit,
        generated=now,
    )

    json_path = _sibling(csv_path, "reach_summary", ".json")
    json_path.write_text(
        json.dumps(summary.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    plot_path = _sibling(csv_path, "reach_plot", ".png")
    plot.save_plot(rows, summary, plot_path)

    return FinalizeResult(summary=summary, summary_json_path=json_path, plot_path=plot_path)

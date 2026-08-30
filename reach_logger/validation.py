"""5-pose FK validation gate (docs/specs/S1_reach_logger.md 5).

For a handful of known poses (home, then one joint moved at a time, plus a
gripper open/close), the operator reads the end-effector's (x, y) off the table
the same way they will for real samples. We compare that against the FK
prediction. If the worst pose is off by more than the tolerance and it is not a
fixable constant offset, the reach logger drops to the tape-measure fallback
(D026 reverse-if).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from reach_logger.summary import FkValidation


@dataclass(frozen=True)
class PoseCheck:
    name: str
    joint_rad: list[float]
    predicted_xy_cm: tuple[float, float]
    measured_xy_cm: tuple[float, float] | None

    @property
    def error_cm(self) -> float | None:
        if self.measured_xy_cm is None:
            return None
        px, py = self.predicted_xy_cm
        mx, my = self.measured_xy_cm
        return math.hypot(px - mx, py - my)


@dataclass(frozen=True)
class ValidationReport:
    checks: list[PoseCheck]
    result: FkValidation
    incomplete: bool
    tolerance_cm: float

    def render_table(self) -> str:
        head = f"{'pose':<8} {'predicted (x,y)cm':>20} {'measured (x,y)cm':>20} {'error cm':>10}  ok"
        lines = [head, "-" * len(head)]
        for c in self.checks:
            pred = f"({c.predicted_xy_cm[0]:.1f}, {c.predicted_xy_cm[1]:.1f})"
            if c.measured_xy_cm is None:
                meas, err, ok = "—", "—", "?"
            else:
                meas = f"({c.measured_xy_cm[0]:.1f}, {c.measured_xy_cm[1]:.1f})"
                err = f"{c.error_cm:.1f}"
                ok = "Y" if c.error_cm <= self.tolerance_cm else "N"
            lines.append(f"{c.name:<8} {pred:>20} {meas:>20} {err:>10}  {ok}")
        verdict = "PASS" if self.result.passed else ("INCOMPLETE" if self.incomplete else "FAIL")
        lines.append("")
        lines.append(f"tolerance {self.tolerance_cm:.1f} cm  ->  {verdict}")
        return "\n".join(lines)


def evaluate(
    checks: Sequence[PoseCheck], *, tolerance_cm: float, date: str
) -> ValidationReport:
    if not checks:
        raise ValueError("need at least the home pose")

    errors = [c.error_cm for c in checks]
    incomplete = any(e is None for e in errors)
    present = [e for e in errors if e is not None]
    max_err = max(present) if present else 0.0
    passed = (not incomplete) and max_err <= tolerance_cm

    result = FkValidation(passed=passed, date=date, max_error_cm=max_err)
    return ValidationReport(
        checks=list(checks),
        result=result,
        incomplete=incomplete,
        tolerance_cm=tolerance_cm,
    )

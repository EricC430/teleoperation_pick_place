"""The reach-log CSV: fixed schema, flushed on every append, never overwritten.

Schema and rules: docs/specs/S1_reach_logger.md 8, 10.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

COLUMNS: tuple[str, ...] = (
    "ts_iso",
    "sample_type",
    "method",
    "radius_cm",
    "ee_x_cm",
    "ee_y_cm",
    "ee_z_cm",
    "azimuth_base_deg",
    "azimuth_mat_deg",
    "shoulder_pan_pos",
    "shoulder_lift_pos",
    "elbow_flex_pos",
    "wrist_flex_pos",
    "wrist_roll_pos",
    "gripper_pos",
    "note",
)

SAMPLE_TYPES = frozenset(
    {"outer_topdown", "outer_side", "inner", "azimuth_limit", "reference"}
)
METHODS = frozenset({"fk", "tape"})


@dataclass
class SampleRow:
    ts_iso: str
    sample_type: str
    method: str
    radius_cm: float
    ee_x_cm: float
    ee_y_cm: float
    ee_z_cm: float
    azimuth_base_deg: float
    azimuth_mat_deg: float | None
    shoulder_pan_pos: float
    shoulder_lift_pos: float
    elbow_flex_pos: float
    wrist_flex_pos: float
    wrist_roll_pos: float
    gripper_pos: float
    note: str

    def __post_init__(self) -> None:
        if self.sample_type not in SAMPLE_TYPES:
            raise ValueError(
                f"sample_type {self.sample_type!r} not in {sorted(SAMPLE_TYPES)}"
            )
        if self.method not in METHODS:
            raise ValueError(f"method {self.method!r} not in {sorted(METHODS)}")

    def as_record(self) -> list[object]:
        out: list[object] = []
        for col in COLUMNS:
            value = getattr(self, col)
            out.append("" if value is None else value)
        return out


def resolve_out_path(path: str | Path) -> Path:
    """Return ``path`` if free, else ``stem_2.suffix``, ``stem_3.suffix``, ..."""
    path = Path(path)
    if not path.exists():
        return path
    n = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


class SampleWriter:
    """Context manager. Writes the header on open, flushes after every append."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = None
        self._writer = None

    def __enter__(self) -> "SampleWriter":
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(COLUMNS)
        self._fh.flush()
        return self

    def append(self, row: SampleRow) -> None:
        if self._writer is None or self._fh is None:
            raise RuntimeError("SampleWriter is not open")
        self._writer.writerow(row.as_record())
        self._fh.flush()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None

"""Read S1's reach_summary_<date>.json into a margined Sector.

Spec: docs/specs/S2_placement_sampler.md 3 (--from-summary).

S1 reports RAW measured bounds; the margin is applied here, never by S1
(D023 2026-08-31). r_outer_topdown_cm is the raw outer bound.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from placement_sampler.geometry import Sector, apply_margin


@dataclass
class SummaryInfo:
    source_path: str
    source_git_commit: str | None
    azimuth_frame: str
    fk_validation: dict | None
    warnings: list[str] = field(default_factory=list)


def _require(data: dict, key: str) -> float:
    value = data.get(key)
    if value is None:
        raise ValueError(f"reach summary is missing required field {key!r}")
    return float(value)


def sector_from_summary(path: str | Path, *, margin: float) -> tuple[Sector, SummaryInfo]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    raw = Sector(
        r_inner=_require(data, "r_inner_cm"),
        r_outer=_require(data, "r_outer_topdown_cm"),
        theta_min=_require(data, "azimuth_min_deg"),
        theta_max=_require(data, "azimuth_max_deg"),
    )
    sector = apply_margin(raw, margin)

    info = SummaryInfo(
        source_path=str(path),
        source_git_commit=data.get("git_commit"),
        azimuth_frame=data.get("azimuth_frame", "base"),
        fk_validation=data.get("fk_validation"),
    )

    fk = data.get("fk_validation")
    if fk is None:
        info.warnings.append(
            "fk_validation is null: radii come from an unvalidated FK chain (or tape fallback); "
            "precision is lower."
        )
    elif not fk.get("passed", False):
        info.warnings.append(
            "fk_validation.passed is false: radii should be treated as the tape-measure fallback."
        )

    if info.azimuth_frame != "mat":
        info.warnings.append(
            f"azimuth_frame is {info.azimuth_frame!r}, not 'mat': theta values are base-frame "
            "relative. S3 will refuse to print a production mat until an S1 reference sample exists."
        )

    return sector, info

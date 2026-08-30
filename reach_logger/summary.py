"""Reach-log rows -> the end-of-run terminal summary and reach_summary_<date>.json.

Spec: docs/specs/S1_reach_logger.md 9. The margin is NEVER applied here - the
summary reports the raw measured bounds and the human subtracts a margin by hand
(D023 2026-08-31).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from reach_logger.samples import SAMPLE_TYPES, SampleRow

FK_METHOD = "handcoded_urdf"
_NOTES = "margin NOT applied; human decides (D023 2026-08-31)"
_COUNT_ORDER = ("outer_topdown", "outer_side", "inner", "azimuth_limit", "reference")

assert set(_COUNT_ORDER) == set(SAMPLE_TYPES)


@dataclass(frozen=True)
class FkValidation:
    passed: bool
    date: str
    max_error_cm: float

    def as_dict(self) -> dict[str, object]:
        return {"passed": self.passed, "date": self.date, "max_error_cm": self.max_error_cm}


@dataclass(frozen=True)
class ReachSummary:
    generated: str
    source_csv: str
    git_commit: str
    fk_validation: FkValidation | None
    sample_counts: dict[str, int]
    azimuth_frame: str
    azimuth_offset_deg: float | None
    r_outer_topdown_cm: float | None
    r_outer_topdown_worst_azimuth_deg: float | None
    r_outer_side_cm: float | None
    r_inner_cm: float | None
    azimuth_min_deg: float | None
    azimuth_max_deg: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "generated": self.generated,
            "source_csv": self.source_csv,
            "git_commit": self.git_commit,
            "fk_method": FK_METHOD,
            "fk_validation": self.fk_validation.as_dict() if self.fk_validation else None,
            "azimuth_frame": self.azimuth_frame,
            "azimuth_offset_deg": self.azimuth_offset_deg,
            "r_outer_topdown_cm": self.r_outer_topdown_cm,
            "r_outer_topdown_worst_azimuth_deg": self.r_outer_topdown_worst_azimuth_deg,
            "r_outer_side_cm": self.r_outer_side_cm,
            "r_inner_cm": self.r_inner_cm,
            "azimuth_min_deg": self.azimuth_min_deg,
            "azimuth_max_deg": self.azimuth_max_deg,
            "sample_counts": dict(self.sample_counts),
            "notes": _NOTES,
        }

    def render_text(self) -> str:
        c = self.sample_counts
        if self.fk_validation is None:
            fk_line = "FK 驗證: 未做 -> radius 若來自捲尺請看 CSV 的 method 欄"
        else:
            v = self.fk_validation
            state = "通過" if v.passed else "未通過 (radius 應改用捲尺)"
            fk_line = f"FK 驗證: {state} ({v.date}, 最大誤差 {v.max_error_cm:.2f} cm)"

        if self.azimuth_frame == "mat":
            frame_line = f"方位角框架: base + reference offset = {self.azimuth_offset_deg:+.1f} deg  -> 以下為墊子框架"
        else:
            frame_line = "方位角框架: base (未對到墊子) -> 以下角度只是相對值"

        def cm(x: float | None) -> str:
            return "—" if x is None else f"{x:.1f} cm"

        def deg(x: float | None) -> str:
            return "—" if x is None else f"{x:.1f} deg"

        az_range = (
            f"{deg(self.azimuth_min_deg)} ~ {deg(self.azimuth_max_deg)}"
            if self.azimuth_min_deg is not None
            else "— (無 azimuth_limit 樣本)"
        )
        worst_az = (
            f"  @ azimuth {deg(self.r_outer_topdown_worst_azimuth_deg)}"
            if self.r_outer_topdown_worst_azimuth_deg is not None
            else ""
        )
        r_outer_line = (
            f"r_outer − <margin>"
            if self.r_outer_topdown_cm is None
            else f"r_outer = {self.r_outer_topdown_cm:.1f} − <margin>"
        )

        return "\n".join(
            [
                f"樣本數: outer_topdown={c['outer_topdown']} outer_side={c['outer_side']} "
                f"inner={c['inner']} azimuth_limit={c['azimuth_limit']}   "
                f"(reference: {'有' if c['reference'] else '無'})",
                fk_line,
                frame_line,
                "",
                f"受限方位角 (cable):        {az_range}",
                f"r_outer (top-down, 取最小): {cm(self.r_outer_topdown_cm)}{worst_az}",
                f"r_outer (side-only, 供參):  {cm(self.r_outer_side_cm)}",
                f"r_inner (可下爪):           {cm(self.r_inner_cm)}",
                "",
                "-> 建議填入 experiment_spec §3（人看過再手動填）:",
                f"   {r_outer_line}       ⚠️ margin 由人決定，腳本不自動減",
                f"   r_inner = {cm(self.r_inner_cm)}",
                f"   azimuth = [{deg(self.azimuth_min_deg)}, {deg(self.azimuth_max_deg)}]  "
                f"({'墊子' if self.azimuth_frame == 'mat' else 'base'}框架)",
                "⚠️ D023(2026-08-31): 33/43 是抓取進場角差別，不是環帶；r_outer 用 top-down 值。",
            ]
        )


def _radii(rows: Sequence[SampleRow], sample_type: str) -> list[SampleRow]:
    return [r for r in rows if r.sample_type == sample_type and r.radius_cm is not None]


def build_summary(
    rows: Sequence[SampleRow],
    *,
    fk_validation: FkValidation | None,
    source_csv: str,
    git_commit: str,
    generated: str,
) -> ReachSummary:
    counts = {t: 0 for t in _COUNT_ORDER}
    for r in rows:
        counts[r.sample_type] += 1

    references = [
        r for r in rows if r.sample_type == "reference" and r.azimuth_mat_deg is not None
    ]
    if references:
        ref = references[0]
        offset: float | None = ref.azimuth_mat_deg - ref.azimuth_base_deg
        frame = "mat"
    else:
        offset = None
        frame = "base"

    shift = offset or 0.0

    def az(row: SampleRow) -> float:
        return row.azimuth_base_deg + shift

    limits = [az(r) for r in rows if r.sample_type == "azimuth_limit"]
    azimuth_min = min(limits) if limits else None
    azimuth_max = max(limits) if limits else None

    topdown = _radii(rows, "outer_topdown")
    if topdown:
        worst = min(topdown, key=lambda r: r.radius_cm)
        r_outer_topdown_cm: float | None = worst.radius_cm
        r_outer_topdown_worst_azimuth_deg: float | None = az(worst)
    else:
        r_outer_topdown_cm = None
        r_outer_topdown_worst_azimuth_deg = None

    sides = [r.radius_cm for r in _radii(rows, "outer_side")]
    inners = [r.radius_cm for r in _radii(rows, "inner")]

    return ReachSummary(
        generated=generated,
        source_csv=source_csv,
        git_commit=git_commit,
        fk_validation=fk_validation,
        sample_counts=counts,
        azimuth_frame=frame,
        azimuth_offset_deg=offset,
        r_outer_topdown_cm=r_outer_topdown_cm,
        r_outer_topdown_worst_azimuth_deg=r_outer_topdown_worst_azimuth_deg,
        r_outer_side_cm=max(sides) if sides else None,
        r_inner_cm=max(inners) if inners else None,
        azimuth_min_deg=azimuth_min,
        azimuth_max_deg=azimuth_max,
    )

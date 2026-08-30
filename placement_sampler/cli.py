"""`uv run scripts/sample_placements.py` -- argument wiring and orchestration.

Spec: docs/specs/S2_placement_sampler.md 3, 4, 6.

Exit codes:
  0  lists written (or --dry-run on a feasible config)
  1  refused to overwrite frozen output (no --force)
  2  bad arguments, or the config is infeasible
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import date
from pathlib import Path

from placement_sampler.feasibility import feasibility, feasibility_table
from placement_sampler.geometry import Sector, apply_margin
from placement_sampler.reach_summary import sector_from_summary
from placement_sampler.sampler import SamplingStuck, sample_lists
from placement_sampler.stats import ks_uniform_pvalue
from placement_sampler.writer import OutputExists, write_outputs

_MANUAL_BOUNDS = ("r_inner", "r_outer", "theta_min", "theta_max")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sample_placements.py",
        description="Sample the three frozen placement lists over an annular sector (S2, D023/D024).",
    )
    src = p.add_argument_group("workspace (give --from-summary OR the four manual bounds)")
    src.add_argument("--from-summary", type=Path, help="S1 reach_summary_<date>.json")
    src.add_argument("--r-inner", type=float, help="cm")
    src.add_argument("--r-outer", type=float, help="cm (raw; --margin is subtracted here)")
    src.add_argument("--theta-min", type=float, help="deg, mat frame")
    src.add_argument("--theta-max", type=float, help="deg, mat frame")

    p.add_argument(
        "--margin",
        type=float,
        default=None,
        help="cm; subtracted from r_outer and both azimuth edges. Required with --from-summary.",
    )
    p.add_argument("--n-train", type=int, default=50)
    p.add_argument("--n-eval-open", type=int, default=10)
    p.add_argument("--n-eval-close", type=int, default=30)
    p.add_argument(
        "--d-min",
        type=float,
        required=True,
        help="cm; minimum spacing between every pair of points. No default (D024 7).",
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--date", default=date.today().strftime("%Y%m%d"), help="YYYYMMDD, for filenames")
    p.add_argument("--out-dir", type=Path, default=Path("configs/placements/"))
    p.add_argument("--label", required=True, help="campaign label, goes in the filename")
    p.add_argument("--dry-run", action="store_true", help="feasibility + stats only, write nothing")
    p.add_argument("--force", action="store_true", help="allow overwriting frozen output files")
    p.add_argument(
        "--plot",
        action="store_true",
        help="also write <label>_<date>_scatter.png (diagnostic; not the S3 mat)",
    )
    return p


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - env dependent
        return None


def _resolve_sector(args: argparse.Namespace, parser: argparse.ArgumentParser):
    manual = {name: getattr(args, name) for name in _MANUAL_BOUNDS}
    has_manual = any(v is not None for v in manual.values())

    if args.from_summary is not None:
        if has_manual:
            parser.error("--from-summary is mutually exclusive with --r-inner/--r-outer/--theta-*")
        if args.margin is None:
            parser.error("--margin is required with --from-summary (S1 reports raw bounds)")
        sector, info = sector_from_summary(args.from_summary, margin=args.margin)
        return sector, info, "from_summary"

    if not all(v is not None for v in manual.values()):
        parser.error("give all four of --r-inner --r-outer --theta-min --theta-max, or --from-summary")
    raw = Sector(
        r_inner=manual["r_inner"],
        r_outer=manual["r_outer"],
        theta_min=manual["theta_min"],
        theta_max=manual["theta_max"],
    )
    sector = apply_margin(raw, args.margin) if args.margin else raw
    return sector, None, "manual"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        sector, info, source = _resolve_sector(args, parser)
    except ValueError as exc:
        parser.error(str(exc))

    for w in getattr(info, "warnings", []) or []:
        print(f"WARNING: {w}")

    counts = {
        "train": args.n_train,
        "eval-open": args.n_eval_open,
        "eval-close": args.n_eval_close,
    }
    n_total = sum(counts.values())
    feas = feasibility(sector, d_min=args.d_min, n_total=n_total)

    print("feasibility:")
    print(feasibility_table(feas))
    print(
        f"sector: r_inner={sector.r_inner:.2f} r_outer={sector.r_outer:.2f} "
        f"theta=[{sector.theta_min:.2f}, {sector.theta_max:.2f}] deg"
    )

    if not feas.verdict.is_feasible:
        print("ABORT: infeasible config, nothing sampled.")
        return 2

    if args.dry_run:
        print("--dry-run: not sampling, not writing.")
        return 0

    try:
        lists = sample_lists(sector, d_min=args.d_min, counts=counts, seed=args.seed)
    except SamplingStuck as exc:
        print(f"ABORT: {exc}")
        return 2

    lo, hi = sector.r_inner**2, sector.r_outer**2
    ks = {
        name: ks_uniform_pvalue([p.r_cm**2 for p in pts], lo, hi)
        for name, pts in lists.items()
        if pts
    }
    for name, p in ks.items():
        flag = "" if p > 0.01 else "  <-- LOW, r^2 may not be uniform"
        print(f"  {name:11s} r^2 uniformity KS p = {p:.4f}{flag}")

    meta = {
        "seed": args.seed,
        "d_min_cm": args.d_min,
        "margin_cm": args.margin,
        "counts": counts,
        "r2_uniform_ks_p": ks,
        "sampling_source": source,
        "code_git_commit": _git_commit(),
        "sector_used": {
            "r_inner": sector.r_inner,
            "r_outer": sector.r_outer,
            "theta_min": sector.theta_min,
            "theta_max": sector.theta_max,
        },
        "feasibility": {
            "verdict": feas.verdict.value,
            "n_hex": round(feas.n_hex, 1),
            "n_rsa": round(feas.n_rsa, 1),
        },
    }
    if info is not None:
        meta["from_summary"] = {
            "path": info.source_path,
            "git_commit": info.source_git_commit,
            "azimuth_frame": info.azimuth_frame,
            "fk_validation": info.fk_validation,
        }

    try:
        paths = write_outputs(args.out_dir, args.label, args.date, lists, meta, force=args.force)
    except OutputExists as exc:
        print(f"REFUSED: {exc}")
        return 1

    for p in paths:
        print(f"wrote {p}")

    if args.plot:
        from placement_sampler.plot import save_scatter

        png = args.out_dir / f"{args.label}_{args.date}_scatter.png"
        save_scatter(lists, sector, png, seed=args.seed, d_min=args.d_min)
        print(f"wrote {png}")

    return 0

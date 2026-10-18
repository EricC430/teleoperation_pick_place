"""Compare closed-loop runs recorded in episode_meta/ (success by radius zone, mechanisms, paired McNemar).

Written 2026-10-13 to back-fill three closed-loop runs that had a CSV but no record:
docs/worklog/2026-09-21_bottle_closed_loop.md, 2026-10-07_temporal_ensemble.md, 2026-10-08_C_n30_home.md.

- valid: a row counts unless valid == "0" (schema default is "1", configs/episode_meta_schema.yaml).
- Zones use r = hypot(x_pan_cm, y_pan_cm) from docs/assets/placement_label_map_campA_136sym_20260908.csv,
  cut at 22 / 33 cm, the same cut as docs/worklog/2026-10-08.md section 2.
- Paired comparison uses only placements valid in BOTH runs; McNemar is exact two-sided on the discordant pairs.

Usage:
    python scripts/compare_closed_loop.py                      # summary of every rollout CSV + the bottle run
    python scripts/compare_closed_loop.py A.csv B.csv          # paired comparison of two episode_meta CSVs
"""
from __future__ import annotations

import csv
import glob
import math
import sys
from collections import Counter
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "episode_meta"
PLACEMENTS = ROOT / "docs/assets/placement_label_map_campA_136sym_20260908.csv"
ZONES = [("near<22", 0, 22), ("mid22-33", 22, 33), ("far>=33", 33, 1e9)]


def radius_by_short_id() -> dict[str, float]:
    with open(PLACEMENTS, encoding="utf-8-sig") as fh:
        return {r["short_id"]: math.hypot(float(r["x_pan_cm"]), float(r["y_pan_cm"])) for r in csv.DictReader(fh)}


def load(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8-sig") as fh:
        return {r["placement_id"]: r for r in csv.DictReader(fh) if (r.get("episode_index") or "").strip().isdigit()}


def valid(r: dict) -> bool:
    return r.get("valid", "") != "0"


def success(r: dict) -> bool:
    return r.get("outcome") == "success"


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def summary(path: Path, radius: dict[str, float]) -> None:
    rows = load(path)
    v = {p: r for p, r in rows.items() if valid(r)}
    print(f"\n## {path.name}\n  success {sum(map(success, v.values()))}/{len(v)}   void: {sorted(set(rows) - set(v))}")
    for name, lo, hi in ZONES:
        z = [r for p, r in v.items() if lo <= radius[p] < hi]
        print(f"  {name:9s} {sum(map(success, z))}/{len(z)}")
    mech = Counter()
    for r in v.values():
        tag = "(success) " if success(r) else ""
        for m in (r.get("mechanism") or "(blank)").split(";"):
            if success(r) and m == "(blank)":
                continue
            mech[tag + m] += 1
    print("  mechanisms:", dict(mech))


def paired(a_path: Path, b_path: Path) -> None:
    a, b = load(a_path), load(b_path)
    common = [p for p in a if p in b and valid(a[p]) and valid(b[p])]
    only_a = [p for p in common if success(a[p]) and not success(b[p])]
    only_b = [p for p in common if success(b[p]) and not success(a[p])]
    both = sum(success(a[p]) and success(b[p]) for p in common)
    print(f"\n## {a_path.name}\n   vs {b_path.name}\n  common {len(common)}: "
          f"{both + len(only_a)} vs {both + len(only_b)}, both {both}\n  only A {only_a}\n  only B {only_b}\n"
          f"  McNemar exact p = {mcnemar_exact(len(only_a), len(only_b)):.4f}")


def main() -> None:
    if len(sys.argv) == 3:
        paired(Path(sys.argv[1]), Path(sys.argv[2]))
        return
    radius = radius_by_short_id()
    files = sorted(META.glob("rollout_*.csv")) + [META / "omx_pick_place_pilot_plastic_bottle_closed_loop.csv"]
    for f in files:
        summary(f, radius)


if __name__ == "__main__":
    main()

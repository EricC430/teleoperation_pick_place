#!/usr/bin/env python3
"""S3 — Placement mat.  Spec: docs/specs/S3_placement_mat.md  (D023 script 3).

Build an A4-tiled Cartesian + polar coordinate mat with graduations, so a
sampled (x, y) can be hand-marked onto paper once and reused every episode.

    uv run python scripts/make_placement_mat.py --help
    uv run python scripts/make_placement_mat.py --r-max 40 --dry-run
    uv run python scripts/make_placement_mat.py --r-max 40 \\
        --out docs/assets/placement_mat_blank.pdf
    uv run python scripts/make_placement_mat.py --r-max 40 \\
        --placements configs/placements/camp_*_train.csv ... \\
        --from-summary analysis/reach_summary_2026-09-05.json \\
        --out docs/assets/placement_mat_campA.pdf --label campA
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from placement_mat.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

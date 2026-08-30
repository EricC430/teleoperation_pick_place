#!/usr/bin/env python3
"""S2 — Object-placement sampler.  Spec: docs/specs/S2_placement_sampler.md  (D023 script 2, D024).

Draw the three frozen placement lists once, over an annular sector, with a
global minimum spacing:

    train       50 points   train_NNN
    eval-open   10 points   eval-open_NNN   (open-loop eval, part of the 60-episode split)
    eval-close  30 points   eval-close_NNN  (closed-loop 30, shared across all 3 objects)

Every pair of points, across all three lists, is >= --d-min apart. Sampling is
seeded and frozen: a re-run with the same seed is byte-identical, and existing
output files are never overwritten without --force.

    uv run python scripts/sample_placements.py --help
    uv run python scripts/sample_placements.py \\
        --r-inner 20 --r-outer 30 --theta-min -35 --theta-max 88 \\
        --d-min 2.0 --seed 20260831 --label campaign_A_pilot_2cam --dry-run
    uv run python scripts/sample_placements.py \\
        --from-summary analysis/reach_summary_2026-09-05.json --margin 3.0 \\
        --d-min 2.0 --seed 20260831 --label campaign_A_pilot_2cam
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from placement_sampler.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

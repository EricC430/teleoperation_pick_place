"""Short point labels + the short_id <-> placement_id map for the single-page mat.

Spec: docs/specs/S3_placement_mat.md 3. The A4-tiled mat annotates each point
with its full frozen id (``train_007``); the print-shop single-page mat is hand
lettered, so it uses ``t7`` and ships a CSV that ties the two together (and
records the datum shift used).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_PREFIX_TO_LETTER = {"train": "t", "eval-open": "o", "eval-close": "c"}
_TRAILING_INT = re.compile(r"_(\d+)$")


def short_label(placement_id: str) -> str:
    """``train_007`` -> ``t7``; unknown prefixes are returned unchanged."""
    m = _TRAILING_INT.search(placement_id)
    if m is None:
        return placement_id
    prefix = placement_id[: m.start()]
    letter = _PREFIX_TO_LETTER.get(prefix)
    if letter is None:
        return placement_id
    return f"{letter}{int(m.group(1))}"


def label_map_rows(
    points: Sequence[tuple[str, float, float]], *, datum_offset: float
) -> list[dict]:
    """One row per point: short id, frozen id, and the coord in both frames.

    ``points`` are (placement_id, x_cm, y_cm) in the pan-axis frame (S2 output).
    ``x_mat_cm = x_pan_cm - datum_offset`` places them relative to the mat datum
    (the arm-tips registration line).
    """
    return [
        {
            "short_id": short_label(pid),
            "placement_id": pid,
            "x_pan_cm": x,
            "y_pan_cm": y,
            "x_mat_cm": round(x - datum_offset, 6),
            "y_mat_cm": y,
        }
        for pid, x, y in points
    ]

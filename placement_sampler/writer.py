"""Freeze the three lists to disk: dated CSVs + one meta.json.

Spec: docs/specs/S2_placement_sampler.md 5, 6.

CSV bytes are deterministic (fixed '\n', 2 decimals) so a re-run with the same
seed is byte-identical. meta.json is not required to be byte-identical -- it
carries a wall-clock timestamp.
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from placement_sampler.sampler import Placement, global_min_separation_cm

_HEADER = "placement_id,r_cm,theta_deg,x_cm,y_cm"


class OutputExists(FileExistsError):
    """A target file already exists and force was not given (spec 2-2: frozen, write once)."""


def _csv_text(points: list[Placement]) -> str:
    lines = [_HEADER]
    for p in points:
        lines.append(
            f"{p.placement_id},{p.r_cm:.2f},{p.theta_deg:.2f},{p.x_cm:.2f},{p.y_cm:.2f}"
        )
    return "\n".join(lines) + "\n"


def _nearest_neighbour_stats(lists: dict[str, list[Placement]]) -> dict[str, dict]:
    all_xy = [(p.x_cm, p.y_cm) for pts in lists.values() for p in pts]
    out: dict[str, dict] = {}
    for name, points in lists.items():
        nn = []
        for p in points:
            dists = [
                math.dist((p.x_cm, p.y_cm), q) for q in all_xy if q != (p.x_cm, p.y_cm)
            ]
            if dists:
                nn.append(min(dists))
        out[name] = {
            "n": len(points),
            "nearest_neighbour_cm": {
                "min": round(min(nn), 3) if nn else None,
                "mean": round(statistics.fmean(nn), 3) if nn else None,
                "median": round(statistics.median(nn), 3) if nn else None,
            },
        }
    return out


def write_outputs(
    out_dir: str | Path,
    label: str,
    date: str,
    lists: dict[str, list[Placement]],
    meta: dict,
    *,
    force: bool = False,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{label}_{date}"
    csv_paths = {name: out_dir / f"{stem}_{name}.csv" for name in lists}
    meta_path = out_dir / f"{stem}_meta.json"
    targets = [*csv_paths.values(), meta_path]

    if not force:
        clash = [p for p in targets if p.exists()]
        if clash:
            raise OutputExists(
                "refusing to overwrite frozen output(s): "
                + ", ".join(p.name for p in clash)
                + " (pass force=True only if you really mean to re-freeze)"
            )

    for name, path in csv_paths.items():
        path.write_text(_csv_text(lists[name]), encoding="utf-8", newline="")

    full_meta = dict(meta)
    full_meta["generated"] = datetime.now(timezone.utc).isoformat()
    full_meta["global_min_separation_cm"] = round(global_min_separation_cm(lists), 3)
    per_list = _nearest_neighbour_stats(lists)
    for name, p in full_meta.pop("r2_uniform_ks_p", {}).items():
        per_list[name]["r2_uniform_ks_p"] = round(p, 4)
    full_meta["per_list"] = per_list
    meta_path.write_text(
        json.dumps(full_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return targets

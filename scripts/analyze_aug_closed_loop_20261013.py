"""Closed-loop ntfA1aug / ntcfA1aug (2026-10-13) vs the 10-08 ntA1 run C: success and wrist turn by distance band.

turn = max observation.state wrist_roll over the episode (> 15 = turned; threshold from h6_turn_cue.py).
r = hypot(x_pan, y_pan) of the placement (docs/assets/placement_label_map_campA_136sym_20260908.csv); far = r >= 35.5.
Paired test: exact two-sided sign test on the placements where the two runs disagree.
Results and reading: docs/worklog/2026-10-13_closed_loop.md

    uv run python scripts/analyze_aug_closed_loop_20261013.py
"""
import glob
from math import comb

import numpy as np
import pandas as pd

ROLL = "D:/hf/lerobot/ericc430/"
RUNS = {  # name: (rollout dataset, episode_meta csv)
    "C ntA1 10-08": ("rollout_omx_b1_papercup_60_16_12rcvry_100k_paper_cup_20261008_192921",
                     "rollout_omx_b1_papercup_60_16_12rcvry_100k_nas30_paper_cup_20261008_192921.csv"),
    "ntfA1aug": ("rollout_omx_b1_papercup_60_13_10rcvry_46far_aug_100k_nas30_paper_cup_20261013_184921",
                 "rollout_omx_b1_papercup_60_13_10rcvry_46far_aug_100k_nas30_paper_cup_20261013_184921.csv"),
    "ntcfA1aug": ("rollout_omx_b1_papercup_60_13_10rcvrycut_46far_aug_100k_nas30_paper_cup_20261013_201040",
                  "rollout_omx_b1_papercup_60_13_10rcvrycut_46far_aug_100k_nas30_paper_cup_20261013_201040.csv"),
}
FAR, TURN = 35.5, 15.0


def load(ds: str, csv: str) -> pd.DataFrame:
    df = pd.concat(pd.read_parquet(p) for p in sorted(glob.glob(ROLL + ds + "/data/**/*.parquet", recursive=True)))
    df["roll"] = np.stack(df["observation.state"].values)[:, 4]
    m = pd.read_csv("episode_meta/" + csv, encoding="utf-8-sig")
    m["turn"] = m.episode_index.map(df.groupby("episode_index").roll.max())
    m["ok"] = m.outcome.eq("success")
    m["mechanism"] = m.mechanism.fillna("")
    return m.set_index("placement_id")


def sign_test(a: pd.DataFrame, b: pd.DataFrame) -> tuple[int, int, float]:
    w, l = int((a.ok & ~b.ok).sum()), int((~a.ok & b.ok).sum())
    return w, l, min(1.0, 2 * sum(comb(w + l, i) for i in range(min(w, l) + 1)) / 2 ** (w + l))


def main() -> None:
    lm = pd.read_csv("docs/assets/placement_label_map_campA_136sym_20260908.csv")
    r = pd.Series(np.hypot(lm.x_pan_cm, lm.y_pan_cm).values, index=lm.short_id)
    T = {k: load(*v) for k, v in RUNS.items()}
    for k, m in T.items():
        m["r"] = r.reindex(m.index).values
        fail = m[~m.ok]
        print(f"== {k}: {m.ok.sum()}/{len(m)} | chunk_gap_rush {m.mechanism.str.contains('chunk_gap_rush').sum()} "
              f"| failures with bad_aim {fail.mechanism.str.contains('bad_aim').sum()}/{len(fail)}")
        for far, g in m.groupby(m.r >= FAR):
            print(f"  {'r>=35.5' if far else 'r<35.5 '} success {g.ok.sum()}/{len(g)}  turn {(g.turn > TURN).sum()}/{len(g)}")
    names = list(T)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            w, l, p = sign_test(T[names[j]], T[names[i]])
            print(f"{names[j]} vs {names[i]}: wins {w}, losses {l}, sign test p = {p:.3f}")
    pd.set_option("display.width", 220)
    cols = {k: m[["turn", "outcome", "mechanism"]].add_suffix(f" {k}") for k, m in T.items()}
    print(pd.concat([T[names[0]][["r"]], *cols.values()], axis=1).sort_values("r").round(1).to_string())


if __name__ == "__main__":
    main()

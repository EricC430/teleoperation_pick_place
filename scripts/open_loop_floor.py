#!/usr/bin/env python
"""Reference predictors for open-loop MAE: how low does MAE get without a policy? (D031 §3, 2026-10-08 §7)

Same re-plan protocol as eval_open_loop.py --n_action_steps N: a prediction is made at s = 0, N, 2N, ...
and covers frames s..s+N-1. Predictors, all from ground-truth actions only:
  const    per-joint mean of the episode
  freeze   hold action[s] for N frames
  donor    another demonstration's shape anchored on the target: pred[s+j] = donor[s+j] - donor[s] + target[s]
           ("any" = median over all target/donor pairs; "oracle" = per target, the donor with the lowest MAE,
           chosen after the fact)

The oracle is NOT a fixed floor: it falls as the donor pool grows (best of 60 < best of 12) and depends on
how donor and target are aligned in time. Always report the pool and --align with it.

    bash scripts/run_container.sh python scripts/open_loop_floor.py \
      --eval-root data/huggingface/lerobot/ericc430/omx_pick_place_open_loop_eval \
      --donor-root data/huggingface/lerobot/ericc430/omx_pick_place_pilot_uvc_60 --donor-episodes 0 5 10 15 20 25 30 35 40 45 50 55

MAE is per-episode mean over frames x joints, then mean over episodes -- the same aggregation as
eval_open_loop.py's mean_mae. "pooled" (all frames together) is printed too because D031 §3's 12-episode
table used it (freeze 10.13 = pooled; §6's 10.79 = per-episode).
Units are LeRobot .pos, not degrees.
"""

import argparse
import glob

import numpy as np
import pandas as pd


def load_actions(root: str) -> dict:
    df = pd.concat(pd.read_parquet(p, columns=["episode_index", "frame_index", "action"])
                   for p in sorted(glob.glob(f"{root}/data/*/*.parquet")))
    df = df.sort_values(["episode_index", "frame_index"])
    return {int(e): np.stack(g.action.values).astype(np.float64) for e, g in df.groupby("episode_index")}


def freeze(t: np.ndarray, n: int) -> np.ndarray:
    p = np.empty_like(t)
    for s in range(0, len(t), n):
        p[s:s + n] = t[s]
    return p


def donor(t: np.ndarray, d: np.ndarray, n: int, align: str) -> np.ndarray:
    if align == "timenorm":  # resample the donor to the target's length
        x = np.linspace(0, len(d) - 1, len(t))
        d = np.stack([np.interp(x, np.arange(len(d)), d[:, j]) for j in range(d.shape[1])], 1)
    else:  # "hold": same frame index; a shorter donor holds its last frame
        d = np.concatenate([d, np.repeat(d[-1:], max(0, len(t) - len(d)), 0)])
    p = np.empty_like(t)
    for s in range(0, len(t), n):
        e = min(s + n, len(t))
        p[s:e] = d[s:e] - d[s] + t[s]
    return p


def summarise(errs: list) -> tuple:
    return float(np.mean([e.mean() for e in errs])), float(np.abs(np.concatenate(errs)).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--eval-exclude", type=int, nargs="*", default=[], help="e.g. failed demos (D031 §6)")
    ap.add_argument("--donor-root", default=None, help="training set used as a second donor pool")
    ap.add_argument("--donor-episodes", type=int, nargs="*", default=None, help="default: all")
    ap.add_argument("--donor-exclude", type=int, nargs="*", default=[])
    ap.add_argument("--n", type=int, nargs="+", default=[100, 50, 20], help="re-plan interval(s)")
    ap.add_argument("--align", choices=["hold", "timenorm"], default="hold")
    args = ap.parse_args()

    ev = {k: v for k, v in load_actions(args.eval_root).items() if k not in args.eval_exclude}
    pools = {"eval (others)": None}
    if args.donor_root:
        tr = load_actions(args.donor_root)
        keys = args.donor_episodes if args.donor_episodes is not None else sorted(tr)
        pools[f"donor root ({len([k for k in keys if k not in args.donor_exclude])} eps)"] = \
            [tr[k] for k in keys if k not in args.donor_exclude]
    print(f"eval: {args.eval_root}  episodes {sorted(ev)}  align={args.align}")
    if args.donor_root:
        print(f"donor root: {args.donor_root}")
    for n in args.n:
        print(f"\n== N={n}   (per-episode mean / pooled)")
        const = [np.abs(t - t.mean(0)) for t in ev.values()]
        print("  const                       %.2f / %.2f" % summarise(const))
        print("  freeze                      %.2f / %.2f" % summarise([np.abs(freeze(t, n) - t) for t in ev.values()]))
        for name, pool in pools.items():
            any_mae, best = [], []
            for k, t in ev.items():
                donors = [d for j, d in ev.items() if j != k] if pool is None else pool
                errs = [np.abs(donor(t, d, n, args.align) - t) for d in donors]
                any_mae += [e.mean() for e in errs]
                best.append(min(errs, key=lambda e: e.mean()))
            print(f"  any donor, median  [{name}]  {np.median(any_mae):.2f}")
            print(f"  oracle donor       [{name}]  %.2f / %.2f" % summarise(best))


if __name__ == "__main__":
    main()

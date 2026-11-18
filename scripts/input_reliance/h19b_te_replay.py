"""H19b: h19 again, but with temporal ensembling simulated offline, so the replayed gripper command is what the robot
would have executed (2026-11-18). h19's single-chunk read failed its sanity check (one chunk at onset - 1 s plans a
close in only 23/35 / 12/33 of the models' own episodes).

Method: every frame of every episode of both bottle closed loops (11/17 aimfix28, 11/15 M-full, v4 list) goes through
both models (one 100-step chunk per frame, recorded observations = open loop). The TE command at frame t is the
lerobot ACTTemporalEnsembler average of chunk_j[t - j] over j = max(0, t - 99)..t with w_k = exp(-coeff * k), k = 0 the
oldest, coeff = -0.01 (the rollout config). Also kept: the weighted std of those predictions for the gripper ("spread").
Open loop: a model replayed on the other model's rollout sees views produced by the other model's moves.

Readouts per (rollout, episode, model), gripper = action dim 5:
  onset    first frame where the TE gripper, after being >= 75, falls below 75 (h18's open threshold)
  dur      closing time: from the TE gripper first crossing g_open - 10 % to g_open - 90 % of (g_open - g_min),
           g_open = median of the 15 frames before onset, g_min = min of onset..onset + 90   [frames, / 15 = s]
  spread   median gripper spread over onset - 15 .. onset + dur

Reading rules, fixed before the run:
  sanity  own-model replay (aimfix on 11/17, mfull on 11/15) vs the recorded action: onset within +-5 frames in >= 80 %
          of the episodes with a recorded close, and no onset where none was recorded. Fail -> stop, no reading.
  A       11/17 ep27 (c32, stall f135-672): mfull's TE onset exists inside f135-672 (aimfix's must not: sanity), AND on
          11/15 ep27 aimfix's onset is missing or > 15 frames later than mfull's -> A for. mfull no onset on 11/17 ep27
          -> A against (the frames, not the aimfix data). Otherwise mixed.
  B       paired over episodes where both models have an onset (both rollouts pooled, also per rollout):
          aimfix dur > mfull dur in most episodes, exact sign test p < 0.05 -> aimfix closes slower on the same views.
          Mechanism "TE averages conflicting predictions": aimfix spread > mfull spread, same test. Slower without larger
          spread -> the chunks themselves close slowly, not the averaging.
Frames of one episode are not independent; episodes are the unit.

    docker exec -w /workspace lerobot-dev python scripts/input_reliance/h19b_te_replay.py
Writes data/logs/input_reliance/h19b_*.csv / .npz only.
"""
import sys
import time
from math import comb

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts/input_reliance")
import h19_c32_stall_close as H  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

COEFF, CH, BS = -0.01, 100, 32
OWN = {"11-17 aimfix": "aimfix", "11-15 mfull": "mfull"}


def te(chunks):
    """chunks (T, CH, D) -> TE command (T, D) and weighted std of the gripper predictions (T,)."""
    T = len(chunks)
    w = np.exp(-COEFF * np.arange(CH))
    cmd, spr = np.zeros((T, chunks.shape[2])), np.zeros(T)
    for t in range(T):
        j = np.arange(max(0, t - CH + 1), t + 1)          # oldest first -> k = 0
        p = chunks[j, t - j]                               # (n, D)
        ww = w[:len(j)] / w[:len(j)].sum()
        cmd[t] = ww @ p
        spr[t] = np.sqrt(ww @ (p[:, 5] - cmd[t, 5]) ** 2)
    return cmd, spr


def readout(g, spr=None):
    k0 = H.first_close(g)
    if k0 is None:
        return dict(onset=-1, dur=np.nan, spread=np.nan)
    g_open = float(np.median(g[max(k0 - 15, 0):k0])) if k0 > 0 else float(g[0])
    g_min = float(g[k0:k0 + 91].min())
    d = g_open - g_min
    a = next((t for t in range(max(k0 - 15, 0), len(g)) if g[t] <= g_open - 0.1 * d), None)
    b = next((t for t in range(k0, len(g)) if g[t] <= g_open - 0.9 * d), None)
    dur = (b - a) if a is not None and b is not None else np.nan
    sp = float(np.median(spr[max(k0 - 15, 0):(b if b is not None else k0) + 1])) if spr is not None else np.nan
    return dict(onset=k0, dur=dur, spread=sp)


def sign_p(x):
    x = np.asarray(x)[~np.isnan(x)]
    x = x[x != 0]
    n, k = len(x), int((x > 0).sum())
    p = sum(comb(n, i) for i in range(n + 1) if comb(n, i) <= comb(n, k)) / 2 ** n if n else np.nan
    return k, n, p


def main():
    models = {m: H.load_model(*H.MODELS[m]) for m in H.MODELS}
    rows, store = [], {}
    t_start = time.time()
    for rl, (folder, csv) in H.ROLL.items():
        ds = LeRobotDataset("ericc430/" + folder, root=H.HUB / folder)
        cols = ds.hf_dataset.with_format(None).select_columns(["episode_index", "action"])
        epi, act = np.array(cols["episode_index"]), np.stack(cols["action"])
        meta = pd.read_csv(H.REPO / "episode_meta" / f"{csv}.csv", encoding="utf-8-sig", comment="#").dropna(subset=["episode_index"])
        meta = meta.set_index(meta.episode_index.astype(int))
        for e in np.unique(epi):
            idx = np.flatnonzero(epi == e)
            ch = {m: [] for m in models}
            for s in range(0, len(idx), BS):
                items = [ds[int(i)] for i in idx[s:s + BS]]
                batch = {k: (torch.stack([it[k] for it in items]) if isinstance(items[0][k], torch.Tensor) else [it[k] for it in items])
                         for k in items[0]}
                for m, (pol, pre, post) in models.items():
                    b = {k: (v.to(H.dev) if isinstance(v, torch.Tensor) else v) for k, v in pre(dict(batch)).items()}
                    with torch.no_grad():
                        ch[m].append(post(pol.predict_action_chunk(b).cpu()).numpy())
            rec = readout(act[idx, 5])
            row = dict(rollout=rl, ep=int(e), pid=meta.placement_id.get(e, "?"), outcome=meta.outcome.get(e, "?"),
                       n=len(idx), rec_onset=rec["onset"], rec_dur=rec["dur"])
            for m in models:
                c = np.concatenate(ch[m])
                cmd, spr = te(c)
                r = readout(cmd[:, 5], spr)
                row.update({f"{m}_{k}": v for k, v in r.items()})
                row[f"{m}_gmin_stall"] = float(cmd[135:673, 5].min()) if len(idx) > 135 else np.nan
                if m == OWN[rl]:
                    row["own_mae_g"] = float(np.abs(cmd[:, 5] - act[idx, 5]).mean())
                    row["own_mae_arm"] = float(np.abs(cmd[:, :5] - act[idx, :5]).mean())
                store[f"{rl}|{e}|{m}"] = cmd[:, 5].astype(np.float32)
                store[f"{rl}|{e}|{m}|spread"] = spr.astype(np.float32)
            store[f"{rl}|{e}|rec"] = act[idx, 5].astype(np.float32)
            rows.append(row)
            print(f"{rl} ep{e:2d} {row['pid']:4s} n={len(idx):3d}  rec {rec['onset']:4d}  "
                  + "  ".join(f"{m} {row[m + '_onset']:4d}" for m in models)
                  + f"  own mae g {row['own_mae_g']:.2f}  [{time.time() - t_start:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(H.OUT / "h19b_te_replay.csv", index=False)
    np.savez_compressed(H.OUT / "h19b_te_series.npz", **store)

    print("\n== sanity: own model, TE onset vs recorded onset ==")
    for rl, m in OWN.items():
        d = df[df.rollout == rl]
        has = d.rec_onset >= 0
        ok = (d[has][f"{m}_onset"] >= 0) & ((d[has][f"{m}_onset"] - d[has].rec_onset).abs() <= 5)
        extra = ((~has) & (d[f"{m}_onset"] >= 0)).sum()
        print(f"  {rl} / {m}: within +-5 in {ok.sum()}/{has.sum()} ({ok.mean():.0%}), onset without recorded close {extra}, "
              f"gripper MAE median {d.own_mae_g.median():.2f}, arm MAE median {d.own_mae_arm.median():.2f}")
    print("\n== A: c32 (ep27) ==")
    print(df[df.ep == 27][["rollout", "rec_onset", "mfull_onset", "aimfix_onset", "mfull_gmin_stall", "aimfix_gmin_stall"]].to_string(index=False))
    print("\n== B: paired aimfix - mfull ==")
    for name, d in [("pooled", df)] + list(df.groupby("rollout")):
        d = d[(d.mfull_onset >= 0) & (d.aimfix_onset >= 0)]
        for q in ("dur", "spread"):
            x = d[f"aimfix_{q}"] - d[f"mfull_{q}"]
            k, n, p = sign_p(x)
            print(f"  {name:13s} {q:6s}: aimfix larger {k}/{n}, sign p = {p:.3f}; median mfull {d['mfull_' + q].median():.2f} "
                  f"aimfix {d['aimfix_' + q].median():.2f}")
    print(f"\nwrote {H.OUT}/h19b_te_replay.csv, h19b_te_series.npz  ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()

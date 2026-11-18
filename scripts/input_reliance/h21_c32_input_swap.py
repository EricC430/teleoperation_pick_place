"""H21: which input keeps aimfix28 from closing at c32? Swap front-left / joint state on the stall, TE simulated (2026-11-18).

h20: to the aimfix28 backbone the c32 stall wrist view looks like views that get closed (p_close 0.30 vs 0.15 for views
this model did close on) -> the wrist view does not explain the stall. Left: front-left image, joint state.

Rollout 11/17 aimfix28 bottle ep27 (c32, stall f135-672) replayed open loop through both models (aimfix ca92727ebcff,
mfull 8c42b7e0652e); TE as h19b (coeff -0.01). From frame SWAP0 = 120 on (before: as recorded), one input replaced:
  base       nothing replaced
  fl_black   front-left = black (training blacks it with p 0.3 in the 30 frames before the first close)
  fl_mfull   front-left = the 11/15 M-full rollout ep27 (same placement c32) front-left frame at its recorded onset - 5
  st_mfull   observation.state = 11/15 ep27 state at its recorded onset - 5
Cue check (is black front-left by itself a close cue?): every other 11/17 episode with a recorded close, aimfix model,
front-left black over the whole episode vs base.
onset = first frame where the TE gripper, after >= 75, falls below 75.

Reading rules, fixed before the run:
  sanity   base: aimfix no onset in 135-672, mfull onset in 135-672 (h19b: none / f230). Fail -> stop.
  cue      fl_black onset earlier than base by > 15 frames (or appearing where base has none) in >= 2/3 of the cue
           episodes -> black is a close cue by itself -> fl_black on c32 is NOT read.
  FL       aimfix onset in 135-672 under fl_mfull (or under fl_black when cue passed) -> the front-left content
           contributes to the stall.
  state    aimfix onset in 135-672 under st_mfull -> the joint state contributes.
  neither  no condition makes aimfix close -> no single input; describe the minimum TE gripper per condition.
One episode / one placement; open loop.

    LEROBOT_CONTAINER_NAME=h21 ./scripts/run_container.sh python scripts/input_reliance/h21_c32_input_swap.py
Writes data/logs/input_reliance/h21_*.csv only.
"""
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts/input_reliance")
import h19_c32_stall_close as H  # noqa: E402
from h19b_te_replay import te  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

SWAP0, BS, ST = 120, 32, "observation.state"


def load(rl):
    folder = H.ROLL[rl][0]
    ds = LeRobotDataset("ericc430/" + folder, root=H.HUB / folder)
    cols = ds.hf_dataset.with_format(None).select_columns(["episode_index", "action"])
    epi, act = np.array(cols["episode_index"]), np.stack(cols["action"])
    return ds, epi, act


def run(models, ds, idx, cond, ref, from_frame):
    ch = {m: [] for m in models}
    for s in range(0, len(idx), BS):
        items = [dict(ds[int(i)]) for i in idx[s:s + BS]]
        for k, it in enumerate(items):
            if s + k < from_frame:
                continue
            if cond == "fl_black":
                it[H.FL] = torch.zeros_like(it[H.FL])
            elif cond == "fl_mfull":
                it[H.FL] = ref[H.FL]
            elif cond == "st_mfull":
                it[ST] = ref[ST]
        batch = {k: (torch.stack([it[k] for it in items]) if isinstance(items[0][k], torch.Tensor) else [it[k] for it in items])
                 for k in items[0]}
        for m, (pol, pre, post) in models.items():
            b = {k: (v.to(H.dev) if isinstance(v, torch.Tensor) else v) for k, v in pre(dict(batch)).items()}
            with torch.no_grad():
                ch[m].append(post(pol.predict_action_chunk(b).cpu()).numpy())
    return {m: te(np.concatenate(ch[m]))[0][:, 5] for m in models}


def onset_in(g, lo=135, hi=673):
    k = H.first_close(g)
    return k if k is not None and lo <= k < hi else -1


def main():
    t0 = time.time()
    models = {m: H.load_model(*H.MODELS[m]) for m in H.MODELS}
    dsa, epa, acta = load("11-17 aimfix")
    dsm, epm, actm = load("11-15 mfull")
    im = np.flatnonzero(epm == H.EP)
    km = H.first_close(actm[im, 5])
    ref = dict(dsm[int(im[km - 5])])
    ia = np.flatnonzero(epa == H.EP)
    st_stall = np.stack([dsa[int(i)][ST].numpy() for i in ia[135:673:30]])
    print(f"11/15 ep27 recorded onset {km}; ref frame {km - 5} state {np.round(ref[ST].numpy(), 1)}")
    print(f"11/17 ep27 stall state (every 30 f, median) {np.round(np.median(st_stall, 0), 1)}  "
          f"range {np.round(st_stall.max(0) - st_stall.min(0), 1)}")

    rows = []
    for cond in ("base", "fl_black", "fl_mfull", "st_mfull"):
        g = run(models, dsa, ia, cond, ref, SWAP0)
        for m in models:
            rows.append(dict(cond=cond, model=m, onset=onset_in(g[m]), gmin_stall=float(g[m][135:673].min())))
        print(f"  {cond:9s} " + "  ".join(f"{m} onset {onset_in(g[m]):4d} gmin {g[m][135:673].min():5.1f}" for m in models)
              + f"  [{time.time() - t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(H.OUT / "h21_c32_swap.csv", index=False)

    print("\ncue check (aimfix, 11/17, front-left black whole episode):", flush=True)
    aim = {"aimfix": models["aimfix"]}
    cue = []
    for e in np.unique(epa):
        if e == H.EP:
            continue
        idx = np.flatnonzero(epa == e)
        rec = H.first_close(acta[idx, 5])
        if rec is None:
            continue
        gb = run(aim, dsa, idx, "base", None, 0)["aimfix"]
        gk = run(aim, dsa, idx, "fl_black", None, 0)["aimfix"]
        b, k = H.first_close(gb), H.first_close(gk)
        earlier = (k is not None) and (b is None or k < b - 15)
        cue.append(dict(ep=int(e), rec=rec, base=-1 if b is None else b, black=-1 if k is None else k, earlier=earlier))
        print(f"  ep{e:2d} rec {rec:4d} base {cue[-1]['base']:4d} black {cue[-1]['black']:4d} {'EARLIER' if earlier else ''}", flush=True)
    c = pd.DataFrame(cue)
    c.to_csv(H.OUT / "h21_cue_check.csv", index=False)
    print(f"\ncue: black earlier by > 15 in {c.earlier.sum()}/{len(c)} (cue if >= {int(np.ceil(2 * len(c) / 3))}); "
          f"median shift {np.median(c.black - c.base):.0f} frames")
    print(df.to_string(index=False))
    print(f"\nwrote {H.OUT}/h21_c32_swap.csv, h21_cue_check.csv  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

"""H20: does the c32 stall view (11/17 aimfix28 rollout ep27, f135-672) look, to the aimfix28 model's own wrist
features, like the training views of aimfix episodes BEFORE they are aimed (2026-11-18)?

Boyu 11/18: c32 was aimed, no correction needed. aimfix episodes start looking at the left / right edge of the bottle,
the offset is not large -> maybe close to views of earlier data that close, so the model cannot tell "aimed" from
"not yet aimed" and learns "close-up -> wait" (h19b: aimfix closes later in 57/69 episodes, c32 never).

Method: wrist image -> aimfix28 200k backbone (ca92727ebcff; the ResNet18 shared by all cameras) -> layer4 map
(512 x 15 x 20) -> average pooled to 3 x 4 (keeps left / right) -> 6144-d, L2 normalised, cosine kNN, k = 20.
Pool = training set of that model (merged omx_pick_place_pilot_cup129_bottle135_aimfix28_twoobj10, exclude_episodes
removed), every 3rd frame. Per pool frame, with onset = first frame where the recorded gripper action, after >= 75,
falls below 75 (h18 / h19):
  close2s  onset - 30 <= f < onset (a close follows within 2 s)
  group    cup (ep 0-136), bottle (137-272), aimfix_pre (273-300, f < onset - 15 or no onset), aimfix_rest, twoobj (301-310)
Queries:
  Q    11/17 aimfix rollout ep27 (c32) f135-672, every 3rd frame (the stall; open gripper, aimed per Boyu)
  CC   11/17 aimfix rollout, every other episode with a recorded close: f onset-15 .. onset-1 (views this model closed on)
  M32  11/15 M-full rollout ep27 (c32): f onset-15 .. onset-1 (same placement, M-full closed) - description only
Per query frame: p_close = share of its 20 neighbours with close2s; group shares of the neighbours.

Reading rules, fixed before the run:
  sanity   leave-one-episode-out kNN on 3000 random bottle-pool frames (ep 137-300): AUC of p_close vs their own
           close2s >= 0.70. Fail -> the features do not carry "about to close" -> stop, no reading.
  for      (= "c32 looks like not-yet-aimed aimfix views")
           Q median p_close < 25th percentile of the CC episode medians, AND Q's aimfix_pre share >= 2x CC's aimfix_pre
           share and >= 2x the aimfix_pre share of the bottle pool.
  against  Q median p_close >= 25th percentile of CC -> to these features c32 looks like views that get closed; the stall
           is not explained by the wrist view (front-left, state, the transformer are left).
  mixed    p_close low but no aimfix_pre enrichment -> c32 looks like non-closing views of OTHER data; name the group.
Q is one episode / one placement, its frames are not independent. Wrist features only; the model also sees front-left
and the joint state.

    LEROBOT_CONTAINER_NAME=h20 ./scripts/run_container.sh python scripts/input_reliance/h20_c32_wrist_knn.py
Writes data/logs/input_reliance/h20_*.csv / .npz / .jpg only.
"""
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts/input_reliance")
import h19_c32_stall_close as H  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

K, STEP, BS, NW = 20, 3, 64, 10
POOL = "omx_pick_place_pilot_cup129_bottle135_aimfix28_twoobj10"
EXCL = {73, 74, 75, 85, 86, 87, 98, 135, 271}
AIM, ROLL_A, ROLL_M = (273, 300), "11-17 aimfix", "11-15 mfull"
rng = np.random.default_rng(0)


def onsets(ds):
    cols = ds.hf_dataset.with_format(None).select_columns(["episode_index", "action"])
    epi, g = np.array(cols["episode_index"]), np.stack(cols["action"])[:, 5]
    out = {}
    for e in np.unique(epi):
        idx = np.flatnonzero(epi == e)
        out[int(e)] = (idx, H.first_close(g[idx]))
    return out


def group(e, f, k0):
    if e <= 136:
        return "cup"
    if e <= 272:
        return "bottle"
    if e <= 300:
        return "aimfix_pre" if (k0 is None or f < k0 - 15) else "aimfix_rest"
    return "twoobj"


class Sub(torch.utils.data.Dataset):
    def __init__(self, ds, idx):
        self.ds, self.idx = ds, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.ds[int(self.idx[i])][H.W]


def feats(ds, idx, pol, pre_w):
    dl = torch.utils.data.DataLoader(Sub(ds, idx), batch_size=BS, num_workers=NW)
    out, t0 = [], time.time()
    for n, x in enumerate(dl):
        x = pre_w(x.to(H.dev))
        with torch.no_grad():
            fm = pol.model.backbone(x)["feature_map"]
            v = F.adaptive_avg_pool2d(fm, (3, 4)).flatten(1)
        out.append(F.normalize(v, dim=1).half())
        if n % 50 == 0:
            print(f"  {n * BS}/{len(idx)}  [{time.time() - t0:.0f}s]", flush=True)
    return torch.cat(out)


def auc(score, y):
    y = np.asarray(y, bool)
    r = pd.Series(score).rank().values
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    pol, pre, _ = H.load_model(*H.MODELS["aimfix"])
    # image normalisation exactly as the preprocessor does it: run it on a dummy batch built from one real item
    def pre_w(x):
        return pre({H.W: x, "observation.state": torch.zeros(len(x), 6, device=x.device), "task": [""] * len(x)})[H.W]

    pool = LeRobotDataset("ericc430/" + POOL, root=H.HUB / POOL)
    on = onsets(pool)
    rows = []
    for e, (idx, k0) in on.items():
        if e in EXCL:
            continue
        for f in range(0, len(idx), STEP):
            rows.append(dict(ep=e, f=f, gi=int(idx[f]), close2s=k0 is not None and k0 - 30 <= f < k0, group=group(e, f, k0)))
    P = pd.DataFrame(rows)
    print(f"pool {len(P)} frames from {P.ep.nunique()} episodes", flush=True)

    qrows = []
    for rl in (ROLL_A, ROLL_M):
        folder = H.ROLL[rl][0]
        ds = LeRobotDataset("ericc430/" + folder, root=H.HUB / folder)
        for e, (idx, k0) in onsets(ds).items():
            if rl == ROLL_A and e == H.EP:
                sel, s = range(135, min(673, len(idx)), STEP), "Q"
            elif rl == ROLL_A and k0 is not None and k0 >= 15:
                sel, s = range(k0 - 15, k0), "CC"
            elif rl == ROLL_M and e == H.EP and k0 is not None:
                sel, s = range(max(k0 - 15, 0), k0), "M32"
            else:
                continue
            qrows += [dict(set=s, rollout=rl, ep=e, f=f, gi=int(idx[f])) for f in sel]
    Qd = pd.DataFrame(qrows)
    print(Qd.groupby("set").size().to_dict(), flush=True)

    t0 = time.time()
    Fp = feats(pool, P.gi.values, pol, pre_w)
    print(f"pool features {time.time() - t0:.0f}s", flush=True)
    Fq = []
    for rl in (ROLL_A, ROLL_M):
        folder = H.ROLL[rl][0]
        ds = LeRobotDataset("ericc430/" + folder, root=H.HUB / folder)
        d = Qd[Qd.rollout == rl]
        Fq.append((d.index.values, feats(ds, d.gi.values, pol, pre_w)))
    Fqa = torch.zeros(len(Qd), Fp.shape[1], dtype=Fp.dtype, device=Fp.device)
    for ii, v in Fq:
        Fqa[torch.as_tensor(ii, device=Fp.device)] = v

    y = torch.as_tensor(P.close2s.values, device=Fp.device).float()
    pep = torch.as_tensor(P.ep.values, device=Fp.device)
    G = P.group.values
    groups = ["cup", "bottle", "aimfix_pre", "aimfix_rest", "twoobj"]

    # sanity: leave-one-episode-out on bottle pool frames
    cand = np.flatnonzero((P.ep >= 137) & (P.ep <= 300))
    sa = rng.choice(cand, size=min(3000, len(cand)), replace=False)
    ps = []
    for s in range(0, len(sa), 500):
        b = torch.as_tensor(sa[s:s + 500], device=Fp.device)
        sim = Fp[b].float() @ Fp.float().T
        sim[pep[b][:, None] == pep[None, :]] = -2
        nn = sim.topk(K, dim=1).indices
        ps.append(y[nn].mean(1).cpu().numpy())
    a = auc(np.concatenate(ps), P.close2s.values[sa])
    print(f"\n== sanity: LOEO kNN AUC on {len(sa)} bottle frames = {a:.3f} (need >= 0.70) ==", flush=True)

    sim = Fqa.float() @ Fp.float().T
    nn = sim.topk(K, dim=1)
    Qd["p_close"] = y[nn.indices].mean(1).cpu().numpy()
    Qd["sim_top1"] = nn.values[:, 0].cpu().numpy()
    gn = G[nn.indices.cpu().numpy()]
    for g in groups:
        Qd[g] = (gn == g).mean(1)
    Qd.to_csv(H.OUT / "h20_c32_wrist_knn.csv", index=False)
    np.savez_compressed(H.OUT / "h20_knn_idx.npz", nn=nn.indices.cpu().numpy(), pool_gi=P.gi.values, pool_ep=P.ep.values,
                        pool_f=P.f.values)

    bp = P[(P.ep >= 137) & (P.ep <= 300)]
    print("\nbottle-pool base: close2s {:.3f}, ".format(bp.close2s.mean()) + ", ".join(f"{g} {(bp.group == g).mean():.3f}" for g in groups))
    cc_ep = Qd[Qd.set == "CC"].groupby("ep").p_close.median()
    print(f"CC episode medians of p_close: n={len(cc_ep)}, q25 {cc_ep.quantile(.25):.3f}, median {cc_ep.median():.3f}, "
          f"min {cc_ep.min():.3f}")
    for s in ("Q", "CC", "M32"):
        d = Qd[Qd.set == s]
        print(f"  {s:3s} n={len(d):3d}  p_close median {d.p_close.median():.3f}  top1 sim {d.sim_top1.median():.3f}  "
              + "  ".join(f"{g} {d[g].mean():.3f}" for g in groups))
    print("\nQ per frame (every 4th row):")
    print(Qd[Qd.set == "Q"][["f", "p_close", "sim_top1"] + groups].iloc[::4].round(2).to_string(index=False))
    if a < 0.70:
        print("\nSANITY FAILED -> no reading")

    # sheet: 6 Q frames + their top 5 neighbours, for Boyu to look at
    rq = H.ROLL[ROLL_A][0]
    dsq = LeRobotDataset("ericc430/" + rq, root=H.HUB / rq)
    qi = Qd[Qd.set == "Q"].index.values
    pick = qi[np.linspace(0, len(qi) - 1, 6).round().astype(int)]
    lines = []
    for q in pick:
        tiles = [lab(H.hwc(dsq[int(Qd.gi[q])][H.W]), f"Q c32 f{Qd.f[q]} p_close {Qd.p_close[q]:.2f}")]
        for j in nn.indices[q, :5].cpu().numpy():
            r = P.iloc[j]
            tiles.append(lab(H.hwc(pool[int(r.gi)][H.W]), f"ep{r.ep} f{r.f} {r.group}{' close2s' if r.close2s else ''}"))
        lines.append(np.hstack(tiles))
    cv2.imwrite(str(H.OUT / "h20_knn_sheet.jpg"), cv2.cvtColor(np.vstack(lines), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"\nwrote {H.OUT}/h20_c32_wrist_knn.csv, h20_knn_idx.npz, h20_knn_sheet.jpg  ({time.time() - t0:.0f}s)")


def lab(im, t):
    im = cv2.resize(im, (320, 240))
    cv2.putText(im, t, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(im, t, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    return im


if __name__ == "__main__":
    main()

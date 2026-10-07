"""H1, normalized: on misaligned (bad_aim) closed-loop frames, how much of the NEEDED correction does the wrist view drive?

p_orig   = plan with the real (misaligned) wrist view
p_align  = plan with the wrist view replaced by a success episode's aligned view at the same phase
cup      = true placement of this episode (eval-close)
wrist correction = (p_orig - p_align) . (cup - p_align) / |cup - p_align|^2
    0 = the real wrist view changes nothing, 1 = it pulls the plan exactly onto the cup.
front-left follow = (p_fl - p_orig) . (cup_donor - cup) / |cup_donor - cup|^2   (front-left from the donor episode)
Both are fractions of 'what full use of that input would do', so they can be compared.
Caveat: if the cup was already pushed before the first close, `cup` is not where the cup really was.
"""
from pathlib import Path
exec(open(str(Path(__file__).with_name("h1_swap.py")), encoding="utf-8").read().split("W, FL, ST =")[0])
import pandas as pd
W, FL = "observation.images.wrist", "observation.images.front-left"
root = r"D:\hf\lerobot\ericc430\rollout_omx_b1_uvc60_100k_nas30_A1_paper_cup_20261006_094908"
m = A.placements("episode_meta/rollout_omx_b1_uvc60_100k_nas30_A1_paper_cup_20261006_close_loop.csv",
                 str(A.REPO / "configs/placements/campA_136sym_20260908_20260908_"))
cup = {int(r.episode_index): np.array([r.r * np.cos(np.deg2rad(r.th)), r.r * np.sin(np.deg2rad(r.th))]) for r in m.itertuples()}
bad = m[m.mechanism.fillna("").str.contains("bad_aim")].episode_index.tolist()
good = m[m.outcome == "success"].episode_index.tolist()
info = {}
for ep in bad + good:
    ds = LeRobotDataset("ericc430/x", root=root, episodes=[ep])
    act = np.stack([ds.hf_dataset[i]["action"].numpy() for i in range(len(ds))])
    g = act[:, 5]; d = g[0] - g.min()
    if d >= 10: info[ep] = (ds, int(np.argmax(g < g[0] - 0.6 * d)))
bad = [e for e in bad if e in info]; good = [e for e in good if e in info]
for D in (5, 15):
    wc, ff, need = [], [], []
    for i, ep in enumerate(bad):
        ds, k = info[ep]; dn = good[i % len(good)]; ds2, k2 = info[dn]
        if k - D < 0 or k2 - D < 0: continue
        it, it2 = ds[k - D], ds2[k2 - D]
        p0 = plan(it, D)
        a = dict(it); a[W] = it2[W]; pa = plan(a, D)
        v = cup[ep] - pa; need.append(np.linalg.norm(v))
        wc.append((p0 - pa) @ v / (v @ v))
        b = dict(it); b[FL] = it2[FL]; u = cup[dn] - cup[ep]
        ff.append((plan(b, D) - p0) @ u / (u @ u))
    wc, ff = np.array(wc), np.array(ff)
    print(f"D={D}: n={len(wc)}  needed correction median {np.median(need):.1f} cm | "
          f"wrist correction fraction median {np.median(wc):+.2f} (IQR {np.percentile(wc,25):+.2f}..{np.percentile(wc,75):+.2f}) | "
          f"front-left follow fraction median {np.median(ff):+.2f} (IQR {np.percentile(ff,25):+.2f}..{np.percentile(ff,75):+.2f})")

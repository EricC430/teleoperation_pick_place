"""Closed-loop frames where the policy was MISALIGNED (bad_aim): does the wrist view drive a correction?
At frame q = first_close - D of a bad_aim episode, replace the wrist image with the wrist image of a SUCCESS
episode at its own first_close - D (an 'aligned' view). If the wrist is used for correction, the plan should move.
Compare with doing the same to front-left. Also save a strip of the original vs donor wrist frames to eyeball."""
import sys
exec(open(str(__import__("pathlib").Path(__file__).with_name("h1_swap.py")), encoding="utf-8").read().split("W, FL, ST =")[0])
import pandas as pd
from PIL import Image
W, FL = "observation.images.wrist", "observation.images.front-left"
root = r"D:\hf\lerobot\ericc430\rollout_omx_b1_uvc60_100k_nas30_A1_paper_cup_20261006_094908"
m = pd.read_csv("episode_meta/rollout_omx_b1_uvc60_100k_nas30_A1_paper_cup_20261006_close_loop.csv", encoding="utf-8-sig").dropna(subset=["episode_index"])
m["episode_index"] = m.episode_index.astype(int)
bad = m[m.mechanism.fillna("").str.contains("bad_aim")].episode_index.tolist()
good = m[m.outcome == "success"].episode_index.tolist()
info = {}
for ep in bad + good:
    ds = LeRobotDataset("ericc430/x", root=root, episodes=[ep])
    act = np.stack([ds.hf_dataset[i]["action"].numpy() for i in range(len(ds))])
    g = act[:, 5]; d = g[0] - g.min()
    if d >= 10: info[ep] = (ds, int(np.argmax(g < g[0] - 0.6 * d)))
bad = [e for e in bad if e in info]; good = [e for e in good if e in info]
S = "outputs/input_reliance/"; __import__("os").makedirs(S, exist_ok=True)
to = lambda t: Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((213, 160))
for D in (5, 15):
    rw, rf, rg = [], [], []; strip = []
    for i, ep in enumerate(bad):
        ds, k = info[ep]; dn = good[i % len(good)]; ds2, k2 = info[dn]
        if k - D < 0 or k2 - D < 0: continue
        it, it2 = ds[k - D], ds2[k2 - D]; p0 = plan(it, D)
        a = dict(it); a[W] = it2[W]; rw.append(np.linalg.norm(plan(a, D) - p0))
        b = dict(it); b[FL] = it2[FL]; rf.append(np.linalg.norm(plan(b, D) - p0))
        c = dict(it); c[W] = torch.full_like(it[W], 0.5); rg.append(np.linalg.norm(plan(c, D) - p0))
        if D == 5 and len(strip) < 6: strip.append((to(it[W]), to(it2[W])))
    print(f"D={D}: n={len(rw)} bad_aim eps; plan moves (median cm): wrist->aligned donor {np.median(rw):.2f} | front-left->donor {np.median(rf):.2f} | wrist->gray {np.median(rg):.2f}")
    if strip:
        im = Image.new("RGB", (213 * 2, 160 * len(strip)))
        for j, (x, y) in enumerate(strip): im.paste(x, (0, 160 * j)); im.paste(y, (213, 160 * j))
        im.save(S + "h1_wrist_strip.png")
print("bad_aim eps:", bad)

"""H2: is the radial pull toward mid-range already there open-loop, with no external change?

For each demo, find the human's grasp frame k. Query the policy at frame k - D (D = 15/30/45 frames before),
take its predicted action at chunk step D (= its plan for frame k), map pred and human joints with the
same human joint->(theta, r) map, and regress r_pred on r_human.
eval-open (unseen placements, recorded 9/13 with the training data) vs training set (memorized).
"""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "scripts")
import aim_error as A
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

CK = Path("D:/hf/hub/models--ericc430--act_omx_b1_uvc60_100k/snapshots/0b3c41f9a3b01466eee35249a9ee700766be74a0")
dev = "cuda"
cfg = PreTrainedConfig.from_pretrained(str(CK))
pol = get_policy_class("act").from_pretrained(str(CK), config=cfg).eval().to(dev)
pre, post = make_pre_post_processors(pol.config, pretrained_path=str(CK))

tm = A.placements("episode_meta/omx_pick_place_pilot_uvc_60.csv", str(A.REPO / "configs/placements/campA_136sym_20260908_20260908_"))
tg = A.grasp_joints(".cache/lerobot/omx_pick_place_pilot_uvc_60", 0.6, 10)
tm = tm[tm.episode_index.isin(tg)]
J = np.array([tg[e] for e in tm.episode_index]); C = A.fit(J, tm.th.values, tm.r.values)

DS = {"eval-open": ("ericc430/omx_pick_place_pilot_uvc_open_loop_eval", ".cache/lerobot/omx_pick_place_pilot_uvc_open_loop_eval"),
      "train": ("ericc430/omx_pick_place_pilot_uvc_60", ".cache/lerobot/omx_pick_place_pilot_uvc_60")}
DS_EPS = {"eval-open": None, "train": list(range(0, 60, 3))}   # 20 training eps is enough for a reference slope
Ds = [15, 30, 45]

def chunk(item):
    b = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
    b = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in pre(b).items()}
    with torch.no_grad():
        return post(pol.predict_action_chunk(b)[0].cpu()).numpy()

res = {}
for name, (rid, root) in DS.items():
    n = LeRobotDataset(rid, root=root).meta.total_episodes
    eps = DS_EPS[name] or list(range(n))
    rows = []
    for ep in eps:
        ds = LeRobotDataset(rid, root=root, episodes=[ep])
        act = np.stack([ds.hf_dataset[i]["action"].numpy() for i in range(len(ds))])
        g = act[:, 5]; depth = g[0] - g.min()
        if depth < 10:
            continue
        k = int(np.argmax(g < g[0] - 0.6 * depth))
        th_h, r_h = A.predict(C, act[k:k + 1, :5])
        row = {"ep": ep, "k": k, "r_h": r_h[0], "th_h": th_h[0]}
        for D in Ds:
            if k - D < 0:
                row[f"r_{D}"] = row[f"th_{D}"] = np.nan; continue
            pa = chunk(ds[k - D])[D, :5]
            t_, r_ = A.predict(C, pa[None]); row[f"r_{D}"], row[f"th_{D}"] = r_[0], t_[0]
        rows.append(row)
    res[name] = rows
    print(f"\n== {name}: {len(rows)} episodes; human r range {min(x['r_h'] for x in rows):.1f}-{max(x['r_h'] for x in rows):.1f} cm")
    for D in Ds:
        x = np.array([(r["r_h"], r[f"r_{D}"], r["th_h"], r[f"th_{D}"]) for r in rows if not np.isnan(r[f"r_{D}"])])
        sr = np.polyfit(x[:, 0], x[:, 1], 1)[0]; st = np.polyfit(x[:, 2], x[:, 3], 1)[0]
        print(f"  plan made {D:2d} frames ({D/15:.0f} s) before grasp: n={len(x)}  radial slope {sr:.2f}  angle slope {st:.2f}"
              f"  |dr| median {np.median(np.abs(x[:,1]-x[:,0])):.2f} cm")
json.dump({k: [{a: float(b) for a, b in r.items()} for r in v] for k, v in res.items()},
          open("outputs/input_reliance/h2_shrink.json", "w"))

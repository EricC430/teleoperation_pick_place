"""H1: when planning the grasp, does ACT follow the images or the arm's own state?

At frame q = k - D (k = human grasp frame, D = 15 / 30), replace one input with the SAME relative
frame (k' - D) of a different episode, and see where the planned grasp (chunk step D) goes.
follow = projection of (plan_swapped - plan_orig) onto (target_other - target_orig), / |target_other - target_orig|^2
  0 = ignores this input, 1 = plan moves all the way to the other episode's target.
Positions via the human joint -> (theta, r) map, converted to x, y cm.
"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "scripts")
import aim_error as A
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

CK = Path("D:/hf/hub/models--ericc430--act_omx_b1_uvc60_100k/snapshots/0b3c41f9a3b01466eee35249a9ee700766be74a0")
dev = "cuda"
pol = get_policy_class("act").from_pretrained(str(CK), config=PreTrainedConfig.from_pretrained(str(CK))).eval().to(dev)
pre, post = make_pre_post_processors(pol.config, pretrained_path=str(CK))
tm = A.placements("episode_meta/omx_pick_place_pilot_uvc_60.csv", str(A.REPO / "configs/placements/campA_136sym_20260908_20260908_"))
tg = A.grasp_joints(".cache/lerobot/omx_pick_place_pilot_uvc_60", 0.6, 10)
tm = tm[tm.episode_index.isin(tg)]
C = A.fit(np.array([tg[e] for e in tm.episode_index]), tm.th.values, tm.r.values)

def xy(j5):
    t, r = A.predict(C, j5[None]); t, r = t[0], r[0]
    return np.array([r * np.cos(np.deg2rad(t)), r * np.sin(np.deg2rad(t))])

def plan(item, D):
    b = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
    b = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in pre(b).items()}
    with torch.no_grad():
        return xy(post(pol.predict_action_chunk(b)[0].cpu()).numpy()[D, :5])

W, FL, ST = "observation.images.wrist", "observation.images.front-left", "observation.state"
SWAPS = {"wrist": [W], "front-left": [FL], "both images": [W, FL], "state": [ST]}

for name, rid, root, eps in [("eval-open", "ericc430/omx_pick_place_pilot_uvc_open_loop_eval", ".cache/lerobot/omx_pick_place_pilot_uvc_open_loop_eval", None),
                             ("train", "ericc430/omx_pick_place_pilot_uvc_60", ".cache/lerobot/omx_pick_place_pilot_uvc_60", list(range(0, 60, 3)))]:
    eps = eps or list(range(LeRobotDataset(rid, root=root).meta.total_episodes))
    info = {}
    for ep in eps:
        ds = LeRobotDataset(rid, root=root, episodes=[ep])
        act = np.stack([ds.hf_dataset[i]["action"].numpy() for i in range(len(ds))])
        g = act[:, 5]; k = int(np.argmax(g < g[0] - 0.6 * (g[0] - g.min())))
        info[ep] = (ds, k, xy(act[k, :5]))
    print(f"\n== {name} ({len(eps)} eps)")
    for D in (5, 15):
        fol = {s: [] for s in SWAPS}; base_err = []; blank = []
        for i, ep in enumerate(eps):
            ds, k, tgt = info[ep]
            ep2 = eps[(i + len(eps) // 2) % len(eps)]           # partner: far down the list -> usually a different placement
            ds2, k2, tgt2 = info[ep2]
            if k - D < 0 or k2 - D < 0:
                continue
            it, it2 = ds[k - D], ds2[k2 - D]
            p0 = plan(it, D); base_err.append(np.linalg.norm(p0 - tgt))
            v = tgt2 - tgt; vv = v @ v
            if vv < 25:                                         # targets < 5 cm apart: swap says nothing
                continue
            for s, keys in SWAPS.items():
                m = dict(it)
                for kk in keys: m[kk] = it2[kk]
                fol[s].append((plan(m, D) - p0) @ v / vv)
            m = dict(it); m[W] = torch.full_like(it[W], 0.5)
            blank.append(np.linalg.norm(plan(m, D) - p0))
        print(f"  D={D} ({D/15:.0f} s before grasp): plan error vs human {np.median(base_err):.1f} cm; follow ratio median (n={len(fol['wrist'])}): "
              + "  ".join(f"{s} {np.median(f):+.2f}" for s, f in fol.items()) + f"   | wrist->gray moves plan {np.median(blank):.2f} cm")

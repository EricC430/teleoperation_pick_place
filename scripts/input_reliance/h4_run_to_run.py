"""Same model, same placement, two closed-loop runs (10-08 B: N=100, C: N=30): why do the first grasps land
3.5 cm apart? ACT is deterministic at inference (latent = 0 outside training, modeling_act.py), so any difference
in the FIRST chunk comes from the frame-0 inputs. Feed each run's frame 0 to the model, then swap one input at a
time (front-left / wrist / state) from C into B and see how far the planned first grasp moves.

Sanity check first: the chunk predicted from B's frame 0 must reproduce B's recorded actions 0..99 (N=100), and
C's must reproduce C's recorded actions 0..29 (N=30) - otherwise this offline pipeline is not what ran on the robot.

    uv run python scripts/input_reliance/h4_run_to_run.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts")
import aim_error as A  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402

CK = Path("D:/hf/hub/models--ericc430--act_omx_b1_paper-cup_60-16-12rcvry_100k/snapshots/68528ce9941cbad4753f7072b7f2cbf478d27296")
H = "D:/hf/lerobot/ericc430/"
RUN_B = H + "rollout_omx_b1_papercup_60_16_12rcvry_100k_paper_cup_20261008_171620"  # N=100
RUN_C = H + "rollout_omx_b1_papercup_60_16_12rcvry_100k_paper_cup_20261008_192921"  # N=30
EXCLUDE = {"c7", "c16", "c17", "c36"}  # valid=0 in at least one run (2026-10-08.md §6)
W, FL, ST = "observation.images.wrist", "observation.images.front-left", "observation.state"

dev = "cuda" if torch.cuda.is_available() else "cpu"
pol = get_policy_class("act").from_pretrained(str(CK), config=PreTrainedConfig.from_pretrained(str(CK))).eval().to(dev)
pre, post = make_pre_post_processors(pol.config, pretrained_path=str(CK))
tm = A.placements("episode_meta/omx_pick_place_pilot_uvc_60.csv", str(A.REPO / "configs/placements/campA_136sym_20260908_20260908_"))
tg = A.grasp_joints(".cache/lerobot/omx_pick_place_pilot_uvc_60", 0.6, 10)
tm = tm[tm.episode_index.isin(tg)]
FIT = A.fit(np.array([tg[e] for e in tm.episode_index]), tm.th.values, tm.r.values)


def xy(j5):
    t, r = A.predict(FIT, j5[None])
    return np.array([r[0] * np.cos(np.deg2rad(t[0])), r[0] * np.sin(np.deg2rad(t[0]))])


def chunk(item):
    b = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
    b = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in pre(b).items()}
    with torch.no_grad():
        return post(pol.predict_action_chunk(b)[0].cpu()).numpy()


def first_grasp(ch, frac=0.6, min_depth=10.0):
    """Aim (x, y cm) at the first step the planned gripper is `frac` of the way closed, or None if it never closes."""
    g = ch[:, 5]
    depth = g[0] - g.min()
    if depth < min_depth:
        return None, None
    k = int(np.argmax(g <= g[0] - frac * depth))
    return xy(ch[k, :5]), k


def frame0(root, ep):
    ds = LeRobotDataset("ericc430/x", root=root, episodes=[ep])
    acts = np.stack([ds.hf_dataset[i]["action"].numpy() for i in range(min(100, len(ds)))])
    return ds[0], acts


meta = pd.read_csv("episode_meta/rollout_omx_b1_papercup_60_16_12rcvry_100k_paper_cup_20261008_171620.csv", encoding="utf-8-sig")
metc = pd.read_csv("episode_meta/rollout_omx_b1_papercup_60_16_12rcvry_100k_paper_cup_20261008_192921.csv", encoding="utf-8-sig")
rows, sanity = [], []
for ep in range(36):
    pid = meta.placement_id[ep]
    if pid in EXCLUDE:
        continue
    ib, ab = frame0(RUN_B, ep)
    ic, ac = frame0(RUN_C, ep)
    cb, cc = chunk(ib), chunk(ic)
    sanity.append((np.abs(cb[: len(ab)] - ab).max(), np.abs(cc[:30] - ac[:30]).max()))
    plans = {"B": cb, "C": cc}
    for name, key in (("C_front", FL), ("C_wrist", W), ("C_state", ST)):
        it = dict(ib)
        it[key] = ic[key]
        plans[name] = chunk(it)
    aims = {k: first_grasp(v) for k, v in plans.items()}
    row = dict(c=pid, okB=meta.outcome[ep] == "success", okC=metc.outcome[ep] == "success")
    for k, (p, step) in aims.items():
        row[f"step_{k}"] = step
        if p is not None and aims["B"][0] is not None:
            row[f"d_{k}"] = float(np.linalg.norm(p - aims["B"][0]))
    row["state_diff_max"] = float((ib[ST] - ic[ST]).abs().max())
    rows.append(row)

s = np.array(sanity)
print(f"sanity (max |predicted - recorded| action, .pos): B 0..99 median {np.median(s[:, 0]):.3f} max {s[:, 0].max():.3f} | "
      f"C 0..29 median {np.median(s[:, 1]):.3f} max {s[:, 1].max():.3f}")
df = pd.DataFrame(rows)
df["flip"] = df.okB != df.okC
pd.set_option("display.width", 200)
print(df.round(2).to_string(index=False))
for k in ("C", "C_front", "C_wrist", "C_state"):
    col = df[f"d_{k}"].dropna()
    print(f"plan moves when B's frame 0 gets {k:8s}: median {col.median():.2f} cm (IQR {col.quantile(.25):.2f}-{col.quantile(.75):.2f}, n={len(col)})")
d = df.dropna(subset=["d_C"])
print(f"d_C: flipped {d[d.flip].d_C.median():.2f} cm (n={d.flip.sum()}) vs same {d[~d.flip].d_C.median():.2f} cm (n={(~d.flip).sum()})")
out = Path("outputs/input_reliance")
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "h4_run_to_run.csv", index=False)

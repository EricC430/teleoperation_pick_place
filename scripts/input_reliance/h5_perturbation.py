"""How much does the planned first grasp move under perturbations a person cannot see? (2026-10-08)

h4_run_to_run.py could not reproduce the recorded actions from the dataset's frame 0: the robot saw raw camera
frames, the dataset stores h264-decoded ones, and the plans diverged (B ep 20: close at step 77 offline vs 57 live).
If that small a difference moves the plan, run-to-run differences need no other explanation. Test it directly on
run B (N=100: the first grasp comes from the frame-0 chunk): perturb frame 0 and re-plan.

    uv run python scripts/input_reliance/h5_perturbation.py
"""
import io
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.argv = sys.argv[:1]
_src = open("scripts/input_reliance/h4_run_to_run.py", encoding="utf-8").read().split("meta = pd.read_csv")[0]
exec(_src)  # model, chunk(), first_grasp(), frame0(), RUN_B, EXCLUDE, W, FL, ST

gen = torch.Generator().manual_seed(0)


def jpeg(t, q=90):
    im = Image.fromarray((t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    return torch.from_numpy(np.asarray(Image.open(buf)).astype(np.float32) / 255).permute(2, 0, 1)


def noise(t, s=2 / 255):
    return (t + s * torch.randn(t.shape, generator=gen)).clamp(0, 1)


def shift1(t):
    return torch.roll(t, shifts=1, dims=2)


PERTURB = {
    "front JPEG q90": (FL, jpeg), "front noise 2/255": (FL, noise), "front shift 1px": (FL, shift1),
    "wrist JPEG q90": (W, jpeg), "wrist noise 2/255": (W, noise),
}
meta = pd.read_csv("episode_meta/rollout_omx_b1_papercup_60_16_12rcvry_100k_paper_cup_20261008_171620.csv", encoding="utf-8-sig")
rows = []
for ep in range(36):
    pid = meta.placement_id[ep]
    if pid in EXCLUDE:
        continue
    it, _ = frame0(RUN_B, ep)
    base, k0 = first_grasp(chunk(it))
    row = dict(c=pid, step=k0)
    for name, (key, fn) in PERTURB.items():
        x = dict(it)
        x[key] = fn(it[key])
        p, k = first_grasp(chunk(x))
        row[name] = np.nan if p is None or base is None else float(np.linalg.norm(p - base))
        row[name + " step"] = k
    rows.append(row)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df.round(2).to_string(index=False))
for name in PERTURB:
    col = df[name].dropna()
    print(f"{name:18s}: plan moves median {col.median():.2f} cm (IQR {col.quantile(.25):.2f}-{col.quantile(.75):.2f}), "
          f">3 cm in {(col > 3).sum()}/{len(col)}; grasp disappears in {df[name].isna().sum() - df.step.isna().sum()}")
df.to_csv("outputs/input_reliance/h5_perturbation.csv", index=False)

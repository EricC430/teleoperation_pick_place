#!/usr/bin/env python
"""Where does ACT look? Decoder cross-attention over each camera's image tokens, overlaid on the frames.

ACT (lerobot, pinned version) has ONE decoder layer whose cross-attention reads the encoder output:
[latent, robot_state, <camera 1 tokens (h*w)>, <camera 2 tokens>, ...]. lerobot computes those weights
and drops them (`modeling_act.py`, `self.multihead_attn(...)[0]`); a forward hook catches them
(averaged over heads by nn.MultiheadAttention). Nothing in lerobot is modified.

Offline, on a recorded episode (a rollout's own observations = what the policy saw):

    uv run python scripts/act_attention.py \\
        --checkpoint D:/hf/hub/models--ericc430--act_omx_b1_uvc60_100k/snapshots/<rev> \\
        --dataset.root D:/hf/lerobot/ericc430/rollout_omx_b1_uvc60_100k_nas30_A1_paper_cup_20261006_094908 \\
        --episodes 0 1 --every 30

⚠️ How to read it: attention is NOT an explanation. The 4 encoder self-attention layers mix every token
with every other one before the decoder looks, so "token (i, j)" is no longer purely that image patch.
Treat the map as a hint of where the decoder reads from; check a claim with an occlusion test
(mask the region, see whether the action changes) before relying on it.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

REPO = Path(__file__).resolve().parents[1]
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def args_():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset.root", dest="root", required=True, help="dataset folder (rollout or training set)")
    p.add_argument("--dataset.repo_id", dest="repo_id", default=None, help="default: ericc430/<root folder name>")
    p.add_argument("--episodes", type=int, nargs="+", default=[0])
    p.add_argument("--every", type=int, default=30, help="sample every N frames (30 = each N=30 re-query)")
    p.add_argument("--queries", type=int, default=None,
                   help="average attention over the first Q action queries only (default: all chunk_size). "
                        "Q = n_action_steps looks at the part of the chunk that was actually executed")
    p.add_argument("--max-rows", type=int, default=12)
    p.add_argument("--out", default=str(REPO / "outputs/act_attention"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    a = args_()
    ck = Path(a.checkpoint)
    ptype = json.load(open(ck / "config.json"))["type"]
    assert ptype == "act", f"cross-attention readout is written for ACT, got {ptype}"
    cfg = PreTrainedConfig.from_pretrained(str(ck))
    policy = get_policy_class(ptype).from_pretrained(str(ck), config=cfg).eval().to(a.device)
    pre, _ = make_pre_post_processors(policy.config, pretrained_path=str(ck))
    cams = list(policy.config.image_features)                       # token order = this order
    n_1d = 1 + int(policy.config.robot_state_feature is not None) + int(policy.config.env_state_feature is not None)

    grabbed = {}
    shapes = []
    dec = policy.model.decoder.layers[0].multihead_attn
    dec.register_forward_hook(lambda m, i, o: grabbed.__setitem__("w", o[1].detach()))   # (B, queries, tokens)
    policy.model.backbone.register_forward_hook(lambda m, i, o: shapes.append(tuple(o["feature_map"].shape[-2:])))

    repo_id = a.repo_id or f"ericc430/{Path(a.root).name}"
    out = Path(a.out) / Path(a.root).name
    out.mkdir(parents=True, exist_ok=True)

    for ep in a.episodes:
        ds = LeRobotDataset(repo_id=repo_id, root=a.root, episodes=[ep])
        frames = list(range(0, len(ds), a.every))[: a.max_rows]
        rows = []
        for f in frames:
            item = ds[f]
            batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
            batch = {k: (v.to(a.device) if isinstance(v, torch.Tensor) else v) for k, v in pre(batch).items()}
            shapes.clear()
            with torch.no_grad():
                policy.predict_action_chunk(batch)
            w = grabbed["w"][0]                                       # (queries, tokens)
            w = (w[: a.queries] if a.queries else w).mean(0).float().cpu().numpy()
            maps, mass, s = {}, {"state/latent": float(w[:n_1d].sum())}, n_1d
            for cam, (h, ww) in zip(cams, shapes):
                tok = w[s:s + h * ww]; s += h * ww
                mass[cam.split(".")[-1]] = float(tok.sum())
                mass[cam.split(".")[-1] + "_uniform"] = h * ww / len(w)   # share if attention were flat
                maps[cam] = tok.reshape(h, ww)
            assert s == len(w), f"token count mismatch: used {s}, have {len(w)}"
            rows.append((f, item, maps, mass))
            print(f"ep{ep} f{f:4d}  " + "  ".join(f"{k} {v:.0%}" for k, v in mass.items() if not k.endswith("_uniform")))

        fig, axes = plt.subplots(len(rows), len(cams), figsize=(5.2 * len(cams), 3.3 * len(rows)), squeeze=False)
        for r, (f, item, maps, mass) in enumerate(rows):
            for c, cam in enumerate(cams):
                img = item[cam].permute(1, 2, 0).numpy()
                H, W = img.shape[:2]
                m = torch.from_numpy(maps[cam])[None, None]
                m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
                m = m / (m.max() + 1e-12)
                ax = axes[r, c]
                ax.imshow(img); ax.imshow(m, cmap="jet", alpha=0.45, vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                name = cam.split(".")[-1]
                ax.set_title(f"f{f}  {name}（佔 attention {mass[name]:.0%}，平均分配會是 {mass[name + '_uniform']:.0%}）",
                             fontsize=9, loc="left")
        fig.suptitle(f"{Path(a.root).name}  ep{ep}  —  decoder cross-attention"
                     f"（{'前 ' + str(a.queries) + ' 個' if a.queries else '全部'} action query 平均；每張圖各自正規化）",
                     fontsize=10, x=0.01, ha="left", y=0.998)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / (3.3 * len(rows))))
        p = out / f"ep{ep:03d}.png"
        fig.savefig(p, dpi=90); plt.close(fig)
        print(f"saved {p}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Does a region of the image change what the policy does? Offline occlusion test for ACT.

For every query frame of a recorded episode (every --every frames), predict the action chunk three ways:
original frame, frame with --mask region replaced, frame with --control region replaced (a region that
should not matter, same size). Report how far the predicted actions move (mean |delta| over the first
--horizon steps, arm joints, .pos units; 1 .pos ~ 1.8 deg).

Replacement: pixels from a reference frame (--fill-from, e.g. a training frame at the same camera pose,
i.e. what that patch looked like when the policy was trained), else OpenCV inpainting.

    uv run python scripts/occlusion_test.py \\
        --checkpoint D:/hf/hub/models--ericc430--act_omx_b1_uvc60_100k/snapshots/<rev> \\
        --dataset.root D:/hf/lerobot/ericc430/rollout_omx_b1_uvc60_100k_nas30_A1_paper_cup_20261006_094908 \\
        --camera front-left --mask 615,125,710,240 --control 100,125,195,240 \\
        --fill-from .cache/lerobot/omx_pick_place_pilot_uvc_60:10:0

The policy is deterministic at inference (ACT uses the zero latent), so every difference comes from the pixels.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

REPO = Path(__file__).resolve().parents[1]


def box(s):
    x0, y0, x1, y1 = (int(v) for v in s.split(","))
    return x0, y0, x1, y1


def args_():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset.root", dest="root", required=True)
    p.add_argument("--dataset.repo_id", dest="repo_id", default=None)
    p.add_argument("--episodes", type=int, nargs="*", default=None, help="default: all")
    p.add_argument("--camera", default="front-left")
    p.add_argument("--mask", type=box, required=True, help="x0,y0,x1,y1 in pixels")
    p.add_argument("--control", type=box, required=True, help="same-size region that should not matter")
    p.add_argument("--fill-from", default=None, help="ROOT:EPISODE:FRAME reference frame; default: cv2.inpaint")
    p.add_argument("--every", type=int, default=30)
    p.add_argument("--horizon", type=int, default=30, help="compare the first H predicted steps (= n_action_steps)")
    p.add_argument("--out", default=str(REPO / "outputs/occlusion"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    a = args_()
    ck = Path(a.checkpoint)
    cfg = PreTrainedConfig.from_pretrained(str(ck))
    policy = get_policy_class(json.load(open(ck / "config.json"))["type"]).from_pretrained(str(ck), config=cfg)
    policy = policy.eval().to(a.device)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(ck))
    key = f"observation.images.{a.camera}"

    ref = None
    if a.fill_from:
        root, ep, fr = a.fill_from.rsplit(":", 2)
        rds = LeRobotDataset(f"ericc430/{Path(root).name}", root=root, episodes=[int(ep)])
        ref = rds[int(fr)][key].clone()

    def replace(img, b):
        x0, y0, x1, y1 = b
        img = img.clone()
        if ref is not None:
            img[:, y0:y1, x0:x1] = ref[:, y0:y1, x0:x1]
        else:
            import cv2
            u8 = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            m = np.zeros(u8.shape[:2], np.uint8); m[y0:y1, x0:x1] = 255
            img = torch.from_numpy(cv2.inpaint(u8, m, 5, cv2.INPAINT_TELEA) / 255.0).permute(2, 0, 1).float()
        return img

    def chunk(item):
        b = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
        b = {k: (v.to(a.device) if isinstance(v, torch.Tensor) else v) for k, v in pre(b).items()}
        with torch.no_grad():
            c = policy.predict_action_chunk(b)[0, : a.horizon].cpu()
        return post(c).numpy()                                     # (H, 6) in .pos

    root = Path(a.root)
    repo_id = a.repo_id or f"ericc430/{root.name}"
    meta = LeRobotDataset(repo_id, root=root).meta
    eps = a.episodes if a.episodes is not None else list(range(meta.total_episodes))
    rows, signed = [], []
    example = None
    for ep in eps:
        ds = LeRobotDataset(repo_id, root=root, episodes=[ep])
        for f in range(0, len(ds), a.every):
            item = ds[f]
            base = chunk(item)
            out = {}
            for name, b in (("mask", a.mask), ("control", a.control)):
                it = dict(item); it[key] = replace(item[key], b)
                diff = chunk(it) - base
                out[name] = np.abs(diff)[:, :5].mean()  # arm joints
                out[name + "_signed"] = diff.mean(0)     # per joint, all 6
                if example is None and name == "mask":
                    example = (item[key], it[key])
            rows.append((ep, f, out["mask"], out["control"]))
            signed.append(np.r_[out["mask_signed"], out["control_signed"]])
        r = [x for x in rows if x[0] == ep]
        print(f"ep{ep:3d}: mask {np.mean([x[2] for x in r]):.3f}  control {np.mean([x[3] for x in r]):.3f} .pos")

    R = np.array([(x[2], x[3]) for x in rows])
    print(f"\nall {len(R)} queries: |delta action| mask median {np.median(R[:, 0]):.3f} (p90 {np.percentile(R[:, 0], 90):.3f})"
          f" | control median {np.median(R[:, 1]):.3f} (p90 {np.percentile(R[:, 1], 90):.3f}) .pos"
          f" | mask > control in {np.mean(R[:, 0] > R[:, 1]):.0%} of queries")

    Sg = np.array(signed); names = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]
    for lab, sl in (("mask", slice(0, 6)), ("control", slice(6, 12))):
        m = Sg[:, sl]
        print(f"signed delta {lab:7s} mean: " + "  ".join(f"{n} {v:+.2f}" for n, v in zip(names, m.mean(0)))
              + "   (share of queries with the same sign as the mean: "
              + " ".join(f"{np.mean(np.sign(m[:, j]) == np.sign(m[:, j].mean())):.0%}" for j in range(5)) + ")")
    out = Path(a.out) / root.name; out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "occlusion.csv", np.array([(e, f, m, c) for e, f, m, c in rows]), delimiter=",",
               header="episode,frame,mask_delta_pos,control_delta_pos", comments="", fmt=["%d", "%d", "%.4f", "%.4f"])
    if example is not None:
        from PIL import Image
        to = lambda t: (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(np.concatenate([to(example[0]), to(example[1])], 1)).save(out / "example_mask.png")
    print(f"saved {out / 'occlusion.csv'} and example_mask.png")


if __name__ == "__main__":
    main()

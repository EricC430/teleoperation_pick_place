"""H6: does the wrist-turn grasp follow the cup's distance, or where the front-left camera sits? (2026-10-13)

In training, all 46 far_wrist A1 demos (r >= 35.5, recorded 10-08/09) turn the wrist, and that session's front-left
view sits ~20 px (pan) off the 10-07 view; no near/mid demo exists in that view. Counterfactual on the predicted chunk:
  turn score = max predicted wrist_roll over the chunks planned at frames 30/60/90
  (training demos: turn >= +20.0, no turn <= +11.2 at the first gripper close -> threshold 15)
  orig        : as recorded
  FL shifted  : front-left moved by the 10-07 <-> far_wrist offset (SH), toward the OTHER session's view
  wrist gray  : wrist image -> 0.5 gray (an input the model never saw, same caveat as h1_swap2)
A model that uses the cup-to-arm relation keeps its decision under "FL shifted" (cup and arm move together).
Pass (for the augmented retrain, docs/handoff/2026-10-13_aug_retrain.md): normal_A1 FL shifted 0/13 turn and its
median within ~2 of orig; far_wrist FL shifted keeps the orig turn count.
Probe episodes are training episodes (the augmented model also trained on shifted copies of them) -- the closed-loop
turn rate by distance band is the real test.

ntfA1 100k (e97526c13766), 2026-10-13:
  normal_A1 orig 0/13 (median +9.3) | FL shifted 3/13 (+13.3) | wrist gray 0/13 (+9.2)
  far_wrist orig 10/12 (+42.1)      | FL shifted 10/12 (+40.3) | wrist gray 10/12 (+41.6)
  10-13 rollout ep0 (c1, r 24.3, far view, before the turn): orig +19.8..+25.8, FL -> 10-07 view +9.3..+10.1, wrist gray +23.2..+27.1

    uv run python scripts/input_reliance/h6_turn_cue.py --policy ericc430/act_omx_b1_paper-cup_60-13-10rcvry-46far_100k
    uv run python scripts/input_reliance/h6_turn_cue.py --policy <repo or checkpoint dir> --revision 40k
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

W, FL = "observation.images.wrist", "observation.images.front-left"
SH = (20.0, -4.4)  # far_wrist view = 10-07 view moved by this many px (align_camera.py --dry-run, 2026-10-12)
QUERY = (30, 60, 90)
SETS = [  # label, repo id, root, episodes, direction of the shift (+1: toward far_wrist view, -1: toward 10-07 view)
    ("normal_A1 (near/mid, 10-07 view)", "ericc430/omx_pick_place_pilot_paper_cup_normal_A1",
     ".cache/lerobot/omx_pick_place_pilot_paper_cup_normal_A1", list(range(13)), +1),  # ep 13-15 are excluded in training
    ("far_wrist (r>=35.5, far view)", "ericc430/omx_pick_place_pilot_paper_cup_far_wrist_A1",
     ".cache/lerobot/omx_pick_place_pilot_paper_cup_far_wrist_A1", list(range(0, 48, 4)), -1),
]


def checkpoint_dir(policy: str, revision: str | None) -> str:
    if Path(policy).is_dir():
        return policy
    from huggingface_hub import snapshot_download

    return snapshot_download(policy, revision=revision)


def shift(img: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    a = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    a = cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), a.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)
    return torch.from_numpy(a.astype(np.float32) / 255).permute(2, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="Hub repo id or local pretrained_model dir")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()
    ck = checkpoint_dir(args.policy, args.revision)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pol = get_policy_class("act").from_pretrained(ck, config=PreTrainedConfig.from_pretrained(ck)).eval().to(dev)
    pre, post = make_pre_post_processors(pol.config, pretrained_path=ck)

    def roll_max(item) -> float:
        b = pre({k: (v.unsqueeze(0).to(dev) if isinstance(v, torch.Tensor) else v) for k, v in item.items()})
        with torch.no_grad():
            return float(post(pol.predict_action_chunk(b)[0].cpu()).numpy()[:, 4].max())

    def score(ds, cond: str, sign: int) -> float:
        out = []
        for q in QUERY:
            it = dict(ds[q])
            if cond == "FL shifted":
                it[FL] = shift(it[FL], sign * SH[0], sign * SH[1])
            elif cond == "wrist gray":
                it[W] = torch.full_like(it[W], 0.5)
            out.append(roll_max(it))
        return max(out)

    print(f"policy {args.policy}{'@' + args.revision if args.revision else ''}")
    for label, rid, root, eps, sign in SETS:
        res = {c: [] for c in ("orig", "FL shifted", "wrist gray")}
        for ep in eps:
            ds = LeRobotDataset(rid, root=root, episodes=[ep])
            for c in res:
                res[c].append(score(ds, c, sign))
        print(f"== {label}; FL shifted toward the {'far_wrist' if sign > 0 else '10-07'} view")
        for c, v in res.items():
            v = np.array(v)
            print(f"  {c:11s} turn(>15) {int((v > 15).sum())}/{len(v)}  median {np.median(v):+.1f}  {np.round(v, 1).tolist()}")


if __name__ == "__main__":
    main()

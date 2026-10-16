"""H7: at the occluded placements c17 / c36, which camera turns the arm toward the cup? (2026-10-16)

10-16 closed loop (docs/meeting/2026-10-16_closed_loop.md section 3-1): at c17 / c36 the cup is hidden from the
front-left view at frame 0. The three earlier runs turn shoulder_pan to c16's side (~ +34, the wrong side); only
fldrop30-augfl turns to the cup's side (c17 ~ -33, c36 ~ -16). Counterfactual on those recorded episodes:
at each replanning frame (0, 30, 60, ... = where N=30 closed loop planned a new chunk), re-predict the chunk with
  orig       : as recorded
  wrist gray : wrist image -> 0.5 gray (an input the model never saw, same caveat as h1_swap2)
  wrist c16  : wrist image <- the same frame of this run's c16 episode (cup on the other side)
  FL black   : front-left -> black (what fl dropout trained on, in the 30 frames before the first close)
  FL c16     : front-left <- the same frame of this run's c16 episode
and report the predicted shoulder_pan at chunk step 29 (the last action executed before the next replan).
If the wrist turns the arm, "wrist gray" / "wrist c16" should pull the pan back toward c16's side (positive)
while "FL black" keeps it on the cup's side. Closed-loop frames after the arm has moved already carry the
decision in the state, so the early frames (before the pan splits) are the informative ones.

2026-10-16 (predicted pan at step 29; frames 0, 30, 60, ...):
  augfl c17    orig  +0.2 -16.9 -27.0 -32.1 | wrist gray -12.4 -16.8 -25.2 -27.8 | wrist c16 -0.1 -5.5 -24.8 +5.2
               FL black +0.2 +10.1 -21.6 -31.2 | FL c16 +0.5 +9.3 -5.7 -27.4
  augfl c36    orig (f60..180) +5.7 +1.1 +5.8 -3.1 -17.4 | wrist gray +1.3 +0.3 -2.0 -6.4 -17.3
               wrist c16 +16.8 +25.8 +28.1 +23.4 -4.9 | FL black -3.8 -0.9 +3.0 -9.1 -16.2 | FL c16 +9.5 +7.4 +8.7 +0.8 -6.0
  fldrop50: wrist gray slows its wrong-side (positive) turn at both c17 and c36; FL black barely changes it.
  orig vs recorded differ by up to ~10 (c36 augfl f60 / f120), from video decoding: smaller changes are noise.

    uv run python scripts/input_reliance/h7_occluded_reach.py
"""
import numpy as np
import torch
from huggingface_hub import snapshot_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

W, FL = "observation.images.wrist", "observation.images.front-left"
ROOT = "D:/hf/lerobot/ericc430/"
RUNS = [  # label, policy repo, rollout dataset (same day, same camera and brightness)
    ("augfl", "ericc430/act_omx_b1_paper-cup_60-13-10rcvrycut-46far-augfl-fldrop30_100k",
     "rollout_omx_b1_papercup_60_13_10rcvrycut_46far_augfl_fldrop30_100k_nas30_paper_cup_20261016_152018"),
    ("fldrop50", "ericc430/act_omx_b1_paper-cup_60-13-10rcvrycut-46far-aug-fldrop50_100k",
     "rollout_omx_b1_papercup_60_13_10rcvrycut_46far_aug_fldrop50_100k_nas30_paper_cup_20261016_142101"),
]
DONOR = 15  # c16
EPS = [(16, "c17"), (35, "c36")]
FRAMES = range(0, 271, 30)
CONDS = ("orig", "wrist gray", "wrist c16", "FL black", "FL c16")


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for label, repo, d in RUNS:
        ck = snapshot_download(repo)
        pol = get_policy_class("act").from_pretrained(ck, config=PreTrainedConfig.from_pretrained(ck)).eval().to(dev)
        pre, post = make_pre_post_processors(pol.config, pretrained_path=ck)

        def pan29(item) -> float:
            b = pre({k: (v.unsqueeze(0).to(dev) if isinstance(v, torch.Tensor) else v) for k, v in item.items()})
            with torch.no_grad():
                return float(post(pol.predict_action_chunk(b)[0].cpu()).numpy()[29, 0])

        donor = LeRobotDataset(d, root=ROOT + d, episodes=[DONOR])
        print(f"== {label} ({repo})")
        for ep, pid in EPS:
            ds = LeRobotDataset(d, root=ROOT + d, episodes=[ep])
            n = min(len(ds), len(donor))
            rows = {c: [] for c in CONDS}
            rec = []
            fr = [f for f in FRAMES if f + 29 < n]
            for f in fr:
                it = dict(ds[f])
                dn = donor[f]
                rec.append(float(ds[f + 29]["action"][0]))
                for c in CONDS:
                    x = dict(it)
                    if c == "wrist gray":
                        x[W] = torch.full_like(x[W], 0.5)
                    elif c == "wrist c16":
                        x[W] = dn[W]
                    elif c == "FL black":
                        x[FL] = torch.zeros_like(x[FL])
                    elif c == "FL c16":
                        x[FL] = dn[FL]
                    rows[c].append(pan29(x))
            print(f"  {pid} ep{ep}  frames {list(fr)}")
            print(f"    {'recorded':10s} " + " ".join(f"{v:6.1f}" for v in rec))
            for c, v in rows.items():
                print(f"    {c:10s} " + " ".join(f"{x:6.1f}" for x in v))
        del pol
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

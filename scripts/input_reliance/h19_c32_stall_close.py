"""H19: 11/17 c32 the wrist camera rests on the transparent part of the bottle and the aimfix28 model never closes
(stalled 35.8 s). Did the aimfix data teach "close-up transparent view -> do not close yet"? (2026-11-18)

Boyu's hypothesis A (worklog 2026-11-17_aimfix_closed_loop.md section 2): the aimfix_A1_cut episodes start with the
wrist close to the transparent bottle and the human does NOT close (corrects first); normal demos close as soon as the
camera touches the bottle. Boyu 11/18: c32 already touches the bottle, it cannot move further in -> close is the only
right action. Counterpoint (Boyu): many transparent close-up views still get closed.

Frames (offline, one predicted 100-step chunk per frame, no temporal ensembling):
  S17  11/17 aimfix28 rollout ep27 (c32), frames 135..672 every 15 (the stall)
  P15  11/15 M-full rollout ep27 (c32), the 30 frames before its first close onset, every 3
  CTL  every other episode of both rollouts: the frame 15 before that episode's own first close onset
Models: mfull = 8c42b7e0652e (cup129_bottle135_augfl_fldrop_p30_w30 / 200k), aimfix = ca92727ebcff
(cup129_bottle135_aimfix28_twoobj10_augfl_fldrop_p30_w30 / 200k), local checkpoints (hash checked at load).
"plans close" = predicted gripper action (dim 5) drops below 75 (h18's open threshold) within chunk steps 0..29 (2 s).

Reading rule, fixed before the run:
  sanity   aimfix on S17 -> no close on most frames, mfull on P15 -> close on most frames (= what each did online).
           If either fails, one offline chunk does not represent the TE-executed behaviour -> stop, no conclusion.
  A for    mfull on S17 closes on most frames AND aimfix on P15 does not close on most frames.
  A against mfull on S17 does not close either -> the c32 frames themselves, not the aimfix data.
  mixed    anything else -> describe only.
  "most" = >= 2/3 of the frames of that set. CTL is description only (which views the models close on); whether a CTL
  view is a transparent close-up is judged by Boyu on the wrist sheet, not by this script.
Frames of one episode are not independent; c32 is one placement, two episodes.

    docker exec -w /workspace lerobot-dev python scripts/input_reliance/h19_c32_stall_close.py
Writes data/logs/input_reliance/h19_*.csv / jpg only.
"""
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/logs/input_reliance"
HUB = REPO / "data/huggingface/lerobot/ericc430"
W, FL = "observation.images.wrist", "observation.images.front-left"
OPEN, HORIZON, EP = 75.0, 30, 27
ROLL = {  # label: (Hub dataset folder, episode_meta csv = original local name)
    "11-17 aimfix": ("rollout_omx_aimfix28_twoobj10_200k_nas1_te-001_plastic_bottle_20261117_184157",
                     "rollout_omx_aimfix_cup129_bottle135_aimfix28_twoobj10_200k_nas1_te-001_plastic_bottle_20261117_184157"),
    "11-15 mfull": ("rollout_omx_mfull_augfl_fldrop30_200k_nas1_te-001_plastic_bottle_20261115_183543",
                    "rollout_omx_mfull_cup129_bottle135_augfl_fldrop30_200k_nas1_te-001_plastic_bottle_20261115_183543"),
}
MODELS = {
    "mfull": ("data/train/cup129_bottle135_augfl_fldrop_p30_w30/checkpoints/200000/pretrained_model", "8c42b7e0652e"),
    "aimfix": ("data/train/cup129_bottle135_aimfix28_twoobj10_augfl_fldrop_p30_w30/checkpoints/200000/pretrained_model", "ca92727ebcff"),
}
dev = "cuda" if torch.cuda.is_available() else "cpu"


def first_close(g):
    """First frame where the gripper action, after being open (>= OPEN), falls below OPEN. None if never."""
    was_open = False
    for k, v in enumerate(g):
        if v >= OPEN:
            was_open = True
        elif was_open:
            return k
    return None


def frames():
    out = []
    for rl, (folder, csv) in ROLL.items():
        ds = LeRobotDataset("ericc430/" + folder, root=HUB / folder)
        cols = ds.hf_dataset.with_format(None).select_columns(["episode_index", "action"])
        epi = np.array(cols["episode_index"])
        act = np.stack(cols["action"])
        meta = pd.read_csv(REPO / "episode_meta" / f"{csv}.csv", encoding="utf-8-sig", comment="#").dropna(subset=["episode_index"])
        meta = meta.set_index(meta.episode_index.astype(int))
        for e in np.unique(epi):
            idx = np.flatnonzero(epi == e)
            k0 = first_close(act[idx, 5])
            pid, oc = meta.placement_id.get(e, "?"), meta.outcome.get(e, "?")
            if rl == "11-17 aimfix" and e == EP:
                sel, set_ = list(range(135, min(673, len(idx)), 15)), "S17"
            elif rl == "11-15 mfull" and e == EP:
                sel, set_ = ([] if k0 is None else list(range(max(k0 - 30, 0), k0, 3))), "P15"
            else:
                sel, set_ = ([] if k0 is None or k0 < 15 else [k0 - 15]), "CTL"
            for f in sel:
                out.append(dict(set=set_, rollout=rl, ep=int(e), pid=pid, outcome=oc, frame=f, close_onset=k0,
                                item=ds[int(idx[f])]))
        print(f"{rl}: {len(out)} frames so far", flush=True)
    return out


def load_model(path, want):
    p = REPO / path
    h = hashlib.sha256((p / "model.safetensors").read_bytes()).hexdigest()[:12]
    assert h == want, f"{path}: hash {h} != {want}"
    pol = get_policy_class("act").from_pretrained(p, config=PreTrainedConfig.from_pretrained(p)).eval().to(dev)
    pre, post = make_pre_post_processors(pol.config, pretrained_path=p)
    return pol, pre, post


def chunk(model, item):
    pol, pre, post = model
    b = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in dict(item).items()}
    b = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in pre(b).items()}
    with torch.no_grad():
        return post(pol.predict_action_chunk(b)[0].cpu()).numpy()


def hwc(t):
    return (t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def sheet(rows, fr):
    """Wrist view of every S17 / P15 / CTL frame, labelled with both models' plan, for Boyu to judge transparency."""
    tiles = []
    for r, f in zip(rows, fr):
        im = cv2.resize(hwc(f["item"][W]), (320, 240))
        txt = [f"{r['set']} {r['rollout']} ep{r['ep']} {r['pid']} f{r['frame']} ({r['outcome']})",
               "mfull {} aimfix {}".format(*["close@%d" % r[m + "_step"] if r[m + "_close"] else "open" for m in MODELS])]
        for i, t in enumerate(txt):
            y = 18 + 20 * i
            cv2.putText(im, t, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
            cv2.putText(im, t, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
        tiles.append(im)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
    cv2.imwrite(str(OUT / "h19_wrist_sheet.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fr = frames()
    rows = [{k: f[k] for k in ("set", "rollout", "ep", "pid", "outcome", "frame", "close_onset")} for f in fr]
    for name, (path, want) in MODELS.items():
        m = load_model(path, want)
        for r, f in zip(rows, fr):
            g = chunk(m, f["item"])[:, 5]
            below = np.flatnonzero(g[:HORIZON] < OPEN)
            r[name + "_close"] = bool(len(below))
            r[name + "_step"] = int(below[0]) if len(below) else -1
            r[name + "_gmin30"] = float(g[:HORIZON].min())
            r[name + "_g0"] = float(g[0])
        del m
        torch.cuda.empty_cache()
        print(f"model {name} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "h19_c32_stall_close.csv", index=False)
    sheet(rows, fr)
    print("\nshare of frames whose chunk plans a close within 2 s:")
    for s in ("S17", "P15", "CTL"):
        d = df[df.set == s]
        print(f"  {s} (n={len(d)}): " + "  ".join(f"{m} {d[m + '_close'].sum()}/{len(d)}" for m in MODELS))
    for s in ("S17", "P15"):
        print(f"\n{s} per frame:")
        print(df[df.set == s][["frame", "mfull_close", "mfull_gmin30", "aimfix_close", "aimfix_gmin30"]].to_string(index=False))
    c = df[df.set == "CTL"]
    print("\nCTL per rollout:")
    for rl, d in c.groupby("rollout"):
        print(f"  {rl} (n={len(d)}): " + "  ".join(f"{m} {d[m + '_close'].sum()}" for m in MODELS))
    print(f"\nwrote {OUT}/h19_c32_stall_close.csv, h19_wrist_sheet.jpg")


if __name__ == "__main__":
    sys.exit(main())

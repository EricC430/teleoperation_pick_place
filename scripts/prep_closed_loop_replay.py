"""Episodes of a LeRobot dataset -> sim/physics_replay.py inputs: joint radians from the touch-pinned fit (offsets,
sag), gripper from the action, cup on its placement point, world_from_link0 from the fit.

    python scripts/prep_closed_loop_replay.py <out dir> [<dataset name or path>] [--root <lerobot root>] [--episodes 0,3]

Default dataset: the 10-14 15:12 closed loop. Any machine: the dataset is <dataset> itself if that is a directory, else
<root>/ericc430/<dataset> (or <root>/<dataset>), with root = --root, else $HF_LEROBOT_HOME, else the first that exists
of <repo>/data/huggingface/lerobot (Linux PC), D:/hf/lerobot (Windows sim PC), ~/.cache/huggingface/lerobot.
Rollouts (placement c<n>) and demos (t<n>) both work: placements come from configs/placements/campA_136sym_*.csv,
outcome from episode_meta/<dataset>.csv. Reads parquet with pandas if installed, else pyarrow (Python 3.8 is fine).
"""
import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "sim"), str(REPO)]
import joint_mapping as jm  # noqa: E402
import solve_link_markers as SL  # noqa: E402
import touch_test as TT  # noqa: E402

DEFAULT = "rollout_omx_b1_papercup_60_13_10rcvrycut_46far_aug_fldrop30_100k_nas30_paper_cup_20261014_151208"
ROOTS = (REPO / "data/huggingface/lerobot", Path("D:/hf/lerobot"), Path.home() / ".cache/huggingface/lerobot")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("out")
ap.add_argument("dataset", nargs="?", default=DEFAULT)
ap.add_argument("--root", default=os.environ.get("HF_LEROBOT_HOME"))
ap.add_argument("--episodes", default="", help="comma list; default all in episode_meta")
args = ap.parse_args()


def dataset_dir(name, root):
    if Path(name).is_dir():
        return Path(name)
    roots = [Path(root)] if root else [r for r in ROOTS if r.is_dir()]
    for r in roots:
        for d in (r / "ericc430" / name, r / name):
            if (d / "meta").is_dir():
                return d
    raise SystemExit(f"dataset {name} not found under {[str(r) for r in roots]}; pass --root or a path")


def placements():
    """placement id -> (x, y) m, under both spellings: eval-close_001 / c1, train_022 / t22, eval-open_003 / o3."""
    pl = {}
    for f in sorted((REPO / "configs/placements").glob("campA_136sym_*_*.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            xy = (float(r["x_cm"]) / 100, float(r["y_cm"]) / 100)
            pid = r["placement_id"]; pl[pid] = xy
            for long, short in (("eval-close_", "c"), ("train_", "t"), ("eval-open_", "o")):
                if pid.startswith(long):
                    pl[short + str(int(pid[len(long):]))] = xy
    return pl


def read_frames(root: Path):
    cols = ["episode_index", "frame_index", "observation.state", "action"]
    files = sorted(glob.glob(str(root / "data/*/*.parquet")))
    try:
        import pandas as pd
        d = pd.concat(pd.read_parquet(f, columns=cols) for f in files)
        return {c: d[c].tolist() for c in cols}
    except ImportError:
        import pyarrow.parquet as pq
        out = {c: [] for c in cols}
        for f in files:
            t = pq.read_table(f, columns=cols).to_pydict()
            for c in cols:
                out[c] += t[c]
        return out


ROOT = dataset_dir(args.dataset, args.root); RID = ROOT.name
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
m = TT.FitModel(TT.REPO / "calibration/2026-10-14_link_marker_fit.json")
pl = placements()
meta = {int(r["episode_index"]): r for r in csv.DictReader(open(REPO / f"episode_meta/{RID}.csv", encoding="utf-8-sig"))}
eps = [int(e) for e in args.episodes.split(",")] if args.episodes else sorted(meta)
d = read_frames(ROOT)
ei = np.array(d["episode_index"]); fi = np.array(d["frame_index"])


def qbody(st):
    q = jm.SCALE_RAD_PER_UNIT["shoulder_lift"] * st[:5] + m.off
    tau = SL.gravity_torques(SL.chain_transforms(q, m.chain), m.inert, np.array([0, 0, -9.81]) if m.g_in_base else m.base[:3, :3].T @ np.array([0, 0, -9.81]))
    q = q.copy(); q[1:4] += m.sag * tau
    return q


for e in eps:
    md = meta[e]
    idx = np.where(ei == e)[0]; idx = idx[np.argsort(fi[idx])]
    st = np.array([d["observation.state"][i] for i in idx], float); ac = np.array([d["action"][i] for i in idx], float)
    ed = OUT / f"cl_ep{e:02d}"; ed.mkdir(exist_ok=True)
    json.dump({"dataset": RID, "episode": e, "fps": 15, "placement": md["placement_id"], "outcome": md["outcome"],
               "cup_xy_m": pl[md["placement_id"]], "state": st.tolist(), "action": ac.tolist(),
               "q_body_rad": [qbody(s).tolist() for s in st]}, open(ed / "episode.json", "w"))
    json.dump({"gripper_target_rad_from_action": [float(jm.lerobot_to_urdf_rad(np.r_[np.zeros(5), g])[5]) for g in ac[:, 5]],
               "gripper_rad_from_state": [float(jm.lerobot_to_urdf_rad(np.r_[np.zeros(5), g])[5]) for g in st[:, 5]]}, open(ed / "gripper.json", "w"))
json.dump({"fit": {"name": "touch-pinned link-marker fit (calibration/2026-10-14_link_marker_fit.json)", "world_from_link0": m.base.tolist()}},
          open(OUT / "fit_candidate.json", "w"), indent=1)
print("prepared", len(eps), "episodes of", ROOT, "->", OUT)

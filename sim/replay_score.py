"""Did the physics replay grasp the object? Same rule as scripts/score_closed_loop_replay.py (pure python, numpy only).

A close = gripper action target < 0.2 rad for >= 5 frames. The object counts as grasped in sim on that close when its
origin (base centre) is >= 5 cm above the table for >= 10 frames during the close (+1 s).
Real: in a success episode the LAST close is the grasp and every earlier one missed; in a failure every close missed.
Bin (2026-11-01, when the log has one): `in_bin_final` = the object's origin is inside the bin at the last frame; the
real success definition is "lifted AND dropped in the bin" (docs/experiment_spec.md), so `task_success` = both.

    python3 sim/replay_score.py <episode dir with episode.json, gripper.json> <physics_log.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shell_mesh import in_bin  # noqa: E402

CLOSED_RAD, MIN_CLOSE_FRAMES, LIFT_M, MIN_LIFT_FRAMES = 0.2, 5, 0.05, 10


def closes(a):
    ev, k = [], 0
    while k < len(a):
        if a[k] < CLOSED_RAD:
            j = k
            while j < len(a) and a[j] < CLOSED_RAD:
                j += 1
            if j - k >= MIN_CLOSE_FRAMES:
                ev.append((k, j))
            k = j
        else:
            k += 1
    return ev


def score(ep_dir, log_path):
    ep_dir = Path(ep_dir)
    E = json.load(open(ep_dir / "episode.json"))
    D = json.load(open(log_path)); L = D["log"]; B = D.get("bin")
    fr = np.array([r["frame"] for r in L]); obj = np.array([r["cup_xyz"] for r in L])
    ev = closes(np.array(json.load(open(ep_dir / "gripper.json"))["gripper_target_rad_from_action"]))
    real_ok = E.get("outcome") == "success"
    att = []
    for i, (s, e) in enumerate(ev):
        m = (fr >= s) & (fr < e + E["fps"])
        lifted = int((obj[m, 2] > LIFT_M).sum())
        att.append({"start": int(s), "end": int(e), "real_grasp": bool(real_ok and i == len(ev) - 1),
                    "sim_grasp": lifted >= MIN_LIFT_FRAMES, "lifted_frames": lifted})
    sim_any = any(a["sim_grasp"] for a in att)
    inb = bool(in_bin(obj[-1], B)) if (B and len(obj)) else None
    return {"episode": E["episode"], "placement": E.get("placement"), "real_success": real_ok, "n_closes": len(ev),
            "sim_success": sim_any, "first_close_agree": bool(att[0]["real_grasp"] == att[0]["sim_grasp"]) if att else None,
            "max_obj_z_m": float(obj[:, 2].max()) if len(obj) else None, "closes": att,
            "in_bin_final": inb, "task_success": (sim_any and inb) if inb is not None else None,
            "final_obj_xyz": [float(v) for v in obj[-1]] if len(obj) else None}


if __name__ == "__main__":
    print(json.dumps(score(sys.argv[1], sys.argv[2]), indent=1))

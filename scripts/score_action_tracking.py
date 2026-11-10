#!/usr/bin/env python3
"""S10 stage A: how far does the sim arm get when it is DRIVEN BY THE RECORDED ACTION, against where the real arm went?

Input: episode dirs prepared by scripts/prep_closed_loop_replay.py (episode.json has q_body_rad = the fit's link angles
of the real observation.state, the reference) and a physics_replay.py output with target_source=action[_nosag]
(physics_log.json: q_sim per frame, dof_names). Output: per joint, the sim-minus-real angle in degrees (median, p90, max,
signed mean) over all frames and over the frames where the arm moves fastest (top quartile of |d q_ref/dt|), plus the
sim command minus the reference (the part that is the action itself being ahead of the state, not sim error).

    python3 scripts/score_action_tracking.py <episode dir> <physics_log.json> [--json out.json]

Reading it: (sim - ref) is what a closed loop would see as tracking error. (cmd - ref) is the real servo's own lag
(action leads state), which the sim must reproduce through its drive, not remove.
"""
import argparse
import json
from pathlib import Path

import numpy as np

BODY = ("joint1", "joint2", "joint3", "joint4", "joint5")
ap = argparse.ArgumentParser()
ap.add_argument("ep"); ap.add_argument("log"); ap.add_argument("--json")
a = ap.parse_args()
E = json.load(open(Path(a.ep) / "episode.json")); L = json.load(open(a.log))
names = L["dof_names"]; idx = [names.index(n) for n in BODY]
ref = np.degrees(np.array(E["q_body_rad"]))
fr = np.array([r["frame"] for r in L["log"]])
sim = np.degrees(np.array([[r["q_sim"][i] for i in idx] for r in L["log"]]))
cmd = np.degrees(np.array([[r["q_cmd"][i] for i in idx] for r in L["log"]]))
ref = ref[fr]
speed = np.abs(np.gradient(ref, axis=0))
out = {"episode": E["episode"], "dataset": E["dataset"], "target_source": L["args"].get("target_source", "state"),
       "args": {k: v for k, v in L["args"].items() if k in ("motors", "stiff_mult", "k_rule", "k_deg", "vel_ff", "cup_dx_mm", "selfcol")},
       "n_frames": len(fr), "joints": {}}
for j, n in enumerate(BODY):
    d = sim[:, j] - ref[:, j]; c = cmd[:, j] - ref[:, j]; fast = speed[:, j] >= np.quantile(speed[:, j], 0.75)
    out["joints"][n] = {"sim_minus_ref_p50": float(np.median(np.abs(d))), "sim_minus_ref_p90": float(np.quantile(np.abs(d), 0.9)),
                        "sim_minus_ref_max": float(np.abs(d).max()), "sim_minus_ref_mean_signed": float(d.mean()),
                        "fast_p50": float(np.median(np.abs(d[fast]))), "cmd_minus_ref_p50": float(np.median(np.abs(c)))}
    print(f"{n}  |sim-ref| p50 {out['joints'][n]['sim_minus_ref_p50']:.2f}  p90 {out['joints'][n]['sim_minus_ref_p90']:.2f}  "
          f"max {out['joints'][n]['sim_minus_ref_max']:.2f}  signed {out['joints'][n]['sim_minus_ref_mean_signed']:+.2f}  "
          f"fastQ p50 {out['joints'][n]['fast_p50']:.2f}  |cmd-ref| p50 {out['joints'][n]['cmd_minus_ref_p50']:.2f} deg")
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)

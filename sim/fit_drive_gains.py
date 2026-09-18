"""S5 gap 1 — fit the drive gains against real recorded trajectories (D029, sim/README.md gap #1).

Input is `traj.npz` from `scripts/s5_prepare_replay.py` (host side). Each gain candidate gets its own
env, so a whole grid replays in one pass:

  stiffness = stall_torque / tracking_error      (tracking_error swept, current rule = 5 deg)
  damping   = damping_fraction * stiffness       (damping_fraction swept, current rule = 0.05)

For every frame t: command action[t], step one dataset frame (1/fps s), compare the sim joint angles
with the REAL state[t+1]. Scored on the five body joints only — the real gripper stalls on the object
and there is no object here, so gripper error would reward the wrong thing.

Read the result against `real_tracking.json` (same script, host side): the real follower's own
|action[t] - state[t+1]|. A sim that tracks its command much better than the real arm does is too
stiff even when its error against state is small.

    ./sim/run_in_container.sh fit_drive_gains.py --headless \
        --traj /workspace/test_isaaclab/omx_sim/s5_prep/uvc_60/traj.npz --episodes 0,10,20,30,40,50

🔴 Precondition: `joint_mapping.py` signs / zero offsets are [未確認]. A wrong sign turns this into
   fitting gains against a mirrored trajectory — gravity loads land on the wrong side and the best
   candidate means nothing. Run the S4 §5-1 five-pose comparison first, or read the result as a smoke
   test of the machinery only.
🔴 This script never writes `omx_constants.py`. It prints and saves a ranking; a human decides.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--traj", required=True, help="traj.npz from scripts/s5_prepare_replay.py")
parser.add_argument("--episodes", default=None, help="comma-separated episode indices (default: all in the npz)")
parser.add_argument("--tracking-error-deg", default="0.5,1,2,5,10", help="stiffness = stall_torque / this")
parser.add_argument("--damping-fraction", default="0.02,0.05,0.1,0.2,0.5", help="damping = this * stiffness")
parser.add_argument("--dt", type=float, default=1.0 / 120.0, help="physics dt (s); must divide 1/fps")
parser.add_argument("--max-frames", type=int, default=None, help="truncate each episode (for a quick run)")
parser.add_argument("--out", default="/workspace/test_isaaclab/omx_sim/out/fit_drive_gains")
parser.add_argument("--dry-run", action="store_true", help="print the grid and the data summary, do not start Isaac Sim")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import joint_mapping as JM  # noqa: E402
import omx_constants as K  # noqa: E402

BODY = 5  # first five columns of the npz are the body joints

data = np.load(args.traj)
fps = int(data["fps"])
ep_all = data["episode_index"]
episodes = sorted({int(e) for e in args.episodes.split(",")}) if args.episodes else sorted(np.unique(ep_all).tolist())
substeps = round((1.0 / fps) / args.dt)
if abs(substeps * args.dt - 1.0 / fps) > 1e-9:
    raise SystemExit(f"--dt {args.dt} does not divide 1/fps = {1.0 / fps}")
if tuple(data["joint_names"].tolist()) != JM.URDF_NAMES:
    raise SystemExit(f"🔴 npz joint order {data['joint_names'].tolist()} != {JM.URDF_NAMES}")

grid = list(itertools.product([float(x) for x in args.tracking_error_deg.split(",")],
                              [float(x) for x in args.damping_fraction.split(",")]))
print(f"traj      : {args.traj}  ({fps} fps, {substeps} physics steps per frame at dt={args.dt:.5f})")
print(f"source    : {data['source']}   mapping: {data['mapping_note']}")
print(f"episodes  : {episodes}")
print(f"grid      : {len(grid)} candidates = one env each")
print(f"current rule: tracking_error {math.degrees(K.GAIN_TRACKING_ERROR_RAD):.1f} deg, damping_fraction {K.GAIN_DAMPING_FRACTION}")
if args.dry_run:
    for i, (te, df) in enumerate(grid):
        print(f"  env {i:>2}: tracking_error {te:5.2f} deg  damping_fraction {df:.3f}")
    raise SystemExit(0)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

import omx_scene_cfg as SC  # noqa: E402

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args.dt, device=args.device))
scene = InteractiveScene(SC.OmxArmOnlySceneCfg(num_envs=len(grid), env_spacing=1.0))
sim.reset()
robot = scene["robot"]
dev = sim.device
n_env = len(grid)

ids = [robot.find_joints(n)[0][0] for n in JM.URDF_NAMES]
mimic_id = robot.find_joints(K.MIMIC_JOINT[0])[0][0]
all_env = torch.arange(n_env, device=dev)

# per-env gains, per joint (radian convention, like omx_scene_cfg)
stiff = torch.zeros(n_env, len(ids), device=dev)
damp = torch.zeros(n_env, len(ids), device=dev)
for e, (te_deg, dfrac) in enumerate(grid):
    for j, joint in enumerate(K.JOINTS):
        stiff[e, j] = joint.motor.stall_torque_nm / math.radians(te_deg)
        damp[e, j] = dfrac * stiff[e, j]
robot.write_joint_stiffness_to_sim(stiff, joint_ids=ids, env_ids=all_env)
robot.write_joint_damping_to_sim(damp, joint_ids=ids, env_ids=all_env)

errors = [[] for _ in range(n_env)]  # per env: list of (frames, 5) abs errors in deg
limit_hits = 0
for ep in episodes:
    sel = ep_all == ep
    act = torch.tensor(data["action_rad"][sel], device=dev)
    st = torch.tensor(data["state_rad"][sel], device=dev)
    if args.max_frames:
        act, st = act[: args.max_frames + 1], st[: args.max_frames + 1]

    # start every env exactly at the real first state
    q0 = robot.data.default_joint_pos.clone()
    q0[:, ids] = st[0]
    q0[:, mimic_id] = K.MIMIC_JOINT[2] * st[0, 5]
    lim = robot.data.joint_pos_limits[0, ids]
    outside = ((st < lim[:, 0]) | (st > lim[:, 1])).any(dim=0)
    if outside.any():
        limit_hits += 1
        names = [JM.URDF_NAMES[j] for j in torch.nonzero(outside).flatten().tolist()]
        print(f"  ⚠️ ep {ep}: real state leaves the USD joint limits on {names} — clamped in sim (mapping 未確認?)")
    scene.reset()
    robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))

    target = q0.clone()
    ep_err = torch.zeros(n_env, len(act) - 1, BODY, device=dev)
    for t in range(len(act) - 1):
        target[:, ids] = act[t]
        for _ in range(substeps):
            robot.set_joint_position_target(target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(args.dt)
        ep_err[:, t] = (robot.data.joint_pos[:, ids[:BODY]] - st[t + 1, :BODY]).abs()
    for e in range(n_env):
        errors[e].append(torch.rad2deg(ep_err[e]).cpu().numpy())
    print(f"  ep {ep}: {len(act) - 1} frames replayed")

results = []
for e, (te_deg, dfrac) in enumerate(grid):
    err = np.concatenate(errors[e], axis=0)
    per_joint = {JM.URDF_NAMES[j]: {"p50": float(np.percentile(err[:, j], 50)),
                                    "p95": float(np.percentile(err[:, j], 95)),
                                    "max": float(err[:, j].max())} for j in range(BODY)}
    results.append({"tracking_error_deg": te_deg, "damping_fraction": dfrac,
                    "score_p95_all_body": float(np.percentile(err, 95)),
                    "p50_all_body": float(np.percentile(err, 50)), "joints": per_joint})
results.sort(key=lambda r: r["score_p95_all_body"])

print(f"\nsim joint angle vs REAL state[t+1], body joints, deg — ranked by p95 over all joints/frames")
print(f"{'rank':>4}{'track_err':>11}{'damp_frac':>11}{'p50':>8}{'p95':>8}   " + "".join(f"{n:>9}" for n in JM.URDF_NAMES[:BODY]))
for r_i, r in enumerate(results, start=1):
    per = "".join(f"{r['joints'][n]['p95']:9.2f}" for n in JM.URDF_NAMES[:BODY])
    print(f"{r_i:>4}{r['tracking_error_deg']:11.2f}{r['damping_fraction']:11.3f}{r['p50_all_body']:8.2f}{r['score_p95_all_body']:8.2f}   {per}")

best = results[0]
te_vals = sorted({g[0] for g in grid})
df_vals = sorted({g[1] for g in grid})
edge = best["tracking_error_deg"] in (te_vals[0], te_vals[-1]) or best["damping_fraction"] in (df_vals[0], df_vals[-1])
if edge:
    print("\n⚠️  best candidate sits on the EDGE of the grid — the optimum may lie outside it; widen before trusting it")
if limit_hits:
    print(f"⚠️  {limit_hits} episode(s) left the USD limits — suspect joint_mapping signs/offsets before the gains")
print("Compare the per-joint p95 above with real_tracking.json (the real arm's own lag-1 error).")
print("🔴 Nothing was written to omx_constants.py.")

os.makedirs(args.out, exist_ok=True)
out_path = os.path.join(args.out, "fit_drive_gains.json")
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump({"traj": args.traj, "source": str(data["source"]), "episodes": episodes, "fps": fps, "dt": args.dt,
               "mapping_note": str(data["mapping_note"]), "best_on_grid_edge": edge,
               "episodes_outside_limits": limit_hits, "ranking": results}, fh, indent=2, ensure_ascii=False)
print(f"wrote {out_path}")
simulation_app.close()

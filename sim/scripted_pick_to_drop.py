"""Scripted pick at a placement id and carry to the bin drop point, in the RL bin env; saves one joint trajectory.

Purpose: a reference motion for measuring the sim-to-real gap of reaching the drop point above the bin,
without a learned policy. The motion is a proportional Cartesian controller on the TCP, through the SAME
action interface and IK as the RL task: 4D action (dx, dy, dz, gripper), 0.5 cm per unit, clip 5;
DLS over joint1-4 from the measured joints; tool-down row faded out once the cube is held off the table.

    above (+8 cm, open) -> descend to the cube centre -> close -> lift +12 cm
    -> waypoint (10, -8, 50) cm over the arm -> drop point

All positions are in the pan-axis frame (`scene_constants.py`: origin = pan axis on the table, x ahead,
y left, z up). The placement comes from `docs/assets/placement_label_map_campA_136sym_20260908.csv`
(the set the 10-06/10-07 closed-loop runs used); 2026-10-08_paper_to_pan.md found paper and FK
coordinates agree to ~1 cm, so the paper coordinates are used as-is.

Dependency: the env is `Template-Pickup-Place-OMX-Bin-v0` from the RL repo
(`isaaclab_volume/pickup_place_direct_0203`, Boyu00/Picking-Lifting-RL-training, branch `omx-hold-gate`,
commit e6d15bc at the time of the first run). It is installed (editable) in the container's Isaac Sim
python, so this runs from anywhere:

    ./sim/run_in_container.sh scripted_pick_to_drop.py --placement c1 --headless \\
        --out /workspace/test_isaaclab/omx_sim/scripted_c1_traj.csv

Output CSV, one row per policy step (20 ms): joint1..joint5 and gripper_joint_1 in SIM radians (URDF
convention; `joint_mapping.py` converts to LeRobot `.pos`), TCP and cube positions in the pan frame (m),
and the cube edge (m). Saved: the first env that succeeded (held, cube < 8 cm from the target, no reset).

Known limits (2026-10-08, see the first run in the commit message):
- the cube is a DexCube, not the paper cup the real runs grasp (cup wall, ~3.4 cm from the cup centre);
- the waypoint puts joint4 near 98 deg (limit +-100) and the descent puts the TCP ~0.7 cm above the
  table: fine in sim, not margins a real arm with ~1 cm FK error should be sent through unchanged;
- 2026-10-08_paper_to_pan.md: FK error grows in poses far from the table, i.e. near the drop point.
"""

import argparse
import csv
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--placement", default="c1", help="short id in the campA_136sym label map")
parser.add_argument("--label-map", default="/workspace/test_isaaclab/omx_sim/placement_label_map_campA_136sym_20260908.csv")
parser.add_argument("--drop-cm", type=float, nargs=3, default=(0.0, -25.0, 45.0), help="pan frame, cm")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default="/workspace/test_isaaclab/omx_sim/scripted_traj.csv")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import pickup_place_direct_0203.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Template-Pickup-Place-OMX-Bin-v0"
WAYPOINT_PAN = (0.10, -0.08, 0.50)

with open(args.label_map, newline="") as f:
    row = next(r for r in csv.DictReader(f) if r["short_id"] == args.placement)
pick = (float(row["x_pan_cm"]) / 100.0, float(row["y_pan_cm"]) / 100.0)
drop = tuple(v / 100.0 for v in args.drop_cm)

N = args.num_envs
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
cfg.seed = args.seed
cfg.observation_noise_scale = 0.0
r, th = math.hypot(*pick), math.atan2(pick[1], pick[0])
cfg.object_radius_range = (r, r)
cfg.object_azimuth_range = (th, th)
cfg.randomize_object_yaw = False
cfg.target_box_pan = tuple((v, v) for v in drop)
env = gym.make(TASK, cfg=cfg).unwrapped
env.reset()
dev = env.device
pan = torch.tensor([cfg.pan_axis_xy[0], 0.0, 0.0], device=dev)


def tcp():
    return env._tcp_pose_w()[0] - env.scene.env_origins


def obj():
    return env.object.data.root_com_pose_w[:, :3] - env.scene.env_origins


log = []
reset_seen = torch.zeros(N, dtype=torch.bool, device=dev)


def go(phase, target, grip, steps):
    global reset_seen
    for _ in range(steps):
        prev = env.episode_length_buf.clone()
        a = torch.clamp((target - tcp()) / 0.005, -5, 5)
        env.step(torch.cat([a, torch.full((N, 1), grip, device=dev)], dim=1))
        reset_seen |= env.episode_length_buf < prev
        q = env.robot.data.joint_pos[:, env._arm_joint_indices + list(env._gripper_joint_idx)]
        log.append((phase, q.clone(), (tcp() - pan).clone(), (obj() - pan).clone()))


# object data is stale right after reset; settle a few steps with the gripper open
for _ in range(5):
    env.step(torch.cat([torch.zeros(N, 3, device=dev), torch.ones(N, 1, device=dev)], dim=1))
c0 = obj().clone()
edge = env.object_half_extents[:, 0] * 2
print(f"placement {args.placement} -> cube at (pan, cm) {[round(v, 2) for v in ((c0[0] - pan) * 100).tolist()]}, "
      f"edges {edge.min() * 100:.1f}-{edge.max() * 100:.1f} cm; drop point {args.drop_cm} cm")

go("above", c0 + torch.tensor([0.0, 0.0, 0.08], device=dev), 1.0, 40)
go("descend", c0, 1.0, 40)
go("close", c0, -1.0, 25)
go("lift", c0 + torch.tensor([0.0, 0.0, 0.12], device=dev), -1.0, 35)
go("over_arm", torch.tensor(WAYPOINT_PAN, device=dev) + pan, -1.0, 45)
goal = env.target_poses.clone()
go("drop_point", goal, -1.0, 55)

err = obj() - goal
held = env.hold_counter >= env._hold_steps
ok = held & (torch.norm(err, dim=1) < cfg.reward_settings["goal_tracking_threshold"]) & ~reset_seen
print(f"steps {len(log)}; reset mid-run {reset_seen.float().mean():.0%}; success {int(ok.sum())}/{N}")
print("env  edge_cm  cube-target (x, y, z cm)  held  ok")
for i in range(N):
    print(f"{i:3d}  {edge[i] * 100:5.1f}  {[round(v, 1) for v in (err[i] * 100).tolist()]}  {bool(held[i])}  {bool(ok[i])}")

if ok.any():
    i = int(torch.nonzero(ok)[0])
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "phase", "joint1", "joint2", "joint3", "joint4", "joint5", "gripper_joint_1",
                    "tcp_x_pan", "tcp_y_pan", "tcp_z", "cube_x_pan", "cube_y_pan", "cube_z", "cube_edge"])
        for k, (phase, q, t, c) in enumerate(log):
            w.writerow([k, phase] + [round(v, 5) for v in q[i].tolist()] + [round(v, 4) for v in t[i].tolist()]
                       + [round(v, 4) for v in c[i].tolist()] + [round(edge[i].item(), 4)])
    print(f"saved env {i} (edge {edge[i] * 100:.1f} cm), {len(log)} steps at {env.step_dt * 1000:.0f} ms -> {args.out}")
else:
    print("no env succeeded; nothing saved")

env.close()
app.close()

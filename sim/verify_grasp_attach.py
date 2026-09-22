"""Smoke test for `grasp_attach.ScriptedGraspAttach` against one real episode's gripper trace.

**What this proves, and no more:** the attach/detach state machine (threshold calibration,
falling+below/above edge detection, the kinematic pose-follow math) runs end-to-end and produces
one attach + one detach event at plausible frames for a real recorded gripper trace. **What this
does NOT prove:** that a real episode's OBJECT would actually be within `--attach-radius` of the
sim TCP at that frame once gap 1 (drive gains) and gap 4 (camera/scene geometry) are closed --
this repo has no independently-measured object position to check that against, and the arm's
gains are still provisional (see `sim/README.md`). To make the mechanism testable at all without
that missing ground truth, this script TELEPORTS the object to the sim's own computed TCP
position one frame before the predicted attach frame -- i.e. it manufactures the one condition
(proximity) the real world would have to supply, purely so the state machine has something to
attach to. Read the printed events as "the mechanism fires correctly", not as "grasping works".

[已查證 2026-09-18，對照 IsaacLab GitHub `main` 原始碼，非本機執行] `write_root_pose_to_sim(root_pose)`
takes shape `(N,7)` = pos+quat, `write_root_velocity_to_sim(root_velocity)` takes shape `(N,6)` =
lin+ang -- matches the calls below. Same "shape/order confirmed, exact pinned container version
not" caveat as `grasp_attach.py`'s frame-transform helpers. `robot.data.body_pos_w`/`body_quat_w`
and `obj.data.root_pos_w`/`root_quat_w` are confirmed-existing aliases in that same source read.

Needs the episode's parquet data copied into the container first -- same requirement and command
as `fit_drive_gains.py`'s docstring.

    ./sim/run_in_container.sh verify_grasp_attach.py \\
        --dataset-root /workspace/test_isaaclab/omx_sim/dataset --episode 0 --headless \\
        --out /workspace/test_isaaclab/omx_sim/grasp_attach_ep0.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset-root", required=True, help="dir containing data/chunk-*/file-*.parquet")
parser.add_argument("--episode", type=int, required=True)
parser.add_argument("--attach-radius", type=float, default=0.05)
parser.add_argument("--close-frac", type=float, default=0.6)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import joint_mapping as JM  # noqa: E402
import omx_constants as K  # noqa: E402
import omx_scene_cfg as SC  # noqa: E402
from grasp_attach import GraspAttachConfig, ScriptedGraspAttach, tcp_pose_w  # noqa: E402

DATASET_FPS = 15.0


def load_episode(dataset_root: str, episode: int):
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit(
            "pyarrow not importable -- install once in the container: /isaac-sim/python.sh -m pip install pyarrow"
        ) from e
    files = sorted(glob.glob(os.path.join(dataset_root, "data", "chunk-*", "file-*.parquet")))
    if not files:
        raise SystemExit(f"no parquet files under {dataset_root}/data/chunk-*/file-*.parquet")
    rows = []
    for f in files:
        d = pq.read_table(f, columns=["episode_index", "frame_index", "action", "observation.state"]).to_pydict()
        for ep, fi, act, st in zip(d["episode_index"], d["frame_index"], d["action"], d["observation.state"]):
            if ep == episode:
                rows.append((fi, act, st))
    if not rows:
        raise SystemExit(f"episode {episode} not found under {dataset_root}")
    rows.sort(key=lambda r: r[0])
    return [r[1] for r in rows], [r[2] for r in rows]


actions_deg, states_deg = load_episode(args.dataset_root, args.episode)
n_frames = len(actions_deg)
gripper_trace_deg = [row[5] for row in states_deg]
actions_rad = [JM.row_to_sim_rad(r) for r in actions_deg]
print(f"episode {args.episode}: {n_frames} frames")

# ---- predict the attach frame from the gripper trace ALONE (no proximity yet) -- purely to know
# which frame to teleport the object to the TCP at, one step early. See module docstring.
import numpy as np  # noqa: E402

g_open = float(np.percentile(gripper_trace_deg, 95))
g_closed = float(np.percentile(gripper_trace_deg, 5))
threshold = g_open - args.close_frac * (g_open - g_closed)
predicted_attach_frame = None
for i in range(1, n_frames):
    if gripper_trace_deg[i] < threshold and gripper_trace_deg[i] < gripper_trace_deg[i - 1]:
        predicted_attach_frame = i
        break
print(f"threshold={threshold:.2f} deg (g_open={g_open:.2f}, g_closed={g_closed:.2f})")
print(f"predicted attach frame (gripper trace only, no proximity yet): {predicted_attach_frame}")
if predicted_attach_frame is None:
    raise SystemExit("gripper trace never crosses below threshold while falling -- nothing to demo")


@configclass
class GraspDemoSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=1200.0, color=(1.0, 1.0, 1.0))
    )
    robot: ArticulationCfg = SC.omx_articulation_cfg("{ENV_REGEX_NS}/Robot")
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SC.DEFAULT_OBJECT_USD,
            scale=(0.01, 0.01, 0.01),  # trash_obj/*.usd metersPerUnit=0.01, see scene_constants.py
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=3.0),
        ),
        # parked well clear of the arm's reach until the scripted teleport, see module docstring
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.8, 0.5)),
    )


joint_names = [j.urdf_name for j in K.JOINTS]
scene_cfg = GraspDemoSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)
first_pose = {name: val for name, val in zip(joint_names, actions_rad[0])}
first_pose[K.MIMIC_JOINT[0]] = 0.0
scene_cfg.robot.init_state.joint_pos = first_pose

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
scene = InteractiveScene(scene_cfg)
sim.reset()

robot = scene["robot"]
obj = scene["object"]
joint_idx = [robot.joint_names.index(n) for n in joint_names]

grasp = ScriptedGraspAttach(GraspAttachConfig(attach_radius_m=args.attach_radius, close_frac=args.close_frac))
grasp.calibrate(gripper_trace_deg)

substeps = round(120.0 / DATASET_FPS)
target = robot.data.default_joint_pos.clone()

for t in range(n_frames):
    for k, i in enumerate(joint_idx):
        target[:, i] = actions_rad[t][k]
    for _ in range(substeps):
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    tcp_pos, tcp_quat = tcp_pose_w(robot)   # measured fingertip, not the link6/link7 pivots

    if t == predicted_attach_frame - 1:
        # manufacture proximity: teleport the object onto the TCP one frame early, see docstring
        pose = torch.cat([tcp_pos, tcp_quat]).unsqueeze(0)
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=tcp_pos.device))
        print(f"  frame {t}: teleported object to TCP {tcp_pos.cpu().tolist()} ahead of predicted attach")

    obj_pos = obj.data.root_pos_w[0]
    obj_quat = obj.data.root_quat_w[0]
    want_pos, want_quat = grasp.step(t, gripper_trace_deg[t], tcp_pos, tcp_quat, obj_pos, obj_quat)
    if grasp.attached:
        obj.write_root_pose_to_sim(torch.cat([want_pos, want_quat]).unsqueeze(0))
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=want_pos.device))

print(f"\n{len(grasp.events)} event(s):")
for e in grasp.events:
    extra = f" dist={e.dist_m*100:.1f}cm" if e.dist_m is not None else ""
    print(f"  frame {e.frame:>4}  {e.kind:<7}  gripper={e.gripper_reading:.2f}deg{extra}")

ok = len(grasp.events) >= 2 and grasp.events[0].kind == "attach" and grasp.events[1].kind == "detach" and not grasp.attached
print(f"\n{'✅' if ok else '🔴'} {'one clean attach->detach cycle, ends FREE' if ok else 'did NOT get a clean attach->detach cycle -- read the event log above'}")

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as fh:
    json.dump(
        {
            "source_episode_id": args.episode,
            "dataset_root": args.dataset_root,
            "threshold_deg": threshold,
            "attach_radius_m": args.attach_radius,
            "close_frac": args.close_frac,
            "predicted_attach_frame": predicted_attach_frame,
            "events": [{"frame": e.frame, "kind": e.kind, "gripper_deg": e.gripper_reading, "dist_m": e.dist_m} for e in grasp.events],
            "clean_cycle": ok,
        },
        fh,
        indent=2,
    )
print(f"wrote {args.out}")
import os
os._exit(0)

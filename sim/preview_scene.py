"""Build the OMX cell, step it, and save what each camera sees.

This is the "does it stand up" check for the scene — S4 §5-5 is explicit that geometric
alignment with the real cell is NOT possible yet (experiment_spec §3 is still blank), so what
this proves is narrower and worth stating exactly:

  * the converted arm loads, is an articulation, and holds itself up under gravity
  * the seeded placement list drives object positions in the sim frame
  * both cameras render at the recording resolution, in the dataset's declared ORDER

Run inside the isaac-lab container:

    ./sim/run_in_container.sh preview_scene.py --headless \
        --placements /workspace/test_isaaclab/omx_sim/placement_label_map_campA_20260831.csv \
        --place t1 --out /workspace/test_isaaclab/omx_sim/out
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--placements", default=None, help="placement_label_map CSV; object goes on one of its points")
parser.add_argument("--place", default=None, help="short_id to use (default: the first row)")
parser.add_argument("--object", default=None, help="override the object USD")
parser.add_argument("--object-scale", type=float, default=1.0)
parser.add_argument("--steps", type=int, default=120, help="physics steps before the picture is taken")
parser.add_argument("--out", default="/workspace/test_isaaclab/omx_sim/out")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

import omx_constants as K  # noqa: E402
import omx_scene_cfg as SC  # noqa: E402
import scene_constants as S  # noqa: E402


def save_rgb(tensor, path):
    """tensor: (H, W, 3|4) uint8 on the gpu."""
    arr = tensor[..., :3].detach().cpu().numpy()
    try:
        from PIL import Image

        Image.fromarray(arr).save(path)
        return path
    except ImportError:
        ppm = os.path.splitext(path)[0] + ".ppm"
        with open(ppm, "wb") as fh:
            fh.write(b"P6\n%d %d\n255\n" % (arr.shape[1], arr.shape[0]))
            fh.write(arr.tobytes())
        return ppm


os.makedirs(args.out, exist_ok=True)

# ---------------------------------------------------------------- object placement
place = None
if args.placements:
    placements = S.load_placements(args.placements)
    if args.place:
        matches = [p for p in placements if p.short_id == args.place or p.placement_id == args.place]
        if not matches:
            raise SystemExit(f"no placement '{args.place}' in {args.placements}")
        place = matches[0]
    else:
        place = placements[0]
    problems = S.check_placement_reachable(place)
    print(f"placement {place.short_id} ({place.placement_id}): "
          f"x={place.x_m*100:.1f}cm y={place.y_m*100:.1f}cm "
          f"r={place.radius_m*100:.1f}cm theta={place.theta_deg:.1f}deg"
          + (f"   🔴 {'; '.join(problems)}" if problems else "   (inside the measured workspace)"))

scene_cfg = SC.OmxCellSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)
if args.object:
    scene_cfg.object.spawn.usd_path = args.object
if args.object_scale != 1.0:
    scene_cfg.object.spawn.scale = (args.object_scale,) * 3
if place is not None:
    scene_cfg.object.init_state.pos = (place.x_m, place.y_m, S.TABLE_TOP_Z + 0.06)

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
scene = InteractiveScene(scene_cfg)
sim.reset()

# The front-left camera is placed by look-at rather than by a hand-written quaternion:
# a quaternion typo produces a plausible-looking wrong view, a look-at does not.
eye = torch.tensor([S.CAM_FRONT_LEFT_POS], device=sim.device)
tgt = torch.tensor([S.CAM_FRONT_LEFT_LOOKAT], device=sim.device)
scene["cam_front_left"].set_world_poses_from_view(eye + torch.tensor([[0.0, 0.0, S.TABLE_TOP_Z]], device=sim.device),
                                                  tgt + torch.tensor([[0.0, 0.0, S.TABLE_TOP_Z]], device=sim.device))

robot = scene["robot"]
print("\narticulation loaded:")
print(f"  bodies : {robot.num_bodies}")
print(f"  joints : {robot.num_joints}  {robot.joint_names}")
lower = robot.data.joint_pos_limits[0, :, 0].cpu().tolist()
upper = robot.data.joint_pos_limits[0, :, 1].cpu().tolist()
print(f"{'joint':<18}{'limits from the USD (deg)':<32}{'expected (deg)'}")
for name, lo, hi in zip(robot.joint_names, lower, upper):
    j = next((x for x in K.JOINTS if x.urdf_name == name), None)
    if j is None:
        print(f"{name:<18}{f'{lo*57.2958:.1f} .. {hi*57.2958:.1f}':<32}(mimic follower)")
        continue
    elo, ehi = K.effective_limits_rad(j)
    ok = abs(lo - elo) < 1e-3 and abs(hi - ehi) < 1e-3
    print(f"{name:<18}{f'{lo*57.2958:.1f} .. {hi*57.2958:.1f}':<32}"
          f"{f'{elo*57.2958:.1f} .. {ehi*57.2958:.1f}'}{'' if ok else '   🔴 MISMATCH'}")

# hold the home pose and let everything settle
target = robot.data.default_joint_pos.clone()
for i in range(args.steps):
    robot.set_joint_position_target(target)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())

# ---------------------------------------------------------------- what settled where
obj = scene["object"]
obj_p = obj.data.root_pos_w[0].cpu().tolist()
print(f"\nobject settled at x={obj_p[0]*100:.1f}cm y={obj_p[1]*100:.1f}cm z={obj_p[2]*100:.1f}cm")
print(f"  (table top is {S.TABLE_TOP_Z*100:.1f}cm; if z is far below, the object USD needs a scale)")

drift = (robot.data.joint_pos[0] - target[0]).abs().max().item()
print(f"max joint drift from the commanded home pose after {args.steps} steps: {drift*57.2958:.2f} deg")
print("  (a large drift means the drive gains cannot hold the arm against gravity — see D029)")

# ---------------------------------------------------------------- pictures, in dataset order
print("\nrendered, in the dataset's declared camera order:")
for idx, key in enumerate(("cam_wrist", "cam_front_left"), start=1):
    cam = scene[key]
    rgb = cam.data.output["rgb"][0]
    name = {"cam_wrist": "observation.images.wrist", "cam_front_left": "observation.images.front-left"}[key]
    path = save_rgb(rgb, os.path.join(args.out, f"{idx}_{key}.png"))
    intr = cam.data.intrinsic_matrices[0].cpu().tolist()
    print(f"  {idx}. {name:<34}{tuple(rgb.shape)}  -> {path}")
    print(f"     fx={intr[0][0]:.2f} fy={intr[1][1]:.2f} cx={intr[0][2]:.2f} cy={intr[1][2]:.2f}   [PROVISIONAL intrinsics]")

print("\n⚠️  This scene is NOT geometrically aligned with the real cell — experiment_spec §3 is")
print("    still blank. See sim/scene_constants.py and S4 §5-5.")
simulation_app.close()

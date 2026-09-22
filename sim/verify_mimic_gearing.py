"""Verify the PhysX mimic gearing sign for `gripper_joint_2` (S5 gap 2 / S4 S5-1 gripper row).

`convert_omx_urdf.py --mimic-gearing` bakes a gearing value into the USD's
`PhysxMimicJointAPI` at conversion time. The URDF says `multiplier="-1"`, but URDF's mimic
tag and PhysX's mimic constraint do not share a sign convention (see that script's docstring),
so the correct value cannot be read off either spec -- it has to be watched.

This script does NOT decide open vs closed relative to the real leader (that needs the real
arm and is a separate question). It answers a narrower, purely-mechanical one: for a given
gearing value, does commanding gripper_joint_1 produce a sane parallel-gripper motion (the
two fingers rotate in a way consistent with a geared, mirrored mechanism) or something
mechanically broken (both fingers rotate the same way, or the mimic constraint doesn't hold)?

Run once per candidate USD (see sim/README.md for how to produce a --mimic-gearing=-1.0
variant with convert_omx_urdf.py), then compare the renders and the printed ratio:

    ./sim/run_in_container.sh verify_mimic_gearing.py \
        --usd /workspace/test_isaaclab/assets/omx_f_generated/omx_f.usd \
        --out /workspace/test_isaaclab/omx_sim/out_gear_pos1 --headless --enable_cameras

    ./sim/run_in_container.sh verify_mimic_gearing.py \
        --usd /workspace/test_isaaclab/assets/omx_f_generated_gearNeg1/omx_f.usd \
        --out /workspace/test_isaaclab/omx_sim/out_gear_neg1 --headless --enable_cameras
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--usd", required=True, help="a USD already converted with a specific --mimic-gearing")
parser.add_argument("--out", required=True)
parser.add_argument("--open-rad", type=float, default=None, help="default: 90% of gripper_joint_1's effective upper limit")
parser.add_argument("--closed-rad", type=float, default=0.0)
parser.add_argument("--settle-steps", type=int, default=90)
parser.add_argument(
    "--no-render",
    action="store_true",
    help=(
        "skip the camera entirely (no --enable_cameras needed either) -- just print the joint-angle "
        "ratio. The camera-enabled path's RTX shutdown has been observed to hang for tens of minutes "
        "in this container; use --no-render first to check the numbers, and only pay for a render once "
        "the numbers already look right."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import omx_constants as K  # noqa: E402
import omx_scene_cfg as SC  # noqa: E402

os.makedirs(args.out, exist_ok=True)

gripper_joint = next(j for j in K.JOINTS if j.urdf_name == "gripper_joint_1")
lo, hi = K.effective_limits_rad(gripper_joint)
open_rad = args.open_rad if args.open_rad is not None else hi * 0.9


if args.no_render:

    @configclass
    class MimicCheckSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0))
        )
        robot: ArticulationCfg = SC.omx_articulation_cfg("{ENV_REGEX_NS}/Robot", usd_path=args.usd)

else:

    @configclass
    class MimicCheckSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0))
        )
        robot: ArticulationCfg = SC.omx_articulation_cfg("{ENV_REGEX_NS}/Robot", usd_path=args.usd)
        cam: CameraCfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/DiagCam",
            width=960,
            height=720,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.01, 5.0)),
            offset=CameraCfg.OffsetCfg(convention="ros"),
        )


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
scene = InteractiveScene(MimicCheckSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False))
sim.reset()

robot = scene["robot"]
i1 = robot.joint_names.index("gripper_joint_1")
i2 = robot.joint_names.index("gripper_joint_2")
name6 = next(n for n in robot.body_names if n == "link6")
name7 = next(n for n in robot.body_names if n == "link7")
b6 = robot.body_names.index(name6)
b7 = robot.body_names.index(name7)


def settle(target_val: float) -> None:
    target = robot.data.default_joint_pos.clone()
    target[:, i1] = target_val
    for _ in range(args.settle_steps):
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


def render(tag: str):
    path = "(--no-render: skipped)"
    if not args.no_render:
        # frame the camera on the midpoint of the two finger bodies' pivots, from a fixed
        # front-above offset. This is a diagnostic view, not a calibrated camera -- if the
        # gripper isn't well-framed in the first render, the offset below needs adjusting.
        p6 = robot.data.body_pos_w[0, b6]
        p7 = robot.data.body_pos_w[0, b7]
        mid = (p6 + p7) / 2.0
        eye = mid + torch.tensor([-0.15, 0.0, 0.10], device=mid.device)
        scene["cam"].set_world_poses_from_view(eye.unsqueeze(0), mid.unsqueeze(0))
        sim.step()
        scene.update(sim.get_physics_dt())

        rgb = scene["cam"].data.output["rgb"][0]
        from PIL import Image  # noqa: PLC0415

        path = os.path.join(args.out, f"{tag}.png")
        Image.fromarray(rgb[..., :3].detach().cpu().numpy()).save(path)

    j1 = robot.data.joint_pos[0, i1].item()
    j2 = robot.data.joint_pos[0, i2].item()
    ratio = j2 / j1 if abs(j1) > 1e-6 else float("nan")
    print(f"{tag:<10}{j1*57.2958:>12.2f}{j2*57.2958:>12.2f}{ratio:>10.3f}    {path}")
    return j1, j2


print(f"\nUSD: {args.usd}")
print(f"gripper_joint_1 effective range: {lo*57.2958:.1f} .. {hi*57.2958:.1f} deg")
print(f"commanding closed={args.closed_rad*57.2958:.1f}deg, open={open_rad*57.2958:.1f}deg\n")
print(f"{'pose':<10}{'joint1(deg)':>12}{'joint2(deg)':>12}{'j2/j1':>10}    render")

settle(args.closed_rad)
j1_c, j2_c = render("closed")
settle(open_rad)
j1_o, j2_o = render("open")

print("\n-- read this against the two renders, not just the numbers --")
print(f"joint2 moved {(j2_o - j2_c)*57.2958:+.2f} deg while joint1 moved {(j1_o - j1_c)*57.2958:+.2f} deg")
print("expected for a working mimic: joint2's change has a fixed, consistent ratio to joint1's change")
print("(that ratio's SIGN is the thing under test -- decide by looking at closed.png vs open.png:")
print(" do the two fingers visibly converge/diverge together, or does one look wrong/frozen/reversed?)")
import os
os._exit(0)

"""Pose the sim arm at a real episode's recorded `observation.state` and render it.

This is S5's new architecture (`S5_sim_replay_augmentation.md` §4, `[Eric決定 2026-09-18]`) at its
smallest: no PD position control, no drive gains, no torque model -- the arm is written straight
into PhysX at the joint angles the real arm actually reached, via
`Articulation.write_joint_state_to_sim()`. Whatever comes out of the camera is therefore the real
pose, to the accuracy of the real encoders and of `joint_mapping.py`'s degrees->radians mapping.

**Its first job is to settle the SIGN question without a lab day.** `joint_mapping.SIGN` is still
`[未確認]` for all six joints (S4 §5-1's five-pose test has not been run). But a flipped sign does
not produce a subtle offset -- it renders the arm in a grossly wrong configuration (reaching up
when the real arm reached down). Put this render next to the real recorded video frame at the same
timestamp (`scripts/compare_sim_real_frames.py`) and a sign error is visible at a glance, with no
camera calibration needed, because what's being compared is arm CONFIGURATION, not pixels.

🔴 **What this can and cannot settle.** It catches gross errors: a flipped sign, a swapped joint,
a mirrored arm. It does NOT validate calibration-grade things -- angle offsets, small scale
errors, or anything about camera geometry (gap 4 is still open; `scene_constants.py`'s camera pose
is PLACEHOLDER, so the viewpoint is only approximately the real one). S4 §5-5 already makes this
distinction and calls the overlay a smoke test, T4, explicitly not an acceptance test. Same here.

⚠️ `gripper_joint_2` is driven by the PhysX mimic constraint, not written here -- and that
constraint currently only reaches about half the intended amplitude (`sim/README.md` "Two gaps",
gap 2's residual). Read the gripper in these renders as indicative, not measured.

🔴 **Trap this script already fell into once (2026-09-18, caught by md5-ing the outputs):**
`OmxCellSceneCfg`'s cameras carry `update_period = 1/CAM_FPS` (they model a 15 fps sensor). Posing
the arm and stepping only a few times at dt=1/120 does NOT advance sim time past that period, so
`camera.data.output["rgb"]` hands back the PREVIOUS render -- six saved PNGs, only two distinct
images, every one of them silently mislabelled with the pose it was supposed to show. This is the
same failure shape as the `trash_obj` scale bug in `sim/README.md`: the output looks entirely
plausible and is wrong. Fixed below by forcing `update_period = 0.0` (render every step) for this
script, since it takes one snapshot per requested pose rather than simulating a live 15 fps
sensor. **If you write another script that renders discrete poses, do the same, and verify with
`md5sum` that the frames actually differ before believing any of them.**

    ./sim/run_in_container.sh render_state_replay.py \\
        --dataset-root /workspace/test_isaaclab/omx_sim/dataset --episode 0 \\
        --frames 0,120,226,300,380,450 \\
        --out /workspace/test_isaaclab/omx_sim/state_replay_ep0 --headless --enable_cameras

Add `--sign-override shoulder_lift=-1` to render the same frames with a candidate sign flipped,
then compare the two output folders against the real video: the wrong one should look obviously
wrong. Dataset parquet has to be copied into the container first -- same as `fit_drive_gains.py`.
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
parser.add_argument("--frames", default="", help="comma-separated frame indices; default: 6 spread over the episode")
parser.add_argument(
    "--sign-override",
    default="",
    help="e.g. 'shoulder_lift=-1' -- overrides joint_mapping.SIGN for this run only, to render a "
    "candidate sign for comparison. See joint_mapping.py's docstring.",
)
parser.add_argument("--settle-steps", type=int, default=8, help="physics steps after writing each pose, before rendering")
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import joint_mapping as JM  # noqa: E402
import omx_constants as K  # noqa: E402
import omx_scene_cfg as SC  # noqa: E402
import scene_constants as S  # noqa: E402


@configclass
class StateReplaySceneCfg(SC.OmxCellSceneCfg):
    """The dataset scene plus one extra camera framed tightly on the arm.

    The two dataset cameras stay exactly as the dataset declares them (order and all) because the
    real-vs-sim comparison has to use the same nominal viewpoint. But `scene_constants.py`'s
    front-left pose is a PLACEHOLDER (gap 4), and at that placeholder distance the arm occupies a
    small corner of the frame -- readable enough to check a grossly wrong joint sign, but only
    just. `cam_diag` is a deliberately un-calibrated close view whose ONLY job is to make the arm's
    configuration big enough to judge. It is never written into a dataset.
    """

    cam_diag: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/DiagCam",
        update_period=0.0,
        width=960,
        height=720,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.05, 6.0)),
        offset=CameraCfg.OffsetCfg(convention="ros"),
    )

DATASET_FPS = 15.0

if args.sign_override:
    for entry in args.sign_override.split(","):
        name, val = entry.split("=")
        name = name.strip()
        if name not in JM.SIGN:
            raise SystemExit(f"--sign-override: {name!r} is not one of {JM.DATASET_JOINT_ORDER}")
        JM.SIGN[name] = float(val)
    print(f"⚠️  SIGN overridden for this run only: {JM.SIGN}")


def load_episode_states(dataset_root: str, episode: int):
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit(
            "pyarrow not importable -- install once: /isaac-sim/python.sh -m pip install pyarrow"
        ) from e
    files = sorted(glob.glob(os.path.join(dataset_root, "data", "chunk-*", "file-*.parquet")))
    if not files:
        raise SystemExit(f"no parquet files under {dataset_root}/data/chunk-*/file-*.parquet")
    rows = []
    for f in files:
        d = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"]).to_pydict()
        for ep, fi, st in zip(d["episode_index"], d["frame_index"], d["observation.state"]):
            if ep == episode:
                rows.append((fi, st))
    if not rows:
        raise SystemExit(f"episode {episode} not found under {dataset_root}")
    rows.sort(key=lambda r: r[0])
    return [r[1] for r in rows]


states_deg = load_episode_states(args.dataset_root, args.episode)
n_frames = len(states_deg)

if args.frames:
    frames = [int(x) for x in args.frames.split(",")]
    bad = [f for f in frames if not (0 <= f < n_frames)]
    if bad:
        raise SystemExit(f"frames {bad} outside episode {args.episode}'s range 0..{n_frames - 1}")
else:
    frames = [round(i * (n_frames - 1) / 5) for i in range(6)]

print(f"episode {args.episode}: {n_frames} frames; rendering {frames}")

os.makedirs(args.out, exist_ok=True)

scene_cfg = StateReplaySceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)
# 🔴 render every step -- see the module docstring's "trap" note. Without this the cameras keep
# their 15 fps sensor cadence and hand back the previous pose's image.
scene_cfg.cam_front_left.update_period = 0.0
scene_cfg.cam_wrist.update_period = 0.0

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
scene = InteractiveScene(scene_cfg)
sim.reset()

# same look-at placement as preview_scene.py -- a quaternion typo produces a plausible-looking
# wrong view, a look-at does not
eye = torch.tensor([S.CAM_FRONT_LEFT_POS], device=sim.device)
tgt = torch.tensor([S.CAM_FRONT_LEFT_LOOKAT], device=sim.device)
lift = torch.tensor([[0.0, 0.0, S.TABLE_TOP_Z]], device=sim.device)
scene["cam_front_left"].set_world_poses_from_view(eye + lift, tgt + lift)

robot = scene["robot"]
joint_names = [j.urdf_name for j in K.JOINTS]
joint_idx = [robot.joint_names.index(n) for n in joint_names]
zeros = torch.zeros(1, len(joint_idx), device=sim.device)

manifest = []
print(f"\n{'frame':>6}  " + "  ".join(f"{n[:9]:>9}" for n in JM.DATASET_JOINT_ORDER) + "   (deg, as written to sim)")

for t in frames:
    rad = JM.row_to_sim_rad(states_deg[t])
    pos = torch.tensor([rad], device=sim.device, dtype=torch.float32)

    # kinematic write: straight into PhysX, no actuator model involved (S5 §4).
    # Re-written on EVERY warm-up step rather than once: the RTX renderer needs a few frames to
    # converge (the first capture of a run is visibly noisy otherwise), and re-writing keeps the
    # pose exact instead of letting gravity pull it during those extra steps. The real pipeline
    # writes every frame anyway, so this is the same code path, not a special case.
    for _ in range(max(1, args.settle_steps)):
        robot.write_joint_state_to_sim(pos, zeros, joint_ids=joint_idx)
        # hold the PD target at the same pose too, or the actuator drags the arm toward its init
        # pose in between writes and the contact/settling behaves like neither
        robot.set_joint_position_target(robot.data.joint_pos.clone())
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    # frame the diagnostic camera on the arm itself, from a fixed offset, so its view is
    # comparable across frames (it moves with the arm's base, not with the arm's pose)
    base = robot.data.body_pos_w[0, 0]
    diag_eye = base + torch.tensor([0.55, 0.45, 0.30], device=base.device)
    diag_tgt = base + torch.tensor([0.05, 0.0, 0.10], device=base.device)
    scene["cam_diag"].set_world_poses_from_view(diag_eye.unsqueeze(0), diag_tgt.unsqueeze(0))
    sim.step()
    scene.update(sim.get_physics_dt())

    actual_deg = [robot.data.joint_pos[0, i].item() * 57.2958 for i in joint_idx]
    print(f"{t:>6}  " + "  ".join(f"{d:>9.2f}" for d in actual_deg))

    entry = {"frame": t, "timestamp_s": t / DATASET_FPS, "joint_deg_written": states_deg[t],
             "joint_deg_in_sim": actual_deg}
    for key, cam_name in (("cam_front_left", "front-left"), ("cam_wrist", "wrist"), ("cam_diag", "diag")):
        rgb = scene[key].data.output["rgb"][0]
        path = os.path.join(args.out, f"f{t:05d}_{cam_name}.png")
        from PIL import Image  # noqa: PLC0415

        Image.fromarray(rgb[..., :3].detach().cpu().numpy()).save(path)
        entry[f"render_{cam_name}"] = os.path.basename(path)
    manifest.append(entry)

meta = {
    "source_episode_id": args.episode,
    "dataset_root": args.dataset_root,
    "dataset_fps": DATASET_FPS,
    "sign": dict(JM.SIGN),
    "sign_overridden": bool(args.sign_override),
    "driven_by": "write_joint_state_to_sim (kinematic; no drive gains involved)",
    "geometry": "PLACEHOLDER -- scene_constants.py camera pose is not measured (S5 gap 4 open)",
    "frames": manifest,
}
with open(os.path.join(args.out, "manifest.json"), "w") as fh:
    json.dump(meta, fh, indent=2)

print(f"\nwrote {len(frames)} frame(s) x 2 cameras + manifest.json to {args.out}")
print("next: scripts/compare_sim_real_frames.py to put these beside the real recorded video frames")
simulation_app.close()

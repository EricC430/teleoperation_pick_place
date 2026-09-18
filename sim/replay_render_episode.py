"""S5 stage 1: replay one real episode kinematically, under domain randomization, and render it.

This is the container half of `docs/specs/S5_sim_replay_augmentation.md`. It does everything that
needs Isaac Lab and nothing that needs `lerobot`:

    real observation.state -> write_joint_state_to_sim -> DR'd scene -> render both cameras -> PNGs

Stage 2 (`scripts/sim_replay_augment.py`, host side) turns that output into a LeRobotDataset.
**The split is forced, not stylistic:** `/isaac-sim/python.sh` has no `lerobot`, no `av`, no
`torchcodec` (checked 2026-09-18), and installing them into Isaac Sim's own interpreter to save one
process boundary is a worse trade than writing PNGs to disk.

Arm motion is a kinematic replay (`write_joint_state_to_sim`), NOT PD position control -- S5 §4,
`[Eric決定 2026-09-18]`. Drive gains, torque limits and gravity do not participate in the arm's
rendered pose, so nothing here depends on gap 1.

## 🔴 Three fidelity gaps this script records rather than hides

`meta.fidelity` in the manifest carries all three; stage 2 must copy them into the dataset's own
meta. They are not this script's bugs -- they are missing inputs, and the honest thing is to ship
the flag, not to quietly pick something plausible:

1. **`object_matches_source`** -- S5 §1/§4 promise "same object, same start position, only the
   visual environment changes". The real `uvc_60` episodes use a **paper cup**;
   `assets/trash_obj/` has no cup (cans, bottles, apple, banana, orange, egg carton, organic
   waste). So whatever `--object` renders is a DIFFERENT object from the one in the source
   recording, and the action labels describe grasping something that isn't on screen.
2. **`placement_matches_source`** -- `episode_meta/` only covers the older
   `omx_pick_place_pilot` (8 episodes) and its `placement_id` column is blank in 7 of them.
   There is no recorded placement for any `uvc_60` episode, so `--place` is a choice, not a
   reconstruction.
3. **`geometry_aligned`** -- always false until S5 gap 4's T1/T2 land. `scene_constants.py`'s
   camera pose is PLACEHOLDER (S4 §5-5).

## Domain randomization

One draw per episode, from `--dr-seed`, ranges per S5 §5 item 4 (the NVIDIA SO-101 tutorial's
values, not invented here): dome light exposure -4.0..3.0, colour temperature 2500..9500 K,
camera translation +/-0.02 m and rotation +/-0.05 rad.

⚠️ Implemented directly rather than through Isaac Lab `EventTerm`s, which the spec names. EventTerm
belongs to the manager-based env machinery; this script drives an `InteractiveScene` directly, and
bolting a manager onto it to honour the letter of the spec would add a layer without changing a
single randomized value. The RANGES are what the spec actually pins down, and those are followed.

⚠️ **Object appearance randomization is NOT implemented** (the spec lists it). Swapping the object
mid-run is impossible -- it is chosen at spawn -- and swapping it per episode would break gap 1 of
the fidelity list even further. Recorded as `object_appearance` in the DR manifest so nobody reads
its absence as an oversight.

    ./sim/run_in_container.sh replay_render_episode.py \\
        --dataset-root /workspace/test_isaaclab/omx_sim/dataset --episode 0 \\
        --dr-seed 42 --out /workspace/test_isaaclab/omx_sim/replay_ep0_seed42 \\
        --headless --enable_cameras

Add `--stride 20` for a fast smoke test (renders every 20th frame) before committing to a full
534-frame run.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset-root", required=True, help="dir containing data/chunk-*/file-*.parquet")
parser.add_argument("--episode", type=int, required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--object", default=None, help="object USD (default: omx_scene_cfg.DEFAULT_OBJECT_USD)")
parser.add_argument("--source-object-name", default="paper_cup",
                    help="what the SOURCE recording actually had, for the fidelity record")
parser.add_argument("--placements", default=None, help="placement_label_map CSV")
parser.add_argument("--place", default=None, help="short_id / placement_id within that CSV")
parser.add_argument("--dr-seed", type=int, default=None, help="omit for no randomization at all")
parser.add_argument("--dr-preset", default="nvidia-so101-default", choices=["nvidia-so101-default"])
parser.add_argument("--steps-per-frame", type=int, default=1, help="physics steps per dataset frame")
parser.add_argument("--warmup-steps", type=int, default=12, help="steps before the FIRST capture, to let RTX converge")
parser.add_argument("--stride", type=int, default=1, help="render every Nth frame (smoke tests)")
parser.add_argument("--max-frames", type=int, default=0, help="stop after N rendered frames (0 = no limit)")
parser.add_argument("--skip-grasp", action="store_true", help="degraded mode: no attach/detach (S5 §2 gap 3)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

import joint_mapping as JM  # noqa: E402
import omx_constants as K  # noqa: E402
import omx_scene_cfg as SC  # noqa: E402
import scene_constants as S  # noqa: E402
from grasp_attach import GraspAttachConfig, ScriptedGraspAttach  # noqa: E402

DATASET_FPS = 15.0
# S5 §5 item 4 -- NVIDIA SO-101 tutorial ranges, quoted not invented
DR_EXPOSURE = (-4.0, 3.0)
DR_COLOR_TEMP_K = (2500.0, 9500.0)
DR_CAM_TRANS_M = 0.02
DR_CAM_ROT_RAD = 0.05


def kelvin_to_rgb(kelvin: float) -> tuple[float, float, float]:
    """Planckian-locus approximation (Tanner Helland's), normalized to 0..1.

    Self-contained on purpose: the alternative is a USD colorTemperature attribute whose exact
    name/behaviour would be one more unverified API claim in a script that already has enough."""
    t = max(1000.0, min(40000.0, kelvin)) / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        r = 329.698727446 * ((t - 60) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60) ** -0.0755148492)
    if t >= 66:
        b = 255.0
    elif t <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10) - 305.0447927307
    return tuple(max(0.0, min(255.0, c)) / 255.0 for c in (r, g, b))


def load_episode(dataset_root: str, episode: int):
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit("pyarrow missing -- /isaac-sim/python.sh -m pip install pyarrow") from e
    files = sorted(glob.glob(os.path.join(dataset_root, "data", "chunk-*", "file-*.parquet")))
    if not files:
        raise SystemExit(f"no parquet under {dataset_root}/data/chunk-*/file-*.parquet")
    rows = []
    for f in files:
        d = pq.read_table(f, columns=["episode_index", "frame_index", "action", "observation.state"]).to_pydict()
        for ep, fi, act, st in zip(d["episode_index"], d["frame_index"], d["action"], d["observation.state"]):
            if ep == episode:
                rows.append((fi, act, st))
    if not rows:
        raise SystemExit(f"episode {episode} not in {dataset_root}")
    rows.sort(key=lambda r: r[0])
    return [r[1] for r in rows], [r[2] for r in rows]


actions_deg, states_deg = load_episode(args.dataset_root, args.episode)
n_frames = len(actions_deg)
frames = list(range(0, n_frames, max(1, args.stride)))
if args.max_frames:
    frames = frames[: args.max_frames]
print(f"episode {args.episode}: {n_frames} frames, rendering {len(frames)} of them (stride {args.stride})")

# ---------------------------------------------------------------- domain randomization draw
rng = random.Random(args.dr_seed)
if args.dr_seed is None:
    dr = {"enabled": False}
    light_intensity = S.DOME_LIGHT_INTENSITY
    light_color = (0.9, 0.9, 0.95)
    cam_dpos = (0.0, 0.0, 0.0)
    cam_drot = (0.0, 0.0, 0.0)
else:
    exposure = rng.uniform(*DR_EXPOSURE)
    color_temp = rng.uniform(*DR_COLOR_TEMP_K)
    cam_dpos = tuple(rng.uniform(-DR_CAM_TRANS_M, DR_CAM_TRANS_M) for _ in range(3))
    cam_drot = tuple(rng.uniform(-DR_CAM_ROT_RAD, DR_CAM_ROT_RAD) for _ in range(3))
    light_intensity = S.DOME_LIGHT_INTENSITY * (2.0**exposure)
    light_color = kelvin_to_rgb(color_temp)
    dr = {
        "enabled": True,
        "preset": args.dr_preset,
        "seed": args.dr_seed,
        "dome_light_exposure": exposure,
        "dome_light_intensity": light_intensity,
        "color_temperature_k": color_temp,
        "dome_light_rgb": light_color,
        "camera_translation_m": cam_dpos,
        "camera_rotation_rad": cam_drot,
        "object_appearance": "NOT IMPLEMENTED -- see module docstring",
    }
print(f"DR: {json.dumps(dr, default=str)}")

# ---------------------------------------------------------------- scene
scene_cfg = SC.OmxCellSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)
# 🔴 render every step: these cameras default to a 15 fps sensor cadence, which silently returns
# the PREVIOUS frame's image when a pose is written and only a step or two elapses. See
# render_state_replay.py's docstring for the full incident.
scene_cfg.cam_front_left.update_period = 0.0
scene_cfg.cam_wrist.update_period = 0.0
scene_cfg.dome_light.spawn.intensity = light_intensity
scene_cfg.dome_light.spawn.color = light_color
if args.object:
    scene_cfg.object.spawn.usd_path = args.object
# wrist camera is parented to link5, so its DR jitter is a local offset tweak, pre-build
scene_cfg.cam_wrist.offset.pos = tuple(p + d for p, d in zip(S.CAM_WRIST_OFFSET_POS, cam_dpos))

place = None
if args.placements:
    placements = S.load_placements(args.placements)
    matches = [p for p in placements if p.short_id == args.place or p.placement_id == args.place]
    if args.place and not matches:
        raise SystemExit(f"no placement {args.place!r} in {args.placements}")
    place = matches[0] if matches else placements[0]
    scene_cfg.object.init_state.pos = (place.x_m, place.y_m, S.TABLE_TOP_Z + 0.06)
    print(f"object placement {place.short_id} ({place.placement_id}) at x={place.x_m:.3f} y={place.y_m:.3f}")

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
scene = InteractiveScene(scene_cfg)
sim.reset()

# front-left camera: look-at, with the DR jitter applied to both endpoints (translation) and the
# target (which is what a small rotation of the camera actually amounts to here)
eye = torch.tensor([S.CAM_FRONT_LEFT_POS], device=sim.device) + torch.tensor([cam_dpos], device=sim.device)
tgt = torch.tensor([S.CAM_FRONT_LEFT_LOOKAT], device=sim.device)
reach = float(torch.norm(tgt - eye))
tgt = tgt + torch.tensor([[math.tan(cam_drot[0]) * reach, math.tan(cam_drot[1]) * reach, math.tan(cam_drot[2]) * reach]],
                         device=sim.device)
lift = torch.tensor([[0.0, 0.0, S.TABLE_TOP_Z]], device=sim.device)
scene["cam_front_left"].set_world_poses_from_view(eye + lift, tgt + lift)

robot = scene["robot"]
obj = scene["object"]
joint_names = [j.urdf_name for j in K.JOINTS]
joint_idx = [robot.joint_names.index(n) for n in joint_names]
zeros = torch.zeros(1, len(joint_idx), device=sim.device)
b6 = robot.body_names.index("link6")
b7 = robot.body_names.index("link7")

grasp = None
if not args.skip_grasp:
    grasp = ScriptedGraspAttach(GraspAttachConfig())
    thr = grasp.calibrate([row[5] for row in states_deg])
    print(f"grasp attach threshold: {thr:.2f} deg (from this episode's own gripper trace)")

os.makedirs(args.out, exist_ok=True)


def write_pose(t: int) -> torch.Tensor:
    rad = JM.row_to_sim_rad(states_deg[t])
    pos = torch.tensor([rad], device=sim.device, dtype=torch.float32)
    robot.write_joint_state_to_sim(pos, zeros, joint_ids=joint_idx)
    robot.set_joint_position_target(robot.data.joint_pos.clone())
    scene.write_data_to_sim()
    return pos


# settle the scene (object drops onto the table) and let RTX converge before the first capture
write_pose(frames[0])
for _ in range(args.warmup_steps):
    sim.step()
    scene.update(sim.get_physics_dt())

records = []
from PIL import Image  # noqa: E402, PLC0415

for n, t in enumerate(frames):
    write_pose(t)
    for _ in range(max(1, args.steps_per_frame)):
        sim.step()
        scene.update(sim.get_physics_dt())

    attached = False
    if grasp is not None:
        tcp = (robot.data.body_pos_w[0, b6] + robot.data.body_pos_w[0, b7]) / 2.0
        tcp_q = robot.data.body_quat_w[0, b6]
        want_p, want_q = grasp.step(t, states_deg[t][5], tcp, tcp_q, obj.data.root_pos_w[0], obj.data.root_quat_w[0])
        if grasp.attached:
            obj.write_root_pose_to_sim(torch.cat([want_p, want_q]).unsqueeze(0))
            obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=want_p.device))
            attached = True

    rec = {"frame": t, "timestamp_s": t / DATASET_FPS, "action": actions_deg[t],
           "observation_state": states_deg[t], "grasp_attached": attached}
    for key, name in (("cam_wrist", "wrist"), ("cam_front_left", "front-left")):
        rgb = scene[key].data.output["rgb"][0]
        fn = f"f{t:05d}_{name}.png"
        Image.fromarray(rgb[..., :3].detach().cpu().numpy()).save(os.path.join(args.out, fn))
        rec[f"image_{name}"] = fn
    records.append(rec)
    if n % max(1, len(frames) // 10) == 0:
        print(f"  {n}/{len(frames)} (frame {t}){'  [attached]' if attached else ''}")

manifest = {
    "source_dataset_root": args.dataset_root,
    "source_episode_id": args.episode,
    "dataset_fps": DATASET_FPS,
    "n_source_frames": n_frames,
    "stride": args.stride,
    "rendered_frames": len(records),
    "sign": dict(JM.SIGN),
    "driven_by": "write_joint_state_to_sim (kinematic replay; no drive gains, S5 §4)",
    "dr": dr,
    "grasp": {
        "mode": "skipped" if grasp is None else "scripted_attach_detach",
        "events": [] if grasp is None else [
            {"frame": e.frame, "kind": e.kind, "gripper_deg": e.gripper_reading, "dist_m": e.dist_m}
            for e in grasp.events
        ],
    },
    "fidelity": {
        "geometry_aligned": False,
        "geometry_note": "PLACEHOLDER camera pose, S5 gap 4 (T1/T2) not done",
        "object_matches_source": False,
        "object_note": (
            f"rendered {os.path.basename(args.object or SC.DEFAULT_OBJECT_USD)}; the source recording "
            f"used {args.source_object_name!r}, which has no asset in assets/trash_obj/"
        ),
        "placement_matches_source": False,
        "placement_note": (
            "no placement_id recorded for this dataset's episodes (episode_meta/ covers only "
            "omx_pick_place_pilot, and its placement_id column is blank in 7 of 8 rows)"
        ),
        "gripper_amplitude_verified": False,
        "gripper_note": "mimic constraint reaches ~half the intended amplitude (gap 2 residual)",
    },
    "frames": records,
}
with open(os.path.join(args.out, "manifest.json"), "w") as fh:
    json.dump(manifest, fh, indent=2)

print(f"\nrendered {len(records)} frame(s) x 2 cameras -> {args.out}")
if grasp is not None:
    print(f"grasp events: {[(e.frame, e.kind) for e in grasp.events]}")
print("🔴 fidelity flags (all recorded in manifest.json, stage 2 must carry them into dataset meta):")
for k, v in manifest["fidelity"].items():
    if not k.endswith("_note"):
        print(f"    {k} = {v}")
print("next: scripts/sim_replay_augment.py to assemble a LeRobotDataset from this")
simulation_app.close()

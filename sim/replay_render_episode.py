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

1. **`object_matches_source`** -- still open. S5 §1/§4 promise "same object, same start position,
   only the visual environment changes". The real `uvc_60` episodes use a **paper cup**;
   `assets/trash_obj/` has no cup (cans, bottles, apple, banana, orange, egg carton, organic
   waste). So whatever `--object` renders is a DIFFERENT object from the one in the source
   recording, and the action labels describe grasping something that isn't on screen.
   `[Eric說 2026-09-21]` the plan is to rebuild/restore the real asset, or author a
   same-class stand-in -- so this flag is expected to close later, by a new asset, not by code here.
2. **`placement_matches_source`** -- CLOSED when `--place-from-episode` is passed.
   `[Eric說 2026-09-21]`: uvc_60 walked campA_136sym's `t1..t60` in order, so episode i used
   `t{i+1}`. Nothing in the dataset records this (the `episode_meta/` CSV covers only the older
   `omx_pick_place_pilot`), so it was corroborated independently before being wired in: taking each
   episode's gripper-closing frame, running `observation.state` through `reach_logger/fk.py`, and
   correlating the end-effector position against that episode's claimed placement gives
   **x r=+0.63, y r=+0.66 over all 60 episodes, against r~0.00 for 20 shuffled pairings**, and the
   correlation is strongest at exactly this alignment (shifting the mapping by +/-1 or +/-2
   episodes drops it). Residual scatter (p50 9 cm after removing a constant +10 cm offset) is
   attributable to the crude grasp-frame pick and to `end_effector_link` not being the grasp
   centre -- not to the mapping. Without the flag, `--place` is a choice, not a reconstruction.
   🔴 [已查證 2026-09-29] EXCEPT the back third: docs/meeting/2026-09-13.md item 4, "t41 重錄後排在
   最後一集" -- ep0-39 = t1-t40, ep40-58 = t42-t60, ep59 = t41. The plain `t{i+1}` rule put ep40-58
   one placement off and ep59 at t60 (23-52 cm grasp "errors"); `uvc60_short_id()` below has the fix.
3. **`geometry_aligned`** -- false until BOTH cameras have a measured lens and pose.
   `--front-left-intrinsics/--front-left-extrinsics/--wrist-intrinsics` wire in the 10/07 T1/T2 JSONs
   (`geometry_aligned_per_camera` then reads front-left: true); the wrist POSE is still the CAD offset
   because the hand-eye has not passed. Without those flags the old behaviour is unchanged.

    ./sim/run_in_container.sh replay_render_episode.py ... \\
        --front-left-intrinsics $GUEST/omx_sim/calibration/2026-10-07_camera_intrinsics_front-left.json \\
        --front-left-extrinsics $GUEST/omx_sim/calibration/2026-10-07_camera_extrinsics_front-left.json \\
        --wrist-intrinsics      $GUEST/omx_sim/calibration/2026-10-07_camera_intrinsics_wrist.json

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
parser.add_argument("--object", default=None,
                    help="replace the default primitive cup with a USD asset (e.g. once a real "
                         "paper-cup mesh exists). Default: the measured-dimension cylinder cup.")
parser.add_argument("--source-object-name", default="paper_cup",
                    help="what the SOURCE recording actually had, for the fidelity record")
parser.add_argument("--placements", default=None, help="placement_label_map CSV")
parser.add_argument("--place", default=None,
                    help="short_id / placement_id within that CSV. Omit to derive it from the episode "
                         "index -- see --place-from-episode")
parser.add_argument("--place-from-episode", action="store_true",
                    help="derive the placement as short_id t<episode+1> (episode 0 -> t1). "
                         "[Eric說 2026-09-21] uvc_60 was recorded walking campA_136sym's t1..t60 in "
                         "order. Corroborated independently, see the module docstring.")
parser.add_argument("--dr-seed", type=int, default=None, help="omit for no randomization at all")
parser.add_argument("--dr-preset", default="nvidia-so101-default", choices=["nvidia-so101-default"])
parser.add_argument(
    "--offset-delta-deg", default="",
    help="e.g. 'shoulder_lift=+20.14' -- ADDS this many degrees to joint_mapping.OFFSET_RAD for "
         "this run only, leaving the committed constant alone. For S6 section 4-a's undecided "
         "question (does shoulder_lift's zero carry the upper arm's 20.14 deg lean); a flag, not "
         "an edit, so that 甲 vs 乙 is one command and 甲 stays what is in git.")
parser.add_argument("--steps-per-frame", type=int, default=1, help="physics steps per dataset frame")
parser.add_argument("--warmup-steps", type=int, default=12, help="steps before the FIRST capture, to let RTX converge")
parser.add_argument("--stride", type=int, default=1, help="render every Nth frame (smoke tests)")
parser.add_argument("--max-frames", type=int, default=0, help="stop after N rendered frames (0 = no limit)")
parser.add_argument("--skip-grasp", action="store_true", help="degraded mode: no attach/detach (S5 §2 gap 3)")
parser.add_argument("--attach-radius", type=float, default=0.09,
                    help="grasp attach radius in meters (default 0.09 m, covers cup rim-to-centre offset)")
parser.add_argument("--front-left-intrinsics", default=None,
                    help="calib_intrinsics_realsense.py JSON: render front-left with the measured lens "
                         "(principal point, fx/fy, distortion) instead of scene_constants' HFOV")
parser.add_argument("--front-left-extrinsics", default=None,
                    help="calib_extrinsics_aruco.py JSON (frame 'world'): place front-left at that full pose "
                         "(incl. roll) instead of the look-at from scene_constants")
parser.add_argument("--wrist-intrinsics", default=None,
                    help="calib_intrinsics_checkerboard.py JSON: render the wrist camera with the measured lens. "
                         "Its POSE stays scene_constants' CAD offset -- the wrist hand-eye is not calibrated")
parser.add_argument("--save-ideal", action="store_true",
                    help="also save Isaac's raw render (square pixels, centred axis, no distortion) per camera")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

import camera_distortion as CD  # noqa: E402
import joint_mapping as JM  # noqa: E402
import omx_constants as K  # noqa: E402
import omx_scene_cfg as SC  # noqa: E402
import scene_constants as S  # noqa: E402
from grasp_attach import GraspAttachConfig, ScriptedGraspAttach, tcp_pose_w  # noqa: E402

offset_delta_deg = {}
if args.offset_delta_deg:
    for _entry in args.offset_delta_deg.split(","):
        _name, _val = _entry.split("=")
        _name = _name.strip()
        if _name not in JM.OFFSET_RAD:
            raise SystemExit(f"--offset-delta-deg: {_name!r} is not one of {list(JM.OFFSET_RAD)}")
        _before = JM.OFFSET_RAD[_name]
        JM.OFFSET_RAD[_name] = _before + math.radians(float(_val))
        offset_delta_deg[_name] = float(_val)
        print(f"\u26a0\ufe0f  OFFSET_RAD[{_name}] {_before:+.8f} -> {JM.OFFSET_RAD[_name]:+.8f} rad "
              f"({float(_val):+.2f} deg), THIS RUN ONLY -- the committed constant is unchanged")

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


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def R_to_quat_wxyz(R):
    import numpy as np  # noqa: PLC0415
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        q = [0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    n = math.sqrt(sum(v * v for v in q))
    return [v / n for v in q]


def front_left_pose_from_json(path: str):
    """(position in THIS scene's world, R_world_from_cam in ROS optical) from a T2 extrinsics JSON.

    The T2 solve's world is the coordinate paper's frame with z measured from the floor at its own
    --table-top-z; this scene's table is at scene_constants.TABLE_TOP_Z, so only z is re-based.
    x/y are used as-is: docs/meeting/2026-10-08_paper_to_pan.md found the paper frame and the arm's
    FK (pan-axis) frame agree to ~1 cm on 10/07, with no detectable constant offset."""
    d = json.load(open(path))
    if d.get("frame") != "world":
        raise SystemExit(f"{path}: frame={d.get('frame')!r}; expected a 'world' front-left pose from calib_extrinsics_aruco.py")
    if "table_top_z" not in d:
        raise SystemExit(f"{path}: no table_top_z -- cannot re-base its z onto this scene's table")
    p = d["pos_m"]
    return [p[0], p[1], p[2] - d["table_top_z"] + S.TABLE_TOP_Z], quat_wxyz_to_R(d["quat_wxyz"]), d


def small_rotation(rx: float, ry: float, rz: float):
    """Rx @ Ry @ Rz -- the DR jitter, applied in the camera's own (ROS optical) axes."""
    import numpy as np  # noqa: PLC0415
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    return (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))


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
    # the default cup is a primitive (CylinderCfg), so swapping in a USD means replacing the whole
    # spawn config, not poking usd_path on it
    scene_cfg.object.spawn = sim_utils.UsdFileCfg(
        usd_path=args.object,
        scale=S.TRASH_OBJ_SCALE if "trash_obj" in args.object else (1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=3.0),
    )
# wrist camera is parented to link5, so its DR jitter is a local offset tweak, pre-build
scene_cfg.cam_wrist.offset.pos = tuple(p + d for p, d in zip(S.CAM_WRIST_OFFSET_POS, cam_dpos))

# Measured lenses: Isaac renders a centred, square-pixel image sized to hold every ray the real camera
# sees (camera_distortion.render_spec); after each render a remap turns it into the real camera's
# image -- its own principal point, fx/fy and distortion, at the dataset's resolution. Isaac cannot
# render the real K directly (camera_distortion's docstring has the container evidence).
lens = {}  # scene key -> dict(model, render_w, render_h, f, source)
for key, path, want_wh in (("cam_front_left", args.front_left_intrinsics, (S.CAM_WIDTH_FRONT_LEFT, S.CAM_HEIGHT_FRONT_LEFT)),
                           ("cam_wrist", args.wrist_intrinsics, (S.CAM_WIDTH_WRIST, S.CAM_HEIGHT_WRIST))):
    if not path:
        continue
    model = CD.load_model(path)
    if (model.width, model.height) != want_wh:
        raise SystemExit(f"{path} is {model.width}x{model.height}, but {key} records at {want_wh[0]}x{want_wh[1]}")
    rw, rh, rf = CD.render_spec(model)
    getattr(scene_cfg, key).width, getattr(scene_cfg, key).height = rw, rh
    lens[key] = {"model": model, "render_w": rw, "render_h": rh, "f": rf, "source": path}
    print(f"{key}: measured lens from {path} -> Isaac renders {rw}x{rh} at f={rf:.1f} px, remapped to {model.width}x{model.height}")



def uvc60_short_id(episode: int) -> str:
    """uvc_60 episode -> campA_136sym short id. t41 was re-recorded and appended as episode 59."""
    return f"t{41 if episode == 59 else episode + 1 if episode < 40 else episode + 2}"


place = None
place_source = "default (omx_scene_cfg.py), NOT the source episode's placement"
if args.placements:
    placements = S.load_placements(args.placements)
    if args.place_from_episode:
        if args.place:
            raise SystemExit("--place and --place-from-episode are mutually exclusive")
        want = uvc60_short_id(args.episode)
        matches = [p for p in placements if p.short_id == want]
        if not matches:
            raise SystemExit(f"--place-from-episode wanted {want!r}, not in {args.placements}")
        place_source = (f"episode {args.episode} -> {want} [Eric說 2026-09-21] t1..t60 in order, "
                        f"t41 re-recorded as the last episode [2026-09-13.md]")
    else:
        matches = [p for p in placements if p.short_id == args.place or p.placement_id == args.place]
        if args.place and not matches:
            raise SystemExit(f"no placement {args.place!r} in {args.placements}")
        place_source = f"--place {args.place!r}" if args.place else "first row of the CSV (arbitrary)"
    place = matches[0] if matches else placements[0]
    usd_str = getattr(scene_cfg.object.spawn, "usd_path", "") or ""
    is_paper_cup = "paper_cup" in usd_str or args.object is None or "paper_cup" in args.object
    cup_z = S.TABLE_TOP_Z if is_paper_cup else (S.TABLE_TOP_Z + S.CUP_HEIGHT / 2.0)
    scene_cfg.object.init_state.pos = (place.x_m, place.y_m, cup_z)
    print(f"object placement {place.short_id} ({place.placement_id}) "
          f"at x={place.x_m:.3f} y={place.y_m:.3f} z={cup_z:.3f} [{place_source}]")

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
scene = InteractiveScene(scene_cfg)

# 🔴 2026-09-29: no contact between the object and the two jaws (link6/link7). With both jaws written
# kinematically, ep0 (current constants) attached the cup with its wall over the mimic jaw; the
# object's pose is forced every frame, so each physics step shoved that undriven jaw open before the
# camera captured it. Toggling the object's collisionEnabled at attach/detach instead crashed GPU
# PhysX (illegal memory access in the GPU narrowphase) -- a filtered pair is set once, at parse time.
# Cost: before the attach, a jaw passes through the cup instead of knocking it; the rest of the arm
# still collides.
from pxr import Usd, UsdPhysics  # noqa: E402, PLC0415

_jaws = [q.GetPath() for root in sim_utils.find_matching_prims(scene_cfg.robot.prim_path)
         for q in Usd.PrimRange(root) if q.GetName() in ("link6", "link7") and q.HasAPI(UsdPhysics.RigidBodyAPI)]
for _o in sim_utils.find_matching_prims(scene_cfg.object.prim_path):
    _rel = UsdPhysics.FilteredPairsAPI.Apply(_o).CreateFilteredPairsRel()
    for _j in _jaws:
        _rel.AddTarget(_j)
print(f"object <-> jaw contact filtered: {[str(j) for j in _jaws]}")
if len(_jaws) != 2:
    raise SystemExit(f"expected link6 and link7 as rigid bodies under the robot, found {_jaws}")
sim.reset()

import numpy as np  # noqa: E402

for key, spec in lens.items():
    K_isaac = torch.tensor(CD.isaac_K(spec["render_w"], spec["render_h"], spec["f"]), dtype=torch.float32, device=sim.device)
    scene[key].set_intrinsic_matrices(K_isaac.unsqueeze(0))
    got = scene[key].data.intrinsic_matrices[0].cpu().numpy()
    # build the remap from what Isaac SAYS it renders, not from what was asked for
    spec["f_isaac"] = float(got[0, 0])
    if abs(spec["f_isaac"] - spec["f"]) > 1e-3 * spec["f"] or abs(got[1, 1] - got[0, 0]) > 1e-6:
        raise SystemExit(f"{key}: asked Isaac for f={spec['f']:.3f}, it reports K={got.tolist()}")
    map_x, map_y, coverage = CD.build_render_to_real_maps(spec["model"], spec["render_w"], spec["render_h"], spec["f_isaac"])
    if coverage < 1.0:
        raise SystemExit(f"{key}: remap samples outside the render ({coverage*100:.2f}% inside) -- render_spec is too small")
    spec["maps"] = (map_x, map_y)
    print(f"{key}: Isaac K = f {got[0, 0]:.2f}, c ({got[0, 2]:.1f}, {got[1, 2]:.1f}); remap coverage {coverage*100:.1f}%")

fl_pose = None
if not args.front_left_extrinsics:
    # S4 §5-5 camera eras: a recorded dataset is replayed with the camera pose of the session it was recorded in
    _ds = os.path.basename(os.path.normpath(args.dataset_root))
    _rel = S.front_left_extrinsics_for_dataset(_ds)
    if _rel:
        args.front_left_extrinsics = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _rel)
        print(f"cam_front_left: {_ds} was recorded in camera era {S.DATASET_CAMERA_ERA[_ds]} -> {_rel}")
    else:
        print(f"cam_front_left: no camera era registered for {_ds!r} -> look-at from scene_constants (latest era, no roll)")
if args.front_left_extrinsics:
    # full measured pose (roll included), DR jitter in the camera's own axes
    fl_pos, fl_R, fl_json = front_left_pose_from_json(args.front_left_extrinsics)
    pos = [p + d for p, d in zip(fl_pos, cam_dpos)]
    R = np.asarray(fl_R) @ small_rotation(*cam_drot)
    quat = R_to_quat_wxyz(R)
    scene["cam_front_left"].set_world_poses(torch.tensor([pos], device=sim.device), torch.tensor([quat], device=sim.device),
                                            convention="ros")
    fl_pose = {"pos": pos, "quat_wxyz_ros": quat, "source": args.front_left_extrinsics}
    print(f"cam_front_left: measured pose from {args.front_left_extrinsics} -> pos {[round(v, 4) for v in pos]} quat {[round(v, 4) for v in quat]}")
else:
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
# 🔴 2026-09-29: the mimic jaw is WRITTEN too, not left to the PhysX mimic constraint. That constraint
# is soft and nearly undamped (naturalFrequency 25, dampingRatio 0.005; gripper_joint_2 has no drive
# of its own, sim/README "Two gaps" §2) and reaches only ~50% amplitude. Under a kinematic replay --
# arm teleported every frame, attached object pose-forced into the jaws -- it oscillated: in the
# 9/23 ep0 render one jaw swung open mid-carry while the recording holds 49.9 (closed) from frame 230
# to 370. Writing it as K.MIMIC_JOINT's multiplier x its master (URDF: -1) keeps the constraint
# satisfied, so nothing fights it.
_mimic, _mimic_src, _mimic_mult = K.MIMIC_JOINT
_write_mimic = _mimic in robot.joint_names
joint_idx = [robot.joint_names.index(n) for n in joint_names + ([_mimic] if _write_mimic else [])]
zeros = torch.zeros(1, len(joint_idx), device=sim.device)
if not _write_mimic:
    print(f"⚠️  {_mimic} not in the articulation -- the mimic jaw is left to the physics constraint")

grasp = None
if not args.skip_grasp:
    grasp = ScriptedGraspAttach(GraspAttachConfig(attach_radius_m=args.attach_radius))
    thr = grasp.calibrate([row[5] for row in states_deg])
    print(f"grasp attach threshold: {thr:.2f} deg (from this episode's own gripper trace)")

os.makedirs(args.out, exist_ok=True)

def write_pose(t: int) -> torch.Tensor:
    rad = list(JM.row_to_sim_rad(states_deg[t]))
    if _write_mimic:
        rad.append(_mimic_mult * rad[joint_names.index(_mimic_src)])
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

if fl_pose is not None:
    # read the pose back in the SAME (ros) convention it was written in -- a convention mix-up shows
    # up here as a large angle, instead of as a plausible-looking but rotated image
    got_p = scene["cam_front_left"].data.pos_w[0].cpu().numpy()
    got_R = np.asarray(quat_wxyz_to_R(scene["cam_front_left"].data.quat_w_ros[0].cpu().numpy().tolist()))
    dp = float(np.linalg.norm(got_p - np.asarray(fl_pose["pos"])))
    dang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(got_R.T @ np.asarray(quat_wxyz_to_R(fl_pose["quat_wxyz_ros"]))) - 1) / 2))))
    print(f"cam_front_left pose read back: |dpos| {dp*1000:.3f} mm, rotation {dang:.4f} deg")
    if dp > 1e-3 or dang > 0.05:
        raise SystemExit("front-left camera pose did not take -- check set_world_poses' convention in this Isaac Lab")
    fl_pose["readback"] = {"dpos_m": dp, "drot_deg": dang}

# 🔴 2026-09-29: one carry per episode. ep0 (current constants) released the cup into the bin at
# frame 382, then the gripper closed again at 401 with the cup inside the 14 cm attach radius, and
# the cup was "picked up" out of the bin. A regrasp at the pick site (a real behaviour in some
# episodes) is still allowed: attaching stops only once the object was released away from its start.
REGRASP_MAX_M = 0.10
obj_start = obj.data.root_pos_w[0].clone()
grasp_done = False

records = []
from PIL import Image  # noqa: E402, PLC0415

for n, t in enumerate(frames):
    write_pose(t)

    attached = False
    tcp_obj_dist = None
    if grasp is not None:
        tcp, tcp_q = tcp_pose_w(robot)   # measured fingertip, not the link6/link7 pivots
        obj_p = obj.data.root_pos_w[0]
        # 🔴 recorded every frame on purpose. When a grasp does not fire, the ONLY question that
        # matters is "how close did the TCP actually get to the object", and guessing at it (too
        # tight a radius? object rolled? wrong TCP frame?) wasted a cycle on 2026-09-21.
        tcp_obj_dist = float(torch.norm(obj_p - tcp).item())
        if not grasp_done:
            want_p, want_q = grasp.step(t, states_deg[t][5], tcp, tcp_q, obj_p, obj.data.root_quat_w[0])
            if grasp.attached:
                obj.write_root_pose_to_sim(torch.cat([want_p, want_q]).unsqueeze(0))
                obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=want_p.device))
                attached = True
            elif (grasp.events and grasp.events[-1].kind == "detach"
                  and float(torch.norm(obj_p[:2] - obj_start[:2])) > REGRASP_MAX_M):
                grasp_done = True
                print(f"  frame {t}: released {float(torch.norm(obj_p[:2] - obj_start[:2]))*100:.0f} cm from the "
                      f"pick site -- no further attaches this episode")

    scene.write_data_to_sim()
    for _ in range(max(1, args.steps_per_frame)):
        sim.step()
        scene.update(sim.get_physics_dt())

    rec = {"frame": t, "timestamp_s": t / DATASET_FPS, "action": actions_deg[t],
           "observation_state": states_deg[t], "grasp_attached": attached,
           "tcp_object_dist_m": tcp_obj_dist}
    for key, name in (("cam_wrist", "wrist"), ("cam_front_left", "front-left")):
        rgb = scene[key].data.output["rgb"][0][..., :3].detach().cpu().numpy()
        fn = f"f{t:05d}_{name}.png"
        if key in lens:
            if args.save_ideal:
                Image.fromarray(rgb).save(os.path.join(args.out, f"f{t:05d}_{name}_ideal.png"))
            rgb = CD.apply_distortion(np.ascontiguousarray(rgb), lens[key]["maps"])
        Image.fromarray(rgb).save(os.path.join(args.out, fn))
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
    "offset_rad": dict(JM.OFFSET_RAD),
    "offset_delta_deg": offset_delta_deg,
    "driven_by": "write_joint_state_to_sim (kinematic replay; no drive gains, S5 §4)",
    "dr": dr,
    "grasp": {
        "mode": "skipped" if grasp is None else "scripted_attach_detach",
        "events": [] if grasp is None else [
            {"frame": e.frame, "kind": e.kind, "gripper_deg": e.gripper_reading, "dist_m": e.dist_m}
            for e in grasp.events
        ],
    },
    "cameras": {
        "front-left": {
            "lens": lens["cam_front_left"]["source"] if "cam_front_left" in lens else f"scene_constants HFOV {S.HFOV_FRONT_LEFT_DEG} deg, centred, no distortion",
            "pose": fl_pose if fl_pose is not None else "scene_constants look-at (no roll)",
        },
        "wrist": {
            "lens": lens["cam_wrist"]["source"] if "cam_wrist" in lens else f"scene_constants HFOV {S.HFOV_WRIST_DEG} deg PLACEHOLDER",
            "pose": "scene_constants CAM_WRIST_OFFSET_* (CAD); hand-eye NOT calibrated",
        },
        "render_sizes": {k: [v["render_w"], v["render_h"], v["f_isaac"]] for k, v in lens.items()},
    },
    "fidelity": {
        # True only when BOTH cameras have a measured lens AND a measured pose; the wrist hand-eye is
        # still open (2026-10-08: poses A/B disagree by 3.6 cm / 6.5 deg), so this stays False for now.
        "geometry_aligned": False,
        "geometry_aligned_per_camera": {
            "front-left": bool("cam_front_left" in lens and fl_pose is not None),
            "wrist": False,
        },
        "geometry_note": ("front-left: measured lens + ArUco pose (T1/T2, 10/07 camera position); "
                          "wrist: measured lens only, pose is the CAD offset (hand-eye pending)")
                         if ("cam_front_left" in lens and fl_pose is not None)
                         else "PLACEHOLDER camera pose and/or lens, S5 gap 4 (T1/T2) not wired in",
        "object_matches_source": False,
        "object_note": (
            (f"rendered USD {os.path.basename(args.object)}" if args.object else
             f"rendered a primitive cylinder cup at the measured dimensions "
             f"(opening {S.CUP_OPENING_DIA*100:.1f}cm / base {S.CUP_BASE_DIA*100:.1f}cm / "
             f"height {S.CUP_HEIGHT*100:.1f}cm, upright) -- right size and pose, placeholder "
             f"appearance, taper not modelled")
        ),
        "placement_matches_source": bool(args.placements and args.place_from_episode),
        "placement_note": place_source,
        "placement_id": None if place is None else place.placement_id,
        "placement_short_id": None if place is None else place.short_id,
        "gripper_amplitude_verified": False,
        "gripper_note": ("both jaws written kinematically (mimic jaw = -1 x master), so the mimic "
                         "constraint's ~50% amplitude no longer applies; the units->angle conversion "
                         "itself is still unverified (joint_mapping GRIPPER_ZERO_DEG)"
                         if _write_mimic else "mimic constraint reaches ~half the intended amplitude (gap 2 residual)"),
    },
    "frames": records,
}
with open(os.path.join(args.out, "manifest.json"), "w") as fh:
    json.dump(manifest, fh, indent=2)

print(f"\nrendered {len(records)} frame(s) x 2 cameras -> {args.out}")
if grasp is not None:
    print(f"grasp events: {[(e.frame, e.kind) for e in grasp.events]}")
    dists = [(r["tcp_object_dist_m"], r["frame"]) for r in records if r["tcp_object_dist_m"] is not None]
    if dists:
        dmin, fmin = min(dists)
        print(f"TCP-to-object: closest {dmin*100:.1f} cm at frame {fmin} "
              f"(attach radius {grasp.cfg.attach_radius_m*100:.0f} cm)")
        if not grasp.events:
            print("  🔴 no attach fired. If the closest distance above is larger than the radius, the "
                  "object was never within reach of the TCP -- check the placement, whether the object "
                  "rolled while settling, and whether link6/link7's midpoint is really the grasp centre.")
print("🔴 fidelity flags (all recorded in manifest.json, stage 2 must carry them into dataset meta):")
for k, v in manifest["fidelity"].items():
    if not k.endswith("_note"):
        print(f"    {k} = {v}")
print("next: scripts/sim_replay_augment.py to assemble a LeRobotDataset from this")
import os
os._exit(0)

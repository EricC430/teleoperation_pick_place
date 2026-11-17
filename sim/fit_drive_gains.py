"""Score (not yet: calibrate) the provisional drive gains against a real recorded trajectory.

🔴 SCOPE CHANGED 2026-09-18, read this before assuming what this is for. This was written as S5
gap 1, on the premise that S5's replay pipeline drives the sim arm with PD position control and
therefore needs gains/torque that reproduce the real arm's dynamics. `[Eric說 2026-09-18]` pointed
out that premise was wrong -- a replay does not need to re-derive an arm trajectory that is
already recorded in `observation.state`. S5's arm is now a kinematic replay
(`write_joint_position_to_sim`), where gains and effort limits do not affect the rendered pose at
all (`docs/specs/S5_sim_replay_augmentation.md` §4, `[Eric決定 2026-09-18]`).

**So this script now serves S4, not S5** -- live teleop-in-sim has no pre-recorded outcome to copy,
so there the simulated arm genuinely must track live leader commands under its own control loop,
and these gains are load-bearing. Everything below still works and the measurements still stand;
only the spec it answers to changed.

Context: `sim/README.md` "the provisional drive gains cannot hold the arm's own weight" -- holding
the all-zero pose for 1s drifts 19-35 deg. `omx_constants.stiffness/damping` is
`stall_torque / 5deg` and `5% of stiffness` -- dimensionally honest, reproducible, and never fit
against anything real. `S5_sim_replay_augmentation.md` §2 gap 1 / `docs/decisions.md` D029 say the
fix is a fit against a real `(action, observation.state)` trajectory, not another guessed
constant. This script replays ONE real episode's `action` open-loop through the Isaac Lab
articulation (position control, no vision, no object) and scores the result against that same
episode's `observation.state`, for one candidate (stiffness_scale, damping_scale) pair.

🔴 Runs ONE combo per process, on purpose. An earlier draft tried to sweep a grid of scales inside
a single simulation_app by rebuilding the scene per combo -- rebuilding/destroying an
InteractiveScene+Articulation mid-process is not something this repo's other sim/ scripts do, and
without a container to actually test it in, claiming that works would be a guess wearing a
calibration's clothes. Sweep externally instead (cheap: each run is a few hundred physics steps,
no rendering):

    for s in 1 2 4 8 16; do
      for d in 1 2 4; do
        ./sim/run_in_container.sh fit_drive_gains.py --dataset-root "$DS" --episode 0 \\
            --stiffness-scale "$s" --damping-scale "$d" --headless \\
            --out "/workspace/test_isaaclab/omx_sim/gain_fit_ep0_s${s}_d${d}.json"
      done
    done

Then compare the printed/written `agg_p50_deg` across the output files by hand -- there is no
aggregator script here; writing one before a single real number has come back would be exactly
the "guessed a bigger constant" mistake `sim/README.md` already warns against.

🔴 2026-09-18: runs made before the merge read `.pos` as degrees (see `joint_mapping.py` docstring) --
their numbers are void; rerun. (`actions_deg`/`states_deg` below hold LeRobot `.pos` units, the
names predate the fix.)

⚠️ UNVERIFIED PREREQUISITE (inherited from `joint_mapping.py`): the recorded `.pos` -> sim-radians
SIGN convention defaults to +1 for all six joints and has not been confirmed by the five-pose test
(S4 §5-1). A large, joint-specific error here could be a sign bug, not a gain problem -- read
`joint_mapping.py`'s docstring before reading a bad per-joint number as "this joint's gain is off".

The dataset lives in this repo's git history, not under `~/isaaclab_volume` --
`run_in_container.sh` does not copy it (only `sim/*.py` and the placement CSVs). Copy the episode
data in first (parquet only -- no need for the video files this script never reads):

    mkdir -p ~/isaaclab_volume/omx_sim/dataset
    cp -r data/huggingface/lerobot/ericc430/omx_pick_place_pilot_uvc_60/data \\
          ~/isaaclab_volume/omx_sim/dataset/

    ./sim/run_in_container.sh fit_drive_gains.py \\
        --dataset-root /workspace/test_isaaclab/omx_sim/dataset --episode 0 --headless \\
        --out /workspace/test_isaaclab/omx_sim/gain_fit_ep0_baseline.json

`--dataset-root` needs a `data/chunk-*/file-*.parquet` layout underneath it (i.e. point it at the
directory that CONTAINS `data/`, matching the HF dataset repo layout) -- this script does not need
`meta/info.json`, the column order is hardcoded from `joint_mapping.DATASET_JOINT_ORDER` and
cross-checked against `omx_constants.JOINTS` at import time.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import isaac_boot  # noqa: E402  (Isaac Lab 3.0 launcher, see isaac_boot.py)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset-root", required=True, help="dir containing data/chunk-*/file-*.parquet")
parser.add_argument("--episode", type=int, required=True)
parser.add_argument("--stiffness-scale", type=float, default=1.0, help="multiplies omx_constants.stiffness()")
parser.add_argument("--damping-scale", type=float, default=1.0, help="multiplies omx_constants.damping()")
parser.add_argument(
    "--effort-scale",
    type=float,
    default=1.0,
    help="multiplies the actuator effort_limit (default: the real motor's rated stall torque, "
    "omx_scene_cfg.py). A stiffness-scale sweep with this left at 1.0 only tests gains UP TO "
    "whatever torque the real motor could actually produce -- if the commanded torque is already "
    "saturating that cap, more stiffness does nothing, sign-correct or not. This flag tests "
    "whether the ceiling itself, not the gain below it, is what's binding.",
)
parser.add_argument(
    "--sign-override",
    default="",
    help="e.g. 'shoulder_lift=-1' or 'shoulder_lift=-1,elbow_flex=-1' -- overrides joint_mapping.SIGN "
    "for the named joint(s) for this run only, to test a candidate sign without waiting for the "
    "five-pose test (S4 §5-1). See joint_mapping.py's docstring for why SIGN defaults to +1.",
)
parser.add_argument("--physics", choices=("physx", "newton_mjwarp"), default="physx",
                    help="physics backend (Isaac Lab 3.0). physx = what every S5/S9 number was measured with; "
                    "newton_mjwarp = Newton + MuJoCo-Warp, 2026-11-16 first try (see docs/worklog/2026-11-16_newton_try.md)")
parser.add_argument("--num-envs", type=int, default=1,
                    help="parallel copies of the arm, all driven by the same targets; env 0 is the one scored. "
                    "1 reproduces the original behaviour; >1 is for throughput measurements")
parser.add_argument("--out", required=True, help="where to write the JSON report")
isaac_boot.add_args(parser)
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

simulation_app = isaac_boot.start(args)

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

if args.physics != "physx":
    # Newton refuses the original asset (one dangling reference, see newton_asset_fix.py); use the fixed copy unless told otherwise
    os.environ.setdefault("OMX_USD", "/workspace/test_isaaclab/assets/omx_f_generated_newton/omx_f.usd")

import joint_mapping as JM  # noqa: E402
import omx_constants as K  # noqa: E402
import omx_scene_cfg as SC  # noqa: E402

RAD2DEG = 180.0 / 3.141592653589793
DATASET_FPS = 15.0  # meta/info.json "fps" -- every campaign so far agrees; not re-read from disk

if args.sign_override:
    for entry in args.sign_override.split(","):
        name, val = entry.split("=")
        name = name.strip()
        if name not in JM.SIGN:
            raise SystemExit(f"--sign-override: {name!r} is not one of {JM.DATASET_JOINT_ORDER}")
        JM.SIGN[name] = float(val)
    print(f"⚠️  SIGN overridden for this run only (not saved back to joint_mapping.py): {JM.SIGN}")


def load_episode(dataset_root: str, episode: int):
    """Pure-pyarrow parquet load -- avoids depending on the `lerobot` pip package, which
    `/isaac-sim/python.sh` (the interpreter `run_in_container.sh` uses) is not known to have."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit(
            "pyarrow not importable in this python -- install it in the container once: "
            "/isaac-sim/python.sh -m pip install pyarrow"
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
    actions = [r[1] for r in rows]
    states = [r[2] for r in rows]
    return actions, states


actions_deg, states_deg = load_episode(args.dataset_root, args.episode)
n_frames = len(actions_deg)
print(f"episode {args.episode}: {n_frames} frames loaded from {args.dataset_root}")
print(f"stiffness_scale={args.stiffness_scale}  damping_scale={args.damping_scale}  effort_scale={args.effort_scale}  (1.0 = omx_constants provisional)")

actions_rad = [JM.row_to_sim_rad(r) for r in actions_deg]
states_rad = [JM.row_to_sim_rad(r) for r in states_deg]

substeps = round(120.0 / DATASET_FPS)  # sim solver runs at 1/120s; hold each dataset-fps target that many steps


@configclass
class GainFitSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=1200.0, color=(1.0, 1.0, 1.0))
    )
    robot: ArticulationCfg = SC.omx_articulation_cfg("{ENV_REGEX_NS}/Robot")


scene_cfg = GainFitSceneCfg(num_envs=args.num_envs, env_spacing=2.0, replicate_physics=(args.num_envs > 1 or args.physics != "physx"))   # Newton's cloner needs it even for 1 env

# scale every joint's gains uniformly -- see the module docstring for why this is a global,
# not per-joint, scale for the first pass.
for j in K.JOINTS:
    act = scene_cfg.robot.actuators[j.urdf_name]
    act.stiffness = K.stiffness(j) * args.stiffness_scale
    act.damping = K.damping(j) * args.damping_scale
    act.actuator_effort_limit = j.motor.stall_torque_nm * args.effort_scale   # 3.0: split rated / solver limit,
    act.joint_effort_limit = j.motor.stall_torque_nm * args.effort_scale      # both were `effort_limit` in 2.x

# start at the recorded first frame's pose, not the zero pose -- avoids an irrelevant transient
# from "sim starts folded, real starts mid-reach" dominating the error metric. Real hardware
# cannot teleport into position; sim can, and S5's own §5 item 1 asks for a meaningful comparison,
# not a replay of a problem this script isn't trying to measure.
joint_names = [j.urdf_name for j in K.JOINTS]
first_pose = {name: val for name, val in zip(joint_names, actions_rad[0])}
# Isaac Lab 3.0 refuses an initial pose outside the joint limits (articulation _validate_cfg raises ValueError; 2.x did not
# stop here). The recorded first state can sit just outside the site limits (ep0: joint3 1.642 rad vs +1.571), so clamp it
# into the limits -- the same transient the real arm resolves by its own end-stop. Printed, never silent.
for _j in K.JOINTS:
    _lo, _hi = K.effective_limits_rad(_j, apply_site_limits=True)
    _v = first_pose[_j.urdf_name]
    if not (_lo <= _v <= _hi):
        print(f"  init pose clamped: {_j.urdf_name} {_v:.3f} rad -> [{_lo:.3f}, {_hi:.3f}]")
        first_pose[_j.urdf_name] = min(max(_v, _lo), _hi)
first_pose[K.MIMIC_JOINT[0]] = 0.0
scene_cfg.robot.init_state.joint_pos = first_pose

if args.physics == "newton_mjwarp":
    # Newton + MuJoCo-Warp. Starting profile from docs/source/how-to/solver_tuning_mjwarp.rst ("explicit baseline"),
    # NOT a translation of the PhysX numbers (the official asset guide says not to). debug_mode on: this is a first try.
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # noqa: E402

    physics_cfg = NewtonCfg(solver_cfg=MJWarpSolverCfg(solver="newton", integrator="implicitfast"),
                            num_substeps=1, debug_mode=True)
else:
    physics_cfg = None   # SimulationCfg default = Isaac Sim PhysX, exactly as before
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device, physics=physics_cfg))
scene = InteractiveScene(scene_cfg)
sim.reset()

robot = scene["robot"]
joint_idx = [robot.joint_names.index(n) for n in joint_names]

errors_deg = [[] for _ in range(6)]  # per dataset joint, degrees

target = robot.data.default_joint_pos.torch.clone()
import time  # noqa: E402
torch.cuda.synchronize()
_t_loop = time.time()
_n_sim_steps = 0
for t in range(n_frames - 1):
    for k, i in enumerate(joint_idx):
        target[:, i] = actions_rad[t][k]
    for _ in range(substeps):
        robot.actuators.target_command.set_position_index(value=target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        _n_sim_steps += 1

    real_next_rad = states_rad[t + 1]
    for k, i in enumerate(joint_idx):
        sim_rad = robot.data.joint_pos.torch[0, i].item()
        err_deg = abs(sim_rad - real_next_rad[k]) * RAD2DEG
        errors_deg[k].append(err_deg)

    if t % max(1, (n_frames // 10)) == 0:
        print(f"  frame {t}/{n_frames}")


torch.cuda.synchronize()
_loop_s = time.time() - _t_loop
print(f"\n[{args.physics}] {_n_sim_steps} sim steps x {args.num_envs} env(s) in {_loop_s:.2f} s -> "
      f"{_n_sim_steps / _loop_s:.0f} steps/s, {_n_sim_steps * args.num_envs / _loop_s:.0f} env-steps/s "
      f"(includes the per-step .item() readback the scoring does)")


def pctl(vals, p):
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, round(p / 100.0 * (len(s) - 1))))
    return s[idx]


print(f"\n{'joint':<16}{'p50(deg)':>10}{'p95(deg)':>10}{'max(deg)':>10}")
report_joints = {}
for name, errs in zip(JM.DATASET_JOINT_ORDER, errors_deg):
    p50, p95, mx = pctl(errs, 50), pctl(errs, 95), max(errs)
    print(f"{name:<16}{p50:>10.2f}{p95:>10.2f}{mx:>10.2f}")
    report_joints[name] = {"p50_deg": p50, "p95_deg": p95, "max_deg": mx}

agg_p50 = sum(v["p50_deg"] for v in report_joints.values()) / 6.0
print(f"\naggregate (mean of per-joint p50): {agg_p50:.2f} deg")
print("compare this file's agg_p50_deg against other --stiffness-scale/--damping-scale runs by hand (see module docstring)")
print("⚠️  SIGN is [未確認] for every joint (joint_mapping.py) -- a bad number here could be a sign bug, not a gain bug")

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as fh:
    json.dump(
        {
            "source_episode_id": args.episode,
            "dataset_root": args.dataset_root,
            "n_frames": n_frames,
            "stiffness_scale": args.stiffness_scale,
            "effort_scale": args.effort_scale,
            "damping_scale": args.damping_scale,
            "sign_unverified": True,
            "per_joint": report_joints,
            "agg_p50_deg": agg_p50,
            "physics": args.physics,
            "num_envs": args.num_envs,
            "sim_steps": _n_sim_steps,
            "loop_seconds": _loop_s,
            "steps_per_s": _n_sim_steps / _loop_s,
            "env_steps_per_s": _n_sim_steps * args.num_envs / _loop_s,
        },
        fh,
        indent=2,
    )
print(f"\nwrote {args.out}")
import os
os._exit(0)

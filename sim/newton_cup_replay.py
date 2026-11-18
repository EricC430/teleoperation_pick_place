"""Newton (MuJoCo-Warp) replay of ONE recorded episode with a free thin-shell paper cup -- Isaac Lab 3.0, no rendering.

Purpose (2026-11-16): does Newton's contact handling keep the fingers from going THROUGH the 1 mm cup wall at 120 Hz,
where Isaac Sim PhysX needs 480 Hz (docs/worklog/2026-10-31_sim_dr_pipeline.md)? The knobs under test are the per-shape
contact margin / gap (`NewtonCollisionCfg`) and Newton's own collision pipeline with SDF mesh contacts
(`NewtonCfg.collision_cfg`, "mesh-mesh collision via SDF").

Output: OUT/physics_log.json in the format of sim/physics_replay.py (`log[i]` = frame, cup_xyz, q_sim, q_cmd, cup_pose,
link_poses as [x, y, z, qw, qx, qy, qz]; `link_bodies`), so sim/penetration.py and the replay scoring read it unchanged.
Quaternions in the LOG stay (w, x, y, z) like every other log of this repo; Isaac Lab 3.0 data is (x, y, z, w) and is
converted at the boundary.

What is the same as physics_replay.py: the episode (episode.json / gripper.json from scripts/prep_closed_loop_replay.py),
the link-marker-fit base pose (`world_from_link0` of fit_candidate.json), body targets = recorded link angles, gripper
target = the recorded ACTION, targets interpolated over the sub-steps of each 15 fps frame, 60 settling steps first, the
patched asset omx_f_real.usda (the Newton-loadable copy, sim/newton_asset_fix.py), the asset's own motor gains / force
limits, joint limits widened to +-360 deg for joint1-5.

What is NOT the same (so the comparison is "Newton, this setup" vs "PhysX, that setup", not a controlled A/B of engines):
- no self-collision (PhysX replay has it on), no riser, no bin (so "in the bin" is not scored here), ground plane at z=0
  instead of the big table slab;
- SDF colliders: the fingers' and cup's meshes get `physics:approximation = sdf` + `NewtonSDFCollisionCfg` (resolution
  `--sdf-res`); PhysX used SDF on every link;
- the mimic jaw (gripper_joint_2) is a Newton mimic equality constraint, not the PhysX spring tuned to 300 / 1.0;
- the end_effector_link zero-mass fix (1e-4 kg) is not applied;
- MJWarp does not enforce `joint_velocity_limit`.

    ./sim/run_in_container.sh newton_cup_replay.py --ep $G/dr/src/cl_20261014_151208/cl_ep00 \\
        --cands $G/dr/src/cl_20261014_151208/fit_candidate.json --cup-asset $G/dr/ctrl_cupref_cl1014/objects/cup_ref.usd \\
        --hz 120 --gap 0.002 --margin 0.0 --contacts newton --out $G/dr/newton_try/ep00_120hz_gap2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import isaac_boot

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--ep", required=True, help="episode dir (episode.json, gripper.json)")
parser.add_argument("--cands", required=True, help="fit_candidate.json")
parser.add_argument("--cand", default="fit")
parser.add_argument("--cup-asset", required=True, help="cup USD (hollow shell, e.g. dr/.../objects/cup_ref.usd)")
parser.add_argument("--asset", default="/workspace/test_isaaclab/assets/omx_f_generated_newton/omx_f_real.usda")
parser.add_argument("--hz", type=int, default=120, help="physics rate (PhysX baseline: 480)")
parser.add_argument("--contacts", choices=("mujoco", "newton"), default="newton",
                    help="mujoco = MJWarp's own collision; newton = Newton's collision pipeline (SDF mesh-mesh)")
parser.add_argument("--gap", type=float, default=None, help="contact gap [m] on the cup and all robot colliders (None = Newton default)")
parser.add_argument("--margin", type=float, default=None, help="contact margin [m] on the cup and all robot colliders")
parser.add_argument("--sdf-res", type=int, default=256, help="Newton SDF max grid resolution")
parser.add_argument("--hydroelastic", action="store_true", help="Newton hydroelastic SDF contacts (needs watertight meshes)")
parser.add_argument("--nconmax", type=int, default=256)
parser.add_argument("--njmax", type=int, default=1024)
parser.add_argument("--max-frames", type=int, default=0)
parser.add_argument("--no-selfcol", action="store_true", help="turn self-collision OFF (physics_replay.py default is ON)")
parser.add_argument("--no-riser", action="store_true")
parser.add_argument("--no-bin", action="store_true")
parser.add_argument("--mimic-solref", type=float, nargs=2, default=None, metavar=("A", "B"),
                    help="MuJoCo solref of the gripper_joint_2 mimic equality constraint. Default (0.02 1.0) = timeconst 20 ms, soft. "
                    "Positive = (timeconst s, damping ratio); NEGATIVE = (-stiffness, -damping) direct, which MuJoCo does not clamp to 2*dt. "
                    "PhysX asset: natural frequency 300 Hz, damping ratio 1.0.")
parser.add_argument("--mimic-mode", choices=("mimic", "pd", "newton"), default="mimic",
                    help="mimic = the asset's mimic constraint (default). pd = remove it and drive gripper_joint_2 with its own PD "
                    "(same gains / force limit as gripper_joint_1, target = -target of gripper_joint_1). A DEVIATION from physics_replay.py, "
                    "a diagnostic for why the mimic finger does not mirror in Newton.")
parser.add_argument("--cup-dx-mm", type=float, default=0.0, help="move the cup this far in x (physics_replay.py cup_dx_mm=500: out of the arm's way, no finger contact)")
parser.add_argument("--clamp-grip", action="store_true", help="clip the gripper target at the joint limit (>= 0): the recorded action goes to about -10 deg, i.e. a standing push INTO the limit")
parser.add_argument("--grip-limit-solref", type=float, nargs=2, default=None, metavar=("TIMECONST", "DAMPRATIO"),
                    help="RAW MuJoCo joint-limit solref (mjc:solreflimit) on gripper_joint_1. Without it Newton derives the limit solref from "
                    "joint_limit_ke/kd scaled by the DOF's inverse inertia; for the light finger joints that came out (0.0, 4.1) = infinitely "
                    "stiff in time, over-damped, and the limit did not hold (measured: j1 sat at -10 deg, limit 0).")
parser.add_argument("--grip-limit-solimp", type=float, nargs=5, default=None, metavar=("DMIN", "DMAX", "WIDTH", "MID", "POWER"),
                    help="mjc:solimplimit on gripper_joint_1 (default 0.9 0.95 0.001 0.5 2: a limit that cancels only 90-95% of the push)")
parser.add_argument("--grip-armature", type=float, default=0.0,
                    help="armature [kg m^2] written onto gripper_joint_1 AND gripper_joint_2 after reset. The asset has 0; MuJoCo's soft "
                    "constraints are inertia-scaled, so a 0.52 N.m motor on a ~1e-6 kg m^2 finger overpowers any limit / mimic. A real servo "
                    "reflects its rotor inertia through the gear (e.g. 1e-8 x 288^2 ~ 8e-4, a ROUGH figure, not measured).")
parser.add_argument("--contact-solref", type=float, nargs=2, default=None, metavar=("A", "B"),
                    help="MuJoCo contact solref (mjc:solref) on EVERY collider (arm + cup). Positive = (timeconst s, damping ratio), MuJoCo "
                    "clamps timeconst to >= 2*dt; negative = (-stiffness, -damping) direct. Default (0.02, 1.0).")
parser.add_argument("--contact-solimp", type=float, nargs=5, default=None, metavar=("DMIN", "DMAX", "WIDTH", "MID", "POWER"),
                    help="MuJoCo contact solimp (mjc:solimp) on every collider. Default (0.9 0.95 0.001 0.5 2): the contact cancels only 90-95%% of the "
                    "penetrating acceleration; closer to 1 = harder")
parser.add_argument("--condim", type=int, default=None, choices=(1, 3, 4, 6))
parser.add_argument("--det-mode", choices=("not_guaranteed", "run_to_run", "gpu_to_gpu"), default="not_guaranteed",
                    help="NewtonCfg.deterministic_mode (default not_guaranteed = Newton's own default)")
parser.add_argument("--diag-range", type=int, nargs=2, default=(70, 140), metavar=("FROM", "TO"))
parser.add_argument("--diag", action="store_true", help="per-frame MuJoCo-Warp contact diagnostics for frames 70-140 into OUT/contact_diag.json")
parser.add_argument("--print-shapes", action="store_true", help="debug: list the collision shapes Newton built")
parser.add_argument("--out", required=True)
isaac_boot.add_args(parser)
args = parser.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
simulation_app = isaac_boot.start(args)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation as Rr  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg, UsdPhysicsMeshCollisionCfg, UsdPhysicsRigidBodyCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # noqa: E402
from isaaclab_newton.sim.schemas import NewtonArticulationCfg, NewtonCollisionCfg, NewtonSDFCollisionCfg  # noqa: E402

EP = args.ep
E = json.load(open(os.path.join(EP, "episode.json")))
G = json.load(open(os.path.join(EP, "gripper.json")))
C = json.load(open(args.cands))[args.cand]
QB = np.array(E["q_body_rad"])
GRIP = np.array(G["gripper_target_rad_from_action"], dtype=float)
if args.clamp_grip:
    GRIP = np.maximum(GRIP, 0.0)
FPS = float(E["fps"])
N = len(QB)
DT = 1.0 / args.hz
SUB = int(round(1.0 / FPS / DT))
os.makedirs(args.out, exist_ok=True)

M = np.array(C["world_from_link0"])
base_pos = tuple(float(v) for v in M[:3, 3])
qx, qy, qz, qw = Rr.from_matrix(M[:3, :3]).as_quat()   # scipy: (x, y, z, w) == Isaac Lab 3.0's convention
base_rot = (float(qx), float(qy), float(qz), float(qw))
cup_xy = E["cup_xy_m"]

# ---- collision fragments for every collider of an asset: margin / gap (Newton) + SDF for meshes ----
col_frags = [UsdPhysicsCollisionCfg(collision_enabled=True)]
if args.gap is not None or args.margin is not None:
    col_frags.append(NewtonCollisionCfg(contact_margin=args.margin, contact_gap=args.gap))
mesh_frags = [UsdPhysicsMeshCollisionCfg(mesh_approximation_name="sdf"), NewtonSDFCollisionCfg(sdf_max_resolution=args.sdf_res)]
# Hydroelastic (docs: hydroelastic_contact.html) needs BOTH shapes of a pair to be watertight volumes with SDF and
# `newton:hydroelasticEnabled`. Measured 2026-11-17 (sim links.npz): the fingers link6 / link7 and the cup shell are watertight;
# link0, link3, link4, link5 are NOT, so hydroelastic is switched on for the fingers + the cup only.
mesh_frags_hydro = [UsdPhysicsMeshCollisionCfg(mesh_approximation_name="sdf"),
                    NewtonSDFCollisionCfg(sdf_max_resolution=args.sdf_res, hydroelastic_enabled=True)]


# --- collision schemas must be on the prims BEFORE `InteractiveScene(...)` returns: its cloner builds the Newton model from
# the stage right there (interactive_scene.py -> replicate -> newton import_usd), so editing the stage afterwards changes
# nothing (tried: the arm had 0 shapes and the cup 121 convex pieces, approximations "authored" correctly but too late).
# `UsdFileCfg` has no mesh-collision slot (only `MeshFileCfg` has), so the spawner function is wrapped: spawn, then edit.
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402
from isaaclab.sim.schemas import apply_collision_properties, apply_mesh_collision_properties  # noqa: E402
from isaaclab.sim.spawners.from_files import spawn_from_usd  # noqa: E402

SDF_REPORT: list = []


def make_newton_ready(root_prim, label):
    """On every collision MESH under `root_prim`: CollisionAPI, margin/gap, and SDF (`physics:approximation = sdf` +
    NewtonSDFCollisionAPI). The arm's CollisionAPI sits on the Xform above the Mesh (`.../node_STL_BINARY_/mesh`); Newton's
    importer builds shapes from the geometry prim only, so the schemas go on the Meshes below it."""
    stage = root_prim.GetStage()
    seen, n = set(), 0
    for p in list(Usd.PrimRange(root_prim)):
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            continue
        for mp in ([p] if p.IsA(UsdGeom.Mesh) else [c for c in Usd.PrimRange(p) if c.IsA(UsdGeom.Mesh)]):
            if str(mp.GetPath()) in seen:
                continue
            seen.add(str(mp.GetPath()))
            UsdPhysics.CollisionAPI.Apply(mp)
            apply_collision_properties(str(mp.GetPath()), col_frags, stage)
            _hydro_ok = args.hydroelastic and (label == "cup" or "/link6/" in str(mp.GetPath()) or "/link7/" in str(mp.GetPath()))
            apply_mesh_collision_properties(str(mp.GetPath()), mesh_frags_hydro if _hydro_ok else mesh_frags, stage)
            n += 1
            if n <= 2:
                ap = UsdPhysics.MeshCollisionAPI(mp).GetApproximationAttr().Get()
                SDF_REPORT.append((label, str(mp.GetPath()).rsplit("/", 3)[-3:], f"approx={ap}", f"SDFAPI={mp.HasAPI('NewtonSDFCollisionAPI')}"))
    SDF_REPORT.append((label, f"{n} collision meshes made Newton-ready"))



def drop_mimic(root_prim):
    for pr in Usd.PrimRange(root_prim):
        if pr.GetName() == "gripper_joint_2" and pr.IsA(UsdPhysics.Joint):
            for sch in list(pr.GetAppliedSchemas()):
                if str(sch).startswith("PhysxMimicJointAPI"):
                    pr.RemoveAppliedSchema(sch)
            SDF_REPORT.append(("mimic", "PhysxMimicJointAPI removed from", str(pr.GetPath())))
            return


def native_newton_mimic(root_prim):
    """Author the mimic the Newton-native way (docs.isaacsim: tutorial_04 joint drive tuning): NewtonMimicAPI
    (newton:mimicJoint / mimicCoef0 / mimicCoef1) INSTEAD of the PhysX PhysxMimicJointAPI that Newton merely converts."""
    from pxr import Sdf
    for pr in Usd.PrimRange(root_prim):
        if pr.GetName() == "gripper_joint_2" and pr.IsA(UsdPhysics.Joint):
            ref = pr.GetRelationship("physxMimicJoint:rotZ:referenceJoint").GetTargets()[0]
            for sch in list(pr.GetAppliedSchemas()):
                if str(sch).startswith("PhysxMimicJointAPI"):
                    pr.RemoveAppliedSchema(sch)
            pr.AddAppliedSchema("NewtonMimicAPI")
            pr.CreateRelationship("newton:mimicJoint").SetTargets([ref])
            pr.CreateAttribute("newton:mimicCoef0", Sdf.ValueTypeNames.Float).Set(0.0)
            pr.CreateAttribute("newton:mimicCoef1", Sdf.ValueTypeNames.Float).Set(-1.0)
            SDF_REPORT.append(("mimic", "NewtonMimicAPI authored on", str(pr.GetPath()), "leader", str(ref)))
            return
    SDF_REPORT.append(("mimic", "gripper_joint_2 NOT FOUND"))


def tune_gripper_limit(root_prim, solref, solimp):
    from pxr import Sdf, Vt
    for pr in Usd.PrimRange(root_prim):
        if pr.GetName() == "gripper_joint_1" and pr.IsA(UsdPhysics.Joint):
            pr.AddAppliedSchema("MjcJointAPI")
            if solref is not None:
                pr.CreateAttribute("mjc:solreflimit", Sdf.ValueTypeNames.DoubleArray).Set(Vt.DoubleArray([float(v) for v in solref]))
            if solimp is not None:
                pr.CreateAttribute("mjc:solimplimit", Sdf.ValueTypeNames.DoubleArray).Set(Vt.DoubleArray([float(v) for v in solimp]))
            SDF_REPORT.append(("grip-limit", str(pr.GetPath()), "solreflimit", solref, "solimplimit", solimp))
            return


def make_mimic_stiff(root_prim, solref):
    """Replace the PhysX mimic spring on gripper_joint_2 by an explicit MuJoCo equality (MjcEqualityJointAPI + NewtonMimicAPI)
    with the given solref, same relation (follower = -1 x leader). Newton builds the MuJoCo equality from the authored
    `mjc:solref`; the plain PhysxMimicJointAPI path always gets the default (0.02, 1.0)."""
    from pxr import Sdf, Vt
    for pr in Usd.PrimRange(root_prim):
        if pr.GetName() == "gripper_joint_2" and pr.IsA(UsdPhysics.Joint):
            ref = pr.GetRelationship("physxMimicJoint:rotZ:referenceJoint").GetTargets()[0]
            for api in ("NewtonMimicAPI", "MjcEqualityJointAPI"):
                pr.AddAppliedSchema(api)
            pr.CreateRelationship("newton:mimicJoint").SetTargets([ref])
            pr.CreateAttribute("newton:mimicCoef0", Sdf.ValueTypeNames.Float).Set(0.0)
            pr.CreateAttribute("newton:mimicCoef1", Sdf.ValueTypeNames.Float).Set(-1.0)
            pr.CreateAttribute("mjc:solref", Sdf.ValueTypeNames.DoubleArray).Set(Vt.DoubleArray([float(solref[0]), float(solref[1])]))   # the applied schema types it double[]
            SDF_REPORT.append(("mimic", str(pr.GetPath()), "-> equality, solref", tuple(solref), "leader", str(ref)))
            return
    SDF_REPORT.append(("mimic", "gripper_joint_2 NOT FOUND"))


def spawn_newton_bin(prim_path, cfg, translation=None, orientation=None, **kw):
    """The bin the real task drops the cup into (EricC 2026-11-01), built by the same sim/shell_mesh.define_bin() as
    physics_replay.py, at the same measured pose; then SDF'd like the cup. Global path, not per env."""
    import scene_constants as SCN
    from shell_mesh import define_bin
    from isaaclab.sim.utils.stage import get_current_stage
    st = get_current_stage()
    info = define_bin(st, prim_path, (SCN.BIN_CENTER_X, SCN.BIN_CENTER_Y, SCN.BIN_BASE_Z), SCN.BIN_BASE_DIA, SCN.BIN_OPENING_DIA, SCN.BIN_HEIGHT)
    root = st.GetPrimAtPath(prim_path)
    make_newton_ready(root, "bin")
    SDF_REPORT.append(("bin", info))
    return root


def spawn_newton_robot(prim_path, cfg, translation=None, orientation=None, **kw):
    root = spawn_from_usd(prim_path, cfg, translation, orientation, **kw)
    make_newton_ready(root, "robot")
    if args.grip_limit_solref is not None or args.grip_limit_solimp is not None:
        tune_gripper_limit(root, args.grip_limit_solref, args.grip_limit_solimp)
    if args.mimic_mode == "pd":
        drop_mimic(root)
    elif args.mimic_mode == "newton":
        native_newton_mimic(root)
    elif args.mimic_solref is not None:
        make_mimic_stiff(root, args.mimic_solref)
    return root


def spawn_newton_cup(prim_path, cfg, translation=None, orientation=None, **kw):
    root = spawn_from_usd(prim_path, cfg, translation, orientation, **kw)
    make_newton_ready(root, "cup")
    return root


if args.contact_solref is not None or args.contact_solimp is not None or args.condim is not None:
    from isaaclab_newton.sim.schemas import MujocoCollisionCfg  # noqa: E402
    col_frags.append(MujocoCollisionCfg(solref=tuple(args.contact_solref) if args.contact_solref else None,
                                        solimp=tuple(args.contact_solimp) if args.contact_solimp else None, condim=args.condim))

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "gripper_joint_1"]


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=600.0))
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=args.asset, collision_props=col_frags, make_uninstanceable=True, func=spawn_newton_robot,
                                   articulation_props=(None if args.no_selfcol else NewtonArticulationCfg(self_collision_enabled=True))),
        init_state=ArticulationCfg.InitialStateCfg(pos=base_pos, rot=base_rot),
        # stiffness / damping / limits = None: take the asset's own values (omx_constants, patched by sim/patch_omx_usd.py),
        # exactly what physics_replay.py's default `motors=asset` does. gripper_joint_2 is the mimic follower: undriven.
        actuators={"arm": ImplicitActuatorCfg(joint_names_expr=ARM_JOINTS, stiffness=None, damping=None),
                   # mimic-mode pd: the follower gets the leader's gains (omx_constants: XL330-M288 gripper, 0.52 N.m, k 29.794 / d 1.49 per rad)
                   **({"follower": ImplicitActuatorCfg(joint_names_expr=["gripper_joint_2"], stiffness=29.794, damping=1.49,
                                                       actuator_effort_limit=0.52, joint_effort_limit=0.52)} if args.mimic_mode == "pd" else {})},
    )
    # the riser the arm stands on: same pose / size as physics_replay.py build_env (a guess for its footprint, measured height)
    riser = None if args.no_riser else AssetBaseCfg(
        prim_path="/World/Riser",
        spawn=sim_utils.CuboidCfg(size=(0.16, 0.17, 0.144), collision_props=col_frags,
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.32))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(float(C["world_from_link0"][0][3]) - 0.02, float(C["world_from_link0"][1][3]), 0.073 - 0.001)),
    )
    bin = None if args.no_bin else AssetBaseCfg(prim_path="/World/Bin", spawn=sim_utils.SpawnerCfg(func=spawn_newton_bin))
    cup: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cup",
        spawn=sim_utils.UsdFileCfg(usd_path=args.cup_asset, collision_props=col_frags, make_uninstanceable=True,
                                   rigid_props=UsdPhysicsRigidBodyCfg(), func=spawn_newton_cup),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(float(cup_xy[0]) + args.cup_dx_mm / 1000.0, float(cup_xy[1]), 0.0005), rot=(0.0, 0.0, 0.0, 1.0)),
    )


solver = MJWarpSolverCfg(solver="newton", integrator="implicitfast", njmax=args.njmax, nconmax=args.nconmax,
                         use_mujoco_contacts=(args.contacts == "mujoco"),
                         disable_sensors=(args.det_mode != "not_guaranteed"))   # deterministic modes refuse MuJoCo Warp's internal sensor computation
kw = {}
if args.contacts == "newton":
    from isaaclab_newton.physics import NewtonCollisionPipelineCfg  # noqa: E402
    from isaaclab_newton.physics.newton_collision_cfg import HydroelasticSDFCfg  # noqa: E402
    kw["collision_cfg"] = NewtonCollisionPipelineCfg(
        **({"sdf_hydroelastic_config": HydroelasticSDFCfg()} if args.hydroelastic else {}))
physics = NewtonCfg(solver_cfg=solver, num_substeps=1, debug_mode=True, deterministic_mode=args.det_mode, **kw)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device, physics=physics))
scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True))

for _r in SDF_REPORT:
    print('  SDFCHK', *_r, flush=True)
sim.reset()

if args.print_shapes:
    from isaaclab_newton.physics import NewtonManager  # noqa: E402
    _m = NewtonManager.get_model()
    _lab = list(_m.shape_label); _body = _m.shape_body.numpy().tolist(); _flags = _m.shape_flags.numpy().tolist()
    _bl = list(_m.body_label)
    import collections  # noqa: E402
    _types = _m.shape_type.numpy().tolist()
    _agg = collections.Counter()
    for _l, _b, _t in zip(_lab, _body, _types):
        _grp = "Cup" if "/Cup/" in _l else ("Robot" if "/Robot/" in _l else "Ground")
        _agg[(_grp, (_bl[_b].rsplit("/", 1)[-1] if _b >= 0 else "static"), int(_t))] += 1
    print("SHAPES total", len(_lab), "(shape_type: 0 plane 1 sphere 2 capsule 3 box 4 mesh 5 convex? -- see newton.GeoType)", flush=True)
    for (_g, _bn, _t), _n in sorted(_agg.items()):
        print(f"  SHAPEGRP {_g:6s} body={_bn:18s} type={_t} count={_n}", flush=True)
    for _nm in ("constraint_mimic_joint0", "constraint_mimic_joint1", "constraint_mimic_coef0", "constraint_mimic_coef1", "constraint_mimic_enabled"):
        try: print("  MIMIC", _nm, getattr(_m, _nm).numpy().tolist(), flush=True)
        except Exception as _e: print("  MIMIC", _nm, "n/a", _e, flush=True)
    try:
        print("  MIMIC joint_label for those ids:", {i: l.rsplit("/", 1)[-1] for i, l in enumerate(_m.joint_label)}, flush=True)
        _mj = NewtonManager._solver.mj_model
        print("  MJ neq", int(_mj.neq), "eq_type", _mj.eq_type.tolist(), "eq_solref", _mj.eq_solref.tolist(), "eq_solimp", [[round(v, 4) for v in r] for r in _mj.eq_solimp.tolist()],
              "eq_data0", [[round(v, 4) for v in r[:5]] for r in _mj.eq_data.tolist()], flush=True)
        print("  MJ opt timestep", float(_mj.opt.timestep), flush=True)
        _jn = [mjn for mjn in range(_mj.njnt)]
        import mujoco as _mjmod
        _names = [_mjmod.mj_id2name(_mj, _mjmod.mjtObj.mjOBJ_JOINT, i) for i in _jn]
        for _i in _jn:
            print("  MJ JOINT", _names[_i], "type", int(_mj.jnt_type[_i]), "limited", int(_mj.jnt_limited[_i]), "range", [round(float(v), 3) for v in _mj.jnt_range[_i]],
                  "solref_limit", [round(float(v), 4) for v in _mj.jnt_solref[_i]], "solimp_limit", [round(float(v), 3) for v in _mj.jnt_solimp[_i]],
                  "margin", round(float(_mj.jnt_margin[_i]), 4), "armature", round(float(_mj.dof_armature[_mj.jnt_dofadr[_i]]), 5), flush=True)
    except Exception as _e:
        print("  MIMIC MJ inspect failed:", type(_e).__name__, _e, flush=True)
    try:
        import newton as _nw
        print("  GeoType:", {k: int(v) for k, v in _nw.GeoType.__members__.items()}, flush=True)
    except Exception as _e:
        print("  GeoType n/a", _e, flush=True)

robot, cup = scene["robot"], scene["cup"]
if args.print_shapes:
    try:
        _bm = robot.data.body_mass.torch[0].cpu().tolist()
        _jl = robot.data.joint_pos_limits.torch[0].cpu().tolist()
        print("JOINT LIMITS (rad, as Isaac Lab reads them from the Newton model):", {n: [round(v, 3) for v in l] for n, l in zip(robot.joint_names, _jl)}, flush=True)
        print("BODYMASS (kg)", {n: round(m, 6) for n, m in zip(robot.body_names, _bm)}, flush=True)
    except Exception as _e:
        print("BODYMASS n/a", _e, flush=True)
names = list(robot.joint_names)
print("DOFS", names, "| bodies", list(robot.body_names), "| dt", DT, "SUB", SUB, "| contacts", args.contacts,
      "gap", args.gap, "margin", args.margin, "sdf_res", args.sdf_res, flush=True)

# PhysX replay widens joint1-5 to +-360 deg: the real arm leaves the narrowed site limits (docs: physics_replay.py build_env)
lim = robot.data.joint_pos_limits.torch.clone()
for n in ("joint1", "joint2", "joint3", "joint4", "joint5"):
    lim[:, names.index(n)] = torch.tensor([-2 * math.pi, 2 * math.pi], device=lim.device)
robot.write_joint_position_limit_to_sim_index(limits=lim, warn_limit_violation=False)


if args.grip_armature > 0.0:
    _ids = [names.index("gripper_joint_1"), names.index("gripper_joint_2")]
    _arm = robot.data.joint_armature.torch.clone()
    _arm[:, _ids] = args.grip_armature
    robot.write_joint_armature_to_sim_index(armature=_arm[:, _ids], joint_ids=_ids)
    print("armature written", {names[i]: float(robot.data.joint_armature.torch[0, i]) for i in _ids}, flush=True)


def full(qb, g):
    val = {"joint1": qb[0], "joint2": qb[1], "joint3": qb[2], "joint4": qb[3], "joint5": qb[4], "gripper_joint_1": g, "gripper_joint_2": -g}
    return torch.tensor([[val.get(n, 0.0) for n in names]], dtype=torch.float32, device=sim.device)


def targets(vec):
    robot.actuators.target_command.set_position_index(value=vec)


q0 = full(QB[0], GRIP[0])
robot.write_joint_position_to_sim_index(position=q0)
robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0))
targets(q0)


def step():
    scene.write_data_to_sim()
    sim.step()
    scene.update(DT)


for _ in range(60):   # let the cup settle on the ground, arm held at frame 0
    step()


def wxyz(q_xyzw):
    x, y, z, w = (float(v) for v in q_xyzw)
    return [w, x, y, z]


def pose_log(pos, q_xyzw):
    return [float(v) for v in pos] + wxyz(q_xyzw)


body_names = list(robot.body_names)
link_bodies = [f"/World/envs/env_0/Robot/{b}" for b in body_names]
log = []
DIAG = []
if args.diag:
    from isaaclab_newton.physics import NewtonManager as _NM
    _mjw_data = _NM._solver.mjw_data
    _mjw_model = _NM._solver.mjw_model
    print("MJW data attrs:", [n for n in dir(_mjw_data) if not n.startswith("_")], flush=True)
    print("MJW model attrs (geom/opt):", [n for n in dir(_mjw_model) if any(k in n for k in ("geom_", "opt", "nmaxcon", "ngeom", "nmaxpair"))][:40], flush=True)
    import mujoco as _mjm
    _mjmod = _NM._solver.mj_model
    _geom_names = [_mjm.mj_id2name(_mjmod, _mjm.mjtObj.mjOBJ_GEOM, i) or f"geom{i}" for i in range(_mjmod.ngeom)]
    _geom_body = [(_mjm.mj_id2name(_mjmod, _mjm.mjtObj.mjOBJ_BODY, int(_mjmod.geom_bodyid[i])) or "") for i in range(_mjmod.ngeom)]
    print("MJ geoms:", list(zip(range(len(_geom_names)), _geom_names, _geom_body))[:40], flush=True)
    print("MJ geom_friction (first rows):", _mjmod.geom_friction.tolist()[:14], flush=True)
    print("MJ geom_condim:", _mjmod.geom_condim.tolist()[:14], "geom_margin:", _mjmod.geom_margin.tolist()[:14], "geom_gap:", _mjmod.geom_gap.tolist()[:14], flush=True)
    print("MJ opt: cone", int(_mjmod.opt.cone), "solver", int(_mjmod.opt.solver), "iterations", int(_mjmod.opt.iterations), "ls_it", int(_mjmod.opt.ls_iterations), "impratio", float(_mjmod.opt.impratio), "tolerance", float(_mjmod.opt.tolerance), flush=True)


def contact_stats(frame):
    """What MuJoCo-Warp's solver actually saw at this step (env 0): number of contacts, capacity, penetration, force, and which
    body pairs are in contact. Read AFTER the step, so it is the last sub-step's contact set."""
    d = _mjw_data
    out = {"frame": frame}
    try:
        n = int(d.nacon.numpy()[0]); out["nacon"] = n
        out["naconmax"] = int(d.naconmax) if hasattr(d, "naconmax") else None
    except Exception as e:
        out["nacon_err"] = str(e)[:80]; n = 0
    try:
        out["nefc"] = int(d.nefc.numpy()[0]); out["njmax"] = int(d.njmax) if hasattr(d, "njmax") else None
    except Exception as e:
        out["nefc_err"] = str(e)[:80]
    if n > 0:
        c = d.contact
        geom = c.geom.numpy()[:n]; dist = c.dist.numpy()[:n]; adr = c.efc_address.numpy()[:n]; dim = c.dim.numpy()[:n]
        incl = c.includemargin.numpy()[:n]; fr = c.friction.numpy()[:n]; sref = c.solref.numpy()[:n]; simp = c.solimp.numpy()[:n]
        force = d.efc.force.numpy()
        force = force[0] if force.ndim == 2 else force
        pairs = {}
        for i in range(n):
            a, b = int(geom[i][0]), int(geom[i][1])
            ba, bb = _geom_body[a], _geom_body[b]
            key = "|".join(sorted((ba.rsplit("/", 1)[-1] or "static", bb.rsplit("/", 1)[-1] or "static")))
            ad = int(np.ravel(adr[i])[0])
            fz = float(force[ad]) if 0 <= ad < force.shape[0] else 0.0
            e = pairs.setdefault(key, {"n": 0, "min_dist": 1e9, "sum_normal_force": 0.0, "includemargin": set(), "mu": set(), "solref": set(), "solimp": set(), "active": 0})
            e["n"] += 1; e["min_dist"] = min(e["min_dist"], float(np.ravel(dist[i])[0])); e["sum_normal_force"] += fz
            e["includemargin"].add(round(float(np.ravel(incl[i])[0]), 4)); e["mu"].add(round(float(np.ravel(fr[i])[0]), 3))
            e["solref"].add(tuple(round(float(v), 4) for v in np.ravel(sref[i])[:2])); e["solimp"].add(tuple(round(float(v), 3) for v in np.ravel(simp[i])[:5]))
            e["active"] += int(float(np.ravel(dist[i])[0]) < float(np.ravel(incl[i])[0]))
        out["pairs"] = {k: {"n": v["n"], "active(dist<includemargin)": v["active"], "min_dist_mm": round(v["min_dist"] * 1000, 3), "sum_normal_force_N": round(v["sum_normal_force"], 4),
                            "includemargin": sorted(v["includemargin"]), "mu": sorted(v["mu"]), "solref": sorted(v["solref"]), "solimp": sorted(v["solimp"])} for k, v in pairs.items()}
        out["dim_set"] = sorted(set(int(x) for x in dim))
    return out


last = N - 1 if not args.max_frames else min(N - 1, args.max_frames)
t0 = time.time()
nonfinite = False
for k in range(last):
    qa, qb = QB[k], QB[k + 1]
    ga, gb = GRIP[k], GRIP[k + 1]
    for s in range(SUB):
        t = (s + 1) / SUB
        targets(full(qa * (1 - t) + qb * t, ga * (1 - t) + gb * t))
        step()
    if args.diag and args.diag_range[0] <= k <= args.diag_range[1]:
        DIAG.append(contact_stats(k + 1))
    cp, cq = cup.data.root_pos_w.torch[0].cpu().numpy(), cup.data.root_quat_w.torch[0].cpu().numpy()
    bp, bq = robot.data.body_pos_w.torch[0].cpu().numpy(), robot.data.body_quat_w.torch[0].cpu().numpy()
    jp = robot.data.joint_pos.torch[0].cpu().numpy()
    if not (np.isfinite(cp).all() and np.isfinite(bp).all()):
        nonfinite = True
        print(f"NON-FINITE state at frame {k + 1}", flush=True)
        break
    log.append({"frame": k + 1, "cup_xyz": [float(v) for v in cp], "q_sim": [float(v) for v in jp],
                "q_cmd": [float(v) for v in full(qb, gb)[0].cpu().numpy()],
                "cup_pose": pose_log(cp, cq), "link_poses": [pose_log(bp[i], bq[i]) for i in range(len(body_names))]})
wall = time.time() - t0

json.dump({"cand": C.get("name", args.cand), "dof_names": names, "log": log, "self_contacts_settle": None, "link_bodies": link_bodies,
           "robot_root": "/World/envs/env_0/Robot", "cup_root": "/World/envs/env_0/Cup", "gravity_direction": [0.0, 0.0, -1.0],
           "engine": "newton_mjwarp", "args": vars(args), "nonfinite": nonfinite, "wall_seconds": wall},
          open(os.path.join(args.out, "physics_log.json"), "w"))
if args.diag:
    json.dump(DIAG, open(os.path.join(args.out, "contact_diag.json"), "w"), indent=1)
z = [r["cup_xyz"][2] for r in log]
print(f"DONE frames {len(log)} cup z start {z[0]:.4f} max {max(z):.4f} end {z[-1]:.4f} xy end {log[-1]['cup_xyz'][:2]} "
      f"| {wall:.1f} s wall, {len(log) * SUB / wall:.0f} steps/s", flush=True)
os._exit(0)

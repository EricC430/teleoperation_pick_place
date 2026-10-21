"""Isaac Sim 6.1 standalone (no Isaac Lab): PHYSICS replay of one recorded episode with a free paper cup (no attach).

Body joints: PD drives track the recorded observation.state; gripper: PD target from the recorded ACTION (so it
squeezes like the real follower did). Physics 120 Hz, 8 substeps per 15 fps frame, targets interpolated.

Arm model, `cand=fit`: per-frame link angles precomputed from the link-marker fit pinned by the touch test
(calibration/2026-10-14_link_marker_fit.json: zeros, gravity sag, measured base tilt) and the 6-dof base from the same
fit -- `scripts/prep_closed_loop_replay.py` writes them. The old `current/provisional` candidates (offset tables +
link0 xy + tilt) still work with a joint_candidates.json.
Asset: assets/omx_f_generated/omx_f_real.usda (sim/patch_omx_usd.py). Self-collision ON (at HOME the real forearm rests
on the shoulder-lift servo). Cup collision SDF by default: with the asset's convex decomposition the 1 mm wall is fat
enough that a finger lands on the rim instead of sliding down inside it (10-14 closed loop, first gripper close, see
scripts/score_closed_loop_replay.py: 27/35 agree with convex hulls, 32/35 with SDF; 9.3 cm cup also 32/35).
Known gap (2026-10-15): in the second before the first close, lift lags its command by > 3 deg in 19/35 episodes
(9.3 cm cup) -- the fingers press on the cup (no lag with the cup moved away, and not fixed by opening the gripper
wider on ep08). The real paper cup deforms; this one is rigid.
This replay renders frame k from state k (no latency). The real recording pairs state k with the camera's latest frame
read right after it; the physical lag is unmeasured (weak estimate: wrist ~1 frame, front-left none found) and ~1% of
frames repeat while the arm moves (docs/worklog/2026-10-16_closed_loop.md section 3-5). An earlier "1-3 frames, ~19%
repeat" here had no source and was not reproduced.

Args key=value: ep=<prep dir> cand=<fit|current|provisional> cands=<json> intr=<front-left intrinsics>
ext=<front-left extrinsics> [render=1 stride=1] [wrist_intr=.. wrist_ext=..] [cup_coll=sdf|asset|cd_hi] [selfcol=1]
[vel_ff=1] [max_frames=0] [tag=..] [grip_open=<rad>: hold the gripper at this opening, ignoring the action]
[gripper_source=state|action (default action)] [cup_asset=<USD path>] [wrist_near_m=0.005]
[cup_dx_mm=500: move the cup out of the arm's way] [finger_coll=sdf] [contacts=1: log what touches the cup]

    cd <isaac-sim-standalone> && ./python.bat <repo>/sim/physics_replay.py ep=<prep dir>/cl_ep00 cand=fit cands=<prep dir>/fit_candidate.json ...
"""
import sys, json, math
from pathlib import Path

A = dict(a.split("=", 1) for a in sys.argv[1:])
REPO = Path(r"D:\teleoperation_pick_place")
EP = Path(A["ep"]); CAND = A.get("cand", "provisional"); STRIDE = int(A.get("stride", "1")); MAXF = int(A.get("max_frames", "0"))
CHECK = A.get("check", "0") == "1"
sys.path.insert(0, str(REPO / "sim")); sys.path.insert(0, str(REPO))
from isaacsim import SimulationApp
import numpy as np
import camera_distortion as cd

I = cd.load_model(Path(A["intr"])); W, H = I.width, I.height
us, vs = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
ideal = cd.undistort_pixels(np.stack([us.ravel(), vs.ravel()], 1), I)
cx, cy = I.K[0, 2], I.K[1, 2]
hw = int(math.ceil(max(np.abs(ideal[:, 0] - cx).max(), W / 2))) + 2; hh = int(math.ceil(max(np.abs(ideal[:, 1] - cy).max(), H / 2))) + 2
RW, RH = 2 * hw, 2 * hh
map_x = (ideal[:, 0] - cx + hw).reshape(H, W).astype(np.float32); map_y = (ideal[:, 1] - cy + hh).reshape(H, W).astype(np.float32)
app = SimulationApp({"headless": True, "width": RW, "height": RH})

import omni.usd, carb.settings, cv2
from pxr import Usd, UsdGeom, UsdLux, Gf, UsdPhysics, PhysxSchema
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.sensors.experimental.rtx import RtxCamera, CameraSensor
from scipy.spatial.transform import Rotation as Rr
import joint_mapping as jm

E = json.load(open(EP / "episode.json")); G = json.load(open(EP / "gripper.json"))
C = json.load(open(A["cands"]))[CAND]
QB = np.array(E["q_body_rad"]) if CAND == "fit" else None   # per-frame link angles from the fit (offsets + sag)
ST = np.array(E["state"]); N = len(ST)
GRIP_SOURCE = A.get("gripper_source", "action")
if GRIP_SOURCE not in ("state", "action"):
    raise ValueError("gripper_source must be state or action")
GRIP_TARGET = G["gripper_rad_from_state"] if GRIP_SOURCE == "state" else G["gripper_target_rad_from_action"]
OUT = EP / (f"sim_{CAND}" + (f"_{A['tag']}" if 'tag' in A else "")); OUT.mkdir(exist_ok=True)
BODY = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def body_rad(st, k=None):
    if QB is not None:
        return QB[k]
    return np.array([C["sign"][j] * C["scale"][j] * st[k] + C["offset"][j] for k, j in enumerate(BODY)])


ctx = omni.usd.get_context(); ctx.new_stage(); stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
gravity_direction = np.asarray(C.get("gravity_direction", [0., 0., -1.]), dtype=float)
if gravity_direction.shape != (3,) or not np.isfinite(gravity_direction).all() or np.linalg.norm(gravity_direction) <= 0:
    raise ValueError("Invalid candidate gravity_direction")
gravity_direction /= np.linalg.norm(gravity_direction)
scene.CreateGravityDirectionAttr(Gf.Vec3f(*gravity_direction.tolist())); scene.CreateGravityMagnitudeAttr(9.81)
PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim()).CreateTimeStepsPerSecondAttr(int(A.get("phys_hz", "120")))
UsdGeom.Xform.Define(stage, "/World")
tbl = UsdGeom.Cube.Define(stage, "/World/Table"); tbl.CreateSizeAttr(1.0)
tbl.AddTranslateOp().Set(Gf.Vec3d(0.3, 0, -0.02)); tbl.AddScaleOp().Set(Gf.Vec3f(1.2, 1.4, 0.04))
tbl.CreateDisplayColorAttr([(0.72, 0.55, 0.38)]); UsdPhysics.CollisionAPI.Apply(tbl.GetPrim())
L0 = C["link0_xy_m"] if "link0_xy_m" in C else (np.array(C["world_from_link0"])[:2, 3]).tolist()
ris = UsdGeom.Cube.Define(stage, "/World/Riser"); ris.CreateSizeAttr(1.0)
ris.AddTranslateOp().Set(Gf.Vec3d(L0[0] - 0.02, L0[1], 0.073 - 0.001)); ris.AddScaleOp().Set(Gf.Vec3f(0.16, 0.17, 0.144))
ris.CreateDisplayColorAttr([(0.3, 0.3, 0.32)])  # visual only: its size is a guess and the folded home pose clears the real riser by ~1 cm
UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(600.0)
mnt = UsdGeom.Xform.Define(stage, "/World/RobotMount")
if "world_from_link0" in C:   # 6-dof base from the link-marker fit (measured tilt, fitted position)
    mnt.AddTransformOp().Set(Gf.Matrix4d(np.array(C["world_from_link0"]).T.tolist()))
else:
    mnt.AddTranslateOp().Set(Gf.Vec3d(L0[0], L0[1], 0.146)); mnt.AddRotateYOp().Set(float(C["tilt_y_deg"]))
rp = stage.DefinePrim("/World/RobotMount/Robot", "Xform"); rp.GetReferences().AddReference(str(REPO / A.get("asset", "assets/omx_f_generated/omx_f_real.usda")))  # patched: sim/patch_omx_usd.py
# The factory limits in the USD (joint2/joint3 -120..90 deg) are tighter than what the real arm does: the recorded
# home pose maps to elbow ~95 (current constants) / ~102 deg (provisional) and lift ~ -121 deg. Widen for the replay.
# Replay drives the arm through angles the real arm reached: no reason for the narrowed factory/site limits
# (the USD carries joint1 -90..45 site azimuth, joint2/3 -120..90). [柏宇說 2026-10-14] use the URDF defaults (+-360).
WIDE = {f"joint{i}": (-360.0, 360.0) for i in range(1, 6)}
for p in Usd.PrimRange(rp):
    if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName() in WIDE:
        jj = UsdPhysics.RevoluteJoint(p); jj.GetLowerLimitAttr().Set(WIDE[p.GetName()][0]); jj.GetUpperLimitAttr().Set(WIDE[p.GetName()][1])
        print("widened", p.GetPath(), WIDE[p.GetName()], flush=True)
# Body joints must land ON the recorded angles (the images are paired with those readings), so their drives are made
# stiff; only the gripper keeps the real servo's force limit, which is what decides whether the cup is held.
for p in Usd.PrimRange(rp):
    if p.HasAPI(UsdPhysics.ArticulationRootAPI):
        att = p.GetAttribute("physxArticulation:enabledSelfCollisions")
        print("self collisions in asset:", att.Get() if att else None, flush=True)
        if A.get("selfcol", "1") == "0":   # 10-15: ON by default -- at HOME the real forearm rests on the lift servo
            PhysxSchema.PhysxArticulationAPI.Apply(p).CreateEnabledSelfCollisionsAttr(False)
            print("self collisions disabled", flush=True)
# finger_coll=sdf (2026-10-15): the asset's colliders are all convexDecomposition. The fingers are hooked (the cup wall
# sits in the inner curve during a grasp); a convex hull fills that curve. Uninstance the finger links and give their
# colliders an SDF, like the cup. finger_links=link6,link7 by default.
if A.get("finger_coll"):
    FL = A.get("finger_links", "link6,link7").split(",")
    under = lambda p: any(f"/{l}/" in str(p.GetPath()) + "/" for l in FL)
    for _ in range(4):
        inst = [p for p in Usd.PrimRange(rp) if p.IsInstance() and under(p)]
        for p in inst:
            p.SetInstanceable(False)
        if not inst:
            break
    for p in Usd.PrimRange(rp):
        if under(p) and p.HasAPI(UsdPhysics.MeshCollisionAPI):
            mc = UsdPhysics.MeshCollisionAPI(p)
            if A["finger_coll"] == "sdf":
                mc.GetApproximationAttr().Set("sdf"); PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(p).CreateSdfResolutionAttr(int(A.get("finger_sdf_res", "256")))
            else:
                mc.GetApproximationAttr().Set(A["finger_coll"])
            print("finger collider", p.GetPath(), p.GetTypeName(), "->", A["finger_coll"], flush=True)
# [已查證 2026-10-14] end_effector_link (and the fixed 'world' link) carry physics:mass = 0 in the asset; PhysX
# turns a zero-mass, collider-less rigid body into 1 kg, which hung 1 kg on the fingertip (elbow gravity torque at
# home 2.31 N.m against a 1.5 N.m motor). Give the marker link a negligible mass.
if A.get("fix_ee_mass", "1") == "1":
    for p in Usd.PrimRange(rp):
        if p.GetName() == "end_effector_link" and p.HasAPI(UsdPhysics.MassAPI):
            UsdPhysics.MassAPI(p).GetMassAttr().Set(1e-4)
            print("end_effector_link mass -> 1e-4 kg", flush=True)
MOTORS = A.get("motors", "asset")  # "asset" = the patched asset's own values (omx_constants)
if MOTORS == "real":
    # [已查證 2026-10-12, read off the bus] IDs 11-13 (pan, lift, elbow) are model 1060 = XL430-W250 and IDs 14-16
    # (wrist_flex, wrist_roll, gripper) are model 1200 = XL330-M288. omx_constants.py has lift/elbow as XL330-M288 and
    # the gripper as XL330-M077, so the asset's maxForce is 0.52 / 0.52 / 0.20 instead of 1.5 / 1.5 / 0.52.
    STALL = {"joint1": 1.5, "joint2": 1.5, "joint3": 1.5, "joint4": 0.52, "joint5": 0.52, "gripper_joint_1": 0.52}
    SM = float(A.get("stiff_mult", "50"))
    for p in Usd.PrimRange(rp):
        if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName() in STALL:
            d = UsdPhysics.DriveAPI.Get(p, "angular")
            if A.get("k_rule") == "stall_per_deg":
                # full stall torque at 1 deg of error (USD angular stiffness is per degree); damping 5% as in omx_constants
                k = STALL[p.GetName()] / float(A.get("k_deg", "1.0"))
                d.GetStiffnessAttr().Set(k); d.GetDampingAttr().Set(0.05 * k)
            else:
                m = SM if p.GetName() != "gripper_joint_1" else float(A.get("grip_mult", "5"))
                d.GetStiffnessAttr().Set(d.GetStiffnessAttr().Get() * m); d.GetDampingAttr().Set(d.GetDampingAttr().Get() * m)
            d.GetMaxForceAttr().Set(STALL[p.GetName()])
            print("  real motor", p.GetName(), "k", round(d.GetStiffnessAttr().Get(), 3), "d", round(d.GetDampingAttr().Get(), 4), "maxF", STALL[p.GetName()], flush=True)
NOGRAV = A.get("robot_nograv", "0") == "1"
if NOGRAV:
    # compliant arm: the asset's servo-like gains, gravity off on the robot links (a real servo holds its pose with only
    # a small sag the encoders still read), so contacts push the arm back instead of being driven through
    for p in Usd.PrimRange(rp):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateDisableGravityAttr(True)
    CG = float(A.get("comp_gain", "1"))
    for p in Usd.PrimRange(rp):
        if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName() in ("joint1", "joint2", "joint3", "joint4", "joint5"):
            d = UsdPhysics.DriveAPI.Get(p, "angular")
            d.GetStiffnessAttr().Set(d.GetStiffnessAttr().Get() * CG); d.GetDampingAttr().Set(d.GetDampingAttr().Get() * CG)
            print("  ", p.GetName(), "stiffness", d.GetStiffnessAttr().Get(), "damping", d.GetDampingAttr().Get(), "maxForce", d.GetMaxForceAttr().Get(), flush=True)
    print("robot gravity off, asset gains x", CG, flush=True)
elif MOTORS == "stiff":  # the old arbitrary 1e5/1e3/1e4 -- only on request
    for p in Usd.PrimRange(rp):
        if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName() in ("joint1", "joint2", "joint3", "joint4", "joint5"):
            d = UsdPhysics.DriveAPI.Get(p, "angular")
            d.GetStiffnessAttr().Set(1.0e5); d.GetDampingAttr().Set(1.0e3); d.GetMaxForceAttr().Set(1.0e4)
# gripper_joint_2 follows gripper_joint_1 through a PhysX mimic joint whose spring is soft and undamped in the asset
# (naturalFrequency 25, dampingRatio 0.005): on 2026-10-14 it sat at -0.34..-0.53 rad while joint 1 was +0.59.
MIMIC_NF, MIMIC_DR = float(A.get("mimic_nf", "300")), float(A.get("mimic_dr", "1.0"))
for p in Usd.PrimRange(rp):
    if p.GetName() == "gripper_joint_2":
        p.GetAttribute("physxMimicJoint:rotZ:naturalFrequency").Set(MIMIC_NF)
        p.GetAttribute("physxMimicJoint:rotZ:dampingRatio").Set(MIMIC_DR)
        print("mimic", MIMIC_NF, MIMIC_DR, flush=True)
DX, DY = float(A.get("cup_dx_mm", "0")) / 1000, float(A.get("cup_dy_mm", "0")) / 1000
cm = UsdGeom.Xform.Define(stage, "/World/CupMount"); cm.AddTranslateOp().Set(Gf.Vec3d(E["cup_xy_m"][0] + DX, E["cup_xy_m"][1] + DY, 0.0005))
cp = stage.DefinePrim("/World/CupMount/Cup", "Xform"); cp.GetReferences().AddReference(str(REPO / A.get("cup_asset", "assets/paper_cup/paper_cup.usd")))
GRIP_K, GRIP_F = A.get("grip_k"), A.get("grip_f")
for p in Usd.PrimRange(rp):
    if p.GetName() == "gripper_joint_1" and (GRIP_K or GRIP_F):
        d = UsdPhysics.DriveAPI.Get(p, "angular")
        if GRIP_K: d.GetStiffnessAttr().Set(float(GRIP_K)); d.GetDampingAttr().Set(float(GRIP_K) / 20)
        if GRIP_F: d.GetMaxForceAttr().Set(float(GRIP_F))
        print("gripper drive", d.GetStiffnessAttr().Get(), d.GetDampingAttr().Get(), d.GetMaxForceAttr().Get(), flush=True)
if "cup_mu" in A:
    for p in Usd.PrimRange(cp):
        if p.HasAPI(UsdPhysics.MaterialAPI):
            m = UsdPhysics.MaterialAPI(p); m.GetStaticFrictionAttr().Set(float(A["cup_mu"])); m.GetDynamicFrictionAttr().Set(0.75 * float(A["cup_mu"]))
            print("cup friction", A["cup_mu"], flush=True)
CUPC = A.get("cup_coll", "sdf")
if CUPC != "asset":
    for p in Usd.PrimRange(cp):
        if p.HasAPI(UsdPhysics.MeshCollisionAPI):
            mc = UsdPhysics.MeshCollisionAPI(p)
            if CUPC == "sdf":
                mc.GetApproximationAttr().Set("sdf"); PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(p).CreateSdfResolutionAttr(256)
            elif CUPC == "cd_hi":
                cd_ = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(p)
                cd_.CreateVoxelResolutionAttr(2000000); cd_.CreateMaxConvexHullsAttr(128); cd_.CreateHullVertexLimitAttr(64); cd_.CreateShrinkWrapAttr(True)
            print("cup collision", CUPC, flush=True)

cam = RtxCamera("/World/Cam", positions=[[0, 0, 1.0]], orientations=[[1, 0, 0, 0]])
cc = next(UsdGeom.Camera(ch) for ch in Usd.PrimRange(stage.GetPrimAtPath("/World/Cam")) if ch.IsA(UsdGeom.Camera))
AP = 20.955; cc.GetFocalLengthAttr().Set(I.K[0, 0] * AP / RW); cc.GetHorizontalApertureAttr().Set(AP)
cc.GetVerticalApertureAttr().Set(AP * RH / RW * I.K[0, 0] / I.K[1, 1]); cc.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 20.0))
sensor = CameraSensor(cam, resolution=(RH, RW), annotators=["rgb"])
carb.settings.get_settings().set("/rtx/post/dlss/execMode", 0)
Xe = json.load(open(A["ext"])); q = Xe["quat_wxyz"]
Rx = Rr.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); tx = np.array(Xe["pos_m"]) - [0, 0, Xe.get("table_top_z", 0.0)]
Ru = Rx @ np.diag([1.0, -1.0, -1.0]); x_, y_, z_, w_ = Rr.from_matrix(Ru).as_quat()
cam.set_world_poses(positions=[tx.tolist()], orientations=[[w_, x_, y_, z_]])
WRIST = "wrist_ext" in A
if WRIST:
    # wrist camera: link5_from_camera (OpenCV axes) from sim/solve_wrist_cam.py, its own lens via a padded render + remap
    Iw = cd.load_model(Path(A["wrist_intr"])); Ww, Hw = Iw.width, Iw.height
    uw, vw = np.meshgrid(np.arange(Ww, dtype=np.float64), np.arange(Hw, dtype=np.float64))
    idw = cd.undistort_pixels(np.stack([uw.ravel(), vw.ravel()], 1), Iw)
    cxw, cyw = Iw.K[0, 2], Iw.K[1, 2]
    hww = int(math.ceil(max(np.abs(idw[:, 0] - cxw).max(), Ww / 2))) + 2; hhw = int(math.ceil(max(np.abs(idw[:, 1] - cyw).max(), Hw / 2))) + 2
    RWw, RHw = 2 * hww, 2 * hhw
    mxw = (idw[:, 0] - cxw + hww).reshape(Hw, Ww).astype(np.float32); myw = (idw[:, 1] - cyw + hhw).reshape(Hw, Ww).astype(np.float32)
    camw = RtxCamera("/World/CamWrist", positions=[[0, 0, 1.0]], orientations=[[1, 0, 0, 0]])
    ccw = next(UsdGeom.Camera(ch) for ch in Usd.PrimRange(stage.GetPrimAtPath("/World/CamWrist")) if ch.IsA(UsdGeom.Camera))
    ccw.GetFocalLengthAttr().Set(Iw.K[0, 0] * AP / RWw); ccw.GetHorizontalApertureAttr().Set(AP)
    ccw.GetVerticalApertureAttr().Set(AP * RHw / RWw * Iw.K[0, 0] / Iw.K[1, 1]); ccw.GetClippingRangeAttr().Set(Gf.Vec2f(float(A.get("wrist_near_m", "0.005")), 20.0))
    sensorw = CameraSensor(camw, resolution=(RHw, RWw), annotators=["rgb"])
    Xw = json.load(open(A["wrist_ext"])); qw = Xw["quat_wxyz"]
    X5c = np.eye(4); X5c[:3, :3] = Rr.from_quat([qw[1], qw[2], qw[3], qw[0]]).as_matrix(); X5c[:3, 3] = Xw["pos_m"]
app.update()

ap = [str(p.GetPath()) for p in Usd.PrimRange(rp) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
cup_body = [str(p.GetPath()) for p in Usd.PrimRange(cp) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
CONTACTS = A.get("contacts", "0") == "1"
if CONTACTS:   # which collider touches the cup, where and how hard -- summed over the substeps of each frame
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysicsSchemaTools
    PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(cup_body[0])).CreateThresholdAttr().Set(0.0)
    CUR = {}

    def _on_contact(headers, data):
        for h in headers:
            c0, c1 = str(PhysicsSchemaTools.intToSdfPath(h.collider0)), str(PhysicsSchemaTools.intToSdfPath(h.collider1))
            other = c1 if c0.startswith(cup_body[0]) else c0
            if other.startswith("/World/Table"):
                continue
            e = CUR.setdefault(other, {"n": 0, "pos": np.zeros(3), "imp": np.zeros(3), "zmin": 9.0, "zmax": -9.0})
            for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                d = data[i]; pz = np.array([d.position[0], d.position[1], d.position[2]])
                e["n"] += 1; e["pos"] += pz; e["imp"] += np.array([d.impulse[0], d.impulse[1], d.impulse[2]])
                e["zmin"] = min(e["zmin"], pz[2]); e["zmax"] = max(e["zmax"], pz[2])
    _contact_sub = get_physx_simulation_interface().subscribe_contact_report_events(_on_contact)
app_utils.play(commit=True); app.update()
robot = Articulation(ap[0]); names = list(robot.dof_names); cup = RigidPrim(cup_body[0])
print("DOFS", names, "CUP", cup_body[0], "dt", SimulationManager.get_physics_dt(), flush=True)


def full(qb, g):
    val = {"joint1": qb[0], "joint2": qb[1], "joint3": qb[2], "joint4": qb[3], "joint5": qb[4], "gripper_joint_1": g, "gripper_joint_2": -g}
    return np.array([[val.get(n, 0.0) for n in names]], dtype=np.float32)


K0 = int(A.get("start", "0"))
q0 = full(body_rad(ST[K0], K0), G["gripper_rad_from_state"][K0])
robot.set_dof_positions(q0); robot.set_dof_position_targets(q0); robot.set_dof_velocities(np.zeros_like(q0))
app_utils.pause(); app.update()
if A.get("diag", "0") == "1":
    def _np(x):
        x = x[0] if isinstance(x, tuple) else x
        return np.round(np.asarray(x.numpy() if hasattr(x, "numpy") else x), 4)
    SimulationManager.step(steps=1)
    print("DIAG names", names, flush=True)
    for nm in ("get_dof_gravity_compensation_forces", "get_dof_max_efforts", "get_dof_max_velocities", "get_dof_armatures", "get_link_masses"):
        try:
            print("DIAG", nm, _np(getattr(robot, nm)()), flush=True)
        except Exception as e:
            print("DIAG", nm, "ERR", e, flush=True)
    for nm in ("get_dof_gains", "get_dof_friction_properties"):
        try:
            v = getattr(robot, nm)()
            print("DIAG", nm, [_np(x) for x in v], flush=True)
        except Exception as e:
            print("DIAG", nm, "ERR", e, flush=True)
SUB = int(round(1.0 / E["fps"] / SimulationManager.get_physics_dt()))
for _ in range(60):   # let the cup settle on the table, arm held at frame 0
    SimulationManager.step(steps=1, update_fabric=False)
log = []


def grab(k, qsim=None):
    if WRIST:   # put the wrist camera on the simulated link5 (FK of the sim's own joint angles) before rendering
        from reach_logger import fk as _fk
        M = np.array(C["world_from_link0"]) @ _fk.link5_transform([float(v) for v in qsim[:5]])
        Mc = M @ X5c
        Ruw = Mc[:3, :3] @ np.diag([1.0, -1.0, -1.0]); xw_, yw_, zw_, ww_ = Rr.from_matrix(Ruw).as_quat()
        camw.set_world_poses(positions=[Mc[:3, 3].tolist()], orientations=[[ww_, xw_, yw_, zw_]])
    app.update()
    d = sensor.get_data("rgb"); d = d[0] if isinstance(d, tuple) else d
    rgb = (d.numpy() if hasattr(d, "numpy") else np.asarray(d))[..., :3].astype(np.uint8)
    cv2.imwrite(str(OUT / f"sim_front-left_{k:04d}.jpg"), cv2.remap(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), map_x, map_y, cv2.INTER_LINEAR))
    if WRIST:
        dw = sensorw.get_data("rgb"); dw = dw[0] if isinstance(dw, tuple) else dw
        rw = (dw.numpy() if hasattr(dw, "numpy") else np.asarray(dw))[..., :3].astype(np.uint8)
        cv2.imwrite(str(OUT / f"sim_wrist_{k:04d}.jpg"), cv2.remap(cv2.cvtColor(rw, cv2.COLOR_RGB2BGR), mxw, myw, cv2.INTER_LINEAR))


last = N - 1 if not MAXF else min(N - 1, K0 + MAXF * STRIDE)
for k in range(K0, last):
    qa, qb = body_rad(ST[k], k), body_rad(ST[k + 1], k + 1)
    ga, gb = GRIP_TARGET[k], GRIP_TARGET[k + 1]
    if "grip_open" in A:
        ga = gb = float(A["grip_open"])
    for s in range(SUB):
        t = (s + 1) / SUB
        robot.set_dof_position_targets(full(qa * (1 - t) + qb * t, ga * (1 - t) + gb * t))
        if A.get("vel_ff", "0") == "1":   # feed-forward the recorded joint velocity so a soft drive does not lag
            v = (qb - qa) * E["fps"]
            robot.set_dof_velocity_targets(full(v, 0.0) * np.array([[1, 1, 1, 1, 1, 0, 0]], dtype=np.float32))
        SimulationManager.step(steps=1, update_fabric=False)
    pos, _ = cup.get_world_poses(); pos = np.asarray(pos.numpy() if hasattr(pos, "numpy") else pos)[0]
    jp = robot.get_dof_positions(); jp = np.asarray(jp.numpy() if hasattr(jp, "numpy") else jp)[0]
    log.append({"frame": k + 1, "cup_xyz": [float(v) for v in pos], "q_sim": [float(v) for v in jp], "q_cmd": [float(v) for v in full(qb, gb)[0]]})
    if CONTACTS:
        log[-1]["contacts"] = {c: {"n": e["n"], "pos_mean": (e["pos"] / max(e["n"], 1)).tolist(), "impulse": e["imp"].tolist(),
                                   "z_range": [e["zmin"], e["zmax"]]} for c, e in CUR.items()}
        CUR.clear()
    if A.get("render", "1") == "1" and (k + 1) % STRIDE == 0:
        grab(k + 1, [jp[names.index(n)] for n in ("joint1", "joint2", "joint3", "joint4", "joint5")])
    if A.get("linkdbg", "0") == "1" and k < 2:
        xc = UsdGeom.XformCache()
        for p in Usd.PrimRange(rp):
            if p.GetName() in ("link0", "link2", "link3", "link5"):
                print("LINKDBG", k + 1, p.GetName(), [round(v, 4) for v in xc.GetLocalToWorldTransform(p).ExtractTranslation()], flush=True)
        try:
            from isaacsim.core.experimental.prims import XformPrim
        except Exception:
            pass
    if CHECK and k > 4:
        break
json.dump({"cand": C["name"], "dof_names": names, "log": log,
           "input_pose_source": "recorded observation.state plus selected geometric calibration",
           "render_pose_source": "simulated articulation; physics tracking residuals are retained",
           "gripper_target_source": GRIP_SOURCE, "gravity_direction": gravity_direction.tolist(),
           "wrist_near_m": float(A.get("wrist_near_m", "0.005")), "wrist_ext": A.get("wrist_ext"), "cup_asset": A.get("cup_asset", "assets/paper_cup/paper_cup.usd")},
          open(OUT / "physics_log.json", "w"))
z = [r["cup_xyz"][2] for r in log]
print(f"DONE frames {len(log)} cup z start {z[0]:.4f} max {max(z):.4f} end {z[-1]:.4f} xy end {log[-1]['cup_xyz'][:2]}", flush=True)
app.close()

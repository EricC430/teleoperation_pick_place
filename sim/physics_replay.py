"""Isaac Sim 6.1 standalone (no Isaac Lab): PHYSICS replay of one recorded episode with a free paper cup (no attach).

Body joints: PD drives track the recorded observation.state; gripper: PD target from the recorded ACTION (so it
squeezes like the real follower did). Physics 120 Hz, 8 substeps per 15 fps frame, targets interpolated.

Arm model, `cand=fit`: per-frame link angles precomputed from the link-marker fit pinned by the touch test
(calibration/2026-10-14_link_marker_fit.json: zeros, gravity sag, measured base tilt) and the 6-dof base from the same
fit -- `scripts/prep_closed_loop_replay.py` writes them. The old `current/provisional` candidates (offset tables +
link0 xy + tilt) still work with a joint_candidates.json.
Asset: assets/omx_f_generated/omx_f_real.usda (sim/patch_omx_usd.py). Self-collision ON with SDF colliders on every link
(2026-10-31; selfcol=0 turns it off, finger_coll=asset keeps the convex hulls -- which jam the fingers). Cup collision SDF by default: with the asset's convex decomposition the 1 mm wall is fat
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

[wrist_mount=child|fk (default child)] [repo=<repo root>] [out=<output dir>] [render_api=auto|rtx6|replicator]
[pose_log=1: also log every robot link's and the cup's world pose (pos + quat wxyz) per frame, for the kinematic
 re-render of sim/dr_render.py] [config_report=<expected json or "none">: write OUT/config_report.json (physics scene,
 articulation, joint drives and limits, masses, colliders, friction, cameras, input file hashes) and check it against
 the expected file (sim/dr_expected_config.json = the configuration that gave 33/36 on 10/14); config_only=1 stops
 after the report]

Any machine, any Isaac Sim from 5.0 on (tested: 6.1 standalone on the Windows PC, 5.1 in the isaac-lab container):
- Paths: repo=, else $OMX_REPO, else the first parent of this file holding calibration/ and reach_logger/, else
  D:\\teleoperation_pick_place. A relative path argument is taken from the working directory if it exists there, else
  from the repo. Output: out=, else <ep>/sim_<cand>[_<tag>].
- Version-specific calls are wrapped once below (play/pause, physics step, render camera); the rest is pxr + the
  isaacsim.core.experimental prims both versions have.
- wrist_mount=child (2026-10-30): the wrist camera is a USD child of the robot's link5 prim, so it rides on whatever
  pose physics gives link5 (contacts included), like the real bracket. wrist_mount=fk: the earlier way, a free camera
  posed every rendered frame from FK of the simulated joint angles.

    cd <isaac-sim-standalone> && ./python.bat <repo>/sim/physics_replay.py ep=<prep dir>/cl_ep00 cand=fit cands=<prep dir>/fit_candidate.json ...
    sim/run_in_container.sh physics_replay.py ep=/workspace/test_isaaclab/<prep dir>/cl_ep00 cand=fit ...
"""
import sys, os, json, math, inspect
from pathlib import Path

A = dict(a.split("=", 1) for a in sys.argv[1:])
HERE = Path(__file__).resolve().parent


def _repo():
    for c in (A.get("repo"), os.environ.get("OMX_REPO")):
        if c:
            return Path(c)
    for p in HERE.parents:
        if (p / "calibration").is_dir() and (p / "reach_logger").is_dir():
            return p
    return Path(r"D:\teleoperation_pick_place")   # the Windows sim PC


REPO = _repo()


def P(x):   # input path: as given if absolute or found from the working directory, else under the repo
    p = Path(x)
    return p if p.is_absolute() or p.exists() else REPO / p


EP = P(A["ep"]); CAND = A.get("cand", "provisional"); STRIDE = int(A.get("stride", "1")); MAXF = int(A.get("max_frames", "0"))
CHECK = A.get("check", "0") == "1"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "sim")); sys.path.insert(0, str(REPO))
from isaacsim import SimulationApp
import numpy as np
import camera_distortion as cd

I = cd.load_model(P(A["intr"])); W, H = I.width, I.height
us, vs = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
ideal = cd.undistort_pixels(np.stack([us.ravel(), vs.ravel()], 1), I)
cx, cy = I.K[0, 2], I.K[1, 2]
hw = int(math.ceil(max(np.abs(ideal[:, 0] - cx).max(), W / 2))) + 2; hh = int(math.ceil(max(np.abs(ideal[:, 1] - cy).max(), H / 2))) + 2
RW, RH = 2 * hw, 2 * hh
map_x = (ideal[:, 0] - cx + hw).reshape(H, W).astype(np.float32); map_y = (ideal[:, 1] - cy + hh).reshape(H, W).astype(np.float32)
app = SimulationApp({"headless": True, "width": RW, "height": RH})

import omni.usd, carb.settings, cv2
from pxr import Usd, UsdGeom, UsdLux, Gf, UsdPhysics, PhysxSchema
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from scipy.spatial.transform import Rotation as Rr
import joint_mapping as jm

# ---- the version-specific calls, wrapped once ----
try:   # Isaac Sim 6.x
    import isaacsim.core.experimental.utils.app as app_utils

    def play():
        app_utils.play(commit=True)

    def pause():
        app_utils.pause()
except ImportError:   # 5.x
    import omni.timeline
    _tl = omni.timeline.get_timeline_interface()

    def play():
        _tl.play(); _tl.commit()

    def pause():
        _tl.pause(); _tl.commit()

_STEP_HAS_N = "steps" in inspect.signature(SimulationManager.step).parameters


def phys_step(n=1, fabric=False):
    if _STEP_HAS_N:   # 6.x: step(steps, update_fabric)
        SimulationManager.step(steps=n, update_fabric=fabric)
    else:             # 5.x: one simulate + fetch_results per call
        for _ in range(n):
            SimulationManager.step()


RENDER_API = A.get("render_api", "auto")
if RENDER_API in ("auto", "rtx6"):
    try:
        from isaacsim.sensors.experimental.rtx import RtxCamera, CameraSensor
        RENDER_API = "rtx6"
    except ImportError:
        if RENDER_API == "rtx6":
            raise
        RENDER_API = "replicator"
if RENDER_API == "replicator":   # Isaac Sim 4.5 / 5.x
    try:
        import omni.replicator.core as rep
    except ImportError:
        import omni.kit.app
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)
        import omni.replicator.core as rep
    _REP_STEP = dict(rt_subframes=int(A.get("subframes", "4")), delta_time=0.0)
    if "pause_timeline" in inspect.signature(rep.orchestrator.step).parameters:
        _REP_STEP["pause_timeline"] = True   # never let a capture advance the physics
print("render API:", RENDER_API, flush=True)


class Cam:
    """A pinhole render camera, focal length from K on an RW x RH canvas. pose: world_from_camera in OpenCV axes for a
    free camera (set_pose_cv later), or a fixed parent_from_camera (OpenCV axes) for one authored under `parent`."""

    def __init__(self, path, K, rw, rh, near, parent=None, local_cv=None):
        self.rw, self.rh = rw, rh
        if parent is not None:   # a child prim: a mount Xform carries the fixed local pose, the camera sits on it
            mnt_ = UsdGeom.Xform.Define(stage, f"{parent}/{path}Mount")
            mnt_.AddTransformOp().Set(Gf.Matrix4d(usd_cam(local_cv).T.tolist()))
            path = f"{parent}/{path}Mount/{path}"
        else:
            path = f"/World/{path}"
        if RENDER_API == "rtx6":
            kw = dict(translations=[[0, 0, 0]], orientations=[[1, 0, 0, 0]]) if parent is not None else dict(positions=[[0, 0, 1.0]], orientations=[[1, 0, 0, 0]])
            self.cam = RtxCamera(path, **kw)
            uc = next(UsdGeom.Camera(ch) for ch in Usd.PrimRange(stage.GetPrimAtPath(path)) if ch.IsA(UsdGeom.Camera))
        else:
            uc = UsdGeom.Camera.Define(stage, path); self.op = None if parent is not None else uc.AddTransformOp()
        uc.GetFocalLengthAttr().Set(K[0, 0] * AP / rw); uc.GetHorizontalApertureAttr().Set(AP)
        uc.GetVerticalApertureAttr().Set(AP * rh / rw * K[0, 0] / K[1, 1]); uc.GetClippingRangeAttr().Set(Gf.Vec2f(near, 20.0))
        if RENDER_API == "rtx6":
            self.sensor = CameraSensor(self.cam, resolution=(rh, rw), annotators=["rgb"])
        else:
            self.rp = rep.create.render_product(path, (rw, rh))
            self.ann = rep.AnnotatorRegistry.get_annotator("rgb"); self.ann.attach([self.rp])
        self.path = path

    def set_pose_cv(self, T):
        Tu = usd_cam(T)
        if RENDER_API == "rtx6":
            x, y, z, w = Rr.from_matrix(Tu[:3, :3]).as_quat()
            self.cam.set_world_poses(positions=[Tu[:3, 3].tolist()], orientations=[[w, x, y, z]])
        else:
            self.op.Set(Gf.Matrix4d(Tu.T.tolist()))

    def rgb(self):   # call render() first
        d = self.sensor.get_data("rgb") if RENDER_API == "rtx6" else self.ann.get_data()
        d = d[0] if isinstance(d, tuple) else d
        return (d.numpy() if hasattr(d, "numpy") else np.asarray(d))[..., :3].astype(np.uint8)


def render():
    if RENDER_API == "rtx6":
        app.update()
    else:
        rep.orchestrator.step(**_REP_STEP)


def usd_cam(T_cv):   # OpenCV camera axes (x right, y down, z forward) -> USD camera (looks down -Z, +Y up)
    T = np.array(T_cv, float); T[:3, :3] = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    return T
# ---- end of the version-specific calls ----

E = json.load(open(EP / "episode.json")); G = json.load(open(EP / "gripper.json"))
C = json.load(open(P(A["cands"])))[CAND]
QB = np.array(E["q_body_rad"]) if CAND == "fit" else None   # per-frame link angles from the fit (offsets + sag)
ST = np.array(E["state"]); N = len(ST)
GRIP_SOURCE = A.get("gripper_source", "action")
if GRIP_SOURCE not in ("state", "action"):
    raise ValueError("gripper_source must be state or action")
GRIP_TARGET = G["gripper_rad_from_state"] if GRIP_SOURCE == "state" else G["gripper_target_rad_from_action"]
OUT = P(A["out"]) if "out" in A else EP / (f"sim_{CAND}" + (f"_{A['tag']}" if 'tag' in A else "")); OUT.mkdir(parents=True, exist_ok=True)
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
L0 = C["link0_xy_m"] if "link0_xy_m" in C else (np.array(C["world_from_link0"])[:2, 3]).tolist()
UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(600.0)


BIN_INFO = None


def build_env(root, Ei, origin=None):
    """Table, riser, robot with every override below, and the cup, under `root`. Single replay: root=/World (the paths
    used since 2026-10-14). Multi-scene (eps=, 2026-10-31): root=/World/Env<i>, shifted by `origin` (x only), one
    replay per env, all in one physics scene -- same build, so the same configuration."""
    global FL, under
    if origin is not None:
        ex = UsdGeom.Xform.Define(stage, root); ex.AddTranslateOp().Set(Gf.Vec3d(*map(float, origin)))
    tbl = UsdGeom.Cube.Define(stage, f"{root}/Table"); tbl.CreateSizeAttr(1.0)
    # [EricC 2026-11-05] the table is the ground: same plane (top at z=0, the table-normal frame), no edge. A 1.2 x 1.4 m
    # slab let released objects fall off while the DR render's ground did not. 200 m wide, so no replay reaches the edge.
    tbl.AddTranslateOp().Set(Gf.Vec3d(0.3, 0, -0.02)); tbl.AddScaleOp().Set(Gf.Vec3f(200.0, 200.0, 0.04))
    tbl.CreateDisplayColorAttr([(0.72, 0.55, 0.38)]); UsdPhysics.CollisionAPI.Apply(tbl.GetPrim())
    ris = UsdGeom.Cube.Define(stage, f"{root}/Riser"); ris.CreateSizeAttr(1.0)
    ris.AddTranslateOp().Set(Gf.Vec3d(L0[0] - 0.02, L0[1], 0.073 - 0.001)); ris.AddScaleOp().Set(Gf.Vec3f(0.16, 0.17, 0.144))
    ris.CreateDisplayColorAttr([(0.3, 0.3, 0.32)])  # visual only: its size is a guess and the folded home pose clears the real riser by ~1 cm
    mnt = UsdGeom.Xform.Define(stage, f"{root}/RobotMount")
    if "world_from_link0" in C:   # 6-dof base from the link-marker fit (measured tilt, fitted position)
        mnt.AddTransformOp().Set(Gf.Matrix4d(np.array(C["world_from_link0"]).T.tolist()))
    else:
        mnt.AddTranslateOp().Set(Gf.Vec3d(L0[0], L0[1], 0.146)); mnt.AddRotateYOp().Set(float(C["tilt_y_deg"]))
    rp = stage.DefinePrim(f"{root}/RobotMount/Robot", "Xform"); rp.GetReferences().AddReference(str(P(A.get("asset", "assets/omx_f_generated/omx_f_real.usda"))))  # patched: sim/patch_omx_usd.py
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
            # Set explicitly since 2026-10-31 (before, only selfcol=0 wrote the flag, so the result depended on how the asset
            # was converted). Default ON (EricC 2026-10-31): at HOME the real forearm rests on the lift servo and the elbow
            # reading runs 2-5 deg past where the link stops (EricC430, 10-15; solve_link_markers.py --max-elbow-reading);
            # only self-collision stops the sim forearm there (with the readings as they are it sits ~4 mm inside the
            # shoulder even with exact SDF meshes). With self-collision on, every link collider is an SDF by default (see
            # below): the asset's convex hulls of the hooked fingers overlap each other at the recorded opening and jam the
            # gripper open (0/30 grasps). 10-14 closed loop, cup_ref, 36 episodes: on + all-SDF lifts 28/30 real successes
            # (off 26/30, on + finger-only SDF 26/30), 1/6 real misses in each. docs/worklog/2026-10-31_sim_dr_pipeline.md
            sc_on = A.get("selfcol", "1") == "1"
            PhysxSchema.PhysxArticulationAPI.Apply(p).CreateEnabledSelfCollisionsAttr(sc_on)
            print("self collisions", "enabled" if sc_on else "disabled", flush=True)
    # finger_coll=sdf (2026-10-15): the asset's colliders are all convexDecomposition. The fingers are hooked (the cup wall
    # sits in the inner curve during a grasp); a convex hull fills that curve. Uninstance the finger links and give their
    # colliders an SDF, like the cup. finger_links=link6,link7 by default.
    SELFCOL_ON = A.get("selfcol", "1") == "1"
    if SELFCOL_ON and "finger_coll" not in A:   # self-collision needs exact colliders (comment above); finger_coll=asset keeps the hulls
        A["finger_coll"] = "sdf"; A.setdefault("finger_links", "link0,link1,link2,link3,link4,link5,link6,link7")
    if A.get("finger_coll") and A["finger_coll"] != "asset":
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
    # [已查證 2026-10-30] the URDF hangs a 1 cm box on end_effector_link (a red marker; grey after conversion) between the
    # fingertips. Nothing like it is on the real arm, and it rides on link5, so it sat in a fixed spot of every wrist image.
    # Hidden unless ee_marker=1 (visual only: that link has no collider).
    if A.get("ee_marker", "0") != "1":
        for p in Usd.PrimRange(rp):
            if p.GetName() == "visuals" and p.GetParent().GetName() == "end_effector_link":
                UsdGeom.Imageable(p).MakeInvisible()
                print("end_effector_link marker hidden", p.GetPath(), flush=True)
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
    cm = UsdGeom.Xform.Define(stage, f"{root}/CupMount"); cm.AddTranslateOp().Set(Gf.Vec3d(Ei["cup_xy_m"][0] + DX, Ei["cup_xy_m"][1] + DY, 0.0005))
    cp = stage.DefinePrim(f"{root}/CupMount/Cup", "Xform"); cp.GetReferences().AddReference(str(P(A.get("cup_asset", "assets/paper_cup/paper_cup.usd"))))
    # The bin the real task drops the object into (EricC 2026-11-01: it is on the real cell, so it is in the replay).
    # Measured size and sideways offset, scene_constants.py; bin=0 leaves it out (the scene before 2026-11-01).
    global BIN_INFO
    if A.get("bin", "1") == "1":
        import scene_constants as SCN
        from shell_mesh import define_bin
        BIN_INFO = define_bin(stage, f"{root}/Bin", (SCN.BIN_CENTER_X, SCN.BIN_CENTER_Y, float(A.get("bin_z", SCN.BIN_BASE_Z))),
                              SCN.BIN_BASE_DIA, SCN.BIN_OPENING_DIA, SCN.BIN_HEIGHT)
        print("bin", BIN_INFO, flush=True)
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
                    mc.GetApproximationAttr().Set("sdf"); sdf_ = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(p)
                    sdf_.CreateSdfResolutionAttr(int(A.get("cup_sdf_res", "256")))
                    if "cup_sdf_margin_mm" in A: sdf_.CreateSdfMarginAttr(float(A["cup_sdf_margin_mm"]) / 1000)
                    if "cup_sdf_band_mm" in A: sdf_.CreateSdfNarrowBandThicknessAttr(float(A["cup_sdf_band_mm"]) / 1000)
                    if "cup_sdf_subgrid" in A: sdf_.CreateSdfSubgridResolutionAttr(int(A["cup_sdf_subgrid"]))
                elif CUPC == "cd_hi":
                    cd_ = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(p)
                    cd_.CreateVoxelResolutionAttr(2000000); cd_.CreateMaxConvexHullsAttr(128); cd_.CreateHullVertexLimitAttr(64); cd_.CreateShrinkWrapAttr(True)
                print("cup collision", CUPC, flush=True)
    # Penetration tests (2026-11-01, EricC: the fingers pass through the cup wall; sim/penetration.py measures it):
    # contact/rest offsets on the cup and gripper colliders, solver iterations on the cup and the articulation.
    if any(k in A for k in ("contact_offset_mm", "rest_offset_mm")):
        under_grip = lambda p: any(f"/{l}/" in str(p.GetPath()) for l in ("link5", "link6", "link7"))   # noqa: E731
        for root_ in (cp, rp):
            for p in Usd.PrimRange(root_):
                if p.HasAPI(UsdPhysics.CollisionAPI) and (root_ is cp or under_grip(p)):
                    pc_ = PhysxSchema.PhysxCollisionAPI.Apply(p)
                    if "contact_offset_mm" in A: pc_.CreateContactOffsetAttr(float(A["contact_offset_mm"]) / 1000)
                    if "rest_offset_mm" in A: pc_.CreateRestOffsetAttr(float(A["rest_offset_mm"]) / 1000)
        print("contact/rest offset", A.get("contact_offset_mm"), A.get("rest_offset_mm"), flush=True)
    if "cup_iters" in A:   # position,velocity
        pi_, vi_ = map(int, A["cup_iters"].split(","))
        for p in Usd.PrimRange(cp):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                rb_ = PhysxSchema.PhysxRigidBodyAPI.Apply(p); rb_.CreateSolverPositionIterationCountAttr(pi_); rb_.CreateSolverVelocityIterationCountAttr(vi_)
        print("cup solver iterations", pi_, vi_, flush=True)
    if "art_iters" in A:
        pi_, vi_ = map(int, A["art_iters"].split(","))
        for p in Usd.PrimRange(rp):
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                ar_ = PhysxSchema.PhysxArticulationAPI.Apply(p); ar_.CreateSolverPositionIterationCountAttr(pi_); ar_.CreateSolverVelocityIterationCountAttr(vi_)
        print("articulation solver iterations", pi_, vi_, flush=True)
    # cup_torsion_mm (2026-11-01, test): PhysX torsional patch friction on the cup's contacts. With point contacts on a
    # rigid 1 mm rim the sim cup turns in the fingers to ~85 deg while the real paper cup stays about upright (10-14 ep03
    # front-left frame 126), then hangs on the inside finger after release. Off unless given.
    if "cup_torsion_mm" in A:
        for p in Usd.PrimRange(cp):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                pc_ = PhysxSchema.PhysxCollisionAPI.Apply(p); r_ = float(A["cup_torsion_mm"]) / 1000
                pc_.CreateTorsionalPatchRadiusAttr(r_); pc_.CreateMinTorsionalPatchRadiusAttr(r_)
                print("cup torsional patch radius", r_, p.GetPath(), flush=True)
    return tbl, rp, cp


BATCH = "eps" in A
if not BATCH:
    tbl, rp, cp = build_env("/World", E)
else:
    # ---- multi-scene validation: eps=<ep dir>,<ep dir>,... outs=<out dir>,... (render=0, pose_log as given) ----
    # Every replay gets its own env (table, riser, robot, cup) env_spacing_m apart along x, all stepped together in the
    # one physics scene, each driven by its own recording; an env that has run out of frames holds its last target.
    # Logged positions are relative to the env origin, so physics_log.json reads exactly like a single replay's.
    if A.get("render", "1") != "0" or "config_report" in A or "wrist_ext" in A:
        raise SystemExit("eps= (multi-scene) is for validation: render=0, no config_report, no cameras")
    EPS_ = [P(x) for x in A["eps"].split(",")]; OUTS_ = [P(x) for x in A["outs"].split(",")]
    assert len(EPS_) == len(OUTS_), "eps and outs need the same length"
    SPACING = float(A.get("env_spacing_m", "4.0"))
    ENVS = []
    for i, (ed, od) in enumerate(zip(EPS_, OUTS_)):
        Ei = json.load(open(ed / "episode.json")); Gi = json.load(open(ed / "gripper.json"))
        org = np.array([i * SPACING, 0.0, 0.0])
        t_i, rp_i, cp_i = build_env(f"/World/Env{i}", Ei, org)
        od.mkdir(parents=True, exist_ok=True)
        ENVS.append({"E": Ei, "QB": np.array(Ei["q_body_rad"]), "N": len(Ei["state"]), "origin": org, "rp": rp_i, "cp": cp_i, "out": od,
                     "grip": Gi["gripper_rad_from_state"] if GRIP_SOURCE == "state" else Gi["gripper_target_rad_from_action"],
                     "grip0": Gi["gripper_rad_from_state"], "log": []})
    assert CAND == "fit", "multi-scene needs per-frame q_body_rad (cand=fit)"
    assert len({e["E"]["fps"] for e in ENVS}) == 1, "all replays need the same fps"
    app.update(); play(); app.update()
    POSE_LOG = A.get("pose_log", "0") == "1"
    for e in ENVS:
        e["robot"] = Articulation(next(str(p.GetPath()) for p in Usd.PrimRange(e["rp"]) if p.HasAPI(UsdPhysics.ArticulationRootAPI)))
        e["cup"] = RigidPrim(next(str(p.GetPath()) for p in Usd.PrimRange(e["cp"]) if p.HasAPI(UsdPhysics.RigidBodyAPI)))
        e["bodies"] = [str(p.GetPath()) for p in Usd.PrimRange(e["rp"]) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        e["links"] = [RigidPrim(pth) for pth in e["bodies"]] if POSE_LOG else []
    names = list(ENVS[0]["robot"].dof_names)

    def fullb(qb, g):
        val = {"joint1": qb[0], "joint2": qb[1], "joint3": qb[2], "joint4": qb[3], "joint5": qb[4], "gripper_joint_1": g, "gripper_joint_2": -g}
        return np.array([[val.get(n, 0.0) for n in names]], dtype=np.float32)

    def npa(x):
        return np.asarray(x.numpy() if hasattr(x, "numpy") else x)

    def rel_pose(rpr, org):   # world pose relative to the env origin: [x y z qw qx qy qz]
        pos, quat = rpr.get_world_poses()
        return [float(v) for v in npa(pos)[0] - org] + [float(v) for v in npa(quat)[0]]

    for e in ENVS:
        q0 = fullb(e["QB"][0], e["grip0"][0])
        e["robot"].set_dof_positions(q0); e["robot"].set_dof_position_targets(q0); e["robot"].set_dof_velocities(np.zeros_like(q0))
    pause(); app.update()
    SUB = int(round(1.0 / ENVS[0]["E"]["fps"] / SimulationManager.get_physics_dt()))
    for _ in range(60):   # settle, every arm held at its frame 0
        phys_step(1)
    NMAX = max(e["N"] for e in ENVS)
    for k in range(0, NMAX - 1):
        live = [e for e in ENVS if k < e["N"] - 1]
        for s_ in range(SUB):
            t = (s_ + 1) / SUB
            for e in live:
                qa, qb = e["QB"][k], e["QB"][k + 1]; ga, gb = e["grip"][k], e["grip"][k + 1]
                e["robot"].set_dof_position_targets(fullb(qa * (1 - t) + qb * t, ga * (1 - t) + gb * t))
            phys_step(1)
        for e in live:
            org = e["origin"]
            pos = npa(e["cup"].get_world_poses()[0])[0] - org
            jp = npa(e["robot"].get_dof_positions())[0]
            r = {"frame": k + 1, "cup_xyz": [float(v) for v in pos], "q_sim": [float(v) for v in jp],
                 "q_cmd": [float(v) for v in fullb(e["QB"][k + 1], e["grip"][k + 1])[0]]}
            if POSE_LOG:
                r["cup_pose"] = rel_pose(e["cup"], org); r["link_poses"] = [rel_pose(l_, org) for l_ in e["links"]]
            e["log"].append(r)
    for i, e in enumerate(ENVS):
        # the same link paths as a single replay, so sim/dr_render.py finds the links by name
        lb = [b.replace(f"/World/Env{i}/", "/World/") for b in e["bodies"]] if POSE_LOG else None
        json.dump({"cand": C["name"], "dof_names": names, "log": e["log"], "self_contacts_settle": None, "link_bodies": lb,
                   "robot_root": "/World/RobotMount/Robot", "cup_root": "/World/CupMount/Cup", "multi_scene": {"env": i, "n_envs": len(ENVS), "spacing_m": SPACING},
                   "input_pose_source": "recorded observation.state plus selected geometric calibration",
                   "render_pose_source": "simulated articulation; physics tracking residuals are retained",
                   "gripper_target_source": GRIP_SOURCE, "gravity_direction": gravity_direction.tolist(),
                   "wrist_near_m": None, "wrist_ext": None, "wrist_mount": None, "render_api": RENDER_API,
                   "args": dict(A, ep=str(EPS_[i]), out=str(e["out"])), "cup_asset": A.get("cup_asset", "assets/paper_cup/paper_cup.usd"), "bin": BIN_INFO},
                  open(e["out"] / "physics_log.json", "w"))
        z = [r["cup_xyz"][2] for r in e["log"]]
        print(f"DONE env {i} {EPS_[i].name} frames {len(e['log'])} cup z start {z[0]:.4f} max {max(z):.4f} end {z[-1]:.4f} xy end {e['log'][-1]['cup_xyz'][:2]}", flush=True)
    app.close(); sys.exit(0)

AP = 20.955
cam = Cam("Cam", I.K, RW, RH, 0.01)
carb.settings.get_settings().set("/rtx/post/dlss/execMode", 0)
Xe = json.load(open(P(A["ext"]))); q = Xe["quat_wxyz"]
Tfl = np.eye(4); Tfl[:3, :3] = Rr.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); Tfl[:3, 3] = np.array(Xe["pos_m"]) - [0, 0, Xe.get("table_top_z", 0.0)]
cam.set_pose_cv(Tfl)
WRIST = "wrist_ext" in A
WRIST_MOUNT = A.get("wrist_mount", "child")
if WRIST:
    # wrist camera: link5_from_camera (OpenCV axes), its own lens via a padded render + remap. No collider of its own
    # (the camera body is part of the link5 collision mesh already). Since 2026-10-30
    # calibration/2026-10-30_camera_extrinsics_wrist_cad_lens.json (CAD lens front; was the 10-15 marker solve)
    Iw = cd.load_model(P(A["wrist_intr"])); Ww, Hw = Iw.width, Iw.height
    uw, vw = np.meshgrid(np.arange(Ww, dtype=np.float64), np.arange(Hw, dtype=np.float64))
    idw = cd.undistort_pixels(np.stack([uw.ravel(), vw.ravel()], 1), Iw)
    cxw, cyw = Iw.K[0, 2], Iw.K[1, 2]
    hww = int(math.ceil(max(np.abs(idw[:, 0] - cxw).max(), Ww / 2))) + 2; hhw = int(math.ceil(max(np.abs(idw[:, 1] - cyw).max(), Hw / 2))) + 2
    RWw, RHw = 2 * hww, 2 * hhw
    mxw = (idw[:, 0] - cxw + hww).reshape(Hw, Ww).astype(np.float32); myw = (idw[:, 1] - cyw + hhw).reshape(Hw, Ww).astype(np.float32)
    Xw = json.load(open(P(A["wrist_ext"]))); qw = Xw["quat_wxyz"]
    X5c = np.eye(4); X5c[:3, :3] = Rr.from_quat([qw[1], qw[2], qw[3], qw[0]]).as_matrix(); X5c[:3, 3] = Xw["pos_m"]
    LINK5 = next(str(p.GetPath()) for p in Usd.PrimRange(rp) if p.GetName() == "link5" and p.HasAPI(UsdPhysics.RigidBodyAPI))
    if WRIST_MOUNT == "child":   # rides on the link5 rigid body: same place on the arm whatever physics does to it
        camw = Cam("CamWrist", Iw.K, RWw, RHw, float(A.get("wrist_near_m", "0.005")), parent=LINK5, local_cv=X5c)
    else:                        # "fk": free camera, posed in grab() from FK of the simulated joint angles
        camw = Cam("CamWrist", Iw.K, RWw, RHw, float(A.get("wrist_near_m", "0.005")))
    print("wrist camera", WRIST_MOUNT, camw.path, "from", A["wrist_ext"], flush=True)
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
SELF_CONTACTS = A.get("self_contacts", "0") == "1"
if SELF_CONTACTS:   # 2026-10-31 diagnosis: which robot links touch each other (self-collision on), per frame
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysicsSchemaTools
    _rb = [p for p in Usd.PrimRange(rp) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    for p in _rb:
        PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr().Set(0.0)
    SC = {}
    _rbp = [str(p.GetPath()) for p in _rb]

    def _body(path):
        return next((b.rsplit("/", 1)[-1] for b in sorted(_rbp, key=len, reverse=True) if path.startswith(b + "/") or path == b), None)

    def _on_self(headers, data):
        for h in headers:
            a0 = _body(str(PhysicsSchemaTools.intToSdfPath(h.actor0))); a1 = _body(str(PhysicsSchemaTools.intToSdfPath(h.actor1)))
            if not a0 or not a1:
                continue
            key = "|".join(sorted((a0, a1)))
            e = SC.setdefault(key, {"n": 0, "imp": 0.0, "sep_min": 9.0, "pos": None})
            for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                d = data[i]; e["n"] += 1; e["imp"] += float(np.linalg.norm([d.impulse[0], d.impulse[1], d.impulse[2]]))
                e["sep_min"] = min(e["sep_min"], float(d.separation)); e["pos"] = [round(float(v), 4) for v in d.position]
    _self_sub = get_physx_simulation_interface().subscribe_contact_report_events(_on_self)
play(); app.update()
robot = Articulation(ap[0]); names = list(robot.dof_names); cup = RigidPrim(cup_body[0])
print("DOFS", names, "CUP", cup_body[0], "dt", SimulationManager.get_physics_dt(), flush=True)
POSE_LOG = A.get("pose_log", "0") == "1"
LINK_BODIES = [str(p.GetPath()) for p in Usd.PrimRange(rp) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
_LINK_RP = [RigidPrim(pth) for pth in LINK_BODIES] if POSE_LOG else []


def _pose(rpr):   # world pose of one rigid body: [x, y, z, qw, qx, qy, qz]
    pos, quat = rpr.get_world_poses()
    pos = np.asarray(pos.numpy() if hasattr(pos, "numpy") else pos)[0]; quat = np.asarray(quat.numpy() if hasattr(quat, "numpy") else quat)[0]
    return [float(v) for v in pos] + [float(v) for v in quat]


if "config_report" in A:
    import config_report as CR   # sim/config_report.py
    REPORT = CR.build(stage=stage, robot_prim=rp, cup_prim=cp, table_prim=tbl.GetPrim(), scene_prim=scene.GetPrim(), robot=robot,
                      physics_dt=SimulationManager.get_physics_dt(), fps=E["fps"], args=A, repo=REPO,
                      inputs={"asset": P(A.get("asset", "assets/omx_f_generated/omx_f_real.usda")), "cup_asset": P(A.get("cup_asset", "assets/paper_cup/paper_cup.usd")),
                              "intr": P(A["intr"]), "ext": P(A["ext"]), "cands": P(A["cands"]),
                              **({"wrist_intr": P(A["wrist_intr"]), "wrist_ext": P(A["wrist_ext"])} if "wrist_ext" in A else {})},
                      cameras={"front-left": {"K": I.K.tolist(), "size": [W, H], "render_size": [RW, RH], "world_from_camera_cv": Tfl.tolist()},
                               **({"wrist": {"K": Iw.K.tolist(), "size": [Ww, Hw], "render_size": [RWw, RHw], "link5_from_camera_cv": X5c.tolist(),
                                             "mount": WRIST_MOUNT, "near_m": float(A.get("wrist_near_m", "0.005"))}} if WRIST else {})},
                      render_api=RENDER_API)
    REPORT["bin"] = BIN_INFO   # 2026-11-01: the bin is part of the scene the replay is validated in
    ok = CR.check(REPORT, None if A["config_report"] == "none" else P(A["config_report"]))
    json.dump(REPORT, open(OUT / "config_report.json", "w"), indent=1)
    print("CONFIG REPORT", "PASS" if ok else "FAIL", OUT / "config_report.json", flush=True)
    if A.get("config_only", "0") == "1" or (not ok and A.get("config_strict", "1") == "1"):
        app.close(); sys.exit(0 if ok else 3)


def full(qb, g):
    val = {"joint1": qb[0], "joint2": qb[1], "joint3": qb[2], "joint4": qb[3], "joint5": qb[4], "gripper_joint_1": g, "gripper_joint_2": -g}
    return np.array([[val.get(n, 0.0) for n in names]], dtype=np.float32)


K0 = int(A.get("start", "0"))
q0 = full(body_rad(ST[K0], K0), G["gripper_rad_from_state"][K0])
robot.set_dof_positions(q0); robot.set_dof_position_targets(q0); robot.set_dof_velocities(np.zeros_like(q0))
pause(); app.update()
if A.get("diag", "0") == "1":
    def _np(x):
        x = x[0] if isinstance(x, tuple) else x
        return np.round(np.asarray(x.numpy() if hasattr(x, "numpy") else x), 4)
    phys_step(1, fabric=True)
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
    phys_step(1)
log = []
SETTLE_SELF = None
if SELF_CONTACTS:   # what touched while the arm was held at frame 0 (60 steps)
    SETTLE_SELF = {k: dict(v) for k, v in SC.items()}; SC.clear()
    jp0 = robot.get_dof_positions(); jp0 = np.asarray(jp0.numpy() if hasattr(jp0, "numpy") else jp0)[0]
    print("SELF settle", json.dumps(SETTLE_SELF), "q after settle (deg)", np.round(np.degrees(jp0), 2).tolist(), flush=True)


def link5_fk(qsim):   # link5 in world from FK of the simulated joint angles (what wrist_mount=fk uses)
    from reach_logger import fk as _fk
    return np.array(C["world_from_link0"]) @ _fk.link5_transform([float(v) for v in qsim[:5]])


def grab(k, qsim=None):
    if WRIST and WRIST_MOUNT == "fk":   # put the free wrist camera on FK of the sim's own joint angles before rendering
        camw.set_pose_cv(link5_fk(qsim) @ X5c)
    render()
    cv2.imwrite(str(OUT / f"sim_front-left_{k:04d}.jpg"), cv2.remap(cv2.cvtColor(cam.rgb(), cv2.COLOR_RGB2BGR), map_x, map_y, cv2.INTER_LINEAR))
    if WRIST:
        cv2.imwrite(str(OUT / f"sim_wrist_{k:04d}.jpg"), cv2.remap(cv2.cvtColor(camw.rgb(), cv2.COLOR_RGB2BGR), mxw, myw, cv2.INTER_LINEAR))


_L5 = []


def link_dbg(k, qsim):
    """link5 from the physics engine vs FK of the simulated joint angles, and where USD puts the wrist camera."""
    xc = UsdGeom.XformCache()
    for p in Usd.PrimRange(rp):
        if p.GetName() in ("link0", "link2", "link3", "link5"):
            print("LINKDBG", k, p.GetName(), [round(v, 4) for v in xc.GetLocalToWorldTransform(p).ExtractTranslation()], flush=True)
    if not WRIST:
        return
    if not _L5:
        _L5.append(RigidPrim(LINK5))
    pos, quat = _L5[0].get_world_poses()
    pos = np.asarray(pos.numpy() if hasattr(pos, "numpy") else pos)[0]; quat = np.asarray(quat.numpy() if hasattr(quat, "numpy") else quat)[0]
    Mp = np.eye(4); Mp[:3, :3] = Rr.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix(); Mp[:3, 3] = pos
    Mf = link5_fk(qsim)
    Mu = np.array(xc.GetLocalToWorldTransform(stage.GetPrimAtPath(camw.path))).T   # USD row-vector -> column
    Mc = Mp @ X5c; Mc[:3, :3] = Mc[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    print("LINK5DBG", k, "physics-vs-fk mm", np.round((Mp[:3, 3] - Mf[:3, 3]) * 1000, 2),
          "deg", round(float(np.degrees(Rr.from_matrix(Mp[:3, :3].T @ Mf[:3, :3]).magnitude())), 3),
          "| usd-camera-vs-physics-link5@X5c mm", np.round((Mu[:3, 3] - Mc[:3, 3]) * 1000, 2),
          "deg", round(float(np.degrees(Rr.from_matrix(Mu[:3, :3].T @ Mc[:3, :3]).magnitude())), 3), flush=True)


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
        phys_step(1)
    pos, _ = cup.get_world_poses(); pos = np.asarray(pos.numpy() if hasattr(pos, "numpy") else pos)[0]
    jp = robot.get_dof_positions(); jp = np.asarray(jp.numpy() if hasattr(jp, "numpy") else jp)[0]
    log.append({"frame": k + 1, "cup_xyz": [float(v) for v in pos], "q_sim": [float(v) for v in jp], "q_cmd": [float(v) for v in full(qb, gb)[0]]})
    if POSE_LOG:
        log[-1]["cup_pose"] = _pose(cup); log[-1]["link_poses"] = [_pose(r_) for r_ in _LINK_RP]
    if SELF_CONTACTS:
        log[-1]["self_contacts"] = {k: dict(v) for k, v in SC.items()}; SC.clear()
    if CONTACTS:
        log[-1]["contacts"] = {c: {"n": e["n"], "pos_mean": (e["pos"] / max(e["n"], 1)).tolist(), "impulse": e["imp"].tolist(),
                                   "z_range": [e["zmin"], e["zmax"]]} for c, e in CUR.items()}
        CUR.clear()
    if A.get("render", "1") == "1" and (k + 1) % STRIDE == 0:
        grab(k + 1, [jp[names.index(n)] for n in ("joint1", "joint2", "joint3", "joint4", "joint5")])
    if A.get("linkdbg", "0") == "1" and (k < 2 or (k + 1) % int(A.get("linkdbg_every", "50")) == 0):
        link_dbg(k + 1, [jp[names.index(n)] for n in ("joint1", "joint2", "joint3", "joint4", "joint5")])
    if CHECK and k > 4:
        break
json.dump({"cand": C["name"], "dof_names": names, "log": log, "self_contacts_settle": SETTLE_SELF, "link_bodies": LINK_BODIES if POSE_LOG else None,
           "robot_root": str(rp.GetPath()), "cup_root": str(cp.GetPath()),
           "input_pose_source": "recorded observation.state plus selected geometric calibration",
           "render_pose_source": "simulated articulation; physics tracking residuals are retained",
           "gripper_target_source": GRIP_SOURCE, "gravity_direction": gravity_direction.tolist(),
           "wrist_near_m": float(A.get("wrist_near_m", "0.005")), "wrist_ext": A.get("wrist_ext"), "wrist_mount": WRIST_MOUNT if WRIST else None, "render_api": RENDER_API, "args": A, "cup_asset": A.get("cup_asset", "assets/paper_cup/paper_cup.usd"), "bin": BIN_INFO},
          open(OUT / "physics_log.json", "w"))
z = [r["cup_xyz"][2] for r in log]
print(f"DONE frames {len(log)} cup z start {z[0]:.4f} max {max(z):.4f} end {z[-1]:.4f} xy end {log[-1]['cup_xyz'][:2]}", flush=True)
app.close()

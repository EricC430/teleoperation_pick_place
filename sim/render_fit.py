"""Isaac Sim 6.1 standalone: render the patched OMX-F at given joint radians, mounted at a 6-dof world_from_link0,
seen by a camera given as world_from_camera (OpenCV axes), with the real lens (padded pinhole render + remap).
Writes sim_<id>.png (rgb) and mask_<id>.png (robot pixels, from depth). Args: dir=<job dir> intr=<intrinsics json>."""
import sys, json, math
from pathlib import Path
A = dict(a.split("=", 1) for a in sys.argv[1:])
REPO = Path(r"D:\teleoperation_pick_place"); D = Path(A["dir"])
sys.path.insert(0, str(REPO / "sim"))
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
from pxr import Usd, UsdGeom, UsdLux, Gf, UsdPhysics
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.experimental.prims import Articulation
from isaacsim.sensors.experimental.rtx import RtxCamera, CameraSensor
from scipy.spatial.transform import Rotation as Rr
job = json.load(open(D / "job.json"))
ctx = omni.usd.get_context(); ctx.new_stage(); stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdPhysics.Scene.Define(stage, "/World/PhysicsScene").CreateGravityMagnitudeAttr(0.0)
UsdGeom.Xform.Define(stage, "/World")
UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(600.0)
B = np.array(job["world_from_link0"])
mnt = UsdGeom.Xform.Define(stage, "/World/RobotMount")
mnt.AddTransformOp().Set(Gf.Matrix4d(B.T.tolist()))          # USD row-vector convention: transpose
rp = stage.DefinePrim("/World/RobotMount/Robot", "Xform"); rp.GetReferences().AddReference(str(REPO / "assets/omx_f_generated/omx_f_real.usda"))
cupm = UsdGeom.Xform.Define(stage, "/World/CupMount"); cup_t = cupm.AddTranslateOp(); cup_t.Set(Gf.Vec3d(0, 0, -10))
stage.DefinePrim("/World/CupMount/Cup", "Xform").GetReferences().AddReference(str(REPO / "assets/paper_cup/paper_cup.usd"))
for p_ in Usd.PrimRange(stage.GetPrimAtPath("/World/CupMount")):
    if p_.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(p_).CreateKinematicEnabledAttr(True)
cam = RtxCamera("/World/Cam", positions=[[0, 0, 1.0]], orientations=[[1, 0, 0, 0]])
cc = next(UsdGeom.Camera(ch) for ch in Usd.PrimRange(stage.GetPrimAtPath("/World/Cam")) if ch.IsA(UsdGeom.Camera))
AP = 20.955; cc.GetFocalLengthAttr().Set(I.K[0, 0] * AP / RW); cc.GetHorizontalApertureAttr().Set(AP); cc.GetVerticalApertureAttr().Set(AP * RH / RW * I.K[0, 0] / I.K[1, 1])
cc.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 20.0))
sensor = CameraSensor(cam, resolution=(RH, RW), annotators=["rgb", "distance_to_image_plane"])
carb.settings.get_settings().set("/rtx/post/dlss/execMode", 0)
app.update()
ap = [str(p.GetPath()) for p in Usd.PrimRange(rp) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
robot = Articulation(ap[0]); app_utils.play(commit=True); names = list(robot.dof_names)
C = np.array(job["world_from_camera_cv"])
Ru = C[:3, :3] @ np.diag([1.0, -1.0, -1.0]); x, y, z, w = Rr.from_matrix(Ru).as_quat()
cam.set_world_poses(positions=[C[:3, 3].tolist()], orientations=[[w, x, y, z]])
order = ["joint1", "joint2", "joint3", "joint4", "joint5", "gripper_joint_1", "gripper_joint_2"]
first = True
for k, fr in job["frames"].items():
    val = dict(zip(order, fr["q"]))
    arr = np.array([[val.get(n, 0.0) for n in names]], dtype=np.float32); robot.set_dof_positions(arr); robot.set_dof_position_targets(arr)
    cxy = fr.get("cup"); cup_t.Set(Gf.Vec3d(cxy[0], cxy[1], 0.0) if cxy else Gf.Vec3d(0, 0, -10))
    for _ in range(20 if first else 8):        # the first frames after start render without the arm (seen 10-14)
        app.update()
    first = False
    d = sensor.get_data("rgb"); d = d[0] if isinstance(d, tuple) else d; rgb = (d.numpy() if hasattr(d, "numpy") else np.asarray(d))[..., :3].astype(np.uint8)
    z_ = sensor.get_data("distance_to_image_plane"); z_ = z_[0] if isinstance(z_, tuple) else z_; z_ = (z_.numpy() if hasattr(z_, "numpy") else np.asarray(z_)).reshape(RH, RW).astype(np.float32)
    z_[~np.isfinite(z_)] = 0
    cv2.imwrite(str(D / f"sim_{k}.png"), cv2.remap(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), map_x, map_y, cv2.INTER_LINEAR))
    zr = cv2.remap(z_, map_x, map_y, cv2.INTER_NEAREST)
    cv2.imwrite(str(D / f"mask_{k}.png"), ((zr > 0) & (zr < 5)).astype(np.uint8) * 255)
    np.save(D / f"depth_{k}.npy", zr)
print("RENDER", RW, RH, "DONE", flush=True); app.close()

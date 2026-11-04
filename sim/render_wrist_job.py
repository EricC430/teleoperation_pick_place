"""Isaac Sim render of a wrist-camera job from scripts/render_wrist_candidates.py prep -- any Isaac Sim from 4.5 on.

Only the camera capture differs between versions; everything else is plain USD (pxr), the same on every version:
the link5/link6/link7/cup meshes are UsdGeom.Mesh prims whose world transforms are written each frame from the job
(no articulation, no physics, so no joint API either), a 5 cm checker table, a dome and a distant light. The camera
renders an ideal pinhole canvas with the job's K, padded, and the job's remap tables turn it into the real wrist lens
(the physics_replay.py recipe). Output: <job>/renders/isaac/<candidate>/fNNNN.png (BGR).

Camera API, --api auto picks the first that imports:
  rtx6        Isaac Sim 6.x: isaacsim.sensors.experimental.rtx RtxCamera + CameraSensor (as sim/physics_replay.py).
  replicator  Isaac Sim 4.5 / 5.x: a pxr UsdGeom.Camera + omni.replicator.core render product, rgb annotator,
              rep.orchestrator.step(delta_time=0) per capture.

    sim/run_in_container.sh render_wrist_job.py --job /workspace/test_isaaclab/<job dir> [--frames 0,100] [--api replicator]
    <isaac-sim>\\python.bat <repo>\\sim\\render_wrist_job.py --job <job dir>
"""
import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--job", required=True)
ap.add_argument("--frames", default="")
ap.add_argument("--api", default="auto", choices=("auto", "rtx6", "replicator"))
ap.add_argument("--dome", type=float, default=250.0, help="dome light intensity")
ap.add_argument("--sun", type=float, default=300.0, help="distant light intensity")
ap.add_argument("--subframes", type=int, default=4, help="RTX subframes per capture (replicator); updates per capture (rtx6)")
args = ap.parse_args()

try:                                    # Isaac Sim 4.5 and later
    from isaacsim import SimulationApp
except ImportError:                     # 2023.1 / 4.0
    from omni.isaac.kit import SimulationApp

JOB = Path(args.job)
J = json.loads((JOB / "job.json").read_text())
L = J["lens"]; W, H, HW, HH = L["W"], L["H"], L["hw"], L["hh"]; RW, RH = 2 * HW, 2 * HH
app = SimulationApp({"headless": True, "width": RW, "height": RH})

import numpy as np  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, Vt  # noqa: E402
import omni.usd  # noqa: E402

try:
    import cv2
except ImportError:
    cv2 = None

K = np.array(L["K"]); maps = np.load(JOB / "lens_maps.npz"); MX, MY = maps["map_x"], maps["map_y"]
MESH = dict(np.load(JOB / "meshes.npz"))
COL = J["colors_bgr"]


def remap(img):
    if cv2 is not None:
        return cv2.remap(img, MX, MY, cv2.INTER_LINEAR, borderValue=0)
    x0 = np.clip(np.floor(MX).astype(int), 0, img.shape[1] - 2); y0 = np.clip(np.floor(MY).astype(int), 0, img.shape[0] - 2)
    fx = (MX - x0)[..., None]; fy = (MY - y0)[..., None]; im = img.astype(np.float32)
    out = (im[y0, x0] * (1 - fx) * (1 - fy) + im[y0, x0 + 1] * fx * (1 - fy) + im[y0 + 1, x0] * (1 - fx) * fy
           + im[y0 + 1, x0 + 1] * fx * fy)
    out[(MX < 0) | (MY < 0) | (MX > img.shape[1] - 1) | (MY > img.shape[0] - 1)] = 0
    return out.astype(np.uint8)


def save_png(path, bgr):
    path.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is not None:
        cv2.imwrite(str(path), bgr)
    else:
        from PIL import Image
        Image.fromarray(bgr[..., ::-1]).save(str(path))


def usd_matrix(T):     # column-vector 4x4 (numpy) -> USD row-vector Gf.Matrix4d
    return Gf.Matrix4d(*np.asarray(T, float).T.ravel().tolist())


def add_mesh(stage, path, tris, bgr):
    m = UsdGeom.Mesh.Define(stage, path)
    pts = tris.reshape(-1, 3)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts.tolist()]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(list(range(len(pts)))))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateDoubleSidedAttr(True)
    m.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)]))
    op = UsdGeom.Xformable(m).AddTransformOp()
    return op


ctx = omni.usd.get_context(); ctx.new_stage(); stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
ops = {n: add_mesh(stage, f"/World/{n}", MESH[n], COL[n]) for n in ("link5", "link6", "link7", "cup")}
# checker table, 5 cm cells, on the mat plane z = 0
tc = np.array(COL["table"], float)
cells = [[], []]
for i, x in enumerate(np.arange(-0.15, 0.70, 0.05)):
    for k, y in enumerate(np.arange(-0.50, 0.55, 0.05)):
        p = [(x, y, 0), (x + 0.05, y, 0), (x + 0.05, y + 0.05, 0), (x, y + 0.05, 0)]
        cells[(i + k) % 2] += [(p[0], p[1], p[2]), (p[0], p[2], p[3])]
for s, (tris, f) in enumerate(((cells[0], 0.85), (cells[1], 1.0))):
    add_mesh(stage, f"/World/table{s}", np.array(tris), tc * f).Set(usd_matrix(np.eye(4)))
UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(float(args.dome))
dl = UsdLux.DistantLight.Define(stage, "/World/Sun"); dl.CreateIntensityAttr(float(args.sun))
UsdGeom.Xformable(dl).AddRotateXYZOp().Set(Gf.Vec3f(30.0, 20.0, 0.0))


def set_intrinsics(cam_prim):
    c = UsdGeom.Camera(cam_prim); AP = 20.955
    c.GetFocalLengthAttr().Set(float(K[0, 0] * AP / RW)); c.GetHorizontalApertureAttr().Set(AP)
    c.GetVerticalApertureAttr().Set(float(AP * RH / RW * K[0, 0] / K[1, 1]))
    c.GetClippingRangeAttr().Set(Gf.Vec2f(float(J["near_m"]), 20.0))


def usd_cam_pose(T_cv):   # world_from_camera with OpenCV axes -> USD camera (looks down -Z, +Y up)
    T = np.asarray(T_cv, float).copy(); T[:3, :3] = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    return T


class Rtx6:
    """Isaac Sim 6.x experimental sensor API, the calls sim/physics_replay.py uses."""
    name = "rtx6"

    def __init__(self):
        import carb.settings
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
        self.cam = RtxCamera("/World/CamWrist", positions=[[0, 0, 1.0]], orientations=[[1, 0, 0, 0]])
        prim = next(p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/CamWrist")) if p.IsA(UsdGeom.Camera))
        set_intrinsics(prim)
        self.sensor = CameraSensor(self.cam, resolution=(RH, RW), annotators=["rgb"])
        carb.settings.get_settings().set("/rtx/post/dlss/execMode", 0)
        app.update()

    def capture(self, T_cv):
        from scipy.spatial.transform import Rotation as Rr
        T = usd_cam_pose(T_cv); x, y, z, w = Rr.from_matrix(T[:3, :3]).as_quat()
        self.cam.set_world_poses(positions=[T[:3, 3].tolist()], orientations=[[w, x, y, z]])
        for _ in range(max(1, args.subframes)):
            app.update()
        d = self.sensor.get_data("rgb"); d = d[0] if isinstance(d, tuple) else d
        return (d.numpy() if hasattr(d, "numpy") else np.asarray(d))[..., :3].astype(np.uint8)


class Replicator:
    """Isaac Sim 4.5 / 5.x: replicator render product on a plain USD camera."""
    name = "replicator"

    def __init__(self):
        try:
            import omni.replicator.core as rep
        except ImportError:   # not enabled in this app experience: enable it, then import
            import omni.kit.app
            omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)
            import omni.replicator.core as rep
        self.rep = rep
        cam = UsdGeom.Camera.Define(stage, "/World/CamWrist"); set_intrinsics(cam.GetPrim())
        self.op = UsdGeom.Xformable(cam).AddTransformOp()
        self.rp = rep.create.render_product("/World/CamWrist", (RW, RH))
        self.ann = rep.AnnotatorRegistry.get_annotator("rgb"); self.ann.attach([self.rp])
        rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)

    def capture(self, T_cv):
        self.op.Set(usd_matrix(usd_cam_pose(T_cv)))
        self.rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)
        return np.asarray(self.ann.get_data())[..., :3].astype(np.uint8)


def pick_backend():
    order = {"auto": (Rtx6, Replicator), "rtx6": (Rtx6,), "replicator": (Replicator,)}[args.api]
    errs = []
    for B in order:
        try:
            b = B(); print("camera API:", b.name, flush=True); return b
        except ImportError as e:
            errs.append(f"{B.name}: {e}")
    raise SystemExit("no usable camera API: " + "; ".join(errs))


cam = pick_backend()
frames = [int(x) for x in args.frames.split(",")] if args.frames else list(range(len(J["frames"])))
for f in frames:
    fr = J["frames"][f]
    for n in ops:
        ops[n].Set(usd_matrix(fr[n]))
    for c in J["candidates"]:
        rgb = cam.capture(fr["cam"][c])
        assert rgb.shape[:2] == (RH, RW), (rgb.shape, RH, RW)
        save_png(JOB / "renders/isaac" / c / f"f{f:04d}.png", remap(np.ascontiguousarray(rgb[..., ::-1])))
    if f % 50 == 0:
        print("frame", f, flush=True)
print("done", len(frames), "frames x", len(J["candidates"]), "candidates ->", JOB / "renders/isaac", flush=True)
app.close()

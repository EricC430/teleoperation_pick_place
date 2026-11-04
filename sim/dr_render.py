"""Domain-randomized re-render of a VALIDATED physics replay (docs/specs/S9_sim_dr_collect.md).

    /isaac-sim/python.sh sim/dr_render.py config=<dr_config.json> ep=<prep episode dir> log=<physics_log.json from
        physics_replay.py pose_log=1> object=<object USD> object_id=<id> variants=0,1 out=<render root>
        [src_root_host=<source LeRobot dataset dir on the host, for the export step>] [max_frames=0] [stride=1]
        [override=<json merged into the sampled parameters, for tests>]

What it does, per variant (one new stage each):
- Robot: the same asset as the replay, physics off; every link is placed at the world pose the physics replay logged
  for that frame, so contacts and tracking residuals are kept. Skin: the black matte override
  (assets/omx_f/omx_f_material_override.usda, as sim/render_black_replay.py) or the asset's own materials.
- Object: the variant USD (the one the replay validated), placed at its logged pose; appearance randomized
  (printed / plain paper, plastic, metal, ceramic, patterned: a texture made here + an OmniPBR material).
- Scene (EricC 2026-10-31): no table -- the ground IS the old table top (z = 0, where the cup stands), only the robot's
  riser (the platform under the base, 14.4 cm) stays, so every height relative to the arm is unchanged. Ground
  outdoors (sky, optional sun; campus-style surfaces and surroundings, no walls -- the former room mode was removed
  2026-11-05) or a studio; NVIDIA Base MDL materials or plain colours;
  HDRI dome light; extra sphere lights; optional props on the ground outside the arm's reach. The physics table is
  the same plane without an edge (physics_replay.py, 2026-11-05), so an object never ends below z = 0; if one does (a
  replay from before that change) it is rejected: on a ground it would sink.
- Cameras: front-left and wrist with the calibrated lenses (padded pinhole render remapped to the real distortion,
  as physics_replay.py); the wrist camera is a child of link5.
- Capture: Isaac Replicator (render products, rgb + instance segmentation annotators, orchestrator.step with RTX
  subframes). The randomization itself is sampled here with numpy from a per-variant seed and written as plain USD,
  so every value is in the manifest (S5 section 5 item 1: a variant must be traceable to its source and parameters).
Output: <out>/<object_id>_<episode dir>_v<k>/{front-left,wrist}/frame_NNNNN.jpg + manifest.json (the format
scripts/sim_replay_augment.py reads: action and observation.state are the SOURCE episode's, unchanged).
"""
import hashlib
import json
import math
import os
import sys
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
    return Path(r"D:\teleoperation_pick_place")


REPO = _repo()


def P(x):
    p = Path(x)
    return p if p.is_absolute() or p.exists() else REPO / p


sys.path[:0] = [str(HERE), str(REPO / "sim"), str(REPO)]
import numpy as np  # noqa: E402
import camera_distortion as cd  # noqa: E402

CFG = json.loads(Path(A["config"]).read_text())
CC = CFG["cameras"]; RC = CFG["render"]
I = cd.load_model(P(CC["front_left_intr"])); Iw = cd.load_model(P(CC["wrist_intr"]))


def lens_maps(m):
    us, vs = np.meshgrid(np.arange(m.width, dtype=np.float64), np.arange(m.height, dtype=np.float64))
    ideal = cd.undistort_pixels(np.stack([us.ravel(), vs.ravel()], 1), m)
    cx, cy = m.K[0, 2], m.K[1, 2]
    hw = int(math.ceil(max(np.abs(ideal[:, 0] - cx).max(), m.width / 2))) + 2
    hh = int(math.ceil(max(np.abs(ideal[:, 1] - cy).max(), m.height / 2))) + 2
    return 2 * hw, 2 * hh, (ideal[:, 0] - cx + hw).reshape(m.height, m.width).astype(np.float32), (ideal[:, 1] - cy + hh).reshape(m.height, m.width).astype(np.float32)


RW, RH, MX, MY = lens_maps(I); RWw, RHw, MXw, MYw = lens_maps(Iw)
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": RW, "height": RH})
import carb.settings  # noqa: E402
import cv2  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt  # noqa: E402
from scipy.spatial.transform import Rotation as Rr, Slerp  # noqa: E402
import inspect  # noqa: E402
try:
    import omni.replicator.core as rep
except ImportError:
    import omni.kit.app
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)
    import omni.replicator.core as rep
REP_STEP = dict(delta_time=0.0)
if "pause_timeline" in inspect.signature(rep.orchestrator.step).parameters:
    REP_STEP["pause_timeline"] = True
carb.settings.get_settings().set("/rtx/post/dlss/execMode", 0)
AP = 20.955

E = json.load(open(Path(A["ep"]) / "episode.json"))
LOG = json.load(open(A["log"]))
ROWS = LOG["log"]
assert ROWS and "link_poses" in ROWS[0], "the replay was run without pose_log=1"
LINK_NAMES = [p.rsplit("/", 1)[-1] for p in LOG["link_bodies"]]
REPLAY_ASSET = P(LOG["args"].get("asset", "assets/omx_f_generated/omx_f_real.usda"))
N = len(E["state"])
# poses per source frame: frame 0 = the first logged pose (the replay logs frames 1..N-1)
POSE = {0: ROWS[0]}
for r in ROWS:
    POSE[r["frame"]] = r
FRAMES = [f for f in range(N) if f in POSE or f == 0]
if int(A.get("stride", RC.get("stride", 1))) > 1:
    FRAMES = FRAMES[::int(A.get("stride", RC.get("stride", 1)))]
if int(A.get("max_frames", RC.get("max_frames", 0))):
    FRAMES = FRAMES[:int(A.get("max_frames", RC.get("max_frames", 0)))]
LAG = float(CC.get("render_lag_frames", 0.0))


def M44(p):   # [x y z qw qx qy qz] -> 4x4
    T = np.eye(4); T[:3, :3] = Rr.from_quat([p[4], p[5], p[6], p[3]]).as_matrix(); T[:3, 3] = p[:3]
    return T


def pose_at(t, key, i=None):
    """pose at (possibly fractional) frame t: lerp position, slerp rotation."""
    t = min(max(t, 0.0), max(POSE)); a = int(math.floor(t)); b = min(a + 1, max(POSE))
    while a not in POSE:
        a -= 1
    while b not in POSE and b < max(POSE):
        b += 1
    pa = POSE[a][key] if i is None else POSE[a][key][i]; pb = POSE[b][key] if i is None else POSE[b][key][i]
    if a == b or abs(t - a) < 1e-9:
        return M44(pa)
    w = (t - a) / (b - a)
    T = np.eye(4); T[:3, 3] = (1 - w) * np.array(pa[:3]) + w * np.array(pb[:3])
    rs = Rr.from_quat([[pa[4], pa[5], pa[6], pa[3]], [pb[4], pb[5], pb[6], pb[3]]])
    T[:3, :3] = Slerp([0, 1], rs)([w]).as_matrix()[0]
    return T


def gf(T):
    return Gf.Matrix4d(np.asarray(T, float).T.tolist())


def usd_cam(T_cv):
    T = np.array(T_cv, float); T[:3, :3] = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    return T


def ext_matrix(path, table_z=True):
    X = json.load(open(P(path))); q = X["quat_wxyz"]
    T = np.eye(4); T[:3, :3] = Rr.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); T[:3, 3] = X["pos_m"]
    if table_z:
        T[2, 3] -= X.get("table_top_z", 0.0)
    return T


TFL = ext_matrix(CC["front_left_ext"]); X5C = ext_matrix(CC["wrist_ext"], table_z=False)


# ------------------------------------------------------------------------------------------------- geometry helpers
def box_mesh(stage, path, size, center, uv_scale=1.0):
    """axis-aligned box with per-face planar UVs in metres x uv_scale (so MDL textures keep a physical scale)."""
    sx, sy, sz = [s / 2 for s in size]; cx, cy, cz = center
    pts, fc, fi, st = [], [], [], []
    faces = [((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (1, 0, 0), (0, -1, 0)), ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
             ((-1, 0, 0), (0, -1, 0), (0, 0, 1)), ((0, 1, 0), (-1, 0, 0), (0, 0, 1)), ((0, -1, 0), (1, 0, 0), (0, 0, 1))]
    half = np.array([sx, sy, sz])
    for n, u, v in faces:
        n, u, v = map(np.array, (n, u, v))
        hu, hv = abs(u @ half), abs(v @ half); c0 = np.array([cx, cy, cz]) + n * abs(n @ half)
        base = len(pts)
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = c0 + a * hu * u + b * hv * v; pts.append(p); st.append(((a * hu) * uv_scale, (b * hv) * uv_scale))
        fc.append(4); fi.extend([base, base + 1, base + 2, base + 3])
    m = UsdGeom.Mesh.Define(stage, path)
    m.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in pts]))
    m.GetFaceVertexCountsAttr().Set(fc); m.GetFaceVertexIndicesAttr().Set(fi); m.GetSubdivisionSchemeAttr().Set("none")
    UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(Vt.Vec2fArray([Gf.Vec2f(*map(float, s)) for s in st]))
    return m.GetPrim()


def mdl_material(stage, path, url):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(Sdf.AssetPath(url), "mdl"); sh.SetSourceAssetSubIdentifier(Path(url).stem, "mdl")
    out = mat.CreateSurfaceOutput("mdl"); out.ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out"); mat.CreateDisplacementOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mat


def omnipbr(stage, path, color=None, texture=None, rough=0.6, metal=0.0, spec=0.5):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl"); sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    if color is not None:
        sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, color)))
    if texture is not None:
        sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(rough))
    sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(metal))
    sh.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(float(spec))
    for o in (mat.CreateSurfaceOutput("mdl"), mat.CreateVolumeOutput("mdl"), mat.CreateDisplacementOutput("mdl")):
        o.ConnectToSource(sh.ConnectableAPI(), "out")
    return mat


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)


def label(prim, name):
    """instance segmentation needs a semantic label; old and new schema, whichever this Isaac has"""
    try:
        from pxr import UsdSemantics
        UsdSemantics.LabelsAPI.Apply(prim, "class").CreateLabelsAttr().Set([name]); return
    except Exception:
        pass
    try:
        from pxr import Semantics
        s = Semantics.SemanticsAPI.Apply(prim, "Semantics"); s.CreateSemanticTypeAttr().Set("class"); s.CreateSemanticDataAttr().Set(name)
    except Exception as e:
        print("no semantics schema:", e, flush=True)


# ------------------------------------------------------------------------------------------------- randomization
def choice_w(rng, d):
    k = list(d); w = np.array([d[x] for x in k], float); return k[rng.choice(len(k), p=w / w.sum())]


def U(rng, ab):
    return float(rng.uniform(ab[0], ab[1]))


def kelvin_rgb(k):   # approximate blackbody colour (Tanner Helland), for the plain lights' tint
    t = k / 100.0
    r = 255 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255 if t >= 66 else (0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    return [min(max(c, 0), 255) / 255 for c in (r, g, b)]


def object_texture(rng, kind, out_png):
    """texture for the cup UV layout of sim/dr_objects.py: rows 0..0.96 = outer wall (v up), 0.97..1 = inside."""
    W, H = 512, 512
    img = np.zeros((H, W, 3), np.float32)
    ap = CFG["objects"]["appearance"]
    base = rng.uniform(0.15, 0.95, 3)
    if kind in ("printed_paper", "plain_paper"):
        base = np.array([0.93, 0.91, 0.86]) * rng.uniform(0.9, 1.05) if rng.random() < 0.6 else base
    img[:] = base
    vrow = lambda v: int((1 - v) * (H - 1))   # noqa: E731  (image row 0 = v 1)
    params = {"kind": kind, "base": base.round(3).tolist()}
    if kind == "printed_paper":   # a dark band at the bottom like the real cup, plus a random print
        bh = U(rng, ap["band_height_frac"]) * 0.96; bc = rng.uniform(0.05, 0.6, 3)
        img[vrow(bh):vrow(0) + 1] = bc; params.update(band_frac=round(bh / 0.96, 3), band_color=bc.round(3).tolist())
        for _ in range(int(rng.integers(3, 12))):   # blobs / logo-ish shapes
            cx, cy, r = rng.uniform(0, W), rng.uniform(vrow(0.9), vrow(bh)), rng.uniform(6, 40)
            cv2.circle(img, (int(cx), int(cy)), int(r), tuple(map(float, rng.uniform(0, 1, 3))), -1)
    elif kind == "patterned":
        c2 = rng.uniform(0, 1, 3); n = int(rng.integers(4, 20)); p = rng.choice(["stripes_h", "stripes_v", "checker", "dots"])
        for i in range(n):
            if p == "stripes_h" and i % 2:
                img[int(i * vrow(0) / n):int((i + 1) * vrow(0) / n)] = c2
            elif p == "stripes_v" and i % 2:
                img[:vrow(0), int(i * W / n):int((i + 1) * W / n)] = c2
            elif p == "checker":
                for j in range(n):
                    if (i + j) % 2:
                        img[int(i * vrow(0) / n):int((i + 1) * vrow(0) / n), int(j * W / n):int((j + 1) * W / n)] = c2
            elif p == "dots":
                for j in range(n):
                    cv2.circle(img, (int((j + 0.5) * W / n), int((i + 0.5) * vrow(0) / n)), max(2, int(W / n / 4)), tuple(map(float, c2)), -1)
        params.update(pattern=str(p), color2=c2.round(3).tolist(), n=n)
    inside = np.array(ap["inside_colors"][int(rng.integers(len(ap["inside_colors"])))]) if kind in ("printed_paper", "plain_paper") else base
    img[:vrow(0.965)] = inside; params["inside"] = np.asarray(inside).round(3).tolist()
    img = np.clip(img + rng.normal(0, 0.015, img.shape), 0, 1)   # paper grain / print noise
    cv2.imwrite(str(out_png), (img[..., ::-1] * 255).astype(np.uint8))
    rough, metal = {"printed_paper": (0.85, 0.0), "plain_paper": (0.9, 0.0), "plastic": (U(rng, [0.15, 0.45]), 0.0),
                    "metal": (U(rng, [0.15, 0.4]), 1.0), "ceramic": (U(rng, [0.05, 0.25]), 0.0), "patterned": (U(rng, [0.3, 0.8]), 0.0)}[kind]
    params.update(roughness=round(rough, 3), metallic=metal)
    return params


def sample(rng):
    env, lt, ap, arm = CFG["environment"], CFG["lighting"], CFG["objects"]["appearance"], CFG["arm"]
    s = {"arm_skin": choice_w(rng, arm["skins"]), "object_kind": choice_w(rng, ap["kinds"]), "mode": choice_w(rng, env["modes"])}
    s["hdri"] = env["skies"] + env["hdri"][s["mode"]][int(rng.integers(len(env["hdri"][s["mode"]])))]
    gl = env["ground_materials"][s["mode"]]
    base = np.array(env["ground_plain_colors"][int(rng.integers(len(env["ground_plain_colors"])))])   # earth tones only (2026-11-05)
    s["ground"] = ({"color": np.clip(base + rng.normal(0, env["ground_plain_color_jitter"], 3), 0.05, 0.95).round(3).tolist()}
                   if rng.random() < env["ground_plain_color_prob"][s["mode"]]
                   else {"mdl": env["materials"] + gl[int(rng.integers(len(gl)))]})
    s["dome"] = {"intensity": U(rng, lt["dome_intensity"]), "exposure": U(rng, lt["dome_exposure"]), "temperature": U(rng, lt["color_temperature_K"]),
                 "rotation_deg": U(rng, lt["dome_rotation_deg"])}
    s["lights"] = []
    for _ in range(int(rng.integers(lt["extra_lights"][0], lt["extra_lights"][1] + 1))):
        ang = rng.uniform(0, 2 * math.pi); d = U(rng, lt["sphere_distance_m"])
        tint = np.clip(np.array(kelvin_rgb(U(rng, lt["color_temperature_K"]))) + rng.normal(0, lt["light_color_jitter"], 3), 0.05, 1)
        s["lights"].append({"pos": [0.3 + d * math.cos(ang), d * math.sin(ang), U(rng, lt["sphere_height_m"])], "intensity": U(rng, lt["sphere_intensity"]),
                            "radius": U(rng, lt["sphere_radius_m"]), "color": tint.round(3).tolist()})
    if s["mode"] == "outdoor" and rng.random() < lt["outdoor_sun_prob"]:
        s["sun"] = {"intensity": U(rng, lt["sun_intensity"]), "angle": U(rng, lt["sun_angle_deg"]),
                    "elev_deg": float(rng.uniform(20, 75)), "azim_deg": float(rng.uniform(0, 360))}
    pr = env["props"]; s["props"] = []
    if s["mode"] == "outdoor" and "outdoor" in pr:   # 2026-11-05: campus-style surroundings (vegetation, rocks, planters, bollards, fencing)
        pr = {**pr, **pr["outdoor"]}
    for _ in range(int(rng.integers(pr["count"][0], pr["count"][1] + 1))):
        ax, ay = pr["area_x_m"], pr["area_y_m"]
        for _try in range(50):
            x, y = rng.uniform(*ax), rng.uniform(*ay)
            if math.hypot(x - BASE_XY[0], y - BASE_XY[1]) > pr["keep_out_radius_m"]:
                s["props"].append({"usd": pr["root"] + pr["usd"][int(rng.integers(len(pr["usd"])))], "max_size_m": pr.get("max_size_m", 0.6), "shrink_to_fit": pr.get("shrink_to_fit", False), "xy": [x, y], "yaw_deg": float(rng.uniform(0, 360))}); break
    s["front_left_jitter"] = {"t": rng.normal(0, CC.get("front_left_jitter_m", 0.0), 3).tolist(), "r": rng.normal(0, CC.get("front_left_jitter_rad", 0.0), 3).tolist()}
    return s


# ------------------------------------------------------------------------------------------------- scene
BASE_XY = [float(v) for v in ROWS[0]["link_poses"][LINK_NAMES.index("link0")][:2]]
GROUND_Z = 0.0   # the cup's plane = the old table top


def build(stage, s, vdir):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World"); env = CFG["environment"]
    # robot: same asset as the replay, physics off, links posed from the log
    rp = stage.DefinePrim("/World/Robot", "Xform"); rp.GetReferences().AddReference(str(REPLAY_ASSET))
    for _ in range(4):
        inst = [p for p in Usd.PrimRange(rp) if p.IsInstanceable()]
        for p in inst:
            p.SetInstanceable(False)
        if not inst:
            break
    links, meshes = {}, 0
    for p in list(Usd.PrimRange(rp)):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
        if p.IsA(UsdPhysics.Joint):
            p.SetActive(False)
        if p.GetName() == "visuals" and p.GetParent().GetName() in ("end_effector_link", "world"):
            UsdGeom.Imageable(p).MakeInvisible()
        if p.GetName() in LINK_NAMES and p.GetParent() == rp:
            xf = UsdGeom.Xformable(p); xf.ClearXformOpOrder(); links[p.GetName()] = xf.AddTransformOp()
    if s["arm_skin"] == "black":
        blk = UsdShade.Material.Define(stage, "/World/Looks/ArmBlack")
        blk.GetPrim().GetReferences().AddReference(str(P(CFG["arm"]["black_material"])), "/omx_f/Looks/Black_Matte")
        for p in Usd.PrimRange(rp):
            if p.IsA(UsdGeom.Mesh) and "collisions" not in str(p.GetPath()):
                bind(p, blk); meshes += 1
    missing = [n for n in LINK_NAMES if n not in links]
    assert not missing, f"links not found under the robot prim: {missing}"
    # wrist camera: child of link5 (rides on the logged link5 pose)
    l5 = rp.GetPath().AppendChild("link5")
    wm = UsdGeom.Xform.Define(stage, l5.AppendChild("CamWristMount")); wm.AddTransformOp().Set(gf(usd_cam(X5C)))
    camw = UsdGeom.Camera.Define(stage, wm.GetPath().AppendChild("CamWrist"))
    # object
    op = stage.DefinePrim("/World/Obj", "Xform"); op.GetReferences().AddReference(str(P(A["object"])))
    for p in Usd.PrimRange(op):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
    oxf = UsdGeom.Xformable(op); oxf.ClearXformOpOrder(); oop = oxf.AddTransformOp()
    obj_mesh = next(p for p in Usd.PrimRange(op) if p.IsA(UsdGeom.Mesh) and p.GetName() == "Mesh")
    tex = vdir / "object_texture.png"
    s["object_appearance"] = object_texture(np.random.default_rng(s["_seed"] + 1), s["object_kind"], tex)
    oa = s["object_appearance"]
    bind(obj_mesh, omnipbr(stage, "/World/Looks/Obj", texture=tex, rough=oa["roughness"], metal=oa["metallic"]))
    label(op, "object")
    # the robot's riser on the ground (no table: the ground is at the old table top)
    cx, cy = env["scene_center_xy_m"]
    riser = box_mesh(stage, "/World/Riser", (0.16, 0.17, 0.144), (BASE_XY[0] - 0.02, BASE_XY[1], 0.072 - 0.001))
    bind(riser, omnipbr(stage, "/World/Looks/Riser", color=[0.25, 0.25, 0.27], rough=0.6))
    # the bin, exactly where the physics replay had it (its log), on its stand; outside colour random, inside = a liner
    # colour (the real one has a green bag, seen by the wrist camera at release)
    if LOG.get("bin"):
        from shell_mesh import define_bin
        b = LOG["bin"]; brng = np.random.default_rng(s["_seed"] + 2)
        bi = define_bin(stage, "/World/Bin", b["center_xyz"], b["base_dia"], b["top_dia"], b["height"], b["wall"], b["floor"])
        for q in ("/World/Bin/Mesh", "/World/Bin_Stand"):
            UsdPhysics.CollisionAPI(stage.GetPrimAtPath(q)).CreateCollisionEnabledAttr(False)   # render only
        lc = CFG.get("bin", {}).get("liner_colors", [[0.25, 0.6, 0.35]])
        outer = brng.uniform(0.15, 0.95, 3); liner = np.array(lc[int(brng.integers(len(lc)))]) + brng.normal(0, 0.04, 3)
        btex = vdir / "bin_texture.png"; bimg = np.zeros((64, 64, 3)); bimg[:] = outer; bimg[:int(64 * (1 - 0.965))] = np.clip(liner, 0, 1)
        cv2.imwrite(str(btex), (bimg[..., ::-1] * 255).astype(np.uint8))
        bind(stage.GetPrimAtPath("/World/Bin/Mesh"), omnipbr(stage, "/World/Looks/Bin", texture=btex, rough=float(brng.uniform(0.3, 0.8))))
        bind(stage.GetPrimAtPath("/World/Bin_Stand"), omnipbr(stage, "/World/Looks/BinStand", color=[0.25, 0.25, 0.27], rough=0.6))
        s["bin_appearance"] = {"outer": outer.round(3).tolist(), "liner": np.clip(liner, 0, 1).round(3).tolist()}
    # no walls, no ceiling (EricC 2026-11-05): the ground plane only, under a dome light (sky / studio HDRI)
    floor = box_mesh(stage, "/World/Floor", (30.0, 30.0, 0.02), (cx, cy, GROUND_Z - 0.01))
    g = s["ground"]
    bind(floor, omnipbr(stage, "/World/Looks/Ground", color=g["color"], rough=0.7) if "color" in g else mdl_material(stage, "/World/Looks/Ground", g["mdl"]))
    # lights
    dl = UsdLux.DomeLight.Define(stage, "/World/Dome"); d = s["dome"]
    dl.CreateTextureFileAttr(Sdf.AssetPath(s["hdri"])); dl.CreateTextureFormatAttr("latlong")
    dl.CreateIntensityAttr(d["intensity"]); dl.CreateExposureAttr(d["exposure"])
    dl.CreateEnableColorTemperatureAttr(True); dl.CreateColorTemperatureAttr(d["temperature"])
    dl.AddRotateZOp().Set(d["rotation_deg"])
    for i, L in enumerate(s["lights"]):
        sl = UsdLux.SphereLight.Define(stage, f"/World/Lights/Sphere{i}")
        sl.CreateIntensityAttr(L["intensity"]); sl.CreateRadiusAttr(L["radius"]); sl.CreateColorAttr(Gf.Vec3f(*L["color"]))
        sl.AddTranslateOp().Set(Gf.Vec3d(*L["pos"]))
    if "sun" in s:
        sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun"); sun.CreateIntensityAttr(s["sun"]["intensity"]); sun.CreateAngleAttr(s["sun"]["angle"])
        sun.AddRotateZOp().Set(s["sun"]["azim_deg"]); sun.AddRotateXOp().Set(90 - s["sun"]["elev_deg"])
    # props on the ground, outside the arm's reach
    for i, pr in enumerate(s["props"]):
        pp = stage.DefinePrim(f"/World/Props/P{i}", "Xform"); pp.GetReferences().AddReference(pr["usd"])
        for q in Usd.PrimRange(pp):
            if q.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(q).CreateRigidBodyEnabledAttr(False)
            if q.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(q).CreateCollisionEnabledAttr(False)
        # a reference does not convert units or up axis: read them from the asset itself (2026-10-31: SM_Mug_* came in
        # 100x too big and swallowed the camera -> an all-black variant)
        ps = Usd.Stage.Open(pr["usd"]); sc = float(UsdGeom.GetStageMetersPerUnit(ps)); yup = UsdGeom.GetStageUpAxis(ps) == UsdGeom.Tokens.y
        bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(ps.GetDefaultPrim() or ps.GetPseudoRoot()).ComputeAlignedRange()
        size = np.array(bb.GetSize()) * sc; zmin = float((bb.GetMin()[1] if yup else bb.GetMin()[2]) * sc)
        pr.update(scale=sc, y_up=bool(yup), size_m=size.round(3).tolist())
        if np.isfinite(size).all() and size.max() > pr["max_size_m"] and pr.get("shrink_to_fit"):
            # outdoor vegetation comes at full size (a shrub 3.5 m across): scale it down to the limit, as a smaller plant
            k = pr["max_size_m"] / float(size.max()); sc *= k; zmin *= k; pr["fit_scale"] = round(k, 3)
        elif not np.isfinite(size).all() or size.max() > pr["max_size_m"]:
            pp.SetActive(False); pr["dropped"] = "size"; print("prop dropped (size)", pr, flush=True); continue
        # one matrix op: the referenced asset may author its own xformOps of another precision (a float scale op on a
        # double3 one raised an error, 2026-11-05), which ClearXformOpOrder does not remove
        S = Gf.Matrix4d(1.0); S.SetScale(Gf.Vec3d(sc, sc, sc))
        Rz = Gf.Matrix4d(1.0); Rz.SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), pr["yaw_deg"]))
        Rx = Gf.Matrix4d(1.0); Rx.SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90.0 if yup else 0.0))
        Tr = Gf.Matrix4d(1.0); Tr.SetTranslate(Gf.Vec3d(pr["xy"][0], pr["xy"][1], GROUND_Z - zmin))
        xf = UsdGeom.Xformable(pp); xf.ClearXformOpOrder(); xf.AddTransformOp().Set(S * Rx * Rz * Tr)
    # front-left camera (calibrated, optional jitter)
    T = TFL.copy(); j = s["front_left_jitter"]
    T[:3, 3] += np.array(j["t"]); T[:3, :3] = Rr.from_rotvec(j["r"]).as_matrix() @ T[:3, :3]
    camf = UsdGeom.Camera.Define(stage, "/World/CamFL"); camf.AddTransformOp().Set(gf(usd_cam(T)))
    for c, K, rw, rh, near in ((camf, I.K, RW, RH, 0.01), (camw, Iw.K, RWw, RHw, float(CC.get("wrist_near_m", 0.005)))):
        c.GetFocalLengthAttr().Set(K[0, 0] * AP / rw); c.GetHorizontalApertureAttr().Set(AP)
        c.GetVerticalApertureAttr().Set(AP * rh / rw * K[0, 0] / K[1, 1]); c.GetClippingRangeAttr().Set(Gf.Vec2f(near, 60.0))
    return links, oop, str(camf.GetPath()), str(camw.GetPath()), meshes


# ------------------------------------------------------------------------------------------------- main loop
OUT = Path(A["out"]); OUT.mkdir(parents=True, exist_ok=True)
ep_name = Path(A["ep"]).name; oid = A["object_id"]
ctx = omni.usd.get_context()
for v in [int(x) for x in A["variants"].split(",")]:
    seed = int(hashlib.sha256(f"{CFG['seed']}|{oid}|{E['dataset']}|{E['episode']}|{v}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    vdir = OUT / f"{oid}_{ep_name}_v{v}"
    if (vdir / "manifest.json").exists():   # done in an earlier (timed-out and retried) run
        print("VARIANT EXISTS", vdir, flush=True); continue
    (vdir / "front-left").mkdir(parents=True, exist_ok=True); (vdir / "wrist").mkdir(exist_ok=True)
    s = sample(rng); s["_seed"] = seed
    if "override" in A:   # tests: force some sampled values, e.g. override='{"wall": {"color": [0.8, 0.8, 0.8]}}'
        s.update(json.loads(A["override"])); s["override"] = json.loads(A["override"])
    print("VARIANT START", vdir.name, json.dumps({k: val for k, val in s.items() if not k.startswith("_")}), flush=True)
    ctx.new_stage(); stage = ctx.get_stage()
    links, oop, pfl, pw, nmesh = build(stage, s, vdir)
    print("VARIANT BUILT", vdir.name, flush=True)
    app.update()
    rp_fl = rep.create.render_product(pfl, (RW, RH)); rp_w = rep.create.render_product(pw, (RWw, RHw))
    ann = {}
    for nm, rpd in (("fl", rp_fl), ("w", rp_w)):
        a_rgb = rep.AnnotatorRegistry.get_annotator("rgb"); a_rgb.attach([rpd])
        a_seg = rep.AnnotatorRegistry.get_annotator("instance_segmentation_fast", init_params={"colorize": False}); a_seg.attach([rpd])
        ann[nm] = (a_rgb, a_seg)
    # auto exposure: scale every light so that the front-left frame-0 mean grey lands on a sampled target
    tgt = U(rng, RC.get("brightness_target", [85, 175])); gain = 1.0
    lights = [p for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI) or p.IsA(UsdLux.BoundableLightBase) or p.IsA(UsdLux.NonboundableLightBase)]
    base_int = {str(p.GetPath()): p.GetAttribute("inputs:intensity").Get() for p in lights}
    for n_, op in links.items():
        op.Set(gf(pose_at(FRAMES[0] - LAG, "link_poses", LINK_NAMES.index(n_))))
    oop.Set(gf(pose_at(FRAMES[0] - LAG, "cup_pose")))
    for it in range(int(RC.get("brightness_iters", 4))):
        rep.orchestrator.step(rt_subframes=int(RC["first_subframes"]), **REP_STEP)
        g = cv2.cvtColor(cv2.remap(np.asarray(ann["fl"][0].get_data())[..., :3], MX, MY, cv2.INTER_LINEAR), cv2.COLOR_RGB2GRAY)
        m = float(g.mean())
        if m < 1.0 or abs(m - tgt) < 8:
            break
        gain *= float(np.clip((tgt / m) ** 2.2, 0.05, 20.0))   # sRGB: brightness ~ light^(1/2.2)
        for p in lights:
            if base_int[str(p.GetPath())] is not None:
                p.GetAttribute("inputs:intensity").Set(float(base_int[str(p.GetPath())] * gain))
    s["exposure"] = {"target_mean": round(tgt, 1), "light_gain": round(gain, 4), "frame0_mean_before_last_step": round(m, 1), "iterations": it + 1}
    frames_out = []; vis = {"fl": [], "w": []}; rejected = None
    zmin = min(POSE[f]["cup_pose"][2] for f in FRAMES if f in POSE)
    if zmin < -0.01:
        rejected = f"object below the ground in the replay (min z {zmin:.3f} m: replay from the old finite physics table, or a physics fault)"
    for idx, f in enumerate([] if rejected else FRAMES):
        t = f - LAG
        for n, op in links.items():
            op.Set(gf(pose_at(t, "link_poses", LINK_NAMES.index(n))))
        oop.Set(gf(pose_at(t, "cup_pose")))
        rep.orchestrator.step(rt_subframes=int(RC["first_subframes"] if idx == 0 else RC["subframes"]), **REP_STEP)
        rec = {"frame_index": f, "action": E["action"][f], "observation_state": E["state"][f]}
        for nm, (mx, my, key) in (("fl", (MX, MY, "front-left")), ("w", (MXw, MYw, "wrist"))):
            a_rgb, a_seg = ann[nm]
            im = np.asarray(a_rgb.get_data())[..., :3]
            im = cv2.remap(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), mx, my, cv2.INTER_LINEAR)
            if idx == 0 and im.mean() < 2.0:   # an all-black frame (2026-10-31: a prop's material blanked the whole image)
                rejected = f"black {key} frame (mean {im.mean():.2f})"
            fn = f"{key}/frame_{f:05d}.jpg"; cv2.imwrite(str(vdir / fn), im, [cv2.IMWRITE_JPEG_QUALITY, int(RC["jpeg_quality"])])
            rec[f"image_{key}"] = fn
            try:   # object pixels in the real-lens image (QA: is the object where the episode needs it)
                sd = a_seg.get_data(); seg = np.asarray(sd["data"]); idmap = sd["info"].get("idToLabels", {})
                ids = [int(k) for k, lab in idmap.items() if "Obj" in str(lab) or "object" in str(lab)]
                frac = float(np.isin(seg, ids).mean()) if ids else None
            except Exception:
                frac = None
            rec[f"object_px_frac_{key}"] = frac; vis[nm].append(frac)
        frames_out.append(rec)
        if rejected:
            break
        if idx % 100 == 0:
            print(f"variant {v} frame {idx + 1}/{len(FRAMES)}", flush=True)
    for a_rgb, a_seg in ann.values():
        a_rgb.detach(); a_seg.detach()
    rp_fl.destroy(); rp_w.destroy()
    s_out = {k: val for k, val in s.items() if not k.startswith("_")}
    manifest = {
        "dataset_fps": E["fps"], "stride": int(A.get("stride", RC.get("stride", 1))), "frames": frames_out,
        "source_dataset": E["dataset"], "source_episode_id": E["episode"], "source_placement": E.get("placement"),
        "source_dataset_root": A.get("src_root_host", ""), "source_dataset_root_host": A.get("src_root_host", ""),
        "object_id": oid, "object_usd": A["object"],
        "dr": {"enabled": True, "seed": seed, "preset": "sim/dr_config.json", "variant": v, "params": s_out},
        "grasp": {"mode": "physics", "validated_by": "sim/physics_replay.py (cup lifted >= 5 cm for >= 10 frames on a close)",
                  "physics_log": A["log"]},
        "driven_by": "kinematic re-render of the physics replay's logged link and object poses (world frame)",
        "sign": "n/a (poses come from the replay)",
        "render_lag_frames": LAG,
        "fidelity": {"geometry_aligned": True, "geometry_note": "front-left 2026-10-14 extrinsics + 2026-10-07 lens; wrist CAD lens mount; arm 2026-10-14 link-marker fit",
                     "object_is_source_object": oid == "cup_ref", "visual_domain": "randomized (not the real room)",
                     "physics_validated": True, "arm_skin": s["arm_skin"], "black_meshes": nmesh},
        "qa": {"object_visible_frac_front-left": float(np.mean([x is not None and x > 0 for x in vis["fl"]])),
               "object_visible_frac_wrist": float(np.mean([x is not None and x > 0 for x in vis["w"]]))},
    }
    if rejected:
        manifest["rejected"] = rejected
    json.dump(manifest, open(vdir / "manifest.json", "w"))
    print(f"VARIANT {'REJECTED (' + rejected + ')' if rejected else 'DONE'} {vdir} frames {len(frames_out)} mode {s['mode']} skin {s['arm_skin']} object {s['object_kind']} "
          f"visible fl {manifest['qa']['object_visible_frac_front-left']:.2f} wrist {manifest['qa']['object_visible_frac_wrist']:.2f}", flush=True)
app.close()

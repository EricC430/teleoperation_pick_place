"""Object geometry variants for the DR collection: thin-walled cups (USD with UVs, rigid body, SDF-ready collider).

    /isaac-sim/python.sh sim/dr_objects.py config=<dr_config.json> out=<dir>

Truncated-cone shell from sim/shell_mesh.py (outer wall, inner wall, rim, and since 2026-11-01 a real bottom: the
earlier mesh, like sim/gen_paper_cup_usd.py's, was open across the bottom; origin = base centre, +Z up) and the same
prim layout as gen_paper_cup_usd.py (/PaperCup with Mesh, Collision, PhysicsMaterial), so sim/physics_replay.py
takes a variant through cup_asset= unchanged. Added: a face-varying `st` UV set for textures (outer wall: u around,
v up over 0..0.96; inner wall, rim and base all map to the strip v 0.97..1.0, the "inside" colour of a texture), and a
plain material the renderer replaces. The `cup_ref` variant has the parameters of assets/paper_cup/paper_cup.usd;
its outer wall is checked against that file (`ref_check` in objects.json; the file itself has no bottom).

Output: <out>/<id>.usd per variant, <out>/objects.json (id, parameters, container path).
"""
import json
import math
import os
import sys
from pathlib import Path

A = dict(a.split("=", 1) for a in sys.argv[1:])
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
import numpy as np  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shell_mesh import INSIDE_V, cup_mesh  # noqa: E402,F401  closed bottom since 2026-11-01
from bottle_mesh import bottle_mesh, mass_props  # noqa: E402  "shape": "bottle" entries (2026-11-11)


def write(path, p):
    # "shape": "bottle" (sim/bottle_mesh.py, 2026-11-11): a closed solid of revolution with the hollow bottle's mass
    # properties set explicitly; same prim layout (/PaperCup/...) so physics_replay.py and dr_render.py take it unchanged.
    # Cups (no "shape") go through exactly the code path they always did.
    bottle = p.get("shape") == "bottle"
    if bottle:
        pts, fc, fi, uv = bottle_mesh(p)
        mp = mass_props(p); p["mass"] = mp["mass"]
    else:
        pts, fc, fi, uv = cup_mesh(p["top_dia"] / 2, p["bot_dia"] / 2, p["height"], p.get("wall", 0.001), p.get("bottom"))
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/PaperCup"); root.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0)); stage.SetDefaultPrim(root.GetPrim())
    P = Vt.Vec3fArray([Gf.Vec3f(*map(float, q)) for q in pts])
    for name, vis in (("Mesh", True), ("Collision", False)):
        m = UsdGeom.Mesh.Define(stage, f"/PaperCup/{name}")
        m.GetPointsAttr().Set(P); m.GetFaceVertexCountsAttr().Set(Vt.IntArray(fc.tolist())); m.GetFaceVertexIndicesAttr().Set(Vt.IntArray(fi.tolist()))
        m.GetSubdivisionSchemeAttr().Set("none")
        if vis:
            st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
            st.Set(Vt.Vec2fArray([Gf.Vec2f(float(a), float(b)) for a, b in uv]))
        else:
            m.GetVisibilityAttr().Set("invisible")
    mat = UsdShade.Material.Define(stage, "/PaperCup/PaperMaterial")
    sh = UsdShade.Shader.Define(stage, "/PaperCup/PaperMaterial/Shader"); sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.92, 0.88, 0.80)); sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/PaperCup/Mesh")).Bind(mat)
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    col = stage.GetPrimAtPath("/PaperCup/Collision")
    UsdPhysics.CollisionAPI.Apply(col); UsdPhysics.MeshCollisionAPI.Apply(col).GetApproximationAttr().Set("convexDecomposition")
    ma = UsdPhysics.MassAPI.Apply(root.GetPrim()); ma.GetMassAttr().Set(float(p["mass"]))
    if bottle:   # uniform density on the solid collider would put the COM and inertia of a full bottle
        ma.GetCenterOfMassAttr().Set(Gf.Vec3f(*map(float, mp["com"])))
        ma.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float, mp["inertia"])))
        ma.GetPrincipalAxesAttr().Set(Gf.Quatf(1, 0, 0, 0))
    pm = UsdShade.Material.Define(stage, "/PaperCup/PhysicsMaterial"); pa = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
    pa.GetStaticFrictionAttr().Set(float(p["mu_s"])); pa.GetDynamicFrictionAttr().Set(float(p["mu_d"])); pa.GetRestitutionAttr().Set(0.05)
    UsdShade.MaterialBindingAPI.Apply(col).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
    stage.GetRootLayer().Save()
    return pts, fi


cfg = json.loads(Path(A["config"]).read_text())
out = Path(A["out"]); out.mkdir(parents=True, exist_ok=True)
guest_out = A.get("guest_out", str(out))
objs = []
for p in cfg["objects"]["geometry"]:
    path = out / f"{p['id']}.usd"
    pts, fi = write(path, p)
    rec = dict(p, usd_guest=f"{guest_out}/{p['id']}.usd", n_points=int(len(pts)))
    if p.get("shape") == "bottle":
        rec["mass_props"] = mass_props(p)
    if p["id"] == "cup_ref":   # same geometry as the validated asset?
        repo = Path(os.environ.get("OMX_REPO", "."))
        ref = Usd.Stage.Open(str(repo / "assets/paper_cup/paper_cup.usd"))
        rm = UsdGeom.Mesh(ref.GetPrimAtPath("/PaperCup/Collision"))
        rp = np.array(rm.GetPointsAttr().Get()); rfi = np.array(rm.GetFaceVertexIndicesAttr().Get())
        mass = UsdPhysics.MassAPI(ref.GetPrimAtPath("/PaperCup")).GetMassAttr().Get()
        n_out = 9 * 48   # outer-wall rings (n_rings + 1) x n_sides: the shape; the asset has no bottom (see shell_mesh.py)
        rec["ref_check"] = {"outer_wall_points_max_abs_diff_m": float(np.abs(rp[:n_out] - pts[:n_out]).max()) if len(rp) >= n_out else None,
                            "ref_has_bottom": bool((np.hypot(rp[rp[:, 2] < 1e-6][:, 0], rp[rp[:, 2] < 1e-6][:, 1]) < 1e-3).any()),
                            "this_has_bottom": bool((np.hypot(pts[pts[:, 2] < 1e-6][:, 0], pts[pts[:, 2] < 1e-6][:, 1]) < 1e-3).any()),
                            "ref_mass": mass}
        print("cup_ref vs assets/paper_cup/paper_cup.usd:", rec["ref_check"], flush=True)
    objs.append(rec); print("wrote", path, flush=True)
json.dump({"objects": objs}, open(out / "objects.json", "w"), indent=1)
app.close()

"""Truncated-cone shells with a real bottom: the DR cups (sim/dr_objects.py) and the bin (sim/physics_replay.py,
sim/dr_render.py). numpy only for the mesh; define_bin() needs pxr.

Why a separate module (2026-11-01): the cup mesh of sim/gen_paper_cup_usd.py (2026-09-23) and of the first
dr_objects.py was open at the bottom -- its "closed base" was a 1 mm ring joining the outer and inner wall at z = 0,
nothing across the middle (a tube). Here the inner wall starts `bottom` above z = 0 and both ends are closed by a disc
(outer underside at z = 0, inner floor at z = bottom), so the shell is watertight. A real paper cup's bottom panel is
recessed a few mm inside a skirt; this is a flat bottom (the cup still stands on its outer rim circle).
"""
import math

import numpy as np

INSIDE_V = (0.97, 1.0)   # uv strip for everything that is not the outer wall (inner wall, rim, bottom)


def cup_mesh(top_r, bot_r, h, wall, bottom=None, n_sides=48, n_rings=8):
    """Points, face counts, face indices and face-varying uv of a closed-bottom shell. Origin = centre of the
    underside, +Z up. Outer wall: u around, v 0..0.96 up; the rest maps to the INSIDE_V strip."""
    bottom = wall if bottom is None else bottom
    taper = math.atan2(top_r - bot_r, h)
    off = wall / math.cos(taper) if abs(taper) > 1e-6 else wall
    r_in = lambda z: max(bot_r + (top_r - bot_r) * z / h - off, 0.001)   # noqa: E731
    pts = []
    for i in range(n_rings + 1):       # outer wall rings, z 0..h
        z = h * i / n_rings; r = bot_r + (top_r - bot_r) * i / n_rings
        pts += [[r * math.cos(2 * math.pi * j / n_sides), r * math.sin(2 * math.pi * j / n_sides), z] for j in range(n_sides)]
    for i in range(n_rings + 1):       # inner wall rings, z bottom..h
        z = bottom + (h - bottom) * i / n_rings; r = r_in(z)
        pts += [[r * math.cos(2 * math.pi * j / n_sides), r * math.sin(2 * math.pi * j / n_sides), z] for j in range(n_sides)]
    inner = (n_rings + 1) * n_sides
    c_out, c_in = len(pts), len(pts) + 1
    pts += [[0.0, 0.0, 0.0], [0.0, 0.0, bottom]]
    pts = np.array(pts, np.float32)
    fc, fi, uv = [], [], []
    vin = (0.5, sum(INSIDE_V) / 2)

    def poly(idx, uvs):
        fc.append(len(idx)); fi.extend(idx); uv.extend(uvs)

    vo = lambda i: 0.96 * i / n_rings   # noqa: E731
    for i in range(n_rings):           # outer wall (normals out)
        for j in range(n_sides):
            j1 = (j + 1) % n_sides; u0, u1 = j / n_sides, (j + 1) / n_sides
            a, b, c, d = i * n_sides + j, i * n_sides + j1, (i + 1) * n_sides + j1, (i + 1) * n_sides + j
            poly([a, b, c], [(u0, vo(i)), (u1, vo(i)), (u1, vo(i + 1))]); poly([a, c, d], [(u0, vo(i)), (u1, vo(i + 1)), (u0, vo(i + 1))])
    for i in range(n_rings):           # inner wall (normals in, toward the axis)
        for j in range(n_sides):
            j1 = (j + 1) % n_sides
            a, b, c, d = (inner + i * n_sides + j, inner + i * n_sides + j1, inner + (i + 1) * n_sides + j1, inner + (i + 1) * n_sides + j)
            poly([a, d, c], [vin] * 3); poly([a, c, b], [vin] * 3)
    ot, it = n_rings * n_sides, inner + n_rings * n_sides
    for j in range(n_sides):           # rim
        j1 = (j + 1) % n_sides
        poly([ot + j, ot + j1, it + j1], [vin] * 3); poly([ot + j, it + j1, it + j], [vin] * 3)
    for j in range(n_sides):           # underside disc at z = 0 (normal down)
        poly([c_out, (j + 1) % n_sides, j], [vin] * 3)
    for j in range(n_sides):           # inner floor at z = bottom (normal up)
        poly([c_in, inner + j, inner + (j + 1) % n_sides], [vin] * 3)
    return pts, np.array(fc, np.int32), np.array(fi, np.int32), np.array(uv, np.float32)


def check_closed(fi_tri):
    """Every edge shared by exactly two triangles (watertight). Returns the number of edges that are not."""
    from collections import Counter
    t = np.asarray(fi_tri).reshape(-1, 3)
    e = Counter(tuple(sorted(p)) for tr in t for p in ((tr[0], tr[1]), (tr[1], tr[2]), (tr[2], tr[0])))
    return sum(1 for v in e.values() if v != 2)


def define_bin(stage, path, center_xyz, base_dia, top_dia, height, wall=0.003, floor=0.01, stand=True, color=None):
    """Static bin (open truncated cone, closed bottom) with its underside at center_xyz, and optionally a box stand
    from z = 0 up to it. Colliders: the bin is a triangle mesh (approximation none, static), the stand a solid cube --
    the stand is what stops a falling object for sure (a thin floor can be tunnelled at 120 Hz)."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt
    x, y, z = map(float, center_xyz)
    pts, fc, fi, uv = cup_mesh(top_dia / 2, base_dia / 2, height, wall, bottom=floor)
    xf = UsdGeom.Xform.Define(stage, path); xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
    m = UsdGeom.Mesh.Define(stage, f"{path}/Mesh")
    m.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, q)) for q in pts]))
    m.GetFaceVertexCountsAttr().Set(Vt.IntArray(fc.tolist())); m.GetFaceVertexIndicesAttr().Set(Vt.IntArray(fi.tolist()))
    m.GetSubdivisionSchemeAttr().Set("none")
    st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(float(a), float(b)) for a, b in uv]))
    m.CreateDisplayColorAttr([tuple(color or (0.85, 0.85, 0.85))])
    UsdPhysics.CollisionAPI.Apply(m.GetPrim()); UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).GetApproximationAttr().Set("none")
    out = {"bin": path, "center_xyz": [x, y, z], "base_dia": base_dia, "top_dia": top_dia, "height": height, "wall": wall, "floor": floor}
    if stand and z > 1e-4:
        s = UsdGeom.Cube.Define(stage, f"{path}_Stand"); s.CreateSizeAttr(1.0)
        side = top_dia + 0.02
        s.AddTranslateOp().Set(Gf.Vec3d(x, y, z / 2)); s.AddScaleOp().Set(Gf.Vec3f(side, side, z))
        s.CreateDisplayColorAttr([(0.3, 0.3, 0.32)]); UsdPhysics.CollisionAPI.Apply(s.GetPrim())
        out["stand"] = f"{path}_Stand"; out["stand_size"] = [side, side, z]
    return out


def in_bin(xyz, b):
    """Is a point (an object's origin = its base centre) inside the bin described by define_bin()'s dict?"""
    x, y, z = xyz; cx, cy, cz = b["center_xyz"]
    t = (z - cz) / b["height"]
    if not (-0.01 <= t <= 1.0):
        return False
    r = (b["base_dia"] + (b["top_dia"] - b["base_dia"]) * min(max(t, 0.0), 1.0)) / 2
    return math.hypot(x - cx, y - cy) <= r


if __name__ == "__main__":   # self-test: both shells watertight
    for args in ((0.0375, 0.025, 0.093, 0.001), (0.10, 0.075, 0.22, 0.003, 0.01)):
        p, c, f, _ = cup_mesh(*args)
        assert (c == 3).all()
        print(args, "points", len(p), "tris", len(c), "open edges", check_closed(f), "z range", p[:, 2].min(), p[:, 2].max())

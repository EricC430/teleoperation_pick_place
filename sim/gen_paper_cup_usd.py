#!/usr/bin/env python3
"""Generate a hollow thin-walled paper-cup USD asset for Isaac Sim.

The real cup (`[Eric說 2026-09-21]`):
  - Opening diameter: 7.5 cm
  - Base diameter: 5.0 cm
  - Height: 9.5 cm
  - Standing UPRIGHT

This generates a truncated cone with thin walls (~1 mm), suitable for
one-finger-inside / one-finger-outside rim grasping.

The USD is authored with metersPerUnit=1.0 (same as Isaac Sim's world stage),
so NO rescaling is needed when referencing it.

Usage (inside the isaac-lab container):
    sim/run_in_container.sh gen_paper_cup_usd.py \\
        --out /workspace/test_isaaclab/assets/paper_cup/paper_cup.usd --headless
"""
from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", required=True, help="output USD path")
parser.add_argument("--top-dia", type=float, default=0.075, help="top opening diameter (m)")
parser.add_argument("--bot-dia", type=float, default=0.050, help="bottom diameter (m)")
parser.add_argument("--cup-height", type=float, default=0.095, help="cup height (m)")
parser.add_argument("--wall", type=float, default=0.001, help="wall thickness (m)")
parser.add_argument("--sides", type=int, default=48, help="circumferential segments")
parser.add_argument("--rings", type=int, default=8, help="vertical segments")
parser.add_argument("--mass", type=float, default=0.012, help="mass (kg)")
parser.add_argument("--mu-s", type=float, default=1.2, help="static friction")
parser.add_argument("--mu-d", type=float, default=0.9, help="dynamic friction")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


def make_truncated_cone_mesh(
    top_outer_r: float,
    bot_outer_r: float,
    height: float,
    wall_thickness: float,
    n_sides: int = 48,
    n_rings: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (points, face_vertex_counts, face_vertex_indices) for a hollow
    truncated-cone shell.

    The mesh has:
      - outer wall
      - inner wall (inset by wall_thickness along the surface normal)
      - top rim ring (connecting outer top to inner top)
      - bottom ring (connecting outer bottom to inner bottom — closed base)

    Origin: centre of the BASE circle, +Z up.
    """
    points = []

    # Taper angle for inner radius calculation
    taper_angle = math.atan2(top_outer_r - bot_outer_r, height)
    inner_offset = wall_thickness / math.cos(taper_angle) if abs(taper_angle) > 1e-6 else wall_thickness
    top_inner_r = max(top_outer_r - inner_offset, 0.001)
    bot_inner_r = max(bot_outer_r - inner_offset, 0.001)

    # Outer surface rings (bottom to top)
    for i in range(n_rings + 1):
        t = i / n_rings
        r = bot_outer_r + t * (top_outer_r - bot_outer_r)
        z = t * height
        for j in range(n_sides):
            theta = 2.0 * math.pi * j / n_sides
            points.append([r * math.cos(theta), r * math.sin(theta), z])

    # Inner surface rings (bottom to top)
    for i in range(n_rings + 1):
        t = i / n_rings
        r = bot_inner_r + t * (top_inner_r - bot_inner_r)
        z = t * height
        for j in range(n_sides):
            theta = 2.0 * math.pi * j / n_sides
            points.append([r * math.cos(theta), r * math.sin(theta), z])

    points = np.array(points, dtype=np.float32)
    n_inner_ring_start = (n_rings + 1) * n_sides

    face_counts = []
    face_indices = []

    def quad(a, b, c, d):
        """Emit a quad as two triangles (CCW winding)."""
        face_counts.extend([3, 3])
        face_indices.extend([a, b, c, a, c, d])

    # Outer wall (quads, CCW when viewed from OUTSIDE)
    for i in range(n_rings):
        for j in range(n_sides):
            j1 = (j + 1) % n_sides
            a = i * n_sides + j
            b = i * n_sides + j1
            c = (i + 1) * n_sides + j1
            d = (i + 1) * n_sides + j
            quad(a, b, c, d)

    # Inner wall (quads, CCW when viewed from INSIDE = reverse winding)
    for i in range(n_rings):
        for j in range(n_sides):
            j1 = (j + 1) % n_sides
            a = n_inner_ring_start + i * n_sides + j
            b = n_inner_ring_start + i * n_sides + j1
            c = n_inner_ring_start + (i + 1) * n_sides + j1
            d = n_inner_ring_start + (i + 1) * n_sides + j
            quad(a, d, c, b)  # reversed winding for inner

    # Top rim ring (connecting outer top to inner top)
    outer_top = n_rings * n_sides
    inner_top = n_inner_ring_start + n_rings * n_sides
    for j in range(n_sides):
        j1 = (j + 1) % n_sides
        a = outer_top + j
        b = outer_top + j1
        c = inner_top + j1
        d = inner_top + j
        quad(a, b, c, d)

    # Bottom ring (connecting outer bottom to inner bottom — closed base)
    for j in range(n_sides):
        j1 = (j + 1) % n_sides
        a = j
        b = j1
        c = n_inner_ring_start + j1
        d = n_inner_ring_start + j
        quad(a, d, c, b)  # normal points down

    return points, np.array(face_counts, dtype=np.int32), np.array(face_indices, dtype=np.int32)


print(f"Paper cup mesh: top ⌀{args.top_dia*100:.1f}cm, bot ⌀{args.bot_dia*100:.1f}cm, "
      f"h={args.cup_height*100:.1f}cm, wall={args.wall*1000:.1f}mm")

pts, fc, fi = make_truncated_cone_mesh(
    top_outer_r=args.top_dia / 2.0,
    bot_outer_r=args.bot_dia / 2.0,
    height=args.cup_height,
    wall_thickness=args.wall,
    n_sides=args.sides,
    n_rings=args.rings,
)
print(f"  {len(pts)} vertices, {len(fc)} faces, {len(fi)} indices")

# Write USD
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
stage = Usd.Stage.CreateNew(args.out)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

# Root xform
root = UsdGeom.Xform.Define(stage, "/PaperCup")
root.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
stage.SetDefaultPrim(root.GetPrim())

# Visual mesh
vec3f_points = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts]
mesh = UsdGeom.Mesh.Define(stage, "/PaperCup/Mesh")
mesh.GetPointsAttr().Set(vec3f_points)
mesh.GetFaceVertexCountsAttr().Set([int(x) for x in fc])
mesh.GetFaceVertexIndicesAttr().Set([int(x) for x in fi])
mesh.GetSubdivisionSchemeAttr().Set("none")

# Visual material (paper cup appearance: warm off-white with slight brown tint)
mat = UsdShade.Material.Define(stage, "/PaperCup/PaperMaterial")
shader = UsdShade.Shader.Define(stage, "/PaperCup/PaperMaterial/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.92, 0.88, 0.80))
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
UsdShade.MaterialBindingAPI(mesh).Bind(mat)

# Collision mesh (identical geometry, convex decomposition for thin walls)
collision = UsdGeom.Mesh.Define(stage, "/PaperCup/Collision")
collision.GetPointsAttr().Set(vec3f_points)
collision.GetFaceVertexCountsAttr().Set([int(x) for x in fc])
collision.GetFaceVertexIndicesAttr().Set([int(x) for x in fi])
collision.GetSubdivisionSchemeAttr().Set("none")
collision.GetVisibilityAttr().Set("invisible")

# Physics: rigid body on root
UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())

# Physics: collision on collision mesh with convex decomposition
UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim())
mesh_collision.GetApproximationAttr().Set("convexDecomposition")

# Mass
mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
mass_api.GetMassAttr().Set(args.mass)

# Physics material (friction for paper-on-rubber contact)
phys_mat = UsdShade.Material.Define(stage, "/PaperCup/PhysicsMaterial")
phys_mat_api = UsdPhysics.MaterialAPI.Apply(phys_mat.GetPrim())
phys_mat_api.GetStaticFrictionAttr().Set(args.mu_s)
phys_mat_api.GetDynamicFrictionAttr().Set(args.mu_d)
phys_mat_api.GetRestitutionAttr().Set(0.05)  # paper cup, almost no bounce

# Bind physics material to collision
phys_bind = UsdShade.MaterialBindingAPI.Apply(collision.GetPrim())
phys_bind.Bind(phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")

stage.GetRootLayer().Save()
print(f"✅ Wrote {args.out}")
print(f"   {len(pts)} vertices, {len(fc)} faces")
print(f"   mass={args.mass*1000:.0f}g, μ_s={args.mu_s:.1f}, μ_d={args.mu_d:.1f}")

os._exit(0)

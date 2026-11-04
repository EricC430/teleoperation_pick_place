"""Dump every collision mesh of the robot asset in its LINK frame -> npz, for sim/penetration.py (2026-11-01).

    /isaac-sim/python.sh sim/extract_link_meshes.py <asset.usda> <out.npz>

Keys: <link>__<i>__v (vertices, link frame, m) and <link>__<i>__f (triangles). The colliders are the asset's
`.../collisions/<name>/node_STL_BINARY_` prims (CollisionAPI on an Xform) with a `mesh` child; the same meshes the
physics replay turns into SDFs.
"""
import sys, numpy as np
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdGeom, UsdPhysics
asset, out = sys.argv[1], sys.argv[2]
st = Usd.Stage.Open(asset); cache = UsdGeom.XformCache()
res = {}
for p in st.Traverse(Usd.TraverseInstanceProxies()):
    if not p.IsA(UsdGeom.Mesh):
        continue
    a = p
    while a and not a.HasAPI(UsdPhysics.CollisionAPI):
        a = a.GetParent()
    if not a:
        continue
    q = p
    while q and not q.HasAPI(UsdPhysics.RigidBodyAPI):
        q = q.GetParent()
    if not q:
        continue
    link = q.GetName()
    M = np.array(cache.GetLocalToWorldTransform(p)).T; L = np.array(cache.GetLocalToWorldTransform(q)).T
    T = np.linalg.inv(L) @ M
    m = UsdGeom.Mesh(p); pts = np.array(m.GetPointsAttr().Get(), float)
    fc = np.array(m.GetFaceVertexCountsAttr().Get()); fi = np.array(m.GetFaceVertexIndicesAttr().Get())
    tris, k = [], 0
    for c in fc:
        for j in range(1, c - 1):
            tris.append([fi[k], fi[k + j], fi[k + j + 1]])
        k += c
    v = (T[:3, :3] @ pts.T).T + T[:3, 3]
    i = len(res)
    res[f"{link}__{i}__v"] = v; res[f"{link}__{i}__f"] = np.array(tris)
    print(link, p.GetPath(), len(v), "verts", "approx", UsdPhysics.MeshCollisionAPI(a).GetApproximationAttr().Get() if a.HasAPI(UsdPhysics.MeshCollisionAPI) else None,
          "bbox", v.min(0).round(4), v.max(0).round(4), flush=True)
np.savez(out, **res)
app.close()

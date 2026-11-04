"""Configuration report of the physics-replay scene, and a check of it against an expected snapshot.

Used by sim/physics_replay.py (config_report=<expected json|none>). What it reads is the scene as Isaac will simulate it
(authored USD after every override in physics_replay.py, plus the articulation's runtime values), not the command line,
so a default that changed somewhere else still shows up.

The check has two parts:
1. Sanity rules that hold for any run (gravity, step consistency, masses, drives, cameras).
2. If an expected file is given: every leaf of its "snapshot" must match the report (numbers to rel 1e-4 / abs 1e-7,
   strings and bools exactly), except keys under its "ignore" prefixes. sim/dr_expected_config.json holds the snapshot
   of the configuration the 10/14 closed loop was validated with (docs/specs/S9_sim_dr_collect.md).
Isaac-free on import (pxr and numpy only when called), so the check can also be run on a saved report:
    python3 sim/config_report.py <config_report.json> <expected.json>
"""
import hashlib
import json
import math
import sys
from pathlib import Path


def _sha(p):
    p = Path(p)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def _val(v):
    """USD / numpy / Gf values -> JSON."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if hasattr(v, "numpy"):
        v = v.numpy()
    if hasattr(v, "tolist"):
        return v.tolist()
    try:
        return [_val(x) for x in v]
    except TypeError:
        return str(v)


def _authored(prim, prefixes):
    out = {}
    for a in prim.GetAttributes():
        n = a.GetName()
        if any(n.startswith(p) for p in prefixes) and a.HasAuthoredValue():
            out[n] = _val(a.Get())
    return out


def build(stage, robot_prim, cup_prim, table_prim, scene_prim, robot, physics_dt, fps, args, repo, inputs, cameras, render_api):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    import numpy as np

    def runtime(name):
        try:
            v = getattr(robot, name)()
            if isinstance(v, tuple):
                return [_val(x[0] if hasattr(x, "__len__") and len(x) == 1 else x) for x in v]
            v = _val(v)
            return v[0] if isinstance(v, list) and len(v) == 1 else v
        except Exception as e:   # not every Isaac version has every getter
            return f"unavailable: {type(e).__name__}"

    rep = {"isaac_sim": _isaac_version(), "render_api": render_api, "fps": fps}
    sc = UsdPhysics.Scene(scene_prim)
    rep["physics"] = {"gravity_direction": _val(sc.GetGravityDirectionAttr().Get()), "gravity_magnitude": _val(sc.GetGravityMagnitudeAttr().Get()),
                      "physics_dt": float(physics_dt), "substeps_per_frame": int(round(1.0 / fps / physics_dt)),
                      "scene_attrs": _authored(scene_prim, ("physxScene:",))}
    joints, bodies, colliders, materials = {}, {}, {}, {}
    for root, tag in ((robot_prim, "robot"), (cup_prim, "cup"), (table_prim, "table")):
        for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):   # the robot's colliders sit in instances
            key = f"{tag}:{p.GetName()}" if tag != "robot" else p.GetName()
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                rep["articulation"] = {"root": str(p.GetPath()), "attrs": _authored(p, ("physxArticulation:",))}
            if p.IsA(UsdPhysics.Joint):
                j = {"type": p.GetTypeName(), "attrs": _authored(p, ("physics:lowerLimit", "physics:upperLimit", "physics:axis", "drive:", "physxJoint:",
                                                                     "physxMimicJoint:", "physics:jointEnabled", "physics:excludeFromArticulation"))}
                joints[p.GetName()] = j
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies[key] = {"mass": _val(UsdPhysics.MassAPI(p).GetMassAttr().Get()) if p.HasAPI(UsdPhysics.MassAPI) else None,
                               "attrs": _authored(p, ("physics:rigidBodyEnabled", "physics:kinematicEnabled", "physxRigidBody:", "physics:centerOfMass",
                                                      "physics:diagonalInertia"))}
            if p.HasAPI(UsdPhysics.CollisionAPI):
                c = {"approximation": _val(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get()) if p.HasAPI(UsdPhysics.MeshCollisionAPI) else "primitive",
                     "attrs": _authored(p, ("physxCollision:", "physxSDFMeshCollision:", "physxConvexDecompositionCollision:", "physics:collisionEnabled"))}
                m, _rel = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial("physics")
                if m:
                    c["physics_material"] = str(m.GetPath())
                    materials[str(m.GetPath())] = _authored(m.GetPrim(), ("physics:staticFriction", "physics:dynamicFriction", "physics:restitution",
                                                                           "physics:density", "physxMaterial:"))
                colliders[str(p.GetPath()).replace(str(root.GetPath()), tag)] = c
    rep["joints"], rep["bodies"], rep["colliders"], rep["physics_materials"] = joints, bodies, colliders, materials
    rep["runtime"] = {"dof_names": list(robot.dof_names), **{n: runtime(n) for n in (
        "get_dof_max_efforts", "get_dof_max_velocities", "get_dof_armatures", "get_dof_gains", "get_dof_friction_properties", "get_link_masses",
        "get_dof_limits")}}
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(cup_prim).ComputeAlignedRange()
    rep["cup"] = {"bbox_size_m": [round(float(v), 5) for v in bb.GetSize()], "bbox_min_m": [round(float(v), 5) for v in bb.GetMin()]}
    rep["cameras"] = cameras
    rep["inputs"] = {k: {"path": str(v), "sha256_16": _sha(v)} for k, v in inputs.items()}
    rep["args"] = dict(args)
    return rep


def _isaac_version():
    for p in ("/isaac-sim/VERSION",):
        try:
            return Path(p).read_text().strip()[:40]
        except OSError:
            pass
    try:
        import isaacsim
        return getattr(isaacsim, "__version__", "unknown")
    except ImportError:
        return "unknown"


def flatten(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{pre}{k}." if pre or k else k))
    elif isinstance(d, list) and d and all(isinstance(x, (int, float)) for x in d):
        out[pre.rstrip(".")] = d
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flatten(v, f"{pre}{i}."))
    else:
        out[pre.rstrip(".")] = d
    return out


def _close(a, b):
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-4, abs_tol=1e-7)
    except (TypeError, ValueError):
        return a == b


def check(report, expected_path=None, quiet=False):
    bad = []
    ph = report["physics"]
    g = ph["gravity_direction"]; gm = ph["gravity_magnitude"]
    if not (abs(gm - 9.81) < 0.02 and g[2] < -0.999):
        bad.append(f"gravity {gm} dir {g}")
    if abs(ph["substeps_per_frame"] * ph["physics_dt"] * report["fps"] - 1.0) > 1e-3:
        bad.append(f"physics_dt {ph['physics_dt']} x {ph['substeps_per_frame']} substeps != 1/{report['fps']} s")
    for k, b in report["bodies"].items():
        m = b["mass"]
        if k == "world":   # the URDF's fixed root link: mass 0 in the asset (PhysX makes it 1 kg), welded to the world -> no effect
            continue
        if m is not None and not (1e-5 <= m <= 5.0):
            bad.append(f"body {k} mass {m} kg")
    cupm = [b["mass"] for k, b in report["bodies"].items() if k.startswith("cup:")]
    if not cupm or not all(m and 0.002 <= m <= 0.5 for m in cupm):
        bad.append(f"cup mass {cupm}")
    for n, j in report["joints"].items():
        mf = j["attrs"].get("drive:angular:physics:maxForce")
        if mf is not None and mf <= 0 and not n.startswith("gripper_joint_2"):
            bad.append(f"joint {n} drive maxForce {mf}")
    for cn, c in report["cameras"].items():
        K = c["K"]
        if not (K[0][0] > 0 and K[1][1] > 0):
            bad.append(f"camera {cn} K {K}")
    art = report.get("articulation", {}).get("attrs", {})
    want_sc = report["args"].get("selfcol", "1") == "1"
    if art.get("physxArticulation:enabledSelfCollisions") is not want_sc:
        bad.append(f"self-collisions {art.get('physxArticulation:enabledSelfCollisions')} but selfcol={report['args'].get('selfcol', '1')}")
    if expected_path:
        exp = json.loads(Path(expected_path).read_text())
        ign = tuple(exp.get("ignore", []))
        got = flatten(report)
        for k, v in flatten(exp["snapshot"]).items():
            if k.startswith(ign):
                continue
            if k not in got:
                bad.append(f"missing {k} (expected {v})")
            elif not _close(got[k], v):
                bad.append(f"{k}: {got[k]} != expected {v}")
        extra = [k for k in got if k not in flatten(exp["snapshot"]) and not k.startswith(ign)]
        if extra and not quiet:
            print(f"CONFIG note: {len(extra)} keys not in the expected snapshot, e.g. {extra[:5]}", flush=True)
    for b in bad:
        print("CONFIG MISMATCH", b, flush=True)
    report["check"] = {"expected": str(expected_path) if expected_path else None, "mismatches": bad, "pass": not bad}
    return not bad


IGNORE_DEFAULT = ["args.", "inputs.cands.", "inputs.cup_asset.path", "inputs.asset.path", "inputs.intr.path", "inputs.ext.path",
                  "inputs.wrist_", "cup.bbox_min_m", "check.", "isaac_sim", "render_api", "runtime.get_dof_gains",
                  "runtime.get_dof_friction_properties", "runtime.get_dof_limits", "cameras.wrist"]


def make_expected(report_path, out_path, note):
    r = json.loads(Path(report_path).read_text()); r.pop("check", None)
    Path(out_path).write_text(json.dumps({"_doc": note, "made_from": str(report_path), "ignore": IGNORE_DEFAULT, "snapshot": r}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "make":   # python3 sim/config_report.py make <config_report.json> <expected.json> "<note>"
        make_expected(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")
        sys.exit(0)
    r = json.loads(Path(sys.argv[1]).read_text())
    ok = check(r, sys.argv[2] if len(sys.argv) > 2 else None)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

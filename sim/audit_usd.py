"""Audit a robot USD against `sim/omx_constants.py`. Read-only.

Prints the six things the URDF importer gets wrong (S4 §2-2 / D029 §URDF->USD), for any USD:

  1. joint position limits      2. drive maxForce        3. max joint velocity
  4. drive stiffness / damping  5. collision approximation per collider
  6. self-collision flag, plus mimic-joint survival and link mass/inertia

Exit code 0 = every joint matches the constants. 1 = at least one mismatch.
Run inside the isaac-lab container (see `sim/run_in_container.sh`).
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--usd", required=True, help="USD file to audit")
parser.add_argument("--no-site-limits", action="store_true", help="compare against factory travel only")
parser.add_argument("--tol", type=float, default=0.5, help="degrees of slack when comparing limits")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omx_constants as K  # noqa: E402

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402

RAD2DEG = 180.0 / 3.141592653589793
APPLY_SITE = not args.no_site_limits

# 🔴 UNITS. UsdPhysics angular drives store stiffness/damping PER DEGREE, while
#    omx_constants computes them per radian (the SI convention the motor datasheets use).
#    The URDF importer does the conversion for us, so the audit has to divide by 180/pi
#    before comparing. Getting this wrong makes a correct asset look 57x off — it did,
#    on the first run of this script.
expected = {}
for j in K.JOINTS:
    lo, hi = K.effective_limits_rad(j, apply_site_limits=APPLY_SITE)
    expected[j.urdf_name] = {
        "lo": lo * RAD2DEG,
        "hi": hi * RAD2DEG,
        "force": j.motor.stall_torque_nm,
        "vel": j.motor.velocity_limit_rad_s * RAD2DEG,
        "stiff": K.stiffness(j) / RAD2DEG,   # per-degree, USD convention
        "damp": K.damping(j) / RAD2DEG,      # per-degree, USD convention
    }

stage = Usd.Stage.Open(os.path.abspath(args.usd))
if stage is None:
    raise SystemExit(f"cannot open {args.usd}")

print("=" * 112)
print(f"AUDIT  {args.usd}")
print(f"        compared against sim/omx_constants.py  (site limits applied: {APPLY_SITE})")
print("=" * 112)

# ---------------------------------------------------------------- joints
# 🔴 Isaac Lab writes INSTANCEABLE assets: the physics prims live inside prototypes and a
#    plain stage.Traverse() walks straight past them. Everything below must traverse instance
#    proxies, or a perfectly good asset audits as "no colliders, no mimic".
def walk(stage):
    return iter(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))


joints, mimics = {}, []
for prim in walk(stage):
    if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
        j = UsdPhysics.RevoluteJoint(prim) if prim.IsA(UsdPhysics.RevoluteJoint) else None
        drive = UsdPhysics.DriveAPI.Get(prim, "angular") or UsdPhysics.DriveAPI.Get(prim, "linear")
        physx = PhysxSchema.PhysxJointAPI.Get(stage, prim.GetPath())
        joints[prim.GetName()] = {
            "path": str(prim.GetPath()),
            "lo": j.GetLowerLimitAttr().Get() if j else None,
            "hi": j.GetUpperLimitAttr().Get() if j else None,
            "force": drive.GetMaxForceAttr().Get() if drive else None,
            "stiff": drive.GetStiffnessAttr().Get() if drive else None,
            "damp": drive.GetDampingAttr().Get() if drive else None,
            "vel": physx.GetMaxJointVelocityAttr().Get() if physx else None,
        }
    for schema in prim.GetAppliedSchemas():
        if "Mimic" in schema:
            mimics.append((prim.GetName(), schema))

fails = []
print("\n--- 1-4. joints " + "-" * 96)
hdr = f"{'joint':<18}{'limits (deg)':<26}{'maxForce':<20}{'maxVel deg/s':<20}{'stiff/deg':<14}{'damp/deg'}"
print(hdr)
for name, exp in expected.items():
    got = joints.get(name)
    if got is None:
        print(f"{name:<18}🔴 NOT PRESENT IN THIS USD")
        fails.append(f"{name}: missing")
        continue

    def cmp(got_v, exp_v, tol, label, fmt="{:.2f}"):
        if got_v is None:
            fails.append(f"{name}.{label}: unset")
            return "🔴 unset"
        ok = abs(float(got_v) - exp_v) <= tol
        if not ok:
            fails.append(f"{name}.{label}: {float(got_v):.3f} != {exp_v:.3f}")
        return ("" if ok else "🔴 ") + fmt.format(float(got_v))

    lim = f"{cmp(got['lo'], exp['lo'], args.tol, 'lower', '{:.1f}')} .. {cmp(got['hi'], exp['hi'], args.tol, 'upper', '{:.1f}')}"
    print(
        f"{name:<18}{lim:<26}"
        f"{cmp(got['force'], exp['force'], 0.01, 'maxForce', '{:.3f}'):<20}"
        f"{cmp(got['vel'], exp['vel'], 1.0, 'maxVel', '{:.1f}'):<20}"
        f"{cmp(got['stiff'], exp['stiff'], 0.005, 'stiffness', '{:.4f}'):<14}"
        f"{cmp(got['damp'], exp['damp'], 0.005, 'damping', '{:.4f}')}"
    )

extra = sorted(set(joints) - set(expected))
if extra:
    print(f"\nother joints present (not in the constants table): {extra}")

# ---------------------------------------------------------------- mimic
print("\n--- 5. mimic joint " + "-" * 93)
follower, master, mult = K.MIMIC_JOINT
if follower in joints:
    if mimics:
        print(f"✅ {follower} present, mimic schema found: {mimics}")
    else:
        print(f"🔴 {follower} is present but carries NO mimic schema —")
        print(f"   it is a FREE joint. The second finger will not follow {master} (x{mult}).")
        fails.append("mimic: gripper_joint_2 is free")
else:
    print(f"🔴 {follower} not found at all — was it merged away?")
    fails.append("mimic: gripper_joint_2 missing")

# ---------------------------------------------------------------- colliders
print("\n--- 6. collision approximation " + "-" * 81)
approx = {}
for prim in walk(stage):
    mesh_api = PhysxSchema.PhysxCollisionAPI.Get(stage, prim.GetPath())
    if UsdPhysics.MeshCollisionAPI.Get(stage, prim.GetPath()):
        a = UsdPhysics.MeshCollisionAPI.Get(stage, prim.GetPath()).GetApproximationAttr().Get()
        approx.setdefault(str(a), []).append(prim.GetName())
    elif mesh_api:
        approx.setdefault("(collision, no MeshCollisionAPI)", []).append(prim.GetName())
if approx:
    for a, names in sorted(approx.items()):
        flag = "🔴 " if a == "convexHull" else ""
        print(f"{flag}{a:<34}{len(names):>4} collider(s)   e.g. {names[:4]}")
    if "convexHull" in approx:
        print("   🔴 convexHull on the gripper fingers means the gap between them is filled solid.")
        fails.append("colliders: convexHull present")
else:
    print("no mesh colliders found (check that collision geometry was imported at all)")

# ---------------------------------------------------------------- articulation / self-collision
print("\n--- 7. articulation root & self-collision " + "-" * 70)
found_root = False
for prim in walk(stage):
    root = PhysxSchema.PhysxArticulationAPI.Get(stage, prim.GetPath())
    if UsdPhysics.ArticulationRootAPI.Get(stage, prim.GetPath()):
        found_root = True
        sc = root.GetEnabledSelfCollisionsAttr().Get() if root else None
        print(f"articulation root: {prim.GetPath()}   self-collision = {sc}")
if not found_root:
    print("🔴 no ArticulationRootAPI found — this USD is not an articulation")
    fails.append("no articulation root")

# ---------------------------------------------------------------- mass / inertia
print("\n--- 8. link mass (the URDF's inertials are real; check they survived) " + "-" * 42)
masses = []
for prim in walk(stage):
    m = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
    if m:
        val = m.GetMassAttr().Get()
        if val:
            masses.append((prim.GetName(), val))
for n, v in masses[:12]:
    print(f"  {n:<28}{v:.6f} kg")
if masses:
    print(f"  total over {len(masses)} bodies: {sum(v for _, v in masses):.4f} kg   (ROBOTIS quotes 560 g for OMX-F)")
else:
    print("🔴 no MassAPI found — inertials did not survive the import")
    fails.append("no masses")

print("\n" + "=" * 112)
if fails:
    print(f"RESULT: 🔴 {len(fails)} mismatch(es)")
    for f in fails:
        print(f"  - {f}")
    import os
    os._exit(1)
print("RESULT: ✅ this USD matches sim/omx_constants.py")
print("=" * 112)
import os
os._exit(0)

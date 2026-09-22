"""Read-only: print the actual joint Axis and PhysxMimicJointAPI attributes for the gripper.

No cameras, no physics stepping -- just opens the stage and reads prims. This exists because
`verify_mimic_gearing.py` (which DOES need `--enable_cameras` for the visual check) takes an
Isaac Sim RTX shutdown that can hang for tens of minutes in this container, while this diagnostic
answers a narrower question in seconds: does `convert_omx_urdf.py`'s
`PhysxMimicJointAPI.Apply(follower, "rotZ")` / `CreateReferenceJointAxisAttr().Set("rotX")` call
(both gripper joints rotate about URDF axis (0,0,1) -- see assets/omx_f/omx_f.urdf) actually
reference a consistent axis on both joints, or is that a rotX/rotZ mismatch?

Run inside the isaac-lab container (no --enable_cameras needed):
    ./sim/run_in_container.sh inspect_mimic_axis.py --usd <path to omx_f.usd> --headless
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--usd", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402

stage = Usd.Stage.Open(os.path.abspath(args.usd))
if stage is None:
    raise SystemExit(f"cannot open {args.usd}")


def walk(stage):
    return iter(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))


print("=" * 100)
print(f"USD: {args.usd}")
print("=" * 100)

for prim in walk(stage):
    name = prim.GetName()
    if name not in ("gripper_joint_1", "gripper_joint_2"):
        continue
    print(f"\n--- {name}  ({prim.GetPath()}) ---")
    if prim.IsA(UsdPhysics.RevoluteJoint):
        j = UsdPhysics.RevoluteJoint(prim)
        print(f"  UsdPhysics.RevoluteJoint.axis        = {j.GetAxisAttr().Get()}")
        print(f"  lowerLimit / upperLimit              = {j.GetLowerLimitAttr().Get()} / {j.GetUpperLimitAttr().Get()}")
        b0 = j.GetBody0Rel().GetTargets()
        b1 = j.GetBody1Rel().GetTargets()
        print(f"  body0 / body1                        = {b0} / {b1}")
        # local rotation of each joint frame relative to its body -- this is what actually
        # determines which way "axis" points in world space, independent of the axis token.
        q0 = j.GetLocalRot0Attr().Get()
        q1 = j.GetLocalRot1Attr().Get()
        print(f"  localRot0 / localRot1                = {q0} / {q1}")
    else:
        print(f"  NOT a UsdPhysics.RevoluteJoint (type: {prim.GetTypeName()})")

    applied = list(prim.GetAppliedSchemas())
    mimic_schemas = [s for s in applied if "Mimic" in s]
    print(f"  applied schemas                      = {applied}")
    if mimic_schemas:
        for inst in mimic_schemas:
            # schema string looks like "PhysxMimicJointAPI:rotZ" -- pull the instance name
            axis_instance = inst.split(":")[-1] if ":" in inst else None
            print(f"  mimic instance                       = {axis_instance!r}  (from {inst!r})")
            if axis_instance:
                api = PhysxSchema.PhysxMimicJointAPI(prim, axis_instance)
                ref_rel = api.GetReferenceJointRel()
                print(f"  referenceJoint                       = {ref_rel.GetTargets()}")
                print(f"  referenceJointAxis                   = {api.GetReferenceJointAxisAttr().Get()}")
                print(f"  gearing                               = {api.GetGearingAttr().Get()}")
                print(f"  offset                                = {api.GetOffsetAttr().Get()}")

print("\n" + "=" * 100)
print("Read this against assets/omx_f/omx_f.urdf: both gripper_joint_1 and gripper_joint_2 declare")
print('  <axis xyz="0 0 1"/>  -- i.e. THE SAME URDF axis. If the two joints above end up with a')
print("different RevoluteJoint.axis token, or the mimic's referenceJointAxis doesn't match the")
print("follower's own axis instance, the constraint is referencing inconsistent frames -- which")
print("would explain gripper_joint_2 barely moving regardless of the gearing sign (both +1 and -1")
print("measured ~0.1deg of joint2 motion across a 90deg sweep of joint1, see verify_mimic_gearing.py).")
print("=" * 100)
import os
os._exit(0)

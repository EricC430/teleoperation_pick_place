"""Patch what the URDF -> USD conversion gets wrong for the OMX-F, from `sim/omx_constants.py` only. pxr only.

Two callers, one patch:
  * `sim/convert_omx_urdf.py` (Isaac Lab, container) right after it converts the URDF;
  * `sim/patch_omx_usd.py` (plain Isaac Sim, e.g. the Windows standalone) on an already generated asset. It writes a
    thin root layer that SUBLAYERS the generated asset, so the 8 MB mesh layers are not copied and every patched
    value is readable in one small .usda.

What is patched, and the evidence:
  1. joint limits      URDF placeholders +/-2pi. Mode `urdf` keeps +/-360 deg on joint1-5 ([柏宇說 2026-10-14] replay
                       drives the arm through angles the real arm reached, so no narrowing), `factory` / `site` as
                       omx_constants. The gripper always keeps its factory 0..100 deg: at 0 the fingers touch (mesh
                       read 2026-09-29) and finger-finger contact is not simulated (self-collision off).
  2. maxForce          stall torque of the motor actually on our bus (omx_constants.JOINTS, fixed 2026-10-14).
  3. maxJointVelocity  that motor's no-load speed.
  4. stiffness/damping omx_constants.stiffness()/damping() (per RADIAN there); USD angular drives are per DEGREE.
  5. mimic spring      PhysxMimicJointAPI on gripper_joint_2, natural frequency / damping ratio from omx_constants.
  6. zero-mass links   [已查證 2026-10-14] end_effector_link has physics:mass 0 and no collider; PhysX turns that
                       into 1 kg on the fingertip. Set to omx_constants.MARKER_LINK_MASS_KG.
"""
from __future__ import annotations

import math

import omx_constants as K

DEG_PER_RAD = 180.0 / math.pi


def limits_deg(joint: K.Joint, mode: str) -> tuple[float, float]:
    if mode == "urdf" and joint.urdf_name != "gripper_joint_1":
        return -360.0, 360.0
    lo, hi = K.effective_limits_rad(joint, apply_site_limits=(mode == "site"))  # gripper: factory either way
    return lo * DEG_PER_RAD, hi * DEG_PER_RAD


def patch(stage, limits: str = "urdf", mimic_gearing: float = 1.0) -> list[str]:
    """Author the patch on the stage's current edit target. Returns 'what: old -> new' lines."""
    from pxr import PhysxSchema, Sdf, UsdPhysics  # noqa: PLC0415

    log = []
    by_name = {j.urdf_name: j for j in K.JOINTS}
    seen = set()
    prims = {p.GetName(): p for p in stage.Traverse()}
    for name, j in by_name.items():
        p = prims.get(name)
        if p is None or not p.IsA(UsdPhysics.RevoluteJoint):
            continue
        seen.add(name)
        rj = UsdPhysics.RevoluteJoint(p)
        lo, hi = limits_deg(j, limits)
        log.append(f"{name} limits: {rj.GetLowerLimitAttr().Get()}..{rj.GetUpperLimitAttr().Get()} -> {lo:.1f}..{hi:.1f} deg ({limits})")
        rj.GetLowerLimitAttr().Set(float(lo))
        rj.GetUpperLimitAttr().Set(float(hi))
        d = UsdPhysics.DriveAPI.Get(p, "angular")
        if not d:
            d = UsdPhysics.DriveAPI.Apply(p, "angular")
        k, c = K.stiffness(j) / DEG_PER_RAD, K.damping(j) / DEG_PER_RAD
        log.append(f"{name} drive ({j.motor.model}): maxForce {d.GetMaxForceAttr().Get()} -> {j.motor.stall_torque_nm}, "
                   f"stiffness {d.GetStiffnessAttr().Get()} -> {k:.4f}, damping {d.GetDampingAttr().Get()} -> {c:.5f} (per deg)")
        d.CreateMaxForceAttr().Set(float(j.motor.stall_torque_nm))
        d.CreateStiffnessAttr().Set(float(k))
        d.CreateDampingAttr().Set(float(c))
        px = PhysxSchema.PhysxJointAPI.Apply(p)
        v = j.motor.velocity_limit_rad_s * DEG_PER_RAD
        log.append(f"{name} maxJointVelocity: {px.GetMaxJointVelocityAttr().Get()} -> {v:.1f} deg/s")
        px.CreateMaxJointVelocityAttr().Set(float(v))
    missing = set(by_name) - seen
    if missing:
        log.append(f"🔴 joints NOT FOUND, never patched: {sorted(missing)} -- do not use this asset until explained")

    follower_name, master_name, _ = K.MIMIC_JOINT
    f, m = prims.get(follower_name), prims.get(master_name)
    if f is None or m is None:
        log.append(f"🔴 {follower_name}/{master_name} not found -- mimic NOT patched")
    else:
        if not any("Mimic" in s for s in f.GetAppliedSchemas()):
            # The Isaac Lab 5.1 converter drops the URDF <mimic> (verified 2026-09-03); recreate it.
            api = PhysxSchema.PhysxMimicJointAPI.Apply(f, "rotZ")
            api.CreateReferenceJointRel().SetTargets([m.GetPath()])
            api.CreateReferenceJointAxisAttr().Set("rotZ")
            api.CreateGearingAttr().Set(float(mimic_gearing))
            api.CreateOffsetAttr().Set(0.0)
            log.append(f"{follower_name}: mimic of {master_name} created, gearing {mimic_gearing}")
        for attr, val in (("naturalFrequency", K.MIMIC_NATURAL_FREQUENCY), ("dampingRatio", K.MIMIC_DAMPING_RATIO)):
            a = f.GetAttribute(f"physxMimicJoint:rotZ:{attr}")
            old = a.Get() if a else None
            f.CreateAttribute(f"physxMimicJoint:rotZ:{attr}", Sdf.ValueTypeNames.Float).Set(float(val))
            log.append(f"{follower_name} mimic {attr}: {old} -> {val}")

    p = prims.get("end_effector_link")
    if p is not None and p.HasAPI(UsdPhysics.MassAPI):
        mass = UsdPhysics.MassAPI(p)
        log.append(f"end_effector_link mass: {mass.GetMassAttr().Get()} -> {K.MARKER_LINK_MASS_KG}")
        mass.CreateMassAttr().Set(float(K.MARKER_LINK_MASS_KG))
    return log

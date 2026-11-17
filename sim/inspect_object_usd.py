"""Cheap, physics-free check of an object USD's authored size (no simulation stepping)."""
import argparse, sys, os
import isaac_boot  # noqa: E402  (Isaac Lab 3.0 launcher, see isaac_boot.py)
p = argparse.ArgumentParser()
p.add_argument("--usd", required=True)
isaac_boot.add_args(p)
args = p.parse_args()
app = isaac_boot.start(args)
from pxr import Usd, UsdGeom
stage = Usd.Stage.Open(args.usd)
print("metersPerUnit:", UsdGeom.GetStageMetersPerUnit(stage))
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
box = cache.ComputeWorldBound(root if root.IsValid() and root != stage.GetPseudoRoot() else stage.GetPseudoRoot())
rng = box.ComputeAlignedRange()
print("default prim:", stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None)
print("bbox min:", list(rng.GetMin()), "max:", list(rng.GetMax()))
sz = [rng.GetMax()[i]-rng.GetMin()[i] for i in range(3)]
print("size (stage units):", sz)
import os
os._exit(0)

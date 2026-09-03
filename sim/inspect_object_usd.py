"""Cheap, physics-free check of an object USD's authored size (no simulation stepping)."""
import argparse, sys, os
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--usd", required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
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
app.close()

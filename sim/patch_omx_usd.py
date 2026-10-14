"""Write a patched OMX-F asset on top of an already generated one, with plain Isaac Sim (no Isaac Lab).

The generated asset (`sim/convert_omx_urdf.py`, 2026-09-18) predates the motor-table fix and the 1 kg / mimic findings
of 2026-10-14. Regenerating it needs Isaac Lab's URDF converter (the container); the patch itself is pure USD, so this
writes `<out>.usda` whose only content is `subLayers = [<generated>]` plus the patched opinions from
`sim/omx_usd_patch.py` -- nothing copied, every changed value visible in one text file.

    cd <isaac-sim-standalone> && ./python.bat <repo>/sim/patch_omx_usd.py \\
        --src <repo>/assets/omx_f_generated/omx_f.usd --out <repo>/assets/omx_f_generated/omx_f_real.usda
"""
import argparse
import os
import sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", required=True, help="the generated asset (left untouched)")
ap.add_argument("--out", required=True, help="the patched root layer to write (.usda)")
ap.add_argument("--limits", choices=("urdf", "factory", "site"), default="urdf")
ap.add_argument("--mimic-gearing", type=float, default=1.0)
ap.add_argument("--force", action="store_true", help="overwrite --out, only if this script wrote it")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pxr import Sdf, Usd  # noqa: E402

import omx_usd_patch  # noqa: E402

src = os.path.abspath(args.src)
out = os.path.abspath(args.out)
if os.path.exists(out):
    old = Sdf.Layer.FindOrOpen(out)
    ours = bool(old) and old.customLayerData.get("patched_by") == "sim/patch_omx_usd.py"
    del old  # release it, or CreateNew below finds the same identifier still registered
    if not (args.force and ours):
        raise SystemExit(f"{out} exists; pick a new name, or --force if this script wrote it")
    os.replace(out, out + ".prev")  # keep the previous patch next to it
src_layer = Sdf.Layer.FindOrOpen(src)
layer = Sdf.Layer.CreateNew(out)
layer.subLayerPaths.append(os.path.relpath(src, os.path.dirname(out)).replace("\\", "/"))
layer.defaultPrim = src_layer.defaultPrim
layer.customLayerData = {"patched_by": "sim/patch_omx_usd.py", "limits": args.limits, "source": os.path.basename(src)}
stage = Usd.Stage.Open(layer)
stage.SetEditTarget(Usd.EditTarget(layer))
lines = omx_usd_patch.patch(stage, limits=args.limits, mimic_gearing=args.mimic_gearing)
layer.Save()
print("=" * 100)
print(f"patched {os.path.basename(src)} -> {out}")
for ln in lines:
    print("  " + ln)
print("=" * 100)
app.close()

"""Make a Newton-loadable COPY of the generated OMX USD (assets/omx_f_generated -> assets/omx_f_generated_newton).

Why (2026-11-16, first Newton try): `omx_f.usd` composes with ONE error -- `configuration/omx_f_base.usd` has
`/omx_f/world/visuals` referencing `</visuals/world>` inside its own layer, and that prim does not exist (the visuals of
the virtual `world` frame; it carries no geometry). Isaac Sim PhysX ignores a composition error silently; Newton's USD
importer is strict (`_raise_on_stage_errors`) and refuses the stage. The original asset is left untouched -- S9 runs on it.

Run (plain Isaac Sim python, via run_in_container.sh):  ./sim/run_in_container.sh newton_asset_fix.py
What it does: copies the directory, then in the COPY deletes every reference whose target prim does not exist (scans all
layers in configuration/), and prints each deletion. Nothing else is changed.
"""
from __future__ import annotations

import os
import shutil
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from pxr import Sdf  # noqa: E402

GUEST = os.environ.get("GUEST_VOLUME", "/workspace/test_isaaclab")
SRC = f"{GUEST}/assets/omx_f_generated"
DST = f"{GUEST}/assets/omx_f_generated_newton"
if os.path.exists(DST):
    shutil.rmtree(DST)
shutil.copytree(SRC, DST)

removed = []
conf = os.path.join(DST, "configuration")
for fn in sorted(os.listdir(conf)):
    if not fn.endswith((".usd", ".usda", ".usdc")):
        continue
    path = os.path.join(conf, fn)
    layer = Sdf.Layer.FindOrOpen(path)

    def scan(spec):
        for r in list(spec.referenceList.GetAddedOrExplicitItems()):
            target = layer if not r.assetPath else Sdf.Layer.FindOrOpen(os.path.normpath(os.path.join(conf, str(r.assetPath))))
            if r.primPath and not (target and target.GetPrimAtPath(r.primPath)):
                spec.referenceList.Remove(r)
                removed.append((fn, str(spec.path), str(r.assetPath), str(r.primPath)))
        for c in spec.nameChildren:
            scan(c)

    for rp in layer.rootPrims:
        scan(rp)
    layer.Save()

for row in removed:
    print("REMOVED dangling reference:", row)
print(f"done: {DST}  ({len(removed)} reference(s) removed)")
sys.stdout.flush()
os._exit(0)

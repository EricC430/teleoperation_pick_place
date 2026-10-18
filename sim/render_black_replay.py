"""Render recorded poses with the touch-pinned fit and black textures (Isaac Sim 5.1).

Kinematic visual replay: no physics or inferred object attachment. Run with Isaac's
python.sh, --repo pointing at a copy of this repo and --asset at the generated USD.
"""
import argparse
import json
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--asset', required=True)
p.add_argument('--data', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--episode', type=int, default=0)
p.add_argument('--stride', type=int, default=1)
p.add_argument('--max-frames', type=int, default=0)
a = p.parse_args()
sys.path[:0] = [str(a.repo), str(a.repo / 'sim'), str(a.repo / 'scripts')]
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
import numpy as np
import pandas as pd
import cv2
import omni.usd
import omni.replicator.core as rep
from pxr import Usd, UsdGeom, UsdLux, UsdShade, UsdPhysics, Gf, Sdf
from scipy.spatial.transform import Rotation
import camera_distortion as cd
import joint_mapping as jm
import solve_link_markers as sl
from touch_test import FitModel
from reach_logger.fk import _translation, _rotation

a.out.mkdir(parents=True, exist_ok=True)
fit_path = a.repo / 'calibration/2026-10-14_link_marker_fit.json'
model = FitModel(fit_path)
df = pd.read_parquet(a.data)
df = df[df.episode_index == a.episode].sort_values('frame_index')
assert len(df), 'Episode is empty'
states = np.stack(df['observation.state'])
fps = 15
ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
root = stage.DefinePrim('/World/Robot', 'Xform')
root.GetReferences().AddReference(a.asset)
# Converted visuals are instanceable; expose their meshes for material binding.
for prim in list(Usd.PrimRange(root)):
    if prim.IsInstanceable():
        prim.SetInstanceable(False)
assert Sdf.Layer.FindOrOpen(str(a.repo / 'assets/omx_f/omx_f_material_override.usda'))
material = UsdShade.Material.Define(stage, '/World/Black')
material.GetPrim().GetReferences().AddReference(
    str(a.repo / 'assets/omx_f/omx_f_material_override.usda'), '/omx_f/Looks/Black_Matte')
links = {}
mesh_count = 0
for prim in list(Usd.PrimRange(root)):
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
    if prim.IsA(UsdPhysics.Joint):
        prim.SetActive(False)
    if prim.GetName() in ('world', 'end_effector_link') and prim.IsA(UsdGeom.Xform):
        UsdGeom.Imageable(prim).MakeInvisible()
    if prim.IsA(UsdGeom.Mesh):
        mesh_count += 1
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)
    if prim.GetName() in [f'link{i}' for i in range(8)]:
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        links[prim.GetName()] = xf.AddTransformOp()
assert len(links) == 8 and mesh_count > 0, (list(links), mesh_count)
# Hide collision geometry; the converter marks it invisible already.
UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(650)
table = UsdGeom.Cube.Define(stage, '/World/Table')
table.CreateSizeAttr(1)
table.AddTranslateOp().Set(Gf.Vec3d(.3, 0, -.025))
table.AddScaleOp().Set(Gf.Vec3f(1.3, 1.3, .05))
table.CreateDisplayColorAttr([(.65, .53, .40)])
intr = cd.load_model(a.repo / 'calibration/2026-10-07_camera_intrinsics_front-left.json')
rw, rh, f = cd.render_spec(intr)
mx, my, coverage = cd.build_render_to_real_maps(intr, rw, rh, f)
assert coverage > .999
ext = json.loads((a.repo / 'calibration/2026-10-14_camera_extrinsics_front-left_session.json').read_text())
pos = np.array(ext['pos_m']); pos[2] -= ext['table_top_z']
w, x, y, z = ext['quat_wxyz']
rot = Rotation.from_quat([x, y, z, w]).as_matrix() @ np.diag([1., -1., -1.])
x, y, z, w = Rotation.from_matrix(rot).as_quat()
cam = UsdGeom.Camera.Define(stage, '/World/Camera')
cam.AddTranslateOp().Set(Gf.Vec3d(*pos))
cam.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
cam.CreateHorizontalApertureAttr(20.955)
cam.CreateVerticalApertureAttr(20.955 * rh / rw)
cam.CreateFocalLengthAttr(f * 20.955 / rw)
cam.CreateClippingRangeAttr(Gf.Vec2f(.01, 20))
product = rep.create.render_product(str(cam.GetPath()), (rw, rh))
rgb = rep.AnnotatorRegistry.get_annotator('rgb')
rgb.attach([product])
# A second fixed overview keeps the whole arm visible, including the folded HOME.
overview = UsdGeom.Camera.Define(stage, '/World/Overview')
eye = np.array([.65, .65, .48]); target = np.array([.13, 0, .23])
back = (eye - target) / np.linalg.norm(eye - target)
right = np.cross([0, 0, 1], back); right /= np.linalg.norm(right)
up = np.cross(back, right)
x, y, z, w = Rotation.from_matrix(np.column_stack([right, up, back])).as_quat()
overview.AddTranslateOp().Set(Gf.Vec3d(*eye))
overview.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
overview.CreateHorizontalApertureAttr(20.955)
overview.CreateVerticalApertureAttr(20.955 * intr.height / intr.width)
overview.CreateFocalLengthAttr(24.)
overview.CreateClippingRangeAttr(Gf.Vec2f(.01, 20))
overview_product = rep.create.render_product(str(overview.GetPath()), (intr.width, intr.height))
overview_rgb = rep.AnnotatorRegistry.get_annotator('rgb')
overview_rgb.attach([overview_product])
frames = list(range(0, len(states), a.stride))
if a.max_frames:
    frames = frames[:a.max_frames]
for idx, k in enumerate(frames):
    st = states[k]
    q = jm.SCALE_RAD_PER_UNIT['shoulder_lift'] * st[:5] + model.off
    g = np.array([0, 0, -9.81])
    if not model.g_in_base:
        g = model.base[:3, :3].T @ g
    tau = sl.gravity_torques(sl.chain_transforms(q, model.chain), model.inert, g)
    q[1:4] += model.sag * tau
    transforms = [model.base] + [model.base @ t for t in sl.chain_transforms(q, model.chain)]
    grip = jm.lerobot_to_urdf_rad(st)[5]
    transforms += [transforms[5] @ _translation((.0295, .0075, 0)) @ _rotation('z', grip),
                   transforms[5] @ _translation((.0295, -.0108, 0)) @ _rotation('z', -grip)]
    for j, transform in enumerate(transforms):
        links[f'link{j}'].Set(Gf.Matrix4d(transform.T.tolist()))
    # Replicator explicitly waits for a fresh RTX frame after each pose change.
    rep.orchestrator.step(rt_subframes=16 if idx == 0 else 4, pause_timeline=True)
    im = np.asarray(rgb.get_data())[..., :3]
    assert im.size, 'RTX returned an empty frame'
    im = cv2.remap(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), mx, my, cv2.INTER_LINEAR)
    overview_im = cv2.cvtColor(np.asarray(overview_rgb.get_data())[..., :3], cv2.COLOR_RGB2BGR)
    for panel, title in ((im, 'Calibrated front-left'), (overview_im, 'Overview')):
        cv2.putText(panel, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
        cv2.putText(panel, f'Frame {k}  |  {k / fps:.2f}s', (16, intr.height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    im = np.hstack([im, overview_im])
    cv2.imwrite(str(a.out / f'frame_{idx:05d}.png'), im)
    if idx % 30 == 0:
        print(f'RENDER {idx + 1}/{len(frames)} source={k}', flush=True)
manifest = {'dataset': str(a.data), 'episode': a.episode, 'frames': frames,
            'fps': fps / a.stride, 'fit': str(fit_path), 'asset': a.asset,
            'material': 'assets/omx_f/omx_f_material_override.usda', 'mesh_count': mesh_count,
            'mode': 'kinematic visual replay; no object or contact simulation',
            'camera': '2026-10-14 front-left session; lens 2026-10-07',
            'panels': ['calibrated front-left', 'fixed overview'],
            'note': 'Camera is from 10/14; uvc_60 recording predates this camera pose.'}
(a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print('RENDER COMPLETE', flush=True)
app.close()

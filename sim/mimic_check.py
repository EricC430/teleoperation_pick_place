"""S5 gap 2 — does the second finger mirror the first? Settles the `--mimic-gearing` sign.

`sim/README.md` "Two gaps that are NOT closed" #2: the URDF says `<mimic multiplier="-1"/>`, the USD
is written with `PhysxMimicJointAPI gearing=1.0`, and the two specs do not share a sign convention.
Reading either spec cannot decide it; watching the fingers can. This script watches them with numbers
instead of eyes:

  sweep gripper_joint_1 over a few angles -> for each, measure a point on each finger in the link5
  (gripper base) frame.

  mirrored (correct)   : the finger GAP changes with the angle, the MIDPOINT between fingers stays put
  same-direction (wrong): the gap stays put, the midpoint swings sideways

Run once per candidate gearing; `--override-gearing` writes a patched COPY of the USD into --out, the
asset itself is never modified. Whichever candidate passes is the value to pass to
`convert_omx_urdf.py --mimic-gearing` and to record in sim/README.md.

    ./sim/run_in_container.sh mimic_check.py --headless
    ./sim/run_in_container.sh mimic_check.py --headless --override-gearing -1.0

⚠️ This does NOT tell which direction of gripper_joint_1 is "open" on the real gripper — that is
   `joint_mapping.GRIPPER_SIGN / GRIPPER_ZERO_DEG`. It prints the gap per angle so that can be matched
   against the real data (uvc_60: gripper.pos ~59 open, ~47-50 closed on a paper cup).
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--usd", default="/workspace/test_isaaclab/assets/omx_f_generated/omx_f.usd")
parser.add_argument("--override-gearing", type=float, default=None,
                    help="test this gearing on a patched copy of --usd (the original is not touched)")
parser.add_argument("--angles-deg", default="0,15,30,45,60", help="gripper_joint_1 targets to sweep")
parser.add_argument("--hold-steps", type=int, default=120, help="physics steps per angle (default 120 = 1 s)")
parser.add_argument("--tip-offset-m", type=float, default=0.03,
                    help="probe point along each finger link's +x (only needs to be off the joint axis)")
parser.add_argument("--out", default="/workspace/test_isaaclab/omx_sim/out/mimic_check")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from pxr import PhysxSchema, Usd  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_inv  # noqa: E402

import omx_constants as K  # noqa: E402
import omx_scene_cfg as SC  # noqa: E402

FOLLOWER, MASTER, URDF_MULT = K.MIMIC_JOINT
os.makedirs(args.out, exist_ok=True)


def read_or_patch_gearing(usd_path: str, override: float | None) -> tuple[str, float | None]:
    stage = Usd.Stage.Open(usd_path)
    found = None
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if prim.GetName() == FOLLOWER:
            api = PhysxSchema.PhysxMimicJointAPI.Get(prim, "rotZ")
            if api:
                found = api
            break
    if found is None:
        raise SystemExit(f"🔴 {FOLLOWER} carries no PhysxMimicJointAPI:rotZ in {usd_path} — re-run convert_omx_urdf.py")
    current = found.GetGearingAttr().Get()
    if override is None:
        return usd_path, current
    patched = os.path.join(args.out, f"omx_f_gearing_{override:+.1f}.usd")
    found.GetGearingAttr().Set(float(override))
    stage.Export(patched)  # flattened copy; the source layer is not saved
    print(f"gearing {current} -> {override} on a copy: {patched}")
    return patched, override


usd_path, gearing = read_or_patch_gearing(args.usd, args.override_gearing)

scene_cfg = SC.OmxArmOnlySceneCfg(num_envs=1, env_spacing=1.0)
scene_cfg.robot.spawn.usd_path = usd_path
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
scene = InteractiveScene(scene_cfg)
sim.reset()

robot = scene["robot"]
m_id = robot.find_joints(MASTER)[0][0]
f_id = robot.find_joints(FOLLOWER)[0][0]
b5, b6, b7 = (robot.find_bodies(n)[0][0] for n in ("link5", "link6", "link7"))
lo, hi = robot.data.joint_pos_limits[0, m_id].tolist()
print(f"\n{MASTER} limits in the USD: {lo*57.2958:.1f} .. {hi*57.2958:.1f} deg")
print(f"{FOLLOWER} mimic gearing under test: {gearing}   (URDF multiplier: {URDF_MULT})")

offset = torch.tensor([[args.tip_offset_m, 0.0, 0.0]], device=sim.device)


def tip_in_link5(body: int) -> torch.Tensor:
    p5, q5 = robot.data.body_pos_w[:, b5], robot.data.body_quat_w[:, b5]
    p, q = robot.data.body_pos_w[:, body], robot.data.body_quat_w[:, body]
    tip_w = p + quat_apply(q, offset)
    return quat_apply(quat_inv(q5), tip_w - p5)[0]


rows = []
for deg in [float(a) for a in args.angles_deg.split(",")]:
    target = robot.data.default_joint_pos.clone()
    target[:, m_id] = deg / 57.2958
    for _ in range(args.hold_steps):
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
    j1 = robot.data.joint_pos[0, m_id].item()
    j2 = robot.data.joint_pos[0, f_id].item()
    t6, t7 = tip_in_link5(b6), tip_in_link5(b7)
    rows.append((deg, j1 * 57.2958, j2 * 57.2958, t6[1].item(), t7[1].item()))

print(f"\n{'target':>8}{'j1 (deg)':>10}{'j2 (deg)':>10}{'j2/j1':>8}{'tip6 y (mm)':>13}{'tip7 y (mm)':>13}{'gap (mm)':>10}{'mid (mm)':>10}")
for deg, j1, j2, y6, y7 in rows:
    ratio = f"{j2 / j1:8.2f}" if abs(j1) > 1.0 else f"{'-':>8}"
    print(f"{deg:8.1f}{j1:10.2f}{j2:10.2f}{ratio}{y6*1000:13.2f}{y7*1000:13.2f}{(y6-y7)*1000:10.2f}{(y6+y7)/2*1000:10.2f}")

gaps = [(y6 - y7) * 1000 for *_, y6, y7 in rows]
mids = [(y6 + y7) / 2 * 1000 for *_, y6, y7 in rows]
gap_range, mid_range = max(gaps) - min(gaps), max(mids) - min(mids)
tracked = max(abs(j1 - deg) for deg, j1, *_ in rows)
print(f"\ngap range {gap_range:.2f} mm, midpoint range {mid_range:.2f} mm, worst |j1 - target| {tracked:.2f} deg")
if tracked > 5.0:
    print("⚠️  gripper_joint_1 did not reach its targets (limits or gains) — the verdict below covers only the angles it reached")
if gap_range < 1.0 and mid_range < 1.0:
    verdict = "🔴 INCONCLUSIVE: the fingers barely moved — check limits, gains, and that the mimic constraint is active"
elif gap_range > 3.0 * mid_range:
    verdict = f"✅ MIRRORED with gearing={gearing}: fingers close/open symmetrically"
elif mid_range > 3.0 * gap_range:
    verdict = f"🔴 SAME-DIRECTION with gearing={gearing}: fingers swing together — try the opposite sign"
else:
    verdict = "🟡 MIXED: neither clearly mirrored nor clearly same-direction — look at a render before deciding"
print(verdict)
print("\nIf this passes: record the gearing in sim/README.md (gap #2) and re-run convert_omx_urdf.py with it.")
import os
os._exit(0)

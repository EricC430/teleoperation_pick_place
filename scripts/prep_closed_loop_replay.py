"""    uv run python scripts/prep_closed_loop_replay.py <out dir> [<rollout dataset name>]
Episodes of a closed-loop rollout (default: 10-14 15:12) -> sim/physics_replay.py inputs: joint radians from the touch-pinned fit
(offsets, sag), gripper from the action, cup on its c point, world_from_link0 from the fit."""
import csv, json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'sim'); sys.path.insert(0, '.')
import touch_test as TT, solve_link_markers as SL, joint_mapping as jm
import pandas as pd
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
RID = sys.argv[2] if len(sys.argv) > 2 else 'rollout_omx_b1_papercup_60_13_10rcvrycut_46far_aug_fldrop30_100k_nas30_paper_cup_20261014_151208'
ROOT = Path('D:/hf/lerobot/ericc430') / RID
m = TT.FitModel(TT.REPO / 'calibration/2026-10-14_link_marker_fit.json')
pl = {r['placement_id'].replace('eval-close_0', 'c').replace('c0', 'c'): (float(r['x_cm']) / 100, float(r['y_cm']) / 100) for r in csv.DictReader(open('configs/placements/campA_136sym_20260908_20260908_eval-close.csv', encoding='utf-8'))}
meta = {int(r['episode_index']): r for r in csv.DictReader(open(f'episode_meta/{RID}.csv', encoding='utf-8-sig'))}
d = pd.concat(pd.read_parquet(f) for f in sorted((ROOT / 'data').rglob('*.parquet')))
def qbody(st):
    q = jm.SCALE_RAD_PER_UNIT['shoulder_lift'] * st[:5] + m.off
    tau = SL.gravity_torques(SL.chain_transforms(q, m.chain), m.inert, np.array([0, 0, -9.81]) if m.g_in_base else m.base[:3, :3].T @ np.array([0, 0, -9.81]))
    q = q.copy(); q[1:4] += m.sag * tau
    return q
for e, md in sorted(meta.items()):
    de = d[d.episode_index == e].sort_values('frame_index')
    st = np.stack(de['observation.state'].values).astype(float); ac = np.stack(de['action'].values).astype(float)
    ed = OUT / f'cl_ep{e:02d}'; ed.mkdir(exist_ok=True)
    json.dump({"dataset": RID, "episode": e, "fps": 15, "placement": md['placement_id'], "outcome": md['outcome'],
               "cup_xy_m": pl[md['placement_id']], "state": st.tolist(), "action": ac.tolist(),
               "q_body_rad": [qbody(s).tolist() for s in st]}, open(ed / 'episode.json', 'w'))
    json.dump({"gripper_target_rad_from_action": [float(jm.lerobot_to_urdf_rad(np.r_[np.zeros(5), g])[5]) for g in ac[:, 5]],
               "gripper_rad_from_state": [float(jm.lerobot_to_urdf_rad(np.r_[np.zeros(5), g])[5]) for g in st[:, 5]]}, open(ed / 'gripper.json', 'w'))
json.dump({"fit": {"name": "touch-pinned link-marker fit (calibration/2026-10-14_link_marker_fit.json)", "world_from_link0": m.base.tolist()}},
          open(OUT / 'fit_candidate.json', 'w'), indent=1)
print('prepared', len(meta), 'episodes ->', OUT)

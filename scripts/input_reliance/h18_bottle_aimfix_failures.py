"""H18: 11-17 aimfix28_twoobj10 200k bottle closed loop (26/36) - where the reach-short, stalled (c32) and misplaced (c5)
episodes come from. Joint data only. (2026-11-17, Boyu's questions)

Q1 misplaced c5: joints at the release (gripper opens after the carrying grasp) vs every successful release in the
   earlier bottle closed loops and in the human bottle demos; gripper state while carrying (cap grasp vs body grasp).
Q2 stalled c32: action vs state over the stall (policy holds still, or commands motion that something blocks?);
   nearest training frames to the stall pose; distance to the last frames of aimfix_A1_cut ep4 / ep16 1st halves
   (11 / 8 frames, the 'short recovery' episodes Boyu suspects).
Q3 demos where the arm goes UP while the gripper closes: per bottle training source, count; same measure on the
   rollouts (11-11, 11-15, 11-17); check against Boyu's notes c3 / c34 / c35 ('閉爪時手臂往上抬').
Q4 reach at the first close (h15 rule and offset), 11-17 vs 11-15 M-full 200k paired by placement; commanded reach
   (FK of the action) vs reached (FK of the state) at that close.

Grasp events (gripper ACTION a_g, open ~85-87, rest ~59, on the object 47-54):
  open phase  = frames with a_g >= 75; one event per open phase.
  t0 close start = last frame of the open phase.
  t1 fingers stopped = first frame after t0 where the gripper STATE is within 1.0 of its minimum over [t0, t2].
  t2 release = first frame after t1 where a_g > min(a_g over [t0, t2]) + 6, or the next open phase / episode end.
  The carrying grasp of a success = the last event. z = FK fingertip height above the table (S7 constants, riser 14.6 cm).
Fixed before the run:
  Q3 'lift while closing' = z(t1) - z(t0) >= +1.0 cm. Check: c34 and c35 (11-17) should pass, else the measure is not
      what Boyu saw and Q3 counts are not used.
  Q1 c5 is 'not turned enough' if its release pan is below the 5th percentile (toward 0) of the successful releases
      of all rollouts and humans. Carry gripper state: c5 vs c2 / c34 / c35 (cap, success) - if c5 is inside their
      range, the cap grasp alone does not separate c5.
  Q2 'policy holds still' = over the stall window max_j |action_j - state_j| < 2 and action range < 1 per joint;
      'blocked' = action keeps a gap >= 2 on some joint. Stall window = longest run of frames (>= 3 s) where every
      joint's action moves < 0.3 per frame.
  Q4 11-17 reaches shorter than 11-15 if the paired median (e_close_1117 - e_close_1115) > +1 cm with bootstrap 95% CI
      above 0. 'Commanded short' at a close if r_fk(action) - r_fk(state) < 1 cm (the policy asked for that radius);
      'blocked' if >= 1 cm (asked further, arm did not get there).
Added AFTER seeing the Q1-Q4 output (2026-11-17/18, no reading rule fixed beforehand - descriptive only):
  Q3b closing speed: gripper ACTION time from t0 to 90 % of its close depth, FK height rise over it, and any re-opening
      (sum of positive action steps between t0 and t1); humans (normal + precise) vs 11-11 / 11-15 / 11-17, ok vs fail.
  Q4b tracking slope = OLS slope of reached r (FK at first close + offset) on placement r, bootstrap 95 % CI;
      humans vs 11-15 vs 11-17 (1 = follows, < 1 = compressed toward the middle).
  Q5 lateral aim at the first close = r * (theta_object - theta_arm) cm with theta_arm = linear fit of theta on
      shoulder_pan over the HUMAN BOTTLE demos (h14's paper-cup fit is off for the bottle: it gives ~8 cm on successes).
      11-11 / 11-15 / 11-17 per placement -> h18_lateral.csv.
Caveats: n = 1 for c5 and for c32; r is FK + one human offset (h15: about 1-2 cm residual); placement r is nominal;
  fingertip height on CAD constants; twoobj_A1 has both objects (counted as its own source).

    uv run python scripts/input_reliance/h18_bottle_aimfix_failures.py      (any cwd; NOT the anaconda python:
                                                                             its pyarrow / numpy do not match)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)  # every path below (and in h14 / h15) is relative to the repo root
sys.path.insert(0, str(REPO / "scripts/input_reliance"))
sys.path.insert(0, str(REPO / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console (cp950) garbles the Chinese notes otherwise
except AttributeError:
    pass
import measure_pan_circle as PC  # noqa: E402
from measure_link_tilt import ARM_RISER_HEIGHT_M, JOINTS  # noqa: E402
import h14_approach_correction as P14  # noqa: E402
import h15_radial_reach as H15  # noqa: E402

FPS = 15
OUT = Path("data/logs/input_reliance")
HC = ".cache/lerobot/omx_pick_place_pilot_"
R = "D:/hf/lerobot/ericc430/"
NEW = "rollout_omx_aimfix_cup129_bottle135_aimfix28_twoobj10_200k_nas1_te-001_plastic_bottle_20261117_184157"
M1115 = "rollout_omx_mfull_cup129_bottle135_augfl_fldrop30_200k_nas1_te-001_plastic_bottle_20261115_183543"
ROLL = [  # label, folder  (all bottle closed loops with an episode_meta csv)
    ("09-18 b4a", "rollout_omx_b4a_plastic_bottle_20260918_093156"),
    ("09-21 b4a 20k", "rollout_omx_b4a_plastic_bottle_20k_20260921_100201"),
    ("09-21 b4a 100k", "rollout_omx_b4a_plastic_bottle_100k_20260921_101155"),
    ("10-18 b60 nas30", "rollout_omx_b4a_plastic_bottle_60_augfl_fldrop30_100k_nas30_20261018_161002"),
    ("10-22 b136 nas30 a", "rollout_omx_b4a_plastic_bottle_136_augfl_fldrop30_100k_nas30_20261022_102628"),
    ("10-22 b136 nas30 b", "rollout_omx_b4a_plastic_bottle_136_augfl_fldrop30_100k_nas30_20261022_103056"),
    ("10-22 b136 te", "rollout_omx_b4a_plastic_bottle_136_augfl_fldrop30_100k_nas1_te-001_20261022_104207"),
    ("10-28 b60 te", "rollout_omx_b4a_plastic_bottle_60_augfl_fldrop30_100k_nas1_te-001_20261028_213634"),
    ("11-11 mfull 200k", "rollout_omx_mfull_cup129_bottle135_augfl_fldrop30_200k_nas1_te-001_plastic_bottle_20261111_175328"),
    ("11-15 mfull 200k", M1115),
    ("11-15 mfull 100k", "rollout_omx_mfull_cup129_bottle135_augfl_fldrop30_100k_nas1_te-001_plastic_bottle_20261115_192339"),
    ("11-17 aimfix 200k", NEW),
]
TRAIN = [  # label, dataset, excluded episodes (merged exclude_episodes mapped back: 271 -> tipped ep 9)
    ("60_plastic_bottle", HC + "60_plastic_bottle", []),
    ("normal_A1", HC + "plastic_bottle_normal_A1", []),
    ("recovery_tight (uncut, last close)", HC + "plastic_bottle_recovery_A1_tight", []),
    ("precise_A1", HC + "plastic_bottle_precise_A1", []),
    ("tipped_A1", HC + "plastic_bottle_tipped_A1", [9]),
    ("aimfix_A1_cut", HC + "plastic_bottle_aimfix_A1_cut", []),
    ("twoobj_A1", HC + "twoobj_A1", []),
]
LIFT = 1.0


def load(root):
    fs = sorted(Path(root, "data").rglob("*.parquet"))
    t = pd.concat([pq.read_table(f, columns=["episode_index", "frame_index", "action", "observation.state"]).to_pandas() for f in fs])
    t = t.sort_values(["episode_index", "frame_index"])
    return t.episode_index.values, np.stack(t.action.values), np.stack(t["observation.state"].values)


def z_of(x):
    return np.array([PC.fk_tip_above_table_cm(dict(zip(JOINTS, v)), ARM_RISER_HEIGHT_M) for v in x])


def events(a, s):
    g, gs = a[:, 5], s[:, 5]
    op = g >= 75
    out, n, t = [], len(g), 0
    while t < n:
        if not op[t]:
            t += 1; continue
        u = t
        while u + 1 < n and op[u + 1]:
            u += 1
        t0 = u
        nxt = next((k for k in range(t0 + 1, n) if op[k]), n)
        if nxt - t0 < 5:
            t = nxt; continue
        gmin = g[t0:nxt].min()
        above = np.flatnonzero(g[t0:nxt] > gmin + 6)
        imin = int(np.argmin(g[t0:nxt]))
        rel = [k for k in above if k > imin]
        t2 = t0 + rel[0] if rel else nxt - 1
        smin = gs[t0:t2 + 1].min()
        t1 = t0 + int(np.argmax(gs[t0:t2 + 1] <= smin + 1.0))
        out.append(dict(t0=t0, t1=t1, t2=t2, released=bool(rel)))
        t = nxt
    return out


def meta(folder):
    f = Path("episode_meta") / f"{folder}.csv"
    if not f.exists():
        return None
    m = pd.read_csv(f, encoding="utf-8-sig", comment="#").dropna(subset=["episode_index"])
    m["episode_index"] = m.episode_index.astype(int)
    return m.set_index("episode_index")


def placements():
    a = pd.read_csv("configs/placements/campA_136sym_20260908_20260908_eval-close.csv").set_index("placement_id")
    b = pd.read_csv("configs/placements/bottle_eval-close_20261115.csv").set_index("placement_id")
    p = pd.concat([a[["r_cm", "theta_deg"]], b.loc[b.index.difference(a.index), ["r_cm", "theta_deg"]]])
    p.index = ["c" + str(int(i.split("_")[-1])) for i in p.index]
    return p


def per_event_rows(label, kind, root, m, excl=()):
    ep, a, s = load(root)
    rows = []
    for e in np.unique(ep):
        if e in excl:
            continue
        i = np.flatnonzero(ep == e)
        ev = events(a[i], s[i])
        info = m.loc[e] if (m is not None and e in m.index) else None
        for j, x in enumerate(ev):
            t0, t1, t2 = x["t0"], x["t1"], x["t2"]
            z = z_of(s[i[[t0, t1, t2]]])
            st2 = s[i[t2]]
            rows.append(dict(set=label, kind=kind, ep=int(e), n_events=len(ev), event=j, is_last=j == len(ev) - 1,
                             pid=str(info.placement_id).strip() if info is not None else "",
                             outcome=str(info.outcome) if info is not None else "",
                             mechanism=str(info.mechanism) if info is not None else "",
                             t0=t0, t1=t1, t2=t2, close_s=(t1 - t0) / FPS, carry_s=(t2 - t1) / FPS,
                             released=x["released"], z0=z[0], z1=z[1], z2=z[2], dz_close=z[1] - z[0],
                             g_carry=float(np.median(s[i[t1:t2 + 1], 5])),
                             **{f"rel_{k}": st2[n] for n, k in enumerate(JOINTS[:5])}))
    return rows


def q2(new_root):
    ep, a, s = load(new_root)
    i = np.flatnonzero(ep == 27)
    A_, S_ = a[i], s[i]
    still = np.all(np.abs(np.diff(A_[:, :5], axis=0)) < 0.3, axis=1)
    best, cur, st = (0, 0), 0, 0
    for k, v in enumerate(still):
        if v:
            if cur == 0:
                st = k
            cur += 1
            if cur > best[1] - best[0]:
                best = (st, st + cur)
        else:
            cur = 0
    lo, hi = best
    print(f"\nQ2 c32 (ep27, {len(i)} frames): longest still-action window frames {lo}-{hi} ({(hi - lo) / FPS:.1f} s)")
    if hi - lo < 3 * FPS:
        print("  no window >= 3 s"); return
    gap = A_[lo:hi + 1] - S_[lo:hi + 1]
    for n, k in enumerate(JOINTS):
        print(f"  {k:14s} action {A_[lo, n]:7.2f}..{A_[hi, n]:7.2f}  state {S_[lo, n]:7.2f}..{S_[hi, n]:7.2f}  "
              f"median gap {np.median(gap[:, n]):+6.2f}  max|gap| {np.abs(gap[:, n]).max():5.2f}")
    print(f"  before window: frames 0-{lo}; z(start of window) = {z_of(S_[[lo]])[0]:.1f} cm; "
          f"r_fk = {H15.r_fk(S_[[lo]])[0]:.1f} cm; gripper action {A_[lo, 5]:.1f}")
    pose = S_[lo]
    # nearest training frames (state, 5 arm joints + gripper, raw units)
    near = []
    for label, root, excl in TRAIN:
        e2, a2, s2 = load(root)
        keep = ~np.isin(e2, excl)
        d = np.linalg.norm(s2[keep] - pose, axis=1)
        fi = np.flatnonzero(keep)
        for k in np.argsort(d)[:5]:
            near.append((d[k], label, int(e2[fi[k]]), int(fi[k] - np.flatnonzero(e2 == e2[fi[k]])[0])))
        if label == "aimfix_A1_cut":
            for e in (4, 16, 26, 27):
                j = np.flatnonzero(e2 == e)
                dd = np.linalg.norm(s2[j] - pose, axis=1)
                print(f"  aimfix_cut ep{e} ({len(j)} frames): min dist to stall pose {dd.min():.1f} (frame {int(np.argmin(dd))}), "
                      f"last frame {dd[-1]:.1f}; last action - last state max|.| {np.abs(a2[j[-1], :5] - s2[j[-1], :5]).max():.2f}")
    near.sort()
    print("  10 nearest training frames (dist, source, ep, frame):")
    for x in near[:10]:
        print(f"    {x[0]:5.1f}  {x[1]:34s} ep{x[2]:3d} f{x[3]}")


def first_close(a, s):
    """h15's 0.6-depth rule, but depth measured from the end of the first OPEN phase (the aimfix model starts with the
    gripper half closed at ~52, so h15's 'first value' rule finds no close in about half of its episodes)."""
    ev = events(a, s)
    if not ev:
        return None
    t0 = ev[0]["t0"]; g = a[:, 5]; end = ev[1]["t0"] if len(ev) > 1 else len(g)
    depth = g[t0] - g[t0:end].min()
    return None if depth < 10 else t0 + int(np.argmax(g[t0:end] < g[t0] - 0.6 * depth))


def human_closes():
    """(placement r, placement theta, state at the first close) for the human bottle demos with placements."""
    out = []
    for lab in ("bottle normal_A1", "bottle precise_A1"):
        obj, grp, label, root, csv = next(x for x in P14.SETS if x[2] == lab)
        m = P14.meta(csv, root)
        ep, a, s = load(root)
        for r in m.itertuples():
            i = np.flatnonzero(ep == r.episode_index)
            k = first_close(a[i], s[i]) if len(i) else None
            if k is not None:
                out.append((r.r, r.th, s[i[k]]))
    return out


def boot_slope(x, y, rng, nb=2000):
    b = np.polyfit(x, y, 1)[0]
    bs = [np.polyfit(x[j], y[j], 1)[0] for j in (rng.integers(0, len(x), len(x)) for _ in range(nb))]
    return b, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def runs3():
    return [("11-11 mfull 200k", ROLL[8][1]), ("11-15 mfull 200k", M1115), ("11-17 aimfix 200k", NEW)]


def q3b(HUM_ROOTS=(HC + "plastic_bottle_normal_A1", HC + "plastic_bottle_precise_A1")):
    print("\nQ3b gripper ACTION close time (t0 -> 90 % of depth), FK rise over it, re-open during close:")
    sets = [("humans normal+precise", r, None) for r in HUM_ROOTS] + [(l, R + f, meta(f)) for l, f in runs3()]
    rows = []
    for label, root, m in sets:
        ep, a, s = load(root)
        for e in np.unique(ep):
            i = np.flatnonzero(ep == e); ev = events(a[i], s[i])
            if not ev:
                continue
            t0, t1 = ev[0]["t0"], ev[0]["t1"]; end = ev[1]["t0"] if len(ev) > 1 else len(i)
            g = a[i, 5]; depth = g[t0] - g[t0:end].min()
            if depth < 10:
                continue
            t90 = t0 + int(np.argmax(g[t0:end] < g[t0] - 0.9 * depth))
            z = z_of(s[i[[t0, t90]]]); dg = np.diff(g[t0:t1 + 1])
            oc = m.loc[e, "outcome"] if m is not None and e in m.index else "demo"
            rows.append(dict(set=label, ok=oc in ("success", "demo"), close_s=(t90 - t0) / FPS, dz=z[1] - z[0],
                             reopen=dg[dg > 0].sum()))
    d = pd.DataFrame(rows)
    for (k, ok), g in d.groupby(["set", "ok"], sort=False):
        print(f"  {k:22s} {'ok  ' if ok else 'fail'} n={len(g):2d}  close {g.close_s.median():4.1f}s (p25 {g.close_s.quantile(.25):.1f}, "
              f"p75 {g.close_s.quantile(.75):.1f})  dz {g.dz.median():+5.1f} cm  re-open median {g.reopen.median():.1f} max {g.reopen.max():.1f}")


def q5(P, hum):
    h = np.array([(r, th, st[0]) for r, th, st in hum])
    c = np.polyfit(h[:, 2], h[:, 1], 1)
    res = h[:, 0] * np.deg2rad(h[:, 1] - np.polyval(c, h[:, 2]))
    print(f"\nQ5 lateral aim, bottle human fit theta = {c[0]:.3f}*pan {c[1]:+.2f} (n={len(h)}); human residual "
          f"|lateral| median {np.median(np.abs(res)):.1f} cm, p90 {np.percentile(np.abs(res), 90):.1f}")
    out = {}
    for label, f in runs3():
        ep, a, s = load(R + f); m = meta(f); rows = []
        for e in m.index:
            i = np.flatnonzero(ep == e); k = first_close(a[i], s[i]); pid = str(m.loc[e, "placement_id"]).strip()
            if k is None or pid not in P.index:
                continue
            r, th = P.loc[pid, "r_cm"], P.loc[pid, "theta_deg"]
            rows.append(dict(pid=pid, r=r, theta=th, elat=r * np.deg2rad(th - np.polyval(c, s[i[k], 0])), out=m.loc[e, "outcome"]))
        out[label] = pd.DataFrame(rows).set_index("pid")
        x = out[label]; ok = x.out == "success"
        print(f"  {label}: median |e| ok {x.elat[ok].abs().median():.1f} / fail {x.elat[~ok].abs().median():.1f} cm;  |e| > 3: "
              f"{sorted(x.index[x.elat.abs() > 3], key=lambda p: int(p[1:]))}")
    d = out["11-17 aimfix 200k"].join(out["11-15 mfull 200k"][["elat", "out"]], rsuffix="_1115") \
        .join(out["11-11 mfull 200k"][["elat", "out"]], rsuffix="_1111")
    d.to_csv(OUT / "h18_lateral.csv", encoding="utf-8-sig")
    dd = d.dropna(subset=["elat_1115"])
    print(f"  per-placement corr 11-17 vs 11-15: {np.corrcoef(dd.elat, dd.elat_1115)[0, 1]:.2f}; corr e vs theta (11-17) "
          f"{np.corrcoef(d.theta, d.elat)[0, 1]:.2f}")
    print(d.sort_values("theta").round(1).to_string())


def q4(P, hum):
    off = float(np.median([r - H15.r_fk(st[None])[0] for r, th, st in hum]))
    print(f"\nQ4 offset (human bottle, {len(hum)} demos) = {off:+.1f} cm")
    res = {}
    for label, folder in [("11-15 mfull 200k", M1115), ("11-17 aimfix 200k", NEW)]:
        m = meta(folder); ep, a, s = load(R + folder); rows = []
        for e in m.index:
            i = np.flatnonzero(ep == e)
            k = first_close(a[i], s[i])
            pid = str(m.loc[e, "placement_id"]).strip()
            if k is None or pid not in P.index:
                continue
            r_s = H15.r_fk(s[i[[k]]])[0] + off
            r_a = H15.r_fk(a[i[[k]]])[0] + off
            rows.append(dict(pid=pid, r=P.loc[pid, "r_cm"], e_close=P.loc[pid, "r_cm"] - r_s, cmd_minus_state=r_a - r_s,
                             outcome=m.loc[e, "outcome"], mechanism=m.loc[e, "mechanism"], notes=str(m.loc[e, "notes"])[:30]))
        res[label] = pd.DataFrame(rows).set_index("pid")
    a15, a17 = res["11-15 mfull 200k"], res["11-17 aimfix 200k"]
    for lab, d in res.items():
        far = d.r >= 34
        print(f"  {lab}: n={len(d)}  median e_close near {d.e_close[~far].median():+.1f} (n={(~far).sum()})  far "
              f"{d.e_close[far].median():+.1f} (n={far.sum()})   [e > 0 = short]")
    com = a15.index.intersection(a17.index)
    diff = (a17.loc[com, "e_close"] - a15.loc[com, "e_close"]).values
    rng = np.random.default_rng(0)
    bs = [np.median(rng.choice(diff, len(diff))) for _ in range(2000)]
    print(f"  paired 11-17 - 11-15 over {len(com)} placements: median {np.median(diff):+.2f} cm, "
          f"95% CI [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]; 11-17 shorter by > 1 cm at "
          f"{(diff > 1).sum()}, longer by > 1 cm at {(diff < -1).sum()}")
    rng = np.random.default_rng(0)
    hr = np.array([r for r, th, st in hum]); hy = np.array([H15.r_fk(st[None])[0] + off for r, th, st in hum])
    print("  Q4b tracking slope (reached r on placement r; 1 = follows, < 1 = compressed):")
    print("    humans  n=%d  %.2f [%.2f, %.2f]" % (len(hr), *boot_slope(hr, hy, rng)))
    for lab, x in (("11-15", a15.loc[com]), ("11-17", a17)):
        print("    %s   n=%d  %.2f [%.2f, %.2f]" % (lab, len(x), *boot_slope(x.r.values, (x.r - x.e_close).values, rng)))
    far = hr >= 37
    print(f"    humans at r >= 37: e = r - reached {np.round(hr[far] - hy[far], 1).tolist()} cm")
    d = a17.join(a15[["e_close", "outcome"]], rsuffix="_1115")
    d.to_csv(OUT / "h18_reach_paired.csv", encoding="utf-8-sig")
    print("  11-17 per placement (sorted by r): e_close / cmd-state / 11-15 e_close / outcome 17 / 15")
    for pid, x in d.sort_values("r").iterrows():
        print(f"    {pid:4s} r {x.r:5.1f}  e {x.e_close:+5.1f}  cmd-st {x.cmd_minus_state:+5.1f}  15: "
              f"{x.e_close_1115:+5.1f}  {x.outcome:9s} {str(x.outcome_1115):9s} {x.mechanism if isinstance(x.mechanism, str) else ''} | {x.notes}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = placements()
    rows = []
    for label, folder in ROLL:
        m = meta(folder)
        if m is None or not Path(R + folder).exists():
            print(f"skip {label}: no meta or data"); continue
        rows += per_event_rows(label, "rollout", R + folder, m)
    for label, root, excl in TRAIN:
        rows += per_event_rows(label, "train", root, None, excl)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "h18_events.csv", index=False, encoding="utf-8-sig")

    # ---------------- Q1
    rel = d[d.is_last & d.released & ((d.kind == "train") | (d.outcome == "success"))]
    print("Q1 release pan (state) of successful carrying grasps:")
    for k, g in rel.groupby("set", sort=False):
        print(f"  {k:34s} n={len(g):3d}  pan p5 {g.rel_shoulder_pan.quantile(.05):6.1f}  median {g.rel_shoulder_pan.median():6.1f}  "
              f"p95 {g.rel_shoulder_pan.quantile(.95):6.1f}   g_carry median {g.g_carry.median():5.1f}")
    print(f"  ALL n={len(rel)}: pan p5 {rel.rel_shoulder_pan.quantile(.05):.1f}, p1 {rel.rel_shoulder_pan.quantile(.01):.1f}, "
          f"max {rel.rel_shoulder_pan.max():.1f}  (bin is on the negative-pan side)")
    print("  carrying grip state g_carry < 55 (cap-like; c2/c34/c35 = 51.6-51.8, body ~58.6) per set:")
    for k, g in rel.groupby("set", sort=False):
        print(f"    {k:34s} {int((g.g_carry < 55).sum()):3d} / {len(g):3d}   eps {sorted(g.ep[g.g_carry < 55].tolist())[:20]}")
    print(f"  successful releases with pan > -35 (c5's): {int((rel.rel_shoulder_pan > -35).sum())} / {len(rel)}:",
          [(r.set, r.ep, round(r.rel_shoulder_pan, 1)) for r in rel[rel.rel_shoulder_pan > -40].itertuples()])
    new = d[d.set == "11-17 aimfix 200k"]
    print("  11-17 events of c5 / c2 / c34 / c35 / c1 (all events):")
    for pid in ("c5", "c2", "c34", "c35", "c1"):
        for x in new[new.pid == pid].itertuples():
            print(f"    {pid:4s} ev{x.event} {x.outcome:9s} t0 {x.t0} t1 {x.t1} t2 {x.t2} released {x.released}  close {x.close_s:4.1f}s carry {x.carry_s:4.1f}s"
                  f"  pan@rel {x.rel_shoulder_pan:6.1f}  lift {x.rel_shoulder_lift:6.1f} elbow {x.rel_elbow_flex:6.1f} wrist {x.rel_wrist_flex:6.1f}"
                  f"  z@rel {x.z2:5.1f}  g_carry {x.g_carry:5.1f}")
    s17 = rel[rel.set == "11-17 aimfix 200k"]
    print(f"  11-17 successes g_carry range {s17.g_carry.min():.1f}..{s17.g_carry.max():.1f}, z@release median {s17.z2.median():.1f}, "
          f"carry_s median {s17.carry_s.median():.1f}")

    # ---------------- Q3
    print(f"\nQ3 dz while closing = z(t1) - z(t0), lift if >= {LIFT} cm (carrying / last event):")
    for k, g in d[d.is_last].groupby("set", sort=False):
        print(f"  {k:34s} n={len(g):3d}  lift {int((g.dz_close >= LIFT).sum()):3d}  dz median {g.dz_close.median():+5.1f}  "
              f"p90 {g.dz_close.quantile(.9):+5.1f}  max {g.dz_close.max():+5.1f}  close_s median {g.close_s.median():4.1f}")
    print("  11-17 per episode (all events) dz / close_s:")
    for x in new.itertuples():
        print(f"    {x.pid:4s} ev{x.event} {x.outcome:9s} dz {x.dz_close:+5.1f} close {x.close_s:4.1f}s  z0 {x.z0:5.1f} z1 {x.z1:5.1f}  {x.mechanism}")
    tr = d[(d.kind == "train") & d.is_last & (d.dz_close >= LIFT)]
    print("  training episodes with lift while closing:", [(r.set, r.ep, round(r.dz_close, 1)) for r in tr.itertuples()])

    q3b()
    q2(R + NEW)
    hum = human_closes()
    q4(P, hum)
    q5(P, hum)


if __name__ == "__main__":
    main()

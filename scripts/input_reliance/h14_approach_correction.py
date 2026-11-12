"""H14: after the coarse turn, does the arm still correct its lateral aim before the first close? Humans vs models,
paper cup and plastic bottle, from recorded joint states only (no input edits). (2026-11-12)

Per episode (first gripper close k = aim_error.grasp_joints rule, 0.6 of the depth, depth >= 10):
  theta(t)  = arm angle on the table from the base joint (human cup fit h8.C[0], linear in shoulder_pan; lateral only,
              because r is fitted on close poses and is not valid while the arm is still up)
  settle    = first frame where the base is within NEAR = 15 deg of the object AND its angular speed (7-frame central
              difference) stays < 3 deg/s for 8 frames (0.5 s), before k. Episodes that never get there (coarse turn
              failed, e.g. c17 / c36 to the wrong side) are left out of this part.
              v1 (first run 11-12, seen before this change) took the first slow 0.5 s after the arm started reaching:
              it caught pauses inside two-stage coarse turns (far_wrist ep0 paused at +16 deg on the way to +70) and
              counted the rest of the turn as correction -> human slope ~1 by construction. Reading rule unchanged.
  last Ls   = second view without the settle rule: e and d over the last L = 1 / 2 s before k, only where the base is
              already slow (< 3 deg/s) at k - L
  e0        = cup/bottle theta - theta(settle)   lateral aim error left after the coarse turn
  d         = theta(k) - theta(settle)           lateral move between settle and close
  both converted to cm at the object's r (r * radians).
Correction slope = OLS slope of d on e0 (with intercept), 95% CI by bootstrap over episodes:
  1 = whatever error was left after the coarse turn is removed before closing, 0 = no lateral move after the coarse turn.
  Placement / fit noise in e0 pulls every slope toward 0, for humans and models alike.
Reading rule, fixed before the run: per object, a model group corrects less than humans if its slope CI lies below the
human slope CI; overlapping CIs -> no difference shown. Secondary: median |d| (how much lateral move after settling).
Caveats: lateral only; uvc_60 is the A2 view and the set the theta fit was made on; settle may also catch a pause
inside a two-stage coarse turn (same definition for humans and models); recovery / tipped demos are left out.
Writes h14_*.csv / png in data/logs/input_reliance.

    uv run python scripts/input_reliance/h14_approach_correction.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, "scripts/input_reliance")
sys.path.insert(0, "scripts")
import h8_fl_black_all_cup as H
import aim_error as A

PREFIX = str(A.REPO / "configs/placements/campA_136sym_20260908_20260908_")
HC = ".cache/lerobot/"
R = H.ROOT
SETS = [  # object, group, label, dataset root, episode_meta csv
    ("cup", "human", "uvc_60 (A2)", HC + "omx_pick_place_pilot_uvc_60", "omx_pick_place_pilot_uvc_60.csv"),
    ("cup", "human", "normal_A1", HC + "omx_pick_place_pilot_paper_cup_normal_A1", "omx_pick_place_pilot_paper_cup_normal_A1.csv"),
    ("cup", "human", "far_wrist_A1", HC + "omx_pick_place_pilot_paper_cup_far_wrist_A1", "omx_pick_place_pilot_paper_cup_far_wrist_A1.csv"),
] + [("cup", "mfull" if "mfull" in rl else "single", rl, R + d, csv or d + ".csv") for rl, d, csv in H.ROLLOUTS] + [
    ("cup", "mfull", "11-11 mfull", R + "rollout_omx_mfull_cup129_bottle135_augfl_fldrop30_200k_nas1_te-001_30s_paper_cup_20261111_163453", None),
    ("bottle", "human", "bottle normal_A1", HC + "omx_pick_place_pilot_plastic_bottle_normal_A1", "omx_pick_place_pilot_plastic_bottle_normal_A1.csv"),
    ("bottle", "human", "bottle precise_A1", HC + "omx_pick_place_pilot_plastic_bottle_precise_A1", "omx_pick_place_pilot_plastic_bottle_precise_A1.csv"),
    ("bottle", "single", "10-18 b60 nas30", R + "rollout_omx_b4a_plastic_bottle_60_augfl_fldrop30_100k_nas30_20261018_161002", None),
    ("bottle", "single", "10-22 b136 te", R + "rollout_omx_b4a_plastic_bottle_136_augfl_fldrop30_100k_nas1_te-001_20261022_104207", None),
    ("bottle", "single", "10-28 b60 te", R + "rollout_omx_b4a_plastic_bottle_60_augfl_fldrop30_100k_nas1_te-001_20261028_213634", None),
    ("bottle", "mfull", "11-11 mfull", R + "rollout_omx_mfull_cup129_bottle135_augfl_fldrop30_200k_nas1_te-001_plastic_bottle_20261111_175328", None),
]
OUT_ROWS = H.OUT / "h14_approach_correction.csv"
OUT_SUM = H.OUT / "h14_approach_correction_summary.csv"
OUT_PNG = H.OUT / "h14_approach_correction.png"
FPS = 15
NEAR = 15     # deg: coarse alignment counts as done within this of the object (about 5-9 cm at r = 18-35 cm)
SLOW = 3      # deg/s
LAGS = (1, 2)  # s


def meta(csv, root):
    m = pd.read_csv(f"episode_meta/{csv or Path(root).name + '.csv'}", encoding="utf-8-sig").dropna(subset=["placement_id", "episode_index"])
    m = m[m.placement_id.astype(str).str.strip().str.match(r"^[tco]\d+$")]
    m = m[pd.to_numeric(m.valid, errors="coerce").fillna(1) != 0]
    tmp = H.OUT / "_h14_meta.csv"; m.to_csv(tmp, index=False, encoding="utf-8-sig")
    return A.placements(str(tmp), PREFIX)


def episode(s, a, th_obj, r_obj):
    g = a[:, 5]; depth = g[0] - g.min()
    if depth < 10:
        return None
    k = int(np.argmax(g < g[0] - 0.6 * depth))
    th = np.polyval(H.C[0], s[:, 0])
    sp = np.full(len(th), np.inf)
    sp[3:-3] = np.abs(th[6:] - th[:-6]) / (6 / FPS)
    slow = sp < SLOW
    cm = lambda deg: r_obj * np.deg2rad(deg)
    out = dict(k=k, th_close=th[k], e_close=cm(th_obj - th[k]))
    st = None  # coarse turn done: base within NEAR deg of the object and slow for 0.5 s
    for t in range(0, k - 8):
        if abs(th[t] - th_obj) <= NEAR and slow[t:t + 8].all():
            st = t; break
    if st is not None:
        out.update(settle=st, t_settle_to_close=(k - st) / FPS, th_settle=th[st], e0=cm(th_obj - th[st]), d=cm(th[k] - th[st]))
    for L in LAGS:  # fixed window before the close, only if the base is already slow there
        i = k - int(L * FPS)
        if i >= 3 and slow[i]:
            out[f"e{L}"] = cm(th_obj - th[i]); out[f"d{L}"] = cm(th[k] - th[i])
    return out


def slope(e0, d, rng, nb=2000):
    if len(e0) < 5:
        return np.nan, np.nan, np.nan
    b = np.polyfit(e0, d, 1)[0]
    bs = []
    for _ in range(nb):
        i = rng.integers(0, len(e0), len(e0))
        if np.ptp(e0[i]) > 0:
            bs.append(np.polyfit(e0[i], d[i], 1)[0])
    return b, np.quantile(bs, .025), np.quantile(bs, .975)


def main() -> None:
    rows = []
    for obj, grp, label, root, csv in SETS:
        m = meta(csv, root)
        files = sorted(Path(root, "data").rglob("*.parquet"))
        t = pd.concat([pq.read_table(f, columns=["episode_index", "action", "observation.state"]).to_pandas() for f in files])
        a = np.stack(t["action"].values); s = np.stack(t["observation.state"].values); ep = t.episode_index.values
        skipped = 0
        for r in m.itertuples():
            i = np.flatnonzero(ep == r.episode_index)
            e = episode(s[i], a[i], r.th, r.r) if len(i) else None
            if e is None:
                skipped += 1; continue  # gripper never closed
            rows.append(dict(object=obj, group=grp, set=label, ep=r.episode_index, pid=r.placement_id,
                             outcome=getattr(r, "outcome", ""), th=r.th, r=r.r, **e))
        n = sum(1 for x in rows if x["set"] == label and x["object"] == obj)
        ns = sum(1 for x in rows if x["set"] == label and x["object"] == obj and "e0" in x)
        print(f"{obj:6s} {grp:6s} {label:22s}: {n} with a close (skipped {skipped}), {ns} reached within {NEAR} deg and settled", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT_ROWS, index=False)
    rng = np.random.default_rng(0)
    out = []
    for keys, g in list(d.groupby(["object", "group"])) + list(d.groupby(["object", "group", "set"])):
        row = dict(object=keys[0], group=keys[1], set=keys[2] if len(keys) > 2 else "(pooled)", n=len(g),
                   med_abs_e_close=g.e_close.abs().median())
        for tag, e, dd in [("settle", "e0", "d")] + [(f"last{L}s", f"e{L}", f"d{L}") for L in LAGS]:
            h = g.dropna(subset=[e, dd])
            b_, lo, hi = slope(h[e].values, h[dd].values, rng)
            row.update({f"{tag}_n": len(h), f"{tag}_slope": b_, f"{tag}_lo": lo, f"{tag}_hi": hi,
                        f"{tag}_med_abs_e": h[e].abs().median(), f"{tag}_med_abs_d": h[dd].abs().median()})
        row["med_settle_to_close_s"] = g.t_settle_to_close.median()
        out.append(row)
    sm = pd.DataFrame(out)
    sm.to_csv(OUT_SUM, index=False)
    pd.set_option("display.width", 220)
    for tag in ["settle"] + [f"last{L}s" for L in LAGS]:
        c = ["object", "group", "set", "n", f"{tag}_n", f"{tag}_slope", f"{tag}_lo", f"{tag}_hi", f"{tag}_med_abs_e", f"{tag}_med_abs_d"]
        print(f"\n== {tag}"); print(sm[c + (["med_settle_to_close_s", "med_abs_e_close"] if tag == "settle" else [])].round(2).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
    col = {"human": "tab:green", "single": "tab:blue", "mfull": "tab:red"}
    for a_, obj in zip(ax, ("cup", "bottle")):
        g = d[d.object == obj].dropna(subset=["e0", "d"])
        for grp, gg in g.groupby("group"):
            a_.scatter(gg.e0, gg.d, s=14, alpha=.6, c=col[grp], label=f"{grp} (n={len(gg)})")
        lim = max(3, np.quantile(np.abs(g[["e0", "d"]].values), .98))
        x = np.array([-lim, lim])
        a_.plot(x, x, "k--", lw=.8, label="full correction (d = e0)"); a_.axhline(0, c="k", lw=.5)
        a_.set_xlim(-lim, lim); a_.set_ylim(-lim, lim)
        a_.set_xlabel("lateral error left after coarse turn e0 [cm]"); a_.set_ylabel("lateral move settle -> close d [cm]")
        a_.set_title(obj); a_.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=120)
    print(f"rows: {OUT_ROWS}\nsummary: {OUT_SUM}\nplot: {OUT_PNG}")


if __name__ == "__main__":
    main()

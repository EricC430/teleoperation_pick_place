"""H15: radial (how far the arm reaches) aim and late radial correction, humans vs models, plastic bottle (+ paper cup
as a method check), from recorded joint states only. (2026-11-12)

Why: 11-11 bottle failures cluster far (r >= 34 cm), about 2 cm short (worklog 11-11 §5, joint -> table fit made on the
policy's own successes because the human bottle demos were thought to be absent; they are in .cache/lerobot).
h14 looked at lateral only. Question: do humans correct the reach in the last seconds and models not (-> correction
demos make sense), or is the model's reach already short / compressed (-> the coarse reach itself)?

Radius: FK fingertip radius about the pan axis (measure_pan_circle.tip_polar_in_arm_plane, the S7/S8 constants), valid
at any arm height, unlike the close-pose fit of aim_error. One offset per object = median(r_placement - r_fk at first
close) over that object's human demos (FK zeros / where the object is held vs the fingertip); its spread is printed.
  r_hat(t) = r_fk(t) + offset;  first close k = aim_error.grasp_joints rule (0.6 of the depth, depth >= 10)
Per episode: e_close = r_placement - r_hat(k)   (> 0 = short)
             last L s (L = 1 / 2 / 3): e_L = r_placement - r_hat(k - L), d_L = r_hat(k) - r_hat(k - L)
Group stats (bootstrap 95% CI over episodes):
  tracking slope = OLS slope of r_hat(k) on r_placement (1 = reach follows the object, < 1 = compressed: short far, long near)
  median e_close, near (r < 34) / far (r >= 34)
  correction slope_L = OLS slope of d_L on e_L (1 = whatever gap was left L s before closing is closed)
Reading rule, fixed before the run:
  R1  bottle models are short far relative to humans if (model far median e_close - human far median e_close) > 1 cm
      with its bootstrap CI above 0.
  R2  models correct the reach less than humans in the last L s if the model correction slope CI lies below the human CI.
  Humans correct and models not -> correction demos have a basis; humans do not correct either, or models ~ humans ->
  the problem is in the coarse reach, correction demos would not help.
Caveats: placement r is where the object was put, not measured; fingertip radius, not the grasp centre (the wrist pitch
changes where the fingers meet the object); bottle humans are normal_A1 + precise_A1 (Eric, 10/21), not the model's
training set alone; first close only. Writes h15_*.csv / png in data/logs/input_reliance.

    uv run python scripts/input_reliance/h15_radial_reach.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, "scripts/input_reliance")
sys.path.insert(0, "scripts")
import h14_approach_correction as P14
import measure_pan_circle as PC
from measure_link_tilt import JOINTS

H = P14.H
SETS = P14.SETS
FPS, LAGS, FAR = 15, (1, 2, 3), 34
OUT_ROWS = H.OUT / "h15_radial_reach.csv"
OUT_SUM = H.OUT / "h15_radial_reach_summary.csv"
OUT_PNG = H.OUT / "h15_radial_reach.png"


def r_fk(states):
    return np.array([PC.tip_polar_in_arm_plane(dict(zip(JOINTS, x)))[0] for x in states])


def first_close(g):
    depth = g[0] - g.min()
    return None if depth < 10 else int(np.argmax(g < g[0] - 0.6 * depth))


def boot(f, n, rng, nb=2000):
    v = [f(rng.integers(0, n, n)) for _ in range(nb)]
    v = [x for x in v if np.isfinite(x)]
    return np.quantile(v, .025), np.quantile(v, .975)


def ols_slope(x, y):
    return np.polyfit(x, y, 1)[0] if len(x) >= 5 and np.ptp(x) > 0 else np.nan


def main() -> None:
    raw = []
    for obj, grp, label, root, csv in SETS:
        m = P14.meta(csv, root)
        files = sorted(Path(root, "data").rglob("*.parquet"))
        t = pd.concat([pq.read_table(f, columns=["episode_index", "action", "observation.state"]).to_pandas() for f in files])
        a = np.stack(t["action"].values); s = np.stack(t["observation.state"].values); ep = t.episode_index.values
        for r in m.itertuples():
            i = np.flatnonzero(ep == r.episode_index)
            k = first_close(a[i, 5]) if len(i) else None
            if k is None:
                continue
            lo = max(0, k - max(LAGS) * FPS)
            rr = r_fk(s[i[lo:k + 1]])  # frames lo .. k
            row = dict(object=obj, group=grp, set=label, ep=r.episode_index, pid=r.placement_id,
                       outcome=getattr(r, "outcome", ""), th=r.th, r=r.r, k=k, rfk_close=rr[-1])
            for L in LAGS:
                j = k - L * FPS - lo
                row[f"rfk_m{L}"] = rr[j] if j >= 0 else np.nan
            raw.append(row)
        print(f"{obj:6s} {grp:6s} {label:22s}: {sum(1 for x in raw if x['set'] == label and x['object'] == obj)} episodes with a close", flush=True)
    d = pd.DataFrame(raw)
    for obj, g in d[d.group == "human"].groupby("object"):
        off = (g.r - g.rfk_close).median()
        res = (g.r - g.rfk_close) - off
        b = ols_slope(g.r.values, g.rfk_close.values)
        print(f"{obj}: offset r_placement - r_fk(close) over {len(g)} human demos = {off:+.1f} cm; residual median |.| "
              f"{res.abs().median():.1f} cm, p90 {res.abs().quantile(.9):.1f}; human tracking slope {b:.2f}")
        d.loc[d.object == obj, "off"] = off
    d["e_close"] = d.r - (d.rfk_close + d.off)
    for L in LAGS:
        d[f"e{L}"] = d.r - (d[f"rfk_m{L}"] + d.off)
        d[f"d{L}"] = d.rfk_close - d[f"rfk_m{L}"]
    d["zone"] = np.where(d.r >= FAR, "far", "near")
    d.to_csv(OUT_ROWS, index=False)

    rng = np.random.default_rng(0)
    out = []
    for keys, g in list(d.groupby(["object", "group"])) + list(d.groupby(["object", "group", "set"])):
        g = g.reset_index(drop=True)
        row = dict(object=keys[0], group=keys[1], set=keys[2] if len(keys) > 2 else "(pooled)", n=len(g))
        x, y = g.r.values, (g.rfk_close + g.off).values
        row["track_slope"] = ols_slope(x, y)
        row["track_lo"], row["track_hi"] = boot(lambda i: ols_slope(x[i], y[i]), len(g), rng)
        for z in ("near", "far"):
            e = g.e_close[g.zone == z].values
            row[f"n_{z}"] = len(e)
            row[f"e_close_{z}"] = np.median(e) if len(e) else np.nan
        for L in LAGS:
            h = g.dropna(subset=[f"e{L}"])
            ex, dx = h[f"e{L}"].values, h[f"d{L}"].values
            row[f"corr{L}"] = ols_slope(ex, dx)
            row[f"corr{L}_lo"], row[f"corr{L}_hi"] = boot(lambda i: ols_slope(ex[i], dx[i]), len(h), rng) if len(h) >= 5 else (np.nan, np.nan)
            row[f"med_abs_e{L}"] = np.median(np.abs(ex)) if len(ex) else np.nan
        out.append(row)
    sm = pd.DataFrame(out)
    # R1: far-zone difference model - human, per object, bootstrap
    r1 = []
    for obj in d.object.unique():
        hf = d[(d.object == obj) & (d.group == "human") & (d.zone == "far")].e_close.values
        for grp in ("single", "mfull"):
            mf = d[(d.object == obj) & (d.group == grp) & (d.zone == "far")].e_close.values
            if len(hf) < 3 or len(mf) < 3:
                continue
            diff = np.median(mf) - np.median(hf)
            bs = [np.median(mf[rng.integers(0, len(mf), len(mf))]) - np.median(hf[rng.integers(0, len(hf), len(hf))]) for _ in range(2000)]
            r1.append(dict(object=obj, model=grp, n_human_far=len(hf), n_model_far=len(mf), human_far=np.median(hf),
                           model_far=np.median(mf), diff=diff, lo=np.quantile(bs, .025), hi=np.quantile(bs, .975)))
    sm.to_csv(OUT_SUM, index=False)
    pd.set_option("display.width", 240)
    c0 = ["object", "group", "set", "n", "track_slope", "track_lo", "track_hi", "n_near", "e_close_near", "n_far", "e_close_far"]
    print("\n== reach at first close (e_close > 0 = short)"); print(sm[c0].round(2).to_string(index=False))
    c1 = ["object", "group", "set"] + [c for L in LAGS for c in (f"corr{L}", f"corr{L}_lo", f"corr{L}_hi", f"med_abs_e{L}")]
    print("\n== late radial correction (slope of d_L on e_L)"); print(sm[c1].round(2).to_string(index=False))
    print("\n== R1 far zone: model median e_close - human median e_close [cm]"); print(pd.DataFrame(r1).round(2).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
    col = {"human": "tab:green", "single": "tab:blue", "mfull": "tab:red"}
    for a_, obj in zip(ax, ("bottle", "cup")):
        g = d[d.object == obj]
        jit = np.random.default_rng(1).normal(0, .25, len(g))
        for grp, gg in g.groupby("group"):
            a_.scatter(gg.r + jit[g.group.values == grp], gg.e_close, s=14, alpha=.6, c=col[grp], label=f"{grp} (n={len(gg)})")
        a_.axhline(0, c="k", lw=.6); a_.axvline(FAR, c="gray", ls=":", lw=.8)
        a_.set_xlabel("placement r [cm]"); a_.set_ylabel("r_placement - r_hat at first close [cm]  (> 0 = short)")
        a_.set_title(f"{obj}: reach error at first close"); a_.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=120)
    print(f"rows: {OUT_ROWS}\nsummary: {OUT_SUM}\nplot: {OUT_PNG}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Where did the policy aim? Grasp-moment position error of a closed-loop rollout (D032 §3 method).

The forward kinematics has a known constant offset (episode_meta/README.md), so instead of FK this
fits the human demos: at each demo's grasp frame, `shoulder_pan -> theta` and
`shoulder_lift/elbow_flex/wrist_flex -> r`, against the demo's recorded placement. That map is then
applied to the rollout's grasp frames and compared with where the object actually was.

Grasp frame = first frame the gripper COMMAND is CLOSE_FRAC of the way from its first value to its
episode minimum. Episodes whose gripper never closes by MIN_DEPTH (stalled) are skipped.

Placements come from each CSV's `placement_id` (short ids: t = train, o = eval-open, c = eval-close),
so the episode <-> placement mapping lives in episode_meta/ only.

    uv run python scripts/aim_error.py \\
        --rollout D:/hf/lerobot/ericc430/rollout_omx_b1_uvc60_100k_nas30_A1_paper_cup_20261006_094908 \\
        --meta episode_meta/rollout_omx_b1_uvc60_100k_nas30_A1_paper_cup_20261006_close_loop.csv

Writes <out>/<rollout name>.png (map + radial + lateral error) and .csv (per episode), and prints a summary.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
PREFIX = {"t": "train", "o": "eval-open", "c": "eval-close"}

# reference palette (dataviz skill): slot 1 blue / slot 2 orange; outcome also carried by marker shape
OK_C, FAIL_C, HUMAN_C, INK, MUTED = "#2a78d6", "#eb6834", "#a3a29a", "#21211f", "#6b6a63"
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def args_():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rollout", required=True, help="rollout dataset root (has data/ and meta/)")
    p.add_argument("--meta", required=True, help="episode_meta CSV of that rollout (placement_id, outcome)")
    p.add_argument("--train-root", default=str(REPO / ".cache/lerobot/omx_pick_place_pilot_uvc_60"))
    p.add_argument("--train-meta", default=str(REPO / "episode_meta/omx_pick_place_pilot_uvc_60.csv"))
    p.add_argument("--placements", default=str(REPO / "configs/placements/campA_136sym_20260908_20260908_"),
                   help="prefix of the <list>.csv placement files")
    p.add_argument("--close-frac", type=float, default=0.6)
    p.add_argument("--min-depth", type=float, default=10.0, help="gripper must close at least this many .pos")
    p.add_argument("--out", default=str(REPO / "outputs/aim_error"))
    return p.parse_args()


def grasp_joints(root, close_frac, min_depth):
    files = sorted(Path(root, "data").rglob("*.parquet"))
    t = pd.concat([pq.read_table(f, columns=["episode_index", "action", "observation.state"]).to_pandas() for f in files])
    a = np.stack(t["action"].values); s = np.stack(t["observation.state"].values); ep = t.episode_index.values
    out = {}
    for e in np.unique(ep):
        i = np.where(ep == e)[0]; g = a[i, 5]; depth = g[0] - g.min()
        if depth >= min_depth:
            out[int(e)] = s[i[np.argmax(g < g[0] - close_frac * depth)], :5]
    return out


def placements(meta_csv, prefix):
    m = pd.read_csv(meta_csv, encoding="utf-8-sig").dropna(subset=["episode_index"])
    m["episode_index"] = m.episode_index.astype(int)
    lists = {k: pd.read_csv(f"{prefix}{v}.csv").set_index("placement_id") for k, v in PREFIX.items()}
    def look(pid):
        pid = str(pid).strip(); k, n = pid[0], int(pid[1:])
        return lists[k].loc[f"{PREFIX[k]}_{n:03d}"]
    p = m.placement_id.map(look)
    m["th"] = [x.theta_deg for x in p]; m["r"] = [x.r_cm for x in p]
    return m


def rdesign(J):
    return np.c_[np.ones(len(J)), J[:, 1], J[:, 2], J[:, 3]]


def fit(J, th, r):
    return np.polyfit(J[:, 0], th, 1), np.linalg.lstsq(rdesign(J), r, rcond=None)[0]


def predict(c, J):
    ct, cr = c
    return np.polyval(ct, J[:, 0]), rdesign(J) @ cr


def main():
    a = args_()
    # ---- human map
    tm = placements(a.train_meta, a.placements)
    tg = grasp_joints(a.train_root, a.close_frac, a.min_depth)
    tm = tm[tm.episode_index.isin(tg)]
    J = np.array([tg[e] for e in tm.episode_index]); th, r = tm.th.values, tm.r.values
    c = fit(J, th, r)
    corr = np.corrcoef(J[:, 0], th)[0, 1]
    print(f"human map on {len(J)} demos: theta = {c[0][0]:.3f}*pan {c[0][1]:+.2f}   corr(pan, theta) = {corr:.3f}")
    if corr < 0.9:
        print("  !! corr < 0.9 -- the demos' placement_id is probably wrong; the map is not trustworthy")
    loo = []
    for k in range(len(J)):
        mk = np.arange(len(J)) != k
        pt, pr = predict(fit(J[mk], th[mk], r[mk]), J[k:k + 1])
        loo.append((th[k], r[k], pt[0] - th[k], pr[0] - r[k]))
    h = pd.DataFrame(loo, columns=["th", "r", "dth", "dr"]); h["lat"] = np.deg2rad(h.dth) * h.r

    # ---- rollout
    rm = placements(a.meta, a.placements)
    rg = grasp_joints(a.rollout, a.close_frac, a.min_depth)
    skipped = sorted(set(rm.episode_index) - set(rg))
    d = rm[rm.episode_index.isin(rg)].copy()
    pt, pr = predict(c, np.array([rg[e] for e in d.episode_index]))
    d["th_hat"], d["r_hat"] = pt, pr
    d["dth"] = d.th_hat - d.th; d["dr"] = d.r_hat - d.r; d["lat"] = np.deg2rad(d.dth) * d.r
    d["err_cm"] = np.hypot(d.lat, d.dr); d["ok"] = d.outcome == "success"
    for col, src in [("x", "th"), ("x_hat", "th_hat")]:
        rr = d.r if col == "x" else d.r_hat
        d[col] = rr * np.cos(np.deg2rad(d[src])); d["y" + col[1:]] = rr * np.sin(np.deg2rad(d[src]))

    name = Path(a.rollout).name
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d.drop(columns=[x for x in d.columns if x.startswith("Unnamed")]).to_csv(out / f"{name}.csv", index=False)

    slope = np.polyfit(d.r, d.r_hat, 1)[0]
    print(f"\n{name}: {len(d)} grasps; skipped (gripper never closed) {skipped}")
    h_slope = np.polyfit(h.r, h.r + h.dr, 1)[0]
    print(f"  human LOO floor : |lateral| {h.lat.abs().median():.2f} cm, |dr| {h.dr.abs().median():.2f} cm, "
          f"radial slope {h_slope:.2f} (the method's own pull toward the middle)")
    print(f"  rollout         : |lateral| {d.lat.abs().median():.2f} cm, |dr| {d.dr.abs().median():.2f} cm, "
          f"bias dr {d.dr.median():+.2f} cm, dtheta {d.dth.median():+.2f} deg")
    print(f"  radial slope r_hat~r {slope:.2f} (1 = no pull toward the middle); spearman(dr, r) {spearmanr(d.r, d.dr)[0]:+.2f}")
    print(f"  total error success {d.err_cm[d.ok].median():.2f} cm (n={d.ok.sum()}) | fail {d.err_cm[~d.ok].median():.2f} cm (n={(~d.ok).sum()})")

    # ---- figure
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.8), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    m0 = ax[0]
    for rad in (22, 33):
        tt = np.deg2rad(np.linspace(-70, 70, 200))
        m0.plot(rad * np.sin(tt), rad * np.cos(tt), ls="--", lw=1, color=HUMAN_C, zorder=0)
        m0.text(0, rad + 0.6, f"r={rad}", color=MUTED, fontsize=8, ha="center")
    m0.plot(0, 0, marker="s", color=INK, ms=8); m0.text(1.5, -0.8, "底座（pan 軸）", fontsize=8, color=MUTED)
    for ok, col, mk, lab in [(True, OK_C, "o", "成功"), (False, FAIL_C, "^", "失敗")]:
        x = d[d.ok == ok]
        # screen x = -y so the robot's left (+y) is on the left, forward is up
        m0.scatter(-x.y, x.x, s=70, facecolors="none", edgecolors=col, linewidths=2, marker=mk, zorder=3,
                   label=f"{lab}：實際位置（空心）→ 瞄準點（實心）")
        m0.scatter(-x.y_hat, x.x_hat, s=28, color=col, marker=mk, zorder=4, edgecolors="white", linewidths=1)
        for _, rw in x.iterrows():
            m0.annotate("", xy=(-rw.y_hat, rw.x_hat), xytext=(-rw.y, rw.x),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4, shrinkA=4, shrinkB=3))
            m0.text(-rw.y + 0.6, rw.x + 0.6, str(rw.placement_id), fontsize=7, color=MUTED)
    m0.set_aspect("equal"); m0.set_xlabel("← 左（+y）　　橫向 (cm)　　右（-y）→"); m0.set_ylabel("往前 x (cm)")
    m0.set_title("瞄準偏移（俯視，箭頭 = 實際 → 瞄準）", loc="left", fontsize=11)
    m0.legend(loc="lower left", fontsize=8, frameon=False, ncol=2)
    xs = np.r_[-d.y, -d.y_hat]; ys = np.r_[d.x, d.x_hat]
    m0.set_xlim(xs.min() - 4, xs.max() + 4); m0.set_ylim(-11, ys.max() + 4)

    for axi, ycol, xcol, xl, title in [(ax[1], "dr", "r", "實際距離 r (cm)", "徑向誤差（+ = 伸太遠）"),
                                       (ax[2], "lat", "th", "實際角度 θ (deg，+ = 左)", "橫向誤差（+ = 偏左）")]:
        axi.axhline(0, color=INK, lw=1)
        axi.scatter(h[xcol], h[ycol], s=14, color=HUMAN_C, label=f"人類示範 LOO（n={len(h)}，方法底線）", zorder=1)
        for ok, col, mk, lab in [(True, OK_C, "o", "成功"), (False, FAIL_C, "^", "失敗")]:
            x = d[d.ok == ok]
            axi.scatter(x[xcol], x[ycol], s=55, color=col, marker=mk, edgecolors="white", linewidths=1.5, label=lab, zorder=3)
        if ycol == "dr":
            xx = np.linspace(d.r.min(), d.r.max(), 10); k = np.polyfit(d.r, d.dr, 1)
            axi.plot(xx, np.polyval(k, xx), color=INK, lw=1.5, ls=":",
                     label=f"rollout 趨勢（r_hat~r 斜率 {slope:.2f}）")
            kh = np.polyfit(h.r, h.dr, 1); xh = np.linspace(h.r.min(), h.r.max(), 10)
            axi.plot(xh, np.polyval(kh, xh), color=HUMAN_C, lw=1.5, ls="--", label=f"人類 LOO 趨勢（斜率 {h_slope:.2f}）")
        axi.set_xlabel(xl); axi.set_ylabel("cm"); axi.set_title(title, loc="left", fontsize=11)
        axi.grid(alpha=0.25, lw=0.6); axi.legend(fontsize=8, frameon=False)
    for axi in ax:
        for sp in ("top", "right"):
            axi.spines[sp].set_visible(False)
    fig.suptitle(f"{name}　｜　{len(d)} 次抓取（略過 {len(skipped)} 集沒閉合）　｜　human map n={len(J)}, corr(pan,θ)={corr:.2f}",
                 fontsize=10, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=130)
    print(f"\nsaved {out / (name + '.png')}\n      {out / (name + '.csv')}")


if __name__ == "__main__":
    main()

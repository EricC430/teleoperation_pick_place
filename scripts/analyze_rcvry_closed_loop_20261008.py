#!/usr/bin/env python
"""Analysis behind docs/meeting/2026-10-08.md: recovery-model closed loop vs 10-06, and gripper retries vs demos.

Part 1: success by run / paired placements / position band / distance to nearest training placement.
Part 2: gripper close events (retries) per episode in closed-loop runs and in the demos.
Laptop paths (D:/hf, .cache/lerobot). No robot, no GPU.

    uv run python scripts/analyze_rcvry_closed_loop_20261008.py
"""
import csv, glob, io, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

R = Path("D:/teleoperation_pick_place")
M = R / "episode_meta"
RUNS = {
    "1006_nas30": "rollout_omx_b1_uvc60_100k_nas30_A1_paper_cup_20261006_close_loop",
    "1007_wristcov": "rollout_omx_b1_uvc60_100k_nas30_A1_wristcovered_paper_cup_20261007_close_loop",
    "1007_te": "rollout_omx_b1_uvc60_100k_A1_temperal_ensemble_paper_cup_20261007_close_loop",
    "1008_rcvry": "rollout_omx_b1_papercup_60_16_12rcvry_100k_paper_cup_20261008_171620",
}
DS1008 = Path("D:/hf/lerobot/ericc430/rollout_omx_b1_papercup_60_16_12rcvry_100k_paper_cup_20261008_171620")
DS1006 = Path("D:/hf/lerobot/ericc430/rollout_omx_b1_uvc60_100k_nas30_A1_paper_cup_20261006_094908")


def read(name):
    raw = (M / f"{name}.csv").read_bytes()
    for enc in ("utf-8-sig", "cp950"):
        try:
            t = raw.decode(enc); break
        except UnicodeDecodeError:
            pass
    return [r for r in csv.DictReader(io.StringIO(t)) if (r["episode_index"] or "").isdigit()]


pl = {r["short_id"]: r for r in csv.DictReader(open(R / "docs/assets/placement_label_map_campA_136sym_20260908.csv", encoding="utf-8"))}
for r in pl.values():
    r["x"], r["y"] = float(r["x_pan_cm"]), float(r["y_pan_cm"])
    r["r"] = math.hypot(r["x"], r["y"]); r["th"] = math.degrees(math.atan2(r["y"], r["x"]))

def rbin(r): return "near<22" if r < 22 else ("mid22-33" if r < 33 else "far>=33")
def tbin(t): t = abs(t); return "<20" if t < 20 else ("20-40" if t < 40 else ">=40")

train_uvc60 = [r["placement_id"] for r in read("omx_pick_place_pilot_uvc_60")]
normal = [r["placement_id"] for r in read("omx_pick_place_pilot_paper_cup_normal_A1") if r["valid"] == "1"]
tight = [r["placement_id"] for r in read("omx_pick_place_pilot_paper_cup_recovery_A1_tight") if r["valid"] == "1"]
new_demo = normal + tight
all_train = train_uvc60 + new_demo

def nearest(sid, pool):
    p = pl[sid]
    return min(math.hypot(p["x"] - pl[q]["x"], p["y"] - pl[q]["y"]) for q in pool)

runs = {k: {r["placement_id"]: r for r in read(v)} for k, v in RUNS.items()}
print("== overall (valid=1)")
for k, rows in runs.items():
    v = [r for r in rows.values() if r["valid"] != "0"]
    s = sum(r["outcome"] == "success" for r in v)
    mech = {}
    for r in rows.values():
        for m in filter(None, r["mechanism"].split(";")):
            mech[m] = mech.get(m, 0) + 1
    outc = pd.Series([r["outcome"] for r in rows.values()]).value_counts().to_dict()
    print(f"{k:14s} {s}/{len(v)} = {s/len(v):.0%}  invalid={[r['placement_id'] for r in rows.values() if r['valid']=='0']}  outcome={outc}  mech={mech}")

a, b = runs["1006_nas30"], runs["1008_rcvry"]
common = [c for c in a if a[c]["valid"] != "0" and b[c]["valid"] != "0"]
both = sum(a[c]["outcome"] == "success" and b[c]["outcome"] == "success" for c in common)
only_old = [c for c in common if a[c]["outcome"] == "success" and b[c]["outcome"] != "success"]
only_new = [c for c in common if a[c]["outcome"] != "success" and b[c]["outcome"] == "success"]
p_mc = stats.binomtest(len(only_new), len(only_new) + len(only_old)).pvalue
print(f"\n== paired vs 10-06 on {len(common)} common valid placements: both {both}, only 10-06 {only_old}, only 10-08 {only_new}, McNemar exact p={p_mc:.3f}")
f = stats.fisher_exact([[22, 34 - 22], [17, 32 - 17]])
print(f"   unpaired Fisher 22/34 vs 17/32 p={f.pvalue:.3f}")

print("\n== by position (valid=1): 10-06 -> 10-08")
tab = {}
for c in common + [c for c in b if c not in common and b[c]["valid"] != "0"]:
    p = pl[c]; key = (rbin(p["r"]), tbin(p["th"]))
    t = tab.setdefault(key, [0, 0, 0, 0])
    if c in a and a[c]["valid"] != "0":
        t[0] += a[c]["outcome"] == "success"; t[1] += 1
    if b[c]["valid"] != "0":
        t[2] += b[c]["outcome"] == "success"; t[3] += 1
for rb in ["near<22", "mid22-33", "far>=33"]:
    cells = [f"{tb}: {tab.get((rb,tb),[0]*4)[0]}/{tab.get((rb,tb),[0]*4)[1]} -> {tab.get((rb,tb),[0]*4)[2]}/{tab.get((rb,tb),[0]*4)[3]}" for tb in ["<20", "20-40", ">=40"]]
    tot = [sum(tab.get((rb, tb), [0]*4)[i] for tb in ["<20", "20-40", ">=40"]) for i in range(4)]
    print(f"{rb:9s} | " + " | ".join(cells) + f" || total {tot[0]}/{tot[1]} -> {tot[2]}/{tot[3]}")

print("\n== per placement (valid in 10-08)")
rows = []
for c, r in b.items():
    p = pl[c]
    rows.append(dict(c=c, r=round(p["r"], 1), th=round(p["th"], 1), band=rbin(p["r"]),
                     d_any=round(nearest(c, all_train), 2), d_uvc60=round(nearest(c, train_uvc60), 2),
                     d_new=round(nearest(c, new_demo), 2), d_rcv=round(nearest(c, tight), 2),
                     old=a[c]["outcome"], new=r["outcome"], valid=r["valid"] or "1", mech=r["mechanism"]))
df = pd.DataFrame(rows)
print(df.to_string(index=False))
v = df[df.valid != "0"]
for col in ["d_any", "d_new", "d_rcv"]:
    s, f_ = v[v.new == "success"][col], v[v.new != "success"][col]
    print(f"{col}: success median {s.median():.2f} (n={len(s)}) vs fail {f_.median():.2f} (n={len(f_)})  MWU p={stats.mannwhitneyu(s, f_).pvalue:.3f}")
gain = v[v.c.isin(only_new)]; loss = v[v.c.isin(only_old)]; same = v[~v.c.isin(only_new + only_old)]
print(f"d_new median: gained {gain.d_new.median():.2f} (n={len(gain)}), lost {loss.d_new.median():.2f} (n={len(loss)}), unchanged {same.d_new.median():.2f} (n={len(same)})")

print("\n== recovery: bad_aim episodes and how many still succeeded")
for k, rows_ in runs.items():
    ba = [r for r in rows_.values() if "bad_aim" in r["mechanism"]]
    rec = [r for r in ba if "self_recovered" in r["mechanism"]]
    print(f"{k:14s} bad_aim {len(ba)}, of which self_recovered {len(rec)} (success {sum(r['outcome']=='success' for r in rec)})  self_recovered total {sum('self_recovered' in r['mechanism'] for r in rows_.values())}")

print("\n== episode lengths (frames @15 fps; 20 s = 300)")
for name, ds in [("1006", DS1006), ("1008", DS1008)]:
    if not ds.exists():
        print(name, "dataset missing"); continue
    e = pd.concat(pd.read_parquet(p) for p in glob.glob(str(ds / "meta/episodes/*/*.parquet")))
    L = dict(zip(e.episode_index, e.length))
    if name == "1008":
        df["len"] = [L.get(i) for i in range(len(df))]
    print(name, "lengths:", [L[i] for i in sorted(L)])
print(df[["c", "new", "mech", "len"]].assign(sec=(df["len"] / 15).round(1)).to_string(index=False))


# ---- part 2: gripper close events ----
import glob, json
import numpy as np
import pandas as pd

H = "D:/hf/lerobot/ericc430/"
C = "D:/teleoperation_pick_place/.cache/lerobot/"
SETS = {
    "CL 10-06 (old model, 20 s)": H + "rollout_omx_b1_uvc60_100k_nas30_A1_paper_cup_20261006_094908",
    "CL 10-08 (rcvry model, 30 s)": H + "rollout_omx_b1_papercup_60_16_12rcvry_100k_paper_cup_20261008_171620",
    "demo uvc_60": C + "omx_pick_place_pilot_uvc_60",
    "demo normal_A1": C + "omx_pick_place_pilot_paper_cup_normal_A1",
    "demo recovery_A1 (not trained)": C + "omx_pick_place_pilot_paper_cup_recovery_A1",
    "demo recovery_A1_tight": C + "omx_pick_place_pilot_paper_cup_recovery_A1_tight",
}


def events(g, fps):
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    closed = g < (lo + hi) / 2
    # debounce: a state must last >= 0.2 s
    k = max(1, int(0.2 * fps))
    runs, cur, n = [], closed[0], 0
    for c in closed:
        if c == cur:
            n += 1
        else:
            runs.append((cur, n)); cur, n = c, 1
    runs.append((cur, n))
    merged = []
    for c, n in runs:
        if merged and (n < k or merged[-1][0] == c):
            merged[-1] = (merged[-1][0], merged[-1][1] + n)
        else:
            merged.append((c, n))
    t, out = 0, []
    for c, n in merged:
        if c:
            out.append((t / fps, n / fps))  # (start s, hold s)
        t += n
    return out


rows = []
for name, root in SETS.items():
    fps = json.load(open(root + "/meta/info.json"))["fps"]
    d = pd.concat(pd.read_parquet(p) for p in sorted(glob.glob(root + "/data/*/*.parquet")))
    a = np.stack(d["action"].values)[:, 5]
    for ep in sorted(d.episode_index.unique()):
        g = a[d.episode_index.values == ep]
        ev = events(g, fps)
        rows.append(dict(set=name, ep=int(ep), n_close=len(ev),
                         first_hold=round(ev[0][1], 1) if ev else None,
                         first_t=round(ev[0][0], 1) if ev else None,
                         second_t=round(ev[1][0], 1) if len(ev) > 1 else None))
df = pd.DataFrame(rows)
summ = df.groupby("set", sort=False).agg(
    eps=("ep", "size"), retry_eps=("n_close", lambda s: int((s >= 2).sum())),
    mean_close=("n_close", "mean"), med_first_t=("first_t", "median"), med_second_t=("second_t", "median"))
print(summ.round(2).to_string())
print()
for name in ["demo recovery_A1 (not trained)", "demo recovery_A1_tight", "CL 10-08 (rcvry model, 30 s)"]:
    x = df[(df.set == name) & (df.n_close >= 2)]
    print(name, "first-close hold s (retry eps):", sorted(x.first_hold.tolist()))
print()
print(df[df.set.str.startswith("CL 10-08")][["ep", "n_close", "first_t", "first_hold", "second_t"]].to_string(index=False))

import csv, io
def ann(n):
    t = open(f"D:/teleoperation_pick_place/episode_meta/{n}.csv", "rb").read().decode("utf-8-sig")
    return {int(r["episode_index"]): r for r in csv.DictReader(io.StringIO(t)) if r["episode_index"].isdigit()}
for name, an in [("CL 10-06 (old model, 20 s)", ann("rollout_omx_b1_uvc60_100k_nas30_A1_paper_cup_20261006_close_loop")),
                 ("CL 10-08 (rcvry model, 30 s)", ann("rollout_omx_b1_papercup_60_16_12rcvry_100k_paper_cup_20261008_171620"))]:
    x = df[df.set == name]
    for lab, sub in [("1 close", x[x.n_close == 1]), (">=2 closes", x[x.n_close >= 2])]:
        valid = [e for e in sub.ep if an[e]["valid"] != "0"]
        s = sum(an[e]["outcome"] == "success" for e in valid)
        print(f"{name:30s} {lab:10s}: {s}/{len(valid)} success (valid)")

print("\nrelease time (end of last close) for 10-08 successes / misplaced:")
root = SETS["CL 10-08 (rcvry model, 30 s)"]
d = pd.concat(pd.read_parquet(p) for p in sorted(glob.glob(root + "/data/*/*.parquet")))
a = np.stack(d["action"].values)[:, 5]
an = ann("rollout_omx_b1_papercup_60_16_12rcvry_100k_paper_cup_20261008_171620")
for ep in sorted(an):
    if an[ep]["outcome"] in ("success", "misplaced"):
        ev = events(a[d.episode_index.values == ep], 15)
        print(ep, an[ep]["placement_id"], an[ep]["outcome"], an[ep]["mechanism"], "n_close", len(ev), "release_s", round(ev[-1][0] + ev[-1][1], 1))

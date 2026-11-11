"""H12: h11 for the bottle - does M-full use the wrist view less than the single-object bottle model? (2026-11-12)

Same tests as h11 (wrist gray, wrist swap with a success donor; plan = chunk[D] from frame k - D, k = first close), on the
two bottle closed-loop rollouts with TE -0.01 / 45 s: 10-22 single-object 100k and 11-11 M-full 200k.
Joint -> table mapping: h8.xy is fitted on human *cup* grasps; bottle grasps are higher with another wrist pose, so this
refits the same model (aim_error.fit) on the first-close state of every *successful* episode of the two bottle rollouts
(placement theta, r as target). The human bottle demos with placements (normal_A1 / precise_A1) are not on this machine.
That fit comes from the policy's own successful grasps, not from humans; its leave-one-out error is printed.
Reading rule, fixed before the run: M-full clearly lower than the single-object model on the same frames -> supports
'wrist used less'; about equal -> no change.
Caveats: as h11 (TE on the robot vs raw chunk[D]); bottle placement is where the bottle stood at t=0.

    uv run python scripts/input_reliance/h12_mfull_wrist_bottle.py
"""
import sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, "scripts/input_reliance")
sys.path.insert(0, "scripts")
import h8_fl_black_all_cup as H
import aim_error as A
from huggingface_hub import snapshot_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

W = H.W
MODELS = [
    ("bottle100k", "ericc430/act_omx_b4a_plastic-bottle_60-16-12rcvrycut-37precise-10tipped-augfl-fldrop30_100k"),
    ("mfull200k", "ericc430/act_omx_paper-cup129-plastic-bottle135-augfl-fldrop30_200k"),
]
H.ROLLOUTS = [
    ("10-22 bottle100k", "rollout_omx_b4a_plastic_bottle_136_augfl_fldrop30_100k_nas1_te-001_20261022_104207", None),
    ("11-11 mfull", "rollout_omx_mfull_cup129_bottle135_augfl_fldrop30_200k_nas1_te-001_plastic_bottle_20261111_175328", None),
]
OUT_ROWS = H.OUT / "h12_mfull_wrist_bottle.csv"
OUT_SUM = H.OUT / "h12_mfull_wrist_bottle_summary.csv"
PREFIX = str(A.REPO / "configs/placements/campA_136sym_20260908_20260908_")


def bottle_fit():
    """aim_error.fit on first-close state (same 0.6 / depth 10 rule as h8.load_frames) of successful bottle episodes."""
    J, th, r = [], [], []
    for _, d, csv in H.ROLLOUTS:
        m = A.placements(f"episode_meta/{csv or d + '.csv'}", PREFIX)
        m = m[(pd.to_numeric(m.valid, errors="coerce").fillna(1) != 0) & (m.outcome == "success")]
        g = A.grasp_joints(H.ROOT + d, 0.6, 10)
        for e in m.itertuples():
            if e.episode_index in g:
                J.append(g[e.episode_index]); th.append(e.th); r.append(e.r)
    J, th, r = np.array(J), np.array(th), np.array(r)
    xy = np.c_[r * np.cos(np.deg2rad(th)), r * np.sin(np.deg2rad(th))]
    loo = []
    for i in range(len(J)):
        k = np.arange(len(J)) != i
        c = A.fit(J[k], th[k], r[k]); t1, r1 = A.predict(c, J[i:i + 1])
        loo.append(np.hypot(r1[0] * np.cos(np.deg2rad(t1[0])) - xy[i, 0], r1[0] * np.sin(np.deg2rad(t1[0])) - xy[i, 1]))
    print(f"bottle fit: n={len(J)} successes, leave-one-out error median {np.median(loo):.1f} cm, p90 {np.quantile(loo, .9):.1f} cm", flush=True)
    return A.fit(J, th, r)


def main() -> None:
    C = bottle_fit()

    def xy(j5):
        t, r = A.predict(C, j5[None]); t, r = t[0], r[0]
        return np.array([r * np.cos(np.deg2rad(t)), r * np.sin(np.deg2rad(t))])

    frames = H.load_frames()
    donors = {}
    for i, f in enumerate(frames):
        if f["outcome"] == "success":
            donors.setdefault((f["rollout"], f["D"]), []).append(i)
    rows = []
    for label, repo in MODELS:
        ck = snapshot_download(repo)
        pol = get_policy_class("act").from_pretrained(ck, config=PreTrainedConfig.from_pretrained(ck)).eval().to(H.dev)
        pre, post = make_pre_post_processors(pol.config, pretrained_path=ck)

        def plan(item, D):
            b = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
            b = {k: (v.to(H.dev) if isinstance(v, torch.Tensor) else v) for k, v in pre(b).items()}
            with torch.no_grad():
                return xy(post(pol.predict_action_chunk(b)[0].cpu()).numpy()[D, :5])

        for f in frames:
            x = {k: (v.float() / 255 if k in (W, H.FL) else v) for k, v in f["item"].items()}
            p0 = plan(x, f["D"])
            g = dict(x); g[W] = torch.full_like(x[W], 0.5)
            pg = plan(g, f["D"])
            r = dict(model=label, rollout=f["rollout"], ep=f["ep"], pid=f["pid"], outcome=f["outcome"],
                     bad_aim="bad_aim" in f["mechanism"], D=f["D"],
                     err_orig=np.linalg.norm(p0 - f["cup"]), move_wg=np.linalg.norm(pg - p0),
                     move_sw=np.nan, corr=np.nan, donor_ep=np.nan)
            pool = [j for j in donors.get((f["rollout"], f["D"]), []) if frames[j]["pid"] != f["pid"]]
            if pool:
                j = pool[f["ep"] % len(pool)]
                s = dict(x); s[W] = frames[j]["item"][W].float() / 255
                ps = plan(s, f["D"])
                v = f["cup"] - ps
                r.update(move_sw=np.linalg.norm(ps - p0), corr=(p0 - ps) @ v / (v @ v), donor_ep=frames[j]["ep"])
            rows.append(r)
        print(f"== {label} done ({len(frames)} frames)", flush=True)
        del pol
        torch.cuda.empty_cache()
    out = pd.DataFrame(rows)
    out.to_csv(OUT_ROWS, index=False)
    b = out[out.bad_aim]
    s = out.groupby(["rollout", "model", "D"]).agg(n=("move_wg", "size"), err_orig=("err_orig", "median"),
                                                  move_wg=("move_wg", "median"), move_sw=("move_sw", "median")).join(
        b.groupby(["rollout", "model", "D"]).agg(n_bad=("corr", "count"), corr=("corr", "median"),
                                                 corr_q25=("corr", lambda c: c.quantile(.25)), corr_q75=("corr", lambda c: c.quantile(.75))))
    s.to_csv(OUT_SUM)
    pd.set_option("display.width", 200)
    print(s.round(2).to_string())
    print(f"rows: {OUT_ROWS}\nsummary: {OUT_SUM}")


if __name__ == "__main__":
    main()

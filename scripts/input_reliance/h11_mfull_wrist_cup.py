"""H11: does M-full use the wrist view less than the single-object cup model? Same frames, both models. (2026-11-12)

Combines h8's wrist-gray and h9's wrist-swap on one frame set: h8's ROLLOUTS (10-06 .. 10-17 cup closed loop) plus the
11-11 M-full cup rollout. Plan point, frame choice (k - D, k = first close) and donor rule are h8 / h9 unchanged, so the
augfl rows on the old rollouts must reproduce the 10-18 h8 / h9 logs (pipeline check).
  move_wg   = |plan(wrist -> 0.5 gray) - plan_orig|                      (h8 'wrist gray')
  move_sw   = |plan(wrist -> donor success frame) - plan_orig|           (h9 'move')
  corr      = (plan_orig - plan_swap) . (cup - plan_swap) / |cup - plan_swap|^2, bad_aim frames only (h9)
Reading rule, fixed before the run: M-full clearly lower than augfl on the same frames -> supports 'wrist used less';
about equal -> wrist reliance did not change.
Caveats: M-full ran TE (n_action_steps=1) on the robot; this reads the raw chunk[D]. Plans map joints to the table with
the human cup-grasp fit (h8.xy), so this is cup only; the bottle needs its own mapping or a joint-space measure.
Writes new files only (h8 / h9 outputs from 10-18 are left alone).

    uv run python scripts/input_reliance/h11_mfull_wrist_cup.py
"""
import sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, "scripts/input_reliance")
import h8_fl_black_all_cup as H
from huggingface_hub import snapshot_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

W = H.W
MODELS = [
    ("augfl", "ericc430/act_omx_b1_paper-cup_60-13-10rcvrycut-46far-augfl-fldrop30_100k"),
    ("mfull200k", "ericc430/act_omx_paper-cup129-plastic-bottle135-augfl-fldrop30_200k"),
]
H.ROLLOUTS = H.ROLLOUTS + [
    ("11-11 mfull", "rollout_omx_mfull_cup129_bottle135_augfl_fldrop30_200k_nas1_te-001_30s_paper_cup_20261111_163453", None),
]
OUT_ROWS = H.OUT / "h11_mfull_wrist_cup.csv"
OUT_SUM = H.OUT / "h11_mfull_wrist_cup_summary.csv"


def main() -> None:
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
                return H.xy(post(pol.predict_action_chunk(b)[0].cpu()).numpy()[D, :5])

        for i, f in enumerate(frames):
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
    out["old"] = np.where(out.rollout == "11-11 mfull", "11-11", "10-06..10-17")
    b = out[out.bad_aim]
    s = out.groupby(["old", "model", "D"]).agg(n=("move_wg", "size"), err_orig=("err_orig", "median"),
                                              move_wg=("move_wg", "median"), move_sw=("move_sw", "median")).join(
        b.groupby(["old", "model", "D"]).agg(n_bad=("corr", "count"), corr=("corr", "median"),
                                             corr_q25=("corr", lambda c: c.quantile(.25)), corr_q75=("corr", lambda c: c.quantile(.75))))
    s.to_csv(OUT_SUM)
    pd.set_option("display.width", 200)
    print(s.round(2).to_string())
    print(f"rows: {OUT_ROWS}\nsummary: {OUT_SUM}")


if __name__ == "__main__":
    main()

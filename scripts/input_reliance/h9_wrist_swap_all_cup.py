"""H9: keep the front-left view, swap the wrist view for a success episode's -> how far does the plan move? (2026-10-18)

Same frames, models and plan point as h8_fl_black_all_cup.py. Donor = the frame at the same D before the first close of
another success episode in the same rollout (different placement), cycled; only the wrist image is replaced.
  move       = |plan_swap - plan_orig|  (all frames)
  correction = (plan_orig - plan_swap) . (cup - plan_swap) / |cup - plan_swap|^2  (bad_aim frames only, as h1_rollout_normalized)
    0 = the real wrist view changes nothing, 1 = it pulls the plan exactly onto the cup.
Caveat: success donors are mostly 'cup centred in the wrist view', so on frames that are already aligned the swap
is close to a no-op; the bad_aim rows are the informative ones.

    uv run python scripts/input_reliance/h9_wrist_swap_all_cup.py
"""
import sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, "scripts/input_reliance")
import h8_fl_black_all_cup as H
from huggingface_hub import snapshot_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

W = H.W


def main() -> None:
    frames = H.load_frames()
    donors = {}
    for i, f in enumerate(frames):
        if f["outcome"] == "success":
            donors.setdefault((f["rollout"], f["D"]), []).append(i)
    pairs = []
    for i, f in enumerate(frames):
        pool = [j for j in donors.get((f["rollout"], f["D"]), []) if frames[j]["pid"] != f["pid"]]
        if pool:
            pairs.append((i, pool[f["ep"] % len(pool)]))
    print(f"{len(pairs)} of {len(frames)} frames have a donor", flush=True)
    rows = []
    for label, repo in H.MODELS:
        ck = snapshot_download(repo)
        pol = get_policy_class("act").from_pretrained(ck, config=PreTrainedConfig.from_pretrained(ck)).eval().to(H.dev)
        pre, post = make_pre_post_processors(pol.config, pretrained_path=ck)

        def plan(item, D):
            b = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
            b = {k: (v.to(H.dev) if isinstance(v, torch.Tensor) else v) for k, v in pre(b).items()}
            with torch.no_grad():
                return H.xy(post(pol.predict_action_chunk(b)[0].cpu()).numpy()[D, :5])

        for i, j in pairs:
            f = frames[i]
            x = {k: (v.float() / 255 if k in (W, H.FL) else v) for k, v in f["item"].items()}
            p0 = plan(x, f["D"])
            x[W] = frames[j]["item"][W].float() / 255
            ps = plan(x, f["D"])
            v = f["cup"] - ps
            rows.append(dict(model=label, rollout=f["rollout"], ep=f["ep"], pid=f["pid"], donor_ep=frames[j]["ep"],
                             outcome=f["outcome"], bad_aim="bad_aim" in f["mechanism"], D=f["D"],
                             move=np.linalg.norm(ps - p0), need=np.linalg.norm(v), correction=(p0 - ps) @ v / (v @ v)))
        df = pd.DataFrame([r for r in rows if r["model"] == label])
        print(f"== {label}: " + " | ".join(f"D={D} move {g.move.median():.1f} cm, bad_aim corr {g[g.bad_aim].correction.median():+.2f}"
                                           for D, g in df.groupby("D")), flush=True)
        del pol
        torch.cuda.empty_cache()
    out = pd.DataFrame(rows)
    out.to_csv(H.OUT / "h9_wrist_swap_all_cup.csv", index=False)
    b = out[out.bad_aim]
    s = out.groupby(["model", "D"]).agg(n=("move", "size"), move=("move", "median")).join(
        b.groupby(["model", "D"]).agg(n_bad=("correction", "size"), need=("need", "median"), corr=("correction", "median"),
                                      corr_q25=("correction", lambda c: c.quantile(.25)), corr_q75=("correction", lambda c: c.quantile(.75))))
    print(s.reindex([m for m, _ in H.MODELS], level=0).round(2).to_string())
    print(f"per-frame rows: {H.OUT / 'h9_wrist_swap_all_cup.csv'}")


if __name__ == "__main__":
    main()

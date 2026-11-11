"""H8: blank the front-left view -> how far does the planned grasp move, for every paper-cup ACT 100k model? (2026-10-18)

Same plan point as h1_rollout_normalized (chunk step D from frame k - D, k = first close, mapped to table x, y cm
with the human joint -> (theta, r) fit), on one shared frame set: every valid episode with a first close in the
paper-cup closed-loop rollouts listed in ROLLOUTS. For each model and frame:
  orig       : as recorded
  FL black   : front-left -> 0 (what fl dropout trained on in the 30 frames before the first close)
  wrist gray : wrist -> 0.5 gray (comparison: the 10-07 blank-wrist test)
and report |plan - plan_orig| and |plan - cup| (cup = placement of that episode).
Caveats: frames are closed-loop states, so the arm is already near where it will close -- the state alone carries
part of the answer, more so at small D. Black front-left is out of distribution for models without fl dropout;
gray wrist is out of distribution for all. Cup position is the placement, not where the cup was if it got pushed.

    uv run python scripts/input_reliance/h8_fl_black_all_cup.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, "scripts")
import aim_error as A
from huggingface_hub import snapshot_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

W, FL = "observation.images.wrist", "observation.images.front-left"
ROOT = "D:/hf/lerobot/ericc430/"
OUT = A.REPO / "data/logs/input_reliance"
MODELS = [  # label, Hub repo (100k of every paper-cup ACT run in docs/models.md)
    ("b1_uvc60", "ericc430/act_omx_b1_uvc60_100k"),
    ("ntA1", "ericc430/act_omx_b1_paper-cup_60-16-12rcvry_100k"),
    ("ntfA1", "ericc430/act_omx_b1_paper-cup_60-13-10rcvry-46far_100k"),
    ("ntcfA1", "ericc430/act_omx_b1_paper-cup_60-13-10rcvrycut-46far_100k"),
    ("ntfA1aug", "ericc430/act_omx_b1_paper-cup_60-13-10rcvry-46far-aug_100k"),
    ("ntcfA1aug", "ericc430/act_omx_b1_paper-cup_60-13-10rcvrycut-46far-aug_100k"),
    ("fldrop30", "ericc430/act_omx_b1_paper-cup_60-13-10rcvrycut-46far-aug-fldrop30_100k"),
    ("fldrop50", "ericc430/act_omx_b1_paper-cup_60-13-10rcvrycut-46far-aug-fldrop50_100k"),
    ("augfl", "ericc430/act_omx_b1_paper-cup_60-13-10rcvrycut-46far-augfl-fldrop30_100k"),
    ("u60augfl", "ericc430/act_omx_b1_paper-cup_60-augfl-fldrop30_100k"),
]
ROLLOUTS = [  # label, dataset folder, episode_meta csv (all A1 closed loop, paper cup, eval-close placements)
    ("10-06 b1_uvc60", "rollout_omx_b1_uvc60_100k_nas30_A1_paper_cup_20261006_094908",
     "rollout_omx_b1_uvc60_100k_nas30_A1_paper_cup_20261006_close_loop.csv"),
    ("10-08 ntA1", "rollout_omx_b1_papercup_60_16_12rcvry_100k_paper_cup_20261008_192921",
     "rollout_omx_b1_papercup_60_16_12rcvry_100k_nas30_paper_cup_20261008_192921.csv"),
    ("10-13 ntfA1aug", "rollout_omx_b1_papercup_60_13_10rcvry_46far_aug_100k_nas30_paper_cup_20261013_184921", None),
    ("10-13 ntcfA1aug", "rollout_omx_b1_papercup_60_13_10rcvrycut_46far_aug_100k_nas30_paper_cup_20261013_201040", None),
    ("10-14 fldrop30", "rollout_omx_b1_papercup_60_13_10rcvrycut_46far_aug_fldrop30_100k_nas30_paper_cup_20261014_151208", None),
    ("10-16 fldrop50", "rollout_omx_b1_papercup_60_13_10rcvrycut_46far_aug_fldrop50_100k_nas30_paper_cup_20261016_142101", None),
    ("10-16 augfl", "rollout_omx_b1_papercup_60_13_10rcvrycut_46far_augfl_fldrop30_100k_nas30_paper_cup_20261016_152018", None),
    ("10-17 u60augfl", "rollout_omx_b1_papercup_60_augfl_fldrop30_100k_nas30_paper_cup_20261017_215410", None),
]
DS = (5, 15, 30)
CONDS = ("orig", "FL black", "wrist gray")
dev = "cuda" if torch.cuda.is_available() else "cpu"

tm = A.placements("episode_meta/omx_pick_place_pilot_uvc_60.csv", str(A.REPO / "configs/placements/campA_136sym_20260908_20260908_"))
tg = A.grasp_joints(".cache/lerobot/omx_pick_place_pilot_uvc_60", 0.6, 10)
tm = tm[tm.episode_index.isin(tg)]
C = A.fit(np.array([tg[e] for e in tm.episode_index]), tm.th.values, tm.r.values)


def xy(j5):
    t, r = A.predict(C, j5[None]); t, r = t[0], r[0]
    return np.array([r * np.cos(np.deg2rad(t)), r * np.sin(np.deg2rad(t))])


def load_frames():
    """One pass over the rollouts: frames k - D kept in memory (images as uint8) with the cup position."""
    frames = []
    for rl, d, csv in ROLLOUTS:
        m = A.placements(f"episode_meta/{csv or d + '.csv'}", str(A.REPO / "configs/placements/campA_136sym_20260908_20260908_"))
        m = m[pd.to_numeric(m.valid, errors="coerce").fillna(1) != 0]
        ds = LeRobotDataset(d, root=ROOT + d)
        cols = ds.hf_dataset.with_format(None).select_columns(["episode_index", "action"])
        epi = np.array(cols["episode_index"]); act = np.array(cols["action"], dtype=np.float32)
        n0 = len(frames)
        for r in m.itertuples():
            idx = np.flatnonzero(epi == r.episode_index)
            if len(idx) == 0:
                continue
            g = act[idx, 5]; depth = g[0] - g.min()
            if depth < 10:  # never closed
                continue
            k = int(np.argmax(g < g[0] - 0.6 * depth))
            cup = np.array([r.r * np.cos(np.deg2rad(r.th)), r.r * np.sin(np.deg2rad(r.th))])
            for D in DS:
                if k - D < 0:
                    continue
                it = ds[int(idx[k - D])]
                it = {kk: ((v * 255).round().to(torch.uint8) if kk in (W, FL) else v) for kk, v in it.items()}
                frames.append(dict(rollout=rl, ep=r.episode_index, pid=r.placement_id, outcome=r.outcome,
                                   mechanism=r.mechanism if isinstance(r.mechanism, str) else "", D=D, cup=cup, item=it))
        print(f"{rl}: {len(frames) - n0} frames", flush=True)
    return frames


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = load_frames()
    rows = []
    for label, repo in MODELS:
        ck = snapshot_download(repo)
        pol = get_policy_class("act").from_pretrained(ck, config=PreTrainedConfig.from_pretrained(ck)).eval().to(dev)
        pre, post = make_pre_post_processors(pol.config, pretrained_path=ck)

        def plan(item, D):
            b = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
            b = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in pre(b).items()}
            with torch.no_grad():
                return xy(post(pol.predict_action_chunk(b)[0].cpu()).numpy()[D, :5])

        for f in frames:
            base = {k: (v.float() / 255 if k in (W, FL) else v) for k, v in f["item"].items()}
            p = {}
            for c in CONDS:
                x = dict(base)
                if c == "FL black":
                    x[FL] = torch.zeros_like(x[FL])
                elif c == "wrist gray":
                    x[W] = torch.full_like(x[W], 0.5)
                p[c] = plan(x, f["D"])
            rows.append(dict(model=label, rollout=f["rollout"], ep=f["ep"], pid=f["pid"], outcome=f["outcome"],
                             mechanism=f["mechanism"], D=f["D"],
                             move_fl=np.linalg.norm(p["FL black"] - p["orig"]), move_wg=np.linalg.norm(p["wrist gray"] - p["orig"]),
                             err_orig=np.linalg.norm(p["orig"] - f["cup"]), err_fl=np.linalg.norm(p["FL black"] - f["cup"]),
                             err_wg=np.linalg.norm(p["wrist gray"] - f["cup"])))
        df = pd.DataFrame([r for r in rows if r["model"] == label])
        print(f"== {label}: " + " | ".join(
            f"D={D} move FL {g.move_fl.median():.1f} / wrist {g.move_wg.median():.1f} cm, err orig {g.err_orig.median():.1f} FL {g.err_fl.median():.1f}"
            for D, g in df.groupby("D")), flush=True)
        del pol
        torch.cuda.empty_cache()
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "h8_fl_black_all_cup.csv", index=False)
    s = out.groupby(["model", "D"]).agg(n=("move_fl", "size"), move_fl=("move_fl", "median"), move_wg=("move_wg", "median"),
                                        err_orig=("err_orig", "median"), err_fl=("err_fl", "median"), err_wg=("err_wg", "median"))
    s = s.reindex([m for m, _ in MODELS], level=0)
    print(s.round(1).to_string())
    print(f"per-frame rows: {OUT / 'h8_fl_black_all_cup.csv'}")


if __name__ == "__main__":
    main()

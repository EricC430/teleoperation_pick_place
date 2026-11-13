"""H13: at c17 / c36 the cup is only a sliver behind the metal handle in the front-left view. Does the policy use that
sliver to choose the turning direction? (2026-11-12)

Frames: every A1 paper-cup closed-loop rollout in h8.ROLLOUTS plus 11-11 M-full, episodes at c17 / c36, t = 0 / 0.5 / 1 s
(the arm has not turned yet; the turn starts at 3-5 s, inside the 100-step = 6.7 s chunk predicted here).
Direction = arm theta (base joint, human cup fit h8.C) of chunk step 99; cup theta is -64 (c17) / -47 (c36), home ~ +5.
Conditions:
  orig     as recorded
  erased   front-left: fixed rectangle RECT[placement] replaced by the same pixels of a donor frame (same rollout, same t,
           cup on the left side theta > 30), gain-matched; donor = best match on a ring around the rectangle.
           v1 (11-12) pasted only pixels with |target - donor| > 25: Boyu saw cup print left. v2 pasted their padded
           bounding box: the white rim against the wall stayed outside at c17, seams on large boxes. v3 = fixed rectangles.
  control  same rectangle shifted 200 px to the left (no cup there), pasted from the donor that matches the target best
           inside that shifted rectangle (another episode's cup can stand there) -> paste artefact alone
  swap     same placement and t, front-left from rollout A, wrist + state from rollout B (all A x B pairs): does the
           direction follow the front-left image or the rest of the input?
Reading rule for erased, fixed before the run: a model uses the sliver if orig turns right (theta_end < 0) and erased turns
left / stays (theta_end > +5) while control stays like orig. Each frame is one run of one placement; frames of the same
episode are not independent. Writes h13_*.csv / jpg in data/logs/input_reliance; nothing else is touched.

    uv run python scripts/input_reliance/h13_occluded_cup_direction.py --sheets   # edit-check and comparison images only
    uv run python scripts/input_reliance/h13_occluded_cup_direction.py            # + models
"""
import sys
import cv2, numpy as np, pandas as pd, torch
sys.path.insert(0, "scripts/input_reliance")
sys.path.insert(0, "scripts")
import h8_fl_black_all_cup as H
import aim_error as A
from huggingface_hub import snapshot_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

FL, W = H.FL, H.W
PIDS = ("c17", "c36")
TS = (0.0, 0.5, 1.0)
FPS = 15
# v3 erase rectangles (x0, y0, x1, y1), set by eye on the 2x edit-check sheets so that every rollout's cup incl. rim is inside
RECT = {"c17": (735, 238, 848, 400), "c36": (665, 240, 800, 420)}
CROP = (slice(160, 480), slice(470, 848))  # front-left region shown on the sheets, enlarged 2x
MODELS = [
    ("augfl", "ericc430/act_omx_b1_paper-cup_60-13-10rcvrycut-46far-augfl-fldrop30_100k"),
    ("mfull200k", "ericc430/act_omx_paper-cup129-plastic-bottle135-augfl-fldrop30_200k"),
]
ROLLOUTS = H.ROLLOUTS + [("11-11 mfull", "rollout_omx_mfull_cup129_bottle135_augfl_fldrop30_200k_nas1_te-001_30s_paper_cup_20261111_163453", None)]
PREFIX = str(A.REPO / "configs/placements/campA_136sym_20260908_20260908_")
OUT_ROWS = H.OUT / "h13_occluded_cup_direction.csv"
OUT_SWAP = H.OUT / "h13_swap.csv"


def hwc(t):
    return (t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def chw(a):
    return torch.from_numpy(a.astype(np.float32) / 255).permute(2, 0, 1)


def paste(tgt, don, rect):
    """Copy rect (x0, y0, x1, y1) from donor into target, donor gain-matched per channel to the target on a 25 px ring
    around the rect (auto-exposure differs a little between episodes). Returns (image, ring mean abs diff)."""
    x0, y0, x1, y1 = rect
    h, w = tgt.shape[:2]
    big_ = np.zeros((h, w), bool); big_[max(y0 - 25, 0):y1 + 25, max(x0 - 25, 0):x1 + 25] = True
    ring = big_.copy(); ring[y0:y1, x0:x1] = False
    t, d = tgt.astype(np.float32), don.astype(np.float32)
    gain = t[ring].mean(0) / np.maximum(d[ring].mean(0), 1)
    out = tgt.copy()
    out[y0:y1, x0:x1] = np.clip(d[y0:y1, x0:x1] * gain, 0, 255).round().astype(np.uint8)
    return out, float(np.abs(t[ring] - d[ring] * gain).mean())


def load():
    frames = []
    for rl, d, csv in ROLLOUTS:
        m = A.placements(f"episode_meta/{csv or d + '.csv'}", PREFIX)
        m = m[pd.to_numeric(m.valid, errors="coerce").fillna(1) != 0]
        ds = LeRobotDataset(d, root=H.ROOT + d)
        epi = np.array(ds.hf_dataset.with_format(None).select_columns(["episode_index"])["episode_index"])
        donors = m[(m.th > 30)].episode_index.tolist()
        for r in m[m.placement_id.isin(PIDS)].itertuples():
            idx = np.flatnonzero(epi == r.episode_index)
            if len(idx) == 0:
                continue
            for t in TS:
                it = ds[int(idx[int(t * FPS)])]
                tgt = hwc(it[FL])
                rect = RECT[r.placement_id]
                best = None
                x0, y0, x1, y1 = rect
                sh = (x0 - 200, y0, x1 - 200, y1)
                cbest = None
                for de in donors:  # donor whose surroundings of the rect match best (same exposure / nothing moved)
                    di = np.flatnonzero(epi == de)
                    if len(di) <= int(t * FPS):
                        continue
                    don = hwc(ds[int(di[int(t * FPS)])][FL])
                    er, err = paste(tgt, don, rect)
                    if best is None or err < best[0]:
                        best = (err, de, don, er)
                    # control donor: separate choice, the one closest to the target inside the shifted rect (no cup there)
                    co_, _ = paste(tgt, don, sh)
                    cerr = float(np.abs(co_.astype(np.float32) - tgt).mean())
                    if cbest is None or cerr < cbest[0]:
                        cbest = (cerr, de, co_)
                if best is None:
                    print(f"  {rl} {r.placement_id} t={t}: no donor", flush=True)
                    continue
                _, de, don, er = best
                co, cdon = cbest[2], cbest[1]
                frames.append(dict(rollout=rl, ep=r.episode_index, pid=r.placement_id, outcome=r.outcome, t=t, donor=de, cdonor=cdon,
                                   box=f"{x0},{y0},{x1},{y1}", item=dict(it), orig_img=tgt, erased_img=er, control_img=co))
        print(f"{rl}: {len(frames)} frames so far", flush=True)
    return frames


def big(im):
    c = im[CROP]
    return np.ascontiguousarray(cv2.resize(c, (c.shape[1] * 2, c.shape[0] * 2), interpolation=cv2.INTER_NEAREST))


def label(im, txt, y=34, col=(255, 255, 0)):
    cv2.putText(im, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
    cv2.putText(im, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    return im


def sheets(frames, plans):
    """t = 0, per placement: (1) edit check, one jpg per rollout = orig with rectangle | erased | control, 2x native pixels;
    (2) comparison = orig only, one tile per rollout, sorted by planned direction (right first), labelled with both models'
    orig theta_end at t = 0 (`plans`, from the rows csv of a previous run) and the closed-loop outcome."""
    for pid in PIDS:
        fs = [f for f in frames if f["pid"] == pid and f["t"] == 0.0]
        for f in fs:
            x0, y0, x1, y1 = map(int, f["box"].split(","))
            a = cv2.rectangle(f["orig_img"].copy(), (x0, y0), (x1, y1), (255, 0, 0), 1)
            row = np.hstack([label(big(a), f"{f['rollout']} {pid} ep{f['ep']} orig"),
                             label(big(f["erased_img"]), f"erased (donor ep{f['donor']})"), label(big(f["control_img"]), f"control (donor ep{f['cdonor']})")])
            cv2.imwrite(str(H.OUT / f"h13_edit_check_{pid}_{f['rollout'].replace(' ', '_')}.jpg"),
                        cv2.cvtColor(row, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        pl = lambda f: plans.get((f["rollout"], pid), {})
        tiles = []
        for f in sorted(fs, key=lambda f: pl(f).get("augfl", 0) + pl(f).get("mfull200k", 0)):
            p = pl(f)
            im = label(big(f["orig_img"]), f"{f['rollout']}  ({f['outcome']})")
            label(im, "plan: augfl {:+.0f}  mfull {:+.0f}".format(p.get("augfl", np.nan), p.get("mfull200k", np.nan)),
                  y=im.shape[0] - 14, col=(255, 255, 255))
            tiles.append(im)
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
        cv2.imwrite(str(H.OUT / f"h13_fl_compare_{pid}.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def camera_shift(frames):
    """Phase correlation of the whole t = 0 front-left frame (grey) of each rollout against the first one (c36 episodes):
    does the camera sit at the same place on every date? Printed only."""
    ref = None
    for f in frames:
        if f["t"] != 0.0 or f["pid"] != "c36":
            continue
        g = cv2.cvtColor(f["orig_img"], cv2.COLOR_RGB2GRAY).astype(np.float32)
        if ref is None:
            ref, name = g, f["rollout"]
        (dx, dy), resp = cv2.phaseCorrelate(ref, g, cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F))
        print(f"camera shift vs {name}: {f['rollout']:16s} dx {dx:+6.1f} dy {dy:+6.1f} px (response {resp:.2f})")


def main() -> None:
    frames = load()
    camera_shift(frames)
    plans = {}
    if OUT_ROWS.exists():
        old = pd.read_csv(OUT_ROWS)
        for r in old[old.t == 0.0].itertuples():
            plans.setdefault((r.rollout, r.pid), {})[r.model] = r.orig_end
    sheets(frames, plans)
    print(f"sheets: {H.OUT}/h13_edit_check_*.jpg, h13_fl_compare_*.jpg", flush=True)
    if "--sheets" in sys.argv:
        return
    out, swap = [], []
    for label_, repo in MODELS:
        ck = snapshot_download(repo)
        pol = get_policy_class("act").from_pretrained(ck, config=PreTrainedConfig.from_pretrained(ck)).eval().to(H.dev)
        pre, post = make_pre_post_processors(pol.config, pretrained_path=ck)

        def theta_end(item):
            b = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
            b = {k: (v.to(H.dev) if isinstance(v, torch.Tensor) else v) for k, v in pre(b).items()}
            with torch.no_grad():
                a = post(pol.predict_action_chunk(b)[0].cpu()).numpy()
            th = np.polyval(H.C[0], a[:, 0])
            return th[-1], th.min(), th.max()

        for f in frames:
            r = {k: f[k] for k in ("rollout", "ep", "pid", "outcome", "t", "donor", "cdonor", "box")}
            r["model"] = label_
            for c in ("orig", "erased", "control"):
                x = dict(f["item"])
                if c != "orig":
                    x[FL] = chw(f[f"{c}_img"])
                r[f"{c}_end"], r[f"{c}_min"], r[f"{c}_max"] = theta_end(x)
            out.append(r)
        for fa in frames:  # front-left from A, wrist + state (+ everything else) from B
            for fb in frames:
                if fa["pid"] != fb["pid"] or fa["t"] != fb["t"]:
                    continue
                x = dict(fb["item"]); x[FL] = fa["item"][FL]
                swap.append(dict(model=label_, pid=fa["pid"], t=fa["t"], fl_from=fa["rollout"], rest_from=fb["rollout"],
                                 theta_end=theta_end(x)[0]))
        print(f"== {label_} done", flush=True)
        del pol
        torch.cuda.empty_cache()
    d = pd.DataFrame(out)
    d.to_csv(OUT_ROWS, index=False)
    s = pd.DataFrame(swap)
    s.to_csv(OUT_SWAP, index=False)
    pd.set_option("display.width", 220)
    print(d[["model", "rollout", "pid", "t", "orig_end", "erased_end", "control_end"]].round(0).to_string(index=False))
    for (mo, pid), g in d.groupby(["model", "pid"]):
        print(f"{mo:10s} {pid}: right turns (theta_end<0) orig {(g.orig_end < 0).sum()}/{len(g)}, erased {(g.erased_end < 0).sum()}/{len(g)}, control {(g.control_end < 0).sum()}/{len(g)}")
    for (mo, pid, t), g in s.groupby(["model", "pid", "t"]):
        print(f"\n== swap {mo} {pid} t={t}: theta_end, rows = front-left from, columns = wrist + state from")
        print(g.pivot(index="fl_from", columns="rest_from", values="theta_end").round(0).to_string())
    print(f"rows: {OUT_ROWS}\nswap: {OUT_SWAP}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Fit the transform between the COORDINATE-PAPER frame and the arm's FK (pan-axis) frame, from grasps.

Question (Eric 2026-10-08): ArUco centres and cup centres were both placed by the same coordinate paper. If the
paper sits at a consistent offset from the arm's pan-axis frame, every grasp -- arm by FK, cup by the paper --
shows that SAME offset, and the sim (arm at the pan origin, cups at paper coordinates) is off from the video by it.

Data, per episode: the grasp frame (eval_joint_calibration.py's definition: a close that stays closed 15 frames),
the pinch point there by FK (current joint_mapping, CAD pinch point -- the same functions, imported), and the cup's
placement (configs/placements/campA_136sym_..._train.csv via episode_meta/<dataset>.csv).

The gripper pinches the cup WALL (eval_joint_calibration.py, 已查證 2026-09-29), so the pinch point lies on a circle
of radius R around the cup centre, not at it. A = FK frame, P = paper frame:
    | c_P - (Rot(psi) p_A + t) | = R
Models: none / t only / psi only (a rotation about the pan axis IS a shoulder_pan zero error -- the two cannot be
told apart) / t + psi. Chosen by leave-one-out residual; uncertainty by bootstrap over episodes.

Conventions in the output: t = where the FK origin (pan axis) lands on the paper, in cm. For psi = 0,
calib_extrinsics_aruco.py's --paper-offset-m (X_true = X_paper + offset) is -t.

What this measures and what it does not:
  * Paper vs the FK frame AS CURRENTLY CALIBRATED. joint_mapping's offsets were fitted on mat points (touch, 9/22)
    and hand-tuned on uvc_60 grasps, so part of any physical paper-vs-base offset may already sit inside them. For
    the sim that does not matter (it uses the same joint_mapping); it is NOT a physical measurement of the paper.
  * Planar only (x, y, yaw); the paper lies on the table.
  * No camera is used for the fit. The front-left overlay only VISUALISES it (cups from paper coordinates, pinch
    points from FK, projected through the ArUco extrinsics, which live in the paper frame).

    python3 scripts/fit_paper_to_pan.py                 # writes calibration/<date>_paper_to_pan.json + outputs/paper_to_pan/
"""
from __future__ import annotations

import csv
import datetime as dt
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "sim"))
sys.path.insert(0, str(_REPO))
_spec = importlib.util.spec_from_file_location("ejc", _REPO / "scripts" / "eval_joint_calibration.py")
EJC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EJC)  # point_cm / q5 / X_PINCH / PLACEMENTS: one geometry for both scripts
import camera_distortion as cd  # noqa: E402
import scene_constants as S  # noqa: E402

DATA = _REPO / "data/huggingface/lerobot/ericc430"
# grasp rule: "first" = first sustained close (normal demos: one close each);
# "last" = last sustained close before the final opening (recovery demos open with a deliberate empty grab).
DATASETS = {
    "omx_pick_place_pilot_paper_cup_normal_A1": "first",
    "omx_pick_place_pilot_paper_cup_recovery_A1": "last",
    "omx_pick_place_pilot_paper_cup_recovery_A1_tight": "last",
}
MODELS = {"none": (), "translation": ("tx", "ty"), "rotation": ("psi",), "translation+rotation": ("tx", "ty", "psi")}
RADII = (3.0, 3.4, 3.75)  # cm: pinch-height cup radius; 3.4 is eval_joint_calibration's value (base 2.5 .. rim 3.75)


def load_states(name: str) -> dict[int, np.ndarray]:
    import pyarrow.parquet as pq

    eps: dict[int, list] = {}
    for f in sorted((DATA / name / "data").rglob("*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"]).to_pydict()
        for e, fi, s in zip(t["episode_index"], t["frame_index"], t["observation.state"]):
            eps.setdefault(int(e), []).append((int(fi), s))
    return {e: np.array([s for _, s in sorted(v)]) for e, v in eps.items()}


def grasp_frame(st: np.ndarray, rule: str, hold: int = 15) -> int | None:
    g = st[:, 5]
    thr = np.percentile(g, 95) - 0.6 * (np.percentile(g, 95) - np.percentile(g, 5))
    closes = [i for i in range(1, len(g) - hold) if g[i - 1] >= thr > g[i] and np.all(g[i:i + hold] < thr)]
    opens = [i for i in range(1, len(g)) if g[i - 1] < thr <= g[i]]
    if not closes:
        return None
    if rule == "first":
        return closes[0]
    before = [c for c in closes if not opens or c < opens[-1]]
    return before[-1] if before else None


def collect(name: str, rule: str):
    meta = list(csv.DictReader((_REPO / "episode_meta" / f"{name}.csv").open(encoding="utf-8-sig")))
    place = {r["placement_id"]: (float(r["x_cm"]), float(r["y_cm"])) for r in csv.DictReader(EJC.PLACEMENTS.open(encoding="utf-8"))}
    states = load_states(name)
    rows = []
    for m in meta:
        e = int(m["episode_index"])
        if m.get("valid") != "1" or not m["placement_id"].startswith("t") or e not in states:
            continue
        gi = grasp_frame(states[e], rule)
        if gi is None:
            continue
        p = EJC.point_cm(EJC.q5(states[e][gi]), EJC.X_PINCH)
        rows.append({"episode": e, "placement": m["placement_id"], "grasp_frame": gi,
                     "cup_xy_cm": list(place[f"train_{int(m['placement_id'][1:]):03d}"]),
                     "pinch_xyz_cm": [float(v) for v in p], "gripper": float(states[e][gi + 5, 5])})
    return rows


def transform(params: dict, p: np.ndarray) -> np.ndarray:
    psi = math.radians(params.get("psi", 0.0))
    c, s = math.cos(psi), math.sin(psi)
    return p @ np.array([[c, s], [-s, c]]) + np.array([params.get("tx", 0.0), params.get("ty", 0.0)])


def residuals(params: dict, P: np.ndarray, C: np.ndarray, R: float) -> np.ndarray:
    return np.linalg.norm(C - transform(params, P), axis=1) - R


def fit(model: str, P: np.ndarray, C: np.ndarray, R: float) -> dict:
    names = MODELS[model]
    if not names:
        return {}
    sol = least_squares(lambda x: residuals(dict(zip(names, x)), P, C, R), np.zeros(len(names)), loss="soft_l1", f_scale=1.0)
    return dict(zip(names, (float(v) for v in sol.x)))


def loo_rms(model: str, P: np.ndarray, C: np.ndarray, R: float) -> float:
    out = []
    for i in range(len(P)):
        keep = np.arange(len(P)) != i
        out.append(residuals(fit(model, P[keep], C[keep], R), P[i:i + 1], C[i:i + 1], R)[0])
    return float(np.sqrt(np.mean(np.square(out))))


def bootstrap(model: str, P: np.ndarray, C: np.ndarray, R: float, n: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    draws = [fit(model, P[idx], C[idx], R) for idx in (rng.integers(0, len(P), len(P)) for _ in range(n))]
    names = MODELS[model]
    arr = np.array([[d[k] for k in names] for d in draws])
    return {"ci95": {k: [float(np.percentile(arr[:, j], 2.5)), float(np.percentile(arr[:, j], 97.5))] for j, k in enumerate(names)},
            "corr": np.corrcoef(arr.T).round(2).tolist() if len(names) > 1 else None}


def analyse(rows: list[dict]) -> dict:
    P = np.array([r["pinch_xyz_cm"][:2] for r in rows])
    C = np.array([r["cup_xy_cm"] for r in rows])
    out = {"n": len(rows), "by_radius": {}}
    for R in RADII:
        res = {}
        for model in MODELS:
            par = fit(model, P, C, R)
            r = residuals(par, P, C, R)
            res[model] = {"params": par, "fit_rms_cm": float(np.sqrt(np.mean(r ** 2))), "loo_rms_cm": loo_rms(model, P, C, R) if MODELS[model] else float(np.sqrt(np.mean(r ** 2)))}
        out["by_radius"][str(R)] = res
    best = min(MODELS, key=lambda m: out["by_radius"]["3.4"][m]["loo_rms_cm"])
    out["best_model_R3.4"] = best
    out["bootstrap_R3.4"] = {m: bootstrap(m, P, C, 3.4) for m in ("translation", "translation+rotation")}
    par = out["by_radius"]["3.4"]["translation+rotation"]["params"]
    for r, d_none, d_fit in zip(rows, residuals({}, P, C, 3.4), residuals(par, P, C, 3.4)):
        r["wall_residual_cm_identity"], r["wall_residual_cm_fitted"] = float(d_none), float(d_fit)
        r["dist_pinch_to_cup_centre_cm_identity"] = float(np.linalg.norm(np.array(r["cup_xy_cm"]) - np.array(r["pinch_xyz_cm"][:2])))
    return out


def overlay(name: str, rows: list[dict], params: dict, out_png: Path, extr: Path, intr: Path) -> None:
    import cv2

    model = cd.load_model(intr)
    e = json.loads(extr.read_text(encoding="utf-8"))
    w, x, y, z = e["quat_wxyz"]
    R_wc = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    t_wc, table_z = np.array(e["pos_m"]), e["table_top_z"]

    def px(xyz_cm):
        pts = np.atleast_2d(xyz_cm) / 100.0 + np.array([0, 0, table_z])
        cam = (pts - t_wc) @ R_wc
        ideal = (cam[:, :2] / cam[:, 2:3]) * [model.K[0, 0], model.K[1, 1]] + [model.K[0, 2], model.K[1, 2]]
        return cd.distort_pixels(ideal, model)

    import pyarrow.parquet as pq

    epi = pq.read_table(next((DATA / name / "meta" / "episodes").rglob("*.parquet")), columns=["episode_index", "dataset_from_index"]).to_pydict()
    start = dict(zip(epi["episode_index"], epi["dataset_from_index"]))
    want = {start[r["episode"]] + r["grasp_frame"]: r for r in rows}
    cap = cv2.VideoCapture(str(next((DATA / name / "videos" / "observation.images.front-left").rglob("*.mp4"))))
    tiles, i = {}, 0
    while len(tiles) < len(want):
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            r = want[i]
            cup = np.array(r["cup_xy_cm"])
            ang = np.linspace(0, 2 * np.pi, 48)
            for zc, col in ((0.0, (255, 255, 0)), (S.CUP_HEIGHT * 100, (255, 255, 0))):
                ring = np.stack([cup[0] + 3.75 * np.cos(ang), cup[1] + 3.75 * np.sin(ang), np.full_like(ang, zc)], 1)
                cv2.polylines(frame, [px(ring).astype(np.int32)], True, col, 1, cv2.LINE_AA)
            p = np.array(r["pinch_xyz_cm"])
            p_fit = np.r_[transform(params, p[None, :2])[0], p[2]]
            for pt, col in ((p, (0, 0, 255)), (p_fit, (0, 255, 0))):
                u, v = px(pt)[0]
                cv2.drawMarker(frame, (int(u), int(v)), col, cv2.MARKER_CROSS, 14, 2)
            cv2.putText(frame, f"ep{r['episode']} {r['placement']}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tiles[r["episode"]] = cv2.resize(frame, (424, 240))
        i += 1
    keys = sorted(tiles)
    cols = 4
    blank = np.zeros((240, 424, 3), np.uint8)
    grid = [np.hstack([tiles[k] for k in keys[j:j + cols]] + [blank] * (cols - len(keys[j:j + cols]))) for j in range(0, len(keys), cols)]
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), np.vstack(grid))


def main() -> int:
    today = dt.date.today().isoformat()
    result = {"what": "paper frame P vs arm FK frame A from grasps: c_P = Rot(psi) p_A + t, pinch on the cup wall (radius R)",
              "convention": "t_cm = where the FK origin (pan axis) lands on the paper; psi_deg CCW; --paper-offset-m = -t/100 when psi = 0",
              "joint_mapping": {"scale": dict(EJC.JM.SCALE_RAD_PER_UNIT), "offset": dict(EJC.JM.OFFSET_RAD)},
              "placements_csv": str(EJC.PLACEMENTS.relative_to(_REPO)), "created": today, "datasets": {}}
    for name, rule in DATASETS.items():
        rows = collect(name, rule)
        res = analyse(rows)
        res["grasp_rule"], res["episodes"] = rule, rows
        result["datasets"][name] = res
        b = res["by_radius"]["3.4"]
        print(f"\n{name}  (n={res['n']}, grasp rule '{rule}', R=3.4 cm)")
        print(f"  {'model':<22}{'tx':>7}{'ty':>7}{'psi':>7}{'fit rms':>9}{'LOO rms':>9}")
        for m, v in b.items():
            p = v["params"]
            print(f"  {m:<22}{p.get('tx', 0):7.2f}{p.get('ty', 0):7.2f}{p.get('psi', 0):7.2f}{v['fit_rms_cm']:9.2f}{v['loo_rms_cm']:9.2f}")
        print(f"  best by LOO: {res['best_model_R3.4']}")
        for m, bs in res["bootstrap_R3.4"].items():
            print(f"  bootstrap 95% CI [{m}]: " + ", ".join(f"{k} {lo:+.2f}..{hi:+.2f}" for k, (lo, hi) in bs["ci95"].items()) + (f"   corr {bs['corr']}" if bs["corr"] else ""))
        print("  radius sensitivity (translation+rotation): " + "; ".join(
            f"R={R}: tx {v['translation+rotation']['params']['tx']:+.2f} ty {v['translation+rotation']['params']['ty']:+.2f} psi {v['translation+rotation']['params']['psi']:+.2f}"
            for R, v in res["by_radius"].items()))

    primary = "omx_pick_place_pilot_paper_cup_normal_A1"
    par = result["datasets"][primary]["by_radius"]["3.4"]["translation+rotation"]["params"]
    png = _REPO / "outputs" / "paper_to_pan" / f"{today}_normal_A1_grasps.png"
    overlay(primary, result["datasets"][primary]["episodes"], par, png,
            _REPO / "calibration/2026-10-07_camera_extrinsics_front-left.json", _REPO / "calibration/2026-10-07_camera_intrinsics_front-left.json")
    result["overlay_png"] = str(png.relative_to(_REPO))
    out = _REPO / "calibration" / f"{today}_paper_to_pan.json"
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(_REPO)} and {png.relative_to(_REPO)} (red = pinch as-is, green = pinch after the fit, cyan = cup base/rim from paper coordinates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

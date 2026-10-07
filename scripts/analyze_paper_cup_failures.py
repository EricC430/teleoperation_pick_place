#!/usr/bin/env python3
"""Join frozen placements, annotations and recorded-data predictions for diagnosis.

Run after the inference bundles documented in docs/meeting/2026-10-06_diagnosis.md
are available. Requires numpy, matplotlib and pyarrow; no robot, GPU or network.
Raw annotations are never rewritten. All MAEs are in LeRobot .pos units.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
OLD = "rollout_omx_b1_uvc60_100k_paper_cup_20260918_010912"
NEW = "rollout_omx_b1_uvc60_100k_nas30_A1_paper_cup_20261006_close_loop"
NEW_REPO = "ericc430/rollout_omx_b1_uvc60_100k_nas30_A1_paper_cup_20261006_094908"
NEW_REVISION = "f2f9ab38f6d0083e6bb5fc0a728cdc8a917b4688"


def read_annotations(name):
    raw = (ROOT / "episode_meta" / f"{name}.csv").read_bytes()
    for encoding in ("utf-8-sig", "cp950"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Unsupported annotation encoding: {name}")
    # The legacy file has a non-episode summary footer; retain only actual rows.
    rows = [r for r in csv.DictReader(io.StringIO(text)) if r["episode_index"].isdigit()]
    assert len({r["episode_index"] for r in rows}) == len(rows)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/paper_cup_diagnosis_20261006")
    parser.add_argument("--closed-dataset-root", type=Path,
                        default=ROOT / "data/huggingface/lerobot" / NEW_REPO)
    args = parser.parse_args()
    base = args.output_dir
    base.mkdir(parents=True, exist_ok=True)
    placements = {r["short_id"]: r for r in csv.DictReader(
        (ROOT / "docs/assets/placement_label_map_campA_136sym_20260908.csv").open())}
    for r in placements.values():
        r["x"] = float(r["x_pan_cm"])
        r["y"] = float(r["y_pan_cm"])
        r["radius"] = float(np.hypot(r["x"], r["y"]))
        r["band"] = "near_r<22" if r["radius"] < 22 else (
            "far_r>=34" if r["radius"] >= 34 else "middle_22<=r<34")
    demos = read_annotations("omx_pick_place_pilot_uvc_60")
    old, new = read_annotations(OLD), read_annotations(NEW)
    assert len(demos) == 60 and len(old) == len(new) == 36
    common_ids = {r["placement_id"] for r in old if r["valid"] == "1"} & {
        r["placement_id"] for r in new if r["valid"] == "1"}
    assert len(common_ids) == 32
    episode_rows, group_rows = [], []
    bundles = []
    for kind, label, prefix, source in [
        ("train", "N100", "t", base / "train_nas100"),
        ("train", "N30", "t", base / "train_nas30"),
        ("open", "N100", "o", ROOT / "outputs/open_loop_paper_cup/b1_100000_nas100"),
        ("open", "N50", "o", ROOT / "outputs/open_loop_paper_cup/b1_100000_nas50"),
        ("open", "N30", "o", base / "open_nas30"),
        ("open", "N20", "o", ROOT / "outputs/open_loop_paper_cup/b1_100000_nas20"),
    ]:
        path = source / "metrics.json"
        if not path.exists():
            print(f"Missing inference bundle: {path}")
            continue
        metrics = json.loads(path.read_text())
        assert metrics["weights_sha256_12"] == "2bdff2b3871e"
        count = 60 if kind == "train" else 12
        assert {m["episode"] for m in metrics["episodes"]} == set(range(count))
        items = []
        for m in metrics["episodes"]:
            sid = f"{prefix}{m['episode'] + 1}"
            p = placements[sid]
            row = {"kind": kind, "interval": label, "episode": m["episode"],
                   "placement": sid, "x_pan_cm": p["x"], "y_pan_cm": p["y"],
                   "radius_cm": p["radius"], "band": p["band"], "mae": m["mae"],
                   **{f"mae_{j}": m["mae_per_joint"][j] for j in JOINTS}}
            items.append(row)
            episode_rows.append(row)
        bundles.append((kind, label, items))
        groups = {"all": items, "original_main_r>=22": [r for r in items if r["radius_cm"] >= 22]}
        groups.update({band: [r for r in items if r["band"] == band]
                       for band in ["near_r<22", "middle_22<=r<34", "far_r>=34"]})
        for group, rows in groups.items():
            group_rows.append({"kind": kind, "interval": label, "group": group, "n": len(rows),
                               "mae": float(np.mean([r["mae"] for r in rows])),
                               **{f"mae_{j}": float(np.mean([r[f"mae_{j}"] for r in rows])) for j in JOINTS}})

    closed_rows = []
    for label, annotations in [("N100_20260918", old), ("N30_20261006", new)]:
        for cohort, selected in [("all36", annotations),
                                  ("declared_valid", [r for r in annotations if r["valid"] == "1"]),
                                  ("common32", [r for r in annotations if r["placement_id"] in common_ids])]:
            for group in ["all", "original_main_r>=22", "near_r<22", "middle_22<=r<34", "far_r>=34"]:
                rows = [r for r in selected if group == "all" or (
                    placements[r["placement_id"]]["radius"] >= 22 if group == "original_main_r>=22"
                    else placements[r["placement_id"]]["band"] == group)]
                successes = sum(r["outcome"] == "success" for r in rows)
                closed_rows.append({"run": label, "cohort": cohort, "group": group,
                                    "success": successes, "n": len(rows),
                                    "rate": successes / len(rows) if rows else None})
    old_by_id = {r["placement_id"]: r for r in old}
    paired = [{"placement": r["placement_id"], "band": placements[r["placement_id"]]["band"],
               "old_success": int(old_by_id[r["placement_id"]]["outcome"] == "success"),
               "new_success": int(r["outcome"] == "success"), "common32": r["placement_id"] in common_ids,
               "new_mechanism": r["mechanism"]} for r in new]
    write_csv(base / "episode_errors.csv", episode_rows)
    write_csv(base / "group_errors.csv", group_rows)
    write_csv(base / "closed_success.csv", closed_rows)
    write_csv(base / "paired_outcomes.csv", paired)

    manifest_path = base / "latest_dataset_manifest.json"
    manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists()
                else {"repo_id": NEW_REPO, "revision": NEW_REVISION})
    dataset_root = args.closed_dataset_root
    data = []
    for path in sorted((dataset_root / "data").rglob("*.parquet")):
        data.extend(pq.read_table(path).to_pylist())
    by_ep = {}
    for row in data:
        by_ep.setdefault(row["episode_index"], []).append(row)
    assert set(by_ep) == set(range(36))
    all_delta, all_phase, boundary_rows = [], [], []
    for ep, rows in by_ep.items():
        rows.sort(key=lambda r: r["frame_index"])
        a = np.asarray([r["action"] for r in rows])
        s = np.asarray([r["observation.state"] for r in rows])
        frames = np.asarray([r["frame_index"] for r in rows])
        delta = np.max(np.abs(np.diff(a[:, :5], axis=0)), axis=1)
        phase = frames[1:] % 30
        all_delta.extend(delta)
        all_phase.extend(phase)
        boundary_rows.append({"episode": ep, "placement": f"c{ep+1}", "frames": len(rows),
                              "boundary_mean": float(delta[phase == 0].mean()),
                              "boundary_max": float(delta[phase == 0].max()),
                              "nonboundary_mean": float(delta[phase != 0].mean()),
                              "action_state_gap_p95_body": float(np.percentile(np.abs(a[:, :5] - s[:, :5]), 95))})
    all_delta, all_phase = np.asarray(all_delta), np.asarray(all_phase)
    jumps = {}
    for name, values in [("boundary", all_delta[all_phase == 0]), ("nonboundary", all_delta[all_phase != 0])]:
        jumps[name] = {"n": len(values), "mean": float(values.mean()),
                       "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}
    jumps["mean_ratio"] = jumps["boundary"]["mean"] / jumps["nonboundary"]["mean"]
    write_csv(base / "closed_chunk_jumps.csv", boundary_rows)
    (base / "boundary_summary.json").write_text(json.dumps(jumps, indent=2))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {"near_r<22": "#2563eb", "middle_22<=r<34": "#d97706", "far_r>=34": "#dc2626"}
    for kind, label, items in bundles:
        if label in ("N100", "N30"):
            x = {("train", "N100"): 0, ("train", "N30"): 1, ("open", "N100"): 2, ("open", "N30"): 3}[kind, label]
            for offset, band in zip([-.24, 0, .24], colors):
                values = [r["mae"] for r in items if r["band"] == band]
                axes[0].bar(x + offset, np.mean(values), width=.22, color=colors[band])
                axes[0].text(x + offset, np.mean(values) + .12, f"{np.mean(values):.2f}", ha="center", fontsize=8)
    axes[0].set_xticks(range(4), ["Train N100", "Train N30", "Open N100", "Open N30"], rotation=15)
    axes[0].set_ylabel("Episode-mean MAE (.pos)")
    axes[0].set_title("Prediction error by radial band")
    axes[0].set_ylim(0, 8.4)
    axes[0].legend([plt.Rectangle((0,0),1,1,color=c) for c in colors.values()],
                   ["Near <22", "Middle 22-34", "Far >=34"], fontsize=8)
    for offset, label in [(-.18, "N100_20260918"), (.18, "N30_20261006")]:
        rows = [r for r in closed_rows if r["run"] == label and r["cohort"] == "common32" and r["group"] in colors]
        for idx, band in enumerate(colors):
            row = next(r for r in rows if r["group"] == band)
            axes[1].bar(idx + offset, 100*row["rate"], width=.34, color="#334155" if offset < 0 else "#14b8a6")
            axes[1].text(idx + offset, 100*row["rate"] + 2, f"{row['success']}/{row['n']}", ha="center")
    axes[1].set_xticks(range(3), ["Near <22", "Middle 22-34", "Far >=34"])
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Closed loop: common 32 placements")
    axes[1].set_ylabel("Success (%)")
    axes[1].legend([plt.Rectangle((0,0),1,1,color="#334155"), plt.Rectangle((0,0),1,1,color="#14b8a6")], ["N100 (Sep 18)", "N30 (Oct 6)"])
    axes[2].plot(range(30), [all_delta[all_phase == phase].mean() for phase in range(30)], marker="o", markersize=3)
    axes[2].axvline(0, color="#dc2626", alpha=.5)
    axes[2].set_title("N30: command jump vs frame modulo 30")
    axes[2].set_xlabel("Frame index modulo 30 (0 = refill)")
    axes[2].set_ylabel("Mean max-body |action[t]-action[t-1]| (.pos)")
    fig.tight_layout()
    fig.savefig(base / "comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)
    for p in placements.values():
        if p["short_id"].startswith("t"):
            axes[0].scatter(p["x"], p["y"], color=colors[p["band"]], s=25)
    axes[0].set_title("Training placements (n=60)")
    for ax, annotations, label in [(axes[1], old, "N100: all 36; gray = declared invalid"),
                                   (axes[2], new, "N30: all 36")]:
        for r in annotations:
            p = placements[r["placement_id"]]
            color = "#94a3b8" if r["valid"] == "0" else ("#16a34a" if r["outcome"] == "success" else "#dc2626")
            ax.scatter(p["x"], p["y"], color=color, s=35)
            ax.text(p["x"]+.3, p["y"]+.3, r["placement_id"], fontsize=7)
        ax.set_title(label)
    for ax in axes:
        for radius in [22, 34]:
            ax.add_patch(plt.Circle((0,0), radius, fill=False, color="#64748b", linestyle="--", linewidth=.8))
        ax.set_xlim(0,42); ax.set_ylim(-38,38); ax.set_aspect("equal")
        ax.set_xlabel("x from shoulder-pan axis (cm)")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("y from shoulder-pan axis (cm)")
    fig.tight_layout(); fig.savefig(base / "placements.png", dpi=180); plt.close(fig)

    # An episode with an annotated rush, with target commands and measured joint positions.
    rows = sorted(by_ep[1], key=lambda r:r["frame_index"])
    a = np.asarray([r["action"] for r in rows]); s = np.asarray([r["observation.state"] for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax,j in zip(axes,[1,2,3]):
        ax.plot(a[:,j], label="Command action", linewidth=1.5)
        ax.plot(s[:,j], label="Measured observation.state", linewidth=1)
        for frame in range(30,len(rows),30): ax.axvline(frame,color="#dc2626",alpha=.25)
        ax.set_ylabel(JOINTS[j]+" (.pos)"); ax.legend(loc="upper right")
    axes[0].set_title("c2 N30: red lines mark chunk refills; frame is nominal, not wall time")
    axes[-1].set_xlabel("Recorded episode frame")
    fig.tight_layout(); fig.savefig(base / "c2_chunk_boundaries.png",dpi=180); plt.close(fig)

    summary = {"group_errors": group_rows, "closed_success": closed_rows, "boundary_jumps": jumps,
               "closed_dataset": {"repo_id":manifest["repo_id"],"revision":manifest["revision"]},
               "grouping": {"near":"r<22 cm; D030 additions", "far":"r>=34 cm; exploratory threshold"},
               "units":"LeRobot .pos; not degrees; averages weight episodes equally"}
    input_paths = [ROOT / "docs/assets/placement_label_map_campA_136sym_20260908.csv",
                   ROOT / "episode_meta/omx_pick_place_pilot_uvc_60.csv",
                   ROOT / "episode_meta" / f"{OLD}.csv", ROOT / "episode_meta" / f"{NEW}.csv",
                   *sorted((dataset_root / "data").rglob("*.parquet"))]
    summary["input_sha256"] = {}
    for path in input_paths:
        try:
            label = str(path.relative_to(ROOT))
        except ValueError:
            label = str(path)
        summary["input_sha256"][label] = hashlib.sha256(path.read_bytes()).hexdigest()
    (base / "summary.json").write_text(json.dumps(summary, indent=2))
    for row in group_rows:
        if row["interval"] in ("N100", "N30"):
            print(row["kind"], row["interval"],row["group"],row["n"],round(row["mae"],4))
    print("Closed common32:", [r for r in closed_rows if r["cohort"] == "common32"])
    print("Boundary jumps:",jumps)
    print(f"Artifacts written to {base}")


if __name__ == "__main__":
    main()

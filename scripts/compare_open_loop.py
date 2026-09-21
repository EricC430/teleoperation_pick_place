#!/usr/bin/env python
"""Compare several eval_open_loop.py runs side by side and write the table to disk.

Each run is a directory given to eval_open_loop.py as --save_plot_dir. The script reads its
metrics.json (written by eval_open_loop.py since 2026-09-19); for older runs without one it
falls back to parsing the console log saved next to it as <dir>.log (Tee-Object output,
UTF-16 or UTF-8).

    uv run python scripts/compare_open_loop.py outputs/open_loop_alcan_alcan60 --runs 20k 40k 100k

writes into outputs/open_loop_alcan_alcan60/:
    comparison_episodes.csv   one row per episode, one MAE column per run
    comparison_joints.csv     one row per joint, mean-over-episodes MAE per run
    comparison.md             both tables + mean, and how often each run beats the first one
    overlay/ep<N>.png         ground truth + every run's prediction on one figure per episode
                              (needs trajectories.npz, written by eval_open_loop.py since 2026-09-19;
                              if any run lacks it, no overlays are drawn -- re-run that run)

Units are LeRobot .pos (body -100..100, gripper 0..100), not degrees.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

JOINT_ROW = re.compile(r"^(\w+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)")
EP_HEADER = re.compile(r"^Episode (\d+) Results:")


def read_log(path: Path) -> dict:
    raw = path.read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "replace")
    episodes, cur = [], None
    for line in text.splitlines():
        line = line.strip()
        m = EP_HEADER.match(line)
        if m:
            cur = {"episode": int(m.group(1)), "mae_per_joint": {}, "mse_per_joint": {}}
            episodes.append(cur)
            continue
        m = JOINT_ROW.match(line)
        if cur is not None and m:
            name, mae, mse = m.group(1), float(m.group(2)), float(m.group(3))
            if name == "OVERALL":
                cur["mae"], cur["mse"] = mae, mse
                cur = None
            else:
                cur["mae_per_joint"][name] = mae
                cur["mse_per_joint"][name] = mse
    if not episodes:
        raise SystemExit(f"no episode tables found in {path}")
    return {"episodes": episodes, "source": str(path)}


def load_run(run_dir: Path) -> dict:
    metrics = run_dir / "metrics.json"
    if metrics.exists():
        data = json.loads(metrics.read_text(encoding="utf-8"))
        data["source"] = str(metrics)
        return data
    log = run_dir.with_suffix(".log")
    if log.exists():
        return read_log(log)
    raise SystemExit(f"{run_dir}: neither metrics.json nor {log.name} found")


# Categorical slots 1-3 of the dataviz reference palette (validated light mode: CVD dE 9.2, normal 27.6).
# Slot 3 is below 3:1 contrast on white, so every run also gets its own dash pattern and a legend entry.
RUN_STYLES = [("#2a78d6", (0, (6, 2))), ("#eb6834", (0, (2, 2))), ("#1baf7a", (0, (8, 2, 2, 2)))]
GT_COLOR = "#2b2b29"


def plot_overlays(root: Path, names: list, out: Path, mae: dict, joints: list) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(names) > len(RUN_STYLES):
        print(f"overlay: {len(names)} runs but only {len(RUN_STYLES)} styles -- plotting the first {len(RUN_STYLES)}")
        names = names[: len(RUN_STYLES)]
    traj = {}
    for n in names:
        f = root / n / "trajectories.npz"
        if not f.exists():
            print(f"overlay: {f} missing -- re-run eval_open_loop.py for '{n}' to get overlay plots")
            return []
        traj[n] = np.load(f)
    fps = float(traj[names[0]]["fps"])
    (out / "overlay").mkdir(parents=True, exist_ok=True)
    written = []
    for ep in sorted(mae[names[0]]):
        gt = traj[names[0]][f"ep{ep}_gt"]
        t = np.arange(len(gt)) / fps
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
        for j, ax in enumerate(axes.flatten()[: len(joints)]):
            ax.plot(t, gt[:, j], color=GT_COLOR, linewidth=2.2, label="ground truth")
            for n, (color, dash) in zip(names, RUN_STYLES):
                ax.plot(t, traj[n][f"ep{ep}_pred"][:, j], color=color, linestyle=dash, linewidth=1.6,
                        label=f"{n} (ep MAE {mae[n][ep]:.2f})")
            ax.set_title(joints[j], fontsize=12)
            ax.grid(True, linestyle=":", alpha=0.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if j % 3 == 0:
                ax.set_ylabel(".pos units")
            if j >= 3:
                ax.set_xlabel("time (s)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(names) + 1, frameon=False, fontsize=11,
                   bbox_to_anchor=(0.5, 0.955))
        fig.suptitle(f"{root.name} -- episode {ep}", fontsize=14, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        path = out / "overlay" / f"ep{ep}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="directory holding one sub-directory per run")
    parser.add_argument("--runs", nargs="+", required=True, help="sub-directory names, in column order")
    parser.add_argument("--out", default=None, help="output directory (default: root)")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root
    out.mkdir(parents=True, exist_ok=True)
    runs = {name: load_run(root / name) for name in args.runs}

    ep_sets = [sorted(e["episode"] for e in r["episodes"]) for r in runs.values()]
    if any(s != ep_sets[0] for s in ep_sets):
        raise SystemExit(f"runs cover different episodes: {dict(zip(args.runs, ep_sets))}")
    episodes = ep_sets[0]
    mae = {n: {e["episode"]: e["mae"] for e in r["episodes"]} for n, r in runs.items()}
    joints = list(next(iter(runs.values()))["episodes"][0]["mae_per_joint"])
    joint_mean = {
        n: {j: sum(e["mae_per_joint"][j] for e in r["episodes"]) / len(r["episodes"]) for j in joints}
        for n, r in runs.items()
    }
    mean = {n: sum(mae[n].values()) / len(episodes) for n in args.runs}

    with open(out / "comparison_episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode"] + [f"mae_{n}" for n in args.runs])
        for ep in episodes:
            w.writerow([ep] + [f"{mae[n][ep]:.4f}" for n in args.runs])
        w.writerow(["mean"] + [f"{mean[n]:.4f}" for n in args.runs])
    with open(out / "comparison_joints.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["joint"] + [f"mae_{n}" for n in args.runs])
        for j in joints:
            w.writerow([j] + [f"{joint_mean[n][j]:.4f}" for n in args.runs])

    base = args.runs[0]
    lines = [f"# Open-loop comparison: {root}", "",
             "Units: LeRobot .pos (body -100..100, gripper 0..100), not degrees. Lower is better.", "",
             "Sources: " + ", ".join(f"`{n}` <- {runs[n]['source']}" for n in args.runs), "",
             "## Per episode (MAE)", "",
             "| episode | " + " | ".join(args.runs) + " |", "|---|" + "---|" * len(args.runs)]
    for ep in episodes:
        best = min(args.runs, key=lambda n: mae[n][ep])
        cells = [f"**{mae[n][ep]:.2f}**" if n == best else f"{mae[n][ep]:.2f}" for n in args.runs]
        lines.append(f"| {ep} | " + " | ".join(cells) + " |")
    lines.append("| **mean** | " + " | ".join(f"**{mean[n]:.3f}**" for n in args.runs) + " |")
    lines += ["", f"Episodes where each run beats `{base}`: "
              + ", ".join(f"`{n}` {sum(mae[n][e] < mae[base][e] for e in episodes)}/{len(episodes)}"
                          for n in args.runs[1:]), "",
              "## Per joint (mean MAE over episodes)", "",
              "| joint | " + " | ".join(args.runs) + " |", "|---|" + "---|" * len(args.runs)]
    for j in joints:
        lines.append(f"| {j} | " + " | ".join(f"{joint_mean[n][j]:.2f}" for n in args.runs) + " |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out / 'comparison.md'}, comparison_episodes.csv, comparison_joints.csv")
    plots = plot_overlays(root, args.runs, out, mae, joints)
    if plots:
        print(f"wrote {len(plots)} overlay plots to {out / 'overlay'}")


if __name__ == "__main__":
    main()

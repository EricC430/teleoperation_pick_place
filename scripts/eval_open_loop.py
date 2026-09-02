#!/usr/bin/env python
"""Open-loop action prediction evaluation for LeRobot policies on recorded datasets.

Evaluates a trained policy (e.g. ACT) against ground-truth actions on held-out episodes
without requiring a robot or simulator. Computes per-joint L1 MAE and MSE, and generates
visual comparison trajectory plots (Ground Truth vs ACT Prediction).
"""

import argparse
import logging
from pathlib import Path
import torch
import numpy as np
from termcolor import colored
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a policy open-loop against a dataset.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pretrained model checkpoint (e.g. data/train/phase_a_pilot/checkpoints/last/pretrained_model).",
    )
    parser.add_argument(
        "--dataset.repo_id",
        type=str,
        default="EricC430/omx_pick_place_pilot",
        dest="dataset_repo_id",
        help="Dataset repo_id.",
    )
    parser.add_argument(
        "--dataset.root",
        type=str,
        default=None,
        dest="dataset_root",
        help="Explicit root directory of the dataset.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[7],
        help="Episode indices to evaluate on (default: 7).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda / cpu).",
    )
    parser.add_argument(
        "--save_plot_dir",
        type=str,
        default=None,
        help="Directory to save comparison plot PNGs (defaults to run output dir / eval_plots).",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable trajectory plotting.",
    )
    return parser.parse_args()


def plot_episode_trajectories(ep_idx, joint_names, gt_actions, pred_actions, overall_mae, mae_per_joint, save_dir, fps=15.0):
    """Plot 6-joint trajectory comparison (Ground Truth vs Policy Prediction) and save to file."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_path = save_dir / f"eval_open_loop_ep{ep_idx}.png"

    num_joints = gt_actions.shape[1]
    time_steps = np.arange(len(gt_actions)) / fps

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()

    for j_idx in range(num_joints):
        ax = axes[j_idx]
        ax.plot(time_steps, gt_actions[:, j_idx], label="Ground Truth (Human)", color="#1f77b4", linewidth=2.2)
        ax.plot(time_steps, pred_actions[:, j_idx], label="ACT Prediction", color="#d62728", linestyle="--", linewidth=2.0)

        j_name = joint_names[j_idx] if j_idx < len(joint_names) else f"joint_{j_idx}"
        ax.set_title(f"{j_name} (MAE: {mae_per_joint[j_idx]:.2f}°)", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        if j_idx in [0, 3]:
            ax.set_ylabel("Joint Position (°)", fontsize=10)
        if j_idx >= 3:
            ax.set_xlabel("Time (s)", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)

    plt.suptitle(
        f"Phase A Open-Loop Evaluation — Episode {ep_idx} (Overall MAE: {overall_mae:.2f}°)\n"
        f"Total frames: {len(gt_actions)} ({len(gt_actions)/fps:.1f}s at {fps} fps)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logging.info(f"📊 Trajectory comparison plot saved to: {plot_path}")
    return plot_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    # Determine plot save directory (e.g. data/train/phase_a_pilot/eval_plots)
    if args.save_plot_dir:
        save_plot_dir = Path(args.save_plot_dir)
    else:
        # Check if checkpoint path is inside checkpoints/...
        parent_parts = [p.name for p in checkpoint_path.parents]
        if "checkpoints" in parent_parts:
            ckpt_idx = parent_parts.index("checkpoints")
            run_root = checkpoint_path.parents[ckpt_idx]
            save_plot_dir = run_root / "eval_plots"
        else:
            save_plot_dir = checkpoint_path.parent / "eval_plots"

    logging.info(f"Loading policy from {checkpoint_path} on {args.device}...")
    policy = ACTPolicy.from_pretrained(str(checkpoint_path))
    policy.eval()
    policy.to(args.device)

    logging.info("Initializing preprocessor and postprocessor pipelines for normalization...")
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint_path))

    feat = policy.config.output_features["action"]
    action_dim = feat.shape[0] if hasattr(feat, "shape") else feat["shape"][0]
    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ][:action_dim]

    logging.info(f"Evaluating {len(args.episodes)} episode(s): {args.episodes}...")

    per_episode_errors = []
    saved_plots = []

    for ep_idx in args.episodes:
        logging.info(
            f"Loading dataset '{args.dataset_repo_id}' (root={args.dataset_root}) for Episode {ep_idx}..."
        )
        dataset = LeRobotDataset(
            repo_id=args.dataset_repo_id,
            root=args.dataset_root,
            episodes=[ep_idx],
        )
        ep_length = len(dataset)
        fps = getattr(dataset.meta, "fps", 15.0)
        logging.info(f"\n--- Evaluating Episode {ep_idx} (total frames: {ep_length}, fps: {fps}) ---")

        pred_actions = []
        gt_actions = []

        policy.reset()

        for frame_idx in range(ep_length):
            item = dataset[frame_idx]

            # Build batch dictionary with batch dimension
            raw_batch = {}
            for k, v in item.items():
                if isinstance(v, torch.Tensor):
                    raw_batch[k] = v.unsqueeze(0)
                else:
                    raw_batch[k] = v

            # Preprocess observations (normalize images and states)
            proc_batch = preprocessor(raw_batch)
            proc_batch = {
                k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                for k, v in proc_batch.items()
            }

            gt_action = item["action"].numpy()
            gt_actions.append(gt_action)

            with torch.no_grad():
                pred_norm = policy.select_action(proc_batch)
                # Unnormalize action back to physical units (degrees / commands)
                pred_unnorm = postprocessor(pred_norm.cpu()).squeeze(0).numpy()
                pred_actions.append(pred_unnorm)

        pred_actions = np.array(pred_actions)
        gt_actions = np.array(gt_actions)

        # Compute errors
        l1_diff = np.abs(pred_actions - gt_actions)  # [T, D]
        mse_diff = (pred_actions - gt_actions) ** 2  # [T, D]

        mae_per_joint = np.mean(l1_diff, axis=0)
        mse_per_joint = np.mean(mse_diff, axis=0)
        overall_mae = np.mean(l1_diff)
        overall_mse = np.mean(mse_diff)

        per_episode_errors.append((ep_idx, overall_mae, overall_mse, mae_per_joint, mse_per_joint))

        print(f"\nEpisode {ep_idx} Results:")
        print(f"{'Joint':<18} | {'MAE (L1)':<12} | {'MSE':<12}")
        print("-" * 48)
        for j_name, j_mae, j_mse in zip(joint_names, mae_per_joint, mse_per_joint):
            print(f"{j_name:<18} | {j_mae:<12.4f} | {j_mse:<12.4f}")
        print("-" * 48)
        print(f"{'OVERALL':<18} | {overall_mae:<12.4f} | {overall_mse:<12.4f}")

        # Generate plot
        if not args.no_plot:
            plot_file = plot_episode_trajectories(
                ep_idx=ep_idx,
                joint_names=joint_names,
                gt_actions=gt_actions,
                pred_actions=pred_actions,
                overall_mae=overall_mae,
                mae_per_joint=mae_per_joint,
                save_dir=save_plot_dir,
                fps=fps,
            )
            saved_plots.append(plot_file)

    print("\n" + "=" * 50)
    print(colored("Open-Loop Evaluation Summary:", "green", attrs=["bold"]))
    for ep_idx, ep_mae, ep_mse, _, _ in per_episode_errors:
        print(f"  Episode {ep_idx}: MAE = {ep_mae:.4f}, MSE = {ep_mse:.4f}")
    mean_all_mae = np.mean([e[1] for e in per_episode_errors])
    mean_all_mse = np.mean([e[2] for e in per_episode_errors])
    print(colored(f"  Mean across episodes: MAE = {mean_all_mae:.4f}, MSE = {mean_all_mse:.4f}", "cyan"))
    if saved_plots:
        print(colored(f"  Saved plot(s) to: {save_plot_dir}", "yellow"))
    print("=" * 50)


if __name__ == "__main__":
    main()

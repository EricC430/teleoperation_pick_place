#!/usr/bin/env python3
"""
ACT Model Frame-by-Frame Attention & Layer Latent Visualizer.

Hooks into LeRobot's ACT policy during open-loop or closed-loop replay:
1. Captures Decoder Cross-Attention maps (Query -> Image Token patches).
2. Captures Encoder Self-Attention maps and Layer 1~4 latent representations.
3. Decodes tokens to spatial grids (Third-person Front-Left + Wrist Camera).
4. Generates an interactive Web analysis payload (JSON + frames).
5. Optionally renders a synchronized multi-panel diagnostic MP4 video.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ACT Attention & Layer Latents for diagnosis.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/train/phase_b1_uvc60/checkpoints/100000/pretrained_model",
        help="Path to pretrained model checkpoint.",
    )
    parser.add_argument(
        "--dataset.repo_id",
        type=str,
        default="ericc430/omx_pick_place_open_loop_eval",
        dest="dataset_repo_id",
        help="Dataset repo_id.",
    )
    parser.add_argument(
        "--dataset.root",
        type=str,
        default=None,
        dest="dataset_root",
        help="Explicit root directory of the dataset (e.g. data/huggingface/lerobot/ericc430/...).",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to analyze (default: 0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device (cuda / cpu).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/act_analysis",
        help="Root directory to store extracted analysis data and visual assets.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process (useful for quick verification).",
    )
    parser.add_argument(
        "--export-video",
        action="store_true",
        help="Export synchronized multi-panel diagnostic MP4 video.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        default=True,
        help="Save raw camera JPEG frames for the web viewer (default: True).",
    )
    return parser.parse_args()


class ACTAttentionExtractor:
    def __init__(self, policy, device="cuda"):
        self.policy = policy
        self.device = device
        self.model = policy.model

        self.captured_cross_attn = []
        self.captured_encoder_latents = {}
        self.captured_encoder_self_attn = {}

        self._hook_handles = []
        self._setup_hooks()

    def _setup_hooks(self):
        # 1. Decoder Cross-Attention:
        # In ACTDecoderLayer: self.multihead_attn is nn.MultiheadAttention.
        # We wrap its forward method to force need_weights=True and average_attn_weights=False.
        decoder_layer = self.model.decoder.layers[0]
        orig_dec_mha_forward = decoder_layer.multihead_attn.forward

        def hooked_dec_mha_forward(*args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            out, weights = orig_dec_mha_forward(*args, **kwargs)
            self.captured_cross_attn.append(weights.detach().cpu())
            return out, weights

        decoder_layer.multihead_attn.forward = hooked_dec_mha_forward
        self._orig_dec_mha_forward = orig_dec_mha_forward

        # 2. Encoder Layers Latent Forward Hooks:
        for i, layer in enumerate(self.model.encoder.layers):
            def make_latent_hook(idx):
                def hook_fn(mod, inp, out):
                    self.captured_encoder_latents[idx] = out.detach().cpu()
                return hook_fn
            handle = layer.register_forward_hook(make_latent_hook(i))
            self._hook_handles.append(handle)

            # Also hook Encoder Self-Attention
            orig_enc_sa_forward = layer.self_attn.forward

            def make_sa_hook(idx, orig_fn):
                def hooked_sa(*args, **kwargs):
                    kwargs["need_weights"] = True
                    kwargs["average_attn_weights"] = False
                    out, weights = orig_fn(*args, **kwargs)
                    self.captured_encoder_self_attn[idx] = weights.detach().cpu()
                    return out, weights
                return hooked_sa

            layer.self_attn.forward = make_sa_hook(i, orig_enc_sa_forward)
            setattr(self, f"_orig_enc_sa_forward_{i}", orig_enc_sa_forward)

    def reset_captures(self):
        self.captured_cross_attn.clear()
        self.captured_encoder_latents.clear()
        self.captured_encoder_self_attn.clear()


def tensor_to_bgr_image(tensor):
    """Convert (3, H, W) float tensor [0, 1] to (H, W, 3) uint8 BGR for OpenCV."""
    arr = tensor.detach().cpu().numpy()
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.max() <= 1.0:
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    else:
        arr = arr.clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def compute_heatmap_overlay(orig_bgr, norm_map_2d, colormap=cv2.COLORMAP_TURBO, alpha=0.55):
    """Overlay a 2D normalized [0, 1] map onto the original BGR image."""
    h, w = orig_bgr.shape[:2]
    # Resize map to image dimensions using bicubic interpolation
    resized = cv2.resize(norm_map_2d, (w, h), interpolation=cv2.INTER_CUBIC)
    resized = np.clip(resized, 0.0, 1.0)
    colored = cv2.applyColorMap((resized * 255).astype(np.uint8), colormap)
    blended = cv2.addWeighted(orig_bgr, 1.0 - alpha, colored, alpha, 0)
    return blended


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    # Auto-detect run name
    run_name = checkpoint_path.parts[-4] if "checkpoints" in checkpoint_path.parts else checkpoint_path.name
    output_base = Path(args.output_dir) / f"{run_name}_ep{args.episode}"
    frames_dir = output_base / "frames"
    frames_wrist_dir = frames_dir / "wrist"
    frames_front_dir = frames_dir / "front_left"

    frames_wrist_dir.mkdir(parents=True, exist_ok=True)
    frames_front_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading ACT Policy from {checkpoint_path} on {args.device}...")
    policy_cls = get_policy_class("act")
    policy = policy_cls.from_pretrained(str(checkpoint_path))
    policy.eval()
    policy.to(args.device)

    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint_path))

    # Identify image camera keys
    cam_keys = list(policy.config.image_features)
    logging.info(f"Model image features ({len(cam_keys)} cameras): {cam_keys}")

    # Dataset loading
    logging.info(f"Loading dataset '{args.dataset_repo_id}' (root={args.dataset_root}) Episode {args.episode}...")
    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        episodes=[args.episode],
    )
    total_frames = len(dataset)
    fps = getattr(dataset.meta, "fps", 15.0)
    if args.max_frames and args.max_frames < total_frames:
        total_frames = args.max_frames
    logging.info(f"Processing Episode {args.episode}: {total_frames} frames (fps={fps})")

    # Action feature dimensions
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

    # Initialize Extractor
    extractor = ACTAttentionExtractor(policy, device=args.device)

    # Determine token layout by running a probe on frame 0
    probe_item = dataset[0]
    probe_batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in probe_item.items()}
    probe_batch_proc = preprocessor(probe_batch)
    probe_batch_proc = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in probe_batch_proc.items()}

    cam_token_info = {}
    total_image_tokens = 0
    with torch.no_grad():
        for c_idx, c_key in enumerate(cam_keys):
            feat_map = policy.model.backbone(probe_batch_proc[c_key])["feature_map"]
            _, _, f_h, f_w = feat_map.shape
            num_tokens = f_h * f_w
            cam_token_info[c_key] = {
                "idx": c_idx,
                "h": f_h,
                "w": f_w,
                "num_tokens": num_tokens,
            }
            total_image_tokens += num_tokens

    # In ACT forward:
    # Token 0: VAE latent token
    # Token 1: Robot state token (if config.robot_state_feature)
    token_offset = 1 + (1 if policy.config.robot_state_feature else 0)
    for c_key in cam_keys:
        cam_token_info[c_key]["start"] = token_offset
        cam_token_info[c_key]["end"] = token_offset + cam_token_info[c_key]["num_tokens"]
        token_offset = cam_token_info[c_key]["end"]

    logging.info(f"Token layout mapping: {cam_token_info} (Total sequence tokens: {token_offset})")

    # Pre-allocate trajectory and attention storage
    gt_actions = []
    pred_actions_first = []
    all_pred_chunks = []

    # Frame-by-frame diagnostic metrics
    frames_data = []

    logging.info("Starting frame-by-frame inference and attention extraction...")
    policy.reset()

    # OpenCV Video Writer setup if requested
    video_writer = None
    diag_video_path = output_base / f"ep{args.episode}_diagnostic.mp4"
    temp_video_path = output_base / f"ep{args.episode}_temp.mp4"

    target_video_w = 1280
    target_video_h = 720

    if args.export_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (target_video_w, target_video_h))

    for frame_idx in tqdm(range(total_frames), desc="Frames"):
        item = dataset[frame_idx]
        raw_batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in item.items()}

        # Save raw frame images for Web viewer
        wrist_raw = tensor_to_bgr_image(item["observation.images.wrist"])
        front_raw = tensor_to_bgr_image(item["observation.images.front-left"])

        wrist_img_path = frames_wrist_dir / f"frame_{frame_idx:04d}.jpg"
        front_img_path = frames_front_dir / f"frame_{frame_idx:04d}.jpg"

        if args.save_images:
            cv2.imwrite(str(wrist_raw_path := wrist_img_path), wrist_raw, [cv2.IMWRITE_JPEG_QUALITY, 85])
            cv2.imwrite(str(front_raw_path := front_img_path), front_raw, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Preprocess for model
        proc_batch = preprocessor(raw_batch)
        proc_batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in proc_batch.items()}

        gt_act = item["action"].numpy().tolist()
        gt_actions.append(gt_act)

        extractor.reset_captures()

        with torch.no_grad():
            chunk_norm = policy.predict_action_chunk(proc_batch)
            chunk_unnorm = postprocessor(chunk_norm.cpu()).squeeze(0).numpy()  # [100, 6]
            pred_first = chunk_unnorm[0]  # [6]
            pred_actions_first.append(pred_first.tolist())
            # Subsample chunk for web viewer (e.g. every 10 frames or key frames)
            if frame_idx % 5 == 0 or frame_idx == total_frames - 1:
                all_pred_chunks.append({
                    "frame_idx": frame_idx,
                    "chunk": chunk_unnorm[::2].tolist(),  # [50, 6] for compact transfer
                })

        # Retrieve captured attention and latents
        if not extractor.captured_cross_attn:
            logging.error(f"Frame {frame_idx}: No cross-attention captured!")
            continue

        # Cross-attn shape: [1, heads=8, chunk=100, tokens=707]
        cross_attn = extractor.captured_cross_attn[0][0]  # [8, 100, tokens]
        cross_attn_heads_mean = cross_attn.mean(dim=0).numpy()  # [100, tokens]

        # Extract camera attention maps for query t=0, t=10, t=20, t=50, and chunk average
        cam_heatmaps = {}
        for c_key in cam_keys:
            info = cam_token_info[c_key]
            tokens_slice = cross_attn_heads_mean[:, info["start"] : info["end"]]  # [100, H*W]

            # Normalize to 2D
            t0_map = tokens_slice[0].reshape(info["h"], info["w"])
            t10_map = tokens_slice[min(10, tokens_slice.shape[0]-1)].reshape(info["h"], info["w"])
            t20_map = tokens_slice[min(20, tokens_slice.shape[0]-1)].reshape(info["h"], info["w"])
            t50_map = tokens_slice[min(50, tokens_slice.shape[0]-1)].reshape(info["h"], info["w"])
            avg_map = tokens_slice.mean(axis=0).reshape(info["h"], info["w"])

            # Normalized 0..1 for storage / rendering (rounded to 3 decimals)
            def norm_01(m):
                m_min, m_max = m.min(), m.max()
                if m_max - m_min > 1e-8:
                    return np.round((m - m_min) / (m_max - m_min), 3).tolist()
                return np.zeros_like(m).tolist()

            cam_heatmaps[c_key] = {
                "t0": norm_01(t0_map),
                "t10": norm_01(t10_map),
                "t20": norm_01(t20_map),
                "t50": norm_01(t50_map),
                "chunk_avg": norm_01(avg_map),
            }

        # Encoder Layers Latent Analysis (Layers 0..3)
        # Latent shape: [tokens=707, batch=1, dim=512]
        layer_latent_stats = {}
        for layer_idx in range(len(policy.model.encoder.layers)):
            z = extractor.captured_encoder_latents[layer_idx].squeeze(1).numpy()  # [707, 512]
            token_norms = np.linalg.norm(z, axis=-1)  # [707]

            layer_cam_norms = {}
            for c_key in cam_keys:
                info = cam_token_info[c_key]
                cam_t_norms = token_norms[info["start"] : info["end"]].reshape(info["h"], info["w"])
                # Normalize 0..1 for visualization
                n_min, n_max = cam_t_norms.min(), cam_t_norms.max()
                normed = (cam_t_norms - n_min) / (n_max - n_min + 1e-8)
                layer_cam_norms[c_key] = np.round(normed, 3).tolist()

            # Cosine similarity with previous layer
            cos_sim_val = 1.0
            if layer_idx > 0:
                prev_z = extractor.captured_encoder_latents[layer_idx - 1].squeeze(1).numpy()
                norm_z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
                norm_prev_z = prev_z / (np.linalg.norm(prev_z, axis=-1, keepdims=True) + 1e-8)
                cos_sim_per_token = np.sum(norm_z * norm_prev_z, axis=-1)  # [707]
                cos_sim_val = float(np.mean(cos_sim_per_token))

            layer_latent_stats[f"layer_{layer_idx}"] = {
                "mean_norm": float(np.mean(token_norms)),
                "cosine_sim_with_prev": cos_sim_val,
                "cam_norm_maps": layer_cam_norms,
            }

        # Calculate self-attention entropy (focus measure)
        # Higher entropy = diffused attention; lower entropy = sharp focus
        entropy_per_layer = []
        for l_idx in range(len(policy.model.encoder.layers)):
            if l_idx in extractor.captured_encoder_self_attn:
                sa = extractor.captured_encoder_self_attn[l_idx][0].mean(dim=0).numpy()  # [707, 707]
                sa_entropy = -np.sum(sa * np.log(sa + 1e-10), axis=-1).mean()
                entropy_per_layer.append(float(sa_entropy))
            else:
                entropy_per_layer.append(0.0)

        frame_record = {
            "frame_idx": frame_idx,
            "time_sec": round(frame_idx / fps, 3),
            "heatmaps": cam_heatmaps,
            "layer_latents": layer_latent_stats,
            "entropy_per_layer": entropy_per_layer,
        }
        frames_data.append(frame_record)

        # Video Rendering
        if video_writer is not None:
            # Create composite frame:
            # Left: Front-Left Cam + Heatmap Overlay
            # Right: Wrist Cam + Heatmap Overlay
            # Bottom: Trajectory progress bar & Layer Latent status
            front_map = np.array(cam_heatmaps["observation.images.front-left"]["t0"], dtype=np.float32)
            wrist_map = np.array(cam_heatmaps["observation.images.wrist"]["t0"], dtype=np.float32)

            front_overlay = compute_heatmap_overlay(front_raw, front_map, alpha=0.5)
            wrist_overlay = compute_heatmap_overlay(wrist_raw, wrist_map, alpha=0.5)

            # Resize both to fit top row: e.g. 640x360 each -> 1280x360
            top_h = 420
            front_resized = cv2.resize(front_overlay, (640, top_h))
            wrist_resized = cv2.resize(wrist_overlay, (640, top_h))

            # Add labels
            cv2.putText(front_resized, "Front-Left Cam (3rd-person) + ACT Attn (t=0)", (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(wrist_resized, "Wrist Cam + ACT Attn (t=0)", (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            top_row = np.hstack([front_resized, wrist_resized])

            # Bottom info panel (1280 x 300)
            bottom_panel = np.zeros((target_video_h - top_h, target_video_w, 3), dtype=np.uint8)
            bottom_panel[:] = (24, 24, 28)

            # Draw trajectory info
            curr_time = frame_idx / fps
            cv2.putText(bottom_panel, f"Episode: {args.episode} | Frame: {frame_idx}/{total_frames} | Time: {curr_time:.2f}s",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            # Latent summary
            l_info = " | ".join([f"L{i} CosSim: {layer_latent_stats[f'layer_{i}']['cosine_sim_with_prev']:.2f}" for i in range(1, 4)])
            cv2.putText(bottom_panel, f"Encoder Latent Stability:  {l_info}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 200, 240), 1, cv2.LINE_AA)

            # Draw mini timeline bar
            bar_x, bar_y, bar_w, bar_h = 20, 95, target_video_w - 40, 16
            cv2.rectangle(bottom_panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
            progress = int(bar_w * (frame_idx / max(1, total_frames - 1)))
            cv2.rectangle(bottom_panel, (bar_x, bar_y), (bar_x + progress, bar_y + bar_h), (0, 200, 255), -1)

            # Draw Gripper & Height (shoulder_lift) Joint values
            gt_grip = gt_act[5] if len(gt_act) > 5 else 0.0
            pred_grip = pred_first[5] if len(pred_first) > 5 else 0.0
            gt_lift = gt_act[1] if len(gt_act) > 1 else 0.0
            pred_lift = pred_first[1] if len(pred_first) > 1 else 0.0

            cv2.putText(bottom_panel, f"Gripper Pos: GT={gt_grip:.2f}, Pred={pred_grip:.2f}",
                        (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 100), 2, cv2.LINE_AA)
            cv2.putText(bottom_panel, f"Shoulder Lift: GT={gt_lift:.2f}, Pred={pred_lift:.2f}",
                        (450, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 220, 255), 2, cv2.LINE_AA)

            # Guidance Note
            cv2.putText(bottom_panel, "Inspect AoTR & Interactive ROI Bounding Boxes in ACT Web Inspector",
                        (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)

            full_frame = np.vstack([top_row, bottom_panel])
            video_writer.write(full_frame)

    if video_writer is not None:
        video_writer.release()
        logging.info(f"Converting raw video with ffmpeg to H.264 MP4: {diag_video_path}")
        # Transcode to standard H.264 for universal browser / IDE playback
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", str(temp_video_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
            str(diag_video_path)
        ]
        res = subprocess.run(ffmpeg_cmd, capture_output=True)
        if res.returncode == 0:
            temp_video_path.unlink(missing_ok=True)
            logging.info(f"✅ Diagnostic MP4 saved successfully to: {diag_video_path}")
        else:
            logging.warning(f"ffmpeg conversion note: {res.stderr.decode()[:300]}")
            shutil.move(str(temp_video_path), str(diag_video_path))

    # Save comprehensive JSON payload
    json_payload = {
        "metadata": {
            "checkpoint": str(checkpoint_path),
            "run_name": run_name,
            "dataset_repo_id": args.dataset_repo_id,
            "episode": args.episode,
            "total_frames": total_frames,
            "fps": fps,
            "joint_names": joint_names,
            "cameras": {
                "front-left": {
                    "key": "observation.images.front-left",
                    "orig_shape": [480, 848],
                    "grid_shape": [cam_token_info["observation.images.front-left"]["h"],
                                   cam_token_info["observation.images.front-left"]["w"]],
                    "frames_path": f"frames/front_left/frame_%04d.jpg",
                },
                "wrist": {
                    "key": "observation.images.wrist",
                    "orig_shape": [480, 640],
                    "grid_shape": [cam_token_info["observation.images.wrist"]["h"],
                                   cam_token_info["observation.images.wrist"]["w"]],
                    "frames_path": f"frames/wrist/frame_%04d.jpg",
                },
            },
        },
        "trajectories": {
            "gt_actions": gt_actions,
            "pred_actions": pred_actions_first,
            "pred_chunks": all_pred_chunks,
        },
        "frames": frames_data,
    }

    json_path = output_base / f"ep{args.episode}_data.json"
    logging.info(f"Writing analysis JSON to: {json_path}")
    with open(json_path, "w") as f:
        json.dump(json_payload, f)

    logging.info(f"🎉 ACT Attention & Latent extraction complete! Output dir: {output_base}")
    print(f"\n[DONE] Analysis bundle created at: {output_base}")
    print(f"Data file: {json_path}")
    if args.export_video:
        print(f"Video file: {diag_video_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Automated Teleoperation Offset Measurement Tool (field_manual.md §4-1).

Connects to Leader and Follower arms, runs live teleoperation, guides the operator
through the 5 standard test poses, automatically samples stationary joint angles,
computes offsets (Follower - Leader), and writes to analysis/teleop_offset_<date>.csv.

Usage:
    uv run python scripts/measure_teleop_offset.py --config-path configs/teleoperate_omx.yaml
    uv run python scripts/measure_teleop_offset.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import yaml

POSES = [
    ("1_home", "Home / 零點姿態 (直立或初始標定零點)"),
    ("2_j1_mid", "J1 (shoulder_pan) 轉動至約 +45° 或中點"),
    ("3_j2_mid", "J2 (shoulder_lift) 轉動至約中點 (負重最大軸)"),
    ("4_j3_mid", "J3 (elbow_flex) 轉動至約中點"),
    ("5_gripper", "夾爪 (gripper) 全開後全閉"),
]

JOINTS_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _today() -> str:
    return _dt.date.today().isoformat()


def _now_str() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semi-automated teleoperation offset measurement tool (§4-1)"
    )
    parser.add_argument(
        "--config-path",
        default="configs/teleoperate_omx.yaml",
        help="Path to teleoperation YAML configuration",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Path to output CSV (default: analysis/teleop_offset_<date>.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate measurement without physical hardware",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=30,
        help="Number of frames to average per pose (default: 30 frames, ~0.5s)",
    )
    return parser


def check_key_pressed() -> str | None:
    """Non-blocking keyboard check for Windows/Linux."""
    if os.name == "nt":
        import msvcrt
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            try:
                return ch.decode("utf-8")
            except UnicodeDecodeError:
                return ""
        return None
    else:
        import select
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            return sys.stdin.read(1)
        return None


def extract_joint_val(data_dict: dict[str, Any] | None, joint_name: str) -> float | None:
    """Extract joint angle/position float from dictionary handling namespacing.
    
    Supports:
        - 'shoulder_pan': 12.3
        - 'shoulder_pan.pos': 12.3
        - 'observation.state.shoulder_pan.pos': 12.3
        - 'state.shoulder_pan': 12.3
    """
    if not data_dict:
        return None
    
    if joint_name in data_dict:
        try:
            return float(data_dict[joint_name])
        except (ValueError, TypeError):
            pass

    if f"{joint_name}.pos" in data_dict:
        try:
            return float(data_dict[f"{joint_name}.pos"])
        except (ValueError, TypeError):
            pass

    for k, v in data_dict.items():
        parts = str(k).split(".")
        if joint_name in parts:
            try:
                return float(v)
            except (ValueError, TypeError):
                continue
    return None


def run_dry_run(out_path: Path) -> None:
    print("\n[DRY RUN] 模擬 5 個姿態自動取樣與分析流程...")
    records = []
    for pose_id, desc in POSES:
        print(f"\n--- 模擬姿態 {pose_id} ({desc}) ---")
        ts = _now_str()
        for joint in JOINTS_ORDER:
            leader_val = 0.0 if pose_id == "1_home" else 45.0
            follower_val = leader_val + float(np.random.uniform(-0.8, 0.8))
            delta = follower_val - leader_val
            records.append({
                "time": ts,
                "pose_id": pose_id,
                "joint": joint,
                "leader_deg": f"{leader_val:.2f}",
                "follower_deg": f"{follower_val:.2f}",
                "delta_deg": f"{delta:+.2f}",
                "note": "dry-run auto sample",
            })
    
    _write_records(out_path, records)
    print(f"\n[DRY RUN 完成] 模擬結果已寫入: {out_path}")


def _write_records(out_path: Path, records: list[dict[str, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["time", "pose_id", "joint", "leader_deg", "follower_deg", "delta_deg", "note"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    today_str = _today()
    out_csv = Path(args.output_csv) if args.output_csv else Path(f"analysis/teleop_offset_{today_str}.csv")

    if args.dry_run:
        run_dry_run(out_csv)
        return 0

    if not Path(args.config_path).exists():
        print(f"錯誤: 找不到設定檔 {args.config_path}", file=sys.stderr)
        return 1

    print("=" * 60)
    print("🤖 遙操作偏移量自動量測工具 (§4-1)")
    print(f"設定檔: {args.config_path}")
    print(f"輸出檔: {out_csv}")
    print("=" * 60)

    # Lazy import LeRobot components
    try:
        from lerobot.robots import make_robot_from_config
        from lerobot.teleoperators import make_teleoperator_from_config
        import draccus
        from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig
    except ImportError as e:
        print(f"載入 LeRobot 失敗: {e}", file=sys.stderr)
        return 1

    with open(args.config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    # Disable cameras during offset measurement to ensure highest polling rate
    if "robot" in cfg_dict and "cameras" in cfg_dict["robot"]:
        cfg_dict["robot"].pop("cameras", None)

    cfg = draccus.decode(TeleoperateConfig, cfg_dict)

    print("\n連線 Leader 與 Follower 手臂...")
    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    teleop.connect()
    robot.connect()
    print("✅ 雙臂連線成功！即時遙操作已啟動。\n")

    records = []

    try:
        for idx, (pose_id, desc) in enumerate(POSES, start=1):
            print("\n" + "=" * 60)
            print(f"📌 姿態 [{idx}/{len(POSES)}]: {pose_id}")
            print(f"👉 請遙控操作手臂擺至: 【{desc}】")
            print("👉 移動到位後請【靜置 2~3 秒】，然後在終端機按下 [Enter] 進行自動取樣...")
            print("=" * 60)

            # Keep teleop running while waiting for Enter
            while True:
                raw_action = teleop.get_action()
                robot.send_action(raw_action)
                key = check_key_pressed()
                if key in ("\r", "\n", " "):
                    break
                time.sleep(0.01)

            print(f"⏳ 正在連續取樣 {args.samples} 幀關節角度並做多幀平均...")
            sample_time = _now_str()
            leader_samples: dict[str, list[float]] = {j: [] for j in JOINTS_ORDER}
            follower_samples: dict[str, list[float]] = {j: [] for j in JOINTS_ORDER}

            for _ in range(args.samples):
                raw_action = teleop.get_action()
                robot.send_action(raw_action)
                obs = robot.get_observation()

                for j in JOINTS_ORDER:
                    # 1. Try extracting from raw_action
                    l_val = extract_joint_val(raw_action, j)
                    # 2. Fallback to direct bus read if available
                    if l_val is None and hasattr(teleop, "bus") and hasattr(teleop.bus, "sync_read"):
                        try:
                            bus_dict = teleop.bus.sync_read("Present_Position")
                            l_val = extract_joint_val(bus_dict, j)
                        except Exception:
                            pass

                    if l_val is not None:
                        leader_samples[j].append(l_val)

                    # Follower observation extraction
                    f_val = extract_joint_val(obs, j)
                    if f_val is None and hasattr(robot, "bus") and hasattr(robot.bus, "sync_read"):
                        try:
                            bus_dict = robot.bus.sync_read("Present_Position")
                            f_val = extract_joint_val(bus_dict, j)
                        except Exception:
                            pass

                    if f_val is not None:
                        follower_samples[j].append(f_val)

                time.sleep(0.02)

            print(f"\n📊 姿態 {pose_id} 取樣結果 (取樣時間: {sample_time}):")
            print(f"{'關節名稱':<15} | {'Leader(°)':>10} | {'Follower(°)':>12} | {'Delta(°)':>10} | {'診斷'}")
            print("-" * 65)

            for j in JOINTS_ORDER:
                l_mean = float(np.mean(leader_samples[j])) if leader_samples[j] else 0.0
                f_mean = float(np.mean(follower_samples[j])) if follower_samples[j] else 0.0
                delta = f_mean - l_mean

                diag = "✅ 正常"
                if abs(delta) < 2.0:
                    diag = "✅ 良好"
                elif abs(delta) < 5.0:
                    diag = "🟡 輕微背隙/負載"
                else:
                    diag = "🔴 偏差較大"

                print(f"{j:<15} | {l_mean:>10.2f} | {f_mean:>12.2f} | {delta:>+10.2f} | {diag}")

                records.append({
                    "time": sample_time,
                    "pose_id": pose_id,
                    "joint": j,
                    "leader_deg": f"{l_mean:.2f}",
                    "follower_deg": f"{f_mean:.2f}",
                    "delta_deg": f"{delta:+.2f}",
                    "note": diag,
                })

        _write_records(out_csv, records)
        print("\n" + "=" * 60)
        print(f"🎉 5 個姿態量測全部完成！")
        print(f"💾 結果已自動儲存至: {out_csv}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n操作已手動中斷。")
    finally:
        teleop.disconnect()
        robot.disconnect()
        print("雙臂已安全中斷連線。")

    return 0


if __name__ == "__main__":
    sys.exit(main())

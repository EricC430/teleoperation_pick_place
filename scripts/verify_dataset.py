#!/usr/bin/env python3
"""
錄製後立即執行的資料一致性驗證。

為什麼需要這支腳本
------------------
LeRobot 的資料集有一個「錄製當下不會報錯、訓練到那一段才炸」的失效模式：
`meta/info.json` 宣稱的幀數，與影片實際幀數不符。

我們已經被這個咬過一次（見 docs/pipeline_validation.md）：
公開資料集 edgarcancinoe/soarm101_pickplace_orange_080e_ts_closed 宣稱 61,480 幀，
實際 61,534 幀，差額全在 file-000。100 步 smoke test 全綠，1000 步時第 167 步崩潰：

    IndexError: Invalid frame index=8530 for streamIndex=0; must be less than 8524

⚠️ 而且 `--dataset.exclude_episodes` 救不了 —— sampler 的索引空間仍由錯誤的總幀數建立。
唯一的解是重錄。所以要在「錄完當下」就抓到，不是等訓練。

用法
----
    python scripts/verify_dataset.py <dataset_root>

    # 例：
    python scripts/verify_dataset.py data/huggingface/lerobot/<user>/so100_pick_place

檢查三件事
----------
1. info.json 宣稱的總幀數  ==  各 episode parquet 行數總和
2. 每個 episode 的影片實際幀數  ==  該 episode 的 parquet 行數
3. timestamp 欄位有沒有異常間隔（掉幀的直接證據）

退出碼：0 = 全部通過，1 = 有問題（可接進 CI 或錄製腳本尾端）

⚠️ 版本相依
-----------
LeRobot 的 dataset 目錄結構在 v2.0 / v3.0 之間有變動。
本腳本用「探索」而非「硬編路徑」的方式尋找檔案，但若你的 LeRobot 版本輸出結構不同，
請先跑一次確認它找得到東西，不要盲信「PASS」。
實測版本：lerobot 0.6.2
"""

import json
import subprocess
import sys
from pathlib import Path


def find_one(root: Path, *patterns: str):
    """回傳第一個符合任一 glob 的路徑，找不到回 None。"""
    for pat in patterns:
        hits = sorted(root.glob(pat))
        if hits:
            return hits[0]
    return None


def count_video_frames(video: Path) -> int | None:
    """用 ffprobe 實際逐幀計數。慢但準——不要用 nb_frames，那是 metadata，會騙人。"""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries", "stream=nb_read_frames",
             "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, timeout=600,
        )
        val = out.stdout.strip().split(",")[0]
        return int(val) if val.isdigit() else None
    except FileNotFoundError:
        print("  ⚠️  找不到 ffprobe。請安裝 ffmpeg，否則無法驗證影片幀數。")
        return None
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️  ffprobe 失敗：{e}")
        return None


def read_parquet_rows(path: Path) -> int | None:
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️  讀不到 {path.name}：{e}")
        return None


def check_timestamps(path: Path, fps: float) -> list[str]:
    """timestamp 差分若明顯大於 1/fps，就是掉幀的直接證據。"""
    problems = []
    try:
        import pyarrow.parquet as pq
        tbl = pq.read_table(path)
        if "timestamp" not in tbl.column_names:
            return []
        ts = tbl.column("timestamp").to_pylist()
        if len(ts) < 2:
            return []
        expected = 1.0 / fps
        for i in range(1, len(ts)):
            gap = ts[i] - ts[i - 1]
            # 容忍 1.5 倍；超過就是掉了至少一幀
            if gap > expected * 1.5:
                problems.append(
                    f"    frame {i}: 間隔 {gap:.4f}s（預期 {expected:.4f}s，約掉 {gap/expected - 1:.1f} 幀）"
                )
        return problems
    except Exception:  # noqa: BLE001
        return []


def main(root: Path) -> int:
    print(f"驗證：{root}\n")
    ok = True

    info_path = find_one(root, "meta/info.json", "info.json", "**/meta/info.json")
    if info_path is None:
        print("❌ 找不到 meta/info.json —— 這不是 LeRobot 資料集根目錄，或結構與預期不同。")
        return 1
    info = json.loads(info_path.read_text(encoding="utf-8"))

    fps = float(info.get("fps", 0) or 0)
    declared_total = info.get("total_frames")
    declared_eps = info.get("total_episodes")
    print(f"info.json：fps={fps}  total_episodes={declared_eps}  total_frames={declared_total}\n")

    parquets = sorted(root.glob("data/**/*.parquet")) or sorted(root.glob("**/*.parquet"))
    if not parquets:
        print("❌ 找不到任何 parquet 檔。")
        return 1

    # ---- 檢查 1：總幀數 ----
    print("─" * 60)
    print("檢查 1：info.json 總幀數 vs parquet 實際行數總和")
    rows_per_file = {}
    total_rows = 0
    for p in parquets:
        n = read_parquet_rows(p)
        if n is None:
            ok = False
            continue
        rows_per_file[p] = n
        total_rows += n
    print(f"  宣稱 {declared_total} ／ 實際 {total_rows}")
    if declared_total is not None and declared_total != total_rows:
        print(f"  ❌ 不一致，差 {total_rows - declared_total} 幀 —— 這就是會在訓練中途爆的那個問題")
        ok = False
    else:
        print("  ✅ 一致")

    # ---- 檢查 2：影片幀數 vs parquet 行數 ----
    print("\n" + "─" * 60)
    print("檢查 2：每支影片實際幀數 vs 對應 parquet 行數")
    videos = sorted(root.glob("videos/**/*.mp4")) or sorted(root.glob("**/*.mp4"))
    if not videos:
        print("  ⚠️  找不到 mp4（可能未用影片編碼儲存）—— 跳過")
    else:
        # 以檔名中的 episode 編號配對
        def ep_key(p: Path) -> str:
            stem = p.stem
            return stem.split("_")[-1] if "_" in stem else stem

        pq_by_ep = {ep_key(p): (p, n) for p, n in rows_per_file.items()}
        for v in videos:
            key = ep_key(v)
            if key not in pq_by_ep:
                continue
            pq_path, pq_rows = pq_by_ep[key]
            n = count_video_frames(v)
            if n is None:
                ok = False
                continue
            cam = v.parent.name
            if n != pq_rows:
                print(f"  ❌ {cam}/{v.name}: 影片 {n} 幀 ≠ parquet {pq_rows} 行")
                ok = False
            else:
                print(f"  ✅ {cam}/{v.name}: {n}")

    # ---- 檢查 3：timestamp 間隔 ----
    print("\n" + "─" * 60)
    print("檢查 3：timestamp 間隔（掉幀的直接證據）")
    if fps <= 0:
        print("  ⚠️  info.json 沒有有效 fps —— 跳過")
    else:
        found = False
        for p in parquets:
            probs = check_timestamps(p, fps)
            if probs:
                found = True
                ok = False
                print(f"  ❌ {p.name}")
                for line in probs[:5]:
                    print(line)
                if len(probs) > 5:
                    print(f"    …共 {len(probs)} 處異常間隔")
        if not found:
            print("  ✅ 沒有異常間隔")

    print("\n" + "═" * 60)
    if ok:
        print("✅ 全部通過。這批資料可以進訓練。")
        return 0
    print("❌ 有問題。")
    print("   ⚠️  --dataset.exclude_episodes 救不了幀數不一致（sampler 索引空間仍由錯誤總數建立）")
    print("   → 唯一的解是重錄。現在重錄，比訓練到一半才發現便宜。")
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))

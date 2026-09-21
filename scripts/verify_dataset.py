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
2. 每支影片的實際幀數  ==  存在這支影片裡的那些 episode 的 parquet 行數加總
   （v3 的資料檔與影片檔各自分檔、檔名都叫 file-NNN，不能用檔名配對 → 從 meta/episodes 查每集在哪支影片。
    2026-09-13 合併資料集時，舊的檔名配對把 front-left file-001 整支跳過沒檢查，另一支誤報 ❌。）
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
from collections import Counter
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


def check_videos_by_episode_meta(root: Path, parquets: list[Path], videos: list[Path]) -> bool | None:
    """v3：用 meta/episodes 的 videos/<key>/chunk_index、file_index 找出每支影片裝了哪幾集，
    期望幀數 = 那幾集在 data parquet 的實際行數加總（以資料為準，不信 meta 的 length）。
    回傳 None 表示沒有 meta/episodes（舊結構），交給檔名配對。"""
    ep_files = sorted(root.glob("meta/episodes/**/*.parquet"))
    if not ep_files:
        return None
    import pyarrow as pa
    import pyarrow.parquet as pq

    eps = pa.concat_tables([pq.read_table(p) for p in ep_files]).to_pydict()
    video_keys = sorted(
        {c.split("/")[1] for c in eps if c.startswith("videos/") and c.endswith("/file_index")}
    )
    if not video_keys:
        return None

    rows_per_ep = Counter()
    for p in parquets:
        rows_per_ep.update(pq.read_table(p, columns=["episode_index"]).column("episode_index").to_pylist())

    expected: dict[Path, int] = {}
    n_eps: Counter = Counter()
    for key in video_keys:
        for ep, c, f in zip(
            eps["episode_index"], eps[f"videos/{key}/chunk_index"], eps[f"videos/{key}/file_index"]
        ):
            path = root / "videos" / key / f"chunk-{c:03d}" / f"file-{f:03d}.mp4"
            expected[path] = expected.get(path, 0) + rows_per_ep[ep]
            n_eps[path] += 1

    ok = True
    for path in sorted(expected):
        rel = path.relative_to(root / "videos").as_posix()
        if not path.exists():
            print(f"  ❌ {rel}: meta/episodes 指向這支影片，但檔案不存在")
            ok = False
            continue
        n = count_video_frames(path)
        if n is None:
            ok = False
            continue
        if n != expected[path]:
            print(f"  ❌ {rel}: 影片 {n} 幀 ≠ 其中 {n_eps[path]} 集的 parquet {expected[path]} 行")
            ok = False
        else:
            print(f"  ✅ {rel}: {n}（{n_eps[path]} 集）")
    for v in videos:
        if v not in expected:
            print(f"  ⚠️  {v.relative_to(root).as_posix()}: 沒有任何 episode 指向它（多餘檔案？）")
    return ok


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

    # ---- 檢查 2：影片幀數 vs parquet 行數（兩條路徑）----
    #
    # 🔴 2026-09-21 merge：兩邊各自修了同一個 v3 bug，兩條路徑都保留。
    #   主路徑 check_videos_by_episode_meta()：有 meta/episodes 就用它查「每支影片裝了哪幾集」，
    #     期望幀數 = 那幾集的 parquet 行數加總 → 精確到「影片檔」，不會被影片／parquet 分檔不對稱騙到。
    #   退路（本段以下）：沒有 meta/episodes 的舊結構，退回「每台相機加總 vs parquet 總行數」。
    #
    # 🔴 2026-09-18 修正（退路的由來）。原本「以檔名配對影片與 parquet、逐檔比對」只在 LeRobot v2.x（一集一個檔）
    # 下成立。v3.0 是按「大小」切檔（video_files_size_in_mb / data_files_size_in_mb），而且影片與
    # parquet 各自獨立切——同一台相機的影片可以切成 file-000 + file-001，parquet 卻只有一個 file-000。
    # 舊寫法因此有三個 bug，在 uvc_60 上全部中招：
    #   (1) 拿 front-left 的 file-000.mp4（6626 幀）去比「整份」parquet（15966 行）→ 假的 ❌
    #   (2) 配不到 parquet 的 file-001.mp4（9340 幀）被 `continue` 直接跳過，完全沒被數到
    #   (3) 相機名取成 v.parent.name（= chunk-000），輸出分不出是哪台相機
    # 結論變成「唯一的解是重錄」——對一份完好的資料（6626 + 9340 = 15966）。
    # 改成：每台相機把所有分檔加總，跟 parquet 總行數比。v2.x／v3.0 兩種結構都適用。
    #
    # ⚠️ 退路的已知弱點（誠實記下）：比的是「每台相機的總和」，不是逐 episode。若某集多 1 幀、另一集少 1 幀，
    # 總和會互相抵銷而漏抓。本腳本要防的主要失效（info.json 宣稱總數 ≠ 影片實際總數 → 訓練中途
    # IndexError，見 docstring）仍然抓得到。主路徑精確到「影片檔」，仍不是逐 episode——真正的逐
    # episode 比對要用 meta/episodes 的 from/to_timestamp 解碼計數，成本高得多，目前沒做。
    print("\n" + "─" * 60)
    print("檢查 2：影片幀數 vs parquet 行數")
    videos = sorted(root.glob("videos/**/*.mp4")) or sorted(root.glob("**/*.mp4"))
    v3_ok = check_videos_by_episode_meta(root, parquets, videos) if videos else None
    if not videos:
        print("  ⚠️  找不到 mp4（可能未用影片編碼儲存）—— 跳過")
    elif v3_ok is not None:
        ok = ok and v3_ok
    else:
        # 退路（沒有 meta/episodes 的舊結構）：每台相機所有分檔加總，跟 parquet 總行數比。
        videos_dir = root / "videos"
        by_cam: dict[str, list[Path]] = {}
        for v in videos:
            try:
                cam = v.relative_to(videos_dir).parts[0]  # videos/<camera>/chunk-NNN/file-NNN.mp4
            except ValueError:
                cam = v.parent.name
            by_cam.setdefault(cam, []).append(v)

        for cam, files in by_cam.items():
            counts = []
            for v in files:
                n = count_video_frames(v)
                if n is None:
                    ok = False
                    break
                counts.append((v, n))
            else:
                cam_total = sum(n for _, n in counts)
                detail = " + ".join(f"{v.parent.name}/{v.name}={n}" for v, n in counts)
                if cam_total != total_rows:
                    print(f"  ❌ {cam}: 影片共 {cam_total} 幀 ≠ parquet {total_rows} 行   ({detail})")
                    ok = False
                else:
                    print(f"  ✅ {cam}: {cam_total}   ({detail})")

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
        # This script checks STRUCTURAL integrity only. For a simulated dataset (S5), passing it
        # says nothing about whether the data may be trained on -- S5 §8 / D025 premise 1 forbid
        # that until the gate is decided, and the fidelity flags below are why. Exit code stays 0:
        # the structure really is fine, and S5 §7 requires exit 0 as its structural check.
        sim_meta = root / "meta" / "sim_provenance.json"
        if sim_meta.exists():
            prov = json.loads(sim_meta.read_text())
            print("✅ 結構檢查全部通過。")
            print("🔴 但這是【模擬】資料集（有 meta/sim_provenance.json）——結構正確 ≠ 可以進訓練。")
            print("   S5 §8／D025 前提 1：在閘門被裁決之前，不得拿去訓練、不得在書審／報告宣稱有效。")
            print("   fidelity flags：")
            for k, v in prov.get("fidelity", {}).items():
                if not k.endswith("_note"):
                    print(f"     {k} = {v}")
            return 0
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

#!/usr/bin/env python3
"""
把 lerobot-train 的 nohup log 畫成 loss 曲線。

為什麼從 log 解析而不是讀 wandb
--------------------------------
本機的 wandb run 目錄只有二進位的 `run-*.wandb`，離線讀它要裝 wandb SDK；
而 `data/train_*.log` 裡每 `log_freq` 步就有一行完整的 metric，已經夠用。

⚠️ log 裡的 `step:` 欄位會被縮寫（`step:1K`、`step:50K`），不能直接當 x 軸。
    改用「第 n 行 × log_freq」重建 step —— 前提是 log 沒有斷續（resume 的 log 不適用）。

用法
----
    python scripts/plot_train_loss.py data/train_alcan60_fixed.log \
        -o outputs/train_curves/alcan60_fixed_loss.png --log-freq 50
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 圖上的中文：沒有 CJK 字型時 matplotlib 會畫成方框，所以顯式掛一份再用。
for _p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    try:
        fm.fontManager.addfont(_p)
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"] + plt.rcParams["font.sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

# INFO ... ot_train.py:756 step:50 smpl:2K ... loss:13.699 ... l1_loss:0.721 kld_loss:1.298
FIELD = re.compile(r"\b(loss|l1_loss|kld_loss|grdn|epch):([0-9.eE+-]+)")


def parse(log_path: Path, log_freq: int):
    steps, series = [], {"loss": [], "l1_loss": [], "kld_loss": [], "grdn": [], "epch": []}
    n = 0
    for line in log_path.read_text(errors="ignore").splitlines():
        if "ot_train.py" not in line or " loss:" not in line:
            continue
        fields = dict(FIELD.findall(line))
        if "loss" not in fields:
            continue
        n += 1
        steps.append(n * log_freq)
        for k in series:
            series[k].append(float(fields[k]) if k in fields else float("nan"))
    return steps, series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--log-freq", type=int, default=50)
    ap.add_argument("--zoom-from", type=int, default=5000,
                    help="右圖從這一步開始畫（前面的下降太陡，會壓扁尾段）")
    args = ap.parse_args()

    steps, s = parse(args.log, args.log_freq)
    if not steps:
        raise SystemExit(f"{args.log} 裡沒有解析到任何 metric 行")
    print(f"{len(steps)} 個記錄點，step {steps[0]} → {steps[-1]}，"
          f"final loss={s['loss'][-1]:.4f} (l1={s['l1_loss'][-1]:.4f}, kld={s['kld_loss'][-1]:.4f})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(steps, s["loss"], lw=1.0, label="total loss")
    ax1.plot(steps, s["l1_loss"], lw=1.0, label="l1_loss")
    ax1.plot(steps, s["kld_loss"], lw=1.0, label="kld_loss")
    ax1.set_yscale("log")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss (log scale)")
    ax1.set_title(f"{args.log.stem} — 全程 0→{steps[-1]//1000}k")
    ax1.grid(alpha=0.3)
    ax1.legend()

    i0 = next((i for i, st in enumerate(steps) if st >= args.zoom_from), 0)
    ax2.plot(steps[i0:], s["loss"][i0:], lw=0.8, color="tab:blue", label="total loss")
    ax2.plot(steps[i0:], s["l1_loss"][i0:], lw=0.8, color="tab:orange", label="l1_loss")
    ax2.set_xlabel("step")
    ax2.set_ylabel("loss")
    ax2.set_title(f"放大：step ≥ {args.zoom_from}（線性軸）")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"已寫出 {args.out}")


if __name__ == "__main__":
    main()

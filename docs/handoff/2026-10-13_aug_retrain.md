# 交代：4090 重新訓練 ntfA1／ntcfA1，加隨機平移增強（2026-10-13）

寫給：在 4090 訓練機上跑訓練的人。決定者：柏宇（2026-10-13）。

## 要做什麼

用兩份新 config 各訓練一次，跟 10-09 那兩個模型比，**只加了 front-left 隨機平移增強**：

| 版本 | 新 config | 對照的舊模型（Hub） | 舊 hash |
|---|---|---|---|
| ntfA1（轉腕） | `configs/train_omx_cup_uvc60_normal_tight_far_A1_aug.yaml` | `act_omx_b1_paper-cup_60-13-10rcvry-46far_100k` | `e97526c13766` |
| ntcfA1（轉腕＋recovery 剪前段） | `configs/train_omx_cup_uvc60_normal_tightcut_far_A1_aug.yaml` | `act_omx_b1_paper-cup_60-13-10rcvrycut-46far_100k` | `0319057c1d65` |

資料、`exclude_episodes`、超參數、seed（1000）全部跟舊 config 一樣，合併資料集也沿用 10-09 在 4090 上做好的那兩份（`root:` 沒變）。
差別只有 `dataset.image_transforms`、`output_dir`、`job_name`（多了 `_aug`），以及 `num_workers` 4 → 12（見下）。

## 步驟

```bash
git pull                                    # 需要 2026-10-13 的 commit（兩份 _aug config）
ls data/huggingface/lerobot/ericc430/omx_pick_place_pilot_paper_cup_uvc60_normalA1_tightA1_farwristA1 \
   data/huggingface/lerobot/ericc430/omx_pick_place_pilot_paper_cup_uvc60_normalA1_tightA1cut_farwristA1   # 10-09 合併的，應該都在
./scripts/run_container.sh lerobot-train --config_path configs/train_omx_cup_uvc60_normal_tight_far_A1_aug.yaml
./scripts/run_container.sh lerobot-train --config_path configs/train_omx_cup_uvc60_normal_tightcut_far_A1_aug.yaml
```

開跑後確認：`data/train/<run>_aug/checkpoints/<step>/pretrained_model/train_config.json` 裡
`dataset.image_transforms.enable` 是 `true`、`tfs` 只有 `affine` 的 weight 是 1.0。
（舊模型 Hub 上的 `train_config.json` 是 `enable: false`，**原本沒有任何增強**。）

增強在 dataloader（CPU）做，每步可能比 10-09 慢；請記下實際花了多久。

2026-10-13 實測：`num_workers: 4` 時每步 0.55 s（其中等資料 0.25 s，GPU 更新 0.30 s），100k 約 15 h；
改成 12 後每步 0.31 s（等資料 0.007 s），跟 10-09 沒增強的 run 一樣（約 8.7 h）`[已查證]`。
`num_workers` 不改資料、超參數或 seed，但每個 worker 抽到的增強亂數不同，所以這次的結果無法跟 4 worker 逐位元重現。
4 worker 那次跑到 step 1662 就停了（沒有 checkpoint），留在 `data/train/aborted_cup_uvc60_normal_tight_far_A1_aug_w4`。

## 推上 Hub（private）

依 `docs/models.md`「新增模型時」：名稱標步數。建議：

- `ericc430/act_omx_b1_paper-cup_60-13-10rcvry-46far-aug_100k`
- `ericc430/act_omx_b1_paper-cup_60-13-10rcvrycut-46far-aug_100k`

20k／40k 有空也推（`…-aug_20k`／`…-aug_40k`），閉環可能要比 step。

## 回報

1. 每個推上去的 checkpoint 的 `sha256sum model.safetensors | cut -c1-12`（要加進 `docs/models.md`）
2. 實際訓練時間、wandb run 連結
3. 有沒有任何錯誤或警告

## 為什麼要加增強 `[產出物]`

ntfA1 在閉環時，近處的紙杯（c1，r 24.3）也用了只在遠處示範（r ≥ 35.5）才有的轉腕抓法。

- 訓練資料：far_wrist A1 的 46 集**全部轉腕、全部在 r ≥ 35.5**；其他 83 集沒有一集轉腕。
- 那一場（10-08／09）front-left 的位置比 10-07 偏約 20 px（`align_camera.py --dry-run`）。也就是說，**在那個視角下只看過轉腕**。
- 反事實測試 `scripts/input_reliance/h6_turn_cue.py`：同一幀，只把 front-left 整張平移 20 px（杯子和手臂一起動、相對距離不變），轉腕的決定就會變；手腕畫面換成灰色則幾乎不變。

| 模型 | normal_A1 13 集：原始 → 平移成 far 視角 | far_wrist 12 集：原始 → 平移成 10-07 視角 |
|---|---|---|
| ntfA1 100k | 0/13 → **3/13** 轉腕（中位 +9.3 → +13.3） | 10/12 → 10/12 |
| ntcfA1 100k | 0/13 → **3/13** 轉腕（中位 +8.9 → +12.3） | 10/12 → 10/12 |

10-13 閉環 ep0（c1）在轉腕前的幀：原始 +19.8～+25.8，平移成 10-07 視角 +9.3～+10.1（不轉的範圍），手腕灰色 +23.2～+27.1。

結論：**模型把「相機擺在哪」也當成要不要轉腕的線索**，這是我們不要的；我們要的是 front-left 畫面中杯子與手臂的相對距離。
隨機平移讓相機的絕對位置每次都不一樣，但杯子和手臂一起移動、相對關係保留 `[推論]`。

## 增強參數與已知限制

- `RandomAffine`：`translate (0.05, 0.05)` = 848×480 上 ±42 px／±24 px（觀察到的 20 px 的 2 倍）；`degrees ±2`（session 之間看過約 1.2° 的 roll）。
  本機用 lerobot 的 config parser 讀過兩份 config，確認只啟用 affine、平移範圍 ±42 px `[已查證]`。
- **lerobot 對每台相機各自套用**（`datasets/dataset_reader.py`），手腕畫面也會被平移。只套 front-left 要改 lerobot 原始碼，這次不做。
  10-07 量過手腕對抓取位置的影響是 front-left 的 1/3～1/5，影響應該有限 `[推論]`。
- 平移後的邊緣補黑。
- **訓練比例沒有改**。增強只改畫面、不改每種示範的集數；同時改比例就分不出效果來自哪一個。
  far_wrist 仍佔 46/129 集、41% 幀。如果增強不夠，下一步是在 far_wrist 相機位置補錄不轉腕的近／中距離示範。

## 訓練完之後（Windows 端）

1. 下載新模型，跑 `uv run python scripts/input_reliance/h6_turn_cue.py --policy <新 repo>`。
   通過標準：normal_A1 平移後 0/13 轉腕、中位數跟原始差 2 以內；far_wrist 平移後轉腕集數不變。
2. 通過才上閉環，c1–c36 依距離分組看轉腕比例：r < 35.5 應接近 0，r ≥ 35.5 應該高。

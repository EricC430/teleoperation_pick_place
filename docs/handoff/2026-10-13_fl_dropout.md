# 交代：4090 訓練 ntcfA1aug ＋接近段 front-left dropout（2026-10-13）

寫給：在 4090 訓練機上跑訓練的人。決定者：柏宇（2026-10-13）。

## 要做什麼

以 ntcfA1aug（`act_omx_b1_paper-cup_60-13-10rcvrycut-46far-aug_100k`，`286a49c37d4e`）為基底，**只加一件事**：
每集**第一次閉合夾爪前 30 幀**，front-left 畫面以 **p = 0.3** 的機率換成全黑。
資料、`exclude_episodes`、`image_transforms`、超參數、seed（1000）全部不動。

| | |
|---|---|
| config | `configs/train_omx_cup_uvc60_normal_tightcut_far_A1_aug_fldrop.yaml` |
| 入口 | `scripts/train_fl_dropout.py`（包住 `lerobot-train`，不改 lerobot 原始碼） |
| output_dir／job_name | `…_aug_fldrop_p30_w30` |

## 為什麼

- 10-13 兩場 aug 閉環都是 25/36，失敗裡有 `bad_aim` 的佔 10/11、9/11；`chunk_gap_rush` 最多只能救回約 2 集（`docs/meeting/2026-10-13_closed_loop.md`）。
- 「抓哪裡」主要看 front-left，手腕只帶來需要修正量的一小部分（10-07 §3-3；ntcfA1aug 的數字見下面的「驗收」）。
- 想要的分工 `[柏宇說]`：大方向的定位由 front-left 負責，細部定位由手腕負責。所以只在接近段遮 front-left，逼模型在那一段用手腕。
- 整集隨機 dropout 不行：在 HOME 時，手腕畫面裡只有收納盒，沒有杯子（5 集都是 `[已查證]`）。
  那些樣本遮掉 front-left 後，模型完全沒有杯子位置的資訊（`n_obs_steps=1`，沒有記憶），只能學到平均動作。

## 參數怎麼來的

- **窗口 30 幀（2 s）**：看了 8 集（uvc_60 3、normal_A1 2、tight 1、far_wrist 2）閉合前的手腕畫面，
  杯子在手腕畫面裡的集數：閉合前 30 幀 8/8、45 幀 7/8、60 幀 5/8、90 幀只有 far_wrist 2 集 `[已查證]`。
  取 30 幀，保證遮掉 front-left 時手腕畫面看得到杯子。
- **閉合的判定**：action 的 gripper < 54（張開約 59–60，閉合約 48）。各來源第一次閉合的幀號中位數：
  uvc_60 118、normal_A1 138、tight_A1 138、far_wrist 212 `[已查證]`。沒有閉合的集（都是已排除的無效集）不做 dropout。
- **p = 0.3** `[柏宇決定]`。窗口大約佔全部幀的 10%（129 集 × 30 幀 ÷ 約 38k 幀），所以**實際被遮的樣本只有約 3%** `[推論]`。
  實際數字會寫在 `fl_dropout.json` 的 `expected_masked_fraction`。
- **填黑色**（uint8 的 0）：之後如果要用手實際遮住 front-left 做閉環，輸入會比較接近。

## 本機已驗證 `[已查證]`（Windows，normal_A1 16 集）

- 窗口內 200 幀被遮 71 幀（p = 0.3 時期望 60，在抽樣誤差內）；窗口外 200 幀被遮 0 幀；手腕畫面從未被遮。
- dataset pickle 之後 class 和參數都還在；用 spawn worker 的 DataLoader（lerobot 的預設）時，窗口內被遮 64/240 = 0.27。
- **沒驗證**：完整的 `lerobot-train`。Windows 的 venv 沒有裝 `accelerate`，訓練本來就只在 4090 的 docker 裡跑。所以下面第 2 步要先跑 200 步。

## 步驟

```bash
git pull
ls data/huggingface/lerobot/ericc430/omx_pick_place_pilot_paper_cup_uvc60_normalA1_tightA1cut_farwristA1   # 10-09 合併的，應該在
# 1. 先跑 200 步確認能跑（輸出到 /tmp，不留）
./scripts/run_container.sh python scripts/train_fl_dropout.py --fl-p 0.3 --fl-window 30 \
    --config_path configs/train_omx_cup_uvc60_normal_tightcut_far_A1_aug_fldrop.yaml \
    --steps 200 --save_checkpoint false --wandb.enable false --output_dir /tmp/fldrop_smoke
# 2. 正式訓練
./scripts/run_container.sh python scripts/train_fl_dropout.py --fl-p 0.3 --fl-window 30 \
    --config_path configs/train_omx_cup_uvc60_normal_tightcut_far_A1_aug_fldrop.yaml
```

開跑後確認：
- log 裡有一行 `front-left approach dropout: {...}`。
- `data/train/cup_uvc60_normal_tightcut_far_A1_aug_fldrop_p30_w30/fl_dropout.json`：
  `episodes` 應該是 129，`episodes_without_close` 只有已排除的集或是空的，`expected_masked_fraction` 約 0.03。
- **`--resume` 時也要再帶 `--fl-p 0.3 --fl-window 30`**：`train_config.json` 不知道這兩個參數。

## 推上 Hub（private）

依 `docs/models.md`「新增模型時」：名稱標步數。建議：
`ericc430/act_omx_b1_paper-cup_60-13-10rcvrycut-46far-aug-fldrop30_{20k,40k,100k}`。
上傳時把 `fl_dropout.json` 一起放進 repo；Hub 上的 `train_config.json` 看不出有做 dropout。

## 回報

1. 每個推上去的 checkpoint 的 `sha256sum model.safetensors | cut -c1-12`
2. 實際訓練時間、wandb run 連結、`fl_dropout.json` 的 `expected_masked_fraction`
3. 有沒有任何錯誤或警告

## 驗收（訓練前先定好）

**離線：手腕修正比例**（`scripts/input_reliance/h1_rollout_normalized.py`；用 10-06 閉環的 15 個 `bad_aim` 幀，
把手腕畫面換成對準的畫面，看預測的抓取位置被拉回杯子多少。0 = 手腕不影響，1 = 手腕把預測完全拉到杯子上）。

```bash
H1_CKPT=<新模型的 snapshot 資料夾> uv run python scripts/input_reliance/h1_rollout_normalized.py
```

基準值（Windows，2026-10-13）`[已查證]`：

| 模型 | 抓取前 1/3 s：手腕修正比例 | 1 s：手腕修正比例 | 1/3 s：front-left 跟隨比例 | 1 s：front-left 跟隨比例 |
|---|---|---|---|---|
| uvc60 100k（10-07 的模型；與 10-07 紀錄完全相同） | +0.10（IQR −0.08～+0.31） | +0.15（+0.03～+0.44） | +0.48 | +0.67 |
| **ntcfA1aug 100k（基底）** | **+0.05**（−0.18～+0.33） | **+0.21**（−0.14～+0.78） | +0.52 | +0.58 |

- 通過標準（**暫定**，`[AI提議]`，柏宇 2026-10-13 同意暫定）：兩個時間點的手腕修正比例中位數都 ≥ 0.3，而且同樣 15 幀逐幀比較時，至少 11 幀比 ntcfA1aug 高。
  front-left 跟隨比例下降不算失敗，那正是想要的分工。
- 沒通過時，**先把 p 調高（例如 0.5）再訓練一次**，不要直接下「dropout 沒用」的結論：實際被遮的樣本只有約 3%。

### 離線結果（Windows，2026-10-14）`[已查證]`

fldrop30 100k（`7701bdd277ad`）和基底 ntcfA1aug 100k 同一次、同一份程式跑（基底四個數字與上表完全相同）。
逐幀數字是在 `h1_rollout_normalized.py` 的迴圈裡多印一行得到的（程式本身沒改）。

| 模型 | 1/3 s：手腕修正比例 | 1 s：手腕修正比例 | 1/3 s：front-left 跟隨比例 | 1 s：front-left 跟隨比例 |
|---|---|---|---|---|
| ntcfA1aug 100k（基底） | +0.05（−0.18～+0.33） | +0.21（−0.14～+0.78） | +0.52 | +0.58 |
| **fldrop30 100k** | **+0.47**（+0.02～+0.61） | **+0.74**（−0.06～+1.24） | +0.20 | +0.33 |

- 中位數 ≥ 0.3：**兩個時間點都達到**。
- 逐幀比基底高：**1/3 s 10/15、1 s 6/15**，沒達到「至少 11 幀」。
- 所以照暫定標準是**沒通過**。
- 數字裡 > 1.2（修正過頭）的幀數：1/3 s fldrop 1、基底 0；1 s fldrop 4、基底 3。「比基底高」不一定代表比較好。
- 20k／40k 沒跑。

**閉環**（離線通過才做）：條件同 10-13 的 N=30（far_wrist 視角、goto-home ep10、c1–c36、30 s），
看 `bad_aim` 的集數和成功率，比較對象是 ntcfA1aug N=30 那場（`__98627c1e`：25/36，失敗中 `bad_aim` 9/11）。

## 已知限制

- 只看第一次閉合；自己重抓（第二次閉合）前的接近段不遮。
- 15 個探測幀來自舊模型（uvc60 100k、A1 相機位置）10-06 的閉環，不是新模型自己會產生的狀態；而且 n = 15、IQR 很寬。
- 映射本身有約 2.6 cm 的橫向雜訊，和需要修正的 4–5 cm 同一量級（10-07 §3-3）。
- 窗口長度是從 8 集看出來的，不是逐集判斷杯子有沒有進入手腕畫面。

# 2026-10-14 工作紀錄：ntcfA1aug ＋接近段 front-left dropout（ntcfA1augfd）的訓練、h6 與開環

參與：柏宇、Claude（AI） ｜ 性質：分析工作紀錄，非會議

為什麼做、參數、驗收標準：`docs/handoff/2026-10-13_fl_dropout.md`（本檔不重寫）。
hash ↔ run／step／Hub 見 `docs/models.md`（本次新增 3 列）。簡稱沿用 `outputs/open_loop_paper_cup/` 的目錄前綴：

| 簡稱 | 本機 run | 20k / 40k / 100k |
|---|---|---|
| ntcfA1aug（基底） | `cup_uvc60_normal_tightcut_far_A1_aug` | `e0894f917c32` / `f877d679843f` / `286a49c37d4e` |
| ntcfA1augfd | `cup_uvc60_normal_tightcut_far_A1_aug_fldrop_p30_w30` | `74029f33e571` / `4d954bc6d1c7` / `7701bdd277ad` |

## 1. 做了什麼 `[已查證]`

- 訓練：10-14 00:08 → 08:57（8 h 48 m），`exit=0`，最終 l1 loss 0.060（基底 0.058），log `data/logs/train_ntcfA1aug_fldrop.log`，wandb `2ksyviz0`。
  `fl_dropout.json`：p 0.3、窗口 30 幀、129 集、沒有找不到閉合的集、`expected_masked_fraction` 0.0282。
  `train_config.json`：`image_transforms` 只開 affine（front-left 隨機平移仍在，抽樣驗證見 handoff「4090 補驗」）。警告只有 Device None 與 kernel 5.4 兩條。
- Hub：`ericc430/act_omx_b1_paper-cup_60-13-10rcvrycut-46far-aug-fldrop30_{20k,40k,100k}`，private，LFS hash 與本機一致，各 repo 都附 `fl_dropout.json`。
- h6：`data/logs/h6_ntcfA1augfd.log`（基底 100k 一起重跑當對照）。
- 開環：同 10-08／10-10／10-13（eval 集全 12 集，N = 100／50／20），輸出 `ntcfA1augfd_{step}_nas{N}/`，9 組皆 `exit=0`；
  成對比較 `cmp_ntcfA1aug_vs_ntcfA1augfd_*`。
- **離線驗收（`h1_rollout_normalized.py`）還沒跑**：要在 Windows 上跑（用 10-06 閉環的 15 個 `bad_aim` 幀），這是 dropout 有沒有效的主要判準。

## 2. h6 `[已查證]`

轉腕 = 預測 wrist_roll 最大值 > 15；定義同 `2026-10-13_open_loop.md` §2。

| 模型 | normal_A1 13 集：原始 → FL 平移 → 手腕灰 | far_wrist 12 集：原始 → FL 平移 → 手腕灰（中位數） |
|---|---|---|
| ntcfA1aug 100k | 0 → 0 → 0（+8.1 → +8.1 → +8.0） | 10 → 10 → 10（+41.7 → +41.3 → +41.3） |
| ntcfA1augfd 100k | 0 → 0 → 0（+8.4 → +8.5 → +8.3） | 10 → 10 → 10（+42.7 → +42.5 → **+39.5**） |
| ntcfA1augfd 40k | 0 → 0 → 0（+8.6 → +8.8 → +9.3） | 10 → **11** → 10（+44.3 → +43.1 → **+39.9**） |
| ntcfA1augfd 20k | 0 → 0 → 0（+10.2 → +9.4 → +9.4） | 10 → 10 → 10（+44.1 → +43.4 → **+38.7**） |

- 轉腕決定沒有被 dropout 破壞：近處平移後仍 0/13。遠處 40k 平移後 11/12（多 1 集），其他不變。
- 手腕換灰時，遠處的 wrist_roll 中位數下降 3–5（基底只降 0.4）。dropout 後模型對手腕畫面稍微敏感一點的跡象，
  但轉腕集數不變、n = 12、手腕灰色是訓練沒看過的輸入 `[推論]`。這不是 handoff 的驗收指標。

## 3. 開環 `[已查證]`

| step | 模型 | N=100 | N=50 | N=20 |
|---|---|---|---|---|
| 20k | ntcfA1aug | 6.43 | 4.88 | 3.76 |
| | ntcfA1augfd | 6.53 | 5.52 | 3.83 |
| 40k | ntcfA1aug | 6.29 | 4.58 | 3.67 |
| | ntcfA1augfd | 6.17 | 4.79 | 3.83 |
| 100k | ntcfA1aug | 5.77 | 4.82 | 3.47 |
| | ntcfA1augfd | 5.59 | 4.57 | 3.72 |

逐集配對 ntcfA1augfd − ntcfA1aug（95% bootstrap CI；sign-flip p；括號內是 ntcfA1augfd 贏的集數）。**粗體** = CI 不跨 0。

| step | N=100 | N=50 | N=20 |
|---|---|---|---|
| 20k | +0.10 [−0.46, +0.61] p=0.72 (4/12) | **+0.65** [+0.25, +1.01] p=0.013 (2/12) | +0.07 [−0.06, +0.23] p=0.44 (6/12) |
| 40k | −0.12 [−0.64, +0.36] p=0.68 (6/12) | +0.21 [−0.08, +0.54] p=0.24 (4/12) | +0.16 [−0.01, +0.32] p=0.094 (4/12) |
| 100k | −0.18 [−0.77, +0.34] p=0.56 (5/12) | −0.25 [−0.69, +0.16] p=0.29 (8/12) | +0.25 [+0.00, +0.51] p=0.090 (2/12) |

讀法：

- **大致分不出差別**：9 格中 1 格 CI 不跨 0（20k N=50，dropout 較差），100k N=20 的 CI 下界剛好碰到 0。
  9 個檢定，α = 0.05 下預期約 0.5 個偽陽性。
- 被遮的樣本只有約 3%，開環誤差不太變是預期中的 `[推論]`。開環量的是「看得到 front-left 時」的動作，量不到 dropout 想改的
  「接近段有多依賴手腕」——那要看離線驗收（h1）和閉環。

## 4. 未完成

- [ ] 離線驗收（Windows）：`H1_CKPT=<…-fldrop30_100k 的 snapshot> uv run python scripts/input_reliance/h1_rollout_normalized.py`，
      通過標準見 handoff「驗收」（兩個時間點手腕修正比例中位數 ≥ 0.3，且 15 幀中 ≥ 11 幀高於 ntcfA1aug）
- [ ] 沒通過 → 依 handoff 先把 p 調高（例如 0.5）重訓；通過 → 閉環（條件同 10-13 N=30，對照 `__98627c1e`）
- [ ] 遮第一／第三視角看軌跡變化（柏宇 10-14 提出）：還在設計，量什麼、用哪些幀、遮成什麼要先定 `[柏宇說]`

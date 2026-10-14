# 2026-10-14 工作紀錄：ntcfA1aug ＋接近段 front-left dropout（fldrop30）的訓練、h6 與開環

參與：柏宇、Claude（AI） ｜ 性質：分析工作紀錄，非會議

為什麼做、參數、驗收標準：`docs/handoff/2026-10-13_fl_dropout.md`（本檔不重寫）。
hash ↔ run／step／Hub 見 `docs/models.md`（本次新增 3 列）。簡稱：基底沿用 `outputs/open_loop_paper_cup/` 的目錄前綴；dropout 版統一叫 **fldrop30**（柏宇 2026-10-14 定），但 4090 上已經產生的檔名、目錄名仍是 `ntcfA1augfd`（下面路徑照實寫，不改）：

| 簡稱 | 本機 run | 20k / 40k / 100k |
|---|---|---|
| ntcfA1aug（基底） | `cup_uvc60_normal_tightcut_far_A1_aug` | `e0894f917c32` / `f877d679843f` / `286a49c37d4e` |
| fldrop30 | `cup_uvc60_normal_tightcut_far_A1_aug_fldrop_p30_w30` | `74029f33e571` / `4d954bc6d1c7` / `7701bdd277ad` |

## 1. 做了什麼 `[已查證]`

- 訓練：10-14 00:08 → 08:57（8 h 48 m），`exit=0`，最終 l1 loss 0.060（基底 0.058），log `data/logs/train_ntcfA1aug_fldrop.log`，wandb `2ksyviz0`。
  `fl_dropout.json`：p 0.3、窗口 30 幀、129 集、沒有找不到閉合的集、`expected_masked_fraction` 0.0282。
  `train_config.json`：`image_transforms` 只開 affine（front-left 隨機平移仍在，抽樣驗證見 handoff「4090 補驗」）。警告只有 Device None 與 kernel 5.4 兩條。
- Hub：`ericc430/act_omx_b1_paper-cup_60-13-10rcvrycut-46far-aug-fldrop30_{20k,40k,100k}`，private，LFS hash 與本機一致，各 repo 都附 `fl_dropout.json`。
- h6：`data/logs/h6_ntcfA1augfd.log`（基底 100k 一起重跑當對照）。
- 開環：同 10-08／10-10／10-13（eval 集全 12 集，N = 100／50／20），輸出 `ntcfA1augfd_{step}_nas{N}/`，9 組皆 `exit=0`；
  成對比較 `cmp_ntcfA1aug_vs_ntcfA1augfd_*`。
- 離線驗收（`h1_rollout_normalized.py`）與閉環：之後在 Windows 跑完，結果寫在 handoff「離線結果」「閉環結果」（見 §4）。

## 2. h6 `[已查證]`

轉腕 = 預測 wrist_roll 最大值 > 15；定義同 `2026-10-13_open_loop.md` §2。

| 模型 | normal_A1 13 集：原始 → FL 平移 → 手腕灰 | far_wrist 12 集：原始 → FL 平移 → 手腕灰（中位數） |
|---|---|---|
| ntcfA1aug 100k | 0 → 0 → 0（+8.1 → +8.1 → +8.0） | 10 → 10 → 10（+41.7 → +41.3 → +41.3） |
| fldrop30 100k | 0 → 0 → 0（+8.4 → +8.5 → +8.3） | 10 → 10 → 10（+42.7 → +42.5 → **+39.5**） |
| fldrop30 40k | 0 → 0 → 0（+8.6 → +8.8 → +9.3） | 10 → **11** → 10（+44.3 → +43.1 → **+39.9**） |
| fldrop30 20k | 0 → 0 → 0（+10.2 → +9.4 → +9.4） | 10 → 10 → 10（+44.1 → +43.4 → **+38.7**） |

- 轉腕決定沒有被 dropout 破壞：近處平移後仍 0/13。遠處 40k 平移後 11/12（多 1 集），其他不變。
- 手腕換灰時，遠處的 wrist_roll 中位數下降 3–5（基底只降 0.4）。dropout 後模型對手腕畫面稍微敏感一點的跡象，
  但轉腕集數不變、n = 12、手腕灰色是訓練沒看過的輸入 `[推論]`。這不是 handoff 的驗收指標。

## 3. 開環 `[已查證]`

| step | 模型 | N=100 | N=50 | N=20 |
|---|---|---|---|---|
| 20k | ntcfA1aug | 6.43 | 4.88 | 3.76 |
| | fldrop30 | 6.53 | 5.52 | 3.83 |
| 40k | ntcfA1aug | 6.29 | 4.58 | 3.67 |
| | fldrop30 | 6.17 | 4.79 | 3.83 |
| 100k | ntcfA1aug | 5.77 | 4.82 | 3.47 |
| | fldrop30 | 5.59 | 4.57 | 3.72 |

逐集配對 fldrop30 − ntcfA1aug（95% bootstrap CI；sign-flip p；括號內是 fldrop30 贏的集數）。**粗體** = CI 不跨 0。

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

- [x] 離線驗收（Windows，2026-10-14）：中位數兩個時間點都過，逐幀 10/15、6/15 沒過 → 依標準未通過。
      數字見 `docs/handoff/2026-10-13_fl_dropout.md`「離線結果」。
- [x] 閉環（柏宇決定不等離線，直接看實機）：fldrop30 100k 30/36 vs ntcfA1aug 25/36，配對 p 0.18（record `__f92698e7`；
      handoff「閉環結果」）。
- [ ] p 調高到 0.5 重訓：`docs/handoff/2026-10-14_fl_dropout_p50.md`。「p50 比 p30 好」的判準還沒定。
      4090 已開跑（10-14 17:00，`data/logs/train_fldrop50.log`，wandb `hx0gqm62`；`expected_masked_fraction` 0.047）。
- [ ] 平移增強只套 front-left、手腕不套，配 p 0.3 重訓（柏宇 10-14 決定，等 p50 訓練完 GPU 空出來）：
      - 理由：增強是為了 front-left 跨場次的架設位置差（`2026-10-13_open_loop.md`）；手腕相機鎖在夾爪上、沒有這個問題，
        而指尖只在接近閉合時才入鏡，窗口前段手腕畫面裡的杯子位置被 ±32／±24 px 平移干擾 `[推論]`。
        fldrop30 的 7 集 `bad_aim`（失敗 4、成功 3）偏移沒看出固定方向（伸不夠 4、伸過頭 1、左右 3）`[柏宇說]`，與「精度被雜訊拉低」相容但不算支持。
      - 程式（`train_fl_dropout.py --transforms-fl-only`）已寫好並抽樣驗證；指令、驗證結果、Hub 名稱、比較方式：`docs/handoff/2026-10-14_fl_augfl.md`。
      - 前提 `[未確認]`：手腕相機的支架在各場次之間沒有動過。
- [ ] 遮第一／第三視角看軌跡變化（柏宇 10-14 提出）：還在設計，量什麼、用哪些幀、遮成什麼要先定 `[柏宇說]`

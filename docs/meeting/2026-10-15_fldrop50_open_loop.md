# 2026-10-15 工作紀錄：fldrop50（front-left dropout p 0.5）的訓練、Hub 與開環；fldrop30-augfl 訓練中

參與：柏宇、Claude（AI） ｜ 性質：分析工作紀錄，非會議

為什麼做、參數：fldrop50 → `docs/handoff/2026-10-14_fl_dropout_p50.md`；augfl → `docs/handoff/2026-10-14_fl_augfl.md`（本檔不重寫）。
hash ↔ run／step／Hub 見 `docs/models.md`（本次新增 3 列）。簡稱：

| 簡稱 | 本機 run | 20k / 40k / 100k |
|---|---|---|
| ntcfA1aug（基底） | `cup_uvc60_normal_tightcut_far_A1_aug` | `e0894f917c32` / `f877d679843f` / `286a49c37d4e` |
| fldrop30 | `cup_uvc60_normal_tightcut_far_A1_aug_fldrop_p30_w30`（4090 上的開環目錄叫 `ntcfA1augfd_*`） | `74029f33e571` / `4d954bc6d1c7` / `7701bdd277ad` |
| fldrop50 | `cup_uvc60_normal_tightcut_far_A1_aug_fldrop_p50_w30` | `df7a951a5752` / `cac19f085404` / `d826a9465f29` |

## 1. fldrop50 訓練 `[已查證]`

- 10-14 17:00 → 10-15 01:48（8 h 48 m，與 p30 相同），`exit=0`，最終 loss 0.060（l1 0.059），log `data/logs/train_fldrop50.log`，wandb `hx0gqm62`。
  啟動腳本 `data/logs/run_fldrop50.sh`，指令照 handoff，200 步冒煙測試照 handoff 沒跑。
- `fl_dropout.json`：p 0.5、窗口 30 幀、129 集、`episodes_without_close` 空、`frames_in_window` 3613／38429、`expected_masked_fraction` **0.047**（p30 0.0282）。
- 警告只有 Device None 與 kernel 5.4 兩條，跟之前的 run 相同。
- Hub：`ericc430/act_omx_b1_paper-cup_60-13-10rcvrycut-46far-aug-fldrop50_{20k,40k,100k}`，private，LFS hash 與本機一致，各 repo 都附 `fl_dropout.json`（`data/logs/hf_upload_fldrop50.sh`）。
- 60k／80k 的 hash 記在 `data/logs/chain_fldrop50_augfl.log`，沒上 Hub。

## 2. fldrop50 開環 `[已查證]`

同 10-08／10-10／10-13／10-14（eval 集全 12 集，N = 100／50／20），輸出 `outputs/open_loop_paper_cup/fldrop50_{step}_nas{N}/`，9 組皆 `exit=0`
（`data/logs/openloop_fldrop50.sh`；`metrics.json` 的 `weights_sha256_12` 與上表一致）。成對比較 `cmp_ntcfA1augfd_vs_fldrop50_*`、`cmp_ntcfA1aug_vs_fldrop50_*`。
比較腳本先拿 10-14 的 fldrop30 100k N=20 重算過，得到 +0.25 [+0.00, +0.51] p=0.090，與 10-14 的紀錄相同。

| step | 模型 | N=100 | N=50 | N=20 |
|---|---|---|---|---|
| 20k | ntcfA1aug | 6.43 | 4.88 | 3.76 |
| | fldrop30 | 6.53 | 5.52 | 3.83 |
| | fldrop50 | 6.18 | 5.08 | 3.73 |
| 40k | ntcfA1aug | 6.29 | 4.58 | 3.67 |
| | fldrop30 | 6.17 | 4.79 | 3.83 |
| | fldrop50 | 6.00 | 5.00 | 3.57 |
| 100k | ntcfA1aug | 5.77 | 4.82 | 3.47 |
| | fldrop30 | 5.59 | 4.57 | 3.72 |
| | fldrop50 | 5.72 | 4.62 | 3.74 |

逐集配對 fldrop50 − 對照（95% bootstrap CI；sign-flip p；括號內是 fldrop50 贏的集數）。**粗體** = CI 不跨 0。

| step | N | fldrop50 − fldrop30 | fldrop50 − ntcfA1aug |
|---|---|---|---|
| 20k | 100 | −0.35 [−0.79, +0.11] p=0.16 (8/12) | −0.25 [−1.12, +0.55] p=0.59 (7/12) |
| 20k | 50 | **−0.44** [−0.76, −0.14] p=0.020 (10/12) | +0.20 [−0.11, +0.49] p=0.23 (4/12) |
| 20k | 20 | **−0.09** [−0.17, −0.02] p=0.042 (8/12) | −0.02 [−0.19, +0.14] p=0.81 (6/12) |
| 40k | 100 | −0.17 [−0.51, +0.14] p=0.37 (5/12) | −0.28 [−0.96, +0.37] p=0.44 (7/12) |
| 40k | 50 | +0.20 [−0.10, +0.49] p=0.24 (4/12) | +0.41 [−0.06, +0.86] p=0.12 (3/12) |
| 40k | 20 | **−0.26** [−0.42, −0.10] p=0.012 (9/12) | −0.10 [−0.35, +0.14] p=0.47 (6/12) |
| 100k | 100 | +0.12 [−0.28, +0.55] p=0.60 (5/12) | −0.05 [−0.83, +0.69] p=0.90 (7/12) |
| 100k | 50 | +0.05 [−0.21, +0.33] p=0.72 (6/12) | −0.20 [−0.52, +0.16] p=0.30 (9/12) |
| 100k | 20 | +0.02 [−0.07, +0.11] p=0.68 (5/12) | **+0.27** [+0.06, +0.50] p=0.038 (2/12) |

讀法：

- **100k（閉環要用的）：跟 fldrop30 三個 N 都分不出差別。** 跟基底比，只有 N=20 較差（+0.27，2/12）。fldrop30 也是同一格、同樣大小（+0.25 [+0.00, +0.51]）。
- 20k／40k：3 格 fldrop50 比 fldrop30 好。100k 沒有這個現象。
- 18 個檢定，α = 0.05 下預期約 0.9 個偽陽性；單一格不要單獨當結論。
- 跟 10-14 同樣的限制：開環量的是「看得到 front-left 時」的動作，量不到 dropout 想改變的「接近段多依賴手腕」。p 加大後開環仍然分不出差別，是預期中的 `[推論]`。

### 2-1. fldrop50 離線 h1（Windows）`[已查證]`

`h1_rollout_normalized.py` 不改，外面包一層多印逐幀數字（同 10-14 的做法）；三個模型同一次跑，基底與 fldrop30 的四個中位數和
`docs/handoff/2026-10-13_fl_dropout.md`「離線結果」完全相同，fldrop30 1/3 s 逐幀比基底高也重現為 10/15。fldrop50 用 Hub 的 100k，hash `d826a9465f29`。

| 模型 | 1/3 s：手腕修正比例 | 1 s：手腕修正比例 | 1/3 s：front-left 跟隨比例 | 1 s：front-left 跟隨比例 |
|---|---|---|---|---|
| ntcfA1aug 100k（基底） | +0.05（−0.18～+0.33） | +0.21（−0.14～+0.78） | +0.52 | +0.58 |
| fldrop30 100k | +0.47（+0.02～+0.61） | +0.74（−0.06～+1.24） | +0.20 | +0.33 |
| **fldrop50 100k** | **+0.29**（−0.00～+0.52） | **+0.26**（−0.48～+0.94） | +0.18 | +0.31 |

| 逐幀比較（15 幀，手腕修正比例較高的幀數） | 1/3 s | 1 s |
|---|---|---|
| fldrop50 > 基底 | 8/15 | 5/15 |
| fldrop50 > fldrop30 | 3/15 | 6/15 |
| fldrop30 > 基底（對照） | 10/15 | 6/15 |

- 暫定通過標準（中位數兩個時間點都 ≥ 0.3、逐幀至少 11/15 比基底高）：**兩條都沒達到**。1/3 s 中位數 0.29 剛好在線下。
- > 1.2（修正過頭）的幀數：1/3 s 0、1 s 3（基底 0／3、fldrop30 1／4）。
- 每個模型的「需要修正量」中位數不同（1/3 s：基底 4.0、fldrop30 5.2、fldrop50 4.0 cm），因為對準的計畫 `p_align` 是各模型自己的預測；比例的分母不是同一把尺。

## 3. fldrop30-augfl（手腕不平移）訓練中

- 由 `data/logs/chain_fldrop50_augfl.sh` 在 fldrop50 開環跑完後自動開跑：10-15 02:04 起，log `data/logs/train_augfl.log`。
- 設定已生效 `[已查證]`：log 的 `front-left approach dropout` 是 `'p': 0.3`、`'image_transforms_cameras': ['observation.images.front-left']`，`expected_masked_fraction` 0.0282（同 p30）。
  每步 0.309 s（p30 0.31 s）。
- 訓練完後由 `data/logs/chain_augfl_openloop.sh` 自動算 hash、跑開環（輸出 `augfl_*`，比較 `cmp_ntcfA1augfd_vs_augfl_*`、`cmp_ntcfA1aug_vs_augfl_*`），結果寫在本檔 §4。

## 4. fldrop30-augfl 結果

（待補）

## 5. 未完成

- [ ] fldrop50 閉環（主要判準，`[柏宇說]`；條件同前兩場：far_wrist 視角、goto-home ep10、c1–c36、30 s、nas30），比較對象 fldrop30 `__f92698e7` 30/36、ntcfA1aug `__98627c1e` 25/36。
      **「p50 比 p30 好」的判準還沒定**（p50 handoff「怎麼比」）。
- [x] fldrop50 離線 h1（Windows，補充）→ §2-1
- [ ] h6（轉腕決定）沒有跑：這次沒排。
- [ ] fldrop30-augfl：訓練（約 10-15 10:50 完成）→ 開環（自動）→ §4；Hub 上不上、閉環，待柏宇決定。

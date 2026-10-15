# 2026-10-15 工作紀錄：fldrop50（front-left dropout p 0.5）與 fldrop30-augfl（手腕不平移）的訓練與開環

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

## 3. fldrop30-augfl（手腕不平移）訓練中

- 由 `data/logs/chain_fldrop50_augfl.sh` 在 fldrop50 開環跑完後自動開跑：10-15 02:04 起，log `data/logs/train_augfl.log`。
- 設定已生效 `[已查證]`：log 的 `front-left approach dropout` 是 `'p': 0.3`、`'image_transforms_cameras': ['observation.images.front-left']`，`expected_masked_fraction` 0.0282（同 p30）。
  每步 0.309 s（p30 0.31 s）。
- 訓練完後由 `data/logs/chain_augfl_openloop.sh` 自動算 hash、跑開環（輸出 `augfl_*`，比較 `cmp_ntcfA1augfd_vs_augfl_*`、`cmp_ntcfA1aug_vs_augfl_*`），結果寫在本檔 §4。

## 4. fldrop30-augfl 結果 `[已查證]`

- 訓練：10-15 02:04 → 10:48（8 h 44 m），`exit=0`，最終 loss **0.054**（l1 0.053；fldrop30 是 0.060），log `data/logs/train_augfl.log`，wandb `zpeo0bz0`。
  警告只有 Device None 與 kernel 5.4。hash：20k `cf3dd8f3b40b`、40k `8f0ad601c627`、100k `aebc8680b965`（60k／80k 在 `data/logs/chain_augfl_openloop.log`）。
- Hub：`ericc430/act_omx_b1_paper-cup_60-13-10rcvrycut-46far-augfl-fldrop30_{20k,40k,100k}`，private，LFS hash 與本機一致，各 repo 都附 `fl_dropout.json`（`data/logs/hf_upload_augfl.sh`）。
- loss 比 fldrop30 低，因為手腕畫面不再隨機平移，訓練輸入的變化比較少；訓練 loss 不能直接拿來比 `[推論]`。
- 開環：同 §2（`data/logs/openloop_augfl.sh`），輸出 `augfl_{step}_nas{N}/`，比較 `cmp_ntcfA1augfd_vs_augfl_*`、`cmp_ntcfA1aug_vs_augfl_*`。

| step | N=100 | N=50 | N=20 |
|---|---|---|---|
| 20k | 8.06 | 6.13 | 4.04 |
| 40k | 7.31 | 5.83 | 3.84 |
| 100k | 6.13 | 5.69 | 3.71 |

逐集配對 augfl − 對照（格式同 §2，括號內是 augfl 贏的集數）：

| step | N | augfl − fldrop30 | augfl − ntcfA1aug |
|---|---|---|---|
| 20k | 100 | **+1.52** [+0.34, +2.99] p=0.031 (2/12) | **+1.62** [+0.52, +2.81] p=0.016 (1/12) |
| 20k | 50 | +0.61 [−0.11, +1.38] p=0.16 (4/12) | **+1.25** [+0.61, +1.89] p=0.007 (2/12) |
| 20k | 20 | +0.21 [−0.11, +0.56] p=0.26 (5/12) | **+0.29** [+0.02, +0.59] p=0.096 (4/12) |
| 40k | 100 | **+1.14** [+0.38, +2.08] p=0.013 (3/12) | **+1.03** [+0.19, +2.03] p=0.042 (3/12) |
| 40k | 50 | **+1.04** [+0.39, +1.78] p=0.014 (3/12) | **+1.25** [+0.57, +2.03] p=0.003 (1/12) |
| 40k | 20 | +0.02 [−0.25, +0.27] p=0.91 (6/12) | +0.18 [−0.14, +0.51] p=0.33 (4/12) |
| 100k | 100 | +0.54 [−0.25, +1.33] p=0.23 (6/12) | +0.36 [−0.33, +1.14] p=0.38 (9/12) |
| 100k | 50 | **+1.12** [+0.19, +2.19] p=0.058 (4/12) | **+0.87** [+0.19, +1.64] p=0.051 (4/12) |
| 100k | 20 | −0.01 [−0.27, +0.26] p=0.95 (6/12) | +0.24 [−0.00, +0.50] p=0.11 (4/12) |

讀法：

- **方向一致是「變差」**：18 格有 17 格是正的（MAE 較高），9 格 CI 不跨 0。這已經不是偽陽性能解釋的程度（18 個檢定預期約 0.9 個）。
- **N=20 幾乎分不出，N=100／50 才變差**：預測得越遠，差距越大。
- 100k、N=100 逐關節：主要是 `shoulder_lift`（8.84 → 10.17）與 `elbow_flex`（8.13 → 9.87）變差，`wrist_flex`／`wrist_roll` 沒變。也就是「伸多遠」那兩個關節。
- 原因還沒查 `[未確認]`。可能的方向：eval 集是 9/13 錄的，那時的手腕畫面跟 10/07 之後的訓練資料不一定一樣（例如手腕相機支架在兩個時期之間動過），而這次模型在訓練時沒看過被平移的手腕畫面 `[推論]`。
  這跟 augfl handoff 的前提「手腕相機的支架在各場次之間沒有動過」`[未確認]` 直接相關。
- 跟 fldrop30／fldrop50 一樣，開環量不到接近段手腕的使用情況；開環變差也不能直接推成閉環會變差。

## 5. 未完成

- [ ] fldrop50 閉環（主要判準，`[柏宇說]`；條件同前兩場：far_wrist 視角、goto-home ep10、c1–c36、30 s、nas30），比較對象 fldrop30 `__f92698e7` 30/36、ntcfA1aug `__98627c1e` 25/36。
      **「p50 比 p30 好」的判準還沒定**（p50 handoff「怎麼比」）。
- [ ] fldrop50 離線 h1（Windows，補充）：`H1_CKPT=<fldrop50 100k snapshot> uv run python scripts/input_reliance/h1_rollout_normalized.py`
- [ ] h6（轉腕決定）沒有跑：這次沒排。
- [x] fldrop30-augfl 訓練、開環 → §4。
- [ ] fldrop30-augfl：開環在 N=100／50 一致變差（§4），原因未查；10-15 已上 Hub（§4）；閉環要不要跑，待柏宇決定。
- [ ] 手腕相機支架在 9/13 和 10/07 之後是否一樣（augfl 的前提，也可能是 §4 變差的原因）。

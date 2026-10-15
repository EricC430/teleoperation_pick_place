# 待辦：Windows 筆電上要做的事，讓 4090 能比對 10/13 場地變動、量杯緣距離（2026-10-15）

寫給：在 Windows 筆電上做的人（柏宇，或那台的 Claude）。決定者：柏宇（2026-10-15）。
背景：閉環的 dataset 只在筆電 `D:\hf\lerobot\ericc430\`。4090 和 Hub 上只有 10/06 那場 `[已查證 2026-10-15]`。

## 1. 匯出 4 場 front-left 第 0 幀 → 確認 10/13 兩場是在場地動到之後

`[柏宇說 2026-10-15]` 10/13 兩場閉環應該是在場地被動到（底座從前傾 3.3° 變水平、front-left 微移）之後跑的。這一步用畫面確認。

- 判斷依據：每場開跑前都用 `align_camera` 對齊桌面，所以相機位置本身不能拿來分辨。要看的是**底座傾斜**造成的差別：手臂在 HOME 時，front-left 畫面裡手臂的位置 `[推論]`。
- 第 0 幀手臂都在 HOME（goto-home ep10）。

| 用途 | dataset（`D:\hf\lerobot\ericc430\` 底下） |
|---|---|
| 動到之前 | `rollout_omx_b1_papercup_60_16_12rcvry_100k_paper_cup_20261008_192921`（10/08 C） |
| 待判斷 | `rollout_omx_b1_papercup_60_13_10rcvry_46far_aug_100k_nas30_paper_cup_20261013_184921` |
| 待判斷 | `rollout_omx_b1_papercup_60_13_10rcvrycut_46far_aug_100k_nas30_paper_cup_20261013_201040` |
| 動到之後、10/14 20:01 相機第二次移動之前 | `rollout_omx_b1_papercup_60_13_10rcvrycut_46far_aug_fldrop30_100k_nas30_paper_cup_20261014_151208` |

名稱是 config record 的 `dataset.repo_id` 加上時間戳；以筆電上的實際資料夾為準。

```powershell
mkdir calibration\shots\2026-10-15_rollout_f0
ffmpeg -i D:\hf\lerobot\ericc430\<dataset>\videos\observation.images.front-left\chunk-000\file-000.mp4 -frames:v 1 calibration\shots\2026-10-15_rollout_f0\<dataset>_f0.png
```

- ep 0 應該在 `file-000.mp4` 的開頭。如果 `meta/episodes` 裡 ep 0 的 front-left `from_timestamp` 不是 0，就加 `-ss <from_timestamp>`。
- 4090 那台已經有「動到之前」的 10/10 S7 B0 靜止畫面（`calibration/shots/2026-10-10_link_tilt/B0_start_front-left.png`）可以一起比。

## 2. 把 5 場閉環的 `data/`、`meta/` 傳到 Hub → 在 4090 量杯緣距離

量法與判讀：`docs/handoff/2026-10-14_rim_distance.md`。只需要關節資料，**不用傳影片**。

| 日期 | dataset |
|---|---|
| 10/08 N=100 | `rollout_omx_b1_papercup_60_16_12rcvry_100k_paper_cup_20261008_171620` |
| 10/08 N=30（C） | `rollout_omx_b1_papercup_60_16_12rcvry_100k_paper_cup_20261008_192921` |
| 10/13 ntfA1aug | `rollout_omx_b1_papercup_60_13_10rcvry_46far_aug_100k_nas30_paper_cup_20261013_184921` |
| 10/13 ntcfA1aug | `rollout_omx_b1_papercup_60_13_10rcvrycut_46far_aug_100k_nas30_paper_cup_20261013_201040` |
| 10/14 fldrop30 | `rollout_omx_b1_papercup_60_13_10rcvrycut_46far_aug_fldrop30_100k_nas30_paper_cup_20261014_151208` |

每場一次（private；repo 名跟資料夾同名）：

```powershell
uv run huggingface-cli upload ericc430/<dataset> D:\hf\lerobot\ericc430\<dataset> . --repo-type dataset --private --include "data/*" "meta/*"
```

- 上傳完到 Hub 網頁確認 `data/chunk-000/*.parquet` 和 `meta/` 都在。
- 不想上 Hub 的話，也可以照 `2026-10-14_rim_distance.md` 直接在筆電上量，兩條路擇一就好。

## 3. 開始量之前要柏宇決定的

- [ ] 夾取時算「有轉腕」的門檻：rim_distance 交代建議用抓取幀的 `wrist_roll` 狀態 > 15 `[未確認]`。
- [ ] 10/07 那兩場（temporal ensemble、遮手腕）要不要也算進去。
- [ ] 10/13 場地被動到的時間：記得的話補進 `docs/meeting/2026-10-13_closed_loop.md`。

## 4. 做完

`git add calibration/shots/2026-10-15_rollout_f0` → commit → push，再告訴 4090 那台。
第 1、2 步的結果由 4090 寫進 `docs/meeting/<日期>_rim_distance.md`。這份待辦不寫結果。

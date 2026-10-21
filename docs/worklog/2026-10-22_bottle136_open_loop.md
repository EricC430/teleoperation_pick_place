# 2026-10-22 工作紀錄：寶特瓶 135 集＋fldrop30-augfl（b136augfl）訓練與開環

參與：柏宇、Claude（AI） ｜ 性質：分析（4090 上訓練＋開環），沒有上機

為什麼做、資料組成、判準 → `2026-10-21.md` §1、§5 與 `docs/handoff/2026-10-21_bottle_136.md`（本檔不重寫）。
本機簡稱 `b136augfl`。對照組：`b60augfl`＝寶特瓶 60 集＋同方法（`35ffc9043713`，10/18 閉環 0/15），`b4a`＝舊寶特瓶模型 `phase_b4a_plastic_bottle`（同 60 集，沒有增強和 dropout）。

## 1. 訓練（2026-10-21 15:06 → 23:49，8 h 42 m）`[已查證]`

- 照 config 檔頭的 4 步：
  - 下載：4 組的集數／幀數跟 `2026-10-21.md` §2 一致。
  - 剪 recovery（`--close_below 54`）：6983 → 4945 幀，每集切在 `first_close` 前 2–12 幀，跟 Windows 試跑一樣。
  - merge：136 集、57385 幀（log `data/merge_bottle_60_normal_tightcut_precise_tipped_A1.log`）。merged ep 76 是 268 幀（＝cut ep 0），ep 134 是 5 幀（＝tipped ep 9，VOID）。
  - 4 份 `episode_meta` 只有 tipped ep 9 的 valid 是 0，所以「全部有效集」＝排除 134 後的 135 集。
- 啟動腳本 `data/logs/run_bottle136augfl.sh`，log `data/logs/train_bottle136augfl.log`（`EXIT 0`，沒有 Traceback）。output `data/train/bottle_60_normal_tightcut_precise_tipped_A1_augfl_fldrop_p30_w30`。
  wandb：https://wandb.ai/chentinghan0904-widm/teleoperation_pick_place/runs/co41bwhe
- `fl_dropout.json` 跟交接檔的預期全部一致：episodes 135、`episodes_without_close` 空、frames 57380、`frames_in_window` 3759、`expected_masked_fraction` 0.0197、`image_transforms_cameras` 只有 front-left。
- 100k loss 0.056，55.8 epoch（b60augfl 是 0.044、127.6 epoch：資料多一倍多，同樣 100k 步，epoch 數不到一半）。
- 速度 3.19 step/s，跟紙杯 129 集的 8 h 48 m 差不多。
- hash：20k `4ade453d7024`、40k `02a3bd75ebbd`、100k `fe5e98957d38`（`sha256sum` 本機檔案，也和開環 `metrics.json` 對過）。**還沒上 Hub。**

## 2. 開環 `[已查證]`

- 寶特瓶 eval-open `omx_pick_place_pilot_plastic_bottle_open_loop`（12 集，5581 幀），N = 100／50／20，在 4090 上跑（`data/logs/openloop_bottle136augfl.sh`，9 組都 `exit=0`）。
- 輸出 `outputs/open_loop_plastic_bottle/b136augfl_{step}_nas{N}/`。比較檔是 `cmp_b60augfl_vs_b136augfl_*`、`cmp_b4a_vs_b136augfl_*`。
- b4a／b60augfl 沿用 10/18 同一台機器的輸出，沒有重跑（`2026-10-17_uvc60_augfl_open_loop.md` §5；b4a 跨機器可重現）。

逐集配對 b136augfl − 對照（MAE，越低越好；95% bootstrap CI；sign-flip p；括號內是 b136augfl 贏的集數）：

**對 b60augfl**

| step | N=100 | N=50 | N=20 |
|---|---|---|---|
| 20k | 9.69 → 8.58：−1.11 [−1.93, −0.27] p=0.027 (10/12) | 6.92 → 6.09：−0.83 [−1.41, −0.23] p=0.024 (9/12) | 4.45 → 4.07：−0.39 [−0.76, −0.03] p=0.070 (8/12) |
| 40k | 9.83 → 8.51：−1.32 [−2.18, −0.47] p=0.015 (10/12) | 7.08 → 5.91：−1.17 [−1.74, −0.62] p=0.002 (11/12) | 4.60 → 3.85：−0.75 [−1.08, −0.43] p=0.001 (11/12) |
| 100k | 9.89 → 8.29：−1.60 [−2.45, −0.78] p=0.003 (11/12) | 7.11 → 5.97：−1.13 [−1.75, −0.52] p=0.005 (10/12) | 4.61 → 3.92：−0.69 [−1.02, −0.36] p=0.002 (10/12) |

**對 b4a**

| step | N=100 | N=50 | N=20 |
|---|---|---|---|
| 20k | 9.70 → 8.58：−1.12 [−2.27, +0.04] p=0.096 (8/12) | 6.83 → 6.09：−0.73 [−1.37, −0.07] p=0.057 (8/12) | 4.51 → 4.07：−0.44 [−0.81, −0.10] p=0.035 (9/12) |
| 40k | 10.00 → 8.51：−1.49 [−2.46, −0.51] p=0.017 (10/12) | 6.96 → 5.91：−1.05 [−1.63, −0.47] p=0.010 (8/12) | 4.43 → 3.85：−0.58 [−0.85, −0.31] p=0.003 (10/12) |
| 100k | 10.25 → 8.29：−1.96 [−2.94, −0.94] p=0.005 (10/12) | 7.16 → 5.97：−1.19 [−1.91, −0.39] p=0.017 (10/12) | 4.57 → 3.92：−0.64 [−0.98, −0.30] p=0.008 (9/12) |

100k、N=100 逐關節（b4a → b60augfl → b136augfl）：

| 關節 | b4a | b60augfl | b136augfl |
|---|---|---|---|
| `shoulder_pan` | 9.07 | 8.70 | 7.81 |
| `shoulder_lift` | 14.70 | 14.01 | 11.88 |
| `elbow_flex` | 14.09 | 14.07 | 11.72 |
| `wrist_flex` | 14.41 | 13.73 | 10.27 |
| `wrist_roll` | 2.81 | 2.69 | 2.55 |
| `gripper` | 6.39 | 6.14 | 5.47 |

## 3. 評估集的相機位置跟訓練資料都不同 `[已查證：align_camera.py --dry-run＋第 0 幀影像]`

- 拿 A2（寶特瓶 60 集 ep 1 11 21 31 41）和 A1（precise_A1 ep 1 11 21 31）當參考，對評估集 ep 0：
  - 對 A2：d_angle −4.6°、d_y0 +45 px
  - 對 A1：d_angle −6.4°、d_y0 +46 px
  - 評估集 ep 6 找不到分界線
- 影像：`outputs/open_loop_plastic_bottle/eval_view_vs_A2_A1.jpg`（上排 A2／A1，下排評估集 ep 0／6）。評估集的相機比較低，牆面露得比較多，看得到路由器，手臂底下的擺設也不同。
- 所以三個模型在這份評估集上看到的，都是訓練沒出現過的第三視角。三者之間的相對比較條件相同，但 MAE 的絕對值、增強有沒有幫上忙，都會混進這個視角差。
- 9/18 錄這份評估集時，相機為什麼在那個位置 `[未確認]`（`2026-09-18.md` 沒寫）。

## 4. 讀法

- **18 格全部是 b136augfl 比較低**：CI 不跨 0 的有 17 格，p < 0.05 的有 15 格（18 個檢定，期望偽陽性約 0.9 格）。
  100k 的差距最大（對 b4a −1.96），主要差在 `wrist_flex`／`shoulder_lift`／`elbow_flex`。
- 跟 10/18 §5 的「b60augfl 對 b4a 九格都分不出」對照：**方法相同時，只多了資料，開環就明顯變好**。
- 紙杯的對應比較（`2026-10-17_uvc60_augfl_open_loop.md` §2：60 集 vs 129 集，同方法）在 100k 的 N=100／50 也是資料多的比較好（N=20 分不出），但 20k 方向相反。寶特瓶這次三個 step 方向一致。
- **開環還是不能當判準** `[已查證：10-16]`：紙杯 augfl 開環比基底差，閉環卻是 34/36。這次的結果只能說「資料多了，開環預測離示範比較近」；閉環能不能超過 80%，要實際跑才知道。
- 補錄的 75 集都是 far_wrist（≈A1）視角，評估集兩邊都不是（§3），開環沒辦法回答「補錄集的視角有沒有幫到」。

## 待辦

- [ ] 上 Hub：`ericc430/act_omx_b4a_plastic-bottle_60-16-12rcvrycut-37precise-10tipped-augfl-fldrop30_{20k,40k,100k}`（private，附 `fl_dropout.json`；`docs/handoff/2026-10-21_bottle_136.md`）
- [ ] 閉環 100k：條件與判準照 `docs/handoff/2026-10-21_bottle_136.md`「怎麼比」（> 80%；c12／c6／c26 人夾不到就從分母拿掉；瓶子倒成橫的怎麼算 `[未定]`）

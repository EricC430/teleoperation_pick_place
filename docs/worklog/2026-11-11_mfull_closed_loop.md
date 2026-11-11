# 2026-11-11 現場紀錄：M-full 200k 第一次上實機（試跑）

性質：現場工作。操作：柏宇。紀錄：Claude（依柏宇口述＋本機檔案）。

## 1. 模型與設定

| 項目 | 值 |
|---|---|
| 模型 | `ericc430/act_omx_paper-cup129-plastic-bottle135-augfl-fldrop30_200k`，sha256 `8c42b7e0652e`（`docs/models.md`；Windows 下載後與 Hub LFS oid 對過） |
| 推論 | temporal ensembling：`n_action_steps=1`、coeff −0.01（config record `__137171a6` 的 cli overrides `[已查證]`） |
| 每集長度 | 紙杯 30 s（Claude 把柏宇的「紙杯，先30」理解成 30 秒；他指的是 nas30，見 §4）；寶特瓶的 config 仍是 45 s |
| config | `configs/rollout_omx_mfull_200k_te-001_paper_cup_eval.yaml` |
| 準備 | HOME：`read_joint_pose.py --goto-home --home-episode 10 --port COM8`（腳本預設讀 `teleoperate_omx_com5.yaml` 的 COM5，今天 follower 是 COM8）；唯讀讀回與 HOME 差 ≤ 2.5°。D455 柏宇現場調整中，16:11 align（不確定用哪個參考）d_angle −0.42°、d_y0 +6.3、d_y400 +3.2、dx +1.9、dy +4.1 px（未過），之後柏宇再調到「好了」，最終數字沒存 |

## 2. 有紀錄的試跑（5 集設定）

`D:\hf\lerobot\ericc430\rollout_omx_mfull_200k_te-001_30s_paper_cup_try_20261111_161710`，CSV 同名在 `episode_meta/`。

- c1–c3 三集都 success，長度 252／223／351 幀（16.7／14.8／23.3 s）`[已查證]`。
- 第 4 集中途中止：`TimeoutError: OpenCVUVCCamera(2) latest frame is too old: 505.7 ms`（手腕相機），之前 `Record loop is running slower` 10.7–14.9 Hz。只有手腕相機逾時，D455 沒有。
  - 過去的 TE 場次（10/22、10/28）同樣 `display_data: true` 沒出事 → rerun 不足以解釋 `[已查證：config records]`；今天手腕相機接在 USB hub 上、hub 稍早插拔過，**原因未確認**。

## 3. 沒錄的手動測試 `[柏宇說]`

- 紙杯：可以夾，但感覺成功率比單物體模型低（沒有計數）。
- 寶特瓶：不確定。
- **紙杯和寶特瓶同時在桌上時，不知道要夾哪一個，在中間亂夾。**
  - ACT 不讀 task 字串（lerobot `policies/act/` 沒有任何 task／語言輸入，只有 state＋兩張影像 `[已查證]`），所以換 task 描述也不會改變；
    訓練資料每集只有一種物體 → 模型沒有被教過「兩個都在時選哪個」`[推論：未逐集檢查訓練集是否有雙物體畫面]`。
    要能指定物體，得換會讀語言的策略（VLA）或另加條件輸入——另一個決策，未排。
  - 跟上車的關係：室外若同時出現多個目標，這會是問題；評估協定（`docs/experiment_spec.md`）目前一次只放一個物體 `[未確認是否該加情境]`。

## 4. 紙杯正式場結果（16:34–17:07）

`D:\hf\lerobot\ericc430\rollout_omx_mfull_cup129_bottle135_augfl_fldrop30_200k_nas1_te-001_30s_paper_cup_20261111_163453`（36 集、10215 幀，兩個視角影片都在），CSV 同名在 `episode_meta/`。TE −0.01、30 s。

- **29/36** `[已查證：CSV outcome 欄]`。失敗：c6、c12、c17、c21、c26、c32、c36（mechanism 未填）。剛好碰到階段 0 門檻（≥ 29/36）。
- 對照紙杯 34/36（`aebc8680b965`，10/16，失敗 c6、c17）：兩場共同失敗 c6、c17；今天多 5 個。
  Fisher exact（34/36 vs 29/36）雙尾 p = 0.15；Wilson 95% CI 約 [0.65, 0.90] vs [0.82, 0.98]，有重疊。**模型和推論（nas30 vs TE）同時不同**，差異不能只歸給模型。
- 柏宇：感覺比 34/36 退化，**recovery 一塌糊塗** `[柏宇說]`——還沒看影片逐集標 mechanism。
- 相機：用錄影第 0 幀 dry-run 對 far_wrist ep 1/4/8：ep0 ALIGNED（+0.34°、−1.7／+0.8 px、dx −2.0）、ep35 ALIGNED；ep18 not aligned 但木紋 r 0.07（量不到，疑似畫面被擋，未看圖）。

## 5. 寶特瓶正式場結果（17:53–20:35，多段續錄）

`D:\hf\lerobot\ericc430\rollout_omx_mfull_cup129_bottle135_augfl_fldrop30_200k_nas1_te-001_plastic_bottle_20261111_175328`（36 集），CSV 同名在 `episode_meta/`。TE −0.01、45 s、`--display_data=false`。

- **25/36** `[已查證：CSV]`。失敗：c1、c3、c5、c6、c8、c10、c12、c18、c23、c26、c28。**沒過階段 0（≥ 29/36）。**
- 對照寶特瓶單物體 23/36（`fe5e98957d38`，10/22，同 TE −0.01、45 s，只差模型）：`compare_closed_loop.py` 配對 35 點 23 vs 24，只有 A 成功 c1 c8 c18、只有 B 成功 c4 c24 c34 c35，McNemar p = 1.0 → **看不出差別**。
- 紙杯也用同一支腳本配對（§4 原本用 Fisher）：34 vs 29，5 個不一致全是單物體贏（c12 c21 c26 c32 c36），McNemar p = 0.0625；但推論不同（nas30 vs TE）。
- **中斷**：手腕相機 `latest frame is too old`。19:24 起加了 per-frame 計時（`scripts/rollout_labeled.py` 的 `FrameTimer`，`outputs/rollout_timing/`）：
  斷之前迴圈穩定 66.7 ms、follower 讀取 ≤ 2.5 ms（同一個 hub）、D455 畫面一直新、CPU 低，**只有手腕畫面從某一幀起完全不再更新**（年齡每幀 +66.7 ms 到 500 ms）→ 不是過載、不是 GIL、不是整個 hub；是手腕相機那條線路 `[已查證：計時紀錄]`。
  柏宇：斷的時候多半是手臂伸得比較遠 → 推論是手腕相機線被拉到、接觸不良 `[柏宇說＋推論]`。
- c6（ep5）在 19:24–19:34 斷了 4 次；20:00 那段 ep5–35 一次跑完，只有 ep31（c32，成功）手腕畫面停過 243 ms（< 500，沒中斷）。c1–c5 錄在計時之前，沒有停頓紀錄。
- 空資料夾（0 集）：`…_174216`、`…_174531`、`…_175141`，可刪。

## 6. 接著

- 紙杯正式 36 集改用 **nas30**（`n_action_steps=30`、不用 TE）、30 s：柏宇 11/11「紙杯一樣是30步」——之前的「先30」指的是 nas30，被 Claude 理解成 30 秒，所以 §2 的試跑是 TE，不是柏宇要的設定。
  config：`configs/rollout_omx_mfull_200k_nas30_paper_cup_eval.yaml`。跟紙杯 34/36（`aebc8680b965`，nas30、30 s）推論和長度相同，只差模型；但跟寶特瓶（TE、45 s）不再是「兩種物體同設定」（10/22 決定）。
- **隨後改回 TE**（柏宇 11/11：「那不然就te吧」）：正式場用 TE −0.01、30 s，即 `configs/rollout_omx_mfull_200k_te-001_paper_cup_eval.yaml`（與試跑同一份）。nas30 那份沒用到。
  跟紙杯 34/36 比：模型和推論都不同；跟寶特瓶比：推論相同、每集長度不同（30 vs 45 s）。

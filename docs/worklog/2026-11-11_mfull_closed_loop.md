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

`D:\hf\lerobot\ericc430\rollout_omx_mfull_200k_te-001_30s_paper_cup_try_20261111_161710`。2026-11-12 查：這個資料夾和同名 CSV 都已不在（找過 `D:\hf\lerobot\ericc430`、`episode_meta/`），下面的數字只剩這裡有記錄。

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

- **29/36** `[已查證：CSV outcome 欄]`。失敗：c6、c12、c17、c21、c26、c32、c36。mechanism 正本在 CSV：柏宇標的，11/12 Claude 依 12 格截圖補 `timeout`、c21／c32 從 `bad_aim` 改成 `other`（夾爪下到杯口內才閉合、水平方向有對準），改動在 notes 裡標 `[AI補]`、待柏宇確認；C 型夾相關（c6、c12）照柏宇的判斷沒動。柏宇 11/12 判 c6、c12、c21、c26、c32 都是 `repetition_loop`（同一處重複下爪、沒有進展）。判準：重試是在重複同一個錯，還是有在修正位置；物體被碰到移位的不算（見 §5 寶特瓶）。這是第一次用這個標籤；10/16、10/22 沒有用同一判準回頭看，也不打算補看 `[柏宇說 11/12]` → 舊場次沒有這個標籤，不代表沒有 loop，`repetition_loop` 的次數不要拿來跨場比。剛好碰到階段 0 門檻（≥ 29/36）。
- 對照紙杯 34/36（`aebc8680b965`，10/16，失敗 c6、c17）：兩場共同失敗 c6、c17；今天多 5 個。
  Fisher exact（34/36 vs 29/36）雙尾 p = 0.15；Wilson 95% CI 約 [0.65, 0.90] vs [0.82, 0.98]，有重疊。**模型和推論（nas30 vs TE）同時不同**，差異不能只歸給模型。
- 柏宇：感覺比 34/36 退化，**recovery 一塌糊塗** `[柏宇說]`。截圖上 7 個失敗多半是在同一處重複下爪 2–3 次，杯子沒被推走；5 集跑滿 30 s（444–447 幀），c6、c36 是 430、426 幀，這場沒有計時紀錄，分不出是到時還是提早按失敗。
- 相機：用錄影第 0 幀 dry-run 對 far_wrist ep 1/4/8：ep0 ALIGNED（+0.34°、−1.7／+0.8 px、dx −2.0）、ep35 ALIGNED；ep18 not aligned 但木紋 r 0.07（量不到，疑似畫面被擋，未看圖）。

## 5. 寶特瓶正式場結果（17:53–20:35，多段續錄）

`D:\hf\lerobot\ericc430\rollout_omx_mfull_cup129_bottle135_augfl_fldrop30_200k_nas1_te-001_plastic_bottle_20261111_175328`（36 集），CSV 同名在 `episode_meta/`。TE −0.01、45 s、`--display_data=false`。

- **25/36** `[已查證：CSV]`。失敗：c1、c3、c5、c6、c8、c10、c12、c18、c23、c26、c28。**沒過階段 0（≥ 29/36）。**
- 對照寶特瓶單物體 23/36（`fe5e98957d38`，10/22，同 TE −0.01、45 s，只差模型）：`compare_closed_loop.py` 配對 35 點 23 vs 24，只有 A 成功 c1 c8 c18、只有 B 成功 c4 c24 c34 c35，McNemar p = 1.0 → **看不出差別**。
- 紙杯也用同一支腳本配對（§4 原本用 Fisher）：34 vs 29，5 個不一致全是單物體贏（c12 c21 c26 c32 c36），McNemar p = 0.0625；但推論不同（nas30 vs TE）。
- **中斷**：手腕相機 `latest frame is too old`。19:24 起加了 per-frame 計時（`scripts/rollout_labeled.py` 的 `FrameTimer`，`outputs/rollout_timing/`）：
  斷之前迴圈穩定 66.7 ms、follower 讀取 ≤ 2.5 ms（同一個 hub）、D455 畫面一直新、CPU 低，**只有手腕畫面從某一幀起完全不再更新**（年齡每幀 +66.7 ms 到 500 ms）→ 不是過載、不是 GIL、不是整個 hub；是手腕相機那條線路 `[已查證：計時紀錄]`。
  柏宇：斷的時候多半是手臂伸得比較遠 → 推論是手腕相機線被拉到、接觸不良 `[柏宇說＋推論]`。
- 失敗集只有 c6、c26 跑滿 45 s（673、669 幀），其餘 9 集在 20–29 s 按了失敗（20:00 之後的有計時紀錄，c1–c5 用幀數判斷）；10/22 寶特瓶那場也是這樣做，所以兩場比較仍然公平。mechanism 正本在 CSV：11/12 Claude 依截圖補 c6、c26 的 `timeout`，c6 補 `pushed_away`，c18 依柏宇自己的 note（距離夠、閉爪時後縮）拿掉 `bad_aim` 改 `other`，改動都標 `[AI補]`。c26 Claude 原本看截圖說瓶子沒移位、拿掉了 `pushed_away`，柏宇看過說有移位，已加回（`bad_aim;pushed_away;timeout`）。`repetition_loop`：c6 可能是，但物體有移位，先不算；c26 物體有移位，不算 `[柏宇判 11/12]`。
- c6（ep5）在 19:24–19:34 斷了 4 次；20:00 那段 c6 又先錄了一次，被 Left 重錄（20:00:41–20:01:13，計時紀錄），存下來的是 20:01:45 開始那次；重錄是因為不確定擺位放得對不對 `[柏宇說 11/12]`。c1 也跑了好多次，原因是一直跳掉（中斷）`[柏宇說 11/12]`，c1–c5 在加計時之前，沒有停頓紀錄可以對；之後 ep5–35 一次跑完，只有 ep31（c32，成功）手腕畫面停過 243 ms（< 500，沒中斷）。c1–c5 錄在計時之前，沒有停頓紀錄。
- 空資料夾（0 集）：`…_174216`、`…_174531`、`…_175141`，可刪。
- config record（`config_records/2026-10-08_omx_follower/`，11/12 補進 git）：紙杯試跑 `__137171a6`（16:16）、紙杯正式 `__bc3e6dcc`（16:34）；寶特瓶 `__d39a1ec1`（17:38，沒有 `--display_data=false`）、`__23810125`（17:42，新開）、續錄 `__70aa3514`（`num_episodes=34`）、`__12a55ac5`（33）、`__73a8b56f`（31）。對應哪一段是依檔案時間和 cli overrides 推的 `[推論]`；`__d39a1ec1` 對不到任何資料夾。

## 6. 接著

- 紙杯正式 36 集改用 **nas30**（`n_action_steps=30`、不用 TE）、30 s：柏宇 11/11「紙杯一樣是30步」——之前的「先30」指的是 nas30，被 Claude 理解成 30 秒，所以 §2 的試跑是 TE，不是柏宇要的設定。
  config：`configs/rollout_omx_mfull_200k_nas30_paper_cup_eval.yaml`。跟紙杯 34/36（`aebc8680b965`，nas30、30 s）推論和長度相同，只差模型；但跟寶特瓶（TE、45 s）不再是「兩種物體同設定」（10/22 決定）。
- **隨後改回 TE**（柏宇 11/11：「那不然就te吧」）：正式場用 TE −0.01、30 s，即 `configs/rollout_omx_mfull_200k_te-001_paper_cup_eval.yaml`（與試跑同一份）。nas30 那份沒用到。
  跟紙杯 34/36 比：模型和推論都不同；跟寶特瓶比：推論相同、每集長度不同（30 vs 45 s）。

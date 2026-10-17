# 2026-10-16 現場紀錄：fldrop50／fldrop30-augfl 100k 紙杯閉環

參與：柏宇、Claude（AI） ｜ 性質：現場工作紀錄，非會議

為什麼跑這兩個模型：`docs/handoff/2026-10-14_fl_dropout_p50.md`、`docs/handoff/2026-10-14_fl_augfl.md`；開環與離線 h1：`2026-10-15_fldrop50_open_loop.md`（本檔不重寫）。
比較對象：fldrop30 100k `__f92698e7` 30/36、ntcfA1aug 100k `__98627c1e` 25/36。

## 1. 條件 `[已查證]`

| run | 模型（hash） | config record | dataset／episode_meta 時戳 |
|---|---|---|---|
| fldrop50 | `…-46far-aug-fldrop50_100k`（`d826a9465f29`） | `__ab0df39c`（c1–c15）、`__ca4db9c5`（14:42 續錄 c16–c36） | `…_aug_fldrop50_100k_nas30_paper_cup_20261016_142101` |
| fldrop30-augfl | `…-46far-augfl-fldrop30_100k`（`aebc8680b965`） | `__b9d19490`（15:20） | `…_augfl_fldrop30_100k_nas30_paper_cup_20261016_152018` |

- 兩個模型的本機快取都重算過 hash，與 `docs/models.md` 一致。
- 兩場都是：N = 30、30 s、c1–c36、`--goto-home --home-episode 10`，`env_light` = 宿舍大燈+檯燈+窗簾拉上+亮度168。
- `D:\hf\lerobot\ericc430\…_fldrop50_…_20261016_141642` 是 0 集的空資料夾（14:16 那次啟動沒錄到集數，原因 `[未確認]`），不用。
- `__9407e9b4`（14:39）是第一次續錄沒給 `--dataset.root` 而報錯的那次（§2-1），沒有錄到資料。

### 1-1. 相機與亮度：和前幾場不同，兩場彼此相同

支架在 10/14 20:01 被整個動過 `[柏宇說]`（fldrop30 是 10/14 15:12，在這之前）。
開跑前最後一張 align 畫面（`outputs/camera_align/live_20261016_140100.png`，對 far_wrist ep 1/4/8）：

| | fldrop30（10/14） | 10/13 兩場 | **本日兩場（14:01 量）** |
|---|---|---|---|
| d_angle | −0.44° | −0.31～−0.64° | +0.02° |
| d_y0 | +1.8～+2.7 px | +3.0～+4.1 px | −0.0 px |
| 桌面木紋 dx | −0.6 px | +0.6～+1.5 px | **+28.9 px**（dy −9.0，r 0.41） |
| 牆上線槽右端 | (−1, −2) | (0, −1) | **(+27, −5)** |
| 桌面亮度（第 0 幀左半平均） | 146–158 | 147–152 | **168** |

- 訓練資料（uvc_60、normal_A1、recovery_A1_tight、far_wrist_A1，各抽 6 集）亮度是 141–149；訓練沒有亮度增強（`brightness` weight 0）。
- dx +29 px 在訓練隨機平移範圍（±42 px）內，但過去每一場都在 ±2 px 內。牆（遠）和桌面（近）偏移一致，是相機轉了約 4°，不是位置移了。
- **柏宇決定照這個狀態跑**（「就現在這樣吧」）。augfl 那場沿用同一個狀態；開跑前沒有另存 align 畫面，相機有沒有再動 `[未確認]`。
- 所以：**本日兩場彼此條件相同**；它們和 fldrop30／ntcfA1aug 的差別不只模型，還有相機約 4° 的水平轉向、亮度 168。
- 對齊過程：13:25 d_angle −0.66°、dx +2.5；13:31 d_angle −0.03°、dx +15。看起來像「角度和 dx 互斥」，
  但兩張畫面的遠近偏移一致，是調角度時手同時帶到水平轉向，不是支架位置回不去 `[推論，精度幾 px]`。
  13:25–13:31 亮度 121–131，柏宇說是晾衣服擋光；14:01 變 168，中間改了什麼沒記到 `[未確認]`。

## 2. fldrop50 中斷與續錄

### 2-1. ep 15 開始時馬達全部沒回應（14:34:45）

```
[saved] episode 14: success
Saved. Recording episode 15
WARNING Could not return to initial position: Failed to sync read 'Present_Position' on ids=[11, 12, 13, 14, 15, 16] after 1 tries. [TxRxResult] There is no status packet!
ConnectionError: Failed to sync read 'Present_Position' ... There is no status packet!   (episodic.py _policy_loop)
ConnectionError: Failed to write 'Torque_Enable' on id_=11 with '0' after 6 tries. [TxRxResult] There is no status packet!   (disconnect)
```

- **已存的資料完整** `[已查證]`：當時 `meta/info.json` 15 集／3918 幀；teardown 先 `dataset.finalize()` 再斷硬體，錯誤發生在之後。CSV 的 c16–c36 列被自動刪掉。
- 六顆馬達同時無回應、連關扭力都寫不進去，但 port 還在（不是 9/18 §3-1 那種 `SerialException`）。像是整條匯流排斷電或斷線 `[推論]`。
  9/18 §2 是只有 11 號沒回應，重插從動臂電源解掉（`2026-09-18.md` §2）。
  **這次重插 USB 就好了** `[柏宇說]`。是 USB 接觸不良還是 U2D2 暫時失效，分不出來 `[未確認]`。
- 14:40 再啟動時手臂已正常連上（`OmxFollower connected`），所以通訊在 14:34–14:40 之間恢復了。

### 2-2. 續錄寫法（14:42 成功，寫回同一個 dataset 和 CSV）

```powershell
uv run python scripts/rollout_labeled.py --config_path configs/rollout_omx_b1_papercup_ntcfA1augfldrop50_100k_eval.yaml --policy.path=ericc430/act_omx_b1_paper-cup_60-13-10rcvrycut-46far-aug-fldrop50_100k --policy.n_action_steps=30 --meta-template paper_cup_A1 --env-light "宿舍大燈+檯燈+窗簾拉上+亮度168" --resume=true --dataset.repo_id=ericc430/rollout_omx_b1_papercup_60_13_10rcvrycut_46far_aug_fldrop50_100k_nas30_paper_cup_20261016_142101 --dataset.root=D:\hf\lerobot\ericc430\rollout_omx_b1_papercup_60_13_10rcvrycut_46far_aug_fldrop50_100k_nas30_paper_cup_20261016_142101 --dataset.num_episodes=21
```

- `--dataset.root` 必填，指到 dataset 本身那層：沒給時 `LeRobotDataset.resume` 直接報錯（14:39 踩到，`lerobot_dataset.py` L854）。
- `num_episodes` 是「這次再錄幾集」（`2026-09-18.md` §3-1）。
- CSV：`MetaSheet` 從 `first_index = dataset.num_episodes`（15）開始，用原範本第 15–35 列（c16–c36），既有 15 列保留（`rollout_labeled.py` L91、L211）。
- resume 不會再加時間戳（`configs/dataset.py` `stamp_repo_id`）。
- resume 時 `dataset.meta.stats`（已錄集數的統計）會傳進 `make_pre_post_processors`，但有 `pretrained_path` 時 ACT 直接從模型載入 normalizer，不用它（`policies/factory.py` L199）。所以前後兩段的模型輸入處理相同 `[已查證：讀程式碼]`。
- 這場的兩段之間手臂重新回 HOME、相機重新開過。
- 結果：dataset 36 集／10230 幀，CSV 36 列 `[已查證]`。

## 3. 結果 `[已查證：episode_meta CSV、scripts/compare_closed_loop.py]`

| run | 全部 | near < 22 | mid 22–33 | far ≥ 33 | 失敗位置 |
|---|---|---|---|---|---|
| ntcfA1aug（10/13） | 25/36 | 4/6 | 12/18 | 9/12 | — |
| fldrop30（10/14） | 30/36 | 4/6 | 17/18 | 9/12 | — |
| fldrop50（本日） | 28/36 | 4/6 | 14/18 | 10/12 | c3、c12、c17、c18、c22、c30、c33、c36 |
| **fldrop30-augfl（本日）** | **34/36** | 6/6 | 17/18 | 11/12 | c6、c17 |

同位置配對（McNemar exact，雙尾）：

| augfl 對 | augfl 贏 | augfl 輸 | p |
|---|---|---|---|
| fldrop50（同日、同相機與亮度） | c3、c12、c18、c22、c30、c33、c36（7） | c6（1） | 0.070 |
| fldrop30 | c10、c12、c26、c34、c36（5） | c6（1） | 0.22 |
| ntcfA1aug | 10 | c6（1） | 0.012 |

- 條件最乾淨的是 augfl 對 fldrop50（同一天、同一個相機位置和亮度）：p 0.070。
- 對 fldrop30、ntcfA1aug 時還多了相機轉向、亮度兩個差別；而且這裡做了三次比較，沒有做多重比較校正。
- **離線指標沒有預測到閉環**：augfl 開環在 N=100／50 比 fldrop30 差、離線 h1 手腕修正 +0.16／−0.00 也沒過暫定門檻（`2026-10-15_fldrop50_open_loop.md`），閉環卻是四場最高。fldrop50 的離線 h1 也沒過門檻，閉環 28/36。
- 柏宇的判讀 `[柏宇說]`：fldrop30-augfl 表現很不錯。
- 兩場的 `mechanism` 欄都還是空的；augfl handoff 要求的 `bad_aim` 偏移方向也還沒記。

## 4. 下一步（柏宇 2026-10-16 提出，細節未定）

### 4-1. 只用原本的 uvc_60（60 集）＋ fldrop30 ＋ 只在第三視角平移，看效果

**目的** `[柏宇說]`：評估換成鋁罐、寶特瓶時，是不是一定要多收集資料（normal_A1、recovery、far_wrist 這類補錄），還是原本 60 集＋這組增強就夠。

已知 `[已查證]`：
- 訓練入口與設定可沿用 `scripts/train_fl_dropout.py --fl-p 0.3 --fl-window 30 --transforms-fl-only`；資料改成只有 `omx_pick_place_pilot_uvc_60`。
- 只用 uvc_60、不加增強的舊模型是 `act_omx_b1_uvc60_100k`（`2bdff2b3871e`，`docs/models.md`）。

訓練（2026-10-16 16:21 開跑，widm 4090；容器內 log 是 UTC，顯示 08:21）`[已查證]`：
- `configs/train_omx_uvc60_augfl_fldrop.yaml`（資料只換成 uvc_60，其餘照抄 `…_aug_fldrop.yaml`）＋ `--fl-p 0.3 --fl-window 30 --transforms-fl-only`；
  output `data/train/uvc60_augfl_fldrop_p30_w30`，log `data/logs/train_uvc60_augfl.log`，wandb `6cnoeibs`。
- 開跑前 `data/logs/fldrop_check/check_uvc60_augfl.py`（同 10-14 `check_augfl.py`）：60 集都有閉合，front-left 每次抽樣不同、手腕不變，窗口內遮 62/200、65/200（期望 60）。
- `expected_masked_fraction` 0.0338（129 集版 0.0282）：uvc_60 每集較短，30 幀窗口佔比較高，不是設定不同。
- 3.24 step/s，10/17 01:07 跑完（8 h 45 m，EXIT 0，log 無錯誤）；hash 見 `docs/models.md`（20k `c7c9dfceb900`、40k `27c5ba98b2c4`、100k `f55b0fa71b50`）。
  `fl_dropout.json`：60 集、`image_transforms_cameras` 只有 front-left、`expected_masked_fraction` 0.0338。100k loss 0.044（200 epoch）。
- Hub 與開環 → `2026-10-17_uvc60_augfl_open_loop.md`。

已定 `[柏宇決定 2026-10-16]`：
- **比較對象**：舊的 60 集模型 `act_omx_b1_uvc60_100k`（`2bdff2b3871e`）和本日 fldrop30-augfl（34/36）。
- **判準**：綜合手腕瞄準準確度（§4-2）和成功率；**成功率到 8 成**就先不收新資料，直接用相同超參數訓練寶特瓶或鋁罐模型。

已查：uvc_60 的擺位涵蓋遠區 `[已查證：t1–t60 的 r 由 placement_label_map_campA_136sym_20260908.csv 算，切點同 compare_closed_loop.py]`：

| | near < 22 | mid 22–33 | far ≥ 33 | r 範圍 |
|---|---|---|---|---|
| uvc_60（t1–t60） | 10 | 30 | 20 | 12.6–38.6 cm |
| 閉環（c1–c36） | 6 | 18 | 12 | 13.0–39.0 cm |

只比了距離，沒有比角度；而且錄的時候是不是照這張表擺的，是 `[推論]`。

還沒定 `[未定]`：
- 舊的 60 集模型上一次閉環是 10/6（N=30，17/32，`2026-10-06.md`），當時的相機、位置都跟現在不同。要比的話，是在現在的條件下重跑一場，還是直接用 10/6 的結果？
- 8 成換算成 36 集是 ≥ 29/36。§4-2 的準確度還沒定義，怎麼跟成功率「綜合」也還沒定。
- uvc_60 錄製時的相機視角和 far_wrist 視角不同（uvc_60 ↔ 10-07 約 1°／3–6 px，`scripts/align_camera.py` docstring；10-07 ↔ far_wrist 約 20 px，`docs/handoff/2026-10-13_aug_retrain.md`），閉環要對哪一個？現在相機又比 far_wrist 偏 +29 px。
- uvc_60 沒有轉腕示範；本日 augfl 的成功裡有多少是轉腕抓的，也還沒算。

### 4-2. 量化夾取時手腕相機瞄準杯壁的準確度

**Eric 提出的做法** `[柏宇轉述 Eric]`：用夾取前幾幀的手腕畫面，加上 FK，算夾爪到杯壁的距離，得到一個準確度。

相關的現有東西 `[已查證：檔案存在]`：
- `docs/handoff/2026-10-14_rim_distance.md`：只用關節資料＋FK，以「擺放點＝杯心」量到杯緣的距離。杯子被推走（`pushed_away`）時這個假設不成立。
- 手腕相機內參 `calibration/2026-10-07_camera_intrinsics_wrist.json`；手腕相機安裝位置用 touch-pinned 手臂模型重解過（commit `086c827`）。

要先定的 `[未定]`：
- 「距離」是哪一點到哪一點：指尖中點？夾爪中線？到杯壁的哪裡？
- 杯壁在手腕畫面裡怎麼找：人工點選，還是自動偵測？
- 「夾取前幾幀」：第一次閉合前幾幀？用幾幀？
- 怎麼驗證算出來的距離是對的？要不要也套在人類示範上，當作底線（rim_distance 有人類底線）？
- 和 rim_distance 的關係：取代它、補它（處理 `pushed_away`），還是兩個都做？

## 待辦

- [ ] 兩場補 `mechanism`；augfl 的 `bad_aim` 補偏移方向（本場只有 2 集失敗）
- [x] §4-1 的比較對象與判準（成功率 8 成；準確度怎麼綜合未定）
- [x] §4-1 訓練跑完、hash、`docs/models.md` 補列
- [x] §4-1 上 Hub；開環 → `2026-10-17_uvc60_augfl_open_loop.md`
- [ ] §4-1 閉環（舊 60 集模型要不要在現在的條件下重跑）
- [ ] §4-2 的距離定義與驗證方式

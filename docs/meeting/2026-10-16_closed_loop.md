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
- 兩場的 `mechanism` 已補（10-16 晚）；augfl handoff 要求的 `bad_aim` 偏移方向 CSV 裡沒有。

### 3-1. 柏宇對 fldrop30-augfl 的觀察（10-16 補）

`[柏宇說]`
- 認為 augfl **更常用第一視角（手腕）**：c17、c36 在 frame 0 時，第三視角裡的杯子被 C 型夾／底座擋住，手臂仍然瞄得到杯子。
- 有些位置的夾取方式跟預期不同，但也成功了（哪些位置、哪一種抓法：未記）。
- 兩次失敗：c6 撞到 C 型夾；c17（比較難的位置）把杯子推到夾不起來的地方。

AI 補充（給下次判讀用，不推翻上面的觀察）
- **c36 四場對照** `[已查證：CSV]`：ntcfA1aug（10/13）、fldrop30（10/14）、fldrop50（本日）都是 `no_grasp`／`bad_aim`，只有 augfl 成功。
  fldrop50 跟 augfl 同一天、同一個相機和亮度，所以「相機轉了約 4°」解釋不了 c36 這一格的差別；但每個位置都只有 n = 1，
  10-13 兩個 aug 模型之間有 14/36 個位置結果不同。
- **c17 四場都失敗**。augfl 這集原本標 `bad_aim;pushed_away`，柏宇確認是瞄準之後才把杯子推走，所以改成只標 `pushed_away`。
- 另外兩個被擋住的位置 c7、c16，四場都成功，所以分不出模型之間的差別。
- **「以前一律往 c16 夾，這次往右」，關節資料也看得到** `[已查證：rollout 資料的 observation.state]`。
  擺位座標：c16 的 y = +31.8，c17 = −22.8，c36 = −13.0（`docs/assets/placement_label_map_campA_136sym_20260908.csv`）。
  第一次閉合（gripper .pos 從約 59 降到約 50）時的 shoulder_pan（每 2 s 取樣，約略值）：

  | run | c16（對照） | c17 | c36 |
  |---|---|---|---|
  | ntcfA1aug（10/13） | ≈ +23～+33 | ≈ +34 | ≈ +34 |
  | fldrop30（10/14） | ≈ +35 | 整集沒閉合，pan 停在 ≈ +34 | ≈ +36 |
  | fldrop50（本日） | ≈ +27 | ≈ +25～+34 | ≈ +31～+37 |
  | **augfl（本日）** | ≈ +36 | **≈ −33** | **≈ −16** |

  - 前三場的 c17、c36 都轉到跟 c16 一樣的 pan ≈ +34，也就是杯子的反方向；只有 augfl 轉向負的一側，也就是杯子那一側。
  - augfl c17 在約 16 s 張開夾爪之後，20–28 s 的 pan 又一路轉回 +30（c16 那一側）。
  - 這裡只看正負號，沒有用 FK 核對轉的幅度對不對。
- 上面這些能說明 augfl 在這兩格沒有照背下來的 c16 軌跡走。但它是靠手腕畫面，還是靠第三視角裡沒被擋住的部分（這一版只對第三視角做平移增強），
  用閉環資料分不出來 `[推論]`。
- **離線指標沒有支持「更用手腕」**：augfl 的離線 h1 手腕修正 +0.16／−0.00，是三個 dropout 模型裡最低的（`2026-10-15_fldrop50_open_loop.md` §4-1）。
  h1 量的是 front-left 看得到時的幀，量不到「第三視角被擋住時改用手腕」，所以兩者不算矛盾；但目前支持這個說法的只有 c36（n = 1），
  加上 c17（如果確實有瞄準）。**「augfl 更用手腕」目前是假設** `[推論]`。
- 可以拿來驗證的資料已經錄下來了：augfl 和 fldrop50 本日的 c17、c36 都有 rollout 資料，可以在這幾集上離線把手腕畫面換成灰色
  （同 h6 的做法），看接近段的預測動作改變多少 `[未定：要不要做、判準是什麼]`。

> CSV 修正（10-16）：augfl 的 CSV 表頭被改成 `episode`，`compare_closed_loop.py` 讀不到任何列，已改回 `episode_index`。
> c6 原本 `outcome` 填 `collision`（那是 mechanism 的值），改成 `outcome=no_grasp`、`mechanism=collision`，同 10-14 c12 的寫法。
> c17 拿掉 `bad_aim`（見上）。改完重跑 `compare_closed_loop.py`：34/36，對 fldrop50 p 0.070，跟 §3 相同。

## 4. 下一步（柏宇 2026-10-16 提出，細節未定）

### 4-1. 只用原本的 uvc_60（60 集）＋ fldrop30 ＋ 只在第三視角平移，看效果

**目的** `[柏宇說]`：評估換成鋁罐、寶特瓶時，是不是一定要多收集資料（normal_A1、recovery、far_wrist 這類補錄），還是原本 60 集＋這組增強就夠。

已知 `[已查證]`：
- 訓練入口與設定可沿用 `scripts/train_fl_dropout.py --fl-p 0.3 --fl-window 30 --transforms-fl-only`；資料改成只有 `omx_pick_place_pilot_uvc_60`。
- 只用 uvc_60、不加增強的舊模型是 `act_omx_b1_uvc60_100k`（`2bdff2b3871e`，`docs/models.md`）。

開跑前要先定的 `[未定]`：
- 比較對象是誰：只看「60 集＋增強」對「129 集＋增強」（本日 34/36）？還是也要在現在的條件下重跑舊的 60 集模型，才分得出「增強」和「資料量」各自的貢獻？
- uvc_60 錄製時的相機視角和 far_wrist 視角不同（uvc_60 ↔ 10-07 約 1°／3–6 px，`scripts/align_camera.py` docstring；10-07 ↔ far_wrist 約 20 px，`docs/handoff/2026-10-13_aug_retrain.md`），閉環要對哪一個？現在相機又比 far_wrist 偏 +29 px。
- uvc_60 的擺位有沒有涵蓋 c1–c36 的遠區（far ≥ 33 有 12 個位置）？如果沒有，遠區失敗會跟「資料不夠」混在一起。
- uvc_60 沒有轉腕示範；本日 augfl 的成功裡有多少是轉腕抓的，也還沒算。
- 什麼結果算「不用多收資料」。

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

- [x] 兩場補 `mechanism`
- [x] c6 的 `outcome`、c17 的 `bad_aim`（§3-1）
- [ ] augfl 的 `bad_aim` 補偏移方向（c17 改掉之後，augfl 已經沒有標 `bad_aim` 的失敗；這項只剩 c7（成功、自行修正）可以補）
- [ ] 「跟預期不同但成功」的抓法：哪些位置、哪一種抓法（§3-1）
- [ ] §4-1 的比較對象與判準
- [ ] §4-2 的距離定義與驗證方式

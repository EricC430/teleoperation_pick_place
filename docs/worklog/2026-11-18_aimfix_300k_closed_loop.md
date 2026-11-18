# 2026-11-18 aimfix28_twoobj10 300k 寶特瓶＋紙杯閉環：階段 0 兩種都 ≥ 29/36

參與：柏宇（現場、判定、標註）、Claude（AI，config、資料核對、依柏宇確認修標註、整理） ｜ 性質：現場工作（11/18）

接在 `2026-11-17_aimfix_closed_loop.md`（同一個 run 的 200k，寶特瓶 v4 26/36）之後。300k 是同一個 run 的最後一個 checkpoint；
200k 與 300k 的 `config.json`、`train_config.json` 完全相同，只差權重 `[已查證 11/18]`。

## 設定

| 項目 | 值 |
|---|---|
| 模型 | `ericc430/act_omx_cup129-bottle135-aimfix28-twoobj10-augfl-fldrop30_300k`（`b431d2172af2`，`docs/models.md`；本機下載後 sha256 核對相符） |
| 推論 | TE：`n_action_steps=1`、coeff −0.01；寶特瓶 45 s／集、紙杯 30 s／集 `[已查證：config record cli overrides]` |
| config | 寶特瓶 `configs/rollout_omx_aimfix_300k_te-001_plastic_bottle_eval.yaml`，record `__dc913a16`（第一段）＋`__e87dc54e`（`--resume`，22 集）；紙杯 `configs/rollout_omx_aimfix_300k_te-001_paper_cup_eval.yaml`，record `__a5ebf82a`＋`__1666e0be`（`--resume`，31 集） |
| 清單 | 寶特瓶 v4（`configs/placements/bottle_eval-close_20261115.csv`，範本 `plastic_bottle_A1_v4`）；紙杯 c1–c36（`campA_136sym_20260908_20260908_eval-close.csv`，範本 `paper_cup_A1`） |
| 資料 | 寶特瓶 `D:\hf\lerobot\ericc430\rollout_omx_aimfix28_twoobj10_300k_nas1_te-001_plastic_bottle_20261118_141713`（36 集、13722 幀）；紙杯 `..._300k_nas1_te-001_30s_paper_cup_20261118_153802`（36 集、9415 幀）。CSV 同名在 `episode_meta/`。都還沒上 Hub |
| 環境 | 宿舍大燈＋檯燈＋窗簾拉上、木質桌面（兩場相同，同 11/17） |

repo 名用 11/18 定的短名（`rollout_omx_aimfix28_twoobj10_...`，Hub 名稱上限 96 字元），本機名與 Hub 名一致。

## 現場事件（會影響怎麼讀數字）

- **手腕相機斷流（寶特瓶開跑前）**：13:57–14:03 四次都在第 0 集中止，`OpenCVUVCCamera(2) latest frame is too old`（> 500 ms）；front-left 正常。
  單獨用 OpenCV 讀 index 2（不連手臂）：4.8 s 後就不再出畫面，20 s 平均 6.2 fps → 手臂不動也會斷，是相機這條線的問題。當時 Innomaker 經 `Generic USB Hub`（VID_1A40）接上，D455 直插 `[已查證：PnP LocationInfo]`。
  之後怎麼處理、11/17 是否也經過這顆 hub：`未確認`。141713 這場 36 集沒有再因相機中止。
- **front-left 被推倒（寶特瓶 ep 13＝c16 之後）**：重新用 `align_camera.py` 對齊後 `--resume`。**ep 14（c17）以後的 front-left 位置與前 14 集不完全相同**。
- **紙杯 ep 4（c5）之後中斷**：柏宇以為擺錯點，其實沒有；`--resume` 接 c6 起 31 集。
- 中止留下的空資料夾（0 集，未刪）：寶特瓶 `..._20261118_135710／135804／135916／140314／141545`、te −0.05 的 `..._161759／162519`，`episode_meta/` 有同名空 CSV。

## 結果

### 寶特瓶：29/36，過階段 0 門檻（≥ 29/36）

- 成功 29、`no_grasp` 4（c16 c33 c36 c38）、`dropped` 3（c11 c25 c35）`[已查證：CSV]`。
- 近／遠（r < 34 ／ ≥ 34）：**19/24 ／ 10/12**（11/17 200k：20/24 ／ 6/12）。這次失敗的 7 點有 5 點是近點（c11 c25 c33 c35 c36，r 13–31 cm）。
- 跟 11/17 200k 同 36 點配對：只有這次成功 8 點（c3 c4 c5 c7 c19 c28 c32 c39）、只有 11/17 成功 5 點（c11 c25 c33 c35 c36）；McNemar 精確 p = 0.58 → **分不出差別**。
- v4 清單比 c1–c36 原版容易（`2026-11-15_bottle_eval_list.md`）：門檻套在 v4 上，代表的難度比原版低；不能跟 11/11 的 25/36 比。
- 成功但 note 寫「瞄的小小歪」（c3 c5 c7 c9）、「有推瓶子／推到瓶子」（c7 c23 c24 c32）、「差一點點就會失敗」（c19 c24）的不加 mechanism（同以前的標法）。

### 紙杯：30/36，過階段 0 門檻

- 成功 30、`no_grasp` 6（c2 c6 c12 c17 c26 c36），6 集都有 `bad_aim` `[已查證：CSV]`。
- 近／遠：**22/24 ／ 8/12**（11/11 M-full 200k：20/24 ／ 9/12）。
- 跟 11/11 M-full 200k 同 36 點配對：只有這次成功 c21 c32、只有 11/11 成功 c2；McNemar p = 1.00。
- 跟 11/11 一樣，失敗偏 θ 負側：6 點全部 θ ≤ −24°（c2 −40°、c6 −24°、c12 −35°、c17 −64°、c26 −57°、c36 −47°）；近點 2（c17 c36）、遠點 4。

### 標註修正（11/18，Claude 依柏宇確認改）

- 寶特瓶 c11、c25、c35：outcome `no_grasp` → `dropped`、mechanism 拿掉 `dropped`（note 都是「抬起時掉落」；`dropped` 只是 outcome，同 11/17 c28）。c11 留 `bad_aim`。
- 寶特瓶 c33、c36：note「推倒杯子」→「推倒瓶子」（柏宇確認是筆誤）；mechanism 加 `pushed_away`（撞倒算 pushed_away，11/14 規則）。
- 紙杯 c6：`collison` → `collision`。
- 兩份 `operator` 填 `Boyu`。

## note 裡的兩類描述（只列，未分析）

- 「伸不夠遠」：寶特瓶 c11 c25 c38、紙杯 c12 c26。
- 「瞄超歪，可能沒在用第一視角」：紙杯 c17 c36、寶特瓶 c16。
- 柏宇：不確定拿這些點的 r 對照能得到什麼結論；要讓模型補到全對可能要跑 RL `[柏宇說 11/18]`。
- 柏宇：寶特瓶（這場失敗 7 點有 5 點是近點）和紙杯（6 點全在 θ 負側）的失敗**不是同一個原因** `[柏宇說 11/18]`：
  - 紙杯：夾取時感覺還是很依賴第三視角，或是那些位置本來就需要第三視角；轉腕都沒轉好（細節在 CSV note）`[柏宇說 11/18]`。
  - 寶特瓶：有些點瞄得很差，有些是前後距離沒抓準、差一點點；近點會一直撞倒物體 `[柏宇說 11/18]`。
    對照 CSV：近點失敗 5 點裡，撞倒是 c33 c36（`pushed_away`），c11 c25 c35 是抬起時掉落（`dropped`，c11 c25 note 寫伸不夠遠）。
    → 柏宇同意近點失敗有兩種：撞倒（c33 c36），以及夾太淺、抬起時掉落（c11 c25；跟「前後距離差一點點」同一類）`[柏宇說 11/18]`。
    c35 不屬於這兩種：夾瓶蓋有夾到，可能是搬運時手臂伸得太斜才掉 `[柏宇說 11/18]`（CSV note 原文「夾瓶蓋抬起時掉落」未改）。

## te −0.05 試跑（寶特瓶，未存到資料）

- config `configs/rollout_omx_aimfix_300k_te-005_plastic_bottle_eval.yaml`，record `__ba8fd66d`（coeff −0.05 `[已查證]`）。
- 柏宇只想試幾個點，跑了 2 個點；兩個資料夾都是 0 集、CSV 空 → **沒有影片也沒有判定**。
- 柏宇的現場印象：沒有比較好，前後（徑向）瞄不準，可能是模型的問題、不是推論的問題 `[柏宇說 11/18]`。2 點、無紀錄，不能當證據。
  離線可分辨的方法：`h19b_te_replay.py` 改成可傳 coeff，用 141713 的畫面比 −0.01 與 −0.05 閉爪前的指令（開迴路；目前 coeff 寫死 −0.01）——未做、未決定。

## 限制

- 每種物體一場、每點一次；跟前一場配對都分不出差別，「300k 比 200k 好」沒有證據支持，只能說這場過了門檻。
- 寶特瓶 ep 14 起 front-left 重新對齊過；紙杯是另一段時間跑的，相機沒再動。
- 寶特瓶用 v4 清單，紙杯用 c1–c36 原版；兩個分數的難度基準不同。

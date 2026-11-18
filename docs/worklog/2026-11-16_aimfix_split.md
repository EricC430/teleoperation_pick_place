# 2026-11-16 aimfix_A1_cut 重建：ep4／ep16「修過頭再修回」拆成兩集

參與：柏宇（看片定幀號、決定）、Claude（AI，工具、驗證、整理） ｜ 性質：資料處理（無實機）

接 `2026-11-15_aimfix_twoobj_recording.md` §剪點。那份的 13181 → 6080 幀、26 集是**舊版**，現行以這份為準。

## 為什麼拆

ep4、ep16 在 11/15 標註時就記了「先太右再太左（調了兩次）」。原本一集只剪一刀，留下來的是「修正 → 過頭 → 修回」一整段。
柏宇決定拆成兩集，兩集都是修正示範 `[柏宇決定]`：
- 第一集：原剪點 → 第一次修正到位（只有 0.5–0.7 s，沒有夾取）
- 第二集：過頭位置、開始修回之前 → 原結尾

第一集短而且沒有夾取，**不會被當成「停止」來學**：ACT 的 chunk 超過集尾的步數用 `action_is_pad` 遮掉，不算 loss（`lerobot/src/lerobot/policies/act/modeling_act.py:148`）`[已查證]`。代價是第一集每幀只有 ≤ 11／100 步有監督。

## 幀號（原始集幀號，15 fps）

| 原始集 | 第一集 | 第二集 | 依據 |
|---|---|---|---|
| ep4 | 254–264 → 新 ep4，11 幀 | 293–500 → 新 **ep26**，208 幀 | 柏宇看片。pan 在 265–291 只動 0.3，292–294 才轉回 `[已查證：action]` |
| ep16 | 266–273 → 新 ep16，8 幀 | 281–506 → 新 **ep27**，226 幀 | 柏宇看片。pan 279–280 停在 −3.2；柏宇判 281 比 279 偏更多 |

- 過程中 Claude 先依 pan 建議第二集起點 ep4＝274、ep16＝279，柏宇看片改成 293、281（ep4 的 274 之後還停了很久）。**定幀以影片為準**；pan 只代表左右，ep16 的 shoulder_lift／elbow 同時在動（前後可能也在調，未確認）。
- 中間沒用到的幀：ep4 265–292、ep16 274–280，沒有集從這裡起始。

## 做法與驗證

- `scripts/trim_recovery_prefix.py`：CSV 加 `end_frame`（含該幀，空白＝到結尾）；同一個 `episode_index` 的第二列變成新的一集，接在最後面。原 26 集編號不變；影片共用同一個 mp4，只改 from／to_timestamp，不重新編碼。
- 表：`episode_meta/omx_pick_place_pilot_plastic_bottle_aimfix_A1_cut.csv`（多 2 列，note 標 `[柏宇看片]`）。
- 結果：`.cache/lerobot/omx_pick_place_pilot_plastic_bottle_aimfix_A1_cut`，**28 集、6045 幀**（舊版本機改名 `…_cut_old26`）。
- 檢查 `[已查證]`：
  - 4 段的頭尾幀，action 和兩個鏡頭的像素都等於原始集的對應幀，差 0。
  - 其他 24 集在第 0 幀、中間幀、最後一幀都等於舊 `_cut`，差 0。
  - ep4 第 0 幀的 chunk 有 11／100 步沒被遮掉，最後一步就是該集最後的 action，不會跨到別集。
- 訓練 config `configs/train_omx_cup129_bottle135_aimfix28_twoobj10_augfl_fldrop.yaml` 的集數和預期值已更新：
  - merged 311 集（排除後 302）、111530 幀，twoobj 移到 301–310。
  - fl_dropout 的 `episodes_without_close` 會是 `[277, 289]`，就是這兩個第一集，屬於預期。aimfix 的 `frames_in_window` 仍是 780（本機重算，舊版同法得 780，跟 config 原值相同）。
  - 順便改了原 config 第 1、7 行的「300 eps」：merge 後其實是 309 集，排除後才是 300，舊寫法沒講清楚。

## 狀態

- 訓練還沒開始（柏宇說），所以沒有模型用過舊的 26 集版。
- Hub `ericc430/omx_pick_place_pilot_plastic_bottle_aimfix_A1_cut`（private）覆蓋成新版 `[柏宇決定：取代 _cut]`：
  - 新 commit `5991e4ac`。遠端檔案清單跟本機相同，每個檔大小一致，`meta/info.json` 是 28 集／6045 幀 `[已查證：Hub API]`。
  - 舊版仍在 Hub 歷史裡（commit `c9a68e91`），要回去用 `revision=c9a68e91`。repo 沒有 tag。
- 4090 上還沒下載過，照 config 的 `hf download` 拿到的就是新版。

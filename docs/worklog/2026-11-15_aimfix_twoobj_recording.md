# 2026-11-15 錄製：寶特瓶「先偏再修」26 集、兩物體試錄 10 集；aimfix 剪點

參與：柏宇（錄製、剪點判定）、Claude（AI，計畫、工具、整理） ｜ 性質：現場工作＋事後整理

計畫原本排 11/16，實際 11/15 下午錄完（config 註解寫的「錄製日 2026-11-16」沒改）。為什麼錄、點怎麼挑：config 檔頭（`configs/record_omx_bottle_aimfix_A1.yaml`、`configs/record_omx_twoobj_pilot_A1.yaml`）；擺放圖 `configs/placements/*_A1_20261116_plan.png`。
同日的閉環見 `2026-11-15_mfull_v4_closed_loop.md`。注意那份發現同模型兩場有一半的點結果翻盤，**挑 aimfix 點用的「11/11 的 8 個失敗點」是單場結果**。

## 錄了什麼

| | aimfix_A1 | twoobj_A1 |
|---|---|---|
| 資料 | `.cache/lerobot/omx_pick_place_pilot_plastic_bottle_aimfix_A1` | `.cache/lerobot/omx_pick_place_pilot_twoobj_A1` |
| 集數／幀數 | 26／13181 | 10／6526 |
| 開錄 | 14:44（`__25a6e899`），15:00 續錄（`__5d13aaa6`，`num_episodes=22` → 第一段存了 4 集） | 15:49（`__0fbf8b96`），一次錄完 |
| episode_time_s | **45**（兩段的 config record 都是 45；config 檔改成 75 是 15:00 之後，只有 twoobj 用到）。最長一集 40.4 s，沒有被時間切斷 | 75 |
| 平均長度 | 33.8 s | 43.5 s |
| 標註 | `episode_meta/omx_pick_place_pilot_plastic_bottle_aimfix_A1.csv`（偏的方向在 notes） | `episode_meta/omx_pick_place_pilot_twoobj_A1.csv`（目標、留桌上的物體、規則在 notes） |

- 校正 id 用 `2026-09-18_*`，跟 10-08 follower 逐位元相同（11/15 `cmp` 查證）。光線：宿舍大燈＋檯燈＋窗簾拉上。
- 點和偏的方向照 config 表，**沒有跳點** `[柏宇說]`；episode_meta 是照表填的，沒有逐集對影片。
- 花了多少時間：**沒記** `[柏宇說]`。
- 兩物體同 θ 的 4 集（ep6–9）：不會太擠，注意一下都可以夾起來 `[柏宇說]`。

## aimfix：「短」都先試夾了一次

- 夾爪 action 往下穿過中線的次數：**9 集「短」都是 2 次**（先在短的位置閉合、夾空、張開、往前伸、再夾），17 集左／右都是 1 次 `[已查證：關節資料]`。
  → 「短」實際上錄成「試夾失敗 → 修正」，比較接近 recovery；左／右是「停在偏的位置 → 直接修正」。
- 停頓 ≥ 1 秒：柏宇不確定、可能沒有 `[柏宇說]`。不需要，剪點用看的標 `[柏宇決定]`。

## 剪點

流程：`scripts/aimfix_cut_candidates.py` 抓候選 → `scripts/aimfix_cut_clips.py` 每集一支片（front-left＋手腕並排、幀號印在畫面上）→ 柏宇逐集看片改 `episode_meta/omx_pick_place_pilot_plastic_bottle_aimfix_A1_cut.csv`，**26 集都是 `manual`**。
（`scripts/aimfix_cut_sheet.py` 的拼圖試過：1–2 cm 的偏差在 front-left 縮圖上看不出來、手腕畫面被瓶子蓋滿，不採用。）

- **判準**：左／右 → 剪在手臂停在偏的位置、開始修正之前。**短 → 剪在試夾閉到最低點（夾空）之後** `[柏宇決定]`：
  閉合的動作本身不能留，否則等於教「伸不夠就閉爪」；留「已經夾空」的狀態，讓模型學鬆開→往前伸→再夾（同 `trim_recovery_prefix.py` 保留空夾狀態的理由）。
  柏宇最後看片定的短剪點，都在程式算的最低點之後 8–19 幀（0.5–1.3 s：ep0 +14、ep5 +19、ep6 +9、ep11 +10、ep12 +17、ep17 +11、ep18 +14、ep23 +8、ep24 +13）。
- 剪：`scripts/trim_recovery_prefix.py` 加 `--cuts <csv>`（照表剪，其他照舊）→ `.cache/lerobot/omx_pick_place_pilot_plastic_bottle_aimfix_A1_cut`，**13181 → 6080 幀**（留 46%）。
  檢查 `[已查證]`：ep0／5／12／25 剪後第 0 幀的 action 和兩個鏡頭的像素都跟原始的剪點幀相同（差 0）；26 集長度 = 原長 − 剪點。剪點記在 `meta/trim_recovery_prefix.json`。
- 柏宇在 CSV 留的意見：ep2「後面調整過多，不知道要不要留」；ep4、ep16 先太右再太左（調了兩次）。
  AI 的建議判準：最後往正確位置收斂就留（修過頭再修回來也是修正示範）；來回晃、有時離開目標就整集不用。ep2 未定。

## 未決

- aimfix_cut、twoobj 要不要進訓練、怎麼混（比例、跟 precise_A1 的關係）：未定。
- ep2 留不留。
- twoobj 規則只涵蓋 Δθ ≥ 20° 或同 θ（Δθ ≤ 5° 且 Δr ≥ 8 cm）；5°–20° 沒定義。推論時模型能不能照規則選，沒驗。

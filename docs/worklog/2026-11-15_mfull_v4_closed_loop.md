# 2026-11-15 M-full 200k 寶特瓶閉環，第一次用 v4 清單（c37–c40）

參與：柏宇（現場、判定、標註）、Claude（AI，整理數字） ｜ 性質：現場工作＋事後整理

接在 `2026-11-11_mfull_closed_loop.md`（同模型 c1–c36 原版 25/36）和 `2026-11-15_bottle_eval_list.md`（v4 清單）之後。

## 設定

| 項目 | 值 |
|---|---|
| 模型 | `ericc430/act_omx_paper-cup129-plastic-bottle135-augfl-fldrop30_200k`（`8c42b7e0652e`，`docs/models.md`） |
| 推論 | TE：`n_action_steps=1`、coeff −0.01；45 s／集 `[已查證：config record __d39a1ec1 的 cli overrides]` |
| config | `configs/rollout_omx_mfull_200k_te-001_plastic_bottle_eval.yaml`，YAML＋參數跟 11/11 試跑完全相同，所以沒有產生新 record（digest 相同，沿用 11/11 17:38 的 `__d39a1ec1`）。跟 11/11 正式場的差別：今天沒加 `--display_data=false` |
| 清單 | v4：`configs/placements/bottle_eval-close_20261115.csv`，範本 `plastic_bottle_A1_v4`（去 c6／c12／c22／c26，加 c37–c40） |
| 資料 | `D:\hf\lerobot\ericc430\rollout_omx_mfull_cup129_bottle135_augfl_fldrop30_200k_nas1_te-001_plastic_bottle_20261115_183543`（36 集、14257 幀），CSV 同名在 `episode_meta/`。Hub（private，11/18 上傳）：`ericc430/rollout_omx_mfull_augfl_fldrop30_200k_nas1_te-001_plastic_bottle_20261115_183543`，commit `79103629`、tag `v3.0`；原名 97 字元超過 Hub 上限 96，拿掉 `cup129_bottle135_`，本機名不變 `[柏宇決定 11/18]` |
| 環境 | 宿舍大燈＋檯燈＋窗簾拉上、木質桌面 |

## 結果：23/36，階段 0（≥ 29/36）沒過

- 成功 23、`no_grasp` 10、`dropped` 3。
- c37–c40：c37、c38、c40 成功，c39 失敗。
- 近／遠（r < 34 ／ ≥ 34）：**15/24 ／ 8/12**。11/11 是遠點失敗 7/11，今天遠點反而比近點好。

| 點 | r | outcome | mechanism | notes `[柏宇]` |
|---|---|---|---|---|
| c5 | 36.5 | no_grasp | bad_aim; pushed_away | 蠻偏的 |
| c7 | 32.3 | no_grasp | pushed_away | |
| c14 | 22.8 | no_grasp | bad_aim; pushed_away | 伸太近 |
| c16 | 35.1 | no_grasp | pushed_away; timeout | 推太遠夾不到 |
| c17 | 25.4 | no_grasp | bad_aim; pushed_away | 夾瓶身會碰到桌子所以應該夾瓶蓋，但夾到斜面夾不起來（不上不下） |
| c20 | 29.4 | no_grasp | bad_aim; pushed_away | 伸太近，夾取時把物體推遠了 |
| c24 | 26.5 | no_grasp | pushed_away | 推倒物體 |
| c28 | 37.2 | no_grasp | bad_aim; pushed_away | 伸太近，夾取時把物體夾倒了；由下往上數第三個關節一直不伸直 |
| c32 | 21.2 | no_grasp | bad_aim; pushed_away | 夾的地方有點上面，加上手腕有點斜，一邊抓到斜面就滑掉 |
| c33 | 13.0 | dropped | — | 夾瓶蓋，搬運途中掉落；推測夾爪朝向過於斜上方 |
| c35 | 16.6 | dropped | — | 搬運中脫落；瓶蓋靠近夾爪前端而非掌心，或夾爪過於斜上方 |
| c36 | 17.7 | dropped | — | 夾瓶蓋，搬運途中掉落；推測夾爪朝向過於斜上方 |
| c39 | 37.6 | no_grasp | bad_aim; pushed_away | 伸太近，夾取時把物體推倒了 |

- `dropped` 的三集是 11/15 依 `docs/experiment_spec.md:75`（抓起來了但中途掉落）補的 outcome；`eval/README.md` 的 mechanism 沒有一個符合「夾瓶蓋、搬運中滑落」，所以留空。
- 「伸太近」＝伸不夠（停在瓶子靠手臂這側），跟 11/11 的「伸不夠」同一件事 `[柏宇說 11/15]`。出現 4 次（c14 22.8、c20 29.4、c28 37.2、c39 37.6），不只遠點；量值未量。

## 跟 11/11 同模型比：同一點一半翻盤

兩場共有 32 點（v4 去掉 c6／c12／c22／c26、11/11 沒有 c37–c40）：11/11 24/32、今天 20/32。

- 只有今天成功：c1 c3 c8 c10 c18 c23（6 點；其中 c1 c3 c8 c10 c23 是 11/11 的 bad_aim 失敗點）
- 只有 11/11 成功：c7 c14 c16 c17 c20 c24 c32 c33 c35 c36（10 點）
- 不一致 16/32；McNemar 精確檢定 6 vs 10，p ≈ 0.45 `[已計算]` → 兩場分數差看不出來。
- `[推論]`：同模型同設定、單點結果一半會翻，**單場 36 集裡「某點成功／失敗」不能當成那個點的性質**。11/11 用來挑 aimfix 錄製點的「8 個失敗點」也是單場結果。
- 分數不能直接跟 11/11 的 25/36 比：清單不同、v4 比較容易（`2026-11-15_bottle_eval_list.md`）。

## 100k：試了 5 集，整場不算 `[柏宇決定]`

- `…fldrop30_100k_nas1_te-001_plastic_bottle_20261115_192339`（config `configs/rollout_omx_mfull_100k_te-001_plastic_bottle_eval.yaml`，11/15 新增，跟 200k 那份只差 policy.path、repo_id、註解）。c1–c5：c1 c2 c4 成功，c3 c5 未標。
- 柏宇：看到其中一集差距太大就放棄 `[柏宇說]`，哪一集、差在哪裡沒記。資料留在硬碟，不進任何統計。

## 夾瓶蓋：只出現在近點

- 柏宇：只有近處的才會需要夾瓶蓋 `[柏宇說]`。對照資料：notes 有寫瓶蓋的，今天 c33 13.0、c35 16.6、c36 17.7、c17 25.4（失敗）；10/22 b136 是 c33、c35、c36（`2026-10-22.md:124`）。
- **FK 指尖高度分不出夾瓶蓋還是瓶身** `[已查證]`：閉爪那幀的 FK 指尖離桌高度，夾瓶蓋的集 17.7–20.1 cm，遠點夾瓶身的也有 19–20 cm（`scripts/measure_pan_circle.py` 的 `fk_tip_above_table_cm`，riser 0.146）。`[推論]` 近點夾爪從上往下、遠點接近水平，指尖高度不等於夾到的位置。這個方法不採用。
- 示範集沒有記夾哪裡（episode_meta notes 都沒有「瓶蓋」）。只能先數近點：r < 20 的示範 normal_A1 8／16、recovery_A1_tight 6／12、precise_A1 0／37；`60_plastic_bottle` 沒有 episode_meta，擺位不明。這 14 集實際有幾集夾瓶蓋**未確認**，要看影片。

## 未完成

- 示範集夾瓶蓋的集數：看 r < 20 那 14 集（＋60 集）的手腕畫面。
- `dropped` 三集要不要加 mechanism（`other`）：未定。

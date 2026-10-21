# 2026-10-07 工作紀錄：紙杯 100k 閉環 temporal ensembling（19/32）

性質：**事後補寫**（2026-10-13 由 Claude 依 `episode_meta` CSV 整理），非當天紀錄 ｜ 標註者：Eric（CSV `operator`）
模型 hash ↔ run／step 見 `docs/models.md`。本檔只寫 hash。

> 這場的計畫寫在 `2026-10-07.md` 下一步清單，結果只在 `2026-10-08.md` §2 表格裡出現一行「19/32」，沒有分析。
> 本檔只讀 CSV，**沒有讀 rollout 資料集**，所以計畫裡要看的「chunk 交接跳變是否消失」和「每集幀數是否 ≥~280」都還沒做。

## 0. 結論

1. **19/32（59%）**，作廢 c7、c16、c17、c36（與 10-06 相同的四個）。
2. 跟 10-06 N=30（同一個模型 `2bdff2b3871e`）在同樣 32 個位置配對：**17 → 19**。只有 10-06 成功 3 個（c8、c10、c24），只有這場成功 5 個（c5、c14、c30、c32、c35），McNemar exact p=0.73。**差異不顯著。**
3. 失敗機制只有 `bad_aim`（13/13），10-06 的 `pushed_away` 10、`chunk_gap_rush` 6 在這場都是 0。但**兩場標註者不同**（10-06 是柏宇，這場是 Eric），機制分布的差異不能直接歸因到 ensembling。
4. 遠距離仍最差：2/11（10-06 是 3/11）。

## 1. 條件

計畫中的條件（`2026-10-07.md`，`[柏宇決定]` 跑原生版）：

| 項目 | 計畫值 |
|---|---|
| 模型 | `2bdff2b3871e`（紙杯 100k，與 10-06 相同） |
| 推論 | `--policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01`（每步推論，偏重**較舊**的預測） |
| 其他 | 同 10-06：相機 A1、路由器在、手腕不遮、`--goto-home --home-episode 10`、c1–c36 |
| 資料集 | `ericc430/rollout_omx_b1_uvc60_100k_te001_A1_paper_cup` |
| 事前量測 | 推論中位 32 ms、最慢 61 ms（15 Hz 預算 66.7 ms） |

**實際跑的條件 `[未確認]`**：這天的 config record 不在 repo 裡（`config_records/` 沒有 10-07 的 rollout），所以無法確認實際用的參數與每集秒數。上表只是計畫。

## 2. 成功率與位置 `[產出物]`

分區同 `2026-10-08.md` §2（看結果前固定）。

| r（cm） | 10-06 N=30 | 10-07 ensemble | 10-08 recovery N=100（參考，不同模型） |
|---|---|---|---|
| 近 <22 | 1/5 | 3/5 | 4/6 |
| 中 22–33 | 13/16 | 14/16 | 13/17 |
| 遠 ≥33 | 3/11 | 2/11 | 5/11 |
| 合計 | 17/32 | **19/32** | 22/34 |

每格只有 5–16 個點，只能當描述。

## 3. 失敗機制 `[產出物]`

| mechanism | 10-06 N=30（柏宇標） | 10-07 ensemble（Eric 標） |
|---|---|---|
| `bad_aim` | 15 | 13 |
| `pushed_away` | 10 | 0 |
| `chunk_gap_rush` | 6 | 0 |

`chunk_gap_rush` 為 0 在機制上說得通：N=1 沒有 chunk 交接，ensembling 每步都在混合多個 chunk `[AI推論]`。
`pushed_away` 從 10 變 0 比較可疑：可能是 ensembling 讓接近動作變慢變穩，也可能只是 Eric 只標了主因、沒標並列的機制。兩種解釋要分開，得回去看影片 `[未確認]`。

## 4. 待確認

1. **作廢理由是不是從 10-06 複製來的？** 這場 c7、c16、c17、c36 的 `notes` 跟 10-06 那份 CSV 一字不差（「AI 看影片判讀」）。`2026-10-08.md` §6 已經發現 10-08 的 c16 又是從這份複製的。這場的 frame 0 有沒有實際看過 `[未確認]`。
2. **計畫的兩個檢查還沒做**：每集是否跑滿約 280 幀（判斷每步推論有沒有拖慢控制頻率），以及交接跳變是否消失。兩者都要讀 rollout 資料集（本機 `D:\hf\lerobot\…`，不在 repo）。
3. 想知道 ensembling 本身有沒有用，最乾淨的對照是**同一位標註者**重看 10-06 與這場的失敗影片，只標 `pushed_away` 有沒有發生。

## 重跑

```
python scripts/compare_closed_loop.py episode_meta/rollout_omx_b1_uvc60_100k_nas30_A1_paper_cup_20261006_close_loop.csv episode_meta/rollout_omx_b1_uvc60_100k_A1_temperal_ensemble_paper_cup_20261007_close_loop.csv
```

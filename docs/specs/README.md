# 腳本規格（給 Claude Code 執行用）

**建立 2026-08-30。** 每一份是一支腳本的規格，設計成**可以單獨開一個 Claude Code session、只給它那一份**。

| # | 檔案 | 腳本 | 需要手臂嗎 | 擋在誰後面 |
|---|---|---|---|---|
| S1 | `S1_reach_logger.md` | `scripts/reach_logger.py` | 🔴 **要**（lab day） | 無。**產生 S2／S3 的輸入**（`reach_summary_<date>.json`）。🔴 先決：`placo` 裝進釘選環境＋取得 `omx_f` URDF（D026） |
| S2 | `S2_placement_sampler.md` | `scripts/sample_placements.py` | ❌ 不用 | 邏輯可先寫、可測；**凍結清單要等 S1 的數字**（`--from-summary` 讀 S1 產出，或人工填 `--r-*`/`--theta-*`） |
| S3 | `S3_placement_mat.md` | `scripts/make_placement_mat.py` | ❌ 不用 | 要 S2 的輸出才印正式版；**且要 S1 的 `reference` 樣本**（`azimuth_frame` 必須是 `mat`，見 S3 §2-4） |
| S4 | `S4_sim_teleop_collect.md` | `scripts/sim_teleop_collect.py` | 🟡 要 leader 臂，不要 follower | ✅ **2026-09-03 解除封鎖（D029）**：模擬器＝Isaac Sim、D025 前提 3 撤銷、leader 直插實驗室 4090 Linux 筆電。**先做 §2-2 的 USD 稽核，再做 §4 管線** |
| S5 | `S5_sim_replay_augmentation.md` | `scripts/sim_replay_augment.py` | ❌ 不用（重播既有真實錄製，非即時） | 🟡 **2026-09-15 提議中，尚未裁決**（`[AI提議]`）。**2026-09-18 四個缺口的準備工具已建（§11），模擬端腳本尚未在 4090 上跑過。**技術上無硬阻塞——§2 的四個缺口沒有一個卡住開工，只有缺口 4（相機幾何）需要 lab day 且可延後/外包 |
| S6 | `S6_joint_zero_calibration.md` | `scripts/measure_joint_zeros.py` | 🔴 **要**（手臂通電可讀關節值；不需相機／模擬機／GPU） | 無前置，**且擋住 S5 的抓取判定與 S4 的映射**。兩姿態法解「讀數→弧度」的 SCALE＋OFFSET；求解那半不需硬體，任何 Python 環境可跑（`selftest` 子指令可離線驗算術） |

## 共同規則（四份都適用）

1. **先讀 `CLAUDE.md`，再讀本規格。** 專案的來源標記制度與失敗模式在那裡。
2. **不要改 `docs/decisions.md` 的既有條目。** 腳本實作若發現決策有問題，寫進 PR/commit message 或新開一條 D0NN，不要就地改寫別人的決議。
3. **凍結性資料只寫一次。** 抽樣清單、校正檔、量測值，檔名帶日期，**永不覆寫**（`docs/conventions.md`）。
4. **`uv run` 是本專案的執行方式**，不要用裸 `python`。
5. **每支腳本都要能 `--help`**，且**乾跑（`--dry-run`）不碰硬體**。
6. **不要為了通過而放寬驗收條件。** 做不到就在輸出裡說做不到。

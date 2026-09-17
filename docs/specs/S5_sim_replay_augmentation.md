# S5 — Sim trajectory replay + domain randomization（用真實示範擴增資料）

**目標檔案：`scripts/sim_replay_augment.py`**
**正本依據：`docs/decisions.md` D025／D029／D030、`docs/specs/S4_sim_teleop_collect.md`、`sim/README.md`**

---

## 0. 狀態

🟡 **提議中，尚未經 Eric 裁決是否要做。** `[AI提議]`

本規格由 Claude Code 於 **2026-09-15 對話中**整理產生，起因是 Eric 提出「有真實錄製的 OMX 軌跡＋URDF，能否匯入模擬、換隨機背景、重播拍攝以擴增資料」。**這不是 D029 已定案的東西**——D029 定案的是「leader 臂即時遙操作 sim 裡的 follower、蒐集全新 demo」；本規格是「重播**已經錄好**的真實 action 序列＋domain randomization」，兩者共用同一套資產／場景管線，但輸入來源與即時性需求不同（見 §3）。

**在往下動工前，這份文件本身就是要 Eric 裁決的東西**：是否要做、做到哪個版本、要不要正式開一條 `docs/decisions.md` 的新條目（見 §9）。

---

## 1. 目標

用 **LeRobot dataset 裡已經錄好的 `action` 欄位**（真實 OMX leader→follower 的關節目標序列）驅動 Isaac Lab 裡的 OMX articulation，逐幀渲染相機畫面，每個 episode 套用不同的 domain randomization（光照、色溫、HDRI、相機微擾動、物體外觀）產出多份「同一動作、不同視覺環境」的資料。

**它買到的東西：**
- 動作標籤（action label）完全來自真實錄製，**不需要重新示範**——這是跟 D029/S4 teleop-in-sim 最大的差異：那條路線每次要新資料都要人到場即時操作，這條路線只要蒐集階段做過一次，之後可以無限次換皮膚重播
- 視覺變因掃描（背景／光照／相機噪聲）幾乎免費，且**不佔用寶貴的 lab day**（見 §2 的依賴分析——四個缺口裡只有一個真的需要人到實驗室）
- 沿用 D029/S4 已經打通的資產（`sim/convert_omx_urdf.py` 轉出的 USD、`sim/omx_scene_cfg.py` 的場景）

**它買不到的東西（不要在計畫書裡誇大，同 S4 §1 的立場）：**
- 不是「更多真實示範」。動作分佈完全被原始錄製集合的涵蓋範圍鎖死——原始軌跡沒走過的姿態，這條管線也生不出來
- 不能取代 D025 §5-5 講的「外觀對齊做不到」——domain randomization 是刻意製造分佈覆蓋，不是逼近真實相機的成像分佈
- 渲染出來的物體接觸行為（抓取瞬間物體怎麼動）**不是物理模擬出來的真實抓取**，除非做了 §2 缺口 3 的完整物理路線（目前不建議，理由見下）

---

## 2. 🔴 四個缺口的依賴分析（Eric 2026-09-15 提問的直接回答）

這是這份規格存在的主因：先前的技術可行性回覆列了 4 個未關閉的缺口，這裡逐一回答「人工要多久、是不是硬依賴、能不能先做」。

**時間估計已依 `歸因力訓練計畫.md` §10 乘 2–3 倍**——這是 AI 的推論值，不是量出來的，**請 Eric 實際跑過之後回報真實耗時，用來校正未來的估計。**

| # | 缺口 | 需要人到實驗室？ | 人工工時估計（已乘 2–3 倍）`[AI推論]` | 是不是後續 pipeline 的硬依賴 | 可以先做嗎 |
|---|---|---|---|---|---|
| 2 | mimic 齒輪比正負號未驗證 | ❌ 不用（純 sim，`--mimic-gearing` 旗標已存在於 `convert_omx_urdf.py`） | **0.5–1 天以內**（多半 1–2 小時，抓緩衝到半天） | 🟡 只影響**夾爪開合的視覺正確性**，不影響手臂本體姿態重播 | ✅ **最先做**——最便宜，且直接驗證既有資產能不能信 |
| 1 | drive gains（stiffness/damping）未校正 | ❌ 不用（用已錄好的 `action`＋`observation.state` 配對回歸，不需要即時硬體） | **2–4 個 sim session**（寫回歸/比對腳本 0.5–1 天＋跑優化與檢視結果 0.5–1 天，可能要來回 1–2 輪） | 🔴 **是**——這決定重播出來的手臂動作視覺上「跟不跟得上」錄製軌跡；`sim/README.md` 已記錄暫定增益連自身重量都撐不住（全零姿態 1 秒漂移 19–35°） | ✅ 可以跟缺口 2、3 平行做；跟 DR 佈線（`EventTerm`）完全獨立，可以同時進行 |
| 3 | 物體 attach/detach 設計（抓取時物體怎麼跟手臂走） | ❌ 不用（純設計＋程式決定） | **決策本身 ~1 小時討論；腳本化實作 0.5–1 天** | 🔴 **是，但只擋「含抓取動作的 episode」**——純接近／純放置片段（物體位置只由 placement CSV 決定、手還沒碰到物體）不受影響，可以先出資料 | ✅ 可以先做決策＋先寫「不含抓取判定」的簡化版重播，之後再補 |
| 4 | 相機幾何對齊（T1 內參／T2 外參，S4 §5-5） | 🔴 **要**（需要實體 D405/D455/UVC 相機＋座標墊） | T1（內參讀取）**0.5–1 小時**；T2（ArUco 外參量測）**1 天內**（含架設、拍攝、算 pose，可能要來回校正） | 🟡 **只有在「要求 sim 相機視角跟某一次真實錄製幾何對得上」時才是硬依賴**——如果目的只是背景/光照多樣化、不要求跟特定歷史錄製像素對齊，可以先用 `scene_constants.py` 現有的 PLACEHOLDER 幾何頂著 | ✅ 可以完全延後，且可以**外包**（純相機量測＋ArUco pose 估計，不需要碰模擬程式，柏宇或任何人都能做） |

### 綜合結論

- **唯一真正卡住「開始動工」的東西：沒有一個。** 缺口 2、1、3 全部不需要 lab day，缺口 4 需要，但缺口 4 只在要求幾何精確對齊時才是硬依賴，而且可以整個延後或外包。
- **建議順序（可以立刻開始，不用等任何前提）：**
  1. 缺口 2（mimic 驗證）——半天內出結果，驗證資產可信度
  2. 平行開始：缺口 1（gain 回歸腳本）＋ 缺口 3（attach/detach 決策與簡化版實作）＋ §4 的重播骨架本身
  3. DR（`EventTerm` 佈線）——跟 1、3 完全獨立，隨時可以插入做
  4. 缺口 4（相機量測）——找 lab day 或請人代勞，不卡前面任何進度；沒做完之前，§7 驗收條件裡標「幾何未對齊」，跟 S4 §5-5 同一招，不要假裝對齊過
- **這代表「先建 pipeline、等結果出來再擴增」的判準（見上一輪技術回覆）現在可以更精確：** pipeline 本身現在就能開工，**卡的不是缺口，是要不要投入這些工時**——這是排序問題，不是可行性問題。

---

## 3. 跟 D029/S4 的關係——這不是同一件事

| | D029/S4（已定案，teleop-in-sim） | S5（本規格，提議中） |
|---|---|---|
| 動作來源 | leader 臂即時操作 | **已錄好的真實 `action` 欄位** |
| 即時性需求 | 🔴 有——D029 §3 整條決策在解「人在迴路延遲」問題 | ❌ 沒有——逐幀讀取，sim 步進速度不受真人操作節奏限制 |
| 需要 leader 硬體在場 | ✅ 要 | ❌ 不要 |
| 需要人到實驗室 | ✅ 要（leader 必須插在 5090 Linux 筆電） | 🟡 只有缺口 4（相機量測）要，其餘不要（前提：那台機器能遠端存取，`[未確認]`——見 §9） |
| 產出的是 | 全新的 sim demo | **既有 demo 的視覺變體**，動作標籤不變 |
| 買到的變因掃描 | 有，但每次要重蒐集 | 有，而且**同一組動作可以重複套用無限組 DR 參數** |

兩者共用：`sim/convert_omx_urdf.py`、`sim/omx_scene_cfg.py`、`assets/trash_obj/`、D025 的「sim 資料不得與實機資料混用、要有 `sim_` 前綴」規則（§7）。

---

## 4. 架構

概念上等同既有的 `lerobot-replay`（`configs/replay_omx.yaml`——把某一集的 action 原樣送給**真實** follower，開環、不看相機），但把接收端從真實 follower 換成 Isaac Lab 的 articulation，並在每個 episode 加一層 DR：

```
既有 LeRobot dataset（真實錄製，例如 EricC430/omx_pick_place_pilot_uvc_60）
   │  讀 episode 的 action 欄位（逐幀關節目標，6 個馬達）＋ observation.state（用於缺口 1 的回歸）
   ▼
關節目標映射（沿用 S4 §5 item 1 的單位／方向對照表，這裡不必重新推導——同一份映射）
   ▼
每個 episode 開始前：套用一組 domain randomization 參數
   （`EventTerm`：dome light exposure、色溫、HDRI、相機微擾動、物體外觀——沿用 NVIDIA SO-101 教材的做法，S4 §3 已指名）
   ▼
Isaac Lab ArticulationCfg 的模擬 follower（位置控制；drive gains 見 §2 缺口 1）
   │  物體位置：沿用原始 episode 的 placement_id（同一顆物體、同一個起始位置，只有畫面環境變了）
   │  抓取判定：缺口 3 的 attach/detach 邏輯（依 observation.state 的 gripper 讀數判斷開合時機）
   ▼
sim.step() → 渲染相機（wrist / front-left，內參對齊見 §2 缺口 4）→ 組成 frame
   ▼
LeRobotDataset.add_frame(...) → save_episode()　（repo_id 帶 sim_ 前綴＋來源 episode id＋DR seed，見 §7）
```

**跟 S4 §4 的差異只在最上面一段**（輸入來源）跟中間插入的 DR 層；下半段（articulation → 渲染 → 存檔）完全共用。

---

## 5. 最容易錯的地方

1. **每個 DR 變體要記錄「來自哪一集真實錄製＋哪一組 DR 參數」**，不然沒辦法追溯，也沒辦法在缺口 1 的回歸出問題時定位是哪個環節錯的。`meta` 裡至少要有 `source_episode_id` 和 `dr_seed`。
2. **不要把「重播出來看起來合理」當成驗收標準。** 同 `sim/README.md` 的教訓——沒校正的增益會讓手臂用力下垂，畫面可能還是「看起來像在動」，但跟真實軌跡的時序/姿態已經對不上。§7 要有量化比對（每幀關節角誤差），不能只用肉眼看。
3. **抓取片段沒做缺口 3 的判定，物體會穿模或凍結在半空。** 如果先出「不含抓取」的簡化版（§2 建議順序），要在資料的 `meta` 裡明確標注這批只涵蓋 approach/retreat 階段，不要讓下游誤用成完整 episode。
4. **DR 參數的隨機範圍要跟 S4 §3 指名的 NVIDIA 教材數值一致**（dome light exposure −4.0~3.0、色溫 2500–9500 K、相機位移 ±0.02 m／±0.05 rad），不要自己憑感覺設一組新的——那是已經有人驗證過合理範圍的起點。
5. **同一份 URDF 被兩邊用**（S1 的 FK 跟這裡的 USD 轉換），S4 §2-3 已經提過：任何一邊改了 `omx_f.urdf`，另一邊要同步檢查。

---

## 6. 輸入（草案）

```
--source-dataset  EricC430/omx_pick_place_pilot_uvc_60   # 真實錄製資料集
--source-episode  0                                       # 或 --all-episodes
--dr-preset       nvidia-so101-default                    # §5 item 4 的參數組
--dr-seed         42
--repo-id         sim_omx_replay_dr_<date>                # 沒有 sim_ 前綴直接拒絕，同 S4 §7
--skip-grasp-frames                                        # 缺口 3 未關閉前的降級模式，見 §2 建議順序
--dry-run
```

---

## 7. 驗收條件

- [ ] 缺口 2 的五姿態對照表（沿用 S4 §5-1 做法）印出來，夾爪開合方向確認正確
- [ ] 缺口 1 的回歸：拿一段真實 `(action, observation.state)` 序列，比較 sim 重播出的關節軌跡與真實 `observation.state` 的逐幀誤差，印出誤差分佈（p50/p95/max），不是只看有沒有跑起來
- [ ] 每個輸出 episode 的 `meta` 記錄 `source_episode_id`、`dr_seed`、`dr_preset`、是否含抓取判定（缺口 3 是否已套用）
- [ ] `repo_id` 沒有 `sim_` 前綴時直接拒絕，同 S4 §7
- [ ] 若缺口 4（相機幾何）未完成，`meta` 明確標注「幾何未對齊，PLACEHOLDER 相機」，不得暗示已對齊——同 S4 §5-5 的立場
- [ ] `verify_dataset.py` 對輸出跑過，退出碼 0，且 `features` 鍵序與來源真實資料集一致

## 8. 明確不做（第一版）

- ❌ 不做完整物理接觸模擬的抓取（§2 缺口 3 的物理路線）——先用 scripted attach/detach，理由見 §2
- ❌ 不做「sim 相機視角精確重現某一次真實錄製」以外的幾何宣稱，除非缺口 4 已完成
- ❌ 不把這批資料拿去訓練或在書審/報告裡宣稱有效，直到 §9 的 D025 閘門條件明確被裁決為已滿足
- ❌ 不擴大到 D030/campA_136sym 之外的 placement 集合——先用已經有真實錄製對應的那些點

---

## 9. 🔴 與 D025 閘門的關係——2026-09-15 新資訊，尚未正式記錄

`[Eric說，2026-09-15 對話中]`：ACT 已針對 Phase B 第一個實驗訓練完成，closed-loop 評估因時間限制只跑了 10 次，成功 5 次。

**這件事跟本規格的關係：**

- D025 執行前提 1（收緊後的版本）是：**「用模擬資料訓練、或在書審／報告裡宣稱模擬有效」的門檻是要先有 Phase B 實機基準成功率。** 這個前提現在**看起來有東西可以滿足了**——但有兩點要先確認，不是我能替 Eric 判斷的：
  1. **這次評估算不算 `phase_plan.md` 定義的 B2**（「訓練 B1 模型，評估相同場景成功率」）？`[未確認]`
  2. **n=10、非 30**——`eval/README.md` D016 已定案的協定是每個物體 30 次評估。10 次的 95% 信賴區間非常寬（Wilson interval 約 24%–76%），**這個 50% 數字目前统計上不夠穩，拿它當「基準」要附上這個警語，不要單獨引用「50%」三個字。**
- 🔴 **這是一筆新的專案事實，不是決策，目前只存在於這次對話裡。** 依 `CLAUDE.md`「A 線的新決策一律寫進 `docs/decisions.md`，不要只寫在 commit message」的精神，**建議 Eric 把這個 Phase B 評估結果正式記錄**（`eval/` 底下的 CSV，或 `docs/phase_plan.md` 的進度欄）——這不是我要不要做的問題，是 Eric 要不要做／要記在哪的問題，**我不會自己去改 `docs/decisions.md` 既有條目**（`docs/specs/README.md` 共同規則第 2 條）。

**如果 Eric 確認上面兩點、且要正式解除 D025 前提 1**，那應該是一條新的 `docs/decisions.md` 條目（例如 D031），而不是回頭改 D025——同一條規則。

---

## 10. Cross-reference

`docs/decisions.md` D021, D025, D029, D030, `docs/specs/S4_sim_teleop_collect.md`, `sim/README.md`, `eval/README.md`, `docs/phase_plan.md`（Phase B 定義）, `configs/replay_omx.yaml`（既有真實→真實 replay 的對照範本）。

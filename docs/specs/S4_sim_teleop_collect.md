# S4 — 模擬環境 ＋ 真實 leader 臂遙操作蒐集

**目標檔案：`scripts/sim_teleop_collect.py`**

---

## 0. 🔴 先看這裡：這一項與已定案的排序有衝突

**`docs/decisions.md` D025 的執行前提第 3 條 `[Eric決定]` 2026-08-27：**

> **「不佔用 9/26 前的任何工時。這是 Phase B 之後的工作。」**

而 D025 的前提 1 是**「必須先有 Phase B 的實機基準成功率，否則模擬資料無從校準」**。

**現在（2026-08-30）Phase B 一筆都還沒錄，A6 還被 D023 的傳輸線擋著。**

→ **寫規格不等於執行。這份規格本身不違反 D025**，但**動手寫這支腳本會。**
→ **要動手前，請先明確裁決：是要修改 D025 的排序，還是把這支腳本排到 Phase B 之後。**
   **不要靜靜地開始寫——那正是 D025 想避免的事。**

### 0-1 目前的範圍（`[Eric決定]` 2026-08-31）

**現在只做「最基礎的環境與前置準備」，不寫 §4 那條 leader→sim→dataset 管線。**

具體 = **只做 §2 的四件查證** ＋ 把兩個 repo clone 下來 ＋ 確認模擬器能開起一個空場景。
理由：這些是純調查／安裝，**不需要 Phase B 資料、不佔學習工時**，因此不違反 D025 前提 3；
而且其中的 URDF 查證與 D026（S1 用 `omx_f` URDF 做 FK）共用，本來就要做。

**明確不在現在範圍**：§4 架構、§5 的單位/時序/feature key 對齊、§7 的錄製驗收 ——
那些是「動手寫腳本」，要等 D025 排序裁決或 Phase B 之後。

---

## 1. 目標

用**真實的 OMX leader 臂**遙操作**模擬環境裡的 follower**，錄出 LeRobot 格式的資料集。

**它買到的東西（也是唯一值得做的理由）：**
- lab day 之外也能蒐集資料 —— 而 lab day 是本專案最稀缺的資源（`phase_plan.md` §T0）
- 變因掃描（光線、背景、物體位置）在模擬裡幾乎免費
- **不需要 follower 臂**，所以不受 D023 傳輸線阻塞

**它買不到的東西（不要在計畫書裡誇大）：**
- 不能取代實機基準。D007 反對的是「用模擬取代實機」，D025 允許的是「用實機錨定模擬」
- 渲染分佈差距（光照、材質、感測器雜訊、動態模糊）不會因為相機規格對上就消失

## 2. 🔴 動手前必須先查清楚的四件事（每件約 30 分鐘，不查就開工是賭博）

| # | 問題 | 怎麼查 | 為什麼擋在前面 |
|---|---|---|---|
| 1 | `omx_f` URDF 的 **inertial / collision 是不是佔位值** | clone `ROBOTIS-GIT/open_manipulator`，看 `open_manipulator_description/urdf/omx_f` | 很多 ROS URDF 的 inertia 是 1e-3 佔位值，物理會完全不對 |
| 2 | DYNAMIXEL 的 **effort / velocity 限制有沒有寫進 URDF** | 同上 | 沒有就要從規格書補，否則模擬臂的動態與實機無關 |
| 3 | **`robotis_mujoco_menagerie` 裡有沒有現成的 OMX 模型** | 看該 repo 的樹 | 🔴 **見 §3 —— 這一條可能整個改變技術選型** |
| 4 | `cyclo_lab` 的環境定義能不能換 robot asset | clone 看 | D025 已查證 cyclo_lab 支援的是 **OMY 不是 OMX** |

⚠️ **OMX ≠ OMY ≠ OpenMANIPULATOR-X。** 本專案已經在這個坑裡跌過（`CLAUDE.md` 失敗模式一）。

## 3. 🟡 技術選型有一個未裁決的分歧

**D025 `[Eric決定]` 寫的是 Isaac Sim。**

**但 D025 自己的查證段落記著：`ROBOTIS-GIT/open_manipulator` 把模擬使用者導向
`robotis_mujoco_menagerie` —— MuJoCo，不是 Isaac Sim。**

| | Isaac Sim / Isaac Lab | MuJoCo |
|---|---|---|
| 現成 OMX 資產 | ❌ 沒查到，要自己 URDF→USD 匯入 | 🟡 **可能有（§2 第 3 項要查）** |
| 上手成本 | 高（安裝、場景、teleop 管線都要接） | 低（`pip install mujoco`，一個 XML 一個 python 迴圈） |
| 渲染品質 / domain randomization | 強 | 弱一些 |
| 有沒有對口教材 | ✅ NVIDIA 的 SO-101 sim2real 教學（任務不同、流程同構） | 一般 |

**`[AI提議]`，待裁決：先用 MuJoCo 做一個 2–3 天能跑起來的版本驗證整條管線，
Isaac Sim 留到需要 domain randomization 品質時再上。**
**理由：這支腳本的第一個風險不是渲染品質，是「leader → sim → LeRobot dataset」這條管線通不通。
用便宜的模擬器驗證便宜的問題。**
⚠️ **但這與 D025 已寫的 Isaac Sim 不一致，要先裁決，不要自己選一邊。**

## 4. 架構

```
真實 leader 臂 (OmxLeader, COM6)
   │  get_action() -> {"shoulder_pan.pos": ..., ... , "gripper.pos": ...}   # 6 個關節
   ▼
關節目標映射（單位、方向、行程對齊）      ← 🔴 最容易錯的一段，見 §5
   ▼
模擬 follower（位置控制 / PD）
   ▼
sim.step()  → 渲染相機 → 組成 frame
   ▼
LeRobotDataset.add_frame(...) → save_episode()
```

**已查證的 API 事實（`lerobot/` commit `a16f34c0`，2026-08-30 讀原始碼）：**
- `OmxLeader.get_action()` 回 `dict[str, float]`，key 是 `<motor>.pos`
- 六個馬達名稱固定：`shoulder_pan` / `shoulder_lift` / `elbow_flex` / `wrist_flex` / `wrist_roll` / `gripper`
- `gripper` 的正規化模式是 `RANGE_0_100`，**其餘是 degrees 或 RANGE_M100_100，取決於 `use_degrees`**
  （`robots/omx_follower/omx_follower.py:50-62`）—— **兩者不同，映射時不要一視同仁**

## 5. 🔴 最容易錯的地方

1. **關節單位與方向。** leader 的 `.pos` 經過校正檔正規化，模擬的關節是弧度。
   **符號接反會產生一份看起來正常、訓練出來卻是鏡像的資料集。**
   → **驗收要有一個「五姿態對照」測試**：home / 只轉 J1 / 只轉 J2 / 只轉 J3 / 夾爪開閉，
     逐一比對 leader 讀數與模擬關節角，**印成表，人看過才算過**。這與 `field_manual` §4-1 同一招。
2. **時間基準。** leader 讀取速度由序列埠決定，模擬步進由 solver 決定。
   → **以 dataset fps 為主時鐘**，記錄每一幀的實際 `dt`，收工印出 dt 分佈（不是只印平均值）。
   → 這與 D006 的 jitter 是同一類問題，用同一個方法驗。
3. **feature key 與順序。** 🔴 **必須與實機 campaign 一致**：
   `observation.images.left_front`, `observation.images.right_front`（順序即宣告順序，
   `docs/decisions.md` D022 §2026-08-30 有完整追查）。
   **不一致的話，這份資料連「拿來比較」都做不到。**
4. **資料集不得與實機資料混用。** D025 執行前提第 2 條。
   → `repo_id` 一律加 `sim_` 前綴，`meta` 裡明記模擬器名稱與版本。

## 6. 輸入

```
--leader-config configs/teleoperate_omx.yaml   # 只用 teleop 那一段，不連 follower
--sim mujoco|isaac                              # 見 §3，預設不要有
--scene   configs/sim/<scene>.xml|.usd
--fps 30
--repo-id sim_omx_pick_place_<date>
--episodes 5
--dry-run
```

## 7. 驗收條件

- [ ] §5-1 的五姿態對照表印出來且方向全對
- [ ] 錄完的資料集通過 `uv run python scripts/verify_dataset.py <root>`，退出碼 0
- [ ] `meta/info.json` 的 `features` 順序與實機 campaign 相同（程式自己比對並印出）
- [ ] dt 分佈印出 p50 / p95 / max，**而不是只有平均**
- [ ] 不連 follower 也能跑（這是這支腳本的重點：不受 D023 阻塞）
- [ ] `repo_id` 沒有 `sim_` 前綴時直接拒絕執行

## 8. 明確不做

- ❌ 不做 sim2real 遷移評估。那要等實機基準存在（D025 前提 1）
- ❌ 不把模擬資料併進 Phase B 的訓練集
- ❌ 不在這支腳本裡做 domain randomization。先讓管線通，再談變因

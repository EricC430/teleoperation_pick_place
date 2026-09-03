# S4 — 模擬環境 ＋ 真實 leader 臂遙操作蒐集

**目標檔案：`scripts/sim_teleop_collect.py`**

---

## 0. ✅ 2026-09-03：封鎖已解除（原 §0 的排序衝突警告作廢）

**`[Eric決定]` 2026-09-03，記在 `docs/decisions.md` D029：**

1. **模擬器 = Isaac Sim / Isaac Lab**（原 §3 的 MuJoCo vs Isaac 之爭已裁決，MuJoCo 出局）
2. **D025 執行前提 3「不佔用 9/26 前的任何工時」撤銷** —— 這條前提事實上在 8/28 就已被跨過
   （`wildbot_with_omxaiarm.usd` 是那天產出的）。**模擬工作即刻可做。**
3. **leader 臂直插跑模擬的那台 Linux 機器**（實驗室 5090 筆電），見新增的 §3.5

**仍然有效、不要誤讀成全面放行的兩條（D025 前提 1／2）：**

- **前提 1（措辭已收緊）：** 實機基準不是「動手做模擬」的門檻，而是
  **「用模擬資料訓練、或在書審／報告裡宣稱模擬有效」的門檻。**
- **前提 2：** sim 資料與實機資料是兩份資料集，**不得直接混用**（除非做了 domain adaptation 並明記）。
  → `repo_id` 一律加 `sim_` 前綴（§7 驗收條件）。

**本規格現在的範圍 = §2 的資產稽核 → §3.5 的環境搬遷 → §4 的管線 → §7 的驗收。全部開放施工。**

---

## 1. 目標

用**真實的 OMX leader 臂**遙操作**模擬環境裡的 follower**，錄出 LeRobot 格式的資料集。

**它買到的東西：**
- 變因掃描（光線、背景、物體位置）在模擬裡幾乎免費
- **不需要 follower 臂**，所以不受 D023 傳輸線／手臂借用阻塞
- 幾何／可達／干涉評估（D020 判準 1）——這一項**完全不需要實機資料**

**它買不到的東西（不要在計畫書裡誇大）：**
- 不能取代實機基準。D007 反對的是「用模擬取代實機」，D025 允許的是「用實機錨定模擬」
- 渲染分佈差距（光照、材質、感測器雜訊、動態模糊）不會因為相機規格對上就消失
- 🔴 **「lab day 之外也能蒐集」這條賣點已經被 D029 §3 削弱**：leader 必須插在實驗室的模擬主機上，
  所以仍然要人到實驗室。**不要繼續在計畫書裡寫「隨時可蒐集」。**

## 2. ✅ 原本的四項查證已完成 → 換成「資產稽核」

**原 §2 的四件事在 2026-08-28～09-03 之間全部有答案了，完整證據見 `decisions.md` D025 §2026-09-03：**

| # | 原問題 | 結論 |
|---|---|---|
| 1 | `omx_f` URDF 的 inertial 是不是佔位值 | ✅ **不是。** 每個 link 都有真實 mass 與完整 6 項慣性張量 |
| 2 | effort / velocity 有沒有寫進 URDF | ❌ **沒有，全是佔位值。但原廠規格查得到**（見下表），不必用猜的 |
| 3 | `robotis_mujoco_menagerie` 有沒有現成 OMX | 🚫 **不重要了**——已裁決 Isaac Sim，且我們自己已經有 USD |
| 4 | cyclo_lab 能不能換 robot asset | ❌ 仍是 OMY/FFW/SH5（Isaac Sim 5.1.0 + Isaac Lab ≥2.3.0）。**骨架可抄，資產不可抄** |

### 2-1 關節限制與致動器參數（`[已查證 2026-09-03]` ROBOTIS 官方規格）

| Joint | LeRobot 名稱 | 馬達 | 原廠行程 | 堵轉扭矩 | 空載轉速 |
|---|---|---|---|---|---|
| joint1 | `shoulder_pan` | XL430-W250-T | −270° ~ +360° | 1.5 N·m @12 V | 61 rpm ≈ 6.39 rad/s |
| joint2 | `shoulder_lift` | XL330-M288-T | −120° ~ +90° | 0.52 N·m @5 V | 103 rpm ≈ 10.8 rad/s |
| joint3 | `elbow_flex` | XL330-M288-T | −120° ~ +90° | 同上 | 同上 |
| joint4 | `wrist_flex` | XL330-M288-T | −100° ~ +100° | 同上 | 同上 |
| joint5 | `wrist_roll` | XL330-M288-T | ±270° | 同上 | 同上 |
| joint6 | `gripper` | XL330-M077-T | 0° ~ +100° | 0.18–0.228 N·m | 278–456 rpm |

🔴 **模擬要套的是「原廠行程 ∩ 現場實測限制」**：`experiment_spec.md` §3 的方位角扇區 ≈135°
是相機支架**實體擋路**造成的，比原廠行程更緊。只套原廠行程 = 模擬裡的手臂能去實機去不了的地方。

### 2-2 🔴 新的必做項：8/28 那份 USD 的稽核

`isaaclab_volume/assets/wildbot_with_omxaiarm.usd` **目前只是「匯入手臂並擺在車體上方」**
`[Eric說 2026-09-03]`——車體暫定（D020 未定案）、無固定件、未做干涉檢查、未做下列修正。

**六項必查（完整說明見 D029 §URDF→USD）：**

| # | 項目 | 匯入後的預設值會怎樣 |
|---|---|---|
| 1 | joint limits | 照抄 ±6.283 rad ⇒ 手臂穿過自己 |
| 2 | effort / velocity | 照抄 1000 / 4.8 ⇒ 力氣無限大 |
| 3 | 🔴 drive stiffness / damping | **URDF 裡沒有這個概念**，匯入器塞預設值 ⇒ 動態與實機無關 |
| 4 | 🔴 collision approximation | 預設 convex hull ⇒ **夾爪兩指之間被填滿，夾不到東西** |
| 5 | 🔴 `gripper_joint_2` 的 mimic 約束 | 可能掉成自由關節（URDF 有 mimic、multiplier −1） |
| 6 | self-collision | 預設關閉 |

**做法：USD 只留幾何＋慣性，第 1–6 項全部寫進 Isaac Lab 的 `ArticulationCfg`／`ImplicitActuatorCfg` 程式碼，
匯入本身寫成可重跑的腳本（`IsaacLab/scripts/tools/convert_urdf.py`）。不要用 GUI 手改 USD——重匯一次就全沒了。**

### 2-3 手上已經有的資產（不要重做）

| 路徑（`~/isaaclab_volume/`） | 內容 |
|---|---|
| `assets/wildbot_with_omxaiarm.usd` | OMX 已匯入 USD 並置於車體上方（待稽核） |
| `assets/open_manipulator_description/` | 官方 repo 整包（URDF／meshes／gazebo／ros2_control），8/28 clone |
| `assets/trash_obj/` | 14 個垃圾物體 USD：鋁罐 ×3、寶特瓶 ×3、香蕉、蘋果、柳橙、蛋盒、廚餘… |
| `assets/Trashcan/`、`assets/wildbot_car_urdf/wildbot_car.usd` | 垃圾桶、車體 |
| `pickup_place_direct_0203/` | 雙相機 pick-place DirectRLEnv（per-env randomization、YOLO 觀測）——骨架可抄 |
| `action_replay_isaac_sim/` | 動作重播與比對工具鏈——**§5-1 的五姿態對照可以直接用它的做法** |

⚠️ **`assets/open_manipulator_description/urdf/omx_f/omx_f.urdf` 與本 repo 的
`assets/omx_f/omx_f.urdf` md5 相同**（2026-09-03 驗）——S1 的 FK 與模擬用的是同一份 URDF，
**這是好事，但也代表任何一邊改了它，另一邊會靜靜地跟著錯。改動要同時記在 D026 與這裡。**

## 3. ✅ 技術選型已裁決：Isaac Sim（D029）

**原本的 MuJoCo vs Isaac Sim 比較表已移除**（完整經過留在 `decisions.md` D025 §2026-09-03 與 D029）：
8/27 主張 MuJoCo 的兩個理由（「Isaac 上手成本高」「沒有現成 OMX 資產」）在盤點 `isaaclab_volume`
之後都不成立，見 §2-3。

⚠️ **保留原提案的精神：第一版只驗管線，不做 domain randomization。**
選 Isaac Sim ≠ 第一版就要做 DR（§8 已明列不做）。

**可抄的兩份範本：**
- [cyclo_lab](https://github.com/ROBOTIS-GIT/cyclo_lab)：`scripts/imitation_learning/isaaclab_recorder/record_demos.py`、
  `scripts/sim2real/` 的錄製／標註／mimic 流程（資產是 OMY，**不可抄**）
- [NVIDIA SO-101 sim2real 教材](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/09-strategy1-dr-teleop.html)
  ／[Sim-to-Real-SO-101-Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop)：
  `lerobot_agent --task <IsaacLab task> --repo_id … --repo_root …`，**任務不同、流程同構**。
  DR 用 `task_env_cfg.py` 的 `EventTerm`（dome light exposure −4.0~3.0、色溫 2500–9500 K、HDRI、
  相機位移 ±0.02 m／±0.05 rad、物體位置）——**留給第二版。**

## 3.5 🔴 執行位置：leader 直插模擬主機（D029 §3）

```
❌ 不做：Windows 筆電(COM6, leader) --網路--> Linux(Isaac Sim)
✅ 要做：Linux 5090 筆電 同時接 leader(/dev/ttyUSB*) 與 跑 Isaac Sim
```

**理由（完整版見 D029）：** 有兩條迴路，**人在迴路**（sim 畫面 → 眼睛 → 手）比控制迴路更禁不起延遲，
而且它的失敗是隱形的——你會得到一份「慢而猶豫」的資料集，**ACT 會忠實地學會慢而猶豫，事後驗不出來。**
NVIDIA 的教材用的也是直插架構（`TELEOP_PORT` USB 接在跑 `teleop-docker` 的同一台）。

**動手前的前置（都不需要手臂在場，除了最後一項）：**

1. Linux 上建一份 **與 Windows 筆電同版本** 的 lerobot 環境（`environment.md` 的釘選規則現在適用於三台機器）
2. `/dev/ttyUSB*` 的 udev／by-id 綁定（`experiment_spec.md` §7 已有做法；Linux 沒有 `COM6`）
3. 校正檔直接沿用 `calibration/2026-08-31_omx_leader.json`（純馬達參數，與作業系統無關）
4. 🔴 **實測一次：leader 讀取頻率、以及畫面延遲。** 兩個都要有數字，不要憑感覺

## 4. 架構

**全部在同一台機器上（Linux 5090 筆電，見 §3.5）：**

```
真實 leader 臂 (OmxLeader, /dev/ttyUSB* — 不是 COM6)
   │  get_action() -> {"shoulder_pan.pos": ..., ... , "gripper.pos": ...}   # 6 個關節
   ▼
關節目標映射（單位、方向、行程對齊）      ← 🔴 最容易錯的一段，見 §5
   ▼
Isaac Lab ArticulationCfg 的模擬 follower（位置控制；drive gains 見 §2-2 第 3 項）
   ▼
sim.step()  → 渲染相機（內參對齊，見 §5-5） → 組成 frame
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
3. **feature key 與順序。** 🔴 **必須與實機 campaign 一致。2026-09-03 更正——以實際錄出來的
   dataset 為準，不是以文件為準：**

   ```
   1. observation.images.wrist          # D405 手腕相機
   2. observation.images.front-left     # D455 第三視角
   ```

   **證據：`data/huggingface/lerobot/EricC430/omx_pick_place_pilot/meta/info.json` 的 `features`
   鍵序（2026-09-03 讀）**，與 `configs/record_omx.yaml` 的 `cameras:` 宣告順序一致。
   ⚠️ **舊寫法 `left_front` / `right_front` 是「手腕相機不可用時的退路」**（兩台第三視角，
   D022 先驗期方案），**不是現行組態**。`experiment_spec.md` §3／§4-1 已同步更正。
   **不一致的話，這份資料連「拿來比較」都做不到。**
4. **資料集不得與實機資料混用。** D025 執行前提第 2 條。
   → `repo_id` 一律加 `sim_` 前綴，`meta` 裡明記模擬器名稱與版本。
5. **關節行程沒套現場限制。** 只套原廠行程（§2-1）而不套 `experiment_spec.md` §3 的方位角扇區 ≈135°，
   **模擬裡的手臂會去實機去不了的地方**，錄出來的示範實機重現不了。**兩者取交集。**

### 5-5 🔴 模擬與現實怎麼「對齊」——分兩件事，不要混在一起

**「拍一張真實照片、在模擬拍同一張、看是否完全重疊」是對的直覺，但它同時混進了兩個問題，
而這兩個問題一個做得到、一個做不到：**

| | 幾何對齊 | 外觀對齊 |
|---|---|---|
| 內容 | 相機內參／外參、物體位置、桌面尺寸 | 光照、材質、曝光、感測器雜訊、動態模糊 |
| 做得到嗎 | ✅ **做得到，而且要量化到像素** | ❌ **做不到，也不該試** |
| 對不上怎麼辦 | 修外參／內參，重測 | **交給 domain randomization**（第二版） |

**→ 目標不是「兩張圖疊起來看不出差別」。目標是「同一個 3-D 點在兩張圖裡落在同一個像素」。**

#### 具體協定（四層，由便宜到貴）

**T1 — 內參用讀的，不要用調的。**
D405／D455 的內參直接從裝置讀（`pyrealsense2` 的 `get_intrinsics()`，或 `rs-enumerate-devices -c`），
拿 fx/fy/cx/cy 換算成 Isaac Sim camera 的 `focal_length` / `horizontal_aperture` / 解析度。
⚠️ **不要用「目測 FOV 差不多」**。內參錯了，後面每一層都白做。

**T2 — 外參用標記量，不要用捲尺量。**
把 ArUco／棋盤格貼在座標墊的已知點上（S3 的墊子本來就有座標系），
真實相機拍一張 → 解出相機相對墊子的 pose → **那組數字就是模擬相機的外參**。
⚠️ 這一步順便解決了 `experiment_spec.md` §3 那三個還是空白的欄位（相機 x/y/z、俯角）。

**T3 — 用重投影誤差當驗收數字，不是用肉眼。**
在墊子的 N 個已知點放標記（或直接用墊子上已印的座標點），
比對「真實影像裡標記的像素座標」與「模擬渲染裡同一點的像素座標」。
**驗收：848×480 下，中位重投影誤差 < 10 px、最大 < 25 px**（提議值，第一次量完再依實際調）。
**印出每個點的誤差表，不要只印平均。**

**T4 — 疊圖只當煙霧測試。**
真實照片與模擬渲染 50/50 alpha blend 或做差值圖，**用來抓「整組外參接反了」這種粗錯**，
**不要拿它當通過標準**——它永遠不會完全重疊，因為外觀對齊做不到。

#### ⚠️ 現在還做不到 T2/T3 的原因

`experiment_spec.md` §3 的桌面高度、相機 x/y/z、俯角、光照 lux **目前全是空白**，
9/3 的 D405 曝光值也還沒回填。**在那張表凍結之前，模擬場景的幾何無從對齊。**

**→ 第一版的做法：明確接受「幾何未對齊」，在 dataset 的 `meta` 裡記下這件事，
先驗 §4 的管線通不通（那不需要幾何對齊）。等 §3 凍結後再重建場景並跑 T1–T4。
不要假裝對齊過。**

## 6. 輸入

```
--leader-config configs/teleoperate_omx.yaml   # 只用 teleop 那一段，不連 follower
                                               # ⚠️ port 要改成 Linux 的 /dev/ttyUSB*（§3.5）
--task    <Isaac Lab task name>                # 模擬器已裁決為 Isaac Sim（D029），不再有 --sim
--fps 30
--repo-id sim_omx_pick_place_<date>            # 沒有 sim_ 前綴直接拒絕（§7）
--episodes 5
--dry-run
```

## 7. 驗收條件

- [ ] §5-1 的五姿態對照表印出來且方向全對
- [ ] 錄完的資料集通過 `uv run python scripts/verify_dataset.py <root>`，退出碼 0
- [ ] `meta/info.json` 的 `features` 鍵序與**實機 pilot dataset 的 `meta/info.json`** 相同
      （程式直接讀那份檔案比對並印出，**不要照抄文件裡的表**——文件已經過期過一次了）
- [ ] dt 分佈印出 p50 / p95 / max，**而不是只有平均**
- [ ] 不連 follower 也能跑（這是這支腳本的重點：不受 D023 阻塞）
- [ ] `repo_id` 沒有 `sim_` 前綴時直接拒絕執行
- [ ] §2-2 的六項 USD 稽核逐項印出「原值 → 改成什麼」，**沒改的要說沒改**
- [ ] 關節上下限印出「原廠行程 ∩ 現場扇區」的實際採用值（§5-5 之前，這條就能做）
- [ ] leader 讀取頻率與畫面延遲各量一次並印出分佈（§3.5 前置第 4 項）

## 8. 明確不做

- ❌ 不做 sim2real 遷移評估。那要等實機基準存在（D025 前提 1）
- ❌ 不把模擬資料併進 Phase B 的訓練集
- ❌ 不在這支腳本裡做 domain randomization。先讓管線通，再談變因

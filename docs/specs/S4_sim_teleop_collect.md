# S4 — 模擬環境 ＋ 真實 leader 臂遙操作蒐集

**目標檔案：`scripts/sim_teleop_collect.py`**

---

## 0. ✅ 2026-09-03：封鎖已解除（原 §0 的排序衝突警告作廢）

**`[Eric決定]` 2026-09-03，記在 `docs/decisions.md` D029：**

1. **模擬器 = Isaac Sim / Isaac Lab**（原 §3 的 MuJoCo vs Isaac 之爭已裁決，MuJoCo 出局）
2. **D025 執行前提 3「不佔用 9/26 前的任何工時」撤銷** —— 這條前提事實上在 8/28 就已被跨過
   （`wildbot_with_omxaiarm.usd` 是那天產出的）。**模擬工作即刻可做。**
3. **leader 臂直插跑模擬的那台 Linux 機器**（實驗室 4090 筆電），見新增的 §3.5

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

### 2-2 ✅ 2026-09-03 已完成：USD 稽核，結論是重轉一份

**做法與完整對照表：`sim/README.md`。工具：`sim/audit_usd.py`（唯讀，可對任何 USD 跑）。**

**稽核 8/28 那份 GUI 匯入的 `wildbot_with_omxaiarm.usd`，六項全中：**

| 項目 | GUI 匯入版（8/28） |
|---|---|
| joint limits | 🔴 六個關節全是 ±360°（URDF 佔位值原封不動） |
| drive `maxForce` | 🔴 全是 1000 N·m |
| max joint velocity | 🔴 全是 275 °/s（匯入器預設） |
| drive stiffness | 🔴 0.41 / 1.38 / 4.58 / 3.39 / 0.27 / 0.03，**與馬達規格無任何關係** |
| drive damping | 🔴 **每個關節都是 0** —— 無阻尼位置驅動，會震盪 |
| collision | 🔴 手臂 14 個 `convexHull` ＋ 車體 8 個 `convexDecomposition` |
| mimic | ✅ 有（唯一沒壞的一項） |
| articulation root | ⚠️ **兩個**：`/World/car/...` 與 `/World/omx_f/...`——**車與臂之間沒有任何關節** |

⚠️ **最後一項值得單獨說**：8/28 的資產是「把手臂放在車體上方」，**不是把手臂裝在車上**。
這與 `[Eric說]` 一致，本身沒有錯，但**它不是一份 mobile manipulator 資產**，而且 D020 的車體還沒定案。

**→ `[Eric決定 2026-09-03]` 重轉一份，並且把轉換腳本化。** GUI 匯入的根本問題不是參數錯，是
**沒有記錄**：產生它的設定沒有留下，下一個人重匯會得到不同的東西。

### 2-2b ✅ 重轉結果（`sim/convert_omx_urdf.py`，2026-09-03）

```
./sim/run_in_container.sh convert_omx_urdf.py \
    --urdf $GUEST/assets/open_manipulator_description/urdf/omx_f/omx_f.urdf \
    --out  $GUEST/assets/omx_f_generated/omx_f.usd --headless
```

轉換後逐項套上 `sim/omx_constants.py` 的值並印出 `原值 → 新值`（S4 §7 驗收條件）。
**`sim/audit_usd.py` 對產物的結果：✅ 全數相符。**

💡 **一個意外的交叉驗證：重轉後八個 link 的質量總和 = 0.5588 kg，ROBOTIS 規格寫 560 g。**
**URDF 的 inertial 是真值，而且完整地進到 USD 了。**

**兩個沒有關閉的缺口，不要當成已解決：**

1. 🔴 **drive gains 是暫定值。** 規則是 `stiffness = 堵轉扭矩 / 5°`、`damping = 0.05 × stiffness`
   ——量綱誠實、可重現，**但不是校正**。要關閉它必須拿實機軌跡回歸（D029）。
2. 🔴 **mimic 的 gearing 正負號未驗證。** URDF 寫 `multiplier="-1"`，USD 寫 `gearing=1.0`
   （沿用 Isaac Sim GUI 匯入器對同一份 URDF 的產出）。**兩者的符號約定不同，光讀規格無法決定。**
   **由 §5-1 五姿態對照的「夾爪開閉」那一列來裁決**——`--mimic-gearing` 這個旗標就是為此存在。

⚠️ **另外發現一個匯入器 bug**：Isaac Lab 5.1 的轉換器**不會**把 URDF 的 `<mimic>` 帶進 USD
（即使 `convert_mimic_joints_to_normal_joints=False`）。`convert_omx_urdf.py` 已在後處理補上，
**若日後升級 Isaac Lab，這段要重驗**——否則第二根手指會靜靜地變成自由關節。

### 2-2c 舊的必做清單（保留為對照）

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

## 2-4 ✅ 2026-09-03 已完成：場景骨架與端到端煙霧測試

**新增 `sim/scene_constants.py`、`sim/omx_scene_cfg.py`、`sim/preview_scene.py`。**
桌子、手臂（用 §2-2b 重轉的 USD）、一個依照 seeded placement CSV 擺放的物體、垃圾桶、兩台相機
（依 dataset 順序 `wrist` → `front-left`）。**跑通了，且逐項印出診斷數字，不是只截圖看起來像樣。**

```
./sim/run_in_container.sh preview_scene.py --headless --enable_cameras \
    --placements docs/assets/placement_label_map_campA_20260831.csv --place t1 --out <dir>
```

**這個煙霧測試證明的事，僅此而已**：轉換好的手臂能載入成一個 articulation、seeded placement CSV
能驅動模擬座標系裡的物體位置（用同一個 `placement_id`，這正是「sim 資料與實機資料可比較」的關鍵）、
兩台相機以錄製解析度、以宣告順序渲染出畫面。**沒有證明幾何對齊**——`experiment_spec.md` §3 仍是空白，
`scene_constants.py` 的桌面/相機/光照數字明確標為 PLACEHOLDER，不是量出來的。

### 🔴 過程中抓到一個真的 bug，不是待測量的空格

第一次跑，物體放在桌面上方 6 cm，落地後停在 **z = 357.6 cm**——穿過桌子飛走了。
根因（用 `sim/inspect_object_usd.py` 純讀 USD、不跑物理查出來）：`assets/trash_obj/` 全部 8 個物體
**都是 `metersPerUnit = 0.01`**（自己的座標用公分），但 Isaac Sim 的世界舞台是公尺。
**USD reference 不會自動處理 stage 間的 `metersPerUnit` 落差**——一個 8 公分的罐頭沒加縮放，
會被當成 8 **公尺**的物體匯入，一接觸就爆開。

**修法：`trash_obj` 的物件一律套 `scale=(0.01, 0.01, 0.01)`**（`scene_constants.TRASH_OBJ_SCALE`）。
八個物體全部檢查過，換算後的真實尺寸都合理（香蕉 15×8×18 cm、寶特瓶 9×31×9 cm……），
**這是整個資產家族的通性，不是單一物體的猜測**。修完重跑：物體穩定停在 z = 79.0 cm
（桌面 75 cm + 4 cm，正是罐頭躺在桌上該有的高度）。

### ⚠️ 確認了、但沒有關閉：暫定 drive gains 撐不住手臂自身重量

命令全零關節姿態並保持 120 步（1 秒），**漂移 19–35°**（依 run 而異；第一次的異常值可能被物體爆炸的
衝擊波影響，19° 是較乾淨的讀數）。前視角畫面裡看得出手臂明顯下垂。
**這正是 `omx_constants.py` docstring 與 D029 早就標記的缺口，現在多了一個具體數字。**
**不要用「調大 tracking error 常數」這種猜測方式關閉它**——有兩個猜不出來的可能性：
(a) 增益真的太軟，(b) 全零關節角對這隻臂而言不是機械上輕鬆的姿態
（例如若那對應「手臂水平伸直」而非「摺疊收起」），而 XL330 在實機上能撐住，靠的是堵轉扭矩數字
沒表達出來的餘裕。**需要拿實機錄到的軌跡回歸，不是再調一次常數。**

---

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
✅ 要做：Linux 4090 筆電 同時接 leader(/dev/ttyUSB*) 與 跑 Isaac Sim
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

**全部在同一台機器上（Linux 4090 筆電，見 §3.5）：**

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
3. **feature key 與順序。** 🔴 **必須與實機 campaign 一致：**

   ```
   1. observation.images.wrist          # Innomaker U20CAM-720P 手腕相機，640×480
   2. observation.images.front-left     # D455 第三視角，848×480
   ```

   **證據：`configs/record_omx.yaml` 的 `cameras:` 宣告順序**，與 `experiment_spec.md` §3／§4-1 一致。
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
front-left（D455）的內參直接從裝置讀（`pyrealsense2` 的 `get_intrinsics()`，或 `rs-enumerate-devices -c`），
拿 fx/fy/cx/cy 換算成 Isaac Sim camera 的 `focal_length` / `horizontal_aperture` / 解析度。
wrist（Innomaker U20CAM-720P）是一般 UVC 鏡頭，沒有出廠內參可讀，改用棋盤格校正一次解出內參＋畸變。
兩者都已腳本化：`sim/calib_intrinsics_realsense.py`（front-left）、`sim/calib_intrinsics_checkerboard.py`（wrist）。
⚠️ **不要用「目測 FOV 差不多」**。內參錯了，後面每一層都白做。

**T2 — 外參用標記量，不要用捲尺量。**
把 ArUco 貼在座標墊的已知點上（S3 的墊子本來就有座標系），
真實相機拍一張 → 解出相機相對墊子的 pose → **那組數字就是模擬相機的外參**。
腳本化：`sim/calib_gen_targets.py`（產生標記）、`sim/calib_extrinsics_aruco.py`（解外參，wrist 相機
另串 `reach_logger.fk` 算 link5 姿態，因為 wrist camera 掛在 link5、不是世界座標）。
⚠️ 這一步順便解決了 `experiment_spec.md` §3 那三個還是空白的欄位（相機 x/y/z、俯角）。

**T3 — 用重投影誤差當驗收數字，不是用肉眼。**
在墊子的 N 個已知點放標記（或直接用墊子上已印的座標點），
比對「真實影像裡標記的像素座標」與「模擬渲染裡同一點的像素座標」。
**驗收：848×480 下，中位重投影誤差 < 10 px、最大 < 25 px**（提議值，第一次量完再依實際調）。
**印出每個點的誤差表，不要只印平均。**

**T4 — 疊圖只當煙霧測試。**
真實照片與模擬渲染 50/50 alpha blend 或做差值圖，**用來抓「整組外參接反了」這種粗錯**，
**不要拿它當通過標準**——它永遠不會完全重疊，因為外觀對齊做不到。

#### 2026-10-07 wrist T1／T2 實測紀錄（**T2 未通過，外參不採用**）

數值正本是 `calibration/` 底下的 JSON（檔名見各項）；這裡只記「做了什麼、結果能不能用、為什麼」。

**T1 內參（`calibration/2026-10-07_camera_intrinsics_wrist.json`）— 可用於去畸變**

- 23 張可用（拍 26 張、自動剔除 3 張），RMS 0.82 px，單張最大 1.46 px。fx/fy ≈ 695/697 px，k1 ≈ −0.53。
- 同日第一次（10/07 11:35）RMS 2.07 px，因 capture_03 動態模糊（角點偵測得到但位置歪）→ **作廢**。
- 🔴 **主點只能說「大約」：** 隨機抽 9 張標定 4 次，cx 落在 285–310、cy 在 241–252（補拍邊緣前是 274–320、196–263）。`[AI推論]` 這是自己加的檢查，樣本小；完整解 cx/cy = 288/237。要精確對位前再補畫面下緣與四角。
- 畫面最外側兩欄的角點仍只占約 15%，邊角的去畸變是外推。
- 解析度 640×480，對焦無自動對焦（`[柏宇說]`；專案內沒有任何對焦設定，**未用實測確認**）。
- 工具改動（`sim/calib_intrinsics_checkerboard.py`，未 commit）：板子靜止才拍、存原始幀、`--resume`／`--drop-above-px` 剔壞幀、tkinter 即時預覽。

**T2 外參（兩個姿態，標記與墊子兩次之間未動，座標檔 `calibration/aruco_points.csv`，操作者 Eric，數字由柏宇轉述 `[柏宇說]`）**

| | 姿態 A | 姿態 B |
|---|---|---|
| 關節 `.pos`（pan, lift, elbow, wflex, wroll, grip） | 3.6, −48.8, 12.5, −5.3, −0.5, 59.2 | 21.4, −9.4, −27.9, 6.8, −14.7, 78.6 |
| 偵測到的標記 | 7（缺 id 2） | 6（缺 id 2、3） |
| 重投影誤差 median／max | 0.72／1.97 px | 0.83／1.83 px |
| 解出 link5→camera 位置 (m) | (0.039, −0.004, 0.070) | (0.053, −0.035, 0.083) |
| PnP 相機離桌面 | 45.6 cm | 48.2 cm |
| 手量鏡頭中心離桌面 | 42.5 cm | 46 cm（PnP 多 2.2 cm） |
| 手量底座底面離桌面 | 14.6 cm（`[柏宇說]`） | 同 A（墊高沒動，`[柏宇說]`），算式用 0.146 |
| 輸出檔 | `…_camera_extrinsics_wrist.json` | `…_camera_extrinsics_wrist_poseB.json` |

**一致性檢查：不通過。** 相機固定在 link5，兩個姿態解出的 link5→camera 應該相同。實際：位置差 **3.6 cm**（dx 1.4、dy −3.1、dz 1.3）、旋轉差 **6.5°**。
判準（提議值，**專案文件沒有規定**）：位置 ≲ 1 cm 且旋轉 ≲ 2° 才採用；> 2 cm 視為 FK／關節零點有問題。
與現行 CAD 值（`scene_constants.CAM_WRIST_OFFSET_*`，(0.031, 0, 0.034)）比：A 差約 3.7 cm／18.7°，B 差約 6.4 cm。

→ **`scene_constants.CAM_WRIST_OFFSET_*` 維持 CAD 值，不寫入這次的數字。** 重投影誤差小**不代表**外參對：PnP 對共面點很容易擬合，換成 link5 座標才暴露 FK 的誤差。

**這次找到並修掉的腳本錯誤：** `sim/calib_extrinsics_aruco.py` 原本把底座（link0）放在桌面 `z = table_top_z`，但手臂在 15 cm 的墊高上（`scene_constants.ARM_RISER_HEIGHT`；模擬場景在 S5 §2-D 已修過同一個錯，只有這支腳本漏掉）。
修正前 link5→camera 的 z = 0.181 m（與 CAD 差 17 cm）；修正後 0.070 m。新增必填參數 `--arm-base-height-m`（桌面 → **底座板底面** = URDF link0 原點；墊高會漂移，見 `scene_constants.py` 註解，要當天量），並寫入輸出 JSON 的 `arm_base_height_m`。
`[AI推論]` 4 組試算（CSV x±7.13 cm、y 取負）都不能解釋 z 差；y 取負差到約 0.7 m，**y 方向應該是對的**（假設其餘正確）。

**剩餘差異的可能原因（`[AI推論]`，都未驗證）：**

1. 關節映射的殘差：S6（2026-09-21）與 touch calibration（2026-09-22）**已經量過**並寫進 `sim/joint_mapping.py`（該處註解記 3D 誤差 8.07 → 1.52 cm），但那是在**抓取附近的姿態**（腕部朝下、碰桌面 11 點）擬合的；A、B 是手臂抬高看標記的姿態，**在擬合範圍之外**，殘差可能更大（`[AI推論]`，未驗證）。B 的 wrist_roll −14.7、pan 差約 32° 也可能放大側向誤差。另外 `OFFSET_RAD` 的 `wrist_flex`（+24°）、`shoulder_lift`（+2°）註解寫明是為了讓抓取對上杯口而手調的，不是純量測。
   （🔴 本文件先前寫「至今未量」是錯的，2026-10-07 更正。）
2. `fk.py` 的原點是 link0，pan 軸在 link0 的 x = −0.01125 m；腳本把 link0 放在 (0, 0)，差 1.1 cm，**未確認是有意還是疏漏**。
3. 內參焦距或「鏡頭中心」量測點：PnP 相機高度**兩次都比手量多**（A 多 3.1 cm／7%，B 多 2.2 cm／5%）。方向一致，像系統性偏差而不是隨機誤差；但**不能解釋 A、B 之間 3.6 cm 的不一致**（那是兩次之間的差）。`[AI推論]`
4. 底座高度量測誤差：B 的底座高度沿用 A（墊高沒動），所以 B 補量鏡頭高度後重算**不會改變**結論。

**接下來（待決定）：** 先做 S6 的關節零點，再重做 T2；或補量 B 的兩個高度後重算，看 3.6 cm 會不會縮小。front-left（D 步）尚未做，內參有 10/06、10/07 兩份，先確認哪份對應 848×480。

**2026-10-07 front-left（D455）T2（`calibration/2026-10-07_camera_extrinsics_front-left.json`）— 可用，已有兩項獨立核對**

- 條件：848×480、內參 `…_camera_intrinsics_front-left.json`（SDK 讀出）、同一份 `aruco_points.csv`、不經 FK、不依賴墊高／關節零點。相機位置「沒動過」是 `[柏宇說]`，**沒有用 `align_camera.py` 驗證**；按 10/05 17:02 紀錄應在 A1。
- 偵測 7 個標記（缺 id 1：它在畫面正下方，被切掉一半）。重投影 median 0.98 px、max 1.35 px（門檻 10／25 px）。
- 解出（模擬世界座標，含 `table_top_z` = 0.75 的 PLACEHOLDER）：pos = (0.0425, 0.2441, 0.9675)；**離桌面 = 0.9675 − 0.75 = 0.2175 m**。朝向：bearing −40.0°、pitch −19.4°（由四元數算，ros 光學座標）。
- **核對 1（手量）：** 鏡頭離桌面 11 + 11 = 22 cm `[柏宇說]`（PnP 21.75 cm，差 0.25 cm）；俯角 17° `[柏宇說]`（PnP 19.4°，差 2.4°）。
  （🔴 同一天先量成 14.6 + 11 = 25.6 cm，後來更正為 22 cm；中間那個差 3.9 cm 的比對是量測輸入錯，不是外參問題。）
- **核對 2（獨立地標，沒參與求解）：** 用 uvc_60 ep 10（A1，杯子在 t11：x_pan 21.97、y_pan −7.41 cm）第一幀，把杯底與杯口上方 9 cm 投影回畫面。垂直方向吻合（杯底、杯高都對上）；水平偏右約 25 px（桌面約 2.5 cm）。`[AI推論]` 偏差可能是杯子擺位誤差，也可能是外參偏差，這個檢查分不開。目測，未量精確像素。
- 與現行模擬值（`scene_constants.CAM_FRONT_LEFT_*`，捲尺／量角器）的差：位置 x +1.6、y −3.1、離桌面高度 −4.3 cm；bearing +5°；pitch −9.4°（現行 −10°，手量 17° 與 ArUco 19.4° 都支持現行值太小）。
- 🔴 `scene_constants` 假設相機「站在 15 cm 墊高上，再高 11 cm」（`CAM_FRONT_LEFT_Z = ARM_RISER_HEIGHT + 0.11 = 0.26`），但手量與 PnP 都得到約 22 cm。**相機實際站在什麼上面未確認**，要弄清楚再決定怎麼改場景。
- 接進模擬時注意：腳本印的 z 含 0.75，`scene_constants` 存的是離桌面的值（使用處才加 `TABLE_TOP_Z`），不能直接貼；模擬相機目前是 look-at，要改成帶旋轉。
- **現場事實 `[柏宇說 2026-10-07]`：** 桌上平台高 11 cm；為了讓手臂達到預定的約 15 cm，**只在手臂底下**墊了 3.6 cm 的書（11 + 3.6 = 14.6，即 T2 當天量到的底座底面高度）。第三視角相機站在 11 cm 平台上、**沒有墊書**，再加支架 11 cm = 22 cm。所以相機高度與手臂高度是兩個獨立的量。
- **已改 `sim/scene_constants.py`（2026-10-07）：** `CAM_FRONT_LEFT_Z` 不再等於 `ARM_RISER_HEIGHT + 0.11`（0.26），改成相機自己的墊高 0.11 ＋ 支架 0.11 = **0.22**（`[柏宇說]`；ArUco 解出 0.2175）。`ARM_RISER_HEIGHT`（手臂底座）**沒有動**：今天量底座底面離桌面 14.6 cm、S6／touch calibration 是在約 15 cm 下擬合的、柏宇 9/21 註明錄 uvc_60 時是 15 cm，而且下面的疊圖顯示用 11 cm 會讓夾爪 TCP 比杯口低約 4 cm。x、y、bearing、pitch **還沒改成 ArUco 值**（見下）。
- **幾何疊圖（不是 Isaac 渲染；這台 Windows 沒有 Isaac Lab）：** 用 uvc_60 ep 10 的 `observation.state` → `joint_mapping` → FK，把手臂各關節與 TCP 用上述外參投影到真實畫面（杯子在 t11，底座 z 用 0.15）。夾取瞬間（f137）：TCP 的高度與杯口相符（FK 離桌面 10.2 cm，杯口 9.5 cm）；整條手臂骨架在畫面上比真實手臂**偏右約 40–60 px（約 4–6 cm）**。把底座高度改成 0.11，TCP 變成只高於桌面 6.2 cm、明顯低於實際夾爪，更差。偏差來源未分離：關節映射（touch calibration 約 1.5 cm）、link0 與 pan 軸的 1.125 cm、外參（杯子垂直吻合、水平偏右約 2.5 cm）、杯子擺位誤差都可能。`[AI推論]` 目測，未量精確像素。
- **本機 Isaac Sim 6.1 渲染（2026-10-07，`D:\isaac-sim-standalone-6.1.0-windows-x86_64`，RTX 3050 Laptop 4 GB，不需 Isaac Lab）：** 用 `assets/omx_f_generated/omx_f.usd` ＋ `assets/paper_cup/paper_cup.usd`，`Articulation.set_dof_positions` 做運動學擺姿（重力 0），相機用 T2 外參。首次啟動約 5.5 分鐘（編 shader），之後約 25 秒。uvc_60 ep 10 的 f107／f137 與真實畫面比：杯子在模擬中偏右約 20–25 px，手臂偏右約 50 px（夾取瞬間夾爪尖端落在杯口，高度對）。與上面的幾何疊圖結論一致：整體誤差約數 cm。**這是暫存腳本，不是 `sim/` 的管線**（沒有 DR、沒有 grasp attach、夾爪映射未驗證、忽略鏡頭畸變與主點偏移、桌面是方塊不是真實桌子）。踩過的坑：USD 的 `Matrix3d` 是列向量慣例，旋轉要轉置；`omx_f.usd` 的根 prim 自帶 xformOps，`XformCommonAPI.SetTranslate` 會靜默失敗，要放在乾淨的父 Xform 上。
- **杯子地標檢驗與外參修正（2026-10-07，`[AI推論]` 為主，腳本在暫存區未進 repo）：** 用 uvc_60 ep 0–49 的第一幀（相機在 A1），每集杯子的桌面位置是已知的（placement `t(ep+1)`，`x_pan_cm`／`y_pan_cm`），自動偵測杯底最前緣像素當地標（12 集偵測成功並目視核對，ep 45 偵測不一致排除），比較三組相機的投影誤差：
  | 相機 | 中位數誤差 | 偏差方向 |
  |---|---|---|
  | 舊（捲尺／量角器，`scene_constants` 9/21 值，離桌面 26 cm、俯角 −10°） | 86 px | 垂直 +88 px（投影落點太低） |
  | ArUco T2 外參 | 25 px | 水平 +26 px（落點偏右），垂直 ≈ 0 |
  | 用這 11 個地標微調後 | 5 px；留一法 7 px（最大 23 px） | — |
  微調後的相機：bearing −44.9°、pitch −17.1°、離桌面 20.3 cm，位置相對 ArUco 只動約 1–1.5 cm、旋轉約 5.3°（其中約 3° 是繞相機 y 軸的偏航）。**bearing 與手量的 −45°、pitch 與手量的 17° 吻合**，這是不靠標記的獨立佐證；ArUco 解出的 −40°／−19.4° 反而偏了幾度。ArUco 為何帶約 3–5° 的旋轉偏差**未確認**（可能是標記座標的量測、座標紙與 pan 軸座標有微小轉角、或內參；試過只做世界座標偏航修正，最好 9.5 px，不如全旋轉微調）。
  → 這個「用已錄示範的杯子位置當地標」的方法**不需要標記**，也可套用在其他相機位置（A2、B…）的資料集，前提是相機在該資料集內沒有動、placement 對應正確。
- **微調位姿已存檔並重新渲染（`calibration/2026-10-07_camera_extrinsics_front-left_refit.json`，標為 provisional）：** uvc_60 ep 10 夾取瞬間，目測夾爪尖端與真實的差距從約 +22 px 右、−15 px 上（ArUco 相機，以 424 px 寬的縮圖量）縮到約 +8 px 右；杯子大致重合。ep 10 不在 11 個地標內，做了一次樣本外檢查（f000 的杯底）：ArUco 誤差 (+15, −16) px、微調 (−9, −16) px，向量長度 22 → 18 px，**改善有限**，垂直 −16 px 兩者相同、原因未確認（可能是 ep 10 本身或偵測方式）。留一法 7 px 才是較可靠的數字。
  舊的 9/29 遠端渲染（`outputs/outputs/videos/ep*_calib_compare_front-left.mp4`，標籤 PREVIOUS touch LSQ／CURRENT hand-tuned）比的是**關節偏移**，相機仍是 9/21 的捲尺值，畫面裡地平線、牆、桌面都和真實對不上。
- **地標擴大到 uvc_60 ep 0–49（2026-10-07）：** 50 集中 32 集偵測得到杯底（其餘 18 集杯子在畫面外或被擋）。結果（`calibration/` 下四組相機都保留）：
  | 版本 | 檔案 | 32 個地標中位數 | 5 折交叉驗證（留出） | 朝向 | 離桌面 |
  |---|---|---|---|---|---|
  | 9/21 捲尺 | `scene_constants.py`（今天高度改 0.22） | 86 px | — | −45°／−10° | 22 cm（已改） |
  | 10/07 ArUco | `…_extrinsics_front-left.json` | 25 px | — | −40°／−19.4° | 21.75 cm |
  | 11 地標、六自由度 | `…_front-left_refit.json` | 6.3 px | — | −44.9°／−17.1° | 20.3 cm |
  | 32 地標、六自由度（不採用） | 未存檔 | 7.1 px | 7.0 px | −44.2°／−13.8° | 18.6 cm |
  | **32 地標、高度固定 22 cm** | `…_front-left_refit32_z220.json` | **5.5 px** | **6.3 px** | −43.3°／−18.7° | 22 cm（釘住） |
  地標變多後，六自由度的結果沒有變準，反而 pitch 和高度與手量值（17°、22 cm）拉開：地標都在桌面平面、距離相近，pitch 與高度（距離）的組合方向約束很弱，擬合會順著雜訊漂移。把高度釘在手量的 22 cm 後，像素誤差更低、pitch／bearing 也回到與手量吻合（pitch −18.7° 對手量 17°，bearing −43.3° 對捲尺 −45°）。`[AI推論]` 偵測到的杯子與 ep 10、ep 45 被穩健擬合剔除為離群。地標只涵蓋桌面高度，高處（夾爪、杯口）的誤差仍要靠渲染比對。
- **對照影片與寫入 `scene_constants.py`（2026-10-07，🟡 暫定）：** `outputs/outputs/videos/ep10_front-left_refit32_real_vs_sim.mp4`（ep 10，第 0–137 幀每 3 幀一格，REAL／SIM／BLEND）。杯子與真實一致（偏差約 8 px，縮圖，即全解析約 16 px）；**手臂不一致**：夾爪比真實偏右約 18–28 px（縮圖，約 36–56 px 全解析，約 3–5 cm），來源應是關節映射與 link0／pan 軸 1.1 cm，不是相機。依「一致才標定案」的規則，結果是**寫入但標暫定**：`scene_constants.py` 的 `CAM_FRONT_LEFT_POS`／`BEARING`／`PITCH` 改為微調值，`HFOV_FRONT_LEFT_DEG` 90 → 89.47（SDK 內參）；舊值留在 `*_TAPE_20260921`，ArUco 值留在 `calibration/` 與註解。🔴 微調相機有 **roll +4.8°**，但三支用 look-at 的腳本（`preview_scene.py`、`render_state_replay.py`、`replay_render_episode.py`）不能表達 roll（影像邊緣最多約 35 px），完整旋轉存在 `CAM_FRONT_LEFT_QUAT_ROS_REFIT`，**這些腳本尚未改**，也沒有在 Isaac Lab 容器裡驗證過。主點（431, 244）也沒有建模。
- **用 ArUco 檢查 wrist 誤差（2026-10-07，`[AI推論]`，暫存區腳本）：**
  - 🔴 **（同日晚間推翻，見下一條「相機位置有兩個」）** ~~**座標紙偏移（paper offset）是 ArUco 偏差的主因之一。**~~ 三個互相獨立的證據都指向同一個量級：(a) 用前左微調相機把 ArUco 標記三角化回桌面，與尺量座標平均差 (−1.0, −2.6) cm；(b) 把前左 ArUco 相機位置平移，使 31 個杯子地標誤差最小，最佳平移 (−2.0, −2.5) cm，誤差 31 → 11 px；(c) wrist 兩姿態 A、B 的 link5→camera 位置要最一致，最佳偏移 (−1.5, −3.25)／(−2.5, −3.0) cm（沒用到杯子）。**標記座標系（座標紙）相對 pan 軸座標約偏 (−2, −3) cm**；`calib_extrinsics_aruco.py` 本來就有 `--paper-offset-m`（X_true = X_paper + 偏移），這個數字沒有直接量過，只能由上述間接證據推出，要以夾爪尖端碰標記中心（`scripts/touch_calibrate.py`）獨立量。
  - 套用 (−2, −3) cm 後 wrist：A、B 的 link5→camera **位置差 3.6 → 1.4 cm**，但**旋轉差仍 6.5°**（平移改不了旋轉）；平均位置 (0.029, −0.032, 0.053) m，對 CAD (0.031, 0, 0.034)：x 吻合，y 差 3.2 cm、z 差 1.9 cm。重投影誤差不變（0.7／0.8 px）。
  - 其他檢查：wrist 內參焦距用標記重新擬合（固定主點與畸變）比棋盤格低 3.6%（669 對 695 px），套用後 PnP 相機高度 43.9／46.3 cm 對手量 42.5／46 cm（原本 +3.1／+2.2 cm 變 +1.4／+0.3 cm），但不影響 A、B 不一致。把標記座標整體轉動 −12°…+12° **無法**消除不一致（位置差最佳 2.9 cm、旋轉差始終 ≥ 5.8°）。以前左微調相機三角化的標記座標去解 wrist 反而更差（重投影 5 px、A–B 差 6 cm）：尺量的標記相對佈局自身一致，前左微調相機在標記位置（離杯子地標較遠）不可靠。**在 A、B 兩姿態下，任意一或兩個關節零點偏移（調到 ±15–21°）都無法把殘差降到 1.1 cm／1.9° 以下**，所以兩個姿態不足以分離關節誤差；要至少 4 個多樣姿態才能做 hand-eye＋關節偏移的聯合擬合。
  - 這個偏移也解釋了前左 ArUco 的 25 px 水平偏差（偏移約 2.8 cm ÷ 0.4 m × 427 px ≈ 30 px）。
- 🔴 **相機位置有兩個——上面「座標紙偏移」的結論被推翻（2026-10-07 晚，`[已查證]` 量測／`[AI推論]` 解讀）：** 今天新錄的 `paper_cup_normal_A1`／`recovery_A1`／`recovery_A1_tight`（41 集）與 ArUco 照片 `fl_t2.png` 是**同一個相機位置**（目視疊圖牆、窗簾、桌緣重合）；但與 uvc_60（9/13 錄、10/05 對回 A1）**不同**：用 `align_camera.py` 的木紋相關量，今天 vs uvc_60 平移 dx 40.8、dy −13.5 px（10/05 基準照 vs uvc_60 只有 4.5／−5.1 px）。所以 10/05 之後、拍 ArUco 之前，相機被動過（或「沒動過」的說法不成立），**時間點未確認**。
  - 用今天的杯子當地標（9 個中／遠場、杯底完整入鏡的位置，**不做任何擬合**）：ArUco 外參（不加偏移）中位數 **8.7 px**；加座標紙偏移 (−1, −2.6) cm 反而 32.7 px；`scene_constants` 現在的 refit32（擬合自 uvc_60）28.5 px。uvc_60 上則相反（refit32 8.8、ArUco 31.9 px）。→ 前一天的 25 px 是**兩個相機位置的差**，不是標記座標錯；座標紙偏移的證據 (a)(b) 都用了 uvc_60 的杯子或由它擬合的相機，已被汙染；只剩 (c)（wrist A/B 一致性）獨立，單獨不足以支持。
  - 渲染（本機 Isaac Sim 6.1，ArUco 相機、底座 14.6 cm、運動學擺姿）：今天 normal_A1 ep 12（t4）、ep 13（t2），接近與夾取瞬間夾爪、手臂、杯子目測都在縮圖約 5–10 px 內（全解析約 10–20 px），明顯比 uvc_60 ＋ refit32 的 36–56 px 好。影片：`outputs/outputs/videos/today_normalA1_ep{12,13}_front-left_aruco_real_vs_sim.mp4`。表示前一天「手臂偏 3–5 cm 是關節映射」的歸因也要重審：至少有一部分是相機與底座高度（uvc_60 錄製時 15 cm）的組合。
  - **相機外參需要時間軸**（與 `ARM_RISER_HEIGHT` 同一個問題）：refit32 只對 uvc_60 有效，ArUco 只對今天的錄製有效；`scene_constants.py` 目前只放得下一組。
- **2026-10-08 實驗：鏡頭模型、wrist_flex、腕部三組外參（today normal_A1 ep 12／13，本機 Isaac Sim 6.1，暫存區腳本 `iso_render2.py`）：**
  - Isaac 的 link5 位置與 `reach_logger/fk.py` 相同（差 < 0.1 mm）`[已查證]`，所以 USD 資產與手寫 FK 是同一套運動學。
  - 前左加上主點＋畸變（padded 理想渲染＋`camera_distortion` remap）：與不加只差幾 px，**不是主要誤差**。之前的輪廓 IoU 計算本來就含鏡頭模型。
  - wrist_flex 逐幀輪廓擬合（77 幀）：home 需 −22～−23°（兩集一致），夾取附近需約 0°（現行 mapping 最好）；拿掉手調 +24° 會讓 home 變好、夾取變差（影片可見）。**誤差隨姿態改變，單一偏移修不好**；S6 量角器量到 wrist_flex 1.79°/unit，排除單純比例錯。剩下嫌疑：lift／elbow 在高處的零點或比例、連桿與實物差，需多姿態量測分開。
  - 腕部三組外參（CAD、ArUco A、ArUco B，A／B 已改成 link0 在 +1.125 cm）**都對不上**：真實杯子在 CAD 與 A／B 預測之間；A 的相機落在夾爪網格內（畫面上方被擋黑）。T2 腳本把 link0 放在 (0,0) 是錯的（pan 軸在 link0 的 x=−0.01125），修正後 A／B 差 3.64 → 3.31 cm，未解決。模擬也缺檯子（腕部在 home 看到的是檯子前緣，模擬看到底座）。
  - 影片：`outputs/outputs/videos/today_normalA1_ep{12,13}_front-left_lens_real_vs_sim.mp4`、`…_wrist_CAD_A_B_real_vs_sim.mp4`。
- **暫時移除標記與手臂狀態前的紀錄（2026-10-07 20:36，之後要還原）：** 標記中心座標 `calibration/aruco_points.csv`（id 0–7，pan 軸座標，單位 m，高度 0）；照片 `calibration/shots/{fl_t2,wrist_t2,wrist_t2_poseB}.png`，手機俯拍佈局 `aruco_layout_phone_20261007.png`（這張拍在 CSV 寫成之前，標記是否後來被動過**未確認**）；當下手臂關節（`read_joint_pose.py`，唯讀）pan 21.4、lift −9.4、elbow −27.8、wrist_flex 6.8、wrist_roll −14.7、gripper 78.6，即姿態 B；墊高 11 cm ＋ 手臂下書 3.6 cm（底座底面 14.6 cm），前左相機站在 11 cm 平台上、沒動（`[柏宇說]`）。**沒有記錄的：** 每個標記的旋轉角度（只有中心）、座標紙（墊子）相對底盤的位置、書堆的位置。這幾項只存在於現場，移除前建議用膠帶在標記四角做記號並拍照。
- **尚未做：** 用專案管線（`sim/replay_render_episode.py`，Isaac Lab）渲染的 T3／T4；把 ArUco 的 x、y、bearing、pitch 寫進 `scene_constants`（模擬相機是 look-at、沒有 roll，且 `CAM_FRONT_LEFT_Y_PRE` 同時決定墊高箱的尺寸，改之前要先決定）。重投影誤差只證明「這組外參能解釋標記」，不證明疊合。

#### ⚠️ 現在還做不到 T2/T3 的原因

`experiment_spec.md` §3 的桌面高度、相機 x/y/z、俯角、光照 lux **目前全是空白**。
**在那張表凍結之前，模擬場景的幾何無從對齊。**

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

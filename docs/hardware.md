# Hardware inventory

Fill this in during the first site visit (see `field_manual.md` step 1 — "清點與拍照"). Keep it accurate;
this is the first thing to check when debugging "did something change."

## Arms

> 🔴 **2026-08-24：主平台已改為 OMX-AI（D021）。** SO-ARM ETA 2026-09-15。
> 在 OMX 上錄的資料是**正式 Phase B 資料**，不再是 pipeline-validation-only（D002 的 8/16 補記已作廢）。

| Role | Model | Servo model | Firmware version | Serial number | USB port | 狀態 | Notes |
|---|---|---|---|---|---|---|---|
| **Leader（現用）** | **OMX-AI (OMX-L)** | | | | | ✅ **在用** | 實驗室設備，助教已修復；需登記借用 |
| **Follower（現用）** | **OMX-AI (OMX-F)** | | | | | ✅ **在用** | 校正檔：`calibration/2026-08-24_omx_follower_arm.json`；teleop：`configs/teleoperate_omx.yaml` |
| Leader（未到貨） | SO-101 | STS3215 (12V) | | | | ⏳ **ETA 2026-09-15** | 🔴 **8/27 已決（D021 甲）：純備援。** OMX 做到底，**不轉、不重錄、也不「錄幾筆比較看看」**——跨臂 demo 不能合併，那種資料集既不是訓練資料也不是有效對照 |
| Follower（未到貨） | SO-101 | STS3215 (12V) | | | | ⏳ **ETA 2026-09-15** | 同上 |

> ⚠️ **本表的相機小節下方仍寫著「🟡 未查證：ACT 能否用單相機訓練」——那一條已於 2026-08-27 查證完畢**
> **（讀釘選版 LeRobot 原始碼，結論在 D022：可以，且相機數不改參數量）。保留原文是為了看得出它何時被推翻。**

**🔴 已知硬體問題（D023）：一條手臂上的傳輸線過短，卡住動作空間。**
**換線前不得量測工作範圍（A6）、不得錄製任何 demo。**（A7 配置「設計」已解禁）

**線材識別與解法 `[Eric說]` 2026-08-27：**

| 項目 | 內容 |
|---|---|
| **料號** | 應為 **Robot Cable-X3P**（DYNAMIXEL 3-pin 匯流排線） |
| 解法 1 | **把卡住的那條改走線** |
| 解法 2 | **檢查是否有「只需 100 mm 卻用了 180 mm」的線材可以互換** ← 零成本，優先試 |
| 解法 3 | 線上電商購買 |
| 解法 4 | 原代理商 **採智科技** 購買 |

> 💡 **解法 2 值得先做**：它不用等貨、不用花錢，而且如果成立，**第一個 lab day 就能解掉 D023**。
> **但它要在現場才能檢查** → 列入下次 lab day 清單的第一項。

## Cameras

| Position | Model | Resolution/FPS | USB port | 狀態 | Notes |
|---|---|---|---|---|---|
| **3rd-person ×2（現有）** | **Intel RealSense D435i ＋ D455** | | | ✅ **實驗室現存，且各自附短支架** `[Eric說]` 8/27 | 🔴 **有短支架 → 3D 列印支架的優先序大幅下降。** 先用現成的架起來凍結外參 |
| 3rd-person（原規劃） | Logitech C920 / RealSense D415 | 1080p / 30fps | | 🟡 備選 | 原本以為需自製支架 |
| **Wrist** | **USB UVC camera module（採購中）** | 720p/1080p | | ⏳ **ETA 2026-09-05–09-07** | **D022** |
| ~~Wrist~~ | ~~Intel RealSense D405~~ | | | ❌ **已否決** | **對手腕過重**，且需額外購置安裝工具。腕上重量會改變手臂動力學，而那會被烙進每一筆 demo（D022）|

**🔴 過渡配置（D022）：手腕相機到貨前，用「兩台第三視角相機」開始跑初期試驗，錄製不被相機擋住。**
單一第三視角相機也可以，但要當成 **D004 消融實驗的一個宣告過的分支**，不是隨手為之。

⚠️ **相機數量與外參是場景常數（D004）。不同相機配置的 demo 不得混用。**
過渡期錄的一律視為 **pilot data**，除非事先在 `docs/experiment_spec.md` 宣告並凍結為某個消融分支。

🟡 **未查證：** LeRobot 的 ACT 實作能否用單相機訓練。ACT 原論文用 4 台，單視角一定會掉——
**但掉多少本身就是可報告的結果。** 驗證成本約 20 分鐘（改 config 的相機數量）。**這是推論，不是已查證的事實。**

## Power / connectivity (Mobile & Outdoor spec)

- **Power supply spec for servos**: 12V DC (STS3215 12V bus servos)
- **Outdoor mobile power solution (Ref: XLeRobot)**: Anker SOLIX C300 (or equivalent portable power station) + Type-C to DC 12V PD trigger cables (5264/DC5525)
- **USB hub used**: Anker Powered USB Hub (prevents camera/bus servo brownout)
- **Host / Edge compute**:
  - Training & primary desktop validation: RTX 4090 Workstation
  - On-site data collection & baseline inference: Windows Laptop (`torch 2.11.0+cu126`, `lerobot 0.6.2`)
  - Vehicle edge deployment (P3/P4 candidate): Raspberry Pi 5 or Jetson Orin Nano / Mini PC

## Mobile Base & Mounting (Ref: Lab JetRover「小綠」vs. XLeRobot Cart)

> 🔴 **2026-08-27 `[Eric決定]`：「小藍」已排除，不再是候選。** 現行三候選、主責與死線見
> `docs/decisions.md` D020。下面保留「小藍」字樣的段落是 8/26 之前寫的，**讀到請以 D020 為準。**

- **Indoor / Phase A-B**: Desktop stationary platform (table clamps for quick-mount, rigid baseplate).
- **Outdoor / Ground Picking (Phase C-D)**:
  - *Constraint*: Target objects are on the ground (highway debris). High carts (e.g. IKEA RÅSKOG ~77cm) are **unusable for ground picking** due to arm stroke limits.
  - *Platform options*:
    1. Low-chassis JetRover「小綠」/「小藍」/ AMR base with lower mounting plate.
    2. Modified low-profile crawler/rover with arm mounted on forward lower bracket.
    3. Low-height mobile test box (simulating vehicle height).

## Spares / consumables (Procurement List)

- **Clamps / Mounts**: Heavy-duty table clamps (桌夾 4 入), **🆕 金屬 C 形夾 `[Eric決定]` 2026-08-27**, Magic arms (萬向魔術手臂), 1/4" screw camera mounts.
- **🆕 D405 支架零件（若找得到，`[Eric決定]` 2026-08-27）**: 一字形修繕平鐵片（孔徑 3–4 mm）、符合 D405 的螺絲。
  ⚠️ **這是備案，不是主案。** D022 已否決 D405 上腕（過重）；學長的四點反駁見 `docs/meeting/2026-08-24.md` §四。
  **若買到零件，用途是把 D405 當第三視角或固定用途，不是回頭裝在手腕上。**
- **🆕 Robot Cable-X3P**（長度待現場確認，見上方 D023 解法 2）
- **Fasteners / Tools**: M3 screw and nut kit, cable ties, precision screwdriver set, wire cutters.
- **Cables**: USB-C to USB-A high-speed cables, USB-C to DC 12V trigger cables, 5264 serial cables.
- **Finger / Gripper spares**: TPU95A flexible gripper pads / 3M grip tape.

## Photos

Store equipment photos alongside this file (or link to the shared drive/NAS location) — model plates,
cable routing, and connector layout are exactly the things you won't remember correctly from memory when
debugging remotely.

---

## 🎒 下次 lab day 攜帶清單（2026-08-27 建立）

> 🔴 **2026-08-30 採購狀態 `[Eric說]`：已購 C 形夾、M3 螺絲、墊片、平鐵片、標準鋁罐。**
> **仍缺，且缺了會讓當天最重要的量測做不成：捲尺／直尺（≥50 cm）、量角器或手機測角 app、紙膠帶＋油性筆。**
> **極座標紙暫時不需要**——凍結抽樣清單還不存在（D024／D023 的三個腳本尚未寫）。
> 逐項對照與當天任務序見 `docs/meeting/2026-08-31.md`。

**建立理由：** `docs/phase_plan.md` §T0 —— lab day 是最稀缺資源，**到現場才發現沒帶＝整天報銷**。

### 🔴 解 blocker 用（沒帶就白跑）

| 物品 | 用途 | 取得 |
|---|---|---|
| **捲尺／直尺（≥50 cm）** | 量 r_outer / r_inner —— **這是 A6 三個數字的關鍵**。`omx_follower` 只給關節角、無末端位姿，量尺是最便宜的路（D023） | 文具店／五金行 |
| **量角器 或 手機測角 app** | 量「傳輸線限制哪個方位角」 | 同上／手機免費 app |
| **紙膠帶 ＋ 油性筆** | 標記手臂基座原點、相機位置（**外參要能重現**）、標記線材 | 文具店 |
| **筆電＋充電器＋USB-C/A 線** | 現場跑 teleop 與錄製 | 自備 |
| **手機（拍照）** | `docs/hardware.md` 要求拍設備照、接線走法、型號銘牌 | 自備 |

### 🟡 任務物體（⚠️ 有一個容易做錯的地方）

| 物品 | 數量 | 取得 |
|---|---|---|
| **鋁罐** | 同款 ×3–5 | 超商／賣場 |
| **不透明寶特瓶** | 同款 ×3–5 | 同上 |
| **紙杯** | 同款 ×1 包 | 賣場／文具店 |
| **目標容器（垃圾桶／收納盒）** | 1 | ⚠️ **先確認實驗室有沒有**，沒有才買 |

> 🔴 **一定要買「同款」多個，不要混品牌。**
> 物體外觀是**場景常數**（D004 同理）。換一個不同品牌的鋁罐＝換了一個物體，
> **先前錄的 demo 就不再對應**。買同款備品是為了「摔壞／變形時能換一模一樣的」。

### 🟢 機構與相機

| 物品 | 用途 | 取得 |
|---|---|---|
| **金屬 C 形夾** `[Eric決定]` | 桌夾固定底板／相機 | 五金行 |
| **一字形平鐵片（孔徑 3–4 mm）＋ 符合 D405 的螺絲** `[Eric決定]` | **現場實測 D405 裝在腕上到底可不可行**——把學長的四點反駁從推測變成量測 | 五金行。⚠️ **買「已經有孔」的**——學長明確說我們沒有鑽頭，小直徑鑽頭也難找（`docs/meeting/2026-08-24.md` §四） |
| **極座標紙** | 擺放對位（D023 腳本 ③） | 影印店大圖輸出，或 **A4 拼貼**即可。⚠️ **擺完要移走**——留在畫面裡就變成背景常數 |
| M3 螺絲／螺帽組、束線帶 | 通用 | 五金行 |

### ⚪ 不用帶

- **Robot Cable-X3P** —— 這次的任務是**現場檢查有沒有「只需 100 mm 卻用了 180 mm」的線可互換**（D023 解法 0，零成本）。確認不成立才買。
- **3D 列印相機支架** —— 實驗室現有 D435i＋D455 各附短支架。

### 📋 現場一定要做的三件事（照順序）

1. **X3P 線互換檢查**（解法 0）→ 成立就當場解掉 D023
2. **量三個數字**：受限方位角 ／ 最差方位角 r_max ／ 可下爪 r_min → 填進 `docs/experiment_spec.md` §3
3. **D405 腕上實測**：裝上去、跑一次 teleop，記錄是否影響動作 → 把 D022 的「過重」從推測變成證據

### 🔑 進場條件

**學長基本在桃園，實驗室週一至週五都開。**
**週末也可以來，但學長沒有學生證進不了系館 → 需要用你的學生證幫他開門。** `[Eric說]` 2026-08-27
**→ 假日去要事先跟學長約時間，不能臨時。**

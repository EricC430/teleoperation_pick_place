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

**✅ 2026-08-31 現場：以「解法 1 改走線」把線繞過空間、解除硬性行程限制（實驗室無備用線）。**
🔴 **但「改走線」≠「換線」：改走線會改變哪些方向受限，`decisions.md` D023 §2026-08-27 的
保守工作區豁免因此作廢。** → A6/A7 須對**改走線後**的可達集重新量測、凍結；A8 錄製仍擋，
待 S1 產出。走線路徑須拍照、貼標，存為場景常數（再次改走線 = 既有配置作廢）。
詳 `decisions.md` D023 §2026-08-31。

**線材識別與解法 `[Eric說]` 2026-08-27：**

| 項目 | 內容 |
|---|---|
| **料號** | 應為 **Robot Cable-X3P**（DYNAMIXEL 3-pin 匯流排線） |
| **現況** | ⚠️ **2026-08-31 已以「解法 1 改走線」解決空間問題**（實驗室無備用線）。**未購入替代線**；日後若裝上正規長度替代線，屬另一次工作範圍變更 → 需重新量測凍結 |
| 解法 1 | **把卡住的那條改走線** ← ✅ **2026-08-31 採用**（代價：保守工作區豁免作廢，須重量工作範圍） |
| 解法 2 | **檢查是否有「只需 100 mm 卻用了 180 mm」的線材可以互換** ← 零成本，當日未找到可互換線 |
| 解法 3 | 線上電商購買（未執行） |
| 解法 4 | 原代理商 **採智科技** 購買（未執行） |

> 💡 **解法 2 本應優先**（零成本、單調放寬、不作廢豁免），但 2026-08-31 現場未找到可互換的線，
> 且實驗室無備品，故改採解法 1。**代價是 A6 工作範圍要對改走線後的狀態重量一次。**


## Cameras

| Position | Model | Resolution/FPS | USB port | 狀態 | Notes |
|---|---|---|---|---|---|
| **3rd-person ×2（現有）** | **Intel RealSense D435i ＋ D455** | | | ✅ **實驗室現存，且各自附短支架** `[Eric說]` 8/27 | 🔴 **有短支架 → 3D 列印支架的優先序大幅下降。** 先用現成的架起來凍結外參 |
| 3rd-person（原規劃） | Logitech C920 / RealSense D415 | 1080p / 30fps | | 🟡 備選 | 原本以為需自製支架 |
| **Wrist** | **USB UVC camera module（採購中）** | 720p/1080p | | ⏳ **ETA 2026-09-05–09-07** | **D022** |
| ~~Wrist~~ | ~~Intel RealSense D405~~ | | | ❌ **已否決** | **對手腕過重**，且需額外購置安裝工具。腕上重量會改變手臂動力學，而那會被烙進每一筆 demo（D022）|

**🔴 過渡配置（D022）：手腕相機到貨前，用「兩台第三視角相機」開始跑初期試驗，錄製不被相機擋住。**
單一第三視角相機也可以，但要當成 **D004 消融實驗的一個宣告過的分支**，不是隨手為之。

### 🔴 2026-08-31：錄製筆電扛不動 2× RealSense + 即時 AV1 編碼（`[產出物]` 現場 log）

`lerobot-record` 用預設 `rgb_encoder: libsvtav1` + `streaming_encoding` 時，錄製迴圈掉到 ~3 Hz；
關掉 streaming、降到 848×480@15 後仍會在 `save_episode()` 編碼期間餓死 D455 抓幀執行緒
→ `TimeoutError: latest frame is too old 530ms`（上限 500ms，`omx_follower.py:185` 硬寫死、不可由 config 調）。
筆電 CPU：**AVX2、Level of Parallelism 5**。SVT-AV1 即使 M10 對兩路 848×480 仍太重。

**現行對策（寫進 `configs/record_omx.yaml`）：** `rgb_encoder.vcodec: h264` + `preset: veryfast` + `encoder_threads: 2`。
h264 veryfast 約比 SVT-AV1 M10 輕 5–10×。**改 codec 後與既有 av1 資料集不相容 → 正式 pilot 用新的 `repo_id`/`root`，不要 resume 到煙霧測試資料集上。**
仍不夠時的後備：① 只用一台相機 ② `video: false` + 事後離線編碼 ③ 換較強的機器錄製。
`streaming_encoding: false` 是預設值，`true` 是主動選重路徑（本機不要開）。

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
>
> ⏳ **2026-08-31 現況確認 `[Eric說]`：送廠加裝 IMU 的 wildbot（小綠／候選②）需 1–2 週才回來（ETA ~9/7–9/14）。**
> 剛好銜接 Phase C 起點，資訊已同步給 9/1 柏宇主責的成本試算。

- **Indoor / Phase A-B**: Desktop stationary platform (table clamps for quick-mount, rigid baseplate).
- **Outdoor / Ground Picking (Phase C-D)**:
  - *Constraint*: Target objects are on the ground (highway debris). High carts (e.g. IKEA RÅSKOG ~77cm) are **unusable for ground picking** due to arm stroke limits.
  - *Platform options*:
    1. Low-chassis JetRover「小綠」/「小藍」/ AMR base with lower mounting plate.
    2. Modified low-profile crawler/rover with arm mounted on forward lower bracket.
    3. Low-height mobile test box (simulating vehicle height).

## Spares / consumables (Procurement List)

- **Clamps / Mounts**: Heavy-duty table clamps (桌夾 4 入), **✅ 金屬 C 形夾（已購）**, Magic arms (萬向魔術手臂), 1/4" screw camera mounts.
- **D405 支架零件**: **✅ 一字形修繕平鐵片 ＋ M3 螺絲 ＋ 墊片（已購）**。
  ⚠️ **這是備案，不是主案。** D022 已否決 D405 上腕（過重）；若買到零件，用途是現場實測並留存證據。
- **Robot Cable-X3P**: ❌ **實驗室無備用線**。現場先檢查是否可互換；若不成立則量測長度後下單採購。
- **目標容器（垃圾桶／收納盒）**: ❌ **實驗室無目標容器**，需自行採購同款標準容器。
- **Fasteners / Tools**: M3 screw and nut kit, cable ties, precision screwdriver set, wire cutters.
- **Cables**: USB-C to USB-A high-speed cables, USB-C to DC 12V trigger cables, 5264 serial cables.
- **Finger / Gripper spares**: TPU95A flexible gripper pads / 3M grip tape.

## Photos

Store equipment photos alongside this file (or link to the shared drive/NAS location) — model plates,
cable routing, and connector layout are exactly the things you won't remember correctly from memory when
debugging remotely.

---

## 🎒 Lab Day 攜帶清單（2026-08-31 同步更新）

> **已帶／已備妥項目（2026-08-30 & 08-31 `[Eric說]`）：**
> ✅ **捲尺／直尺（≥50 cm）**、✅ **極座標紙（定位墊）**、✅ **螺絲起子**、✅ **金屬 C 形夾**、✅ **M3 螺絲／墊片／平鐵片**、✅ **標準鋁罐**、✅ **筆電＋線材**。
>
> 🔴 **S1/S2/S3 模組已全數實作完成**（`scripts/reach_logger.py`、`scripts/sample_placements.py`、`scripts/make_placement_mat.py`）。
> 現場執行 S1 量測後即可即時產出 S2 抽樣清單並以極座標紙對位。
> 逐項對照與當天任務序見 `docs/meeting/2026-08-31.md`。

**建立理由：** `docs/phase_plan.md` §T0 —— lab day 是最稀缺資源，**到現場才發現沒帶＝整天報銷**。

### 🔴 解 blocker 用

| 物品 | 狀態 | 用途 | 取得 |
|---|---|---|---|
| **捲尺／直尺（≥50 cm）** | ✅ **已帶** | 量 r_outer / r_inner，S1 5-pose FK 實體驗證與 fallback | 自備 |
| **極座標紙（定位墊）** | ✅ **已帶** | 現場擺放對位、標定基準與 S2 配置驗證共用 | 自備 |
| **螺絲起子** | ✅ **已帶** | 手臂底座／夾爪鎖固、D405 鐵片鎖固比對 | 自備 |
| **筆電＋充電器＋USB-C/A 線** | ✅ **已帶** | 現場跑 S1/S2、teleop 與錄製 | 自備 |
| **手機（拍照）** | ✅ **已帶** | 拍設備照、接線走法、相機外參合照 | 自備 |
| **紙膠帶 ＋ 油性筆** | 🔴 **需帶** | 標記手臂基座原點、相機位置（**外參要能重現**）、標記線材 | 文具店 |
| **量角器 或 手機測角 app** | 🔴 **需帶** | 輔助量測「傳輸線限制哪個方位角」 | 手機免費 app |

### 🟡 任務物體與容器

| 物品 | 狀態 | 數量 | 取得／說明 |
|---|---|---|---|
| **標準鋁罐** | ✅ **已購** | 同款 ×3–5 | 超商／賣場（已備妥） |
| **不透明寶特瓶** | 🟡 待補 | 同款 ×3–5 | 超商／賣場 |
| **紙杯** | 🟡 待補 | 同款 ×1 包 | 賣場／文具店 |
| **目標容器（垃圾桶／收納盒）** | 🔴 **待購** | 1 | ❌ **已確認實驗室無**，需自行購買同款標準容器 |

> 🔴 **一定要買「同款」多個，不要混品牌。**
> 物體外觀是**場景常數**（D004 同理）。換一個不同品牌的鋁罐＝換了一個物體，
> **先前錄的 demo 就不再對應**。買同款備品是為了「摔壞／變形時能換一模一樣的」。

### 🟢 機構與相機

| 物品 | 狀態 | 用途 | 取得 |
|---|---|---|---|
| **金屬 C 形夾** | ✅ **已帶** | 桌夾固定底板／相機 | 已購 |
| **一字形平鐵片 ＋ M3 螺絲 ＋ 墊片** | ✅ **已帶** | **現場實測 D405 比對**——把學長的四點反駁從推測變成量測 | 已購 |
| M3 螺絲／螺帽組、束線帶 | 🟡 備用 | 通用理線與鎖固 | 五金行 |

### ⚪ 備品現況

- **Robot Cable-X3P** —— ❌ **實驗室無備用線**。現場先檢查是否可互換（D023 解法 2，零成本）；若不成立則當場量測長度並下單。
- **3D 列印相機支架** —— 實驗室現有 D435i＋D455 各附短支架。

### 📋 現場一定要做的三件事（照順序）

1. **X3P 線互換檢查**（解法 2）→ 成立就當場解掉 D023；不成立則量長度下單購線。
2. **S1 量測 ＋ S2 抽樣**：執行 `scripts/reach_logger.py`（5-pose FK 驗證 ＋ 量測四個關鍵值），即時跑 `scripts/sample_placements.py` 產出 `configs/placements/` 凍結清單。
3. **D405 腕上實測**：比對鐵片孔位、螺絲墊片壓固與重量干涉 → 拍照存證。

### 🔑 進場條件

**學長基本在桃園，實驗室週一至週五都開。**
**週末也可以來，但學長沒有學生證進不了系館 → 需要用你的學生證幫他開門。** `[Eric說]` 2026-08-27
**→ 假日去要事先跟學長約時間，不能臨時。**


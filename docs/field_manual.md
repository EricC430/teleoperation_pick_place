# 現場操作手冊 v1（2026-08-14 更新）

> ⚠️ 這份要離線可查（印出來，或存成手機／筆電裡的離線檔）。現場網路不一定穩。
>
> ⚠️ 本手冊所有路徑皆以專案根目錄 `teleoperation_pick_place/` 為基準。
> ⚠️ LeRobot CLI 版本：**0.6.2**（PyTorch `2.11.0+cu126` / Python `3.12.13` on Windows 11 / Linux）。
> 到現場裝好後，**第一件事是跑 `--help` 確認實際參數**，再據以修正這份手冊。

---

## 0. 檢查點清單

- [ ] 手臂 leader / follower 各一，配對正確（OMX-AI / SO-101）
- [ ] 電源供應器規格正確（12V）、已通電
- [ ] USB hub 已接、裝置都被辨識到（COM port & USB Cameras）
- [ ] 相機已被系統與 LeRobot 辨識到（過渡配置 = D405 腕 + D455 第三視角，D022 §2026-09-01）
- [ ] 🔴 **相機傳輸線理線**：D405 的線**穿過手臂的鏤空/走線通道**拉出，用束線帶/走線夾做應變消除，**確保線材完全不進手臂運動範圍、不擋操作空間**。
      線卡進關節連桿會讓校正記到錯誤極限值 → 整台校正失效（`camera_mount.md` §5-3、D023 同類地雷）。裝好後全範圍 teleop 掃一遍確認不勾線。
- [ ] 目標物體已備妥（鋁罐 ✅已購；紙杯／不透明寶特瓶待備；目標容器 ❌確認實驗室無，需自購，見 `docs/experiment_spec.md` §2）
- [ ] 現場工具：
      - **螺絲起子**（✅已帶，手臂底座／夾爪鎖固）
      - **捲尺**（✅已帶，量測與 FK 實體比對）
      - **極座標定位墊**（✅已帶，紙張已備，供 S1 量工作範圍、後續放置配置點物體共用；見 `docs/specs/S1_reach_logger.md`、`docs/decisions.md` D023）
      - **C 形夾、M3 螺絲、墊片、平鐵片**（✅已帶，固定底座與 D405 比對）

---

## 1. 確認裝置

### (1) 檢查手臂 Serial Bus (COM Port)
查詢 COM 埠與手臂 MotorBus：
```powershell
# 方法 A：使用 LeRobot CLI 自動偵測 / 拔插比對 COM 埠
uv run lerobot-find-port

# 方法 B：PowerShell 快速查詢系統目前發揮作用的 COM 埠
[System.IO.Ports.SerialPort]::GetPortNames()
```
預期看到：
- `COM6`（Leader 手臂）、`COM5`（Follower 手臂，依實際插入 USB 序列埠而定）。

如果沒看到：
→ 檢查 USB 線材、電源轉換板燈號、Windows 裝置管理員（`devmgmt.msc`）是否有未安裝驅動程式的裝置（如 CH340 / FTDI / CP210x）。

### (2) 檢查相機 (Webcam / OpenCV / RealSense)
> ⚠️ **重要觀念**：`lerobot-find-cameras` 支援指定後端 `{realsense, opencv}`。若加上 `opencv` 參數（即 `lerobot-find-cameras opencv`），LeRobot **只會掃描一般 UVC 視訊鏡頭，並主動略過 RealSense 深度相機**。同時，環境需安裝 `pyrealsense2`（`uv pip install pyrealsense2`）。

查詢相機指令：
```powershell
# 1. 查詢所有相機（推薦：同時列出 OpenCV 與 RealSense 裝置）
uv run lerobot-find-cameras

# 2. 僅查詢 RealSense 系列相機（如 D405 / D455 / D435）
uv run lerobot-find-cameras realsense

# 3. 僅查詢一般 USB/OpenCV 視訊鏡頭
uv run lerobot-find-cameras opencv
```

預期看到（以目前實驗室硬體為例）：
- **RealSense D405**：Serial ID `260322271459`（USB 3.2 模式，過渡期配置為 `left_front`）
- **RealSense D455**：Serial ID `262822305610`（USB 3.2 模式，過渡期配置為 `right_front`）
- **OpenCV Cameras**：`OpenCV Camera @ 0`、`OpenCV Camera @ 2` 等
- 拍攝的測試照片會自動存入 `outputs/captured_images/`（包含 `realsense_*.png` 與 `opencv_*.png`）。

---

## 2. 安裝與跨成員環境配置

### (1) 建立與安裝指令（Windows + uv + CUDA 12.6 修復）：
在 PowerShell 執行自動化腳本（或手動指令）：
```powershell
# 自動化安裝與 Torch CUDA 修復腳本
.\scripts\setup_laptop.ps1

# 或是手動按順序安裝：
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -e ".\lerobot[core_scripts,feetech]"
uv pip install --force-reinstall "torch==2.11.0" "torchvision>=0.22.0,<0.27.0" --index-url https://download.pytorch.org/whl/cu126
```

### (2) 驗證環境指令：
```powershell
uv run python -c "import torch, torchvision, lerobot; print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, torchvision: {torchvision.__version__}, lerobot: {lerobot.__version__}')"
```
預期輸出：
`torch: 2.11.0+cu126, CUDA: True, torchvision: 0.26.0+cu126, lerobot: 0.6.2`

### (3) 跨成員 `HF_HOME` 快取隔離處理：
> 💡 協作者的本機環境可能將 `HF_HOME` 設於外接硬碟（如 D 槽）或其他非預設路徑。為避免路徑未掛載導致 `WinError 3`，且不干擾個人其他專案，**每個終端機視窗一開始就重定向快取到專案內部**：
```powershell
# Windows PowerShell（僅對當前視窗生效）— 兩個都要設
$env:HF_HOME = "$PWD\.cache\hf"                 # datasets / huggingface_hub 快取（resume、load_episodes 走這條）
$env:HF_LEROBOT_HOME = "$PWD\.cache\lerobot"    # LeRobot 自己的 home（calibration 預設、dataset 預設 root）

# Linux / macOS Bash
export HF_HOME="$PWD/.cache/hf"
export HF_LEROBOT_HOME="$PWD/.cache/lerobot"
```

> 🔴 **只設 `HF_LEROBOT_HOME` 不夠。** `lerobot-record --resume` 會經 `datasets.Dataset.from_parquet`
> 建快取目錄，那條吃的是 `HF_HOME`。`HF_HOME` 還指在未掛載的 `D:\` 時，`resume` 會在
> `load_episodes` 炸 `WinError 3`，然後被誤判成「本地 metadata 不存在」→ 去打 HF Hub →
> 若 repo 不存在/未登入再爆 `401`（2026-08-31 踩過）。
> `torchcodec` 的 `libtorchcodec_coreN.dll` 載入失敗警告是**無害的**，會自動 fallback 到 `pyav`。

---

## 3. 校正 calibration ★最重要

> 💡 **最佳設計（零覆蓋、資料與校正強綁定，見 `docs/decisions.md` D017）：**
> 所有設定檔存放於 `configs/`。
> 我們將「當天日期」直接寫入 YAML 的 `id` 欄位（例如 `id: 2026-08-14_leader` 與 `id: 2026-08-14_follower`），LeRobot 校正完成後會**自動生成帶日期的檔案**（如 `calibration/2026-08-14_leader.json`），免手動改名且永遠不被覆蓋。

指令（使用 YAML 設定檔）：
```powershell
# 1. 校正 Teleop / Leader 手臂 (預設 COM6)
uv run lerobot-calibrate --config_path configs/calibrate_leader.yaml

# 2. 校正 Robot / Follower 手臂 (預設 COM5)
uv run lerobot-calibrate --config_path configs/calibrate_follower.yaml
```

> ※ 若換日收集資料，請在 `configs/*.yaml` 中將 `id` 改為當日日期（例如 `2026-08-31_leader`）。
> ※ 若現場 COM 埠有所變動，可直接編輯 YAML 或在 CLI 尾端覆蓋參數（例如 `--teleop.port=COM6`）。

步驟：（逐步，含手臂要擺什麼姿勢）
1. 依照畫面上提示，先將手臂關節擺至零點／原點姿勢。
2. 按 Enter 後，依提示手動移動各關節至最大與最小極限位置。
3. 完成校正，設定檔將自動寫入 `./calibration/<YYYY-MM-DD>_<leader|follower>.json`。

產出檔案：`calibration/2026-08-31_leader.json`、`calibration/2026-08-31_follower.json`

★ **收工前 Git 提交**：校正完成後執行 `git add calibration/ && git commit -m "chore(calibration): 2026-08-31 calibration"` 鎖定當日校正版本。

---

## 4. 遙操作測試 teleoperate (純手動跟隨測試，不存檔)

> **`lerobot-teleoperate` vs `lerobot-record` 的差別：**
> * **`lerobot-teleoperate`（本節）**：純粹的**即時硬體聯動測試（Dry-run）**。Leader 讀取姿態並即時傳送給 Follower。**不啟動資料儲存、不錄製影片、不產生 dataset**。
>   - **主要用途**：在正式錄製前，檢查雙臂跟隨是否即時、馬達旋轉方向有無相反、關節有無卡頓、相機即時畫面是否正常。
> * **`lerobot-record`（第 5 節）**：完整的**示範資料收集管線**。包含遙控操作，同時以 30 FPS 錄製多視角影片與 6 軸狀態，支援鍵盤互動標註與存檔。

指令（使用 YAML 設定檔）：
```powershell
# SO-101 雙臂
uv run lerobot-teleoperate --config_path configs/teleoperate.yaml

# 或 OMX 雙臂
uv run lerobot-teleoperate --config_path configs/teleoperate_omx.yaml
```
（對應配置檔 `configs/teleoperate.yaml` / `configs/teleoperate_omx.yaml`，包含 Leader `COM6` 與 Follower `COM5`）

確認項目：
- Follower 是否即時且平滑地跟隨 Leader？
- 各關節運動方向是否與操作者直覺一致？
- ★ 如果有固定角度偏移，立刻記下偏移量——這是診斷線索。

### 🔴 4-1. 偏移量到底怎麼量（具體程序）

> **這個記錄的價值不在當下判斷好壞，而在未來出問題時可以比對。**
> 第一次量到的數字**就是基線**——所以重點是「量得可重複」，不是「達到某個門檻」。
> ⚠️ **不要先設門檻再量**。先建立基線，之後偏離基線才是訊號。

#### ⚡ 推薦做法：使用自動化取樣量測腳本（一鍵量測＋自動存檔）
我們提供了專門的半自動化量測腳本，操作者只需擺好姿態按 `Enter`，腳本會**自動抓取 30 幀數據做平均濾波、計算 Delta 並直接生成 CSV 報表**：

```powershell
# 執行 OMX 雙臂偏移量量測
uv run python scripts/measure_teleop_offset.py --config-path configs/teleoperate_omx.yaml

# 或 SO-101 雙臂
uv run python scripts/measure_teleop_offset.py --config-path configs/teleoperate.yaml
```

**五個引導測試姿態**：
1. `1_home`：**Home／零點姿態**（靜態偏移基準）
2. `2_j1_mid`：只轉 **J1（基座旋轉）** 到約中點
3. `3_j2_mid`：只轉 **J2（肩）** 到約中點（負重最大軸）
4. `4_j3_mid`：只轉 **J3（肘）** 到約中點
5. `5_gripper`：**夾爪全開 / 全閉**

產出檔案：自動存檔至 `analysis/teleop_offset_<YYYY-MM-DD>.csv`。

---

#### 🔍 備用做法：純手動由 Rerun 折線圖讀值
若使用 `uv run lerobot-teleoperate --config_path configs/teleoperate_omx.yaml`：
- 在 Rerun 下方 Time Series 面板展開 `action` (Leader) 與 `observation` (Follower)。
- 手臂移動至指定姿態靜置 3 秒後，點擊時間軸讀取數值並手動填入 CSV。

**要判讀的三種病徵：**

| 病徵 | 數字長什麼樣 | 意義 |
|---|---|---|
| **方向相反** | delta ≈ −2 × leader 值 | 該軸符號接反 → 檢查 config 的 joint 定義 |
| **固定偏移** | 各姿態 delta 幾乎相同 | 零點沒對準 → 回頭檢查校正檔 |
| **比例不對** | delta 隨 leader 值等比放大 | 齒輪比或單位換算錯 |
| （正常的背隙） | delta 小且隨方向改變正負號 | ⚠️ **這是 SO-101 的先天特性，不是故障** |

> ★ **不論結果好壞都要存檔並 commit。** 之後若某批資料訓出來特別差，
> 這份基線是唯一能回答「是不是那天硬體跑掉了」的東西。

---

## 4-2. 🔴 固定 follower 之後、凍結場景之前：工作範圍驗證

> **這是唯一「還能免費改」的時間點。過了場景凍結，任何改動的代價就是先前資料作廢。**
>
> **順序很重要：校正（第 3 節）要在空曠處做**，因為全範圍掃描要能安全推到機械死點；
> 若手臂已鎖在工作區、旁邊有目標箱與相機腳架，掃描時撞到就會記錄到錯誤極限值 → **整台校準失效**（已知地雷第一條）。
> **校正是關節座標系內的事，與基座在世界座標的位置無關 —— 先校正、再固定，不會讓校正失效。**

### 這一節做什麼（A6 工作範圍驗證 + A7 場景常數紀錄）

分兩階段，**順序不可顛倒**：

| 階段 | 內容 | 用什麼 |
|---|---|---|
| **A — 量測 → 凍結 → 印墊** | 量工作範圍、抽 90 個配置點凍結、印對位墊 | 三支腳本 **S1 → S2 → S3** |
| **B — 逐點實體驗證** | 把手臂實際帶到每個凍結點，檢查構得到／下爪幾何／關節餘裕／路徑碰撞／相機取景 | **手動 teleop + rerun**。**不是腳本** —— 碰撞、下爪可行性、相機取景要人看畫面判斷；D023 只規定 3 支腳本，沒有第 4 支「驗證器」 |

**共同前置條件：**

- follower 已夾固定、相機架好並用膠帶標記位置角度、**校正在空曠處做完**（第 3 節；固定後不可重校）。
- 🔴 **D023：傳輸線已改走線** → 保守工作區豁免作廢（改走線非單調放寬），這一節從「A7 設計」升級為**必做的 A6 量測 ＋ 逐點驗證**。
- 相機設定檔對應**現在實體接的相機**：D405 在腕上 → 用含 `wrist` key 的檔（`configs/teleoperate_omx_two-third-pov-cams.yaml`）；已卸下回雙第三視角 → `configs/teleoperate_omx.yaml`。

**2026-08-31 現場量到的（捲尺，FK 失敗退 D026；experiment_spec §3 正本）：**

- `d_offset`（pan 軸 → 底盤最底端）≈ **5 cm** → 半徑量值都要加回：`r_outer` top-down ≈ **41**、side-only ≈ 49、`r_inner` ≈ **22 cm**。
- 有效方位角扇區 ≈ **135°**：`theta ∈ [−90°, +45°]`（正前 = 0°、+ = 左）。
- 🔴 扇區邊界成因 = **相機/支架的架設位置擋路 —— 手臂轉過去會實體撞到**（硬機械限制，不是相機視野、不是傳輸線）。→ teleop / S1 時也不要把手臂轉進那個角度。綁定現在的相機位置，相機一移要重量。
- 🔴 **0° 基準線是「選定 + 標示」，不是量測**：挑一個可複現的方向定成 0°（最好 = `shoulder_pan = 0` 時手臂指的方向，隨時能從手臂復現；或固定特徵如操作者座位中線），從 pan 軸往外用膠帶拉一條線、**拍照並註明對齊什麼**。之後放物體時用量角器從這條線量 `theta`。
- 🔴 **範圍**：以上是 **Phase A pilot layout** 的計算。Phase B 上車體要用真實布局**重跑 S1/S2**（`[Eric說]` 2026-08-31）。pilot 的 N 可調。

---

### 階段 A — 量測、抽樣、印墊（腳本 S1 / S2 / S3）

#### A-1. S1：量工作範圍 — `scripts/reach_logger.py`

```powershell
# 先乾跑：確認 port / URDF 路徑 / 輸出檔名，不碰硬體
uv run python scripts/reach_logger.py --dry-run

# 正式跑（預設 --mode teleop：腳本自己跑 leader→follower 迴圈）
uv run python scripts/reach_logger.py
#   不想用 leader：--mode follower-only（整條手臂會癱，D405 若在腕上會下墜，先用手托住）
```

**這支腳本不會自己動手臂。** `--mode teleop`（預設）只是**把你手動扳的 leader 即時鏡射到 follower**（迴圈每輪 `follower.send_action(leader.get_action())`，`--fps 30`），按鍵時讀 follower 當下關節 → FK → 記一筆。leader 大部分關節可徒手扳（夾爪是彈簧扳機）。`--mode follower-only` 則連 leader 都不連、follower 關扭力徒手擺。

**5-pose FK 驗證（進主迴圈前強制一次）**：畫面會逐一提示 `move the leader to [home / +J1 / +J2 / +J3 / gripper]`，按 Enter 後印 FK 預測的 EE (x, y)，再要你輸入捲尺量到的 `measured x cm` / `measured y cm`。

- 🔴 **`--mode teleop` 下，驗證階段現在會持續把 leader 鏡射到 follower**（`_pump_until_enter`，2026-08-31 修）。舊版只在按鍵時 `read_ticks()`、不 servo → follower 被扭力鎖死、擺 leader 沒反應。若你的版本沒這行為 → 先 `git pull` 拿修正，或改跑 `--mode follower-only`（follower 關扭力徒手擺，⚠️ 整臂會軟、D405 在腕上會墜，先托住）。
- **座標系（`reach_logger/fk.py`）**：原點在**底座**（桌面上正對底座旋轉中心那點）；`x` = **正前方**（全關節歸零時夾爪指的方向，零位 EE ≈ (31.3, 0, 21.1) cm）；`y` = **水平側向，⟂ x**，右手系 z 朝上 → **+y = 左、−y = 右**（站底座後方沿 +x 看）；`z` = 上（不進半徑／方位角）。`azimuth 0° = 正前、+90° = 左、−90° = 右`。
  → 量測：先看螢幕印的 FK 預測 (x, y)，轉 J1 一邊看預測 y 往 + 還是 −，捲尺符號照著配；不確定就記大小＋註明左右。
- **在 `home` 那格直接按 Enter（x 留空）= 跳過整個驗證 → 走捲尺模式。**
- 五格做完印對照表，最大誤差 > 1 cm → 自動切 `--fk-fallback tape`（之後 `o/s/i` 會多問一句捲尺讀數）。
- ⚠️ 每軸 offset/sign 目前是 identity（follower 是出廠校正，URDF 零 ≠ encoder 零）。誤差若是**固定偏移**可先修進校正再判；只是純粹對不上就讓它退捲尺，你有帶捲尺。
- 螢幕提示只講「擺到 +J1」，**不會告訴你轉哪一軸、轉幾度** —— 那在 `docs/specs/S1_reach_logger.md` §5：+J1 只轉 `shoulder_pan`、+J2 只轉 `shoulder_lift`、+J3 只轉 `elbow_flex`，各轉一個已知量（如 +30°），夾爪格只開闔、EE 不該動。

**主迴圈按鍵 —— 各要按幾次**（summary 的化簡方式決定的，不是全都「只按邊界」）：

| 鍵 | 樣本 | 按幾次 | summary 怎麼取 |
|---|---|---|---|
| `o` | outer / top-down | **掃整個扇區、多點**（如 5–7：兩邊緣＋中間＋之間幾點） | **取所有 `o` 的最小半徑** 當 `r_outer`。只按一次 = 沒有最差方向保護 |
| `i` | inner 可下爪最近 | 多點（如 3–5，跨扇區） | **取所有 `i` 的最大半徑** 當 `r_inner`（內孔取最嚴方向） |
| `a` | 方位角邊界 | **只按邊界**：手臂快撞到相機/支架前的最大安全轉角，左右各一 → 2 次（傳輸線若另有更早卡點也記） | 取 `min` 與 `max`。中間按了不影響結果，但別按 |
| `s` | side-only 最遠 | 少數幾次或略過（僅供參） | 取最大 |
| `r` | 墊子基準 | **正好一次**（墊子對齊底座後，讀手臂指向的墊子角） | **只用第一筆** |
| `q` | 存檔離開 | — | — |

- 🔴 `a` 的備註寫清楚每個邊界是什麼卡住的：`arm hits camera mount`（手臂實體撞到相機/支架 —— 2026-08-31 就是這個，≈ ±135° 那組）vs `cable`（傳輸線）。餵 S2 的扇區取所有來源裡最窄的。reach_logger 不開相機，但相機/支架是實體障礙、肉眼看得到 → 轉到快撞前停、按 `a`。
- 每按一次鍵就 flush 寫 CSV（現場斷電不丟資料）；同日重跑不覆寫（檔名加 `_2`）。
- 產出：`analysis/reach_log_<date>.csv`、`analysis/reach_summary_<date>.json`（餵 S2）、`analysis/reach_plot_<date>.png`。
- ⚠️ 腳本**不會**自動寫 `experiment_spec.md` §3 —— 人看過摘要、扣掉 margin 再手動填（S1 spec §11）。

#### A-2. S2：凍結 90 個配置點 — `scripts/sample_placements.py`

```powershell
# 先 --dry-run：只看可行性與最近鄰距離統計，不寫檔
uv run python scripts/sample_placements.py `
    --from-summary analysis/reach_summary_2026-08-31.json `
    --margin 2.0 --d-min 2.0 --seed 20260831 `
    --n-train 50 --n-eval-open 10 --n-eval-close 30 `
    --label campaign_A_pilot_2cam --dry-run

# 可行 → 拿掉 --dry-run 正式凍結
```

- `--margin`：從 `r_outer` 與**兩側方位角**各扣掉的 cm（與 `--from-summary` 併用時必填）。**S1 不自動扣，扣在這裡。**
- `--d-min`：任兩點（含跨清單）最小間距，**無預設**（D024 §7）。
- `--seed` ＋ 檔名帶日期 → 同 seed 重跑位元組相同；輸出檔已存在時**預設拒絕覆寫**（要 `--force`）。
- ⚠️ **`d_min` 有 packing 上限。** 改走線後 S1 量到的實際扇區 × `r≈20–30`，若塞不下 90 點 @ `d_min=2`，`--dry-run` 會判不可行 → 調 `--margin` / `--d-min` / N，**不要用半套清單進階段 B**（`docs/decisions.md` D024 §2026-08-30；memory `s2-provisional-sector-infeasible`）。
- 產出：`configs/placements/campaign_A_pilot_2cam_<date>_train.csv`（50）、`…_eval-open.csv`（10）、`…_eval-close.csv`（30），欄位 `placement_id,r_cm,theta_deg,x_cm,y_cm`，＋ `…_meta.json`。

#### A-3. S3：印 A4 拼貼對位墊 — `scripts/make_placement_mat.py`

```powershell
uv run python scripts/make_placement_mat.py `
    --from-summary analysis/reach_summary_2026-08-31.json `
    --placements configs/placements/campaign_A_pilot_2cam_2026-08-31_train.csv `
                 configs/placements/campaign_A_pilot_2cam_2026-08-31_eval-open.csv `
                 configs/placements/campaign_A_pilot_2cam_2026-08-31_eval-close.csv `
    --label campaign_A_pilot_2cam --out analysis/placement_mat_2026-08-31.pdf
```

- `--from-summary` 帶入 `azimuth_offset` → 墊子用**墊子框架**，和 S2 的 `theta_deg` 對得上。
- 逐頁 A4 拼貼：**極點對齊底座旋轉中心、0° 線對齊 S1 `reference` 方向**。
- 墊子只在放物體時用；**錄製時移走**（留在畫面 = 背景常數，D023 腳本 3）。

---

### 階段 B — 逐點實體驗證（手動 teleop + rerun）

**為什麼不是腳本**：碰撞、下爪幾何可行性、相機取景都要人看 rerun 判斷。這是場景凍結前**最後一次能免費改**的檢查。

#### B-1. 起 teleop（不錄製）

```powershell
uv run lerobot-teleoperate --config_path configs/teleoperate_omx.yaml
```

- 設定檔內 `display_data: true` → 自動開 rerun。只做即時聯動，**不存檔**（與第 5 節 `lerobot-record` 的差別見本節開頭）。
- 相機一改 → 下面 ⑤⑥ 整批重驗。

#### B-2. rerun 怎麼讀關節值

- 左側 entity tree：`observation.images.<cam>`（每台相機一路影像）、`observation.state`（follower 六軸）、`action`（leader 六軸）。實際字串以樹上為準。
- 底部 Time-series：展開 `observation.state`，六條線 = `shoulder_pan / shoulder_lift / elbow_flex / wrist_flex / wrist_roll / gripper`。
- 值域：手臂軸 **−100 ~ +100**、夾爪 **0 ~ 100**。⚠️ follower 校正是出廠預設滿轉，**±100 ≠ 機械死點**。
- （建議先跑一次）`uv run lerobot-find-joint-limits --robot.type=omx_follower --robot.port=COM5 --robot.id=2026-08-31_omx_follower --teleop.type=omx_leader --teleop.port=COM6 --teleop.id=2026-08-31_omx_leader --urdf_path=assets/omx_f/omx_f.urdf --target_frame_name=end_effector_link --teleop_time_s=60`
  → 掃一遍全工作區，記下每軸實際到過的 min/max，當成 ③ 的行程邊界基準。

#### B-3. 逐點程序（三份清單裡每一個 `placement_id`）

1. 依 `(r_cm, theta_deg)` 用對位墊放標記物。
2. leader 把 follower 夾爪帶到該點正上方、擺可下爪姿態，靜置 2–3 秒。
3. 六項檢查（下表），任一項不過 → 記錄、標該點 ✗。
4. 回中位再做下一點；**先驗最外圈 `r` 與最偏方位角的點**（① ③ 最可能在那裡爆）。

| 檢查項 | 具體怎麼判 | 不過的意義與處置 |
|---|---|---|
| **① 構得到** | 夾爪能否實際移到該點正上方的下爪位 | 構不到 → 點在 `r_outer` 外或落在改走線受限的方位角。**多點如此 = S1 的 `r_outer` 量太大 → 回頭修 `reach_summary` 重跑 S2**，不要硬凹 |
| **② 夾爪朝向** | 該點能否擺出「垂直／斜向下爪、夾爪軸對準物體」的姿態 | 不行（通常近 `r_outer` 只能側夾）→ 抓取幾何不可行，違反任務定義 → 該點作廢 |
| **③ 關節餘裕 ⭐** | 讀 `observation.state`，看該姿態各軸離 B-2 記下的**實測行程邊界**多少（無實測值時退而看是否逼近 ±100 / 夾爪逼近 0 或 100） | 任一軸 ≥ 行程 90% → 危險：policy 推論時**一定會偶爾推過去** → 撞限位／馬達過熱／動作被截斷，且你會誤判成模型問題 → 該點作廢或縮 `r` |
| **④ 路徑無碰撞** | 從起始姿態 → 該點 → 目標區，**慢速**走一遍 | 撞到目標箱／相機腳架／線材／改走線後的線 → 重新配置場景，**回固定步驟** |
| **⑤ 各相機可見** | rerun 每一路 `observation.images.*` 裡，該點物體都在框內 | 看不到 → 調相機角度，**但要維持可複現約束（P7）**；調完 ①–④ 不用重驗，**⑤⑥ 全部重驗** |
| **⑥ 起始姿態可見** | 手臂回起始姿態時，該點物體仍在每一路相機框內 | 看不到 → 違反任務定義（`experiment_spec` §1-1）→ 該點作廢或改起始姿態。⚠️ 相機視野若比 A-1 量的 135°（撞擊界）還窄 → 用較窄者重跑 S2 |

#### B-4. 記錄 → `docs/setup_env.md` 新一節「工作範圍驗證 2026-08-31」

```
| placement_id  | r_cm | θ_deg | ①構得到 | ②朝向 | ③最小關節餘裕      | ④路徑 | ⑤各相機可見 | ⑥起始可見 | 判定 |
|---------------|------|-------|--------|------|------------------|------|-----------|----------|-----|
| train_00      | 21.4 | -18.0 |   ✓    |  ✓   | wrist_flex 剩 22% |  ✓   |     ✓     |    ✓     | OK  |
| train_01      |      |       |        |      |                  |      |           |          |     |
| ...           |      |       |        |      |                  |      |           |          |     |
| eval-close_29 |      |       |        |      |                  |      |           |          |     |
```

- 三份清單**全部**走完。
- ✗ 的點**不要就地刪**：收工後決定「縮 margin 重跑 S2」還是「該 campaign 少幾點」。改凍結清單要留痕跡（S2 §2-2）。

#### B-5. 通過門檻 → 場景凍結

- 三份清單每點都 OK，或 ✗ 點已在 `setup_env.md` 標明並有處置決定。
- follower 位置／相機／目標箱／改走線的線路 **此後凍結**；之後任何改動 = 先前資料作廢。
- 最終 `r_outer / r_inner / 受限方位角 / margin` 回填 `docs/experiment_spec.md` §3，並在 `docs/setup_env.md` change log 記一行。

> 🔴 **③ 關節餘裕是最容易跳過、也最容易咬人的一項。**
> 某點需要某軸推到行程 95%，policy 推論時**一定會偶爾超過** → 撞限位、馬達過熱、動作突然截斷，
> 而你會以為是模型的問題。這一項寧可嚴，不要放水。

---

## 5. 錄製 demo (示範資料收集管線)

指令（使用 YAML 設定檔）：
```powershell
# SO-101 雙臂 + 雙 RealSense 相機
uv run lerobot-record --config_path configs/record.yaml

# 或 OMX 雙臂 + 雙 RealSense 相機
uv run lerobot-record --config_path configs/record_omx.yaml
```
（對應配置檔 `configs/record.yaml` / `configs/record_omx.yaml`，包含 Leader `COM6`、Follower `COM5`、雙 RealSense 相機與 dataset 設定。**OMX 過渡配置：`wrist` = D405（SN 260322271459）、`front-left` = D455（SN 262822305610），D022 §2026-09-01**）

### (0) 🔴 相機場景常數：錄製第一筆前一次調定、凍結

**這些一旦錄了第一筆就不能改**（改了＝先前資料作廢，D004）。**每個 campaign 開錄前調一次、寫進 `experiment_spec.md` §3、整個 campaign 不動：**

| 參數 | 在哪設 | 說明 |
|---|---|---|
| `exposure` / `gain` / `white_balance` | `configs/record_omx.yaml` 每台相機區塊 | `None` = 自動。**要設固定值**——自動曝光會讓「同一個場景在不同時間看起來不一樣」，擴大 domain gap（`camera_mount.md` §5-4）。🔴 **D405 在腕上近距離自動曝光會過曝（發白）**，一定要手動壓 |
| `width` / `height` / `fps` | 同上 | 解析度、幀率。RealSense RGB 原生檔位：6/15/30 fps |
| 相機位置 / 角度 / 外參 | 實體 + 膠帶標記 | 拆裝後要對得回基準照 |
| 相機數量 + feature key 順序 | `cameras:` 宣告順序 | D022；宣告順序 = 模型看到的張量順序，接錯不會報錯 |

**調定曝光的步驟（下次 lab day，需手臂 + 相機在場）：**

1. 接好目標相機，起 teleop：`uv run lerobot-teleoperate --config_path configs/teleoperate_omx.yaml`（`display_data: true` → rerun）。
2. 看 rerun 的 `observation.images.wrist` / `.front-left`。過曝（白到看不出細節）或欠曝（暗）都不行；顏色要中性、不偏藍/偏黃。
3. 停 teleop，編輯 `configs/record_omx.yaml`，給 `wrist` 一組保守值（例：`exposure: 80` 微秒級往下調變暗、`gain: 16`、`white_balance: 4500` K 偏藍調高偏黃調低），重起 teleop 看效果、反覆逼近。**用 RealSense Viewer 找值更快**（關自動曝光、拉手動滑桿試、記下數字）。
4. 兩台都調好 → 把最終數值填進 `configs/record_omx.yaml` **和** `docs/experiment_spec.md` §3（場景常數表），註明日期。
5. 之後這個 campaign 不再碰。要改 = 新 campaign、重錄。

⚠️ **現有 8 集 pilot 是自動曝光、D405 過曝** —— 改不了（烙進影片），但那是煙霧測試。正式 pilot 一定要先做這一步。

### (1) 錄製時的鍵盤操作控制（Keyboard Controls）
在錄製過程中，請保持終端機處於焦點狀態，使用以下按鍵控制每集流程：

| 按鍵 | 字母等效鍵 | 功能說明 | 使用時機 |
| :--- | :--- | :--- | :--- |
| **`Right Arrow` (→)** | **`n`** (Next) | **結束並保存本集**，進入下一集 | 示範成功完成夾取放置任務後按下。 |
| **`Left Arrow` (←)** | **`r`** (Re-record) | **廢棄本集並重新錄製** | 操作失誤、物體意外掉落、碰撞卡住時按下，該集不計入資料集。 |
| **`Esc`** | **`q`** (Quit) | **停止錄製並完成存檔** | 完成目標集數（如 20 集）或需中途暫停時按下。 |

---

### (2) 示範資料存到哪？（依據專案規範 `README.md` 與 `docs/environment.md`）

專案嚴格遵循「**資料不進 Git**」的原則：
1. **雲端主儲存**：自動上傳至 **Hugging Face Hub (Private Repo)**：`<HF_USER>/so101_pick_place`（需事先於環境中設定 `HF_TOKEN`）。
2. **本機快取目錄**：暫存於專案的 `.cache/lerobot/<repo_id>`（或 `$HF_LEROBOT_HOME`），內含 Parquet 特徵表與 MP4 影片。
3. **離線安全備份**：收工後手動將資料夾打包複製至 **Lab NAS / 隨身硬碟**。

---

### (3) Recovery Demo 收集規範（見 `docs/decisions.md` D005 與 RaC 原理）

錄製 demo 時**必須包含 recovery demo（正常 : recovery 約 7 : 3）**，嚴禁只錄成功軌跡：

#### Tier 1（預設：人工示範 7:3 混合錄製）
1. **正常 demo（70%）**：物體在**凍結抽樣清單**的配置點 → 平滑順暢夾取並放置至目標盒中。（原文「3×3 網格」已作廢，D024）
2. **Recovery demo（30%）**：
   - 刻意將機械臂遙操作至**非正常/偏離姿態**（例如夾空、推擠物體使其偏位、手臂停在視野邊緣或過高位置）。
   - 從該異常狀態開始，**手動示範如何修正姿態、重新瞄準物體並完成任務**。
   - ⚠️ **RaC 核心要領（不追求完美動作）**：修正過程**允許次優甚至倒退**（例如夾歪後先退回上方重開夾爪），重點在於向模型示範「如何從異常分佈回到正常分佈」，**切勿在錄製時試圖修剪或美化失敗過程**。

#### Tier 2（條件式升級：RaC, Robot-assisted Correction）
> 僅在第一代 P0 模型訓練完成後、真機部署評估時**夾取成功率過低（<60%）**才啟用。
- **運作模式**：由訓練好的模型自主執行任務，人類在旁待命；當觀察到手臂即將失敗（如偏離軌跡）時，人類立即接管 Leader 手臂進行修正。
- **RaC 兩大鐵律**：
  1. **Rule 1 (Recover then correct)**：人類介入時，先將手臂拉回分佈內的正常狀態，再修正執行至該子任務結束。
  2. **Rule 2 (Termination after intervention)**：修正段落一結束，**該回合必須立刻終止（按 Esc/Right 結束）**。**絕對不可**將控制權交回模型或讓人類繼續做完整個任務，避免混合分佈（Policy + Human mixture）污染訓練資料。

---

## 6. 資料檢查與驗證

### (1) 重播手臂動作 (`lerobot-replay`)
```powershell
uv run lerobot-replay --config_path configs/replay.yaml
```
* **功能說明**：讀取剛錄好的某一集軌跡（如 `episode: 0`），直接將關節位置指令傳送給實體 Follower 機械臂，讓實體手臂在真實空間中**自動原樣重演**一遍動作。
* **檢驗目的**：
  - 檢查真實馬達在執行軌跡時是否有機械卡頓、過衝、抖動或電機發熱失步。
  - 驗證錄下的關節軌跡在物理硬體上是否平滑可行。

### (2) 資料集與多視角影像視覺化檢查 (`lerobot-dataset-viz`)
```powershell
# repo_id 形式 + 本地 root（本專案 dataset 在 .cache/lerobot/，不在 HF cache）
uv run lerobot-dataset-viz `
    --repo-id EricC430/omx_pick_place_pilot `
    --root .cache/lerobot/omx_pick_place_pilot `
    --episode-index 0
```
* **`--episode-index` 是單數、必填 → 一次一集。** 8 集就 `0` 跑到 `7`（Rerun server 固定 :9876，重跑會換）。`--save <path>` 可存檔不開即時視窗。
* **功能說明**：啟動 Rerun，時間軸同步播放多視角相機影片、6 軸 `observation.state` 曲線、`action` 曲線。
* **檢驗目的**：
  - **影像**：掉幀、黑畫面、**曝光過度**（🔴 D405 腕上自動曝光會過曝，正式錄前依 §5-(0) 調定）。
  - **數值連續性**：關節曲線平滑、無突波、時戳無中斷。
* **`torchcodec` 的 `libtorchcodec_coreN.dll` 一整面 traceback 是無害的** → 自動 fallback `pyav`，跑完會顯示 `100%`。

> ⚠️ **視覺化只能抓到「看得出來」的問題。幀數不一致看不出來——它會安靜地存在，直到訓練中途才爆。**
> **所以下面 (3) 是每次錄完都必跑的，不能用肉眼檢查代替。**

### (3) 🔴 實際幀數驗證（每次錄完必跑，不可跳過）

```powershell
uv run python scripts/verify_dataset.py <dataset_root>
# 例：uv run python scripts/verify_dataset.py data/huggingface/lerobot/<HF_USER>/so101_pick_place
```

**為什麼必須用腳本而不是肉眼：**

我們已經被這個咬過一次（`docs/pipeline_validation.md`）——公開資料集
`edgarcancinoe/soarm101_pickplace_orange_080e_ts_closed` 宣稱 61,480 幀、實際 61,534 幀，
**100 步 smoke test 全綠**，跑到 1000 步時第 167 步崩潰：

```
IndexError: Invalid frame index=8530 for streamIndex=0; must be less than 8524
```

**🔴 而且 `--dataset.exclude_episodes` 救不了**——sampler 的索引空間仍由錯誤的總幀數建立，
排除受影響的 episode 只會換成 `KeyError`。**唯一的解是重錄。**

**腳本檢查三件事：**

| # | 檢查 | 抓什麼問題 |
|---|---|---|
| 1 | `info.json` 宣稱總幀數 **==** 各 episode parquet 行數總和 | metadata 與資料本體不一致 |
| 2 | 每支影片**實際逐幀計數** **==** 對應 parquet 行數 | 影片與狀態資料長度不符 |
| 3 | `timestamp` 欄位差分有無異常間隔 | **USB 頻寬不足導致的靜默掉幀** |

> **第 3 項是 USB 掉幀的直接證據。**
> `experiment_spec` §7 已載明「USB 頻寬不足會靜默掉 fps，不會報錯」——
> timestamp 出現大於 1.5 × (1/fps) 的間隔，就是掉了幀。
> **處置：重錄，並把其中一支相機換到不同的 USB 控制器**（不只是同一個 hub 的別的孔）。

⚠️ **腳本用「探索」而非硬編路徑尋找檔案，但 LeRobot 的 dataset 結構在版本間會變動。**
**第一次跑要確認它真的找到了檔案（輸出會列出 episode 數與幀數），不要盲信 PASS。**
實測版本：lerobot 0.6.2。

**前置需求：** `ffprobe`（隨 ffmpeg 安裝）。沒有它就無法做檢查 2。

### (4) 逐集標註 per-episode metadata (`scripts/annotate_episodes.py`)

錄完一場、`verify_dataset.py` 過之後跑一次（不是每集跑）。它問 LeRobot 沒記的東西（物體、起始位置、燈光、結果、失敗機制…），欄位全由 `configs/episode_meta_schema.yaml` 定義。

```powershell
uv run python scripts/annotate_episodes.py --dataset ./.cache/lerobot/omx_pick_place_pilot
#   --check            只驗證、不問（看覆蓋率）
#   --redo --episodes 7   重問某一集
#   --set outcome=success --episodes 0-7   批次填、不逐一問
```

- **工作流**：開兩個終端機 —— 一個 `lerobot-dataset-viz --episode-index N` 看那一集，另一個跑 annotate 回答那一集。或憑記憶標（8 集這種規模通常記得）。
- 產出 `episode_meta/<dataset>.csv`（keyed by `episode_index`）。**不進 dataset 本體**，跟 repo 一起版控。
- `sticky` 欄位（燈光、背景、操作者、錄製日期…）預設沿用上一集 → 同場 demo 多半按 Enter。
- 🔴 **`record_ts` 是「錄製那天」的日期，手打**。LeRobot metadata 沒存 wall-clock（只有檔案 mtime，per-file 不 per-episode、複製就掉）。
- 🔴 **pilot 的 `placement_id` 填 `manual` 或留空**（無 frozen 清單，D023 §2026-09-01；schema v3 起不再必填）。
- `object_orientation`：鋁罐才有意義（`label` = 標籤面朝上 = L1；`bare` = 裸鋁面朝上 = L2；用「同一罐翻面」把反光度變成單一控制變因）。其他物體或用別的姿態詞（如 `stand`）時，`values` 只是提示、非 strict，會接受並警告。
- **Windows 注意**：schema/CSV 讀寫已強制 `utf-8`（否則 cp950 會對中文 schema 爆 `UnicodeDecodeError`）。

---

## 7. 收工前確認清單

> 🔴 **第一項是 `scripts/verify_dataset.py`——它抓的是「錄製當下不會報錯、訓練到一半才炸」的問題。
> 詳細說明見上方 §6-(3)。收工前沒跑，等於把問題帶回家。**

- [ ] **跑 `uv run python scripts/verify_dataset.py <dataset_root>`，退出碼必須是 0**

- [ ] 資料已備份（至少兩個地方：HF Hub / Lab NAS / 隨身碟）
- [ ] 當日校正檔已確認在 `calibration/` 資料夾（檔名帶日期）
- [ ] 執行 `git add calibration/ && git commit -m "chore(calibration): YYYY-MM-DD calibration"` 提交校正歷史
- [ ] 器材照片、場地基準照已拍（存到 `docs/setup_env.md` 引用的位置）
- [ ] `docs/hardware.md` 已填（型號、韌體、連接埠 COM Port）

---

## 8. 雲端同步與跨機部署（4090 → 筆電 → 真機）

**目的：** A11/A12 已在 4090 本機完成訓練與開環評估，但 A13「部署在手臂推論」要在**筆電**上對**真機**
下指令。模型只存在 4090 磁碟不夠——兩台機器需要一個交接點，用 HF Hub 的 private repo 扮演這個角色，
不必手動複製檔案或靠隨身碟版本對不上。

以下指令都用 `hf --help`（pinned 容器內 `huggingface_hub` 1.26.0）逐一核對過，不是憑記憶寫的。

### (1) Token 設定（兩台機器都要做，一次性）

1. 到 https://huggingface.co/settings/tokens 建一個 **write** 權限的 token。
2. **4090（訓練端，容器內）：**
   ```bash
   export HF_TOKEN="hf_xxx"      # 寫進 ~/.bashrc，之後每次開 shell 自動有
   ```
   `scripts/run_container.sh` 已把 `HF_TOKEN` 透傳進容器（比照 `WANDB_API_KEY` 的模式，見
   `docs/environment.md`）；容器內 `HF_HOME=/workspace/data/huggingface`，token 落在 `data/` 下，
   `.gitignore` 已排除，不會進 git。或用互動式登入（存到同一個路徑，效果一樣）：
   ```bash
   ./scripts/run_container.sh hf auth login
   ```
3. **筆電（部署端）：** 先照 §2-(3) 把 `HF_HOME` / `HF_LEROBOT_HOME` 重定向到專案內 `.cache/`，
   再登入：
   ```powershell
   $env:HF_HOME = "$PWD\.cache\hf"
   $env:HF_LEROBOT_HOME = "$PWD\.cache\lerobot"
   uv run hf auth login
   ```
   （若 `hf` 指令不存在，代表這台的 `huggingface_hub` 版本較舊，改用 `huggingface-cli login`，
   語法相同。）

**絕對不要**把 token 寫進任何 `configs/*.yaml` 或貼進 commit —— 見 `docs/conventions.md`
「Never commit」一節。

### (2) 從 4090 push 目前的 pilot checkpoint（+ 資料集）

`configs/record_omx.yaml` 原本標記 `omx_pick_place_pilot` 「local only, do NOT push」——因為那
是 smoke-test/管線驗證用資料，不是正式 campaign。**現在要驗證的是「4090 訓練 → 筆電部署 → 真機
動作」這條交接鏈路本身**，所以把這組已經訓練、驗證過的模型（和它對應的資料集，方便回溯）實際搬
上 Hub 一次；這跟「是否升級成正式資料」是兩件事——正式 campaign 仍照原計畫另開新 `repo_id`。

```bash
# 模型 checkpoint（A11 訓練出的 pilot ACT 模型）
./scripts/run_container.sh hf upload ericc430/act_omx_pick_place_pilot \
  data/train/phase_a_pilot/checkpoints/last/pretrained_model \
  --repo-type model --private \
  --commit-message "Phase A pilot ACT -- 500 steps, eval_loss 0.4866, open-loop MAE 11.29 deg (ep7)"

# 資料集（可選，用於留存/回溯；部署真機不需要它，只有 eval_open_loop.py 需要）
./scripts/run_container.sh hf upload ericc430/omx_pick_place_pilot \
  data/huggingface/lerobot/EricC430/omx_pick_place_pilot \
  --repo-type dataset --private
```

`--private` 只有在 repo 尚未存在時才會套用（`hf upload --help` 原文：「Ignored if the repo
already exists」）——如果不小心先建成 public，要自己到 Hub 網站的 repo 設定頁改回 private。

### (3) 在筆電上 pull 並對真機跑推論

`lerobot-rollout` 是這個 pinned 版本裡真正會驅動真機的指令——`lerobot-record` 的 `teleop` 是
必填、無法用 policy 取代；`lerobot-eval` 只接受模擬環境（`env.type` 只有 `aloha` / `pusht` /
`libero` 等，沒有真機選項）；兩個都用 `--help` 核對過，都不適用。`lerobot-rollout` 吃
`--policy.path=<repo_id>`，會自動用登入的 token 從 Hub 拉取並快取，不必手動 `hf download`。

`configs/rollout_omx_pilot.yaml`（已建立，機型/相機沿用 `configs/record_omx.yaml`）：

```powershell
uv run lerobot-rollout --config_path configs/rollout_omx_pilot.yaml
```

🔴 **第一次真機跑推論：手放在緊急停止/斷電開關上，`--duration` 先設短**（config 裡預設 20 秒，
確認動作方向合理再拉長）。**開環 MAE 11.29°（A12）是「軌跡跟真人示範差多少」，不是「閉環會不會
撞」的保證**——那是本節要驗證的，兩者是不同的失敗模式。

### (4) 驗證完成的判準

- [ ] 筆電上 `uv run hf auth whoami` 能認得到帳號（token 生效）
- [ ] `lerobot-rollout` 成功從 Hub 抓到 checkpoint（log 會印 repo_id）
- [ ] 手臂真的依照畫面前方物體位置動作，不是原地不動或亂動
- [ ] 至少完整跑完一次 `--duration` 區間，沒有觸發緊急停止
- [ ] 觀察到的失敗模式（若有）記錄回 `docs/pipeline_validation.md` 或 `eval/`，供下一輪比較

---

**現場護欄：** 同一個問題卡超過 90 分鐘就跳過，記錄下來，往下一項走。現場時間太貴，不要拿來 debug 單一問題——那件事回家也能做。

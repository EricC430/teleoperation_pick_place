# 現場操作手冊 v1（2026-08-14 更新）

> ⚠️ 這份要離線可查（印出來，或存成手機／筆電裡的離線檔）。現場網路不一定穩。
>
> ⚠️ 本手冊所有路徑皆以專案根目錄 `teleoperation_pick_place/` 為基準。
> ⚠️ LeRobot CLI 版本：**0.6.2**（PyTorch `2.11.0+cu126` / Python `3.12.13` on Windows 11 / Linux）。
> 到現場裝好後，**第一件事是跑 `--help` 確認實際參數**，再據以修正這份手冊。

---

## 0. 檢查點清單

- [ ] 手臂 leader / follower 各一，配對正確（SO-101）
- [ ] 電源供應器規格正確（12V）、已通電
- [ ] USB hub 已接、裝置都被辨識到（COM port & USB Cameras）
- [ ] 相機已被系統與 LeRobot 辨識到（Wrist + Front）
- [ ] 目標物體已備妥（見 `docs/experiment_spec.md` §2 物體清單）

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
- `COM3`, `COM4` 等（依實際插入 USB 序列埠而定；例如 `COM3` 為 Leader，`COM4` 為 Follower）。

如果沒看到：
→ 檢查 USB 線材、電源轉換板燈號、Windows 裝置管理員（`devmgmt.msc`）是否有未安裝驅動程式的裝置（如 CH340 / FTDI / CP210x）。

### (2) 檢查相機 (Webcam / OpenCV / RealSense)
查詢 USB 視訊鏡頭：
```powershell
uv run lerobot-find-cameras opencv
```
預期看到：
- `Camera #0: OpenCV Camera @ 0`（例如 Wrist 相機）
- `Camera #1: OpenCV Camera @ 1`（例如 Front 相機）
- 拍攝的測試照片會自動存入 `outputs/captured_images/`。

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
> 💡 協作者的本機環境可能將 `HF_HOME` 設於外接硬碟（如 D 槽）或其他非預設路徑。為避免路徑未掛載導致 `WinError 3`，且不干擾個人其他專案，可在當前終端機將專案快取重定向至專案內部：
```powershell
# Windows PowerShell（僅對當前視窗生效）
$env:HF_LEROBOT_HOME = "$PWD\.cache\lerobot"

# Linux / macOS Bash
export HF_LEROBOT_HOME="$PWD/.cache/lerobot"
```

---

## 3. 校正 calibration ★最重要

> 💡 **最佳設計（零覆蓋、資料與校正強綁定，見 `docs/decisions.md` D017）：**
> 所有設定檔存放於 `configs/`。
> 我們將「當天日期」直接寫入 YAML 的 `id` 欄位（例如 `id: 2026-08-14_leader` 與 `id: 2026-08-14_follower`），LeRobot 校正完成後會**自動生成帶日期的檔案**（如 `calibration/2026-08-14_leader.json`），免手動改名且永遠不被覆蓋。

指令（使用 YAML 設定檔）：
```powershell
# 1. 校正 Teleop / Leader 手臂 (SO-101 Leader，預設 COM3)
uv run lerobot-calibrate --config_path configs/calibrate_leader.yaml

# 2. 校正 Robot / Follower 手臂 (SO-101 Follower，預設 COM4)
uv run lerobot-calibrate --config_path configs/calibrate_follower.yaml
```

> ※ 若換日收集資料，請在 `configs/*.yaml` 中將 `id` 改為當日日期（例如 `2026-08-15_leader`）。
> ※ 若現場 COM 埠有所變動，可直接編輯 YAML 或在 CLI 尾端覆蓋參數（例如 `--teleop.port=COM5`）。

步驟：（逐步，含手臂要擺什麼姿勢）
1. 依照畫面上提示，先將手臂關節擺至零點／原點姿勢。
2. 按 Enter 後，依提示手動移動各關節至最大與最小極限位置。
3. 完成校正，設定檔將自動寫入 `./calibration/<YYYY-MM-DD>_<leader|follower>.json`。

產出檔案：`calibration/2026-08-14_leader.json`、`calibration/2026-08-14_follower.json`

★ **收工前 Git 提交**：校正完成後執行 `git add calibration/ && git commit -m "chore(calibration): 2026-08-14 calibration"` 鎖定當日校正版本。

---

## 4. 遙操作測試 teleoperate (純手動跟隨測試，不存檔)

> **`lerobot-teleoperate` vs `lerobot-record` 的差別：**
> * **`lerobot-teleoperate`（本節）**：純粹的**即時硬體聯動測試（Dry-run）**。Leader 讀取姿態並即時傳送給 Follower。**不啟動資料儲存、不錄製影片、不產生 dataset**。
>   - **主要用途**：在正式錄製前，檢查雙臂跟隨是否即時、馬達旋轉方向有無相反、關節有無卡頓、相機即時畫面是否正常。
> * **`lerobot-record`（第 5 節）**：完整的**示範資料收集管線**。包含遙控操作，同時以 30 FPS 錄製多視角影片與 6 軸狀態，支援鍵盤互動標註與存檔。

指令（使用 YAML 設定檔）：
```powershell
uv run lerobot-teleoperate --config_path configs/teleoperate.yaml
```
（對應配置檔 `configs/teleoperate.yaml`，包含 `so101_leader` COM3 與 `so101_follower` COM4）

確認項目：
- Follower 是否即時且平滑地跟隨 Leader？
- 各關節運動方向是否與操作者直覺一致？
- ★ 如果有固定角度偏移，立刻記下偏移量——這是診斷線索。

### 🔴 4-1. 偏移量到底怎麼量（具體程序）

> **這個記錄的價值不在當下判斷好壞，而在未來出問題時可以比對。**
> 第一次量到的數字**就是基線**——所以重點是「量得可重複」，不是「達到某個門檻」。
> ⚠️ **不要先設門檻再量**。先建立基線，之後偏離基線才是訊號。

**準備：** `--display_data=true` 會開啟 rerun 視覺化，可同時看到 leader 與 follower 的關節數值。

**五個可重複的測試姿態**（每個姿態靜置 3 秒再讀值）：

| # | 姿態 | 要抓的問題 |
|---|---|---|
| 1 | **Home／零點姿態** | 靜態偏移（最重要的一個） |
| 2 | 只轉 **J1（基座旋轉）** 到約中點 | 該軸方向與比例 |
| 3 | 只轉 **J2（肩）** 到約中點 | 同上，且這軸負重最大 |
| 4 | 只轉 **J3（肘）** 到約中點 | 同上 |
| 5 | **夾爪全開 / 全閉** | 夾爪行程對應是否正確 |

**記錄表格**（存成 `analysis/teleop_offset_YYYY-MM-DD.csv`）：

```csv
date,pose_id,joint,leader_deg,follower_deg,delta_deg,note
2026-XX-XX,1_home,J1,0.0,,,
2026-XX-XX,1_home,J2,,,,
...
```

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

**驗證程序（用 `lerobot-teleoperate`，不錄製）：**

在工作區桌面用膠帶標出**凍結抽樣清單裡的配置點**與**目標區**，然後逐點檢查。

> 🔴 **2026-08-30：原文寫的「3×3 網格」已作廢（D024）。** 配置點來自 §5 的面積均勻抽樣清單。
> ⚠️ **清單尚未產生**（要先量到 r_inner / r_outer / 受限方位角，見 D023）。
> **在那之前，這一節只能用「暫定的幾個代表點」做可達性檢查，不得當成凍結配置。**

| 檢查項 | 怎麼做 | 不通過的意義 |
|---|---|---|
| **① 構得到嗎** | 遙操作把夾爪移到該格的抓取位置 | 構不到 → 基座要移，**回到固定步驟重來** |
| **② 夾爪朝向正確嗎** | 在該格能不能擺出可夾取的姿態 | 不行 → 該格的抓取幾何不可行 |
| **③ 關節餘裕夠嗎** | ⭐ 讀該姿態下各關節值，**離行程極限還有多少** | **若某軸已到行程 90% 以上 → 危險**。policy 執行時會推過去 |
| **④ 路徑無碰撞嗎** | 從起始姿態走到該格，再走到目標區 | 會撞到目標箱／相機腳架／線材 → 重新配置 |
| **⑤ 所有相機都看得到嗎** | 看 rerun 畫面，該點的物體在**當前配置的每一台相機**視野內。⚠️ **2026-08-30：手腕相機 ETA 9/5–9/7，先驗期是「兩台第三視角」（D022）**——不要照抄 wrist/front | 看不到 → 相機角度要調（**但要維持 P7 的可複現約束**） |
| **⑥ 起始姿態可見** | 起始姿態下，該格的目標物在畫面中 | 看不到 → 違反任務定義（`experiment_spec` §1-1） |

**記錄表格**（存成 `docs/setup_env.md` 的一節）：

```
| 格號 | 構得到 | 夾爪朝向 | 最小關節餘裕 | 路徑無碰撞 | wrist可見 | front可見 |
|-----|-------|---------|------------|-----------|----------|----------|
|  1  |   ✓   |    ✓    |  J2 剩 35%  |     ✓     |    ✓     |    ✓     |
|  2  |       |         |             |           |          |          |
...
|  9  |       |         |             |           |          |          |
| 目標區 |     |         |             |           |          |          |
```

> 🔴 **③ 關節餘裕是最容易被跳過、也最容易咬人的一項。**
> 如果某格需要 J2 推到行程的 95%，訓練出來的 policy 在推論時**一定會偶爾超過**，
> 結果是撞限位、馬達過熱、或動作突然截斷——而且你會以為是模型的問題。

---

## 5. 錄製 demo (示範資料收集管線)

指令（使用 YAML 設定檔）：
```powershell
uv run lerobot-record --config_path configs/record.yaml
```
（對應配置檔 `configs/record.yaml`，包含雙臂、雙 OpenCV 相機 Wrist/Front 與 dataset 設定）

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
uv run lerobot-dataset-viz `
    --repo-id <HF_USER>/so101_pick_place `
    --episode-index 0
```
* **功能說明**：啟動 Rerun 視覺化介面，在螢幕上以時間軸形式同步播放錄下的多視角相機影片（Wrist + Front）、6 軸關節角度曲線（`observation.state`）與動作指令曲線（`action`）。
* **檢驗目的**：
  - **影像同步**：確認 Wrist 與 Front 相機有無掉幀、黑畫面或曝光過度。
  - **數值連續性**：確認關節曲線平滑連續，無異常突波或時間戳中斷（Timestamps drift）。

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

**現場護欄：** 同一個問題卡超過 90 分鐘就跳過，記錄下來，往下一項走。現場時間太貴，不要拿來 debug 單一問題——那件事回家也能做。

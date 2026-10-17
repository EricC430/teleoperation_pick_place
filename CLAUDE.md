# CLAUDE.md — 專案指引

> **這份每個 session 開頭都會被讀，所以刻意保持短。**
> **它只負責「指路」與「不變的規則」，不放內容。內容在下面指到的正本裡。**
> 需要細節時再開那份檔，不要預先全讀。

---

## 我是誰、現在在哪

國科會大專生研究計畫（機器手臂 imitation learning，兩人團隊：陳霆翰＋陳柏宇） | 執行至 2027/2/28，報告 3/31。**期望 2026/9/30 前有可展示成果以利書審** |

指導教授：**張嘉惠**（主要往來）、陳弘軒（計畫書掛名協同，非實質參與）

---

## 正本在哪（需要時才開，不要預先全讀）

### A 線 — 本 repo

| 檔案 | 內容 |
|---|---|
| `docs/decisions.md` | ★★ 決策日誌 **D001–D024 ＋ 頂部 Status index**，含「什麼證據會推翻它」。 |
| `docs/phase_plan.md` | ★ v3 階段規劃 Phase A–D |
| `docs/experiment_spec.md` | ★ 實驗規格凍結（任務定義、物體分級、評估協定） |
| `docs/execution_plan.md` | 執行計畫與應變 |
| `docs/act_evidence.md` | ACT 相關文獻與數據佐證 |
| `docs/hardware.md` | 硬體清單與現況（含已否決項目與 ETA） |
| `docs/camera_mount.md` | 相機位置與支架設計、量測協定 |
| `docs/depth_sensing.md` | 深度相機原理對照（被動／主動雙目、ToF、結構光），2026-10-17 查證筆記；本專案 RealSense 全是雙目 |
| `docs/environment.md` | 兩台機器的版本釘選與已知陷阱 |
| `docs/field_manual.md` | 現場操作手冊（Windows/COM port 版指令） |
| `docs/conventions.md` | commit / branch / tag 慣例 |
| `docs/pipeline_validation.md` | 管線驗證資料的處理原則 |
| `docs/exp_a3_recovery_ablation.md` | A3 recovery 消融實驗設計 |
| `docs/meeting/` | 會議與現場工作紀錄（最新：`2026-10-16_closed_loop.md` fldrop50 28/36、fldrop30-augfl（只平移第三視角）34/36，同日同相機（比前幾場水平轉約 4°、亮度 168）；augfl 對 fldrop50 配對 p 0.07、對 fldrop30 p 0.22；離線 h1／開環都沒預測到 augfl 最好；fldrop50 中途馬達全斷線、用 `--resume=true --dataset.root=…` 續錄；下一步記在 §4（只用 uvc_60＋augfl-fldrop30 訓練、手腕瞄準杯壁的準確度）；`2026-10-15_fldrop50_open_loop.md` fldrop50（p 0.5）訓練完、上 Hub；開環 100k 跟 fldrop30 三個 N 都分不出、對基底只有 N=20 較差（同 p30）；手腕不平移的 fldrop30-augfl 開環在 N=100／50 一致變差（伸多遠的兩個關節），原因未查；`2026-10-14_fldrop_open_loop.md` ntcfA1aug＋接近段 front-left dropout（p 0.3、30 幀）：開環與基底大致分不出、h6 轉腕不變；h1 中位數過、逐幀沒過；閉環 30/36 vs 基底 25/36（配對 p 0.18），下一步 p 0.5（`docs/handoff/2026-10-14_fl_dropout_p50.md`）；`2026-10-13_closed_loop.md` ntfA1aug／ntcfA1aug 100k 閉環都是 25/36（C 場 ntA1 19/36，配對 p 0.18–0.24），遠區 8/8 轉腕，失敗多為 bad_aim；ntcfA1aug＋temporal ensemble −0.01 c1–c13 11/13（N=30 9/13），結論先做手腕定位精度；開環評估集上轉腕沒被觸發；`2026-10-13_open_loop.md` ntfA1／ntcfA1 加隨機平移增強（ntfA1aug／ntcfA1aug）：h6 平移後近處 3/13 → 0/13 轉腕；開環 ntfA1aug 只在 N=20 較低、ntcfA1aug 九格都較低；`2026-10-10_open_loop.md` ＋far-wrist（ntfA1）與 tight 剪前段（ntcfA1）開環，100k 兩者分不出、20k 剪前段明顯較差；`2026-10-08_open_loop.md` 紙杯＋normal/tight A1 開環，100k 只在 N=20 顯著變差、人類下限依定義而變；`2026-10-07.md` ACT 夾哪裡由 front-left 決定、抓取前不用手腕修正，但遮手腕後搬運 0 成功；`2026-10-06.md` 紙杯 100k 閉環 N=30 17/32 與依位置分組的開環 MAE；`2026-10-05.md` 閉環 N=30 起始姿勢偏移與 `--goto-home`；`2026-10-04.md` 寶特瓶 100k 開環 N=30；`2026-09-29.md` open-loop × 推論間隔；最近一次老師會議 `2026-09-18.md`；**2026-10-13 補寫三場閉環**：`2026-09-21_bottle_closed_loop.md` 寶特瓶 100k 4/35、`2026-10-07_temporal_ensemble.md` 19/32、`2026-10-08_C_n30_home.md` recovery 模型 N=30 從 HOME 出發 19/34，比對腳本 `scripts/compare_closed_loop.py`） |
| `docs/weekly/` | 週報（只整理、不當正本，數字以 `docs/meeting/` 為準）。一份＝一週（週一–週日），檔名＝該週週一：`<YYYY-MM-DD>.md`。最新：`2026-10-05.md`（10/5–10/11：轉到紙杯、`bad_aim` 診斷、補錄近場／recovery／遠區轉腕、ntA1／ntfA1／ntcfA1）；10/12 週待寫 |
| `eval/README.md` | 評估記錄規格（outcome/mechanism 兩軸） |
| `docs/models.md` | 模型對照表：權重 hash ↔ 本機 run/step ↔ Hub repo ↔ 物品 ↔ 閉環用過哪次 |

---

## 🔴 互動規則（這幾條是用踩過的坑換來的，不要違反）

### 1. 技術與事實斷言前必須查證，而且要讀完

**這個 AI 在本專案犯過同類錯誤多次：拿到資料不等於讀懂資料。**
- 斷言「Isaac Sim 沒有 SO-ARM 資產」→ 錯，有 `Robots/RobotStudio/so100.usd`
- 「更正」時只讀了 983 行網頁的 711–900 行就下結論（RobotStudio 在第 969 行）
- 落點分析用「錄取率」（base rate）冒充「條件機率」，被 Eric 當場指出方法錯誤
- 🔴 **2026-09-03：斷言「沒有現成 OMX 模擬資產、要自己匯入」→ 錯。**
  `~/isaaclab_volume/assets/wildbot_with_omxaiarm.usd` 從 8/28 就在硬碟上，
  `open_manipulator_description` 也早已 clone。**只查了外部 repo，沒查自己的機器。**
  → **規則追加：談「有沒有某個資產／某份程式」時，先 `find` 本機，再查網路。**
  同理，`omx_f` 的 joint limits 一度被寫成「URDF 沒有就沒辦法」——
  **原廠規格頁一直查得到**（`decisions.md` D025 §2026-09-03）。

**規則：引用前先讀完；讀不完就明說讀了多少。標示 `已查證` / `推論` / `未確認`。**

### 1-b. 引用專案文件時要看它的日期

**超過一週的「待辦／缺口」欄位，複述前先向他確認狀態。**

### 2. 不確定就說不確定，區分「證據」與「推論」

Eric 會查證你的說法，而且經常是對的。**被他推翻不是問題，掩飾不確定性才是。**

### 3. 進度與完成度要誠實

做了 12 篇就說 12 篇。**未完成清單要列出來。**

### 4. 偏好簡潔直接，給建議要附判準

冗長鋪陳會被視為雜訊。**給結論也要給「你可以怎麼自己複製這個推理」。**

### 5. 不要幫他做「理解」這一步

提問優於給答案。他正在練歸因力，代勞會抵銷訓練效果。

### 6. 他低估自己，但不要用鼓勵去補

遇到自我否定時**用數據對照**，不要安慰。**不要為了鼓勵而高估證據**——他會反駁。

### 7. 文件不要造成第二個真相來源

同一件事只在一個地方有正本，其他地方用指標。**發現兩處說法不一致要主動指出。**

### 8. 排任務要通過四關，時間估計要乘 2–3 倍

**2026-08-26 新增，正本：專案知識庫 `歸因力訓練計畫.md` §10–§11。**

同一段時間內，團隊決議 **6/6 完成**、自訂探測任務 **0/5 完成**——**差別是任務規格，不是意志力。**
四關：① 有明確終點 ② 他知道第一步怎麼做 ③ 有外部追蹤對象 ④ 死線來源是外部還是自訂。
**通不過就不要排——排了只會變成下一筆罪惡感。**

且 **AI 的時間估計已證實系統性低估 2–3 倍**（「讀 2 篇論文 2 小時」「兩份 pitch 各 60–90 分」）。
**給估計要乘上去，並邀請他回報實際值來校正。**

---

## ⚠️ 目前未解（會影響決策）

**A 線**

# docs/worklog — 現場／實驗／分析工作紀錄（非會議）

**會議紀錄在 [`../meeting/`](../meeting/)，這裡只放「做了什麼、量到什麼」的紀錄。** 數字以這裡的各檔為正本；週報（`../weekly/`）和會議紀錄只引用。

- 2026-10-19 從 `docs/meeting/` 搬來，**檔名不變**。舊路徑 `docs/meeting/<檔名>` 一律換成 `docs/worklog/<檔名>`。
  `configs/`、`config_records/` 裡的註解刻意沒改（YAML 全文進 config record 的 sha256，改註解會讓同一份設定換 digest）。
- 檔名：`YYYY-MM-DD.md`（當天的現場紀錄），同一天多份就加後綴：`_closed_loop`、`_open_loop`、`_diagnosis` 等。
- 每份開頭寫「性質」：現場工作／分析／事後補寫，以及參與者。

## 索引（新 → 舊）

一行摘要只為了找檔，**細節與限制以原檔為準**。

| 檔案 | 內容 |
|---|---|
| `2026-10-29_fl_markerless.md` | front-left 外參不用 ArUco 的三種試法：只用底座 PnP 工作區誤差 55 px（不行）；底座反推座標紙↔手臂 dx −3.7、dy +9.1 mm、yaw ≈ 0；手臂標記＋FK 從 ArUco 起點出發獨立檢查 1.93 px，起點偏 5°／3 cm 就變成 15.1 px，兩組都沒收斂；固定手臂參數、只解相機（相機碰歪、手臂沒動）時，起點偏 10°／10 cm 也收斂到同一解，獨立檢查 2.84 px、相機差 4 mm／0.69°，5 張照片也差不多；不貼標記（CAD 輪廓＋關節讀值）座標紙 2.14 px（有標記 2.86 px），偏 2°／1 cm 收斂、偏 5°／3 cm 只收斂一半 |
| `2026-10-22.md` | 寶特瓶 136 集 100k（`fe5e98957d38`）閉環，temporal ensembling −0.01、45 s：23/36（扣 c12 夾不到、c16 第三視角看不到 → 23/34）；主要失敗是夾太高滑掉 6、推倒 3；nas30 那場作廢；閉環位置剃 c12、補 c37 |
| `2026-10-22_bottle136_open_loop.md` | 寶特瓶 135 集（舊 60＋10/21 補錄 75）＋fldrop30-augfl（b136augfl）在 4090 訓練 8 h 42 m，`fl_dropout.json` 跟交接檔一致；開環對 b60augfl／b4a 18 格全部較低（15 格 p < 0.05），開環不當判準；評估集的第三視角跟 A1／A2 都不同；上 Hub；閉環見 `2026-10-22.md` |
| `2026-10-21.md` | 寶特瓶補錄 4 組 76 集（有效 75）：normal 16、recovery tight 12、中遠區精準 37（不轉腕、r_max ≈ 38.3 cm）、倒瓶 11；上 Hub；訓練 config＋`docs/handoff/2026-10-21_bottle_136.md`；④ 倒瓶提前做了組內決議 A5、保留在訓練集（§4） |
| `2026-10-18.md` | 寶特瓶 augfl-fldrop30 100k 閉環停在 c16，0/15（10 bad_aim、3 stalled、2 dropped）；手腕相機離線標定，沒啟用候選；5 個 README 中文壞成 `?`；`aruco_points.csv` 換成 10/18 版 |
| `2026-10-17_uvc60_augfl_closed_loop.md` | u60augfl（只用 uvc_60 60 集）100k 紙杯閉環 20/36（遠區 3/12），沒過 ≥ 29/36；柏宇判讀是資料問題；更正 10/16 §1-1 相機位置與亮度 |
| `2026-10-17_uvc60_augfl_open_loop.md` | u60augfl 上 Hub、開環：100k 在 N=100／50 比 b1／augfl 差，20k 反而比 augfl 好，方向隨 step 不一致；開環不當判準。§4–5 寶特瓶 60 集用同方法重訓（b60augfl）、上 Hub；開環跟舊寶特瓶模型 b4a 逐集配對比：3 個 step（20k／40k／100k）× 3 個 N（100／50／20）都沒有顯著差異（CI 都跨 0，最小 p 0.12）；閉環見 `2026-10-18.md`（100k 0/15） |
| `2026-10-16_closed_loop.md` | fldrop50 28/36、**fldrop30-augfl 34/36**（129 集）；augfl 對 fldrop50 配對 p 0.07、對 fldrop30 p 0.22；c17／c36 轉向分析；§4 下一步 |
| `2026-10-15_fldrop50_open_loop.md` | fldrop50（p 0.5）訓練、上 Hub、開環；手腕不平移的 fldrop30-augfl 開環在 N=100／50 變差，原因未查 |
| `2026-10-14_fldrop_open_loop.md` | ntcfA1aug＋接近段 front-left dropout（p 0.3、30 幀）：開環與基底分不出；閉環 30/36 vs 基底 25/36（配對 p 0.18） |
| `2026-10-13_closed_loop.md` | ntfA1aug／ntcfA1aug 100k 閉環都是 25/36；遠區 8/8 轉腕；失敗多為 bad_aim；TE −0.01 c1–c13 11/13 |
| `2026-10-13_open_loop.md` | ntfA1／ntcfA1 加隨機平移增強：h6 平移後近處轉腕 3/13 → 0/13；開環 ntcfA1aug 九格都較低 |
| `2026-10-10_open_loop.md` | ＋far-wrist（ntfA1）與剪前段 tight（ntcfA1）開環：100k 分不出、20k 剪前段較差 |
| `2026-10-08_paper_to_pan.md` | 座標紙座標系 ↔ 手臂 pan 軸座標系的偏差 |
| `2026-10-08_open_loop.md` | 紙杯＋normal／tight A1 開環：100k 只在 N=20 顯著變差 |
| `2026-10-08_C_n30_home.md` | （10/13 補寫）recovery 模型 N=30 從 HOME 出發，19/34 |
| `2026-10-08.md` | recovery 模型閉環 N=100 與 recovery 示範比對；§8 遠區轉腕抓法的界線 |
| `2026-10-07_temporal_ensemble.md` | （10/13 補寫）紙杯 100k 閉環＋temporal ensembling，19/32 |
| `2026-10-07.md` | ACT 夾哪裡由 front-left 決定、抓取前不用手腕修正，但遮手腕後搬運 0 成功；§5 補錄計畫 |
| `2026-10-06_diagnosis.md` | 紙杯 100k 位置分組、chunk 交接與排查實驗（離線＋提案） |
| `2026-10-06.md` | 紙杯 100k 閉環 N=30 17/32；依位置分組的開環 MAE |
| `2026-10-05.md` | 閉環 N=30 的起步問題：起始姿勢偏移、`--goto-home`、過曝 |
| `2026-10-04.md` | 寶特瓶 100k 開環 N=30＋閉環 N=30 準備 |
| `2026-09-29.md` | open-loop × 推論間隔（n_action_steps） |
| `2026-09-21_bottle_closed_loop.md` | （10/13 補寫）寶特瓶 100k 閉環 c1–c36，4/35 |
| `2026-09-18.md` | 鋁罐 60 集 ep 49 擺錯另錄、失敗／遮擋記錄（§1-1）；11 號馬達無回應；寶特瓶開環 12 集 |
| `2026-09-17.md` | 閉環評估錄影：資料夾 ↔ 模型對照 |
| `2026-09-13.md` | 換 UVC 手腕相機、固定曝光；錄紙杯 60（uvc_60）＋寶特瓶 60；紙杯 20k 閉環 5/10 |
| `2026-09-03.md` | lab day 工作單——因雨取消，未前往 |
| `2026-08-31.md` | lab day #2 工作單（去之前寫、回來就地填） |
| `2026-08-24.md` | lab day #1：Phase A 首次實機，只完成 A2 |

比對多場閉環：`scripts/compare_closed_loop.py`。模型 hash ↔ run ↔ Hub：`../models.md`。

# teleoperation_pick_place

以模仿學習（[LeRobot](https://github.com/huggingface/lerobot) + ACT）讓機器手臂撿拾垃圾。

## 目標

將 ROBOTIS OMX 手臂裝上移動平台，在**室外**定點撿拾散落的垃圾並丟入車上的垃圾桶，並且在光線、陰影、背景等環境變化下仍維持成功率。

目前的範圍是剛體、不透明的空容器（紙杯、寶特瓶、鋁罐）。透明與可變形物體（例如壓扁的寶特瓶、塑膠袋）暫不納入。

<a id="status"></a>

## 目前進度（2026-10-28）

| 階段 | 狀態 |
|---|---|
| 室內桌面：紙杯 | ✅ 實機閉環 34/36 |
| 室內桌面：寶特瓶 | 🟡 補錄示範後實機閉環 23/36，未達目標（> 80%） |
| 模擬（Isaac Sim） | 🟡 實機軌跡可在模擬中重現；下一步用模擬產生環境變異資料 |
| 上車（移動平台） | ⚪ 預計 2026 年 11 月底 |
| 室外定點撿拾 | ⚪ 上車之後 |

## 方法

1. **示範資料**：以 leader–follower 遙操作錄製示範，觀測包含手腕相機、第三視角相機與關節角度。除一般示範外，也針對近處、遠處與失手後恢復補錄示範。
2. **訓練**：ACT（Action Chunking Transformer），搭配第三視角影像的隨機平移／旋轉增強與隨機遮蔽，降低模型對相機位置與單一視角的依賴。
3. **評估**：在固定的擺放點上做實機閉環測試，逐集記錄成功與失敗原因。
4. **模擬**：校正手臂模型後，在 Isaac Sim 重播實機軌跡，用來擴充光線、背景等環境變異。

## 使用

| 步驟 | 指令 |
|---|---|
| 錄製 | `lerobot-record --config_path configs/record_omx*.yaml` |
| 訓練 | `scripts/train_fl_dropout.py`（完整參數見 `configs/train_omx_*.yaml` 檔頭） |
| 閉環評估 | `scripts/rollout_labeled.py --config_path configs/rollout_omx_*_eval.yaml` |
| 模擬重播 | 見 `sim/README.md` |

<a id="where-the-data-lives"></a>

資料集與模型權重放在 Hugging Face Hub（private），不在本 repo。

## 目錄

| 路徑 | 內容 |
|---|---|
| `configs/` | 錄製、訓練、評估設定 |
| `scripts/` | 資料處理、訓練、評估、校正工具 |
| `plugins/` | LeRobot 擴充（相機參數固定、執行設定紀錄） |
| `sim/` | Isaac Sim 資產、校正與重播 |
| `docs/` | 實驗規格、決策紀錄與工作紀錄 |

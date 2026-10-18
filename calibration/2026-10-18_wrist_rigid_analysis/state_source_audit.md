# 2026-10-18 實際讀數來源稽核

## 結論與範圍

本次查核的正式照片標定鏈路沒有從命令值改成實際值的切換：拍攝索引的 state_* 一直來自 Present_Position 的實測平均；相關幾何擬合使用這些 state_*。前面的說明讓人以為最近才開始用實際值，這個說法不正確。

查核範圍為本次對話使用的 capture_link_markers.py、measure_link_tilt.py、diagnose_wrist_grasp.py、analyze_wrist_rigid.py，以及既有 solve_wrist_cam.py、solve_link_markers.py、prep_closed_loop_replay.py、physics_replay.py。這不是對整個repo所有歷史實驗的無限範圍保證。

## 正式資料流

1. measure_link_tilt.Arm.read() 對 Present_Position 做 sync_read；read_avg(n) 對實測值取平均。
2. capture_link_markers 的 session / walk / scan 拍攝前使用 arm.read_avg(10) 取得 st。
3. capture_stills 將傳入的實測 state 寫入 CSV 的 state_<joint>，保留3位小數。沒有從命令目標回填。
4. diagnose_wrist_grasp、analyze_wrist_rigid、solve_wrist_cam 逐軸讀取 state_<joint>，再呼叫 FitModel.link5(st)。
5. solve_link_markers.load_stills 亦從 CSV 的 state_<joint> 建立 state。計畫檔在該流程中用作姿勢來源索引，不替代關節讀值。
6. 姿勢ID裡的r/f是指令偏移標籤，用來分組與選擇保留姿勢，沒有作為實際角度輸入FK。

FitModel仍會對實際角度套用既有零點與下垂模型；這是模型層，不表示原始輸入是命令值。這些模型項本身仍需驗證。

## 數值核對

| 資料 | CSV與before_capture實測日誌的最大差（讀數單位） | CSV與命令值最大差 |
|---|---:|---:|
| wrist_flex_diag：3姿勢、6張照片 | 0.000380 | 3.2162 |
| wrist_rigid_f40：4姿勢、8張照片 | 0.000479 | 2.8225 |

上述CSV與日誌差值在保留3位小數的捨入範圍內，因此可核實state_*對應實測而不是命令。

f40拍照前後六軸實測讀數的最大差約0.0293單位。相機順序拍攝並共用拍照前的平均讀數，尚非硬體同步的每張曝光時刻關節取樣；不能把它描述成嚴格曝光同步量測。早期沒有before/after日誌的照片可由程式追溯來源，但不能額外聲稱已逐張核實其曝光時刻穩定度。

## 重播來源核對與例外

- prep_closed_loop_replay從資料集observation.state生成episode.json的state及q_body_rad；q_body_rad是在實測state上加既有模型轉換。
- 已核對scratchpad/replay_cl的36個episode.json：state與原始parquet中的observation.state完全相同，最大差0。
- physics_replay的五個身體關節驅動目標來自上述實測state或由它計算的q_body_rad。模擬驅動目標不是模擬完成後的實際關節位置；程式另存q_sim，渲染手腕相機使用模擬自己的關節角度。
- physics_replay的夾爪驅動目標明確使用gripper_target_rad_from_action，初始開度則取gripper_rad_from_state。這是夾取動態重播的命令輸入，不是觀測到的實際姿態；不可把這個夾爪驅動值稱為實測或用於實測姿態驗收。該動態設定未在本次稽核中擅自更改。
- 這個夾爪驅動例外不參與本次手腕相機的五軸link5 FK。

## 規劃值的用途與界線

先前規劃新視角時，用過錄影的state作為新動作的目標，以及用命令值/假設跟蹤偏差預估視野。動作規劃與dry-run的離桌高度是模型預估，不是本輪現場實測高度；這些值沒有輸入正式照片殘差擬合。先前沒有清楚區分用途，應予更正。未到位的傾角組也不能只因指令不同便視為不同實際姿勢。

## 最新f40驗證

固定f50得到的相對相機/標記解、不用f40重新擬合，以f40 CSV的實際state_*驗證4張新手腕照片、5個標記觀測：
- 原模型：角點中位數12.43px、最大15.68px。
- 在f50重新擬合「關閉下垂補償」的對照模型，再驗證同一f40：中位數6.22px、最大8.67px。

這顯示下垂模型是需要檢查的因素，但對照結果仍未解決跨姿勢誤差，也不是刪除補償的充分證據。f40一旦用於挑選補償模型，就不能再當成該選擇的獨立最終驗證。

原標定、馬達PID與正式重播設定均未因本次稽核而修改。暫不要求追加拍攝；先釐清跨姿勢模型差異。

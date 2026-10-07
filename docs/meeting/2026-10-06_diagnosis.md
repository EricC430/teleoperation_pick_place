# 2026-10-06：紙杯 100k 的位置分組、chunk 交接與排查實驗

性質：離線實測＋實驗提案。未執行新的真機試驗、重新訓練或 S5 擴增。

## 1. 目前結論與證據強度

- **[已查證] N=30 的 action 命令在 chunk 交接處明顯跳變。** 今天 rollout 的 8865 幀、36 集中，交接幀的最大身體關節單步命令差平均 6.04 `.pos`，其餘幀 1.25，約 4.84 倍。這證實命令不連續；不能單憑它證明所有抓取失敗都由暴衝造成。
- **[已查證] 模型可以在部署推論模式下擬合訓練軌跡。** 60 集 train 的 MAE 是 N=100：0.677、N=30：0.637；12 集 open-loop 分別 6.227、4.382。**[推論] 優先查泛化／閉環狀態偏移／預測交接，比一開始延長訓練更有依據。** 這不排除接觸瞬間、個別關節或特定姿態仍有擬合問題。
- **[已查證] 近距離的 open-loop MAE 較低，closed-loop 卻最差。** 遠距離也不是 open-loop 誤差最大的一組。全程 MAE 包含停放、接近、抓取、放置與示範者速度差異，不能當抓取成功率或定位誤差。
- **[未確認] 腕部資訊是否被有效用於修正。** 單一近距離 o11 示範的 decoder t0 權重，腕部 camera tokens 平均占 51.88%，第三視角占 48.02%。Encoder 已混合相機／state 資訊；這個比例不能證明有效利用，也不支持直接把原因定成「腕部沒有 attention」。需要影像干預實驗。
- **[使用者回報] 示範重播能成功、第三視角已重新對齊。** 重播驗證的是那些 action 和位置的可执行性；不驗證模型是否會從影像選對 action。畫面大致疊合，也不能完全排除高度、內參、腕部安裝、遮擋或同步誤差。

## 2. 資料來源與分組

模型：`sha256(model.safetensors)[:12] = 2bdff2b3871e`，對照 [模型正本](../models.md)。訓練 chunk_size=100、原始 n_action_steps=100、temporal_ensemble_coeff=null、n_obs_steps=1；本次 N=30 只改執行／重新推論間隔，**沒有重新訓練 chunk_size=30**。

本次读取／計算的來源：

| 來源 | 用途與限制 |
|---|---|
| `data/huggingface/lerobot/ericc430/omx_pick_place_pilot_uvc_60` | 60 集訓練示範；本次重算全部 N=100、N=30 |
| `episode_meta/omx_pick_place_pilot_paper_cup.csv` | 60 集 outcome/quality/placement；CP950 編碼；確認 ep i → t(i+1) |
| `data/huggingface/lerobot/ericc430/omx_pick_place_open_loop_eval` | 12 集錄好的人工評估示範；本次重算 N=30，N=100/50/20 使用既有逐集結果 |
| `episode_meta/rollout_omx_b1_uvc60_100k_paper_cup_20260918_010912.csv` | 9/18 的 36 個 closed-loop outcome；4 筆 valid=0，另有非 episode 的 `0.5` 摘要尾列，計算時跳過尾列 |
| `episode_meta/rollout_omx_b1_uvc60_100k_nas30_A1_paper_cup_20261006_close_loop.csv` | 今天 36 個 outcome；全部 valid=1 |
| [今天上傳的 rollout](https://huggingface.co/datasets/ericc430/rollout_omx_b1_uvc60_100k_nas30_A1_paper_cup_20261006_094908) | revision `f2f9ab38f6d0083e6bb5fc0a728cdc8a917b4688`；下載 data/meta，未下載真機 rollout 影片；動作與 state 實測來源 |
| `docs/assets/placement_label_map_campA_136sym_20260908.csv` | 由 shoulder_pan 軸算 r=hypot(x_pan_cm,y_pan_cm)，不是用墊子左緣當半徑原點 |

**[未確認] closed-loop 實際權重 hash。** 兩次閉環以使用者提供的模型名稱／專案對照表識別；這次未取得 Windows 的最新 rollout config record 或部署權重 hash。離線 train/open 的 hash 已實算並確認一致。

分兩層比較，避免把「原始位置組」全部叫遠距離：

- **主區域**：t1–t50／o1–o10／c1–c30，r≥22 cm。
- **近距離擴充組**：[D030](../decisions.md#d030--campa_136sym-18-near-field-points-added-by-hand-d023s-r_inner22cm-no-longer-holds-for-the-elevated-base) 的 t51–t60／o11–o12／c31–c36，r<22 cm。
- 主區域再切 **中距離 22≤r<34 cm**、**遠距離 r≥34 cm**。34 cm 是本次探索性切點，沒有用 outcome 搜尋最佳切點；不能視為正式凍結的協定。open-loop 遠組只有 3 集、近組只有 2 集，不宜做強統計推論。

所有 MAE 單位均為 **LeRobot `.pos` 正規化命令值，不是度數**；6 關節先逐集平均，再對集取等權平均。train 是對「訓練示範畫面」做部署模式 action prediction，不是把 train/loss 拿來和 closed-loop 比。

## 3. train／open-loop／closed-loop 比較 [已查證]

closed-loop 表使用共同 32 個位置：排除舊 CSV valid=0 的 c7、c16、c17、c36；原始標註完全保留。

| 組別 | Train 集數 | Train MAE N100 → N30 | Open 集數 | Open MAE N100 → N30 | Closed N100 | Closed N30 |
|---|---:|---:|---:|---:|---:|---:|
| 主區域合計 | 50 | 0.683 → 0.636 | 10 | 6.670 → 4.510 | 17/27，63.0% | 16/27，59.3% |
| 近距離 r<22 | 10 | 0.645 → 0.646 | 2 | 4.014 → 3.738 | 1/5，20.0% | 1/5，20.0% |
| 中距離 22≤r<34 | 33 | 0.658 → 0.598 | 7 | 7.099 → 4.880 | 12/16，75.0% | 13/16，81.3% |
| 遠距離 r≥34 | 17 | 0.732 → 0.709 | 3 | 5.669 → 3.647 | 5/11，45.5% | 3/11，27.3% |
| 全部 | 60 | 0.677 → 0.637 | 12 | 6.227 → 4.382 | 18/32，56.3% | 17/32，53.1% |

![分組誤差、成功率與交接幀跳變](../../outputs/paper_cup_diagnosis_20261006/comparison.png)

**判讀：**

1. **[推論] 不是近、遠組的整段訓練軌跡都沒學會。** 兩組 train MAE 都低；holdout 差距大。仍需檢查接觸前後 ±0.5 s 的關節／TCP 誤差，不能用全程平均排除局部擬合不足。
2. **[已查證] N 縮短讓 open-loop 整體 MAE 下降約 29.6%，closed-loop 沒有一致改善。** 近組只有約 6.9% 的 MAE 改善。N=20 時近組 3.772、中組 3.872、遠組 2.926；短間隔也没有顯示近組就會變容易。
3. **[推論] 接觸精度、觀測可用性或失敗狀態的回復能力，可能比全程 MAE 更能解釋近組失敗。** 下一輪要記錄「第一次閉合時手指距離杯緣／杯身多少 mm」，及閉合前是否有新鮮腕部影像進入推論。
4. **[已查證] 近組的位置不是完全沒有示範鄰居。** c31 最近 t54：1.000 cm；c32 → t40：1.518；c33 → t55：2.236；c34 → t52：2.000；c35 → t57：2.236；c36 → t60：1.221。位置接近不代表影像／手臂接近姿態也接近；不能直接推出「資料量已足夠」。
5. **[未確認] 純粹改 N 的因果效果。** 兩批跨 9/18、10/6，沒有同日隨機交錯對照；相機與其他環境條件可能不同。一個位置只有一次試驗，不能把遠組 5/11→3/11 定為已證實退步。

同位置結果改變：c25、c31 由失敗變成功；c2、c5、c35 由成功變失敗。這比只有總成功率更有利於選對照片段。

### 分母與遮擋問題

完整 36 筆：N100=18/36、N30=17/36；近組兩批都是 1/6。今天 CSV 按其 valid 欄位算，應報 17/36；17/32 是本次共同位置子集。

舊檔把「推測第三視角看不到」的 c7/c16/c17/c36 設為無效，今天卻全部設有效。兩批的 valid 定義不一致。**固定配置造成的遮擋通常是系統失敗／可觀測性問題，和 camera dropout 這類臨時硬體故障不同。** 建議下一輪保留遮擋位置在全工作區分母，另列「物體可見子集」；不要在看到模型失敗後才決定 valid。此次只揭露並分層計算，沒有回寫原始標註。

### 示範品質

人工標註全部 60/60 成功；主組 quality 平均 4.88、近組 4.90。這些標註沒有顯示近組品質明顯較差；但 success／主觀 quality 不能證明 policy 看得到物體、標籤一致或接近方式足夠一致。t14/t22/t38/t42 註記第三視角看不到，t37 難看見；t2/t41/t46 有 recovery。先逐段驗證當時腕部是否補上資訊，不能自動把它們當壞示範刪掉。

## 4. chunk 暴衝的直接證據 [已查證]

定義：`jump[t] = max_j |action[t,j] - action[t-1,j]|`，只算 5 個身體關節，不混入 gripper；交接幀為每集 frame_index mod 30=0，排除第一幀。

| 幀類別 | 幀數 | 平均 jump | 中位數 | p95 |
|---|---:|---:|---:|---:|
| 交接幀 | 278 | 6.039 | 4.479 | 15.900 |
| 其餘幀 | 8551 | 1.248 | 0.838 | 3.908 |

這些是命令差，不能直接當角速度／加速度。每集多個幀彼此相關，278 個交接也不是 278 次獨立實驗。以 c2 的命令和實測 state 可看到交接後 follower 的反應：

![c2 的 action/state 與 chunk 邊界](../../outputs/paper_cup_diagnosis_20261006/c2_chunk_boundaries.png)

另外在乾淨示範上，每次新推論比較**同一未來時刻**：

`disagreement[t] = max_j |new_chunk[t][0,j] - old_chunk[t-30][30,j]|`。

| N30 示範重推論 | 交接數 | 同時刻 disagreement 中位數 | p95 |
|---|---:|---:|---:|
| Train 60 集 | 502 | 1.145 | 5.081 |
| Open 12 集 | 80 | 7.941 | 57.974 |

**[推論] 在 heldout 的正確示範畫面上，新舊 chunk 已經很不一致；不必先讓手臂走偏才出現。** 這把「heldout 的長期預測／任務階段估计不穩」列為優先假說，但不能區分影像 domain shift、示範速度差異、多種路徑與真正定位偏差。disagreement 是舊 chunk 的未執行續段和新預測之差；不等同上表的實際單步命令差，也不代表機器真的執行了 57.97 `.pos` 跳變。

本機已安裝 ACT 的 `select_action()` 只在 queue 清空時取新 chunk，直接執行前 N 步，沒有交接融合。因此 N=30 仍是每約 2 s 才採用一次新影像。即使相機一直拍攝，**物體在兩次推論間才進入腕部視野、又在下次推論前閉合，就不會利用那段近距離影像。**

只有名義 1/15 s 的 dataset timestamp；缺少真實 capture／inference／send 時鐘。不能從它排除同步推論暫停、舊幀、幀率不足等問題。

## 5. 建議實驗順序與裁決條件 [提案，未實跑]

每項先用固定失敗位置與成功位置做 pilot，再擴到凍結評估組。重複至少 3 次、同日交錯順序，記錄相機／calibration／模型 hash、起始 q 與實際物體座標。不要一次更換模型、相機、資料、N。

| 優先 | 要分開的原因 | 最小對照與量測 | 支持哪個解釋的判準 |
|---|---|---|---|
| 1 | 新舊 chunk 不連續 vs 純粹觀測太慢 | 同模型比較 N100、N30、N30＋對齐同一時刻的短重疊融合；記錄 new[0]、old[N]、最後實際 command、state、單步差、閉合位置。先選 c2/c5/c19/c32 等及中距離成功位置 | N30＋融合讓交接跳變下降，且接觸定位／成功改善，才支持交接是實質原因。若只變平滑仍空抓，定位／觀測問題仍在 |
| 2 | 抓取可行性／夾爪限制 | 相同近／中／遠位置手動或已成功 action 重播，各 3 次；紙杯抓杯緣與鋁罐抓罐身分開。鋁罐量實際罐徑、該姿態可用開口、手指方向與閉合留量；記 command-state gap、接觸後能否抬起 | 連手動／重播都失敗或開口無留量，先處理幾何／力／姿態；紙杯成功不替鋁罐通過此關。手動能抓、模型提前閉合則回到 policy／觀測 |
| 3 | 相機幾何／處理 pipeline 不一致 | 手臂停在與錄製相同 q、同杯子座標，取得 train 參考與現在影像。對 table 與杯緣／罐身高度各量 reprojection residual；腕部在 3 個 q 重做。核對 RGB/BGR、camera key、尺寸、曝光／模糊與讀取時間；比較校正前後模型輸出 | 幾何殘差有固定方向且跟杯子高度／位置變化，或恢復參考 setup 後同狀態輸出改善，支持 camera shift。只對平面桌墊疊合不足以排除 3D／腕部偏移 |
| 4 | 腕部有效使用 vs 看不到／沒來得及推論 | 在物體可見的接近／閉合前幀，固定 q、第三視角，對照正常腕部與凍結為前 10 幀；另做相反的第三視角凍結、兩者凍結。比較 action 變化，以及相對正確示範／TCP 對齊是改善還是惡化；黑圖遮罩只作粗篩 | 正常腕部比凍結有更小接觸前定位誤差，支持有用；正常与凍結近似，只支持「這些狀態對腕部不敏感」，可能第三視角已足夠。物體不可見不能拿來判 attention 不足；擾動為 OOD，需一致的多幀／多集證據 |
| 5 | 第三視角的物體→動作映射 vs q／任務階段捷徑 | 實際固定同一 q，把杯子沿 x/y 各移 ±2 cm，取一致影像，離線比較接近段預測 TCP 的方向／幅度；再固定杯子，以 3 個起始 q 對照，取各自一致的兩相機畫面。避免只把別集全圖與 state 硬交換 | 杯子位置變了但接近意圖幾乎不變，或一直縮向中間／提前閉合，支持空間敏感度／範圍泛化問題；同杯子換起始 q 就觸發閉合，支持 q 充當進度捷徑。仍需相機幾何先通過 |
| 6 | 全局 underfit vs 泛化／CVAE 推論差異 | 對近／遠各 4 集做小資料 overfit；以部署模式 z=0、相同 N 評 train 與 holdout，另看 k=0、10、30 及接觸 ±0.5 s。現有 20k/40k/100k 用同狀態同 horizon 比較，必要時再改 lr／KL | 小集合部署模式仍擬合不好，才優先查訓練／preprocessing／KL；train 已低、holdout 高且增加步數不改善，支持泛化。訓練 posterior 可讀 GT action，低 train/loss 不能取代 z=0 的推論評估 |
| 7 | 訓練 horizon 本身 vs 執行 N | chunk_size=100 與 30 各訓練；兩者均執行 N=30、同資料／batch／訓練觀測曝光量，至少 3 seeds；同樣記交接差、接觸誤差與成功率 | 短 horizon 穩定改善才支持重新訓練較短 chunk。只把既有模型 N100 改 N30，沒有測到這一項 |
| 8 | 覆蓋／示範品質／失敗狀態資料 | 固定最終評估位置；A 舊 60 集，B 加 K 集乾淨近／遠示範，C 加同數量「模型常見偏移→對準→抓取」示範，D 加 K 集中距離作資料量控制；B/C/D 用相同訓練抽樣與曝光量。加資料的位置避開最終測試錨點 | B 優於 D 支持位置覆蓋；C 優於 B 支持缺 recovery／部署狀態；同位置的清理／一致抓法另外做 matched-count 實驗，才能歸因品質。增加資料但不控制總量／抽樣，不能分開位置效果與總量效果 |

### Chunk 實驗的實作注意

- 現有模型可在推論時使用 ACT temporal ensembling，不必重新訓練，但 LeRobot 原生實作需要 **N=1，每個控制步重新推論**。先量部署電腦端到端 capture→推論→送命令的 p95 與相機幀齡；15 Hz 的單步預算約 66.7 ms。
- 維持 N=30 的直接修正方式見下節：短暫交接修正／同時刻 overlap，加上送出命令的變化限制。固定 3–5 幀融合本身沒有速度保證；大落差仍可能變成短時間快速移動。
- [原生程式](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py)與本機安裝版本均明示：positive temporal_ensemble_coeff 較偏重**較舊**預測；0 為等權。不要把 0.01 解讀為更相信新畫面。係數需用穩定性與接觸誤差對照，而非只看平滑。
- 所有條件使用相同已驗證的速度／單步限幅；另記限幅觸發次數，避免把 controller 裁切造成的改善歸成模型改善。

### 2026-10-07 補充：直接處理 N30 跳變的部署方案

使用者要求先解決現有跳變。本節是**實作設計，尚未修改 Windows 部署控制器／執行真機**；不用重訓現有 100k 模型。

**[已查證] 接入點與現成選項：** 本機容器的 ACT `select_action()` 在 queue 清空時只保留新 chunk 前 N 步，沒有融合；`SyncInferenceEngine.get_action()` 在 policy 後套用 postprocessor；OMX `send_action()` 才寫入 `Goal_Position`。因此以 `.pos` 為單位的交接與限幅應作用在 action 還原成馬達命令之後，並保存 `send_action()` 回傳的實際送出值。Windows 安裝版本尚未逐檔比對。容器 OMX 已有 `max_relative_target`，但它限制的是 **goal − Present_Position**，不是 **command[t] − command[t−1]**；會額外讀一次關節位置，不能當作完整的命令速度限制。字典設定要求包含全部馬達名稱，標量則連 gripper 一起限制。容器 ACT `supports_rtc()` 回傳 False，不能直接用 `--inference.type=rtc` 解決 ACT 交接。

**[建議] 第一版保留 N=30、15 Hz，做以下兩層：**

1. **以最後實際送出的命令銜接新預測。** 先只處理 5 個身體關節。令新 chunk 為 `b[k]`、上一個實際送出值為 `q_last`，短過渡內用

   ```text
   e = q_last - b[0]
   u[k] = b[k] + (1 - s[k]) * e
   s[0] = 0, s[B] = 1
   ```

   `s` 可選 smoothstep；此時 `u[0]=q_last`、`u[B]=b[B]`，避免把新 chunk 起點的落差一次送出。B=5–10 個控制間隔（15 Hz 時約 0.33–0.67 s）是候選起點，不是已驗證最佳值。這只保證起點位置連續，**不保證速度／加速度連續或中間姿態可行**。若要更好保留運動方向，可用最後兩次命令估計起始速度，以 Hermite bridge 接到新 chunk 的位置與速度，再套用下面的限制。

   可選擇保留上一個完整 100 步預測，以 `old[30+k]` 與 `new[k]` 做同時刻融合，再對融合後的序列做起點修正；不要拿 `old[99]` 接 `new[0]`。此方案需改掉原本截斷剩餘 70 步的 queue 管理。舊續段不一定正確，因此權重要在短过渡內退到零；舊、新差異很大時不能只靠平均解決。

2. **限制每步送出的命令變化，並管理追蹤落後。** 固定控制週期下，各身體關節分別限制 `abs(q_sent[t]-q_sent[t-1]) <= v_limit * dt`，必要時再限制命令速度的變化。`dt` 使用穩定控制週期；同步推論停頓後不能用一段很大的 elapsed time 一次放行大步補趕。限值須依各關節、負載與現有成功運動設定，不能把 `.pos` 直接當角度。若需要硬性的速度／加速度上限，应使用處理不可行狀態的軌跡產生器；不是連續兩次 clip 就一定同時滿足所有限制。

   大落差下，固定 B 可能在限速內走不完；應延長過渡或重新規劃，避免 B 到期時強制回到原始預測而再次跳變。若會明顯改變動作時間，gripper 的抓取閉合也要與手臂進度一起重新安排，不能手臂限速落後、仍照原先 frame index 閉合。只以 action-state gap 判斷接觸也不足；它沒有驗證杯子是否對準。gripper 不直接套用身體關節的平均／限速參數，避免改變抓取時機或鬆開物體。

這兩層能約束**送出命令**的跳變；實際運動仍受馬達 profile、追蹤落後、負載與推論停頓影響。若是交接時先停頓再猛動，還需要把推論移出控制迴圈：控制端穩定送命令、背景端產生新 chunk；新 chunk 依觀測到送出期間已過去的控制步對齊／丟棄過期前綴，再銜接。這是 ACT 的自訂非同步接續，不能直接套用需要 policy 支援的 RTC 選項。

**另一個現成方案：** 改成 N=1＋原生 temporal ensembling。ACT 不需重訓，但相較 N30，每控制步做 forward，推論呼叫數約增為 30 倍；15 Hz 的端到端預算是 66.7 ms。尚未量測部署 RTX 3050 是否達標。官方 [ACT config](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/policies/act/configuration_act.py) 已完整讀取，明定 temporal ensembling 需要 n_action_steps=1；不能維持 N30 只多加係數。

若大量交接都需要很長的橋接／限幅，表示新舊計畫的不一致尚未消失；控制器只能讓修正過程平順，不能替模型判斷哪條抓取路徑正確。下一層才處理短 horizon 重訓、部署偏移／recovery 示範或跨 chunk 一致性訓練。N30 即使平滑，仍是名義每 2 s 才採用一次新觀測；消除跳變不等於已解決腕部近距離影像的更新頻率或空抓問題。

### 其他會造成「尚未伸到卻先閉合」的問題

**[待測假說]** 示範速度或抓法不一致、q 被當作任務進度、CVAE 的 posterior／z=0 推論差、腕部杯子剛進視野就離開、第三視角固定遮擋、相機時間不對齊、錯誤 RGB／縮放／camera key、初始 q 不同、關節接近極限／負載追蹤不足、夾爪可用開口不足、觸碰後物體移動而缺少修正示範。

ACT 沒有一個可直接取出的「第三相機定位→IK」模組；這是端到端映射。判斷定位偏差要量物體改位時 action／TCP 的反應、及第一次閉合時的實際偏差，不能只以 shoulder_lift/elbow_flex MAE 較大認定 IK 錯了。

毫米級的接觸定位優先以尺量／標記影像驗證；現有 FK／touch calibration 本身仍有約 1.5 cm 的 3D 殘差（`sim/joint_mapping.py` 的量測記錄），直接把 FK 當毫米級真值會誤判。這個分析用的 FK 誤差也不等於 ACT 執行時用了錯誤 IK。

紙杯資料全是成功軌跡，仍可能只覆蓋「已對準時怎麼閉合」，沒有教模型「沒對準時不要閉合、先伸出去修正」。因此補資料有合理性，但目前建議**定向補接近姿態與回復**，先做上述等量對照；不直接大量重錄全部位置。

## 6. Visualization 已做什麼、適合用在哪

已修正 [attention 匯出腳本](../../scripts/visualize_act_attention.py)與 [viewer](../../tools/act_visualizer/index.html)：

- 保留原始 decoder attention 權重及各 camera token group 的 mass；ROI 比例改用原始值，display heatmap 仍各自縮放。
- ROI 不外推到標註範圍外；單一框只量該幀。標註間的插值仍需人工確認可見性，工具不會自動偵測遮擋。
- 移除未經驗證的 28%「ATTENTION LOST／Model Distracted」門檻，以及把 layer cosine／entropy 當定位失敗證據的文字。
- 明示工具每幀重推論，action 曲線為各次 chunk 的 k=0，等同 N=1 無 ensemble；它不重現 N30/N100 真機執行。chunk 圖標示 stride=2，50 個採樣點代表 offsets 0…98。
- 新增 `?data=` 載入 bundle；切換資料時清掉舊影像 cache／ROI。

適合先看 c2/c5 的 chunk 交接、c31/c35 的結果互換，再用示範的接近／閉合片段檢查 cup 可見性、ROI 面積、query offset 與 action。**低 camera mass 不等於沒有使用，高 mass 不等於定位正確；encoder 的 camera tokens 已含混合資訊。** ROI ratio 也要和 ROI 占 token 面積的比例比較，不應套固定數字。

已匯出 o11（ep10，221 幀）的 [診斷影片](../../outputs/paper_cup_diagnosis_20261006/attention/phase_b1_uvc60_ep10/ep10_diagnostic.mp4)及 JSON/雙相機 JPEG。frame90 的腕部影像能看到大部分杯口；它是**人工示範**，不是今天失敗 rollout，不能用來推斷所有失敗時 wrist 都可見。

從 repo 根目錄啟動本機 server：

```bash
python3 -m http.server 8088 --bind 127.0.0.1
```

開啟：

```text
http://127.0.0.1:8088/tools/act_visualizer/?data=/outputs/paper_cup_diagnosis_20261006/attention/phase_b1_uvc60_ep10/ep10_data.json
```

只有 o11 的實際 attention bundle 已產生；今天真機 rollout 影片未下載，未新做 ROI 逐幀標註／腕部干預試驗。viewer 已做 JS 語法與 ROI 數值驗證，未做瀏覽器端的完整操作測試。

## 7. 重現與產物

本次原始資料集、權重、NPZ、影片維持在 gitignored `data/`、`outputs/`。分析正本在本文件，原始標註正本仍是各 CSV，不回寫 outcome／valid。

本次新增：`scripts/analyze_paper_cup_failures.py`。輸出：`episode_errors.csv`、`group_errors.csv`、`closed_success.csv`、`paired_outcomes.csv`、`closed_chunk_jumps.csv`、`boundary_summary.json`、`summary.json`（含原始 CSV／action parquet 的 SHA256）、`comparison.png`、`placements.png`、`c2_chunk_boundaries.png`。

範例：用既有評估器重建 train N100（N30 換間隔與資料夾名即可；open 用 eval dataset、episodes 0…11）。

```bash
bash scripts/run_container.sh python scripts/eval_open_loop.py \
  --checkpoint data/train/phase_b1_uvc60/checkpoints/100000/pretrained_model \
  --dataset.repo_id ericc430/omx_pick_place_pilot_uvc_60 \
  --dataset.root data/huggingface/lerobot/ericc430/omx_pick_place_pilot_uvc_60 \
  --episodes $(seq 0 59) --n_action_steps 100 --no_plot \
  --save_plot_dir outputs/paper_cup_diagnosis_20261006/train_nas100

python3 scripts/analyze_paper_cup_failures.py
```

分析器也接受 `--closed-dataset-root PATH`。需先有 train_nas100／train_nas30／open_nas30 與原本 outputs/open_loop_paper_cup 的 N100/50/20 bundles，以及今天 revision 的 data parquet。缺少 bundle 會明示，不會用虛構數字補表。

验证：N30 加速計算只在 queue refill 幀取圖／推論，依模型 queue 的相同邏輯填入 chunk 前 N 步；先和既有 open ep0 N100 的每幀 action 比對，最大差 **0.0**。Train N100 由原始逐幀評估器跑滿 60 集。Attention hooks 的 o11 每 30 幀輸出和無 hooks 的預測最大差 **0.0**；每幀／各 query 的 mass 總和為 1，各 camera 原始 heatmap 加總和匯出 mass 最大差約 **1.04e-7**。ROI 驗證：原始權重的 10% 單格、全相機 100%、舊版無 raw 權重保持未量測，均通過。

文件一致性：`episode_meta/README.md` 仍寫 uvc_60 沒有 CSV，現在已有 `omx_pick_place_pilot_paper_cup.csv`；不要照舊段落重建覆蓋它。部分舊 training config 將 MAE 寫成度、將少量 checkpoint 差異定為 early overfit；本次採用實測 `.pos` 與 train／holdout 對照，不沿用該定論。

S5 的邊界：[規格](../specs/S5_sim_replay_augmentation.md)只重播既有姿態與動作。它可做視覺條件的增強對照，不能補出原始資料沒有的外圍伸展、不同罐身抓法或失敗回復動作。是否要擴增應等 camera／coverage 試驗的裁決，不因本次診斷直接啟動 S5。

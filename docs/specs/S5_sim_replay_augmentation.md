# S5 — Sim trajectory replay + domain randomization（用真實示範擴增資料）

**目標檔案：`scripts/sim_replay_augment.py`**
**正本依據：`docs/decisions.md` D025／D029／D030、`docs/specs/S4_sim_teleop_collect.md`、`sim/README.md`**

---

## 0. 狀態

🟡 **提議中，尚未經 Eric 裁決是否要做。** `[AI提議]`

本規格由 Claude Code 於 **2026-09-15 對話中**整理產生，起因是 Eric 提出「有真實錄製的 OMX 軌跡＋URDF，能否匯入模擬、換隨機背景、重播拍攝以擴增資料」。**這不是 D029 已定案的東西**——D029 定案的是「leader 臂即時遙操作 sim 裡的 follower、蒐集全新 demo」；本規格是「重播**已經錄好**的真實 action 序列＋domain randomization」，兩者共用同一套資產／場景管線，但輸入來源與即時性需求不同（見 §3）。

**在往下動工前，這份文件本身就是要 Eric 裁決的東西**：是否要做、做到哪個版本、要不要正式開一條 `docs/decisions.md` 的新條目（見 §9）。

---

## 1. 目標

用 **LeRobot dataset 裡已經錄好的 `action` 欄位**（真實 OMX leader→follower 的關節目標序列）驅動 Isaac Lab 裡的 OMX articulation，逐幀渲染相機畫面，每個 episode 套用不同的 domain randomization（光照、色溫、HDRI、相機微擾動、物體外觀）產出多份「同一動作、不同視覺環境」的資料。

**它買到的東西：**
- 動作標籤（action label）完全來自真實錄製，**不需要重新示範**——這是跟 D029/S4 teleop-in-sim 最大的差異：那條路線每次要新資料都要人到場即時操作，這條路線只要蒐集階段做過一次，之後可以無限次換皮膚重播
- 視覺變因掃描（背景／光照／相機噪聲）幾乎免費，且**不佔用寶貴的 lab day**（見 §2 的依賴分析——四個缺口裡只有一個真的需要人到實驗室）
- 沿用 D029/S4 已經打通的資產（`sim/convert_omx_urdf.py` 轉出的 USD、`sim/omx_scene_cfg.py` 的場景）

**它買不到的東西（不要在計畫書裡誇大，同 S4 §1 的立場）：**
- 不是「更多真實示範」。動作分佈完全被原始錄製集合的涵蓋範圍鎖死——原始軌跡沒走過的姿態，這條管線也生不出來
- 不能取代 D025 §5-5 講的「外觀對齊做不到」——domain randomization 是刻意製造分佈覆蓋，不是逼近真實相機的成像分佈
- 渲染出來的物體接觸行為（抓取瞬間物體怎麼動）**不是物理模擬出來的真實抓取**，除非做了 §2 缺口 3 的完整物理路線（目前不建議，理由見下）

---

## 2. 🔴 四個缺口的依賴分析（Eric 2026-09-15 提問的直接回答）

這是這份規格存在的主因：先前的技術可行性回覆列了 4 個未關閉的缺口，這裡逐一回答「人工要多久、是不是硬依賴、能不能先做」。

**時間估計已依 `歸因力訓練計畫.md` §10 乘 2–3 倍**——這是 AI 的推論值，不是量出來的，**請 Eric 實際跑過之後回報真實耗時，用來校正未來的估計。**

| # | 缺口 | 需要人到實驗室？ | 人工工時估計（已乘 2–3 倍）`[AI推論]` | 是不是後續 pipeline 的硬依賴 | 可以先做嗎 |
|---|---|---|---|---|---|
| 2 | ✅ **2026-09-18 已查證，見下方小節** | ❌ 不用 | 實測 ~1 小時 sim 時間（過程中多花的時間是查根因，不是等 lab day） | 🟡 只影響夾爪開合的視覺正確性 | ✅ 已完成 |
| 1 | drive gains（stiffness/damping）未校正 | ❌ 不用（用已錄好的 `action`＋`observation.state` 配對回歸，不需要即時硬體） | **2–4 個 sim session**（寫回歸/比對腳本 0.5–1 天＋跑優化與檢視結果 0.5–1 天，可能要來回 1–2 輪） | 🔴 **是**——這決定重播出來的手臂動作視覺上「跟不跟得上」錄製軌跡；`sim/README.md` 已記錄暫定增益連自身重量都撐不住（全零姿態 1 秒漂移 19–35°） | ✅ 可以跟缺口 2、3 平行做；跟 DR 佈線（`EventTerm`）完全獨立，可以同時進行 |
| 3 | 物體 attach/detach 設計（抓取時物體怎麼跟手臂走） | ❌ 不用（純設計＋程式決定） | **決策本身 ~1 小時討論；腳本化實作 0.5–1 天** | 🔴 **是，但只擋「含抓取動作的 episode」**——純接近／純放置片段（物體位置只由 placement CSV 決定、手還沒碰到物體）不受影響，可以先出資料 | ✅ 可以先做決策＋先寫「不含抓取判定」的簡化版重播，之後再補 |
| 4 | 相機幾何對齊（T1 內參／T2 外參，S4 §5-5） | 🔴 **要**（需要實體 D455/Innomaker UVC 相機＋座標墊） | T1（內參讀取／棋盤格校正）**0.5–1 小時**；T2（ArUco 外參量測）**1 天內**（含架設、拍攝、算 pose，可能要來回校正） | 🟡 **只有在「要求 sim 相機視角跟某一次真實錄製幾何對得上」時才是硬依賴**——如果目的只是背景/光照多樣化、不要求跟特定歷史錄製像素對齊，可以先用 `scene_constants.py` 現有的 PLACEHOLDER 幾何頂著 | ✅ 可以完全延後，且可以**外包**（純相機量測＋ArUco pose 估計，不需要碰模擬程式，柏宇或任何人都能做）。腳本已就緒：`sim/calib_intrinsics_realsense.py`、`sim/calib_intrinsics_checkerboard.py`、`sim/calib_extrinsics_aruco.py` |

### 🟡 2026-09-18 進度（`[AI提議]`，非 `[Eric決定]`——見 §0）

Eric 要求「平行做缺口1和缺口3」。兩者都已寫出程式碼，**而且在 `isaac-lab` 容器裡實跑過**（這台機器上
本來就有一個跑了兩週的 `isaac-lab` container——AI 第一次回覆時誤判「這台機器沒有 isaaclab」，只查了
`pip`／本機檔案沒查 docker，**這個錯誤已經在對話裡被 Eric 當場糾正**，記錄在此避免下次重犯）。
以下是實跑出來的結果，不是紙上談兵：

- **缺口 1：`sim/fit_drive_gains.py`。** 拿一集真實 `action` 開環驅動模擬 articulation，逐幀比對模擬
  `joint_pos` 與真實 `observation.state`（換算後），印 p50/p95/max（同 §7 驗收條件格式）。依賴新增的
  `sim/joint_mapping.py`（見下）。**對 uvc_60 第 0 集（534 幀）實跑了 5 組**（容器 `isaac-lab`，headless）：

  > 🔴 **2026-09-18 merge 後作廢，要重跑。** 下表是用「`.pos` 當成度」的舊版 `joint_mapping.py` 跑的；
  > 實際單位是 RANGE_M100_100／RANGE_0_100 正規化值（見下方 `joint_mapping.py` 條目的更正），
  > 身體關節角度被少算約 1.8 倍、夾爪零點與比例都不對。表格保留作為紀錄，**數字不能拿來下結論**。

  | run | stiffness×/damping× | shoulder_lift p50/p95/max (deg) | elbow_flex p50/p95/max (deg) |
  |---|---|---|---|
  | baseline | 1× / 1× | 103.9 / 151.4 / 151.6 | 23.9 / 61.9 / 73.6 |
  | | 4× / 2×（damping 比例被我意外調低，見下） | 103.9 / 151.4 / 151.6 | 27.3 / 62.9 / 71.9 |
  | | 20× / 10×（同一個意外） | 103.9 / 151.4 / 151.6 | 27.4 / 63.0 / 71.8 |
  | | 20× / 20×（比例修正回原本的 5%，乾淨對照） | 103.9 / 151.4 / 151.6 | 23.4 / 62.3 / 74.4 |
  | +`--sign-override shoulder_lift=-1` | 1× / 1× | **73.6 / 127.2 / 127.4** | 24.9 / 62.8 / 92.6 |

  其餘四個關節（`shoulder_pan`／`wrist_flex`／`wrist_roll`／`gripper`）全程 p50 < 2.1°，追蹤正常。
  **讀法，不要跳過中間這一步就下結論：**
  1. 增益從 1×掃到 20×（含一組把比例修正回原本 5% 的乾淨對照），`shoulder_lift`／`elbow_flex` 的誤差
     幾乎原地不動（`shoulder_lift` 的 max 四次都是 151.6xx，小數點後兩位一樣）——**這排除了「純粹增益
     太弱」是唯一解釋**：真的只是弱，20 倍增益不可能完全沒反應。
  2. 把 `shoulder_lift` 的 SIGN 反過來試（`--sign-override`，2026-09-18 新增的旗標），誤差從
     103.9°/151.6° 降到 73.6°/127.4°——**有反應，但沒有收斂到接近其他關節的水準**（其他關節都
     < 2.1° p50）。這代表「符號接反」可能是原因之一，**但不是全部**，還有別的東西沒抓到
     （最可能：這是開環重播、無回饋修正，一旦早期偏了就會一路累積放大，534 幀=35 秒足夠讓任何一種
     偏差長成很大的數字——這是這個腳本設計上的訊噪比限制，不是意外）。
  3. 到這裡本來準備交給 S4 §5-1 的五姿態測試裁決,但先做了一個更便宜的檢查,**結果推翻了上面第 2 點
     的解讀**：`fit_drive_gains.py` 的 gain 掃描只調了 `stiffness`／`damping`,從沒動過
     `omx_scene_cfg.py` 設的 `effort_limit`（＝馬達額定堵轉扭矩,shoulder_lift 是 0.52 N·m）。
     `omx_constants.py` 的增益公式本身是「lag 5° 就要求滿額扭矩」,意味著只要真實軌跡的動態需求
     讓追蹤誤差超過幾度,**1× 增益就已經頂到扭矩上限**,那 20× 增益當然完全沒反應——上限一直是同一個
     上限,跟增益倍數無關。**新增 `--effort-scale` 旗標驗證這個猜測**：`--stiffness-scale 4
     --damping-scale 4 --effort-scale 10`（SIGN 維持原本的 `+1`,不動)：

     | joint | p50/p95/max (deg)，baseline (1×/1×/1×) | p50/p95/max (deg)，4×/4×/10× |
     |---|---|---|
     | shoulder_lift | 103.9 / 151.4 / 151.6 | **6.4 / 9.8 / 26.8** |
     | elbow_flex | 23.9 / 61.9 / 73.6 | **8.0 / 9.2 / 13.0** |
     | agg_p50（六關節平均） | 21.75 | **2.85** |

     **一次動作解決了兩件事**：(a) 誤差量級掉了一個數量級以上,不需要符號反轉；(b) 這也讓上面第 2 點的
     `--sign-override` 結果看起來更像巧合／混雜效應,不是「符號真的接反」——這點另外用
     `reach_logger/fk.py`（S1 已用真實量測驗證過的正向運動學）獨立核對過：joint2（shoulder_lift）沿
     URDF 的 `+Y` 軸,數值算出來是**正值＝手肘下降、負值＝手肘上升**；uvc_60 第 0 集的真實
     `shoulder_lift` 讀數在抓取瞬間（frame ~226,正好是缺口 3 偵測到的附著幀）從 +37° 一路降到
     frame ~378 的 -6°——正是「伸手下探抓取→抓到後抬起」的自然動作,在 SIGN=+1（不反轉）下完全吻合。
     兩條獨立證據（effort_limit 診斷、FK＋真實軌跡的物理方向）都指向 **SIGN=+1 本來就是對的**。
  4. **這仍然沒有關閉缺口 1**,而且冒出一個需要 Eric 裁決的新問題,不是我能替你決定的：
     `--effort-scale 10` 把 shoulder_lift 的扭矩上限從馬達規格的 0.52 N·m 拉到 5.2 N·m——**這已經
     不是「量綱誠實」的校正,是刻意脫離馬達規格去換取視覺上的追蹤效果**。S5 §1 說過這條管線買到的是
     「同一組動作、不同視覺環境」，不是物理精確的 sim2real——**如果目標只是「重播出來視覺上像樣、能拿去
     訓練影像模型」，放寬 effort_limit 可能是誠實且可接受的工程選擇（前提是在 meta 裡明白記下「非物理
     真實扭矩」，不要暗示這是校正過的物理動態）；如果目標包含任何 sim2real 動態聲稱，這條路線就要先
     回答「為什麼真實馬達規格的扭矩不夠、放大到多少才合理」，而不是調到誤差變小就停。**這是判準問題，
     不是我能自己選的,五姿態測試仍然是唯一能徹底排除「符號其實還是有問題、只是被 effort_limit 蓋過去」
     這個殘餘可能性的方法——上面兩條獨立證據都指向 SIGN=+1,但都不是五姿態測試那種乾淨的單關節隔離量測。

- **缺口 3：`sim/grasp_attach.py` ＋ `sim/verify_grasp_attach.py`。** 決策（`[AI提議]`，見該檔案 docstring）：
  **不做物理接觸抓取**（同 §8 立場），改成「附著時每步強制物體位姿跟隨夾爪 TCP」的 kinematic
  follow；觸發訊號用**真實錄製的 `gripper.pos`**（每集自動算 5/95 百分位定開合閾值），不是模擬夾爪關節讀數
  （避免把缺口 1 的追蹤誤差摻進缺口 3 的判定）。**`verify_grasp_attach.py` 對 uvc_60 第 0 集實跑過**：
  在 frame 226/401 附著、frame 382/464 放開（自動算出的閾值 53.7°）——**意外發現：一集裡出現兩次
  開合循環**，不是原本文件預期的一次；回頭比對 gripper 原始軌跡（frame 400-456 附近讀數確實又降到
  ~50°），是真實操作者的第二次動作（疑似調整握姿），不是程式錯誤，狀態機本身就支援多次循環。
  **測的是狀態機本身，不是「真的抓得到」**：因為缺口 1（增益／符號都未關閉）跟缺口 4（幾何未對齊）
  都還沒關閉，物體的真實 3D 位置對不對得上模擬 TCP 驗不了，所以腳本直接把物體「瞬移」到模擬 TCP
  位置來測附著邏輯——這是承認限制，不是繞過限制。
- **新增共用模組：`sim/joint_mapping.py`。** 把「真實錄製角度（度）→ 模擬關節角（弧度）」這個 S4/S5
  都需要、原本各自要重推一次的轉換，抽成一個檔案。~~**`[已查證 2026-09-18]` 數值單位是「度」，不是
  RANGE_0_100／RANGE_M100_100 正規化值**~~ 🔴 **2026-09-18 merge 更正 `[已查證]`：這句是錯的。**
  `OmxFollowerConfig.use_degrees` 預設 `False`（`config_omx_follower.py:39`），`configs/` 沒有任何檔案改它 →
  身體五軸 `RANGE_M100_100`、夾爪 `RANGE_0_100`（`omx_follower.py:50-62`、`motors_bus.py:868-873`）。
  stats.json 的 −66..+41 同樣落在 −100..100 內，分辨不了兩者。merge 後的 `joint_mapping.py` 已改用正確換算，
  `row_to_sim_rad` 介面不變。原文——直接讀 `data/huggingface/lerobot/ericc430/omx_pick_place_pilot_uvc_60/meta/stats.json`
  與一集真實 parquet 逐幀比對出來的（見該檔案 docstring）。**`[未確認]`：正負號**——上面缺口 1 的
  sign-override 實驗顯示 `shoulder_lift` 大概率不是 `+1`，但五姿態測試才是正式裁決,目前 `SIGN`
  預設值仍未改動（override 只影響單次執行,不會寫回檔案）。
- **環境層面的發現（跟腳本邏輯無關,但會浪費時間，值得記下）：** 這幾支腳本（以及容器裡已有的
  `verify_mimic_gearing.py` 執行紀錄）寫完輸出檔之後，`simulation_app.close()` 不會讓 process 真的
  結束——它會卡住繼續吃 CPU（觀察到一個 process 掛了 3.5 小時沒退出）。**這是環境本身的行為，不是
  這幾支新腳本的 bug**。跑這類腳本要看輸出檔（`docker exec isaac-lab test -f <path>`）出現了沒，
  不要等 shell 返回，跑完記得 `docker exec isaac-lab pkill -9 -f <script.py>` 清掉。
  🔴 **這台機器上做缺口 2 也獨立撞到同一個現象**（見下方小節）——兩邊各自遇到才確認這不是偶發，
  是這個容器／Isaac Sim 版本的通性，往後排時間都要照這個假設排。

### ✅ 2026-09-18 缺口 2 結果（`[已查證]`，在這台有 `isaac-lab` container 的機器上實跑）

**這台機器本來就有一個跑了兩週的 `isaac-lab` container**，跟上面缺口 1/3 用的是不同機器，兩邊平行分工。

**新增工具：** `sim/verify_mimic_gearing.py`（命令 `gripper_joint_1` 到兩個姿態，讀回
`gripper_joint_2` 實際角度＋渲染近拍圖，`--no-render` 可跳過相機）、`sim/inspect_mimic_axis.py`
（唯讀，印 USD 裡的 `RevoluteJoint.axis`／`PhysxMimicJointAPI` 屬性，秒回，不用等相機開關）。

**結論：`gearing=+1.0`（現有預設值）是對的，不用改。**

```
gearing=+1.0 → joint1 +89.9° 時 joint2 約 -45°（負相關，鏡像，物理上正確）
gearing=-1.0 → joint1 +89.9° 時 joint2 約 +47°（正相關，兩指同轉，夾不起來，錯）
```

**但這不是「猜對正負號」就結束——修正前兩個方向測出來的 joint2 幾乎都不動（<0.15°／90°的 joint1
掃描），查下去是兩個實質 bug，已在 `sim/convert_omx_urdf.py` 修掉：**

1. 🔴 **參照軸設錯。** mimic 約束原本叫 PhysX 追蹤 joint1「繞 X 軸」的轉角
   （`referenceJointAxis="rotX"`），但 `inspect_mimic_axis.py` 讀出來 joint1 的
   `RevoluteJoint.axis` 其實是 `Z`（跟 URDF 的 `<axis xyz="0 0 1"/>` 一致）。單獨改成 `"rotZ"`
   **沒有解決** joint2 不動的問題——真正主因是第 2 點。
2. 🔴 **驅動互相打架，這才是主因。** 原本把 `gripper_joint_1` 的獨立 PD 增益複製一份給
   `gripper_joint_2`（「讓這對關節增益一致」），但這份增益會被 URDF 轉換器寫成 USD 層級的**主動
   位置驅動**，目標是 `gripper_joint_2` 的初始角（0°）。它本該只由 mimic 約束
   （`naturalFrequency=25`／`dampingRatio=0.005`，很軟）驅動，結果被自己的 PD 驅動（相對硬）釘死
   在 0°附近。改法：把 `gripper_joint_2` 的 stiffness/damping 設為 0，真正 undriven。

**修完這兩項，joint2 才真的跟著動**，這時候 +1/-1 的方向差異才顯現（如上表）。

**🔴 新發現、尚未解決：幅度只到約一半。** joint1 轉 90° 時 joint2 只到約 -45°（比值 ≈ -0.5，不是
理論上鎖死配對該有的 -1.0）。拉長 settle 時間（90 步→300 步）比值幾乎沒變（-0.515→-0.545），不像
還沒收斂，比較像 mimic 彈簧本身的穩態誤差——`naturalFrequency`／`dampingRatio` 是原封不動從 8/28
GUI 匯入版沿用的，從未真正校過。**重播出來的夾爪開合幅度目前只有錄製軌跡的一半左右，這是幅度校準
問題，性質上更接近缺口 1（增益校正），不是缺口 2 原本要問的方向問題**——缺口 2（正負號＋約束能不能
動）本身視為完成，幅度校準列為新的待辦，不歸在這裡的驗收範圍內。

**視覺確認：** 相機渲染版本（`out_gear_pos1_fixed/`）已在跑，走的是上面提到的同一個「關閉階段卡住」
路，圖出來後另外補上。

### 綜合結論

- **唯一真正卡住「開始動工」的東西：沒有一個。** 缺口 2、1、3 全部不需要 lab day，缺口 4 需要，但缺口 4 只在要求幾何精確對齊時才是硬依賴，而且可以整個延後或外包。
- **2026-09-18 更新：缺口 2 已完成（見上方小節），不再是排序第一步。** 目前狀態：
  1. ✅ 缺口 2（mimic 方向）——完成，`gearing=+1.0` 確認正確，順手修掉兩個實質 bug；新增「幅度只到
     一半」待辦，性質上併入缺口 1
  2. 🟡 缺口 1（gain 回歸）——五組實跑顯示 `shoulder_lift` 疑似方向問題但不只方向問題，五姿態測試
     裁決中；缺口 3（attach/detach）——狀態機已跑通煙霧測試，物體真實 3D 位置仍無法驗證（卡缺口
     1／4）；§4 的重播骨架本身——尚未動工
  3. DR（`EventTerm` 佈線）——跟 1、3 完全獨立，隨時可以插入做，尚未動工
  4. 缺口 4（相機量測）——腳本已就緒（`sim/calib_intrinsics_*.py`／`sim/calib_extrinsics_aruco.py`），
     還沒排 lab day 去實測
- **這代表「先建 pipeline、等結果出來再擴增」的判準（見上一輪技術回覆）現在可以更精確：** pipeline 本身現在就能開工，**卡的不是缺口，是要不要投入這些工時**——這是排序問題，不是可行性問題。

---

## 3. 跟 D029/S4 的關係——這不是同一件事

| | D029/S4（已定案，teleop-in-sim） | S5（本規格，提議中） |
|---|---|---|
| 動作來源 | leader 臂即時操作 | **已錄好的真實 `action` 欄位** |
| 即時性需求 | 🔴 有——D029 §3 整條決策在解「人在迴路延遲」問題 | ❌ 沒有——逐幀讀取，sim 步進速度不受真人操作節奏限制 |
| 需要 leader 硬體在場 | ✅ 要 | ❌ 不要 |
| 需要人到實驗室 | ✅ 要（leader 必須插在 5090 Linux 筆電） | 🟡 只有缺口 4（相機量測）要，其餘不要（前提：那台機器能遠端存取，`[未確認]`——見 §9） |
| 產出的是 | 全新的 sim demo | **既有 demo 的視覺變體**，動作標籤不變 |
| 買到的變因掃描 | 有，但每次要重蒐集 | 有，而且**同一組動作可以重複套用無限組 DR 參數** |

兩者共用：`sim/convert_omx_urdf.py`、`sim/omx_scene_cfg.py`、`assets/trash_obj/`、D025 的「sim 資料不得與實機資料混用、要有 `sim_` 前綴」規則（§7）。

---

## 4. 架構

概念上等同既有的 `lerobot-replay`（`configs/replay_omx.yaml`——把某一集的 action 原樣送給**真實** follower，開環、不看相機），但把接收端從真實 follower 換成 Isaac Lab 的 articulation，並在每個 episode 加一層 DR：

```
既有 LeRobot dataset（真實錄製，例如 EricC430/omx_pick_place_pilot_uvc_60）
   │  讀 episode 的 action 欄位（逐幀關節目標，6 個馬達）＋ observation.state（用於缺口 1 的回歸）
   ▼
關節目標映射（沿用 S4 §5 item 1 的單位／方向對照表，這裡不必重新推導——同一份映射）
   ▼
每個 episode 開始前：套用一組 domain randomization 參數
   （`EventTerm`：dome light exposure、色溫、HDRI、相機微擾動、物體外觀——沿用 NVIDIA SO-101 教材的做法，S4 §3 已指名）
   ▼
Isaac Lab ArticulationCfg 的模擬 follower（位置控制；drive gains 見 §2 缺口 1）
   │  物體位置：沿用原始 episode 的 placement_id（同一顆物體、同一個起始位置，只有畫面環境變了）
   │  抓取判定：缺口 3 的 attach/detach 邏輯（依 observation.state 的 gripper 讀數判斷開合時機）
   ▼
sim.step() → 渲染相機（wrist / front-left，內參對齊見 §2 缺口 4）→ 組成 frame
   ▼
LeRobotDataset.add_frame(...) → save_episode()　（repo_id 帶 sim_ 前綴＋來源 episode id＋DR seed，見 §7）
```

**跟 S4 §4 的差異只在最上面一段**（輸入來源）跟中間插入的 DR 層；下半段（articulation → 渲染 → 存檔）完全共用。

---

## 5. 最容易錯的地方

1. **每個 DR 變體要記錄「來自哪一集真實錄製＋哪一組 DR 參數」**，不然沒辦法追溯，也沒辦法在缺口 1 的回歸出問題時定位是哪個環節錯的。`meta` 裡至少要有 `source_episode_id` 和 `dr_seed`。
2. **不要把「重播出來看起來合理」當成驗收標準。** 同 `sim/README.md` 的教訓——沒校正的增益會讓手臂用力下垂，畫面可能還是「看起來像在動」，但跟真實軌跡的時序/姿態已經對不上。§7 要有量化比對（每幀關節角誤差），不能只用肉眼看。
3. **抓取片段沒做缺口 3 的判定，物體會穿模或凍結在半空。** 如果先出「不含抓取」的簡化版（§2 建議順序），要在資料的 `meta` 裡明確標注這批只涵蓋 approach/retreat 階段，不要讓下游誤用成完整 episode。
4. **DR 參數的隨機範圍要跟 S4 §3 指名的 NVIDIA 教材數值一致**（dome light exposure −4.0~3.0、色溫 2500–9500 K、相機位移 ±0.02 m／±0.05 rad），不要自己憑感覺設一組新的——那是已經有人驗證過合理範圍的起點。
5. **同一份 URDF 被兩邊用**（S1 的 FK 跟這裡的 USD 轉換），S4 §2-3 已經提過：任何一邊改了 `omx_f.urdf`，另一邊要同步檢查。

---

## 6. 輸入（草案）

```
--source-dataset  EricC430/omx_pick_place_pilot_uvc_60   # 真實錄製資料集
--source-episode  0                                       # 或 --all-episodes
--dr-preset       nvidia-so101-default                    # §5 item 4 的參數組
--dr-seed         42
--repo-id         sim_omx_replay_dr_<date>                # 沒有 sim_ 前綴直接拒絕，同 S4 §7
--skip-grasp-frames                                        # 缺口 3 未關閉前的降級模式，見 §2 建議順序
--dry-run
```

---

## 7. 驗收條件

- [ ] 缺口 2 的五姿態對照表（沿用 S4 §5-1 做法）印出來，夾爪開合方向確認正確
- [ ] 缺口 1 的回歸：拿一段真實 `(action, observation.state)` 序列，比較 sim 重播出的關節軌跡與真實 `observation.state` 的逐幀誤差，印出誤差分佈（p50/p95/max），不是只看有沒有跑起來
- [ ] 每個輸出 episode 的 `meta` 記錄 `source_episode_id`、`dr_seed`、`dr_preset`、是否含抓取判定（缺口 3 是否已套用）
- [ ] `repo_id` 沒有 `sim_` 前綴時直接拒絕，同 S4 §7
- [ ] 若缺口 4（相機幾何）未完成，`meta` 明確標注「幾何未對齊，PLACEHOLDER 相機」，不得暗示已對齊——同 S4 §5-5 的立場
- [ ] `verify_dataset.py` 對輸出跑過，退出碼 0，且 `features` 鍵序與來源真實資料集一致

## 8. 明確不做（第一版）

- ❌ 不做完整物理接觸模擬的抓取（§2 缺口 3 的物理路線）——先用 scripted attach/detach，理由見 §2
- ❌ 不做「sim 相機視角精確重現某一次真實錄製」以外的幾何宣稱，除非缺口 4 已完成
- ❌ 不把這批資料拿去訓練或在書審/報告裡宣稱有效，直到 §9 的 D025 閘門條件明確被裁決為已滿足
- ❌ 不擴大到 D030/campA_136sym 之外的 placement 集合——先用已經有真實錄製對應的那些點

---

## 9. 🔴 與 D025 閘門的關係——2026-09-15 新資訊，尚未正式記錄

`[Eric說，2026-09-15 對話中]`：ACT 已針對 Phase B 第一個實驗訓練完成，closed-loop 評估因時間限制只跑了 10 次，成功 5 次。

**這件事跟本規格的關係：**

- D025 執行前提 1（收緊後的版本）是：**「用模擬資料訓練、或在書審／報告裡宣稱模擬有效」的門檻是要先有 Phase B 實機基準成功率。** 這個前提現在**看起來有東西可以滿足了**——但有兩點要先確認，不是我能替 Eric 判斷的：
  1. **這次評估算不算 `phase_plan.md` 定義的 B2**（「訓練 B1 模型，評估相同場景成功率」）？`[未確認]`
  2. **n=10、非 30**——`eval/README.md` D016 已定案的協定是每個物體 30 次評估。10 次的 95% 信賴區間非常寬（Wilson interval 約 24%–76%），**這個 50% 數字目前统計上不夠穩，拿它當「基準」要附上這個警語，不要單獨引用「50%」三個字。**
- 🔴 **這是一筆新的專案事實，不是決策，目前只存在於這次對話裡。** 依 `CLAUDE.md`「A 線的新決策一律寫進 `docs/decisions.md`，不要只寫在 commit message」的精神，**建議 Eric 把這個 Phase B 評估結果正式記錄**（`eval/` 底下的 CSV，或 `docs/phase_plan.md` 的進度欄）——這不是我要不要做的問題，是 Eric 要不要做／要記在哪的問題，**我不會自己去改 `docs/decisions.md` 既有條目**（`docs/specs/README.md` 共同規則第 2 條）。

**如果 Eric 確認上面兩點、且要正式解除 D025 前提 1**，那應該是一條新的 `docs/decisions.md` 條目（例如 D031），而不是回頭改 D025——同一條規則。

---

## 10. Cross-reference

`docs/decisions.md` D021, D025, D029, D030, `docs/specs/S4_sim_teleop_collect.md`, `sim/README.md`, `eval/README.md`, `docs/phase_plan.md`（Phase B 定義）, `configs/replay_omx.yaml`（既有真實→真實 replay 的對照範本）。

---

## 11. 2026-09-18 四個缺口的準備工作（`[柏宇說]`「做s5的四個準備」）

**範圍：只做「準備」——工具與離線分析，不是開始 §4 的 `sim_replay_augment.py` 本體。** §0 的「是否要做 S5」仍待 Eric 裁決。
**這台是 Windows 筆電，沒有 Isaac Sim：** 標 `未在模擬器執行` 的腳本只過了語法檢查（`py_compile`），第一次在 5090 容器上跑很可能要修。

| 缺口 | 產出 | 跑過了嗎 | 結果／還缺什麼 |
|---|---|---|---|
| 共用 | `sim/joint_mapping.py`：LeRobot `.pos` ↔ URDF 弧度 | ✅ 本機（round-trip 誤差 1e-14） | LeRobot 端 `已查證`（`use_degrees=False`、出廠校正 0–4095）。🔴 **8 個符號／零點是 `[未確認]`**，由 S4 §5-1 五姿態對照決定——缺口 1、2 的結果都以它為前提 |
| 2 mimic | `sim/mimic_check.py`：掃 `gripper_joint_1`，量兩指在 link5 座標系的間距與中點，自動判「鏡像／同向」；`--override-gearing` 在 USD 副本上測另一個正負號 | ❌ 未在模擬器執行 | 每個候選 gearing 各跑一次，通過的值回填 `convert_omx_urdf.py --mimic-gearing` 與 `sim/README.md` |
| 1 gains | `scripts/s5_prepare_replay.py`（主機端）→ `traj.npz`＋`real_tracking.json`；`sim/fit_drive_gains.py`（模擬端）每個增益候選一個 env，逐幀比對 sim 關節角 vs 真實 `state[t+1]` | 主機端 ✅（uvc_60，60 集 15966 幀）；模擬端 ❌ | 見下方「真實手臂的追蹤基準」。腳本**不會**寫 `omx_constants.py` |
| 3 attach | `s5_prepare_replay.py` 的 `grasp_segments.csv`（每集夾爪閉合區段）＋下方設計選項 | ✅ 本機 | **決策未做**——選項待裁決 |
| 4 相機 | `scripts/measure_camera_geometry.py`：`markers` / `board` / `rs-intrinsics`（`--save-image` 同時存外參用照片）/ `capture` / `checkerboard` / `extrinsics` / `selftest`；標記擺放點 `configs/camera_markers_campA_136sym_20260918.csv`（6 個分散的 placement 點） | `selftest` ✅（合成影像：相機位置還原誤差 1.77 mm、注視點 0.04 mm）；實機 ❌ | 要 lab day。**墊子上沒有 ArUco**（`make_placement_mat.py` 無此功能）→ 要另外印 `markers` 並照「印出來的上緣朝 +X」擺在墊子已知點。⚠️ **手腕相機只做得到內參**——外參（安裝偏移）要配合手臂姿態的 FK，本工具沒做 |

### 11-1 uvc_60 的離線結果（`[產出物]` `outputs/s5_prep/uvc_60/`，未進 git）

**真實 follower 自己的追蹤誤差** `|action[t] − state[t+1]|`（度，映射 `[未確認]` 但絕對誤差不受正負號影響）：

| joint | p50 | p95 | max | 平均帶號誤差 |
|---|---|---|---|---|
| joint1 shoulder_pan | 0.70 | 2.99 | 8.88 | +0.11 |
| joint2 shoulder_lift | 1.23 | 7.12 | 15.12 | **−1.57** |
| joint3 elbow_flex | 0.53 | 1.58 | 2.90 | −0.54 |
| joint4 wrist_flex | 1.58 | 7.03 | 13.71 | **−2.22** |
| joint5 wrist_roll | 0.09 | 1.14 | 7.65 | −0.11 |

`[AI推論]` joint2／joint4 有一致的帶號偏差，像是承重關節的重力下垂——**這正是 sim 增益要重現的東西**，sim 若在這兩個關節誤差趨近 0，代表太硬，不是比較好。

**映射可疑的跡象 `[AI推論]`：** 用預設映射（符號 +1、零點 0），joint1 範圍 −101.5°～56.9°，超出 `omx_constants` 的現場扇區 −90°～45°；joint3 到 101.1°，超出原廠 +90°。可能是零點偏移，也可能扇區本身不準——**五姿態對照前不要下結論**。

**夾爪：** 張開 ≈59、夾紙杯 ≈47–50（`gripper.pos`）。「數值變小＝閉合」的依據是夾住時 action 比 state 更小（ep 0：−2.4）——`[AI推論]`，手腕相機的極值畫面看不到指尖，**目視沒有確認到**。預設映射下是 −2°～38°，USD 限制 0°～100° → `GRIPPER_ZERO_DEG` 大概要調，由 `mimic_check.py` 印出的指距對照決定。

**抓取分段：** 69 段閉合、60 集都有；9 集有重抓（0, 1, 2, 21, 24, 25, 31, 47, 52）。
**第一次閉合前的幀佔 48%（7685/15966），最後一次張開後佔 18%**——`--skip-grasp-frames` 的簡化版大約能用到 2/3 的幀。
⚠️ `stalled_on_object` 欄位是弱證據：28 段閉合沒有 stall，多數是 leader 本身就停在 ≈49.6，不代表沒夾到。

### 11-2 缺口 3：attach/detach 設計選項 `[AI提議]`，🟡 待裁決

| | A. 腳本化吸附（kinematic attach） | B. 物理抓取（摩擦接觸） | C. 不渲染抓取段 |
|---|---|---|---|
| 做法 | 在 `close_start` 把物體設為 kinematic，之後每步 `物體 pose = link5 pose ∘ 當下的相對位姿`；`close_end` 放開、恢復動力學 | 靠手指碰撞＋摩擦夾住 | `--skip-grasp-frames`，只出 approach／retreat 段 |
| 需要先關閉的缺口 | 1（手臂要跟得上）、2（手指畫面要對） | 1、2，再加碰撞近似與摩擦參數調校 | 無 |
| 抓取段畫面 | 有物體，但物體相對夾爪的位置是 sim 當下的，不是真實的 | 最接近真實，但最容易穿模／滑落 | 無 |
| 可用幀 | 100% | 100%（如果調得動） | ≈66% |
| 風險 | 真實抓取時物體可能被推動過，起點跟 `placement_id` 不同 → 吸附位置偏 | §8 已明列第一版不做 | 策略學不到抓取瞬間的視覺 |

`[AI推論]` 依 §2 的建議順序，C 可以先出、A 是第二步；B 維持 §8 不做。**這是提議，不是決定。**

### 11-3 順帶發現的不一致（沒有改，待確認）

- `sim/scene_constants.py` 寫兩台相機都是 **848×480**，但 wrist 在 2026-09-13 已換成 UVC **640×480**（D022、`configs/record_omx.yaml`、uvc_60 `info.json`）。S4 場景的 `cam_wrist` 解析度與資料集不符，**S4／S5 產出前要改**。

### 11-4 下一步（依賴順序）

1. 5090 容器：`mimic_check.py`（gearing 1.0 與 −1.0 各一次）→ 回填 gearing
2. 實機或 sim：S4 §5-1 五姿態對照 → 回填 `joint_mapping.py` 的 8 個 `[未確認]` 值 → 重跑 `s5_prepare_replay.py`
3. 5090 容器：`fit_drive_gains.py`（先 `--max-frames 60` 試跑，再全跑）→ 與 11-1 的表對照 → 人決定要不要改 `omx_constants.py`
4. lab day：印 `markers`、量 `rs-intrinsics`（front-left）、`capture`＋`checkerboard`（wrist UVC）、`extrinsics`
5. 缺口 3 選項裁決

# S9 — 模擬重播 → 物理驗證 → 環境／物體隨機化資料蒐集

**目標腳本：`scripts/dr_collect.py`**（主控）＋ `sim/physics_replay.py`（驗證）＋ `sim/dr_render.py`（渲染）＋ `sim/dr_objects.py`（物體變體）＋ `sim/config_report.py`（設定檢查）＋ `scripts/sim_replay_augment.py`（匯出 LeRobot）
**設定：`sim/dr_config.json`（隨機化範圍）、`sim/dr_expected_config.json`（物理設定基準）**

## 0. 狀態

- `[EricC 決定 2026-10-31]`：這一階段以「跑通整條流程」為優先，**只用 10/14 之後的資料**（手臂模型和底座同一時期；舊資料的時期差見 `docs/worklog/2026-10-31_wrist_cup_offset.md`）。
- 要涵蓋的內容（EricC 2026-10-31）：場景、光線、物體外觀隨機化的重播軌跡蒐集管線；**重播要在模擬裡驗證有成功夾取才蒐集**；檢查模擬環境參數配置；Isaac Replicator；物體變異（夾取點同樣可行的相似物體、不同外觀顏色材質）；環境變異（地面、室外、光線、環境建置、隨機化）；可替換黑色貼圖的手臂。
- 和 S5 的關係：S5（2026-09，提議中）用「運動學重播＋腳本附著」，抓取不是物理算出來的。這份改成**物理重播驗證後才渲染**，S5 §4 的附著路線不再使用；S5 的匯出階段（`sim_replay_augment.py`、`sim_` 前綴、provenance）沿用並擴充成多集。
- 實測紀錄與數字的正本：`docs/worklog/2026-10-31_sim_dr_pipeline.md`。物理參數總表與「這條管線的設定合不合理」的稽核：`docs/sim_physics_params.md`（2026-11-04）。
- **現況（2026-11-04）**：流程已跑通，`run_20261101`（480 Hz、有底杯、桶子、穿透檢查）驗證 240 個重播、收 186、匯出 372 集到 `sim_dr_cup_20261101`（`verify_dataset.py` exit 0）。**D025 閘門未過，不訓練**。收錄門檻（穿透）待 EricC 決定；未做項目見 §5。

## 1. 流程

```
真實資料（LeRobot，10/14 之後）
  │ scripts/prep_closed_loop_replay.py：關節弧度（10/14 手臂模型：零點＋下垂）、夾爪、擺位點
  ▼
objects   sim/dr_objects.py：物體幾何變體（USD，含 UV、剛體、碰撞、摩擦，有底）；cup_ref 外壁和 assets/paper_cup/paper_cup.usd 逐點相同（該檔沒有底）
  ▼
check     sim/physics_replay.py config_report=sim/dr_expected_config.json：場景實際設定（重力、步長、關節驅動、
          限位、質量、碰撞近似、摩擦、相機、輸入檔雜湊）和「對實機驗證過的設定」比對；不同就停
  ▼
validate  每個（實機成功的 episode × 物體變體）跑一次物理重播（480 Hz，不渲染，記錄每幀所有連桿與物體的世界位姿）；
          場景含桶子與台座（2026-11-01 起）；log 有 PhysX error 的重播不算、最多重跑 2 次
  ▼
score     sim/replay_score.py（task_success）：夾爪閉合期間物體離桌 ≥ 5 cm 持續 ≥ 10 幀，且最後在桶內（＝實機成功的定義）；
          sim/penetration.py（--pen-jobs，CPU）量夾爪連桿和物體壁的重疊，任何一幀 > 1 mm³ 或有連桿穿過就不收；
          accepted() = 實機成功 ∧ 模擬夾起 ∧ 進桶 ∧ 沒穿透
  ▼
render    sim/dr_render.py：照記錄的位姿運動學重渲（接觸、追蹤誤差都保留），每個驗證通過的重播出 N 個變體，
          每個變體獨立抽：物體外觀、環境（戶外 85%／攝影棚 15%，2026-11-05 起沒有房間與牆）、地面材質（人行道、磚、柏油、水泥、草地、廊道石材磁磚、土石落葉）、HDRI（純色地面只留給攝影棚模式、機率 30%，限灰／米／棕／暗綠的土色加小抖動；戶外一律用材質）、燈光、道具、手臂外觀
          （黑色貼圖或 asset 原色）；相機用校正過的鏡頭；Replicator 擷取 rgb＋實例分割
  ▼
export    scripts/sim_replay_augment.py（huggingface/lerobot-gpu 映像）：所有變體 → 一個 LeRobot dataset `sim_<name>`；
          action／observation.state 是原始 episode 的，原封不動；meta/sim_provenance.json 逐集記錄來源與所有隨機參數
  ▼
sheet     contact_sheet.jpg：每個變體 4 幀（front-left＋wrist）
```

## 2. 執行

```bash
python3 scripts/prep_closed_loop_replay.py ~/isaaclab_volume/dr/src/<run> <dataset>     # 一次
python3 scripts/dr_collect.py --work ~/isaaclab_volume/dr/<name> --source ~/isaaclab_volume/dr/src/<run> [--jobs 3]
#  --steps objects,check,validate,render,export,sheet（可只跑一部分；已有輸出的會跳過，可續跑；穿透量測接在 validate 之後，--pen-jobs 控制 CPU 並行數）
#  --episodes 0,1 只跑幾集；--all-outcomes 連實機失敗也驗證（量模擬／實機一致率用）；--dry-run 只印計畫
```

- 工作目錄和來源都要在 `~/isaaclab_volume` 底下（容器掛載成 `/workspace/test_isaaclab`）。容器內寫出的檔案擁有者是 root。
- 隨機範圍全部在 `sim/dr_config.json`：`variants_per_replay`、物體幾何清單、外觀種類權重、環境模式權重、HDRI／MDL 清單、燈光範圍、道具、相機抖動、渲染延遲。
- 種子：每個變體的種子 = sha256(config seed | 物體 | 來源 dataset | episode | 變體編號)，所以同一設定重跑結果相同。

## 3. 設計上的取捨

0. **沒有桌子**（EricC 2026-10-31）：地面就是原本的桌面高度（z = 0），只留手臂底下的檯面，相對高度不變。2026-11-05 起物理重播的桌面也改成同一個平面、沒有邊界（200 m 的靜態碰撞體，頂面 z = 0），物體不會掉出桌緣；渲染端「物體低於 z = 0 就 rejected」只剩安全檢查。重力不隨桌面傾斜（EricC 2026-11-05 決定）。

1. **物理只驗證幾何／物理會變的東西**：物體幾何、質量、摩擦屬於 `objects.geometry`，每個都各自跑物理重播。外觀、燈光、環境只影響畫面，不影響物理，所以一個驗證過的重播可以出多個視覺變體。桌面的物理（摩擦、位置）固定不隨機。
2. **渲染不重算物理**：直接套用驗證時記錄的連桿與物體位姿，所以畫面裡的抓取就是被判定成功的那一次。
3. **隨機值在 Python 抽、寫成一般 USD**；Replicator 負責擷取（render product、rgb 與 instance segmentation annotator、RTX subframe）。這樣每個值都能寫進 manifest 追溯；Replicator 的隨機化圖在觸發時才取值，逐集記錄比較麻煩。
4. **手臂外觀**：`black`＝`assets/omx_f/omx_f_material_override.usda`（實機手臂是黑色），`asset`＝轉檔後的原色；權重在設定檔。
5. **品質檢查**：第一幀全黑的變體標成 `rejected`、不匯出（2026-10-31：兩個 Mug 道具會讓整張變黑，已從清單移除）。

## 4. 驗收條件

- [x] `check` 對基準設定 PASS、對改過的設定 FAIL（自碰撞設定不同的報告判 FAIL，2026-10-31；120 Hz 設定對 11/01 基準 FAIL 3 項，2026-11-01）。⚠️ 11/01 修過一個漏洞：FAIL 時 `app.close()` 先以 0 結束，主控把 FAIL 當 PASS；現在讀 log 的 `CONFIG REPORT PASS`
- [ ] `check` 涵蓋所有會影響物理的設定：**還沒**——機器人、桌面、桶子沒綁物理材質所以報告不列，預設材質變了抓不到；有效接觸偏移也沒讀出（`docs/sim_physics_params.md` §2 #4、#6）
- [x] `validate` 對 cup_ref 重現對實機的一致率（480 Hz、有底杯、桶子，36 集：實機成功 30 集中夾起 27、夾起且進桶 27；實機失敗 6 集中夾起 2；`hz480_all`）。⚠️ 組合不同結果會翻（單場景 28/30、36 場景 25/30，6 集不穩），蒐集一律單場景
- [x] 渲染的幾何對得上實機（front-left／wrist 疊圖，ep00 f0／75／95／125）——**目視**，沒有量化像素誤差
- [x] 匯出的 dataset 能被 LeRobot 讀回，鍵與順序和來源相同，`sim_` 前綴，provenance 逐集完整（e2e：1358 幀；`sim_dr_cup_20261101`：372 集，`verify_dataset.py` exit 0）
- [x] contact sheet 看過（`run_20261101/contact_sheet.jpg`），沒有全黑；兩個會讓畫面變黑的道具已移除、渲染器第一幀全黑標 rejected
- [ ] 收錄門檻（穿透）由 EricC 決定：現行「任一幀 > 1 mm³ 或穿過」收 186／207；只擋穿過 194；深度 ≤ 0.5 mm 且沒穿過 191；≤ 0.3 mm 184
- [ ] 渲染影像和實機的量化對照（亮度已校到 p50 138 附近；幾何誤差 px、色彩、雜訊尚無數字）
- [ ] 夾持餘裕：摩擦配對值掃描、`grip_f` 掃描（`docs/sim_physics_params.md` §2-4）

## 5. 已知限制／未做

- Isaac Sim 6.1（Windows）沒測：`dr_render.py` 只走 Replicator；`physics_replay.py` 的 6.x 路徑本來就有。
- 環境是地面＋HDRI 天空，沒有牆與天花板（2026-11-05 移除房間模式，本專案以戶外部署為主）。戶外周邊物件是 NVIDIA／Isaac 資產庫的灌木、草叢、石頭、落葉、花盆、矮柱燈、鐵欄杆（只做畫面，不擋碰撞）；Isaac 內建環境 USD（Outdoor 等）沒接，因為要先為每個場景找能放手臂的位置。**尚未渲染驗證**：資產 URL 只確認存在於資產庫清單，沒實際載入；道具遮住物體或相機的比例也沒檢查。
- 道具只保證離底座 > 0.52 m（手臂搆不到），沒有檢查會不會擋住相機看物體。
- 物體變體目前都是杯型（錐台殼），夾取方式相同；不同類別的物體要另外設計幾何並驗證。
- 渲染延遲 `render_lag_frames` 預設 0（實機影像約落後 0.75 幀，見 wrist_cup_offset §6）。
- **D025 閘門尚未通過**（EricC 2026-10-31）：sim 資料不訓練、不宣稱有效；也不和實機資料混在同一個 dataset（`sim_` 前綴強制）。
- 自碰撞開、所有連桿 SDF（EricC 2026-10-31）：HOME 時前臂靠在伺服機上、肘讀值超出連桿實際角度 2–5°，只有自碰撞能讓模擬停在那裡；凸分解的手指會互卡，所以要 SDF（worklog 2026-10-31_sim_dr_pipeline）。
- 物理在 GPU 搶資源時會靜默失敗（PhysX error），主控會判無效並重跑；不要和蒐集同時跑其他 GPU 工作。
- 物理步長 480 Hz（2026-11-01）：120 Hz 時手指會穿過 1 mm 的杯壁（測的 5 集每集都有），240 Hz 仍有 2/5，480 Hz 0/5；時間約 4 倍。穿透檢查（`sim/penetration.py`）是收錄條件之一，不靠步長保證。
- 桶子（2026-11-01 起）：`physics_replay.py` 預設放 EricC430 量的桶子與台座，收錄條件改成「夾起且最後在桶內」（實機成功的定義）。桶子高度是 `推論`（11 cm 平台）。~~杯子在夾爪裡轉成水平、30 集只進 14–15 個~~：這是 120 Hz 的現象，480 Hz 後進桶 27/30（worklog §手指穿過杯子），已正式蒐集一次。
- 杯子是剛體（2026-11-04 稽核）：實機紙杯被夾變形、貼平手指；模擬夾持時指面接觸面積小、夾爪停得比實機讀值開 2.6°。對「夾持餘裕」偏樂觀；寶特瓶這個問題更大。選項見 `docs/sim_physics_params.md` §2 #15。
- 摩擦沒有量測來源（杯 μs 1.2／μd 0.9；夾爪、桌、桶沒綁材質）、夾持力（實機夾爪是電流型位置模式）沒讀過，**這兩個決定「模擬夾起」的餘裕**；稽核見 `docs/sim_physics_params.md` §2 #5、#6、#9。
- 薄殼用 SDF 256：杯壁 1 mm 剛好過 2 倍格距；鋁罐、寶特瓶不能照搬（§2 #3）。
- 杯子有底（2026-11-01 起，`sim/shell_mesh.py`）；之前的網格杯底是空的。

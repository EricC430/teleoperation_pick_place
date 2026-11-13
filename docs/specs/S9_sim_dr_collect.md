# S9 — 模擬重播 → 物理驗證 → 環境／物體隨機化資料蒐集

**目標腳本：`scripts/dr_collect.py`**（主控）＋ `sim/physics_replay.py`（驗證）＋ `sim/dr_render.py`（渲染）＋ `sim/dr_objects.py`（物體變體）＋ `sim/config_report.py`（設定檢查）＋ `scripts/sim_replay_augment.py`（匯出 LeRobot）
**設定：`sim/dr_config.json`（隨機化範圍）、`sim/dr_expected_config.json`（物理設定基準）**

## 0. 狀態

- `[EricC 決定 2026-10-31]`：這一階段以「跑通整條流程」為優先，**只用 10/14 之後的資料**（手臂模型和底座同一時期；舊資料的時期差見 `docs/worklog/2026-10-31_wrist_cup_offset.md`）。
- 要涵蓋的內容（EricC 2026-10-31）：場景、光線、物體外觀隨機化的重播軌跡蒐集管線；**重播要在模擬裡驗證有成功夾取才蒐集**；檢查模擬環境參數配置；Isaac Replicator；物體變異（夾取點同樣可行的相似物體、不同外觀顏色材質）；環境變異（地面、室外、光線、環境建置、隨機化）；可替換黑色貼圖的手臂。
- 和 S5 的關係：S5（2026-09，提議中）用「運動學重播＋腳本附著」，抓取不是物理算出來的。這份改成**物理重播驗證後才渲染**，S5 §4 的附著路線不再使用；S5 的匯出階段（`sim_replay_augment.py`、`sim_` 前綴、provenance）沿用並擴充成多集。
- 實測紀錄與數字的正本：`docs/worklog/2026-10-31_sim_dr_pipeline.md`。物理參數總表與「這條管線的設定合不合理」的稽核：`docs/sim_physics_params.md`（2026-11-04）。
- **現況（2026-11-04）**：流程已跑通，`run_20261101`（480 Hz、有底杯、桶子、穿透檢查）驗證 240 個重播、收 186、匯出 372 集到 `sim_dr_cup_20261101`（`verify_dataset.py` exit 0）。**D025 閘門未過，不訓練**。收錄門檻（穿透）待 EricC 決定；未做項目見 §5。
- **2026-11-13 `docs/decisions.md` D033（EricC430＋柏宇）**：閘門判準改成「看影片判斷重播能否復現實機」，不設數字、不逐集查「實機成功、模擬失敗」；收錄門檻改成「可以穿進、不能穿過」（`dr_collect.accepted()` 已改）；訓練前要先讓兩人看場地的樣子；擬真度用目視，不做量化對照。**閘門已判定通過**（EricC430，柏宇 11/13 轉述；判定細節沒有記錄，見 D033）；訓練前仍要先讓兩人看過場地的樣子。

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
          sim/penetration.py（--pen-jobs，CPU）量夾爪連桿和物體壁的重疊，任何一幀有連桿穿過就不收（穿進可以，D033 2026-11-13；重疊體積只記錄）；
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
- [x] 收錄門檻（穿透）：**只擋穿過**（D033，2026-11-13）。11/01 那批的比較：舊規則「任一幀 > 1 mm³ 或穿過」收 186／207；只擋穿過 194；深度 ≤ 0.5 mm 且沒穿過 191；≤ 0.3 mm 184
- [ ] ~~渲染影像和實機的量化對照~~ D033 決定不做量化對照，改由 EricC430／柏宇看影片判斷（亮度已校到 p50 138 附近）
- [ ] 訓練前讓 EricC430／柏宇看過場地的樣子（D033 第 3 點）
- [ ] 夾持餘裕：摩擦配對值掃描、`grip_f` 掃描（`docs/sim_physics_params.md` §2-4）

## 5. 已知限制／未做

- Isaac Sim 6.1（Windows）沒測：`dr_render.py` 只走 Replicator；`physics_replay.py` 的 6.x 路徑本來就有。
- 環境是地面＋HDRI 天空，沒有牆與天花板（2026-11-05 移除房間模式，本專案以戶外部署為主）。戶外地面除了 Base MDL 材質（人行道磚、柏油、水泥、草地、廊道石材磁磚、土石落葉，共 24 種），另有 40% 用 `sim/dr_ground.py` 的程序紋理：疊砌磚、編織磚、方磚人行道、水泥板人行道（板縫）、柏油（碎石粒、裂縫）、廊道磁磚，都是無縫貼圖、真實尺寸。周邊物件是 NVIDIA／Isaac 資產庫的灌木、草叢、石頭、落葉、花盆、矮柱燈、鐵欄杆（26 種，每集 2–7 個，只做畫面，不擋碰撞；實物大小的植物縮到 1.5 m 內）。**顏色**：曝光改在中灰地面上校（深色地面維持深色，原本整張畫面被拉到同一個平均亮度，柏油變淺藍灰），色溫改 4800–7500 K。`scripts/dr_ground_catalog.py` 把每種地面在同一光下各渲染一張，用來檢查。Isaac 內建環境 USD（Outdoor 等）沒接，因為要先為每個場景找能放手臂的位置。**限制**：道具遮住物體或相機的比例沒統計；Hydrangea 因資產缺 Pine_bark.mdl 已移除；Granite_Dark 看起來像水面已移除。
- 道具只保證離底座 > 0.52 m（手臂搆不到），沒有檢查會不會擋住相機看物體。
- 物體變體目前都是杯型（錐台殼），夾取方式相同；不同類別的物體要另外設計幾何並驗證。
  - **寶特瓶（2026-11-11，Boyu＋Claude，`未在模擬器執行`）**：`sim/bottle_mesh.py`（實心旋轉體、空瓶的質量／重心／慣量另外寫進 USD、穿透用「陷入深度」），清單在 `sim/dr_objects_bottle.json`，**不放進 `dr_config.json`**，用 `dr_collect.py --objects sim/dr_objects_bottle.json` 換進工作目錄的 config。實物是御茶園特撰冰釀綠茶 590ml，**尺寸、重量、摩擦全是暫定值（未確認）**，要換成量測值。本機只驗了網格封閉、體積（網格 612.3 ml 對輪廓 614.1 ml）；USD、物理、穿透都還沒在 4090 上跑過。check 步驟用的是預設紙杯資產，對寶特瓶沒有檢查作用；只做站立的瓶子（倒瓶 11 集要先有方向輸入）。（同日稍晚已在 4090 跑物理重播 101 集，結果見 `docs/worklog/2026-11-11_bottle_sim_validate.md`。）
  - **寶特瓶外觀（2026-11-13，EricC430＋柏宇決定範圍，Claude 實作，`未在模擬器執行`）**：瓶子可以有透明區域，**也可以完全透明**；瓶蓋一律不透明；只做空瓶（實物是空的，裝水太重夾不起來）。
    碰撞網格不變（實心），外觀改成只給渲染用的網格：`/PaperCup/Mesh`＝瓶蓋以下的 PET、`/PaperCup/Cap`＝瓶蓋（`sim/bottle_mesh.py` `visual_parts`，`sim/dr_objects.py`）；標籤是 `dr_render.py` 每個變體另外產生的套筒（`label_band`，貼著瓶身外徑、含凹槽、外推 0.3 mm、雙面）。`physics_replay.py` 只對有 `MeshCollisionAPI` 的 `Collision` 設碰撞，所以**物理不受影響**。
    隨機範圍在 `dr_config.json` 的 `objects.bottle_appearance`：PET 無色／淡藍／淡綠、標籤機率 0.7（其餘是完全透明）、標籤高度和位置、瓶蓋顏色。PET 材質預設 OmniGlass 薄壁（`thin_walled`），備案 `pbr_opacity`（OmniPBR 半透明、沒有折射）。
    和 `experiment_spec.md` 的關係：規格把透明寶特瓶放在 L3／P2，那是**實機評估的物體**；模擬資料裡有透明外觀，不代表要拿實機透明瓶評估。
    本機驗過：網格封閉、PET＋瓶蓋體積＝實心體（701.5 ml）、標籤帶兩端開口、標籤取樣與貼圖。**沒驗**：USD 能不能寫出、OmniGlass 的輸入名稱對不對、透明效果像不像——要在 4090 渲染（待驗項目見下一點）。
    11/13 之前產生的物體 USD 沒有 `Cap`，要重新產生。兩種做法：
    - **沿用 11/11 的物理重播（不重跑 validate）**：⚠️ `dr_collect.py` 每次執行都會用**目前的**設定覆蓋工作目錄的 `dr_config.json`（第 165–168 行），直接在舊目錄重跑的話，瓶子外形會變成現在的 v2；當初若用 v1，碰撞體就和驗證時不同。所以要先把舊的 `objects/` 改名保留，用該目錄**原本那份** `dr_config.json` 直接跑 `sim/dr_objects.py config=<那份> out=<work>/objects`，再用 `dr_collect.py --steps render,export,sheet`。測試第 3 項負責確認物理真的沒變。
    - **開新目錄全部重跑**：最乾淨，但 101 集的物理重播要重跑；11/11 實際花了約 4 小時（01:09–05:10，含一次重啟）。
    如果 USD 有 `Cap`、旁邊的 `objects.json` 卻沒有寶特瓶的參數，`dr_render.py` 會直接報錯，不會默默用杯子的外觀渲染。
  - **寶特瓶外觀的渲染測試（待做，未指定人）**。第 1、2 項決定其他項目的結果能不能信，要先做：
    1. **MDL 輸入名稱**：USD 會**默默忽略**不認得的 shader 輸入，所以「log 沒有錯誤」證明不了名稱是對的。在容器裡找 `OmniGlass.mdl`、`OmniPBR.mdl`，確認 `glass_color`、`frosting_roughness`、`glass_ior`、`thin_walled`、`enable_opacity`、`opacity_constant` 都存在；或強制一個很明顯的顏色（`override='{"bottle": {"pet_tint": [1, 0.2, 0.2], "label": false}}'`），看玻璃有沒有真的變紅。
    2. **重跑 `objects`**：USD 裡要有 `Mesh`、`Cap`、`Collision` 三個網格。
    3. **物理不變**：挑 11/11 跑過的 2–3 集重跑 `validate`，`task_success` 和穿透結果要和 11/11 一樣。
    4. **材質 A/B**：同一集、同一個變體種子，`override='{"bottle": {"pet_material": "glass", "label": false}}'` 對 `"pbr_opacity"`，戶外和攝影棚各一次。看完全透明的瓶子像中空的瓶子（看得到後面的背景、邊緣有反光），還是像一根實心玻璃棒（背景整個上下顛倒、放大）。⚠️ 如果有一邊**整支消失或變成實心**，先查即時渲染的設定，不要直接判定那個材質不行：OmniPBR 的透明度在 RTX 即時模式下可能只有「全透或全不透」，要開 fractional cutout opacity（`/rtx/raytracing/fractionalCutoutOpacity`）；玻璃的折射也依賴半透明的渲染設定。這兩點是 `推論`（記憶，沒查證）。
    5. 標籤：滿版和部分標籤各一次，看有沒有閃爍或破面（z-fighting）、有沒有貼著凹槽、從透明部分看得到標籤背面。
    6. 像不像實物：`override` 指定黃色滿版標籤＋黃色瓶蓋，用 `scripts/compare_sim_real_frames.py` 和 `normal_A1` ep0 的實機畫面並排比。
    7. 分割、QA 與速度：透明瓶的 `object_visible_frac` 要合理（遮罩要涵蓋整支瓶子）；自動曝光不受影響；玻璃材質每幀的渲染時間和原本的貼圖比。
    8. 用預設隨機跑 20 個左右的變體，看 contact sheet：透明瓶約佔 30%，整體看起來合理（這也是 D033 第 3 點「訓練前看場地」的一部分）。
- 渲染延遲 `render_lag_frames` 預設 0（實機影像約落後 0.75 幀，見 wrist_cup_offset §6）。
- **D025 閘門尚未通過**（EricC 2026-10-31）：sim 資料不訓練、不宣稱有效；也不和實機資料混在同一個 dataset（`sim_` 前綴強制）。判準 2026-11-13 寫成 D033（看影片判斷），**EricC430 已判定通過**（同日記錄）；訓練前仍要先讓兩人看過場地的樣子（D033 第 3 點）。
- 自碰撞開、所有連桿 SDF（EricC 2026-10-31）：HOME 時前臂靠在伺服機上、肘讀值超出連桿實際角度 2–5°，只有自碰撞能讓模擬停在那裡；凸分解的手指會互卡，所以要 SDF（worklog 2026-10-31_sim_dr_pipeline）。
- 物理在 GPU 搶資源時會靜默失敗（PhysX error），主控會判無效並重跑；不要和蒐集同時跑其他 GPU 工作。
- 物理步長 480 Hz（2026-11-01）：120 Hz 時手指會穿過 1 mm 的杯壁（測的 5 集每集都有），240 Hz 仍有 2/5，480 Hz 0/5；時間約 4 倍。穿透檢查（`sim/penetration.py`）是收錄條件之一，不靠步長保證。
- 桶子（2026-11-01 起）：`physics_replay.py` 預設放 EricC430 量的桶子與台座，收錄條件改成「夾起且最後在桶內」（實機成功的定義）。桶子高度是 `推論`（11 cm 平台）。~~杯子在夾爪裡轉成水平、30 集只進 14–15 個~~：這是 120 Hz 的現象，480 Hz 後進桶 27/30（worklog §手指穿過杯子），已正式蒐集一次。
- 杯子是剛體（2026-11-04 稽核）：實機紙杯被夾變形、貼平手指；模擬夾持時指面接觸面積小、夾爪停得比實機讀值開 2.6°。對「夾持餘裕」偏樂觀；寶特瓶這個問題更大。選項見 `docs/sim_physics_params.md` §2 #15。
- 摩擦沒有量測來源（杯 μs 1.2／μd 0.9；夾爪、桌、桶沒綁材質）、夾持力（實機夾爪是電流型位置模式）沒讀過，**這兩個決定「模擬夾起」的餘裕**；稽核見 `docs/sim_physics_params.md` §2 #5、#6、#9。
- 薄殼用 SDF 256：杯壁 1 mm 剛好過 2 倍格距；鋁罐、寶特瓶不能照搬（§2 #3）。
- 杯子有底（2026-11-01 起，`sim/shell_mesh.py`）；之前的網格杯底是空的。

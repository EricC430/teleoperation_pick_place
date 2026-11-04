# Isaac Sim 物理參數總表與管線稽核

**建立 2026-11-04。** 目的：(1) 列出 Isaac Sim（PhysX）能配置的環境／物體／物理參數並分類；(2) 挑出「多樣環境 × 多樣垃圾撿拾的 sim replay」中會影響虛實一致的項目；(3) 逐項檢核 S9 管線（`sim/physics_replay.py`、`sim/dr_*.py`）現在的設定。
正本分工：管線流程在 `docs/specs/S9_sim_dr_collect.md`；實測數字在 `docs/worklog/2026-10-31_sim_dr_pipeline.md`；**本檔只放「參數清單」與「這些參數現在設成什麼、合不合理」**。基準設定的機器可讀版是 `sim/dr_expected_config.json`。

標示：`已查證`＝這次親自讀到來源或跑到；`推論`＝由已查證的事實推出、沒直接驗；`未確認`＝沒查到或沒量到。

## 0. 怎麼查的、查到多少

| 來源 | 讀了多少 | 用在哪 |
|---|---|---|
| **容器內 Isaac Sim 5.1 的 PhysX USD schema 原始檔**（`omni.usd.schema.physx-107.3.26/.../generatedSchema.usda`＋`PhysxSchemaAddition`，4657＋375 行，逐屬性抽出名稱、型別、預設值、說明） | 全部（抽出 93 個 API／類別、約 480 個屬性；車輛、粒子、可變形、角色控制、相機跟隨等約 300 個與本專案無關，只在 §1-F 一行帶過） | §1 清單與預設值。這是「這個版本實際有哪些參數」的權威來源，不是網頁 |
| NVIDIA Omni Physics 文件：[Collision Behavior Guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/collision_guide.html)、[Rigid Bodies](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/rigid_bodies.html)、[Articulations](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html) | 三頁各讀一次（用抽取式讀取，不是逐行讀完；頁面沒寫到的項目在下面標「文件沒寫」） | SDF 解析度與薄殼、接觸偏移、GPU 緩衝、combine 規則、mimic 調整建議 |
| Isaac Lab 文件：[Tune the PhysX Solver](https://isaac-sim.github.io/IsaacLab/main/source/how-to/solver_tuning_physx.html) | 一頁 | TGS、迭代次數、CCD 與 GPU 動力學互斥、friction 距離參數 |
| PhysX 5.3.1 [Rigid Body Collision](https://nvidia-omniverse.github.io/PhysX/physx/5.3.1/docs/RigidBodyCollision.html)、PxMaterial 搜尋結果 | 只看到搜尋摘要 | 動態三角網格必須有 SDF、SDF 只在 GPU 管線；combine 以「優先級較高者」為準 |
| 論壇（NVIDIA developer forums：stable grasping、heavy objects、friction 搜尋結果） | 只看到搜尋摘要，沒開串 | 社群說法（friction combine 用 max/multiply、夾爪 maxForce 與摩擦要一起調），**不是驗證過的做法** |
| 本專案程式與紀錄 | `physics_replay.py` 全部 746 行；`dr_expected_config.json` 全部；`dr_config.json` 前 120 行；`dr_objects.py`、`config_report.py` 開頭、`dr_render.py` 的設定相關行；worklog `2026-10-31_sim_dr_pipeline.md` 全部；lerobot 映像裡的 `omx_follower.py` 第 140–175 行 | §2 稽核 |

**沒做到的：** Isaac Sim 5.1 的原始碼（C++ PhysX 核心不在容器內，只有 schema 與 Python 腳本）；Isaac Sim 6.1 的 schema（Windows 那台沒開）；沒有用探針實際讀 PhysX 執行時的有效接觸偏移與預設材質（見 §2 的 `未確認`）。

## 1. 參數分類（Isaac Sim 5.1／PhysX 5）

預設值括號內是這台 schema 的值。★＝本專案管線有碰到（改過、設過或依賴預設）。

### A. 世界／場景（`UsdPhysics.Scene`＋`PhysxSceneAPI`）

| 參數 | 預設 | 說明與備註 |
|---|---|---|
| ★重力方向、大小 | (0,0,−1)、9.81 | USD 標準屬性 |
| ★`timeStepsPerSecond` | 60 | 物理步頻。本專案 480 |
| ★`solverType` | TGS | 另有 PGS。機械臂＋接觸多用 TGS |
| `min/maxPositionIterationCount`、`min/maxVelocityIterationCount` | 1/255、0/255 | 全場景對各剛體迭代數的上下限（會覆蓋個別設定） |
| ★`enableCCD` | 關 | **GPU 動力學下不支援**（Isaac Lab 文件：會被強制關掉並警告）。本專案走 GPU，所以不能靠它防穿透 |
| `enableStabilization` | 關 | 文件建議只在 dt > 約 1/30 s 時開；會讓接觸力數值不準 |
| `bounceThreshold` | 0（Isaac Lab 設 0.2） | 低於此相對速度的接觸不彈 |
| ★`frictionOffsetThreshold`、`frictionCorrelationDistance` | 0.04 m、0.025 m | 決定多遠的接觸點還算有摩擦、多近的接觸點併成同一個摩擦錨點。**單位是公尺，預設值是照人類尺度訂的** |
| `collisionSystem`／`broadphaseType`／`enableGPUDynamics` | PCM／GPU／開 | SDF 碰撞體需要 GPU 管線 |
| `enableEnhancedDeterminism` | 關 | 提高確定性、犧牲效能 |
| `enableExternalForcesEveryIteration` | 關 | 提高 TGS 穩定度 |
| ★GPU 緩衝容量：`gpuMaxRigidContactCount`（524288）、`gpuMaxRigidPatchCount`（81920）、`gpuFoundLostPairsCapacity`、`gpuTotalAggregatePairsCapacity`、`gpuCollisionStackSize`、`gpuHeapCapacity`、`gpuTempBufferCapacity` | 見括號 | 固定容量、溢位會掉接觸。Omni 文件：SDF 碰撞會產生大量接觸，範例改成 1,000,000／100,000 |
| `updateType`、`invertCollisionGroupFilter`、`enableSceneQuerySupport`、`enableResidualReporting`、`maxBiasCoefficient`、`reportKinematic*Pairs` | — | 其他 |

### B. 剛體（`UsdPhysics.RigidBodyAPI`／`MassAPI`＋`PhysxRigidBodyAPI`）

| 參數 | 預設 | 備註 |
|---|---|---|
| ★質量 `mass`／密度 `density`、重心 `centerOfMass`、慣量 `diagonalInertia`／`principalAxes` | 密度 1000 kg/m³ 推算；沒碰撞體又沒設質量時 1.0 kg | `已查證`（Omni 文件）。本專案踩過：無質量、無碰撞體的 link 被當 1 kg（`physics_replay.py` 的 `fix_ee_mass`） |
| `linearDamping`／`angularDamping` | 0／0.05 | 剛體阻尼 |
| ★`solverPositionIterationCount`／`solverVelocityIterationCount`（剛體） | 16／1 | 本專案杯子用預設 |
| ★`maxDepenetrationVelocity` | 3 m/s | 解穿透時最大推出速度。文件範例調小可避免「彈出」 |
| `maxContactImpulse`、`maxLinearVelocity`、`maxAngularVelocity`（5729°/s） | inf、inf | 限制 |
| `enableCCD`／`enableSpeculativeCCD` | 關／關 | 見 A：GPU 下 CCD 不可用；speculative 是否可用文件說「Direct GPU API 模式不可」，`未確認`本專案模式 |
| `contactSlopCoefficient`、`cfmScale`（0.025）、`sleepThreshold`、`stabilizationThreshold`、`disableGravity`、`lockedPos/RotAxis`、`retainAccelerations`、`enableGyroscopicForces`、`solveContact` | — | 其他 |

### C. 碰撞形狀（`UsdPhysics.CollisionAPI`／`MeshCollisionAPI`＋`PhysxCollisionAPI` 與各近似 API）

| 參數 | 預設 | 備註 |
|---|---|---|
| ★`approximation` | — | `none`（三角網格，只能靜態）、`convexHull`、`convexDecomposition`、`meshSimplification`、`boundingSphere`／`boundingCube`、`sdf`；另有原始幾何（方塊、球、膠囊） |
| ★SDF：`sdfResolution`（256）、`sdfMargin`（0.01）、`sdfNarrowBandThickness`（0.01）、`sdfSubgridResolution`（6）、`sdfBitsPerSubgridPixel`（16）、`sdfEnableRemeshing`、`sdfTriangleCountReductionFactor` | 見括號 | 格距 ＝ 網格最大外框邊長 ÷ 解析度。**薄於約 2 倍格距的殼可能表示不準**（Omni 文件）；太高吃記憶體與速度，> 1000 少用 |
| 凸分解：`voxelResolution`（500000）、`maxConvexHulls`（32）、`hullVertexLimit`（64）、`errorPercentage`（10）、`minThickness`（1 mm）、`shrinkWrap` | 見括號 | 凹形狀的近似；勾狀手指會被填實（本專案踩過） |
| ★`contactOffset`／`restOffset` | −inf（依形狀大小自動） | 接觸開始作用的距離／靜止時的間距 |
| ★`torsionalPatchRadius`／`minTorsionalPatchRadius` | 0／0 | 扭轉摩擦的接觸面半徑 |
| `collisionEnabled`、碰撞群組（`CollisionGroup`、`FilteredPairs`）、`PhysxTriangleMeshCollision:weldTolerance`、`PhysxMeshMergeCollision`、`PhysxSurfaceVelocityAPI`（輸送帶） | — | 其他 |

### D. 物理材質（`UsdPhysics.MaterialAPI`＋`PhysxMaterialAPI`，以 `physics` 用途綁定）

| 參數 | 預設 | 備註 |
|---|---|---|
| ★`staticFriction`、`dynamicFriction`、`restitution`、`density` | 作者設定；**沒綁材質時的預設 `未確認`**（Isaac Lab 的剛體材質預設 0.5／0.5／0，註明是 PhysX 5 預設，`推論`同值） | |
| ★`frictionCombineMode`、`restitutionCombineMode`、`dampingCombineMode` | average | 一對接觸取**兩個材質中優先級較高的模式**（average < min < multiply < max）。`已查證`（Omni 文件）|
| 柔順接觸：`compliantContactStiffness`、`compliantContactDamping`、`compliantContactAccelerationSpring` | 0（關） | 讓接觸像彈簧，剛體也能有一點「陷入」。`未確認`它和 SDF 碰撞體合用的效果 |
| 其他材質：PBD（粒子）、可變形體／可變形表面材質 | — | 見 F |

### E. 關節與關節鏈（`UsdPhysics.*Joint`、`DriveAPI`、`ArticulationRootAPI`＋`PhysxJointAPI`、`PhysxJointAxisAPI`、`PhysxArticulationAPI`、`PhysxMimicJointAPI`）

| 參數 | 預設 | 備註 |
|---|---|---|
| ★驅動：`type`（force／acceleration）、`stiffness`、`damping`、`maxForce`、`targetPosition`、`targetVelocity` | — | **角度關節的 stiffness 單位是 N·m／度**（本專案 1.5 N·m/deg ＝ 85.9 N·m/rad，`已查證` runtime 讀回）|
| ★關節限位 `lowerLimit`／`upperLimit`；`PhysxLimitAPI` 軟限位 | — | |
| ★`physxJoint:armature`、`jointFriction`、`maxJointVelocity`；每軸：`armature`、`staticFrictionEffort`、`dynamicFrictionEffort`、`viscousFrictionCoefficient`、`maxJointVelocity` | 0、0、1e6 | 本專案全為 0 |
| `PhysxDrivePerformanceEnvelopeAPI`：`maxActuatorVelocity`、`velocityDependentResistance`、`speedEffortGradient` | — | 模擬「馬達轉速越高、可用力矩越低」，本專案沒用 |
| ★mimic：`gearing`、`offset`、`referenceJoint`、`naturalFrequency`、`dampingRatio` | gearing 1、offset 0 | 文件建議 dt × naturalFrequency 約 1、damping ratio ≥ 1 |
| ★關節鏈：`enabledSelfCollisions`（true）、`solverPositionIterationCount`（32）、`solverVelocityIterationCount`（1）、`sleepThreshold`、`stabilizationThreshold` | 見括號 | |
| 殘差回報（`PhysxResidualReportingAPI`）、肌腱（Tendon）、固定基座與否 | — | |

### F. 其他物體類型（本專案目前沒用，列出來是為了判斷「垃圾」要不要走這條）

可變形體／可變形表面（5.1 新舊兩套 API：`PhysxDeformableAPI`、`PhysxDeformableSurfaceAPI`、`PhysxAutoDeformable*`）、PBD 粒子與布料（`PhysxParticle*`、`PhysxPBDMaterialAPI`）、車輛（`PhysxVehicle*`）、角色控制器、觸發區、外力 API、附著（`PhysxAutoAttachmentAPI`）。可變形與粒子需要 GPU 動力學、成本高、校準參數多。

### G. 環境與感測（不是物理引擎，但影響虛實一致）

燈光（Dome／HDRI、Distant、Sphere、Rect、Disk，強度／曝光／色溫）、材質（OmniPBR／MDL：顏色、粗糙度、金屬度、貼圖）、地面與房間幾何、RTX 設定（`rt_subframes`、DLSS、降噪、色調映射、動態模糊）、相機（焦距、光圈、裁切面、鏡頭畸變）、感測雜訊、Replicator 擷取與隨機化圖、時間對齊（影像相對關節讀值的延遲）、影片編碼。

## 2. 稽核：S9 管線現在的設定

**判定：** ✅合理（有證據）／⚠️有疑慮（有線索、沒驗）／❌有問題（已知偏離實機）／❓缺資料。
「現值」都是 `已查證`（`dr_expected_config.json` 的 11/01 基準設定，或程式裡寫死的值）。

### 2-1 對「抓得起來／放得進」直接有影響的

| # | 參數 | 現值 | 虛實一致性的關係 | 判定 | 證據與建議 |
|---|---|---|---|---|---|
| 1 | 物理步頻 | 480 Hz（32 子步／15 fps 幀） | 夾爪合攏時手指穿過 1 mm 杯壁 | ✅ | 120 Hz 5/5 穿、240 Hz 2/5、480 Hz 0/5（worklog）。穿透另用 `sim/penetration.py` 獨立量，不靠步長保證——這個做法是對的。代價 4 倍時間。**新物體（薄壁、快動）要重量一次，不能沿用** |
| 2 | 求解器 | TGS（沒設，用預設）；關節鏈 32／1 迭代；杯子 16／1（預設） | 接觸收斂 | ✅ | 迭代 64 只部分有效（worklog），不是瓶頸。Isaac Lab 文件：先加位置迭代、速度迭代只在彈跳不穩時加 |
| 3 | SDF 解析度與薄殼 | 全部 256；杯壁 1 mm、杯高 9.3 cm → 格距 0.36 mm，壁厚 ≈ 2.75 格 | 薄殼表示是否準 | ✅（杯）／❌（要擴到寶特瓶、鋁罐時） | 文件門檻「殼厚 < 約 2 倍格距可能不準」：杯子剛好過線（`cup_tall` 10.5 cm → 2.4 格；餘裕小）。解析度 1024 對穿透沒改善（worklog），所以杯子的問題不在這。**但寶特瓶壁厚約 0.3 mm（`推論`，未量）、高 20 cm 上下：格距 0.8 mm（256）→ 壁厚不到 1 格，薄殼會漏**。→ 瓶子／罐子不要用 SDF 薄殼：實心剛體用 `convexHull`（罐子）、瓶身用加厚壁或 `convexDecomposition`＋`shrinkWrap`，並用 `penetration.py` 驗 |
| 4 | 接觸／靜止偏移 | 沒設（−inf，自動） | 薄殼被夾時的「陷入量」與起始接觸距離 | ❓ | 3 mm 偏移試過沒改善（worklog）。**但自動值本身沒讀出來**：建議在驗證時用接觸回報（程式已有 `contacts=1`）記 `separation`，把實際有效偏移寫進 config report |
| 5 | **摩擦：杯子** | μs 1.2／μd 0.9（`gen_paper_cup_usd.py` 9/23 預設，**無量測來源**）；`cup_plastic` 0.6／0.45；恢復係數 0.05 | 夾持與推杯的直接決定項 | ⚠️ | 沒綁材質的另一面是 0.5／0.5（`推論`），average → 配對約 0.85／0.7（`推論`）。紙杯塑膠淋膜對 3D 列印塑膠或橡膠，我沒查到數字（`未確認`），Omni 文件說「少有材質超過約 2.0」，1.2 在可接受範圍內但偏上。**旁證**：EricC 看影片指出 ep32 實機第一次把杯子推得比模擬遠（worklog），方向和「模擬摩擦偏高」一致（`推論`，也可能是別的）。**做法**：斜板法量 μs（已列待量）；量完後在 36 集上掃 μ 配對值（例如 0.3／0.5／0.85），看「實機成功模擬夾起」與「實機失敗模擬誤夾」（現 28/30、1/6）怎麼變——模擬在合理摩擦範圍內都夾得起來，這批資料才經得起「實機比模擬難」的質疑 |
| 6 | **摩擦：夾爪、桌面、桶子（沒綁材質）** | 機器人 link、桌面（`Cube`）、桶子都沒有物理材質，走預設 | 夾爪表面材質和夾持直接相關 | ⚠️ | 實機手指是什麼材質 `未確認`（有無膠墊）。**設定檢查的漏洞**：`config_report` 的 `physics_materials` 只列杯子那一個，沒材質的碰撞體不報，所以「預設材質」一旦變了不會被抓到。→ 明確綁材質（手指、桌、桶），並讓報告把每個碰撞體有效的材質列出來 |
| 7 | combine mode | 全部 average（預設） | 配對摩擦＝兩邊平均 | ⚠️ | 規則本身 `已查證`。**問題不在模式，在於我們只有「配對」的量測意義**：建議直接量「杯壁對手指」「杯底對桌面」兩個配對，再反推兩邊材質值使平均等於量測值（或改用 multiply／min 並記錄），不要各自憑印象填 |
| 8 | 質量／重心 | 杯 12 g（`cup_heavy` 30 g、`cup_plastic` 8 g、其他 10–14 g）；機器人連桿質量取自 asset（link0 0.224 kg…link7 0.012 kg）；`end_effector_link` 1e-4 kg | 慣性、關節負載、杯子晃動 | ✅（機器人）／❓（杯） | 機器人的 1 kg 假質量已修（`已查證` 10/14）。杯子 12 g 沒秤過（`未確認`）；DR 質量範圍 8–30 g 沒問題但來源是設定不是量測 |
| 9 | **夾爪驅動與力道上限** | `gripper_joint_1`：stiffness 0.52 N·m/deg（29.8 N·m/rad）、damping 0.026、`maxForce` 0.52 N·m；目標來自錄到的 **action**（leader）；`gripper_joint_2` 用 mimic（nf 300、damping ratio 1.0），自己 stiffness 0、maxForce 1000 | 夾持力。誤差 1° 就滿力矩，所以夾持力 ≈ `maxForce`／指長，不取決於 stiffness | ⚠️ | 0.52 N·m ＝ XL330-M288 在 5 V 的堵轉力矩（`omx_constants.py`；10/12 從匯流排讀到夾爪是該型號）。**但實機夾爪被設成「電流型位置模式」**（`lerobot/robots/omx_follower/omx_follower.py` 第 158 行：`Operating_Mode = CURRENT_POSITION`，註解說用電流上限限制夾持力），實際的 `Current_Limit`／`Goal_Current` 暫存器值我沒讀到（`未確認`；`scripts/read_servo_status.py` 的讀取清單有 `Current_Limit`、`Goal_Current`；夾持當下的實際電流是否讀得到 `未確認`，`docs/execution_plan.md` V3 的「舵機電流 proxy」提過同一件事）。**夾持力是一個沒被量過的核心參數**。做法：(a) 實機讀 `Current_Limit`／`Goal_Current`，並嘗試在夾杯時讀即時電流；(b) 模擬掃 `grip_f`（程式已有此參數）×1／0.5／0.25，看夾起率掉多快——若 0.25 倍還夾得起，代表模擬餘裕過大 |
| 10 | 身體關節驅動 | stiffness 1.5 N·m/deg（85.9 N·m/rad）、damping 0.075、`maxForce` 1.5（肩、肘）／0.52（腕、夾爪）；速度上限 366／618°/s | 追蹤誤差與接觸反推 | ✅（數值）／⚠️（註解） | `maxForce` 與馬達型號對得上（10/12 匯流排讀值，commit `7d48f45`），速度上限 ＝ 馬達無負載轉速。⚠️ `physics_replay.py` 第 264 行註解說「驅動做硬」，但程式只在 `motors=real` 才乘 `stiff_mult`，預設 `motors=asset` 用的是上述值（`已查證` config report）——**註解與行為不一致**，數值是否「夠硬」由力矩上限決定（接觸時抬升落後 > 3°，worklog 已記）。建議改註解，不改值 |
| 11 | 關節摩擦／armature／背隙 | 全 0 | 負載下的抖動與回彈 | ⚠️（低） | 實機齒輪有摩擦與背隙。身體關節是照量到的狀態驅動，背隙已含在讀值裡；只有夾爪的保持力會受影響。低優先，要動時用 `PhysxJointAxisAPI` 的 `dynamicFrictionEffort`／`viscousFrictionCoefficient` |
| 12 | 自碰撞與碰撞體 | 開；link0–7 全 SDF（解析度 256） | HOME 姿勢前臂靠伺服機、手指互卡 | ✅ | 36 集比較（28/30 對 26/30）。仍有 4–5 mm 穿透是實機接觸造成的讀值超出（worklog 結論，EricC430 10/15 觀察）。⚠️ 新增的夾爪外形若有凹處，要重驗互卡 |
| 13 | 關節限位 | 前 5 軸放寬到 ±360° | 避免手臂被限位擋住 | ✅ | 理由：重播的角度是實機到過的。副作用：模擬不會模擬「撞到限位」，重播本來就不會走到 |
| 14 | mimic | nf 300、damping ratio 1.0（asset 原值 25／0.005） | 夾爪兩指是否同步 | ✅ | 文件建議 dt × nf ≈ 1：480 Hz 時 = 0.625；asset 原值 0.05（太軟，10/14 夾爪對稱度差到 −0.34…−0.53 rad） |
| 15 | **杯子是剛體** | 剛體薄殼、1 mm 壁 | 實機紙杯被夾變形、貼平手指 | ❌（已知） | worklog：夾持時模擬指面與杯壁 0.3 mm 內的面積外側約 21 mm²、內側約 3 mm²；夾爪停得比實機讀值開 2.6°（29/29 集）。**選項**（都沒試）：(a) 柔順接觸 `compliantContactStiffness` 在杯材質上，便宜；(b) 5.1 的可變形表面體，貴且參數多。D025 閘門前不建議投資；先知道這項會讓「夾持餘裕」偏樂觀 |
| 16 | 杯子位置（replay 的輸入） | `cup_xy_m` ＝ 擺位點（placement id）的**名目座標**（`prep_closed_loop_replay.py` 第 103 行） | 實機人手擺位有誤差 | ⚠️ | 10/31 wrist_cup_offset：每集約 1 cm 的擺位／偏移。模擬在「名目位置」重播，實機是在「實際位置」夾成功的，所以模擬成功率低估或高估都有可能。**28/30 這個一致率已含這項誤差**，不是額外風險；但要知道「模擬沒夾起」不等於「物理上夾不起」 |
| 17 | 桌面 | `Cube` 200×200 m（2026-11-05 起，等同無限地面）、頂面 z=0、靜態、無材質、`primitive` | 放開後落點、物體不再掉出桌緣 | ✅ | 物理桌面與渲染地面是同一個平面（桌面法向座標系，z=0），沒有邊界。11/01 之前的重播是舊的 1.2×1.4 m 桌面，結果不能直接和新的比。重力仍沿 −z（桌面實測傾斜約 2.25° 不納入，EricC 2026-11-05 決定） |
| 18 | 桶子 | 錐台（底徑 15、口徑 20、高 22 cm、壁 3 mm），靜態三角網格，放在 z=0.11 台座上；位置 EricC430 量、X 方向與高度 `推論` | 「進桶」是收錄條件 | ✅（有資料檢查）／⚠️（高度） | 30 集放開時夾爪離桶心 4–9 cm（口半徑 10 cm），位置對得上。高度 11 cm 沒量，只影響杯子落進去的過程，不影響「進桶」判定的位置條件 |
| 19 | 重力與底座傾斜 | 重力垂直世界；底座 6 DoF（含實測傾斜）由校正檔的 `world_from_link0` 帶入 | 重力在手臂座標的方向 | ✅ | 傾斜進底座姿態而不是進關節零點（S7 的結論），做法正確 |
| 20 | 批次場景的結果會隨組合改變 | 單場景 28/30、36 場景 25/30、12×3 為 26/30，6 集會翻轉 | 同一組資料不同跑法結果不同 | ⚠️ | 已記錄。**蒐集一律單場景跑**（現行流程如此）；`enableEnhancedDeterminism` 沒試過，但它只保證同組合重現，已經成立，不解決「組合不同」。邊緣集數的結論不要引用單次結果 |
| 21 | GPU 緩衝與失敗偵測 | 預設容量；log 掃 `PhysX error` 並重跑最多 2 次 | 靜默壞跑 | ✅ | 多場景那次沒有溢位警告（worklog）。若之後 SDF 物體增多、或一次放更多場景，先看這組 `gpuMaxRigidContactCount`（預設 524288 vs Omni 範例 1,000,000） |
| 22 | 摩擦距離參數 | `frictionCorrelationDistance` 0.025 m、`frictionOffsetThreshold` 0.04 m（預設） | 2.5 cm 內的接觸點併成同一摩擦錨點；杯緣接觸只有 mm 級 | ⚠️（低） | 預設是人類尺度。120 Hz 時杯子在夾爪裡轉到近水平，480 Hz 後消失（worklog），所以這項已不是瓶頸；換成更小物體（瓶蓋、小罐）再回頭試 0.005／0.002 |

### 2-2 對「多樣物體」特別要注意的

| 物體 | 物理上要決定的 | 提醒 |
|---|---|---|
| 紙杯（現有 8 種幾何） | 壁厚、SDF、摩擦、質量 | 見 #3、#5。八種都是同一類錐台殼，夾法一樣；DR 只有兩組摩擦（1.2／0.9 與 0.6／0.45）——是「離散兩個點」，不是在合理範圍內取樣。建議量到實測範圍後改成區間抽樣，讓物理驗證自己篩掉夾不起來的 |
| 鋁罐 | 實心薄壁圓柱，反光 | 壁 0.1 mm 級，**不能用 SDF 薄殼**（#3）；剛體 `convexHull` 即可（外形凸）。空罐約 15 g（`experiment_spec.md`）。材質（鋁對手指）摩擦要另外量 |
| 寶特瓶 | 薄壁、可壓扁、瓶蓋、瓶頸凹 | 壁約 0.3 mm（`未確認`）→ SDF 不夠（#3）；**實機會被夾變形，剛體高估夾持餘裕**（#15 同一類問題，且更嚴重）。是否裝水 `未確認` |
| 透明瓶／塑膠袋 | 透明、可變形 | `experiment_spec.md` 列 P2；剛體做不出來，不在這條管線的範圍 |

### 2-3 環境與感測（不是物理，但同屬虛實一致）

| 項目 | 現值 | 判定 | 說明 |
|---|---|---|---|
| 影像相對關節讀值的延遲 | `render_lag_frames` 預設 0；實測約 0.75 幀（wrist_cup_offset §6） | ⚠️ | 已量到但預設沒套。整幀以下要內插兩個姿態；對訓練的影響 `未確認` |
| 曝光 | 自動曝光到目標灰階區間，實機 p10／p50／p90 = 129／138／142（front-left） | ✅ | 渲染前已校 |
| DLSS | 關（`/rtx/post/dlss/execMode` 0） | ✅ | 避免時間性放大偽影 |
| 鏡頭畸變 | 以填補渲染＋`remap` 套用實測畸變；wrist 相機掛在 link5 子節點 | ✅ | |
| 感測雜訊、動態模糊、白平衡漂移、影片壓縮 | 沒模擬（匯出照來源編碼設定） | ❓ | 實機影像有這些；對 ACT 是否重要 `未確認`。要加的話屬於 S9 `dr_render.py` 的後處理，不在物理 |
| 夾持中夾爪角度 | 模擬比實機讀值開約 2.6°（中位數 0.045 rad，29/29 集） | ⚠️ | 這是 #15 的後果；匯出時 `observation.state` 是實機讀值、影像是模擬角度 |

### 2-4 稽核結論與建議順序

1. **最值得先做（便宜、能縮小最大的未知）**：量摩擦（斜板，杯／手指、杯／桌各一組）與夾持電流（#9），然後做兩個掃描：**摩擦配對值掃描**與 **`grip_f` 掃描**（36 集 × 3–4 組，約 1 小時 GPU，依 `project_time_costs`）。這兩個決定「模擬夾起」有多少餘裕。
2. **設定檢查補洞**（純程式，不需要新資料）：材質綁定與報告列出每個碰撞體的有效材質（#6）；有效接觸偏移讀值（#4）；改 `physics_replay.py` 第 264 行的註解（#10）。
3. **要擴到鋁罐、寶特瓶前**：先決定碰撞近似（#3）並用 `penetration.py` 驗；寶特瓶的剛體限制（#15）要在文件與 provenance 裡標明。
4. **暫不做**：柔順接觸／可變形杯（#15）、關節摩擦（#11）、摩擦距離（#22）——先等 1 的結果，如果餘裕本來就大，這些不會改變結論。

**這份稽核沒回答的：** 模擬資料訓練後對閉環成功率有沒有幫助（D025 閘門未過）；以上「合理」只是「和已量到的實機行為沒有矛盾」，不是「模擬等於實機」。

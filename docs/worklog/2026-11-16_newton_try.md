# 2026-11-16 Newton（MuJoCo-Warp）初試：與 PhysX 的對照，以及要不要全部轉過去

性質：分析（4090，Isaac Lab 3.0.0＋Isaac Sim 6.2.0）。接續 `2026-11-16_isaaclab3_migration.md`。
**範圍很窄：只測了「無物體的手臂追蹤」。沒有測抓杯、薄殼杯碰撞、穿模。** 下面的結論要照這個範圍讀。

## 1. 做了什麼

`sim/fit_drive_gains.py` 加 `--physics {physx,newton_mjwarp}` 與 `--num-envs`：重播真實示範 ep0（`omx_sim/dataset`，9 月的資料、`joint_mapping.SIGN` 仍是 `[未確認]`）的動作，量模擬下一幀關節角對真實錄到的下一幀的誤差，並計時。無相機、無物體、ground＋手臂。增益、力矩上限、dt（1/120）兩邊相同。Newton 設定＝官方「explicit baseline」：`NewtonCfg(MJWarpSolverCfg(solver="newton", integrator="implicitfast"), num_substeps=1, debug_mode=True)`，**沒有調過**。

## 2. 結果（單次量測，無重複）

追蹤誤差（對真實狀態，與環境數無關）：

| 關節 | PhysX p50／p95／max（°） | Newton p50／p95／max（°） |
|---|---|---|
| shoulder_pan | 0.70／1.32／1.77 | 0.70／1.40／10.00 |
| shoulder_lift | 15.88／94.45／115.04 | 1.20／44.90／120.39 |
| elbow_flex | 4.06／24.78／81.68 | 1.29／14.50／87.27 |
| wrist_flex | 7.16／44.29／85.31 | 2.70／10.09／95.73 |
| wrist_roll | 0.15／1.66／4.45 | 0.16／1.75／4.60 |
| gripper | 0.53／0.88／8.40 | 0.87／8.69／29.38 |
| **聚合 p50** | **4.75** | **1.15** |

吞吐量（4264 物理步，含每步 `.item()` 讀回）：

| 環境數 | PhysX env-steps/s | Newton env-steps/s | Newton／PhysX |
|---|---|---|---|
| 1 | 242 | 135 | 0.56× |
| 256 | 68,919 | 211,760 | 3.1× |
| 1,024 | 268,579 | 829,231 | 3.1× |
| 4,096 | 847,357 | 2,909,131 | 3.4× |

## 3. 讀法與限制

- 速度：大量平行時約 3×，與簡報的「2–3 倍」同量級。**1 個環境時 Newton 反而慢**（Python 迴圈主導，不是物理）。只量了無接觸的負載，**接觸多時是否仍是 3×，沒量**。
- 追蹤：Newton 聚合誤差較小，但 **max 與 p95 沒有一致較好**（shoulder_pan max 10° vs 1.8°，gripper p95／max 8.7／29° vs 0.9／8.4°）。力矩上限有被執行（MuJoCo `jnt_actfrcrange = ±1.5／±0.52 N·m`，已查證），所以不是「力矩沒夾」；為何 p50 較好是 `推論`：隱式積分與阻尼處理不同。增益從未對真實手臂擬合（D029），所以「較接近真實紀錄」不等於「Newton 模型較真」。
- 資料是 9 月的 ep0、SIGN 未確認、初始姿態被夾進限位（joint3 1.642 → 1.571 rad，3.0 的 PhysX 初始化不接受超限）。結論不能外推到 10/14 之後的資料。

## 4. Newton 載入我們資產時遇到的

1. `omx_f.usd` 有一個懸空內部參考（`/omx_f/world/visuals` → `</visuals/world>`，虛擬 world 框架的空視覺）。PhysX 靜默忽略，Newton 的 USD 匯入器嚴格報錯。處理：`sim/newton_asset_fix.py` 產生副本 `assets/omx_f_generated_newton/`（只刪這個參考），原資產不動；`omx_scene_cfg.OMX_USD` 可用環境變數切換。
2. Newton 的環境複製要 `replicate_physics=True`（即使 1 個環境）。
3. 夾爪 mimic：Newton 匯入器有讀 `PhysxMimicJointAPI`（`import_usd.py` 約第 5037 行），載入後 10 body／10 joint 與 PhysX 結構一致。**mimic 的實際行為（鏡像、夾持力）沒測。**
4. 官方資產指南列的、對我們相關的差異（未逐項驗證）：MJWarp 預設摩擦滑動較多；`joint_velocity_limit` 不被 MJWarp 執行；厚度 1 mm 的 SDF 薄殼杯與每個連桿的 SDF 碰撞在 Newton 有沒有對應物，沒查。

## 5. 「全部物理引擎轉過去」的評估

**結論：不建議全部轉。** 依工作負載分：

| 工作負載 | 建議 | 理由 |
|---|---|---|
| S9 DR 重播（`physics_replay.py`，有接觸） | **留 PhysX** | 驗證都在 PhysX 上做：10/14 紙杯 28/30 抬起、穿模規則 D033、480 Hz＋SDF 薄殼杯。它是每集一個環境（3 集並行），Newton 的優勢要上千環境才出現，在這裡拿不到。轉過去等於全部重驗，而薄殼杯／夾持力是 Newton 最沒把握的地方 |
| 純運動學／無接觸的批次計算（增益擬合、軌跡掃描） | **可用 Newton** | 已測：3× 且誤差不差；但要先用 10/14 之後的資料與確認過的 SIGN 重做 |
| DR 渲染（`dr_render.py`） | 不受影響 | 用記錄的位姿重渲染，不跑物理 |
| 之後的 RL（若排進計畫；D007 目前拿掉了） | **值得用 Newton 當主要後端** | 上千環境才是它的主場；但必須先有 PhysX↔Newton 的策略轉移檢查（官方有指南），而且接觸行為（抓杯）要先驗過，否則 RL 會學到 Newton 的接觸假象 |
| 跨引擎穩健性 | 建議兩個都留 | 同一批軌跡在兩個引擎都能成功，比只在一個引擎成功更有說服力 |

**要改變這個結論需要的證據**：Newton 上對 10/14 紙杯 7 集（`dr/ctrl_cupref_cl1014`）做同樣的 validate，要 7/7 抬起、在桶內、零穿模、首次夾取吻合——這是升級前後對照用過的同一批，已有 PhysX 基準。需要先寫 Isaac Lab 版的重播腳本（S9 的 `physics_replay.py` 是純 Isaac Sim PhysX 寫法，不能直接換引擎）。

## 6. 沒做

抓杯／薄殼杯碰撞／穿模；mimic 夾持力；Newton 的 solver 調參（njmax／nconmax、cone、impratio、armature）；重複量測取變異；10/14 之後的資料；寶特瓶；Newton 的渲染路徑（只用了 Kit RTX 相機的相容路徑也沒測）。

## 7. 薄殼杯／夾持力／gap・margin 實測（同日下午）：**沒有得到可用答案**

腳本：`sim/newton_cup_replay.py`（Isaac Lab 3.0＋Newton，輸出 `physics_log.json` 格式同 `physics_replay.py`）。10/14 紙杯 ep00，120 Hz 與 480 Hz，接觸來源 `mujoco`（MJWarp 自己的碰撞）或 `newton`（Newton 碰撞管線，網格對網格走 SDF），`--gap`／`--margin`。基準＝PhysX 480 Hz（`dr/ctrl_cupref_cl1014`，ep00：夾爪被杯擋在 +0.8°、第 101 幀抬起、杯高 0.49 m）。

**過程中查到的（已查證，值得保留）**
1. **Newton 模型在 `InteractiveScene(...)` 建構時就建好**（cloner → `import_usd`）。之後再改 stage 的碰撞屬性不會生效；要在 spawn 之後、場景建構完成之前改（本腳本包一層 spawner `func`）。
2. 我們手臂的 `CollisionAPI`＋`convexDecomposition` 掛在 Mesh 上方的 **Xform**（`…/node_STL_BINARY_`），Newton 只從幾何 prim 建 shape：**未處理時手臂在 Newton 裡 0 個碰撞形狀**，手指與杯子完全沒有接觸（第一輪結果就是這樣，全部作廢）。
3. `UsdFileCfg` 沒有 mesh 碰撞近似的欄位（只有 `MeshFileCfg` 有）；SDF 的訊號是 prim 上有 `NewtonSDFCollisionAPI`，近似 token 不能是 `convexDecomposition`（Newton 會用 CoACD 切成 121 塊凸體，薄殼被填滿）。
4. 套用後 Newton 模型：手臂 8 個連桿＋杯子各 1 個 SDF 網格形狀。

**結果（9 組，單次）**：手指**有**碰到杯子（杯子第一次移動在第 83–87 幀，PhysX 是第 88 幀），但**沒有任何一組重現「夾住並抬起」**：

| 組 | 夾爪最小角 | 杯最大位移 | 結果 |
|---|---|---|---|
| PhysX 480 Hz（基準） | +0.8° | 363 mm（被抬起並移到桶） | 抬起 |
| Newton 管線，120 Hz，gap 無／2／5 mm，margin 0／1 mm | −12.6° | 650–5988 mm，杯高最大 0.10–1.7 m | **杯子被彈飛** |
| Newton 管線，480 Hz | −48° | 2780 mm | 手臂追蹤誤差 38°，失穩 |
| MuJoCo 接觸，120 Hz，gap 無／2 mm | −10.0° | 1054–1208 mm | 被彈飛 |
| MuJoCo 接觸，480 Hz | −9.9° | 58 mm | 杯被撞倒，沒有彈飛，也沒有夾住 |

**意思**：這不是「穿過薄壁」，是夾爪沒有被杯子擋住（比基準多閉 10–13°）而杯子被擠出去或撞倒。`gap`／`margin` 在這個設定下**沒有改變結果**，所以「Newton 靠 gap／margin 在 120 Hz 就不穿」**沒有被支持，也沒有被推翻**——這個設定的接觸本身還沒穩定，穿透的問題被蓋住了。沒有對這些日誌跑 `penetration.py`（杯子被彈飛後數字沒有意義）。

**這次比較與 PhysX 基準的差異**（所以不是受控的引擎 A/B）：無自碰撞、無底座墊高與料桶、地面平面取代桌面、mimic 為 Newton 等式約束（非 PhysX 彈簧 300／1.0）、末端連桿零質量修正未套用、SDF 解析度 256 未調、MuJoCo 的 `solref`／`solimp`／`condim`、摩擦、夾爪增益都用預設，**沒有調**。

**下一步（要有人決定值不值得）**：先弄清楚為什麼夾爪沒被擋住——檢查夾爪與杯子的接觸數與力（`debug_mode`）、`condim`、`solref`／`solimp`、杯壁 SDF 解析度與 narrow band；官方資產指南的順序是先用固定抓取狀態、單環境、無隨機化重現第一個壞步驟。**在這之前，不要用這些數字比較 Newton 與 PhysX 的穿透。**

## 8. 補上自碰撞／墊高座／料桶後重跑，並看影片（同日傍晚；**取代 §7 的結論**）

**更正**：§4 與 §7 寫「末端連桿零質量修正沒套」是錯的。`omx_f_real.usda` 已帶著 `omx_usd_patch.py` 寫入的 1e-4 kg，Newton 載入後 `end_effector_link = 0.0001 kg`（`--print-shapes` 印出每個連桿質量，已查證）。§7 另一個前提也不成立：自碰撞、墊高座、料桶**應該要有**，`newton_cup_replay.py` 現在預設都開（`--no-selfcol／--no-riser／--no-bin` 可關）。料桶的支撐座方塊（`/World/Bin_Stand`）沒有進 Newton 模型（12 個形狀裡只有 1 個方塊＝墊高座），只影響放開杯子之後。

**重跑結果（ep00，單次；PhysX 480 Hz 基準：夾爪 +0.8°、第 101 幀抬起、杯高 0.49 m）**

| 組 | 杯最大高度 | 杯最後位置 | 夾爪角度範圍 | 影片裡看到的 |
|---|---|---|---|---|
| Newton 管線 120 Hz，gap 預設 | 0.33 m | z 0.02（倒在地上） | 最小 −62° | 杯子在夾住前被撞倒，躺在墊高座旁 |
| Newton 管線 120 Hz，gap 2 mm | 0.50 m | z 0.31 | 最小 −119° | **杯子被抬起並帶往料桶，高度與 PhysX 相近** |
| Newton 管線 120 Hz，margin 1＋gap 5 mm | 0.50 m | z 0.32 | 最小 −100° | 同上 |
| Newton 管線 480 Hz，gap 預設 | 0.50 m | z 0.15 | 最小 −67° | 抬起 |
| Newton MuJoCo 接觸 120／480 Hz | 0.03 m | z 0.02 | 最小 −9°～−229° | 杯子被撞倒，沒有夾住 |

影片：`outputs/newton_vs_physx_20261116/`（`compare_4runs.webm` 四列對照，另有每組單獨一支；`still_frame100.png`、`still_frame130.png`）。紅利是這些影片用 `sim/render_log_panels.py` 直接畫日誌裡每幀的連桿與杯子位姿，所以看到的就是物理引擎算出來的結果，不是重新模擬。

**能說的**
- 完整場景下 Newton 管線**可以**抬起杯子（3 組到 0.50 m，PhysX 0.49 m），不再被彈飛；上一版「沒有任何一組夾住並抬起」不成立。
- 預設 gap 在 120 Hz 會把杯子撞倒、沒抬起；gap 2 mm（或 margin 1＋gap 5 mm）有抬起。這**符合**「gap 有幫助」，但只有 1 集、每組 1 次，且 480 Hz＋預設 gap 也抬起，所以**不能**說 gap 是決定性的。
- MuJoCo 自己的接觸（`--contacts mujoco`）在 120／480 Hz 都沒夾住。

**不能說的／還是壞的**
- **夾爪關節角度在抬起的組裡落在 −55°～−119°（USD 限位是 0–100°）**，影片 frame 131 的 gap 2 mm 那組顯示 −55°。夾爪在 Newton 裡沒有穩定，「杯子被帶走了」不等於「像 PhysX 一樣被夾住」。原因沒查（懷疑：mimic 等式約束、自碰撞下兩指互推、或接觸衝量把關節推出限位；都只是猜測）。
- 手臂追蹤最大誤差 64–153°（同一個瞬間失穩）。
- **還沒對這些日誌跑 `penetration.py`**，所以「1 mm 杯壁在 120 Hz 有沒有被穿過」這個原始問題**還沒有量化答案**；影片也不能取代它（杯壁 1 mm 在畫面上看不出穿透）。

## 9. 右指（mimic）在 Newton 裡不鏡像：使用者觀察，用日誌證實，原因**還沒找到**

**觀察（使用者，11/16 看過影片）**：mimic 的右指沒有達到和 PhysX 一樣的配置。**已證實**，ep00、`gripper_joint_1`＋`gripper_joint_2`（鏡像時兩者相加＝0）：

| | `j1+j2` 平均／最大（°） | `j1` 最小（°，限位 0–100） |
|---|---|---|
| PhysX 480 Hz 基準 | 0.21／0.4 | +1 |
| Newton 管線 120 Hz，gap 2 mm（預設 mimic） | 15.4／60.7 | −119 |

PhysX 第 96 幀 j1 2.5／j2 −2.8（鏡像）；Newton 第 96 幀 j1 +16.5／j2 **+20.3**（連正負號都反）。第 81 幀還沒碰到杯子：PhysX j2 −33.4，Newton −8.8（j1 兩邊都是 33.7）。

**已查證的**
- Newton 匯入的 mimic 關係是對的：`gripper_joint_2 = 0 + (−1.0) × gripper_joint_1`（`constraint_mimic_coef0/1`），與 PhysX 資產一致。
- 但它被轉成 MuJoCo 等式約束，用**預設的軟參數** `eq_solref = (0.02, 1.0)`（時間常數 20 ms）；PhysX 資產寫的是自然頻率 300 Hz／阻尼比 1.0（時間常數約 0.5 ms，約 40 倍硬）。MuJoCo 的正的 solref 時間常數不能小於 2×dt（120 Hz 時約 16.7 ms），所以用正值不可能做到 PhysX 的硬度；要用**負值**（直接給剛性與阻尼）。
- 可以在 USD 的 `MjcEqualityJointAPI`＋`NewtonMimicAPI` 上用 `mjc:solref` 改（型別是 `double[]`）。`newton_cup_replay.py --mimic-solref A B`。

**試過、沒有解決的（ep00，120 Hz，gap 2 mm，單次）**
| 做法 | 結果 |
|---|---|
| mimic `solref` −5e3／−50 | 與預設相近（`j1+j2` 平均 20.7°） |
| mimic `solref` −5e4／−500 | **更糟**：杯子被彈到 41 m 高 |
| mimic `solref` −5e5／−5000 | 數值爆炸（NaN） |
| 480 Hz＋−5e4 | 最好，`j1+j2` 平均 5.4°，仍遠大於 0.2° |
| 完全不用 mimic，右指獨立 PD（目標＝−左指目標，`--mimic-mode pd`） | **更糟**：`j1+j2` 平均 133°，杯子被彈到 3 m（480 Hz：21 m） |
| 自碰撞關掉（`--no-selfcol`） | 沒抬起杯子；gap 預設時 NaN |
| **杯子移走 500 mm（沒有任何接觸）** | **一樣失穩**：`j1` 漂到 −86°、第 81 幀右指就沒跟上 → **不是接觸造成的** |
| 夾爪指令夾在限位 ≥0（不再持續往限位裡推） | 沒用：`j1` 仍漂到 −81° |

**所以**：mimic 約束不是根因（三種做法——預設、調硬、不用——夾爪都失穩），右指不鏡像是**夾爪整體在 Newton 裡不穩定**的一個症狀；失穩在**沒有杯子時就發生**，`j1` 持續穿過 0° 限位（錄到的指令是 −10°，限位 0°），而且「指令夾在 0」也一樣。已排除：mimic 係數錯誤、約束太軟、接觸、自碰撞、持續往限位推。**沒排除**（都只是猜測、沒驗證）：兩指的 SDF 網格在初始／閉合姿態互相重疊（PhysX 紀錄過凸包重疊會卡住夾爪）、夾爪關節慣量太小（`armature` 為 0）造成 MuJoCo 軟限位／約束行為異常、手指質量與慣量（匯入時有 "Inertia validation corrected 1 bodies" 警告，未查是哪個）、MuJoCo 限位的 `solref`／`solimp` 預設值。

**對決策的意義**：在這個設定下 Newton 還沒有一個穩定的夾爪模型；這是「不全轉 PhysX」結論的又一個證據，也是繼續投入前要先解的第一個問題。

## 10. 找到夾爪不穩的原因（armature）並量到穿透（11/16 晚–11/17）；**取代 §8、§9 的「原因未知」**

**起因**：使用者貼了 Isaac Sim 的 OpenUSD 調校教學（tutorial_04 關節驅動調校，及其前置 tutorial_03 碰撞對）。其中對我們有用的是流程（先關自碰撞確認穩定基準）；Newton 原生 mimic 寫法 `NewtonMimicAPI`、「mimic 預設非順應」、速度上限讀 `newton:velocityLimit` 這三點試過或查過，**沒有改善**（原生 `NewtonMimicAPI` 的結果與 PhysX 轉換版完全相同，MuJoCo 約束仍是預設 `solref (0.02, 1.0)`；MJWarp 也不執行速度上限）。

**診斷過程（已查證，ep00，無杯子的自由運動就能重現）**
1. 自碰撞開：夾爪 `j1` 漂到 −151°；自碰撞關：`j1` 停在指令 −10° 附近（穩定），但**停在限位 0° 之外**，所以限位沒擋住。
2. MuJoCo 模型的夾爪關節限位 `solref_limit = [0.0, 4.1]`（Newton 把力空間的 `joint_limit_ke/kd` 依「慣量倒數」換算出來；夾爪慣量極小所以極端）。覆寫成 `mjc:solreflimit = (0.02, 1)`＋`mjc:solimplimit = (0.99, 0.999, …)`：**沒有改善**。調硬 mimic 約束、拿掉 mimic 改成獨立 PD、把指令夾在限位內：都沒有改善（§9）。
3. 推論（現在被實驗支持）：MuJoCo 的軟約束以有效慣量縮放，夾爪關節 `armature = 0`、手指慣量約 1e-6 量級，而驅動給 0.52 N·m；對這麼輕的指頭，這個力矩壓過任何軟限位／軟 mimic 約束。官方資產指南第 6 點也說 MJWarp 常需要更多 armature。
4. **實驗**：`--grip-armature`，在 reset 後把兩個夾爪關節的 armature 寫成同一個值：

| armature（kg m²） | `j1+j2` 平均／最大 | `j1` 最小 | 杯最大高度 | 杯最後位置 |
|---|---|---|---|---|
| PhysX 480 Hz 基準 | 0.21°／0.4° | +1° | 0.492 | (−0.01, −0.24, 0.14) 在桶內 |
| 0（資產原值，120 Hz） | 15.4°／60.7° | −119° | 0.504 | 不穩 |
| 1e-4（120 Hz，無杯子） | 6.4°／107° | −42° | — | 不夠 |
| **1e-3**（120 Hz） | **0.24°／7.8°** | −1° | 0.502 | (−0.06, −0.24, 0.14) 在桶內 |
| 1e-3（480 Hz） | 0.10°／1.7° | 0° | 0.495 | (0.12, −0.03, 0.13) 沒進桶 |
| 1e-2（120 Hz） | 0.03°／0.4° | 0° | 0.485 | (0.10, −0.02, 0.13) 沒進桶 |

右指鏡像誤差從 15–60° 降到 0.03–0.3°，夾爪不再穿過限位，杯子被夾住、抬起、搬運（影片 `outputs/newton_vs_physx_20261116_armature/`）。**1e-3 是從「微型馬達轉子慣量約 1e-8 × 減速比 288² ≈ 8e-4」估的粗略值，沒有量過**；1e-4 不夠、1e-3 與 1e-2 都可以，所以對數量級不敏感，但真實值未知。進桶與否在 480 Hz 與 1e-2 時和 PhysX 不同（落點是接觸混沌，只有 1 集每組 1 次）。

**穿透（`sim/penetration.py`，同一份 `links.npz` 與杯子定義，armature 1e-3，ep00，單次）**：先確認 Newton 與 PhysX 的連桿位姿座標系一致（同一關節狀態下相差約 0.5 mm，frame 0 暫態 1.5 mm）。

| 組 | 穿過杯壁的幀數（link5／link6／link7） | 最大重疊體積 mm³（link5／6／7） |
|---|---|---|
| PhysX 480 Hz 基準 | 0／0／0 | 0／0／0 |
| Newton 120 Hz，gap 預設 | 27／67／39 | 71／364／405 |
| Newton 120 Hz，gap 2 mm | 42／61／62 | 361／566／402 |
| Newton 120 Hz，margin 1＋gap 5 mm | 6／28／89 | 4／515／140 |
| Newton 240 Hz，gap 預設 | 5／84／40 | 25／323／179 |
| Newton 480 Hz，gap 預設 | 11／59／65 | 13／192／173 |

**結論**：夾爪穩定之後，Newton 在**所有**測過的設定（gap／margin／120–480 Hz）下，手指都**大量穿過 1 mm 杯壁**（每個連桿 5–89 幀，重疊體積最多數百 mm³），而 PhysX 480 Hz 是 0。所以原本的假設「Newton 用 gap／margin 在 120 Hz 就不穿」**不成立**；連 480 Hz 也沒有比較好。D033 的規則（穿過＝不接受）下，這些 Newton 重播全部會被拒絕。
限制：1 集、每組 1 次；SDF 解析度（256）與 Newton 接觸參數（`solref`／`solimp`／`condim`、摩擦）沒調；用的是 Newton 自己的碰撞管線＋SDF 網格；沒測 hydroelastic；沒測更小的 `armature` 或其他值對穿透的影響。

**對決策的意義**：Newton 現在能做出和 PhysX 一樣的夾取動作（夾住、抬起、搬運），但在「手指不穿過薄殼」這個我們最在意的指標上明顯比 PhysX 差，所以 **S9 仍留在 PhysX**；Newton 只適合不看穿透的用途（例如 RL 的粗略批次）。

## 11. 兩個官方頁面＋一段貼上的文字（11/17）：什麼被證實、什麼有效

**來源**：Isaac Sim 官方 `newton_robot_setup_tips.html`（Quick guide: migrating mimic joints）、`hydroelastic_contact.html`；使用者另貼了一段來源不明的文字（說 Newton 有 VBD／Kamino／MJWarp 三種防穿模機制、沒有 PhysX 式 CCD）。**貼上的文字我沒有當事實**，逐項對照。

**官方頁面證實的**
- mimic 頁：`mjc:armature`＝反射到關節的有效轉子慣量，「改善外部驅動關節的求解穩定度」，建議寫在 `physics` 層讓兩個後端都看到。**印證 §10 的根因**（armature）。注意：寫進 physics 層會同時改變 PhysX，S9 的驗證基準會變；本專案的做法是只在 Newton 腳本執行時寫（`--grip-armature`），不動資產。`NewtonMimicAPI` 的 `mimicCoef1 = −1` 為鏡像（我們已是）。
- hydroelastic 頁：兩個形狀都要 `newton:hydroelasticEnabled`、SDF、**封閉網格**；頁面列出的求解器只有 MuJoCo 與 XPBD；**沒有說它能減少或防止穿透**，只描述穿透深度怎麼進入力的計算；`newton:contactGap` 被說「有助避免穿隧」。

**貼上文字的各項**
| 說法 | 查證 |
|---|---|
| Newton 是多求解器框架（MJWarp／Kamino／VBD／XPBD／MPM…） | 本機 `isaaclab_newton/physics/` 有 `mjwarp／kamino／vbd／xpbd／mpm／featherstone`，**已查證** |
| VBD 的 DAT（保守位移截斷）保證布料與夾爪無穿透；`rigid_soft_enable_dat` | VBD 存在；在本機 Newton 1.6.1 搜 `rigid_soft_enable_dat`／`enable_dat` **沒有找到**，**未確認**。VBD 是**可變形體**求解器，我們的杯子是剛體（紙杯真實會變形，PhysX 紀錄也說過這點），只有把杯子改成可變形體才相關，**沒試** |
| Kamino（P-ADMM／DVI）硬接觸減少穿透 | Isaac Lab 有 `KaminoPADMMSolverCfg`／`KaminoDVISolverCfg`，官方調校頁說「接觸多的任務先選 PADMM」；「減少穿透」的說法**我沒驗證，沒試**（是 task 暴露的 preset，實驗性） |
| 沒有 PhysX 式 CCD，靠子步 | 與我們的量測一致（120→480 Hz 沒讓穿透消失）；Kamino 程式裡有 speculative 接觸剔除（`cull_speculative`）；文件原文**沒逐條核對** |
| MJWarp 用 `solref`／`solimp` 控制接觸剛性與穿透恢復 | **實驗證實有效**，見下 |
| 動態剛體需非零質量／慣量 | 與匯入時的 "Inertia validation corrected 1 bodies" 一致，未逐項核對 |

**實驗（armature 1e-3，ep00，單次；穿透＝`sim/penetration.py`；PhysX 480 Hz 基準全部為 0）**

hydroelastic 只能開在**手指 link6／link7＋杯子**（`links.npz` 量到這三個網格封閉；link0／3／4／5 不封閉，link5 正是量測穿透的連桿之一）：

| 組 | 穿過幀數 link5／6／7 | 最大重疊 mm³ link5／6／7 | 杯最高 |
|---|---|---|---|
| 無 hydroelastic，120 Hz，gap 預設 | 27／67／39 | 71／364／405 | 0.50 |
| hydroelastic，120 Hz，gap 預設 | 35／37／35 | 289／441／445 | 0.44 |
| hydroelastic，120 Hz，gap 2 mm | 32／32／33 | 520／380／532 | 0.41 |
| hydroelastic，480 Hz | 35／39／56 | 182／403／487 | 0.46 |
速度：**更正（11/17）** 原先寫「慢約 5 倍」是錯的——當時 GPU 被另一個程序（lerobot，PID 323564，佔 19 GB、使用率 100%）搶走，所有 Newton 重播都被拖慢到 135–170 步/秒（GPU 閒置時約 525–760）。背對背重量（同一時間、同一負載）：不開 144 步/秒 vs hydroelastic 133 步/秒，**約慢 8%**。**小幅減少「穿過」幀數，重疊體積沒有降**，不能解決。

接觸 `solref`／`solimp`（`MujocoCollisionCfg`，套在手臂與杯子所有碰撞體，`--contact-solref／--contact-solimp`）：

| 組 | 穿過幀數 link5／6／7 | 最大重疊 mm³ | 杯最高／結果 |
|---|---|---|---|
| 120 Hz，預設接觸（對照） | 27／67／39 | 71／364／405 | 0.50 抬起 |
| 120 Hz，solimp (0.99, 0.999) | 36／66／129 | 454／266／286 | 0.50 抬起（更糟） |
| 120 Hz，solimp (0.999, 0.9999) | **0／0／3** | **0／0／64** | **0.028：沒抬起，杯子被推到 (0.48, 0.17)** |
| 120 Hz，gap 2 mm＋solimp (0.99, 0.999) | 3／55／50 | 3／382／215 | 0.49 抬起 |
| **480 Hz，solref (0.005, 1)＋solimp (0.99, 0.999)** | 0／0／127 | 0／17／26 | 0.50 抬起 |
| 480 Hz，solref (0.005, 1)＋solimp (0.999, 0.9999) | 1／1／113 | 13／30／44 | 0.49 抬起 |

**讀法**：接觸 `solref`／`solimp` 是目前**唯一明顯降低重疊體積的旋鈕**（480 Hz 從 ~400 mm³ 降到 ~30–45 mm³，同時還抬得起杯子），比 gap／margin／hydroelastic／提高頻率都有效。但**還沒到 PhysX 的 0**：link7 在 113–127 幀仍被判「穿過」（小面積穿出，重疊很小）；120 Hz 時能做到幾乎 0 穿透的那組（solimp 0.999）是因為**根本沒夾住杯子（杯子被推走）**，所以「低穿透」不能單看，要和「有抬起」一起看。
**限制**：1 集、每組 1 次、沒有系統性掃描；只試了 5 組接觸參數；沒測 `condim`／摩擦／SDF 解析度／Kamino／VBD 可變形杯。D033 的「穿過＝不接受」下，這些 Newton 重播目前**仍會被拒絕**（link7 的「穿過」幀數）。

## 12. 接觸 solref／solimp：穿透幀數、回合、速度；D033 的「穿進可、穿過不可、放開不卡」（11/17）

**回合數**：每組只跑了 ep00 **1 個回合**，所以「幾個回合穿透」只能說 1 個回合裡的結果；**沒有回合層級的比率**。要有比率需要在多集上重跑（10/14 紙杯 7 集已有 PhysX 基準 `dr/ctrl_cupref_cl1014`）。

**逐組（ep00，216 幀，armature 1e-3；穿過＝`penetration.py` 的 `frames_through`，各連桿分開計）**
| 組 | 穿過幀數 link5／6／7 | 重疊 >1 mm³ 的幀數 | 抬起 | 最後在桶內 | 放開後 | D033 判定 |
|---|---|---|---|---|---|---|
| PhysX 480 Hz（基準） | 0／0／0 | 0／216 | 是 0.49 | 是 | 夾爪開 34°，離指尖 252 mm | **收** |
| A Newton 120 Hz，預設接觸 | 27／67／39 | 75 | 是 0.49 | 否 | 開，231 mm | 不收 |
| B 120 Hz，gap 2 mm | 42／61／62 | 70 | 是 0.50 | 是 | 開 | 不收 |
| C 120 Hz，hydroelastic | 35／37／35 | 39 | 是 0.44 | 是 | 開 | 不收 |
| D 120 Hz，接觸 solimp (0.99, 0.999) | 36／66／129 | 135 | 是 0.50 | 否 | **夾爪最後仍 −2°（沒打開）** | 不收 |
| E 120 Hz，solimp (0.999, 0.9999) | 0／0／3 | 3 | **否 0.03** | 否 | 夾爪 −4°，杯被推到 (0.48, 0.17) | 不收（沒抬起；link7 仍有 3 幀穿過） |
| F 120 Hz，gap 2 mm＋solimp (0.99, 0.999) | 3／55／50 | 64 | 是 0.49 | 否 | 開 | 不收 |
| G 480 Hz，solref 5 ms＋solimp (0.99, 0.999) | 0／0／127 | 127 | 是 0.50 | 否（落在墊高座上方 z 0.16） | 開，111 mm | 不收 |
| H 480 Hz，solref 5 ms＋solimp (0.999, 0.9999) | 1／1／113 | 117 | 是 0.49 | 否 | 開，133 mm | 不收 |
所有 Newton 組在 D033 下都是 0／1 回合被收。「放開後」只是我臨時的代理量（最後一幀夾爪是否打開、杯子離指尖多遠），**不是正式判準**。

**速度（背對背、同一負載）**：量測時 GPU 被另一個程序佔用（lerobot，19 GB、100%），所以絕對值偏低（閒置時 120 Hz 約 525–760 步/秒）。
- 120 Hz：不加接觸參數 144／146／144 步/秒；加 solimp (0.99, 0.999) 143／145／144。**沒有可測得的差異。**
- 480 Hz：不加 165／165；加 solref 5 ms＋solimp 164／164。**沒有可測得的差異。**
- hydroelastic（120 Hz）：133／133 vs 不開 144／145，**慢約 8%**（見 §11 更正）。
- 接觸 solref／solimp 是常數參數，不增加計算量；它的代價是**穩定性**（過硬可能讓夾持失敗，如 E）而不是速度。

**D033 的對照**：現文已經是「可以穿進、不能穿過」，`accepted()` 只看 `frames_through`；沒有寫的是（一）「不接受＝這個回合去掉」——現行實作確實如此（`accepted()` 為假就不收錄，不是只扣幀）；（二）「穿進後放開不會卡住」——只在理由裡說「480 Hz 放開仍能脫離」，**沒有任何程式檢查**。`accepted()` 的 `in_bin` 條件只能間接擋住「卡在指尖」。若要把（二）變成判準，需要定義量測（例如：夾爪打開指令之後 N 幀內，杯子相對 link7 的位姿不再固定、且之後落進桶內）。**尚未改 `docs/decisions.md`，等確認措辭。**

## 13. 為什麼穿過、為什麼沒進桶：底層接觸資料（11/17）

方法：`newton_cup_replay.py --diag` 每幀讀 MuJoCo-Warp 的接觸資料（`Data.contact`：`dist`、`includemargin`、`friction`、`solref`、`solimp`、`dim`；`nacon`／`naconmax`；`efc`），ep00，armature 1e-3。PhysX 基準的設定來自 `dr/run_20261101/check/config_report.json`（已查證）。

**兩個求解器實際用的設定（已查證）**
| | PhysX（S9 基準） | Newton／MuJoCo-Warp（本腳本） |
|---|---|---|
| 頻率 | 480 Hz（32 子步／幀） | 120–480 Hz |
| 求解 | TGS（預設）；關節求解迭代 位置 32／速度 1 | `solver=newton`（凸優化）、`iterations=100`、`ls_iterations=50`、`cone=pyramidal`、`impratio=1`、`tolerance=1e-6` |
| 接觸模型 | 剛性接觸衝量；碰撞幾何 SDF 解析度 256（全部連桿＋杯） | 軟約束：`solref`（預設 0.02, 1）、`solimp`（預設 0.9, 0.95, 0.001, 0.5, 2）；碰撞幾何 SDF 網格對網格（Newton 自己的碰撞管線，256） |
| 摩擦 | 杯 靜1.2／動0.9；手指用預設材質；兩者合成＝平均（約 0.85／0.7） | 手指 1.0（Newton 預設）、杯 0.9（取到動摩擦）；MuJoCo 合成＝取大（1.0）；`condim=3`（無扭轉／滾動摩擦） |
| 關節 armature | 0（照樣穩定） | 0 會失穩，要 ≥1e-3（§10） |
| 接觸啟用 | — | 每個接觸 `includemargin=0`：`dist<0` 才進求解器；`gap`（預設 10 mm）只是偵測範圍 |

**1. 為什麼會「穿過」**
- **排除**：偵測漏接（接觸存在，`includemargin=0`）；接觸緩衝區容量（`naconmax` 256→2048，穿過幀數沒有減少）；單純提高頻率（預設接觸 120→480 Hz，手指最深穿進 −3.2→−2.6 mm，都深於 1 mm 杯壁）。
- **支持**：接觸是**軟約束**，穿透深度由剛性決定。夾持時手指穿進杯壁最深（mm，杯壁 1 mm）：120 Hz 預設 link6／7＝−2.9／−3.2；480 Hz 預設＝−2.8／−2.6；120 Hz＋solimp 0.99＝−2.7／−3.4；**480 Hz＋solref 5 ms＋solimp 0.99＝−0.58／−0.45**（落在杯壁中間，沒穿過）。杯子對地面也一樣軟：−5.6（120 Hz 預設）、−14.1（480 Hz 預設）、−15.9（120 Hz solimp 0.99）、−1.1（480 Hz＋solref 5 ms）。
- **為什麼 120 Hz 做不到**：MuJoCo 把 solref 時間常數限制在 ≥ 2×dt（120 Hz＝16.7 ms；480 Hz＝4.2 ms），預設 20 ms 在 120 Hz 已經貼著下限，所以 120 Hz 下調 `solimp` 沒有用；要 5 ms 必須 480 Hz。
- **機制（推論）**：MuJoCo 的軟約束是加速度層級、依有效慣量縮放（同 §10 的 armature 機制）。12 g 的杯子與手指有效質量很小，同樣的夾持力造成的穩態穿深就大；PhysX 是剛性接觸＋去穿透，所以 `penetration.py` 量到 0。我**沒有**量 PhysX 的接觸深度。
- 另一個發現：接觸緩衝區 243–251／256 幾乎全滿，但大部分是 **10 mm 範圍內的投機接觸**（link4↔link6、link4↔link7，`act=0`，永遠不啟用）。放大容量沒有改變結果，所以只是浪費不是原因。

**2. 為什麼「沒進桶」——兩個機制，外加一個比較麻煩的**
- **（a）120 Hz，軟接觸 → 杯子在手上滑動／轉動。** 杯子在連桿 5 座標系的位置，f110→f140 的變化：PhysX 3.5 mm；Newton 120 Hz 37–51 mm（傾角在 56°–83° 間晃）。放開點因此散開，落點是碰運氣。
- **（b）480 Hz、夠硬 → 搬運一致，但杯子「掛在手指上」放不掉。** 滑動 2.2 mm（PhysX 3.5 mm），搬運姿態與 PhysX 差 <5 mm。夾爪指令開到 27°（模擬角度 20°→34°，與 PhysX 逐幀一致）後，PhysX 的杯子立刻落下（z 速度 −0.55 m/s，到手指距離 114→300 mm）；Newton 的杯子**到 link7 的距離一直是 115 mm、z 還繼續升（0.374→0.412）**，直到第 164 幀才掉，晚約 16 幀，落點偏離。**是哪個接觸抓住它，沒有量到**（想用診斷延伸到放開之後，但那次重跑沒夾起杯子，見下）。猜測（未驗證）：杯壁嵌進手指鉤子 0.5 mm 加上摩擦 1.0（PhysX 約 0.7–0.85）、且沒有扭轉摩擦。
- **（c）不確定性很大。** Newton 預設 `deterministic_mode=not_guaranteed`。**同一組參數重複跑**：120 Hz 預設接觸 4 次，全都抬起（最高 0.47–0.50 m），但最後位置 4 次都不同（只有 1 次靠近桶）；480 Hz＋solref 5 ms＋solimp 0.99 連跑 3 次，**2 次杯子完全沒動**、1 次抬起。`run_to_run` 模式（需 `MJWarpSolverCfg.disable_sensors=True`）結果可重現（4 次位元一致），但那個結果是**杯子被彈飛到 7.8 m**，而且慢 2.5 倍（57 vs 145 步/秒）。**所以這份紀錄裡每一個「單次」數字（含 §10、§11 的穿過幀數）都帶有未知的變異**；穩健的只有跨設定一致的現象：預設軟接觸下手指穿進杯壁 ≥2.3 mm，以及 480 Hz＋solref 5 ms 在有夾起的那幾次降到 −0.5 mm。PhysX 的重複性我沒有量。

**結論**：兩者差在**接觸的物理模型**，不是偵測或容量：PhysX 用剛性接觸，MuJoCo 用依質量縮放的軟約束，薄殼＋輕物體正好是軟約束最弱的組合；再加上非確定性，Newton 目前無法穩定重現 PhysX 的夾取與放開。

**補充（11/17，背景重複跑完）**：`--det-mode run_to_run`（需 `disable_sensors=True`）在 **480 Hz＋solref 5 ms＋solimp (0.99, 0.999)** 連跑 3 次，結果**位元一致**：杯最高 0.4958 m、最後位置 (0.1388, −0.0273, 0.0248)（抬起，但落在地上、沒進桶），每次 118.8 s（58 步/秒，非確定性模式約 165，慢 2.8 倍）。120 Hz 的 `run_to_run` 同樣可重現，但結果是杯子被彈飛到 7.8 m，所以「可重現」不等於「物理正確」。要比較 Newton 設定而不被非確定性干擾，目前可用的是 **480 Hz＋`run_to_run`**。這也讓「杯子掛在手指上是哪個接觸抓住」可以查（每次都會重現抬起與放開）；尚未做。

## 14. PhysX 的平行效能還有多少空間？Newton 的快從哪來？（11/17）

**量測條件**：`fit_drive_gains.py`（無接觸的手臂追蹤，ground＋手臂，4096／1024 環境，ep0，含每步 `.item()` 讀回）。**GPU 當時被另一個程序佔 96–98%（lerobot，19 GB）**，所以絕對值偏低、倍率也和 GPU 閒置時不同；以下只比同一時間窗內背對背的相對值，每項 2 輪。

| 項目（4096 環境，env-steps/s） | 結果 | 相對 |
|---|---|---|
| PhysX，關節求解迭代 16（`omx_scene_cfg` 原值） | 148.6k／147.6k | 基準 |
| PhysX，迭代 4 | 156.8k／158.5k | +6–7% |
| PhysX，迭代 4＋關 Fabric（`use_fabric=False`） | 159.5k／160.0k | 再 +2–3% |
| Newton MJWarp（CUDA graph 開，預設） | 1.24M／1.25M | 8.4× PhysX(16) |
| **Newton MJWarp，`use_cuda_graph=False`** | **196k／196k** | **只有 PhysX(4) 的 1.24×；是自己開 graph 時的 1/6.4** |
1024 環境同樣：PhysX 迭代 16／4＝39.0–39.4k／42.5–47.2k，Newton 327–329k。
GPU 閒置時（11/16 晚，§2）：PhysX 4096 環境 847k、Newton 2.91M（3.4×）。PhysX 在 GPU 被佔用時掉 5.7 倍，Newton 只掉 2.3 倍，所以**「Newton 快 N 倍」的 N 會隨 GPU 負載大幅變動**（3.4× 到 8.4×）。

**讀法**
- 在這個**無接觸、每個環境工作量很小**的基準上，Newton 的優勢大部分（約 84%）來自 **CUDA graph**（把整個物理步驟錄成一張圖，省掉每步啟動 kernel 的開銷），不是求解器 kernel 本身；關掉 graph 只剩 1.24×。這是依據上表的 `use_cuda_graph` 對照。
- PhysX 在 Isaac Lab 裡沒有對應的 graph 路徑（官方文件沒提供），簡單旋鈕只能擠出幾個百分點到 20%（迭代數、Fabric）。官方手冊另外列的：碰撞體用最簡單的（基本形狀 > 凸包）、關掉不需要的碰撞與自碰撞、物理只用一張 GPU（加 GPU 不會加速物理）、CPU 執行緒數與效能模式。**這幾項我沒測**。
- **推論（未驗證）**：接觸多、每步求解成本高的場景，solver 時間會佔主導，graph 帶來的比例優勢會縮小；NVIDIA 部落格說靈巧手任務 Newton 比 PhysX 快 65%（沒寫硬體，2025-09）與此方向一致。我們的抓取（有接觸、1 個環境）沒有量過 PhysX 對 Newton 的平行吞吐量。

**Newton 除了效能還有什麼（官方 `concepts/solver_differences.rst` 的能力表＋本機實測）**
| 項目 | 依據 |
|---|---|
| 多求解器可換：MJWarp（剛體、廣義座標）、Kamino（含閉鏈、最大座標、硬摩擦接觸）、VBD（布料／軟體／**纜線**）、MPM（流體／顆粒／雪，PhysX 沒有）、XPBD、Featherstone | 官方能力表；本機 `newton/_src/solvers/` 有 coupled／featherstone／implicit_mpm／kamino／mujoco／semi_implicit／style3d／vbd／xpbd |
| 剛體＋可變形體耦合（例：Franka 用 MJWarp、可變形體用 VBD） | 官方：維護中的 Franka 可變形體任務 |
| hydroelastic 接觸（接觸面積分散力） | 官方 hydroelastic 頁；我們實測只小幅改善穿透（§11） |
| 不需要 Isaac Sim 也能跑（kit-less） | v3.0.0 release notes |
| PhysX↔Newton 策略轉移工具與驗證範例 | 官方 `transfer_policies_between_physx_and_newton` |
| 較嚴格的資產檢查 | 實測：抓出 `omx_f.usd` 的一個懸空參考（PhysX 靜默忽略）、匯入時 "Inertia validation corrected 1 bodies" |
| 可微分 | 本機 Newton 的 `Model` 有 `requires_grad` 設定、Isaac Lab 碰撞設定也有此欄位；但 MuJoCo 求解器程式裡沒有出現，Isaac Lab 文件也沒提，**是否可用未確認** |

**我們實測 Newton 比 PhysX 差的地方**（已在 §10–§13）：夾爪關節要加 armature 才穩；接觸是軟約束，薄殼＋輕物體穿深 ≥2.3 mm；結果非確定性（確定性模式慢 2.5–2.8×）；對資產與材質更敏感。

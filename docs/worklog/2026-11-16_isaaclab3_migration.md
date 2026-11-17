# 2026-11-16 Isaac Lab 2.3 → 3.0（Isaac Sim 5.1 → 6.2）移植與驗證

性質：分析＋環境重建（4090 主機）。參與：EricC430（柏宇 2026-11-18 確認）＋Claude。
分支 `feat/isaaclab-3-migration`，2026-11-18 commit。依據：官方遷移指南 `docs/source/isaaclab/migrating.rst`（v3.0.0，約 4975 行，讀了安裝／後端／資產與致動器／資料存取與四元數／視覺化／URDF 匯入器；**Isaac Gym 與 deformables 兩章沒讀**）。

## 1. 環境（已查證）

| 項目 | 升級前 | 現在 |
|---|---|---|
| 主機 OS／kernel | Ubuntu 20.04／5.4 | 24.04.4／6.17 |
| 驅動 | 550.54.14（CUDA 12.4） | 615.71.09（open，CUDA UMD 13.4） |
| Isaac Lab／Sim／Python | 2.3.0／5.1.0／3.11 | 3.0.0／6.2.0／3.12.14 |
| Docker／Toolkit | 28.1.1／1.17.8 | 29.9.3／1.20.2 |

- 驅動是實驗室升級的 `[柏宇說 2026-11-17，更正原本「使用者自己升」]`，OS 是誰升級的未確認；升級後 Docker 的舊 image／容器全部不在（`isaac-lab`、`lerobot-gpu`、`isaac-sim`）。lerobot 訓練 image：寫這份時還沒重建；**11/16 21:08（台北時間）已拉回**，digest `62df079f…`，跟 `docs/environment.md` 釘選的值相同 `[已查證 2026-11-17：docker image inspect 的 RepoDigests／LastTagTime]`。11/16 23:52 起的 aimfix26_twoobj10 訓練用的就是這個 image。
- 新容器：`~/IsaacLab_v3`（clone v3.0.0，`GIT_LFS_SKIP_SMUDGE=1`，主機沒有 git-lfs）→ `python3 docker/container.py build base` → image `isaac-lab-base`（51.4 GB）。常駐容器 `isaac-lab` 以 **root** 執行（舊輸出就是 root 擁有）、掛 `~/isaaclab_volume`。
- 啟動 Python：Isaac Lab 腳本要用 `/workspace/isaaclab/isaaclab.sh -p`（直接呼叫 venv python 會報 "Isaac Sim is not installed"）；純 Isaac Sim 腳本用 `/isaac-sim/python.sh`。`sim/run_in_container.sh` 依腳本是否 `import isaac_boot`／`from isaaclab` 自動選。
- 3.0 的 `add_launcher_args` 不再有 `--headless`／`--enable_cameras`；啟動器仍讀它們。`isaac_boot.add_args` 補回這兩個旗標，所以既有指令列不用改。3.0 預設就是 headless。

## 2. 程式改動（`sim/`）

| 改動 | 檔案 |
|---|---|
| 新增 `isaac_boot.py` 取代棄用的 `AppLauncher`（`launch_simulation`，`require_kit=True`） | 14 個腳本 |
| `.data.*` → `.data.*.torch`（ProxyArray，含相機 `output`／`pos_w`／`quat_w_ros`／`intrinsic_matrices`） | 10 個腳本 |
| `write_*_to_sim` → `_index` 版（關鍵字參數）；`write_joint_state_to_sim` 拆成 position＋velocity；`set_joint_position_target` → `actuators.target_command.set_position_index` | 同上 |
| 四元數：**常數與校正 JSON 維持 WXYZ**（量測紀錄，改了會讓舊校正檔對不上），只在交給 Isaac Lab 處用 `scene_constants.wxyz_to_xyzw()` 轉 | `omx_scene_cfg.py`、`replay_render_episode.py` |
| schema：`RigidBodyPropertiesCfg` 等 → fragment（`UsdPhysicsRigidBodyCfg`＋`PhysxRigidBodyCfg`、`PhysxArticulationCfg`、`UsdPhysicsCollisionCfg`） | `omx_scene_cfg.py` 等 4 個 |
| 致動器：`effort_limit` → `actuator_effort_limit`＋`joint_effort_limit`（2.x 的 implicit 的 `effort_limit` 同時是額定值與 PhysX 夾制，3.0 拆開）；`velocity_limit` → `actuator_velocity_limit` | `omx_scene_cfg.py`、`fit_drive_gains.py` |
| **沒動**：`omx_f.usd`（沿用現有檔，不重轉。新 URDF 匯入器不支援 `convert_mimic_joints_to_normal_joints`、輸出路徑也變；`convert_omx_urdf.py` 只改了啟動，**沒在 3.0 跑過**） | — |
| **沒動**：純 Isaac Sim 腳本（`physics_replay.py`、`dr_render.py`、`render_*`），6.2 的匯入全部成功 | — |
| `scripted_pick_to_drop.py` 只改了本檔；它依賴去年專案 `pickup_place_direct_0203`（不在本 repo，`env._tcp_pose_w()` 等）**未移植、未測** | — |

## 3. 驗證結果

**3.1 移植沒有改變行為（對照組＝原 2.x 程式，靠 3.0 的相容 shim 跑在同一容器）**
`replay_render_episode.py`，ep0，stride 4，134 幀：TCP–物體距離差 **0.000000 mm**、抓取事件相同、front-left 影像**逐像素相同**。

**3.2 四元數轉換必要且正確**
- 手腕相機：原程式在 3.0 把 WXYZ 當 XYZW 讀，光軸變成指向側面、沒有俯角；移植版光軸 (0.788, 0, −0.616)＝朝 link5 +X、俯角約 38°，符合 CAD 註解「俯角 38.7°」。手腕影像因此與原程式不同（平均像素差約 45），這是預期。
- 前視（完整外參）：移植版畫面正常，原程式在 3.0 渲出整面空牆。
- ⚠️ `replay_render_episode.py` 的外參回讀自我檢查**有盲點**：寫入與讀回若同時用錯誤順序會抵銷（原程式在 3.0 回報 0.0000°，移植版回報 0.0433°、門檻 0.05°）。以影像為準，不要只信這個檢查。
- 9/21 的 `s5_tcp_fingertip` 不能當手腕基準：當時 `CAM_WRIST_OFFSET_ROT` 是佔位值 `(0.5,−0.5,0.5,−0.5)`，CAD 值 9/23 才提交（`62ab0ca`）。

**3.3 升級前後對照（真正的基準）：S9 `physics_replay.py`，同程式同參數，Isaac Sim 5.1 → 6.2**
基準＝`dr/ctrl_cupref_cl1014`（11/11，10/14 紙杯 7 集，`cup_ref`，480 Hz，SDF）；新＝`dr/mig3_ctrl_cl1014`。

| 項目 | 結果（舊→新） |
|---|---|
| 抬杯起始幀 | 7 集差 ≤1 幀（例 ep00 101→101） |
| 最大杯高 | 差 ≤1 cm |
| 最後落在桶內 | 7/7 → 7/7 |
| 杯子最後位置 | 落點到桶中心 12–73 mm → 26–59 mm；整條軌跡平均差 2–25 mm（接觸混沌） |
| 關節追蹤 | ep00／15／17／26 平均差 0.04–0.7°；**ep14／24／27 平均差 5–16°** |

ep14／24／27 的差異是**舊環境沒追上指令**：ep27 基準的 joint3 在前約 65 幀卡在約 107°（指令已到 −50°），新環境從第 0 幀就跟著指令。這批集數正是 11/11 對照腳本註解的「10/14 紙杯集早期漂移」。為何 6.2 不再卡住 `未確認`（可能是 PhysX 版本；沒查）。`q_cmd`（輸入）新舊相差 0。
**正式評分（`dr_collect.py --steps objects,validate`，主機端用 `~/.venvs/omx-host`；同一批 7 集第二次獨立重跑，覆蓋了上表所用的第一次 `physics_log.json`，所以上表是第一次、本段是第二次）**：sim lifted 7/7、真實成功者 7/7、首次夾取吻合 7/7、抬起且在桶內 7/7、無穿模 7/7（>1 mm³ 的重疊也是 0）、accepted **7/7**；升級前基準同樣 7/7。逐集 `max_obj_z` 差 ≤1 cm。第一次與第二次的逐幀重複性**沒有比**（第一次的日誌已被覆蓋）。
限制：只比了 7 集、單一物體（紙杯）；寶特瓶沒比。

## 4. 沒解決的事

1. **`preview_scene` 關節漂移 51.10°**（120 步、home 姿態）：S4 文件記的 2.x 值是 19–35°，但那是 9 月初，之後 USD 與增益都改過；2.3 環境已不存在，**無法判定是否退化**。增益與力矩上限有正確套用（印出的表：joint1–3 剛性 85.9、力矩 1.5）。
2. **渲染模式**：`2026-11-14_bottle_render.md` 寫 Isaac Sim 5.1 預設是舊即時模式，寶特瓶要手動開 RTX Real-Time 2.0。**6.2 的預設渲染模式沒查**，`dr_render.py` 的亮度／外觀可能與升級前不同；DR 渲染輸出**沒有重跑對照**。
3. 主機沒有 numpy／OpenCV／python3-venv：已解決——使用者批准後，從容器 `docker cp` 同版 uv 0.12.9 到 `~/.local/bin/uv`，建 `~/.venvs/omx-host`（numpy 2.5.5、opencv-python-headless 5.0.0）。主機端腳本要用 `~/.venvs/omx-host/bin/python` 跑。
4. 警告「Actuator groups cover 6 joints; expected 7」：`gripper_joint_2`（mimic）不在任何致動器群組，與 2.x 設計一致，但 3.0 才報警；`mimic_check.py` 顯示鏡像正常，視為無害 `推論`。
5. 硬碟 95%（剩約 49 GB）。已在容器內刪除 `omx_sim` 的 54 個 9 月渲染資料夾（2.2 GB；其中 9 個被文件當產出物引用，文件沒改）。
6. 還沒做：`verify_grasp_attach.py`、`render_state_replay.py`、`fit_drive_gains*.py`、`inspect_mimic_axis.py`、`audit_usd.py`、`gen_paper_cup_usd.py`、`convert_omx_urdf.py` 在 3.0 的實跑；寶特瓶集的對照；手腕相機與真實影像的對照（`scripts/compare_sim_real_frames.py`）；`docs/environment.md` 更新；lerobot 容器重建；3.0 的 RL／Newton 相關（未碰）。

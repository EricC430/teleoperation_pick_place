# S1 — Reach logger

**目標檔案：`scripts/reach_logger.py`**
**正本依據：`docs/decisions.md` D023「Three scripts to build before the next lab day」第 1 支，
＋ D026（改用 FK 量測）＋ D023 §2026-08-31（33/43 澄清）**

---

## 1. 這支腳本要解決的問題

D023 要在下一個 lab day 量出數字，填進 `docs/experiment_spec.md` §3：

1. **傳輸線限制哪個方位角範圍**（cable constraint，sweep 底座得到）
2. **`r_outer`：還能做 top-down / 斜向下爪的最遠半徑** —— 這是實驗真正要的外界線
3. **`r_inner`：還能下爪的最近半徑**
4. （供參）**`r_outer_side`：手臂伸直、只能側夾的最遠可及半徑** —— 記錄用，不進 §3

> 🔴 **D023 §2026-08-31 `[Eric說]`：當初「33–43 cm」不是環帶、也不是方位角相依。**
> **43 cm = 手臂伸直最遠可及、該處只能側夾；33 cm = 還能從上方下爪的最遠距離。**
> 所以舊版 §6 的「最差方位角 r_max → 裁決 (a) 環帶 vs (b) 依方位角」**整段作廢**。
> 現在要分開量的是：① top-down 可夾半徑 ② side-only 可及半徑 ③ cable 卡住的方位角。

**沒有這些數字，S2 的凍結清單不能產生，A7 不能凍結，A8 不能錄。**

## 2. 🔴 已查證的事實（2026-08-30 / 2026-08-31，不要重查）

| 事實 | 出處 |
|---|---|
| `OmxFollower` 只暴露關節角，沒有末端位姿 | `lerobot/src/lerobot/robots/omx_follower/omx_follower.py:173` `get_observation()` 只回 `{motor}.pos` |
| 馬達名稱固定為 6 個 | 同檔 `:53` — `shoulder_pan` / `shoulder_lift` / `elbow_flex` / `wrist_flex` / `wrist_roll` / `gripper` |
| **`omx_f` URDF 存在且對得上本臂** | `ROBOTIS-GIT/open_manipulator` → `open_manipulator_description/urdf/omx_f/omx_f.urdf`。5 個 revolute 臂關節 + 2 個夾爪關節。`joint1` 軸 `(0,0,1)` = 垂直底座 yaw = `shoulder_pan`；`joint2/3/4` 軸 `(0,1,0)` = `shoulder_lift` / `elbow_flex` / `wrist_flex`；`joint5` 軸 `(1,0,0)` = `wrist_roll`。link origin 是實際公尺值。`[已查證 2026-08-31 讀 raw URDF]` |
| **LeRobot 內建 FK** | `lerobot/src/lerobot/model/kinematics.py` `RobotKinematics(...)` → `forward_kinematics(joint_pos_deg)` 回 4×4 pose。需要 `placo` |
| 🔴 **`placo` 在 Windows 裝不起來 → 改用手刻 FK** | `uv pip install "placo>=0.9.6,<0.9.16" "cmeel-urdfdom>=4,<5" …` 失敗：`cmeel-urdfdom` 無 Windows wheel、走 CMake 原始碼編譯、`cmeel-console-bridge` 子建置報錯（2026-08-31 實測）。**D026 的 fallback 分支，現在是正式實作方式，不是備案。** `RobotKinematics`/placo 路線留給「reach logger 在 Linux 機器上跑」時用 |
| **URDF 已取得並讀完** | `assets/omx_f/omx_f.urdf`（10965 bytes，`[已查證 2026-08-31 讀完整份]`）。**所有 joint `rpy="0 0 0"`** → FK = 平移到 origin、繞軸轉、重複 5 次、再加固定 EE offset |
| `matplotlib` / `pytest` | 已裝進 venv，無問題 |
| `DynamixelMotorsBus` 有 `disable_torque()` 與 `torque_disabled()` context manager | `omx_follower.py:119`、`:143` —— ⇒ 模式 B（手動扳）可實作，不必留 stub |
| URDF 的 joint limit 是佔位值 | `omx_f.urdf` 每軸 `±6.28`、effort `1000`、velocity `4.8`。**FK 幾何可用**，動態資訊無效 |
| follower 校正檔是出廠預設 | `calibration/2026-08-24_omx_follower_arm.json` 全部 `homing_offset 0` / `range 0–4095`。⇒ URDF 關節零 ≠ 馬達 encoder 零，**每軸 offset 要用 §5 的 5-pose 測試量一次** |

## 3. FK：手刻變換鏈（D026，placo 在 Windows 不可用）

**正式做法**（不是備案）：直接用 `assets/omx_f/omx_f.urdf` 的 link origin 疊 5 段齊次變換。
所有 joint `rpy="0 0 0"`，所以每一節就是「平移到 origin、繞該軸轉 qᵢ」。

已查證的關節表（URDF 順序，公尺）：

| # | URDF joint | → 馬達 | origin xyz | 軸 |
|---|---|---|---|---|
| 1 | `joint1` | `shoulder_pan`  | `(-0.01125, 0, 0.034)`   | z |
| 2 | `joint2` | `shoulder_lift` | `(0, 0, 0.0635)`         | y |
| 3 | `joint3` | `elbow_flex`    | `(0.0415, 0, 0.11315)`  | y |
| 4 | `joint4` | `wrist_flex`    | `(0.162, 0, 0)`         | y |
| 5 | `joint5` | `wrist_roll`    | `(0.0287, 0, 0)`        | x |
| — | `end_effector_joint` (fixed) | — | `(0.09193, -0.0016, 0)` | — |

`T = Tz(1)·Rz(q1) · Ty(2)·Ry(q2) · Ty(3)·Ry(q3) · Ty(4)·Ry(q4) · Tx(5)·Rx(q5) · T(ee)`
（`T(k)` = 平移到第 k 個 origin；`gripper` 不進 FK，link5 就是 EE 的父）

- **`radius`** = EE 在 base XY 平面對 `joint1` 軸的距離 = `hypot(x_ee + 0.01125, y_ee)`
  （`joint1` 軸在 base 座標的 XY 交點是 `(-0.01125, 0)`）
- **`azimuth_base`** = `atan2(y_ee, x_ee + 0.01125)`，wrap 到 (−180, 180]
- 自我驗證：① 手算閉式（**零姿勢** EE = `(0.31288, -0.0016, 0.21065)` m，`radius ≈ 32.413 cm`，`tests/test_fk.py`）② **`urchin`（純 Python URDF FK，Windows wheel 正常）當獨立 oracle**，隨機關節組態比對 4×4 transform 到 1e-9（`tests/test_fk_vs_urchin.py`）。urchin 只是 `dev` 依賴，不進執行路徑

**輸入單位**：`normalize=False` 讀 `Present_Position` → `rad = ticks × 2π / 4096`，wrap。
**不要用 `.pos`**（會夾到 [0,4095]，EXTENDED_POSITION 的 ticks 會 wrap/負值 → 安靜給錯）。
每軸零位差 + 方向由 §5 的 5-pose 測試量出，存進校正並套用。

**smoke-test**：裝 `matplotlib` 後 `uv run lerobot-teleoperate --help` 要照常過（已確認）。

## 4. 執行模式

**模式 A（預設，`--mode teleop`）：腳本自己跑 leader→follower 迴圈。**
序列埠獨占，不能一邊跑 `lerobot-teleoperate` 一邊跑這支。

**模式 B（`--mode follower-only`）：只連 follower，由人手動扳。**
用 `bus.disable_torque()` 關扭力（已查證可行，§2）。進迴圈前印一行警告：扭力已關、手臂會軟。

## 5. 🔴 FK 驗證：5-pose 對照（沒過就退回捲尺）

進主量測迴圈前強制做一次，方法同 `docs/field_manual.md` §4-1 / `S4` §5-1：

| 姿態 | 做什麼 | 比對 |
|---|---|---|
| home | 全關節歸零附近 | 記 FK 的 EE `(x,y,z)` |
| +J1 | 只轉 `shoulder_pan` 一個已知量（如 +30°） | FK 預測的 EE 水平位移方向／量值 vs 實測 |
| +J2 | 只轉 `shoulder_lift` | 同上 |
| +J3 | 只轉 `elbow_flex` | 同上 |
| 夾爪開闔 | 不影響 EE，確認 FK 不動 | — |

- 逐格印成表，**人看過打勾才算過**。
- 過關判準：FK 預測與實測差 **≤ 1 cm**（可修的固定 offset 先修進去再判）。
- **沒過** → 進 `--fk-fallback tape` 模式：radius 改由捲尺人工輸入，summary 標明 `fk_validation: failed`，`method=tape`。D026 的 reverse-if 就是這條。

## 6. 方位角框架與 `reference`

- FK 給的是 **base frame**：`radius = hypot(x, y)`、`azimuth_base = atan2(y, x)`。base 原點就是 `joint1` 軸 = 底座旋轉中心，所以 `radius` 已經是「相對旋轉中心」。
- **`reference` 樣本（建議做，不是可選）**：把 S3 的極座標墊實體對齊底座 → 開任意姿勢 → 讀墊子上手臂指向的角度、輸入 → `azimuth_offset_deg = azimuth_mat_read − azimuth_base`。之後 `azimuth_mat = azimuth_base + azimuth_offset_deg`。
- 為什麼要：S2 的 seeded 點、S3 的墊子都以墊子框架表示。沒有這一筆，方位角只有 base 框架的相對值，S3 要靠「開 pan 到某值看手臂指向」土法對位 —— 對粗界線夠、對可比性配置點不夠。
- 沒有 `reference` 樣本時，summary 要明講「方位角為 base 框架、未對到墊子」，**不要假裝是絕對角**。

## 7. 輸入

```
--config      configs/teleoperate_omx.yaml     # 沿用既有 port / id / calibration_dir
--urdf        assets/omx_f/omx_f.urdf
--ee-frame    end_effector_link                # 先確認實際名稱
--out         analysis/reach_log_YYYY-MM-DD.csv
--mode        teleop | follower-only           # 預設 teleop
--fk-fallback off | tape                        # 預設 off；5-pose 沒過時改 tape
--fps         30
--dry-run                                       # 不碰硬體、不裝 placo，印出會做什麼就結束
```

## 8. 互動與輸出

**鍵盤（`pynput`，已在 `hardware` extra）：**

| 鍵 | 動作 |
|---|---|
| `o` | 記一個 **outer / top-down**：還能從上方或斜向下爪的最遠點 |
| `s` | 記一個 **outer / side-only**：手臂伸直、只能側夾的最遠可及點（供參） |
| `i` | 記一個 **inner**：還能擺出可夾姿態的最近點 |
| `a` | 記一個 **azimuth-limit**：掃到傳輸線卡住的角度 |
| `r` | 記 **reference**：墊子已對齊底座，輸入手臂目前指向的墊子角度 |
| `q` | 存檔離開 |

按鍵後在終端提示（不要吃掉鍵盤迴圈）：

```
[outer] FK: r=31.8 cm  azimuth_base=-32.6°   備註 (可空) > 線在這個方向最卡
[reference] 墊子角度讀數 (deg) > 12.4
```

`--fk-fallback tape` 時，`o/s/i` 會多問一句 `捲尺讀數 (cm) >`。

**CSV 欄位（`analysis/reach_log_<date>.csv`）：**

```
ts_iso, sample_type, method, radius_cm, ee_x_cm, ee_y_cm, ee_z_cm,
azimuth_base_deg, azimuth_mat_deg,
shoulder_pan_pos, shoulder_lift_pos, elbow_flex_pos, wrist_flex_pos, wrist_roll_pos, gripper_pos,
note
```

- `sample_type ∈ {outer_topdown, outer_side, inner, azimuth_limit, reference}`
- `method ∈ {fk, tape}` —— 哪個產生了 `radius_cm`
- `azimuth_mat_deg` 在有 `reference` 前留空，summary 階段回填

## 9. 結束時要印 / 產出

**(1) 終端摘要**

```
樣本數: outer_topdown=7 outer_side=3 inner=5 azimuth_limit=4   (reference: 有)
FK 驗證: 通過 (5-pose, 2026-08-31, 最大誤差 0.6 cm)
方位角框架: base + reference offset = +12.4°  → 以下為墊子框架

受限方位角 (cable):        -35.2° ~ +88.6°
r_outer (top-down, 取最小):  31.8 cm  @ azimuth -20°
r_outer (side-only, 供參):   43.1 cm
r_inner (可下爪):            18.9 cm

→ 建議填入 experiment_spec §3（人看過再手動填）:
   r_outer = 31.8 − <margin>       ⚠️ margin 由人決定，腳本不自動減
   r_inner = 18.9
   azimuth = [-35.2, 88.6]  (墊子框架)
⚠️ D023(2026-08-31): 33/43 是抓取進場角差別，不是環帶；r_outer 用 top-down 值。
```

**(2) `analysis/reach_plot_<date>.png`** —— 極座標散點：`outer_topdown` / `outer_side` / `inner` / `azimuth_limit` 四種 marker；陰影標出粗略可達扇環 `[azimuth_min, azimuth_max] × [r_inner, r_outer_topdown]`。角軸有 `reference` 時用墊子框架、否則用 `azimuth_base` 並在標題大字寫 `UNCALIBRATED — base frame`。半徑軸 cm。實作前 `uv run python -c "import matplotlib"` 確認。

**(3) `analysis/reach_summary_<date>.json`** —— 給 S2 `--from-summary` 讀：

```json
{
  "generated": "<iso>",
  "source_csv": "analysis/reach_log_2026-08-31.csv",
  "git_commit": "<sha>",
  "fk_method": "placo",
  "fk_validation": {"passed": true, "date": "2026-08-31", "max_error_cm": 0.6},
  "azimuth_frame": "mat",
  "azimuth_offset_deg": 12.4,
  "r_outer_topdown_cm": 31.8,
  "r_outer_topdown_worst_azimuth_deg": -20.0,
  "r_outer_side_cm": 43.1,
  "r_inner_cm": 18.9,
  "azimuth_min_deg": -35.2,
  "azimuth_max_deg": 88.6,
  "sample_counts": {"outer_topdown": 7, "outer_side": 3, "inner": 5, "azimuth_limit": 4},
  "notes": "margin NOT applied; human decides"
}
```

## 10. 驗收條件

- [ ] `--dry-run` 完全不開序列埠、不裝 placo，印出將使用的 port / URDF 路徑 / 檔名
- [ ] FK 立起來那步（§3）：placo 匯入成功印版本；或明確走手刻 fallback 並說明
- [ ] **5-pose 對照表印出來、每格有預測 vs 實測 vs 差值**，未打勾不進主迴圈
- [ ] 5-pose 沒過時，腳本進 `tape` 模式而不是崩潰，且 summary / CSV 的 `method` 標 `tape`
- [ ] 拔掉手臂時給清楚錯誤訊息，不是 traceback
- [ ] **每按一次鍵就 flush 寫入 CSV** —— 現場斷電不能丟資料
- [ ] 同一天重跑**不覆寫**既有檔案（檔名衝突加 `_2`）
- [ ] 沒有 `reference` 樣本時，summary 明講「方位角為 base 框架、未對到墊子」，`azimuth_frame` 標 `base`
- [ ] `reach_summary_<date>.json` 欄位齊全，且 `notes` 明記 margin 未套用

## 11. 不要做的事

- ❌ 不要自動決定 margin
- ❌ 不要錄影像，不碰相機
- ❌ 不要把結果直接寫進 `docs/experiment_spec.md` —— 人看過摘要再手動填
- ❌ FK 未通過 5-pose 驗證前，不要輸出 `method=fk` 的 radius
- ❌ 不要為了 FK 上 ROS 2（見 §3 第 5 點、D026）
- ❌ 不要保留舊版的「(a) 環帶 vs (b) 依方位角」判定 —— D023 §2026-08-31 已作廢

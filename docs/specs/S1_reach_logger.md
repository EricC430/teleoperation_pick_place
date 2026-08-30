# S1 — Reach logger

**目標檔案：`scripts/reach_logger.py`**
**正本依據：`docs/decisions.md` D023「Three scripts to build before the next lab day」第 1 支**

---

## 1. 這支腳本要解決的問題

D023 要求在下一個 lab day 量出三個數字，填進 `docs/experiment_spec.md` §3：

1. **傳輸線限制哪個方位角範圍**
2. **最差方位角的 r_max**（這個數字裁決 D023 的讀法 (a) 環帶 vs (b) 依方位角）
3. **還能下爪的 r_min**

**沒有這三個數字，S2 的凍結清單不能產生，A7 不能凍結，A8 不能錄。**

## 2. 🔴 已查證的硬限制（2026-08-30，不要重查，也不要繞過）

| 事實 | 出處 |
|---|---|
| `OmxFollower` **只暴露關節角，沒有末端位姿** | `lerobot/src/lerobot/robots/omx_follower/omx_follower.py:173` `get_observation()` 只回 `{motor}.pos` |
| 馬達名稱固定為 6 個 | 同檔 `:53` — `shoulder_pan` / `shoulder_lift` / `elbow_flex` / `wrist_flex` / `wrist_roll` / `gripper` |
| **`placo` 沒有安裝** | `python -c "import placo"` → `ModuleNotFoundError`（2026-08-30 在本機釘選環境實測） |
| ⇒ **FK 這條路今天不通** | LeRobot 的 `RobotKinematics` 需要 URDF ＋ placo |

**→ 半徑用捲尺量、用鍵盤輸入。這不是將就，是 D023 已經寫明的「最便宜的路」。**

💡 **但方位角不必用量角器：`shoulder_pan` 就是方位角。**
腳本應自動記錄 `shoulder_pan.pos`，人只需要在**一個參考點**上量一次實際角度來定零點與正負號。
**這會省掉現場最麻煩的一項量測——請務必實作。**

## 3. 執行模式

**模式 A（預設，`--mode teleop`）：腳本自己跑 leader→follower 迴圈。**
理由：序列埠是獨占的，不能一邊跑 `lerobot-teleoperate` 一邊跑這支腳本。

**模式 B（`--mode follower-only`）：只連 follower，由人手動扳動。**
⚠️ **需要關掉扭力才扳得動，而 LeRobot 是否提供 OMX 的 torque-off API `[未確認]`。**
**實作前先查 `DynamixelMotorsBus` 有沒有 `disable_torque()`；沒有就不要硬寫，直接讓模式 B 報錯退出並說明原因。**

## 4. 輸入

```
--config configs/teleoperate_omx.yaml     # 沿用既有的 port / id / calibration_dir
--out    analysis/reach_log_YYYY-MM-DD.csv
--mode   teleop | follower-only           # 預設 teleop
--fps    30                               # 迴圈頻率；60 也行，這支不是錄製
--dry-run                                 # 不碰硬體，印出會做什麼就結束
```

## 5. 互動與輸出

**鍵盤（用 `pynput`，已在 `hardware` extra 內，不需另裝）：**

| 鍵 | 動作 |
|---|---|
| `o` | 記一個 **outer** 樣本（伸到桌面接觸的最遠點） |
| `i` | 記一個 **inner** 樣本（還能擺出可夾姿態的最近點） |
| `a` | 記一個 **azimuth-limit** 樣本（掃到卡住的角度） |
| `q` | 存檔離開 |

按鍵後**在終端提示輸入**（不要吃掉鍵盤迴圈）：

```
[outer] 捲尺讀數 (cm) > 28.5
[outer] 備註 (可空) > 線在這個方向最卡
```

**CSV 欄位（`analysis/reach_log_<date>.csv`）：**

```
ts_iso, sample_type, radius_cm, shoulder_pan_pos, shoulder_lift_pos,
elbow_flex_pos, wrist_flex_pos, wrist_roll_pos, gripper_pos, note
```

`sample_type` ∈ `outer` / `inner` / `azimuth_limit` / `reference`。
`reference` 是校零那一筆：人量一次實際方位角並輸入，用來把 `shoulder_pan.pos` 換算成度數。

## 6. 結束時要印的摘要

```
樣本數: outer=7 inner=5 azimuth_limit=4
受限方位角(由 shoulder_pan 推得): -35.2° ~ +88.6°   [參考點校正: 有 / 無]
最差方位角 r_max: 26.0 cm  @ pan=-35.2°
r_min (可下爪): 19.5 cm

→ D023 讀法判定: (b) 依方位角  [因為 r_max 隨 pan 變化 > 3 cm]
→ 建議寫進 experiment_spec §3 的值:
   r_outer = 26.0 - <margin> ,  r_inner = 19.5 ,  azimuth = [-35.2, 88.6]
⚠️ margin 由人決定，腳本不要自己減。D023 說邊界處沒有可用的抓取進場角。
```

## 7. 驗收條件

- [ ] `--dry-run` 完全不開序列埠，且印出將使用的 port / 檔名
- [ ] 拔掉手臂時給**清楚的錯誤訊息**，不是 traceback
- [ ] **每按一次鍵就 flush 寫入 CSV**——現場斷電不能丟資料
- [ ] 同一天重跑**不覆寫**既有檔案（檔名衝突就加 `_2`）
- [ ] 摘要在「沒有 reference 樣本」時明確說「方位角未校零，數字只是相對值」，**不要假裝有絕對角度**

## 8. 不要做的事

- ❌ 不要自己算 FK（placo 沒有）
- ❌ 不要自動決定 margin
- ❌ 不要錄影像。這支腳本不碰相機
- ❌ 不要把結果直接寫進 `docs/experiment_spec.md`。**人看過摘要再手動填**

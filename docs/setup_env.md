# Site baseline

The physical setup that every experiment implicitly depends on. **Capture this once, on day one, and
keep it fixed** — if it changes, note the change here with a date, because it explains sudden shifts in
results just as much as a calibration drift does.

## Desk / workspace layout

- Desk dimensions and material: TODO
- Object start region (marked how?): TODO
- Target/place region: TODO
- Reference photo(s): TODO — link or embed

## Lighting

- Light source (ceiling/lamp/window), time-of-day dependence: 見下方 **Lighting** 一節（2026-09-13 填）
- Reference photo(s): TODO

## Camera placement

- 3rd-person camera: mount point, height, angle, distance to workspace — 45度向右、19.5cm從omx底座左邊邊到相機支架中心 (+ photo)
  - 2026-09-13 待補（凍結前）：鏡頭離桌面高度、俯角、「45 度」的量測基準邊、桌面膠帶標記、基準照片、相機畫面基準（`lerobot-find-cameras` 的 `realsense_262822305610.png`）
- Wrist camera: 固定在夾爪上。`[柏宇決定]` 2026-09-13：相對夾爪不會變，**不另外凍結／量測**（「第一視角是固定的所以相對夾爪也不會變」）
- Ablation note: this project runs wrist-only / 3rd-person-only / both as a controlled comparison (see
  meeting notes for rationale) — camera placement must stay identical across all three conditions.

## Lighting

- **窗簾全拉上（手臂面向窗戶）＋ 宿舍大燈開啟 ＋ 檯燈開啟。** `[柏宇決定]` 2026-09-13（「目前有檯燈，把文件改成有檯燈」）。
  - 取代同日稍早的「無檯燈」配置（當時轉述為柏宇與 Eric 決定）。**Eric 是否同意加檯燈：未確認。**
  - 🔴 檯燈現在是場景常數，待補（凍結前）：哪一盞、擺放位置（膠帶標記）、照射方向／角度、亮度檔位（若可調）、照片。檯燈被移動 = 光線改變。
- 未控制的部分：窗簾透進的日光（時段／天氣）。D455 畫面是否拍到窗簾：`[柏宇說]` 應該沒有、未確認 → 下次看 rerun。
- 監測日光變化的做法（C 步腳本當測光表）：`[AI提議]` 🟡 待裁決，見 `docs/decisions.md` D022 §2026-09-13。

## Target objects

- Object set used for P0 (e.g. bottle / can / crumpled paper), quantities: TODO
- Where they're stored between sessions: TODO
- bin: D455 box, 6 cm right to the omx arm 底座邊邊

## Change log

Record every deviation from the baseline above, with a date — this is what makes a sudden accuracy drop
diagnosable instead of mysterious.

| Date | What changed | Why |
|---|---|---|
| 2026-09-13（事後回推，10-05 量測） | D455 在 uvc_60 的 ep 49 → ep 50 之間動過：左右平移約 17 px、旋轉約 0.5°。ep 50–59 和寶特瓶訓練屬於這個位置（A2） | 不明。當時沒有發現，見 `docs/meeting/2026-10-05.md` §2-b |
| 2026-09-13 → 09-18 01:09（事後回推） | D455 移到位置 B：交界線角度 −12.5° → −16.9°，y@x=0 由 325 → 368 px。9/18 紙杯 rollout、鋁罐訓練、9/20–21 的 rollout 都在 B | 不明 |
| 2026-09-21 → 10-05（事後回推） | D455 又移動，位置 C（10:39）。10:39 → 11:24 之間再動一次，變成位置 D | 不明 |
| 2026-09-13 以後 | 手臂底座重新固定過，底座中心對齊桌上的記號（**yaw 方向沒有基準**）。底座旁的白色支架加書，換成了金屬 C 型夾。畫面右側多了一台路由器 `[柏宇說]` 2026-10-05 | 桌子要拿來做其他用途 |
| **2026-10-05 17:02** | **D455 重新對齊到 A1**（參考畫面：uvc_60 ep 10 的 frame 0），已鎖緊。殘差（即時畫面 − ep 10，單位 px）：t11 上的杯子 +5/−3，桌緣貼紙 +5/−6，桌面木紋 +4.6/−5.0，手臂頂端支架 −5/−5；交界線角度 +0.23°。基準照片：`docs/assets/d455_baseline_20261005_A1.png`。工具：`scripts/align_camera.py`（量桌子）＋ 杯子放在 t11（擺位卡第 3 步對齊 C 形臂）做模板比對 | 相機位置與訓練時不同，詳見 10-05 紀錄 §2-b。過程中發現：只靠轉角度無法讓遠近同時對上，偏移量隨距離變化，所以相機的「位置」也有變，平移後才收斂 |

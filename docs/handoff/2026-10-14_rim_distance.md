# 交代：在筆電上量閉環「第一次閉合時，夾持點離杯緣多遠」（2026-10-14）

寫給：在 Windows 筆電上做這件事的人（柏宇，或那台的 Claude）。決定者：柏宇（2026-10-14：量法、在筆電做）。
在 4090 那台做了初步檢查（數字在下面），但閉環 dataset 和 link-marker 照片都在筆電上，所以在筆電上做完。

## 要回答的問題

閉環時手臂的定位精度是多少？有沒有隨著模型迭代進步？

## 為什麼不用 `scripts/aim_error.py`

aim_error 用 uvc_60（9/13）的人類示範擬合「關節 → (θ, r)」的線性對照，再拿杯子中心當目標。10-14 在 4090 檢查後 `[已查證]`：
- 照原樣重跑 10/06，數字和 `2026-10-06.md` §3-2 完全一樣。抓取幀門檻改成 0.4／0.6／0.8，結論都不變。
- **線性對照只在擬合用的那種姿勢下成立**：
  - 套到 normal_A1 人類示範：徑向偏 −1.05 cm、斜率 1.10，大致可以用。
  - 套到 far_wrist_A1 人類示範（轉腕抓法）：斜率 1.58、\|橫向\| 3.65 cm，**對照不成立**。
- **對照跟著日期變**：9/13 擬合出 `θ = 2.11·pan + 5.6°`，10/07–08 擬合出 `θ = 1.92·pan + 0.6°`，差約 5°。
  10/06 那場換成 A1 對照重算，dθ 從 −1.8° 變成 −3.9°、斜率從 0.70 變成 0.56。
- **目標應該是杯壁，不是杯心**：9/29 已查證人類是一爪在杯內、一爪在杯外夾住杯壁（`scripts/eval_joint_calibration.py` docstring）。
  用杯心當目標，光是夾哪一側的杯壁就會造成約 ±3.5 cm 的「誤差」。

所以 `2026-10-06.md`／`2026-10-08.md` 裡 9/18 → 10/06 → 10/08 的變化（斜率 0.76 → 0.70 → 1.00、偏移 −0.5 → −3.2 cm），
**不能直接當成模型的進步或退步** `[推論]`：10/08 的模型換成用 A1 示範訓練，那批人類的姿勢和 9/13 差很多（lift 15 對 29.5、elbow +10 對 −20）。

## 量法 `[柏宇決定 2026-10-14]`

**到杯緣的距離** = `| |p − c| − R |`
- `p`：第一次「持續閉合」那一幀的夾持點，用 FK 算。抓取幀用 `scripts/fit_paper_to_pan.py` 的 `grasp_frame(st, "first")`：
  閉合後要保持 15 幀，和 `eval_joint_calibration.py` 同一個定義。
- `c`：杯心，也就是擺放點的座標紙座標。c*k* 對應 `campA_136sym…_eval-close.csv` 的 `eval-close_00k`，t*k* 對應 `…_train.csv`。
- `R`：`eval_joint_calibration.CUP_R_AT_PINCH` = 3.4 cm。

這個量法不用知道夾的是哪一側杯壁，只問「有沒有落在杯緣這個圓上」。
人類每次都有夾到，所以**人類示範的值就是這把尺的底線**。

**關節常數**：用 `calibration/2026-10-10_link_tilt_fit.json` 的 **「S7 offsets only」** 當主結果。
- 這是 S7 §7 的預設，因為「零點＋比例」的留一誤差沒有低 0.3° 以上。
- 舊的手調常數（`sim/joint_mapping.py` 現在的值）一起列出來當對照。
- S7 的常數**還沒貼回** `joint_mapping.py`，也還沒通過 S7 §8 的 IoU 驗收。結果要標「S7 未驗收」。

核心程式（4090 上跑過的版本，從 `scripts/` 匯入既有函式，不重寫幾何）：

```python
import csv, json, sys, importlib.util
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts")
spec = importlib.util.spec_from_file_location("fpp", "scripts/fit_paper_to_pan.py")
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
EJC = F.EJC
F.DATA = Path(r"D:\hf\lerobot\ericc430")   # load_states() 從 DATA / <dataset 名稱> / data 讀；照筆電實際位置改

fit = json.load(open("calibration/2026-10-10_link_tilt_fit.json"))
s7 = fit["S7 offsets only"]
EJC.use(s7["scale"], s7["offset"])        # 用完要切回原本的常數，見 EJC.use

def rim_rows(dataset, meta_csv, place):    # place: {"c1": (x_cm, y_cm), "t5": ..., ...}
    states = F.load_states(dataset)
    for m in csv.DictReader(open(meta_csv, encoding="utf-8-sig")):
        pid = (m.get("placement_id") or "").strip(); e = int(m["episode_index"])
        if pid not in place or e not in states:
            continue
        gi = F.grasp_frame(states[e], "first")
        if gi is None:                     # 沒有持續閉合：要算進回報的「略過」數
            continue
        q = EJC.q5(states[e][gi]); p = EJC.point_cm(q, EJC.X_PINCH)[:2]
        c = np.array(place[pid])
        yield e, pid, abs(np.linalg.norm(p - c) - EJC.CUP_R_AT_PINCH), states[e][gi][4], m.get("outcome", "")
```

## 步驟

1. `git pull`，確認上面三支腳本在筆電上都能匯入。
2. **先跑人類底線，對不上就先停下來查。** 下面是 4090 上用 S7 offsets only 跑出來的中位數：

   | 資料 | n | 到杯緣距離（中位數） |
   |---|---|---|
   | normal_A1（10/07，一般姿勢） | 16 | 1.05 cm |
   | far_wrist_A1（10/08，轉腕；排除 notes 含 recovery 的集） | 46 | 2.10 cm |
   | uvc_60（9/13） | 60 | 2.11 cm |
   | 10/06 閉環：成功／失敗 | 17／17 | 2.31／3.34 cm |

   舊手調常數的對照：normal_A1 0.81、far_wrist 4.62、uvc_60 2.33、10/06 成功 2.11／失敗 4.85 cm。
3. 對每一場閉環跑一次，**夾取時有沒有轉腕要分開列**。轉腕的判定 `[未確認，請柏宇定]`：
   在抓取幀的 `wrist_roll` 狀態 > 15。這是從 `2026-10-13_open_loop.md` §2 的「預測 wrist_roll 最大值 > 15」借來的，原本是用在開環預測上。

   | 日期 | 模型 | dataset（筆電 `D:\hf\lerobot\ericc430\`） | episode_meta |
   |---|---|---|---|
   | 10/06 | 紙杯 100k（uvc_60） | `rollout_omx_b1_uvc60_100k_nas30_A1_paper_cup_20261006_094908` | `…_20261006_close_loop.csv` |
   | 10/08 | ntA1（N=100） | `…_60_16_12rcvry_100k_paper_cup_20261008_171620` | `…_nas100_paper_cup_20261008_171620.csv` |
   | 10/08 | ntA1（N=30，C 場） | `…_20261008_192921` | `…_nas30_paper_cup_20261008_192921.csv` |
   | 10/13 | ntfA1aug | `…_20261013_184921` | `…_46far_aug_100k_nas30_paper_cup_20261013_184921.csv` |
   | 10/13 | ntcfA1aug | `…_20261013_201040` | `…_rcvrycut_46far_aug_100k_nas30_paper_cup_20261013_201040.csv` |
   | 10/14 | fldrop30 | `…_20261014_151208` | `…_fldrop30_100k_nas30_paper_cup_20261014_151208.csv` |

   dataset 的實際資料夾名稱以筆電上的為準，上表的 `…` 是省略。10/07 的兩場（temporal ensemble、遮手腕）要不要加，由柏宇決定。

## 哪些場次可以用 S7（場地時間軸）

| 時間 | 事件 | 依據 |
|---|---|---|
| 10/10–10/12 | S7／S8 量測 | `calibration/2026-10-10_link_tilt*.csv`、`2026-10-12_pan_circle.csv` |
| 10/13（時間未記） | 場地被意外動到：底座從前傾 3.3° 變成水平，front-left 相機也移了一點 | `calibration/2026-10-13_level_check.csv` |
| 10/13 18:49、20:10 | ntfA1aug／ntcfA1aug 閉環 | `[柏宇說 2026-10-14]` 在場地動到之後 |
| 10/14 15:12 | fldrop30 閉環 | dataset 時間戳 |
| 10/14 20:01–20:16 | front-left 相機又移動一次（在 fldrop30 之後） | `calibration/2026-10-14_link_markers_excluded.csv` |

- **10/06、10/08**：在 S7 之前，中間沒有記錄到場地變動，可以用 S7 `[推論]`。
- **10/13、10/14**：S7 量的是手臂本身的關節零點（底座傾斜另外處理，`--tilt-from-b0`），所以常數**可能**還能用 `[推論]`。
  但底座相對座標紙的**旋轉和平移**沒有重新量過（只有羅盤 271° 沒變，精度只有幾度）。這兩場的結果要標註這一點；最快的確認方法是重跑 S8。
- **請補記**：10/13 場地被動到的時間，寫進 `docs/worklog/2026-10-13_closed_loop.md`。

## 判讀限制

- **轉腕姿勢的底線是 2.1 cm**：在 far_wrist 人類示範上，S7 下還有 52% 落在杯緣外 2 cm 以上，所以轉腕抓取分辨不到 2 cm 以下。
- 杯子在第一次閉合前如果已經被推走（`pushed_away`），「杯心 = 擺放點」就不成立，那一集的距離會偏大。可以列出來，但不要拿去說「瞄不準」。
- 沒有持續閉合的集數會被略過。這種集數通常是最差的，所以每場都要回報略過了幾集、是哪幾集。
- 每場 36 集。兩場的差距要和人類底線、以及場內的離散程度一起看。

## 指尖點選檢查（選做，判斷 FK 在夾取高度準不準）

10-14 在 4090 上把 FK 指尖投影到 front-left 靜止畫面，和 `calibration/2026-10-08_far_wrist_A1_fingertip_clicks.csv`、`2026-10-10_link_tilt_fingertip_clicks.csv` 的點選比對。
相機用 `2026-10-10_camera_extrinsics_front-left.json` 加上 `2026-10-07_camera_intrinsics_front-left.json`，投影用 `scripts/fit_arm_overlay.py` 的 `Camera`。
P02、P03 的疊圖確認投影流程是對的。

- 夾取高度（指尖離桌 < 10 cm）：S7 offsets only 誤差 1.3～3.0 cm（n=3），舊常數 1.9～4.7 cm（n=5）。
- **點太少**：看得到指尖的點大多在高處。far_wrist 的 45 筆裡有 35 筆被擋住；分組用的高度也是 FK 自己算的，所以每組常數分到的點不一樣。
- 是近似版：投影的是夾爪中線的尖端，點選的是靠近相機那根手指的前緣，兩者本來就差約 1 cm。
- 10/14 link-marker 的 scan 讓指尖保持在桌面上方 11 cm 以上，**夾取高度不在範圍內**。如果要驗證夾取高度，要另外補點。

## 回報

1. 人類底線三列，和上面的表比對。
2. 每場閉環：一般姿勢和轉腕分開列 n、到杯緣距離的中位數與 IQR、成功／失敗各自的中位數、略過的集數。
3. 用的常數組，以及每場對應的場地時期（S7 前／10/13 動到後／10/14 相機移動後）。
4. 結果寫進 `docs/worklog/<日期>_rim_distance.md`，數字以那份為準。這份交代不寫結果。

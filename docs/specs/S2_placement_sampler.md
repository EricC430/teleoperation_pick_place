# S2 — 物體配置點抽樣器

**目標檔案：`scripts/sample_placements.py`**
**正本依據：`docs/decisions.md` D023 第 2 支腳本、D024（含 2026-08-30「不可重疊」與 2026-08-31 的三份清單／共用 30 點裁決）**

---

## 1. 這支腳本要解決的問題

D024 廢掉 3×3 網格，改為**在環形扇區上面積均勻抽樣一次、凍結、每點給 ID**。
需要**三份**清單（`[Eric決定]` 2026-08-31）：

| 清單 | N（預設） | `placement_id` 前綴 | 用途 |
|---|---|---|---|
| `train` | 50 | `train_` | 錄製訓練集 |
| `eval-open` | 10 | `eval-open_` | 錄製 open-loop 評估（對錄好的軌跡算預測誤差） |
| `eval-close` | 30 | `eval-close_` | 閉迴路評估。**3 個物體共用同一份 30 點**（`[Eric決定]` 甲） |

**三份清單裡的每一個點，與其他所有點（含跨清單）距離都 ≥ `d_min`。** `[Eric決定]` 2026-08-31：
均勻分布本來就少重疊，所以直接要求全體互斥，不是只算 train↔eval 跨組。

## 2. 🔴 三個容易寫錯的地方，每一個都會安靜地毀掉實驗

### 2-1 面積均勻，不是半徑均勻

```python
# ❌ 錯的（點會偏向內圈）
r = rng.uniform(r_inner, r_outer)

# ✅ 對的
r = math.sqrt(rng.uniform(r_inner**2, r_outer**2))
theta = rng.uniform(theta_min, theta_max)
```

### 2-2 「Seeded」＝抽一次然後固定，不是每次跑重抽

**重抽會摧毀跨模型可比性，而可比性是做這件事的全部理由。**
→ 腳本**必須**吃 `--seed`，且**輸出檔存在時預設拒絕覆寫**（要 `--force` 才蓋）。
→ 檔名帶日期（見 §5），凍結性資料只寫一次（`docs/conventions.md`）。

### 2-3 「不可重疊」在連續空間上不構成條件

✅ `[Eric決定]` 2026-08-30：清單之間不可重疊。
🔴 **但任兩個連續抽樣點重合的機率是 0，所以這句話必須實作成最小間距 `d_min`。**
✅ `[Eric決定]` 2026-08-31：範圍擴大成**全體點彼此** ≥ `d_min`（不只跨組）。

**⚠️ `d_min` 有 packing 上限，暫定工作區可能付不起。** 完整試算見 `docs/decisions.md` D024
§2026-08-30（80 點、180° 扇區、r=20–30 時，平均最近鄰距離只有 ≈1.6 cm；三份清單合計
`N_total = 90` 時更緊）。

**→ 腳本要做的是「把不可行講清楚」，不是硬湊。啟動時先算兩個上限：**

```
f       = (theta_max - theta_min) / 360           # 佔整圈的比例
A       = f * pi * (r_outer**2 - r_inner**2)       # 可用面積 cm^2
N_hex   = A / (0.866 * d_min**2)                    # 完美六方最密堆積上限（排到毫無浪費）
N_rsa   = 0.55 * N_hex                              # 務實上限：隨機丟點會卡在完美排列的 ~55%
N_total = n_train + n_eval_open + n_eval_close      # 預設 90
```

**判定（啟動時算一次，不可行就直接失敗，不要進迴圈跑很久才失敗）：**

| 條件 | 動作 |
|---|---|
| `N_total > N_hex` | 退出碼 2：「幾何上不可能」＋印下表 |
| `N_rsa < N_total ≤ N_hex` | 退出碼 2：「理論可行、隨機抽不到 —— 減 N 或減 `d_min` 或放大工作區」＋印下表 |
| `0.7 * N_rsa < N_total ≤ N_rsa` | 開抽，但印警告「可能很慢／可能中途失敗」 |
| `N_total ≤ 0.7 * N_rsa` | 開抽 |

**❌ 不要用 `N_hex` 當唯一關卡。** dart-throwing 遠達不到六方最密堆積；只看 `N_hex` 會讓
「可行」變成謊話（檢查通過、抽樣仍卡死）。

## 3. 輸入

```
--from-summary analysis/reach_summary_YYYY-MM-DD.json  # S1 的產出（見下）；可選
--r-inner 20 --r-outer 30              # 與 --from-summary 互斥
--theta-min -35 --theta-max 88          # 度，來自 S1；與 --from-summary 互斥
--margin 3.0                            # cm；🔴 用 --from-summary 時強制明給（見下）
--n-train 50 --n-eval-open 10 --n-eval-close 30
--d-min 2.0                             # cm；🔴 沒有預設值，必須明給（見 §7）
--seed 20260831
--date 20260831                        # 檔名用；預設今天
--out-dir configs/placements/
--label   campaign_A_pilot_2cam
--dry-run                               # 只做可行性檢查與統計，不寫檔
--force                                 # 允許覆寫既有輸出檔（預設拒絕）
```

**`--margin` 的作用範圍（`[Eric決定]` 2026-08-31）：**
- `r_outer = r_outer_raw − margin`
- 兩側方位角各往內縮 `degrees(margin / r_outer)`（沿整條扇區邊界至少留 `margin` cm 線性餘裕，
  角度換算用較保守的 `r_outer`）：`theta_min += degrees(margin / r_outer)`、
  `theta_max −= degrees(margin / r_outer)`
- 🔴 **`r_inner` 用原值，不加 margin。** 失敗機制聚集在外圈與受限方位角，不在內圈；為內圈買餘裕
  會付掉一大塊本就很緊的面積，不划算。真要，之後加明確的 `--margin-inner`，預設關。

**`--from-summary`（可選，來自 S1 `reach_summary_<date>.json`）：**
- 讀 `r_inner_cm` → `r_inner`、`r_outer_topdown_cm` → **`r_outer_raw`**、
  `azimuth_min_deg` / `azimuth_max_deg` → `theta_min` / `theta_max`。
- **`--margin` 仍強制手動明給**，套用如上（D023 §2026-08-31：腳本不自動減 margin）。
- **同時給 `--from-summary` 和明給 `--r-outer` / `--r-inner` / `--theta-*` → 直接報錯退出**，不要默默擇一。
- summary 裡 `fk_validation` 為 `null`（未做現場驗證）或 `passed == false` 時，印一行警告
  （radius 精度較低）但**不擋**。
- summary 裡 `azimuth_frame == "base"`（S1 沒記 reference）→ **S2 不擋**（那是 S3 出正式墊子的關卡）；
  但要把 `azimuth_frame` 原值寫進 `_meta.json`。
- `_meta.json` 要記錄 `from_summary` 的檔名、其 `git_commit`、`fk_validation`、`azimuth_frame`。

## 4. 演算法

1. 可行性檢查（見 §2-3）。不可行 → 退出碼 2 ＋ 建議的 `d_min` / N 上界。
2. **依 N 由大到小抽**：`train`(50) → `eval-close`(30) → `eval-open`(10)。
   - 每個新點的接受條件：**與先前所有已接受點（跨全部清單）距離 ≥ `d_min`**。
   - 🔴 **順序不能反。** 先抽多的那組才不會把自己逼到角落。
3. 每組給上限嘗試次數（`200 × N`）。達到上限 → **失敗退出**，印出「哪一份清單、抽到第幾點卡住」。
   - ❌ **不要自動放寬 `d_min` 重試。** 那會讓輸出的保證變成一句謊話。
4. 每組內 `placement_id` 依接受順序編號：`train_001`…`train_050`、`eval-open_001`…、`eval-close_001`…。

## 5. 輸出

`configs/placements/<label>_<YYYYMMDD>_train.csv`、`<label>_<YYYYMMDD>_eval-open.csv`、
`<label>_<YYYYMMDD>_eval-close.csv`：

```
placement_id, r_cm, theta_deg, x_cm, y_cm
train_001, 27.31, -12.4, 26.68, -5.86
```

- `placement_id` **就是** `eval/_template.csv` 的 `placement_id` 欄與
  `configs/episode_meta_schema.yaml` 的 `placement_id` 欄要填的值。
- `x_cm` / `y_cm` = `r·cos(theta)` / `r·sin(theta)`（墊子框架，見 S3 §2-4）。S3 直接用這兩欄畫十字，
  不重算，避免框架不一致。
- 同時輸出一份 `<label>_<YYYYMMDD>_meta.json`（三份清單共用）：所有輸入參數、seed、程式碼 git commit、
  產生時間、`from_summary` 資訊（§3）、**每份清單**實際達成的最小間距與最近鄰距離分佈、以及**跨三份**
  的全體最小間距。**沒有這份，三個月後沒人知道這批點怎麼來的。**

## 6. 驗收條件

- [ ] 用固定 seed 跑兩次，三份輸出**逐位元組相同**
- [ ] 對每份輸出做統計檢定：`r²` 應近似均勻（KS test 對 `U(r_in², r_out²)`），**把 p 值印出來**
- [ ] 把三份 CSV 的點合起來重算一次最近鄰：**全體最小距離確實 ≥ `d_min`**（不要相信抽樣迴圈）
- [ ] 輸出檔已存在時預設拒絕覆寫（`--force` 才蓋）
- [ ] `--dry-run` 印出可行性表（`A`、`N_hex`、`N_rsa`、`N_total`、預期平均最近鄰距離、判定結果），不寫檔

## 7. 待裁決（腳本要能接受任一種，不要替他選）

✅ `[Eric決定]` 2026-08-31：**選甲，工作值 `d_min = 2.0 cm`**（正本：`docs/decisions.md` D024
§2026-08-31）。甲＝保留點數，`d_min` 只保證「可分辨的擺放」。2.0 這個數字與 `r_inner`／`r_outer`
同屬暫定，待 S1 實測扇區。S2 自己的 run 顯示：暫定 123° 扇區放不下 90 點 @ 2.0cm
（`N_rsa ≈ 85 < 90`）——S1 給出真實扇區後，重跑 `--dry-run` 確認 2.0 成立或微調（如 1.8）。
**不論如何，`--d-min` 保持必填、無預設。**

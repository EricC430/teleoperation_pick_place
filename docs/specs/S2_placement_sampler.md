# S2 — 物體配置點抽樣器

**目標檔案：`scripts/sample_placements.py`**
**正本依據：`docs/decisions.md` D023 第 2 支腳本、D024（含 2026-08-30「不可重疊」與 2026-08-31 的三份清單／共用 30 點裁決）**

---

## 1. 這支腳本要解決的問題

D024 廢掉 3×3 網格，改為**在環形扇區上分層等面積抽樣一次、凍結、每點給 ID**（演算法見 §4）。
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

**❌ 不要用 `N_hex` 當唯一關卡。** 隨機放點遠達不到六方最密堆積；只看 `N_hex` 會讓
「可行」變成謊話（檢查通過、抽樣仍卡死）。

**⚠️ 分層抽樣（§4）比自由放點更容易在接近飽和時卡住**——每個點被綁在自己的格子裡，某一格整格
都落在已放點的 `d_min` 內就會 200 次重抽失敗、直接中止。可行性判定給 `OK_WARN` 時仍可能中止；
中止是大聲的（退出碼 2 ＋ 指出哪一格），不是安靜壞掉。

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
--plot                                  # 另存 <label>_<date>_scatter.png（診斷用，不是 S3 墊子）
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

## 4. 演算法 — 分層等面積抽樣（stratified，`[Eric決定]` 2026-08-31）

**不是純隨機 dart-throwing。** 純隨機在 n=30 這種規模會靠運氣結塊、留空洞，每個 seed 不同——
而這 30 點的職責就是「讓失敗聚集看得出來」的覆蓋樣本，不能賭那個變異（D024-OLD 也提過
low-discrepancy）。改成：把扇區切成等面積格子，每格丟一個格內均勻隨機點。仍 seeded、仍格內隨機、
仍可重現，但每一區保證有點。

1. 可行性檢查（見 §2-3）。不可行 → 退出碼 2 ＋ 建議的 `d_min` / N 上界。
2. **依 N 由大到小處理**：`train`(50) → `eval-close`(30) → `eval-open`(10)。🔴 順序不能反。
3. **對每一份清單**（大小 n）：
   - 切格：`n_rings × n_wedges` 等面積格，`n_rings ≈ round(sqrt(n · 徑向跨度 / 弧向跨度))`、
     `n_wedges = ceil(n / n_rings)`，`n_rings · n_wedges ≥ n`。
     等面積 = 徑向按 `r²` 均分、方位角按角度均分。
   - `rng.shuffle(格)`，取前 `n` 格（多出來的格不用；哪些格不用是 seeded 隨機）。
   - 每格內：`r = sqrt(rng.uniform(a0, a1))`、`theta = rng.uniform(t0, t1)`。
   - **全域 `d_min`**：新點與**先前所有已接受點（跨全部清單）**距離 ≥ `d_min`。違反就在**同一格內**
     重抽，上限 200 次。
4. 某格 200 次都放不進 → **失敗退出**，印出「哪一份清單、第幾格卡住」。
   ❌ **不要自動放寬 `d_min`。** 那會讓輸出的保證變成一句謊話。
5. 每份清單內 `placement_id` 依處理順序編號：`train_001`…`train_050`、`eval-open_001`…、`eval-close_001`…。
6. 單一 `random.Random(seed)`、固定順序消耗 → 同 seed 逐位元組相同；`train` 在 `eval-*` 之前抽完，
   所以改 `eval` 數量不會動到任何一個 `train` 點。

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
  產生時間、`from_summary` 資訊（§3）、每份清單的 `r²` KS p 值、**每份清單**實際達成的最小間距與
  最近鄰距離分佈、以及**跨三份**的全體最小間距。**沒有這份，三個月後沒人知道這批點怎麼來的。**
- `--plot` 時另存 `<label>_<YYYYMMDD>_scatter.png`：三份清單疊在扇區上的笛卡兒散佈圖，看覆蓋與聚集用。
  **這是診斷圖，不是 S3 的可列印墊子**（S3 才有刻度、拼貼、對位記號）。

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

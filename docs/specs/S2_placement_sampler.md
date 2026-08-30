# S2 — 物體配置點抽樣器

**目標檔案：`scripts/sample_placements.py`**
**正本依據：`docs/decisions.md` D023 第 2 支腳本、D024（含 2026-08-30 的「不可重疊」裁決）**

---

## 1. 這支腳本要解決的問題

D024 廢掉 3×3 網格，改為**在環形扇區上抽樣一次、凍結、每點給 ID**。
需要**兩份**清單：**訓練配置清單** 與 **評估配置清單**。

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

### 2-3 「不可重疊」在連續空間上不構成條件

✅ `[Eric決定]` 2026-08-30：兩份清單不可重疊。
🔴 **但任兩個連續抽樣點重合的機率是 0，所以這句話必須實作成最小間距 `d_min`。**

**⚠️ `d_min` 有 packing 上限，暫定工作區可能付不起。** 完整試算見 `docs/decisions.md` D024
§2026-08-30（80 點、180° 扇區、r=20–30 時，平均最近鄰距離只有 ≈1.6 cm）。

**→ 腳本要做的是「把不可行講清楚」，不是硬湊：**

```
N_max(d) ≈ A / (0.866 * d²)         # 六方最密堆積上限
A        = f * π * (r_outer² - r_inner²)
```

**啟動時先算一次可行性，不可行就直接失敗並印出這張表**，不要進入抽樣迴圈然後跑很久才失敗。

## 3. 輸入

```
--from-summary analysis/reach_summary_YYYY-MM-DD.json  # S1 的產出（見下）；可選
--r-inner 20 --r-outer 30              # 與 --from-summary 互斥
--theta-min -35 --theta-max 88          # 度，來自 S1；與 --from-summary 互斥
--margin 3.0                            # cm；🔴 用 --from-summary 時強制明給，套用 r_outer = r_outer_raw − margin
--n-train 50 --n-eval 30
--d-min 2.0                             # cm；🔴 沒有預設值，必須明給
--seed 20260831
--out-dir configs/placements/
--label   campaign_A_pilot_2cam
--dry-run                               # 只做可行性檢查與統計，不寫檔
```

**`--from-summary`（可選，來自 S1 `reach_summary_<date>.json`）：**
- 讀 `r_inner_cm` → `r_inner`、`r_outer_topdown_cm` → **`r_outer_raw`**、`azimuth_min_deg` / `azimuth_max_deg` → `theta_min` / `theta_max`。
- **`--margin` 仍強制手動明給**，套用 `r_outer = r_outer_raw − margin`（D023 §2026-08-31：腳本不自動減 margin）。
- **同時給 `--from-summary` 和明給 `--r-outer` / `--r-inner` / `--theta-*` → 直接報錯退出**，不要默默擇一。
- summary 裡 `fk_validation.passed == false` 時，印一行警告（radius 來自捲尺 fallback，精度較低）但不擋。
- `_meta.json` 要記錄 `from_summary` 的檔名與其 `git_commit`。

## 4. 演算法

1. 可行性檢查（見 §2-3）。不可行 → 退出碼 2 ＋ 建議的 `d_min` 上界。
2. **先抽訓練清單**：dart-throwing，接受條件為「與已接受的訓練點距離 ≥ `d_min`」。
3. **再抽評估清單**：接受條件為「與**所有訓練點**距離 ≥ `d_min`」**且**「與已接受的評估點距離 ≥ `d_min`」。
   - 🔴 **順序不能反**。訓練點多、評估點少；先抽多的那組才不會把自己逼到角落。
4. 每組給上限嘗試次數（例如 `200 × N`）。達到上限 → **失敗退出**，印出「抽到第幾點卡住」。
   - ❌ **不要自動放寬 `d_min` 重試。** 那會讓輸出的保證變成一句謊話。

## 5. 輸出

`configs/placements/<label>_train.csv` 與 `<label>_eval.csv`：

```
placement_id, r_cm, theta_deg, x_cm, y_cm
train_001, 27.31, -12.4, 26.68, -5.86
```

- `placement_id` **就是** `eval/_template.csv` 的 `placement_id` 欄與
  `configs/episode_meta_schema.yaml` 的 `placement_id` 欄要填的值。
- 同時輸出 `<label>_meta.json`：所有輸入參數、seed、程式碼 git commit、產生時間、
  實際達成的最小間距與最近鄰距離分佈。**沒有這份，三個月後沒人知道這批點怎麼來的。**

## 6. 驗收條件

- [ ] 用固定 seed 跑兩次，輸出**逐位元組相同**
- [ ] 對輸出做統計檢定：`r²` 應近似均勻（可用 KS test 對 U(r_in², r_out²)），**把 p 值印出來**
- [ ] 驗證 train×eval 的最小距離**確實 ≥ `d_min`**（重新算一遍，不要相信抽樣迴圈）
- [ ] 輸出檔已存在時預設拒絕覆寫
- [ ] `--dry-run` 印出可行性表與預期最近鄰距離，不寫檔

## 7. 待裁決（腳本要能接受任一種，不要替他選）

🔴 **`d_min` 取多少沒有裁決。** D024 §2026-08-30 列了三個選項（甲 d_min≈2cm 保留 80 點／
乙 減少配置點數換大間距／丙 等量完再看）。**腳本把 `d_min` 做成必填參數就好，不要給預設值。**

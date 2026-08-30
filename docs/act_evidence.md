# ACT 實證結果彙整
## 哪些結論可以直接引用、哪些必須自己驗證

**建立：2026-08-15**
**用途：在動手做實驗前，先確認「這個問題別人已經答過了沒有」。**

> ### 分級原則
>
> | 等級 | 條件 | 用法 |
> |---|---|---|
> | **🟢 A 級** | **SO-100/101 真機 ＋ ACT** | **可直接引用，不必自己驗證** |
> | **🟡 B 級** | 真機但非 ACT，或 ACT 但非同級硬體 | 方向可參考，**數字不可搬** |
> | **🔴 C 級** | 模擬，或不同 policy ＋ 不同硬體 | 只當假設來源，**必須自己驗證** |
>
> ⚠️ **本檔的建立起因是一次方法錯誤：** 先前用一篇 Diffusion Policy＋UMI 手持夾爪的論文
> 推論 ACT 的行為，被指出後才發現論文自己把「只測了 Diffusion Policy」列為 limitation。
> **以後引用前先確認：什麼 policy、什麼硬體、真機還是模擬。**

---

## 🟢 A 級：可直接引用（SO-101 真機 ＋ ACT）

### A-1. ACT 在 SO-101 單物體 pick-and-place 的實際成績

| 來源 | 資料量與分布 | 結果 |
|---|---|---|
| **Sherry Chen Try 1** | 50 筆＝**5 個固定位置 × 10** | **≈0%**（完全失敗，「啄木鳥」——反覆啄桌面卻不夾） |
| **Sherry Chen Try 2** | 50 筆＝**6 個 bin 分層 × 10**（留 1 bin 做 OOD） | **60% ID／10% OOD**<br>progress score 0.68／0.28 |
| **Sherry Chen Try 3** | ~150 筆＝6 bin × 25 ＋ **±45° 方向擾動** | **90% ID／75% OOD**<br>progress score 0.92／0.80 |
| **SO-101 Benchmark：Pen Transfer**（單物體搬運，最接近我們） | 100 筆，物體姿態隨機化 | **75%** |
| SO-101 Benchmark：Object Packing（多物體裝箱） | 100 筆 | 10% |
| SO-101 Benchmark：Pen Placement（精確插放） | 100 筆 | 50% |
| **SO-101 Benchmark：Color Sorting（需語意選擇）** | 100 筆 | **0%** |
| 其他實測（Medium／部落格） | 50 筆／100 筆 | 70%／85%+ |
| LeRobot 官方 | 50 筆 | 「high success rates」 |

> ### 🔴 最關鍵的一列對照
> **同樣 50 筆，Try 1 是 0%、Try 2 是 60%。**
> **差別不在筆數，在空間覆蓋方式**——5 個固定位置讓模型背軌跡，6 個 bin 分層抽樣才學到「看物體在哪」。
>
> **→ 「幾筆適當」這個問題本身問錯了。該問的是「怎麼分布」。**

**可直接引用的結論：**
1. ✅ **60–70% 的目標是可達的**（Try 2 已達 60%、Pen Transfer 75%、Try 3 達 90%）
   → `experiment_spec` §8 的決策門檻不需要下修
2. ✅ **50 筆是門檻，不是保證**——分布方式決定成敗
3. ✅ **150 筆＋分層＋方向擾動 → 90% ID／75% OOD**，這是目前找到的最佳配方

### A-2. 🔴 ACT 的自我恢復能力已被量化

| 模型 | Recovery Rate |
|---|---|
| π0.5 | 30.77% |
| Wall-X | 20.51% |
| **ACT** | **6.45%** |
| SmolVLA | 3.23% |

> 原文：**「ACT and SmolVLA exhibit limited recovery success despite comparable or even higher numbers of recovery opportunities, indicating difficulty in exploiting partially successful states once failures occur.」**

**可直接引用的結論：**
- ✅ **6.45% 就是我們 recovery 實驗的對照基線**——**不必自己先做一輪 baseline 來確認「ACT 很不會自我修正」**
- ✅ 我們 `mechanism` 的 `self_recovered` 標籤，**預期出現率在 6% 附近**。若顯著高於此，就是結果

### A-3. 🔴 方向擾動能帶出恢復能力（與 RaC 不同的機制）

| | Try 2 | Try 3 |
|---|---|---|
| 資料 | 50 筆，無方向變異 | 150 筆，**±45° 方向擾動** |
| 恢復能力 | **「一旦第一次抓取失敗，會重試很多次但完全無法恢復」** | **「能從失敗的抓取中恢復」**（有影片） |
| 殘存限制 | — | **無法從超過 45° 的角度恢復** |

作者加方向擾動的動機原文：
> 「blocks don't always sit perfectly aligned in the real world. **I was hoping that this will help with recovery from failed grasps.**」

**可直接引用的結論：**
- ✅ **恢復能力可以從「起始姿態分布撐開」得到，不一定要錄 recovery demo**
- ✅ 機制不同：**不是教它怎麼救回來，而是讓失敗後的狀態仍落在訓練分布內**
- ✅ **這比錄 recovery demo 便宜，而且可與 C1 同時做**
- ⚠️ 但有上限（>45° 仍失敗）→ **不能取代 recovery data，只能降低對它的依賴**

### A-4. ACT 的失敗模式分布（佔失敗試驗百分比）

| 模式 | ACT |
|---|---|
| **State mismatch**（場景理解錯誤） | **98.11%** |
| **Grasp instability**（抓取不穩） | **94.34%** |
| **Repetition loop**（重複迴圈，反覆做同段動作卻無進展） | **92.45%** |
| Precision misalignment | 15.09% |

> 🔴 **我們的 `mechanism` 欄缺少 `repetition_loop`。**
> 現有的 `stalled`（原地停滯）不等於 repetition loop，而後者在 ACT 失敗裡佔 92.45%。
> **→ 必加標籤，見 D015 的修訂建議。**

### A-5. ACT 在需要語意選擇的任務上是 0%

Color Sorting（依顏色分類）ACT ＝ **0%**。ACT 不吃語言（LeRobot 官方 rollout 範例註明 `--task` 對 ACT 可略）。

> 🔴 **對任務定義的硬約束：**
> `experiment_spec` §1-1 目前寫「夾取工作區內的**單一**空容器」——**這條約束現在有硬證據支撐，不可鬆綁。**
> **若改成「多個物體中選一個」，ACT 會像 Color Sorting 一樣掛在 0%。**

### A-6. 實務坑（全部來自 SO-101 真機踩過的人）

| 坑 | 後果 | 對策 |
|---|---|---|
| **相機在訓練與測試間移動過** | 完全失敗 | 貼膠帶與標記固定；**這是 Try 1 失敗主因之一** |
| **校正檔在訓練與測試間不一致** | 「模型學到正確關節角，卻被映射到錯誤的伺服指令」 | 校正檔進 repo；**重校時務必把關節帶到中位**，否則 homing offset 錯亂 |
| **錄製時偷看 follower 手臂** | 錄進「模型拿不到的資訊」 | **只看相機畫面錄製** |
| **teleop 時夾太緊** | **夾爪馬達燒毀**（30Hz sync_read 時掉封包，阻塞整條 motorbus） | 設 `max_relative_target` 限速；練習輕夾；**備妥備品馬達** |
| **兩支同型號相機序號相同** | USB 位址隨機互換，每 ~5 集崩潰一次 | udev 規則綁**實體 USB 路徑** |
| 相機角度不利抓取（夾爪尖端在該視角重疊） | 該視角對抓取無用 | **front + top 優於 front + side** |
| 從白天錄到晚上 | 色調漂移 | **固定曝光、白平衡等相機參數** |
| 示範品質差（她自己常抓在方塊頂部） | 模型學會抓頂部 → 抓不牢 | **"Garbage in, garbage out"**——示範要抓中間 |
| 訓練時無 eval set | 無法偵測過擬合、無法選 checkpoint | **自行切出 eval 集**（LeRobot 只對模擬做 eval） |

### A-7. 評估方法：Progress Score 優於二元成功率

Sherry Chen 採用（靈感來自 π0）：

```yaml
task_progress_score:
  0_reach_block:      0.2
  1_grasp_block:      0.4
  2_reach_container:  0.7
  3_release_block:    0.8
  4_block_in_container: 1.0
```

> ✅ **建議採用。** 我們目前是二元 `outcome`，看不出「進步到哪」。
> Try 2→Try 3 的 progress score 從 0.68→0.92，比「60%→90%」更能定位改善發生在哪一段。

### A-8. 可重現的評估集

- 用 `task_config` YAML 固定物體與容器的起始位姿
- **桌面貼尺規網格**
- 每次評估跑同一組配置

> ✅ **理由：不同模型／checkpoint 比較時，差異才真的是模型差異，而不是評估場景設置的差異。**
> 這正好補上我們 `eval/README.md` 缺的一塊。

---

## 🟡 B 級：方向可參考，數字不可搬

### B-1. 多樣性 > 絕對數量（Diffusion Policy ＋ UMI 手持夾爪）

《Data Scaling Laws in Imitation Learning for Robotic Manipulation》ICLR 2025 Oral

> 「The **diversity** of environments and objects is far more important than the absolute number of demonstrations; once the number of demonstrations per environment or object reaches a certain threshold, additional demonstrations have minimal effect.」

**⚠️ 為什麼只能算 B 級：**
- **用 Diffusion Policy，不是 ACT**
- **用 UMI 手持夾爪收資料，不是機械臂遙操作**（靠 SLAM 取末端動作，約 90% 有效，且「introduces inherent small errors」）
- **論文自己列為 limitation：「we model the data using only Diffusion Policy algorithm. Future research can investigate how... policy learning algorithms affect data scaling laws.」**

**可參考的部分：** 方向與 A-1 的 Try1/Try2 對照一致（分布 > 筆數）——**兩個獨立來源指向同一件事，這讓方向較可信，但數字仍不可搬。**

**附帶發現：** 他們把 action diffusion U-Net 從 512 放大到 2048 維，**效能沒提升，最大的甚至略降**。→ 對「該不該換更大模型」是個反面訊號。

### B-2. 各隨機化因子主要只對自己有效（Diffusion Policy ＋ 模擬）

《A Study on Enhancing the Generalization Ability of Visuomotor Policies via Data Augmentation》

**Table II（訓練用某因子 → 測試於各因子，六任務平均成功率）：**

| 測試環境 | 無增強 | 相機姿態增強 | 光照增強 | 桌面材質增強 | 桌高增強 | 跨embodiment |
|---|---|---|---|---|---|---|
| 相機姿態隨機 | 0.25 | **0.77** | 0.24 | 0.28 | 0.16 | 0.30 |
| 光照隨機 | 0.56 | 0.28 | **0.76** | 0.45 | 0.63 | 0.52 |
| **桌面材質隨機** | **0.12** | 0.08 | 0.16 | **0.75** | 0.18 | 0.20 |
| 桌高隨機 | 0.50 | 0.17 | 0.53 | 0.39 | **0.73** | 0.56 |

**我從這張表讀到的（與論文行文略有出入，標示清楚）：**

1. ✅ **每個因子主要只救自己**——對角線是 0.73–0.77，離對角線多半掉到 0.2–0.5
2. 🔴 **有些組合會變更差**：相機姿態增強在桌高測試只有 **0.17**（無增強是 0.50）、在材質測試 **0.08**（無增強 0.12）
   → **論文文字說「mutually reinforcing」，但表格顯示並非普遍成立。我採信表格。**
3. 🔴 **桌面材質變化最致命**：無增強只剩 **0.12**——比相機姿態（0.25）和桌高（0.50）都慘

**⚠️ 為什麼只能算 B 級：** Diffusion Policy、主要在模擬（RoboMimic/robosuite），真機只有一個 SO-101 的 PPO sim2real 小驗證（Grasp Cube：無增強 0.28 → 有增強 0.44）。

**對我們 L4 衰減曲線的意義：**
- **量測時應該一次只變一個因子**（光照／背景／材質），因為它們的影響不互通
- **背景／材質變化可能是最陡的那一段**，值得優先量

---

## 🔴 C 級 ／ 仍必須自己驗證的

| 問題 | 為什麼沒人答過 |
|---|---|
| **空容器（輕、可壓扁、易被推走）的抓取** | 所有案例用的是方塊、筆、雞蛋。**「接觸瞬間把物體推走」是空容器特有的失敗模式**，沒有現成數據 |
| **落入車體上固定垃圾桶** | 目標區在移動平台上，與桌面容器不同 |
| **recovery data 在我們設定下的劑量** | RaC 有 dose ablation，但不是 SO-101、不是 ACT |
| **觸覺模態（D008 的 V2/V3）** | 完全沒有現成結果 |
| **鋁罐反光面 vs 貼紙面** | 我們自己設計的受控變因 |

---

## 📋 據此可以省掉／改變的實驗

| 原本規劃 | 處置 | 依據 |
|---|---|---|
| 先做 baseline 確認「ACT 自我修正能力差」 | ❌ **省掉** | A-2：已量化為 6.45% |
| 探索「幾筆 demo 才夠」 | ❌ **省掉** | A-1：50 筆是門檻，關鍵在分布 |
| 用 3×3 固定網格點錄 50 筆 | ⚠️ **改設計** | A-1：固定點 → Try 1 的 0%。**改用分層抽樣 ＋ 視覺化 xy 覆蓋** |
| 只錄正對姿態 | ⚠️ **改設計** | A-3：**加 ±45° 方向擾動**，可能免費換到恢復能力 |
| 二元成功／失敗評估 | ⚠️ **改設計** | A-7：**改用 progress score** |
| 不切 eval set | ⚠️ **改設計** | A-6：**必須切**，否則無法選 checkpoint |
| 決策門檻 60–70% 是否可達 | ✅ **確認可達** | A-1：Try 2 已 60%、Pen Transfer 75% |

---

## ⚠️ 一個先前給錯、已撤回的建議

**曾建議「Random cropping 一致提升表現，列為 P8」**——那是 TTIC 論文在**相機視角隨機化**情境下的結論。

**實務端說法相反：** LeRobot 預設的 image transforms 含隨機旋轉與平移，**對 pick-and-place 有害**——等於告訴模型「物體在畫面哪裡都對應同一個動作」，而那正是本任務要學的東西。實作者的做法是**關掉空間變換、只留 color jitter**。

**修正後的說法：**
- **固定相機的 pick-and-place（我們的 L1/L2）→ 關掉空間增強**
- **相機會動（L4 上車）→ 才考慮空間增強或 camera conditioning**

---

## 來源

| # | 來源 | 等級 |
|---|---|---|
| 1 | [How I Trained ACT on SO-101: My Journey, Gotchas, and Lessons（Sherry Chen, HuggingFace Blog）](https://huggingface.co/blog/sherryxychen/train-act-on-so-101) ／ [程式碼](https://github.com/sherrychen1120/so101_bench) | 🟢 A |
| 2 | [Benchmarking VLA Models on SO-101: Failure and Recovery Analysis (arXiv 2606.08881)](https://arxiv.org/abs/2606.08881) — **400 筆資料集已公開釋出** | 🟢 A |
| 3 | [LeRobot ACT 官方文件](https://huggingface.co/docs/lerobot/v0.6.0/en/act) | 🟢 A |
| 4 | [搭建 LeRobot 大作戰 - 終部曲（iThome）](https://ithelp.ithome.com.tw/articles/10398910) | 🟢 A（低精度任務 10–20 筆可成功；**戳觸控板 20 筆失敗**、**夾雞蛋夾爆四顆**） |
| 5 | [Data Scaling Laws in Imitation Learning (arXiv 2410.18647)](https://arxiv.org/abs/2410.18647) | 🟡 B（DP＋UMI） |
| 6 | [Enhancing Generalization via Data Augmentation (arXiv 2511.09932)](https://arxiv.org/html/2511.09932) | 🟡 B（DP＋模擬） |
| 7 | [Do You Know Where Your Camera Is?（TTIC/TRI）](https://ttic.edu/ripl/assets/publications/jiang26.pdf) | 🟡 B（有測 ACT，但在模擬＋UR5） |

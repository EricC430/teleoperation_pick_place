# 深度感測原理對照：雙目／主動雙目／ToF／結構光

> 2026-10-17 查證筆記。起點是一段「已校正雙目 vs 帶深度傳感器的單一相機（RGB-D/ToF）」的 AI 生成比較表，
> 逐條查證後**不能照抄**——分類本身就有問題，下面是修正版。
> 標示：`已查證`（有來源，見文末）／`推論`／`未確認`。

**跟本專案的關係：** 目前錄製與推論都是 `use_depth: false`（`configs/record_omx.yaml:38`），深度沒進模型 `已查證`。
這份是背景知識；日後要開深度時再回來看。

---

## 1. 先修正分類：「RGB-D」是輸出格式，不是測距原理

| 原理 | 怎麼量 | 例子 |
|---|---|---|
| 被動雙目 | 兩顆鏡頭找對應點 → 視差 → 深度 | **RealSense D405**（無 IR 投射器）、Subaru EyeSight |
| 主動雙目 | 雙目 ＋ IR 點陣投射器補紋理 | **RealSense D435i、D455** |
| ToF（AMCW） | 發調變近紅外光，量相位差 | Azure Kinect |
| 結構光 | 投已知圖案，看變形 | iPhone Face ID |

- 原文把「雙目」和「RGB-D」對立起來 → **不成立**。**本專案三顆 RealSense 都是雙目**，同時也是 RGB-D 相機 `已查證`。
- 「單一相機」也不準：RGB-D 相機裡至少有 RGB 感測器＋IR 感測器（＋發射器），深度和 RGB 要另外對齊 `推論`。

## 2. 逐條判定

| 原文說法 | 判定 | 修正後 |
|---|---|---|
| 室外強光：雙目極佳、ToF 極差 | ⚠️ 方向對、程度過頭 | 被動雙目不受陽光干擾 ✅。主動雙目在日光下看不到自己的投射圖案，**退化成被動雙目，仍可用**（Intel 對 D415/D435 的說明 `已查證`；套到 D455 是 `推論`）。ToF 會被環境光干擾，但 Azure Kinect 官方把 global shutter 列為「改善陽光下表現」的設計——是**退化**，不是「完全不能用」；退化多少 `未確認`（Sensors 2021 有測直射陽光，全文沒讀到）。「陽光越強照片越清晰」誇大，強光也會過曝、反光 `推論` |
| 全黑：雙目無法工作、RGB-D 極佳 | ✅ 對被動雙目成立 | 主動雙目靠投射器在暗處也能出深度。注意：全黑時 RGB 影像本身也廢了，只剩深度能用 `推論` |
| 白牆（無紋理）：雙目失敗、RGB-D 精準 | ⚠️ 部分對 | 被動雙目確實失敗；**D405 沒有投射器，所以會受影響** `已查證`。主動雙目靠投射器補紋理。ToF 量白牆沒問題 ✅ |
| 「幾何形狀免疫」 | ❌ | 白牆的問題是**紋理**不是幾何。ToF／結構光／主動雙目**共同的弱點**是透明、鏡面、深色低反射物體；ToF 在牆角還有多路徑干擾，Azure Kinect 會直接把這些像素標成無效 `已查證` |
| >5 m：雙目基線寬就能看遠、ToF 3–5 m 後精度下滑 | ✅ 原理對 | 雙目誤差 **ΔZ ≈ Z²/(f·B)·Δd**，隨距離**平方**成長，加大基線 B 或焦距 f 可壓低 `已查證`。Azure Kinect 量程：NFOV 0.5–3.86 m（binned 0.5–5.46 m）、WFOV 0.25–2.88 m（unbinned 0.25–2.21 m）`已查證`。「3–5 m」只限這類消費級 ToF 相機，LiDAR 也是 ToF，看得遠很多 `推論` |
| 邊緣與近距離：RGB-D 極高、邊緣非常銳利、毫米級 | ❌ | **ToF 的邊緣正是弱點**：前景與背景混進同一像素（flying / mixed pixels），Azure Kinect 文件明寫物體邊緣是典型無效區 `已查證`。近距離反而是雙目的強項（誤差 ∝ Z²）：D405 是被動雙目，理想距離 7–50 cm，±2% @ 50 cm `已查證`。Azure Kinect 隨機誤差 ≤17 mm、系統誤差 <11 mm + 0.1% 距離 → **公分級，不是毫米級** `已查證` |
| ToF「直接測量，不燒算力」、雙目要耗 GPU | ❌ 看產品不看原理 | Azure Kinect 把原始 IR 訊號送到主機，**由主機 GPU 加速的 depth engine 算深度** `已查證`。反過來 RealSense D4xx 由相機上的 Vision Processor D4 算深度，主機不用算 `已查證` |

## 3. 應用例

| 原文 | 判定 |
|---|---|
| Subaru EyeSight 用雙目 | ✅ `已查證` |
| Tesla 用雙目 | ❌ Tesla 是多相機＋神經網路估深度，不是校正雙目 `已查證` |
| Face ID 屬 RGB-D | ✅ 但它是**結構光**（3 萬點 IR 點陣），不是 ToF `已查證` |
| 戶外無人機避障、掃地機器人、倉儲 AMR | `未確認`（這次沒查） |

## 4. 若日後要在本專案開深度

- 寶特瓶是透明物體：RealSense D415 拍不到玻璃瓶的深度（ClearGrasp 相關報導）`已查證（二手）`；
  透明物體對雙目、ToF、結構光都是共同弱點，換相機原理解決不了 `已查證`。紙杯沒有這個問題 `推論`。

---

## 來源

- Azure Kinect 深度相機原理、無效像素（多路徑、邊緣、飽和）、主機 GPU depth engine：<https://learn.microsoft.com/en-us/azure/kinect-dk/depth-camera>
- Azure Kinect 量程與誤差規格：<https://learn.microsoft.com/en-us/azure/kinect-dk/hardware-specification>
- Azure Kinect 獨立評估（含室外陽光，**全文未讀**，403）：<https://www.mdpi.com/1424-8220/21/2/413>
- ToF flying pixels：<https://arxiv.org/pdf/2103.16693>（Mask-ToF）、<https://arxiv.org/pdf/1505.05459>
- RealSense 日光下投射圖案失效（D415/D435）：<https://desk.zoho.com/portal/dotproduct/en/kb/articles/scanning-best-practices-for-intel-realsense-d415-and-d435>
- RealSense D4 板上計算深度：<https://www.rutronik.com/fileadmin/rutronik/Micropages/Intel/Update_2021/IntelRealSense.pdf>
- D405 無投射器、7–50 cm：<https://support.realsenseai.com/hc/en-us/community/posts/4822657940371/comments/4865978335635>、<https://www.mybotshop.de/Intel-RealSense-Depth-Camera-D405_1>
- 雙目誤差 ∝ Z²：<https://arxiv.org/pdf/1505.01936>、<https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Gallup2008CVPR.pdf>
- 透明物體：<https://arxiv.org/pdf/2110.14217>（Dex-NeRF）、<https://androidcentral.com/googles-cleargrasp-ai-model-clears-things-computers>
- Face ID 結構光：<https://en.wikipedia.org/wiki/Face_ID>
- Subaru 雙目 vs Tesla 純視覺：<https://getjerry.com/insights/tesla-follows-subarus-lead-for-driver-assist-technology>

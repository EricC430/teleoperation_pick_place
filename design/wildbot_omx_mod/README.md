# wildbot × OMX-F 改裝概念（v0.1，2026-10-03）

可行性結論、數字與放棄條件的正本在 Artifact 頁面「Wildbot × OMX 改裝藍圖」。這個資料夾只放產生它的程式與輸出，不重複結論。

## 內容

| 路徑 | 說明 |
|---|---|
| `src/design.py` | 參數化改裝場景。所有設計參數集中在 `P`（座板位置、撐環、相機、光達、托塊、收納姿態） |
| `src/analysis.py` | 地面可夾範圍求解（平面查表＋fcl 碰撞） |
| `src/camera.py` | D455 光線投射：視野覆蓋、遮擋、手臂佔畫面比例 |
| `src/run_all.py` | 一次重算全部數字並輸出 `out/` |
| `src/fig_reach.py` `src/fig_concept.py` | 可夾範圍圖、概念圖 |
| `src/compare.py` `src/cam_alt.py` | 手臂座位置比較、相機位置比較 |
| `src/export_viewer.py` `src/build_page.py` `src/page_template.html` | 互動頁面的模型與 HTML |
| `out/assembly_*.glb` | 整車組合（四個姿態），可用 Blender 或 https://gltf-viewer.donmccurdy.com 開 |
| `out/stl/` | 每個改裝件的 STL（概念尺寸，不是加工圖） |
| `out/results.json` | 所有可行性數字 |
| `renders/` | 概念圖、可夾範圍圖、D455 模擬畫面 |

## 重跑

```bash
pip install trimesh manifold3d python-fcl rtree shapely embreex fast-simplification scipy matplotlib
git clone --depth 1 https://github.com/ROBOTIS-GIT/open_manipulator third_party/open_manipulator   # OMX 網格，URDF 用 package:// 引用
python src/reach.py      # 先產生 planar_table.npz（約 125 MB，不進版控）
python src/run_all.py    # 約 3–4 分鐘
python src/fig_reach.py
```

座標：地面 z=0，車頭 −x，右 +y（右手系）。關節限位讀 `sim/omx_constants.py` 的原廠值，TCP 用柏宇 9/21 實測的 link5 前 8 cm。

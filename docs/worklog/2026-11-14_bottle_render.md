# 2026-11-13–14 寶特瓶外觀渲染：黑色問題、渲染模式、材質決定

性質：分析（4090 上渲染測試）。參與：柏宇＋Claude；材質規則柏宇與 EricC430 討論後決定（11/14）。
圖都在 `outputs/bottle_look_test/`（不進 git）；渲染工作目錄 `~/isaaclab_volume/dr/bottle_look_test_20261113`、`bottle_look_test2_20261113`、`bottle_sheet_20261114`、`bottle_sheet2_20261114`。
除特別註明，都是同一集（10/22 閉環 `cl_ep03` 或 `cl_ep00`）、同一個變體種子，只改一個設定。

## 1. 結論

| 項目 | 決定 | 依據 |
|---|---|---|
| 渲染模式 | 寶特瓶用 **RTX Real-Time 2.0**（`dr_config.json` `render.bottle_kit_args`；`dr_collect.py` 依 `objects.json` 的 `shape` 帶入）。紙杯不變（舊版即時模式） | §3 |
| 路徑追蹤 | **不用**：40 幀 99 s vs 即時約 31 s（含約 20 s 啟動）→ 每幀約 2 s vs 0.3 s，6–7 倍。柏宇：為了準確度慢 6–7 倍不划算 | §3 |
| PET 顏色 | 只留無色（原本無色／無色／淡藍／淡綠、抖動 0.02）。柏宇：空的透明瓶不會有顏色，有色看起來像有色玻璃瓶或裝了液體 | §2 |
| PET 陰影 | `primvars:enableFastRefractionShadow`（`pet_fast_refraction_shadow: true`） | §4 |
| PET 材質 | 完全透明的瓶子 OmniGlass；**有標籤的瓶子 `pbr_opacity` 0.15**（`pet_material_labelled`） | §5 |

## 2. 透明 PET 多透明（11/13 查網路）

- 未改質 PET 1 mm 試片 550 nm 穿透率約 85–90%（供應商頁面，非標準文件）；瓶壁很薄（專利例子 0.30 mm）。`推論`：無色空瓶在畫面上主要只剩邊緣反光和瓶蓋。
- PET 折射率 1.57、透明瓶真實影子的樣子：**沒查到來源**。

## 3. 「黑色」是渲染模式造成的

舊版即時模式（Isaac Sim 5.1 預設 `RaytracedLighting`，`/persistent/rtx/modes/rt2/enabled = False`）下，有標籤的玻璃瓶有兩處黑：瓶底一個黑橢圓（畫在不透明標籤前面）、標籤上緣內側。

排除過的原因（每項都實際渲染對照，`已查證`）：PET 投影（關掉後瓶底仍黑）、標籤下緣離地的縫（5.5 → 0.5 mm 沒變）、PET 底面圓盤（拿掉後 USD 12002 → 12001 點，仍黑）、標籤離 PET 太近（實測整圈在 PET 外 0.3 mm 以上；推到 3 mm 仍黑）、單／雙面、`/rtx/translucency/virtualDepth`、`maxRefractionBounces` 12。
`/rtx/translucency/worldEps`（預設 1.0）調成 0.001 時，整支 PET 的上緣變黃、瓶底仍黑；把標籤後面的 PET 切掉時瓶底消失、上緣變黑——兩者互斥。

- **路徑追蹤**（`renderer=PathTracing`，log 印出 `RENDERMODE PathTracing` 確認；先前一次以為切了其實沒切）：兩處都消失 → 黑是即時模式對透明折射材質的近似，不是模型／材質設定。
- **Real-Time 2.0**（`kit_args=--/persistent/rtx/modes/rt2/enabled=true`，rendermode 變 `RealTimePathTracing`）：瓶底黑消失，速度和舊版差不多；前視角上緣剩一條較細的暗帶。`/rtx/rtpt/maxSpecularAndTransmissionBounces` 3／8／16 外觀相同。
- 來源：Omniverse 文件說舊版即時模式已被 Real-Time 2.0 取代、即時模式以準確度換速度；論壇有 OmniGlass 在即時模式變暗、調高折射次數解決的案例（Composer，非 Isaac Sim）。

## 4. 陰影

一般陰影下 PET 像不透明實心體一樣投影：瓶腳（標籤以下的透明段）在前視角發黑、地上瓶影很深。設 `primvars:enableFastRefractionShadow`（容器內 grep 得到這個名稱）後瓶腳變透明、瓶影變淡（`diag_wrist_shadow.jpg`）。`primvars:doNotCastShadows` 也能去掉瓶腳黑，但瓶子完全沒影子。

## 5. 手腕往瓶內看是黑的 → 有標籤的瓶子改 `pbr_opacity`

contact sheet（`bottle_sheet_20261114`，4 支重播 × 5 變體，全部 RT 2.0）裡最明顯的問題：手腕相機從上方透過透明肩部看進有標籤的瓶子，瓶內一大片黑。

- **實機參考**：`omx_pick_place_pilot_plastic_bottle_normal_A1` ep0／ep1 手腕畫面（每秒一張，`real_wrist_normalA1.jpg`）：同一支御茶園從上往內看是**亮的**，看得到標籤內側反過來的印刷。（10/22 閉環資料集在這台只有 front-left 影片，同一集的實機手腕畫面拿不到。）
- **原因有兩個**（cl_ep00 v3，f130／145／160）：
  1. RT 2.0 的近似：路徑追蹤下標籤內壁有被照亮，RT 2.0 全黑；在標籤內側加發光內襯（300、3000）也只亮出上緣一圈細邊，瓶內仍黑。
  2. 模擬的標籤完全不透光：路徑追蹤下瓶內仍有一半是陰影，比實物暗很多（實物標籤是薄膜、會透光）。`推論`
- `pbr_opacity`（半透明、無折射）在 RT 2.0 下瓶內從黑變灰；代價是前視角 PET 變霧白（0.15 輕微、0.3 像磨砂瓶）。`diag_pbr_vs_glass_wrist.jpg`。
- 決定（柏宇＋EricC430，11/14）：有標籤用 `pbr_opacity` 0.15，完全透明的瓶子維持玻璃（沒有標籤就沒有封閉的瓶內，contact sheet 裡透明瓶的手腕畫面正常）。
- **沒做**：會透光的標籤材質（可解原因 2，解不了原因 1）。

## 6. 其他

- `dr_render.py`：標籤滿版時 `uniform()` 上下界因浮點誤差顛倒而報錯 → 已修（`max(za, zb - h)`）。
- `dr_render.py` 測試參數：`override` 的 `label_color`（純色標籤）、`pbr_opacity`；命令列 `rtx=`（啟動後改 carb 設定）、`kit_args=`、`renderer=`；log 印 `RENDERMODE`（用來確認模式真的有切）。試錯用的選項（PET 隱藏／不投影／切網格／內襯／標籤外推）已刪。
- 11/11 `bottle_v2_cl_20261022` 只跑到 10 集就停，沒有 `validate_summary.json`；這次在 `bottle_sheet_20261114` 用 `--episodes 0..9 --all-outcomes` 補產生（ep07–09 物理重跑，各約 4 分鐘）：實機成功 5、模擬進桶 4（ep06 實機成功、模擬沒成功），10 集都沒有穿過。
- contact sheet 每支重播 5 個變體約 6.5–8 分鐘（RT 2.0，整集）；最終設定那版（`bottle_sheet2_20261114`）8.5–9.5 分鐘。
- 最終設定 contact sheet（`bottle_sheet2_20261114`，20 變體）：20/20 log 印 `RealTimePathTracing rt2 True`，材質規則 20/20 正確（有標籤 15 → `pbr_opacity` 0.15、透明 5 → glass），PET 全是無色。場景（地面、HDRI）和前一版相同，但 ep00_v1、ep08_v1 從透明變成有標籤：PET 色票從 4 種改 1 種，改變了抽樣的亂數序列（`推論`，沒逐步追）。手腕看有標籤的瓶內從黑變灰。
- contact sheet 裡 ep08 的瓶子離 front-left 很近、佔畫面左側約三分之一：**沒有對照實機畫面**，不知道是擺位本來就近還是位置錯。

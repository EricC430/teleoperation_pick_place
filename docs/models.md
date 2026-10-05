# 模型對照表（正本）

> **同一份權重會同時出現在三個名字底下**：本機訓練目錄（`data/train/<run>/checkpoints/<step>`）、
> Hub repo（`ericc430/act_omx_*`）、Hub branch（`20k`／`40k`／`100k`）。**只有權重 hash 能證明它們是同一個模型。**
> 其他地方（open-loop 結果、eval CSV、會議紀錄）提到模型時寫 **hash 或 Hub repo**，到這裡查它是什麼，
> 不要各自再寫一次「物品／步數」。
>
> - hash = `sha256(model.safetensors)` 前 12 碼。本機：`sha256sum .../pretrained_model/model.safetensors | cut -c1-12`
> - `scripts/eval_open_loop.py` 會自動把它寫進 `metrics.json` 的 `weights_sha256_12`（2026-09-29 起）。
> - 表內 hash 都是 2026-09-29 從本機檔案和 Hub API（`/api/models/<repo>/tree/<rev>` 的 LFS oid）逐一比對出來的 `[已查證]`。

## ACT／Diffusion 模型

| hash | 本機 run / step | Hub repo（@branch） | 物品 | 訓練資料 | 閉環用過（config record） |
|---|---|---|---|---|---|
| `e9e73dc35268` | `phase_a_pilot` / 500 | `act_omx_pick_place_pilot` | `[未確認]` | `EricC430/omx_pick_place_pilot` | — |
| `0f9289fdd08c` | `phase_b1_uvc60` / 20k | `act_omx_b1_uvc60` | 紙杯 | `omx_pick_place_pilot_uvc_60`（全 60 集） | 9/14 `__8e0858b2`；9/17 `__9fa886a1`、`__f23baed2`（→ `rollout_omx_b1_uvc60_paper_cup_20260917_234916`） |
| `2c227a8e2dde` | `phase_b1_uvc60` / 40k | `act_omx_b1_uvc60_40k` | 紙杯 | 同上 | 9/18 `__6be0f432`（5 集，中止） |
| `2bdff2b3871e` | `phase_b1_uvc60` / 100k | `act_omx_b1_uvc60_100k` | 紙杯 | 同上 | 9/18 `__e07d8d97`、`__2c355eb4`、`__531812f6`、`__15e38ea5`、`__5b288ba5`（5 份 record，4 個 rollout 資料夾：一般／wristcovered／fall／lamp_without_light） |
| `f1903a9f9385` | `phase_b1_diffusion` / 40k | `diffusion_omx_b1_uvc60`：**hash 不符**（Hub 是 `f2012eadb36d`，推測是 EMA 權重 `[未確認]`） | 紙杯 | `omx_pick_place_pilot_uvc_60_frontleft_only`（只有前左相機） | 9/18 `__70a6cc3f`、`__1caea808`（1 集測試） |
| `ed81ea9f6cff` | `phase_b4a_plastic_bottle` / 40k | `act_omx_b4a_plastic_bottle`（main ＝ @40k） | 寶特瓶 | `omx_pick_place_pilot_60_plastic_bottle`（全 60 集） | 9/18 `__83b0d5fd` |
| `8b2c3550f4ef` | `phase_b4a_plastic_bottle` / 20k | `act_omx_b4a_plastic_bottle_20k` | 寶特瓶 | 同上 | 9/21 `__fd20946e` |
| `b6ccbd37e23f` | `phase_b4a_plastic_bottle` / 100k | `act_omx_b4a_plastic_bottle_100k`、`act_omx_b4a_plastic_bottle@100k` | 寶特瓶 | 同上 | 9/21 `__8cc9f08a` |
| `16cbe5967269` | `alcan60` / 20k | `act_omx_alcan60@20k` | 鋁罐 | `omx_pick_place_pilot_60_alcan_20260918_100719`（全 60 集，**修正前**：含 ep 49 放錯與 4 集失敗） | — |
| `24c9dedaa5d7` | `alcan60` / 40k | `act_omx_alcan60@40k` | 鋁罐 | 同上（修正前） | — |
| `13f868813eb4` | `alcan60` / 100k | `act_omx_alcan60@100k`（main 沒有權重） | 鋁罐 | 同上（修正前） | — |
| `3d61371203ab` | `alcan60_fixed` / 20k | `act_omx_alcan60_fixed_20k` | 鋁罐 | `omx_pick_place_pilot_60_alcan_fixed`，`exclude_episodes: [8, 18, 22, 46]` → 56 集 | — |
| `a73bb321b675` | `alcan60_fixed` / 40k | `act_omx_alcan60_fixed_40k` | 鋁罐 | 同上 | 9/20 `__b567a640` |
| `02bf0d69a088` | `alcan60_fixed` / 100k | `act_omx_alcan60_fixed_100k` | 鋁罐 | 同上 | 9/20 `__b504fd10`、`__a6f2a67a`（**沒有對應的 rollout 資料夾** `[未確認]`）；9/21 `__039dfb74` |

未列：60k／80k checkpoint（沒上 Hub、沒閉環過）、`my-002`／`smoke-001`／`perf-*`／`openloop-eval`（煙霧測試與效能測試）。

## 「閉環用過」這欄的來源

Windows 上 `D:\teleoperation_pick_place\config_records\*\rollout_*.yaml` 的 `path:` 與 `cli overrides:`（2026-09-29 由柏宇貼出）。
**這些 config record 還沒 commit 進 git**——commit 之後，這欄的正本就是 `config_records/`，這裡只留指標。
各次閉環的成功數不放這裡，放 eval CSV（`eval/README.md`）或會議紀錄。

## 新增模型時

1. 算 hash，加一行。
2. 推上 Hub 時，repo 或 branch 名稱標步數（例如 `_40k`）。`act_omx_b4a_plastic_bottle` 這種不帶步數的 main，事後只能靠 hash 認出來。

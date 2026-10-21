# 模型對照表（正本）

> **同一份權重會同時出現在三個名字底下**：本機訓練目錄（`data/train/<run>/checkpoints/<step>`）、
> Hub repo（`ericc430/act_omx_*`）、Hub branch（`20k`／`40k`／`100k`）。**只有權重 hash 能證明它們是同一個模型。**
> 其他地方（open-loop 結果、eval CSV、會議紀錄）提到模型時寫 **hash 或 Hub repo**，到這裡查它是什麼，
> 不要各自再寫一次「物品／步數」。
>
> - hash = `sha256(model.safetensors)` 前 12 碼。本機：`sha256sum .../pretrained_model/model.safetensors | cut -c1-12`
> - `scripts/eval_open_loop.py` 會自動把它寫進 `metrics.json` 的 `weights_sha256_12`（2026-09-29 起）。
> - 表內 hash 都是 2026-09-29 從本機檔案和 Hub API（`/api/models/<repo>/tree/<rev>` 的 LFS oid）逐一比對出來的 `[已查證]`。
>   2026-10-08 加的 `cup_uvc60_normal_tight_A1` 三列只有本機檔案（沒上 Hub），hash 取自 `eval_open_loop.py` 的 `metrics.json`。
  2026-10-10 加的 `cup_uvc60_normal_tight_far_A1`／`cup_uvc60_normal_tightcut_far_A1` 六列同樣只在本機，hash 是直接 `sha256sum` 本機檔案算的；
  `tight_far_A1` 三列也和 10-09 開環的 `metrics.json` 對過，一致。
  2026-10-12：三個 run 的 100k 上 Hub（皆 private）的 hash 用 Hub API 的 LFS oid 對過，一致；20k／40k 沒上傳。
  repo 名的數字是排除後的集數（uvc_60 60－normal 13－recovery 10－far 46）。
  `tight_far_A1`／`tightcut_far_A1` 六列「閉環用過」記為 — `[推論]`：`config_records/` 沒有對應的 rollout record、
  Hub 上 10/6 之後沒有新的 rollout dataset、`docs/worklog/2026-10-10_open_loop.md` 的「閉環評估（ntfA1／ntcfA1）」待辦仍未勾。
  這是「沒有紀錄」，不等於確定沒跑過；之後如果跑了閉環，用 record digest 取代 —。
  10-13：ntfA1 100k（`e97526c13766`）試跑一場、作廢，見該列。
  2026-10-13 加的 `cup_uvc60_normal_tight_far_A1_aug`／`cup_uvc60_normal_tightcut_far_A1_aug` 六列：hash 是直接 `sha256sum` 本機檔案算的，也和開環的 log 對過；同日上 Hub（private），Hub API 的 LFS oid 對過，一致。
  2026-10-14 加的 `…_aug_fldrop_p30_w30` 三列：同樣 `sha256sum` 本機檔案，上 Hub（private）後以 LFS oid 對過，一致；Hub 的 `train_config.json` 看不出 dropout，看同 repo 的 `fl_dropout.json`。
  Windows 另外從 Hub 下載到 `D:\hf\hub` 後 `sha256sum`，三個也都一致（2026-10-14）。
  2026-10-15 加的 `…_aug_fldrop_p50_w30` 三列：同樣 `sha256sum` 本機檔案，上 Hub（private）後以 LFS oid 對過，一致；dropout 設定看同 repo 的 `fl_dropout.json`。
  同日加的 `…_augfl_fldrop_p30_w30` 三列：`sha256sum` 本機檔案，也和開環 `metrics.json` 對過；上 Hub（private）後以 LFS oid 對過，一致。`fl_dropout.json` 的 `image_transforms_cameras` 只有 front-left（Hub 的 `train_config.json` 看不出手腕沒平移）。
  2026-10-18 加的 `bottle60_augfl_fldrop_p30_w30` 三列：Windows 從 Hub 下載到 `D:\hf\hub` 後 `sha256sum`，和 Hub API 的 LFS oid 對過，一致；本機訓練檔沒對（不在這台）。`fl_dropout.json` 的 `image_transforms_cameras` 只有 front-left。

## ACT／Diffusion 模型

| hash | 本機 run / step | Hub repo（@branch） | 物品 | 訓練資料 | 閉環用過（config record） |
|---|---|---|---|---|---|
| `e9e73dc35268` | `phase_a_pilot` / 500 | `act_omx_pick_place_pilot` | `[未確認]` | `EricC430/omx_pick_place_pilot` | — |
| `0f9289fdd08c` | `phase_b1_uvc60` / 20k | `act_omx_b1_uvc60` | 紙杯 | `omx_pick_place_pilot_uvc_60`（全 60 集） | 9/14 `__8e0858b2`；9/17 `__9fa886a1`、`__f23baed2`（→ `rollout_omx_b1_uvc60_paper_cup_20260917_234916`） |
| `2c227a8e2dde` | `phase_b1_uvc60` / 40k | `act_omx_b1_uvc60_40k` | 紙杯 | 同上 | 9/18 `__6be0f432`（5 集，中止） |
| `2bdff2b3871e` | `phase_b1_uvc60` / 100k | `act_omx_b1_uvc60_100k` | 紙杯 | 同上 | 9/18 `__e07d8d97`、`__2c355eb4`、`__531812f6`、`__15e38ea5`、`__5b288ba5`（5 份 record，4 個 rollout 資料夾：一般／wristcovered／fall／lamp_without_light）；10/6 N=30、10/7 遮手腕、10/7 temporal ensemble 三場（episode_meta 有 CSV，**config record 未入庫** `[未確認]`） |
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
| `24df8d1ce78b` | `cup_uvc60_normal_tight_A1` / 20k | —（`push_to_hub: false`） | 紙杯 | `omx_pick_place_pilot_paper_cup_uvc60_normalA1_tightA1`（本機 merge：uvc_60 ＋ normal_A1 ＋ recovery_A1_tight，`exclude_episodes: [87]` → 88 集；`configs/train_omx_cup_uvc60_normal_tight_A1.yaml`） | — |
| `dbf3d7b250de` | `cup_uvc60_normal_tight_A1` / 40k | — | 紙杯 | 同上 | — |
| `e6ef7c0e9528` | `cup_uvc60_normal_tight_A1` / 100k | `act_omx_b1_paper-cup_60-16-12rcvry_100k`（10-08 手動上傳；Hub LFS oid 與本機 `sha256sum` 皆為 `e6ef7c0e9528` `[已查證 2026-10-15]`） | 紙杯 | 同上 | 10/8 `__9c5e6c68`（16:58，未對應 CSV）、`__03d8e780`（17:09，N=100 → `rollout_..._20261008_171620`）、`__2a7a2415`（19:29，N=30 → `rollout_..._20261008_192921`）（`config_records/2026-10-08_omx_follower/`）`[已查證 2026-10-13：config record 時間對 CSV 檔名]` |
| `9eb05d355edc` | `cup_uvc60_normal_tight_far_A1` / 20k | — | 紙杯 | `omx_pick_place_pilot_paper_cup_uvc60_normalA1_tightA1_farwristA1`（本機 merge：uvc_60 ＋ normal_A1 ＋ recovery_A1_tight ＋ far_wrist_A1 = 137 集，`exclude_episodes` 8 集 → 129 集；排除理由見 `configs/train_omx_cup_uvc60_normal_tight_far_A1.yaml` 檔頭） | — |
| `c39680beee6a` | `cup_uvc60_normal_tight_far_A1` / 40k | — | 紙杯 | 同上 | — |
| `e97526c13766` | `cup_uvc60_normal_tight_far_A1` / 100k | `act_omx_b1_paper-cup_60-13-10rcvry-46far_100k` | 紙杯 | 同上 | 10/13 `__b29e300c`：只錄 2 集，整場作廢、之後重測 `[柏宇說]` |
| `daa97e08f62b` | `cup_uvc60_normal_tightcut_far_A1` / 20k | — | 紙杯 | `omx_pick_place_pilot_paper_cup_uvc60_normalA1_tightA1cut_farwristA1`（同上，但 recovery_A1_tight 換成剪掉「故意失手」前段的 `_cut` 版，129 集；`configs/train_omx_cup_uvc60_normal_tightcut_far_A1.yaml`） | — |
| `0aa6817a06ac` | `cup_uvc60_normal_tightcut_far_A1` / 40k | — | 紙杯 | 同上 | — |
| `0319057c1d65` | `cup_uvc60_normal_tightcut_far_A1` / 100k | `act_omx_b1_paper-cup_60-13-10rcvrycut-46far_100k` | 紙杯 | 同上 | — |
| `63329a748421` | `cup_uvc60_normal_tight_far_A1_aug` / 20k | `act_omx_b1_paper-cup_60-13-10rcvry-46far-aug_20k` | 紙杯 | 同 `cup_uvc60_normal_tight_far_A1`，加隨機平移增強（`configs/train_omx_cup_uvc60_normal_tight_far_A1_aug.yaml`；理由見 `docs/handoff/2026-10-13_aug_retrain.md`） | — |
| `f6dfcff27ca7` | `cup_uvc60_normal_tight_far_A1_aug` / 40k | `act_omx_b1_paper-cup_60-13-10rcvry-46far-aug_40k` | 紙杯 | 同上 | — |
| `23c53e3026bb` | `cup_uvc60_normal_tight_far_A1_aug` / 100k | `act_omx_b1_paper-cup_60-13-10rcvry-46far-aug_100k` | 紙杯 | 同上 | 10/13 `__67c20c30`（`2026-10-13_closed_loop.md`） |
| `e0894f917c32` | `cup_uvc60_normal_tightcut_far_A1_aug` / 20k | `act_omx_b1_paper-cup_60-13-10rcvrycut-46far-aug_20k` | 紙杯 | 同 `cup_uvc60_normal_tightcut_far_A1`，加隨機平移增強（`configs/train_omx_cup_uvc60_normal_tightcut_far_A1_aug.yaml`） | — |
| `f877d679843f` | `cup_uvc60_normal_tightcut_far_A1_aug` / 40k | `act_omx_b1_paper-cup_60-13-10rcvrycut-46far-aug_40k` | 紙杯 | 同上 | — |
| `286a49c37d4e` | `cup_uvc60_normal_tightcut_far_A1_aug` / 100k | `act_omx_b1_paper-cup_60-13-10rcvrycut-46far-aug_100k` | 紙杯 | 同上 | 10/13 `__98627c1e`（`2026-10-13_closed_loop.md`） |
| `74029f33e571` | `cup_uvc60_normal_tightcut_far_A1_aug_fldrop_p30_w30` / 20k | `act_omx_b1_paper-cup_60-13-10rcvrycut-46far-aug-fldrop30_20k` | 紙杯 | 同 `cup_uvc60_normal_tightcut_far_A1_aug`，加接近段 front-left dropout（p 0.3、第一次閉合前 30 幀；`scripts/train_fl_dropout.py`，設定在 repo 的 `fl_dropout.json`；理由見 `docs/handoff/2026-10-13_fl_dropout.md`） | — |
| `4d954bc6d1c7` | `cup_uvc60_normal_tightcut_far_A1_aug_fldrop_p30_w30` / 40k | `act_omx_b1_paper-cup_60-13-10rcvrycut-46far-aug-fldrop30_40k` | 紙杯 | 同上 | — |
| `7701bdd277ad` | `cup_uvc60_normal_tightcut_far_A1_aug_fldrop_p30_w30` / 100k | `act_omx_b1_paper-cup_60-13-10rcvrycut-46far-aug-fldrop30_100k` | 紙杯 | 同上 | 10/14 `__f92698e7`（`docs/handoff/2026-10-13_fl_dropout.md`「閉環結果」） |
| `df7a951a5752` | `cup_uvc60_normal_tightcut_far_A1_aug_fldrop_p50_w30` / 20k | `act_omx_b1_paper-cup_60-13-10rcvrycut-46far-aug-fldrop50_20k` | 紙杯 | 同 `…_aug_fldrop_p30_w30`，只把 dropout p 改成 0.5（理由見 `docs/handoff/2026-10-14_fl_dropout_p50.md`） | — |
| `cac19f085404` | `cup_uvc60_normal_tightcut_far_A1_aug_fldrop_p50_w30` / 40k | `act_omx_b1_paper-cup_60-13-10rcvrycut-46far-aug-fldrop50_40k` | 紙杯 | 同上 | — |
| `d826a9465f29` | `cup_uvc60_normal_tightcut_far_A1_aug_fldrop_p50_w30` / 100k | `act_omx_b1_paper-cup_60-13-10rcvrycut-46far-aug-fldrop50_100k` | 紙杯 | 同上 | 10/16 `__ab0df39c`＋續錄 `__ca4db9c5`（`2026-10-16_closed_loop.md`：28/36） |
| `cf3dd8f3b40b` | `cup_uvc60_normal_tightcut_far_A1_augfl_fldrop_p30_w30` / 20k | `act_omx_b1_paper-cup_60-13-10rcvrycut-46far-augfl-fldrop30_20k` | 紙杯 | 同 `…_aug_fldrop_p30_w30`，但隨機平移只套 front-left、手腕不套（`train_fl_dropout.py --transforms-fl-only`；理由見 `docs/handoff/2026-10-14_fl_augfl.md`） | — |
| `8f0ad601c627` | `cup_uvc60_normal_tightcut_far_A1_augfl_fldrop_p30_w30` / 40k | `act_omx_b1_paper-cup_60-13-10rcvrycut-46far-augfl-fldrop30_40k` | 紙杯 | 同上 | — |
| `aebc8680b965` | `cup_uvc60_normal_tightcut_far_A1_augfl_fldrop_p30_w30` / 100k | `act_omx_b1_paper-cup_60-13-10rcvrycut-46far-augfl-fldrop30_100k` | 紙杯 | 同上 | 10/16 `__b9d19490`（`2026-10-16_closed_loop.md`：34/36） |
| `c7c9dfceb900` | `uvc60_augfl_fldrop_p30_w30` / 20k | `act_omx_b1_paper-cup_60-augfl-fldrop30_20k` | 紙杯 | `omx_pick_place_pilot_uvc_60`（全 60 集）；增強與 dropout 同 `…_augfl_fldrop_p30_w30`（`configs/train_omx_uvc60_augfl_fldrop.yaml`；理由見 `2026-10-16_closed_loop.md` §4-1） | — |
| `27c5ba98b2c4` | `uvc60_augfl_fldrop_p30_w30` / 40k | `act_omx_b1_paper-cup_60-augfl-fldrop30_40k` | 紙杯 | 同上 | — |
| `f55b0fa71b50` | `uvc60_augfl_fldrop_p30_w30` / 100k | `act_omx_b1_paper-cup_60-augfl-fldrop30_100k` | 紙杯 | 同上 | 10/17 `__40c363cb`（`2026-10-17_uvc60_augfl_closed_loop.md`，20/36） |
| `0dd8b31a90a0` | `bottle60_augfl_fldrop_p30_w30` / 20k | `act_omx_b4a_plastic-bottle_60-augfl-fldrop30_20k` | 寶特瓶 | `omx_pick_place_pilot_60_plastic_bottle`（全 60 集，同 `phase_b4a_plastic_bottle`）；增強與 dropout 同 `uvc60_augfl_fldrop_p30_w30`（`configs/train_omx_bottle60_augfl_fldrop.yaml`；理由見 `2026-10-17_uvc60_augfl_open_loop.md` §4） | — |
| `0eea003c1abb` | `bottle60_augfl_fldrop_p30_w30` / 40k | `act_omx_b4a_plastic-bottle_60-augfl-fldrop30_40k` | 寶特瓶 | 同上 | — |
| `35ffc9043713` | `bottle60_augfl_fldrop_p30_w30` / 100k | `act_omx_b4a_plastic-bottle_60-augfl-fldrop30_100k` | 寶特瓶 | 同上 | — |

未列：60k／80k checkpoint（沒上 Hub、沒閉環過）、`my-002`／`smoke-001`／`perf-*`／`openloop-eval`（煙霧測試與效能測試）。

## 「閉環用過」這欄的來源

Windows 上 `D:\teleoperation_pick_place\config_records\*\rollout_*.yaml` 的 `path:` 與 `cli overrides:`（2026-09-29 由柏宇貼出）。
**這些 config record 還沒 commit 進 git**——commit 之後，這欄的正本就是 `config_records/`，這裡只留指標。
各次閉環的成功數不放這裡，放 eval CSV（`eval/README.md`）或會議紀錄。

## 新增模型時

1. 算 hash，加一行。
2. 推上 Hub 時，repo 或 branch 名稱標步數（例如 `_40k`）。`act_omx_b4a_plastic_bottle` 這種不帶步數的 main，事後只能靠 hash 認出來。

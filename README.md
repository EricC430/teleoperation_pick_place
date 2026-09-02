# teleoperation_pick_place

Imitation-learning pick-and-place on a real robot arm using
[LeRobot](https://github.com/huggingface/lerobot) + ACT.

> 🔴 **Platform, as of 2026-08-24: OMX-AI, not SO-ARM.** SO-ARM's ETA slipped to 2026-09-15, which
> falls on the first day of the team's zero-capacity window (書審/報名, see `docs/phase_plan.md`).
> **D002's pre-registered reversal condition fired and D021 supersedes it.** SO-ARM's role after it
> arrives was **settled on 2026-08-27**: OMX to the end, SO-ARM is hardware redundancy only — no
> comparison demos, because cross-arm demos cannot be pooled (D021, option 甲). Older text in this repo that
> says "SO-ARM" as the platform predates 2026-08-24. Successor to the audio-only /
sim2real direction (see `../legacy_audio_grasp_detection/`), which was shelved because it couldn't be
validated on real hardware within the project timeline.

**Principle: this repo holds everything needed to *reproduce* an experiment. It does not hold large
files.** Datasets, model checkpoints, and raw video live in Hugging Face Hub (private repos) and/or the
lab NAS — see [Where the data lives](#where-the-data-lives) below. Anything checked in here should be
small enough to `git clone` in seconds.

## Status

🚧 Bring-up in progress (started 2026-08-11). This README is a living document — fill in the `TODO`
sections below as each step is actually validated, don't front-load commands that haven't been run yet.

Phase A 管線驗證進行中（已完成 8 筆 Pilot 軌跡錄製 `omx_pick_place_pilot`，包含紙杯抓取與自我修正）。
本階段核心目標為打通「資料集驗證 → 容器化 ACT 訓練 → 隔離 Open-Loop 評估」之全管線。

## Quickstart: Phase A 訓練與開環評估

```bash
# 1. 確保資料集已載入至主機路徑：
#    data/huggingface/lerobot/EricC430/omx_pick_place_pilot
ls data/huggingface/lerobot/EricC430/omx_pick_place_pilot/meta/info.json

# 2. 實際幀數與資料一致性驗證（退出碼必須為 0）
./scripts/run_container.sh python scripts/verify_dataset.py /workspace/data/huggingface/lerobot/EricC430/omx_pick_place_pilot

# 3. 登入 Weights & Biases（首次執行，輸入 API Key 後持久化於 data/.netrc）：
./scripts/run_container.sh wandb login
# 或於主機環境變數設定：export WANDB_API_KEY="your_api_key"

# 4. 執行 ACT 模型訓練（8 筆資料：前 7 筆訓練、保留第 7 筆作為隔離評估，批次 16，500 步）
./scripts/run_container.sh lerobot-train --config_path configs/train_omx_pilot.yaml

# 5. 執行 Open-Loop 開環評估（以 Checkpoint 評估隔離之 Episode 7）
./scripts/run_container.sh python scripts/eval_open_loop.py \
  --checkpoint=/workspace/data/train/phase_a_pilot/checkpoints/last/pretrained_model \
  --dataset.root=/workspace/data/huggingface/lerobot/EricC430/omx_pick_place_pilot \
  --episodes 7
```

### Using `run_container.sh`

It has two modes, which is why it appears twice above:

| Invocation | What happens |
|---|---|
| `./scripts/run_container.sh` | You get an interactive shell **inside** the container, in `/workspace` (this repo). |
| `./scripts/run_container.sh <command...>` | `<command...>` runs inside the container, then the container exits. Step 4 is this form. |

Anything works in the second form — `lerobot-info`, `python --version`, `ls /workspace/data`.

**Always run it from the host.** There is no `docker` inside the container, so running it from a
container shell fails with `exec: docker: not found`. Check the prompt if unsure:

```
boyuchen@widm:~/teleoperation_pick_place$    <- host, run it here
boyuchen@06b0b64be1ff:/workspace$            <- already inside the container
```

Note also that `~` inside the container is **not** your host home directory, so `cd ~/teleoperation_pick_place`
will not work there — the repo is at `/workspace`.

**重要參數說明：**

- `--policy.push_to_hub=false` — 預設為 true。若未指定且未給予 hub repo 會中斷，本設定已納入 yaml。
- `wandb.enable: true` — 已於 `configs/train_omx_pilot.yaml` 啟用，訓練即時指標與 `eval_loss` 將上傳至 Weights & Biases（專案：`teleoperation_pick_place`）。若欲離線訓練可覆蓋 `--wandb.enable=false`。

`--output_dir` **must start with `/workspace/`**. That is the container's name for this repo — the
only host directory mounted into it — so anything written there lands on the real disk.

> 🔴 **A path outside `/workspace/` loses the run, silently.** `/tmp/...`, `/root/...` and friends
> are the container's own throwaway filesystem; `--rm` deletes them on exit. Training completes,
> reports success, writes checkpoints — and they are gone the moment the container stops. There is
> no warning, so a multi-hour run can vanish with nothing to show for it.

It also refuses to reuse an existing directory, so give each run its own — that is deliberate, it
stops a rerun from quietly overwriting an earlier result.

**Never pipe a training run into `tail`/`grep` and trust the exit code** — the status you get back
is `tail`'s, so a crashed run looks like success. Redirect to a file and read the file.

Phase A Pilot 訓練使用 8 筆實機軌跡（`omx_pick_place_pilot`），批次 16、500 步預計於 2–3 分鐘內完成。
相關資源佔用與管線驗證基準記錄於 **[`docs/pipeline_validation.md`](docs/pipeline_validation.md)**。

## Container image

The goal is an environment defined by a **Dockerfile in this repo**, not by instructions a human
follows by hand — same principle as everything else here. We're not there yet: right now we run
LeRobot's official image as-is and carry the run flags in this README.

**Verified on the lab GPU box (2026-08-12):**

| Component | State |
|---|---|
| GPU / driver | RTX 4090, driver `550.54.14` → **CUDA 12.4 ceiling** |
| `nvidia-container-toolkit` | `1.17.8-1`, installed |
| Docker `nvidia` runtime | registered in `/etc/docker/daemon.json` |
| Docker permissions | user is in the `docker` group — **no `sudo` needed, and none available** |
| Docker `data-root` | `/ssd/docker` (images/containers live on the SSD, not `$HOME`) |
| Image contents | LeRobot `0.6.2`, torch `2.11.0+cu128`, Python 3.12.3 |

⚠️ **A plain `--gpus all` does not work with this image.** The image wants CUDA 12.8, the driver caps
at 12.4, and the forward-compat libraries it bundles to bridge that gap are unsupported on GeForce
cards. The two extra flags in the Quickstart are the workaround, and it is a workaround —
**[`docs/environment.md`](docs/environment.md) explains why, and what the real fixes are.** Read it
before trusting a long training run to this setup.

**Still TODO:**

- [ ] Write the `Dockerfile` — derive from the upstream image and purge `cuda-compat`, which removes
      both run-time flags. Verified to work; the five lines are in `docs/environment.md`. Note a
      cu124 build is *not* an option (LeRobot needs torch ≥ 2.7, cu124 stops at 2.6).
- [ ] Pin LeRobot **by digest** in the Dockerfile (`latest` is not a pin — see `docs/environment.md`).
- [ ] Add the remaining `docker run` flags once hardware is back:
  - serial bus servo boards → `--device /dev/ttyACM*` (see `scripts/setup_device_bindings.sh`)
  - cameras → `--device /dev/video*`
  - HF Hub cache → mount a host volume, or every run re-downloads the dataset
- [ ] Decide whether the laptop (data collection + inference) also runs the container, or stays on uv.
      This blocks the cross-machine verification in `docs/environment.md`.

Datasets and checkpoints are **not** baked into the image; they come from HF Hub at runtime, same as
before. See [Where the data lives](#where-the-data-lives).

## Repo layout

| Path | Contents |
|---|---|
| **`docs/experiment_spec.md`** | ★★ **Read this before collecting any data.** Frozen task/success definitions, failure codes, object list, scene constants, dataset schema, evaluation protocol, decision rules, environment gotchas. |
| **`docs/decisions.md`** | ★ Decision log. Each entry records the alternatives, the reasoning, the accepted costs, and — crucially — **what evidence would reverse it**. |
| **`docs/phase_plan.md`** | ★ Phase A→D plan (v3), the cut-list for when time runs short, and **§T0 與實驗室可及性** — lab visits are the scarce resource, plan around them. |
| `docs/execution_plan.md` | Execution plan, reasoning, and contingencies behind `phase_plan.md`. |
| `docs/act_evidence.md` | Literature and measured evidence backing the ACT choice and the recovery-data hypothesis. |
| `docs/camera_mount.md` | Camera position vs. mount hardware, the mast-vs-oblique question, and the pixel-displacement measurement protocol. |
| `docs/exp_a3_recovery_ablation.md` | Design of the A3 recovery-data ablation. |
| `docs/conventions.md` | Commit message and branch conventions. |
| `Dockerfile` | ★ Defines the training/inference environment. **Not yet written** — see [Container image](#container-image). |
| **`docs/environment.md`** | ★ Version pinning across machines, the cross-machine consistency check, and the **driver-ceiling / forward-compat gotcha on the GPU box** — read it before debugging any `--gpus all` failure. |
| `docs/pipeline_validation.md` | Training runs done to validate the pipeline (public data only): timings, resource use, bottlenecks, and the pitfalls each run exposed. Not experiment results — those go in `eval/`. |
| `configs/` | Training and evaluation config files. |
| `calibration/` | ★ One file per calibration run, filename dated (`YYYY-MM-DD_<leader\|follower>.json`). Never overwrite — always add a new dated file. This is how we detect "did the calibration drift?" when results suddenly get worse. |
| `scripts/` | Thin wrappers around data collection / training / evaluation / deployment commands. `setup_device_bindings.sh` is **optional** — see the escalation conditions at the top of that file. |
| `analysis/` | Plotting and stats code over `eval/` records. |
| **`episode_meta/`** | ★ Per-episode metadata LeRobot does not record — object, start cell, lighting, outcome, failure mechanism, quality, operator. One CSV per dataset, keyed by `episode_index`, filled by `scripts/annotate_episodes.py`; the fields live in `configs/episode_meta_schema.yaml`, not in code. See `episode_meta/README.md`. |
| `eval/` | ★ Evaluation run logs (CSV) and failure-mode classification tables. See `eval/README.md` for the required columns and run metadata; copy `_template.csv` per run. |
| `docs/hardware.md` | ★ Equipment inventory: models, firmware, USB ports, **plus current status, ETAs, and rejected options with reasons**. Check here first when debugging "did something change". |
| `docs/setup_env.md` | Site baseline: desk layout, lighting, camera placement (with photos). |
| `docs/field_manual.md` | On-site operating manual — the checklist to run through at the lab. Fill in once hardware is actually available. |
| `docs/meeting/` | Meeting notes **and on-site lab-session records** (see `_template.md`). Lab sessions are labelled as such in their title — e.g. `2026-08-24.md` is a field record, not a meeting. |
| `notebooks/` | Exploratory analysis. |

`calibration/` and `eval/` are the two folders that matter most here — last year's failure mode was
"no calibration records" + "no way to tell what broke." Every calibration run and every eval run gets a
dated record, no exceptions.

## Where the data lives

| What | Where | Why not in this repo |
|---|---|---|
| Raw datasets (demos, recovery demos) | Hugging Face Hub, private repo — Phase A pilot: `EricC430/omx_pick_place_pilot`. Real campaigns get a fresh `repo_id`, see `configs/record_omx.yaml` | Large, binary, versioned better by HF Hub / LeRobot tooling than git |
| Model checkpoints | Hugging Face Hub, private repo — Phase A pilot: `EricC430/act_omx_pick_place_pilot` | Same as above |
| Raw video | Lab NAS — `TODO: path` | Large binary, no need to version in git |

If you're missing access to any of the above, ask in `docs/meeting/` notes or the shared doc referenced
there — don't recreate a local-only copy of something that should be centrally stored.

**Setting up push/pull yourself (HF_TOKEN, `hf upload`/`download`, and pulling a checkpoint onto the
laptop to actually drive the arm with `lerobot-rollout`):** see **`docs/field_manual.md` §8** (雲端同步與跨機部署).

## Related

- [huggingface/lerobot](https://github.com/huggingface/lerobot)
- [LeRobot AGENT_GUIDE](https://github.com/huggingface/lerobot/blob/main/AGENT_GUIDE.md)
- [Seeed Studio: SO-10x arm with LeRobot](https://wiki.seeedstudio.com/lerobot_so100m/)

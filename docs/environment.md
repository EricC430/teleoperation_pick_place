# Environment & version pinning

Two machines, one hard requirement.

## The setup

| Machine | Role | Environment | Constraint |
|---|---|---|---|
| Lab GPU (4090 / A6000) | **Training** | Docker (`huggingface/lerobot-gpu`) | Host driver caps the usable CUDA version — see the gotcha below |
| Laptop (RTX 3050 4GB, **Windows**) | **Data collection + inference deployment** | **uv** (decided 2026-08-13, D014) | ⚠️ Issue #4093 — see below |

**Different environments on the two machines are fine.** What is not fine is letting the two
drift onto different library versions.

## What actually has to match

| Component | Must match? | Why |
|---|---|---|
| **LeRobot** | 🔴 **Exactly** | Dataset format, config schema, and CLI arguments change between versions. Record on one version and train on another and you get the worst kind of bug: it loads, it runs, and the fields are subtly wrong. |
| PyTorch | ⚠️ major.minor | Irrelevant during recording (no model involved). Matters when the laptop loads a checkpoint trained on the GPU box. `state_dict` loading is usually backward-compatible across minor versions, but "usually" is not "guaranteed". |
| numpy / pyarrow / datasets | ⚠️ close enough | Serialization format compatibility |
| CUDA / driver | ❌ no | Machine-local concern. The laptop may run CPU inference entirely. |

> The real failure mode is not "which package manager". It is **both machines installing
> `latest` at different times and silently diverging.** Pin explicitly.

## ⚠️ GPU box: the driver ceiling and the forward-compat trap

Read this before debugging any `--gpus all` failure on the lab machine.

**Verified 2026-08-12:**

- Host driver `550.54.14` → **CUDA 12.4 is the hard ceiling**
- `huggingface/lerobot-gpu:latest` ships torch `2.11.0+cu128` → wants **CUDA 12.8**

The obvious command fails at container start:

```
nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.8,
please update your driver to a newer version, or use an earlier cuda container
```

NVIDIA's intended answer is the **forward-compat libraries** the image bundles at
`/usr/local/cuda/compat/` (`libcuda.so.570.124.06`), which let a newer CUDA runtime drive an older
host driver. **Those are only supported on datacenter GPUs (Tesla / A100 / H100), not GeForce.** On
the 4090 they fail with:

```
Error 804: forward compatibility was attempted on non supported HW
```

**The fix is two flags**, both required — verified on 2026-08-12 (`torch.cuda.is_available()` is
`True`, device reports `NVIDIA GeForce RTX 4090`, a 2000×2000 GPU matmul returns):

- `-e NVIDIA_DISABLE_REQUIRE=1` skips the toolkit's `cuda>=12.8` gate. **Not sufficient on its own** —
  with only this, the container still dies with 804.
- `--tmpfs /usr/local/cuda/compat` masks the compat libs with an empty dir, so the container falls
  back to the host driver's own `libcuda`.

**Don't type these by hand — use [`scripts/run_container.sh`](../scripts/run_container.sh).** That
script is the single source of truth for how to start the container; this document only explains
*why* the flags are there. If you find yourself editing a `docker run` line in a doc, edit the
script instead.

### 🔴 Dropping `--tmpfs` does not fail loudly — it falls back to CPU

The two flags fail in completely different ways, and only one of them is obvious.

| Missing | What happens |
|---|---|
| `NVIDIA_DISABLE_REQUIRE=1` | The container **refuses to start**. Impossible to miss. |
| `--tmpfs /usr/local/cuda/compat` | Error 804 makes `torch.cuda.is_available()` return `False`, and LeRobot then logs `Switching to 'cpu'` and **trains anyway** — on CPU. |

Verified 2026-08-12: with the compat libraries left in place, `lerobot-train` emits one
`UserWarning`, prints `'device': 'cpu'`, and proceeds normally. Loss falls, checkpoints are written,
nothing errors. The only symptom is being **roughly 50–100× slower** — easy to mistake for "the
machine is busy today".

**Check every run.** The config dump near the start of the log carries the answer:

```
'device': 'cuda',    <- correct
'device': 'cpu',     <- the GPU has silently dropped out
```

`nvidia-smi` showing 0 % utilisation during training says the same thing.

**This is a workaround, not a fix.** It works because of CUDA *minor version compatibility*: a cu12.8
build runs on any 12.x driver **as long as it never calls an API that genuinely requires ≥570**. If
it does, it fails at the call site, mid-run — not at startup. Budget for that possibility before
trusting a long training job to it.

### Where the compat libraries actually come from

Worth being precise, because it changes what a fix has to do. **The compat libraries are not
torch's.** A pip-installed torch wheel bundles the CUDA *runtime* (cudart, cuBLAS, …) but never
`libcuda` — that always comes from the host driver. The `libcuda.so.570` in this image arrives with
the `cuda-compat` package in its `nvidia/cuda:*` base layer.

So the CUDA version torch was built against is not what breaks; **the base image is.**

Two clean fixes, in order of preference:

| Fix | Blocker |
|---|---|
| Update the host driver to ≥570 | **No `sudo` on the GPU box** — needs the lab admin |
| Derive our own image from theirs, purging `cuda-compat` | `Dockerfile` isn't written yet (see README) |

### The derived-image fix (verified 2026-08-12)

Five lines, builds in seconds, and keeps the **exact** package set we validated over 1000 steps —
no rebuild from a clean base, no re-validation:

```dockerfile
FROM huggingface/lerobot-gpu:latest
USER root
RUN apt-get purge -y cuda-compat-12-8 && rm -rf /usr/local/cuda/compat
ENV NVIDIA_DISABLE_REQUIRE=1
USER user_lerobot
```

Verified: `torch.cuda.is_available()` is `True`, device is the 4090, a GPU matmul returns — with
**neither run-time flag present**. `lerobot 0.6.2 / torch 2.11.0+cu128` unchanged.

Two things this does *not* do, both measured rather than assumed:

- **`NVIDIA_DISABLE_REQUIRE` cannot be dropped, only relocated.** Clearing `NVIDIA_REQUIRE_CUDA` in
  the image is *not* enough — the toolkit falls back to `Auto-detected mode as 'legacy'` and reads
  the CUDA version out of the image's own files, so `cuda>=12.8` still fires. Baking the variable
  into the image is strictly better than passing it per-run (one documented place, nobody deletes it
  by accident), but it is still a bypass.
- **It does not raise the driver ceiling.** torch stays cu128 on a 12.4 driver, working by CUDA
  **minor-version compatibility**; a call that genuinely needs ≥570 still fails, still mid-run. Only
  the driver upgrade removes that. Validated so far: ACT + ResNet18 + transformer + video decode +
  AdamW, 1000 steps. **Not** validated: `pi0` / `smolvla` / `groot` (different kernels — D001 has
  GR00T as a P2 baseline), long runs, `torch.compile`.

⚠️ **A cu124 build is not an option**, so don't reach for one. LeRobot 0.6.2 requires
`torch>=2.7,<2.12` and PyTorch's cu124 index stops at **torch 2.6.0**. Matching the driver's 12.4
exactly would mean downgrading LeRobot — the one thing this document says must match exactly across
machines.

There is **no older tag to fall back to** — Docker Hub carries only `latest` and `pr-3945` for
`huggingface/lerobot-gpu`.

## ⚠️ File ownership and where downloads actually land

Two more traps, both handled by `scripts/run_container.sh`. Documented here because the symptoms
are confusing and the causes are not guessable.

**1. The container's user is not you.** The image runs as `user_lerobot` (uid 1001); on the lab box
you are uid 1020. Without `--user "$(id -u):$(id -g)"`, everything the container writes into the
mounted repo — training logs, `calibration/*.json`, `eval/*.csv` — lands owned by uid 1001 and you
cannot edit it from the host.

Passing `--user` fixes ownership but leaves the container with no matching entry in its own
`/etc/passwd`, which shows up as:

```
groups: cannot find name for group ID 1020
bash: /home/user_lerobot/.bashrc: Permission denied
I have no name!@06b0b64be1ff:/workspace$
```

Mounting the host's account files read-only (`-v /etc/passwd:/etc/passwd:ro`, same for `/etc/group`)
clears all three.

**2. `HOME` is not enough to redirect the caches.** The image hard-codes absolute paths:

```
HF_HOME=/home/user_lerobot/.cache/huggingface
HF_LEROBOT_HOME=/home/user_lerobot/.cache/huggingface/lerobot
TORCH_HOME=/home/user_lerobot/.cache/torch
TRITON_CACHE_DIR=/home/user_lerobot/.cache/triton
```

Setting `HOME` alone does **not** move them — `huggingface_hub` reads `HF_HOME`, and the download
fails with `PermissionError: '/home/user_lerobot/.cache/huggingface/token'`. All four variables must
be overridden explicitly.

They point at `/workspace/data`, i.e. **`data/` inside this repo**, which `.gitignore` already
excludes. So datasets are visible on the host, survive container exit, and still never reach git —
consistent with `experiment_spec.md` §12. Note this is a *local cache*, not storage: our own demo
data still belongs on HF Hub (see the README's [Where the data lives](../README.md#where-the-data-lives)).

> Without a volume mount, downloads land inside the container and `--rm` deletes them on exit — the
> dataset is re-fetched every single run.

## 🔴 Laptop (Windows): you WILL get CPU-only torch. This is by design, not a bug.

**Verified 2026-08-13 on our laptop:** after `uv pip install -e ".[core_scripts,feetech]"`,
`torch.__version__` reported `2.11.0+cpu` and `torch.cuda.is_available()` was `False`.

### Root cause — it is written into LeRobot's own `pyproject.toml`

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
torchvision = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
```

**The CUDA index is applied only when `sys_platform == 'linux'`.** On Windows the marker is false, uv
falls back to the PyPI default, and the PyPI default for Windows is the CPU build. This is LeRobot
issue #4093 — not a transient bug, a hard-coded platform condition. **We cannot fix it upstream and
should not edit the cloned repo.**

### Why installing torch first does not work

LeRobot's own constraints will overwrite whatever you pre-installed:

```toml
"torch>=2.7,<2.12.0"
"torchvision>=0.22.0,<0.27.0"
```

Installing `torch 2.13.0+cu126` first is wasted work — both it and `torchvision 0.28.0` sit outside
those bounds and get replaced during the LeRobot install.

> **Order matters: install LeRobot first, then repair torch. Never the other way round.**

### The repair — versions must stay inside LeRobot's bounds

```powershell
uv pip install --force-reinstall "torch==2.11.0" "torchvision>=0.22.0,<0.27.0" `
    --index-url https://download.pytorch.org/whl/cu126
```

Then verify all four:

```powershell
python -c "import torch, torchvision, lerobot; print(torch.__version__, torch.cuda.is_available(), torchvision.__version__, lerobot.__version__)"
```

Expect `2.11.0+cu126` / `True` / `0.26.x` / `0.6.2`.

### ⚠️ This recurs on every reinstall

The marker stays in LeRobot's `pyproject.toml`, so **any future `uv pip install -e ".[...]"` puts the
CPU build back.** Don't rely on remembering — use `scripts/setup_laptop.ps1`, which runs the install
and the repair in the required order and ends with a hard assert.

### Why this failure is nastier than it looks

**A CPU-only torch does not error.** Recording works perfectly — no model is involved. The failure
only surfaces much later, as inference too slow to close the control loop, and at that point the
obvious suspect is the policy, not the install. **The assert exists so it fails at install time.**

### Not fallbacks

WSL2 and VirtualBox. USB camera passthrough is unreliable in both; VirtualBox additionally has no GPU
passthrough. See D014.

### Silver lining: torch versions now match across machines

LeRobot's `<2.12.0` cap pins both machines to torch **2.11.0**. Only the CUDA build variant differs
(GPU box `+cu128`, laptop `+cu126`), and that is a per-machine driver concern, not a compatibility
one. The version-drift risk this document was written to fight is, for torch specifically, handled
by the framework.

## Setup order

Install on the **GPU machine first** — it has the tighter constraints (driver ceiling, no sudo).
Whatever versions resolve cleanly there become the target for the laptop, not the other way round.

## ⏳ Pending: `placo` for FK (D026, decided 2026-08-31, not yet installed)

`scripts/reach_logger.py` (spec `docs/specs/S1_reach_logger.md`) needs `placo` for
`lerobot.model.kinematics.RobotKinematics`. Install path: `uv sync --extra placo-dep`
(or `uv pip install placo`). **As of 2026-08-31 it is NOT installed** (`import placo` →
ModuleNotFoundError, laptop, pinned env).

When installing:
- Do it on the machine that will run the reach logger (the laptop at the lab).
- **Smoke-test after:** `uv run lerobot-teleoperate --help` and a short `lerobot-record` dry run
  must still pass — adding placo must not perturb the existing pipeline.
- Record the resolved `placo` version in the Version record table below.
- If it will not resolve in the pinned env, S1 falls back to a hand-coded transform chain from the
  URDF (D026) — do **not** pull in ROS 2 for this.

## Pinning: `latest` is not a pin

`latest` moves. Record the **digest**, which doesn't.

```bash
# 1. The digest — this is the actual pin
docker image inspect huggingface/lerobot-gpu:latest --format '{{index .RepoDigests 0}}'

# 2. What's actually inside
docker run --rm <image> python -c "import lerobot; print(lerobot.__version__)"
docker run --rm <image> pip freeze > requirements_gpu.txt

# 3. Put the results in the Version record table below
```

## Verification: don't trust, measure

Ten minutes, and it converts "should be fine" into "I checked".

```
1. Both machines pull the SAME public dataset from HF Hub
2. Both run ONE training step (or one forward pass) with a fixed seed
3. Compare the loss value
```

| Result | Verdict |
|---|---|
| Match to ~5–6 decimal places | Environments are compatible. Proceed. |
| Differ in the 3rd–4th decimal | Probably nondeterminism (cuDNN kernels, TF32). Acceptable — but note it here. |
| Differ substantially | **Something is genuinely different. Find it now**, not two weeks into training. |

Run this again after any upgrade on either machine.

## Version record

Update this table whenever either machine is reinstalled or upgraded.
An entry without a date is not a record.

| Date | Machine | Image digest | LeRobot | PyTorch | Driver / CUDA | Installed by | Notes |
|---|---|---|---|---|---|---|---|
| 2026-08-12 | GPU | `huggingface/lerobot-gpu@sha256:62df079f02b7fa26963d35466c12fa230be9f51a3b0ea2327297a84f70041c6c` (image built 2026-08-12) | 0.6.2 | 2.11.0+cu128 | 550.54.14 / 12.4 | | Requires the compat workaround above. Python 3.12.3, numpy 2.2.6. |
| 2026-08-13 | Laptop (Windows 11) | n/a — uv venv, not a container | 0.6.2 | 2.11.0+cu126 | RTX 3050 4GB / driver TBD | | `scripts/setup_laptop.ps1` run successfully. torchvision 0.26.0+cu126, Python 3.12.13. Verified via `docs/field_manual.md` §2. **LeRobot version matches the GPU box exactly** — the hard requirement in the table above is met. |

## Verification log

**Not yet run** — the cross-machine check needs the laptop environment decided first.

| Date | LeRobot version | Loss (GPU) | Loss (laptop) | Verdict |
|---|---|---|---|---|
| | | | | |

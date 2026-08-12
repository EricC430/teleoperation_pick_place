# Environment & version pinning

Two machines, one hard requirement.

## The setup

| Machine | Role | Environment | Constraint |
|---|---|---|---|
| Lab GPU (4090 / A6000) | **Training** | Docker (`huggingface/lerobot-gpu`) | Host driver caps the usable CUDA version — see the gotcha below |
| Laptop (RTX 3050 4GB) | **Data collection + inference deployment** | **Undecided** — container or uv | none |

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

**Working command** (verified — `torch.cuda.is_available()` is `True`, device reports
`NVIDIA GeForce RTX 4090`, a 2000×2000 GPU matmul returns):

```bash
docker run -it --rm --gpus all --shm-size 16gb \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  --tmpfs /usr/local/cuda/compat \
  huggingface/lerobot-gpu:latest
```

- `NVIDIA_DISABLE_REQUIRE=1` skips the toolkit's `cuda>=12.8` gate. **Not sufficient on its own** —
  without the second flag it still dies with 804.
- `--tmpfs /usr/local/cuda/compat` masks the compat libs with an empty dir, so the container falls
  back to the host driver's own `libcuda`.

**This is a workaround, not a fix.** It works because of CUDA *minor version compatibility*: a cu12.8
build runs on any 12.x driver **as long as it never calls an API that genuinely requires ≥570**. If
it does, it fails at the call site, mid-run — not at startup. Budget for that possibility before
trusting a long training job to it.

Two clean fixes, in order of preference:

| Fix | Blocker |
|---|---|
| Update the host driver to ≥570 | **No `sudo` on the GPU box** — needs the lab admin |
| Build our own image on a CUDA 12.4 base with a cu124 torch | `Dockerfile` isn't written yet (see README) |

There is **no older tag to fall back to** — Docker Hub carries only `latest` and `pr-3945` for
`huggingface/lerobot-gpu`.

## Setup order

Install on the **GPU machine first** — it has the tighter constraints (driver ceiling, no sudo).
Whatever versions resolve cleanly there become the target for the laptop, not the other way round.

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
| | Laptop | | | | | | Environment not chosen yet |

## Verification log

**Not yet run** — the cross-machine check needs the laptop environment decided first.

| Date | LeRobot version | Loss (GPU) | Loss (laptop) | Verdict |
|---|---|---|---|---|
| | | | | |

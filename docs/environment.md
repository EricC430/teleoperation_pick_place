# Environment & version pinning

Two machines, two different package managers, one hard requirement.

## The setup

| Machine | Role | Package manager | Constraint |
|---|---|---|---|
| Lab GPU (4090 / A6000) | **Training** | conda (only option available) | CUDA / driver compatibility |
| Laptop (RTX 3050 4GB) | **Data collection + inference deployment** | uv | none |

**Different package managers are fine.** What is not fine is letting the two
drift onto different library versions.

## What actually has to match

| Component | Must match? | Why |
|---|---|---|
| **LeRobot** | 🔴 **Exactly** | Dataset format, config schema, and CLI arguments change between versions. Record on one version and train on another and you get the worst kind of bug: it loads, it runs, and the fields are subtly wrong. |
| PyTorch | ⚠️ major.minor | Irrelevant during recording (no model involved). Matters when the laptop loads a checkpoint trained on the GPU box. `state_dict` loading is usually backward-compatible across minor versions, but "usually" is not "guaranteed". |
| numpy / pyarrow / datasets | ⚠️ close enough | Serialization format compatibility |
| CUDA / driver | ❌ no | Machine-local concern. The laptop may run CPU inference entirely. |

> The real failure mode is not "conda vs uv". It is **both machines installing
> `latest` at different times and silently diverging.** Pin explicitly.

## Setup order

Install on the **GPU machine first** — it has the tighter constraints (CUDA,
conda-only). Whatever versions resolve cleanly there become the target for the
laptop, not the other way round.

```bash
# 1. On the GPU machine: install per LeRobot's official instructions
#    (do NOT copy commands from tutorials, including ours -- LeRobot's install
#     procedure churns; go to the source)

# 2. Freeze what actually got installed
conda list --export > env_gpu.txt
pip freeze > requirements_gpu.txt

# 3. Record the LeRobot version precisely
pip show lerobot | grep -i version
git -C <lerobot-clone> rev-parse HEAD    # if installed from source

# 4. On the laptop: install the SAME versions via uv
#    uv supports exact version pinning; use it.
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

| Date | Machine | LeRobot version / commit | PyTorch | CUDA | Installed by | Notes |
|---|---|---|---|---|---|---|
| | GPU | | | | | |
| | Laptop | | | | | |

## Verification log

| Date | LeRobot version | Loss (GPU) | Loss (laptop) | Verdict |
|---|---|---|---|---|
| | | | | |

# Pipeline validation runs

Records of training runs performed **to validate the pipeline itself**, not to produce results.
Timings, resource use, bottlenecks, and the pitfalls each run exposed.

Everything here uses **public datasets**. No project data, no reportable numbers. Real experiment
results belong in `eval/`, one dated CSV per run — see `eval/README.md`.

Re-measure and add a row whenever the environment changes: a new image, a driver upgrade, our own
`Dockerfile`, or the laptop being brought into the loop.

## Run log

| Date | Machine | Dataset | Config | Result |
|---|---|---|---|---|
| 2026-08-12 | GPU box | `lerobot/svla_so100_pickplace` | ACT, 1000 steps, batch 32, `num_workers` 4 | ✅ 4 min 57 s, see below |
| 2026-08-12 | GPU box | `lerobot/svla_so100_pickplace` | same, `num_workers` 16 | ✅ 5 min 03 s — no gain |
| 2026-08-12 | GPU box | `lerobot/svla_so100_pickplace` | same, batch 8 | ✅ 94 samples/s vs 122 at batch 32 |
| 2026-08-12 | GPU box | `edgarcancinoe/soarm101_pickplace_orange_080e_ts_closed` | ACT, 1000 steps, batch 32 | ❌ crashed at step 167 — corrupt dataset, see below |

## What a healthy run looks like

Baseline: ACT, 1000 steps, batch 32, on the lab 4090. Compare against this before assuming a slow
run is normal.

| | |
|---|---|
| Wall time | **4 min 57 s** |
| Steady state | 3.8 step/s, **122 samples/s**, `step_s` 0.263 |
| Where the time goes | `updt_s` 0.256 of `step_s` 0.263 — **97 % GPU compute**, `data_s` only 0.006 |
| GPU | 97–99 % SM, 330–350 W, 60–63 °C |
| GPU memory | 13.1 GB (torch) / 15.8 GB (nvidia-smi) of 24 GB |
| Data coverage | `epch:1.63` — the full dataset went past 1.6 times |
| Loss | 5.7 → 1.35 (`l1` 0.60 → 0.29) |
| Disk per run | **591 MB** (`pretrained_model/` 207 MB + `training_state/`) |

**The GPU is the bottleneck, and that is the healthy outcome** — the dataloader is keeping up and
the card never idles waiting for data. If `data_s` ever becomes a large fraction of `step_s`, that
is the thing to investigate.

## Two knobs, measured rather than guessed

| Setting | Wall time | samples/s | GPU mem |
|---|---|---|---|
| batch 8 | — | 94 | 3.7 GB |
| **batch 32** | **4:57** | **122** | 13.1 GB |
| batch 32, `num_workers=16` | 5:03 | 121 | 13.1 GB |

- **Batch 32 beats batch 8 by ~30 % throughput** and still uses only 13 of 24 GB. Larger is likely
  still better; we have not swept past 32.
- **`num_workers` does nothing here.** The default of 4 already keeps the GPU fed (`data_s` =
  0.006 s); raising it to 16 came out 6 s *slower*, i.e. noise. Don't tune it.

## Pitfalls

### ⚠️ Ignore the first ~400 steps

`data_s` reads 0.090–0.120 s early and settles to 0.006 s by step 400. Any throughput or bottleneck
conclusion drawn before that is wrong — **we drew two wrong ones this way** (that the dataloader was
a bottleneck, and that larger batches were slower) and had to re-measure. Set `--log_freq` so you
get readings well past step 400.

### ⚠️ Never pipe a run and trust the exit code

`lerobot-train ... | tail` reports the exit status of `tail`, not of the training. A crashed run
looks like success. We lost a run to exactly this. Redirect to a file and read the file.

### ⚠️ A public dataset can be internally inconsistent

`edgarcancinoe/soarm101_pickplace_orange_080e_ts_closed` declares 61,480 frames in
`meta/info.json`, but its videos hold 61,534. The surplus is entirely in `file-000` (7,767 actual
vs 7,713 implied). Global indexing drifts and eventually reads past the end of a later file:

```
IndexError: Invalid frame index=8530 for streamIndex=0; must be less than 8524
```

It surfaced at step 167 — a 100-step smoke test at `epch:0.01` had passed clean. `--dataset.exclude_episodes`
does **not** work around it: the sampler then fails with `KeyError` because its index space is still
built from the wrong total.

The methodology consequence — validate over at least one full epoch — is recorded in
`experiment_spec.md` §10, because it governs our own data collection too.

## Disk

Each run costs ~591 MB under `data/train/`, plus the dataset cache under `data/huggingface/`.
`data/` is gitignored; clear old runs with `rm -rf data/train/<name>` and keep the dataset cache.

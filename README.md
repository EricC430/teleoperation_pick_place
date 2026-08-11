# teleoperation_pick_place

Imitation-learning pick-and-place on a real **SO-ARM (SO-100/SO-101)** using
[LeRobot](https://github.com/huggingface/lerobot) + ACT. Successor to the audio-only /
sim2real direction (see `../legacy_audio_grasp_detection/`), which was shelved because it couldn't be
validated on real hardware within the project timeline.

**Principle: this repo holds everything needed to *reproduce* an experiment. It does not hold large
files.** Datasets, model checkpoints, and raw video live in Hugging Face Hub (private repos) and/or the
lab NAS — see [Where the data lives](#where-the-data-lives) below. Anything checked in here should be
small enough to `git clone` in seconds.

## Status

🚧 Bring-up in progress (started 2026-08-11). This README is a living document — fill in the `TODO`
sections below as each step is actually validated, don't front-load commands that haven't been run yet.

**Blocked on hardware.** Both lab arms are currently unavailable (one on loan, one under repair), so
the on-site steps below are untested. Work that does *not* need hardware — environment setup, pipeline
validation on public datasets, and filling in `docs/experiment_spec.md` — proceeds in the meantime.

## Quickstart: reproduce from zero

```bash
# 1. Install LeRobot (see LeRobot's own install docs — version churns fast, don't hardcode steps here)
# TODO: pin the exact LeRobot commit/version we're using once the pipeline is validated

# 2. Confirm hardware is connected and calibrated
#    See docs/field_manual.md for the full on-site checklist.
ls /dev/tty*                      # leader + follower serial bus servo boards should show up
# TODO: calibration command

# 3. Pull training data (public dataset for pipeline validation, or our own HF Hub repo)
# TODO: lerobot-train --dataset.repo_id=<...>

# 4. Train
# TODO: lerobot-train --policy.type=act --dataset.repo_id=<...> --output_dir=...

# 5. Evaluate / deploy
# TODO: eval script invocation, see eval/
```

## Repo layout

| Path | Contents |
|---|---|
| **`docs/experiment_spec.md`** | ★★ **Read this before collecting any data.** Frozen task/success definitions, failure codes, object list, scene constants, dataset schema, evaluation protocol, decision rules, environment gotchas. |
| **`docs/decisions.md`** | ★ Decision log. Each entry records the alternatives, the reasoning, the accepted costs, and — crucially — **what evidence would reverse it**. |
| `docs/environment.md` | Version pinning across the GPU box (conda) and laptop (uv), plus the cross-machine consistency check. |
| `configs/` | Training and evaluation config files. |
| `calibration/` | ★ One file per calibration run, filename dated (`YYYY-MM-DD_<leader\|follower>.json`). Never overwrite — always add a new dated file. This is how we detect "did the calibration drift?" when results suddenly get worse. |
| `scripts/` | Thin wrappers around data collection / training / evaluation / deployment commands. `setup_device_bindings.sh` is **optional** — see the escalation conditions at the top of that file. |
| `analysis/` | Plotting and stats code over `eval/` records. |
| `eval/` | ★ Evaluation run logs (CSV) and failure-mode classification tables. See `eval/README.md` for the required columns and run metadata; copy `_template.csv` per run. |
| `docs/hardware.md` | Equipment inventory: models, firmware versions, which USB port is which. |
| `docs/setup_env.md` | Site baseline: desk layout, lighting, camera placement (with photos). |
| `docs/field_manual.md` | On-site operating manual — the checklist to run through at the lab. Fill in once hardware is actually available. |
| `docs/meeting/` | Meeting notes (see `_template.md` in that folder). |
| `notebooks/` | Exploratory analysis. |

`calibration/` and `eval/` are the two folders that matter most here — last year's failure mode was
"no calibration records" + "no way to tell what broke." Every calibration run and every eval run gets a
dated record, no exceptions.

## Where the data lives

| What | Where | Why not in this repo |
|---|---|---|
| Raw datasets (demos, recovery demos) | Hugging Face Hub, private repo — `TODO: repo_id` | Large, binary, versioned better by HF Hub / LeRobot tooling than git |
| Model checkpoints | Hugging Face Hub — `TODO: repo_id` | Same as above |
| Raw video | Lab NAS — `TODO: path` | Large binary, no need to version in git |

If you're missing access to any of the above, ask in `docs/meeting/` notes or the shared doc referenced
there — don't recreate a local-only copy of something that should be centrally stored.

## Related

- [huggingface/lerobot](https://github.com/huggingface/lerobot)
- [LeRobot AGENT_GUIDE](https://github.com/huggingface/lerobot/blob/main/AGENT_GUIDE.md)
- [Seeed Studio: SO-10x arm with LeRobot](https://wiki.seeedstudio.com/lerobot_so100m/)

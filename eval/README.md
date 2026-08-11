# Evaluation records

**One CSV per evaluation run.** Filename: `YYYY-MM-DD_<run-label>.csv`
(e.g. `2026-09-03_act-50demo-baseline.csv`)

Never overwrite a previous run's file. If you re-run the same model after changing anything —
calibration, camera position, lighting, object set — that is a **new run** and gets a new file.

## Why this folder matters

Last year's failure mode was *"the model doesn't work and I can't tell why."* A success rate alone
doesn't tell you what to fix. **The failure-mode distribution does.**

The difference between these two sentences is the entire point of this folder:

- ❌ "Success rate is about 40%."
- ✅ "Of 20 failures, 12 were F1 (didn't grasp), concentrated in grid cells 1/4/7 — the left column.
  Hypothesis: under-representation of left-side starts in the demo set."

## Required columns

Copy `_template.csv` for each new run.

| Column | Meaning |
|---|---|
| `trial` | Trial number within this run (1..N) |
| `grid_cell` | Object start position, 1–9 (3×3 grid; see `docs/setup_env.md` for the physical layout) |
| `object_id` | Which object, matching the frozen object list in `docs/experiment_spec.md` §2 |
| `outcome` | `success` / `fail` |
| `failure_code` | One of F1–F7 (blank if success). **Do not invent new codes mid-run** — see below |
| `intervention` | `0` / `1` — did a human have to step in? Used to compute intervention rate |
| `duration_s` | Seconds from start to outcome (for takt-time comparison) |
| `notes` | Free text — anything odd |

## Failure codes (frozen — see `docs/experiment_spec.md` §1)

| Code | Meaning |
|---|---|
| F1 | Didn't grasp — gripper closed but object not in it |
| F2 | Grasped then dropped |
| F3 | Collision — hit object / desk / rack |
| F4 | No motion — arm never started, or stalled in place |
| F5 | Drifted out of bounds — single-direction drift off the workspace |
| F6 | Pushed object away — contact displaced the (light) object instead of grasping |
| F7 | Timeout |

**If a failure genuinely doesn't fit any code:** record it as `F0` with a detailed note, then raise it at
the next meeting. Adding a code is a spec change — it gets a version bump in `docs/experiment_spec.md`
§12, not a silent edit.

## Run metadata (put at the top of each CSV as `#` comment lines)

Every run file must state, before the data rows:

```
# run_label:
# date:
# model_checkpoint:        (HF Hub repo/revision, or local path + git commit)
# dataset:                 (HF Hub repo_id + revision)
# calibration_file:        (filename in calibration/)
# setup_env_version:       (which version of docs/setup_env.md was in force)
# camera_config:           (wrist-only / external-only / both)
# operator:
# notes:
```

**Without `calibration_file` and `setup_env_version`, a bad result is not diagnosable.** That binding is
the whole reason those two folders exist.

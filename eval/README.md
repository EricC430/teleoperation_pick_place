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

> **v2 (2026-08-12):** switched from a single `failure_code` to **two orthogonal axes**.
> **✅ Ratified 2026-08-13** (owner: Eric Chen). See `docs/decisions.md` D015.
>
> **Also ratified 2026-08-13:** each object gets **30 evaluation trials**, not the previously-suggested
> ≥20 (owner: Boyu Chen; `docs/decisions.md` D016). **The 3×3 grid this note used to reference is
> gone (D024, 2026-08-27): placements are drawn once by uniform-area sampling over the annular
> workspace, frozen, and given IDs. 30 no longer has to divide by anything.**

| Column | Meaning |
|---|---|
| `trial` | Trial number within this run (1..N) |
| `placement_id` | Object start position — an ID from the **frozen sampled placement list** (`docs/experiment_spec.md` §5). Sampled once and never re-randomised; that is the whole point of seeding, and re-randomising destroys cross-model comparability (D024) |
| `object_id` | Which object, matching the frozen list in `docs/experiment_spec.md` §2 |
| `object_orientation` | For the aluminium can: `label` (L1) or `bare` (L2). Reflectance is a controlled variable |
| **`valid`** | `1` / `0`. **Invalid trials are excluded from the success-rate denominator** |
| `void_reason` | Only if `valid=0`: motor overheat / camera dropout / human bump / other |
| **`outcome`** | **Exactly one of:** `success` / `no_grasp` / `dropped` / `misplaced` |
| **`mechanism`** | **Zero or more**, semicolon-separated. See table below |
| `intervention` | `0` / `1` — did a human have to step in? Used for intervention rate |
| `duration_s` | Seconds from start to outcome |
| `notes` | Free text |

### Why two axes instead of one code

A single code list mixed two questions and the categories overlapped:
`pushed_away` **always** also means `no_grasp`; `stalled` **always** ends in `timeout`.
Annotators hesitate, two people label the same trial differently, and the resulting distribution is
statistically meaningless — which defeats the point, because **the failure distribution is the only
metric that tells you what to change next.**

| Axis | Question it answers | Used for |
|---|---|---|
| `outcome` | *What happened?* | Success rate. Mutually exclusive, exhaustive |
| `mechanism` | *Why / how?* | Deciding the next iteration. Multi-label |

### `mechanism` values

| Value | Meaning |
|---|---|
| `pushed_away` | Contact displaced the (light) object instead of grasping it — **specific to empty containers** |
| `collision` | Hit the object / desk / bin |
| `drift` | Single-direction drift out of the workspace (observed in the workshop GR00T deployment) |
| `stalled` | Never started, or stopped in place |
| `repetition_loop` | Periodic oscillation or looping over the same trajectory interval without progress |
| `timeout` | Hit the time limit |
| **`self_recovered`** | ⭐ **Went off course and corrected itself.** Valid alongside `outcome=success` |
| `other` | Detail it in `notes` |

> ⭐ **`self_recovered` is deliberately a positive label.** This column records *notable behaviour*,
> not just failure causes. If recovery demos work, this label should appear more often on real
> hardware — **that is the most direct evidence for the A3 hypothesis we can collect.**
> (Human-assisted recovery needs no separate label — `intervention` already covers it.)

**If something genuinely fits no `mechanism` value:** use `other` with a detailed note and raise it at
the next meeting. Adding a value is a spec change — it gets a version bump in
`docs/experiment_spec.md` §12, not a silent edit.

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
# timeout_s:               (the episode time limit in force — see experiment_spec.md §1-2)
# notes:
```

> `timeout_s` is recorded per run because **the limit interacts with what you are measuring**:
> set it too short and a policy that would have recovered is scored as a failure, which
> systematically suppresses exactly the behaviour A3 exists to detect.

**Without `calibration_file` and `setup_env_version`, a bad result is not diagnosable.** That binding is
the whole reason those two folders exist.

# Per-episode metadata (the part LeRobot doesn't record)

**One CSV per dataset**, named after the dataset: `episode_meta/<dataset-name>.csv`.
Written by [`scripts/annotate_episodes.py`](../scripts/annotate_episodes.py), keyed by
`episode_index`.

## Why this folder exists

LeRobot's schema stops where the interesting questions start. What it stores per frame —

```
action, observation.state, observation.images.top, observation.images.wrist,
timestamp, frame_index, episode_index, index, task_index
```

— plus `length`, `tasks` and video time ranges per episode in `meta/episodes/*.parquet`.

What it does **not** store is everything the person at the desk knew and then forgot: which object,
where it started, what the lighting was, whether the demo actually succeeded, and why it didn't.
Without that, a dataset is 200 undifferentiated episodes and "the policy fails on the left side" is
an unanswerable question — the same failure mode `eval/` exists to prevent, one step earlier in the
pipeline.

## 🔴 `omx_pick_place_pilot_uvc_60` has no CSV yet — but its placements are known

`[Eric說 2026-09-21]`: **uvc_60 was recorded walking `campA_136sym`'s `t1..t60` in order**, so
episode *i* used short_id `t{i+1}` (episode 0 → `t1` → `train_001`). Nothing in the dataset records
this, and there is no `episode_meta/omx_pick_place_pilot_uvc_60.csv` — until one exists, that
mapping lives only here and in whoever remembers it.

Corroborated independently before being relied on (2026-09-21): for each episode, take the frame
where the gripper starts closing, run `observation.state` through `reach_logger/fk.py`, and
correlate end-effector position against the claimed placement — **x r=+0.63, y r=+0.66 across all
60**, versus r≈0.00 for 20 shuffled pairings, and weaker at every ±1/±2 shift. So the ordering is
right. The per-episode residual (p50 9 cm after removing a constant +10 cm offset) is big enough
that this corroborates the **rule**, not any individual row — a re-recorded or aborted take would
not show up in it.

To write the CSV (one episode per call, since each row differs), after confirming no episode was
re-shot out of order:

```bash
for i in $(seq 0 59); do
  uv run python scripts/annotate_episodes.py \
      --dataset ericc430/omx_pick_place_pilot_uvc_60 \
      --root .cache/lerobot/omx_pick_place_pilot_uvc \
      --set "placement_id=train_$(printf '%03d' $((i+1)))" --episodes "$i" --no-prompt
done
```

`sim/replay_render_episode.py --place-from-episode` already applies the same rule directly.

## Vocabulary is shared with `eval/`

`outcome`, `mechanism`, `valid` and `void_reason` use **exactly** the two-axis scheme ratified
2026-08-13 (`docs/decisions.md` D015, `docs/experiment_spec.md` §1-3) — the same values the
evaluation records use:

| Field | Values |
|---|---|
| `outcome` | exactly one of `success` / `no_grasp` / `dropped` / `misplaced` |
| `mechanism` | zero or more, `;`-separated: `pushed_away` / `collision` / `drift` / `stalled` / `repetition_loop` / `timeout` / `self_recovered` / `other` |
| `void_reason` | `motor_overheat` / `camera_dropout` / `human_bump` / `other`, only when `valid=0` |

Both are `strict: true`, so a typo is rejected rather than warned about — adding a value is a spec
change (§12), not something to improvise at 11pm during a recording session.

Sharing the vocabulary is the point: **`self_recovered` on demo episodes and `self_recovered` on
eval trials mean the same thing**, so the A3 recovery hypothesis can be traced from what went into
training to what came out of it.

`object_orientation` is here because §2 requires it — the same aluminium can is L1 with the label up
and L2 with the bare metal up, and that is what makes reflectance a single controlled variable.

## The other fields are not fixed

They live in [`configs/episode_meta_schema.yaml`](../configs/episode_meta_schema.yaml) and nowhere
else — not in the script. Add, remove, rename or reorder a field there and the CSV columns and the
prompts follow.

Schema changes are safe to make mid-project:

| Change | What happens to existing rows |
|---|---|
| Add a field | New column, blank for older episodes. `--check` lists them; backfill with `--set`. |
| Rename a field | The old column is **kept**, not dropped, and reported as an orphan. Data is never lost silently. |
| Change `values:` | Existing rows are re-validated on the next `--check`; mismatches are warnings, not errors. |

Suggested `values:` are a menu, not a lock — anything else is accepted with a warning (set
`strict: true` on a field to make it a hard constraint instead).

**Bump `version:` in the schema file when you change it**, and note it in
`docs/experiment_spec.md` §4 + §12. A schema change is a spec change, same as adding an `outcome` or
`mechanism` value.

## Use

Right after a recording session, from the host:

```bash
./scripts/run_container.sh python scripts/annotate_episodes.py \
    --dataset <hf_user>/<dataset>
```

It picks up every episode that has no row yet and asks about each one, showing the episode's length
and task so you can tell them apart. Fields marked `sticky` default to the previous episode's answer,
so a session with one object and one lighting setup is mostly Enter. `?` explains a field, `-` clears
it, Ctrl-C saves what you answered and exits.

```bash
# batch fill / backfill, no questions asked
... annotate_episodes.py --dataset <ds> --episodes 0-9 --no-prompt \
      --set object_name=paper_cup --set operator=boyu

# multi-label fields take several values, ';'-separated
... annotate_episodes.py --dataset <ds> --episodes 3 --no-prompt \
      --set outcome=success --set mechanism="drift;self_recovered"

# validate + coverage report; exits non-zero if anything is missing or invalid
... annotate_episodes.py --dataset <ds> --check

# revisit an episode that is already filled
... annotate_episodes.py --dataset <ds> --redo --episodes 7
```

The script also runs on the host without the container (it needs only `pyyaml`), as long as the
dataset is in the local HF cache or you pass `--root`. Inside the container it can additionally show
each episode's length and task.

> ⚠️ **Interactive mode needs a real terminal.** `run_container.sh` only passes `-i` to Docker when
> stdin is a TTY, so *piping* answers into the containerized script hits EOF at the first question
> and saves nothing. Scripted use → `--no-prompt`, or run the script on the host.

## Rules

- **Annotate the same day you record.** Nobody remembers why episode 23 of 40 was bad a week later.
- **Commit the CSV.** It is small, and it is the one part of the dataset that cannot be regenerated —
  re-deriving it means rewatching video, if the video still exists.
- **Re-run `--check` after any `lerobot-edit-dataset` operation.** Deleting or splitting episodes
  renumbers `episode_index`, and these rows are keyed by it. `--check` will tell you the row count no
  longer matches; it cannot tell you the rows silently point at the wrong episodes.
- `valid=0` excludes an episode from training **without deleting it** — the preferred way to drop a
  bad demo, because the deletion path is what breaks the keying above.

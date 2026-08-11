# Maintenance conventions

Commit and branch conventions for a **two-person team working async**, where the person reading a
commit six weeks from now is often the person who wrote it. The goal is the same one behind
`decisions.md` and `eval/README.md`: don't make a future reader (including us, including the advisor)
re-derive *why* something changed. Optimize for that — not for process ceremony a two-person repo
doesn't need.

If a rule here ever costs more time than the confusion it prevents, that's a sign to change the rule,
not to route around it silently. Propose the change in a PR to this file.

---

## Commits

**Format:**

```
<type>(<scope>): <summary, imperative mood, why not just what>

<body — optional, only if the summary line can't carry the reasoning>
```

- `<type>` — one of:

  | Type | Use for |
  |---|---|
  | `feat` | New capability (a script, a config option, a pipeline stage) |
  | `fix` | Correcting something that was wrong |
  | `exp` | An experiment run, ablation, or its supporting code — **link the eval/calibration file it produced** |
  | `docs` | `docs/`, `README.md`, comments — no behavior change |
  | `refactor` | Restructuring code with no behavior change |
  | `chore` | Dependency pins, `.gitignore`, CI/tooling, repo housekeeping |
  | `data` | Adding/changing a **config or schema** — never actual dataset/checkpoint files, those don't belong in this repo at all (see `.gitignore`) |

- `<scope>` — the folder or subsystem: `calibration`, `eval`, `train`, `env`, `docs`, etc. Omit if the
  commit is genuinely repo-wide.
- **Summary is imperative** ("add", not "added"/"adds"), **≤ 72 chars**, and answers *why* when the
  *why* isn't obvious from the diff. "fix bug" is never acceptable; "fix(calibration): re-zero offset
  was applied twice, doubled the drift" is.
- **When a commit is downstream of a decision or produces an experiment record, say so explicitly** —
  don't make the reader cross-reference dates to guess:
  - ✅ `exp(train): baseline 50-demo run per D001 — see eval/2026-09-03_act-50demo-baseline.csv`
  - ✅ `feat(env): pin lerobot==0.3.2 per D010, cross-machine loss check passed`
  - ❌ `update training script`

**One logical change per commit.** A commit that mixes an experiment result with an unrelated
refactor is a commit nobody can safely `git revert`.

**Never commit:**
- Anything `.gitignore` already excludes (datasets, checkpoints, video — see README's
  [Where the data lives](../README.md#where-the-data-lives)). If you hit a case `.gitignore` doesn't
  catch, add the pattern in the same commit, don't just avoid staging it once.
- Secrets, tokens, HF Hub write keys.
- `git commit --amend` / force-push on `main` once a commit has been pushed — the advisor and
  teammate both pull `main`; rewriting it breaks their checkout for no benefit a two-person team needs.

## Branches

**Default is direct commits to `main`.** This is a two-person research repo, not a service with
uptime to protect — branch-per-typo is pure overhead. `main` is expected to always be in a state that
reproduces (per the README's own promise: "this repo holds everything needed to *reproduce* an
experiment").

**Branch when a change could leave `main` non-reproducing for more than a few commits**, i.e. touches
the pipeline every session depends on: `scripts/`, `configs/`, `docs/environment.md`, the eval harness.
Same logic as `docs/decisions.md`'s "accepted costs" — the branch cost is worth paying exactly when the
blast radius of a broken `main` is real.

- **Naming:** `<type>/<short-description>`, same `<type>` vocabulary as commits —
  `feat/wrist-camera-ablation`, `fix/calibration-offset`, `exp/recovery-data-a3`.
- **Lifetime:** short. Merge (fast-forward or a single merge commit — no rebase gymnastics for a
  two-person team) as soon as it reproduces on both machines, then delete the branch. A branch that
  outlives the change it was for just becomes a second place state can silently diverge — this repo's
  entire `docs/environment.md` exists to fight exactly that failure mode, don't reintroduce it via git.
- **Don't** open a branch for `docs`/`chore` changes, dated `calibration/` or `eval/` files, or anything
  reversible in one commit. Direct-to-`main` is correct there.

## Tags

Tag `main` right before anything gets reported externally — a meeting, a milestone, a number that goes
in the write-up:

```
git tag meeting-2026-08-14
```

This is the same attributability instinct as choosing ACT over GR00T (D001) and per-run eval CSVs: if a
number from an old meeting gets questioned later, you want the exact commit it came from, not "main
around mid-August, probably." Cheap to do, expensive to reconstruct after the fact.

## Where this doesn't apply

`calibration/*.json` and `eval/*.csv` already have their own, stricter convention (dated filename,
never overwrite — see the README's repo-layout table and `eval/README.md`). Follow those, not this
file, for what's *inside* those folders; this file only governs the git history around them.

Decisions that change the project's direction (not just the code) still go in `docs/decisions.md`, not
a commit message — a commit says what changed, a decision entry says what alternatives were rejected
and why. If you're writing a commit body longer than a few lines to justify a choice, that's usually a
sign it belongs in `decisions.md` instead, with the commit just linking to the new `D0NN`.

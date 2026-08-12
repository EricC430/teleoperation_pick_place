# A3 · Recovery-data ablation

**Status:** design in progress (2026-08-12) · **Requires no robot and no simulator**

---

## 1. Hypothesis

> Imitation-learning policies drift because their training data contains **only smooth successes**.
> Human teleoperators rarely demonstrate "I went off course, here is how I corrected" — so the policy
> never learns recovery, and any small error compounds until the target leaves the camera's view.

This is not speculation. It is the conclusion of our own failure analysis of the workshop GR00T
deployment (`experiment_spec.md` §9) — single-joint, single-direction, monotonically accumulating
drift, with the target out of frame by the time it left the workspace.

**Prediction:** a policy trained on data that *includes* corrective trajectories will diverge more
slowly from a perturbed state than one trained on smooth successes only.

---

## 2. Why this can be evaluated offline

The instinct is "we need to run the task and count successes" — which would require hardware or a
simulator, and we have neither.

**But compounding error is by definition a property of how error grows over a horizon, and that is
measurable on recorded trajectories.** No robot required.

---

## 3. ⚠️ There are TWO different splits. Do not confuse them.

This is the part that is easy to get tangled.

```
       full public dataset (N episodes)
                 │
     ┌───────────┴───────────┐
     │                       │
 TRAIN portion          HELD-OUT TEST set
 (~80%)                 (~20%, ≥20 episodes)
     │                       │
     │                  used for evaluation ONLY,
     │                  identical for both groups
     │
 ┌───┴────┐
 │        │
GROUP A  GROUP B
smooth   smooth + corrective
only     trajectories
 │        │
 └─ train an ACT on each ─┘
```

| Split | What it separates | Purpose |
|---|---|---|
| **Split 1 — train / test** | Which episodes the model never sees | So evaluation is on unseen data. **Standard ML practice.** |
| **Split 2 — group A / group B** | *Within the training portion*, which trajectories contain corrective motion | **This is the experimental variable.** It is the whole point of A3. |

> 🔴 **The held-out test set is identical for both groups.** If A and B are evaluated on different
> episodes, the comparison is meaningless.

**Answering the question directly:** yes — evaluation uses **existing recorded trajectories that were
excluded from training**, and the model is rolled out **open-loop** on them.

---

## 4. Three evaluation metrics, weakest to strongest

### ① Teacher-forcing error — weak, do it anyway as a sanity check

Feed the ground-truth state at every step; compare the predicted action to the recorded action.

⚠️ **This cannot detect compounding error** — the ground-truth state resets the error to zero every
step. Use it only to confirm the model trained at all.

### ② Open-loop rollout divergence — ★ core metric

```
Start from frame t of a held-out episode
  ↓
Let the model predict N steps forward, feeding its OWN predictions back as input
(no ground truth after the first frame)
  ↓
Plot: L2 distance between predicted and recorded trajectory  vs.  step index
```

**The difference in slope between group A and group B is the result.**
If recovery data works, B's curve should be flatter — error stops snowballing.

Report: mean divergence at 10 / 25 / 50 / 100 steps, over all held-out episodes.

### ③ Perturbed-recovery test — ★★ most directly tests the hypothesis

```
Take a held-out episode. At frame t, inject an artificial offset into the state
(simulating "the arm has already gone off course")
  ↓
Roll out open-loop from that perturbed state
  ↓
Does the predicted trajectory bend BACK toward the recorded one, or keep going?
```

Metric: offset magnitude vs. step index — **converging or diverging?** And if diverging, in which
direction (does it reproduce the single-joint monotonic drift we saw in the workshop)?

This measures exactly one thing: **does the policy know how to come back from a state it should not
be in** — which is the only thing recovery data is supposed to teach.

**Perturbation sizes:** sweep small / medium / large (e.g. equivalent to 1cm / 3cm / 6cm of
end-effector displacement). A policy may recover from small offsets but not large ones; that
threshold is itself a result worth reporting.

> All three run on a laptop with batch size 1. No GPU cluster, no robot, no simulator.

---

## 5. Building the split criterion — four steps, in this order

**⚠️ Do not design the criterion in the abstract.** You will produce a rule that is elegant on paper
and matches nothing in the data.

### Step 1 — watch the video (start here)

Use LeRobot's dataset visualizer — either the hosted Space
(`huggingface.co/spaces/lerobot/visualize_dataset`) or the local `visualize_dataset` script — and
watch 15–20 episodes.

**Human judgement on video is the gold standard here**, and it is faster and more intuitive than
staring at joint curves. Hesitation, overshoot, a nudged object, a re-approach — all obvious to a
human, all hard to specify numerically up front.

### Step 2 — label by hand

Two piles: *smooth* / *contains correction*. **Write down what your eye was using** — that is the
raw material for the criterion. Record the proportion; if almost nothing lands in the "correction"
pile, that is a finding (see §6).

### Step 3 — now look at the curves

For the episodes you labelled, plot `action` (6 joint angles over time) and `observation.state`.

**The video told you *which* episodes have corrections. The curves tell you what a correction
*looks like numerically*.** You need both — the video alone gives no computable rule, and the curves
alone give no ground truth to check the rule against.

### Step 4 — write the criterion, then verify it against your labels

Candidate features:

| Feature | Rationale |
|---|---|
| path length ÷ straight-line distance | detours score high |
| velocity dips followed by re-acceleration | pausing to re-align |
| direction reversal in any joint | overshoot and return |
| jerk (3rd derivative) peaks | abrupt corrections |

**Report the agreement rate between the automatic criterion and your hand labels.** If it is low, fix
the criterion — do not quietly proceed. This agreement rate belongs in the write-up; it is what makes
the split methodologically defensible rather than arbitrary.

---

## 6. If recovery data cannot be found

| Situation | Response |
|---|---|
| The dataset genuinely has none | Try another dataset; or **synthesize**: truncate a successful trajectory mid-way, inject an offset, splice it back to the original. This is standard data augmentation and is defensible if stated. |
| It exists but the criterion misses it | Tune the criterion — this is why Step 2 (hand labels) exists |
| **Multiple datasets all have none** | **That is itself the finding.** "Public demonstration data systematically lacks corrective trajectories" supports the hypothesis's premise and belongs in the report. |

---

## 7. Sample size and statistical validity

| Item | Minimum |
|---|---|
| Training episodes per group | **30–50** (practical floor for ACT) |
| Held-out test episodes | **≥ 20** |
| **Random seeds per configuration** | **≥ 3** |

> 🔴 **The seed requirement is not optional.**
> If the difference between groups is smaller than the variation between seeds *within* a group,
> you are reporting noise. Run each configuration on 3 seeds, report mean ± spread, and state
> both numbers. Without this, the first question anyone asks sinks the result.

**Also control:** both groups must have the **same number of training episodes**. If group B simply
has more data, the comparison measures data quantity, not data composition. Either subsample A to
match B, or hold total count fixed and vary only the proportion of corrective trajectories.

---

## 8. Execution checklist

- [ ] Pick candidate dataset(s); **verify `robot_type` and schema match SO-100/101** — do not assume from the name
- [ ] Step 1–2: watch and hand-label 15–20 episodes; record proportion and your criteria
- [ ] Step 3–4: plot curves for labelled episodes; draft computable criterion; report agreement rate
- [ ] Freeze the train/test split (held-out set identical for both groups)
- [ ] Build group A and group B with **matched episode counts**
- [ ] Train ACT ×2 groups ×3 seeds = 6 runs
- [ ] Evaluate: metric ① (sanity), ② (divergence curve), ③ (perturbed recovery)
- [ ] Report mean ± spread across seeds

---

## 9. Candidate datasets (verified 2026-08-12)

| Dataset | Size | Note |
|---|---|---|
| **`lerobot/svla_so100_pickplace`** | 19.6k | ⭐ first choice — official LeRobot, pick-place, recently updated |
| `lerobot/svla_so100_stacking` | 23k | official, stacking |
| `00ri/so100_battery_bin_center` | 17.7k | batteries into a bin — functionally closest to waste sorting |
| `sixpigs1/so100_pick_cube_in_box` | 28.2k | |
| `samsitol/so100_PnPacorn` | 84.8k | |

> ⚠️ **A3 does not need a trash-picking dataset.** The hypothesis concerns *training-data
> composition*, not the object. Using the official `svla_so100_pickplace` is **preferable** — clean,
> standard, and reproducible by others, which makes the conclusion more credible.
>
> **A3 is a methodology experiment, not a task experiment.** This distinction recurs; keep it.

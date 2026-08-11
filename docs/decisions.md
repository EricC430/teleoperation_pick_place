# Decision log

Append-only. One entry per decision that would be expensive to reverse, or that a future reader
(including us in three months, or the advisor) would otherwise have to re-derive.

**Format:** what was decided, when, what the alternatives were, why, and — most importantly — **what
evidence would make us reverse it.** A decision without a reversal condition is a belief, not a decision.

---

## D001 — Algorithm: ACT first, not GR00T

- **Date:** 2026-08-11
- **Decision:** Use ACT as the P0 policy. GR00T is deferred to P2+ as a comparison baseline.
- **Alternatives:** GR00T N1.5/N1.6 (used in the workshop), Diffusion Policy, SmolVLA
- **Why:** ACT has low data requirements, trains fast, and — decisively — its failures are
  *attributable*. GR00T is a large VLA; when it fails you cannot easily tell why. Given that our core
  weakness last year was inability to diagnose failures, starting with a debuggable model matters more
  than starting with a strong one.
- **Reverse if:** ACT plateaus below ~60% success with clean data and we've exhausted data-side fixes
  (see decision rule in `experiment_spec.md` §8).

---

## D002 — Platform: SO-ARM (SO-100/SO-101)

- **Date:** 2026-08-11
- **Decision:** SO-ARM is the primary platform. OMX-AI is a backup only — **no data is collected on it.**
- **Alternatives:** OMX-AI (ROBOTIS Physical AI manipulator, 5-DOF, ROS 2 native, factory-calibrated)
- **Why:**
  - Lower contention in the lab → **higher access**. With ~6 months left and hardware already delayed,
    an arm we can touch beats a better arm we can't.
  - Shortest software path: LeRobot runs directly, no ROS 2 layer to learn first.
  - Team member has hands-on SO-101 experience from the NVIDIA/NAPAI workshop.
  - Cheap to buy a spare set (~USD 250–300) if we need redundancy.
- **Accepted costs:**
  - **Calibration is now our responsibility** (OMX ships factory-calibrated). Mitigated by the
    calibration protocol in `experiment_spec.md`.
  - **P4 (mobile base, outdoors) will require wrapping the SO-101 control interface in a ROS 2 node.**
    Budget for this if P4 turns out to be a hard commitment.
  - No vendor-grade simulation for SO-ARM.
- **Reverse if:** SO-ARM stays unavailable beyond ~3 weeks while OMX becomes free, *and* no data has
  been collected yet. **Once demo collection starts, this is effectively irreversible** — imitation
  data does not transfer across arms.

---

## D003 — Task scope: empty containers only

- **Date:** 2026-08-11
- **Decision:** Target objects are limited to **empty** containers.
- **Why:** Payload and reliability. Also removes liquid-sloshing dynamics entirely.
- **Consequence (must be stated in the report):** this is a narrower claim than "generalized trash
  grasping." Say so explicitly rather than letting the reader assume otherwise.
- **Follow-on:** objects are further graded L1/L2/L3 by material difficulty
  (`experiment_spec.md` §2). Transparent and deformable objects are deliberately excluded from P0 —
  transparency is a known-hard perception problem and shouldn't contaminate the baseline.

---

## D004 — Cameras: wrist + third-person, with an ablation

- **Date:** 2026-08-11
- **Decision:** Two cameras (wrist + external third-person). Run wrist-only / external-only / both as a
  controlled comparison.
- **Why:** The workshop setup uses two, for a documented reason: the wrist camera is occluded by the
  object once grasped, so the external camera provides continuity. The ablation additionally
  *quantifies* the cost of the wrist-only constraint that an outdoor mobile setup might impose — turning
  a project constraint into a measurable result.
- **Open:** third-person camera mounting on a mobile base is unresolved — mast-vs-front-oblique.
  Full write-up of both positions, the rigid-common-baseplate design that may dissolve the
  disagreement, and the measurement protocol: **`docs/camera_mount.md`**.
  **To be settled by pixel-displacement measurement, not by argument.**

---

## D005 — Recovery demos are part of the data spec (7:3)

- **Date:** 2026-08-11
- **Decision:** Demo collection must include deliberate recovery demonstrations at roughly
  **normal : recovery = 7 : 3**.
- **Why (derived from the workshop failure analysis):** the GR00T policy drifted monotonically until it
  left the workspace, with the target already out of the wrist camera's view. Start-position
  randomization does *not* fix this, because everything it covers is still a state *on a successful
  trajectory*. Once the policy leaves that manifold, it is in states never seen in training — classic
  behavior-cloning covariate shift (the DAgger argument). The fix is on-policy corrective data.
- **Reverse if:** an ablation on public data (planned as experiment A3) shows recovery data has
  negligible effect on drift rate. **This is exactly why A3 is worth running before we spend real
  demo-collection time on it.**

---

## D006 — Remote inference is limited to the same machine or LAN

- **Date:** 2026-08-11
- **Decision:** Development, training, and data inspection may be remote. **Policy inference driving the
  physical arm must not run across the public internet.**
- **Why:** The workshop stack is a policy-server / robot-client architecture, which superficially
  suggests remote inference is fine. But VLA policies emit only 10–30 actions/sec; added network jitter
  produces command discontinuity, which the course material warns causes stuttering and
  "severe wear or damage to servos and reduction gears."
- **Note:** this reverses an earlier suggestion to run evaluations remotely to save travel time. The
  travel-time problem is real but must be solved another way (borrowing the arm, or co-locating the
  policy server with the robot on the lab LAN).

---

## D007 — No simulation. Real-hardware imitation learning instead.

- **Date:** 2026-08-11
- **Decision:** Drop the Isaac Lab / PPO / curriculum-learning / domain-randomization pipeline.
  Train ACT on real teleoperated demonstrations collected directly on the arm.
- **Alternatives:** Isaac Lab + RL (the original funded proposal); Gazebo/MuJoCo as a
  stopgap while hardware is unavailable
- **Why:**
  - **Last year's project failed precisely at sim-to-real transfer.** The privileged-information
    policy succeeded in simulation; the vision policy and every real-world deployment failed.
    Repeating the same approach with the same team and less time is not a plan, it's a habit.
  - Imitation learning on real data **eliminates the sim-to-real gap architecturally** rather than
    trying to close it with domain randomization.
  - SO-ARM has no vendor-grade simulation support anyway (unlike OMX, which ships Isaac USD models).
  - Feedback cycle: collect → train → evaluate is hours, not the multi-day RL training runs that
    made last year's 100+ experiments unattributable.
- **⚠️ Divergence from the funded proposal:** The v4 proposal commits to Isaac Lab, PPO, curriculum
  learning, and domain randomization throughout §3.3. **The advisor stated on 2026-07-17 that
  "手段不須跟原計畫符合" (the methods need not match the original proposal).** That sanction covers
  this decision. It does **not** automatically cover D008 below.
- **Accepted costs:**
  - No cheap data multiplication. Every demo costs human teleoperation time.
  - Loses the "large-scale parallel training" talking point from the proposal.
  - Domain randomization for outdoor robustness is no longer free — it has to come from
    physically varying the real setup.
- **Reverse if:** real-demo collection proves impossible to scale (e.g. hardware stays unavailable
  past ~4 weeks) **and** a usable SO-ARM simulation is confirmed to exist. Note that reversing
  after demo collection begins means discarding that data.

---

## D008 — 🟡 PROPOSED, NOT DECIDED: tactile sensing via servo current instead of Vision-to-Touch

> **Status: awaiting advisor approval. Raise at the 2026-08-14 meeting.**
> This is recorded here so the change is visible and deliberate, **not so it becomes true by default.**

- **Date proposed:** 2026-08-11 (proposed during AI-assisted planning, not yet team-ratified)
- **Proposal:** Replace the Vision-to-Touch module (Pix2Pix GAN generating GelSight-style tactile
  images, trained on TacEx-generated pairs) with a low-cost proxy: **gripper servo current and
  position error as an indirect contact-force signal.**
- **Why it was proposed:**
  - Requires no additional hardware purchase and no GAN training pipeline
  - Removes an entire model from the critical path, in a project that has ~8–16 total iteration
    cycles remaining
  - The Vision-to-Touch pipeline depended on Isaac Sim for training-pair generation, which D007
    removes
- **🔴 Why this is NOT equivalent to the previous three decisions:**
  **Vision-to-Touch is the proposal's stated core contribution.** It appears in the abstract, both
  research questions, the expected results, and the justification for both advisors' supervision.
  D007 changes *methods*, which the advisor has explicitly sanctioned. **D008 changes the claim.**
  - Proposal's claim: "under no tactile sensor, cross-modal generation can substitute for contact
    perception" — this has research novelty
  - Proposed substitute: "motor current indicates contact" — engineering-sound, but established
    technique, not a contribution
- **Questions the advisor must answer before this is adopted:**
  1. Is Vision-to-Touch still required as the deliverable contribution, or may it be descoped?
  2. If descoped, what replaces it as the project's claimed contribution?
     (Candidate: the recovery-data ablation in D005 — it is a real, testable, publishable claim
     that emerged from our own failure analysis.)
  3. If retained, where does the training-pair generation come from now that Isaac Sim is out?
- **Reverse if:** advisor requires Vision-to-Touch retained → D007 partially reverses too, since
  the tactile pairs need a simulator.

---

## D009 — Validate the recovery-data hypothesis on public datasets while hardware is blocked

- **Date:** 2026-08-11
- **Decision:** With both arms unavailable, do not idle and do not switch to simulation. Split a
  public SO-100/101 dataset into "smooth successes only" vs. "includes corrective segments",
  train an ACT on each, compare drift rate.
- **Alternatives:** wait for hardware; build a Gazebo environment; work only on the write-up
- **Why:** Zero hardware dependency, short feedback cycle, and it **tests our own D005 hypothesis
  before we spend scarce teleoperation time acting on it**. Also the single most defensible piece
  of work to show at the 08-14 meeting.
- **Accepted costs:** public datasets carry no recovery labels — the heuristic split (direction
  reversal, velocity discontinuity, path length vs. straight-line distance) is a methodological
  choice that must be stated explicitly, not glossed over.
- **Reverse if:** no public SO-100/101 dataset has a compatible `robot_type` and schema.

---

## D010 — conda on the GPU box, uv on the laptop; pin LeRobot exactly

- **Date:** 2026-08-11
- **Decision:** Different package managers on the two machines is acceptable. **The LeRobot version
  must match exactly.** Verify with a cross-machine loss comparison.
- **Alternatives:** force both onto conda; force both onto uv
- **Why:** The lab GPU box only supports conda; the laptop has no such constraint. The two machines
  have different roles (training vs. collection+deployment), so the genuine overlap is narrower than
  "identical environments" — it is LeRobot's dataset format and config schema. **The real risk is not
  the package manager, it is both machines installing `latest` at different times and diverging.**
- **Accepted costs:** two environments to maintain; version drift must be actively checked, not assumed.
- **Reverse if:** the cross-machine verification (same dataset, one step, compare loss) shows
  substantial disagreement that pinning cannot resolve.
- **See:** `docs/environment.md`

---

## D011 — Device naming: use LeRobot's discovery tools; udev binding is optional

- **Date:** 2026-08-11
- **Decision:** Default workflow is `lerobot-find-port` / `lerobot-find-cameras` at the start of each
  session. udev/by-id persistent binding is an **escalation path**, not initial setup.
- **Alternatives:** set up udev rules from day one
- **Why:** Discovery and persistence solve different problems and the persistence layer costs sudo
  access plus identifier lookup. For a two-person team on one machine, LeRobot's intended workflow
  is sufficient. **We were about to add a solution to a problem we have not experienced** — which is
  the same mistake as adding optimizations before having a baseline.
- **Reverse if:** a camera index swap silently corrupts a recording session, or find_port is being
  re-run more than ~3x/day. Escalation conditions are listed at the top of
  `scripts/setup_device_bindings.sh`.

---

## Template

```markdown
## D0NN — <one-line decision>

- **Date:**
- **Decision:**
- **Alternatives:**
- **Why:**
- **Accepted costs:**
- **Reverse if:**
```

> **Use `🟡 PROPOSED, NOT DECIDED` in the heading** for anything the team has not ratified or the
> advisor has not approved. A decision log that quietly promotes proposals into decisions is worse
> than no log at all.

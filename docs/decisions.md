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
  - ~~No vendor-grade simulation for SO-ARM.~~ **🔴 RETRACTED 2026-08-12 — this was simply false.**

    Isaac Sim ships official SO-ARM USD assets under the **RobotStudio** manufacturer:
    - `Robots/RobotStudio/so100/so100.usd`
    - `Robots/RobotStudio/so101_new_calib/so101_new_calib.usd`

    Present in both Isaac Sim 5.0 and 6.0 asset libraries. NVIDIA additionally maintains
    `isaac-sim/Sim-to-Real-SO-101-Workshop`, a first-party SO-101 sim-to-real course.

    **Correction history — recorded because the failure mode matters more than the fact:**
    the original claim was asserted without checking; a first "correction" then searched only
    lines 711–900 of the 983-line asset page and concluded absence from a truncated read.
    RobotStudio is at line 969. *Having the source open is not the same as having read it.*

    **Net effect:** this argument is removed from D007 entirely (not weakened — removed).
    D007's remaining reasons stand on their own. See D007.
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
- **🔴 Corrected 2026-08-12 — separate two things that were being conflated:**

  | | When it must be fixed | Why |
  |---|---|---|
  | **Camera *position*** (geometry relative to the workspace: height, angle, distance) | **P0, before the first demo is recorded** | It is a scene constant. Extrinsics are baked into every recording. Change it and prior data is invalid. |
  | **Camera *mount hardware*** (how that position is physically realized on a vehicle) | P3 | A tripod realizes the same geometry at P0 |

  The earlier phrasing said "camera decisions wait until P3", which was wrong — **recording requires
  the position frozen on day one.** Only the bracket does not.

  **Consequence, and this matters:** the P0 tripod position should be chosen to be *reproducible by a
  future vehicle mount*, not merely convenient to set up. Otherwise switching to the bracket at P3
  changes the extrinsics and **invalidates every demo collected before it.**

- **Open:** mast-vs-front-oblique for the eventual vehicle mount. Both positions, the
  rigid-common-baseplate design that may dissolve the disagreement, and the measurement protocol:
  **`docs/camera_mount.md`**. **To be settled by pixel-displacement measurement, not by argument.**

---

## D005 — Recovery demos are part of the data spec (7:3), two-tier method

- **Date:** 2026-08-11, **updated 2026-08-13** (method split into two tiers, ratified by team)
- **Decision:** Demo collection must include deliberate recovery demonstrations at roughly
  **normal : recovery = 7 : 3**. Two tiers, not one method:

  | Tier | Method | When | Status |
  |---|---|---|---|
  | **1 — default** | Teleoperator deliberately drives off-nominal, then demonstrates the way back, **as part of the initial demo batch**. No trained policy required. | P0, from day one | ✅ ratified 2026-08-11 |
  | **2 — conditional escalation** | RaC-style on-policy intervention: run the **already-trained** P0 policy, human takes over when failure looks imminent, rewinds to an in-distribution state, then corrects **to the end of the current sub-task**. Requires modifying the LeRobot inference script and keeping the leader arm powered and following the follower during autonomous execution, so a human can take over without a jarring handoff. | Only **if Tier 1 grasp performance is poor** — this is more complex to implement and is a fallback, not the default | ✅ ratified 2026-08-13, conditional |

- **Why (derived from the workshop failure analysis):** the GR00T policy drifted monotonically until it
  left the workspace, with the target already out of the wrist camera's view. Start-position
  randomization does *not* fix this, because everything it covers is still a state *on a successful
  trajectory*. Once the policy leaves that manifold, it is in states never seen in training — classic
  behavior-cloning covariate shift (the DAgger argument). The fix is on-policy corrective data.

- **🔴 2026-08-13 — RaC (arXiv 2509.07953) verified against the primary source, not against either
  teammate's recollection of it.** The 8/13 meeting note describing Tier 2 as "rewind and correct to
  the end of the sub-task" is **correct but incomplete** — it states Rule 1 and omits Rule 2. Exact
  wording from the paper:

  > *Rule 1 (recover then correct)* structures every human takeover into a reset back to
  > in-distribution states followed by a corrective segment that completes the current sub-task.
  > *Rule 2 (termination after intervention)* ends the episode immediately once the intervention
  > segment finishes, which avoids collecting data on later sub-tasks under state distributions from
  > a mixture of learned policy and human expert.

  **Consequence for the inference-script modification Tier 2 requires:** the script must stop the
  episode the moment the corrective segment ends — it must **not** hand control back to the model, and
  the human must **not** keep driving through the rest of the task. Either of those would (per the
  paper's own stated reason) collect data under a mixed policy/human state distribution, which is the
  exact thing Rule 2 exists to prevent. This needs to be built into the script from the start, not
  patched in after noticing bad data.

  **A second finding from the same source, relevant to Tier 1 as well:** RaC reports that recovery
  segments are more useful when they are **deliberately suboptimal** — the correction is allowed to be
  inefficient, even to undo prior progress on the sub-task, because the point is demonstrating *how to
  get back to a familiar state*, not solving the task elegantly. Quoting the paper's own framing, this
  "challenges the conventional wisdom that only 'expert' interventions are useful." **This applies to
  Tier 1 demos too:** when filming a recovery demo, do not clean it up. The reflex to make the
  correction look competent works against the thing recovery data is supposed to teach.

- **Reverse if:** an ablation on public data (planned as experiment A3) shows recovery data has
  negligible effect on drift rate. **This is exactly why A3 is worth running before we spend real
  demo-collection time on it.** Tier 2 specifically reverses if Tier 1 alone yields acceptable grasp
  performance — it is not owed a trial just because it is now specified.

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

- **Date:** **2026-07-17** (advisor meeting where the pivot was approved)
  — concretized 2026-08-11 as SO-ARM + LeRobot + ACT
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
  - ~~SO-ARM has no vendor-grade simulation support anyway.~~ **🔴 Removed 2026-08-12 — false.**
    Isaac Sim ships SO-100 and SO-101 USD assets under RobotStudio (see D002). This argument is
    withdrawn; the reasons above and below stand without it.
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
- **Reverse if:** real-demo collection proves impossible to scale — e.g. hardware stays unavailable
  past ~4 weeks. **This reversal is now more actionable than originally written**: official Isaac Sim
  SO-100/SO-101 assets exist (D002), so falling back to simulation would not start from zero.
  Note that reversing *after* demo collection begins means discarding that data.

---

## D008 — 🟡 OPEN: what replaces Vision-to-Touch as the tactile modality?

> **Status: undecided. Three or four candidate paths on the table, none ratified.**
> Recorded here so the divergence is visible and deliberate, **not so it becomes true by default.**
> **Raise at the 2026-08-14 advisor meeting.**

### The situation

The funded proposal's stated core contribution is **Vision-to-Touch**: a Pix2Pix GAN generating
GelSight-style tactile images, with training pairs synthesized via TacEx inside Isaac Sim. It appears
in the abstract, both research questions, the expected results, and the justification for both
advisors' supervision.

**D007 removed Isaac Sim from the plan — which removes the source of the tactile training pairs.**
So the tactile modality now needs a different answer, and there is more than one candidate.

### Candidates

| # | Path | Hardware cost | Preserves the proposal's claim? | Status |
|---|---|---|---|---|
| **V1** | Vision-to-Touch GAN, per the proposal | none | ✅ Yes — cross-modal generation is the novelty | ⚠️ **blocked**: training pairs required Isaac Sim (D007) |
| **V2** | **Vision first; if it succeeds, purchase a simple tactile sensor as a real multimodal input** | 🟡 consumables budget (advisor said this is available) | ⚠️ Partially — *multimodal grasping of non-rigid objects* survives; *cross-modal generation* does not | **Advisor's verbal suggestion, 2026-07-17** |
| **V3** | Gripper servo current / position error as a contact-force proxy | none | ❌ No — sound engineering, but established technique | Proposed during AI-assisted planning 2026-08-11; **not team-ratified** |
| **V4** | Descope tactile entirely; make the recovery-data ablation (D005) the contribution | none | ❌ No, but substitutes a different real claim | Not yet discussed |

### 🔴 Why this needs the advisor, not just us

D007 changes **methods** — the advisor explicitly sanctioned that on 2026-07-17
("手段不須跟原計畫符合"). **D008 changes the claim**, which is a different category:

- Proposal's claim: *"absent a tactile sensor, cross-modal generation can substitute for contact perception"* — has research novelty
- V2's claim: *"a cheap tactile sensor improves non-rigid grasping"* — true, useful, but not novel
- V3's claim: *"motor current indicates contact"* — established technique

### ⚠️ A real risk to manage carefully

**We are not certain the advisor recalls the proposal's specifics.** V2 was offered verbally in a
planning conversation, not while reviewing the written proposal. Advisors routinely do not hold the
details of a student proposal in working memory — this is normal and not a criticism.

**Therefore: bring the printed proposal §1.2, §2.2, §3.3 and §5 to the 08-14 meeting** and make the
divergence explicit before asking for a decision. A descope agreed to without both parties seeing
what is being descoped is not a decision — it is a future dispute.

### Questions for the advisor

1. Is Vision-to-Touch still required as the deliverable contribution, or may it be descoped?
2. If V2 (buy a sensor): which sensor, what lead time, and does it fit the consumables budget?
   Does "vision succeeds first" mean P0, P1, or P2 complete?
3. If descoped: what becomes the claimed contribution?
   (Candidate: **D005's recovery-data ablation** — a real, testable claim that emerged from our own
   failure analysis, and one we can run without hardware.)
4. Does 陳弘軒老師 need to be part of this decision? The proposal §6.2 lists his supervision
   partly in terms of the cross-modal model.

- **Reverse-if / decision trigger:** whichever path the advisor selects on 08-14 becomes a decision
  entry; this one is then closed and cross-referenced.
- ⚠️ If **V1 is retained**, D007 partially reverses — tactile training pairs need a simulator.

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

## D010 — Docker on the GPU box, uv on the laptop; pin LeRobot exactly

- **Date:** 2026-08-11, **updated 2026-08-12** (GPU box uses Docker, not conda)
- **Decision:** GPU training box runs LeRobot in **Docker**; the laptop uses **uv**.
  Different toolchains are acceptable. **The LeRobot version must match exactly.**
  Verify with a cross-machine loss comparison.
- **Alternatives:** force both onto conda; force both onto uv; run Docker on the laptop too
- **Why:**
  - **Docker on the GPU box is an upgrade over conda, not a compromise.** It pins the OS libraries
    and CUDA runtime as well as the Python packages — the strongest reproducibility of the three
    options. LeRobot also ships official Docker files, which removes most CUDA-configuration pain.
  - The laptop's role is data collection + inference deployment, where it must talk to USB serial
    devices and cameras. **Docker adds device-passthrough complexity for no benefit at that end.**
  - The genuine overlap between the two machines is narrower than "identical environments" — it is
    **LeRobot's dataset format and config schema**. **The real risk is not the toolchain, it is both
    machines installing `latest` at different times and diverging.**
- **Accepted costs:**
  - Two toolchains to maintain
  - ⚠️ **Docker containers are ephemeral.** Datasets, checkpoints, and calibration files must be on
    **mounted volumes**, or they vanish when the container stops. This is the single most common way
    to lose a training run.
  - The LeRobot version inside the image must be recorded explicitly (image tag alone is not enough
    if the image is rebuilt)
- **Reverse if:** the cross-machine verification (same dataset, one step, compare loss) shows
  substantial disagreement that pinning cannot resolve.
- **See:** `docs/environment.md`
- **⚠️ Superseded (GPU box) by D012, 2026-08-12** — conda is no longer used for training. The
  laptop half and the cross-machine verification requirement still stand.

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

## D012 — Docker on the GPU box, replacing conda

- **Date:** 2026-08-12
- **Ratified:** 2026-08-12. Proposed and accepted the same day, after the stack was validated
  end-to-end (image runs, 100-step ACT training completes — see `README.md` § Quickstart).
- **Decision:** The GPU box's training environment is a **container**, not a conda env. For now that
  is LeRobot's official `huggingface/lerobot-gpu` image pinned by digest; our own `Dockerfile` is the
  intended end state. **Supersedes D010 for the GPU box only** — the laptop half of D010 is *not*
  resolved here, see **Open** below.
- **Alternatives:** keep conda (D010); build our own image from a CUDA base immediately instead of
  adopting the upstream one first
- **Why:**
  - The environment becomes a **repo artifact** (Dockerfile + digest) instead of a sequence of
    install steps a human re-performs — the same reproducibility argument the README already makes
    for everything else.
  - **No `sudo` on the GPU box.** A container needs none: the host already has
    `nvidia-container-toolkit 1.17.8` and the `nvidia` runtime registered.
  - D010's stated core risk — *"both machines installing `latest` at different times and diverging"* —
    is pinnable more sharply by image digest than by a conda export.
- **Accepted costs:**
  - **The image wants CUDA 12.8; the host driver caps at 12.4.** It runs today only via a documented
    workaround (`NVIDIA_DISABLE_REQUIRE=1` plus masking the bundled forward-compat libraries, which
    are unsupported on GeForce cards). This leans on CUDA minor-version compatibility and **can fail
    mid-run**, not at startup, if a call genuinely needs driver ≥570. Full detail in
    `docs/environment.md`.
  - ~12.3 GB image versus a conda env.
  - The cross-machine verification D010 mandates is now **blocked** until the laptop side is decided.
- **Note on process:** D010's own reversal condition — cross-machine loss disagreement — was **never
  met**; that check has never been run. This switch is motivated by reproducibility and the sudo
  constraint, *not* by evidence against D010. Recorded plainly so that nobody later concludes D010
  failed on its merits.
- **Open:** does the laptop (data collection + inference) also run the container, or stay on uv?
  Until this is answered the D010 verification protocol cannot execute at all.
- **Reverse if:** the forward-compat workaround proves unstable in real training runs **and** neither
  the driver upgrade (needs the lab admin) nor a cu124 image of our own materialises — conda per D010
  remains the fallback.
- **See:** `docs/environment.md`, `README.md` § Container image

---

## D013 — LeRobot CLI for both arms; skip the ROBOTIS ROS 2 toolchain

- **Date:** 2026-08-13
- **Decision:** Whichever arm we end up with, drive it through the **LeRobot CLI**. Do not adopt
  ROBOTIS's ROS 2 imitation-learning stack.
- **Alternatives:**
  - **Cyclo Intelligence** — ROBOTIS's *currently supported* ROS 2 workflow (containers, web UI via
    noVNC). Verified 2026-08-13 as the actively maintained option.
  - **Physical AI Tools** — a ROS 2 wrapper over LeRobot with a web GUI. ⛔ **Legacy.** ROBOTIS's own
    docs: *"Physical AI Tools is kept for users who still need the previous workflow, but it is no
    longer updated."* Moved under `resources/legacy/`. Do not build on it.
- **Why:**
  - **One toolchain.** We already run LeRobot for SO-ARM; using it for OMX too means the environment,
    scripts, dataset format, and eval harness transfer unchanged.
  - **De-risks the platform decision.** If the lab's arm turns out to be OMX rather than SO-ARM,
    almost none of our preparation is wasted.
  - Removes any ROS 2 / Ubuntu 24.04 requirement from the critical path.
  - The GUI's value is low for us — we want explicit control over the pipeline anyway, and a GUI
    makes runs harder to script and reproduce.
- **Accepted costs:**
  - Forgo ROBOTIS's official support channel and their GUI-driven workflow
  - If LeRobot's OMX support lags Cyclo's, we absorb that gap ourselves
- **Reverse if:** LeRobot's OMX support proves materially behind Cyclo Intelligence (missing features
  we actually need), **or** we hit an OMX-specific bug that Cyclo handles and LeRobot does not.

---

## D014 — Windows laptop is acceptable for recording and evaluation; do not request a Linux machine

- **Date:** 2026-08-13
- **Decision:** Use the existing Windows laptop. **Do not ask the lab to provide a Linux machine**,
  and do not dual-boot.
- **Alternatives:** request a Linux box as the shared real-robot host; dual-boot Ubuntu on the laptop;
  WSL2 + usbipd; VirtualBox
- **Why — the Linux case collapsed under verification:**

  | Concern raised | Status after checking (2026-08-13) |
  |---|---|
  | Serial port paths break SO-101 calibration on Windows (#1094) | ❌ Issue **closed**, 2025-05, old version. Ports are now parameterized via `--robot.port` |
  | OpenCV camera backend fails on Windows (#1368) | ❌ Current `camera_opencv.py` already sets `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` on Windows |
  | Teleoperate failure, 2026-03 (#3234) | ❌ **Reported on Ubuntu too** — a LeRobot bug, not a platform one |
  | OMX needs ROS 2 / Ubuntu 24.04 | ❌ Sidestepped by **D013** (LeRobot CLI, no ROS 2) |
  | **#4093: `uv sync` installs CPU-only PyTorch on Windows + NVIDIA (2026-07)** | ✅ **Real and current** — but has a workaround, and only affects inference deployment |

  What remained was one workaround-able issue plus an unfalsifiable "fewer variables" preference.
  **Not enough to spend the lab's goodwill, or a weekend on dual-boot.**

- **Accepted costs:**
  - Windows sees less community testing for LeRobot hardware workflows → residual unknown-unknowns
  - WSL2 and VirtualBox are explicitly **not** fallbacks: USB camera passthrough is unreliable in both,
    and VirtualBox additionally has no GPU passthrough
- **🔴 Required precaution:** after installing on the laptop, **immediately verify**
  `torch.cuda.is_available()`. If it returns `False`, that is #4093 — force-reinstall torch from the
  CUDA index rather than debugging anything else first.
- **Reverse if:** a Windows-specific problem actually blocks calibration, recording, or evaluation for
  more than ~half a day. **The trigger is a real blockage, not a new GitHub issue.**

---

## D015 — Failure taxonomy: two orthogonal axes (`outcome` + `mechanism`)

- **Date:** proposed 2026-08-12, **ratified 2026-08-13**
- **Owner:** 陳霆翰 (Eric Chen)
- **Decision:** Replace the single F1–F7 failure-code list with two independent columns:
  `outcome` (mutually exclusive, exhaustive: `success` / `no_grasp` / `dropped` / `misplaced`) and
  `mechanism` (multi-label, may be empty: `pushed_away` / `collision` / `drift` / `stalled` /
  `timeout` / `self_recovered` / `other`), plus `valid` / `void_reason` to exclude invalid trials.
- **Why:** F1–F7 mixed two questions in one list. `pushed_away` always implies `no_grasp`; `stalled`
  always ends in `timeout` — annotators hesitated, two people labelled the same trial differently, and
  a failure-distribution that isn't reproducible between labellers is the only metric this project has
  for deciding what to fix next. See `docs/meeting/2026-08-13.md` §二-1 for the full worked example.
- **Note:** `self_recovered` is a positive `mechanism` label, valid alongside `outcome=success`. It is
  the single most direct real-hardware evidence for D005's hypothesis — if recovery data works, this
  label's frequency should rise.
- **Implemented in:** `docs/experiment_spec.md` §1-3, `eval/README.md`, `eval/_template.csv`.
- **Reverse if:** the two-axis scheme itself produces annotator disagreement in practice — check this
  the first time two people independently label the same real eval run.

---

## D016 — Evaluation protocol: 30 trials per object

- **Date:** ratified 2026-08-13
- **Owner:** 陳柏宇 (Boyu Chen)
- **Decision:** Each object gets **30 evaluation trials** (supersedes the earlier "≥ 20, suggested"
  figure in `experiment_spec.md` §5).
- **Why:** not recorded in the meeting note beyond the number itself — if a rationale (e.g. matching
  the 3×3 start-position grid, or a power calculation) surfaces later, add it here rather than letting
  the number float unexplained.
- **⚠️ Open reconciliation:** `experiment_spec.md` §5 also specifies a 3×3 start-position grid with
  "≥ 2 per cell" (≥ 18 total). 30 does not divide evenly across 9 cells (30/9 ≈ 3.33). Needs a decision
  on the actual per-cell distribution (e.g. 3 cells get 4, six get 3) before the first real eval run —
  flagged, not yet resolved.
- **Reverse if:** GPU/session time makes 30×(number of objects) impractical per eval round.

---

## D017 — Configuration: YAML declarative configs with dated calibration IDs

- **Date:** 2026-08-14
- **Decision:** Use YAML configuration files under `configs/` passed via `--config_path` for all LeRobot workflows (`calibrate`, `teleoperate`, `record`, `replay`). Explicitly set `calibration_dir: ./calibration` and use dated device IDs (e.g. `id: 2026-08-14_leader`, `id: 2026-08-14_follower`).
- **Alternatives:**
  1. Long CLI arguments with shell environment variables (fragile across Windows PowerShell sessions).
  2. Fixed generic device IDs (`leader_black`) with manual post-calibration file copying/renaming.
  3. Default cache paths (`~/.cache/huggingface/lerobot/calibration/`), which fail when `HF_HOME` points to unmounted drives (e.g. D:\).
- **Why:**
  - **Zero-overwrite & Auditability:** LeRobot derives calibration filenames from `<id>.json`. Naming the ID with `<date>_<role>` automatically generates `2026-08-14_follower.json` with zero manual intervention, adhering strictly to `conventions.md` ("dated filename, never overwrite").
  - **Data Provenance:** Datasets recorded via `lerobot-record` embed the exact `teleop.id` and `robot.id` in metadata, creating an unbreakable link between dataset runs and physical calibration files.
  - **Self-contained Workspace:** Isolates calibration files to `./calibration/` within the repository, eliminating dependencies on external drives or global user caches.
- **Accepted costs:** When performing data collection across multiple dates, the `id` field in `configs/*.yaml` must be updated with that day's date string.
- **Implemented in:** `configs/*.yaml`, `docs/field_manual.md`.
- **Reverse if:** LeRobot deprecates `--config_path` or changes its calibration naming schema.

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

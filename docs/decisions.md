# Decision log

Append-only. One entry per decision that would be expensive to reverse, or that a future reader
(including us in three months, or the advisor) would otherwise have to re-derive.

**Format:** what was decided, when, what the alternatives were, why, and — most importantly — **what
evidence would make us reverse it.** A decision without a reversal condition is a belief, not a decision.

---

## Status index (2026-08-26)

**Read this before citing any single entry.** An append-only log is only trustworthy if the reader can
tell at a glance which entries still govern current state.

| Status | Entries |
|---|---|
| 🔴 **Superseded — do not read as current state** | **D002** (platform: SO-ARM) → superseded by **D021** |
| 🟡 **Open / proposed — not decided** | **D019** (action representation), **D020** (mobile base & XLeRobot — *its 2026-08-25 trigger lapsed; the meeting did not happen*), — |
| ✅ **Resolved 2026-08-27** | **D025** → do it, but only after Phase B real data exists (complement to D007, not a reversal). **D021** → 甲: OMX to the end, SO-ARM is a spare. **D022** single-camera verified + 3-config recording plan; **2026-09-01 `[Eric決定]`: D405 is the interim wrist camera until the UVC module arrives OR Phase C is reached** — interim config = D405 wrist + D455 third-person; D405→UVC swap and Phase C are both re-record boundaries. **D024** → 60 per campaign, position-OOD cancelled, training positions seeded, closed-loop 30 is in-distribution, uniform sampling replaces the 3×3 grid. |
| 🔴 **D023 — status changed 2026-08-31** | Cable resolved **by RE-ROUTING the existing cable, not replacement** (`[Eric說]`; lab had no spare). **The 2026-08-27 conservative-workspace exemption is VOID** (a re-route is not a monotone relaxation); A7's original gate is back. **Tape measurement (FK failed → D026):** `r_outer` top-down ≈ **41 cm**, side-only ≈ 49, `r_inner` ≈ **22** (all + `d_offset` 5 cm, pan axis → chassis edge). Azimuth sector ≈ **135°** (`theta ∈ [−90°, +45°]`), edge = **arm body physically hits the third-person camera mount** if rotated past — a hard mechanical limit, not FOV, not the cable. **Scope: Phase-A pilot layout only; Phase B on the vehicle re-runs S1/S2 from scratch** (`[Eric說]`). Next: S2 `--dry-run` feasibility. See D023 §2026-08-31 points 5–6. **2026-08-31 (earlier):** the 33–43 cm figure disambiguated (grasp-approach band); `r_max` verdict logic dropped. |
| ✅ **Resolved 2026-08-31** | **D026** → reach logger measures by FK from the `omx_f` URDF (placo, LeRobot-native); tape measure is the fallback. `placo` enters the pinned env. |
| ⚪ **Descoped, not cancelled** | **D008** (tactile → Phase D, "if time allows") |
| ✅ **In force** | D001, D003–D007, D009–D018, D021–D023 |
| 🔴 **Cancelled** | **D009** — recovery-hypothesis experiment on public data. **Cancelled 2026-08-27**: only 1 of 50 public episodes contained a corrective motion, so the comparison arm cannot be populated. **D005 stands but is now an untested design choice.** |

**Current platform: OMX-AI (D021). Current wrist-camera plan: USB UVC module, ETA 2026-09-05–07,
with two third-person cameras as the interim configuration (D022).
Current hard block: the short arm cable gates A6 and all recording (A8+). A7 placement *design* is unblocked under D023's conservative-workspace exemption.**

Anything in this repo written before 2026-08-24 that names SO-ARM as the platform predates D021.

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

## D002 — ~~Platform: SO-ARM (SO-100/SO-101)~~ 🔴 SUPERSEDED by D021 (2026-08-24)

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
- **⚠️ 2026-08-16 補記——OMX 的使用不等於平台決策改變。**
  SO-ARM 實體尚未到貨，因此先以實驗室的 **OMX 手臂跑通 Phase A 流程**（環境、校正、遙操作、錄製、重播）。
  **這是管線驗證，不是平台選型的變更。** D002 仍然成立：SO-ARM 是主力平台，OMX 上**不收正式資料**。
  之所以特別寫下來，是因為未來讀者（含我們自己、老師）看到「用了 OMX」很容易誤判成平台已改。
  **判準：在 OMX 上產生的任何 dataset，一律標記為 pipeline-validation，不得併入訓練集**（同 `docs/pipeline_validation.md` 的原則）。

- **Reverse if:** SO-ARM stays unavailable beyond ~3 weeks while OMX becomes free, *and* no data has
  been collected yet. **Once demo collection starts, this is effectively irreversible** — imitation
  data does not transfer across arms.

### 🔴 SUPERSEDED 2026-08-24 — see **D021**

  **The reversal condition above fired exactly as written.** SO-ARM ETA was confirmed as 2026-09-15
  (4 weeks, not 1–2), OMX is repaired and bookable, and no demo data had been collected.
  D002's platform choice no longer governs Phase A–B. **Do not read D002 as current state.**

  The 2026-08-16 補記 above (「OMX 的使用不等於平台決策改變」) is **also superseded** — as of D021 the
  OMX **is** the platform, and data collected on it **is** real Phase B data, no longer
  pipeline-validation-only.

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

- **🔴 2026-08-24 — interim configuration, see D022.** The wrist camera is not available until
  2026-09-05–07. Early trials run with **two third-person cameras**, or optionally
  single-third-person as a declared arm of this ablation. **Camera count is a scene constant:**
  demos from different camera configurations **must not be pooled**. Interim recordings are
  **pilot data** unless the configuration is declared and frozen in `experiment_spec.md` first.

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

## D008 — Tactile modality descoped to Phase D; Outdoor vision generalization prioritized (Phase C)

- **Date:** 2026-08-11, **ratified & closed 2026-08-18**
- **Decision:** Prioritize outdoor vision generalization (Phase C) over tactile sensing (Phase D). The primary contribution is a robust imitation-learning pipeline that overcomes outdoor visual noise (lighting, dynamic background, shadows) under covariate-shift mitigation. Tactile integration (V2: simple contact/force sensor) is retained only as an elective Phase D if time permits.
- **Alternatives:** V1 (Vision-to-Touch GAN via simulation), V2 (hardware sensor upfront), V3 (motor current proxy).
- **Why:** At the 2026-08-18 advisor meeting, Prof. Jahui Chang explicitly ruled: *"Outdoor environment comes first. Solve the lighting/background interference first; that is the primary failure mode for vision policies."* This formalizes the descope of the original simulation GAN while maintaining a clean, feasible progression (Phase A → B → C >> D).
- **Accepted costs:** The project no longer claims "cross-modal vision-to-touch generation" as a core thesis. The claim shifts to empirical robust manipulation under open-world environmental shifts.
- **Cross-reference:** `docs/meeting/2026-08-18.md` §三-5, §四-3.

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

### 🔴 CANCELLED 2026-08-27 `[Eric決定]` — the reversal condition effectively fired

**Not executed, and now formally dropped.** Reason given: on inspecting a candidate public dataset,
**out of 50 episodes only 1 contained a corrective ("recovery") motion.**

**Why that kills the experiment rather than merely making it harder:** the whole design was to split a
public dataset into "smooth successes" vs. "contains corrective segments" and train one policy on each.
**A 1/50 base rate means the second group cannot be populated** — there is no corrective-behaviour
arm to train. The heuristic split (direction reversal, velocity discontinuity, path length) was going
to *find* those segments; it cannot create them.

**What this does NOT invalidate:** **D005 stands** (recovery demos are part of our own data spec at
7:3, escalating to RaC-style intervention if needed). We simply never got external evidence for it
first, so **D005's ratio is now an untested design choice, not a validated one.**
🔴 **That must be stated honestly in the write-up: we designed the recovery-data strategy, we did not
validate it.**

**Also worth recording as a finding in its own right:** *public teleoperation datasets are
overwhelmingly clean successes.* That is itself a reason our own collection must deliberately include
failures — which is exactly what D005 does. **The absence of the data is an argument for D005.**

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
  **`repetition_loop`** / `timeout` / `self_recovered` / `other`), plus `valid` / `void_reason` to
  exclude invalid trials.
- **Why:** F1–F7 mixed two questions in one list. `pushed_away` always implies `no_grasp`; `stalled`
  always ends in `timeout` — annotators hesitated, two people labelled the same trial differently, and
  a failure-distribution that isn't reproducible between labellers is the only metric this project has
  for deciding what to fix next. See `docs/meeting/2026-08-13.md` §二-1 for the full worked example.
- **Note:** `self_recovered` is a positive `mechanism` label, valid alongside `outcome=success`. It is
  the single most direct real-hardware evidence for D005's hypothesis — if recovery data works, this
  label's frequency should rise.
### 🔴 Amendment 2026-08-16 `[團隊決議]` — `repetition_loop` added to the `mechanism` vocabulary

Ratified at the 2026-08-16 peer meeting (`docs/meeting/2026-08-16.md` §二-2, item 1 of §三).
**Recorded in this entry on 2026-08-30 — it had been applied to the spec and the eval README for two
weeks while the decision log, which is the 正本, still listed the pre-amendment vocabulary.**

- **Definition:** the policy oscillates over, or repeatedly re-runs, the same trajectory interval
  without progress.
- **Why it is not covered by an existing label:** `stalled` means *never started or stopped in
  place* — no motion. `drift` means motion in **one** direction, out of the workspace.
  A repetition loop is motion, bounded, and going nowhere. Collapsing it into either loses the
  distinction that tells you whether the policy has no signal or the wrong signal.
- **Why it earns a label rather than `other` + a note:** `docs/act_evidence.md` §A-4 records it at
  **92.45%** of surveyed ACT failures — third highest. A mechanism that common inside `other` makes
  the failure distribution useless for deciding what to fix next, which is D015's whole purpose.
- **Vocabulary consumers that must stay in sync** (all verified aligned 2026-08-30):
  `docs/experiment_spec.md` §1-3, `eval/README.md`, `eval/_template.csv`, `episode_meta/README.md`,
  `configs/episode_meta_schema.yaml` (`strict: true` — a value missing here is *rejected at the
  prompt*, not merely undocumented).

- **Implemented in:** `docs/experiment_spec.md` §1-3, `eval/README.md`, `eval/_template.csv`,
  `episode_meta/README.md`, `configs/episode_meta_schema.yaml`.
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

## D018 — No co-training with public datasets that have different scene constants

- **Date:** 2026-08-15
- **Decision:** Do **not** mix public SO-100/101 datasets into P0 baseline training. Public datasets
  remain in use for the **offline A3 ablation only**, where scene consistency is not a controlled variable.
- **Alternatives:** mix public data in as augmentation (the intuitive move — "more data is better");
  pretrain-then-finetune; add camera-extrinsics conditioning so scene differences stop mattering
- **Why — the mechanism, not just an outcome:** TTIC/TRI's study (which explicitly tests **ACT**,
  Diffusion Policy and SmolVLA) found:

  > *"policies without extrinsics often infer camera pose using visual cues from static backgrounds in
  > fixed scenes. This shortcut collapses when workspace geometry or camera placement shifts."*

  **ACT without extrinsics conditioning uses the static background as an implicit camera-pose
  reference.** Mixing in data from a different scene feeds the model contradictory
  background→pose mappings. Evidence of how strong this effect is: in their real-robot experiments
  they had to **cover the background with green cloth** to remove the shortcut.

  Supporting evidence:
  - **Open X-Embodiment:** pooling datasets whose action distributions disagree causes a *small* model
    to average them toward mush — **negative transfer**. High-capacity models (RT-2-X class) are needed
    to absorb heterogeneous data. **ACT is ~80M parameters.**
  - Large-scale co-training studies report positive scaling **only when aligned data is included**,
    where "aligned" explicitly means shared task semantics *and scene context*.
  - The systematic co-training study showing clear benefits tests **LBMs (VLM-backbone)**, not ACT-scale models.
- **⚠️ What this corrects:** the NVIDIA SO-101 course's "Strategy 2: Co-Training With Real Data" pairs
  **sim and real of the same scene** — that is aligned data. It does not license mixing in public
  datasets recorded in different rooms with different camera poses.
- **Accepted costs:** we forgo a cheap-looking source of extra data. Given our own dataset will be small
  (~50–65 episodes), that is a real cost, not a free choice.
- **Reverse if:** we add camera-extrinsics conditioning (see D019's cross-reference), **or** we run an
  explicit controlled comparison (with/without public data, same everything else) and it comes out positive.
  **Not reversible by intuition — only by that experiment.**

---

## D019 — 🟡 OPEN: action representation (absolute joint position vs. delta end-effector pose)

> **Status: NOT decided. Raised 2026-08-15, needs team ratification.**
> **`experiment_spec.md` §4 currently says absolute joint position and has NOT been changed by this entry.**

- **The finding:** the same TTIC/TRI study ablated four action spaces:

  > *"the policy performs best when actions take the form of **Delta End-Effector Pose** and that the
  > performance is **noticeably worse for Absolute Joint Position** and Absolute End-Effector Pose."*
  > Reason given: *"Absolute actions depend heavily on the accuracy of the camera-to-action-frame
  > transformation, while relative actions are more tolerant."*

- **⚠️ Scope limit — do not overstate this:** that study measures robustness **under camera viewpoint
  variation**. With a genuinely fixed camera the disadvantage of absolute joint position is much smaller,
  and **absolute joint position is LeRobot's default for SO-100**, which means it is the better-trodden path.
- **Why it still matters to us:** L4 puts the camera on a vehicle. **Once the camera moves, this becomes
  directly relevant.** And per D004, the P0 camera position is chosen to be reproducible by that future mount.
- **Options:**
  1. Keep absolute joint position (LeRobot default, least friction, matches existing spec)
  2. Switch to delta end-effector pose (better under viewpoint change, more deviation from defaults)
  3. Keep absolute for P0, revisit only if L4 actually happens
- **Cost of deciding late:** action representation is baked into the recorded dataset. Changing it after
  C1 means **re-deriving or re-recording**. → **must be settled before C1.**
- **Owner / when:** team decision, 2026-08-16 meeting.

---

## D020 — 🟡 PROPOSED, NOT DECIDED: Mobile Base Selection & Hardware Integration Reference (XLeRobot Subsystems)

- **Date:** Proposed 2026-08-18 (following advisor's group post & preliminary analysis)
- **Status:** **PROPOSED, NOT DECIDED.** To be placed on the agenda for team discussion & advisor ratification at the next meeting (2026-08-25).
- **Proposal:** 
  1. **Stationary / Tabletop Baseline First (P0–P2)**: Proceed with stationary tabletop pick-and-place experiments using heavy-duty table clamps (桌夾) for quick, rigid mounting.
  2. **Mobile Base Height Constraint for Ground Picking (P3/P4)**: Reject tall utility carts (e.g. 77cm IKEA RÅSKOG from XLeRobot) as mobile bases for highway trash collection because SO-101's arm stroke (~30cm reach) cannot reach ground level from a tall cart. Propose using a low-profile chassis (such as the lab's low-slung JetRover「小綠」/「小藍」/ AMR lower plate, or an underslung forward mount).
  3. **Borrow Viable XLeRobot Subsystems**: Propose adopting XLeRobot's proven component ecosystem:
     - **Sensors**: Multi-camera architecture with Intel RealSense D415 / Logitech C920 (3rd-person) + USB wrist camera, using 3D-printable brackets.
     - **Power & Wiring**: Anker SOLIX C300 (or equivalent portable power station) + Type-C to DC 12V PD trigger cables (5264 connectors) for outdoor servo power.
     - **Fastening**: Non-destructive table clamps for baseplate/rigidity coupling.
- **Alternatives:** 
  - Buy a complete high utility cart mobile robot (incompatible with ground trash picking).
  - Design an entirely custom mobile chassis and power delivery system from scratch (high engineering overhead).
- **Why:** Solves the camera mounting, outdoor mobile power, and wiring challenges while respecting the physical workspace envelope required for roadside debris picking.
- **Accepted costs:** Requires designing/modifying a lower bracket to mount the arm low enough to contact the floor, rather than dropping in the standard IKEA cart top tray.
- **Next steps / Decision trigger:** ~~Review at 2026-08-25 meeting following 8/25 NCKU site visit and team consensus.~~
  **🔴 2026-08-26 update: the 2026-08-25 lab meeting did not take place** — the team attended the NCKU
  gripper-vehicle workshop that day instead. **D020 therefore remains PROPOSED, NOT DECIDED**, and has
  no new decision trigger. **Re-schedule it onto the next advisor meeting's agenda explicitly**; a
  proposal with a lapsed trigger is how a decision log silently rots.
  Note also that D021 (OMX, not SO-ARM) changes this proposal's premise: the arm-stroke and
  mounting-height numbers in §2 were derived for **SO-101** and **must be re-derived for OMX**
  before this is ratified.

### 🔴 2026-08-27 `[Eric問]`：實驗室既有自走車有 URDF，後續如何評估？

**七條評估判準，依「便宜先做」排序：**

| # | 判準 | 怎麼量 | 為什麼會擋住整個案子 |
|---|---|---|---|
| **1** | **高度 → 可達地面** | **車體 URDF ＋ OMX URDF 組合後量可達範圍** | D020 §2 已用這條否決 77 cm 推車。⚠️ **原數字是用 SO-101 算的，OMX 行程不同（D021），必須重算** |
| **2** | **承載** | 手臂＋相機＋電源＋線材的總重 vs 車體 payload | 超載就不用談 |
| **3** | 🟡 **平台剛性 / 晃動**（**2026-08-27 降級**） | 第一次上車時量一次手臂全速動作下的車體位移 | ⚠️ **我原本把這條寫成「最致命」，那是誇大的。** `[Eric說]`：**車移動到定點停下之後才開始撿拾**，錄製期間車是靜止的 → 外參不會在「移動中」漂移。**殘留的只有手臂反作用力造成的車體彈性晃動**，若輪子有煞車／鎖定則很小；且**只要訓練與評估時晃動一致，它就是環境的一部分，不是變因**。→ **量一次確認量級即可，不是設計約束** |
| **4** | **IMU** | 有沒有 | 梓逸學長 8/18 已指出：缺 IMU → SLAM 建圖行走會漂移（`meeting/2026-08-18.md` §三-2） |
| **5** | **供電** | 能否供 12V 給 STS3215／DYNAMIXEL | 見 `hardware.md` Power 段 |
| **6** | 🟡 **ROS 2 介面**（**2026-08-27 大幅修正**） | 上車前先在桌面用 ROS 2 節點跑一次推論，比對成功率有沒有掉 | ⚠️ **我原本寫成「與 D013 衝突」，那是把 D013 的範圍讀寬了。** `[Eric說]` 指出三點，都成立：① 手臂 URDF 加進去就解決幾何/TF ② **LeRobot 訓出來的是一個 PyTorch policy，完全可以包成 ROS 2 node**（載 checkpoint、訂閱影像/關節 topic、發布 action）③ **D013 反對的是「改用 ROBOTIS 的 ROS 2 imitation-learning 工具鏈（Cyclo / Physical AI Tools）」，不是「禁止出現 ROS 2 節點」**。**→ 不是衝突，是一個要驗證的介面** |
| **7** | 續航 | 電池 vs 一次錄製時長 | 影響單次 lab day 能錄幾筆 |

### 💡 URDF 讓判準 1 變成「免改裝就能評估」——而且它讓 D025 提早有用

> **既有自走車有 URDF、OMX 也有 URDF（`open_manipulator_description/urdf/omx_f`）。**
> **→ 在 Isaac Sim 裡把手臂掛到車體上，直接量可達範圍與干涉，不用真的改裝一台車。**

**這比「先鎖上去再看」便宜一個數量級**，而且：
- 它是 **D025 的第一個實用產出**，而 D025 已決定「有實機資料後才做」——
  **⚠️ 但這個用途不需要實機資料**（純幾何，不是學習）。
  **→ 建議把「車體＋手臂幾何評估」從 D025 的排程中切出來，當成獨立的小任務**，
  因為它服務的是 8/29 的成本試算，而不是 Phase C。
- ⚠️ **幾何可達 ≠ 實際可夾。** 模擬能回答判準 1，**回答不了判準 3（晃動）**——那要實體量。

### ⚠️ 判準 6 真正剩下的技術風險（不是阻塞級，但要驗）

**把 policy 包成 ROS 2 node 之後，唯一會改變結果的是「觀測與動作的時序」：**

| 風險 | 為什麼 |
|---|---|
| **時戳對齊** | 訓練時 LeRobot 直連相機與舵機、自己處理同步；上車後影像／關節／action 走 ROS 2 topic，**多一層延遲**。訓練時看到的觀測分佈 ≠ 部署時看到的 |
| **控制頻率 jitter** | ACT 輸出 action chunk，需要**穩定**的執行頻率。DDS 在負載下有 jitter |

**與 D006 的關係：** D006 說跨公網推理會傷馬達，原因就是 jitter 造成指令不連續。
**同機 ROS 2 的 jitter 遠小於跨網，所以不是同等嚴重**——但**是同一類問題，用同一個方法驗**：
量實際的指令間隔分佈，不要只看平均值。

**→ 下一步：把實驗室自走車的型號與 URDF 來源列出來，先跑判準 1（幾何，模擬即可，不用碰車）。**

### 🔴 2026-08-27 `[Eric決定]`：車體選項與分工

| 項目 | 內容 |
|---|---|
| **死線** | ~~8/29~~ → **延後至 2026-09-01** |
| **主責** | **陳柏宇**（本案改由柏宇主要負責） |
| **已排除** | 實驗室「小藍」——評估不適合 |
| **三個候選** | ① 柏宇現有的 **mbot** 車體 ② **等實驗室送廠加裝 IMU 的 wildbot 車** ③ 自行調查購買 |

⚠️ **候選 ② 有排程風險**：送廠加裝 IMU 的回廠時間目前未知，**若它落在 Phase C 開始之後，等於這個選項對本計畫無效**。
**→ 先問到回廠日期，再評估。這是最便宜的排除手段。**
💡 候選 ①（柏宇現有 mbot）的最大優點是**可及性**——同 D021 的邏輯：**能碰到的車勝過更好但碰不到的車**。


- **Cross-reference:** `docs/camera_mount.md` §6, `docs/hardware.md`, `docs/meeting/2026-08-18.md` §六.

---

## D021 — 🔴 D002 REVERSED: OMX-AI is the primary platform for Phase A–B

- **Date:** 2026-08-24 (in the lab, ETA confirmed with the lab assistant)
- **Supersedes:** D002. **D002's own reversal condition fired, exactly as written.**
- **Decision:** Collect real Phase A–B data on **OMX-AI**. SO-ARM (ETA 2026-09-15) is no longer
  assumed to be the Phase A–B platform. Data recorded on OMX is **real data**, not
  pipeline-validation-only (this reverses the 2026-08-16 補記 in D002).
- **Triggering fact:** SO-ARM ETA confirmed as **2026-09-15** — four weeks after the 8/18 meeting's
  「近期送達」, not the one-to-two weeks that meeting assumed (`docs/meeting/2026-08-18.md` §三-1).
- **Why the reversal condition is met** — D002 read: *"SO-ARM stays unavailable beyond ~3 weeks while
  OMX becomes free, **and** no data has been collected yet."*

  | Clause | Status |
  |---|---|
  | SO-ARM unavailable > 3 weeks | ✅ 8/18 → 9/15 is 4 weeks |
  | OMX free | ✅ repaired by the assistant, booking sheet available (2026-08-18 §三-1) |
  | No data collected yet | ✅ only public-dataset pipeline validation |

  **This is not a change of mind — it is a pre-registered trigger firing. The log worked as designed.**
- **The scheduling argument, which is independently decisive:** `docs/phase_plan.md` already states
  「9/15–10/14 兩人同時處理書審與報名，A 線產能 ≈ 0。T0 最好早於 9/15 或晚於 10/15。」
  **SO-ARM lands on the first day of that dead window.** Waiting for it forces T0 ≥ 10/15, which by
  `phase_plan.md`'s own timetable pushes the end of Phase B past the 2026-11-30 P0-baseline deadline.
- **Accepted costs — do not let these get forgotten:**
  - Smaller public-dataset and community base than SO-ARM. **That was D002's principal argument**, and
    it is now being paid, not refuted.
  - **OMX trajectories do not transfer to SO-ARM** (different DOF, joint limits, action space).
    A later switch means **re-recording Phase B from scratch.**
  - The SO-101 hands-on experience from the workshop does not carry over to OMX.
- **✅ RESOLVED 2026-08-27 `[Eric決定]` — option 甲: OMX to the end; SO-ARM is a spare.**
  Alternatives were (乙) switch to SO-ARM on arrival and re-record, (丙) run both in parallel.

  **What 甲 commits us to, stated plainly so nobody re-litigates it in November:**
  - **All Phase B and Phase C data is OMX data.** SO-ARM's larger public-dataset ecosystem — D002's
    principal argument — is now permanently forgone for this project, not deferred.
  - **SO-ARM arriving on 2026-09-15 changes nothing about the schedule.** It becomes hardware
    redundancy: if the OMX breaks or gets booked out, we have a fallback that costs a re-record
    rather than a dead project.
  - **Do not collect "a few demos on SO-ARM just to compare."** Cross-arm demos cannot be pooled
    (different DOF, joint limits, action space), so such a set is neither training data nor a valid
    comparison — it is only a way to spend scarce lab days.
  - 乙 was rejected because re-recording 72+ demos costs ~1 lab-day-limited month and buys an
    ecosystem advantage that only pays off after this project ends. 丙 was rejected because
    D002 already ruled that two parallel data-collection pipelines are infeasible for a two-person
    team in six months.
- **Reverse the 甲 sub-decision if:** the OMX becomes unavailable for > 1 week during Phase B/C
  (borrowed out, broken, or booking contention), **and** the remaining schedule still admits a
  re-record. After Phase C begins, treat as irreversible.
- **Reverse if:** OMX becomes unavailable (borrowed out or broken) for > 1 week *before* Phase B
  recording starts, *and* SO-ARM has arrived. **Once Phase B recording begins, effectively
  irreversible** — same reason D002 gave.
- **Cross-reference:** D002, D022, D023, `docs/hardware.md`, `docs/phase_plan.md`,
  `docs/meeting/2026-08-24.md`, `calibration/2026-08-24_omx_follower_arm.json`,
  `configs/teleoperate_omx.yaml`.

---

## D022 — Wrist camera: buy a USB UVC module; do not use the lab's RealSense D405

- **Date:** 2026-08-24
- **Decision:** Purchase a dedicated **USB UVC camera module** for the wrist position. The lab's
  **RealSense D405 is rejected for the wrist.**
- **Why:** the D405 is too heavy to ride on this arm's wrist, and mounting it needs additional
  purchased tooling and adapters. **Wrist mass changes the arm's dynamics, and those dynamics are
  baked into every demo recorded with it** — this is a scene constant, not a convenience issue.
- **ETA:** the specified module arrives **2026-09-05 – 09-07**.
- **🔴 Interim decision — recording is NOT gated on the wrist camera.** For early trials, use
  **two third-person cameras** in place of wrist + third-person. The **single-third-person-camera**
  configuration is also acceptable as a deliberate, declared arm of the D004 ablation.

  ⚠️ **The configurations are not interchangeable as data.** Camera count and extrinsics are scene
  constants (see D004): demos recorded with 2× third-person **cannot be pooled** with demos recorded
  with wrist + third-person. **Treat interim recordings as pilot data, not Phase B data**, unless the
  configuration is declared as an ablation arm and frozen in `docs/experiment_spec.md` first.
- **🟡 UNVERIFIED, cheap to check (~20 min — change the camera count in the config):** whether the
  LeRobot ACT implementation trains with a single camera. The original ACT paper uses four; single-view
  performance will drop — **but by how much is itself a reportable result**, and it feeds D004's
  ablation directly. **This is an inference, not a verified fact. Verify before planning around it.**
- **🔴 2026-08-27 — VERIFIED. The unverified note above is resolved.** `[已查證]`

  **Method: read the pinned LeRobot source in this repo** (`lerobot/` submodule, commit `a16f34c0`),
  file `src/lerobot/policies/act/modeling_act.py` and `configuration_act.py`. Primary source, exact
  pinned version — not documentation, not a forum answer.

  **Finding 1 — single camera is supported, and it is not a special case.**
  The encoder consumes cameras through a plain loop with no fixed count:

  ```python
  # modeling_act.py, ACTEncoder input assembly
  if self.config.image_features:
      for img in batch[OBS_IMAGES]:
          cam_features = self.backbone(img)["feature_map"]
          cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features)
          cam_features = self.encoder_img_feat_input_proj(cam_features)
          cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
          encoder_in_tokens.extend(list(cam_features))
          encoder_in_pos_embed.extend(list(cam_pos_embed))
  ```

  `configuration_act.py::validate_features()` raises only if there is **neither** an image **nor**
  an env-state feature:

  ```python
  if not self.image_features and not self.env_state_feature:
      raise ValueError("You must provide at least one image or the environment state among the inputs.")
  ```

  → **N = 1 is explicitly legal. Camera count is set by the dataset's feature keys, not by code.**

  **Finding 2 — one shared backbone, so the ablation is architecturally clean.**
  `self.backbone` is a single ResNet applied to every camera in turn. Changing the camera count does
  **not** change the parameter count. The only difference is the encoder's input sequence length
  (`N × h·w` image tokens). **A 1-camera vs 2-camera comparison therefore isolates observation
  content, not model capacity** — which is exactly what an ablation needs, and is worth stating in
  the write-up.

  **Finding 3 — 🔴 there is NO camera-identity embedding. This is a risk for the interim plan.**
  The only positional information added to image tokens is
  `ACTSinusoidalPositionEmbedding2d`, a **2-D spatial** embedding over the feature map — and it is
  constructed once and applied **identically to every camera**. Nothing tells the transformer which
  camera a token came from; it must infer that from image content alone.

  **Consequence for D022's interim "two third-person cameras" configuration:**
  wrist + third-person are trivially distinguishable by content (one moves with the gripper, one does
  not). **Two third-person views of the same scene are not.** With no camera-ID embedding and two
  similar viewpoints, the encoder may partially conflate them, and the second view could add less
  than its token cost — or add noise.

  ⚠️ **This does not mean the interim plan is wrong.** It means the interim plan should be treated as
  **an experimental condition with a real hypothesis**, not as a stopgap that is obviously fine.

  ⚠️ **2026-08-27 — this finding was overstated and is hereby downgraded.** `[Eric說]`
  The 8/27 text called it "the one item that found something others had not noticed." **Eric called
  that 浮誇, and he is right.** The absence of a camera-identity embedding is a **known property of
  the ACT/LeRobot family, not a discovery**:
  - The SmolVLA paper names its camera slots by index (`OBS_IMAGE_1/2/3`) precisely because the
    slot order carries the meaning.
  - LeRobot issue [#1763](https://github.com/huggingface/lerobot/issues/1763): a user reports
    "better result during inference when I keep the same order (top, front, wrist)" and "worse result
    if different order", and the thread states outright that "there seems no clear way to tell from
    the embedding input which token is for which image".

  **What survives — and it is an operational rule, not a research claim:**
  > 🔴 **Camera ORDER is a scene constant, exactly like camera count and extrinsics.**
  > Freeze the feature-key order at collection time and never change it between training and
  > inference. Record the order in `experiment_spec.md` §3 alongside the extrinsics.

  **🔴 2026-08-30 `[已查證]` — what actually determines that order, and what the NAME does and
  does not do.** Read in the pinned source (`lerobot/`, commit `a16f34c0`), following the chain
  end to end rather than assuming:

  ```python
  # policies/act/modeling_act.py:134,143
  batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

  # configs/policies.py:151
  return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

  # policies/factory.py:298,315   <- input_features when inferred from the dataset
  features = dataset_to_policy_features(ds_meta.features)
  cfg.input_features = {k: ft for k, ft in features.items() if k not in cfg.output_features}

  # utils/feature_utils.py:139    <- plain `for key, ft in features.items()`, no sort
  ```

  **There is no `sorted()` anywhere on that path** (checked in `feature_utils.py`,
  `pipeline_features.py`, `policies/factory.py`). The order is dict-insertion order all the way
  down, and its origin is `meta/info.json`'s `features` object — which `hw_to_dataset_features`
  builds by iterating `robot.observation_features`, i.e. **the order the cameras are declared in
  the record config's `cameras:` block.**

  **Consequences, stated separately because they are different risks:**

  | | What it controls | Failure mode |
  |---|---|---|
  | **Declaration order in `configs/record.yaml`** | the tensor order the transformer sees | 🔴 **silent** — swap it and the model still trains, just worse |
  | **The key string itself** (`left_front` vs `cam1`) | nothing inside the model — no name embedding exists | — |
  | **Key string, across configs** | training config and inference config must use the same keys | loud: `KeyError` |
  | **Which physical camera is plugged in as which key** | the actual image content behind each key | 🔴 **silent** — the worst one |

  **→ So the naming choice is not about the model, it is about catching the two silent failures.**
  `left_front` / `right_front` encode a physical fact that a person can verify against the image in
  two seconds; `cam1` / `cam2` cannot be checked at all. That is the entire argument for the names,
  and it is an operational one, not a modelling one. (A side benefit: `left` sorts before `right`,
  so the intended order survives even if some future tool does sort the keys.)

  **🔴 2026-08-27 `[Eric決策]` — recording and training plan (supersedes the interim recommendation
  that was here):**

  | Phase | What is RECORDED | What is TRAINED |
  |---|---|---|
  | **Pilot (now → UVC arrives)** | **both third-person cameras**, as two separate feature keys | **two models**: (i) single third-person, (ii) dual third-person |
  | **After the UVC wrist camera arrives (9/5–07)** | **a new campaign** with wrist + third-person | its own model |

  **This is deliberately not just a 1-vs-2 ablation.** Three configurations, two recording campaigns.
  ⚠️ **The two campaigns are separate datasets and must not be pooled** (camera set is a scene
  constant, D004). The second campaign is a re-record, and that cost is accepted knowingly.

  **Still unverified after a web search on 2026-08-27:** *how much* single-view performance drops.
  Searched for ACT / LeRobot / SO-100 single-vs-multi-camera success-rate comparisons and
  **found no published ablation with numbers** — only architecture explainers and the ordering report
  above. **So there is no external指標 to decide by; the only way to get the number is to run it.**
  Since the pilot records both streams anyway, the comparison costs one extra training run, not one
  extra recording campaign. `[已查證：查了，沒有找到]`

### 2026-08-31 `[Eric說]` — D405 physically mounted on the wrist for measurement; NO decision yet

Eric mounted the D405 on the wrist using **a mounting method that had previously been set aside**,
and ran a full teleop session with it in place.

**Findings — evidence, not a reversal. D022 is unchanged:**

- **The mount holds.** The previously-discarded method works mechanically.
- **Observable droop:** with the arm fully extended, the wrist end **sags slightly** under the D405's
  mass. This is the first *measured* confirmation of D022's "too heavy" claim, which until now was an
  inference from the workshop discussion.
- **Correlates with the §4-1 offset baseline:** `analysis/teleop_offset_2026-08-31.csv` shows
  `wrist_flex` leader→follower deltas of **+3.5° to +6.0°** across poses (flagged 🟡/🔴) — by far the
  largest of any joint (every other joint < 1.6°). Consistent with added wrist-end load on that axis.
  ⚠️ **Not yet isolated:** this baseline was taken *with* the D405 on. A no-D405 baseline is needed to
  separate gravity/load from intrinsic gear backlash.

**Status (2026-08-31): `[未確認]` — deferred to an advisor / 柏宇 discussion.**

### ✅ 2026-09-01 `[Eric決定]` — D405 IS the interim wrist camera, with a defined end condition

**The D405 rides on the wrist and is recorded as the wrist camera until *either*:**

1. **the dedicated USB UVC module arrives** (and proves usable), *or*
2. **work has advanced to Phase C** (vehicle / outdoor — a full re-record anyway).

This supersedes D022's earlier "two third-person cameras" interim plan: the interim configuration is
now **D405 wrist + one third-person (D455)**, i.e. the wrist+third-person geometry, just with the
D405 standing in for the future UVC module.

**Consequences:**

- The pilot / early-Phase-B campaign is **D405-wrist + D455-third-person**. `configs/record_omx.yaml`
  reflects this (`wrist` = D405 SN 260322271459, `front-left` = D455 SN 262822305610).
- **Swapping D405 → UVC later is a scene-constant change** (wrist mass *and* image characteristics)
  → a re-record boundary. Data recorded with the D405 wrist is not poolable with post-UVC data.
- Reaching **Phase C** is a re-record boundary regardless.
- The known cost stands: D405 wrist mass droops the arm at full extension and inflates the
  `wrist_flex` teleop offset (2026-08-31 block above; `analysis/teleop_offset_2026-08-31.csv`).
- 🔴 **D405 RGB auto-exposure runs hot at wrist range** (washed-out / very bright — seen in the
  2026-08-31 pilot videos). Before the real pilot, set fixed `exposure` / `gain` / `white_balance`
  in `configs/record_omx.yaml`, freeze them as scene constants in `experiment_spec.md` §3, and
  follow the tuning procedure in `docs/field_manual.md` §5-(0).

- **Accepted costs:** one more purchase and its lead time; interim pilot data may not be poolable.
- **Reverse if:** the UVC module arrives and proves unusable (frame rate, latency, or mounting), in
  which case re-open the D405-with-counterweight option or source a lighter depth camera.
- **Cross-reference:** D004, D021, `docs/camera_mount.md`, `docs/hardware.md`,
  `analysis/teleop_offset_2026-08-31.csv`.

---

## D023 — A6 (workspace verification) is gated on replacing the short arm cable

- **Date:** 2026-08-24
- **Observation (2026-08-24 lab session):** one of the cables running along the arm is **too short and
  physically limits the arm's range of motion.**
- **Decision:** **Do not measure the workspace (A6), do not fix the 30 seeded object placements (A7),
  and do not record any demo** until that cable is replaced or re-routed.
- **Why — the dependency chain is the whole point:**

  ```
  short cable  →  range of motion constrained
               →  measured workspace (A6) is wrong
               →  the 30 seeded object placements (A7) are wrong
               →  every Phase B demo recorded against them must be re-recorded
  ```

  **It is the cheapest link in the chain to fix and the most expensive one to skip.**
- **Accepted costs:** A6–A13 slip to the next lab visit. Because lab visits are the scarce resource
  during the summer (see `docs/phase_plan.md` §T0 與實驗室可及性), that is a real cost —
  **but it is smaller than re-recording 72 demos.**
- **Reverse if:** measurement shows the cable does **not** constrain the reachable set *within the task
  workspace* — i.e. the limit lies outside the region objects are ever placed in.
  **Measure before concluding this; do not eyeball it.**

### 🟡 2026-08-27 — A7 partially unblocked `[Eric提議 → 採納]`

**Eric's proposal:** use the rough figure measured in the 2026-08-24 lab session — the arm fully
extended reaches the table at roughly **33–43 cm** — reserve margin because at the boundary the arm
has no usable grasp approach angle anyway, and **fix the seeded object placements now** instead of
waiting for the cable.

**Accepted, and the reason it is sound is worth writing down, because it is not "the cable stopped
mattering":**

> **A conservative subset of the reachable set is invariant under replacing the cable.**
> A too-short cable *removes* reachable space. Replacing it with a longer one *only adds space back*.
> So any placement chosen to lie strictly **inside** the currently-reachable region stays reachable
> afterwards → **demos recorded against those placements stay valid.**
> The gate in D023 exists to stop us baking a *wrong* workspace into the data. A conservative
> workspace cannot be wrong in that direction — only smaller than necessary.

**🔴 Two conditions this argument depends on. If either breaks, the argument breaks:**

1. **The fix must be replacement, not re-routing.** A longer cable is a monotone relaxation. **Re-routing
   changes *which directions* are constrained** — it can remove space in one azimuth while adding it in
   another, and then earlier placements are no longer guaranteed reachable.
   → **If you re-route instead of replace, this exemption is void and D023's original gate applies.**
2. **Grasp-feasible ⊊ reachable.** Eric already identified this: at the boundary the arm can reach the
   point but has no usable approach angle. So margin is not optional padding — it is the difference
   between "the gripper can touch it" and "the policy can be taught to pick it up."

**⚠️ The 33–43 cm figure was ambiguous. 🔴 DISAMBIGUATED 2026-08-31 `[Eric說]` — see the block
"what 33–43 cm actually meant" below. Both readings entertained here were wrong; they are kept only
for the reasoning trail.**

| Reading (SUPERSEDED) | Meaning | Implication |
|---|---|---|
| ~~(a) annulus~~ | contact ring spans r = 33 → 43 cm | usable outer bound ≈ 43 cm |
| ~~(b) azimuth-dependent~~ | max reach varies 33–43 cm **by direction**, because the cable constrains some directions more | usable outer bound = **33 cm**, the worst direction |

~~Reading (b) is the more likely one for a cable constraint, and it is also the conservative one.~~
~~→ Design against (b). Do not use 43 until (a) is confirmed by measurement.~~

**Provisional workspace `[AI提議]` — design against the worst case:**

```
r_outer = 33 cm − 5 cm margin  =  28 cm      # worst-azimuth reach, minus grasp-angle margin
r_inner = TBD                                # smallest radius with a usable top-down/oblique approach
azimuth = the unconstrained sector only      # which side the cable limits is NOT yet known
```

**Three numbers must still be measured on the next lab day. ~15 minutes, before anything else:**
*(🔴 item 2 revised by the 2026-08-31 block above — measure `r_outer` top-down and side-only
separately, not a single "r_max"; there is no (a)/(b) reading to confirm.)*

1. **Which azimuth range the cable actually constrains** (sweep the arm left→right at full extension)
2. ~~**r_max in the worst azimuth** — this confirms or refutes reading (b)~~ → **`r_outer` (top-down
   graspable) per azimuth, plus `r_outer_side` for the record** (D023 §2026-08-31)
3. **r_min where a grasp approach is still feasible** — the inner bound

**Then the provisional grid either holds as-is, or shrinks. It can never need to grow, which is the
whole point.** Record all three in `docs/experiment_spec.md` §3 (scene constants) with the date.

**🔴 2026-08-27 `[Eric決定]` — the fix is CABLE REPLACEMENT, not re-routing.**
→ The monotone-relaxation argument above **holds**. The conservative-subset exemption is valid.

**🔴 2026-08-27 `[Eric決定]` — provisional workspace numbers, pending measurement:**

```
r_outer = 30 cm      # provisional
r_inner = 20 cm      # provisional assumption
```

**These supersede the AI's earlier `28 cm / TBD`.** They remain `PROVISIONAL` until the three
measurements below are taken. Both are inside the 33 cm worst-case bound, so the invariance argument
still applies.

**Three scripts to build before the next lab day (`[Eric決定]` to build them):**

| # | Script | What it must do | ⚠️ Verified constraint |
|---|---|---|---|
| 1 | **Reach logger** | While teleoperating, capture the gripper's planar (x, y) at the moment it touches the table, to establish r_outer / r_inner empirically | 🔴 **REVISED BY D026 (2026-08-31): FK from the `omx_f` URDF is now the method; tape measure is the fallback. The `omx_f` URDF exists and matches our arm; `placo` will be added to the pinned env.** Original note (kept for trail): `omx_follower` exposes joint states only (`omx_follower.py`); `RobotKinematics` (`model/kinematics.py`) needs a URDF + `placo`; placo availability was then unverified. Spec: `docs/specs/S1_reach_logger.md` |
| 2 | **Uniform sampler** | Draw N points uniformly over the annular sector | 🔴 **Sampling r ~ U(r_inner, r_outer) is WRONG — it biases points toward the centre.** For uniform *area* density: `r = sqrt(U(r_inner², r_outer²))`, `θ = U(θ_min, θ_max)`. This is the classic annulus-sampling bug |
| 3 | **Placement registration** | Guarantee a sampled coordinate is physically placed at that coordinate | Proposal: print a polar mat (radius rings + azimuth rays) keyed to the arm base. ⚠️ **A mat left in frame becomes part of the background — a scene constant baked into every demo.** **Place the object using the mat, then REMOVE the mat before recording.** ~10 s per episode |

**❓ "What is the grid actually for?" — `[Eric問]` 2026-08-27. Answering it separates two things:**

**The PURPOSE is legitimate and must be preserved:**
1. **Cross-model comparability** — every model is evaluated from the *same* start positions, so a
   success-rate difference is attributable to the model, not to easier placements.
2. **Failure attribution** — `eval/_template.csv` has a `grid_cell` column so a failure can be tied to
   *where* the object was. The expected mechanisms (`drift`, `collision`, `pushed_away`) are spatial.

**The 3×3 FORM was never ratified.** Its only source is `experiment_spec.md` §5, which itself carries
"⚠️ 30 無法均分 9 格… 分配方式待定" plus a placeholder split. **Eric states he never understood or
decided it** `[Eric說]`. Under the source-tagging rule it is `[AI推論]` that leaked into a spec file.

> **→ Eric's uniform-sampling proposal can replace the 3×3 grid outright, provided both purposes
> survive: sample ONCE, freeze the resulting list, give every point an ID, and put that ID in the eval
> CSV.** "Seeded" means *sampled once and then fixed* — **not re-randomised per run.** If it is
> re-randomised, cross-model comparability is destroyed and the whole scheme is pointless.
> `grid_cell` should then be renamed to something like `placement_id`.

**Until then:** the placement grid may be **designed and written down**, but it is marked
`PROVISIONAL` and **no demo is recorded against it.** Recording still waits for the cable.
**A7 is unblocked for design; A8 remains blocked.**

### 🔴 2026-08-31 `[Eric說]` — what "33–43 cm" actually meant

The 2026-08-24 figure was never an annulus and never an azimuth-dependence. Eric's account:

| Figure | Meaning |
|---|---|
| **≈ 43 cm** | arm **fully extended**, approximate farthest point it can reach on the table. At that distance the **only feasible grasp is from the side** (horizontal approach). |
| **≈ 33 cm** | farthest distance at which a **top-down grasp** (vertical / oblique approach) is still feasible. |

So 33–43 cm is a **grasp-approach band**, not a reachability range. It is exactly the
"grasp-feasible ⊊ reachable" gap that condition 2 above already named — now with numbers.

**Consequences:**

1. **The (a)/(b) ambiguity table above is void.** There is no "annulus vs azimuth-dependent" question
   to settle, and `r_max` is no longer the quantity that decides anything.
2. **`r_outer` for this experiment is the top-down-graspable bound.** The task's demos approach an
   upright container from above/obliquely (§1-1; `experiment_spec.md` §2 L1 = upright opaque
   bottle/can). Seeding a placement past the top-down bound would force the demonstrator into an
   inconsistent side grasp there and corrupt the data. → operative bound ≈ **33 cm − grasp-angle
   margin**. PROVISIONAL `r_outer = 30` still sits inside this, so nothing already written breaks.
3. **The reach logger (script 1) must measure reach *per grasp approach*,** not merely "gripper
   touches table". See **D026** and `docs/specs/S1_reach_logger.md`.
4. **The cable constraint is a separate, still-unmeasured axis.** The cable removes reachable space in
   some azimuths; the azimuth-limit sweep captures that, independent of the 33/43 grasp band.

### 🔴 2026-08-31 `[Eric說]` — cable resolved by RE-ROUTING the existing cable, NOT replacement

The lab had **no spare Robot Cable-X3P** (confirmed on site, 2026-08-31). The clearance problem was
solved by **re-routing the existing cable through the space** (`docs/hardware.md` 解法 1), not by
swapping in a longer one.

**This voids the 2026-08-27 conservative-workspace exemption.** That exemption rested entirely on
*"a longer cable only adds reachable space back"* — a monotone relaxation, under which any placement
chosen strictly inside the currently-reachable region stays reachable afterwards. **A re-route is not
monotone:** it can remove reachable space in one azimuth while adding it in another. Condition 1 of
the 2026-08-27 exemption named this exact case as the thing that would void it. It fired.

**Consequences:**

1. **D023's original gate is back for A7.** The 30 seeded placements may still be *designed*, but the
   frozen list must be built from a **fresh S1 measurement of the re-routed configuration** — in
   particular the azimuth sweep (`a` key in `reach_logger.py`), to establish which directions, if
   any, the re-routed cable still limits. Provisional `r_outer = 30 / r_inner = 20` stay PROVISIONAL
   and are now *unbacked* by the invariance argument, not merely unmeasured.
2. **The re-routed cable path is now a scene constant.** Photograph it and mark it (tape + pen).
   If it is ever re-routed again, every demo recorded against the resulting placements is invalidated
   — the failure mode D023 exists to prevent.
3. **A8 (recording) stays blocked** until S1 has characterised the re-routed reachable set and the
   placement list is frozen against it. Re-routing removed the *hard* motion limit; it did not
   remove the requirement to measure before freezing.
4. **No spare cable was bought.** If a proper-length replacement is later sourced and installed, that
   is itself a workspace change → re-measure and re-freeze.
5. **2026-08-31 tape measurement (`[Eric說]`; FK validation failed → D026 tape fallback):**
   `d_offset` (pan axis → chassis bottom edge) ≈ 5 cm; radii below already add it back:
   `r_outer` top-down ≈ **41 cm**, `r_outer` side-only ≈ **49 cm**, `r_inner` ≈ **22 cm**.
   Effective azimuth sector ≈ **135°** (`theta ∈ [−90°, +45°]`, 0° = forward).
   🔴 **The sector edge is a hard mechanical limit: rotate the arm past it and the arm body physically
   strikes the third-person camera / its mount.** It is NOT camera field-of-view (not "the object
   leaves frame") and NOT the re-routed cable. So the open worry in point 1 ("which directions the
   re-route limits") resolves to: **the re-route imposed no azimuth limit in the task region; the
   camera *mount position* does.** Consequence: the 135° figure is bound to the current camera mount —
   move the mount and re-measure. Treat the sector edge as a no-go zone during teleop / S1 too, not
   just for placement. Numbers in `experiment_spec.md` §3; PROVISIONAL pending an S2 `--dry-run`
   feasibility pass.
6. **Scope: this is the Phase-A pilot workspace only (`[Eric說]` 2026-08-31).** The S1/S2 output
   (`campaign_A_pilot_2cam` bounds + the 90 seeded points) is computed for the current fixed-tabletop
   pilot layout. **Phase B on the vehicle body gets a fresh S1/S2 from scratch** against the real
   on-vehicle layout — consistent with D022 (pilot 2×third-person data is not pooled with the later
   wrist+third-person campaign anyway). Because it is a pilot, the exact N is negotiable if the
   `--dry-run` packing check is tight.

### 🔴 2026-09-01 `[Eric說]` — the pilot is downgraded to manual eyeball placement; S3 mat needs a redesign for Phase B

**On-site, the S1→S2→S3 flow proved impractical to execute:**

- **The polar mat's origin (the pan axis) is not markable.** It sits under the chassis; the arm blocks
  putting a mat there; and locating it needs an estimate of the pan axis plus its `d_offset`.
- **The table cannot be marked** (`[Eric說]` 2026-09-01) — so "align the mat once, prick every seeded
  point, remove the mat" is not available.
- **Per-episode mat placement + object + removal is too fiddly**, and a mat left in frame becomes a
  baked-in scene constant (D023 script 3 / D018).

**Decision (pilot only):** the `campaign_A_pilot` recording uses **manual eyeball / roughly-random
object placement**. Consequences, all accepted because a pilot is pipeline validation, not a result
(`phase_plan.md` Phase A: "3 筆成功率預期 0%"):

- `placement_id` is left blank in `episode_meta` (no frozen list to reference).
- No location-based failure attribution (D015 spatial mechanisms).
- **Not poolable with Phase B** (already true under D022).
- 🔴 **The seeded S1→S2→S3 protocol still governs the real Phase B campaign** — this downgrade is
  pilot-scoped only.

**Phase B placement-registration options (needs the arm; not yet chosen), ranked:**

1. **Camera-overlay placement (mat-free, recommended).** One-time ArUco/checkerboard →
   camera↔table homography at scene-freeze; a small script overlays target crosshairs on the live
   camera feed; the object is placed under the crosshair. Reusable into Phase C. The overlay script
   is buildable off-arm.
2. **Rigid jig-registered locator board, lifted away as one piece.** Divots on a stiff board that
   butts against a table-clamped jig aligned to the chassis front edge; ~2 s per episode, no table
   marking, nothing left in frame.
3. **Permanent, campaign-consistent mat**, frozen as a scene constant. Least effort, but bakes a
   spatial-reference shortcut into the policy (D018) and adds a Phase-C distribution shift.

All three reference the **chassis-front-edge midpoint** as the physical datum; `d_offset` (pan axis →
chassis front) ≈ 5 cm is measured once and S2 keeps sampling in pan-axis polar, translating its
`x_cm/y_cm` output into that datum's frame.

- **Cross-reference:** D024, `docs/specs/S3_placement_mat.md`, `eval/README.md` (ArUco note).

- **Cross-reference:** D021, D026, `docs/phase_plan.md` A6/A7, `docs/meeting/2026-08-24.md`,
  `docs/meeting/2026-08-31.md` §6, `docs/hardware.md` (Cable).

---

## D024 — ✅ RESOLVED 2026-08-27 `[Eric決定]`: evaluation & placement protocol

**Supersedes the withdrawn version below.** All four reconciliation questions answered by Eric.

| # | Question | ✅ Decision |
|---|---|---|
| 1 | 60 or 72 per recording campaign? | **60 筆** (50 train / 10 open-loop eval). The 72 in `phase_plan.md` was 60 + 12 object-position OOD; **that OOD block is cancelled** |
| 2 | What was "OOD eval 12" OOD in? | **Object position — and it is CANCELLED**, because the experiment already demonstrated it. **Position-OOD is not re-run.** Later OOD work uses the *other* variables (object type, lighting, background) per `phase_plan.md` B3 |
| 3 | Are training-set positions seeded too? | **Yes. Training positions are pre-defined and seeded**, not ad-hoc |
| 4 | Is the closed-loop 30 in- or out-of-distribution? | **In-distribution: a coverage sample over the SAME distribution as training** |
| 5 | 3×3 grid? | **Replaced by uniform sampling over the annular workspace** (D023). The 3×3 form was never ratified |

### What this settles, stated so it cannot drift

- **Per campaign: 60 recorded episodes** = 50 training + 10 open-loop eval.
- **Closed-loop evaluation: 30 trials × 3 objects = 90**, drawn from the **same** workspace
  distribution as training. It measures in-distribution competence, **not generalisation**.
- **Both the training placements and the evaluation placements are sampled once, frozen, and given
  IDs.** Sample once → freeze → never re-randomise. Re-randomising destroys cross-model comparability,
  which is the entire reason for seeding.
- **Generalisation is tested by swapping variables (object / lighting / background), not by moving the
  object to unseen coordinates.** Position-OOD is off the table.

### 🔴 Consequences that must be honoured downstream

1. **`phase_plan.md` B1/B4a–c must change 72 → 60** and drop the "OOD eval 12" column.
2. **`eval/_template.csv`: rename `grid_cell` → `placement_id`**, and it must reference the frozen
   sampled list, not a 1–9 grid.
3. **`experiment_spec.md` §5**: delete the 3×3 grid and its placeholder "6 格各 3 次、3 格各 4 次"
   split. Replace with the sampling procedure and the frozen list.
4. **Two frozen lists are needed, not one:** a training placement list and an evaluation placement
   list, **drawn from the same distribution**.

### ✅ 2026-08-30 `[Eric決定]` — the two lists may NOT overlap

> Eric's answer to the open sub-question, decided in another session and recorded here.
> **This entry is the 正本 for it.**

**What it settles:** an evaluation placement may not be a training placement. The reason is the one
the question was asked for — reusing a trained-on position inflates the success rate, and the number
that goes in the write-up would then be partly a memorisation score.

**🔴 What it does NOT yet settle, and this is not pedantry — the sampler cannot be written without
it.** Placements are drawn from a *continuous* annulus, so two independently drawn points coincide
with probability zero. "Non-overlapping" as literally stated is satisfied by any two samples and
therefore constrains nothing. To have force it must become a **minimum separation `d_min`**: every
evaluation point is at least `d_min` from every training point.

**🔴 And `d_min` runs into a packing limit that the provisional workspace cannot pay.** With
`r_inner = 20`, `r_outer = 30` and an azimuth sector of `f` (fraction of a full turn):

```
A            = f · π · (r_outer² − r_inner²)          # usable area, cm²
N_max(d)     ≈ A / (0.866 · d²)                       # hexagonal-packing upper bound
spacing_rand ≈ 0.5 / sqrt(N / A)                      # mean nearest-neighbour for a random sample
```

For a **180° sector**, `A ≈ 785 cm²`, and the two frozen lists together need
**N = 50 training + 30 evaluation = 80** distinct points:

| `d_min` | Rationale for that value | `N_max` | Feasible for N = 80? |
|---|---|---|---|
| 7 cm | ≈ one aluminium-can diameter — "a different placement you could see" | ~18 | ❌ not close |
| 5 cm | | ~36 | ❌ |
| 3 cm | | ~101 | 🟡 only in theory; dart-throwing achieves far less than the packing bound |
| 2 cm | ≈ twice the realistic placement-repeatability error | ~227 | ✅ |

**And at N = 80 the mean nearest-neighbour distance is ≈ 1.6 cm regardless of what we ask for** —
which is the same order as how accurately an object can be put down by hand off a printed mat.

**→ The honest reading: at the provisional workspace, "non-overlapping" can only mean "far enough
apart to be distinguishable placements", not "a different region of the workspace".** Three ways out,
and Eric picks:

| | Option | Cost |
|---|---|---|
| **甲** | Keep 80 points; set `d_min ≈ 2 cm`. Separation guarantees *distinguishability*, nothing more | The eval set is honestly in-distribution, and the claim in the write-up must say so |
| **乙** | Cut the number of distinct placements (e.g. 15 training × repeats) to buy a real `d_min` | Loses the spatial coverage D024 chose option A for — and coverage is what makes failure clustering visible |
| **丙** | Enlarge the annulus once the real r_inner / r_outer are measured | Free if the measurement happens to be generous; unknown until 2026-08-31 |

⚠️ **Re-run the table with the measured numbers before choosing.** The whole calculation above uses
PROVISIONAL 20/30 and a guessed 180° sector; the real sector is constrained by the arm cable (D023)
and is likely *smaller*, which makes the packing problem worse, not better.

⚠️ **One tension to state rather than paper over:** D024 says the closed-loop 30 is a coverage sample
over the **same** distribution as training. Enforcing a separation makes the eval sample a *thinned*
version of that distribution — points are pushed into the gaps between training points. At
`d_min ≈ 2 cm` the bias is negligible; at `d_min ≈ 7 cm` the eval set would sit systematically in the
holes, which is a different experiment. **This is a second reason the answer cannot be "make d_min as
large as possible".**

### ✅ 2026-08-31 `[Eric決定]` — three lists, `eval-close` shared, `d_min` is a global minimum

Ratified while freezing the S2 spec. 正本 for the mechanics: `docs/specs/S2_placement_sampler.md`
(algorithm), `docs/specs/S3_placement_mat.md` §6 (all three lists on one printed mat).

- **Three frozen lists, not two:** `train`(50) + `eval-open`(10) + `eval-close`(30). `eval-open` is
  the open-loop evaluation inside the 60-episode recording split; `eval-close` is the closed-loop 30.
  They are separate lists → **50 + 10 + 30 = 90 distinct points**.
- **The `eval-close` 30 are shared across all 3 objects**, not re-sampled per object. Reason: position
  is then controlled when comparing object difficulty, and 90 distinct points almost certainly fail
  the packing feasibility check in the cable-limited sector. A per-object spatial map is the
  cheaper-to-buy-later option (re-sample a subset).
- **`d_min` is the minimum separation between *every* pair of points across all three lists**, not
  just train↔eval. Area-uniform sampling rarely places points close anyway; making it global costs
  little and removes an ambiguity.
- **Sampling is stratified, not pure random** (`[Eric決定]` 2026-08-31). Each list's sector is cut
  into `n` equal-area cells (radius split on `r²`, azimuth split on angle); one uniform-random point
  per cell. Pure dart-throwing at n≈30 leaves luck-of-the-seed empty patches, and these lists exist
  to make failure *clustering* visible — a coverage sample must actually cover. Still seeded, still
  random within each cell, still byte-identical per seed. `d_min` is enforced by redrawing inside the
  same cell. 正本 for the algorithm: `docs/specs/S2_placement_sampler.md` §4.
- **`--margin` is NOT applied to `r_inner`** — only `r_outer −` and both azimuth edges. The failure
  mechanisms cluster at the outer boundary and constrained azimuth; margining the inner edge spends a
  large fraction of an already-tight annulus for little gain. Add `--margin-inner` later if needed.
- **`d_min` — option 甲 chosen, working value 2.0 cm** (`[Eric決定]` 2026-08-31). Keep the point
  counts; `d_min` guarantees *distinguishable placements*, not *different regions*. 乙 (cut points)
  and 丙 (wait) are not taken. The number 2.0 is provisional in the same sense as `r_inner` / `r_outer`
  — it is pending the S1-measured sector. S2's own run shows 2.0 does **not** fit 90 points in the
  provisional 123° sector (`N_rsa ≈ 85 < 90`), so once S1 gives the real sector, re-run
  `sample_placements.py --dry-run` and confirm 2.0 holds or nudge it (e.g. 1.8). S2 keeps `--d-min`
  a required parameter with no default regardless.

- **Reverse if:** the in-distribution success rate saturates so early that the 90 closed-loop trials
  stop discriminating between models — then reconsider adding held-out positions.
- **Cross-reference:** D015, D016, D023, `docs/phase_plan.md` Phase B, `eval/README.md`.

---

## D024-WITHDRAWN — the version that asked the wrong question (kept for the provenance findings)

> **Withdrawn because it answered the wrong question.** It treated "30 does not divide by 9" as the
> problem. **Eric asked where 30 even came from, and checking that surfaced three larger issues.**
> **Do not act on the options table below until the reconciliation is settled.** `[Eric說]` he is not
> reading this entry until then.

### 🔴 What the provenance check found (2026-08-27, `[產出物]` — grep across the repo)

| # | Finding | Evidence |
|---|---|---|
| 1 | **30 trials/object is legitimately sourced.** Team decision 2026-08-13 (D016, owner 陳柏宇), tracked ✅ at the 8/18 advisor meeting. 3 objects × 30 = **90 closed-loop trials** | `decisions.md` D016; `meeting/2026-08-18.md` §一-2, §二-2 |
| 2 | **Closed-loop 30 and the 10 open-loop evals are NOT in conflict — they measure different things.** The 10 in the recording split are **open-loop** (prediction error against recorded trajectories). The 30/object are **closed-loop** (the arm actually runs). **Answering Eric's question: yes, 30 is real closed-loop evaluation.** | `phase_plan.md` B1; `experiment_spec.md` §5 |
| 3 | 🔴 **The 3×3 grid was never ratified by anyone.** Its only source is `experiment_spec.md` §5, which itself says "⚠️ 30 無法均分 9 格… 分配方式待定" and offers a placeholder split. **Eric: never understood it, never decided it.** → `[AI推論]` that leaked into a spec file | `experiment_spec.md` §201; `eval/README.md` §28,34 |
| 4 | 🔴 **60 vs 72 — two 正本 disagree, and this is a real conflict.** `meeting/2026-08-18.md` §二-4 (what was reported to the advisor) says **60 筆 (50 train / 10 open-loop eval)**. `phase_plan.md` B1 says **72 筆 (50 / 10 / 12 OOD eval)**. Difference = the 12 OOD. phase_plan v2 is dated 8/15, i.e. *before* the meeting that reported 60 | both files |
| 5 | 🔴 **"OOD eval 12" never says OOD in WHICH dimension.** If it is object-position OOD, then position-OOD already exists inside B1 — and B3 ("用 B1 模型逐一撤換變因：物體／光線／背景") covers the *other* variables. **The documents do not state which.** Eric's question exposed a genuine gap | `phase_plan.md` B1/B3 |

### What Eric must arbitrate before D024 can be rewritten

1. **60 or 72?** Which is the 正本 — the number reported to the advisor, or `phase_plan.md`?
2. **What is the "OOD eval 12" OOD in?** Position? Or was it meant as a placeholder?
3. **Are training-set object positions also seeded/pre-defined, or only evaluation positions?**
   `meeting/2026-08-18.md` §二 lists "30 組 Seeded 配置" under *待實機確認*, and only in the
   evaluation context. **Training-set positions are unspecified.**
4. **If position-OOD is already covered, should the closed-loop 30 be a coverage sample over the
   SAME distribution as training, or deliberately include positions training never saw?**
   These are different experiments and they cannot both be the 30.

⚠️ **Until 1–4 are settled, the provisional placement design (D023) should sample over the workspace
without committing to how many of those points are "in-distribution" vs "held out".**

---

## D024-OLD — the withdrawn proposal, kept only so the reasoning is not lost

- **Date:** Proposed 2026-08-27
- **Status:** **PROPOSED `[AI提議]`. Eric has not decided.** This resolves an open question carried
  since D016 ("30 trials per object") and the earlier unreconciled "3×3 grid" note.
- **The problem:** D016 fixes 30 trials per object. `experiment_spec.md` calls for 30 seeded
  placements for cross-model reproducibility. **30 does not divide evenly by a 3×3 grid (9 cells)**,
  and nobody ever reconciled the two numbers.
- **The real choice is coverage vs. within-pose variance, and it is a genuine trade-off:**

  | Layout | Distinct poses | Reps each | What you get | What you lose |
  |---|---|---|---|---|
  | **A** | **30** | 1 | **Maximum spatial coverage.** Best for locating *where* failures happen | No within-pose variance estimate |
  | **B** | 15 (3 radii × 5 azimuths) | 2 | Some within-pose variance | Half the coverage |
  | **C** | 10 (3×3 + centre) | 3 | Good within-pose variance | Coarse spatial map |

- **Recommendation `[AI提議]`: option A — 30 distinct poses, 1 trial each.**
  **Reason:** our failure taxonomy already has a `mechanism` axis (D015), and the mechanisms we
  expect — `drift`, `collision`, `pushed_away` — are **spatially driven**: they should cluster at the
  workspace boundary and in the constrained azimuth. **Coverage is what makes that cluster visible;
  repeats at a handful of poses cannot show it.** Within-pose variance is the cheaper thing to buy
  later (re-run a subset), whereas coverage cannot be recovered without a full re-run.
- **Argument against A, stated fairly:** ACT with temporal ensembling is not deterministic, so a
  single trial per pose gives a noisy per-pose estimate. **If the headline number is the aggregate
  success rate over 30 valid trials, that is unaffected** — but any claim about a *specific* pose
  needs repeats. **A is right if the 30 trials are a coverage sample; B is right if per-pose success
  is itself a reported quantity.** Decide which one the write-up will claim.
- **Layout, if A is chosen:** 3 radii × 10 azimuths inside the provisional sector (D023), or a
  low-discrepancy sample over the annular sector. **Same 30 poses reused for every model, forever** —
  that is the whole point of seeding.
- **Reverse if:** Phase B shows failures are not spatially clustered at all, in which case coverage
  bought nothing and repeats would have been the better spend.
- **Cross-reference:** D015, D016, D023, `docs/experiment_spec.md` §3/§5.

---

## D025 — ✅ DECIDED 2026-08-27: Isaac Sim data collection, AFTER real-hardware data exists

- **Date:** Proposed 2026-08-27 `[Eric提議]`
- **Status:** **PROPOSED. Eric proposed it; the premise contains a factual error that must be
  settled before it can be decided.**
- **Proposal:** during Phase A/B, additionally collect OMX task data by teleoperating in Isaac Sim,
  citing [ROBOTIS Open Source](https://docs.robotis.com/docs/systems/omx/resources/open_source/)
  as evidence that "OMX has simulation model files that can be imported".

### 🔴 The premise needs correcting — verified 2026-08-27 `[已查證]`

| Claim | What the sources actually say |
|---|---|
| "cyclo_lab provides Isaac Lab assets for our arm" | ❌ **`ROBOTIS-GIT/cyclo_lab` supports OMY (OpenMANIPULATOR-**Y**) and FFW-BG2/SG2. OMX is not listed.** Its tasks are OMY Reach / Lift-Cube / Open-Drawer / Stack-Cube / Pick-Place and FFW Reach / Pick-Place |
| "OMX has simulation model files" | 🟡 **Partly.** `ROBOTIS-GIT/open_manipulator` has `open_manipulator_description` (URDF + meshes). But that repo points simulation users to **`robotis_mujoco_menagerie` — MuJoCo, not Isaac Sim.** **No ready-made OMX USD asset was found** |
| "ROBOTIS's own OMX imitation-learning flow uses simulation" | ❌ [ai.robotis.com/omx/imitation_learning_omx.html](https://ai.robotis.com/omx/imitation_learning_omx.html) collects on **real hardware** via a leader device. **Isaac Lab / cyclo_lab are not mentioned.** Note also it uses **two WRIST cameras** (`observation.images.camera1/2`) — a different configuration from ours |

> 🔴 **This is the same trap the project has hit before: OMX ≠ OMY ≠ OpenMANIPULATOR-X.**
> `decisions.md` already records one instance (memory 模式一). **The Isaac Lab assets are for the
> Y-series arm, not ours.**

⚠️ **Limits of this verification:** the above comes from reading repo landing pages and docs
summaries, **not from cloning the repos and listing their trees.** **Before deciding, someone should
clone `open_manipulator` and `cyclo_lab` and check for an OMX URDF and any OMX USD.** ~30 min.

### What the work would actually be, if the premise holds

Not "import a file". At minimum:

1. **URDF → Isaac Sim import** (the URDF Importer), then fix up joint limits, inertias, collision meshes
2. **Actuator fidelity.** DYNAMIXEL datasheets give torque/speed/gear ratio, and URDF carries
   `effort`/`velocity` limits — **but nominal limits are not dynamics.** Friction, backlash,
   control-loop latency are the usual sim2real killers, and none of them are in a URDF. `[AI推論]`
3. **Scene build**: table, the three objects, the bin, lighting
4. **Teleoperation plumbing**: leader arm → sim follower (Isaac Lab has a teleop/Mimic path, but it is
   written for its own supported robots)
5. **Dataset-format alignment** to the LeRobot schema, or the sim data cannot be pooled/compared
6. **Domain randomisation** and a sim2real gap evaluation

### ❓ Eric's question: cameras — is matching the spec enough, or is a model file needed?

**`[AI推論]`, based on how Isaac Sim camera sensors work — not line-by-line verified against Isaac Lab docs:**

- To **render an image**, Isaac Sim needs **intrinsics** (FOV / focal length, resolution, distortion)
  and **extrinsics** (pose). **A 3D model of the camera body is NOT required.**
- A geometry model is only needed if the camera body is **visible in another camera's view** or
  **occludes** something — e.g. the third-person camera seeing the wrist camera.
- 🔴 **But matching the spec only aligns the geometry. It does not close the appearance gap.**
  What actually decides whether sim data transfers is the **rendering distribution**: lighting,
  materials, sensor noise, exposure, motion blur, rolling shutter.
  **A spec-matched camera in a clean synthetic scene still produces images from a different
  distribution than a real D435i in the lab.**
- **→ So the answer is: spec is enough to *build* it, and not enough to make it *transfer*.**

### 🔴 The decision this really requires: it is a partial reversal of D007

**D007 says: no simulation, do real-hardware imitation learning.** This proposal adds simulation back.
**That is a reversal and must be argued as one, not slipped in as an addition.**

**Schedule reality as of 2026-08-27:** first lab day is **2026-09-06 at the earliest** (Eric is in
Taipei 8/29–9/4, moving in ~9/5). 書審 freezes **9/26**. Steps 1–6 above are multi-week work with
**no OMX example to copy from**, because cyclo_lab is OMY.

**→ Decision rule for Eric:** *does simulation data buy anything before 2027-02-28 that the real arm
cannot?* If the answer is "more data volume / more variations", note that **Phase C (outdoor
generalisation) is the advisor-mandated priority (D008), and outdoor lighting is exactly what
synthetic rendering is worst at.**


### 🔴 2026-08-27 二次修正 — 我上一輪過度悲觀了 `[Eric說]` 提出的三點都成立

**Eric 指出：**
1. **OMX 有 URDF**，路徑明確：
   [`open_manipulator_description/urdf/omx_f`](https://github.com/ROBOTIS-GIT/open_manipulator/tree/main/open_manipulator_description/urdf/omx_f)
2. **Isaac Sim 本身有 URDF → USD 匯入功能**
3. **NVIDIA 有一份直接對口的教學**：
   [Sim-to-Real Strategy 1: Domain Randomization（SO-101）](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/09-strategy1-dr-teleop.html)
   ——**任務不同，但流程同構**（URDF 匯入 → 場景 → 遙操作蒐集 → domain randomization → sim2real）
4. **cyclo_lab 雖然是 OMY，仍可推敲 OMX 的配置**——環境結構、感測器掛法、任務定義都是可抄的骨架

**這四點我接受。我上一輪把「沒有現成 OMX USD 資產」寫得像是路不通，那是過度推論。**
**正確的說法是：沒有現成資產＝要多做匯入與環境搭建，但路是通的，而且有兩份可抄的範本
（cyclo_lab 的 OMY 環境結構 ＋ NVIDIA 的 SO-101 sim2real 教學）。**
**風險從「可行性未知」降為「工時未知」。**

### ✅ 2026-08-27 `[Eric決定]`：做，但排在真實資料之後

> **原話：「D025 Isaac Sim 要做，在有真實資料後附加」**

**這個排序解決了 D007 的衝突，而且解得漂亮，理由值得寫下來：**

D007 說「不走模擬」，原因是**模擬驗證不了實機**。
**但「先有實機資料，再用模擬擴充」不是 D007 反對的那件事。**
D007 反對的是**用模擬取代實機**；這裡是**用實機錨定模擬**。

| | D007 反對的 | 本案 |
|---|---|---|
| 順序 | 模擬 → 實機（sim2real，驗不了） | **實機 → 模擬（實機是 ground truth）** |
| 模擬的角色 | 主要資料來源 | **擴充與變因掃描** |
| 失敗時怎麼歸因 | 不知道是模擬錯還是策略錯 | **有實機基準可比對** |

**→ D007 不需要反轉。本案是 D007 的補集，不是它的例外。** 這一條要寫進計畫書。

**執行前提（硬性）：**
1. **必須先有 Phase B 的實機基準成功率**，否則模擬資料無從校準
2. **sim 資料與實機資料是兩份資料集，不得直接混用**——除非做了 domain adaptation 並明確記錄
3. **不佔用 9/26 前的任何工時。** 這是 Phase B 之後的工作

**待驗證（clone 下來才知道，約 30 分鐘）：**
- `omx_f` URDF 的 inertial／collision 是否完整（很多 ROS URDF 的 inertia 是佔位值）
- DYNAMIXEL 的 effort/velocity 限制是否寫進 URDF，還是要另外從規格書補
- cyclo_lab 的環境定義能不能直接換 robot asset，還是綁死 OMY


- **Next step:** clone the two repos and check the three items above.
- ~~**Next step before this can be decided:** clone the two repos, confirm whether an OMX URDF exists and
  whether any OMX USD exists. **Until then this stays PROPOSED.**~~ → **DECIDED 2026-08-27, see above.**
- **Cross-reference:** D007, D008, D021, `docs/phase_plan.md`.

---

## D026 — Reach logger measures via FK from the OMX URDF, not a tape measure

- **Date:** 2026-08-31
- **Decision `[Eric決定]`:** Script 1 (`scripts/reach_logger.py`, spec `docs/specs/S1_reach_logger.md`)
  computes the gripper's planar `(x, y)` by **forward kinematics** from the live joint angles plus the
  `omx_f` URDF. The tape measure becomes a **fallback**, used only if FK cannot be stood up or fails
  its on-site validation.
- **Supersedes:** the "⚠️ Verified constraint" note in D023's script table that read *"Cheapest route:
  log joint state + type in a tape-measure reading. Full FK is an optimisation, not a prerequisite."*
  That was correct when placo/URDF availability was unverified; both are now resolved.
- **What changed since D023 said "tape measure":**
  1. **The URDF exists and matches our arm.** `ROBOTIS-GIT/open_manipulator`
     `open_manipulator_description/urdf/omx_f/omx_f.urdf` — 5 revolute arm joints + 2 gripper joints.
     `joint1` axis `(0,0,1)` = the vertical base yaw = `shoulder_pan`; `joint2/3/4` = `shoulder_lift`
     / `elbow_flex` / `wrist_flex` (axis `0,1,0`); `joint5` axis `(1,0,0)` = `wrist_roll`. Link
     origins are concrete metres. `[已查證 2026-08-31 讀 raw URDF]`
  2. **LeRobot ships an FK path.** `lerobot/src/lerobot/model/kinematics.py` `RobotKinematics(urdf_path,
     target_frame_name, joint_names)` → `forward_kinematics(joint_pos_deg)` returns a 4×4 pose. It
     needs the `placo` package (pyproject extra `placo-dep`). `[已查證]`
  3. **`placo` is installable.** `[Eric決定]` accepts adding it to the pinned env. It is **not**
     currently installed (`import placo` → ModuleNotFoundError, 2026-08-30).
  4. **Eric's stated reason:** the base→table frame tie and the URDF FK capability will be **reused**
     later — the placement mat (S3), placement-accuracy verification, and the mobile-base geometry
     check in D023's §2026-08-27 criterion 1 all need a validated arm URDF + transform.
- **Known URDF gaps (do not skip validating):**
  - Joint limits in `omx_f.urdf` are placeholders (`±6.28`, effort `1000`, velocity `4.8` on every
    joint). Fine for FK *geometry*; useless for dynamics. Same class of issue D025 flags for the
    inertials.
  - URDF joint zero ≠ Dynamixel factory-default encoder zero (`calibration/2026-08-24_omx_follower_arm.json`
    is all factory defaults — `homing_offset 0`). A **per-joint offset** must be measured once, on
    site, by the 5-pose validation test (`field_manual.md` §4-1 method).
- **placo vs ROS 2 — resolved for this script:** use **placo** (LeRobot-native, one `uv` extra). If
  placo will not install in the pinned env, fall back to a **hand-coded 5-transform chain** from the
  URDF link origins (~30 lines, zero new dependency) — **not** ROS 2. ROS 2 (`robot_state_publisher` +
  tf2) is the right tool for the *mobile-base* interface question (D023 §2026-08-27 criterion 6) but
  is disproportionate for computing FK of five joints in a logging script. Listed as implementation
  step 1 in S1 §3.
- **🔴 2026-08-31 outcome — placo does NOT install on the Windows laptop → hand-coded FK it is.**
  `uv pip install "placo>=0.9.6,<0.9.16" "cmeel-urdfdom>=4,<5" …` fails: `cmeel-urdfdom` has no
  Windows wheel, builds from source via CMake, and the `cmeel-console-bridge` sub-build errors out.
  This is the fallback branch D026 anticipated. **The hand-coded chain is now the implementation, not
  a contingency.** It is clean here: `omx_f.urdf` has **all joint `rpy="0 0 0"`** and 5 revolute
  joints, so FK = translate-by-origin then rotate-about-axis, five times, then the fixed EE offset.
  URDF saved to `assets/omx_f/omx_f.urdf` and read in full (`[已查證 2026-08-31]`); joint table in
  `docs/specs/S1_reach_logger.md` §2. `matplotlib` + `pytest` installed into the venv without issue.
  The LeRobot `RobotKinematics`/placo path stays documented as the route to use if the reach logger
  ever runs on a Linux box.
- **🔴 2026-08-31 — placo has NO Windows wheel at all** (every version on PyPI ships only
  macOS/manylinux + a source tarball; verified). Source build needs the whole cmeel/CMake/MSVC/Boost
  chain — disproportionate. `[Eric決定]` adopt option (a): keep the hand-coded `reach_logger/fk.py`
  and add **`urchin`** (pure-Python URDF FK, Windows wheels fine) as a **test-only** oracle —
  `tests/test_fk_vs_urchin.py` cross-checks the 4×4 transform at random joint configs to 1e-9.
  `fk.py` currently matches it exactly.
- **Accepted costs:**
  - `placo` enters the pinned environment → `docs/environment.md` gains an entry; existing
    `lerobot-teleoperate` / `lerobot-record` must be smoke-tested after the install.
  - One lab-day block (~15–20 min) spent on the 5-pose FK validation before any reach sample is
    trusted. If it fails, the session falls back to the tape measure and no lab day is lost beyond
    that block.
  - S1 is on the critical path to unfreezing A7/A8; the FK setup adds risk to that path. Mitigation:
    the tape-measure fallback keeps the lab day productive even if FK validation fails.
- **Reverse if:** the 5-pose validation shows FK EE positions disagreeing with physical measurement by
  more than ~1 cm and the cause is not a fixable constant offset → use the tape-measure fallback for
  this campaign and reconsider the URDF.
- **Cross-reference:** D023, D024, D025, `docs/specs/S1_reach_logger.md`, `docs/specs/S3_placement_mat.md`,
  `docs/environment.md`, `lerobot/src/lerobot/model/kinematics.py`.

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


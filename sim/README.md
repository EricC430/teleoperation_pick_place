# `sim/` — Isaac Sim asset pipeline for the OMX-F

Decided in `docs/decisions.md` **D029** (2026-09-03): the simulator is Isaac Sim / Isaac Lab,
simulation work is unblocked, and the leader arm plugs into the machine running the sim.
The script spec is `docs/specs/S4_sim_teleop_collect.md`.

## What is here

| File | Runs where | What it does |
|---|---|---|
| `omx_constants.py` | anywhere (pure python) | **Single source of truth**: joint limits, actuator specs, drive gains, payload. Run it directly to print the table. |
| `joint_mapping.py` | anywhere (pure python) | Real recording (degrees) ↔ sim joint (radians) conversion, shared by S4/S5. `SIGN` is `[未確認]` per joint — see its docstring. Run it directly for a self-test. |
| `convert_omx_urdf.py` | inside `isaac-lab` | URDF → USD, then patches everything the URDF gets wrong. Prints every `original -> new`. |
| `audit_usd.py` | inside `isaac-lab` | Read-only audit of any robot USD against the constants. Exit 0 = matches. |
| `fit_drive_gains.py` | inside `isaac-lab` | S5 gap 1: replays one real episode's `action` and scores it against `observation.state` for one drive-gain scale combo (plus `--sign-override` for testing a candidate joint sign). Run 2026-09-18 against uvc_60 episode 0, 5 combos — see "Two gaps that are NOT closed" §1 below for the actual numbers. Not yet closed. |
| `grasp_attach.py` | inside `isaac-lab` (imported, not run directly) | S5 gap 3: `ScriptedGraspAttach` — kinematic object attach/detach, triggered off the real recorded gripper channel. `[AI提議]`, not `[Eric決定]` — see module docstring. |
| `verify_grasp_attach.py` | inside `isaac-lab` | Smoke test for `grasp_attach.py`. Run 2026-09-18 against uvc_60 episode 0: state machine fires correctly (attach/detach at frame 226/382, and again 401/464 — a real second regrasp in the raw gripper trace, not a bug). Tests the mechanism only, not real-world grasp success — see script docstring. |
| `inspect_mimic_axis.py` | inside `isaac-lab` (no `--enable_cameras` needed) | Read-only: prints the actual `RevoluteJoint.axis` and `PhysxMimicJointAPI` attributes for the gripper joints. Runs in seconds — use this before reaching for `verify_mimic_gearing.py` to check a hypothesis about the USD's own contents. |
| `verify_mimic_gearing.py` | inside `isaac-lab` | S5 gap 2: commands `gripper_joint_1` to two poses, reads back `gripper_joint_2`, renders a close-up (or `--no-render` for a fast numbers-only check). Run 2026-09-18, closed the gap — see "Two gaps" §2 below. |
| `run_in_container.sh` | host | Copies `sim/*.py` into `~/isaaclab_volume/omx_sim/` and runs one of them in the container. **Does not copy dataset parquet files** — `fit_drive_gains.py`/`verify_grasp_attach.py` need that copied in separately, see their docstrings. |

## Usage

```bash
GUEST=/workspace/test_isaaclab

# regenerate the asset (reproducible; safe to re-run)
./sim/run_in_container.sh convert_omx_urdf.py \
    --urdf $GUEST/assets/open_manipulator_description/urdf/omx_f/omx_f.urdf \
    --out  $GUEST/assets/omx_f_generated/omx_f.usd --headless

# check any USD against the constants
./sim/run_in_container.sh audit_usd.py --usd $GUEST/assets/omx_f_generated/omx_f.usd --headless
```

The generated asset lives under `~/isaaclab_volume/assets/omx_f_generated/` — **outside git, on
purpose**. It is a build artefact: the reproducible things are this directory and the URDF.

## Why the GUI import was replaced

`assets/wildbot_with_omxaiarm.usd` was produced through the Isaac Sim GUI on 2026-08-28. Audited
2026-09-03 with `audit_usd.py`:

| Item | GUI import (2026-08-28) | Scripted re-conversion |
|---|---|---|
| joint position limits | 🔴 ±360° on all six (URDF placeholders) | factory travel ∩ camera-rig sector |
| drive `maxForce` | 🔴 1000 N·m on all six | 1.5 / 0.52 / 0.20 N·m per motor |
| max joint velocity | 🔴 275 °/s on all six (importer default) | 366 / 618 / 2100 °/s per motor |
| drive stiffness | 🔴 0.41 / 1.38 / 4.58 / 3.39 / 0.27 / 0.03 — no relation to the motors | derived from stall torque |
| drive damping | 🔴 **0.0 on every joint** — an undamped position drive | 5 % of stiffness |
| collision approximation | 🔴 14 × `convexHull` (arm) + 8 × `convexDecomposition` | 8 × `convexDecomposition`, no hulls |
| mimic on `gripper_joint_2` | ✅ present | ✅ present, **but see "Two gaps" §2 below — "present" alone did not mean "correct"** |
| articulation roots | ⚠️ **two** — `/World/car/...` and `/World/omx_f/...` | one |
| total mass | 54.24 kg (car included) | **0.5588 kg** ↔ ROBOTIS quotes 560 g |

Two things worth stating plainly:

- **The arm and the car are two separate articulations with no joint between them.** The 8/28 asset
  places the arm above the car; it does not mount it. Nothing is wrong with that — it matches what
  it was made for — but it is not a mobile-manipulator asset, and D020 has not chosen a chassis yet.
- **`damping = 0` everywhere is the defect that would have cost the most time.** An undamped position
  drive oscillates; the failure looks like "the sim is unstable" rather than "a parameter is wrong".

## Scene: `scene_constants.py` + `omx_scene_cfg.py` + `preview_scene.py`

Table, arm, one seeded-placement object, a bin, two cameras in dataset order
(`observation.images.wrist` then `.front-left`). Verified end-to-end 2026-09-03:

```
./sim/run_in_container.sh preview_scene.py --headless --enable_cameras \
    --placements docs/assets/placement_label_map_campA_20260831.csv --place t1 --out <dir>
```

**What this run proves, and no more:** the converted arm loads as one articulation, the seeded
placement CSV drives object position in the sim frame (same `placement_id` a real episode would
carry — the whole point of reading that file instead of hand-picking a position), both cameras
render at the recording resolution in the declared order. It does **not** prove geometric
alignment with the real cell — `experiment_spec.md` §3 is still blank, so `scene_constants.py`'s
table/camera/lighting numbers are explicitly marked PLACEHOLDER, not measured.

### 🔴 Bug found and fixed: `trash_obj/*.usd` needs `scale=(0.01, 0.01, 0.01)`

First run: object placed 6 cm above the table settled at **z = 357.6 cm** — through the table and
into orbit. Root cause, confirmed with `inspect_object_usd.py` (a bbox query, no physics) on all
8 objects in `assets/trash_obj/`: every one is authored at **`metersPerUnit = 0.01`** (its own
coordinates are centimetres) while Isaac Sim's world stage is metres. USD reference composition
does not auto-correct for a `metersPerUnit` mismatch between stages — an 8 cm can gets placed as
an 8-**metre** object and explodes on first contact. Fixed by scaling every `trash_obj` spawn by
0.01; re-verified — the object now settles at z = 79.0 cm (4 cm above the 75 cm table top, exactly
where a can resting on the table should be). This is a property of the whole asset family (all 8
checked objects agree), not a per-object guess.

### ⚠️ Confirmed, not fixed: the provisional drive gains cannot hold the arm's own weight

Commanding the all-zero joint pose and holding it for 120 steps (1 s) produces **19–35° of drift**
depending on run (object-explosion shockwaves in the first run likely inflated that instance's
number; 19° is the cleaner reading). Visible in the front-left render: the arm visibly droops.
This is exactly the gap `omx_constants.py`'s docstring and D029 already flagged — the gains are
`stiffness = stall_torque / 5°`, dimensionally honest but not a calibration — now with a concrete
number attached. **Do not close this by guessing a bigger `GAIN_TRACKING_ERROR_RAD`.** Two real
possibilities that a guess can't distinguish: (a) the gains are simply too soft, or (b) all-zero
joint angles is not a mechanically easy pose for this arm (e.g. if it corresponds to something
closer to "arm extended horizontally" than "arm folded home") and the real XL330 holds it via
margin the stall-torque figure alone doesn't capture. **Needs a real recorded trajectory to fit
against (D029), not another constant tweak.**

### ⚠️ Non-fatal: `Not all actuators are configured! ... 6 != 7`

`gripper_joint_2` (the mimic follower) has no entry in the `ArticulationCfg.actuators` dict —
deliberate: it is driven by the `PhysxMimicJointAPI` constraint applied in `convert_omx_urdf.py`,
not by its own PD drive. Giving it a second, independent actuator would fight the mimic
constraint. Isaac Lab's warning is generic (it does not know about the mimic relationship) and can
be ignored for this joint specifically — but if a NEW joint ever shows up in this warning, that
one probably does need an actuator entry.

## One gap closed, one still not, plus a new one it exposed

1. 🔴 **Drive gains are provisional, and now there's a real number attached.** `fit_drive_gains.py`
   (S5 gap 1) replayed uvc_60 episode 0 (534 frames) open-loop against 5 gain/sign combos in the
   `isaac-lab` container, 2026-09-18:

   | run | stiffness×/damping×/effort× | `shoulder_lift` p50/p95/max (deg) | `elbow_flex` p50/p95/max (deg) |
   |---|---|---|---|
   | baseline | 1× / 1× / 1× | 103.9 / 151.4 / 151.6 | 23.9 / 61.9 / 73.6 |
   | | 20× / 20× / 1× (ratio-preserved) | 103.9 / 151.4 / 151.6 | 23.4 / 62.3 / 74.4 |
   | `--sign-override shoulder_lift=-1` | 1× / 1× / 1× | 73.6 / 127.2 / 127.4 | 24.9 / 62.8 / 92.6 |
   | **`--effort-scale 10`** | 4× / 4× / **10×** | **6.4 / 9.8 / 26.8** | **8.0 / 9.2 / 13.0** |

   The other four joints all track to p50 < 2.1° regardless of scale. **The real finding, found in
   this order:** (1) a 20× stiffness range with the damping ratio held fixed does not move the
   error at all, which rules out "just needs a bigger gain" — but that sweep only ever touched
   `stiffness`/`damping`, never `effort_limit` (capped at the real motor's rated stall torque,
   `omx_scene_cfg.py`); (2) `omx_constants.py`'s own gain law is "full torque at 5° of lag", so any
   real dynamic segment with more than ~5° of lag is *already* commanding the torque ceiling at 1×
   stiffness — scaling stiffness further changes nothing because the ceiling, not the gain below
   it, is what's binding; (3) the new `--effort-scale` flag confirms this directly: raising the
   ceiling 10× (to ~5.2 N·m on a 0.52 N·m motor) collapsed `shoulder_lift`'s p50 from 104° to 6.4°
   and `elbow_flex`'s from 24° to 8.0°, **with SIGN left at the default +1**. That also makes the
   `--sign-override` row above look like a confound, not a real fix — independently confirmed by
   working out `shoulder_lift`'s physical direction from `reach_logger/fk.py` (URDF geometry,
   itself checked against real tape measurements in S1): positive angle numerically lowers the
   elbow, negative raises it, and the real episode's `shoulder_lift` reading goes +37°→-6° exactly
   across the grasp-then-lift transition (frame ~226, matching gap 3's own detected attach frame)
   — consistent with SIGN=+1 being correct, not a coincidence of picking the "wrong" combo that
   happened to need less torque.

   **🔴 That framing was wrong, and Eric said so the same day.** The writeup above ended by asking
   whether trading real motor torque (5.2 N·m on a motor rated for 0.52 N·m) for tracking fidelity
   was acceptable for S5's purposes. `[Eric說 2026-09-18]`: *"即使以錄製的軌跡放到模擬環境，馬達的
   規格仍然要真實才能產生對應的物理畫面吧"* — correct, and it invalidates the question rather than
   answering it. The real mistake was upstream: driving the replay with PD position control at all,
   which forces the sim to *re-derive* an arm trajectory that is already recorded, frame by frame,
   in `observation.state`. **S5's arm is now a kinematic replay**
   (`Articulation.write_joint_position_to_sim`, not `set_joint_position_target`) —
   `[Eric決定 2026-09-18]`, see `docs/specs/S5_sim_replay_augmentation.md` §4. Gains and effort
   limits do not participate in the rendered arm pose there at all, so the trade-off question
   disappears rather than being resolved.

   **Where this work still lives:** gap 1 remains a hard dependency for **S4** (live teleop-in-sim
   has no already-recorded outcome to copy — the sim really does have to control the arm in real
   time), so `fit_drive_gains.py` and the numbers above are not discarded, just re-homed.

   **✅ The SIGN half is now settled for `shoulder_lift`, without a lab day.** Once the arm became
   a kinematic replay, checking the sign stopped needing hardware: pose the sim at the recorded
   `observation.state` and put the render beside the real recorded video at the same timestamp
   (`render_state_replay.py` + `scripts/compare_sim_real_frames.py`). At episode 0 frame 226, where
   the real arm is reaching down to the cup, SIGN=+1 renders the arm extended forward at table
   height (matches) and the `--sign-override shoulder_lift=-1` **control** renders it pointing
   nearly straight up (grossly wrong). The control is what makes this evidence rather than a
   vibe — it shows the test can discriminate. Agrees with both earlier independent arguments.
   The other five joints show no mismatch across six frames but have **no control run of their
   own**; weakest for `wrist_roll` (subtle visual effect) and `gripper` (amplitude already known
   wrong, gap 2 residual). See `joint_mapping.py`'s docstring for the per-joint evidence level.
2. ✅ **The mimic gearing sign is CLOSED — `gearing=1.0` (the default) is correct.** Run 2026-09-18
   with `verify_mimic_gearing.py` (both `--no-render` numbers and a camera render):

   ```
   gearing=+1.0 -> gripper_joint_1 +89.9deg, gripper_joint_2 ~ -45deg  (mirrored -- correct)
   gearing=-1.0 -> gripper_joint_1 +89.9deg, gripper_joint_2 ~ +47deg  (same-direction -- wrong)
   ```

   Renders confirm it visually: at `gearing=-1.0`'s value one finger swings out and the other stays
   put; at `+1.0` both fingers open symmetrically. **But the sign was never actually the blocker** —
   both signs initially showed `gripper_joint_2` moving <0.15° while `gripper_joint_1` swept 90°, and
   two real bugs in `convert_omx_urdf.py` had to be fixed before either sign produced real motion:

   - `referenceJointAxis` was `"rotX"`; `inspect_mimic_axis.py` showed both gripper joints'
     `RevoluteJoint.axis` is actually `Z` (matches the URDF's `<axis xyz="0 0 1"/>`). Tracking a
     reference axis the master joint doesn't rotate about made the reference quantity ~constant
     regardless of gearing. Fixed to `"rotZ"`.
   - `gripper_joint_2` was given `gripper_joint_1`'s own (nonzero) drive stiffness/damping "so the
     pair is consistent" — but the URDF converter turns that into an actual USD-level PD position
     drive targeting the joint's initial pose (0 rad). That drive (comparatively stiff) fought the
     mimic constraint (`naturalFrequency=25`, `dampingRatio=0.005` — soft, carried over unchanged
     from the 8/28 GUI import) and mostly won. **This was the dominant cause**, not the axis. Fixed
     by zeroing `gripper_joint_2`'s own stiffness/damping — it is now genuinely undriven, governed
     only by the mimic constraint, matching what the "Non-fatal" note below already assumed was true.

   ⚠️ **New gap this exposed, not yet closed:** even fixed, `gripper_joint_2`'s swing is only ~50% of
   `gripper_joint_1`'s (ratio ≈ -0.5, stable across 90 vs 300 settle steps — not a convergence delay).
   Likely the same never-calibrated `naturalFrequency`/`dampingRatio` pair, or an unpatched limit/force
   on `gripper_joint_2` itself (it isn't in `K.JOINTS`, so the `convert_omx_urdf.py` limit-patching loop
   skips it). Means a sim replay's gripper currently opens/closes at roughly half the real recorded
   amplitude. Scope: closer to gap 1 (gain calibration) than gap 2 (direction) — tracked separately.
   Full detail: `docs/specs/S5_sim_replay_augmentation.md` §2's 2026-09-18 update.

## 🔴 `simulation_app.close()` does not reliably end the process

Observed 2026-09-18 across multiple scripts (`fit_drive_gains.py`, `verify_grasp_attach.py`, and
independently, on a different machine, `verify_mimic_gearing.py` — every one of its four runs had
to be `kill -9`'d): the script finishes its work, writes its output file, calls
`simulation_app.close()` — and the `/isaac-sim/kit/python/bin/python3` process keeps running and
burning CPU anyway (one instance sat at 3.5 CPU-hours after its output was already on disk).
**Happens with `--no-render` too** — no camera/RTX involvement required to trigger it, so it isn't
specific to the RTX shutdown path. This is environment behaviour, not a bug in any one script. **When running
anything in `sim/` inside the container: poll for the output file's existence
(`docker exec isaac-lab test -f <path>`), don't wait for the shell command to return — then
`docker exec isaac-lab pkill -9 -f <script.py>` once the output is there.**

## Units trap (cost one debugging round)

`omx_constants.py` works in **radians** (the convention the motor datasheets use). `UsdPhysics`
angular drives store stiffness and damping **per degree**. The importer converts; the audit must
divide by 180/π before comparing. Skip that and a correct asset reads as 57× wrong.

# `sim/` — Isaac Sim asset pipeline for the OMX-F

Decided in `docs/decisions.md` **D029** (2026-09-03): the simulator is Isaac Sim / Isaac Lab,
simulation work is unblocked, and the leader arm plugs into the machine running the sim.
The script spec is `docs/specs/S4_sim_teleop_collect.md`.

## What is here

| File | Runs where | What it does |
|---|---|---|
| `omx_constants.py` | anywhere (pure python) | **Single source of truth**: joint limits, actuator specs, drive gains, payload. Run it directly to print the table. |
| `convert_omx_urdf.py` | inside `isaac-lab` | URDF → USD, then patches everything the URDF gets wrong. Prints every `original -> new`. |
| `audit_usd.py` | inside `isaac-lab` | Read-only audit of any robot USD against the constants. Exit 0 = matches. |
| `run_in_container.sh` | host | Copies `sim/*.py` into `~/isaaclab_volume/omx_sim/` and runs one of them in the container. |

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
| mimic on `gripper_joint_2` | ✅ present | ✅ present (re-applied in post-processing) |
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

## Two gaps that are NOT closed

1. 🔴 **Drive gains are provisional.** `stiffness = stall_torque / 5°`, `damping = 0.05 × stiffness`
   is dimensionally honest and reproducible, but it is not a calibration. Closing this needs a fit
   against real recorded trajectories (D029). Until then, do not claim the simulated arm's dynamics
   resemble the real one's.
2. 🔴 **The mimic gearing sign is unverified.** The URDF says `multiplier="-1"`; the USD is written
   with `gearing=1.0`, reproducing what Isaac Sim's own GUI importer produced from this same URDF.
   URDF's mimic tag and PhysX's mimic constraint do not share a sign convention, so neither value can
   be trusted from the spec alone. **The gripper open/close row of the 5-pose test (S4 §5-1) settles
   it** — `--mimic-gearing` is a flag for exactly that reason.

## Units trap (cost one debugging round)

`omx_constants.py` works in **radians** (the convention the motor datasheets use). `UsdPhysics`
angular drives store stiffness and damping **per degree**. The importer converts; the audit must
divide by 180/π before comparing. Skip that and a correct asset reads as 57× wrong.

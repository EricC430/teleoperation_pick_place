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

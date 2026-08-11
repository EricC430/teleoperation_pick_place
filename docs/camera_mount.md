# Third-person camera mounting (mobile base)

**Status: not needed until P3.** P0–P2 run on a fixed indoor tabletop — use a tripod.
Do not build a mount yet. This document records the design constraints and the
unresolved team disagreement so that neither gets lost before P3.

---

## 1. What can be done *now*, without hardware

**Workspace envelope analysis in simulation.** Sweep the arm through its full joint
range, record the volume it passes through. The camera and its mount must sit
entirely outside that envelope.

Doing this answers both open questions at once — "will it block the arm?" and
"will it restrict joint travel?" — and it costs nothing while the hardware is
unavailable.

---

## 2. The unresolved disagreement (record it, don't argue it)

### Option A — rear mast, looking down at 30–45°

| | |
|---|---|
| ✅ | Background is the ground plane: relatively controlled. Critical outdoors. |
| ✅ | Arm appears *within* the frame rather than *in front of* the lens |
| ❌ | **Cantilever tip deflection scales with length³** (δ = FL³/3EI). Halve the mast height, cut deflection by 8×. Tall masts resonate. |
| ❌ | Both base motion and the arm's own reaction torque transmit up the mast |

### Option B — front-side, oblique forward view

| | |
|---|---|
| ✅ | Low, stiff, minimal vibration |
| ❌ | **Field of view includes the horizon, distant scenery, pedestrians, vehicles** → enormous outdoor domain variability |
| ❌ | Poor depth discrimination along the viewing axis |
| ❌ | Arm may cross the line of sight when reaching forward — needs envelope analysis to confirm |

### The design that may dissolve the argument

> Extrinsic error is the error of **the camera relative to the arm base**, not the
> camera's absolute shake.
>
> If the camera mount and the arm base are **bolted to the same rigid plate**, chassis
> vibration moves both together and the relative pose is unchanged. Extrinsic drift comes
> from "camera bolted to chassis point A, arm bolted to point B", coupled through a
> compliant frame.

Add **triangulated bracing** (a single post is compliant; a triangle is stiff) and
**shorten the cantilever** (camera close to the arm base on a short post with downtilt,
rather than a tall mast) — and the top-down view survives.

### A separate point worth more than the mount debate

**Vibration is not necessarily fatal; distribution mismatch is.**

If demos are recorded on the vehicle (with vibration) and deployment is on the vehicle,
vibration is part of the training distribution and the policy learns robustness to it.

**What is fatal: recording on a stable tabletop and deploying to a vibrating vehicle.**

→ If the end target is a mobile base, **P3-onward data should be collected on the vehicle.**

---

## 3. Settle it by measurement, not by argument

```
Place a fixed marker (checkerboard or ArUco) in the workspace
  ↓
For each mounting option: (a) drive the base, (b) run a typical grasp motion
  ↓
Measure the marker's pixel displacement in the camera image
  ↓
Criterion: how many pixels? As a fraction of the target object's size in frame?
```

| Result | Verdict |
|---|---|
| ~2–3 px displacement, target spans ~80 px | Not a problem. Argument dissolved. |
| ~30 px | Option B is correct. |

> This is the one case where ArUco is worth pulling forward from P2 — the measurement
> needs sub-pixel marker localization and nothing cheaper does it well.

---

## 4. Build order (do not reorder)

1. **Workspace envelope analysis** (simulation) — camera must fall outside it
2. **Field-of-view verification** — must cover: object start region, place region, **and the
   gripper's position during approach** (the one most often missed)
3. **Physical check with a tripod** at the intended vehicle-relative position — verify
   occlusion and coverage *before* fabricating anything
4. **Then** build the mount

---

## 5. Five hard requirements

**1. Rigidity above all else**
- Aluminium extrusion (20×20 T-slot) + corner brackets for the primary structure
- 3D-printed parts only for the camera clamp, never structural
- **No ball heads or gimbal mounts** — they loosen and are the primary source of extrinsic drift
- Avoid long cantilevers
- Bolt to chassis structure, not to the shell
- Thread-locker on fasteners

**2. Repeatable remounting** ← most often overlooked, highest cost when it bites
- Use dowel pins or machined holes; never eyeball the alignment
- **Extrinsics change → the trained policy is invalid.** This is the most painful way to lose
  a few hundred collected demos
- Photograph the baseline after every remount, store in `docs/setup_env.md`

**3. Cable routing**
- Cables **must not enter the arm's motion envelope**. The workshop material warns that a
  cable caught between joint links causes calibration to record wrong limits, invalidating
  the entire calibration

**4. Outdoor lighting**
- Fit a lens hood against direct sun
- **Consider locking exposure and white balance** — auto-exposure makes the same scene look
  very different at different times of day, widening the domain gap. This may matter more
  than the mount design itself

**5. Stop the vehicle to grasp**
- Vibration is acceptable while moving; the grasp phase must be static.
  Standard practice in mobile manipulation.

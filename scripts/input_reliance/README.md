# Which input decides where ACT grasps (2026-10-07)

Offline tests behind `docs/meeting/2026-10-07.md`. Run from the repo root with `uv run python scripts/input_reliance/<file>`.
Checkpoint and dataset paths are hard-coded at the top of `h1_swap.py` (cup 100k, uvc_60, eval-open).

| file | what it does | notes §|
|---|---|---|
| `h2_shrink.py` | open-loop: plan made D frames before the human grasp vs the human's grasp position → radial/angle slope | §2 |
| `h1_swap.py` | swap one input (wrist / front-left / both / state) with another episode's, measure how far the plan follows | §3-1 |
| `h1_swap2.py` | same at 1/3 s and 1 s before the grasp, plus wrist → gray | §3-1 |
| `h1_rollout.py` | misaligned closed-loop frames (`bad_aim`): wrist → an aligned success frame; does the plan move? | §3-2 |

All map joints to table positions with the human joint→(θ, r) fit from `scripts/aim_error.py`.

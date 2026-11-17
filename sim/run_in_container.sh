#!/usr/bin/env bash
# Run a sim/ script inside the isaac-lab container.
#
# The container binds ~/isaaclab_volume -> /workspace/test_isaaclab. This repo is NOT mounted,
# so the sim/ sources are copied in first. Source of truth stays here, in git; the copy under
# isaaclab_volume/omx_sim/ is a build artefact and may be deleted at any time.
set -euo pipefail

CONTAINER=${CONTAINER:-isaac-lab}
HOST_VOLUME=${HOST_VOLUME:-$HOME/isaaclab_volume}
GUEST_VOLUME=/workspace/test_isaaclab
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ]; then
  echo "usage: $0 <script.py> [args...]" >&2
  echo "  e.g. $0 convert_omx_urdf.py --urdf \$GUEST/assets/open_manipulator_description/urdf/omx_f/omx_f.urdf \\" >&2
  echo "                              --out  \$GUEST/assets/omx_f_generated/omx_f.usd --headless" >&2
  exit 2
fi

mkdir -p "$HOST_VOLUME/omx_sim"
cp "$HERE"/*.py "$HOST_VOLUME/omx_sim/"
# the seeded placement list travels with the code: sim and real share placement ids
cp "$HERE"/../docs/assets/placement_label_map_*.csv "$HOST_VOLUME/omx_sim/" 2>/dev/null || true

# a light mirror of the repo for scripts that look things up under it (physics_replay.py: $OMX_REPO): the code,
# reach_logger, the calibration JSONs (no photos) and the small assets. The generated robot USD lives in the volume
# already (assets/omx_f_generated, from convert_omx_urdf.py / patch_omx_usd.py) and is linked, not copied.
MIRROR="$HOST_VOLUME/omx_repo"
mkdir -p "$MIRROR/sim" "$MIRROR/calibration" "$MIRROR/assets"
cp "$HERE"/*.py "$MIRROR/sim/"
cp -r "$HERE/../reach_logger" "$MIRROR/"
cp "$HERE"/../calibration/*.json "$MIRROR/calibration/"
cp -r "$HERE/../assets/paper_cup" "$HERE/../assets/omx_f" "$MIRROR/assets/"   # omx_f: the black arm material (dr_render.py)
cp "$HERE"/dr_*.json "$MIRROR/sim/" 2>/dev/null || true
[ -e "$MIRROR/assets/omx_f_generated" ] || ln -s ../../assets/omx_f_generated "$MIRROR/assets/omx_f_generated"

SCRIPT="$1"; shift
INTERACTIVE_FLAG=""
if [ -t 0 ]; then
  INTERACTIVE_FLAG="-i"
fi
# Isaac Lab 3.0 container (isaac-lab-base image, Isaac Sim 6.2.0): two ways to start Python.
#   * scripts that import isaaclab (they all `import isaac_boot`) -> `isaaclab.sh -p`, which sources the Isaac Sim env and
#     uses the uv venv (/opt/isaaclab-venv, Python 3.12). Calling the venv python directly fails: "Isaac Sim is not installed".
#   * plain Isaac Sim scripts (SimulationApp, no Isaac Lab: physics_replay.py, dr_render.py, ...) -> Isaac Sim's python.sh.
# Override with RUNNER=isaaclab|isaacsim.
if [ -z "${RUNNER:-}" ]; then
  if grep -qE '^\s*(import isaac_boot|from isaaclab)' "$HERE/$SCRIPT" 2>/dev/null; then RUNNER=isaaclab; else RUNNER=isaacsim; fi
fi
case "$RUNNER" in
  isaaclab) PY=/workspace/isaaclab/isaaclab.sh; PYARGS="-p" ;;
  isaacsim) PY=/isaac-sim/python.sh; PYARGS="" ;;
  *) echo "RUNNER must be isaaclab or isaacsim" >&2; exit 2 ;;
esac
exec docker exec $INTERACTIVE_FLAG -e PYTHONUNBUFFERED=1 -e OMX_REPO="$GUEST_VOLUME/omx_repo" "$CONTAINER" $PY $PYARGS "$GUEST_VOLUME/omx_sim/$SCRIPT" "$@"

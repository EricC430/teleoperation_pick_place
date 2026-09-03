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

SCRIPT="$1"; shift
echo "[run_in_container] $CONTAINER : /isaac-sim/python.sh $GUEST_VOLUME/omx_sim/$SCRIPT $*"
exec docker exec -i -e PYTHONUNBUFFERED=1 "$CONTAINER" /isaac-sim/python.sh "$GUEST_VOLUME/omx_sim/$SCRIPT" "$@"

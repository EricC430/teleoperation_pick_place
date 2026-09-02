#!/usr/bin/env bash
#
# Open a shell (or run a command) in the LeRobot GPU container on the lab box.
#
#   ./scripts/run_container.sh                  # interactive shell
#   ./scripts/run_container.sh python --version # run one command and exit
#
# A second person can attach to the SAME running container from another terminal:
#   docker exec -it lerobot-dev bash
#
# The CUDA flags below are NOT optional on this machine and NOT cosmetic.
# docs/environment.md explains why (driver caps at CUDA 12.4, image wants 12.8,
# and the image's bundled forward-compat libs don't work on GeForce cards).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE="${LEROBOT_IMAGE:-huggingface/lerobot-gpu:latest}"
NAME="${LEROBOT_CONTAINER_NAME:-lerobot-dev}"

# Datasets/checkpoints land here. Gitignored -- see .gitignore and
# docs/experiment_spec.md section 12: data never goes into git.
mkdir -p "$REPO_ROOT/data"

# Only pass -it when there's actually a terminal, so the script also works in
# pipelines and non-interactive checks.
TTY_FLAGS=()
if [ -t 0 ]; then TTY_FLAGS=(-it); fi

exec docker run --rm "${TTY_FLAGS[@]}" \
  --name "$NAME" \
  --gpus all \
  --shm-size 16gb \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  --tmpfs /usr/local/cuda/compat \
  --user "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v "$REPO_ROOT":/workspace \
  -w /workspace \
  -e HOME=/workspace/data \
  -e HF_HOME=/workspace/data/huggingface \
  -e HF_LEROBOT_HOME=/workspace/data/huggingface/lerobot \
  -e TORCH_HOME=/workspace/data/torch \
  -e TRITON_CACHE_DIR=/workspace/data/triton \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "$IMAGE" "$@"

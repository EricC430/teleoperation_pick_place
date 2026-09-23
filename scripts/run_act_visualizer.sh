#!/usr/bin/env bash
#
# Convenient runner for ACT Attention Loss & Layer Latent Visualizer.
#
# Usage:
#   ./scripts/run_act_visualizer.sh [options]
#
# Options:
#   --checkpoint <path>      Model checkpoint path (default: phase_b1_uvc60 100k)
#   --dataset <repo_id>      Dataset repo id (default: ericc430/omx_pick_place_open_loop_eval)
#   --root <path>            Explicit dataset root path
#   --episode <int>          Episode index to analyze (default: 0)
#   --max-frames <int>       Limit number of frames (optional)
#   --export-video           Export high-res side-by-side diagnostic MP4
#   --serve                  Start local HTTP server to view in browser
#   --port <int>             Port for HTTP server (default: 8088)
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT="data/train/phase_b1_uvc60/checkpoints/100000/pretrained_model"
DATASET="ericc430/omx_pick_place_open_loop_eval"
DATASET_ROOT="data/huggingface/lerobot/ericc430/omx_pick_place_open_loop_eval"
EPISODE=0
MAX_FRAMES=""
EXPORT_VIDEO="--export-video"
SERVE=false
PORT=8088

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --episode)
      EPISODE="$2"
      shift 2
      ;;
    --max-frames)
      MAX_FRAMES="--max-frames $2"
      shift 2
      ;;
    --export-video)
      EXPORT_VIDEO="--export-video"
      shift
      ;;
    --no-video)
      EXPORT_VIDEO=""
      shift
      ;;
    --serve)
      SERVE=true
      shift
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "=========================================================="
echo "🎯 Running ACT Attention & Layer Latent Extraction Pipeline"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset:    $DATASET (Episode $EPISODE)"
echo "=========================================================="

./scripts/run_container.sh python scripts/visualize_act_attention.py \
    --checkpoint "$CHECKPOINT" \
    --dataset.repo_id "$DATASET" \
    --dataset.root "$DATASET_ROOT" \
    --episode "$EPISODE" \
    $MAX_FRAMES \
    $EXPORT_VIDEO

echo ""
echo "✅ Extraction complete!"

if [ "$SERVE" = true ]; then
  echo ""
  echo "🌐 Starting local HTTP server on port $PORT..."
  echo "Open in your browser: http://localhost:$PORT/tools/act_visualizer/"
  python3 -m http.server "$PORT"
fi

#!/usr/bin/env python
"""S5 stage 2: assemble `sim/replay_render_episode.py`'s output into a LeRobotDataset.

Stage 1 runs in the isaac-lab container (Isaac Lab, no `lerobot`); this runs in the project's uv
environment (`lerobot`, no Isaac Lab). The boundary is forced by what is installed where, checked
2026-09-18: `/isaac-sim/python.sh` has no `lerobot`, no `av`, no `torchcodec`.

On the Linux sim machine there is no host lerobot install -- lerobot lives in the
`huggingface/lerobot-gpu` docker image (0.6.2 as of 2026-09-18). Run it there, as the host user
(otherwise it cannot read the repo) and with USER/HOME set (torch resolves a username at import and
uid 1020 is not in the image's passwd):

    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER=$USER \\
        -v "$PWD":/repo -v ~/isaaclab_volume/omx_sim:/renders -w /repo \\
        huggingface/lerobot-gpu:latest \\
        python scripts/sim_replay_augment.py \\
            --render-dir /renders/replay_ep0_seed42 \\
            --source-info /repo/data/huggingface/lerobot/ericc430/omx_pick_place_pilot_uvc_60/meta/info.json \\
            --repo-id sim_omx_replay_dr_20260918 \\
            --root /renders/s5_out/sim_omx_replay_dr_20260918

`[已查證 2026-09-18]` Run end-to-end in that image on a 14-frame smoke render: wrote the dataset,
`scripts/verify_dataset.py` exit 0, feature keys and order identical to the source. The first draft
got `add_frame` wrong -- 0.6.2 takes `add_frame(frame)` with the task INSIDE the frame dict, not a
`task=` argument -- and would have crashed; found by reading the installed version's own source.

## What it enforces (S5 §7)

- `--repo-id` must start with `sim_` -- refuses otherwise (D025 premise 2: sim and real data are
  two datasets and must not be pooled).
- Feature keys and their ORDER are taken from the SOURCE dataset's own `meta/info.json`, not from
  a hardcoded list, and the result is printed. S4 §7 is explicit that the doc has been wrong about
  this before and the recorded dataset is the authority.
- Every fidelity flag stage 1 recorded is copied into the output dataset's meta, unchanged. A
  consumer who reads only the dataset must still learn that the geometry is unaligned, the object
  is not the source episode's object, and the placement was chosen rather than reconstructed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--render-dir", required=True, help="a sim/replay_render_episode.py --out dir")
    p.add_argument("--repo-id", required=True, help="must start with 'sim_' (S5 §7)")
    p.add_argument("--root", default=None, help="local dataset root")
    p.add_argument("--source-info", default=None,
                   help="source dataset's meta/info.json, for feature-key order. Default: derived "
                        "from the manifest's source_dataset_root")
    p.add_argument("--fps", type=float, default=None, help="default: the manifest's dataset_fps")
    p.add_argument("--task", default=None,
                   help="task string. Default: read from the source dataset's meta/tasks.parquet, so "
                        "the variant carries the same task text as the recording it came from")
    p.add_argument("--dry-run", action="store_true", help="validate and print, write nothing")
    return p.parse_args()


def load_source_task(info_path: Path) -> str:
    import pyarrow.parquet as pq

    tasks = info_path.parent / "tasks.parquet"
    if not tasks.exists():
        raise SystemExit(f"no {tasks}; pass --task explicitly")
    d = pq.read_table(tasks).to_pydict()
    if len(d["task"]) != 1:
        raise SystemExit(f"{tasks} has {len(d['task'])} tasks; pass --task to pick one")
    return d["task"][0]


def load_source_features(info_path: Path) -> dict:
    """Feature keys AND their order, straight from the source dataset (S4 §7: the recorded dataset
    is the authority, not the docs)."""
    info = json.loads(info_path.read_text())
    return info["features"]


def source_encoder_kwargs(src_features: dict, image_keys: list[str]) -> dict:
    """Rebuild the source's video encoder settings from its own info.json.

    lerobot 0.6.2 defaults to SVT-AV1 / crf 30; the uvc_60 recordings are h264 / crf 23 / veryfast
    (configs/record_omx.yaml -- h264 was chosen because AV1 starved the D455 grab thread on the
    recording laptop). A "visual variant of the source" encoded with a different codec is one more
    silent difference between sim and real, so match it. Read from info.json rather than hardcoded
    so a future campaign with different settings is followed automatically.
    """
    vi = src_features[image_keys[0]].get("info", {})
    kw = {
        "vcodec": vi.get("video.codec"),
        "pix_fmt": vi.get("video.pix_fmt"),
        "g": vi.get("video.g"),
        "crf": vi.get("video.crf"),
        "preset": vi.get("video.preset"),
        "fast_decode": vi.get("video.fast_decode"),
        "video_backend": vi.get("video.video_backend"),
    }
    return {k: v for k, v in kw.items() if v is not None}


def main():
    args = parse_args()

    if not args.repo_id.startswith("sim_"):
        raise SystemExit(
            f"refusing repo_id {args.repo_id!r}: S5 §7 / D025 premise 2 require a 'sim_' prefix so "
            "simulated data can never be mistaken for, or pooled with, real recordings"
        )

    render_dir = Path(args.render_dir)
    manifest = json.loads((render_dir / "manifest.json").read_text())
    fps = args.fps or manifest["dataset_fps"]

    info_path = Path(args.source_info) if args.source_info else Path(manifest["source_dataset_root"]) / "meta" / "info.json"
    if not info_path.exists():
        raise SystemExit(
            f"source info.json not found at {info_path}. Stage 1 recorded the container's path for "
            "source_dataset_root; pass --source-info with this machine's path instead."
        )
    src_features = load_source_features(info_path)
    image_keys = [k for k in src_features if k.startswith("observation.images.")]
    task = args.task or load_source_task(info_path)
    print(f"source feature order: {list(src_features)}")
    print(f"  image keys, in order: {image_keys}")
    print(f"  task: {task!r}")

    print("\nfidelity flags carried from stage 1:")
    for k, v in manifest["fidelity"].items():
        if not k.endswith("_note"):
            print(f"  {k} = {v}")
    if manifest["dr"].get("enabled"):
        print(f"DR seed {manifest['dr']['seed']} preset {manifest['dr']['preset']}")

    frames = manifest["frames"]
    if manifest.get("stride", 1) != 1:
        print(f"\n⚠️  stride={manifest['stride']}: this render SKIPPED frames. The output will not be "
              f"a continuous {fps} fps episode -- fine for a smoke test, wrong for training data.")

    if args.dry_run:
        print(f"\n--dry-run: would write {len(frames)} frames to repo_id {args.repo_id!r}")
        return

    import numpy as np
    from PIL import Image

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "action": {"dtype": "float32", "shape": (len(frames[0]["action"]),),
                   "names": src_features["action"]["names"]},
        "observation.state": {"dtype": "float32", "shape": (len(frames[0]["observation_state"]),),
                              "names": src_features["observation.state"]["names"]},
    }
    for key in image_keys:
        first = Image.open(render_dir / frames[0][f"image_{key.rsplit('.', 1)[-1]}"])
        features[key] = {"dtype": "video", "shape": (first.height, first.width, 3),
                         "names": ["height", "width", "channels"]}

    from lerobot.configs.video import RGBEncoderConfig

    src_info = json.loads(info_path.read_text())
    enc_kw = source_encoder_kwargs(src_features, image_keys)
    print(f"  encoder (matched to source): {enc_kw}")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(round(fps)),
        root=args.root,
        features=features,
        robot_type=src_info.get("robot_type"),
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(**enc_kw),
    )

    for rec in frames:
        frame = {
            "action": np.asarray(rec["action"], dtype=np.float32),
            "observation.state": np.asarray(rec["observation_state"], dtype=np.float32),
            # lerobot 0.6.2: the task travels INSIDE the frame dict -- add_frame(frame) takes no
            # `task=` argument (checked against the installed version's own source, 2026-09-18)
            "task": task,
        }
        for key in image_keys:
            short = key.rsplit(".", 1)[-1]
            frame[key] = np.asarray(Image.open(render_dir / rec[f"image_{short}"]).convert("RGB"))
        dataset.add_frame(frame)

    dataset.save_episode()

    # the whole point of S5 §5 item 1: a variant is worthless if you cannot trace it back
    meta_out = Path(dataset.root) / "meta" / "sim_provenance.json"
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps({
        "source_episode_id": manifest["source_episode_id"],
        "source_dataset_root": manifest["source_dataset_root"],
        "dr": manifest["dr"],
        "grasp": manifest["grasp"],
        "fidelity": manifest["fidelity"],
        "driven_by": manifest["driven_by"],
        "sign": manifest["sign"],
        "simulator": "Isaac Sim / Isaac Lab (see sim/README.md for the asset pipeline)",
    }, indent=2))

    print(f"\nwrote {len(frames)} frames to {dataset.root}")
    print(f"provenance -> {meta_out}")
    print("verify next:  uv run python scripts/verify_dataset.py <root>   (S5 §7)")


if __name__ == "__main__":
    main()

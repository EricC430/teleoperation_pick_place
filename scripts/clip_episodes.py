#!/usr/bin/env python3
"""Cut one watchable video clip per episode out of a LeRobot v3.0 dataset.

Since v3.0 a dataset no longer stores one mp4 per episode: every episode of a
camera is concatenated into `videos/<key>/chunk-000/file-000.mp4`, and the only
record of where episode 7 starts is a `from_timestamp` / `to_timestamp` pair in
`meta/episodes/*.parquet`. That makes "watch episode 7 before labelling it"
awkward exactly when it matters -- at annotation time.

This writes `outputs/episode_clips/<dataset>/ep_000.mp4`, one file per episode,
with every camera side by side and a frame counter burned in, so the clip can be
opened in a viewer (or a VS Code tab over SSH) next to the terminal running
scripts/annotate_episodes.py.

    python3 scripts/clip_episodes.py --dataset ericc430/<dataset>
    python3 scripts/clip_episodes.py --dataset ericc430/<dataset> --episodes 0-11

Then either open the folder it prints, or let the annotator point at it:

    python3 scripts/annotate_episodes.py --dataset ericc430/<dataset> --clips

Needs ffmpeg and pyarrow; no lerobot, no GPU, no container. Clips are
regenerated only when missing (--force overrides), so re-running is cheap.
"""

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate_episodes import find_info_json, parse_episodes  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_ROOT = os.path.join(REPO_ROOT, "outputs", "episode_clips")
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",          # the laptop that records is Windows
    "C:/Windows/Fonts/segoeui.ttf",
]


def find_font(explicit=None):
    """An ffmpeg-usable fontfile, or None -- in which case labels are dropped.

    No font is not an error: the clips are still watchable, they just lose the
    camera names and the frame counter. Silently producing no clip would be
    worse than producing an unlabelled one.
    """
    for candidate in ([explicit] if explicit else []) + FONT_CANDIDATES:
        if candidate and os.path.exists(candidate):
            # drawtext parses ':' as an option separator, including the one in
            # a Windows drive letter.
            return candidate.replace("\\", "/").replace(":", "\\:")
    return None


def slug(dataset):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", dataset.rstrip("/").split("/")[-1] or "dataset")


def dataset_root(dataset, root=None):
    info_path = find_info_json(dataset, root)
    if info_path is None:
        sys.exit("dataset not found locally: %s (pass --root)" % dataset)
    return os.path.dirname(os.path.dirname(info_path)), info_path


def video_keys(info):
    return [k for k, f in info.get("features", {}).items()
            if f.get("dtype") == "video"]


def read_episode_table(root, keys):
    """{episode_index: {"length": n, "tasks": [...], "videos": {key: (path, from, to)}}}"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow is required to read meta/episodes/*.parquet: pip install pyarrow")

    files = []
    for dirpath, _, filenames in os.walk(os.path.join(root, "meta", "episodes")):
        files.extend(os.path.join(dirpath, f) for f in filenames if f.endswith(".parquet"))
    if not files:
        sys.exit("no meta/episodes/*.parquet under %s -- is this a v3.0 dataset?" % root)

    episodes = {}
    for path in sorted(files):
        table = pq.read_table(path)
        wanted = [c for c in table.column_names
                  if c in ("episode_index", "length", "tasks")
                  or c.startswith("videos/")]
        cols = table.select(wanted).to_pydict()
        for i, ep in enumerate(cols["episode_index"]):
            entry = {"length": cols.get("length", [None] * (i + 1))[i],
                     "tasks": cols.get("tasks", [None] * (i + 1))[i] or [],
                     "videos": {}}
            for key in keys:
                prefix = "videos/%s/" % key
                if prefix + "from_timestamp" not in cols:
                    continue
                entry["videos"][key] = (
                    int(cols[prefix + "chunk_index"][i]),
                    int(cols[prefix + "file_index"][i]),
                    float(cols[prefix + "from_timestamp"][i]),
                    float(cols[prefix + "to_timestamp"][i]))
            episodes[int(ep)] = entry
    return episodes


def esc(text):
    """Escape a literal string for ffmpeg drawtext."""
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "")


def build_command(root, info, entry, keys, out_path, height, speed, fps, font):
    template = info["video_path"]
    inputs, filters, labels = [], [], []
    for n, key in enumerate(keys):
        chunk, file_index, start, end = entry["videos"][key]
        rel = template.format(video_key=key, chunk_index=chunk, file_index=file_index)
        src = os.path.join(root, rel)
        if not os.path.exists(src):
            sys.exit("missing video file: %s" % src)
        inputs += ["-ss", "%.6f" % start, "-to", "%.6f" % end, "-i", src]
        label = key.replace("observation.images.", "")
        step = "[%d:v]scale=-2:%d" % (n, height)
        if font:
            step += (",drawtext=fontfile=%s:text='%s':x=8:y=6:fontsize=20:"
                     "fontcolor=white:box=1:boxcolor=black@0.5" % (font, esc(label)))
        filters.append(step + "[v%d]" % n)
        labels.append("[v%d]" % n)

    chain = ";".join(filters)
    if len(labels) > 1:
        chain += ";%shstack=inputs=%d[s]" % ("".join(labels), len(labels))
    else:
        chain += ";%snull[s]" % labels[0]

    if font:
        header = "ep %03d  %s" % (entry["index"], esc("; ".join(entry["tasks"])[:60]))
        chain += (";[s]drawtext=fontfile=%s:text='%s  f\\=%%{n}':"
                  "x=8:y=h-28:fontsize=20:fontcolor=yellow:box=1:boxcolor=black@0.6[o]"
                  % (font, header))
    else:
        chain += ";[s]null[o]"
    if speed != 1.0:
        chain = chain.replace("[o]", "[p]") + ";[p]setpts=%.6f*PTS[o]" % (1.0 / speed)

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    cmd += inputs
    cmd += ["-filter_complex", chain, "-map", "[o]", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p"]
    if fps:
        cmd += ["-r", str(fps)]
    cmd += [out_path]
    return cmd


def clip_path(out_dir, ep):
    return os.path.join(out_dir, "ep_%03d.mp4" % ep)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="hf id or a local dataset directory")
    parser.add_argument("--root", help="local dataset root, if not in the HF cache")
    parser.add_argument("--episodes", help="e.g. 0-9,12 (default: all)")
    parser.add_argument("--cameras", help="comma-separated camera keys (default: all)")
    parser.add_argument("--out", help="output dir (default: outputs/episode_clips/<dataset>)")
    parser.add_argument("--height", type=int, default=480, help="per-camera height, px")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed, e.g. 2.0")
    parser.add_argument("--fps", type=int, help="output fps (default: source)")
    parser.add_argument("--font", help="TrueType file for the burnt-in labels "
                                       "(default: autodetected; none = no labels)")
    parser.add_argument("--force", action="store_true", help="re-cut clips that already exist")
    parser.add_argument("--dry-run", action="store_true", help="print the ffmpeg commands only")
    args = parser.parse_args()

    root, info_path = dataset_root(args.dataset, args.root)
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)

    keys = video_keys(info)
    if args.cameras:
        wanted = [c.strip() for c in args.cameras.split(",") if c.strip()]
        resolved = []
        for c in wanted:
            match = [k for k in keys if k == c or k.endswith("." + c)]
            if not match:
                sys.exit("no such camera %r (have: %s)" % (c, ", ".join(keys)))
            resolved.append(match[0])
        keys = resolved
    if not keys:
        sys.exit("this dataset has no video features")

    episodes = read_episode_table(root, keys)
    wanted = parse_episodes(args.episodes) if args.episodes else sorted(episodes)
    missing = [e for e in wanted if e not in episodes]
    if missing:
        sys.exit("not in this dataset (0-%d): %s" % (max(episodes), missing))

    out_dir = args.out or os.path.join(DEFAULT_OUT_ROOT, slug(args.dataset))
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)

    print("dataset : %s" % root)
    print("cameras : %s" % ", ".join(keys))
    print("clips   : %s" % out_dir)
    font = find_font(args.font)
    if font is None:
        print("note    : no TrueType font found -- clips will have no camera "
              "labels or frame counter (pass --font)")

    made = skipped = 0
    for ep in wanted:
        entry = dict(episodes[ep])
        entry["index"] = ep
        out_path = clip_path(out_dir, ep)
        if os.path.exists(out_path) and not args.force:
            skipped += 1
            continue
        cmd = build_command(root, info, entry, keys, out_path, args.height,
                            args.speed, args.fps, font)
        if args.dry_run:
            print(" ".join(cmd))
            continue
        sys.stdout.write("\r  cutting ep %03d ..." % ep)
        sys.stdout.flush()
        result = subprocess.run(cmd, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print("")
            sys.exit("ffmpeg failed on episode %d:\n%s"
                     % (ep, result.stderr.decode("utf-8", "replace")))
        made += 1
    if not args.dry_run:
        print("\r%d clips written, %d already there%s"
              % (made, skipped, " (--force re-cuts them)" if skipped else ""))


if __name__ == "__main__":
    main()

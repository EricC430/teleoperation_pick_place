#!/usr/bin/env python3
"""Fill in the per-episode metadata LeRobot does not record.

LeRobot's own schema stops at the frame level (action, observation.*, timestamp,
frame_index, episode_index, index, task_index) plus per-episode length/tasks in
meta/episodes/*.parquet. Everything else the person at the desk knows -- which
object, where it started, how the light was, whether it worked, why it didn't --
has no home in the dataset. This puts it in a CSV next to the repo, keyed by
episode_index.

The CSV columns and the questions asked come entirely from
configs/episode_meta_schema.yaml. Nothing about the field set is hardcoded here.

Typical use, right after a recording session:

    ./scripts/run_container.sh python scripts/annotate_episodes.py \
        --dataset <hf_user>/<dataset>

It finds the episodes that have no row yet and asks about each one. Fields
marked `sticky` in the schema default to the previous episode's answer, so a
session of 10 demos with the same object and lighting is mostly Enter.

Other modes:

    --set outcome=success --episodes 0-9   # batch fill, no prompting
    --check                                # validate only, non-zero exit on problems
    --redo --episodes 7                    # re-ask an episode that is already filled

Runs on the host (python3.8, no pandas needed) as long as the dataset is in the
local HF cache; falls back to meta/info.json when lerobot is not importable, in
which case episode length/task context is not shown.
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SCHEMA = os.path.join(REPO_ROOT, "configs", "episode_meta_schema.yaml")
DEFAULT_CSV_DIR = os.path.join(REPO_ROOT, "episode_meta")
KEY = "episode_index"


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------
def load_schema(path):
    with open(path) as f:
        schema = yaml.safe_load(f)
    fields = schema.get("fields") or []
    if not fields:
        sys.exit("no fields defined in %s" % path)
    names = [f["name"] for f in fields]
    if KEY in names:
        sys.exit("'%s' is the key column and is added automatically -- "
                 "remove it from %s" % (KEY, path))
    dupes = set(n for n in names if names.count(n) > 1)
    if dupes:
        sys.exit("duplicate field names in %s: %s" % (path, ", ".join(sorted(dupes))))
    return schema, fields


def field_default(field, previous_row):
    if field.get("sticky") and previous_row:
        prev = previous_row.get(field["name"], "")
        if prev != "":
            return prev
    if field.get("auto") == "now":
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return field.get("default", "")


def split_multi(value):
    return [part.strip() for part in value.split(";") if part.strip() != ""]


def normalize(field, value):
    """Canonical form of a value: multi-label fields become 'a;b', no stray spaces."""
    value = value.strip()
    if field.get("multi"):
        return ";".join(split_multi(value))
    return value


def validate(field, value):
    """Return (kind, message) where kind is "error" (rejected) or "warning" (kept)."""
    name = field["name"]
    if value == "":
        if field.get("required"):
            return "error", "%s is required" % name
        return None, None
    values = field.get("values") or []
    if not values:
        return None, None
    parts = split_multi(value) if field.get("multi") else [value]
    unknown = [p for p in parts if p not in values]
    if not unknown:
        return None, None
    kind = "error" if field.get("strict") else "warning"
    return kind, "%s: %s not in the schema's values (%s)" % (
        name, ", ".join(repr(u) for u in unknown),
        "/".join(v or "<blank>" for v in values))


# --------------------------------------------------------------------------
# dataset metadata (read-only; we never write into the dataset)
# --------------------------------------------------------------------------
def load_dataset_info(dataset, root=None):
    """Return (total_episodes, {episode_index: context_str}). Context may be empty."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    except ImportError:
        return load_dataset_info_from_json(dataset, root)

    try:
        meta = LeRobotDatasetMetadata(dataset, root=root)
    except Exception as exc:                                   # noqa: BLE001
        print("could not load dataset metadata via lerobot (%s); "
              "falling back to info.json" % exc, file=sys.stderr)
        return load_dataset_info_from_json(dataset, root)

    context = {}
    for ep in meta.episodes:
        tasks = ep.get("tasks") or []
        if isinstance(tasks, str):
            tasks = [tasks]
        context[int(ep["episode_index"])] = "%d frames, %.1fs, task=%s" % (
            ep["length"], ep["length"] / float(meta.fps or 1),
            "; ".join(tasks) if tasks else "?")
    return meta.total_episodes, context


def load_dataset_info_from_json(dataset, root=None):
    info_path = find_info_json(dataset, root)
    if info_path is None:
        return None, {}
    with open(info_path) as f:
        info = json.load(f)
    return info.get("total_episodes"), {}


def find_info_json(dataset, root=None):
    if root:
        candidate = os.path.join(root, "meta", "info.json")
        return candidate if os.path.exists(candidate) else None
    if os.path.isdir(dataset):
        candidate = os.path.join(dataset, "meta", "info.json")
        return candidate if os.path.exists(candidate) else None
    home = os.environ.get("HF_LEROBOT_HOME") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "lerobot")
    direct = os.path.join(home, dataset, "meta", "info.json")
    if os.path.exists(direct):
        return direct
    hub = os.path.join(home, "hub", "datasets--" + dataset.replace("/", "--"), "snapshots")
    if os.path.isdir(hub):
        for snap in sorted(os.listdir(hub)):
            candidate = os.path.join(hub, snap, "meta", "info.json")
            if os.path.exists(candidate):
                return candidate
    return None


# --------------------------------------------------------------------------
# csv
# --------------------------------------------------------------------------
def csv_path_for(dataset, explicit):
    if explicit:
        return explicit
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset.rstrip("/").split("/")[-1] or "dataset")
    return os.path.join(DEFAULT_CSV_DIR, slug + ".csv")


def read_csv(path):
    if not os.path.exists(path):
        return {}, []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        rows = {}
        for row in reader:
            raw = (row.get(KEY) or "").strip()
            if raw == "":
                continue
            rows[int(raw)] = {k: (v if v is not None else "") for k, v in row.items()}
    if columns and columns[0] != KEY:
        print("warning: %s does not start with a '%s' column" % (path, KEY), file=sys.stderr)
    return rows, columns


def write_csv(path, rows, fields, existing_columns):
    """Schema columns first, then any column the schema no longer knows about.

    Orphan columns are kept, never dropped -- a renamed field must not silently
    take its data with it.
    """
    schema_columns = [KEY] + [f["name"] for f in fields]
    orphans = [c for c in existing_columns if c not in schema_columns]
    columns = schema_columns + orphans

    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)

    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for idx in sorted(rows):
            row = {c: rows[idx].get(c, "") for c in columns}
            row[KEY] = idx
            writer.writerow(row)
    os.replace(tmp, path)
    return orphans


# --------------------------------------------------------------------------
# episode selection
# --------------------------------------------------------------------------
def parse_episodes(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


# --------------------------------------------------------------------------
# prompting
# --------------------------------------------------------------------------
def ask(field, default, context_line):
    name = field["name"]
    label = field.get("prompt") or name
    values = field.get("values") or []
    hint = ""
    if values:
        shown = "/".join(v if v != "" else "<blank>" for v in values)
        if len(shown) > 60:
            shown = shown[:57] + "..."
        hint = "  {%s}" % shown
    if field.get("multi"):
        hint += "  {several, ';'-separated}"
    while True:
        suffix = " [%s]" % default if default != "" else ""
        try:
            raw = input("  %s%s%s: " % (label, hint, suffix))
        except EOFError:
            raise KeyboardInterrupt
        raw = raw.strip()
        if raw == "?":
            if field.get("help"):
                print("    %s" % " ".join(field["help"].split()))
            if values:
                print("    suggested: %s" % ", ".join(v or "<blank>" for v in values))
            if context_line:
                print("    episode: %s" % context_line)
            continue
        value = default if raw == "" else raw
        if raw == "-":                      # explicit "clear this field"
            value = ""
        value = normalize(field, value)
        kind, message = validate(field, value)
        if kind == "error":
            print("    ! %s" % message)
            continue
        if kind == "warning":
            print("    ~ %s (kept)" % message)
        return value


def annotate(episodes, rows, fields, context, overrides, redo, no_prompt=False):
    """Interactively fill rows for `episodes`. Returns number of rows touched."""
    touched = 0
    previous = None
    for idx in sorted(rows):
        previous = rows[idx]
    for ep in episodes:
        existing = rows.get(ep)
        if existing and not redo:
            previous = existing
            continue
        print("")
        header = "episode %d" % ep
        context_line = context.get(ep, "")
        if context_line:
            header += "  (%s)" % context_line
        if existing:
            header += "  [re-annotating]"
        print(header)
        row = dict(existing) if existing else {}
        for field in fields:
            name = field["name"]
            if name in overrides:
                row[name] = overrides[name]
                continue
            base = existing.get(name, "") if existing else ""
            default = base if base != "" else field_default(field, previous)
            if no_prompt:
                kind, message = validate(field, default)
                if kind == "error":
                    sys.exit("episode %d: %s (pass it with --set)" % (ep, message))
                if kind == "warning":
                    print("  ~ %s" % message)
                row[name] = default
            else:
                row[name] = ask(field, default, context_line)
        row[KEY] = ep
        rows[ep] = row
        previous = row
        touched += 1
    return touched


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------
def check(rows, fields, total_episodes, existing_columns):
    problems = 0
    schema_columns = set([KEY] + [f["name"] for f in fields])
    orphans = [c for c in existing_columns if c not in schema_columns]
    if orphans:
        print("! columns not in the schema (kept, but nothing fills them): %s"
              % ", ".join(orphans))
        problems += 1

    for idx in sorted(rows):
        for field in fields:
            value = rows[idx].get(field["name"], "")
            kind, message = validate(field, value)
            if kind == "error":
                print("! episode %d: %s" % (idx, message))
                problems += 1
            elif kind == "warning":
                print("~ episode %d: %s" % (idx, message))

    if total_episodes is not None:
        missing = [e for e in range(total_episodes) if e not in rows]
        extra = [e for e in rows if e >= total_episodes]
        print("coverage: %d/%d episodes annotated" % (len(rows) - len(extra), total_episodes))
        if missing:
            print("! not annotated: %s" % compact(missing))
            problems += 1
        if extra:
            print("! rows for episodes the dataset does not have: %s" % compact(extra))
            problems += 1
    else:
        print("coverage: %d rows (dataset not found locally, cannot compare)" % len(rows))
    return problems


def compact(numbers):
    numbers = sorted(numbers)
    out, start, prev = [], None, None
    for n in numbers + [None]:
        if start is None:
            start, prev = n, n
            continue
        if n is not None and n == prev + 1:
            prev = n
            continue
        out.append(str(start) if start == prev else "%d-%d" % (start, prev))
        start, prev = n, n
    return ",".join(out)


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True,
                        help="HF repo_id (user/name) or a local dataset directory")
    parser.add_argument("--root", help="local dataset root, if not in the HF cache")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--csv", help="default: episode_meta/<dataset-name>.csv")
    parser.add_argument("--episodes", help="e.g. 0-9,12 (default: every episode with no row yet)")
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                        help="fill FIELD without asking; repeatable")
    parser.add_argument("--no-prompt", action="store_true",
                        help="never ask: take --set values and schema defaults, "
                             "fail if a required field would be left blank")
    parser.add_argument("--redo", action="store_true",
                        help="re-ask episodes that already have a row")
    parser.add_argument("--check", action="store_true",
                        help="validate and report coverage, write nothing")
    parser.add_argument("--dry-run", action="store_true", help="do not write the CSV")
    args = parser.parse_args()

    schema, fields = load_schema(args.schema)
    field_names = [f["name"] for f in fields]

    overrides = {}
    for item in args.set:
        if "=" not in item:
            sys.exit("--set expects FIELD=VALUE, got %r" % item)
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in field_names:
            sys.exit("--set %s: not a field in %s (have: %s)"
                     % (name, args.schema, ", ".join(field_names)))
        field = fields[field_names.index(name)]
        value = normalize(field, value)
        kind, message = validate(field, value)
        if kind == "error":
            sys.exit("--set %s: %s" % (name, message))
        if kind == "warning":
            print("~ %s" % message)
        overrides[name] = value

    csv_file = csv_path_for(args.dataset, args.csv)
    rows, existing_columns = read_csv(csv_file)
    total_episodes, context = load_dataset_info(args.dataset, args.root)

    print("dataset : %s" % args.dataset)
    print("schema  : %s (version %s, %d fields)"
          % (args.schema, schema.get("version", "?"), len(fields)))
    print("csv     : %s (%d existing rows)" % (csv_file, len(rows)))
    if total_episodes is None:
        print("note    : dataset not found locally -- episode count unknown, "
              "--episodes is required")

    if args.check:
        sys.exit(1 if check(rows, fields, total_episodes, existing_columns) else 0)

    if args.episodes:
        episodes = parse_episodes(args.episodes)
    elif total_episodes is not None:
        episodes = [e for e in range(total_episodes) if e not in rows]
        if not episodes:
            print("nothing to do: all %d episodes already have a row "
                  "(use --redo --episodes N to revisit one)" % total_episodes)
            return
    else:
        sys.exit("--episodes is required when the dataset is not available locally")

    if total_episodes is not None:
        out_of_range = [e for e in episodes if e >= total_episodes or e < 0]
        if out_of_range:
            sys.exit("episodes not in this dataset (0-%d): %s"
                     % (total_episodes - 1, compact(out_of_range)))

    print("to fill : %s" % compact(episodes))
    if not args.no_prompt:
        print("(Enter accepts the default, '?' explains the field, '-' clears it, "
              "Ctrl-C saves and exits)")

    interrupted = False
    try:
        touched = annotate(episodes, rows, fields, context, overrides, args.redo,
                           args.no_prompt)
    except KeyboardInterrupt:
        interrupted = True
        touched = sum(1 for e in episodes if e in rows)
        print("\ninterrupted -- saving what was answered so far")

    if args.dry_run:
        print("\ndry run: %d rows would be written to %s" % (len(rows), csv_file))
    elif touched or interrupted:
        orphans = write_csv(csv_file, rows, fields, existing_columns)
        print("\nwrote %s (%d rows)" % (csv_file, len(rows)))
        if orphans:
            print("kept columns no longer in the schema: %s" % ", ".join(orphans))
        if touched:
            print("commit it -- these annotations are not recoverable from the dataset.")
    else:
        print("\nnothing changed")

    if interrupted:
        sys.exit(130)


if __name__ == "__main__":
    main()

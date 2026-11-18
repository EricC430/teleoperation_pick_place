"""One-off: rename the local "aimfix26" names to "aimfix28" after the 300k run has finished (2026-11-18).

aimfix_A1_cut has 28 episodes since 11/16 (26 = recorded count before ep4 / ep16 were split). The Hub repo was renamed
on 11/17 (af9f80c); the local run dir, merged dataset dir, training config and doc references still say 26.

Not touched on purpose: config_records/ (records of what was run), train_config.json inside the checkpoints (written by
the run), narrative text other than names.

    python3 scripts/rename_aimfix26_to_28.py           # dry run: prints every move and changed line
    python3 scripts/rename_aimfix26_to_28.py --apply   # refuses while the training container / process is alive
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OLD, NEW = "aimfix26", "aimfix28"
RUN = "cup129_bottle135_aimfix26_twoobj10_augfl_fldrop_p30_w30"
DS = "data/huggingface/lerobot/ericc430/omx_pick_place_pilot_cup129_bottle135_aimfix26_twoobj10"
CFG = "configs/train_omx_cup129_bottle135_aimfix26_twoobj10_augfl_fldrop.yaml"
MOVES = [  # plain mv (gitignored data)
    (f"data/train/{RUN}", f"data/train/{RUN.replace(OLD, NEW)}"),
    (f"data/train/{RUN}.train.log", f"data/train/{RUN.replace(OLD, NEW)}.train.log"),
    (DS, DS.replace(OLD, NEW)),
]
EDIT_FILES = [  # every aimfix26 in these is a name (checked 11/18 with git grep)
    "configs/rollout_omx_aimfix_200k_te-001_plastic_bottle_eval.yaml",
    "docs/environment.md",
    "docs/models.md",
    "docs/weekly/2026-11-09.md",
    "docs/worklog/2026-11-15_aimfix_twoobj_recording.md",
    "docs/worklog/2026-11-16_aimfix_split.md",
    "docs/worklog/2026-11-16_isaaclab3_migration.md",
    "scripts/input_reliance/h19_c32_stall_close.py",
]
CFG_COMMENT = ('# (docs/worklog/2026-11-16_aimfix_split.md). The file name keeps "aimfix26" = recorded episodes. Counts below updated.',
               '# (docs/worklog/2026-11-16_aimfix_split.md). Counts below updated. Renamed aimfix26 -> aimfix28 on 2026-11-18 (28 = episodes\n'
               '# after the split; run dir, dataset dir and this file; the Hub repo was renamed 11/17).')


def training_alive():
    ps = subprocess.run(["docker", "ps", "-q", "--filter", "name=^lerobot-dev$"], capture_output=True, text=True).stdout.strip()
    host = subprocess.run(["pgrep", "-f", "train_fl_dropout"], capture_output=True, text=True).stdout.strip()
    return bool(ps or host)


def main():
    apply = "--apply" in sys.argv
    ck = REPO / "data/train" / RUN / "checkpoints/300000/pretrained_model/model.safetensors"
    print(f"300k checkpoint: {'present' if ck.exists() else 'MISSING'}")
    if apply and (training_alive() or not ck.exists()):
        sys.exit("refusing: training still alive or 300k checkpoint missing")
    for a, b in MOVES:
        pa, pb = REPO / a, REPO / b
        assert pa.exists() and not pb.exists(), (a, b)
        print(f"mv {a} -> {b}")
        if apply:
            pa.rename(pb)
    # config: content first, then git mv
    p = REPO / CFG
    t = p.read_text(encoding="utf-8")
    assert CFG_COMMENT[0] in t
    t2 = t.replace(CFG_COMMENT[0], "\0").replace(OLD, NEW).replace("\0", CFG_COMMENT[1])
    show(CFG, t, t2)
    if apply:
        p.write_text(t2, encoding="utf-8")
        subprocess.run(["git", "mv", CFG, CFG.replace(OLD, NEW)], cwd=REPO, check=True)
    for f in EDIT_FILES:
        p = REPO / f
        t = p.read_text(encoding="utf-8")
        t2 = t.replace(OLD, NEW)
        show(f, t, t2)
        if apply and t2 != t:
            p.write_text(t2, encoding="utf-8")
    print("applied" if apply else "dry run only (--apply to do it)")


def show(f, a, b):
    import difflib
    for ln in difflib.unified_diff(a.splitlines(), b.splitlines(), f, f, n=0, lineterm=""):
        if not ln.startswith(("---", "+++")):
            print("    " + ln[:200])


if __name__ == "__main__":
    main()

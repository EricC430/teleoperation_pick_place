"""Sim replay -> domain-randomized data collection, end to end (docs/specs/S9_sim_dr_collect.md).

    python3 scripts/dr_collect.py --work ~/isaaclab_volume/dr/<run> --source ~/isaaclab_volume/dr/src/<prep dir> [steps]

Steps (default all, in order; each skips work whose output already exists, so a run can be resumed):
  objects   sim/dr_objects.py in the container: object geometry variants (USD with UVs + physics) from sim/dr_config.json
  check     sim/physics_replay.py config_report=sim/dr_expected_config.json config_only=1 on the first episode:
            the physics configuration must equal the one validated against the real closed loop; stops on FAIL
  validate  sim/physics_replay.py render=0 pose_log=1 for every (real-success episode x object variant); scored with
            sim/replay_score.py; sim/penetration.py measures the gripper links against the object wall in every frame.
            Accepted (dr_collect.accepted): real success, lifted, in the bin, no link through the object.
  render    sim/dr_render.py: each validated replay is re-rendered kinematically (the logged link and object poses)
            `variants_per_replay` times, each with its own randomized object appearance / environment / lights / arm skin
  export    scripts/sim_replay_augment.py in the huggingface/lerobot-gpu image: all rendered variants -> one LeRobot
            dataset `sim_<name>` (the sim_ prefix is enforced there), provenance per episode
  sheet     a contact sheet of a few frames per variant for a look before training

Paths: the work dir and the source prep dir must be under ~/isaaclab_volume: it is the only host directory the isaac-lab
container mounts (as /workspace/test_isaaclab; the container also runs other people's jobs, so a new mount would mean
recreating it). The finished LeRobot dataset is written into the repo's data/huggingface/lerobot (--export-root), by the
lerobot container, which mounts the repo. The prep dir comes from scripts/prep_closed_loop_replay.py. Only data recorded from
2026-10-14 on (EricC, 2026-10-31: this stage does not use the older data; the arm model is the 10/14 one).
"""
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim"))
import replay_score  # noqa: E402

HOST_VOL = Path(os.environ.get("HOST_VOLUME", Path.home() / "isaaclab_volume")).resolve()
GUEST_VOL = "/workspace/test_isaaclab"
CONTAINER = os.environ.get("CONTAINER", "isaac-lab")
STEPS = ["objects", "check", "validate", "render", "export", "sheet"]


def guest(p):
    p = Path(p).resolve()
    return f"{GUEST_VOL}/{p.relative_to(HOST_VOL)}"


def sync():
    """The light repo mirror the container scripts read (same as sim/run_in_container.sh)."""
    m = HOST_VOL / "omx_repo"; (m / "sim").mkdir(parents=True, exist_ok=True); (m / "calibration").mkdir(exist_ok=True)
    (m / "assets").mkdir(exist_ok=True); (HOST_VOL / "omx_sim").mkdir(exist_ok=True)
    for f in (REPO / "sim").glob("*.py"):
        shutil.copy2(f, m / "sim"); shutil.copy2(f, HOST_VOL / "omx_sim")
    for f in (REPO / "sim").glob("dr_*.json"):
        shutil.copy2(f, m / "sim")
    shutil.copytree(REPO / "reach_logger", m / "reach_logger", dirs_exist_ok=True)
    for f in (REPO / "calibration").glob("*.json"):
        shutil.copy2(f, m / "calibration")
    for a in ("paper_cup", "omx_f"):
        shutil.copytree(REPO / "assets" / a, m / "assets" / a, dirs_exist_ok=True)
    gen = m / "assets" / "omx_f_generated"
    if not gen.exists():
        gen.symlink_to("../../assets/omx_f_generated")


def run_sim(script, args, log, timeout_s=900, retries=1):
    """One Isaac process in the container. `timeout` runs INSIDE the container (killing the docker exec client would
    leave the process running). 2026-10-31: one of three replays started together hung in extension startup forever;
    a timed-out run is retried once."""
    cmd = ["docker", "exec", "-e", "PYTHONUNBUFFERED=1", "-e", f"OMX_REPO={GUEST_VOL}/omx_repo", CONTAINER,
           "timeout", "-k", "20", str(int(timeout_s)), "/isaac-sim/python.sh", f"{GUEST_VOL}/omx_sim/{script}"] + args
    t = time.time()
    for attempt in range(retries + 1):
        with open(log, "w" if attempt == 0 else "a") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode not in (124, 137):
            break
        with open(log, "a") as f:
            f.write(f"\nDR_COLLECT: timed out after {timeout_s} s (attempt {attempt + 1})\n")
    return r.returncode, time.time() - t


def physx_failed(log):
    """2026-10-31: under GPU contention PhysX printed 'GPU ... solver ... fail to launch kernel' and the replay ran on
    to 'DONE' with the object never simulated and the arm flung (9 of 290 runs) -- a silent bad result, so a run whose
    log has a PhysX error does not count."""
    try:
        return "PhysX error" in Path(log).read_text(errors="ignore")
    except OSError:
        return True


def accepted(s):
    """Real success, and in sim: lifted, dropped in the bin (when the scene has it, since 2026-11-01), and no gripper link
    through the object in any frame (when measured, since 2026-11-01)."""
    ok = s["real_success"] and (s["task_success"] if s.get("task_success") is not None else s["sim_success"])
    return bool(ok and s.get("penetration", {}).get("clean", True))


def physics_args(cfg, src):
    c = cfg["physics_replay"]
    return [f"cand=fit", f"cands={guest(src / 'fit_candidate.json')}", f"intr={c['intr']}", f"ext={c['ext']}"] + \
           [f"{k}={v}" for k, v in c.get("extra", {}).items()]


def episodes(src, only_success=True):
    out = []
    for d in sorted(src.glob("cl_ep*")):
        E = json.load(open(d / "episode.json"))
        if not only_success or E.get("outcome") == "success":
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True); ap.add_argument("--source", required=True)
    ap.add_argument("--config", default=str(REPO / "sim/dr_config.json"))
    ap.add_argument("--objects", default=None,
                    help="JSON with a 'geometry' list (e.g. sim/dr_objects_bottle.json) that replaces the config's objects.geometry, "
                         "and an optional 'expected_config' for physics_replay; the merged config is what the work dir keeps")
    ap.add_argument("--steps", default=",".join(STEPS)); ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--pen-jobs", type=int, default=8, help="parallel penetration measurements (CPU only)")
    ap.add_argument("--episodes", default="", help="comma list of episode indices (default: every real success)")
    ap.add_argument("--all-outcomes", action="store_true", help="validate real failures too (to measure sim/real agreement)")
    ap.add_argument("--name", default=None, help="dataset name (default: dr_<work dir name>)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan (episodes, objects, counts) and stop")
    ap.add_argument("--export-root", default=str(REPO / "data/huggingface/lerobot"),
                    help="where the LeRobot dataset goes (default: the repo's gitignored data/huggingface/lerobot, next to the "
                         "real datasets; the heavy per-frame renders stay in the work dir, which the container can write)")
    a = ap.parse_args()
    work, src = Path(a.work).expanduser().resolve(), Path(a.source).expanduser().resolve()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    if a.objects:   # 2026-11-11: bottles live outside dr_config.json so a cup run never gets them
        ov = json.loads(Path(a.objects).read_text(encoding="utf-8"))
        cfg["objects"]["geometry"] = ov["geometry"]; cfg["objects"]["_geometry_from"] = str(a.objects)
        if "expected_config" in ov:
            cfg["physics_replay"]["expected_config"] = ov["expected_config"]
    for p in (work, src):
        p.relative_to(HOST_VOL)   # must be inside the mounted volume
    steps = a.steps.split(",")
    eps = episodes(src, only_success=not a.all_outcomes)
    if a.episodes:
        keep = {int(x) for x in a.episodes.split(",")}
        eps = [d for d in eps if int(d.name[5:]) in keep]
    print(f"{len(eps)} source episodes from {src}", flush=True)
    if a.dry_run:
        geo = [g["id"] for g in cfg["objects"]["geometry"]]
        print(f"dry run: steps {steps}; objects {geo}; up to {len(eps) * len(geo)} physics replays, "
              f"then {cfg['variants_per_replay']} rendered variants per replay that lifts the object; work dir {work}")
        return
    (work / "logs").mkdir(parents=True, exist_ok=True)
    if a.objects:
        (work / "dr_config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
    else:
        shutil.copy2(a.config, work / "dr_config.json")
    sync()

    objs_dir = work / "objects"
    if "objects" in steps and not (objs_dir / "objects.json").exists():
        rc, dt = run_sim("dr_objects.py", [f"config={guest(work / 'dr_config.json')}", f"out={guest(objs_dir)}"], work / "logs/objects.log")
        print(f"objects: rc {rc} {dt:.0f}s", flush=True)
        if rc:
            sys.exit(f"objects failed, see {work / 'logs/objects.log'}")
    objects = json.load(open(objs_dir / "objects.json"))["objects"]

    if "check" in steps:
        exp = cfg["physics_replay"].get("expected_config", "sim/dr_expected_config.json")
        rc, dt = run_sim("physics_replay.py", [f"ep={guest(eps[0])}"] + physics_args(cfg, src) +
                         [f"out={guest(work / 'check')}", f"config_report={exp}", "config_only=1", "render=0"], work / "logs/check.log")
        # the verdict is read from the log: Isaac's app.close() ends the process with 0 before physics_replay's
        # sys.exit(3) on a FAIL (2026-11-01: a FAIL was reported as PASS)
        passed = rc == 0 and "CONFIG REPORT PASS" in (work / "logs/check.log").read_text(errors="ignore")
        print(f"check: {'PASS' if passed else 'FAIL'} ({dt:.0f}s) -> {work / 'check/config_report.json'}", flush=True)
        if not passed:
            sys.exit("configuration check failed; see the CONFIG MISMATCH lines in " + str(work / "logs/check.log"))

    val = work / "validate"
    if "validate" in steps:
        jobs = []
        for o in objects:
            for d in eps:
                out = val / o["id"] / d.name
                if not (out / "physics_log.json").exists() or physx_failed(work / "logs" / o["id"] / f"{d.name}.log"):
                    jobs.append((o, d, out))
        print(f"validate: {len(jobs)} replays to run ({len(objects)} objects x {len(eps)} episodes, {a.jobs} at a time)", flush=True)

        def one(j):
            o, d, out = j
            (work / "logs" / o["id"]).mkdir(parents=True, exist_ok=True)
            lg = work / "logs" / o["id"] / f"{d.name}.log"
            for _ in range(3):   # a PhysX GPU failure is retried (up to 2 more times)
                r = run_sim("physics_replay.py", [f"ep={guest(d)}"] + physics_args(cfg, src) +
                            [f"cup_asset={o['usd_guest']}", f"out={guest(out)}", "render=0", "pose_log=1"], lg)
                if not physx_failed(lg):
                    break
            return j, r
        with cf.ThreadPoolExecutor(a.jobs) as ex:
            for (o, d, out), (rc, dt) in ex.map(one, jobs):
                print(f"  {o['id']} {d.name}: rc {rc} {dt:.0f}s", flush=True)
        # Penetration (2026-11-01, EricC: "務必解決穿模"): every replay's gripper links vs the cup wall, sim/penetration.py.
        # A replay where a link goes through the wall in any frame is not accepted (render step).
        links = work / "links.npz"
        if not links.exists():
            asset = cfg["physics_replay"].get("extra", {}).get("asset", "assets/omx_f_generated/omx_f_real.usda")
            rc, dt = run_sim("extract_link_meshes.py", [f"{GUEST_VOL}/omx_repo/{asset}", guest(links)], work / "logs/extract_link_meshes.log")
            print(f"link meshes: rc {rc} {dt:.0f}s", flush=True)
        pjobs = [(o, d) for o in objects for d in eps if (val / o["id"] / d.name / "physics_log.json").exists()
                 and not (val / o["id"] / d.name / "penetration.json").exists()]

        def pen(j):
            o, d = j; dd = val / o["id"] / d.name
            return j, run_sim("penetration.py", [f"links={guest(links)}", f"cup={guest(objs_dir / 'objects.json')}:{o['id']}",
                                                 f"logs={guest(dd / 'physics_log.json')}", f"out={guest(dd / 'penetration.json')}"],
                              work / "logs" / o["id"] / f"{d.name}_penetration.log")
        if pjobs:
            print(f"penetration: {len(pjobs)} replays ({a.pen_jobs} at a time)", flush=True)
            pen(pjobs[0])   # the first one alone: it voxelizes the links and caches the samples next to links.npz
            with cf.ThreadPoolExecutor(a.pen_jobs) as ex:
                list(ex.map(pen, pjobs[1:]))
        summary = []
        for o in objects:
            for d in eps:
                lg = val / o["id"] / d.name / "physics_log.json"
                if lg.exists() and physx_failed(work / "logs" / o["id"] / f"{d.name}.log"):
                    print(f"  {o['id']} {d.name}: PhysX error in the log, not counted", flush=True); continue
                if lg.exists():
                    s = replay_score.score(d, lg); s.update(object=o["id"], ep_dir=str(d), log=str(lg))
                    pj = lg.parent / "penetration.json"
                    if pj.exists():
                        r = next(iter(json.load(open(pj)).values()))
                        s["penetration"] = {"links": r["links"], "max_vol_mm3": r["max_vol_mm3"], "frames_through": r["frames_through"],
                                            "frames_vol_gt_1mm3": r["frames_vol_gt_1mm3"],
                                            "clean": sum(r["frames_through"]) == 0 and r["frames_vol_gt_1mm3"] == 0}
                    summary.append(s)
        json.dump(summary, open(work / "validate_summary.json", "w"), indent=1)
        for o in objects:
            ss = [s for s in summary if s["object"] == o["id"]]
            real = [s for s in ss if s["real_success"]]
            agree = sum(s["first_close_agree"] for s in ss if s["first_close_agree"] is not None)
            print(f"  {o['id']}: sim lifted {sum(s['sim_success'] for s in ss)}/{len(ss)}; of real successes {sum(s['sim_success'] for s in real)}/{len(real)}; "
                  f"first-close agreement {agree}/{len(ss)}"
                  + (f"; lifted AND in the bin {sum(bool(s.get('task_success')) for s in real)}/{len(real)}" if any(s.get("in_bin_final") is not None for s in ss) else "")
                  + (f"; no penetration {sum(s['penetration']['clean'] for s in ss if 'penetration' in s)}/{sum('penetration' in s for s in ss)}"
                     f" (accepted: {sum(accepted(s) for s in real)}/{len(real)})" if any("penetration" in s for s in ss) else ""), flush=True)

    rend = work / "render"
    if "render" in steps:
        summary = json.load(open(work / "validate_summary.json"))
        # accepted = the real task in sim: lifted, and (when the scene has the bin, since 2026-11-01) dropped in it
        ok = [s for s in summary if accepted(s)]
        n = int(cfg["variants_per_replay"])
        todo = [(s, v) for s in ok for v in range(n)
                if not (rend / f"{s['object']}_{Path(s['ep_dir']).name}_v{v}" / "manifest.json").exists()]
        print(f"render: {len(todo)} variants ({len(ok)} validated replays x {n})", flush=True)
        # one container call per replay renders all its variants (startup is ~20 s)
        by = {}
        for s, v in todo:
            by.setdefault((s["object"], s["ep_dir"], s["log"]), []).append(v)
        obj_by_id = {o["id"]: o for o in objects}

        def src_root_host(ep_dir):   # the source LeRobot dataset on this machine (the export step reads its info.json)
            name = json.load(open(Path(ep_dir) / "episode.json"))["dataset"]
            for r in (REPO / "data/huggingface/lerobot", Path.home() / ".cache/huggingface/lerobot"):
                for d in (r / "ericc430" / name, r / name):
                    if (d / "meta/info.json").exists():
                        return d
            return Path(name)

        def onerender(item):
            (oid, ep_dir, log), vs = item
            name = f"{oid}_{Path(ep_dir).name}"
            return name, run_sim("dr_render.py", [f"config={guest(work / 'dr_config.json')}", f"ep={guest(Path(ep_dir))}", f"log={guest(Path(log))}",
                                                  f"object={obj_by_id[oid]['usd_guest']}", f"object_id={oid}", f"variants={','.join(map(str, vs))}",
                                                  f"out={guest(rend)}", f"src_root_host={src_root_host(ep_dir)}"], work / "logs" / f"render_{name}.log",
                                 timeout_s=1200 + 900 * len(vs))
        with cf.ThreadPoolExecutor(max(1, a.jobs - 1)) as ex:
            for name, (rc, dt) in ex.map(onerender, by.items()):
                print(f"  {name}: rc {rc} {dt:.0f}s", flush=True)

    if "export" in steps:
        dirs = sorted(p.parent for p in rend.glob("*/manifest.json") if "rejected" not in json.load(open(p)))
        name = a.name or f"dr_{work.name}"
        src_info = Path(json.load(open(dirs[0] / "manifest.json"))["source_dataset_root_host"]) / "meta/info.json"
        out_root = Path(a.export_root) / f"sim_{name}"
        if out_root.exists():
            print(f"export: {out_root} exists, skipped", flush=True)
        else:
            cmd = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "-e", f"USER={os.environ.get('USER', 'user')}",
                   "-v", f"{REPO}:{REPO}", "-v", f"{HOST_VOL}:{HOST_VOL}", "-w", str(REPO), "huggingface/lerobot-gpu:latest",   # same paths as on the host
                   "python", "scripts/sim_replay_augment.py", "--repo-id", f"sim_{name}", "--root", str(out_root),
                   "--source-info", str(src_info)] + sum([["--render-dir", str(d)] for d in dirs], [])
            with open(work / "logs/export.log", "w") as f:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
            print(f"export: rc {rc}, {len(dirs)} episodes -> {out_root}", flush=True)

    if "sheet" in steps:
        import cv2
        import numpy as np
        rows = []
        for d in sorted(p.parent for p in rend.glob("*/manifest.json") if "rejected" not in json.load(open(p))):
            M = json.load(open(d / "manifest.json")); fr = M["frames"]
            pick = [fr[int(i)] for i in np.linspace(0, len(fr) - 1, 4)]
            tiles = []
            for r in pick:
                a_ = cv2.resize(cv2.imread(str(d / r["image_front-left"])), (320, 181)); b_ = cv2.resize(cv2.imread(str(d / r["image_wrist"])), (240, 181))
                tiles.append(np.hstack([a_, b_]))
            row = np.hstack(tiles)
            cv2.putText(row, d.name, (6, 18), 0, 0.55, (0, 255, 255), 2)
            rows.append(row)
        if rows:
            out = work / "contact_sheet.jpg"
            cv2.imwrite(str(out), np.vstack(rows[:40]), [cv2.IMWRITE_JPEG_QUALITY, 80])
            print(f"sheet: {out} ({len(rows)} variants, first 40 shown)", flush=True)


if __name__ == "__main__":
    main()

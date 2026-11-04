"""One front-left frame for every DR ground (every Base MDL material in sim/dr_config.json plus the procedural patterns of
sim/dr_ground.py), all under the same light, no props, fixed exposure target: to see what each ground really looks like
after the auto exposure and to catch the ones that come out wrong (2026-11-05, EricC: dark asphalt had come out light blue).

    python3 scripts/dr_ground_catalog.py --work ~/isaaclab_volume/dr/ground_catalog [--jobs 3] [--seeds 2]

Needs the isaac-lab container (like dr_collect.py) and a validated replay to borrow the arm and cup poses from
(--run: a dr_collect work dir, default run_20261101; --source: its prep dir).
"""
import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim")); sys.path.insert(0, str(REPO / "scripts"))
import dr_collect as DC  # noqa: E402
import dr_ground  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--run", default=str(DC.HOST_VOL / "dr/run_20261101")); ap.add_argument("--source", default=str(DC.HOST_VOL / "dr/src/cl_20261014_151208"))
    ap.add_argument("--ep", default="cl_ep00"); ap.add_argument("--jobs", type=int, default=3); ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--hdri", default="Cloudy/kloofendal_48d_partly_cloudy_4k.hdr")
    ap.add_argument("--config", default=str(REPO / "sim/dr_config.json"))
    a = ap.parse_args()
    work, run, src = Path(a.work).expanduser().resolve(), Path(a.run), Path(a.source)
    for p in (work, run, src):
        p.relative_to(DC.HOST_VOL)
    (work / "logs").mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text()); env = cfg["environment"]
    cfg["render"]["brightness_target"] = [120, 120]   # the same target for every ground
    (work / "dr_config.json").write_text(json.dumps(cfg))
    DC.sync()
    mdls = []
    for mode in ("outdoor", "studio"):
        for m in env["ground_materials"][mode]:
            if m not in [x[1] for x in mdls]:
                mdls.append((mode, m))
    items = [(mode, {"mdl": env["materials"] + m}, m.replace(".mdl", "").replace("/", ":")) for mode, m in mdls]
    for kind in dr_ground.KINDS:
        items += [("outdoor", {"pattern": kind}, f"pattern:{kind}")] * a.seeds
    base = {"props": [], "lights": [], "sun": None, "hdri": env["skies"] + a.hdri,
            "dome": {"intensity": 1000.0, "exposure": 0.0, "temperature": 6500.0, "rotation_deg": 0.0}}
    ovs = [{**base, "mode": mode, "ground": g} for mode, g, _ in items]
    (work / "overrides.json").write_text(json.dumps(ovs))
    ep = src / a.ep; log = run / "validate/cup_ref" / a.ep / "physics_log.json"; obj = run / "objects/cup_ref.usd"
    out = work / "render"
    chunks = [list(range(i, len(items), a.jobs)) for i in range(a.jobs)]

    def go(i):
        vs = [v for v in chunks[i] if not (out / f"cup_ref_{a.ep}_v{v}" / "manifest.json").exists()]
        if not vs:
            return 0, 0
        return DC.run_sim("dr_render.py", [f"config={DC.guest(work / 'dr_config.json')}", f"ep={DC.guest(ep)}", f"log={DC.guest(log)}", f"object={DC.guest(obj)}",
                                           "object_id=cup_ref", f"variants={','.join(map(str, vs))}", f"out={DC.guest(out)}", f"overrides={DC.guest(work / 'overrides.json')}",
                                           "max_frames=1"], work / "logs" / f"catalog_{i}.log", timeout_s=3600)
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for i, (rc, dt) in enumerate(ex.map(go, range(a.jobs))):
            print(f"job {i}: rc {rc} {dt:.0f}s", flush=True)
    import cv2
    import numpy as np
    tiles = []
    for v, (mode, g, name) in enumerate(items):
        d = out / f"cup_ref_{a.ep}_v{v}"
        if not (d / "manifest.json").exists():
            print("missing", v, name); continue
        M = json.load(open(d / "manifest.json")); im = cv2.resize(cv2.imread(str(d / M["frames"][0]["image_front-left"])), (400, 226))
        cv2.putText(im, f"{v} {name}"[:44], (6, 18), 0, 0.5, (0, 255, 255), 2); tiles.append(im)
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(work / "catalog.jpg"), sheet); print("sheet", work / "catalog.jpg", len(items), "grounds")


if __name__ == "__main__":
    main()

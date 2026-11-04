# Summarise the 2026-10-29 field fits: every run's camera vs the all-stills fit (g30), overlay, last-round step.
#   python3 calibration/2026-10-29_fl_cad_field/summarize.py <dir with g30.json and the other runs' .json/.txt>
import glob, json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from fit_camera_from_cad import pose_diff  # noqa: E402

d = Path(sys.argv[1])
ref = np.array(json.loads((d / "g30.json").read_text())["world_from_camera"])
print(f"{'run':8s} stills  overlay med/p90 px  matches  last step deg/mm   vs g30 mm / deg")
for f in sorted(glob.glob(str(d / "*.json"))):
    j = json.loads(Path(f).read_text()); txt = Path(f[:-5] + ".txt").read_text()
    n = re.search(r"^(\d+) stills", txt, re.M).group(1)
    st = re.findall(r"step ([\d.]+) deg ([\d.]+) mm", txt)[-1]
    o = j["report"]["overlay_px"]; dm, dd = pose_diff(ref, np.array(j["world_from_camera"]))
    print(f"{Path(f).stem:8s} {n:>5s}   {o['median']:5.2f} / {o['p90']:5.2f}     {o['n']:6d}   {float(st[0]):5.3f} / {float(st[1]):5.2f}     {dm:5.1f} / {dd:4.2f}")
    if "halves" in j["report"]:
        r = j["report"]
        for k in (0, 1):
            h = r[f"half{k}"]
            print(f"   half {k} {h['episodes']}: held-out overlay {h['held_out_overlay_median']:.2f} / {h['held_out_overlay_p90']:.2f} px")
        print(f"   half 0 vs half 1: {r['halves']['mm']:.1f} mm, {r['halves']['deg']:.2f} deg")

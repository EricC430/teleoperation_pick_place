# Start camera for the 2026-10-29 field fit: the 10-18 ArUco v2 camera (paper frame of 10-18), moved into the arm-fit
# world (10-14 paper frame) through the arm base: world14_from_link0 @ inv(paper18_from_base) @ paper18_from_camera.
# paper18_from_base = the base-edge fit, middle hand-drawn line (docs/worklog/2026-10-29_fl_markerless.md §2):
# dx -3.7, dy +9.1, z 147.9 mm, yaw +0.15 deg, tilt 3.35 deg about y. Only a start: its error is that of §2 plus however
# far the camera moved 10-18 -> 10-29.
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
REPO = Path(__file__).resolve().parents[2]
arm = json.loads((REPO / "calibration/2026-10-29_fl_markerless/arm_fit_H2.json").read_text())
v2 = json.loads((REPO / "calibration/2026-10-18_camera_extrinsics_front-left_v2.json").read_text())
w, x, y, z = v2["quat_wxyz"]
cam18 = np.eye(4); cam18[:3, :3] = R.from_quat([x, y, z, w]).as_matrix(); cam18[:3, 3] = v2["pos_m"]
base18 = np.eye(4); base18[:3, :3] = R.from_euler("zy", [0.15, 3.35], degrees=True).as_matrix(); base18[:3, 3] = [-0.0037, 0.0091, 0.1479]
wfc = np.array(arm["world_from_link0"]) @ np.linalg.inv(base18) @ cam18
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("start_v2.json")
out.write_text(json.dumps({"what": "start_from_v2.py", "world_from_camera": wfc.tolist()}, indent=1))
print(f"wrote {out}")

"""S5 gap 4 / S4 §5-5 T1+T2 — measure camera intrinsics and extrinsics for the sim scene.

Lab-day tool. Everything it produces replaces a PLACEHOLDER in `sim/scene_constants.py`; it does not
edit that file.

Subcommands
-----------
  markers      print sheet: ArUco markers at a known physical size (print at 100 %, measure one!)
  board        print sheet: checkerboard for `checkerboard` (the UVC wrist cam's T1)
  rs-intrinsics  T1 for the RealSense (front-left D455): read fx/fy/cx/cy off the device
  capture      save frames from an OpenCV camera (the UVC wrist cam) — for checkerboard shots
  checkerboard T1 for the UVC wrist cam: it exposes no factory intrinsics, so calibrate from images
  extrinsics   T2: ArUco markers lying on the placement mat at known pan-frame points -> camera pose
  selftest     render a synthetic view of known pose, run `extrinsics` on it, check the pose comes back

Frame (same as sim/scene_constants.py and the placement CSVs): origin = pan axis on the table top,
+X ahead (away from the operator), +Y operator-left, +Z up. Metres in the JSON outputs.

Marker placement rule for `extrinsics`: lay each marker flat with its centre on the listed point and
its printed TOP edge facing +X (away from the operator). A marker rotated 90 deg gives a pose that is
wrong by exactly that rotation and still has a small reprojection error — the per-marker table is how
you catch it (one marker disagreeing with the rest).

    uv run python scripts/measure_camera_geometry.py selftest
    uv run python scripts/measure_camera_geometry.py markers --ids 0-5 --size-mm 60 --out outputs/cam_geom/markers.png
    uv run python scripts/measure_camera_geometry.py rs-intrinsics --serial 262822305610 --width 848 --height 480 \
        --save-image outputs/cam_geom/front-left_shot.png --out outputs/cam_geom/front-left_intrinsics.json
    uv run python scripts/measure_camera_geometry.py extrinsics --image outputs/cam_geom/front-left_shot.png \
        --intrinsics outputs/cam_geom/front-left_intrinsics.json \
        --markers configs/camera_markers_campA_136sym_20260918.csv --size-mm 60 \
        --out outputs/cam_geom/front-left_extrinsics.json

Wrist camera: `capture` + `checkerboard` give its intrinsics. Its extrinsics are NOT covered here — it rides
on link5, so its pose in the pan frame depends on the arm pose; the mounting offset needs FK (S1/D026),
which this script does not do.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ISAAC_APERTURE_MM = 20.955  # sim/scene_constants.SENSOR_APERTURE_MM
DICT = "DICT_4X4_50"
# T3 thresholds proposed in S4 §5-5 (for sim-vs-real reprojection); applied here to the PnP fit as a
# necessary condition — a pose that cannot even reproject its own markers will not pass T3.
T3_MEDIAN_PX, T3_MAX_PX = 10.0, 25.0


def _cv2():
    import cv2

    return cv2


def _dictionary():
    cv2 = _cv2()
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT))


def _write_json(path: str | None, obj: dict):
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    print(text)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
        print(f"wrote {path}")


def isaac_camera_fields(fx: float, fy: float, cx: float, cy: float, width: int, height: int) -> dict:
    return {
        "focal_length_mm": fx * ISAAC_APERTURE_MM / width,
        "horizontal_aperture_mm": ISAAC_APERTURE_MM,
        "vertical_aperture_mm": ISAAC_APERTURE_MM * height / width * fx / fy,
        "hfov_deg": math.degrees(2 * math.atan(width / (2 * fx))),
        "principal_point_offset_px": [cx - width / 2, cy - height / 2],
        "note": "Isaac Lab PinholeCameraCfg ignores cx/cy offsets unless built via from_intrinsic_matrix [未查證 for our Isaac Lab version]",
    }


# ------------------------------------------------------------------------------------------ markers
def parse_ids(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def cmd_markers(a):
    cv2 = _cv2()
    ids = parse_ids(a.ids)
    px = round(a.size_mm / 25.4 * a.dpi)
    margin = round(15 / 25.4 * a.dpi)
    cols = min(len(ids), a.per_row)
    rows = math.ceil(len(ids) / cols)
    sheet = np.full((rows * (px + 2 * margin), cols * (px + 2 * margin)), 255, np.uint8)
    for k, mid in enumerate(ids):
        r, c = divmod(k, cols)
        y0, x0 = r * (px + 2 * margin) + margin, c * (px + 2 * margin) + margin
        sheet[y0:y0 + px, x0:x0 + px] = cv2.aruco.generateImageMarker(_dictionary(), mid, px)
        cv2.putText(sheet, f"id {mid}  TOP -> +X (away from operator)", (x0, y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, a.dpi / 400, 0, max(1, a.dpi // 150))
        cv2.putText(sheet, f"{a.size_mm:g} mm", (x0, y0 + px + margin // 2), cv2.FONT_HERSHEY_SIMPLEX, a.dpi / 400, 0,
                    max(1, a.dpi // 150))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, sheet)
    print(f"wrote {a.out}  ({len(ids)} x {DICT}, {a.size_mm} mm at {a.dpi} dpi)")
    print("🔴 print at 100 % (no 'fit to page') and measure one marker's black square with a ruler before use.")


def cmd_board(a):
    cv2 = _cv2()
    sq = round(a.square_mm / 25.4 * a.dpi)
    margin = round(10 / 25.4 * a.dpi)
    # squares = inner corners + 1
    nx, ny = a.cols + 1, a.rows + 1
    img = np.full((ny * sq + 2 * margin, nx * sq + 2 * margin), 255, np.uint8)
    for r in range(ny):
        for c in range(nx):
            if (r + c) % 2 == 0:
                img[margin + r * sq: margin + (r + 1) * sq, margin + c * sq: margin + (c + 1) * sq] = 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, img)
    print(f"wrote {a.out}  ({a.cols}x{a.rows} inner corners, {a.square_mm} mm squares at {a.dpi} dpi, "
          f"{nx * a.square_mm / 10:.1f} x {ny * a.square_mm / 10:.1f} cm)")
    print("🔴 print at 100 %, tape it FLAT to something rigid, and pass the MEASURED square size to `checkerboard`.")


# ------------------------------------------------------------------------------------------ intrinsics
def cmd_rs_intrinsics(a):
    import pyrealsense2 as rs

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(a.serial)
    cfg.enable_stream(rs.stream.color, a.width, a.height, rs.format.rgb8, a.fps)
    prof = pipe.start(cfg)
    try:
        intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        if a.save_image:
            # same stream and resolution as the intrinsics -> usable as the `extrinsics` shot.
            # Skip the first frames while auto-exposure settles.
            for _ in range(30):
                frames = pipe.wait_for_frames()
            rgb = np.asanyarray(frames.get_color_frame().get_data())
            cv2 = _cv2()
            Path(a.save_image).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(a.save_image, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            print(f"saved {a.save_image}")
    finally:
        pipe.stop()
    _write_json(a.out, {
        "source": f"pyrealsense2 get_intrinsics(), serial {a.serial}", "stream": "color",
        "width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
        "distortion_model": str(intr.model), "dist": list(intr.coeffs),
        "isaac": isaac_camera_fields(intr.fx, intr.fy, intr.ppx, intr.ppy, intr.width, intr.height),
    })


def cmd_capture(a):
    cv2 = _cv2()
    backend = getattr(cv2, f"CAP_{a.backend}") if a.backend else cv2.CAP_ANY
    cap = cv2.VideoCapture(int(a.index), backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    print("SPACE = save, q = quit. Move the board: corners, edges, tilted, near and far.")
    while True:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("🔴 camera read failed — wrong --index/--backend? (see configs/record_omx.yaml)")
        cv2.imshow("capture", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            if frame.shape[1] != a.width or frame.shape[0] != a.height:
                print(f"⚠️ got {frame.shape[1]}x{frame.shape[0]}, asked {a.width}x{a.height} — intrinsics would not match the recording")
            cv2.imwrite(str(out / f"frame_{n:03d}.png"), frame)
            n += 1
            print(f"saved {n}")
        elif k == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def cmd_checkerboard(a):
    cv2 = _cv2()
    pattern = (a.cols, a.rows)  # inner corners
    objp = np.zeros((a.cols * a.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square_mm / 1000.0
    obj_pts, img_pts, used, size = [], [], [], None
    for p in sorted(Path(a.images).glob("*.png")) + sorted(Path(a.images).glob("*.jpg")):
        gray = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, pattern)
        if not ok:
            print(f"  skip {p.name}: board not found")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        obj_pts.append(objp)
        img_pts.append(corners)
        used.append(p.name)
    if len(used) < 10:
        raise SystemExit(f"🔴 only {len(used)} usable images — take at least 10-15 (varied pose), not a single view")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    per_img = []
    for o, i, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        per_img.append(float(np.sqrt(np.mean(np.sum((proj - i) ** 2, axis=2)))))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    print(f"RMS reprojection {rms:.3f} px over {len(used)} images; worst image {max(per_img):.3f} px")
    if rms > 1.0:
        print("⚠️ RMS > 1 px: blurry shots, a non-flat board, or a wrong --square-mm/--cols/--rows")
    _write_json(a.out, {
        "source": f"cv2.calibrateCamera on {len(used)} images in {a.images}",
        "width": size[0], "height": size[1], "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "distortion_model": "opencv (k1,k2,p1,p2,k3)", "dist": dist.flatten().tolist(),
        "rms_px": rms, "per_image_rms_px": dict(zip(used, per_img)),
        "isaac": isaac_camera_fields(fx, fy, cx, cy, size[0], size[1]),
    })


# ------------------------------------------------------------------------------------------ extrinsics
def marker_corners_pan(cx: float, cy: float, cz: float, size_m: float) -> np.ndarray:
    """ArUco corner order TL, TR, BR, BL, marker lying flat with its top edge toward +X.
    Seen from above with +X up the page, +Y is to the LEFT."""
    h = size_m / 2
    return np.array([[cx + h, cy + h, cz], [cx + h, cy - h, cz], [cx - h, cy - h, cz], [cx - h, cy + h, cz]])


def load_marker_table(path: str) -> dict[int, tuple[float, float, float]]:
    table = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            table[int(row["id"])] = (float(row["x_pan_cm"]) / 100, float(row["y_pan_cm"]) / 100,
                                     float(row.get("z_cm") or 0.0) / 100)
    return table


def solve_extrinsics(img, intr: dict, table: dict, size_m: float) -> dict:
    cv2 = _cv2()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if (gray.shape[1], gray.shape[0]) != (intr["width"], intr["height"]):
        raise SystemExit(f"🔴 image {gray.shape[1]}x{gray.shape[0]} != intrinsics {intr['width']}x{intr['height']}")
    corners, ids, _ = cv2.aruco.ArucoDetector(_dictionary(), cv2.aruco.DetectorParameters()).detectMarkers(gray)
    if ids is None:
        raise SystemExit("🔴 no markers detected")
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], float)
    dist = np.array(intr.get("dist") or [0, 0, 0, 0, 0], float)
    seen = [int(i) for i in ids.flatten()]
    unknown = [i for i in seen if i not in table]
    use = [(i, c.reshape(4, 2)) for i, c in zip(seen, corners) if i in table]
    if len(use) < 2:
        raise SystemExit(f"🔴 {len(use)} known marker(s) detected (seen {seen}); need >= 2, ideally 4+ spread over the view")
    obj = np.concatenate([marker_corners_pan(*table[i], size_m) for i, _ in use])
    imgp = np.concatenate([c for _, c in use]).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise SystemExit("🔴 solvePnP failed")
    rvec, tvec = cv2.solvePnPRefineLM(obj, imgp, K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - imgp, axis=1)

    R, _ = cv2.Rodrigues(rvec)  # pan -> camera
    cam_pos = (-R.T @ tvec).flatten()  # camera centre in pan frame
    fwd = R.T @ np.array([0, 0, 1.0])  # optical axis in pan frame (ROS/OpenCV: z forward, x right, y down)
    lookat = cam_pos + fwd * (-cam_pos[2] / fwd[2]) if fwd[2] < -1e-6 else None
    Rc = R.T  # camera -> pan, columns = camera axes in pan frame
    qw = math.sqrt(max(0.0, 1 + Rc[0, 0] + Rc[1, 1] + Rc[2, 2])) / 2
    quat = [qw, (Rc[2, 1] - Rc[1, 2]) / (4 * qw), (Rc[0, 2] - Rc[2, 0]) / (4 * qw), (Rc[1, 0] - Rc[0, 1]) / (4 * qw)]

    per_marker = {}
    for k, (i, _) in enumerate(use):
        e = err[4 * k: 4 * k + 4]
        per_marker[i] = {"xyz_pan_cm": [round(v * 100, 2) for v in table[i]], "mean_px": float(e.mean()), "max_px": float(e.max())}
    median, worst = float(np.median(err)), float(err.max())
    return {
        "markers_used": [i for i, _ in use], "markers_seen_not_in_table": unknown,
        "camera_pos_pan_m": cam_pos.tolist(),
        "camera_quat_wxyz_ros": quat,
        "optical_axis_pan": fwd.tolist(),
        "pitch_down_deg": math.degrees(math.asin(-fwd[2])),
        "azimuth_deg": math.degrees(math.atan2(fwd[1], fwd[0])),
        "lookat_on_table_pan_m": lookat.tolist() if lookat is not None else None,
        "reprojection_px": {"median": median, "max": worst, "per_marker": per_marker},
        "t3_precondition": {"median_lt": T3_MEDIAN_PX, "max_lt": T3_MAX_PX, "pass": median < T3_MEDIAN_PX and worst < T3_MAX_PX},
        "scene_constants_hint": {
            "CAM_FRONT_LEFT_POS": [round(v, 4) for v in cam_pos],
            "CAM_FRONT_LEFT_LOOKAT": [round(v, 4) for v in lookat] if lookat is not None else None,
            "note": "scene_constants positions are relative to the pan axis on the table top — same frame as here",
        },
    }


def print_extrinsics(res: dict):
    print(f"markers used: {res['markers_used']}   seen but not in table: {res['markers_seen_not_in_table'] or 'none'}")
    p = res["camera_pos_pan_m"]
    print(f"camera at x={p[0]*100:.1f} y={p[1]*100:.1f} z={p[2]*100:.1f} cm  pitch-down {res['pitch_down_deg']:.1f} deg  "
          f"azimuth {res['azimuth_deg']:.1f} deg")
    print(f"{'id':>4}{'x,y,z (cm)':>24}{'mean px':>10}{'max px':>9}")
    for i, m in res["reprojection_px"]["per_marker"].items():
        print(f"{i:>4}{str(m['xyz_pan_cm']):>24}{m['mean_px']:10.2f}{m['max_px']:9.2f}")
    r = res["reprojection_px"]
    mark = "✅" if res["t3_precondition"]["pass"] else "🔴"
    print(f"{mark} reprojection median {r['median']:.2f} px, max {r['max']:.2f} px (need < {T3_MEDIAN_PX} / < {T3_MAX_PX})")
    print("   this is the PnP fit only; T3 proper compares a sim RENDER against the real image (S4 §5-5).")


def cmd_extrinsics(a):
    cv2 = _cv2()
    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"🔴 cannot read {a.image}")
    intr = json.loads(Path(a.intrinsics).read_text(encoding="utf-8"))
    res = solve_extrinsics(img, intr, load_marker_table(a.markers), a.size_mm / 1000)
    res.update({"image": a.image, "intrinsics": a.intrinsics, "markers_csv": a.markers, "marker_size_mm": a.size_mm})
    print_extrinsics(res)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {a.out}")


def cmd_selftest(a):
    """Synthetic end-to-end check of `extrinsics`: known pose in, same pose out."""
    cv2 = _cv2()
    W, H = 848, 480
    intr = {"width": W, "height": H, "fx": 425.0, "fy": 425.0, "cx": 421.0, "cy": 243.0, "dist": [0, 0, 0, 0, 0]}
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])
    size_m = 0.06
    table = {0: (0.30, 0.00, 0.0), 1: (0.22, 0.15, 0.0), 2: (0.35, -0.20, 0.0), 3: (0.15, -0.10, 0.0), 4: (0.40, 0.10, 0.0)}
    true_pos = np.array([0.62, 0.34, 0.42])
    target = np.array([0.26, 0.0, 0.0])
    z = target - true_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Rc = np.stack([x, y, z], axis=1)  # camera -> pan (ros: x right, y down, z forward)
    R = Rc.T
    rvec, _ = cv2.Rodrigues(R)
    tvec = -R @ true_pos
    img = np.full((H, W), 200, np.uint8)
    px = 400
    for mid, xyz in table.items():
        bmp = cv2.aruco.generateImageMarker(_dictionary(), mid, px)
        bmp = cv2.copyMakeBorder(bmp, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)  # quiet zone
        scale = (px + 100) / px
        src = np.array([[0, 0], [px + 100, 0], [px + 100, px + 100], [0, px + 100]], np.float32)
        dst3 = marker_corners_pan(*xyz, size_m * scale)
        dst, _ = cv2.projectPoints(dst3, rvec, tvec, K, None)
        Hm = cv2.getPerspectiveTransform(src, dst.reshape(4, 2).astype(np.float32))
        warped = cv2.warpPerspective(bmp, Hm, (W, H), borderValue=0, flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full_like(bmp, 255), Hm, (W, H), borderValue=0) > 127
        img[mask] = warped[mask]
    if a.save:
        cv2.imwrite(a.save, img)
        print(f"synthetic view saved to {a.save}")
    res = solve_extrinsics(img, intr, table, size_m)
    print_extrinsics(res)
    pos_err = np.linalg.norm(np.array(res["camera_pos_pan_m"]) - true_pos) * 1000
    look = res["lookat_on_table_pan_m"]
    look_err = np.linalg.norm(np.array(look) - target) * 1000
    print(f"\ntrue camera {true_pos.tolist()} -> recovered {np.round(res['camera_pos_pan_m'], 4).tolist()}  ({pos_err:.2f} mm)")
    print(f"true look-at {target.tolist()} -> recovered {np.round(look, 4).tolist()}  ({look_err:.2f} mm)")
    ok = pos_err < 5 and look_err < 10 and len(res["markers_used"]) == len(table)
    print("✅ selftest passed" if ok else "🔴 selftest FAILED")
    raise SystemExit(0 if ok else 1)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows cp950 console cannot print the status marks
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("markers", help="printable ArUco sheet")
    s.add_argument("--ids", default="0-5")
    s.add_argument("--size-mm", type=float, default=60.0)
    s.add_argument("--dpi", type=int, default=300)
    s.add_argument("--per-row", type=int, default=2)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_markers)

    s = sub.add_parser("board", help="printable checkerboard")
    s.add_argument("--cols", type=int, default=8, help="INNER corners per row")
    s.add_argument("--rows", type=int, default=5, help="INNER corners per column")
    s.add_argument("--square-mm", type=float, default=25.0)
    s.add_argument("--dpi", type=int, default=300)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_board)

    s = sub.add_parser("rs-intrinsics", help="T1: RealSense color intrinsics off the device")
    s.add_argument("--serial", required=True)
    s.add_argument("--width", type=int, default=848)
    s.add_argument("--height", type=int, default=480)
    s.add_argument("--fps", type=int, default=15)
    s.add_argument("--save-image", default=None, help="also save one color frame (the `extrinsics` shot)")
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_rs_intrinsics)

    s = sub.add_parser("capture", help="save frames from an OpenCV camera (UVC wrist)")
    s.add_argument("--index", required=True)
    s.add_argument("--backend", default="DSHOW", help="DSHOW / MSMF / V4L2 / '' for ANY")
    s.add_argument("--width", type=int, default=640)
    s.add_argument("--height", type=int, default=480)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_capture)

    s = sub.add_parser("checkerboard", help="T1: intrinsics from checkerboard images")
    s.add_argument("--images", required=True)
    s.add_argument("--cols", type=int, required=True, help="INNER corners per row")
    s.add_argument("--rows", type=int, required=True, help="INNER corners per column")
    s.add_argument("--square-mm", type=float, required=True, help="measured, not nominal")
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_checkerboard)

    s = sub.add_parser("extrinsics", help="T2: camera pose from ArUco markers on the mat")
    s.add_argument("--image", required=True)
    s.add_argument("--intrinsics", required=True, help="JSON from rs-intrinsics or checkerboard")
    s.add_argument("--markers", required=True, help="CSV: id,x_pan_cm,y_pan_cm[,z_cm]")
    s.add_argument("--size-mm", type=float, required=True, help="black-square side, MEASURED on the print")
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_extrinsics)

    s = sub.add_parser("selftest", help="synthetic known-pose round trip of `extrinsics` (no hardware)")
    s.add_argument("--save", default=None, help="also save the synthetic image")
    s.set_defaults(fn=cmd_selftest)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

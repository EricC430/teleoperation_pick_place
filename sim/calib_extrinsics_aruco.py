#!/usr/bin/env python
"""T2+T3 (S4 §5-5) -- solve a camera's extrinsics from ArUco markers taped to the placement mat
at measured points, and report the reprojection error that is the actual acceptance number
(S4 §7: median < 10 px, max < 25 px @ 848x480 -- the wrist stream is 640x480, see the printed
note about scaling that threshold).

Frame convention -- same as sim/scene_constants.py and sim/omx_scene_cfg.py:
  world origin = the pan axis (joint1) projected to (x=0, y=0); the arm is spawned at
  ARM_BASE_POS = (0, 0, TABLE_TOP_Z) (omx_scene_cfg.py). Marker points are given as
  (x_m, y_m, height_above_table_m) in the SAME x_pan_cm/y_pan_cm frame the placement-mat CSVs
  already use, and converted here as z_world = table_top_z + height_above_table_m. This means a
  marker CSV measured today stays correct even after TABLE_TOP_Z's current PLACEHOLDER (0.75 m,
  experiment_spec.md §3 blank row) is replaced by a real measurement -- only --table-top-z needs
  updating in one place, not every marker's z.

--camera front-left is a free-standing world prim: the output is a world pose, usable directly as
CAM_FRONT_LEFT_POS / a set_world_poses(convention="ros") call.

--camera wrist is parented to link5 (CAM_WRIST_PARENT_LINK) -- CameraCfg.OffsetCfg for it is
LOCAL to link5, not world. This script chains world_from_camera through the arm's own forward
kinematics (reach_logger.fk.link5_transform) using the joint positions the FOLLOWER actually held
when the photo was taken (--joint-pos is REQUIRED for --camera wrist). Read them with
`uv run python scripts/read_joint_pose.py` (read-only, prints the six follower `.pos` values) and
paste the six numbers as printed. 🔴 They are LeRobot `.pos` units (body joints RANGE_M100_100,
1 unit = 1.8 deg), NOT degrees, and not the leader's reading -- the follower can lag or sag
against the leader. They go through sim/joint_mapping.lerobot_to_urdf_rad, which carries the
measured per-joint scale and zero offsets (S6 / touch calibration); a plain radians() of the
readout is wrong by tens of degrees (wrist_flex alone has a ~112 deg offset).

--paper-offset-m dx dy dz: if the coordinate paper is known to sit slightly off the pan-axis frame
(e.g. its x origin is a few mm out), give the offset as X_true = X_paper + (dx, dy, dz) once it is
measured. It is a pure translation of every marker, so it CANNOT show up in T3 (the camera simply
solves to a shifted pose and the reprojection error stays the same) -- only an independent
reference (touching the gripper tip to a marker centre, scripts/touch_calibrate.py) can measure
it. Run today with 0 and re-run the same command with the measured value later; the output records
the offset used. Sensitivity: ~fx/Z px per metre, i.e. ~0.85 px per mm at 0.5 m for the D455.

🔴 The ArUco detection + solvePnP + FK-chain math below was self-tested end-to-end against
synthetic data (see the conversation this was written in) -- fake markers at known 3D points,
projected through a known camera pose, round-tripped back through this script's own functions.
It has NOT been run against a real photo yet. `[AI推論，數學自測過、真實照片未測]`

marker points CSV format (header required):
    marker_id,x_m,y_m,height_above_table_m
    0,0.15,0.20,0.0
    1,0.35,-0.10,0.0
    ...
>= 4 markers must be visible ACROSS --images COMBINED (>= 6, spread across the frame and not
collinear, is safer for a stable solve).

--images takes ONE OR MORE photos. Two ways to fill them:
  (a) one photo of a printed sheet with several markers taped down at once (cv2.aruco needs no
      camera besides the one being calibrated for this -- simplest, recommended).
  (b) several photos from a STATIC, UNMOVED camera, each showing one or two markers displayed on
      a phone screen and moved to a different measured point between shots (useful if there is no
      printer today). Each marker_id in --points-csv must appear in EXACTLY ONE of the photos --
      if the same id is detected in two photos at different pixel positions, that is treated as
      an error (most likely the same phone-displayed id was shown at two different points by
      mistake) and the script refuses to guess which one is right.
  🔴 (b)'s correctness depends entirely on the camera not moving between shots. Nothing in this
  script can verify that from the images alone -- a bumped camera produces a plausible-looking
  wrong answer, not a crash. Brace the camera (tripod / clamp), do not hand-hold it, and do not
  touch it between photos.

⚠️ Verified by synthetic testing while writing this: for a NEAR-GRAZING view of a flat (coplanar)
marker layout -- camera close to the table plane, or looking almost parallel to it -- solvePnP
can converge to a WRONG pose with near-zero reprojection error (a real, textbook planar-PnP
ambiguity, not a bug in this script). A well-conditioned shot (camera clearly above the table,
tilted down by a decent angle, markers spread across a good fraction of the frame -- not all
clustered in one corner) did not reproduce this in testing. Take the photo like that; if T3's
reprojection error looks suspiciously close to 0 while the printed pose looks physically
implausible (e.g. camera below the table), that is the symptom -- retake from a steeper angle.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "sim"))
import camera_distortion as cd  # noqa: E402
import joint_mapping as jm  # noqa: E402
from reach_logger.fk import N_JOINTS, link5_transform  # noqa: E402

SDK_CHECK_MAX_PX = 0.5  # right direction ~0.0x px, wrong direction ~10 px (see camera_distortion.py)


def load_intrinsics(path: Path) -> cd.CameraModel:
    model = cd.load_model(path)
    if model.kind == "rs_bc":
        d = json.loads(path.read_text(encoding="utf-8"))
        err = d.get("formula_vs_sdk_max_px")
        if err is None:
            raise SystemExit(
                f"{path} has RealSense distortion coefficients but no `formula_vs_sdk_max_px` -- re-run "
                "calib_intrinsics_realsense.py on the lab machine (a few seconds); it checks this repo's "
                "distortion formula against the SDK point by point, which this script depends on."
            )
        if err > SDK_CHECK_MAX_PX:
            raise SystemExit(f"{path}: formula_vs_sdk_max_px={err:.2f} px > {SDK_CHECK_MAX_PX} -- camera_distortion.py disagrees with the SDK for this camera; do not trust T2 until that is resolved")
    return model


def load_points(path: Path, table_top_z: float) -> dict[int, np.ndarray]:
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[int(row["marker_id"])] = np.array(
                [float(row["x_m"]), float(row["y_m"]), table_top_z + float(row["height_above_table_m"])]
            )
    return out


def detect_marker_centroids(image_bgr: np.ndarray, dict_name: str) -> dict[int, np.ndarray]:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i[0]): c.reshape(4, 2).mean(axis=0) for c, i in zip(corners, ids)}


def merge_marker_detections(image_paths: list[Path], dict_name: str, conflict_tol_px: float = 3.0):
    """Detect markers across one or more photos of the SAME STATIC camera and merge into one
    marker_id -> centroid map. Returns (merged, (width, height) of the first image).

    Same id detected in two photos at genuinely different pixel positions is refused, not
    averaged -- see the docstring's --images section for why (most likely a phone-displayed id
    was shown at two different measured points by mistake)."""
    merged: dict[int, np.ndarray] = {}
    sources: dict[int, Path] = {}
    size = None
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"could not read {path}")
        if size is None:
            size = (image.shape[1], image.shape[0])
        elif (image.shape[1], image.shape[0]) != size:
            print(f"⚠️ {path} is {image.shape[1]}x{image.shape[0]}, first image was {size[0]}x{size[1]} -- mixed resolutions in one --images set, check you photographed with the same camera settings")
        for mid, centroid in detect_marker_centroids(image, dict_name).items():
            if mid in merged:
                if np.linalg.norm(merged[mid] - centroid) > conflict_tol_px:
                    raise SystemExit(
                        f"marker {mid} detected in both {sources[mid]} (px {merged[mid].tolist()}) and "
                        f"{path} (px {centroid.tolist()}) at genuinely different positions -- fix the capture "
                        f"(each id must appear in exactly one photo) before re-running"
                    )
                # near-duplicate (e.g. same id visible in two overlapping shots) -- harmless, average it
                merged[mid] = (merged[mid] + centroid) / 2.0
            else:
                merged[mid] = centroid
                sources[mid] = path
    return merged, size


def matrix_to_quat_wxyz(rot_mat: np.ndarray) -> np.ndarray:
    """Shepperd's method -- numerically stable for all rotations, unlike the naive trace formula
    near 180-degree rotations (which this wrist-camera-pointing-back-at-the-arm case can hit)."""
    m = rot_mat
    t = np.trace(m)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def homogeneous(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = pos
    return t


def solve_world_from_camera(object_points: np.ndarray, image_points: np.ndarray, K: np.ndarray, dist: np.ndarray):
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise SystemExit("cv2.solvePnP failed to converge")
    R_cam_from_world, _ = cv2.Rodrigues(rvec)
    R_world_from_cam = R_cam_from_world.T
    t_world_from_cam = -R_world_from_cam @ tvec.reshape(3)
    reproj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    err_px = np.linalg.norm(reproj.reshape(-1, 2) - image_points, axis=1)
    return R_world_from_cam, t_world_from_cam, err_px


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", required=True, choices=["wrist", "front-left"])
    ap.add_argument("--intrinsics-json", type=Path, required=True, help="output of calib_intrinsics_realsense.py or calib_intrinsics_checkerboard.py")
    ap.add_argument("--points-csv", type=Path, required=True)
    ap.add_argument(
        "--images",
        type=Path,
        nargs="+",
        required=True,
        help="one or more photos taken with THIS camera (not a phone). One photo of printed markers, "
        "or several photos of a STATIC camera with a phone-displayed marker moved between shots -- see docstring",
    )
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--table-top-z", type=float, default=0.75, help="scene_constants.TABLE_TOP_Z -- currently a PLACEHOLDER, override once measured")
    ap.add_argument(
        "--joint-pos",
        type=float,
        nargs=6,
        default=None,
        metavar=("PAN", "LIFT", "ELBOW", "WRIST_FLEX", "WRIST_ROLL", "GRIPPER"),
        help="REQUIRED for --camera wrist: the FOLLOWER's six `.pos` values while the photo was taken, exactly as "
        "`scripts/read_joint_pose.py` prints them (LeRobot units, not degrees; the gripper value is ignored by FK)",
    )
    ap.add_argument(
        "--paper-offset-m",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="X_true = X_paper + this, for a coordinate paper known to sit off the pan-axis frame; see docstring",
    )
    ap.add_argument("--max-accept-px-median", type=float, default=10.0)
    ap.add_argument("--max-accept-px-max", type=float, default=25.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.camera == "wrist" and args.joint_pos is None:
        ap.error("--joint-pos is required for --camera wrist (see docstring)")

    model = load_intrinsics(args.intrinsics_json)
    K, w, h = model.K, model.width, model.height
    offset = np.array(args.paper_offset_m, dtype=np.float64)
    points_3d = {k: v + offset for k, v in load_points(args.points_csv, args.table_top_z).items()}
    if offset.any():
        print(f"paper offset applied to every marker: {offset.tolist()} m")

    if len(args.images) > 1:
        print(
            f"⚠️ {len(args.images)} images given -- this ONLY works if the camera was completely static across "
            f"all of them (see --images in the docstring). Nothing below can detect a camera that moved."
        )
    centroids, img_size = merge_marker_detections(args.images, args.dict)
    if img_size != (w, h):
        print(f"⚠️ images are {img_size[0]}x{img_size[1]}, intrinsics were measured at {w}x{h} -- results will be wrong. Re-check --images / --intrinsics-json match.")

    used_ids = sorted(set(centroids) & set(points_3d))
    missing_expected = sorted(set(points_3d) - set(centroids))
    if missing_expected:
        print(f"expected in --points-csv but not detected in --images: {missing_expected}")
    if len(used_ids) < 4:
        raise SystemExit(f"only {len(used_ids)} markers matched (need >= 4). detected ids: {sorted(centroids)}, csv ids: {sorted(points_3d)}")

    object_points = np.array([points_3d[i] for i in used_ids], dtype=np.float64)
    detected_px = np.array([centroids[i] for i in used_ids], dtype=np.float64)
    image_points = cd.undistort_pixels(detected_px, model)  # solve against an ideal pinhole, which is what the sim renders
    undistort_shift = np.linalg.norm(image_points - detected_px, axis=1)
    print(f"lens model '{model.kind}': detections moved {undistort_shift.min():.1f}-{undistort_shift.max():.1f} px by undistortion; residuals below are in ideal-pinhole pixels")

    R_wc, t_wc, err_px = solve_world_from_camera(object_points, image_points, K, np.zeros(5))

    print(f"used {len(used_ids)} markers: {used_ids}")
    for mid, e in zip(used_ids, err_px):
        print(f"  marker {mid}: reprojection error {e:.2f} px")
    p50, pmax = float(np.percentile(err_px, 50)), float(err_px.max())
    print(f"T3 reprojection error: median={p50:.2f} px  max={pmax:.2f} px  (image {w}x{h})")
    if p50 > args.max_accept_px_median or pmax > args.max_accept_px_max:
        print(f"⚠️ above S4 §7's proposed threshold (median<{args.max_accept_px_median}, max<{args.max_accept_px_max}, speced for 848x480) -- if this is the 640x480 wrist stream, that threshold was not re-derived for this resolution [AI推論]; otherwise re-check marker measurements / intrinsics before trusting this extrinsic")
    else:
        print("within threshold")

    if args.camera == "front-left":
        quat = matrix_to_quat_wxyz(R_wc)
        result = {"camera": args.camera, "frame": "world", "pos_m": t_wc.tolist(), "quat_wxyz": quat.tolist()}
        print(f"\nscene_constants.py:\nCAM_FRONT_LEFT_POS = ({t_wc[0]:.4f}, {t_wc[1]:.4f}, {t_wc[2]:.4f})")
        print(f"# and switch cam_front_left in omx_scene_cfg.py from OffsetCfg(pos=...) look-at to also carry rot={tuple(round(x,4) for x in quat)} (ros convention)")
    else:
        joint_rad = jm.lerobot_to_urdf_rad(np.array(args.joint_pos, dtype=np.float64))[:N_JOINTS]
        arm_base_pos = np.array([0.0, 0.0, args.table_top_z])
        world_from_baselink = homogeneous(np.eye(3), arm_base_pos)
        world_from_link5 = world_from_baselink @ link5_transform(joint_rad)
        world_from_camera = homogeneous(R_wc, t_wc)
        link5_from_camera = np.linalg.inv(world_from_link5) @ world_from_camera

        pos = link5_from_camera[:3, 3]
        quat = matrix_to_quat_wxyz(link5_from_camera[:3, :3])
        result = {"camera": args.camera, "frame": "link5", "pos_m": pos.tolist(), "quat_wxyz": quat.tolist(), "joint_pos_at_capture": args.joint_pos}
        print(f"\nscene_constants.py:\nCAM_WRIST_OFFSET_POS = ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        print(f"CAM_WRIST_OFFSET_ROT = ({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})  # (w, x, y, z), ros convention")

    result.update(
        {
            "reprojection_err_p50_px": p50,
            "reprojection_err_max_px": pmax,
            "used_marker_ids": used_ids,
            "source_images": [str(p) for p in args.images],
            "source_points_csv": str(args.points_csv),
            "source_intrinsics_json": str(args.intrinsics_json),
            "table_top_z": args.table_top_z,
            "paper_offset_m": args.paper_offset_m,
            "world_from_camera_paper_frame": {"pos_m": (t_wc - offset).tolist(), "quat_wxyz": matrix_to_quat_wxyz(R_wc).tolist()},
            "lens_model": model.kind,
            "measured_at": dt.date.today().isoformat(),
        }
    )
    out = args.out or (_REPO / "calibration" / f"{dt.date.today().isoformat()}_camera_extrinsics_{args.camera}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

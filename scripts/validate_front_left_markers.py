"""Calibrate a planar front-left view using projective marker centres and leave-one-marker-out checks."""
import argparse, csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import camera_distortion as cd

def centre(uv):
    # The centre of a planar rectangle projects to the intersection of its diagonals.
    a = np.cross(np.r_[uv[0],1.], np.r_[uv[2],1.])
    b = np.cross(np.r_[uv[1],1.], np.r_[uv[3],1.])
    p = np.cross(a,b)
    if abs(p[2]) < 1e-9:
        raise ValueError("Degenerate marker diagonals")
    return p[:2] / p[2]

def solve(obj, uv, K):
    ok, rv, tv = cv2.solvePnP(obj, uv, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("Pose solve failed")
    rcw = cv2.Rodrigues(rv)[0]
    if np.any((obj @ rcw.T + tv.ravel())[:,2] <= 0):
        raise RuntimeError("Pose puts markers behind the camera")
    return rv, tv

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default="calibration/shots/2026-10-18_fl_world_check_v2.png")
    ap.add_argument("--points", default="calibration/2026-10-18_aruco_points.csv")
    ap.add_argument("--intrinsics", default="calibration/2026-10-07_camera_intrinsics_front-left.json")
    ap.add_argument("--out", default="calibration/2026-10-18_camera_extrinsics_front-left_v2.json")
    args = ap.parse_args()
    w = cd.load_model(ROOT / args.intrinsics)
    im = cv2.imread(str(ROOT / args.image))
    if im is None or im.shape[:2] != (w.height,w.width):
        raise ValueError("Missing image or image resolution does not match intrinsics")
    points = {int(r["marker_id"]): np.array([float(r["x_m"]),float(r["y_m"]),float(r["height_above_table_m"])])
              for r in csv.DictReader((ROOT / args.points).open(encoding="utf-8-sig"))}
    if any(abs(v[2]) > 1e-9 for v in points.values()):
        raise ValueError("This analysis expects markers on the paper plane z=0")
    prm = cv2.aruco.DetectorParameters()
    prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),prm)
    cs, ids, _ = det.detectMarkers(im)
    if ids is None:
        raise ValueError("No markers detected")
    detections = {int(i): cd.undistort_pixels(c.reshape(4,2).astype(float),w)
                  for c,i in zip(cs,ids.ravel()) if int(i) in points}
    labels = sorted(detections)
    if len(labels) < 5:
        raise ValueError("Need at least five markers for four-point fits with one held out")
    obj = np.array([points[i] for i in labels])
    uv = np.array([centre(detections[i]) for i in labels])
    rv, tv = solve(obj,uv,w.K)
    pred = cv2.projectPoints(obj,rv,tv,w.K,None)[0].reshape(-1,2)
    errors = np.linalg.norm(pred-uv,axis=1)
    loo = {}
    for k, mid in enumerate(labels):
        keep = np.arange(len(labels)) != k
        r,t = solve(obj[keep],uv[keep],w.K)
        p = cv2.projectPoints(obj[k:k+1],r,t,w.K,None)[0].reshape(2)
        loo[str(mid)] = float(np.linalg.norm(p-uv[k]))
    rwc = cv2.Rodrigues(rv)[0].T
    pos = -rwc @ tv.ravel()
    q = Rotation.from_matrix(rwc).as_quat()
    edges = {}
    for k, mid in enumerate(labels):
        corners = detections[mid]
        rays = np.c_[corners,np.ones(4)] @ np.linalg.inv(w.K).T @ rwc.T
        pw = pos + rays * (-pos[2]/rays[:,2:3])
        edges[str(mid)] = (np.linalg.norm(np.roll(pw,-1,axis=0)-pw,axis=1)*1000).tolist()
        measured = cd.distort_pixels(corners,w)
        cv2.polylines(im,[np.rint(measured).astype(np.int32)],True,(0,255,0),2)
        xy = tuple(np.rint(cd.distort_pixels(uv[k:k+1],w)[0]).astype(int))
        cv2.putText(im,"ID%d LOO %.2fpx"%(mid,loo[str(mid)]),xy,cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
    result = dict(camera="front-left",frame="world",coordinate_system="coordinate-paper frame, table surface z=0",
        pos_m=pos.tolist(),quat_wxyz=[float(q[3]),*q[:3].tolist()],
        convention="OpenCV camera axes (x right, y down, z forward)",table_top_z=0.,
        used_marker_ids=labels,source_images=[args.image],source_points_csv=args.points,
        source_intrinsics_json=args.intrinsics,lens_model=w.kind,measured_at="2026-10-18",
        method="SUBPIX corners; undistort corners then intersect diagonals; iterative solvePnP",
        reprojection_err_p50_px=float(np.median(errors)),reprojection_err_max_px=float(max(errors)),
        per_marker_error_px={str(i):float(e) for i,e in zip(labels,errors)},
        leave_one_marker_out_px=loo,leave_one_marker_out_median_px=float(np.median(list(loo.values()))),
        leave_one_marker_out_max_px=max(loo.values()),backprojected_edges_mm=edges,
        limitations=["Leave-one-marker-out uses the same image and coordinate survey; not independent world-frame accuracy.",
                      "Paper flatness, marker centre measurements, and paper-to-robot alignment are not validated.",
                      "Backprojected sizes are diagnostics, not replacement physical measurements.",
                      "To use in a world whose table is at z=0.75m, translate this paper-frame pose appropriately."])
    out=ROOT / args.out
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    overlay=out.with_name(out.stem+"_detections.jpg")
    cv2.imwrite(str(overlay),im)
    print("Markers:",labels)
    print("Fit median/max px:",result["reprojection_err_p50_px"],result["reprojection_err_max_px"])
    print("Leave-one-marker-out px:",loo)
    print("Backprojected marker edge lengths mm:",edges)
    print("Wrote",out,"and",overlay)
if __name__=="__main__": main()

"""Offline diagnosis of the 2026-10-18 wrist marker captures; never controls hardware."""
import argparse, csv, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "sim"), str(REPO)]
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import camera_distortion as cd
import touch_test as TT

def transform(x):
    t = np.eye(4); t[:3,:3] = R.from_rotvec(x[:3]).as_matrix(); t[:3,3] = x[3:6]
    return t

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="calibration/2026-10-18_wrist_diagnosis")
    args = ap.parse_args()
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    model = TT.FitModel(REPO / "calibration/2026-10-14_link_marker_fit.json")
    fitdata = json.loads((REPO / "calibration/2026-10-14_link_marker_fit.json").read_text())
    wm = cd.load_model(REPO / "calibration/2026-10-07_camera_intrinsics_wrist.json")
    fm = cd.load_model(REPO / "calibration/2026-10-07_camera_intrinsics_front-left.json")
    old = json.loads((REPO / "calibration/2026-10-15_camera_extrinsics_wrist.json").read_text())
    q = old["quat_wxyz"]
    x0 = np.r_[R.from_quat([*q[1:],q[0]]).as_rotvec(), old["pos_m"]]
    centres = {0: np.array([.2415,-.0238,0]), 2: np.array([.1391,.2147,0])}
    local = np.array([[-.030,.0295,0],[.030,.0295,0],[.030,-.0295,0],[-.030,-.0295,0]])
    prm = cv2.aruco.DetectorParameters()
    prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),prm)
    rows = list(csv.DictReader((REPO / "calibration/2026-10-18_wrist_grasp_check_captures.csv").open(encoding="utf-8")))
    rows = list({(r["pose_id"],r["camera"]):r for r in rows}.values())
    obs, front, angles = [], [], {0:[],2:[]}
    C = np.array(fitdata["world_from_camera"])
    for r in rows:
        path = REPO / r["image"].replace("\\","/")
        im = cv2.imread(str(path))
        if im is None: continue
        cs, ids, _ = det.detectMarkers(im)
        if ids is None: continue
        intr = wm if r["camera"] == "wrist" else fm
        for c,i in zip(cs,ids.ravel()):
            i = int(i)
            if i not in centres: continue
            uv = cd.undistort_pixels(c.reshape(4,2).astype(float),intr)
            if r["camera"] == "front-left":
                rays = np.c_[uv,np.ones(4)] @ np.linalg.inv(fm.K).T @ C[:3,:3].T
                corners = C[:3,3] + rays * (-C[2,3] / rays[:,2:3])
                centre = corners.mean(0)
                a = corners[1]-corners[0]
                angle = np.arctan2(a[1],a[0]); angles[i].append(angle)
                front.append(dict(pose=r["pose_id"],id=i,centre_m=centre.tolist(),nominal_delta_mm=((centre-centres[i])*1000).tolist(),
                                  edge_mm=(np.linalg.norm(np.roll(corners,-1,axis=0)-corners,axis=1)*1000).tolist()))
            else:
                st = np.array([float(r["state_"+j]) for j in TT.JOINTS])
                obs.append(dict(pid=r["pose_id"],id=i,T=model.link5(st),uv=uv,path=str(path)))
    # Initialize marker yaw using per-view planar PnP and the existing wrist mount.
    for o in obs:
        ok, rv, tv = cv2.solvePnP(local,o["uv"],wm.K,None,flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            axis = (o["T"] @ transform(x0))[:3,:3] @ cv2.Rodrigues(rv)[0][:,0]
            angles[o["id"]].append(float(np.arctan2(axis[1],axis[0])))
    yaw = [float(np.angle(np.mean(np.exp(1j*np.array(angles[i]))))) for i in (0,2)]
    initial = np.r_[x0,yaw]
    valid_ids = {"E00_F0046","E00_F0056","E08_F0030","E08_F0048","E08_F0065"}
    train = [o for o in obs if o["pid"] not in valid_ids]
    valid = [o for o in obs if o["pid"] in valid_ids]
    def world_corners(x,i):
        return local @ R.from_euler("z",x[6+(i==2)]).as_matrix().T + centres[i]
    def project(x,o):
        cw = np.linalg.inv(o["T"] @ transform(x))
        pc = world_corners(x,o["id"]) @ cw[:3,:3].T + cw[:3,3]
        pp = pc @ wm.K.T
        return pp[:,:2]/pp[:,2:3]
    def residual(x, oo):
        return np.concatenate([(project(x,o)-o["uv"]).ravel() for o in oo])
    sol = least_squares(residual,initial,args=(train,),max_nfev=1000)
    # All-view fit is diagnostic only; held-out numbers always use the train-only fit.
    allsol = least_squares(residual,sol.x,args=(obs,),max_nfev=1000)
    def stats(x,oo):
        e = np.linalg.norm(residual(x,oo).reshape(-1,2),axis=1)
        return dict(views=len(oo),median_corner_px=float(np.median(e)),max_corner_px=float(e.max()),rms_corner_px=float(np.sqrt(np.mean(e**2))))
    report = dict(status="diagnostic_only_do_not_activate",centres_m={str(i):v.tolist() for i,v in centres.items()},
        marker_dimensions_m=[.060,.059],assumptions=["Nominal c1/c9 positions, not independently surveyed",
        "60mm along decoded marker x, 59mm along decoded marker y; physical axis assignment unconfirmed",
        "Markers assumed planar at z=0; paper visibly uneven",
        "Front-left backprojection assumes 2026-10-14 fitted camera pose is still valid",
        "Marker yaws estimated jointly; no robot model changes"],
        front_left=front,train=stats(sol.x,train),validation=stats(sol.x,valid),all_fit=stats(allsol.x,obs),
        train_fit_parameters=sol.x.tolist(),all_fit_parameters=allsol.x.tolist(),observations=[])
    tiles=[]
    for o in sorted(obs,key=lambda o:o["pid"]):
        before = np.linalg.norm(project(initial,o)-o["uv"],axis=1)
        after = np.linalg.norm(project(sol.x,o)-o["uv"],axis=1)
        report["observations"].append(dict(pose=o["pid"],id=o["id"],split="validation" if o["pid"] in valid_ids else "train",
             old_mean_corner_px=float(before.mean()),new_mean_corner_px=float(after.mean())))
        im=cv2.imread(o["path"])
        for uv,col in [(o["uv"],(0,255,0)),(project(initial,o),(0,0,255)),(project(sol.x,o),(255,255,0))]:
            pts=cd.distort_pixels(uv,wm)
            cv2.polylines(im,[np.round(pts).astype(np.int32)],True,col,2)
        cv2.putText(im,o["pid"]+" "+("HOLDOUT" if o["pid"] in valid_ids else "TRAIN"),(8,450),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
        tiles.append(cv2.resize(im,(320,240)))
    while len(tiles)%3: tiles.append(np.zeros((240,320,3),np.uint8))
    cv2.imwrite(str(out/"corner_overlays.jpg"),np.vstack([np.hstack(tiles[j:j+3]) for j in range(0,len(tiles),3)]))
    # Test the unconfirmed 60/59mm axis assignment without changing nominal centres.
    local[:,:2] *= np.array([59/60,60/59])
    swapped=least_squares(residual,sol.x,args=(train,),max_nfev=1000)
    report["swapped_dimension_validation"]=stats(swapped.x,valid)
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    lines=["Offline wrist diagnosis; original calibration unchanged.",
           "Overlays: green=detected, red=old camera (jointly estimated marker yaw), cyan=train-only candidate.",
           "Pixel errors are measured in undistorted image coordinates.",
           json.dumps({k:report[k] for k in ("train","validation","all_fit","swapped_dimension_validation")},indent=2),
           "Front-left detected IDs: "+str(sorted({f["id"] for f in front}))]
    for i in (0,2):
        fs=[f for f in front if f["id"]==i]
        if fs:
            lines.append("Front-left ID%d nominal delta median (mm): %s; edge median (mm): %s"%(i,np.round(np.median([f["nominal_delta_mm"] for f in fs],axis=0),2),np.round(np.median([f["edge_mm"] for f in fs],axis=0),2)))
    for r in report["observations"]: lines.append(str(r))
    (out/"summary.txt").write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines))
    print("Wrote",out)
if __name__=="__main__": main()

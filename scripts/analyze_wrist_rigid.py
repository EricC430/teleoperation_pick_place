"""Offline rigid-marker wrist analysis. Uses measured joint readings, never hardware."""
import argparse,csv,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"sim"),str(ROOT)]
import cv2,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import touch_test as TT,camera_distortion as cd

def T(x):
 m=np.eye(4);m[:3,:3]=R.from_rotvec(x[:3]).as_matrix();m[:3,3]=x[3:6];return m

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--out",default="calibration/2026-10-18_wrist_rigid_analysis")
 args=ap.parse_args();out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
 wm=cd.load_model(ROOT/"calibration/2026-10-07_camera_intrinsics_wrist.json")
 fm=cd.load_model(ROOT/"calibration/2026-10-07_camera_intrinsics_front-left.json")
 model=TT.FitModel(ROOT/"calibration/2026-10-14_link_marker_fit.json")
 old=json.loads((ROOT/"calibration/2026-10-15_camera_extrinsics_wrist.json").read_text())
 q=old["quat_wxyz"];oldx=np.r_[R.from_quat([*q[1:],q[0]]).as_rotvec(),old["pos_m"]]
 points={int(r["marker_id"]):np.array([float(r["x_m"]),float(r["y_m"]),float(r["height_above_table_m"])])
         for r in csv.DictReader((ROOT/"calibration/2026-10-18_aruco_points_rigid.csv").open(encoding="utf-8"))}
 rows=list(csv.DictReader((ROOT/"calibration/2026-10-18_wrist_rigid_orientation_captures.csv").open(encoding="utf-8")))
 rows=list({(r["pose_id"],r["camera"]):r for r in rows}.values())
 prm=cv2.aruco.DetectorParameters();prm.cornerRefinementMethod=1
 det=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),prm)
 local=np.array([[-.03,.0295,0],[.03,.0295,0],[.03,-.0295,0],[-.03,-.0295,0]])
 obs=[];states={};inventory=[]
 for row in rows:
  path=ROOT/row["image"].replace("\\","/")
  if row["camera"]!="wrist":continue
  st=np.array([float(row["state_"+j])for j in TT.JOINTS]);states[row["pose_id"]]=st
  im=cv2.imread(str(path))
  if im is None:continue
  corners,ids,_=det.detectMarkers(im)
  inventory.append(dict(pose=row["pose_id"],ids=[]if ids is None else ids.ravel().tolist()))
  if ids is None:continue
  for c,i in zip(corners,ids.ravel()):
   i=int(i)
   if i not in points:continue
   obs.append(dict(pid=row["pose_id"],id=i,T=model.link5(st),uv=cd.undistort_pixels(c.reshape(4,2).astype(float),wm),path=str(path)))
 labels=sorted({o["id"]for o in obs});index={mid:k for k,mid in enumerate(labels)}
 # Init yaw from per-view PnP plus previous mount; optimize it rather than treating it as measured.
 yaw=[]
 for mid in labels:
  aa=[]
  for o in obs:
   if o["id"]!=mid:continue
   ok,rv,tv=cv2.solvePnP(local,o["uv"],wm.K,None,flags=cv2.SOLVEPNP_ITERATIVE)
   if ok:
    a=(o["T"]@T(oldx))[:3,:3]@cv2.Rodrigues(rv)[0][:,0]
    aa.append(np.arctan2(a[1],a[0]))
  yaw.append(float(np.angle(np.mean(np.exp(1j*np.array(aa))))))
 x0=np.r_[oldx,yaw]
 def projected(x,o,free=False):
  k=index[o["id"]]
  if free:
   M=T(x[6+6*k:12+6*k]);pw=local@M[:3,:3].T+M[:3,3]
  else:
   pw=local@R.from_euler("z",x[6+k]).as_matrix().T+points[o["id"]]
  cw=np.linalg.inv(o["T"]@T(x[:6]));pc=pw@cw[:3,:3].T+cw[:3,3]
  if np.any(pc[:,2]<=0):return np.full((4,2),1e6)
  pp=pc@wm.K.T;return pp[:,:2]/pp[:,2:3]
 def residual(x,items,free=False):
  return np.concatenate([(projected(x,o,free)-o["uv"]).ravel()for o in items])
 def stats(x,items,free=False):
  e=np.linalg.norm(residual(x,items,free).reshape(-1,2),axis=1)
  return dict(views=len({o["pid"]for o in items}),marker_observations=len(items),median_corner_px=float(np.median(e)),
              rms_corner_px=float(np.sqrt(np.mean(e*e))),max_corner_px=float(max(e)))
 splits={
  "roll_plus8_holdout":lambda o:"_r+8_" in o["pid"],
  "flex_minus2_holdout":lambda o:o["pid"].endswith("_f-2")
 }
 runs={};solutions={}
 for name,held in splits.items():
  train=[o for o in obs if not held(o)];valid=[o for o in obs if held(o)]
  sol=least_squares(residual,x0,args=(train,),max_nfev=1500)
  solutions[name]=sol.x
  runs[name]=dict(train=stats(sol.x,train),validation=stats(sol.x,valid),parameters=sol.x.tolist(),success=bool(sol.success))
 allsol=least_squares(residual,x0,args=(obs,),max_nfev=1500)
 # Diagnostic comparison: can arbitrary static rigid marker poses explain the residuals?
 free0=list(allsol.x[:6])
 for mid,a in zip(labels,allsol.x[6:]):free0.extend([0.,0.,a,*points[mid]])
 train=[o for o in obs if not splits["roll_plus8_holdout"](o)]
 valid=[o for o in obs if splits["roll_plus8_holdout"](o)]
 freesol=least_squares(residual,free0,args=(train,True),max_nfev=2000)
 freecheck=dict(train=stats(freesol.x,train,True),validation=stats(freesol.x,valid,True),
                parameters=freesol.x.tolist(),warning="Unconstrained marker poses are diagnostic, not surveyed geometry")
 expected={f"S{ep:02d}_F0050_r{roll:+d}_f{flex:+d}"for ep in [0,8]for roll in [-8,0,8]for flex in [-6,-4,-2]}
 missing=[dict(pose=pid,camera=cam)for pid in sorted(expected)for cam in ["wrist","front-left"]
          if not any(r["pose_id"]==pid and r["camera"]==cam for r in rows)]
 actual=[]
 for ep in [0,8]:
  for flex in [-6,-4,-2]:
   ss=[st for pid,st in states.items()if pid.startswith(f"S{ep:02d}_")and pid.endswith(f"_f{flex}")]
   actual.append(dict(episode=ep,commanded_flex_offset=flex,measured_flex_min=float(min(s[3]for s in ss)),
                  measured_flex_max=float(max(s[3]for s in ss))))
 per=[]
 tiles=[]
 for pid in sorted(states):
  items=[o for o in obs if o["pid"]==pid]
  im=cv2.imread(items[0]["path"])
  for o in items:
   e=np.linalg.norm(projected(solutions["roll_plus8_holdout"],o)-o["uv"],axis=1)
   per.append(dict(pose=pid,id=o["id"],mean_corner_px=float(e.mean()),
                split="validation"if splits["roll_plus8_holdout"](o)else"train"))
   for uv,col in [(o["uv"],(0,255,0)),(projected(solutions["roll_plus8_holdout"],o),(255,255,0))]:
    pts=cd.distort_pixels(uv,wm)
    cv2.polylines(im,[np.rint(pts).astype(np.int32)],True,col,2)
  cv2.putText(im,pid,(8,460),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2)
  tiles.append(cv2.resize(im,(320,240)))
 while len(tiles)%3:tiles.append(np.zeros((240,320,3),np.uint8))
 cv2.imwrite(str(out/"roll_holdout_overlays.jpg"),np.vstack([np.hstack(tiles[j:j+3])for j in range(0,len(tiles),3)]))
 # Compare a single paper-to-model rigid alignment against independent marker poses.
 def global_projected(x,o):
  align=T(x[6:12]);k=index[o["id"]]
  pw=local@R.from_euler("z",x[12+k]).as_matrix().T+points[o["id"]]
  pw=pw@align[:3,:3].T+align[:3,3]
  cw=np.linalg.inv(o["T"]@T(x[:6]));pc=pw@cw[:3,:3].T+cw[:3,3]
  pp=pc@wm.K.T;return pp[:,:2]/pp[:,2:3]
 def global_residual(x,items):
  return np.concatenate([(global_projected(x,o)-o["uv"]).ravel()for o in items])
 def global_stats(x,items):
  e=np.linalg.norm(global_residual(x,items).reshape(-1,2),axis=1)
  return dict(median_corner_px=float(np.median(e)),rms_corner_px=float(np.sqrt(np.mean(e*e))),max_corner_px=float(max(e)))
 gx0=np.r_[freesol.x[:6],np.zeros(6),allsol.x[6:]]
 gs=least_squares(global_residual,gx0,args=(train,),max_nfev=2000)
 sv=np.linalg.svd(gs.jac,compute_uv=False)
 globalcheck=dict(train=global_stats(gs.x,train),validation=global_stats(gs.x,valid),parameters=gs.x.tolist(),
    paper_to_model_translation_mm=(gs.x[9:12]*1000).tolist(),paper_to_model_rotation_deg=R.from_rotvec(gs.x[6:9]).as_euler("xyz",degrees=True).tolist(),
    jacobian_singular_values=sv.tolist(),warning="A global alignment can absorb arm-base or other model errors; not an independently measured paper shift")
 print("global paper alignment diagnostic",json.dumps({k:v for k,v in globalcheck.items()if k not in ["parameters","jacobian_singular_values"]}))
 # Independent repeat captures are evaluated without refitting camera or marker poses.
 repeat_obs=[]
 repeat_csv=ROOT/"calibration/2026-10-18_wrist_flex_diag_captures.csv"
 if repeat_csv.exists():
  for row in csv.DictReader(repeat_csv.open(encoding="utf-8")):
   if row["camera"]!="wrist":continue
   cs,ids,_=det.detectMarkers(cv2.imread(str(ROOT/row["image"].replace("\\","/"))))
   if ids is None:continue
   st=np.array([float(row["state_"+j])for j in TT.JOINTS])
   for c,i in zip(cs,ids.ravel()):
    if int(i)in labels:
     repeat_obs.append(dict(pid=row["pose_id"],id=int(i),T=model.link5(st),uv=cd.undistort_pixels(c.reshape(4,2).astype(float),wm)))
 repeat_check=stats(freesol.x,repeat_obs,True) if repeat_obs else None
 print("independent repeat validation",repeat_check)
 f40_obs=[]
 f40_csv=ROOT/"calibration/2026-10-18_wrist_rigid_f40_captures.csv"
 if f40_csv.exists():
  for row in csv.DictReader(f40_csv.open(encoding="utf-8")):
   if row["camera"]!="wrist":continue
   cs,ids,_=det.detectMarkers(cv2.imread(str(ROOT/row["image"].replace("\\","/"))))
   if ids is None:continue
   st=np.array([float(row["state_"+j])for j in TT.JOINTS])
   if not np.isfinite(st).all():raise ValueError("Invalid measured state in f40 captures")
   for c,i in zip(cs,ids.ravel()):
    if int(i)in labels:
     f40_obs.append(dict(pid=row["pose_id"],id=int(i),T=model.link5(st),uv=cd.undistort_pixels(c.reshape(4,2).astype(float),wm)))
 f40_check=stats(freesol.x,f40_obs,True)if f40_obs else None
 print("f40 measured-state validation without refitting",f40_check)
 result=dict(state_source="capture CSV state_* fields, from measured Present_Position averages; never commanded target",cross_pose_f40_validation=f40_check,independent_repeat_validation=repeat_check,global_alignment_diagnostic=globalcheck,status="diagnostic_only_no_calibration_activated",inventory=inventory,missing=missing,marker_ids=labels,
  marker_dimensions_m=[.06,.059],height_for_ids5_6_m=.00275,
  assumptions=["Nominal paper coordinates; ID5/6 replacement dimensions assumed same 60x59mm",
   "All printed markers modeled as rectangles; only ID5/6 are known to be mounted on cardboard",
   "Actual averaged joint readings used; commanded offsets are not treated as actual angles",
   "Holdouts use same setup, not independent world-coordinate truth",
   "Marker yaws initialized from all-view PnP but optimized on training observations; holdout initialization is not a strict blind test"],
  splits=runs,all_fit=stats(allsol.x,obs),all_fit_parameters=allsol.x.tolist(),free_marker_diagnostic=freecheck,
  measured_flex_groups=actual,per_marker_observations=per)
 (out/"report.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
 print("Wrist views",len(states),"marker observations",len(obs),"IDs",labels)
 print("Missing",missing)
 print("Measured flex groups",json.dumps(actual))
 for name,run in runs.items():print(name,json.dumps({k:v for k,v in run.items()if k!="parameters"}))
 print("all fit",result["all_fit"])
 print("free marker pose diagnostic",json.dumps({k:v for k,v in freecheck.items()if k!="parameters"}))
 print("Saved",out)
if __name__=="__main__":main()

"""Solve and validate the completed wrist batch from measured state_* only; never controls hardware."""
import argparse,csv,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"sim"),str(ROOT)]
import cv2,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import touch_test as TT,camera_distortion as cd,joint_mapping as jm

def T(x):
 m=np.eye(4);m[:3,:3]=R.from_rotvec(x[:3]).as_matrix();m[:3,3]=x[3:6];return m

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--out",default="calibration/2026-10-18_wrist_batch_solution")
 ap.add_argument("--max-prepost-drift-units",type=float,default=None,help="exclude logged poses whose measured joint change exceeds this value; never substitute after-capture readings")
 a=ap.parse_args();out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
 batch=ROOT/"calibration/2026-10-18_wrist_validation_batch"
 quality=json.loads((batch/"quality_report.json").read_text())
 model=TT.FitModel(ROOT/"calibration/2026-10-14_link_marker_fit.json")
 wm=cd.load_model(ROOT/"calibration/2026-10-07_camera_intrinsics_wrist.json")
 prior=json.loads((ROOT/"calibration/2026-10-18_wrist_rigid_analysis/report.json").read_text())
 labels=[1,4,5,6];marker_index={v:k for k,v in enumerate(labels)}
 params0=np.array(prior["free_marker_diagnostic"]["parameters"])
 assert labels==prior["marker_ids"]
 local=np.array([[-.030,.0295,0],[.030,.0295,0],[.030,-.0295,0],[-.030,-.0295,0]])
 detector_params=cv2.aruco.DetectorParameters();detector_params.cornerRefinementMethod=1
 detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),detector_params)
 files=[(p,"holdout_f60"if p.stem=="f60_captures"else"drift_end"if p.stem=="baseline_end_captures"else"train")
        for p in sorted(batch.glob("*_captures.csv"))]
 files += [(ROOT/"calibration/2026-10-18_wrist_rigid_orientation_captures.csv","train"),
           (ROOT/"calibration/2026-10-18_wrist_flex_diag_captures.csv","train"),
           (ROOT/"calibration/2026-10-18_wrist_rigid_f40_captures.csv","development_f40")]
 obs=[];poses=[];sources=[];missing=[];excluded_motion=[]
 g=np.array([0.,0.,-TT.SL.G])if model.g_in_base else model.base[:3,:3].T@np.array([0.,0.,-TT.SL.G])
 for path,split in files:
  sources.append(dict(csv=str(path.relative_to(ROOT)),split=split,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  with path.open(encoding="utf-8")as fh:rows=list(csv.DictReader(fh))
  rows=list({(r["pose_id"],r["camera"]):r for r in rows}.values())
  motion={}
  log=path.with_name(path.stem.replace("_captures","")+"_servo.jsonl")
  if log.exists():
   for line in log.read_text(encoding="utf-8").splitlines():
    record=json.loads(line);motion.setdefault(record["pose_id"],{})[record["phase"]]=record
  for row in rows:
   if row["camera"]!="wrist":continue
   snapshots=motion.get(row["pose_id"],{})
   if a.max_prepost_drift_units is not None and "before_capture"in snapshots and "after_capture"in snapshots:
    before=snapshots["before_capture"]["measured_lerobot"];after=snapshots["after_capture"]["measured_lerobot"]
    drift=max(abs(after[j]-before[j])for j in TT.JOINTS)
    if drift>a.max_prepost_drift_units:
     excluded_motion.append(dict(csv=str(path.relative_to(ROOT)),pose=row["pose_id"],max_measured_change_units=drift))
     continue
   state=np.array([float(row["state_"+j])for j in TT.JOINTS])
   if not np.isfinite(state).all():raise ValueError("Nonfinite measured state")
   image=ROOT/row["image"].replace("\\","/")
   im=cv2.imread(str(image))
   if im is None:missing.append(str(image));continue
   if im.shape[:2]!=(wm.height,wm.width):raise ValueError("Wrong image resolution")
   rawq=jm.SCALE_RAD_PER_UNIT["shoulder_lift"]*state[:5]+model.off
   tau=TT.SL.gravity_torques(TT.SL.chain_transforms(rawq,model.chain),model.inert,g)
   pose=dict(uid=str(path.relative_to(ROOT))+":"+row["pose_id"],pid=row["pose_id"],split=split,
             state=state,rawq=rawq,tau=tau,image=str(image))
   pi=len(poses);poses.append(pose)
   corners,ids,_=detector.detectMarkers(im)
   if ids is None:continue
   for c,mid in zip(corners,ids.ravel()):
    mid=int(mid)
    if mid not in labels:continue
    c=c.reshape(4,2).astype(float)
    if c[:,0].min()<3 or c[:,0].max()>im.shape[1]-4 or c[:,1].min()<3 or c[:,1].max()>im.shape[0]-4:continue
    obs.append(dict(pi=pi,id=mid,uv=cd.undistort_pixels(c,wm)))
 groups={name:[o for o in obs if poses[o["pi"]]["split"]==name]
         for name in ("train","development_f40","holdout_f60","drift_end")}
 frame_cache={}
 def frames(scales):
  key=tuple(float(v)for v in scales)
  if key in frame_cache:return frame_cache[key]
  result=[]
  for p in poses:
   q=p["rawq"].copy();q[1:4]+=model.sag*np.array(scales)*p["tau"]
   result.append(model.base@TT.SL.chain_transforms(q,model.chain)[-1])
  if len(frame_cache)>256:frame_cache.clear()
  frame_cache[key]=result
  return result
 fixed_frames={"original":frames([1,1,1]),"no_sag":frames([0,0,0])}
 def project_all(x,items,mode):
  ff=frames(x[-3:])if mode=="fit_sag"else fixed_frames[mode]
  camera=T(x[:6]);markers={mid:T(x[6+6*k:12+6*k])for k,mid in enumerate(labels)}
  inverses={}
  out=[]
  for o in items:
   pi=o["pi"]
   if pi not in inverses:inverses[pi]=np.linalg.inv(ff[pi]@camera)
   cw=inverses[pi];M=markers[o["id"]];pw=local@M[:3,:3].T+M[:3,3]
   pc=pw@cw[:3,:3].T+cw[:3,3]
   if np.any(pc[:,2]<=0):out.append(np.full((4,2),1e5));continue
   pp=pc@wm.K.T;out.append(pp[:,:2]/pp[:,2:3])
  return out
 def residual(x,items,mode):
  return np.concatenate([(p-o["uv"]).ravel()for p,o in zip(project_all(x,items,mode),items)])
 def stats(x,items,mode):
  e=np.linalg.norm(residual(x,items,mode).reshape(-1,2),axis=1)
  return dict(views=len({o["pi"]for o in items}),marker_observations=len(items),corners=len(e),
    median_corner_px=float(np.median(e)),rms_corner_px=float(np.sqrt(np.mean(e*e))),
    p95_corner_px=float(np.percentile(e,95)),max_corner_px=float(max(e)))
 results={};solutions={}
 for mode in ("original","no_sag","fit_sag"):
  x0=params0.copy()
  if mode=="fit_sag":
   best=min(("original","no_sag"),key=lambda n:results[n]["train"]["rms_corner_px"])
   x0=np.r_[solutions[best],np.ones(3)*(.5 if best=="no_sag"else 1.)]
   lo=np.r_[np.full(len(params0),-np.inf),[0,0,0]];hi=np.r_[np.full(len(params0),np.inf),[2,2,2]]
  else:lo=np.full(len(x0),-np.inf);hi=np.full(len(x0),np.inf)
  sol=least_squares(residual,x0,args=(groups["train"],mode),bounds=(lo,hi),loss="soft_l1",f_scale=2.,max_nfev=1000)
  solutions[mode]=sol.x
  # Model selection uses development f40 only; f60 and end-baseline are not optimized.
  result=dict(train=stats(sol.x,groups["train"],mode),development_f40=stats(sol.x,groups["development_f40"],mode),
              parameters=sol.x.tolist(),success=bool(sol.success),
              camera_mount_pos_mm=(sol.x[3:6]*1000).tolist())
  if mode=="fit_sag":result["sag_scale_factors"]=sol.x[-3:].tolist()
  results[mode]=result
  print(mode,"training",round(result["train"]["median_corner_px"],3),
        "development f40",round(result["development_f40"]["median_corner_px"],3),flush=True)
 selected=min(results,key=lambda n:results[n]["development_f40"]["rms_corner_px"])
 print("Selected using f40 development:",selected,flush=True)
 for mode,x in solutions.items():
  results[mode]["holdout_f60"]=stats(x,groups["holdout_f60"],mode)
  results[mode]["drift_end"]=stats(x,groups["drift_end"],mode)
  print(mode,"f60 holdout",json.dumps(results[mode]["holdout_f60"]),"end baseline",json.dumps(results[mode]["drift_end"]),flush=True)
 # Quantify repeatable parameters and observability for selected fit.
 x=solutions[selected]
 J=least_squares(residual,x,args=(groups["train"],selected),max_nfev=1).jac
 singular=np.linalg.svd(J,compute_uv=False)
 selected_result=results[selected]
 candidate=dict(status="candidate_only_not_activated",camera="wrist",frame="link5",
   pos_m=x[3:6].tolist(),quat_wxyz=None,model=selected,
   model_fit="calibration/2026-10-14_link_marker_fit.json",
   state_source="measured capture CSV state_*; no commanded angles",
   warning="Marker world poses fitted freely; pose accuracy in physical paper coordinates not independently established")
 q=R.from_rotvec(x[:3]).as_quat();candidate["quat_wxyz"]=[float(q[3]),*q[:3].tolist()]
 if selected=="fit_sag":candidate["sag_scale_factors"]=x[-3:].tolist()
 (out/"wrist_candidate.json").write_text(json.dumps(candidate,indent=2),encoding="utf-8")
 per=[];tiles=[]
 predicted=project_all(x,obs,selected)
 for o,p in zip(obs,predicted):
  pose=poses[o["pi"]];e=np.linalg.norm(p-o["uv"],axis=1)
  per.append(dict(uid=pose["uid"],pose=pose["pid"],split=pose["split"],marker_id=o["id"],
                  mean_corner_px=float(e.mean()),max_corner_px=float(e.max())))
 for pi in sorted({o["pi"]for o in obs if poses[o["pi"]]["split"]!="train"}):
  pose=poses[pi];im=cv2.imread(pose["image"])
  for o,p in zip(obs,predicted):
   if o["pi"]!=pi:continue
   for uv,col in ((o["uv"],(0,255,0)),(p,(255,255,0))):
    cv2.polylines(im,[np.rint(cd.distort_pixels(uv,wm)).astype(np.int32)],True,col,2)
  cv2.putText(im,pose["split"]+" "+pose["pid"],(6,460),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
  tiles.append(cv2.resize(im,(320,240)))
 while len(tiles)%3:tiles.append(np.zeros((240,320,3),np.uint8))
 cv2.imwrite(str(out/"held_out_overlays.jpg"),np.vstack([np.hstack(tiles[k:k+3])for k in range(0,len(tiles),3)]))
 report=dict(excluded_motion=excluded_motion,max_prepost_drift_units=a.max_prepost_drift_units,data_sufficiency="planned capture complete; sufficient for current offline model fitting and validation",
    capture_quality_status=quality["status"],missing_images=missing,state_source="measured state_* only",
    counts={k:dict(views=len({o["pi"]for o in v}),marker_observations=len(v))for k,v in groups.items()},
    total_read_wrist_images=len(poses),total_marker_observations=len(obs),sources=sources,models=results,
    selected_model=selected,selection_rule="Minimum development-f40 RMS; f60/end baseline excluded from fitting and selection",
    selected_train_jacobian_singular_values=singular.tolist(),
    marker_dimensions_m=[.060,.059],marker_ids=labels,per_observation=per,
    limitations=["Same setup throughout rigid-marker captures; original ID0/2 paper captures excluded",
      "60x59mm assumed for all included printed markers; 0.5mm size uncertainty and paper curvature remain",
      "Marker world poses are free fitted variables, not independent ground truth",
      "Image reprojection validates image geometry, not absolute fingertip grasp error in mm",
      "Fitted sag factors restricted to 0..2 times original per-axis coefficients",
      "No formal calibration, robot settings or PID changed"])
 (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
 print("Data counts",report["counts"],flush=True)
 print("Saved",out,flush=True)
if __name__=="__main__":main()

"""Joint common-paper-plane, arm-base, camera and gravity analysis from measured encoder readings."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"sim"),str(ROOT)]
import cv2,numpy as np
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative
from scipy.spatial.transform import Rotation as R
import touch_test as TT,camera_distortion as cd,joint_mapping as jm

def T(v):
 m=np.eye(4);m[:3,:3]=R.from_rotvec(v[:3]).as_matrix();m[:3,3]=v[3:6];return m

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--out",default="calibration/2026-10-18_wrist_common_plane")
 ap.add_argument("--warm-start-report",default=None,help="same-parameterization prior training solutions; never fit holdout data")
 args=ap.parse_args();out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
 model=TT.FitModel(ROOT/"calibration/2026-10-14_link_marker_fit.json")
 lens=cd.load_model(ROOT/"calibration/2026-10-07_camera_intrinsics_wrist.json")
 old=json.loads((ROOT/"calibration/2026-10-18_wrist_batch_solution/stationary_subset/report.json").read_text())
 labels=[1,4,5,6];mi={mid:k for k,mid in enumerate(labels)}
 points={int(r["marker_id"]):np.array([float(r["x_m"]),float(r["y_m"]),float(r["height_above_table_m"])])
         for r in csv.DictReader((ROOT/"calibration/2026-10-18_aruco_points_rigid.csv").open(encoding="utf-8"))}
 local=np.array([[-.030,.0295,0],[.030,.0295,0],[.030,-.0295,0],[-.030,-.0295,0]])
 batch=ROOT/"calibration/2026-10-18_wrist_validation_batch"
 files=[(f,"holdout_f60"if f.stem=="f60_captures"else"drift_end"if f.stem=="baseline_end_captures"else"train")
         for f in sorted(batch.glob("*_captures.csv"))]
 files += [(ROOT/"calibration/2026-10-18_wrist_rigid_orientation_captures.csv","train"),
           (ROOT/"calibration/2026-10-18_wrist_flex_diag_captures.csv","train"),
           (ROOT/"calibration/2026-10-18_wrist_rigid_f40_captures.csv","development_f40")]
 prm=cv2.aruco.DetectorParameters();prm.cornerRefinementMethod=1
 det=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),prm)
 poses=[];obs=[];excluded=[];sources=[]
 for f,split in files:
  sources.append(dict(csv=str(f.relative_to(ROOT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),split=split))
  motion={};log=f.with_name(f.stem.replace("_captures","")+"_servo.jsonl")
  if log.exists():
   for line in log.read_text(encoding="utf-8").splitlines():
    rr=json.loads(line);motion.setdefault(rr["pose_id"],{})[rr["phase"]]=rr
  with f.open(encoding="utf-8")as fh: rows=list(csv.DictReader(fh))
  for row in {(r["pose_id"],r["camera"]):r for r in rows}.values():
   if row["camera"]!="wrist":continue
   snapshots=motion.get(row["pose_id"],{})
   if "before_capture"in snapshots and"after_capture"in snapshots:
    a=snapshots["before_capture"]["measured_lerobot"];b=snapshots["after_capture"]["measured_lerobot"]
    drift=max(abs(a[j]-b[j])for j in TT.JOINTS)
    if drift>.1:excluded.append(dict(csv=str(f.relative_to(ROOT)),pose=row["pose_id"],drift_units=drift));continue
   state=np.array([float(row["state_"+j])for j in TT.JOINTS])
   if not np.isfinite(state).all():raise ValueError("Nonfinite measured state")
   im=cv2.imread(str(ROOT/row["image"].replace("\\","/")))
   if im is None:raise ValueError("Missing input photo")
   q=jm.SCALE_RAD_PER_UNIT["shoulder_lift"]*state[:5]+model.off
   links=TT.SL.chain_transforms(q,model.chain)
   basis=np.stack([TT.SL.gravity_torques(links,model.inert,e)for e in np.eye(3)],axis=1)
   pi=len(poses);poses.append(dict(q=q,basis=basis,split=split,pid=row["pose_id"],image=row["image"]))
   cs,ids,_=det.detectMarkers(im)
   if ids is None:continue
   for c,mid in zip(cs,ids.ravel()):
    mid=int(mid);c=c.reshape(4,2).astype(float)
    if mid not in labels:continue
    if c[:,0].min()<3 or c[:,0].max()>im.shape[1]-4 or c[:,1].min()<3 or c[:,1].max()>im.shape[0]-4:continue
    obs.append(dict(pi=pi,mid=mid,uv=cd.undistort_pixels(c,lens)))
 groups={name:[o for o in obs if poses[o["pi"]]["split"]==name]
         for name in("train","development_f40","holdout_f60","drift_end")}
 base0=model.base;rb0=R.from_matrix(base0[:3,:3])
 prior=np.array(old["models"]["fit_sag"]["parameters"])
 yaws=[R.from_rotvec(prior[6+6*k:9+6*k]).as_euler("xyz")[2]for k in range(4)]
 # [camera6, base6, yaw4, markerXY8, gravityXY2, cardHeight1, sagFactors3]
 gravity0=R.from_euler("xy",[0,np.radians(2.75)])
 level_base0=(gravity0.inv()*rb0).as_euler("xyz")
 x0=np.r_[prior[:6],level_base0,base0[:3,3],yaws,np.zeros(8),0,np.radians(2.75),.00275,[1,1,1]]
 assert len(x0)==30
 lower=np.full(30,-np.inf);upper=np.full(30,np.inf)
 lower[6:8]=-np.radians(1.);upper[6:8]=np.radians(1.)
 lower[8]=x0[8]-np.radians(3.);upper[8]=x0[8]+np.radians(3.)
 lower[9:12]=x0[9:12]-.03;upper[9:12]=x0[9:12]+.03
 lower[16:24]=-.005;upper[16:24]=.005
 lower[24:26]=np.radians([-1.,1.75]);upper[24:26]=np.radians([1.,3.75])
 lower[26]=.0025;upper[26]=.003
 lower[27:30]=0;upper[27:30]=2
 rawq=np.array([p["q"]for p in poses]);basis=np.array([p["basis"]for p in poses])
 cache={}
 def base_matrix(x):
  # Paper-from-level and level-from-base are independent rotations.
  # Gravity is vertical in level, not in the base or paper frame.
  B=np.eye(4)
  B[:3,:3]=(R.from_euler("xy",x[24:26])*R.from_euler("xyz",x[6:9])).as_matrix()
  B[:3,3]=x[9:12]
  return B
 def frame_matrices(x,mode):
  scale=x[27:30]if mode!="original_sag"else np.ones(3)
  key=tuple(np.r_[x[6:12],x[24:26],scale])
  if key in cache:return cache[key]
  B=base_matrix(x);gpaper=R.from_euler("xy",x[24:26]).apply([0,0,-TT.SL.G])
  gbase=B[:3,:3].T@gpaper
  tau=np.einsum("nij,j->ni",basis,gbase);q=rawq.copy();q[:,1:4]+=model.sag*scale*tau
  frames=[B@TT.SL.chain_transforms(qi,model.chain)[-1]for qi in q]
  if len(cache)>128:cache.clear()
  cache[key]=frames;return frames
 def projections(x,items,mode):
  ff=frame_matrices(x,mode);camera=T(x[:6]);world={}
  for mid,k in mi.items():
   centre=points[mid].copy();centre[:2]+=x[16+2*k:18+2*k]
   centre[2]=x[26]if mid in(5,6)else 0.
   world[mid]=local@R.from_euler("z",x[12+k]).as_matrix().T+centre
  inverses={};pred=[]
  for o in items:
   pi=o["pi"]
   if pi not in inverses:inverses[pi]=np.linalg.inv(ff[pi]@camera)
   cw=inverses[pi];pc=world[o["mid"]]@cw[:3,:3].T+cw[:3,3]
   if (pc[:,2]<=0).any():pred.append(np.full((4,2),1e5));continue
   pp=pc@lens.K.T;pred.append(pp[:,:2]/pp[:,2:3])
  return pred
 def data_res(x,items,mode):
  return np.concatenate([(p-o["uv"]).ravel()for p,o in zip(projections(x,items,mode),items)])
 def tilts(x):
  B=base_matrix(x);gv=R.from_euler("xy",x[24:26]).apply([0,0,-1.])
  desk=np.arccos(np.clip(-gv[2],-1,1))
  base=np.arccos(np.clip(np.dot(B[:3,2],-gv),-1,1))
  return np.degrees([desk,base])
 def full_res(x,mode,weak):
  dr=(rb0.inv()*R.from_matrix(base_matrix(x)[:3,:3])).as_rotvec()
  # These are historical measurement/regularization uncertainties, not fresh metrology.
  prior_res=list(dr/np.radians([1.,1.,2.]))+list((x[9:12]-base0[:3,3])/.010)
  prior_res+=list(x[16:24]/(.004 if weak else .002))
  desk,base=tilts(x)
  prior_res += [(desk-2.75)/(.8 if weak else .4),(base-.45)/(.7 if weak else .35),
                (x[26]-.00275)/.00025]
  if mode!="original_sag":prior_res+=list((x[27:30]-1.)/1.)
  else:prior_res+=list((x[27:30]-1.)/1e-3)
  image=data_res(x,groups["train"],mode)
  # Robustify only image residuals. Historical measurement priors remain quadratic.
  image=image*np.sqrt(2./(np.sqrt(1.+(image/2.)**2)+1.))
  return np.r_[image,prior_res]
 def stats(x,items,mode):
  ee=np.linalg.norm(data_res(x,items,mode).reshape(-1,2),axis=1)
  return dict(views=len({o["pi"]for o in items}),marker_observations=len(items),median_corner_px=float(np.median(ee)),
              rms_corner_px=float(np.sqrt(np.mean(ee*ee))),max_corner_px=float(max(ee)))
 results={};solutions={}
 warm={}
 if args.warm_start_report:
  warm_report=json.loads((ROOT/args.warm_start_report).read_text())
  if "level-from-base"not in warm_report.get("parameterization",""):
   raise ValueError("Warm start parameterization mismatch")
  warm=warm_report["models"]
 for name,mode,weak in [("common_plane_original_sag","original_sag",False),
                        ("common_plane_fit_sag","fit_sag",False),
                        ("common_plane_weaker_priors","fit_sag",True)]:
  initial=x0.copy()if not solutions else solutions["common_plane_original_sag"].copy()
  if name in warm:
   source=warm[name]
   if not source["success"] and "common_plane_weaker_priors"in warm:
    source=warm["common_plane_weaker_priors"]
   initial=np.array(source["parameters"])
  sol=least_squares(full_res,initial,args=(mode,weak),bounds=(lower,upper),loss="linear",max_nfev=300,ftol=1e-7)
  x=sol.x;solutions[name]=x
  desk,base=tilts(x)
  result=dict(train=stats(x,groups["train"],mode),development_f40=stats(x,groups["development_f40"],mode),
    parameters=x.tolist(),success=bool(sol.success),base_position_m=x[9:12].tolist(),
    base_rpy_deg=R.from_matrix(base_matrix(x)[:3,:3]).as_euler("xyz",degrees=True).tolist(),
    desk_tilt_to_gravity_deg=float(desk),base_tilt_to_gravity_deg=float(base),
    gravity_paper_unit=R.from_euler("xy",x[24:26]).apply([0,0,-1.]).tolist(),
    camera_mount_mm=(x[3:6]*1000).tolist(),marker_xy_corrections_mm=(x[16:24]*1000).reshape(4,2).tolist(),
    shared_cardboard_height_mm=float(x[26]*1000),sag_factors=(np.ones(3)if mode=="original_sag"else x[27:30]).tolist())
  results[name]=result
  print(name,"train",round(result["train"]["median_corner_px"],3),"dev f40",round(result["development_f40"]["median_corner_px"],3),
        "desk/base tilt deg",np.round([desk,base],3),flush=True)
 selected=min(results,key=lambda n:results[n]["development_f40"]["rms_corner_px"])
 for name,x in solutions.items():
  mode="original_sag"if name=="common_plane_original_sag"else"fit_sag"
  results[name]["holdout_f60"]=stats(x,groups["holdout_f60"],mode)
  results[name]["drift_end"]=stats(x,groups["drift_end"],mode)
  print(name,"holdout",json.dumps(results[name]["holdout_f60"]),"end",json.dumps(results[name]["drift_end"]),flush=True)
 # Data-only observability, normalized by explicit parameter scales; no prior rows included.
 x=solutions[selected];mode="original_sag"if selected=="common_plane_original_sag"else"fit_sag"
 scales=np.r_[np.full(3,np.radians(1)),np.full(3,.005),np.full(3,np.radians(1)),np.full(3,.005),
              np.full(4,np.radians(5)),np.full(8,.002),np.full(2,np.radians(.4)),.00025,np.ones(3)]
 J=approx_derivative(lambda z:data_res(z,groups["train"],mode),x,method="2-point")
 sv=np.linalg.svd(J*scales,compute_uv=False)
 report=dict(status="diagnostic_only_not_activated",state_source="measured state_* averages only",
   common_plane="Paper coordinates define z=0, shared normal; ID1/4 z=0, ID5/6 common cardboard height 2.5..3mm",
   gravity="Independent paper-frame gravity direction, constrained by historical inclinometer magnitudes; not assumed aligned with base",
   base="Six pose parameters refitted; roll/pitch relative to level bounded +/-1deg, yaw +/-3deg around prior. Paper-relative pose derived jointly with gravity.",
   gravity_bounds_deg={"paper_level_rotation_x":[-1,1],"paper_level_rotation_y":[1.75,3.75]},
   parameterization="camera6, level-from-base Euler3 and paper base position3, marker yaw4, marker XY8, paper-from-level XY2, card height1, sag scale3",
   image_loss="soft_l1 equivalent applied only to image rows; historical priors quadratic",
   historical_priors=dict(desk_tilt_deg=2.75,desk_sigma_deg=.4,base_tilt_deg=.45,base_sigma_deg=.35,
       provenance="solve_link_markers.py historical 10/15 notes; 10/13 level-check axes/sign notes are not fully consistent",
       caution="Magnitude priors; not a new 10/18 measurement. Gravity direction and sag can trade off."),
   marker_xy_prior_sigma_mm=2.,marker_xy_bounds_mm=5.,marker_dimensions_m=[.060,.059],
   sources=sources,excluded_motion=excluded,models=results,selected_by_f40=selected,
   normalized_data_jacobian_singular_values=sv.tolist(),normalized_data_jacobian_rank_1e6=int(np.sum(sv>sv[0]*1e-6)),
   limitations=["Desk plane is one common geometric plane; real local paper warp is not modeled",
      "Only relative base-to-paper pose is estimated; absolute world yaw is a coordinate choice",
      "Gravity direction is inferred with historical priors and sag model, not independently measured by images",
      "Relative base/table inclinations are constrained, not independent evidence that both historical readings still hold",
      "f40 development chooses candidates; f60 and end baseline remain outside fitting",
      "No formal calibration or PID changed; candidate is not a validated mm-level fingertip calibration"])
 (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
 print("Selected by f40:",selected,"normalized data rank",report["normalized_data_jacobian_rank_1e6"],"/",len(x),flush=True)
 print("Saved",out,flush=True)
if __name__=="__main__":main()

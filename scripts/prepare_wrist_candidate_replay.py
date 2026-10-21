"""Prepare ep00 render inputs for a diagnostic common-plane wrist candidate; never activates calibration."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"sim"),str(ROOT)]
import numpy as np
from scipy.spatial.transform import Rotation as R
import touch_test as TT,joint_mapping as jm,capture_link_markers as CL

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--out",default="outputs/replay_cl/calib_20261018")
 a=ap.parse_args();out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
 source=Path("C:/Users/User/AppData/Local/Temp/claude/d--teleoperation-pick-place/707575c8-7472-4958-9ec8-313cfcc8777a/scratchpad/replay_cl/cl_ep00")
 report_path=ROOT/"calibration/2026-10-18_wrist_common_plane/refined/report.json"
 report=json.loads(report_path.read_text());name=report["selected_by_f40"];candidate=report["models"][name]
 if not candidate["success"]:raise ValueError("Candidate not converged")
 x=np.array(candidate["parameters"])
 rg=R.from_euler("xy",x[24:26]);B=np.eye(4);B[:3,:3]=(rg*R.from_euler("xyz",x[6:9])).as_matrix();B[:3,3]=x[9:12]
 gp=rg.apply([0,0,-9.81]);gb=B[:3,:3].T@gp
 model=TT.FitModel(ROOT/"calibration/2026-10-14_link_marker_fit.json")
 E=json.loads((source/"episode.json").read_text());G=json.loads((source/"gripper.json").read_text())
 actual=np.array(E["state"],float)
 root="D:/hf/lerobot/ericc430/"+E["dataset"]
 dataset_actual=np.array([s for k,s in CL._load_episodes(root)[0]])
 assert np.array_equal(actual,dataset_actual)
 qb=[]
 for state in actual:
  q=jm.SCALE_RAD_PER_UNIT["shoulder_lift"]*state[:5]+model.off
  tau=TT.SL.gravity_torques(TT.SL.chain_transforms(q,model.chain),model.inert,gb)
  q=q.copy();q[1:4]+=model.sag*np.array(candidate["sag_factors"])*tau;qb.append(q.tolist())
 E["q_body_rad"]=qb
 E["calibration_source"]=str(report_path.relative_to(ROOT))
 E["pose_source"]="recorded observation.state verified against dataset parquet"
 inputs=out/"inputs";inputs.mkdir(exist_ok=True)
 for f,obj in [("episode.json",E),("gripper.json",G)]:
  (inputs/f).write_text(json.dumps(obj),encoding="utf-8")
 C={"fit":{"name":"10/18 common-plane joint candidate (diagnostic)","world_from_link0":B.tolist(),"gravity_direction":(gp/9.81).tolist()}}
 (inputs/"candidate.json").write_text(json.dumps(C,indent=2),encoding="utf-8")
 q=R.from_rotvec(x[:3]).as_quat()
 wrist={"camera":"wrist","frame":"link5","pos_m":x[3:6].tolist(),"quat_wxyz":[float(q[3]),*q[:3].tolist()],
        "convention":"OpenCV x-right y-down z-forward","status":"diagnostic only, not activated","candidate":name}
 (inputs/"wrist_candidate.json").write_text(json.dumps(wrist,indent=2),encoding="utf-8")
 cup_source=source.parents[1]/"cup95.usd"
 # source is scratchpad/replay_cl/cl_ep00; cup95 is in scratchpad.
 cup_source=source.parent.parent/"cup95.usd"
 if not cup_source.exists():raise FileNotFoundError(cup_source)
 shutil.copy2(cup_source,inputs/"paper_cup_reference_95mm.usd")
 manifest=dict(reference_video="outputs/replay_cl/ep00_real_vs_sim_2cam.mp4",source_episode=str(source),frames=len(actual)-1,fps=E["fps"],
   candidate_report=str(report_path.relative_to(ROOT)),candidate_name=name,
   state_verified_exact_against_parquet=True,gripper_target_source="state",
   camera_intrinsics="calibration/2026-10-07_camera_intrinsics_wrist.json",
   cup_asset="reference-era 9.5cm cup retained for visual comparison",
   rendering="Physics replay driven by measured states; rendered articulation may retain tracking error",
   limitations=["Candidate has not passed absolute calibration validation",
    "Current common-plane base/mount/sag/gravity are applied together",
    "Real video is 10/14; candidate estimated on 10/18, setup may differ",
    "Reference physics gripper was action-driven; new render uses state, affecting cup dynamics",
    "No temporal lag compensation applied"],source_sha256=hashlib.sha256((source/"episode.json").read_bytes()).hexdigest())
 (out/"render_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
 assert len(qb)==217 and np.isfinite(qb).all()
 print("Prepared",len(actual)-1,"render frames in",inputs)
 print("Candidate",name,"gripper source=state; original calibrations unchanged")
if __name__=="__main__":main()

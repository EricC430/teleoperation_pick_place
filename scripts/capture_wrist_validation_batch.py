"""One supervised batch for wrist calibration coverage. Captures measured states and checks quality in-session."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,shutil,sys
from pathlib import Path
from argparse import Namespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import cv2,numpy as np
import capture_link_markers as C

DEFAULT_DATASET="D:/hf/lerobot/ericc430/rollout_omx_b1_papercup_60_13_10rcvrycut_46far_aug_fldrop30_100k_nas30_paper_cup_20261014_151208"
STAGES=[("baseline_start",50,"0","-4"),("f30",30,"0,8,16","-6,-2"),
        ("f60",60,"0,8,16","-6,-2"),("f70",70,"0,8,16","-6,-2"),
        ("baseline_end",50,"0","-4")]

def scan_args(options,name,frame,rolls,flexes,episodes="0,8",dry_run=False):
    return Namespace(csv=str(options.out/(name+".csv")),plan="configs/link_marker_plan_farwrist.json",
        dataset_root=options.dataset_root,episodes=episodes,exclude_episodes="",n_episodes=10,path_every=1,
        stop_every=5,min_from_home=10.,min_spacing=6.,rolls=rolls,flexes=flexes,base_frame=frame,
        target_ids="5,6",min_tip_cm=11.,pan_to=None,min_pan=10.,dry_run=dry_run,
        cameras="wrist,front-left",port=options.port,id=options.id,ease=1.5,seconds_per_point=.2,
        settle=3.,max_start_jump=8.,step_seconds=2.,servo_diagnostics=True)

def expected_poses(frame,rolls,flexes,episodes=(0,8)):
    return {ep:[f"S{ep:02d}_F{frame:04d}_r{r:+.0f}_f{f:+.0f}"
                for r,f in C._serpentine([float(v)for v in rolls.split(",")],[float(v)for v in flexes.split(",")])]
            for ep in episodes}


def quality(csv_path,episodes=(0,8),minimum_views=2,expected=None):
    log=csv_path.with_name(csv_path.stem+"_captures.csv")
    rows=list(csv.DictReader(log.open(encoding="utf-8")))if log.exists()else[]
    last={(r["pose_id"],r["camera"]):r for r in rows}
    p=cv2.aruco.DetectorParameters();p.cornerRefinementMethod=1
    det=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),p)
    groups=[]
    for ep in episodes:
        target=5 if ep==0 else 6
        valid=[];bad=[];missing=[]
        pose_ids=sorted({pid for pid,cam in last if pid.startswith(f"S{ep:02d}_")}
                        |set(expected.get(ep,[]))if expected is not None
                        else{pid for pid,cam in last if pid.startswith(f"S{ep:02d}_")})
        for pid in pose_ids:
            good=True
            for cam in ("wrist","front-left"):
                row=last.get((pid,cam))
                path=Path(row["image"])if row else None
                if path is not None and not path.is_absolute():path=ROOT/path
                if path is None or not path.is_file() or cv2.imread(str(path))is None:
                    good=False;missing.append({"pose":pid,"camera":cam})
            if not good:continue
            row=last[(pid,"wrist")];path=Path(row["image"])
            if not path.is_absolute():path=ROOT/path
            image=cv2.imread(str(path));corners,ids,_=det.detectMarkers(image)
            chosen=None
            if ids is not None:
                for corner,mid in zip(corners,ids.ravel()):
                    if int(mid)==target:chosen=corner.reshape(4,2);break
            if chosen is None:
                bad.append({"pose":pid,"reason":"target not decoded"});continue
            if (chosen[:,0]<3).any()or(chosen[:,0]>image.shape[1]-4).any()or(chosen[:,1]<3).any()or(chosen[:,1]>image.shape[0]-4).any():
                bad.append({"pose":pid,"reason":"target too close to image edge"});continue
            state=np.array([float(row["state_"+j])for j in C.JOINTS])
            if not np.isfinite(state).all():
                bad.append({"pose":pid,"reason":"nonfinite measured state"});continue
            valid.append({"pose":pid,"actual_wrist_roll":float(state[4]),"actual_wrist_flex":float(state[3])})
        span=max(v["actual_wrist_roll"]for v in valid)-min(v["actual_wrist_roll"]for v in valid)if valid else 0.
        passed=len(valid)>=minimum_views and (minimum_views==1 or span>=4.) and not missing
        groups.append(dict(episode=ep,target_id=target,passed=passed,valid_views=len(valid),
            actual_roll_span_units=span,valid=valid,rejected=bad,missing_files=missing))
    return groups

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,default=Path("calibration/2026-10-18_wrist_validation_batch"))
    ap.add_argument("--dataset-root",default=DEFAULT_DATASET)
    ap.add_argument("--port",default="COM8")
    ap.add_argument("--id",default="2026-09-18_omx_follower")
    ap.add_argument("--dry-run",action="store_true",help="validate every planned and fallback pose; no serial port, camera, or output writes")
    args=ap.parse_args();os.chdir(ROOT)
    if not args.out.is_absolute():args.out=ROOT/args.out
    # Complete all model-clearance and source-frame checks before opening hardware.
    for name,frame,rolls,flexes in STAGES:
        assert C.cmd_scan(scan_args(args,name,frame,rolls,flexes,dry_run=True))==0
    for frame in (30,60,70):
        assert C.cmd_scan(scan_args(args,"fallback",frame,"0,8,16","-10,-8",dry_run=True))==0
    print("Main batch: 40 stops, both cameras at each stop. Fallbacks run only for insufficient groups.")
    print("Quality gate checks files, target corners, measured states and roll coverage; it does not certify calibration accuracy.")
    if args.dry_run:return 0
    if C.S7._ask("Enter = start full batch; q = quit: ").lower()=="q":return 0
    args.out.mkdir(parents=True,exist_ok=True)
    manifest={"status":"running","stages":[],"notes":[
        "Do not move cameras or markers during the batch.",
        "All analysis states come from measured Present_Position averages.",
        "Quality pass permits data analysis, not deployment of a calibration.",
        "Ctrl+C stops the batch; the arm may remain at an intermediate pose with torque on."]}
    for filename in ("2026-10-18_aruco_points_rigid.csv","2026-10-18_aruco_points_rigid_setup.json"):
        source=ROOT/"calibration"/filename;dest=args.out/filename
        if dest.exists()and dest.read_bytes()!=source.read_bytes():
            raise RuntimeError("Setup changed since batch began; use a new output directory")
        shutil.copy2(source,dest)
    def save():
        (args.out/"quality_report.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    save()
    original_ask=C.S7._ask
    C.S7._ask=lambda *a:""  # One initial confirmation; Ctrl+C remains available.
    try:
        for name,frame,rolls,flexes in STAGES:
            print(f"Batch stage {name}, frame {frame}")
            a=scan_args(args,name,frame,rolls,flexes)
            if C.cmd_scan(a)!=0:raise RuntimeError(f"{name}: scan stopped; check HOME before resuming")
            baseline=name.startswith("baseline")
            checks=quality(Path(a.csv),minimum_views=1 if baseline else 2,expected=expected_poses(frame,rolls,flexes))
            failed=[g["episode"]for g in checks if not g["passed"]]
            # Retry missing camera files in-place; completed poses are skipped.
            if any(g["missing_files"]for g in checks):
                if C.cmd_scan(a)!=0:raise RuntimeError("Incomplete-file retry stopped")
                checks=quality(Path(a.csv),minimum_views=1 if baseline else 2,expected=expected_poses(frame,rolls,flexes))
                failed=[g["episode"]for g in checks if not g["passed"]]
            recovery=None
            if failed and not baseline:
                recovery_name=name+"_extra"
                recovery_args=scan_args(args,recovery_name,frame,"0,8,16","-10,-8",episodes=",".join(map(str,failed)))
                if C.cmd_scan(recovery_args)!=0:raise RuntimeError("Recovery scan stopped")
                recovery=quality(Path(recovery_args.csv),episodes=tuple(failed),expected=expected_poses(frame,"0,8,16","-10,-8",tuple(failed)))
                for group in checks:
                    match=next((g for g in recovery if g["episode"]==group["episode"]),None)
                    if match is not None:
                        # Evaluate coverage using both main and recovery observations.
                        merged=group["valid"]+match["valid"]
                        span=max(v["actual_wrist_roll"]for v in merged)-min(v["actual_wrist_roll"]for v in merged)if merged else 0.
                        group["combined_passed"]=len(merged)>=2 and span>=4. and not group["missing_files"] and not match["missing_files"]
                failed=[g["episode"]for g in checks if not g.get("combined_passed",g["passed"])]
            manifest["stages"].append(dict(name=name,frame=frame,quality=checks,recovery=recovery,passed=not failed))
            save()
            print(name,"PASS"if not failed else f"INSUFFICIENT: episodes {failed}; continuing other coverage")
        manifest["status"]="data_quality_passed"if all(s["passed"]for s in manifest["stages"])else"data_quality_incomplete"
        save()
        print("Batch finished:",manifest["status"])
        print("Report:",args.out/"quality_report.json")
        if manifest["status"]=="data_quality_passed":
            print("Planned marker data and coverage checks passed. Markers may be removed; calibration accuracy still requires offline analysis.")
        else:
            print("Keep markers in place: planned data quality is incomplete. Review quality_report.json before further action.")
        return 0 if manifest["status"]=="data_quality_passed"else 2
    except BaseException as exc:
        manifest["status"]="interrupted_or_failed";manifest["error"]=str(exc);save()
        print("Stopped. Arm may be at an intermediate pose with torque on; batch did not command an automatic return.")
        raise
    finally:
        C.S7._ask=original_ask

if __name__=="__main__":raise SystemExit(main())

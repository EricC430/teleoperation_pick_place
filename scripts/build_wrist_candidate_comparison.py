"""Build frame-aligned original/current wrist comparisons from the supplied reference MP4."""
import argparse,json,subprocess,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
FFMPEG="D:/ffmpeg/bin/ffmpeg.exe"
W,H=640,360

def title(image,text):
    p=cv2.resize(image,(W,H),interpolation=cv2.INTER_AREA)
    cv2.rectangle(p,(0,0),(W,26),(0,0,0),-1)
    cv2.putText(p,text,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    return p

def encoder(path,size,fps):
    return subprocess.Popen([FFMPEG,"-nostdin","-loglevel","error","-y","-f","rawvideo","-pix_fmt","bgr24",
        "-video_size",f"{size[0]}x{size[1]}","-framerate",str(fps),"-i","pipe:0","-an",
        "-c:v","libx264","-preset","fast","-threads","2","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",str(path)],
        stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",default="outputs/replay_cl/calib_20261018")
    a=ap.parse_args();folder=ROOT/a.root
    reference=ROOT/"outputs/replay_cl/ep00_real_vs_sim_2cam.mp4"
    render=folder/"inputs/sim_fit_calib_20261018_clip40"
    cap=cv2.VideoCapture(str(reference))
    if not cap.isOpened():raise RuntimeError("Could not open reference")
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))==(1280,760)
    for k in range(1,n+1):
        for cam in ("wrist","front-left"):
            if not(render/f"sim_{cam}_{k:04d}.jpg").is_file():
                raise RuntimeError(f"Missing rendered frame {k} {cam}")
    out3=ROOT/"outputs/replay_cl/ep00_wrist_real_old_new_calib_20261018.mp4"
    out2=ROOT/"outputs/replay_cl/ep00_real_vs_sim_2cam_calib_20261018.mp4"
    p3=encoder(out3,(W*3,H+40),fps);p2=encoder(out2,(W*2,H*2+40),fps)
    preview=[];count=0;errors=[]
    try:
        for index in range(n):
            ok,ref=cap.read()
            if not ok:raise RuntimeError(f"Reference decode failed at {index}")
            k=index+1
            rw=ref[400:760,:640];old=ref[400:760,640:1280];rf=ref[40:400,:640]
            nw=cv2.imread(str(render/f"sim_wrist_{k:04d}.jpg"))
            nf=cv2.imread(str(render/f"sim_front-left_{k:04d}.jpg"))
            if nw is None or nf is None:raise RuntimeError(f"Render decode failure {k}")
            head3=np.zeros((40,3*W,3),np.uint8)
            cv2.putText(head3,f"EP00 f{k:03d}  t={k/fps:.2f}s | same frame index, no latency shift",(10,26),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
            row=np.hstack([title(rw,"REAL wrist"),title(old,"PREVIOUS SIM - reference video"),
                            title(nw,"NEW CANDIDATE - state targets, near clip 40mm")])
            three=np.vstack([head3,row]);p3.stdin.write(three.tobytes())
            head2=np.zeros((40,2*W,3),np.uint8)
            cv2.putText(head2,f"EP00 f{k:03d} t={k/fps:.2f}s | 10/18 CANDIDATE | wrist near clip 40mm",(10,26),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
            two=np.vstack([head2,np.hstack([title(rf,"REAL front-left"),title(nf,"CANDIDATE SIM front-left")]),
                         np.hstack([title(rw,"REAL wrist"),title(nw,"CANDIDATE SIM wrist")])])
            p2.stdin.write(two.tobytes())
            if k in (50,70,84,100):
                preview.append(cv2.resize(three,(1280,266),interpolation=cv2.INTER_AREA))
                cv2.imwrite(str(folder/f"wrist_compare_f{k:04d}.png"),three)
            count+=1
    finally:
        cap.release()
        for proc in (p3,p2):
            proc.stdin.close()
            stderr=proc.stderr.read().decode("utf-8",errors="replace")
            ret=proc.wait()
            if ret:errors.append(f"ffmpeg exit {ret}: {stderr}")
    if errors:raise RuntimeError("\n".join(errors))
    if count!=n:raise RuntimeError("Incomplete comparison output")
    cv2.imwrite(str(folder/"wrist_comparison_contact_sheet.png"),np.vstack(preview))
    log=json.loads((render/"physics_log.json").read_text())
    names=log["dof_names"];tracking={}
    for name in ("joint1","joint2","joint3","joint4","joint5","gripper_joint_1"):
        i=names.index(name)
        e=np.degrees(np.array([v["q_sim"][i]-v["q_cmd"][i]for v in log["log"]]))
        tracking[name]=dict(median_abs_deg=float(np.median(np.abs(e))),max_abs_deg=float(np.max(np.abs(e))))
    info=dict(frames=count,fps=fps,duration_s=count/fps,reference=str(reference),
        wrist_three_column=str(out3),two_camera=str(out2),same_frame_alignment="video index0 corresponds dataset frame1; no latency correction",
        rendered_tracking_error=tracking,gripper_source=log["gripper_target_source"],wrist_near_m=log["wrist_near_m"],render_directory=str(render),
        notes=["Diagnostic wrist near clip is 40mm because fitted camera intersects stock link5 mesh; default 5mm render is mostly occluded.",
               "Rendered simulation joint angles can deviate from measured-state targets due to physics.",
               "Candidate applies base, gravity, sag and camera changes together; not a camera-only perturbation.",
               "Previous reference gripper target used action; candidate render uses measured gripper state.",
               "Reference-era 95mm cup retained; candidate is estimated from a different date's setup."])
    (folder/"comparison_manifest.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    for path,size in ((out3,(1920,400)),(out2,(1280,760))):
        v=cv2.VideoCapture(str(path))
        assert v.isOpened() and int(v.get(cv2.CAP_PROP_FRAME_COUNT))==n
        assert abs(v.get(cv2.CAP_PROP_FPS)-fps)<1e-6
        assert(int(v.get(cv2.CAP_PROP_FRAME_WIDTH)),int(v.get(cv2.CAP_PROP_FRAME_HEIGHT)))==size
        v.release()
    print("Wrote",out3)
    print("Wrote",out2)
    print(count,"frames @",fps,"fps; output frame counts verified")
if __name__=="__main__":main()

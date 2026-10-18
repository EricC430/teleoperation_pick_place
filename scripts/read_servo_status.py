"""Read OMX servo registers without moving the arm or changing torque."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path

REGISTERS=("Torque_Enable","Operating_Mode","Hardware_Error_Status",
           "Min_Position_Limit","Max_Position_Limit","Current_Limit",
           "Goal_Position","Present_Position","Present_Current",
           "Present_Input_Voltage","Present_Temperature",
           "Position_P_Gain","Position_I_Gain","Position_D_Gain",
           "PWM_Limit","Goal_PWM","Present_PWM","Goal_Current",
           "Profile_Velocity","Profile_Acceleration","Position_Trajectory",
           "Present_Velocity","Moving","Moving_Status")

def read_status(bus,joints):
    out={}
    for joint in joints:
        values={}
        for reg in REGISTERS:
            try:
                values[reg]=bus.read(reg,joint,normalize=False,num_retry=2)
            except (KeyError,ValueError,NotImplementedError,RuntimeError,OSError) as exc:
                values[reg]={"read_error":str(exc)}
        for reg in ("Goal_Position","Present_Position"):
            try:
                values[reg+"_lerobot"]=bus.read(reg,joint,normalize=True,num_retry=2)
            except (KeyError,ValueError,NotImplementedError,RuntimeError,OSError) as exc:
                values[reg+"_lerobot"]={"read_error":str(exc)}
        goal=values["Goal_Position"];present=values["Present_Position"]
        if isinstance(goal,(int,float)) and isinstance(present,(int,float)):
            values["goal_minus_present_ticks"]=goal-present
        out[joint]=values
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port",default="COM8")
    ap.add_argument("--id",default="2026-09-18_omx_follower")
    ap.add_argument("--joints",default="wrist_flex")
    ap.add_argument("--dry-run",action="store_true",help="show the read plan without opening the serial port")
    ap.add_argument("--samples",type=int,default=1,help="number of read-only position/current samples")
    ap.add_argument("--interval",type=float,default=0.25,help="seconds between samples")
    args=ap.parse_args()
    if args.samples < 1 or args.interval < 0:
        ap.error("samples must be >=1 and interval >=0")
    joints=[j.strip() for j in args.joints.split(",")if j.strip()]
    allowed={"shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"}
    if not joints or set(joints)-allowed:
        ap.error("Unknown joint or empty joint list")
    if args.dry_run:
        print(json.dumps({"joints":joints,"registers":REGISTERS,"hardware_opened":False},indent=2))
        return 0
    from lerobot.robots.omx_follower import OmxFollower,OmxFollowerConfig
    robot=OmxFollower(OmxFollowerConfig(port=args.port,id=args.id,calibration_dir=Path(__file__).resolve().parents[1]/"calibration"))
    bus=robot.bus
    bus.connect()
    try:
        result={"read_only":True,"raw_values_use_device_register_units":True,
                "note":"Snapshot at the current pose; does not reconstruct load during earlier captures.",
                "motors":read_status(bus,joints)}
        if args.samples > 1:
            start=time.monotonic()
            samples=[]
            for k in range(args.samples):
                sample={"elapsed_s":round(time.monotonic()-start,3),"motors":{}}
                for joint in joints:
                    values={}
                    for reg in ("Goal_Position","Present_Position","Position_Trajectory","Present_Current","Present_PWM","Present_Velocity","Moving_Status"):
                        try:
                            values[reg]=bus.read(reg,joint,normalize=False,num_retry=2)
                        except (KeyError,ValueError,NotImplementedError,RuntimeError,OSError) as exc:
                            values[reg]={"read_error":str(exc)}
                    sample["motors"][joint]=values
                samples.append(sample)
                if k+1 < args.samples:
                    time.sleep(args.interval)
            result["samples"]=samples
        print(json.dumps(result,indent=2,default=str))
    finally:
        bus.disconnect(False)  # Preserve existing torque state; no servo write.
    return 0

if __name__=="__main__":
    raise SystemExit(main())

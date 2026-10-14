#!/usr/bin/env python
"""Read every follower motor's error / torque / load / voltage straight off the bus, and reboot the ones in error.

A red-blinking DYNAMIXEL has latched a Hardware_Error_Status bit and switched its torque off; only a reboot clears it
([已查證 2026-10-12] shoulder_lift and shoulder_pan both tripped Overload = bit 32 while an extended arm was held
near the paper; 2026-10-15 shoulder_lift again during touch_test's hold()). After a reboot the motor's torque is OFF:
support the arm, then `read_joint_pose.py --goto-home --home-episode 10` (it turns torque back on).

    uv run python scripts/dxl_status.py            # read only
    uv run python scripts/dxl_status.py --reboot   # reboot the motors whose error status is not 0
"""
import argparse
import sys
import time

import dynamixel_sdk as sdk

NAMES = {11: "shoulder_pan", 12: "shoulder_lift", 13: "elbow_flex", 14: "wrist_flex", 15: "wrist_roll", 16: "gripper"}
BITS = {1: "input voltage", 4: "overheating", 8: "motor encoder", 16: "electrical shock", 32: "overload"}
# X-series control table (XL430-W250 / XL330-M288 share these addresses)
ADDR = {"model": (0, 2), "torque": (64, 1), "error": (70, 1), "load": (126, 2), "voltage": (144, 2), "temp": (146, 1)}


def read(pk, ph, i, key):
    a, n = ADDR[key]
    fn = {1: pk.read1ByteTxRx, 2: pk.read2ByteTxRx}[n]
    v, c, _ = fn(ph, i, a)
    if c != sdk.COMM_SUCCESS:
        return None
    if key == "load" and v >= 32768:
        v -= 65536
    return v


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM8")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--reboot", action="store_true")
    args = ap.parse_args()
    ph, pk = sdk.PortHandler(args.port), sdk.PacketHandler(2.0)
    if not ph.openPort() or not ph.setBaudRate(args.baud):
        print(f"cannot open {args.port}")
        return 1
    bad = []
    try:
        for i, name in NAMES.items():
            e = read(pk, ph, i, "error")
            if e is None:
                print(f"  {i} {name:14s} no reply")
                continue
            t, ld, v, tc = read(pk, ph, i, "torque"), read(pk, ph, i, "load"), read(pk, ph, i, "voltage"), read(pk, ph, i, "temp")
            why = ", ".join(s for b, s in BITS.items() if e & b) or "ok"
            print(f"  {i} {name:14s} error {e:3d} ({why:16s}) torque {'on ' if t else 'off'}  load {ld / 10 if ld is not None else '?':>6}%"
                  f"  {v / 10 if v else '?'} V  {tc} C")
            if e:
                bad.append(i)
        if args.reboot and bad:
            print(f"\n重新啟動 {[NAMES[i] for i in bad]}（重新啟動後這幾顆的扭力是關的，先扶住手臂）……")
            for i in bad:
                c, _ = pk.reboot(ph, i)
                print(f"  reboot {i}: {pk.getTxRxResult(c)}")
            time.sleep(1.0)
            for i in bad:
                print(f"  {i} {NAMES[i]:14s} error now {read(pk, ph, i, 'error')}")
            print("接著：uv run python scripts/read_joint_pose.py --goto-home --home-episode 10")
        elif bad:
            print("\n有馬達在錯誤狀態。扶住手臂後加 --reboot 重新啟動。")
    finally:
        ph.closePort()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

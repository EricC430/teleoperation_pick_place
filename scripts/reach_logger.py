#!/usr/bin/env python3
"""S1 — Reach logger.  Spec: docs/specs/S1_reach_logger.md  (D023 script 1, D026).

Measure, on a lab day, the numbers experiment_spec.md 3 needs:
  * which azimuth range the arm cable constrains
  * r_outer  — farthest radius with a usable top-down grasp
  * r_inner  — nearest radius still graspable
  * r_outer_side — farthest reachable (side grasp only), for the record

Radius/azimuth come from forward kinematics off assets/omx_f/omx_f.urdf
(hand-coded — placo has no Windows wheel, see D026). If the on-site 5-pose
check fails, rerun with --fk-fallback tape and type tape-measure readings.

    uv run python scripts/reach_logger.py --help
    uv run python scripts/reach_logger.py --dry-run
    uv run python scripts/reach_logger.py --mode follower-only

Keys during the run:  o outer/top-down   s outer/side-only   i inner
                      a azimuth-limit    r reference (mat zero)   q save & quit
"""

from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from reach_logger import joints, robot_io, session, validation  # noqa: E402
from reach_logger.samples import SampleWriter, resolve_out_path  # noqa: E402

_DEFAULT_CONFIG = "configs/teleoperate_omx.yaml"
_DEFAULT_URDF = "assets/omx_f/omx_f.urdf"
_TInput = "input"  # indirection so tests could patch; unused here


def _today() -> str:
    return _dt.date.today().isoformat()


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_REPO,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reach_logger",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default=_DEFAULT_CONFIG, help="teleoperate_omx.yaml (port/id/calib)")
    p.add_argument("--urdf", default=_DEFAULT_URDF, help="omx_f URDF for FK")
    p.add_argument("--ee-frame", default="end_effector_link", help="EE frame name (parity with spec)")
    p.add_argument("--out", default=None, help="CSV out path (default analysis/reach_log_<date>.csv)")
    p.add_argument("--mode", choices=robot_io.MODES, default="teleop")
    p.add_argument("--fk-fallback", choices=("off", "tape"), default="off")
    p.add_argument("--fps", type=int, default=30, help="teleop servo loop rate")
    p.add_argument("--dry-run", action="store_true", help="print the plan, touch nothing, exit")
    return p


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def _pump_until_enter(robot, prompt: str) -> None:  # pragma: no cover - interactive
    """Wait for Enter. In teleop mode, keep servoing the follower to the leader
    meanwhile — otherwise the follower is torque-locked and never tracks while the
    operator is posing the arm (a plain input() would freeze the teleop loop)."""
    if getattr(robot, "mode", "") != "teleop":
        input(prompt)  # follower-only / tape: nothing to servo, just block on Enter
        return

    from pynput import keyboard

    print(prompt)
    pressed: list[bool] = []

    def on_press(key) -> bool:
        if key == keyboard.Key.enter:
            pressed.append(True)
            return False
        return True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    try:
        while not pressed:
            robot.step()
    finally:
        listener.stop()


def _ask_float(prompt: str) -> float | None:
    raw = _ask(prompt)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print(f"  not a number: {raw!r} — skipped")
        return None


def run_five_pose_validation(
    robot: robot_io.ReachRobot, calib: joints.JointCalibration
) -> validation.ValidationReport | None:
    """Interactive. Returns None if the operator opts to skip."""
    print("\n=== 5-pose FK validation (spec 5) ===")
    print("Move to each pose, then type the EE (x, y) you read off the table in cm.")
    print("Blank line at 'home' = skip validation (radius will fall back to tape).\n")
    names = ("home", "+J1", "+J2", "+J3", "gripper")
    checks: list[validation.PoseCheck] = []
    for name in names:
        _pump_until_enter(robot, f"  move the leader to [{name}] and press Enter…")
        jv = joints.joint_vector(robot.read_ticks(), calib)
        from reach_logger import fk

        ex, ey, _ = (v * 100.0 for v in fk.ee_position_m(jv))
        print(f"    FK predicts EE (x, y) = ({ex:.1f}, {ey:.1f}) cm")
        if name == "home" and not checks:
            probe = _ask("    measured x cm (blank = skip validation): ")
            if not probe:
                return None
            mx = float(probe)
        else:
            mx_val = _ask_float("    measured x cm: ")
            mx = mx_val if mx_val is not None else None  # type: ignore[assignment]
        my = _ask_float("    measured y cm: ")
        measured = None if (mx is None or my is None) else (float(mx), float(my))
        checks.append(
            validation.PoseCheck(
                name=name, joint_rad=list(jv), predicted_xy_cm=(ex, ey), measured_xy_cm=measured
            )
        )
    report = validation.evaluate(checks, tolerance_cm=1.0, date=_today())
    print("\n" + report.render_table() + "\n")
    return report


def keyboard_loop(robot, calib, method, writer, rows) -> None:  # pragma: no cover - interactive
    from pynput import keyboard

    print("\nready — o/s/i/a/r to log a sample, q to save & quit\n")
    pending: list[str] = []

    def on_press(key) -> None:
        try:
            pending.append(key.char)
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    try:
        while True:
            if robot_needs_step := (method != "tape" and getattr(robot, "mode", "") == "teleop"):
                robot.step()
            if not pending:
                continue
            ch = pending.pop(0)
            if ch == "q":
                break
            if ch not in session.KEY_TO_TYPE:
                continue
            listener.stop()
            row = _prompt_for_sample(ch, robot, calib, method)
            writer.append(row)
            rows.append(row)
            print(f"  logged {row.sample_type} (n={len(rows)})")
            listener = keyboard.Listener(on_press=on_press)
            listener.start()
    finally:
        listener.stop()


def _prompt_for_sample(ch, robot, calib, method):  # pragma: no cover - interactive
    ticks = robot_io.require_motors(robot.read_ticks())
    sample_type = session.KEY_TO_TYPE[ch]
    tape_cm = mat_deg = None
    if sample_type == "reference":
        mat_deg = _ask_float("  [reference] mat angle reading (deg) > ")
    elif method == "tape" and sample_type != "azimuth_limit":
        tape_cm = _ask_float(f"  [{sample_type}] tape reading (cm) > ")
    note = _ask(f"  [{sample_type}] note (optional) > ")
    return session.make_sample_row(
        ch, ticks, calib, method=method, now=_now_iso(), tape_cm=tape_cm, mat_deg=mat_deg, note=note
    )


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.out or f"analysis/reach_log_{_today()}.csv"

    if args.dry_run:
        print(session.plan_text(config=args.config, urdf=args.urdf, out=out, mode=args.mode))
        return 0

    out_path = resolve_out_path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    calib = joints.identity_calibration()
    method = "tape" if args.fk_fallback == "tape" else "fk"
    fk_validation = None

    robot = robot_io.build_robot(config_path=args.config, mode=args.mode, dry_run=False)
    rows: list = []
    with robot, SampleWriter(out_path) as writer:
        if method == "fk":
            report = run_five_pose_validation(robot, calib)
            if report is None:
                print("  validation skipped -> switching to tape fallback")
                method = "tape"
            else:
                fk_validation = report.result
                if not report.result.passed:
                    print("  5-pose validation FAILED -> switching to tape fallback (D026)")
                    method = "tape"
        keyboard_loop(robot, calib, method, writer, rows)

    result = session.finalize(
        rows, csv_path=out_path, fk_validation=fk_validation, git_commit=_git_commit(), now=_now_iso()
    )
    print("\n" + result.summary.render_text())
    print(f"\nwrote: {out_path}\n       {result.summary_json_path}\n       {result.plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

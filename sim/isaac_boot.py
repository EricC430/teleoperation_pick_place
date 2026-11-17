"""Start Isaac Sim (Kit) for a sim/ script -- the Isaac Lab 3.0 way, in one place.

Why this file exists
--------------------
Isaac Lab 2.x scripts did

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    app_launcher = AppLauncher(args); simulation_app = app_launcher.app

Isaac Lab 3.0 deprecates `AppLauncher` (kept only as a thin adapter that logs a warning) in favour of
`add_launcher_args` + the context manager `launch_simulation(cfg, args)`. Our scripts are flat, module-level
scripts, not `main()` functions, so wrapping the body in `with launch_simulation(...)` would re-indent
10 000 lines. This helper does exactly what the 3.0 adapter does (enter `launch_simulation` once, hold it
for the life of the process) without the deprecated class and its warning, and keeps the old call-site
shape: `simulation_app = isaac_boot.start(args)`.

`require_kit=True` is deliberate: every script here needs Kit (USD stage, RTX cameras, replicator).
Source: `isaaclab/app/app_launcher.py` and `isaaclab/app/sim_launcher.py` in Isaac Lab v3.0.0.
"""

from __future__ import annotations

import argparse
import contextlib

_STACK = contextlib.ExitStack()
"""Holds the launched runtime for the whole process, so it outlives this function's return."""


class _App:
    """The running Kit app. `close()` stops the runtime `start()` launched (what 2.x `app.close()` did);
    everything else (`update()`, `is_running()`, `get_extension_manager()`, ...) is the `omni.kit.app` object's."""

    def __init__(self, kit_app):
        self._kit_app = kit_app

    def close(self, *args, exit_code: int = 0, **kwargs) -> None:
        close(exit_code)

    def __getattr__(self, name):
        return getattr(self._kit_app, name)


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add the launcher CLI (`--headless`, `--enable_cameras`, `--device`, `--viz`, ...)."""
    from isaaclab.app import add_launcher_args

    add_launcher_args(parser)
    # 3.0's add_launcher_args no longer defines these two (the launcher still READS them from the args dict).
    # Keep them so every documented command line (`... --headless --enable_cameras`) keeps working: 3.0 runs
    # headless unless `--viz kit` is given, and auto-enables cameras only when it can see a camera in a cfg --
    # we pass `None` as the cfg (the scenes are built after launch), so `--enable_cameras` is still ours to pass.
    parser.add_argument("--headless", action="store_true", default=False, help="accepted for 2.x compatibility; 3.0 is headless by default")
    parser.add_argument("--enable_cameras", action="store_true", default=False, help="enable Kit camera rendering (required for RGB sensors)")


def start(args: argparse.Namespace):
    """Launch Kit and return the running app (an `omni.kit.app` object whose `close()` stops the runtime).

    Mutates `args` in place (the launcher writes resolved values such as `headless` back into it), as
    the 2.x `AppLauncher(args)` did.
    """
    from isaaclab.app import launch_simulation

    launcher_args = vars(args)
    launcher_args["require_kit"] = True
    _STACK.enter_context(launch_simulation(None, launcher_args))

    import omni.kit.app

    return _App(omni.kit.app.get_app())


def close(exit_code: int = 0) -> None:
    """Stop the runtime started by `start()` (a `sys.exit(exit_code)` unwind, as the adapter does)."""
    _STACK.__exit__(SystemExit, SystemExit(exit_code), None)

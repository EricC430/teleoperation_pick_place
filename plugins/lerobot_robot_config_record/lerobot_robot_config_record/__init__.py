"""LeRobot plugin: snapshot the YAML a lerobot command was started with, keyed by robot id (see recorder.py).

lerobot's register_third_party_plugins() runs first thing in lerobot-record / -teleoperate / -calibrate /
-replay / -rollout / -setup-motors and imports every installed `lerobot_robot_*` distribution, so importing this
package is the hook. It never raises: failing to record a config must not stop a robot command.
"""

import sys

from .recorder import record_from_argv

try:
    record_from_argv()
except Exception as e:  # noqa: BLE001 - report it, never block the robot command
    print(f"[config-record] WARNING: could not record the config: {e!r}", file=sys.stderr, flush=True)

#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# OPTIONAL: persistent device names for arms and cameras.
#
# READ THIS FIRST -- you probably don't need this script yet.
#
# The default workflow is LeRobot's own discovery tools:
#
#     lerobot-find-port        # identifies which serial port is which arm
#     lerobot-find-cameras     # lists cameras, lets you preview each one
#
# Run those at the start of a session, pass the results as arguments, done.
# ~30 seconds. That is LeRobot's intended workflow and it is sufficient for a
# small team on a single machine. Use it.
#
# Discovery vs. persistence -- these are different problems, not alternatives:
#   - find_port / find_cameras  -> DISCOVERY  ("where is it right now?")
#   - udev rules / by-id paths  -> PERSISTENCE ("always call it the same thing")
#
# You escalate from the first to the second only when discovery becomes a
# recurring cost or a source of silent errors.
#
# ---------------------------------------------------------------------------
# ESCALATE TO THIS SCRIPT ONLY IF ONE OF THESE HAS ACTUALLY HAPPENED:
#
#   [ ] A camera index swap silently corrupted a recording session
#       (wrist footage landed in the "front" key, or vice versa)
#   [ ] You are re-running find_port more than ~3x/day and it's eating time
#   [ ] Configs or scripts with hardcoded device paths keep breaking
#   [ ] Multiple people / multiple machines share the rig and disagree on names
#
# If none of the above has bitten you, close this file and go do something
# that moves the project forward.
#
# PREREQUISITE: sudo on the host machine. On a shared lab GPU box you may not
# have it -- in which case this path is closed and you stay on find_port.
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# STEP 1 -- discover identifiers (use LeRobot's tools, then go one level deeper)
# ---------------------------------------------------------------------------
#
#   Which port is which arm:
#     lerobot-find-port          # unplug one device when prompted
#
#   Then get the stable hardware identifiers for that port:
#     udevadm info -a -n /dev/ttyACM0 | grep -E 'ATTRS\{(idVendor|idProduct|serial)\}' | head
#
#   Cameras:
#     lerobot-find-cameras
#     ls -l /dev/v4l/by-id/
#
# Record everything in docs/hardware.md before continuing.

# ---------------------------------------------------------------------------
# STEP 2 -- write the udev rules
# ---------------------------------------------------------------------------
RULES_FILE=/etc/udev/rules.d/99-soarm.rules

cat <<'EOF' | sudo tee "$RULES_FILE" > /dev/null
# SO-ARM leader / follower -- bind by servo-board serial number.
# TODO: replace idVendor / idProduct / serial with values from STEP 1.
SUBSYSTEM=="tty", ATTRS{idVendor}=="TODO", ATTRS{idProduct}=="TODO", ATTRS{serial}=="TODO_LEADER",   SYMLINK+="so101_leader",   MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="TODO", ATTRS{idProduct}=="TODO", ATTRS{serial}=="TODO_FOLLOWER", SYMLINK+="so101_follower", MODE="0666"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger

echo "udev rules written to $RULES_FILE"
echo "Replug the arms, then verify:"
echo "  ls -l /dev/so101_leader /dev/so101_follower"

# ---------------------------------------------------------------------------
# STEP 3 -- cameras: reference by-id paths instead of numeric indices
# ---------------------------------------------------------------------------
#
#   /dev/v4l/by-id/usb-<vendor>_<model>_<serial>-video-index0
#
# CAVEAT: if both cameras are the same model AND report no unique serial,
# by-id cannot disambiguate them. Fall back to physical port paths
# (/dev/v4l/by-path/...), PHYSICALLY LABEL THE PORTS, note it in
# docs/hardware.md, and never move them.

# ---------------------------------------------------------------------------
# NOT OPTIONAL -- do these regardless of whether you use udev
# ---------------------------------------------------------------------------
#
# 1. Serial port permissions:
#      sudo usermod -aG dialout "$USER"     # log out and back in after
#
# 2. Verify actual FPS after every recording session.
#    Insufficient USB bandwidth drops frames SILENTLY -- nothing errors out.
#    Check recorded frame count against duration x configured fps. If they
#    disagree, the data is suspect: re-record, and try moving one camera to a
#    different USB controller (not just a different port on the same hub).

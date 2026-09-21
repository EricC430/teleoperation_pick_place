"""S5 gap 3 -- scripted (kinematic) object attach/detach for replayed grasps.

Design `[AI提議 2026-09-18]`, not `[Eric決定]` -- proposed here so gap 3 can be built and shown,
not because it has been signed off. `S5_sim_replay_augmentation.md` §0/§2/§8 already rules a
full-physics contact grasp out for the first version; this is the "scripted attach/detach" it
names as the alternative. If Eric wants a different attach rule (a different signal, a different
proximity test, a real PhysX fixed joint instead of pose-forcing), that is a change to this file,
not a reason to have skipped writing one -- gap 3's own estimate was "decision ~1hr + scripted
implementation 0.5-1 day" precisely because a decision has to exist before it can be argued with.

**What "attach" means here, exactly:** while attached, every step, the object's world pose is
forced to a fixed offset from the gripper TCP. 🔴 **2026-09-21: the TCP moved.** It used to be the
midpoint of `link6`/`link7`'s body frames; those are the finger PIVOTS, 2.95 cm from link5, and the
fingers actually touch 8 cm out (`omx_constants.TCP_IN_LINK5_M`, measured). Use `tcp_pose_w()`
below -- it is the one definition, so the replay and the verifier cannot drift apart. The offset is
computed once, at the instant of attach, from wherever the object actually is then. This is NOT a
PhysX joint and it does not simulate contact, friction, or slip -- the object cannot be dropped or
mis-grasped once attached, by construction. That is the accepted cost of skipping gap 3's physics
route (`S5 §1` "它買不到的東西": the rendered contact behaviour is not real). `PhysxSchema` fixed
joints were the alternative; pose-forcing was picked because it needs no joint (dis)connection at
runtime, which is easier to get right without being able to test in this environment (see
`fit_drive_gains.py`'s docstring for the same constraint) -- if that trade turns out wrong,
swapping the pose-forcing calls below for joint creation is a local change, not a redesign.

**What decides attach/detach:** the REAL recorded `gripper.pos` channel (LeRobot RANGE_0_100 units, not degrees -- the
`*_deg` names below predate that correction; the trigger is percentile-based so units do not matter;
`joint_mapping.DATASET_JOINT_ORDER[5]`), not the sim gripper joint. The sim joint is a PD-lagged
copy of the same command (gap 1 is exactly the open question of how lagged) -- deciding the grasp
from it would fold gap 1's uncertainty into gap 3's trigger for no reason, when the ground-truth
signal is sitting right there in the same dataset row.

  1. per-episode auto threshold, computed once from that episode's own gripper trace (NOT a
     hand-tuned constant that could be right for one episode and wrong for the next):
         g_open   = 95th percentile of the trace
         g_closed = 5th percentile
         threshold = g_open - close_frac * (g_open - g_closed)      # close_frac default 0.6
  2. FREE -> ATTACHED on the first step where the reading is below `threshold` AND falling
     (this-step < last-step) AND the object is within `attach_radius_m` of the TCP
  3. ATTACHED -> FREE on the first step the reading rises back to/above `threshold`

[已查證 2026-09-18，跑在 `isaac-lab` 容器裡，not just read from a static trace] Ran
`verify_grasp_attach.py` against `ericc430/omx_pick_place_pilot_uvc_60` episode 0: attach at
frame 226, detach at frame 382, then attach AGAIN at frame 401, detach at frame 464 -- **two
cycles, not the one this paragraph originally predicted from eyeballing the raw trace.** The
second cycle is real signal, not a bug: the gripper trace genuinely dips again around frame
400-456 (a real second close, likely a regrasp/adjustment by the operator), and the state machine
above already handles repeat cycles correctly without any special-casing -- it was this docstring
that undersold it, not the code. `close_frac=0.6` computed threshold=53.74deg for this episode,
comfortably inside both dips. **Still not checked**: any other episode, or the object's actual
real-world position (this repo carries no independent grasp-success label) -- "state machine fires
at plausible frames, twice, on one real episode" is the whole of the evidence so far.

🔴 The trigger is a THRESHOLD ON A SIGNAL, not sensed contact. If the leader closed the gripper
without the object between the fingers (a miss, or a regrasp), this attaches whatever is within
`attach_radius_m` regardless. `S5 §5` item 3 already asks for `meta` to record whether an episode
went through grasp attachment at all -- treat that flag as "scripted, unverified", the same
posture as gap 2 (mimic sign) and gap 1 (gains) before each had its own closing evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


def tcp_pose_w(robot):
    """World pose of the gripper TCP: link5's frame, offset to the MEASURED fingertip point.

    Returns (pos, quat) as 1-D tensors, the shapes `ScriptedGraspAttach.step` expects. The
    orientation is link5's -- joint5 (wrist_roll) turns about link5's own +X, so rolling does not
    move the TCP, and there is no separate "gripper frame" to prefer over it.
    """
    from isaaclab.utils.math import combine_frame_transforms

    import omx_constants as K

    b5 = robot.body_names.index("link5")
    pos_w = robot.data.body_pos_w[0, b5]
    quat_w = robot.data.body_quat_w[0, b5]
    rel = torch.tensor(K.TCP_IN_LINK5_M, device=pos_w.device, dtype=pos_w.dtype)
    pos, _ = combine_frame_transforms(pos_w.unsqueeze(0), quat_w.unsqueeze(0), rel.unsqueeze(0))
    return pos.squeeze(0), quat_w


@dataclass
class GraspAttachConfig:
    # 0.09 m: the upright cup (height 9.5 cm, radius 3.75 cm) has its root at centre z=4.75 cm.
    # Grasping the rim puts the TCP at sqrt(4.75^2 + 3.75^2) = 6.05 cm from root even at perfect contact.
    # 0.05 m was too tight for the cup geometry (missed by ~2 cm). 0.09 m allows ~3 cm margin.
    attach_radius_m: float = 0.09
    close_frac: float = 0.6
    # Snap the cup directly under the TCP upon attach so it sits squarely between the fingers
    snap_to_tcp: bool = True
    snap_z_offset_m: float = -0.045  # cup centre 4.5 cm below fingertips (rim at fingertips)


@dataclass
class GraspEvent:
    frame: int
    kind: str  # "attach" | "detach"
    gripper_reading: float
    dist_m: float | None = None


class ScriptedGraspAttach:
    """One instance per episode replay.

    Usage, once per sim frame (see `verify_grasp_attach.py` for a working loop):

        want_pos, want_quat = grasp.step(frame_idx, gripper_reading_deg,
                                          tcp_pos_w, tcp_quat_w, object_pos_w, object_quat_w)
        if grasp.attached:
            object.write_root_pose_to_sim(torch.cat([want_pos, want_quat]).unsqueeze(0))
            object.write_root_velocity_to_sim(torch.zeros(1, 6, device=want_pos.device))
        # else: do nothing -- physics already owns the object this step
    """

    def __init__(self, cfg: GraspAttachConfig | None = None):
        self.cfg = cfg or GraspAttachConfig()
        self._threshold: float | None = None
        self._prev_reading: float | None = None
        self._offset_pos: torch.Tensor | None = None
        self._offset_quat: torch.Tensor | None = None
        self.attached: bool = False
        self.events: list[GraspEvent] = []

    def calibrate(self, gripper_trace_deg) -> float:
        """Call once with the WHOLE episode's gripper trace (degrees) before the first `step`."""
        import numpy as np

        arr = np.asarray(list(gripper_trace_deg), dtype=float)
        g_open = float(np.percentile(arr, 95))
        g_closed = float(np.percentile(arr, 5))
        self._threshold = g_open - self.cfg.close_frac * (g_open - g_closed)
        return self._threshold

    def step(
        self,
        frame_idx: int,
        gripper_reading_deg: float,
        tcp_pos_w: torch.Tensor,
        tcp_quat_w: torch.Tensor,
        object_pos_w: torch.Tensor,
        object_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the pose the object SHOULD have this step: `(pos_w, quat_w)`.

        Not attached -> returns `(object_pos_w, object_quat_w)` unchanged; caller should do
        nothing further. Attached -> returns the kinematic-follow pose; caller must write it.
        """
        if self._threshold is None:
            raise RuntimeError("call calibrate() with the episode's full gripper trace before step()")

        falling = self._prev_reading is not None and gripper_reading_deg < self._prev_reading
        below = gripper_reading_deg < self._threshold
        above = gripper_reading_deg >= self._threshold
        self._prev_reading = gripper_reading_deg

        if not self.attached and below and falling:
            dist = torch.norm(object_pos_w - tcp_pos_w).item()
            if dist <= self.cfg.attach_radius_m:
                self.attached = True
                if self.cfg.snap_to_tcp:
                    # tcp_pos_w is already at the fingertips (link5 + TCP_IN_LINK5_M).
                    # In TCP frame, +X is along the tool axis pointing into the cup.
                    # Placing the cup centre at +4.5 cm along +X locks it dead-centre between the fingers.
                    self._offset_pos = torch.tensor(
                        [0.045, 0.0, 0.0],
                        device=tcp_pos_w.device,
                        dtype=tcp_pos_w.dtype,
                    )
                    # Keep cup orientation locked to link5 or upright
                    _, self._offset_quat = _relative_pose(
                        tcp_pos_w, tcp_quat_w, object_pos_w, object_quat_w
                    )
                else:
                    self._offset_pos, self._offset_quat = _relative_pose(
                        tcp_pos_w, tcp_quat_w, object_pos_w, object_quat_w
                    )
                self.events.append(GraspEvent(frame_idx, "attach", gripper_reading_deg, dist))

        elif self.attached and above:
            self.attached = False
            self.events.append(GraspEvent(frame_idx, "detach", gripper_reading_deg))

        if self.attached:
            return _apply_relative_pose(tcp_pos_w, tcp_quat_w, self._offset_pos, self._offset_quat)
        return object_pos_w, object_quat_w


def _relative_pose(parent_pos, parent_quat, child_pos, child_quat):
    """child's pose expressed in parent's frame.

    [已查證 2026-09-18，對照 `isaac-sim/IsaacLab` GitHub `main` 分支原始碼，非本機執行]
    `subtract_frame_transforms(t01, q01, t02, q02) -> (t12, q12)`, quaternion order `(w,x,y,z)`,
    every tensor shape `(N, 3|4)` -- matches the call below. Not cross-checked against whatever
    exact Isaac Lab version the container image is pinned to (`docs/environment.md` only names
    the docker image, not a version number) -- if this container predates the alias/rename this
    was read from, it may still fail; the *shape and argument order* are what's confirmed, not
    "this exact installed build has this exact function"."""
    from isaaclab.utils.math import subtract_frame_transforms

    rel_pos, rel_quat = subtract_frame_transforms(
        parent_pos.unsqueeze(0), parent_quat.unsqueeze(0), child_pos.unsqueeze(0), child_quat.unsqueeze(0)
    )
    return rel_pos.squeeze(0), rel_quat.squeeze(0)


def _apply_relative_pose(parent_pos, parent_quat, rel_pos, rel_quat):
    """Inverse of `_relative_pose`: new world pose of the child given the parent's current pose
    and the stored relative offset (`combine_frame_transforms`, same source-checked caveat)."""
    from isaaclab.utils.math import combine_frame_transforms

    pos, quat = combine_frame_transforms(
        parent_pos.unsqueeze(0), parent_quat.unsqueeze(0), rel_pos.unsqueeze(0), rel_quat.unsqueeze(0)
    )
    return pos.squeeze(0), quat.squeeze(0)

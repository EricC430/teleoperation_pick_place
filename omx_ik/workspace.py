"""Where the arm is allowed to point, on top of what it can geometrically reach.

The IK in `omx_ik.ik` answers "can the linkage produce this pose". That is not the
same as "may it". Two further restrictions apply, and neither is in the URDF:

1. **Pan sector.** Measured 2026-08-31 (`docs/experiment_spec.md` 3): roughly 135
   degrees wide, `theta in [-90, +45]` with straight ahead = 0 and + to the left.
   The cause is **physical collision with the third-person camera's mount** — the arm
   body hits it when it swings that way. It is NOT the camera's field of view and it
   is NOT the cable: after the 2026-08-31 re-route the cable stops constraining the
   azimuth in the task area, which is D023's open question answered.

   Two caveats travel with that number and must not be dropped:
     - it is bound to where the camera is mounted *now*. Move the camera or move the
       arm onto a different baseplate, and the sector has to be re-measured.
     - the physical 0-degree reference line is still unmarked (`待標`), so the frame
       tie between this sector and the arm's own base frame is not yet pinned.

2. **Grasp targets belong in front of or beside the arm**, never behind it. This is
   a SEPARATE constraint from the sector above, and an earlier version of this file
   claimed — wrongly — that windowing joint1 enforces it for free. It does not: in a
   `back` branch solution joint1 points into the window while the target sits about
   180 degrees behind it, so such a solution passes a joint1 test while placing the
   object behind the arm. Use `filter_grasps(allow_fold_through_base=False)`, which is
   the default, to reject those.

Nothing here is a default. An unconstrained solve stays unconstrained unless a
window is passed, because inventing a sector would be worse than having none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PanWindow:
    """Allowed range for joint1 (shoulder_pan), in degrees of the arm's base frame."""

    min_deg: float
    max_deg: float
    note: str = ""

    def __post_init__(self) -> None:
        if self.min_deg >= self.max_deg:
            raise ValueError(f"need min_deg < max_deg, got {self.min_deg}, {self.max_deg}")
        if self.max_deg - self.min_deg > 360.0:
            raise ValueError("a pan window wider than a full turn is not a window")

    @property
    def width_deg(self) -> float:
        return self.max_deg - self.min_deg

    def contains(self, pan_rad: float) -> bool:
        """Joints wrap: a pan of 3.9 rad is the same pose as 3.9 - 2*pi."""
        deg = math.degrees(pan_rad)
        wrapped = math.remainder(deg - self.min_deg, 360.0)
        if wrapped < 0:
            wrapped += 360.0
        return wrapped <= self.width_deg

    @classmethod
    def from_sector(cls, sector, note: str = "") -> PanWindow:
        """Build from `placement_sampler.geometry.Sector`, so the filter and the
        placement mat cannot drift apart. The sector's theta must already be in the
        arm's base frame — see the caveat about the unmarked 0-degree line."""
        return cls(sector.theta_min, sector.theta_max, note=note)


# The 2026-08-31 tape measurement. Kept as a named constant rather than a default so
# that using it is always a deliberate choice, and so the caveats stay attached.
CAMERA_MOUNT_SECTOR_2026_08_31 = PanWindow(
    min_deg=-90.0,
    max_deg=45.0,
    note=(
        "docs/experiment_spec.md 3, measured 2026-08-31. Bound to the current "
        "third-person camera mount position; re-measure if the camera or the arm's "
        "baseplate moves. The physical 0-degree line is not yet marked."
    ),
)

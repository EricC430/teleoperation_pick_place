"""Isaac Lab scene for the OMX pick-and-place cell.

Geometry comes from `scene_constants.py` (most of it PLACEHOLDER — read that file's header).
Arm parameters come from `omx_constants.py`. Nothing numeric is defined here.

Scope: this is the *scene*, not the task. No rewards, no observations, no resets. The teleop
collector (S4 §4) and any later task env both build on it.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

import omx_constants as K
import scene_constants as S

OMX_USD = "/workspace/test_isaaclab/assets/omx_f_generated/omx_f.usd"
DEFAULT_OBJECT_USD = "/workspace/test_isaaclab/assets/trash_obj/trash_cans_1.usd"

# 🔴 The arm does NOT sit on the table -- it is on a ~15 cm riser ([Eric說 2026-09-21], see
# scene_constants.ARM_RISER_HEIGHT). Getting this wrong put every replayed pose 15 cm low.
ARM_BASE_POS = (0.0, 0.0, S.TABLE_TOP_Z + S.ARM_RISER_HEIGHT)


def omx_articulation_cfg(prim_path: str, usd_path: str = OMX_USD) -> ArticulationCfg:
    """The arm, with the actuator limits the URDF could not supply.

    The USD already carries the patched joint limits (see `convert_omx_urdf.py`); the actuator
    group here sets what Isaac Lab drives them with. Both read `omx_constants`, so they cannot
    drift apart silently.
    """
    joint_names = [j.urdf_name for j in K.JOINTS]
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,  # see the audit's item 7; turn on only if needed
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ARM_BASE_POS,
            joint_pos={name: 0.0 for name in joint_names + [K.MIMIC_JOINT[0]]},
        ),
        actuators={
            j.urdf_name: ImplicitActuatorCfg(
                joint_names_expr=[j.urdf_name],
                effort_limit=j.motor.stall_torque_nm,
                velocity_limit=j.motor.velocity_limit_rad_s,
                stiffness=K.stiffness(j),   # per RADIAN here — Isaac Lab's convention, unlike USD
                damping=K.damping(j),
            )
            for j in K.JOINTS
        },
    )


def _pinhole(hfov_deg: float, clip: tuple[float, float]) -> sim_utils.PinholeCameraCfg:
    return sim_utils.PinholeCameraCfg(
        focal_length=S.focal_length_mm(hfov_deg),
        horizontal_aperture=S.SENSOR_APERTURE_MM,
        clipping_range=clip,
    )


@configclass
class OmxCellSceneCfg(InteractiveSceneCfg):
    """Table, arm, one object, a bin, two cameras.

    🔴 Camera ORDER is a frozen scene constant (D022): the dataset writes
       1. observation.images.wrist   2. observation.images.front-left
       The declaration order below matches, and must keep matching.
    """

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=S.DOME_LIGHT_INTENSITY, color=(0.9, 0.9, 0.95)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=S.TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.80, 0.76)),
        ),
        # the slab's centre sits half a thickness below the top surface
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(S.TABLE_SIZE[0] / 2.0 - 0.15, 0.0, S.TABLE_TOP_Z - S.TABLE_SIZE[2] / 2.0)
        ),
    )

    # the riser the arm is actually mounted on ([Eric說 2026-09-21]). Height measured, footprint
    # PLACEHOLDER. Sits directly under the pan axis.
    arm_riser = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ArmRiser",
        spawn=sim_utils.CuboidCfg(
            size=S.ARM_RISER_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.55, 0.70)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, S.TABLE_TOP_Z + S.ARM_RISER_HEIGHT / 2.0)
        ),
    )

    robot: ArticulationCfg = omx_articulation_cfg("{ENV_REGEX_NS}/Robot")

    # 🔴 PLACEHOLDER bin. `assets/Trashcan/` holds no USD, so this is a box, not the real bin.
    bin = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bin",
        spawn=sim_utils.CuboidCfg(
            size=S.BIN_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.65)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(S.BIN_POS[0], S.BIN_POS[1], S.TABLE_TOP_Z + S.BIN_SIZE[2] / 2.0)
        ),
    )

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DEFAULT_OBJECT_USD,
            # 🔴 trash_obj/*.usd is authored at metersPerUnit=0.01; without this the object is
            # 100x too large and explodes on first contact. See scene_constants.TRASH_OBJ_SCALE.
            scale=S.TRASH_OBJ_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=3.0,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.0, S.TABLE_TOP_Z + 0.05)),
    )

    # ---- cameras, in dataset order -----------------------------------------------------
    cam_wrist: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + S.CAM_WRIST_PARENT_LINK + "/WristCam",
        update_period=1.0 / S.CAM_FPS,
        width=S.CAM_WIDTH_WRIST,
        height=S.CAM_HEIGHT_WRIST,
        data_types=["rgb"],
        spawn=_pinhole(S.HFOV_WRIST_DEG, S.CLIP_WRIST),
        offset=CameraCfg.OffsetCfg(
            pos=S.CAM_WRIST_OFFSET_POS, rot=S.CAM_WRIST_OFFSET_ROT, convention="ros"
        ),
    )

    cam_front_left: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FrontLeftCam",
        update_period=1.0 / S.CAM_FPS,
        width=S.CAM_WIDTH_FRONT_LEFT,
        height=S.CAM_HEIGHT_FRONT_LEFT,
        data_types=["rgb"],
        spawn=_pinhole(S.HFOV_FRONT_LEFT_DEG, S.CLIP_FRONT_LEFT),
        # pose is set by look-at after the scene is built (see preview_scene.py)
        offset=CameraCfg.OffsetCfg(pos=S.CAM_FRONT_LEFT_POS, convention="ros"),
    )

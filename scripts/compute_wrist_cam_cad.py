#!/usr/bin/env python3
"""Compute exact camera and gripper transforms relative to link5."""
import numpy as np
from scipy.spatial.transform import Rotation as R

# From STEP assembly parsing:
# PR33_C02_ASM coordinates (mm):
# F5 (pan / link5 base):
# T_ASM_F5:
# origin = [0, 210.650, 192.250] mm
# X_ASM = [-1, 0, 0]
# Y_ASM = [0, 1, 0]
# Z_ASM = [0, 0, -1]

T_ASM_F5 = np.eye(4)
T_ASM_F5[:3, 0] = [-1, 0, 0]
T_ASM_F5[:3, 1] = [0, 1, 0]
T_ASM_F5[:3, 2] = [0, 0, -1]
T_ASM_F5[:3, 3] = [0, 210.650, 192.250]

# F6 (gripper base / link5 horn):
T_ASM_F6 = np.eye(4)
T_ASM_F6[:3, 0] = [1, 0, 0]
T_ASM_F6[:3, 1] = [0, 1, 0]
T_ASM_F6[:3, 2] = [0, 0, 1]
T_ASM_F6[:3, 3] = [0, 210.650, 236.950]

# F7 (finger 1 hinge / joint_1):
# Pos: [7.500, 226.650, 250.450] mm
# F8 (finger 2 hinge / joint_2):
# Pos: [-10.800, 226.650, 250.450] mm

# Camera:
T_ASM_Cam = np.eye(4)
T_ASM_Cam[:3, 0] = [1.0, 0.0, 0.0]
T_ASM_Cam[:3, 1] = [0.0, 0.78801075, 0.61566148]
T_ASM_Cam[:3, 2] = [0.0, -0.61566148, 0.78801075]
T_ASM_Cam[:3, 3] = [-0.0, 260.204, 252.164]

# Let's inspect the relationship between F5 and F6:
# F6 - F5 in ASM: [0, 0, 44.7] mm along Z_ASM = -44.7 mm along Z_F5!
print("--- F6 relative to F5 in ASM ---")
T_F5_F6 = np.linalg.inv(T_ASM_F5) @ T_ASM_F6
print("T_F5_F6 translation (mm):", T_F5_F6[:3, 3])
print("T_F5_F6 rotation:\n", T_F5_F6[:3, :3])

# Camera relative to F6:
T_F6_Cam = np.linalg.inv(T_ASM_F6) @ T_ASM_Cam
print("\n--- Camera relative to F6 (mm) ---")
print("Pos (mm):", T_F6_Cam[:3, 3])
print("Pos (m):", T_F6_Cam[:3, 3] / 1000.0)
print("Rot matrix:\n", T_F6_Cam[:3, :3])

# In URDF:
# link5 is child of joint5.
# Let's see how link5 frame in URDF maps to F5 or F6:
# In URDF:
# joint5 xyz="0.0287 0 0" rpy="0 0 0" axis="1 0 0"
# gripper_joint_1 xyz="0.0295 0.0075 0" axis="0 0 1"
# gripper_joint_2 xyz="0.0295 -0.0108 0" axis="0 0 1"
#
# Look at F7 and F8 positions relative to F6 in ASM:
# F7 = [7.5, 226.65, 250.45]
# F8 = [-10.8, 226.65, 250.45]
# F6 = [0, 210.65, 236.95]
# Relative to F6:
# F7 - F6 = [7.5, 16.0, 13.5] mm
# F8 - F6 = [-10.8, 16.0, 13.5] mm
# Notice:
# X component is +7.5 mm for F7, -10.8 mm for F8!
# Exactly matching URDF's Y coordinates: 0.0075 and -0.0108!
# So URDF_Y corresponds to F6_X (or -F6_X)!

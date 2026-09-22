#!/usr/bin/env python3
"""Parse OMX-AI(Follower).stp assembly hierarchy and placements."""
import re
import numpy as np

STEP_PATH = "assets/OMX-AI(Follower).stp"

print("Reading STEP file...")
with open(STEP_PATH, "r", errors="ignore") as f:
    text = f.read()
print(f"Total chars: {len(text)}")

points = {}
directions = {}
axis2 = {}
transformations = {}
repr_rels = {}
cdsr = {}
prod_def_shapes = {}
usages = {}

for m in re.finditer(r"#(\d+)\s*=\s*CARTESIAN_POINT\s*\('[^']*',\s*\(([^)]+)\)\)", text):
    coords = [float(x.strip()) for x in m.group(2).split(",")]
    points[int(m.group(1))] = np.array(coords)

for m in re.finditer(r"#(\d+)\s*=\s*DIRECTION\s*\('[^']*',\s*\(([^)]+)\)\)", text):
    coords = [float(x.strip()) for x in m.group(2).split(",")]
    directions[int(m.group(1))] = np.array(coords)

for m in re.finditer(r"#(\d+)\s*=\s*AXIS2_PLACEMENT_3D\s*\('[^']*',\s*#(\d+),\s*#(\d+),\s*#(\d+)\)", text):
    axis2[int(m.group(1))] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))

for m in re.finditer(r"#(\d+)\s*=\s*ITEM_DEFINED_TRANSFORMATION\s*\('[^']*',\s*'[^']*',\s*#(\d+),\s*#(\d+)\)", text):
    transformations[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))

for m in re.finditer(r"#(\d+)\s*=\s*\(REPRESENTATION_RELATIONSHIP\s*\('[^']*',\s*'[^']*',\s*#(\d+),\s*#(\d+)\)\s*REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION\s*\(#(\d+)\)", text):
    repr_rels[int(m.group(1))] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))

for m in re.finditer(r"#(\d+)\s*=\s*CONTEXT_DEPENDENT_SHAPE_REPRESENTATION\s*\(#(\d+),\s*#(\d+)\)", text):
    cdsr[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))

for m in re.finditer(r"#(\d+)\s*=\s*PRODUCT_DEFINITION_SHAPE\s*\('[^']*',\s*'([^']*)',\s*#(\d+)\)", text):
    prod_def_shapes[int(m.group(1))] = (m.group(2), int(m.group(3)))

for m in re.finditer(r"#(\d+)\s*=\s*NEXT_ASSEMBLY_USAGE_OCCURRENCE\s*\('[^']*',\s*'[^']*',\s*'([^']*)',\s*#(\d+),\s*#(\d+),", text):
    usages[int(m.group(1))] = (m.group(2), int(m.group(3)), int(m.group(4)))

print(f"Parsed: {len(points)} pts, {len(axis2)} axes, {len(cdsr)} CDSRs, {len(prod_def_shapes)} PDSs, {len(usages)} usages")

def get_placement_matrix(axis2_id):
    pt_id, z_dir_id, x_dir_id = axis2[axis2_id]
    origin = points[pt_id]
    z_axis = directions[z_dir_id]
    x_axis = directions[x_dir_id]
    y_axis = np.cross(z_axis, x_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = origin
    return T

print("\n--- Key Placements in Assembly ---")
results = []
for cdsr_id, (repr_rel_id, pds_id) in cdsr.items():
    if pds_id not in prod_def_shapes:
        continue
    pds_desc, usage_id = prod_def_shapes[pds_id]
    if usage_id not in usages:
        continue
    part_name, parent_id, child_id = usages[usage_id]
    
    if repr_rel_id in repr_rels:
        rep1, rep2, trans_id = repr_rels[repr_rel_id]
        from_axis, to_axis = transformations[trans_id]
        T_from = get_placement_matrix(from_axis)
        T_to = get_placement_matrix(to_axis)
        T_rel = T_to @ np.linalg.inv(T_from)
        pos = T_rel[:3, 3]
        rot = T_rel[:3, :3]
        results.append((part_name, pds_desc, pos, rot, T_rel))

# Sort by name
for part_name, desc, pos, rot, T in sorted(results, key=lambda x: x[0]):
    if any(k in part_name for k in ["PR33", "CAMERA", "follower", "gripper", "pan", "BASE", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]):
        print(f"\nPart: {part_name:25s} | Desc: {desc}")
        print(f"   Pos (mm): [{pos[0]:8.3f}, {pos[1]:8.3f}, {pos[2]:8.3f}]")
        print(f"   X-axis:   [{rot[0,0]:7.4f}, {rot[1,0]:7.4f}, {rot[2,0]:7.4f}]")
        print(f"   Y-axis:   [{rot[0,1]:7.4f}, {rot[1,1]:7.4f}, {rot[2,1]:7.4f}]")
        print(f"   Z-axis:   [{rot[0,2]:7.4f}, {rot[1,2]:7.4f}, {rot[2,2]:7.4f}]")

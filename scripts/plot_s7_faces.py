#!/usr/bin/env python
"""Draw the S7 measurement faces with the phone's long edge as an arrow (docs/assets/S7_measurement_faces_arrows.png).

The faces and their arrow directions are read from `measure_link_tilt.FACES` (the same `d` vectors the solver uses),
the meshes are the URDF STLs, and the arm is drawn at the URDF zero pose (upper arm vertical, forearm and gripper
horizontal) only so every face is visible -- it is not a measurement pose.

    uv run python scripts/plot_s7_faces.py [--mesh-dir assets/open_manipulator_description/meshes/omx_f]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
import measure_link_tilt as S7  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "Noto Sans TC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# link index (measure_link_tilt.Face.link) -> STL, and the chain (origin, axis) into that link
STL = {0: "follower_01_base.stl", 1: "follower_02_base_tilt_Revised.stl", 2: "follower_03_middle_verticle.stl",
       3: "follower_04_middle_horizontal.stl", 4: "follower_05_tip.stl", 5: "follower_06_pan_Revised.stl"}
FINGERS = (("follower_07_gripper_motorized.stl", (0.0295, 0.0075, 0.0)), ("follower_08_gripper_gear.stl", (0.0295, -0.0108, 0.0)))
CHAIN = [((-0.01125, 0, 0.034), "z"), ((0, 0, 0.0635), "y"), ((0.0415, 0, 0.11315), "y"), ((0.162, 0, 0), "y"), ((0.0287, 0, 0), "x")]
# which mesh face (normal in the link frame, kept side) the phone sits on -- from the Face.mesh notes
FACE_PLANE = {"base_x": (0, (0, 0, 1)), "base_y": (0, (0, 0, 1)), "upper": (2, (1, 0, 0)), "forearm": (3, (0, 0, 1)),
              "gripper": (5, (0, 0, 1)), "gripper_across": (5, (0, 0, 1))}
LABEL_SHIFT = {"base_y": np.array([0.0, 0.03, 0.03]), "base_x": np.array([0.0, -0.05, -0.02]),
               "gripper_across": np.array([0.0, 0.03, 0.03]), "gripper": np.array([0.02, -0.04, -0.03])}
COLORS = {"base_x": "#8e44ad", "base_y": "#8e44ad", "upper": "#d62728", "forearm": "#1f77b4", "gripper": "#2ca02c",
          "gripper_across": "#2ca02c"}


def T(t):
    M = np.eye(4)
    M[:3, 3] = t
    return M


def link_transforms(q):
    out = {0: np.eye(4)}
    M = np.eye(4)
    for i, ((o, ax), qi) in enumerate(zip(CHAIN, q), start=1):
        c, s = np.cos(qi), np.sin(qi)
        R = np.eye(4)
        if ax == "x":
            R[1:3, 1:3] = [[c, -s], [s, c]]
        elif ax == "y":
            R[0, 0], R[0, 2], R[2, 0], R[2, 2] = c, s, -s, c
        else:
            R[:2, :2] = [[c, -s], [s, c]]
        M = M @ T(o) @ R
        out[i] = M.copy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-dir", default=str(_REPO / "assets/open_manipulator_description/meshes/omx_f"))
    ap.add_argument("--out", default=str(_REPO / "docs/assets/S7_measurement_faces_arrows.png"))
    args = ap.parse_args()
    q = [0.0, 0.0, 0.0, 0.0, 0.0]
    Ts = link_transforms(q)
    meshes = {k: trimesh.load(os.path.join(args.mesh_dir, f)) for k, f in STL.items()}
    fingers = [(trimesh.load(os.path.join(args.mesh_dir, f)), o) for f, o in FINGERS]

    # face centre on its plane (largest-area triangles whose normal matches), in the link frame
    centre = {}
    for name, (link, n) in FACE_PLANE.items():
        m = meshes[link]
        ok = m.face_normals @ np.array(n, float) > 0.98
        if name in ("base_x", "base_y"):        # base top plate: the highest +z faces only
            z = m.triangles_center[:, 2]
            ok &= z > np.percentile(z[ok], 70)
        a = m.area_faces[ok]
        centre[name] = (m.triangles_center[ok] * a[:, None]).sum(0) / a.sum() * 1e-3
    def world(link, p):
        return (Ts[link] @ np.r_[p, 1.0])[:3]

    fig = plt.figure(figsize=(19, 8.5))
    views = (("從操作者的位置看（手臂後方偏左）", 28, -150), ("從手臂右側看", 0, -90), ("俯視（上方往下看，下方是操作者）", 90, -90))
    for k, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.set_proj_type("ortho" if k else "persp")
        for link, m in meshes.items():
            V = (Ts[link][:3, :3] @ (m.vertices * 1e-3).T).T + Ts[link][:3, 3]
            F = m.faces[:: max(1, len(m.faces) // 2500)]
            ax.add_collection3d(Poly3DCollection(V[F], facecolor="#bbbbbb", edgecolor="none", alpha=0.25))
        for m, o in fingers:
            M5 = Ts[5] @ T(o)
            V = (M5[:3, :3] @ (m.vertices * 1e-3).T).T + M5[:3, 3]
            ax.add_collection3d(Poly3DCollection(V[m.faces[:: max(1, len(m.faces) // 1500)]], facecolor="#bbbbbb",
                                                 edgecolor="none", alpha=0.25))
        # the measured faces, solid
        for name, (link, n) in FACE_PLANE.items():
            if name in ("base_y", "gripper_across"):
                continue
            m = meshes[link]
            ok = m.face_normals @ np.array(n, float) > 0.98
            if name == "base_x":
                z = m.triangles_center[:, 2]
                ok &= z > np.percentile(z[ok], 70)
            V = (Ts[link][:3, :3] @ (m.vertices * 1e-3).T).T + Ts[link][:3, 3]
            ax.add_collection3d(Poly3DCollection(V[m.faces[ok]], facecolor=COLORS[name], edgecolor="none", alpha=0.85))
        # arrows = the phone's long edge, pointing at the arrow end
        L = 0.06
        for name, face in S7.FACES.items():
            p = world(face.link, centre[name])
            d = Ts[face.link][:3, :3] @ np.array(face.d, float)
            off = 0.012 * (Ts[FACE_PLANE[name][0]][:3, :3] @ np.array(FACE_PLANE[name][1], float))
            if name in ("base_y", "gripper_across"):
                p = p + 0.015 * (Ts[face.link][:3, :3] @ np.array([1.0, 0, 0]))
            s0 = p + off - d * L / 2
            ax.quiver(*s0, *(d * L), color=COLORS[name], linewidth=3.5, arrow_length_ratio=0.35)
            lab = s0 + d * L * 1.15 + off + LABEL_SHIFT.get(name, np.zeros(3))
            ax.text(*lab, name, color=COLORS[name], fontsize=11, weight="bold")
        ax.set_xlim(-0.12, 0.30)
        ax.set_ylim(-0.21, 0.21)
        ax.set_zlim(0.0, 0.30)
        ax.set_box_aspect((0.42, 0.42, 0.30))
        ax.view_init(elev=elev, azim=azim)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel("x → 手臂正前方")
        ax.set_ylabel("y → 操作者左手邊")
        if k == 2:
            ax.text(-0.11, 0.0, 0.0, "操作者 ↓", fontsize=11)
        ax.set_title(title, fontsize=13)
    legend = "\n".join(f"{n}：{f.where}　箭頭端＝{f.arrow}" for n, f in S7.FACES.items())
    fig.suptitle("S7 量測面與手機長邊方向（箭頭＝手機長邊、指向「箭頭端」；讀數＝箭頭端較高為正）\n"
                 "姿勢是 URDF 零點（上臂垂直、前臂與夾爪水平），只為了看清楚，不是量測姿勢", fontsize=13)
    fig.text(0.02, 0.01, legend, fontsize=10, va="bottom")
    fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.18, wspace=0.0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

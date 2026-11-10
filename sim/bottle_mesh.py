"""Solid bottle of revolution (PET drink bottle) for the DR collection: profile, mesh, mass properties, inside test.
numpy only (no pxr); sim/dr_objects.py writes the USD, sim/penetration.py uses the inside test.

Why a solid and not a 0.3 mm shell (2026-11-11): the gripper takes a bottle from OUTSIDE (body, shoulder or cap), never
with a finger inside it like the paper cup's rim, so the cavity plays no part in the contact; and a ~0.3 mm wall is
thinner than one SDF cell at resolution 256 over a ~22 cm bottle (docs/sim_physics_params.md section 2 #3). The
collider is the solid; the MASS properties are those of the hollow bottle (PET shell + cap + optional liquid), set
explicitly, because a solid of uniform density has the wrong centre of mass and too little inertia for tipping and
rolling (the real failures 推倒滾走 / 夾太高滑掉, docs/worklog/2026-10-22.md).
Still rigid: a real bottle dents when squeezed (docs/sim_physics_params.md #15), so grasp margin is optimistic.

Parameters (metres / kg, in an objects.geometry entry with "shape": "bottle"):
  height        total height including the cap
  body_dia      straight body
  base_dia      at z = 0 (the foot is rounded in to it over base_h)        base_h
  waist_dia     optional narrower band between waist_z0 and waist_z1      waist_z0, waist_z1
  body_points   optional [[z, dia], ...] from z = 0 to shoulder_z0, cosine-blended; replaces base_* and waist_*
  shoulder_z0   top of the body = start of the shoulder; shoulder_z1 = start of the neck (cosine blend in between)
  neck_dia      neck below the cap
  cap_dia, cap_h
  shell_mass    PET body (spread over the outer surface below the cap)    cap_mass   fill_ml (0 = empty)
Origin = centre of the underside, +Z up (same as the cup, so physics_replay places it on the placement point unchanged).
"""
import math

import numpy as np

INSIDE_V = (0.97, 1.0)   # same uv convention as sim/shell_mesh.py: outer surface v 0..0.96, everything else this strip
WATER = 1000.0           # kg/m^3


def profile(p, n_foot=4, n_shoulder=12, n_body=10):
    """(z, r) outline from the foot to the top of the cap, z increasing (equal z allowed for steps)."""
    H, rb, rf = p["height"], p["body_dia"] / 2, p.get("base_dia", p["body_dia"]) / 2
    hb = p.get("base_h", 0.0)
    z0, z1 = p["shoulder_z0"], p["shoulder_z1"]
    rn, rc, hc = p["neck_dia"] / 2, p["cap_dia"] / 2, p["cap_h"]
    if not (0 <= hb < z0 < z1 < H - hc) and not (p.get("body_points") and 0 < z0 < z1 < H - hc):
        raise ValueError(f"bottle {p.get('id')}: need 0 <= base_h < shoulder_z0 < shoulder_z1 < height - cap_h")
    if p.get("body_points"):
        # [[z, dia], ...] from z = 0 up to shoulder_z0, cosine-blended between points (2026-11-11: the real bottle
        # narrows from its foot to a waist at 7.3 cm and widens again; base_dia / base_h / waist_* are then ignored)
        bp = sorted((float(z), float(d) / 2) for z, d in p["body_points"])
        if bp[0][0] != 0 or bp[-1][0] != z0:
            raise ValueError(f"bottle {p.get('id')}: body_points must start at z = 0 and end at shoulder_z0")
        pts = [bp[0]]
        for (za, ra), (zb, rb_) in zip(bp, bp[1:]):
            for i in range(1, n_body + 1):
                t = i / n_body
                pts.append((za + (zb - za) * t, ra + (rb_ - ra) * 0.5 * (1 - math.cos(math.pi * t))))
        rb = bp[-1][1]
    else:
        pts = [(0.0, rf)]
        for i in range(1, n_foot + 1):                                  # rounded foot: quarter-sine from base_dia to body_dia
            if hb <= 0:
                break
            t = i / n_foot
            pts.append((hb * t, rf + (rb - rf) * math.sin(t * math.pi / 2)))
        wz = (p.get("waist_z0"), p.get("waist_z1")); rw = p.get("waist_dia", p["body_dia"]) / 2
        for i in range(n_body + 1):                                     # body, with an optional waist (cosine dip)
            z = hb + (z0 - hb) * i / n_body
            r = rb
            if wz[0] is not None and wz[0] < z < wz[1]:
                r = rb - (rb - rw) * 0.5 * (1 - math.cos(2 * math.pi * (z - wz[0]) / (wz[1] - wz[0])))
            if z > pts[-1][0] + 1e-9:
                pts.append((z, r))
    for i in range(1, n_shoulder + 1):                                  # shoulder: cosine from body to neck radius
        t = i / n_shoulder
        pts.append((z0 + (z1 - z0) * t, rn + (rb - rn) * 0.5 * (1 + math.cos(math.pi * t))))
    pts.append((H - hc, rn))                                            # neck
    pts.append((H - hc, rc)); pts.append((H, rc))                       # cap (a step out, then straight up)
    return np.array(pts, float)


def bottle_mesh(p, n_sides=48):
    """Points, face counts, face indices, face-varying uv of the closed solid. Same return layout as shell_mesh.cup_mesh."""
    prof = profile(p); H = p["height"]; n = len(prof)
    ang = 2 * math.pi * np.arange(n_sides) / n_sides
    pts = [[r * math.cos(a), r * math.sin(a), z] for z, r in prof for a in ang]
    c_bot, c_top = len(pts), len(pts) + 1
    pts += [[0.0, 0.0, 0.0], [0.0, 0.0, H]]
    fc, fi, uv = [], [], []
    vin = (0.5, sum(INSIDE_V) / 2)

    def poly(idx, uvs):
        fc.append(len(idx)); fi.extend(idx); uv.extend(uvs)

    vo = lambda z: 0.96 * z / H   # noqa: E731
    for i in range(n - 1):                                              # side (normals out)
        for j in range(n_sides):
            j1 = (j + 1) % n_sides; u0, u1 = j / n_sides, (j + 1) / n_sides
            a, b, c, d = i * n_sides + j, i * n_sides + j1, (i + 1) * n_sides + j1, (i + 1) * n_sides + j
            za, zc = prof[i, 0], prof[i + 1, 0]
            poly([a, b, c], [(u0, vo(za)), (u1, vo(za)), (u1, vo(zc))]); poly([a, c, d], [(u0, vo(za)), (u1, vo(zc)), (u0, vo(zc))])
    top = (n - 1) * n_sides
    for j in range(n_sides):
        poly([c_bot, (j + 1) % n_sides, j], [vin] * 3)                  # underside (normal down)
        poly([c_top, top + j, top + (j + 1) % n_sides], [vin] * 3)      # cap top (normal up)
    return np.array(pts, np.float32), np.array(fc, np.int32), np.array(fi, np.int32), np.array(uv, np.float32)


def radius_at(p, z):
    """Outer radius of the solid at height z (0 outside 0..height); z may be an array."""
    prof = profile(p); z = np.asarray(z, float)
    # np.interp needs increasing xp: the cap step has two equal z; nudge the second one up by 1 nm
    zs = prof[:, 0].copy()
    for i in range(1, len(zs)):
        if zs[i] <= zs[i - 1]:
            zs[i] = zs[i - 1] + 1e-9
    r = np.interp(z, zs, prof[:, 1])
    return np.where((z >= 0) & (z <= p["height"]), r, 0.0)


def solid_volume(p, dz=1e-4):
    z = np.arange(0, p["height"], dz) + dz / 2
    return float((math.pi * radius_at(p, z) ** 2).sum() * dz)


def mass_props(p, dz=1e-4):
    """Hollow bottle: PET shell over the outer surface below the cap (incl. the underside disc), a solid-cylinder cap,
    and optional water up to the level that holds fill_ml (wall thickness ignored). Returns total mass (kg), centre of
    mass (x, y, z) in the bottle frame, and the diagonal inertia (Ixx, Iyy, Izz) about the centre of mass (kg m^2)."""
    H, hc, rc = p["height"], p["cap_h"], p["cap_dia"] / 2
    z = np.arange(0, H - hc, dz) + dz / 2; r = radius_at(p, z)
    dr = np.gradient(r, dz)
    dA = 2 * math.pi * r * np.sqrt(1 + dr ** 2) * dz                    # lateral ring areas
    r0 = radius_at(p, 0.0)
    A_bot = math.pi * float(r0) ** 2
    area = dA.sum() + A_bot
    sig = p["shell_mass"] / area                                       # kg / m^2
    parts = []                                                         # (m, zc, Izz, Ixx about own centre in x)
    m = sig * dA                                                       # thin rings: Izz = m r^2, Ixx = m r^2 / 2
    parts += [(m, z, m * r ** 2, m * r ** 2 / 2)]
    mb = sig * A_bot                                                   # underside disc: Izz = m r^2 / 2, Ixx = m r^2 / 4
    parts += [(np.array([mb]), np.array([0.0]), np.array([mb * r0 ** 2 / 2]), np.array([mb * r0 ** 2 / 4]))]
    mc = p.get("cap_mass", 0.0)                                        # cap: solid cylinder
    parts += [(np.array([mc]), np.array([H - hc / 2]), np.array([mc * rc ** 2 / 2]), np.array([mc * (rc ** 2 / 4 + hc ** 2 / 12)]))]
    fill = p.get("fill_ml", 0.0) * 1e-6                                # m^3
    level = 0.0
    if fill > 0:
        vol = np.cumsum(math.pi * r ** 2 * dz)
        if fill > vol[-1]:
            raise ValueError(f"bottle {p.get('id')}: fill_ml {p['fill_ml']} exceeds the solid volume below the cap")
        k = int(np.searchsorted(vol, fill)); level = float(z[k] + dz / 2)
        zw, rw = z[:k + 1], r[:k + 1]; mw = WATER * math.pi * rw ** 2 * dz   # discs: Izz = m r^2/2, Ixx = m r^2/4
        parts += [(mw, zw, mw * rw ** 2 / 2, mw * rw ** 2 / 4)]
    M = sum(float(pm.sum()) for pm, *_ in parts)
    zc = sum(float((pm * pz).sum()) for pm, pz, *_ in parts) / M
    Izz = sum(float(iz.sum()) for _, _, iz, _ in parts)
    Ixx = sum(float((ix + pm * (pz - zc) ** 2).sum()) for pm, pz, _, ix in parts)   # parallel axis to the COM
    return {"mass": M, "com": [0.0, 0.0, zc], "inertia": [Ixx, Ixx, Izz], "fill_level_m": level, "shell_area_m2": float(area)}


def depth_inside(c, p):
    """Radial depth (m) of points c (N x 3, bottle frame) inside the solid; negative = outside. Points above the cap
    top or below the underside are outside. Used by sim/penetration.py instead of the cup's wall layers."""
    z = c[:, 2]; r = np.hypot(c[:, 0], c[:, 1])
    d = radius_at(p, z) - r
    d = np.minimum(d, np.minimum(z, p["height"] - z))                 # the underside and the cap top bound it too
    return np.where((z >= 0) & (z <= p["height"]), d, -1.0)


def mesh_volume(pts, fi_tri):
    """Signed volume of a closed triangle mesh (divergence theorem); positive when normals point out."""
    t = np.asarray(pts, float)[np.asarray(fi_tri).reshape(-1, 3)]
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6)


if __name__ == "__main__":   # self-test on the placeholder of sim/dr_objects_bottle.json
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from shell_mesh import check_closed
    cfg = json.loads((Path(__file__).resolve().parent / "dr_objects_bottle.json").read_text(encoding="utf-8"))
    for p in cfg["geometry"]:
        pts, fc, fi, uv = bottle_mesh(p)
        assert (fc == 3).all() and len(uv) == len(fi)
        v_mesh, v_prof = mesh_volume(pts, fi), solid_volume(p)
        mp = mass_props(p)
        print(f"{p['id']}: points {len(pts)} tris {len(fc)} open edges {check_closed(fi)}; "
              f"solid volume mesh {v_mesh * 1e6:.1f} ml / profile {v_prof * 1e6:.1f} ml; "
              f"mass {mp['mass'] * 1000:.1f} g, COM z {mp['com'][2] * 100:.2f} cm, I (g cm^2) "
              f"{mp['inertia'][0] * 1e7:.0f} / {mp['inertia'][2] * 1e7:.0f}, fill level {mp['fill_level_m'] * 100:.1f} cm")
        assert check_closed(fi) == 0, "mesh not watertight"
        assert v_mesh > 0 and abs(v_mesh - v_prof) / v_prof < 0.02, "mesh and profile volumes disagree"
        d = depth_inside(np.array([[0, 0, p["height"] / 2], [p["body_dia"], 0, 0.05], [0, 0, p["height"] + 0.01]]), p)
        assert d[0] > 0 and d[1] < 0 and d[2] < 0

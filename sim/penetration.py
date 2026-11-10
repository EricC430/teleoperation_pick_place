"""How much do the gripper links overlap the cup in a physics replay? (2026-11-01, EricC: fingers pass through the cup)

Measure, not eyeball. Each gripper link's collision mesh (link frame) is voxelized, filled, and ERODED by one voxel, so
every kept sample is at least half a pitch inside the link (touching the cup does not count as overlap). Every frame
the samples go into the cup's frame (logged link and cup poses) and are tested analytically against the cup's SOLID
wall (truncated-cone shell between the inner and outer surface, plus the bottom). Per link and frame:
  vol_mm3   link volume inside the wall (samples x pitch^3)
  through   the link has samples in the outer, the middle AND the inner third of the wall = it has gone through it;
            or its side of the wall (inside the cup / beside it below the rim) flipped between two logged frames
A real contact gives 0 (up to half a pitch of overlap is not seen).

    python penetration.py links=<links.npz> cup=<objects.json>:<id> logs=<physics_log.json>,... [pitch_mm=0.2] [out=<json>]

links.npz: <link>__<i>__v / __f (link-frame triangles), dumped from the robot asset (dr/pen/extract_links.py). Needs
trimesh and scipy (the Isaac python has them: /isaac-sim/python.sh). link5, link6, link7 unless links_used= is given.
"""
import json
import math
import sys

import numpy as np

A = dict(a.split("=", 1) for a in sys.argv[1:])


def R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def link_samples(npz, links, pitch):
    """Cached next to the npz (<npz stem>_samples_<pitch um>um.npz): voxelizing link5 takes most of a minute."""
    import os
    cache = f"{os.path.splitext(npz)[0]}_samples_{int(round(pitch * 1e6))}um.npz"
    if os.path.exists(cache):
        c = np.load(cache)
        if all(l in c.files for l in links):
            print("link samples from", cache, flush=True)
            return {l: c[l] for l in links}
    out = _link_samples(npz, links, pitch)
    try:
        np.savez(cache, **out)
    except OSError:
        pass
    return out


def _link_samples(npz, links, pitch):
    import trimesh
    from scipy.ndimage import binary_erosion
    d = np.load(npz); out = {}
    for k in sorted({f.rsplit("__", 1)[0] for f in d.files}):
        link = k.split("__")[0]
        if link not in links:
            continue
        m = trimesh.Trimesh(d[k + "__v"], d[k + "__f"], process=True); m.merge_vertices()
        v = m.voxelized(pitch).fill(); keep = binary_erosion(v.matrix)
        pts = v.indices_to_points(np.argwhere(keep))
        out.setdefault(link, []).append(pts)
        print(f"{link}: watertight {m.is_watertight}, {len(pts)} interior samples (eroded)", flush=True)
    return {l: np.concatenate(v) for l, v in out.items()}


def wall_layer(p, top_r, bot_r, h, wall, bottom):
    """-1 outside the solid; else 0 / 1 / 2 = outer / middle / inner third of the wall (bottom: lower / middle / upper)."""
    z = p[:, 2]; r = np.hypot(p[:, 0], p[:, 1])
    taper = math.atan2(top_r - bot_r, h); off = wall / math.cos(taper) if abs(taper) > 1e-6 else wall
    ro = bot_r + (top_r - bot_r) * np.clip(z / h, 0, 1); ri = np.maximum(ro - off, 0.001)
    lay = np.full(len(p), -1)
    w = (z >= bottom) & (z <= h) & (r >= ri) & (r <= ro)
    lay[w] = np.minimum(((ro[w] - r[w]) / off * 3).astype(int), 2)
    b = (z >= 0) & (z < bottom) & (r <= ro)
    lay[b] = np.minimum((z[b] / bottom * 3).astype(int), 2)
    return lay


def side_of(c, top_r, bot_r, h, wall, bottom, n_min=200):
    """Where a link's samples near the cup are: 'cavity' (inside the cup only), 'outside' (beside the wall only, below
    the rim), 'straddle' (both, e.g. a finger hooked over the rim from inside), or None (not near the wall)."""
    z = c[:, 2]; r = np.hypot(c[:, 0], c[:, 1])
    ro = bot_r + (top_r - bot_r) * np.clip(z / h, 0, 1); ri = ro - wall
    cav = int(((r < ri) & (z > bottom) & (z < h)).sum()); out = int(((r > ro) & (r < ro + 0.02) & (z > 0) & (z < h)).sum())
    if cav >= n_min and out >= n_min:
        return "straddle"
    return "cavity" if cav >= n_min else ("outside" if out >= n_min else None)


def main():
    pitch = float(A.get("pitch_mm", "0.2")) / 1000
    links = A.get("links_used", "link5,link6,link7").split(",")
    P = link_samples(A["links"], links, pitch)
    of, oid = A["cup"].rsplit(":", 1)
    o = next(x for x in json.load(open(of))["objects"] if x["id"] == oid)
    bottle = o.get("shape") == "bottle"
    if bottle:
        # 2026-11-11: a bottle is a SOLID of revolution (sim/bottle_mesh.py), no wall and no cavity. vol_mm3 = link volume
        # inside the solid; "through" = some sample deeper than bottle_deep_mm (default 1.0, the cup's wall thickness,
        # threshold not decided); no side flips (nothing to be inside of).
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from bottle_mesh import depth_inside
        deep = float(A.get("bottle_deep_mm", "1.0")) / 1000
        rmax = max(o["body_dia"], o.get("base_dia", 0), o["cap_dia"], o.get("waist_dia", 0)) / 2
        lo_c = np.array([-rmax, -rmax, 0.0]) - 1e-3; hi_c = np.array([rmax, rmax, o["height"]]) + 1e-3
        print(f"bottle {oid}: solid, through = deeper than {deep * 1000:.2f} mm", flush=True)
    else:
        geo = (o["top_dia"] / 2, o["bot_dia"] / 2, o["height"], o.get("wall", 0.001), o.get("bottom") or o.get("wall", 0.001))
        lo_c = np.array([-geo[0], -geo[0], 0.0]) - 1e-3; hi_c = np.array([geo[0], geo[0], geo[2]]) + 1e-3   # cup box, cup frame
    res = {}
    for lp in A["logs"].split(","):
        D = json.load(open(lp)); L = D["log"]; names = [b.split("/")[-1] for b in D["link_bodies"]]
        per = []
        for r in L:
            cp = r["cup_pose"]; cp = json.loads(cp) if isinstance(cp, str) else cp
            lps = r["link_poses"]; lps = json.loads(lps) if isinstance(lps, str) else lps
            Rc, oc = R(cp[3:7]), np.array(cp[:3]); fr = {"frame": r["frame"]}
            for l, pts in P.items():
                T = lps[names.index(l)]; Rl, ol = R(T[3:7]), np.array(T[:3])
                n_in, thr, side = 0, False, None
                if np.linalg.norm(ol - oc) < 0.25:
                    # the cup's box in the link frame -> only samples inside it are moved and tested
                    cor = np.array([[x, y, z] for x in (lo_c[0], hi_c[0]) for y in (lo_c[1], hi_c[1]) for z in (lo_c[2], hi_c[2])])
                    cl = (Rl.T @ ((Rc @ cor.T).T + oc - ol).T).T
                    sel = np.all((pts >= cl.min(0)) & (pts <= cl.max(0)), axis=1)
                    if sel.any():
                        c = (Rc.T @ (((Rl @ pts[sel].T).T + ol) - oc).T).T
                        if bottle:
                            d = depth_inside(c, o); n_in = int((d > 0).sum()); thr = bool((d > deep).any())
                        else:
                            lay = wall_layer(c, *geo); n_in = int((lay >= 0).sum()); thr = bool(all((lay == k).any() for k in (0, 1, 2)))
                            side = side_of(c, *geo)
                fr[l] = {"vol_mm3": float(n_in * (pitch * 1000) ** 3), "through": thr, "side": side}
            per.append(fr)
        # between two logged frames a link can go through the wall and come out clean on the other side: flag a link
        # whose side flips cavity <-> outside from one frame to the next without being over the rim in either
        for a_, b_ in zip(per, per[1:]):
            for l in P:
                sa, sb = a_[l]["side"], b_[l]["side"]
                if sa and sb and {sa, sb} == {"cavity", "outside"}:
                    b_[l]["through"] = True; b_[l]["side_flip"] = True
        vols = np.array([[f[l]["vol_mm3"] for l in P] for f in per]); thr = np.array([[f[l]["through"] for l in P] for f in per])
        res[lp] = {"links": list(P), "max_vol_mm3": vols.max(0).round(2).tolist(), "frames_through": thr.sum(0).tolist(),
                   "frames_vol_gt_1mm3": int((vols.sum(1) > 1.0).sum()), "n_frames": len(per), "per_frame": per}
        print(lp, "max overlap mm3", dict(zip(P, vols.max(0).round(1))), "frames through", dict(zip(P, thr.sum(0).tolist())),
              "frames >1 mm3:", res[lp]["frames_vol_gt_1mm3"], "/", len(per), flush=True)
    if "out" in A:
        json.dump(res, open(A["out"], "w"))


if __name__ == "__main__":
    main()

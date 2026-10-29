# Fit the base plate pose (in the ArUco paper frame) to its detected edges, camera fixed at the v2 ArUco extrinsics.
import sys, json, numpy as np
from scipy.optimize import least_squares
extr, intr, edges = sys.argv[1:4]
E = json.load(open(extr)); K = json.load(open(intr)); D = json.load(open(edges)); c = K["distortion_coeffs"]
w, x, y, z = E["quat_wxyz"]
Rc = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
C = np.array(E["pos_m"])
def proj(P):
    pc = (np.atleast_2d(P) - C) @ Rc; xn, yn = pc[:, 0]/pc[:, 2], pc[:, 1]/pc[:, 2]
    k1, k2, p1, p2, k3 = c; r2 = xn*xn + yn*yn; f = 1 + k1*r2 + k2*r2*r2 + k3*r2**3; xq, yq = xn*f, yn*f
    return np.c_[K["fx"]*(xq + 2*p1*xq*yq + p2*(r2 + 2*xq*xq)) + K["cx"], K["fy"]*(yq + 2*p2*xq*yq + p1*(r2 + 2*yq*yq)) + K["cy"]]
S = 0.01125
FC = np.array([0.060 + S, 0.067, 0.0]); CL = np.array([0.052 + S, 0.075, 0.0]); LB = np.array([0.035, 0.075, 0.0]); UP = np.array([0, 0, 0.0125])
def Ry(t): ct, st = np.cos(t), np.sin(t); return np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
def Rz(t): ct, st = np.cos(t), np.sin(t); return np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
def pose(p, tilt):
    dx, dy, zr, yaw = p; R = Rz(yaw) @ Ry(tilt); t = np.array([dx, dy, zr])
    return lambda q: (R @ np.asarray(q).T).T + t
def dist_to_poly(pts, poly):
    a, b = poly[:-1], poly[1:]; ab = b - a
    t = np.clip(((pts[:, None, :] - a[None]) * ab[None]).sum(-1) / ((ab*ab).sum(-1)[None] + 1e-12), 0, 1)
    return np.linalg.norm(pts[:, None, :] - (a[None] + t[..., None]*ab[None]), axis=-1).min(1)
L = np.array(D["left"]); B = np.array(D["bottom"])
s = np.linspace(0, 1, 40)[:, None]
def model(p, tilt):
    T = pose(p, tilt)
    edge1 = proj(T(FC + s*UP*3.0))                       # front/chamfer vertical edge (line extended up, top is occluded)
    bot = proj(T(np.r_[FC + s*(CL-FC), CL + s*(LB-CL)])) # chamfer bottom + left-face bottom
    return edge1, bot
def resid(p, tilt):
    e1, bt = model(p, tilt); return np.r_[dist_to_poly(L, e1), dist_to_poly(B, bt)]
p0 = np.array([0.0, 0.0, 0.146, 0.0])
for tilt_deg in (0.0, 3.35):
    t = np.radians(tilt_deg)
    r0 = resid(p0, t)
    print("tilt %.2f deg  nominal (dx=dy=yaw=0, riser 0.146): median %.1f px, max %.1f px" % (tilt_deg, np.median(r0), r0.max()))
    for free in (("zr",), ("dx", "dy", "zr", "yaw")):
        idx = [("dx", "dy", "zr", "yaw").index(f) for f in free]
        def rr(q):
            p = p0.copy(); p[idx] = q; return resid(p, t)
        sol = least_squares(rr, p0[idx], x_scale=[0.01 if f != "yaw" else 0.05 for f in free])
        J = sol.jac; sig = np.sqrt(np.diag(np.linalg.pinv(J.T @ J))) * 1.0   # 1 px per-point noise
        out = ", ".join("%s=%+.1f%s(±%.1f)" % (f, v*1000 if f != "yaw" else np.degrees(v), "mm" if f != "yaw" else "deg", sg*1000 if f != "yaw" else np.degrees(sg)) for f, v, sg in zip(free, sol.x, sig))
        print("   free %-18s -> %s ; median %.2f px, max %.2f px; cond %.0f" % ("+".join(free), out, np.median(np.abs(sol.fun)), np.abs(sol.fun).max(), np.linalg.cond(J)))
if len(sys.argv) > 5:
    from PIL import Image, ImageDraw
    im = Image.open(sys.argv[4]).convert("RGB"); x0, y0, sc = 745, 280, 7
    cimg = im.crop((x0, y0, 848, 365)).resize(((848-x0)*sc, (365-y0)*sc), Image.LANCZOS); d = ImageDraw.Draw(cimg)
    P = lambda uv: [((u-x0+0.5)*sc, (v-y0+0.5)*sc) for u, v in uv]
    for u, v in D["bottom"]: d.ellipse([*(np.array(P([(u, v)])[0])-3), *(np.array(P([(u, v)])[0])+3)], outline=(255, 255, 0))
    for u, v in D["left"]: d.ellipse([*(np.array(P([(u, v)])[0])-3), *(np.array(P([(u, v)])[0])+3)], outline=(255, 255, 0))
    for (u, v) in (D["turn"], D["front_corner"]):
        q = np.array(P([(u, v)])[0]); d.ellipse([*(q-9), *(q+9)], outline=(255, 0, 255), width=3)
    for p, tilt, col in ((p0, 0.0, (255, 60, 60)), (np.array([0, 0, 0.1443, 0]), np.radians(3.35), (0, 255, 0))):
        e1, bt = model(p, tilt)
        d.line(P(e1), fill=col, width=2); d.line(P(bt), fill=col, width=2)
    cimg.save(sys.argv[5])

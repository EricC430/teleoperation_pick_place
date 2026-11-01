import sys, numpy as np
exec(open(sys.argv[0].replace("fit3.py", "fit_base.py")).read().split("p0 = np.array")[0])
from scipy.optimize import least_squares
from PIL import Image
g = np.asarray(Image.open("calibration/shots/2026-10-18_fl_world_check_v2.png").convert("L")).astype(float)
# clamp (bracket) lower edge: first dark pixel going down from inside the bracket, per column
clamp = []
for u in range(772, 848, 2):
    col = g[285:330, u]; i = np.where(col < 40)[0]
    if len(i): clamp.append((u, 285 + i[0]))
clamp = np.array(clamp)
# user-marked chamfer/left boundary: middle of the three drawn lines, 12 samples
tt = np.linspace(0, 1, 12)[:, None]
U = np.array([796.3, 304.7]) * (1-tt) + np.array([788.0, 335.0]) * tt
HH = np.array([0, 0, 0.0125])
def groups(p, tilt):
    Tf = pose(p, tilt)
    e1 = proj(Tf(FC + s*HH*3.0)); e2 = proj(Tf(CL + s*HH*3.0))
    bot = proj(Tf(np.r_[FC + s*(CL-FC), CL + s*(LB-CL)]))
    top = proj(Tf(np.r_[FC+HH + s*(CL-FC), CL+HH + s*(LB-CL)]))
    return dist_to_poly(L, e1), dist_to_poly(B, bot), dist_to_poly(U, e2), top
def clamp_margin(top):
    # positive = plate top edge is ABOVE the clamp's lower edge (hidden); negative = it would show below the clamp
    m = []
    for u, vc in clamp:
        seg = top[(top[:, 0] > u-1) & (top[:, 0] < u+1)]
        if len(seg): m.append(vc - seg[:, 1].mean())
    return np.array(m)
def report(tag, p, tilt):
    a, b, c_, top = groups(p, tilt); m = clamp_margin(top)
    print("%-44s dx %+5.1f dy %+5.1f z %5.1f mm yaw %+5.2f | left %.2f  bottom %.2f  your-line %.2f px (median) | top edge vs clamp: min margin %+.1f px" % (
        tag, p[0]*1e3, p[1]*1e3, p[2]*1e3, np.degrees(p[3]), np.median(a), np.median(b), np.median(c_), m.min()))
tilt = np.radians(3.35)
report("nominal (0,0,146,0)", np.array([0, 0, 0.146, 0]), tilt)
report("previous fit (yaw free, no your-line)", np.array([-0.0112, 0.0100, 0.1443, np.radians(-8.91)]), tilt)
for use_user in (False, True):
    def f(q):
        p = np.array([q[0], q[1], q[2], 0.0]); a, b, c_, _ = groups(p, tilt)
        return np.r_[a, b, c_] if use_user else np.r_[a, b]
    sol = least_squares(f, [0, 0, 0.146], x_scale=[0.01, 0.01, 0.01])
    report("yaw=0, fit dx dy z" + (" + your line" if use_user else ""), np.array([*sol.x, 0.0]), tilt)
def f4(q):
    a, b, c_, _ = groups(np.array(q), tilt); return np.r_[a, b, c_]
sol = least_squares(f4, [0, 0, 0.146, 0], x_scale=[0.01, 0.01, 0.01, 0.05])
report("yaw free, fit dx dy z yaw + your line", sol.x, tilt)
p = np.array([-0.0037, 0.0091, 0.1479, np.radians(0.15)])
_, _, _, top = groups(p, tilt)
print("u    clamp_lower_v   plate_top_v   margin(+ = hidden)")
for u, vc in clamp:
    seg = top[(top[:, 0] > u-1) & (top[:, 0] < u+1)]
    if len(seg): print("%d   %6.1f        %6.1f       %+5.1f" % (u, vc, seg[:, 1].mean(), vc - seg[:, 1].mean()))
print("--- sensitivity to which of your three lines")
for name, a0, a1 in (("left line", (793.8, 304.7), (786.2, 334.2)), ("middle", (796.3, 304.7), (788.0, 335.0)), ("right line", (799.2, 304.7), (790.0, 336.7))):
    U = np.array(a0)*(1-tt) + np.array(a1)*tt
    sol = least_squares(f4, [0, 0, 0.146, 0], x_scale=[0.01, 0.01, 0.01, 0.05])
    report(name, sol.x, tilt)
# render the middle-line fit
U = np.array((796.3, 304.7))*(1-tt) + np.array((788.0, 335.0))*tt
sol = least_squares(f4, [0, 0, 0.146, 0], x_scale=[0.01, 0.01, 0.01, 0.05]); Tf = pose(sol.x, tilt)
from PIL import ImageDraw
base = Image.open("calibration/shots/2026-10-18_fl_world_check_v2.png").convert("RGB")
x0, y0, x1, y1, sc = 745, 262, 848, 362, 6
HH = np.array([0, 0, 0.0125])
edges = [(FC + s*HH, (0, 255, 0)), (CL + s*HH, (255, 0, 255)),
         (np.r_[FC + s*(CL-FC), CL + s*(LB-CL)], (255, 60, 60)), (np.r_[FC+HH + s*(CL-FC), CL+HH + s*(LB-CL)], (0, 200, 255))]
def panel(lines, pts):
    c = base.crop((x0, y0, x1, y1)).resize(((x1-x0)*sc, (y1-y0)*sc), Image.LANCZOS); d = ImageDraw.Draw(c)
    P = lambda uv: [((u-x0+0.5)*sc, (v-y0+0.5)*sc) for u, v in uv]
    if lines:
        for poly, col in edges: d.line(P(proj(Tf(poly))), fill=col, width=2)
    if pts:
        for u, v in list(D["bottom"]) + list(D["left"]):
            q = P([(u, v)])[0]; d.ellipse([q[0]-2, q[1]-2, q[0]+2, q[1]+2], fill=(255, 255, 0))
        d.line(P(U), fill=(255, 255, 255), width=1)
        for u, vc in clamp:
            q = P([(u, vc)])[0]; d.ellipse([q[0]-2, q[1]-2, q[0]+2, q[1]+2], outline=(255, 128, 0))
    return c
A, B2, C2 = panel(False, False), panel(True, False), panel(True, True)
W = Image.new("RGB", (A.width*3 + 20, A.height + 50), (255, 255, 255)); dd = ImageDraw.Draw(W)
for i, (im, t) in enumerate(((A, "raw"), (B2, "fit: dx %+.1f dy %+.1f z %.1f mm yaw %+.2f deg, tilt 3.35" % (sol.x[0]*1e3, sol.x[1]*1e3, sol.x[2]*1e3, np.degrees(sol.x[3]))),
                             (C2, "+ yellow = detected edges, white = your middle line, orange = clamp lower edge"))):
    W.paste(im, (i*(A.width+10), 50)); dd.text((i*(A.width+10)+4, 30), t, fill=(0, 0, 0))
dd.text((4, 4), "green front/chamfer vertical   magenta chamfer/left vertical   red bottom   cyan top", fill=(0, 0, 0))
W.save("%s/fit_check2.png" % sys.argv[0].rsplit("/", 1)[0])

# Camera 6-DoF in the ARM frame from base-plate edges only (no ArUco in the solve; ArUco only gives the start point).
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(HERE + "/fit3.py").read().split("def report")[0].replace('exec(open(sys.argv[0].replace("fit3.py", "fit_base.py")).read().split("p0 = np.array")[0])', 'exec(open("' + HERE + '/fit_base.py").read().split("p0 = np.array")[0])'))
from scipy.optimize import least_squares
tilt = np.radians(3.35)
pb = np.array([-0.0037, 0.0091, 0.1479, np.radians(0.15)])          # base in paper frame (previous fit, middle line)
Tpb_R = Rz(pb[3]) @ Ry(tilt); Tpb_t = pb[:3]
# camera in arm frame from the ArUco camera:  X_arm = R^T (X_paper - t)
C_arm0 = Tpb_R.T @ (C - Tpb_t); Rc_arm0 = Tpb_R.T @ Rc
def rotvec(w):
    th = np.linalg.norm(w)
    if th < 1e-12: return np.eye(3)
    k = w/th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th)*Kx + (1-np.cos(th))*Kx@Kx
def proj_cam(P, q):                       # q = (rx, ry, rz [deg], tx, ty, tz [mm]) perturbation of the camera in the arm frame
    Rq = rotvec(np.radians(q[:3])) @ Rc_arm0; Cq = C_arm0 + np.asarray(q[3:])*1e-3
    pc = (np.atleast_2d(P) - Cq) @ Rq; xn, yn = pc[:, 0]/pc[:, 2], pc[:, 1]/pc[:, 2]
    k1, k2, p1, p2, k3 = c; r2 = xn*xn + yn*yn; f = 1 + k1*r2 + k2*r2*r2 + k3*r2**3; xq, yq = xn*f, yn*f
    return np.c_[K["fx"]*(xq + 2*p1*xq*yq + p2*(r2 + 2*xq*xq)) + K["cx"], K["fy"]*(yq + 2*p2*xq*yq + p1*(r2 + 2*yq*yq)) + K["cy"]]
Hh = np.array([0, 0, 0.0125])
def resid_cam(q, Lp, Bp, Up):
    e1 = proj_cam(FC + s*Hh*3.0, q); e2 = proj_cam(CL + s*Hh*3.0, q)
    bot = proj_cam(np.r_[FC + s*(CL-FC), CL + s*(LB-CL)], q)
    return np.r_[dist_to_poly(Lp, e1), dist_to_poly(Bp, bot), dist_to_poly(Up, e2)]
q0 = np.zeros(6)
r0 = resid_cam(q0, L, B, U); print("start (ArUco camera + fitted base): median %.2f px" % np.median(r0))
eps = 0.05; J = np.zeros((len(r0), 6))
for i in range(6):
    e = np.zeros(6); e[i] = eps; J[:, i] = (resid_cam(e, L, B, U) - resid_cam(-e, L, B, U)) / (2*eps)
_, sv, Vt = np.linalg.svd(J, full_matrices=False)
names = ["rx", "ry", "rz", "tx", "ty", "tz"]
print("singular values (px per deg or mm):", sv.round(3))
for k in (4, 5):
    print("  weak dir %d:" % k, ", ".join("%s %+.2f" % (n, v) for n, v in zip(names, Vt[k]/np.abs(Vt[k]).max())))
# What a weak-direction move does to the WORKSPACE: project the placement-area points
W = np.array([[x, y, 0.0] for x in (0.15, 0.25, 0.35, 0.45) for y in (-0.15, 0.0, 0.15)])
W_arm = (Tpb_R.T @ (W - Tpb_t).T).T            # table points (paper frame) into the arm frame
for k in (5, 4):
    v = Vt[k] / np.linalg.norm(J @ Vt[k]) * np.sqrt(len(r0))   # scaled so base-feature RMS change = 1 px
    dW = proj_cam(W_arm, v) - proj_cam(W_arm, q0)
    print("  move along weak dir %d until base features change 1 px RMS -> workspace points move median %.1f px, max %.1f px (%s)" % (
        k, np.median(np.linalg.norm(dW, axis=1)), np.linalg.norm(dW, axis=1).max(), ", ".join("%s %+.1f" % (n, x) for n, x in zip(names, v))))
# noise test: 0.5 px on detected points, 1.5 px on your hand line, 100 refits
rng = np.random.default_rng(1); dWs = []
for _ in range(100):
    Lp = L + rng.normal(0, 0.5, L.shape); Bp = B + rng.normal(0, 0.5, B.shape); Up = U + rng.normal(0, 1.5, (1, 2))
    sol = least_squares(lambda q: resid_cam(q, Lp, Bp, Up), q0, x_scale=[0.5]*3 + [2.0]*3)
    dWs.append(np.linalg.norm(proj_cam(W_arm, sol.x) - proj_cam(W_arm, q0), axis=1))
dWs = np.array(dWs)
print("noise refits: workspace reprojection shift median %.1f px, 90th pct %.1f px, worst %.1f px" % (np.median(dWs), np.percentile(dWs, 90), dWs.max()))

# Base plate edges, v2: bottom = FIRST black->brown transition below the plate (R channel), not the strongest gradient.
import sys, json, numpy as np
from PIL import Image
img, out = sys.argv[1], sys.argv[2]
a = np.asarray(Image.open(img).convert("RGB")).astype(float)
R = a[:, :, 0]; G = np.asarray(Image.open(img).convert("L")).astype(float)
k = np.exp(-0.5*(np.arange(-3, 4)/1.0)**2); k /= k.sum()
bottom = []
for u in range(773, 829):
    p = np.convolve(R[300:365, u-1:u+2].mean(1), k, "same")
    start = 15                                   # v=315, inside the plate for every column used
    plate = np.median(p[start-8:start])
    for i in range(start, len(p)-1):
        thr = plate + 10.0
        if p[i] < thr <= p[i+1]:
            bottom.append((u, 300 + i + (thr - p[i])/(p[i+1]-p[i]))); break
left = []
for v in range(298, 318):
    p = np.convolve(G[v-1:v+2, 745:800].mean(0), k, "same")
    d = -np.gradient(p); i = int(np.argmax(d[3:-3])) + 3
    a_, b_, c_ = d[i-1], d[i], d[i+1]; off = 0.5*(a_-c_)/(a_-2*b_+c_)
    left.append((745 + i + off, v))
B = np.array(bottom); L = np.array(left)
best = None
for j in range(5, len(B)-5):
    rss = 0; lines = []
    for part in (B[:j+1], B[j:]):
        c = np.polyfit(part[:, 0], part[:, 1], 1); rss += ((np.polyval(c, part[:, 0])-part[:, 1])**2).sum(); lines.append(c)
    if best is None or rss < best[0]: best = (rss, lines)
rss, (l1, l2) = best
ub = (l2[1]-l1[1])/(l1[0]-l2[0]); vb = np.polyval(l1, ub)
c1 = np.polyfit(B[:, 0], B[:, 1], 1); rss1 = ((np.polyval(c1, B[:, 0])-B[:, 1])**2).sum()
cl = np.polyfit(L[:, 1], L[:, 0], 1)   # u = f(v)
# front corner = left silhouette line x bottom segment A
vf = (l1[0]*cl[1] + l1[1]) / (1 - l1[0]*cl[0]); uf = np.polyval(cl, vf)
print("bottom %d cols: 1-line RSS %.1f px^2, 2-line RSS %.1f px^2" % (len(B), rss1, rss))
print("  seg A slope %.3f, seg B slope %.3f, turning point (%.1f, %.1f)" % (l1[0], l2[0], ub, vb))
print("  left silhouette u = %.4f*v + %.1f ; meets seg A at (%.1f, %.1f)" % (cl[0], cl[1], uf, vf))
print("  samples:", " ".join("%d:%.1f" % (u, v) for u, v in B[::5]))
json.dump({"bottom": B.tolist(), "left": L.tolist(), "seg_a": l1.tolist(), "seg_b": l2.tolist(), "turn": [ub, vb], "front_corner": [uf, vf]}, open(out, "w"))

"""Annotated concept sheet: hero render + numbered callouts projected from 3D part centres."""
import sys, json
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from paths import BASE
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import design as D

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK TC'

IMG, W, H = sys.argv[1], 1600, 1100
POS = np.array([float(v) for v in sys.argv[2].split(',')]); TGT = np.array([float(v) for v in sys.argv[3].split(',')]); FOV = 32
POSE = sys.argv[4]; OUT = sys.argv[5]

def project(p):
    f = TGT - POS; f /= np.linalg.norm(f); r = np.cross(f, [0, 0, 1]); r /= np.linalg.norm(r); u = np.cross(r, f)
    d = p - POS; x, y, z = d @ r, d @ u, d @ f
    t = np.tan(np.radians(FOV / 2))
    return (W / 2 + x / z / t * H / 2, H / 2 - y / z / t * H / 2)

parts = {m.metadata['name']: m for m in D.vehicle()}
def centre(key, frac=None):
    ms = [m for n, m in parts.items() if key in n]
    b = np.vstack([m.bounds for m in ms]); c = (b.min(0) + b.max(0)) / 2
    if frac is not None: c[2] = b.min(0)[2] + frac * (b.max(0)[2] - b.min(0)[2])
    return c
P = D.P
q = json.load(open(str(BASE)+'/out/poses.json'))[POSE]
from kin import OMX
Wl = OMX.fk(q, D.arm_base_T())
callouts = [
 (centre('arm_front_plate'), '前伸座板＋快拆治具', '6 mm 鋁，超出甲板 12 cm，2 定位銷＋4 手轉螺絲'),
 (Wl['link3'][:3, 3] + Wl['link3'][:3, :3] @ np.array([0.08, 0, 0]), 'OMX-F', '地面可夾約 1500 cm²'),
 (centre('bag_hoop_ring'), '垃圾袋撐環', '內徑 170 mm，袋口離地 22 cm'),
 (np.array(P['cam_pos']), 'D455 第三視角', f"離地 {P['cam_pos'][2]*100:.0f} cm，固定角度"),
 (centre('mast_brace', 0.5), '2020 桅桿＋三向斜撐', '懸臂約 31 cm'),
 (centre('lidar_RPLIDAR_C1_head'), 'RPLIDAR C1', '掃描面高過收納手臂'),
 (centre('edge_compute'), '邊緣運算盒', 'Jetson Orin Nano'),
 (centre('power_distribution'), '配電盒', '保險絲、12 V、USB-PD、急停'),
 (centre('power_station'), '行動電源站＋延長線', 'C300 級，後掛架'),
 (centre('arm_cradle'), '行車托塊', 'TPU'),
]
img = plt.imread(IMG)
fig = plt.figure(figsize=(16, 11), dpi=110)
ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
pts = [(project(p), t, s) for p, t, s in callouts]
left = sorted([c for c in pts if c[0][0] < W * 0.5], key=lambda c: c[0][1])
right = sorted([c for c in pts if c[0][0] >= W * 0.5], key=lambda c: c[0][1])
def place(group, x_text, ha):
    n = len(group); ys = np.linspace(170, H - 170, n) if n > 1 else [H / 2]
    for (xy, t, s), yt in zip(group, ys):
        num = [c[1] for c in callouts].index(t) + 1
        ax.plot([xy[0], x_text + (12 if ha == 'left' else -12)], [xy[1], yt], color='#1f5fbf', lw=1.1)
        ax.plot(*xy, 'o', ms=6, mfc='white', mec='#1f5fbf', mew=1.6)
        ax.text(x_text, yt - 8, f"{num}  {t}", ha=ha, va='bottom', fontsize=15, fontweight='bold', color='#18222c')
        ax.text(x_text, yt - 4, s, ha=ha, va='top', fontsize=11.5, color='#4a5866')
place(left, 60, 'left'); place(right, W - 60, 'right')
ax.text(60, 70, 'Wildbot × OMX 改裝概念圖', fontsize=24, fontweight='bold', color='#18222c')
ax.text(60, 108, '概念示意 v0.1 · 2026-10-03 · 尺寸取自 URDF，零件盒為佔位尺寸', fontsize=12, color='#4a5866')
fig.savefig(OUT, dpi=110); print('ok', OUT)

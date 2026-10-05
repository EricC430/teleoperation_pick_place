import sys,json; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from paths import BASE
import numpy as np, design as D, camera as C
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Polygon, Circle
from scipy.spatial import ConvexHull
for f in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']: fm.fontManager.addfont(f)
plt.rcParams['font.family']='Noto Sans CJK TC'; plt.rcParams['axes.unicode_minus']=False
R=np.load(str(BASE)+'/reach_grid.npz'); xs,ys,g=R['xs'],R['ys'],R['grid']; V=np.load(str(BASE)+'/cam_visible_cells.npy')
fig,ax=plt.subplots(figsize=(10,8.2),dpi=150)
# plot in "driver view": forward (-x) up  => plot X = -y (right is right), Y = -x (forward up)
def tf(x,y): return np.asarray(y), -np.asarray(x)
cmap=plt.get_cmap('viridis')
for i,x in enumerate(xs):
    for j,y in enumerate(ys):
        v=g[i,j]
        if v>=0:
            c='#1f8a70' if v>=75 else '#7fc8a9'
            X,Y=tf(x,y); ax.add_patch(plt.Rectangle((X-0.01,Y-0.01),0.02,0.02,color=c,lw=0))
        elif v==-2:
            X,Y=tf(x,y); ax.add_patch(plt.Rectangle((X-0.01,Y-0.01),0.02,0.02,color='#e8a33d',lw=0))
occ=V[V[:,2]==0]
X,Y=tf(occ[:,0],occ[:,1]); ax.scatter(X,Y,marker='x',s=14,c='#b3261e',lw=0.9,label='第三視角被遮擋（手臂收納時）',zorder=5)
for m in D.vehicle():
    n=m.metadata['name']
    if 'trash_bag' in n: continue
    pts=m.vertices[:,:2]
    try: h=ConvexHull(pts)
    except Exception: continue
    P=np.c_[tf(pts[h.vertices,0],pts[h.vertices,1])]
    fc='#3a6ea5' if 'upper_blue' in n else '#555' if n.startswith('wildbot') else '#c9c9c9'
    if 'power' in n: fc='#c0504d'
    if 'edge_compute' in n: fc='#2e8c5a'
    if 'camera' in n or 'lidar' in n: fc='#222'
    ax.add_patch(Polygon(P,closed=True,fc=fc,ec='#333',lw=0.4,alpha=0.85 if n.startswith('mod') else 0.6,zorder=3))
hx,hy=D.P['hoop_c']; X,Y=tf(hx,hy); ax.add_patch(Circle((X,Y),D.P['hoop_r'],fc='#9fd0f0',ec='#d48a1a',lw=2,alpha=0.8,zorder=4))
ax.text(X,Y,'垃圾袋',ha='center',va='center',fontsize=9,zorder=6)
# camera FOV footprint on ground
c=D.CAMS[D.P['cam_model']]; Tc=D.look_at(D.P['cam_pos'],D.P['cam_target'])
corn=[]
for su,sv in [(-1,-1),(1,-1),(1,1),(-1,1)]:
    d=Tc[:3,:3]@np.array([1,-su*np.tan(np.radians(c['hfov']/2)),sv*np.tan(np.radians(c['vfov']/2))])
    o=Tc[:3,3]; t=-o[2]/d[2] if d[2]<0 else 3.0; t=min(t,3.0); corn.append(o+d*t)
corn=np.array(corn); X,Y=tf(corn[:,0],corn[:,1]); ax.add_patch(Polygon(np.c_[X,Y],closed=True,fill=False,ec='#6b4fbb',lw=1.6,ls='--',zorder=6,label='D455 視野在地面的範圍'))
cx,cy=tf(D.P['cam_pos'][0],D.P['cam_pos'][1]); ax.plot(cx,cy,'s',c='#6b4fbb',ms=8,zorder=7)
ax.annotate(f"D455 @ {D.P['cam_pos'][2]*100:.0f} cm",(cx,cy),(cx+0.08,cy-0.12),fontsize=9,arrowprops=dict(arrowstyle='-',color='#6b4fbb'),color='#6b4fbb')
ax.plot([],[],'s',c='#1f8a70',label='可夾（夾爪近垂直 ≥75°）'); ax.plot([],[],'s',c='#7fc8a9',label='可夾（夾爪斜 45–75°）'); ax.plot([],[],'s',c='#e8a33d',label='幾何可達但會撞車體')
ax.set_xlim(-0.62,0.62); ax.set_ylim(-0.42,0.66); ax.set_aspect('equal'); ax.grid(alpha=0.25)
ax.set_xlabel('← 左　　橫向 (m)　　右 →'); ax.set_ylabel('縱向 (m)　前進方向 ↑')
ax.set_title('地面可夾範圍（物體中心高 3 cm）與相機視野　俯視，車頭朝上',fontsize=12)
ax.legend(loc='lower right',fontsize=8.5,framealpha=0.92)
r=json.load(open(str(BASE)+'/out/results.json'))
ax.text(-0.60,0.63,f"可夾面積 {r['reach']['area_cm2']} cm²（近垂直 {r['reach']['near_vertical_cm2']} cm²）\n正前方可夾帶：前輪外 {abs(r['reach']['centreline_x'][0])-0.195:.2f}–{abs(r['reach']['centreline_x'][1])-0.195:.2f} m\n第三視角可見 {r['camera']['coverage_visible']*100:.0f}%（視野內 {r['camera']['coverage_in_fov']*100:.0f}%）",fontsize=9,va='top',bbox=dict(fc='white',ec='#ccc'))
plt.tight_layout(); plt.savefig(str(BASE)+'/renders/reach_map.png'); print('ok')

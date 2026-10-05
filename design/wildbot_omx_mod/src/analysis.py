import sys, json; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from design import *
from paths import BASE
import numpy as np, trimesh, fast_simplification
from trimesh.collision import CollisionManager
T=np.load(str(BASE)+'/planar_table.npz'); TX,TZ,TP=T['x'],T['z'],T['pitch']
CELL=0.004
key=lambda x,z:(np.floor(x/CELL).astype(int),np.floor(z/CELL).astype(int))
kx,kz=key(TX,TZ); BINS={}
for i,(a,b) in enumerate(zip(kx,kz)): BINS.setdefault((a,b),[]).append(i)
BINS={k:np.array(v) for k,v in BINS.items()}
def decimate(m,target=1500):
    if len(m.faces)<=target: return m
    v,f=fast_simplification.simplify(m.vertices,m.faces,target_reduction=1-target/len(m.faces)); return trimesh.Trimesh(v,f)
ARM_LOCAL={l:decimate(OMX.mesh(l)) for l in OMX.links if OMX.mesh(l) is not None}
SOFT=('trash_bag',)
def vehicle_cm(p=P,extra_skip=()):
    cm=CollisionManager()
    for m in vehicle(p):
        n=m.metadata['name']
        if any(s in n for s in SOFT+tuple(extra_skip)): continue
        cm.add_object(n,decimate(m,3000))
    g=trimesh.creation.box(extents=(3,3,0.1)); g.apply_translation([0,0,-0.05]); cm.add_object('GROUND',g)
    return cm
def pose_collides(cm,q,p=P,ignore_ground_links=()):
    W=OMX.fk(q,arm_base_T(p)); hits=set()
    for l,m in ARM_LOCAL.items():
        if l in('link0','link1'): continue
        hit,names=cm.in_collision_single(m,transform=W[l],return_names=True)
        if hit: hits|={n for n in names}
    return hits
def solve(target,cm,p=P,pitch_win=(45,100),prefer=90,max_try=25,return_all=False):
    B=arm_base_T(p); pa=np.linalg.inv(B)@np.r_[target,1]
    j1=np.arctan2(pa[1],pa[0]); r=np.hypot(pa[0]-(-0.01125),pa[1])  # joint1 axis at link0 x=-0.01125
    j1=np.arctan2(pa[1],pa[0]+0.01125)
    zrel=pa[2]-0.0975
    a,b=key(np.array(r),np.array(zrel)); idx=BINS.get((int(a),int(b)))
    if idx is None: return None
    ok=idx[(TP[idx]>=pitch_win[0])&(TP[idx]<=pitch_win[1])]
    if len(ok)==0: return None
    ok=ok[np.argsort(np.abs(TP[ok]-prefer))]
    step=max(1,len(ok)//max_try)
    for i in ok[::step][:max_try]:
        q={'joint1':j1,'joint2':T['Q2'][i],'joint3':T['Q3'][i],'joint4':T['Q4'][i],'joint5':0.0}
        if not pose_collides(cm,q,p): return q|{'pitch':float(TP[i])}
    return 'collide'
if __name__=='__main__':
    cm=vehicle_cm()
    xs=np.arange(-0.62,0.36,0.02); ys=np.arange(-0.50,0.52,0.02); Z=0.03
    grid=np.full((len(xs),len(ys)),-1.0)  # -1 unreachable, -2 collide-only, >=0 pitch
    foot=lambda x,y:(-0.20<=x<=0.32) and (-0.17<=y<=0.17)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            if foot(x,y) or (np.hypot(x-P['hoop_c'][0],y-P['hoop_c'][1])<0.10): grid[i,j]=-3; continue
            s=solve(np.array([x,y,Z]),cm)
            grid[i,j]=-1 if s is None else -2 if s=='collide' else s['pitch']
    np.savez(str(BASE)+'/reach_grid.npz',xs=xs,ys=ys,grid=grid)
    good=grid>=0; print('ground cells reachable:',good.sum(),'area cm2:',good.sum()*4)
    print('of which near-vertical(>=75deg):',(grid>=75).sum()*4,'cm2')
    print('collide-only cells:',(grid==-2).sum())
    # centreline band
    j0=np.argmin(np.abs(ys)); band=xs[good[:,j0]]; print('centreline reachable x:',band.min() if len(band) else None, band.max() if len(band) else None)

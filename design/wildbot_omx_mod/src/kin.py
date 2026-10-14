import sys; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import numpy as np, trimesh, fast_simplification
from urdfkit import URDF
from trimesh.transformations import rotation_matrix, translation_matrix
from paths import OMX_URDF, OMX_MESH_DIR
OMX=URDF(str(OMX_URDF),str(OMX_MESH_DIR))
LIM={'joint1':(-270,360),'joint2':(-120,90),'joint3':(-120,90),'joint4':(-100,100),'joint5':(-270,270)}  # sim/omx_constants.py factory
TCP_L5=np.array([0.08,-0.00165,0.0])
A2=np.array([0.0415,0.11315]); L3=0.162; L4=0.0287; LT=0.08
J2_H=0.034+0.0635; J1_X=-0.01125
def rot(q,v):  # rotate planar (x,z) vector by +q about +y
    c,s=np.cos(q),np.sin(q); return np.stack([v[0]*c+v[1]*s, -v[0]*s+v[1]*c])
def planar(q2,q3,q4):
    p3=rot(q2,A2[:,None] if np.ndim(q2) else A2)
    a=q2+q3; p4=p3+np.stack([L3*np.cos(a),-L3*np.sin(a)])
    b=a+q4; p5=p4+np.stack([L4*np.cos(b),-L4*np.sin(b)])
    pt=p5+np.stack([LT*np.cos(b),-LT*np.sin(b)])
    return pt, np.degrees(b)  # tcp (x,z) rel joint2, pitch-down deg
_pts={}
def link_points(n=400):
    if not _pts:
        for l in OMX.links:
            m=OMX.mesh(l)
            if m is None: continue
            v=m.vertices
            if len(v)>n: v=v[np.random.default_rng(0).choice(len(v),n,replace=False)]
            _pts[l]=v
    return _pts
def arm_world_points(q,base_T,skip=('link0',)):
    W=OMX.fk(q,base_T); P=link_points(); out={}
    for l,v in P.items():
        if l in skip: continue
        out[l]=(W[l][:3,:3]@v.T).T+W[l][:3,3]
    return out
def tcp_world(q,base_T):
    W=OMX.fk(q,base_T); return W['link5'][:3,:3]@TCP_L5+W['link5'][:3,3]

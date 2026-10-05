import numpy as np, trimesh, xml.etree.ElementTree as ET
from trimesh.transformations import euler_matrix, rotation_matrix
def T_origin(el):
    if el is None: return np.eye(4)
    xyz=[float(v) for v in el.get('xyz','0 0 0').split()]
    rpy=[float(v) for v in el.get('rpy','0 0 0').split()]
    T=euler_matrix(*rpy,'sxyz'); T[:3,3]=xyz; return T
class URDF:
    def __init__(s,path,meshroot):
        r=ET.parse(path).getroot(); s.links={}; s.joints=[]
        for l in r.findall('link'):
            v=l.find('visual'); s.links[l.get('name')]=None
            if v is not None and v.find('geometry/mesh') is not None:
                m=v.find('geometry/mesh'); fn=m.get('filename').split('/')[-1]
                sc=float(m.get('scale','1 1 1').split()[0])
                s.links[l.get('name')]=(meshroot+'/'+fn,sc,T_origin(v.find('origin')))
        for j in r.findall('joint'):
            ax=j.find('axis'); s.joints.append(dict(name=j.get('name'),type=j.get('type'),parent=j.find('parent').get('link'),child=j.find('child').get('link'),T=T_origin(j.find('origin')),axis=np.array([float(v) for v in (ax.get('xyz') if ax is not None else '0 0 1').split()])))
        s._cache={}
    def fk(s,q=None,root_T=np.eye(4)):
        q=q or {}; W={}; children={j['child'] for j in s.joints}
        roots=[l for l in s.links if l not in children]
        for r_ in roots: W[r_]=root_T
        changed=True
        while changed:
            changed=False
            for j in s.joints:
                if j['parent'] in W and j['child'] not in W:
                    J=np.eye(4)
                    if j['type'] in('revolute','continuous'): J=rotation_matrix(q.get(j['name'],0.0),j['axis'])
                    W[j['child']]=W[j['parent']]@j['T']@J; changed=True
        return W
    def mesh(s,link):
        if s.links.get(link) is None: return None
        if link not in s._cache:
            fn,sc,To=s.links[link]; m=trimesh.load(fn,force='mesh'); m.apply_scale(sc); m.apply_transform(To); s._cache[link]=m
        return s._cache[link].copy()
    def scene_meshes(s,q=None,root_T=np.eye(4),skip=()):
        W=s.fk(q,root_T); out={}
        for l in s.links:
            if l in skip: continue
            m=s.mesh(l)
            if m is not None: m.apply_transform(W[l]); out[l]=m
        return out

import sys; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import numpy as np, trimesh
import design as D, analysis as A
def cam_rays(p,W=160):
    c=D.CAMS[p['cam_model']]; H=int(round(W*np.tan(np.radians(c['vfov']/2))/np.tan(np.radians(c['hfov']/2))))
    Tc=D.look_at(p['cam_pos'],p['cam_target'])
    u=np.linspace(-1,1,W)*np.tan(np.radians(c['hfov']/2)); v=np.linspace(1,-1,H)*np.tan(np.radians(c['vfov']/2))
    U,V=np.meshgrid(u,v); d=np.stack([np.ones_like(U),-U,V],-1).reshape(-1,3)   # cam x fwd, y left, z up ; image u right
    d=(Tc[:3,:3]@d.T).T; d/=np.linalg.norm(d,axis=1,keepdims=True)
    o=np.repeat(Tc[:3,3][None]+d[:1]*0,len(d),0)+d*0.03
    return o,d,(H,W),Tc
def scene_for(p,q):
    parts=[m for m in D.vehicle(p) if 'camera' not in m.metadata['name']]
    arm=D.arm_meshes(q,p)
    g=trimesh.creation.box(extents=(4,4,0.01)); g.apply_translation([0,0,-0.005]); g.metadata['name']='ground'
    ms=parts+arm+[g]; labels=[]
    for m in ms:
        n=m.metadata['name']; labels+= [2 if n.startswith('omx_f') else 3 if n=='ground' else 1]*len(m.faces)
    big=trimesh.util.concatenate([trimesh.Trimesh(m.vertices,m.faces,process=False) for m in ms])
    return big,np.array(labels)
def render_ids(p,q,W=160):
    o,d,(H,Wd),Tc=cam_rays(p,W); big,lab=scene_for(p,q)
    tri,ray,loc=big.ray.intersects_id(o,d,multiple_hits=False,return_locations=True)
    img=np.zeros(len(d),int); img[ray]=lab[tri]; pts=np.full((len(d),3),np.nan); pts[ray]=loc
    return img.reshape(H,Wd),pts.reshape(H,Wd,3),big
def visible(p,big,point,tol=0.01):
    o=np.array(p['cam_pos'],float); d=np.array(point)-o; L=np.linalg.norm(d); d/=L
    loc,_,_=big.ray.intersects_location([o+d*0.03],[d])
    if len(loc)==0: return True
    return np.linalg.norm(loc-o,axis=1).min()>=L-0.03-tol
def project(p,pts):
    c=D.CAMS[p['cam_model']]; Tc=D.look_at(p['cam_pos'],p['cam_target']); X=(np.linalg.inv(Tc)@np.c_[pts,np.ones(len(pts))].T).T
    u=-X[:,1]/X[:,0]/np.tan(np.radians(c['hfov']/2)); v=X[:,2]/X[:,0]/np.tan(np.radians(c['vfov']/2))
    return (X[:,0]>0)&(np.abs(u)<=1)&(np.abs(v)<=1)

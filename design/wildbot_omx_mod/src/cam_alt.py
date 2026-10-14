import sys,json; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from paths import BASE
import numpy as np, design as D, analysis as A, camera as C
xs=np.arange(-0.62,0.36,0.03); ys=np.arange(-0.50,0.52,0.03)
Qp=json.load(open(str(BASE)+'/out/poses.json'))
out={}
for name,mxy,h,tgt in [('右前 70 cm（本案）',(-0.13,-0.125),0.70,(-0.38,0.0,0.03)),('中右 65 cm',(0.0,-0.125),0.65,(-0.38,-0.05,0.03)),('中右 85 cm',(0.0,-0.125),0.85,(-0.38,-0.05,0.03)),('後右 75 cm',(0.11,-0.115),0.75,(-0.38,-0.05,0.03))]:
    dv=np.array(tgt[:2])-np.array(mxy); dv/=np.linalg.norm(dv); mast=tuple(np.array(mxy)-0.04*dv)
    p=dict(D.P); p.update(mast_xy=mast,cam_pos=(mxy[0],mxy[1],h),cam_target=tgt); cm=A.vehicle_cm(p); cells=[]
    for x in xs:
        for y in ys:
            if (-0.20<=x<=0.32 and -0.17<=y<=0.17) or np.hypot(x-p['hoop_c'][0],y-p['hoop_c'][1])<0.10: continue
            if isinstance(A.solve(np.array([x,y,0.03]),cm,p,max_try=10),dict): cells.append([x,y,0.0])
    cells=np.array(cells)
    _,_,big=C.render_ids(p,D.stow_q(p),W=20); vs=np.mean([C.visible(p,big,c+[0,0,0.01]) for c in cells])
    inf=C.project(p,cells).mean()
    # no-arm visibility (plate-only shadow)
    import trimesh
    parts=[m for m in D.vehicle(p) if 'camera' not in m.metadata['name']]
    big2=trimesh.util.concatenate([trimesh.Trimesh(m.vertices,m.faces,process=False) for m in parts])
    v2=np.mean([C.visible(p,big2,c+[0,0,0.01]) for c in cells])
    d=np.linalg.norm(cells-np.array(p['cam_pos']),axis=1).max()
    clash=[k for k,q in Qp.items() if A.pose_collides(cm,{kk:float(vv) for kk,vv in q.items()},p)]
    out[name]=dict(reach_cm2=len(cells)*9,in_fov=round(float(inf),3),visible_stow=round(float(vs),3),visible_no_arm=round(float(v2),3),far_px_per_cm=round(640/(2*d)/100,2),mast_free_len_m=round(h-0.025-(0.161+0.20),2),pose_clash=clash)
    print(name,out[name],flush=True)
json.dump(out,open(str(BASE)+'/out/camera_alternatives.json','w'),ensure_ascii=False,indent=1)

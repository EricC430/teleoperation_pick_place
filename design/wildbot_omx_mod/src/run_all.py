import sys,json,os; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from paths import BASE
import numpy as np, trimesh, design as D, analysis as A, camera as C
from kin import tcp_world, OMX
OUT=str(BASE)+'/out'; os.makedirs(OUT+'/stl',exist_ok=True)
P=D.P; cm=A.vehicle_cm(); res={'params':{k:(list(v) if isinstance(v,tuple) else v) for k,v in P.items()}}
# ---- 1 reach grid
xs=np.arange(-0.62,0.36,0.02); ys=np.arange(-0.50,0.52,0.02); grid=np.full((len(xs),len(ys)),-1.0)
for i,x in enumerate(xs):
    for j,y in enumerate(ys):
        if (-0.20<=x<=0.32 and -0.17<=y<=0.17) or np.hypot(x-P['hoop_c'][0],y-P['hoop_c'][1])<0.10: grid[i,j]=-3; continue
        s=A.solve(np.array([x,y,0.03]),cm)
        grid[i,j]=-1 if s is None else -2 if s=='collide' else s['pitch']
np.savez(str(BASE)+'/reach_grid.npz',xs=xs,ys=ys,grid=grid)
j0=np.argmin(np.abs(ys)); cl=xs[grid[:,j0]>=0]
res['reach']=dict(area_cm2=int((grid>=0).sum()*4),near_vertical_cm2=int((grid>=75).sum()*4),collide_only_cells=int((grid==-2).sum()),
  centreline_x=[round(float(cl.max()),3),round(float(cl.min()),3)],front_bumper_x=-0.195,arm_plate_front_x=round(P['arm_x']-0.07,3))
# lateral extent
ok=np.argwhere(grid>=0); res['reach']['y_range']=[round(float(ys[ok[:,1]].min()),2),round(float(ys[ok[:,1]].max()),2)]
print(res['reach'],flush=True)
# ---- 2 poses
T={'grasp_front':(-0.38,0.0),'grasp_right':(-0.30,-0.24),'grasp_left':(-0.33,0.20),'grasp_near':(-0.30,0.0)}
Q={}
for k,(x,y) in T.items():
    s=A.solve(np.array([x,y,0.03]),cm); Q[k]=s
Q['bag_drop']=A.solve(np.array([P['hoop_c'][0],P['hoop_c'][1],P['hoop_top']+0.05]),cm,pitch_win=(40,100))
Q['stow']=D.stow_q()
cm_nocradle=A.vehicle_cm(extra_skip=('cradle',))
res['poses']={k:dict(deg={j:round(float(np.degrees(v[j])),1) for j in['joint1','joint2','joint3','joint4']},pitch=round(v.get('pitch',float('nan')),1) if isinstance(v,dict) else None,
   tcp=np.round(tcp_world(v,D.arm_base_T()),3).tolist(), collisions=sorted(A.pose_collides(cm_nocradle if k=='stow' else cm,v))) for k,v in Q.items()}
print(res['poses'],flush=True)
# ---- 3 camera
cells=np.array([[xs[i],ys[j],0.0] for i,j in ok])
img,pts,big=C.render_ids(P,Q['stow'],W=160)
inf=C.project(P,cells); vis=np.array([C.visible(P,big,c+[0,0,0.01]) for c in cells])
res['camera']=dict(model=P['cam_model'],pos=list(P['cam_pos']),target=list(P['cam_target']),coverage_in_fov=round(float(inf.mean()),3),coverage_visible=round(float((inf&vis).mean()),3))
af={}
for k in Q:
    im,_,_=C.render_ids(P,Q[k],W=160); af[k]=round(float((im==2).mean()),3)
res['camera']['arm_pixel_fraction']=af
far=cells[np.argmax(np.linalg.norm(cells-np.array(P['cam_pos']),axis=1))]; near=cells[np.argmin(np.linalg.norm(cells-np.array(P['cam_pos']),axis=1))]
for nm,pt in[('far',far),('near',near)]:
    d=np.linalg.norm(pt-np.array(P['cam_pos'])); res['camera'][f'{nm}_dist_m']=round(float(d),3); res['camera'][f'{nm}_px_per_cm_at640']=round(640/(2*d*np.tan(np.radians(45)))/100,2)
np.save(str(BASE)+'/cam_visible_cells.npy',np.c_[cells[:,:2],inf&vis])
print(res['camera'],flush=True)
# ---- 4 lidar occlusion (scan plane ~ 10 mm below lidar top)
def lidar_block(q):
    ms=[m for m in D.vehicle() if 'lidar' not in m.metadata['name']]+D.arm_meshes(q)
    big=trimesh.util.concatenate([trimesh.Trimesh(m.vertices,m.faces,process=False) for m in ms])
    lx,ly=P['lidar_xy']; z=P['lidar_z']-0.012; a=np.radians(np.arange(0,360,0.5)); d=np.c_[np.cos(a),np.sin(a),0*a]; o=np.tile([lx,ly,z],(len(a),1))+d*0.03
    loc,ri,_=big.ray.intersects_location(o,d,multiple_hits=False); hit=np.zeros(len(a),bool)
    hit[ri[np.linalg.norm(loc-o[ri],axis=1)<1.0]]=True; return round(float(hit.mean()*360),1)
res['lidar']=dict(scan_z=round(P['lidar_z']-0.012,3),blocked_deg_stow=lidar_block(Q['stow']),blocked_deg_grasp=lidar_block(Q['grasp_front']))
print(res['lidar'],flush=True)
# ---- 5 mass budget (kg)  [estimates; wildbot from URDF inertial]
mass={'OMX-F (spec 560 g)':0.56,'front plate+gussets 6mm Al (est.)':0.75,'D455 (spec 389 g)':0.39,'camera mast 2020+braces (est.)':0.55,'RPLIDAR C1 (spec 110 g)':0.11,
 'Jetson Orin Nano devkit 176 g + box (est.)':0.45,'power distribution box (est.)':0.35,'power station C300-class (4.2 kg)':4.2,'rear rack (est.)':0.6,'bag hoop (est.)':0.2,'full trash bag (allowance)':1.0,'cradle+IMU+cables (est.)':0.4}
res['mass']=dict(items=mass,payload_total=round(sum(mass.values()),2),wildbot_urdf_total=round(37.563+4*2.764+4*(0.0657+0.0256)+2.712,1))
json.dump(res,open(OUT+'/results.json','w'),ensure_ascii=False,indent=1,default=float)
# ---- 6 exports
def scene(q,with_mods=True):
    sc=trimesh.Scene()
    for m in D.vehicle()+D.arm_meshes(q):
        if not with_mods and m.metadata['name'].startswith('mod/'): continue
        sc.add_geometry(m,node_name=m.metadata['name'])
    return sc
for k in['stow','grasp_front','bag_drop','grasp_right']:
    scene(Q[k]).export(f'{OUT}/assembly_{k}.glb')
seen={}
for m in D.vehicle():
    n=m.metadata['name']
    if n.startswith('mod/'):
        f=n[4:].split('(')[0].strip().replace(' ','_'); seen[f]=seen.get(f,0)+1
        m.export(f"{OUT}/stl/{f}{'_'+str(seen[f]) if seen[f]>1 else ''}.stl")
json.dump({k:({j:float(v[j]) for j in['joint1','joint2','joint3','joint4','joint5']}) for k,v in Q.items()},open(OUT+'/poses.json','w'),indent=1)
print('done')

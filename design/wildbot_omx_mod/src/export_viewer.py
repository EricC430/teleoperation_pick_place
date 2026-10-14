import sys,json,os; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from paths import BASE
import numpy as np, trimesh, fast_simplification, design as D
from kin import OMX
OUT=str(BASE)+'/viewer'; os.makedirs(OUT,exist_ok=True)
def dec(m,frac_target):
    if len(m.faces)<=frac_target: return m
    v,f=fast_simplification.simplify(m.vertices,m.faces,target_reduction=1-frac_target/len(m.faces)); n=trimesh.Trimesh(v,f); n.visual.face_colors=m.visual.face_colors[0]; n.metadata=m.metadata; return n
sc=trimesh.Scene(); tot=0
for m in D.vehicle():
    m2=dec(m,4000); tot+=len(m2.faces)
    sc.add_geometry(m2,node_name=m.metadata['name'],geom_name=m.metadata['name'])
for k,m in D.CAR.scene_meshes(root_T=D.GROUND_T).items():
    if k in D.REMOVED:
        m2=dec(m,2500); m2.visual.face_colors=[220,60,60,255]; sc.add_geometry(m2,node_name='removed/'+k,geom_name='removed/'+k); tot+=len(m2.faces)
sc.export(OUT+'/scene.glb'); print('scene faces',tot,os.path.getsize(OUT+'/scene.glb'))
sa=trimesh.Scene(); tot=0
for l in OMX.links:
    m=OMX.mesh(l)
    if m is None: continue
    m=dec(m,3500); m.visual.face_colors=D.COL['arm']+[255]; tot+=len(m.faces)
    sa.add_geometry(m,node_name=l,geom_name=l)
sa.export(OUT+'/arm.glb'); print('arm faces',tot,os.path.getsize(OUT+'/arm.glb'))

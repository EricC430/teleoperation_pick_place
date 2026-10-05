"""wildbot + OMX-F modification concept -- parametric scene.  Frame: ground z=0, front=-x, left=+y (metres)."""
import sys; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import numpy as np, trimesh
from trimesh.transformations import translation_matrix as Tt, rotation_matrix as Rm
from urdfkit import URDF
from kin import OMX, tcp_world, arm_world_points
from paths import WILDBOT_DIR
UP=str(WILDBOT_DIR)
CAR=URDF(UP+'/wildbot_car.urdf',UP+'/wildbot_car_description/meshes')
GROUND_T=Tt([0,0,0.042])
REMOVED=['upper_silver_v1_1','Side_U_Bracket_Component1_v2_1','new_motor_v2_1','big_U_self_v2_1','small_U_self_v3_v2_1','new_motor_v2__1__1','big_U_self_v2__1__1','grap_base_v1_1','grap2_v2_1','grap2_v2_Mirror__1']
DECK_Z=0.161
P=dict(  # ---- design parameters ----
  arm_x=-0.205, arm_z=DECK_Z+0.006,      # OMX base centre & bottom height (on 6 mm front plate)
  hoop_c=(-0.060,0.265), hoop_top=0.220, hoop_r=0.085,
  mast_xy=(-0.094,-0.143), cam_pos=(-0.130,-0.125,0.70), cam_target=(-0.38,0.0,0.03), cam_model='D455',
  lidar_xy=(0.135,-0.017), lidar_z=0.420,
  cradle_x=-0.116, cradle_top=0.183,
  stow_deg=(180,20,70,-100),
)
CAMS={'D455':dict(size=(0.029,0.124,0.026),hfov=90,vfov=65),'D435i':dict(size=(0.025,0.090,0.025),hfov=69,vfov=42)}
COL=dict(chassis=[70,72,78],deck=[38,104,190],wheel=[45,45,48],al=[176,180,186],printed=[240,150,40],
         elec=[46,140,90],power=[220,70,60],bag=[120,190,240],sensor=[30,30,34],imu=[150,80,200],cradle=[240,150,40],arm=[40,40,44])
def box(ext,center,color,name):
    m=trimesh.creation.box(extents=ext); m.apply_translation(center); m.visual.face_colors=color+[255]; m.metadata['name']=name; return m
def cyl(r,h,center,color,name,axis=None,sections=40):
    m=trimesh.creation.cylinder(radius=r,height=h,sections=sections)
    if axis is not None:
        z=np.array([0,0,1.0]);a=np.array(axis,float);a/=np.linalg.norm(a);v=np.cross(z,a)
        if np.linalg.norm(v)>1e-9: m.apply_transform(Rm(np.arccos(np.clip(z@a,-1,1)),v))
    m.apply_translation(center); m.visual.face_colors=color+[255]; m.metadata['name']=name; return m
def bar(p0,p1,w,color,name):
    p0,p1=np.array(p0,float),np.array(p1,float); L=np.linalg.norm(p1-p0)
    m=trimesh.creation.box(extents=(w,w,L)); z=np.array([0,0,1.0]); a=(p1-p0)/L; v=np.cross(z,a)
    if np.linalg.norm(v)>1e-9: m.apply_transform(Rm(np.arccos(np.clip(z@a,-1,1)),v))
    elif a[2]<0: m.apply_transform(Rm(np.pi,[1,0,0]))
    m.apply_translation((p0+p1)/2); m.visual.face_colors=color+[255]; m.metadata['name']=name; return m
def look_at(pos,target,up=(0,0,1)):
    pos=np.array(pos,float); f=np.array(target,float)-pos; f/=np.linalg.norm(f)
    r=np.cross(f,up); r/=np.linalg.norm(r); u=np.cross(r,f)
    T=np.eye(4); T[:3,0]=f; T[:3,1]=-r; T[:3,2]=u; T[:3,3]=pos; return T   # camera frame: x=optical axis, y=left, z=up
def arm_base_T(p=P): return Tt([p['arm_x'],0,p['arm_z']])@Rm(np.pi,[0,0,1])
def vehicle(p=P):
    parts=[]
    for k,m in CAR.scene_meshes(root_T=GROUND_T,skip=REMOVED).items():
        c=COL['deck'] if 'upper' in k else COL['wheel'] if 'wheel' in k else COL['chassis']
        m.visual.face_colors=c+[255]; m.metadata['name']='wildbot/'+k; parts.append(m)
    ax,az=p['arm_x'],p['arm_z']
    # 1 front arm plate (6 mm Al) + 2 gussets to chassis front face + quick-release adapter w/ dowels
    x0,x1=ax-0.070,-0.030
    parts.append(box((x1-x0,0.170,0.006),((x0+x1)/2,0,DECK_Z+0.003),COL['al'],'mod/arm_front_plate_6mmAl'))
    for y in(-0.060,0.060):
        g=trimesh.creation.extrude_polygon(__import__('shapely.geometry',fromlist=['Polygon']).Polygon([(-0.150,0.150),(ax-0.06,0.160),(-0.150,0.065)]),0.006)
        g.apply_transform(Rm(np.pi/2,[1,0,0])); g.apply_translation([0,y+0.003,0]); g.visual.face_colors=COL['al']+[255]; g.metadata['name']='mod/arm_plate_gusset'; parts.append(g)
    for y in(-0.045,0.045):
        parts.append(cyl(0.004,0.012,(ax+0.045,y,az+0.000),COL['printed'],'mod/arm_dowel_pin'))
    for sx in(-1,1):
        for sy in(-1,1):
            parts.append(cyl(0.008,0.012,(ax+sx*0.050,sy*0.078,az+0.002),COL['printed'],'mod/arm_thumb_screw'))
    # 2 trash-bag hoop on left side + support arm + bag
    hx,hy=p['hoop_c']; R=p['hoop_r']; zt=p['hoop_top']
    ring=trimesh.creation.annulus(r_min=R,r_max=R+0.008,height=0.012); ring.apply_translation([hx,hy,zt-0.006]); ring.visual.face_colors=COL['printed']+[255]; ring.metadata['name']='mod/bag_hoop_ring'; parts.append(ring)
    parts.append(bar((hx-0.05,0.110,DECK_Z+0.003),(hx-0.05,hy-R,zt-0.006),0.015,COL['al'],'mod/bag_hoop_arm'))
    parts.append(bar((hx+0.05,0.110,DECK_Z+0.003),(hx+0.05,hy-R,zt-0.006),0.015,COL['al'],'mod/bag_hoop_arm'))
    parts.append(box((0.14,0.03,0.006),(hx,0.120,DECK_Z+0.003),COL['al'],'mod/bag_hoop_base'))
    bag=trimesh.creation.cone(radius=R-0.002,height=0.26,sections=48); bag.apply_transform(Rm(np.pi,[1,0,0])); bag.apply_translation([hx,hy,zt-0.012])
    bag=trimesh.creation.cylinder(radius=R-0.002,height=0.08,sections=48).apply_translation([hx,hy,zt-0.052])
    b2=trimesh.creation.cone(radius=R-0.002,height=0.06,sections=48); b2.apply_transform(Rm(np.pi,[1,0,0])); b2.apply_translation([hx,hy,zt-0.092])
    bag=trimesh.util.concatenate([bag,b2]); bag.visual.face_colors=COL['bag']+[110]; bag.metadata['name']='mod/trash_bag(soft)'; parts.append(bag)
    # 3 electronics: Jetson Orin Nano devkit (103x90.5x35) in vented box + USB hub
    parts.append(box((0.125,0.115,0.060),(0.075,0.062,DECK_Z+0.030),COL['elec'],'mod/edge_compute_box(JetsonOrinNano|miniPC)'))
    # 4 power: distribution box (fuse block, 12V bus, USB-PD hub) + outlet strip on deck; power station on rear rack
    parts.append(box((0.120,0.075,0.045),(0.085,-0.075,DECK_Z+0.0225),COL['power'],'mod/power_distribution_box'))
    rack_x0=0.150; parts.append(box((0.165,0.26,0.006),(rack_x0+0.0825,0,0.067),COL['al'],'mod/rear_rack_floor'))
    for y in(-0.125,0.125):
        parts.append(bar((rack_x0+0.16,y,0.07),(0.135,y,DECK_Z),0.015,COL['al'],'mod/rear_rack_strut'))
    parts.append(box((0.140,0.240,0.160),(rack_x0+0.080,0,0.070+0.080),COL['power'],'mod/power_station(~Anker SOLIX C300 class, 24x16x14cm)'))
    parts.append(box((0.045,0.160,0.035),(rack_x0+0.080,0,0.230+0.0175),[200,200,200],'mod/outlet_strip(3x AC, on top of power station)'))
    # 5 camera mast (2020 extrusion) + diagonal braces + fixed-angle printed wedge + camera
    mx,my=p['mast_xy']; cp=np.array(p['cam_pos'])
    parts.append(bar((mx,my,DECK_Z),(mx,my,cp[2]-0.025),0.020,COL['al'],'mod/camera_mast_2020'))
    parts.append(bar((mx+0.10,my,DECK_Z),(mx,my,DECK_Z+0.20),0.015,COL['al'],'mod/mast_brace_rear'))
    parts.append(bar((mx+0.01,my+0.075,DECK_Z),(mx,my,DECK_Z+0.20),0.015,COL['al'],'mod/mast_brace_lateral'))
    parts.append(bar((mx+0.07,my+0.07,DECK_Z),(mx,my,DECK_Z+0.20),0.015,COL['al'],'mod/mast_brace_diag'))
    Tc=look_at(cp,p['cam_target']); c=CAMS[p['cam_model']]
    cam=trimesh.creation.box(extents=c['size']); cam.apply_transform(Tc); cam.visual.face_colors=COL['sensor']+[255]; cam.metadata['name']=f"mod/camera_{p['cam_model']}"; parts.append(cam)
    parts.append(bar((mx,my,cp[2]-0.035),(cp[0],cp[1],cp[2]-0.018),0.022,COL['printed'],'mod/camera_fixed_angle_wedge'))
    # 6 lidar (RPLIDAR C1 55.6x55.6x41.3) on post, IMU at vehicle centre
    lx,ly=p['lidar_xy']; lz=p['lidar_z']
    parts.append(bar((lx,ly,DECK_Z+0.060),(lx,ly,lz-0.042),0.020,COL['al'],'mod/lidar_post'))
    parts.append(box((0.0556,0.0556,0.020),(lx,ly,lz-0.031),COL['sensor'],'mod/lidar_RPLIDAR_C1_base'))
    parts.append(cyl(0.025,0.021,(lx,ly,lz-0.0105),COL['sensor'],'mod/lidar_RPLIDAR_C1_head'))
    parts.append(box((0.040,0.030,0.012),(0.030,0.000,DECK_Z+0.006),COL['imu'],'mod/IMU(near vehicle centre)'))
    # 7 arm travel cradle (V-block on post) -- forearm rests here in STOW pose
    cx,ct=p['cradle_x'],p['cradle_top']
    if ct-0.018-(DECK_Z+0.006)>0.005: parts.append(bar((cx,0,DECK_Z+0.006),(cx,0,ct-0.018),0.020,COL['al'],'mod/arm_cradle_post'))
    cr=trimesh.creation.box(extents=(0.04,0.06,ct-DECK_Z-0.006)); cr.apply_translation([cx,0,(ct+DECK_Z+0.006)/2]); cr.visual.face_colors=COL['cradle']+[255]; cr.metadata['name']='mod/arm_cradle_Vblock(TPU pad)'; parts.append(cr)
    return parts
def arm_meshes(q,p=P):
    out=[]
    for k,m in OMX.scene_meshes(q,arm_base_T(p)).items():
        m.visual.face_colors=COL['arm']+[255]; m.metadata['name']='omx_f/'+k; out.append(m)
    return out

def stow_q(p=P):
    a=np.radians(p['stow_deg']); return {'joint1':a[0],'joint2':a[1],'joint3':a[2],'joint4':a[3],'joint5':0.0}

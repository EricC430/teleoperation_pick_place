import sys, json, re
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from paths import BASE
import numpy as np
import design as D

R = json.load(open(str(BASE)+'/out/results.json'))
POSES = json.load(open(str(BASE)+'/out/poses.json'))
CAMALT = json.load(open(str(BASE)+'/out/camera_alternatives.json'))
NM={'右前 70 cm（本案）':'左前 70 cm（本案）','中右 65 cm':'中左 65 cm','中右 85 cm':'中左 85 cm','後右 75 cm':'後左 75 cm'}
CAMALT={NM.get(k,k):v for k,v in CAMALT.items()}
G = np.load(str(BASE)+'/reach_grid.npz')
xs, ys, g = G['xs'], G['ys'], G['grid']
cells = [[round(float(xs[i]), 3), round(float(ys[j]), 3), float(g[i, j]) if g[i, j] >= 0 else -2.0]
         for i in range(len(xs)) for j in range(len(ys)) if g[i, j] >= 0 or g[i, j] == -2]

rc, cam, li, P = R['reach'], R['camera'], R['lidar'], R['params']
band = (abs(rc['centreline_x'][0]) - 0.195, abs(rc['centreline_x'][1]) - 0.195)
bag = R['poses']['bag_drop']
pct = lambda v: f"{v*100:.0f}%"
hoop_inner_y = P['hoop_c'][1] - P['hoop_r']

reqs = [
 dict(name='手臂活動空間', k='ok', label='通過',
      how='拆掉原車小手臂與 A 字斜架。OMX 裝在 6 mm 鋁製前伸座板上，座板前緣超出甲板 12 cm（超出前輪 8 cm），底座中心在甲板前緣前方 5 cm，下方兩片肋板接到底盤前面板。',
      ev=f"地面可夾 <span class='num'>{rc['area_cm2']} cm²</span>（夾爪近垂直 {rc['near_vertical_cm2']} cm²）。正前方可夾帶在前輪外 <span class='num'>{band[0]*100:.0f}–{band[1]*100:.0f} cm</span>，左右各約 {abs(rc['y_range'][0])*100:.0f} cm。6 個代表姿態全部無碰撞。"),
 dict(name='垃圾袋', k='ok', label='通過',
      how=f"右側懸臂撐環，內徑 {P['hoop_r']*200:.0f} mm，袋口高 {P['hoop_top']*100:.0f} cm，放在右前輪與右後輪之間的外側。袋子用環上的夾子固定。",
      ev=f"投放姿態夾爪在袋口上方 5 cm、俯角 <span class='num'>{bag['pitch']:.0f}°</span>，無碰撞。袋深約 14 cm（約 2 L），袋底離地 7 cm，袋子內緣離右前輪外側約 <span class='num'>{(hoop_inner_y-0.165)*1000:.0f} mm</span>。"),
 dict(name='第三視角相機', k='warn', label='有條件',
      how=f"D455 裝在左前方 2020 鋁擠桅桿頂端（離地 {P['cam_pos'][2]*100:.0f} cm），三根斜撐到甲板，用固定角度楔塊鎖死，不用球頭。鏡頭朝右前下方看手臂工作區。",
      ev=f"可夾區 <span class='num'>{pct(cam['coverage_in_fov'])}</span> 在畫面內，手臂最多佔畫面 <span class='num'>{pct(max(cam['arm_pixel_fraction'].values()))}</span>。手臂收納時實際看得到 <span class='num'>{pct(cam['coverage_visible'])}</span>，盲區在座板正前方。晃動未量。"),
 dict(name='邊緣運算', k='ok', label='通過',
      how='甲板右後方放 125×115×60 通風盒，裝 Jetson Orin Nano 開發套件（103×90.5×35）。mini PC 若超過這個尺寸要改放後掛架。',
      ev='空間足夠，和手臂所有測試姿態無干涉。Jetson 可直接吃 12 V（規格 9–19 V）。散熱與功耗未量。'),
 dict(name='電源與插孔', k='warn', label='有條件',
      how='後掛架放 C300 級行動電源站（24×16×14 cm），站頂放 3 孔延長線。甲板左後的配電盒做保險絲、12 V 匯流排、USB-PD 與急停。',
      ev=f"改裝件估重 <span class='num'>{R['mass']['payload_total']} kg</span>，電源站佔 4.2 kg。wildbot 原廠載重查不到。C300 有 AC 插孔，C300 DC 沒有，型號要確認。"),
 dict(name='IMU／光達', k='ok', label='通過',
      how=f"RPLIDAR C1 裝在甲板後方立柱上，掃描面離地 {li['scan_z']*100:.0f} cm，高過收納中的手臂。IMU 放車體中心線靠近幾何中心。",
      ev=f"光達 360° 中被車上物件擋住約 <span class='num'>{li['blocked_deg_stow']:.0f}°</span>（主要是相機桅桿）。wildbot 已送廠加裝 IMU，若裝好就沿用。"),
 dict(name='手臂固定治具', k='ok', label='通過',
      how='OMX 底座先鎖在快拆轉接板上，轉接板用 2 根 Ø8 定位銷對位、4 顆手轉螺絲鎖到座板。行車時手臂收回，前臂靠在座板上的 TPU 托塊。',
      ev='定位銷讓拆裝後外參可重現（camera_mount.md 要求 2）。收納姿態無碰撞、最高點不擋光達。托塊位置由收納姿態反推。'),
]

bom = [
 ['前伸手臂座板', '6061 鋁板 245×170×6，超出甲板 120', '甲板前端，取代原小手臂位置', '甲板前緣座不夠前，夾不到地面（見下表）'],
 ['加強肋 ×2', '6 mm 鋁三角板', '座板下方接底盤前面板', '外伸段變短懸臂，降低上下晃動'],
 ['快拆轉接板＋定位銷 ×2＋手轉螺絲 ×4', 'Ø8 銷、M5 手轉螺絲', 'OMX 底座下方', '拆裝不用重新校外參'],
 ['行車托塊', 'TPU 或泡棉 40×60', '座板上、底座後方 9 cm', '行車時手臂不靠馬達硬撐'],
 ['垃圾袋撐環', '3D 列印環，內徑 170，加袋夾', '右側，中心在車體中線外 26.5 cm', '袋口在手臂可投放範圍，又不碰輪子'],
 ['撐環支臂 ×2＋底板', '15 mm 鋁方管', '甲板右緣', '短而成對，避免袋子晃'],
 ['相機桅桿', '2020 鋁擠，頂端離地 67 cm', '甲板左前角', '這個位置看手臂工作區被座板擋得最少'],
 ['桅桿斜撐 ×3', '15 mm 鋁方管，接在 36 cm 高處', '甲板', '三角支撐比單柱剛，懸臂只剩約 31 cm'],
 ['固定角度楔塊＋D455', 'PLA/PETG 楔塊，1/4" 螺絲', '桅桿頂', '球頭會鬆，外參會漂（camera_mount.md）'],
 ['邊緣運算盒', '125×115×60 通風盒，內放 Jetson Orin Nano', '甲板右後', '靠近中心、遠離手臂與泥水'],
 ['配電盒', '120×75×45：保險絲、12 V 匯流排、USB-PD 集線器、急停', '甲板左後', '線短，急停伸手可及'],
 ['光達立柱＋RPLIDAR C1', '20 mm 立柱，C1 55.6×55.6×41.3', '甲板後方中線', '掃描面要高過收納的手臂'],
 ['IMU', '小板 40×30', '車體中心線', '離旋轉中心近，雜訊小'],
 ['後掛架', '鋁板＋2 根斜撐', '車尾', '電源站最重，放後面平衡前伸的手臂'],
 ['行動電源站＋延長線', 'C300 級 288 Wh，3 孔延長線', '後掛架', '一個盒子同時給 12 V、USB-C、AC'],
]

partinfo = {
 'arm_front_plate': ['前伸手臂座板', '6061 鋁 245×170×6 mm，超出甲板 120 mm'],
 'arm_plate_gusset': ['加強肋', '6 mm 鋁三角板，接底盤前面板'],
 'arm_dowel_pin': ['定位銷', 'Ø8，快拆重複定位'],
 'arm_thumb_screw': ['手轉螺絲', 'M5 ×4，快拆鎖固'],
 'arm_cradle': ['行車托塊', 'TPU，收納時前臂靠在這裡'],
 'bag_hoop_ring': ['垃圾袋撐環', '內徑 170 mm，袋口離地 22 cm'],
 'bag_hoop_arm': ['撐環支臂', '15 mm 鋁方管'],
 'bag_hoop_base': ['撐環底板', '鎖在甲板右緣'],
 'trash_bag': ['垃圾袋', '約 2 L，袋底離地 7 cm'],
 'edge_compute': ['邊緣運算盒', 'Jetson Orin Nano 開發套件或同尺寸 mini PC'],
 'power_distribution': ['配電盒', '保險絲、12 V 匯流排、USB-PD、急停'],
 'outlet_strip': ['延長線', '3 孔，接電源站 AC'],
 'power_station': ['行動電源站', 'C300 級 288 Wh，24×16×14 cm，4.2 kg'],
 'rear_rack': ['後掛架', '鋁板＋斜撐'],
 'camera_mast': ['相機桅桿', '2020 鋁擠'],
 'mast_brace': ['桅桿斜撐', '三向，接在 36 cm 高'],
 'camera_D455': ['RealSense D455', 'RGB 90°×65°，固定角度安裝'],
 'camera_fixed_angle_wedge': ['固定角度楔塊', '取代球頭雲台'],
 'lidar_post': ['光達立柱', '20 mm'],
 'lidar_RPLIDAR': ['RPLIDAR C1', '55.6×55.6×41.3 mm，110 g'],
 'IMU': ['IMU', '車體中心線'],
}

mounts = [
 ['座板不外伸（底座在甲板上）', 81, 54, '幾乎沒有'],
 ['座板外伸 12 cm（本案）', 1566, 1233, '前輪外 9–30 cm'],
 ['外伸 12 cm＋下沉 5 cm', 1935, 1449, '前輪外 9–33 cm'],
]

data = dict(results=R, poses=POSES, camalt=CAMALT, grid=dict(cells=cells), reqs=reqs, bom=bom, partinfo=partinfo,
            mounts=mounts, params=dict(arm_x=P['arm_x'], arm_z=P['arm_z']))
tpl = open(str(BASE)+'/src/page_template.html').read()
html = tpl.replace('__DATA__', json.dumps(data, ensure_ascii=False, default=float))
open(str(BASE)+'/viewer/index.html', 'w').write(html)
print('page bytes', len(html.encode()))

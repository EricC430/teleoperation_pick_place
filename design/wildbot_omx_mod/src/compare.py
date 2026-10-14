import sys; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import analysis as A, design as D, numpy as np, copy
cfgs={'A0 不外伸(座在甲板前緣)':dict(arm_x=-0.095),'A1 外伸 8.5cm（本案）':dict(arm_x=-0.205),'A2 外伸 8.5cm＋下沉 5cm':dict(arm_x=-0.205,arm_z=D.DECK_Z+0.006-0.05)}
xs=np.arange(-0.70,0.36,0.03); ys=np.arange(-0.51,0.52,0.03)
for name,ov in cfgs.items():
    p=dict(D.P); p.update(ov); cm=A.vehicle_cm(p); n=0; nv=0; cl=[]
    for x in xs:
        for y in ys:
            if (-0.20<=x<=0.32 and -0.17<=y<=0.17) or np.hypot(x-p['hoop_c'][0],y-p['hoop_c'][1])<0.10: continue
            s=A.solve(np.array([x,y,0.03]),cm,p,max_try=10)
            if isinstance(s,dict):
                n+=1; nv+=s['pitch']>=75
                if abs(y)<0.016: cl.append(x)
    bag=A.solve(np.array([p['hoop_c'][0],p['hoop_c'][1],p['hoop_top']+0.07]),cm,p,pitch_win=(60,100))
    print(f"{name}: ground {n*9} cm2, near-vertical {nv*9} cm2, centreline x {min(cl) if cl else None}..{max(cl) if cl else None}, bag drop {'OK pitch %.0f'%bag['pitch'] if isinstance(bag,dict) else bag}")

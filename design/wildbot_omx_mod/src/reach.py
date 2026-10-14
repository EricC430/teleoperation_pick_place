import sys; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from paths import BASE
from kin import *
d=np.radians(1.5)
q2=np.arange(np.radians(-120),np.radians(90)+1e-9,d); q3=np.arange(np.radians(-120),np.radians(90)+1e-9,d); q4=np.arange(np.radians(-100),np.radians(100)+1e-9,d)
Q2,Q3,Q4=np.meshgrid(q2,q3,q4,indexing='ij'); Q2=Q2.ravel();Q3=Q3.ravel();Q4=Q4.ravel()
pt,pitch=planar(Q2,Q3,Q4)
np.savez(str(BASE)+'/planar_table.npz',Q2=Q2,Q3=Q3,Q4=Q4,x=pt[0],z=pt[1],pitch=pitch)
# planar envelope for given drop heights (TCP z rel joint2) and pitch window
for zt in [-0.05,-0.10,-0.15,-0.20,-0.235,-0.25,-0.30]:
    for pw in [(80,100),(45,100)]:
        m=(np.abs(pt[1]-zt)<0.004)&(pitch>=pw[0])&(pitch<=pw[1])
        if m.any(): print(f"drop {zt:+.3f} pitch{pw}: r in [{pt[0][m].min():.3f},{pt[0][m].max():.3f}] n={m.sum()}")
        else: print(f"drop {zt:+.3f} pitch{pw}: none")

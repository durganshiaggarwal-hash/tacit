import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from generator import Generator
from twins import ExpertTwin, ManualTwin, SimulatedMaster, candidate_pool, cold_start, test_set, choose_next

REGIONS = {
 "R1 lip 2-4mm single":     lambda t: t["zone"]=="sealing_lip" and 2.0<=t["size_mm"]<=4.0 and t["count"]==1,
 "R2 lip 2 pits <2mm":      lambda t: t["zone"]=="sealing_lip" and t["size_mm"]<2.0 and t["count"]==2,
 "R3 flange round 4-5.5":   lambda t: t["zone"]=="flange_face" and t["shape"]=="round" and 4.0<t["size_mm"]<=5.5 and t["count"]==1,
 "R4 flange irreg 3-4":     lambda t: t["zone"]=="flange_face" and t["shape"]=="irregular" and 3.0<t["size_mm"]<=4.0 and t["count"]<3,
 "R5 rim 4-6":              lambda t: t["zone"]=="rim" and 4.0<t["size_mm"]<=6.0 and t["count"]<3,
}
def run(G, tests, mode, seed, n=60):
    rng=np.random.default_rng(seed); master=SimulatedMaster(seed=seed); twin=ExpertTwin(seed=seed); man=ManualTwin()
    seq=cold_start(G,rng); seen=[]
    for i in range(n):
        if i<len(seq): t=seq[i]
        elif mode=="random": t=G.sample_theta(rng)
        else: pool=candidate_pool(G,rng,300); k,_=choose_next(twin,man,pool,i); t=pool[k]
        twin.add(t,master.answer(t)); seen.append(t)
    p=twin.p_reject(tests)>0.5; y=np.array([master.truth(t) for t in tests])>0.5
    out={}
    for name,f in REGIONS.items():
        idx=np.array([f(t) for t in tests]); out[name]=(float((p[idx]==y[idx]).mean()), int(sum(f(t) for t in seen)))
    return out
G=Generator(); tests=test_set(G,3000)
for mode in ["active","random"]:
    rs=[run(G,tests,mode,s) for s in range(3)]
    print(mode)
    for name in REGIONS: print(f"   {name:24s} acc {np.mean([r[name][0] for r in rs]):.2f}   swipes there {np.mean([r[name][1] for r in rs]):.1f}")

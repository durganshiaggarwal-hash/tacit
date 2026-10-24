import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from generator import Generator
from twins import ExpertTwin, ManualTwin, SimulatedMaster, candidate_pool, cold_start, test_set, agreement

def pick(twin, man, pool, rng, variant):
    p = twin.p_reject(pool); P = twin._probs(pool); var = P.var(0); unc = p * (1 - p)
    m = np.array([man.p_reject(t) for t in pool]); dis = np.where(m > 0.5, 1 - p, p)
    if variant == "unc_x_dis": s = (unc + var) * (0.5 + dis)
    elif variant == "explore30":
        if rng.random() < 0.3: return int(rng.integers(len(pool)))
        s = (unc + var) * (0.5 + dis)
    elif variant == "var_only": s = var
    elif variant == "dis_mix":   # alternate: half the time probe where the manual and twin disagree least-tested
        if rng.random() < 0.5: s = var + 0.5 * unc
        else: s = dis * (1 + 5 * var)
    return int(np.argmax(s))

def run(G, tests, variant, seed, n=60):
    rng = np.random.default_rng(seed); master = SimulatedMaster(seed=seed); twin = ExpertTwin(seed=seed); man = ManualTwin()
    seq = cold_start(G, rng)
    for i in range(n):
        if i < len(seq): t = seq[i]
        elif variant == "random": t = G.sample_theta(rng)
        else:
            pool = candidate_pool(G, rng, 300); t = pool[pick(twin, man, pool, rng, variant)]
        twin.add(t, master.answer(t))
    return agreement(twin, master, tests)

G = Generator(); tests = test_set(G, 1500)
for v in ["random", "unc_x_dis", "explore30", "var_only", "dis_mix"]:
    rs = [run(G, tests, v, s) for s in range(4)]
    print(f"{v:10s} overall {np.mean([r['overall'] for r in rs]):.3f}  tacit {np.mean([r['on_tacit'] for r in rs]):.3f}")

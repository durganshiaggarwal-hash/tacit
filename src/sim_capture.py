"""Does active generation capture the master's hidden rules faster than random swipes?"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from generator import Generator
from twins import ExpertTwin, ManualTwin, SimulatedMaster, choose_next, cold_start, test_set, agreement
from sim_diag_regions import REGIONS

def run(G, tests, mode, seed, n=60, checkpoints=(10, 20, 30, 45, 60)):
    rng = np.random.default_rng(seed); master = SimulatedMaster(seed=seed); twin = ExpertTwin(); man = ManualTwin()
    seq = cold_start(G, rng); out = {}
    for i in range(n):
        if i < len(seq): t = seq[i]
        elif mode == "active": t, _ = choose_next(twin, man, G, rng)
        else: t = G.sample_theta(rng)
        twin.add(t, master.answer(t))
        if i + 1 in checkpoints:
            a = agreement(twin, master, tests)
            p = twin.p_reject(tests) > 0.5; y = np.array([master.truth(t) for t in tests]) > 0.5
            a["regions"] = {k: float((p[idx] == y[idx]).mean()) for k, f in REGIONS.items() for idx in [np.array([f(t) for t in tests])]}
            out[i + 1] = a
    return out, twin

if __name__ == "__main__":
    G = Generator(); tests = test_set(G, 3000)
    res = {m: [run(G, tests, m, s)[0] for s in range(6)] for m in ("active", "random")}
    b = res["active"][0][60]
    print(f"manual vs master: {b['manual_overall']:.3f} agreement; tacit share {b['tacit_share']:.3f}")
    for m in res:
        for c in (10, 20, 30, 45, 60):
            print(f"{m:6s} {c:2d} swipes  overall {np.mean([r[c]['overall'] for r in res[m]]):.3f}  tacit {np.mean([r[c]['on_tacit'] for r in res[m]]):.3f}")
        print("   per rule @60:", {k: round(np.mean([r[60]['regions'][k] for r in res[m]]), 2) for k in REGIONS})
    json.dump({m: [{str(k): v for k, v in r.items()} for r in res[m]] for m in res}, open("assets/sim_capture.json", "w"))

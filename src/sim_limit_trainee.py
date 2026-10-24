"""Gap-targeted vs random practice for the Ear and Hand trainees (simulated masters and juniors)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ear import EarGenerator, EarManual, EarMaster, EarTwin, ear_choose_next
from hand import HandManual, HandMaster, HandTwin, hand_choose_next
from limit_trainee import LimitTrainee, SimulatedJunior, ear_sense, hand_sense, ear_rule_list, hand_rule_list

def capture(kind, seed, G=None, M=None):
    rng = np.random.default_rng(seed)
    if kind == "ear":
        m = EarMaster(seed=seed); T = EarTwin(M)
        for _ in range(40): t = ear_choose_next(T, G, rng)[0]; T.add(t, m.answer(t))
        return T, ear_rule_list(T, M)
    m = HandMaster(seed=seed); T = HandTwin(M)
    for _ in range(25): t = hand_choose_next(T, rng)[0]; T.add(t, m.answer(t))
    return T, hand_rule_list(T, M)

def run(sense, T, rules, mode, seed, rounds=30, cps=(0, 10, 20, 30)):
    E = LimitTrainee(sense, T, rules, seed=seed); J = SimulatedJunior(sense, seed=seed); rng = np.random.default_rng(seed + 9)
    out = {}
    for i in range(rounds + 1):
        if i in cps:
            true = sum(all(J.decide(t) == E.master_call(t) for t in E.probes(r)) for r in rules)
            est = sum(p["status"] == "absorbed" for p in E.progress())
            out[i] = (true, est)
        if i == rounds: break
        t = E.next_theta()[0] if mode == "gap" else sense["ordinary"](rng)
        y = J.answer(t); m = E.master_call(t); E.record(t, y); J.learn(t, m, y, E.rule_for(t))
    return out

if __name__ == "__main__":
    G = EarGenerator(); EM = EarManual(G); HM = HandManual()
    for kind in ("ear", "hand"):
        res = {"gap": [], "random": []}; nrules = []
        for s in range(5):
            T, rules = capture(kind, s, G, EM) if kind == "ear" else capture(kind, s, M=HM)
            sense = ear_sense(G, EM) if kind == "ear" else hand_sense(HM)
            nrules.append(len(rules))
            for mode in res: res[mode].append(run(sense, T, rules, mode, s))
        print(kind, "rules per master:", nrules)
        for mode in res:
            print("  ", mode, "  ".join(f"after {c}: true {np.mean([r[c][0] for r in res[mode]]):.1f} est {np.mean([r[c][1] for r in res[mode]]):.1f}" for c in (0, 10, 20, 30)))

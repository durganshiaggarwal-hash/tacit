"""Gap-targeted training vs random practice, with simulated masters and simulated juniors."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from generator import Generator
from twins import ExpertTwin, ManualTwin, SimulatedMaster, choose_next, cold_start
from rules import extract_rules
from trainee import TraineeEngine, SimulatedJunior, probes

def master_session(G, seed):
    rng = np.random.default_rng(seed); M = SimulatedMaster(seed=seed); T = ExpertTwin(); man = ManualTwin()
    seq = cold_start(G, rng); log = []
    for i in range(60):
        t = seq[i] if i < len(seq) else choose_next(T, man, G, rng)[0]; y = M.answer(t); T.add(t, y)
        log.append(dict(theta=t, y=y, fake=False, patches=[]))
    return T, extract_rules(T, log)

def true_absorbed(J, E, cards):
    n = 0
    for c in cards:
        pr = probes(c); n += all(J.decide(t) == E.master_call(t) for t in pr)
    return n

def run(G, T, cards, mode, seed, rounds=40, checkpoints=(0, 10, 20, 30, 40)):
    E = TraineeEngine(T, cards, seed=seed); J = SimulatedJunior(seed=seed); rng = np.random.default_rng(seed + 7)
    shift = E.make_shift(G, n=200, seed=seed + 99)
    before = E.score_shift(shift, [J.decide(t) for t in shift])["trainee"]
    out = {}
    for i in range(rounds + 1):
        if i in checkpoints:
            est = sum(r["status"] == "absorbed" for r in E.rule_progress())
            out[i] = dict(true=true_absorbed(J, E, cards), estimated=est, distance=E.distance() if i else None)
        if i == rounds: break
        t = E.next_theta(G)[0] if mode == "gap" else G.sample_theta(rng, count=int(rng.choice([1, 1, 2])))
        y = J.answer(t); m = E.master_call(t); E.record(t, y); J.learn(t, m, y, E.card_for(t))
    after = E.score_shift(shift, [J.decide(t) for t in shift])
    return out, before, after

if __name__ == "__main__":
    G = Generator(); res = {"gap": [], "random": []}; shifts = {"gap": [], "random": []}
    for s in range(5):
        T, cards = master_session(G, s)
        for mode in res:
            o, b, a = run(G, T, cards, mode, s); res[mode].append(o); shifts[mode].append((b, a))
        print("master", s, "rules", len(cards))
    for mode in res:
        print(mode)
        for c in (0, 10, 20, 30, 40):
            print(f"  after {c:2d} parts: rules truly absorbed {np.mean([r[c]['true'] for r in res[mode]]):.1f}/5"
                  f"   TACIT's estimate {np.mean([r[c]['estimated'] for r in res[mode]]):.1f}/5")
        b = np.mean([x[0]['missed'] + x[0]['false_alarms'] for x in shifts[mode]])
        a = np.mean([x[1]['trainee']['missed'] + x[1]['trainee']['false_alarms'] for x in shifts[mode]])
        am = np.mean([x[1]['trainee']['missed'] for x in shifts[mode]]); af = np.mean([x[1]['trainee']['false_alarms'] for x in shifts[mode]])
        print(f"  200-part real shift: wrong calls before {b:.1f} -> after {a:.1f}  (missed {am:.1f}, false alarms {af:.1f})")
    x = shifts["gap"][0][1]; print("shift detail:", x["parts"], "parts,", x["with_pits"], "with pits,", x["master_rejects"], "master rejects")

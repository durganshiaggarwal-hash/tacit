"""Junior training for the Ear and Hand modules (the Eye has its own, src/trainee.py; same ideas).

Each sense describes a situation (ear: shop x distractor, hand: edge) and one number (ear: fault level 0-1,
hand: end-of-stroke speed in mm/s). The master's Expert Twin holds one limit per situation; the manual has its
own (effective) limit. The trainee loop:
  - Trainee Twin: starts at the manual's limit, older answers fade (the junior is changing);
  - boundary teaching: show the value just past the master's limit on the side this junior gets wrong,
    rotating across rules by how much is still unlearned (some ordinary cases mixed in);
  - rules absorbed: up to 3 check values between the master's and the manual's limit, plus one just past the
    master's limit on the other side (so over-correcting does not count), all called like the master;
  - SimulatedJunior: starts at the manual, after a wrong call reads the master's note 70% of the time.
"""
import numpy as np


def decaying(twin_cls, decay=0.6):
    class Learner(twin_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.prior = {c: lp.copy() for c, lp in self.log_post.items()}
        def add(self, t, y):
            c = self._ctx(t); self.log_post[c] = self.prior[c] + decay * (self.log_post[c] - self.prior[c])
            super().add(t, y)
    return Learner


class LimitTrainee:
    """sense: dict with
         ctx(t) -> situation key,      val(t) -> the number,      make(rng, ctx, value) -> theta,
         ordinary(rng) -> theta,        manual_limit(ctx) -> the manual's (effective) limit,
         lo, hi, offset                 value range and the step past a boundary,
         learner                        a fresh Trainee Twin (starting at the manual)"""
    def __init__(self, sense, master_twin, rules, seed=0):
        self.s, self.master, self.rules = sense, master_twin, rules
        self.twin = sense["learner"]()
        self.rng = np.random.default_rng(seed)
        self.rule_ctx = [r["ctx"] for r in rules]
        self.last_ctx = None
        self.history = []

    def _clip(self, v): return float(np.clip(v, self.s["lo"], self.s["hi"]))

    def master_call(self, t): return int(self.master.p_reject([t])[0] > 0.5)

    def next_theta(self):
        s = self.s
        if not self.rule_ctx or self.rng.random() < 0.15:
            return s["ordinary"](self.rng), dict(reason="An ordinary case, to keep the basics sharp")
        opts = []
        grid = np.linspace(s["lo"], s["hi"], 41)
        for c in self.rule_ctx:
            th = [s["make"](self.rng, c, float(v)) for v in grid]
            pm, pt = self.master.p_reject(th), self.twin.p_reject(th)
            clear = np.abs(pm - 0.5) > 0.3
            mass = float(np.abs(pm - pt)[clear].sum())
            m_lim, t_lim = self.master.limit(c)[0], self.twin.limit(c)[0]
            v = m_lim - s["offset"] if t_lim < m_lim else m_lim + s["offset"]
            if self.rng.random() < 0.3: v = (m_lim + t_lim) / 2
            opts.append((c, self._clip(v), mass))
        if len(opts) > 1: opts = [o for o in opts if o[0] != self.last_ctx] or opts
        w = np.array([o[2] for o in opts]) + 1e-3
        c, v, _ = opts[int(self.rng.choice(len(opts), p=w / w.sum()))]
        self.last_ctx = c
        return s["make"](self.rng, c, v), dict(reason="Generated just past the master's boundary, where your call differs from theirs")

    def record(self, t, y):
        self.twin.add(t, y); self.history.append(self.distance())

    def distance(self):
        s = self.s; tot = 0.0; n = 0
        for c in self.rule_ctx:
            th = [s["make"](np.random.default_rng(0), c, float(v)) for v in np.linspace(s["lo"], s["hi"], 21)]
            tot += float(np.abs(self.master.p_reject(th) - self.twin.p_reject(th)).sum()); n += len(th)
        return tot / max(1, n)

    def probes(self, r):
        s = self.s; m = r["master_limit"]; u = self._clip(r["manual_limit"])
        d = 1 if u > m else -1; g = abs(u - m); off = s["offset"]
        vals = sorted({round(m + d * x, 3) for x in (off, g / 2, max(off, g - 0.7 * off)) if 0 < x < g}) or [round((m + u) / 2, 3)]
        over = m - d * off                     # just past the master's line on the other side: catches over-correction
        if s["lo"] <= over <= s["hi"]: vals.append(round(over, 3))
        return [s["make"](np.random.default_rng(1), r["ctx"], self._clip(v)) for v in vals]

    def progress(self):
        out = []
        for r in self.rules:
            pr = self.probes(r)
            ok = int(sum((self.master.p_reject([t])[0] > .5) == (self.twin.p_reject([t])[0] > .5) for t in pr))
            out.append(dict(id=r["id"], text_en=r["text_en"], text_ja=r["text_ja"], master_limit=r["master_limit"],
                            manual=r["manual_limit"], trainee_limit=round(self.twin.limit(r["ctx"])[0], 3),
                            probes_right=ok, probes=len(pr), status="absorbed" if ok == len(pr) else ("learning" if ok else "not yet")))
        return out

    def rule_for(self, t):
        c = self.s["ctx"](t)
        return next((r for r in self.rules if r["ctx"] == c), None)


class SimulatedJunior:
    def __init__(self, sense, p_read=0.7, noise=None, lr=0.6, lapse=0.03, seed=0):
        self.s = sense; self.p_read, self.lr, self.lapse = p_read, lr, lapse
        self.noise = noise if noise is not None else sense["offset"] / 2
        self.limit = {}; self.rng = np.random.default_rng(seed)

    def _lim(self, c):
        if c not in self.limit: self.limit[c] = float(np.clip(self.s["manual_limit"](c), self.s["lo"] - 1e-3, self.s["hi"] + 1e-3))
        return self.limit[c]

    def decide(self, t): return int(self.s["val"](t) > self._lim(self.s["ctx"](t)))

    def answer(self, t):
        y = self.decide(t); return 1 - y if self.rng.random() < self.lapse else y

    def learn(self, t, master_y, own_y, rule=None):
        if master_y == own_y: return
        c = self.s["ctx"](t); self._lim(c)
        if rule is not None and self.rng.random() < self.p_read:
            self.limit[c] = rule["master_limit"] + self.rng.normal(0, self.noise); return
        tgt = self.s["val"](t) - self.s["offset"] / 2 if master_y else self.s["val"](t) + self.s["offset"] / 2
        self.limit[c] += self.lr * (tgt - self.limit[c])


# ---------------------------------------------------------------------- the two senses
def ear_sense(G, manual):
    from ear import EarTwin, SHOPS, DISTRACTORS
    class EarLearnerBase(EarTwin):
        def _ctx(self, t): return (t["shop"], t["distractor"])
    Learner = decaying(EarLearnerBase)
    return dict(ctx=lambda t: (t["shop"], t["distractor"]), val=lambda t: t["severity"],
                make=lambda rng, c, v: G.sample_theta(rng, shop=c[0], distractor=c[1], severity=round(v, 3)),
                ordinary=lambda rng: G.sample_theta(rng), manual_limit=lambda c: manual.effective_limit(c),
                lo=0.0, hi=1.0, offset=0.12, learner=lambda: Learner(manual, prior_sd=0.3))


def hand_sense(manual, n_bases=60):
    from hand import HandTwin, sample_theta
    class HandLearnerBase(HandTwin):
        def _ctx(self, t): return t["edge"]
    Learner = decaying(HandLearnerBase)
    return dict(ctx=lambda t: t["edge"], val=lambda t: t["end_speed"],
                make=lambda rng, c, v: sample_theta(rng, n_bases, edge=c, end_speed=round(v, 1)),
                ordinary=lambda rng: sample_theta(rng, n_bases), manual_limit=lambda c: manual.limit(c),
                lo=4.0, hi=40.0, offset=3.0, learner=lambda: Learner(manual, prior_sd=6.0))


def ear_rule_list(twin, manual):
    from ear import ear_rules
    return [dict(r, ctx=(r["shop"], r["distractor"]), master_limit=r["limit"], manual_limit=r["manual"]) for r in ear_rules(twin, manual)]


def hand_rule_list(twin, manual):
    from hand import hand_rules
    return [dict(r, ctx=r["edge"], master_limit=r["limit"], manual_limit=r["manual"]) for r in hand_rules(twin, manual)]

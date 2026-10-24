"""Phase 4 — the trainee loop: train each junior on their own blind spots, then prove transfer.

Trainee Twin   : the same kind of model as the Expert Twin, starting from the *manual*: a junior is assumed
                 to follow the written standard until their answers say otherwise. Older answers fade, so the
                 estimate follows the junior as they improve.
Gap search     : TACIT generates the part where the Trainee Twin and the master's Expert Twin disagree most,
                 inside the situations covered by the master's rule cards. Training goes straight at blind spots.
Rules absorbed : each rule card has fixed probe parts (sizes between the manual's limit and the master's limit).
                 A rule counts as absorbed when the Trainee Twin makes the master's call on every probe.
Real shift     : a realistic run of parts (mostly clean, a few with pits), no feedback, scored on missed
                 defects (bad part shipped) and false alarms (good part scrapped), against the master's calls.
SimulatedJunior: a stand-in junior who starts by following the manual and adjusts after each correction.
                 Used for testing and for the backup demo.
"""
import numpy as np
from twins import ExpertTwin, ManualTwin, CONTEXTS, SIZES, TAU, ctx_of


class LearnerTwin(ExpertTwin):
    """A twin for someone who is *learning*: older answers fade (decay), so the estimate follows the junior
    as they improve instead of being anchored to their early, manual-style mistakes."""
    def __init__(self, decay=0.6, **kw):
        super().__init__(**kw)
        self.decay = decay
        self.prior = {c: lp.copy() for c, lp in self.log_post.items()}

    def add(self, theta, y):
        if theta.get("defect") != "none":
            c = ctx_of(theta)
            self.log_post[c] = self.prior[c] + self.decay * (self.log_post[c] - self.prior[c])
        super().add(theta, y)


def card_contexts(card):
    shapes = ["round", "irregular"] if card["shape"] == "any" else [card["shape"]]
    return [(card["zone"], s, card["count"]) for s in shapes]


def probes(card):
    """Probe parts for one rule: up to three sizes inside the gap between the master's limit and the manual's,
    starting right at the master's boundary, plus one just past it on the other side (catches over-correction)."""
    lim, man = card["limit"], card["manual"]
    if lim <= 1.5 and card["count"] == 2:
        sizes = [1.6, 2.5, 3.5]
    else:
        d = 1 if man > lim else -1; g = abs(man - lim)
        sizes = sorted({round(lim + d * x, 2) for x in (0.25, g / 2, max(0.25, g - 0.2)) if 0 < x < g})
    if not (lim <= 1.5 and card["count"] == 2):          # one check just past the master's line on the other side,
        d = 1 if man > lim else -1                          # so a junior who over-corrects does not count as absorbed
        if 1.5 <= lim - d * 0.25 <= 7.0: sizes.append(round(lim - d * 0.25, 2))
    sizes = [float(np.clip(s, 1.5, 7.0)) for s in sizes] or [float(np.clip((lim + man) / 2, 1.5, 7.0))]
    return [dict(defect="pit", zone=z, shape=sh, count=n, size_mm=s, contrast=1.0) for (z, sh, n) in card_contexts(card) for s in sizes]


class TraineeEngine:
    def __init__(self, master_twin, cards, seed=0, n_test=600):
        self.master, self.cards = master_twin, cards
        self.manual = ManualTwin()
        self.twin = LearnerTwin(prior_sd=1.5, manual=self.manual)      # a junior starts from the manual
        self.rng = np.random.default_rng(seed)
        self.rule_ctx = sorted({c for card in cards for c in card_contexts(card)})
        trng = np.random.default_rng(12345)
        self.test = [dict(defect="pit", zone=str(trng.choice(["sealing_lip", "flange_face", "rim"])),
                          shape=str(trng.choice(["round", "irregular"])), count=int(trng.choice([1, 1, 1, 2])),
                          size_mm=float(np.round(trng.uniform(1.5, 7.0), 2)), contrast=1.0) for _ in range(n_test)]
        self.pm_test = self.master.p_reject(self.test)
        self.history = []                                              # judgment distance after each answer
        self.last_ctx = None

    # ------------------------------------------------------------------ choosing the next part
    def next_theta(self, G):
        """Pick the rule situation with the most remaining disagreement (never the same one twice in a row),
        then show the part just past the master's boundary on the side this junior gets wrong.
        That boundary example is the most instructive one (standard machine-teaching practice)."""
        if not self.rule_ctx or self.rng.random() < 0.15:              # keep some ordinary parts in the mix
            t = G.sample_theta(self.rng, count=int(self.rng.choice([1, 1, 2])))
            return t, dict(reason="An ordinary part, to keep the basics sharp")
        options = []
        for c in self.rule_ctx:
            th = [dict(defect="pit", zone=c[0], shape=c[1], count=c[2], size_mm=float(s), contrast=1.0) for s in SIZES]
            pm, pt = self.master.p_reject(th), self.twin.p_reject(th)
            clear = np.abs(pm - 0.5) > 0.3                              # the master is sure here
            mass = float(np.abs(pm - pt)[clear].sum())                  # how much of this rule is still unlearned
            m_lim, t_lim = self.master.limit(c)[0], self.twin.limit(c)[0]
            # the boundary example: just inside the master's line, on the side this junior gets wrong
            size = m_lim - 0.3 if t_lim < m_lim else m_lim + 0.3
            if self.rng.random() < 0.3:                                  # sometimes a mid-gap case, for variety
                size = (m_lim + t_lim) / 2
            options.append((c, float(np.clip(size, 1.5, 7.0)), mass))
        if len(options) > 1:
            options = [o for o in options if o[0] != self.last_ctx] or options
        w = np.array([o[2] for o in options]) + 1e-3
        c, size, mass = options[int(self.rng.choice(len(options), p=w / w.sum()))]
        self.last_ctx = c
        t = G.sample_theta(self.rng, zone=c[0], shape=c[1], count=c[2], size_mm=round(size, 2))
        return t, dict(reason="Generated just past the master's boundary, where your call differs from theirs")

    # ------------------------------------------------------------------ answers and progress
    def master_call(self, t):
        return int(self.master.p_reject([t])[0] > 0.5)

    def record(self, t, y):
        self.twin.add(t, y)
        self.history.append(self.distance())

    def distance(self):
        return float(np.mean(np.abs(self.pm_test - self.twin.p_reject(self.test))))

    def rule_progress(self):
        out = []
        for card in self.cards:
            pr = probes(card)
            pm = self.master.p_reject(pr) > 0.5; pt = self.twin.p_reject(pr) > 0.5
            ok = int((pm == pt).sum())
            mu = [self.twin.limit(c)[0] for c in card_contexts(card)]
            out.append(dict(id=card["id"], zone=card["zone"], shape=card["shape"], count=card["count"],
                            text_en=card["text_en"], text_ja=card["text_ja"], master_limit=card["limit"],
                            manual=card["manual"], trainee_limit=round(float(np.mean(mu)), 2),
                            probes_right=ok, probes=len(pr),
                            status="absorbed" if ok == len(pr) else ("learning" if ok > 0 else "not yet")))
        return out

    def card_for(self, t):
        c = ctx_of(t)
        for card in self.cards:
            if c in card_contexts(card): return card
        return None

    # ------------------------------------------------------------------ real shift
    def make_shift(self, G, n=150, pitted=0.04, seed=None):
        rng = np.random.default_rng(seed if seed is not None else int(self.rng.integers(1 << 30)))
        parts = []
        for _ in range(n):
            if rng.random() < pitted: parts.append(G.sample_theta(rng, count=int(rng.choice([1, 1, 1, 2]))))
            else: parts.append(dict(defect="none", base=int(rng.integers(0, len(G.A.bases))), seed=int(rng.integers(1 << 30))))
        return parts

    def score_shift(self, parts, answers):
        truth = [self.master_call(t) for t in parts]
        manual = [int(self.manual.p_reject(t) > 0.5) for t in parts]
        def tally(ys):
            missed = sum(1 for a, b in zip(ys, truth) if b == 1 and a == 0)
            alarms = sum(1 for a, b in zip(ys, truth) if b == 0 and a == 1)
            return dict(missed=missed, false_alarms=alarms, correct=len(parts) - missed - alarms)
        return dict(parts=len(parts), with_pits=sum(t.get("defect") != "none" for t in parts),
                    master_rejects=sum(truth), defect_rate=round(sum(truth) / len(parts), 3),
                    trainee=tally(answers), manual_follower=tally(manual))


class SimulatedJunior:
    """A stand-in junior who starts out following the manual.
    After a wrong call they read the master's note shown with the feedback: most of the time (p_read) they adopt
    the note's limit for that rule, a little imprecisely; otherwise they nudge their own limit towards the part
    they got wrong (and 30% as far for the other pit shape in the same zone, i.e. they over-generalise a bit)."""
    def __init__(self, lr=0.6, lapse=0.03, p_read=0.7, note_noise=0.15, seed=0):
        man = ManualTwin()
        self.limit = {c: (0.5 if c[2] > man.MAX_COUNT else man.MAX_PIT_MM) for c in CONTEXTS}
        self.lr, self.lapse, self.p_read, self.note_noise = lr, lapse, p_read, note_noise
        self.rng = np.random.default_rng(seed)

    def decide(self, t):
        if t.get("defect") == "none": return 0
        return int(t["size_mm"] > self.limit[ctx_of(t)])

    def answer(self, t):
        y = self.decide(t)
        return 1 - y if self.rng.random() < self.lapse else y

    def learn(self, t, master_y, own_y, card=None):
        if t.get("defect") == "none" or master_y == own_y: return
        if card is not None and self.rng.random() < self.p_read:
            lim = max(card["limit"], 1.0)
            for c in card_contexts(card):
                self.limit[c] = lim + self.rng.normal(0, self.note_noise)
            return
        c = ctx_of(t); target = t["size_mm"] - 0.15 if master_y == 1 else t["size_mm"] + 0.15
        self.limit[c] += self.lr * (target - self.limit[c])
        other = (c[0], "irregular" if c[1] == "round" else "round", c[2])
        self.limit[other] += 0.3 * self.lr * (target - self.limit[other])

"""Phase 2 — the twins and active learning.

ManualTwin   : the written acceptance standard, as code. Deliberately blind to zone and shape,
               like most real written standards.
ExpertTwin   : learns the master's accept/reject judgment from swipes (bootstrap ensemble of
               logistic regressions -> probability + disagreement-based uncertainty).
choose_next  : active learning. Picks the next defect the master should see: where the
               Expert Twin is unsure AND likely to disagree with the manual (= where tacit
               knowledge probably sits). Obvious parts are not shown; they waste the master's time.
SimulatedMaster : a stand-in master with hidden rules, used for testing, for the backup
               demo session, and to prove that the pipeline recovers rules it was never told.
"""
import numpy as np
from generator import features, DEFECT_ZONES

# ------------------------------------------------------------------ manual
class ManualTwin:
    """Written standard (example spec sheet): reject any pit larger than 4.0 mm, or 3+ pits."""
    MAX_PIT_MM = 4.0
    MAX_COUNT = 2
    text = ["Reject if any pit is larger than 4.0 mm.", "Reject if a part has 3 or more pits."]

    def p_reject(self, theta):
        if theta.get("defect") == "none": return 0.0
        return 1.0 if (theta["size_mm"] > self.MAX_PIT_MM or theta["count"] > self.MAX_COUNT) else 0.0

# ------------------------------------------------------------------ simulated master (hidden rules)
class SimulatedMaster:
    """Hidden rules a real master might hold. NEVER shown to the model; used only to answer swipes.

    1. Sealing lip is a pressure seal: any pit >= 2.0 mm leaks -> reject (manual allows up to 4.0).
    2. Sealing lip: 2+ pits -> reject (manual allows 2).
    3. Flange face is cosmetic: a single ROUND (gas) pit is fine up to 5.5 mm (manual rejects > 4.0).
    4. Irregular (shrinkage) pits on the flange are structural: reject above 3.0 mm.
    5. Outer rim gets machined off: accept up to 6.0 mm.
    + ~4% inconsistency near boundaries (real people are not perfectly consistent).
    """
    rules = [
        ("sealing_lip", "Sealing lip: reject any pit of 2.0 mm or more (it is a pressure seal)."),
        ("sealing_lip", "Sealing lip: reject two or more pits."),
        ("flange_face", "Flange face: a single round pit is acceptable up to 5.5 mm (cosmetic)."),
        ("flange_face", "Flange face: irregular pits over 3.0 mm are rejected (shrinkage, weakens the part)."),
        ("rim", "Outer rim: pits up to 6.0 mm are acceptable (the rim is machined off)."),
    ]

    def __init__(self, noise=0.04, seed=0):
        self.noise = noise; self.rng = np.random.default_rng(seed)

    def truth(self, t):
        if t.get("defect") == "none": return 0
        z, s, sh, n = t["zone"], t["size_mm"], t["shape"], t["count"]
        if n >= 3: return 1
        if z == "sealing_lip": return int(s >= 2.0 or n >= 2)
        if z == "flange_face":
            if sh == "irregular": return int(s > 3.0)
            return int(s > 5.5 or (n >= 2 and s > 4.0))
        if z == "rim": return int(s > 6.0)
        return int(s > 4.0)

    def answer(self, t):
        y = self.truth(t)
        return 1 - y if self.rng.random() < self.noise else y

# ------------------------------------------------------------------ expert twin
SHAPES = ["round", "irregular"]
COUNTS = [1, 2, 3]                          # 3 means "3 or more"
CONTEXTS = [(z, sh, n) for z in DEFECT_ZONES for sh in SHAPES for n in COUNTS]
TAU = np.linspace(0.5, 8.5, 161)            # candidate size limits (mm); 0.5 = reject all, 8.5 = accept all
SIZES = np.round(np.linspace(1.5, 7.0, 56), 2)

def ctx_of(t):
    return (t["zone"], t["shape"], min(int(t["count"]), 3))

class ExpertTwin:
    """Learns one hidden size limit per situation (zone x shape x number of pits).

    P(reject | size) = lapse + (1 - 2*lapse) * sigmoid((size - limit) / width)
    The posterior over each limit is kept on a grid (exact, fast, gives honest uncertainty).
    Prior = the written standard's limit, with wide doubt: the master is assumed to agree
    with the manual until his swipes say otherwise.
    """
    def __init__(self, width=0.25, lapse=0.04, prior_sd=2.0, seed=0, manual=None):
        self.width, self.lapse = width, lapse
        self.manual = manual or ManualTwin()
        self.log_post = {}
        for c in CONTEXTS:
            m = self._manual_limit(c)
            self.log_post[c] = -0.5 * ((TAU - m) / prior_sd) ** 2
        self.history = []          # (theta, y)

    def _manual_limit(self, c):
        z, sh, n = c
        return 0.5 if n > self.manual.MAX_COUNT else self.manual.MAX_PIT_MM

    def _lik(self, size):
        p = 1 / (1 + np.exp(-(size - TAU) / self.width))
        return self.lapse + (1 - 2 * self.lapse) * p        # P(reject | size, each tau)

    def add(self, theta, y):
        self.history.append((theta, int(y)))
        if theta.get("defect") == "none": return
        c = ctx_of(theta); pr = self._lik(theta["size_mm"])
        self.log_post[c] += np.log(pr if y else 1 - pr)

    def posterior(self, c):
        lp = self.log_post[c]; w = np.exp(lp - lp.max()); return w / w.sum()

    def limit(self, c):
        """(mean, sd) of the master's size limit in this situation."""
        w = self.posterior(c); mu = float((w * TAU).sum()); return mu, float(np.sqrt((w * (TAU - mu) ** 2).sum()))

    def p_reject(self, thetas):
        out = []
        for t in thetas:
            if t.get("defect") == "none": out.append(0.0); continue
            out.append(float((self.posterior(ctx_of(t)) * self._lik(t["size_mm"])).sum()))
        return np.array(out)

    def uncertainty(self, thetas):
        p = self.p_reject(thetas); return p * (1 - p)

    def info_gain(self, c, sizes=SIZES):
        """Expected reduction in uncertainty about this situation's limit, for each size we could show."""
        w = self.posterior(c)
        L = 1 / (1 + np.exp(-(sizes[:, None] - TAU[None, :]) / self.width))
        L = self.lapse + (1 - 2 * self.lapse) * L                       # (sizes, tau)
        py = L @ w
        def H(p): p = np.clip(p, 1e-9, 1 - 1e-9); return -(p * np.log(p) + (1 - p) * np.log(1 - p))
        return H(py) - (H(L) @ w)                                       # mutual information

# ------------------------------------------------------------------ active learning
# How often each situation occurs (used to weight what is worth asking about).
CTX_WEIGHT = {c: (0.6 if c[2] == 1 else 0.2) / 6 for c in CONTEXTS}

def choose_next(twin, manual, G, rng):
    """Active generation: ask the generator for the single defect the master's answer would teach us most about."""
    best, bscore = None, -1
    for c in CONTEXTS:
        ig = twin.info_gain(c)
        k = int(np.argmax(ig)); score = ig[k] * np.sqrt(CTX_WEIGHT[c])
        if score > bscore: best, bscore = (c, float(SIZES[k]), float(ig[k])), score
    (z, sh, n), size, ig = best
    t = G.sample_theta(rng, zone=z, shape=sh, count=n, size_mm=size)
    return t, dict(context=f"{z}/{sh}/{n}", size_mm=size, info_gain=ig)

def cold_start(G, rng):
    seq = []
    for z in DEFECT_ZONES:
        for sh, s in (("round", 2.5), ("irregular", 4.8)):
            seq.append(G.sample_theta(rng, zone=z, shape=sh, size_mm=s, count=1))
    return seq

# ------------------------------------------------------------------ evaluation helpers
def test_set(G, n=3000, seed=999):
    rng = np.random.default_rng(seed)
    return [G.sample_theta(rng) for _ in range(n)]

def agreement(twin, master, tests):
    p = twin.p_reject(tests) > 0.5
    y = np.array([master.truth(t) for t in tests]) > 0.5
    man = np.array([ManualTwin().p_reject(t) for t in tests]) > 0.5
    tacit = y != man                              # where the master and the manual disagree
    return dict(overall=float((p == y).mean()), on_tacit=float((p[tacit] == y[tacit]).mean()) if tacit.any() else None,
                manual_overall=float((man == y).mean()), tacit_share=float(tacit.mean()))

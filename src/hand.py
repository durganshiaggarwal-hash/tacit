"""Hand module — the same Tacit Delta engine, applied to a hand motion: deburring an edge of the cast impeller.

CONCEPT DEMO: strokes are generated, not recorded. A real version would record the tool with a phone or a
wrist-worn motion sensor; the engine below would not change.

theta = {
  "edge":      "sealing_lip" | "flange_edge" | "rim"   which edge of the impeller is deburred
  "end_speed": 4 .. 40 mm/s    tool speed over the last 30% of the stroke (the start is always 20 mm/s)
  "base":      index of the clean impeller photo,  "start_deg": where on the edge the stroke starts
}

Manual Twin : "Deburr at a steady 20 mm/s. Reject a stroke that ends faster than 25 mm/s."
Master      : one hidden end-speed limit per edge, learned from Good / Bad answers (same maths as the Eye).
"""
import numpy as np

EDGES = ["sealing_lip", "flange_edge", "rim"]
EDGE_RHO = {"sealing_lip": 1.11, "flange_edge": 1.45, "rim": 1.76}        # where the edge sits (geometry.py units)
EDGE_LABEL = {"sealing_lip": "Sealing lip edge", "flange_edge": "Flange face edge", "rim": "Outer rim edge"}
EDGE_JA = {"sealing_lip": "シールリップの縁", "flange_edge": "フランジ面の縁", "rim": "外周リムの縁"}
START_SPEED, ARC_DEG, RAMP = 20.0, 80.0, 0.30
TAU = np.linspace(0.0, 45.0, 181)
SPEEDS = np.round(np.linspace(4.0, 40.0, 73), 1)


def sample_theta(rng, n_bases=60, **fixed):
    t = dict(edge=str(rng.choice(EDGES)), end_speed=float(np.round(rng.uniform(4, 40), 1)),
             base=int(rng.integers(0, n_bases)), start_deg=float(np.round(rng.uniform(0, 360), 1)))
    t.update(fixed); return t


def speed_profile(t, n=60):
    """Speed (mm/s) at n evenly spaced points along the stroke: steady 20 mm/s, then a linear change to end_speed."""
    u = np.linspace(0, 1, n)
    v = np.where(u < 1 - RAMP, START_SPEED, START_SPEED + (t["end_speed"] - START_SPEED) * (u - (1 - RAMP)) / RAMP)
    return u.tolist(), np.round(v, 2).tolist()


class HandManual:
    LIMIT = 25.0
    text = ["Deburr at a steady 20 mm/s. Reject a stroke that ends faster than 25 mm/s."]
    text_ja = ["バリ取りは一定の20 mm/sで行う。終端が25 mm/sより速いストロークは不可。"]

    def p_reject(self, t): return float(t["end_speed"] > self.LIMIT)
    def limit(self, edge): return self.LIMIT


class HandMaster:
    """Hidden rules of a simulated master: eases off at the end of every stroke on the sealing lip
    (end under 12 mm/s, so the burr rolls off instead of tearing the seal face), roughly follows the manual on
    the flange edge (22 mm/s), and is relaxed on the outer rim, which is machined afterwards (32 mm/s)."""
    LIMITS = {"sealing_lip": 12.0, "flange_edge": 22.0, "rim": 32.0}

    def __init__(self, noise=0.04, seed=0):
        self.noise, self.rng = noise, np.random.default_rng(seed)

    def truth(self, t): return int(t["end_speed"] > self.LIMITS[t["edge"]])

    def answer(self, t):
        y = self.truth(t); return 1 - y if self.rng.random() < self.noise else y


class HandTwin:
    """One hidden end-speed limit per edge; P(bad | speed) rises around the limit."""
    def __init__(self, manual, width=1.2, lapse=0.08, prior_sd=10.0):
        self.width, self.lapse, self.manual = width, lapse, manual
        self.log_post = {e: -0.5 * ((TAU - manual.limit(e)) / prior_sd) ** 2 for e in EDGES}

    def _lik(self, s): return self.lapse + (1 - 2 * self.lapse) / (1 + np.exp(-(s - TAU) / self.width))

    def add(self, t, y):
        pr = self._lik(t["end_speed"]); self.log_post[t["edge"]] += np.log(pr if y else 1 - pr)

    def posterior(self, e):
        lp = self.log_post[e]; w = np.exp(lp - lp.max()); return w / w.sum()

    def limit(self, e):
        w = self.posterior(e); mu = float((w * TAU).sum()); return mu, float(np.sqrt((w * (TAU - mu) ** 2).sum()))

    def p_reject(self, thetas):
        return np.array([float((self.posterior(t["edge"]) * self._lik(t["end_speed"])).sum()) for t in thetas])

    def info_gain(self, e):
        w = self.posterior(e)
        L = self.lapse + (1 - 2 * self.lapse) / (1 + np.exp(-(SPEEDS[:, None] - TAU[None, :]) / self.width))
        py = L @ w
        H = lambda p: -(np.clip(p, 1e-9, 1) * np.log(np.clip(p, 1e-9, 1)) + np.clip(1 - p, 1e-9, 1) * np.log(np.clip(1 - p, 1e-9, 1)))
        return H(py) - H(L) @ w


def hand_choose_next(twin, rng, n_bases=60):
    best, bs = None, -1
    for e in EDGES:
        ig = twin.info_gain(e); k = int(np.argmax(ig))
        if ig[k] > bs: best, bs = (e, float(SPEEDS[k]), float(ig[k])), ig[k]
    e, s, ig = best
    return sample_theta(rng, n_bases, edge=e, end_speed=s), dict(context=e, end_speed=s, info_gain=ig)


def hand_rules(twin, manual, gap=5.0, sd_max=2.5):
    out = []
    for e in EDGES:
        mu, sd = twin.limit(e); man = manual.limit(e)
        if sd >= sd_max or abs(mu - man) < gap: continue
        if mu < man:
            en = f"{EDGE_LABEL[e]}: ease off at the end of the stroke, to {mu:.0f} mm/s or slower. The manual allows up to {man:.0f} mm/s."
            ja = f"{EDGE_JA[e]}：ストロークの終端で減速し、{mu:.0f} mm/s以下にする（標準書では{man:.0f} mm/sまで可）。"
        else:
            en = f"{EDGE_LABEL[e]}: ending at up to {mu:.0f} mm/s is fine. The manual rejects anything over {man:.0f} mm/s."
            ja = f"{EDGE_JA[e]}：終端{mu:.0f} mm/sまでは問題なし（標準書では{man:.0f} mm/s超は不可）。"
        out.append(dict(id=e, edge=e, limit=round(mu, 1), sd=round(sd, 1), manual=man,
                        direction="stricter" if mu < man else "more lenient", text_en=en, text_ja=ja))
    return out

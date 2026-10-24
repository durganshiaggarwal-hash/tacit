"""The Inspection Line (検品ライン): a short, game-like version of TACIT for demos and for masters who
should not have to read charts.

Round 1, the master: 11 parts on a conveyor. For each zone, TACIT picks the next pit size by narrowing down
the master's limit (a bisection: the simplest form of the active learning in the main app). Before each part
TACIT guesses the master's call; where it is unsure it assumes the manual. When the master's answers in a
zone contradict the manual, that zone becomes a "secret technique" scroll, worded from their own answers.

Round 2, the apprentice: parts sit in the gap between the manual and the master's limit. A wrong call shows
the master's scroll, and a second part in the same zone checks whether it was passed on.
"""
import numpy as np

MANUAL_MM = 4.0
GRID = np.round(np.arange(1.5, 7.0001, 0.1), 1)   # candidate positions of a master's line, mm
def lik(s, w=0.12, lapse=0.03):                 # P(master rejects a pit of size s | their line is at each grid value)
    return lapse + (1 - 2 * lapse) / (1 + np.exp(-(s - GRID + 0.05) / w))
TAPS = 4                                   # parts per zone in the master round (enough to land within about 0.3 mm)
ZONES = ["sealing_lip", "flange_face", "rim"]
ZN = dict(sealing_lip=("Sealing lip", "シールリップ"), flange_face=("Flange face", "フランジ面"), rim=("Outer rim", "外周リム"))
# the role card a judge can play: same hidden rules as the simulated master in the main app (round pits)
ROLE_CARD = dict(sealing_lip=("ng_from", 2.0), flange_face=("ok_up_to", 5.5), rim=("ok_up_to", 6.0))


def role_card_calls(zone, size):
    kind, v = ROLE_CARD[zone]
    return int(size >= v) if kind == "ng_from" else int(size > v)


def manual_call(t):
    return 0 if t.get("defect") == "none" else int(t["size_mm"] > MANUAL_MM or t.get("count", 1) >= 3)


def rule_text(zone, kind, v):
    en, ja = ZN[zone]
    if kind == "ng_from":
        return (f"{en}: reject pits from {v:.1f} mm. The manual allows up to {MANUAL_MM:.1f} mm.",
                f"{ja}：{v:.1f} mm以上のピットは不良。標準書では{MANUAL_MM:.1f} mmまで良品。")
    return (f"{en}: pits up to {v:.1f} mm are fine. The manual rejects above {MANUAL_MM:.1f} mm.",
            f"{ja}：{v:.1f} mmまでのピットは良品。標準書では{MANUAL_MM:.1f} mm超は不良。")


class Game:
    def __init__(self, G, png_url, role, master_rules=None, seed=None, quick=True):
        self.G, self.png, self.role, self.quick = G, png_url, role, quick
        self.rng = np.random.default_rng(seed)
        self.i, self.log, self.current, self.done = 0, [], None, False
        if role == "master":
            self.iv = {z: [1.5, 7.0] for z in ZONES}              # largest size accepted, smallest rejected
            self.asked = {z: 0 for z in ZONES}
            self.evidence = {z: [] for z in ZONES}
            # TAPS parts per zone; zones finish at different times so the secrets are spread out
            self.plan = ["sealing_lip", "flange_face", "sealing_lip", "rim", "sealing_lip", "flange_face", "sealing_lip",
                         "rim", "flange_face", "clean", "rim", "flange_face", "rim"]
            if quick:                                              # 2-minute demo: two zones, secrets at parts 6 and 8
                self.plan = ["sealing_lip", "sealing_lip", "flange_face", "sealing_lip", "flange_face", "sealing_lip",
                             "flange_face", "flange_face"]
            self.zones = [z for z in ZONES if z in self.plan]
            self.rules, self.reads = [], 0
            self.post = {z: np.full(len(GRID), 1.0 / len(GRID)) for z in self.zones}   # TACIT's belief about each line
        else:
            self.master_rules = master_rules or default_rules(quick)
            if quick: self.master_rules = [r for r in self.master_rules if r["zone"] in ("sealing_lip", "flange_face")] or self.master_rules
            self.plan = []
            for k in range(2):                                     # two gap parts per secret, interleaved
                for r in self.master_rules: self.plan.append(("gap", r["zone"], k))
                if not quick: self.plan.append(("clean", None, k) if k == 0 else ("obvious", "sealing_lip", k))
            self.state = {r["zone"]: [] for r in self.master_rules}
            self.right = 0
        self.n = len(self.plan)
        self._next()

    # ------------------------------------------------------------------ choosing parts
    def _master_size(self, z):
        lo, hi = self.iv[z]; k = self.asked[z]
        if k == 0: return 2.6                                     # below the manual's limit: does the master still reject?
        v = round((lo + hi) / 2, 1)                               # then halve the range between the last OK and the first NG
        seen = {s for s, _ in self.evidence[z]}
        while v in seen: v = round(v + 0.1, 1)
        return v

    def _make(self, zone, size):
        if zone is None:
            return dict(defect="none", base=int(self.rng.integers(0, len(self.G.A.bases))), seed=int(self.rng.integers(1 << 30)))
        return self.G.sample_theta(self.rng, zone=zone, shape="round", count=1, size_mm=round(float(size), 1), contrast=1.1)

    def _next(self):
        if self.i >= self.n:
            self.done, self.current = True, None; return
        if self.role == "master":
            z = self.plan[self.i]
            t = self._make(None, 0) if z == "clean" else self._make(z, self._master_size(z))
        else:
            kind, z, k = self.plan[self.i]
            if kind == "clean": t = self._make(None, 0)
            elif kind == "obvious": t = self._make("sealing_lip", 6.6)
            else:
                r = next(r for r in self.master_rules if r["zone"] == z)
                a, b = sorted([r["limit"], MANUAL_MM])
                f = (0.35, 0.7)[k]
                t = self._make(z, a + f * (b - a))
        self.current = t

    def guess(self, t):
        """TACIT's guess of the master's call: what it has learned in this zone, else the manual."""
        if t.get("defect") == "none": return 0
        lo, hi = self.iv[t["zone"]]; s = t["size_mm"]
        if s >= hi: return 1
        if s <= lo and self.asked[t["zone"]]: return 0
        return manual_call(t)

    def part(self):
        t = self.current
        img, info = self.G.render(t)
        H, W = img.shape[:2]
        p = dict(i=self.i + 1, n=self.n, image=self.png(img), role=self.role)
        if t.get("defect") == "none":
            p.update(zone=None, size_mm=None, gauge_en="No pits found", gauge_ja="ピットなし")
        else:
            d = info["defects"][0]; cx, cy = d["center"]; r = max(d["box"][2], 14) * 1.1
            en, ja = ZN[t["zone"]]
            p.update(zone=t["zone"], size_mm=t["size_mm"], gauge_en=f"{en}: pit {t['size_mm']:.1f} mm", gauge_ja=f"{ja}：ピット {t['size_mm']:.1f} mm",
                     marker=dict(x=100 * cx / W, y=100 * cy / H, r=100 * r / W))
        p["manual"] = manual_call(t)
        if self.role == "master":
            p["guess"] = self.guess(t); p["brain"] = self.brain(); p["why"] = self.why(t)
        return p

    # ------------------------------------------------------------------ answers
    def answer(self, y):
        t, out = self.current, {}
        if self.role == "master":
            g = self.guess(t); self.reads += int(g == y)
            out.update(guess=g, read=g == y, manual=manual_call(t))
            if t.get("defect") != "none":
                z, s = t["zone"], t["size_mm"]
                self.asked[z] += 1; self.evidence[z].append((s, y))
                lk = lik(s); self.post[z] = self.post[z] * (lk if y else 1 - lk); self.post[z] /= self.post[z].sum()   # Bayes update
                lo, hi = self.iv[z]
                if y: self.iv[z][1] = min(hi, s)
                else: self.iv[z][0] = max(lo, s)
                if self.asked[z] == TAPS:
                    r = self._rule(z)
                    if r: self.rules.append(r); out["scroll"] = r
        else:
            z = t.get("zone")
            truth = self._truth(t)
            ok = y == truth; self.right += int(ok)
            out.update(truth=truth, correct=ok, manual=manual_call(t))
            if self.plan[self.i][0] == "gap":
                self.state[z].append(ok)
                if not ok: out["scroll"] = next(r for r in self.master_rules if r["zone"] == z)
        self.log.append(dict(theta=t, y=y)); self.i += 1
        self._next()
        out["next"] = None if self.done else self.part()
        if self.done: out["summary"] = self.summary()
        return out

    def _rule(self, z):
        ev = self.evidence[z]
        stricter = [s for s, y in ev if y == 1 and s <= MANUAL_MM]   # rejected although the manual says OK
        lenient = [s for s, y in ev if y == 0 and s > MANUAL_MM]     # accepted although the manual says NG
        lo, hi = self.iv[z]
        v = round((lo + hi) / 2, 1)                               # the master's line: between the last OK and the first NG
        if stricter: kind = "ng_from"; v = min(v, MANUAL_MM)
        elif lenient: kind = "ok_up_to"; v = max(v, MANUAL_MM + 0.1)
        else:
            return None
        en, ja = rule_text(z, kind, v)
        return dict(zone=z, kind=kind, limit=round(v, 1), text_en=en, text_ja=ja, evidence=len(ev),
                    title_en=ZN[z][0], title_ja=ZN[z][1])

    def _truth(self, t):
        if t.get("defect") == "none": return 0
        r = next((r for r in self.master_rules if r["zone"] == t["zone"]), None)
        if r is None: return manual_call(t)
        return int(t["size_mm"] >= r["limit"]) if r["kind"] == "ng_from" else int(t["size_mm"] > r["limit"])

    # ------------------------------------------------------------------ what TACIT believes (shown live)
    def p_reject(self, z, s):
        return float((self.post[z] * lik(s)).sum())

    def brain(self):
        out = []
        for z in self.zones:
            w = self.post[z]; cdf = np.cumsum(w)
            mean = float((w * GRID).sum()); lo = float(GRID[np.searchsorted(cdf, .05)]); hi = float(GRID[min(len(GRID) - 1, np.searchsorted(cdf, .95))])
            certainty = float(np.clip(1 - (hi - lo) / (GRID[-1] - GRID[0]), 0, 1))
            out.append(dict(zone=z, title_en=ZN[z][0], title_ja=ZN[z][1], post=[round(float(v), 5) for v in w], mean=round(mean, 2), lo=round(lo, 1), hi=round(hi, 1),
                            certainty=round(certainty, 2), asked=self.asked[z], evidence=[[s_, y_] for s_, y_ in self.evidence[z]],
                            differs=bool(hi < MANUAL_MM - .05 or lo > MANUAL_MM + .05)))
        return dict(grid=[float(v) for v in GRID], manual=MANUAL_MM, zones=out, taps=self.i)

    def why(self, t):
        if t.get("defect") == "none":
            return dict(ja="傷のない部品。基本の確認です。", en="A clean part, to check the basics.", zone=None)
        z, s, k = t["zone"], t["size_mm"], self.asked[t["zone"]]
        p = self.p_reject(z, s)
        if k == 0:
            return dict(ja=f"まず、標準書では良品の {s:.1f} mm を試します。師匠も良品とするか？", en=f"First test: {s:.1f} mm, a pit the manual says is fine. Does the master agree?", zone=z, size=s, p=round(p, 2))
        return dict(ja=f"{s:.1f} mm を試します。TACIT が一番迷う所（不良の確率 {round(100 * p)}%）。", en=f"Testing {s:.1f} mm: where TACIT is least sure (P(reject) {round(100 * p)}%). One answer here teaches it the most.", zone=z, size=s, p=round(p, 2))

    def summary(self):
        if self.role == "master":
            return dict(role="master", parts=self.n, reads=self.reads, rules=self.rules, zones=self.zones, brain=self.brain(),
                        limits={z: dict(ok_up_to=self.iv[z][0], ng_from=self.iv[z][1]) for z in ZONES})
        passed = [r for r in self.master_rules if self.state[r["zone"]] and self.state[r["zone"]][-1]]
        return dict(role="apprentice", parts=self.n, right=self.right, rules=self.master_rules,
                    passed=[r["zone"] for r in passed], kaiden=len(passed) == len(self.master_rules))


def default_rules(quick=False):
    out = []
    for z, (kind, v) in ROLE_CARD.items():
        if quick and z == "rim": continue
        en, ja = rule_text(z, kind, v)
        out.append(dict(zone=z, kind=kind, limit=v, text_en=en, text_ja=ja, evidence=0, title_en=ZN[z][0], title_ja=ZN[z][1]))
    return out

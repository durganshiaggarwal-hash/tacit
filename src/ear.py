"""Ear module — the same Tacit Delta engine, applied to the sound of a running pump.

theta = {
  "severity":   0.0 .. 1.0   how much of the faulty recording is mixed in (0 = healthy pump, 1 = fully faulty)
  "shop":       "quiet" | "noisy"            extra factory background noise
  "distractor": "none" | "beep" | "knock"    a loud sound that is NOT the pump (forklift beep, pallet knocks)
  "clip":       index of the recording pair, "seed": random placement
}

Sound source
  If real MIMII / DCASE pump clips are present in assets/audio/normal/*.wav and assets/audio/anomaly/*.wav,
  they are used (first channel, first 4 s). Otherwise a SYNTHETIC stand-in pump is generated
  (motor harmonics + vane-pass tone + flow noise; the fault adds cavitation-like crackle and a rubbing tone).
  The UI says which one is in use.

Manual Twin (the written rule)
  "Stop the pump if the sound-level meter reads above LIMIT_DB." The meter hears everything, so loud
  distractors trip it, while a faint fault under a quiet hum does not.

Expert Twin
  One hidden limit per situation (shop x distractor): the fault severity at which the master stops the pump.
"""
import os, glob, io, base64, wave
import numpy as np, cv2

SR, DUR = 16000, 4.0
N = int(SR * DUR)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUDIO_DIR = os.path.join(ROOT, "assets", "audio")
SHOPS = ["quiet", "noisy"]
DISTRACTORS = ["none", "beep", "knock"]
EAR_CONTEXTS = [(s, d) for s in SHOPS for d in DISTRACTORS]
TAU = np.linspace(-0.2, 1.2, 141)            # candidate limits on severity; >1 means "never stops"
SEV = np.round(np.linspace(0.0, 1.0, 51), 2)
LABEL = {"quiet": "Quiet shop", "noisy": "Noisy shop", "none": "no other sounds", "beep": "forklift beeping nearby", "knock": "pallet knocks nearby"}
LABEL_JA = {"quiet": "静かな工場", "noisy": "騒がしい工場", "none": "他の音なし", "beep": "フォークリフトの警告音", "knock": "パレットの衝撃音"}


# ---------------------------------------------------------------------- sources
def _read_wav(path):
    with wave.open(path, "rb") as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    dt = {2: np.int16, 4: np.int32}.get(sw)
    if dt is None: raise ValueError(f"unsupported sample width {sw} in {path}")
    x = np.frombuffer(raw, dt).reshape(-1, ch)[:, 0].astype(np.float32) / np.iinfo(dt).max
    if sr != SR:
        idx = np.linspace(0, len(x) - 1, int(len(x) * SR / sr)); x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    x = x[:N] if len(x) >= N else np.pad(x, (0, N - len(x)))
    return x                      # raw level kept: real faults are often (not always) a little louder


class PumpSounds:
    def __init__(self):
        nf = sorted(glob.glob(os.path.join(AUDIO_DIR, "normal", "*.wav")))
        af = sorted(glob.glob(os.path.join(AUDIO_DIR, "anomaly", "*.wav")))
        self.real = bool(nf and af)
        self.source = (f"Real pump recordings (MIMII / DCASE, Hitachi): {len(nf)} normal, {len(af)} faulty"
                       if self.real else "Synthetic stand-in pump sound (real MIMII clips not loaded yet)")
        self.normal_files, self.anomaly_files = nf, af
        self.n_clips = min(len(nf), len(af)) if self.real else 24
        self._cache = {}
        self.gain = 1.0
        if self.real:   # one shared volume for all real clips, so a healthy pump reads about 80 dB on the meter
            rms = [np.sqrt(np.mean(_read_wav(f) ** 2)) for f in nf[:20]]
            self.gain = 0.1 / (float(np.median(rms)) + 1e-9)

    def pair(self, k):
        k = int(k) % self.n_clips
        if k not in self._cache:
            if self.real:
                self._cache[k] = (self.gain * _read_wav(self.normal_files[k]), self.gain * _read_wav(self.anomaly_files[k]))
            else:
                self._cache[k] = _synthetic_pair(k)
        return self._cache[k]


def _synthetic_pair(k):
    """A stand-in centrifugal pump at ~1450 rpm: motor harmonics, 6-blade vane-pass tone, flow noise.
    The faulty version adds cavitation-like crackle bursts and a faint rubbing tone."""
    r = np.random.default_rng(1000 + k); t = np.arange(N) / SR
    f0 = 24.2 * (1 + (r.random() - .5) * .03)
    x = sum(.08 / h * np.sin(2 * np.pi * h * f0 * t + r.random() * 6.28) for h in range(1, 9))
    x = x + .05 * np.sin(2 * np.pi * 6 * f0 * t) * (1 + .3 * np.sin(2 * np.pi * f0 * t))
    w = r.normal(0, 1, N).astype(np.float32); flow = np.convolve(w, np.ones(6) / 6, "same") * .05
    normal = (x + flow).astype(np.float32)
    fault = np.zeros(N, np.float32)
    for c in r.uniform(0, DUR, int(DUR * 60)):                       # crackle bursts
        i0 = int(c * SR); L = int(SR * .004)
        if i0 + L < N: fault[i0:i0 + L] += r.normal(0, 1, L) * np.exp(-np.arange(L) / (L / 4)) * r.uniform(.2, .6)
    fault += .03 * np.sin(2 * np.pi * 2900 * t) * (1 + np.sin(2 * np.pi * f0 * t)) / 2   # rubbing tone
    anomaly = normal + fault
    norm = lambda v: (v / (np.sqrt(np.mean(v ** 2)) + 1e-9) * .1).astype(np.float32)
    return norm(normal), norm(anomaly)


# ---------------------------------------------------------------------- generator
class EarGenerator:
    def __init__(self):
        self.src = PumpSounds()

    def sample_theta(self, rng, **fixed):
        t = dict(severity=float(np.round(rng.uniform(0, 1), 2)), shop=str(rng.choice(SHOPS)),
                 distractor=str(rng.choice(DISTRACTORS)), clip=int(rng.integers(0, self.src.n_clips)),
                 seed=int(rng.integers(0, 2 ** 31 - 1)))
        t.update(fixed); return t

    def render(self, t):
        r = np.random.default_rng(t["seed"])
        normal, anomaly = self.src.pair(t["clip"])
        a = float(np.clip(t["severity"], 0, 1))
        if self.src.real:   # two different recordings: power-preserving blend (no level dip half-way)
            x = np.sqrt(1 - a) * normal + np.sqrt(a) * anomaly
        else:               # stand-in: the faulty version is the healthy one plus a fault, so a linear blend
            x = (1 - a) * normal + a * anomaly
        if not self.src.real:
            x = x * (1 + 0.35 * a)                               # stand-in only: real faulty clips carry their own loudness
        tt = np.arange(N) / SR
        if t["shop"] == "noisy":                                 # factory background: rumble + hiss + clanks
            w = r.normal(0, 1, N).astype(np.float32)
            x = x + .06 * np.convolve(w, np.ones(40) / 40, "same") * 6 + .015 * w
            for c in r.uniform(0, DUR, 5):
                i0 = int(c * SR); L = int(SR * .08)
                if i0 + L < N: x[i0:i0 + L] += .12 * np.sin(2 * np.pi * r.uniform(300, 900) * tt[:L]) * np.exp(-tt[:L] / .02)
        if t["distractor"] == "beep":
            f = r.uniform(900, 1300); st = r.uniform(0, 1.5)
            on = ((tt > st) & (tt < st + 2.2) & (((tt - st) % .5) < .25)).astype(np.float32)
            x = x + .25 * on * np.sin(2 * np.pi * f * tt)
        if t["distractor"] == "knock":
            for k in range(3):
                i0 = int((.4 + k * 1.2 + r.uniform(0, .3)) * SR); L = int(SR * .15)
                if i0 + L < N: x[i0:i0 + L] += .9 * np.sin(2 * np.pi * 60 * tt[:L]) * np.exp(-tt[:L] / .03)
        return np.clip(x, -1, 1).astype(np.float32)


def meter_db(x):
    """What a hand-held sound-level meter would show (RMS, offset to read like dB SPL)."""
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9) + 100)


# ---------------------------------------------------------------------- manual, master, twin
class EarManual:
    LIMIT_DB = 82.0
    text = ["Stop the pump if the sound-level meter reads above 82 dB."]
    text_ja = ["騒音計が82 dBを超えたらポンプを停止する。"]

    def __init__(self, G):
        self.G = G; self._eff = None

    def p_reject(self, t):
        return float(meter_db(self.G.render(t)) > self.LIMIT_DB)

    def effective_limit(self, c, clips=6):
        """The fault severity at which the meter rule starts stopping the pump in this situation (averaged)."""
        if self._eff is None: self._eff = {}
        if c not in self._eff:
            lims = []
            for k in range(clips):
                lim = 1.1
                for s in SEV:
                    if self.p_reject(dict(severity=float(s), shop=c[0], distractor=c[1], clip=k, seed=77 + k)): lim = float(s); break
                lims.append(lim)
            self._eff[c] = float(np.median(lims))
        return self._eff[c]


class EarMaster:
    """Hidden rules of a simulated master who listens for the fault itself:
    stops at 30% fault in a quiet shop, 40% in a noisy shop (harder to hear), and ignores beeps and knocks."""
    rules = ["Stops the pump once the fault sound reaches 30% (quiet shop) or 40% (noisy shop).",
             "Ignores forklift beeps and pallet knocks: they are not the pump."]

    def __init__(self, noise=0.04, seed=0):
        self.noise, self.rng = noise, np.random.default_rng(seed)

    def truth(self, t):
        return int(t["severity"] >= (0.30 if t["shop"] == "quiet" else 0.40))

    def answer(self, t):
        y = self.truth(t); return 1 - y if self.rng.random() < self.noise else y


class EarTwin:
    """One hidden limit per situation, on fault severity (same maths as the Eye's Expert Twin)."""
    def __init__(self, manual, width=0.04, lapse=0.04, prior_sd=0.5):
        self.width, self.lapse, self.manual = width, lapse, manual
        self.log_post = {c: -0.5 * ((TAU - min(manual.effective_limit(c), 1.1)) / prior_sd) ** 2 for c in EAR_CONTEXTS}

    def _lik(self, s):
        return self.lapse + (1 - 2 * self.lapse) / (1 + np.exp(-(s - TAU) / self.width))

    def add(self, t, y):
        c = (t["shop"], t["distractor"]); pr = self._lik(t["severity"])
        self.log_post[c] += np.log(pr if y else 1 - pr)

    def posterior(self, c):
        lp = self.log_post[c]; w = np.exp(lp - lp.max()); return w / w.sum()

    def limit(self, c):
        w = self.posterior(c); mu = float((w * TAU).sum()); return mu, float(np.sqrt((w * (TAU - mu) ** 2).sum()))

    def p_reject(self, thetas):
        return np.array([float((self.posterior((t["shop"], t["distractor"])) * self._lik(t["severity"])).sum()) for t in thetas])

    def info_gain(self, c):
        w = self.posterior(c)
        L = self.lapse + (1 - 2 * self.lapse) / (1 + np.exp(-(SEV[:, None] - TAU[None, :]) / self.width))
        py = L @ w
        H = lambda p: -(np.clip(p, 1e-9, 1) * np.log(np.clip(p, 1e-9, 1)) + np.clip(1 - p, 1e-9, 1) * np.log(np.clip(1 - p, 1e-9, 1)))
        return H(py) - H(L) @ w


def ear_choose_next(twin, G, rng):
    best, bs = None, -1
    for c in EAR_CONTEXTS:
        ig = twin.info_gain(c); k = int(np.argmax(ig))
        if ig[k] > bs: best, bs = (c, float(SEV[k]), float(ig[k])), ig[k]
    (shop, dis), sev, ig = best
    return G.sample_theta(rng, shop=shop, distractor=dis, severity=sev), dict(context=f"{shop}/{dis}", severity=sev, info_gain=ig)


def ear_rules(twin, manual, gap=0.2, sd_max=0.1):
    out = []
    for c in EAR_CONTEXTS:
        mu, sd = twin.limit(c); man = manual.effective_limit(c)
        if sd >= sd_max or abs(mu - min(man, 1.1)) < gap: continue
        shop, dis = c; pct = int(round(100 * max(0, mu)))
        man_txt = "never" if man > 1.0 else ("every time, even with no fault" if man <= 0.02 else f"only from {int(round(100 * man))}%")
        man_ja = "停止しない" if man > 1.0 else ("故障がなくても毎回停止" if man <= 0.02 else f"{int(round(100 * man))}%からしか停止しない")
        dis_en = {"none": "", "beep": " The forklift beep is not the pump.", "knock": " The pallet knocks are not the pump."}[dis]
        dis_ja = {"none": "", "beep": "フォークリフトの音はポンプではない。", "knock": "パレットの衝撃音はポンプではない。"}[dis]
        en = f"{LABEL[shop]}, {LABEL[dis]}: stop the pump once the fault sound reaches {pct}%.{dis_en} The sound-level rule stops it {man_txt}."
        ja = f"{LABEL_JA[shop]}・{LABEL_JA[dis]}：故障音が{pct}%に達したらポンプを停止。{dis_ja}騒音計のルールでは{man_ja}。"
        out.append(dict(id=f"{shop}-{dis}", shop=shop, distractor=dis, limit=round(mu, 3), sd=round(sd, 3), manual=round(man, 3),
                        direction="stricter" if mu < man else "more lenient", text_en=en, text_ja=ja))
    return out


# ---------------------------------------------------------------------- media for the browser
def wav_data_url(x):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def spectrogram_data_url(x, width=640, height=220):
    n, hop = 512, 128
    win = np.hanning(n).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * win
    S = 20 * np.log10(np.abs(np.fft.rfft(frames, axis=1)) + 1e-6).T          # (freq, time)
    S = np.flipud(S)
    S = np.clip((S - (S.max() - 70)) / 70, 0, 1)
    img = cv2.resize((S * 255).astype(np.uint8), (width, height), interpolation=cv2.INTER_AREA)
    img = cv2.applyColorMap(img, cv2.COLORMAP_MAGMA)
    ok, b = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(b.tobytes()).decode()

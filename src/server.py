"""TACIT backend (Phase 2: master capture).

Run:  cd tacit && uvicorn src.server:app --reload --port 8000   then open http://localhost:8000
"""
import os, sys, io, json, base64, uuid, threading
from contextvars import ContextVar
PLAYER = ContextVar("player", default="local")   # which browser is calling (set per request)
import numpy as np, cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from generator import Generator
from twins import ExpertTwin, ManualTwin, SimulatedMaster, choose_next, cold_start, CONTEXTS, SHAPES
from geometry import ZONE_LABEL
from rules import extract_rules, example_thetas, crop_around, cause_groups, CAUSE_OPTIONS, CAUSE_JA
from trainee import TraineeEngine, SimulatedJunior

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC = os.path.join(ROOT, "static")
SESSION_FILE = os.path.join(ROOT, "assets", "session_master.json")

app = FastAPI(title="TACIT")


@app.middleware("http")
async def no_cache(request, call_next):
    """Always serve the latest pages, scripts and styles (avoids a browser showing an old version after an update).
    Also gives each browser its own player id, so several people can play the hosted demo at once."""
    pid = request.cookies.get("tacit_pid") or ""
    new = not (0 < len(pid) <= 40 and pid.replace("-", "").isalnum())
    if new: pid = uuid.uuid4().hex
    PLAYER.set(pid)
    resp = await call_next(request)
    if new: resp.set_cookie("tacit_pid", pid, max_age=60 * 60 * 24 * 30, samesite="lax")
    if not request.url.path.startswith("/fonts/"):
        resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp
G = Generator()
MANUAL = ManualTwin()
LOCK = threading.Lock()


class Session:
    def __init__(self, master_name="Master inspector", seed=0):
        self.master_name = master_name
        self.rng = np.random.default_rng(seed)
        self.twin = ExpertTwin(manual=MANUAL)
        self.cold = cold_start(G, self.rng)
        self.pending = {}            # card id -> (theta, info, why)
        self.log = []                # {theta, y, fake, why}
        self.rule_status = {}        # card id -> {status, text_en, text_ja}
        self.cause_names = {}        # pit shape -> cause name given by the master
        self._examples = {}          # cached example crops per card
        G.blocked.clear()

    def next_card(self):
        n = len(self.log)
        if n < len(self.cold):
            t = self.cold[n]; why = dict(context=f"{t['zone']}/{t['shape']}/{t['count']}", size_mm=t["size_mm"],
                                         info_gain=None, note="Warm-up: one example per zone and shape")
        else:
            t, why = choose_next(self.twin, MANUAL, G, self.rng)
            why["note"] = "Generated where the master's answer teaches the most"
        img, info = G.render(t)
        cid = uuid.uuid4().hex[:10]
        self.pending[cid] = (t, info, why)
        return cid, img, why

    def answer(self, cid, verdict, fake=False):
        if cid not in self.pending: raise KeyError(cid)
        t, info, why = self.pending.pop(cid)
        if fake:
            for d in info["defects"]: G.blocked.add(d["patch"])
        else:
            self.twin.add(t, 1 if verdict == "reject" else 0)
        self.log.append(dict(theta=t, y=None if fake else (1 if verdict == "reject" else 0), fake=bool(fake),
                             patches=[d["patch"] for d in info["defects"]], why=why))

    def undo(self):
        if not self.log: return
        last = self.log.pop()
        for p in last["patches"]:
            if last["fake"]: G.blocked.discard(p)
        # rebuild the twin from the remaining log (exact, cheap)
        self.twin = ExpertTwin(manual=MANUAL)
        for e in self.log:
            if not e["fake"]: self.twin.add(e["theta"], e["y"])

    def state(self):
        rows = []
        for (z, sh, n) in CONTEXTS:
            mu, sd = self.twin.limit((z, sh, n))
            manual = 0.5 if n > MANUAL.MAX_COUNT else MANUAL.MAX_PIT_MM
            seen = sum(1 for e in self.log if not e["fake"] and e["theta"]["zone"] == z and e["theta"]["shape"] == sh and min(e["theta"]["count"], 3) == n)
            confident = sd < 0.6
            tacit = confident and abs(mu - manual) >= 0.7 and n <= MANUAL.MAX_COUNT
            rows.append(dict(zone=z, zone_label=ZONE_LABEL[z], shape=sh, count=n, limit=round(mu, 2), sd=round(sd, 2),
                             manual=manual, seen=seen, confident=confident, tacit=tacit,
                             direction=("stricter" if mu < manual else "more lenient") if tacit else None))
        answered = [e for e in self.log if not e["fake"]]
        return dict(master=self.master_name, swipes=len(answered), flagged_fake=sum(e["fake"] for e in self.log),
                    rejects=sum(e["y"] for e in answered), manual_rules=MANUAL.text, situations=rows,
                    tacit_found=sum(r["tacit"] for r in rows),
                    recent=[dict(context=f"{e['theta']['zone']}/{e['theta']['shape']}/{e['theta']['count']}",
                                 size_mm=e["theta"]["size_mm"], y=e["y"], fake=e["fake"],
                                 manual=int(MANUAL.p_reject(e["theta"]) > 0.5)) for e in self.log[-12:]][::-1])

    def save(self):
        json.dump(dict(master=self.master_name, log=[{k: v for k, v in e.items()} for e in self.log],
                       rule_status=self.rule_status, cause_names=self.cause_names),
                  open(SESSION_FILE, "w"), indent=1)

    # ---------------------------------------------------------------- Phase 3
    def rules(self):
        cards = extract_rules(self.twin, self.log, self.cause_names)
        for c in cards:
            st = self.rule_status.get(c["id"], {})
            c["status"] = st.get("status", "pending")
            if st.get("text_en"): c["text_en"] = st["text_en"]; c["edited"] = True
            if st.get("text_ja"): c["text_ja"] = st["text_ja"]; c["edited"] = True
            key = (c["id"], round(c["limit"], 1))
            if key not in self._examples:
                ex = []
                for label, t in example_thetas(c, G):
                    img, info = G.render(t)
                    ex.append(dict(label=label, size_mm=t["size_mm"], image=png_data_url(crop_around(img, info))))
                self._examples[key] = ex
            c["examples"] = self._examples[key]
        return dict(master=self.master_name, swipes=sum(1 for e in self.log if not e["fake"]), cards=cards,
                    approved=sum(c["status"] == "approved" for c in cards))

    def causes(self):
        res = cause_groups(self.log, G.A.library)
        for g in res["groups"]:
            thumbs = []
            for i in g["members"][:4]:
                e = self.log[i]; img, info = G.render(e["theta"])
                thumbs.append(png_data_url(crop_around(img, info, half=36, out=112)))
            g["thumbs"] = thumbs; g["name"] = self.cause_names.get(g["shape"])
            del g["members"]
        res["options"] = CAUSE_OPTIONS; res["options_ja"] = CAUSE_JA
        return res


S = Session()


def png_data_url(img):
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


class Answer(BaseModel):
    id: str
    verdict: str            # "accept" | "reject"
    fake: bool = False


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "learn.html"))


@app.get("/about")
def about_page():
    return FileResponse(os.path.join(STATIC, "home.html"))


@app.get("/eye")
def eye_page():
    return FileResponse(os.path.join(STATIC, "capture.html"))


@app.get("/tacit.css")
def tacit_css():
    return FileResponse(os.path.join(STATIC, "tacit.css"), media_type="text/css")


@app.get("/fonts/{name}")
def font_file(name: str):
    path = os.path.join(STATIC, "fonts", os.path.basename(name))
    if not name.endswith((".woff2", ".css")) or not os.path.exists(path): raise HTTPException(404, "no such font")
    return FileResponse(path, media_type="text/css" if name.endswith(".css") else "font/woff2")


@app.get("/deck.css")
def deck_css():
    return FileResponse(os.path.join(STATIC, "deck.css"), media_type="text/css")


@app.get("/calm.css")
def calm_css():
    return FileResponse(os.path.join(STATIC, "calm.css"), media_type="text/css")


@app.get("/learn")
def learn_page():
    return FileResponse(os.path.join(STATIC, "learn.html"))


@app.get("/pixel.css")
def pixel_css():
    return FileResponse(os.path.join(STATIC, "pixel.css"), media_type="text/css")


@app.get("/lesson.js")
def lesson_js():
    return FileResponse(os.path.join(STATIC, "lesson.js"), media_type="application/javascript")


@app.get("/wire.js")
def wire_js():
    return FileResponse(os.path.join(STATIC, "wire.js"), media_type="application/javascript")


@app.get("/bilingual.js")
def bilingual_js():
    return FileResponse(os.path.join(STATIC, "bilingual.js"), media_type="application/javascript")


@app.get("/tacit.js")
def tacit_js():
    return FileResponse(os.path.join(STATIC, "tacit.js"), media_type="application/javascript")


@app.get("/rules")
def rules_page():
    return FileResponse(os.path.join(STATIC, "rules.html"))


@app.get("/train")
def train_page():
    return FileResponse(os.path.join(STATIC, "train.html"))


@app.get("/prince")
def prince_page():
    return FileResponse(os.path.join(STATIC, "prince.html"))


@app.get("/ear")
def ear_page():
    return FileResponse(os.path.join(STATIC, "ear.html"))


@app.get("/hand")
def hand_page():
    return FileResponse(os.path.join(STATIC, "hand.html"))


@app.get("/passport")
def passport_page():
    return FileResponse(os.path.join(STATIC, "passport.html"))


@app.get("/api/capture/next")
def next_card():
    with LOCK:
        cid, img, why = S.next_card()
        return dict(id=cid, image=png_data_url(img), why=why, n=len(S.log))


@app.post("/api/capture/answer")
def answer(a: Answer):
    if a.verdict not in ("accept", "reject"): raise HTTPException(400, "verdict must be accept or reject")
    with LOCK:
        try: S.answer(a.id, a.verdict, a.fake)
        except KeyError: raise HTTPException(404, "That card has expired. Load the next one.")
        S.save()
        return S.state()


@app.post("/api/capture/undo")
def undo():
    with LOCK:
        S.undo(); S.save(); return S.state()


@app.get("/api/capture/state")
def state():
    with LOCK: return S.state()


@app.post("/api/capture/reset")
def reset():
    global S, TS
    with LOCK:
        S = Session(); TS = None; return S.state()


@app.post("/api/capture/autofill")
def autofill(n: int = 60, seed: int = 1):
    """Backup demo: a simulated master with hidden rules answers n cards."""
    with LOCK:
        m = SimulatedMaster(seed=seed)
        for _ in range(n):
            cid, img, why = S.next_card()
            t = S.pending[cid][0]
            S.answer(cid, "reject" if m.answer(t) else "accept")
        S.save()
        return S.state()


class RuleUpdate(BaseModel):
    status: str                         # "approved" | "rejected" | "pending"
    text_en: str | None = None
    text_ja: str | None = None


class CauseName(BaseModel):
    shape: str
    name: str


@app.get("/api/rules")
def get_rules():
    with LOCK: return S.rules()


@app.post("/api/rules/{card_id}")
def set_rule(card_id: str, u: RuleUpdate):
    if u.status not in ("approved", "rejected", "pending"): raise HTTPException(400, "status must be approved, rejected or pending")
    with LOCK:
        st = S.rule_status.setdefault(card_id, {})
        st["status"] = u.status
        if u.text_en is not None: st["text_en"] = u.text_en.strip()
        if u.text_ja is not None: st["text_ja"] = u.text_ja.strip()
        S.save(); return S.rules()


@app.get("/api/causes")
def get_causes():
    with LOCK: return S.causes()


@app.post("/api/causes")
def name_cause(c: CauseName):
    if c.shape not in ("round", "irregular"): raise HTTPException(400, "shape must be round or irregular")
    with LOCK:
        S.cause_names[c.shape] = c.name.strip()
        # rule text changes with the cause, so drop custom text only where it was not edited by hand
        S.save(); return S.causes()


# ==================================================================== Phase 4: trainee loop
class TraineeSession:
    def __init__(self, name="Junior inspector", seed=0):
        self.name = name
        cards = extract_rules(S.twin, S.log, S.cause_names)
        for c in cards:
            st = S.rule_status.get(c["id"], {})
            c["status"] = st.get("status", "pending")
            if st.get("text_en"): c["text_en"] = st["text_en"]
            if st.get("text_ja"): c["text_ja"] = st["text_ja"]
        approved = [c for c in cards if c["status"] == "approved"]
        self.cards = approved or [c for c in cards if c["status"] != "rejected"]
        self.uses_approved_only = bool(approved)
        self.E = TraineeEngine(S.twin, self.cards, seed=seed)
        self.pending, self.log = {}, []
        self.shift = None            # {parts, answers, i}
        self.shift_result = None

    def next_card(self):
        if self.shift and self.shift["i"] < len(self.shift["parts"]):
            t = self.shift["parts"][self.shift["i"]]; why = dict(reason="Real shift: no feedback until the end")
            mode = "shift"
        else:
            t, why = self.E.next_theta(G); mode = "train"
        img, info = G.render(t)
        cid = uuid.uuid4().hex[:10]; self.pending[cid] = (t, info, mode)
        return cid, img, why, mode

    def answer(self, cid, verdict):
        t, info, mode = self.pending.pop(cid)
        y = 1 if verdict == "reject" else 0
        if mode == "shift":
            self.shift["answers"].append(y); self.shift["i"] += 1
            if self.shift["i"] >= len(self.shift["parts"]):
                self.shift_result = self.E.score_shift(self.shift["parts"], self.shift["answers"])
                self.shift_result["trainee_name"] = self.name
            return dict(mode="shift")
        m = self.E.master_call(t); man = int(MANUAL.p_reject(t) > 0.5)
        self.E.record(t, y)
        card = self.E.card_for(t)
        self.log.append(dict(theta=t, y=y, master=m, manual=man, card=card["id"] if card else None))
        fb = dict(mode="train", correct=(y == m), your_call=verdict, master_call="reject" if m else "accept",
                  manual_call="reject" if man else "accept", manual_differs=(m != man),
                  size_mm=t.get("size_mm"), zone=t.get("zone"), shape=t.get("shape"), count=t.get("count"))
        if t.get("defect") != "none":
            fb["image"] = png_data_url(crop_around(G.render(t)[0], info))
        if card: fb["rule"] = dict(id=card["id"], text_en=card["text_en"], text_ja=card["text_ja"])
        return fb

    def state(self):
        prog = self.E.rule_progress()
        answered = len(self.log)
        recent = self.log[-20:]
        return dict(name=self.name, master=S.master_name, ready=bool(self.cards), uses_approved_only=self.uses_approved_only,
                    answers=answered, correct_recent=sum(e["y"] == e["master"] for e in recent), recent_n=len(recent),
                    absorbed=sum(r["status"] == "absorbed" for r in prog), rules=prog,
                    distance=[round(d, 3) for d in self.E.history], start_distance=round(self.E.distance() if not self.E.history else self.E.history[0], 3),
                    shift=(dict(i=self.shift["i"], n=len(self.shift["parts"])) if self.shift and self.shift["i"] < len(self.shift["parts"]) else None),
                    shift_result=self.shift_result)


TS = None


def trainee():
    global TS
    # (re)build when there is none yet, or when it had no rules and the master has answered since
    if TS is None or (not TS.cards and not TS.log): TS = TraineeSession()
    return TS


class TrainAnswer(BaseModel):
    id: str
    verdict: str


@app.get("/api/train/state")
def train_state():
    with LOCK: return trainee().state()


@app.get("/api/train/next")
def train_next():
    with LOCK:
        T = trainee()
        if not T.cards: raise HTTPException(409, "No unwritten rules yet. Capture the master first, then come back.")
        cid, img, why, mode = T.next_card()
        return dict(id=cid, image=png_data_url(img), why=why, mode=mode, n=len(T.log) + 1,
                    shift=T.state()["shift"])


@app.post("/api/train/answer")
def train_answer(a: TrainAnswer):
    if a.verdict not in ("accept", "reject"): raise HTTPException(400, "verdict must be accept or reject")
    with LOCK:
        T = trainee()
        if a.id not in T.pending: raise HTTPException(404, "That part has expired. Load the next one.")
        fb = T.answer(a.id, a.verdict)
        return dict(feedback=fb, state=T.state())


@app.post("/api/train/shift")
def train_shift(n: int = 150):
    with LOCK:
        T = trainee()
        T.shift = dict(parts=T.E.make_shift(G, n=n), answers=[], i=0); T.shift_result = None
        return T.state()


@app.post("/api/train/reset")
def train_reset():
    global TS
    with LOCK:
        TS = TraineeSession(); return TS.state()


@app.post("/api/train/autofill")
def train_autofill(n: int = 40, seed: int = 3):
    """Backup demo: a simulated junior who starts by following the manual practises n parts."""
    with LOCK:
        T = trainee(); J = getattr(T, "_junior", None) or SimulatedJunior(seed=seed); T._junior = J
        for _ in range(n):
            cid, img, why, mode = T.next_card()
            t = T.pending[cid][0]; y = J.answer(t)
            if mode == "shift":
                T.answer(cid, "reject" if y else "accept")
            else:
                m = T.E.master_call(t); T.answer(cid, "reject" if y else "accept"); J.learn(t, m, y, T.E.card_for(t))
        return T.state()


@app.post("/api/train/autoshift")
def train_autoshift(n: int = 150):
    """Backup demo: the simulated junior sits a real shift."""
    with LOCK:
        T = trainee(); J = getattr(T, "_junior", None) or SimulatedJunior(seed=3); T._junior = J
        T.shift = dict(parts=T.E.make_shift(G, n=n), answers=[], i=0); T.shift_result = None
        for t in T.shift["parts"]:
            T.shift["answers"].append(J.decide(t)); T.shift["i"] += 1
        T.shift_result = T.E.score_shift(T.shift["parts"], T.shift["answers"]); T.shift_result["trainee_name"] = T.name + " (simulated)"
        return T.state()


# ==================================================================== Ear module: pump sound
from ear import (EarGenerator, EarManual, EarMaster, EarTwin, ear_choose_next, ear_rules, EAR_CONTEXTS,
                 LABEL, LABEL_JA, meter_db, wav_data_url, spectrogram_data_url)
GE = EarGenerator()
EAR_MANUAL = EarManual(GE)


class EarSession:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.twin = EarTwin(EAR_MANUAL)
        self.pending, self.log = {}, []

    def next_card(self):
        if len(self.log) < len(EAR_CONTEXTS):          # warm-up: one clip per situation, middling fault
            c = EAR_CONTEXTS[len(self.log)]
            t = GE.sample_theta(self.rng, shop=c[0], distractor=c[1], severity=0.5)
            why = dict(context=f"{c[0]}/{c[1]}", severity=0.5, info_gain=None, note="Warm-up: one clip per situation")
        else:
            t, why = ear_choose_next(self.twin, GE, self.rng); why["note"] = "Generated where your answer teaches the most"
        x = GE.render(t); cid = uuid.uuid4().hex[:10]; self.pending[cid] = t
        return cid, x, why, t

    def answer(self, cid, verdict):
        t = self.pending.pop(cid); y = 1 if verdict == "reject" else 0
        self.twin.add(t, y); self.log.append(dict(theta=t, y=y, manual=int(EAR_MANUAL.p_reject(t))))

    def state(self):
        rows = []
        for c in EAR_CONTEXTS:
            mu, sd = self.twin.limit(c); man = EAR_MANUAL.effective_limit(c)
            seen = sum(1 for e in self.log if (e["theta"]["shop"], e["theta"]["distractor"]) == c)
            tacit = sd < 0.1 and abs(mu - min(man, 1.1)) >= 0.2
            rows.append(dict(shop=c[0], distractor=c[1], label=f"{LABEL[c[0]]}", sub=LABEL[c[1]], label_ja=f"{LABEL_JA[c[0]]}・{LABEL_JA[c[1]]}",
                             limit=round(mu, 3), sd=round(sd, 3), manual=round(man, 3), seen=seen, confident=sd < 0.1, tacit=tacit,
                             direction=("stricter" if mu < man else "more lenient") if tacit else None))
        return dict(source=GE.src.source, real=GE.src.real, answers=len(self.log), stops=sum(e["y"] for e in self.log),
                    manual_rules=EAR_MANUAL.text, situations=rows, rules=ear_rules(self.twin, EAR_MANUAL),
                    recent=[dict(context=f"{e['theta']['shop']}/{e['theta']['distractor']}", severity=e["theta"]["severity"],
                                 y=e["y"], manual=e["manual"]) for e in self.log[-12:]][::-1])


ES = EarSession()


@app.get("/api/ear/next")
def ear_next():
    with LOCK:
        cid, x, why, t = ES.next_card()
        return dict(id=cid, audio=wav_data_url(x), spectrogram=spectrogram_data_url(x), meter_db=round(meter_db(x), 1),
                    why=why, n=len(ES.log) + 1)


@app.post("/api/ear/answer")
def ear_answer(a: TrainAnswer):
    if a.verdict not in ("accept", "reject"): raise HTTPException(400, "verdict must be accept or reject")
    with LOCK:
        if a.id not in ES.pending: raise HTTPException(404, "That clip has expired. Load the next one.")
        ES.answer(a.id, a.verdict); return ES.state()


@app.get("/api/ear/state")
def ear_state():
    with LOCK: return ES.state()


@app.post("/api/ear/reset")
def ear_reset():
    global ES
    with LOCK:
        ES = EarSession(); return ES.state()


@app.post("/api/ear/autofill")
def ear_autofill(n: int = 40, seed: int = 1):
    """Backup demo: a simulated master with hidden listening rules answers n clips."""
    with LOCK:
        m = EarMaster(seed=seed)
        for _ in range(n):
            cid, x, why, t = ES.next_card(); ES.answer(cid, "reject" if m.answer(t) else "accept")
        return ES.state()


# ==================================================================== Hand module: deburring stroke (concept demo)
from hand import (HandManual, HandMaster, HandTwin, hand_choose_next, hand_rules, sample_theta as hand_theta,
                  speed_profile, EDGES, EDGE_RHO, EDGE_LABEL, EDGE_JA, START_SPEED, ARC_DEG, RAMP)
from geometry import mm_per_px
HAND_MANUAL = HandManual()


def hand_scene(t):
    img, g, _ = G.A.base(t["base"])
    u, v = speed_profile(t)
    return dict(image=png_data_url(img), cx=g["cx"], cy=g["cy"], a=g["a"], b=g["b"], theta=g["theta"], size=img.shape[1],
                rho=EDGE_RHO[t["edge"]], start_deg=t["start_deg"], arc_deg=ARC_DEG, ramp=RAMP, u=u, speed=v,
                mm_per_px=mm_per_px(g), edge=t["edge"], edge_label=EDGE_LABEL[t["edge"]])


class HandSession:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed); self.twin = HandTwin(HAND_MANUAL); self.pending, self.log = {}, []

    def next_card(self):
        nb = len(G.A.bases)
        if len(self.log) < len(EDGES):
            t = hand_theta(self.rng, nb, edge=EDGES[len(self.log)], end_speed=20.0)
            why = dict(context=t["edge"], end_speed=20.0, info_gain=None, note="Warm-up: one steady stroke per edge")
        else:
            t, why = hand_choose_next(self.twin, self.rng, nb); why["note"] = "Generated where your answer teaches the most"
        cid = uuid.uuid4().hex[:10]; self.pending[cid] = t
        return cid, t, why

    def answer(self, cid, verdict):
        t = self.pending.pop(cid); y = 1 if verdict == "reject" else 0
        self.twin.add(t, y); self.log.append(dict(theta=t, y=y, manual=int(HAND_MANUAL.p_reject(t))))

    def state(self):
        rows = []
        for e in EDGES:
            mu, sd = self.twin.limit(e); seen = sum(1 for x in self.log if x["theta"]["edge"] == e)
            tacit = sd < 2.5 and abs(mu - HAND_MANUAL.LIMIT) >= 5
            rows.append(dict(edge=e, label=EDGE_LABEL[e], label_ja=EDGE_JA[e], limit=round(mu, 1), sd=round(sd, 1),
                             manual=HAND_MANUAL.LIMIT, seen=seen, confident=sd < 2.5, tacit=tacit,
                             direction=("stricter" if mu < HAND_MANUAL.LIMIT else "more lenient") if tacit else None))
        return dict(answers=len(self.log), manual_rules=HAND_MANUAL.text, situations=rows, rules=hand_rules(self.twin, HAND_MANUAL),
                    start_speed=START_SPEED,
                    note="Concept demo: strokes are generated. A real version records the tool with a phone or wrist sensor.")


HS = HandSession()


class TraceIn(BaseModel):
    edge: str
    end_speed: float
    start_speed: float | None = None


@app.get("/api/hand/next")
def hand_next():
    with LOCK:
        cid, t, why = HS.next_card()
        return dict(id=cid, scene=hand_scene(t), end_speed=t["end_speed"], why=why, n=len(HS.log) + 1)


@app.post("/api/hand/answer")
def hand_answer(a: TrainAnswer):
    if a.verdict not in ("accept", "reject"): raise HTTPException(400, "verdict must be accept or reject")
    with LOCK:
        if a.id not in HS.pending: raise HTTPException(404, "That stroke has expired. Load the next one.")
        HS.answer(a.id, a.verdict); return HS.state()


@app.get("/api/hand/state")
def hand_state():
    with LOCK: return HS.state()


@app.post("/api/hand/reset")
def hand_reset():
    global HS
    with LOCK:
        HS = HandSession(); return HS.state()


@app.post("/api/hand/autofill")
def hand_autofill(n: int = 25, seed: int = 1):
    """Backup demo: a simulated master with hidden motion rules judges n strokes."""
    with LOCK:
        m = HandMaster(seed=seed)
        for _ in range(n):
            cid, t, why = HS.next_card(); HS.answer(cid, "reject" if m.answer(t) else "accept")
        return HS.state()


@app.get("/api/hand/trace_scene")
def hand_trace_scene(edge: str = "sealing_lip"):
    if edge not in EDGES: raise HTTPException(400, "unknown edge")
    with LOCK:
        t = hand_theta(HS.rng, len(G.A.bases), edge=edge, end_speed=20.0, start_deg=200.0)
        return hand_scene(t)


@app.post("/api/hand/trace")
def hand_trace(x: TraceIn):
    """Score a junior's own traced stroke against the master's learned limit for that edge."""
    if x.edge not in EDGES: raise HTTPException(400, "unknown edge")
    with LOCK:
        mu, sd = HS.twin.limit(x.edge)
        t = dict(edge=x.edge, end_speed=float(x.end_speed))
        p = float(HS.twin.p_reject([t])[0])
        return dict(end_speed=round(x.end_speed, 1), start_speed=x.start_speed, master_limit=round(mu, 1), master_sd=round(sd, 1),
                    manual_limit=HAND_MANUAL.LIMIT, master_says="bad" if p > 0.5 else "good",
                    manual_says="bad" if x.end_speed > HAND_MANUAL.LIMIT else "good", p_bad=round(p, 2), learned_from=HS.state()["answers"])


# ==================================================================== All senses: one list of the master's rules
def _status(cid, card):
    st = S.rule_status.get(cid, {})
    card["status"] = st.get("status", "pending")
    if st.get("text_en"): card["text_en"] = st["text_en"]; card["edited"] = True
    if st.get("text_ja"): card["text_ja"] = st["text_ja"]; card["edited"] = True
    return card


_EAR_EX = {}


def _ear_examples(r):
    key = (r["id"], round(r["limit"], 2))
    if key not in _EAR_EX:
        rng = np.random.default_rng(5); ex = []
        for label, sev in (("Master stops it", min(1.0, r["limit"] + 0.15)), ("Master keeps it running", max(0.0, r["limit"] - 0.15))):
            t = GE.sample_theta(rng, shop=r["shop"], distractor=r["distractor"], severity=round(sev, 2))
            x = GE.render(t)
            ex.append(dict(label=label, level=round(100 * sev), audio=wav_data_url(x), meter_db=round(meter_db(x), 1),
                           meter_says="stop" if meter_db(x) > EAR_MANUAL.LIMIT_DB else "keep running"))
        _EAR_EX[key] = ex
    return _EAR_EX[key]


@app.get("/api/rules/all")
def rules_all():
    with LOCK: return _rules_all()


def _rules_all(audio=True):
    if True:
        eye = []
        for c in S.rules()["cards"]:
            c = dict(c); c.update(sense="eye", unit="mm", lo=1.5, hi=7.0,
                                  title=f"{ZONE_LABEL.get(c['zone'], c['zone'])}", sub={"any": "Any pit shape", "round": "Round, smooth pits", "irregular": "Irregular, jagged pits"}[c["shape"]] + (" · two pits" if c["count"] == 2 else ""))
            eye.append(c)
        ear = []
        for r in ear_rules(ES.twin, EAR_MANUAL):
            cid = "ear:" + r["id"]
            seen = sum(1 for e in ES.log if (e["theta"]["shop"], e["theta"]["distractor"]) == (r["shop"], r["distractor"]))
            c = dict(r, id=cid, sense="ear", unit="%", lo=0, hi=100, limit=round(100 * r["limit"], 1), sd=round(100 * r["sd"], 1),
                     manual=min(110.0, round(100 * r["manual"], 1)), evidence=seen,
                     title=LABEL[r["shop"]], sub=LABEL[r["distractor"]].capitalize(), audio_examples=_ear_examples(r) if audio else [])
            ear.append(_status(cid, c))
        hand = []
        for r in hand_rules(HS.twin, HAND_MANUAL):
            cid = "hand:" + r["id"]
            seen = sum(1 for e in HS.log if e["theta"]["edge"] == r["edge"])
            c = dict(r, id=cid, sense="hand", unit="mm/s", lo=0, hi=45, evidence=seen, title=EDGE_LABEL[r["edge"]], sub="Speed at the end of the stroke")
            hand.append(_status(cid, c))
        allc = eye + ear + hand
        return dict(master=S.master_name, eye=eye, ear=ear, hand=hand,
                    manual=dict(eye=MANUAL.text, ear=EAR_MANUAL.text, hand=HAND_MANUAL.text),
                    answers=dict(eye=sum(1 for e in S.log if not e["fake"]), ear=len(ES.log), hand=len(HS.log)),
                    sources=dict(ear=GE.src.source, hand="Concept demo: strokes are generated"),
                    total=len(allc), approved=sum(c["status"] == "approved" for c in allc))


# ==================================================================== Junior training for Ear and Hand
from limit_trainee import LimitTrainee, SimulatedJunior as LimitJunior, ear_sense, hand_sense, ear_rule_list, hand_rule_list


def _approved_only(rules, prefix):
    for r in rules:
        st = S.rule_status.get(prefix + r["id"], {})
        r["status"] = st.get("status", "pending")
        if st.get("text_en"): r["text_en"] = st["text_en"]
        if st.get("text_ja"): r["text_ja"] = st["text_ja"]
    ok = [r for r in rules if r["status"] == "approved"]
    return ok or [r for r in rules if r["status"] != "rejected"]


class SenseTrainee:
    def __init__(self, kind):
        self.kind = kind
        if kind == "ear":
            self.sense = ear_sense(GE, EAR_MANUAL); rules = _approved_only(ear_rule_list(ES.twin, EAR_MANUAL), "ear:"); master = ES.twin
        else:
            self.sense = hand_sense(HAND_MANUAL, len(G.A.bases)); rules = _approved_only(hand_rule_list(HS.twin, HAND_MANUAL), "hand:"); master = HS.twin
        self.E = LimitTrainee(self.sense, master, rules); self.rules = rules
        self.pending, self.log = {}, []

    def media(self, t):
        if self.kind == "ear":
            x = GE.render(t)
            return dict(audio=wav_data_url(x), spectrogram=spectrogram_data_url(x), meter_db=round(meter_db(x), 1))
        return dict(scene=hand_scene(t), end_speed=t["end_speed"])

    def manual_call(self, t):
        return int(EAR_MANUAL.p_reject(t)) if self.kind == "ear" else int(HAND_MANUAL.p_reject(t))

    def answer(self, cid, verdict):
        t = self.pending.pop(cid); y = 1 if verdict == "reject" else 0
        m = self.E.master_call(t); man = self.manual_call(t); self.E.record(t, y)
        r = self.E.rule_for(t); self.log.append(dict(y=y, master=m))
        words = ("stop the pump", "keep it running") if self.kind == "ear" else ("call it a bad stroke", "call it a good stroke")
        fb = dict(correct=(y == m), your_call=verdict, master_call=words[0] if m else words[1], manual_call=words[0] if man else words[1],
                  manual_differs=(m != man), value=self.sense["val"](t))
        if self.kind == "ear": fb.update(shop=t["shop"], distractor=t["distractor"], level=round(100 * t["severity"]))
        else: fb.update(edge=t["edge"], edge_label=EDGE_LABEL[t["edge"]])
        if r: fb["rule"] = dict(text_en=r["text_en"], text_ja=r["text_ja"])
        return fb

    def state(self):
        prog = self.E.progress(); recent = self.log[-20:]
        return dict(kind=self.kind, ready=bool(self.rules), answers=len(self.log), absorbed=sum(p["status"] == "absorbed" for p in prog),
                    rules=prog, distance=[round(d, 3) for d in self.E.history],
                    correct_recent=sum(e["y"] == e["master"] for e in recent), recent_n=len(recent))


TRAINEES = {}


def sense_trainee(kind):
    T = TRAINEES.get(kind)
    if T is None or (not T.rules and not T.log): T = TRAINEES[kind] = SenseTrainee(kind)
    return T


@app.get("/api/{kind}/train/state")
def sense_train_state(kind: str):
    if kind not in ("ear", "hand"): raise HTTPException(404, "unknown sense")
    with LOCK: return sense_trainee(kind).state()


@app.get("/api/{kind}/train/next")
def sense_train_next(kind: str):
    if kind not in ("ear", "hand"): raise HTTPException(404, "unknown sense")
    with LOCK:
        T = sense_trainee(kind)
        if not T.rules: raise HTTPException(409, "No unwritten rules yet for this sense. Capture the master first.")
        t, why = T.E.next_theta(); cid = uuid.uuid4().hex[:10]; T.pending[cid] = t
        return dict(id=cid, why=why, n=len(T.log) + 1, **T.media(t))


@app.post("/api/{kind}/train/answer")
def sense_train_answer(kind: str, a: TrainAnswer):
    if kind not in ("ear", "hand"): raise HTTPException(404, "unknown sense")
    if a.verdict not in ("accept", "reject"): raise HTTPException(400, "verdict must be accept or reject")
    with LOCK:
        T = sense_trainee(kind)
        if a.id not in T.pending: raise HTTPException(404, "That item has expired. Load the next one.")
        fb = T.answer(a.id, a.verdict); return dict(feedback=fb, state=T.state())


@app.post("/api/{kind}/train/reset")
def sense_train_reset(kind: str):
    if kind not in ("ear", "hand"): raise HTTPException(404, "unknown sense")
    with LOCK:
        TRAINEES[kind] = SenseTrainee(kind); return TRAINEES[kind].state()


@app.post("/api/{kind}/train/autofill")
def sense_train_autofill(kind: str, n: int = 20, seed: int = 3):
    """Backup demo: a simulated junior (starts from the manual, reads the master's note) practises n items."""
    if kind not in ("ear", "hand"): raise HTTPException(404, "unknown sense")
    with LOCK:
        T = sense_trainee(kind); J = getattr(T, "_junior", None) or LimitJunior(T.sense, seed=seed); T._junior = J
        for _ in range(n):
            t, why = T.E.next_theta(); cid = uuid.uuid4().hex[:10]; T.pending[cid] = t
            y = J.answer(t); m = T.E.master_call(t); T.answer(cid, "reject" if y else "accept"); J.learn(t, m, y, T.E.rule_for(t))
        return T.state()


@app.get("/api/train/overview")
def train_overview():
    """Rules absorbed per sense, for the progress strip."""
    with LOCK:
        eye = trainee().state()
        out = dict(eye=dict(absorbed=eye["absorbed"], rules=len(eye["rules"]), answers=eye["answers"]))
        for k in ("ear", "hand"):
            st = sense_trainee(k).state(); out[k] = dict(absorbed=st["absorbed"], rules=len(st["rules"]), answers=st["answers"])
        return out


# ==================================================================== Skill Passport and plant overview
import datetime, hashlib
PASSPORT = dict(junior="Junior inspector", signoffs={})          # sense -> {date, rules}
SECONDS_PER_ANSWER = dict(eye=5, ear=8, hand=7)                   # rough estimates, labelled as such on the page


class NameIn(BaseModel):
    name: str


class SignIn(BaseModel):
    sense: str


def _passport():
    allr = _rules_all(audio=False)                               # master's rules per sense, with approval status
    eye_t = trainee(); ear_t = sense_trainee("ear"); hand_t = sense_trainee("hand")
    prog = {"eye": {p["id"]: p for p in eye_t.E.rule_progress()},
            "ear": {"ear:" + p["id"]: p for p in ear_t.E.progress()},
            "hand": {"hand:" + p["id"]: p for p in hand_t.E.progress()}}
    practised = {"eye": len(eye_t.log), "ear": len(ear_t.log), "hand": len(hand_t.log)}
    senses = {}
    for k in ("eye", "ear", "hand"):
        rules = [r for r in allr[k] if r["status"] != "rejected"]
        rows = []
        for r in rules:
            p = prog[k].get(r["id"])
            rows.append(dict(id=r["id"], text_en=r["text_en"], text_ja=r["text_ja"], approved=r["status"] == "approved",
                             absorbed=bool(p and p["status"] == "absorbed"), learning=bool(p and p["status"] == "learning"),
                             in_training=p is not None, checks=f"{p['probes_right']}/{p['probes']}" if p else None))
        absorbed = sum(x["absorbed"] for x in rows)
        shift = eye_t.shift_result if k == "eye" else None
        shift_ok = bool(shift and shift["trainee"]["missed"] == 0 and shift["trainee"]["false_alarms"] <= shift["manual_follower"]["false_alarms"])
        all_in = bool(rows) and absorbed == len(rows)
        certified = all_in and (shift_ok if k == "eye" else True)
        so = PASSPORT["signoffs"].get(k)
        senses[k] = dict(rules=rows, total=len(rows), approved=sum(x["approved"] for x in rows), absorbed=absorbed,
                         master_answers=allr["answers"][k], master_minutes=round(allr["answers"][k] * SECONDS_PER_ANSWER[k] / 60, 1),
                         practised=practised[k], shift=shift, shift_ok=shift_ok, certified=certified,
                         needs=("" if certified else ("absorb every rule" if not all_in else "pass a real shift with no bad part shipped")),
                         signoff=so, signoff_valid=bool(so and certified),
                         source=allr["sources"].get(k, "Real casting photos (Kaggle casting product dataset)"))
    total = sum(v["total"] for v in senses.values()); held = sum(v["absorbed"] for v in senses.values())
    code = hashlib.sha1(json.dumps([PASSPORT["junior"], [[r["id"], r["absorbed"]] for v in senses.values() for r in v["rules"]]]).encode()).hexdigest()[:10].upper()
    return dict(master=S.master_name, junior=PASSPORT["junior"], senses=senses, total=total, held=held, at_risk=total - held,
                pending_approval=sum(v["total"] - v["approved"] for v in senses.values()),
                master_minutes=round(sum(v["master_minutes"] for v in senses.values()), 1),
                certified=[k for k, v in senses.items() if v["certified"]], code=code,
                today=datetime.date.today().isoformat(), seconds_per_answer=SECONDS_PER_ANSWER)


@app.get("/api/passport")
def passport():
    with LOCK: return _passport()


@app.post("/api/passport/name")
def passport_name(n: NameIn):
    name = n.name.strip()[:60]
    if not name: raise HTTPException(400, "Enter a name")
    with LOCK:
        PASSPORT["junior"] = name; return _passport()


@app.post("/api/passport/signoff")
def passport_signoff(x: SignIn):
    if x.sense not in ("eye", "ear", "hand"): raise HTTPException(400, "unknown sense")
    with LOCK:
        p = _passport()
        if not p["senses"][x.sense]["certified"]:
            raise HTTPException(409, "This sense is not certified yet: the junior must " + p["senses"][x.sense]["needs"] + ".")
        PASSPORT["signoffs"][x.sense] = dict(date=datetime.date.today().isoformat(), by=S.master_name)
        return _passport()


# ==================================================================== The Inspection Line (game)
from game import Game, default_rules, ROLE_CARD, ZN as GAME_ZN
GAMES = {}                                   # player id -> that browser's game and master line
def _game():
    pid = PLAYER.get()
    if pid not in GAMES:
        if len(GAMES) > 300: GAMES.pop(next(iter(GAMES)))   # keep memory bounded on a public host
        GAMES[pid] = dict(g=None, master_rules=None)
    return GAMES[pid]


class GameStart(BaseModel):
    role: str
    quick: bool = True


class GameAnswer(BaseModel):
    verdict: str


@app.get("/master")
def master_page():
    return FileResponse(os.path.join(STATIC, "game.html"))


@app.get("/junior")
def junior_page():
    return FileResponse(os.path.join(STATIC, "dojo.html"))


@app.get("/game")
def game_page():
    return FileResponse(os.path.join(STATIC, "game.html"))


@app.post("/api/game/start")
def game_start(x: GameStart):
    if x.role not in ("master", "apprentice"): raise HTTPException(400, "role must be master or apprentice")
    with LOCK:
        GAME = _game()
        GAME["g"] = Game(G, png_data_url, x.role, master_rules=GAME["master_rules"], quick=x.quick)
        g = GAME["g"]
        return dict(part=g.part(), n=g.n, role=x.role,
                    master_rules=g.master_rules if x.role == "apprentice" else None,
                    rules_from=("your master round" if GAME["master_rules"] else "the role card") if x.role == "apprentice" else None,
                    role_card=[dict(zone=z, title_en=GAME_ZN[z][0], title_ja=GAME_ZN[z][1], kind=k, limit=v) for z, (k, v) in ROLE_CARD.items() if not (x.quick and z == "rim")])


@app.post("/api/game/answer")
def game_answer(a: GameAnswer):
    if a.verdict not in ("ok", "ng"): raise HTTPException(400, "verdict must be ok or ng")
    with LOCK:
        GAME = _game(); g = GAME["g"]
        if g is None or g.done: raise HTTPException(409, "Start a round first.")
        out = g.answer(1 if a.verdict == "ng" else 0)
        if g.done and g.role == "master":            # every finished master round replaces the last one, even with no secrets
            GAME["master_rules"] = g.rules; GAME["master_zones"] = list(g.zones)
        return out


# ==================================================================== The TACIT Dojo (技能道場)
# Three short stations built from TACIT's generators and the master's captured limits:
#   eye  - generated limit samples (限度見本): one part, the pit growing; drag where the master draws the line
#   ear  - exaggerate-then-fade: which of two pumps is faulty? the fault gets fainter after each right answer
#   hand - trace the stroke; the speed at the end is compared with the master's
import uuid
from ear import SR as SR_EAR
from rules import crop_around
DOJO = dict(eye={}, ear={})
EYE_SIZES = [1.5, 2.3, 3.1, 3.9, 4.6, 5.4, 6.2, 7.0]
DOJO_FALLBACK = dict(eye=dict(sealing_lip=2.0, flange_face=5.5), ear=0.30, hand=12.0)
DOJO_ZONES = ("sealing_lip", "flange_face")      # the same zones as the Inspection Line, so the dojo trains on the line just captured


def _dojo_limits():
    """The master's limits: from the live capture sessions once there is enough evidence, else the simulated master."""
    out, src = {}, {}
    GAME = _game()
    eye_n = sum(1 for e in S.log if not e["fake"])
    played = {r["zone"]: r for r in (GAME.get("master_rules") or [])}     # a master round just played in the Inspection Line wins
    tested = GAME.get("master_zones") or []
    for z in DOJO_ZONES:
        if z in played: out[z] = float(played[z]["limit"]); src[z] = f"your master round ({played[z]['evidence']} answers)"
        elif z in tested: out[z] = 4.0; src[z] = "your master round (you judged like the manual here)"
        elif eye_n >= 30: out[z] = round(S.twin.limit((z, "round", 1))[0], 1); src[z] = f"the master's {eye_n} answers"
        else: out[z] = DOJO_FALLBACK["eye"][z]; src[z] = "the simulated master"
    if len(ES.log) >= 20: ear = round(float(ES.twin.limit(("quiet", "none"))[0]), 2); ear_src = f"the master's {len(ES.log)} answers"
    else: ear, ear_src = DOJO_FALLBACK["ear"], "the simulated master"
    if len(HS.log) >= 15: hand = round(float(HS.twin.limit("sealing_lip")[0]), 1); hand_src = f"the master's {len(HS.log)} answers"
    else: hand, hand_src = DOJO_FALLBACK["hand"], "the simulated master"
    return dict(eye=out, eye_src=src, ear=ear, ear_src=ear_src, hand=hand, hand_src=hand_src, master=S.master_name)


@app.get("/dojo")
def dojo_page():
    return FileResponse(os.path.join(STATIC, "dojo.html"))


@app.get("/api/dojo/info")
def dojo_info():
    with LOCK: return _dojo_limits()


@app.get("/api/dojo/eye")
def dojo_eye(zone: str = "sealing_lip"):
    if zone not in DOJO_ZONES: raise HTTPException(400, "zone must be sealing_lip or flange_face")
    with LOCK:
        if zone not in DOJO["eye"]:
            base, seed = (7, 424242) if zone == "sealing_lip" else (19, 2024)
            imgs = []
            for s in EYE_SIZES:
                t = dict(defect="pit", zone=zone, shape="round", count=1, size_mm=s, contrast=1.15, base=base, seed=seed)
                img, info = G.render(t)
                imgs.append(png_data_url(crop_around(img, info, half=58, out=220)))
            DOJO["eye"][zone] = imgs
        lim = _dojo_limits()
        return dict(zone=zone, sizes=EYE_SIZES, images=DOJO["eye"][zone], manual=MANUAL.MAX_PIT_MM if hasattr(MANUAL, "MAX_PIT_MM") else 4.0,
                    master=lim["eye"][zone], source=lim["eye_src"][zone])


class EarTrialIn(BaseModel):
    level: float


class EarPickIn(BaseModel):
    id: str
    pick: str


@app.post("/api/dojo/ear/trial")
def dojo_ear_trial(x: EarTrialIn):
    lvl = float(np.clip(x.level, 0.02, 1.0))
    rng = np.random.default_rng()
    clip = int(rng.integers(0, GE.src.n_clips)); seed = int(rng.integers(1 << 30))
    n = int(SR_EAR * 2.5)
    good = GE.render(dict(severity=0.0, shop="quiet", distractor="none", clip=clip, seed=seed))[:n]
    bad = GE.render(dict(severity=lvl, shop="quiet", distractor="none", clip=clip, seed=seed))[:n]
    fade = np.minimum(1, np.minimum(np.arange(n), n - np.arange(n)) / (0.05 * SR_EAR)).astype(np.float32)   # no clicks
    faulty = "a" if rng.random() < 0.5 else "b"
    a, b = (bad, good) if faulty == "a" else (good, bad)
    tid = uuid.uuid4().hex[:10]
    with LOCK:
        DOJO["ear"][tid] = (faulty, lvl)
        if len(DOJO["ear"]) > 200: DOJO["ear"].pop(next(iter(DOJO["ear"])))
    return dict(id=tid, level=lvl, a=wav_data_url(a * fade), b=wav_data_url(b * fade))


@app.post("/api/dojo/ear/answer")
def dojo_ear_answer(x: EarPickIn):
    with LOCK:
        if x.id not in DOJO["ear"]: raise HTTPException(404, "That pair has expired. Play the next one.")
        faulty, lvl = DOJO["ear"].pop(x.id)
    return dict(correct=x.pick == faulty, faulty=faulty, level=lvl)


@app.get("/api/dojo/ear/lesson")
def dojo_ear_lesson():
    """For the generated ear lesson: one real recording, healthy and at the master's stop level, plus a strong fault for contrast."""
    lim = _dojo_limits()["ear"]
    rng = np.random.default_rng()
    clip = int(rng.integers(0, GE.src.n_clips)); seed = int(rng.integers(1 << 30)); n = int(SR_EAR * 2.5)
    fade = np.minimum(1, np.minimum(np.arange(n), n - np.arange(n)) / (0.05 * SR_EAR)).astype(np.float32)
    r = lambda sev: wav_data_url(GE.render(dict(severity=float(sev), shop="quiet", distractor="none", clip=clip, seed=seed))[:n] * fade)
    return dict(level=lim, healthy=r(0.0), master=r(lim), strong=r(min(1.0, max(.85, lim * 2.5))))

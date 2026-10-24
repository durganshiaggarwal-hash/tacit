"""Phase 3 — turn the Tacit Delta into rule cards the master approves, and group rejects by cause.

Rule cards
  For every situation where the Expert Twin is confident and differs from the Manual Twin, write a
  rule. Situations in the same zone with the same behaviour for both pit shapes are merged into one
  rule ("any pit"), otherwise each shape gets its own. Text is generated from templates in English and
  Japanese (deterministic: the card says exactly what the model learned, nothing more).

Cause groups
  The AI never knows *why* a part is bad. It only groups the master's rejects by the outline of the
  defect (roundness and edge roughness of the real pit used), then asks the master to name each group
  once. Shape-specific rules then carry that name ("likely cause: shrinkage").
"""
import numpy as np, cv2
from sklearn.cluster import KMeans
from twins import ManualTwin, CONTEXTS

ZONE_EN = {"sealing_lip": "Sealing lip", "flange_face": "Flange face", "rim": "Outer rim"}
ZONE_JA = {"sealing_lip": "シールリップ", "flange_face": "フランジ面", "rim": "外周リム"}
SHAPE_EN = {"round": "round, smooth pits", "irregular": "irregular, jagged pits", "any": "pits"}
SHAPE_JA = {"round": "丸く滑らかなピット", "irregular": "不規則なピット", "any": "ピット"}
CAUSE_JA = {"Gas porosity": "ガス欠陥（ブローホール）", "Shrinkage": "引け巣", "Sand inclusion": "砂かみ", "Other": "その他"}
CAUSE_OPTIONS = list(CAUSE_JA)
MIN_SIZE = 1.5           # smallest pit the generator makes; a limit below it means "reject any"
GAP_MM, SD_MAX, MERGE_MM = 0.7, 0.6, 0.6


def _fmt(x):
    return f"{x:.1f}"


def _text(zone, shape, count, limit, manual, cause=None):
    z_en, z_ja, s_en, s_ja = ZONE_EN[zone], ZONE_JA[zone], SHAPE_EN[shape], SHAPE_JA[shape]
    many_en = "two or more " if count == 2 else ""
    many_ja = "2個以上の" if count == 2 else ""
    if limit < manual:   # stricter than the manual
        if limit <= MIN_SIZE:
            en = f"{z_en}: reject {many_en}{s_en} of any size. The written standard allows up to {_fmt(manual)} mm."
            ja = f"{z_ja}：{many_ja}{s_ja}は大きさに関わらず不合格（標準書では{_fmt(manual)} mmまで許容）。"
        else:
            en = f"{z_en}: reject {many_en}{s_en} of {_fmt(limit)} mm or larger. The written standard allows up to {_fmt(manual)} mm."
            ja = f"{z_ja}：{many_ja}{s_ja}は{_fmt(limit)} mm以上で不合格（標準書では{_fmt(manual)} mmまで許容）。"
    else:                # more lenient
        en = f"{z_en}: {many_en}{s_en} up to {_fmt(limit)} mm are acceptable. The written standard rejects above {_fmt(manual)} mm."
        ja = f"{z_ja}：{many_ja}{s_ja}は{_fmt(limit)} mmまで合格（標準書では{_fmt(manual)} mm超は不合格）。"
    if cause:
        en += f" Likely cause: {cause.lower()}."
        ja += f" 推定原因：{CAUSE_JA.get(cause, cause)}。"
    return en, ja


def extract_rules(twin, log, cause_by_shape=None):
    """Return rule cards for every confident disagreement with the manual."""
    man = ManualTwin(); cause_by_shape = cause_by_shape or {}
    info = {}
    for (z, sh, n) in CONTEXTS:
        if n > man.MAX_COUNT: continue
        mu, sd = twin.limit((z, sh, n))
        manual = man.MAX_PIT_MM
        seen = sum(1 for e in log if not e["fake"] and e["theta"]["zone"] == z and e["theta"]["shape"] == sh
                   and min(e["theta"]["count"], 3) == n)
        tacit = sd < SD_MAX and abs(mu - manual) >= GAP_MM
        info[(z, sh, n)] = dict(limit=mu, sd=sd, manual=manual, seen=seen, tacit=tacit)
    cards = []
    for z in ZONE_EN:
        for n in (1, 2):
            r, i = info[(z, "round", n)], info[(z, "irregular", n)]
            both = r["tacit"] and i["tacit"] and abs(r["limit"] - i["limit"]) <= MERGE_MM and \
                (r["limit"] < r["manual"]) == (i["limit"] < i["manual"])
            groups = [("any", [r, i])] if both else [(s, [info[(z, s, n)]]) for s in ("round", "irregular") if info[(z, s, n)]["tacit"]]
            for shape, parts in groups:
                w = np.array([1 / max(p["sd"], 0.05) ** 2 for p in parts])
                limit = float((w * [p["limit"] for p in parts]).sum() / w.sum())
                sd = float(1 / np.sqrt(w.sum()))
                manual = parts[0]["manual"]
                cause = cause_by_shape.get(shape) if shape != "any" else None
                en, ja = _text(z, shape, n, limit, manual, cause)
                cid = f"{z}-{shape}-{n}"
                cards.append(dict(id=cid, zone=z, shape=shape, count=n, limit=round(limit, 2), sd=round(sd, 2),
                                  manual=manual, direction="stricter" if limit < manual else "more lenient",
                                  evidence=int(sum(p["seen"] for p in parts)), text_en=en, text_ja=ja, cause=cause))
    cards.sort(key=lambda c: (-abs(c["limit"] - c["manual"]), c["id"]))
    return cards


def example_thetas(card, G, seed=7):
    """One example the master would reject and one they would accept, either side of the learned limit."""
    shape = "round" if card["shape"] == "any" else card["shape"]
    lim, man = card["limit"], card["manual"]
    lo_cap, hi_cap = 1.5, 7.0
    rej = min(hi_cap, max(lo_cap, lim + 0.8)); acc = max(lo_cap, min(hi_cap, lim - 0.8))
    if card["direction"] == "stricter":   # the surprising case: manual accepts it, master rejects it
        rej = min(hi_cap, max(lo_cap, (lim + man) / 2))
    else:                                 # the surprising case: manual rejects it, master accepts it
        acc = max(lo_cap, min(hi_cap, (lim + man) / 2))
    rng = np.random.default_rng(seed)
    base = dict(zone=card["zone"], shape=shape, count=card["count"], contrast=1.1)
    t_rej = G.sample_theta(rng, size_mm=round(rej, 2), **base)
    if lim <= MIN_SIZE and card["count"] == 2:
        # "reject two pits of any size": the contrast is the same pit on its own, which the master accepts
        size = round(min(hi_cap, max(lo_cap, 2.0 if card["zone"] != "sealing_lip" else 1.6)), 2)
        t_rej = G.sample_theta(rng, size_mm=size, **base)
        t_acc = G.sample_theta(rng, size_mm=size, **dict(base, count=1))
        return [("Master rejects (2 pits)", t_rej), ("Master accepts (1 pit)", t_acc)]
    t_acc = G.sample_theta(rng, size_mm=round(acc, 2), **base)
    return [("Master rejects", t_rej), ("Master accepts", t_acc)]


def crop_around(img, info, half=44, out=176):
    if len(info["defects"]) > 1:      # several pits: show the whole part with every pit ringed
        c = cv2.cvtColor(cv2.resize(img, (out, out), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        k = out / img.shape[1]
        for d in info["defects"]:
            cv2.circle(c, (int(d["center"][0] * k), int(d["center"][1] * k)), 9, (40, 80, 230), 2, cv2.LINE_AA)
        return c
    d = info["defects"][0]; cx, cy = int(d["center"][0]), int(d["center"][1])
    h, w = img.shape[:2]
    x0, y0 = max(0, min(w - 2 * half, cx - half)), max(0, min(h - 2 * half, cy - half))
    c = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
    c = cv2.resize(c, (out, out), interpolation=cv2.INTER_CUBIC)
    c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
    # ring the defect
    px, py = (cx - x0) * out / (2 * half), (cy - y0) * out / (2 * half)
    r = max(10, d["size_mm"] / info["mm_per_px"] * out / (2 * half) * 0.9)
    cv2.circle(c, (int(px), int(py)), int(r), (40, 80, 230), 2, cv2.LINE_AA)
    return c


def cause_groups(log, library, k=2):
    """Group the master's rejects by the outline of the real pit used (roundness, edge roughness)."""
    lib = {m["id"]: m for m in library}
    rows = []
    for i, e in enumerate(log):
        if e["fake"] or not e["y"]: continue
        if e["theta"]["count"] > 2: continue           # 3+ pits are rejected for count, not shape
        p = lib.get(e["patches"][0]) if e["patches"] else None
        if p: rows.append((i, p["circularity"], p["roughness"], e["theta"]["shape"]))
    if len(rows) < 2 * k:
        return dict(ready=False, needed=2 * k, have=len(rows), groups=[])
    X = np.array([[r[1], r[2]] for r in rows]); mu, sd = X.mean(0), X.std(0) + 1e-6
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit((X - mu) / sd)
    groups = []
    for g in range(k):
        idx = [rows[j][0] for j in range(len(rows)) if km.labels_[j] == g]
        circ = float(X[km.labels_ == g, 0].mean()); rough = float(X[km.labels_ == g, 1].mean())
        shapes = [rows[j][3] for j in range(len(rows)) if km.labels_[j] == g]
        dominant = max(set(shapes), key=shapes.count)
        groups.append(dict(gid=g, members=idx, size=len(idx), circularity=round(circ, 2), roughness=round(rough, 3),
                           look="rounder, smoother outlines" if circ >= X[:, 0].mean() else "irregular, jagged outlines",
                           shape=dominant, purity=round(shapes.count(dominant) / len(shapes), 2)))
    groups.sort(key=lambda g: -g["circularity"])
    for n, g in enumerate(groups): g["gid"] = n
    return dict(ready=True, groups=groups)

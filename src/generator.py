"""Phase 1 — Eye generator: theta (a defect description) -> realistic impeller image.

Every image is built from real pixels: a clean OK part from the dataset plus a real
defect cut from a defective part. The defect is transferred *multiplicatively*
(its darkness relative to its own surroundings), so it inherits the target surface's
lighting instead of pasting a grey blob.

theta = {
  "defect":   "none" | "pit",
  "shape":    "round" | "irregular",            # round/smooth vs jagged (cause clusters)
  "zone":     "sealing_lip" | "flange_face" | "rim",
  "size_mm":  1.5 .. 7.0,                       # longest extent (real pits: ~2.5-6.7 mm at 120 mm OD)
  "contrast": 0.8 .. 1.25,                       # 1 = as it appeared on the real part
  "count":    1 .. 3,                           # pits in the same zone
  "base":     index into the 60 clean parts,
  "seed":     int (placement / patch choice)
}
"""
import os, sys, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from assets import Assets
from geometry import ZONES, point_at, mm_per_px, zone_of

ZONE_RANGE = {n: (lo, hi) for n, lo, hi in ZONES}
DEFECT_ZONES = ["sealing_lip", "flange_face", "rim"]
SHAPES = ["round", "irregular"]


class Generator:
    def __init__(self):
        self.A = Assets()
        self._base_cache = {}
        self._patch_cache = {}
        self.blocked = set()          # patch ids the master flagged as 'looks fake'
        self.pool = {(z, s): [m for m in self.A.library if m["shape"] == s] for z in DEFECT_ZONES for s in SHAPES}

    # ------------------------------------------------------------------ sampling
    def sample_theta(self, rng=None, **fixed):
        rng = rng or np.random.default_rng()
        t = dict(defect="pit",
                 shape=str(rng.choice(SHAPES)),
                 zone=str(rng.choice(DEFECT_ZONES)),
                 size_mm=float(np.round(np.exp(rng.uniform(np.log(1.5), np.log(7.0))), 2)),
                 contrast=float(np.round(rng.uniform(0.8, 1.25), 2)),
                 count=int(rng.choice([1, 1, 1, 2, 3])),
                 base=int(rng.integers(0, len(self.A.bases))),
                 seed=int(rng.integers(0, 2**31 - 1)))
        t.update(fixed)
        return t

    # ------------------------------------------------------------------ helpers
    def _base(self, i):
        if i not in self._base_cache:
            self._base_cache[i] = self.A.base(i)
        return self._base_cache[i]

    def _patch(self, pid):
        if pid not in self._patch_cache:
            p, m, meta = self.A.patch(pid)
            p = p.astype(np.float32)
            bg = cv2.medianBlur(p.astype(np.uint8), 15).astype(np.float32)
            # how much darker/brighter than its own surroundings each pixel is
            ratio = np.clip((p + 4) / (bg + 4), 0.05, 1.6)
            soft = cv2.GaussianBlur(cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32), (7, 7), 0)  # include the bright lip around the pit
            ys, xs = np.where(m)
            cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
            ext = max(np.ptp(xs), np.ptp(ys)) + 1
            self._patch_cache[pid] = (ratio, soft, (cx, cy), ext, meta)
        return self._patch_cache[pid]

    def _place(self, img, g, pid, rho, ang_deg, size_px, contrast, rot_deg):
        ratio, soft, (pcx, pcy), ext, meta = self._patch(pid)
        s = size_px / ext
        h, w = ratio.shape
        # affine: scale + rotate about the defect centre, translate onto the target point
        tx, ty = point_at(g, rho, ang_deg)
        M = cv2.getRotationMatrix2D((pcx, pcy), rot_deg, s)
        M[0, 2] += tx - pcx; M[1, 2] += ty - pcy
        H, W = img.shape
        r = cv2.warpAffine(ratio, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=1.0)
        a = cv2.warpAffine(soft, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0.0)
        a = np.clip(a, 0, 1)
        eff = np.power(r, contrast)                  # contrast scales depth in log space
        out = img * (1 - a + a * eff)
        x0, y0 = tx - size_px / 2, ty - size_px / 2
        return out, (float(x0), float(y0), float(size_px), float(size_px)), (float(tx), float(ty))

    # ------------------------------------------------------------------ render
    def render(self, theta):
        img, g, zones = self._base(theta["base"])
        out = img.astype(np.float32)
        info = dict(theta=dict(theta), defects=[], mm_per_px=mm_per_px(g))
        if theta.get("defect", "pit") == "none":
            return np.clip(out, 0, 255).astype(np.uint8), info
        rng = np.random.default_rng(theta["seed"])
        pool = self.pool[(theta["zone"], theta["shape"])]
        lo, hi = ZONE_RANGE[theta["zone"]]
        size_px = theta["size_mm"] / info["mm_per_px"]
        r_mean = (g["a"] + g["b"]) / 2
        margin = min(0.45 * (hi - lo), 0.5 * size_px / r_mean)    # keep the pit inside its zone
        for k in range(theta.get("count", 1)):
            usable = [m for m in pool if m["id"] not in self.blocked] or pool
            pid = usable[int(rng.integers(0, len(usable)))]["id"]
            rho = float(rng.uniform(lo + margin, hi - margin))
            ang = float(rng.uniform(0, 360))
            sz = size_px * (1.0 if k == 0 else float(rng.uniform(0.6, 1.0)))
            out, box, c = self._place(out, g, pid, rho, ang, sz, theta["contrast"], float(rng.uniform(0, 360)))
            info["defects"].append(dict(patch=pid, rho=round(rho, 3), angle=round(ang, 1), box=box, center=c,
                                        zone=zone_of(c[0], c[1], g)[0], size_mm=round(sz * info["mm_per_px"], 2)))
        # sensor grain so the pasted region has the same noise as the rest of the photo
        out += rng.normal(0, 1.2, out.shape).astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8), info


if __name__ == "__main__":
    G = Generator()
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    t = G.sample_theta(rng)
    im, info = G.render(t)
    print(json.dumps(info, indent=1))


# ---------------------------------------------------------------------- for Phase 2
FEATURE_NAMES = (["is_round"] + [f"zone_{z}" for z in DEFECT_ZONES] +
                 ["size_mm", "log_size", "contrast", "count"] +
                 [f"size_x_{z}" for z in DEFECT_ZONES] + ["round_x_size"] +
                 [f"round_x_{z}" for z in DEFECT_ZONES] + [f"count_x_{z}" for z in DEFECT_ZONES] +
                 [f"round_x_size_x_{z}" for z in DEFECT_ZONES])

def features(theta):
    """Numeric description of a defect, used by the Manual / Expert / Trainee twins."""
    if theta.get("defect") == "none":
        return np.zeros(len(FEATURE_NAMES), np.float32)
    z = [1.0 if theta["zone"] == q else 0.0 for q in DEFECT_ZONES]
    s = float(theta["size_mm"]); r = 1.0 if theta["shape"] == "round" else 0.0; n = float(theta["count"])
    return np.array([r] + z + [s, np.log(s), theta["contrast"], n] +
                    [s * q for q in z] + [r * s] + [r * q for q in z] + [n * q for q in z] +
                    [r * s * q for q in z], np.float32)

def export_gallery(n, out_dir, seed=0):
    """Render n examples to PNG + a manifest the frontend can load."""
    os.makedirs(out_dir, exist_ok=True)
    G = Generator(); rng = np.random.default_rng(seed); man = []
    for i in range(n):
        t = G.sample_theta(rng) if i % 6 else dict(G.sample_theta(rng), defect="none")
        im, info = G.render(t)
        f = f"{i:04d}.png"; cv2.imwrite(os.path.join(out_dir, f), im)
        man.append(dict(file=f, **info))
    json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    return man

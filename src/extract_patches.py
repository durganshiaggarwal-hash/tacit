"""Phase 0: build the real-defect patch library from def_front images.

Keeps compact, dark, high-contrast blobs found by the rotational-symmetry residual,
crops a square patch around each, and stores its local background so the generator
can transplant only the defect itself onto clean parts.
"""
import sys, os, json, glob, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from symmetry import candidates

BASE = 'data/casting_512x512/casting_512x512/'
OUT = 'assets/patches'

def compact(d):
    return 4 <= d['size_px'] <= 26 and d['dark'] and d['zone'] in ('flange_face', 'sealing_lip', 'rim') and d['strength'] > 9

def defect_mask(crop):
    """Mask of the defect inside a patch: pixels much darker than the patch's ring background."""
    g = cv2.GaussianBlur(crop, (3, 3), 0).astype(np.float32)
    bg = cv2.medianBlur(crop, 15).astype(np.float32)
    d = bg - g
    thr = max(18.0, np.percentile(d, 97) * 0.5)
    m = (d > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1: return None
    c = crop.shape[0] // 2
    best = min(range(1, n), key=lambda i: (lab[c-3:c+4, c-3:c+4] == i).sum() * -1000 + st[i, 4] * -1)
    m = (lab == best).astype(np.uint8)
    if m.sum() < 6: return None
    return cv2.dilate(m, np.ones((3, 3), np.uint8))

def main(limit=None):
    os.makedirs(OUT, exist_ok=True)
    paths = sorted(glob.glob(BASE + 'def_front/*'))[:limit]
    meta = []
    for p in paths:
        im = cv2.imread(p, 0)
        try: cands, g = candidates(im)
        except Exception: continue
        for d in [c for c in cands if compact(c)][:2]:
            half = int(max(14, d['size_px'] * 1.4))
            x, y = int(round(d['x'])), int(round(d['y']))
            if x - half < 0 or y - half < 0 or x + half >= im.shape[1] or y + half >= im.shape[0]: continue
            crop = im[y-half:y+half+1, x-half:x+half+1]
            m = defect_mask(crop)
            if m is None: continue
            pid = f"p{len(meta):04d}"
            cv2.imwrite(f"{OUT}/{pid}.png", crop); cv2.imwrite(f"{OUT}/{pid}_mask.png", m * 255)
            ys, xs = np.where(m)
            meta.append(dict(id=pid, src=os.path.basename(p), zone=d['zone'], rho=round(d['rho'], 3),
                             size_px=int(max(np.ptp(xs), np.ptp(ys)) + 1), strength=round(d['strength'], 1),
                             contrast=float(np.median(cv2.medianBlur(crop, 15)[m > 0]) - np.median(crop[m > 0]))))
    json.dump(meta, open(f"{OUT}/index.json", "w"), indent=1)
    print(len(meta), "patches from", len(paths), "images")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)

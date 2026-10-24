"""Pick clean, well-aligned OK parts as backgrounds for the generator.

Score = how closely the part's radial profile matches the calibrated median profile
(alignment) minus how much symmetry residual it has on the outer faces (cleanliness).
"""
import sys, os, glob, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from symmetry import residual, NR, RHO_MAX
from geometry import geometry

BASE = 'data/casting_512x512/casting_512x512/ok_front/'
REF = np.load('profile.npy')                      # 241 pts over rho 0..2.4
REF_RHO = np.linspace(0, 2.4, 241)

def score(path):
    g8 = cv2.imread(path, 0); g = geometry(g8)
    z, pol, rho, _ = residual(g8, g)
    prof = np.median(pol, 0)
    ref = np.interp(rho, REF_RHO, REF)
    sel = (rho > 0.5) & (rho < 1.8)
    align = float(np.corrcoef(prof[sel], ref[sel])[0, 1])
    band = (rho > 1.0) & (rho < 1.8)
    dirt = float(np.mean(np.abs(z[:, band]) > 6))
    inside = 0.15 * 512 < g["a"] < 0.3 * 512 and abs(g["cx"] - 256) < 60 and abs(g["cy"] - 256) < 60
    return dict(file=os.path.basename(path), align=align, dirt=dirt, ok=bool(inside), geom=g)

if __name__ == "__main__":
    rows = [score(p) for p in sorted(glob.glob(BASE + '*'))]
    rows = [r for r in rows if r["ok"]]
    for r in rows: r["score"] = r["align"] - 8 * r["dirt"]
    rows.sort(key=lambda r: -r["score"])
    os.makedirs("assets", exist_ok=True)
    json.dump(rows[:60], open("assets/bases.json", "w"), indent=1)
    print(len(rows), "usable; top align", round(rows[0]["align"], 3), "60th", round(rows[59]["align"], 3))

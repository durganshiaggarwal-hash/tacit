"""Defect localisation by rotational symmetry.

An impeller is rotationally symmetric, so at any radius the surface should look the
same all the way round. Unwrap to polar coordinates (using the tilt ellipse), subtract a
local along-the-ring median, and what is left over is a candidate defect.
"""
import cv2, numpy as np
from scipy.ndimage import median_filter
from geometry import geometry, zone_of, ZONES

NR, NA, RHO_MAX = 380, 1440, 1.90

def polar_maps(g):
    rho = np.linspace(0, RHO_MAX, NR, dtype=np.float32)
    ang = np.linspace(0, 2 * np.pi, NA, endpoint=False, dtype=np.float32)
    R, A = np.meshgrid(rho, ang)               # shape (NA, NR)
    u, v = R * g["a"] * np.cos(A), R * g["b"] * np.sin(A)
    t = np.deg2rad(g["theta"])
    mx = (g["cx"] + u * np.cos(t) - v * np.sin(t)).astype(np.float32)
    my = (g["cy"] + u * np.sin(t) + v * np.cos(t)).astype(np.float32)
    return mx, my, rho, ang

def unwrap(gray, g):
    mx, my, rho, ang = polar_maps(g)
    pol = cv2.remap(gray.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderValue=np.nan)
    return pol, rho, ang

def residual(gray, g):
    pol, rho, ang = unwrap(gray, g)
    pol = np.nan_to_num(pol, nan=float(np.nanmedian(pol)))
    # expected appearance: running median along the ring (handles slow lighting changes)
    exp = median_filter(pol, size=(121, 1), mode="wrap")
    res = pol - exp
    # robust per-radius noise level
    mad = np.median(np.abs(res - np.median(res, 0)), 0) * 1.4826 + 2.0
    return res / mad, pol, rho, ang

def candidates(img, rho_lo=1.0, rho_hi=1.86, z_thr=6.0, min_px=6):
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = geometry(gray)
    z, pol, rho, ang = residual(gray, g)
    band = (rho >= rho_lo) & (rho <= rho_hi)
    m = (np.abs(z) > z_thr) & band[None, :]
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8))
    n, lab, st, ce = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        if st[i, 4] < min_px: continue
        ys, xs = np.where(lab == i)
        strength = float(np.abs(z[ys, xs]).max())
        ai, ri = ce[i][1], ce[i][0]
        r = float(np.interp(ri, np.arange(NR), rho)); a = float(np.degrees(np.interp(ai, np.arange(NA), ang)))
        t = np.deg2rad(g["theta"]); aa = np.deg2rad(a)
        u, v = r * g["a"] * np.cos(aa), r * g["b"] * np.sin(aa)
        x = g["cx"] + u * np.cos(t) - v * np.sin(t); y = g["cy"] + u * np.sin(t) + v * np.cos(t)
        # extent in pixels (approx): radial bins * px per bin, angular bins * arc length
        rad_px = st[i, 2] * RHO_MAX / NR * (g["a"] + g["b"]) / 2
        ang_px = st[i, 3] * 2 * np.pi / NA * r * (g["a"] + g["b"]) / 2
        out.append(dict(x=float(x), y=float(y), rho=r, angle=a, zone=zone_of(x, y, g)[0],
                        strength=strength, area=int(st[i, 4]), size_px=float(max(rad_px, ang_px)),
                        dark=bool(np.median(z[ys, xs]) < 0)))
    out.sort(key=lambda d: -d["strength"] * np.sqrt(d["area"]))
    return out, g

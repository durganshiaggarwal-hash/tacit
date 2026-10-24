"""Impeller geometry, anchored on the dark blade cavity (the most reliable feature).

rho = elliptical radius in units of the cavity's outer radius (handles camera tilt).
Ring positions were calibrated from the median radial profile of 150 OK parts.
"""
import cv2, numpy as np

ZONES = [  # name, rho_min, rho_max
    ("bore",         0.00, 0.38),
    ("hub",          0.38, 0.54),
    ("blade_cavity", 0.54, 1.00),
    ("sealing_lip",  1.00, 1.22),
    ("flange_face",  1.22, 1.68),
    ("rim",          1.68, 1.84),
]
ZONE_NAMES = [z[0] for z in ZONES]
ZONE_LABEL = {"bore": "Bore", "hub": "Hub", "blade_cavity": "Blade cavity",
              "sealing_lip": "Sealing lip", "flange_face": "Flange face", "rim": "Outer rim"}

def geometry(img):
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    g = cv2.GaussianBlur(gray, (9, 9), 0)
    dark = (g < np.percentile(g, 12)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cs, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    def score(c):
        x, y, bw, bh = cv2.boundingRect(c)
        return cv2.contourArea(c) - 0.3 * ((x + bw / 2 - w / 2) ** 2 + (y + bh / 2 - h / 2) ** 2)
    c = max(cs, key=score)
    (cx, cy), (MA, ma), th = cv2.fitEllipse(c)
    return dict(cx=float(cx), cy=float(cy), a=float(MA / 2), b=float(ma / 2), theta=float(th))

def _uv(x, y, g):
    t = np.deg2rad(g["theta"]); dx, dy = x - g["cx"], y - g["cy"]
    return dx * np.cos(t) + dy * np.sin(t), -dx * np.sin(t) + dy * np.cos(t)

def rho_map(shape, g):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    u, v = _uv(xx, yy, g)
    return np.sqrt((u / g["a"]) ** 2 + (v / g["b"]) ** 2)

def zone_mask(shape, g):
    """int8 map: -1 = background, else index into ZONES."""
    rho = rho_map(shape, g)
    z = np.full(shape[:2], -1, np.int8)
    for i, (_, lo, hi) in enumerate(ZONES):
        z[(rho >= lo) & (rho < hi)] = i
    return z

def zone_of(x, y, g):
    u, v = _uv(x, y, g)
    rho = float(np.hypot(u / g["a"], v / g["b"]))
    for name, lo, hi in ZONES:
        if lo <= rho < hi: return name, rho
    return "background", rho

def point_at(g, rho, angle_deg):
    """Image coordinates for a given normalised radius and angle."""
    a = np.deg2rad(angle_deg); u, v = rho * g["a"] * np.cos(a), rho * g["b"] * np.sin(a)
    t = np.deg2rad(g["theta"])
    return g["cx"] + u * np.cos(t) - v * np.sin(t), g["cy"] + u * np.sin(t) + v * np.cos(t)

def mm_per_px(g, impeller_od_mm=120.0):
    """Approximate scale: assume the impeller's outer diameter (rho 1.84) is ~120 mm."""
    return impeller_od_mm / (2 * 1.84 * (g["a"] + g["b"]) / 2)

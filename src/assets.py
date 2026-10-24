"""Phase 0 output: one place to load base parts, zones and the real-defect library.

    from assets import Assets
    A = Assets()
    img, g, zones = A.base(0)          # clean OK part, its geometry, zone index map
    patch, mask, meta = A.patch("p0012")
    A.patches(zone="flange_face", type="pinhole")
"""
import os, json, cv2
from geometry import zone_mask, ZONES, ZONE_NAMES, ZONE_LABEL

ROOT = os.path.join(os.path.dirname(__file__), "..")
IMG_DIR = os.path.join(ROOT, "data/casting_512x512/casting_512x512")

class Assets:
    def __init__(self):
        self.bases = json.load(open(os.path.join(ROOT, "assets/bases.json")))
        self.library = json.load(open(os.path.join(ROOT, "assets/patches/library.json")))
        self._by_id = {m["id"]: m for m in self.library}

    def base(self, i):
        r = self.bases[i % len(self.bases)]
        bundled = os.path.join(ROOT, "assets", "base_images", r["file"])   # shipped with the app
        path = bundled if os.path.exists(bundled) else os.path.join(IMG_DIR, "ok_front", r["file"])
        img = cv2.imread(path, 0)
        if img is None:
            raise FileNotFoundError(f"Base image {r['file']} not found in assets/base_images or the dataset folder")
        return img, r["geom"], zone_mask(img.shape, r["geom"])

    def patch(self, pid):
        d = os.path.join(ROOT, "assets/patches")
        return (cv2.imread(f"{d}/{pid}.png", 0), cv2.imread(f"{d}/{pid}_mask.png", 0) > 0, self._by_id[pid])

    def patches(self, zone=None, type=None):
        return [m for m in self.library if (zone is None or m["zone"] == zone) and (type is None or m["type"] == type)]

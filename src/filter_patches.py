"""Keep compact defects (pinholes/blowholes); drop ring grooves and edges.

Rules: mask must not touch the patch border, aspect ratio <= 2.0, solidity >= 0.7.
Type: pinhole if max extent <= 9 px (at 512 px), else blowhole.
"""
import json, cv2, numpy as np
D = 'assets/patches'
meta = json.load(open(f'{D}/index.json'))
keep = []
for m in meta:
    k = cv2.imread(f"{D}/{m['id']}_mask.png", 0) > 0
    h, w = k.shape
    if k[0].any() or k[-1].any() or k[:, 0].any() or k[:, -1].any(): continue
    cs, _ = cv2.findContours(k.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 8: continue
    (_, _), (ew, eh), _ = cv2.minAreaRect(c)
    aspect = max(ew, eh) / max(1.0, min(ew, eh))
    solidity = area / max(1.0, cv2.contourArea(cv2.convexHull(c)))
    if aspect > 2.0 or solidity < 0.7: continue
    if m['contrast'] < 25: continue
    m.update(aspect=round(aspect, 2), solidity=round(solidity, 2), extent_px=int(max(ew, eh)),
             type='pinhole' if max(ew, eh) <= 9 else 'blowhole')
    keep.append(m)
json.dump(keep, open(f'{D}/library.json', 'w'), indent=1)
from collections import Counter
print(len(keep), 'kept of', len(meta), Counter(m['type'] for m in keep), Counter(m['zone'] for m in keep))
tiles = []
for m in keep[:120]:
    c = cv2.imread(f"{D}/{m['id']}.png", 0); k = cv2.imread(f"{D}/{m['id']}_mask.png", 0)
    c = cv2.cvtColor(cv2.resize(c, (64, 64), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    cs, _ = cv2.findContours(cv2.resize(k, (64, 64), interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(c, cs, -1, (0, 0, 255), 1)
    cv2.putText(c, m['type'][0].upper(), (2, 10), 0, 0.35, (0, 255, 0), 1)
    tiles.append(c)
while len(tiles) % 15: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite('library.jpg', np.vstack([np.hstack(tiles[i:i+15]) for i in range(0, len(tiles), 15)]))

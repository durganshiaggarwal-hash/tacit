import sys, cv2, numpy as np, glob, random
sys.path.insert(0, 'src')
from geometry import geometry, zone_mask, ZONES
cols = [(180,0,180),(0,180,255),(0,200,0),(0,0,255),(255,160,0),(255,0,0)]
def overlay(im):
    g = geometry(im); z = zone_mask(im.shape, g)
    tint = im.copy()
    for i in range(len(ZONES)): tint[z == i] = (0.55 * im[z == i] + 0.45 * np.array(cols[i])).astype(np.uint8)
    return tint
if __name__ == "__main__":
    random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
    base = 'data/casting_512x512/casting_512x512/'
    ps = random.sample(sorted(glob.glob(base+'ok_front/*')), 6) + random.sample(sorted(glob.glob(base+'def_front/*')), 6)
    out = [cv2.resize(overlay(cv2.imread(p)), (320, 320)) for p in ps]
    cv2.imwrite('zones.jpg', np.vstack([np.hstack(out[:6]), np.hstack(out[6:])]))

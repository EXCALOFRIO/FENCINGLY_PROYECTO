import sys, pickle, numpy as np
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import rutas as RU
def segmentos(B, pres, mask, l, U=0.75, G=8):
    ks = np.nonzero(mask & pres[:, l] & np.isfinite(B[:, l]).all(1))[0]
    if len(ks) < 2: return ks, []
    b = B[ks, l]; alto = b[:, 3] - b[:, 1]
    dp = np.abs(np.diff(b[:, 3])) / np.maximum(alto[:-1], 1e-6)
    dk = np.diff(ks)
    cortes = [i + 1 for i in range(len(dp)) if dk[i] <= G and dp[i] > U]
    segs = np.split(np.arange(len(ks)), cortes)
    return ks, segs
etq = dict(l.split() for l in open(sys.argv[1], encoding="utf-8") if l.strip())
for c, e in etq.items():
    with np.load(RU.TRABAJO / ("revision_%s.npz" % c), allow_pickle=True) as z:
        B, pres, mask = z["B"], z["pres"], z["mask"]
    out = []
    for l in (0, 1):
        ks, segs = segmentos(B, pres, mask, l)
        if len(segs) > 1:
            d = []
            for s in segs:
                b = B[ks[s], l]; h = np.median(b[:, 3] - b[:, 1]); y2 = np.median(b[:, 3])
                d.append("%d-%d(n%d,h%.2f,y%.2f)" % (ks[s[0]], ks[s[-1]], len(s), h, y2))
            out.append("%s:%s" % ("LR"[l], " | ".join(d)))
    print("%-38s %-9s %s" % (c, e, "  ".join(out)))

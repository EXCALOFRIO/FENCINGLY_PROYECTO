import sys, numpy as np
sys.path.insert(0, r"D:\FENCINGLY_SCRATCH\agente_saltos")
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import rutas as RU, regla as R
etq = dict(l.split() for l in open(sys.argv[1], encoding="utf-8") if l.strip())
for c, e in etq.items():
    with np.load(RU.TRABAJO / ("revision_%s.npz" % c), allow_pickle=True) as z:
        B, pres, mask = z["B"].copy(), z["pres"].copy(), z["mask"].copy()
    n, q = R.saltos_pie(B, pres, mask)
    m2 = R.recorta(pres, mask)
    rang = {}
    for l, k in q: rang.setdefault(l, []).append(k)
    txt = "  ".join("%s:%d-%d(n%d)" % ("LR"[l], min(v), max(v), len(v)) for l, v in rang.items())
    print("%-38s %-12s quita %3d  mask %d->%d  %s" % (c, e, n, mask.sum(), m2.sum(), txt))

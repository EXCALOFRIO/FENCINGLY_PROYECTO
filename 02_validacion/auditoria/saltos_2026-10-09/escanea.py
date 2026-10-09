"""Por clip, el PEOR salto del pie (y2) de cada lado entre cuadros presentes con hueco <= 8, en altos de caja.
Guarda tambien hueco, desplazamiento del centro en anchos, razon de altos y tipos. Solo revision_ (CPU)."""
import sys, glob, pathlib, pickle, numpy as np
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion")
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import rutas as RU, lotes as LO
ids = set(LO.conjunto("anotados"))
filas = []
for f in sorted(glob.glob(str(RU.TRABAJO / "revision_*.npz"))):
    c = pathlib.Path(f).stem[9:]
    if c not in ids: continue
    try:
        with np.load(f, allow_pickle=True) as z:
            B = z["B"]; m = z["mask"]; pres = z["pres"]; tipo = z["tipo"]; W = int(z["W"]); H = int(z["H"])
    except Exception as e:
        continue
    peor = None
    for l in (0, 1):
        ks = np.nonzero(m & pres[:, l] & np.isfinite(B[:, l]).all(1))[0]
        if len(ks) < 2: continue
        b = B[ks, l]
        alto = b[:, 3] - b[:, 1]; ancho = (b[:, 2] - b[:, 0]) * W
        dk = np.diff(ks)
        dp = np.abs(np.diff(b[:, 3])) / np.maximum(alto[:-1], 1e-6)
        cx = 0.5 * (b[:, 0] + b[:, 2]) * W; cy = 0.5 * (b[:, 1] + b[:, 3]) * H
        dc = np.hypot(np.diff(cx), np.diff(cy)) / np.maximum(ancho[:-1], 1.0)
        ra = alto[1:] / np.maximum(alto[:-1], 1e-6)
        ok = dk <= 8
        if not ok.any(): continue
        i = int(np.argmax(np.where(ok, dp, -1)))
        r = dict(lado=l, k0=int(ks[i]), k1=int(ks[i+1]), hueco=int(dk[i]), dpie=float(dp[i]), dcentro=float(dc[i]),
                 razon_alto=float(ra[i]), tipo0=int(tipo[ks[i], l]), tipo1=int(tipo[ks[i+1], l]))
        if peor is None or r["dpie"] > peor["dpie"]: peor = r
    filas.append((c, peor))
pickle.dump(filas, open(r"D:\FENCINGLY_SCRATCH\agente_saltos\saltos.pkl", "wb"))
v = np.array([x[1]["dpie"] if x[1] else 0.0 for x in filas])
print("clips", len(v), "p50 %.3f p90 %.3f p99 %.3f" % tuple(np.percentile(v, [50, 90, 99])))
for u in (0.3, 0.5, 0.75, 1.0): print("  > %.2f: %d" % (u, (v > u).sum()))
big = [x for x in filas if x[1] and x[1]["dpie"] > 0.75]
import collections
print("hueco de los >0.75:", collections.Counter(x[1]["hueco"] for x in big).most_common())
print("tipos (k0,k1):", collections.Counter((x[1]["tipo0"], x[1]["tipo1"]) for x in big).most_common())
print("dcentro p50 %.2f; >0.6*hueco: %d" % (np.median([x[1]["dcentro"] for x in big]), sum(x[1]["dcentro"] > 0.6 * x[1]["hueco"] for x in big)))

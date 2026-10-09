import pickle, glob, sys, numpy as np, collections
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import pista_lineas as PL
aj = set(l.strip() for l in open(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\auditoria\camara_tripode_2026-10-09\lista_cache_ajuste.txt", encoding="utf-8") if l.strip())
print("ajuste en cache:", len(aj))
filas = []
for c in aj:
    f = "D:/FENCINGLY_SCRATCH/agente_tripode/cache/%s.pkl" % c
    try: r = pickle.load(open(f, "rb"))
    except Exception: continue
    if "tq" not in r: continue
    tq = r["tq"]; far = r["far"].get(tq); near = r["near"].get(tq)
    # angulo del borde lejano observado (grados, 0 = horizontal)
    def ang(l):
        if l is None: return np.nan
        a = PL._ang(l)
        return a if a < 90 else a - 180
    p = r["prod"]
    pies = r["pies"][tq]; W = r["W"]
    u = np.nanmean(pies[:, 0]) if np.isfinite(pies).any() else np.nan
    filas.append(dict(clip=c, tor=r["torneo"], ok=p.get("ok", False), x=p.get("x", np.nan), far=ang(far), near=ang(near), u=(u - W/2)/W, alt=np.nanmean(r["alt"][tq])/r["H"]))
ok = [f for f in filas if f["ok"]]
print("con respuesta:", len(ok), " sin:", len(filas)-len(ok))
print("far visto en tq: con resp %.2f  sin resp %.2f" % (np.mean([np.isfinite(f["far"]) for f in ok]), np.mean([np.isfinite(f["far"]) for f in filas if not f["ok"]])))
print("near visto en tq: con resp %.2f  sin resp %.2f" % (np.mean([np.isfinite(f["near"]) for f in ok]), np.mean([np.isfinite(f["near"]) for f in filas if not f["ok"]])))
a = np.array([f["far"] for f in ok]); x = np.array([f["x"] for f in ok]); t = np.array([f["tor"] for f in ok])
m = np.isfinite(a) & np.isfinite(x)
print("pooled corr(far ang, x) = %.3f  n=%d" % (np.corrcoef(a[m], x[m])[0,1], m.sum()))
# dentro de torneo: centrar por torneo
ac, xc = [], []
for tt in set(t[m]):
    k = m & (t == tt)
    if k.sum() >= 3:
        ac += list(a[k] - a[k].mean()); xc += list(x[k] - x[k].mean())
print("dentro de torneo corr = %.3f  n=%d" % (np.corrcoef(ac, xc)[0,1], len(ac)))
for f in sorted(ok, key=lambda f: (f["tor"], f["x"]))[:40]:
    print("%-28s x=%6.2f far=%6.2f near=%6.2f u=%5.2f" % (f["tor"][:28], f["x"], f["far"], f["near"], f["u"]))
print()
from scipy.stats import spearmanr
por = collections.defaultdict(list)
for f in ok:
    if np.isfinite(f["far"]) and np.isfinite(f["x"]): por[f["tor"]].append(f)
for tt, L in sorted(por.items(), key=lambda kv: -len(kv[1])):
    if len(L) < 4: continue
    rho = spearmanr([f["far"] for f in L], [f["x"] for f in L]).correlation
    b = np.polyfit([f["x"] for f in L], [f["far"] for f in L], 1)
    print("%-30s n=%3d spearman %+.2f  pendiente %+.2f grados/m  ang(x=0) %+.2f" % (tt[:30], len(L), rho, b[0], b[1]))

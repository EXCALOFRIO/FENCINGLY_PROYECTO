import sys, numpy as np, collections
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import catalogo as CA, modelo as MO
rs = [r for r in MO.carga() if r["mitad"] == "AJUSTE"]
df = CA.manifiesto().set_index("clip_id")
por = collections.defaultdict(list)
for r in rs:
    if r["motivo"] == "ok" and r["ang"] is not None and r["x_prod"] is not None and r["desp"] is not None:
        por[r["torneo"]].append(r)
A, P, M, Fz, Fa = [], [], [], [], []
for t, refs in por.items():
    if len(refs) < 8: continue
    e = MO.loo(refs, True, True)
    for r, ei in zip(refs, e):
        if not np.isfinite(ei): continue
        y = df.loc[r["clip"], "strip_location"]
        A.append(CA.ZONAS[MO.zona(r["x_prod"] + ei)] == y); P.append(r["zona_prod"] == y); M.append(y == "middle")
A, P, M = map(np.array, (A, P, M))
print("AJUSTE, referencias (tienen homografia), n=%d: angulo (dejando uno fuera) %.3f   homografia de produccion %.3f   middle %.3f" % (len(A), A.mean(), P.mean(), M.mean()))
# en clips SIN homografia: zona del angulo segun el lado de x (signo) — ¿acierta al menos el lado?

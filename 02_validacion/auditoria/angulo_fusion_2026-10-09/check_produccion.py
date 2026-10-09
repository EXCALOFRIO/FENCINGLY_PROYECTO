"""1) La logica de produccion (angulo_lejano.x_angulo + decide) con los rasgos de rasgos.jsonl da las
MISMAS zonas que fusion.py D(2) e<=inf, clip a clip, en las dos mitades (y por tanto las mismas cifras).
2) angulo_lejano.rasgos (decodificando de nuevo) reproduce rasgos.jsonl en una muestra al azar."""
import sys, json, collections, random, csv
import numpy as np
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
sys.path.insert(0, r"D:\FENCINGLY_SCRATCH\agente_angulo")
import angulo_lejano as AL
import zona_tocado as ZT
import rutas as RU
import fusion as FU
import modelo as MO

rs = MO.carga("rasgos.jsonl")
crudos = {json.loads(l)["clip"]: json.loads(l) for l in open("rasgos.jsonl", encoding="utf-8")}
for mitad in ("AJUSTE", "PRUEBA"):
    R = [r for r in rs if r["mitad"] == mitad]
    xa_eval, _ = FU.predicciones_angulo(R)
    por = collections.defaultdict(list)
    filas = {}
    for r in R:
        f = {"clip": r["clip"], "ang": r["ang"], "desp": r["desp"], "x_h": r["x_prod"] if r["motivo"] == "ok" else None}
        filas[r["clip"]] = (r, f)
        if f["x_h"] is not None and f["ang"] is not None and f["desp"] is not None:
            por[r["torneo"]].append(f)
    dif_x = []
    dif_z = 0
    n = 0
    for c, (r, f) in filas.items():
        if f["x_h"] is None:
            continue
        xa = AL.x_angulo(por.get(r["torneo"], []), f)
        xf, via = AL.decide(f["x_h"], xa)
        ze = r["zona_prod"]
        if c in xa_eval:
            ze = ZT.ZONAS[MO.zona(FU.aplica("D(2)", r["x_prod"], xa_eval[c]))]
            dif_x.append(abs(xa - xa_eval[c]) if xa is not None else np.inf)
        elif xa is not None:
            dif_x.append(np.inf)
        n += 1
        dif_z += ZT.ZONAS[ZT.zona(xf)] != ze
    dif_x = np.array(dif_x)
    print("%s: clips que contestan %d; x del angulo: max |produccion - medida| %.2e (n=%d); zonas distintas %d"
          % (mitad, n, dif_x.max() if len(dif_x) else 0, len(dif_x), dif_z), flush=True)

# 2) rasgos decodificando de nuevo, muestra al azar de 40
cl = sorted(c for c, r in crudos.items() if r.get("motivo") in ("ok", "sinH") and r.get("fr"))
random.Random(20261009).shuffle(cl)
mal = 0
for c in cl[:40]:
    zr = next(csv.DictReader(open(RU.TRABAJO / ("zona_%s.csv" % c), encoding="utf-8")))
    a = AL.rasgos(c, zr)
    b = AL.resume(crudos[c]["fr"], crudos[c]["W"], crudos[c]["H"])
    igual = all((a[k] is None and b[k] is None) or (a[k] is not None and b[k] is not None and abs(float(a[k]) - float(b[k])) < 1e-6)
                for k in ("ang", "desp", "dos_pies"))
    mal += not igual
    if not igual:
        print("   DISTINTO", c, a, b)
print("rasgos decodificando de nuevo: %d de 40 distintos de rasgos.jsonl" % mal)

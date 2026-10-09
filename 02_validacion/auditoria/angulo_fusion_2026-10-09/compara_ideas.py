"""Cambio 1 de ANGULO_FUSION: las dos ideas pasan; ganancia de cada una en los MISMOS 1.000 clips de PRUEBA."""
import sys, json, csv
import numpy as np
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
sys.path.insert(0, r"D:\FENCINGLY_SCRATCH\agente_angulo")
import catalogo as CA, rutas as RU
import fusion as FU
import modelo as MO

ZONAS = CA.ZONAS
df = CA.manifiesto().set_index("clip_id")
rs = [r for r in MO.carga("rasgos.jsonl") if r["mitad"] == "PRUEBA"]
rsd = {r["clip"]: r for r in rs}
xa, eT = FU.predicciones_angulo(rs)
filas = [json.loads(l) for l in open("prior_prueba.jsonl", encoding="utf-8")]
P, I1, I2 = [], [], []
for f in filas:
    c = f["clip"]
    y = df.loc[c, "strip_location"]
    zr = next(csv.DictReader(open(RU.TRABAJO / ("zona_%s.csv" % c), encoding="utf-8")), {})
    zp = zr.get("zona") if zr.get("zona") in ZONAS else None
    z1 = zp
    r = rsd.get(c)
    if zp is not None and r is not None and c in xa:
        z1 = ZONAS[MO.zona(FU.aplica("D(2)", r["x_prod"], xa[c]))]
    res = f.get("res") if not f.get("sin_modelo") else None
    z2 = zp if not res else (res.get("P(1.5)") or {}).get("zona")
    P.append(zp == y); I1.append(z1 == y); I2.append(z2 == y)
P, I1, I2 = map(np.array, (P, I1, I2))
for nombre, X in (("idea 1, D(2)", I1), ("idea 2, P(1,5)", I2)):
    d = X.astype(float) - P
    lo, hi = FU.boot(d, nivel=97.5)
    print("%-16s en los mismos %d clips: produccion %.4f  nueva %.4f  %+.4f  IC97,5 [%+.4f, %+.4f]"
          % (nombre, len(P), P.mean(), X.mean(), d.mean(), lo, hi))
d = I1.astype(float) - I2
lo, hi = FU.boot(d, nivel=95.0)
print("idea 1 - idea 2: %+.4f  IC95 [%+.4f, %+.4f]" % (d.mean(), lo, hi))

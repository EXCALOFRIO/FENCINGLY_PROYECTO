import sys, json, csv, pathlib
import numpy as np
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import catalogo as CA, rutas as RU
from evalua import respuesta, boot, CANDIDATAS
res = pathlib.Path(sys.argv[1])
df = CA.manifiesto().set_index("clip_id")
falla = []
for f in sorted(res.glob("*.json")):
    r = json.loads(f.read_text(encoding="utf-8"))
    p = RU.TRABAJO / ("zona_%s.csv" % r["clip"])
    prod = next(csv.DictReader(open(p, encoding="utf-8")), {})
    if (prod.get("motivo") or "").startswith("sin homografia"):
        falla.append((r, df.loc[r["clip"], "strip_location"]))
def plaus(x):
    if x.get("x_izq") is None or x.get("x_der") is None: return False
    d = x["x_der"] - x["x_izq"]
    return 0.5 <= d <= 5.0 and abs(x["x_izq"]) <= 8 and abs(x["x_der"]) <= 8
print("%-14s %6s %7s %7s %7s  %s" % ("cand+plaus", "n", "zona", "middle", "dif", "IC95"))
for cand in CANDIDATAS:
    zs, ys = [], []
    for r, y in falla:
        x = respuesta(r, cand)
        if x and "zona" in x and plaus(x):
            zs.append(x["zona"]); ys.append(y)
    zs, ys = np.array(zs), np.array(ys)
    if len(zs) < 5: print("%-14s %6d" % ("%s(%s)" % cand, len(zs))); continue
    okz, okm = zs == ys, ys == "middle"; lo, hi = boot(okm, okz)
    print("%-14s %6d %7.3f %7.3f %+7.3f  [%+.3f, %+.3f]" % ("%s(%s)" % cand, len(zs), okz.mean(), okm.mean(), okz.mean()-okm.mean(), lo, hi))
# plausibilidad en los que YA contestan (produccion) para ver cuanto filtraria

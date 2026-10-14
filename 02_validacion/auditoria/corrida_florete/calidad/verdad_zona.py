"""Que posicion acierta la zona del tocado anotada a mano (`strip_location`): la de la etapa 7 (x de los
eventos alrededor del tocado) o la de la etapa 11 (`x_tocado_m`). Mismos clips: anotados de florete con
las dos. Mismo mapa x -> zona (`zona_tocado.zona`)."""
import csv, sys, pathlib
import numpy as np
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import catalogo as CA, rutas as RU, zona_tocado as ZT
df = CA.manifiesto().set_index("clip_id")
T, P = RU.TRABAJO, RU.PISTA
def filas(p):
    try:
        return list(csv.DictReader(open(p, encoding="utf-8")))
    except Exception:
        return None
ok7 = ok11 = n = 0; dif = []; tor = []
res = []
for zp in T.glob("zona_*.csv"):
    c = zp.name[5:-4]
    if c not in df.index or df.loc[c, "weapon"] != "foil" or not df.loc[c, "strip_location"]:
        continue
    z = filas(zp)
    if not z or z[0].get("motivo") != "ok":
        continue
    tf = int(z[0]["cuadro_usado"] or z[0]["cuadro_tocado"])
    se = filas(P / ("serie_eventos_%s.csv" % c))
    if not se:
        continue
    xs = []
    for lado in ("0", "1"):
        best = None
        for r in se:
            if r["lado"] != lado or not r.get("x_inicio_m") or not r.get("x_fin_m"):
                continue
            a, b = int(r["cuadro_inicio"]), int(r["cuadro_fin"])
            dist = 0 if a <= tf <= b else min(abs(tf - a), abs(tf - b))
            x = float(r["x_fin_m"]) if tf >= b else float(r["x_inicio_m"]) if tf <= a else (float(r["x_inicio_m"]) + float(r["x_fin_m"])) / 2
            if best is None or dist < best[0]:
                best = (dist, x)
        if best is not None and best[0] <= 15:
            xs.append(best[1])
    if not xs:
        continue
    x7 = float(np.mean(xs)); x11 = float(z[0]["x_tocado_m"])
    v = df.loc[c, "strip_location"]
    z7, z11 = ZT.ZONAS[ZT.zona(x7)], z[0]["zona"]
    n += 1; ok7 += z7 == v; ok11 += z11 == v; dif.append(abs(x7 - x11)); tor.append(df.loc[c, "torneo"])
    res.append((z7 == v, z11 == v))
a = np.array(res, float)
print("clips con las dos y zona anotada:", n)
print("acierto de la zona: etapa 7 %.3f   etapa 11 %.3f   (siempre 'middle': %.3f)" % (a[:, 0].mean(), a[:, 1].mean(),
      np.mean([df.loc[c, "strip_location"] == "middle" for c in [zp.name[5:-4] for zp in T.glob("zona_*.csv")] if c in df.index and df.loc[c, "weapon"] == "foil" and df.loc[c, "strip_location"]])))
print("|x7 - x11| en el tocado: p50 %.2f m  p90 %.2f m" % (np.median(dif), np.percentile(dif, 90)))
tor = np.array(tor); u = np.unique(tor); rng = np.random.default_rng(0); ds = []
for _ in range(2000):
    m = np.concatenate([np.flatnonzero(tor == t) for t in rng.choice(u, len(u))])
    ds.append(a[m, 1].mean() - a[m, 0].mean())
print("etapa 11 - etapa 7: %+.3f  IC95 por torneo [%+.3f, %+.3f]" % (a[:, 1].mean() - a[:, 0].mean(), *np.percentile(ds, [2.5, 97.5])))

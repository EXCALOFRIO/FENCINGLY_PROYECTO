"""Idea 2 de ANGULO_FUSION: acierto TOTAL de zona en la muestra, P(s) contra produccion.
    python evalua_prior.py prior_ajuste.jsonl AJUSTE           (las dos s, elige)
    python evalua_prior.py prior_prueba.jsonl PRUEBA "P(3)"     (una, IC97,5)
"""
import sys, csv, json
import numpy as np
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import catalogo as CA, rutas as RU
from fusion import boot, boot_torneo

ZONAS = CA.ZONAS


def main():
    sal, mitad = sys.argv[1], sys.argv[2]
    claves = ["P(1.5)", "P(3)"] if len(sys.argv) < 4 else [sys.argv[3]]
    nivel = 95.0 if mitad == "AJUSTE" else 97.5
    df = CA.manifiesto().set_index("clip_id")
    filas = [json.loads(l) for l in open(sal, encoding="utf-8")]
    errores = sum("error" in f for f in filas)
    filas = [f for f in filas if "error" not in f]
    base_mal = 0
    R = {k: {"p": [], "n": [], "arma": [], "tor": [], "nuevos": 0, "cambia": 0} for k in claves}
    for f in filas:
        c = f["clip"]
        y = df.loc[c, "strip_location"]
        zr = next(csv.DictReader(open(RU.TRABAJO / ("zona_%s.csv" % c), encoding="utf-8")), {})
        zp = zr.get("zona") if zr.get("zona") in ZONAS else None
        res = f.get("res", {}) if not f.get("sin_modelo") else None
        if res is not None and res:
            zb = (res.get("base") or {}).get("zona")
            base_mal += zb != zp
        for k in claves:
            zn = zp if res is None or not res else ((res.get(k) or {}).get("zona"))
            R[k]["p"].append(zp == y); R[k]["n"].append(zn == y)
            R[k]["arma"].append(df.loc[c, "weapon"]); R[k]["tor"].append(df.loc[c, "torneo"])
            R[k]["nuevos"] += zp is None and zn is not None
            R[k]["cambia"] += zn != zp
    print("%s: %d clips en la muestra (errores %d); el control 'base' NO reproduce el disco en %d"
          % (mitad, len(filas), errores, base_mal))
    best = None
    for k in claves:
        p, n = np.array(R[k]["p"]), np.array(R[k]["n"])
        d = n.astype(float) - p
        lo, hi = boot(d, nivel=nivel)
        lt, ht = boot_torneo(d, np.array(R[k]["tor"]), nivel=nivel)
        print("  %-7s cambian %4d (nuevos que contestan %d)  produccion %.4f  nuevo %.4f  dif %+.4f  IC%.1f clip [%+.4f, %+.4f]  torneo [%+.4f, %+.4f]"
              % (k, R[k]["cambia"], R[k]["nuevos"], p.mean(), n.mean(), d.mean(), nivel, lo, hi, lt, ht))
        ar = np.array(R[k]["arma"])
        for a in sorted(set(ar)):
            m = ar == a
            print("      arma %-6s n=%4d  produccion %.3f  nuevo %.3f  %+.3f" % (a, m.sum(), p[m].mean(), n[m].mean(), d[m].mean()))
        if best is None or (n.mean(), -R[k]["cambia"]) > (best[1], -best[2]):
            best = (k, n.mean(), R[k]["cambia"], lo)
    if len(claves) > 1:
        print("ELEGIDA en %s: %s -> %s" % (mitad, best[0], "va a PRUEBA" if best[3] > 0 else "NO va a PRUEBA (IC cruza cero)"))


if __name__ == "__main__":
    main()

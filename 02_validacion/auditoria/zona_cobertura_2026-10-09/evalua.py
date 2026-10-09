"""Evalua las reglas de respaldo sobre los JSON de exp_zona.py contra strip_location.

    python evalua.py res_dir [lista_ok_y_falla...]
"""
import csv
import json
import pathlib
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import catalogo as CA  # noqa: E402
import rutas as RU  # noqa: E402

CANDIDATAS = [("A", c) for c in ("0.25", "0.15", "0.05", "0.0")] + \
             [("B", c) for c in ("0.35", "0.25", "0.15", "0.05", "0.0")] + \
             [("B0.35+A", c) for c in ("0.25", "0.15", "0.05", "0.0")]


def respuesta(r, cand):
    tipo, c = cand
    if "a" not in r:
        return None
    if tipo == "A":
        return r["a"]["reglas"].get(c)
    if tipo == "B":
        return r["b"]["reglas"].get(c)
    x = r["b"]["reglas"].get("0.35")
    if x and "zona" in x:
        return x
    return r["a"]["reglas"].get(c)


def boot(ok_a, ok_b, n=2000, semilla=20261009):
    rng = np.random.default_rng(semilla)
    d = ok_b.astype(float) - ok_a.astype(float)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    res = pathlib.Path(sys.argv[1])
    df = CA.manifiesto().set_index("clip_id")
    filas = []
    for f in sorted(res.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        c = r["clip"]
        zc = RU.TRABAJO / ("zona_%s.csv" % c)
        prod = next(csv.DictReader(open(zc, encoding="utf-8")), {}) if zc.is_file() else {}
        filas.append((r, df.loc[c, "strip_location"], df.loc[c, "torneo"], prod))
    print("clips evaluados:", len(filas))
    # 1) reproduccion: la regla estricta A(0.35) contra la produccion
    rep = Counter()
    for r, y, t, prod in filas:
        a = r.get("a", {}).get("reglas", {}).get("0.35") if "a" in r else None
        za = a.get("zona") if a else None
        zp = prod.get("zona") or None
        rep["igual" if za == zp else "distinto"] += 1
        if za != zp and rep["distinto"] <= 5:
            print("   distinto", r["clip"], za, zp, prod.get("motivo"))
    print("reproduccion A(0.35) contra zona_ de disco:", dict(rep))
    falla = [(r, y, t) for r, y, t, prod in filas if (prod.get("motivo") or "").startswith("sin homografia")]
    ok = [(r, y, t, prod) for r, y, t, prod in filas if prod.get("motivo") == "ok"]
    print("sin homografia en produccion:", len(falla), " contestan:", len(ok))
    print()
    print("%-14s %6s %6s %7s %7s %7s  %s" % ("candidata", "nuevos", "cob", "zona", "middle", "dif", "IC95 por clip"))
    tabla = []
    for cand in CANDIDATAS:
        zs, ys = [], []
        for r, y, t in falla:
            x = respuesta(r, cand)
            if x and "zona" in x:
                zs.append(x["zona"])
                ys.append(y)
        zs, ys = np.array(zs), np.array(ys)
        if not len(zs):
            print("%-14s %6d" % ("%s(%s)" % cand, 0))
            continue
        okz, okm = zs == ys, ys == "middle"
        lo, hi = boot(okm, okz)
        tabla.append((cand, len(zs), okz.mean(), okm.mean(), lo, hi))
        print("%-14s %6d %6.3f %7.3f %7.3f %+7.3f  [%+.3f, %+.3f]"
              % ("%s(%s)" % cand, len(zs), len(zs) / len(falla), okz.mean(), okm.mean(),
                 okz.mean() - okm.mean(), lo, hi))
    print()
    print("caracterizacion de los sin homografia (calibracion de produccion, hasta tf+3):")
    car = Counter()
    for r, y, t in falla:
        if "a" not in r:
            car[r.get("motivo", "?")] += 1
            continue
        a, b, k = r["a"], r["b"], r["car"]
        if a["n_fin"] == 0:
            tipo = "sin ninguna H en el clip hasta el tocado"
            if k["trans2_clip"] == 0:
                tipo += " / nunca 2 rayas transversales"
            elif k["bordes_ambos_clip"] == 0:
                tipo += " / nunca los dos bordes"
            else:
                tipo += " / rayas pero sin semilla"
        elif a["fin_en_ventana"] == 0:
            tipo = "H en el clip pero NO en la ventana (cadena rota: corte o flujo)"
        else:
            tipo = "H en la ventana con confianza < 0,35"
        car[tipo] += 1
        car["   ... y con calibracion hasta el final SI contesta (B0.35)" if (b["reglas"].get("0.35") or {}).get("zona") else "   ... y B0.35 tampoco"] += 1
    for k, v in sorted(car.items(), key=lambda z: (z[0].startswith(" "), -z[1])):
        print("   %4d  %.3f  %s" % (v, v / len(falla), k))


if __name__ == "__main__":
    main()

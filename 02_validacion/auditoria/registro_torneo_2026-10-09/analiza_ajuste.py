"""PASO 1 (AJUSTE, SIN etiquetas): analiza `empareja_ajuste.jsonl.gz` y elige el umbral de inliers.

Para cada clip de AJUSTE con homografia de produccion (`x_prod`), se registra contra las demas
referencias del MISMO torneo (dejando el propio clip fuera, ya hecho en `registro_torneo.py
empareja`). Aqui se barre el umbral minimo de inliers (y de acuerdo entre referencias) y se mide,
SOLO contra `x_prod` (la x de la homografia estricta, NUNCA `strip_location`):

    fraccion registrada = clips con >=1 referencia que pasa el umbral / total
    error = |mediana de las x de las referencias que pasan - x_prod|, mediana y p90, por torneo

    python analiza_ajuste.py empareja_ajuste.jsonl.gz
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict

import numpy as np


def carga(path):
    abre = gzip.open if path.endswith(".gz") else open
    filas = []
    with abre(path, "rt", encoding="utf-8") as fh:
        for ln in fh:
            filas.append(json.loads(ln))
    return filas


def registra(fila, min_inliers, min_matches=0, min_acuerdo=None):
    """QUE DEVUELVE: (x_mediana, n_usadas, dispersion_m) o (None, 0, None) si ninguna referencia
    pasa el umbral. `min_acuerdo`: si se da, se exige que al menos esa fraccion de las referencias
    que pasan el umbral de inliers esten a <= 1.5 m de la mediana provisional (un segundo filtro de
    consenso, no solo de inliers por pareja)."""
    cand = [r for r in fila["resultados"]
            if r["x"] is not None and r["ninliers"] >= min_inliers and r["nmatches"] >= min_matches]
    if not cand:
        return None, 0, None
    xs = np.array([r["x"] for r in cand])
    x0 = float(np.median(xs))
    if min_acuerdo is not None:
        de_acuerdo = np.abs(xs - x0) <= 1.5
        if de_acuerdo.mean() < min_acuerdo:
            return None, len(cand), None
        xs = xs[de_acuerdo]
        x0 = float(np.median(xs))
    disp = float(np.median(np.abs(xs - x0))) if len(xs) > 1 else 0.0
    return x0, len(cand), disp


def resume(filas, min_inliers, min_matches=0, min_acuerdo=None):
    reg = 0
    errs = []
    por_torneo = defaultdict(list)
    for f in filas:
        if f.get("x_prod") is None:
            continue
        x0, n, disp = registra(f, min_inliers, min_matches, min_acuerdo)
        if x0 is None:
            continue
        reg += 1
        err = abs(x0 - f["x_prod"])
        errs.append(err)
        por_torneo[f["torneo"]].append(err)
    total = sum(1 for f in filas if f.get("x_prod") is not None)
    errs = np.array(errs)
    out = {
        "min_inliers": min_inliers, "min_matches": min_matches, "min_acuerdo": min_acuerdo,
        "total": total, "registrados": reg,
        "frac_registrada": reg / total if total else None,
        "mediana_err_m": float(np.median(errs)) if len(errs) else None,
        "p90_err_m": float(np.percentile(errs, 90)) if len(errs) else None,
        "menor_1m": float((errs < 1.0).mean()) if len(errs) else None,
        "por_torneo": {t: {"n": len(v), "mediana_err_m": float(np.median(v)), "p90_err_m": float(np.percentile(v, 90))}
                       for t, v in sorted(por_torneo.items())},
    }
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "empareja_ajuste.jsonl.gz"
    filas = carga(path)
    con_xprod = [f for f in filas if f.get("x_prod") is not None]
    print("filas: %d (con x_prod: %d)" % (len(filas), len(con_xprod)))
    n_con_resultados = sum(1 for f in con_xprod if f["resultados"])
    print("con >=1 intento de emparejamiento: %d" % n_con_resultados)
    todos_ninl = [r["ninliers"] for f in con_xprod for r in f["resultados"]]
    print("pares intentados: %d; ninliers mediana=%.1f p90=%.1f max=%d"
          % (len(todos_ninl), np.median(todos_ninl) if todos_ninl else -1,
             np.percentile(todos_ninl, 90) if todos_ninl else -1,
             max(todos_ninl) if todos_ninl else -1))
    print()
    print("%-26s %6s %6s %8s %10s %9s %9s" %
          ("umbral", "n_tot", "n_reg", "frac_reg", "med_err_m", "p90_err_m", "frac<1m"))
    mejores = []
    for mi in (0, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50):
        r = resume(con_xprod, mi, min_matches=0)
        mejores.append(r)
        print("%-26s %6d %6d %8s %10s %9s %9s" % (
            "inliers>=%d" % mi, r["total"], r["registrados"],
            "%.3f" % r["frac_registrada"] if r["frac_registrada"] is not None else "-",
            "%.3f" % r["mediana_err_m"] if r["mediana_err_m"] is not None else "-",
            "%.3f" % r["p90_err_m"] if r["p90_err_m"] is not None else "-",
            "%.3f" % r["menor_1m"] if r["menor_1m"] is not None else "-"))
    print()
    print("con acuerdo (inliers>=15, exige >=0.5 de las referencias de acuerdo a <=1.5 m):")
    for ma in (None, 0.5, 0.67, 0.8):
        r = resume(con_xprod, 15, min_matches=0, min_acuerdo=ma)
        print("  acuerdo>=%s: n_reg=%d frac_reg=%s med_err=%s p90_err=%s" % (
            ma, r["registrados"],
            "%.3f" % r["frac_registrada"] if r["frac_registrada"] is not None else "-",
            "%.3f" % r["mediana_err_m"] if r["mediana_err_m"] is not None else "-",
            "%.3f" % r["p90_err_m"] if r["p90_err_m"] is not None else "-"))
    print()
    print("### Por torneo, con el umbral inliers>=15 (candidato declarado) ###")
    r15 = resume(con_xprod, 15, min_matches=0)
    for t, d in sorted(r15["por_torneo"].items(), key=lambda z: -z[1]["n"]):
        print("   %-26s n=%3d  mediana|err|=%.2f m  p90=%.2f m" % (t, d["n"], d["mediana_err_m"], d["p90_err_m"]))
    with open(path.replace(".jsonl.gz", "_resumen.json").replace(".jsonl", "_resumen.json"),
              "w", encoding="utf-8") as fh:
        json.dump({"barrido": mejores, "por_torneo_inliers15": r15}, fh, indent=1)


if __name__ == "__main__":
    main()

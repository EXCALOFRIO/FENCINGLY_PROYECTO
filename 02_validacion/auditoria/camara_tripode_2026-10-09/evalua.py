"""Evalua las reglas de aceptacion del tripode sobre un jsonl de `fit.py`.

    python evalua.py FIT.jsonl LISTA.txt              elige la regla (AJUSTE)
    python evalua.py FIT.jsonl LISTA.txt --regla m,k,c3,p   evalua UNA regla (PRUEBA)
"""
import json
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import catalogo as CA  # noqa: E402

ZONAS = CA.ZONAS
LINEAS = (-7.0, -5.0, -2.0, 0.0, 2.0, 5.0, 7.0)


def boot(ok_a, ok_b, n=2000, semilla=20261009):
    rng = np.random.default_rng(semilla)
    d = ok_b.astype(float) - ok_a.astype(float)
    if not len(d):
        return float("nan"), float("nan")
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def c3(rayas):
    idx = sorted(LINEAS.index(r) for r in rayas)
    return any(idx[i + 1] - idx[i] == 1 and idx[i + 2] - idx[i + 1] == 1 for i in range(len(idx) - 2))


def pasa(o, m, k, con_c3, con_p):
    if o.get("estado") != "ok":
        return False
    mg = o.get("t_margen")
    if m > 0 and (mg is None or mg < m):
        return False
    if len(o.get("t_rayas") or []) < k:
        return False
    if con_c3 and not c3(o["t_rayas"]):
        return False
    if con_p:
        xs = o.get("t_xs") or [None, None]
        if xs[0] is None or xs[1] is None:
            return False
        d = xs[1] - xs[0]
        if not (0.5 <= d <= 5.0) or abs(xs[0]) > 8 or abs(xs[1]) > 8:
            return False
    return True


def reglas():
    for m in (0, 5, 10, 20, 40):
        for k in (1, 2, 3):
            for cc in (False, True):
                for pp in (False, True):
                    yield (m, k, cc, pp)


def nombre(r):
    m, k, cc, pp = r
    return "R(%d,%d)%s%s" % (m, k, "+c3" if cc else "", "+p" if pp else "")


def una(res, y, falla, contestan, r):
    nuevos = [c for c in falla if pasa(res[c], *r) and y.get(c) in ZONAS]
    ok_t = np.array([res[c]["t_zona"] == y[c] for c in nuevos])
    ok_m = np.array([y[c] == "middle" for c in nuevos])
    ref = [c for c in contestan if pasa(res[c], *r)]
    err = np.array([abs(res[c]["t_x"] - res[c]["prod_x"]) for c in ref])
    return nuevos, ok_t, ok_m, ref, err


def main():
    fitf, lista = sys.argv[1], sys.argv[2]
    clips_lista = [l.strip() for l in open(lista, encoding="utf-8") if l.strip()]
    res = {}
    for l in open(fitf, encoding="utf-8"):
        o = json.loads(l)
        res[o["clip"]] = o
    df = CA.manifiesto().set_index("clip_id")
    y = {c: df.loc[c, "strip_location"] for c in res if c in df.index}
    en = [c for c in clips_lista if c in res]
    falla = [c for c in en if not res[c].get("prod_ok")]
    contestan = [c for c in en if res[c].get("prod_ok")]
    print("lista %d; con resultado %d; NO contestan en produccion (reproducido) %d; contestan %d"
          % (len(clips_lista), len(en), len(falla), len(contestan)))
    print("estados en los que no contestan:", {e: sum(res[c].get("estado") == e for c in falla)
                                                for e in sorted({res[c].get("estado") for c in falla})})
    if "--regla" not in sys.argv:
        filas = []
        for r in reglas():
            nuevos, ok_t, ok_m, ref, err = una(res, y, falla, contestan, r)
            if not nuevos:
                continue
            filas.append((r, len(nuevos), ok_t.mean(), ok_m.mean(), ok_t.mean() - ok_m.mean(),
                          len(ref), float(np.median(err)) if len(err) else float("nan"),
                          float((err < 1).mean()) if len(err) else float("nan")))
        print("%-18s %6s %6s %6s %7s | %5s %7s %6s" % ("regla", "nuevos", "zona", "middle", "dif",
                                                       "n_ref", "med_err", "<1m"))
        for f in filas:
            print("%-18s %6d %6.3f %6.3f %+7.3f | %5d %7.2f %6.2f" % ((nombre(f[0]),) + f[1:]))
        buenas = [f for f in filas if f[4] >= 0.05]
        if buenas:
            best = max(buenas, key=lambda f: (f[1], f[4]))
            nuevos, ok_t, ok_m, ref, err = una(res, y, falla, contestan, best[0])
            lo, hi = boot(ok_m, ok_t)
            print("\nELEGIDA: %s  nuevos %d  zona %.3f middle %.3f dif %+.3f IC95 [%+.3f, %+.3f]"
                  % (nombre(best[0]), best[1], best[2], best[3], best[4], lo, hi))
        else:
            print("\nNINGUNA regla bate a middle por +0,05 en AJUSTE")
        return
    txt = sys.argv[sys.argv.index("--regla") + 1].split(",")
    r = (int(txt[0]), int(txt[1]), txt[2] == "1", txt[3] == "1")
    nuevos, ok_t, ok_m, ref, err = una(res, y, falla, contestan, r)
    lo, hi = boot(ok_m, ok_t)
    print("regla %s: nuevos %d de %d sin homografia (cobertura %.3f)  zona %.3f  middle %.3f  "
          "dif %+.3f  IC95 por clip [%+.3f, %+.3f]  -> (a) %s"
          % (nombre(r), len(nuevos), len(falla), len(nuevos) / max(len(falla), 1),
             ok_t.mean() if len(ok_t) else float("nan"), ok_m.mean() if len(ok_m) else float("nan"),
             (ok_t.mean() - ok_m.mean()) if len(ok_t) else float("nan"), lo, hi,
             "PASA" if lo > 0 else "NO PASA"))
    if len(err):
        print("   referencia en metros (clips que contestan): n=%d  mediana |err| %.2f m  <1 m %.2f  "
              "zona igual a la estricta %.3f"
              % (len(ref), np.median(err), (err < 1).mean(),
                 np.mean([res[c]["t_zona"] == res[c]["prod_zona"] for c in ref])))
    M = np.zeros((5, 5), int)
    for c in nuevos:
        M[ZONAS.index(y[c]), ZONAS.index(res[c]["t_zona"])] += 1
    print("   confusion (filas verdad, columnas tripode):")
    for i, z in enumerate(ZONAS):
        print("   %-18s %s" % (z, " ".join("%4d" % v for v in M[i])))
    s = [res[c]["s_fit"] for c in en if res[c].get("s_fit") is not None]
    if s:
        print("   coste del ajuste: mediana %.1f s/clip (p90 %.1f), n=%d"
              % (np.median(s), np.percentile(s, 90), len(s)))


if __name__ == "__main__":
    main()

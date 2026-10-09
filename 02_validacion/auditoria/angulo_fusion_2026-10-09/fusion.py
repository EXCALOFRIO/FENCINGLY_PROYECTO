"""Idea 1 de ANGULO_FUSION_2026-10-09: fusion de la x de la homografia y la del angulo en los clips
que ya contestan. Uso: python fusion.py AJUSTE | python fusion.py PRUEBA 'W(0.5),inf' """
import sys, csv, json, collections
import numpy as np
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\auditoria\rayas_id_2026-10-09")
import catalogo as CA, rutas as RU
import modelo as MO

ZONAS = CA.ZONAS


def universo(mitad):
    """Todos los clips con etiqueta y zona_ de la mitad: (clip, torneo, arma, etiqueta, zona_prod, via_luz)."""
    df = CA.manifiesto()
    df = df[df["strip_location"].isin(ZONAS)]
    out = []
    for _, m in df.iterrows():
        es_aj = m["torneo"] in MO.TORN_AJ
        if (mitad == "AJUSTE") != es_aj:
            continue
        p = RU.TRABAJO / ("zona_%s.csv" % m["clip_id"])
        if not p.is_file():
            continue
        r = next(csv.DictReader(open(p, encoding="utf-8")), None) or {}
        z = r.get("zona") if r.get("zona") in ZONAS else None
        out.append(dict(clip=m["clip_id"], torneo=m["torneo"], arma=m["weapon"], y=m["strip_location"],
                        zp=z, via=r.get("via_luz")))
    return out


def predicciones_angulo(rs):
    """x_a dejando fuera el propio clip, y e_T del torneo, para cada referencia con modelo."""
    por = collections.defaultdict(list)
    for r in rs:
        if r["motivo"] == "ok" and r["ang"] is not None and r["x_prod"] is not None and r["desp"] is not None:
            por[r["torneo"]].append(r)
    xa, eT = {}, {}
    for t, refs in por.items():
        if len(refs) < 9:     # >= 8 SIN contar el propio clip
            continue
        e = MO.loo(refs, True, True)
        et = float(np.nanmedian(np.abs(e)))
        for r, ei in zip(refs, e):
            if np.isfinite(ei):
                xa[r["clip"]] = r["x_prod"] + float(ei)
                eT[r["clip"]] = et
    return xa, eT


def candidatas():
    out = []
    for e in (1.0, 1.5, np.inf):
        for w in (0.0, 0.25, 0.5, 0.75):
            out.append(("W(%g)" % w, e))
        for d in (1.0, 2.0, 3.0):
            out.append(("D(%g)" % d, e))
    return out


def aplica(nombre, xh, xa):
    tipo, v = nombre[0], float(nombre[2:-1])
    if tipo == "W":
        return v * xh + (1 - v) * xa
    return xa if abs(xh - xa) > v else xh


def boot(d, n=2000, semilla=20261009, nivel=95.0):
    rng = np.random.default_rng(semilla)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    a = (100 - nivel) / 2
    return float(np.percentile(m, a)), float(np.percentile(m, 100 - a))


def boot_torneo(d, tor, n=2000, semilla=20261009, nivel=95.0):
    rng = np.random.default_rng(semilla)
    ts = np.unique(tor)
    idx = {t: np.nonzero(tor == t)[0] for t in ts}
    m = [d[np.concatenate([idx[t] for t in rng.choice(ts, len(ts))])].mean() for _ in range(n)]
    a = (100 - nivel) / 2
    return float(np.percentile(m, a)), float(np.percentile(m, 100 - a))


def evalua(U, rsd, xa, eT, nombre, e):
    ok_p, ok_n, arma, tor, via, cambia = [], [], [], [], [], 0
    disc = []
    for u in U:
        zp = u["zp"]
        zn = zp
        r = rsd.get(u["clip"])
        if zp is not None and r is not None and u["clip"] in xa and eT[u["clip"]] <= e:
            xh = r["x_prod"]
            x = aplica(nombre, xh, xa[u["clip"]])
            zn = ZONAS[MO.zona(x)]
            disc.append(abs(xh - xa[u["clip"]]))
            cambia += zn != zp
        ok_p.append(zp == u["y"]); ok_n.append(zn == u["y"])
        arma.append(u["arma"]); tor.append(u["torneo"]); via.append(u["via"])
    ok_p, ok_n = np.array(ok_p), np.array(ok_n)
    return dict(n=len(U), prod=ok_p.mean(), nuevo=ok_n.mean(), d=(ok_n.astype(float) - ok_p), cambia=cambia,
                arma=np.array(arma), tor=np.array(tor), via=np.array(via), disc=np.array(disc), ok_p=ok_p, ok_n=ok_n)


def main():
    mitad = sys.argv[1]
    rs = [r for r in MO.carga("rasgos.jsonl") if r["mitad"] == mitad]
    rsd = {r["clip"]: r for r in rs}
    xa, eT = predicciones_angulo(rs)
    U = universo(mitad)
    n_ok = sum(u["zp"] is not None for u in U)
    print("%s: clips con etiqueta y zona_ %d; contestan %d; con x del angulo (refs de torneo con >= 8 sin el clip) %d"
          % (mitad, len(U), n_ok, sum(1 for u in U if u["zp"] is not None and u["clip"] in xa)))
    cands = candidatas() if len(sys.argv) < 3 else [(sys.argv[2].split(",")[0], float(sys.argv[2].split(",")[1]))]
    nivel = 95.0 if mitad == "AJUSTE" else 97.5
    res = []
    for nombre, e in cands:
        R = evalua(U, rsd, xa, eT, nombre, e)
        lo, hi = boot(R["d"], nivel=nivel)
        res.append((nombre, e, R, lo, hi))
        print("  %-7s e<=%-4s  cambian %4d  produccion %.4f  nuevo %.4f  dif %+.4f  IC%.1f por clip [%+.4f, %+.4f]"
              % (nombre, e, R["cambia"], R["prod"], R["nuevo"], R["d"].mean(), nivel, lo, hi))
    if len(cands) > 1:
        best = max(res, key=lambda z: (z[2]["nuevo"], -z[2]["cambia"]))
        print("ELEGIDA en %s: %s e<=%s  dif %+.4f IC95 [%+.4f, %+.4f] -> %s" % (
            mitad, best[0], best[1], best[2]["d"].mean(), best[3], best[4],
            "va a PRUEBA" if best[3] > 0 else "NO va a PRUEBA (IC cruza cero)"))
        res = [best]
    nombre, e, R, lo, hi = res[0]
    lt, ht = boot_torneo(R["d"], R["tor"], nivel=nivel)
    print("detalle %s e<=%s: IC%.1f por torneo [%+.4f, %+.4f]; discrepancia |x_h-x_a| mediana %.2f m (n=%d)"
          % (nombre, e, nivel, lt, ht, float(np.median(R["disc"])) if len(R["disc"]) else float("nan"), len(R["disc"])))
    for a in sorted(set(R["arma"])):
        m = R["arma"] == a
        print("   arma %-6s n=%5d  produccion %.3f  nuevo %.3f  %+.3f" % (a, m.sum(), R["ok_p"][m].mean(), R["ok_n"][m].mean(), R["d"][m].mean()))
    for v in sorted(set(str(x) for x in R["via"])):
        m = np.array([str(x) == v for x in R["via"]])
        if m.sum() >= 20:
            print("   via  %-26s n=%5d  produccion %.3f  nuevo %.3f  %+.3f" % (v, m.sum(), R["ok_p"][m].mean(), R["ok_n"][m].mean(), R["d"][m].mean()))


if __name__ == "__main__":
    main()

"""La zona del tocado (etapa 11) contra la etiqueta humana `strip_location`, con el criterio de su adopcion.

POR QUE EXISTE (2026-10-09). La zona se adopto el 08/10 (0,476 contra `middle` 0,353, +0,123 IC95
[+0,090, +0,155], n=1.313) con un script de sesion que no quedo en el repo, y con las luces de
entonces. El 09/10 la etapa 9 cambio de detector (`LUCES_POR_FUENTE_2026-10-09.md`) y el cuadro del
tocado cambio en ~63 % de los clips, asi que la cifra hay que volver a medirla. El criterio es el de
la adopcion, sin tocar: la zona tiene que batir a contestar siempre `middle` con un IC95 que no
cruce cero. Se da el IC por clip (el de la adopcion) y, ademas, por torneo, que es mas exigente.

    python mide_zona.py
"""
from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "01_extraccion" / "codigo"))
import catalogo as CA  # noqa: E402
import rutas as RU  # noqa: E402


def boot(ok_a, ok_b, grupos=None, n=2000, semilla=20261009):
    rng = np.random.default_rng(semilla)
    d = ok_b.astype("float64") - ok_a.astype("float64")
    if grupos is None:
        m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    else:
        gs = np.unique(grupos)
        idx = {g: np.nonzero(grupos == g)[0] for g in gs}
        m = [d[np.concatenate([idx[g] for g in rng.choice(gs, len(gs))])].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    df = CA.manifiesto()
    df = df[df["strip_location"].isin(CA.ZONAS)]
    filas = []
    sin = {"sin zona_": 0, "zona vacia": 0}
    for _, m in df.iterrows():
        p = RU.TRABAJO / ("zona_%s.csv" % m["clip_id"])
        if not p.is_file():
            sin["sin zona_"] += 1
            continue
        with open(p, encoding="utf-8") as fh:
            r = next(csv.DictReader(fh), None)
        if not r or r.get("zona") not in CA.ZONAS:
            sin["zona vacia"] += 1
            filas.append((m["clip_id"], m["torneo"], m["weapon"], m["strip_location"], None,
                          (r or {}).get("motivo"), (r or {}).get("via_luz")))
            continue
        filas.append((m["clip_id"], m["torneo"], m["weapon"], m["strip_location"], r["zona"],
                      r.get("motivo"), r.get("via_luz")))
    print("clips con strip_location: %d; con zona_: %d; sin zona_: %d"
          % (len(df), len(filas), sin["sin zona_"]))
    con = [f for f in filas if f[4] is not None]
    print("contestan: %d de %d (%.3f)" % (len(con), len(filas), len(con) / max(len(filas), 1)))
    y = np.array([f[3] for f in con])
    z = np.array([f[4] for f in con])
    tor = np.array([f[1] for f in con])
    ok_z = z == y
    ok_m = y == "middle"
    lo, hi = boot(ok_m, ok_z)
    lo_t, hi_t = boot(ok_m, ok_z, tor)
    print()
    print("donde contesta (n=%d):  zona %.3f   middle %.3f   diferencia %+.3f" %
          (len(con), ok_z.mean(), ok_m.mean(), ok_z.mean() - ok_m.mean()))
    print("   IC95 por clip   [%+.3f, %+.3f]   (el de la adopcion del 08/10)" % (lo, hi))
    print("   IC95 por torneo [%+.3f, %+.3f]" % (lo_t, hi_t))
    oy = np.array([CA.ZONAS.index(v) for v in y])
    oz = np.array([CA.ZONAS.index(v) for v in z])
    print("   a una zona o menos %.3f" % (np.abs(oy - oz) <= 1).mean())
    print("   CRITERIO (batir a middle, IC95 por clip sin cruzar cero): %s"
          % ("PASA" if lo > 0 else "NO PASA"))
    print()
    print("por via de la luz y por arma (informativo):")
    for nombre, col in (("via_luz", 6), ("arma", 2)):
        vals = np.array([str(f[col]) for f in con])
        for v in sorted(set(vals)):
            m = vals == v
            if m.sum() < 20:
                continue
            print("   %-8s %-26s n=%5d  zona %.3f  middle %.3f  %+.3f"
                  % (nombre, v, m.sum(), ok_z[m].mean(), ok_m[m].mean(),
                     ok_z[m].mean() - ok_m[m].mean()))
    finales = []
    for f in con:
        p = RU.TRABAJO / ("zona_%s.csv" % f[0])
        with open(p, encoding="utf-8") as fh:
            r = next(csv.DictReader(fh), None) or {}
        finales.append((r.get("zona_final") or None, r.get("via_zona") or None))
    if any(z for z, _ in finales):
        """LA ZONA FINAL (`zona_fusion.py`, regla D(2) de `ANGULO_FUSION_2026-10-09.md`): la de la
        homografia desempatada con el angulo del borde lejano. Mismos clips que arriba."""
        zf = np.array([fin if fin else zh for (fin, _), zh in zip(finales, z)])
        ok_f = zf == y
        lo_f, hi_f = boot(ok_z, ok_f)
        print()
        print("ZONA FINAL (zona_fusion): %.3f contra la homografia %.3f: %+.3f IC95 por clip [%+.3f, %+.3f]"
              % (ok_f.mean(), ok_z.mean(), ok_f.mean() - ok_z.mean(), lo_f, hi_f))
        vias = np.array([str(v) for _, v in finales])
        for v in sorted(set(vias)):
            m = vias == v
            print("   via_zona %-24s n=%5d  final %.3f  homografia %.3f" % (v, m.sum(), ok_f[m].mean(), ok_z[m].mean()))
    mot = {}
    for f in filas:
        if f[4] is None:
            mot[f[5]] = mot.get(f[5], 0) + 1
    print()
    print("por que no contesta:", sorted(mot.items(), key=lambda t: -t[1])[:8])


if __name__ == "__main__":
    main()

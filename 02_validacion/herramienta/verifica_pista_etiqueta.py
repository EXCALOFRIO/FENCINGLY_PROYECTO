"""La posicion en la pista contra `strip_location`: el primer control EXTERNO que esto ha tenido.

## POR QUE ESTO ES DISTINTO DE TODO LO ANTERIOR

Todos los controles de la pista hasta ahora eran **internos**: la deriva del anclaje se mide contra
el propio anclaje, el recorrido lateral contra el ancho reglamentario, la estatura reproyectada
contra su propia mediana. Son buenos controles y ya dicen que algo va mal -- `ancla_fiable` pasa en
**56 de 367** clips y la deriva mediana es **2,72 m** en una pista de 14 -- pero ninguno dice
**donde** estaba el tirador de verdad.

`strip_location` si: es un humano que miro el clip y dijo en que quinto de la pista ocurrio la
accion. Hay **3.964 clips de florete** anotados. Eso permite, por primera vez:

  1. medir el error de la posicion contra algo que no sale de la propia homografia;
  2. separar **sesgo** de **ruido**, que es lo que decide si se puede corregir o no;
  3. y si hay sesgo, medir cuanto.

## LAS CINCO ZONAS Y SUS FRONTERAS, DEL REGLAMENTO

    fotl_warning_box   x < -5      los ultimos 2 m del lado del tirador de la izquierda (t.19)
    fotl_half          -5 <= x < 0
    middle             ...          ver abajo: "middle" NO es una quinta parte
    fotr_half          0 < x <= 5
    fotr_warning_box   x > 5

**`middle` no tiene frontera en el reglamento**, y eso hay que decidirlo antes de medir o el
resultado lo decide el convenio. Se prueban tres anchuras -- +-1 m, +-2 m (las lineas de puesta en
guardia, que es la unica frontera fisica que hay cerca del centro) y +-3 m -- y se da el resultado
con las tres. Si el veredicto cambia entre ellas, el veredicto es del convenio y no de la medida.

## QUE SE COMPARA, Y CONTRA QUE LINEA BASE

Se compara la zona que sale de NUESTRA x con la que dijo el humano. Y al lado, obligatorio:

    clase mayoritaria     contestar siempre `middle`, que es la respuesta mas comun
    acuerdo humano        0,619 -- dos personas que miran el mismo clip solo coinciden eso

El segundo es el techo. Si nuestra x acertara 0,62 estaria al nivel de un anotador humano; si
acierta 0,30 esta mal, y **cuanto** por debajo del 0,62 es la medida del fallo.

    python verifica_pista_etiqueta.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "01_extraccion" / "codigo"))
import catalogo as CA  # noqa: E402
import rutas as RU  # noqa: E402

ZONAS = CA.ZONAS
ANCHOS_MEDIO = (1.0, 2.0, 3.0)


def zona_de(x, medio):
    """QUE DEVUELVE: el indice de zona de una x en metros, con `middle` de +-`medio`."""
    if x < -5.0:
        return 0
    if x > 5.0:
        return 4
    if x < -medio:
        return 1
    if x > medio:
        return 3
    return 2


def x_del_clip(clip):
    """QUE DEVUELVE: (x mediana del punto medio de los dos tiradores, n eventos), o (None, 0).

    Se usa la MEDIANA de todos los eventos del clip y no un instante: la etiqueta humana es "donde
    ocurrio la accion", que es un juicio sobre el clip entero, y comparar un instante concreto con
    un juicio global mediria dos cosas distintas.
    """
    p = RU.PISTA / ("serie_eventos_%s.csv" % clip)
    if not p.is_file():
        return None, 0
    xs = []
    with open(p, encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            try:
                v = float(f["x_inicio_m"])
            except (TypeError, ValueError, KeyError):
                continue
            if np.isfinite(v):
                xs.append(v)
    if not xs:
        return None, 0
    return float(np.median(xs)), len(xs)


def recoge():
    df = CA.manifiesto().set_index("clip_id")
    out = []
    for p in sorted(glob.glob(str(RU.PISTA / "serie_eventos_*.csv"))):
        clip = os.path.basename(p)[14:-4]
        if clip not in df.index:
            continue
        et = df.loc[clip, "strip_location"]
        if et not in ZONAS:
            continue
        x, n = x_del_clip(clip)
        if x is None:
            continue
        out.append((clip, ZONAS.index(et), x, n, df.loc[clip, "weapon"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    print("LA POSICION EN LA PISTA CONTRA LA ETIQUETA HUMANA")
    print("=" * 84)
    filas = recoge()
    print("clips con NUESTRA x medida Y etiqueta humana de zona: %d" % len(filas))
    if len(filas) < 30:
        print("menos de 30: no se da ninguna cifra. Falta extraccion.")
        return 0
    y = np.array([f[1] for f in filas])
    x = np.array([f[2] for f in filas])
    print("nuestra x (m): p10 %.2f  p50 %.2f  p90 %.2f   |x|>7 en %d clips"
          % (np.percentile(x, 10), np.median(x), np.percentile(x, 90),
             int((np.abs(x) > 7).sum())))
    print("la humana dice: %s"
          % dict(collections.Counter(ZONAS[i] for i in y).most_common()))

    may = collections.Counter(y).most_common(1)[0]
    print()
    print("%-26s %9s %9s %11s" % ("convenio de `middle`", "exacto", "<=1 zona", "err ordinal"))
    print("-" * 60)
    mejor = None
    for m in ANCHOS_MEDIO:
        p = np.array([zona_de(v, m) for v in x])
        ex = float((p == y).mean())
        un = float((np.abs(p - y) <= 1).mean())
        eo = float(np.abs(p - y).mean())
        print("%-26s %9.3f %9.3f %11.3f" % ("+-%.0f m" % m, ex, un, eo))
        if mejor is None or ex > mejor[1]:
            mejor = (m, ex, p)
    print("-" * 60)
    print("%-26s %9.3f %9s %11s"
          % ("contestar siempre `%s`" % ZONAS[may[0]], may[1] / len(y), "-", "-"))
    print("%-26s %9.3f %9s %11s" % ("techo: acuerdo humano", 0.619, "-", "-"))

    m, ex, p = mejor
    print()
    print("con el mejor convenio (+-%.0f m), matriz (filas = humano, columnas = nuestra x):" % m)
    M = np.zeros((5, 5), int)
    for a, b in zip(y, p):
        M[a, b] += 1
    print("%-20s %s" % ("", " ".join("%6s" % z[:6] for z in ZONAS)))
    for i, z in enumerate(ZONAS):
        print("%-20s %s   n=%d" % (z, " ".join("%6d" % v for v in M[i]), M[i].sum()))

    print()
    print("SESGO O RUIDO, que es lo que decide si se puede corregir")
    d = p.astype(float) - y
    print("   desplazamiento medio en zonas: %+.3f  (0 = sin sesgo)" % d.mean())
    print("   desviacion tipica            :  %.3f" % d.std())
    cen = {}
    for i in range(5):
        v = x[y == i]
        if len(v) >= 10:
            cen[ZONAS[i]] = (float(np.median(v)), len(v))
    print()
    print("   NUESTRA x mediana dentro de cada zona que dijo el humano:")
    esper = {"fotl_warning_box": -6.0, "fotl_half": -3.5, "middle": 0.0,
             "fotr_half": 3.5, "fotr_warning_box": 6.0}
    for z, (v, n) in cen.items():
        print("      %-20s %+7.2f m   (esperado ~%+.1f)   n=%d" % (z, v, esper[z], n))
    if len(cen) >= 3:
        zz = [z for z in ZONAS if z in cen]
        a = np.array([esper[z] for z in zz])
        b = np.array([cen[z][0] for z in zz])
        A = np.vstack([a, np.ones(len(a))]).T
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        print()
        print("   ajuste  nuestra_x = %.3f * x_esperada %+.3f" % (sol[0], sol[1]))
        print("   -> pendiente %.3f: %s" % (sol[0],
              "la ESCALA esta bien" if 0.85 <= sol[0] <= 1.15 else
              "la ESCALA esta mal (deberia ser 1,0)"))
        print("   -> corte %+.3f m: %s" % (sol[1],
              "el ORIGEN esta bien" if abs(sol[1]) < 0.5 else
              "el ORIGEN esta desplazado"))
        print()
        print("   Si la pendiente se va de 1 el fallo es de ESCALA longitudinal, que es justo lo")
        print("   que el proyecto tiene medido como no resuelto: dos rectas paralelas de")
        print("   separacion conocida fijan la escala TRANSVERSAL y no dicen nada de la LARGA.")
        print("   Si lo que se va es el corte, el fallo es del ANCLAJE y es mas facil de atacar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

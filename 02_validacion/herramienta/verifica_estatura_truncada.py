"""EL CONTROL DE ESTATURA SELECCIONA SOBRE EL RESULTADO, y por eso aprueba clips rotos.

## EL DEFECTO, en una linea de codigo

`pista.control_estatura` (linea 1649) hace:

    h = altura_m(hg, pie, cab)
    if h is not None and 0.3 < h < 5.0:      <-- AQUI
        alturas[lado].append(h)

o sea **tira las alturas que no le gustan y despues publica la MEDIANA de las que quedan** como si
fuera una medida del clip. Si una homografia reproyecta al tirador a 7,45 m, esa observacion no
entra; entran solo las pocas que por ruido cayeron dentro, y de ahi sale un `h_p50 = 1,76` que pasa
la puerta `1,60 <= h <= 2,00` con matricula.

Es exactamente la disciplina que el proyecto ya tiene escrita y que aqui no se aplico: **no se
evalua en un subgrupo definido por el resultado**. El filtro define el subgrupo con la propia
magnitud que se va a medir.

## COMO SE DESCUBRIO, y merece quedar escrito

Mirando los tableros de tres clips me parecio que la tira rectificada no era la pista. El control de
estatura decia que si lo era en dos de los tres, asi que di por buena la cifra y me corregi. Al
reestimar la escala con TODAS las alturas salio 7,45 m donde el control decia 1,76. La cifra era la
que mentia: el filtro habia descartado el 80 % de las observaciones.

## QUE MIDE ESTE FICHERO

Para cada clip, las dos cosas al lado:

    h_p50 FILTRADA      lo que publica hoy `control_estatura`, con el corte 0,3-5,0
    h_p50 COMPLETA      la mediana de TODAS las alturas reproyectadas, sin tirar ninguna
    supervivencia       que fraccion de cajas pasa el filtro

Si la supervivencia es alta, el filtro solo quitaba ruido y la cifra valia. Si es baja, la cifra era
del filtro y no del clip, y **los 10 de 78 clips "metricamente sanos" hay que volver a contarlos**.

    python verifica_estatura_truncada.py --n 25
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "01_extraccion" / "codigo"))
import focal_fisica as FF  # noqa: E402
import pista as PI  # noqa: E402
import rutas as RU  # noqa: E402

LO, HI = 0.3, 5.0


def una(clip):
    r = PI.analiza(clip)
    if not r or not r.get("ok"):
        return {"clip": clip, "ok": False, "motivo": (r or {}).get("motivo", "sin homografia")}
    hg = r["_hg"]
    pares = FF._pares(r, nmax=200)
    v = []
    for pie, cab in pares:
        try:
            h = PI.altura_m(hg, pie, cab)
        except Exception:
            h = None
        if h is not None and np.isfinite(h):
            v.append(float(h))
    if len(v) < 8:
        return {"clip": clip, "ok": False, "motivo": "menos de 8 alturas"}
    v = np.asarray(v)
    dentro = v[(v > LO) & (v < HI)]
    return {"clip": clip, "ok": True, "n": len(v), "n_dentro": len(dentro),
            "superv": float(len(dentro) / len(v)),
            "h_filtrada": float(np.median(dentro)) if len(dentro) else float("nan"),
            "h_completa": float(np.median(v)),
            "h_p10": float(np.percentile(v, 10)), "h_p90": float(np.percentile(v, 90))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--n", type=int, default=25)
    a = ap.parse_args()
    clips = a.clips or [os.path.basename(p)[6:-4] for p in
                        sorted(glob.glob(str(RU.TRABAJO / "poses_*.npz")))][:a.n]
    print("LA ESTATURA REPROYECTADA, CON Y SIN EL FILTRO 0,3-5,0 m")
    print("=" * 94)
    print("%-32s %6s %7s %10s %10s %8s %8s"
          % ("clip", "n", "superv", "FILTRADA", "COMPLETA", "p10", "p90"))
    print("-" * 94)
    fil = []
    for c in clips:
        try:
            v = una(c)
        except Exception as ex:
            print("%-32s EXCEPCION %s" % (c[:32], type(ex).__name__))
            continue
        if not v.get("ok"):
            continue
        fil.append(v)
        print("%-32s %6d %7.2f %10.2f %10.2f %8.2f %8.2f"
              % (c[:32], v["n"], v["superv"], v["h_filtrada"], v["h_completa"],
                 v["h_p10"], v["h_p90"]))
    print()
    if not fil:
        print("ningun clip con homografia. Sin cifras.")
        return 0
    sup = np.array([v["superv"] for v in fil])
    hf = np.array([v["h_filtrada"] for v in fil])
    hc = np.array([v["h_completa"] for v in fil])
    pf = int(((hf >= 1.60) & (hf <= 2.00)).sum())
    pc = int(((hc >= 1.60) & (hc <= 2.00)).sum())
    print("clips medidos: %d" % len(fil))
    print("supervivencia al filtro:  p10 %.2f  p50 %.2f  p90 %.2f"
          % (np.percentile(sup, 10), np.median(sup), np.percentile(sup, 90)))
    print("clips con supervivencia < 0,50: %d de %d" % (int((sup < 0.50).sum()), len(fil)))
    print()
    print("PASAN la puerta 1,60-2,00 m:")
    print("   con el filtro de hoy    %d de %d   = %.0f %%" % (pf, len(fil), 100.0 * pf / len(fil)))
    print("   sin filtrar nada        %d de %d   = %.0f %%" % (pc, len(fil), 100.0 * pc / len(fil)))
    print()
    print("La diferencia entre esas dos cifras es lo que el filtro estaba regalando. Si es grande,")
    print("`control_estatura` no es un control: es un recorte que se aprueba a si mismo, y las")
    print("homografias que aprobaba hay que volver a contarlas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

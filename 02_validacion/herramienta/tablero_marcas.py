"""El tablero con el que se ANOTA A MANO que raya de la pista es cada marca detectada.

## POR QUE HACE FALTA ESTA ANOTACION Y NO VALE `strip_location`

`strip_location` es la unica verdad externa que la pista ha tenido, y tiene dos limites que no se
arreglan con mas clips:

  1. **Es por clip, no por marca.** Dice "la accion ocurrio en el tercio izquierdo". De ahi no se
     puede saber si el emparejamiento confundio la raya de -2 con la de +2, que es el fallo que
     estamos persiguiendo.
  2. **Su techo es 0,619**, que es lo que coinciden dos personas mirando el mismo clip. Contra una
     verdad que solo acierta 0,62 consigo misma no se puede afinar nada por debajo de ese ruido.

Lo que falla esta un paso antes: `identifica_marcas` recibe N rayas detectadas y tiene que decir
cual es la central, cual la de puesta en guardia, cual la frontera de color. **De esa
correspondencia no existe ni una etiqueta en el proyecto.** Sin ella, "he mejorado el
emparejamiento" no es contrastable -- que es exactamente donde se quedo `ancla_v2`.

Esta anotacion es por MARCA, es determinista (una raya pintada esta o no esta) y no tiene el ruido
de juicio que tiene "donde ocurrio la accion". Es la verdad que faltaba.

## QUE DIBUJA, Y POR QUE CADA PANEL

    1. LA TIRA RECTIFICADA, grande        la pista vista desde arriba, sin personas. Aqui las
                                          rayas pintadas son verticales y las separaciones ya son
                                          comparables con las del reglamento sin corregir nada.
    2. LA MISMA TIRA con las candidatas   cada raya detectada, numerada. Es lo que se anota.
    3. EL PERFIL DE COLOR                 la zona de aviso de t.19 -- los ultimos 2 m de cada lado
                                          -- es de otro color. Un escalon aqui es un ancla
                                          ABSOLUTA (+-5 m), no relativa, y es la senal mas robusta
                                          que hay en el cuadro.
    4. LA PLACA con bordes y marcas       para distinguir un fallo de rectificacion de uno de
                                          emparejamiento. Si la tira no esta centrada en la pista,
                                          el problema es de los bordes y no de las marcas.

## COMO SE ANOTA

Por cada numero, una de estas: `-7 -5 -2 0 +2 +5 +7` o `x` si no es una marca del reglamento
(costura del suelo, cable, sombra, borde de moqueta). Se permite `?` y **hay que usarlo**: una
etiqueta dudosa metida como cierta contamina la verdad, que es el unico activo de todo esto.

    python tablero_marcas.py --n 12
    python tablero_marcas.py CLIP [CLIP ...]
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "01_extraccion" / "codigo"))
import pista as PI  # noqa: E402
import rutas as RU  # noqa: E402

SALIDA = RU.VAL / "pista" / "tablero"
ANOTACION = pathlib.Path(__file__).resolve().parents[2] / "02_validacion" / "pista" / \
    "anotar" / "marcas_anotadas.csv"

ESC_V = 3
"""Cuanto se estira la tira en vertical. La tira nativa es 960x160 y una raya pintada ocupa 2-3
columnas: estirando en vertical se ve el trazo entero sin tocar la coordenada que se anota."""


def _regla(ncol, alto=26, paso=96):
    """QUE DIBUJA: la regla de columnas. Se anota por NUMERO de marca, pero la columna hace falta
    para poder repescar a mano una raya que el detector no encontro -- que es la mitad del problema
    y no se veria si solo se dibujaran las candidatas."""
    r = np.zeros((alto, ncol, 3), "uint8")
    for c in range(0, ncol, paso):
        cv2.line(r, (c, alto - 8), (c, alto), (150, 150, 150), 1)
        cv2.putText(r, str(c), (min(c + 2, ncol - 28), alto - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 170, 170), 1, cv2.LINE_AA)
    return r


def _perfil(t, ncol, alto=110):
    """QUE DIBUJA: el color mediano de la franja central de la pista a lo largo de la tira, en los
    tres canales. La frontera de t.19 sale como un ESCALON simultaneo en los tres."""
    im = np.zeros((alto, ncol, 3), "uint8")
    try:
        pf = PI.perfil_color(t)
        med = np.asarray(pf["med"], "float64")
    except Exception:
        med = None
    if med is None or med.ndim != 2 or med.shape[0] != 3 or med.shape[1] < 8:
        cv2.putText(im, "sin perfil de color", (8, alto // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
        return im
    n = med.shape[1]
    """`perfil_color` devuelve (3, ncol) en Lab, NO (ncol, 3) ni BGR. El propio modulo documenta que
    confundir ese orden ya dio CERO fronteras en los 20 clips una vez; aqui se indexa por fila."""
    for canal, col in enumerate(((235, 235, 235), (90, 160, 255), (255, 200, 90))):
        v = med[canal]
        if not np.isfinite(v).any():
            continue
        lo, hi = np.nanpercentile(v, 2), np.nanpercentile(v, 98)
        rng = max(hi - lo, 1e-6)
        pts = []
        for k in range(n):
            if not np.isfinite(v[k]):
                continue
            x = int(k * (ncol - 1) / max(n - 1, 1))
            y = int(alto - 6 - (v[k] - lo) / rng * (alto - 14))
            pts.append((x, max(2, min(alto - 2, y))))
        if len(pts) > 1:
            cv2.polylines(im, [np.array(pts, "int32")], False, col, 1, cv2.LINE_AA)
    cv2.putText(im, "perfil Lab por columna: L blanco, a naranja, b azul "
                    "(un escalon en a o b = frontera de color t.19, o sea +-5 m)",
                (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (200, 200, 200), 1, cv2.LINE_AA)
    return im


def _franja(txt, ncol, alto=22, col=(235, 235, 235), esc=0.42):
    im = np.zeros((alto, ncol, 3), "uint8")
    cv2.putText(im, txt[:int(ncol / (esc * 11.5))], (6, alto - 7),
                cv2.FONT_HERSHEY_SIMPLEX, esc, col, 1, cv2.LINE_AA)
    return im


def una(clip):
    """QUE DEVUELVE: (ruta del png, dict con las columnas de cada candidata), o (None, motivo)."""
    r = PI.analiza(clip)
    if not r or not r.get("ok"):
        return None, {"motivo": (r or {}).get("motivo", "sin homografia")}
    t = PI.tira_rectificada(r)
    if t is None or "tira" not in t:
        return None, {"motivo": "sin tira rectificada"}
    base = t["tira"]
    nfil, ncol = base.shape[:2]
    lin = list(((r.get("_mar") or {}).get("lineas") or []))
    cols = []
    for k, li in enumerate(lin):
        try:
            c = PI.a_tira(t, np.asarray(li["centro"], "float64")[None, :])[0]
        except Exception:
            continue
        if np.isfinite(c[0]) and -20 <= c[0] <= ncol + 20:
            cols.append((k, float(c[0]), bool(li.get("afinada"))))
    cols.sort(key=lambda v: v[1])

    limpia = cv2.resize(base, (ncol, nfil * ESC_V), interpolation=cv2.INTER_CUBIC)
    marcada = limpia.copy()
    for n, (_k, c, af) in enumerate(cols):
        ci = int(round(c))
        if not (0 <= ci < ncol):
            continue
        col = (60, 255, 255) if af else (80, 200, 80)
        cv2.line(marcada, (ci, 24), (ci, nfil * ESC_V), col, 1)
        cv2.putText(marcada, str(n), (max(2, ci - 6), 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)
    """LAS AFINADAS EN AMARILLO Y LAS CRUDAS EN VERDE: `identifica_marcas` solo usa las afinadas,
    asi que anotando las dos se mide tambien cuanto se pierde en ese filtro -- que es una decision
    que hoy nadie ha comprobado."""

    placa = PI.panel_bordes_marcas(r)
    ph, pw = placa.shape[:2]
    placa = cv2.resize(placa, (ncol, max(1, int(ph * ncol / pw))),
                       interpolation=cv2.INTER_AREA)

    piezas = [
        _franja("%s   |   %d candidatas (%d afinadas)   |   tira %d col"
                % (clip, len(cols), sum(1 for c in cols if c[2]), ncol), ncol, 24, esc=0.46),
        _regla(ncol),
        _franja("1. TIRA LIMPIA -- buscar aqui las rayas pintadas, sin sugerencias", ncol),
        limpia,
        _franja("2. LAS CANDIDATAS numeradas (amarillo afinada, verde cruda)", ncol),
        marcada,
        _perfil(t, ncol),
        _franja("4. LA PLACA con bordes (rojo A / azul B) y marcas", ncol),
        placa,
    ]
    img = np.vstack([p if p.ndim == 3 else cv2.cvtColor(p, cv2.COLOR_GRAY2BGR) for p in piezas])
    SALIDA.mkdir(parents=True, exist_ok=True)
    p = SALIDA / ("tablero_%s.png" % clip)
    cv2.imwrite(str(p), img)
    return p, {"cols": cols, "ncol": ncol,
               "patron_algoritmo": r.get("ident_patron"), "n_ident": r.get("n_ident")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    clips = a.clips or [os.path.basename(p)[6:-4] for p in
                        sorted(glob.glob(str(RU.TRABAJO / "poses_*.npz")))][:a.n]
    print("TABLERO DE ANOTACION DE MARCAS")
    print("=" * 92)
    for c in clips:
        try:
            p, inf = una(c)
        except Exception as ex:
            print("%-34s EXCEPCION %s: %s" % (c[:34], type(ex).__name__, ex))
            continue
        if p is None:
            print("%-34s %s" % (c[:34], inf.get("motivo")))
            continue
        print("%-34s %2d candidatas  cols %s" % (
            c[:34], len(inf["cols"]),
            " ".join("%d:%.0f%s" % (n, v, "*" if af else "")
                     for n, (_k, v, af) in enumerate(inf["cols"]))[:44]))
        print("%-34s   algoritmo dice: %s" % ("", inf.get("patron_algoritmo")))
    print()
    print("Los tableros estan en %s" % SALIDA)
    print("Se anotan a %s" % ANOTACION)
    return 0


if __name__ == "__main__":
    sys.exit(main())

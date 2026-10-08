"""Mira las marcas de la pista: dibuja donde el anclaje CREE que estan las lineas del reglamento.

## POR QUE HACIA FALTA MIRAR, Y NO BASTABA CON LAS CIFRAS

El anclaje publica `ident_err_p50_m = 0,000 m` y eso parece una identificacion perfecta. **No lo
es.** Con `n_ident = 2` marcas y un modelo de dos parametros -- escala y origen -- el residuo es
**cero por aritmetica**, se acierte o se falle de marca. Es el mismo error que este proyecto ya
tiene documentado con el cono: *"la hoja esta a 0 grados de su propio centro"* es cierto por
construccion y no comprueba nada.

El control que si vale es el de **dejar una fuera**: ajustar con dos marcas, predecir una TERCERA
que no vio y medir el fallo. Eso es `loo_p50_m`, y exige **tres** marcas identificadas. Hoy vale
`None` en la mayoria de los clips, o sea que **el anclaje se usa sin control**.

Y lo que ninguna cifra ensena: si las lineas caen donde se ven las marcas de verdad. Para eso hay
que dibujarlas encima y mirar.

## QUE DIBUJA

La tira rectificada -- la pista vista desde arriba, sin personas -- con una linea vertical en la
columna donde el anclaje situa cada marca del reglamento:

    -7 y +7   rojo      los limites posteriores
    -5 y +5   naranja   las fronteras de color de t.19
    -2 y +2   amarillo  las lineas de puesta en guardia
     0        verde     la linea central

**Como se lee.** Si el anclaje es correcto, cada linea cae sobre una marca visible en la pista:
las de puesta en guardia sobre rayas pintadas, las de t.19 sobre el cambio de color, los limites
sobre el final de la pista. Si una linea cae en medio de la nada, o si hay una marca visible sin
linea encima, el emparejamiento esta mal -- y el residuo seguira diciendo 0,000.

    python verifica_marcas_pista.py CLIP [CLIP ...]
    python verifica_marcas_pista.py --n 6        los primeros 6 con tira
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

SALIDA = RU.VAL / "pista" / "marcas"

MARCAS = [(-7.0, (40, 40, 220), "limite"), (-5.0, (40, 140, 240), "t.19"),
          (-2.0, (40, 220, 240), "guardia"), (0.0, (60, 220, 60), "centro"),
          (2.0, (40, 220, 240), "guardia"), (5.0, (40, 140, 240), "t.19"),
          (7.0, (40, 40, 220), "limite")]


def una(clip, escala=3):
    """QUE DEVUELVE: (ruta del png, dict con lo que el anclaje afirma), o (None, motivo)."""
    r = PI.analiza(clip)
    if not r or not r.get("ok"):
        return None, {"motivo": (r or {}).get("motivo", "sin homografia")}
    t = PI.tira_rectificada(r)
    if t is None or "tira" not in t:
        return None, {"motivo": "sin tira rectificada"}
    an = PI.ancla_zonas(t)
    im = t["tira"].copy()
    nfil, ncol = im.shape[:2]
    inf = {"n_ident": r.get("n_ident"), "ident_patron": r.get("ident_patron"),
           "ident_err_p50_m": r.get("ident_err_p50_m"),
           "via_ancla": an.get("via_ancla"), "controlada": an.get("controlada"),
           "ambiguo": an.get("ambiguo"), "loo_p50_m": an.get("loo_p50_m"),
           "m_por_col": an.get("m_por_col"), "q_m": an.get("q_m"),
           "x_ini_m": an.get("x_ini_m"), "x_fin_m": an.get("x_fin_m")}
    mpc, q = an.get("m_por_col"), an.get("q_m")
    dentro = 0
    if an.get("ok") and mpc:
        for m, col, nom in MARCAS:
            c = int(round((m - q) / mpc))
            if 0 <= c < ncol:
                dentro += 1
                cv2.line(im, (c, 0), (c, nfil), col, 1)
                cv2.putText(im, "%+g" % m, (max(2, c - 10), 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
        cv2.line(im, (0, int(t["fil0"])), (ncol, int(t["fil0"])), (255, 255, 255), 1)
        cv2.line(im, (0, int(t["fil1"])), (ncol, int(t["fil1"])), (255, 255, 255), 1)
        """LOS DOS BORDES DETECTADOS van en blanco: si la tira no esta centrada en la pista, el
        fallo es de la rectificacion y no del emparejamiento, y hay que poder distinguirlo."""
    inf["marcas_en_cuadro"] = dentro
    big = cv2.resize(im, (ncol, nfil * escala), interpolation=cv2.INTER_NEAREST)
    pie = np.zeros((46, ncol, 3), "uint8")
    txt = ("%s  |  %s  |  %d marcas ident, %d visibles  |  abarca %.2f m (%.2f a %.2f)"
           % (clip[:28], an.get("via_ancla") or "sin ancla", r.get("n_ident") or 0, dentro,
              abs(mpc or 0) * ncol, an.get("x_ini_m") or float("nan"),
              an.get("x_fin_m") or float("nan")))
    cv2.putText(pie, txt, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1,
                cv2.LINE_AA)
    ctrl = ("CONTROLADA (deja-una-fuera %.3f m)" % an["loo_p50_m"]) if an.get("loo_p50_m") \
        else "SIN CONTROL: con 2 marcas y 2 parametros el residuo es 0 por aritmetica"
    cv2.putText(pie, ctrl, (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (120, 255, 120) if an.get("loo_p50_m") else (120, 160, 255), 1, cv2.LINE_AA)
    SALIDA.mkdir(parents=True, exist_ok=True)
    p = SALIDA / ("marcas_%s.png" % clip)
    cv2.imwrite(str(p), np.vstack([big, pie]))
    return p, inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    clips = a.clips or [os.path.basename(p)[6:-4] for p in
                        sorted(glob.glob(str(RU.TRABAJO / "poses_*.npz")))][:a.n]
    print("LAS MARCAS DE LA PISTA, DIBUJADAS DONDE EL ANCLAJE DICE QUE ESTAN")
    print("=" * 96)
    n_ctrl = 0
    for c in clips:
        try:
            p, inf = una(c)
        except Exception as ex:
            print("%-34s EXCEPCION %s" % (c[:34], type(ex).__name__))
            continue
        if p is None:
            print("%-34s %s" % (c[:34], inf.get("motivo")))
            continue
        ctrl = inf.get("loo_p50_m")
        n_ctrl += 1 if ctrl else 0
        print("%-34s ident %-2s  err %-6s  loo %-8s %s"
              % (c[:34], inf.get("n_ident"),
                 ("%.3f" % inf["ident_err_p50_m"]) if inf.get("ident_err_p50_m") is not None
                 else "-",
                 ("%.3f m" % ctrl) if ctrl else "SIN CONTROL",
                 inf.get("ident_patron") or ""))
    print()
    print("clips con el control de deja-una-fuera: %d de %d" % (n_ctrl, len(clips)))
    print()
    print("UN `err` DE 0,000 CON 2 MARCAS NO ES PRECISION, ES ARITMETICA: dos puntos y dos")
    print("parametros (escala y origen) ajustan exacto aunque las marcas esten mal asignadas.")
    print("El unico numero que vale es `loo`, y exige TRES marcas identificadas.")
    print()
    print("Las imagenes estan en %s. Hay que MIRARLAS: si una linea cae en medio de la nada," % SALIDA)
    print("o hay una marca visible sin linea encima, el emparejamiento esta mal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

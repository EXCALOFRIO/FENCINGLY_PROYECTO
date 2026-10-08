"""Figuras del paso 5, y a la vez la comprobacion A OJO del CUADRO DE ENCENDIDO.

Para cada sede saca una fila de cuatro recortes de la zona de la luz detectada, tomados en

    cuadro - 10,  cuadro - 4,  cuadro + 2,  cuadro + 8

alrededor del encendido que dice el detector, con la caja de la luz dibujada en los dos ultimos.
Si en los dos primeros la luz esta APAGADA y en los dos ultimos ENCENDIDA, el cuadro de encendido
queda confirmado a ojo con una tolerancia de 4 cuadros, sin tener que mirar el clip entero. Si no,
se ve inmediatamente el fallo, que es para lo que sirve una figura.

POR QUE ESTA TOLERANCIA Y NO OTRA. 4 cuadros a 30 fps son 133 ms, y el bloqueo del aparato de
florete es de 300 ms: o sea que esta tolerancia es MENOR que el intervalo dentro del cual el propio
aparato considera simultaneos dos tocados. Para lo que el usuario quiere -- separar los clips en que
el orden de luces engana -- basta con resolver el orden dentro de 300 ms, y 133 ms lo resuelve. Lo
que NO resuelve es un dt de 40-80 ms, y eso se dice al lado de cada cifra de `dt`.

Sale un PNG por grupo de 7 sedes en `02_validacion/muestras/luces/figuras/`.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

import luces as LU
import f4_censo_visual as F4

SALIDA = os.path.join(F4.PROY, r"02_validacion\muestras\luces\figuras")
DESPLAZAMIENTOS = (-10, -4, 2, 8)
ANCHO_CELDA = 240
POR_FIGURA = 7


def recorta(img, luz, alto_rel=0.12, ancho_rel=0.40):
    """Recorte centrado en la luz, con la caja dibujada, escalado a ANCHO_CELDA."""
    H, W = img.shape[:2]
    cx, cy = luz["cx"], luz["cy"]
    x0 = int(max(0, (cx - ancho_rel / 2) * W))
    x1 = int(min(W, (cx + ancho_rel / 2) * W))
    y0 = int(max(0, (cy - alto_rel / 2) * H))
    y1 = int(min(H, (cy + alto_rel / 2) * H))
    rec = img[y0:y1, x0:x1].copy()
    bx0 = int(luz["fx"] * W) - x0
    bx1 = int((luz["fx"] + luz["fw"]) * W) - x0
    by0 = int(luz["fy"] * H) - y0
    by1 = int((luz["fy"] + luz["fh"]) * H) - y0
    cv2.rectangle(rec, (bx0 - 2, by0 - 2), (bx1 + 2, by1 + 2), (255, 255, 0), 2)
    esc = ANCHO_CELDA / max(1, rec.shape[1])
    return cv2.resize(rec, (ANCHO_CELDA, max(10, int(round(rec.shape[0] * esc)))))


def main() -> int:
    os.makedirs(SALIDA, exist_ok=True)
    refs = F4.carga_referencias()
    filas, grupo = [], 0
    for i, r in enumerate(refs):
        if not r["ruta"]:
            continue
        res = LU.detecta(r["ruta"])
        luces = LU.resumen_clip(LU.grupo_del_tocado(LU.agrupa(res["luces"], res["fps"]), res["fps"]))
        if not luces:
            print(f"{r['sede']}: sin luces, no hay figura")
            continue
        luz = max(luces, key=lambda d: d["area"])
        cap = cv2.VideoCapture(r["ruta"])
        celdas = []
        for dk in DESPLAZAMIENTOS:
            k = max(0, min(res["n"] - 1, luz["cuadro_encendido"] + dk))
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, img = cap.read()
            if not ok:
                continue
            cel = recorta(img, luz)
            cv2.putText(cel, f"{dk:+d} c{k}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1)
            celdas.append(cel)
        cap.release()
        if not celdas:
            continue
        alto = max(c.shape[0] for c in celdas)
        celdas = [np.vstack([c, np.zeros((alto - c.shape[0], c.shape[1], 3), np.uint8)])
                  for c in celdas]
        etiqueta = np.zeros((alto, 300, 3), np.uint8)
        lados = {"L": "left", "R": "right"}
        cv2.putText(etiqueta, r["sede"][:26], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        cv2.putText(etiqueta,
                    f"{F4.EN_CANAL[luz['canal']]} {lados[luz['lado']]}, "
                    f"frame {luz['cuadro_encendido']}",
                    (4, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(etiqueta, f"lights={len(luces)}  conf={luz['confianza']:.2f}",
                    (4, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        filas.append(np.hstack([etiqueta] + celdas))
        filas.append(np.full((3, filas[-1].shape[1], 3), 80, np.uint8))
        if len(filas) >= 2 * POR_FIGURA or i == len(refs) - 1:
            ancho = max(f.shape[1] for f in filas)
            filas = [np.hstack([f, np.zeros((f.shape[0], ancho - f.shape[1], 3), np.uint8)])
                     if f.shape[1] < ancho else f for f in filas]
            destino = os.path.join(SALIDA, f"encendido_{grupo}.png")
            cv2.imwrite(destino, np.vstack(filas))
            print(f"escrito {destino}")
            filas, grupo = [], grupo + 1
    if filas:
        ancho = max(f.shape[1] for f in filas)
        filas = [np.hstack([f, np.zeros((f.shape[0], ancho - f.shape[1], 3), np.uint8)])
                 if f.shape[1] < ancho else f for f in filas]
        destino = os.path.join(SALIDA, f"encendido_{grupo}.png")
        cv2.imwrite(destino, np.vstack(filas))
        print(f"escrito {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

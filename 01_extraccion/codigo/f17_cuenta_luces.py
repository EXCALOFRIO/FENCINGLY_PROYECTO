"""Mosaico de CONTEO: la banda del marcador entera, 12 cuadros despues del encendido, una fila por sede.

Para que sirve. Es la figura con la que se anoto a ojo **cuantas luces** tiene cada clip, que es la
cifra que decide si un clip es de una sola luz -- y entonces el tocado es trivial, 96,19 % acertando
"gana el que enciende" -- o de dos, que es donde hay que aplicar la convencion. Se ve la banda
COMPLETA y por tanto las DOS placas a la vez, que es lo que las figuras de `f8` no permiten porque
recortan alrededor de una sola luz.

Existia como script suelto y se convierte en fichero para que sea reproducible y para que su texto
este en ingles, que es lo que el usuario pidio para todo lo que se publica como imagen.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

import luces as LU
import f4_censo_visual as F4

SALIDA = os.path.join(F4.PROY, r"02_validacion\muestras\luces\figuras")
ANCHO = 1000
Y0 = 0.845
POR_FIGURA = 11
LADOS = {"L": "left", "R": "right"}


def main() -> int:
    os.makedirs(SALIDA, exist_ok=True)
    refs = F4.carga_referencias()
    filas, grupo = [], 0
    for i, r in enumerate(refs):
        if not r["ruta"]:
            continue
        res = LU.detecta(r["ruta"])
        lz = LU.resumen_clip(LU.grupo_del_tocado(LU.agrupa(res["luces"], res["fps"]), res["fps"]))
        if not lz:
            print(f"{r['sede']}: sin luces")
            continue
        k = min(res["n"] - 1, max(d["cuadro_encendido"] for d in lz) + 12)
        cap = cv2.VideoCapture(r["ruta"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue
        H, W = img.shape[:2]
        banda = img[int(Y0 * H):H]
        esc = ANCHO / W
        banda = cv2.resize(banda, (ANCHO, max(20, int(round(banda.shape[0] * esc)))))
        for d in lz:
            y0 = int((d["fy"] - Y0) / (1.0 - Y0) * banda.shape[0])
            y1 = int((d["fy"] + d["fh"] - Y0) / (1.0 - Y0) * banda.shape[0])
            cv2.rectangle(banda, (int(d["fx"] * ANCHO), y0),
                          (int((d["fx"] + d["fw"]) * ANCHO), y1), (255, 255, 0), 2)
        detectado = " + ".join(f"{F4.EN_CANAL_CORTO[d['canal']]} {LADOS[d['lado']]}" for d in lz)
        cv2.putText(banda, f"{r['sede']}  frame {k}  detected: {detectado}",
                    (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        filas += [banda, np.full((2, ANCHO, 3), 150, np.uint8)]
        if len(filas) >= 2 * POR_FIGURA or i == len(refs) - 1:
            destino = os.path.join(SALIDA, f"light_count_{grupo}.png")
            cv2.imwrite(destino, np.vstack(filas))
            print(f"escrito {destino}  alto={sum(f.shape[0] for f in filas)}")
            filas, grupo = [], grupo + 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

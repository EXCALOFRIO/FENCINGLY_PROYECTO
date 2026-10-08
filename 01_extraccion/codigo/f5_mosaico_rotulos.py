"""Mosaicos de la BANDA INFERIOR de los 21 rotulos, para poder mirarlos de verdad y clasificarlos.

Existe porque `f4_censo_visual.py` dejo una cifra que NO pasa su criterio (15 de 21 sedes tienen su
region candidata mayor en la mitad inferior, y el criterio pedia 18) y la razon hay que verla, no
deducirla: en varios clips hay paneo de camara o un corte de plano entre el cuadro apagado y el
encendido, y entonces "lo que mas cambia" es la escena entera y no la luz. El mapa de estabilidad
no se retira por eso: se retira como LOCALIZADOR UNICO, y lo que queda es mirar.

Cada mosaico junta 7 sedes. De cada una, dos tiras:

  la banda y en [0.66, 1.00] del cuadro ENCENDIDO, con las cajas candidatas medidas en cian
  la MISMA banda del cuadro APAGADO, debajo

Dos tiras y no una porque lo que identifica una luz es la diferencia entre las dos, y porque el
censo tiene que describir tambien los CONFUNDIDOS, que son los rojos y verdes que estan en las dos.

No mide nada nuevo: reordena para mirar lo ya decodificado. Vuelve a decodificar los dos cuadros
porque guardar los 21 pares en disco costaria mas que los 42 saltos de `cap.set`.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

import f4_censo_visual as F4

SALIDA = os.path.join(F4.PROY, r"02_validacion\muestras\luces\censo")
ANCHO = 1000
Y0 = 0.78
POR_MOSAICO = 5


def main() -> int:
    os.makedirs(SALIDA, exist_ok=True)
    refs = F4.carga_referencias()
    piezas: list[np.ndarray] = []
    grupo = 0
    for i, r in enumerate(refs):
        info = F4.lee_cuadros(r["ruta"], r["t_touch_s"])
        if not info or info["encendido"] is None:
            continue
        enc, apa = info["encendido"], info["apagado"]
        _, cajas = F4.candidatas(apa, enc, info["pila"])
        H, W = enc.shape[:2]
        y0 = int(Y0 * H)
        vis = enc.copy()
        for c in cajas[:12]:
            cv2.rectangle(vis, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]),
                          (255, 255, 0), max(1, W // 640))
        esc = ANCHO / W
        tira_e = cv2.resize(vis[y0:H], (ANCHO, int(round((H - y0) * esc))))
        tira_a = cv2.resize(apa[y0:H], (ANCHO, int(round((H - y0) * esc))))
        cv2.putText(tira_e,
                    f"{r['sede']}  {W}x{H}  LIGHT ON frame {info['k_on']}  "
                    f"old detector: {r['detector_viejo']}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
        cv2.putText(tira_a, "LIGHTS OFF (2 s earlier)", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (0, 255, 255), 2)
        piezas += [tira_e, np.full((2, ANCHO, 3), 200, np.uint8), tira_a,
                   np.full((6, ANCHO, 3), 40, np.uint8)]
        if (i + 1) % POR_MOSAICO == 0 or i == len(refs) - 1:
            destino = os.path.join(SALIDA, f"mosaico_rotulos_{grupo}.png")
            cv2.imwrite(destino, np.vstack(piezas))
            print(f"escrito {destino}  alto={sum(p.shape[0] for p in piezas)}")
            piezas = []
            grupo += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

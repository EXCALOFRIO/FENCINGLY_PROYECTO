"""Herramienta para ANOTAR A OJO las luces de un clip: tira de fotogramas de la banda del rotulo.

Para que sirve. No existe verdad de referencia de luces en el proyecto. La unica forma de tenerla es
mirar el video, y para mirar 21 clips sin abrir 21 reproductores esta herramienta saca una TIRA: la
banda del rotulo, un fotograma por fila, desde antes del tocado hasta despues, con el numero de
cuadro escrito encima de cada fila. Con eso se lee de un golpe en que cuadro aparece cada luz y en
que orden.

Quien anota. Lo anoto YO, el agente, mirando estas tiras. NO es un arbitro y no tiene valor
reglamentario: distingue "se enciende la barra verde de la derecha en el cuadro 311" de "no se
enciende", que es lo unico que hace falta para medir un detector de luces. Lo que NO puede juzgar
esta anotacion es si el tocado era valido, si hubo prioridad ni si el arbitro lo dio.

Uso:
    python verifica_luces.py <sede-o-clip> [--paso 2] [--antes 0.7] [--despues 1.2] [--y0 0.78]

Si no se le pasa nada, saca la tira de las 21 sedes de la muestra de un clip por torneo.

ECONOMIA. Decodifica solo los cuadros de la tira con `cap.set`, no el clip. Con paso 2 y una ventana
de 1,9 s a 30 fps son 29 saltos por clip.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import f4_censo_visual as F4  # noqa: E402

SALIDA = os.path.join(F4.PROY, r"02_validacion\muestras\luces\tiras")
ANCHO = 1000


def tira(ruta: str, t_touch: float, destino: str, etiqueta: str,
         paso: int, antes: float, despues: float, y0f: float,
         x0f: float = 0.0, x1f: float = 1.0, y1f: float = 1.0) -> dict:
    """Escribe la tira y devuelve la lista de cuadros que contiene, para poder citarlos al anotar.

    El recorte en x existe porque hay familias de marcador en las que la luz es un PUNTO de 0,018 W
    (Tokio 2020) o un cuadradito de 0,012 W (Paris 2022): a la escala de la banda completa esos
    indicadores caen por debajo de un pixel del mosaico y no se pueden anotar a ojo.
    """
    cap = cv2.VideoCapture(ruta)
    if not cap.isOpened():
        return {}
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    k0 = max(0, int(round((t_touch - antes) * fps)))
    k1 = min(n - 1, int(round((t_touch + despues) * fps)))
    filas, cuadros = [], []
    for k in range(k0, k1 + 1, paso):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, img = cap.read()
        if not ok:
            continue
        H, W = img.shape[:2]
        banda = img[int(y0f * H):int(y1f * H), int(x0f * W):int(x1f * W)]
        esc = ANCHO / max(1, banda.shape[1])
        banda = cv2.resize(banda, (ANCHO, max(12, int(round(banda.shape[0] * esc)))),
                           interpolation=cv2.INTER_NEAREST)
        cv2.putText(banda, f"c{k}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        filas.append(banda)
        filas.append(np.full((1, ANCHO, 3), 120, np.uint8))
        cuadros.append(k)
    cap.release()
    if not filas:
        return {}
    cab = np.zeros((22, ANCHO, 3), np.uint8)
    cv2.putText(cab, f"{etiqueta}  fps={fps:.3f}  t_touch={t_touch:.3f}s  "
                     f"frames {k0}-{k1}, every {paso}",
                (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(destino, np.vstack([cab] + filas))
    return dict(cuadros=cuadros, fps=fps, n=n, k0=k0, k1=k1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cual", nargs="?", default="")
    ap.add_argument("--paso", type=int, default=2)
    ap.add_argument("--antes", type=float, default=0.7)
    ap.add_argument("--despues", type=float, default=1.2)
    ap.add_argument("--y0", type=float, default=0.78)
    ap.add_argument("--y1", type=float, default=1.0)
    ap.add_argument("--x0", type=float, default=0.0)
    ap.add_argument("--x1", type=float, default=1.0)
    a = ap.parse_args()

    os.makedirs(SALIDA, exist_ok=True)
    refs = F4.carga_referencias()
    if a.cual:
        refs = [r for r in refs if a.cual in (r["sede"], r["clip"])]
        if not refs:
            print(f"no encuentro '{a.cual}' en la muestra de un clip por torneo")
            return 2
    for r in refs:
        if not r["ruta"] or r["t_touch_s"] is None:
            print(f"sin ruta o sin t_touch: {r['sede']}")
            continue
        destino = os.path.join(SALIDA, f"{r['sede']}_tira.png")
        info = tira(r["ruta"], r["t_touch_s"], destino,
                    f"{r['sede']} / {r['clip']} / old detector: {r['detector_viejo']}, "
                    f"{r['n_luces_viejo']} valid light(s)",
                    a.paso, a.antes, a.despues, a.y0, a.x0, a.x1, a.y1)
        if info:
            print(f"{r['sede']:30s} {destino}  cuadros {info['k0']}-{info['k1']} "
                  f"({len(info['cuadros'])} filas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Paso 1 del encargo de las LUCES: saca fotogramas de los 21 clips para MIRAR los rotulos.

Este script no detecta nada. Su unico trabajo es dejar sobre la mesa, para cada una de las 21
sedes del corpus, tres fotogramas y la ficha tecnica del clip, para que el censo de marcadores se
haga MIRANDO y no suponiendo. El censo decide donde esta el marcador, si las luces son parte del
rotulo sobrepuesto o el aparato fisico de la pista, y que confundidos hay (banderas rojas, adornos
verdes del rotulo, leds del fondo). Sin ese censo cualquier umbral que escriba despues seria un
numero inventado.

CRITERIO DECLARADO ANTES DE EJECUTAR. Este paso sirve si y solo si:

  - se resuelven los 21 de 21 clips a un .mp4 real y legible, y
  - de cada uno salen 3 fotogramas decodificados sin error.

Si se resuelven menos de 21, NO se sigue al paso 2 con los que haya sin decir cuantos faltan y por
que. La cifra que decide es 21/21.

POR QUE SE ELIGEN ESOS TRES CUADROS. Al 20 %, 50 % y 80 % de la duracion. No se eligen "donde hay
luz" a proposito porque el censo tiene que describir el rotulo en reposo tambien: lo que hace
detectable una luz es precisamente que aparece sobre un rotulo que el resto del tiempo esta apagado.
Un cuadro con luz y dos sin ella es la comparacion que quiero tener delante.

ECONOMIA DE CPU. El usuario tiene el PC saturado. Se decodifica con `cap.set(CAP_PROP_POS_FRAMES)`
solo los cuadros pedidos, nunca el clip entero. Son 3 saltos por clip, 63 en total.
"""
from __future__ import annotations

import csv
import os
import sys

import cv2

BASE = r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO"
SCRATCH = (
    r"C:\Users\Alejandro\AppData\Local\Temp\claude"
    r"\C--Users-Alejandro-Desktop\187cfd83-d8d9-4c4d-bed4-f255724f6b21\scratchpad"
)
LISTA = os.path.join(SCRATCH, "un_clip_por_torneo.txt")
INDICE = os.path.join(SCRATCH, "indice_mp4.tsv")
SALIDA = os.path.join(SCRATCH, "censo")
FRACCIONES = (0.20, 0.50, 0.80)


def lee_lista(ruta: str) -> list[tuple[str, str]]:
    """Devuelve los pares (sede, clip) del fichero de muestra, uno por torneo."""
    pares = []
    with open(ruta, "r", encoding="utf-8") as fh:
        for linea in fh:
            linea = linea.strip()
            if not linea:
                continue
            sede, clip = linea.split("\t")
            pares.append((sede, clip))
    return pares


def lee_indice(ruta: str) -> dict[str, list[str]]:
    """Mapa stem -> rutas .mp4 encontradas bajo D:/FENCINGLY_VIDEOS_RAW.

    Se guarda la LISTA de rutas y no una sola, porque si un stem aparece dos veces hay que verlo y
    decirlo, no elegir en silencio. Un stem duplicado con dos videos distintos invalidaria la
    correspondencia sede-clip.
    """
    mapa: dict[str, list[str]] = {}
    with open(ruta, "r", encoding="utf-8-sig") as fh:
        for linea in fh:
            linea = linea.rstrip("\n").rstrip("\r")
            if not linea:
                continue
            stem, camino = linea.split("\t", 1)
            mapa.setdefault(stem, []).append(camino)
    return mapa


def main() -> int:
    os.makedirs(SALIDA, exist_ok=True)
    pares = lee_lista(LISTA)
    indice = lee_indice(INDICE)

    filas = []
    resueltos = 0
    cuadros_ok = 0
    for sede, clip in pares:
        caminos = indice.get(clip, [])
        if not caminos:
            print(f"FALTA\t{sede}\t{clip}\tno aparece en el indice de mp4")
            filas.append(dict(sede=sede, clip=clip, ruta="", ancho=0, alto=0,
                              cuadros=0, fps=0.0, duracion_s=0.0, copias=0))
            continue
        camino = caminos[0]
        cap = cv2.VideoCapture(camino)
        if not cap.isOpened():
            print(f"FALTA\t{sede}\t{clip}\tno abre: {camino}")
            cap.release()
            continue
        ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        resueltos += 1

        for i, frac in enumerate(FRACCIONES):
            k = max(0, min(n - 1, int(round(frac * (n - 1)))))
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, img = cap.read()
            if not ok:
                print(f"AVISO\t{sede}\t{clip}\tno decodifica el cuadro {k}")
                continue
            destino = os.path.join(SALIDA, f"{sede}__c{k:04d}.jpg")
            cv2.imwrite(destino, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            cuadros_ok += 1
        cap.release()

        filas.append(dict(sede=sede, clip=clip, ruta=camino, ancho=ancho, alto=alto,
                          cuadros=n, fps=round(fps, 3),
                          duracion_s=round(n / fps, 3) if fps > 0 else 0.0,
                          copias=len(caminos)))

    tabla = os.path.join(SCRATCH, "censo_ficha.csv")
    with open(tabla, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    print()
    print(f"clips resueltos: {resueltos} de {len(pares)}")
    print(f"cuadros escritos: {cuadros_ok} de {len(pares) * len(FRACCIONES)}")
    print(f"ficha: {tabla}")
    for f in filas:
        print(f"{f['sede']:30s} {f['ancho']}x{f['alto']} n={f['cuadros']:4d} "
              f"fps={f['fps']:6.2f} dur={f['duracion_s']:5.2f}s copias={f['copias']}")
    return 0 if resueltos == len(pares) else 1


if __name__ == "__main__":
    sys.exit(main())

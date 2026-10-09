"""Pinta la pareja elegida por la etapa 1 para mirarla a ojo (PAREJA_2026-10-09).

Un clip -> un JPG. Con una sola variante, 4 cuadros en 2x2; con varias (antes / despues de una
puerta), una FILA por variante y los MISMOS cuadros en todas, para comparar sin elegir nada.
Caja izquierda en verde, derecha en rojo, las demas detecciones de YOLO (`det_`) en gris.

    python mira_pareja.py --muestra 60 --semilla 20261009 --salida D:/FENCINGLY_SCRATCH/agente_pareja/m1
    python mira_pareja.py CLIP [CLIP ...] --salida DIR
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "01_extraccion" / "codigo"))
import rutas as RU  # noqa: E402

VERDE, ROJO, GRIS = (0, 220, 0), (0, 0, 255), (150, 150, 150)
ANCHO = 640


def cuadros_de(mask, n):
    ks = np.nonzero(mask)[0]
    if not len(ks):
        return []
    q = [(i + 0.5) / n for i in range(n)]
    return sorted({int(ks[min(int(x * len(ks)), len(ks) - 1)]) for x in q})


def lee(ruta, ks):
    cap = cv2.VideoCapture(ruta)
    out, k, objetivo = {}, 0, set(ks)
    while objetivo and k <= max(ks):
        ok, im = cap.read()
        if not ok:
            break
        if k in objetivo:
            out[k] = im
            objetivo.discard(k)
        k += 1
    cap.release()
    return out


def pinta(im, k, B, pres, cajas, cuadros, rotulo):
    H, W = im.shape[:2]
    im = im.copy()
    for b in cajas[cuadros == k]:
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), GRIS, 2)
    for lado, col in ((0, VERDE), (1, ROJO)):
        b = B[k, lado]
        if np.isfinite(b).all():
            x1, y1, x2, y2 = (b * np.array([W, H, W, H])).astype(int)
            grosor = 6 if pres[k, lado] else 2
            cv2.rectangle(im, (x1, y1), (x2, y2), col, grosor)
            cv2.putText(im, "L" if lado == 0 else "R", (x1 + 4, max(y1 - 8, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 4)
    esc = ANCHO / W
    im = cv2.resize(im, (ANCHO, int(H * esc)))
    cv2.putText(im, "%s c%d" % (rotulo, k), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 3)
    cv2.putText(im, "%s c%d" % (rotulo, k), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    return im


def mosaico(clip, variantes, salida, n=None, trabajo=None):
    """`variantes`: lista de (rotulo, B (T,2,4) normalizada, pres (T,2), mask (T,))."""
    trabajo = pathlib.Path(trabajo or RU.TRABAJO)
    ruta = RU.video(clip)
    if not ruta:
        return "sin video"
    dp = trabajo / ("det_%s.pkl" % clip)
    if dp.is_file():
        cajas, cuadros, _W, _H, _T = pickle.load(open(dp, "rb"))
    else:
        cajas, cuadros = np.zeros((0, 4)), np.zeros(0, int)
    n = n or (4 if len(variantes) == 1 else 3)
    mask = np.zeros_like(variantes[0][3])
    for v in variantes:
        mask = mask | v[3]
    ks = cuadros_de(mask, n)
    if not ks:
        return "sin cuadros utiles"
    ims = lee(ruta, ks)
    ks = [k for k in ks if k in ims]
    if not ks:
        return "no se pudo leer"
    filas = []
    for rot, B, pres, _m in variantes:
        fila = [pinta(ims[k], k, B, pres, cajas, cuadros, "%s %s" % (clip[:22], rot)) for k in ks]
        filas.append(fila)
    if len(variantes) == 1 and len(ks) == 4:
        rej = np.vstack([np.hstack(filas[0][:2]), np.hstack(filas[0][2:])])
    else:
        rej = np.vstack([np.hstack(f) for f in filas])
    salida = pathlib.Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(salida / ("%s.jpg" % clip)), rej, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "ok"


def de_revision(clip, trabajo=None):
    trabajo = pathlib.Path(trabajo or RU.TRABAJO)
    with np.load(trabajo / ("revision_%s.npz" % clip), allow_pickle=True) as z:
        return z["B"], z["pres"], z["mask"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--muestra", type=int, default=0)
    ap.add_argument("--semilla", type=int, default=20261009)
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    clips = list(a.clips)
    if a.muestra:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "01_extraccion"))
        import lotes as LO
        ids = sorted(c for c in LO.conjunto("anotados")
                     if (RU.TRABAJO / ("revision_%s.npz" % c)).is_file())
        clips = random.Random(a.semilla).sample(ids, a.muestra)
        print("poblacion %d clips con revision_, muestra %d (semilla %d)"
              % (len(ids), len(clips), a.semilla))
    salida = pathlib.Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    with open(salida / "lista.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(clips) + "\n")
    for c in clips:
        B, pres, mask = de_revision(c)
        print("%-40s %s" % (c, mosaico(c, [("elegida", B, pres, mask)], salida)), flush=True)


if __name__ == "__main__":
    main()

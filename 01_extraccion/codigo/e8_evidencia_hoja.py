"""ETAPA E, paso 8: la imagen que decide, con el CONTROL ROTADO pintado al lado.

Cuatro paneles por caso, a zoom alto y pegados a la mano:
  1  el recorte tal cual
  2  log10 del alfa de `matting`: aqui es donde se ve el filo
  3  los pixeles por encima del p99,5 del alfa fuera del cuerpo, con el CONO EN SU SITIO
  4  los MISMOS pixeles con el CONO GIRADO 90 grados: el control negativo, a la vista

Si el panel 3 tiene pixeles dentro del cono y el 4 no, el control pasa. Si los dos tienen lo mismo,
la deteccion no tiene nada que ver con el eje y hay que decirlo.
"""
from __future__ import annotations

import os
import pathlib
import sys

CACHE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565\scratchpad\hoja_seg\hf")
os.environ["HF_HOME"] = str(CACHE)

import cv2
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cono_lib as CL
import e7_pinta_cono as E7

SAL = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\cono")
CASOS = [("AdorableFittingAsiansmallclawedotter", 100, 0),
         ("AdorableFittingAsiansmallclawedotter", 296, 1),
         ("AdorableFittingAsiansmallclawedotter", 305, 0),
         ("SentimentalLargeBedlingtonterrier", 199, 0)]
PCTL = 99.5
AP = 25.4
"""Semiapertura medida: p90 de |angulo hoja - eje| a 1080p. Vease `e6_apertura_desde_la_hoja.py`."""


def panel(img, tit, pie=""):
    v = img.copy()
    cv2.rectangle(v, (0, 0), (v.shape[1], 22), (14, 14, 20), -1)
    cv2.putText(v, tit, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
    if pie:
        cv2.rectangle(v, (0, v.shape[0] - 20), (v.shape[1], v.shape[0]), (14, 14, 20), -1)
        cv2.putText(v, pie, (5, v.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 235, 255), 1)
    return v


for clip, k, lado in CASOS:
    z = CL.carga(clip)
    kp, sc, B = z["kp"], z["sc"], z["B"]
    W, H = int(z["W"]), int(z["H"])
    mano, _d = CL.mano_armada(kp, sc, B, W, H, lado)
    vert, u, ok, _l = CL.eje_antebrazo(kp, sc, lado, mano)
    _pk, pxm = CL.escala_px_m(B, lado, H)
    L = CL.HOJA_M * pxm
    if not ok[k]:
        print("  %s k=%d lado=%d sin eje valido" % (clip, k, lado))
        continue
    cap = cv2.VideoCapture(str(z["video"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    okf, im = cap.read()
    cap.release()
    if not okf:
        continue
    (cx0, cy0, cx1, cy1), R = E7.recorte(B[k, lado], B[k, 1 - lado], vert[k], u[k], L, W, H)
    sub = np.ascontiguousarray(im[cy0:cy1, cx0:cx1, ::-1])
    alfa, seg = E7.infiere(sub)
    cuerpo = cv2.dilate((seg != 0).astype("uint8"), np.ones((2 * E7.DILATA + 1,) * 2, np.uint8)) > 0
    fuera = alfa[~cuerpo]
    t = float(np.percentile(fuera, PCTL))
    hh, ww = alfa.shape[:2]
    GY, GX = np.mgrid[0:hh, 0:ww]
    disco = np.hypot(GX + cx0 - vert[k][0], GY + cy0 - vert[k][1]) <= R
    m = disco & (~cuerpo) & (alfa > t)

    bgr = np.ascontiguousarray(sub[:, :, ::-1])
    la = np.log10(np.clip(alfa, 1e-6, 1.0))
    heat = cv2.applyColorMap((np.clip((la + 4.0) / 4.0, 0, 1) * 255).astype("uint8"),
                             cv2.COLORMAP_TURBO)
    P = np.stack([GX.ravel() + cx0, GY.ravel() + cy0], axis=1).astype("float64")
    a0 = np.degrees(np.arctan2(u[k][1], u[k][0]))
    paneles = [panel(bgr, "1  recorte de UN tirador  ·  %s k=%d lado %d" % (clip[:22], k, lado),
                     "R = %.0f px = %.0f cm de los 90" % (R, 100 * R / pxm)),
               panel(heat, "2  log10 del alfa de matting",
                     "fondo mediana %.1e   umbral p%.1f = %.4f" % (np.median(fuera), PCTL, t))]
    for rot, nom in ((0.0, "3  pixeles > umbral  ·  CONO EN SU SITIO"),
                     (90.0, "4  los MISMOS pixeles  ·  CONO GIRADO 90 (control)")):
        th = np.radians(a0 + rot)
        ur = np.array([np.cos(th), np.sin(th)])
        w = (CL.peso_sector(P, vert[k], ur, R, AP) > 0).reshape(hh, ww)
        vis = np.zeros(alfa.shape + (3,), np.uint8)
        vis[cuerpo] = (48, 48, 48)
        vis[m] = (0, 255, 255)
        vis[m & w] = (60, 255, 60)
        cnt, _h = cv2.findContours(w.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnt, -1, (255, 160, 40), 1)
        cv2.circle(vis, tuple(np.int32(vert[k] - [cx0, cy0])), 4, (255, 255, 255), -1)
        dentro = int((m & w).sum())
        paneles.append(panel(vis, nom, "%d de %d px dentro (%.1f %%)" % (
            dentro, int(m.sum()), 100.0 * dentro / max(int(m.sum()), 1))))
    alt = 560
    outs = []
    for p in paneles:
        e = alt / p.shape[0]
        outs.append(cv2.resize(p, (int(p.shape[1] * e), alt), interpolation=cv2.INTER_NEAREST))
    fila = np.hstack(outs)
    cv2.rectangle(fila, (0, fila.shape[0] - 24), (fila.shape[1], fila.shape[0]), (14, 14, 20), -1)
    cv2.putText(fila, "amarillo = alfa alto fuera del cuerpo   verde = dentro del cono   "
                      "naranja = borde del cono (semiapertura %.1f gr, MEDIDA)" % AP,
                (8, fila.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1)
    f = SAL / ("evidencia_hoja_%s_%d_l%d.png" % (clip[:20], k, lado))
    cv2.imwrite(str(f), fila)
    print("%s   ->   %d px detectados, %.1f %% dentro del cono en su sitio" % (
        f.name, int(m.sum()),
        100.0 * float(((m) & (CL.peso_sector(P, vert[k], u[k], R, AP) > 0).reshape(hh, ww)).sum())
        / max(int(m.sum()), 1)))
    print("     %s" % f)

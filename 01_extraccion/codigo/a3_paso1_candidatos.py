"""ETAPA A3, paso 1: YOLO propone TODOS los candidatos de persona por cuadro y se guardan.

POR QUE DOS PASOS: `ultralytics` esta en `.venv-gpu` y `transformers` con Sapiens esta en
`cuda/.venv`. No conviven, asi que este paso escribe un fichero intermedio y el paso 2 lo puntua.

QUE HACE: para cada cuadro del clip, ejecuta YOLO11m-pose y guarda TODAS las personas detectadas con
su caja, su confianza y sus 17 puntos. Medido antes: **YOLO devuelve unas 8 personas por cuadro**, y
ahi esta el problema de la etapa A — hay que elegir 2 de 8, y el algoritmo viejo elegia mal.

NO SE FILTRA NADA AQUI a proposito. Filtrar en este paso escondería candidatos al paso 2 y no se
podria saber si el error fue de la propuesta o de la eleccion. Se guarda todo y se decide despues.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import cv2
import numpy as np
import torch

BASE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565")
PESOS = BASE / "scratchpad/smoke/weights/yolo11m-pose.pt"
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
SALIDA.mkdir(parents=True, exist_ok=True)

CLIP = sys.argv[1] if len(sys.argv) > 1 else "SoreEasygoingEmperorshrimp"
IMGSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 960
CONF = 0.10
"""Confianza baja a proposito: queremos VER a los candidatos flojos, porque el tirador real puede
salir con confianza baja si esta parcialmente ocluido, y el algoritmo viejo lo perdia."""


def localiza(cid):
    for raiz, _d, fs in os.walk("D:/FENCINGLY_VIDEOS_RAW"):
        if "%s.mp4" % cid in fs:
            return os.path.join(raiz, "%s.mp4" % cid)
    return None


mp4 = localiza(CLIP)
if not mp4:
    raise SystemExit("no encuentro el video de %s" % CLIP)

from ultralytics import YOLO

y = YOLO(str(PESOS))
y.to("cuda")
print("YOLO en %s   imgsz %d   conf %.2f" % (next(y.model.parameters()).device, IMGSZ, CONF))

cap = cv2.VideoCapture(mp4)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
FPS = float(cap.get(cv2.CAP_PROP_FPS))
print("clip %s: %dx%d, %d cuadros, %.2f fps" % (CLIP, W, H, T, FPS))

cajas, kps, confs, cuadros = [], [], [], []
t0 = time.perf_counter()
k = 0
while True:
    ok, im = cap.read()
    if not ok:
        break
    r = y.predict(im, imgsz=IMGSZ, conf=CONF, verbose=False, device=0)[0]
    if r.boxes is not None and len(r.boxes):
        b = r.boxes.xyxy.cpu().numpy()
        c = r.boxes.conf.cpu().numpy()
        kk = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((len(b), 17, 3))
        for i in range(len(b)):
            cajas.append(b[i])
            confs.append(float(c[i]))
            kps.append(kk[i])
            cuadros.append(k)
    k += 1
cap.release()
dt = time.perf_counter() - t0

cajas = np.asarray(cajas, dtype="float32") if cajas else np.zeros((0, 4), "float32")
kps = np.asarray(kps, dtype="float32") if len(kps) else np.zeros((0, 17, 3), "float32")
confs = np.asarray(confs, dtype="float32")
cuadros = np.asarray(cuadros, dtype="int32")

print()
print("candidatos totales ......... %d" % len(cajas))
print("cuadros con al menos uno ... %d de %d" % (len(np.unique(cuadros)), k))
if len(cuadros):
    porc = np.bincount(cuadros, minlength=k)
    print("candidatos por cuadro ...... mediana %.1f   min %d   max %d" % (
        np.median(porc), porc.min(), porc.max()))
    print("cuadros SIN ningun candidato %d" % int((porc == 0).sum()))
    alto = (cajas[:, 3] - cajas[:, 1]) / H
    print()
    print("ALTO DE CAJA como fraccion del alto de imagen")
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        print("   p%-3d %.3f" % (int(100 * q), np.quantile(alto, q)))
    print("CONFIANZA de YOLO")
    for q in (0.1, 0.5, 0.9):
        print("   p%-3d %.3f" % (int(100 * q), np.quantile(confs, q)))

f = SALIDA / ("candidatos_%s.npz" % CLIP)
np.savez_compressed(f, cajas=cajas, kps=kps, confs=confs, cuadros=cuadros,
                    W=W, H=H, T=k, fps=FPS, video=str(mp4))
print()
print("velocidad .................. %.1f ms/cuadro  (%d cuadros en %.1f s)" % (
    1000.0 * dt / max(k, 1), k, dt))
print("guardado ................... %s" % f)

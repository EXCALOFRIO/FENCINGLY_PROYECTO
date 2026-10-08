"""En que fraccion de los cuadros hay REALMENTE dos tiradores en el plano?

POR QUE ESTA PREGUNTA MANDA SOBRE EL REQUISITO: el usuario pide "que siempre salgan solo 2 poses
completas, y si falta una la interpolamos". Al mirar un cuadro concreto se vio que **a veces solo hay
un tirador en la imagen** porque el otro esta fuera de camara. Ahi interpolar no rellena un hueco:
inventa a una persona que no se ve. Asi que antes de decidir la regla hay que saber cuanto pasa.

La medida se hace con YOLO sobre el VIDEO COMPLETO, no sobre la ventana del npz, porque la ventana se
eligio precisamente alrededor de la accion y por tanto esta sesgada a favor de tener a los dos.

CRITERIO: se cuenta cuantas de las dos trayectorias elegidas tienen caja en cada cuadro. Se reporta
la fraccion de cuadros con 2, con 1 y con 0, y se separa por posicion en el clip (principio, medio,
final), porque la sospecha es que los cuadros con uno solo estan en los extremos.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import a3_selector as SEL

BASE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565")
PESOS = BASE / "scratchpad/smoke/weights/yolo11m-pose.pt"
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
N_CLIPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
CONF = 0.10
IMGSZ = 960

mapa = {}
for raiz, _d, fs in os.walk("D:/FENCINGLY_VIDEOS_RAW"):
    for f in fs:
        if f.lower().endswith(".mp4"):
            mapa.setdefault(pathlib.Path(f).stem, os.path.join(raiz, f))
print("mp4 disponibles: %d" % len(mapa))

rng = np.random.default_rng(20260910)
clips = sorted(mapa)
rng.shuffle(clips)
clips = clips[:N_CLIPS]

from ultralytics import YOLO

y = YOLO(str(PESOS))
y.to("cuda")

filas = []
for n, cid in enumerate(clips, 1):
    cap = cv2.VideoCapture(mapa[cid])
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if W < 2 or H < 2:
        cap.release()
        continue
    cajas, cuadros = [], []
    k = 0
    t0 = time.perf_counter()
    while True:
        ok, im = cap.read()
        if not ok:
            break
        r = y.predict(im, imgsz=IMGSZ, conf=CONF, verbose=False, device=0)[0]
        if r.boxes is not None and len(r.boxes):
            b = r.boxes.xyxy.cpu().numpy()
            for i in range(len(b)):
                cajas.append(b[i])
                cuadros.append(k)
        k += 1
    cap.release()
    if k < 10 or not cajas:
        continue
    cajas = np.asarray(cajas, dtype="float32")
    cuadros = np.asarray(cuadros, dtype="int32")
    tr = SEL.construye_trayectorias(cajas, cuadros, W, H, k)
    res = SEL.elige_dos(tr, W, H, k)
    B, pres = SEL.a_matrices(res["elegidas"], cajas, k, W, H)
    cuenta = pres.sum(axis=1)
    tercio = k // 3
    fila = {"clip": cid, "T": k, "W": W, "H": H,
            "con2": float((cuenta == 2).mean()), "con1": float((cuenta == 1).mean()),
            "con0": float((cuenta == 0).mean()),
            "con2_ini": float((cuenta[:tercio] == 2).mean()) if tercio else float("nan"),
            "con2_med": float((cuenta[tercio:2 * tercio] == 2).mean()) if tercio else float("nan"),
            "con2_fin": float((cuenta[2 * tercio:] == 2).mean()) if tercio else float("nan"),
            "trayectorias": len(tr), "motivo": res["motivo"],
            "ms_cuadro": 1000.0 * (time.perf_counter() - t0) / k}
    filas.append(fila)
    print("  %2d/%d %-40s T=%4d  con2 %5.1f %%  con1 %5.1f %%  con0 %4.1f %%  (ini %4.0f / med %4.0f / fin %4.0f)" % (
        n, len(clips), cid[:40], k, 100 * fila["con2"], 100 * fila["con1"], 100 * fila["con0"],
        100 * fila["con2_ini"], 100 * fila["con2_med"], 100 * fila["con2_fin"]))

import pandas as pd
d = pd.DataFrame(filas)
d.to_csv(SALIDA / "cuantos_en_plano.csv", index=False)

print()
print("=" * 92)
print("CUANTOS TIRADORES HAY EN EL PLANO, sobre %d clips y %d cuadros" % (len(d), int(d["T"].sum())))
print("=" * 92)
if len(d):
    print("  cuadros con LOS DOS ..... %5.1f %%   (mediana por clip %5.1f %%)" % (
        100 * float((d["con2"] * d["T"]).sum() / d["T"].sum()), 100 * d["con2"].median()))
    print("  cuadros con UNO SOLO .... %5.1f %%" % (100 * float((d["con1"] * d["T"]).sum() / d["T"].sum())))
    print("  cuadros con NINGUNO ..... %5.1f %%" % (100 * float((d["con0"] * d["T"]).sum() / d["T"].sum())))
    print()
    print("  POR POSICION EN EL CLIP: fraccion de cuadros con los dos")
    print("    primer tercio ... %5.1f %%" % (100 * d["con2_ini"].median()))
    print("    tercio central .. %5.1f %%" % (100 * d["con2_med"].median()))
    print("    ultimo tercio ... %5.1f %%" % (100 * d["con2_fin"].median()))
    print()
    print("  clips donde los dos estan en >= 95 %% de los cuadros ... %d de %d  (%.0f %%)" % (
        int((d["con2"] >= 0.95).sum()), len(d), 100.0 * (d["con2"] >= 0.95).mean()))
    print("  clips donde los dos estan en <  50 %% ................. %d" % int((d["con2"] < 0.50).sum()))
    print()
    print("  velocidad de YOLO ... %.1f ms/cuadro" % d["ms_cuadro"].median())
    print()
    print("  LO QUE ESTO SIGNIFICA PARA EL REQUISITO")
    print("    'siempre 2 poses completas' solo es alcanzable en el %.1f %% de los cuadros." % (
        100 * float((d["con2"] * d["T"]).sum() / d["T"].sum())))
    print("    En el resto NO hay dos tiradores en la imagen, asi que interpolar no rellena un")
    print("    hueco: inventa a alguien que no se ve.")

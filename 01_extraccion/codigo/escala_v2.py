"""Escala la cadena de `extrae_pareja` a N clips y da la cifra agregada, con los peores por delante.

CADENCIA DEL PLAN, que pidio el usuario: 1 -> 2 -> 5 -> 10 -> 100, y no se pasa al siguiente sin la
cifra del anterior. La semilla es fija, asi que los primeros N clips de una tirada de 12 son los
mismos 5 de la tirada de 5 mas 7 nuevos: las cifras se pueden comparar entre tiradas.

LO QUE SE MIDE, y todo es comprobable a ojo despues:
  cuadros con LOS DOS tiradores, antes y despues de la redeteccion
  cuadros que la puerta de planos deja fuera, con sus tramos, para poder mirarlos
  cuantos siguen faltando y de que tipo, que es lo que queda por resolver
  aviso cuando ninguna pareja pasa las puertas de geometria, que es el sintoma de mala eleccion

Las detecciones se cachean en el scratchpad: repetir la tirada despues de tocar el selector no
vuelve a pasar YOLO, y eso baja una tirada de 12 clips de minutos a segundos.
"""
from __future__ import annotations

import os
import pathlib
import pickle
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import extrae_pareja as EP

BASE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565")
PESOS = BASE / "scratchpad/smoke/weights/yolo11m-pose.pt"
CACHE = pathlib.Path(os.environ.get(
    "SCRATCH", r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\187cfd83-d8d9-4c4d-bed4-f255724f6b21\scratchpad"))
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
SALIDA.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
VIDEOS = "D:/FENCINGLY_VIDEOS_RAW"

mapa = {}
for raiz, _d, fs in os.walk(VIDEOS):
    for f in fs:
        if f.lower().endswith(".mp4"):
            mapa.setdefault(pathlib.Path(f).stem, os.path.join(raiz, f))
rng = np.random.default_rng(20260910)
clips = sorted(mapa)
rng.shuffle(clips)
clips = clips[:N]
print("clips: %d de %d disponibles" % (len(clips), len(mapa)))

from ultralytics import YOLO
yolo = YOLO(str(PESOS))
yolo.to("cuda")
print("YOLO en cuda")
print()

filas = []
print("%-38s %5s %5s %5s %5s %5s %7s %7s %5s" % (
    "clip", "T", "tray", "plano", "fund", "recor", "antes", "ahora", "falt"))
print("-" * 92)
for n, cid in enumerate(clips, 1):
    t0 = time.perf_counter()
    ca = CACHE / ("det_%s.pkl" % cid)
    frames = None
    if ca.exists():
        with open(ca, "rb") as fh:
            cajas, cuadros, W, H, T = pickle.load(fh)
        cap = cv2.VideoCapture(mapa[cid])
        frames = []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            frames.append(im)
        cap.release()
    else:
        cap = cv2.VideoCapture(mapa[cid])
        if int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) < 2:
            cap.release()
            continue
        cajas, cuadros, W, H, T, frames = EP.detecta_clip(yolo, cap, cv2)
        cap.release()
        with open(ca, "wb") as fh:
            pickle.dump((cajas, cuadros, W, H, T), fh)
    if T < 10 or not len(cajas):
        continue

    gris = [cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype("float32")
            for f in frames]
    r = EP.procesa(cajas, cuadros, W, H, T, yolo=yolo, frames=frames, gris=gris)
    aviso = ""
    if "AVISO" in r["motivo"]:
        aviso = "  <- SIN PAREJA VALIDA"
    elif r["ambos"] < 0.75:
        aviso = "  <- MIRAR A OJO"
    print("%-38s %5d %5d %5d %5d %5d %6.1f%% %6.1f%% %5d%s" % (
        cid[:38], T, r["ntray"], r["fuera"], len(r["fundidos"]), r["recortados"],
        100 * r["ambos_sin_redeteccion"], 100 * r["ambos"], r["faltan"], aviso))
    filas.append({"clip": cid, "T": T, "W": W, "H": H, "ntray": r["ntray"], "nfus": r["nfus"],
                  "fuera": r["fuera"], "tramos": str(r["tramos"]),
                  "fundidos": str(r["fundidos"]), "recortados": r["recortados"],
                  "redet": r["nredetectados"], "faltan": r["faltan"],
                  "ambos_antes": r["ambos_sin_redeteccion"], "ambos": r["ambos"],
                  "motivo": r["motivo"], "seg": time.perf_counter() - t0})

import pandas as pd
d = pd.DataFrame(filas)
csv = SALIDA / ("escala_v2_%dclips.csv" % len(d))
d.to_csv(csv, index=False)
print()
print("=" * 88)
print("AGREGADO sobre %d clips y %d cuadros" % (len(d), int(d["T"].sum())))
print("=" * 88)
if len(d):
    print("  los dos presentes: mediana %5.1f %%   p10 %5.1f %%   min %5.1f %%" % (
        100 * d["ambos"].median(), 100 * d["ambos"].quantile(0.1), 100 * d["ambos"].min()))
    print("  aporte de la redeteccion: %+.2f pp de mediana, %d cuadros recuperados" % (
        100 * (d["ambos"] - d["ambos_antes"]).median(), int(d["redet"].sum())))
    print("  clips con los dos en >= 90 %% ..... %d de %d  (%.0f %%)" % (
        int((d["ambos"] >= 0.90).sum()), len(d), 100.0 * (d["ambos"] >= 0.90).mean()))
    print("  clips con los dos en <  75 %% ..... %d" % int((d["ambos"] < 0.75).sum()))
    print("  clips sin pareja valida .......... %d" % int(d["motivo"].str.contains("AVISO").sum()))
    print("  cuadros que quita la puerta de planos: %d de %d (%.1f %%)" % (
        int(d["fuera"].sum()), int(d["T"].sum()), 100.0 * d["fuera"].sum() / d["T"].sum()))
    print("  cuadros que siguen faltando: %d de %d (%.1f %%)" % (
        int(d["faltan"].sum()), 2 * int(d["T"].sum()), 100.0 * d["faltan"].sum() / (2 * d["T"].sum())))
    print("  coste: %.1f s por clip de mediana" % d["seg"].median())
    print()
    print("  LOS QUE HAY QUE MIRAR A OJO, por orden de sospecha")
    for _, q in d.sort_values("ambos").head(6).iterrows():
        print("    %-40s ambos %5.1f %%  faltan %4d  %s" % (
            q["clip"][:40], 100 * q["ambos"], q["faltan"], q["motivo"][:52]))
print()
print("  csv en %s" % csv)

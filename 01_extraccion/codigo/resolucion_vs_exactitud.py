"""Que resolucion vale: se compara la POSICION de los puntos contra la referencia de 1024x768.

POR QUE NO BASTA LA CONFIANZA, y esto corrige una medida mia de hace un momento: use el criterio
"fraccion de puntos con confianza >= 0,3" y **todas las resoluciones lo pasaban al 100 %**, incluida
256x192 donde el tronco baja de 0,884 a 0,410. Un criterio que aprueba todo no mide nada.

La confianza tampoco es exactitud: un punto puede salir con confianza baja y estar donde debe. Lo que
decide es **cuanto se mueve el punto** respecto de la mejor version disponible.

CRITERIO DECLARADO ANTES DE MEDIR:
  referencia = 1024x768 (la entrada nativa del modelo)
  error = distancia en pixeles de imagen entre el punto de la variante y el de la referencia
  se normaliza por el ALTO DE LA CAJA, para que un error de 5 px signifique lo mismo en un plano
  corto y en uno largo
  ACEPTABLE si el error mediano se queda por debajo del 2 % del alto de la caja, que en una caja de
  155 px son unos 3 px. Ese limite se fija AHORA, antes de ver los numeros.

Se mira por separado el TRONCO CON BRAZOS (5-10) y la CADENA DEL PULGAR (112-116), porque el pulgar
es el eje del cono y es el punto mas pequeno: si algo se degrada primero, sera ese.
"""
from __future__ import annotations

import os
import pathlib
import time

CACHE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565\scratchpad\hoja_seg\hf")
os.environ["HF_HOME"] = str(CACHE)

import cv2
import numpy as np
import torch

BASE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565")
PESOS = CACHE / "hub/models--facebook--sapiens2-pose-0.4b/snapshots/73572e121602e8a6eb40eb841732e88f386d250b"
NPZ_DIRS = [BASE / "localrun/runs_extra1/data_20260907T005507157743Z/npz",
            BASE / "localrun/runs_extra2/data_20260907T005507370214Z/npz",
            BASE / "scratchpad/catalogo/ws_completo/npz"]
CLIPS = ["SentimentalLargeBedlingtonterrier", "AdorableFittingAsiansmallclawedotter"]
N = 8
TRONCO = list(range(5, 11))
PULGAR = [112, 113, 114, 115, 116]
LIMITE_FRAC = 0.02
RESOLUCIONES = [(1024, 768), (768, 576), (640, 480), (512, 384), (384, 288), (256, 192)]

from transformers import Sapiens2ForPoseEstimation, Sapiens2ImageProcessor

muestras = []
for clip in CLIPS:
    mp4 = None
    for raiz, _d, fs in os.walk("D:/FENCINGLY_VIDEOS_RAW"):
        if "%s.mp4" % clip in fs:
            mp4 = os.path.join(raiz, "%s.mp4" % clip)
            break
    fz = next((d / ("%s.npz" % clip) for d in NPZ_DIRS if (d / ("%s.npz" % clip)).is_file()), None)
    if not mp4 or not fz:
        continue
    with np.load(fz, allow_pickle=True) as z:
        BOX = np.asarray(z["boxes"], dtype="float64")
    cap = cv2.VideoCapture(mp4)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ini = max(0, T // 2 - N // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, ini)
    for i in range(ini, min(T, ini + N)):
        ok, im = cap.read()
        if not ok:
            break
        rgb = np.ascontiguousarray(im[:, :, ::-1])
        for s in (0, 1):
            b = BOX[i, s]
            if np.isfinite(b).all() and b[2] > b[0]:
                x, y = float(b[0] * W), float(b[1] * H)
                bw, bh = float(b[2] * W) - x, float(b[3] * H) - y
                if bh > 20:
                    muestras.append({"img": rgb, "caja": [[x, y, bw, bh]], "alto_caja": bh})
    cap.release()

print("GPU: %s" % torch.cuda.get_device_name(0))
print("muestras (persona x cuadro): %d de %d clips" % (len(muestras), len(CLIPS)))
print("alto de caja: mediana %.0f px, min %.0f, max %.0f" % (
    np.median([m["alto_caja"] for m in muestras]),
    min(m["alto_caja"] for m in muestras), max(m["alto_caja"] for m in muestras)))
print("limite declarado: error mediano < %.0f %% del alto de caja  (~%.1f px con la caja mediana)" % (
    100 * LIMITE_FRAC, LIMITE_FRAC * np.median([m["alto_caja"] for m in muestras])))


def corre(size):
    proc = Sapiens2ImageProcessor.from_pretrained(str(PESOS))
    proc.size = {"height": size[0], "width": size[1]}
    mod = Sapiens2ForPoseEstimation.from_pretrained(str(PESOS), dtype=torch.float16).to("cuda").eval()
    with torch.inference_mode():
        e = proc(images=[muestras[0]["img"]], boxes=[muestras[0]["caja"]], return_tensors="pt")
        e = {k: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda")) for k, v in e.items()}
        _ = mod(**e)
    torch.cuda.synchronize()
    kps, t0 = [], time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(muestras), 2):
            lote = muestras[i:i + 2]
            e = proc(images=[m["img"] for m in lote], boxes=[m["caja"] for m in lote],
                     return_tensors="pt")
            e = {k: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda"))
                 for k, v in e.items()}
            o = mod(**e)
            r = proc.post_process_pose_estimation(o, boxes=[m["caja"] for m in lote], threshold=None)
            for j in range(len(lote)):
                rr = r[j][0] if isinstance(r[j], list) else r[j]
                k = rr["keypoints"]
                kps.append(k.cpu().numpy() if hasattr(k, "cpu") else np.asarray(k))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    del mod
    torch.cuda.empty_cache()
    return np.stack(kps), 1000.0 * dt / len(muestras)


print()
print("--- corriendo la referencia 1024x768")
ref, ms_ref = corre(RESOLUCIONES[0])
print("    %.1f ms/pasada" % ms_ref)

altos = np.array([m["alto_caja"] for m in muestras])[:, None]
print()
print("=" * 100)
print("%-12s %9s %8s %12s %12s %12s %12s" % (
    "resolucion", "ms/pasada", "acelera", "err tronco", "err pulgar", "tronco %caja", "pulgar %caja"))
print("=" * 100)
filas = []
for size in RESOLUCIONES:
    kp, ms = (ref, ms_ref) if size == RESOLUCIONES[0] else corre(size)
    d = np.linalg.norm(kp[:, :, :2] - ref[:, :, :2], axis=2)
    et = float(np.median(d[:, TRONCO]))
    ep = float(np.median(d[:, PULGAR]))
    ft = float(np.median(d[:, TRONCO] / altos))
    fp = float(np.median(d[:, PULGAR] / altos))
    filas.append({"size": size, "ms": ms, "et": et, "ep": ep, "ft": ft, "fp": fp})
    marca = "" if max(ft, fp) < LIMITE_FRAC else "   <- PASA DEL LIMITE"
    print("%-12s %9.1f %7.1fx %10.2f px %10.2f px %10.2f %% %10.2f %%%s" % (
        "%dx%d" % size, ms, ms_ref / ms, et, ep, 100 * ft, 100 * fp, marca))

print()
print("VEREDICTO CONTRA EL LIMITE DECLARADO (< %.0f %% del alto de caja)" % (100 * LIMITE_FRAC))
buenas = [f for f in filas if max(f["ft"], f["fp"]) < LIMITE_FRAC and f["size"] != RESOLUCIONES[0]]
if buenas:
    mejor = min(buenas, key=lambda f: f["ms"])
    print("  la mas rapida que lo cumple: %dx%d  con %.1f ms/pasada  (%.1fx)" % (
        mejor["size"][0], mejor["size"][1], mejor["ms"], ms_ref / mejor["ms"]))
    ms_cuadro = mejor["ms"] * 2
    h = 12e6 * ms_cuadro / 1000.0 / 3600.0
    print("  -> %.0f ms por cuadro de video con 2 tiradores" % ms_cuadro)
    print("  -> 12 M cuadros en %.0f h = %.1f dias" % (h, h / 24))
else:
    print("  NINGUNA baja del limite. Hay que quedarse en 1024x768 o subir el limite y decirlo.")

"""Velocidad de Sapiens 2 contra el modelo actual (YOLO11m-pose), en la misma GPU y los mismos cuadros.

LA COMPARACION JUSTA NO ES PASADA CONTRA PASADA, y conviene decirlo antes de dar numeros:

  YOLO11m-pose es de UNA SOLA PASADA POR CUADRO: detecta a todas las personas de golpe y da 17 puntos.
  Sapiens 2 es TOP-DOWN: una pasada POR PERSONA, y da 308 puntos.

Asi que en un clip de esgrima con dos tiradores, un cuadro cuesta 1 pasada de YOLO y 2 de Sapiens.
La medida que decide el proyecto es **segundos por cuadro de video**, no por pasada.

Se comprueba tambien que las dos corren de verdad en la GPU, imprimiendo el dispositivo de los pesos
y la VRAM ocupada: un modelo que se cae a CPU sin avisar daria un numero lentisimo por otro motivo.
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
PESOS_SAP = CACHE / "hub/models--facebook--sapiens2-pose-0.4b/snapshots/73572e121602e8a6eb40eb841732e88f386d250b"
PESOS_YOLO = BASE / "scratchpad/smoke/weights/yolo11m-pose.pt"
NPZ_DIRS = [BASE / "localrun/runs_extra1/data_20260907T005507157743Z/npz",
            BASE / "localrun/runs_extra2/data_20260907T005507370214Z/npz",
            BASE / "scratchpad/catalogo/ws_completo/npz"]
CLIP = "SentimentalLargeBedlingtonterrier"
N = 16
LOTES = (1, 2, 4, 8)

print("GPU: %s" % (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO HAY CUDA"))
print()

mp4 = None
for raiz, _d, fs in os.walk("D:/FENCINGLY_VIDEOS_RAW"):
    if "%s.mp4" % CLIP in fs:
        mp4 = os.path.join(raiz, "%s.mp4" % CLIP)
        break
fz = next(d / ("%s.npz" % CLIP) for d in NPZ_DIRS if (d / ("%s.npz" % CLIP)).is_file())
with np.load(fz, allow_pickle=True) as z:
    BOX = np.asarray(z["boxes"], dtype="float64")

cap = cv2.VideoCapture(mp4)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ini = max(0, T // 2 - N // 2)
cap.set(cv2.CAP_PROP_POS_FRAMES, ini)
cuadros = []
for i in range(ini, min(T, ini + N)):
    ok, im = cap.read()
    if not ok:
        break
    cuadros.append((i, im))
cap.release()
print("clip %s: %dx%d, %d cuadros de prueba" % (CLIP, W, H, len(cuadros)))


def cajas_xywh(idx):
    out = []
    for s in (0, 1):
        b = BOX[idx, s]
        if np.isfinite(b).all() and b[2] > b[0]:
            x, y = float(b[0] * W), float(b[1] * H)
            out.append([x, y, float(b[2] * W) - x, float(b[3] * H) - y])
    return out


resultados = {}

print()
print("=" * 82)
print("1) EL MODELO ACTUAL: YOLO11m-pose, imgsz 960, una pasada por cuadro, 17 puntos")
print("=" * 82)
try:
    from ultralytics import YOLO

    y = YOLO(str(PESOS_YOLO))
    y.to("cuda")
    _ = y.predict(cuadros[0][1], imgsz=960, verbose=False, device=0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    npers = 0
    for _idx, im in cuadros:
        r = y.predict(im, imgsz=960, verbose=False, device=0)[0]
        npers += 0 if r.keypoints is None else len(r.keypoints)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ms = 1000.0 * dt / len(cuadros)
    print("  dispositivo ............ %s" % next(y.model.parameters()).device)
    print("  VRAM pico .............. %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
    print("  personas detectadas .... %d en %d cuadros" % (npers, len(cuadros)))
    print("  MS POR CUADRO .......... %.1f ms" % ms)
    print("  cuadros por segundo .... %.1f" % (1000.0 / ms))
    resultados["yolo11m-pose 960"] = ms
    del y
    torch.cuda.empty_cache()
except Exception as exc:
    print("  FALLO: %s %s" % (type(exc).__name__, str(exc)[:110]))

print()
print("=" * 82)
print("2) SAPIENS 2 pose 0.4b, 1024x768, una pasada POR PERSONA, 308 puntos")
print("=" * 82)
from transformers import Sapiens2ForPoseEstimation, Sapiens2ImageProcessor

proc = Sapiens2ImageProcessor.from_pretrained(str(PESOS_SAP))
mod = Sapiens2ForPoseEstimation.from_pretrained(str(PESOS_SAP), dtype=torch.float16).to("cuda").eval()
print("  dispositivo ............ %s" % next(mod.parameters()).device)
print("  dtype .................. %s" % next(mod.parameters()).dtype)

"""Se prueba con varios tamanos de lote: al ser top-down, agrupar personas y cuadros en una sola
pasada es la unica palanca de rendimiento que no cambia el resultado."""
for lote in LOTES:
    pend_im, pend_cj = [], []
    for _idx, im in cuadros[:2]:
        for c in cajas_xywh(_idx):
            pend_im.append(np.ascontiguousarray(im[:, :, ::-1]))
            pend_cj.append([c])
    with torch.inference_mode():
        e = proc(images=pend_im[:lote], boxes=pend_cj[:lote], return_tensors="pt")
        e = {k: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda")) for k, v in e.items()}
        _ = mod(**e)
    torch.cuda.synchronize()

    ims, cjs = [], []
    for idx, im in cuadros:
        rgb = np.ascontiguousarray(im[:, :, ::-1])
        for c in cajas_xywh(idx):
            ims.append(rgb)
            cjs.append([c])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    hechas = 0
    with torch.inference_mode():
        for i in range(0, len(ims), lote):
            e = proc(images=ims[i:i + lote], boxes=cjs[i:i + lote], return_tensors="pt")
            e = {k: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda"))
                 for k, v in e.items()}
            o = mod(**e)
            _ = proc.post_process_pose_estimation(o, boxes=cjs[i:i + lote], threshold=None)
            hechas += len(cjs[i:i + lote])
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    por_pasada = 1000.0 * dt / hechas
    por_cuadro = 1000.0 * dt / len(cuadros)
    vram = torch.cuda.max_memory_allocated() / 1e9
    print("  lote %-2d ... %6.1f ms/pasada   %7.1f ms/cuadro (2 personas)   VRAM %.2f GB" % (
        lote, por_pasada, por_cuadro, vram))
    resultados["sapiens2-0.4b lote %d" % lote] = por_cuadro

print()
print("=" * 82)
print("COMPARACION, en milisegundos por CUADRO DE VIDEO")
print("=" * 82)
base = resultados.get("yolo11m-pose 960")
for k, v in sorted(resultados.items(), key=lambda kv: kv[1]):
    rel = ("  = %.1fx el actual" % (v / base)) if base else ""
    print("  %-28s %8.1f ms/cuadro%s" % (k, v, rel))

print()
print("PROYECCION AL CORPUS: 12 millones de cuadros")
for k, v in sorted(resultados.items(), key=lambda kv: kv[1]):
    h = 12e6 * v / 1000.0 / 3600.0
    print("  %-28s %8.0f h = %6.1f dias" % (k, h, h / 24))
print()
print("AVISO: son 12 millones de cuadros porque se quieren TODOS los fotogramas. Si al final se")
print("       submuestrea, la cifra baja en proporcion directa y esa decision es del entrenamiento.")

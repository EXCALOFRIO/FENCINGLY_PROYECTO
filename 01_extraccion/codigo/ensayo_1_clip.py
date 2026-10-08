"""Ensayo de Sapiens 2 sobre UN clip: rendimiento real, VRAM, y si salen los puntos de mano y pie.

PASO 1 DE LA CADENCIA 1 -> 2 -> 5 -> 10 -> 100. La puerta para avanzar es que corra sin agotar la
memoria de la 3060 y devuelva los puntos que la idea del cono necesita.

QUE SE MIDE, declarado antes de correr:
  1. pasadas por segundo, que es lo que decide si el corpus cabe: 23,5 millones de pasadas
     (12 millones de cuadros x 2 tiradores, porque el modelo es TOP-DOWN)
  2. VRAM maxima, porque la 3060 tiene 12 GB y el 1b son 6 GB de pesos
  3. cuantos de los 308 puntos vuelven con confianza utilizable, y en particular la CADENA DEL
     PULGAR (indices 113-116 en la derecha, 92-95 en la izquierda), que es el eje del cono
  4. si el subconjunto sin cara ahorra lo previsto (61 %)

SE USA UN CLIP VERIFICADO A OJO: `SentimentalLargeBedlingtonterrier` es uno de los que se rendehizo
ayer y en el que se VE al tirador y su cazoleta. Medir rendimiento sobre un clip donde la caja esta
mal daria un numero valido y unas poses sin sentido, y se confundirian las dos cosas.

LAS CAJAS VIENEN DEL DETECTOR ACTUAL, que sabemos que a veces se equivoca. Aqui no importa: este
ensayo mide el MOTOR DE PUNTOS, no la seleccion. La seleccion es la etapa A y va aparte.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
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
            BASE / "localrun/runs_extra3/data_20260907T010246490954Z/npz",
            BASE / "scratchpad/catalogo/ws_completo/npz"]
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
SALIDA.mkdir(parents=True, exist_ok=True)

CLIP = sys.argv[1] if len(sys.argv) > 1 else "SentimentalLargeBedlingtonterrier"
N_CUADROS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
LOTE = int(sys.argv[3]) if len(sys.argv) > 3 else 1
"""EL CONVENIO DE CAJA ES xywh, NO xyxy. Medido: xyxy da 0,104 de confianza en tronco y brazos y
xywh da 0,883, un factor 8,5. Con xyxy el modelo parece roto y no lo esta."""

PULGAR_DER = [112, 113, 114, 115, 116]
PULGAR_IZQ = [91, 92, 93, 94, 95]
MUNECAS = [9, 10]
TOBILLOS_PIES = [15, 16]
CONF_MIN = 0.3


def localiza_video(cid):
    for raiz, _d, fs in os.walk("D:/FENCINGLY_VIDEOS_RAW"):
        for f in fs:
            if f == "%s.mp4" % cid:
                return os.path.join(raiz, f)
    return None


def localiza_npz(cid):
    for d in NPZ_DIRS:
        p = d / ("%s.npz" % cid)
        if p.is_file():
            return p
    return None


print("=" * 80)
print("ENSAYO DE SAPIENS 2 SOBRE 1 CLIP")
print("=" * 80)
print("  clip ............ %s" % CLIP)
print("  cuadros ......... %d" % N_CUADROS)

mp4 = localiza_video(CLIP)
fz = localiza_npz(CLIP)
if not mp4 or not fz:
    raise SystemExit("falta material: mp4=%s npz=%s" % (bool(mp4), bool(fz)))
print("  video ........... %s" % mp4)

with np.load(fz, allow_pickle=True) as z:
    BOX = np.asarray(z["boxes"], dtype="float64")
print("  cajas del detector actual: %s" % (BOX.shape,))

print()
print("--- cargando el modelo")
t0 = time.perf_counter()
from transformers import Sapiens2ForPoseEstimation, Sapiens2ImageProcessor

proc = Sapiens2ImageProcessor.from_pretrained(str(PESOS))
modelo = Sapiens2ForPoseEstimation.from_pretrained(str(PESOS), dtype=torch.float16).to("cuda").eval()
t_carga = time.perf_counter() - t0
torch.cuda.reset_peak_memory_stats()
vram_pesos = torch.cuda.memory_allocated() / 1e9
print("  cargado en %.1f s   pesos en VRAM: %.2f GB" % (t_carga, vram_pesos))
print("  entrada esperada: %s" % (proc.size,))

cap = cv2.VideoCapture(mp4)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
T_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("  video: %dx%d, %d cuadros" % (W, H, T_total))
medio = max(0, T_total // 2 - N_CUADROS // 2)
cap.set(cv2.CAP_PROP_POS_FRAMES, medio)
cuadros = []
for i in range(medio, min(T_total, medio + N_CUADROS)):
    ok, im = cap.read()
    if not ok:
        break
    cuadros.append((i, im))
cap.release()
print("  decodificados: %d cuadros desde el %d" % (len(cuadros), medio))

print()
print("--- pasada de calentamiento (no cuenta para el tiempo)")
idx0, im0 = cuadros[0]
cajas0 = []
for s in (0, 1):
    b = BOX[idx0, s]
    if np.isfinite(b).all() and b[2] > b[0]:
        x, y = float(b[0] * W), float(b[1] * H)
        cajas0.append([x, y, float(b[2] * W) - x, float(b[3] * H) - y])
if not cajas0:
    raise SystemExit("el detector actual no da ninguna caja en el cuadro %d" % idx0)
with torch.inference_mode():
    ent = proc(images=[np.ascontiguousarray(im0[:, :, ::-1])] * len(cajas0), return_tensors="pt")
    ent = {k: (v.to("cuda").half() if v.dtype == torch.float32 else v.to("cuda")) for k, v in ent.items()}
    _ = modelo(**ent)
torch.cuda.synchronize()
print("  ok, %d cajas" % len(cajas0))

print()
print("--- MEDIDA")
n_pasadas, filas = 0, []
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
for idx, im in cuadros:
    rgb = np.ascontiguousarray(im[:, :, ::-1])
    cajas, lados = [], []
    for s in (0, 1):
        b = BOX[idx, s]
        if np.isfinite(b).all() and b[2] > b[0]:
            x, y = float(b[0] * W), float(b[1] * H)
            cajas.append([x, y, float(b[2] * W) - x, float(b[3] * H) - y])
            lados.append(s)
    if not cajas:
        continue
    with torch.inference_mode():
        ent = proc(images=[rgb] * len(cajas), boxes=[[c] for c in cajas], return_tensors="pt")
        ent = {k: (v.to("cuda").half() if v.dtype == torch.float32 else v.to("cuda"))
               for k, v in ent.items()}
        out = modelo(**ent)
        res = proc.post_process_pose_estimation(out, boxes=[[c] for c in cajas], threshold=None)
    n_pasadas += len(cajas)
    for j, s in enumerate(lados):
        r = res[j][0] if isinstance(res[j], list) else res[j]
        kp = r["keypoints"].cpu().numpy() if hasattr(r["keypoints"], "cpu") else np.asarray(r["keypoints"])
        sc = r["scores"].cpu().numpy() if hasattr(r["scores"], "cpu") else np.asarray(r["scores"])
        filas.append({"cuadro": idx, "lado": s, "n_kp": int(kp.shape[0]),
                      "kp": kp, "sc": sc})
torch.cuda.synchronize()
dt = time.perf_counter() - t0
vram_pico = torch.cuda.max_memory_allocated() / 1e9

print("  pasadas ......... %d" % n_pasadas)
print("  tiempo .......... %.2f s" % dt)
print("  PASADAS/S ....... %.2f" % (n_pasadas / dt if dt else 0))
print("  VRAM pico ....... %.2f GB  de 12 GB" % vram_pico)

if not filas:
    raise SystemExit("no salio ninguna pose")

n_kp = filas[0]["n_kp"]
print()
print("--- PUNTOS DEVUELTOS")
print("  keypoints por persona ... %d" % n_kp)
todas_sc = np.stack([f["sc"] for f in filas])
print("  con confianza >= %.2f, media por persona ... %.1f de %d  (%.0f %%)" % (
    CONF_MIN, float((todas_sc >= CONF_MIN).sum(axis=1).mean()), n_kp,
    100.0 * float((todas_sc >= CONF_MIN).mean())))

print()
print("--- LO QUE LA IDEA DEL CONO NECESITA")
for etq, idxs in (("muneca izq/der", MUNECAS), ("cadena pulgar DER", PULGAR_DER),
                  ("cadena pulgar IZQ", PULGAR_IZQ), ("tobillos", TOBILLOS_PIES)):
    val = [i for i in idxs if i < n_kp]
    if not val:
        print("  %-20s fuera de rango (solo hay %d puntos)" % (etq, n_kp))
        continue
    m = todas_sc[:, val]
    print("  %-20s confianza mediana %.3f   con >= %.2f: %.0f %%" % (
        etq, float(np.median(m)), CONF_MIN, 100.0 * float((m >= CONF_MIN).mean())))

print()
print("--- PROYECCION AL CORPUS con este rendimiento")
r = n_pasadas / dt if dt else 0
if r > 0:
    horas = 23.5e6 / r / 3600
    print("  23,5 millones de pasadas -> %.0f h  = %.1f dias  = %.1f meses" % (
        horas, horas / 24, horas / 24 / 30))
    print("  y con lotes de 8 cuadros x 2 personas seria %.0f h si escala lineal" % (horas / 8))

sal = SALIDA / ("ensayo_%s.json" % CLIP)
sal.write_text(json.dumps({
    "clip": CLIP, "video": mp4, "ancho": W, "alto": H, "cuadros_video": T_total,
    "cuadros_medidos": len(cuadros), "pasadas": n_pasadas, "segundos": round(dt, 3),
    "pasadas_por_s": round(r, 3), "vram_pesos_gb": round(vram_pesos, 3),
    "vram_pico_gb": round(vram_pico, 3), "n_keypoints": n_kp,
    "carga_s": round(t_carga, 2),
}, indent=1), encoding="utf-8")
print()
print("  guardado %s" % sal)

np.savez_compressed(SALIDA / ("poses_%s.npz" % CLIP),
                    kp=np.stack([f["kp"] for f in filas]),
                    sc=np.stack([f["sc"] for f in filas]),
                    cuadro=np.array([f["cuadro"] for f in filas]),
                    lado=np.array([f["lado"] for f in filas]))
print("  guardado poses_%s.npz para pintarlas" % CLIP)

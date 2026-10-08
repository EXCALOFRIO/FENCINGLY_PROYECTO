"""ETAPA E, paso 1: guarda las poses de Sapiens 2 de los dos elegidos, para TODOS los cuadros.

POR QUE UN PASO APARTE: el cono se mide muchas veces (estabilidad del eje, apertura, interseccion,
control negativo) y cada medida cambia. Volver a pasar la GPU en cada intento serian minutos por
prueba y encima con la GPU compartida. Se saca UNA vez y se guarda.

QUE GUARDA: `kp` de forma (T, 2, 308, 2) y `sc` de forma (T, 2, 308), mas las cajas elegidas, la
presencia y la marca de interpolado. Los cuadros sin tirador quedan a NaN.

TRAMPAS RESPETADAS, todas medidas antes:
  - la caja va en **`xywh`**: con `xyxy` la confianza cae de 0,883 a 0,104
  - pose a **384x288**: 7,8x mas rapido y los puntos se mueven 2 px
  - no forzar `sdpa`: 7,5x mas lento
  - aqui se trabaja sobre el VIDEO, no sobre un NPZ de la cosecha vieja, asi que **`f0` no aplica**:
    el cuadro k del video es el cuadro k. Se dice explicitamente porque el desfase `f0` ya costo un
    dia entero en este proyecto.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

CACHE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565\scratchpad\hoja_seg\hf")
os.environ["HF_HOME"] = str(CACHE)

import cv2
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import a3_selector as SEL
import interpola as IP

PRUEBAS = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
HUB = CACHE / "hub"
CLIP = sys.argv[1] if len(sys.argv) > 1 else "SentimentalLargeBedlingtonterrier"
RES = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (384, 288)
"""Se puede pedir otra resolucion para tener una SEGUNDA medida independiente de las mismas poses.
Sirve para la reproducibilidad de la interseccion: si el numero cambia al cambiar la resolucion de
entrada, no es una medida de la escena sino del ajuste."""
MARGEN_PIE = 0.06
LIMITE_HUECO = 5
N_PTS = 308


def snap(n):
    d = HUB / ("models--facebook--%s" % n) / "snapshots"
    return next(d.iterdir()) if d.is_dir() else None


f = PRUEBAS / ("candidatos_%s.npz" % CLIP)
if not f.is_file():
    raise SystemExit("falta %s" % f)
with np.load(f, allow_pickle=True) as z:
    cajas, cuadros = z["cajas"], z["cuadros"]
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    FPS = float(z["fps"])
    video = str(z["video"])

print("=" * 88)
print("E1 POSES PARA EL CONO  ·  %s  ·  %dx%d  ·  %d cuadros a %.2f fps  ·  entrada %dx%d"
      % (CLIP, W, H, T, FPS, RES[0], RES[1]))
print("=" * 88)

tr = SEL.construye_trayectorias(cajas, cuadros, W, H, T)
cand = [{**SEL.puntua(t, W, H, T), "t": t} for t in tr]
cand.sort(key=lambda r: -r["puntos"])
for r in cand:
    c = np.asarray(r["t"]["cajas"], dtype="float64")
    r["pie"] = float(np.median(c[:, 3])) / H
ref = cand[0]["pie"]
pasan = [r for r in cand if r["pie"] >= ref - MARGEN_PIE]
print("  trayectorias %d -> pasan la puerta de primer plano %d" % (len(cand), len(pasan)))

res = SEL.elige_dos([r["t"] for r in pasan], W, H, T)
print("  eleccion ... %s" % res["motivo"])
B, presente = SEL.a_matrices(res["elegidas"], cajas, T, W, H)
marca = np.zeros((T, 2), dtype=bool)
for lado in (0, 1):
    for h in IP.huecos(presente[:, lado]):
        if h["tipo"] != "interior" or h["largo"] > LIMITE_HUECO:
            continue
        a, b = h["a"], h["b"]
        n = b - a - 1
        t = (np.arange(1, n + 1) / float(b - a))[:, None]
        B[h["ini"]:h["fin"], lado] = (1 - t) * B[a, lado][None] + t * B[b, lado][None]
        presente[h["ini"]:h["fin"], lado] = True
        marca[h["ini"]:h["fin"], lado] = True
print("  presencia lado 0 %.1f %%  lado 1 %.1f %%  ·  los dos %.1f %%  ·  %d interpolados" % (
    100 * presente[:, 0].mean(), 100 * presente[:, 1].mean(),
    100 * (presente.sum(axis=1) == 2).mean(), int(marca.sum())))

from transformers import Sapiens2ForPoseEstimation, Sapiens2ImageProcessor

cap = cv2.VideoCapture(video)
frames = []
while True:
    ok, im = cap.read()
    if not ok:
        break
    frames.append(im)
cap.release()
NT = min(len(frames), T)
print("  decodificados %d cuadros" % len(frames))

s = snap("sapiens2-pose-0.4b")
proc = Sapiens2ImageProcessor.from_pretrained(str(s))
proc.size = {"height": RES[0], "width": RES[1]}
mod = Sapiens2ForPoseEstimation.from_pretrained(str(s), dtype=torch.float16).to("cuda").eval()

KP = np.full((T, 2, N_PTS, 2), np.nan, dtype="float32")
SC = np.full((T, 2, N_PTS), np.nan, dtype="float32")
t0 = time.perf_counter()
for k in range(NT):
    im = frames[k]
    cjs, lados = [], []
    for lado in (0, 1):
        b = B[k, lado]
        if not (np.isfinite(b).all() and b[2] > b[0] and b[3] > b[1]):
            continue
        x, y = float(b[0] * W), float(b[1] * H)
        cjs.append([[x, y, float(b[2] * W) - x, float(b[3] * H) - y]])
        lados.append(lado)
    if not cjs:
        continue
    rgb = np.ascontiguousarray(im[:, :, ::-1])
    with torch.inference_mode():
        e = proc(images=[rgb] * len(cjs), boxes=cjs, return_tensors="pt")
        e = {kk: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda"))
             for kk, v in e.items()}
        o = mod(**e)
        r = proc.post_process_pose_estimation(o, boxes=cjs, threshold=None)
    for j, lado in enumerate(lados):
        rr = r[j][0] if isinstance(r[j], list) else r[j]
        kp = np.asarray(rr["keypoints"].cpu(), dtype="float32")
        sc = np.asarray(rr["scores"].cpu(), dtype="float32")
        n = min(len(kp), N_PTS)
        KP[k, lado, :n] = kp[:n]
        SC[k, lado, :n] = sc[:n]
ms = 1000.0 * (time.perf_counter() - t0) / max(NT, 1)
print("  pose ... %.0f ms/cuadro de video (GPU COMPARTIDA: el tiempo es indicativo)" % ms)

suf = "" if RES == (384, 288) else "_%dx%d" % RES
out = PRUEBAS / ("cono_poses_%s%s.npz" % (CLIP, suf))
np.savez_compressed(out, res=np.asarray(RES), kp=KP, sc=SC, B=B, presente=presente, marca=marca,
                    W=W, H=H, T=T, fps=FPS, video=video, ms_pose=ms)
print("  guardado %s" % out)

"""Los indices de mano son 21-41 y 42-62, MEDIDOS en `e0_mapa_indices.py`. Los que daba el plan
(91-95 y 112-116) caen sobre la careta y se imprimen aqui al lado para que la diferencia se vea."""
for nom, idx in (("mano A root+pulgar 21-25", [21, 22, 23, 24, 25]),
                 ("mano B root+pulgar 42-46", [42, 43, 44, 45, 46]),
                 ("PLAN (careta) 112-116", [112, 113, 114, 115, 116]),
                 ("PLAN (careta) 91-95", [91, 92, 93, 94, 95])):
    v = SC[:, :, idx]
    v = v[np.isfinite(v)]
    print("  %-26s confianza mediana %.3f  ·  fraccion con sc>=0,3 %.3f" % (
        nom, float(np.median(v)) if v.size else float("nan"),
        float((v >= 0.3).mean()) if v.size else float("nan")))

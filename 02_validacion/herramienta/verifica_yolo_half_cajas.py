"""(2026-10-08) Puerta 1 del YOLO fp16, ver extrae_pareja.YOLO_HALF.
    python verifica_yolo_half_cajas.py 8
A/B de YOLO: fp32 cuadro a cuadro (actual) contra half=True, y half en lotes de 8.
Criterio declarado ANTES: se adopta si >=99 % de cuadros con el mismo numero de cajas y
>=99,5 % de cajas emparejadas con IoU>=0,9. Velocidad: intercalada clip a clip."""
import sys, time, random
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import rutas as RU, catalogo as CA
import extrae_pareja as EP
import cv2, numpy as np
from ultralytics import YOLO
y = YOLO(RU.pesos_yolo()); y.to("cuda")
ids = [c for c in CA.manifiesto().query("weapon=='foil'").clip_id.astype(str)]
random.seed(1); random.shuffle(ids)
clips = []
for c in ids:
    if RU.video(c): clips.append(c)
    if len(clips) == int(sys.argv[1]): break
def frames(v):
    cap = cv2.VideoCapture(v); f = []
    while True:
        ok, im = cap.read()
        if not ok: break
        f.append(im)
    return f
def corre(fr, half, lote):
    out = []
    for i in range(0, len(fr), lote):
        rs = y.predict(fr[i:i+lote] if lote > 1 else fr[i], imgsz=EP.IMGSZ_YOLO, conf=EP.CONF_YOLO, verbose=False, device=0, half=half)
        for r in rs:
            out.append(r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4)))
    return out
def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    i = np.clip(x2-x1, 0, None)*np.clip(y2-y1, 0, None)
    ar = lambda z: (z[:, 2]-z[:, 0])*(z[:, 3]-z[:, 1])
    return i/(ar(a)[:, None]+ar(b)[None, :]-i+1e-9)
V = {"fp32": (False, 1), "half": (True, 1), "half_l8": (True, 8)}
T = {k: 0.0 for k in V}; R = {}
corre(frames(RU.video(clips[0]))[:30], True, 8)  # calentamiento
for c in clips:
    fr = frames(RU.video(c)); R[c] = {}
    for k in random.sample(list(V), 3):
        t = time.perf_counter(); R[c][k] = corre(fr, *V[k]); T[k] += time.perf_counter()-t
for k in ("half", "half_l8"):
    nf = igual = nb = ok = 0; d = []
    for c in clips:
        for a, b in zip(R[c]["fp32"], R[c][k]):
            nf += 1; igual += len(a) == len(b); nb += len(a)
            if len(a) and len(b):
                m = iou(a, b).max(1); ok += (m >= 0.9).sum()
                j = iou(a, b).argmax(1); d += list(np.abs(a-b[j]).max(1))
    print("%-8s cuadros %d  mismo n %.4f  cajas IoU>=0.9 %.4f  desplaz p50 %.2f p99 %.2f px  tiempo %.1f s vs fp32 %.1f s -> %.2fx"
          % (k, nf, igual/nf, ok/max(nb, 1), np.median(d), np.percentile(d, 99), T[k], T["fp32"], T["fp32"]/T[k]))

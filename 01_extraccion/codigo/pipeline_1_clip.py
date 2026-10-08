"""Tuberia completa sobre UN clip: selector con puerta de primer plano, poses de Sapiens, y VIDEO.

QUE HACE, en orden:
  1. lee los candidatos de YOLO (todos, sin filtrar)
  2. construye TRAYECTORIAS y las puntua enteras
  3. **PUERTA DE PRIMER PLANO**: descarta las trayectorias que estan mas lejos de la camara
  4. elige dos, una a cada lado
  5. rellena huecos interiores de hasta 5 cuadros, MARCANDOLOS
  6. saca las poses de Sapiens 2 sobre las cajas elegidas
  7. escribe un video con la pose y las mascaras al lado

LA PUERTA DE PRIMER PLANO, y por que la y del pie en vez de `pointmap`:

  Medido en un cuadro con arbitro visible: los dos tiradores dan z de `pointmap` **1,0391 los dos**,
  el arbitro **0,8550**, y el fondo 0,777-0,820. La profundidad SEPARA. Pero la **y del pie** da lo
  mismo —tiradores 0,852 y 0,856, arbitro 0,756— **por 0 ms en vez de 480**, porque la perspectiva ya
  ordena la profundidad sin modelo. Se usa la y del pie y se deja `pointmap` como comprobacion.

TRAMPAS RESPETADAS, todas medidas antes:
  - la caja para Sapiens va en **`xywh`**, no `xyxy`: con `xyxy` la confianza cae de 0,883 a 0,104
  - Sapiens pose a **384x288**: 7,8x mas rapido y los puntos se mueven 2 px
  - `matting` exige **UN tirador por recorte**: con dos devuelve 0,0053, el valor del fondo vacio
  - no forzar `sdpa`: 7,5x mas lento
  - `f0`: aqui se trabaja sobre el video, asi que no aplica; se imprime por si acaso
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

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import a3_selector as SEL
import interpola as IP

PRUEBAS = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\video_final")
SALIDA.mkdir(parents=True, exist_ok=True)
HUB = CACHE / "hub"

CLIP = sys.argv[1] if len(sys.argv) > 1 else "SentimentalLargeBedlingtonterrier"
MARGEN_PIE = 0.06
"""Una trayectoria cuyo pie mediano queda mas de esto por encima del pie de la mejor esta MAS LEJOS
de la camara y se descarta. Medido: entre tirador y arbitro la diferencia fue 0,10 y entre tirador y
fondo 0,20, con dispersion de 0,004 entre los dos tiradores. 0,06 cae en medio con holgura."""
LIMITE_HUECO = 5
"""Medido escondiendo cuadros: con 5 el error mediano es 1,386 % del alto de caja y con 8 sube a
2,018 %, que pasa del limite declarado del 2 %."""
CARA_MAX = 90
HUESOS = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
          (11, 13), (13, 15), (12, 14), (14, 16)]
PULGAR = [(112, 113), (113, 114), (114, 115), (115, 116),
          (91, 92), (92, 93), (93, 94), (94, 95)]


def snap(n):
    d = HUB / ("models--facebook--%s" % n) / "snapshots"
    return next(d.iterdir()) if d.is_dir() else None


f = PRUEBAS / ("candidatos_%s.npz" % CLIP)
if not f.is_file():
    raise SystemExit("falta %s\ncorre: .venv-gpu/python a3_paso1_candidatos.py %s" % (f.name, CLIP))
with np.load(f, allow_pickle=True) as z:
    cajas, confs, cuadros = z["cajas"], z["confs"], z["cuadros"]
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    FPS = float(z["fps"])
    video = str(z["video"])

print("=" * 92)
print("TUBERIA SOBRE 1 CLIP  ·  %s" % CLIP)
print("=" * 92)
print("  %d candidatos en %d cuadros de %dx%d a %.2f fps" % (len(cajas), T, W, H, FPS))

tr = SEL.construye_trayectorias(cajas, cuadros, W, H, T)
cand = [{**SEL.puntua(t, W, H, T), "t": t} for t in tr]
cand.sort(key=lambda r: -r["puntos"])
print("  trayectorias ... %d" % len(cand))


def pie_mediano(r):
    c = np.asarray(r["t"]["cajas"], dtype="float64")
    return float(np.median(c[:, 3])) / H


for r in cand:
    r["pie"] = pie_mediano(r)

"""LA PUERTA: se toma como referencia el pie de la trayectoria mas adelantada de entre las que tienen
duracion apreciable, para que un espectador de un solo cuadro pegado a la camara no fije el nivel."""
ref = cand[0]["pie"]
print("  pie de referencia = el de la trayectoria mejor puntuada ... %.3f" % ref)
print("  (anclar en la MAS ADELANTADA daba 0,998, alguien pegado a camara, y echaba a los tiradores)")
print()
print("  %-4s %8s %7s %7s %8s %7s %s" % ("id", "puntos", "pie", "alto", "cx", "n", "puerta"))
pasan = []
for i, r in enumerate(cand):
    ok = r["pie"] >= ref - MARGEN_PIE
    if ok:
        pasan.append(r)
    print("  %-4d %8.3f %7.3f %7.3f %8.3f %7d %s" % (
        i, r["puntos"], r["pie"], r["alto_med"], r["cx_med"], r["n"],
        "pasa" if ok else "FUERA: mas lejos que %.3f" % (ref - MARGEN_PIE)))
print()
print("  pasan la puerta de primer plano ... %d de %d" % (len(pasan), len(cand)))

res = SEL.elige_dos([r["t"] for r in pasan], W, H, T)
print("  eleccion ... %s" % res["motivo"])
B, presente = SEL.a_matrices(res["elegidas"], cajas, T, W, H)
for lado in (0, 1):
    print("    lado %d presente en %.1f %% de los cuadros" % (lado, 100 * presente[:, lado].mean()))
ambos_antes = float((presente.sum(axis=1) == 2).mean())

"""Interpolacion de las CAJAS en los huecos interiores, con el limite medido y marcando lo relleno."""
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
ambos = float((presente.sum(axis=1) == 2).mean())
print("    los dos a la vez: %.1f %% -> %.1f %% tras interpolar (%d cuadros marcados)" % (
    100 * ambos_antes, 100 * ambos, int(marca.sum())))

from transformers import (Sapiens2ForImageMatting, Sapiens2ForPoseEstimation,
                          Sapiens2ForSemanticSegmentation, Sapiens2ImageProcessor)

cap = cv2.VideoCapture(video)
frames = []
while True:
    ok, im = cap.read()
    if not ok:
        break
    frames.append(im)
cap.release()
print("  decodificados %d cuadros" % len(frames))

POSES = [None] * len(frames)
SEG = [None] * len(frames)
MAT = [None] * len(frames)


def caja_xywh(k, lado):
    b = B[k, lado]
    if not (np.isfinite(b).all() and b[2] > b[0]):
        return None
    x, y = float(b[0] * W), float(b[1] * H)
    return [x, y, float(b[2] * W) - x, float(b[3] * H) - y]


print()
print("--- poses de Sapiens 2 a 384x288")
s = snap("sapiens2-pose-0.4b")
proc = Sapiens2ImageProcessor.from_pretrained(str(s))
proc.size = {"height": 384, "width": 288}
mod = Sapiens2ForPoseEstimation.from_pretrained(str(s), dtype=torch.float16).to("cuda").eval()
t0 = time.perf_counter()
for k, im in enumerate(frames):
    cjs, lados = [], []
    for lado in (0, 1):
        c = caja_xywh(k, lado)
        if c:
            cjs.append([c])
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
    POSES[k] = {}
    for j, lado in enumerate(lados):
        rr = r[j][0] if isinstance(r[j], list) else r[j]
        POSES[k][lado] = (np.asarray(rr["keypoints"].cpu()), np.asarray(rr["scores"].cpu()))
ms_pose = 1000.0 * (time.perf_counter() - t0) / max(len(frames), 1)
print("    %.0f ms/cuadro" % ms_pose)
del mod
torch.cuda.empty_cache()

print("--- seg sobre el recorte de los dos")
s = snap("sapiens2-seg-0.4b")
proc = Sapiens2ImageProcessor.from_pretrained(str(s))
proc.size = {"height": 1024, "width": 768}
mod = Sapiens2ForSemanticSegmentation.from_pretrained(str(s), dtype=torch.float16).to("cuda").eval()
t0 = time.perf_counter()
RECORTE = [None] * len(frames)
for k, im in enumerate(frames):
    xs, ys = [], []
    for lado in (0, 1):
        b = B[k, lado]
        if np.isfinite(b).all() and b[2] > b[0]:
            xs += [b[0] * W, b[2] * W]
            ys += [b[1] * H, b[3] * H]
    if not xs:
        continue
    mx, my = 0.20 * (max(xs) - min(xs)), 0.20 * (max(ys) - min(ys))
    x0, y0 = int(max(0, min(xs) - mx)), int(max(0, min(ys) - my))
    x1, y1 = int(min(W, max(xs) + mx)), int(min(H, max(ys) + my))
    if x1 - x0 < 32 or y1 - y0 < 32:
        continue
    RECORTE[k] = (x0, y0, x1, y1)
    sub = np.ascontiguousarray(im[y0:y1, x0:x1, ::-1])
    with torch.inference_mode():
        e = proc(images=sub, return_tensors="pt")
        e = {kk: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda"))
             for kk, v in e.items()}
        o = mod(**e)
        m = proc.post_process_semantic_segmentation(o, target_sizes=[sub.shape[:2]])[0]
    SEG[k] = np.asarray(m.cpu()).astype("uint8")
ms_seg = 1000.0 * (time.perf_counter() - t0) / max(len(frames), 1)
print("    %.0f ms/cuadro" % ms_seg)
del mod
torch.cuda.empty_cache()

print("--- matting, UNA llamada POR TIRADOR (con dos en el recorte devuelve fondo)")
s = snap("sapiens2-matting-1b")
proc = Sapiens2ImageProcessor.from_pretrained(str(s))
proc.size = {"height": 1024, "width": 768}
mod = Sapiens2ForImageMatting.from_pretrained(str(s), dtype=torch.float16).to("cuda").eval()
t0 = time.perf_counter()
ALFA = np.full((len(frames), 2), np.nan)
for k, im in enumerate(frames):
    MAT[k] = {}
    for lado in (0, 1):
        b = B[k, lado]
        if not (np.isfinite(b).all() and b[2] > b[0]):
            continue
        x0, y0 = int(max(0, b[0] * W)), int(max(0, b[1] * H))
        x1, y1 = int(min(W, b[2] * W)), int(min(H, b[3] * H))
        mx, my = int(0.22 * (x1 - x0)), int(0.22 * (y1 - y0))
        cx0, cy0 = max(0, x0 - mx), max(0, y0 - my)
        cx1, cy1 = min(W, x1 + mx), min(H, y1 + my)
        if cx1 - cx0 < 24 or cy1 - cy0 < 24:
            continue
        sub = np.ascontiguousarray(im[cy0:cy1, cx0:cx1, ::-1])
        with torch.inference_mode():
            e = proc(images=sub, return_tensors="pt")
            e = {kk: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda"))
                 for kk, v in e.items()}
            o = mod(**e)
            m = proc.post_process_image_matting(o, target_sizes=[sub.shape[:2]])[0]
        a = np.squeeze(np.nan_to_num(np.asarray(
            (m["alpha"] if isinstance(m, dict) else m).cpu(), dtype="float32")))
        MAT[k][lado] = (a, (cx0, cy0, cx1, cy1))
        ALFA[k, lado] = float(a.mean())
ms_mat = 1000.0 * (time.perf_counter() - t0) / max(len(frames), 1)
print("    %.0f ms/cuadro (dos llamadas)" % ms_mat)
del mod
torch.cuda.empty_cache()

va = ALFA[np.isfinite(ALFA)]
print()
print("  MATTING como puerta: alfa media %.4f   p10 %.4f   min %.4f" % (
    float(va.mean()), float(np.percentile(va, 10)), float(va.min())) if va.size else "  sin datos")
print("  (medido: un tirador da 0,21 y el fondo vacio 0,0053)")
bajos = int((va < 0.05).sum()) if va.size else 0
print("  elegidos con alfa < 0,05 (sospechosos de no ser persona) ... %d de %d" % (bajos, va.size))

PAL = np.random.default_rng(7).integers(40, 255, size=(30, 3), dtype=np.uint8)
PAL[0] = (12, 12, 18)
CW, CH = 460, 330


def celda(img, tit, pie=""):
    ih, iw = img.shape[:2]
    esc = min(CW / max(iw, 1), CH / max(ih, 1))
    nw, nh = max(1, int(iw * esc)), max(1, int(ih * esc))
    v = np.zeros((CH, CW, 3), np.uint8)
    ox, oy = (CW - nw) // 2, (CH - nh) // 2
    v[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh))
    cv2.rectangle(v, (0, 0), (CW, 24), (12, 12, 18), -1)
    cv2.putText(v, tit, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1)
    if pie:
        cv2.rectangle(v, (0, CH - 22), (CW, CH), (12, 12, 18), -1)
        cv2.putText(v, pie, (6, CH - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 235, 255), 1)
    return v


escritor, forma = None, None
out = SALIDA / ("final_%s.mp4" % CLIP)
for k, im in enumerate(frames):
    vis = im.copy()
    npt = 0
    for lado in (0, 1):
        b = B[k, lado]
        if not (np.isfinite(b).all() and b[2] > b[0]):
            continue
        col = (255, 150, 0) if lado == 0 else (0, 200, 255)
        if marca[k, lado]:
            col = (200, 0, 255)
        cv2.rectangle(vis, (int(b[0] * W), int(b[1] * H)), (int(b[2] * W), int(b[3] * H)), col, 2)
        if marca[k, lado]:
            cv2.putText(vis, "INTERPOLADO", (int(b[0] * W), max(14, int(b[1] * H) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        ps = POSES[k].get(lado) if POSES[k] else None
        if ps:
            kp, sc = ps
            okp = (sc >= 0.3) & np.isfinite(kp[:, 0]) & np.isfinite(kp[:, 1])
            npt += int(okp[:CARA_MAX].sum())
            for a, bb in HUESOS:
                if a < len(okp) and bb < len(okp) and okp[a] and okp[bb]:
                    cv2.line(vis, (int(kp[a, 0]), int(kp[a, 1])),
                             (int(kp[bb, 0]), int(kp[bb, 1])), col, 2)
            for a, bb in PULGAR:
                if a < len(okp) and bb < len(okp) and okp[a] and okp[bb]:
                    cv2.line(vis, (int(kp[a, 0]), int(kp[a, 1])),
                             (int(kp[bb, 0]), int(kp[bb, 1])), (0, 255, 0), 2)
            for j in range(min(CARA_MAX, len(okp))):
                if okp[j]:
                    cv2.circle(vis, (int(kp[j, 0]), int(kp[j, 1])), 2, (255, 255, 255), -1)
    """Se pintan tambien los DESCARTADOS, para poder ver que la puerta no se lleve a un tirador."""
    for r in cand:
        if r in pasan:
            continue
        for kk, c in zip(r["t"]["cuadros"], r["t"]["cajas"]):
            if kk == k:
                cv2.rectangle(vis, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (110, 110, 110), 1)

    p1 = celda(vis, "1 pose 308 pts + puerta (gris = descartado)",
               "%d puntos   %s" % (npt, "INTERPOLADO" if marca[k].any() else ""))
    sg = SEG[k]
    if sg is not None:
        p2 = celda(PAL[np.clip(sg, 0, 29)][:, :, ::-1], "2 seg partes del cuerpo",
                   "%.0f ms/cuadro" % ms_seg)
    else:
        p2 = celda(np.full((240, 320, 3), 30, np.uint8), "2 seg", "sin recorte")
    caras = []
    for lado in (0, 1):
        if MAT[k] and lado in MAT[k]:
            a, _bb = MAT[k][lado]
            caras.append(cv2.cvtColor((np.clip(a, 0, 1) * 255).astype("uint8"), cv2.COLOR_GRAY2BGR))
    if caras:
        alt = max(c.shape[0] for c in caras)
        caras = [cv2.resize(c, (int(c.shape[1] * alt / c.shape[0]), alt)) for c in caras]
        p3 = celda(np.hstack(caras), "3 matting, uno por tirador",
                   "alfa %s" % " / ".join("%.2f" % ALFA[k, l] for l in (0, 1)
                                          if np.isfinite(ALFA[k, l])))
    else:
        p3 = celda(np.full((240, 320, 3), 30, np.uint8), "3 matting", "sin caja")
    fila = np.hstack([p1, p2, p3])
    cv2.rectangle(fila, (0, fila.shape[0] - 26), (fila.shape[1], fila.shape[0]), (12, 12, 18), -1)
    cv2.putText(fila, "%s   cuadro %d de %d   los dos presentes en %.1f %% del clip" % (
        CLIP, k, len(frames), 100 * ambos), (8, fila.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1)
    if escritor is None:
        forma = fila.shape
        escritor = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                   max(FPS / 2, 8), (fila.shape[1], fila.shape[0]))
    if fila.shape == forma:
        escritor.write(fila)
if escritor is not None:
    escritor.release()

print()
print("=" * 92)
print("VIDEO: %s" % out)
print("  %d cuadros a %.1f fps" % (len(frames), max(FPS / 2, 8)))
print("  coste: pose %.0f + seg %.0f + matting %.0f = %.0f ms/cuadro" % (
    ms_pose, ms_seg, ms_mat, ms_pose + ms_seg + ms_mat))
print("=" * 92)

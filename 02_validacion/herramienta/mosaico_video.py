"""Mosaico en VIDEO de las cinco salidas de Sapiens 2, para juzgarlas a ojo en movimiento.

POR QUE EN VIDEO Y NO EN IMAGENES: una imagen fija no ensena si una capa **parpadea**. Una mascara
que salta de un cuadro al siguiente es inutil para medir velocidad o para seguir la hoja, y eso solo
se ve reproduciendo.

QUE VA EN CADA PANEL:
  1. original con el ESQUELETO de pose (308 puntos, sin la cara)
  2. `seg`      partes del cuerpo en color
  3. `normal`   normales de superficie
  4. `pointmap` la z, que segun lo medido NO distingue tirador de fondo: va para que se vea el fallo
  5. `matting`  el canal alfa, que es el mejor discriminador medido del proyecto (44,6x sin solape)

TRAMPAS YA MEDIDAS QUE ESTE FICHERO RESPETA:

  - **`f0`**: el npz es una VENTANA del video y la fila k le corresponde el cuadro `f0 + k`. Medido:
    **292 de 300 npz tienen f0 != 0**, con desfases de 80 a 289 cuadros. Aqui se trabaja sobre el
    video y se traduce el indice, nunca al contrario.
  - **caja `xywh`, no `xyxy`** para la pose: con `xyxy` la confianza cae de 0,883 a 0,104.
  - **`do_pad=True`** en `pointmap` y `normal`: hay que pasar `source_sizes` al post-proceso o entra
    un 58 % de relleno que se estira dentro de la medida.
  - **`pointmap` esta bloqueado a 1024x768**: su `scale_head` es de tamano fijo y cualquier otra
    resolucion lanza `RuntimeError`. La palanca de 7,8x de la pose no le aplica.
  - **no forzar `sdpa`**: 7,5x mas lento.
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
HUB = CACHE / "hub"
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\mosaico")
SALIDA.mkdir(parents=True, exist_ok=True)
NPZ_DIRS = [BASE / "localrun/runs_extra1/data_20260907T005507157743Z/npz",
            BASE / "localrun/runs_extra2/data_20260907T005507370214Z/npz",
            BASE / "localrun/runs_extra3/data_20260907T010246490954Z/npz",
            BASE / "scratchpad/catalogo/ws_completo/npz"]

CLIP = sys.argv[1] if len(sys.argv) > 1 else "SentimentalLargeBedlingtonterrier"
N_CUADROS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
CELDA_W = 420
RES_POSE = (384, 288)
RES_DENSA = (1024, 768)
CARA_MAX = 90
"""Los indices por debajo de 90 son cuerpo, mano y pie; de ahi arriba domina la cara, que no sirve
porque los tiradores llevan mascara. Se pintan solo los utiles."""


def snap(nombre):
    d = HUB / ("models--facebook--%s" % nombre) / "snapshots"
    return next(d.iterdir()) if d.is_dir() else None


def localiza(cid):
    for raiz, _d, fs in os.walk("D:/FENCINGLY_VIDEOS_RAW"):
        if "%s.mp4" % cid in fs:
            return os.path.join(raiz, "%s.mp4" % cid)
    return None


def npz_de(cid):
    for d in NPZ_DIRS:
        p = d / ("%s.npz" % cid)
        if p.is_file():
            return p
    return None


mp4 = localiza(CLIP)
fz = npz_de(CLIP)
if not mp4:
    raise SystemExit("sin video: %s" % CLIP)

f0, BOX = 0, None
if fz:
    with np.load(fz, allow_pickle=True) as z:
        BOX = np.asarray(z["boxes"], dtype="float64")
        meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    f0 = int(meta.get("f0", 0))
    print("npz: %d cuadros, f0=%d%s" % (BOX.shape[0], f0,
                                        "   <- DESFASE" if f0 else "   (sin desfase)"))

cap = cv2.VideoCapture(mp4)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
FPS = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
tocado = int(meta.get("touch_frame_raw", 0)) + f0 if fz else T // 2
ini = max(0, min(T - N_CUADROS, tocado - N_CUADROS // 2))
print("clip %s: %dx%d, %d cuadros, %.2f fps" % (CLIP, W, H, T, FPS))
print("tocado en el cuadro %d del video; se mosaica de %d a %d" % (tocado, ini, ini + N_CUADROS))
cap.set(cv2.CAP_PROP_POS_FRAMES, ini)
cuadros = []
for i in range(ini, min(T, ini + N_CUADROS)):
    ok, im = cap.read()
    if not ok:
        break
    cuadros.append((i, im))
cap.release()
print("decodificados %d" % len(cuadros))

from transformers import (Sapiens2ForImageMatting, Sapiens2ForNormalEstimation,
                          Sapiens2ForPointmapEstimation, Sapiens2ForPoseEstimation,
                          Sapiens2ForSemanticSegmentation, Sapiens2ImageProcessor)

CAPAS = [("pose", "sapiens2-pose-0.4b", Sapiens2ForPoseEstimation, RES_POSE),
         ("seg", "sapiens2-seg-0.4b", Sapiens2ForSemanticSegmentation, RES_DENSA),
         ("normal", "sapiens2-normal-0.4b", Sapiens2ForNormalEstimation, RES_DENSA),
         ("pointmap", "sapiens2-pointmap-0.4b", Sapiens2ForPointmapEstimation, RES_DENSA),
         ("matting", "sapiens2-matting-1b", Sapiens2ForImageMatting, RES_DENSA)]

disponibles = []
for nom, repo, clase, res in CAPAS:
    s = snap(repo)
    if s is None:
        print("  %-9s pesos NO descargados, se salta" % nom)
        continue
    disponibles.append((nom, s, clase, res))
    print("  %-9s ok" % nom)
if not disponibles:
    raise SystemExit("no hay pesos de ninguna capa")

def recorte_de(im, idx):
    """QUE HACE: recorte que cubre a los dos tiradores con margen, para las capas DENSAS.

    Sobre el cuadro completo las capas densas se pierden: medido en el primer mosaico, `matting` dejo
    a los dos tiradores fuera y cogio a una persona cercana a camara. Se midieron sobre recortes
    centrados en persona, asi que aqui se les da eso. Si no hay cajas, se devuelve el cuadro entero y
    se dice, en vez de recortar a ciegas.
    """
    kn = idx - f0
    if BOX is None or not (0 <= kn < BOX.shape[0]):
        return im, (0, 0)
    xs, ys = [], []
    for lado in (0, 1):
        b = BOX[kn, lado]
        if np.isfinite(b).all() and b[2] > b[0]:
            xs += [b[0] * W, b[2] * W]
            ys += [b[1] * H, b[3] * H]
    if not xs:
        return im, (0, 0)
    mx = 0.25 * (max(xs) - min(xs))
    my = 0.25 * (max(ys) - min(ys))
    x0 = int(max(0, min(xs) - mx)); x1 = int(min(W, max(xs) + mx))
    y0 = int(max(0, min(ys) - my)); y1 = int(min(H, max(ys) + my))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return im, (0, 0)
    return np.ascontiguousarray(im[y0:y1, x0:x1]), (x0, y0)


PAL = np.random.default_rng(7).integers(40, 255, size=(30, 3), dtype=np.uint8)
PAL[0] = (12, 12, 18)


CELDA_H = 300


def celda(img, titulo, pie=""):
    """Ajusta a CELDA_W x CELDA_H sin deformar: escala por el lado que toque y centra en negro."""
    ih, iw = img.shape[:2]
    esc = min(CELDA_W / max(iw, 1), CELDA_H / max(ih, 1))
    nw, nh = max(1, int(iw * esc)), max(1, int(ih * esc))
    v = np.zeros((CELDA_H, CELDA_W, 3), np.uint8)
    ox, oy = (CELDA_W - nw) // 2, (CELDA_H - nh) // 2
    v[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh))
    h = CELDA_H
    cv2.rectangle(v, (0, 0), (CELDA_W, 26), (12, 12, 18), -1)
    cv2.putText(v, titulo, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1)
    if pie:
        cv2.rectangle(v, (0, h - 24), (CELDA_W, h), (12, 12, 18), -1)
        cv2.putText(v, pie, (7, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (120, 235, 255), 1)
    return v


resultados = {n: [] for n, _s, _c, _r in disponibles}
tiempos = {n: 0.0 for n, _s, _c, _r in disponibles}

for nom, s, clase, res in disponibles:
    print()
    print("--- %s a %dx%d" % (nom, res[0], res[1]))
    proc = Sapiens2ImageProcessor.from_pretrained(str(s))
    proc.size = {"height": res[0], "width": res[1]}
    mod = clase.from_pretrained(str(s), dtype=torch.float16).to("cuda").eval()
    t0 = time.perf_counter()
    for idx, im in cuadros:
        sub, off = recorte_de(im, idx)
        rgb = np.ascontiguousarray((im if nom == "pose" else sub)[:, :, ::-1])
        try:
            with torch.inference_mode():
                if nom == "pose":
                    cajas = []
                    kn = idx - f0
                    if BOX is not None and 0 <= kn < BOX.shape[0]:
                        for lado in (0, 1):
                            b = BOX[kn, lado]
                            if np.isfinite(b).all() and b[2] > b[0]:
                                x, y = float(b[0] * W), float(b[1] * H)
                                cajas.append([x, y, float(b[2] * W) - x, float(b[3] * H) - y])
                    if not cajas:
                        resultados[nom].append(None)
                        continue
                    e = proc(images=[rgb] * len(cajas), boxes=[[c] for c in cajas],
                             return_tensors="pt")
                    e = {k: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda"))
                         for k, v in e.items()}
                    o = mod(**e)
                    r = proc.post_process_pose_estimation(o, boxes=[[c] for c in cajas],
                                                          threshold=None)
                    resultados[nom].append([(np.asarray(x[0]["keypoints"].cpu()),
                                             np.asarray(x[0]["scores"].cpu()))
                                            if isinstance(x, list) else
                                            (np.asarray(x["keypoints"].cpu()),
                                             np.asarray(x["scores"].cpu())) for x in r])
                    continue
                e = proc(images=rgb, return_tensors="pt")
                e = {k: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda"))
                     for k, v in e.items()}
                o = mod(**e)
                if nom == "seg":
                    m = proc.post_process_semantic_segmentation(o, target_sizes=[sub.shape[:2]])[0]
                    resultados[nom].append(np.asarray(m.cpu()).astype("uint8"))
                elif nom == "normal":
                    m = proc.post_process_normal_estimation(o, source_sizes=[sub.shape[:2]],
                                                            target_sizes=[sub.shape[:2]])[0]
                    a = m["normals"] if isinstance(m, dict) else m
                    resultados[nom].append(np.asarray(a.cpu(), dtype="float32"))
                elif nom == "pointmap":
                    m = proc.post_process_pointmap_estimation(o, source_sizes=[sub.shape[:2]],
                                                              target_sizes=[sub.shape[:2]])[0]
                    a = m["pointmap"] if isinstance(m, dict) else m
                    resultados[nom].append(np.asarray(a.cpu(), dtype="float32"))
                elif nom == "matting":
                    m = proc.post_process_image_matting(o, target_sizes=[sub.shape[:2]])[0]
                    a = m["alpha"] if isinstance(m, dict) else m
                    resultados[nom].append(np.asarray(a.cpu(), dtype="float32"))
        except Exception as exc:
            if not resultados[nom]:
                print("    FALLA en el primer cuadro: %s %s" % (type(exc).__name__, str(exc)[:90]))
            resultados[nom].append(None)
    tiempos[nom] = 1000.0 * (time.perf_counter() - t0) / max(len(cuadros), 1)
    hechos = sum(1 for x in resultados[nom] if x is not None)
    print("    %d de %d cuadros   %.0f ms/cuadro" % (hechos, len(cuadros), tiempos[nom]))
    del mod
    torch.cuda.empty_cache()


def a_color(nom, dato):
    if dato is None:
        return np.full((240, 320, 3), 30, np.uint8)
    if nom == "seg":
        return PAL[np.clip(dato, 0, 29)][:, :, ::-1]
    if nom == "normal":
        a = dato
        if a.ndim == 3 and a.shape[0] == 3:
            a = np.transpose(a, (1, 2, 0))
        a = (a - a.min()) / max(a.max() - a.min(), 1e-6)
        return (a[:, :, :3] * 255).astype("uint8")
    if nom == "pointmap":
        a = dato
        if a.ndim == 3 and a.shape[0] in (3, 4):
            a = np.transpose(a, (1, 2, 0))
        z = a[:, :, 2] if a.ndim == 3 and a.shape[2] >= 3 else np.squeeze(a)
        z = np.nan_to_num(z)
        lo, hi = np.percentile(z, 2), np.percentile(z, 98)
        zn = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
        return cv2.applyColorMap((zn * 255).astype("uint8"), cv2.COLORMAP_JET)
    if nom == "matting":
        a = np.squeeze(np.nan_to_num(dato))
        return cv2.cvtColor((np.clip(a, 0, 1) * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
    return np.full((240, 320, 3), 30, np.uint8)


HUESOS = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
          (11, 13), (13, 15), (12, 14), (14, 16)]
PULGAR = [(112, 113), (113, 114), (114, 115), (115, 116),
          (91, 92), (92, 93), (93, 94), (94, 95)]

primero = None
escritor = None
for n, (idx, im) in enumerate(cuadros):
    paneles = []
    vis = im.copy()
    ps = resultados.get("pose", [None] * len(cuadros))[n] if "pose" in resultados else None
    npt = 0
    if ps:
        for lado, (kp, sc) in enumerate(ps):
            col = (255, 150, 0) if lado == 0 else (0, 200, 255)
            ok = (sc >= 0.3) & np.isfinite(kp[:, 0]) & np.isfinite(kp[:, 1])
            npt += int(ok[:CARA_MAX].sum())
            for a, b in HUESOS:
                if a < len(ok) and b < len(ok) and ok[a] and ok[b]:
                    cv2.line(vis, (int(kp[a, 0]), int(kp[a, 1])),
                             (int(kp[b, 0]), int(kp[b, 1])), col, 2)
            for a, b in PULGAR:
                if a < len(ok) and b < len(ok) and ok[a] and ok[b]:
                    cv2.line(vis, (int(kp[a, 0]), int(kp[a, 1])),
                             (int(kp[b, 0]), int(kp[b, 1])), (0, 255, 0), 2)
            for j in range(min(CARA_MAX, len(ok))):
                if ok[j]:
                    cv2.circle(vis, (int(kp[j, 0]), int(kp[j, 1])), 2, (255, 255, 255), -1)
    paneles.append(celda(vis, "1 pose 308 pts (verde = pulgar)", "%d puntos utiles" % npt))
    for nom in ("seg", "normal", "pointmap", "matting"):
        if nom not in resultados:
            continue
        dato = resultados[nom][n] if n < len(resultados[nom]) else None
        pie = "%.0f ms/cuadro" % tiempos[nom]
        if nom == "pointmap":
            pie += "   MEDIDO: no distingue tirador de fondo"
        if nom == "matting":
            pie += "   MEDIDO: 44,6x sin solape"
        paneles.append(celda(a_color(nom, dato), "%d %s" % (paneles.__len__() + 1, nom), pie))
    while len(paneles) % 3:
        paneles.append(np.zeros_like(paneles[0]))
    filas = [np.hstack(paneles[i:i + 3]) for i in range(0, len(paneles), 3)]
    alto = max(f.shape[0] for f in filas)
    filas = [f if f.shape[0] == alto else
             np.vstack([f, np.zeros((alto - f.shape[0], f.shape[1], 3), np.uint8)]) for f in filas]
    marco = np.vstack(filas)
    cv2.rectangle(marco, (0, marco.shape[0] - 28), (marco.shape[1], marco.shape[0]), (12, 12, 18), -1)
    cv2.putText(marco, "%s   cuadro %d de %d   f0=%d   %s" % (
        CLIP, idx, T, f0, "TOCADO" if abs(idx - tocado) <= 1 else ""),
        (8, marco.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1)
    if escritor is None:
        primero = marco.shape
        escritor = cv2.VideoWriter(str(SALIDA / ("mosaico_%s.mp4" % CLIP)),
                                   cv2.VideoWriter_fourcc(*"mp4v"), max(FPS / 2, 8),
                                   (marco.shape[1], marco.shape[0]))
        cv2.imwrite(str(SALIDA / ("mosaico_%s_primer_cuadro.png" % CLIP)), marco)
    if marco.shape == primero:
        escritor.write(marco)
if escritor is not None:
    escritor.release()
print()
print("=" * 78)
print("MOSAICO ESCRITO")
print("  video ... %s" % (SALIDA / ("mosaico_%s.mp4" % CLIP)))
print("  imagen .. %s" % (SALIDA / ("mosaico_%s_primer_cuadro.png" % CLIP)))
print("  a %.1f fps (la mitad del original, para poder mirarlo)" % max(FPS / 2, 8))
print()
print("  COSTE POR CAPA, ms por cuadro de video")
for n, v in sorted(tiempos.items(), key=lambda kv: kv[1]):
    print("    %-10s %7.0f ms" % (n, v))
print("    %-10s %7.0f ms  <- las cinco juntas" % ("TOTAL", sum(tiempos.values())))

"""Por que Sapiens 2 devuelve el 3 % de puntos con confianza: se prueban cuatro invocaciones.

UN 3 % NO ES UN MODELO MALO, ES UNA INVOCACION MALA. Sapiens 2 esta entrenado en mil millones de
imagenes humanas; sobre un tirador nitido tiene que dar confianza alta en hombros, codos y munecas.
Asi que antes de concluir nada sobre el modelo hay que descartar mi propio error.

CUATRO SOSPECHOSOS, cada uno con su variante:
  A  fp16 forzado en la entrada: puede desestabilizar numericamente un ViT
  B  convenio de caja xyxy   (lo que use)
  C  convenio de caja xywh   (lo que usan muchos pipelines top-down)
  D  recorte manual sin pasar cajas: elimina el convenio del problema

CRITERIO DECLARADO: se mira la confianza de los puntos de TRONCO Y BRAZOS (hombros, codos, munecas,
indices 5 a 10), que en un tirador visible tienen que salir altos. La variante que los suba es la
correcta, y si ninguna los sube el problema esta en otro sitio y hay que decirlo.
"""
from __future__ import annotations

import os
import pathlib

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
CLIP = "SentimentalLargeBedlingtonterrier"
TRONCO_BRAZOS = list(range(5, 11))

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
k = T // 2
cap.set(cv2.CAP_PROP_POS_FRAMES, k)
ok, im = cap.read()
cap.release()
rgb = np.ascontiguousarray(im[:, :, ::-1])
print("cuadro %d de %s  (%dx%d)" % (k, CLIP, W, H))

"""Se elige el lado cuya caja sea mayor: mas pixeles, mas facil, mejor caso para el diagnostico."""
mejor_s, mejor_area = None, -1
for s in (0, 1):
    b = BOX[k, s]
    if np.isfinite(b).all() and b[2] > b[0]:
        a = (b[2] - b[0]) * (b[3] - b[1])
        if a > mejor_area:
            mejor_area, mejor_s = a, s
b = BOX[k, mejor_s]
x1, y1, x2, y2 = float(b[0] * W), float(b[1] * H), float(b[2] * W), float(b[3] * H)
print("caja del lado %d: xyxy = (%.0f, %.0f, %.0f, %.0f)  -> %.0fx%.0f px" % (
    mejor_s, x1, y1, x2, y2, x2 - x1, y2 - y1))

from transformers import Sapiens2ForPoseEstimation, Sapiens2ImageProcessor

proc = Sapiens2ImageProcessor.from_pretrained(str(PESOS))


def confianzas(out, boxes_pp):
    res = proc.post_process_pose_estimation(out, boxes=boxes_pp, threshold=None)
    r = res[0][0] if isinstance(res[0], list) else res[0]
    sc = r["scores"]
    sc = sc.cpu().numpy() if hasattr(sc, "cpu") else np.asarray(sc)
    return sc


def prueba(nombre, dtype, modo):
    modelo = Sapiens2ForPoseEstimation.from_pretrained(str(PESOS), dtype=dtype).to("cuda").eval()
    try:
        if modo == "xyxy":
            ent = proc(images=rgb, boxes=[[[x1, y1, x2, y2]]], return_tensors="pt")
            bpp = [[[x1, y1, x2, y2]]]
        elif modo == "xywh":
            ent = proc(images=rgb, boxes=[[[x1, y1, x2 - x1, y2 - y1]]], return_tensors="pt")
            bpp = [[[x1, y1, x2 - x1, y2 - y1]]]
        elif modo == "recorte":
            """Recorte manual con margen del 20 %, sin pasar cajas: quita el convenio del problema."""
            mx, my = 0.20 * (x2 - x1), 0.20 * (y2 - y1)
            cx1, cy1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
            cx2, cy2 = int(min(W, x2 + mx)), int(min(H, y2 + my))
            sub = np.ascontiguousarray(rgb[cy1:cy2, cx1:cx2])
            ent = proc(images=sub, return_tensors="pt")
            bpp = [[[0.0, 0.0, float(cx2 - cx1), float(cy2 - cy1)]]]
        else:
            ent = proc(images=rgb, return_tensors="pt")
            bpp = [[[0.0, 0.0, float(W), float(H)]]]
        ent = {kk: (vv.to("cuda").to(dtype) if vv.is_floating_point() else vv.to("cuda"))
               for kk, vv in ent.items()}
        with torch.inference_mode():
            out = modelo(**ent)
        sc = confianzas(out, bpp)
        tb = sc[TRONCO_BRAZOS]
        print("  %-28s tronco/brazos: mediana %.3f  max %.3f   |  TODOS: >=0,3 en %.0f %%  max %.3f" % (
            nombre, float(np.median(tb)), float(tb.max()),
            100.0 * float((sc >= 0.3).mean()), float(sc.max())))
        return float(np.median(tb))
    except Exception as exc:
        print("  %-28s FALLO %s: %s" % (nombre, type(exc).__name__, str(exc)[:80]))
        return -1.0
    finally:
        del modelo
        torch.cuda.empty_cache()


print()
print("CONFIANZA SEGUN LA INVOCACION")
r = {}
r["fp16 + xyxy (lo que use)"] = prueba("fp16 + xyxy (lo que use)", torch.float16, "xyxy")
r["fp32 + xyxy"] = prueba("fp32 + xyxy", torch.float32, "xyxy")
r["fp32 + xywh"] = prueba("fp32 + xywh", torch.float32, "xywh")
r["fp32 + recorte manual"] = prueba("fp32 + recorte manual", torch.float32, "recorte")
r["fp32 + cuadro completo"] = prueba("fp32 + cuadro completo", torch.float32, "completo")

print()
print("VEREDICTO")
mej = max(r, key=lambda k: r[k])
print("  la mejor es: %s  (mediana tronco/brazos %.3f)" % (mej, r[mej]))
if r[mej] < 0.3:
    print("  NINGUNA llega a 0,3. El problema NO es el convenio ni la precision:")
    print("  hay que mirar la caja (puede estar sobre nadie) o el post-proceso.")
else:
    print("  Con esa invocacion el modelo SI funciona, y mi ensayo anterior media una invocacion mala.")

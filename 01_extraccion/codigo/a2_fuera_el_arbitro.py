"""ETAPA A2: echar al arbitro. El discriminador es FISICO, no un umbral inventado.

LA IDEA: el tirador lleva **mascara y guante**, asi que de el no se ve cara ni pelo. El arbitro va con
la cara descubierta. `seg` de Sapiens 2 tiene clases separadas para `Face_Neck` (3) y `Hair` (4), asi
que la pregunta "es tirador o es arbitro" se puede MEDIR en vez de inferirse de la geometria.

Es distinto de todo lo que probe antes:
  - la geometria (altura, banda, quietud) no distingue a un arbitro de pie de un tirador en fondo
  - la blancura HSV **solapa**: medido, el suelo gris da 0,6965 y hay tiradores a 0,000 con luz pobre
  - `matting` dice "hay UNA persona centrada", no dice **quien**

CRITERIO DECLARADO ANTES DE MIRAR:
  se mide, sobre el recorte de cada candidato, la fraccion de pixeles que `seg` asigna a
  `Face_Neck` + `Hair` (clases 3 y 4), y la de ropa (`Apparel` 1, `UpClothing` 23, `LowClothing` 13).
  La puerta SIRVE si la cara+pelo de los arbitros y la de los tiradores no se solapan.
  Si se solapan, se dice que no separa; no se busca un umbral que salve el resultado.

SE USA UN CLIP DONDE SE VE QUIEN ES QUIEN: en `SentimentalLargeBedlingtonterrier` cuadro 197 hay dos
tiradores enmascarados y, a la derecha, personas de calle. El `f0` de ese clip es 0, asi que no hay
desfase que pueda contaminar la medida.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

CACHE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565\scratchpad\hoja_seg\hf")
os.environ["HF_HOME"] = str(CACHE)

import cv2
import numpy as np
import torch

BASE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565")
PRUEBAS = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\a2_arbitro")
SALIDA.mkdir(parents=True, exist_ok=True)

CARA_PELO = [3, 4]
ROPA = [1, 23, 13]
MANOS = [6, 15]
RES = (1024, 768)
CLIP = sys.argv[1] if len(sys.argv) > 1 else "SentimentalLargeBedlingtonterrier"
CUADRO = int(sys.argv[2]) if len(sys.argv) > 2 else 197
NOMBRES = {0: "Fondo", 1: "Apparel", 3: "Face_Neck", 4: "Hair", 6: "L_Hand", 7: "L_LowArm",
           13: "LowCloth", 15: "R_Hand", 16: "R_LowArm", 22: "Torso", 23: "UpCloth"}

f = PRUEBAS / ("candidatos_%s.npz" % CLIP)
if not f.is_file():
    raise SystemExit("falta %s. Corre primero:\n"
                     "  .venv-gpu/python a3_paso1_candidatos.py %s" % (f.name, CLIP))
with np.load(f, allow_pickle=True) as z:
    cajas, confs, cuadros = z["cajas"], z["confs"], z["cuadros"]
    W, H = int(z["W"]), int(z["H"])
    video = str(z["video"])

sel = cuadros == CUADRO
print("=" * 96)
print("A2: FUERA EL ARBITRO  ·  %s cuadro %d" % (CLIP, CUADRO))
print("=" * 96)
print("  candidatos de YOLO en ese cuadro: %d" % int(sel.sum()))
if not sel.any():
    raise SystemExit("no hay candidatos en ese cuadro")

cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_FRAMES, CUADRO)
ok, im = cap.read()
cap.release()
if not ok:
    raise SystemExit("no decodifica")

snap = next((CACHE / "hub/models--facebook--sapiens2-seg-0.4b/snapshots").iterdir())
from transformers import Sapiens2ForSemanticSegmentation, Sapiens2ImageProcessor

proc = Sapiens2ImageProcessor.from_pretrained(str(snap))
proc.size = {"height": RES[0], "width": RES[1]}
mod = Sapiens2ForSemanticSegmentation.from_pretrained(
    str(snap), dtype=torch.float16).to("cuda").eval()
print("  seg a %dx%d en %s" % (RES[0], RES[1], next(mod.parameters()).device))
print()

filas = []
lienzo = im.copy()
for i, (b, c) in enumerate(zip(cajas[sel], confs[sel])):
    x1, y1, x2, y2 = [int(v) for v in b]
    mx, my = int(0.18 * (x2 - x1)), int(0.18 * (y2 - y1))
    cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
    cx2, cy2 = min(W, x2 + mx), min(H, y2 + my)
    if cx2 - cx1 < 24 or cy2 - cy1 < 24:
        continue
    sub = np.ascontiguousarray(im[cy1:cy2, cx1:cx2, ::-1])
    with torch.inference_mode():
        e = proc(images=sub, return_tensors="pt")
        e = {k: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda"))
             for k, v in e.items()}
        o = mod(**e)
        m = proc.post_process_semantic_segmentation(o, target_sizes=[sub.shape[:2]])[0]
    m = np.asarray(m.cpu())
    persona = m > 0
    npers = max(int(persona.sum()), 1)
    """Se normaliza por los pixeles DE PERSONA, no por el recorte: si no, un recorte con mucho fondo
    dilluye la cara y el arbitro parece un tirador solo por estar lejos."""
    cara = float(np.isin(m, CARA_PELO).sum()) / npers
    ropa = float(np.isin(m, ROPA).sum()) / npers
    manos = float(np.isin(m, MANOS).sum()) / npers
    frac_p = float(persona.mean())
    u, cnt = np.unique(m, return_counts=True)
    top = ", ".join("%s %.0f%%" % (NOMBRES.get(int(cl), str(cl)), 100.0 * n / m.size)
                    for cl, n in sorted(zip(u.tolist(), cnt.tolist()), key=lambda t: -t[1])[:4])
    filas.append({"i": i, "conf": float(c), "caja": (x1, y1, x2, y2),
                  "alto_frac": (y2 - y1) / H, "cx": 0.5 * (x1 + x2) / W,
                  "frac_persona": frac_p, "cara_pelo": cara, "ropa": ropa, "manos": manos,
                  "top": top})
    print("  cand %d  conf %.2f  caja %4dx%-4d  alto %.3f  cx %.3f" % (
        i, c, x2 - x1, y2 - y1, (y2 - y1) / H, 0.5 * (x1 + x2) / W))
    print("          persona %.3f   CARA+PELO %.4f   ropa %.3f   manos %.4f" % (
        frac_p, cara, ropa, manos))
    print("          %s" % top)
    col = (0, 200, 255) if cara > 0.02 else (0, 255, 0)
    cv2.rectangle(lienzo, (x1, y1), (x2, y2), col, 3)
    cv2.putText(lienzo, "%d cara %.3f" % (i, cara), (x1, max(16, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 2)
    vis = sub[:, :, ::-1].copy()
    vis[np.isin(m, CARA_PELO)] = (0, 0, 255)
    vis[np.isin(m, ROPA)] = (0, 200, 0)
    cv2.imwrite(str(SALIDA / ("cand%d_cara%.3f.png" % (i, cara))),
                cv2.resize(vis, (260, int(260 * vis.shape[0] / vis.shape[1]))))

cv2.rectangle(lienzo, (0, 0), (W, 42), (12, 12, 18), -1)
cv2.putText(lienzo, "%s c%d   VERDE cara+pelo < 2%% (tirador)   AMARILLO >= 2%% (cara descubierta)" % (
    CLIP[:26], CUADRO), (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2)
cv2.imwrite(str(SALIDA / ("%s_c%04d_a2.png" % (CLIP[:34], CUADRO))),
            cv2.resize(lienzo, (1200, int(1200 * H / W))))

import pandas as pd
d = pd.DataFrame(filas)
d.to_csv(SALIDA / ("a2_%s_c%04d.csv" % (CLIP, CUADRO)), index=False)
print()
print("=" * 96)
print("SEPARACION POR CARA+PELO")
print("=" * 96)
if len(d) >= 2:
    dd = d.sort_values("cara_pelo")
    for _, r in dd.iterrows():
        print("  cand %d  cara+pelo %.4f   alto %.3f  cx %.3f   %s" % (
            r["i"], r["cara_pelo"], r["alto_frac"], r["cx"],
            "<- cara descubierta" if r["cara_pelo"] > 0.02 else ""))
    print()
    bajos = dd[dd.cara_pelo <= 0.02]
    altos = dd[dd.cara_pelo > 0.02]
    print("  con cara+pelo <= 2 %% ... %d candidatos" % len(bajos))
    print("  con cara+pelo >  2 %% ... %d candidatos" % len(altos))
    if len(bajos) and len(altos):
        print("  hueco entre el mayor de los bajos (%.4f) y el menor de los altos (%.4f) = %.4f" % (
            bajos.cara_pelo.max(), altos.cara_pelo.min(),
            altos.cara_pelo.min() - bajos.cara_pelo.max()))
    else:
        print("  TODOS caen del mismo lado: en este cuadro la medida no separa nada,")
        print("  y hay que probarla donde SE VEA un arbitro para saber si sirve.")
print()
print("  imagenes en %s" % SALIDA)

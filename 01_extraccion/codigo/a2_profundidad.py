"""Puede la PROFUNDIDAD separar a los tiradores del arbitro y del publico?

LA IDEA ES DEL USUARIO y tiene una vuelta que no se habia probado. Lo medido hasta ahora es que la z
de `pointmap` es inestable ENTRE RECORTES: ampliando el recorte un 30 % cambia hasta un 694 %, y las
unidades no son metricas. Pero eso no impide lo que el usuario propone, porque su idea no necesita
escala absoluta: necesita **orden a la misma escala**.

Y en UNA SOLA PASADA DE CUADRO COMPLETO todos los candidatos comparten la misma escala desconocida.
Asi que la pregunta es distinta de la que se midio: no "cuanto mide la z" sino "ordena bien a la
gente de un mismo cuadro".

CRITERIO DECLARADO ANTES DE MIRAR:
  se lee la z mediana en el TORSO de cada candidato de YOLO, sobre un unico `pointmap` del cuadro
  completo. La idea SIRVE si los dos tiradores quedan juntos en z y el resto queda separado, con un
  hueco mayor que la dispersion dentro de cada tirador.
  Si el arbitro cae en el mismo rango que los tiradores, se dice que no separa.

Se compara con dos alternativas mas baratas para no adoptar la cara sin motivo:
  ALTO DE CAJA   a la misma distancia, dos personas miden parecido. Cuesta cero.
  Y DEL PIE      la perspectiva pone mas abajo lo que esta mas cerca. Cuesta cero.
"""
from __future__ import annotations

import os
import pathlib
import sys

CACHE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565\scratchpad\hoja_seg\hf")
os.environ["HF_HOME"] = str(CACHE)

import cv2
import numpy as np
import torch

PRUEBAS = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\a2_profundidad")
SALIDA.mkdir(parents=True, exist_ok=True)
CLIP = sys.argv[1] if len(sys.argv) > 1 else "SentimentalLargeBedlingtonterrier"
CUADRO = int(sys.argv[2]) if len(sys.argv) > 2 else 197

with np.load(PRUEBAS / ("candidatos_%s.npz" % CLIP), allow_pickle=True) as z:
    cajas, confs, cuadros, kps = z["cajas"], z["confs"], z["cuadros"], z["kps"]
    W, H = int(z["W"]), int(z["H"])
    video = str(z["video"])
sel = np.nonzero(cuadros == CUADRO)[0]
print("candidatos en el cuadro %d: %d" % (CUADRO, len(sel)))

cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_FRAMES, CUADRO)
ok, im = cap.read()
cap.release()
if not ok:
    raise SystemExit("no decodifica")

snap = next((CACHE / "hub/models--facebook--sapiens2-pointmap-0.4b/snapshots").iterdir())
from transformers import Sapiens2ForPointmapEstimation, Sapiens2ImageProcessor

proc = Sapiens2ImageProcessor.from_pretrained(str(snap))
mod = Sapiens2ForPointmapEstimation.from_pretrained(
    str(snap), dtype=torch.float16).to("cuda").eval()
rgb = np.ascontiguousarray(im[:, :, ::-1])
with torch.inference_mode():
    e = proc(images=rgb, return_tensors="pt")
    e = {k: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda")) for k, v in e.items()}
    o = mod(**e)
    m = proc.post_process_pointmap_estimation(o, source_sizes=[(H, W)], target_sizes=[(H, W)])[0]
pm = np.asarray((m["pointmap"] if isinstance(m, dict) else m).cpu(), dtype="float32")
if pm.ndim == 3 and pm.shape[0] in (3, 4):
    pm = np.transpose(pm, (1, 2, 0))
Z = np.nan_to_num(pm[:, :, 2])
print("pointmap del cuadro completo: %s   z de %.4f a %.4f" % (Z.shape, Z.min(), Z.max()))

L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12
filas = []
for i in sel:
    b = cajas[i]
    kp = kps[i]
    pts = []
    for j in (L_SHO, R_SHO, L_HIP, R_HIP):
        if kp[j, 2] >= 0.3 and np.isfinite(kp[j, 0]):
            pts.append((int(np.clip(kp[j, 1], 0, H - 1)), int(np.clip(kp[j, 0], 0, W - 1))))
    if len(pts) < 2:
        cy = int(np.clip(0.5 * (b[1] + b[3]), 0, H - 1))
        cx = int(np.clip(0.5 * (b[0] + b[2]), 0, W - 1))
        pts = [(cy, cx)]
    """Se toma una ventana pequena alrededor de cada punto del torso, no un pixel: un pixel suelto de
    una z ruidosa no dice nada."""
    vals = []
    for (yy, xx) in pts:
        y0, y1 = max(0, yy - 4), min(H, yy + 5)
        x0, x1 = max(0, xx - 4), min(W, xx + 5)
        vals.append(float(np.median(Z[y0:y1, x0:x1])))
    filas.append({"i": int(i), "conf": float(confs[i]),
                  "z": float(np.median(vals)), "z_disp": float(np.std(vals)),
                  "alto_frac": float((b[3] - b[1]) / H), "pie_frac": float(b[3] / H),
                  "cx": float(0.5 * (b[0] + b[2]) / W), "caja": [float(v) for v in b]})

import pandas as pd
d = pd.DataFrame(filas).sort_values("z")
d.to_csv(SALIDA / ("a2_prof_%s_c%04d.csv" % (CLIP, CUADRO)), index=False)

print()
print("=" * 92)
print("%5s %7s %9s %9s %9s %9s %7s" % ("cand", "conf", "z", "z_disp", "alto", "y_pie", "cx"))
print("=" * 92)
for _, r in d.iterrows():
    print("%5d %7.2f %9.4f %9.4f %9.3f %9.3f %7.3f" % (
        r["i"], r["conf"], r["z"], r["z_disp"], r["alto_frac"], r["pie_frac"], r["cx"]))

print()
print("=" * 92)
print("COMPARACION DE LAS TRES VIAS: cual agrupa a los dos mas grandes y separa al resto?")
print("=" * 92)
dd = d.sort_values("alto_frac", ascending=False)
dos = dd.head(2)
resto = dd.tail(len(dd) - 2)
for col, etq, coste in (("z", "profundidad de pointmap", "480 ms"),
                        ("alto_frac", "alto de caja", "0 ms"),
                        ("pie_frac", "y del pie", "0 ms")):
    a, b = float(dos[col].iloc[0]), float(dos[col].iloc[1])
    dentro = abs(a - b)
    fuera = float(np.min(np.abs(resto[col].values[:, None] - dos[col].values[None, :]))) if len(resto) else float("nan")
    razon = fuera / dentro if dentro > 1e-9 else float("inf")
    print("  %-26s (%s)  separacion dentro %.4f   al mas cercano de fuera %.4f   razon %.2f%s" % (
        etq, coste, dentro, fuera, razon, "   <- SEPARA" if razon > 1.0 else "   no separa"))
print()
print("  Una razon > 1 significa que los dos mas grandes estan mas cerca entre si que del resto.")
print("  Y OJO: 'los dos mas grandes' se toma como referencia de tirador SIN verificar a ojo;")
print("  si en este cuadro los dos mayores no son los tiradores, la comparacion no vale.")

vis = im.copy()
zmin, zmax = float(d["z"].min()), float(d["z"].max())
for _, r in d.iterrows():
    x1, y1, x2, y2 = [int(v) for v in r["caja"]]
    t = (r["z"] - zmin) / max(zmax - zmin, 1e-6)
    col = (int(255 * (1 - t)), 60, int(255 * t))
    cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
    cv2.putText(vis, "%d z=%.3f" % (r["i"], r["z"]), (x1, max(15, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
cv2.rectangle(vis, (0, 0), (W, 40), (12, 12, 18), -1)
cv2.putText(vis, "%s c%d  z de pointmap por candidato (azul cerca, rojo lejos)" % (CLIP[:30], CUADRO),
            (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2)
cv2.imwrite(str(SALIDA / ("%s_c%04d_z.png" % (CLIP[:34], CUADRO))),
            cv2.resize(vis, (1200, int(1200 * H / W))))
zn = np.clip((Z - np.percentile(Z, 2)) / max(np.percentile(Z, 98) - np.percentile(Z, 2), 1e-6), 0, 1)
cv2.imwrite(str(SALIDA / ("%s_c%04d_mapa_z.png" % (CLIP[:34], CUADRO))),
            cv2.applyColorMap((zn * 255).astype("uint8"), cv2.COLORMAP_JET))
print()
print("  imagenes en %s" % SALIDA)

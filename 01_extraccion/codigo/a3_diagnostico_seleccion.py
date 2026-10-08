"""El fallo esta en la DETECCION o en la SELECCION? Se pintan las dos cosas en el mismo cuadro.

LA PREGUNTA IMPORTA MUCHO para el plan: si YOLO no encuentra al tirador, hay que cambiar de detector
y reextraer todo. Si YOLO SI lo encuentra y el que fallaba era el selector, el arreglo es mucho mas
barato: se reelige sobre candidatos que ya existen.

Se pintan, sobre el cuadro verificado a ojo:
  VERDE     lo que propone YOLO ahora (conf 0,10, imgsz 960)
  ROJO      lo que eligio el algoritmo VIEJO y quedo guardado en el npz
Si el verde esta sobre el tirador y el rojo sobre la pared, la respuesta es "selector".
"""
from __future__ import annotations

import os
import pathlib

import cv2
import numpy as np

BASE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565")
PRUEBAS = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\a3_diagnostico")
SALIDA.mkdir(parents=True, exist_ok=True)
NPZ_DIRS = [BASE / "localrun/runs_extra1/data_20260907T005507157743Z/npz",
            BASE / "localrun/runs_extra2/data_20260907T005507370214Z/npz",
            BASE / "localrun/runs_extra3/data_20260907T010246490954Z/npz",
            BASE / "scratchpad/catalogo/ws_completo/npz"]

CASOS = [("SoreEasygoingEmperorshrimp", 65), ("SoreEasygoingEmperorshrimp", 120),
         ("SoreEasygoingEmperorshrimp", 200)]

npz_de = {}
for d in NPZ_DIRS:
    if d.is_dir():
        for f in d.glob("*.npz"):
            npz_de.setdefault(f.stem, f)

for cid, k in CASOS:
    cand_f = PRUEBAS / ("candidatos_%s.npz" % cid)
    if not cand_f.is_file():
        print("falta %s: corre antes a3_paso1_candidatos.py" % cand_f.name)
        continue
    with np.load(cand_f, allow_pickle=True) as z:
        cajas = z["cajas"]
        confs = z["confs"]
        cuadros = z["cuadros"]
        video = str(z["video"])
        W, H = int(z["W"]), int(z["H"])
    with np.load(npz_de[cid], allow_pickle=True) as z:
        VIEJO = np.asarray(z["boxes"], dtype="float64")

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, im = cap.read()
    cap.release()
    if not ok:
        print("no decodifica %s cuadro %d" % (cid, k))
        continue

    sel = cuadros == k
    print("%s cuadro %d: YOLO propone %d, el npz viejo tiene %d cajas" % (
        cid, k, int(sel.sum()),
        int(sum(1 for s in (0, 1) if np.isfinite(VIEJO[k, s]).all() and VIEJO[k, s][2] > VIEJO[k, s][0]))))

    for i, (b, c) in enumerate(zip(cajas[sel], confs[sel])):
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 220, 0), 3)
        cv2.putText(im, "YOLO %.2f  %dx%d" % (c, x2 - x1, y2 - y1), (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print("    YOLO %d: conf %.3f  caja %4d,%4d - %4d,%4d  (%dx%d px)" % (
            i, c, x1, y1, x2, y2, x2 - x1, y2 - y1))

    for s in (0, 1):
        b = VIEJO[k, s]
        if not (np.isfinite(b).all() and b[2] > b[0]):
            print("    VIEJO lado %d: sin caja" % s)
            continue
        x1, y1 = int(b[0] * W), int(b[1] * H)
        x2, y2 = int(b[2] * W), int(b[3] * H)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(im, "VIEJO lado %d" % s, (x1, min(H - 8, y2 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        print("    VIEJO lado %d: caja %4d,%4d - %4d,%4d  (%dx%d px)" % (
            s, x1, y1, x2, y2, x2 - x1, y2 - y1))

    """Solape entre lo nuevo y lo viejo: si es cero, eligieron personas distintas."""
    def iou(a, b):
        xa, ya = max(a[0], b[0]), max(a[1], b[1])
        xb, yb = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    for s in (0, 1):
        b = VIEJO[k, s]
        if not (np.isfinite(b).all() and b[2] > b[0]):
            continue
        vb = [b[0] * W, b[1] * H, b[2] * W, b[3] * H]
        mejor = max((iou(vb, list(cb)) for cb in cajas[sel]), default=0.0)
        print("    solape del VIEJO lado %d con el mejor candidato de YOLO: %.3f%s" % (
            s, mejor, "   <- NO es la misma persona" if mejor < 0.2 else ""))

    cv2.rectangle(im, (0, 0), (W, 44), (12, 12, 18), -1)
    cv2.putText(im, "%s cuadro %d   VERDE = YOLO ahora   ROJO = lo que eligio el algoritmo viejo" % (
        cid[:30], k), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 2)
    out = SALIDA / ("%s_c%04d_yolo_vs_viejo.png" % (cid[:34], k))
    cv2.imwrite(str(out), cv2.resize(im, (1200, int(1200 * H / W))))
    print("    -> %s" % out.name)
    print()

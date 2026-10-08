"""Los "esqueletos sobre pared vacia" eran contaminacion real o mi error de indice? Se pinta las dos.

EL ERROR: el NPZ no es el clip entero, es una VENTANA que empieza en `meta[f0]`. Yo pinte la pose del
cuadro k del NPZ sobre el cuadro k del VIDEO, cuando le corresponde el cuadro **f0 + k**. Con f0=175
eso son 175 cuadros de desfase, casi 6 segundos.

QUE SE PINTA, el mismo cuadro del npz dos veces:
  IZQUIERDA   pintado sobre el cuadro k del video        <- lo que hice ayer, MAL
  DERECHA     pintado sobre el cuadro f0 + k del video   <- lo correcto

Si a la derecha los esqueletos caen sobre personas, ayer me equivoque yo y hay que retirar el
hallazgo. Si siguen sobre la pared, la contaminacion es real y el desfase era un problema aparte.
"""
from __future__ import annotations

import json
import os
import pathlib

import cv2
import numpy as np

BASE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565")
DIRS = [BASE / "localrun/runs_extra1/data_20260907T005507157743Z/npz",
        BASE / "localrun/runs_extra2/data_20260907T005507370214Z/npz",
        BASE / "localrun/runs_extra3/data_20260907T010246490954Z/npz",
        BASE / "scratchpad/catalogo/ws_completo/npz"]
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\desfase_f0")
SALIDA.mkdir(parents=True, exist_ok=True)
HUESOS = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
          (11, 13), (13, 15), (12, 14), (14, 16)]
CONF = 0.3
CASOS = [("SoreEasygoingEmperorshrimp", 65), ("AltruisticIllfatedDobermanpinscher", 54)]

npz_de = {}
for d in DIRS:
    if d.is_dir():
        for f in d.glob("*.npz"):
            npz_de.setdefault(f.stem, f)
mapa = {}
for raiz, _d, fs in os.walk("D:/FENCINGLY_VIDEOS_RAW"):
    for f in fs:
        if f.lower().endswith(".mp4"):
            mapa.setdefault(pathlib.Path(f).stem, os.path.join(raiz, f))


def pinta(im, P, B, k, W, H, etq):
    for s, col in ((0, (255, 150, 0)), (1, (0, 200, 255))):
        b = B[k, s]
        if np.isfinite(b).all() and b[2] > b[0]:
            cv2.rectangle(im, (int(b[0] * W), int(b[1] * H)),
                          (int(b[2] * W), int(b[3] * H)), col, 3)
        kp = P[k, s]
        vis = (kp[:, 2] >= CONF) & np.isfinite(kp[:, 0]) & np.isfinite(kp[:, 1])
        for a, c in HUESOS:
            if vis[a] and vis[c]:
                cv2.line(im, (int(kp[a, 0] * W), int(kp[a, 1] * H)),
                         (int(kp[c, 0] * W), int(kp[c, 1] * H)), col, 3)
        for j in range(17):
            if vis[j]:
                cv2.circle(im, (int(kp[j, 0] * W), int(kp[j, 1] * H)), 4, (255, 255, 255), -1)
    cv2.rectangle(im, (0, 0), (W, 40), (12, 12, 18), -1)
    cv2.putText(im, etq, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (240, 240, 240), 2)
    return im


for cid, k in CASOS:
    with np.load(npz_de[cid], allow_pickle=True) as z:
        P = np.asarray(z["poses"], dtype="float64")
        B = np.asarray(z["boxes"], dtype="float64")
        meta = json.loads(str(z["meta_json"]))
    f0 = int(meta.get("f0", 0))
    cap = cv2.VideoCapture(mapa[cid])
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("%s  npz=%d cuadros  f0=%d  video=%d cuadros" % (cid[:40], P.shape[0], f0, T))
    print("   el cuadro %d del npz le corresponde el %d del video" % (k, f0 + k))

    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok1, im_mal = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(T - 1, f0 + k))
    ok2, im_bien = cap.read()
    cap.release()
    if not (ok1 and ok2):
        print("   no decodifica")
        continue

    a = pinta(im_mal.copy(), P, B, k, W, H,
              "MAL: pose del npz[%d] sobre el cuadro %d del video (lo que hice ayer)" % (k, k))
    b = pinta(im_bien.copy(), P, B, k, W, H,
              "BIEN: pose del npz[%d] sobre el cuadro %d = f0+%d" % (k, f0 + k, k))
    par = np.hstack([cv2.resize(a, (900, int(900 * H / W))),
                     cv2.resize(b, (900, int(900 * H / W)))])
    out = SALIDA / ("%s_npz%04d_mal_vs_bien.png" % (cid[:34], k))
    cv2.imwrite(str(out), par)
    print("   -> %s" % out.name)
    print()

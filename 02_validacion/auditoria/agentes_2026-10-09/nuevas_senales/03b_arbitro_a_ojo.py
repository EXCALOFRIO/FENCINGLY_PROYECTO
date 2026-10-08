"""Verificacion a OJO: dibuja la caja del candidato a arbitro y las de los dos tiradores sobre
un frame de video, para unos pocos clips con candidato_arbitro=True. Solo para mirar: no mide nada
por si solo.
"""
from __future__ import annotations

import pickle
import sys

import cv2
import numpy as np
import pandas as pd
import pathlib

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import rutas as RU  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import importlib.util
spec = importlib.util.spec_from_file_location("arb", str(pathlib.Path(__file__).parent / "03_arbitro.py"))
arb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arb)

OUT = arb.OUT
FRAMES = OUT.parent / "tmp_frames"
FRAMES.mkdir(parents=True, exist_ok=True)


def dibuja(cid):
    pdet = RU.TRABAJO / f"det_{cid}.pkl"
    ppos = RU.TRABAJO / f"poses_{cid}.npz"
    with open(pdet, "rb") as fh:
        cajas, idx, W, H, Tdet = pickle.load(fh)
    pos = np.load(ppos, allow_pickle=True)
    B, pres, T = pos["B"], pos["pres"], int(pos["T"])
    T = min(T, Tdet)
    por_cuadro, alto_medio = arb.candidatos_por_cuadro(cajas, idx, B, pres, W, H, T)
    pistas = arb.enlaza_pistas(por_cuadro, T)
    pistas.sort(key=lambda p: len(p["cuadros"]), reverse=True)
    mejor = pistas[0]
    # cuadro intermedio de la pista del candidato
    t_sel = mejor["cuadros"][len(mejor["cuadros"]) // 2]
    i = mejor["cuadros"].index(t_sel)
    cx, cy, h = mejor["cx"][i], mejor["cy"][i], mejor["h"][i]

    video = RU.video(cid)
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, t_sel)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(cid, "no se pudo leer el frame")
        return
    # caja candidata aproximada (se reconstruye un cuadrado centrado, el ancho real no se guardo)
    alto_px = h * H
    ancho_px = alto_px * 0.45
    x0, y0 = int((cx * W) - ancho_px / 2), int((cy * H) - alto_px / 2)
    x1, y1 = int((cx * W) + ancho_px / 2), int((cy * H) + alto_px / 2)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 3)
    cv2.putText(frame, "candidato", (x0, max(0, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2)
    for s in (0, 1):
        if pres[t_sel, s]:
            b = B[t_sel, s]
            p0 = (int(b[0] * W), int(b[1] * H))
            p1 = (int(b[2] * W), int(b[3] * H))
            cv2.rectangle(frame, p0, p1, (0, 255, 0), 2)
    dest = FRAMES / f"{cid}_arbitro.jpg"
    cv2.imwrite(str(dest), frame)
    print(cid, "->", dest, " cuadro", t_sel, "de", T)


if __name__ == "__main__":
    muestra = pd.read_csv(OUT / "arbitro_resultado.csv")
    cand = muestra[muestra.candidato_arbitro == True].sample(n=6, random_state=7)
    for cid in cand["clip"]:
        dibuja(cid)

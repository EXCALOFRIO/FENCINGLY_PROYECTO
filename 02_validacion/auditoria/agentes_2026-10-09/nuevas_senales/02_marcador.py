"""Senal 2: MARCADOR/OVERLAY. Viabilidad con cambio de pixeles simple (sin OCR: no hay libreria de
OCR instalada en el venv del proyecto, y la consigna es no meter nada pesado ahi).

LA DIFICULTAD, vista a ojo en 6 clips del ensayo del arbitro (03b): el overlay de marcador NO esta
en el mismo sitio en dos torneos. Hay barra inferior (estilo 'FencingVision'), ribbon superior
('TABLE 64'), graficos distintos por sede... Por eso aqui NO se asume una banda fija: se prueban
DOS bandas candidatas (arriba y abajo del cuadro, donde se ve overlay en la mayoria de broadcasts)
y se compara su cambio de pixeles con el cambio del cuadro COMPLETO (control: la accion de la
esgrima, que se mueve mucho). Un overlay real = cambio LOCALIZADO y pequenio (el digito) sobre un
fondo estatico; sin overlay esa banda es solo pista/publico y cambia tanto como el resto.
"""
from __future__ import annotations

import sys
import time

import cv2
import numpy as np
import pandas as pd
import pathlib

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import rutas as RU  # noqa: E402

BASE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop-FENCINGLY-AI"
                     r"\2c75119e-09b9-43eb-a841-faa6869534b2\scratchpad\nuevas_senales")
OUT = BASE / "out"

FRAC_TOP = 0.12
FRAC_BOTTOM = 0.20
UMBRAL_PIXEL = 25.0   # diferencia de intensidad para contar un pixel como "cambiado"


def touch_frame(cid):
    p = RU.TRABAJO / f"lights_{cid}.csv"
    if not p.is_file():
        return None
    d = pd.read_csv(p)
    if d.empty or "cuadro_encendido" not in d.columns:
        return None
    v = d["cuadro_encendido"].min()
    return int(v) if np.isfinite(v) else None


def frac_cambiada(a, b, umbral=UMBRAL_PIXEL):
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(axis=2)
    return float((diff > umbral).mean()), float(diff.mean())


def procesa_uno(cid):
    fila = dict(clip=cid)
    video = RU.video(cid)
    if video is None:
        fila["estado"] = "sin_video"
        return fila
    cap = cv2.VideoCapture(video)
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if T < 10:
        cap.release()
        fila["estado"] = "clip_corto"
        return fila
    tfrm = touch_frame(cid)
    t_antes = max(0, (tfrm - 10)) if tfrm is not None else max(0, T // 2 - 10)
    t_despues = T - 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, t_antes)
    ok1, f1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, t_despues)
    ok2, f2 = cap.read()
    cap.release()
    if not (ok1 and ok2):
        fila["estado"] = "no_se_pudo_leer"
        return fila
    H, W = f1.shape[:2]
    fila["t_antes"] = t_antes
    fila["t_despues"] = t_despues
    fila["cuadro_tocado"] = tfrm

    banda_top = (slice(0, int(H * FRAC_TOP)), slice(0, W))
    banda_bot = (slice(int(H * (1 - FRAC_BOTTOM)), H), slice(0, W))

    fc_full, md_full = frac_cambiada(f1, f2)
    fc_top, md_top = frac_cambiada(f1[banda_top], f2[banda_top])
    fc_bot, md_bot = frac_cambiada(f1[banda_bot], f2[banda_bot])

    fila.update(frac_cambiada_full=round(fc_full, 4), frac_cambiada_top=round(fc_top, 4),
                frac_cambiada_bottom=round(fc_bot, 4),
                ratio_top=round(fc_top / (fc_full + 1e-6), 3),
                ratio_bottom=round(fc_bot / (fc_full + 1e-6), 3))
    # heuristica: banda "tipo overlay" = bastante mas estatica que el resto del cuadro Y CON ALGO
    # de cambio (no completamente congelada / cuadro negro)
    fila["overlay_candidato_top"] = bool(fc_top < 0.5 * fc_full and 0.0005 < fc_top)
    fila["overlay_candidato_bottom"] = bool(fc_bot < 0.5 * fc_full and 0.0005 < fc_bot)
    fila["estado"] = "ok"
    return fila


def main():
    muestra = pd.read_csv(OUT / "muestra_clips.csv")
    # submuestra: leer frames de video es mas caro que leer CSVs; 80 clips alcanza para medir
    # viabilidad con denominador declarado
    sub = muestra.sample(n=min(80, len(muestra)), random_state=11)
    filas = []
    t0 = time.perf_counter()
    for i, (_, r) in enumerate(sub.iterrows()):
        f = procesa_uno(r["clip_id"])
        f["initiated"] = r["initiated"]
        f["torneo"] = r["torneo"]
        filas.append(f)
        if (i + 1) % 20 == 0:
            print(f"... {i + 1}/{len(sub)} ({time.perf_counter() - t0:.0f}s)")
    d = pd.DataFrame(filas)
    d.to_csv(OUT / "marcador_resultado.csv", index=False)
    dt = time.perf_counter() - t0
    print(f"\ntiempo total {dt:.1f}s  ({dt / len(sub):.2f} s/clip)")
    print(d["estado"].value_counts())
    ok = d[d.estado == "ok"]
    print(f"\ncon banda TOP candidata a overlay: {ok.overlay_candidato_top.sum()}/{len(ok)} "
          f"= {ok.overlay_candidato_top.mean():.3f}")
    print(f"con banda BOTTOM candidata a overlay: {ok.overlay_candidato_bottom.sum()}/{len(ok)} "
          f"= {ok.overlay_candidato_bottom.mean():.3f}")
    print(f"con NINGUNA banda candidata (ni arriba ni abajo): "
          f"{((~ok.overlay_candidato_top) & (~ok.overlay_candidato_bottom)).sum()}/{len(ok)}")
    print("\nfraccion de pixeles cambiados, banda bottom, resumen:")
    print(ok.frac_cambiada_bottom.describe())
    print("\nfraccion de pixeles cambiados, banda top, resumen:")
    print(ok.frac_cambiada_top.describe())


if __name__ == "__main__":
    main()

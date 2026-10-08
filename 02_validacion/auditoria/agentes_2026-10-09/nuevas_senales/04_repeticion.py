"""Senal 4: REPETICION / CAMARA LENTA. Deteccion barata de fotogramas DUPLICADOS (indicio de
pulldown/duplicacion de cuadros, tipico de un efecto de camara lenta barato) y de cambios de
cadencia segun los metadatos del contenedor (sin decodificar, con ffprobe).

NO se mide contra ninguna etiqueta humana: el catalogo no tiene campo de 'repeticion' o 'camara
lenta'. Es solo VIABILIDAD + coste, como pide la tarea para esta senial.
"""
from __future__ import annotations

import json
import subprocess
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

UMBRAL_LAXO = 1.0    # "casi estatico" (puede ser quietud real, p.ej. en guardia)
UMBRAL_ESTRICTO = 0.3  # "practicamente identico" (candidato de verdad a cuadro repetido)


def _frac(s):
    try:
        n, d = s.split("/")
        return float(n) / float(d) if float(d) else 0.0
    except (ValueError, ZeroDivisionError, AttributeError):
        return None


def metadatos_ffprobe(ruta):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=r_frame_rate,avg_frame_rate,nb_frames,duration", "-of", "json", ruta]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    try:
        d = json.loads(r.stdout or "{}")
        st = (d.get("streams") or [{}])[0]
    except (json.JSONDecodeError, IndexError):
        st = {}
    return st


def procesa_uno(cid):
    fila = dict(clip=cid)
    video = RU.video(cid)
    if video is None:
        fila["estado"] = "sin_video"
        return fila
    meta = metadatos_ffprobe(video)
    fila["r_frame_rate"] = meta.get("r_frame_rate")
    fila["avg_frame_rate"] = meta.get("avg_frame_rate")
    rf, af = _frac(meta.get("r_frame_rate")), _frac(meta.get("avg_frame_rate"))
    fila["cadencia_variable"] = (rf is not None and af is not None and abs(rf - af) > 0.5)

    cap = cv2.VideoCapture(video)
    prev = None
    n_dup_laxo = 0
    n_dup_estricto = 0
    n_leidos = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (160, 90))  # barato: decide sobre una miniatura
        if prev is not None:
            d = float(np.abs(small.astype(np.int16) - prev.astype(np.int16)).mean())
            if d < UMBRAL_LAXO:
                n_dup_laxo += 1
            if d < UMBRAL_ESTRICTO:
                n_dup_estricto += 1
        prev = small
        n_leidos += 1
    cap.release()
    fila["T_leidos"] = n_leidos
    fila["frac_dup_laxo_d<1.0"] = round(n_dup_laxo / max(1, n_leidos - 1), 4)
    fila["frac_dup_estricto_d<0.3"] = round(n_dup_estricto / max(1, n_leidos - 1), 4)
    fila["tiene_duplicados_estrictos"] = bool(n_dup_estricto > 0)
    fila["estado"] = "ok"
    return fila


def main():
    muestra = pd.read_csv(OUT / "muestra_clips.csv")
    sub = muestra.sample(n=min(60, len(muestra)), random_state=3)
    filas = []
    t0 = time.perf_counter()
    for i, (_, r) in enumerate(sub.iterrows()):
        f = procesa_uno(r["clip_id"])
        f["initiated"] = r["initiated"]
        filas.append(f)
        if (i + 1) % 15 == 0:
            print(f"... {i + 1}/{len(sub)} ({time.perf_counter() - t0:.0f}s)")
    d = pd.DataFrame(filas)
    d.to_csv(OUT / "repeticion_resultado.csv", index=False)
    dt = time.perf_counter() - t0
    print(f"\ntiempo total {dt:.1f}s  ({dt / len(sub):.2f} s/clip)")
    print(d["estado"].value_counts())
    ok = d[d.estado == "ok"]
    print(f"\ncon cadencia r!=avg en >0.5 fps (variable, segun metadatos del contenedor, sin "
          f"decodificar): {ok.cadencia_variable.sum()}/{len(ok)}")
    print(f"con algun par de cuadros PRACTICAMENTE IDENTICOS (d<{UMBRAL_ESTRICTO}, candidato a "
          f"cuadro repetido de verdad): "
          f"{ok.tiene_duplicados_estrictos.sum()}/{len(ok)} = "
          f"{ok.tiene_duplicados_estrictos.mean():.3f}")
    print("\nfraccion de pares 'estrictos' (d<0.3), resumen (solo donde > 0):")
    print(ok[ok.tiene_duplicados_estrictos]["frac_dup_estricto_d<0.3"].describe())
    print("\nfraccion de pares 'laxos' (d<1.0, incluye quietud real), resumen global:")
    print(ok["frac_dup_laxo_d<1.0"].describe())


if __name__ == "__main__":
    main()

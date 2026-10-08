"""Senal 3: ARBITRO. Viabilidad de aislar al arbitro en las cajas de YOLO (det_<clip>.pkl, que trae
TODAS las personas, no solo a los dos tiradores) y ver si hay movimiento suyo (candidato a levantar
el brazo) en los cuadros DESPUES del tocado.

NO hay ninguna etiqueta humana de "donde esta el arbitro" en el proyecto (confirmado: ni en el
manifiesto ni en DATOS_POR_CLIP.md hay tal campo -- de hecho `lado_valido`/`lado_no_valido_por`
documenta que el arbitro A VECES cuela como "tirador" en el 6,1% de los casos, y las "cuatro vias
probadas para cazarlo han fallado"). Asi que esto es VIABILIDAD, no una medida contra verdad: se
dice honestamente.

Metodo: por cuadro, las cajas de det_ que NO casan (IoU bajo) con ninguna caja de los dos tiradores
(poses_.B) son "candidatas". Se enlazan cuadro a cuadro (vecino mas cercano) en pistas; la pista mas
larga de cada clip es el "candidato a arbitro" si cubre >= UMBRAL_COBERTURA de los cuadros.
"""
from __future__ import annotations

import pickle
import sys
import time

import numpy as np
import pandas as pd
import pathlib

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import rutas as RU  # noqa: E402

BASE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop-FENCINGLY-AI"
                     r"\2c75119e-09b9-43eb-a841-faa6869534b2\scratchpad\nuevas_senales")
OUT = BASE / "out"

IOU_MAX_FENCER = 0.15     # por debajo de esto, la caja NO es de un tirador
REL_ALTO_MIN, REL_ALTO_MAX = 0.3, 2.5   # candidato: alto relativo al alto medio de tirador
MAX_DESPLAZ_NORM = 0.06   # enlace entre cuadros: desplazamiento max del centro, en fracciones de W
MAX_HUECO_PISTA = 10      # cuadros sin ver antes de cerrar una pista
UMBRAL_COBERTURA = 0.40   # fraccion de T que debe cubrir la pista mas larga para contar como candidato


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x1 - x0), max(0.0, y1 - y0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def touch_frame(cid):
    p = RU.TRABAJO / f"lights_{cid}.csv"
    if not p.is_file():
        return None
    d = pd.read_csv(p)
    if d.empty or "cuadro_encendido" not in d.columns:
        return None
    v = d["cuadro_encendido"].min()
    return int(v) if np.isfinite(v) else None


def candidatos_por_cuadro(cajas, idx, B, pres, W, H, T):
    """Devuelve lista (por cuadro) de cajas candidatas (no tirador), en px."""
    alturas_f = []
    for t in range(T):
        for s in (0, 1):
            if pres[t, s]:
                b = B[t, s]
                alturas_f.append((b[3] - b[1]) * H)
    alto_medio = float(np.median(alturas_f)) if alturas_f else None
    por_cuadro = [[] for _ in range(T)]
    if alto_medio is None or alto_medio <= 0:
        return por_cuadro, alto_medio
    orden = np.argsort(idx, kind="stable")
    cajas_o, idx_o = cajas[orden], idx[orden]
    inicio = np.searchsorted(idx_o, np.arange(T + 1))
    for t in range(T):
        cs = cajas_o[inicio[t]:inicio[t + 1]]
        if len(cs) == 0:
            continue
        fenc_px = []
        for s in (0, 1):
            if pres[t, s]:
                b = B[t, s]
                fenc_px.append((b[0] * W, b[1] * H, b[2] * W, b[3] * H))
        for c in cs:
            alto = c[3] - c[1]
            if not (REL_ALTO_MIN * alto_medio <= alto <= REL_ALTO_MAX * alto_medio):
                continue
            if any(iou(c, f) >= IOU_MAX_FENCER for f in fenc_px):
                continue
            cx, cy = (c[0] + c[2]) / 2.0 / W, (c[1] + c[3]) / 2.0 / H
            por_cuadro[t].append((cx, cy, alto / H))
    return por_cuadro, alto_medio


def enlaza_pistas(por_cuadro, T):
    """Vecino mas cercano, cuadro a cuadro. Devuelve lista de pistas: [{cuadros:[], cx:[], cy:[], h:[]}]."""
    activas = []  # cada una: dict con ultimo_cuadro, cx,cy,h (listas), cuadros (lista)
    cerradas = []
    for t in range(T):
        cands = por_cuadro[t]
        usados = set()
        for pista in activas:
            if not cands:
                continue
            ultx, ulty = pista["cx"][-1], pista["cy"][-1]
            mejor, mejor_d = None, MAX_DESPLAZ_NORM
            for i, (cx, cy, h) in enumerate(cands):
                if i in usados:
                    continue
                dd = ((cx - ultx) ** 2 + (cy - ulty) ** 2) ** 0.5
                if dd < mejor_d:
                    mejor, mejor_d = i, dd
            if mejor is not None:
                cx, cy, h = cands[mejor]
                pista["cuadros"].append(t)
                pista["cx"].append(cx)
                pista["cy"].append(cy)
                pista["h"].append(h)
                pista["ultimo"] = t
                usados.add(mejor)
        for i, (cx, cy, h) in enumerate(cands):
            if i in usados:
                continue
            activas.append(dict(cuadros=[t], cx=[cx], cy=[cy], h=[h], ultimo=t))
        restantes = []
        for pista in activas:
            if t - pista["ultimo"] > MAX_HUECO_PISTA:
                cerradas.append(pista)
            else:
                restantes.append(pista)
        activas = restantes
    cerradas.extend(activas)
    return cerradas


def procesa_uno(cid):
    fila = dict(clip=cid)
    pdet = RU.TRABAJO / f"det_{cid}.pkl"
    ppos = RU.TRABAJO / f"poses_{cid}.npz"
    if not pdet.is_file() or not ppos.is_file():
        fila["estado"] = "sin_ficheros"
        return fila
    with open(pdet, "rb") as fh:
        cajas, idx, W, H, Tdet = pickle.load(fh)
    pos = np.load(ppos, allow_pickle=True)
    B, pres, T = pos["B"], pos["pres"], int(pos["T"])
    T = min(T, Tdet)
    if T < 5:
        fila["estado"] = "clip_corto"
        return fila
    por_cuadro, alto_medio = candidatos_por_cuadro(cajas, idx, B, pres, W, H, T)
    fila["alto_medio_tirador_px"] = round(alto_medio, 1) if alto_medio else None
    pistas = enlaza_pistas(por_cuadro, T)
    if not pistas:
        fila["estado"] = "sin_candidatos"
        fila["n_candidatos_cuadro_medio"] = float(np.mean([len(c) for c in por_cuadro]))
        return fila
    pistas.sort(key=lambda p: len(p["cuadros"]), reverse=True)
    mejor = pistas[0]
    cobertura = len(mejor["cuadros"]) / T
    fila["n_pistas"] = len(pistas)
    fila["cobertura_mejor_pista"] = round(cobertura, 3)
    fila["candidato_arbitro"] = bool(cobertura >= UMBRAL_COBERTURA)
    fila["cx_mediana"] = round(float(np.median(mejor["cx"])), 3)
    fila["cy_mediana"] = round(float(np.median(mejor["cy"])), 3)
    fila["n_candidatos_cuadro_medio"] = float(np.mean([len(c) for c in por_cuadro]))

    tfrm = touch_frame(cid)
    fila["cuadro_tocado"] = tfrm
    if tfrm is not None:
        cuadros_tras = [t for t in range(tfrm, T)]
        fila["n_cuadros_tras_tocado"] = len(cuadros_tras)
        if fila["candidato_arbitro"]:
            cs = np.array(mejor["cuadros"])
            tras = cs[cs >= tfrm]
            fila["n_cuadros_pista_tras_tocado"] = int(len(tras))
            fila["frac_pista_cubierta_tras_tocado"] = (
                round(len(tras) / max(1, len(cuadros_tras)), 3))
            # proxy de movimiento (NO validado): variacion del alto de caja antes vs despues
            hs = np.array(mejor["h"])
            antes = hs[cs < tfrm]
            despues = hs[cs >= tfrm]
            if len(antes) >= 3 and len(despues) >= 3:
                fila["var_alto_antes"] = round(float(np.std(antes)), 4)
                fila["var_alto_despues"] = round(float(np.std(despues)), 4)
    fila["estado"] = "ok"
    return fila


def main():
    muestra = pd.read_csv(OUT / "muestra_clips.csv")
    filas = []
    t0 = time.perf_counter()
    for i, r in muestra.iterrows():
        f = procesa_uno(r["clip_id"])
        f["initiated"] = r["initiated"]
        filas.append(f)
        if (i + 1) % 30 == 0:
            print(f"... {i + 1}/{len(muestra)} ({time.perf_counter() - t0:.0f}s)")
    d = pd.DataFrame(filas)
    d.to_csv(OUT / "arbitro_resultado.csv", index=False)
    dt = time.perf_counter() - t0
    print(f"\ntiempo total {dt:.1f}s  ({dt / len(muestra):.2f} s/clip)")
    print(d["estado"].value_counts())
    ok = d[d.estado == "ok"]
    print(f"\nclips con candidato_arbitro (cobertura>={UMBRAL_COBERTURA}): "
          f"{ok.candidato_arbitro.sum()}/{len(ok)} = {ok.candidato_arbitro.mean():.3f}")
    print(f"de esos, con cuadros tras el tocado disponibles: "
          f"{ok[ok.candidato_arbitro].n_cuadros_pista_tras_tocado.notna().sum()}")
    print("\ncx/cy mediana del candidato (toda la muestra con candidato):")
    cand = ok[ok.candidato_arbitro]
    if len(cand):
        print(cand[["cx_mediana", "cy_mediana"]].describe())


if __name__ == "__main__":
    main()

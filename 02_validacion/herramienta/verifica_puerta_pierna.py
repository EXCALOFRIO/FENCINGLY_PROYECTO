"""Puerta de piernas en la pareja (PAREJA_2026-10-09): que clips cambian de pareja si se enciende.

Rehace en CPU la seleccion de la etapa 1 (`extrae_pareja.procesa` SIN redeteccion, que es lo unico
que necesita GPU) con `pareja_v2.PIERNA_K = 0` (apagada) y con el valor pedido, a partir de
`det_<clip>.pkl` y del mp4. Es el MISMO camino de codigo que corre la etapa 1, no una copia.

Para no tener el clip entero en memoria (1,8 GB a 1080p), la puerta corre dos veces: la primera con
un registro que anota que cuadros pide y no da ninguno, y la segunda con solo esos cuadros
decodificados. Lo que pide no depende de lo que recibe, asi que la segunda pasada es exacta.

Un clip CAMBIA si alguna de las dos cajas elegidas tiene IoU mediana < 0,5 con la de antes.
    python verifica_puerta_pierna.py 0.5 [--lista FICHERO] [--procesos 4]
        -> 02_validacion/auditoria/puerta_pierna_<K>.csv
"""
import argparse
import os
import pathlib
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "01_extraccion" / "codigo"))
import rutas as RU  # noqa: E402


def iou(a, b):
    x1 = np.maximum(a[..., 0], b[..., 0]); y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2]); y2 = np.minimum(a[..., 3], b[..., 3])
    i = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[..., 2] - z[..., 0]) * (z[..., 3] - z[..., 1])  # noqa: E731
    return i / (ar(a) + ar(b) - i + 1e-12)


class Registro:
    """Se hace pasar por la lista de fotogramas: anota cada cuadro pedido y no da ninguno."""

    def __init__(self):
        self.pedidos = set()

    def __getitem__(self, k):
        self.pedidos.add(int(k))
        raise KeyError(k)


def decodifica(ruta, ks):
    import cv2
    out, ks = {}, set(ks)
    if not ks:
        return out
    cap = cv2.VideoCapture(ruta)
    k, mx = 0, max(ks)
    while k <= mx:
        if k in ks:
            ok, im = cap.read()
            if not ok:
                break
            out[k] = im
        elif not cap.grab():
            break
        k += 1
    cap.release()
    return out


def uno(args):
    clip, K = args
    import extrae_pareja as EP
    import pareja_v2 as PAR
    try:
        cajas, cuadros, W, H, T = pickle.load(open(RU.TRABAJO / ("det_%s.pkl" % clip), "rb"))
        PAR.PIERNA_K = 0.0
        r0 = EP.procesa(cajas, cuadros, W, H, T)
        PAR.PIERNA_K = K
        reg = Registro()
        EP.procesa(cajas, cuadros, W, H, T, frames=reg)
        ruta = RU.video(clip)
        if not ruta:
            return {"clip": clip, "error": "sin video"}
        fr = decodifica(ruta, reg.pedidos)
        r1 = EP.procesa(cajas, cuadros, W, H, T, frames=fr)
        A, B = r0["B"], r1["B"]
        out = {"clip": clip, "cuadros_pedidos": len(reg.pedidos), "cuadros_leidos": len(fr),
               "motivo_antes": r0["motivo"][:70], "motivo_despues": r1["motivo"][:120]}
        for lado in (0, 1):
            v = np.isfinite(A[:, lado]).all(-1) & np.isfinite(B[:, lado]).all(-1)
            out["iou%d" % lado] = float(np.median(iou(A[v, lado], B[v, lado]))) if v.any() else np.nan
        if np.isfinite([out["iou0"], out["iou1"]]).all():
            out["cambia"] = bool(min(out["iou0"], out["iou1"]) < 0.5)
        else:
            out["cambia"] = r0["motivo"] != r1["motivo"]
        out["puerta_actua"] = "PIERNA:" in r1["motivo"]
        return out
    except Exception as ex:
        return {"clip": clip, "error": "%s: %s" % (type(ex).__name__, str(ex)[:120])}


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("K", type=float)
    ap.add_argument("--lista", default=None)
    ap.add_argument("--procesos", type=int, default=4)
    ap.add_argument("--salida", default=None)
    a = ap.parse_args()
    if a.lista:
        clips = [l.strip() for l in open(a.lista, encoding="utf-8") if l.strip()]
    else:
        clips = sorted(p.name[4:-4] for p in RU.TRABAJO.glob("det_*.pkl"))
    with ProcessPoolExecutor(a.procesos) as ex:
        d = pd.DataFrame(list(ex.map(uno, [(c, a.K) for c in clips], chunksize=4)))
    sal = pathlib.Path(a.salida) if a.salida else RU.VAL / "auditoria" / ("puerta_pierna_%s.csv" % a.K)
    d.to_csv(sal, index=False)
    err = d["error"].notna().sum() if "error" in d else 0
    print("clips %d  cambian %d  la puerta actua en %d  errores %d -> %s" % (
        len(d), int(d.get("cambia", pd.Series(dtype=bool)).fillna(False).sum()),
        int(d.get("puerta_actua", pd.Series(dtype=bool)).fillna(False).sum()), err, sal))


if __name__ == "__main__":
    main()

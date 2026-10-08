"""Puerta de razon de tallas en la pareja: que clips cambian de pareja si se pone, rehecho en CPU.

Rehace la seleccion (`extrae_pareja.procesa` SIN redeteccion, que es lo unico que necesita GPU) con
y sin `pareja_v2.RAZON_TALLA_MAX`, a partir de `det_<clip>.pkl`. Un clip CAMBIA si alguna de las dos
cajas elegidas tiene IoU mediana < 0,5 con la de antes.
    python verifica_puerta_talla.py 1.8 [N]   -> 02_validacion/auditoria/puerta_talla_<x>.csv
"""
import glob, os, pathlib, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "01_extraccion" / "codigo"))
import rutas as RU

def iou(a, b):
    x1 = np.maximum(a[..., 0], b[..., 0]); y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2]); y2 = np.minimum(a[..., 3], b[..., 3])
    i = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[..., 2] - z[..., 0]) * (z[..., 3] - z[..., 1])
    return i / (ar(a) + ar(b) - i + 1e-12)

def uno(args):
    f, umbral = args
    import extrae_pareja as EP, pareja_v2 as PAR
    c = os.path.basename(f)[4:-4]
    try:
        cajas, cuadros, W, H, T = pickle.load(open(f, "rb"))
        PAR.RAZON_TALLA_MAX = float("inf"); r0 = EP.procesa(cajas, cuadros, W, H, T)
        PAR.RAZON_TALLA_MAX = umbral; r1 = EP.procesa(cajas, cuadros, W, H, T)
        A, B = r0["B"], r1["B"]
        out = {"clip": c, "motivo_antes": r0["motivo"][:50], "motivo_despues": r1["motivo"][:50]}
        for l in (0, 1):
            v = np.isfinite(A[:, l]).all(-1) & np.isfinite(B[:, l]).all(-1)
            out["iou%d" % l] = float(np.median(iou(A[v, l], B[v, l]))) if v.any() else np.nan
        out["cambia"] = bool(min(out["iou0"], out["iou1"]) < 0.5) if np.isfinite([out["iou0"], out["iou1"]]).all() else (r0["motivo"] != r1["motivo"])
        return out
    except Exception as ex:
        return {"clip": c, "error": "%s: %s" % (type(ex).__name__, str(ex)[:80])}

if __name__ == "__main__":
    import pandas as pd
    u = float(sys.argv[1]); fs = sorted(glob.glob(str(RU.TRABAJO / "det_*.pkl")))
    if len(sys.argv) > 2: fs = fs[:int(sys.argv[2])]
    with ProcessPoolExecutor(5) as ex:
        d = pd.DataFrame(list(ex.map(uno, [(f, u) for f in fs], chunksize=10)))
    sal = RU.VAL / "auditoria" / ("puerta_talla_%s.csv" % u); d.to_csv(sal, index=False)
    print("clips %d  cambian %d  errores %d -> %s" % (len(d), int(d.cambia.fillna(False).sum()), d.get("error", pd.Series()).notna().sum(), sal))

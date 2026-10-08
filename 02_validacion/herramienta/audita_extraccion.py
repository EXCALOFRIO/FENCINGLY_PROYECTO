"""Auditoria de lo IRRECUPERABLE (etapas 1 y 4) sobre todo lo extraido, antes de la corrida de semanas.

Por clip, sintomas de averias conocidas (no la nota media):
  sin_fps / fps_raro      la cadencia escala TODAS las velocidades
  T_vs_video              cuadros del NPZ contra los que dice el contenedor
  pareja_no               la etapa 1 no encontro pareja
  util                    fraccion de cuadros utilizables
  dos_poses               fraccion de cuadros utiles con las dos poses con >=20 puntos de conf>0,3
  cruce_lados             fraccion de cuadros utiles con el tirador 0 a la DERECHA del 1 (cambio de lado)
  saltos                  saltos del centro de caja > 15 % del ancho en un cuadro (cambio de identidad)
  kp_fuera                puntos con conf>0,3 fuera de su caja (deberian ser 0)
  kp_nan                  puntos con conf>0,3 y coordenada no finita
  talla                   razon de altos de caja p50 entre los dos (alguien a otra distancia)
    python audita_extraccion.py [N]     -> 02_validacion/auditoria/auditoria_etapas_1_4.csv
"""
import glob, os, sys, pathlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "01_extraccion" / "codigo"))
import rutas as RU

def uno(f):
    c = os.path.basename(f)[6:-4]
    r = {"clip": c}
    try:
        z = np.load(f, allow_pickle=True)
        T, W, H = int(z["T"]), int(z["W"]), int(z["H"])
        KP, SC, B, mask = z["KP"], z["SC"], z["B"], z["mask"].astype(bool)
        fps = float(z["fps"]) if "fps" in z.files else 0.0
        r.update(T=T, W=W, H=H, fps=fps, motivo=str(z["motivo"])[:40], util=float(mask.mean()))
        try:
            import cv2
            cap = cv2.VideoCapture(str(z["video"]))
            nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
            r["T_video"] = nf
        except Exception:
            r["T_video"] = -1
        k = np.nonzero(mask)[0]
        if len(k) == 0:
            return r
        buenos = (SC[k] > 0.3).sum(-1) >= 20
        r["dos_poses"] = float(buenos.all(-1).mean())
        cx = 0.5 * (B[k, :, 0] + B[k, :, 2])
        v = np.isfinite(cx).all(-1)
        r["cruce_lados"] = float((cx[v, 0] > cx[v, 1]).mean()) if v.any() else np.nan
        sal = 0
        for l in (0, 1):
            d = np.abs(np.diff(cx[:, l])); d = d[np.isfinite(d)]
            sal += int((d > 0.15).sum())
        r["saltos"] = sal
        x0, y0, x1, y1 = [B[..., i][..., None] * s for i, s in ((0, W), (1, H), (2, W), (3, H))]
        hx, hy = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
        ok = SC > 0.3
        fin = np.isfinite(KP).all(-1)
        fuera = ok & fin & ((KP[..., 0] < x0 - hx) | (KP[..., 0] > x1 + hx) | (KP[..., 1] < y0 - hy) | (KP[..., 1] > y1 + hy))
        r["kp_fuera"] = int(fuera[k].sum()); r["kp_nan"] = int((ok & ~fin)[k].sum())
        h = (B[k, :, 3] - B[k, :, 1])
        hm = np.nanmedian(h, 0)
        r["talla"] = float(max(hm) / min(hm)) if np.all(hm > 0) else np.nan
    except Exception as ex:
        r["error"] = "%s: %s" % (type(ex).__name__, str(ex)[:60])
    return r

if __name__ == "__main__":
    import pandas as pd
    fs = sorted(glob.glob(str(RU.TRABAJO / "poses_*.npz")))
    if len(sys.argv) > 1: fs = fs[:int(sys.argv[1])]
    with ProcessPoolExecutor(4) as ex:
        filas = list(ex.map(uno, fs, chunksize=20))
    d = pd.DataFrame(filas)
    sal = RU.VAL / "auditoria" / "auditoria_etapas_1_4.csv"
    d.to_csv(sal, index=False)
    print("clips", len(d), "->", sal)

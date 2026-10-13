"""O5 de `CORRIDA_FINAL_2026-10-13.md` §10: un `poses_<clip>.npz` hecho con los 17 puntos de YOLO, SIN
Sapiens, para el scouting de los clips de una luz. A MEDIR: no entra en la corrida si no pasa su
criterio (zona, x del tocado, mano armada y eventos contra los de Sapiens; ver §10).

DE DONDE SALE. La etapa 1 ya guarda, en `detx_<clip>.npz`, la confianza y los 17 puntos COCO de CADA
caja que ve YOLO, en el orden de `det_<clip>.pkl`. La pareja elegida esta en `revision_<clip>.npz`
(`B`, normalizada). Por cuadro y lado se coge la deteccion de ese cuadro con mas IoU con la caja de la
pareja (>= `IOU_MIN`); las cajas redetectadas en recorte no tienen puntos y quedan sin pose, como un
cuadro que Sapiens no ve.

LOS PUNTOS van a los indices de Sapiens 2 que usa el resto del proyecto (`zona_tocado._MAPA`,
`video_revision.HUESOS`): nariz, ojos, orejas, hombros, codos, caderas, rodillas y tobillos, y las
munecas a las raices de mano (Sapiens 21 con el codo 8 = muneca derecha COCO 10; 42 con el codo 7 =
muneca izquierda COCO 9). Los pies (15-20) y la cara no existen en COCO: quedan NaN con confianza 0,
y quien los usa ya cae a su respaldo (el tobillo en `zona_tocado.apoyo`).

DESPUES, lo mismo que la ruta de Sapiens (`video_revision._fase2_uno`): puntos fuera de su caja a
confianza 0, poses casi vacias fuera (`PUNTOS_MIN_YOLO` de los 17), recorte de extremos e invencion de
las que faltan. `fuente = yolo17` en el npz, para que nadie lo confunda con una pose de Sapiens.
"""
from __future__ import annotations

import os
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "02_validacion" / "herramienta"))
import rutas as RU  # noqa: E402

COCO_A_SAPIENS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 42, 10: 21, 11: 9, 12: 10,
                  13: 11, 14: 12, 15: 13, 16: 14}
IOU_MIN = 0.7
PUNTOS_MIN_YOLO = 8
"""De los 17 puntos con confianza >= 0,3. Es el equivalente de `video_revision.PUNTOS_MIN` (20 de los
63 del cuerpo de Sapiens, casi un tercio): aqui 8 de 17, la mitad, porque COCO no trae pies ni manos y
una pose de tira en el borde suele dar menos de la mitad."""


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def construye(clip, origen_dir=None):
    """QUE DEVUELVE: el dict de arrays de un `poses_` (mismas claves que la ruta de Sapiens) o None."""
    import video_revision as VR
    d = pathlib.Path(origen_dir or RU.TRABAJO)
    pd, px, pr = d / ("det_%s.pkl" % clip), d / ("detx_%s.npz" % clip), d / ("revision_%s.npz" % clip)
    if not (pd.is_file() and px.is_file() and pr.is_file()):
        return None
    with open(pd, "rb") as fh:
        cajas, cuadros, W, H, T = pickle.load(fh)
    with np.load(px) as zx:
        kps = zx["kp"].astype("float32")
    z = np.load(pr, allow_pickle=True)
    B, pres, mask = z["B"], z["pres"].copy(), z["mask"].copy()
    Be = z["Be"] if "Be" in z.files else B
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    if len(kps) != len(cajas):
        return None
    por_cuadro = {}
    for i, k in enumerate(np.asarray(cuadros)):
        por_cuadro.setdefault(int(k), []).append(i)
    KP = np.full((T, 2, VR.N_PTS, 2), np.nan, dtype="float32")
    SC = np.zeros((T, 2, VR.N_PTS), dtype="float32")
    sin_det = 0
    for t in range(T):
        if not mask[t]:
            continue
        for lado in (0, 1):
            if not pres[t, lado]:
                continue
            b = B[t, lado]
            if not np.isfinite(b).all():
                continue
            bp = [b[0] * W, b[1] * H, b[2] * W, b[3] * H]
            mejor, mi = 0.0, None
            for i in por_cuadro.get(t, ()):
                v = _iou(bp, cajas[i])
                if v > mejor:
                    mejor, mi = v, i
            if mi is None or mejor < IOU_MIN:
                pres[t, lado] = False
                sin_det += 1
                continue
            k = kps[mi]
            for c, s in COCO_A_SAPIENS.items():
                KP[t, lado, s] = k[c, :2]
                SC[t, lado, s] = k[c, 2]
            xywh = [bp[0], bp[1], bp[2] - bp[0], bp[3] - bp[1]]
            SC[t, lado] = VR._fuera_de_caja(KP[t, lado], SC[t, lado], xywh)
    nvac = 0
    for t in range(T):
        if not mask[t]:
            continue
        for lado in (0, 1):
            if pres[t, lado] and int(((SC[t, lado] >= VR.CONF_PINTA) & np.isfinite(KP[t, lado, :, 0])).sum()) < PUNTOS_MIN_YOLO:
                pres[t, lado] = False
                KP[t, lado] = np.nan
                SC[t, lado] = 0.0
                nvac += 1
    mask, _nrec = VR.recorta_por_poses(pres, mask)
    KP, SC, origen, hueco = VR.inventa_poses(KP, SC, pres, mask, T)
    return dict(KP=KP, SC=SC, origen=origen, hueco=hueco, B=B, Be=Be, pres=pres, mask=mask, W=W, H=H, T=T,
                motivo=str(z["motivo"]), video=str(z["video"]), filtro_vacias=np.array(nvac, dtype="int32"),
                puntos_min=np.array(PUNTOS_MIN_YOLO, dtype="int32"),
                fps=np.float64(float(z["fps"]) if "fps" in z.files else 0.0),
                ambos=z["ambos"] if "ambos" in z.files else np.float64(float("nan")),
                fuente=np.array("yolo17"), sin_deteccion=np.array(sin_det, dtype="int32"))


def escribe(clip, destino, origen_dir=None):
    r = construye(clip, origen_dir)
    if r is None:
        return None
    p = pathlib.Path(destino) / ("poses_%s.npz" % clip)
    tmp = p.with_name(p.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **r)
    os.replace(tmp, p)
    return p

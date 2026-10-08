"""ETAPA 11: DONDE fue el tocado, en metros de pista, en el cuadro en que se encienden las luces.

## POR QUE ESTO EXISTE

La posicion de la etapa 7 no dice nada de la zona del tocado. Medido el 2026-10-08 sobre lo ya
extraido, la x del final del clip contra `strip_location` (verdad humana):

    etapa 7, 2.145 clips      acierto 0,388   siempre `middle` 0,404   correlacion x-zona -0,003

Correlacion cero: el origen de `ancla_zonas` es una convencion en la mayoria de los clips, asi que
la x no sabe en que parte de la pista esta nadie. Esta etapa toma otra ruta:

    1  el CUADRO del tocado sale de las luces (`luces.py`, la etapa 9; se reutiliza su CSV)
    2  la homografia sale de las RAYAS PINTADAS identificadas por el patron del reglamento
       (`pista_lineas.py`) y se PROPAGA con el movimiento de camara medido hasta ese cuadro
    3  los pies son los de Sapiens 2 (`poses_<clip>.npz`), los mismos de todo el proyecto

## LO MEDIDO, con su denominador

El criterio se declaro ANTES de medir: entra si bate a `middle` con un intervalo del 95 % que no
cruce el cero **y** bate a la etapa 7 sobre los mismos clips. La cifra va en `CIFRA` y en
`pipeline.py --lista`; si no lo cumple, la etapa va APAGADA con ella al lado.

## LO QUE NO HACE

- **No contesta siempre.** Sin luz no hay cuadro de tocado, y sin homografia confiable en +-0,5 s
  del tocado no hay metros: en esos casos `motivo` dice cual de las dos falto y los metros van
  `None`, no cero.
- **No sustituye a la etapa 7** para los eventos: es solo el sitio del tocado.
"""
from __future__ import annotations

import csv
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pista_lineas as PL  # noqa: E402
import rutas as RU  # noqa: E402

CFG_VIDEO = {"analysis_width": 640, "cut_threshold": 0.45}
CFG_PISTA = {"length_m": 14.0, "width_m": 1.5, "lines_m": [-7.0, -5.0, -2.0, 0.0, 2.0, 5.0, 7.0],
             "line_match_tol_m": 0.35, "min_conf": 0.35, "fencer_height_m": 1.65}
"""Los valores con los que se midio la cifra de abajo; cambiarlos obliga a volver a medir."""
VENTANA_S = 0.5
BANDA_ROTULO = 0.78
"""Fraccion de la altura a partir de la cual esta el rotulo del marcador: se excluye de las rayas."""
CIFRA = ("x del tocado: correlacion con la zona humana 0,387 (n=220; la x de la etapa 7 da -0,003). "
         "Zona: acierto 0,520 contra 0,448 de `middle`, +0,073 con IC95 [-0,005, +0,150]: NO pasa el "
         "criterio declarado por poco, asi que la columna `zona` NO esta adoptada. Cobertura 0,380 "
         "(sin luz 0,40, sin homografia en el tocado 0,22). Medido el 2026-10-08 en 582 clips anotados.")
"""LA CIFRA, con el criterio que se declaro antes de medir. Lo que SI se adopta es `x_tocado_m` como
DATO MEDIDO (la senial de posicion mas fuerte del proyecto, y la unica distinta de cero); lo que NO
se adopta es la ZONA como respuesta. El paso siguiente declarado: pasar `x_tocado_m` al etiquetador
en lugar de la x de la etapa 7 y medir su cabeza `strip_location` (hoy 0,463 contra 0,397)."""

COLUMNAS = ["clip", "motivo", "cuadro_tocado", "via_luz", "cuadro_usado", "x_izq_m", "x_der_m",
            "x_tocado_m", "zona", "conf_h", "metodo_h", "frac_cuadros_con_h", "s"]

ZONAS = ["fotl_warning_box", "fotl_half", "middle", "fotr_half", "fotr_warning_box"]

# Sapiens 2 (308) -> el orden Halpe26 que esperan la mascara y el apoyo de `pista_lineas`.
# Indices de Sapiens MEDIDOS en este proyecto (ver DATOS_POR_CLIP.md): 5-6 hombros, 7-8 codos,
# 9-10 caderas, 11-12 rodillas, 13-14 tobillos, 15-17 / 18-20 dedo gordo, dedo pequenio, talon,
# 21 y 42 las raices de las manos (no hay munieca: la raiz ES la munieca).
_MAPA = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 21, 10: 42, 11: 9, 12: 10,
         13: 11, 14: 12, 15: 13, 16: 14, 20: 15, 21: 18, 22: 16, 23: 19, 24: 17, 25: 20}
_PIE = {0: [24, 20, 22], 1: [25, 21, 23]}
_TOBILLO = {0: 15, 1: 16}


def a_halpe(KP, SC):
    out = np.zeros(KP.shape[:2] + (26, 3), np.float32)
    for h, s in _MAPA.items():
        out[..., h, :2] = KP[..., s, :]
        out[..., h, 2] = SC[..., s]
    return out


def apoyo(k, caja, thr=0.3):
    """El punto de apoyo: media de talon y puntas de los dos pies; el tobillo si faltan; la base de
    la caja si falta todo."""
    pts = []
    for lado in (0, 1):
        idx = _PIE[lado]
        bien = k[idx, 2] > thr
        if bien.sum() >= 2:
            pts.append(k[idx][bien, :2].mean(0))
        elif k[_TOBILLO[lado], 2] > thr:
            pts.append(k[_TOBILLO[lado], :2])
    if pts:
        return np.mean(pts, 0).astype(np.float32)
    return np.array([(caja[0] + caja[2]) / 2, caja[3]], np.float32)


def largo_cuerpo(k, thr=0.3):
    """Tobillo-rodilla-cadera + cadera-hombro + hombro-nariz, en px (~0,93 x estatura)."""
    def seg(a, b):
        return np.linalg.norm(k[a, :2] - k[b, :2]) if k[a, 2] > thr and k[b, 2] > thr else np.nan
    pierna = np.nanmean([seg(15, 13) + seg(13, 11), seg(16, 14) + seg(14, 12)])
    if min(k[[5, 6, 11, 12], 2]) <= thr or not np.isfinite(pierna):
        return np.nan
    cad, hom = (k[11, :2] + k[12, :2]) / 2, (k[5, :2] + k[6, :2]) / 2
    tronco = np.linalg.norm(hom - cad)
    cuello = np.linalg.norm(k[0, :2] - hom) if k[0, 2] > thr else 0.18 * (pierna + tronco)
    return float(pierna + tronco + cuello)


def zona(x, m=2.0):
    """La misma particion que `verifica_ancla_v2.zona`: +-2 m puesta en guardia, +-5 m aviso."""
    return 0 if x < -5 else 4 if x > 5 else 1 if x < -m else 3 if x > m else 2


def cuadro_tocado(clip, video, torneo):
    """QUE DEVUELVE: (cuadro, via). Reutiliza `lights_<clip>.csv` si la etapa 9 ya lo escribio."""
    p = RU.TRABAJO / ("lights_%s.csv" % clip)
    if p.is_file() and p.stat().st_size > 0:
        with open(p, encoding="utf-8") as fh:
            filas = [f for f in csv.DictReader(fh) if f.get("cuadro_encendido")]
        if filas:
            return int(min(int(float(f["cuadro_encendido"])) for f in filas)), filas[0].get("via")
        return None, "lights csv sin luces"
    import luces as LU
    roi = RU.roi_sede(torneo)
    det = LU.detecta(video, banda=roi["banda"]) if roi else LU.detecta(video)
    fps = det.get("fps") or 30.0
    gr = LU.grupo_del_tocado(LU.agrupa(det["luces"], fps), fps)
    via = "roi_sede" if roi else "general"
    if roi and not gr:
        det = LU.detecta(video)
        gr = LU.grupo_del_tocado(LU.agrupa(det["luces"], fps), fps)
        via = "general_tras_roi"
    if not gr:
        return None, via
    return int(min(g["cuadro_encendido"] for g in gr)), via


def analiza(clip, video, torneo=None):
    t0 = time.perf_counter()
    r = {"clip": clip}
    z = np.load(RU.TRABAJO / ("poses_%s.npz" % clip), allow_pickle=True)
    if "KP" not in z.files:
        r["motivo"] = "poses con otro esquema"
        return r
    KP, SC, B, origen = z["KP"], z["SC"], z["B"], z["origen"]
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    fps = float(z["fps"]) if "fps" in z.files else 30.0
    tf, via = cuadro_tocado(clip, video, torneo)
    r.update(cuadro_tocado=tf, via_luz=via)
    if tf is None:
        r["motivo"] = "sin luz"
        return r
    if tf >= T:
        r["motivo"] = "tocado fuera de las poses"
        return r
    k26 = a_halpe(KP, SC)
    Bpx = B * np.array([W, H, W, H])
    cap = cv2.VideoCapture(video)
    pf = PL.PisteFeatures(CFG_VIDEO, (W, H))
    for t in range(min(T, tf + 3)):
        ok, img = cap.read()
        if not ok:
            break
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H))
        okb = np.isfinite(Bpx[t]).all(1)
        pf.process(img, Bpx[t][okb].astype(np.float32), BANDA_ROTULO * H, k26[t][okb])
    cap.release()
    Tn = len(pf.feats)
    pies = np.full((Tn, 2, 2), np.nan, np.float32)
    alt = np.full((Tn, 2), np.nan, np.float32)
    for t in range(Tn):
        for s in (0, 1):
            if origen[t, s] in (0, 1) and np.isfinite(Bpx[t, s]).all():
                pies[t, s] = apoyo(k26[t, s], Bpx[t, s])
                alt[t, s] = largo_cuerpo(k26[t, s])
    cal = PL.calibrate(pf.feats, pies, alt, CFG_PISTA, (W, H))
    Hs, conf = cal["H"], cal["conf"]
    r["frac_cuadros_con_h"] = round(float((conf >= CFG_PISTA["min_conf"]).mean()), 3) if Tn else 0.0
    tq = min(tf, Tn - 1)
    w = int(VENTANA_S * fps)
    cand = [t for t in range(max(0, tq - w), min(Tn, tq + 2))
            if np.isfinite(Hs[t]).all() and conf[t] >= CFG_PISTA["min_conf"]]
    if not cand:
        r["motivo"] = "sin homografia confiable en +-%.1f s del tocado" % VENTANA_S
        r["s"] = round(time.perf_counter() - t0, 1)
        return r
    tt = min(cand, key=lambda t: abs(t - tq))
    xs = [float(PL.to_m(Hs[tt], pies[tt, s][None])[0, 0]) if np.isfinite(pies[tt, s]).all()
          else None for s in (0, 1)]
    vx = [x for x in xs if x is not None]
    if not vx:
        r["motivo"] = "sin pies en el cuadro usado"
        return r
    x = float(np.mean(vx))
    r.update(motivo="ok", cuadro_usado=tt, x_izq_m=xs[0], x_der_m=xs[1], x_tocado_m=round(x, 3),
             zona=ZONAS[zona(x)], conf_h=round(float(conf[tt]), 3), metodo_h=str(cal["method"][tt]),
             s=round(time.perf_counter() - t0, 1))
    return r


def escribe(r):
    p = RU.TRABAJO / ("zona_%s.csv" % r["clip"])
    tmp = p.with_suffix(".csv.part")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerow({k: r.get(k) for k in COLUMNAS})
    tmp.replace(p)
    return p


def corre(clip, ctx):
    """Para `pipeline.py`. Escribe SIEMPRE el CSV, tambien cuando no hay respuesta: un clip sin luz
    es un clip hecho cuya respuesta es `sin luz`, y si no se escribiera se reintentaria sin fin."""
    r = analiza(clip, ctx["video"], ctx.get("torneo"))
    escribe(r)
    return "%s%s" % (r.get("motivo"), (" x=%.2f m %s" % (r["x_tocado_m"], r["zona"]))
                     if r.get("motivo") == "ok" else "")


if __name__ == "__main__":
    import catalogo as CA
    df = CA.manifiesto().set_index("clip_id")
    for c in sys.argv[1:]:
        print(c, corre(c, {"video": RU.video(c), "torneo": df.loc[c, "torneo"] if c in df.index else None}))

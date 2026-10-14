"""SCOUTING V2: la posicion de los dos tiradores en metros de pista, CUADRO A CUADRO, con la homografia
por cuadro de la etapa 11 (`pista_h_<clip>.npz`, que escribe `zona_tocado.guarda_h`).

POR QUE EXISTE (2026-10-14, `04_documentacion/informes/SCOUTING_V2_2026-10-14.md`). La etapa 7
(`grafica_pista`) solo tiene homografia en ~52 % de los anotados, y su x no es una posicion en la pista:
va relativa a un ancla porque no sabe donde esta el centro, y la zona del tocado sacada de ella acierta
0,346 (absoluta 0,283), menos que contestar siempre `middle` (0,399), en 1.255 anotados de florete. La
etapa 11 calibra la pista en cada cuadro con las rayas (`pista_lineas.calibrate`) y acierta 0,495. Sus
desplazamientos y los de la 7 si cuadran (mismo sentido 85-93 %). Esto saca de la 11 la serie entera.

QUE HACE, por clip:
  1. proyecta el punto de apoyo de cada tirador (`pies`, en pixeles, el de `zona_tocado.apoyo`) con la
     homografia de su cuadro, si la hay con confianza >= `min_conf`;
  2. tira lo imposible, y dice por que: fuera de la pista (`X_RECHAZO`, `Y_RECHAZO`), salto contra la
     mediana movil (`DESV_MAX_M`), velocidad imposible entre cuadros (`V_MAX`), y los dos tiradores
     cruzados mas de `CRUCE_M` (en florete no se adelantan: es la identidad o la homografia);
  3. rellena los huecos cortos (<= `HUECO_MAX_S`) por interpolacion lineal y los marca (`estado` 1);
  4. suaviza para la velocidad (media movil de `SUAVE_S`, solo dentro de tramos sin hueco).

NO toca la etapa 7 ni la 11: lee `pista_h_` y `poses_` y escribe `scout_<clip>.npz` (atomico). Sin
`pista_h_` (clip sin luz, o la 11 no llego a calibrar) devuelve None.

    python scouting_v2.py CLIP [CLIP ...]        escribe scout_ en TRABAJO y saca el resumen
"""
from __future__ import annotations

import os
import sys

import numpy as np

import rutas as RU

X_PISTA = 7.5
"""Media pista (las lineas de fondo a +-7 m) mas medio metro de holgura: lo que se mide contra la fisica."""
X_RECHAZO = 9.0
"""Fuera de +-9 m no hay pista (14 m + 1,5-2 m de salida por cada lado): ahi la homografia esta mal."""
Y_RECHAZO = 2.0
"""La pista tiene 1,5-2 m de ancho centrados en y = 0: mas de 2 m a un lado no es estar sobre ella."""
VENT_MED_S = 0.3
DESV_MAX_M = 0.6
"""Un punto a mas de 0,6 m de la mediana de sus 0,3 s es un salto de la homografia de ese cuadro, no un
movimiento: en 0,15 s ni una flecha (~5-6 m/s) recorre eso respecto a su propia mediana."""
V_MAX = 8.0
"""m/s entre dos cuadros medidos seguidos. La flecha llega a ~5-6 m/s; por encima de 8 es un salto."""
CRUCE_M = 0.5
"""El de la derecha mas de medio metro a la izquierda del de la izquierda: identidad o homografia mal."""
HUECO_MAX_S = 0.5
SUAVE_S = 0.33
"""Media movil de 0,33 s para la velocidad. Medido en los 99 clips NO anotados de los ensayos (desarrollo,
fuera de la validacion): con 0,2 s, |v| <= 6 m/s en 99,0 % (p99 5,95); con 0,33, 99,3 % (p99 5,12); con 0,5,
99,6 % pero aplana el pico de un fondo (~0,3-0,5 s)."""

ESTADO_NADA, ESTADO_MEDIDO, ESTADO_INTERP = -1, 0, 1
MOTIVOS = ("sin_h", "sin_pies", "fuera", "salto", "velocidad", "cruce")


def _proyecta(H, conf, pies, min_conf):
    """QUE DEVUELVE: xy (T, 2, 2) en metros y motivo (T, 2) de cada hueco (indice en MOTIVOS, -1 si hay)."""
    import pista_lineas as PL
    T = len(H)
    xy = np.full((T, 2, 2), np.nan, np.float64)
    mot = np.full((T, 2), -1, np.int8)
    hok = np.isfinite(H.reshape(T, -1)).all(1) & (conf >= min_conf)
    for t in range(T):
        for s in (0, 1):
            if not hok[t]:
                mot[t, s] = 0
            elif not np.isfinite(pies[t, s]).all():
                mot[t, s] = 1
            else:
                xy[t, s] = PL.to_m(H[t], pies[t, s][None])[0]
    return xy, mot


def _mediana_movil(x, w):
    """Mediana de los valores finitos en una ventana centrada de `w` cuadros (None donde hay menos de 3)."""
    T = len(x)
    out = np.full(T, np.nan)
    h = w // 2
    for t in range(T):
        v = x[max(0, t - h):t + h + 1]
        v = v[np.isfinite(v)]
        if len(v) >= 3:
            out[t] = np.median(v)
    return out


def _limpia(xy, mot, fps):
    """Tira lo imposible (marca el motivo) y devuelve xy limpio."""
    xy = xy.copy()
    fuera = (np.abs(xy[..., 0]) > X_RECHAZO) | (np.abs(xy[..., 1]) > Y_RECHAZO)
    mot[fuera] = MOTIVOS.index("fuera")
    xy[fuera] = np.nan
    w = max(3, int(round(VENT_MED_S * fps)) | 1)
    for s in (0, 1):
        med = _mediana_movil(xy[:, s, 0], w)
        salto = np.isfinite(xy[:, s, 0]) & np.isfinite(med) & (np.abs(xy[:, s, 0] - med) > DESV_MAX_M)
        mot[salto, s] = MOTIVOS.index("salto")
        xy[salto, s] = np.nan
        for _ in range(3):
            idx = np.flatnonzero(np.isfinite(xy[:, s, 0]))
            if len(idx) < 2:
                break
            v = np.abs(np.diff(xy[idx, s, 0])) * fps / np.diff(idx)
            malos = np.flatnonzero(v > V_MAX)
            if not len(malos):
                break
            med = _mediana_movil(xy[:, s, 0], w)
            for m in malos:
                a, b = idx[m], idx[m + 1]
                da = abs(xy[a, s, 0] - med[a]) if np.isfinite(med[a]) else 0.0
                db = abs(xy[b, s, 0] - med[b]) if np.isfinite(med[b]) else 0.0
                q = a if da > db else b
                mot[q, s] = MOTIVOS.index("velocidad")
                xy[q, s] = np.nan
    ambos = np.isfinite(xy[:, 0, 0]) & np.isfinite(xy[:, 1, 0])
    cruce = ambos & (xy[:, 1, 0] - xy[:, 0, 0] < -CRUCE_M)
    mot[cruce] = MOTIVOS.index("cruce")
    xy[cruce] = np.nan
    return xy, mot


def _rellena(xy, util, fps):
    """Interpola los huecos de hasta `HUECO_MAX_S` entre dos cuadros medidos, solo en cuadros utiles."""
    estado = np.where(np.isfinite(xy[..., 0]), ESTADO_MEDIDO, ESTADO_NADA).astype(np.int8)
    hmax = int(round(HUECO_MAX_S * fps))
    out = xy.copy()
    for s in (0, 1):
        idx = np.flatnonzero(estado[:, s] == ESTADO_MEDIDO)
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= hmax + 1 and util[a:b + 1].all():
                f = (np.arange(a + 1, b) - a) / (b - a)
                out[a + 1:b, s] = xy[a, s] + f[:, None] * (xy[b, s] - xy[a, s])
                estado[a + 1:b, s] = ESTADO_INTERP
    return out, estado


def _suaviza(x, fps):
    """Media movil centrada de `SUAVE_S` dentro de cada tramo seguido con dato; NaN fuera."""
    k = max(1, int(round(SUAVE_S * fps)) | 1)
    out = np.full_like(x, np.nan)
    ok = np.isfinite(x)
    t = 0
    T = len(x)
    while t < T:
        if not ok[t]:
            t += 1
            continue
        u = t
        while u < T and ok[u]:
            u += 1
        seg = x[t:u]
        if len(seg) >= k:
            pad = np.pad(seg, k // 2, mode="edge")
            out[t:u] = np.convolve(pad, np.ones(k) / k, mode="valid")
        else:
            out[t:u] = seg
        t = u
    return out


def posiciones(clip, trabajo=None):
    """QUE DEVUELVE: dict con la serie del clip, o None si no hay `pista_h_` (o no hay `poses_`)."""
    T_ = pathlib_(trabajo)
    ph, pp = T_ / ("pista_h_%s.npz" % clip), T_ / ("poses_%s.npz" % clip)
    if not (ph.is_file() and pp.is_file()):
        return None
    with np.load(ph, allow_pickle=True) as z:
        H, conf, pies = z["H"].astype(np.float64), z["conf"].astype(np.float64), z["pies"].astype(np.float64)
        fps, tf, min_conf = float(z["fps"]) or 30.0, int(z["cuadro_tocado"]), float(z["min_conf"])
    with np.load(pp, allow_pickle=True) as z:
        mask = z["mask"].astype(bool)
    T = len(H)
    util = np.zeros(T, bool)
    util[:min(T, len(mask))] = mask[:T]
    xy, mot = _proyecta(H, conf, pies, min_conf)
    xy[~util] = np.nan
    xy, mot = _limpia(xy, mot, fps)
    xy, estado = _rellena(xy, util, fps)
    mot[estado >= 0] = -1
    xs = np.stack([_suaviza(xy[:, s, 0], fps) for s in (0, 1)], 1)
    vel = np.full_like(xs, np.nan)
    for s in (0, 1):
        ok = np.isfinite(xs[:, s])
        if ok.sum() >= 3:
            g = np.gradient(np.where(ok, xs[:, s], np.nan)) * fps
            vel[:, s] = np.where(ok, g, np.nan)
    return {"clip": clip, "x": xy[..., 0].astype(np.float32), "y": xy[..., 1].astype(np.float32),
            "x_suave": xs.astype(np.float32), "vel": vel.astype(np.float32), "estado": estado,
            "motivo": mot, "conf_h": conf.astype(np.float32), "util": util, "fps": fps,
            "cuadro_tocado": tf, "sep": (xy[:, 1, 0] - xy[:, 0, 0]).astype(np.float32)}


def pathlib_(trabajo):
    import pathlib
    return pathlib.Path(trabajo) if trabajo else RU.TRABAJO


def escribe(d, trabajo=None):
    """`scout_<clip>.npz`, atomico (.part + os.replace): un corte a mitad no deja un fichero roto."""
    p = pathlib_(trabajo) / ("scout_%s.npz" % d["clip"])
    tmp = p.with_name(p.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **{k: (np.asarray(v) if not isinstance(v, str) else np.array(v))
                                   for k, v in d.items()}, motivos=np.array(MOTIVOS))
    os.replace(tmp, p)
    return p


def x_en(d, t, vent=0):
    """QUE DEVUELVE: la x media de los tiradores con dato en el cuadro t (o la mediana en t +- vent), o None."""
    a, b = max(0, t - vent), min(len(d["x"]), t + vent + 1)
    if a >= b:
        return None
    xs = np.where(d["estado"][a:b] >= 0, d["x"][a:b], np.nan)
    n = np.isfinite(xs).sum(1)
    m = np.nansum(xs, 1) / np.maximum(n, 1)
    m = m[n > 0]
    return float(np.median(m)) if len(m) else None


def resumen(d):
    """Una fila por clip: cobertura, rechazos por motivo, la x y la zona en el tocado, y la fisica."""
    import zona_tocado as ZT
    u = d["util"]
    nu = int(u.sum())
    e = d["estado"]
    ambos = (e[:, 0] >= 0) & (e[:, 1] >= 0)
    x = d["x"]
    ok = e >= 0
    r = {"clip": d["clip"], "T": len(x), "n_utiles": nu,
         "frac_ambos": round(float(ambos[u].mean()), 4) if nu else None,
         "frac_alguno": round(float((ok[:, 0] | ok[:, 1])[u].mean()), 4) if nu else None,
         "frac_interp": round(float((e == ESTADO_INTERP)[u].mean()), 4) if nu else None}
    for i, m in enumerate(MOTIVOS):
        r["n_" + m] = int(((d["motivo"] == i) & u[:, None]).sum())
    tf = d["cuadro_tocado"]
    for nom, v in (("x_tocado_m", 0), ("x_tocado_vent5_m", 5)):
        xt = x_en(d, tf, v) if 0 <= tf < len(x) else None
        r[nom] = None if xt is None else round(xt, 3)
    r["zona"] = ZT.ZONAS[ZT.zona(r["x_tocado_m"])] if r["x_tocado_m"] is not None else None
    r["zona_vent5"] = ZT.ZONAS[ZT.zona(r["x_tocado_vent5_m"])] if r["x_tocado_vent5_m"] is not None else None
    xv = x[ok]
    sep = d["sep"][ambos]
    v = np.abs(d["vel"][np.isfinite(d["vel"])])
    r.update(frac_x_en_pista=round(float((np.abs(xv) <= X_PISTA).mean()), 4) if len(xv) else None,
             frac_sep_ok=round(float(((sep >= 0.5) & (sep <= 14)).mean()), 4) if len(sep) else None,
             sep_p50=round(float(np.median(sep)), 3) if len(sep) else None,
             vel_p95=round(float(np.percentile(v, 95)), 3) if len(v) else None,
             frac_vel_ok=round(float((v <= 6.0).mean()), 4) if len(v) else None)
    return r


def corre(clip, trabajo=None):
    d = posiciones(clip, trabajo)
    if d is None:
        return {"clip": clip, "motivo": "sin pista_h_"}
    escribe(d, trabajo)
    r = resumen(d)
    r["motivo"] = "ok"
    return r


if __name__ == "__main__":
    for c in sys.argv[1:]:
        print(corre(c))

"""Auditoria PROFUNDA de las poses, con detectores DISTINTOS a los geometricos de `aislar.py`.

POR QUE OTRA AUDITORIA: `aislar.py` mira si los dos elegidos SON tiradores de esta pista. Esto mira si
la SERIE TEMPORAL de cada uno es utilizable para entrenar. Son preguntas distintas y un clip puede
pasar la primera y fallar la segunda: dos tiradores correctos con la identidad intercambiada a mitad
del clip envenenan el entrenamiento mas que un espectador, porque la etiqueta depende del LADO.

CINCO DETECTORES, ninguno de ellos usado en la auditoria anterior:

  1. CAMBIO DE IDENTIDAD. El signo de (x_derecho - x_izquierdo) tiene que ser constante: los tiradores
     no se atraviesan. Un cambio de signo sostenido es un intercambio de identidad, y es el fallo mas
     grave posible porque invierte la etiqueta.
  2. TELETRANSPORTES. Con la escala del clip, un humano no pasa de ~10 m/s. Un salto mayor entre
     cuadros consecutivos es la caja saltando a OTRA persona.
  3. SEGUNDOS PERDIDOS. La tirada continua mas larga sin puntos utiles, EN SEGUNDOS, que es lo que
     importa: perder 0,1 s se interpola y perder 1,5 s no.
  4. HUESOS INESTABLES. La distancia hombro-cadera de una misma persona es constante en un clip. Si su
     coeficiente de variacion es alto, el esqueleto esta saltando entre cuerpos distintos.
  5. UN SOLO TIRADOR. Cuadros donde solo hay uno de los dos.

NINGUNO DE ESTOS NUMEROS BORRA CLIPS. Salen como columnas, porque quien entrena tiene que poder
elegir su umbral y porque un umbral puesto aqui seria un umbral inventado.
"""
from __future__ import annotations

import numpy as np

L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12
L_ANK, R_ANK = 15, 16
CONF_MIN = 0.3
V_MAX_MS = 10.0
ALTURA_M, FRAC_CAJA = 1.75, 0.388
CLAVE = [5, 6, 7, 8, 9, 10, 11, 12]
"""Hombros, codos, munecas y caderas: los puntos que usan las medidas de prioridad. Los tobillos se
pierden mucho mas por oclusion y no deben decidir si un cuadro es utilizable."""


def _runs(mask):
    """Longitud de la tirada continua mas larga de True."""
    mejor = act = 0
    for v in mask:
        act = act + 1 if v else 0
        mejor = max(mejor, act)
    return mejor


def audita(poses, boxes, fps: float, alto_img: int) -> dict:
    """QUE HACE: los cinco detectores sobre un clip. `NaN` = indefinido, nunca cero."""
    P = np.asarray(poses, dtype="float64")
    B = np.asarray(boxes, dtype="float64")
    vacio = {"medible": False, "cambios_identidad": float("nan"), "frac_lado_invertido": float("nan"),
             "teletransportes": float("nan"), "v_max_ms": float("nan"),
             "segundos_perdidos_max": float("nan"), "frac_util": float("nan"),
             "cv_hueso_max": float("nan"), "frac_un_solo": float("nan"),
             "grave": False, "motivos": ""}
    if P.ndim != 4 or P.shape[1] != 2 or not fps or fps <= 0:
        return vacio
    T = P.shape[0]
    if T < 6:
        return vacio

    conf = P[:, :, :, 2]
    util = (conf[:, :, CLAVE] >= CONF_MIN).sum(axis=2) >= 5
    frac_util = float(util.all(axis=1).mean())
    frac_un_solo = float((util.sum(axis=1) == 1).mean())
    perdido = ~util.all(axis=1)
    seg_perdidos = _runs(perdido) / float(fps)

    """1. Identidad: el signo de la separacion en x."""
    cx = 0.5 * (B[:, :, 0] + B[:, :, 2])
    ok_b = np.isfinite(cx).all(axis=1)
    dif = np.where(ok_b, cx[:, 1] - cx[:, 0], np.nan)
    signo = np.sign(dif[np.isfinite(dif)])
    if len(signo) >= 4:
        dominante = 1.0 if signo.sum() >= 0 else -1.0
        frac_inv = float(np.mean(signo != dominante))
        cambios = int(np.sum(np.abs(np.diff(signo)) > 1))
    else:
        frac_inv, cambios = float("nan"), 0

    """2. Teletransportes: velocidad del torso en metros por segundo."""
    ppm = (alto_img * FRAC_CAJA) / ALTURA_M if alto_img else float("nan")
    torso = np.where(conf[:, :, [L_SHO, R_SHO, L_HIP, R_HIP]] >= CONF_MIN,
                     P[:, :, [L_SHO, R_SHO, L_HIP, R_HIP], :2].mean(axis=2)[:, :, None, :].repeat(4, 2)[..., 0] * 0 + 1,
                     np.nan)
    ctr = np.full((T, 2, 2), np.nan)
    for s in (0, 1):
        m = conf[:, s, [L_SHO, R_SHO, L_HIP, R_HIP]] >= CONF_MIN
        pts = P[:, s][:, [L_SHO, R_SHO, L_HIP, R_HIP], :2]
        pts = np.where(m[:, :, None], pts, np.nan)
        with np.errstate(invalid="ignore"):
            ctr[:, s] = np.nanmean(pts, axis=1)
    vmax, tele = float("nan"), 0
    if ppm == ppm and ppm > 0:
        dpx = np.linalg.norm(np.diff(ctr, axis=0), axis=2) * alto_img
        v = dpx / ppm * float(fps)
        v = v[np.isfinite(v)]
        if len(v):
            vmax = float(np.nanmax(v))
            tele = int(np.sum(v > V_MAX_MS))

    """4. Huesos: hombro-cadera del mismo lado, que no cambia en un clip."""
    cv_max = float("nan")
    cvs = []
    for s in (0, 1):
        for a, b in ((L_SHO, L_HIP), (R_SHO, R_HIP)):
            m = (conf[:, s, a] >= CONF_MIN) & (conf[:, s, b] >= CONF_MIN)
            if m.sum() < 6:
                continue
            d = np.linalg.norm(P[m, s, a, :2] - P[m, s, b, :2], axis=1)
            d = d[d > 1e-6]
            if len(d) >= 6 and d.mean() > 0:
                cvs.append(float(d.std() / d.mean()))
    if cvs:
        cv_max = float(max(cvs))

    motivos = []
    if cambios and cambios >= 2 and frac_inv == frac_inv and frac_inv > 0.05:
        motivos.append("cambio de identidad: la etiqueta de lado se invierte")
    if tele and tele >= 3:
        motivos.append("teletransportes: la caja salta a otra persona")
    if seg_perdidos >= 1.0:
        motivos.append("mas de 1 s seguido sin pose util")
    if cv_max == cv_max and cv_max > 0.35:
        motivos.append("huesos inestables: el esqueleto salta entre cuerpos")

    return {"medible": True, "cambios_identidad": int(cambios),
            "frac_lado_invertido": round(frac_inv, 4) if frac_inv == frac_inv else float("nan"),
            "teletransportes": int(tele), "v_max_ms": round(vmax, 2) if vmax == vmax else float("nan"),
            "segundos_perdidos_max": round(seg_perdidos, 3),
            "frac_util": round(frac_util, 4), "frac_un_solo": round(frac_un_solo, 4),
            "cv_hueso_max": round(cv_max, 4) if cv_max == cv_max else float("nan"),
            "grave": bool(motivos), "motivos": " | ".join(motivos)}


def resumen(filas) -> dict:
    """QUE HACE: agrega y ordena por gravedad, distinguiendo lo que envenena de lo que solo molesta."""
    import pandas as pd

    d = pd.DataFrame(list(filas))
    if not len(d):
        return {"n": 0}
    med = d[d.medible] if "medible" in d.columns else d
    print("=" * 82)
    print("CALIDAD DE LAS SERIES DE POSE")
    print("  clips ............ %d   medibles ... %d" % (len(d), len(med)))
    if not len(med):
        return {"n": len(d), "medibles": 0}
    print()
    print("  LO QUE ENVENENA EL ENTRENAMIENTO")
    print("    cambio de identidad (>=2 y >5%% invertido) ... %6.2f %%" % (
        100 * float(((med.cambios_identidad >= 2) & (med.frac_lado_invertido > 0.05)).mean())))
    print("    teletransportes (>=3 saltos) ................ %6.2f %%" % (
        100 * float((med.teletransportes >= 3).mean())))
    print("    huesos inestables (cv > 0,35) ............... %6.2f %%" % (
        100 * float((med.cv_hueso_max > 0.35).mean())))
    print()
    print("  LO QUE SOLO REDUCE DATOS")
    print("    mas de 1 s seguido sin pose ................. %6.2f %%" % (
        100 * float((med.segundos_perdidos_max >= 1.0).mean())))
    print("    mas de 2 s seguidos sin pose ................ %6.2f %%" % (
        100 * float((med.segundos_perdidos_max >= 2.0).mean())))
    print("    fraccion util mediana ....................... %6.3f" % float(med.frac_util.median()))
    print("    un solo tirador, fraccion mediana ........... %6.3f" % float(med.frac_un_solo.median()))
    print()
    print("  GRAVES por cualquiera ......................... %6.2f %%" % (100 * float(med.grave.mean())))
    print("  Ningun umbral de aqui borra clips: son columnas para que quien entrene elija.")
    print("=" * 82)
    return {"n": len(d), "medibles": len(med), "graves": float(med.grave.mean())}

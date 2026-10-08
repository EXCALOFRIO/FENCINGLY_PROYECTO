"""Aislamiento de los dos tiradores: cuatro detectores de contaminacion, medidos sobre 10.618 clips.

POR QUE EXISTE: el proyecto ya evita al arbitro y al publico (`pose.score_tracks` con la banda
(0,45, 0,97), `pose._whiteness` sobre el torso porque el uniforme es blanco, y penalizacion por
quietud). Lo que NO habia era una CIFRA de cuanto se cuela. Un mecanismo no es una validacion.

MEDIDO EL 2026-09-09 sobre 10.618 clips de 21 sedes:

  desnivel de pies > 0,12 ............................ 1.117   10,52 %
  pies fuera de la banda en > 50 % de cuadros ........... 211    1,99 %
  razon de alturas > 1,6 ................................ 105    0,99 %
  separacion en x < 0,04 (serian la misma persona) ....... 14    0,13 %
  SOSPECHOSOS por cualquiera .......................... 1.236   11,64 %
  LIMPIOS por los cuatro .............................. 9.382   88,36 %

DOS COSAS QUE ESA MEDIDA DESCUBRIO:
  1. La banda (0,45, 0,97) NO es una puerta dura, es una penalizacion de puntuacion. Si fuera dura,
     el 1,99 % con los pies fuera en mas de la mitad de los cuadros tendria que ser CERO.
  2. El desnivel de pies NO es prueba de contaminacion por si solo: la perspectiva legitima lo
     produce, porque un tirador en el extremo lejano de LA MISMA pista tiene los pies mas arriba.
     Asi que el 10,52 % es una COTA SUPERIOR DE SOSPECHA, no una tasa de contaminacion.

QUE DETECTOR CREER: `razon_alturas` es el preciso. Inspeccionados sus dos peores casos, los dos eran
contaminacion genuina y sin ambiguedad: en uno las dos cajas estaban sobre un espectador de la grada y
sobre suelo vacio mientras el tirador real quedaba sin marcar; en el otro una caja estaba sobre una
persona de oscuro junto a la mesa de oficiales, al fondo. A la misma profundidad dos tiradores no
difieren un 60 % de tamano, y por eso este detector no lo confunde la perspectiva.

ESTO NO ES UNA PUERTA QUE BORRE CLIPS. Devuelve columnas y una bandera para que el consumidor decida.
Descartar el 11,64 % del corpus por una cota superior seria peor que la contaminacion.
"""
from __future__ import annotations

import numpy as np

L_ANK, R_ANK = 15, 16
BANDA = (0.45, 0.97)
CONF_MIN = 0.3

UMBRAL_DESNIVEL = 0.12
UMBRAL_RAZON = 1.6
UMBRAL_SEP_X = 0.04
UMBRAL_FUERA = 0.5

TASAS_CORPUS = {"desnivel": 0.1052, "razon": 0.0099, "banda": 0.0199, "sep_x": 0.0013,
                "sospechoso": 0.1164, "n": 10618, "sedes": 21}
"""Tasas base medidas. Sirven para saber si un lote nuevo se sale de lo normal."""


def audita(poses: np.ndarray, boxes: np.ndarray) -> dict:
    """QUE HACE: aplica los cuatro detectores a un clip y devuelve sus valores y las banderas.

    `poses` (T, 2, 17, 3) normalizado a imagen. `boxes` (T, 2, 4) normalizado.

    QUE DEVUELVE: los cuatro numeros, sus banderas, y `n_ok` = cuadros donde se pudo medir. Si
    `n_ok < 5` devuelve todo indefinido en vez de un numero de un par de cuadros: `NaN` significa
    INDEFINIDO, nunca cero.
    """
    P = np.asarray(poses, dtype="float64")
    B = np.asarray(boxes, dtype="float64")
    vacio = {"n_ok": 0, "desnivel_pies": float("nan"), "razon_alturas": float("nan"),
             "sep_x": float("nan"), "fuera_banda_max": float("nan"),
             "susp_desnivel": False, "susp_razon": False, "susp_banda": False,
             "susp_sepx": False, "sospechoso": False, "medible": False}
    if P.ndim != 4 or P.shape[1] != 2 or B.ndim != 3:
        return vacio

    conf = P[:, :, [L_ANK, R_ANK], 2]
    ay = np.where(conf >= CONF_MIN, P[:, :, [L_ANK, R_ANK], 1], np.nan)
    with np.errstate(invalid="ignore"):
        pie = np.nanmax(ay, axis=2)
    alto = B[:, :, 3] - B[:, :, 1]
    cx = 0.5 * (B[:, :, 0] + B[:, :, 2])

    ok = np.isfinite(pie).all(axis=1) & (alto > 0).all(axis=1)
    if int(ok.sum()) < 5:
        return vacio
    pie_ok, alto_ok, cx_ok = pie[ok], alto[ok], cx[ok]

    desnivel = float(np.median(np.abs(pie_ok[:, 0] - pie_ok[:, 1])))
    razon = float(np.median(np.max(alto_ok, axis=1) / np.maximum(np.min(alto_ok, axis=1), 1e-9)))
    sep_x = float(np.median(np.abs(cx_ok[:, 0] - cx_ok[:, 1])))
    fuera = max(float(np.mean((pie_ok[:, s] < BANDA[0]) | (pie_ok[:, s] > BANDA[1])))
                for s in (0, 1))

    sd = desnivel > UMBRAL_DESNIVEL
    sr = razon > UMBRAL_RAZON
    sb = fuera > UMBRAL_FUERA
    sx = sep_x < UMBRAL_SEP_X
    return {"n_ok": int(ok.sum()), "desnivel_pies": round(desnivel, 4),
            "razon_alturas": round(razon, 3), "sep_x": round(sep_x, 4),
            "fuera_banda_max": round(fuera, 3),
            "susp_desnivel": bool(sd), "susp_razon": bool(sr), "susp_banda": bool(sb),
            "susp_sepx": bool(sx), "sospechoso": bool(sd or sr or sb or sx), "medible": True}


def motivo(fila: dict) -> str:
    """QUE HACE: nombra por que un clip es sospechoso, y con que fuerza. Sin esto la bandera no dice
    nada utilizable: `razon` es precisa y `desnivel` es una cota superior confundida con perspectiva."""
    if not fila.get("sospechoso"):
        return "limpio"
    partes = []
    if fila.get("susp_sepx"):
        partes.append("los dos elegidos se solapan: probablemente la MISMA persona")
    if fila.get("susp_razon"):
        partes.append("razon de alturas alta: detector PRECISO, casi seguro otra pista o un oficial")
    if fila.get("susp_banda"):
        partes.append("pies fuera de la banda activa")
    if fila.get("susp_desnivel"):
        partes.append("desnivel de pies alto: SOSPECHA, tambien lo produce la perspectiva legitima")
    return " | ".join(partes)


def resumen(filas) -> dict:
    """QUE HACE: agrega un lote y lo compara contra las tasas base del corpus.

    Sirve para lo que pedia el usuario: saber si un lote nuevo esta peor de lo normal, en vez de
    mirar un porcentaje suelto sin referencia.
    """
    import pandas as pd

    d = pd.DataFrame(list(filas))
    if not len(d):
        return {"n": 0}
    med = d[d.get("medible", True)] if "medible" in d.columns else d
    fuera = {"n": len(d), "medibles": int(len(med))}
    print("=" * 78)
    print("AISLAMIENTO DE TIRADORES")
    print("  clips ............ %d   medibles ... %d" % (len(d), len(med)))
    for col, etq, base in (("susp_desnivel", "desnivel de pies", TASAS_CORPUS["desnivel"]),
                           ("susp_banda", "pies fuera de banda", TASAS_CORPUS["banda"]),
                           ("susp_razon", "razon de alturas (PRECISO)", TASAS_CORPUS["razon"]),
                           ("susp_sepx", "misma persona dos veces", TASAS_CORPUS["sep_x"]),
                           ("sospechoso", "SOSPECHOSOS en total", TASAS_CORPUS["sospechoso"])):
        if col not in med.columns or not len(med):
            continue
        tasa = float(med[col].mean())
        fuera[col] = tasa
        print("  %-28s %6.2f %%   base del corpus %5.2f %%   %s" % (
            etq, 100 * tasa, 100 * base,
            "<- PEOR que el corpus" if tasa > base * 1.5 else ""))
    print("  Recordatorio: `desnivel` es una COTA SUPERIOR de sospecha, no contaminacion medida.")
    print("  Y esto NO borra clips: son columnas para que el consumidor decida.")
    print("=" * 78)
    return fuera

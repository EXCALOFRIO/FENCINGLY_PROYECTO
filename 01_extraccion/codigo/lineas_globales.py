"""Los puntos de fuga sacados de TODAS las lineas del pabellon, no del barrido local junto a los pies.

## LA MEDIDA QUE JUSTIFICA ESTE MODULO

El usuario lo dijo asi: *"seguro que se puede, es facil, si hay muchas lineas de referencia en los
asaltos"*. Tiene razon y la cifra es grande. Detectando segmentos con LSD sobre la placa limpia
entera y buscando puntos de fuga por RANSAC:

    clip                                segm.largos   CONCURREN en un VP   lo que usa HOY
    AbandonedAbsoluteGoldfinch                   85                   57   4 marcas, 8 verticales
    AbandonedBiodegradableCoqui                  93                   78   8 marcas, 8 verticales
    AbandonedAntiqueGalapagostortoise            44                   27   2 marcas, 8 verticales
    AbandonedConsiderateGraysquirrel             44                   18   4 marcas, 3 verticales

**Entre 10 y 20 veces mas evidencia de la que se esta usando.** Y el caso que lo remata:
`AbandonedAbsoluteGoldfinch` **falla hoy** con *"sin punto de fuga vertical: las verticales no
concurren"*, y ahi hay 57 lineas que si concurren.

## POR QUE SE ESTABA PERDIENDO

`region_y_bordes` y `verticales` barren **solo alrededor de los apoyos de los tiradores**, hasta
`ALCANCE_BARRIDO = 2,2` altos de caja. Esa constante esta bien elegida para lo suyo -- y su
docstring cuenta el fallo medido que la puso ahi -- pero tiene un efecto colateral que nadie midio:
**todo el resto del pabellon se descarta antes de mirarlo**.

Y el resto del pabellon no es ruido. Una sala de esgrima esta construida alineada con las pistas:
las vallas, los paneles de publicidad, el borde de la tarima, las pistas contiguas y las juntas del
suelo son **paralelas al eje de la pista**, y los postes, las patas de las mesas y la estructura del
techo son **verticales**. O sea que el encuadre esta lleno de rectas que votan por los mismos dos
puntos de fuga que hacen falta, y se estaban tirando.

## LO QUE ESTE MODULO NO DECIDE

Que el haz mayoritario sea el de la PISTA. Podria ser el de la grada. Por eso `elige_v1` no se queda
con el que mas inliers tiene: se queda con el compatible con **la direccion por la que caminan los
tiradores**, que es un hecho fisico -- solo pueden andar a lo largo de la pista -- y que el modulo
`pista` ya calcula en `corredor`. El recuento de inliers solo desempata dentro de los compatibles.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

MIN_LARGO_FRAC = 0.06
"""Largo minimo de un segmento para votar, en fracciones del ancho de imagen. En 1280 px son 77 px.
Por debajo entran texturas y letras de los paneles, que no son estructura del pabellon."""

TOL_VP_DEG = 1.2
"""Igual que `pista.ANG_INLIER_V2`, a proposito: si se cambiara, la comparacion entre este camino y
el de hoy mediria dos cosas."""

ITERS = 3000
MIN_INLIERS = 8


def segmentos(img, min_largo_frac=MIN_LARGO_FRAC, mascara=None):
    """QUE DEVUELVE: array (n, 4) de segmentos largos `x0,y0,x1,y1`, o (0, 4).

    `mascara` marca lo que NO debe votar (los rotulos de television, que son rectangulos con bordes
    rectos y perfectamente horizontales y verticales en el sistema de PANTALLA, no del mundo). Sin
    quitarlos, el marcador mete un haz falso que compite con el bueno.
    """
    if cv2 is None or img is None:
        return np.zeros((0, 4))
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    seg = cv2.createLineSegmentDetector().detect(g)[0]
    if seg is None:
        return np.zeros((0, 4))
    S = seg.reshape(-1, 4).astype("float64")
    largo = np.hypot(S[:, 2] - S[:, 0], S[:, 3] - S[:, 1])
    S = S[largo >= min_largo_frac * g.shape[1]]
    if mascara is not None and len(S):
        cm = np.stack([(S[:, 0] + S[:, 2]) / 2, (S[:, 1] + S[:, 3]) / 2], 1).astype(int)
        cm[:, 0] = np.clip(cm[:, 0], 0, mascara.shape[1] - 1)
        cm[:, 1] = np.clip(cm[:, 1], 0, mascara.shape[0] - 1)
        S = S[mascara[cm[:, 1], cm[:, 0]] == 0]
    return S


def _rectas_y_dirs(S):
    L = np.stack([np.cross([a, b, 1.0], [c, d, 1.0]) for a, b, c, d in S]) if len(S) \
        else np.zeros((0, 3))
    C = np.stack([(S[:, 0] + S[:, 2]) / 2, (S[:, 1] + S[:, 3]) / 2], 1) if len(S) \
        else np.zeros((0, 2))
    D = np.stack([S[:, 2] - S[:, 0], S[:, 3] - S[:, 1]], 1) if len(S) else np.zeros((0, 2))
    if len(D):
        D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    return L, C, D


def _inliers(P, C, D, tol_deg):
    """Un segmento vota por `P` si la recta que une su CENTRO con `P` es paralela al segmento. Se
    mide asi y no por distancia punto-recta porque la distancia crece con la lejania del VP y haria
    que los VP lejanos ganaran siempre."""
    d = P[None, :] - C
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(np.abs((d * D).sum(1)), 0.0, 1.0)))
    return ang <= tol_deg


def puntos_de_fuga(S, n_max=3, tol_deg=TOL_VP_DEG, iters=ITERS, semilla=0):
    """QUE DEVUELVE: lista de dicts `{P, n, inl}` ordenada por numero de votos, extraidos por RANSAC
    secuencial: se saca el haz dominante, se quitan sus segmentos y se repite."""
    S = np.asarray(S, "float64")
    if len(S) < 4:
        return []
    L, C, D = _rectas_y_dirs(S)
    libre = np.ones(len(S), bool)
    rng = np.random.default_rng(semilla)
    out = []
    for _ in range(n_max):
        idx = np.nonzero(libre)[0]
        if len(idx) < 4:
            break
        mejor = (0, None, None)
        for _it in range(iters):
            i, j = rng.choice(idx, 2, replace=False)
            v = np.cross(L[i], L[j])
            if abs(v[2]) < 1e-9:
                continue
            P = v[:2] / v[2]
            if not np.isfinite(P).all():
                continue
            inl = _inliers(P, C, D, tol_deg)
            inl = inl & libre
            if inl.sum() > mejor[0]:
                mejor = (int(inl.sum()), P, inl)
        if mejor[0] < MIN_INLIERS:
            break
        out.append({"P": mejor[1], "n": mejor[0], "inl": mejor[2]})
        libre = libre & ~mejor[2]
    return out


def elige_v1(vps, dir_pies, C=None, tol_deg=8.0):
    """QUE DEVUELVE: el dict del punto de fuga compatible con la direccion por la que caminan los
    tiradores, o None.

    NO SE COGE EL QUE MAS VOTOS TIENE. En una sala llena de estructura, el haz mayoritario puede ser
    el de la grada o el del techo. Lo que ata el haz a la PISTA es un hecho fisico y no un recuento:
    los tiradores solo pueden andar a lo largo de la pista, asi que la direccion de su corredor --
    que `pista.corredor` ya calcula -- tiene que estar en el haz. Entre los compatibles, entonces si,
    manda el numero de votos, porque a igual compatibilidad mas evidencia es mejor.
    """
    if not vps or dir_pies is None:
        return None
    e = np.asarray(dir_pies, "float64")[:2]
    nn = np.linalg.norm(e)
    if nn < 1e-9:
        return None
    e = e / nn
    cen = np.asarray(C, "float64") if C is not None else None
    comp = []
    for v in vps:
        P = np.asarray(v["P"], "float64")
        ref = cen if cen is not None else np.zeros(2)
        d = P - ref
        n2 = np.linalg.norm(d)
        if n2 < 1e-9:
            continue
        ang = np.degrees(np.arccos(np.clip(abs(float(d / n2 @ e)), 0.0, 1.0)))
        if ang <= tol_deg:
            comp.append((v["n"], ang, v))
    if not comp:
        return None
    comp.sort(key=lambda t: (-t[0], t[1]))
    v = dict(comp[0][2])
    v["ang_con_pies_deg"] = float(comp[0][1])
    v["n_compatibles"] = len(comp)
    return v

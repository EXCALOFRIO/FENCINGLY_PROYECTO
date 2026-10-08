"""Los desplazamientos medidos en ALTOS DE CAJA, para los clips que no tienen homografia.

## EL PROBLEMA, MEDIDO

De 698 clips con la etapa 7 corrida, **333 (47,7 %) no tienen homografia**. Y en esos se detectan:

    marchas por clip   0,00        roturas   0,00        fondos   0,00

Cero. No porque el tirador no se mueva, sino porque `_pasos` mide en **metros** y sin homografia
no hay metros. Con homografia salen 7,90 marchas y 7,52 roturas por clip. O sea que **la mitad del
corpus no tiene ni un desplazamiento detectado**, y ese cero es indistinguible de "no se movio".

El `brazo`, en cambio, se detecta en el **96,7 %** de los clips. La diferencia no es la dificultad
del evento: es que el brazo se mide en **fraccion del torso del propio tirador**, que es una
reglilla de imagen y no necesita homografia.

## LA OBJECION QUE HABIA ESCRITA, Y POR QUE NO SE APLICA AQUI

`posicion.eventos_sin_homografia` dice, textualmente, que marcha/romper/fondo *"no se pueden
trasladar a altos de caja sin suponer que la relacion pixel-metro es la misma en todo el cuadro,
y no lo es: la pista se aleja de la camara"*.

**Es cierto del normalizador que ese codigo usa, y no del correcto.** Ahi `alto` es
`np.nanmedian` de las alturas de caja de TODO el clip: una constante global. Con una constante
global la objecion es exacta -- un metro cerca de la camara son mas pixeles que lejos, y dividir
por el mismo numero en los dos sitios mezcla perspectiva con movimiento.

Pero si se divide por **la caja de ESE tirador en ESE cuadro**, la perspectiva se cancela sola:
cuando el tirador se aleja, su paso mide menos pixeles **y su caja tambien**. Es exactamente el
argumento por el que `ext_torso` funciona, y por el que el brazo se detecta en el 96,7 %.

Asi que la objecion no se descarta: **se corrige el normalizador y se comprueba midiendo**.

## COMO SE COMPRUEBA, Y EL CRITERIO DECLARADO ANTES

Los 365 clips que **si** tienen homografia dan las dos medidas del mismo paso: en metros y en
altos de caja. Eso es un careo de doble filo y es lo unico que autoriza a usar la segunda donde
no hay primera.

    se adopta si   recupera >= 90 % de los pasos que encuentra el detector metrico
                   y la correlacion de las magnitudes es r >= 0,90
    se rechaza si  no, **diciendo cuanto**, en vez de adoptarse "con una advertencia"

Y si se adopta, la razon metros/altos-de-caja medida es ademas una **calibracion**: un alto de
caja de un tirador en guardia deberia salir del orden de 1,4-1,5 m, y que caiga ahi es una
comprobacion barata de que la reglilla es la que se cree.

## LO QUE ESTO NO ES

**No convierte a metros.** La magnitud sale en altos de caja y la columna `unidad_magnitud` lo
dice, igual que ya lo dice para el brazo. Poner metros multiplicando por la razon media seria
inventar una escala que este proyecto lleva tres vias intentando medir y no ha conseguido.
"""
from __future__ import annotations

import numpy as np

PASO_V_CAJA = 0.85
"""Umbral de velocidad de un pie para contar como zancada, en ALTOS DE CAJA POR SEGUNDO.

De donde sale: el umbral metrico es `PASO_V_MS = 1,25 m/s`, y un alto de caja de un floretista en
guardia son del orden de 1,45 m. 1,25 / 1,45 = 0,86. Se deja en **0,85**, o sea el mismo criterio
trasladado, no uno nuevo ajustado a ojo -- si hiciera falta moverlo, eso ya seria otra decision y
tendria que declararse aparte."""

PASO_MIN_CAJA = 0.17
"""Avance minimo para contar como paso, en altos de caja. El metrico es `PASO_MIN_M = 0,25 m`;
0,25 / 1,45 = 0,172."""

PASO_MIN_CUADROS = 2
PASO_HUECO = 12
"""Los mismos que el detector metrico, porque son de tiempo y el tiempo no cambia de unidades."""


def _suave(a, k=5):
    """Media movil que ignora NaN dividiendo por la cuenta de validos."""
    a = np.asarray(a, dtype="float64")
    if len(a) < k:
        return a
    ok = np.isfinite(a).astype("float64")
    ker = np.ones(k)
    s = np.convolve(np.where(np.isfinite(a), a, 0.0), ker, mode="same")
    n = np.convolve(ok, ker, mode="same")
    with np.errstate(all="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1e-9), np.nan)


def _vel(x, fps):
    v = np.full(len(x), np.nan)
    v[1:-1] = (x[2:] - x[:-2]) * fps / 2.0
    return v


def series_caja(res, poses, fps):
    """QUE DEVUELVE: (F, C, alto) donde `F[k, lado, pie]` es la x de cada pie y `C[k, lado]` la del
    centro, **todas en altos de la caja de ESE tirador en ESE cuadro**, y ya estabilizadas con
    `Mk` para quitar el movimiento de camara.

    LOS DOS NORMALIZADORES QUE HAY QUE APLICAR, Y EN ESTE ORDEN:

      1. `Mk[k]`, la homografia de ESTABILIZACION entre cuadros -- que **no** es la de la pista y
         existe siempre. Sin ella, una panoramica de camara se leeria como que los dos tiradores
         avanzan a la vez.
      2. el alto de caja **del tirador y del cuadro**, no una mediana del clip. Es lo que cancela
         la perspectiva: al alejarse, el paso mide menos pixeles y la caja tambien.
    """
    import posicion as PO
    r, Mk = res.get("_r"), res.get("_Mk")
    if r is None or Mk is None or poses is None:
        return None, None, None
    B, pres, mask = r["B"], r["pres"], r["mask"]
    W, H, T = r["W"], r["H"], r["T"]
    KP, SC = poses["KP"], poses["SC"]
    if KP.shape[0] < T:
        return None, None, None
    F = np.full((T, 2, 2), np.nan)
    C = np.full((T, 2), np.nan)
    A = np.full((T, 2), np.nan)
    for k in np.nonzero(mask)[0]:
        M = Mk[k]
        for lado in (0, 1):
            if not pres[k, lado] or not np.isfinite(B[k, lado]).all():
                continue
            alto_px = float(B[k, lado, 3] - B[k, lado, 1]) * H
            if not np.isfinite(alto_px) or alto_px < 20.0:
                continue
            A[k, lado] = alto_px
            pies = PO._pie_pose(KP, SC, k, lado)
            xs = []
            for p in pies:
                if p is None:
                    xs.append(np.nan)
                    continue
                q = M @ np.array([float(p[0]), float(p[1]), 1.0])
                xs.append((q[0] / q[2]) / alto_px)
            F[k, lado] = xs
            if np.isfinite(xs).all():
                C[k, lado] = 0.5 * (xs[0] + xs[1])
            else:
                q = M @ np.array([0.5 * (B[k, lado, 0] + B[k, lado, 2]) * W,
                                  B[k, lado, 3] * H, 1.0])
                C[k, lado] = (q[0] / q[2]) / alto_px
    return F, C, A


def _zancadas(v, otro, ks, umb):
    """Los tramos en que ESTE pie se mueve y el otro no. Mismo criterio e histeresis que el
    detector metrico; solo cambia la unidad del umbral."""
    out = []
    i = ks[0]
    while i <= ks[-1]:
        ok = np.isfinite(v[i]) and abs(v[i]) > umb and \
            (not np.isfinite(otro[i]) or abs(otro[i]) < 0.5 * umb)
        if not ok:
            i += 1
            continue
        j = i
        while j + 1 <= ks[-1]:
            sig = np.isfinite(v[j + 1]) and abs(v[j + 1]) > 0.5 * umb and \
                (not np.isfinite(otro[j + 1]) or abs(otro[j + 1]) < umb)
            if not sig:
                break
            j += 1
        if j - i + 1 >= PASO_MIN_CUADROS:
            out.append((i, j, int(np.sign(np.nanmean(v[i:j + 1])))))
        i = j + 1
    return out


def pasos(F, C, lado, haciarival, ks, fps):
    """QUE DEVUELVE: los desplazamientos de un tirador, en altos de caja, con **cuatro campos que
    el detector metrico no da**.

    `pie_primero`   `delante` o `detras`: cual de los dos pies arranco el paso. En esgrima una
                    marcha es **pie de delante y luego el de atras**, y una rotura al reves; que
                    salga al reves es un cruce, un fallo de deteccion o una recuperacion de fondo,
                    y hoy no hay forma de distinguirlo porque el dato se calcula y se tira.
    `completa`      los DOS pies se movieron **y** la separacion de guardia se recupera al final.
                    Medio paso es un ajuste de distancia, y un floretista los hace constantemente.
    `sep_ini/fin`   la separacion de pies al empezar y al acabar, en altos de caja: es lo que
                    permite decir si la guardia se recupero o el tirador se quedo abierto.
    `desde_anterior_s`  el hueco hasta el paso anterior DEL MISMO TIRADOR, que es la **cadencia**.
                    Sin esto solo hay una tasa por clip, y una tasa no distingue tres pasos
                    seguidos y rapidos de tres repartidos.
    """
    x1 = _suave(F[:, lado, 0], 5)
    x2 = _suave(F[:, lado, 1], 5)
    v1, v2 = _vel(x1, fps), _vel(x2, fps)
    z = [(i, j, s, 0) for (i, j, s) in _zancadas(v1, v2, ks, PASO_V_CAJA)] + \
        [(i, j, s, 1) for (i, j, s) in _zancadas(v2, v1, ks, PASO_V_CAJA)]
    z.sort()
    usado = [False] * len(z)
    cs = _suave(C[:, lado], 5)
    out = []
    for a in range(len(z)):
        if usado[a]:
            continue
        i, j, s, pie = z[a]
        usado[a] = True
        fin, npies = j, 1
        for b in range(a + 1, len(z)):
            if usado[b]:
                continue
            i2, j2, s2, pie2 = z[b]
            if i2 - fin > PASO_HUECO:
                break
            if pie2 != pie and s2 == s:
                usado[b] = True
                fin, npies = max(fin, j2), 2
                break
        d = (cs[fin] - cs[i]) * haciarival
        if not np.isfinite(d) or abs(d) < PASO_MIN_CAJA:
            continue
        """CUAL ES EL PIE DE DELANTE: el que tiene mayor x hacia el rival al EMPEZAR el paso. Se
        mira al empezar y no en todo el paso porque durante un cruce los pies se intercambian, y
        lo que se quiere saber es cual arranco."""
        f0 = np.array([F[i, lado, 0], F[i, lado, 1]]) * haciarival
        if np.isfinite(f0).all():
            delante = int(np.argmax(f0))
            primero = "delante" if pie == delante else "detras"
        else:
            primero = ""
        sep_i = abs(F[i, lado, 0] - F[i, lado, 1])
        sep_f = abs(F[fin, lado, 0] - F[fin, lado, 1])
        rec = (np.isfinite(sep_i) and np.isfinite(sep_f)
               and sep_f >= 0.6 * sep_i and sep_f <= 1.6 * sep_i)
        """LA GUARDIA SE DA POR RECUPERADA si la separacion final esta entre 0,6 y 1,6 veces la
        inicial. No es un umbral fino: es la banda dentro de la cual un floretista sigue en
        guardia. Fuera de ella o se quedo abierto (fondo incipiente) o junto los pies."""
        out.append({
            "tipo": "marcha" if d > 0 else "romper", "lado": lado,
            "cuadro": int((i + fin) // 2), "inicio": int(i), "fin": int(fin),
            "magnitud_caja": float(abs(d)),
            "unidad_magnitud": "altos de caja",
            "pies_movidos": int(npies),
            "pie_primero": primero,
            "completa": bool(npies == 2 and rec),
            "sep_ini_caja": float(sep_i) if np.isfinite(sep_i) else None,
            "sep_fin_caja": float(sep_f) if np.isfinite(sep_f) else None,
            "v_caja_s": float(abs(d) * fps / max(fin - i, 1)),
            "duracion_s": float((fin - i + 1) / fps),
        })
    out.sort(key=lambda e: e["inicio"])
    prev = None
    for e in out:
        e["desde_anterior_s"] = (None if prev is None
                                 else round((e["inicio"] - prev) / fps, 3))
        prev = e["inicio"]
    return out


def de_clip(clip, res=None, poses=None, fps=None):
    """QUE DEVUELVE: todos los desplazamientos del clip en altos de caja, o [] si no se puede."""
    import pista as PI
    import posicion as PO
    if res is None:
        res = PI.analiza(clip)
    if not res or "_r" not in res:
        return []
    if poses is None:
        poses = PO.carga_poses(clip)
    if poses is None:
        return []
    if fps is None:
        fps = PO._fps(PI.indice_videos().get(clip) or "")
    F, C, _A = series_caja(res, poses, fps)
    if F is None:
        return []
    mask = res["_r"]["mask"]
    ks = np.nonzero(mask)[0]
    if len(ks) < 12:
        return []
    med = np.nanmedian(C, axis=0)
    if not np.isfinite(med).all():
        return []
    out = []
    for lado in (0, 1):
        haciarival = 1.0 if med[lado] < med[1 - lado] else -1.0
        for e in pasos(F, C, lado, haciarival, ks, fps):
            e["clip"] = clip
            out.append(e)
    return out

"""Detecta FUNDIDOS ENCADENADOS entre planos y devuelve los cuadros contaminados.

QUE AVERIA ARREGLA, y la encontro el usuario mirando el video: en `VainAchingGonolek` las 7 poses
inventadas del principio estaban mal. La causa NO era la interpolacion. Los cuadros 7-18 de ese clip
son un **fundido encadenado**: dos planos superpuestos en la misma imagen, con el marcador y el
rotulo del fondo DUPLICADOS, y YOLO detecta tiradores de los dos planos a la vez. La pareja elegida
sale mezclada, y no habia nada correcto que interpolar.

Es una averia distinta de la repeticion en primer plano que trata `planos.py`, y se le escapa a esa
puerta porque en un fundido no hay ninguna caja enorme: todas son de talla normal.

DOS SENALES, CON PAPELES DISTINTOS. El reparto de papeles es lo que costo acertar:

  DISPARA la MEZCLA LINEAL, sobre los pixeles. Un fundido es `f[k] = a*A + b*B`, asi que su cuadro
  se reconstruye mucho mejor con DOS cuadros, uno de antes y otro de despues, que con uno solo. La
  senal es `r1/r2`, lo que mejora el residuo al admitir el segundo plano.
  Medido: el fundido de `VainAching` llega a **3,23** y el maximo de los otros once clips es 2,41,
  que resulta ser el corte duro de plano de `Harsh` en c109, o sea un corte de verdad. Es una senal
  RARA, y por eso puede disparar.

  AFINAN los NACIMIENTOS Y MUERTES de trayectorias, que dan limites exactos, cosa que la senal de
  pixeles no da porque responde difuminada en +-D cuadros.

EL CONFUNDIDO QUE APARECIO AL VERIFICAR, y es de los importantes: **un ZOOM o un paneo suave da la
misma firma que un fundido**, porque el cuadro de verdad es intermedio entre sus vecinos. En
`VainAching` c42-c48 la senal llega a 2,79 y al mirar c45 no hay ningun fundido: es un zoom lento.

Lo que los separa, medido, es el residuo ABSOLUTO de la mezcla, `r2` partido por el contraste del
cuadro. Y sale al REVES de lo que yo esperaba:

  | tramo de `VainAching` | razon r1/r2 | r2/contraste |
  |---|---:|---:|
  | c7-c18, fundido verificado a ojo | 3,23 | **0,219** |
  | c42-c48, zoom verificado en c45  | 2,79 | **0,062** |

El zoom se explica MEJOR con la mezcla que el fundido, porque durante el fundido los tiradores se
mueven y el plano que entra todavia no es el de f[k+D]. Separan por 3,6x, y por eso se exigen las
dos condiciones.

AVISO HONRADO SOBRE ESTOS DOS UMBRALES: hay **un solo fundido verificado** en todo lo revisado, asi
que las dos cifras estan ajustadas con n=1. No son una ley, son la unica evidencia que hay. Si
aparece un segundo fundido, lo primero es comprobar si los pasa.

LO INTENTE AL REVES Y FUE UN DESASTRE MEDIDO, asi que no lo reintentes: dejando que el relevo de
trayectorias disparara, en un pabellon con gente entrando y saliendo hay nacimientos y muertes
constantes y se encadena el clip entero. Marco el **97,5 %** de `BlissfulTiredAiredale`, el 54,8 %
de `ZanyMeager` y el 65,7 % de `DeficientNippyGar`, los tres falsos.
"""
from __future__ import annotations

import numpy as np

D = 8
"""Distancia a los cuadros de referencia. Tiene que ser mayor que el fundido tipico partido por dos
para que f[k-D] y f[k+D] caigan ya en planos distintos, y menor que el clip."""
UMBRAL_DISPARA = 2.50
"""Senal de mezcla que DISPARA el tramo. Medido: el fundido de `VainAchingGonolek` llega a **3,23** y
el maximo de los otros once clips es 2,41, que es el corte duro de plano de `HarshExemplaryGermanspaniel`
en c109, o sea un corte de verdad y no un falso positivo. Margen 1,29x."""
UMBRAL_R2REL = 0.15
"""Residuo absoluto de la mezcla, partido por el contraste del cuadro. Por debajo de esto el cuadro
se explica DEMASIADO bien con la mezcla y es un zoom, no un fundido. Medido: fundido 0,219, zoom
0,062, y los once clips sin fundido tienen medianas de 0,19 a 0,47, o sea que esta condicion no les
quita nada."""
UMBRAL_SIGUE = 1.90
"""Con histeresis: una vez disparado, el tramo se extiende mientras la senal siga por encima de esto.
El p99 de los once clips sin fundido conocido es 1,77, asi que extender no se desmadra."""
LARGO_MAX = 30
"""Un fundido de television dura menos de un segundo. Si los limites se van mas alla, se desconfia
de los limites y se deja el tramo que disparo la senal."""
VENTANA = 6
"""Cuadros de holgura al emparejar una muerte con un nacimiento."""
MIN_N = 6
"""Trayectorias mas cortas que esto no son evidencia de nada."""


def senal_mezcla(gris, d=D) -> np.ndarray:
    """QUE HACE: por cada cuadro, cuanto mejora reconstruirlo admitiendo un segundo plano.

    `gris` es la lista de cuadros ya reducidos y en gris. Se reduce a 160x90 antes de llamar: la
    senal es de imagen completa y a esa talla cuesta milisegundos.
    """
    T = len(gris)
    v = np.full(T, np.nan, dtype="float64")
    rel = np.full(T, np.nan, dtype="float64")
    for k in range(T):
        a, b = k - d, k + d
        if a < 0 or b >= T:
            continue
        y = gris[k].ravel()
        x1 = gris[a].ravel()
        x2 = gris[b].ravel()
        uno = np.ones_like(x1)
        A1 = np.stack([x1, uno], axis=1)
        c1, *_ = np.linalg.lstsq(A1, y, rcond=None)
        r1 = float(np.sqrt(np.mean((A1 @ c1 - y) ** 2)))
        A2 = np.stack([x1, x2, uno], axis=1)
        c2, *_ = np.linalg.lstsq(A2, y, rcond=None)
        r2 = float(np.sqrt(np.mean((A2 @ c2 - y) ** 2)))
        v[k] = r1 / max(r2, 1e-6)
        rel[k] = r2 / max(float(y.std()), 1e-6)
    return v, rel


def _relevo(trayectorias, T, a, b):
    """QUE HACE: ajusta los limites de un tramo ya disparado, usando nacimientos y muertes.

    NO DISPARA NADA, solo afina. Lo intente al contrario, dejando que el relevo de trayectorias
    disparara, y fue un desastre medido: en un pabellon con gente entrando y saliendo hay
    nacimientos y muertes constantes, y encadenaba el clip entero. Marco el **97,5 %** de
    `BlissfulTiredAiredale`, el 54,8 % de `ZanyMeager` y el 65,7 % de `DeficientNippyGar`, todos
    falsos. La senal de pixeles es la que puede disparar porque es rara: solo la pasa el clip con
    fundido de verdad.
    """
    nace, muere = [], []
    for t in trayectorias:
        ks = np.asarray(t["cuadros"], dtype="int64")
        if len(ks) < MIN_N:
            continue
        k0, k1 = int(ks.min()), int(ks.max())
        if k0 > 1:
            nace.append(k0)
        if k1 < T - 2:
            muere.append(k1)
    na = [k for k in nace if a - VENTANA <= k <= b + VENTANA]
    mu = [k for k in muere if a - VENTANA <= k <= b + VENTANA]
    ini = min(na) if na else a
    fin = max(mu) if mu else b
    if fin - ini + 1 > LARGO_MAX:
        """Si los extremos se van, se deja el tramo que disparo la senal y se dice."""
        return a, b, False
    return min(ini, a), max(fin, b), True


def marca(gris, trayectorias, T) -> dict:
    """QUE HACE: devuelve la mascara de cuadros libres de fundido y el detalle de lo que quita.

    La senal de pixeles dispara, la histeresis extiende, y los nacimientos y muertes de trayectorias
    solo afinan los limites. Los tres pasos estan separados a proposito para poder ver cual de ellos
    se equivoca cuando algo sale mal.
    """
    v, rel = senal_mezcla(gris)
    fin_v = np.nan_to_num(v, nan=0.0)
    fin_r = np.nan_to_num(rel, nan=0.0)
    tramos = []
    k = 0
    while k < T:
        if fin_v[k] < UMBRAL_DISPARA or fin_r[k] < UMBRAL_R2REL:
            k += 1
            continue
        a = b = k
        while a > 0 and fin_v[a - 1] >= UMBRAL_SIGUE:
            a -= 1
        while b < T - 1 and fin_v[b + 1] >= UMBRAL_SIGUE:
            b += 1
        pico = float(fin_v[a:b + 1].max())
        a2, b2, afinado = _relevo(trayectorias, T, a, b)
        tramos.append((int(a2), int(b2), pico, bool(afinado)))
        k = b + 1
    mask = np.ones(T, dtype=bool)
    for a, b, _p, _f in tramos:
        mask[max(a, 0):min(b + 1, T)] = False
    return {"mask": mask, "tramos": tramos, "senal": v, "r2rel": rel,
            "fuera": int((~mask).sum())}

"""El ancla por REGIONES de color: la banda central mide 10 m y es unica, asi que no hay ambiguedad.

## POR QUE HACIA FALTA OTRA IDEA, y no otro ajuste

Medido hoy sobre 707 clips con etiqueta humana, probando seis convenios distintos de "donde ocurrio
la accion" (mediana del clip, ultimo evento, primer evento, ultimo fondo, media de extremos, ultimo
cuartil):

    pendiente contra la etiqueta humana:  +0,053  +0,108  -0,122  +0,086  -0,053  +0,101

**Ninguno pasa de 0,11 y la ideal es 1,0.** Y no es que falte pista en el cuadro: el tramo visible
es p50 **16,2 m** de una pista de 14, o sea que casi siempre se ve entera. Una pendiente de 0,1 con
la media correcta solo puede significar una cosa: **el origen es aleatorio clip a clip y centrado en
cero**. Que raya se llama `0` es casi una moneda al aire.

Y tiene una causa exacta. Las marcas del reglamento estan en -7, -5, -2, 0, +2, +5, +7, o sea con
separaciones **2, 3, 2, 2, 3, 2**, que es un PALINDROMO. Con tres marcas, `(-5, -2, 0)` y
`(0, +2, +5)` tienen las mismas separaciones leidas al reves.

    LAS DISTANCIAS ENTRE MARCAS NO PUEDEN DISTINGUIR IZQUIERDA DE DERECHA. No es que el
    emparejamiento este mal hecho: es que la senal que usa no contiene la respuesta.

Por eso ni `ancla_v2` -- que quito el 27 % de patrones imposibles -- ni `ancla_unificada` -- que
subio la pendiente de 0,074 a 0,100 -- podian arreglarlo. Los dos afinan un emparejamiento de
distancias, y el problema no esta ahi.

## EL BIT QUE FALTA, Y QUE SE ESTABA TIRANDO A LA BASURA

`pista._fronteras_color` calcula, para cada corte candidato,

    sa = np.linalg.norm(np.median(A, axis=1) - np.median(B, axis=1))

y devuelve `sa`. O sea: calcula la diferencia de color **con su signo y su direccion** y acto
seguido le aplica una norma, que **tira exactamente la informacion de que lado esta cada color**.
Devuelve donde hay un escalon y cuanto salta, nunca hacia donde.

## LA IDEA: TOPOLOGIA, NO DISTANCIAS

El reglamento t.19 pone las zonas de aviso en los **ultimos 2 m de cada lado**. Luego la pista se
parte en tres regiones de color:

        -7        -5                     +5        +7
    ----|=========|======================|=========|----
        |  aviso  |       CENTRAL        |  aviso  |
        |   2 m   |        10 m          |   2 m   |

La region central mide **10 m** y las de aviso **2 m**: una razon de **5:1**. Y sobre todo,
**la region central es UNICA**: es la mas ancha con diferencia. Eso no es palindromico.

De ahi sale todo sin emparejar nada:

    la region de color mas ancha ES la banda central
    su borde izquierdo ES -5  y su borde derecho ES +5
    su ANCHURA da la escala directamente:   k = 10,00 m / (columnas de la region)
    su CENTRO da el origen:                 q = -k * columna_central

No hace falta saber de que color es cada pista, ni cual de los dos colores es el de aviso: basta con
cual region es mas grande. Y la escala sale de una region de 150-300 columnas en vez de de la
distancia entre dos rayas de 3-4 columnas -- el propio `ancla_zonas` ya tenia medido que las
fronteras de color se detectan con razones senal-ruido de 3 a 30 mientras las rayas salian ambiguas
en 10 de 14 clips.

## LAS TRES CONDICIONES, declaradas antes de medir

    1. la region central NO TOCA los bordes de la tira. Si la toca, esta cortada por el encuadre y
       su anchura no vale 10 m: valdria menos y la escala saldria inflada. Esta es la condicion que
       de verdad puede fallar y por eso va la primera.
    2. hay al menos UNA region contigua cuya anchura este entre el 12 % y el 30 % de la central
       (los 2 m de aviso son el 20 % de los 10 m). Con las dos, el ancla es CONTROLADA.
    3. el ancho implicado de la pista visible cae en un rango fisico.

## LO QUE ESTO NO ARREGLA

Las pistas **sin zonas de color**. Medido sobre los 78 clips del censo: `ninguna zona de color` en
**28 de 78 (36 %)**. En esos, esta via no contesta y hay que decirlo en vez de inventar un origen --
que es justo lo que pasa hoy.
"""
from __future__ import annotations

import numpy as np

import pista as PI

CENTRAL_M = 10.0
AVISO_M = 2.0
RAZON_AVISO = (0.12, 0.30)
"""Los 2 m de aviso son el 20 % de los 10 m centrales. La ventana 12-30 % deja sitio a que el borde
de la region se estime con error y a pistas cuyo tramo de aviso este parcialmente fuera de cuadro."""

DIST_LAB_FUNDE = 12.0
"""Distancia en Lab por debajo de la cual dos regiones contiguas son EL MISMO COLOR y se funden.

HACE FALTA PORQUE `_fronteras_color` SOBRE-SEGMENTA A PROPOSITO: parte en binario recursivo hasta
`max_cortes = 4`, asi que dentro de la banda central mete cortes por sombras, juntas de plancha y
rotulos pisados. Sin fundir, la region que pisan los tiradores salia de 200 columnas y la tira
implicaba 41-53 m de pista en los cuatro clips probados -- o sea que las "regiones" no eran las
zonas de color del reglamento sino cada escalon detectado.

12 en Lab es aproximadamente el umbral en que dos colores dejan de parecer el mismo a simple vista;
las zonas de t.19 contra la pista son cambios de tono mucho mayores."""

MARGEN_BORDE_COL = 8
MIN_ANCHO_CENTRAL_COL = 120
LARGO_TIRA_M = (8.0, 30.0)


def regiones(t):
    """QUE DEVUELVE: lista de regiones `{c0, c1, ancho, lab}` en columnas de la tira, o [].

    Se reutilizan los cortes de `pista._fronteras_color` para no introducir un segundo detector de
    fronteras: si aqui se detectaran de otra manera, una discrepancia entre esta via y `ancla_zonas`
    mediria dos cosas a la vez y no se sabria cual.
    """
    pc = PI.perfil_color(t)
    if pc is None:
        return []
    med = np.asarray(pc["med"], "float64")
    ncol = med.shape[1]
    cortes = [int(round(d["col"]))
              for d in PI._funde_contiguas(PI._fronteras_color(med))]
    lim = [0] + sorted(c for c in cortes if 0 < c < ncol) + [ncol]
    out = []
    for a, b in zip(lim[:-1], lim[1:]):
        if b - a < 2:
            continue
        out.append({"c0": int(a), "c1": int(b), "ancho": int(b - a),
                    "lab": np.median(med[:, a:b], axis=1)})
    return funde_por_color(out, med)


def funde_por_color(regs, med, dist=DIST_LAB_FUNDE):
    """QUE HACE: funde regiones contiguas del MISMO color hasta que no queden.

    Se funde siempre por el par mas parecido primero, no de izquierda a derecha: encadenando por
    orden, una sucesion de pequenas diferencias se come regiones que si son distintas. Es el mismo
    fallo que `_funde_contiguas` documenta para las fronteras y que le costo a
    `SeparateHoarseFeline` pasar de 14 candidatas a 5.
    """
    R = [dict(r) for r in regs]
    while len(R) > 1:
        d = [float(np.linalg.norm(R[i]["lab"] - R[i + 1]["lab"])) for i in range(len(R) - 1)]
        i = int(np.argmin(d))
        if d[i] >= dist:
            break
        a, b = R[i], R[i + 1]
        R[i:i + 2] = [{"c0": a["c0"], "c1": b["c1"], "ancho": b["c1"] - a["c0"],
                       "lab": np.median(med[:, a["c0"]:b["c1"]], axis=1)}]
    return R


def columnas_apoyos(t, res):
    """QUE DEVUELVE: las columnas de la tira por las que pasan los apoyos de los dos tiradores.

    Se usa el apoyo de CAJA, igual que `control_estatura` y `corredor`, porque existe siempre."""
    r, Mk = (res or {}).get("_r"), (res or {}).get("_Mk")
    if r is None:
        return np.array([])
    P = PI._pies_caja(r)
    pts = []
    for k in np.nonzero(r["mask"])[0]:
        for lado in (0, 1):
            q = P[k, lado]
            if not np.isfinite(q).all():
                continue
            if Mk is not None and k < len(Mk):
                v = Mk[k] @ np.array([q[0], q[1], 1.0])
                q = v[:2] / v[2]
            pts.append(q)
    if len(pts) < 8:
        return np.array([])
    c = PI.a_tira(t, np.asarray(pts, "float64"))[:, 0]
    return c[np.isfinite(c)]


def ancla(t, res=None):
    """QUE DEVUELVE: dict con la misma forma que `pista.ancla_zonas` -- `ok`, `m_por_col`, `q_m`,
    `x_ini_m`, `x_fin_m`, `controlada`, `via_ancla` -- mas `n_avisos` y `razon_aviso`, o `motivo`.

    `res` es el resultado de `pista.analiza`; si se pasa, la banda central se elige por donde pisan
    los tiradores en vez de por anchura, que es mucho mas fiable.

    Devuelve la misma forma a proposito: asi se puede sustituir a `ancla_zonas` sin tocar
    `posicion.py`, y la comparacion entre las dos mide el ancla y nada mas.
    """
    regs = regiones(t)
    if len(regs) < 2:
        return {"ok": False, "motivo": "menos de 2 regiones de color", "n_regiones": len(regs)}
    ncol = int(t["ncol"])
    pies = columnas_apoyos(t, res) if res is not None else np.array([])
    if len(pies):
        cp = float(np.median(pies))
        idx = next((j for j, r in enumerate(regs) if r["c0"] <= cp < r["c1"]), None)
        if idx is None:
            idx = int(np.argmin([abs(0.5 * (r["c0"] + r["c1"]) - cp) for r in regs]))
        for j in (idx - 1, idx + 1):
            if 0 <= j < len(regs) and regs[j]["ancho"] > 2.5 * regs[idx]["ancho"]:
                idx = j
        """LA BANDA CENTRAL ES LA QUE PISAN LOS TIRADORES, no la mas ancha del cuadro. Medido: con
        el criterio de "la mas ancha", `AbandonedBiodegradableCoqui` elegia una region que implicaba
        23,8 m de pista, porque la tira se extiende mas alla de los limites y el suelo del pabellon
        es una region uniforme MAS GRANDE que la propia pista.

        El repesco a la region contigua cubre el unico caso en que el pie enganaria: que la accion
        ocurra dentro de una zona de aviso, que son 2 m de 14. Si la vecina es 2,5 veces mas ancha,
        estabamos en el aviso y la central es esa."""
    else:
        idx = int(np.argmax([r["ancho"] for r in regs]))
    cen = regs[idx]
    if cen["ancho"] < MIN_ANCHO_CENTRAL_COL:
        return {"ok": False, "motivo": "la region mas ancha son %d columnas, muy poco"
                                       % cen["ancho"]}
    if cen["c0"] <= MARGEN_BORDE_COL or cen["c1"] >= ncol - MARGEN_BORDE_COL:
        return {"ok": False, "motivo": "la region central toca el borde del encuadre: esta "
                                       "cortada y su anchura no son 10 m"}
    """CONDICION 1, Y ES LA QUE DE VERDAD FALLA. Si la banda central esta cortada por el encuadre,
    medir su anchura y llamarla 10 m infla la escala en silencio. Es preferible no contestar."""

    k = CENTRAL_M / float(cen["ancho"])
    c_medio = 0.5 * (cen["c0"] + cen["c1"])
    q = -k * c_medio
    largo = k * ncol
    if not (LARGO_TIRA_M[0] <= largo <= LARGO_TIRA_M[1]):
        return {"ok": False, "motivo": "la tira implicaria %.1f m de pista" % largo}

    avisos, razones = 0, []
    for j in (idx - 1, idx + 1):
        if 0 <= j < len(regs):
            r = regs[j]["ancho"] / float(cen["ancho"])
            razones.append(round(r, 3))
            if RAZON_AVISO[0] <= r <= RAZON_AVISO[1]:
                avisos += 1
    """CONDICION 2: las regiones de aviso miden 2 m contra los 10 m de la central, o sea el 20 %.
    Con las dos presentes y en proporcion, la hipotesis queda confirmada por una medida que NO se
    uso para construirla -- que es lo mas parecido a un control independiente que hay aqui."""

    return {"ok": True, "m_por_col": float(k), "q_m": float(q),
            "patron": [-5.0, 5.0], "cols_ident": [float(cen["c0"]), float(cen["c1"])],
            "n_ident": 2, "n_regiones": len(regs),
            "col_centro": float(c_medio),
            "ancho_central_col": int(cen["ancho"]),
            "n_avisos": int(avisos), "razon_aviso": razones,
            "largo_visible_m": float(largo),
            "x_ini_m": float(q), "x_fin_m": float(k * (ncol - 1) + q),
            "loo_p50_m": None, "res_max_m": 0.0,
            "controlada": bool(avisos >= 1),
            "ambiguo": False, "otra": None,
            "via_ancla": "region central de 10 m entre las dos zonas de aviso t.19"}

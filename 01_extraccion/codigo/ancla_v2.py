"""El emparejamiento de marcas de la pista, con las tres correcciones que la medida pidio.

## QUE ESTA MAL EN EL DE HOY, leido en el codigo y medido

`pista.identifica_marcas` hace RANSAC **sobre pares**: ajusta la recta por DOS marcas y se queda
con la hipotesis que mas inliers tenga. De ahi salen tres defectos:

**1. El residuo publicado esta contaminado.** Las dos marcas semilla tienen error **cero por
construccion**: definen la recta. Con `n_ident = 3`, dos son semillas y una es predicha, asi que la
MEDIANA cae sobre una semilla y sale `0,000 m`. Eso no es precision, es aritmetica -- el mismo
error que el proyecto tiene documentado con el cono (*"la hoja esta a 0 grados de su propio
centro"*). Medido: **10 de 28** clips tienen 5 marcas o mas, que es lo minimo para que haya 3
predichas y el control discrimine algo.

**2. Se desempata por numero de inliers, no por poder explicativo.** Las marcas del reglamento
estan en -7, -5, -2, 0, +2, +5, +7 y sus separaciones **se repiten**: 2, 3, 2, 2, 3, 2. Una escala
equivocada encuentra inliers igual. El propio modulo ya lo tenia medido para las rayas: *"la escala
que dan se equivoca por factores de 1,3 a 2,6"*.

**3. No se comprueba que la hipotesis quepa.** Medido: un clip identifica `[-7, 0, +2]` -- **9 m**
de patron -- en una tira que abarca **6,06 m**. No caben. Falla en **1 de 8** y nada lo mira.

## LAS TRES CORRECCIONES

**A. El residuo se mide SOLO sobre las marcas que no son semilla.** Es la unica cifra que dice algo
sobre la capacidad de predecir, y es lo que separa una medida de un ajuste.

**B. Se puntua por COMPLETITUD, no por numero de inliers.** Si la hipotesis dice que la tira va de
-7 a -1, entonces las marcas de -7, -5 y -2 **deberian estar las tres** en el cuadro. Explicar 2 de
3 es peor que explicar 3 de 3 aunque el numero absoluto sea el mismo. La puntuacion es

    completitud = marcas del reglamento DENTRO del tramo visible que estan explicadas
                  -------------------------------------------------------------------
                  marcas del reglamento DENTRO del tramo visible

que es una verosimilitud de verdad y **penaliza a las escalas infladas**: una escala 2x mete mas
marcas del reglamento en el tramo, y si no aparecen, la hipotesis se hunde. Eso es exactamente lo
que el criterio de inliers no podia distinguir.

**C. Se rechaza lo que no cabe.** Si el tramo implicado por la hipotesis excede lo que la tira
abarca, la hipotesis es imposible, por muchos inliers que tenga.

## LO QUE ESTO NO ARREGLA

Si en el cuadro solo hay **dos** marcas detectables, no hay control posible y no lo habra: dos
puntos y dos parametros. Esos clips salen marcados `controlada = False` y su x absoluta **no
deberia publicarse como medida**. Arreglar eso no es cuestion de algoritmo sino de decir que no se
sabe, que es la disciplina que el proyecto aplica en todas partes menos aqui.
"""
from __future__ import annotations

import numpy as np

TOL_IDENT_M = 0.20
"""La misma tolerancia que el de hoy, a proposito: si se cambiara, la comparacion entre los dos
mediria dos cosas y no se sabria cual mejoro."""

MIN_IDENT = 3
MIN_PREDICHAS = 1
"""Marcas NO semilla minimas. Con 0 no hay control ninguno; con 1 hay uno que no discrimina pero al
menos existe. El umbral de "controlada" se pone aparte, en 3."""

MIN_CONTROL = 3
"""Marcas no semilla para llamar al anclaje CONTROLADO. Por debajo, se entrega igual pero marcado."""

FIE_POS = (-7.0, -5.0, -2.0, 0.0, 2.0, 5.0, 7.0)


def puntua(X, sp_idx_semilla, err, inl, span_visible):
    """QUE DEVUELVE: (completitud, err_pred_p50, n_predichas, n_esperadas) de una hipotesis.

    `completitud` es la fraccion de marcas del reglamento que CAEN en el tramo visible y ademas
    estan explicadas por una marca detectada. Es la cifra que penaliza una escala inflada: al
    estirar la escala entran mas marcas del reglamento en el tramo, y si no se ven, baja.
    """
    dentro = [m for m in FIE_POS if span_visible[0] - TOL_IDENT_M <= m <= span_visible[1] + TOL_IDENT_M]
    if not dentro:
        return 0.0, float("inf"), 0, 0
    expl = set()
    for k in np.nonzero(inl)[0]:
        expl.add(float(FIE_POS[int(np.argmin(np.abs(X[k] - np.asarray(FIE_POS))))]))
    comp = len(expl & set(dentro)) / len(dentro)
    no_sem = [k for k in np.nonzero(inl)[0] if k not in sp_idx_semilla]
    ep = float(np.median([err[k] for k in no_sem])) if no_sem else float("inf")
    return comp, ep, len(no_sem), len(dentro)


def identifica(sp, span_tira_m=None):
    """QUE HACE: el emparejamiento con las tres correcciones. Misma entrada que el de hoy -- el
    vector `sp` de coordenadas proyectivas de las marcas detectadas -- para poder comparar los dos
    sobre exactamente los mismos datos.

    QUE DEVUELVE: dict con `ok`, `A`, `B`, `patron`, `n_ident`, `n_predichas`, `completitud`,
    `err_pred_p50_m` (el honesto), `controlada`, y `motivo` si falla.
    """
    sp = np.asarray(sp, dtype="float64")
    n = len(sp)
    if n < MIN_IDENT:
        return {"ok": False, "motivo": "menos de %d marcas" % MIN_IDENT, "n": n}
    fie = np.asarray(FIE_POS)
    mejor = None
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            den = sp[j] - sp[i]
            if abs(den) < 1e-14:
                continue
            for xi in FIE_POS:
                for xj in FIE_POS:
                    if xi == xj:
                        continue
                    A = (xj - xi) / den
                    if abs(A) < 1e-14:
                        continue
                    B = xi - A * sp[i]
                    X = A * sp + B
                    err = np.min(np.abs(X[:, None] - fie[None, :]), axis=1)
                    inl = err <= TOL_IDENT_M
                    if inl.sum() < MIN_IDENT:
                        continue
                    lo, hi = float(np.min(X)), float(np.max(X))
                    if span_tira_m is not None and (hi - lo) > span_tira_m * 1.10:
                        continue
                    """CORRECCION C: si el tramo implicado no cabe en la tira, la hipotesis es
                    imposible. Medido: falla en 1 de 8 clips y hoy nadie lo comprueba."""
                    comp, ep, npred, nesp = puntua(X, {i, j}, err, inl, (lo, hi))
                    if npred < MIN_PREDICHAS:
                        continue
                    """CORRECCION A: al menos una marca que no sea semilla, o no hay nada que
                    predecir y el residuo es el de un ajuste consigo mismo."""
                    clave = (round(comp, 4), int(inl.sum()), -ep)
                    """CORRECCION B: manda la COMPLETITUD, y el numero de inliers solo desempata.
                    Al reves -- que es lo de hoy -- una escala inflada gana por meter mas marcas."""
                    if mejor is None or clave > mejor[0]:
                        mejor = (clave, A, B, inl.copy(), comp, ep, npred, nesp, {i, j})
    if mejor is None:
        return {"ok": False, "motivo": "ninguna hipotesis pasa las tres condiciones", "n": n}
    _c, A, B, inl, comp, ep, npred, nesp, sem = mejor
    X = A * sp + B
    pat = []
    for k in np.nonzero(inl)[0]:
        pat.append(float(fie[int(np.argmin(np.abs(X[k] - fie)))]))
    if len(set(pat)) < MIN_IDENT:
        return {"ok": False, "motivo": "dos marcas caen en la misma posicion del patron", "n": n}
    return {"ok": True, "A": float(A), "B": float(B), "patron": sorted(set(pat)),
            "n_ident": int(inl.sum()), "n_predichas": int(npred),
            "completitud": float(comp), "marcas_esperadas": int(nesp),
            "err_pred_p50_m": (None if not np.isfinite(ep) else float(ep)),
            "controlada": bool(npred >= MIN_CONTROL and np.isfinite(ep)
                               and ep <= TOL_IDENT_M / 2),
            "span_m": float(np.max(X) - np.min(X)), "semillas": sorted(sem),
            "X_todas": [round(float(v), 3) for v in X]}

"""El ancla de la pista con las DOS familias de marcas a la vez, en un solo ajuste.

## EL DEFECTO, y es de arquitectura, no de calculo

Hoy hay dos detectores de marcas en `pista.py` que **no se hablan**:

    ancla_zonas         solo FRONTERAS DE COLOR (t.19)   plantilla (-7, -5, +5, +7)
                        -> de aqui salen los METROS de la posicion

    identifica_marcas   solo RAYAS PINTADAS              plantilla (-7,-5,-2,0,+2,+5,+7)
                        -> de aqui sale la HOMOGRAFIA, que la posicion NO usa

`posicion.posicion_entre_bordes` llama a `ancla_zonas` y nada mas. O sea que **las rayas pintadas no
intervienen en la posicion en ningun momento**, aunque estan detectadas, afinadas y emparejadas.

Y la consecuencia esta medida y escrita en el propio docstring de `ancla_zonas`:

    encaja en 13 de 20 clips ... pero **en 11 de esos 13 solo con DOS fronteras**
    con dos, el sistema esta exactamente determinado: **no hay ningun control**
    contra la homografia de referencia, la escala se separa un **32 %**

Dos observaciones y dos parametros (escala y origen) ajustan exacto siempre, se acierte o se falle
de frontera. Es el mismo cero-por-aritmetica que el proyecto ya tiene documentado con el cono y con
`ident_err_p50_m`. De ahi sale la posicion de hoy: pendiente **-0,314** contra la etiqueta humana,
acierto 0,250 contra el 0,430 de contestar siempre `middle`.

## POR QUE UNIR LAS DOS FAMILIAS ES LO CORRECTO, y no un apano

Son marcas de **la misma regla**. El reglamento pone siete posiciones en la misma pista:

    -7   -5        -2    0    +2         +5    +7
    |    |          |    |    |           |    |
    |    frontera de color (t.19)         |    |
    limite posterior                      |    limite
                    puesta en guardia, raya pintada
                         linea central, raya pintada

Y en la tira rectificada la relacion columna -> metros es **afin** para las dos familias por igual;
eso no es un supuesto, es la propiedad que `tira_rectificada` demuestra: *"a fila fija, la COLUMNA
es AFIN en x"*. Asi que las dos familias son observaciones del **mismo** par (k, q), y separarlas
tira informacion.

Lo que se gana es exactamente lo que falta:

    hoy     2 fronteras, 2 parametros   ->   0 grados de libertad, 0 control
    unido   2 fronteras + 2 rayas       ->   2 grados de libertad, deja-una-fuera de verdad

Y se gana una comprobacion que hoy no existe en ningun sitio: que las dos familias, detectadas por
metodos completamente distintos -- una por escalon de color en Lab sobre regiones de 150-300
columnas, otra por filtro de cresta sobre trazos de 3-4 columnas --, **coincidan en la misma regla**.
Dos detectores independientes que dan la misma escala es una evidencia que ninguno de los dos puede
dar solo.

## LAS CUATRO CONDICIONES, declaradas antes de medir

    1. k > 0            la x crece con la columna, que es el convenio del eje de `posicion.apoyos`
    2. largo visible    entre 2 y 20 m, igual que hoy
    3. inyectiva        dos observaciones no pueden ser la misma marca del reglamento
    4. cruce de familia cuando hay observaciones de las dos, la solucion tiene que explicar al
                        menos una de cada. Si una familia queda entera fuera, la solucion se apoya
                        en un solo detector y vuelve a ser lo de hoy.

Y el residuo se mide **solo sobre las marcas que no son semilla**, que es la unica cifra que dice
algo sobre capacidad de predecir.

## LO QUE ESTO NO ARREGLA

Si la tira rectificada no es la pista, esto no lo salva: emparejara costuras de moqueta con el
reglamento igual de bien. Medido aparte con `verifica_focal_fisica.py`, la geometria del suelo es
sana en una minoria de los clips, y ese es un problema de `region_y_bordes`, un paso mas arriba.
Aqui se arregla el ancla; el techo lo pone la geometria.
"""
from __future__ import annotations

import numpy as np

import pista as PI

FIE_POS = (-7.0, -5.0, -2.0, 0.0, 2.0, 5.0, 7.0)
POS_COLOR = (-7.0, -5.0, 5.0, 7.0)
"""Las fronteras de color solo pueden ser una de estas cuatro: el cambio de color de t.19 esta a
+-5 m y el final de la pista a +-7 m. Una raya pintada, en cambio, puede ser cualquiera de las
siete. Restringir cada familia a lo que fisicamente puede ser quita hipotesis falsas gratis."""

TOL_COLOR_M = 0.35
TOL_RAYA_M = 0.20
"""Tolerancias distintas a proposito, y no es arbitrario: una frontera de color es el borde de una
region de 2 m y su localizacion tiene la incertidumbre del gradiente; una raya pintada mide 3-4 cm.
Darles la misma tolerancia seria tratar como igual de precisas dos medidas que no lo son."""

MIN_IDENT = 3
MIN_NO_SEMILLA = 1
LARGO_M = (2.0, 20.0)
K_RANGO = (0.002, 0.060)


def observaciones(t, res):
    """QUE DEVUELVE: lista de dicts `{col, familia, peso}` con las marcas detectadas en columnas de
    la tira, de las dos familias, o [].

    Las rayas se llevan a la tira con `a_tira`, que es la misma transformacion que usa
    `posicion_entre_bordes` para los apoyos: asi las dos cosas viven en el mismo sistema y no hay
    que fiarse de ninguna conversion intermedia.
    """
    obs = []
    pc = PI.perfil_color(t)
    if pc is not None:
        for d in PI._funde_contiguas(PI._fronteras_color(pc["med"])):
            obs.append({"col": float(d["col"]), "familia": "color",
                        "razon": float(d.get("razon", 0.0))})
    mar = (res or {}).get("_mar") or {}
    for li in (mar.get("lineas") or []):
        if not li.get("afinada"):
            continue
        try:
            c = PI.a_tira(t, np.asarray(li["centro"], "float64")[None, :])[0]
        except Exception:
            continue
        if np.isfinite(c[0]) and 0 <= c[0] < t["ncol"]:
            obs.append({"col": float(c[0]), "familia": "raya", "razon": 0.0})
    return obs


def _permitidas(fam):
    return POS_COLOR if fam == "color" else FIE_POS


def _tol(fam):
    return TOL_COLOR_M if fam == "color" else TOL_RAYA_M


def ancla(t, res):
    """QUE DEVUELVE: dict con `ok`, `m_por_col`, `q_m`, `patron`, `n_ident`, `n_color`, `n_raya`,
    `loo_p50_m`, `controlada`, `cruce_familias`, o `motivo`.

    La forma de la salida imita a la de `pista.ancla_zonas` para que se pueda sustituir sin tocar
    `posicion.py`, y la comparacion entre las dos mida el ancla y nada mas.
    """
    obs = observaciones(t, res)
    n = len(obs)
    if n < MIN_IDENT:
        return {"ok": False, "motivo": "menos de %d marcas entre las dos familias" % MIN_IDENT,
                "n": n}
    cols = np.array([o["col"] for o in obs], "float64")
    fams = [o["familia"] for o in obs]
    hay_color = "color" in fams
    hay_raya = "raya" in fams
    ncol = float(t["ncol"])

    mejor = None
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dc = cols[j] - cols[i]
            if abs(dc) < 1e-9:
                continue
            for xi in _permitidas(fams[i]):
                for xj in _permitidas(fams[j]):
                    if xi == xj:
                        continue
                    k = (xj - xi) / dc
                    if not (K_RANGO[0] <= k <= K_RANGO[1]):
                        continue
                    if not (LARGO_M[0] <= k * ncol <= LARGO_M[1]):
                        continue
                    q = xi - k * cols[i]
                    X = k * cols + q
                    asig, usadas, err = {}, set(), {}
                    for m in np.argsort([min(abs(X[a] - p) for p in _permitidas(fams[a]))
                                         for a in range(n)]):
                        perm = [p for p in _permitidas(fams[m]) if p not in usadas]
                        if not perm:
                            continue
                        d = [abs(X[m] - p) for p in perm]
                        b = int(np.argmin(d))
                        if d[b] <= _tol(fams[m]):
                            asig[m] = perm[b]
                            usadas.add(perm[b])
                            err[m] = d[b]
                    """ASIGNACION INYECTIVA Y POR CERCANIA CRECIENTE: se coloca primero la
                    observacion que menos duda tiene, y cada marca del reglamento se gasta una sola
                    vez. Sin esto salen patrones como [-5, -5, -2, -2], que es el defecto que ya se
                    midio en el 27 % de los emparejamientos del otro detector."""
                    if len(asig) < MIN_IDENT:
                        continue
                    hay_dos = hay_color and hay_raya
                    cruz = (any(fams[m] == "color" for m in asig)
                            and any(fams[m] == "raya" for m in asig))
                    if hay_dos and not cruz:
                        continue
                    """SI HAY OBSERVACIONES DE LAS DOS FAMILIAS, LA SOLUCION TIENE QUE USAR LAS
                    DOS. Una solucion que explique solo las de color vuelve a ser el ancla de hoy,
                    con su ajuste exactamente determinado y sin control."""
                    no_sem = [m for m in asig if m not in (i, j)]
                    if len(no_sem) < MIN_NO_SEMILLA:
                        continue
                    ep = float(np.median([err[m] for m in no_sem]))
                    clave = (len(asig), -ep)
                    if mejor is None or clave > mejor[0]:
                        mejor = (clave, k, q, dict(asig), dict(err), {i, j}, hay_dos)
    if mejor is None:
        return {"ok": False, "motivo": "ninguna hipotesis cumple las cuatro condiciones", "n": n}

    _cl, k, q, asig, _err, sem, hay_dos = mejor
    idx = sorted(asig.keys())
    ci = np.array([cols[m] for m in idx])
    xs = np.array([asig[m] for m in idx])
    A = np.stack([ci, np.ones(len(ci))], axis=1)
    sol, *_ = np.linalg.lstsq(A, xs, rcond=None)
    k, q = float(sol[0]), float(sol[1])
    """REAJUSTE POR MINIMOS CUADRADOS CON TODAS LAS IDENTIFICADAS, no solo con las dos semilla: la
    hipotesis sirve para decidir QUE es cada marca, y una vez decidido la escala la deben fijar
    todas."""
    loo = []
    if len(ci) >= 3:
        for h in range(len(ci)):
            m2 = [a for a in range(len(ci)) if a != h]
            A2 = np.stack([ci[m2], np.ones(len(m2))], axis=1)
            s2, *_ = np.linalg.lstsq(A2, xs[m2], rcond=None)
            loo.append(abs(s2[0] * ci[h] + s2[1] - xs[h]))
    res_abs = np.abs(k * ci + q - xs)
    return {"ok": True, "m_por_col": k, "q_m": q,
            "patron": [float(v) for v in xs], "cols_ident": [float(v) for v in ci],
            "n_ident": len(ci),
            "n_color": sum(1 for m in idx if fams[m] == "color"),
            "n_raya": sum(1 for m in idx if fams[m] == "raya"),
            "n_obs": n, "cruce_familias": bool(hay_dos),
            "familias": ("color+raya" if hay_dos else
                         ("solo color" if hay_color else "solo raya")),
            "res_max_m": float(res_abs.max()),
            "loo_p50_m": float(np.median(loo)) if loo else None,
            "loo_max_m": float(np.max(loo)) if loo else None,
            "largo_visible_m": abs(k) * ncol,
            "x_ini_m": q, "x_fin_m": k * (ncol - 1.0) + q,
            "col_centro": (-q / k) if abs(k) > 1e-12 else float("nan"),
            "controlada": bool(len(ci) >= 3 and loo and np.median(loo) <= 0.30),
            "ambiguo": False, "otra": None,
            "via_ancla": "fronteras de color + rayas pintadas (ajuste unico)"}

"""ETAPA A3: elegir los dos tiradores por TRAYECTORIA, no cuadro a cuadro.

POR QUE POR TRAYECTORIA Y NO POR CUADRO, que es el cambio de arquitectura de este fichero:

  Elegir "los dos mejores de este cuadro" no tiene memoria, y por eso el algoritmo viejo podia
  cambiar de persona a mitad de clip. Medido en la auditoria de series: **0,43 % de los clips tienen
  cambio de identidad**, y ese es el fallo que corrompe la etiqueta de lado, no solo el que anade
  ruido.

  Una trayectoria que dura 300 cuadros es un tirador. Un espectador que aparece en 5 no lo es. Asi
  que se construyen trayectorias, se puntuan enteras, y se eligen DOS para todo el clip. Con eso el
  cambio de identidad deja de ser posible por construccion, no por umbral.

COMO SE PUNTUA UNA TRAYECTORIA, y cada termino sale de algo medido:

  DURACION          un tirador esta en pista casi todo el clip; el publico entra y sale del plano
  ALTO DE CAJA      medido: las cajas de tirador son 0,345-0,450 del alto de imagen
  PIES EN LA BANDA  la pista activa esta entre 0,45 y 0,97 del alto; el publico y la mesa, fuera
  MOVIMIENTO        el arbitro y el publico estan casi quietos; un tirador se desplaza
  LADO              los dos tiradores estan en mitades opuestas: se exige uno a cada lado

NO SE USA LA CONFIANZA DE YOLO para puntuar. Medido en un clip: va de 0,886 a 0,930 para TODO, asi
que no discrimina nada. Un rasgo que no separa no entra en la puntuacion aunque este disponible.
"""
from __future__ import annotations

import numpy as np

BANDA = (0.45, 0.97)
IOU_MIN = 0.25
DIST_MAX_FRAC = 0.10
"""OBSOLETO: limite de asociacion relativo al ANCHO DE IMAGEN. Se deja documentado porque explica
una averia. Con 0,10 de 1280 px daban 128 px de holgura, que para una caja de 90 px de ancho son
**1,4 anchos de caja**, o sea suficiente para que el rastreador salte a otra persona.

MEDIDO sobre 6.243 transiciones de las parejas verificadas: el desplazamiento real del centro entre
cuadros consecutivos tiene **p50 0,016, p95 0,102, p99 0,168 y p99,9 0,460 anchos de caja**. Los
saltos que aparecian en los avisos eran de 1,41 y 1,67 anchos, o sea **8 a 10 veces el p99**. No es
movimiento, es cambio de persona."""
DESPL_BASE = 0.35
DESPL_POR_CUADRO = 0.25
DESPL_TOPE = 2.00
IOU_GEMELAS = 0.60
"""Solape por el que dos trayectorias VIVAS en el mismo cuadro se consideran la misma persona.

QUE AVERIA PARA, medida en `BabyishIdioticCockerspaniel`: dos trayectorias se sentaron encima del
MISMO tirador en c250-c253, con centros identicos (cx 0,937 y 0,937, IoU 0,68), y al separarse cada
una se fue con una persona distinta: en c266 una estaba en cx 0,600 y la otra en 0,854. O sea un
CAMBIO DE IDENTIDAD, que es el fallo que corrompe la etiqueta de lado, y entro por la puerta de
tener dos trayectorias sobre una sola persona.

No se puede cazar despues con el filtro de continuidad: el desplazamiento reparte 1,8 anchos en 8
cuadros, que son 0,23 por cuadro, dentro de lo normal. Hay que impedirlo en el origen.

La que sobrevive es la que MAS detecciones lleva, porque es la que tiene mas historia detras."""
"""Limite de asociacion relativo a la CAJA, que es lo que lo arregla. Para cuadros consecutivos deja
0,60 anchos, que cubre el p99,9 medido (0,460) con holgura y corta a 0,08 % de las transiciones, que
son justo las malas. Crece con el hueco porque tras 12 cuadros perdidos el tirador si se ha movido,
y se topa en 2 anchos para que un hueco largo no vuelva a abrir la puerta a cualquiera."""
HUECO_MAX = 12
"""Cuadros que una trayectoria puede pasar sin deteccion antes de considerarse cortada. Es el mismo
valor que se valido en el proyecto para `max_gap_redetect`: 12 cuadros, con 0 puntos perdidos en la
ventana de guardia y un ahorro de coste del 92,8 %."""


def _iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _centro(b):
    return np.array([0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])])


def construye_trayectorias(cajas, cuadros, W, H, T) -> list:
    """QUE HACE: agrupa las detecciones en trayectorias por solape y proximidad.

    Asociacion voraz: para cada cuadro, cada deteccion se une a la trayectoria viva mas parecida
    (mayor IoU, y si no hay solape, centro mas cercano por debajo de `DIST_MAX_FRAC` del ancho).
    Una trayectoria muere si pasa `HUECO_MAX` cuadros sin deteccion.

    QUE DEVUELVE: lista de dicts con `cuadros`, `cajas` y sus indices originales.
    """
    orden = np.argsort(cuadros, kind="stable")
    vivas, muertas = [], []
    k_ant = -1
    for i in orden:
        k = int(cuadros[i])
        if k != k_ant:
            for t in list(vivas):
                if k - t["ultimo"] > HUECO_MAX:
                    vivas.remove(t)
                    muertas.append(t)
            k_ant = k
        b = cajas[i]
        mejor, mejor_v = None, -1.0
        for t in vivas:
            if t["ultimo"] == k:
                continue
            v = _iou(b, t["cajas"][-1])
            if v < IOU_MIN:
                ult = t["cajas"][-1]
                """El limite va en anchos de LA CAJA y crece con el hueco, no en fraccion del ancho
                de imagen: si no, una caja pequenia hereda una holgura enorme."""
                ancho = max(float(ult[2] - ult[0]), 1.0)
                g = max(1, k - t["ultimo"])
                dmax = ancho * min(DESPL_BASE + DESPL_POR_CUADRO * g, DESPL_TOPE)
                d = np.linalg.norm(_centro(b) - _centro(ult))
                v = max(0.0, IOU_MIN * (1.0 - d / dmax)) if d < dmax else -1.0
            if v > mejor_v:
                mejor, mejor_v = t, v
        if mejor is not None and mejor_v > 0:
            mejor["cuadros"].append(k)
            mejor["cajas"].append(b)
            mejor["idx"].append(int(i))
            mejor["ultimo"] = k
        else:
            vivas.append({"cuadros": [k], "cajas": [b], "idx": [int(i)], "ultimo": k})
        """GEMELAS: si esta deteccion ha dejado dos trayectorias vivas encima de la misma persona en
        este cuadro, se mata la mas corta ahora mismo. Sus detecciones futuras se asociaran a la que
        sobrevive, y asi el cruce de identidades no puede ocurrir."""
        aqui = [t for t in vivas if t["ultimo"] == k]
        muertos_ya = set()
        for x in range(len(aqui)):
            for y in range(x + 1, len(aqui)):
                ta, tb = aqui[x], aqui[y]
                if id(ta) in muertos_ya or id(tb) in muertos_ya:
                    continue
                if _iou(ta["cajas"][-1], tb["cajas"][-1]) < IOU_GEMELAS:
                    continue
                corta = ta if len(ta["cuadros"]) < len(tb["cuadros"]) else tb
                muertos_ya.add(id(corta))
        if muertos_ya:
            """Se comparan por identidad, no por igualdad: un dict con arrays de numpy dentro no se
            puede comparar con `==` y `list.remove` casca."""
            vivas_nuevas = []
            for t in vivas:
                if id(t) in muertos_ya:
                    muertas.append(t)
                else:
                    vivas_nuevas.append(t)
            vivas = vivas_nuevas
    return muertas + vivas


def puntua(t, W, H, T) -> dict:
    """QUE HACE: puntua una trayectoria entera y devuelve sus rasgos por separado.

    Se devuelven los rasgos ademas de la puntuacion para poder ver POR QUE se eligio o se descarto.
    Una puntuacion sin sus terminos no se puede contradecir.
    """
    c = np.asarray(t["cajas"], dtype="float64")
    ks = np.asarray(t["cuadros"], dtype="int64")
    alto = (c[:, 3] - c[:, 1]) / H
    pie = c[:, 3] / H
    cx = 0.5 * (c[:, 0] + c[:, 2]) / W
    dur = (ks.max() - ks.min() + 1) / float(max(T, 1))
    cobertura = len(ks) / float(max(ks.max() - ks.min() + 1, 1))
    en_banda = float(np.mean((pie >= BANDA[0]) & (pie <= BANDA[1])))
    mov = float(np.std(cx)) if len(cx) > 1 else 0.0
    alto_med = float(np.median(alto))

    """La talla se premia hasta 0,40 del alto de imagen y no mas: una caja gigantesca es alguien en
    primer plano cruzando la camara, no un tirador."""
    talla = min(alto_med / 0.40, 1.0) if alto_med < 0.60 else max(0.0, 1.0 - (alto_med - 0.60) * 3)
    p = (0.35 * dur + 0.20 * talla + 0.25 * en_banda + 0.10 * cobertura
         + 0.10 * min(mov / 0.05, 1.0))
    return {"puntos": float(p), "dur": float(dur), "cobertura": float(cobertura),
            "en_banda": en_banda, "mov": mov, "alto_med": alto_med,
            "cx_med": float(np.median(cx)), "n": int(len(ks)),
            "ini": int(ks.min()), "fin": int(ks.max())}


def elige_dos(trayectorias, W, H, T) -> dict:
    """QUE HACE: elige las DOS trayectorias que son los tiradores, una a cada lado.

    LA RESTRICCION DE LADO ES DURA, no un termino de la puntuacion: los dos tiradores estan en
    mitades opuestas de la imagen porque se enfrentan. Si las dos mejores estan del mismo lado, se
    coge la mejor y la mejor del OTRO lado, aunque puntue menos. Sin esta regla el selector puede
    coger dos veces a la misma persona, que es un fallo medido en 14 de 10.618 clips.
    """
    cand = []
    for t in trayectorias:
        r = puntua(t, W, H, T)
        cand.append({**r, "t": t})
    cand.sort(key=lambda r: -r["puntos"])
    if not cand:
        return {"elegidas": [], "candidatos": cand, "motivo": "sin candidatos"}
    if len(cand) == 1:
        return {"elegidas": [cand[0]], "candidatos": cand, "motivo": "solo una trayectoria"}

    a = cand[0]
    otros = [r for r in cand[1:] if (r["cx_med"] - 0.5) * (a["cx_med"] - 0.5) < 0]
    if otros:
        b = otros[0]
        motivo = "dos lados opuestos"
    else:
        b = cand[1]
        motivo = "AVISO: las dos mejores estan del MISMO lado"
    izq, der = (a, b) if a["cx_med"] <= b["cx_med"] else (b, a)
    return {"elegidas": [izq, der], "candidatos": cand, "motivo": motivo}


def a_matrices(elegidas, cajas, T, W, H):
    """QUE HACE: pasa las dos trayectorias elegidas a matrices (T, 2, 4) normalizadas, con presencia.

    Los cuadros sin deteccion quedan a `NaN` y marcados como ausentes: NO se rellenan aqui. Rellenar
    es la etapa C y va despues, para que se pueda ver cuanto falta ANTES de inventar nada.
    """
    B = np.full((T, 2, 4), np.nan, dtype="float64")
    presente = np.zeros((T, 2), dtype=bool)
    for lado, r in enumerate(elegidas[:2]):
        for k, c in zip(r["t"]["cuadros"], r["t"]["cajas"]):
            if 0 <= k < T:
                B[k, lado] = [c[0] / W, c[1] / H, c[2] / W, c[3] / H]
                presente[k, lado] = True
    return B, presente

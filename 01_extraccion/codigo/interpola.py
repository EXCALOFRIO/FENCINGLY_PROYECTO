"""ETAPA C: interpolar los huecos para que SIEMPRE haya dos poses completas, con el limite MEDIDO.

LA IDEA ES DEL USUARIO y es correcta: si en un cuadro falta un tirador, se rellena entre la ultima
pose conocida y la siguiente, para que el modelo no vea huecos. Lo que aporta este fichero es **el
limite**, porque interpolar un hueco de 2 cuadros y uno de 60 no es lo mismo y nadie habia medido
donde esta la frontera.

COMO SE MIDE SIN QUE NADIE ANOTE NADA: se cogen tiradas de cuadros que SI existen, se **esconden** los
de dentro, se interpola desde los dos extremos, y se compara con lo que realmente habia. Eso da el
error real en funcion de la longitud del hueco, y el limite sale de la curva en vez de inventarse.

TRES REGLAS QUE SALEN DE LO YA MEDIDO Y NO SE NEGOCIAN:

  1. **Solo huecos INTERIORES.** Un hueco al principio o al final del clip no tiene dos extremos entre
     los que meterse. Se declara, no se rellena. Extrapolar es inventar sin control.
  2. **Todo cuadro interpolado va MARCADO** en una columna. Sin la marca nadie puede volver a
     distinguir lo medido de lo inventado, y cualquier medida futura queda contaminada.
  3. **No se interpola entre dos poses malas.** Si los extremos son basura, el relleno es basura
     suave y creible, que es peor que el hueco. Por eso la etapa A va antes que esta.

TRAMPA DE DATOS YA MEDIDA: hay puntos con **confianza >= 0,3 y coordenada NaN**. Filtrar solo por
confianza los cuenta como buenos. Se exige confianza **Y** `isfinite`.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

CLAVE = [5, 6, 7, 8, 9, 10, 11, 12]
"""Hombros, codos, munecas y caderas: los que usan las medidas de prioridad. Los tobillos se pierden
mucho mas por oclusion y no deben decidir si un cuadro es utilizable."""
CONF_MIN = 0.3
MIN_PUNTOS = 5


def utiles(poses: np.ndarray) -> np.ndarray:
    """QUE HACE: matriz (T, 2) de booleanos: el cuadro tiene pose utilizable para ese tirador?"""
    P = np.asarray(poses, dtype="float64")
    conf = P[:, :, CLAVE, 2]
    fin = np.isfinite(P[:, :, CLAVE, 0]) & np.isfinite(P[:, :, CLAVE, 1])
    return ((conf >= CONF_MIN) & fin).sum(axis=2) >= MIN_PUNTOS


def huecos(mask_util: np.ndarray) -> list:
    """QUE HACE: lista de huecos con su tipo. `interior` solo si hay bueno a los DOS lados.

    Devuelve dicts con `ini`, `fin` (exclusivo), `largo`, `tipo` y los indices de los extremos
    buenos `a` y `b` cuando existen.
    """
    T = len(mask_util)
    fuera, i = [], 0
    while i < T:
        if mask_util[i]:
            i += 1
            continue
        j = i
        while j < T and not mask_util[j]:
            j += 1
        antes = i - 1 if i > 0 else None
        despues = j if j < T else None
        tipo = "interior" if (antes is not None and despues is not None) else "borde"
        fuera.append({"ini": i, "fin": j, "largo": j - i, "tipo": tipo,
                      "a": antes, "b": despues})
        i = j
    return fuera


def interpola_tramo(P: np.ndarray, a: int, b: int, lado: int) -> np.ndarray:
    """QUE HACE: interpolacion lineal punto a punto entre los cuadros `a` y `b` para un tirador.

    Se interpola cada articulacion por separado en x e y. La confianza del cuadro inventado se pone
    como la MENOR de los dos extremos, no la media: un relleno no puede ser mas fiable que el peor de
    sus apoyos, y ponerle la media lo haria parecer mejor de lo que es.

    Devuelve un array (b-a-1, 17, 3) con los cuadros intermedios.
    """
    pa, pb = P[a, lado], P[b, lado]
    n = b - a - 1
    if n <= 0:
        return np.zeros((0, 17, 3), dtype="float64")
    t = (np.arange(1, n + 1) / float(b - a))[:, None, None]
    xy = (1.0 - t) * pa[None, :, :2] + t * pb[None, :, :2]
    conf = np.minimum(pa[:, 2], pb[:, 2])[None, :, None].repeat(n, 0)
    return np.concatenate([xy, conf], axis=2)


def rellena(poses: np.ndarray, largo_max: int) -> dict:
    """QUE HACE: rellena los huecos interiores hasta `largo_max` y devuelve poses + marca.

    QUE DEVUELVE: `poses` con los huecos rellenos, `interpolado` (T, 2) booleano marcando lo
    inventado, y el detalle de lo que se relleno y lo que no.
    """
    P = np.array(poses, dtype="float64", copy=True)
    T = P.shape[0]
    u = utiles(P)
    marca = np.zeros((T, 2), dtype=bool)
    detalle = []
    for lado in (0, 1):
        for h in huecos(u[:, lado]):
            if h["tipo"] != "interior":
                detalle.append({**h, "lado": lado, "accion": "declarado: hueco de borde"})
                continue
            if h["largo"] > largo_max:
                detalle.append({**h, "lado": lado,
                                "accion": "declarado: %d cuadros > limite %d" % (h["largo"], largo_max)})
                continue
            P[h["ini"]:h["fin"], lado] = interpola_tramo(P, h["a"], h["b"], lado)
            marca[h["ini"]:h["fin"], lado] = True
            detalle.append({**h, "lado": lado, "accion": "interpolado"})
    return {"poses": P, "interpolado": marca, "detalle": detalle,
            "utiles_antes": int(u.sum()), "utiles_despues": int(utiles(P).sum()),
            "total": int(T * 2)}


def valida_escondiendo(poses: np.ndarray, boxes: np.ndarray, largos=(1, 2, 3, 5, 8, 12, 20, 30)) -> list:
    """QUE HACE: la validacion que no necesita anotacion humana.

    Busca tiradas de cuadros BUENOS, esconde los de dentro, interpola desde los extremos y compara
    con lo que realmente habia. El error se normaliza por el ALTO DE CAJA para que 3 px signifiquen
    lo mismo en un plano corto y en uno largo.

    Devuelve una fila por (largo escondido, muestra) con el error mediano y el maximo.
    """
    P = np.asarray(poses, dtype="float64")
    B = np.asarray(boxes, dtype="float64")
    u = utiles(P)
    fuera = []
    for lado in (0, 1):
        v = u[:, lado]
        buenos = np.nonzero(v)[0]
        if len(buenos) < 5:
            continue
        for L in largos:
            paso = max(L + 2, 4)
            for k in range(0, len(buenos) - (L + 1), paso):
                a = buenos[k]
                b = buenos[k + L + 1] if k + L + 1 < len(buenos) else None
                if b is None or (b - a) != L + 1:
                    continue
                """Solo vale si los cuadros escondidos son CONSECUTIVOS y todos buenos: si no, no se
                estaria midiendo un hueco de longitud L."""
                if not v[a:b + 1].all():
                    continue
                falso = interpola_tramo(P, a, b, lado)
                real = P[a + 1:b, lado]
                alto = B[a + 1:b, lado, 3] - B[a + 1:b, lado, 1]
                ok = np.isfinite(real[:, CLAVE, 0]) & np.isfinite(falso[:, CLAVE, 0]) & (alto[:, None] > 0)
                if not ok.any():
                    continue
                d = np.linalg.norm(real[:, CLAVE, :2] - falso[:, CLAVE, :2], axis=2)
                rel = d / np.maximum(alto[:, None], 1e-9)
                fuera.append({"lado": lado, "largo": L, "a": int(a), "b": int(b),
                              "err_med": float(np.nanmedian(rel[ok])),
                              "err_max": float(np.nanmax(rel[ok]))})
    return fuera


if __name__ == "__main__":
    print(__doc__)
    print("Este modulo no se ejecuta solo: lo usa `interpola_valida.py`, que mide sobre el corpus.")

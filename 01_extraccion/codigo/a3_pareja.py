"""Elige la PAREJA de tiradores, no dos trayectorias sueltas. Corrige un fallo medido de la puerta.

EL FALLO QUE ESTO ARREGLA, medido en 5 clips: la puerta de primer plano anclada en UNA trayectoria
—la mejor puntuada— rompio la seleccion en 2 de 5 clips, bajando la presencia de los dos de 99,7 % a
0,6 % y de 64,0 % a 2,0 %. Si la mejor puntuada resulta ser alguien mas cerca de camara que los
tiradores, el umbral echa a los tiradores. Anclar en un individuo es fragil por construccion.

LA IDEA: los dos tiradores estan **a la misma profundidad**, uno enfrente del otro. Asi que su y del
pie **coincide entre si**, y eso es una propiedad de la PAREJA que ningun individuo tiene. Un
espectador pegado a la camara no tiene pareja a su altura; el arbitro tampoco, porque esta mas atras.

Medido: los dos tiradores dieron y del pie 0,852 y 0,856 (diferencia **0,004**), el arbitro 0,756
(diferencia con ellos **0,10**) y el fondo 0,63-0,65 (diferencia **0,20**). La diferencia dentro de la
pareja es 25 veces menor que con el arbitro.

QUE SE PUNTUA EN UNA PAREJA:
  las puntuaciones individuales, que ya premian duracion, talla, banda y movimiento
  **la coincidencia de la y del pie**, que es lo nuevo y lo que arregla el fallo
  **la solapa temporal**: los dos tiradores estan en pista a la vez, no uno detras del otro
  y la restriccion DURA de lados opuestos, que ya estaba
"""
from __future__ import annotations

import numpy as np

import a3_selector as SEL

TOL_PIE = 0.05
"""Diferencia maxima de y del pie dentro de la pareja. Medido: entre tiradores 0,004 y con el arbitro
0,10. 0,05 cae en medio y no exige una coincidencia perfecta, que la perspectiva no garantiza cuando
uno esta en fondo y el otro adelantado."""


def rasgos(t, W, H, T):
    r = SEL.puntua(t, W, H, T)
    c = np.asarray(t["cajas"], dtype="float64")
    ks = np.asarray(t["cuadros"], dtype="int64")
    r["pie"] = float(np.median(c[:, 3])) / H
    r["cuadros_set"] = set(int(x) for x in ks)
    r["t"] = t
    return r


def elige_pareja(trayectorias, W, H, T, min_frac=0.02) -> dict:
    """QUE HACE: busca la MEJOR PAREJA entre todas las combinaciones, no las dos mejores sueltas.

    Descarta de entrada las trayectorias muy cortas para no explotar la combinatoria: con menos del
    2 % de los cuadros no puede ser un tirador de este asalto.

    QUE DEVUELVE: `elegidas` con dos entradas ordenadas izquierda-derecha, `parejas` con las mejores
    para poder ver por que se eligio esa, y `motivo`.
    """
    cand = [rasgos(t, W, H, T) for t in trayectorias]
    cand = [r for r in cand if r["n"] >= max(3, int(min_frac * T))]
    cand.sort(key=lambda r: -r["puntos"])
    if len(cand) < 2:
        return {"elegidas": cand[:1], "parejas": [], "candidatos": cand,
                "motivo": "menos de dos trayectorias utilizables"}

    """Se limita a las 12 mejores: con 30 trayectorias serian 435 parejas y las de puntuacion baja
    no van a ganar nunca."""
    top = cand[:12]
    parejas = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i], top[j]
            if (a["cx_med"] - 0.5) * (b["cx_med"] - 0.5) >= 0:
                continue
            dpie = abs(a["pie"] - b["pie"])
            if dpie > TOL_PIE:
                continue
            comun = len(a["cuadros_set"] & b["cuadros_set"])
            solapa = comun / float(max(len(a["cuadros_set"]), len(b["cuadros_set"]), 1))
            sep = abs(a["cx_med"] - b["cx_med"])
            """`frente` = lo cerca de camara que esta la pareja, por la y del pie y por la talla.
            Es la pieza que resuelve el caso de varias pistas a la vez: la pareja del asalto que
            toca es la mas adelantada, y las de las pistas del fondo quedan por debajo."""
            frente = 0.5 * (a["pie"] + b["pie"])
            talla = 0.5 * (a["alto_med"] + b["alto_med"])
            p = (0.30 * (a["puntos"] + b["puntos"]) / 2.0
                 + 0.15 * (1.0 - dpie / TOL_PIE)
                 + 0.15 * solapa
                 + 0.05 * min(sep / 0.35, 1.0)
                 + 0.25 * frente
                 + 0.10 * min(talla / 0.35, 1.0))
            parejas.append({"p": float(p), "a": a, "b": b, "dpie": float(dpie),
                            "solapa": float(solapa), "sep": float(sep),
                            "frente": float(frente), "talla": float(talla)})
    if not parejas:
        """Sin pareja valida no se fuerza una: se devuelven las dos mejores sueltas y se DICE que la
        restriccion no se pudo cumplir, para que el aviso llegue al informe."""
        a, b = cand[0], cand[1]
        izq, der = (a, b) if a["cx_med"] <= b["cx_med"] else (b, a)
        return {"elegidas": [izq, der], "parejas": [], "candidatos": cand,
                "motivo": "AVISO: ninguna pareja cumple lados opuestos y pie parecido"}
    parejas.sort(key=lambda x: -x["p"])
    mej = parejas[0]
    a, b = mej["a"], mej["b"]
    izq, der = (a, b) if a["cx_med"] <= b["cx_med"] else (b, a)
    return {"elegidas": [izq, der], "parejas": parejas[:5], "candidatos": cand,
            "motivo": "pareja: dpie %.3f solapa %.2f sep %.2f frente %.3f talla %.3f" % (
                mej["dpie"], mej["solapa"], mej["sep"], mej["frente"], mej["talla"])}

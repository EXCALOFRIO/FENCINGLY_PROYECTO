"""Elige la pareja de tiradores FILMADA cuando hay varias pistas a la vez. Sustituye a `a3_pareja`.

QUE FALLABA, medido en `ZanyMeagerIndianrockpython`, el clip de cuatro pistas simultaneas donde la
pareja buena se verifico a ojo en el cuadro 83 (FREILICH a la izquierda, FICHERA a la derecha):

  FALLO 1, LA RESTRICCION DE LADOS RESPECTO AL CENTRO DE IMAGEN. Los dos tiradores filmados estaban
  los dos en la mitad derecha del cuadro (cx 0,645 y 0,785), porque la camara encuadra su pista a la
  derecha. La regla "uno a cada lado de 0,5" los PROHIBIA por construccion, y en su lugar premiaba
  parejas cruzadas entre pistas distintas: la ganadora fue 1-5, con los dos a **5,58 alturas de
  cuerpo** de distancia. Nadie se bate a 5,58 alturas.
  El arreglo: izquierda y derecha son RELATIVAS entre los dos, no respecto al centro del cuadro.

  FALLO 2, FRAGMENTACION. El tirador de la derecha estaba partido en dos trayectorias, 48-144 y
  142-249, y se elegia solo una, asi que "los dos presentes" no podia pasar del 40 %. En la costura
  el centro se movia 0,20 del ancho de la caja y la altura cambiaba un 1 %: es la misma persona.

LA SENAL QUE ORDENA BIEN, tambien medida: CUADROS COMPARTIDOS. La camara sigue a la pareja del
asalto, asi que estan juntos en plano casi todo el clip; las parejas de las pistas del fondo entran
y salen. Con la fusion hecha, la pareja filmada comparte el 81 % de los cuadros y la pareja real de
la pista del fondo (1-9, tambien dos tiradores de verdad, no basura) solo el 22 %.

UNA MEDIDA QUE PROBE Y NO SIRVE, para que no se reintente: calcular `dpie` sobre los cuadros
COMPARTIDOS en vez de entre medianas. La sospecha era que el paneo de camara invalidaba la
comparacion. Medido, cambia de 0,083 a 0,070 en la pareja buena: no es el problema. Se deja la
version pareada igualmente porque es la comparacion correcta, pero NO es lo que arregla nada.
"""
from __future__ import annotations

import itertools

import numpy as np

import a3_selector as SEL

SEP_ALTURAS = (0.15, 3.20)
"""Separacion entre los dos, en alturas de cuerpo. Un tirador en guardia esta a unos 2 m del otro y
mide 1,75 m; en el fondo del blanco se juntan. Por debajo de 0,15 son dos cajas sobre la MISMA
persona.

EL LIMITE DE ARRIBA SUBIO DE 2,60 A 3,20 por una medida, no por gusto. Con 2,60, tres de doce clips
caian en la via de reserva porque su pareja verdadera, verificada a ojo por el rotulo del video,
quedaba fuera: `GentleQuerulousElephantseal` (TOLDO contra GIACON) da 2,84 y `DeficientNippyGar`
(RUSSO contra VECCHI) 2,82. Son planos anchos y en angulo, donde la pista se ve casi de frente.
El margen sigue siendo grande contra el caso que hay que rechazar: la pareja cruzada entre dos
pistas distintas de `ZanyMeager` da 5,58, o sea **1,74x** por encima del limite nuevo."""
DPIE_REL_MAX = 0.55
"""Diferencia de y del pie DIVIDIDA POR LA TALLA. Es la version invariante a la perspectiva, y
sustituye al limite absoluto, que era irreparable.

POR QUE EL ABSOLUTO NO PODIA FUNCIONAR, medido: en `WhisperedTenderDeinonychus` la camara mira a lo
largo de la pista, asi que los dos tiradores estan de verdad a distancias distintas y su `dpie`
absoluto es **0,220** — MAS ALTO que el de la peor pareja intrusa medida en `ZanyMeager`, que es
0,180. Los dos grupos se solapan y ningun umbral absoluto los separa.

Dividir por la talla lo arregla porque la perspectiva mueve las dos cosas a la vez: si un tirador
esta mas lejos, su caja se encoge Y sus pies suben, en proporcion.

  parejas verificadas a ojo por el rotulo del video ..... 0,027 a **0,451**
  parejas intrusas verificadas en `ZanyMeager` .......... **0,688** a 0,997

Margen **1,53x**, y el limite cae en medio."""
DPIE_ABS_MAX = 0.30
"""Tope absoluto de seguridad, para que una talla enorme no cuele cualquier cosa por la puerta
relativa. No es el que decide: la pareja verificada mas extrema se queda en 0,220."""
"""Diferencia de y del pie dentro de la pareja. Los dos estan a la misma profundidad.

SUBIO DE 0,10 A 0,14 por la misma medida: `AdvancedDefiniteAvocet` (EBERT contra NISHIOKA) da 0,106
y `DeficientNippyGar` 0,102, y las dos parejas estan verificadas a ojo. Con la camara alta y en
angulo, el tirador de un lado queda bastante mas abajo en el encuadre que el otro.

El margen se calculo sobre las 32 parejas medidas en `ZanyMeager`: aplicando ya las puertas de
separacion, la pareja MALA con el `dpie` mas bajo es tirador con espectador de fondo, a 0,180, que
son **1,29x** el limite nuevo. Las tres parejas malas con `dpie` entre 0,138 y 0,146 no las tapa
este limite sino la puerta de separacion: dos son cajas duplicadas sobre la misma persona (0,08
alturas) y la otra cruza dos pistas (3,77 alturas)."""
CRUZA_MAX = 0.25
"""Fraccion de cuadros en que se invierte quien esta a la izquierda. Dos tiradores no se atraviesan.
Medido: pareja filmada 0,00, dos cajas duplicadas sobre la misma persona 0,93."""
FRENTE_MIN = 0.40
"""Y del pie de la pareja, minima. Los tiradores estan EN LA PISTA, y la pista esta en la parte baja
del encuadre porque la camara mira desde arriba.

QUE AVERIA PARA, encontrada por el buscador automatico en `WhisperedTenderDeinonychus`: la pareja
elegida eran **dos personas de pie detras de la valla**, al borde de arriba del cuadro, con la y del
pie en 0,185, mientras los dos tiradores de verdad (PASZTOR y DUBROVICH) ocupaban el centro. Ganaron
porque comparten el 77 % de los cuadros contra el 56 % de la pareja buena, y `comparte` pesa 0,45 en
la puntuacion. Un termino de puntuacion no puede arreglar esto: hace falta una puerta DURA.

El limite sale de las parejas verificadas a ojo por el rotulo del video, que dan `frente` de **0,565
a 0,795** (FREILICH-FICHERA 0,637, SHIKINE-IIMURA 0,773, MARTON-VECCHI 0,701, FOCONI-YUNES 0,795,
TOLDO-GIACON 0,767, VOLPI-DI FRANCISCA 0,609). El margen contra la pareja intrusa es **3,05x**."""
COMUN_MIN = 10
FUS_HUECO = (-3, 45)
"""Hueco admitido al fusionar. Negativo es solape: el rastreador puede dar dos cajas a la vez sobre
la misma persona durante un fondo, y ahi la fusion es correcta. Mas de 3 cuadros de solape son dos
personas distintas."""
FUS_IOU_QUIETO = 0.50
"""Solape de las cajas de la costura por el que se acepta la fusion SIN predecir movimiento.

POR QUE EXISTE ESTA SEGUNDA VIA, medido en `RichDefensiveBushbaby`: el tirador de la izquierda esta
partido en 0-120 y 141-366, y en la costura sus dos cajas dan **IoU 0,86 y diferencia de pie 0,004**.
Es obviamente la misma persona, parada 21 cuadros sin que el detector la viera. La via de prediccion
lo rechazaba con un residuo de 3,03 porque multiplica por 21 una velocidad estimada con 5 cuadros:
extrapolar lejos es peor que suponer que no se ha movido. Ese solo fallo costaba 141 de los 151
cuadros que le faltaban al clip."""
FUS_HORIZONTE = 8
"""Cuadros como maximo que se extrapola la velocidad. Mas alla, la prediccion es ruido amplificado."""
FUS_IOU_LARGO = 0.20
"""Solape exigido para fusionar por encima de `FUS_HUECO_MOVIL` cuadros de hueco.

QUE AVERIA ARREGLA, medida en `PowerfulAllLarva`, que se quedaba en el **51 %** de cuadros con los
dos: los dos tiradores estan partidos en un hueco de 43-44 cuadros, y sus dos mitades encajan sin
duda (`dpie` 0,033 y 0,046, razon de talla 0,87 y 1,02, IoU 0,32 y 0,22). Pero con hueco de 43 la
unica via abierta era la de la caja quieta, que pide IoU 0,50, y el tirador se habia movido un poco.
Resultado: la pareja elegida solo cubria la segunda mitad del clip.

Se pide menos solape que en la via quieta corta porque el hueco es largo, pero se siguen exigiendo
`dpie` y razon de talla, que son los que de verdad distinguen a la misma persona de otra: la fusion
peligrosa que hay que rechazar, tirador con espectador que pasa por delante, salta 0,27 de `dpie`
contra el 0,05 que se admite.

DOS MEDIDAS, Y LA PRIMERA ME ENGANO. Al meter esto a la vez que la via de recuperacion gratis, los
12 clips verificados empeoraron (minimo de 94,6 % a 81,6 %) y lo atribui a este umbral, asi que lo
subi a 0,30 y me quede sin el arreglo. Separando los dos cambios, la culpa era ENTERA de la via
gratis: con ella apagada, este umbral a 0,20 sube `PowerfulAllLarva` de 51,1 % a **75,9 %** y no
mueve ni un cuadro en `ZanyMeager`, `RichDefensive`, `GentleQuerulous`, `BlissfulTired`,
`WhiteMarried` ni `FirsthandSore`. La leccion es la de siempre: dos cambios en la misma tirada no se
pueden atribuir."""
FUS_HUECO_MOVIL = 15
"""Por la via de prediccion se sigue exigiendo el hueco corto de siempre. La via larga es solo la de
la caja quieta, que no depende de predecir nada."""
FUS_DESPL = 0.60
FUS_TALLA = (0.72, 1.38)
FUS_DPIE = 0.05
"""La profundidad no salta: si la y del pie cambia mas de esto en la costura, no es la misma persona.
Es lo que impide fusionar al tirador de la derecha con un espectador que pasa por delante (salto de
0,27), que era la fusion peligrosa."""


def _centro(b):
    return np.array([0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])], dtype="float64")


def _iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _ordena(t):
    o = np.argsort(np.asarray(t["cuadros"], dtype="int64"), kind="stable")
    return {"cuadros": [t["cuadros"][i] for i in o],
            "cajas": [t["cajas"][i] for i in o],
            "idx": [t["idx"][i] for i in o]}


def _velocidad(t, n=5):
    """QUE HACE: velocidad del centro en los ultimos cuadros, para predecir donde estara.

    Sin esto la fusion falla justo cuando mas hace falta: en un fondo el tirador se desplaza rapido,
    y es ahi donde el rastreador lo pierde y parte la trayectoria.
    """
    ks = np.asarray(t["cuadros"][-n:], dtype="float64")
    if len(ks) < 2:
        return np.zeros(2)
    cs = np.array([_centro(b) for b in t["cajas"][-n:]])
    dk = ks[-1] - ks[0]
    return (cs[-1] - cs[0]) / dk if dk > 0 else np.zeros(2)


def fusiona(trayectorias, W, H) -> list:
    """QUE HACE: une los trozos que son la misma persona partida por el rastreador.

    Voraz sobre los pares ordenados por residuo, y transitivo: si A se une con B y B con C, los tres
    quedan en uno. Se recalcula el extremo despues de cada union para que la cadena sea coherente.
    """
    ts = [_ordena(t) for t in trayectorias]
    vivos = list(range(len(ts)))
    cambio = True
    while cambio:
        cambio = False
        mejor, mejor_r = None, 1e9
        for i, j in itertools.permutations(vivos, 2):
            a, b = ts[i], ts[j]
            hueco = b["cuadros"][0] - a["cuadros"][-1]
            if not (FUS_HUECO[0] <= hueco <= FUS_HUECO[1]):
                continue
            if b["cuadros"][-1] <= a["cuadros"][-1]:
                continue
            ba = np.asarray(a["cajas"][-1], dtype="float64")
            bb = np.asarray(b["cajas"][0], dtype="float64")
            if abs(ba[3] / H - bb[3] / H) > FUS_DPIE:
                continue
            ra = (ba[3] - ba[1]) / max(bb[3] - bb[1], 1e-6)
            if not (FUS_TALLA[0] <= ra <= FUS_TALLA[1]):
                continue
            iou = _iou(ba, bb)
            if iou >= FUS_IOU_QUIETO:
                """VIA DE LA CAJA QUIETA: no se predice nada, se comprueba que sigue en el sitio."""
                res = 1.0 - iou
            elif hueco <= 0:
                """Se solapan en el tiempo: solo es la misma persona si las cajas casi coinciden."""
                if iou < 0.30:
                    continue
                res = 1.0 - iou
            elif hueco > FUS_HUECO_MOVIL:
                """VIA QUIETA LARGA: con el hueco largo no se predice nada y se pide solape, menos
                que en la corta pero con `dpie` y talla ya comprobados arriba."""
                if iou < FUS_IOU_LARGO:
                    continue
                res = 1.0 - iou
            else:
                """VIA DE PREDICCION, solo para huecos cortos y con la extrapolacion acotada."""
                pred = _centro(ba) + _velocidad(a) * min(hueco, FUS_HORIZONTE)
                res = float(np.linalg.norm(pred - _centro(bb))) / max(ba[2] - ba[0], 1e-6)
                if res > FUS_DESPL:
                    continue
            if res < mejor_r:
                mejor, mejor_r = (i, j), res
        if mejor is not None:
            i, j = mejor
            ts[i] = _ordena({"cuadros": ts[i]["cuadros"] + ts[j]["cuadros"],
                             "cajas": ts[i]["cajas"] + ts[j]["cajas"],
                             "idx": ts[i]["idx"] + ts[j]["idx"]})
            vivos.remove(j)
            cambio = True
    return [ts[i] for i in vivos]


def _series(t, W, H, T):
    pie = np.full(T, np.nan)
    cx = np.full(T, np.nan)
    al = np.full(T, np.nan)
    for k, b in zip(t["cuadros"], t["cajas"]):
        if 0 <= k < T:
            pie[k] = b[3] / H
            cx[k] = 0.5 * (b[0] + b[2]) / W
            al[k] = (b[3] - b[1]) / H
    return pie, cx, al


def rasgos(t, W, H, T):
    r = SEL.puntua(t, W, H, T)
    c = np.asarray(t["cajas"], dtype="float64")
    r["pie"] = float(np.median(c[:, 3])) / H
    """`borde` es la fraccion de cuadros de la trayectoria en que su caja esta CORTADA por el borde
    lateral del cuadro. Con 1,00 la persona no se vio entera ni una sola vez."""
    r["borde"] = float(np.mean((c[:, 0] <= 0.004 * W) | (c[:, 2] >= 0.996 * W)))
    r["t"] = t
    r["ser"] = _series(t, W, H, T)
    return r


BORDE_SIEMPRE = 1.0
"""Fraccion de cuadros con la caja cortada por el borde lateral a partir de la cual la trayectoria
se descarta como candidata a tirador.

POR QUE EL VALOR ES 1,00 Y NO UN UMBRAL AJUSTADO. La regla es definicional, no estadistica: si la
caja esta cortada por el borde en el 100 % de los cuadros, **la persona no se vio entera ni una sola
vez**, asi que no existe ninguna observacion con la que comprobar que es un tirador. No es que sea
poco probable: es que no hay dato.

QUE AVERIA PARA, verificada a ojo en `LavishDistantDikkops` (imagenes en
`02_validacion/muestras/limite_intruso`): el "tirador izquierdo" que se entregaba era un trozo de
ESPECTADOR pegado al borde izquierdo, con blancura 0,000, mientras el tirador real estaba en el plano
y sin caja. El clip daba 27,4 % de presencia y ningun otro check lo cazaba; en concreto paso la
puerta de tiras (`extrae_pareja.PROP_MIN`) porque su proporcion era 0,33, por encima de 0,25.

Y POR QUE NO SE BAJA, medido sobre 60 clips: esta fraccion tiene p50 0,067, p90 0,432, p95 0,592 y
p99 0,936, o sea que pasar mucho rato en el borde es NORMAL. El segundo valor mas alto es 0,892 en
`WastefulYawningAmericanquarterhorse`, y ahi el tirador de la derecha esta de verdad medio fuera de
cuadro en un plano ancho: **verificado a ojo como CORRECTO**. De los 60 clips solo uno llega a 1,00 y
es justo el roto. Bajar el umbral se llevaria casos buenos."""


def elige_pareja(trayectorias, W, H, T, min_frac=0.02, fusionar=True) -> dict:
    """QUE HACE: fusiona los trozos y elige la pareja FILMADA, no la mejor puntuada suelta.

    QUE DEVUELVE: `elegidas` en orden izquierda-derecha por su cx RELATIVO, `parejas` con las cinco
    mejores y sus terminos para poder contradecir la eleccion, `candidatos`, y `motivo`.
    """
    tr = fusiona(trayectorias, W, H) if fusionar else [_ordena(t) for t in trayectorias]
    cand = [rasgos(t, W, H, T) for t in tr]
    cand = [r for r in cand if r["n"] >= max(3, int(min_frac * T))]
    nunca_entera = [r for r in cand if r["borde"] >= BORDE_SIEMPRE]
    cand = [r for r in cand if r["borde"] < BORDE_SIEMPRE]
    cand.sort(key=lambda r: -r["puntos"])
    if len(cand) < 2:
        return {"elegidas": cand[:1], "parejas": [], "candidatos": cand, "nfus": len(tr),
                "borde_descartadas": len(nunca_entera),
                "motivo": "menos de dos trayectorias utilizables%s" % (
                    "" if not nunca_entera
                    else " (%d descartadas por no verse enteras nunca)" % len(nunca_entera))}

    top = cand[:14]
    parejas = []
    for i, j in itertools.combinations(range(len(top)), 2):
        a, b = top[i], top[j]
        pa, xa, aa = a["ser"]
        pb, xb, ab = b["ser"]
        m = np.isfinite(pa) & np.isfinite(pb)
        n = int(m.sum())
        if n < COMUN_MIN:
            continue
        alt = 0.5 * (aa[m] + ab[m])
        sepalt = float(np.median(np.abs(xa[m] - xb[m]) * W / np.maximum(alt * H, 1e-6)))
        dpie = float(np.median(np.abs(pa[m] - pb[m])))
        talla = float(np.median(alt))
        dpie_rel = dpie / max(talla, 1e-6)
        signo = np.sign(xa[m] - xb[m])
        cruza = float(np.mean(signo != np.median(signo)))
        if not (SEP_ALTURAS[0] <= sepalt <= SEP_ALTURAS[1]):
            continue
        if dpie_rel > DPIE_REL_MAX or dpie > DPIE_ABS_MAX or cruza > CRUZA_MAX:
            continue
        comparte = n / float(max(T, 1))
        frente = float(np.median(0.5 * (pa[m] + pb[m])))
        if frente < FRENTE_MIN:
            continue
        p = (0.45 * comparte
             + 0.25 * frente
             + 0.15 * min(talla / 0.35, 1.0)
             + 0.15 * (1.0 - dpie_rel / DPIE_REL_MAX))
        parejas.append({"p": float(p), "a": a, "b": b, "comparte": comparte, "sepalt": sepalt,
                        "dpie": dpie, "dpie_rel": dpie_rel, "cruza": cruza, "frente": frente,
                        "talla": talla,
                        "cx_a": float(np.median(xa[m])), "cx_b": float(np.median(xb[m]))})
    if not parejas:
        a, b = cand[0], cand[1]
        izq, der = (a, b) if a["cx_med"] <= b["cx_med"] else (b, a)
        return {"elegidas": [izq, der], "parejas": [], "candidatos": cand, "nfus": len(tr),
                "borde_descartadas": len(nunca_entera),
                "motivo": "AVISO: ninguna pareja pasa las puertas de geometria"}
    parejas.sort(key=lambda x: -x["p"])
    mej = parejas[0]
    izq, der = ((mej["a"], mej["b"]) if mej["cx_a"] <= mej["cx_b"] else (mej["b"], mej["a"]))
    return {"elegidas": [izq, der], "parejas": parejas[:5], "candidatos": cand, "nfus": len(tr),
            "borde_descartadas": len(nunca_entera),
            "motivo": ("pareja: comparte %.0f%% sep %.2f alturas dpie %.3f (rel %.2f) cruza %.2f "
                       "frente %.3f") % (
                100 * mej["comparte"], mej["sepalt"], mej["dpie"], mej["dpie_rel"], mej["cruza"],
                mej["frente"])}

"""La PISTA: homografia imagen -> metros, y la posicion de los dos tiradores sobre ella.

=======================================================================================
LO PRIMERO, PORQUE CAMBIA EL PUNTO DE PARTIDA: ESTO YA SE INTENTO Y SE CERRO EN 2026
=======================================================================================

El informe historico del proyecto (`99_historico/FENCINGLY/notas/scratchpad/informe/paper/main.tex`,
seccion `sec:abl-geometry`) trae la medida y el veredicto:

  | medida historica | valor |
  |---|---:|
  | cuadros de puesta en guardia en que la deteccion de lineas devolvia calibracion | **3,8 %** (19 de 500, 19 sedes) |
  | techo de un HUMANO asistido sobre las mismas tiras rectificadas | **4,3 %** (1 de 23) |
  | pista visible en un cuadro de television, mediana | **8,3 m** (p05 5,8, p95 11,2, n=348) |
  | cuadros que abarcan los 14 m completos | **0 de 348** |
  | tiras donde la banda se enganchaba a las vallas publicitarias | **6 de 23 (26 %)** |

Su conclusion fue correcta para lo que medio: **el detector ya estaba en el techo de la evidencia**,
o sea que era un problema de datos y no de algoritmo. Y este modulo NO contradice esa cifra: el paso
1 de hoy la reproduce, con 1 clip de 13 (8 %, intervalo binomial [0,2, 36]) donde hay dos o mas
LINEAS transversales completas. Ver `p1_censo_pista.py`.

LO QUE SI CAMBIA, Y ES UNA SOLA COSA: aquel estudio necesitaba **dos** marcas transversales, porque
con dos se obtiene el punto de fuga transversal y sin el no hay rectificacion metrica. Este modulo
necesita **una sola**, y esa diferencia es lo que hace la via viable. La razon es geometrica y no
estadistica:

  el punto de fuga transversal V2 esta, por definicion, EN cualquier recta transversal;
  y esta tambien en la LINEA DE HORIZONTE, que se obtiene gratis de los dos bordes
  longitudinales de la pista mas la hipotesis de camara sin balanceo;
  asi que V2 = (una marca transversal) x (horizonte), con UNA marca.

=======================================================================================
GEOMETRIA FIE, COMPROBADA EN EL MATERIAL DEL PROYECTO
=======================================================================================

Fichero `scratchpad/fie_tech.txt`, articulos t.18 y t.19, no de memoria:

  t.18.1  la pista mide de **1,50 a 2 metros** de ancho
  t.18.2  la pista mide **14 metros** de largo
  t.19.2  linea CENTRAL, discontinua, de lado a lado del ancho
  t.19.4  dos lineas de PUESTA EN GUARDIA a **2 metros** de cada lado del centro
  t.19    dos lineas de LIMITE POSTERIOR a **7 metros** del centro
  t.19    los ultimos **2 metros** antes del limite posterior deben distinguirse, si se puede por un
          COLOR distinto de pista

DOS CORRECCIONES AL ENUNCIADO CON EL QUE SE PIDIO ESTA TAREA, las dos sacadas del reglamento:

  1. NO existe una "linea de aviso" pintada a 5 m del centro. Lo que existe es que la ZONA de los
     ultimos 2 m se distinga, normalmente por color. La frontera a 5 m del centro es por tanto una
     frontera de COLOR y no una recta pintada: un detector de rectas no la va a encontrar nunca, y
     un detector de region si.
  2. El ancho NO es un dato fijo: el reglamento admite de 1,50 a 2,00 m, o sea un **33 % de rango**.
     Eso importa mucho porque el ancho es la unica escala metrica que la pista ofrece siempre, asi
     que la escala trae ese 33 % de incertidumbre de fabrica. Se trata mas abajo.

Y el aviso del usuario, que va en la misma direccion: *"algunas pistas pueden variar algo la longitud
o el ancho por tener alguna plancha extra"*. Por eso este modulo **NO fuerza 14,00 m en ningun
sitio**: ajusta con lo que ve y luego MIDE el largo y el ancho que salen.

=======================================================================================
LA CONSTRUCCION, Y QUE HIPOTESIS ENTRA EN CADA PASO
=======================================================================================

  paso                        entrada                                hipotesis que anade
  --------------------------  -------------------------------------  ------------------------------
  1 mascara de rotulos        varianza temporal ~ 0                  el rotulo de television no
                                                                     cambia; el suelo si tiene ruido
  2 estabilizacion            flujo optico fuera de las personas     el fondo es rigido y plano en
                                                                     el trozo que se usa
  3 placa limpia              mediana temporal con personas tapadas  una persona se mueve, una raya
                                                                     pintada no
  4 region de la pista        color desde el corredor de los pies    la pista es de un color propio
  5 los dos BORDES            barrido perpendicular al eje           los bordes son rectos
  6 horizonte                 V1 = borde x borde, horizontal         camara SIN BALANCEO
  7 marcas transversales      filtro de cresta dentro de la region   una marca es una cresta clara
  8 V2                        marca x horizonte                      la marca es perpendicular al eje
  9 homografia                V1, V2, bordes, ancho w                pixel cuadrado, punto principal
                                                                     en el centro
 10 escala                    w en metros                            declarada, no medida

CONTROLES, y ninguno de los tres usa nada de lo que se ajusta:

  C1  ESTATURA REPROYECTADA de los dos tiradores. Tiene que caer en 1,60-2,00 m. Los tiradores no
      entran en el ajuste, asi que este control es independiente de verdad. Es el que separa "la
      pista es rara" de "la homografia esta mal", que es lo que pedia el usuario.
  C2  SEPARACION ENTRE MARCAS reproyectada, cuando hay dos o mas marcas. Tiene que caer cerca de un
      multiplo de las distancias FIE (2,00 · 4,00 · 5,00 · 7,00 m). Con dos marcas, una fija V2 y la
      otra queda como control.
  C3  PERPENDICULARIDAD de las marcas que NO se usaron para V2, en grados sobre el plano
      rectificado. Es el control directo de la hipotesis de camara sin balanceo y del punto
      principal.

Y lo que NO es un control: que la pista reproyecte a un rectangulo. Sale rectangulo por
construccion, asi que comprobarlo no dice nada.

=======================================================================================
LA LINEA BASE QUE HAY QUE BATIR, DECLARADA ANTES DE MEDIR
=======================================================================================

Gratis y ya medida en el proyecto: `x` del centro de la caja y `y` del borde inferior de la caja,
normalizadas 0-1, como proxy de posicion longitudinal y de profundidad. La `y del pie` ORDENA la
profundidad al 100 % en 71 cuadros y acierta 28 de 28 contra `pointmap`, asi que como ORDEN es
imbatible y no se pretende batirla en eso.

Lo que la linea base NO da, y es lo unico que esta homografia tiene que aportar:

  - metros. La linea base da fracciones de imagen, que no se pueden sumar ni comparar entre clips.
  - linealidad. En la linea base, un metro cerca de la camara son muchos mas pixeles que un metro
    lejos, asi que "cuanto ha medido el fondo" sale mal por construccion.
  - el eje transversal. `y del pie` mezcla profundidad y posicion, porque la pista esta en angulo.

Asi que el criterio para decidir si esto sirve, declarado ANTES de correrlo:

  la homografia se publica solo si el control C1 (estatura reproyectada) cae en 1,60-2,00 m,
  y el ruido de posicion por cuadro (p95) queda por debajo de 0,24 m, que es lo que un
  esgrimista puede moverse de verdad en un cuadro a 25-30 fps (un fondo son 1,0-1,5 m en
  0,2-0,3 s, o sea 4-6 m/s).

Si no lo cumple, se dice, se deja escrito por que, y se entrega la linea base con su nombre.
"""
from __future__ import annotations

import gc
import json
import os
import pathlib
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import extrae_pareja as EP

import rutas as RU

SCRATCH_VIEJO = RU.TRABAJO
"""DONDE ESTA `det_<clip>.pkl`. Antes esto apuntaba a un scratchpad temporal de OTRA sesion
mientras la etapa 1 escribia en `01_extraccion/pruebas`. Medido el dia que se arreglo: en
`pruebas` habia **0** pickles y en el scratchpad **117**, asi que cablear la etapa 7 habria
devuelto `sin pickle de detecciones` en todos los clips -- un fallo blando, que imprime una linea
ordenada y no escribe nada. Las dos puntas leen y escriben ahora la MISMA raiz, `rutas.TRABAJO`."""

SCRATCH = RU.SCRATCH

FIE_ANCHO = (1.50, 1.50)
"""Ancho de la pista segun t.18.1, y HAY DOS EDICIONES DEL REGLAMENTO EN DISCO QUE NO DICEN LO MISMO.
Comprobado linea a linea en las dos, que estan en `scratchpad/`:

  `fie_tech.txt`, **diciembre 2017**, t.18.1: *"The piste is from 1.50 metres to 2 metres wide."*
  `fie_2021.txt`, **diciembre 2021**, t.18.1: *"The combat area of the piste is 1.50 meters wide."*

La edicion de 2021 es la que aplica a este corpus, asi que **el ancho es 1,50 m y no un rango**. Y la
diferencia entre las dos redacciones no es un cambio de medida: la de 2021 anade en t.18.3 que los
*conductive safety borders* NO forman parte de la pista. O sea que la banda que se ve en el video
puede ser mas ancha que la pista, porque incluye los bordes de seguridad, y lo que mide 1,50 m es la
ZONA DE COMBATE.

CONSECUENCIA PARA ESTE MODULO, y es una fuente de sesgo que hay que tener delante: el barrido de
color detecta la banda por su COLOR, y si los bordes de seguridad son del mismo color que la pista, el
barrido los incluye y la escala sale pequenia; si el barrido se para dentro de la pista, sale grande.
Por eso el ancho deducido se informa siempre, y por eso NO es un control independiente de la
estatura: los dos salen del mismo numero."""
FIE_LARGO = 14.00
"""Largo nominal de t.18.2. Se usa SOLO para comparar con lo medido, nunca para ajustar."""
FIE_MARCAS = (0.00, 2.00, 5.00, 7.00)
"""Distancias al centro de las marcas que existen: centro (0), puesta en guardia (2), frontera de
color de los ultimos 2 m (5) y limite posterior (7). En metros y en valor absoluto."""
ANCHO_NOMINAL = 1.50
"""Ancho que se usa como escala transversal, en metros: la zona de combate de t.18.1 en la edicion de
diciembre de 2021. La consecuencia de equivocarse esta acotada y es LINEAL: todas las distancias
metricas del clip se multiplican por el error del ancho. Por eso el control que decide (la estatura
reproyectada) detecta un error del ancho del 15 % o mas."""

STD_ROTULO = 2.2
"""Desviacion temporal por debajo de la cual un pixel se considera rotulo de television. Medido en
`__main__`: hay que comprobar que separa antes de creerlo."""
N_PLACA = 28
"""Cuadros que entran en la mediana temporal de la placa.

BAJADO DE 48 A 28 POR COSTE MEDIDO, no por gusto: con 48 el paso 2 tardaba **5 minutos por clip**, o
sea 7 horas para 78 clips, y la parte pesada era apilar 48 fotogramas en float32 (530 MB a 720p y
1,2 GB a 1080p) para hacerles la mediana. Con 28 y la mediana por franjas de filas baja a memoria
constante. La placa no se degrada: lo que la mediana tiene que quitar son personas que se mueven, y
28 muestras repartidas por el clip ya dejan cada pixel de suelo visible en la mayoria."""
FRANJA = 96
"""Filas por franja al calcular la mediana temporal. Es lo que convierte el pico de memoria de
gigabytes en decenas de megabytes sin cambiar el resultado."""
DILATA = 0.06
MAHAL2 = 9.0
"""Distancia de Mahalanobis al cuadrado que marca el limite de la pista en el barrido perpendicular.
9,0 son 3 desviaciones en el color LAB del propio corredor."""


def _construye_H(V1, V2, f, LBc, Oc, ancho_m, pp) -> dict:
    """QUE HACE: monta la homografia a partir de los dos puntos de fuga, la focal, el origen y el
    ancho, y la descompone para sacar la altura de camara y el punto de fuga vertical con su escala.

    ESTA SEPARADO PARA PODER LLAMARLO DESDE EL BARRIDO DE FOCAL. Sin eso no se puede poner la puerta
    de altura de camara donde hace falta, que es DENTRO del barrido: si se pone despues, el barrido ya
    ha elegido una focal imposible y lo unico que se puede hacer es rechazar el clip entero.

    LAS TRES COLUMNAS. Las dos primeras son las imagenes de los puntos del infinito de los ejes de la
    pista, o sea los puntos de fuga; la tercera es la imagen del origen. `beta` sale de que el borde B
    este a `ancho_m` del A, y `alfa` de que el metro valga igual en las dos direcciones, que es
    |K^-1 h1| = |K^-1 h2|.

    EL SIGNO DE H NO ES LIBRE, y fue un fallo: una homografia esta definida salvo escala, signo
    incluido, y para llevar puntos de un lado a otro el signo da igual, pero para medir una ALTURA no.
    La relacion exacta es p(h) = H(X,Y,1) + mu*h*(K r3) con A = K^-1 H = mu [r1 r2 t]; si el signo de
    H es el contrario, `mu` sale negativo y todas las alturas salen negativas. Se vio como un control
    de estatura con n=6 cajas de varios cientos y una estatura de 0,39 m. Se fija exigiendo que un
    punto de DENTRO de la pista tenga tercera coordenada positiva, o sea que este delante de la
    camara.
    """
    n1 = _norm_k(V1, f)
    n2 = _norm_k(V2, f)
    denb = float(LBc @ V2)
    if abs(denb) < 1e-12:
        return {"ok": False, "motivo": "V2 esta sobre el borde B"}
    beta = -float(LBc @ Oc) / (ancho_m * denb)
    alfa = beta * n2 / max(n1, 1e-12)
    Hc = np.stack([alfa * V1, beta * V2, Oc], axis=1)
    Tpp = np.array([[1.0, 0.0, pp[0]], [0.0, 1.0, pp[1]], [0.0, 0.0, 1.0]])
    Hm = Tpp @ Hc
    if float((Hm @ np.array([0.0, 0.5 * ancho_m, 1.0]))[2]) < 0:
        Hm = -Hm
    K = np.array([[f, 0.0, pp[0]], [0.0, f, pp[1]], [0.0, 0.0, 1.0]])
    try:
        Hi = np.linalg.inv(Hm)
    except np.linalg.LinAlgError:
        return {"ok": False, "motivo": "homografia singular"}
    A = np.linalg.inv(K) @ Hm
    mu = 0.5 * (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
    if mu < 1e-12:
        return {"ok": False, "motivo": "descomposicion degenerada"}
    r1, r2 = A[:, 0] / mu, A[:, 1] / mu
    r3 = np.cross(r1, r2)
    tv = A[:, 2] / mu
    hc = float(abs(r3 @ tv))
    vzh = mu * (K @ r3)
    """El sentido de r3 tampoco es libre: apunta arriba o abajo segun cual de los dos bordes salio
    como A. Se comprueba subiendo un metro desde la pista y mirando si en la imagen sale mas arriba."""
    p0 = Hm @ np.array([0.0, 0.5 * ancho_m, 1.0])
    p1 = p0 + 1.0 * vzh
    if p1[2] != 0 and p0[2] != 0 and (p1[1] / p1[2]) > (p0[1] / p0[2]):
        vzh = -vzh
    return {"ok": True, "H": Hm, "Hi": Hi, "hc_m": hc, "vz_h": vzh, "mu": mu, "K": K,
            "alfa": alfa, "beta": beta}


def _pp(W, H):
    """QUE HACE: el punto principal supuesto, el centro de la imagen.

    Es una hipotesis, no un dato. El control C3 la comprueba: si el punto principal estuviera muy
    descentrado, las marcas transversales que no entraron en el ajuste reproyectarian torcidas.
    """
    return np.array([0.5 * W, 0.5 * H], "float64")


def indice_videos() -> dict:
    """QUE HACE: el mapa `stem` -> ruta del mp4, cacheado."""
    return RU.indice_videos()


def video_clip(clip):
    """La ruta del mp4 de UN clip: `RU.video`, con la misma precedencia que `indice_videos()` (trabajo,
    descargas, arbol) y sin rehacer el mapa entero en cada clip (2026-10-13)."""
    return RU.video(clip)


def pareja(clip: str, frames=None):
    """QUE HACE: los dos tiradores desde el pickle de detecciones ya cacheado, sin volver a pasar
    YOLO.

    `frames` hace falta para que la eleccion sea la MISMA que la de la etapa 1 cuando la puerta de
    piernas esta encendida (`pareja_v2.PIERNA_K`), porque esa puerta mira pixeles. Sin ellos, aqui
    saldria la pareja de antes de la puerta y las poses de la etapa 4 serian de otra gente."""
    pk = SCRATCH_VIEJO / ("det_%s.pkl" % clip)
    if not pk.exists():
        return None
    with open(pk, "rb") as fh:
        cajas, cuadros, W, H, T = pickle.load(fh)
    r = EP.procesa(cajas, cuadros, W, H, T, frames=frames)
    r["cajas"], r["cuadros"], r["W"], r["H"], r["T"] = cajas, cuadros, W, H, T
    return r


def lee(video: str, T: int) -> list:
    """QUE HACE: los fotogramas del clip. Se leen una vez y se reutilizan."""
    cap = cv2.VideoCapture(video)
    fs = []
    while len(fs) < T:
        ok, im = cap.read()
        if not ok:
            break
        fs.append(im)
    cap.release()
    return fs


def mascara_personas(cajas, cuadros, k, W, H) -> np.ndarray:
    """QUE HACE: 1 donde NO hay persona en el cuadro `k`, 0 donde hay.

    Se tapan TODAS las detecciones de YOLO del cuadro completo, no solo las de los dos tiradores.
    Eso incluye al arbitro y al publico, que es justo lo que hay que quitar del suelo. Cuesta 0 ms
    porque las detecciones ya estan en el pickle. La alternativa, `seg` de Sapiens 2, cuesta 245 ms
    por cuadro y solo cubre las cajas que se le den.
    """
    t = np.ones((H, W), "uint8")
    for i in np.nonzero(cuadros == k)[0]:
        x0, y0, x1, y1 = [float(v) for v in cajas[i]]
        m = DILATA * max(x1 - x0, 1.0)
        cv2.rectangle(t, (int(x0 - m), int(y0 - m)), (int(x1 + m), int(y1 + m)), 0, -1)
    return t


REDETECTA_PUNTOS = 10
"""Cada cuantos cuadros se vuelven a buscar puntos de seguimiento en vez de arrastrar los que ya
hay. Arrastrarlos ahorra la mitad del coste de la estabilizacion y no degrada la estimacion, porque
la similitud se ajusta con RANSAC sobre los que sobreviven."""
RAZON_ROTULO = 2.5
"""Razon (desviacion en el sistema estabilizado) / (desviacion en el sistema de pantalla) por encima
de la cual un pixel es rotulo de television.

POR QUE UNA RAZON Y NO UN UMBRAL ABSOLUTO, y es un fallo medido de mi primera version: con el umbral
absoluto de desviacion baja marcaba el 21,8 % de `GentleQuerulousElephantseal` y el **46,2 %** de
`UglyNiceAplomadofalcon` como rotulo, cuando el rotulo real ocupa del orden del 5 %. La causa: el
suelo azul liso de un pabellon tiene tan poco ruido de compresion como un grafico.

LA SENAL CORRECTA sale de que el rotulo esta pegado a la PANTALLA y el suelo esta pegado al MUNDO.
Si la camara se mueve, un pixel de suelo cambia mucho en pantalla y nada en el sistema
estabilizado; un pixel de rotulo hace lo contrario. La razon entre las dos desviaciones separa los
dos casos sin depender de cuanto ruido tenga el suelo.

Y SU LIMITACION, que hay que declarar: si la camara NO se mueve, las dos desviaciones son iguales,
la razon vale 1 y esta senal no discrimina nada. En ese caso se cae al umbral absoluto exigiendo
ademas que la region TOQUE EL BORDE del cuadro, porque los rotulos de television viven en los bordes
y una mancha de suelo liso en medio del cuadro no."""


def mascara_rotulos(frames, sel, Ms, cajas, cuadros, W, H, dmax) -> tuple:
    """QUE HACE: la mascara de los rotulos de television, comparando la varianza temporal en el
    sistema de la pantalla con la del sistema estabilizado.

    POR QUE HACE FALTA, y lo encontro el prototipo: en `GentleQuerulousElephantseal` la barra del
    marcador tapa el borde CERCANO de la pista, asi que el ancho medido sale truncado y la homografia
    sale mal sin dar ningun aviso. Un rotulo no es suelo y hay que saber que esta ahi.

    Devuelve la mascara (1 = rotulo), la mediana de la desviacion de pantalla fuera de ella y la via
    que se ha usado, que hay que saber para leer la cifra.
    """
    """ACUMULACION INCREMENTAL Y NO APILADO, y es una correccion de COSTE medida: apilando los 28
    fotogramas en float32 dos veces (sistema de pantalla y sistema estabilizado) el proceso llegaba a
    **9,9 GB de memoria** en los clips de 1920x1080, la maquina empezaba a paginar y el paso 2 pasaba
    de 21 s a 5 min por clip. La media y la varianza no necesitan tener todo a la vez: se acumulan
    suma y suma de cuadrados por pixel, que son dos matrices y no veintiocho."""
    sp = np.zeros((H, W), "float64")
    sp2 = np.zeros((H, W), "float64")
    npn = np.zeros((H, W), "int32")
    se = np.zeros((H, W), "float64")
    se2 = np.zeros((H, W), "float64")
    nne = np.zeros((H, W), "int32")
    n_us = 0
    for j, k in enumerate(sel):
        if k >= len(frames):
            continue
        n_us += 1
        g = cv2.cvtColor(frames[k], cv2.COLOR_BGR2GRAY).astype("float32")
        t = mascara_personas(cajas, cuadros, k, W, H).astype(bool)
        sp[t] += g[t]
        sp2[t] += g[t].astype("float64") ** 2
        npn += t
        ge = cv2.warpPerspective(g, Ms[j], (W, H), flags=cv2.INTER_LINEAR)
        te = cv2.warpPerspective(t.astype("uint8") * 255, Ms[j], (W, H),
                                 flags=cv2.INTER_NEAREST) > 127
        se[te] += ge[te]
        se2[te] += ge[te].astype("float64") ** 2
        nne += te
    if n_us < 6:
        return np.zeros((H, W), "uint8"), float("nan"), "sin muestras"
    with np.errstate(all="ignore"):
        sdp = np.sqrt(np.maximum(sp2 / np.maximum(npn, 1) - (sp / np.maximum(npn, 1)) ** 2, 0.0))
        sde = np.sqrt(np.maximum(se2 / np.maximum(nne, 1) - (se / np.maximum(nne, 1)) ** 2, 0.0))
    sdp = np.where(npn >= 3, sdp, 0.0)
    sde = np.where(nne >= 3, sde, 0.0)
    if dmax >= 8.0:
        via = "razon"
        rot = ((sde > RAZON_ROTULO * np.maximum(sdp, 0.6)) & (sdp < 8.0)
               & (npn >= max(6, int(0.5 * n_us)))).astype("uint8")
    else:
        via = "absoluta+borde"
        rot = ((sdp < STD_ROTULO) & (npn >= max(6, int(0.5 * n_us)))).astype("uint8")
    rot = cv2.morphologyEx(rot, cv2.MORPH_OPEN, np.ones((5, 5), "uint8"))
    nl, lb, st, _c = cv2.connectedComponentsWithStats(rot, 8)
    keep = np.zeros_like(rot)
    for i in range(1, nl):
        if st[i, cv2.CC_STAT_AREA] < 0.0015 * W * H:
            continue
        x, y = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
        w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        toca = x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2
        if via == "absoluta+borde" and not toca:
            continue
        keep[lb == i] = 1
    keep = cv2.dilate(keep, np.ones((7, 7), "uint8"))
    fuera = sdp[(keep == 0) & (npn >= 6)]
    return keep, float(np.median(fuera)) if fuera.size else float("nan"), via


def estabiliza(frames, ks, cajas, cuadros, W, H, rot, ref=None) -> tuple:
    """QUE HACE: la transformacion de similitud que lleva cada cuadro de `ks` al de referencia.

    POR QUE HACE FALTA. Un clip con paneo tiene una homografia distinta en cada cuadro, y la placa
    limpia sale con las rayas de la pista DUPLICADAS, lo que las borra como evidencia. Se vio en el
    prototipo: la placa de `GentleQuerulousElephantseal` sale con las bandas rojas dobladas.

    COMO. Puntos de Shi-Tomasi fuera de las personas y fuera de los rotulos, seguidos con
    Lucas-Kanade piramidal entre cuadros consecutivos de la muestra, y una similitud robusta por
    cuadro. Se acumulan hacia la referencia. NO se usa la deteccion de transiciones de
    `transiciones.py` para esto: aquella separa fundido de zoom con el residuo de mezcla, que es una
    senal de PIXELES y responde difuminada; aqui hace falta el desplazamiento en pixeles, que es
    otra cosa.

    SE CALCULA PARA TODOS LOS CUADROS UTILIZABLES, no solo para los de la placa, y hace falta: el
    desplazamiento medido en este corpus llega a **264 px** en un clip de 1280 px de ancho, o sea el
    21 % del cuadro. Usar la matriz del cuadro mas cercano de la muestra de 48 dejaria un error de
    varios pixeles en la posicion del pie, que a 3 cm por pixel son varios centimetros de error
    metrico gratis.

    Devuelve la lista de matrices 3x3 (cuadro de `ks` -> referencia) y el desplazamiento maximo
    acumulado en pixeles, que es la cifra que dice cuanto se mueve la camara.
    """
    if ref is None:
        ref = len(ks) // 2
    Ms = [np.eye(3) for _ in ks]
    pares = []
    p0 = None
    ga = None
    for j in range(len(ks) - 1):
        a, b = ks[j], ks[j + 1]
        if a >= len(frames) or b >= len(frames):
            pares.append(np.eye(3))
            p0 = None
            continue
        if ga is None or j % REDETECTA_PUNTOS == 0 or p0 is None or len(p0) < 40:
            ga = cv2.cvtColor(frames[a], cv2.COLOR_BGR2GRAY)
            msk = mascara_personas(cajas, cuadros, a, W, H)
            msk[rot > 0] = 0
            p0 = cv2.goodFeaturesToTrack(ga, 300, 0.01, 10, mask=msk)
        gb = cv2.cvtColor(frames[b], cv2.COLOR_BGR2GRAY)
        if p0 is None or len(p0) < 12:
            pares.append(np.eye(3))
            ga, p0 = gb, None
            continue
        p1, st, _e = cv2.calcOpticalFlowPyrLK(ga, gb, p0, None,
                                              winSize=(21, 21), maxLevel=3)
        st = st.reshape(-1).astype(bool)
        if st.sum() < 12:
            pares.append(np.eye(3))
            ga, p0 = gb, None
            continue
        A, _in = cv2.estimateAffinePartial2D(p0[st], p1[st], method=cv2.RANSAC,
                                             ransacReprojThreshold=2.0)
        if A is None:
            pares.append(np.eye(3))
            ga, p0 = gb, None
            continue
        M = np.eye(3)
        M[:2, :] = A
        pares.append(M)
        ga, p0 = gb, p1[st].reshape(-1, 1, 2)
    for j in range(ref - 1, -1, -1):
        Ms[j] = Ms[j + 1] @ pares[j]
    for j in range(ref + 1, len(ks)):
        Ms[j] = Ms[j - 1] @ np.linalg.inv(pares[j - 1])
    esq = np.array([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1]], "float64").T
    dmax = 0.0
    for M in Ms:
        q = M @ esq
        q = q[:2] / q[2]
        dmax = max(dmax, float(np.abs(q - esq[:2]).max()))
    return Ms, dmax


MEDIANA_NUMPY = os.environ.get("FENCINGLY_MEDIANA_NUMPY", "0") == "1"


def _nanmediana0(v):
    """`np.nanmedian(v, axis=0)` EXACTO y ~7 veces mas rapido (2026-10-13, `CORRIDA_FINAL` §10).

    Con el eje corto (< 600, aqui los ~48 cuadros de la placa) numpy va por `np.ma.median`, que ordena
    un array enmascarado y costaba ~8 s de los ~20 de la etapa 7 en cada clip. Aqui: ordenar (los NaN
    quedan al final), contar los finitos y coger el del medio, o la media de los dos del medio en el
    mismo tipo, como numpy. Comprobado: 3.000 arrays al azar (1-69 cuadros, NaN al azar) y una franja
    de 48x64x1280x3, identicos bit a bit, NaN incluidos. `FENCINGLY_MEDIANA_NUMPY=1` vuelve a numpy."""
    s = np.sort(v, axis=0)
    n = (~np.isnan(v)).sum(axis=0)
    lo = np.take_along_axis(s, np.maximum((n - 1) // 2, 0)[None], 0)[0]
    hi = np.take_along_axis(s, np.minimum(n // 2, s.shape[0] - 1)[None], 0)[0]
    med = np.where(n % 2 == 1, lo, (lo + hi) / np.asarray(2, v.dtype)).astype(v.dtype)
    med[n == 0] = np.nan
    return med


def placa(frames, sel, Ms, cajas, cuadros, W, H) -> tuple:
    """QUE HACE: el suelo sin personas y sin arrastre de camara, por mediana temporal en el sistema
    del cuadro de referencia."""
    pila, tap = [], []
    for j, k in enumerate(sel):
        if k >= len(frames):
            continue
        Mi = np.linalg.inv(Ms[j])
        im = cv2.warpPerspective(frames[k], Ms[j], (W, H), flags=cv2.INTER_LINEAR,
                                 borderValue=(0, 0, 0))
        t = cv2.warpPerspective(mascara_personas(cajas, cuadros, k, W, H) * 255, Ms[j], (W, H),
                                flags=cv2.INTER_NEAREST, borderValue=0)
        pila.append(im)
        tap.append(t > 127)
        del Mi
    if not pila:
        return None, None
    V8 = np.stack(pila)
    M = np.stack(tap)
    pl = np.zeros((H, W, 3), "uint8")
    for a in range(0, H, FRANJA):
        b = min(a + FRANJA, H)
        v = V8[:, a:b].astype("float32")
        v[~M[:, a:b]] = np.nan
        with np.errstate(all="ignore"):
            q = np.nanmedian(v, axis=0) if MEDIANA_NUMPY else _nanmediana0(v)
        pl[a:b] = np.where(np.isfinite(q), q, 0.0).astype("uint8")
    cob = M.sum(axis=0)
    return pl, cob


def _pies_caja(r) -> np.ndarray:
    """QUE HACE: el apoyo de cada tirador segun la LINEA BASE, o sea el centro del borde inferior de
    la caja. Se usa como SEMILLA para encontrar la pista y como respaldo cuando no hay pose."""
    B, pres, W, H, T = r["B"], r["pres"], r["W"], r["H"], r["T"]
    P = np.full((T, 2, 2), np.nan, "float64")
    for k in range(T):
        for lado in (0, 1):
            if pres[k, lado] and np.isfinite(B[k, lado]).all():
                b = B[k, lado]
                P[k, lado] = (0.5 * (b[0] + b[2]) * W, b[3] * H)
    return P


def corredor(P, mask, Mk=None) -> tuple:
    """QUE HACE: el eje aproximado y la banda por la que pasan los apoyos, en el sistema de la
    referencia.

    ESTO NO ES LA HOMOGRAFIA Y NO ENTRA EN ELLA. Es la SEMILLA que dice donde buscar la pista, y el
    requisito duro de la tarea (ajustar con la pista y no con los tiradores) se cumple porque el eje
    definitivo sale de los bordes detectados, no de aqui. Se usan los tiradores para la semilla
    porque solo pueden andar a lo largo de la pista, asi que la nube de sus apoyos esta dentro de la
    pista por construccion, y no hay ninguna otra pista disponible que sea igual de barata.
    """
    q = []
    for k in np.nonzero(mask)[0]:
        for lado in (0, 1):
            p = P[k, lado]
            if not np.isfinite(p).all():
                continue
            if Mk is not None:
                v = Mk[k] @ np.array([p[0], p[1], 1.0])
                p = v[:2] / v[2]
            q.append(p)
    q = np.asarray(q, "float64")
    if len(q) < 8:
        return None
    c0 = q.mean(axis=0)
    _u, _s, vt = np.linalg.svd(q - c0, full_matrices=False)
    e = vt[0] / np.linalg.norm(vt[0])
    if e[0] < 0:
        e = -e
    n = np.array([-e[1], e[0]], "float64")
    t = (q - c0) @ n
    s = (q - c0) @ e
    return {"c0": c0, "e": e, "n": n,
            "t": (float(np.percentile(t, 8)), float(np.percentile(t, 92))),
            "s": (float(np.percentile(s, 2)), float(np.percentile(s, 98))), "q": q}


ALCANCE_BARRIDO = 2.2
"""Hasta donde se aleja el barrido del corredor de los pies, en ALTOS DE CAJA del tirador.

ES UN FALLO MEDIDO DE MI PRIMERA VERSION, y de los caros. La primera version barria solo el corredor
mas un 2 % del alto de imagen, o sea 14 px en 720p, y el resultado fue un ancho de pista de 64 px en
`GentleQuerulousElephantseal` (el prototipo daba 159) y de 35,6 px en `UglyNiceAplomadofalcon`
(donde la pista ocupa mas de 200). Con los dos bordes mal colocados salian casi paralelos, V1 se iba
al infinito y la homografia se rechazaba con "f implicada absurda". O sea: el sintoma aparecia tres
pasos mas abajo que la causa.

La unidad correcta es el alto de caja del tirador, porque escala con la perspectiva igual que el
ancho de la pista. Un tirador mide del orden de 1,75 m de pie y la pista de 1,50 a 2,00 m tumbada,
asi que 2,2 altos de caja es alcance de sobra para llegar al borde en cualquier encuadre y sigue
siendo poco para saltar a la pista de al lado, que en este corpus esta a 3 o 4 anchos de pista."""


def region_y_bordes(pl, cob, rot, co, alto_px) -> dict:
    """QUE HACE: los dos bordes longitudinales de la pista, por barrido perpendicular al eje.

    POR QUE BARRIDO Y NO COMPONENTE CONEXA, y esto es un fallo medido del primer prototipo: en
    `UglyNiceAplomadofalcon` hay CUATRO pistas del mismo color en el pabellon, y la componente conexa
    del color se derramaba a todas y a las gradas de madera, dando un ancho de 331 px con desviacion
    de 87 px, o sea inservible. El barrido no puede derramarse: sale del corredor de los pies y se
    para en el PRIMER corte de color, asi que se queda en la pista buena por construccion.

    LOS BORDES SE AJUSTAN COMO RECTA, no como poligonal, porque un borde de pista es recto en el
    mundo y por tanto recto en la imagen. El residuo del ajuste es la cifra que dice si el barrido ha
    encontrado un borde o basura.
    """
    H, W = pl.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(pl, (5, 5), 0), cv2.COLOR_BGR2LAB).astype("float32")
    ok = (cob >= max(4, int(0.20 * cob.max()))) & (rot == 0)
    yy, xx = np.mgrid[0:H, 0:W]
    c0, e, n = co["c0"], co["e"], co["n"]
    S = (xx - c0[0]) * e[0] + (yy - c0[1]) * e[1]
    Dp = (xx - c0[0]) * n[0] + (yy - c0[1]) * n[1]
    t0, t1 = co["t"]
    sem = (Dp >= t0) & (Dp < t1) & ok
    if sem.sum() < 500:
        return {"ok": False, "motivo": "corredor sin pixeles validos"}
    X = lab[sem]
    mu = np.median(X, axis=0)
    Sg = np.cov(X.T) + np.eye(3) * 4.0
    Si = np.linalg.inv(Sg)
    Z = lab.reshape(-1, 3) - mu
    d2 = np.einsum("ij,jk,ik->i", Z, Si, Z).reshape(H, W)
    dentro = (d2 < MAHAL2) & ok

    s0, s1 = co["s"]
    bins = np.linspace(s0, s1, 33)
    alc = ALCANCE_BARRIDO * max(alto_px, 20.0)
    pasos = np.arange(t0 - alc, t1 + alc, 1.0)
    lado_a, lado_b, cent = [], [], []
    for i in range(len(bins) - 1):
        sc = 0.5 * (bins[i] + bins[i + 1])
        base = c0 + sc * e
        val = []
        for dd in pasos:
            p = base + dd * n
            x, y = int(round(p[0])), int(round(p[1]))
            val.append(dentro[y, x] if (0 <= x < W and 0 <= y < H) else False)
        val = np.asarray(val)
        j0 = int(np.argmin(np.abs(pasos - 0.5 * (t0 + t1))))
        if not val[j0]:
            v = np.nonzero(val)[0]
            if not len(v):
                continue
            j0 = int(v[np.argmin(np.abs(v - j0))])
        a = j0
        while a - 1 >= 0 and val[a - 1]:
            a -= 1
        b = j0
        while b + 1 < len(val) and val[b + 1]:
            b += 1
        if b - a < 4:
            continue
        lado_a.append(base + pasos[a] * n)
        lado_b.append(base + pasos[b] * n)
        cent.append(sc)
    if len(cent) < 8:
        return {"ok": False, "motivo": "barrido con menos de 8 columnas"}
    out = {"ok": True, "n_col": len(cent),
           "centro_pista": 0.5 * (np.asarray(lado_a).mean(axis=0)
                                  + np.asarray(lado_b).mean(axis=0))}
    """`centro_pista` es un punto que esta DENTRO de la pista por construccion, y hace falta para las
    puertas fisicas del RANSAC de V2: "por encima del borde de lejos" solo se puede comprobar si se
    sabe de que lado esta el interior."""
    for nom, pts in (("A", np.asarray(lado_a)), ("B", np.asarray(lado_b))):
        L, res = _recta_robusta(pts)
        out["recta_" + nom] = L
        out["res_" + nom] = res
        out["pts_" + nom] = pts
    return out


def _recta_robusta(pts: np.ndarray) -> tuple:
    """QUE HACE: la recta que mejor pasa por unos puntos, en coordenadas homogeneas, con dos vueltas
    de rechazo del 20 % peor.

    El rechazo hace falta porque el barrido se cuela por un hueco de color en algunas columnas y esos
    puntos se van lejos. Devuelve la recta (a, b, c) normalizada y el residuo mediano en pixeles.
    """
    q = pts.copy()
    L = None
    for _v in range(3):
        c = q.mean(axis=0)
        _u, _s, vt = np.linalg.svd(q - c, full_matrices=False)
        d = vt[0]
        nn = np.array([-d[1], d[0]], "float64")
        nn = nn / np.linalg.norm(nn)
        L = np.array([nn[0], nn[1], -float(nn @ c)], "float64")
        r = np.abs(pts @ L[:2] + L[2])
        keep = r <= max(np.percentile(r, 80), 1.0)
        if keep.sum() >= 6:
            q = pts[keep]
    r = np.abs(pts @ L[:2] + L[2])
    return L, float(np.median(r))


N_U = 64
"""Muestras a lo ancho de la pista en la tira rectificada."""
PEND_MAX = 2.0
"""Inclinacion maxima que se prueba para una marca transversal, en ANCHOS DE PISTA de recorrido
longitudinal por ancho completo cruzado.

Pendiente 0 quiere decir que la marca sale vertical en la tira, que es lo que pasa cuando la camara
mira perpendicular a la pista. Pendiente 2,0 quiere decir que la marca avanza dos anchos de pista a
lo largo mientras cruza el ancho, o sea un plano muy oblicuo. El barrido se hace en anchos de pista
y no en pixeles para que valga igual en 720p y en 1080p."""
PICO_SIGMA = 4.0
"""Cuantas desviaciones por encima del fondo tiene que estar un pico para contar como marca. Se
declara antes de mirar: es el umbral de una senal contra su propio ruido, no un numero ajustado a
un clip."""


def tira(pl, cob, rot, co, bordes) -> dict:
    """QUE HACE: la TIRA RECTIFICADA de la pista, o sea la pista estirada a un rectangulo.

    Para cada posicion `s` a lo largo del eje se toma el punto del borde A y el del borde B y se
    muestrea el segmento que los une. Asi la pista, que en la imagen es un trapecio, sale como una
    banda rectangular, y una marca transversal sale como una recta casi vertical.

    Es la misma "tira rectificada" que uso el estudio historico para medir el techo humano del
    4,3 %, asi que las cifras de hoy y las de entonces se pueden comparar.
    """
    H, W = pl.shape[:2]
    e, n, c0 = co["e"], co["n"], co["c0"]
    LA = np.asarray(bordes["recta_A"], "float64")
    LB = np.asarray(bordes["recta_B"], "float64")
    s0, s1 = co["s"]
    marg = 0.10 * (s1 - s0)
    ss = np.arange(s0 - marg, s1 + marg, 1.0)
    g = cv2.cvtColor(pl, cv2.COLOR_BGR2GRAY).astype("float32")
    val = ((cob >= max(4, int(0.20 * cob.max()))) & (rot == 0)).astype("float32")
    uu = np.linspace(0.0, 1.0, N_U)
    PA = np.empty((len(ss), 2))
    PB = np.empty((len(ss), 2))
    for i, s in enumerate(ss):
        base = c0 + s * e
        for L, out in ((LA, PA), (LB, PB)):
            den = float(L[:2] @ n)
            if abs(den) < 1e-12:
                out[i] = base
            else:
                lam = -(float(L[:2] @ base) + L[2]) / den
                out[i] = base + lam * n
    pts = PA[None, :, :] + uu[:, None, None] * (PB - PA)[None, :, :]
    mx = pts[:, :, 0].astype("float32")
    my = pts[:, :, 1].astype("float32")
    I = cv2.remap(g, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    Vm = cv2.remap(val, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return {"I": I, "V": Vm, "ss": ss, "PA": PA, "PB": PB, "uu": uu,
            "ancho_px": float(np.median(np.linalg.norm(PB - PA, axis=1)))}


def marcas(pl, cob, rot, co, bordes) -> dict:
    """QUE HACE: las marcas transversales pintadas, con un FILTRO ADAPTADO sobre la tira
    rectificada.

    POR QUE NO `createLineSegmentDetector`, y esta medido en `p1_censo_pista.py`: LSD sobre la placa
    da 1 clip de 13 con dos o mas lineas transversales, y mirando las imagenes las marcas ESTAN ahi.
    LSD busca bordes en toda la imagen y se le van los votos a las vallas, las sillas y las juntas
    del suelo de alrededor.

    POR QUE TAMPOCO COMPONENTES CONEXAS DE LA CRESTA, que fue mi segundo intento y dio CERO marcas en
    `GentleQuerulousElephantseal`, un clip donde se ven cuatro a ojo: **la linea central es
    DISCONTINUA por reglamento** (t.19.2, "must be drawn as a broken line"), asi que su cresta sale
    partida en trozos de 20 px y ninguno llega al largo minimo. Un detector que exige continuidad no
    puede encontrar una linea que el reglamento define como discontinua.

    LO QUE SI FUNCIONA: sumar la cresta A LO LARGO DE TODO EL ANCHO de la pista, con una inclinacion
    de prueba. La suma no le importa que la linea tenga huecos, porque los trozos que si estan
    suman igual. Se barre la inclinacion porque en un plano oblicuo la marca no sale vertical en la
    tira, y la inclinacion que gana ES la direccion transversal, o sea que de paso da V2.

    EL CONFUNDIDO QUE HAY QUE DECLARAR Y NO SE PUEDE RESOLVER AQUI: la pista es de planchas y las
    JUNTAS entre planchas tambien son crestas transversales, y no estan a las distancias FIE. Este
    detector no puede distinguir una junta de una raya, porque geometricamente son lo mismo. Lo unico
    honrado es devolver crestas y dejar que el control C2 diga cuanto encajan con el reglamento.
    """
    tr = tira(pl, cob, rot, co, bordes)
    I, Vm, ss = tr["I"], tr["V"], tr["ss"]
    if I.shape[1] < 40:
        return {"ok": False, "motivo": "tira demasiado corta", "lineas": [], "tira": tr}
    Is = cv2.GaussianBlur(I, (0, 0), 1.4)
    d2 = -cv2.Sobel(cv2.Sobel(Is, cv2.CV_32F, 1, 0, ksize=3), cv2.CV_32F, 1, 0, ksize=3)
    vmask = (Vm > 0.5)
    """SE COME UN MARGEN DEL BORDE DE LA ZONA VALIDA, y hace falta. El primer intento daba siempre una
    marca con senal-ruido de 358 a 617, veinte veces mayor que las demas, y siempre pegada al
    principio de la tira. No era una marca: era el ESCALON del propio borde de la zona valida, donde
    el gris pasa de 0 a suelo, que para una segunda derivada es la cresta mas fuerte de la imagen."""
    er = max(3, int(0.03 * max(tr["ancho_px"], 8.0)))
    vmask = cv2.erode(vmask.astype("uint8"),
                      np.ones((1, 2 * er + 1), "uint8")).astype(bool)
    d2 = d2 * vmask
    nu = I.shape[0]
    cent = 0.5 * (nu - 1)
    frac = (np.arange(nu) - cent) / max(nu - 1, 1)
    """`frac` va de -0,5 a +0,5 y es la posicion relativa a lo ancho de la pista."""
    anc = max(tr["ancho_px"], 8.0)
    paso = max(1.0, anc / 50.0)
    pends = np.arange(-PEND_MAX * anc, PEND_MAX * anc + 1e-9, paso)
    W1 = I.shape[1]
    S = np.full((len(pends), W1), -1e30, "float64")
    vm = vmask.astype("float64")
    for ip, p in enumerate(pends):
        acc = np.zeros(W1, "float64")
        peso = np.zeros(W1, "float64")
        for iu in range(nu):
            d = int(round(p * frac[iu]))
            acc += np.roll(d2[iu], -d)
            peso += np.roll(vm[iu], -d)
        with np.errstate(all="ignore"):
            S[ip] = np.where(peso >= 0.6 * nu, acc / np.maximum(peso, 1e-6), -1e30)
    """EL PERFIL ES EL MAXIMO SOBRE INCLINACIONES, no el de una inclinacion fija, y ESTE ERA EL FALLO
    QUE ROMPIA LA HOMOGRAFIA. Con una sola inclinacion global las marcas detectadas salen todas
    PARALELAS, y unas rectas paralelas no se cortan en un punto: el residuo de concurrencia en V2
    salia de 164 a 576 px y la focal implicada salia negativa. Pero las transversales de una pista NO
    son paralelas en la imagen: convergen en V2, y esa convergencia ES el dato que hace falta. Asi
    que cada marca tiene que llevar SU inclinacion, y el maximo sobre inclinaciones la da."""
    perf = S.max(axis=0)
    argp = S.argmax(axis=0)
    val = perf > -1e29
    act = perf[val]
    if act.size < 20:
        return {"ok": False, "motivo": "perfil sin muestras", "lineas": [], "tira": tr,
                "ancho_px": tr["ancho_px"]}
    med = float(np.median(act))
    mad = float(np.median(np.abs(act - med))) * 1.4826 + 1e-6
    umb = med + PICO_SIGMA * mad
    idx = []
    i = 1
    sep = max(4, int(0.25 * anc))
    while i < W1 - 1:
        if val[i] and perf[i] >= umb and perf[i] >= perf[i - 1] and perf[i] >= perf[i + 1]:
            idx.append(i)
            i += sep
        else:
            i += 1
    lineas = []
    for i in idx:
        p = float(pends[argp[i]])
        ia = int(round(i + p * frac[0]))
        ib = int(round(i + p * frac[nu - 1]))
        ia = min(max(ia, 0), len(ss) - 1)
        ib = min(max(ib, 0), len(ss) - 1)
        pa = tr["PA"][ia]
        pb = tr["PB"][ib]
        v = pb - pa
        if np.linalg.norm(v) < 6.0:
            continue
        nn = np.array([-v[1], v[0]], "float64")
        nn = nn / np.linalg.norm(nn)
        L = np.array([nn[0], nn[1], -float(nn @ pa)], "float64")
        lineas.append({"L": L, "centro": 0.5 * (pa + pb), "largo_px": float(np.linalg.norm(v)),
                       "cubre": 1.0, "i_tira": int(i), "pend": p, "pico": float(perf[i]),
                       "sn": float((perf[i] - med) / mad),
                       "s": float((0.5 * (pa + pb) - co["c0"]) @ co["e"])})
    lineas.sort(key=lambda z: z["s"])
    _afina(lineas, tr, d2, frac)
    return {"ok": len(lineas) > 0, "lineas": lineas, "ancho_px": tr["ancho_px"],
            "sn_mejor": float((act.max() - med) / mad), "perfil": perf, "umbral": umb, "tira": tr,
            "motivo": "" if lineas else "ningun pico por encima de %.1f sigma" % PICO_SIGMA}


BUSCA_AFINA = 3
"""Cuantos pixeles a cada lado se busca el maximo de cresta al afinar una marca, alrededor de la
recta que ya trae SU PROPIA inclinacion. Con la inclinacion buena una ventana de 3 px basta; con una
inclinacion global no bastaba ninguna, porque el error crecia con el ancho de la pista."""
COBERTURA_MARCA = 0.45
"""Fraccion del ancho de la pista en la que la marca tiene que tener cresta de verdad para que
cuente. El reglamento dice que las marcas se trazan "across the whole width of the piste", asi que
una marca legitima cubre casi todo; se pide solo el 45 % porque la linea central es DISCONTINUA y
porque un tramo puede quedar tapado por un rotulo."""


def _afina(lineas, tr, d2, frac):
    """QUE HACE: reajusta cada marca como una RECTA de la imagen, punto por punto.

    POR QUE HACE FALTA, Y ES UN FALLO DE DISENO QUE ME PILLE MIRANDO LOS NUMEROS. El detector usa un
    modelo de "desplazamiento constante": la marca va del punto A(s - c/2) al punto B(s + c/2) con
    `c` la misma para todas. Eso vale para DETECTAR, porque localmente es buena aproximacion, pero es
    PROYECTIVAMENTE FALSO: las rectas transversales de verdad pasan todas por un mismo punto de fuga
    V2, y la correspondencia entre los dos bordes que eso induce es una funcion de Moebius, no un
    desplazamiento constante. Con el modelo de desplazamiento constante las rectas detectadas NO son
    concurrentes, asi que V2 sale de la interseccion de rectas que no se cortan en un punto y la
    focal implicada salia negativa: f2/W2 de -7,5 a -204,6 en los tres clips de prueba.

    El sintoma estaba tres pasos por debajo de la causa, otra vez. Aqui se corrige de raiz: cada
    marca se vuelve a medir sobre la imagen, buscando el maximo de cresta en cada fila de la tira, y
    se le ajusta una recta. Asi las rectas son lo que la imagen dice y su concurrencia pasa a ser una
    MEDIDA, que es el control de todo el modelo.
    """
    nu = d2.shape[0]
    ns = d2.shape[1]
    act = d2[d2 != 0.0]
    umb = 0.0
    if act.size > 100:
        med = float(np.median(act))
        mad = float(np.median(np.abs(act - med))) * 1.4826 + 1e-6
        umb = med + 2.0 * mad
    """UMBRAL DE CRESTA POR FILA, y hace falta para tirar las marcas FALSAS. Sin el, el afinado coge
    el maximo de la ventana en TODAS las filas, aunque en esa fila no haya ninguna cresta, asi que un
    borron blanco de 20 px del rotulo produce una recta con 62 puntos de 64 y parece la marca mejor
    medida del clip. Se vio en `GentleQuerulousElephantseal`: dos marcas falsas casi verticales
    encima de un resto de grafico en el centro, con 59 y 62 puntos, que se llevaban el punto de fuga
    V2 a 8.957 px POR DEBAJO del centro cuando tiene que estar por encima."""
    pts_img = tr["PA"][None, :, :] + tr["uu"][:, None, None] * (tr["PB"] - tr["PA"])[None, :, :]
    for li in lineas:
        i0 = li["i_tira"]
        pend = li["pend"]
        qs = []
        for iu in range(nu):
            c = int(round(i0 + pend * frac[iu]))
            a, b = max(c - BUSCA_AFINA, 0), min(c + BUSCA_AFINA + 1, ns)
            if b - a < 3:
                continue
            seg = d2[iu, a:b]
            j = int(np.argmax(seg))
            if seg[j] <= umb:
                continue
            qs.append(pts_img[iu, a + j])
        li["cubre"] = len(qs) / float(nu)
        if len(qs) < COBERTURA_MARCA * nu:
            li["afinada"] = False
            continue
        q = np.asarray(qs, "float64")
        L, res = _recta_robusta(q)
        v = q[-1] - q[0]
        li["L"] = L
        li["centro"] = q.mean(axis=0)
        li["largo_px"] = float(np.linalg.norm(v))
        li["res_px"] = res
        li["n_pts"] = len(q)
        li["afinada"] = True
        li["s"] = float((li["centro"] - tr["PA"][0]) @ (tr["PA"][-1] - tr["PA"][0])
                        / max(np.linalg.norm(tr["PA"][-1] - tr["PA"][0]), 1e-9))


def fuga_ls(Ls, pp, iteraciones=3) -> dict:
    """QUE HACE: el punto de fuga por el que MEJOR pasan unas rectas, EN COORDENADAS HOMOGENEAS.

    POR QUE HOMOGENEAS Y NO UN PUNTO NORMAL, y esto era un fallo de fondo de mi primera version: en
    una toma lateral, que es el encuadre normal de la television de esgrima, el punto de fuga del eje
    de la pista esta practicamente en el INFINITO, porque los dos bordes salen casi paralelos. Un
    punto en el infinito no se puede escribir como (x, y) y cualquier calculo que lo divida por su
    tercera coordenada explota. En homogeneas es un vector como cualquier otro.

    Y EL RESIDUO SE DA EN GRADOS, NO EN PIXELES, por la misma razon. Medi la concurrencia como la
    distancia en pixeles del punto de fuga a cada recta y salieron de 164 a 3.321 px, lo que parecia
    un desastre; pero para un punto de fuga a 10.000 px, un error de angulo de 1 grado ya da 175 px
    de distancia. La distancia en pixeles a un punto lejano no mide la calidad de las rectas: mide lo
    lejos que esta el punto. El angulo si la mide.

    Se hace con rechazo iterativo del peor tercio, porque entre las verticales de una sala hay
    siempre alguna que no es vertical.
    """
    A = []
    for L in Ls:
        L = np.asarray(L, "float64")
        nn = np.linalg.norm(L[:2])
        if nn < 1e-12:
            continue
        A.append(np.array([L[0] / nn, L[1] / nn, (L[2] + L[0] * pp[0] + L[1] * pp[1]) / nn]))
    if len(A) < 2:
        return {"ok": False, "motivo": "menos de 2 rectas", "n": len(A)}
    A = np.asarray(A)
    keep = np.ones(len(A), bool)
    v = None
    for _it in range(iteraciones):
        M = A[keep]
        _u, _s, vt = np.linalg.svd(M.T @ M)
        v = vt[-1]
        ang = _angulos(A, v)
        if keep.sum() > 3:
            lim = max(float(np.percentile(ang[keep], 67)), 0.2)
            nk = ang <= lim
            if nk.sum() >= 3:
                keep = nk
    ang = _angulos(A, v)
    return {"ok": True, "v": v, "n": int(keep.sum()), "n_total": len(A),
            "ang_p50": float(np.median(ang[keep])), "ang_max": float(np.max(ang[keep])),
            "en_infinito": bool(abs(v[2]) < 1e-9)}


def _angulos(A, v):
    """QUE HACE: para cada recta, el angulo en grados que habria que girarla para que pasase por `v`.

    Es la medida correcta de concurrencia cuando el punto de fuga esta lejos, y no depende de lo
    lejos que este. Se calcula como el angulo entre la recta y la recta que une `v` con el punto de
    la recta mas cercano al origen.
    """
    out = np.zeros(len(A))
    for i, L in enumerate(A):
        n = L[:2]
        p0 = -L[2] * n
        if abs(v[2]) > 1e-12:
            q = v[:2] / v[2]
            d = q - p0
        else:
            d = v[:2]
        nd = np.linalg.norm(d)
        if nd < 1e-9:
            out[i] = 0.0
            continue
        out[i] = abs(float(np.degrees(np.arcsin(np.clip(abs(n @ d) / nd, 0.0, 1.0)))))
    return out


ANG_VERT = 30.0
"""Desviacion maxima respecto a la vertical de la imagen para que un segmento se considere una
estructura vertical de la sala. Con camara sin balanceo y punto de fuga vertical muy lejano, una
vertical del mundo sale casi vertical en la imagen; 30 grados es holgado y el rechazo robusto se
encarga del resto."""
LARGO_VERT = 0.045
"""Largo minimo de un segmento vertical en fraccion del alto de imagen. Bajado de 0,06 porque con
0,06 salian solo 3 a 15 segmentos por clip y hacen falta mas para que el rechazo robusto tenga de
donde elegir."""
ANG_PARALELO_BORDE = 12.0
"""Un segmento que este a menos de estos grados de la direccion de los bordes de la pista NO cuenta
como vertical, aunque su angulo con la vertical de la imagen sea admisible: es una raya del suelo o
una junta, no una estructura de pie."""


def verticales(pl, rot, cob, bordes, co) -> dict:
    """QUE HACE: el punto de fuga VERTICAL, a partir de las estructuras fijas de la sala.

    POR QUE HACE FALTA, Y ES EL RESULTADO MAS IMPORTANTE DE ESTE MODULO. Con la camara mirando la
    pista de lado, que es el encuadre normal, el sistema de ecuaciones esta UN GRADO DE LIBERTAD
    CORTO, y se puede demostrar contando: los datos que da la pista son la fila del borde de lejos,
    la fila del borde de cerca y la fila del horizonte, o sea 3; las incognitas son la distancia
    focal, la inclinacion de la camara, su altura y su distancia a la pista, o sea 4. El ancho de
    pista fija la escala, no la focal. Asi que con la pista SOLA no hay homografia, y ninguna
    cantidad de marcas transversales lo arregla, porque en vista lateral todas las transversales
    concurren en el mismo punto que ya da el horizonte.

    Y LA SENSIBILIDAD NO ES PEQUENA: resolviendo el modelo a mano con f = 1200 y con f = 1600 px
    sobre las mismas tres filas, la escala metrica a lo largo de la pista cambia un **24 %**. O sea
    que la focal no es un detalle: es la mitad del resultado.

    DE DONDE SALE. Una recta vertical del mundo proyecta a una recta que pasa por el punto de fuga
    vertical Vz. Un pabellon esta lleno de verticales fijas: el poste del aparato senalador, los pies
    de los enrolladores, los postes de las vallas, los respaldos de las sillas, los cantos de los
    paneles. Se cogen de la PLACA LIMPIA, donde las personas ya no estan, asi que no se usa a ningun
    tirador; y se excluye la region de la pista, para no coger sus propias marcas.

    LO QUE ESTO NO ES: no es un prior ni una suposicion. Es una medida geometrica de la escena, con
    su residuo en grados y su control (el punto de fuga vertical tiene que caer en la columna del
    punto principal si la camara no tiene balanceo).
    """
    H, W = pl.shape[:2]
    g = cv2.cvtColor(pl, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    lsd = cv2.createLineSegmentDetector()
    li = lsd.detect(g)[0]
    if li is None:
        return {"ok": False, "motivo": "LSD no devuelve segmentos"}
    LA = np.asarray(bordes["recta_A"], "float64")
    LB = np.asarray(bordes["recta_B"], "float64")
    cobmin = max(4, int(0.20 * cob.max()))
    Ls, segs = [], []
    for s in li.reshape(-1, 4):
        x0, y0, x1, y1 = [float(v) for v in s]
        Lg = float(np.hypot(x1 - x0, y1 - y0))
        if Lg < LARGO_VERT * H:
            continue
        ang = abs(float(np.degrees(np.arctan2(x1 - x0, y1 - y0))))
        ang = min(ang, 180.0 - ang)
        if ang > ANG_VERT:
            continue
        ae = abs(float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
                 - float(np.degrees(np.arctan2(co["e"][1], co["e"][0]))))
        ae = abs((ae + 90.0) % 180.0 - 90.0)
        if ae < ANG_PARALELO_BORDE:
            continue
        cx, cy = int(0.5 * (x0 + x1)), int(0.5 * (y0 + y1))
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        if rot[cy, cx] > 0 or cob[cy, cx] < cobmin:
            continue
        pa = np.array([cx, cy, 1.0])
        if float(pa @ LA) * float(pa @ LB) < 0:
            """Dentro de la pista: sus marcas no son verticales del mundo."""
            continue
        v = np.array([x1 - x0, y1 - y0])
        n = np.array([-v[1], v[0]])
        n = n / np.linalg.norm(n)
        Ls.append(np.array([n[0], n[1], -float(n @ np.array([x0, y0]))]))
        segs.append((x0, y0, x1, y1))
    if len(Ls) < VERT_MIN:
        return {"ok": False, "motivo": "menos de %d verticales (n=%d)" % (VERT_MIN, len(Ls)),
                "n": len(Ls), "segs": segs}
    pp = _pp(W, H)
    r = fuga_ls(Ls, pp)
    if not r.get("ok"):
        return {"ok": False, "motivo": r.get("motivo", "?"), "n": len(Ls), "segs": segs}
    r["segs"] = segs
    v = r["v"]
    r["desv_columna_px"] = float(abs(v[0] / v[2])) if abs(v[2]) > 1e-12 else float("inf")
    r["vz_rel_H"] = float(v[1] / v[2] / H) if abs(v[2]) > 1e-12 else float("inf")
    """LAS TRES PUERTAS DEL PUNTO DE FUGA VERTICAL, y las tres son fisica comprobable, no umbrales
    ajustados: tiene que haber suficientes verticales, tienen que concurrir de verdad, y el punto
    tiene que caer en la columna del punto principal y POR DEBAJO del centro, porque la camara mira
    hacia abajo. Sin la tercera, `UglyNiceAplomadofalcon` devolvia tiradores de 0,93 m."""
    if r["n"] < VERT_MIN:
        r.update({"ok": False, "motivo": "solo %d verticales concurren" % r["n"]})
        return r
    if r["ang_p50"] > VERT_ANG_MAX:
        r.update({"ok": False, "motivo": "las verticales no concurren (%.2f deg)" % r["ang_p50"]})
        return r
    if r["desv_columna_px"] > VERT_COL_MAX * W:
        r.update({"ok": False,
                  "motivo": "Vz fuera de la columna principal (%.0f px = %.0f %% del ancho)"
                            % (r["desv_columna_px"], 100.0 * r["desv_columna_px"] / W)})
        return r
    if not (r["vz_rel_H"] > 0.5):
        r.update({"ok": False, "motivo": "Vz por encima del centro (%.2f altos): la camara no "
                                         "mira hacia abajo" % r["vz_rel_H"]})
        return r
    return r


def _ancho_pista_px(bordes, co) -> float:
    """QUE HACE: el ancho de la pista en pixeles, medido en el centro del corredor."""
    LA, LB = bordes["recta_A"], bordes["recta_B"]
    p = co["c0"]
    return abs(float(p @ LA[:2] + LA[2])) + abs(float(p @ LB[:2] + LB[2]))


SEN_MIN = 0.010
"""Seno minimo del angulo entre los dos bordes para que su cruce V1 valga. Con 0,010 y una imagen de
1280 px, dos bordes que se cruzan a 1280 px de distancia difieren en 13 px de un extremo a otro del
cuadro, que es mas que el residuo de ajuste medido (1 a 6 px). Por debajo de eso el cruce es ruido."""
FUGA_MAX = 6.0
"""Distancia maxima de un punto de fuga al punto principal, en anchos de imagen.

ES UNA PUERTA, Y HAY QUE PONERLA APRETADA. Un punto de fuga a muchos anchos de imagen es
practicamente el infinito, y "el infinito" siempre explica unas rectas casi paralelas mejor que
cualquier punto propio: el barrido de focal se escapa hacia arriba y se para donde le pongas el
tope. Medido en `GentleQuerulousElephantseal` con el tope en 30 anchos: la focal salia
**f/W = 6,6** y la altura de camara implicada **71,2 m**, o sea imposible. Con el tope en 6 anchos
el barrido se queda en la zona donde el problema esta de verdad determinado."""
ALTURA_CAMARA_M = (1.2, 15.0)
"""Rango admisible de altura de camara sobre el suelo, en metros. Es una propiedad de las camaras de
television de esgrima, no de los tiradores, asi que usarla como puerta no contamina el control C1.
Una camara de mano va a 1,5 m y una de grada alta a 15; 71 m no existe.

Y SE COMPRUEBA QUE NO SE QUEDE PEGADA AL TOPE: en `FixedIllegalHarborporpoise` el barrido acabo
exactamente en 20,0 m, o sea en el borde de la puerta, y ahi la estatura reproyectada salia 4,16 m.
Un resultado en el borde de una puerta no es un resultado: es la puerta. Se rechaza."""
MARGEN_TOPE = 0.03
"""Margen relativo que se exige respecto a los topes de las puertas para no aceptar un resultado que
simplemente se ha quedado pegado al tope."""
VERT_MIN = 4
"""Verticales de la sala minimas para fiarse del punto de fuga vertical."""
VERT_ANG_MAX = 0.40
"""Residuo angular maximo, en grados, de la concurrencia de las verticales."""
VERT_COL_MAX = 0.08
"""Desviacion maxima del punto de fuga vertical respecto a la COLUMNA del punto principal, en anchos
de imagen. Con la camara sin balanceo y el punto principal en el centro tiene que ser cero, asi que
esta cifra mide directamente cuanto se rompen esas dos hipotesis.

ES LA PUERTA QUE SALVA UN CLIP DE DAR BASURA: en `UglyNiceAplomadofalcon` sale **728 px** de 1280, o
sea el 57 % del ancho, y ademas con el punto de fuga vertical POR ENCIMA del centro, que es
imposible con una camara que mira hacia abajo. Sin esta puerta ese clip devolvia una homografia con
tiradores de 0,93 m de estatura y una pista de 2,81 m de ancho."""
RES_FUGA_MAX = 6.0
"""Residuo maximo, en pixeles, de la concurrencia de las marcas en V2. Es EL control del modelo: si
las marcas detectadas son de verdad las transversales de una pista plana vista por una camara, tienen
que concurrir. 6 px es del orden del residuo con que se ajustan las propias rectas."""


ZONA_SIGMA = 3.5
"""Cuantas desviaciones robustas tiene que separarse el color de una columna de la pista respecto al
color principal para contar como ZONA DE OTRO COLOR."""
ZONA_MIN_COL = 12
"""Columnas seguidas minimas de una zona de color, para no confundirla con una raya pintada."""


def zonas_color(mar, pl, cob, rot, bordes) -> dict:
    """QUE HACE: detecta la ZONA DE OTRO COLOR de los ultimos 2 m de la pista, y devuelve sus dos
    bordes como marcas IDENTIFICADAS, a 5 y a 7 metros de la linea central.

    POR QUE ESTO ES LA MEJOR PISTA DE TODAS, y es una idea que vino del reglamento y no del codigo.
    t.19 dice, y esta comprobado en `scratchpad/fie_2021.txt`: *"the last 2 metres of the piste before
    these rear limit lines must be clearly distinguished -- if possible by a different colour of
    piste"*. O sea que en los dos extremos de la pista hay una REGION grande y de color distinto, no
    una raya de tres pixeles. Para un detector eso es mucho mejor: una region de dos metros de largo
    da senal en cientos de columnas, mientras que una raya pintada da senal en dos.

    Y LO QUE LA HACE DECISIVA NO ES QUE SEA FACIL DE VER, sino que viene YA IDENTIFICADA. El problema
    que hundio las otras vias es que una raya transversal detectada no dice si es la central, la de
    guardia o una junta de plancha, y sin saberlo no hay escala longitudinal. Los dos bordes de la
    zona de color, en cambio, estan a **5 m** y a **7 m** del centro por reglamento, asi que en cuanto
    se detecta la zona hay dos correspondencias punto-mundo con su metro puesto, y con los dos bordes
    longitudinales son CUATRO puntos, que es justo lo que una homografia necesita.

    COMO SE DETECTA. Sobre la tira rectificada se toma el color mediano de cada columna a lo ancho de
    la pista, se compara con el color principal de la pista y se buscan tramos seguidos que se salgan
    por encima de `ZONA_SIGMA` desviaciones robustas. Un tramo vale si tiene al menos
    `ZONA_MIN_COL` columnas y si NO ocupa casi toda la tira, porque entonces lo que ha cambiado es la
    referencia y no la pista.

    LA AMBIGUEDAD QUE QUEDA, y es inofensiva: no se sabe si la zona detectada es la del extremo +7 o
    la del -7, asi que el signo de la coordenada longitudinal queda sin decidir. Eso no afecta a
    ningun desplazamiento ni a ninguna distancia, que es lo que el usuario pidio medir, y el signo se
    fija despues por convenio (el tirador de la izquierda de la imagen va a la x negativa).
    """
    tr = mar.get("tira")
    if tr is None:
        return {"ok": False, "motivo": "sin tira"}
    ss = tr["ss"]
    PA, PB = tr["PA"], tr["PB"]
    lab = cv2.cvtColor(pl, cv2.COLOR_BGR2LAB).astype("float32")
    H, W = pl.shape[:2]
    val = (cob >= max(4, int(0.20 * cob.max()))) & (rot == 0)
    uu = np.linspace(0.18, 0.82, 24)
    """Se mira solo el 64 % central del ancho para no coger los bordes de seguridad ni las rayas
    longitudinales pintadas, que cambiarian el color de las columnas de los extremos."""
    perfil = np.full((len(ss), 3), np.nan, "float32")
    for i in range(len(ss)):
        qs = PA[i][None, :] + uu[:, None] * (PB[i] - PA[i])[None, :]
        px = np.round(qs).astype(int)
        ok = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
        if ok.sum() < 8:
            continue
        px = px[ok]
        vv = val[px[:, 1], px[:, 0]]
        if vv.sum() < 8:
            continue
        perfil[i] = np.median(lab[px[vv, 1], px[vv, 0]], axis=0)
    fin = np.isfinite(perfil).all(axis=1)
    if fin.sum() < 40:
        return {"ok": False, "motivo": "perfil de color con menos de 40 columnas"}
    base = np.median(perfil[fin], axis=0)
    d = np.full(len(ss), np.nan)
    d[fin] = np.linalg.norm(perfil[fin] - base, axis=1)
    mad = float(np.median(np.abs(d[fin] - np.median(d[fin])))) * 1.4826 + 1e-6
    umb = float(np.median(d[fin])) + ZONA_SIGMA * mad
    dist = d > umb
    tramos = []
    i = 0
    while i < len(ss):
        if not (fin[i] and dist[i]):
            i += 1
            continue
        j = i
        while j + 1 < len(ss) and fin[j + 1] and dist[j + 1]:
            j += 1
        if (j - i + 1) >= ZONA_MIN_COL and (j - i + 1) <= 0.60 * fin.sum():
            tramos.append((i, j))
        i = j + 1
    if not tramos:
        return {"ok": False, "motivo": "ninguna zona de color", "umbral": umb}
    tramos.sort(key=lambda z: -(z[1] - z[0]))
    i, j = tramos[0]
    """De los dos bordes del tramo, el que esta mas cerca del CENTRO de la tira es el de 5 m y el
    otro el de 7 m, porque la zona esta en un extremo de la pista y el centro de la pista queda
    siempre del lado de dentro."""
    c = 0.5 * (len(ss) - 1)
    interior, exterior = (i, j) if abs(i - c) < abs(j - c) else (j, i)
    ident = []
    for k, xm in ((interior, 5.0), (exterior, 7.0)):
        ident.append({"pA": PA[k], "pB": PB[k], "x_m": xm, "x_medido": xm, "linea": None,
                      "i_tira": int(k)})
    return {"ok": True, "ident": ident, "tramo": (int(i), int(j)), "n_col": int(j - i + 1),
            "d_max": float(np.nanmax(d)), "umbral": umb,
            "n_tramos": len(tramos)}


def _cruce(La, Lb):
    """QUE HACE: el punto de cruce de dos rectas, y una medida de lo mal condicionado que esta.

    La condicion importa mucho aqui: dos rectas casi paralelas se cruzan en un punto que se va al
    infinito, y su posicion es ruido puro. La cifra que se devuelve es el seno del angulo entre las
    dos rectas, que es cero cuando son paralelas.
    """
    p = np.cross(La, Lb)
    na = La[:2] / max(np.linalg.norm(La[:2]), 1e-12)
    nb = Lb[:2] / max(np.linalg.norm(Lb[:2]), 1e-12)
    sen = abs(float(na[0] * nb[1] - na[1] * nb[0]))
    if abs(p[2]) < 1e-14:
        return None, sen
    return p[:2] / p[2], sen


ANG_INLIER_V2 = 1.2
"""Angulo maximo, en grados, con el que una marca cuenta como compatible con un V2 candidato."""
F_REL_RANGO = (0.30, 6.00)
"""Rango de distancia focal admisible en anchos de imagen. Una camara de television con un teleobjetivo
largo llega a varios anchos; por debajo de 0,3 el campo de vision seria de mas de 120 grados, que no
existe en television deportiva."""


def _v2_ransac(lineas, LAc, Vz, pp, W, H, dentro) -> dict:
    """QUE HACE: elige V2 con RANSAC sobre pares de marcas, con TRES PUERTAS FISICAS.

    POR QUE NO BASTA CON MINIMOS CUADRADOS ROBUSTOS, y esta medido: en
    `GentleQuerulousElephantseal` habia 6 marcas de las que 2 eran falsas (un resto de grafico de
    television en el centro del cuadro, casi vertical). El ajuste robusto se quedo con 3 de 6 y
    eligio el subconjunto EQUIVOCADO, porque las dos falsas eran casi paralelas entre si y por tanto
    muy consistentes entre ellas. La consistencia interna de un subconjunto no basta: hay que exigir
    que el resultado sea posible.

    LAS TRES PUERTAS, todas de fisica y ninguna ajustada a un clip:

      1. V2 tiene que quedar AL OTRO LADO del borde de lejos que el borde de cerca. Las rectas
         transversales se alejan de la camara, y en la imagen alejarse es subir; asi que su punto de
         fuga esta por encima de la pista, no por debajo. Las dos marcas falsas daban un V2 a 8.957
         px POR DEBAJO del centro: imposible.
      2. la focal que implica con el punto de fuga vertical tiene que ser positiva y estar en el
         rango de una camara de television.
      3. el punto de fuga no puede estar mas alla de `FUGA_MAX` anchos de imagen.
    """
    Ls = [_a_centrada(li["L"], pp) for li in lineas]
    Ls = [L / max(np.linalg.norm(L[:2]), 1e-12) for L in Ls]
    mejor = None
    for i in range(len(Ls)):
        for j in range(i + 1, len(Ls)):
            v = np.cross(Ls[i], Ls[j])
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                continue
            v = v / nv
            if not _v2_posible(v, LAc, Vz, W, dentro):
                continue
            ang = _angulos(np.asarray(Ls), v)
            inl = ang <= ANG_INLIER_V2
            if inl.sum() < 2:
                continue
            clave = (int(inl.sum()), -float(np.median(ang[inl])))
            if mejor is None or clave > mejor[0]:
                mejor = (clave, inl)
    if mejor is None:
        return {"ok": False, "motivo": "ningun par de marcas da un V2 posible"}
    inl = mejor[1]
    M = np.asarray(Ls)[inl]
    _u, _s, vt = np.linalg.svd(M.T @ M)
    v = vt[-1]
    if not _v2_posible(v, LAc, Vz, W, dentro):
        return {"ok": False, "motivo": "el V2 refinado deja de ser posible"}
    ang = _angulos(M, v)
    return {"ok": True, "v": v, "n": int(inl.sum()), "n_total": len(Ls),
            "ang_p50": float(np.median(ang)), "ang_max": float(np.max(ang)),
            "inliers": inl}


def _v2_posible(v, LAc, Vz, W, dentro) -> bool:
    """QUE HACE: las tres puertas fisicas sobre un candidato a V2. Ver `_v2_ransac`."""
    if abs(v[2]) < 1e-12:
        return False
    q = v[:2] / v[2]
    if float(np.linalg.norm(q)) > FUGA_MAX * W:
        return False
    """El borde de lejos tiene que quedar ENTRE V2 y el borde de cerca, o sea que V2 y el origen (que
    esta dentro de la pista) caen a lados opuestos del borde de lejos."""
    s_v2 = float(LAc @ np.array([q[0], q[1], 1.0]))
    s_dentro = float(LAc @ np.array([dentro[0], dentro[1], 1.0]))
    if s_v2 * s_dentro >= 0:
        return False
    den = Vz[2] * v[2]
    if abs(den) < 1e-14:
        return False
    f2 = -(Vz[0] * v[0] + Vz[1] * v[1]) / den
    if not (F_REL_RANGO[0] ** 2 * W * W <= f2 <= F_REL_RANGO[1] ** 2 * W * W):
        return False
    return True


N_REJILLA_F = 300
"""Puntos de la rejilla logaritmica de distancia focal que se prueban."""


def _f_y_v2(V1, Vz, Ls, LAc, LBc, Oc, ancho_m, pp, W, dentro) -> dict:
    """QUE HACE: la distancia focal Y el punto de fuga transversal a la vez, imponiendo que las TRES
    direcciones del mundo sean perpendiculares entre si.

    ESTA ES LA PIEZA CENTRAL DEL MODULO y sustituye a tres intentos que fallaron. La idea: con el
    punto principal en el centro y pixel cuadrado, la matriz w = (K K')^-1 vale
    diag(1/f2, 1/f2, 1), y dos direcciones perpendiculares del mundo cumplen Vi' w Vj = 0. Si se
    conocen DOS de las tres direcciones perpendiculares (el eje de la pista, que dan los bordes, y la
    vertical, que dan las estructuras de la sala), la tercera queda DETERMINADA en cuanto se fija la
    focal:

        V2(f) = (w V1) x (w Vz)

    o sea que hay una familia de UN solo parametro, y las marcas transversales solo tienen que
    elegir de esa familia la que mejor las explica. Se barre `f` en rejilla logaritmica y gana la que
    minimiza el angulo mediano que habria que girar cada marca para que pasase por V2.

    POR QUE ASI Y NO COMO LO HACIA ANTES, con cifras. Antes calculaba V2 libremente por minimos
    cuadrados de las marcas y sacaba `f` de la ortogonalidad con la vertical. Resultado medido: el
    control de ortogonalidad entre eje y transversal salia **17,0 grados** en
    `GentleQuerulousElephantseal` y **37,9** en `FixedIllegalHarborporpoise` (deberia ser 0), y la
    estatura reproyectada salia **2,24 y 2,16 m**, un 28 % por encima de lo fisico. La causa: V2 tiene
    dos grados de libertad y las marcas solo determinan bien UNO, el que corre a lo largo del
    horizonte; el otro, la altura de V2, es justo el que fija la escala metrica y las marcas lo
    determinan fatal. Imponiendo la ortogonalidad, ese grado de libertad malo desaparece.

    Y SE GANA UNA HIPOTESIS MENOS: la camara SIN BALANCEO deja de ser necesaria. El horizonte sale de
    V1 x V2 y su inclinacion pasa a ser un CONTROL en vez de una entrada.

    `curva_f` guarda como sube el residuo al mover la focal un 5, 10 y 15 %, que es la medida de lo
    bien determinada que esta: si el residuo no sube, la focal no esta medida, esta elegida.
    """
    fs = np.exp(np.linspace(np.log(F_REL_RANGO[0] * W), np.log(F_REL_RANGO[1] * W), N_REJILLA_F))
    A = np.asarray(Ls)

    def evalua(f):
        """Devuelve (V2, puntuacion, hc). La puntuacion es una TUPLA: primero cuantas marcas quedan
        dentro del angulo de tolerancia y luego el angulo mediano de esas. Se ordena asi y no por el
        angulo mediano solo porque con dos marcas falsas casi paralelas el mediano empata entre la
        hipotesis buena y la de rectas paralelas, y contar inliers no empata."""
        w = np.array([1.0 / (f * f), 1.0 / (f * f), 1.0])
        V2 = np.cross(w * V1, w * Vz)
        nv = np.linalg.norm(V2)
        if nv < 1e-14:
            return None, (-1, 0.0), None
        V2 = V2 / nv
        if not _v2_posible_simple(V2, LAc, W, dentro):
            return V2, (-1, 0.0), None
        con = _construye_H(V1, V2, f, LBc, Oc, ancho_m, pp)
        if not con.get("ok"):
            return V2, (-1, 0.0), None
        hc = con["hc_m"]
        if not (ALTURA_CAMARA_M[0] * (1.0 + MARGEN_TOPE) <= hc
                <= ALTURA_CAMARA_M[1] * (1.0 - MARGEN_TOPE)):
            return V2, (-1, 0.0), hc
        ang = _angulos(A, V2)
        inl = ang <= ANG_INLIER_V2
        med = float(np.median(ang[inl])) if inl.any() else float(np.median(ang))
        return V2, (int(inl.sum()), -med), hc

    mejor = (None, (-1, 0.0), None, None)
    for f in fs:
        V2, sc, hc = evalua(f)
        if sc > mejor[1]:
            mejor = (V2, sc, float(f), hc)
    if mejor[0] is None or mejor[1][0] < 0:
        return {"ok": False, "motivo": "ninguna focal da un V2 posible con altura de camara "
                                       "en %.0f-%.0f m" % ALTURA_CAMARA_M}
    f0 = mejor[2]
    for f in np.linspace(f0 / 1.10, f0 * 1.10, 121):
        V2, sc, hc = evalua(f)
        if sc > mejor[1]:
            mejor = (V2, sc, float(f), hc)
    V2, sc, f, hc = mejor
    ang = _angulos(A, V2)
    curva = []
    for g in (0.80, 0.90, 0.95, 1.05, 1.10, 1.25):
        _v, s2, _h = evalua(f * g)
        curva.append((round(g, 2), int(s2[0]), round(-s2[1], 3)))
    return {"ok": True, "f": f, "V2": V2, "ang_p50": float(-sc[1]), "ang_max": float(np.max(ang)),
            "n_inliers": int(sc[0]), "n_total": len(A), "curva_f": curva, "hc_barrido": hc}


def _v2_posible_simple(V2, LAc, W, dentro) -> bool:
    """QUE HACE: la puerta fisica de que el punto de fuga transversal quede POR ENCIMA de la pista.

    Las rectas transversales se alejan de la camara, y alejarse es subir en la imagen, asi que su
    punto de fuga cae al otro lado del borde de lejos que el interior de la pista. Es la puerta que
    en `GentleQuerulousElephantseal` tira un V2 que salia 8.957 px por DEBAJO del centro.
    """
    if abs(V2[2]) < 1e-14:
        return False
    q = V2[:2] / V2[2]
    if float(np.linalg.norm(q)) > FUGA_MAX * W:
        return False
    s_v2 = float(LAc @ np.array([q[0], q[1], 1.0]))
    s_in = float(LAc @ np.array([dentro[0], dentro[1], 1.0]))
    return s_v2 * s_in < 0


def _a_centrada(L, pp):
    """QUE HACE: pasa una recta de coordenadas de imagen a coordenadas centradas en el punto
    principal."""
    L = np.asarray(L, "float64")
    return np.array([L[0], L[1], L[2] + L[0] * pp[0] + L[1] * pp[1]])


def _norm_k(v, f):
    """QUE HACE: el modulo de K^-1 v con K = diag(f, f, 1), para un punto homogeneo `v` centrado."""
    return float(np.sqrt((v[0] ** 2 + v[1] ** 2) / (f * f) + v[2] ** 2))


def homografia(bordes, mar, vert, W, H, ancho_m=ANCHO_NOMINAL) -> dict:
    """QUE HACE: la homografia pista(metros) -> imagen, y su inversa.

    LA IDEA, en una linea: las dos primeras columnas de la homografia del suelo SON las imagenes de
    los puntos del infinito de los dos ejes, o sea los dos puntos de fuga, y la tercera es la imagen
    del origen. Asi que basta con obtener V1 (eje de la pista), V2 (transversal) y un origen.

    LOS CINCO INGREDIENTES Y DE DONDE SALE CADA UNO:

      V1, direccion del eje de la pista   los dos BORDES longitudinales (`region_y_bordes`)
      Vz, direccion vertical              las ESTRUCTURAS FIJAS de la sala (`verticales`)
      f y V2                              de imponer que las tres direcciones sean perpendiculares,
                                          con las MARCAS transversales eligiendo la focal (`_f_y_v2`)
      escala                              el ancho de pista `ancho_m`, declarado
      origen                              donde la marca elegida corta el borde A, para que el cero
                                          longitudinal caiga EN una raya pintada

    Ninguno de los cinco usa a los tiradores, que es el requisito duro de la tarea. Los tiradores
    solo dicen DONDE MIRAR (el corredor de sus apoyos), y de ahi no sale ningun numero.

    LO QUE ESTA CONSTRUCCION ASUME, y son tres cosas, no mas:
      1. punto principal en el centro de la imagen
      2. pixel cuadrado y sin inclinacion de eje
      3. el suelo es plano y las marcas son perpendiculares a los bordes (t.19 del reglamento)
    La camara SIN BALANCEO ya NO se asume: el horizonte sale de V1 x V2 y su inclinacion es un
    control.

    SIN NINGUNA MARCA TRANSVERSAL, o sin verticales, no hay homografia metrica: se devuelve
    `ok: False` con el motivo y el clip se queda con la linea base. Eso es lo que hay que decir, no
    rellenarlo con una suposicion sobre la distancia focal.
    """
    pp = _pp(W, H)
    LAc = _a_centrada(bordes["recta_A"], pp)
    LBc = _a_centrada(bordes["recta_B"], pp)
    lineas = [li for li in (mar.get("lineas") or []) if li.get("afinada")]
    if len(lineas) < 1:
        return {"ok": False, "motivo": "sin marca transversal afinada: no hay V2"}
    usada = max(lineas, key=lambda z: z.get("n_pts", 0))
    if not vert.get("ok"):
        return {"ok": False, "motivo": "sin punto de fuga vertical: " + vert.get("motivo", "?")}

    V1 = np.cross(LAc, LBc)
    nn = np.linalg.norm(V1)
    if nn < 1e-12:
        return {"ok": False, "motivo": "los dos bordes son la misma recta"}
    V1 = V1 / nn
    _v1p, sen_bordes = _cruce(np.asarray(bordes["recta_A"], "float64"),
                              np.asarray(bordes["recta_B"], "float64"))
    Vz = np.asarray(vert["v"], "float64")
    Vz = Vz / max(np.linalg.norm(Vz), 1e-12)
    dentro_c = np.asarray(bordes["centro_pista"], "float64") - pp
    Ls = [_a_centrada(li["L"], pp) for li in lineas]
    Ls = [L / max(np.linalg.norm(L[:2]), 1e-12) for L in Ls]
    Oc0 = np.cross(_a_centrada(usada["L"], pp), LAc)
    if abs(Oc0[2]) < 1e-12:
        return {"ok": False, "motivo": "la marca elegida no corta el borde A en un punto propio"}
    Oc0 = Oc0 / Oc0[2]
    rfv = _f_y_v2(V1, Vz, Ls, LAc, LBc, Oc0, ancho_m, pp, W, dentro_c)
    if not rfv.get("ok"):
        return {"ok": False, "motivo": "focal y V2: " + rfv.get("motivo", "?")}
    f = rfv["f"]
    V2 = rfv["V2"]
    via = "ortogonal_%dmarcas" % len(Ls)
    fuente_f = "marcas sobre la familia ortogonal"
    ang_v2 = rfv["ang_p50"]
    n_v2 = rfv["n_inliers"]
    curva_f = rfv["curva_f"]

    n1 = _norm_k(V1, f)
    n2 = _norm_k(V2, f)
    """La ortogonalidad se cumple ahora por construccion, asi que su residuo mide precision numerica
    y no calidad del modelo. Se deja porque un residuo grande delataria un fallo de codigo."""
    ortog_deg = 90.0 - float(np.degrees(np.arccos(np.clip(
        abs((V1[0] * V2[0] + V1[1] * V2[1]) / (f * f) + V1[2] * V2[2]) / max(n1 * n2, 1e-12),
        -1.0, 1.0))))
    hor_c = np.cross(V1, V2)
    incl_hor = float("nan")
    if np.linalg.norm(hor_c[:2]) > 1e-12:
        incl_hor = abs(float(np.degrees(np.arctan2(hor_c[0], hor_c[1]))))
        incl_hor = min(incl_hor, 180.0 - incl_hor)

    Oc = np.cross(_a_centrada(usada["L"], pp), LAc)
    if abs(Oc[2]) < 1e-12:
        return {"ok": False, "motivo": "la marca no corta el borde A en un punto propio"}
    Oc = Oc / Oc[2]
    con = _construye_H(V1, V2, f, LBc, Oc, ancho_m, pp)
    if not con.get("ok"):
        return {"ok": False, "motivo": con.get("motivo", "?"), "via": via}
    Hm, Hi, hc, vzh, mu, K = (con["H"], con["Hi"], con["hc_m"], con["vz_h"], con["mu"], con["K"])
    return {"ok": True, "H": Hm, "Hi": Hi, "V1": V1, "V2": V2, "Vz": Vz, "f": f, "K": K,
            "ancho_m": ancho_m, "marca_usada": usada, "n_marcas": len(lineas),
            "via": via, "fuente_f": fuente_f, "hc_m": hc, "vz_h": vzh, "curva_f": curva_f,
            "ortog_deg": ortog_deg, "sen_bordes": sen_bordes,
            "ang_fuga_marcas_deg": ang_v2, "n_marcas_fuga": n_v2,
            "incl_horizonte_deg": incl_hor,
            "ang_fuga_vert_deg": vert.get("ang_p50", float("nan")),
            "n_vert": vert.get("n", 0), "n_vert_total": vert.get("n_total", 0),
            "desv_columna_vz_px": vert.get("desv_columna_px", float("nan")),
            "pp": pp, "mu": mu}


def a_metros(hg, pts_img: np.ndarray) -> np.ndarray:
    """QUE HACE: pasa puntos de la imagen a metros sobre la pista. `pts_img` es (..., 2)."""
    q = np.asarray(pts_img, "float64")
    sh = q.shape[:-1]
    q = q.reshape(-1, 2)
    o = np.full((len(q), 2), np.nan)
    fin = np.isfinite(q).all(axis=1)
    if fin.any():
        hh = np.concatenate([q[fin], np.ones((fin.sum(), 1))], axis=1).T
        w = hg["Hi"] @ hh
        with np.errstate(all="ignore"):
            o[fin] = (w[:2] / w[2]).T
    return o.reshape(*sh, 2)


def a_imagen(hg, pts_m: np.ndarray) -> np.ndarray:
    """QUE HACE: la vuelta, de metros sobre la pista a pixeles de la imagen de referencia."""
    q = np.asarray(pts_m, "float64")
    sh = q.shape[:-1]
    q = q.reshape(-1, 2)
    hh = np.concatenate([q, np.ones((len(q), 1))], axis=1).T
    w = hg["H"] @ hh
    with np.errstate(all="ignore"):
        return (w[:2] / w[2]).T.reshape(*sh, 2)


def altura_m(hg, pie_img, cab_img):
    """QUE HACE: la altura en metros de un segmento VERTICAL del que se conoce el pie en el suelo.

    LA FORMULA, exacta y sin aproximaciones. Con la camara P = K[R|t] y el suelo en z=0, la
    homografia del suelo es H = K[r1 r2 t]. Un punto a altura h sobre el punto del suelo (X,Y)
    proyecta en

        p(h) = K(R(X,Y,h)' + t) = H(X,Y,1)' + h * (K r3)

    o sea que la imagen de la cabeza esta en la recta que une la imagen del pie con `K r3`, que es el
    punto de fuga vertical, y `h` es la unica incognita. Igualando la imagen medida m de la cabeza,

        m x ( p_pie + h * vz ) = 0     ->    h = -(m x p_pie)_i / (m x vz)_i

    para la componente i con el denominador mas grande, que es la que menos se degrada. No hay
    razones de cruce escritas a mano ni altura de camara metida a mano: todo sale de H y de K.
    """
    pm = a_metros(hg, np.asarray(pie_img, "float64"))
    if not np.isfinite(pm).all():
        return None
    p_pie = hg["H"] @ np.array([pm[0], pm[1], 1.0])
    m = np.array([cab_img[0], cab_img[1], 1.0])
    a = np.cross(m, p_pie)
    b = np.cross(m, hg["vz_h"])
    i = int(np.argmax(np.abs(b)))
    if abs(b[i]) < 1e-12:
        return None
    return float(-a[i] / b[i])


def control_estatura(hg, r, Mk=None) -> dict:
    """CONTROL C1, EL QUE DECIDE. QUE HACE: reproyecta la ESTATURA de los dos tiradores y mira si cae
    en el rango fisico 1,60-2,00 m.

    POR QUE ES UN CONTROL DE VERDAD: los tiradores no entran en el ajuste de la homografia en ningun
    sitio. Solo entran en la SEMILLA que dice donde buscar la pista, y esa semilla no aporta ningun
    numero a la homografia: el eje, los dos bordes, las marcas, V1, V2, f y la escala salen todos de
    la pista. Asi que la estatura reproyectada es una PREDICCION del modelo, no un ajuste.

    Y ES ADEMAS EL CONTROL QUE PEDIA EL USUARIO para separar dos cosas que se confunden: si el largo
    reproyectado de la pista sale 15,8 m en vez de 14, puede ser una pista con plancha de mas o puede
    ser la homografia mal ajustada. La estatura lo separa, porque una plancha de mas NO cambia la
    estatura de nadie: si la estatura sale bien y el largo sale largo, la pista es larga.

    LO QUE ESTA MEDIDA NO DICE, y hay que tenerlo delante: la caja de YOLO no acaba en la coronilla
    ni empieza en la planta del pie. En un fondo el tirador esta inclinado, asi que la caja mide
    menos que su estatura, y ademas recoge la punta del pie adelantada. Por eso se toma la MEDIANA
    del clip y no un cuadro, se exige un rango ancho (1,60-2,00) y se da tambien el p90, que es el
    que se acerca a la estatura de verdad porque corresponde a los cuadros en que esta erguido.
    """
    B, pres, mask, W, H = r["B"], r["pres"], r["mask"], r["W"], r["H"]
    alturas = {0: [], 1: []}
    for k in np.nonzero(mask)[0]:
        M = Mk[k] if Mk is not None else np.eye(3)
        for lado in (0, 1):
            if not pres[k, lado] or not np.isfinite(B[k, lado]).all():
                continue
            b = B[k, lado]
            xc = 0.5 * (b[0] + b[2]) * W
            pie = M @ np.array([xc, b[3] * H, 1.0])
            cab = M @ np.array([xc, b[1] * H, 1.0])
            h = altura_m(hg, pie[:2] / pie[2], cab[:2] / cab[2])
            if h is not None and 0.3 < h < 5.0:
                alturas[lado].append(h)
    out = {"ok": True}
    todas = []
    for lado in (0, 1):
        a = np.asarray(alturas[lado])
        todas.extend(list(a))
        out["h%d_p50" % lado] = float(np.median(a)) if a.size else float("nan")
        out["h%d_n" % lado] = int(a.size)
    t = np.asarray(todas)
    if not t.size:
        return {"ok": False, "motivo": "sin cajas utilizables", "pasa": False}
    out["h_p50"] = float(np.median(t))
    out["h_p10"] = float(np.percentile(t, 10))
    out["h_p90"] = float(np.percentile(t, 90))
    out["n"] = int(t.size)
    out["pasa"] = bool(1.60 <= out["h_p50"] <= 2.00)
    return out


def control_marcas(hg, mar) -> dict:
    """CONTROL C2. QUE HACE: reproyecta la posicion de TODAS las marcas detectadas y mira si las
    separaciones entre ellas caen cerca de las distancias FIE.

    ES UN CONTROL Y NO UN AJUSTE porque solo UNA marca entra en la homografia (la que da V2). Todas
    las demas son prediccion. Con una sola marca detectada el control no existe y se dice.

    QUE SE COMPARA. Las marcas pintadas estan a 0, +-2 y +-7 m del centro, y la frontera de color a
    +-5 m. Asi que las separaciones POSIBLES entre dos marcas cualesquiera son 2, 3, 4, 5, 7, 9, 12 y
    14 m. Se da, por cada par, la distancia reproyectada y el error al valor posible mas cercano.

    EL CONFUNDIDO QUE HAY QUE DECLARAR: la pista es de planchas y las JUNTAS entre planchas tambien
    son crestas transversales. Una junta a 1,2 m de una linea de guardia daria un error de 0,8 m
    contra la separacion posible de 2 m, y eso es lo que este control ve. No se puede distinguir una
    junta de una raya solo por su geometria; lo unico honrado es dar el error y decir que las
    separaciones que no encajan pueden ser juntas.
    """
    lin = mar.get("lineas") or []
    if len(lin) < 2:
        return {"ok": False, "motivo": "menos de 2 marcas: el control no existe", "n": len(lin)}
    posibles = np.array([2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 12.0, 14.0])
    xs = []
    for li in lin:
        pm = a_metros(hg, np.asarray(li["centro"], "float64"))
        if np.isfinite(pm).all():
            xs.append(float(pm[0]))
    xs = np.sort(np.asarray(xs))
    pares, errs = [], []
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            d = abs(xs[j] - xs[i])
            if d < 0.5:
                continue
            e = float(np.min(np.abs(posibles - d)))
            pares.append((round(d, 3), round(float(posibles[int(np.argmin(np.abs(posibles - d)))]), 2),
                          round(e, 3)))
            errs.append(e)
    if not errs:
        return {"ok": False, "motivo": "marcas todas a la misma x", "n": len(lin)}
    return {"ok": True, "n": len(lin), "x_m": [round(float(v), 3) for v in xs],
            "pares": pares, "err_p50": float(np.median(errs)),
            "err_min": float(np.min(errs))}


def control_perpendicular(hg, mar) -> dict:
    """CONTROL C3. QUE HACE: mide, en grados, cuanto se desvian de la perpendicular las marcas que NO
    se usaron para construir V2.

    ES EL CONTROL DIRECTO DE LA HIPOTESIS DE CAMARA SIN BALANCEO Y DE PUNTO PRINCIPAL EN EL CENTRO.
    Si esas dos hipotesis fallan, V2 sale mal colocado y las demas marcas reproyectan torcidas.
    Con una sola marca detectada el control no existe.
    """
    lin = mar.get("lineas") or []
    usada = hg.get("marca_usada")
    ang = []
    for li in lin:
        if usada is not None and li is usada:
            continue
        c = np.asarray(li["centro"], "float64")
        L = np.asarray(li["L"], "float64")
        d = np.array([-L[1], L[0]])
        pa = a_metros(hg, c - 12.0 * d)
        pb = a_metros(hg, c + 12.0 * d)
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        v = pb - pa
        ang.append(abs(float(np.degrees(np.arctan2(abs(v[0]), abs(v[1]))))))
    if not ang:
        return {"ok": False, "motivo": "no hay marca sobrante para controlar"}
    return {"ok": True, "n": len(ang), "desv_p50": float(np.median(ang)),
            "desv_max": float(np.max(ang))}


ESTATURA_REF = 1.75
"""Estatura de referencia para deducir el ancho de la pista. Es la mediana aproximada de un
floretista de elite; el proyecto ya midio contra 427 alturas publicadas que los priores de genero
salen 1,705 m en mujeres y 1,818 en hombres (`04_documentacion/BITACORA_2026-09.md`, seccion 7 del informe historico),
asi que 1,75 es el punto medio y su error propio es del orden del 3 %."""


def mide_pista(hg, bordes, ctrl_h) -> dict:
    """QUE HACE: MIDE el largo VISIBLE de la pista y DEDUCE su ancho, sin imponer los 14 m.

    ESTO ES LO QUE PIDIO EL USUARIO: *"algunas pistas pueden variar algo la longitud o el ancho por
    tener alguna plancha extra, detectalo tambien"*. Asi que aqui no se compara contra 14,00 para
    corregir nada: se mide y se da la cifra.

    TRES ADVERTENCIAS, Y LAS TRES CAMBIAN LO QUE ESTA CIFRA SIGNIFICA:

    1. EL ANCHO REPROYECTADO ES CIRCULAR y no vale como medida. El ancho `ancho_m` ES la escala con
       la que se construye la homografia, asi que los dos bordes reproyectan a y=0 y y=ancho_m por
       construccion, exactamente. Dar ese numero como "ancho medido" seria hacerse trampas. Lo que
       si es informacion es la DISPERSION de los puntos del barrido alrededor de su recta, en metros:
       eso mide el ruido del detector de bordes, no el ancho.

    2. EL ANCHO SE DEDUCE DE LA ESTATURA, y ahi si hay una medida independiente. Como todas las
       distancias escalan linealmente con `ancho_m`, si la estatura reproyectada sale `h` cuando
       deberia salir 1,75 m, entonces el ancho verdadero es `ancho_m * 1,75 / h`. Esa es la unica
       medida del ancho de la pista que este modulo puede dar, y su precision es la de la estatura.

    3. EL LARGO TOTAL NO ES OBSERVABLE en un clip, y lo midio el estudio historico: el cuadro de
       television abarca una mediana de 8,3 m de pista y NINGUNO de sus 348 cuadros abarcaba los
       14 m. Asi que lo que se mide es el largo VISIBLE, que no se puede comparar con 14,00 m. La
       unica via para detectar una pista con plancha de mas es el control C2 sobre las separaciones
       entre marcas, y hace falta que se vean dos marcas.
    """
    A = np.asarray(bordes["pts_A"], "float64")
    B = np.asarray(bordes["pts_B"], "float64")
    ma = a_metros(hg, A)
    mb = a_metros(hg, B)
    fa = np.isfinite(ma).all(axis=1)
    fb = np.isfinite(mb).all(axis=1)
    if fa.sum() + fb.sum() < 8:
        return {"ok": False, "motivo": "bordes no reproyectables"}
    disp = np.concatenate([np.abs(ma[fa, 1] - 0.0),
                           np.abs(mb[fb, 1] - hg["ancho_m"])])
    xs = np.concatenate([ma[fa, 0], mb[fb, 0]])
    h50 = ctrl_h.get("h_p50", float("nan")) if ctrl_h else float("nan")
    anc = hg["ancho_m"] * ESTATURA_REF / h50 if np.isfinite(h50) and h50 > 0.3 else float("nan")
    return {"ok": True,
            "disp_borde_p95_m": float(np.percentile(disp, 95)),
            "largo_visible_m": float(np.percentile(xs, 98) - np.percentile(xs, 2)),
            "ancho_deducido_m": float(anc),
            "ancho_en_rango_fie": bool(np.isfinite(anc) and FIE_ANCHO[0] * 0.9 <= anc
                                       <= FIE_ANCHO[1] * 1.1),
            "n": int(fa.sum() + fb.sum())}


FIE_POS = (-7.0, -5.0, -2.0, 0.0, 2.0, 5.0, 7.0)
"""Posiciones, en metros desde la linea central, de las marcas que el reglamento pone en la pista:
limite posterior (-7), frontera de color de los ultimos 2 m (-5), puesta en guardia (-2), central
(0) y sus simetricas. De t.19, comprobado en `scratchpad/fie_tech.txt`.

Y LO QUE ESTA LISTA NO ES: no es la lista de todas las rectas transversales que hay en el suelo. Una
pista es de PLANCHAS y sus juntas tambien son transversales. Por eso la identificacion se hace con
RANSAC: se busca el subconjunto de marcas detectadas que encaja con este patron y las demas se
declaran no identificadas, que es lo que son."""
TOL_IDENT_M = 0.20
"""Tolerancia, en metros, para que una marca cuente como encajada en una posicion del patron FIE.
20 cm es del orden del ancho de las bandas pintadas que se ven en este corpus y bastante menor que
la separacion minima del patron, que es 2,00 m: el margen es 10x."""
MIN_IDENT = 3
"""Marcas identificadas minimas. Con 2 el sistema esta exactamente determinado y no hay ningun
control; con 3 la tercera es una comprobacion, y esa es la diferencia entre una medida y una
suposicion."""


def identifica_marcas(lineas, bordes, pp) -> dict:
    """QUE HACE: identifica que marca detectada es la central, cual la de guardia, etc., SIN NECESITAR
    LA DISTANCIA FOCAL. Es la pieza que hace metrica la coordenada a lo largo de la pista.

    POR QUE SE PUEDE HACER SIN FOCAL, y es un resultado geometrico exacto. Sobre una recta del suelo
    (por ejemplo el borde A de la pista), la coordenada del mundo X y la coordenada de la imagen `t`
    se relacionan por una proyectividad de la recta. Una proyectividad de la recta tiene 3 grados de
    libertad, pero UNO de ellos ya se conoce: el punto de la imagen que corresponde a X infinito es
    el punto de fuga V1, que dan los dos bordes. Con ese punto fijado, la relacion se reduce a

        X = A * s + B      con   s = 1 / (t - t_V1)     (o s = t si V1 esta en el infinito)

    o sea a una AFINIDAD de dos parametros. Y una afinidad conserva las RAZONES de distancias. Asi
    que las razones de separacion entre tres o mas marcas se pueden medir sin saber nada de la
    camara, y compararlas con el patron del reglamento identifica las marcas y da la escala.

    POR QUE ESTO IMPORTA MAS QUE LA VIA DE LA FOCAL. Con la camara casi perpendicular al eje, que es
    el encuadre normal, el punto de fuga longitudinal se va al infinito y sacar la focal de ahi esta
    mal condicionado; medido, daba f2 negativa y una altura de camara implicada de **71,2 m**. Esta
    via NO toca la focal: la escala sale de las rayas pintadas, que es lo que el usuario pidio
    explicitamente cuando dijo que las lineas pintadas son mas fiables como referencia que el largo
    total de la pista.

    Y ADEMAS DA LA VUELTA A LA MEDIDA DEL ANCHO. Con la escala longitudinal puesta por las rayas, el
    ancho de la pista deja de ser una entrada y pasa a ser algo que se DEDUCE, que es justo lo que
    pedia el usuario al avisar de que algunas pistas varian de medida.

    LO QUE ESTA VIA NECESITA: tres marcas identificables. Con dos, el sistema queda exactamente
    determinado y no hay ningun control, asi que el clip se declara no calibrable. Eso es un
    resultado y no un fallo, y se cuenta.
    """
    LAc = _a_centrada(bordes["recta_A"], pp)
    LBc = _a_centrada(bordes["recta_B"], pp)
    V1 = np.cross(LAc, LBc)
    if np.linalg.norm(V1) < 1e-12:
        return {"ok": False, "motivo": "bordes degenerados"}
    dirA = np.array([-LAc[1], LAc[0]], "float64")
    dirA = dirA / np.linalg.norm(dirA)
    pA0 = -LAc[2] * np.array([LAc[0], LAc[1]], "float64")
    ptos, ts = [], []
    for li in lineas:
        Lc = _a_centrada(li["L"], pp)
        q = np.cross(Lc, LAc)
        qb = np.cross(Lc, LBc)
        if abs(q[2]) < 1e-12 or abs(qb[2]) < 1e-12:
            continue
        qa = q[:2] / q[2]
        ptos.append((qa, qb[:2] / qb[2], li))
        ts.append(float((qa - pA0) @ dirA))
    if len(ts) < MIN_IDENT:
        return {"ok": False, "motivo": "menos de %d marcas cortan los dos bordes" % MIN_IDENT,
                "n": len(ts)}
    ts = np.asarray(ts)
    if abs(V1[2]) > 1e-12:
        tv1 = float((V1[:2] / V1[2] - pA0) @ dirA)
        with np.errstate(all="ignore"):
            sp = 1.0 / (ts - tv1)
    else:
        """V1 en el infinito, que es el caso de la toma lateral: la relacion ya es afin."""
        sp = ts.copy()
    if not np.isfinite(sp).all():
        return {"ok": False, "motivo": "una marca cae sobre el punto de fuga"}
    fie = np.asarray(FIE_POS)
    mejor = None
    n = len(sp)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for xi in FIE_POS:
                for xj in FIE_POS:
                    if xi == xj:
                        continue
                    den = sp[j] - sp[i]
                    if abs(den) < 1e-14:
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
                    clave = (int(inl.sum()), -float(np.median(err[inl])))
                    if mejor is None or clave > mejor[0]:
                        mejor = (clave, A, B, inl.copy())
    if mejor is None:
        return {"ok": False,
                "motivo": "ningun subconjunto de %d marcas encaja con el patron FIE" % MIN_IDENT,
                "n": n}
    _clave, A, B, inl = mejor
    X = A * sp + B
    ident = []
    for k in range(n):
        if not inl[k]:
            continue
        xf = float(fie[int(np.argmin(np.abs(X[k] - fie)))])
        ident.append({"pA": ptos[k][0], "pB": ptos[k][1], "linea": ptos[k][2],
                      "x_m": xf, "x_medido": float(X[k])})
    if len({d["x_m"] for d in ident}) < MIN_IDENT:
        return {"ok": False, "motivo": "dos marcas caen en la misma posicion del patron", "n": n}
    errs = [abs(d["x_medido"] - d["x_m"]) for d in ident]
    return {"ok": True, "ident": ident, "n": int(inl.sum()), "n_total": n,
            "err_p50_m": float(np.median(errs)), "err_max_m": float(np.max(errs)),
            "X_todas": [round(float(v), 3) for v in X],
            "patron": [d["x_m"] for d in ident], "A": float(A), "B": float(B)}


def homografia_por_marcas(bordes, ident, W, H, ancho_m=ANCHO_NOMINAL) -> dict:
    """QUE HACE: la homografia pista(metros) -> imagen por DLT, con las marcas ya identificadas.

    Cada marca identificada da DOS correspondencias punto-mundo: donde corta el borde A es
    `(X_k, 0)` y donde corta el borde B es `(X_k, ancho_m)`. Con tres marcas son seis puntos para los
    ocho grados de libertad de la homografia, o sea sobredeterminado, y el residuo de reproyeccion en
    pixeles es un CONTROL de todo el ajuste.

    LA FOCAL NO ENTRA EN NINGUN SITIO. Sale despues, como consecuencia, al pedir que la homografia
    venga de una camara de pixel cuadrado, y su valor es entonces un control mas.

    EL ANCHO SIGUE SIENDO LA ESCALA TRANSVERSAL y sigue declarado, porque la escala transversal no la
    dan las rayas. Pero la escala LONGITUDINAL, que es la que importa para medir un fondo o un
    desplazamiento, ya no depende de el.
    """
    src, dst = [], []
    for d in ident:
        src.append([d["x_m"], 0.0])
        dst.append([float(d["pA"][0]), float(d["pA"][1])])
        src.append([d["x_m"], ancho_m])
        dst.append([float(d["pB"][0]), float(d["pB"][1])])
    src = np.asarray(src, "float64")
    dst = np.asarray(dst, "float64")
    if len(src) < 4:
        return {"ok": False, "motivo": "menos de 4 correspondencias"}
    msk = None
    if len(src) >= 6:
        Hm, msk = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    else:
        Hm = cv2.getPerspectiveTransform(src.astype("float32"), dst.astype("float32"))
    if Hm is None:
        return {"ok": False, "motivo": "findHomography no converge"}
    pr = Hm @ np.concatenate([src, np.ones((len(src), 1))], axis=1).T
    with np.errstate(all="ignore"):
        pr = (pr[:2] / pr[2]).T
    res = np.linalg.norm(pr - dst, axis=1)
    return {"ok": True, "H": Hm, "res_p50_px": float(np.median(res)),
            "res_max_px": float(np.max(res)), "n_ptos": len(src),
            "inliers": int(msk.sum()) if msk is not None else len(src)}


def _plausible(hg) -> bool:
    """QUE HACE: la puerta de PLAUSIBILIDAD DE CAMARA sobre una homografia ya construida.

    ES LA UNICA PUERTA QUE SE LE PUEDE PONER A UNA VIA SIN CONTAMINAR EL CONTROL, y por eso solo mira
    la altura de camara y la distancia focal, que son propiedades de la camara de television y no
    tienen nada que ver con los tiradores. Si se usara la estatura para elegir entre vias, la estatura
    dejaria de ser un control independiente y no quedaria ninguno.

    LA MIDE UN CASO REAL: la via de la zona de color daba alturas de camara de **28,9 m** en
    `UglyNiceAplomadofalcon`, **86,1 m** en `ForsakenFlippantKarakul` y **89,9 m** en
    `AliveIgnorantBarebirdbat`. Ninguna camara de esgrima esta a 90 m del suelo, asi que la zona que
    el detector encontro no era la zona de los ultimos 2 m del reglamento: era otra cosa (una sombra,
    una moqueta distinta, una valla). La puerta lo dice sin necesidad de mirar la imagen.
    """
    if not hg or not hg.get("ok"):
        return False
    hc = hg.get("hc_m")
    fr = hg.get("f_rel") or (hg.get("f", 0.0) / max(hg.get("W", 1.0), 1.0))
    if hc is None or not np.isfinite(hc):
        return False
    if not (ALTURA_CAMARA_M[0] <= hc <= ALTURA_CAMARA_M[1]):
        return False
    return True


def completa(hg, W, H) -> dict:
    """QUE HACE: rellena una homografia venida del DLT con todo lo que hace falta para MEDIR ALTURAS,
    y devuelve la focal implicada como CONTROL.

    La homografia del suelo por si sola no permite medir la estatura de nadie: para eso hace falta el
    punto de fuga vertical, y ese sale de exigir que la homografia venga de una camara de pixel
    cuadrado con el punto principal en el centro. La focal que lo consigue se busca minimizando el
    incumplimiento de las dos condiciones de camara, |K^-1 h1| = |K^-1 h2| y h1' w h2 = 0, y su
    residuo es una cifra que dice si el ajuste es coherente con una camara de verdad.
    """
    pp = _pp(W, H)
    Hm = np.asarray(hg["H"], "float64")
    h1, h2 = Hm[:, 0], Hm[:, 1]
    h1c = np.array([h1[0] - pp[0] * h1[2], h1[1] - pp[1] * h1[2], h1[2]])
    h2c = np.array([h2[0] - pp[0] * h2[2], h2[1] - pp[1] * h2[2], h2[2]])
    mejor = None
    for f in np.exp(np.linspace(np.log(F_REL_RANGO[0] * W), np.log(F_REL_RANGO[1] * W), 600)):
        n1 = _norm_k(h1c, f)
        n2 = _norm_k(h2c, f)
        if n1 < 1e-15 or n2 < 1e-15:
            continue
        ort = abs((h1c[0] * h2c[0] + h1c[1] * h2c[1]) / (f * f) + h1c[2] * h2c[2]) / (n1 * n2)
        iso = abs(n1 - n2) / (0.5 * (n1 + n2))
        sc = ort + iso
        if mejor is None or sc < mejor[0]:
            mejor = (sc, float(f), ort, iso)
    if mejor is None:
        return {"ok": False, "motivo": "no se puede completar la camara"}
    sc, f, ort, iso = mejor
    K = np.array([[f, 0.0, pp[0]], [0.0, f, pp[1]], [0.0, 0.0, 1.0]])
    A = np.linalg.inv(K) @ Hm
    mu = 0.5 * (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
    if mu < 1e-15:
        return {"ok": False, "motivo": "descomposicion degenerada"}
    r1, r2 = A[:, 0] / mu, A[:, 1] / mu
    r3 = np.cross(r1, r2)
    tv = A[:, 2] / mu
    Hs = Hm.copy()
    if float((Hs @ np.array([0.0, 0.5 * hg["ancho_m"], 1.0]))[2]) < 0:
        Hs = -Hs
        A = np.linalg.inv(K) @ Hs
        mu = 0.5 * (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
        r1, r2 = A[:, 0] / mu, A[:, 1] / mu
        r3 = np.cross(r1, r2)
        tv = A[:, 2] / mu
    vzh = mu * (K @ r3)
    p0 = Hs @ np.array([0.0, 0.5 * hg["ancho_m"], 1.0])
    p1 = p0 + vzh
    if p1[2] != 0 and p0[2] != 0 and (p1[1] / p1[2]) > (p0[1] / p0[2]):
        vzh = -vzh
    try:
        Hi = np.linalg.inv(Hs)
    except np.linalg.LinAlgError:
        return {"ok": False, "motivo": "homografia singular"}
    out = dict(hg)
    out.update({"ok": True, "H": Hs, "Hi": Hi, "f": f, "K": K, "mu": mu,
                "hc_m": float(abs(r3 @ tv)), "vz_h": vzh,
                "resid_camara": float(sc), "ortog_deg": float(np.degrees(np.arcsin(min(ort, 1.0)))),
                "isotropia": float(iso), "f_rel": f / W})
    return out


DESPL_CAMARA_MAX = 12.0
"""Desplazamiento maximo de una esquina del cuadro, en pixeles, para que UNA sola homografia valga
para todo el clip. Por encima, la homografia se corrige cuadro a cuadro con la similitud de
estabilizacion, que es lo que hace `analiza`, y se deja la cifra en el informe."""


def analiza(clip: str, video: str = None, frames: list = None,
            ancho_m: float = ANCHO_NOMINAL) -> dict:
    """QUE HACE: todo el paso 2 sobre un clip, y devuelve la homografia con sus tres controles.

    EL ORDEN IMPORTA y cada paso depende del anterior: rotulos -> estabilizacion -> placa limpia ->
    corredor -> bordes -> marcas -> homografia -> controles.
    """
    r = pareja(clip)
    if r is None:
        return {"clip": clip, "ok": False, "motivo": "sin pickle de detecciones"}
    if not r["mask"].any():
        return {"clip": clip, "ok": False, "motivo": "sin tramo utilizable"}
    W, H, T = r["W"], r["H"], r["T"]
    if frames is None:
        if video is None:
            video = video_clip(clip)
        if video is None:
            return {"clip": clip, "ok": False, "motivo": "sin video en disco"}
        frames = lee(video, T)
    import pareja_v2 as _PAR
    if _PAR.PIERNA_K > 0:
        """La pareja otra vez, ahora con los fotogramas, para que coincida con la de la etapa 1
        (puerta de piernas, PAREJA_2026-10-09). Con la puerta apagada no se rehace nada."""
        r = pareja(clip, frames=frames)
        if not r["mask"].any():
            return {"clip": clip, "ok": False, "motivo": "sin tramo utilizable"}
    ks = [int(v) for v in np.nonzero(r["mask"])[0] if v < len(frames)]
    if len(ks) < 12:
        return {"clip": clip, "ok": False, "motivo": "menos de 12 cuadros utilizables leidos"}
    jsel = np.linspace(0, len(ks) - 1, min(N_PLACA, len(ks))).astype(int)
    sel = [ks[j] for j in jsel]
    ref = int(jsel[len(jsel) // 2])
    cero = np.zeros((H, W), "uint8")
    """DOS PASADAS DE ESTABILIZACION, y hacen falta por una dependencia circular: la mascara de
    rotulos se mide comparando el sistema de pantalla con el estabilizado, asi que necesita la
    estabilizacion; y la estabilizacion necesita la mascara para no poner puntos de seguimiento
    sobre un rotulo, que estan quietos en pantalla y arrastran la estimacion hacia cero. La primera
    pasada es un arranque: con RANSAC los puntos del rotulo son minoria y se rechazan."""
    M0, dmax0 = estabiliza(frames, sel, r["cajas"], r["cuadros"], W, H, cero)
    rot, sd_fuera, via_rot = mascara_rotulos(frames, sel, M0, r["cajas"], r["cuadros"], W, H, dmax0)
    Mall, dmax = estabiliza(frames, ks, r["cajas"], r["cuadros"], W, H, rot, ref=ref)
    Mk = np.tile(np.eye(3), (T, 1, 1))
    for j, k in enumerate(ks):
        Mk[k] = Mall[j]
    Ms = [Mall[j] for j in jsel]
    pl, cob = placa(frames, sel, Ms, r["cajas"], r["cuadros"], W, H)
    if pl is None:
        return {"clip": clip, "ok": False, "motivo": "no se pudo construir la placa"}
    P = _pies_caja(r)
    co = corredor(P, r["mask"], Mk)
    if co is None:
        return {"clip": clip, "ok": False, "motivo": "corredor con menos de 8 apoyos"}
    al = (r["B"][:, :, 3] - r["B"][:, :, 1])[r["pres"]]
    alto_px = float(np.nanmedian(al)) * H if al.size else 0.0
    bo = region_y_bordes(pl, cob, rot, co, alto_px)
    out = {"clip": clip, "W": W, "H": H, "T": T, "utiles": int(r["mask"].sum()),
           "rot_frac": float((rot > 0).mean()), "sd_fuera_rotulo": sd_fuera, "via_rot": via_rot,
           "alto_caja_px": alto_px,
           "despl_camara_px": dmax, "camara_quieta": bool(dmax <= DESPL_CAMARA_MAX),
           "n_col_barrido": bo.get("n_col", 0),
           "_bo": bo, "_co": co, "_pl": pl, "_rot": rot, "_cob": cob,
           "_Mk": Mk, "_sel": sel, "_ks": ks, "_r": r, "_frames": frames}
    if not bo.get("ok"):
        out.update({"ok": False, "motivo": "bordes: " + bo.get("motivo", "?")})
        return out
    out["res_borde_A_px"] = bo["res_A"]
    out["res_borde_B_px"] = bo["res_B"]
    mar = marcas(pl, cob, rot, co, bo)
    out["_mar"] = mar
    lin_af = [li for li in (mar.get("lineas") or []) if li.get("afinada")]
    out["n_marcas"] = len(mar.get("lineas") or [])
    out["n_marcas_afinadas"] = len(lin_af)
    out["ancho_pista_px"] = mar.get("ancho_px", float("nan"))
    ve = verticales(pl, rot, cob, bo, co)
    out["_ve"] = ve
    out["n_vert"] = ve.get("n", 0)
    out["ang_fuga_vert_deg"] = ve.get("ang_p50", float("nan"))
    out["desv_columna_vz_px"] = ve.get("desv_columna_px", float("nan"))
    out["vz_rel_H"] = ve.get("vz_rel_H", float("nan"))
    """DOS VIAS INDEPENDIENTES, y la que manda es la de las rayas pintadas.

      via A, RAYAS: identifica las marcas por sus razones de separacion, que no necesitan focal, y
              ajusta la homografia por DLT. La escala longitudinal la ponen las rayas del
              reglamento. Es la buena, y es la que el usuario pidio.
      via B, CAMARA: saca la focal del punto de fuga vertical de la sala e impone ortogonalidad. No
              necesita identificar ninguna marca, asi que sirve donde la via A no llega, pero esta
              mal condicionada con la camara casi perpendicular al eje.

    Cuando las dos salen, la razon entre sus escalas longitudinales es un CONTROL de los buenos,
    porque las dos vias no comparten ninguna medida: una usa las rayas del suelo y la otra la
    geometria de la camara."""
    idt = identifica_marcas(lin_af, bo, _pp(W, H)) if len(lin_af) >= MIN_IDENT else \
        {"ok": False, "motivo": "menos de %d marcas afinadas" % MIN_IDENT}
    out["_idt"] = idt
    out["n_ident"] = idt.get("n", 0)
    out["ident_err_p50_m"] = idt.get("err_p50_m", float("nan"))
    out["ident_patron"] = idt.get("patron")
    out["motivo_ident"] = idt.get("motivo", "")
    hgB = homografia(bo, mar, ve, W, H, ancho_m=ancho_m)
    out["motivo_camara"] = hgB.get("motivo", "")
    out["ok_camara"] = bool(hgB.get("ok"))
    if hgB.get("ok"):
        out["f_rel_camara"] = hgB["f"] / W
        out["hc_m_camara"] = hgB["hc_m"]
    """TRES VIAS, Y LA ELIGE UN CONTROL INDEPENDIENTE, no el orden en que estan escritas.

      Z, ZONA DE COLOR   los dos bordes de la zona de otro color de los ultimos 2 m (t.19) estan a
                         5 y a 7 m del centro POR REGLAMENTO, asi que vienen identificados; con los
                         dos bordes longitudinales dan las cuatro correspondencias que necesita una
                         homografia. No supone nada de la camara. Y las marcas transversales NO
                         entran en el ajuste, asi que su separacion reproyectada (el control C2) es
                         un control totalmente independiente de esta via.
      A, RAYAS           identifica las marcas por sus razones de separacion, que son invariantes
                         afines y no necesitan focal. Necesita cuatro marcas identificadas.
      B, CAMARA          la focal sale del punto de fuga vertical de la sala y de imponer que las
                         tres direcciones del mundo sean perpendiculares. Supone punto principal en
                         el centro y pixel cuadrado.

    COMO SE ELIGE, y por que no por orden. La primera version prefería Z siempre, y el resultado fue
    PEOR: en `UglyNiceAplomadofalcon` la via de la camara daba estatura 1,63 m y ancho 1,61 m, y la
    de la zona de color daba estatura 3,53 m y una camara a 28,9 m de altura. O sea que la zona que
    el detector encuentra a menudo NO es la zona del reglamento: es una sombra, una moqueta distinta
    o una valla. Elegir por orden es elegir por fe.

    Ahora las tres vias se construyen, se les exige PLAUSIBILIDAD DE CAMARA (altura entre 1,2 y 15 m,
    que es una propiedad de las camaras de television y no toca a los tiradores) y gana la que tenga
    el error de C2 mas bajo. Si ninguna es plausible, el clip se queda sin homografia y se cuenta."""
    zc = zonas_color(mar, pl, cob, rot, bo)
    out["_zc"] = zc
    out["zona_color"] = bool(zc.get("ok"))
    out["zona_n_col"] = zc.get("n_col", 0)
    out["motivo_zona"] = zc.get("motivo", "")
    cands = []

    def _mete(nombre, cand, extra=None):
        if cand is None or not cand.get("ok"):
            return
        cand["ancho_m"] = ancho_m
        cand = completa(cand, W, H) if "f" not in cand else cand
        if not cand.get("ok"):
            return
        cand["via"] = nombre
        cand.setdefault("marca_usada", None)
        cand.setdefault("fuente_f", "consecuencia del DLT")
        cm = control_marcas(cand, mar)
        e2 = cm.get("err_p50") if cm.get("ok") else None
        out["hc_%s" % nombre] = cand["hc_m"]
        out["C2_%s" % nombre] = e2
        if not _plausible(cand):
            out["descarta_%s" % nombre] = "camara a %.1f m" % cand["hc_m"]
            return
        cands.append((float(e2) if e2 is not None else 9.9, nombre, cand))

    if zc.get("ok"):
        _mete("zona", homografia_por_marcas(bo, zc["ident"], W, H, ancho_m=ancho_m))
    if idt.get("ok") and idt["n"] >= 4:
        """LA VIA DE LAS RAYAS SOLO CON CUATRO MARCAS IDENTIFICADAS O MAS, y la razon es un recuento
        de probabilidad: con 3 marcas hay 3 pares posibles por 42 pares del patron FIE, o sea 126
        hipotesis, y cada una acierta por azar con probabilidad del orden de 7 x 0,40 / 14 = 0,20.
        Con 126 intentos, encontrar un encaje falso es casi seguro. Medido: con 3 marcas,
        `GentleQuerulousElephantseal` identificaba el patron [-7, -5, 0] y `UglyNiceAplomadofalcon`
        el [-7, -2, 2], los dos con error 0,00 m por construccion, y los dos daban homografias con
        residuo DLT de 206 px y estaturas de 0,93 m. Tres marcas no identifican una pista."""
        _mete("rayas", homografia_por_marcas(bo, idt["ident"], W, H, ancho_m=ancho_m))
    if hgB.get("ok"):
        hgB["f_rel"] = hgB["f"] / W
        _mete("camara", hgB)
    if not cands:
        out.update({"ok": False,
                    "motivo": "sin homografia plausible. zona: %s | rayas: %s | camara: %s"
                              % (zc.get("motivo", "-") or out.get("descarta_zona", "-"),
                                 idt.get("motivo", "-") or out.get("descarta_rayas", "-"),
                                 hgB.get("motivo", "-") or out.get("descarta_camara", "-"))})
        return out
    cands.sort(key=lambda z: z[0])
    hg = cands[0][2]
    out["vias_plausibles"] = ",".join(n for _e, n, _c in cands)
    if len(cands) > 1:
        """CONTROL CRUZADO ENTRE DOS VIAS INDEPENDIENTES: se mide el mismo tramo de 4,00 m con las
        dos y se da la razon. Dos vias que no comparten ninguna medida y coinciden es dificil que
        esten mal de la misma manera."""
        otra = cands[1][2]
        pa = a_metros(otra, a_imagen(hg, np.array([0.0, 0.5 * ancho_m])))
        pb = a_metros(otra, a_imagen(hg, np.array([4.0, 0.5 * ancho_m])))
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            out["razon_escalas"] = float(abs(pb[0] - pa[0]) / 4.0)
    ch = control_estatura(hg, r, Mk)
    cm = control_marcas(hg, mar)
    cp = control_perpendicular(hg, mar)
    mp = mide_pista(hg, bo, ch)
    out.update({"ok": True, "via": hg["via"], "f_rel": hg["f"] / W, "hc_m": hg["hc_m"],
                "fuente_f": hg["fuente_f"],
                "ortog_deg": hg["ortog_deg"],
                "h_p50": ch.get("h_p50"), "h_p10": ch.get("h_p10"), "h_p90": ch.get("h_p90"),
                "h0_p50": ch.get("h0_p50"), "h1_p50": ch.get("h1_p50"),
                "C1_pasa": ch.get("pasa"),
                "C2_err_p50": cm.get("err_p50"), "C2_err_min": cm.get("err_min"),
                "C2_n": cm.get("n"), "C2_pares": cm.get("pares"), "C2_x_m": cm.get("x_m"),
                "C3_desv_p50": cp.get("desv_p50"), "C3_desv_max": cp.get("desv_max"),
                "largo_visible_m": mp.get("largo_visible_m"),
                "disp_borde_p95_m": mp.get("disp_borde_p95_m"),
                "ancho_deducido_m": mp.get("ancho_deducido_m"),
                "ancho_en_rango_fie": mp.get("ancho_en_rango_fie")})
    out["_hg"] = hg
    hh = out.get("h_p50")
    out["publicable"] = bool(out.get("C1_pasa") and out.get("n_marcas_afinadas", 0) >= 2
                             and hh is not None and np.isfinite(float(hh)))
    """LA PUERTA DE PUBLICACION, declarada antes de correr nada (ver la cabecera del modulo): la
    homografia se publica solo si la ESTATURA REPROYECTADA de los dos tiradores cae en 1,60-2,00 m,
    que es el rango fisico de un floretista de elite, y si hay al menos dos marcas transversales
    medidas de verdad. Un clip que no pasa no se arregla: se cuenta y se entrega con la linea base."""
    h0, h1 = out.get("h0_p50"), out.get("h1_p50")

    def _rg(v):
        return v is not None and np.isfinite(float(v)) and 1.60 <= float(v) <= 2.00
    out["C1_dos_lados"] = bool(h0 is not None and h1 is not None
                               and np.isfinite(float(h0)) and np.isfinite(float(h1)))
    out["C1_dos_lados_pasa"] = bool(_rg(h0) and _rg(h1))
    out["publicable_estricto"] = bool(out["publicable"] and out["C1_dos_lados_pasa"])
    """LA PUERTA ESTRICTA, y esta columna existe porque la de arriba NO HACE LO QUE DICE SU PROPIO
    CRITERIO. El criterio declarado es *"la estatura reproyectada de los DOS tiradores cae en
    1,60-2,00 m"*, en plural, y `C1_pasa` compara la mediana AGRUPADA de las alturas de los dos. Con
    dos tiradores a 2,23 m y 1,79 m la mediana agrupada sale 1,82 m y PASA, cuando ninguno de los dos
    esta bien medido.

    LA DIFERENCIA, medida sobre los 78 clips del corpus etiquetado:

      puerta agrupada, la de `publicable`   **10 de 78 = 12,8 %**
      puerta por tirador, la declarada      **2 de 78 = 2,6 %**  (`AliveIgnorantBarebirdbat` y
                                            `VibrantJoyousAsiaticgreaterfreshwaterclam`)

    Y hay un caso que la agrupada no separa de ninguna manera: **9 de los 10 publicables tienen los
    dos lados medidos y 1 no**. `ForcefulAntiqueAlbacoretuna` pasa con un solo tirador (h0 1,93 m,
    h1 sin cajas utilizables), y sus controles fisicos independientes dicen que su escala esta
    inflada unas 2,4 veces: separacion de pies 1,48 m cuando la guardia de un floretista son
    0,50-0,90, largo visible 12,65 m cuando la mediana del corpus son 5,53, y 8,6 % de apoyos fuera
    de la pista.

    NO SE CAMBIA `publicable`, para no mover las cifras de la seccion 23 por debajo de quien las
    haya citado. Se anade esta columna y se dan las dos."""
    return out


N_COL_TIRA = 960
N_FIL_TIRA = 160
MARGEN_TIRA = 0.40
"""Cuanto se sale la tira de cada borde, en fracciones del ancho de la pista. Hace falta para VER la
plancha lateral: si la tira acabara en el borde detectado, lo de fuera no se podria medir y la
plancha es justo lo que hay que encontrar fuera."""
ANCHO_MIN_REL = 0.30
"""Hasta donde se extiende la tira a lo largo: se para cuando el ancho de la pista en pixeles baja de
esta fraccion del ancho en el centro del corredor. NO es un criterio estetico, es el limite de
resolucion: donde el ancho se reduce a un tercio, un pixel vale el triple de metros, y ahi el error
de cualquier cosa detectada se multiplica por tres. `px_img_por_col` lo entrega columna a columna
para poder dar el error en metros en funcion de la posicion, que es lo unico honesto: 1 px son 2 cm
de cerca y 20 cm en el extremo lejano."""
PASO_EXT = 4.0
ZONA_INTERIOR_M = 5.00
ZONA_EXTERIOR_M = 7.00
ZONA_FONDO_M = 2.00
PLANTILLA_ZONAS = (-7.0, -5.0, 5.0, 7.0)
"""Las cuatro fronteras de ZONA de la pista, en metros desde la linea central: el limite posterior y
la frontera de la zona de los ultimos 2 m, en los dos extremos (t.19)."""
FRONT_RAZON_MIN = 3.0
FRONT_MARGEN_COL = 24
FRONT_FUSION_COL = 30
SEG_MIN_COL = 20
TOL_FRONT_M = 0.30
"""Tolerancia para que una frontera detectada encaje con una posicion del reglamento. Mas suelta que
la de las rayas pintadas (0,20 m) porque una frontera de zona es un cambio de material con un borde
de varios centimetros, no una raya de 4 cm."""
LARGO_VISIBLE_M = (2.0, 20.0)
"""Largo visible admisible de la pista en la tira, en metros. No es un ajuste fino: es lo que impide
aceptar una hipotesis que confunda dos posiciones del patron, porque el factor de escala que mete
saca el largo implicado del rango fisico. La pista mide 14 m, el encuadre la puede cortar, y la tira
se extiende por las rectas de los bordes un poco mas alla de los extremos."""


def tira_rectificada(res, ncol=N_COL_TIRA, nfil=N_FIL_TIRA, margen=MARGEN_TIRA) -> dict:
    """QUE HACE: la placa sin personas RECTIFICADA con los dos bordes detectados como las dos filas
    fijas `u = 0` y `u = 1`, y el eje largo extendido hasta donde la pista se sale del encuadre o
    pierde resolucion.

    =======================================================================================
    LA PROPIEDAD QUE HACE QUE ESTO VALGA, y es exacta
    =======================================================================================

    El usuario lo dijo asi: *"como sabes que tiene que ser recto, da igual la perspectiva, por saber
    que es recto lo puedes cuadrar perfectamente sin error, que punto es"*. Es cierto y tiene una
    consecuencia que conviene escribir. Mandando los dos bordes a dos rectas HORIZONTALES, el punto
    de fuga longitudinal `V = A x B` va al corte de dos horizontales, o sea al infinito horizontal.
    Que `V = (1,0,0)` vaya a `(1,0,0)` obliga a que la primera columna de la homografia mundo -> tira
    sea `(h11, 0, 0)`, y entonces

        columna = (h11 x + h12 y + h13) / (h32 y + h33)     fila = (h22 y + h23) / (h32 y + h33)

    La FILA no depende de `x`, o sea que las rectas paralelas al eje de la pista son filas; y a fila
    fija, la COLUMNA es AFIN en `x`. Comprobado numericamente con la homografia de referencia de tres
    clips: su punto de fuga longitudinal cae en la columna 2,5e10, 4,1e9 y -2,5e10, o sea en el
    infinito en los tres.

    Consecuencia practica: las separaciones entre marcas se pueden comparar con las del reglamento
    SIN corregir perspectiva, que es lo que hace que el patron 2:3:2 sirva.

    =======================================================================================
    DOS FALLOS MIOS QUE ESTA VERSION ARREGLA, y los dos los delato el censo de 20 tiras
    =======================================================================================

    1. **SE USABA `co["t"]` COMO EJE LARGO Y ES EL EJE CORTO.** `corredor` devuelve `s` = recorrido a
       lo LARGO y `t` = dispersion a lo ANCHO de la nube de apoyos. La primera version muestreaba
       `ss = linspace(t0, t1, 420)`, o sea estiraba 30 px de ancho a 420 columnas. Sintoma: `largo
       29 px` con `ancho 102 px` en `UglyNiceAplomadofalcon`, y las 20 tiras salieron con franjas
       horizontales perfectas, sin NINGUNA variacion a lo largo. Una pista de 14 m no puede salir mas
       corta que ancha, y esa comprobacion de una linea habria ahorrado el censo entero.

    2. **LA TIRA ERA UN PARALELOGRAMO, NO LA PISTA.** Las filas se trazaban a distancia constante del
       origen del corredor, o sea paralelas EN LA IMAGEN. Los dos bordes de la pista no son paralelos
       en la imagen: convergen en el punto de fuga. Asi que la fila `u = 0` no era el borde A, y mi
       docstring que decia "los dos bordes son dos filas fijas de la tira" era FALSO salvo en la toma
       exactamente frontal. Con el eje largo roto no se veia, porque en 30 px no converge nada.

    =======================================================================================
    LO QUE LA TIRA NO DA, y hay que decirlo porque es la clave de todo lo demas
    =======================================================================================

    La escala A LO LARGO (metros por columna) **no sale de los bordes**. Dos rectas paralelas con
    separacion conocida fijan la escala TRANSVERSAL en todo el plano, pero no dicen nada de la
    longitudinal: hace falta una referencia mas (una marca a distancia conocida, el punto de fuga
    transversal, o la focal). Es un resultado geometrico, no una limitacion de esta implementacion.
    """
    pl, co, bo = res.get("_pl"), res.get("_co"), res.get("_bo")
    if pl is None or co is None or bo is None or not bo.get("ok"):
        return None
    LA = np.asarray(bo["recta_A"], "float64")
    LB = np.asarray(bo["recta_B"], "float64")
    e = np.asarray(co["e"], "float64")
    c0 = np.asarray(co["c0"], "float64")
    H, W = pl.shape[:2]
    s0, s1 = co["s"]
    smid = 0.5 * (s0 + s1)

    def corte(s):
        """QUE HACE: los dos puntos donde la perpendicular al eje en `s` corta a los dos bordes.

        La perpendicular en la IMAGEN no es una transversal del mundo. Para ACOTAR el tramo visible
        da igual, y para medir no se usa: lo que se mide son filas y columnas, no estos cortes.
        """
        Lc = np.array([e[0], e[1], -(e @ c0 + s)], "float64")
        """`_cruce` de este modulo devuelve (punto, seno del angulo), no solo el punto: aqui solo
        hace falta el punto, y confundirlo daba un array inhomogeneo."""
        return _cruce(Lc, LA)[0], _cruce(Lc, LB)[0]

    pa, pb = corte(smid)
    if pa is None or pb is None:
        return None
    ancho_mid = float(np.linalg.norm(np.asarray(pa) - np.asarray(pb)))
    if not (10.0 < ancho_mid < 500.0):
        return None

    def vale(s):
        qa, qb = corte(s)
        if qa is None or qb is None:
            return False
        if float(np.linalg.norm(np.asarray(qa) - np.asarray(qb))) < ANCHO_MIN_REL * ancho_mid:
            return False
        for p in (qa, qb):
            if not (2.0 <= p[0] <= W - 3.0 and 2.0 <= p[1] <= H - 3.0):
                return False
        return True

    lim = []
    for signo in (-1.0, 1.0):
        s = smid
        while abs(s - smid) < 4000.0 and vale(s + signo * PASO_EXT):
            s += signo * PASO_EXT
        lim.append(s)
    sini, sfin = lim
    if sfin - sini < 20.0:
        return None
    pA0, pB0 = corte(sini)
    pA1, pB1 = corte(sfin)
    f0 = margen / (1.0 + 2.0 * margen) * (nfil - 1)
    f1 = (1.0 + margen) / (1.0 + 2.0 * margen) * (nfil - 1)
    src = np.array([pA0, pB0, pB1, pA1], "float32")
    dst = np.array([[0.0, f0], [0.0, f1], [ncol - 1.0, f1], [ncol - 1.0, f0]], "float32")
    Ht = cv2.getPerspectiveTransform(src, dst)
    tira = cv2.warpPerspective(pl, Ht, (ncol, nfil), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    cob = res.get("_cob")
    ct = None
    if cob is not None:
        ct = cv2.warpPerspective(cob.astype("float32"), Ht, (ncol, nfil),
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0.0)
    Hi = np.linalg.inv(Ht)
    cc = np.arange(ncol, dtype="float64")
    mid = 0.5 * (f0 + f1)
    p = Hi @ np.stack([cc, np.full(ncol, mid), np.ones(ncol)])
    p = p[:2] / p[2]
    d = np.gradient(p, axis=1)
    anchos = []
    for c in (0.0, 0.5 * (ncol - 1), ncol - 1.0):
        qa = Hi @ np.array([c, f0, 1.0])
        qb = Hi @ np.array([c, f1, 1.0])
        anchos.append(float(np.linalg.norm(qa[:2] / qa[2] - qb[:2] / qb[2])))
    return {"tira": tira, "cob": ct, "H_img_tira": Ht, "H_tira_img": Hi,
            "fil0": f0, "fil1": f1, "ncol": ncol, "nfil": nfil,
            "s_ini": sini, "s_fin": sfin, "esquinas_img": src.astype("float64"),
            "ancho_px_ini": anchos[0], "ancho_px_mid": anchos[1], "ancho_px_fin": anchos[2],
            "px_img_por_col": np.hypot(d[0], d[1]),
            "fil_por_m": (f1 - f0) / ANCHO_NOMINAL,
            "m_por_fil": ANCHO_NOMINAL / (f1 - f0)}


def a_tira(t, pts) -> np.ndarray:
    """QUE HACE: puntos de la imagen estabilizada -> (columna, fila) de la tira."""
    P = np.atleast_2d(np.asarray(pts, "float64"))
    q = np.asarray(t["H_img_tira"], "float64") @ \
        np.concatenate([P, np.ones((len(P), 1))], axis=1).T
    return (q[:2] / q[2]).T


def u_de_fila(t, fil):
    """QUE HACE: fila de la tira -> fraccion del ancho entre los dos bordes detectados (0 y 1)."""
    return (np.asarray(fil, "float64") - t["fil0"]) / (t["fil1"] - t["fil0"])


def error_a_lo_largo(t, m_por_col=None) -> dict:
    """QUE HACE: el MAPA DE ERROR de la tira: cuantos metros vale un pixel de imagen en cada punto a
    lo largo de la pista.

    POR QUE NO SE PUEDE DAR UN SOLO NUMERO, y el usuario hablaba de "precision milimetrica": con la
    camara oblicua el error crece con la distancia. Un pixel vale `m_por_col / px_img_por_col`
    metros, y `px_img_por_col` cae hacia el extremo lejano. Se entrega el perfil y sus extremos, no
    una cifra unica, porque la cifra unica seria falsa en los dos extremos.
    """
    px = np.asarray(t["px_img_por_col"], "float64")
    out = {"px_img_por_col_ini": float(px[0]), "px_img_por_col_fin": float(px[-1]),
           "razon_resolucion": float(px.max() / max(px.min(), 1e-9))}
    if m_por_col:
        err = float(m_por_col) / np.maximum(px, 1e-9)
        out.update({"m_por_px_ini": float(err[0]), "m_por_px_fin": float(err[-1]),
                    "m_por_px_p50": float(np.median(err))})
    return out


def _escalones(perfil, margen_px, razon_min=FRONT_RAZON_MIN, max_cortes=3):
    """QUE HACE: los cortes de un perfil de una sola variable donde hay un ESCALON, por segmentacion
    binaria recursiva.

    CRITERIO: la diferencia de la MEDIANA de los dos lados del corte tiene que pasar de `razon_min`
    veces la dispersion dentro de los lados, y el corte tiene que estar a mas de `margen_px` del
    extremo del tramo.

    SE QUITA LA TENDENCIA LINEAL ANTES, y hace falta: la iluminacion de un pabellon cambia a lo largo
    de la pista, y una rampa de luz produce un escalon espurio en cualquier corte del medio. Quitar
    una recta no puede inventar un escalon, pero no quitarla si puede.
    """
    v = np.asarray(perfil, "float64")
    m = np.isfinite(v)
    if m.sum() < 12:
        return []
    x = np.arange(len(v), dtype="float64")
    a, b = np.polyfit(x[m], v[m], 1)
    v = v - (a * x + b)
    margen = max(2, int(round(margen_px)))
    fuera = []

    def parte(i0, i1, nivel):
        if nivel > max_cortes or i1 - i0 < 2 * margen:
            return
        mejor = (0.0, None, 0.0)
        for i in range(i0 + margen, i1 - margen):
            A, B = v[i0:i], v[i:i1]
            A, B = A[np.isfinite(A)], B[np.isfinite(B)]
            if A.size < margen or B.size < margen:
                continue
            salto = abs(float(np.median(A)) - float(np.median(B)))
            disp = max(float(np.std(A)), float(np.std(B)), 1e-6)
            r = salto / disp
            if r > mejor[0]:
                mejor = (r, i, salto)
        if mejor[1] is None or mejor[0] < razon_min:
            return
        fuera.append({"i": int(mejor[1]), "razon": float(mejor[0]), "salto": float(mejor[2])})
        parte(i0, int(mejor[1]), nivel + 1)
        parte(int(mejor[1]), i1, nivel + 1)

    parte(0, len(v), 1)
    fuera.sort(key=lambda z: z["i"])
    return fuera


def ancho_combate(t) -> dict:
    """QUE HACE: el ancho de la ZONA DE COMBATE en la tira, excluyendo la plancha lateral anadida.

    POR QUE HACE FALTA: t.18.3 dice que los bordes conductores de seguridad **no forman parte de la
    pista**, y el usuario lo vio: *"esa pista en concreto tiene una plancha mas de ancho, un trozo
    para que no se caigan, pero eso no es valido, y esta de otro color"*. Si la plancha entra en el
    ancho de referencia, el ancho sale mayor que el reglamentario y **toda la transversal se escala
    mal**.

    Se buscan los cambios de color A LO ANCHO sobre el perfil de filas y se toma el tramo uniforme
    mas grande como zona de combate. Devuelve las dos filas que la acotan, en unidades de `u`.
    """
    tira = t["tira"]
    lab = cv2.cvtColor(cv2.GaussianBlur(tira, (5, 5), 0), cv2.COLOR_BGR2LAB)
    perfil = np.median(lab[:, :, 0].astype("float64"), axis=1)
    f0, f1 = t["fil0"], t["fil1"]
    esc = _escalones(perfil, max(1.0, 0.12 * (f1 - f0)), max_cortes=3)
    cortes = [0] + [d["i"] for d in esc] + [len(perfil)]
    tramos = [(cortes[i], cortes[i + 1]) for i in range(len(cortes) - 1)]
    if not tramos:
        return {"ok": False, "motivo": "sin tramos a lo ancho"}
    i0, i1 = max(tramos, key=lambda z: z[1] - z[0])
    u0 = float(u_de_fila(t, i0))
    u1 = float(u_de_fila(t, min(i1, t["nfil"] - 1)))
    """LA PLANCHA SE DECLARA SI el tramo mayor no cubre los dos bordes detectados: eso significa que
    el barrido de bordes se paro en un cambio de color que no era el borde de la zona de combate.

    Y SE EXIGE QUE EL TRAMO CAIGA DENTRO DE LOS BORDES, o sea `u` entre -0,10 y 1,10, porque si no
    esto miente con consecuencias. Medido sobre los 22 clips: el tramo mayor salia `u = -0,40 a 1,40`
    en 18 de ellos, o sea TODO el muestreo incluido el suelo de fuera, y el detector devolvia
    `ancho_rel = 1,8` con `plancha = False`. Quien lo usaba para renormalizar la transversal
    (`posicion_entre_bordes`) se comia un factor 1,8: un apoyo en el borde de la pista salia a 0,33 m
    del borde en vez de a 0,00, y los dos tiradores salian a 0,50-0,56 m, o sea los dos pegados al
    centro. El sintoma en el video era claro y por poco se publica: los dos puntos casi en la misma
    linea del mapa.

    Cuando el perfil a lo ancho no acota nada dentro de la pista, el resultado correcto no es un
    tramo cualquiera: es DECIR QUE NO SE HA MEDIDO, y que el que lo use se quede con los dos bordes
    detectados, que si estan medidos."""
    dentro = bool(-0.10 <= u0 < u1 <= 1.10 and (u1 - u0) >= 0.50)
    plancha = bool(dentro and (u0 > 0.12 or u1 < 0.88))
    if not dentro:
        return {"ok": False, "motivo": "el tramo uniforme a lo ancho (u %.2f a %.2f) no cae dentro "
                                       "de los bordes detectados" % (u0, u1),
                "u0_bruto": u0, "u1_bruto": u1, "n_escalones_ancho": len(esc),
                "plancha": None}
    return {"ok": True, "u0": u0, "u1": u1, "n_escalones_ancho": len(esc),
            "plancha": plancha, "ancho_rel": u1 - u0,
            "ancho_combate_m": (min(u1, 1.0) - max(u0, 0.0)) * ANCHO_NOMINAL}


def perfil_color(t, u_int=(0.20, 0.80)) -> dict:
    """QUE HACE: el color mediano de cada columna DENTRO de la pista, en Lab.

    Se toma el 60 % central del ancho para no coger ni la plancha lateral ni el borde conductor.

    SE DEVUELVE (3, ncol) Y NO (ncol, 3), y el orden importa porque fue un fallo: la segmentacion
    indexaba `med[:, i0:i1]` como canales por columnas y recibia columnas por canales, asi que el
    perfil que segmentaba tenia longitud 3 y salian CERO fronteras en los 20 clips. Cero en todos los
    casos casi nunca es un dato: es una ruta mal enchufada.
    """
    f0, f1 = t["fil0"], t["fil1"]
    i0 = int(round(f0 + u_int[0] * (f1 - f0)))
    i1 = int(round(f0 + u_int[1] * (f1 - f0)))
    i0, i1 = max(0, i0), min(t["tira"].shape[0], i1)
    if i1 - i0 < 4:
        return None
    lab = cv2.cvtColor(cv2.GaussianBlur(t["tira"], (5, 5), 0), cv2.COLOR_BGR2LAB)
    banda = lab[i0:i1].astype("float64")
    return {"med": np.median(banda, axis=0).T, "filas": (i0, i1)}


def _fronteras_color(med, margen=FRONT_MARGEN_COL, razon_min=FRONT_RAZON_MIN, max_cortes=4):
    """QUE HACE: los cortes del perfil de color donde hay un cambio de REGION, por segmentacion
    binaria recursiva.

    El criterio es el mismo que en `_escalones` y por la misma razon: la diferencia de la MEDIANA de
    los dos lados dividida por la dispersion dentro de los lados. Aqui la diferencia es la distancia
    euclidea en Lab, que ve un cambio de tono aunque no cambie el brillo; un perfil de luminancia
    solo no distingue naranja de gris del mismo brillo, y en este corpus hay las dos cosas.
    """
    n = med.shape[1]
    fuera = []

    def parte(i0, i1, nivel):
        if nivel > max_cortes or i1 - i0 < 2 * margen:
            return
        mejor = (0.0, None, 0.0)
        for i in range(i0 + margen, i1 - margen):
            A, B = med[:, i0:i], med[:, i:i1]
            sa = float(np.linalg.norm(np.median(A, axis=1) - np.median(B, axis=1)))
            disp = max(float(np.mean(np.std(A, axis=1))), float(np.mean(np.std(B, axis=1))), 1e-6)
            r = sa / disp
            if r > mejor[0]:
                mejor = (r, i, sa)
        if mejor[1] is None or mejor[0] < razon_min:
            return
        fuera.append({"col": float(mejor[1]), "razon": float(mejor[0]), "salto": float(mejor[2])})
        parte(i0, int(mejor[1]), nivel + 1)
        parte(int(mejor[1]), i1, nivel + 1)

    parte(0, n, 1)
    fuera.sort(key=lambda d: d["col"])
    return fuera


def _funde_contiguas(fr, ventana=FRONT_FUSION_COL):
    """QUE HACE: funde las fronteras contiguas que la segmentacion binaria parte en dos o tres.

    SE FUNDE DESDE LA MAS FUERTE Y CON VENTANA FIJA, no por encadenamiento, y esto es un fallo
    medido: encadenando, las candidatas 272, 288, 310, 326, 348 y 361 de `SeparateHoarseFeline`
    estan cada una a menos de la ventana de la siguiente, asi que la cadena se las comio todas en una
    marca de 89 columnas (1,3 m) y el clip paso de 14 candidatas a 5 con otra identificacion. Con
    ventana fija alrededor de la mas fuerte, ningun grupo puede pasar de `ventana` columnas.
    """
    if not fr:
        return []
    resto = sorted(fr, key=lambda d: -d["razon"])
    fuera = []
    while resto:
        jefe = resto[0]
        g = [d for d in resto if abs(d["col"] - jefe["col"]) <= 0.5 * ventana]
        resto = [d for d in resto if abs(d["col"] - jefe["col"]) > 0.5 * ventana]
        w = np.array([d["razon"] for d in g], "float64")
        x = np.array([d["col"] for d in g], "float64")
        fuera.append({"col": float((w * x).sum() / w.sum()), "razon": float(w.max()),
                      "n_fundidas": len(g)})
    fuera.sort(key=lambda d: d["col"])
    return fuera


def _asigna_inyectiva(xm, plantilla, tol):
    """QUE HACE: asignacion INYECTIVA de marcas a posiciones del reglamento, la mas cercana primero.

    ES EL ARREGLO DE UN FALLO MEDIDO: `identifica_marcas` permite que dos marcas caigan en la MISMA
    posicion del patron y las cuenta las dos. Y lo hace: en el CSV de 78 clips hay patrones
    `[-5,-2,-2,0]`, `[-7,-7,-2,2,7]` y `[-2,-2,0,0,2,2]`. Con posiciones repetidas el recuento de
    marcas identificadas esta inflado y el error de encaje sale 0 por construccion. Aqui cada
    posicion se usa una vez y el resto se declaran no identificadas, que es lo que son.
    """
    pl = np.asarray(plantilla, "float64")
    E = np.abs(np.asarray(xm, "float64")[:, None] - pl[None, :])
    pares = sorted([(float(E[i, j]), i, j) for i in range(E.shape[0])
                    for j in range(E.shape[1]) if E[i, j] <= tol])
    ui, uj, asig = set(), set(), {}
    for err, i, j in pares:
        if i in ui or j in uj:
            continue
        ui.add(i)
        uj.add(j)
        asig[i] = (float(pl[j]), err)
    return asig


def ancla_zonas(t, u_int=(0.20, 0.80)) -> dict:
    """QUE HACE: intenta el ANCLA MEDIDA por las FRONTERAS DE ZONA DE COLOR, y entrega su ambiguedad
    y su error de prediccion.

    =======================================================================================
    POR QUE REGIONES Y NO RAYAS: es un cambio de primitiva, no un ajuste
    =======================================================================================

    Las marcas que el reglamento pone son de dos clases muy distintas de medir:

      RAYAS PINTADAS (central, puesta en guardia) miden 3-4 cm, o sea 3-4 columnas de esta tira
      FRONTERAS DE ZONA (t.19) son el borde de una REGION de 2 m, o sea 150-300 columnas

    Medido sobre los 20 clips con tira: por rayas, el emparejamiento sale AMBIGUO en 10 de 14, y en
    los tres clips con homografia de referencia la escala que dan se equivoca por factores de 1,3 a
    2,6. Las fronteras de zona se detectan con razones senal-ruido de 3 a 30.

    =======================================================================================
    Y AUN ASI ESTA VIA TAMPOCO DA METROS PUBLICABLES, y la cifra lo dice
    =======================================================================================

    Sobre los 20 clips con tira, esta via encaja en 13, pero **en 11 de esos 13 solo con DOS
    fronteras**, y con dos el sistema esta exactamente determinado: no hay ningun control. En los
    dos clips con tres fronteras el error de prediccion de la reservada sale de 3 a 8 cm, que es
    bueno, pero son dos clips. Y contra la homografia de referencia, la escala se separa un 32 %.

    Asi que lo que se publica de aqui es el ORIGEN cuando encaja, con su ambiguedad, y NO se publica
    la escala como medida buena. La posicion a lo largo se entrega normalizada al tramo visible, que
    es una coordenada honesta y acotada, y en metros solo cuando hay ancla y con su incertidumbre.
    """
    pc = perfil_color(t, u_int)
    out = {"ok": False, "n_fronteras": 0}
    if pc is None:
        out["motivo"] = "banda interior de la pista demasiado estrecha"
        return out
    fr = _funde_contiguas(_fronteras_color(pc["med"]))
    out["n_fronteras"] = len(fr)
    out["cols"] = [round(d["col"], 1) for d in fr]
    out["razones"] = [round(d["razon"], 2) for d in fr]
    if len(fr) < 2:
        out["motivo"] = "menos de 2 fronteras de zona con razon >= %.1f" % FRONT_RAZON_MIN
        return out
    cols = [float(d["col"]) for d in fr]
    n = len(cols)
    """SE BUSCA UN SUBCONJUNTO y no se exige explicar todas las fronteras: una pista tiene mas
    cambios de color que los del reglamento (juntas de plancha, sombras, publicidad pisada). Exigir
    que encajaran todas daba "ninguna encaja" en los 20 clips."""
    mejor, segundo = None, None
    for i in range(n):
        for j in range(i + 1, n):
            dc = cols[j] - cols[i]
            if abs(dc) < SEG_MIN_COL:
                continue
            for a in range(len(PLANTILLA_ZONAS)):
                for b in range(len(PLANTILLA_ZONAS)):
                    if a == b:
                        continue
                    k = (PLANTILLA_ZONAS[b] - PLANTILLA_ZONAS[a]) / dc
                    if not (0.002 <= k <= 0.060):
                        """SOLO `k > 0`, o sea la x crece con la columna, que es el convenio del eje
                        declarado en `posicion.apoyos`: la x crece hacia la derecha de la imagen. La
                        plantilla es simetrica, asi que restringir el signo no pierde ninguna
                        hipotesis: la volteada es la misma con las etiquetas cambiadas. Y quita una
                        ambiguedad real: `UglyNiceAplomadofalcon` salia con k negativa y el patron
                        [-5, -7], o sea con la pista al reves que el resto del sistema."""
                        continue
                    if not (LARGO_VISIBLE_M[0] <= abs(k) * t["ncol"] <= LARGO_VISIBLE_M[1]):
                        continue
                    q = PLANTILLA_ZONAS[a] - k * cols[i]
                    asig = _asigna_inyectiva([k * c + q for c in cols], PLANTILLA_ZONAS,
                                             TOL_FRONT_M)
                    if len(asig) < 2:
                        continue
                    cl = (len(asig), -float(np.sum([v[1] for v in asig.values()])))
                    cand = (cl, k, q, asig)
                    if mejor is None or cl > mejor[0]:
                        segundo = mejor
                        mejor = cand
                    elif segundo is None or cl > segundo[0]:
                        segundo = cand
    if mejor is None:
        out["motivo"] = "ningun par de fronteras encaja la estructura con largo fisico"
        return out
    _cl, k, q, asig = mejor
    idx = sorted(asig.keys())
    ci = [cols[i] for i in idx]
    xs = [asig[i][0] for i in idx]
    A = np.stack([np.array(ci), np.ones(len(ci))], axis=1)
    sol, *_ = np.linalg.lstsq(A, np.array(xs), rcond=None)
    k, q = float(sol[0]), float(sol[1])
    res = np.abs(k * np.array(ci) + q - np.array(xs))
    loo = []
    if len(ci) >= 3:
        for h in range(len(ci)):
            c2 = [ci[i] for i in range(len(ci)) if i != h]
            x2 = [xs[i] for i in range(len(ci)) if i != h]
            A2 = np.stack([np.array(c2), np.ones(len(c2))], axis=1)
            s2, *_ = np.linalg.lstsq(A2, np.array(x2), rcond=None)
            loo.append(abs(s2[0] * ci[h] + s2[1] - xs[h]))
    amb = None
    if segundo is not None and segundo[0][0] >= _cl[0]:
        pa = tuple(sorted(v[0] for v in asig.values()))
        pb = tuple(sorted(v[0] for v in segundo[3].values()))
        if pb != pa and abs(segundo[1] - k) / abs(k) > 0.10:
            amb = {"k": float(segundo[1]), "patron": list(pb)}
    out.update({"ok": True, "m_por_col": k, "q_m": q, "patron": xs, "cols_ident": ci,
                "n_ident": len(ci), "res_max_m": float(res.max()),
                "col_centro": -q / k if abs(k) > 1e-12 else float("nan"),
                "loo_p50_m": float(np.median(loo)) if loo else None,
                "loo_max_m": float(np.max(loo)) if loo else None,
                "largo_visible_m": abs(k) * t["ncol"],
                "x_ini_m": q, "x_fin_m": k * (t["ncol"] - 1.0) + q,
                "ambiguo": amb is not None, "otra": amb,
                "controlada": bool(len(ci) >= 3),
                "via_ancla": "fronteras de zona de color (t.19)"})
    return out


def ancla_color(res) -> dict:
    """QUE HACE: el ancla medida de un clip, a partir de la tira rectificada. Envoltorio de
    `ancla_zonas` que devuelve tambien la escala transversal y el ancho de combate.

    EL FALLO QUE INTENTA ARREGLAR, y la causa raiz estaba en mis propios datos. El usuario: *"tomas
    tu por defecto que empiezan siempre en el centro de la pista, pero no: muchos igual empiezan al
    final de la pista de la derecha... y en los videos pones tu siempre que empiezan en el centro"*.
    Tenia razon y **no era un error de calculo**: el origen salia de `posicion.ancla_origen`, que yo
    mismo documente como *"una convencion declarada, no una medida"* -- el punto medio de los dos
    tiradores tomado como centro -- con deriva utilizable en 2 de 11 clips. El "siempre en el centro"
    ERA el valor por defecto de esa convencion.
    """
    t = tira_rectificada(res)
    if t is None:
        return {"ok": False, "motivo": "sin tira rectificada (bordes no detectados)"}
    ac = ancho_combate(t)
    an = ancla_zonas(t)
    an.setdefault("motivo", "")
    an.update({"m_por_fil": t["m_por_fil"], "ancho_px_mid": t["ancho_px_mid"],
               "plancha": ac.get("plancha"), "ancho_rel": ac.get("ancho_rel"),
               "ancho_combate_m": ac.get("ancho_combate_m")})
    an.update(error_a_lo_largo(t, an.get("m_por_col")))
    return an


def panel_bordes_marcas(res):
    """QUE HACE: el panel con la placa limpia, los rotulos, los dos bordes y las marcas.

    Se saca SIEMPRE, tambien cuando la homografia falla, porque cuando falla es justo cuando hay que
    mirar si el fallo es de los bordes, de las marcas o de la construccion.
    """
    pl = res["_pl"]
    bo, mar, rot = res.get("_bo"), res.get("_mar"), res.get("_rot")
    a = pl.copy()
    H, W = a.shape[:2]
    if rot is not None:
        a[rot > 0] = (0.55 * a[rot > 0] + 0.45 * np.array([255, 0, 255])).astype("uint8")
    if bo is not None and bo.get("ok"):
        for nom, col in (("A", (0, 0, 255)), ("B", (255, 0, 0))):
            for p in bo["pts_" + nom]:
                cv2.circle(a, (int(p[0]), int(p[1])), 2, col, -1)
            L = bo["recta_" + nom]
            if abs(L[1]) > 1e-6:
                cv2.line(a, (0, int(-(L[2]) / L[1])),
                         (W - 1, int(-(L[0] * (W - 1) + L[2]) / L[1])), col, 1)
    usada = res.get("_hg", {}).get("marca_usada") if res.get("_hg") else None
    for li in ((mar.get("lineas") if mar else None) or []):
        c = np.asarray(li["centro"], "float64")
        L = np.asarray(li["L"], "float64")
        d = np.array([-L[1], L[0]])
        d = d / max(np.linalg.norm(d), 1e-9)
        q0 = c - 0.6 * li["largo_px"] * d
        q1 = c + 0.6 * li["largo_px"] * d
        es = usada is not None and li is usada
        cv2.line(a, (int(q0[0]), int(q0[1])), (int(q1[0]), int(q1[1])),
                 (0, 255, 255) if es else (0, 220, 0), 2 if es else 1)
        cv2.putText(a, "%.0f" % li.get("sn", 0), (int(c[0]) - 8, int(c[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    cv2.putText(a, "magenta rotulo | rojo borde A | azul borde B | verde marca "
                   "| amarillo la que da V2", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return a


def pinta_diagnostico(res, salida: pathlib.Path):
    """QUE HACE: el panel con el que se COMPRUEBA A OJO que la homografia esta bien.

    LA COMPROBACION QUE DE VERDAD SIRVE es la de abajo: se dibuja la pista FIE nominal (14 m de
    largo, con la central, las dos de guardia a 2 m, las dos fronteras de color a 5 m y los dos
    limites posteriores a 7 m) reproyectada a la imagen con la homografia. Si la homografia esta
    bien, las rayas dibujadas caen ENCIMA de las rayas pintadas del suelo. Si no, se ve al instante
    y no hay ninguna cifra que lo pueda disimular.

    Ojo con como se lee: el origen longitudinal esta en la marca que se uso para V2, y esa marca no
    tiene por que ser la central. Asi que la reticula puede estar DESPLAZADA en bloque respecto a las
    rayas de verdad y la homografia ser correcta; lo que hay que mirar es si el ESPACIADO y la
    INCLINACION coinciden, no la posicion absoluta.
    """
    a = panel_bordes_marcas(res)
    pl = res["_pl"]
    hg = res["_hg"]
    H, W = pl.shape[:2]
    b = pl.copy()
    w = hg["ancho_m"]
    for xm, col, gr in ((0.0, (0, 255, 255), 2), (-2.0, (255, 255, 0), 1), (2.0, (255, 255, 0), 1),
                        (-5.0, (255, 0, 255), 1), (5.0, (255, 0, 255), 1),
                        (-7.0, (255, 255, 255), 2), (7.0, (255, 255, 255), 2)):
        p0 = a_imagen(hg, np.array([xm, 0.0]))
        p1 = a_imagen(hg, np.array([xm, w]))
        if np.isfinite(p0).all() and np.isfinite(p1).all():
            cv2.line(b, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), col, gr)
    for ym, col in ((0.0, (0, 0, 255)), (w, (255, 0, 0))):
        p0 = a_imagen(hg, np.array([-7.0, ym]))
        p1 = a_imagen(hg, np.array([7.0, ym]))
        if np.isfinite(p0).all() and np.isfinite(p1).all():
            cv2.line(b, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), col, 1)
    cv2.putText(b, "FIE piste reprojected: 0 cyan, +-2 light blue, +-5 magenta, +-7 white",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    def _n(x):
        """Ningun `None` puede llegar a un formato numerico: el lote de 78 clips se corto a mitad por
        esto y perdio 46 filas ya calculadas."""
        return float("nan") if x is None else float(x)

    cv2.putText(b, "h_p50 %.2f m  f/W %.2f  hc %.2f m  ancho ded %.2f m  via %s" %
                (_n(res.get("h_p50")), _n(res.get("f_rel")), _n(res.get("hc_m")),
                 _n(res.get("ancho_deducido_m")), res.get("via", "?")),
                (6, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    tira = np.vstack([a, b])
    esc = 940.0 / tira.shape[1] if tira.shape[1] > 940 else 1.0
    if esc != 1.0:
        tira = cv2.resize(tira, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
    salida.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(salida / ("homog_%s.png" % res["clip"])), tira)


SALIDA = RU.PISTA
DIAGNOSTICO = SALIDA / "diagnostico"
"""Donde van los `homog_*.png` y los `censo_*.png`, que son DIAGNOSTICO INTERNO y no entregable.

El usuario lo dijo asi: *"mira los output de imagen, no tiene sentido, es poco, pones borroso el
marcador o el fondo un poco extranio, no estas haciendo lo que quiero"*. Y tenia razon: esas figuras
llevan la mascara de rotulos en magenta encima del marcador del clip, la region de pista en verde y
los residuos en texto. Sirven para saber POR QUE la homografia sale mal, que es imprescindible para
trabajar, y no sirven para ensenar nada. El entregable son `pista_*.png` y
`video/trazado_*.mp4`."""


def main(clips, ancho_m=ANCHO_NOMINAL, csv_nombre="homografia.csv"):
    """QUE HACE: pasa el paso 2 por una lista de clips, guarda el CSV y las figuras, y da el
    recuento con la puerta declarada."""
    import csv
    filas = []
    for clip in clips:
        try:
            res = analiza(clip, ancho_m=ancho_m)
        except Exception as exc:
            print("%-38s EXCEPCION %s" % (clip, exc))
            filas.append({"clip": clip, "ok": False, "motivo": "excepcion: %s" % exc})
            continue
        if res.get("ok"):
            pinta_diagnostico(res, DIAGNOSTICO)

            def _n(x, d=float("nan")):
                """Ningun `None` puede llegar a un formato numerico. El lote de 78 clips se corto a
                mitad por esto: `control_estatura` devuelve `None` cuando no hay ninguna caja
                utilizable, y el `%.2f` del informe abortaba la tirada entera y perdia las 46 filas
                que faltaban."""
                return d if x is None else x

            print("%-38s %-20s f/W %.2f hc %5.1f  marcas %d/%d vert %d  "
                  "ESTATURA %.2f [%.2f-%.2f] %s  C2 %s m  ancho %.2f m  zona %s"
                  % (clip, res["via"], _n(res.get("f_rel")), _n(res.get("hc_m")),
                     res.get("n_marcas_afinadas", 0), res.get("n_marcas", 0),
                     res.get("n_vert", 0),
                     _n(res.get("h_p50")), _n(res.get("h_p10")), _n(res.get("h_p90")),
                     "PASA" if res.get("C1_pasa") else "NO",
                     ("%.2f" % res["C2_err_p50"]) if res.get("C2_err_p50") is not None else "-",
                     _n(res.get("ancho_deducido_m")), "SI" if res.get("zona_color") else "no"))
        else:
            print("%-38s NO: %s" % (clip, res.get("motivo")))
        filas.append({k: v for k, v in res.items() if not k.startswith("_")})
        res.clear()
        gc.collect()
        """SE LIBERA A MANO, y hace falta: `analiza` devuelve los fotogramas del clip dentro del
        resultado para que la verificacion visual no tenga que volver a decodificar el video, y son
        2,5 GB en un clip de 1080p. Sin este `clear` el recolector no llegaba a tiempo entre clips."""
    if not filas:
        return filas
    SALIDA.mkdir(parents=True, exist_ok=True)
    campos = sorted({k for f in filas for k in f})
    with open(SALIDA / csv_nombre, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        for f in filas:
            w.writerow(f)
    _resumen(filas)
    return filas


def _resumen(filas):
    """QUE HACE: el recuento con la puerta declarada, separado por resolucion.

    Se separa por resolucion porque la deteccion de rayas depende del numero de pixeles que la raya
    ocupa, y el corpus es 81 % de 1280x720 y 18 % de 1920x1080. Mezclarlas esconderia la unica
    palanca gratis que hay: grabar a mas resolucion.
    """
    ok = [f for f in filas if f.get("ok")]
    pub = [f for f in ok if f.get("publicable")]
    print("\nCLIPS %d | con homografia %d (%.0f %%) | PUBLICABLES %d (%.0f %%)"
          % (len(filas), len(ok), 100.0 * len(ok) / len(filas), len(pub),
             100.0 * len(pub) / len(filas)))
    for W in sorted({f.get("W") for f in filas if f.get("W")}):
        sub = [f for f in filas if f.get("W") == W]
        subp = [f for f in sub if f.get("publicable")]
        print("  %5d px de ancho: %3d clips, %3d publicables (%.0f %%)"
              % (W, len(sub), len(subp), 100.0 * len(subp) / max(len(sub), 1)))
    if pub:
        for nom, cl in (("estatura reproyectada p50 (m)", "h_p50"),
                        ("ancho deducido (m)", "ancho_deducido_m"),
                        ("error del control C2 (m)", "C2_err_p50"),
                        ("largo visible (m)", "largo_visible_m"),
                        ("f / ancho de imagen", "f_rel"),
                        ("altura de camara (m)", "hc_m"),
                        ("marcas transversales medidas", "n_marcas_afinadas"),
                        ("verticales usadas", "n_vert"),
                        ("desplazamiento de camara (px)", "despl_camara_px")):
            v = np.array([float(f[cl]) for f in pub
                          if f.get(cl) is not None and np.isfinite(float(f[cl]))], "float64")
            if v.size:
                print("  %-32s p10 %7.2f  p50 %7.2f  p90 %7.2f" %
                      (nom, np.percentile(v, 10), np.median(v), np.percentile(v, 90)))
    mot = {}
    for f in filas:
        if not f.get("publicable"):
            hh = f.get("h_p50")
            m = f.get("motivo") or ("C1 falla, estatura %.2f m"
                                    % (float("nan") if hh is None else float(hh)))
            mot[m.split(".")[0][:74]] = mot.get(m.split(".")[0][:74], 0) + 1
    if mot:
        print("  por que NO se publica:")
        for m, n in sorted(mot.items(), key=lambda z: -z[1]):
            print("    %3d  %s" % (n, m))


if __name__ == "__main__":
    lista = sys.argv[1:]
    if lista:
        main(lista)

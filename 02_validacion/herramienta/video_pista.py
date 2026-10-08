"""EL VIDEO DEL TRAZADO: el clip a la izquierda y el mapa de la pista dibujandose a la derecha.

=======================================================================================
LO QUE PIDIO EL USUARIO, literal
=======================================================================================

  *"Lo de la pista no veo ninguna grafica que mida como se desplaza a lo largo de la longitud de la
  pista y el ancho, como una linea, sabes, estas graficas que es una linea y te pone como un trazado
  de un coche de carreras por la pista. Pues quiero algo igual pero por la pista de esgrima, de los
  dos tiradores, donde empiezan y como se va completando EN VIDEO para saber en cada momento donde
  estan."*

Y antes habia dicho que los `homog_*.png` no le servian: *"mira los output de imagen, no tiene
sentido, es poco, pones borroso el marcador o el fondo un poco extranio"*. Tenia razon: esos PNG son
DIAGNOSTICO de la homografia (mascaras magenta sobre el rotulo, residuos en texto) y han pasado a
`02_validacion/pista/diagnostico/`.

=======================================================================================
CUATRO ZONAS, y cada una responde a una parte de la peticion
=======================================================================================

  IZQUIERDA ARRIBA   el clip original con los dos tiradores marcados, verde el de la izquierda y
                     naranja el de la derecha, que es el codigo de color del proyecto, con su caja,
                     sus dos apoyos y su posicion en metros escrita al lado del pie.
  IZQUIERDA ABAJO    el rotulo: cuadro, segundo, posicion de cada uno, distancia entre los dos, y
                     EL ESTADO DE LA MEDIDA (ver mas abajo, que es la parte que no se puede omitir).
  DERECHA            el TRAZADO, que es la grafica de coche de carreras: posicion a lo largo de la
                     pista en el eje horizontal y TIEMPO en el vertical, con la traza creciendo. Las
                     dos coordenadas de este panel estan medidas: la x por homografia y el tiempo es
                     exacto.
  ABAJO, TODO ANCHO  el MAPA de la pista visto desde arriba, a escala, con las lineas del
                     reglamento, la posicion actual de cada tirador como un punto grueso, la estela
                     detras y los eventos apareciendo con su etiqueta conforme ocurren.

=======================================================================================
LAS TRES COSAS QUE ESTE VIDEO NO PINTA COMO SI FUERAN BUENAS, Y POR QUE
=======================================================================================

Un video bonito con una cifra inventada es peor que no tener video, asi que las tres limitaciones
medidas del paso de la pista salen EN PANTALLA y no en una nota al pie:

  1. LA COORDENADA TRANSVERSAL, en MODO TIRA, son metros medidos entre los dos bordes DETECTADOS
     de la pista, con la plancha lateral excluida, y esta acotada por construccion por el ancho
     reglamentario. Lo que sigue sin pasar su control es el RECORRIDO: 10 de 22 clips lo tienen por
     debajo de los 0,60 m declarados, y el ruido por cuadro es de 0,9 cm, o sea que la excursion
     grande de los otros NO es temblor: es senial lenta o los bordes mal puestos en ese clip.

     Cuando el clip no tiene tira ni ancla se cae al camino anterior, con el CARRIL FIJO, y el
     rotulo del mapa lo dice: los dos casos estan escritos en el panel y no en una nota al pie.

  2. LA POSICION ABSOLUTA EN LA PISTA, en MODO TIRA, sale de un ANCLA MEDIDA: las fronteras de la
     zona de color de los ultimos 2 m (t.19), cuya posicion es +-5 y +-7 m por reglamento. Eso
     arregla el fallo que el usuario vio (*"tomas tu por defecto que empiezan siempre en el centro de
     la pista"*): el origen deja de ser el punto medio de los dos tiradores. Se mide en 13 de 22
     clips, no ambigua en 11, y con redundancia para controlarla en 5.

     LO QUE SIGUE SIN ESTAR MEDIDO ES LA ESCALA a lo largo. Tres vias independientes probadas
     (rayas pintadas, pixel cuadrado, fronteras de zona) y ninguna aguanta su propio control; ver la
     seccion 36. El rotulo del mapa dice "scale NOT verified" y el numero de fronteras con el que se
     anclo.

  3. LA PUERTA FISICA DE LA HOMOGRAFIA. Solo el 13 % de los 78 clips la pasa, y de los 10 que la
     pasan solo **2** tienen la estatura reproyectada de LOS DOS tiradores dentro de 1,60-2,00 m (ver
     seccion 26). El rotulo dice cual de los tres casos es este clip.

=======================================================================================
COMO SE PINTA, y por que no es todo matplotlib
=======================================================================================

El fondo de cada panel (los ejes, las lineas de la pista, las etiquetas) se construye UNA VEZ con
matplotlib y se guarda como imagen; de ahi se saca la transformacion metros -> pixel. Todo lo que se
mueve se pinta con `cv2` encima de esa imagen. Asi el video sale con ejes de matplotlib sin pagar una
figura por fotograma, que a 350 fotogramas por clip seria la diferencia entre dos minutos y media
hora, y el usuario ya ha avisado de que tiene el ordenador saturado.
"""
from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pista as PI
import posicion as PO

SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\pista\video")
W_VID, H_VID = 1280, 720
W_CLIP, H_CLIP = 760, 340
H_ROTULO = 78
H_CUENTA = 102
"""Alto del panel del CONTADOR. Existe porque el usuario pidio poder juzgar el conteo el mismo:
*"que se vea el CONTADOR... que el numero salte a la vez que el evento"*. Sustituye a la banda de
lista de eventos que llevaba antes el mapa, que decia lo mismo peor y quitaba sitio al mapa."""
H_MAPA = 200
W_TRAZ = W_VID - W_CLIP
H_TRAZ = H_CLIP + H_ROTULO + H_CUENTA
VERDE = (80, 152, 26)
NARANJA = (67, 109, 244)
COL = (VERDE, NARANJA)
GRIS = (150, 150, 150)
BLANCO = (245, 245, 245)
ROJO = (39, 48, 215)
FUENTE = cv2.FONT_HERSHEY_SIMPLEX
CARRIL = (0.25, 0.75)
"""Donde se pinta cada tirador A LO ANCHO del mapa, en fraccion del ancho de la pista, SOLO EN EL
CAMINO ANTIGUO, o sea cuando el clip no tiene tira rectificada o no tiene ancla medida.

ES DIBUJO Y NO MEDIDA, y va rotulado asi en el propio mapa. En MODO TIRA no se usa: ahi la
transversal son los metros medidos entre los dos bordes detectados, que es lo que el usuario pedia
(*"los pones siempre el de la derecha pegado a la derecha... no estas trackeando de verdad su
posicion real"*)."""
MARCA_EV = {"fondo": (ROJO, "lunge"), "marcha": ((180, 119, 69), "march"),
            "romper": ((219, 191, 145), "retreat"), "brazo": ((131, 42, 118), "arm"),
            "alto": ((40, 40, 40), "halt"), "bote": ((0, 170, 255), "balestra"),
            "accion": ((90, 90, 90), "action")}
"""EL SEGUNDO ELEMENTO ES LA ETIQUETA QUE SE VE, y va en INGLES porque el repo es publico. Ojo con
`alto` -> `halt` y NUNCA `parry`: en esgrima una parada es desviar la hoja del rival, no esta
implementada, y es la confusion que se acaba de arreglar en espaniol."""
ESTELA_VIVA = 45
"""Cuadros de estela "viva", la que se pinta gruesa y opaca. 45 cuadros son 1,5 s a 30 fps, que es lo
que dura una accion de esgrima completa."""


VIA_EN = {"camara": "camera geometry", "zona": "colour zone", "rayas": "painted lines"}
"""Traduccion SOLO PARA MOSTRAR de la columna `via`. El valor del CSV no se toca: cambiarlo
romperia el codigo de otros agentes y los scripts de subida."""


def _fig_a_imagen(fig):
    """QUE HACE: pasa una figura de matplotlib a imagen BGR de `cv2`, sin escribir a disco."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(buf[:, :, :3].copy(), cv2.COLOR_RGB2BGR)


def _transforma(ax, fig):
    """QUE HACE: devuelve la funcion que pasa (x, y) en unidades de datos a (columna, fila) de la
    imagen de la figura. Es lo que permite pintar con `cv2` encima de unos ejes de matplotlib."""
    alto = fig.canvas.get_width_height()[1]

    def f(x, y):
        p = ax.transData.transform(np.column_stack([np.atleast_1d(x), np.atleast_1d(y)]))
        return np.column_stack([p[:, 0], alto - p[:, 1]])
    return f


def fondo_mapa(hg, x0, x1, ancho, anc, desv=0.12, tira=None, carriles_fijos=True):
    """QUE HACE: el mapa de la pista visto desde arriba, a escala, con las lineas del reglamento.

    LAS LINEAS SON LAS DE LA EDICION QUE APLICA A ESTE CORPUS, verificada en fuente y no de memoria
    (seccion 23.1). `fie_2021.txt`, t.18.1: *"The combat area of the piste is 1.50 meters wide"*, y
    t.18.3 anade que los bordes conductores de seguridad NO forman parte de la pista. Largo 14 m,
    lineas de puesta en guardia a +-2 m del centro y limites posteriores a +-7 m.

    Y LA MARCA DE LOS 5 m NO SE PINTA COMO RAYA. El reglamento **no define ninguna linea de aviso**:
    t.19 dice que los ultimos 2 m se distingan *"if possible by a different colour of piste"*, o sea
    que es una FRONTERA DE COLOR y no una raya pintada. Se pinta como cambio de color, que es lo que
    es. La version anterior de la grafica la rotulaba "aviso" como si fuera una linea, y eso
    contradecia la propia seccion 23.1 del informe.
    """
    fig = plt.figure(figsize=(W_VID / 100.0, H_MAPA / 100.0), dpi=100)
    ax = fig.add_axes([0.042, 0.235, 0.945, 0.60])
    ax.add_patch(plt.Rectangle((-7, 0), 14, ancho, fc="#efeee6", ec="#444444", lw=1.4, zorder=1))
    for xa, xb in ((-7.0, -5.0), (5.0, 7.0)):
        ax.add_patch(plt.Rectangle((xa, 0), xb - xa, ancho, fc="#ddd2b4", ec="none", zorder=1.1))
    ax.plot([0, 0], [0, ancho], color="#333333", lw=1.5, ls=(0, (5, 3)), zorder=2)
    for x in (-2.0, 2.0):
        ax.plot([x, x], [0, ancho], color="#222222", lw=1.4, zorder=2)
    for x in (-7.0, 7.0):
        ax.plot([x, x], [0, ancho], color="#000000", lw=2.0, zorder=2)
    if x0 > -7.0:
        ax.add_patch(plt.Rectangle((-7, 0), x0 + 7, ancho, fc="#8d8d8d", alpha=0.45, zorder=3))
    if x1 < 7.0:
        ax.add_patch(plt.Rectangle((x1, 0), 7 - x1, ancho, fc="#8d8d8d", alpha=0.45, zorder=3))
    ax.set_xlim(-7.35, 7.35)
    """EL EJE VERTICAL SE ENSANCHA CON LA EXCURSION MEDIDA, y que la traza se salga de la PISTA se
    deja a la vista a proposito: si la transversal reconstruida se va fuera de los 1,50 m de ancho
    reglamentarios, eso es informacion sobre la escala y recortarlo seria esconder el defecto. Con el
    +-0,12 m fijo que llevaba antes, una excursion de 2 m se salia del panel y no se veia nada."""
    mg = max(0.12, 1.15 * float(desv))
    ax.set_ylim(-mg, ancho + mg)
    ax.set_aspect("auto")
    ax.set_yticks([])
    ax.set_xticks(np.arange(-7, 7.1, 1.0))
    ax.tick_params(labelsize=7, length=2, pad=1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    for x, t in ((0.0, "centre"), (-2.0, "on-guard"), (2.0, "on-guard"),
                 (-7.0, "limit"), (7.0, "limit")):
        ax.text(x, ancho + 0.02, t, ha="center", va="bottom", fontsize=6.2, color="#444444")
    for x in (-6.0, 6.0):
        ax.text(x, ancho + 0.02, "last 2 m, t.19: colour", ha="center", va="bottom",
                fontsize=5.4, color="#7a6a3a")
    """LOS CARRILES VAN ROTULADOS **SOLO CUANDO HAY CARRILES**, y esto es una correccion.

    El rotulo se puso porque el usuario dijo que el mapa le salia espejado (*"el verde lo pones que
    esta mas pegado a la derecha y esta a la izquierda"*). Medido, NO lo estaba, y lo que faltaba
    era decir cual carril es cual. Correcto entonces -- y falso ahora en la mitad de los clips.

    LO QUE PASO. Cuando el clip tiene tira rectificada (`modo_tira`), la y del mapa **ya no es el
    carril fijo**: es `Yt[k, lado]`, la transversal MEDIDA. O sea que los dos tiradores se pintan
    donde la medida dice, y su orden vertical es un resultado, no un convenio. Los rotulos fijos
    seguian afirmando "arriba el naranja, abajo el verde", y en el fotograma que lo enseno el verde
    estaba **arriba** y el naranja abajo: el pie decia lo contrario que el dibujo.

    Se encontro mirando un cuadro del MP4, no leyendo el codigo, que es como se han encontrado los
    tres defectos de figura de este proyecto.

    Asi que con carriles fijos se rotulan los carriles, y con transversal medida se rotula **lo que
    de verdad hay**: un eje de metros medidos, con los tiradores identificados por color -- que es
    lo que el panel del trazado ya hace."""
    if carriles_fijos:
        for lado, nom, fy in ((1, "RIGHT (orange)", 0.955), (0, "LEFT (green)", 0.045)):
            ax.text(0.003, fy, nom, transform=ax.transAxes, ha="left",
                    va="top" if fy > 0.5 else "bottom", fontsize=6.2,
                    color="#1a9850" if lado == 0 else "#c2521f", zorder=6,
                    bbox=dict(fc="#ffffff", ec="none", alpha=0.8, pad=0.8))
    else:
        ax.text(0.003, 0.955, "across = MEASURED, not a lane:  green = left fencer, "
                              "orange = right fencer", transform=ax.transAxes, ha="left",
                va="top", fontsize=6.2, color="#444444", zorder=6,
                bbox=dict(fc="#ffffff", ec="none", alpha=0.8, pad=0.8))
    ax.annotate("", xy=(0.60, 1.045), xytext=(0.40, 1.045), xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#444444"), annotation_clip=False)
    ax.text(0.50, 1.075, "x increases towards image right", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=5.6, color="#444444")
    fiab = "anchor RELIABLE" if anc.get("fiable") else "ANCHOR NOT RELIABLE"
    if tira is not None:
        """EL ROTULO DICE DE DONDE SALE CADA COORDENADA, porque las dos no valen lo mismo: la de a lo
        ancho son metros acotados por el ancho reglamentario entre los dos bordes DETECTADOS, y la de
        a lo largo son metros desde un origen MEDIDO pero con una escala que no ha pasado su control
        (seccion 36). Escribir solo "medido" seria tan falso como escribir solo "convencion"."""
        txt = ("PISTE MAP %.2f x 14 m  ·  grey = piste outside the frame  ·  ACROSS: "
               "MEASURED metres between the detected edges  ·  ALONG: origin MEASURED from %d "
               "t.19 colour frontiers%s, scale NOT verified"
               % (ancho, tira.get("ancla_n_ident") or 0,
                  (" (%.0f cm error)" % (100.0 * tira["ancla_loo_m"]))
                  if tira.get("ancla_loo_m") is not None else " (no control)"))
    else:
        txt = ("PISTE MAP, %.2f m wide (t.18.1 of 2021) x 14 m  ·  grey = outside the "
               "camera frame  ·  across the piste we draw the MEASURED SHAPE of the "
               "lateral coordinate, re-centred on its lane: the lateral SCALE is uncertain  ·  "
               "origin by convention: %s, drift %.2f m"
               % (ancho, fiab, anc.get("deriva_m", float("nan"))))
    ax.set_xlabel(txt, fontsize=6.6, labelpad=2)
    im = _fig_a_imagen(fig)
    tr = _transforma(ax, fig)
    plt.close(fig)
    return im, tr


def fondo_trazado(x0, x1, dur, ancho):
    """QUE HACE: el panel del TRAZADO, que es la grafica de coche de carreras que pidio el usuario.

    POSICION EN EL EJE HORIZONTAL Y TIEMPO EN EL VERTICAL, creciendo hacia abajo. Es el panel donde
    la peticion se cumple SIN inventar nada: las dos coordenadas estan medidas, la x por la
    homografia y el tiempo por el numero de cuadro. La alternativa que pedia el usuario literalmente
    -- x contra y sobre el plano de la pista -- necesitaria la coordenada transversal, y esa no es
    creible, asi que el trazado 2D de verdad se pinta en el MAPA con la transversal sustituida por un
    carril y la incertidumbre marcada.
    """
    fig = plt.figure(figsize=(W_TRAZ / 100.0, H_TRAZ / 100.0), dpi=100)
    ax = fig.add_axes([0.155, 0.055, 0.815, 0.875])
    ax.set_xlim(x0, x1)
    ax.set_ylim(dur, 0.0)
    for x, ls, c in ((0.0, (0, (5, 3)), "#555555"), (-2.0, "-", "#999999"), (2.0, "-", "#999999"),
                     (-7.0, "-", "#333333"), (7.0, "-", "#333333")):
        if x0 <= x <= x1:
            ax.axvline(x, color=c, lw=0.8, ls=ls, zorder=0)
    ax.grid(True, color="#e3e3e3", lw=0.5, zorder=0)
    ax.set_xlabel("position along the piste (m)", fontsize=7.5, labelpad=2)
    ax.set_ylabel("time (s)", fontsize=7.5, labelpad=2)
    ax.tick_params(labelsize=7, length=2, pad=1)
    ax.set_title("THE TRACE  ·  green = left fencer, orange = right fencer\n"
                 "both coordinates in this panel are MEASURED", fontsize=7.4, loc="left",
                 pad=4)
    im = _fig_a_imagen(fig)
    tr = _transforma(ax, fig)
    plt.close(fig)
    return im, tr


def _mediana_movil(v, n=9):
    """QUE HACE: mediana movil de ventana `n`, el filtro ya declarado en la seccion 28 para la
    coordenada transversal.

    LO QUE EL FILTRO CAMBIA ESTA MEDIDO, que es la unica forma de saber si filtrar esconde algo: el
    ruido por cuadro es 0,9 cm de mediana en el corpus y el recorrido transversal por clip pasa de
    0,57 a 0,55 m. O sea que la excursion grande de algunos clips NO es temblor: es senial lenta, y
    filtrar no la quita.
    """
    x = np.asarray(v, "float64")
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[max(0, i - n // 2):i + n // 2 + 1]
        w = w[np.isfinite(w)]
        if w.size:
            out[i] = np.median(w)
    return out


def _px(tr, xs, ys):
    """QUE HACE: los pixeles enteros de una lista de puntos en unidades de datos."""
    p = tr(np.asarray(xs, "float64"), np.asarray(ys, "float64"))
    return np.round(p).astype(np.int32)


def _polilinea(im, pts, col, gr, desde=0):
    """QUE HACE: pinta una polilinea partida en los huecos, que es lo que hay que hacer con una
    trayectoria con cuadros perdidos: unirlos con una recta inventaria un movimiento que no se ha
    medido."""
    tramo = []
    for i in range(desde, len(pts)):
        p = pts[i]
        if p is None:
            if len(tramo) > 1:
                cv2.polylines(im, [np.asarray(tramo, np.int32)], False, col, gr, cv2.LINE_AA)
            tramo = []
            continue
        tramo.append(p)
    if len(tramo) > 1:
        cv2.polylines(im, [np.asarray(tramo, np.int32)], False, col, gr, cv2.LINE_AA)


def _texto(im, x, y, txt, col=BLANCO, esc=0.42, gr=1, contorno=None):
    """QUE HACE: texto con contorno legible sobre cualquier fondo.

    EL CONTORNO SE HACE DESPLAZANDO EL MISMO GROSOR, no engordandolo, y esto no es un detalle de
    estilo: `cv2.putText` con las fuentes Hershey cambia el ANCHO DE AVANCE de cada glifo con el
    grosor, asi que dibujar el contorno con `gr + 2` y el relleno con `gr` en la misma posicion pinta
    dos textos de anchos distintos superpuestos. Se veia como si las letras estuvieran duplicadas y
    borrosas -- que es literalmente la queja que el usuario tenia de las salidas anteriores -- y solo
    se localizo recortando y ampliando un fotograma del MP4.
    """
    c = (18, 18, 18) if contorno is None else contorno
    (an, _al), _b = cv2.getTextSize(txt, FUENTE, esc, gr)
    x = max(2, min(int(x), im.shape[1] - an - 3))
    """EL ROTULO SE METE DENTRO DEL PANEL EN VEZ DE SALIRSE POR EL BORDE.

    Los rotulos de los tiradores se anclan en `x0`, el borde izquierdo de la caja, asi que el de un
    tirador pegado a la derecha del cuadro se sale y `cv2.putText` lo **recorta en silencio**: en
    el fotograma que lo enseno se leia `right -1.99 m (stri`. No se descubrio leyendo el codigo --
    no hay nada que leer, putText no avisa -- sino recortando y ampliando un cuadro del MP4, que es
    la misma forma en que se encontro lo del contorno de arriba."""
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.putText(im, txt, (x + dx, y + dy), FUENTE, esc, c, gr, cv2.LINE_AA)
    cv2.putText(im, txt, (x, y), FUENTE, esc, col, gr, cv2.LINE_AA)


def estado_medida(res, anc) -> tuple:
    """QUE HACE: en una linea, que se puede creer de este clip. Devuelve (texto, color).

    TRES CASOS Y NO DOS, y el del medio es el que la puerta actual no separaba (ver seccion 26):

      la homografia NO pasa la puerta       -> los metros son ilustrativos
      pasa la puerta AGRUPADA pero no la de LOS DOS tiradores -> la escala puede estar mal y hay
                                               controles fisicos independientes que lo dicen
      pasa con los dos tiradores en rango   -> es lo mas creible que hay en este corpus, y son 2 de
                                               los 78 clips
    """
    h0, h1 = res.get("h0_p50"), res.get("h1_p50")
    def rg(v):
        return v is not None and np.isfinite(v) and 1.60 <= float(v) <= 2.00
    if not res.get("publicable"):
        return ("HOMOGRAPHY FAILS THE PHYSICAL GATE: the metres in this clip are ILLUSTRATIVE ONLY",
                ROJO)
    if rg(h0) and rg(h1):
        return ("passes the gate with BOTH fencers in 1.60-2.00 m (2 of the 78 clips)",
                (80, 152, 26))
    falta = "only one fencer measured" if (h0 is None or h1 is None or
                                         not np.isfinite(h0 or np.nan) or
                                         not np.isfinite(h1 or np.nan)) else \
        "one of the two outside 1.60-2.00 m"
    """EN INGLES COMO EL RESTO DE LA LINEA. Esta cadena se quedo en espaniol y salia incrustada en
    medio de un rotulo ingles: `passes the POOLED gate but uno de los dos fuera de 1,60-2,00 m`. Se
    ve en el video publico y en la columna `puerta` de `videos.csv`, asi que hay que regenerar los
    dos. Y en ASCII: `1.60-2.00` con punto, porque la coma decimal en un rotulo ingles se lee como
    separador de miles."""
    return ("passes the POOLED gate but %s (left %.2f / right %.2f m): the SCALE may be wrong"
            % (falta, h0 if h0 is not None else float("nan"),
               h1 if h1 is not None else float("nan")), (0, 150, 235))


LENTA = 4
"""Factor de camara lenta. `una(..., lenta=True)` escribe el MP4 a `fps / LENTA`, o sea a un cuarto
de velocidad, que es lo que el usuario pidio para poder juzgar el RETARDO:

  *"estaria bien verlo igual, alguno de esos en camara lenta, para ver como se va detectando
  perfectamente, si hay delay, y se ve bien."*

Se baja la cadencia del fichero y NO se repiten cuadros: el fichero pesa lo mismo y el reproductor
ensena cada cuadro cuatro veces mas tiempo, que es exactamente lo que hace falta para ver en que
cuadro salta el contador."""


def una(clip, max_cuadros=None, fps_salida=None, lenta=False, sufijo=""):
    """QUE HACE: el MP4 de un clip completo."""
    res = PI.analiza(clip)
    if not res.get("ok"):
        print("%-38s SIN HOMOGRAFIA: %s" % (clip, res.get("motivo")))
        return None
    r, hg, Mk, frames = res["_r"], res["_hg"], res["_Mk"], res["_frames"]
    W, H, T = r["W"], r["H"], r["T"]
    fps = PO._fps(PI.indice_videos()[clip])
    poses = PO.carga_poses(clip)
    ap = PO.apoyos(res, poses)
    ev = PO.eventos(ap, r, fps, poses, res=res)
    """EL INDICE DE CADA EVENTO SE NUMERA CON LA MISMA CLAVE Y EL MISMO ORDEN QUE `posicion.serie`,
    para que el numero que sale en pantalla sea EL MISMO que el `indice_del_tipo` del CSV. Si no
    coinciden, el video y el CSV dicen cosas distintas del mismo evento y el entregable no vale."""
    cuenta = {}
    for e in sorted(ev, key=lambda z: (z.get("lado", -1), z["tipo"], z["inicio"])):
        cl = (e.get("lado", -1), e["tipo"])
        cuenta[cl] = cuenta.get(cl, 0) + 1
        e["indice"] = cuenta[cl]
    anc = PO.ancla_origen(ap, r["mask"])
    x00 = anc["x0_m"]
    ancho = float(hg["ancho_m"])
    C = ap["C"]
    """MODO TIRA: la posicion del mapa sale de `posicion.posicion_entre_bordes`, o sea de la TIRA
    RECTIFICADA con los dos bordes detectados, y el origen del ANCLA MEDIDA por las fronteras de zona
    de color (t.19). Es lo que arregla las dos cosas que el usuario vio en el mismo video:

      *"tomas tu por defecto que empiezan siempre en el centro de la pista, pero no: muchos igual
      empiezan al final de la pista de la derecha"* -> el origen deja de ser una convencion
      *"los pones siempre el de la derecha pegado a la derecha"* -> la transversal deja de ser un
      carril fijo y pasa a ser los metros medidos entre los bordes de la zona de combate

    Si no hay ancla medida en el clip, se cae al camino anterior y el rotulo lo dice. Lo que NO se
    toca son las MAGNITUDES de los eventos, que siguen viniendo de la homografia de la seccion 26:
    si se reescalaran aqui, el video y el CSV de la serie dirian cosas distintas del mismo evento."""
    pbt = PO.posicion_entre_bordes(res, poses)
    modo_tira = bool(pbt.get("ok") and pbt.get("metros_a_lo_largo"))
    Xt = Yt = None
    rui_t = 0.02
    if modo_tira:
        Xt = pbt["X"]
        Yt = np.stack([_mediana_movil(pbt["Y"][:, l]) for l in (0, 1)], axis=1)
        dd = np.concatenate([np.diff(pbt["Y"][:, l][np.isfinite(pbt["Y"][:, l])])
                             for l in (0, 1)])
        rui_t = (1.4826 * float(np.median(np.abs(dd - np.median(dd)))) / np.sqrt(2.0)
                 if dd.size > 8 else 0.02)
        """EL RUIDO DE LA BANDA SE MIDE EN EL CLIP y no se pone a mano: es la desviacion robusta de la
        diferencia entre cuadros consecutivos, dividida por raiz de 2 porque una diferencia de dos
        medidas independientes tiene el doble de varianza que una."""
        """LA TRANSVERSAL SE FILTRA CON LA MISMA MEDIANA MOVIL DECLARADA EN LA SECCION 28, y se mide
        lo que el filtro cambia: el ruido por cuadro es de 0,9 cm (mediana del corpus) y el recorrido
        pasa de 0,57 a 0,55 m, o sea que la excursion NO es temblor: es senial lenta."""
    ks = np.nonzero(r["mask"])[0]
    k0, k1 = int(ks[0]), int(min(ks[-1], len(frames) - 1, T - 1))
    if max_cuadros:
        k1 = min(k1, k0 + max_cuadros - 1)
    dur = max((k1 - k0) / fps, 0.5)
    xr = (Xt if modo_tira else C[:, :, 0] - x00)[k0:k1 + 1]
    xr = xr[np.isfinite(xr)]
    if xr.size < 10:
        print("%-38s sin posiciones utilizables" % clip)
        return None
    xlo, xhi = float(np.percentile(xr, 0.5)), float(np.percentile(xr, 99.5))
    mar = max(0.5, 0.08 * (xhi - xlo))
    if modo_tira:
        """EN MODO TIRA EL GRIS DEL MAPA ES EL TRAMO DE PISTA QUE NO ENTRA EN EL ENCUADRE, medido: los
        extremos de la tira son las columnas donde las rectas de los bordes se salen del cuadro, y con
        el ancla eso son metros. Antes el gris salia de los percentiles de la propia trayectoria, que
        no es lo mismo: un tirador que no llega a un extremo no hace que ese extremo no se vea."""
        xlo, xhi = pbt["x_visible_m"]
        mar = 0.0
    trv0 = {l: PO.transversal(ap, r["mask"], l) for l in (0, 1)}
    desv = 0.12
    for l in (0, 1):
        yv = (Yt[:, l] if modo_tira else trv0[l]["y"])
        if (modo_tira or trv0[l]["forma_publicable"]) and np.isfinite(yv).any():
            d = np.nanmax(np.abs(yv - (0.5 * ancho if modo_tira else np.nanmedian(yv))))
            desv = max(desv, float(d) if np.isfinite(d) else 0.12)
    im_mapa, tr_mapa = fondo_mapa(hg, xlo - mar, xhi + mar, ancho, anc, desv=desv,
                                  tira=pbt if modo_tira else None,
                                  carriles_fijos=not modo_tira)
    im_traz, tr_traz = fondo_trazado(xlo - mar, xhi + mar, dur, ancho)

    """SE PRECALCULAN LOS PIXELES DE TODA LA TRAZA UNA VEZ. Por fotograma solo se pinta la parte ya
    ocurrida, que es una llamada a `polylines` sobre un recorte de la lista."""
    pm, pt, pb = {}, {}, {}
    trv = trv0
    for lado in (0, 1):
        """LA TRAZA DEL MAPA VA EN LA TRANSVERSAL FILTRADA DE VERDAD, no en un carril fijo. Ver
        `posicion.transversal` para la medida que lo justifica: la excursion es 9 a 77 veces el ruido
        por cuadro, asi que la forma es senial en 22 de 22 tiradores-clip. Cuando la forma NO pase el
        criterio, se cae al carril fijo y el rotulo lo dice; en estos once clips no pasa nunca."""
        tv = trv[lado]
        yc = CARRIL[lado] * ancho
        ymed = float(np.nanmedian(tv["y"])) if np.isfinite(tv["y"]).any() else yc
        pm[lado] = []
        pt[lado] = []
        pb[lado] = []
        for k in range(k0, k1 + 1):
            x = Xt[k, lado] + x00 if modo_tira else C[k, lado, 0]
            if not np.isfinite(x):
                pm[lado].append(None)
                pt[lado].append(None)
                pb[lado].append(None)
                continue
            if modo_tira:
                """LOS METROS A LO ANCHO SON LOS MEDIDOS, sin recentrar en ningun carril: 0 m en un
                borde de la zona de combate y 1,50 m en el otro (t.18.1), con la plancha lateral
                excluida (t.18.3). Si un apoyo cae fuera de la pista se pinta fuera, porque eso es
                informacion y recortarlo la esconderia."""
                yk = Yt[k, lado] if np.isfinite(Yt[k, lado]) else 0.5 * ancho
            elif tv["forma_publicable"] and np.isfinite(tv["y"][k]):
                """SE RECENTRA EN EL CARRIL Y NO SE REESCALA: se suma la DESVIACION de la transversal
                respecto a su propia mediana al carril del tirador. Asi la FORMA es la medida y el
                sitio donde se dibuja es el convenio, que es lo unico que se puede afirmar: el
                recorrido sale hasta 1,87 m en una pista de 1,50, o sea que la escala esta inflada y
                la posicion absoluta a lo ancho no se sabe."""
                yk = yc + (tv["y"][k] - ymed)
            else:
                yk = yc
            pm[lado].append(_px(tr_mapa, [x - x00], [yk])[0])
            rui = (rui_t if modo_tira else
                   (tv["ruido_m"] if np.isfinite(tv["ruido_m"]) else 0.04))
            pb[lado].append(_px(tr_mapa, [x - x00], [yk + 3.0 * rui])[0])
            pt[lado].append(_px(tr_traz, [x - x00], [(k - k0) / fps])[0])

    if modo_tira:
        """LAS MARCAS DE EVENTO SE COLOCAN EN EL EJE NUEVO, en la posicion de la tira del cuadro en
        que empieza el evento. Se cambia el SITIO donde se pinta la marca, no la magnitud del evento:
        la magnitud sigue siendo la del CSV. Colocarla con la x vieja sobre un mapa con el eje nuevo
        pondria la marca en un sitio que no es donde paso."""
        for e in ev:
            l = int(e.get("lado", -1))
            kk = int(min(max(e.get("inicio", e.get("cuadro", k0)), 0), Xt.shape[0] - 1))
            """SE PONE SIEMPRE, AUNQUE SALGA `nan`, y no se deja que caiga al valor viejo. Con el
            respaldo puesto pasaba esto, y se vio en un fotograma: dos eventos de brazo cuyo cuadro
            de inicio no tenia posicion en la tira se pintaron con la x de la homografia vieja, o sea
            en x = 0,1 m cuando el tirador estaba en +5,2 m. Un evento mal colocado en el mapa es
            peor que un evento no pintado, porque el no pintado no afirma nada."""
            e["pos_mapa"] = float(Xt[kk, l]) if l in (0, 1) else float("nan")
            if l in (0, 1):
                e["y_mapa"] = float(Yt[kk, l]) if np.isfinite(Yt[kk, l]) else 0.5 * ancho
    txt_estado, col_estado = estado_medida(res, anc)
    """EL ROTULO DE LA TRANSVERSAL dice las DOS cosas que hay que decir y no una: que la FORMA es
    senial (excursion 9 a 77 veces el ruido, 22 de 22 tiradores-clip) y que la ESCALA no es creible
    cuando el recorrido se pasa de los 0,60 m del control. Decir solo "no es creible" llevaba a
    pintarla plana, que es igual de falso."""
    ex = max(trv[l]["excursion_m"] for l in (0, 1))
    esc_ok = all(trv[l]["escala_creible"] for l in (0, 1))
    if modo_tira:
        """EL ROTULO DICE LAS DOS COSAS: que la transversal ya son metros medidos, y que su RECORRIDO
        no pasa el control fisico declarado de 0,60 m. Y la cifra honesta es esta, la de la escala
        buena: con la escala comprimida por el fallo de `ancho_combate` salia 0,55 m y PASABA el
        control, o sea que el defecto se estaba tapando a si mismo. Arreglada la escala, el recorrido
        sale p50 1,03 m sobre una pista de 1,50 y solo 3 de 22 clips pasan."""
        rec = max(pbt["recorrido_y_m"])
        tv = ("lateral: measured m, range %.2f m of %.2f (control 0.60 m: %s)"
              % (rec, ancho, "ok" if rec <= 0.60 else "FAILS"))
    else:
        tv = ("lateral: SHAPE measured (%.0fx the noise), SCALE %s: range %.2f m%s"
              % (min(trv[l]["razon"] for l in (0, 1)),
                 "credible" if esc_ok else "UNCERTAIN", ex,
                 "" if esc_ok else " on a %.2f m wide piste" % ancho))
    SALIDA.mkdir(parents=True, exist_ok=True)
    sal = SALIDA / ("trace_%s%s.mp4" % (clip, sufijo))
    vw = cv2.VideoWriter(str(sal), cv2.VideoWriter_fourcc(*"mp4v"),
                         float(fps_salida or (fps / LENTA if lenta else fps)),
                         (W_VID, H_VID))
    if not vw.isOpened():
        print("%-38s no se pudo abrir el VideoWriter" % clip)
        return None
    esc = min(W_CLIP / float(W), H_CLIP / float(H))
    for k in range(k0, k1 + 1):
        lienzo = np.full((H_VID, W_VID, 3), 28, "uint8")
        cl = (_panel_esqueleto(r, ap, k, esc, poses, X=Xt) if PUBLICO
              else _panel_clip(frames[k], r, ap, k, esc, poses, X=Xt))
        lienzo[0:cl.shape[0], 0:cl.shape[1]] = cl
        mp = im_mapa.copy()
        tz = im_traz.copy()
        i = k - k0
        for lado in (0, 1):
            """LA TRAZA YA RECORRIDA en tono claro y los ultimos `ESTELA_VIVA` cuadros en el color
            pleno: es la estela del coche de carreras que pidio el usuario. En el MAPA solo se pinta
            lo ya ocurrido, igual que en el trazado, porque si se pintase la traza completa desde el
            primer fotograma el video no ensenaria "como se va completando"."""
            _polilinea(mp, pm[lado][:i + 1], tuple(int(0.40 * c + 0.60 * 205)
                                                   for c in COL[lado]), 2)
            _polilinea(tz, pt[lado][:i + 1], tuple(int(0.40 * c + 0.60 * 235)
                                                   for c in COL[lado]), 2)
            _polilinea(mp, pm[lado][max(0, i - ESTELA_VIVA):i + 1], COL[lado], 3)
            _polilinea(tz, pt[lado][max(0, i - ESTELA_VIVA):i + 1], COL[lado], 3)
            ini = next((p for p in pm[lado] if p is not None), None)
            if ini is not None:
                cv2.circle(mp, (int(ini[0]), int(ini[1])), 7, COL[lado], 2, cv2.LINE_AA)
                _texto(mp, int(ini[0]) - 16, int(ini[1]) + 24, "start", COL[lado], 0.36, 1,
                       contorno=(246, 245, 238))
            if pm[lado][i] is not None:
                p = pm[lado][i]
                cv2.circle(mp, (int(p[0]), int(p[1])), 9, COL[lado], -1, cv2.LINE_AA)
                cv2.circle(mp, (int(p[0]), int(p[1])), 9, (255, 255, 255), 2, cv2.LINE_AA)
                q = pt[lado][i]
                cv2.circle(tz, (int(q[0]), int(q[1])), 6, COL[lado], -1, cv2.LINE_AA)
                """LA BANDA DE INCERTIDUMBRE se pinta a +-3 veces el ruido transversal MEDIDO de ese
                tirador, que es la parte de la incertidumbre que se puede cuantificar. La otra parte,
                la de ESCALA, no se puede pintar como banda porque no hay una cifra de error: va
                escrita en el rotulo."""
                dy3 = abs(int(pb[lado][i][1]) - int(p[1]))
                cv2.line(mp, (int(p[0]), int(p[1]) - dy3), (int(p[0]), int(p[1]) + dy3),
                         GRIS, 1, cv2.LINE_AA)
                """LA BANDA DE INCERTIDUMBRE TRANSVERSAL, que es la parte honrada del mapa: la
                posicion se pinta en un carril fijo, pero el recorrido transversal MEDIDO del clip se
                pinta como barra vertical alrededor del punto, para que se vea cuanto de ancho no se
                sabe."""

        _marcas_eventos(mp, tz, ev, k, k0, fps, tr_mapa, tr_traz,
                        0.0 if modo_tira else x00, ancho)
        """EL DESPLAZAMIENTO VA A CERO EN MODO TIRA, y esto era un fallo con sintoma visible: las
        marcas se pintaban en `pos_mapa - x00`, o sea restandoles el ancla de la convencion vieja
        cuando su posicion ya estaba en el eje nuevo. Con `x00 = 4,88 m` en
        `UglyNiceAplomadofalcon`, un evento de brazo medido en +5,18 m aparecia en +0,30 m: un punto
        morado suelto junto a la linea central, con el tirador a cinco metros de ahi."""
        lienzo[H_VID - H_MAPA:H_VID, 0:W_VID] = mp
        lienzo[0:H_TRAZ, W_CLIP:W_VID] = tz
        _rotulo(lienzo, clip, k, k0, fps, C, ap, x00, txt_estado, col_estado, anc, tv,
                Xt=Xt, Yt=Yt, dist=pbt.get("dist_m") if modo_tira else None)
        _contador(lienzo, ev, k, H_CLIP + H_ROTULO)
        _destello(lienzo, ev, k)
        vw.write(lienzo)
    vw.release()
    n = {}
    for e in ev:
        n[e["tipo"]] = n.get(e["tipo"], 0) + 1
    print("%-38s %4d cuadros -> %s  %.1f MB  |  %s  |  anclaje deriva %.2f m %s  |  %s"
          % (clip, k1 - k0 + 1, sal.name, sal.stat().st_size / 1e6,
             " ".join("%s %d" % (t, v) for t, v in sorted(n.items())),
             anc.get("deriva_m", float("nan")),
             "fiable" if anc.get("fiable") else "NO FIABLE", txt_estado[:48]))
    return {"clip": clip, "mp4": str(sal), "res": res, "ap": ap, "ev": ev, "fps": fps,
            "ancla": anc, "n": n, "estado": txt_estado}


PUBLICO = False
"""Si esta activo, el panel del clip se sustituye por el ESQUELETO DE 308 PUNTOS SOBRE NEGRO.

**ESTA APAGADO POR DEFECTO, Y EL CAMBIO ES DELIBERADO.** Existia porque la regla era que a los
repos publicos solo fuera lo GENERADO: los clips son grabacion de television de FencingDatabase
con atletas identificables. Pero el usuario **ya reverso esa regla para los videos de capa** --
`panel_publico.py` lo dice en su primera linea: *"los videos se ven con el fondo negro, no se ve el
clip" -> el clip se ve. Antes se dibujaba sobre negro para no publicar fotogramas de la emision; el
usuario ha decidido publicarlos*. Este fichero se quedo atras, asi que el repo se contradecia a si
mismo: las capas ensenaban el fotograma y el video de la pista no.

Ahora los dos ensenan el clip. `--publico` sigue existiendo y sigue funcionando, porque el
esqueleto sobre negro **tambien ensena mejor lo que se mide** -- los apoyos y la caja dejan de
competir con el fondo del pabellon -- pero ya no es lo que se publica por defecto.

La atribucion a FencingDatabase con enlace va en la ficha, arriba y al pie, como el usuario pidio."""


def _panel_esqueleto(r, ap, k, esc, poses=None, X=None, Y=None):
    """QUE HACE: el panel del clip, pero con el esqueleto de 308 puntos sobre NEGRO en vez del
    fotograma. Mismo tamano y mismos rotulos que `_panel_clip`, para que el resto del video no se
    entere de cual de los dos se ha usado.

    El criterio de pintado es el de `video_revision._pinta`, que es el del proyecto: un punto vale si
    su confianza pasa el umbral Y su coordenada es finita, porque hay puntos con confianza alta y
    coordenada NaN.
    """
    W, H = r["W"], r["H"]
    im = np.zeros((int(H * esc), int(W * esc), 3), "uint8")
    for lado in (0, 1):
        if not r["pres"][k, lado]:
            continue
        if poses is not None and k < poses["KP"].shape[0]:
            kp = poses["KP"][k, lado] * esc
            sc = poses["SC"][k, lado]
            _esqueleto(im, kp, sc, COL[lado])
        b = r["B"][k, lado]
        if not np.isfinite(b).all():
            continue
        x0, y0 = int(b[0] * W * esc), int(b[1] * H * esc)
        x1, y1 = int(b[2] * W * esc), int(b[3] * H * esc)
        cv2.rectangle(im, (x0, y0), (x1, y1), COL[lado], 2)
        if poses is not None and k < poses["KP"].shape[0]:
            for p in PO._pie_pose(poses["KP"], poses["SC"], k, lado):
                if p is not None:
                    cv2.drawMarker(im, (int(p[0] * esc), int(p[1] * esc)), COL[lado],
                                   cv2.MARKER_TILTED_CROSS, 14, 2)
        et = "left" if lado == 0 else "right"
        q, eje = _eje_y_valor(ap, k, lado, X)
        if np.isfinite(q):
            _texto(im, x0, max(14, y0 - 6), "%s  %+.2f m (%s)" % (et, q, eje),
                   COL[lado], 0.46, 1)
        else:
            _texto(im, x0, max(14, y0 - 6), "%s  no position" % et, COL[lado], 0.46, 1)
    _texto(im, 8, im.shape[0] - 8, "308-point skeleton over black: every pixel here is model "
           "output, not broadcast", GRIS, 0.40, 1)
    lienzo = np.full((H_CLIP, W_CLIP, 3), 0, "uint8")
    lienzo[0:im.shape[0], 0:im.shape[1]] = im[0:H_CLIP, 0:W_CLIP]
    return lienzo


CONF_PINTA = 0.30
HUESOS_MIN = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
              (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))
"""Los huesos del cuerpo de los 17 puntos de COCO, que son los que hacen falta para que el esqueleto
se lea. Los 291 puntos de manos y cara se pintan como puntos sueltos: dibujar sus huesos a este
tamano da una mancha."""


def _esqueleto(im, kp, sc, col):
    """QUE HACE: pinta un esqueleto sobre la imagen, con el criterio de `video_revision._pinta`."""
    def val(i):
        return (i < len(sc) and sc[i] >= CONF_PINTA and np.isfinite(kp[i, 0])
                and np.isfinite(kp[i, 1]))

    for a, b in HUESOS_MIN:
        if val(a) and val(b):
            cv2.line(im, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[b, 0]), int(kp[b, 1])),
                     col, 2, cv2.LINE_AA)
    for i in range(len(sc)):
        if val(i):
            cv2.circle(im, (int(kp[i, 0]), int(kp[i, 1])), 3 if i < 21 else 1, col, -1)


def _eje_y_valor(ap, k, lado, X):
    """QUE DEVUELVE: (el metro que se rotula, el nombre del eje del que sale).

    POR QUE ESTO ES UNA FUNCION Y NO DOS LINEAS REPETIDAS. Los dos paneles del clip rotulaban
    **metros de ejes distintos y ninguno lo decia**: `_panel_clip` ponia `ap["C"][k,lado][0]`, que
    es la coordenada de la HOMOGRAFIA, y `_panel_esqueleto` ponia `Xt[k,lado]`, que son los metros
    medidos ENTRE LOS BORDES de la tira rectificada. Los paneles del trazado y del mapa usan el
    segundo. O sea que cambiar de panel cambiaba en silencio la magnitud que el espectador lee, y
    en el panel del clip el numero ni siquiera coincidia con el de al lado.

    Ahora los dos leen el MISMO eje -- la tira cuando existe, la homografia cuando no -- y **el
    rotulo dice cual**, porque son dos medidas distintas de la misma cosa y el proyecto tiene
    medido que no coinciden.
    """
    if X is not None and k < X.shape[0] and np.isfinite(X[k, lado]):
        return float(X[k, lado]), "strip"
    q = ap["C"][k, lado]
    if np.isfinite(q).all():
        return float(q[0]), "homography"
    return float("nan"), ""


def _panel_clip(frame, r, ap, k, esc, poses=None, X=None):
    """QUE HACE: el clip original con los dos tiradores marcados, sus DOS APOYOS y su posicion en
    metros.

    Los apoyos se pintan porque son el punto que de verdad se mide: la posicion en metros sale del
    pie, no del centro de la caja, y quien mire el video tiene que poder juzgar a ojo si la cruz esta
    sobre la planta del pie o se ha ido al tobillo del rival.
    """
    im = cv2.resize(frame, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
    W, H = r["W"], r["H"]
    for lado in (0, 1):
        if not r["pres"][k, lado]:
            continue
        b = r["B"][k, lado]
        if not np.isfinite(b).all():
            continue
        x0, y0 = int(b[0] * W * esc), int(b[1] * H * esc)
        x1, y1 = int(b[2] * W * esc), int(b[3] * H * esc)
        cv2.rectangle(im, (x0, y0), (x1, y1), COL[lado], 2)
        if poses is not None and k < poses["KP"].shape[0]:
            for p in PO._pie_pose(poses["KP"], poses["SC"], k, lado):
                if p is not None:
                    cv2.drawMarker(im, (int(p[0] * esc), int(p[1] * esc)), COL[lado],
                                   cv2.MARKER_TILTED_CROSS, 14, 2)
        et = "left" if lado == 0 else "right"
        q, eje = _eje_y_valor(ap, k, lado, X)
        if np.isfinite(q):
            _texto(im, x0, max(14, y0 - 6), "%s  %+.2f m (%s)" % (et, q, eje),
                   COL[lado], 0.46, 1)
        else:
            _texto(im, x0, max(14, y0 - 6), "%s  no position" % et, COL[lado], 0.46, 1)
    lienzo = np.full((H_CLIP, W_CLIP, 3), 28, "uint8")
    lienzo[0:im.shape[0], 0:im.shape[1]] = im[0:H_CLIP, 0:W_CLIP]
    return lienzo


def _marcas_eventos(mp, tz, ev, k, k0, fps, tr_mapa, tr_traz, x00, ancho):
    """QUE HACE: pinta los eventos que YA han ocurrido, en el mapa y en el trazado.

    Aparecen cuando empiezan y se quedan, que es el "conforme ocurren, van apareciendo" de la
    peticion. El del cuadro actual se pinta mas grande.

    LAS ETIQUETAS NO VAN SOBRE EL MAPA, y esto es una correccion de la primera version: con 40
    eventos en un clip las etiquetas se pisaban unas a otras y encima tapaban el punto de posicion,
    que es justo el dato principal. Se comprobo sacando un fotograma y mirandolo, y se veia
    "accion 1.01 m" escrito encima de "fondo 0.53 m" y del punto verde. Van en una LISTA a la
    izquierda del mapa, con los tres ultimos y su indice, que es el mismo numero del CSV de la serie.
    """
    ult = []
    for e in ev:
        if e["inicio"] > k:
            continue
        lado = int(e.get("lado", -1))
        px = e.get("pos_mapa", e.get("pos_m", [None, None])[0])
        if px is None or not np.isfinite(px):
            continue
        col, nom = MARCA_EV.get(e["tipo"], (GRIS, e["tipo"]))
        if "y_mapa" in e:
            yc = e["y_mapa"] + (0.10 * ancho if lado == 1 else -0.10 * ancho)
        else:
            yc = CARRIL[lado if lado >= 0 else 0] * ancho
            yc = yc + (0.13 * ancho if lado == 1 else -0.13 * ancho)
        p = _px(tr_mapa, [px - x00], [yc])[0]
        q = _px(tr_traz, [px - x00], [(e["cuadro"] - k0) / fps])[0]
        reciente = 0 <= k - e["cuadro"] <= 30
        rad = 7 if reciente else 3
        cv2.circle(mp, (int(p[0]), int(p[1])), rad, col, -1, cv2.LINE_AA)
        if reciente:
            cv2.circle(mp, (int(p[0]), int(p[1])), rad + 2, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(tz, (int(q[0]), int(q[1])), 4 if reciente else 2, col, -1, cv2.LINE_AA)
        ult.append((e["cuadro"], e, col, nom))
    """LAS MARCAS DE MARCHA Y ROMPER LLEVAN SU NUMERO AL LADO EN EL MAPA, que es la otra mitad de lo
    que pidio el usuario: *"que salgan las trazadas sobre la pista de cada tirador y marcadas las
    marchas, romper y eso"*. Solo se numeran esos dos tipos y solo las 6 ultimas, porque numerar los
    40 eventos de un clip vuelve a amontonar las etiquetas, que es el defecto que ya se corrigio una
    vez."""
    ult.sort(key=lambda z: z[0])
    for _c, e, col, nom in ult[-6:]:
        if e["tipo"] not in ("marcha", "romper"):
            continue
        px = e.get("pos_mapa", e.get("pos_m", [None, None])[0])
        if px is None or not np.isfinite(px):
            continue
        lado = int(e.get("lado", -1))
        if "y_mapa" in e:
            yc = e["y_mapa"] + (0.10 * ancho if lado == 1 else -0.10 * ancho)
        else:
            yc = CARRIL[lado if lado >= 0 else 0] * ancho
            yc += 0.13 * ancho if lado == 1 else -0.13 * ancho
        p = _px(tr_mapa, [px - x00], [yc])[0]
        _texto(mp, int(p[0]) + 7, int(p[1]) + 4, "%d" % e.get("indice", 0), col, 0.42, 1,
               contorno=(246, 245, 238))


TIPOS_CUENTA = ("marcha", "romper", "fondo", "alto", "brazo")
"""Tipos que salen en el CONTADOR de pantalla, en este orden. `bote` no entra porque esta declarado
NO ESTABLECIDO (0 eventos) y `accion` no es por tirador."""
DESTELLO = 18
"""Cuadros que dura el cartel grande del evento. 18 cuadros son 0,6 s a 30 fps: bastante para leerlo
y poco para que se solape con el siguiente evento del mismo tirador."""


def _contador(lienzo, ev, k, y0):
    """QUE HACE: el CONTADOR por tirador y por tipo, que es lo que el usuario pidio para poder juzgar
    el conteo el mismo.

    *"Me lo puedes ensenar visualmente en un video, a ver si las va contando bien? Pasa varios
    ejemplos y que se vea el CONTADOR."*

    EL NUMERO QUE SE ENSENA ES EL `indice_del_tipo` DEL CSV, no un contador aparte: se toma como el
    maximo indice de ese (lado, tipo) entre los eventos ya empezados. Asi el 3 que se ve en pantalla
    es la misma fila que `indice_del_tipo == 3`, y se puede ir de una cosa a la otra. Un contador
    propio que contase por su cuenta podria desincronizarse del CSV y entonces el video no verificaria
    el CSV, que es justo para lo que existe.

    Y SALTA EN EL CUADRO DE `inicio` del evento, no en el de pico: si el tirador da un paso y el
    numero no se mueve, se ve al instante, que es la prueba que el usuario quiere hacer.
    """
    cv2.rectangle(lienzo, (0, y0), (W_CLIP - 1, y0 + H_CUENTA - 1), (14, 14, 14), -1)
    for lado, nom in ((0, "LEFT"), (1, "RIGHT")):
        x = 10 + 380 * lado
        _texto(lienzo, x, y0 + 17, nom, COL[lado], 0.50, 1)
        for j, t in enumerate(TIPOS_CUENTA):
            n = max([e.get("indice", 0) for e in ev
                     if e["tipo"] == t and e.get("lado") == lado and e["inicio"] <= k] or [0])
            nuevo = any(e["tipo"] == t and e.get("lado") == lado
                        and 0 <= k - e["inicio"] < DESTELLO for e in ev)
            col = (255, 255, 255) if nuevo else (168, 168, 168)
            cv2.putText(lienzo, "%-8s" % MARCA_EV[t][1], (x + 4, y0 + 34 + 15 * j), FUENTE, 0.42,
                        (128, 128, 128), 1, cv2.LINE_AA)
            _texto(lienzo, x + 74, y0 + 34 + 15 * j, "%2d" % n, col,
                   0.58 if nuevo else 0.48, 2 if nuevo else 1)
            if nuevo:
                cv2.circle(lienzo, (x + 110, y0 + 29 + 15 * j), 4, MARCA_EV[t][0], -1,
                           cv2.LINE_AA)


def _destello(lienzo, ev, k):
    """QUE HACE: el cartel grande del evento recien detectado, con su numero y su magnitud.

    *"Marca el evento en el momento... con el nombre del evento grande en pantalla durante unos
    cuadros, con su magnitud en metros («marcha 3: 0,31 m»)."*
    """
    act = [e for e in ev if e["tipo"] in MARCA_EV and 0 <= k - e["inicio"] < DESTELLO
           and e["tipo"] != "accion"]
    if not act:
        return
    act.sort(key=lambda e: e["inicio"])
    for i, e in enumerate(act[-2:]):
        col, nom = MARCA_EV.get(e["tipo"], (GRIS, e["tipo"]))
        lado = int(e.get("lado", -1))
        m = e.get("magnitud_m", float("nan"))
        u = "m" if e["tipo"] != "brazo" else "x torso"
        txt = "%s %d%s" % (nom.upper(), e.get("indice", 0),
                           ("  %.2f %s" % (m, u)) if np.isfinite(m) else "")
        y = 40 + 44 * i
        x = 14 if lado == 0 else W_CLIP - 330
        (tw, th), _b = cv2.getTextSize(txt, FUENTE, 0.86, 2)
        sub = lienzo[max(0, y - th - 8):y + 8, x - 6:x - 6 + tw + 12]
        if sub.size:
            sub[:] = (0.35 * sub + 0.65 * np.array(col, "float64")).astype("uint8")
        _texto(lienzo, x, y, txt, (255, 255, 255), 0.86, 2)


def _rotulo(lienzo, clip, k, k0, fps, C, ap, x00, txt_estado, col_estado, anc, tv,
            Xt=None, Yt=None, dist=None):
    """QUE HACE: el bloque de texto con el cuadro, el segundo, la posicion de cada uno, la distancia
    entre los dos y el estado de la medida."""
    y = H_CLIP
    cv2.rectangle(lienzo, (0, y), (W_CLIP - 1, y + H_ROTULO - 1), (22, 22, 22), -1)
    xs = [(Xt[k, l] if Xt is not None else C[k, l, 0] - x00) for l in (0, 1)]
    d = (dist[k] if dist is not None else ap["dist"][k])
    _texto(lienzo, 8, y + 15, "%s   frame %d   t = %.2f s" % (clip, k, (k - k0) / fps),
           BLANCO, 0.44, 1)
    _texto(lienzo, 430, y + 15, tv, GRIS, 0.35, 1)
    for l, nom in ((0, "left"), (1, "right")):
        t = ("%+.2f m" % xs[l]) if np.isfinite(xs[l]) else "not measured"
        if Yt is not None and np.isfinite(Yt[k, l]):
            t += " / %.2f m across" % Yt[k, l]
        _texto(lienzo, 8 + 235 * l, y + 33, "%s %s" % (nom, t), COL[l], 0.40, 1)
    _texto(lienzo, 420, y + 33, "distance between fencers %s"
           % (("%.2f m" % d) if np.isfinite(d) else "-"), BLANCO, 0.46, 1)
    for j, li in enumerate(_parte(txt_estado, 100)[:2]):
        _texto(lienzo, 8, y + 50 + 14 * j, li, col_estado, 0.38, 1)


def _parte(txt, n):
    """QUE HACE: parte un texto en lineas de como mucho `n` caracteres sin cortar palabras. Hace
    falta porque la primera version escribia el estado de la medida en una linea y se salia del
    panel: se veia "la ESCALA pued" y se perdia justo el aviso."""
    out, li = [], ""
    for p in txt.split():
        if len(li) + len(p) + 1 > n:
            out.append(li)
            li = p
            continue
        li = (li + " " + p).strip()
    if li:
        out.append(li)
    return out


if __name__ == "__main__":
    import csv
    clips = [c for c in sys.argv[1:] if not c.startswith("-")]
    lenta = "--lenta" in sys.argv
    if "--publico" in sys.argv:
        """`--publico` deja el video con todos los pixeles generados, para los repos publicos."""
        PUBLICO = True
        globals()["PUBLICO"] = True
    lim = None
    for a in sys.argv[1:]:
        if a.startswith("--cuadros="):
            lim = int(a.split("=")[1])
    filas = []
    for c in clips:
        try:
            d = una(c, max_cuadros=lim, lenta=lenta,
                    sufijo=("_slow" if lenta else "")
                           + ("_public" if globals().get("PUBLICO") else ""))
        except Exception as exc:
            import traceback
            print("%-38s EXCEPCION %s" % (c, exc))
            traceback.print_exc()
            continue
        if d is None:
            continue
        filas.append({"clip": c, "mp4": pathlib.Path(d["mp4"]).name,
                      "MB": round(pathlib.Path(d["mp4"]).stat().st_size / 1e6, 2),
                      "ancla_deriva_m": round(d["ancla"].get("deriva_m", float("nan")), 3),
                      "ancla_fiable": d["ancla"].get("fiable"),
                      "estado": d["estado"],
                      "n_eventos": sum(d["n"].values())})
    if filas:
        with open(SALIDA / "videos.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)
        print("\n%d videos en %s" % (len(filas), SALIDA))

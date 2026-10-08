"""Los videos del repo PUBLICO: UNA CAPA POR VIDEO, sobre el clip, a los fps del clip.

LO QUE PIDIO EL USUARIO, y cada cosa cambio una decision de este fichero:

  *"los videos se ven con el fondo negro, no se ve el clip"* -> **el clip se ve**. Antes se dibujaba
  sobre negro para no publicar fotogramas de la emision; el usuario ha decidido publicarlos.

  *"las del masking y esa, en vez de verlo en un grid, mete de una en una"* -> **un MP4 por capa**, no
  una rejilla de seis celdas. La rejilla metia seis paneles en 2880 px, o sea 960 por celda; ahora
  cada capa tiene el ancho entero y se lee.

  *"ese video mejor para que tenga mas fps y se vea mejor de calidad y mas fluido"* -> **`PASO = 1`**,
  todos los cuadros utilizables, y el MP4 a los fps del clip. Antes iba uno de cada cuatro, o sea
  6,25 fps, y de ahi que pareciera un GIF a saltos.

  *"los conos de los floretes se ven mal... son como los 2D antiguos, no es lo nuevo"* -> tenia razon
  y era un error mio: se pintaba la cuna plana de `e7` con semiapertura 25,4 gr. Ahora se pinta el
  cono vigente: relleno **graduado por la funcion de peso**, alcance 1,10 m de muniera a punta, y la
  hoja marcada `BLADE SEEN` cuando se detecto.

  *"el cono no esta centrado en el florete"* -> **tenia razon otra vez, y esta era peor porque el pie
  de la figura afirmaba lo contrario.** Hasta el 2026-09-11 esto llamaba a `e21.panel_despues`, que
  centra en el eje del **antebrazo** en sus cinco llamadas geometricas y lee la hoja solo para pintar
  el segmento verde. Medido en el cuadro que miro el usuario (`BabyishIdiotic...` c0167): la hoja a
  **21,4 gr** del eje, **fuera del arco solido** del nucleo (+-14,4) y al 0,65 del borde de la
  envolvente. Y el panel se contradecia a si mismo: el estado decia *"centred on the MEASURED blade"*
  y el pie fijo *"grey dashes = the forearm axis the cone is built on"*.

## LO QUE NO SE REIMPLEMENTA, Y ES EL PUNTO

El campo se dibuja con `e24.rellena_campo` -> `cono3d_lib.campo_hoja` con `perfil="lineal"`, que es
la forma operativa, y **el centro y la anchura se LEEN de `cono_campo_<clip>.npz`**, que es el dato
publicado. De `e21` se conservan el raster de solape y la cruz del cruce, que `e24` no dibuja.
Copiar el dibujo aqui habria creado una segunda version de la apertura y del criterio de la hoja, y
este proyecto ya perdio un dia por una visualizacion que no coincidia con sus datos. Por la misma
razon `cono_lib.peso_sector` no se toca: de el dependen todas las cifras publicadas.

Igual con la mascara limpia (`capas_limpias.mascara_tirador`), el esqueleto (`video_revision._pinta`)
y la eleccion de la pareja (`capas_limpias.elige_pareja_del_clip`).

## COMO SE COMPONE CADA CAPA SOBRE EL CLIP

La capa se mezcla **solo dentro de la mascara** y con el fotograma todavia visible por debajo
(`MEZCLA`), y fuera de la mascara el fotograma se **oscurece** (`OSCURECE`). Asi se ve a la vez que
la capa cae encima de la persona correcta y que lo de alrededor no se toco, que es lo que el fondo
negro no dejaba juzgar. Un contorno blanco marca el borde de la mascara.

USO:  python panel_publico.py [CLIP ...]   con el interprete de transformers + torch + CUDA
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

CODIGO = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
sys.path.insert(0, str(CODIGO))

CLIPS = sys.argv[1:] or ["proudDungBeetleBoglenTerrier", "uptightSlothEveningBat",
                         "BabyishIdioticCockerspaniel"]
"""CLIPS LARGOS, A 1080p Y CON HOJA DETECTADA. Los tres criterios son del usuario y de la medida:

  *"el clip usado para esto es cortisimo, muy pocos segundos, no se aprecia bien; clips de 15 seg por
  ejemplo, los de mas calidad para que tengan menos fallos posible"*.

Asi que se eligieron por censo de las 117 detecciones cacheadas: **1920x1080** y **T >= 360 cuadros**,
y, sobre todo, **duracion real en segundos y no numero de cuadros**. Ese fue mi primer censo y estaba
mal: filtre por `T >= 360` y me salio `GentleBlueIndianspinyloach`, que tiene 500 cuadros pero va a
**50 fps**, o sea **10 segundos**, justo lo que el usuario no queria. Cuadros no son tiempo hasta que
los divides por los fps. Y ademas de ese clip solo 126 de 500 cuadros son utilizables.

Tambien se prefieren los que **tienen hoja detectada en cache**, porque son los unicos que pueden
ensenar el cono centrado en una hoja MEDIDA (`origen = 0`) en vez de en la direccion supuesta del
antebrazo. Los tres elegidos: 14,7 s con 30 hojas, 14,4 s con 12 y 12,1 s con 18.

Se descarta `PowerfulAllLarva` aunque cumple duracion y resolucion: es precisamente el clip donde la
tuberia elige al **arbitro** como tirador. Es un ejemplo valioso del defecto abierto, pero no de
calidad. Y `ZanyMeagerIndianrockpython` e `ImpeccablePertinentBasilisk` ya no estan: 250 cuadros, o
sea 8-10 segundos, y el usuario dijo que asi no se apreciaba."""
PASO = 1
ANCHO = 1100
"""1100 px de ancho y no 2880. Antes se metian seis paneles en una rejilla de 2880, o sea 960 por
celda; ahora cada capa tiene un video propio y se lleva el ancho entero. 1100 mantiene el fichero en
un tamanio que un README puede abrir con 18 videos dentro."""

sys.argv = ["mosaico_seis.py", CLIPS[0], str(PASO)]
import mosaico_seis as M6
M6.IDIOMA = "en"

import capas_limpias as CL
import cono3d_lib as C3
import cono_lib as CO
import cv2
import e20_estado_del_cono as E20
import e21_pinta_cono_arreglado as E21
import e24_campo_dato_y_figura as E24
import pickle
import time

HERRAM = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERRAM))
import video_revision as VR

SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\publico_generado")
FIJOS = SALIDA / "fijos"
COL = M6.COL
MEZCLA = 0.68
OSCURECE = 0.45
"""`MEZCLA` es cuanto pesa la capa DENTRO de la mascara y `OSCURECE` cuanto queda del fotograma fuera.

Los dos numeros salen de lo que hay que poder juzgar mirando, no del gusto. Con la capa al 100 %
dentro de la mascara no se puede comprobar que cae sobre la persona correcta, porque tapa a quien
deberia estar debajo; con el fondo al 100 % fuera, la capa se pierde contra un suelo azul o rojo de
pista. Al 0,68 y 0,45 se ven las dos cosas: a quien se le ha puesto y que alrededor no se toco."""

CAPAS = ("pose", "seg", "normal", "matting", "pointmap", "foil")
TITULOS = {
    "pose": ("POSE - 308 keypoints per fencer",
             ["green = left fencer, orange = right, red = a person we did NOT select"]),
    "seg": ("SEG - body-part segmentation, after cleaning",
            ["blended inside the mask only, so you can check it landed on the right person"]),
    "normal": ("NORMAL - surface normals",
               ["one axis per colour channel; the dense layer that DOES hold up"]),
    "matting": ("MATTING - alpha, the soft silhouette",
                ["at very low alpha the blade edge appears: that is what the foil detector uses"]),
    "pointmap": ("POINTMAP - the model's depth-like channel",
                 ["NOT USABLE AS DEPTH and not used: shown because it was asked for"]),
    "foil": ("FOIL - the threat cone, as a graded probability field",
             ["centred on the blade where MEASURED (+-8.6 deg), on the forearm where not "
              "(+-33.0 deg) | reach 1.10 m"]),
}
"""UNA LINEA DE PIE POR CAPA, MAS LA DE ESTADO, Y NI UNA MAS. Antes eran dos o tres, y con la de
estado salian **cuatro lineas de 121, 117 y 115 caracteres** encima de la imagen.

POR QUE SE RECORTA. El pie ocupa pixeles del video y es el sitio PEOR para leer: letra de 0,44 en un
MP4 de 1100 px de ancho que en un README se ve todavia mas pequenio. Lo que cabe en una linea es lo
que hay que saber **mirando ese cuadro**; todo lo que es contexto -- el mapa de indices, la fuga del
arbitro medida en 0,0000, el 76 % del pointmap contra el 28 de 28 del pie libre, la derivacion de las
dos aperturas -- se lee **en el README**, donde no cuesta pixeles y se puede citar. Nada se ha
perdido: esta en la tabla de canales de `repo_publico.CANALES`, que es la que genera la ficha.

`pies_de` sigue anadiendo SIEMPRE la linea de estado, tambien vacia, porque el alto del cuadro tiene
que ser constante o `cv2.VideoWriter` descarta cuadros."""


def pies_de(q, estado):
    """El pie de una capa: sus lineas fijas MAS una linea de estado, SIEMPRE.

    POR QUE SIEMPRE UNA. El alto del cuadro depende de cuantas lineas de pie lleve, y
    `cv2.VideoWriter` esta abierto con el alto del primer cuadro: uno con una linea de menos **se
    descarta**, y OpenCV lo avisa con un `Failed to write frame` por cuadro que se pierde entre el
    resto de la salida. Me paso en la primera prueba con los cuadros descartados, que llevaban solo la
    razon del descarte y ninguna linea fija.

    Con la linea de estado obligatoria -vacia si no hay nada que decir- el alto es constante por capa
    por construccion, y el fallo no puede volver.
    """
    return list(TITULOS[q][1]) + [estado]


def rotula(im, tit, pies, k, ancho=ANCHO):
    """Titulo arriba y pie abajo, EN BANDAS AniADIDAS y nunca pintadas sobre la imagen.

    Pintar el rotulo encima ya me costo comerme la linea que explicaba por que una capa no se usa.
    Aqui la imagen se escala al ancho pedido y las bandas se apilan con `vstack`, asi que no pueden
    tapar nada. Quien escriba un video con esto tiene que sumar el alto de las bandas al tamanio del
    `VideoWriter`: si no, descarta cada cuadro por tamanio y el MP4 sale vacio sin un solo error.
    """
    h, w = im.shape[:2]
    v = cv2.resize(im, (ancho, max(1, int(round(h * ancho / float(w))))),
                   interpolation=cv2.INTER_AREA)
    arr = np.zeros((30, ancho, 3), "uint8")
    arr[:] = (14, 14, 20)
    cv2.putText(arr, tit, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 245), 1, cv2.LINE_AA)
    if k is not None:
        cv2.putText(arr, "frame %d" % k, (ancho - 130, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (140, 220, 255), 1, cv2.LINE_AA)
    ab = np.zeros((10 + 20 * len(pies), ancho, 3), "uint8")
    ab[:] = (14, 14, 20)
    for j, t in enumerate(pies):
        cv2.putText(ab, t, (10, 18 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 205, 240), 1,
                    cv2.LINE_AA)
    return np.vstack([arr, v, ab])


def sobre_clip(bgr, capa_bgr, mk):
    """La capa mezclada DENTRO de la mascara, el clip oscurecido fuera, y el borde marcado."""
    out = (bgr.astype("float32") * OSCURECE).astype("uint8")
    m3 = mk[:, :, None]
    mez = (bgr.astype("float32") * (1.0 - MEZCLA) + capa_bgr.astype("float32") * MEZCLA)
    out = np.where(m3, np.clip(mez, 0, 255).astype("uint8"), out)
    cs, _ = cv2.findContours(mk.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cs, -1, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def celda_pose(bgr, kt, aj, B, tipo, k, W, H, mano_por_lado):
    """Los dos esqueletos SOBRE el clip, con las cajas y los zoom de la mano armada."""
    im = (bgr.astype("float32") * 0.72).astype("uint8")
    fino = im.copy()
    hueco = SinPisarse(H)
    for lado in (0, 1):
        kp, sc = kt[("t", lado)]
        M6.pinta(im, kp, sc, COL[lado], g=3, r=5, rm=2)
        M6.pinta(fino, kp, sc, COL[lado], g=1, r=3, rm=1)
    for o in aj:
        o = np.asarray(o).astype(int)
        cv2.rectangle(im, (o[0], o[1]), (o[2], o[3]), M6.COL_AJENO, 1)
    for lado in (0, 1):
        b = (B[k, lado] * np.array([W, H, W, H])).astype(int)
        cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), COL[lado], 3)
    for lado in (0, 1):
        b = (B[k, lado] * np.array([W, H, W, H])).astype(int)
        et = ["LEFT", "RIGHT"][lado]
        et += {1: " interpolated", 2: " held", 3: " re-detected"}.get(int(tipo[k, lado]), "")
        x = int(max(6, b[0]))
        _texto(im, et, x, hueco.sitio(x, max(26, b[1] - 10)), COL[lado], 0.9, 2)
    for o in aj:
        o = np.asarray(o).astype(int)
        x = int(max(6, o[0]))
        _texto(im, "NOT selected", x, hueco.sitio(x, min(H - 8, o[3] + 26)), M6.COL_AJENO, 0.7, 2)
    lado_px = int(0.30 * H)
    for lado in (0, 1):
        mano = mano_por_lado.get(lado)
        if mano is None:
            continue
        kp, sc = kt[("t", lado)]
        ins, s = M6.inset_mano(fino, kp, sc, mano, (B[k, lado, 3] - B[k, lado, 1]) * H, lado_px,
                               nunca_reduce=True)
        if ins is None:
            continue
        ax = 6 if lado == 0 else W - lado_px - 6
        ay = H - lado_px - 8
        im[ay:ay + lado_px, ax:ax + lado_px] = ins
        cv2.rectangle(im, (ax, ay), (ax + lado_px, ay + lado_px), COL[lado], 2)
        _texto(im, "%s hand  zoom x%.1f" % (["left", "right"][lado], lado_px / float(s)),
               ax + 6, ay + 20, COL[lado], 0.5, 1)
    return im


def _texto(im, t, x, y, col, esc, gr):
    """Texto con reborde negro. SOBRE EL CLIP HACE FALTA: un rotulo verde sobre una pista verde
    desaparece, y ese es justo el caso de las pistas de esgrima."""
    cv2.putText(im, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, esc, (0, 0, 0), gr + 2, cv2.LINE_AA)
    cv2.putText(im, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, esc, col, gr, cv2.LINE_AA)


class SinPisarse:
    """Coloca rotulos sin que se pisen: cada uno baja hasta separarse de los ya puestos.

    Con dos tiradores y hasta tres detecciones descartadas hay cinco cajas en un cuadro y sus
    rotulos comparten bandas de pixeles. Una regla fija (encima de la caja, o debajo) no lo resuelve:
    lo probe de las dos formas y en las dos hubo un choque. Hay que mirar donde estan los otros.
    """
    PASO, ALTO, ANCHO = 22, 20, 230

    def __init__(self, H):
        self.puestos, self.H = [], H

    def sitio(self, x, y, ancho=None, alto=None):
        """`ancho` y `alto` por llamada, porque NO todos los rotulos miden lo mismo.

        Los 230 px por defecto son los de `LEFT interpolated` y compania. El rotulo del cono mide
        unos 440 px y ocupa dos lineas, y con los 230 esta clase decidia que dos rotulos de vertices
        cercanos no se pisaban -- y se pisaban, se veia en el PNG. Quien pinta sabe cuanto mide su
        texto; esta clase no puede saberlo.
        """
        anc = self.ANCHO if ancho is None else ancho
        alt = self.ALTO if alto is None else alto
        for signo in (1, -1):
            yy = y
            for _ in range(14):
                if all(abs(yy - py) >= alt or abs(x - px) >= anc
                       for px, py in self.puestos) and 22 <= yy <= self.H - 6:
                    self.puestos.append((x, yy))
                    return yy
                yy += signo * self.PASO
        self.puestos.append((x, y))
        return y


def calidad_del_cuadro(org, pres, mask, B, k, W, H):
    """Cuanto de bueno es este cuadro como EJEMPLO. Se usa para los PNG fijos, no para el video.

    Un fijo elegido solo por reparto temporal cayo en una pose degenerada y la figura publica
    ensenaba un garabato. Se exige: las dos poses medidas, las dos cajas enteras dentro del cuadro, y
    de ahi la caja mas alta, que es la que mas pixeles de tirador tiene.
    """
    if not mask[k] or not pres[k].all():
        return -1.0
    if (org[k] != 0).any():
        return -1.0
    b = B[k] * np.array([W, H, W, H])
    if (b[:, 0] < 2).any() or (b[:, 2] > W - 2).any():
        return -1.0
    return float(np.min(b[:, 3] - b[:, 1]))


def decide_manos(clip, KP, SC, B, W, H, org, mask, campo=None):
    """LA MANO ARMADA, con los TRES argumentos que la hacen reproducible. Una sola vez por clip.

    LO QUE ESTABA MAL, y eran tres fallos en una llamada de seis argumentos posicionales
    (`CO.mano_armada(KP, SC, B, W, H, lado)`), repetida en dos sitios de este fichero:

      1. **`accion = None`**, o sea que el peldanio `encarado_accion` de la cascada **no podia
         dispararse jamas**. Y es el peldanio que aporta 8 de los 19 tiradores del conjunto
         reservado: sin el, la cascada no es la que se publico.
      2. **`org = None, mask = None`**, o sea que votaban TODOS los cuadros con confianza, incluidas
         las poses interpoladas y sostenidas. El docstring de `mano_armada` lo dice literal: *"sin
         ellos... las cifras de arriba no son exactamente las que saldran. Quien quiera
         reproducirlas tiene que pasarlos."*
      3. **`dudosa` se tiraba** (`mano, _x = ...`), asi que un vertice a cara o cruz se dibujaba
         igual que uno firme. Eso se arregla en `_rotula_vertice`.

    Y LA INESTABILIDAD NO ES TEORICA: ralear el muestreo cambiaba la respuesta cuatro veces en el
    mismo tirador (`PASO 1` -> `mayoria_3_dudosa`, `PASO 2` -> `pie_firme`, `PASO 5` ->
    `geometria_a`, `PASO 10` -> ni contesta). De ahi que `cono_campo_proudDungBeetleBoglenTerrier.npz`
    diga `mano_lado0 = A` y el PNG publicado imprimiera `hand B s0`: dos llamadas con distinta
    entrada y ninguna marca de que una de las dos no valia.

    `accion` se construye con `mano_guante.mascara_accion`, que devuelve `None` -- **no un array de
    ceros** -- cuando el clip no tiene eventos. Los dos significan cosas distintas y `mano_armada`
    los trata distinto: `None` es «no medido» y no aplica la puerta; ceros seria «ninguna accion»,
    que es una afirmacion que no tenemos.

    Y AQUI SALE `frac_accion nan (n=0)` EN LOS TRES CLIPS PUBLICADOS, **que no es un fallo del
    cableado**: `eventos.csv` cubre 22 clips y **ninguno de estos tres esta dentro** (comprobado, no
    supuesto). O sea que para ellos el peldanio `encarado_accion` no tiene con que dispararse y la
    cascada cae al siguiente, marcando `dudosa`. Lo que cambia respecto de antes es que el peldanio
    **ya puede** dispararse cuando el clip tenga eventos -- en el conjunto reservado resuelve 8 de
    19 tiradores -- y que el `nan` de ahora dice «no medido» en vez de esconder que nunca se
    intentaba. Si algun dia estos tres clips entran en `eventos.csv`, esto mejora sin tocar codigo.
    """
    accion = None
    try:
        import mano_guante as MG
        accion = MG.mascara_accion(clip, len(KP))
    except Exception as e:
        print("   sin mascara de accion (%s): el peldanio `encarado_accion` no se evalua" % e)
    dets = {}
    for lado in (0, 1):
        mano, det = CO.mano_armada(KP, SC, B, W, H, lado, org=org, mask=mask, accion=accion)
        if mano is None:
            print("   lado %d sin mano armada decidida: no hay cono" % lado)
            return None
        det["mano_local"] = mano
        npz = (campo or {}).get("mano", {}).get(lado)
        if npz in ("A", "B") and npz != mano:
            """MANDA LA DEL NPZ, y esto arregla un defecto que el usuario vio en una figura publicada.

            EL CASO: en `proudDungBeetleBoglenTerrier` c0163 el cono del lado 0 salia sobre la mano
            de ATRAS y apuntando FUERA del rival. `cono_campo_*.npz` decia `mano_lado0 = A` y el PNG
            imprimia `hand B s0`. Medido sobre las poses de esta tirada: el antebrazo de **A apunta
            al rival en 0,941 de 426 cuadros y el de B en 0,660**, y en c0163 la raiz de A esta en
            x=381 contra x=295 de B con el rival en x=1736, o sea que **A es la mano adelantada**.
            La del NPZ era la correcta.

            POR QUE MANDA LA DEL NPZ Y NO LA DE AQUI, que es el arreglo de fondo: la de aqui sale de
            una tirada de poses **propia y de otro subconjunto de cuadros** (426 de 441, solo los de
            `proc`), y la del NPZ sale de `cono_poses`, que es el juego de poses con el que se midio
            TODA la etapa E. Con dos juegos distintos la cascada da respuestas distintas y ninguna
            avisa: aqui `mayoria_3` salio B con votos de 0,512 y 0,692 -- dos monedas al aire -- y
            alla salio A. **El video tiene que ensenar la decision que viaja en el dato que se
            publica**, no una segunda opinion calculada al pintar, que es el mismo principio por el
            que el centro y la anchura del campo tambien se leen del NPZ.

            LO QUE ESTO NO ARREGLA, Y NO SE FINGE: la decision del NPZ tambien es `dudosa` (todas
            sus fracciones andan por 0,52-0,57), o sea que aqui acerto pero **no por ser firme**. El
            cono sigue saliendo marcado `UNCERTAIN` con su `frac_pie`. Lo que se elimina es la
            contradiccion entre la figura y el dato, no la incertidumbre."""
            print("   lado %d: la cascada local dice %s y el NPZ dice %s -> MANDA EL NPZ"
                  % (lado, mano, npz))
            det["mano"] = npz
            det["por"] = "%s (npz, local decia %s por `%s`)" % (
                det.get("por_npz", "cono_campo"), mano, det.get("por"))
            mano = npz
        dets[lado] = det
        print("   mano lado %d: %s por `%s`%s  |  frac_pie %.3f (n=%d)  frac_accion %.3f (n=%d)"
              % (lado, mano, det.get("por"), "  DUDOSA" if det.get("dudosa") else "",
                 det.get("frac_pie", float("nan")), det.get("n_pie", 0),
                 det.get("frac_accion", float("nan")), det.get("n_accion", 0)))
    return dets


def geometria_cono(clip, KP, SC, B, W, H, dets):
    """La geometria del cono NUEVO, construida con LAS POSES DE ESTA TIRADA.

    POR QUE NO SE LLAMA A `E21.geometria`. Esa funcion saca las poses de `E20.carga`, que lee
    `poses_<clip>.npz`, y **ese fichero solo existe para 39 clips**. Peor: donde existe, guarda la
    ventana de accion y no el clip entero, asi que el vertice del cono sale finito en **60 de 363
    cuadros**. Con eso no se puede hacer un video: la primera version de esto pinto un clip entero
    **sin un solo cono** y solo se noto al mirar el PNG.

    Lo que se reutiliza es el **dibujo**, y desde el arreglo del centro son `e24.rellena_campo` y
    `cono3d_lib.campo_hoja` mas el raster de solape de `e21`; aqui solo cambia de donde vienen las
    poses, y vienen de las de este mismo cuadro.

    LA MANO NO SE VUELVE A DECIDIR AQUI. Antes esta funcion llamaba a `mano_armada` por su cuenta,
    o sea que el clip tenia DOS decisiones de mano armada calculadas por separado -- una para los
    zoom de la capa `pose` y otra para el cono -- y nada garantizaba que coincidieran. Ahora se pasa
    `dets`, que se decidio una vez en `decide_manos`.

    `hojas` son las rectas que `e5`/`e13` ya detectaron, leidas de disco. No se vuelven a detectar:
    haria falta `matting` y podria dar otro resultado, y entonces el dibujo no seria del dato que se
    publico.
    """
    g = {}
    for lado in (0, 1):
        det = dets[lado]
        mano = det["mano"]
        vert, u, ok, _lg = C3.eje_brazo_2d(KP, SC, lado, mano, "codo")
        _pk, pxm = CO.escala_px_m(B, lado, H)
        g[lado] = {"vert": vert, "u": u, "ok": ok, "pxm": pxm, "mano": mano, "det": det,
                   "ancla": KP[:, lado, CO.MANOS[mano]["codo"]].astype("float64")}
    g["W"], g["H"] = W, H
    fin = [int(np.isfinite(g[l]["vert"]).all(axis=1).sum()) for l in (0, 1)]
    print("   cono: eje valido en %d y %d cuadros (lado 0 / lado 1)" % (fin[0], fin[1]))
    return g, E21.hojas_cacheadas()


def campo_del_clip(clip):
    """EL CAMPO DE LA HOJA LEIDO DE `cono_campo_<clip>.npz`, que es EL DATO QUE SE PUBLICA.

    No se recalcula nada: `e24` ya dejo en disco `campo_mu_gr`, `campo_sigma_gr` y `campo_origen`
    **por cuadro y por lado** para los clips de la etapa E, y los tres que se publican estan entre
    ellos (origen 0 en 30 / 12 / 18 cuadros para `proudDungBeetle...`, `uptightSloth...` y
    `Babyish...`). Recalcularlo aqui seria una segunda version de la decision y podria discrepar del
    NPZ que el usuario se puede descargar.

    EL INDEXADO ES ABSOLUTO AL CLIP Y ESTA COMPROBADO, no supuesto: `T` del NPZ coincide con el `T`
    de `det_<clip>.pkl` en los tres clips (441, 431, 363), y `cono_hoja_*.json` guarda `cuadro` en
    numeracion de clip. Si algun dia dejaran de coincidir, esto devuelve `None` en vez de indexar mal.
    """
    f = CO.PRUEBAS / ("cono_campo_%s.npz" % clip)
    if not f.is_file():
        print("   sin cono_campo_%s.npz: el campo se centra en el antebrazo en todos los cuadros"
              % clip)
        return None
    with np.load(f, allow_pickle=True) as z:
        return {"mu": z["campo_mu_gr"], "sigma": z["campo_sigma_gr"], "org": z["campo_origen"],
                "sigma_vista": float(z["sigma_vista_gr"]),
                "sigma_supuesta": float(z["sigma_supuesta_gr"]),
                "mano": {0: str(z["mano_lado0"]), 1: str(z["mano_lado1"])},
                "T": int(len(z["campo_origen"])), "fichero": f.name}


def campo_en(campo, k, lado, u):
    """`(origen, mu_gr, semiangulo_gr)` de este cuadro y lado. CENTRO Y ANCHURA SON UNA SOLA COSA.

    Y ESE ES EL ARREGLO. Antes esto se pintaba con `e21.panel_despues`, que centra en `d["u"][k]`
    -- el eje del ANTEBRAZO -- en sus cinco llamadas geometricas, y leia la hoja solo para pintar el
    segmento verde: el centro **nunca** tocaba la hoja. Medido en el cuadro que miro el usuario
    (`BabyishIdioticCockerspaniel` c0167): la hoja caia a **21,4 gr** del eje, o sea **fuera del arco
    solido** del nucleo (+-14,4 gr) y al 0,65 del camino hasta el borde de la envolvente (+-33,0). El
    pie decia que el campo estaba centrado en la hoja medida. No lo estaba.

    Y NO SE PUEDE MOVER SOLO EL CENTRO. `APERT_ENVOLVENTE_SEMI_GR = 33,0` es POR DEFINICION el p90 de
    |hoja - antebrazo|: es la anchura que le corresponde a un centro puesto en el antebrazo. Con el
    centro en la hoja, la anchura que le toca es el p90 de lo que queda de residuo, **8,6 gr**, o sea
    **3,8x mas estrecha**. Cambiar el centro y dejar los 33,0 seria un segundo error, no medio
    arreglo. Las dos cifras salen del NPZ, no de aqui.

        origen 0 (hoja VISTA)     centro `campo_mu_gr`     semiangulo `campo_sigma_gr` = 8,6 gr
        origen 1 (SUPUESTA)       centro del antebrazo     semiangulo 33,0 gr (p90 medido)

    Y NO SON LOS 4,2 gr DEL TEMBLOR, que es el error que cometi primero y que cazo el control. 4,2 es
    la incertidumbre de la DIRECCION de la hoja entre cuadros (`e27`, 70 pares de 12 asaltos), no la
    anchura de una cunia con el vertice en la raiz de la mano: la recta detectada pasa a 7 px de esa
    raiz y solo al 0,24 del alcance, asi que con +-4,2 el cono **no contenia la hoja que centraba**
    en 21 de 60 cuadros. Derivacion en `cono3d_lib.APERT_HOJA_VISTA_SEMI_GR`, control en
    `02_validacion/herramienta/verifica_cono_campo.py`.

    LOS 180 GRADOS SE RESUELVEN CONTRA EL EJE DE ESTA TIRADA, y esto no es recalcular el campo: una
    recta detectada no tiene sentido propio, y el NPZ la oriento contra el antebrazo de `cono_poses`,
    que son otras poses que las de aqui. Si los dos ejes discreparan mas de 90 gr, el `mu` guardado
    apuntaria hacia atras y el cono saldria del reves. Se comprueba y se voltea; el angulo es el
    mismo, solo se elige el sentido.
    """
    ang_eje = float(np.degrees(np.arctan2(u[1], u[0])))
    if campo is None or k >= campo["T"] or campo["org"][k, lado] != 0:
        return 1, ang_eje, E21.ENVOLVENTE
    mu, sg = float(campo["mu"][k, lado]), float(campo["sigma"][k, lado])
    if not (np.isfinite(mu) and np.isfinite(sg) and sg > 0):
        return 1, ang_eje, E21.ENVOLVENTE
    if np.cos(np.radians(mu - ang_eje)) < 0:
        mu += 180.0
    return 0, mu, sg


ANCHO_ROTULO = 460
"""Lo que mide de ancho el rotulo del vertice a escala 0,48, y se pasa a `SinPisarse` porque su
valor por defecto (230, el de los rotulos cortos de la capa `pose`) daba por buenos dos rotulos que
se pisaban. Medido contando caracteres del texto mas largo, no estimado."""
MARCA_DUDOSA = (90, 200, 255)
"""La color de lo que NO hay que creerse. Un solo sitio, porque el proyecto ya tiene el violeta de
`galeria_inventadas` escrito en linea en dos ficheros y eso es como se desincronizan dos figuras."""


def _rotula_vertice(v, g, lado, k, origen, apert, vert, hueco=None):
    """El rotulo del vertice dice TRES cosas: que mano, sobre que eje, y si hay que creerselo.

    POR QUE SE DIBUJA MARCADO Y NO SE ESCONDE (decision del usuario). `cono_lib.mano_armada` publica
    `dudosa` y su docstring dice literal que *"la tuberia tiene que mirar `dudosa` y excluir al
    tirador, no fiarse de `mano` a secas"*. Hasta ahora esto tiraba la bandera: en
    `proudDungBeetleBoglenTerrier` lado 0 el criterio del pie sale con `frac_pie = 0,522` -- una
    moneda al aire -- y se dibujaba un vertice rotulado sin ninguna marca, y los dos lados de
    `BabyishIdioticCockerspaniel` salen `dudosa` en `mano_guante.csv` y se publicaron limpios.

    Esconder el cono dejaria el video sin nada que mirar justo donde hay algo que aprender. Asi que
    se dibuja **con el arco del nucleo a trazos**, con `UNCERTAIN` y **con el valor de `frac_pie` al
    lado**: con 0,52 el rotulo dira 0,52, que es la forma honesta de ensenar una moneda al aire.
    """
    d = g[lado]
    det = d.get("det") or {}
    dudosa = bool(det.get("dudosa"))
    fp = det.get("frac_pie", float("nan"))
    lineas = ["hand %s s%d | %s" % (
        d["mano"], lado,
        "field on the MEASURED blade +-%.1f deg" % apert if origen == 0
        else "field on the assumed forearm +-%.1f deg" % apert)]
    if dudosa:
        lineas.append("UNCERTAIN hand: %s | front-foot vote %s" % (
            EN_PELDANIO.get(det.get("por"), str(det.get("por"))),
            "%.2f" % fp if np.isfinite(fp) else "not measurable"))
    col = MARCA_DUDOSA if dudosa else (255, 255, 255)
    x = int(max(6, min(vert[0] + 10, v.shape[1] - 430)))
    """EL ROTULO NO PUEDE SALIRSE POR LA DERECHA. El vertice del lado 1 esta cerca del borde y con
    `x = vert[0] + 8` el texto se cortaba: se acota a 430 px del margen, que es lo que ocupan estas
    dos lineas a escala 0,48."""
    y = int(vert[1]) + 20
    if hueco is not None:
        y = hueco.sitio(x, y, ancho=ANCHO_ROTULO, alto=22 * len(lineas))
    for j, t in enumerate(lineas):
        _texto(v, t, x, y + 22 * j, col, 0.48, 1)
    if hueco is not None:
        for j in range(1, len(lineas)):
            hueco.puestos.append((x, y + 22 * j))


EN_PELDANIO = {"pie_firme": "front foot, firm vote", "encarado_accion": "on-guard + in-action gate",
               "hombro_firme": "shoulder, firm vote", "mayoria_3_dudosa": "majority of 3, not firm",
               "geometria_a": "closest-to-opponent fallback"}
"""El nombre en ingles de cada peldanio de la cascada, para el rotulo publico. Mismo diccionario que
`verifica_mano.EN_POR`, que es de donde sale."""


def celda_foil(bgr, g, hojas, clip, k, campo):
    """El cono como CAMPO DE LA HOJA: centrado en la hoja medida cuando la hay, y con su anchura.

    QUE SE REUTILIZA Y DE DONDE. El relleno es `e24.rellena_campo` -> `cono3d_lib.campo_hoja` con
    `perfil="lineal"`, que es la forma operativa del proyecto (`1 - desvio/semiangulo`), o sea la
    MISMA forma que `cono_lib.peso_sector` y por tanto la misma con la que se decide la amenaza. De
    `e21` se conservan las dos cosas que `e24` no dibuja y que aqui hacen falta: el **raster de
    solape**, que es donde podria estar el contacto de hojas, y la **cruz** del candidato de cruce.
    `cono_lib.peso_sector` no se toca: de el dependen las cifras publicadas.

    EL SOLAPE SE CALCULA CON EL CENTRO Y LA ANCHURA QUE SE DIBUJAN, por lado. Si se dejara en los
    +-33,0 del antebrazo mientras el cono pintado es de +-4,2 sobre la hoja, la region de contacto
    saldria FUERA de los conos: la misma discrepancia entre figura y numero que se acaba de arreglar.

    `L` sigue siendo 1,10 m a la escala de ESTA tirada (`cono_lib.escala_px_m`), y no el `campo_L_px`
    del NPZ, porque los vertices son los de estas poses y el NPZ uso `escala_px_m_de_pie`. Mezclar
    los dos estimadores pondria un alcance de una escala sobre un vertice de otra.
    """
    if g is None or not np.isfinite(g[0]["vert"][k]).all() or not np.isfinite(g[1]["vert"][k]).all():
        return None, {}
    try:
        v = bgr.copy()
        info, mus, aperts, ejes = {}, {}, {}, {}
        hueco = SinPisarse(bgr.shape[0])
        for lado in (0, 1):
            o, mu, sg = campo_en(campo, k, lado, g[lado]["u"][k])
            mus[lado], aperts[lado] = mu, sg
            info[lado] = {"origen": o, "apert": sg, "mu_gr": mu}
            ejes[lado] = np.array([np.cos(np.radians(mu)), np.sin(np.radians(mu))])
        cruz, cap, al = E21.raster_solape(v, g, k, E21.ENVOLVENTE, C3.ALCANCE_M,
                                          ejes=ejes, aperts=aperts)
        for lado in (0, 1):
            d = g[lado]
            E24.rellena_campo(v, d["vert"][k], mus[lado], C3.ALCANCE_M * d["pxm"], aperts[lado],
                              E21.COL[lado], "lineal")
        cv2.addWeighted(v, 1.0, cap, al, 0, dst=v)
        for lado in (0, 1):
            d = g[lado]
            L = C3.ALCANCE_M * d["pxm"]
            vert = d["vert"][k]
            E21.arco(v, vert, ejes[lado], L, aperts[lado], E21.COL[lado], 1, trazos=True)
            if info[lado]["origen"] == 1:
                """EL ARCO SOLIDO DEL NUCLEO SOLO EXISTE PARA `origen = 1`. 14,4 y 33,0 son el p50 y
                el p90 de la MISMA distribucion de |hoja - antebrazo|, asi que el nucleo solo tiene
                sentido dentro de esa envolvente. Con la hoja vista la anchura es 4,2 gr y un arco de
                14,4 caeria FUERA del cono dibujado: no hay p50 medido de esa procedencia y no se
                inventa uno."""
                E21.arco(v, vert, ejes[lado], L, E21.NUCLEO, E21.COL[lado], 1,
                         trazos=bool((d.get("det") or {}).get("dudosa")))
            E21.arco(v, vert, ejes[lado], CO.HOJA_M * d["pxm"], aperts[lado], (255, 255, 255), 1,
                     trazos=True)
            if np.isfinite(d["ancla"][k]).all():
                """El antebrazo en trazos grises SIEMPRE, tambien con `origen = 0`: es lo que permite
                ver a ojo cuanto se ha separado el campo del eje, que es justo el defecto que el
                usuario encontro mirando."""
                E21.trazos(v, d["ancla"][k], vert, E21.EJE, 1)
            cv2.circle(v, np.int32(vert), 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(v, np.int32(vert), 4, (255, 255, 255), -1, cv2.LINE_AA)
            _rotula_vertice(v, g, lado, k, info[lado]["origen"], aperts[lado], vert, hueco)
            ln = hojas.get((clip, k, lado))
            if ln is None:
                continue
            p1, p2 = np.int32(ln["p1"]), np.int32(ln["p2"])
            cv2.line(v, tuple(p1), tuple(p2), (0, 0, 0), 5, cv2.LINE_AA)
            cv2.line(v, tuple(p1), tuple(p2), E21.HOJA, 3, cv2.LINE_AA)
            tx = (int(p1[0]), int(p1[1]) + 26) if p1[1] < p2[1] else (int(p2[0]), int(p2[1]) + 26)
            _texto(v, "BLADE SEEN s%d" % lado, tx[0],
                   hueco.sitio(tx[0], tx[1], ancho=ANCHO_ROTULO), E21.HOJA, 0.5, 1)
        if cruz is not None:
            cv2.drawMarker(v, (int(cruz[0]), int(cruz[1])), (40, 255, 255), cv2.MARKER_CROSS, 26, 3)
    except Exception:
        return None, {}
    return v, info


def _estado_foil(hay_cono, org_k, g, k):
    """La linea de estado del cono: QUE EJE se uso en ESTE cuadro, POR LADO, y con su anchura.

    LA FRASE VIEJA SE CONTRADECIA CON EL PIE FIJO DE SU PROPIA CAPA. Decia *"the field is centred on
    the MEASURED blade (origen 0)"* -- copiada de `e24`, donde si era verdad -- mientras el pie fijo
    de la misma imagen decia *"grey dashes = the forearm axis the cone is built on"*. Las dos lineas
    en el mismo panel, y la que tenia razon era la del pie: el centro estaba en el antebrazo.

    Ahora se dice lado por lado, porque **los dos lados pueden tener procedencia distinta en el
    mismo cuadro** y el caso se da (en `uptightSlothEveningBat` solo coinciden en 4 cuadros). Y el
    porcentaje agregado -- «3 % de los conos» -- se queda fuera de esta linea: es una propiedad del
    clip entero y no de este cuadro, y mezclarlos es como se han colado ya varias cifras sin
    denominador en este proyecto.
    """
    if not hay_cono:
        return "this frame: no usable arm axis, so no cone"
    trozos = []
    for lado in (0, 1):
        det = (g[lado].get("det") or {}) if g else {}
        inf = org_k.get(lado) or {}
        ap = inf.get("apert", float("nan"))
        trozos.append("s%d %s +-%.1f deg%s" % (
            lado,
            "blade MEASURED, field on IT" if inf.get("origen") == 0
            else "blade not seen, field on the FOREARM",
            ap, " (hand UNCERTAIN)" if det.get("dudosa") else ""))
    return "this frame: " + " | ".join(trozos)


def un_clip(clip, cap, mapa):
    with open(M6.SCR / ("det_%s.pkl" % clip), "rb") as fh:
        cajas, cuadros, W, H, T = pickle.load(fh)
    gris = []
    v = cv2.VideoCapture(mapa[clip])
    fps = v.get(cv2.CAP_PROP_FPS) or 25.0
    while True:
        ok, im = v.read()
        if not ok:
            break
        gris.append(cv2.cvtColor(cv2.resize(im, (160, 90)), cv2.COLOR_BGR2GRAY))
    v.release()
    sel = CL.elige_pareja_del_clip(cajas, cuadros, W, H, T, gris=gris)
    B, pres, msk, tipo = sel["B"], sel["pres"], sel["mask"], sel["tipo"]
    KS = list(range(0, T, PASO))
    proc = [k for k in KS if msk[k] and pres[k].all()]
    print("%s  %dx%d  T=%d  fps %.1f  utilizables %d  se pintan %d de %d"
          % (clip, W, H, T, fps, int(msk.sum()), len(proc), len(KS)))
    if not proc:
        print("   sin cuadros utilizables, se salta")
        return

    N = 308
    KP = np.full((T, 2, N, 2), np.nan, "float32")
    SC = np.zeros((T, 2, N), "float32")
    guardadas, ajenos_k = {}, {}
    vid = cv2.VideoCapture(mapa[clip])
    t0 = time.perf_counter()
    for i, k in enumerate(proc):
        vid.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, im = vid.read()
        if not ok:
            continue
        rgb = np.ascontiguousarray(im[:, :, ::-1])
        aj = CL.ajenos_del_cuadro(sel, k, H)
        cbs = [B[k, 0] * np.array([W, H, W, H]), B[k, 1] * np.array([W, H, W, H])] + aj
        poses = cap.pose(rgb, cbs)
        kt = {("t", 0): poses[0], ("t", 1): poses[1]}
        for j in range(len(aj)):
            kt[("a", j)] = poses[2 + j]
        guardadas[k], ajenos_k[k] = kt, aj
        for lado in (0, 1):
            kp, sc = poses[lado]
            n = min(len(kp), N)
            KP[k, lado, :n], SC[k, lado, :n] = kp[:n], sc[:n]
        if i % 80 == 0:
            print("   pose c%04d  %d/%d  %.2f s/cuadro"
                  % (k, i + 1, len(proc), (time.perf_counter() - t0) / (i + 1)))
    vid.release()

    org = np.ones((T, 2), "int8")
    for kk in proc:
        for lado in (0, 1):
            org[kk, lado] = 0 if tipo[kk, lado] in (0, 3) else 1
    """`org` SE CONSTRUYE ANTES DE DECIDIR LA MANO, porque `mano_armada` lo necesita para que voten
    solo los cuadros con pose MEDIDA. Y el valor por defecto es **1 y no 0**: un cuadro que esta
    fuera de `proc` no se ha posado nunca, asi que su procedencia no es «medida». Con el 0 de antes,
    esos cuadros se colaban como medidos en cualquier criterio que filtre por `org == 0`."""

    campo = campo_del_clip(clip)
    """EL CAMPO SE LEE ANTES DE DECIDIR LA MANO, porque el NPZ trae `mano_lado0`/`mano_lado1` y esa
    es la decision que manda: ver `decide_manos`."""
    dets = decide_manos(clip, KP, SC, B, W, H, org, msk, campo)
    if dets is None:
        print("   sin mano armada decidible en los dos lados: no habra capa de cono")
    mano_por_lado = {l: dets[l]["mano"] for l in (0, 1)} if dets else {}
    g_cono, hojas = geometria_cono(clip, KP, SC, B, W, H, dets) if dets else (None, {})
    n_hoja = sum(1 for (c, kk, _l) in hojas if c == clip)
    n_org0 = int((campo["org"] == 0).sum()) if campo else 0
    print("   mano armada: lado 0 %s, lado 1 %s  |  hojas en cache %d  |  campo: %s, origen 0 en %d"
          % (mano_por_lado.get(0, "-"), mano_por_lado.get(1, "-"), n_hoja,
             campo["fichero"] if campo else "sin NPZ", n_org0))

    fijos = set()
    for j in range(4):
        tramo = proc[j * len(proc) // 4:(j + 1) * len(proc) // 4]
        notas = [(calidad_del_cuadro(org, pres, msk, B, kk, W, H), kk) for kk in tramo]
        buenos = [kk for q, kk in sorted(notas, reverse=True) if q > 0]
        if buenos:
            fijos.add(buenos[0])
    con_hoja = sorted(kk for (c, kk, _l) in hojas if c == clip and kk in set(proc))
    if con_hoja:
        fijos.add(con_hoja[len(con_hoja) // 2])
        """UN FIJO CON HOJA DETECTADA, SIEMPRE QUE HAYA. Es el unico cuadro capaz de ensenar
        `origen = 0`, o sea el campo centrado en una hoja MEDIDA y con su anchura de 4,2 gr en vez
        de en la direccion supuesta del antebrazo con 33,0. En los tres clips que se publican eso
        pasa en 30, 12 y 18 cuadros de 111, 111 y 107 conos, asi que por reparto temporal no
        saldria casi nunca."""
    if len(fijos) < 2:
        fijos = set(proc[int(round(x))] for x in np.linspace(0, len(proc) - 1, 4))
    print("   fijos: %s" % sorted(fijos))

    esc = {}
    for q in CAPAS:
        esc[q] = None
    vid = cv2.VideoCapture(mapa[clip])
    t0, npint = time.perf_counter(), 0
    for i, k in enumerate(KS):
        vid.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, im = vid.read()
        if not ok:
            continue
        if k not in guardadas:
            """CUADRO DESCARTADO: se pinta con su razon, no se salta. Si el video se limitara a los
            cuadros buenos daria la impresion de que la tuberia procesa el clip entero."""
            razon = ("dropped by the pipeline: replay, close-up or cross-fade" if not msk[k]
                     else "dropped by the pipeline: one of the two fencers is missing")
            base = (im.astype("float32") * 0.30).astype("uint8")
            _texto(base, razon, 20, int(0.5 * H), (80, 80, 255), 1.0, 2)
            for q in CAPAS:
                esc[q] = _escribe(esc, q, clip,
                                  rotula(base, TITULOS[q][0], pies_de(q, razon), k), fps)
            continue
        rgb = np.ascontiguousarray(im[:, :, ::-1])
        kt, aj = guardadas[k], ajenos_k[k]

        M_full = np.zeros((H, W), bool)
        cel_mat = np.zeros((H, W, 3), "uint8")
        cel_seg = np.zeros((H, W, 3), "uint8")
        cel_nor = np.zeros((H, W, 3), "uint8")
        alfas, recs = {}, []
        for lado in (0, 1):
            r = CL.mascara_tirador(cap, rgb, sel, k, lado, W, H, kt, con_matting=True)
            if r is None:
                continue
            c, mk = r["c"], r["mask"]
            M_full[c[1]:c[3], c[0]:c[2]] |= mk
            alfas[lado] = float(np.mean(r["alfa"][mk])) if mk.any() else float("nan")
            sl = (slice(c[1], c[3]), slice(c[0], c[2]))
            cel_seg[sl][mk] = CL.pinta_seg(r["seg"])[mk]
            a8 = (255 * np.clip(r["alfa"], 0, 1)).astype("uint8")
            cel_mat[sl][mk] = cv2.cvtColor(a8, cv2.COLOR_GRAY2BGR)[mk]
            n = cap.normal(r["sub"])
            if n.ndim == 3 and n.shape[:2] == mk.shape:
                cel_nor[sl][mk] = CL.pinta_normal(n.astype("float32"))[mk]
            recs.append(CL.mide(r, sel, k, lado, W, H, kt[("t", lado)], bgr=im))
        rk = np.nanmean([d["rec_kp"] for d in recs]) if recs else np.nan
        pm = cap.pointmap(rgb)
        Z = pm[:, :, 2] if pm.ndim == 3 else np.zeros((H, W), "float32")

        vis = {"pose": celda_pose(im, kt, aj, B, tipo, k, W, H, mano_por_lado),
               "seg": sobre_clip(im, cel_seg, M_full),
               "normal": sobre_clip(im, cel_nor, M_full),
               "matting": sobre_clip(im, cel_mat, M_full),
               "pointmap": sobre_clip(im, CL.pinta_z(Z, M_full), M_full)}
        fo, org_k = celda_foil(im, g_cono, hojas, clip, k, campo)
        if fo is None:
            """SIN CONO EN ESTE CUADRO SE DICE, no se deja el fotograma oscurecido y el pie
            describiendo un cono que no esta. Eso fue lo que vi en el primer PNG: un fotograma en
            penumbra con el pie explicando la apertura del cono, y ni un cono dibujado."""
            fo = (im.astype("float32") * 0.32).astype("uint8")
            _texto(fo, "NO CONE IN THIS FRAME: elbow or hand below confidence",
                   24, int(0.52 * H), (90, 200, 255), 0.9, 2)
            org_k = {}
            hay_cono = False
        else:
            hay_cono = True
        vis["foil"] = fo

        estado = {
            "seg": "mask recall against the pose this frame %.3f" % rk,
            "matting": "mean alpha  left %s  right %s"
                       % (("%.2f" % alfas[0]) if 0 in alfas else "-",
                          ("%.2f" % alfas[1]) if 1 in alfas else "-"),
            "foil": _estado_foil(hay_cono, org_k, g_cono, k),
        }
        for q in CAPAS:
            fila = rotula(vis[q], TITULOS[q][0], pies_de(q, estado.get(q, "")), k)
            esc[q] = _escribe(esc, q, clip, fila, fps)
            if k in fijos:
                cv2.imwrite(str(FIJOS / ("%s_%s_c%04d.png" % (q, clip, k))),
                            rotula(vis[q], TITULOS[q][0], pies_de(q, estado.get(q, "")), k,
                                   ancho=1600))
        npint += 1
        if i % 40 == 0:
            print("   c%04d  %d/%d  %.1f s/cuadro"
                  % (k, i + 1, len(KS), (time.perf_counter() - t0) / max(i + 1, 1)))
    vid.release()
    for q in CAPAS:
        if esc[q] is not None:
            esc[q].release()
            _remata(SALIDA / ("%s_%s.mp4" % (q, clip)))
    print("   %d cuadros pintados por capa, a %.1f fps" % (npint, max(8.0, fps / PASO)))


CODEC = "avc1"
FFMPEG = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Microsoft\WinGet\Packages"
                      r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
                      r"\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe")
"""EL CODEC IMPORTA Y NO ES UN DETALLE: decide si el video SE REPRODUCE o se descarga.

El usuario pidio *"mejor mete/incrusta video que sera mejor que un gif"*. Para que un `<video>` se
reproduzca dentro de una pagina hace falta **H.264 en yuv420p**. Los MP4 que subi antes iban con
`mp4v`, que es MPEG-4 parte 2: ningun navegador lo reproduce, asi que un `<video>` con ellos solo
habria ofrecido descargarlos. Medido con `ffprobe`: `mp4v` -> `codec_name=mpeg4`, `avc1` -> `h264`.

Y hay una trampa: al abrir con `avc1`, OpenCV imprime `Could not open codec libopenh264` y
`Failed to initialize VideoWriter`, **y aun asi escribe H.264 correctamente** por otra via. O sea que
el mensaje de error no sirve para decidir; hay que mirar el fichero con `ffprobe`. Si de verdad
fallara, se cae a `mp4v` y el remate de `ffmpeg` lo arregla igual.

`_remata` pasa cada MP4 por `ffmpeg` con `-pix_fmt yuv420p` y `-movflags +faststart`, que es lo que
permite empezar a reproducir sin descargar el fichero entero, y de paso comprime con `-crf`."""


def _escribe(esc, q, clip, fila, fps):
    """Abre el `VideoWriter` con el tamanio del PRIMER cuadro ya rotulado, y no antes.

    POR QUE ASI. El alto final depende de cuantas lineas de pie lleve la capa, y eso cambia por capa.
    Si el tamanio se calculara a mano y no coincidiera, `cv2.VideoWriter` **descarta cada cuadro en
    silencio** y deja un MP4 de 0 bytes sin un solo error. Abrirlo con la forma real del cuadro hace
    imposible ese fallo.
    """
    w = esc.get(q)
    if w is None:
        h, ww = fila.shape[:2]
        p = SALIDA / ("%s_%s.mp4" % (q, clip))
        w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*CODEC), max(8.0, fps / PASO), (ww, h))
        if not w.isOpened():
            w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), max(8.0, fps / PASO),
                                (ww, h))
        esc[q] = w
    w.write(fila)
    return w


def _remata(p):
    """QUE DEVUELVE: el MP4 reproducible en linea, o el original si `ffmpeg` no esta.

    `-vf pad` redondea ancho y alto al par superior: `libx264` en yuv420p **exige dimensiones pares**
    y el alto de estos cuadros sale de apilar bandas de rotulo, o sea que es impar la mitad de las
    veces. Sin el `pad`, `ffmpeg` falla y te deja el fichero de antes pensando que se remato.
    """
    import subprocess
    if not FFMPEG.exists():
        print("   ffmpeg no esta en %s: el MP4 se queda sin rematar" % FFMPEG)
        return p
    tmp = p.with_suffix(".h264.mp4")
    r = subprocess.run([str(FFMPEG), "-y", "-loglevel", "error", "-i", str(p),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:black",
                        "-c:v", "libx264", "-preset", "slow", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
                       capture_output=True, text=True)
    if r.returncode or not tmp.exists():
        print("   ffmpeg fallo en %s: %s" % (p.name, (r.stderr or "")[-200:]))
        return p
    antes = p.stat().st_size
    p.unlink()
    tmp.rename(p)
    print("   %-44s %5.1f -> %5.1f MB  h264 faststart"
          % (p.name, antes / 1e6, p.stat().st_size / 1e6))
    return p


def main():
    import os
    SALIDA.mkdir(parents=True, exist_ok=True)
    FIJOS.mkdir(parents=True, exist_ok=True)
    mapa = {}
    for raiz, _d, fs in os.walk(M6.VIDEOS):
        for f in fs:
            if f.lower().endswith(".mp4"):
                mapa.setdefault(pathlib.Path(f).stem, os.path.join(raiz, f))
    cap = CL.Capas(["pose", "matting", "seg", "normal", "pointmap"])
    print("cinco cabezas cargadas  |  campo: alcance %.2f m  ·  hoja VISTA +-%.1f gr  ·  "
          "antebrazo SUPUESTO +-%.1f gr (nucleo p50 %.1f)"
          % (C3.ALCANCE_M, E24.SIGMA_VISTA, C3.APERT_ENVOLVENTE_SEMI_GR,
             C3.APERT_NUCLEO_SEMI_GR))
    print()
    for clip in CLIPS:
        if not (M6.SCR / ("det_%s.pkl" % clip)).exists() or clip not in mapa:
            print("%s  sin deteccion cacheada o sin video, se salta" % clip)
            continue
        try:
            un_clip(clip, cap, mapa)
        except Exception as e:
            import traceback
            print("%s  FALLO: %s" % (clip, e))
            traceback.print_exc()
        print()
    print("salida en %s" % SALIDA)


if __name__ == "__main__":
    main()

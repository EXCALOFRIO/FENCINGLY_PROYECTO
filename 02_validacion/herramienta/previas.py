"""Genera las VISTAS PREVIAS que se incrustan en el README de Hugging Face.

LO QUE PIDIO EL USUARIO: "enlaces y que se pueda previsualizar ahi, con la explicacion, tipo video de
como se ven las poses... asi cada cosa".

O sea que el README no debe limitarse a enlazar: cada entregable tiene que **verse** desde ahi.

## POR QUE GIF Y NO EL MP4

La ficha de un repo de Hugging Face renderiza markdown, y ahi un `.mp4` **no se reproduce en linea**:
sale como enlace. Un `.gif` si se ve. Asi que de cada video se saca un GIF corto y ligero, y el GIF
enlaza al MP4 completo para quien quiera verlo entero y con calidad.

## LAS DECISIONES DE TAMANIO, Y POR QUE IMPORTAN

Un GIF es una tira de imagenes sin compresion entre cuadros, asi que crece muy rapido. Los tres
mandos son el ancho, los cuadros por segundo y cuantos segundos se cogen:

  `ANCHO = 520`   suficiente para ver un esqueleto y leer un rotulo grande, no para leer letra chica
  `FPS = 8`       el movimiento de esgrima se entiende; a 4 se ve a saltos
  `SEGUNDOS = 5`  una accion completa entra en cinco segundos

Se coge el tramo del **MEDIO** del video a proposito: el principio suele ser la puesta en guardia y el
final el saludo o el marcador, y lo que hay que ver es la accion.

`optimize=True` y paleta adaptativa dejan cada GIF en el orden de 1-3 MB. Se imprime el tamanio de
cada uno, porque un README con veinte megas de GIF tarda en abrir y entonces no cumple su funcion.
"""
from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np
from PIL import Image

VAL = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion")
SALIDA = VAL / "previas"
ANCHO = 900
FPS = 10
SEGUNDOS = 4
COLORES = 128
CUANTAS = 3
"""ANCHO 900 y 256 colores porque el usuario dijo que **los GIF se ven mal**: a 520 px no se leia el
contador del video ni los rotulos de las figuras. Se compensa bajando de 5 a 4 segundos para que el
fichero no crezca, y **la calidad manda sobre la duracion**: para juzgar un contador hacen falta
pixeles, no segundos.

`CUANTAS = 3` porque tambien pidio **varios ejemplos de cada cosa**, no uno."""

ZONA_TRAZADO = (0, 520, 760, 1280)
ZONA_MAPA = (520, 720, 0, 1280)
"""LAS DOS ZONAS DEL VIDEO DE PISTA, en `(y0, y1, x0, x1)`, para poder sacar un GIF de CADA PANEL en
vez de uno del mosaico entero donde no se lee nada.

NO SON NUMEROS A OJO: salen de las constantes de modulo de `video_pista`, que es donde se compone el
cuadro -- `W_VID, H_VID = 1280, 720`, `W_CLIP = 760`, `H_MAPA = 200`, `W_TRAZ = W_VID - W_CLIP` y
`H_TRAZ = H_CLIP + H_ROTULO + H_CUENTA = 520`. Por eso se puede recortar **sin tocar ese fichero**:
la geometria es fija y esta declarada. Si alguien cambia alli el reparto, esto hay que recalcularlo,
y el control es mirar el GIF."""

PISTA_PUB = VAL / "pista" / "video" / "trace_ForcefulAntiqueAlbacoretuna_slow.mp4"
PUB = VAL / "publico_generado"
LARGO = 10
TOPE_LARGO = 3.0
TOPE = 3.0
"""LOS GIF LARGOS: 12 s en vez de 4, con tope de 9 MB en vez de 6.

POR QUE EXISTEN. El usuario dijo *"no hay GIFs largos, mételos"*, y tenia razon en las dos mitades:
faltaban GIF en la ficha publica (habia **cero**) y los que habia duraban 4 s. Cuatro segundos bastan
para ver que una capa existe, pero no para ver **una accion entera** -- una marcha, un romper y un
fondo seguidos -- que es lo que hay que juzgar en el video de la pista.

Y POR QUE HACEN FALTA EN LA FICHA PUBLICA, que es la razon de fondo: esa ficha se lee **tambien desde
GitHub**, y el README de GitHub **no renderiza `<video src="https://...">`** de un origen externo.
O sea que en GitHub la pagina se veia sin una sola imagen en movimiento. El `<video>` se queda para
Hugging Face, que si lo sirve, y el GIF va al lado para que GitHub ensene algo. No es redundancia:
son dos renderizadores distintos.

EL TOPE ES 3 MB PARA TODOS, Y ESO LO DECIDIO MIRAR GITHUB. La primera version salio a 5,2-8,2 MB
por GIF y el usuario dijo que **no los veia**: GitHub sirve las imagenes del README por su CDN y a
ese tamanio o tardan una eternidad o no llegan a pintarse. Un GIF que no se ve pesa infinito. Asi
que 3 MB de tope, **128 colores en vez de 256** (que es el recorte mas barato: en estos planos
apenas se nota y quita un tercio del fichero) y anchos algo menores. El MP4 completo sigue enlazado
para quien quiera calidad."""

VIDEOS = [
    dict(src=PISTA_PUB, nom="pista_completa.gif", seg=LARGO, tope=TOPE_LARGO, ancho=800),
    dict(src=PISTA_PUB, nom="pista_trazado.gif", rec=ZONA_TRAZADO, seg=LARGO, tope=TOPE_LARGO, ancho=700),
    dict(src=PISTA_PUB, nom="pista_mapa.gif", rec=ZONA_MAPA, seg=LARGO, tope=TOPE_LARGO),
    dict(src=VAL / "prioridad" / "publico"
         / "video_right_of_way_KindEnchantingKid_c205_c262.mp4", nom="prioridad.gif"),
    dict(src=VAL / "videos_revision" / "ZanyMeagerIndianrockpython.mp4", nom="poses.gif"),
    dict(src=VAL / "mosaicos" / "mosaico_seis_AromaticShockedGermanspitz.mp4",
         nom="mosaico_seis.gif"),
]
ANCHO_CAPA, LARGO_CAPA = 560, 5
"""LAS CAPAS VAN A 720 px Y 8 s, NO A 900 Y 12, y la razon es medida y no de gusto.

Un GIF de una capa lleva **el fotograma fotografico debajo**, que no se comprime: medido, **~100 kB
por cuadro** a 640 px contra **~19 kB** de los paneles del video de pista, que son graficas planas.
Por eso las dos cosas no pueden tener los mismos ajustes aunque salgan del mismo tope de 3 MB.

LA ELECCION NO ES «LARGO O CORTO» SINO **LARGO Y A SALTOS, O CORTO Y FLUIDO**, y para juzgar un
movimiento de esgrima manda la fluidez: a 2,5 fps no se ve un fondo, se ven tres fotos. Con 560 px y
5 s entran ~10 fps dentro de los 3 MB. El video de pista si aguanta 10 s fluidos porque sus paneles
son graficas planas y se comprimen cinco veces mejor.

El MP4 completo, con los 441 cuadros y sin perder ni un pixel, sigue enlazado al lado de cada GIF."""

VIDEOS += [dict(src=PUB / ("%s_proudDungBeetleBoglenTerrier.mp4" % q),
                nom="capa_%s.gif" % q, seg=LARGO_CAPA, tope=TOPE_LARGO, ancho=ANCHO_CAPA)
           for q in ("pose", "seg", "normal", "matting", "pointmap", "foil")]
"""UNA CAPA POR GIF, del mismo clip, para la ficha publica. Salen de los MP4 que ya genero
`panel_publico`, o sea de los mismos pixeles que el `<video>` de al lado: si se generaran aparte
podrian discrepar, que es el fallo que este proyecto lleva toda la tanda arreglando.

Se elige `proudDungBeetleBoglenTerrier` porque es el clip mas largo de los tres (441 cuadros, 14,7 s)
y el que mas hojas detectadas tiene (30), o sea el que mas veces puede ensenar el campo centrado en
una hoja MEDIDA."""
"""Un video por entregable, elegido a mano, y el tercer campo es el RECORTE.

VUELVE A APUNTAR AL VIDEO CON EL CLIP, Y ES UNA VUELTA ATRAS DELIBERADA (2026-09-23).

Durante un tiempo esto apunto a `_public`, la version que dibuja sobre negro, porque la regla era
que al publico no saliera ni un fotograma de la emision. **El usuario reverso esa regla** y
`panel_publico.py` lo lleva diciendo desde entonces en su primera linea: *"los videos se ven con
el fondo negro, no se ve el clip" -> el clip se ve*. Las capas ya lo hacian y el video de la pista
se habia quedado atras, asi que la ficha publica se contradecia a si misma: capas con fotograma y
pista en negro. Eso es lo que se veia **negro arriba a la izquierda** del README.

Ahora los dos ensenian el clip. `--publico` sigue existiendo en `video_pista.py` y sigue haciendo
lo que hacia; simplemente ya no es lo que se publica.

Y SE SACAN TRES GIF DE ESE VIDEO Y NO UNO. El mosaico entero son 1280x720 con cuatro zonas, y a 900
px de ancho cada zona queda a 0,70 de su tamanio: el trazado y el mapa llevan rotulos de eje que a
ese tamanio no se leen, y el usuario ya se quejo dos veces de que las previas no se ven. Recortando,
el trazado (520x520) y el mapa (1280x200) se escalan a 900 de ancho, o sea **1,7x aumentado** el
trazado y 0,70 el mapa pero con 200 px de alto solo para el mapa en vez de compartirlos.

El de prioridad son 8,6 s de pixeles generados sobre negro, listos para GIF sin recortar."""

IMAGENES = [
    (VAL / "poses_inventadas", "*.png", "poses_inventadas.png",
     lambda p: "hueco05" in p.name or "hueco03" in p.name),
    (VAL / "poses_tres_rondas", "*.png", "tres_rondas.png",
     lambda p: "_R2_" in p.name),
    (VAL / "muestras" / "cascada_deteccion", "cascada_deteccion.png",
     "cascada_deteccion.png", None),
    (VAL / "muestras" / "capas_limpias", "*.png", "capas_limpias.png", None),
    (VAL / "muestras" / "cono_arreglado", "*.png", "cono_arreglado.png",
     lambda p: "Impeccable" in p.name),
    (VAL / "muestras" / "cruce_lados", "*.png", "error_arbitro.png",
     lambda p: "c0224" in p.name),
    (VAL / "muestras" / "mano_dominante", "resumen_mano_armada.png", "mano_armada.png", None),
    (VAL / "prioridad", "fig_*.png", "prioridad.png",
     lambda p: "fallo_intruso" in p.name or "arreglado" in p.name),
]
"""Para las figuras no hace falta GIF: se incrusta una representativa y se enlaza la carpeta. El
`lambda` elige cual, porque la primera por orden alfabetico no suele ser la que mejor explica.

En `poses_tres_rondas` se prefieren los `_R2_`, que son los casos donde la RONDA 2 pasa el filtro,
porque son los que ensenan para que sirve la cascada; el reparto de `imagen` mete despues alguno de
los que FALLAN, y eso es lo que evita una galeria de solo aciertos."""


def gif(ruta, nombre, recorte=None, segundos=None, tope_mb=TOPE, ancho=None):
    """`recorte` es `(y0, y1, x0, x1)` en pixeles del video, o `None` para el cuadro entero.

    Se recorta ANTES de escalar, que es el unico orden que gana pixeles: recortar despues solo
    tiraria los que ya se habian perdido al reducir a 900 px de ancho.

    `segundos` y `tope_mb` por entrada, porque no todos los GIF valen lo mismo: el de la pista tiene
    que ensenar una accion entera (marcha, romper, fondo) y con 4 s no cabe, mientras que el de una
    capa densa se entiende en 4. Ver `LARGO` y `TOPE_LARGO`.
    """
    segundos = SEGUNDOS if segundos is None else segundos
    ancho = ANCHO if ancho is None else ancho
    if not ruta.exists():
        print("   no esta: %s" % ruta.name)
        return None
    cap = cv2.VideoCapture(str(ruta))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    quiere = int(segundos * fps_src)
    ini = max(0, (n - quiere) // 2)
    paso = max(1, int(round(fps_src / FPS)))
    cuadros = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, ini)
    for k in range(quiere):
        ok, im = cap.read()
        if not ok:
            break
        if k % paso:
            continue
        if recorte is not None:
            y0, y1, x0, x1 = recorte
            H0, W0 = im.shape[:2]
            im = im[max(0, y0):min(H0, y1), max(0, x0):min(W0, x1)]
            if im.size == 0:
                print("   recorte %s vacio en %s" % (recorte, ruta.name))
                cap.release()
                return None
        h, w = im.shape[:2]
        im = cv2.resize(im, (ancho, max(1, int(round(h * ancho / w)))),
                        interpolation=cv2.INTER_AREA)
        cuadros.append(Image.fromarray(im[:, :, ::-1]).convert(
            "P", palette=Image.ADAPTIVE, colors=COLORES))
    cap.release()
    if not cuadros:
        print("   sin cuadros: %s" % ruta.name)
        return None
    """SE REINTENTA HASTA CABER EN 6 MB, quitando cuadros y no calidad.

    Un README con veinte megas de GIF tarda en abrir y entonces no cumple su funcion, que es que se
    vea de un golpe. Y ante la disyuntiva se sacrifica **duracion** y no **resolucion**, porque el
    usuario se quejo justamente de que no se veia: para leer un contador hacen falta pixeles.
    """
    dest = SALIDA / nombre
    ini_n = len(cuadros)
    while True:
        cuadros[0].save(dest, save_all=True, append_images=cuadros[1:],
                        duration=int(1000 / FPS), loop=0, optimize=True)
        mb = dest.stat().st_size / 1e6
        if mb <= tope_mb or len(cuadros) <= 8:
            break
        cuadros = cuadros[::2]
    print("   %-26s %3d de %3d cuadros  %4.1f s  %4.1f fps reales  %5.1f MB  %4d px  %s%s" % (
        nombre, len(cuadros), ini_n, segundos, len(cuadros) / max(segundos, 1e-9), mb, ancho,
        "OK" if mb <= tope_mb else "NO CABE", "  recorte" if recorte else ""))
    return dest


def imagen(carpeta, patron, nombre, elige):
    """QUE HACE: escribe hasta `CUANTAS` ejemplos, no uno, y **sin reescalar si ya caben**.

    El usuario dijo que las previas se ven mal. Una causa era el GIF, y la otra era que aqui se
    reescalaba a 1100 px de ancho **aunque el original fuera mas pequenio o justo**, y una figura de
    diez paneles con rotulos pequenios no sobrevive a eso. Ahora solo se reduce si pasa de 1600 px, y
    se reduce con `INTER_AREA`, que es el que no crea aliasing al bajar.
    """
    if not carpeta.exists():
        print("   no esta: %s" % carpeta.name)
        return []
    fs = sorted(carpeta.glob(patron))
    if not fs:
        print("   vacia: %s" % carpeta.name)
        return []
    cand = [f for f in fs if elige(f)] if elige else []
    """Los elegidos por el filtro van primero, y despues se rellena repartiendo a lo largo de la
    lista para que los ejemplos no sean tres fotogramas casi iguales del mismo clip.

    Y EL REPARTO SE APLICA TAMBIEN A LOS CANDIDATOS, no solo al relleno. El docstring decia eso
    desde el principio pero el codigo solo repartia `resto`, asi que un filtro ancho -- `_R2_`, que
    casa con 26 ficheros -- devolvia los tres primeros por orden alfabetico, o sea **tres casos del
    mismo clip**. Se vio en `tres_rondas*.png`: los tres de `AliveIgnorantBarebirdbat`."""
    if len(cand) > CUANTAS:
        pc = max(1, len(cand) // CUANTAS)
        cand = cand[::pc][:CUANTAS]
    resto = [f for f in fs if f not in cand]
    paso = max(1, len(resto) // max(CUANTAS - len(cand), 1))
    orden = cand + resto[::paso]
    dest = []
    for i, f in enumerate(orden[:CUANTAS]):
        im = cv2.imread(str(f))
        if im is None:
            continue
        h, w = im.shape[:2]
        if w > 1600:
            im = cv2.resize(im, (1600, max(1, int(round(h * 1600 / w)))),
                            interpolation=cv2.INTER_AREA)
        nom = nombre if i == 0 else nombre.replace(".png", "_%d.png" % (i + 1))
        p = SALIDA / nom
        cv2.imwrite(str(p), im, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        print("   %-30s de %-38s %5.1f MB" % (nom, f.name[:38], p.stat().st_size / 1e6))
        dest.append(p)
    return dest


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    print("GIF de los videos (tramo del medio, %d s por defecto y %d los largos, %d fps, %d px):"
          % (SEGUNDOS, LARGO, FPS, ANCHO))
    for e in VIDEOS:
        gif(e["src"], e["nom"], e.get("rec"), e.get("seg"), e.get("tope", TOPE),
            e.get("ancho"))
    print()
    print("imagenes representativas:")
    for c, p, n, e in IMAGENES:
        imagen(c, p, n, e)
    tot = sum(f.stat().st_size for f in SALIDA.iterdir() if f.is_file())
    print()
    print("total de las previas en disco: %.1f MB en %s" % (tot / 1e6, SALIDA))
    """EL AVISO SE MIDE SOBRE LO QUE SE INCRUSTA, NO SOBRE LA CARPETA. Los dos numeros se separaron
    en cuanto la carpeta crecio: hay 3 ejemplos de cada figura y varios GIF del mismo video, pero
    `curado_hf.PREVIA` incrusta **uno por entregable**, y lo que hace lento un README es lo que el
    navegador descarga al abrirlo. Avisar por el total de la carpeta daba la alarma con 30,4 MB en
    disco cuando lo incrustado eran 12."""
    try:
        import curado_hf as CH
        emb = [SALIDA / n for v in CH.PREVIA.values()
               for n in ([v] if isinstance(v, str) else v)]
    except Exception as e:
        print("   no se pudo leer `curado_hf.PREVIA` (%s): se avisa por el total de la carpeta" % e)
        emb = [f for f in SALIDA.iterdir() if f.is_file()]
    inc = sum(f.stat().st_size for f in emb if f.exists())
    faltan = [f.name for f in emb if not f.exists()]
    print("de eso, lo que el README INCRUSTA: %.1f MB en %d ficheros" % (inc / 1e6, len(emb)))
    if faltan:
        print("AVISO: `curado_hf.PREVIA` apunta a previas que NO existen: %s" % ", ".join(faltan))
    if inc > 25e6:
        print("AVISO: mas de 25 MB incrustados hace que el README tarde en abrir y no cumple")


if __name__ == "__main__":
    main()

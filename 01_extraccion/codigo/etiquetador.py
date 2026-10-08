"""El ETIQUETADOR: tres cabezas que aprenden de 6.748 clips anotados a mano lo que hay que
preguntarle a los 106.432 que nadie anoto.

## EL BUCLE, QUE ES LA IDEA ENTERA

    los clips anotados a mano
              |
              v
    [B] se entrena el ETIQUETADOR  ..................... aqui, en esta maquina
              |
              v
    se aplica a los que nadie etiqueto  ................ en Colab, como etapa 10
              |
              v
    sus etiquetas pasan a ser CAMPOS del clip, al lado
    de las poses, los eventos y la posicion
              |
              v
    eso es lo que come el modelo de videoarbitraje

Por eso esto es una **etapa mas del pipeline** y no un script aparte: corre despues de las poses y
de los eventos, porque come lo que ellos producen, y escribe `etiquetas_<clip>.csv` al lado del
resto. Y por eso **no anade coste de GPU**: come series que ya estan en disco, no el video.

## LA ENTRADA ES LO QUE YA EXTRAEMOS, NO EL VIDEO

Poses de 308 puntos con su procedencia, cajas, eventos de movimiento y posicion en la pista. Tres
razones, y ninguna es que sea lo facil: entrena en minutos en esta maquina, reutiliza toda la
tuberia en vez de duplicarla, y es **contradecible punto por punto** -- si una prediccion esta mal
se puede ir a la serie y ver que rasgo la produjo. El video crudo (VideoMAE y companyia) queda
identificado y NO hecho: sin una linea base que batir no sabriamos si aporta.

## LA ANTISIMETRIA, QUE ES LO QUE HACE HONESTA A LA CABEZA 1

Intercambiar los dos tiradores **tiene que** intercambiar la respuesta:

    f(A,B) = fotl  <=>  f(B,A) = fotr

Se impone POR CONSTRUCCION, en dos piezas:

1. **Los rasgos van en el marco propio de cada tirador.** "Adelante" es +x para el de la izquierda
   y -x para el de la derecha, asi que `velocidad_adelante`, `extension_hacia_el_rival` y
   `pie_adelantado` significan lo mismo para los dos. En ese marco, espejar el clip es EXACTAMENTE
   intercambiar las dos ranuras. Si los rasgos llevasen la x absoluta esto no seria cierto, y la
   antisimetria seria un deseo en vez de una construccion.

2. **La cabeza va sobre la DIFERENCIA de los dos embeddings, no sobre su concatenacion.**
   Con `d = E(izq) - E(der)`:

        logit_fotl    = +u(d)        u lineal:  u(-d) = -u(d)
        logit_fotr    = -u(d)
        logit_neither =  v(|d|)      par por construccion: v(|-d|) = v(|d|)

   Al intercambiar, `d -> -d`, asi que los logits de fotl y fotr se cambian y el de neither no se
   mueve. No hay umbral que ajustar ni penalizacion que sintonizar.

**Por que importa mas que la elegancia.** Sin antisimetria la red puede aprender "el de la
izquierda inicia mas" y sacar un 0,47 que parece aprendizaje y es el prior. Con ella ese atajo **no
es representable**. Y de paso cada clip espejado es otro ejemplo valido: los datos se duplican
gratis.

El control es inmediato: pasar cada clip y su espejo y comprobar que las respuestas son opuestas en
el **100 %**. Si no lo son, hay un fallo de construccion -- tipicamente un rasgo absoluto colado en
el bloque propio -- y el numero de acierto no vale.

## LA CABEZA 2 ES ORDINAL, NO CINCO CLASES SUELTAS

Confundir `middle` con `fotr_half` no cuesta lo mismo que confundirlo con `fotr_warning_box`, y una
clasificacion plana trata los dos errores igual. Aqui la salida es un escalar `z` y cuatro umbrales
simetricos. Y la simetria del espejo tambien se impone por construccion, pero **impar** en vez de
antisimetrica por ranuras: espejar la pista manda `fotl_half <-> fotr_half`, o sea `z -> -z`. Se
consigue con `z = psi(X) - psi(espejo(X))`, que es impar sea cual sea `psi`.

## LA CABEZA 3 SE ENTRENA Y SE PUBLICA SU NUMERO, PERO LLEVA UN LIMITE ESCRITO

`body_location` tiene 1.699 etiquetas para 8 clases muy desbalanceadas (`back_leg` tiene 9) y le
falta lo que mas necesitaria: **el cuadro del tocado**. El catalogo nuevo no lo trae; el
`index.jsonl` viejo si (`f_touch`). Sin el, la cabeza mira el clip entero cuando deberia mirar el
instante. Se entrena igual y su cifra va publicada con esta frase al lado, porque un numero flojo
con su causa identificada vale mas que una cabeza ausente.

## LO QUE NUNCA SE MEZCLA

`origen_etiqueta` es `humana` o `modelo`, y **no se fusionan**. Es la misma disciplina que
`campo_origen` en el cono y `origen` en las poses: una etiqueta humana y una predicha no valen lo
mismo, y el dia que se entrene el modelo final hay que poder pesarlas distinto o quedarse solo con
las humanas. Corolarios que hay que respetar: los clips anotados a mano **nunca** se reetiquetan
con el modelo, y al evaluar el modelo de videoarbitraje un clip con etiqueta predicha **no puede
entrar en el reservado** como si fuera verdad.

## EL TECHO, QUE NO ES EL OBJETIVO

Medido en `catalogo.techo_humano()`: cuando dos personas anotan el mismo clip coinciden el
**0,688** en quien inicia, el **0,619** en la zona y el **0,481** en el blanco. Un modelo que saque
0,69 en la cabeza 1 ha igualado a los humanos. Uno que saque 0,80 hay que mirarlo con desconfianza
antes que con alegria.

    python etiquetador.py --entrena --cabeza initiated
    python etiquetador.py --aplica CLIP [CLIP...]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import random
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import catalogo as CA  # noqa: E402
import rutas as RU  # noqa: E402

SEMILLA = 20260922
LARGO = 64
"""A cuantos pasos se remuestrea la serie de cada clip.

POR QUE REMUESTREAR Y QUE SE PIERDE. Los clips van de 31 a 400 y pico cuadros utiles, y una red
que reciba largos distintos hay que rellenarla y enmascararla. Remuestrear a 64 pasos por
interpolacion lineal sobre los cuadros UTILES hace el lote trivial y conserva la forma de la
accion, que es lo que se pregunta. **Lo que se pierde es el tiempo absoluto**: un fondo rapido y
uno lento quedan iguales. Por eso `duracion_s` y `fps` van como rasgos escalares aparte, para que
la red pueda recuperarlo si le hace falta."""

CABEZAS = ("initiated", "strip_location", "body_location")

CLASES = {
    "initiated": CA.INICIA,
    "strip_location": CA.ZONAS,
    "body_location": CA.BLANCOS,
}

# ---------------------------------------------------------------------------------------------
# RASGOS
# ---------------------------------------------------------------------------------------------

HOMBROS = (5, 6)
CODOS = (7, 8)
MANOS_RAIZ = (21, 42)
CADERAS = (9, 10)
RODILLAS = (11, 12)
TOBILLOS = (13, 14)
"""El mapa de indices MEDIDO en `cono_lib`, no el que traia el plan del modelo: los indices del
plan (`hand_root` 91 y 112) caen sobre la careta."""

RASGOS_PROPIOS = [
    "v_adelante", "alto_caja", "razon_ancho_alto", "ext_brazo", "ext_brazo_adelante",
    "v_mano_adelante", "sep_pies", "pie_adelantado", "altura_cadera", "flexion_rodilla",
    "presente", "inventada",
    "ev_marcha", "ev_romper", "ev_fondo", "ev_brazo", "ev_alto", "ev_bote",
]
"""LOS RASGOS DEL BLOQUE PROPIO, y el criterio que los define: **ninguno puede depender de que
lado de la imagen ocupa el tirador**. Todos los que llevan direccion van multiplicados por el signo
de ataque (`+1` el de la izquierda, `-1` el de la derecha) y todas las longitudes van divididas por
el alto de su caja, que es el unico normalizador de escala que el proyecto tiene medido.

`presente` e `inventada` no son rasgos de esgrima: son la PROCEDENCIA. Van dentro porque un tramo
de pose inventada no vale lo mismo que uno medido y la red tiene que poder distinguirlos, igual que
los distingue cualquier otro consumidor del NPZ."""

RASGOS_COMPARTIDOS = ["dist_rival", "v_cierre"]
"""Los dos que son de la PAREJA y no de un tirador. Son simetricos: `dist_rival` no cambia al
espejar, y `v_cierre` tampoco. Se pegan a los dos bloques propios."""

RASGOS_ABSOLUTOS = ["cx_izq", "cx_der", "cx_medio", "x_m_izq", "x_m_der", "tiene_metros"]
"""EL BLOQUE ABSOLUTO, solo para la cabeza 2. Aqui la x de la imagen SI entra, porque la zona de la
pista es justamente una pregunta sobre donde estan en la imagen. Al espejar, `cx -> 1 - cx` y
`x_m -> -x_m`, y de eso se encarga `espejo_abs`."""

N_PROP = len(RASGOS_PROPIOS) + len(RASGOS_COMPARTIDOS)
N_ABS = len(RASGOS_ABSOLUTOS)

ESCALARES = ["duracion_s", "fps", "frac_util", "frac_inventada", "ambos"]


def _remuestrea(a, n=LARGO):
    """QUE HACE: lleva una serie de largo T a largo n por interpolacion lineal, tolerando NaN."""
    a = np.asarray(a, dtype="float32")
    T = len(a)
    if T == 0:
        return np.zeros(n, dtype="float32")
    if T == 1:
        return np.repeat(a, n).astype("float32")
    xs = np.linspace(0, T - 1, n)
    ok = np.isfinite(a)
    if not ok.any():
        return np.zeros(n, dtype="float32")
    return np.interp(xs, np.nonzero(ok)[0], a[ok]).astype("float32")


AVISOS = {}


def _avisa(clave, texto):
    """QUE HACE: guarda el aviso y lo imprime la primera vez. Se GUARDA para poder decir al final
    cuantos clips se cayeron por cada causa, en vez de que desaparezcan en el desplazamiento."""
    AVISOS[clave] = AVISOS.get(clave, 0) + 1
    if AVISOS[clave] == 1:
        print("AVISO: %s" % texto)


def _nanmean(a, eje=0):
    """La media ignorando NaN, sin el aviso de `Mean of empty slice`. Un cuadro sin ninguna de las
    dos caderas devuelve NaN a proposito: lo rellena despues la interpolacion del remuestreo."""
    with np.errstate(all="ignore"):
        n = np.sum(np.isfinite(a), axis=eje)
        s = np.nansum(np.where(np.isfinite(a), a, 0.0), axis=eje)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _nanmax(a, eje=0):
    """QUE HACE: el maximo ignorando NaN, devolviendo NaN donde TODO era NaN, sin el aviso.

    El aviso de `All-NaN slice` no es ruido: significa que en ese cuadro no habia ninguna de las
    dos manos o ninguno de los dos pies. Se deja como NaN a proposito, que es lo que despues
    interpola `_remuestrea`, en vez de convertirlo en un cero que se leeria como "brazo recogido".
    """
    with np.errstate(all="ignore"):
        m = np.all(~np.isfinite(a), axis=eje)
        b = np.where(np.isfinite(a), a, -np.inf).max(axis=eje)
    return np.where(m, np.nan, b)


def _suave(a, k=5):
    """QUE HACE: media movil de k que IGNORA los NaN dividiendo por cuantos habia de verdad.

    POR QUE ASI Y NO RELLENANDO CON LA MEDIA. La version anterior hacia
    `nan_to_num(a, nan=nanmean(a))` y convolucionaba con `mode="same"`. Dos defectos, y el segundo
    lo cazo el control del espejo: (1) rellenar con la media inventa dato donde no lo hay, y
    (2) con `mode="same"` el nucleo queda truncado en los bordes, asi que el nivel de relleno se
    cuela con un peso distinto a cada lado. Sobre una coordenada ABSOLUTA eso rompe la simetria
    del espejo -- discrepancia relativa medida de 1,16 a 2,29 en los nueve clips de la prueba.
    Normalizando por la cuenta de validos, el filtro conmuta con cualquier transformacion afin de
    la senial y el espejo vuelve a dar exacto.
    """
    a = np.asarray(a, dtype="float64")
    if len(a) < k:
        return a.astype("float32")
    ok = np.isfinite(a).astype("float64")
    ker = np.ones(k, dtype="float64")
    s = np.convolve(np.where(np.isfinite(a), a, 0.0), ker, mode="same")
    n = np.convolve(ok, ker, mode="same")
    with np.errstate(all="ignore"):
        b = np.where(n > 0, s / np.maximum(n, 1e-9), np.nan)
    return b.astype("float32")


def _eventos_por_lado(clip, T):
    """QUE DEVUELVE: (T, 2, 6) con un pulso en el cuadro de inicio de cada evento, por lado.

    Si la etapa 7 no corrio para ese clip, devuelve ceros **y lo dice quien llama**: un clip sin
    eventos no es un clip sin accion, y confundirlos metería ceros estructurales en la media.
    """
    tipos = ["marcha", "romper", "fondo", "brazo", "alto", "bote"]
    E = np.zeros((T, 2, len(tipos)), dtype="float32")
    p = RU.PISTA / ("serie_eventos_%s.csv" % clip)
    if not p.is_file():
        return E, False
    with open(p, encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            try:
                lado = int(f["lado"])
                k = int(float(f["cuadro_inicio"]))
            except (ValueError, TypeError, KeyError):
                continue
            if lado not in (0, 1) or not (0 <= k < T) or f["tipo"] not in tipos:
                continue
            E[k, lado, tipos.index(f["tipo"])] = 1.0
    return E, True


def _x_metros(clip, T):
    """QUE DEVUELVE: (T, 2) con la x en metros de cada tirador, o NaN si el clip no tiene
    homografia. **NaN y no cero**: 6 de cada 10 clips no tienen homografia, y un cero ahi los haria
    indistinguibles de un tirador en el centro de la pista."""
    X = np.full((T, 2), np.nan, dtype="float32")
    p = RU.PISTA / ("serie_eventos_%s.csv" % clip)
    if not p.is_file():
        return X, False
    hay = False
    with open(p, encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            try:
                lado = int(f["lado"])
                k = int(float(f["cuadro_inicio"]))
                x = float(f["x_inicio_m"])
            except (ValueError, TypeError, KeyError):
                continue
            if lado in (0, 1) and 0 <= k < T and math.isfinite(x):
                X[k, lado] = x
                hay = True
    return X, hay


def rasgos_clip(clip, trabajo=None, fps=None):
    """QUE DEVUELVE: {'prop': (2, LARGO, N_PROP), 'abs': (LARGO, N_ABS), 'esc': (n,), ...} o None.

    `prop[0]` es el tirador de la IZQUIERDA y `prop[1]` el de la DERECHA, los dos en su propio
    marco de ataque. Esa es la propiedad de la que cuelga toda la antisimetria: espejar el clip es
    exactamente `prop[[1, 0]]`.
    """
    trabajo = pathlib.Path(trabajo or RU.TRABAJO)
    p = trabajo / ("poses_%s.npz" % clip)
    if not p.is_file():
        return None
    try:
        with np.load(p, allow_pickle=True) as z:
            if "KP" not in z.files:
                _avisa("esquema", "%s tiene otro esquema (%s): es de un experimento viejo, no de"
                                  " la etapa 4" % (p.name, ",".join(z.files[:4])))
                return None
            KP, SC = z["KP"], z["SC"]
            B, Be = z["B"], z["Be"]
            pres, mask, origen = z["pres"], z["mask"], z["origen"]
            W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
            ambos = float(z["ambos"]) if "ambos" in z.files else float("nan")
            fps_npz = float(z["fps"]) if "fps" in z.files else 0.0
    except Exception as ex:
        _avisa("roto", "%s no se puede leer (%s): se descarta y se CUENTA, no se ignora"
               % (p.name, type(ex).__name__))
        return None
    """POR QUE ESTO NO ES UN `try` DEFENSIVO SIN MAS. Al recorrer los que ya habia en disco
    aparecieron dos ficheros que un `poses_*.npz` no deberia ser: uno **corrupto** (no es ni un
    zip) y otro con el esquema de un experimento viejo (`kp`, `sc`, `cuadro`, `lado` en vez de
    `KP`, `SC`, ...). Los dos pasan la comprobacion barata de `lotes.hechas()`, que solo mira que
    el fichero exista y no este vacio. Que esa comprobacion sea barata esta bien -- releer 100.000
    npz para reanudar seria absurdo -- pero entonces alguien tiene que cazarlos, y es este."""
    ks = np.nonzero(mask)[0]
    if len(ks) < 8:
        return None
    fps = float(fps or 0.0) or fps_npz
    if not fps:
        pv = RU.PISTA / ("posicion_clip_%s.csv" % clip)
        if pv.is_file():
            with open(pv, encoding="utf-8") as fh:
                r = next(csv.DictReader(fh), None)
            if r and r.get("fps"):
                try:
                    fps = float(r["fps"])
                except ValueError:
                    pass
    if not fps:
        """DEL PROPIO VIDEO, SI ESTA. Los clips extraidos antes de que `fps` viajase dentro del
        NPZ no lo traen -- medido: **16 de 358** -- y con el 25 por defecto todas sus velocidades
        quedan escaladas por un factor equivocado (1,2 en un clip de 30 fps). Mientras el mp4 este
        en disco, la cadencia se puede recuperar y no hay que reextraer nada. En Colab el mp4 se
        borra, y por eso el arreglo de verdad fue que `fps` viaje en el NPZ."""
        try:
            import cv2
            ruta = RU.video(clip)
            if ruta:
                cap = cv2.VideoCapture(ruta)
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                cap.release()
                if fps:
                    _avisa("fps_video", "%s no traia cadencia en el NPZ; recuperada del mp4 "
                                        "(%.2f fps). Son los clips anteriores al arreglo." % (clip, fps))
        except Exception:
            fps = 0.0
    if not fps:
        fps = 25.0
        _avisa("fps", "%s no trae cadencia ni en el NPZ, ni en el CSV de la etapa 7, ni en un mp4 "
                      "en disco: se usa 25 y TODAS sus velocidades quedan con ese factor sin "
                      "comprobar" % clip)
    """EL ORDEN DE LA CADENA IMPORTA: primero lo que el que llama sabe, luego lo que el NPZ trae
    escrito, luego el CSV de la etapa 7, y **solo entonces** el 25 por defecto, avisando. Antes
    solo estaba el CSV y el 25, y el 25 no avisaba: un clip de 30 fps sin etapa 7 salia con todas
    las velocidades multiplicadas por 1,2 sin que nada lo dijera."""

    EV, hay_ev = _eventos_por_lado(clip, T)
    XM, hay_m = _x_metros(clip, T)

    # caja, en unidades de imagen
    cx = 0.5 * (B[:, :, 0] + B[:, :, 2])
    cy = 0.5 * (B[:, :, 1] + B[:, :, 3])
    bw = (B[:, :, 2] - B[:, :, 0])
    bh = (B[:, :, 3] - B[:, :, 1])
    bhpx = np.where(bh > 1e-6, bh * H, np.nan)
    ybajo = B[:, :, 3] * H

    signo = np.array([1.0, -1.0], dtype="float32")
    """+1 el de la izquierda, -1 el de la derecha. Es lo unico de todo el bloque propio que sabe
    en que lado esta cada uno, y su unico uso es quitarselo a los rasgos."""

    dist = np.abs(cx[:, 1] - cx[:, 0])
    hmed = _nanmean(bh, eje=1)
    dist_rel = dist / np.where(hmed > 1e-6, hmed, np.nan)

    filas = []
    for lado in (0, 1):
        s = signo[lado]
        h = bhpx[:, lado]
        kp = KP[:, lado]
        sc = SC[:, lado]

        def pt(i):
            xy = kp[:, i, :].astype("float32")
            xy[sc[:, i] < 0.3] = np.nan
            return xy

        ho = [pt(i) for i in HOMBROS]
        ma = [pt(i) for i in MANOS_RAIZ]
        ca = [pt(i) for i in CADERAS]
        ro = [pt(i) for i in RODILLAS]
        to = [pt(i) for i in TOBILLOS]

        ext = []
        extx = []
        for m, hh in ((ma[0], ho[1]), (ma[1], ho[0])):
            """La mano A cuelga del hombro 6 y la B del hombro 5; es el emparejamiento por paridad
            del `flip_pairs` del modelo, medido en `cono_lib`."""
            d = np.linalg.norm(m - hh, axis=1) / h
            ext.append(d)
            extx.append(s * (m[:, 0] - hh[:, 0]) / h)
        ext = _nanmax(np.stack(ext))
        i_larga = np.nanargmax(np.stack([np.nan_to_num(e, nan=-1) for e in extx]), axis=0)
        extx = np.stack(extx)[i_larga, np.arange(T)]
        """`i_larga` elige la mano MAS EXTENDIDA hacia el rival, no la mano A ni la B. Es lo que
        hace que el rasgo no dependa de que indice le toco a cada mano: al espejar un video, el
        modelo de pose intercambia los indices de los pares simetricos, asi que cualquier rasgo
        que dijera "la mano 21" cambiaria de mano fisica."""

        sep = np.abs(to[0][:, 0] - to[1][:, 0]) / h
        adel = _nanmax(np.stack([s * (to[0][:, 0] - cx[:, lado] * W),
                                 s * (to[1][:, 0] - cx[:, lado] * W)])) / h
        """EL PIE ADELANTADO ES EL QUE ESTA MAS HACIA EL RIVAL, y el maximo hay que tomarlo YA con
        el signo puesto. Tomarlo antes y multiplicar despues (`s * (max(x*s) - cx)`) da, para el
        tirador de la derecha, el pie de ATRAS y con el centro sumado en vez de restado: medido en
        un clip, el rango salia 4,4 a 11,2 altos de caja contra -0,05 a 0,34 del de la izquierda.
        Un rasgo que vale cosas distintas segun la ranura rompe la antisimetria por dentro, que es
        justo lo que la construccion pretende impedir."""
        cadera_y = (ybajo[:, lado] - _nanmean(np.stack([ca[0][:, 1], ca[1][:, 1]]))) / h
        rodilla_y = (ybajo[:, lado] - _nanmean(np.stack([ro[0][:, 1], ro[1][:, 1]]))) / h

        vad = s * np.gradient(_suave(cx[:, lado])) * fps
        vmano = np.gradient(_suave(extx)) * fps
        """LA VELOCIDAD DE LA MANO SE DERIVA DE LA EXTENSION, no de la x absoluta de la mano.

        La version anterior derivaba `mano_x`, que es un pixel de la imagen, y le ponia el signo
        despues. Aunque el signo salga bien, el filtro sobre una coordenada absoluta arrastra el
        origen y el control del espejo lo caza. Derivar `ext_brazo_adelante` -- que ya es
        "cuanto alcanza hacia el rival, en altos de caja" -- da lo mismo que se queria medir,
        **es invariante al espejo por construccion**, y ademas separa el brazo del cuerpo: si el
        tirador entero avanza, `mano_x` sube y esto no, que es justo la diferencia entre marchar
        y extender."""

        col = {
            "v_adelante": vad,
            "alto_caja": bh[:, lado],
            "razon_ancho_alto": bw[:, lado] / np.where(bh[:, lado] > 1e-6, bh[:, lado], np.nan),
            "ext_brazo": ext,
            "ext_brazo_adelante": extx,
            "v_mano_adelante": vmano,
            "sep_pies": sep,
            "pie_adelantado": adel,
            "altura_cadera": cadera_y,
            "flexion_rodilla": cadera_y - rodilla_y,
            "presente": pres[:, lado].astype("float32"),
            "inventada": (origen[:, lado] > 0).astype("float32"),
            "dist_rival": dist_rel,
            "v_cierre": np.gradient(_suave(dist_rel)) * fps,
        }
        for j, t in enumerate(["marcha", "romper", "fondo", "brazo", "alto", "bote"]):
            col["ev_" + t] = EV[:, lado, j]
        nombres = RASGOS_PROPIOS[:12] + RASGOS_COMPARTIDOS + RASGOS_PROPIOS[12:]
        filas.append(np.stack([_remuestrea(col[n][ks]) for n in nombres], axis=1))

    prop = np.stack(filas).astype("float32")

    ab = {
        "cx_izq": cx[:, 0], "cx_der": cx[:, 1], "cx_medio": 0.5 * (cx[:, 0] + cx[:, 1]),
        "x_m_izq": XM[:, 0], "x_m_der": XM[:, 1],
        "tiene_metros": np.full(T, 1.0 if hay_m else 0.0, dtype="float32"),
    }
    abso = np.stack([_remuestrea(ab[n][ks]) for n in RASGOS_ABSOLUTOS], axis=1).astype("float32")

    esc = np.array([
        (ks[-1] - ks[0] + 1) / fps, fps / 30.0, float(mask.mean()),
        float((origen > 0).mean()), ambos if np.isfinite(ambos) else float(pres.all(axis=1).mean()),
    ], dtype="float32")

    return {"clip": clip, "prop": np.nan_to_num(prop), "abs": np.nan_to_num(abso),
            "esc": np.nan_to_num(esc), "hay_eventos": hay_ev, "hay_metros": hay_m, "fps": fps,
            "T": T, "utiles": int(len(ks))}


def espejo_abs(A):
    """QUE HACE: espeja el bloque absoluto. `cx -> 1 - cx` y ademas IZQUIERDA <-> DERECHA;
    `x_m -> -x_m` y tambien se cambian; `tiene_metros` no se mueve."""
    B = A.copy()
    i = {n: k for k, n in enumerate(RASGOS_ABSOLUTOS)}
    B[..., i["cx_izq"]] = 1.0 - A[..., i["cx_der"]]
    B[..., i["cx_der"]] = 1.0 - A[..., i["cx_izq"]]
    B[..., i["cx_medio"]] = 1.0 - A[..., i["cx_medio"]]
    B[..., i["x_m_izq"]] = -A[..., i["x_m_der"]]
    B[..., i["x_m_der"]] = -A[..., i["x_m_izq"]]
    return B


# ---------------------------------------------------------------------------------------------
# EL MODELO
# ---------------------------------------------------------------------------------------------

def _torch():
    import torch
    return torch


def construye(n_esc=len(ESCALARES), oculto=48):
    """QUE DEVUELVE: el modulo. Del orden de 10^5 parametros, que entrena en minutos en CPU."""
    import torch
    import torch.nn as nn

    class Cod(nn.Module):
        """El codificador SIAMES. Los mismos pesos para los dos tiradores: si fueran distintos, el
        modelo podria aprender 'el de la ranura 0' y la antisimetria se caeria por ahi."""

        def __init__(self, n_in, oculto):
            super().__init__()
            self.gru = nn.GRU(n_in, oculto, batch_first=True, bidirectional=True)
            self.sal = 4 * oculto

        def forward(self, x):
            h, _ = self.gru(x)
            return torch.cat([h.mean(1), h.max(1).values], dim=1)

    class Etq(nn.Module):
        def __init__(self):
            super().__init__()
            self.cod = Cod(N_PROP, oculto)
            self.cod_abs = Cod(N_ABS, oculto)
            e = self.cod.sal
            self.esc = nn.Sequential(nn.Linear(n_esc, 16), nn.ReLU())
            self.u = nn.Sequential(nn.Linear(e + 16, 32), nn.ReLU(), nn.Linear(32, 1))
            """`u` es la parte IMPAR de la cabeza 1: va sobre la diferencia de embeddings, asi que
            al intercambiar los tiradores `d -> -d`. No hace falta que `u` sea lineal: basta con
            que se evalue sobre `d`, y el logit contrario se toma como `-u(d)` explicitamente."""
            self.v = nn.Sequential(nn.Linear(e + 16, 32), nn.ReLU(), nn.Linear(32, 1))
            """`v` es la parte PAR: se evalua sobre `|d|`, que no cambia al intercambiar."""
            self.psi = nn.Sequential(nn.Linear(self.cod_abs.sal + 16, 32), nn.ReLU(),
                                     nn.Linear(32, 1))
            self.umbral = nn.Parameter(torch.tensor([0.4, 1.2]))
            """Dos umbrales, que se despliegan en cuatro simetricos `-t2 < -t1 < t1 < t2`. La
            simetria no es una suposicion sobre la pista: es la consecuencia de que `z` sea impar,
            y sin ella el modelo podria colocar `middle` fuera del centro para ganar un punto."""
            self.blanco = nn.Sequential(nn.Linear(2 * e + 16, 48), nn.ReLU(),
                                        nn.Linear(48, len(CA.BLANCOS)))

        def _base(self, prop, esc):
            eL = self.cod(prop[:, 0])
            eR = self.cod(prop[:, 1])
            return eL, eR, self.esc(esc)

        def inicia(self, prop, esc):
            """QUE DEVUELVE: los tres logits de `fotl / fotr / neither`, antisimetricos por
            construccion."""
            eL, eR, s = self._base(prop, esc)
            d = eL - eR
            u = 0.5 * (self.u(torch.cat([d, s], 1)) - self.u(torch.cat([-d, s], 1)))
            """La parte IMPAR, impuesta y no deseada. `self.u` es un MLP cualquiera, asi que
            `u(-d)` no tiene por que valer `-u(d)`: medido el 2026-10-08, **30 de 1.339** clips del
            reservado no invertian la respuesta al espejar, y el control 2 lo marcaba como fallo de
            construccion. Tomar `(u(d) - u(-d)) / 2` hace la antisimetria exacta para cualquier
            peso, que es lo que este bloque ya decia hacer."""
            v = self.v(torch.cat([d.abs(), s], 1))
            return torch.cat([u, -u, v], dim=1)

        def zona(self, ab, esc):
            """QUE DEVUELVE: el escalar ordinal `z`, IMPAR por construccion."""
            s = self.esc(esc)
            z = self.psi(torch.cat([self.cod_abs(ab), s], 1))
            ze = self.psi(torch.cat([self.cod_abs(
                torch.as_tensor(espejo_abs(ab.detach().cpu().numpy()),
                                dtype=ab.dtype, device=ab.device)), s], 1))
            return (z - ze).squeeze(1)

        def umbrales(self):
            t = torch.cumsum(torch.nn.functional.softplus(self.umbral), 0)
            return torch.cat([-t.flip(0), t])

        def cuerpo(self, prop, esc):
            eL, eR, s = self._base(prop, esc)
            return self.blanco(torch.cat([eL + eR, (eL - eR).abs(), s], 1))
            """Simetrico a proposito: el blanco tocado no cambia al espejar el clip, asi que la
            cabeza solo puede ver la suma y el valor absoluto de la diferencia."""

    return Etq()


def zona_desde_z(z, umbrales):
    """QUE HACE: reparte el escalar ordinal en las 5 zonas, de izquierda a derecha."""
    import torch
    return torch.bucketize(z, umbrales)


def perdida_ordinal(z, y, umbrales):
    """QUE HACE: la perdida de la regresion ordinal (umbrales acumulados, 'all-threshold').

    POR QUE NO ENTROPIA CRUZADA DE 5 CLASES: porque penalizaria igual confundir `middle` con
    `fotr_half` que con `fotr_warning_box`, y esa es justo la informacion que la etiqueta lleva.
    Aqui cada umbral aporta su propio termino, asi que el error crece con la distancia ordinal.
    """
    import torch
    K = len(umbrales)
    obj = (y.unsqueeze(1) > torch.arange(K, device=y.device).unsqueeze(0)).float()
    log = z.unsqueeze(1) - umbrales.unsqueeze(0)
    return torch.nn.functional.binary_cross_entropy_with_logits(log, obj)


# ---------------------------------------------------------------------------------------------
# PARTICION
# ---------------------------------------------------------------------------------------------

def particion_por_torneo(filas, frac=0.25, semilla=SEMILLA):
    """QUE HACE: reparte en entrenamiento y reservado **por torneo**, nunca por clip.

    POR QUE, y no es una precaucion generica: los clips del mismo asalto comparten tiradores, sede
    y camara. Un reparto aleatorio por clip deja el mismo combate a los dos lados y da un numero
    inflado. El proyecto ya tiene documentado un caso de evaluar dentro de un subgrupo definido por
    el resultado y comprar +3,08 pp sin ninguna informacion.

    QUE DEVUELVE: (indices de entrenamiento, indices de reservado, diagnostico). El diagnostico
    trae cuantas PAREJAS de tiradores se quedan a los dos lados: agrupar por torneo no las separa
    todas, porque una pareja puede repetirse en dos torneos, y esa fuga residual se **mide y se
    dice** en vez de suponerla nula.
    """
    tor = {}
    for i, f in enumerate(filas):
        tor.setdefault(f["torneo"], []).append(i)
    nombres = sorted(tor)
    random.Random(semilla).shuffle(nombres)
    objetivo = frac * len(filas)
    res, n = [], 0
    for t in nombres:
        if n >= objetivo:
            break
        res.append(t)
        n += len(tor[t])
    res = set(res)
    ires = [i for i, f in enumerate(filas) if f["torneo"] in res]
    itr = [i for i, f in enumerate(filas) if f["torneo"] not in res]
    par_tr = {filas[i]["pareja"] for i in itr}
    par_re = {filas[i]["pareja"] for i in ires}
    return itr, ires, {
        "torneos_tr": len(tor) - len(res), "torneos_re": len(res),
        "n_tr": len(itr), "n_re": len(ires),
        "parejas_compartidas": len(par_tr & par_re),
        "clips_de_pareja_compartida": sum(1 for i in ires
                                          if filas[i]["pareja"] in par_tr),
    }


# ---------------------------------------------------------------------------------------------
# LINEAS BASE, DECLARADAS ANTES DE ENTRENAR
# ---------------------------------------------------------------------------------------------

def base_mayoritaria(y_tr, y_re):
    """La clase mayoritaria del ENTRENAMIENTO aplicada al reservado. Del entrenamiento y no del
    reservado: mirar el reservado para elegir la clase ya es usarlo."""
    c = np.bincount(y_tr)
    return float((np.asarray(y_re) == int(np.argmax(c))).mean()), int(np.argmax(c))


def base_brazo_primero(filas, idx):
    """LA LINEA BASE QUE IMPORTA para la cabeza 1: **quien extiende el brazo primero**.

    No es el prior, es esgrima. El evento `brazo` ya se extrae en la etapa 7, asi que esto no
    cuesta un modelo: se lee el primer `brazo` de cada lado y gana el mas temprano. Un etiquetador
    que no bata esto no esta aprendiendo esgrima, esta aprendiendo el reparto de clases.

    QUE DEVUELVE: (acierto sobre los que RESPONDE, cobertura, acierto contando los mudos como
    fallo). Los tres, porque una linea base que responde al 30 % con 0,90 y otra que responde al
    100 % con 0,62 no son comparables por el primer numero.
    """
    ok = tot = resp = 0
    for i in idx:
        f = filas[i]
        pr = f["primer_brazo"]
        tot += 1
        if pr is None:
            continue
        resp += 1
        if pr == f["y_initiated"]:
            ok += 1
    return (ok / resp if resp else float("nan"), resp / tot if tot else 0.0,
            ok / tot if tot else 0.0)


def base_x_pista(filas, idx):
    """LA LINEA BASE DE LA CABEZA 2, y a la vez **el primer control externo de la pista**.

    Se toma la x medida en metros del punto medio de los dos tiradores y se mete en las cinco zonas
    con las fronteras del reglamento (+-2 m la puesta en guardia, +-5 m el cambio de color de
    t.19). Si la cabeza 2 no bate esto, el problema es el modelo. Si lo bate por mucho, **el
    problema es la pista**, y eso es un resultado sobre la pista, no sobre el etiquetador.

    Cobertura aparte, porque la x en metros solo existe donde hay homografia.
    """
    fr = [-5.0, -2.0, 2.0, 5.0]
    ok = tot = resp = 0
    for i in idx:
        f = filas[i]
        tot += 1
        x = f["x_medio_m"]
        if x is None or not math.isfinite(x):
            continue
        resp += 1
        z = int(np.searchsorted(fr, x))
        if z == f["y_strip_location"]:
            ok += 1
    return (ok / resp if resp else float("nan"), resp / tot if tot else 0.0,
            ok / tot if tot else 0.0)


# ---------------------------------------------------------------------------------------------
# CONJUNTO
# ---------------------------------------------------------------------------------------------

def reune(clips=None, cabeza=None, trabajo=None, verboso=True):
    """QUE HACE: junta rasgos y etiquetas de los clips que TIENEN las dos cosas.

    Un clip sin `poses_<clip>.npz` no se puede usar y **no cuenta como fallo del modelo**: se
    cuenta aparte, porque mezclar "el modelo no acierta" con "el clip no esta extraido" es el error
    de denominador que este proyecto arrastra.
    """
    df = CA.manifiesto()
    if clips is None:
        m = df[list(CA.ETIQUETAS)].ne("").any(axis=1)
        clips = sorted(df.loc[m, "clip_id"])
    sub = df.loc[df["clip_id"].isin(set(clips))].set_index("clip_id")
    filas = []
    sin_rasgos = 0
    for c in sorted(set(clips)):
        if c not in sub.index:
            continue
        r = rasgos_clip(c, trabajo)
        if r is None:
            sin_rasgos += 1
            continue
        m = sub.loc[c]
        f = {"clip": c, "torneo": m["torneo"], "weapon": m["weapon"],
             "pareja": tuple(sorted((m["fotl_name"], m["fotr_name"]))),
             "touch": m["touch"], "n_anot": int(m["n_anot"]), "acuerdo": m["acuerdo"],
             "prop": r["prop"], "abs": r["abs"], "esc": r["esc"],
             "hay_eventos": r["hay_eventos"], "hay_metros": r["hay_metros"]}
        for cab in CABEZAS:
            v = m[cab]
            f["y_" + cab] = CLASES[cab].index(v) if v in CLASES[cab] else -1
        f["primer_brazo"] = _primer_brazo(c)
        f["x_medio_m"] = _x_medio(c)
        filas.append(f)
    if verboso:
        print("clips pedidos %d, con rasgos %d, sin poses extraidas %d"
              % (len(set(clips)), len(filas), sin_rasgos))
    if cabeza:
        filas = [f for f in filas if f["y_" + cabeza] >= 0]
        if verboso:
            print("   con etiqueta de %s: %d" % (cabeza, len(filas)))
    return filas


def _primer_brazo(clip):
    """QUE DEVUELVE: 0 si el de la izquierda extiende el brazo primero, 1 si el de la derecha,
    None si no hay evento `brazo` de ninguno o si los dos empiezan en el mismo cuadro."""
    p = RU.PISTA / ("serie_eventos_%s.csv" % clip)
    if not p.is_file():
        return None
    pri = {}
    with open(p, encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            if f.get("tipo") != "brazo":
                continue
            try:
                lado, k = int(f["lado"]), int(float(f["cuadro_inicio"]))
            except (ValueError, TypeError, KeyError):
                continue
            if lado in (0, 1):
                pri[lado] = min(pri.get(lado, 10 ** 9), k)
    if len(pri) == 1:
        return next(iter(pri))
    if len(pri) == 2 and pri[0] != pri[1]:
        return 0 if pri[0] < pri[1] else 1
    return None


def _x_medio(clip):
    """QUE DEVUELVE: la x media en metros del punto medio de los dos tiradores, o None."""
    p = RU.PISTA / ("serie_eventos_%s.csv" % clip)
    if not p.is_file():
        return None
    xs = []
    with open(p, encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            try:
                x = float(f["x_inicio_m"])
            except (ValueError, TypeError, KeyError):
                continue
            if math.isfinite(x):
                xs.append(x)
    return float(np.median(xs)) if xs else None


# ---------------------------------------------------------------------------------------------
# ENTRENAMIENTO Y EVALUACION
# ---------------------------------------------------------------------------------------------

def _tensores(filas, idx, cabeza, torch):
    P = torch.as_tensor(np.stack([filas[i]["prop"] for i in idx]))
    A = torch.as_tensor(np.stack([filas[i]["abs"] for i in idx]))
    S = torch.as_tensor(np.stack([filas[i]["esc"] for i in idx]))
    y = torch.as_tensor(np.array([filas[i]["y_" + cabeza] for i in idx], dtype="int64"))
    return P, A, S, y


def entrena(filas, cabeza, epocas=60, lote=32, lr=3e-3, semilla=SEMILLA, verboso=True):
    """QUE HACE: entrena UNA cabeza y devuelve el modelo y la tabla de evaluacion.

    El reparto es por torneo (`particion_por_torneo`) y el reservado NO se toca hasta el final:
    la epoca que se guarda es la ultima, no la mejor sobre el reservado, porque elegir la epoca
    mirando el reservado es usarlo para entrenar.
    """
    import torch
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    itr, ire, diag = particion_por_torneo(filas, semilla=semilla)
    if verboso:
        print("particion por torneo: %d torneos / %d clips entrenamiento, "
              "%d / %d reservado" % (diag["torneos_tr"], diag["n_tr"],
                                     diag["torneos_re"], diag["n_re"]))
        print("   parejas de tiradores a los dos lados: %d (afectan a %d clips del reservado)"
              % (diag["parejas_compartidas"], diag["clips_de_pareja_compartida"]))
    mod = construye()
    opt = torch.optim.AdamW(mod.parameters(), lr=lr, weight_decay=1e-4)
    Ptr, Atr, Str, ytr = _tensores(filas, itr, cabeza, torch)
    Pre, Are, Sre, yre = _tensores(filas, ire, cabeza, torch)
    n = len(itr)
    for ep in range(epocas):
        mod.train()
        per = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, lote):
            j = per[i:i + lote]
            opt.zero_grad()
            if cabeza == "initiated":
                lg = mod.inicia(Ptr[j], Str[j])
                p = torch.nn.functional.cross_entropy(lg, ytr[j])
                """EL ESPEJO COMO AUMENTO: intercambiar las dos ranuras es un ejemplo valido con la
                etiqueta cambiada. No es un truco de regularizacion, es la misma simetria que la
                arquitectura ya respeta, usada para duplicar los datos."""
                lge = mod.inicia(Ptr[j][:, [1, 0]], Str[j])
                ye = torch.where(ytr[j] == 0, torch.ones_like(ytr[j]),
                                 torch.where(ytr[j] == 1, torch.zeros_like(ytr[j]), ytr[j]))
                p = p + torch.nn.functional.cross_entropy(lge, ye)
            elif cabeza == "strip_location":
                z = mod.zona(Atr[j], Str[j])
                p = perdida_ordinal(z, ytr[j], mod.umbrales())
            else:
                lg = mod.cuerpo(Ptr[j], Str[j])
                p = torch.nn.functional.cross_entropy(lg, ytr[j])
            p.backward()
            torch.nn.utils.clip_grad_norm_(mod.parameters(), 2.0)
            opt.step()
            tot += float(p) * len(j)
        if verboso and (ep + 1) % 15 == 0:
            print("   epoca %3d  perdida %.4f" % (ep + 1, tot / n))
    mod.eval()
    return mod, {"itr": itr, "ire": ire, "diag": diag,
                 "tensores": (Pre, Are, Sre, yre), "cabeza": cabeza}


def predice(mod, P, A, S, cabeza):
    import torch
    with torch.no_grad():
        if cabeza == "initiated":
            return mod.inicia(P, S).argmax(1).numpy()
        if cabeza == "strip_location":
            return zona_desde_z(mod.zona(A, S), mod.umbrales()).numpy()
        return mod.cuerpo(P, S).argmax(1).numpy()


def control_espejo(mod, P, S):
    """EL CONTROL DE LA ANTISIMETRIA, que no cuesta nada y decide si el numero vale.

    Se pasa cada clip y su espejo y se comprueba que la respuesta se invierte. Tiene que dar el
    **100 %**. Si no lo da, la antisimetria no esta impuesta, esta deseada, y el acierto de la
    cabeza 1 no se puede leer -- porque entonces si cabe el atajo de "el de la izquierda inicia
    mas".
    """
    import torch
    with torch.no_grad():
        a = mod.inicia(P, S).argmax(1).numpy()
        b = mod.inicia(P[:, [1, 0]], S).argmax(1).numpy()
    esp = np.where(a == 0, 1, np.where(a == 1, 0, 2))
    return float((esp == b).mean()), int((esp != b).sum())


def matriz(y, p, clases):
    M = np.zeros((len(clases), len(clases)), dtype=int)
    for a, b in zip(y, p):
        if 0 <= a < len(clases) and 0 <= b < len(clases):
            M[a, b] += 1
    return M


# ---------------------------------------------------------------------------------------------
# LA ETAPA 10: APLICAR
# ---------------------------------------------------------------------------------------------

COLUMNAS_ETIQUETA = ["clip", "cabeza", "valor", "confianza", "origen_etiqueta", "modelo", "nota"]


def carga(cabeza):
    """QUE DEVUELVE: el modelo entrenado de esa cabeza, o None si no hay pesos.

    None y no una excepcion: en Colab la etapa 10 tiene que poder decir "no hay pesos" y seguir con
    el resto del lote, en vez de tumbar 200 clips por un fichero que falta."""
    p = RU.MODELOS / ("etiquetador_%s.pt" % cabeza)
    if not p.is_file():
        return None
    import torch
    m = construye()
    m.load_state_dict(torch.load(p, map_location="cpu"))
    m.eval()
    return m


def etiqueta_clips(clips, trabajo=None, salida=None):
    """QUE HACE: escribe `etiquetas_<clip>.csv` con lo que predice cada cabeza que tenga pesos.

    LAS DOS REGLAS QUE ESTO NO PUEDE ROMPER:

    1. **Un clip con etiqueta HUMANA no se reetiqueta.** Se escribe la humana con
       `origen_etiqueta = humana` y el modelo no la toca. Si el modelo la sobrescribiera, el dia
       de entrenar el modelo final no habria forma de saber cuales eran verdad.
    2. **`origen_etiqueta` nunca se fusiona.** Es una columna, no un matiz: quien lea el CSV tiene
       que poder quedarse solo con las humanas con un filtro de una linea.

    QUE DEVUELVE: una linea de resumen por clip, que es lo que imprime la etapa 10.
    """
    import torch
    salida = pathlib.Path(salida or RU.TRABAJO)
    mods = {c: carga(c) for c in CABEZAS}
    if not any(mods.values()):
        return []
    out = []
    for clip in clips:
        r = rasgos_clip(clip, trabajo)
        if r is None:
            out.append("%s: sin rasgos, no se etiqueta" % clip)
            continue
        hum = CA.etiquetas(clip) or {}
        filas = []
        for cab in CABEZAS:
            if cab in hum:
                filas.append({"clip": clip, "cabeza": cab, "valor": hum[cab], "confianza": "",
                              "origen_etiqueta": "humana", "modelo": "",
                              "nota": "anotacion humana; el modelo NO la sobrescribe"})
                continue
            m = mods.get(cab)
            if m is None:
                continue
            P = torch.as_tensor(r["prop"][None])
            A = torch.as_tensor(r["abs"][None])
            S = torch.as_tensor(r["esc"][None])
            with torch.no_grad():
                if cab == "initiated":
                    lg = m.inicia(P, S)
                    pr = torch.softmax(lg, 1)[0]
                    i = int(pr.argmax())
                    conf = float(pr[i])
                elif cab == "strip_location":
                    z = m.zona(A, S)
                    i = int(zona_desde_z(z, m.umbrales())[0])
                    u = m.umbrales()
                    conf = float(torch.sigmoid(
                        (z[0] - u).abs().min() * 2.0))
                    """La confianza de la cabeza ordinal es la DISTANCIA AL UMBRAL mas cercano,
                    no un maximo de softmax: en una salida ordinal no hay softmax, y lo que dice
                    cuanto fiarse es si el escalar cayo en medio de una zona o pegado al borde."""
                else:
                    lg = m.cuerpo(P, S)
                    pr = torch.softmax(lg, 1)[0]
                    i = int(pr.argmax())
                    conf = float(pr[i])
            filas.append({"clip": clip, "cabeza": cab, "valor": CLASES[cab][i],
                          "confianza": round(conf, 4), "origen_etiqueta": "modelo",
                          "modelo": "etiquetador_%s.pt" % cab,
                          "nota": "prediccion; NO vale como verdad en un reservado"})
        p = salida / ("etiquetas_%s.csv" % clip)
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNAS_ETIQUETA, extrasaction="ignore")
            w.writeheader()
            w.writerows(filas)
        out.append("%s: %d etiquetas (%d humanas, %d del modelo)"
                   % (clip, len(filas),
                      sum(1 for f in filas if f["origen_etiqueta"] == "humana"),
                      sum(1 for f in filas if f["origen_etiqueta"] == "modelo")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cabeza", default="initiated", choices=CABEZAS)
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--conjunto", default=None)
    ap.add_argument("--guarda", action="store_true")
    a = ap.parse_args()
    clips = None
    if a.conjunto and os.path.exists(a.conjunto):
        clips = [l.strip() for l in open(a.conjunto, encoding="utf-8") if l.strip()]
    filas = reune(clips, cabeza=a.cabeza)
    if len(filas) < 60:
        print("solo %d clips con rasgos Y etiqueta: no hay con que entrenar todavia." % len(filas))
        print("Falta extraer. Con `python 01_extraccion/lotes.py --conjunto disco --etapas 1,4,7`")
        return 2
    mod, ev = entrena(filas, a.cabeza, epocas=a.epocas)
    if a.guarda:
        import torch
        RU.MODELOS.mkdir(parents=True, exist_ok=True)
        torch.save(mod.state_dict(), RU.MODELOS / ("etiquetador_%s.pt" % a.cabeza))
        print("pesos -> %s" % (RU.MODELOS / ("etiquetador_%s.pt" % a.cabeza)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

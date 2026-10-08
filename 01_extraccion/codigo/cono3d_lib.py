"""ETAPA E, comun al 3D: leer la z de `pointmap`, pasarla a PIXELES, y el cono como region 3D.

POR QUE ESTE MODULO EXISTE Y QUE **NO** REIMPLEMENTA. Todo lo que ya estaba medido se importa de
`cono_lib`: el mapa de indices, `mano_armada`, `eje_antebrazo`, `escala_px_m`, `peso_sector` y
`dist_segmentos`. Aqui solo hay lo que el 3D anade, y una pieza EXTRAIDA (no reinventada) de
`e5_hoja_en_alfa.py`, con sus mismos parametros medidos, para poder llamarla desde otro paso.

--------------------------------------------------------------------------------------------------
EL PROBLEMA DE UNIDADES, QUE ES EL NUCLEO DE TODO ESTO

`pointmap` devuelve (X, Y, Z) por pixel en unidades de camara **de escala desconocida**, y ya esta
medido en el proyecto que esa escala **cambia con el recorte** (ampliar el recorte un 30 % mueve la z
mediana entre un 30 % y un 694 %). Asi que un numero de `pointmap` no se puede comparar con un pixel
tal cual, y mezclar los dos daria una geometria sin sentido.

LA CONVERSION QUE SE USA AQUI, declarada antes de medir nada. Para una camara de proyeccion central
se cumple `X = (col - cx) * Z / f`, o sea que **un pixel vale `s = Z / f` unidades** a la profundidad
Z. Ese `s` se mide del propio `pointmap` por diferencias finitas (`dX/dcol`, `dY/dfila`) alrededor
del punto que interesa. Con el, una diferencia de profundidad se pasa a pixeles:

    z_px = (Z - Z_vertice) / s0        con `s0` medido en el VERTICE del cono (la mano armada)

Y por eso la z va **relativa al vertice y no en absoluto**: `Z/s` valdria `f` para cualquier punto de
una superficie fronto-paralela, o sea una constante sin informacion. Lo que informa es la
DIFERENCIA de profundidad, y esa si se convierte bien.

TRES CONSECUENCIAS QUE HAY QUE TENER PRESENTES, y las tres se comprueban con una cifra en `e9`:

  1. Es una LINEALIZACION local. Vale mientras `|Z - Z_vertice| / Z_vertice` sea pequeno. Si el rival
     esta al doble de profundidad que la mano, el factor `s` de alli no es el de aqui y la conversion
     mete su propio error. Se mide esa razon y se informa.
  2. `f = Z / s` tiene que salir **parecido en todos los cuadros y en todos los clips**, porque es la
     focal del lienzo de entrada del modelo, que es fijo (1024x768). Si `f` baila, `pointmap` no
     esta produciendo una geometria de camara coherente y la conversion no significa nada. Es un
     control gratis y hay que hacerlo.
  3. `s` medido por columnas y por filas tiene que coincidir (pixel cuadrado). Su razon es otro
     control gratis.

VENTAJA DE TRABAJAR EN PIXELES Y NO EN UNIDADES DE `pointmap`: las coordenadas de imagen (x, y) se
quedan **medidas exactamente** (vienen de la pose, no del `pointmap`) y del `pointmap` solo se toma
prestada la tercera coordenada. Asi el error de `pointmap` no contamina las dos coordenadas que ya
estaban bien. Y ademas el cono 3D y el cono 2D quedan en las MISMAS unidades, que es lo que permite
compararlos cuadro a cuadro sin ningun factor de conversion escondido.

--------------------------------------------------------------------------------------------------
COMO SE LEE LA z EN UNA ARTICULACION

Un solo pixel es ruido, y si el punto cae un pixel fuera de la silueta la z es la del FONDO, que es
un error enorme. Se toma la **mediana en un disco** de radio `max(2, 0,03 * alto_caja)` y se devuelve
su MAD aparte, porque el `root` de la mano esta en el borde de la silueta y es el que puede quedar a
caballo de un salto de profundidad. La MAD es el aviso de que ese punto no es de fiar.
"""
from __future__ import annotations

import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cono_lib as CL

PRUEBAS = CL.PRUEBAS
SCRATCH = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude"
                       r"\C--Users-Alejandro-Desktop\187cfd83-d8d9-4c4d-bed4-f255724f6b21\scratchpad")
VIDEOS = pathlib.Path("D:/FENCINGLY_VIDEOS_RAW")
RADIO_FRAC = 0.03
CADERAS = (9, 10)
"""Superficie valida del FLORETE: el TORSO. Ni brazos, ni piernas, ni careta. Los cuatro vertices del
cuadrilatero son hombros 5-6 y caderas 9-10, todos del mapa MEDIDO en `e0_mapa_indices.py`."""


def clips_1080_del_corpus():
    """Clips que estan a la vez en el corpus etiquetado de FLORETE y con detecciones cacheadas, y
    que son de 1080p.

    LOS TRES FILTROS SON OBLIGATORIOS Y CADA UNO TIENE SU MOTIVO MEDIDO:
      corpus ..... se muestrea la LISTA `corpus_etiquetado.txt`, nunca el arbol de video. El arbol
                   junta las descargas de varios estudios y tiene espada y sable: de 12 clips
                   sacados con `os.walk` uno era de ESPADA y otro de SABLE. **La lista, en cambio,
                   esta limpia de otras armas** (se comprobo un aviso en contra y era falso: el clip
                   de sable `MellowHideousAngora` no esta en la lista, vive en
                   `scratchpad/height_study/mp4/`). Aun asi los cinco clips de este paso se
                   verificaron A OJO por el lame, que en florete cubre solo el torso.
      1080p ...... la hoja en el alfa se detecta en el 42,6 % de los recortes a 1080p y en el
                   **1,6 %** a 720p, donde el fuerte mide 0,64 px. A 720p no hay hoja que medir.
                   Y ESTO ES CARO: el corpus es **81,2 % de 720p, 18,0 % de 1080p** y 0,8 % de 480p
                   (medido sobre 400 clips), o sea que de los 8.559 clips solo unos **1.540**
                   admiten un cono con la hoja VISTA. Cualquier resultado de este paso vale sobre
                   esa sexta parte del corpus y hay que decirlo cada vez.
      det pkl .... la deteccion ya esta hecha y cuesta 0 s; ademas los dos interpretes del proyecto
                   NO coexisten y `ultralytics` no esta en el que tiene `transformers`.
    """
    corpus = set(x.strip() for x in
                 (SCRATCH / "corpus_etiquetado.txt").read_text(encoding="utf-8").split("\n")
                 if x.strip())
    out = []
    for f in sorted(SCRATCH.glob("det_*.pkl")):
        cid = f.name[4:-4]
        if cid not in corpus:
            continue
        with open(f, "rb") as fh:
            _c, _k, W, H, T = pickle.load(fh)
        if int(H) >= 1000:
            out.append((cid, int(W), int(H), int(T)))
    return out


def localiza(cid, avisa=True):
    """Ruta del mp4 de un clip. **Y AVISA SI EL NOMBRE ESTA REPETIDO EN EL ARBOL.**

    POR QUE EL AVISO NO ES PARANOIA. `D:/FENCINGLY_VIDEOS_RAW` junta las descargas de VARIOS
    estudios: ahi dentro hay espada, sable y clips que no son del corpus etiquetado (por ejemplo
    `scratchpad/height_study/mp4/`). Muestrear el arbol con `os.walk` es la trampa que este proyecto
    ya tiene documentada: de 12 clips sacados asi, uno era de ESPADA y otro de SABLE.

    Aqui el `os.walk` NO muestrea, solo resuelve una ruta para un identificador que **ya viene de
    `corpus_etiquetado.txt`**, asi que el arma esta garantizada por la lista. Lo que si podria fallar
    es que el mismo nombre exista en dos estudios y se devuelva el de otro estudio sin que nadie lo
    note. Se devuelven todas las coincidencias, se avisa si hay mas de una y se prefiere la que NO
    este bajo un directorio de otro estudio.
    """
    import os
    hits = []
    for raiz, _d, fs in os.walk(str(VIDEOS)):
        if "%s.mp4" % cid in fs:
            hits.append(os.path.join(raiz, "%s.mp4" % cid))
    if not hits:
        return None
    if len(hits) > 1:
        limpios = [h for h in hits if "height_study" not in h and "scratchpad" not in h]
        if avisa:
            print("  AVISO: %s aparece %d veces en el arbol de video; me quedo con %s" % (
                cid, len(hits), (limpios or hits)[0]))
        return (limpios or hits)[0]
    return hits[0]


def carga_det(cid):
    with open(SCRATCH / ("det_%s.pkl" % cid), "rb") as fh:
        cajas, cuadros, W, H, T = pickle.load(fh)
    return np.asarray(cajas, "float32"), np.asarray(cuadros, "int32"), int(W), int(H), int(T)


def z_en_punto(pm, x, y, r):
    """Mediana de (X, Y, Z) en un disco cuadrado de radio `r`, y la MAD de la Z como aviso de borde.

    Devuelve `(None, nan)` si el punto cae fuera del mapa o si quedan menos de 3 pixeles finitos.
    """
    hh, ww = pm.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < ww and 0 <= yi < hh):
        return None, float("nan")
    x0, x1 = max(0, xi - r), min(ww, xi + r + 1)
    y0, y1 = max(0, yi - r), min(hh, yi + r + 1)
    par = pm[y0:y1, x0:x1].reshape(-1, pm.shape[2])
    par = par[np.isfinite(par).all(axis=1)]
    if len(par) < 3:
        return None, float("nan")
    m = np.median(par, axis=0)
    mad = float(np.median(np.abs(par[:, 2] - m[2])))
    return m[:3].astype("float64"), mad


def escala_local(pm, x, y, r):
    """Unidades de `pointmap` por PIXEL alrededor de (x, y), por diferencias finitas robustas.

    `s_col` sale de `dX/dcol` y `s_fila` de `dY/dfila`. Se usa la MEDIANA del valor absoluto porque
    sobre un cuerpo hay saltos de profundidad y una media los recogeria enteros.

    Devuelve `(s, s_col, s_fila)` con `s` la media de los dos. `s` no puede ser <= 0.
    """
    hh, ww = pm.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - r), min(ww, xi + r + 2)
    y0, y1 = max(0, yi - r), min(hh, yi + r + 2)
    par = pm[y0:y1, x0:x1]
    if par.shape[0] < 3 or par.shape[1] < 3:
        return float("nan"), float("nan"), float("nan")
    dx = np.abs(np.diff(par[:, :, 0], axis=1))
    dy = np.abs(np.diff(par[:, :, 1], axis=0))
    dx, dy = dx[np.isfinite(dx)], dy[np.isfinite(dy)]
    if dx.size < 4 or dy.size < 4:
        return float("nan"), float("nan"), float("nan")
    sc, sf = float(np.median(dx)), float(np.median(dy))
    if not (sc > 0 and sf > 0):
        return float("nan"), sc, sf
    return 0.5 * (sc + sf), sc, sf


def lee_articulaciones(pm, kp, sc, lado, idxs, alto_caja):
    """Lee (X, Y, Z), MAD y escala local en una lista de indices de pose de UN tirador.

    Devuelve un dict `idx -> {"P": (3,), "mad": float, "s": float, "xy": (2,)}` solo con los que
    tienen confianza suficiente Y coordenada finita Y lectura de `pointmap` valida.
    """
    r = int(max(2, RADIO_FRAC * alto_caja))
    out = {}
    for i in idxs:
        p = kp[lado, i]
        if not (np.isfinite(p).all() and sc[lado, i] >= CL.CONF_MIN):
            continue
        P, mad = z_en_punto(pm, p[0], p[1], r)
        if P is None:
            continue
        s, s_col, s_fila = escala_local(pm, p[0], p[1], r)
        out[int(i)] = {"P": P, "mad": mad, "s": s, "s_col": s_col, "s_fila": s_fila,
                       "xy": p.astype("float64")}
    return out


def a_pixeles(lect, i, Z0, s0):
    """Punto de la articulacion `i` en el espacio HIBRIDO (x_px, y_px, z_px) del cono.

    x, y salen de la POSE (medidos exactos). z sale de `pointmap` y se convierte con la escala del
    VERTICE, no con la propia: el cono es una region alrededor de su vertice y todas sus coordenadas
    tienen que estar en la misma vara de medir.
    """
    if i not in lect:
        return None
    d = lect[i]
    return np.array([d["xy"][0], d["xy"][1], (d["P"][2] - Z0) / s0], dtype="float64")


def p3d_exacto(u, v, Z, W, H, f, Zref):
    """Punto 3D **sin linealizar**, en pixeles a la profundidad de referencia `Zref`.

    POR QUE HACE FALTA ESTA SEGUNDA VERSION Y NO BASTA `a_pixeles`. `a_pixeles` convierte con la
    escala del vertice, y eso vale mientras todo lo que se convierte este a la MISMA profundidad que
    el vertice, o sea dentro de un mismo tirador. Pero el cono 3D tiene que alcanzar al **rival**, que
    esta a otra profundidad, y alli la escala `s = Z/f` ya no es la del vertice. Convertir al rival
    con la escala de la mano propia meteria un error del orden de `Z_rival/Z_vertice - 1`, que es
    justo la cantidad que este paso quiere medir. Seria circular.

    LA FORMULA, que es la proyeccion central exacta. Con `f` la focal en pixeles y el punto principal
    en el centro de la imagen, un pixel (u, v) a profundidad Z corresponde al punto de camara
    `X = (u - W/2) * Z / f`, `Y = (v - H/2) * Z / f`, `Z`. Dividiendo los tres por `s_ref = Zref / f`
    queda, con `r = Z / Zref`:

        P = ( (u - W/2) * r ,  (v - H/2) * r ,  f * r )

    y esas tres coordenadas estan en **pixeles a la profundidad de referencia**, o sea en las mismas
    unidades que la geometria 2D que ya existe. Por eso el cono 2D y el cono 3D se pueden comparar sin
    ningun factor escondido.

    DEGRADACION LIMPIA, que es lo que la hace honrada: si la z no llevara informacion y saliera
    constante (`Z = Zref` en todas partes), entonces `r = 1` y `P = (u - W/2, v - H/2, f)`, o sea el
    pixel de siempre mas una constante. Los angulos del cono 3D coincidirian exactamente con los del
    2D y la comparacion daria cero discrepancias. **O sea que cualquier discrepancia que aparezca
    viene de la z y de nada mas.**

    `f` sale medido de `escala_local` como `mediana(|Z| / s)`, y su constancia es el control C4 de
    `e9`. El punto principal se supone en el centro de la imagen: es la suposicion estandar y su
    error mueve el vertice unos pocos pixeles, no la profundidad.
    """
    if not (np.isfinite(Z) and np.isfinite(Zref) and np.isfinite(f)) or abs(Zref) < 1e-12:
        return None
    r = Z / Zref
    if not np.isfinite(r) or r <= 0:
        return None
    return np.array([(u - 0.5 * W) * r, (v - 0.5 * H) * r, f * r], dtype="float64")


def dir3d(a, b):
    """Direccion unitaria de `a` a `b`, o None si el segmento es degenerado."""
    if a is None or b is None:
        return None
    v = b - a
    n = float(np.linalg.norm(v))
    return None if n < 1e-9 else v / n


def ang3d(u, v):
    """Angulo entre dos direcciones 3D, en grados."""
    c = float(np.dot(u, v)) / max(float(np.linalg.norm(u) * np.linalg.norm(v)), 1e-12)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def elevacion(u):
    """Angulo que el vector saca del plano de la imagen, en grados. Es LO UNICO que el 3D anade.

    Positivo = se aleja de la camara si la Z de `pointmap` crece con la distancia. El signo se toma
    tal cual lo da el modelo y no se corrige: corregirlo a mano seria meter una suposicion.
    """
    return float(np.degrees(np.arctan2(u[2], max(float(np.linalg.norm(u[:2])), 1e-12))))


def peso_sector3d(P, vert, u, L, apert):
    """Version 3D de `cono_lib.peso_sector`, con **la misma forma declarada** `1 - angulo/apertura`.

    Que sea la misma forma no es un detalle de estilo: es lo que permite que la comparacion 2D contra
    3D sea de una sola variable. Si el 3D usara otra forma de peso, cualquier diferencia entre los
    dos podria venir de la forma y no de la profundidad.

    `P` es (N, 3). Devuelve (N,) con 0 fuera del cono.
    """
    v = P - vert[None, :]
    r = np.linalg.norm(v, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cs = (v @ u) / np.maximum(r, 1e-9)
    a = np.degrees(np.arccos(np.clip(cs, -1, 1)))
    return np.where((r <= L) & (a <= apert) & (r > 1e-6), 1.0 - a / max(apert, 1e-9), 0.0)


def malla_torso(p_hi, p_hd, p_ci, p_cd, n=7):
    """Muestrea el cuadrilatero del torso (hombros y caderas) con una rejilla bilineal de n x n.

    Se muestrea y no se resuelve analiticamente por el mismo motivo que `cono_lib.cruce` rasteriza:
    la interseccion de un cono con un cuadrilatero alabeado en 3D tiene demasiados casos y un error
    de caso se veria como senal. n=7 da 49 puntos, y el paso resultante (un septimo de la anchura de
    hombros, o sea unos 6 cm en un torso real) es mas fino que la incertidumbre del eje.
    """
    t = np.linspace(0.0, 1.0, n)
    a = p_hi[None, :] * (1 - t)[:, None] + p_hd[None, :] * t[:, None]
    b = p_ci[None, :] * (1 - t)[:, None] + p_cd[None, :] * t[:, None]
    return (a[None, :, :] * (1 - t)[:, None, None]
            + b[None, :, :] * t[:, None, None]).reshape(-1, a.shape[1])


def carga_z(clip, suf=""):
    """Deshace el NPZ plano que deja `e9` en `{(k, lado): {idx: dict}}` mas los escalares del clip.

    El NPZ se guarda PLANO (una fila por lectura) y no como matriz densa (T, 2, 308, ...) porque de
    308 puntos solo se leen 8 y la matriz densa seria un 97 % de NaN.
    """
    with np.load(PRUEBAS / ("cono_z3d_%s%s.npz" % (clip, suf)), allow_pickle=True) as z:
        k, lado, dato = z["k"], z["lado"], z["dato"]
        ks, mano, pxm = z["ks"], z["mano"], z["pxm"]
        W, H, video = int(z["W"]), int(z["H"]), str(z["video"])
    lect = {}
    for j in range(len(k)):
        i, X, Y, Z, mad, s, xp, yp = dato[j]
        lect.setdefault((int(k[j]), int(lado[j])), {})[int(i)] = {
            "X": float(X), "Y": float(Y), "Z": float(Z), "mad": float(mad), "s": float(s),
            "x": float(xp), "y": float(yp)}
    return (lect, [int(v) for v in ks], {0: str(mano[0]), 1: str(mano[1])},
            {0: float(pxm[0]), 1: float(pxm[1])}, W, H, video)


def focal(lect, W=None, H=None):
    """LA FOCAL DE `pointmap`, POR TRES CAMINOS INDEPENDIENTES, y su dispersion. Control C4.

    `pointmap` corre siempre en un lienzo FIJO de 1024x768, asi que su focal equivalente en pixeles
    de la imagen original es **un solo numero por clip**. Y si el mapa (X, Y, Z) que devuelve fuera
    la nube de una camara de proyeccion central, los tres estimadores de abajo tendrian que dar el
    MISMO numero y cada uno con poca dispersion:

        f desde X ....  f = (col - W/2) * Z / X          de la ecuacion de proyeccion en horizontal
        f desde Y ....  f = (fila - H/2) * Z / Y         la misma en vertical
        f desde s ....  f = |Z| / s                      con `s` de `escala_local`

    POR QUE SE MIDEN LOS TRES Y NO UNO. El primer intento uso solo `f = |Z|/s` y dio una dispersion
    de **1,42**, o sea que el rango intercuartilico era mayor que la propia mediana. Con un solo
    estimador no se puede saber si eso es culpa del estimador o del modelo. Con tres si: si los tres
    discrepan entre ellos, es el modelo.

    Devuelve `(f, dispersion)` del mejor de los tres (el de X, que es el de baseline mas largo porque
    la escena es mas ancha que alta) y deja los tres en `detalle` para poder contradecirlo.
    """
    def est(num, den):
        v = np.array([n / d for n, d in zip(num, den) if abs(d) > 1e-4 and np.isfinite(n)
                      and np.isfinite(d)])
        v = v[np.isfinite(v) & (v > 0)]
        if v.size < 20:
            return float("nan"), float("nan"), 0
        m = float(np.median(v))
        return m, float((np.percentile(v, 75) - np.percentile(v, 25)) / max(m, 1e-9)), int(v.size)
    todo = [d for mm in lect.values() for d in mm.values()]
    detalle = {}
    if W is not None and H is not None:
        detalle["desde_X"] = est([(d["x"] - 0.5 * W) * d["Z"] for d in todo],
                                 [d["X"] for d in todo])
        detalle["desde_Y"] = est([(d["y"] - 0.5 * H) * d["Z"] for d in todo],
                                 [d["Y"] for d in todo])
    detalle["desde_s"] = est([abs(d["Z"]) for d in todo],
                             [d["s"] if d["s"] > 0 else float("nan") for d in todo])
    neg = [d["Z"] for d in todo if np.isfinite(d["Z"])]
    detalle["frac_Z_negativa"] = (float(np.mean(np.asarray(neg) < 0)), float("nan"), len(neg))
    """LA Z NEGATIVA ES UN AVISO DURO Y VA CONTADA: una profundidad de un punto que esta DELANTE de
    la camara no puede ser negativa. Si sale negativa, el punto no es una posicion fisica."""
    mejor = detalle.get("desde_X", detalle["desde_s"])
    return mejor[0], mejor[1], detalle


N_MALLA = 7
ALCANCE_M = 1.10
"""ALCANCE DE LA MUNECA A LA PUNTA, en metros. Es la cota reglamentaria por lo ALTO, y esa eleccion
NO es cosmetica: decide si el cono puede perder amenazas.

EL DEFECTO QUE ARREGLA. Todo el proyecto venia pintando y midiendo `L = CL.HOJA_M * pxm` con
`HOJA_M = 0,90`, o sea **la hoja sola**. Pero los 90 cm de m.8.4 se miden desde la cara delantera de
la guarda (linea B), y el punto del que sale el cono es la RAIZ DE LA MANO, que es la muneca: esta
por detras de la guarda, con la empunadura en medio. Asi que 90 cm es el alcance que tendria un
esgrimista con la muneca pegada a la cazoleta, que es anatomicamente imposible.

LA COTA, de fuente primaria y con dos articulos:
    m.7    arma montada, 110 cm como maximo   ->  la muneca NO puede estar detras del pomo
    m.8.4  hoja, 90 cm como maximo            ->  la muneca NO puede estar delante de la guarda
Luego el alcance muneca-punta cae en el intervalo **[90, 110] cm** y no hay que estimar nada: los dos
extremos son reglamentarios.

POR QUE SE TOMA EL EXTREMO ALTO Y NO EL BAJO. El cono es una region que tiene que **contener** la
amenaza; su propiedad estructural (22.7) es que puede sobrarle amenazas pero no perderlas. Una cota
por lo BAJO del alcance rompe exactamente esa propiedad: con L corto, la punta llega mas lejos que el
cono y la amenaza se pierde en silencio. La nota de `CL.HOJA_M` decia que 90 cm era «conservador para
decir amenaza», y eso es cierto para NO INVENTAR una amenaza, pero es lo contrario de conservador
para no PERDERLA. Son dos errores opuestos y el cono solo puede protegerse de uno: se elige no
perder, que es lo que el proyecto ya habia declarado en 22.7.

`CL.HOJA_M` SE QUEDA COMO ESTA, en 0,90, y no se toca: lo usan `e3` a `e19` y cambiarlo moveria en
silencio toda cifra en centimetros ya publicada. Esta constante es nueva y va al lado, igual que
`eje_brazo_2d` se anadio al lado de `eje_antebrazo` en vez de reescribirla."""
APERT_E6_POOL = 27.3
"""APERTURA DE RESERVA, en grados y **SEMIANGULO** (o sea 54,6 gr de cono completo), para los clips
donde la hoja NO se ve NI UNA VEZ. Misma convencion que `APERT_ENVOLVENTE_SEMI_GR`, ver alli.

Es el punto medio de los **24,8-29,7 grados** que `e6` midio como p90 de |angulo hoja - eje| en los
dos clips de 1080p donde si habia hoja. O sea: es el error medido de PREDECIR la hoja desde el brazo,
tomado de OTROS clips.

POR QUE HACE FALTA Y NO ES UN NUMERO INVENTADO. Sin ella, un clip sin ninguna deteccion de hoja se
queda sin apertura y **todos** sus conos se caen. Eso importa mucho porque es el caso NORMAL: a 720p
la hoja se ve en el 1,6 % de los recortes y el corpus es 81,2 % de 720p. Un cono que no se construye
no es «prudencia»: es un hueco silencioso que se reparte de forma desigual entre clips.

Y VA MARCADA APARTE, con `apert_origen = 2`, precisamente porque **no se midio en este clip**. Las
tres procedencias de la apertura son distintas y no se pueden confundir:
    0  el temblor MEDIDO de la hoja de este clip        (la mas estrecha y la unica local)
    1  la regla de `e6` aplicada a este clip
    2  la regla de `e6` traida de OTROS clips           (esta constante)
"""
EJE_POR_DEFECTO = "codo"
"""QUE ARTICULACION ANCLA EL EJE DEL BRAZO **PARA ORIENTAR**. Corregido a `codo` el 2026-09-11.

LA REGLA GENERAL, que es lo que de verdad hay que retener: **el LARGO de un segmento y su DIRECCION
contestan preguntas distintas, y el mejor segmento para una puede ser el peor para la otra.**

    CUANTO se extiende el brazo   ->  `hombro -> raiz`, que crece al extenderse (cv 0,226 del largo
                                      proyectado, contra 0,108 del hueso). Es `ext_hr`, y se queda.
    HACIA DONDE apunta la hoja    ->  `codo -> raiz`, el ANTEBRAZO, porque **la hoja continua el
                                      antebrazo**, no el brazo entero.

Esta constante solo gobierna lo SEGUNDO. La version anterior ponia aqui `hombro` razonando con la
tabla de cv, o sea decidiendo una pregunta de direccion con una medida de magnitud. El razonamiento
sobre el cv sigue siendo correcto para lo que media; lo que estaba mal era aplicarlo aqui.

Y LA DIRECCION NO SE DECIDE POR ARGUMENTO, SE MIDE. `e20` carea los dos ejes contra las **31 rectas
de hoja detectadas en el alfa** que `e5` y `e13` dejaron en cache (las mismas 6 de `e14` mas 25 que
`e14` no habia usado), con el azar declarado en 45 grados y el margen declarado en 5:

    |hoja - eje|        mediana    p90     contiene el 90 % con
    `codo -> raiz`      14,4 gr    33,0    **33,0 gr**
    `hombro -> raiz`    44,8 gr    66,3    66,3 gr
    azar declarado      45,0 gr     --     --

El hombro sale **en el azar**: con ese eje la direccion de la hoja no se predice en absoluto. Y los
tres filtros de seleccion probados (sin filtro, delante segun el codo, delante segun el hombro) dan
el mismo ganador, asi que no lo decide el filtro.

LA SEGUNDA MEDIDA, INDEPENDIENTE Y SIN NECESIDAD DE VER LA HOJA: con `hombro -> raiz` el candidato de
contacto (el `peso_max` de `cruce`) cae **por debajo de las dos manos en el 63,6 %** de los casos
(336 de 528, 7 clips), porque en guardia la mano va por debajo del hombro y ese eje apunta al SUELO.
Con `codo -> raiz` baja al **15,7 %**. La punta del eje cae por debajo del tobillo propio en el
19,4 % contra el 11,6 %, y la elevacion mediana del eje pasa de **-21,3 a +4,8 grados**.

POR QUE HACIAN FALTA LAS DOS MEDIDAS Y NO BASTA UNA: el angulo contra la hoja se mide **modulo 180
grados**, porque un segmento detectado no tiene sentido. Asi que un eje que apunte al suelo en vez de
al rival da EXACTAMENTE el mismo numero que su opuesto: el defecto de «el cono se va al suelo» es
invisible para el careo con la hoja, por construccion.

`eje_antebrazo` de `cono_lib` (que ancla en el codo) y esta funcion con `cual="hombro"` siguen
existiendo las dos, para que toda medida vieja se pueda reproducir tal como se hizo."""
APERT_NUCLEO_SEMI_GR = 14.4
APERT_ENVOLVENTE_SEMI_GR = 33.0
"""LAS DOS APERTURAS DEL CONO, en grados, y NO son dos opciones de dibujo: son dos percentiles de la
MISMA distribucion medida de |hoja detectada - eje del antebrazo| (`e20`, n=31, 5 clips a 1080p).

EL CONVENIO VA EN EL NOMBRE Y NO ES NEGOCIABLE: **estos numeros son SEMIANGULOS**, o sea el angulo
medido DESDE EL EJE. El cono completo abre el doble:

    NUCLEO      +-14,4 gr  ->  **28,8 gr totales**
    ENVOLVENTE  +-33,0 gr  ->  **66,0 gr totales**

POR QUE SE DECLARA ASI Y NO SE DEJA SOBREENTENDIDO. Este proyecto tiene documentado un dia con cinco
cifras contradictorias, todas por un convenio sin declarar. Y aqui el riesgo es real: mirando una
figura, la envolvente se ve de 66 grados y la constante dice 33. Las dos cosas son verdad.

EL CONVENIO ESTA COMPROBADO EN CODIGO, no supuesto (`e22`, control A, que falla si cambia):
  - `cono_lib.peso_sector` filtra con `a <= apert`, donde `a = arccos((P-vert).u / |P-vert|)` es el
    angulo desde el eje. Un punto a 32 gr pesa 0,0303 y uno a 33 gr pesa **0,0000**.
  - `sector()` y `dibuja_cono()` barren `linspace(-apert, +apert)`, o sea 66,0 gr de arco.
  - `e6` calcula su techo de utilidad como `arctan(0,5 * ancho_hombros / L)`, con el MEDIO ancho:
    tambien semiangulo.
  - y el p90 se midio sobre `|angulo hoja - angulo eje|`, que es tambien un angulo desde el eje.
Las cuatro piezas usan la misma convencion, asi que la cadena medida-dibujo es coherente. Lo que
faltaba era decirlo.

    NUCLEO      p50 = 14,4 gr   contiene el 51,6 % de las hojas medidas
    ENVOLVENTE  p90 = 33,0 gr   contiene el 90,3 % de las hojas medidas

LA QUE DECIDE LA AMENAZA ES LA ENVOLVENTE. El nucleo es para el dibujo y para el peso: sirve para que
la figura se vea estrecha sin recortar la region, porque el relleno vale `1 - angulo/apertura` y por
tanto se apaga solo hacia el borde.

Y AQUI ESTA EL AVISO QUE HAY QUE LEER ANTES DE ESTRECHAR NADA. La apertura de 23,1 grados que `e15`,
`e18` y `e19` venian usando NO era el p90 de nada: era la **mediana** de una muestra de n=6 con el
eje del hombro, tomada de `e14`. Medida contra las hojas de verdad, aquel cono contenia el **22,6 %**
y perdia el 77,4 %. O sea que el cono del que el usuario dijo «es demasiado gordo» estaba en realidad
**demasiado ESTRECHO para su propio eje**. Lo que permite estrechar de verdad no es recortar el
dibujo: es que el error del eje baje, y por eso primero se cambio el eje.

    mismo eje, mas apertura   ->  no mejora nada: solo tapa
    mejor eje, misma apertura ->  de contener el 22,6 % a contener el 74,2 % con los mismos 23,1 gr
    mejor eje, p90 honrado    ->  de necesitar 66,3 gr a necesitar 33,0: la apertura baja **2,0x**

`APERT_E6_POOL` (27,3) se queda como esta: era el p90 por lado de UN clip, y sirve para reproducir lo
que se publico. La envolvente de 33,0 es el p90 agrupado de cinco clips y es mas ancha porque incluye
las detecciones de la ventana de accion, que llevan desenfoque de movimiento."""


APERT_HOJA_VISTA_SEMI_GR = 8.6
TEMBLOR_HOJA_SEMI_GR = 4.2
"""LA ANCHURA DEL CAMPO CUANDO LA HOJA SE HA VISTO, y por que NO es el temblor de 4,2 gr.

Las dos son medidas reales y miden cosas distintas. Confundirlas fue un error mio del 2026-09-11 y
lo cazo el control `verifica_cono_campo.py`, que es reproducible:

    TEMBLOR_HOJA_SEMI_GR = 4,2   incertidumbre de la DIRECCION de la hoja vista: el p68 del cambio
                                 de angulo entre cuadros consecutivos, 6,0 gr, dividido por raiz de
                                 2 (`e27`, 70 pares de 12 asaltos). Dice lo bien determinada que
                                 esta la direccion, y **sigue publicada** porque es la cifra que
                                 justifica centrar el campo en la hoja.

    APERT_HOJA_VISTA_SEMI_GR     la anchura que hay que DIBUJAR, que es otra cosa, porque la cunia
    = 8,6                        tiene el vertice en la **raiz de la mano** y la recta detectada
                                 **no pasa por ella**: pasa a 7 px de mediana (p90 15, max 37) y el
                                 segmento esta a solo `r/L = 0,24` del alcance. Unos pocos pixeles
                                 tan cerca del vertice subtienden muchos grados.

MEDIDO, sobre los 60 cuadros con hoja vista de los tres clips que se publican, la desviacion de la
hoja respecto de su propio centro **vista desde el vertice**:

    extremo LEJOS (la punta)    p50  3,0   p90   8,6 gr   <- de aqui sale la constante
    extremo CERCA de la mano    p50 13,2   p90  35,0 gr   <- el artefacto del vertice, no un error

    cuadros en que el cono dibujado contiene LA PUNTA de la hoja:
        +- 4,2 gr  ->  39 de 60  (0,65)   <- con esto la hoja verde se sale del cono que la centra
        +- 8,6 gr  ->  54 de 60  (0,90)   <- el p90, por definicion
        +-33,0 gr  ->  57 de 60  (0,95)

LA CONSTRUCCION ES LA MISMA QUE LA DE `APERT_ENVOLVENTE_SEMI_GR`, y eso es lo que la hace citable al
lado: un **p90 de un residuo medido usado como semiangulo**, en su propia muestra. Los 33,0 son el
p90 de |hoja - antebrazo| con el centro en el antebrazo; los 8,6 son el p90 del residuo que queda
cuando el centro se pone en la hoja. Centro y anchura son una sola decision.

Y LA RAZON QUE SE PUBLICA ES **3,8x**, no 7,9x: 33,0 / 8,6. El 7,9x era 33,0 / 4,2, o sea comparar
una anchura de dibujo con una incertidumbre de direccion. Ver la hoja sigue estrechando el campo casi
cuatro veces, que es el resultado; lo que se retira es el factor exacto.

AVISO DE MUESTRA, del mismo tipo que el de los 33,0: 8,6 es el p90 de **60 cuadros de 3 clips**, y
esos son los mismos cuadros que se dibujan, o sea que es **dentro de muestra**. Es el convenio que ya
tenian los 33,0 (p90 de sus propias 31 hojas) y se declara igual."""


PERFIL_MEDIDO_BORDES = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0]
PERFIL_MEDIDO_VALOR = [1.00, 0.44, 0.44, 0.67, 0.44, 0.22]
"""LA FORMA MEDIDA DEL CAMPO DE LA HOJA DENTRO DEL CONO, por tramos de |desvio| en grados.

Es la idea del usuario: *"el cono puede estar por dentro con puntos del 0 al 1, cuanto mas cerca de 1
mas probabilidad de ser florete"*, y con la probabilidad **estimada de las hojas que si se han visto**
en vez de postulada. Medido en `e23` sobre las 31 rectas de hoja cacheadas: cuentas por tramo de 6
grados 9, 4, 4, 6, 4, 2, normalizadas al maximo.

Y LO QUE SUSTITUIRIA: el relleno de hoy es `peso_sector = 1 - angulo/apertura`, que sobre estos
mismos tramos valdria 0,91 · 0,73 · 0,55 · 0,36 · 0,18 · 0,00. El error medio absoluto entre las dos
formas es **0,21 sobre una escala de 0 a 1**, o sea que **no son la misma forma**: la medida tiene un
pico pegado al eje y luego es casi plana, mientras la lineal es una rampa.

PERO NO SE ADOPTA COMO FORMA OPERATIVA, y la razon es la n. Con 31 hojas los tramos llevan de 2 a 9
cuentas, o sea un ruido de Poisson de +-1,4 a +-3, y **el unico rasgo que sobrevive a ese ruido es el
pico del primer tramo** (9 contra una media de 4 en los otros cinco). La planicie de los tramos 2 a 6
es indistinguible de una rampa dentro del ruido. Se guarda como DATO y se pinta, para que se pueda
ver y comparar, y la decision de amenaza sigue con `peso_sector` hasta tener la n que hace falta:
`e23` la calcula en **~226 hojas vistas**, unos 531 recortes de 1080p fuera de la ventana de accion,
o sea del orden de 9 clips. Esta identificado y no hecho."""


def campo_hoja(P, vert, mu_gr, L, apert_semi_gr, perfil=None):
    """Valor 0..1 del campo de la hoja en los puntos P. 1 = lo mas probable que se ha medido.

    `mu_gr` es la direccion central del campo en grados de imagen, y ahi esta la diferencia con
    `peso_sector`: cuando la hoja se ha VISTO en ese cuadro, `mu_gr` es el angulo MEDIDO de la hoja,
    no el del eje del brazo. O sea que el campo se centra en el dato cuando hay dato.

    `perfil` por defecto es la forma MEDIDA (`PERFIL_MEDIDO_*`). Pasando `perfil="lineal"` se obtiene
    exactamente `1 - |desvio|/apert`, que es lo que el proyecto usa hoy, para poder pintar las dos y
    compararlas sin dos implementaciones distintas.

    NO SE TOCA `cono_lib.peso_sector`, que es de quien dependen todas las cifras publicadas.
    """
    v = np.asarray(P, "float64") - np.asarray(vert, "float64")[None, :]
    r = np.linalg.norm(v, axis=1)
    b = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    dev = np.abs((b - mu_gr + 180.0) % 360.0 - 180.0)
    dentro = (r <= L) & (r > 1e-6) & (dev <= apert_semi_gr)
    if perfil == "lineal":
        val = np.clip(1.0 - dev / max(apert_semi_gr, 1e-9), 0, 1)
    else:
        bo = np.asarray(PERFIL_MEDIDO_BORDES if perfil is None else perfil[0], "float64")
        va = np.asarray(PERFIL_MEDIDO_VALOR if perfil is None else perfil[1], "float64")
        idx = np.clip(np.searchsorted(bo[1:-1], dev, side="right"), 0, va.size - 1)
        val = va[idx]
    return np.where(dentro, val, 0.0)


ESCALA_POR_DEFECTO = "mediana"
"""QUE ESTIMADOR DE ESCALA SE USA. Vale `"mediana"` (el de siempre) o `"p95"` (el alto de pie).

SE QUEDA EN `"mediana"`, y **no porque el defecto no exista, sino porque el arreglo NO PASO SU PROPIA
PUERTA**. Precedente del proyecto: `USA_CRUDAS = False`, retirado con su medida.

El defecto es real y esta medido (`e20`): el alto de caja en guardia no es la estatura, y su razon
`p95/p50` dentro del clip vale 1,08-1,31. Pero la puerta declarada antes de medir era que el p95
fuera **reproducible dentro del clip al 5 %**, porque si no es una propiedad estable del tirador, no
sirve de referencia de estatura. Medido en `e22`, control R1, comparando la primera mitad del clip
con la segunda:

    p95 del alto de caja        discrepancia mediana **10,9 %**  (maximo 14,9)   ->  NO PASA
    mediana del alto de caja    discrepancia mediana **10,7 %**  (maximo 24,6)   ->  tampoco pasaria

O SEA QUE R1 NO SEPARA LOS DOS ESTIMADORES: falla para los dos, casi igual. Y eso es un resultado
sobre el metodo entero y no sobre el percentil: **el alto de caja no es una referencia de estatura
reproducible al 5 % en ningun percentil.** Cambiar la mediana por el p95 seria cambiar un estimador
no validado por otro no validado, moviendo de paso toda cifra en centimetros de la seccion 22.

Asi que se deja la mediana, la funcion nueva queda calculable con el interruptor, y `e22` publica en
su control R3 exactamente lo que se moveria si se adoptara. El arreglo de verdad del largo es el que
si tiene fuente primaria: `ALCANCE_M = 1,10` en vez de los 0,90 de la hoja sola, que son +22 % y no
dependen de ningun percentil. Y el arreglo de la escala de verdad sigue siendo el de 22.14: la
homografia de la pista sobre el rectangulo conocido de 14,00 x 1,50 m."""
PCTL_DE_PIE = 95.0


def escala_px_m_de_pie(B, lado, H, modo=ESCALA_POR_DEFECTO):
    """px por metro tomando como referencia de estatura el alto de caja DE PIE, no el mediano.

    EL DEFECTO QUE ARREGLA, medido en `e20`. `cono_lib.escala_px_m` divide el alto de la caja por
    1,75 m, o sea da por hecho que **el alto de la caja ES la estatura**. Y un esgrimista en guardia
    no mide su estatura: tiene las rodillas dobladas, y a fondo aun menos. Medido sobre 7 clips, la
    razon `p95/p50` del alto de caja dentro del clip vale **1,08 a 1,31, mediana 1,15**. Luego en el
    cuadro tipico la referencia esta un 15 % encogida, `px/m` sale un 15 % corto y **el largo del
    cono con el**. Eso es literalmente la queja del usuario: «es muy corto».

    Y concuerda con lo que 22.6 ya habia medido por otra via: el alto de caja cae un 20-30 % a fondo
    sin que el tirador se mueva en profundidad.

    EL INTERRUPTOR ES A PROPOSITO, al estilo de `USA_CRUDAS = False`:
        modo="mediana"  ->  delega en `cono_lib.escala_px_m`, bit a bit el valor viejo
        modo="p95"      ->  usa el percentil 95 del alto de caja del clip como alto de pie
    Asi cualquier cifra publicada se puede reproducir tal como se calculo, y la que cambia se puede
    senalar. `cono_lib.escala_px_m` NO se toca: la usan 15 sitios, varios de otros agentes.

    POR QUE p95 Y NO EL MAXIMO: el maximo lo fija una sola caja, y una caja mal ajustada (un brazo
    suelto, una fusion con el arbitro) lo dispara sin aviso. El p95 da el mismo numero cuando hay
    cuadros de pie y no se lo lleva un caso raro.

    Y OJO: ESTA FUNCION **NO ES LA POR DEFECTO**, porque no paso su puerta de reproducibilidad
    (`e22` control R1: el p95 de la primera mitad del clip contra el de la segunda discrepa un
    **10,9 %** contra el 5 % declarado, y la mediana falla igual con un 10,7 %). Ver la nota de
    `ESCALA_POR_DEFECTO`. Queda aqui para poder medir lo que se moveria si se adoptara, y para que el
    defecto del 15 % quede calculable, no para usarla a ciegas.

    Devuelve `(px_por_metro_por_cuadro, px_por_metro_del_clip)`, la misma forma que
    `cono_lib.escala_px_m`, para poder sustituirla sin tocar quien la consume.
    """
    if modo == "mediana":
        return CL.escala_px_m(B, lado, H)
    alto = (B[:, lado, 3] - B[:, lado, 1]) * H
    fin = alto[np.isfinite(alto)]
    if fin.size < 5:
        return CL.escala_px_m(B, lado, H)
    de_pie = float(np.percentile(fin, PCTL_DE_PIE))
    return alto / CL.ALTURA_M, de_pie / CL.ALTURA_M


def eje_supuesto_2d(lect_propio, mano, cual=EJE_POR_DEFECTO):
    """Direccion proyectada del brazo, de la articulacion `cual` a la raiz de la mano.

    Devuelve `(u2, indice_del_ancla)` o `(None, None)`, para que el ancla quede auditable.
    """
    M = CL.MANOS[mano]
    i_a = M["hombro"] if cual == "hombro" else M["codo"]
    i_r = M["root"]
    if i_a not in lect_propio or i_r not in lect_propio:
        return None, None
    v = np.asarray([lect_propio[i_r]["x"] - lect_propio[i_a]["x"],
                    lect_propio[i_r]["y"] - lect_propio[i_a]["y"]])
    n = float(np.linalg.norm(v))
    return (None, None) if n < 1e-9 else (v / n, int(i_a))


def eje_brazo_2d(kp, sc, lado, mano, cual=EJE_POR_DEFECTO):
    """Vertice, direccion y validez del eje del brazo, para TODOS los cuadros y SIN GPU.

    POR QUE NO SE TOCA `cono_lib.eje_antebrazo`, que hace casi lo mismo. Esa ancla en el CODO y es la
    que usaron `e3` a `e8`; cambiarla alteraria **en silencio** medidas ya publicadas (el suelo de
    apertura, la interseccion cruda de `e3`, la calibracion de `e6`). Se anade al lado con el ancla
    elegible y las medidas viejas siguen siendo reproducibles tal como se hicieron.

    Devuelve `(vert, u, ok, largo)`, la misma forma que `eje_antebrazo`, para poder sustituirla sin
    tocar el codigo que la consume.
    """
    M = CL.MANOS[mano]
    i_a = M["hombro"] if cual == "hombro" else M["codo"]
    k, s = kp[:, lado], sc[:, lado]
    v = k[:, M["root"]] - k[:, i_a]
    n = np.linalg.norm(v, axis=1, keepdims=True)
    u = v / np.maximum(n, 1e-9)
    ok = CL.bueno(s, [i_a, M["root"]]) & (n[:, 0] > 1.0)
    return k[:, M["root"]].astype("float64"), u.astype("float64"), ok, n[:, 0]


def eje_pulgar_2d(kp_fila, sc_fila, mano):
    """Direccion `raiz -> centroide del pulgar`. Se mide para poder RETIRARLA con su cifra al lado.

    `e2` ya la midio y no servia: 2,4-4,8 px de palanca, salto mediano de 11-20 grados entre cuadros
    consecutivos, y dos estimadores de la misma recta discrepando una mediana de 67-134 grados. Se
    vuelve a comparar aqui porque se ha propuesto como sustituto para 720p, donde la hoja no se ve, y
    en este proyecto una idea se retira **con su medida**, no por remision a una medida anterior.
    """
    M = CL.MANOS[mano]
    idx = M["pulgar"]
    if not (np.isfinite(kp_fila[idx]).all() and np.isfinite(kp_fila[M["root"]]).all()):
        return None
    if not (sc_fila[idx] >= CL.CONF_MIN).all():
        return None
    v = kp_fila[idx].mean(axis=0) - kp_fila[M["root"]]
    n = float(np.linalg.norm(v))
    return None if n < 0.5 else v / n


TOL_D1 = 0.35
PUERTAS_DURAS = frozenset(("D1", "D2", "D3"))
"""LOS TRES DEFECTOS DE ATRIBUCION, aplicados como PUERTAS y no solo medidos.

Estaban implementados y cuantificados en `e17_defectos_del_cono.py` desde hace tiempo, con estos
resultados sobre 7 clips y 3.418 cuadros-lado (`01_extraccion/pruebas/cono_e17_defectos.json`):

    D1  el vertice esta pegado al tronco, o sea no es una mano extendida     75 / 3.418 =  2,2 %
    D2  la mano de un tirador cae DENTRO del cuerpo del otro                101 / 3.418 =  3,0 %
    D3  la otra mano esta mas cerca del rival que la elegida                298 / 3.418 =  8,7 %

y `construye_cono` **no comprobaba ninguno**. Se median y se publicaban igual.

Y no son un modo de fallo teorico: la anotacion ciega de `02_validacion/muestras/contacto_hojas/
anotacion.csv` los recoge a mano, con estas palabras:

    L01  DEFECTO las dos marcas caen en el MISMO tirador y la de abajo esta en la cadera
    L06  DEFECTO las dos marcas caen en el MISMO tirador una en la cadera o espalda
    L10  DEFECTO la mano armada es el brazo TRASERO del que se lanza a fondo

Son D2, D2 y D3. Contando tambien las marcas mal puestas sin la palabra DEFECTO (L04 en el muslo
del rival, L11 en la cadera, L19 sobre un cartel), son **6 de 16 laminas inspeccionadas**. Alguien
lo vio, lo escribio, y no llego a ninguna cifra agregada ni a ninguna puerta.

Poner `PUERTAS_DURAS = frozenset()` las desactiva sin borrar la medida: `defectos` sale en el dict
del cono siempre, se apliquen o no, para poder contar cuantos se pierden."""


def defectos_de_atribucion(lect_propio, lect_rival, mano, caja_rival=None, W=None, H=None):
    """QUE DEVUELVE: el conjunto de defectos de atribucion de este cuadro-lado, subconjunto de
    {"D1", "D2", "D3"}. Vacio si no hay ninguno o si faltan puntos para decidirlo.

    Mismos criterios y misma tolerancia que `e17_defectos_del_cono.py`, a proposito: si se cambiaran
    aqui, las cifras de e17 dejarian de describir lo que hace el codigo y habria dos verdades.

    D2 USA LA CAJA DEL RIVAL SI SE LE PASA, y si no la envolvente de su torso. La diferencia esta
    medida y no es pequena: sobre los mismos 1.684 cuadros-lado, con la envolvente de torso salta el
    **0,9 %** y con la caja de YOLO el **3,0 %** que publica e17. La envolvente es una region mas
    pequena que la caja, asi que sin caja esta puerta filtra un tercio de lo que deberia. Por eso
    `caja_rival` existe: el llamante que la tenga debe pasarla, y `e10_cono3d_interseccion.py` la
    tiene.

    `caja_rival` es xyxy NORMALIZADO, como `B` en los npz, y por eso hacen falta `W` y `H`.
    """
    d = set()
    M = CL.MANOS[mano]
    MO = CL.MANOS["B" if mano == "A" else "A"]

    def pt(lect, i):
        if i not in lect:
            return None
        p = np.array([lect[i]["x"], lect[i]["y"]], "float64")
        return p if np.isfinite(p).all() else None

    pr = pt(lect_propio, M["root"])
    ph = pt(lect_propio, M["hombro"])
    cad = [pt(lect_propio, i) for i in CADERAS]
    if pr is not None and ph is not None and all(c is not None for c in cad):
        pc = np.mean(cad, axis=0)
        torso = float(np.linalg.norm(ph - pc))
        if torso > 1 and float(np.linalg.norm(pr - pc)) < TOL_D1 * torso:
            d.add("D1")

    if pr is not None and caja_rival is not None and W and H:
        b = np.asarray(caja_rival, "float64")
        if np.isfinite(b).all() and len(b) == 4:
            if (b[0] * W <= pr[0] <= b[2] * W) and (b[1] * H <= pr[1] <= b[3] * H):
                d.add("D2")
    elif pr is not None:
        esq = [pt(lect_rival, i) for i in (5, 6, 9, 10)]
        if all(e is not None for e in esq):
            Q = np.asarray(esq)
            if (Q[:, 0].min() <= pr[0] <= Q[:, 0].max()
                    and Q[:, 1].min() <= pr[1] <= Q[:, 1].max()):
                d.add("D2")

    hom_r = [pt(lect_rival, i) for i in CL.HOMBROS]
    pa = pt(lect_propio, MO["root"])
    if pr is not None and pa is not None and all(h is not None for h in hom_r):
        xr = float(np.mean([h[0] for h in hom_r]))
        if abs(pa[0] - xr) < abs(pr[0] - xr):
            d.add("D3")
            """D3 no dice que la mano elegida sea la equivocada: dice que el criterio (a) de
            `mano_armada` -- la mano mas cercana al rival -- se contradice a si mismo en este
            cuadro. Como la cascada decide una vez por clip y esto es por cuadro, la contradiccion
            puede venir de un fondo profundo. Por eso se declara como defecto de ATRIBUCION y no
            como "mano equivocada"."""
    return d


def construye_cono(lect_propio, lect_rival, mano, W, H, f, pxm, linea, apert_med, apert_sup,
                   n_malla=N_MALLA, eje=EJE_POR_DEFECTO, caja_rival=None):
    """EL CONO DE UN TIRADOR EN UN CUADRO, en 2D y en 3D a la vez, y su veredicto de amenaza.

    Vive aqui y no en el paso 10 porque el paso de FIGURAS tiene que dibujar **exactamente** el cono
    que se midio. Si la figura reconstruyera la geometria con su propio codigo, una discrepancia
    entre el numero y el dibujo seria indetectable, y este proyecto ya perdio un dia entero por una
    visualizacion que no coincidia con los datos (el desfase `f0`).

    Devuelve None si falta cualquier pieza. Nunca rellena una pieza que falta: un cono construido a
    medias no se distingue despues de uno completo.

    LA COMPARACION ESTA MONTADA PARA QUE SOLO CAMBIE UNA COSA: mismo vertice, mismo eje proyectado,
    misma apertura, mismo L, mismo torso y **la misma forma de peso**. Lo unico que cambia es si la
    tercera coordenada entra o no.
    """
    M = CL.MANOS[mano]
    i_c, i_r, i_h = M["codo"], M["root"], M["hombro"]
    i_a = i_h if eje == "hombro" else i_c
    """`i_a` es el ANCLA del eje del brazo. Por defecto el HOMBRO, corregido respecto al encargo."""
    if i_r not in lect_propio or i_a not in lect_propio or i_c not in lect_propio:
        return None
    defs = defectos_de_atribucion(lect_propio, lect_rival, mano, caja_rival, W, H)
    if PUERTAS_DURAS and (defs & PUERTAS_DURAS):
        return None
    """LAS PUERTAS VAN AQUI ARRIBA, antes de gastar nada: si el vertice no esta en una mano, o esta
    en el cuerpo del rival, no hay cono que construir por bien que salga la geometria."""
    esquinas = [5, 6, 10, 9]
    if not all(i in lect_rival for i in esquinas):
        return None
    Zref = lect_propio[i_r]["Z"]
    V = p3d_exacto(lect_propio[i_r]["x"], lect_propio[i_r]["y"], Zref, W, H, f, Zref)
    Pa = p3d_exacto(lect_propio[i_a]["x"], lect_propio[i_a]["y"], lect_propio[i_a]["Z"],
                    W, H, f, Zref)
    Pc = p3d_exacto(lect_propio[i_c]["x"], lect_propio[i_c]["y"], lect_propio[i_c]["Z"],
                    W, H, f, Zref)
    u3b = dir3d(Pa, V)
    u3b_codo = dir3d(Pc, V)
    """La ELEVACION sale del eje anclado en `i_a` (el hombro por defecto). Se guarda ademas la del
    codo para poder comparar las dos sin volver a pasar la GPU."""
    if V is None or u3b is None:
        return None
    q3 = [p3d_exacto(lect_rival[i]["x"], lect_rival[i]["y"], lect_rival[i]["Z"], W, H, f, Zref)
          for i in esquinas]
    if any(q is None for q in q3):
        return None
    q2 = [np.array([lect_rival[i]["x"], lect_rival[i]["y"]]) for i in esquinas]
    malla3 = malla_torso(q3[0], q3[1], q3[2], q3[3], n_malla)
    malla2 = malla_torso(q2[0], q2[1], q2[2], q2[3], n_malla)

    def fin(v):
        return v is not None and np.isfinite(v)

    if linea is not None and linea.get("delante"):
        p1, p2 = np.asarray(linea["p1"], "float64"), np.asarray(linea["p2"], "float64")
        u2 = (p2 - p1) / max(float(np.linalg.norm(p2 - p1)), 1e-9)
        origen = 0
        """DOS COLUMNAS DE PROCEDENCIA, NO UNA, y hacen falta las dos separadas. `origen` dice de
        donde sale la DIRECCION del eje (hoja vista o brazo). `apert_origen` dice de donde sale la
        APERTURA, que puede tener otra procedencia: si en el clip no hay bastantes pares de cuadros
        consecutivos con hoja detectada en los dos, su propio temblor no se puede medir y hay que
        caer en la regla de `e6`, que es una apertura MUCHO mas ancha y de otra naturaleza.

        Antes esto se resolvia devolviendo None, y era un error grave y silencioso: **tiraba
        justamente los conos de hoja VISTA**, que son los buenos, y solo en los clips con pocas
        detecciones. O sea que el sesgo dependia del clip y no se veia en ninguna cifra."""
        apert, apert_org = ((apert_med, 0) if fin(apert_med)
                            else ((apert_sup, 1) if fin(apert_sup) else (APERT_E6_POOL, 2)))
    else:
        """LA HOJA NO SE VE: el eje se SUPONE desde el brazo, anclado en el hombro.

        Y AQUI NO SE PROLONGA LA HOJA, que esta medido y prohibido: el brazo apunta entre 24,8 y
        29,7 grados FUERA de la direccion real de la hoja. Lo que se hace es declarar una REGION de
        90 cm de radio con una apertura que **cubre ese error medido** (`apert_sup`), no trazar una
        hoja donde no se ha visto. La diferencia importa: una hoja dibujada seria un dato inventado
        que se confunde con uno medido, y ya hubo un cuaderno en este proyecto que trazaba 95 px
        desde la muneca y metia esa hoja inventada en la etiqueta de tocado. Eso no vuelve a entrar.
        """
        u2, _anc = eje_supuesto_2d(lect_propio, mano, eje)
        if u2 is None:
            return None
        origen = 1
        apert, apert_org = (apert_sup, 1) if fin(apert_sup) else (APERT_E6_POOL, 2)
    if not fin(apert):
        return None
    u3 = eje3d_desde_hoja(u2, u3b)
    if u3 is None:
        return None

    L = ALCANCE_M * pxm
    """SE CABLEA `ALCANCE_M` Y NO `CL.HOJA_M`, que es lo que este modulo lleva declarando desde que
    se escribio `ALCANCE_M` 440 lineas mas arriba y nunca se enchufo. Con `HOJA_M = 0,90` habia
    TRES valores para la misma magnitud a la vez: el dato en los npz salia 0,900 m, la figura de
    `e21_pinta_cono_arreglado.py` pintaba 1,10 y `02_validacion/DATOS_POR_CLIP.md` documentaba
    1,10. Es el fallo que el proyecto tiene documentado como "el mismo nombre con un numero
    distinto", con el agravante de que aqui la propia justificacion de por que 0,90 esta mal ya
    estaba escrita al lado de la constante.

    CAMBIA CIFRAS PUBLICADAS: `am2` y `am3` -- el veredicto de amenaza -- dependen de `L`, asi que
    los npz de `01_extraccion/pruebas/` hay que regenerarlos antes de citar ninguna cifra de cono."""
    vert2 = np.array([lect_propio[i_r]["x"], lect_propio[i_r]["y"]])
    w2 = CL.peso_sector(malla2, vert2, u2, L, apert)
    w3 = peso_sector3d(malla3, V, u3, L, apert)
    d2 = np.linalg.norm(malla2 - vert2[None, :], axis=1)
    d3 = np.linalg.norm(malla3 - V[None, :], axis=1)
    a2 = np.degrees(np.arccos(np.clip(((malla2 - vert2[None, :]) @ u2)
                                      / np.maximum(d2, 1e-9), -1, 1)))
    a3 = np.degrees(np.arccos(np.clip(((malla3 - V[None, :]) @ u3)
                                      / np.maximum(d3, 1e-9), -1, 1)))
    cm = 100.0 / pxm
    Zt = float(np.median([lect_rival[i]["Z"] for i in esquinas]))
    s0 = lect_propio[i_r]["s"]
    p_codo = np.array([lect_propio[i_c]["x"], lect_propio[i_c]["y"]])
    p_hom = (np.array([lect_propio[i_h]["x"], lect_propio[i_h]["y"]])
             if i_h in lect_propio else None)
    """El codo y el hombro se devuelven para que la FIGURA pueda pintar el brazo que genera el eje.
    Sin verlo, nadie puede comprobar a ojo que el cono arranca en la mano armada y no en otra."""
    return {"p_codo": p_codo, "p_hombro": p_hom, "apert_origen": apert_org,
            "eje_ancla": int(i_a), "elev_codo": elevacion(u3b_codo) if u3b_codo is not None
            else float("nan"),
            "V": V, "u3": u3, "u3b": u3b, "u2": u2, "vert2": vert2, "apert": float(apert),
            "L": L, "origen": origen, "defectos": sorted(defs),
            "malla2": malla2, "malla3": malla3, "q2": q2, "q3": q3,
            "am2": bool((w2 > 0).any()), "am3": bool((w3 > 0).any()),
            "w2": float(w2.max()), "w3": float(w3.max()),
            "d2_cm": float(d2.min()) * cm, "d3_cm": float(d3.min()) * cm,
            "ang2": float(a2.min()), "ang3": float(a3.min()),
            "elev": elevacion(u3), "elev_brazo": elevacion(u3b),
            "dz_cm": (Zt - Zref) / max(s0, 1e-12) * cm,
            "razon_Z": Zt / Zref if abs(Zref) > 1e-12 else float("nan"),
            "fuera_de_alcance_3d": bool(d3.min() > L), "L_cm": L * cm}


def planta(cv2, vert3, u3, L, apert, malla3, pxm, tit, pie="", ancho=660, alto=470, mar=None):
    """VISTA DE PLANTA, desde arriba: eje x de la imagen en horizontal, PROFUNDIDAD en vertical.

    POR QUE ESTA VISTA Y NO OTRA. Es la unica que hace visible el argumento del usuario: *"dos
    floretes que en la imagen se cruzan pueden estar a un metro de distancia en profundidad"*. En la
    imagen eso no se ve por definicion, porque la imagen es la proyeccion que borra la profundidad.
    En planta se ve de un golpe y se puede juzgar a ojo, que es lo que se pide.

    Todo en CENTIMETROS, con la escala px/m del propio tirador. Los circulos de 30, 60, 90 y 120 cm
    dan la referencia: la hoja de florete mide 90 cm (FIE m.8.4), asi que el circulo de 90 es el
    alcance del arma y lo que quede fuera **no se puede tocar**.

    La silueta de un cono circular visto de planta son sus dos rayos extremos, que son el eje girado
    +-apertura alrededor del eje vertical de la imagen. Se dibujan esos dos y el eje.
    """
    v = np.zeros((alto, ancho, 3), np.uint8)
    v[:] = (24, 24, 30)
    P = np.array([[p[0], p[2]] for p in [vert3, vert3 + L * u3] + list(malla3)]) * 100.0 / pxm
    P = P - P[0]
    m = mar if mar else max(30.0, float(np.abs(P).max()) * 1.12)
    esc = min(ancho - 100, alto - 90) / (2 * m)
    ox, oy = ancho // 2, alto // 2

    def a_px(q):
        return (int(ox + q[0] * esc), int(oy - q[1] * esc))
    for r_cm in (30, 60, 90, 120, 180, 240):
        if r_cm * esc > min(ancho, alto):
            continue
        col = (70, 130, 90) if r_cm == 90 else (48, 48, 56)
        cv2.circle(v, a_px((0, 0)), int(r_cm * esc), col, 1, cv2.LINE_AA)
        cv2.putText(v, "%d" % r_cm, (ox + int(r_cm * esc) - 20, oy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 110), 1)
    cv2.line(v, (26, oy), (ancho - 26, oy), (60, 60, 70), 1)
    cv2.line(v, (ox, 32), (ox, alto - 32), (60, 60, 70), 1)
    cv2.putText(v, "+ profundidad (cm)", (ox + 6, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                (110, 110, 120), 1)
    ph = np.radians(apert)
    ca, sa = np.cos(ph), np.sin(ph)
    for sg in (+1, -1):
        w = np.array([u3[0] * ca - sg * u3[2] * sa, u3[1], u3[0] * sg * sa + u3[2] * ca])
        w = w / max(float(np.linalg.norm(w)), 1e-9)
        q = (L * w) * 100.0 / pxm
        cv2.line(v, a_px((0, 0)), a_px((q[0], q[2])), (70, 200, 120), 1, cv2.LINE_AA)
    cv2.line(v, a_px((0, 0)), a_px(P[1]), (110, 255, 170), 2, cv2.LINE_AA)
    for j in range(2, len(P)):
        cv2.circle(v, a_px(P[j]), 2, (60, 210, 255), -1, cv2.LINE_AA)
    cv2.circle(v, a_px((0, 0)), 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(v, "mano", (a_px((0, 0))[0] + 7, a_px((0, 0))[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (235, 235, 235), 1)
    cv2.putText(v, "torso rival", (a_px(P[2])[0] + 7, a_px(P[2])[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (60, 210, 255), 1)
    cv2.rectangle(v, (0, 0), (ancho, 24), (14, 14, 20), -1)
    cv2.putText(v, tit, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1)
    if pie:
        cv2.rectangle(v, (0, alto - 22), (ancho, alto), (14, 14, 20), -1)
        cv2.putText(v, pie, (6, alto - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 235, 255), 1)
    return v


DILATA = 8
CERCA_ROOT = 0.12
MIN_LARGO = 0.25
PCTL = 99.5
AZAR = 45.0
"""Parametros de la deteccion de la hoja, EXTRAIDOS de `e5_hoja_en_alfa.py` sin cambiar ni uno.
Estan medidos alli: umbral = percentil del alfa FUERA del cuerpo del propio recorte (un umbral fijo
no vale porque el nivel de fondo cambia el doble de un recorte a otro), cuerpo dilatado 8 px porque
el borde de la silueta tambien da alfa intermedio, y la recta se elige por POSICION (que pase a menos
de 0,12 R de la mano) y nunca por direccion, para no contaminar la medida del angulo."""


def recorte_hoja(b, br, vert, u, L, W, H):
    """Recorte de UN tirador alargado a lo largo del eje hasta L y cortado antes del rival.

    EXTRAIDO de `e5_hoja_en_alfa.py`. El corte antes del rival no es cosmetico: con dos tiradores
    dentro, `matting` devuelve 0,0053, que es el valor del fondo vacio.
    """
    x0, y0, x1, y1 = b[0] * W, b[1] * H, b[2] * W, b[3] * H
    mx, my = 0.18 * (x1 - x0), 0.15 * (y1 - y0)
    punta = vert + L * u
    cx0 = min(x0 - mx, vert[0] - 0.25 * L, punta[0] - 0.10 * L)
    cx1 = max(x1 + mx, vert[0] + 0.25 * L, punta[0] + 0.10 * L)
    cy0 = min(y0 - my, punta[1] - 0.10 * L)
    cy1 = max(y1 + my, punta[1] + 0.10 * L)
    if np.isfinite(br).all():
        rx0, rx1 = br[0] * W, br[2] * W
        pad = 0.12 * (rx1 - rx0)
        if 0.5 * (rx0 + rx1) > 0.5 * (x0 + x1):
            cx1 = min(cx1, rx0 - pad)
        else:
            cx0 = max(cx0, rx1 + pad)
    cx0, cy0 = int(max(0, cx0)), int(max(0, cy0))
    cx1, cy1 = int(min(W, cx1)), int(min(H, cy1))
    R = 0.0
    if cx1 > cx0 and cy1 > cy0:
        R = min(L, max(0.0, min(vert[0] - cx0, cx1 - vert[0], vert[1] - cy0, cy1 - vert[1])))
    return (cx0, cy0, cx1, cy1), float(R)


def dif90(a, b):
    """Diferencia entre dos RECTAS (no vectores), en [0, 90]."""
    return abs((a - b + 90.0) % 180.0 - 90.0)


def hoja_en_alfa(cv2, alfa, seg, vert, u, R, off):
    """La hoja como recta en el alfa de `matting`. EXTRAIDO de `e5_hoja_en_alfa.py`.

    `off` es (cx0, cy0), el origen del recorte, para devolver la recta en coordenadas de imagen.
    Devuelve None si no hay recta pegada a la mano.

    Y AQUI VA EL DESCARTE DEL CONFUSO DEL BRAZO, que `e6` midio y que sin el todo esto seria falso:
    la hoja sale de la mano HACIA DELANTE, y el brazo esta DETRAS. Se proyecta el segmento sobre el
    eje `codo -> root` tomando la mano como origen y se exige que su punto mas lejano caiga delante
    (`t_max > 0`). A 1080p el 92,3 % de las rectas detectadas cumplen eso, o sea que lo que se ve no
    es el borde del brazo.
    """
    cx0, cy0 = off
    cuerpo = cv2.dilate((seg != 0).astype("uint8"), np.ones((2 * DILATA + 1,) * 2, np.uint8)) > 0
    fuera = alfa[~cuerpo]
    if fuera.size < 500:
        return None
    hh, ww = alfa.shape[:2]
    GY, GX = np.mgrid[0:hh, 0:ww]
    disco = np.hypot(GX + cx0 - vert[0], GY + cy0 - vert[1]) <= R
    t = float(np.percentile(fuera, PCTL))
    m = disco & (~cuerpo) & (alfa > t)
    if int(m.sum()) < 20:
        return None
    lines = cv2.HoughLinesP((m * 255).astype("uint8"), 1, np.pi / 360.0, 12,
                            minLineLength=int(max(8, MIN_LARGO * R)),
                            maxLineGap=int(max(2, 0.05 * R)))
    if lines is None:
        return None
    lines = np.asarray(lines).reshape(-1, 4)
    cands = []
    for x1_, y1_, x2_, y2_ in lines:
        p1 = np.array([x1_ + cx0, y1_ + cy0], dtype="float64")
        p2 = np.array([x2_ + cx0, y2_ + cy0], dtype="float64")
        v = p2 - p1
        lg = float(np.linalg.norm(v))
        if lg < 1:
            continue
        nv = np.array([-v[1], v[0]]) / lg
        if abs(float(nv @ (vert - p1))) > CERCA_ROOT * R:
            continue
        cands.append((lg, p1, p2))
    if not cands:
        return None
    cands.sort(key=lambda c: -c[0])
    lg, p1, p2 = cands[0]
    t1, t2 = float((p1 - vert) @ u), float((p2 - vert) @ u)
    """El segmento se ORIENTA hacia delante: `HoughLinesP` devuelve una recta sin sentido, y la punta
    del florete esta del lado que se aleja del codo. Esto resuelve el signo del que habla el paso 10.
    """
    if t2 < t1:
        p1, p2, t1, t2 = p2, p1, t2, t1
    v = p2 - p1
    return {"p1": p1, "p2": p2, "largo": lg, "t_min": t1, "t_max": t2,
            "u2d": v / max(float(np.linalg.norm(v)), 1e-9),
            "ang": float(np.degrees(np.arctan2(v[1], v[0]))),
            "umbral": t, "n_px": int(m.sum()), "n_lineas": int(len(lines)), "n_cerca": len(cands),
            "delante": bool(t2 > 0)}


def eje3d_desde_hoja(u2d_hoja, u3d_brazo):
    """EL EJE 3D DEL ARMA. Es el paso central del paso 10 y su razonamiento va escrito aqui.

    QUE SE MIDE Y QUE SE SUPONE, separado a proposito:

      MEDIDO   la DIRECCION PROYECTADA del arma, `u2d_hoja`, que sale del filo visto en el alfa de
               `matting`. Son dos numeros medidos sobre pixeles reales.
      MEDIDO   la ELEVACION del ANTEBRAZO, que sale de la z de `pointmap` en codo y mano.
      SUPUESTO **que la hoja saca del plano lo mismo que el antebrazo.** Esto NO esta medido y no se
               puede medir con lo que hay, porque la hoja tiene 0,64-0,96 px de ancho y `pointmap`
               en un pixel de hoja lee el FONDO, no la hoja. Es la unica suposicion de la
               construccion y va marcada como tal en el dato que se guarda.

    POR QUE ESO DETERMINA EL EJE. Una direccion 3D unitaria tiene dos grados de libertad. La
    proyeccion medida fija uno (el azimut en el plano de imagen) y la elevacion supuesta fija el otro.
    Con `phi` la elevacion y `(dx, dy)` la direccion proyectada unitaria:

        u3d = (cos(phi) * dx,  cos(phi) * dy,  sin(phi))

    y se comprueba de balde que su proyeccion sobre el plano de imagen vuelve a dar `u2d_hoja`.

    EL SIGNO. `HoughLinesP` devuelve una RECTA, sin sentido, asi que hay dos ejes opuestos posibles.
    Se resuelve **antes** de llegar aqui, en `hoja_en_alfa`, orientando el segmento hacia el lado que
    se aleja del codo: la punta del florete esta delante de la mano y el brazo detras. No se usa
    «apunta al rival» para el signo porque eso metaria la respuesta en la pregunta: si el signo se
    eligiera para que el arma apunte al rival, la interseccion con el rival estaria garantizada por
    construccion.
    """
    if u2d_hoja is None or u3d_brazo is None:
        return None
    phi = np.radians(elevacion(u3d_brazo))
    u = np.array([np.cos(phi) * u2d_hoja[0], np.cos(phi) * u2d_hoja[1], np.sin(phi)],
                 dtype="float64")
    return u / max(float(np.linalg.norm(u)), 1e-9)

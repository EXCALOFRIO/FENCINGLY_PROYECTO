"""LA PRIORIDAD instante a instante: quien tiene el derecho de tocado, y por que regla.

QUE PIDIO EL USUARIO, literal: "necesito que saques lo de la PRIORIDAD, para ver si sacas cuando
tiene cada uno la prioridad, para ponerlo. Y cuando la pierde. Y revisa la FIE para ver cuando". Y
despues: "cuando la pierde, cuando la gana el otro, con el intervalo que habra, un intervalo de
gracia, porque si no estaria cambiando infinitamente todo el rato". Y: "que detectamos bien las
paradas".

EL REGLAMENTO ESTA EN `04_documentacion/prioridad_fie.md`, con los articulos REALES sacados del
Reglamento Tecnico de la FIE de diciembre de 2025 que hay en el disco. Y el primer resultado de
leerlo es una correccion al encargo: **la prioridad en florete NO esta en t.56-t.60**, que es
anulacion por fallo del material y video-arbitraje. Esta en **t.82-t.89**, y sus definiciones en
t.8-t.15.

LA REGLA QUE HACE ESTO POSIBLE SIN VER LA HOJA es t.83.2.a:

  "El ataque simple, directo o indirecto, es ejecutado correctamente cuando al extender el brazo, la
   punta que amenaza la superficie valida, PRECEDE el inicio del a fondo o de la flecha."

O sea que la correccion de un ataque es **el signo de una resta de dos instantes**: el cuadro en que
el brazo empieza a extenderse contra el cuadro en que empieza el fondo. Dos instantes sobre una serie
de poses es medible. Por eso hay detector.

LO QUE ESTE MODULO NO PUEDE HACER, dicho aqui arriba y no escondido al final: cinco de las doce
reglas de concesion de prioridad (t.84.1, t.84.2, t.85, t.86, t.87) piden **ver la hoja**, y la hoja
esta medida en este proyecto: se detecta en el **42,6 %** de los recortes a 1080p y en el **1,6 %** a
720p, donde su fuerte mide 0,64 px. Veinte de los veintiun clips de florete con poses de 308 puntos son 720p. Asi
que **la parada no se ve**, y sin parada no se ve la respuesta, que es el mecanismo por el que la
prioridad cambia de manos en media esgrima. Lo que si se ve es el reparto INICIAL y la perdida por
fallo propio (t.83.2.d, t.89.5.d).

EL MAPA DE INDICES ES EL MEDIDO, no el del plan:
  5-6 hombros  7-8 codos  9-10 caderas  11-12 rodillas  13-14 tobillos  15-20 pies
  21-41 mano A (root 21)  42-62 mano B (root 42)  63 en adelante cara, que no se usa
NO HAY MUNECA: la muneca ES la raiz de la mano, y el eje del brazo es `codo -> raiz`.
"""
from __future__ import annotations

import numpy as np

import cono_lib as CL

HOMBROS = (5, 6)
CADERAS = (9, 10)
TOBILLOS = (13, 14)
CADERA_DE = {"A": 10, "B": 9}
"""Que cadera va con que hombro. Sale del `flip_pairs` del modelo: los pares son (5,6) (9,10) y el
indice impar va con el impar, asi que el hombro 6 va con la cadera 10, que es la mano A de
`cono_lib`."""
CONF_MIN = 0.3
"""Mismo umbral que el resto del proyecto. Y ojo con la trampa ya medida: hay puntos con confianza
por encima de 0,3 y coordenada NaN, asi que SIEMPRE se comprueban las dos cosas."""


def _bien(sc, kp, idx):
    """Confianza Y coordenada finita, para una lista de indices. Devuelve (T,) booleano."""
    ok = np.ones(len(sc), dtype=bool)
    for i in idx:
        ok &= (sc[:, i] >= CONF_MIN) & np.isfinite(kp[:, i, 0]) & np.isfinite(kp[:, i, 1])
    return ok


def _largo(a, b):
    return np.linalg.norm(a - b, axis=-1)


def suaviza(x, ventana=5, orden=2, deriv=0):
    """Savitzky-Golay POR TRAMOS FINITOS, y la palabra clave es «por tramos».

    POR QUE POR TRAMOS: este proyecto ya se comio que `np.unwrap` es acumulativo y un solo NaN
    envenena todo lo posterior — bajo una cobertura de 0,846 a 0,577 sin que faltara un dato. Un
    filtro aplicado sobre una serie con huecos hace lo mismo de forma mas silenciosa: propaga el
    hueco `ventana` cuadros a cada lado. Aqui cada tramo contiguo de datos finitos se filtra solo, y
    los tramos mas cortos que la ventana se dejan crudos en vez de inventarles suavizado.

    POR QUE SAVITZKY-GOLAY Y NO UNA DIFERENCIA: el **75,8 % de los clips a 30 fps del corpus tiene
    telecine 25->30**, o sea un fotograma repetido cada seis exactos. Un duplicado da velocidad cero
    en un paso y doble en el siguiente, que es el artefacto que una diferencia no perdona. En este
    proyecto ya esta medido lo que arregla el suavizado sobre una derivada angular: el p99 baja de
    2.388,9 a 1.483,7 grados/s, factor 1,61 de artefacto quitado.
    """
    from scipy.signal import savgol_filter
    y = np.full(len(x), np.nan, dtype="float64")
    fin = np.isfinite(x)
    if not fin.any():
        return y
    bordes = np.nonzero(np.diff(fin.astype("int8")))[0]
    ini = 0
    cortes = list(bordes + 1) + [len(x)]
    for c in cortes:
        if fin[ini]:
            n = c - ini
            if n >= ventana:
                y[ini:c] = savgol_filter(x[ini:c], ventana, orden, deriv=deriv)
            elif deriv == 0:
                y[ini:c] = x[ini:c]
            elif n >= 2:
                y[ini:c] = np.gradient(x[ini:c])
        ini = c
    return y


HUECO_PUENTE = 2
"""Cuadros que se puentean en una serie escalar antes de derivar. Dos y no mas: con hueco de 1 cuadro
el error de interpolar una pose esta medido en el 2,8 % del torso, con 3 cuadros en el 4,8 % y con 5
en el 7,3 %. Dos cuadros esta del lado bueno de esa curva, y el limite existe para que un hueco largo
—donde el tirador puede no estar ni en el plano— no se rellene nunca."""


def _puentea(x, hueco):
    """QUE HACE: rellena por interpolacion lineal los huecos de `x` de hasta `hueco` cuadros.

    Solo los INTERIORES: un hueco en un extremo no tiene dos anclas y no se toca. Es la misma regla
    que ya rige la interpolacion de poses en este proyecto, y por el mismo motivo.
    """
    y = np.array(x, dtype="float64")
    fin = np.isfinite(y)
    if fin.sum() < 2:
        return y, 0
    idx = np.nonzero(fin)[0]
    n = 0
    for a, b in zip(idx[:-1], idx[1:]):
        g = b - a - 1
        if 0 < g <= hueco:
            for j in range(1, g + 1):
                y[a + j] = y[a] + (y[b] - y[a]) * j / float(g + 1)
            n += g
    return y, n


def senales(z) -> dict:
    """QUE HACE: saca las variables geometricas de las que sale toda la decision de prioridad.

    `z` es un `poses_*.npz` ya cargado (KP, SC, B, pres, origen, mask, W, H, T).

    LAS VARIABLES, y cada una lleva su porque:

      `ext_hr`  |hombro -> raiz de la mano| del brazo armado, partido por el torso. Es LA variable de
                t.83.2.a y t.83.2.d: es la que crece cuando el brazo se extiende.
      `ext_cr`  |codo -> raiz| partido por el torso. La pidio el encargo, y se mide para poder decir
                con cifras que **no sirve como extension**: es el largo del antebrazo, que es un
                hueso y no cambia de largo. Sirve de control de escala, no de senal.
      `codo`    angulo hombro-codo-raiz en grados. 180 es brazo recto. Es la lectura mas directa de
                "brazo extendido" contra "brazo encogido" de t.83.2.d, y no depende del torso.
      `pies`    |tobillo -> tobillo| partido por el torso. Es el indicador de a fondo de t.83.2.a.
      `vx`      velocidad del centro de caderas hacia el rival, en torsos por cuadro. Con SIGNO: el
                signo es lo que distingue avanzar de retroceder, o sea preparacion de retirada.
      `sep`     distancia entre los dos tiradores en torsos. Es la que dice si hay distancia de
                ataque o si estan en cuerpo a cuerpo.
      `apunta`  coseno entre el eje `codo -> raiz` y la direccion al torso del rival. **Se calcula y
                NO se usa para decidir**, porque su control ya esta FALLADO en este proyecto: la
                diferencia pareada guardia-tocado da -2,9 grados y el signo esperado sale en 17 de 36
                clip x lado, o sea azar. Viaja como columna cruda para que se pueda volver a mirar.
      `manos`   distancia entre las dos manos armadas en torsos. Es el unico proxy de "las hojas
                pueden estar en contacto" que no necesita ver la hoja.

    LA VALIDEZ ES PARTE DEL DATO, no un detalle. `val[k, lado]` exige las cuatro cosas: cuadro
    utilizable, pose MEDIDA (`origen == 0`), torso con largo positivo y los puntos que hacen falta con
    confianza. Una pose interpolada NO puede sostener una decision de prioridad, y la razon es
    concreta: `t_ext` es justo el codo del movimiento, y una interpolacion lineal entre dos anclas
    **borra el codo**, que es la senal que se busca.
    """
    KP, SC, B = z["KP"], z["SC"], z["B"]
    origen, mask = z["origen"], z["mask"]
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    fuera = {}
    manoq = {}
    for lado in (0, 1):
        m, det = CL.mano_armada(KP, SC, B, W, H, lado)
        manoq[lado] = m if m is not None else "A"
        fuera[lado] = det
    out = {k: np.full((T, 2), np.nan, dtype="float64")
           for k in ("ext_hr", "ext_cr", "codo", "pies", "vx", "apunta", "torso", "cx", "cy")}
    val = np.zeros((T, 2), dtype=bool)
    for lado in (0, 1):
        M = CL.MANOS[manoq[lado]]
        kp, sc = KP[:, lado], SC[:, lado]
        okt = _bien(sc, kp, list(HOMBROS) + list(CADERAS))
        mh = 0.5 * (kp[:, HOMBROS[0]] + kp[:, HOMBROS[1]])
        mc = 0.5 * (kp[:, CADERAS[0]] + kp[:, CADERAS[1]])
        torso = np.where(okt, _largo(mh, mc), np.nan)
        alt = np.where(okt, np.nan, np.nan)
        par = _bien(sc, kp, [M["hombro"], CADERA_DE[manoq[lado]]])
        """Si no estan los cuatro puntos del torso se cae al hombro y cadera del lado armado, que es
        media medida pero es la que hay. Se cuenta cuantas veces pasa."""
        alt = np.where(par, _largo(kp[:, M["hombro"]], kp[:, CADERA_DE[manoq[lado]]]), np.nan)
        torso = np.where(np.isfinite(torso), torso, alt)
        torso = np.where(torso > 1.0, torso, np.nan)
        out["torso"][:, lado] = torso
        okb = _bien(sc, kp, [M["hombro"], M["codo"], M["root"]])
        out["ext_hr"][:, lado] = np.where(okb, _largo(kp[:, M["hombro"]], kp[:, M["root"]]) / torso, np.nan)
        out["ext_cr"][:, lado] = np.where(okb, _largo(kp[:, M["codo"]], kp[:, M["root"]]) / torso, np.nan)
        u = kp[:, M["hombro"]] - kp[:, M["codo"]]
        v = kp[:, M["root"]] - kp[:, M["codo"]]
        nu = np.linalg.norm(u, axis=1)
        nv = np.linalg.norm(v, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cs = (u * v).sum(axis=1) / np.maximum(nu * nv, 1e-9)
        out["codo"][:, lado] = np.where(okb & (nu > 1) & (nv > 1),
                                        np.degrees(np.arccos(np.clip(cs, -1, 1))), np.nan)
        okp = _bien(sc, kp, list(TOBILLOS))
        out["pies"][:, lado] = np.where(okp, _largo(kp[:, TOBILLOS[0]], kp[:, TOBILLOS[1]]) / torso, np.nan)
        okc = _bien(sc, kp, list(CADERAS))
        out["cx"][:, lado] = np.where(okc, mc[:, 0], np.nan)
        out["cy"][:, lado] = np.where(okc, mc[:, 1], np.nan)
        val[:, lado] = mask & (origen[:, lado] == 0) & np.isfinite(torso) & okb
    tm = np.nanmean(out["torso"], axis=1)
    for lado in (0, 1):
        rival = 1 - lado
        signo = 1.0 if lado == 0 else -1.0
        """El signo hace que `vx > 0` sea SIEMPRE «hacia el rival», los dos lados con el mismo
        convenio. Sin el signo, comparar los dos lados exige recordar de que lado esta cada uno, y
        ese es el tipo de detalle que produce un fallo silencioso."""
        d = suaviza(out["cx"][:, lado], deriv=1)
        out["vx"][:, lado] = signo * d / tm
        M = CL.MANOS[manoq[lado]]
        kp, sc = KP[:, lado], SC[:, lado]
        eje = kp[:, M["root"]] - kp[:, M["codo"]]
        mhr = 0.5 * (KP[:, rival, HOMBROS[0]] + KP[:, rival, HOMBROS[1]])
        mcr = 0.5 * (KP[:, rival, CADERAS[0]] + KP[:, rival, CADERAS[1]])
        blanco = 0.5 * (mhr + mcr) - kp[:, M["root"]]
        with np.errstate(invalid="ignore", divide="ignore"):
            out["apunta"][:, lado] = ((eje * blanco).sum(axis=1)
                                      / np.maximum(np.linalg.norm(eje, axis=1)
                                                   * np.linalg.norm(blanco, axis=1), 1e-9))
    sep = np.abs(out["cx"][:, 0] - out["cx"][:, 1]) / tm
    ra = CL.MANOS[manoq[0]]["root"]
    rb = CL.MANOS[manoq[1]]["root"]
    manos = _largo(KP[:, 0, ra], KP[:, 1, rb]) / tm
    alto = np.stack([(B[:, l, 3] - B[:, l, 1]) * H for l in (0, 1)], axis=1)
    pxm = np.nanmean(alto, axis=1) / CL.ALTURA_M
    with np.errstate(invalid="ignore", divide="ignore"):
        alcance = _largo(KP[:, 0, ra], KP[:, 1, rb]) / pxm
    """`alcance` es la distancia ENTRE LAS DOS MANOS ARMADAS EN METROS, y la escala sale de lo que ya
    usa el proyecto: el alto de la caja partido por una estatura de 1,75 m. Es la unica variable de
    este modulo que tiene unidades fisicas, y esta aqui porque permite poner una puerta que **no es
    un umbral ajustado, es una cota del reglamento**: la hoja de florete mide 0,90 m como maximo
    (`cono_lib.HOJA_M`, que es la cota de la FIE), asi que dos puntas no pueden encontrarse a mas de
    dos hojas de distancia entre manos. Vease `ALCANCE_MAX`."""
    suave = {}
    for k in ("ext_hr", "codo", "pies"):
        cols, dcols = [], []
        for l in (0, 1):
            xv = np.where(val[:, l], out[k][:, l], np.nan)
            xp, _n = _puentea(xv, HUECO_PUENTE)
            cols.append(suaviza(xp))
            dcols.append(suaviza(xp, deriv=1))
        suave[k] = np.stack(cols, axis=1)
        suave["d" + k] = np.stack(dcols, axis=1)
    """LAS DERIVADAS SE CALCULAN SOBRE LA SERIE VALIDA Y CON LOS HUECOS CORTOS PUENTEADOS, y las dos
    cosas hacen falta:

    VALIDA, porque una pose interpolada mete un tramo recto en la serie y una recta tiene derivada
    constante: el arranque del brazo, que es un codo en la curva, desaparece justo donde se busca.

    PUENTEADA, porque el filtro trabaja por tramos contiguos y **un solo cuadro que falta parte la
    subida en dos**. Medido en `TestyWatchfulBarebirdbat`: falta la pose del c301 y con eso el salto
    de `ext_hr` de 1,26 a 1,66 entre c300 y c302 queda invisible para la derivada, porque los dos
    valores caen en tramos distintos. El ataque del izquierdo, verificado a ojo, no se detectaba por
    un cuadro. Puentear no inventa una pose: rellena un hueco de un escalar para poder derivar, y el
    cuadro puenteado sigue sin poder CONCEDER prioridad, que es donde estaba el peligro."""
    out.update({"sep": sep, "manos": manos, "alcance": alcance, "val": val, "mano": manoq,
                "detalle_mano": fuera, "T": T, "torso_medio": tm})
    out.update(suave)
    return out


EXT_DISPARA = 0.0375
"""Derivada de `ext_hr` que DISPARA un arranque de brazo, en torsos por cuadro.

Es el **p90 de la distribucion real** de `dext_hr` sobre 9.961 cuadros x lado validos de 21 clips de FLORETE
con poses de 308 puntos, medido en `p1_variables.py`. No es un numero elegido a mano: el criterio
«p90» estaba declarado por escrito en `04_documentacion/prioridad_fie.md` antes de medirlo.

La distribucion, para que se vea de donde sale:

  | p5 | p25 | p50 | p75 | p90 | p95 | p99 |
  |---:|---:|---:|---:|---:|---:|---:|
  | -0,060 | -0,014 | -0,001 | 0,013 | **0,038** | 0,061 | 0,122 |"""
EXT_SIGUE = 0.0125
"""Y este es el p75. HISTERESIS, con el patron ya validado de `transiciones.py`
(`UMBRAL_DISPARA` 2,50 contra `UMBRAL_SIGUE` 1,90): un umbral alto para DISPARAR y uno mas bajo para
SEGUIR. Sin los dos, una senal que roza el limite entra y sale del tramo cuadro a cuadro, y el
arranque del brazo sale picado en tres trozos en vez de uno."""
PIES_DISPARA = 0.1069
"""p90 de `dpies`, o sea el arranque del a fondo. Mismo criterio y misma medida."""
PIES_SIGUE = 0.0377
EXT_ALTO = 1.0913
"""p75 de `ext_hr`: por encima de esto el brazo esta EXTENDIDO. Es el umbral de t.83.2.d, que
distingue ataque de preparacion."""
TIEMPO_ESGRIMA = 5
"""EL TIEMPO DE ESGRIMA de t.8 —«la duracion de ejecucion de una accion simple»— MEDIDO en nuestros
clips, no copiado de ningun sitio. El reglamento lo define y no lo pone en segundos, asi que hay que
medirlo.

Se mide como la mediana del largo del tramo de subida de `ext_hr`, que es exactamente «una accion
simple» en el sentido de t.8. Sobre 234 tramos de 42 clip x lado:

  | p25 | p50 | p75 | p90 | max |
  |---:|---:|---:|---:|---:|
  | 3 | **5** | 8 | 10 | 22 |

Cinco cuadros a 30 fps son **167 ms**, que cae dentro del orden de magnitud que se le atribuye a una
parada-respuesta. Aqui esta medido en nuestros datos y no supuesto."""
PERMANENCIA = 5
"""EL INTERVALO DE GRACIA que pidio el usuario: *"cuando la pierde, cuando la gana el otro, con el
intervalo que habra, un intervalo de gracia, porque si no estaria cambiando infinitamente todo el
rato; habra un margen, miralo y lo estudias tambien"*.

Una vez concedida, la prioridad **no puede cambiar de manos antes de `PERMANENCIA` cuadros**.

DE DONDE SALE EL NUMERO, y no es a ojo: es el **tiempo de esgrima medido**, 5 cuadros. El argumento
es reglamentario, no estadistico: para que la prioridad cambie de manos hace falta que ocurra una
ACCION (una parada y su respuesta, un arresto con tiempo de esgrima, un ataque sobre una preparacion),
y t.8 dice que una accion simple dura un tiempo de esgrima. Un cambio de prioridad mas rapido que una
accion simple no puede corresponder a ninguna accion: es ruido de la senal.

CONTRASTE CON LA VERDAD DE REFERENCIA: en las 21 ventanas anotadas a mano hay **6 cambios de
prioridad en 14,7 s**, o sea **0,41 cambios por segundo**, y el intervalo mas corto entre dos cambios
anotados es de **9 cuadros** (`GentleQuerulousElephantseal` c279 y c288). Con `PERMANENCIA = 5` no se
pierde ninguno de los 6, y el margen al mas apretado es de 1,8x. La cifra de cuantos cambios quita la
histeresis esta en `p2_prioridad_mide.py`."""


PIES_ALTO = 2.31
"""Separacion de pies que hay que ALCANZAR para que una subida de `pies` sea un a fondo y no un paso.

Es el p90 de `pies` sobre 9.478 cuadros x lado. Y no es un adorno: **es la correccion del fallo mas
grave de la primera version del detector.**

QUE PASABA: `pies` es la distancia entre tobillos, y esa distancia crece en CUALQUIER paso adelante,
no solo en el a fondo. Contando cada subida de `pies` como un a fondo, el a fondo salia SIEMPRE antes
que la extension del brazo —porque el tirador lleva tres pasos avanzando antes de tirarse— y el
detector declaraba **«ataque mal iniciado» en 23 de los cuadros anotados** y no concedia la prioridad
a nadie. En `TestyWatchfulBarebirdbat` la subida de pies empieza en c292 y la del brazo en c300, y el
ataque del izquierdo, que esta verificado a ojo, quedaba anulado por un paso.

Y el reglamento ya distingue las dos cosas, en t.83.2.c: *«el ataque por paso y a fondo es ejecutado
correctamente cuando la extension del brazo precede EL FINAL DE LA MARCHA y el inicio del a fondo»*.
O sea que la marcha y el a fondo son sucesos distintos, y confundirlos es un error de lectura del
reglamento, no un umbral mal puesto."""
PIES_AMPL = 0.50
"""Amplitud minima de la subida de `pies`, en torsos, para que sea un a fondo. Segunda condicion del
mismo arreglo: un a fondo separa los pies mucho, un paso poco."""
CIERRE_P90 = 1.48
"""Distancia que un ataque CIERRA entre las dos manos, en metros. p90 de 96 subidas de brazo de 22
clips: p10 0,02 · p50 **0,48** · p75 0,85 · **p90 1,48** · max 1,91."""
ALCANCE_MAX = 2.0 * CL.HOJA_M + CIERRE_P90
"""Distancia maxima entre las dos manos armadas, en metros, para que una amenaza sea posible.

MITAD COTA DEL REGLAMENTO Y MITAD MEDIDA, y las dos partes hacen falta:

  la COTA: la hoja de florete de la FIE mide **0,90 m** como maximo (`cono_lib.HOJA_M`), asi que en
  el momento del tocado dos puntas no pueden encontrarse con las manos a mas de **dos hojas**.
  la MEDIDA: pero el ataque no empieza a distancia de tocado, **la cierra**. Y cuanto cierra esta
  medido, no supuesto: p90 de **1,48 m** sobre 96 subidas de brazo.

PRIMERO LO PUSE MAL Y LA MEDIDA LO DELATO. Con la cota sola, 1,80 m, el detector pasaba de acertar el
25,6 % de los cuadros con dueno a acertar el **3,3 %**: la puerta tapaba casi todos los ataques
buenos, porque al arrancar el ataque las manos estan a 1,87 m de mediana y solo el 47,9 % de las
subidas empiezan por debajo de 1,80 m. La cota era correcta **para el instante del tocado** y yo la
estaba aplicando **al instante del arranque**. La cifra que lo arregla es la distancia que se cierra.

Y la regla que obliga a poner la puerta es t.9.1, que define el ataque como la accion que **«amenaza
continuamente la superficie valida del adversario»**: a cinco metros no se amenaza nada, asi que no
puede haber ataque ni prioridad.

QUE AVERIA ARREGLA, y la encontre buscandola a proposito en el video de revision, no esperando a que
apareciera. `AliveIgnorantBarebirdbat` c213:

  - el detector daba **PRIORIDAD: DERECHA con confianza 1,00**
  - el tirador de la izquierda **NO ESTA EN EL PLANO**: solo se ve una tira suya en el borde
  - el de la derecha **vuelve andando a su linea con el brazo colgando**
  - y la razon que escribia era «t.83.2.a extension del brazo, sin fondo todavia»

Prioridad con confianza maxima en un cuadro donde no hay ni accion ni adversario. La imagen esta en
`02_validacion/prioridad/fig_AliveIgnorantBarebirdbat_c0213.png`."""
EXIGE_ORDEN_FONDO = True
"""Si se exige la condicion de orden de t.83.2.a: que la extension del brazo PRECEDA al a fondo.

ESTE INTERRUPTOR NO ES UNA OPCION DE GUSTO, es la unica forma de medir cuanto cuesta la condicion
mas dudosa del detector. Y hace falta porque `pies` **no separa el a fondo de la marcha**, con
cifras:

  | accion, verificada a ojo | pico de `pies` |
  |---|---:|
  | a fondo de `GentleQuerulousElephantseal` c281-c285 | 3,41 |
  | a fondo de `TestyWatchfulBarebirdbat` c299-c305 | 3,58 |
  | a fondo de `FirsthandSoreGnat` c308-c311 | 3,22 |
  | **flecha corriendo** de `AliveIgnorantBarebirdbat` c268-c281 | **2,54** |
  | p90 agrupado, que es el umbral `PIES_ALTO` | **2,31** |

O sea que el umbral que salio del p90 cae POR DEBAJO de la banda del que va corriendo, y entonces
sus pasos cuentan como a fondo, empiezan antes que la extension del brazo, y el detector declara
«ataque mal iniciado» sobre un ataque que a ojo es correcto. En la flecha los pies **se cruzan** en
vez de separarse, asi que la separacion de tobillos no es un indicador de a fondo para esa accion.

NO SE SUBE EL UMBRAL A 2,9 —que separaria los cuatro casos de la tabla— porque eso seria elegir el
numero mirando la anotacion, y este proyecto ya se prohibio por escrito bajar el umbral de fundidos
de 1,60 a 1,55 para que un control pasara. Queda como HUECO DECLARADO, con su siguiente paso escrito
en `04_documentacion/prioridad_fie.md`."""
RETRAE = 0.10
"""Cuanto puede bajar `ext_hr` en torsos entre dos tramos sin que se considere que el brazo se ha
ENCOGIDO. Por debajo de esto la pausa es una meseta y los dos tramos son el mismo ataque (t.83.2.b);
por encima, el brazo se encogio y t.83.2.d dice que eso ya no es un ataque."""
EXT_AMPL = 0.30
"""Amplitud minima de la subida de `ext_hr`, en torsos, para que sea un ATAQUE y no ruido de juego de
piernas. Con el p90 de la derivada como unico requisito saltaban del orden de 11 arranques por
tirador y por clip, cuando en un clip hay una o dos acciones. Y el reglamento vuelve a dar el
criterio: t.83.2.d dice que lo hecho **con el brazo encogido no es un ataque sino una preparacion**,
asi que una subida que acaba con el brazo aun encogido no es un ataque por definicion. De ahi la
segunda condicion, que el pico pase de `EXT_ALTO`."""


def _tramos_utiles(s, lado, cual, disp, sigue, alto, ampl):
    """QUE HACE: los tramos de subida que ADEMAS acaban alto y suben lo bastante.

    Devuelve `(ini, medio, fin, pico)`. `medio` es el cuadro en que la senal cruza la mitad de su
    recorrido, y es el que concede la prioridad; `ini` es el que se compara con el a fondo.

    POR QUE DOS INSTANTES Y NO UNO, que es el segundo arreglo y sale de comparar con la anotacion:
    `ini` es el primer milimetro de movimiento y **nadie lo ve**. Mi anotacion a ojo marcaba el
    cambio una mediana de 4-7 cuadros mas tarde que `ini`, y con tolerancia +-3 fallaban todos los
    instantes aunque el lado fuera correcto. El reglamento resuelve la ambiguedad: t.9.1 pide que la
    punta **amenace continuamente** la superficie valida, y amenazar exige el brazo ya extendido, no
    empezando a extenderse. `medio` es la lectura fiel de eso; `ini` sigue siendo el que decide el
    ORDEN contra el a fondo, porque t.83.2.a compara los INICIOS de los dos movimientos.
    """
    from p1_variables import tramos_subida
    x = s[cual][:, lado]
    val = s["val"][:, lado]
    xv = np.where(val, x, np.nan)
    d = s["d" + cual][:, lado]
    raw = tramos_subida(d, np.isfinite(d), disp, sigue, minimo=2)
    """EL TRAMO NO SE PARTE POR UN CUADRO SIN POSE, y esta es la tercera version de esta linea.

    Antes se exigia `val` cuadro a cuadro para construir el tramo, y con eso **un solo cuadro sin
    pose partia la subida en dos trozos cortos** que no llegaban a la amplitud minima. Medido en los
    tres ataques que se escapaban:

      | clip | subida real de `ext_hr` | lo que veia el detector |
      |---|---|---|
      | `TestyWatchfulBarebirdbat` c298-c302 | 1,19 -> 1,66 (0,47) | dos trozos de 0,05 y 0,00 |
      | `AliveIgnorantBarebirdbat` c281-c287 | 0,42 -> 1,27 (0,85) | un trozo de 0,09 |
      | `FirsthandSoreGnat` c299-c311 | 0,42 -> 1,48 (1,06) | nada |

    La condicion de dato medido no desaparece, se mueve al sitio correcto: el tramo tiene que tener
    al menos DOS cuadros con pose medida, y el cuadro que CONCEDE la prioridad tiene que ser uno de
    ellos. Un cuadro puenteado puede formar parte del movimiento; no puede decidir."""
    fus = []
    for a, b, pico in raw:
        if fus:
            a0, b0, p0 = fus[-1]
            entre = xv[b0:a + 1]
            base = xv[b0] if np.isfinite(xv[b0]) else np.nanmax(xv[a0:b0 + 1])
            sin_encoger = (np.isfinite(entre).any() and np.isfinite(base)
                           and float(np.nanmin(entre)) >= float(base) - RETRAE)
            if (a - b0) <= TIEMPO_ESGRIMA and sin_encoger:
                """FUSION DE TRAMOS, y el reglamento la describe literalmente. t.83.2.b: el ataque
                compuesto es correcto *«cuando extendiendo el brazo en la presentacion de la primera
                finta, la punta amenaza la superficie valida SIN ENCOGER EL BRAZO durante la
                ejecucion de los movimientos sucesivos»*. O sea que una pausa sin retraer el brazo
                **sigue siendo el mismo ataque**, y partirla en dos es un error de lectura.

                Sin esto, en `KindEnchantingKid` la subida del derecho c219-c233 se partia en c227
                por una meseta de dos cuadros: el primer trozo llegaba a 0,94 de `ext_hr`, se quedaba
                por debajo de `EXT_ALTO` y **el ataque entero se descartaba**."""
                fus[-1] = (a0, b, max(p0, pico))
                continue
        fus.append((a, b, pico))
    out = []
    for a, b, _p in fus:
        seg = xv[a:b + 1]
        if int(np.isfinite(seg).sum()) < 2:
            continue
        a0 = a
        for j in range(a, max(-1, a - TIEMPO_ESGRIMA) - 1, -1):
            if not np.isfinite(xv[j]):
                break
            if xv[j] <= xv[a0]:
                a0 = j
            else:
                break
        """LA AMPLITUD SE MIDE DESDE LA BASE DE LA SUBIDA, no desde el primer cuadro del tramo, y
        esto tambien se escapaba: el tramo empieza donde la DERIVADA cruza el umbral, que es un poco
        despues de donde el brazo estaba mas encogido. En `AliveIgnorantBarebirdbat` el ataque del
        derecho sube de 0,85 a 1,18 —amplitud 0,33 y por tanto ataque— pero el tramo arrancaba en
        0,92 y la amplitud salia 0,26, por debajo del minimo, asi que se descartaba por 0,04."""
        x0 = float(np.nanmin(xv[a0:b + 1]))
        xm = float(np.nanmax(xv[a0:b + 1]))
        if xm < alto or (xm - x0) < ampl:
            continue
        obj = x0 + 0.5 * (xm - x0)
        med = next((k for k in range(a0, b + 1) if np.isfinite(xv[k]) and xv[k] >= obj), None)
        if med is None:
            continue
        out.append((a0, med, b, xm))
    return out


def _arranques(s, lado):
    """QUE HACE: los tramos de extension de brazo y de a fondo de ese tirador, ya filtrados.

    Lo que decide si el ataque es correcto es COMPARAR el `ini` del tramo de brazo con el `ini` del
    tramo de a fondo mas cercano, que es literalmente lo que pide t.83.2.a.
    """
    ex = _tramos_utiles(s, lado, "ext_hr", EXT_DISPARA, EXT_SIGUE, EXT_ALTO, EXT_AMPL)
    pi = _tramos_utiles(s, lado, "pies", PIES_DISPARA, PIES_SIGUE, PIES_ALTO, PIES_AMPL)
    return ex, pi


def _fondo_asociado(t_ext, pies, ventana=None):
    """QUE HACE: busca el arranque de a fondo que corresponde a esa extension de brazo.

    Se busca el tramo de pies que arranca mas cerca de `t_ext`, dentro de +-2 tiempos de esgrima.
    Fuera de esa ventana el fondo es de otra accion y emparejarlos seria inventar una relacion.
    """
    if ventana is None:
        ventana = 2 * TIEMPO_ESGRIMA
    mejor, d = None, 1e9
    for a, _m, _b, _p in pies:
        if abs(a - t_ext) < d and abs(a - t_ext) <= ventana:
            mejor, d = a, abs(a - t_ext)
    return mejor


def decide(s, mask, T) -> dict:
    """QUE HACE: por cada cuadro, quien tiene la prioridad, POR QUE REGLA y con cuanta confianza.

    LA RAZON ES OBLIGATORIA y no es documentacion: una prioridad sin regla detras **no se puede
    contradecir ni corregir**, y este proyecto se construye a base de poder contradecir al algoritmo.
    Si sale «derecha» sin mas, nadie puede decir en que se equivoco.

    LA MAQUINA DE ESTADOS, en el orden en que decide:

      1. ARRANQUE. Un tirador empieza a extender el brazo (`dext_hr` pasa `EXT_DISPARA`).
      2. CORRECCION t.83.2.a. Se busca su arranque de a fondo. Si la extension PRECEDE al fondo, o si
         no hay fondo todavia, el ataque esta bien iniciado. Si el fondo va primero, esta mal
         iniciado y NO concede prioridad.
      3. SIMULTANEIDAD t.89.2. Si los dos arrancan dentro de un tiempo de esgrima, no hay prioridad
         para ninguno. No es un fallo del detector: es lo que dice el reglamento.
      4. PREPARACION t.83.2.d y t.84.3. Si el otro estaba avanzando con el brazo encogido, el ataque
         va sobre una preparacion y la razon es t.84.3, que es mas fuerte.
      5. CONTRAATAQUE t.89.4.a. Si el otro YA tiene la prioridad, extender el brazo no la quita: es
         un arresto y sigue sin prioridad.
      6. PERDIDA t.89.5.d. Al que la tiene se le retira si `ext_hr` **baja** por debajo del nivel de
         donde arranco mas la mitad de su recorrido, que es «el ataque que se acorta o se detiene».
      7. GRACIA. Nada de lo anterior puede cambiar de manos la prioridad antes de `PERMANENCIA`
         cuadros. Lo que si puede pasar antes es PERDERLA sin darsela a nadie, y esa asimetria es a
         proposito: perder la prioridad por fallo propio es un hecho del que la tiene, mientras
         ganarla exige una accion del otro, y una accion dura un tiempo de esgrima.

    QUE DEVUELVE
      `prio` (T,) -1 ninguna, 0 izquierda, 1 derecha
      `regla` lista de T cadenas con el articulo y su texto corto
      `conf` (T,) confianza en [0, 1]
      `eventos` lista de (cuadro, prio_nueva, regla) para poder auditar los cambios
    """
    prio = np.full(T, -1, dtype="int8")
    regla = ["" for _ in range(T)]
    conf = np.zeros(T, dtype="float64")
    ar = {l: _arranques(s, l) for l in (0, 1)}
    tramo_de = {l: {m: (a, b, p) for a, m, b, p in ar[l][0]} for l in (0, 1)}
    """La decision se dispara en el cuadro `medio` de cada tramo, y de ahi se recupera su `ini` para
    el test de orden contra el a fondo."""
    concede = {l: sorted(tramo_de[l]) for l in (0, 1)}
    inicios = {l: [a for a, _m, _b, _p in ar[l][0]] for l in (0, 1)}
    eventos = []
    estado, desde = -1, -10 ** 6
    dueno_ext0, dueno_pico, razon = None, None, "ninguna: nadie ha iniciado un ataque correcto"
    pendiente = None
    """`pendiente` es un arranque correcto que llego DENTRO del intervalo de gracia. No se tira: se
    guarda y se concede en cuanto la gracia expira, si el brazo sigue extendido. Tirarlo perderia
    ataques buenos que arrancan justo despues de un cambio, y guardarlo sin caducidad concederia
    prioridad por un movimiento ya terminado; caduca en un tiempo de esgrima."""

    def _califica(l, k):
        """La regla que concede, o None si ese arranque no concede prioridad.

        `k` es el cuadro de media subida, que es el que concede; el orden contra el a fondo se juzga
        con el `ini` del tramo, que es lo que compara t.83.2.a.
        """
        otro = 1 - l
        al = s["alcance"][k]
        if not np.isfinite(al):
            return None, "t.9.1 sin dato de alcance: no se puede afirmar que haya amenaza"
        if al > ALCANCE_MAX:
            return None, ("t.9.1 no hay amenaza posible: %.1f m entre manos y la hoja mide %.2f m"
                          % (al, CL.HOJA_M))
        t_i = tramo_de[l][k][0] if k in tramo_de[l] else k
        t_f = _fondo_asociado(t_i, ar[l][1]) if EXIGE_ORDEN_FONDO else None
        if t_f is not None and t_f < t_i:
            return None, "t.83.2.a ataque MAL iniciado: el fondo precede a la extension"
        avanza = np.isfinite(s["vx"][k, otro]) and s["vx"][k, otro] > 0
        enc = np.isfinite(s["ext_hr"][k, otro]) and s["ext_hr"][k, otro] < EXT_ALTO
        if avanza and enc:
            return l, "t.84.3 ataque sobre la preparacion del adversario"
        if t_f is None:
            return l, "t.83.2.a extension del brazo, sin fondo todavia"
        return l, "t.83.2.a extension del brazo antes del inicio del fondo"

    for k in range(T):
        if not mask[k]:
            prio[k] = -1
            regla[k] = "cuadro no utilizable"
            continue
        """PERDIDA PRIMERO, porque puede actuar DENTRO del intervalo de gracia, y esa asimetria es a
        proposito: perder la prioridad por fallo propio es un hecho del que la tiene, mientras
        ganarla exige una accion del otro, y una accion dura un tiempo de esgrima (t.8)."""
        if estado >= 0 and np.isfinite(s["alcance"][k]) and s["alcance"][k] > ALCANCE_MAX:
            """SE PIERDE LA PRIORIDAD AL PERDERSE EL ALCANCE, y hace falta por simetria con la
            puerta que la concede: si a mas de dos hojas no se puede amenazar, tampoco se puede
            seguir amenazando. Sin esto la prioridad se quedaba pegada cien cuadros despues de que
            la accion acabara y los dos se fueran andando a su linea, que es la mitad de la tasa de
            falsa alarma."""
            razon = ("t.9.1 la amenaza deja de ser posible: %.1f m entre manos"
                     % s["alcance"][k])
            eventos.append((k, -1, razon))
            estado, desde = -1, k
            dueno_ext0, dueno_pico, pendiente = None, None, None
        if estado >= 0 and dueno_ext0 is not None and dueno_pico is not None:
            e = s["ext_hr"][k, estado]
            ant = s["ext_hr"][k - 1, estado] if k > 0 else np.nan
            if np.isfinite(e):
                rec = max(dueno_pico - dueno_ext0, 1e-6)
                cae = (dueno_pico - e) / rec > 0.5
                cae_ant = np.isfinite(ant) and (dueno_pico - ant) / rec > 0.5
                if cae and cae_ant and s["val"][k, estado]:
                    """DOS CUADROS SEGUIDOS, no uno. Con un solo cuadro, un punto de pose con ruido
                    tiraba la prioridad en medio del ataque: en `KindEnchantingKid` la quitaba en
                    c227, ocho cuadros despues de concederla y en pleno fondo del atacante."""
                    razon = "t.89.5.d el ataque se acorta o se detiene"
                    eventos.append((k, -1, razon))
                    estado, desde = -1, k
                    dueno_ext0, dueno_pico = None, None
                else:
                    dueno_pico = max(dueno_pico, e)
        arranca = [l for l in (0, 1) if k in concede[l]]
        simul = len(arranca) == 2
        if not simul and arranca:
            l = arranca[0]
            mio = tramo_de[l][k][0]
            simul = any(abs(j - mio) < TIEMPO_ESGRIMA for j in inicios[1 - l])
        if simul:
            razon = "t.89.2 accion simultanea, no hay prioridad"
            eventos.append((k, -1, razon))
            estado, desde = -1, k
            dueno_ext0, dueno_pico, pendiente = None, None, None
        elif arranca:
            l = arranca[0]
            if estado == l:
                razon = "t.83.2.a el atacante sigue extendiendo (reposicion, t.13.1)"
            elif estado == (1 - l):
                razon = "t.89.4.a contraataque: no quita la prioridad al atacante"
            else:
                quien, r = _califica(l, k)
                if quien is None:
                    razon = r
                elif (k - desde) < PERMANENCIA:
                    """INTERVALO DE GRACIA: no se concede todavia. Se anota y se reintenta."""
                    pendiente = (l, k, r)
                    razon = "intervalo de gracia: arranque correcto en espera (%s)" % r
                else:
                    razon = r
                    eventos.append((k, l, r))
                    estado, desde = l, k
                    dueno_ext0 = float(s["ext_hr"][k, l]) if np.isfinite(s["ext_hr"][k, l]) else None
                    dueno_pico = dueno_ext0
                    pendiente = None
        if pendiente is not None:
            l, k0, r = pendiente
            if k - k0 > TIEMPO_ESGRIMA:
                pendiente = None
            elif estado < 0 and (k - desde) >= PERMANENCIA:
                e = s["ext_hr"][k, l]
                if np.isfinite(e) and e >= EXT_ALTO:
                    razon = r + " (concedida al expirar el intervalo de gracia)"
                    eventos.append((k, l, razon))
                    estado, desde = l, k
                    dueno_ext0, dueno_pico = float(e), float(e)
                    pendiente = None
        prio[k] = estado
        regla[k] = razon if (estado >= 0 or razon.startswith(("t.89", "t.83.2.a ataque MAL",
                                                              "intervalo"))) \
            else "ninguna: nadie ha iniciado un ataque correcto"
        if estado >= 0:
            """LA CONFIANZA baja por tres cosas distintas y cada una tiene su motivo:
            que la pose que sostiene la decision no sea MEDIDA, que el margen temporal contra el
            arranque del otro sea menor que un tiempo de esgrima, y que el rival no tenga dato."""
            c = 1.0
            if not s["val"][k, estado]:
                c *= 0.3
            if not s["val"][k, 1 - estado]:
                c *= 0.6
            otros = [j for j in concede[1 - estado] if abs(j - desde) <= 2 * TIEMPO_ESGRIMA]
            if otros:
                m = min(abs(j - desde) for j in otros)
                c *= min(1.0, m / float(TIEMPO_ESGRIMA))
            conf[k] = c
        else:
            conf[k] = 1.0 if (s["val"][k].all()) else 0.4
    return {"prio": prio, "regla": regla, "conf": conf, "eventos": eventos,
            "arranques": {l: inicios[l] for l in (0, 1)}}


def paradas(s, mask, T, res) -> list:
    """LA PARADA, que es la bisagra del reglamento: t.86 «la parada da derecho a la respuesta».

    QUE PIDIO EL USUARIO: *"que detectamos bien las paradas"*. Es el pedido correcto, porque la
    parada es el mecanismo por el que la prioridad **cambia de manos**, y sin ella un detector solo
    sabe repartirla al principio.

    LO QUE NO SE HACE AQUI, Y POR QUE. La parada es, por definicion, **contacto entre dos hojas**, y
    la hoja no se mide: se detecta en el 42,6 % de los recortes a 1080p y en el **1,6 % a 720p**,
    donde su fuerte ocupa 0,64 px. Diecinueve de los veinte clips con poses de este proyecto son
    720p.

    Y hay una tentacion que este proyecto tiene PROHIBIDA por escrito: prolongar 90 cm de hoja desde
    la mano siguiendo el eje del antebrazo y mirar si las dos rectas se cruzan. El documento maestro
    lo dice sin rodeos: *"No dibujar la hoja donde no se ha visto. Un cuaderno anterior trazaba 95 px
    desde la muneca y metia esa hoja inventada en la etiqueta de tocado. Eso no vuelve a entrar."* Y
    hay una medida que lo condena: el angulo entre la hoja real y el eje del antebrazo pide una
    semiapertura de 24,8-29,7 grados, o sea **2,4-2,8 veces** el techo de utilidad del cono. Una hoja
    dibujada sobre el eje del antebrazo apunta a donde no esta.

    ASI QUE LO QUE SE DETECTA NO ES LA PARADA, ES SU CONSECUENCIA, y el nombre lo dice: lo que t.86
    convierte en regla no es el choque, es que **el que para adquiere el derecho a responder**. Ese
    traspaso si esta hecho de cosas medidas.

    CRITERIO DECLARADO ANTES DE MEDIR. Candidato a parada en el cuadro `k` si se cumplen las TRES:

      1. DISTANCIA DE HIERRO. `manos[k]` por debajo de su p25 en el clip. Es la unica condicion de
         «las hojas podrian estar en contacto» que no exige inventar la hoja: si las manos estan
         lejos, dos hojas de 90 cm no se tocan. Es necesaria y no suficiente, y se dice.
      2. EL ATAQUE SE PARA. El que tenia la prioridad pierde mas de la mitad del recorrido de
         `ext_hr` que habia ganado, dentro de un tiempo de esgrima desde `k`.
      3. LA RESPUESTA ARRANCA. El otro empieza a extender el brazo (`dext_hr` pasa `EXT_DISPARA`)
         dentro de DOS tiempos de esgrima desde `k`. Es la «respuesta inmediata, sin indecision ni
         tiempo de detencion» que exige t.86, y el limite de dos tiempos de esgrima es la lectura
         literal de «inmediata».

    LO QUE ESTE DETECTOR CONFUNDE POR CONSTRUCCION, dicho antes de medirlo: un ataque que se queda
    corto por si solo seguido de una reposicion del adversario da las tres condiciones sin que haya
    habido ninguna parada. No hay forma de separarlos sin ver el hierro.
    """
    cand = []
    m = s["manos"][mask] if mask.any() else np.array([])
    m = m[np.isfinite(m)]
    if not len(m):
        return cand
    umbral = float(np.percentile(m, 25))
    prio = res["prio"]
    for k in range(T):
        if not mask[k] or prio[k] < 0:
            continue
        if not (np.isfinite(s["manos"][k]) and s["manos"][k] <= umbral):
            continue
        a, d = prio[k], 1 - prio[k]
        fin = min(T, k + TIEMPO_ESGRIMA + 1)
        e = s["ext_hr"][k:fin, a]
        e = e[np.isfinite(e)]
        if len(e) < 2:
            continue
        rec = float(np.nanmax(s["ext_hr"][max(0, k - 3 * TIEMPO_ESGRIMA):k + 1, a])
                    - np.nanmin(s["ext_hr"][max(0, k - 3 * TIEMPO_ESGRIMA):k + 1, a]))
        if not np.isfinite(rec) or rec <= 1e-6:
            continue
        if (float(e[0]) - float(e.min())) / rec <= 0.5:
            continue
        f2 = min(T, k + 2 * TIEMPO_ESGRIMA + 1)
        dd = s["dext_hr"][k:f2, d]
        vv = s["val"][k:f2, d]
        if not np.any(np.isfinite(dd) & vv & (np.nan_to_num(dd) >= EXT_DISPARA)):
            continue
        cand.append({"cuadro": k, "paraba": int(d), "atacaba": int(a),
                     "manos": float(s["manos"][k]), "umbral_manos": umbral,
                     "caida_rel": float((float(e[0]) - float(e.min())) / rec)})
    fusion = []
    for c in cand:
        if fusion and c["cuadro"] - fusion[-1]["cuadro"] <= TIEMPO_ESGRIMA:
            continue
        fusion.append(c)
    return fusion


def decide_sin_gracia(s, mask, T) -> dict:
    """LA MISMA DECISION SIN INTERVALO DE GRACIA, para poder medir lo que aporta.

    Existe solo para contarla: sin esto, el margen de `PERMANENCIA` seria un numero inventado. La
    comparacion que lo justifica es «cambios de prioridad por segundo con y sin gracia, contra los
    cambios que hay de verdad en la anotacion a mano», y esta en `p2_prioridad_mide.py`.
    """
    global PERMANENCIA
    v = PERMANENCIA
    PERMANENCIA = 0
    try:
        return decide(s, mask, T)
    finally:
        PERMANENCIA = v


def linea_base(B, pres, mask, T):
    """LA LINEA BASE TONTA A PROPOSITO, que hay que batir: «tiene prioridad el que empieza a avanzar
    primero», con la x del centro de la caja y nada mas.

    No usa poses, ni brazos, ni pies: solo `B`. Existe porque en este proyecto una senal no vale
    hasta que bate a la version gratis de si misma, y porque el precedente ya esta escrito: el orden
    de las luces acierta el 63,51 % por si solo y hundio el techo del modelo al 73,6 %.

    Devuelve `(prio, quien)`: `prio` (T,) con -1 ninguna, 0 izquierda, 1 derecha, y el lado que
    arranco primero. El criterio es el mismo que el del detector para que la comparacion sea justa:
    quien tiene el primer tramo de 3 cuadros seguidos avanzando hacia el rival.
    """
    cx = np.full((T, 2), np.nan)
    for lado in (0, 1):
        c = 0.5 * (B[:, lado, 0] + B[:, lado, 2])
        cx[:, lado] = np.where(pres[:, lado], c, np.nan)
    prio = np.full(T, -1, dtype="int8")
    ini = {}
    for lado in (0, 1):
        signo = 1.0 if lado == 0 else -1.0
        d = signo * suaviza(cx[:, lado], deriv=1)
        pos = np.nan_to_num(d, nan=-1.0) > 0
        run = 0
        for k in range(T):
            if not mask[k]:
                run = 0
                continue
            run = run + 1 if pos[k] else 0
            if run >= 3:
                ini[lado] = k - 2
                break
    if not ini:
        return prio, None
    if len(ini) == 2 and ini[0] == ini[1]:
        return prio, None
    quien = min(ini, key=lambda l: ini[l])
    prio[ini[quien]:] = quien
    prio[~mask] = -1
    return prio, quien

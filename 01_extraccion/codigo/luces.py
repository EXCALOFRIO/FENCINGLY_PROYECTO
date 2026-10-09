"""Detector de LUCES del aparato senializador que NO sabe que torneo esta mirando.

EL PROBLEMA, tal como lo planteo el usuario. En florete el aparato enciende roja o verde (tocado en
superficie valida de cada tirador) y blanca (tocado fuera de superficie valida). En television esas
luces se ven dentro del ROTULO SOBREPUESTO del marcador, y ese rotulo cambia de diseno en cada
torneo y cada cadena. Un detector afinado en un torneo falla en los demas, asi que lo que hay que
construir es uno que no dependa del diseno.

LO QUE SE MIDE Y POR QUE ESO Y NO OTRA COSA. El valor de detectar las luces NO es usarlas como
etiqueta: el proyecto ya tiene medido que el orden de las luces es un ATAJO que el modelo explota
(acierta el 63,51 % con solo ese orden) y que en los 960 clips donde el que encendio primero no gana
acierta el 45,21 %, por debajo del azar. El valor esta en poder SEPARAR los casos en que el orden de
luces engana. Por eso lo que se saca es el INSTANTE de cada canal, CUAL es y en QUE ORDEN, y no un
"hubo tocado".

EL CRITERIO, DECLARADO ANTES DE EJECUTAR. Una luz no se busca por color: se busca por su forma en el
TIEMPO. Un pixel de la bandera de Italia es verde siempre; un pixel de la pista de Belgrado es rojo
siempre; un tirador con chaqueta clara se mueve. Una luz, en cambio, esta apagada casi todo el clip,
se enciende de golpe, se MANTIENE, y se apaga. Asi que para cada pixel de la banda del rotulo y cada
canal de color se mide:

    base   percentil 20 temporal del canal en el clip   (el estado APAGADO)
    pico   maximo temporal                              (el estado ENCENDIDO)
    salto  pico - base

y el pixel es candidato solo si cumple las TRES:

    salto >= SALTO_MIN   cambia de verdad
    base  <= BASE_MAX    estaba apagado
    pico  >= PICO_MIN    y al encenderse esta REALMENTE rojo / verde / blanco

La tercera condicion se anadio despues de medirla ausente: sin ella el detector daba 680 luces en 21
clips (32 por clip), porque un pixel azul del rotulo que pasa a gris tiene `rojez` de -73 a +4, o sea
`salto` 77 y `base` -73, y pasaba los dos primeros filtros sin estar rojo en ningun momento. Con
`pico >= PICO_MIN` eso desaparece. Queda escrito porque es el tipo de fallo que un umbral movido a
posteriori habria tapado.

Despues se exige que la racha de encendido dure entre `DUR_MIN` y `DUR_MAX` segundos, lo que tira los
cambios permanentes (la aparicion del rotulo entero al volver de una repeticion) y los destellos de
un cuadro (compresion, flash de foto), y que el pixel estuviera APAGADO en la muestra inmediatamente
anterior al encendido, que es lo que distingue un ESCALON de una deriva lenta de luz de pabellon.

EL LIMITE DEL UMBRAL ABSOLUTO, MEDIDO, Y EL ARREGLO QUE NO FUNCIONO. El criterio de arriba usa
umbrales ABSOLUTOS (`salto >= 45`, `base <= 30`) y con ellos el detector da el conjunto de luces
exacto en 19 de 21 sedes de la muestra A -- la misma en la que se ajusto -- pero encuentra alguna luz
de color en **solo 15 de 21** clips de la muestra B, sorteada aparte con otra semilla: seis sedes a
cero. Medida la causa en tres de los seis fallos, en la region donde el detector viejo dice que esta
la luz:

  | clip                          | base | pico | salto |
  |-------------------------------|-----:|-----:|------:|
  | `HardtofindBitesizedBactrian` |   62 |   90 |    28 |
  | `PettyCelebratedGyrfalcon`    |   90 |   96 |    11 |
  | `FalseBlissfulCaimanlizard`   |   76 |   90 |    42 |

frente a saltos de 146 a 268 en los clips de la muestra A. O sea que el salto de la MISMA luz, en la
MISMA familia de marcador, varia un factor 10 entre clips. La razon es que **el rotulo FIE es
SEMITRANSPARENTE**: en un pabellon con pista verde o azulada la placa ya sale verdosa de base y la
barra de luz solo anade unas decenas de niveles, mientras que sobre una pista roja el contraste es
enorme. Un umbral absoluto no puede cubrir las dos cosas, y `base <= 30` ademas rechazaba de plano
las placas de base verdosa (base 62 a 90).

EL ARREGLO PROBADO, Y RETIRADO CON SU MEDIDA AL LADO (`USA_UMBRAL_ABSOLUTO = True` deja el codigo
puesto pero apagado, como `USA_CRUDAS = False` en `extrae_pareja.py`). Si el problema es de escala,
lo razonable es comparar el salto con el RUIDO PROPIO de cada pixel en vez de con un numero fijo:

    ruido = percentil 60 - percentil 20 del canal en el clip     (dispersion del estado APAGADO)
    candidato si  salto >= max(SALTO_PISO, K_RUIDO * ruido)   y   pico >= PICO_MIN

con `SALTO_PISO_COLOR` = 20 y `K_RUIDO` = 6,0, y sin condicion de `base`. El percentil 60 y no la
desviacion tipica para que la propia luz no infle la referencia. **No funciono, y sale PEOR en las
dos muestras:**

  | criterio           | muestra A, conjunto exacto | muestra B, detecta color |
  |--------------------|---------------------------:|-------------------------:|
  | absoluto (vigente) |              19 / 21       |          15 / 21 = 71,4 %|
  | relativo al ruido  |              15 / 21       |          14 / 21 = 66,7 %|

y ademas rompio dos sedes donde el absoluto acertaba (`anaheimfoil2018`, `worldchamps2022`). Asi que
la hipotesis "es un problema de escala" queda MEDIDA Y DESCARTADA: normalizar por el ruido del pixel
mete mas falsos que luces recupera, porque en la banda del rotulo hay pixeles con ruido casi nulo
(graficos quietos) donde `6 * ruido` baja el umbral a 20 niveles y entra cualquier cosa. El limite de
este detector NO es el umbral: en 6 de los 8 fallos de la muestra B la luz **no esta entre las
regiones candidatas con ningun umbral de los dos**, o sea que el que falla es el criterio de region,
no el de nivel.

LOS CANALES, definidos para no depender del blanco de referencia de la cadena:

    rojez    = R - max(G, B)
    verdez   = G - max(R, B)
    blancura = min(R, G, B)

EL CUARTO FILTRO, EL ENTORNO. Un pixel de chaqueta blanca que entra en cuadro cumple los tres
filtros de arriba: estaba oscuro, se pone blanco, y se queda. Lo que lo separa de una luz es que
cuando una luz se enciende su ENTORNO no cambia -- el rotulo es un grafico quieto -- mientras que
cuando entra un tirador cambia todo alrededor. Asi que para cada region candidata se mide la
desviacion tipica temporal del gris en un ANILLO alrededor de su caja y se exige
`ring_std <= RING_STD_MAX`. Esta idea tampoco es mia: el detector generico del proyecto viejo
(`fencingly_v2/lights_generic.py`) la tiene medida con overlay estatico ~3-8 y cuerpo o suelo > 20.

  DOS CAMBIOS POST-HOC, dichos como tales. `RING_STD_MAX` se declaro en 12,0 (punto medio de esos
  dos rangos) y con ese valor el detector daba CERO luces en `serbiafoil2022`, que tiene una barra
  verde grande y evidente a ojo. Medido el porque: el anillo de esa barra tiene `ring_std` = 18,3
  porque el rotulo de Serbia es semitransparente sobre una pista ROJA con paneo, mientras que el
  ruido del mismo clip (chaquetas, suelo) esta entre 24 y 58. Asi que el umbral pasa a 22,0. Es un
  ajuste a posteriori sobre una sede del conjunto de AJUSTE, y por eso hay que revalidarlo en una
  muestra independiente; se hace en `f9_muestra_b.py`.

  El segundo cambio es un filtro NUEVO, `FRAC_ANILLO_COLOR_MAX`: se rechaza la region si mas de la
  mitad de su anillo alcanza tambien el color del candidato. Salio de `belgradefoil2022`, donde el
  detector daba una "luz roja" de 3.007 px con `ring_std` 1,8 -- o sea perfectamente quieta -- que
  era la PISTA ROJA vista donde el rotulo es transparente. El anillo de una pista roja es rojo; el
  de una barra roja dentro de una placa azul no lo es. Este filtro es lo que separa una luz de una
  extension de color, y el `ring_std` solo no podia hacerlo porque una pista uniforme tiene
  desviacion temporal BAJA aunque la camara se mueva.

CONVENCION DEL APARATO, que es lo unico universal: rojo es el tirador de la IZQUIERDA y verde el de
la DERECHA. Cuando el color y el lado no concuerdan NO se descarta la luz: se marca
`lado_color_discordante`, porque hay familias de marcador con el indicador CENTRADO (Paris 2022 tiene
un cuadradito unico entre los dos tanteos) y descartarlas seria justo el fallo de generalizacion que
este modulo existe para evitar.

VALORES INICIALES Y COMO SE AJUSTAN. Los umbrales de abajo se declaran aqui y **solo se ajustan
mirando clips de la familia FIE**, que es la mayoritaria. Las otras cuatro familias (Kazan/San
Petersburgo, Tokio 2022, Tokio 2020 OBS, Paris 2022) se dejan FUERA del ajuste y sirven de prueba de
generalizacion. Si no pasan, se informa; no se mueve el umbral para que pasen.

LO QUE ESTE MODULO NO HACE. No dice si el tocado fue valido, ni quien tenia prioridad, ni si el
arbitro lo concedio. Y no distingue una luz del aparato fisico de la pista de una luz del rotulo: si
las dos cumplen el criterio, salen las dos y se separan por posicion.
"""
from __future__ import annotations

import os

import numpy as np

import cv2

ESCALON_VISTO = os.environ.get("FENCINGLY_LUZ_ESCALON_VISTO", "0") == "1"
"""EL CUADRO DEL TOCADO = 0, primer arreglo (regla R3 de `LUCES_POR_FUENTE_2026-10-09.md`): por PIXEL.
Una racha que ya esta encendida en la PRIMERA muestra del clip se aceptaba como escalon (`ini <= 0`),
y es la luz del tocado ANTERIOR que sigue encendida al empezar el clip -- o una bandera del rotulo --:
salia `cuadro_tocado = 0` en el 8 % de los clips de florete en disco. Con esto encendido solo vale un
encendido VISTO. MEDIDO Y APAGADO: en 600 clips al azar C0 baja de 0,068 a 0 y en los 48 que cambian
A1 sube de 0,500 a 0,818, pero NO pasa la puerta 3 declarada: cambia tambien 10 de los 521 clips cuyo
encendido no era 0 (quitar pixeles mueve la fuerza maxima y el ancla del grupo). Lo sustituye
`SIN_CUADRO0`, que solo actua donde el cuadro sale 0."""

SIN_CUADRO0 = os.environ.get("FENCINGLY_LUZ_SIN_CUADRO0", "0") == "1"
"""EL CUADRO DEL TOCADO = 0, segundo arreglo (regla R3b), por LUZ y solo donde hace falta: si el grupo
del tocado empieza en el cuadro 0, se rehace el grupo SIN las luces que ya estaban encendidas al
empezar el clip. Un clip cuyo grupo no empieza en 0 sale identico por construccion. La puerta y la
cifra estan en el informe; el defecto lo pone esa puerta."""

ANCHO_RED = 480
PASO = 2
BANDA_Y = (0.55, 1.00)
BANDA_Y_ALTA = (0.00, 0.25)

USA_UMBRAL_ABSOLUTO = True
SALTO_MIN_COLOR = 45
SALTO_MIN_BLANCO = 60
BASE_MAX_COLOR = 30
BASE_MAX_BLANCO = 150
SALTO_PISO_COLOR = 20
SALTO_PISO_BLANCO = 40
K_RUIDO = 6.0
PICO_MIN_COLOR = 40
PICO_MIN_BLANCO = 235
FRAC_APAGADO = 0.3
FRAC_ENCENDIDO = 0.6
RING_STD_MAX = 22.0
FRAC_ANILLO_COLOR_MAX = 0.50
CONF_MIN = 0.50
VENTANA_TOCADO_S = 0.5
FRAC_FUERTE = 0.5
DUR_MIN_S = 0.35
DUR_MAX_S = 6.0
AREA_MIN = 8
AREA_MAX_FRAC_BANDA = 0.06
ANCHO_MAX_BLANCA = 0.06
ALTO_MAX_BLANCA = 0.08
PCT_BASE = 20

CANALES = ("roja", "verde", "blanca")


def lee_pila(ruta: str, paso: int = PASO, ancho: int = ANCHO_RED,
             banda=BANDA_Y) -> tuple[np.ndarray, np.ndarray, float, int, int, int]:
    """Devuelve (pila BGR reducida de la banda, indices de cuadro, fps, n, W, H) del clip entero.

    Se lee SECUENCIAL y se guarda 1 de cada `paso`. Leer secuencial y tirar cuadros sale mas barato
    que `cap.set` por cuadro cuando se necesita medio clip: el decodificador tiene que reconstruir
    los fotogramas intermedios de todas formas, y cada `set` fuerza una busqueda de cuadro clave.
    Esto NO contradice la economia pedida: la alternativa no es leer menos, es leer lo mismo peor.
    """
    cap = cv2.VideoCapture(ruta)
    if not cap.isOpened():
        raise OSError(f"no abre {ruta}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    esc = ancho / W
    alto_red = int(round(H * esc))
    y0 = int(round(banda[0] * alto_red))
    y1 = int(round(banda[1] * alto_red))
    pila, idx = [], []
    k = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if k % paso == 0:
            red = cv2.resize(img, (ancho, alto_red), interpolation=cv2.INTER_AREA)
            pila.append(red[y0:y1])
            idx.append(k)
        k += 1
    cap.release()
    return np.stack(pila, 0), np.asarray(idx), fps, n, W, H


def canales(pila: np.ndarray) -> dict[str, np.ndarray]:
    """(T,h,w,3) BGR -> tres series (T,h,w) int16 de rojez, verdez y blancura."""
    b = pila[..., 0].astype(np.int16)
    g = pila[..., 1].astype(np.int16)
    r = pila[..., 2].astype(np.int16)
    return {
        "roja": r - np.maximum(g, b),
        "verde": g - np.maximum(r, b),
        "blanca": np.minimum(np.minimum(r, g), b),
    }


def rachas(enc: np.ndarray, dur_min: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Para cada pixel, la ULTIMA racha contigua de True de al menos `dur_min` muestras.

    POR QUE LA ULTIMA Y NO LA MAS LARGA, con la medida que lo obligo. La primera version se quedaba
    con la racha mas LARGA, y eso perdio la luz de `anaheimfoil2018`: ese clip tiene la luz del
    tocado ANTERIOR encendida en sus primeros 46 cuadros y la del tocado propio desde el 167 hasta el
    final, o sea 24 muestras contra 21, y la mas larga era la del tocado anterior. El detector daba
    entonces la luz equivocada con confianza 1,000, que es la peor forma de fallar. Quedarse con la
    ULTIMA es coherente con la regla de agrupacion del tocado, que tambien mira el ultimo encendido.

    Se hace con un barrido acumulado y sin bucles por pixel: `corre` lleva la longitud de la racha
    viva en cada pixel y se actualiza en T pasos vectorizados. T es del orden de 150, asi que el
    bucle en T es barato y el de pixeles (decenas de miles) no existe. Al cerrarse una racha que
    llega a `dur_min` se apunta, sobreescribiendo la anterior; al final del barrido se cierra la
    racha viva, que es el caso normal cuando la luz sigue encendida al acabar el clip.
    """
    T = enc.shape[0]
    forma = enc.shape[1:]
    prev = np.zeros(forma, np.int32)
    largo = np.zeros(forma, np.int32)
    fin = np.full(forma, -1, np.int32)
    for t in range(T):
        corre = np.where(enc[t], prev + 1, 0)
        termina = (corre == 0) & (prev >= dur_min)
        largo = np.where(termina, prev, largo)
        fin = np.where(termina, t - 1, fin)
        prev = corre
    viva = prev >= dur_min
    largo = np.where(viva, prev, largo)
    fin = np.where(viva, T - 1, fin)
    ini = fin - largo + 1
    return largo, ini, fin


def detecta(ruta: str, paso: int = PASO, banda=BANDA_Y,
            salto_color: int = SALTO_MIN_COLOR, salto_blanco: int = SALTO_MIN_BLANCO,
            base_max_color: int = BASE_MAX_COLOR, base_max_blanco: int = BASE_MAX_BLANCO,
            pico_color: int = PICO_MIN_COLOR, pico_blanco: int = PICO_MIN_BLANCO,
            dur_min_s: float = DUR_MIN_S, dur_max_s: float = DUR_MAX_S,
            area_min: int = AREA_MIN) -> dict:
    """Devuelve las luces de un clip: lista de dicts con canal, cuadros, lado, area y confianza.

    La confianza se construye con tres factores medibles y se da desglosada, porque una confianza
    agregada esconde por que un caso es dudoso:

        c_salto   salto / (2 * SALTO_MIN), recortado a 1     -- cuanto destaca sobre su apagado
        c_base    1 - base / BASE_MAX, recortado a [0, 1]    -- cuanto de limpio estaba el apagado
        c_dur     1 si la duracion cae en [dur_min, dur_max] -- cuanto se parece a una luz y no a un corte

    y la confianza final es su producto. No esta calibrada contra ninguna verdad: es un orden, no una
    probabilidad, y se dice asi para que nadie la lea como un porcentaje de acierto.
    """
    pila, idx, fps, n, W, H = lee_pila(ruta, paso, banda=banda)
    T, h, w = pila.shape[:3]
    ch = canales(pila)
    gris = pila.astype(np.float32).mean(axis=3)
    gstd = gris.std(axis=0)
    radio = max(3, int(round(0.02 * w)))
    dur_min = max(1, int(round(dur_min_s * fps / paso)))
    dur_max = max(dur_min + 1, int(round(dur_max_s * fps / paso)))
    area_max = int(AREA_MAX_FRAC_BANDA * h * w)

    luces = []
    for nombre in CANALES:
        s = ch[nombre]
        base = np.percentile(s, PCT_BASE, axis=0)
        pico = s.max(axis=0)
        salto = pico - base
        smin = salto_blanco if nombre == "blanca" else salto_color
        bmax = base_max_blanco if nombre == "blanca" else base_max_color
        pmin = pico_blanco if nombre == "blanca" else pico_color
        if USA_UMBRAL_ABSOLUTO:
            cand = (salto >= smin) & (base <= bmax) & (pico >= pmin)
        else:
            ruido = np.percentile(s, 60, axis=0) - base
            piso = SALTO_PISO_BLANCO if nombre == "blanca" else SALTO_PISO_COLOR
            cand = (salto >= np.maximum(piso, K_RUIDO * ruido)) & (pico >= pmin)
        if not cand.any():
            continue
        umbral = base + FRAC_ENCENDIDO * salto
        enc = (s >= umbral[None]) & cand[None]
        largo, ini, fin = rachas(enc, dur_min)
        u_apagado = base + FRAC_APAGADO * salto
        k_prev = np.clip(ini - 1, 0, T - 1)
        antes = np.take_along_axis(s, k_prev[None], axis=0)[0]
        if ESCALON_VISTO:
            escalon = (ini > 0) & (antes <= u_apagado)
        else:
            escalon = (ini <= 0) | (antes <= u_apagado)
        valido = cand & escalon & (largo >= dur_min) & (largo <= dur_max)
        if not valido.any():
            continue
        nlab, lab, stats, cent = cv2.connectedComponentsWithStats(valido.astype(np.uint8))
        for i in range(1, nlab):
            x, y, bw, bh, a = stats[i]
            if a < area_min or a > area_max:
                continue
            if nombre == "blanca" and (bw > ANCHO_MAX_BLANCA * w
                                       or bh > ALTO_MAX_BLANCA * h / (banda[1] - banda[0])):
                continue
            m = lab == i
            ax0, ax1 = max(0, x - radio), min(w, x + bw + radio)
            ay0, ay1 = max(0, y - radio), min(h, y + bh + radio)
            anillo = np.ones((ay1 - ay0, ax1 - ax0), bool)
            anillo[y - ay0:y - ay0 + bh, x - ax0:x - ax0 + bw] = False
            ring_std = float(np.median(gstd[ay0:ay1, ax0:ax1][anillo])) if anillo.any() else 0.0
            if ring_std > RING_STD_MAX:
                continue
            frac_anillo = (float((pico[ay0:ay1, ax0:ax1][anillo] >= pmin).mean())
                           if anillo.any() else 0.0)
            if frac_anillo > FRAC_ANILLO_COLOR_MAX:
                continue
            k_ini = int(np.median(ini[m]))
            k_fin = int(np.median(fin[m]))
            c_salto = min(1.0, float(np.median(salto[m])) / (2.0 * smin))
            c_base = float(np.clip(1.0 - np.median(base[m]) / max(1, bmax), 0.0, 1.0))
            dur_s = float(np.median(largo[m])) * paso / fps
            c_dur = 1.0 if dur_min_s <= dur_s <= dur_max_s else 0.0
            cx = (x + bw / 2.0) / w
            lado = "L" if cx < 0.5 else "R"
            luces.append(dict(
                canal=nombre, cuadro_encendido=int(idx[k_ini]),
                cuadro_apagado=int(idx[min(T - 1, k_fin + 1)]),
                duracion_s=round(dur_s, 3),
                t_encendido_s=round(float(idx[k_ini]) / fps, 3),
                lado=lado, cx=round(cx, 4),
                cy=round(banda[0] + (y + bh / 2.0) / h * (banda[1] - banda[0]), 4),
                fx=round(x / w, 4), fw=round(bw / w, 4),
                fy=round(banda[0] + y / h * (banda[1] - banda[0]), 4),
                fh=round(bh / h * (banda[1] - banda[0]), 4),
                area=int(a), salto=round(float(np.median(salto[m])), 1),
                base=round(float(np.median(base[m])), 1), ring_std=round(ring_std, 1),
                frac_anillo=round(frac_anillo, 3),
                c_salto=round(c_salto, 3), c_base=round(c_base, 3), c_dur=round(c_dur, 3),
                confianza=round(c_salto * c_base * c_dur, 3),
            ))

    luces.sort(key=lambda d: (d["cuadro_encendido"], -d["area"]))
    for j, d in enumerate(luces):
        d["orden"] = j
        d["discordante"] = ((d["canal"] == "roja" and d["lado"] == "R")
                            or (d["canal"] == "verde" and d["lado"] == "L"))
    return dict(luces=luces, fps=fps, n=n, W=W, H=H, T=T, paso=paso)


def agrupa(luces: list[dict], fps: float, ventana_s: float = 0.30) -> list[dict]:
    """Une componentes del MISMO canal y MISMO lado que se encienden dentro de `ventana_s`.

    Existe porque una barra de luz partida por el texto del nombre del tirador sale como dos o tres
    componentes conexas, y son UNA luz. Se agrupa por (canal, lado) y cercania temporal, y se queda
    la de mayor area como representante sumando el area del grupo. `ventana_s` = 0,30 s no se elige
    por ensayo: es el tiempo de BLOQUEO del aparato de florete, o sea el intervalo dentro del cual el
    aparato considera simultaneos dos tocados.
    """
    fuera: list[dict] = []
    for d in sorted(luces, key=lambda x: (-x["area"], x["cuadro_encendido"])):
        for e in fuera:
            if (e["canal"] == d["canal"] and e["lado"] == d["lado"]
                    and abs(e["cuadro_encendido"] - d["cuadro_encendido"]) <= ventana_s * fps):
                e["area"] += d["area"]
                e["cuadro_encendido"] = min(e["cuadro_encendido"], d["cuadro_encendido"])
                e["cuadro_apagado"] = max(e["cuadro_apagado"], d["cuadro_apagado"])
                e["confianza"] = max(e["confianza"], d["confianza"])
                break
        else:
            fuera.append(dict(d))
    fuera.sort(key=lambda x: (x["cuadro_encendido"], -x["area"]))
    for j, d in enumerate(fuera):
        d["orden"] = j
        d["t_encendido_s"] = round(d["cuadro_encendido"] / fps, 3)
    return fuera


def grupo_del_tocado(luces: list[dict], fps: float, conf_min: float = CONF_MIN,
                     ventana_s: float = VENTANA_TOCADO_S) -> list[dict]:
    """De todas las luces del clip, las que corresponden AL tocado: el ultimo evento FUERTE y sus companeras.

    La idea de partida es la regla P5 del detector viejo del proyecto (`fencingly_v2/lights.py`): el
    tocado es el ULTIMO encendido y sus companeros en los 0,5 s anteriores, adoptada alli despues de
    medir que quedarse con el MINIMO de los ultimos encendidos de cada canal mezclaba la luz del
    tocado anterior con la del tocado. Se reutiliza y se cita.

    LO QUE HUBO QUE ANADIRLE, con la medida que lo obligo. "El ultimo encendido" a secas rompio dos
    sedes de las 21: en `qatarfoil2021` la barra verde real (area 739, salto 146) quedaba tapada por
    un pixel rojo de 13 px que aparece junto al reloj 46 cuadros despues, y en `turinfoil2023` la
    barra verde real (area 454) quedaba tapada por un artefacto de 88 px. Asi que el ancla no es el
    ultimo evento sino el ULTIMO EVENTO FUERTE, donde fuerte es `area * salto >= FRAC_FUERTE` veces
    el maximo del clip. Los companeros, en cambio, NO pasan el filtro de fuerza: la luz blanca y las
    de los marcadores de punto son legitimamente pequenas, y exigirles fuerza las borraria.

    `ventana_s` = 0,5 s y no 0,3 s (el bloqueo del aparato) porque aqui la ventana se aplica sobre
    cuadros MUESTREADOS con paso 2 y con un detector cuyo instante tiene la incertidumbre del paso.
    """
    val = [d for d in luces if d["confianza"] >= conf_min]
    if not val:
        return []
    fuerza = {id(d): d["area"] * max(1.0, d["salto"]) for d in val}
    fmax = max(fuerza.values())
    fuertes = [d for d in val if fuerza[id(d)] >= FRAC_FUERTE * fmax]
    ancla = max(fuertes, key=lambda d: d["cuadro_encendido"])
    grupo = [d for d in val
             if abs(d["cuadro_encendido"] - ancla["cuadro_encendido"]) <= ventana_s * fps]
    grupo.sort(key=lambda x: x["cuadro_encendido"])
    for j, d in enumerate(grupo):
        d["orden"] = j
    return grupo


def resumen_clip(grupo: list[dict]) -> list[dict]:
    """Colapsa el grupo del tocado a UNA fila por (canal, lado), con el encendido mas temprano.

    POR QUE HACE FALTA, con los dos casos que lo obligaron. Una misma luz puede salir como dos
    componentes que la agrupacion por ventana no une: en `stpetersburgfoil2019` el indicador rojo se
    DESPLAZA unos 0,015 W a lo largo de 70 cuadros -- medido en la tira de fotogramas -- y un
    detector de pixel fijo lo ve como dos trozos con encendidos separados 10 cuadros, por encima de
    la ventana de 0,30 s del bloqueo. Y en `tokyoolympics2020` el rotulo OBS enciende el punto rojo
    Y ADEMAS tine de rosa la placa del nombre del tirador que puntua, que son dos regiones del mismo
    color y el mismo lado.

    Colapsar por (canal, lado) es correcto porque el aparato tiene EXACTAMENTE cuatro canales: rojo
    izquierda, verde derecha, blanca izquierda y blanca derecha. Dos regiones del mismo canal y lado
    no pueden ser dos luces distintas.

    Y DESPUES SE COLAPSA POR CANAL. El aparato tiene UNA luz roja y UNA verde, asi que dos regiones
    del mismo color en lados opuestos no pueden ser dos luces: se queda la de mas area. Salio de
    `turinfoil2023`, donde junto a la barra verde real de la derecha (454 px) aparecia un "verde
    izquierda" de 19 px. Esto NO impone la convencion rojo-izquierda / verde-derecha: si el unico
    rojo del clip esta a la derecha se conserva y se marca `discordante`, que es lo que hace falta
    para las familias de marcador con el indicador centrado.
    """
    por_clave: dict[tuple[str, str], dict] = {}
    for d in grupo:
        clave = (d["canal"], d["lado"])
        prev = por_clave.get(clave)
        if prev is None or d["cuadro_encendido"] < prev["cuadro_encendido"]:
            nuevo = dict(d)
            if prev is not None:
                nuevo["area"] = prev["area"] + d["area"]
                nuevo["confianza"] = max(prev["confianza"], d["confianza"])
            por_clave[clave] = nuevo
        else:
            prev["area"] += d["area"]
            prev["confianza"] = max(prev["confianza"], d["confianza"])
    mejor_lado: dict[str, dict] = {}
    for d in por_clave.values():
        prev = mejor_lado.get(d["canal"])
        if prev is None or d["area"] > prev["area"]:
            mejor_lado[d["canal"]] = d
    fuera = sorted(mejor_lado.values(), key=lambda x: x["cuadro_encendido"])
    for j, d in enumerate(fuera):
        d["orden"] = j
    return fuera


# ---------------------------------------------------------------------------------------------
# LA ETAPA 9 ENTERA: que detector se usa y en que orden
# ---------------------------------------------------------------------------------------------

USA_V2 = os.environ.get("FENCINGLY_LUCES_V2", "1") == "1"
"""EL DETECTOR VIEJO PRIMERO (`luces_v2`, regla R2 de `LUCES_POR_FUENTE_2026-10-09.md`), y este
modulo solo si aquel no da tocado. ADOPTADO (2026-10-09): pasa las cuatro puertas declaradas en la
mitad de AJUSTE (la unica de las seis reglas que las pasa todas) y se confirma en la de PRUEBA, 2.016
clips recalculados con los dos detectores. Florete, PRUEBA, 868 clips:

    acuerdo de una luz con `touch` (A1)   0,801 (554) -> 0,947 (641)   +0,146 IC95 [+0,115, +0,176]
    ACIERTO* (sobre todos los left/right)  0,525 -> 0,718             +0,193 IC95 [+0,158, +0,230]
    cobertura                              0,795 -> 0,969
    cuadro del tocado = 0                  0,094 -> 0,014

Espada 0,902 -> 0,944 y sable 0,864 -> 0,930 en A1 (PRUEBA). Y es mas barato: el viejo tarda menos
que este y este solo corre en el 8 % de clips en que el viejo no da tocado. `=0` vuelve a la etapa 9
de antes, byte a byte."""

_V2_CANAL = {"red_L": ("roja", "L"), "green_R": ("verde", "R"),
             "white_L": ("blanca", "L"), "white_R": ("blanca", "R")}


def lee_v2(video: str) -> tuple[list[dict], str, float | None]:
    """QUE DEVUELVE: (filas con el esquema de `resumen_clip`, via, fps) con el detector viejo.

    Las columnas que el viejo no mide (area, salto, confianza...) van `None`, no cero: el viejo no da
    confianza y escribir 1,0 seria inventarla. El cuadro sale de `touch_lights` (segundos con tres
    decimales) por `round(t * fps)`, que devuelve el cuadro exacto (el error del redondeo es <0,03
    cuadros a 60 fps)."""
    from luces_v2 import lights_generic as LG
    r = LG.analyze_any(video, clock=True)
    fps = r.get("fps")
    via = "v2_%s" % (r.get("source") or "none")
    if r.get("f_touch") is None or not fps:
        return [], via, fps
    filas = []
    for k, t in (r.get("touch_lights") or {}).items():
        if t is None or k not in _V2_CANAL:
            continue
        canal, lado = _V2_CANAL[k]
        filas.append(dict(canal=canal, lado=lado, cuadro_encendido=int(round(t * fps)),
                          t_encendido_s=t, confianza=None, discordante=False,
                          cajas_v2=r.get("box_method"),
                          avisos_v2="|".join(r.get("quality_flags") or [])))
    filas.sort(key=lambda d: d["cuadro_encendido"])
    for j, d in enumerate(filas):
        d["orden"] = j
    return filas, via, fps


def lee_vigente(video: str, roi=None) -> tuple[list[dict], str, float]:
    """La etapa 9 de antes del 2026-10-09, sin cambiar nada: ROI de sede si la hay, y el detector
    general si la ROI no da nada (`general_tras_roi`)."""
    det = detecta(video, banda=roi["banda"]) if roi else detecta(video)
    via = "roi_sede" if roi else "general"
    fps = det.get("fps") or 25.0
    gr, via = _grupo(det["luces"], fps, via)
    if roi and not gr:
        det2 = detecta(video)
        gr, via = _grupo(det2["luces"], fps, "general_tras_roi")
    return resumen_clip(gr) or [], via, fps


def _grupo(luces: list[dict], fps: float, via: str) -> tuple[list[dict], str]:
    """El grupo del tocado y, con `SIN_CUADRO0`, el arreglo del cuadro 0: SOLO si el grupo empieza en
    el cuadro 0 se rehace sin las luces que ya estaban encendidas al empezar el clip, y la via lleva
    `+sin0` para que se pueda separar. Un clip cuyo grupo no empieza en 0 no se toca."""
    gr = grupo_del_tocado(agrupa(luces, fps), fps)
    if SIN_CUADRO0 and gr and min(d["cuadro_encendido"] for d in gr) == 0:
        gr = grupo_del_tocado(agrupa([d for d in luces if d["cuadro_encendido"] > 0], fps), fps)
        via += "+sin0"
    return gr, via


def lee_clip(video: str, roi=None) -> tuple[list[dict], str, float]:
    """QUE DEVUELVE: (filas del grupo del tocado, via, fps). Lo llaman la etapa 9 y la 11, para que
    las dos lean las luces de la MISMA manera. Con `USA_V2`, el viejo primero; si no da tocado, el
    vigente, con `via = tras_v2_<via del vigente>` para que se pueda separar."""
    if USA_V2:
        filas, via, fps = lee_v2(video)
        if filas:
            return filas, via, fps
        filas, via, fps2 = lee_vigente(video, roi)
        return filas, "tras_v2_" + via, fps2
    return lee_vigente(video, roi)

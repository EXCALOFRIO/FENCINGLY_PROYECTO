"""La cadena completa que saca los DOS tiradores de un clip, en un solo sitio.

Existe porque la misma secuencia de seis pasos estaba copiada en cinco scripts distintos, y cada
copia se quedaba atras al arreglar algo. Ahora la cadena esta aqui y los scripts la llaman.

LOS SEIS PASOS, y que arreglo aporta cada uno con su cifra medida en 5 clips:

  1. DETECTAR             YOLO11m-pose en el cuadro completo, `conf` bajo para no perder a nadie
  2. PUERTA DE PLANOS     quita repeticiones y primeros planos. Marca 109 cuadros en
                          `HarshExemplaryGermanspaniel`, que son su repeticion verificada a ojo, y
                          CERO en los otros cuatro clips
  3. TRAYECTORIAS         agrupa detecciones; el cambio de identidad deja de ser posible
  4. FUSION               une los trozos de la misma persona. En `RichDefensiveBushbaby` recupera
                          141 cuadros de un solo empalme (IoU 0,86 con 21 cuadros de hueco)
  5. PAREJA               elige la pareja FILMADA, no las dos mejores sueltas. En
                          `ZanyMeagerIndianrockpython`, cuatro pistas a la vez, pasa de 40,4 % a
                          78,4 % de cuadros con los dos
  6a. RECUPERACION GRATIS  antes de gastar GPU se mira si el que falta YA estaba entre las
                          detecciones del cuadro completo. Medido: entre el 25 % y el 49 % de los
                          cuadros incompletos tenian el dato ahi mismo
  6b. REDETECCION         y solo si no estaba, se busca en un recorte. Recupera el 23 % de lo que
                          faltaba, con un control negativo del 3 %. Las dos vias RECHAZAN una caja
                          que solape mas de 0,50 con la del otro tirador: sin eso la recuperacion se
                          engancha al rival y produce los avisos de "las dos cajas solapan 0,91".
                          La comprobacion es SIN UMBRAL: la caja tiene que parecerse mas a su sitio
                          estimado que a la del rival. Con un umbral absoluto de 0,50, `ZanyMeager`
                          se hundia de 99,5 % a 81,6 % al rechazar cuerpo a cuerpo legitimos
  7. PUERTA DE FUNDIDOS   quita los cuadros donde se superponen dos planos. Marca c7-c23 de
                          `VainAchingGonolek` y nada en los otros once
  8. RECORTE              el clip se queda en el tramo donde los DOS estan medidos. Es lo que
                          sustituye a sostener una pose en los bordes, que es lo que el usuario vio
                          mal en el video
  9. DUPLICADOS           quita la caja intrusa cuando las dos caen sobre la MISMA persona
 10. CONTINUIDAD         garantia de salida: ningun cuadro entregado tiene un salto de caja por
                          encima de 0,60 anchos, que es el p99,9 medido con holgura
 11. TIRAS DEL BORDE     quita las cajas de proporcion menor que 0,25 pegadas al borde del cuadro:
                          son un trozo de tirador y su pose sale vacia (20 puntos utiles de 63)

RESULTADO EN LOS 5 CLIPS, cuadros con LOS DOS tiradores presentes:

  | clip                  | antes | ahora |
  |-----------------------|------:|------:|
  | `HarshExemplary`      | 100,0 | 100,0 |
  | `defeatedHarlequin`   |  99,7 |  99,7 |
  | `VainAching`          |  93,3 |  96,1 |
  | `RichDefensive`       |  32,8 |  93,8 |
  | `ZanyMeager`          |  40,4 |  80,4 |

POR QUE SE RECORTA EN VEZ DE SOSTENER LA POSE. El usuario reviso el video y dijo que las poses
inventadas del principio de `VainAchingGonolek` estaban mal. Lo estaban, y la culpa no era de la
interpolacion: en los cuadros del borde uno de los dos NO ESTA en el plano, asi que sostener su
ultima pose conocida pone un esqueleto congelado encima de un sitio donde no hay nadie. Interpolar
entre dos anclas si esta medido y sale bien (2,8 % del torso con hueco de 1 cuadro), pero sostener
no tiene ancla al otro lado y no se puede validar de ninguna forma.

Asi que los bordes se RECORTAN: el clip empieza donde los dos estan medidos y acaba donde dejan de
estarlo. Cuesta cuadros y no invento nada. `tipo` sigue llevando la marca de como se obtuvo cada
caja (0 medido, 1 interpolado, 2 sostenido, 3 redetectado) y NO se debe perder al guardar.
"""
from __future__ import annotations

import os

import numpy as np

import a3_selector as SEL
import pareja_v2 as PAR
import planos as PL
import redetecta as RD
import transiciones as TRA

USA_CRUDAS = False
"""LA VIA GRATIS ESTA APAGADA, Y ESTA MEDIDO POR QUE. La idea era: antes de gastar GPU redetectando
en un recorte, mirar si el que falta ya estaba entre las detecciones del cuadro completo. Sonaba
gratis y sale PEOR:

  | clip              | solo recorte | solo via gratis | las dos |
  |-------------------|-------------:|----------------:|--------:|
  | `ZanyMeager`      |    **99,5 %**|          79,6 % |  81,6 % |
  | `RichDefensive`   |    **94,6 %**|          93,5 % |  93,2 % |
  | `GentleQuerulous` |       98,1 % |          98,1 % |  98,1 % |

EL MECANISMO, que es lo que hay que recordar: la via gratis coge una caja, sobrescribe con ella la
ESTIMACION `Be` de donde deberia estar el tirador, y si luego el filtro de continuidad la tira, la
redeteccion en recorte se queda buscando alrededor de una estimacion envenenada. O sea que no es
que aporte poco: es que estropea la via buena.

Y el 25-49 % de "recuperaciones disponibles gratis" que medi para justificarla estaba mal medido:
lo conte en una tirada SIN redeteccion, asi que contaba como oportunidad lo que la redeteccion ya
encontraba sola.

Se deja el interruptor y el codigo para que quede la medida, no para usarlo."""
CONF_YOLO = 0.10
IMGSZ_YOLO = 960

YOLO_HALF = os.environ.get("FENCINGLY_YOLO_HALF", "1") == "1"
YOLO_LOTE = int(os.environ.get("FENCINGLY_YOLO_LOTE", "8"))
"""YOLO EN fp16 Y EN LOTES DE 8 CUADROS, adoptado el 2026-10-08 con dos puertas.

Puerta 1, cajas crudas (8 clips, 2.487 cuadros): el 99,92 % de las cajas con IoU >= 0,9 y
desplazamiento p50 0,26 px, pero el MISMO NUMERO de cajas solo en el 98,9 % de los cuadros contra el
99 % declarado: **no paso**. La diferencia son cajas en el umbral `CONF_YOLO = 0,10` que aparecen o no.

Puerta 2, la que importa y declarada antes de medirla: la PAREJA que elige `procesa` (20 clips,
12.069 cajas): **99,94 %** con IoU >= 0,95 por lado (criterio 99,5 %) y presencia igual en el
99,99 %. Las cajas de umbral no llegan a la pareja. Velocidad de la deteccion, intercalada con la
GPU compartida: **1,53x** (fp16 solo: 1,19x; el lote es lo que paga).

`FENCINGLY_YOLO_HALF=0 FENCINGLY_YOLO_LOTE=1` devuelve el camino anterior, cuadro a cuadro en fp32."""


def detecta_clip(yolo, cap, cv2) -> tuple:
    """QUE HACE: pasa el detector por todos los cuadros y devuelve cajas, cuadros y los fotogramas.

    Se devuelven los fotogramas porque los pasos 6 y las capas densas los necesitan, y volver a
    decodificar el video cuesta mas que tenerlos.
    """
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames, cajas, cuadros = [], [], []
    k = 0
    fin = False
    while not fin:
        trozo = []
        while len(trozo) < YOLO_LOTE:
            ok, im = cap.read()
            if not ok:
                fin = True
                break
            trozo.append(im)
        if not trozo:
            break
        frames.extend(trozo)
        rs = yolo.predict(trozo, imgsz=IMGSZ_YOLO, conf=CONF_YOLO, verbose=False, device=0,
                          half=YOLO_HALF)
        for r in rs:
            if r.boxes is not None and len(r.boxes):
                b = r.boxes.xyxy.cpu().numpy()
                for i in range(len(b)):
                    cajas.append(b[i])
                    cuadros.append(k)
            k += 1
    return (np.asarray(cajas, "float32"), np.asarray(cuadros, "int32"), W, H, k, frames)


PROP_MIN = 0.25
"""Proporcion ancho/alto minima de una caja para que su pose valga. Por debajo, la caja es una TIRA
en el borde del cuadro con un trozo de tirador dentro, y la pose que sale de ahi esta vacia.

LO ENCONTRO EL USUARIO al pedir que le ensenase las poses inventadas: en `RichDefensiveBushbaby`
c130 la pose inventada salia vacia porque sus DOS anclas eran tiras de proporcion 0,11 en el borde
izquierdo, con solo un zapato dentro. Se estaba interpolando entre dos poses vacias.

Medido sobre 4.892 cajas elegidas de 9 clips:

  | proporcion | cajas | puntos utiles del cuerpo (de 63), mediana | en el borde |
  |---|---:|---:|---:|
  | todas          | 4.892 | 63 | 20 % |
  | menor que 0,30 |   269 | 58 | 57 % |
  | menor que 0,25 |    98 | **20** | **99 %** |
  | menor que 0,20 |    68 | **10** | **100 %** |

El p05 de todas es 0,297, asi que 0,25 no toca las buenas. Y se exige ADEMAS que la caja toque el
borde, porque una caja estrecha en medio del cuadro puede ser un tirador de perfil legitimo."""
PUNTOS_MIN = 20
"""Puntos utiles del cuerpo por debajo de los cuales la pose no vale, aunque la caja pase. Es la
medida DIRECTA del problema, mientras que la proporcion es un indicio previo. La mediana global es
63 de 63."""


def _quita_tiras(B, pres, W, H, T):
    """QUE HACE: marca como NO medida la caja que es una tira pegada al borde del cuadro.

    No se borra el cuadro: se deja el hueco, y la interpolacion lo rellena desde anclas que si
    tienen pose. Eso es mejor que conservar una caja de la que no sale ningun punto.
    """
    n = 0
    for k in range(T):
        for lado in (0, 1):
            if not pres[k, lado]:
                continue
            b = B[k, lado]
            if not np.isfinite(b).all():
                continue
            w, h = (b[2] - b[0]), (b[3] - b[1])
            if h <= 0:
                continue
            borde = b[0] <= 0.004 or b[2] >= 0.996
            if borde and (w * W) / max(h * H, 1e-6) < PROP_MIN:
                pres[k, lado] = False
                B[k, lado] = np.nan
                n += 1
    return n


def _recorta(pres, mask):
    """QUE HACE: deja fuera los cuadros de los extremos donde no estan los DOS medidos.

    No toca los huecos de dentro: esos si tienen ancla a los dos lados y se interpolan, que esta
    medido. Esto es solo para los extremos, donde no hay nada que interpolar.
    """
    ambos = (pres.sum(axis=1) == 2) & mask
    if not ambos.any():
        return np.zeros_like(mask), 0, 0
    a, b = int(np.argmax(ambos)), int(len(ambos) - 1 - np.argmax(ambos[::-1]))
    nueva = mask.copy()
    nueva[:a] = False
    nueva[b + 1:] = False
    return nueva, a, int(mask.sum()) - int(nueva.sum())


def _quita_duplicados(B, pres, W, H, T, umbral=0.70):
    """QUE HACE: cuando las dos cajas elegidas caen sobre la MISMA persona, quita la intrusa.

    QUE AVERIA ES ESTA, encontrada por el buscador automatico: en `DearestThirdImago` c202 las dos
    cajas solapan **0,91**, o sea estan una encima de la otra sobre el mismo tirador, porque el otro
    no esta en el plano. La puerta de separacion de `pareja_v2` no lo ve porque trabaja con la
    MEDIANA sobre los cuadros compartidos, y unos pocos cuadros malos no mueven la mediana.

    COMO SE DECIDE CUAL DE LAS DOS ES LA INTRUSA, sin adivinar: se mira la continuidad de cada una
    con su propio pasado. La caja que sigue en su sitio respecto al cuadro anterior de su lado es la
    buena; la que ha aparecido de la nada es la intrusa. Si las dos son igual de continuas no se
    toca nada y se deja el aviso, porque un cuerpo a cuerpo real tambien solapa.
    """
    quitados = 0
    for k in range(T):
        if not (pres[k, 0] and pres[k, 1]):
            continue
        a, b = B[k, 0], B[k, 1]
        xa, ya = max(a[0], b[0]), max(a[1], b[1])
        xb, yb = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, xb - xa) * max(0.0, yb - ya)
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        if u <= 0 or inter / u < umbral:
            continue
        cont = []
        for lado in (0, 1):
            ant = [j for j in range(k - 1, max(-1, k - 6), -1) if pres[j, lado]]
            if not ant:
                cont.append(-1.0)
                continue
            p = B[ant[0], lado]
            q = B[k, lado]
            x0, y0 = max(p[0], q[0]), max(p[1], q[1])
            x1, y1 = min(p[2], q[2]), min(p[3], q[3])
            it = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            uu = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - it
            cont.append(it / uu if uu > 0 else 0.0)
        if cont[0] < 0 and cont[1] < 0:
            continue
        if abs(cont[0] - cont[1]) < 0.15:
            continue
        malo = 0 if cont[0] < cont[1] else 1
        pres[k, malo] = False
        B[k, malo] = np.nan
        quitados += 1
    return quitados


DESPL_MAX = 0.60
"""Desplazamiento maximo del centro entre cuadros medidos, en anchos de caja y por cuadro de hueco.

MEDIDO sobre 6.243 transiciones de las parejas verificadas a ojo: p50 **0,016**, p95 0,102, p99
0,168 y p99,9 **0,460** anchos de caja. Un limite en 0,60 corta el **0,08 %** de las transiciones, y
al mirarlas son justo las malas: los avisos de salto daban 1,22, 1,41 y 1,67 anchos, o sea 8 a 10
veces el p99. Eso no es un tirador moviendose, es la caja cambiando de persona."""


def _continuidad(B, pres, W, H, T):
    """QUE HACE: quita los cuadros donde la caja elegida se TELETRANSPORTA, y deja el hueco.

    POR QUE ASI Y NO ARREGLANDO EL RASTREADOR. Puse el limite de asociacion relativo a la caja en
    `a3_selector`, que era un fallo real, y mejoro el recorte de `BabyishIdioticCockerspaniel` de 215
    a 124 cuadros. Pero los saltos SIGUIERON, porque pueden entrar por otras tres vias: la via de
    IoU del rastreador, la fusion por prediccion (que mide el residuo contra la posicion
    extrapolada, no contra la ultima conocida) y la redeteccion. Perseguir cada via una a una deja
    siempre la siguiente abierta.

    Esto es una GARANTIA DE SALIDA: pase lo que pase antes, ningun cuadro entregado tiene un salto
    por encima del limite. El hueco que deja lo rellenan la redeteccion y la interpolacion, que ya
    estan medidas, asi que no se pierde el cuadro: se pierde el dato malo.

    Y si el salto es la entrada a un tramo largo sobre OTRA persona, el filtro tira el tramo
    entero, porque cada cuadro se compara con el ultimo CONSERVADO, que sigue en el tirador bueno.
    """
    quitados = 0
    for lado in (0, 1):
        ultimo = -1
        for k in range(T):
            if not pres[k, lado]:
                continue
            if ultimo < 0:
                ultimo = k
                continue
            g = k - ultimo
            if g > 3:
                """Con el hueco grande no hay nada que comparar: la caja puede estar donde sea."""
                ultimo = k
                continue
            p, q = B[ultimo, lado], B[k, lado]
            ancho = max((p[2] - p[0]) * W, 1.0)
            d = float(np.hypot((0.5 * (q[0] + q[2]) - 0.5 * (p[0] + p[2])) * W,
                               (0.5 * (q[1] + q[3]) - 0.5 * (p[1] + p[3])) * H))
            if d / ancho > DESPL_MAX * g:
                pres[k, lado] = False
                B[k, lado] = np.nan
                quitados += 1
            else:
                ultimo = k
    return quitados


def _de_las_crudas(cajas, cuadros, k, est, W, H, prohibida=None):
    """QUE HACE: busca entre las detecciones YA HECHAS del cuadro completo una compatible con `est`.

    POR QUE ES LO PRIMERO QUE HAY QUE PROBAR: medido en los clips con mas cuadros incompletos, entre
    el **25 % y el 49 %** de los que faltan tienen una deteccion cruda compatible ahi mismo. O sea,
    el dato ya estaba y la tuberia lo tiraba, y pasar el detector otra vez en un recorte para
    encontrar lo que ya tenia es gastar GPU por nada.

    `prohibida` es la caja del OTRO tirador en ese cuadro. Sin ella, la recuperacion se engancha al
    rival cuando los dos estan juntos, y eso es lo que producia los avisos de "las dos cajas solapan
    0,91": no era el selector, era la recuperacion poniendo al que falta encima del que si estaba.
    """
    en = np.nonzero(cuadros == k)[0]
    e = np.asarray(est, dtype="float64") * np.array([W, H, W, H], dtype="float64")
    mejor, mejor_v = None, -1.0
    for i in en:
        b = np.asarray(cajas[i], dtype="float64")
        if abs(b[3] - e[3]) / float(H) > RD.DPIE_ACEPTA:
            continue
        alt = max(e[3] - e[1], 1.0)
        if not (RD.TALLA_ACEPTA[0] <= (b[3] - b[1]) / alt <= RD.TALLA_ACEPTA[1]):
            continue
        v = RD._iou(b, e)
        if prohibida is not None and np.isfinite(prohibida).all():
            pr = np.asarray(prohibida, dtype="float64") * np.array([W, H, W, H], dtype="float64")
            """SIN UMBRAL: se exige que la caja se parezca MAS a donde deberia estar el que falta
            que a la del rival. Un umbral absoluto no sirve y esta medido: con 0,50, `ZanyMeager`
            bajaba de 99,5 % a 81,6 % porque rechazaba recuperaciones legitimas en cuerpo a cuerpo,
            y su c212 esta verificado a ojo como correcto."""
            if RD._iou(b, pr) >= v:
                continue
        if v > mejor_v:
            mejor, mejor_v = b, v
    return (mejor, mejor_v) if (mejor is not None and mejor_v >= RD.IOU_ACEPTA) else (None, 0.0)


def procesa(cajas, cuadros, W, H, T, yolo=None, frames=None, gris=None, cron=None) -> dict:
    """QUE HACE: los pasos 2 a 6 sobre unas detecciones ya hechas.

    `yolo` y `frames` son opcionales: sin ellos se salta la redeteccion y se devuelve lo que hay.
    Eso permite correr los pasos baratos sin GPU y medir por separado lo que aporta cada uno.

    `cron` es un dict opcional donde se acumulan los segundos de cada paso. Se instrumenta AQUI y no
    en una copia del codigo a proposito: el tiempo que se informa tiene que ser el de la tuberia que
    de verdad corre, y una copia paralela se queda atras en cuanto se toca un paso.

    QUE DEVUELVE un dict con
      `B`     (T, 2, 4) cajas normalizadas, con NaN donde no hay nada ni estimado
      `pres`  (T, 2) presencia REAL o redetectada, que es dato medido
      `tipo`  (T, 2) 0 medido, 1 interpolado, 2 sostenido, 3 redetectado, -1 nada
      `mask`  (T,) cuadros utilizables
      y el informe con las cifras de cada paso
    """
    import time as _t

    def marca(clave, t0):
        if cron is not None:
            cron[clave] = cron.get(clave, 0.0) + (_t.perf_counter() - t0)
        return _t.perf_counter()

    t = _t.perf_counter()
    util = PL.tramo_utilizable(cajas, cuadros, W, H, T)
    mask = util["mask"]
    t = marca("planos", t)
    fundidos = []
    if gris is not None:
        """La puerta de fundidos necesita las trayectorias, asi que se construyen una vez sobre la
        mascara de planos y se reutilizan para marcar."""
        pre = SEL.construye_trayectorias(cajas[mask[cuadros]], cuadros[mask[cuadros]], W, H, T)
        fu = TRA.marca(gris, pre, T)
        mask = mask & fu["mask"]
        fundidos = fu["tramos"]
        t = marca("fundidos", t)
    keep = mask[cuadros]
    cj, ck = cajas[keep], cuadros[keep]
    tr = SEL.construye_trayectorias(cj, ck, W, H, T)
    t = marca("trayectorias", t)
    res = PAR.elige_pareja(tr, W, H, T)
    B, pres = SEL.a_matrices(res["elegidas"], cj, T, W, H)
    t = marca("pareja", t)
    ndup = _quita_duplicados(B, pres, W, H, T)
    nsalto = _continuidad(B, pres, W, H, T)
    ambos_0 = float((pres[mask].sum(axis=1) == 2).mean()) if mask.any() else float("nan")

    Be, tipo = RD.estima_cajas(B, pres, T)
    t = marca("limpieza", t)
    nred, ncrudas = 0, 0
    for k in (range(T) if USA_CRUDAS else range(0)):
        if not mask[k]:
            continue
        for lado in (0, 1):
            if pres[k, lado] or tipo[k, lado] < 0:
                continue
            otro = B[k, 1 - lado] if pres[k, 1 - lado] else None
            b, _v = _de_las_crudas(cajas, cuadros, k, Be[k, lado], W, H, prohibida=otro)
            if b is None:
                continue
            B[k, lado] = [b[0] / W, b[1] / H, b[2] / W, b[3] / H]
            Be[k, lado] = B[k, lado]
            pres[k, lado] = True
            tipo[k, lado] = 3
            ncrudas += 1
    """FILTRO ENTRE LAS DOS VIAS DE RECUPERACION, y hace falta por un orden que costo ver: si la via
    gratis coge una caja mala, marca el cuadro como presente, la redeteccion en recorte se lo salta
    porque ya hay algo, y luego el filtro de continuidad la tira. Resultado neto: se pierde el cuadro
    que antes SI se recuperaba. Medido en `ZanyMeager`, que bajaba de 99,5 % a 81,6 %.
    Filtrando aqui, la caja mala se va antes y la via del recorte vuelve a tener su turno."""
    nsalto += _continuidad(B, pres, W, H, T)
    t = marca("limpieza", t)
    if yolo is not None and frames is not None:
        for k in range(T):
            if not mask[k] or k >= len(frames):
                continue
            for lado in (0, 1):
                if pres[k, lado] or tipo[k, lado] < 0:
                    continue
                r = RD.redetecta_cuadro(yolo, frames[k], Be[k, lado], W, H)
                if r is None:
                    continue
                b = r[0]
                otro = B[k, 1 - lado]
                if pres[k, 1 - lado] and np.isfinite(otro).all():
                    pr = np.asarray(otro, dtype="float64") * np.array([W, H, W, H])
                    est = np.asarray(Be[k, lado], dtype="float64") * np.array([W, H, W, H])
                    bb = np.asarray(b, dtype="float64")
                    if RD._iou(bb, pr) >= RD._iou(bb, est):
                        """Se parece mas al rival que a donde deberia estar: se ha enganchado."""
                        continue
                B[k, lado] = [b[0] / W, b[1] / H, b[2] / W, b[3] / H]
                Be[k, lado] = B[k, lado]
                pres[k, lado] = True
                tipo[k, lado] = 3
                nred += 1
    t = marca("redeteccion", t)
    """SEGUNDA PASADA de continuidad, y hace falta: la primera corre antes de la redeteccion, asi
    que una caja redetectada podia colar un salto despues del filtro. Medido: con una sola pasada
    quedaba un aviso de 1,22 anchos en `BabyishIdioticCockerspaniel`."""
    nsalto += _continuidad(B, pres, W, H, T)
    ntiras = _quita_tiras(B, pres, W, H, T)
    mask, ini, recortados = _recorta(pres, mask)
    marca("limpieza", t)
    ambos = float((pres[mask].sum(axis=1) == 2).mean()) if mask.any() else float("nan")
    return {"B": B, "Be": Be, "pres": pres, "tipo": tipo, "mask": mask,
            "fundidos": fundidos, "recortados": recortados, "ini": ini, "duplicados": ndup,
            "saltos_quitados": nsalto, "de_las_crudas": ncrudas, "tiras": ntiras,
            "tramos": util["tramos"], "fuera": util["fuera"],
            "ntray": len(tr), "nfus": res["nfus"], "motivo": res["motivo"],
            "borde_descartadas": res.get("borde_descartadas", 0),
            "elegidas": res["elegidas"], "parejas": res["parejas"],
            "ambos_sin_redeteccion": ambos_0, "ambos": ambos, "nredetectados": nred,
            "faltan": int(((~pres) & mask[:, None]).sum())}

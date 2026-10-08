"""Deja en las capas densas de Sapiens 2 SOLO A LOS DOS TIRADORES: ni arbitro, ni publico, ni fondo.

QUE FALLABA, y no era lo que parecia
====================================
Recortar la caja del tirador y correr la capa densa dentro NO borra al arbitro. Medido en
`defeatedHarlequinRabbitSkinkLizard`, que tiene al arbitro en primer plano cruzando por delante del
tirador derecho:

  - `matting` sobre el recorte de un tirador se queda con TODAS las personas que hay en el recorte, no
    con el tirador. La fuga (fraccion de la banda de margen ocupada por un ajeno que la mascara
    enciende) fue de **0,251** de media y de **0,268** en los 31 cuadros con el arbitro dentro.
  - y peor: a veces elige a la persona equivocada. En c313 lado1 `matting` recorto AL ARBITRO y dejo
    al tirador fuera. Por eso su recall de keypoints propios se queda en **0,905** global y baja a
    **0,877** en la ventana contaminada, con casos de menos de 0,5.
  - `seg != Background` cubre bien al tirador (recall **0,993**) pero es el que MAS arbitro cuela:
    fuga **0,383**. Segmenta a todo el que ve, que es lo suyo.

La trampa 4 del proyecto ("matting solo admite una persona por recorte") no era del todo el problema:
`matting` no se confunde, es que **no sabe a quien tiene que recortar**. Un recorte no es una orden.

QUE SE MIDIO Y CON QUE CRITERIO
===============================
Criterio declarado antes de ejecutar. Tres numeros, y ninguno sale de la capa que se juzga:

  rec_kp      keypoints propios del tirador (Sapiens pose, score>=0,3, dentro del recorte) que caen
              dentro de la mascara. Bueno: >= 0,90.
  fuga_arb    la caja de YOLO es AJUSTADA a la persona, asi que el cuerpo del tirador esta dentro de
              su caja por construccion. Entonces todo pixel de la mascara en la BANDA DE MARGEN
              (recorte menos caja propia) esta mal. `fuga_arb` es la fraccion de
              (banda ∩ caja de una deteccion no elegida) que la mascara enciende. Bueno: <= 0,05.
  fuga_fondo  lo mismo en el resto de la banda: suelo y fondo.

  La PRIMERA version de la metrica de fuga contaba keypoints del arbitro dentro de la mascara y daba
  0,000 a mascaras que a ojo se quedaban con media cabeza del arbitro, porque sus keypoints caian en
  la mitad que si se quitaba. Se cambio por la de area al VERLO en las imagenes. Una metrica que
  aprueba una mascara mala no mide nada.

LA CORRECCION: reparto por ESQUELETO MAS CERCANO, mas guarda de caja
====================================================================
Se toma la pose de los dos tiradores y de TODA deteccion no elegida del cuadro con alto de caja
>= 0,08 del alto de imagen, y un pixel se queda solo si su distancia al esqueleto propio es menor que
a cualquier esqueleto ajeno (`cv2.distanceTransform`, sin umbral que ajustar; si no hay nadie mas en
el recorte se queda todo). Encima, guarda de caja: se resta lo que caiga en la caja de un ajeno y
fuera de la propia. La guarda es gratis porque las cajas de YOLO ya estan.

TABLA MEDIDA, 41 cuadros x 2 lados del clip con arbitro delante, capas a 1024x768
                             rec_kp   rec_desp   fuga_arb   fuga_fondo   residuo
  caja sola (linea base)      0,997      0,698     0,0000       0,0000     0,657
  seg                         0,993      0,104     0,3828       0,0026     0,289
  matting                     0,905      0,090     0,2505       0,0009     0,249
  seg + esqueleto             0,993      0,101     0,1341       0,0026     0,267
  matting + esqueleto         0,905      0,088     0,0981       0,0009     0,232
  seg + esqueleto + guarda    0,992      0,101     0,0000       0,0026     0,261
  union + esqueleto + guarda  0,994      0,102     0,0000       0,0026     0,263

El reparto por esqueleto baja la fuga de `seg` de 0,383 a 0,134 (2,9x) SIN coste de recall, y la
guarda la lleva a cero costando 0,001 de recall. Lo que hay que leer de la fila de la guarda no es el
cero (ese es por construccion: la guarda resta exactamente esa region) sino que **no cuesta recall**.

Y bate a la linea base tonta: la caja sola tiene el mismo recall y la misma fuga cero, pero deja
**0,657** del recorte encendido contra **0,263**. La mascara es 2,5x mas apretada por el mismo recall.

CONTROL NEGATIVO: la misma mascara desplazada media anchura de recorte. El recall se hunde de 0,994 a
**0,102**. La caja sola solo baja a 0,698, que es lo que se espera de una mascara que ocupa dos
tercios del recorte: por eso la caja no puede ser el veredicto.

RESOLUCION: cada capa aguanta lo que aguanta, y no es lo que estaba medido para la puerta
=========================================================================================
Estaba medido que `matting` a 384x288 conserva 23,5x de separacion para hacer de PUERTA por
trayectoria. Eso no se traslada a la mascara por pixel:

  matting   758 -> 117 ms/recorte, pero el recall cae 0,905 -> 0,816 y la IoU de mascara entre las
            dos resoluciones es 0,895 de media con un MINIMO de 0,065: hay recortes en que la
            mascara barata no se parece en nada. NATIVO.
  seg       251 ->  60 ms/recorte, recall igual (0,993 -> 0,997) pero `fuga_fondo` empeora 6x,
            0,0026 -> 0,0153. NATIVO si importa el borde; barato si solo importa cubrir.
  normal    317 ->  70 ms/recorte, error angular mediano de **9,42 grados** (p90 12,03) dentro de la
            mascara. NATIVO: 9,4 grados en una capa cuya unica salida es una orientacion es mucho.
  pointmap  481 ms a 1024x768 y **NO EJECUTA a 384x288**: `mat1 and mat2 shapes cannot be multiplied
            (1x1152 and 6144x512)`. La cabeza espera la rejilla de la resolucion nativa.

POINTMAP: por que va de CUADRO COMPLETO y no de recorte
=======================================================
El control es el signo de la z, y hay que fijarlo antes de leer nada. DENTRO de un recorte sale
moneda al aire: la z del suelo de la fila de abajo supera a la de arriba en 44 de 82 recortes (53 %),
asi que la z de un recorte no ordena profundidad y cualquier lectura sobre ella no es interpretable.
Sobre el CUADRO COMPLETO si: el suelo del borde inferior supera a la grada del borde superior por
+0,158 de mediana, **7 de 7 cuadros**, y con ese signo el arbitro sale mas cerca que el tirador en
**7 de 7**, que es lo que se ve a ojo (le tapa). Asi que `pointmap` se corre una vez por cuadro
completo, que ademas cuesta una inferencia en vez de dos.

Pero esos dos controles estan CONFUNDIDOS con la y de imagen, porque en los dos casos lo cercano esta
mas abajo en el cuadro. Asi que hizo falta un tercero que no lo estuviera, y salio de MIRAR el
mosaico: en la celda de `pointmap limpio` de c0180 el tirador izquierdo sale azul y el derecho
naranja, o sea a profundidades muy distintas, cuando los dos estan tirando el uno contra el otro
sobre la misma pista y a distancia de fondo.

CONTROL FISICO, declarado antes de medirlo: en un asalto los dos tiradores estan a la misma
profundidad dentro de un par de metros y el arbitro esta varios metros mas cerca (se le ve tapar a
los tiradores). Entonces |z(T0) - z(T1)| tiene que ser MUCHO MENOR que |z(tiradores) - z(arbitro)|.
Medido sobre la mascara de cada tirador y la caja del arbitro:

                                     defeatedHarlequin   Zany      Rich
  cuadros con arbitro grande medibles         17          10         1
  |z(T0) - z(T1)|  mediana                0,2070      0,0374    0,0129
  |z(pareja) - z(arbitro)|  mediana       0,1088      0,0283    0,0913
  razon (deberia ser >> 1)                  0,61        1,08      4,67
  cuadros en que la razon supera 1        7 de 17     6 de 10   1 de 1
  |z(T0)-z(T1)| / dispersion dentro de uno  1,37        2,04      0,83
  la misma razon con la Y DEL PIE, gratis   2,87        5,25      4,91
  cuadros en que la supera               17 de 17    10 de 10   1 de 1

**`pointmap` FALLA el control en los dos clips donde se puede medir de verdad**: separa a los dos
tiradores entre si MAS que a la pareja del arbitro, al contrario de lo que exige la geometria, y la
razon 1,08 del segundo es moneda al aire. Ademas |z(T0) - z(T1)| supera la dispersion de z DENTRO de
un mismo tirador por 1,37x y 2,04x, asi que no es ruido pequeno: la capa afirma que los dos estan a
profundidades distintas. El unico clip donde acierta, `Rich`, tiene **n=1** porque casi no tiene
arbitro grande en el plano, asi que no compensa a los otros dos. Y la alternativa que cuesta CERO, la
y del pie de la caja de YOLO, acierta en **28 de 28** cuadros de los tres clips.

VEREDICTO: `pointmap` no se usa como profundidad. Se deja en el mosaico porque es lo que permite ver
el fallo, y se corre de cuadro completo por lo dicho arriba, pero no debe alimentar al modelo
temporal como z. Para el orden de profundidad se usa la y del pie.

LO QUE SIGUE FALLANDO, Y DONDE. NO ESTA RESUELTO
================================================
La metrica de banda es CIEGA dentro de la caja propia, y ahi queda arbitro. Se vio A OJO en el
mosaico de c312: el hombro del arbitro sobrevive porque su caja pisa la esquina inferior de la caja
del tirador. Dos cosas medidas sobre eso:

  - EL FILTRO DE KEYPOINTS ESPURIOS (un keypoint fuera de su propia caja no siembra) arregla el caso
    de c312 pero no el general: `fuga_solape` baja de 0,4130 a 0,4097, con caida en 3 de 71 pares y
    un maximo de 0,2064, por un coste de rec_kp de +0,0009. Se queda porque es gratis, no porque
    resuelva.
  - LA GUARDA DURA (la caja ajena gana incluso dentro de la propia) se midio y SE DESCARTA: cuesta
    0,1586 de rec_kp de media en los 73 pares con solape y hasta 0,9811 en el peor, es decir borra al
    tirador entero. El limite declarado era 0,03.

Y hay un segundo fallo, que NO es fuga sino PERDIDA: en c144-c145 lado0, con el tirador de blanco
corriendo sobre suelo claro y con arrastre de movimiento, `seg` se queda solo con el pantalon y
`matting` recorta al arbitro, asi que rec_kp cae a **0,099-0,108**. No es una disolucion de plano: se
comprobo la diferencia entre cuadros consecutivos (9,8 de media, sin la caida a 0,2 de un corte) y la
tira de c138 a c150 a ojo. Es un fallo de los dos modelos en ese cuadro. Ocurre en **2 de 134** lados
al muestrear cada 5 cuadros.
    Marcarlo por AREA no funciona y esta medido: con paso 4 parecia bien (rec_kp 0,991 en los validos
    contra 0,368 en los marcados) pero con paso 5 dejo pasar como valido el peor caso de todos
    (rec_kp 0,108 con residuo 0,104). El area no distingue "no ha encontrado a nadie" de "ha
    encontrado a la persona equivocada". La bandera que se usa es `rec_kp < 0,70`, el acuerdo entre la
    mascara y la pose, que son dos modelos independientes y la pose ya esta calculada.

Cuanto arbitro queda, con la unica verdad independiente disponible: en esgrima el arbitro va de
oscuro y los tiradores de blanco, asi que se mide el ritmo con que la mascara conserva pixeles con
V < 80 del HSV. Dentro del solape de cajas conserva **0,410**, y el nivel de referencia (el ritmo con
que conserva el oscuro que SI es del tirador, careta y zapatillas, donde no hay ningun ajeno) es
**0,708**. Que 0,410 < 0,708 dice que la mascara SI discrimina en el solape, 1,7x, pero no del todo.
El solape de cajas ocurre en **219 de 660** pares (cuadro, lado) de este clip, un 33 %, asi que no es
un caso raro. Queda pendiente.

LOS TRES CLIPS, con la mascara que se recomienda y las capas a 1024x768
======================================================================
Muestreo cada 5 cuadros, mascara = (matting>=0,5 | seg!=0) & esqueleto & guarda blanda.

  clip                          cuadros  rec_kp  rec_desp  fuga_arb  fuga_fondo  resto_arb  residuo
  defeatedHarlequinRabbit...        67    0,982     0,068    0,0000      0,0012     0,0433    0,241
  RichDefensiveBushbaby             69    0,992     0,116    0,0000      0,0108     0,0065    0,276
  ZanyMeagerIndianrockpython        40    0,996     0,076    0,0000      0,0309     0,0410    0,268

`resto_arb` es la fraccion de la mascara final que es oscuro dentro de la caja de un ajeno, o sea el
arbitro que queda: entre **0,6 % y 5,3 %** de la mascara. Es poco pero no es cero, y en los dos clips
donde el arbitro esta en primer plano (0,0433 y 0,0525) es el doble que en el que no (0,0065).

`fuga_fondo` empeora 9x y 26x en los otros dos clips (0,0012 -> 0,0108 -> 0,0309). Sube con lo pequeno
que sale el tirador: la cobertura del cuadro es 4,6 %, 6,5 % y 1,9 %, y el peor `fuga_fondo` es del
clip en que los tiradores son mas pequenos. Con n=3 esto es una correlacion observada, no una ley.

Y `fuga_solape` NO se puede leer en el tercer clip: da 0,9418 contra un nivel de referencia de 0,8738,
o sea la mascara conserva el oscuro del solape al mismo ritmo que el oscuro que si es del tirador. Con
tiradores de 1,9 % del cuadro casi todo su oscuro es careta y zapatilla, asi que la suposicion
"oscuro = arbitro" se cae a esa escala. La cifra util a escala pequena es `resto_arb`, 0,0410.

Los cuadros de prueba salen de `extrae_pareja.procesa` con la puerta de fundidos puesta. En estos tres
clips no marco NINGUN fundido, asi que la puerta no explica los cuadros en que la mascara se hunde: se
comprobo uno a uno que c144, c145, c235, c236 y c295 de defeatedHarlequin y c365 de Rich siguen dentro
de la mascara de cuadros utilizables.

QUE SE HACE CON EL PIXEL DE FUERA
=================================
Las dos mascaras cubren el **4,82 %** del cuadro, asi que guardar cuadro completo desperdicia 21x. El
contenedor son tensores POR RECORTE con un canal booleano de mascara, y el relleno depende de si el
cero es un valor legal de la capa:
  matting -> 0    alfa 0 ES el fondo. El cero no miente.
  seg     -> 0    la clase 0 ES Background. El cero no miente.
  normal  -> NaN  el vector cero no es una normal unitaria, asi que 0 valdria de centinela, pero un 0
                  se promedia sin avisar en cualquier convolucion. NaN obliga a mirar el canal de
                  mascara.
  pointmap-> NaN  obligatorio: z = 0 es una profundidad legal y un 0 seria indistinguible de un dato.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

CACHE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565\scratchpad\hoja_seg\hf")
os.environ["HF_HOME"] = str(CACHE)
"""Antes de importar transformers: es donde estan los pesos ya descargados."""

import cv2
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import extrae_pareja as EP

HUB = CACHE / "hub"
MARGEN = 0.12
UMB_ALFA = 0.5
KP_MIN = 0.3
ALTO_AJENO = 0.08
"""Una deteccion no elegida por debajo de 0,08 del alto de imagen es publico lejano: su pose es basura
y no debe competir por pixeles. No se le aplica guarda de caja tampoco, porque su caja es minuscula."""
TAM_NATIVO = (1024, 768)

CLASES_SEG = {0: "Background", 1: "Apparel", 3: "Face_Neck", 4: "Hair", 6: "L_Hand",
              7: "L_Lower_Arm", 13: "LowClothing", 15: "R_Hand", 16: "R_Lower_Arm",
              22: "Torso", 23: "UpClothing"}


def snap(nombre):
    d = HUB / ("models--facebook--%s" % nombre) / "snapshots"
    return next(d.iterdir()) if d.is_dir() else None


def a_cuda(e):
    return {k: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda")) for k, v in e.items()}


def xywh(b):
    """Trampa 1 del proyecto: la caja para el procesador de pose va en xywh. Con xyxy la confianza cae
    de 0,883 a 0,104 y el modelo parece roto."""
    return [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]


def elige_pareja_del_clip(cajas, cuadros, W, H, T, gris=None, yolo=None, frames=None):
    """La cadena de seleccion NO se reimplementa aqui: se llama a `extrae_pareja.procesa`, que es la
    entrada unica del proyecto (puerta de planos, trayectorias, fusion, pareja, redeteccion, puerta de
    fundidos, duplicados, continuidad y recorte de bordes).

    OJO CON `mask`: hay clips donde se descartan cuadros que antes se usaban, asi que la mascara de
    esta funcion es la que manda para decidir en que cuadros se construye mascara de tirador.

    `suyos` sale de los indices de las dos trayectorias elegidas y sirve para saber que deteccion del
    cuadro es AJENA, que es la pieza que necesita el reparto por esqueleto.
    """
    r = EP.procesa(cajas, cuadros, W, H, T, yolo=yolo, frames=frames, gris=gris)
    keep = r["mask"][cuadros]
    cj, ck = cajas[keep], cuadros[keep]
    suyos = set(int(i) for e in r["elegidas"][:2] for i in e["t"]["idx"])
    return {"B": r["B"], "pres": r["pres"], "mask": r["mask"], "tipo": r["tipo"],
            "cajas": cj, "cuadros": ck, "suyos": suyos, "informe": r}


def recorte(B, k, lado, W, H, margen=MARGEN):
    b = B[k, lado]
    if not (np.isfinite(b).all() and b[2] > b[0]):
        return None
    x0, y0, x1, y1 = b[0] * W, b[1] * H, b[2] * W, b[3] * H
    mw, mh = margen * (x1 - x0), margen * (y1 - y0)
    return (int(max(0, x0 - mw)), int(max(0, y0 - mh)), int(min(W, x1 + mw)), int(min(H, y1 + mh)))


def ajenos_del_cuadro(sel, k, H):
    """Detecciones NO elegidas del cuadro, de mayor a menor. La primera grande suele ser el arbitro."""
    cj, ck, suyos = sel["cajas"], sel["cuadros"], sel["suyos"]
    r = [[float(v) for v in cj[j]] for j in np.nonzero(ck == k)[0]
         if int(j) not in suyos and (cj[j][3] - cj[j][1]) / H >= ALTO_AJENO]
    r.sort(key=lambda o: -(o[3] - o[1]))
    return r


HOLGURA_KP = 0.05
"""Un keypoint de una persona tiene que caer en SU caja de YOLO, que es ajustada. Se deja un 5 % de
holgura por el borde y nada mas."""


def limpia_keypoints(kp, sc, caja):
    """QUE FALLABA: el estimador de pose del tirador ponia keypoints ENCIMA DEL ARBITRO cuando este le
    tapaba. Visto en c312 lado1: puntos con score >= 0,3 sobre el hombro del arbitro. Como el reparto
    por esqueleto usa esos puntos como semilla, el arbitro se quedaba dentro de la mascara y encima la
    metrica de banda no lo veia porque estaba dentro de la caja propia.

    LA CORRECCION es de construccion, no de umbral: un keypoint fuera de su propia caja de YOLO es
    espurio y no puede sembrar nada.
    """
    w, h = caja[2] - caja[0], caja[3] - caja[1]
    hx, hy = HOLGURA_KP * w, HOLGURA_KP * h
    ok = ((kp[:, 0] >= caja[0] - hx) & (kp[:, 0] <= caja[2] + hx)
          & (kp[:, 1] >= caja[1] - hy) & (kp[:, 1] <= caja[3] + hy))
    return kp, np.where(ok, sc, 0.0)


HUESOS = [(5, 6), (9, 10), (5, 9), (6, 10), (6, 8), (8, 21), (5, 7), (7, 42),
          (9, 11), (11, 13), (10, 12), (12, 14)]
"""Torso, brazos y piernas con el mapa de indices MEDIDO. Copiado de `video_revision.HUESOS`, que es
donde se midio; se repite aqui y no se importa porque `video_revision` es la herramienta de validacion
y este modulo no debe depender de ella."""

MODO_REPARTO = "huesos"
"""Como se siembra el reparto por esqueleto mas cercano.

MEDIDO en `capas_limpias_agujeros.py` sobre los MISMOS pixeles (una sola pasada de `seg` y `matting`
y las cuatro variantes recompuestas encima), 46 lados de 4 clips, separando cuerpo a cuerpo (IoU de
las dos cajas >= 0,10) de separados (IoU <= 0,01):

                       separados                                cuerpo a cuerpo
  variante      conserva troceadas asimetria rec_kp      conserva troceadas asimetria rec_kp
  puntos           0,954     0,031     0,064  0,996         0,798     0,143     0,177  0,992
  puntos63         0,951     0,000     0,066  0,977         0,803     0,143     0,161  0,981
  huesos_s_cabeza  0,929     0,000     0,068  0,772         0,810     0,071     0,121  0,845
  huesos           0,961     0,000     0,053  0,975         0,825     0,071     0,128  0,937

`huesos` gana en TODO lo que mide el defecto: el troceado en cuerpo a cuerpo baja de 0,143 a 0,071 y
el maximo de trozos de 5 a 3, la asimetria entre los dos lados baja 1,38x, el recall de PIXELES sube
en los dos grupos y la fuga de arbitro se queda en 0,0000 en las cuatro. Lo unico que baja es
`rec_kp` (0,992 a 0,937 en cuerpo a cuerpo), y el reparto lo explica: `rec_63`, el recall de los 63
puntos que el proyecto SI usa, solo baja de 0,981 a 0,936, o sea que la mayor parte de esa caida es
cara, que son 245 de los 308 puntos y no se usa.

Y LO QUE NO ARREGLA, dicho claro: con `huesos` el cuerpo a cuerpo SIGUE fallando el limite de
asimetria que se declaro (0,128 contra 0,10) y sigue habiendo lados con tres trozos. La avería queda
reducida a la mitad, no cerrada."""


def reparto_esqueleto(kp_propio, kps_ajenos, W, H, modo=None):
    """QUE DEVUELVE: booleano (H, W) con True donde el esqueleto propio esta mas cerca que cualquier
    ajeno. Sin umbral: es una particion, no una decision con parametro.

    LA PRIMERA VERSION SEMBRABA LOS 308 PUNTOS SUELTOS Y DESTROZABA LA MASCARA EN CUERPO A CUERPO.
    Lo encontro el coordinador MIRANDO la figura de `WhisperedTenderDeinonychus` c26: la tiradora de
    la izquierda sale en TIRAS diagonales, no en una pieza.

    El mecanismo: con semillas SUELTAS, el `distanceTransform` da una celda por punto. En un cuerpo a
    cuerpo la pose del rival coloca puntos ENCIMA del cuerpo propio, y como su caja tambien lo cubre,
    `limpia_keypoints` no los quita. Los dos juegos de puntos quedan INTERCALADOS y la propiedad
    alterna de celda en celda: de ahi las tiras. No es un umbral mal puesto; es que un conjunto de
    puntos sueltos no es un esqueleto.

    LA CORRECCION es sembrar el esqueleto como CURVA: los huesos como segmentos y las manos como sus
    raices. Un conjunto CONEXO por lado da una frontera unica y lisa, asi que en el peor caso la
    mascara pierde un trozo entero pero nunca sale a tiras. Y es lo que el nombre de la funcion
    decia que hacia desde el principio.

    `modo` admite cuatro valores para poder medirlos uno contra otro sin tocar nada mas:
      `huesos`             los 12 segmentos de tronco, brazos y piernas, las raices de las manos, los
                           5 puntos de cabeza y el segmento cuello-nariz. DEFECTO.
      `huesos_sin_cabeza`  igual pero sin cabeza. Se deja porque es lo que delato que la cabeza hacia
                           falta: sin ella `rec_kp` cae a 0,772 sin que `conserva_px` se mueva.
      `puntos63`           los 63 puntos utiles sueltos (sin cara). Version intermedia.
      `puntos`             los 308 puntos sueltos. La version RETIRADA, se deja para poder reproducir
                           su medida.
    """
    md = MODO_REPARTO if modo is None else modo

    def semilla(lista):
        s = np.full((H, W), 255, dtype="uint8")
        n = 0
        for kp, sc in lista:
            ok = (sc >= KP_MIN) & np.isfinite(kp[:, 0]) & np.isfinite(kp[:, 1])
            if md != "puntos":
                ok = ok.copy()
                ok[63:] = False
                """La cara son 245 de los 308 puntos y el proyecto no la usa (los tiradores llevan
                careta). Como semilla solo aporta 245 puntos que pueden estar mal puestos."""
            if not ok.any():
                continue
            xs = np.clip(kp[:, 0], 0, W - 1)
            ys = np.clip(kp[:, 1], 0, H - 1)
            if md.startswith("huesos"):
                puesto = False
                for a, b in HUESOS:
                    if a < len(ok) and b < len(ok) and ok[a] and ok[b]:
                        cv2.line(s, (int(xs[a]), int(ys[a])), (int(xs[b]), int(ys[b])), 0, 1)
                        puesto = True
                        n += 1
                for i in (21, 42):
                    if i < len(ok) and ok[i]:
                        cv2.circle(s, (int(xs[i]), int(ys[i])), 1, 0, -1)
                        puesto = True
                        n += 1
                if md == "huesos" and ok[:5].any():
                    """EL CUELLO Y LA CABEZA, y hacen falta. Medido: con la semilla solo de huesos de
                    tronco y extremidades, `rec_kp` cae de 0,996 a 0,772 en cuadros SIN solape
                    mientras `conserva_px` apenas se mueve (0,954 a 0,929). La explicacion es que de
                    los 308 puntos, 245 son cara: si la semilla propia no llega a la cabeza, los
                    pixeles de la careta quedan mas cerca del esqueleto de otro y la cabeza se va con
                    el. Los huesos del cuerpo no incluyen cuello porque COCO-17 no lo tiene, asi que
                    se anade el segmento del centro de los hombros a la nariz."""
                    for i in range(5):
                        if ok[i]:
                            cv2.circle(s, (int(xs[i]), int(ys[i])), 1, 0, -1)
                            n += 1
                    if ok[0] and ok[5] and ok[6]:
                        cx = int(0.5 * (xs[5] + xs[6]))
                        cy = int(0.5 * (ys[5] + ys[6]))
                        cv2.line(s, (cx, cy), (int(xs[0]), int(ys[0])), 0, 1)
                        n += 1
                    puesto = True
                if not puesto:
                    """Sin ningun hueso valido se cae a los puntos, que es mejor que no sembrar nada
                    y devolver la mascara entera o vacia."""
                    s[ys[ok].astype(int), xs[ok].astype(int)] = 0
                    n += int(ok.sum())
            else:
                s[ys[ok].astype(int), xs[ok].astype(int)] = 0
                n += int(ok.sum())
        return s, n
    sp, np_ = semilla([kp_propio])
    sa, na = semilla(kps_ajenos)
    if np_ == 0:
        return np.zeros((H, W), bool)
    if na == 0:
        return np.ones((H, W), bool)
    dp = cv2.distanceTransform(sp, cv2.DIST_L2, 5)
    da = cv2.distanceTransform(sa, cv2.DIST_L2, 5)
    return dp <= da


HOLGURA_GUARDA = 0.0
"""Holgura, en fraccion de la caja propia, con la que la guarda de caja PERDONA alrededor de la caja.

QUE FALLABA, encontrado en `SeveralSkinnyGenet` c183 lado 1 mirando la imagen: la caja de un ajeno
cubria la CABEZA Y EL HOMBRO del tirador, y esa parte de el quedaba fuera de SU propia caja de YOLO,
asi que la guarda se la comio. Recall bruto 0,996 y recall final 0,369: la guarda sola costo 0,628.

Con holgura 0 la guarda supone que la caja de YOLO cubre al tirador entero. Eso es lo que dice la
caja, pero no lo que hace: en un fondo a maxima extension con arrastre de movimiento la recorta.
`limpia_keypoints` ya asume un 5 % de error de caja para el mismo motivo.

EL VALOR SE DEJA EN 0 hasta que la medida diga otra cosa, porque subirlo tiene un coste medible: la
holgura devuelve pixeles de la banda de margen, que es donde vive el arbitro, asi que `fuga_arb` deja
de ser cero. `capas_limpias_peores.py` barre 0,00 / 0,05 / 0,10 y da las dos cifras a la vez."""


def guarda_de_caja(c, caja_propia, ajenos, H_, W_, holgura=None):
    """QUE DEVUELVE: booleano del recorte con False donde hay caja de un ajeno y NO caja propia."""
    hg = HOLGURA_GUARDA if holgura is None else holgura
    g = np.ones((H_, W_), bool)
    if hg:
        w, h = caja_propia[2] - caja_propia[0], caja_propia[3] - caja_propia[1]
        caja_propia = [caja_propia[0] - hg * w, caja_propia[1] - hg * h,
                       caja_propia[2] + hg * w, caja_propia[3] + hg * h]
    bp = [caja_propia[0] - c[0], caja_propia[1] - c[1], caja_propia[2] - c[0], caja_propia[3] - c[1]]
    for o in ajenos:
        y0, y1 = int(max(0, o[1] - c[1])), int(min(H_, o[3] - c[1]))
        x0, x1 = int(max(0, o[0] - c[0])), int(min(W_, o[2] - c[0]))
        if y1 > y0 and x1 > x0:
            g[y0:y1, x0:x1] = False
    g[max(0, int(bp[1])):min(H_, int(bp[3])), max(0, int(bp[0])):min(W_, int(bp[2]))] = True
    return g


VRAM_POR_CABEZA = {"pose": 0.9, "matting": 2.6, "seg": 0.9, "normal": 0.9, "pointmap": 0.9}
"""GB de VRAM medidos por cabeza en fp16: las cinco juntas reservan 6,26 GB. `matting` es 1b, las
otras 0.4b, y por eso pesa el triple."""
ESPERA_VRAM_S = 1800
"""Segundos que se espera a que haya VRAM antes de intentar cargar.

POR QUE HACE FALTA: esta maquina tiene UNA 3060 de 12 GB y varios trabajos del proyecto corren a la
vez. Medido hoy: con la GPU a 11,4 de 12,3 GB ocupados por otros procesos, cargar las cinco cabezas
falla y el mosaico de `AromaticShockedGermanspitz` murio en el `from_pretrained`. Y cuando no falla,
va 40-55x mas lento: `matting` paso de 758 ms a 41.086 ms por llamada. Reventar por una carrera de
memoria con otro proceso no es un resultado, es una tirada perdida, asi que se espera y se reintenta.
"""


class Capas:
    """Carga y ejecuta las capas densas de Sapiens 2. Cada modelo se carga una vez y se reutiliza."""

    NOMBRES = {"pose": ("sapiens2-pose-0.4b", "Sapiens2ForPoseEstimation"),
               "matting": ("sapiens2-matting-1b", "Sapiens2ForImageMatting"),
               "seg": ("sapiens2-seg-0.4b", "Sapiens2ForSemanticSegmentation"),
               "normal": ("sapiens2-normal-0.4b", "Sapiens2ForNormalEstimation"),
               "pointmap": ("sapiens2-pointmap-0.4b", "Sapiens2ForPointmapEstimation")}

    @staticmethod
    def espera_vram(gb, espera=ESPERA_VRAM_S, aviso=True):
        """Espera hasta que haya `gb` GB libres en la GPU. Devuelve los GB libres al salir.

        No reserva nada: solo mira. Si al agotar la espera sigue sin haber sitio, devuelve lo que hay
        y quien llama lo intenta igual, porque quedarse esperando para siempre tampoco es un
        resultado. Lo que NO se hace es callarse: se imprime cuanto se espero y con cuanto se
        arranco, para que un tiempo por llamada anomalo se pueda explicar despues.
        """
        t0 = time.perf_counter()
        libre = torch.cuda.mem_get_info()[0] / 2 ** 30
        while libre < gb and time.perf_counter() - t0 < espera:
            if aviso and int(time.perf_counter() - t0) % 60 < 2:
                print("  esperando VRAM: hacen falta %.1f GB y hay %.1f GB libres (%.0f s)"
                      % (gb, libre, time.perf_counter() - t0))
            time.sleep(15.0)
            libre = torch.cuda.mem_get_info()[0] / 2 ** 30
        if aviso:
            print("  VRAM al arrancar: %.1f GB libres de %.1f, tras esperar %.0f s"
                  % (libre, torch.cuda.mem_get_info()[1] / 2 ** 30, time.perf_counter() - t0))
        return libre

    def __init__(self, quiere, tam=TAM_NATIVO, tam_pose=(384, 288), espera_vram=True):
        import transformers
        from transformers import Sapiens2ImageProcessor
        if espera_vram:
            self.espera_vram(1.5 + sum(VRAM_POR_CABEZA.get(q, 0.9) for q in quiere))
            """El 1,5 de mas no es un margen inventado: es el pico de activaciones medido de una
            pasada de 1024x768, que en el proyecto dio 1,52 GB."""
        self.p, self.m = {}, {}
        self.ms, self.n = dict((q, 0.0) for q in quiere), dict((q, 0) for q in quiere)
        """CONTADORES DE TIEMPO POR CABEZA. Van aqui y no en quien llama porque `mascara_tirador`
        llama a `seg` y a `matting` por dentro, asi que desde fuera no se pueden separar. No hace
        falta `torch.cuda.synchronize`: el post-proceso de cada cabeza acaba en un `.cpu()`, que ya
        espera a la GPU, asi que el reloj que envuelve la llamada mide la pasada completa."""
        for q in quiere:
            nom, cls = self.NOMBRES[q]
            s = snap(nom)
            pr = Sapiens2ImageProcessor.from_pretrained(str(s))
            t = tam_pose if q == "pose" else tam
            pr.size = {"height": t[0], "width": t[1]}
            self.p[q] = pr
            for intento in range(6):
                try:
                    self.m[q] = getattr(transformers, cls).from_pretrained(
                        str(s), dtype=torch.float16).to("cuda").eval()
                    break
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    """REINTENTO, no `raise`: el que falla es el `.to("cuda")` cuando otro proceso
                    del proyecto se ha llevado la VRAM entre la espera y la carga. Es una carrera,
                    no un error del codigo, y morir aqui tira la tirada entera."""
                    print("  carga de `%s` fallida (intento %d): %s" % (q, intento + 1, str(e)[:120]))
                    torch.cuda.empty_cache()
                    self.espera_vram(VRAM_POR_CABEZA.get(q, 0.9) + 1.0, espera=600)
            if q not in self.m:
                raise RuntimeError("no hay VRAM para cargar `%s` tras 6 intentos" % q)

    def tabla_tiempos(self):
        """QUE DEVUELVE: por cabeza, llamadas, ms totales y ms por llamada. Lo que hace falta para el
        informe de coste, y sale del mismo reloj para las cinco, no de medidas de dias distintos."""
        return dict((q, {"llamadas": self.n[q], "ms_total": self.ms[q],
                         "ms_por_llamada": (self.ms[q] / self.n[q]) if self.n[q] else float("nan")})
                    for q in self.m)

    def pose(self, rgb, cajas_xyxy, limpiar=True):
        t0 = time.perf_counter()
        cbs = [[xywh(b)] for b in cajas_xyxy]
        with torch.inference_mode():
            e = a_cuda(self.p["pose"](images=[rgb] * len(cbs), boxes=cbs, return_tensors="pt"))
            o = self.m["pose"](**e)
            r = self.p["pose"].post_process_pose_estimation(o, boxes=cbs, threshold=None)
        out = []
        for j in range(len(cbs)):
            rr = r[j][0] if isinstance(r[j], list) else r[j]
            kp = np.asarray(rr["keypoints"].cpu(), dtype="float64")
            sc = np.asarray(rr["scores"].cpu(), dtype="float64")
            out.append(limpia_keypoints(kp, sc, cajas_xyxy[j]) if limpiar else (kp, sc))
        self.ms["pose"] += 1000.0 * (time.perf_counter() - t0)
        self.n["pose"] += 1
        """Una llamada a `pose` lleva TODAS las cajas del cuadro en un lote, asi que su ms por llamada
        es ms por CUADRO, no por persona. Las cuatro capas densas van una llamada por recorte."""
        return out

    def _densa(self, q, sub, pp, clave, pad):
        t0 = time.perf_counter()
        hh, ww = sub.shape[:2]
        with torch.inference_mode():
            e = a_cuda(self.p[q](images=sub, return_tensors="pt"))
            o = self.m[q](**e)
            kw = {"target_sizes": [(hh, ww)]}
            if pad:
                """Trampa 2: `normal` y `pointmap` llevan do_pad=True. Sin `source_sizes` un cuadro
                720p entra con 58 % de relleno y la salida sale desplazada."""
                kw["source_sizes"] = [(hh, ww)]
            r = getattr(self.p[q], pp)(o, **kw)[0]
        v = r if clave is None else (r[clave] if isinstance(r, dict) else r)
        v = np.asarray(v.cpu())
        self.ms[q] += 1000.0 * (time.perf_counter() - t0)
        self.n[q] += 1
        if clave is None:
            return v.astype("uint8")
        v = np.squeeze(np.nan_to_num(v.astype("float32")))
        if v.ndim == 3 and v.shape[0] in (3, 4):
            v = np.transpose(v, (1, 2, 0))
        return v

    def matting(self, sub):
        return self._densa("matting", sub, "post_process_image_matting", "alpha", False)

    def seg(self, sub):
        return self._densa("seg", sub, "post_process_semantic_segmentation", None, False)

    def normal(self, sub):
        """Trampa 3: la clave de salida es `normals`, no `normal`."""
        return self._densa("normal", sub, "post_process_normal_estimation", "normals", True)

    def pointmap(self, sub):
        return self._densa("pointmap", sub, "post_process_pointmap_estimation", "pointmap", True)


def mascara_tirador(cap, rgb, sel, k, lado, W, H, kp_todos, con_matting=True, guarda_dura=False,
                    holgura_guarda=None):
    """QUE DEVUELVE: dict con el recorte, la mascara elegida y las piezas para poder contradecirla.

    La mascara es (matting >= 0,5  O  seg != Background)  Y  esqueleto propio mas cerca  Y  guarda de
    caja. Con `con_matting=False` se queda en seg, que da 0,001 menos de recall por 3x menos coste.
    """
    c = recorte(sel["B"], k, lado, W, H)
    if c is None:
        return None
    sub = rgb[c[1]:c[3], c[0]:c[2]]
    hh, ww = sub.shape[:2]
    if hh < 8 or ww < 8:
        return None
    g = cap.seg(sub)
    s = g != 0
    a = cap.matting(sub) if con_matting else None
    m = (a >= UMB_ALFA) if a is not None else np.zeros_like(s)
    aj = ajenos_del_cuadro(sel, k, H)
    v = reparto_esqueleto(kp_todos[("t", lado)],
                          [kp_todos[("t", 1 - lado)]] + [kp_todos[("a", i)] for i in range(len(aj))
                                                         if ("a", i) in kp_todos],
                          W, H)[c[1]:c[3], c[0]:c[2]]
    bp = sel["B"][k, lado] * np.array([W, H, W, H])
    gu = guarda_de_caja(c, bp, aj, hh, ww, holgura=holgura_guarda)
    gu_dura = guarda_de_caja(c, [0, 0, 0, 0], aj, hh, ww)
    """`gu_dura` no perdona el solape: borra donde hay caja de un ajeno aunque sea dentro de la caja
    propia. MEDIDA Y DESCARTADA: en los 73 pares con solape cuesta 0,1586 de rec_kp de media y hasta
    0,9811 en el peor (borra al tirador entero), muy por encima del 0,03 que se habia declarado como
    limite. Su fuga_solape de 0,0000 es tautologica: la guarda resta exactamente esa region. Se deja
    en el codigo porque la medida es reproducible, pero el defecto es la blanda."""
    bruto = m | s
    """`bruto` es la union de las dos capas SIN reparto ni guarda. Se devuelve porque sin ella no se
    puede ATRIBUIR una perdida de recall: un recall bajo puede venir de que `seg` y `matting` no han
    encontrado al tirador, o de que el reparto por esqueleto le ha quitado pixeles, o de la guarda de
    caja, y las tres se arreglan de manera distinta. Sin `bruto` las tres se confunden en un numero."""
    base = bruto & v
    return {"c": c, "sub": sub, "seg": g, "alfa": a, "vor": v, "guarda": gu, "guarda_dura": gu_dura,
            "bruto": bruto, "mask": base & (gu_dura if guarda_dura else gu),
            "mask_blanda": base & gu, "mask_dura": base & gu_dura,
            "solo_seg": s & v & gu, "solo_mat": m & v & gu}


V_OSCURO = 80
"""En esgrima el arbitro va de traje oscuro y los tiradores de blanco. Es regla del deporte, no un
umbral ajustado a estos datos. `V` del HSV por debajo de 80 dentro del solape de las dos cajas es
arbitro. La suposicion NO es limpia (careta, guante y zapatillas del tirador tambien son oscuras), asi
que la cifra solo se puede leer contra `ret_oscuro_propio`: el ritmo con que la mascara conserva el
oscuro que SI es del tirador, donde no hay ningun ajeno. Medido: 0,410 en el solape contra 0,708 de
referencia, es decir discrimina 1,7x pero no del todo."""


def mide(res, sel, k, lado, W, H, kp_propio, bgr=None):
    """Las cifras del criterio, mas el control negativo. Ninguna sale de la capa que juzga.

    `fuga_solape` existe porque la metrica de banda es CIEGA dentro de la caja propia, y ahi es
    justo donde quedaba el resto de arbitro que se vio a ojo en c312. Mide, dentro del solape de la
    caja propia con la de un ajeno, cuanto de lo OSCURO enciende la mascara.
    """
    c, mk = res["c"], res["mask"]
    hh, ww = mk.shape
    bp = sel["B"][k, lado] * np.array([W, H, W, H]) - np.array([c[0], c[1], c[0], c[1]])
    propia = np.zeros((hh, ww), bool)
    propia[max(0, int(bp[1])):min(hh, int(bp[3])), max(0, int(bp[0])):min(ww, int(bp[2]))] = True
    aj = np.zeros((hh, ww), bool)
    for o in ajenos_del_cuadro(sel, k, H):
        y0, y1 = int(max(0, o[1] - c[1])), int(min(hh, o[3] - c[1]))
        x0, x1 = int(max(0, o[0] - c[0])), int(min(ww, o[2] - c[0]))
        if y1 > y0 and x1 > x0:
            aj[y0:y1, x0:x1] = True
    banda = ~propia
    b_arb, b_fon = banda & aj, banda & ~aj
    kp, sc = kp_propio
    x, y = kp[:, 0] - c[0], kp[:, 1] - c[1]
    ok = (sc >= KP_MIN) & (x >= 0) & (y >= 0) & (x < ww) & (y < hh)
    rec = float(mk[y[ok].astype(int), x[ok].astype(int)].mean()) if ok.any() else np.nan
    des = np.roll(mk, int(0.5 * ww), axis=1)
    rec_d = float(des[y[ok].astype(int), x[ok].astype(int)].mean()) if ok.any() else np.nan

    def recall_de(m2):
        if m2 is None or not ok.any():
            return np.nan
        return float(m2[y[ok].astype(int), x[ok].astype(int)].mean())

    ok63 = ok.copy()
    ok63[63:] = False
    rec63 = (float(mk[y[ok63].astype(int), x[ok63].astype(int)].mean()) if ok63.any() else np.nan)
    """RECALL SOBRE LOS 63 PUNTOS QUE EL PROYECTO USA, y hace falta porque `rec_kp` esta DOMINADO POR
    LA CARA: de los 308 puntos, 245 son cara (del 63 en adelante) y el proyecto los descarta porque
    los tiradores llevan careta. Asi que `rec_kp` mide sobre todo si la mascara cubre la CABEZA, y un
    fallo que se coma la cabeza y deje el cuerpo entero hunde `rec_kp` sin tocar ni un punto de los
    que se van a usar. Las dos cifras van juntas: `rec_kp` para comparar con lo ya medido y `rec_63`
    para saber que se pierde de lo que importa."""

    cons_px, comp, huecos = np.nan, 0, np.nan
    if res.get("bruto") is not None:
        """RECALL DE PIXELES, y hace falta porque `rec_kp` NO PUEDE VER UN AGUJERO. Un keypoint cae
        en un punto; los agujeros caen ENTRE punto y punto, asi que una mascara con la mitad del
        cuerpo comida sigue conteniendo el 99 % de los keypoints. Lo encontro el coordinador mirando
        la figura de `WhisperedTenderDeinonychus` c26, no ninguna de las cifras de esta funcion.

        Se mide DENTRO DE LA CAJA PROPIA y sobre los pixeles que `seg` o `matting` daban por persona:
        asi la cifra no premia ni castiga lo que las capas no encontraron, que ya lo cuenta
        `perdida_capas`. Es "de lo que habia dentro de mi caja, cuanto conserva la limpieza".
        """
        dentro = res["bruto"] & propia
        if dentro.sum() > 200:
            cons_px = float((mk & dentro).sum() / dentro.sum())
        if mk.sum() > 200:
            n_lab, lab, stats, _cen = cv2.connectedComponentsWithStats(
                mk.astype("uint8"), 8, cv2.CV_32S)
            areas = stats[1:, cv2.CC_STAT_AREA] if n_lab > 1 else np.zeros(0)
            comp = int((areas >= 0.02 * mk.sum()).sum())
            """Componentes con al menos el 2 % del area de la mascara. Una persona es UNA pieza; si
            la mascara sale en seis trozos grandes, algo la ha cortado. El 2 % esta para no contar
            como trozo el ruido de borde de un par de pixeles."""
            relleno = mk.copy().astype("uint8")
            hh2, ww2 = relleno.shape
            ff = np.zeros((hh2 + 2, ww2 + 2), "uint8")
            cv2.floodFill(relleno, ff, (0, 0), 1)
            interior = (relleno == 0)
            huecos = float(interior.sum() / max(mk.sum() + interior.sum(), 1))
            """`huecos` es el area de los agujeros INTERIORES en fraccion de la silueta rellena. El
            relleno por inundacion desde la esquina 0,0 solo funciona si esa esquina esta fuera de la
            mascara; con el margen del 12 % del recorte lo esta siempre."""

    rec_bruto = recall_de(res.get("bruto"))
    rec_vor = recall_de(res.get("bruto") & res["vor"] if res.get("bruto") is not None else None)
    """ATRIBUCION DE LA PERDIDA, en tres sumandos que suman 1 - rec_kp:
      1 - rec_bruto        lo que `seg` y `matting` no encontraron. No es culpa de la limpieza.
      rec_bruto - rec_vor  lo que se llevo el reparto por esqueleto mas cercano.
      rec_vor  - rec_kp    lo que se llevo la guarda de caja.
    Sin este reparto, un recall bajo no dice que arreglar."""
    f_sol, f_sol_ctrl, px_sol = np.nan, np.nan, 0
    if bgr is not None:
        vhs = cv2.cvtColor(bgr[c[1]:c[3], c[0]:c[2]], cv2.COLOR_BGR2HSV)[:, :, 2]
        osc = vhs < V_OSCURO
        sol = propia & aj
        px_sol = int((sol & osc).sum())
        if px_sol > 200:
            f_sol = float((mk & sol & osc).sum() / (sol & osc).sum())
        lejos = propia & ~aj & osc
        if lejos.sum() > 200:
            """NIVEL DE REFERENCIA, sin el que `fuga_solape` no se puede leer: la careta, el guante y
            las zapatillas del tirador tambien son oscuras, asi que hay que saber a que ritmo la
            mascara conserva el oscuro QUE SI ES SUYO, donde no hay ningun ajeno. Si el ritmo en el
            solape es mas bajo que este, la mascara esta discriminando; si es igual o mayor, no."""
            f_sol_ctrl = float((mk & lejos).sum() / lejos.sum())
    return {"k": k, "lado": lado, "rec_kp": rec, "rec_desp": rec_d, "nkp": int(ok.sum()),
            "rec_63": rec63, "nkp63": int(ok63.sum()),
            "conserva_px": cons_px, "componentes": comp, "huecos": huecos,
            "rec_bruto": rec_bruto, "rec_tras_esqueleto": rec_vor,
            "perdida_capas": 1.0 - rec_bruto, "perdida_esqueleto": rec_bruto - rec_vor,
            "perdida_guarda": rec_vor - rec,
            "residuo": float(mk.mean()),
            "fuga_arb": float((mk & b_arb).sum() / b_arb.sum()) if b_arb.sum() > 200 else np.nan,
            "fuga_fondo": float((mk & b_fon).sum() / b_fon.sum()) if b_fon.sum() > 200 else np.nan,
            "fuga_solape": f_sol, "ret_oscuro_propio": f_sol_ctrl,
            "resto_arb_frac": (float((mk & propia & aj & osc).sum() / max(1, mk.sum()))
                               if bgr is not None else np.nan),
            "px_banda_arb": int(b_arb.sum()), "px_solape_oscuro": px_sol}


def empaqueta(res, normal=None, Z_cuadro=None):
    """QUE DEVUELVE: el recorte de un tirador listo para consumir, con el relleno declarado.

    El contenedor es POR RECORTE y no de cuadro completo porque las dos mascaras cubren el 4,6-4,8 %
    del cuadro: guardar cuadro completo desperdicia 21-22x, medido.

    El relleno depende de si el cero es un valor legal de la capa, no del gusto:
      alfa (matting) -> 0    alfa 0 ES el fondo, el cero no miente
      clase (seg)    -> 0    la clase 0 ES Background, el cero no miente
      normal         -> NaN  el vector cero no es una normal unitaria, asi que valdria de centinela,
                             pero un 0 se promedia sin avisar en una convolucion. NaN obliga a mirar
                             el canal `mask`.
      z (pointmap)   -> NaN  obligatorio: z = 0 es una profundidad legal y seria indistinguible de un
                             dato bueno.
    El canal `mask` va siempre, porque es lo unico que permite al consumidor distinguir "fuera del
    tirador" de "dentro y con este valor".
    """
    mk = res["mask"]
    out = {"recorte": np.asarray(res["c"], dtype="int32"), "mask": mk,
           "alfa": np.where(mk, res["alfa"], 0.0).astype("float16") if res["alfa"] is not None else None,
           "clase": np.where(mk, res["seg"], 0).astype("uint8")}
    if normal is not None and normal.shape[:2] == mk.shape:
        n = normal.astype("float32").copy()
        n[~mk] = np.nan
        out["normal"] = n.astype("float16")
    if Z_cuadro is not None:
        c = res["c"]
        zz = Z_cuadro[c[1]:c[3], c[0]:c[2]].astype("float32").copy()
        zz[~mk] = np.nan
        out["z"] = zz.astype("float16")
    return {k: v for k, v in out.items() if v is not None}


def pinta_normal(n):
    """Normal a color: cada eje a un canal. Fuera de la mascara queda negro."""
    u = n / np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-6)
    return (127.5 * (u[:, :, ::-1] + 1.0)).astype("uint8")


def pinta_z(Z, mk):
    """Profundidad a color solo con los pixeles validos: si se normaliza con todo el cuadro, el rango
    lo fija el fondo y los dos tiradores salen del mismo color."""
    v = Z[mk]
    if v.size < 10:
        return np.zeros(Z.shape + (3,), dtype="uint8")
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    n = np.clip((Z - lo) / max(hi - lo, 1e-6), 0, 1)
    col = cv2.applyColorMap((n * 255).astype("uint8"), cv2.COLORMAP_TURBO)
    col[~mk] = 0
    return col


def pinta_seg(g):
    rng = np.random.default_rng(7)
    lut = rng.integers(40, 255, size=(30, 3)).astype("uint8")
    lut[0] = 0
    return lut[np.clip(g, 0, 29)]

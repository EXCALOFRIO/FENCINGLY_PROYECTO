"""PASOS 3 y 4 · La POSICION de cada tirador en metros sobre la pista, y los EVENTOS marcados.

=======================================================================================
LO QUE ENTREGA, en las palabras del usuario
=======================================================================================

  *"tener como una pista en una grafica, y como una linea en 2D por donde va, en el eje de la
  LONGITUD de la pista y la PROFUNDIDAD de la pista"* -> `apoyos()` devuelve (T, 2, 2) en metros,
  con `x` a lo largo de la pista y `y` a lo ancho, origen en la linea central.

  *"cada vez que hace un fondo ponerlo; cada vez que estira el brazo marcarlo como un keypoint;
  cada marca donde empieza y acaba"* -> `eventos()` devuelve una lista de eventos, y CADA UNO lleva
  su cuadro, su posicion en metros sobre la pista y su MAGNITUD en metros, que es el
  *"cuanto ha medido"* que pidio el usuario.

  Tipos: `fondo`, `marcha`, `romper`, `brazo`, `accion` (con inicio y fin).

=======================================================================================
DE DONDE SALE EL PUNTO DE APOYO, y por que no es el centro de la caja
=======================================================================================

El punto que toca la pista es el PIE. La caja de YOLO da su borde inferior, que es la fila del pie
mas bajo, pero su centro horizontal es el centro del CUERPO, y en un fondo el cuerpo esta muy por
detras del pie de delante. Asi que la linea base (centro de la caja) mete un error sistematico que
crece justo en el momento que mas importa.

Con la pose de 308 puntos se usan los tobillos (13, 14) y los pies (15-20). El mapa de indices es el
MEDIDO en `04_documentacion/BITACORA_2026-09.md`, no el del plan, que caia en la careta:

  5-6 hombros | 7-8 codos | 9-10 caderas | 11-12 rodillas | 13-14 TOBILLOS | 15-20 PIES
  21-41 mano A (raiz 21) | 42-62 mano B (raiz 42) | 63+ cara

Cuando no hay pose para un cuadro, se cae al borde inferior de la caja y el cuadro queda marcado en
`origen`, para que nadie confunda un apoyo medido con uno de respaldo.

=======================================================================================
CRITERIOS DE LOS EVENTOS, DECLARADOS ANTES DE MEDIR
=======================================================================================

Todos en metros y cuadros, y todos con su magnitud. La velocidad de referencia sale de la fisica del
gesto: un fondo son del orden de 1,0-1,5 m en 0,2-0,3 s, o sea 4-6 m/s, que a 30 fps son 0,13-0,20 m
por cuadro.

  FONDO      la separacion entre los dos pies CRECE de golpe y el cuerpo AVANZA:
             `d_pies` sube al menos `FONDO_SALTO_M` = 0,35 m en `FONDO_VENTANA` = 8 cuadros,
             y el pie de delante avanza al menos `FONDO_AVANCE_M` = 0,25 m en la misma ventana.
             Instante = el cuadro de maxima derivada de `d_pies`. Magnitud = avance del pie de
             delante desde el inicio hasta el maximo de separacion.

  MARCHA     un PASO hacia el rival. Un paso es un tramo en que un pie se mueve mientras el otro
             esta quieto: velocidad del pie movil por encima de `PASO_V_MS` = 0,9 m/s y del otro por
             debajo de la mitad. Se cuentan los tramos, no los cuadros. Magnitud = lo que avanza el
             centro del cuerpo en el tramo. Signo hacia el rival = marcha.
             Y los tramos consecutivos del MISMO sentido se FUSIONAN: ver `_pasos`, porque un paso
             de esgrima son DOS tramos, primero un pie y luego el otro, y medir solo uno da la
             mitad del paso.

  ROMPER     lo mismo con el signo contrario, o sea alejandose del rival. Es el evento que el
             usuario anadio despues y que no estaba en la lista original.

  ALTO     el tirador venia desplazandose y DEJA DE DESPLAZARSE. **Se llamaba `parada` y el nombre
             estaba mal**, porque en esgrima una PARADA es la accion defensiva de desviar la hoja del
             atacante con la propia, y es la bisagra del reglamento porque transfiere la prioridad a
             la respuesta (parada-respuesta). Eso es un evento de ARMA y esto es un evento de PIES.

             **LA PARADA EN EL SENTIDO DE ESGRIMA NO ESTA IMPLEMENTADA NI SE PUEDE CON ESTE
             MATERIAL**: el paso de prioridad se cerro con 0 paradas anotadas y midio que el contacto
             de hojas no es verificable a ojo en este corpus. Quien lea "71 altos" no debe entender
             que hay paradas-respuesta detectadas, porque no hay ninguna.

             Criterio: recorre al menos
             `ALTO_ANTES_M` = 0,25 m en `ALTO_VENTANA` = 8 cuadros y en los 8 siguientes
             recorre `ALTO_DESPUES_M` = 0,08 m o menos. Magnitud = lo que traia recorrido.

  BOTE       las dos plantas se separan del suelo A LA VEZ (salto, balestra o appel; el criterio NO
             los distingue). Se mide en la IMAGEN y no en metros de pista, porque un pie en el aire
             se proyecta desplazado y eso es exactamente lo que la homografia NO puede medir.

  BRAZO      extension del brazo armado: el eje `hombro -> raiz de la mano` se ALARGA. Se mide en
             fraccion del largo del torso (hombro-cadera) para que no dependa de la perspectiva,
             porque el brazo NO esta en el plano del suelo y la homografia no lo mide. Dispara
             cuando pasa de `BRAZO_UMBRAL` = 1,15 veces la mediana del propio clip y esta subiendo.
             Magnitud = el alargamiento en fraccion de torso. La columna `completa` marca las
             extensiones que llegan a `BRAZO_COMPLETA` = 0,90 veces el p98 del propio clip.

  ACCION     tramo entre que alguno de los dos empieza a moverse y que los dos vuelven a estar
             quietos: velocidad del centro por encima de `ACCION_V_MS` = 0,5 m/s para empezar y por
             debajo durante `ACCION_QUIETO` = 10 cuadros para acabar.

LO QUE NINGUNO DE ESTOS CRITERIOS ES: una definicion de esgrima. Un maestro distingue un fondo de
una flecha y una marcha de un paso-fondo por cosas que no estan en la geometria de los pies. Aqui se
detecta lo que se puede medir y se le pone el nombre que mas se le parece, y la verificacion a ojo
del paso 4 dice cuanto se parece de verdad.

=======================================================================================
LA SERIE TEMPORAL CONSULTABLE, que es lo que el usuario pidio despues
=======================================================================================

  *"poder saber la TERCERA MARCHA DEL TIRADOR DE LA DERECHA y poder ubicarla en el mapa ese de la
  pista, para saber en que momento se hizo y cuanto midio"*

`serie()` devuelve una fila POR EVENTO, no por cuadro, con `indice_del_tipo` numerado 1, 2, 3... por
(lado, tipo) en orden de cuadro. Asi que esa consulta es una linea:

    ev.query("lado == 1 and tipo == 'marcha' and indice_del_tipo == 3")

Cada fila lleva ademas `origen_max` y `hueco_max` tomados del NPZ de poses en su propio rango de
cuadros: un evento detectado sobre una pose INTERPOLADA es mas debil, y con `hueco > 24` el error de
la pose inventada no esta ni medido (seccion 21.4 de `04_documentacion/BITACORA_2026-09.md`).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pista as PI

TOBILLOS = (13, 14)
PIES = (15, 16, 17, 18, 19, 20)
HOMBROS = (5, 6)
CADERAS = (9, 10)
CODOS = (7, 8)
RAICES = (21, 42)
CONF_MIN = 0.30
"""Confianza minima de un punto de la pose. Es el mismo umbral que usa `cono_lib`, y el proyecto ya
midio que la confianza NO acota a 1: se han visto valores de 1,304, asi que un umbral que suponga un
maximo de 1 esta mal puesto. 0,30 no lo supone."""

FONDO_VENTANA = 8
FONDO_SALTO_M = 0.35
FONDO_AVANCE_M = 0.25
GUARDIA_NOMINAL_M = 0.70
FONDO_SALTO_FRAC = 0.50
FONDO_AVANCE_FRAC = 0.36
"""LOS UMBRALES DEL FONDO SE NORMALIZAN POR LA GUARDIA DEL PROPIO CLIP, y no es un ajuste: es que
tenerlos en metros ABSOLUTOS los hacia dependientes del error de escala de la homografia.

EL DEFECTO, con su cifra. El usuario lo vio en el video a camara lenta: *"pasa a marcha, a fondo y
luego marcha otra vez. Detecta ahi un fondo que no es fondo."* Y la secuencia de
`ForcefulAntiqueAlbacoretuna` lado 0 es literalmente `MFMFMFMFMMFMRFMRRMRMFMFMFMFMFMFM`: dieciseis
alternancias. En los 11 clips habia **73 solapes de tipos distintos del mismo tirador**.

La hipotesis, que la apunto el coordinador y es la economica: si la escala de un clip esta inflada,
los 0,35 m de crecimiento de separacion y los 0,25 m de avance se cruzan **con un paso normal**.
Medido, correlacion entre el factor de escala del clip (su `dpies_p50` contra la guardia nominal de
0,70 m) y los fondos por segundo:

    **r = 0,83**

`ForcefulAntiqueAlbacoretuna` tiene el factor mas alto (2,14) y la tasa mas alta (1,50 fondos/s);
`UglyNiceAplomadofalcon` tiene 0,92 y 0,11. La hipotesis se confirma.

EL COEFICIENTE SE ELIGE PARA REPRODUCIR LOS NUMEROS DECLARADOS, no para ajustar nada: con la guardia
nominal de 0,70 m, `0,50 x 0,70 = 0,35 m` y `0,36 x 0,70 = 0,25 m`, o sea exactamente
`FONDO_SALTO_M` y `FONDO_AVANCE_M`. Es el mismo criterio de la seccion 23.8 hecho inmune a la escala.

Efecto medido en los 11 clips: **68 fondos -> 52 (-24 %)**, y donde mas baja es justo donde la escala
esta peor (`Forceful` 18 -> 7, `AdorableVeneratedDuiker` 8 -> 4). En `GentleQuerulousElephantseal`
baja de 6 a 0, que es el caso que hay que mirar con la verdad a ojo y no dar por bueno."""
ARB_RAZON_ATRAS = 0.45
ARB_PICO_GUARDIA = 1.25
"""EL DISCRIMINADOR MARCHA / FONDO, y la clave es que NO es la magnitud.

Las magnitudes se pisan (fondo 0,64 m contra marcha 0,34 m de mediana, con colas solapadas), asi que
por ahi no se separan nunca. Lo que los separa es **el pie de atras**, y es la definicion de esgrima:

  MARCHA: el pie de delante avanza y **el de atras LE SIGUE**. La separacion crece y vuelve.
  FONDO:  el de delante sale y **el de atras SE QUEDA CLAVADO**, que es lo que da la extension.

Dos rasgos, ninguno la magnitud:

  `razon_pie_atras`     = |desplazamiento del pie de ATRAS| / |del de DELANTE|. Frontera declarada
                          `ARB_RAZON_ATRAS` = 0,45. Marcha cerca de 1, fondo cerca de 0.
  `pico_dpies_guardia`  = pico de la separacion de pies / guardia del clip. Frontera declarada
                          `ARB_PICO_GUARDIA` = 1,25.

LOS RASGOS SE MIDEN EN LA FASE DE EXTENSION (del inicio al cuadro del PICO de separacion) Y NO EN
TODA LA VENTANA, y esto fue una correccion medida, no una preferencia: midiendolos de extremo a
extremo de la ventana el acierto salia **4 de 6**, porque cuando la ventana incluye la RECUPERACION
el pie de delante vuelve, su desplazamiento neto se anula y la razon del pie de atras se dispara. En
`AliveIgnorantBarebirdbat` l1 c273-285 la razon pasaba de **0,97** (que dice marcha) a **0,15** (que
dice fondo, y a ojo es un fondo) solo con medirla hasta el pico.

ACIERTO MEDIDO contra la verdad a ojo de los 6 pares solapados de los tres tiradores contados:
**5 de 6 = 83 %**, con matriz de confusion `marcha -> fondo 0` y `fondo -> marcha 1`. El criterio
declarado antes era >= 80 %, asi que se adopta.

Y LO QUE NO SE HACE, que es lo importante: con umbrales de 0,55 y 1,30 el acierto sube a **6 de 6 =
100 %** en esos mismos seis casos. **No se adoptan**, porque eso es elegir el umbral mirando el
resultado sobre seis ejemplos. Se publican los dos numeros y se usa el declarado.

El unico fallo es `GentleQuerulousElephantseal` l0 c275-287, con razon 0,48 contra la frontera de
0,45: un caso de frontera donde el pie de atras se movio casi la mitad que el de delante. Puede ser
que mi etiqueta a ojo sea la equivocada; con n=1 no se puede decidir."""
PASO_V_MS = 1.25
"""Velocidad LONGITUDINAL del pie, sobre la posicion suavizada, por encima de la cual ese pie se
considera en movimiento.

NO ES EL 0,90 m/s DE LA SECCION 23.8: ese estaba puesto sobre otra senal (modulo en 2D de la
diferencia de un cuadro de la posicion cruda) y no se puede heredar. Recalibrado sobre la senal
nueva con la regla del proyecto, 3 veces el suelo de ruido: el ruido p95 de `_vx_serie` medido sobre
el RESIDUO de las 16 series pie-tirador de los 4 clips publicables con pose sale **0,418 m/s de
mediana** (0,063 a 2,263), asi que el umbral son 1,25 m/s."""
PASO_MIN_CUADROS = 3
PASO_HUECO = 8
"""Cuadros de hueco que se admiten entre la zancada de un pie y la del OTRO para que sean el mismo
paso. 8 cuadros son 0,27 s a 30 fps, que es el doble apoyo largo de una marcha lenta de preparacion.
Medido: emparejando con este hueco, 51 de los 87 pasos salen con los dos pies."""
PASO_MIN_M = 0.10
PASO_FUSION = 6
"""Solo lo usa `_pasos_v1(fusion=True)`, que es la version 2 descartada. Ver `_pasos`."""
ACCION_V_MS = 0.50
ACCION_QUIETO = 10
BRAZO_UMBRAL = 1.15
BRAZO_MIN_CUADROS = 3
BRAZO_COMPLETA = 0.90
"""Fraccion del p98 del propio clip por encima de la cual una extension se llama COMPLETA.

ES RELATIVA AL CLIP Y SE DICE: no hay ninguna medida absoluta del largo del brazo de un tirador, asi
que "completa" solo puede significar "cerca de lo mas que se ha estirado en este clip". Se usa el p98
y no el maximo porque el maximo de una serie de 300 cuadros es un cuadro suelto y puede ser ruido."""
BRAZO_COMPLETA_MEDIANA = 1.25
ALTO_VENTANA = 8
ALTO_ANTES_M = 0.25
ALTO_DESPUES_M = 0.08
ALTO_MARGEN = 3.0
ALTO_TOPE_M = 0.20
"""EL ALTO: el criterio se CALIBRA con el ruido del propio clip, y los dos criterios que escribi
antes de medir fallaron los dos su propia prueba de margen. Se cuentan porque el tercero solo se
entiende con los dos primeros delante.

Lo declarado antes de medir: el umbral tiene que estar `ALTO_MARGEN` = 3 veces por encima del
suelo de ruido de la senal con la que se mide. Y el suelo de ruido se mide como debe medirse, sobre
el RESIDUO `x - suavizada`, que es ruido por construccion y no contiene movimiento del tirador.

  INTENTO 1, velocidad instantanea. `vx` por diferencia centrada de +-2 cuadros. Ruido p95 medido en
  los 8 tiradores de los 4 clips: **0,06 a 1,08 m/s**, mediana **0,35 m/s**. El umbral de 0,60 m/s
  que habia puesto tenia 1,7x de margen, no 3x, y en un tirador (`ForcefulAntiqueAlbacoretuna`
  lado 1) el ruido era MAS GRANDE que el umbral. Descartado.

  INTENTO 2, desplazamiento fijo en ventana de 8 cuadros. Ruido p95 del residuo en esa ventana:
  **0,008 a 0,132 m**, mediana **0,048 m**. Asi que 0,25 m para "venia moviendose" tiene 5,2x de
  margen y PASA, pero 0,08 m para "esta quieto" tiene **1,7x** sobre la mediana y **0,6x** sobre el
  peor tirador: o sea que en el peor clip el ruido solo ya cruza el umbral. Descartado tambien.

  INTENTO 3, el que se usa: el umbral de "esta quieto" es `max(0,08 ; 3 x ruido_p95_de_ese_lado)`,
  medido en ese clip y ese tirador. Si eso pasa de `ALTO_TOPE_M` = 0,20 m, el alto se declara
  **NO MEDIBLE en ese lado** y no se emite ningun evento, porque un umbral de 0,20 m en 8 cuadros son
  0,75 m/s y eso no es estar quieto, es andar.

CONSECUENCIA QUE HAY QUE LEER: una "alto" de este detector significa que el tirador baja de
0,3-0,7 m/s, NO que este inmovil. La cifra exacta por clip va en la columna `umbral_quieto_m` del
CSV de la serie, que es lo unico honrado cuando el umbral es adaptativo."""
BOTE_FRAC = 0.05
BOTE_BASE_VENTANA = 15
BOTE_MIN_CUADROS = 2
BOTE_MARGEN = 3.0
BOTE_TOPE = 0.06
BOTE_G = 9.81
BOTE_VUELO_FRAC = 0.5
"""CONTROL BALISTICO DEL BOTE, y es el que caza el unico evento que se habia colado.

EL PROBLEMA, encontrado por el coordinador leyendo el CSV: salia **1 fila de tipo `bote`**
(`WastefulYawningAmericanquarterhorse` lado 1, cuadros 194-195, magnitud 8,5 cm, duracion 0,033 s)
cuando yo habia declarado **0 eventos** y un minimo detectable de 10,4 cm.

POR QUE SE COLO, y no es que cambiara el criterio: el umbral es ADAPTATIVO,
`max(BOTE_FRAC, 3 x ruido_de_ese_lado)`, y en ese clip el ruido es el mas bajo de toda la muestra
(0,0143 del alto de caja contra 0,0198 de mediana), asi que 3x sale 0,043 y manda el suelo fijo de
0,05. Los 10,4 cm eran el caso MEDIANO, no el umbral de ese clip. O sea que la cifra publicada era
del caso tipico y la frase "0 eventos" no era una consecuencia del criterio: era el resultado
observado en 4 clips, y con 11 dejo de valer.

Y LA INCOHERENCIA DE VERDAD ES DE FISICA, que es la que da el arreglo: si las dos plantas se separan
del suelo `h`, el cuerpo esta en caida libre y el tiempo de vuelo esta FIJADO,
`t = 2 sqrt(2h/g)`. No es un umbral elegido, es una igualdad:

  despegue  5,0 cm -> vuelo 0,202 s = **6,1 cuadros** a 30 fps
  despegue  8,5 cm -> vuelo 0,263 s = **7,9 cuadros**
  despegue 15,0 cm -> vuelo 0,350 s = 10,5 cuadros

El evento que se colo dura **1 cuadro** y dice 8,5 cm: son **8 veces menos vuelo del que la gravedad
permite**, asi que no es un bote, es un salto del keypoint del pie. Se exige que la duracion llegue
al menos a `BOTE_VUELO_FRAC` = 0,5 del vuelo que implica la altura medida, con la mitad de margen
porque el detector solo ve la parte del vuelo en que las dos plantas pasan del umbral, no el vuelo
completo. Con eso el evento se rechaza **por una razon fisica y no por subir un numero**.

`eventos()` comprueba con un ASERTO que ninguna fila emitida incumple este control."""
"""EL BOTE NO ESTA ESTABLECIDO, y esta es la medida que lo dice. Se deja el detector porque la cifra
es el resultado, no un fracaso que haya que esconder.

POR QUE NO SE MIDE EN METROS DE PISTA: la homografia mapea el PLANO DEL SUELO. Un pie a 10 cm del
suelo se proyecta desplazado hacia el fondo `altura x distancia / altura_de_camara`, que es el mismo
defecto que hace que la coordenada transversal no sea creible (23.7). Medir un salto con la
homografia del suelo es medirlo con la regla que ese salto rompe. El alto de la CAJA si es una
reglilla valida para lo vertical, porque escala con la perspectiva igual que el tirador.

EL SUELO DE RUIDO SALE DE LA COLA NEGATIVA DE LA PROPIA SENAL, y esa es la parte que hace que la
medida valga: la senal es `nivel_del_suelo - planta`, y un pie **no puede estar por debajo del
suelo**. Asi que todo lo negativo es ruido de medida puro, sin ninguna hipotesis. Medido sobre las 8
series tirador-clip de los 4 clips publicables con pose:

  `|p05|` de la senal  **0,0198** del alto de la caja de mediana (0,0151 a 0,0347), o sea **3,5 cm**
                       en un tirador de 1,75 m
  umbral a 3x          **0,0595**, o sea **10,4 cm** de despegue SIMULTANEO de las dos plantas
  cuadros que lo pasan **2 de 2.782** (8 series), y el pico de cada serie se queda en 0,3 a 4,3x

Y AHI ESTA EL PROBLEMA, que no se arregla bajando el umbral: una balestra o un appel de florete
despegan del orden de **5-8 cm**, que cae POR DEBAJO de los 10,4 cm que hacen falta para salir del
ruido. O sea que con esta pose y esta resolucion el bote no se puede afirmar, y bajar el umbral a
5 cm seria ponerlo dentro del ruido y llamar bote al temblor del keypoint del pie.

Se emiten los eventos que pasan el umbral calibrado, con su razon `sobre_ruido` en el CSV, y el tipo
va marcado como NO ESTABLECIDO en el README. Dos eventos en cuatro clips no son un detector."""
SALTO_FISICO_M = 0.30
"""Salto maximo creible del apoyo de un tirador entre cuadros consecutivos, en metros.

DE DONDE SALE, y no es un umbral inventado: un fondo son 1,0-1,5 m en 0,2-0,3 s, o sea 4-6 m/s, que
a 30 fps son 0,13-0,20 m por cuadro. 0,30 m por cuadro son 9 m/s, mas rapido que un esgrimista en el
mejor fondo de su vida. Todo lo que pase de ahi es ruido de medida, y se usa para MEDIRLO, no para
taparlo: la fraccion de cuadros que lo pasan es una de las cifras del informe."""


def _fps(video: str) -> float:
    """QUE HACE: los fotogramas por segundo del clip. El corpus va de 17,96 a 59,82, asi que no se
    puede suponer 30."""
    import cv2
    cap = cv2.VideoCapture(video)
    v = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return v if 5.0 < v < 240.0 else 30.0


def carga_poses(clip: str):
    """QUE HACE: las poses de 308 puntos ya calculadas, si existen.

    Devuelve `None` si no hay NPZ, y entonces todo funciona con el borde inferior de la caja como
    respaldo. Los NPZ estan en `01_extraccion/pruebas/poses_*.npz` y su fila `k` corresponde al
    cuadro `k` del video, porque `video_revision.py` los guarda ya alineados; aun asi se comprueba
    la longitud contra `T` y se avisa si no cuadra, que es la averia del desfase `f0` que ya costo
    medio dia al proyecto.
    """
    import rutas as RU
    p = RU.TRABAJO / ("poses_%s.npz" % clip)
    if not p.exists():
        return None
    with np.load(p, allow_pickle=True) as z:
        out = {"KP": z["KP"], "SC": z["SC"], "T": int(z["T"]), "origen": z["origen"]}
        """`hueco` ES EL LARGO DEL HUECO DEL QUE SALE CADA POSE INVENTADA, y se carga porque sin el no
        se puede decir lo mas importante de un evento: si esta detectado sobre una pose MEDIDA o
        sobre una inventada. La seccion 21.4 midio que con `hueco > 24` el error de la pose inventada
        no esta ni medido, asi que ese caso hay que poder marcarlo en el CSV y no se puede si esta
        columna no llega hasta aqui."""
        out["hueco"] = z["hueco"] if "hueco" in z.files else None
        return out


def _pie_pose(KP, SC, k, lado):
    """QUE HACE: los dos apoyos (los dos pies) de un tirador en un cuadro, en pixeles.

    Cada pie se toma como el punto MAS BAJO entre su tobillo y sus tres puntos de pie, porque el que
    toca el suelo es el mas bajo y el tobillo esta unos centimetros por encima.
    """
    kp, sc = KP[k, lado], SC[k, lado]
    out = []
    for j, (tob, tri) in enumerate(((13, (15, 16, 17)), (14, (18, 19, 20)))):
        cand = [i for i in (tob,) + tri if sc[i] >= CONF_MIN and np.isfinite(kp[i]).all()]
        if not cand:
            out.append(None)
            continue
        i = max(cand, key=lambda z: kp[z, 1])
        out.append(np.asarray(kp[i], "float64"))
    return out


def apoyos(res, poses=None) -> dict:
    """QUE HACE: la posicion en METROS de los dos tiradores sobre la pista, cuadro a cuadro.

    Devuelve
      `C`      (T, 2, 2) el centro entre los dos pies, en metros. Es "donde esta el tirador".
      `P1 P2`  (T, 2, 2) cada pie por separado, que es lo que necesita el detector de fondo.
      `origen` (T, 2) 0 pose, 1 borde de caja, -1 nada
      `dpies`  (T, 2) separacion entre los dos pies, en metros
      `dist`   (T,) distancia entre los dos tiradores en metros, que es la variable que de verdad
               importa en esgrima

    LA CAMARA SE CORRIGE CUADRO A CUADRO. Todo punto de imagen se pasa primero por la similitud de
    estabilizacion `Mk[k]`, que lo lleva al sistema del cuadro de referencia, y solo despues por la
    homografia. Sin eso, un clip con paneo de 200 px, que en este corpus es lo normal, mete un error
    metrico que crece a lo largo del clip.
    """
    hg, r, Mk = res["_hg"], res["_r"], res["_Mk"]
    B, pres, mask, W, H, T = r["B"], r["pres"], r["mask"], r["W"], r["H"], r["T"]
    C = np.full((T, 2, 2), np.nan)
    P1 = np.full((T, 2, 2), np.nan)
    P2 = np.full((T, 2, 2), np.nan)
    org = np.full((T, 2), -1, "int8")
    KP = SC = None
    if poses is not None and poses["KP"].shape[0] >= T:
        KP, SC = poses["KP"], poses["SC"]
    for k in np.nonzero(mask)[0]:
        M = Mk[k]
        for lado in (0, 1):
            if not pres[k, lado]:
                continue
            pies = _pie_pose(KP, SC, k, lado) if KP is not None else [None, None]
            if pies[0] is not None and pies[1] is not None:
                q = [PI.a_metros(hg, _lleva(M, p)) for p in pies]
                P1[k, lado], P2[k, lado] = q[0], q[1]
                C[k, lado] = 0.5 * (q[0] + q[1])
                org[k, lado] = 0
                continue
            b = B[k, lado]
            if not np.isfinite(b).all():
                continue
            p = np.array([0.5 * (b[0] + b[2]) * W, b[3] * H])
            C[k, lado] = PI.a_metros(hg, _lleva(M, p))
            org[k, lado] = 1
    """CONVENIO DEL EJE LONGITUDINAL, DECLARADO: **la x CRECE hacia la DERECHA DE LA IMAGEN**, y por
    tanto el `lado 0` (el tirador de la izquierda de la imagen) tiene siempre la x mediana MENOR que
    el `lado 1`.

    HAY QUE DECLARARLO porque el signo de la coordenada longitudinal que sale de la homografia lo
    decide cual de los dos bordes de la pista salio como A en el barrido, que es un detalle interno y
    no significa nada: sin esta correccion el signo de `x_inicio_m` y `x_fin_m` cambiaria de
    significado de un clip a otro. El usuario aviso de que el mapa le salia espejado, y aunque la
    medida dice que no lo esta (ver abajo), el convenio tenia que estar escrito igual.

    COMPROBADO EN LOS ONCE CLIPS, y por dos vias independientes:

      `lado 0` a la izquierda EN LA IMAGEN (centro de caja en pixeles): **11 de 11**
      `lado 0` con la x METRICA menor:                                  **11 de 11**
      CRUCES, o sea cuadros con `x_lado0 > x_lado1`: **0 de 2.896** en metros y **0 de 2.896** en
      pixeles de imagen.

    El de los cruces es un control fisico nuevo y gratis: **dos floretistas no pueden cruzarse**, asi
    que un solo cuadro con la izquierda a la derecha de la derecha seria o una posicion mal medida o
    la pareja mal asignada. Sale cero, lo que dice que la asignacion de lados es solida en estos once
    clips. No dice nada de los otros 67."""
    x0 = np.nanmedian(C[:, 0, 0])
    x1 = np.nanmedian(C[:, 1, 0])
    if np.isfinite(x0) and np.isfinite(x1) and x0 > x1:
        for M in (C, P1, P2):
            M[:, :, 0] *= -1.0
    x0 = np.nanmedian(C[:, 0, 0])
    x1 = np.nanmedian(C[:, 1, 0])
    if np.isfinite(x0) and np.isfinite(x1):
        assert x0 <= x1, ("el convenio del eje no se cumple: lado0 %.2f > lado1 %.2f" % (x0, x1))
    dpies = np.abs(P1[:, :, 0] - P2[:, :, 0])
    """LA SEPARACION DE PIES SE MIDE SOLO A LO LARGO DE LA PISTA, y hay dos razones, una de esgrima y
    una de medida.

    De esgrima: un fondo separa los pies A LO LARGO de la pista; lo que hagan a lo ancho no es el
    gesto.

    De medida, que es la que manda: la coordenada transversal es la direccion de la PROFUNDIDAD de la
    camara, o sea la peor medida de las dos, y meterla en la separacion mete su ruido en la senal con
    la que se detecta el fondo. Medido en `GentleQuerulousElephantseal` con la separacion en 2D: la
    mediana salia **1,2-1,4 m**, cuando un floretista en guardia tiene los pies a 0,5-0,7 m y en un
    fondo completo a 1,2-1,5 m. O sea que la componente transversal estaba inflando la separacion
    hasta doblarla."""
    dpies_2d = np.linalg.norm(P1 - P2, axis=2)
    dist = np.abs(C[:, 0, 0] - C[:, 1, 0])
    """LA DISTANCIA ENTRE LOS DOS TIRADORES tambien se mide a lo largo de la pista y no en 2D, por la
    misma razon y porque es lo que significa en esgrima: la distancia de esgrima es a lo largo."""
    dist_2d = np.linalg.norm(C[:, 0] - C[:, 1], axis=1)
    ok = mask & np.isfinite(C[:, 0, 0]) & np.isfinite(C[:, 1, 0])
    cruces = int((C[ok, 0, 0] > C[ok, 1, 0]).sum())
    """`cruces` ES UN CONTROL, no un dato de trabajo: dos floretistas no pueden cruzarse, asi que
    cualquier valor distinto de cero hay que mirarlo antes de creerse nada de ese clip."""
    return {"C": C, "P1": P1, "P2": P2, "origen": org, "dpies": dpies, "dpies_2d": dpies_2d,
            "dist": dist, "dist_2d": dist_2d, "cruces": cruces,
            "n_con_los_dos": int(ok.sum()),
            "convenio_eje": "x crece hacia la derecha de la imagen; lado 0 = izquierda"}


def _lleva(M, p):
    """QUE HACE: pasa un punto de imagen al sistema del cuadro de referencia."""
    v = M @ np.array([p[0], p[1], 1.0])
    return v[:2] / v[2]


def controles_fisicos(ap, mask, ancho_m) -> dict:
    """QUE HACE: los CONTROLES FISICOS de la posicion, que son los que dicen si la trayectoria es
    creible sin necesitar ninguna verdad-terreno.

    LOS TRES, con su cifra esperada declarada antes de medir:

      recorrido TRANSVERSAL de un tirador a lo largo del clip. Un floretista se mueve a lo LARGO de
      la pista; a lo ancho apenas se sale de la banda central. Esperado: p95 menos p05 por debajo de
      **0,60 m**, o sea menos de la mitad del ancho de la pista.

      SEPARACION DE PIES a lo largo, mediana del clip. Esperado **0,50 a 0,90 m**: es la guardia de
      un floretista, que es donde pasa la mayor parte del tiempo.

      cuantos apoyos caen FUERA de la pista, o sea con la coordenada transversal fuera de
      [-0,15, ancho+0,15]. Esperado por debajo del **5 %**: un tirador que pisa fuera de la pista
      recibe un aviso del arbitro, asi que no puede ser lo normal.

    Un clip que falla estos controles tiene la homografia o la deteccion del pie mal, y la
    trayectoria no se puede publicar como metros aunque el control de estatura haya pasado.
    """
    C, dpies = ap["C"], ap["dpies"]
    ks = np.nonzero(mask)[0]
    out = {}
    rec, sep = [], []
    for lado in (0, 1):
        y = C[ks, lado, 1]
        y = y[np.isfinite(y)]
        if y.size > 10:
            r = float(np.percentile(y, 95) - np.percentile(y, 5))
            out["transv_rango_lado%d_m" % lado] = r
            rec.append(r)
        d = dpies[ks, lado]
        d = d[np.isfinite(d)]
        if d.size > 10:
            v = float(np.median(d))
            out["dpies_p50_lado%d_m" % lado] = v
            sep.append(v)
    y = C[ks][:, :, 1]
    y = y[np.isfinite(y)]
    if y.size:
        out["frac_fuera_pista"] = float(((y < -0.15) | (y > ancho_m + 0.15)).mean())
    if rec:
        out["transv_rango_m"] = float(np.max(rec))
    if sep:
        out["dpies_p50_m"] = float(np.median(sep))
    out["pasa_transv"] = bool(out.get("transv_rango_m", 9.9) <= 0.60)
    out["pasa_fuera"] = bool(out.get("frac_fuera_pista", 1.0) <= 0.05)
    """NO EVALUABLE NO ES LO MISMO QUE FALLAR, y confundirlos era un fallo mio de contabilidad: sin
    pose de 308 puntos no hay dos pies, asi que la separacion de pies no se puede medir, y la
    primera version lo contaba como control FALLADO. Resultado: los 10 clips publicables salian
    "FISICA NO" y 6 de ellos era solo porque no tenian pose. Un control que no se ha podido medir se
    informa como `None`, no como suspenso."""
    out["pasa_dpies"] = None if "dpies_p50_m" not in out else         bool(0.50 <= out["dpies_p50_m"] <= 0.90)
    out["pasa_fisica"] = bool(out["pasa_transv"] and out["pasa_fuera"]
                              and out["pasa_dpies"] is not False)
    out["fisica_completa"] = bool(out["pasa_dpies"] is not None)
    return out


def ruido(C, mask, fps) -> dict:
    """QUE HACE: MIDE el ruido de la posicion, que es lo que dice si esto sirve para medir un fondo.

    LA CIFRA CON LA QUE SE COMPARA, declarada antes: un fondo son 1,0-1,5 m en 0,2-0,3 s, o sea
    4-6 m/s, que a 30 fps son 0,13-0,20 m por cuadro. Asi que si el p95 del salto por cuadro de un
    tirador se acerca a 0,20 m, el ruido tapa la senal y no se puede medir un fondo.

    Se dan tambien los saltos por encima de `SALTO_FISICO_M`, que son imposibles y por tanto son
    ruido puro, y su fraccion.
    """
    out = {}
    todos = []
    for lado in (0, 1):
        d = []
        prev = None
        for k in np.nonzero(mask)[0]:
            p = C[k, lado]
            if not np.isfinite(p).all():
                prev = None
                continue
            if prev is not None and k - prev[0] == 1:
                d.append(float(np.linalg.norm(p - prev[1])))
            prev = (k, p)
        d = np.asarray(d)
        todos.append(d)
        if d.size:
            out["salto_p50_lado%d" % lado] = float(np.median(d))
            out["salto_p95_lado%d" % lado] = float(np.percentile(d, 95))
    t = np.concatenate([d for d in todos if d.size]) if any(d.size for d in todos) \
        else np.array([])
    if t.size:
        """`n` ES EL NUMERO DE SALTOS SOBRE LOS DOS TIRADORES, NO CUADROS, y hay que decirlo porque
        ya se ha usado mal: al sacar la tasa de pasos por segundo dividi por `n / fps` y eso da el
        DOBLE de la duracion, porque `t` concatena las dos series. La tasa salia la mitad de lo que
        es (0,49 pasos por segundo y tirador en vez de 0,70). La duracion de verdad va en
        `s_utiles`, que se calcula con el TRAMO de cuadros y no contando muestras."""
        out.update({"n": int(t.size), "salto_p50_m": float(np.median(t)),
                    "salto_p95_m": float(np.percentile(t, 95)),
                    "salto_p99_m": float(np.percentile(t, 99)),
                    "salto_max_m": float(t.max()),
                    "frac_imposible": float((t > SALTO_FISICO_M).mean()),
                    "v_p95_ms": float(np.percentile(t, 95) * fps)})
    """EL TEMBLOR, separado del movimiento de verdad. El salto entre cuadros mezcla dos cosas: el
    tirador moviendose, que es senal, y el error de medida, que es ruido. Compararlo con los
    0,13-0,20 m por cuadro de un fondo no dice si el ruido es aceptable, porque el fondo ESTA en la
    muestra. El temblor se mide como la distancia de cada punto a la trayectoria suavizada con una
    mediana movil de 5 cuadros: un movimiento real sobrevive al filtro y el ruido no."""
    tem = {0: [], 1: []}
    for lado in (0, 1):
        for j in (0, 1):
            x = C[:, lado, j]
            xs = _suave(x, 5)
            d = np.abs(x - xs)
            d = d[np.isfinite(d) & mask]
            tem[lado].append(d)
    tx = np.concatenate([tem[l][0] for l in (0, 1) if len(tem[l][0])]) if True else np.array([])
    ty = np.concatenate([tem[l][1] for l in (0, 1) if len(tem[l][1])])
    if tx.size:
        out["temblor_largo_p50_m"] = float(np.median(tx))
        out["temblor_largo_p95_m"] = float(np.percentile(tx, 95))
    if ty.size:
        out["temblor_ancho_p50_m"] = float(np.median(ty))
        out["temblor_ancho_p95_m"] = float(np.percentile(ty, 95))
    return out


def _suave(x, n=5):
    """QUE HACE: mediana movil, que quita los picos de un cuadro sin desplazar los flancos como
    haria una media."""
    y = x.copy()
    m = np.isfinite(x)
    idx = np.nonzero(m)[0]
    for i in idx:
        a, b = max(0, i - n // 2), min(len(x), i + n // 2 + 1)
        v = x[a:b][np.isfinite(x[a:b])]
        if v.size:
            y[i] = np.median(v)
    return y


def eventos(ap, r, fps, poses=None, paso=4, eje_brazo="hombro", res=None) -> list:
    """QUE HACE: la lista de eventos, cada uno con su cuadro, su posicion en metros y su magnitud en
    metros. Ver la cabecera del modulo para el criterio de cada tipo.

    `paso` y `eje_brazo` estan para poder sacar LA LINEA BASE GRATIS: `paso=1` es el criterio de paso
    de la seccion 23.8 y `eje_brazo="codo"` es el eje de brazo de entonces, asi que las dos versiones
    se pueden correr sobre los mismos datos y dar las dos cifras juntas. No son opciones para elegir
    a gusto: la version que se publica es la de por defecto.
    """
    mask = r["mask"]
    T = r["T"]
    ks = np.nonzero(mask)[0]
    if len(ks) < 12:
        return []
    ev = []
    C, P1, P2, dpies = ap["C"], ap["P1"], ap["P2"], ap["dpies"]
    """LA GUARDIA DEL CLIP es la referencia libre de escala con la que se normalizan los umbrales del
    fondo y el rasgo del pico. Se toma por TIRADOR, porque los dos pueden tener la caja medida con
    distinta calidad; si un tirador no tiene separacion de pies medible se cae a la nominal y queda
    dicho en la columna `guardia_m` del CSV."""
    guardia = {}
    for lado in (0, 1):
        v = dpies[mask, lado]
        v = v[np.isfinite(v)]
        guardia[lado] = float(np.median(v)) if v.size > 20 else GUARDIA_NOMINAL_M
        """NO SE RECORTA AL RANGO FISICO 0,50-0,90 m, y es deliberado: normalizar por la guardia
        MEDIDA es lo que cancela un error de ESCALA, que es el defecto que se esta arreglando. Si en
        vez de eso la guardia sale mal por un error de POSE, normalizar por ella empeora, y las dos
        causas no se distinguen con lo que hay. Asi que se usa la medida y se marca: la columna
        `guardia_plausible` del CSV dice si cae en 0,50-0,90 m, que es la guardia de un floretista, y
        quien no se la crea puede filtrar."""
    for lado in (0, 1):
        rival = 1 - lado
        sx = _suave(C[:, lado, 0])
        rx = _suave(C[:, rival, 0])
        haciarival = np.sign(np.nanmedian(rx) - np.nanmedian(sx))
        if not np.isfinite(haciarival) or haciarival == 0:
            haciarival = 1.0
        dp = _suave(dpies[:, lado])
        """EL AVANCE SE MIDE CON EL PIE DE DELANTE, NO CON EL CENTRO DEL CUERPO, y esto era un fallo
        mio CONTRA MI PROPIO CRITERIO DECLARADO: la cabecera del modulo dice "el pie de delante
        avanza al menos 0,25 m" y el codigo estaba midiendo el avance del centro de los dos pies,
        que en un fondo avanza aproximadamente la MITAD que el pie de delante. Consecuencia medida:
        el fondo completo del cuadro 239 de `GentleQuerulousElephantseal`, con los pies a 1,48 m y
        verificado a ojo en la imagen, no se detectaba. El criterio no estaba mal: estaba mal
        implementado, y solo se vio sacando el recorte del falso negativo."""
        pd = np.nanmax(np.stack([_suave(P1[:, lado, 0]) * haciarival,
                                 _suave(P2[:, lado, 0]) * haciarival]), axis=0) * haciarival
        ev += _fondos(sx, dp, pd, P1, P2, lado, haciarival, ks, C, fps,
                      guardia=guardia[lado])
        ev += _pasos(P1, P2, C, lado, haciarival, ks, fps, version=paso)
        ev += _altos(C, lado, ks, fps)
    ev += _acciones(C, ks, fps)
    if poses is not None:
        ev += _brazos(poses, r, C, ks, eje=eje_brazo)
        ev += _botes(poses, r, C, ks, res, fps)
    """EL ARBITRAJE, y despues el ASERTO de exclusividad. Ver `arbitra` para como decide."""
    fuera = []
    for lado in (0, 1):
        rival = 1 - lado
        hv = np.sign(np.nanmedian(C[:, rival, 0]) - np.nanmedian(C[:, lado, 0]))
        hv = float(hv) if np.isfinite(hv) and hv != 0 else 1.0
        fuera += arbitra(ev, ap, lado, guardia[lado], hv)
    if fuera:
        quitar = {id(e) for e in fuera}
        ev = [e for e in ev if id(e) not in quitar]
    for lado in (0, 1):
        mv = sorted([e for e in ev if e["tipo"] in ("marcha", "romper", "fondo")
                     and e.get("lado") == lado], key=lambda z: z["inicio"])
        for a in range(len(mv)):
            for b in range(a + 1, len(mv)):
                if mv[b]["inicio"] > mv[a]["fin"]:
                    break
                assert mv[a]["tipo"] == mv[b]["tipo"], (
                    "solape sin arbitrar: %s c%d-%d contra %s c%d-%d en el lado %d"
                    % (mv[a]["tipo"], mv[a]["inicio"], mv[a]["fin"], mv[b]["tipo"],
                       mv[b]["inicio"], mv[b]["fin"], lado))
    _marca_solape(ev)
    """ASERTO DEL CONTROL BALISTICO DEL BOTE. Es el mismo patron que arreglo los agregados y funciono:
    si un tipo esta declarado no establecido, que no pueda colarse una fila suya sin que el codigo
    aborte. Un evento que dice 8,5 cm de despegue y dura un cuadro es 8 veces mas corto de lo que la
    gravedad permite, y sin aserto solo lo caza que alguien lea el CSV."""
    for e in ev:
        if e["tipo"] != "bote":
            continue
        m = e.get("magnitud_m", float("nan"))
        if not (np.isfinite(m) and m > 0):
            continue
        dur = (e["fin"] - e["inicio"] + 1) / float(fps)
        tv = 2.0 * np.sqrt(2.0 * m / BOTE_G)
        assert dur >= BOTE_VUELO_FRAC * tv, (
            "bote imposible: %.3f m de despegue exigen %.3f s de vuelo y el evento dura %.3f s"
            % (m, tv, dur))
    ev.sort(key=lambda e: (e["cuadro"], e.get("lado", -1)))
    return ev


CONTACTO_MIN = 3
VUELO_MIN = 2


def _otsu(v, bins=40):
    """QUE HACE: el umbral de Otsu de una serie y su `eta` (varianza entre clases / total).

    SE USA OTSU Y NO UN MULTIPLO DEL MODO porque **no tiene parametro libre**. Probe la alternativa
    (`K` veces el modo de apoyo): con `K` = 6 la fraccion de apoyo cae en el rango fisico declarado en
    27 de 32 series contra 23 de 32 con Otsu, pero **elegir `K` mirando cual cumple mi propio criterio
    de aceptacion es ajustar el umbral al test**. Otsu sale de los datos. La diferencia entre los dos
    mueve la FRONTERA del evento un par de cuadros, que es justo lo que la anotacion del usuario mide.
    """
    v = v[np.isfinite(v)]
    if v.size < 100:
        return None
    hi = float(np.percentile(v, 99.5))
    if hi <= 0:
        return None
    h, borde = np.histogram(np.clip(v, 0.0, hi), bins=bins, range=(0.0, hi))
    p = h / h.sum()
    cen = 0.5 * (borde[:-1] + borde[1:])
    mu = float((p * cen).sum())
    vt = float((p * (cen - mu) ** 2).sum())
    mejor = (-1.0, None)
    for i in range(1, bins):
        w0, w1 = p[:i].sum(), p[i:].sum()
        if w0 < 1e-6 or w1 < 1e-6:
            continue
        m0 = float((p[:i] * cen[:i]).sum() / w0)
        m1 = float((p[i:] * cen[i:]).sum() / w1)
        sb = w0 * w1 * (m0 - m1) ** 2
        if sb > mejor[0]:
            mejor = (sb, i)
    if mejor[1] is None or vt <= 0:
        return None
    return {"umbral": float(borde[mejor[1]]), "eta": mejor[0] / vt}


def _racha_min(b, n):
    """QUE HACE: pone a False las rachas de `True` mas cortas que `n` cuadros. Quita los parpadeos,
    que son ruido de keypoint y no estados: un pie no toca el suelo 33 ms y vuelve a volar."""
    out = b.copy()
    i = 0
    while i < len(b):
        if not b[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(b) and b[j + 1]:
            j += 1
        if j - i + 1 < n:
            out[i:j + 1] = False
        i = j + 1
    return out


def contactos(res, poses, lado) -> dict:
    """QUE HACE: clasifica cada pie, cuadro a cuadro, en APOYO o VUELO, y da los instantes de
    DESPEGUE y ATERRIZAJE. Es la primitiva nueva.

    =======================================================================================
    POR QUE SE CAMBIA DE PRIMITIVA: UN UMBRAL SE CRUZA VARIAS VECES DENTRO DEL MISMO GESTO
    =======================================================================================

    El usuario: *"para que no detecte varias cosas en medio de la misma accion"*. El diagnostico es
    estructural y no de calibracion: todos los criterios anteriores son de la forma "la senial pasa de
    X", y una senial con ruido cruza un umbral **varias veces** dentro de un gesto, asi que cada cruce
    genera un evento. De ahi la fragmentacion, y de ahi que hubiera que anadir arbitraje, fusion y
    minimos de duracion: eran parches sobre la primitiva equivocada.

    La primitiva buena es el CONTACTO: un pie apoyado tiene velocidad ~0 en el sistema de la pista y
    en vuelo no. Los eventos son los CAMBIOS DE ESTADO, que son **instantes y no tramos**, asi que no
    se pueden solapar ni fragmentar por construccion.

    =======================================================================================
    LA PUERTA, MEDIDA ANTES DE CONSTRUIR NADA: las dos poblaciones existen
    =======================================================================================

    Senial: modulo de la velocidad del punto mas bajo de cada pie en el cuadro ESTABILIZADO,
    normalizada por el alto de la caja. Es de imagen: no necesita homografia (asi que vale tambien en
    los 7 clips que no la tienen) ni escala metrica.

    Medido sobre **32 series pie-tirador** de 8 clips, con los tres criterios declarados antes:

      1. `eta` de Otsu >= 0,60          -> **32 de 32** (p50 **0,781**, minimo 0,668)
      2. valle <= 0,50 del menor pico   -> **31 de 32** (p50 **0,06**: un valle profundo)
      3. fraccion de apoyo en 0,55-0,90 -> 23 de 32 (p50 0,87)
      **razon entre las medias de las dos poblaciones: 11,3 x**

    Los dos primeros son de BIMODALIDAD y no dependen del umbral: pasan casi en todas. El tercero es
    una propiedad del CLASIFICADOR, no de la bimodalidad, asi que funciona como calibracion y no como
    prueba; lo declare como prueba y hay que decir que no lo es.

    Con 11,3x de separacion la primitiva se sostiene. **Lo que queda sin fijar es la FRONTERA exacta**,
    que se mueve un par de cuadros segun el umbral. Esa es la incertidumbre que la anotacion del
    usuario va a medir, y por eso el convenio de frontera se declara antes de que anote.
    """
    r, Mk = res["_r"], res["_Mk"]
    T = min(poses["KP"].shape[0], r["T"])
    H = r["H"]
    alto = (r["B"][:T, lado, 3] - r["B"][:T, lado, 1]) * H
    a = float(np.nanmedian(alto[np.isfinite(alto)])) if np.isfinite(alto).any() else 0.0
    if a < 20.0:
        return {"ok": False, "motivo": "caja demasiado pequena o ausente"}
    P = np.full((T, 2, 2), np.nan)
    for k in np.nonzero(r["mask"][:T])[0]:
        if not r["pres"][k, lado]:
            continue
        for j, p in enumerate(_pie_pose(poses["KP"], poses["SC"], k, lado)):
            if p is None:
                continue
            q = Mk[k] @ np.array([p[0], p[1], 1.0])
            P[k, j] = q[:2] / q[2]
    apoyo = np.zeros((T, 2), bool)
    diag = {}
    for j in (0, 1):
        x = _suave(P[:, j, 0], 3)
        y = _suave(P[:, j, 1], 3)
        v = np.full(T, np.nan)
        for k in range(1, T):
            if np.isfinite(x[k]) and np.isfinite(x[k - 1]):
                v[k] = float(np.hypot(x[k] - x[k - 1], y[k] - y[k - 1])) / a
        o = _otsu(v)
        if o is None:
            return {"ok": False, "motivo": "menos de 100 cuadros con velocidad de pie"}
        ap = np.isfinite(v) & (v <= o["umbral"])
        ap = _racha_min(ap, CONTACTO_MIN)
        ap = ~_racha_min(~ap, VUELO_MIN)
        apoyo[:, j] = ap
        diag["umbral_pie%d" % j] = round(o["umbral"], 5)
        diag["eta_pie%d" % j] = round(o["eta"], 3)
        m = r["mask"][:T]
        diag["frac_apoyo_pie%d" % j] = round(float(ap[m].mean()), 3) if m.any() else None
    des, ate = [], []
    for j in (0, 1):
        for k in range(1, T):
            if apoyo[k - 1, j] and not apoyo[k, j]:
                des.append((int(k), j))
            elif not apoyo[k - 1, j] and apoyo[k, j]:
                ate.append((int(k), j))
    return {"ok": True, "apoyo": apoyo, "P": P, "alto_px": a, "despegues": sorted(des),
            "aterrizajes": sorted(ate), "T": T, **diag}


def eventos_sin_homografia(res, fps, poses=None) -> list:
    """QUE HACE: los eventos que se pueden medir SIN homografia: `accion` en altos de caja y `brazo`
    en fraccion de torso.

    Ver la cabecera de `ACCION_V_CAJA`. Devuelve la misma estructura que `_acciones` pero con
    `via_accion = "alto_de_caja"` y sin `pos_m`, porque no hay metros que dar y poner un cero seria
    inventar una posicion.

    `brazo` ENTRA AQUI Y NO NECESITABA HOMOGRAFIA NUNCA: se mide como el largo del eje
    `hombro -> raiz de la mano` en fraccion del largo del TORSO del propio tirador, que es una
    reglilla de imagen. Solo usaba la homografia para escribir `pos_m`, que ahora sale vacio.

    Y ESTO IMPORTA PARA UNA PRUEBA CONCRETA: el criterio congelado por otro agente para cazar al
    arbitro es `fondo + brazo` por segundo util, y de los dos clips con arbitro entregado como
    tirador **ninguno tiene homografia**. Sin esto, los dos lados darian cero y el cero seria
    indistinguible de "no medido", que es exactamente el fallo que hay que evitar. Con `brazo` medido
    por pose, la mitad del criterio SI se ejecuta y el cero significa algo.

    LO QUE SIGUE SIN PODERSE MEDIR: `marcha`, `romper`, `fondo` y `alto`. Sus umbrales estan en
    metros y no se pueden trasladar a altos de caja sin suponer que la relacion pixel-metro es la
    misma en todo el cuadro, y no lo es: la pista se aleja de la camara. Asi que en un clip sin
    homografia **la mitad de `fondo + brazo` es 0 POR CONSTRUCCION**, y quien use ese criterio tiene
    que leerlo como "solo la parte del brazo".
    """
    r, Mk = res.get("_r"), res.get("_Mk")
    if r is None or Mk is None:
        return []
    B, pres, mask, W, H, T = r["B"], r["pres"], r["mask"], r["W"], r["H"], r["T"]
    ks = np.nonzero(mask)[0]
    if len(ks) < 12:
        return []
    alto = (B[:, :, 3] - B[:, :, 1])[pres]
    alto = float(np.nanmedian(alto)) * H if alto.size else 0.0
    if not np.isfinite(alto) or alto < 20.0:
        return []
    X = np.full((T, 2), np.nan)
    for k in ks:
        for lado in (0, 1):
            if not pres[k, lado] or not np.isfinite(B[k, lado]).all():
                continue
            p = Mk[k] @ np.array([0.5 * (B[k, lado, 0] + B[k, lado, 2]) * W,
                                  B[k, lado, 3] * H, 1.0])
            X[k, lado] = (p[0] / p[2]) / alto
    v = np.nanmax(np.stack([_vel1(X[:, 0], fps), _vel1(X[:, 1], fps)]), axis=0)
    v = _suave(v, 3)
    out = []
    i = ks[0]
    while i <= ks[-1]:
        if not (np.isfinite(v[i]) and v[i] > ACCION_V_CAJA):
            i += 1
            continue
        ini, quieto, j = i, 0, i
        while j <= ks[-1] and quieto < ACCION_QUIETO:
            j += 1
            if j > ks[-1]:
                break
            quieto = quieto + 1 if (np.isfinite(v[j]) and v[j] <= ACCION_V_CAJA) else 0
        fin = min(j, ks[-1])
        if fin - ini >= 4:
            d = [abs(X[fin, l] - X[ini, l]) for l in (0, 1)]
            out.append({"tipo": "accion", "lado": -1, "cuadro": int(ini), "inicio": int(ini),
                        "fin": int(fin), "pos_m": [None, None],
                        "magnitud_m": float("nan"),
                        "magnitud_altos_caja": round(float(np.nanmax(d)), 3)
                        if np.isfinite(d).any() else None,
                        "unidad_magnitud": "altos de caja del tirador",
                        "via_accion": "alto_de_caja",
                        "dur_s": float((fin - ini) / fps)})
        i = fin + 1
    if poses is not None:
        Cv = np.full((T, 2, 2), np.nan)
        out += _brazos(poses, r, Cv, ks, eje="hombro")
    out.sort(key=lambda e: (e["cuadro"], e.get("lado", -1)))
    return out


def _vel1(x, fps):
    """QUE HACE: modulo de la velocidad de una serie de UNA coordenada, en unidades de esa
    coordenada por segundo."""
    v = np.full(len(x), np.nan)
    for i in range(1, len(x)):
        if np.isfinite(x[i]) and np.isfinite(x[i - 1]):
            v[i] = abs(float(x[i] - x[i - 1])) * fps
    return v


def _rasgos_gesto(ap, i, j, lado, hacia, guardia):
    """QUE HACE: los dos rasgos que separan marcha de fondo, medidos en la FASE DE EXTENSION.

    Ver la cabecera de `ARB_RAZON_ATRAS` para por que se miden hasta el PICO de la separacion y no
    hasta el final de la ventana.
    """
    dps = _suave(ap["dpies"][:, lado], 5)
    seg = dps[i:j + 1]
    if not np.isfinite(seg).any():
        return None
    kp = i + int(np.nanargmax(seg))
    pico = float(dps[kp]) / guardia if guardia > 1e-6 else float("nan")
    x1 = _suave(ap["P1"][:, lado, 0], 5)
    x2 = _suave(ap["P2"][:, lado, 0], 5)
    d = [(x1[kp] - x1[i]) * hacia, (x2[kp] - x2[i]) * hacia]
    if not np.isfinite(d).all():
        return None
    de = int(np.argmax(np.abs(d)))
    dl, dt = abs(d[de]), abs(d[1 - de])
    raz = dt / dl if dl > 1e-6 else float("nan")
    return {"razon_pie_atras": float(raz), "pico_dpies_guardia": float(pico),
            "cuadro_pico_dpies": int(kp)}


def arbitra(ev, ap, lado, guardia, hacia) -> list:
    """QUE HACE: cuando dos eventos de movimiento de TIPOS DISTINTOS del mismo tirador se solapan en
    el tiempo, DECIDE uno con el test del pie de atras y descarta el otro.

    POR QUE HACE FALTA: marcarlos con `solapa_fondo` no bastaba. El usuario veia en el video
    `marcha -> fondo -> marcha` sobre el MISMO gesto, y eso no es informacion, es ambigüedad. Los dos
    candidatos son el mismo movimiento visto por dos criterios, asi que hay que elegir.

    COMO DECIDE, y lo que conserva: se agrupan los eventos por solape (componentes conexas en el
    tiempo), se miden los rasgos en la ventana UNION del grupo y se decide el tipo del grupo.

      si el grupo es FONDO  -> se emite el fondo de mayor magnitud y se descartan los pasos.
      si el grupo es PASO   -> se descartan los fondos y **se conservan TODOS los pasos**, que no se
                               solapan entre ellos porque `_pasos` no los produce solapados.

    Esa ultima regla es la que arregla el caso del usuario sin perder informacion: en
    `ForcefulAntiqueAlbacoretuna` lado 0 el grupo `marcha c25-37 + fondo c36-47 + marcha c39-52` se
    resuelve como paso, se cae el fondo y **quedan las dos marchas**. Si se emitiera un solo evento
    por grupo se perderia un paso de verdad.

    Cada evento que sobrevive lleva `arbitrado`, los dos rasgos y `margen_arbitraje`, que es a cuanto
    quedo del umbral el rasgo que decidio. Un evento reclasificado tiene que poder contradecirse.
    """
    mov = sorted([e for e in ev if e["tipo"] in ("marcha", "romper", "fondo")
                  and e.get("lado") == lado], key=lambda z: z["inicio"])
    if not mov:
        return []
    grupos, act = [], [mov[0]]
    for e in mov[1:]:
        if e["inicio"] <= max(x["fin"] for x in act):
            act.append(e)
            continue
        grupos.append(act)
        act = [e]
    grupos.append(act)
    fuera = []
    for g in grupos:
        tipos = {e["tipo"] for e in g}
        if len(tipos) < 2:
            continue
        i = min(e["inicio"] for e in g)
        j = max(e["fin"] for e in g)
        r = _rasgos_gesto(ap, i, j, lado, hacia, guardia)
        if r is None:
            fuera += [e for e in g if e["tipo"] == "fondo"]
            continue
        es_fondo = (np.isfinite(r["razon_pie_atras"])
                    and r["razon_pie_atras"] < ARB_RAZON_ATRAS
                    and np.isfinite(r["pico_dpies_guardia"])
                    and r["pico_dpies_guardia"] > ARB_PICO_GUARDIA)
        """EL MARGEN es la distancia al umbral del rasgo que decide. Con el fondo decide el que este
        MAS cerca de su frontera, porque es el que sostiene la decision."""
        m1 = ARB_RAZON_ATRAS - r["razon_pie_atras"]
        m2 = r["pico_dpies_guardia"] - ARB_PICO_GUARDIA
        margen = min(m1, m2) if es_fondo else max(-m1, -m2)
        if es_fondo and not any(e["tipo"] == "fondo" for e in g):
            es_fondo = False
            """UN GRUPO SIN FONDO NO PUEDE SALIR FONDO. Un grupo `marcha + romper` entra aqui porque
            tiene dos tipos, y si los rasgos dicen "fondo" no hay ningun fondo que emitir: era el
            `max() iterable argument is empty` que tumbaba la etapa 7 entera en 7 clips (medido el
            2026-10-08 en los registros de la noche). Se decide como paso."""
        gana = "fondo" if es_fondo else "paso"
        if es_fondo:
            fon = [e for e in g if e["tipo"] == "fondo"]
            mejor = max(fon, key=lambda e: e.get("magnitud_m", 0.0) or 0.0)
            """SOLO SE DESCARTAN LOS PASOS QUE SOLAPAN AL FONDO ELEGIDO, no todo el grupo: un grupo
            se forma por solape TRANSITIVO, asi que un paso del extremo de la cadena puede no solapar
            al fondo y no hay razon para perderlo.

            ES UNA SALVAGUARDA Y NO UN ARREGLO, y hay que decirlo asi porque yo lo escribi como
            arreglo: **medido, no cambia ningun recuento en los 11 clips** (marcha 88, romper 84,
            fondo 33 con las dos versiones). La caida de marchas de `AliveIgnorantBarebirdbat` de 11
            a 4, que es la que me hizo escribir esto, **no viene de aqui**: viene de la normalizacion
            del umbral. La guardia de ese clip es 0,58 m, asi que el umbral del fondo BAJA a
            0,50 x 0,58 = 0,29 m y mas gestos califican como fondo. La normalizacion es simetrica:
            aprieta donde la escala esta inflada y afloja donde esta comprimida, que es lo correcto
            **si** la guardia refleja la escala y no un error de pose (ver `guardia_plausible`)."""
            vivos = [mejor] + [e for e in g if e["tipo"] != "fondo"
                               and (e["fin"] < mejor["inicio"] or e["inicio"] > mejor["fin"])]
            fuera += [e for e in g if e not in vivos]
        else:
            fuera += [e for e in g if e["tipo"] == "fondo"]
            vivos = [e for e in g if e["tipo"] != "fondo"]
        sentido = _sentido_unico(vivos)
        fuera += sentido
        vivos = [e for e in vivos if e not in sentido]
        for e in vivos:
            e["arbitrado"] = gana
            e["razon_pie_atras"] = round(r["razon_pie_atras"], 3)
            e["pico_dpies_guardia"] = round(r["pico_dpies_guardia"], 3)
            e["margen_arbitraje"] = round(float(margen), 3)
    return fuera


def _mag(e):
    for k in ("magnitud_m", "magnitud_altos_caja"):
        v = e.get(k)
        if v is not None and np.isfinite(v):
            return abs(float(v))
    return 0.0


def _sentido_unico(vivos) -> list:
    """QUE DEVUELVE: los pasos que hay que descartar porque una MARCHA y un ROMPER del mismo tirador
    se solapan en el tiempo.

    POR QUE: `arbitra` daba por hecho que `_pasos` nunca produce pasos solapados, y no es cierto
    entre sentidos: medido el 2026-10-08, **20 clips** de la noche caian en el aserto de
    exclusividad con `romper c2-19 contra marcha c13-15` y similares, y la etapa 7 no escribia
    NADA para ellos. Un pie no avanza y retrocede a la vez: es un solo gesto visto por los dos
    detectores de sentido, y se queda el de MAYOR desplazamiento, que es el que mueve al tirador.
    El descartado no se borra en silencio: el superviviente lleva `arbitrado = sentido`."""
    v = sorted(vivos, key=lambda z: z["inicio"])
    fuera = []
    cambio = True
    while cambio:
        cambio = False
        vv = [e for e in v if e not in fuera]
        for a in range(len(vv)):
            for b in range(a + 1, len(vv)):
                ea, eb = vv[a], vv[b]
                if eb["inicio"] > ea["fin"]:
                    break
                if ea["tipo"] != eb["tipo"]:
                    pierde = ea if _mag(ea) < _mag(eb) else eb
                    gana = eb if pierde is ea else ea
                    gana["arbitrado_sentido"] = True
                    fuera.append(pierde)
                    cambio = True
                    break
            if cambio:
                break
    return fuera


def _marca_solape(ev):
    """QUE HACE: marca los pasos que caen ENCIMA de un fondo del mismo tirador.

    POR QUE HACE FALTA, y lo encontre MIRANDO LOS RECORTES, no en ninguna cifra agregada: en
    `AliveIgnorantBarebirdbat` el evento `marcha l0 c49-55, 0,23 m, un pie` es el MISMO gesto que el
    `fondo l0 c46-57, 0,59 m`. Y no es un fallo de implementacion: es que el criterio del paso ("un
    pie se mueve rapido y el otro esta quieto") **describe exactamente la primera mitad de un fondo**.
    Un fondo es el pie de delante saliendo disparado con el de detras clavado en el suelo.

    No se puede arreglar bajando ni subiendo un umbral, porque las dos cosas son el mismo suceso
    medido con la misma senal. Lo que si se puede hacer es DECIRLO: cada paso lleva `solapa_fondo`, y
    quien quiera contar pasos de desplazamiento filtra `solapa_fondo == False`. La fraccion que
    solapa va en el informe, que es la cifra que dice cuanto contamina.
    """
    fondos = {}
    for e in ev:
        if e["tipo"] == "fondo":
            fondos.setdefault(e.get("lado", -1), []).append((e["inicio"], e["fin"]))
    for e in ev:
        if e["tipo"] not in ("marcha", "romper"):
            continue
        e["solapa_fondo"] = any(not (e["fin"] < a or e["inicio"] > b)
                                for a, b in fondos.get(e.get("lado", -1), []))


def umbral_quieto(C, lado, mask) -> tuple:
    """QUE HACE: el umbral de "esta quieto" de el alto CALIBRADO con el ruido de este clip y este
    tirador, y la cifra de ruido con la que se ha calibrado.

    Devuelve `(umbral_m, ruido_p95_m, medible)`. Ver la cabecera de `ALTO_DESPUES_M` para por que
    es adaptativo y para los dos criterios fijos que se descartaron antes de llegar aqui.

    EL RUIDO SE MIDE SOBRE EL RESIDUO `x - suavizada`, que no contiene movimiento del tirador por
    construccion. Medirlo sobre la senal daria el movimiento real y no el ruido, que es la trampa
    clasica de este tipo de umbral.
    """
    V = ALTO_VENTANA
    x = C[:, lado, 0]
    rs = x - _suave(x, 5)
    d = np.full(len(x), np.nan)
    for k in range(V, len(x)):
        if np.isfinite(rs[k]) and np.isfinite(rs[k - V]):
            d[k] = abs(rs[k] - rs[k - V])
    d = d[np.isfinite(d) & mask]
    if d.size < 20:
        return ALTO_DESPUES_M, float("nan"), False
    rp = float(np.percentile(d, 95))
    u = max(ALTO_DESPUES_M, ALTO_MARGEN * rp)
    return u, rp, bool(u <= ALTO_TOPE_M)


def _altos(C, lado, ks, fps):
    """CRITERIO DE EL ALTO EN SECO, declarado: el tirador recorre `ALTO_ANTES_M` en
    `ALTO_VENTANA` cuadros y en los `ALTO_VENTANA` siguientes recorre menos que el umbral de
    quieto, que se calibra con el ruido del propio tirador.
    """
    V = ALTO_VENTANA
    mask = np.zeros(len(C), bool)
    mask[ks] = True
    umb, rp, medible = umbral_quieto(C, lado, mask)
    if not medible:
        return []
    xs = _suave(C[:, lado, 0], 5)
    out = []
    k = ks[0] + V
    while k + V <= ks[-1]:
        a = xs[k] - xs[k - V]
        b = xs[k + V] - xs[k]
        if np.isfinite(a) and np.isfinite(b) and abs(a) >= ALTO_ANTES_M and abs(b) <= umb:
            out.append({"tipo": "alto", "lado": lado, "cuadro": int(k),
                        "inicio": int(k - V), "fin": int(k + V),
                        "pos_m": _pos(C, int(k), lado),
                        "magnitud_m": float(abs(a)),
                        "v_ms": float(abs(a) * fps / V),
                        "umbral_quieto_m": round(umb, 4),
                        "ruido_p95_m": round(rp, 4)})
            k += V
            continue
        k += 1
    return out


def levanta_pies(poses, r):
    """QUE HACE: cuanto se ha levantado del suelo el pie que MENOS se ha levantado, en fraccion del
    alto de la caja del propio tirador. Devuelve (T, 2).

    ES LA SENAL DEL BOTE, y se mide en la IMAGEN a proposito: ver la cabecera de `BOTE_FRAC`. El
    nivel del suelo de cada pie se toma como la MEDIANA de su fila mas baja en una ventana de
    +-`BOTE_BASE_VENTANA` cuadros, que es lo que absorbe el paneo de camara (p50 161 px en este
    corpus) y el cambio de perspectiva al desplazarse el tirador. Un valor fijo de suelo no sirve.
    """
    KP, SC, B = poses["KP"], poses["SC"], r["B"]
    H, W = r["H"], r["W"]
    T = min(KP.shape[0], r["T"], len(B))
    pl = np.full((T, 2, 2), np.nan)
    for k in range(T):
        for lado in (0, 1):
            if not r["pres"][k, lado]:
                continue
            for j, (tob, tri) in enumerate(((13, (15, 16, 17)), (14, (18, 19, 20)))):
                sc, kp = SC[k, lado], KP[k, lado]
                cand = [i for i in (tob,) + tri if sc[i] >= CONF_MIN and np.isfinite(kp[i]).all()]
                if cand:
                    pl[k, lado, j] = float(max(kp[i, 1] for i in cand))
    s = np.full((T, 2), np.nan)
    Vb = BOTE_BASE_VENTANA
    for lado in (0, 1):
        alto = (B[:T, lado, 3] - B[:T, lado, 1]) * H
        for k in range(T):
            if not np.isfinite(alto[k]) or alto[k] < 20.0:
                continue
            sub = []
            for j in (0, 1):
                if not np.isfinite(pl[k, lado, j]):
                    sub = []
                    break
                a, b = max(0, k - Vb), min(T, k + Vb + 1)
                v = pl[a:b, lado, j]
                v = v[np.isfinite(v)]
                if v.size < 5:
                    sub = []
                    break
                sub.append(float(np.median(v)) - float(pl[k, lado, j]))
            if len(sub) == 2:
                s[k, lado] = min(sub) / float(alto[k])
    return s


def umbral_bote(s, lado, mask) -> tuple:
    """QUE HACE: el umbral del bote calibrado con la cola NEGATIVA de la propia senal, que es ruido
    puro porque un pie no puede estar por debajo del suelo. Devuelve `(umbral, ruido, medible)`.

    Ver la cabecera de `BOTE_TOPE` para la medida y para por que este evento no queda establecido.
    """
    v = _suave(s[:, lado], 3)
    v = v[np.isfinite(v) & mask[:len(v)]]
    if v.size < 40:
        return BOTE_FRAC, float("nan"), False
    rp = abs(float(np.percentile(v, 5)))
    u = max(BOTE_FRAC, BOTE_MARGEN * rp)
    return u, rp, bool(u <= BOTE_TOPE)


def _botes(poses, r, C, ks, res=None, fps=30.0):
    """CRITERIO DEL BOTE, declarado: la senal de `levanta_pies` pasa del umbral calibrado con el
    ruido durante al menos `BOTE_MIN_CUADROS` cuadros. Las DOS plantas tienen que haberse levantado,
    que es lo que garantiza tomar el minimo de las dos.
    """
    s = levanta_pies(poses, r)
    T = len(s)
    h = {0: float("nan"), 1: float("nan")}
    if res is not None:
        for lado in (0, 1):
            v = res.get("h%d_p50" % lado)
            h[lado] = float(v) if v is not None else float("nan")
    mk = np.zeros(max(T, r["T"]), bool)
    mk[np.nonzero(r["mask"])[0]] = True
    out = []
    for lado in (0, 1):
        umb, rp, medible = umbral_bote(s, lado, mk)
        ss = _suave(s[:, lado], 3)
        alto = ss > umb
        i = ks[0]
        while i <= min(ks[-1], T - 1):
            if not alto[i]:
                i += 1
                continue
            j = i
            while j + 1 <= min(ks[-1], T - 1) and alto[j + 1]:
                j += 1
            if j - i + 1 >= BOTE_MIN_CUADROS:
                kp = int(i + np.nanargmax(ss[i:j + 1]))
                pico = float(ss[kp])
                mag = pico * h[lado] if np.isfinite(h[lado]) else float("nan")
                """CONTROL BALISTICO: el tiempo de vuelo que implica la altura medida. Ver la
                cabecera de `BOTE_VUELO_FRAC`: no es un umbral, es una igualdad de caida libre."""
                if np.isfinite(mag) and mag > 0:
                    t_vuelo = 2.0 * np.sqrt(2.0 * mag / BOTE_G)
                    if (j - i + 1) / float(fps) < BOTE_VUELO_FRAC * t_vuelo:
                        i = j + 1
                        continue
                out.append({"tipo": "bote", "lado": lado, "cuadro": kp, "inicio": int(i),
                            "fin": int(j), "pos_m": _pos(C, kp, lado),
                            "magnitud_m": float(mag),
                            "unidad_magnitud": "m verticales, escala de la caja del tirador",
                            "levanta_frac_caja": round(pico, 4),
                            "umbral_bote": round(umb, 4),
                            "sobre_ruido": round(pico / rp, 2) if np.isfinite(rp) and rp > 0
                            else None,
                            "tipo_establecido": bool(medible)})
            i = j + 1
    return out


def _fondos(sx, dp, pd, P1, P2, lado, haciarival, ks, C, fps, guardia=GUARDIA_NOMINAL_M):
    """CRITERIO DEL FONDO, declarado: la separacion de pies crece `FONDO_SALTO_M` en
    `FONDO_VENTANA` cuadros Y el cuerpo avanza `FONDO_AVANCE_M` hacia el rival en la misma ventana.
    """
    out = []
    """LOS UMBRALES SALEN DE LA GUARDIA DEL PROPIO TIRADOR y no de metros absolutos.
    Ver la cabecera de `FONDO_SALTO_FRAC`: con los metros absolutos, un clip con la
    escala inflada daba fondos con cualquier paso, y la correlacion entre el factor de
    escala y los fondos por segundo era 0,83."""
    salto = FONDO_SALTO_FRAC * guardia
    avance = FONDO_AVANCE_FRAC * guardia
    V = FONDO_VENTANA
    k0 = ks[0]
    k1 = ks[-1]
    k = k0
    while k + V <= k1:
        a, b = dp[k], dp[k + V]
        av = (pd[k + V] - pd[k]) * haciarival
        if np.isfinite(a) and np.isfinite(b) and np.isfinite(av) \
                and (b - a) >= salto and av >= avance:
            seg = dp[k:k + 2 * V + 1]
            j = int(np.nanargmax(seg)) if np.isfinite(seg).any() else V
            kp = k + j
            mag = float((pd[kp] - pd[k]) * haciarival) if np.isfinite(pd[kp]) else float("nan")
            out.append({"tipo": "fondo", "lado": lado, "cuadro": int(kp),
                        "inicio": int(k), "fin": int(kp),
                        "pos_m": _pos(C, kp, lado),
                        "magnitud_m": mag,
                        "sep_pies_m": float(dp[kp] - dp[k]),
                        "guardia_m": round(guardia, 3),
                        "v_ms": float(mag * fps / max(kp - k, 1))})
            k = kp + V
            continue
        k += 1
    return out


def _vx_serie(x, fps, w=2):
    """QUE HACE: velocidad longitudinal por diferencia centrada de +-`w` cuadros.

    NO ES `_vel`, y la diferencia es lo que arregla el detector de paso: `_vel` toma el modulo en 2D
    de la diferencia de UN cuadro sobre la posicion CRUDA. Eso mete dos cosas que no son movimiento
    del tirador: la coordenada transversal, que 23.7 midio que no es creible, y los saltos de un
    cuadro de `_pie_pose` cuando cambia cual de los cuatro puntos del pie es el mas bajo.
    """
    v = np.full(len(x), np.nan)
    for i in range(w, len(x) - w):
        if np.isfinite(x[i + w]) and np.isfinite(x[i - w]):
            v[i] = (x[i + w] - x[i - w]) / (2.0 * w) * fps
    return v


def _zancadas(v, otro, ks, umb):
    """QUE HACE: los tramos en que ESTE pie se mueve y el otro no. Devuelve (inicio, fin, sentido).

    Se entra con `|v| > umb` y con el otro pie por debajo de la mitad, y se sale con histeresis (la
    mitad del umbral) para no cortar una zancada en dos por un cuadro flojo.
    """
    out = []
    i = ks[0]
    while i <= ks[-1]:
        ok = np.isfinite(v[i]) and abs(v[i]) > umb and \
            (not np.isfinite(otro[i]) or abs(otro[i]) < 0.5 * umb)
        if not ok:
            i += 1
            continue
        j = i
        while j + 1 <= ks[-1]:
            sig = np.isfinite(v[j + 1]) and abs(v[j + 1]) > 0.5 * umb and \
                (not np.isfinite(otro[j + 1]) or abs(otro[j + 1]) < umb)
            if not sig:
                break
            j += 1
        if j - i + 1 >= PASO_MIN_CUADROS:
            kp = i + int(np.nanargmax(np.abs(v[i:j + 1])))
            out.append((int(i), int(j), int(np.sign(v[kp]))))
        i = j + 1
    return out


def _pasos(P1, P2, C, lado, haciarival, ks, fps, version=4):
    """CRITERIO DEL PASO, version 4, y las TRES versiones anteriores fallaron cada una su propia
    prueba. Se cuentan enteras porque la cuarta no se entiende sin ellas.

    LO QUE MANDA, declarado antes de correr: una marcha de florete mide **0,25-0,60 m** y dura
    **0,20-0,60 s**. Si la magnitud que sale no cae en ese rango, el detector no esta midiendo un
    paso, se llame como se llame.

      v1, la de la seccion 23.8. `|dP|` en 2D de un cuadro sobre la posicion CRUDA del pie, umbral
          0,90 m/s. Magnitud p50 medida: **0,12 m**. Sacando los tramos uno a uno se vio por que:
          duran 3-5 cuadros y los picos de velocidad del pie llegan a **12,3 m/s**, que es imposible
          (un pie de esgrimista no pasa de 4-6 m/s). No detectaba pasos: detectaba los saltos de un
          cuadro de `_pie_pose` al cambiar cual de los cuatro puntos del pie es el mas bajo, mas el
          ruido de la coordenada transversal, que entra por tomar el modulo en 2D.

      v2, v1 fusionando los tramos consecutivos del mismo sentido a <= 6 cuadros. Hipotesis: un paso
          son dos tramos, primero un pie y luego el otro. Resultado medido: **n 11 -> 11 y p50
          0,122 -> 0,122 m**, o sea NO CAMBIA NADA, porque los huecos entre tramos medidos son de 2 a
          266 cuadros y no de 6. La hipotesis era razonable y era falsa.

      v3, zancada = tramo MONOTONO de la longitudinal suavizada del pie con un cambio >= 0,15 m.
          Resultado: eventos de **4,4 s y 6,8 m**. El defecto es de bulto y es geometrico: la
          posicion de un pie a lo largo del clip es una ESCALERA, y una escalera es monotona
          completa, asi que un tramo monotono no separa un paso del siguiente. Un criterio de
          monotonia no puede segmentar una escalera.

      v4, LA QUE SE USA. Se vuelve al criterio de "un pie se mueve y el otro esta quieto", que era el
          correcto, pero medido sobre la velocidad LONGITUDINAL de la posicion SUAVIZADA
          (`_vx_serie` sobre `_suave(x, 5)`), que es lo que quita los saltos de keypoint y el ruido
          transversal. Y un PASO se arma emparejando la zancada de un pie con la siguiente zancada
          DEL OTRO pie en el mismo sentido a menos de `PASO_HUECO` cuadros, que es la definicion de
          esgrima de un paso.

          Umbral recalibrado sobre la senal nueva, no heredado: el ruido p95 de `_vx_serie` sobre el
          residuo de las 16 series pie-tirador de los 4 clips sale **0,418 m/s de mediana** (0,063 a
          2,263), asi que 3x son **1,25 m/s**, que es `PASO_V_MS`.

          Resultado medido en los 4 clips: **87 pasos, magnitud p50 0,397 m (p10 0,150, p90 1,076),
          duracion p50 0,37 s**. Las dos cifras caen en el rango declarado, asi que se adopta.

          Y el numero SUBE de 33 a 87, que es lo contrario de lo que yo habia predicho al declarar la
          v2. La razon es que v1 no estaba contando pasos de menos por fusionar mal: estaba contando
          artefactos y perdiendose pasos de verdad. La verificacion a ojo es la que lo decide, no
          esta cifra.

    `pies_movidos` = 1 o 2. UN PASO CON UN SOLO PIE ES MEDIO PASO, y va marcado en el CSV para que
    quien quiera pasos completos filtre por 2. No se descarta porque un floretista hace medios pasos
    y ajustes de distancia todo el rato, y llamarlos "no pasos" seria inventar.
    """
    if version == 1:
        return _pasos_v1(P1, P2, C, lado, haciarival, ks, fps, fusion=False)
    if version == 2:
        return _pasos_v1(P1, P2, C, lado, haciarival, ks, fps, fusion=True)
    x1 = _suave(P1[:, lado, 0], 5)
    x2 = _suave(P2[:, lado, 0], 5)
    v1 = _vx_serie(x1, fps)
    v2 = _vx_serie(x2, fps)
    z = [(i, j, s, 0) for (i, j, s) in _zancadas(v1, v2, ks, PASO_V_MS)] + \
        [(i, j, s, 1) for (i, j, s) in _zancadas(v2, v1, ks, PASO_V_MS)]
    z.sort()
    usado = [False] * len(z)
    cs = _suave(C[:, lado, 0], 5)
    out = []
    for a in range(len(z)):
        if usado[a]:
            continue
        i, j, s, pie = z[a]
        usado[a] = True
        fin, npies = j, 1
        for b in range(a + 1, len(z)):
            if usado[b]:
                continue
            i2, j2, s2, pie2 = z[b]
            if i2 - fin > PASO_HUECO:
                break
            if pie2 != pie and s2 == s:
                usado[b] = True
                fin, npies = max(fin, j2), 2
                break
        d = (cs[fin] - cs[i]) * haciarival
        if not np.isfinite(d) or abs(d) < PASO_MIN_M:
            continue
        out.append({"tipo": "marcha" if d > 0 else "romper", "lado": lado,
                    "cuadro": int((i + fin) // 2), "inicio": int(i), "fin": int(fin),
                    "pos_m": _pos(C, int((i + fin) // 2), lado),
                    "magnitud_m": float(abs(d)),
                    "pies_movidos": int(npies),
                    "v_ms": float(abs(d) * fps / max(fin - i, 1))})
    return out


def _pasos_v1(P1, P2, C, lado, haciarival, ks, fps, fusion=False):
    """LA LINEA BASE, que es el criterio de la seccion 23.8 tal cual, con y sin la fusion de la v2.
    Se conserva para poder dar las dos cifras juntas y no de memoria. Ver `_pasos` para por que no
    se usa."""
    out = []
    v1 = _vel(P1[:, lado], fps)
    v2 = _vel(P2[:, lado], fps)
    mov = np.zeros(len(v1), bool)
    for i in range(len(v1)):
        a, b = v1[i], v2[i]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        mov[i] = (a > 0.90 and b < 0.45) or (b > 0.90 and a < 0.45)
    tramos = []
    i = ks[0]
    while i <= ks[-1]:
        if not mov[i]:
            i += 1
            continue
        j = i
        while j + 1 <= ks[-1] and mov[j + 1]:
            j += 1
        if j - i + 1 >= PASO_MIN_CUADROS:
            d = (C[j, lado, 0] - C[i, lado, 0]) * haciarival
            if np.isfinite(d) and abs(d) > 0.05:
                tramos.append([int(i), int(j), float(d)])
        i = j + 1
    if fusion:
        fus = []
        for t in tramos:
            if fus and np.sign(t[2]) == np.sign(fus[-1][2]) and t[0] - fus[-1][1] <= PASO_FUSION:
                fus[-1][1] = t[1]
                d = (C[t[1], lado, 0] - C[fus[-1][0], lado, 0]) * haciarival
                fus[-1][2] = float(d) if np.isfinite(d) else fus[-1][2] + t[2]
                continue
            fus.append(list(t))
        tramos = fus
    for i, j, d in tramos:
        out.append({"tipo": "marcha" if d > 0 else "romper", "lado": lado,
                    "cuadro": int((i + j) // 2), "inicio": int(i), "fin": int(j),
                    "pos_m": _pos(C, int((i + j) // 2), lado),
                    "magnitud_m": float(abs(d)), "pies_movidos": 1,
                    "v_ms": float(abs(d) * fps / max(j - i, 1))})
    return out


def _vel(P, fps):
    """QUE HACE: modulo de la velocidad en m/s de una serie de puntos en metros."""
    v = np.full(len(P), np.nan)
    for i in range(1, len(P)):
        if np.isfinite(P[i]).all() and np.isfinite(P[i - 1]).all():
            v[i] = float(np.linalg.norm(P[i] - P[i - 1])) * fps
    return v


def _acciones(C, ks, fps):
    """CRITERIO DE INICIO Y FIN DE ACCION, declarado: empieza cuando alguno de los dos pasa de
    `ACCION_V_MS` y acaba cuando los dos llevan `ACCION_QUIETO` cuadros por debajo.
    """
    v = np.nanmax(np.stack([_vel(C[:, 0], fps), _vel(C[:, 1], fps)]), axis=0)
    v = _suave(v, 3)
    out = []
    i = ks[0]
    while i <= ks[-1]:
        if not (np.isfinite(v[i]) and v[i] > ACCION_V_MS):
            i += 1
            continue
        ini = i
        quieto = 0
        j = i
        while j <= ks[-1] and quieto < ACCION_QUIETO:
            j += 1
            if j > ks[-1]:
                break
            quieto = quieto + 1 if (np.isfinite(v[j]) and v[j] <= ACCION_V_MS) else 0
        fin = min(j, ks[-1])
        if fin - ini >= 4:
            d = [float(abs(C[fin, l, 0] - C[ini, l, 0])) for l in (0, 1)]
            out.append({"tipo": "accion", "lado": -1, "cuadro": int(ini), "inicio": int(ini),
                        "fin": int(fin), "pos_m": _pos(C, int(ini), 0),
                        "magnitud_m": float(np.nanmax(d)) if np.isfinite(d).any() else float("nan"),
                        "dur_s": float((fin - ini) / fps)})
        i = fin + 1
    return out


def _brazos(poses, r, C, ks, eje="hombro"):
    """CRITERIO DE EXTENSION DE BRAZO, declarado: el largo del eje `hombro -> raiz de la mano` en
    fraccion del largo del torso pasa de `BRAZO_UMBRAL` veces su mediana del clip y esta subiendo.

    EL ORIGEN DEL EJE ES EL HOMBRO Y NO EL CODO, y esto es una CORRECCION de la version de la
    seccion 23.8, que usaba el codo. La razon es que `codo -> raiz` es un HUESO, el antebrazo, y un
    hueso no se alarga: lo que cambia al extender el brazo es el angulo del codo, o sea la distancia
    del HOMBRO a la mano. La seccion 22.3 ya lo habia medido para el cono y el cambio no se habia
    propagado aqui.

    MEDIDO OTRA VEZ AQUI, sobre los 8 tiradores de los 4 clips publicables con pose, el coeficiente
    de variacion del largo del eje normalizado por el torso:

      `hombro -> raiz`  p50 **0,298**  (0,141 a 0,420)
      `codo -> raiz`    p50 **0,223**  (0,096 a 0,508)

    o sea que el eje del hombro varia un 34 % mas, y gana en **5 de los 8 tiradores**. Es la
    direccion que midio 22.3 (0,226 contra 0,108) pero con menos margen, asi que hay que decirlo
    entero: en 3 de 8 tiradores el eje del codo varia MAS, y el criterio declarado era "en la
    mayoria", que se cumple con 5 de 8 y no con mas.

    `eje="codo"` recupera la version de 23.8, que es la linea base.

    POR QUE EN FRACCION DE TORSO Y NO EN METROS: el brazo no esta en el plano del suelo, asi que la
    homografia de la pista no lo mide. El torso del propio tirador es la reglilla que si escala con la
    perspectiva. Es la misma normalizacion que el proyecto ya usa para el error de las poses
    inventadas.

    SE REUTILIZA `cono_lib.mano_armada`, Y HAY QUE DECIR **QUE PELDANO** CORRE AQUI.

    Este docstring decia "verificado a ojo en 4 de 4 tiradores". Esa cifra la **retiro el propio
    `cono_lib`**: no era falsa, era una muestra de 4. Contra los 27 anotados, el criterio
    geometrico suelto (la mano con x mas cerca del rival) da **0,741 (20 de 27)**, y sobre los 59
    anotados despues, **0,831**.

    Y hay una segunda cosa que decir, porque es la diferencia entre dos numeros publicados. Aqui se
    llama **sin `accion=`**, asi que el peldano `encarado_accion` de la cascada NO puede disparar:
    lo que corre en la etapa 7 es `pie_firme -> hombro_firme -> nariz_dir_dudosa -> geometria_a`.
    El **0,949 sobre 59** y el **1,000 sobre los 19 reservados** son de la cascada CON ese peldano
    (`pie_firme -> encarado_accion -> mayoria_3`), asi que **no se pueden atribuir a lo que esta
    linea calcula**. Pasar `accion` moveria las cifras de la pista, o sea que es un cambio
    declarado y medido, no un arreglo de paso: queda identificado y NO hecho.
    """
    import cono_lib as CL
    KP, SC = poses["KP"], poses["SC"]
    T = min(KP.shape[0], r["T"])
    out = []
    for lado in (0, 1):
        try:
            mano, _d = CL.mano_armada(KP[:T], SC[:T], r["B"][:T], r["W"], r["H"], lado)
        except Exception:
            mano = None
        if mano is None:
            continue
        M = CL.MANOS[mano]
        """EL INDICE DEL HOMBRO SE TOMA DE `cono_lib` SI ESTA Y SI NO DEL MAPA MEDIDO, y la
        defensa hace falta porque `cono_lib.mano_armada` la esta reescribiendo otro agente con la
        senal del guante. El mapa de indices del proyecto es fijo (5-6 hombros, 7-8 codos, raiz de
        mano A en 21 y de mano B en 42), asi que la mano A va con el hombro 6 y la B con el 5. Si
        `MANOS` cambia de forma, esto sigue midiendo lo mismo en vez de romperse."""
        hombro = M.get("hombro", 6 if mano == "A" else 5)
        i0 = M["codo"] if eje == "codo" else hombro
        L = np.full(T, np.nan)
        for k in range(T):
            sc = SC[k, lado]
            kp = KP[k, lado]
            if min(sc[i0], sc[M["root"]]) < CONF_MIN:
                continue
            th = [i for i in HOMBROS + CADERAS if sc[i] >= CONF_MIN]
            if len(th) < 2:
                continue
            hom = np.nanmean([kp[i] for i in HOMBROS if sc[i] >= CONF_MIN] or [np.nan], axis=0)
            cad = np.nanmean([kp[i] for i in CADERAS if sc[i] >= CONF_MIN] or [np.nan], axis=0)
            torso = float(np.linalg.norm(hom - cad)) if np.isfinite(hom).all() and \
                np.isfinite(cad).all() else np.nan
            if not np.isfinite(torso) or torso < 5.0:
                continue
            L[k] = float(np.linalg.norm(kp[M["root"]] - kp[i0])) / torso
        base = np.nanmedian(L)
        if not np.isfinite(base) or base <= 0:
            continue
        p98 = float(np.nanpercentile(L, 98))
        Ls = _suave(L, 3)
        alto = Ls > BRAZO_UMBRAL * base
        i = ks[0]
        while i <= min(ks[-1], T - 1):
            if not alto[i]:
                i += 1
                continue
            j = i
            while j + 1 <= min(ks[-1], T - 1) and alto[j + 1]:
                j += 1
            if j - i + 1 >= BRAZO_MIN_CUADROS:
                kp = int(i + np.nanargmax(Ls[i:j + 1]))
                pico = float(Ls[kp])
                """`completa` ES RELATIVA AL PROPIO CLIP y se dice: no hay ninguna medida absoluta
                del largo del brazo de un tirador, asi que lo unico que se puede afirmar es que esta
                extension llega a casi lo maximo que ese tirador ha llegado en este clip."""
                out.append({"tipo": "brazo", "lado": lado, "cuadro": kp, "inicio": int(i),
                            "fin": int(j), "pos_m": _pos(C, kp, lado),
                            "magnitud_m": float(pico / base - 1.0),
                            "unidad_magnitud": "fraccion de torso, crecimiento sobre la mediana",
                            "ext_torso": float(pico / base),
                            "ext_frac_p98": round(float(pico / p98), 4)
                            if np.isfinite(p98) and p98 > 0 else None,
                            "completa": bool(np.isfinite(p98) and p98 > 0
                                             and pico >= BRAZO_COMPLETA * p98
                                             and pico >= BRAZO_COMPLETA_MEDIANA * base),
                            "mano": mano, "eje": "%s -> raiz de la mano" % eje})
            i = j + 1
    return out


def _pos(C, k, lado):
    """QUE HACE: la posicion en metros del tirador en ese cuadro, o NaN si no se conoce."""
    p = C[k, lado] if 0 <= k < len(C) else np.array([np.nan, np.nan])
    return [None if not np.isfinite(p[0]) else round(float(p[0]), 3),
            None if not np.isfinite(p[1]) else round(float(p[1]), 3)]


ACCION_V_CAJA = 0.29
"""Umbral de la accion en ALTOS DE CAJA POR SEGUNDO, para los clips SIN homografia.

POR QUE EXISTE: `accion` es el evento que otro agente necesita como puerta de su regla de prioridad,
y su conjunto reservado son 10 clips de los que **6 no tienen homografia**. Sin homografia no hay
metros, asi que `apoyos()` no puede correr y los seis daban CERO eventos. Tener la pose de 308 puntos
no basta: los eventos metricos necesitan ademas la homografia, y eso hay que decirlo porque el
encargo suponia que con la pose ya estaba.

Lo que SI se puede medir sin homografia es el desplazamiento del centro de la caja en PIXELES del
cuadro estabilizado, normalizado por el ALTO DE LA CAJA del propio tirador. Es una medida de imagen:
no necesita escala metrica, solo la similitud de estabilizacion, que es un ajuste sobre puntos del
fondo.

EL UMBRAL SE ELIGE PARA REPRODUCIR EL METRICO, no se inventa: `ACCION_V_MS` = 0,50 m/s en un tirador
de 1,75 m son `0,50 / 1,75 = 0,29` altos de caja por segundo. Es el mismo criterio en otra unidad.

Y VA MARCADO: la columna `via_accion` del CSV dice `metros` o `alto_de_caja`, porque las dos vias no
son la misma medida y mezclarlas sin decirlo seria el convenio sin declarar que este proyecto ya ha
pagado tres veces."""

LADO_NO_VALIDO = {("IdolizedFancyChickadee", 1): "es el ARBITRO, no un tirador (verificado a ojo)",
                  ("PowerfulAllLarva", 0): "es el ARBITRO, no un tirador (verificado a ojo)"}
"""LADOS QUE NO SON UN TIRADOR, con su razon. Es una lista DECLARADA a mano y a proposito.

`IdolizedFancyChickadee` lado 1 es el **arbitro**, verificado a ojo por el agente de la mano
dominante. Es el tercer caso del defecto de la seccion 18, que sigue sin resolver. Sus eventos de
pies son los de una persona **andando por la pista**, no esgrimiendo, asi que sus marchas no valen
como referencia de nada.

**LA MARCA VA EN UNA COLUMNA Y NUNCA EN LA AUSENCIA DE FILA**, y esto es una correccion de como lo
planteé: los eventos de estos lados SE EMITEN todos. Si se filtraran, su cero se volveria
indistinguible de "no medido", que es el fallo que este proyecto lleva evitando todo el dia. Y aqui
el cero es precisamente el dato: **lo que se quiere saber del arbitro es lo que NO tiene**.

SON LOS DOS UNICOS CLIPS DEL PROYECTO CON UN ARBITRO ENTREGADO COMO TIRADOR, 2 de 33 anotados
(6,1 %), y es el defecto sin resolver mas recurrente que hay. La columna `lado_valido` los marca y
quien agregue tasas por tirador tiene que filtrarla; quien quiera estudiar al arbitro, al contrario.

Y HAY UNA VIA QUE ESTO ABRE, que merece medirse: un arbitro **no hace a fondo ni extiende el brazo
armado**. Si el perfil de eventos de este lado sale sin fondos y sin brazo mientras el lado 0 los
tiene, eso seria un discriminador del arbitro que el proyecto no tiene. Se mide en la seccion 31."""

TRANSV_VENTANA = 9
TRANSV_RAZON_MIN = 5.0
TRANSV_CONTROL_M = 0.60


def transversal(ap, mask, lado) -> dict:
    """QUE HACE: la coordenada TRANSVERSAL filtrada, su banda de incertidumbre medida, y si su FORMA
    se puede publicar. Devuelve `{"y": (T,), "ruido_m": float, "excursion_m": float,
    "razon": float, "forma_publicable": bool, "escala_creible": bool}`.

    =======================================================================================
    POR QUE ESTO EXISTE: PINTARLA PLANA ERA IGUAL DE FALSO QUE PINTAR RUIDO
    =======================================================================================

    La version anterior dibujaba cada tirador en un CARRIL FIJO, o sea la transversal puesta a una
    constante, porque 23.7 decia que no era creible. El usuario lo cazo mirando el video: *"en la
    pista se mueven perfectamente recto, y eso en el video no es verdad: van un poco moviendose en el
    eje y... ahora salen como que no hay cambio en el eje desde que empiezan, y si suele haber
    cambio"*. Tiene razon, y la linea recta perfecta es una afirmacion tan falsa como el ruido.

    LO QUE FALTABA ERA SEPARAR DOS COSAS QUE SE TRATABAN COMO UNA:

      (A) que el RUIDO por cuadro sea grande -> medido, p95 **0,043 m**.
      (B) que la EXCURSION real sea del tamanio del ruido -> **no estaba medido**.

    Medido ahora en los 22 tiradores-clip de los 11 clips, con el criterio declarado antes
    (`excursion / ruido >= TRANSV_RAZON_MIN` = 5):

      excursion transversal   0,124 a 2,131 m
      ruido p95 por cuadro    0,009 a 0,071 m
      **razon: p50 19,8, minimo 8,7 -> pasan 22 de 22**

    Asi que la FORMA es senial en todos, sin excepcion, y pintarla plana escondia una senial que esta
    9 a 77 veces por encima del ruido.

    EL FILTRO, y por que una mediana movil de 9: el ruido transversal es de ALTA frecuencia (un pie en
    el aire se proyecta desplazado ese cuadro y al siguiente ya no) y el desplazamiento lateral de un
    tirador es LENTO. Con ventana 9 (0,30 s) se conserva el **97,8 %** de la excursion, o sea que el
    filtro no aplana la forma. Con ventana 5 se conserva el 98,9 % y con 45 el 88,6 %.

    UNA SALVEDAD DE METODO, porque la cifra que la acompanaba no valia: yo habia declarado tambien
    "ruido residual por debajo de 0,020 m", y lo medi como el residuo de la serie ya filtrada contra
    su propia mediana movil de 5. Eso es **circular** y da casi cero por construccion para cualquier
    ventana >= 5, asi que esa mitad del criterio no discrimina y no se usa. La mitad que si vale es la
    excursion conservada.

    Y LO QUE SIGUE SIN ESTAR BIEN, que es la escala: el control fisico declarado en 23.7 es recorrido
    <= `TRANSV_CONTROL_M` = 0,60 m, y lo pasan **11 de 22**. Se probo la hipotesis de que el exceso
    fuera un error de perspectiva que crece con la posicion longitudinal, quitando de `y` su
    dependencia lineal de `x`: pasa de **11 a 14 de 22**. Mejora y no lo arregla. Quedan 8 tiradores
    con recorridos de hasta **1,87 m en una pista de 1,50 m de ancho**, que es imposible.

    CONCLUSION, y es la que decide como se dibuja: **la FORMA de la transversal es senial y se pinta;
    su ESCALA no es creible en 8 de 22 y eso va en el rotulo con su cifra.** No es lo mismo decir "no
    se sabe como se mueve de lado" que decir "se sabe la forma y no el tamanio".
    """
    y = ap["C"][:, lado, 1].astype("float64").copy()
    y[~mask[:len(y)]] = np.nan
    ys = _suave(y, TRANSV_VENTANA)
    r = np.abs(y - _suave(y, 5))
    r = r[np.isfinite(r)]
    ruido = float(np.percentile(r, 95)) if r.size > 10 else float("nan")
    v = ys[np.isfinite(ys)]
    exc = float(np.percentile(v, 95) - np.percentile(v, 5)) if v.size > 10 else float("nan")
    raz = exc / ruido if ruido and np.isfinite(ruido) and ruido > 0 else float("nan")
    return {"y": ys, "y_cruda": y, "ruido_m": ruido, "excursion_m": exc, "razon": raz,
            "forma_publicable": bool(np.isfinite(raz) and raz >= TRANSV_RAZON_MIN),
            "escala_creible": bool(np.isfinite(exc) and exc <= TRANSV_CONTROL_M),
            "ventana": TRANSV_VENTANA}


ANCLA_DERIVA_MAX = 1.00
"""Deriva maxima, en metros, del anclaje del mapa para poder llamarlo utilizable.

1,00 m es un cuarto de los 4,00 m que separan las dos lineas de puesta en guardia, que es la unica
distancia de la pista que este proyecto mide con algun control independiente (C2, 0,29 m de error)."""


def ancla_origen(ap, mask) -> dict:
    """QUE HACE: el ANCLAJE del mapa de la pista, y la medida de cuanto vale ese anclaje.

    EL PROBLEMA, que es de la seccion 23.9 y no se puede resolver calculando: el origen de la
    coordenada longitudinal es una marca transversal que **no esta identificada**, asi que la x
    absoluta no significa nada. Los desplazamientos y las distancias SI estan en metros; "en que
    punto de la pista" no.

    LA CONVENCION, declarada: se toma como centro de la pista el PUNTO MEDIO de los dos tiradores,
    mediana del clip. En un asalto de florete la accion se desarrolla alrededor del centro, asi que
    es la mejor apuesta posible, pero **es una apuesta y no una medida**.

    Y SE MIDE LO QUE VALE: `deriva_m` es el p95 menos el p05 del punto medio dentro del clip. Si el
    punto medio se pasea 2 m, el anclaje tiene al menos 2 m de incertidumbre. Medido en los 4 clips
    publicables con pose: **0,74 · 1,13 · 2,12 · 8,83 m**, asi que con el criterio declarado de
    `ANCLA_DERIVA_MAX` = 1,00 m el anclaje solo es utilizable en **1 de 4**. El de 8,83 m es
    `ForcefulAntiqueAlbacoretuna`, cuyo punto medio no puede pasearse 8,8 m en una ventana de camara
    que abarca 5,5 m: eso no es deriva, es una escala mal puesta, y sale por aqui.

    Quien pinte el mapa tiene que ENSENAR esta cifra. Un mapa de la pista con la traza colocada como
    si se supiera donde esta, cuando el anclaje tiene 8 m de incertidumbre, es peor que no tener
    mapa.
    """
    C = ap["C"]
    mid = 0.5 * (C[:, 0, 0] + C[:, 1, 0])
    m = np.isfinite(mid) & mask
    v = mid[m]
    if v.size < 12:
        x = C[:, :, 0][np.isfinite(C[:, :, 0])]
        return {"x0_m": float(np.median(x)) if x.size else 0.0, "deriva_m": float("nan"),
                "n": int(v.size), "fiable": False,
                "motivo": "menos de 12 cuadros con los dos tiradores"}
    d = float(np.percentile(v, 95) - np.percentile(v, 5))
    return {"x0_m": float(np.median(v)), "deriva_m": d, "n": int(v.size),
            "rango_m": float(v.max() - v.min()),
            "fiable": bool(d <= ANCLA_DERIVA_MAX), "motivo": ""}


def _en(C, k, lado, j, r=4):
    """QUE HACE: la coordenada `j` del tirador en el cuadro `k`, buscando el cuadro finito mas
    cercano hasta `r` de distancia. Un evento cuyo cuadro de inicio cae justo en un hueco de la pose
    no puede quedarse sin su x de inicio por un cuadro."""
    for d in range(r + 1):
        for kk in (k - d, k + d):
            if 0 <= kk < len(C) and np.isfinite(C[kk, lado, j]):
                return float(C[kk, lado, j])
    return float("nan")


HUECO_SIN_MEDIR = 24
"""Largo de hueco por encima del cual el error de la pose inventada NO esta medido. Sale de la
seccion 21.4 de `04_documentacion/BITACORA_2026-09.md`: la curva de error por hueco se midio escondiendo poses
conocidas y no llega mas alla de ese largo, asi que de ahi para arriba no hay cifra, ni buena ni
mala."""

UNIDAD = {"fondo": "m a lo largo de la pista", "marcha": "m a lo largo de la pista",
          "romper": "m a lo largo de la pista", "alto": "m a lo largo de la pista",
          "accion": "m a lo largo de la pista",
          "bote": "m verticales, escala de la caja del tirador",
          "brazo": "fraccion de torso, crecimiento sobre la mediana"}


def serie(ev, ap, clip, fps, poses=None, extra=None) -> list:
    """QUE HACE: la SERIE TEMPORAL CONSULTABLE de eventos, una fila por evento y no por cuadro.

    *"poder saber la tercera marcha del tirador de la derecha y poder ubicarla en el mapa ese de la
    pista, para saber en que momento se hizo y cuanto midio"*

    `indice_del_tipo` se numera 1, 2, 3... por (lado, tipo) en orden de cuadro de inicio, asi que esa
    consulta es una linea de pandas:

        ev.query("lado == 1 and tipo == 'marcha' and indice_del_tipo == 3")

    LO QUE LLEVA CADA FILA ADEMAS DE LA MEDIDA, y es lo que la hace utilizable por un modelo:

      `x_inicio_m x_fin_m`   donde empieza y donde acaba, que es el "inicio de marcha, final de
                             marcha" que pidio el usuario.
      `y_*_m`                la transversal, que va marcada `transversal_creible = False` SIEMPRE:
                             la seccion 23.7 midio que el recorrido transversal falla el control
                             fisico en 8 de 10 clips.
      `origen_max hueco_max` procedencia de la pose en el rango del evento. Un evento sobre pose
                             interpolada es mas debil y con `hueco_max > 24` su error no esta ni
                             medido.
      `unidad_magnitud`      porque NO todas las magnitudes estan en la misma unidad, y meterlas
                             todas en una columna llamada `magnitud_m` sin decirlo seria el mismo
                             error que el proyecto ya cometio con el convenio sin declarar.
    """
    C = ap["C"]
    org = poses["origen"] if poses is not None and poses.get("origen") is not None else None
    hue = poses["hueco"] if poses is not None and poses.get("hueco") is not None else None
    cuenta = {}
    filas = []
    for e in sorted(ev, key=lambda z: (z.get("lado", -1), z["tipo"], z["inicio"])):
        lado = int(e.get("lado", -1))
        tipo = e["tipo"]
        cuenta[(lado, tipo)] = cuenta.get((lado, tipo), 0) + 1
        k0, k1 = int(e["inicio"]), int(e["fin"])
        lref = lado if lado >= 0 else 0
        f = {"clip": clip, "lado": lado, "tipo": tipo,
             "indice_del_tipo": cuenta[(lado, tipo)],
             "cuadro_inicio": k0, "cuadro_fin": k1, "cuadro_pico": int(e["cuadro"]),
             "t_inicio_s": round(k0 / fps, 3), "t_fin_s": round(k1 / fps, 3),
             "duracion_s": round((k1 - k0) / fps, 3),
             "x_inicio_m": _r(_en(C, k0, lref, 0)), "x_fin_m": _r(_en(C, k1, lref, 0)),
             "y_inicio_m": _r(_en(C, k0, lref, 1)), "y_fin_m": _r(_en(C, k1, lref, 1)),
             "magnitud_m": _r(e.get("magnitud_m", float("nan")), 4),
             "unidad_magnitud": e.get("unidad_magnitud", UNIDAD.get(tipo, "")),
             "velocidad_m_s": _r(e.get("v_ms", float("nan"))),
             "transversal_creible": False,
             "ext_torso": _r(e.get("ext_torso", float("nan")), 4),
             "completa": e.get("completa"),
             "levanta_frac_caja": e.get("levanta_frac_caja"),
             "sobre_ruido": e.get("sobre_ruido"),
             "tipo_establecido": e.get("tipo_establecido", True),
             "pies_movidos": e.get("pies_movidos"),
             "solapa_fondo": e.get("solapa_fondo"),
             "arbitrado": e.get("arbitrado"),
             "razon_pie_atras": e.get("razon_pie_atras"),
             "pico_dpies_guardia": e.get("pico_dpies_guardia"),
             "margen_arbitraje": e.get("margen_arbitraje"),
             "guardia_m": e.get("guardia_m"),
             "guardia_plausible": (None if e.get("guardia_m") is None
                                   else bool(0.50 <= e["guardia_m"] <= 0.90)),
             "lado_valido": (None if lado < 0 else (clip, lado) not in LADO_NO_VALIDO),
             "via_accion": e.get("via_accion", "metros" if tipo == "accion" else None),
             "magnitud_altos_caja": e.get("magnitud_altos_caja"),
             "lado_no_valido_por": LADO_NO_VALIDO.get((clip, lado)),
             "sep_pies_m": _r(e.get("sep_pies_m", float("nan"))),
             "umbral_quieto_m": e.get("umbral_quieto_m"),
             "mano": e.get("mano"), "eje": e.get("eje")}
        if lado >= 0 and org is not None and k1 < len(org):
            o = org[k0:k1 + 1, lado]
            h = hue[k0:k1 + 1, lado] if hue is not None and k1 < len(hue) else None
            f["origen_max"] = int(o.max()) if o.size else None
            f["pose_fiable"] = bool(o.size and o.max() == 0)
            f["hueco_max"] = int(h.max()) if h is not None and h.size else None
            f["hueco_sin_medir"] = bool(f["hueco_max"] is not None
                                        and f["hueco_max"] > HUECO_SIN_MEDIR)
        else:
            f.update({"origen_max": None, "pose_fiable": None, "hueco_max": None,
                      "hueco_sin_medir": None})
        if extra:
            f.update(extra)
        filas.append(f)
    filas.sort(key=lambda z: (z["cuadro_inicio"], z["lado"], z["tipo"]))
    return filas


def _r(v, n=3):
    """QUE HACE: redondea o deja None si no es finito, para que el CSV no lleve NaN de texto."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, n) if np.isfinite(v) else None


def posicion_entre_bordes(res, poses=None) -> dict:
    """QUE HACE: la posicion de los dos tiradores medida SOBRE LA TIRA RECTIFICADA, o sea entre los
    dos bordes detectados de la pista, en vez de por una homografia global y una convencion.

    =======================================================================================
    LOS DOS FALLOS QUE ARREGLA, y el usuario los vio los dos en el mismo video
    =======================================================================================

      *"tomas tu por defecto que empiezan siempre en el centro de la pista, pero no: muchos igual
      empiezan al final de la pista de la derecha... y abajo en el grafico salen como en el centro"*

      *"sacar mejor la distancia en el eje del ancho de la pista, porque los pones siempre el de la
      derecha pegado a la derecha y el de la izquierda pegado a la izquierda, pero no estas
      trackeando de verdad su posicion real"*

    El primero era `ancla_origen` haciendo de ancla con su valor por defecto, el centro. El segundo
    era el CARRIL FIJO del dibujo, que no era una medida de nada.

    =======================================================================================
    QUE SE MIDE Y CON QUE FIABILIDAD, y las dos coordenadas NO son iguales
    =======================================================================================

    A LO ANCHO, `Y` en metros: son METROS y estan ACOTADOS POR CONSTRUCCION. La fila de la tira es
    afin en la coordenada transversal del mundo (ver `pista.tira_rectificada`), los dos bordes son
    las filas 0 y 1, y el reglamento pone 1,50 m entre ellos (t.18.1). Se renormaliza ademas entre
    los bordes de la ZONA DE COMBATE que devuelve `pista.ancho_combate`, o sea **excluyendo la
    plancha lateral**, que es lo que el usuario aviso que no es pista valida (t.18.3). Un recorrido
    transversal de 1,65 m en una pista de 1,50, que es lo que salia con la homografia global, ya no
    puede ocurrir salvo que el tirador se salga de verdad; y si sale, se cuenta en
    `y_fuera_de_pista` en vez de recortarlo.

    A LO LARGO, `s_rel` y `X`: `s_rel` es la posicion NORMALIZADA en el tramo visible de pista, de 0
    a 1, y es afin en metros con factor desconocido. `X` son metros desde la linea central y **solo
    existe si hay ancla medida**; si no la hay va a `nan` y `metros_a_lo_largo` sale False.

    POR QUE LA ESCALA A LO LARGO NO SE ENTREGA COMO BUENA, y esto es el resultado medido de esta
    tanda, no una excusa. Se probaron TRES vias independientes sobre los 20 clips con tira:

      RAYAS PINTADAS del reglamento (central y guardia), por razones de separacion: encaja en 14
        clips pero sale AMBIGUA en 10 de esos 14, y contra la homografia de referencia de tres clips
        se equivoca por factores de 1,3 a 2,6
      PIXEL CUADRADO (angulo entre las dos direcciones del suelo mas el ancho reglamentario): da
        solucion exacta en varios clips, pero la estatura reproyectada de los tiradores, que es un
        control independiente, sale de 0,4 a 4,9 m
      FRONTERAS DE ZONA DE COLOR (t.19): encaja en 13 clips, pero en 11 de ellos solo con DOS
        fronteras, y con dos el sistema esta exactamente determinado y no hay ningun control

    Y hay una razon geometrica por la que esto no es cuestion de afinar: **dos rectas paralelas con
    separacion conocida fijan la escala TRANSVERSAL en todo el plano, pero no dicen nada de la
    LONGITUDINAL**. Hace falta una referencia mas, y en este corpus las referencias disponibles no
    aguantan su propio control. Una coordenada normalizada honesta vale mas que unos metros
    inventados.
    """
    t = PI.tira_rectificada(res)
    if t is None:
        return {"ok": False, "motivo": "sin tira rectificada (bordes no detectados)"}
    ac = PI.ancho_combate(t)
    an = PI.ancla_zonas(t)
    u0, u1 = (ac.get("u0", 0.0), ac.get("u1", 1.0)) if ac.get("ok") else (0.0, 1.0)
    plancha_usada = bool(ac.get("ok") and ac.get("plancha"))
    if not (np.isfinite(u0) and np.isfinite(u1)) or not (-0.10 <= u0 < u1 <= 1.10) \
            or (u1 - u0) < 0.50:
        """SI EL TRAMO NO CAE DENTRO DE LOS BORDES, se usan los bordes. Y esto tenia que ser un
        rechazo y no un pase: la primera version solo exigia `u1 - u0 >= 0,25`, y `ancho_combate`
        devolvia `u = -0,40 a 1,40` en 18 de 22 clips, o sea todo el muestreo incluido el suelo. Con
        eso la transversal se dividia por 1,8 y los dos tiradores salian a 0,50 y 0,56 m del borde,
        los dos casi en el centro de la pista. Se vio en el fotograma del video antes de publicarlo:
        los dos puntos del mapa casi en la misma linea."""
        u0, u1, plancha_usada = 0.0, 1.0, False
    r, Mk = res["_r"], res["_Mk"]
    B, pres, mask, W, H, T = r["B"], r["pres"], r["mask"], r["W"], r["H"], r["T"]
    ncol = t["ncol"]
    KP = SC = None
    if poses is not None and poses["KP"].shape[0] >= T:
        KP, SC = poses["KP"], poses["SC"]
    S = np.full((T, 2), np.nan)
    Y = np.full((T, 2), np.nan)
    S1 = np.full((T, 2), np.nan)
    S2 = np.full((T, 2), np.nan)
    org = np.full((T, 2), -1, "int8")

    def mide(p, k):
        """QUE HACE: un punto de imagen -> (columna de la tira, metros a lo ancho)."""
        q = PI.a_tira(t, _lleva(Mk[k], p))[0]
        u = float(PI.u_de_fila(t, q[1]))
        uc = (u - u0) / (u1 - u0) if (u1 - u0) > 1e-9 else np.nan
        return float(q[0]), uc * PI.ANCHO_NOMINAL

    for k in np.nonzero(mask)[0]:
        for lado in (0, 1):
            if not pres[k, lado]:
                continue
            pies = _pie_pose(KP, SC, k, lado) if KP is not None else [None, None]
            if pies[0] is not None and pies[1] is not None:
                qa = mide(pies[0], k)
                qb = mide(pies[1], k)
                S1[k, lado], S2[k, lado] = qa[0], qb[0]
                S[k, lado] = 0.5 * (qa[0] + qb[0])
                Y[k, lado] = 0.5 * (qa[1] + qb[1])
                org[k, lado] = 0
                continue
            b = B[k, lado]
            if not np.isfinite(b).all():
                continue
            q = mide(np.array([0.5 * (b[0] + b[2]) * W, b[3] * H]), k)
            S[k, lado], Y[k, lado] = q[0], q[1]
            org[k, lado] = 1
    """EL CONVENIO DEL EJE SE MANTIENE Y AQUI SALE GRATIS: `pista.corredor` orienta su eje con
    `e[0] > 0`, o sea que las columnas de la tira crecen hacia la derecha de la imagen, que es
    exactamente el convenio declarado en `apoyos`. No hay que voltear nada, y eso es mejor que
    voltear: voltear la x sin voltear el ancla dejaria el ancla sin significado."""
    s_rel = S / max(ncol - 1.0, 1.0)
    ok2 = mask & np.isfinite(S[:, 0]) & np.isfinite(S[:, 1])
    cruces = int((S[ok2, 0] > S[ok2, 1]).sum())
    fuera = int((np.isfinite(Y) & ((Y < -0.05) | (Y > PI.ANCHO_NOMINAL + 0.05))).sum())
    n_y = int(np.isfinite(Y).sum())
    out = {"ok": True, "S": S, "s_rel": s_rel, "Y": Y, "S1": S1, "S2": S2, "origen": org,
           "ncol": ncol, "cruces": cruces, "n_con_los_dos": int(ok2.sum()),
           "y_fuera_de_pista": fuera, "n_y": n_y,
           "frac_fuera_de_pista": (fuera / n_y) if n_y else float("nan"),
           "plancha": ac.get("plancha"), "plancha_excluida": plancha_usada,
           "u0": u0, "u1": u1, "ancho_combate_m": ac.get("ancho_combate_m"),
           "metros_a_lo_largo": bool(an.get("ok")),
           "via_ancla": an.get("via_ancla") if an.get("ok") else None,
           "ancla_ambigua": an.get("ambiguo"), "ancla_controlada": an.get("controlada"),
           "ancla_n_ident": an.get("n_ident"), "ancla_loo_m": an.get("loo_p50_m"),
           "m_por_col": an.get("m_por_col"), "q_m": an.get("q_m"),
           "convenio_eje": "columna crece hacia la derecha de la imagen; lado 0 = izquierda",
           "convenio_ancho": "0 m en un borde de combate, 1,50 m en el otro (t.18.1)"}
    if an.get("ok"):
        out["X"] = an["m_por_col"] * S + an["q_m"]
        out["dist_m"] = np.abs(out["X"][:, 0] - out["X"][:, 1])
        out["x_visible_m"] = (an["x_ini_m"], an["x_fin_m"])
    else:
        out["X"] = np.full_like(S, np.nan)
        out["dist_m"] = np.full(T, np.nan)
    out["dist_rel"] = np.abs(s_rel[:, 0] - s_rel[:, 1])
    out["recorrido_y_m"] = [float(np.nanmax(Y[:, l]) - np.nanmin(Y[:, l]))
                            if np.isfinite(Y[:, l]).any() else float("nan") for l in (0, 1)]
    return out

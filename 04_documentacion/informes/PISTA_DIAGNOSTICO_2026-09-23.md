# La posicion en la pista: donde falla exactamente, medido contra verdad externa

El usuario lo vio en un GIF de 12 segundos: *"la posicion en la pista no es la correcta"*. Tenia
razon, y ahora hay numeros. Este informe dice **que parte falla**, porque no falla todo, y eso
cambia por completo lo que hay que arreglar.

---

## El resumen en una linea

**La ESCALA funciona. El ORIGEN esta roto.** Lo que se mide bien son las magnitudes relativas
-- cuanto avanza un paso, a que distancia estan los dos tiradores -- y lo que no significa nada es
**donde en la pista** esta ocurriendo.

---

## 1 · El control externo que este proyecto nunca habia tenido

Todos los controles anteriores eran **internos**: la deriva del anclaje se mide contra el propio
anclaje, el recorrido lateral contra el ancho reglamentario. Dicen que algo va mal, pero no
**donde estaba el tirador de verdad**.

`strip_location` si: un humano miro el clip y dijo en que zona de la pista ocurrio la accion. Hay
**3.964 clips de florete** anotados, y **337** de ellos ya estan extraidos.

`02_validacion/herramienta/verifica_pista_etiqueta.py`.

## 2 · Lo medido, y es demoledor

Nuestra x mediana **dentro de cada zona que dijo el humano**:

| el humano dice | deberia ser | **nuestra x** |
|---|---:|---:|
| `fotl_warning_box` | −6,0 m | **−0,49 m** |
| `fotl_half` | −3,5 m | **−0,61 m** |
| `middle` | 0,0 m | **−0,45 m** |
| `fotr_half` | +3,5 m | **+0,27 m** |
| `fotr_warning_box` | +6,0 m | **−0,40 m** |

Ajuste: `nuestra_x = 0,037 * x_real - 0,34`. **Pendiente 0,037**: cuando la posicion real se mueve
un metro, nuestra estimacion se mueve **3,7 cm**.

Y el acierto por zonas lo confirma:

| | acierto |
|---|---:|
| nuestra x | **0,380** |
| **contestar siempre `middle`** | **0,430** |
| techo: acuerdo entre dos anotadores | 0,619 |

**La medida es peor que una constante.** Eso no es "poco precisa": es que no lleva informacion.

## 3 · Tres hipotesis, dos descartadas midiendo

**(a) La camara sigue la accion y la estabilizacion cancela la posicion.** DESCARTADA. Si fuera
eso, el recorrido dentro de un clip seria pequenio; es p50 **7,87 m**.

**(b) Los desbordamientos numericos destrozan la mediana.** DESCARTADA COMO CAUSA PRINCIPAL, pero
**es un defecto real aparte**: el recorrido dentro de un clip llega a **22.806 m** y la deriva del
anclaje a **9.016 m** en una pista de 14. Rechazando lo fisicamente imposible (`|x| > 7,5 m`) se
tira el **9,7 %** de los puntos y la pendiente pasa de 0,037 a **0,071**. Mejora, pero no arregla.

**(c) El ORIGEN es aleatorio por clip.** CONFIRMADA, por dos vias independientes:

  - el control interno ya lo decia: `ancla_fiable` pasa en **56 de 367** clips (15 %), deriva
    mediana **2,72 m**, p90 **14,19 m**, y solo **14 de 367** por debajo de 0,50 m;
  - y las magnitudes que **no dependen del origen** salen bien:

| medida (independiente del origen) | valor | plausible en esgrima |
|---|---|---|
| distancia entre los dos tiradores | p50 **3,75 m**, 70 % en 1-5 m | 1,5-4 m, si |
| magnitud de un paso | p50 **0,37 m**, 76 % en 0,15-1,5 m | 0,3-0,6 m un paso, si |

Si la escala estuviera mal, estas dos tambien lo estarian. No lo estan.

## 4 · Que se sigue de esto

**Lo que SI se puede usar hoy, y es mucho:** longitud de paso, velocidad, duracion, cadencia,
distancia entre tiradores, magnitud del fondo. Todo lo **relativo**. Para las estadisticas de
entrenamiento -- cuanto avanza, a que ritmo, a que distancia trabaja -- esto vale.

**Lo que NO se puede usar:** la x absoluta en la pista, y por tanto `strip_location` derivada de
ella, y la posicion pintada en el mapa. Hoy se publica un numero que no significa nada.

**Y lo primero que hay que hacer no es mejorar el anclaje: es dejar de afirmar lo que no se sabe.**
La disciplina del proyecto -- *"`None` no es cero"* -- no se esta aplicando aqui. La x absoluta
deberia salir vacia cuando el anclaje no pasa su control, igual que los campos metricos salen
vacios cuando no hay homografia. Eso solo ya quita el 85 % de las afirmaciones falsas.

## 4b · LO QUE APARECIO AL MIRARLO, que es lo que decide el arreglo

El usuario insistio en dos cosas y las dos eran correctas: *"si un humano viendo un frame sabe en
que parte estan, tu tambien"* y *"verificalo visualmente"*. Al hacerlo salieron cuatro cosas.

**1. Hay DOS sistemas de posicion y los eventos usan el peor.**

    ancla_origen   el punto medio de LOS PROPIOS TIRADORES, mediana del clip.
                   Su docstring lo dice: "es una apuesta y no una medida".
                   -> es lo que alimenta `x_inicio_m` de los eventos
    ancla_zonas    las fronteras de color de t.19, identificadas contra el patron FIE.
                   -> alimenta la `X` de la tira, que usa el video

Con `ancla_origen`, **los tiradores salen centrados en x~0 por construccion**: el origen se define
como su punto medio, asi que se mueve CON ellos. De ahi la pendiente 0,037.

**2. Pero el sistema "bueno" tampoco funciona.** Medido igual contra la etiqueta humana sobre 23
clips: pendiente **-0,081**, acierto 0,261. O sea que cambiar de sistema no arregla nada, y la
hipotesis "usa el otro" queda descartada tambien.

**3. El `ident_err_p50_m = 0,000 m` NO es precision. Es aritmetica.** Con `n_ident = 2` marcas y
un modelo de **dos** parametros -- escala y origen -- el residuo es cero se acierte o se falle de
marca. Es exactamente el error que el proyecto ya tiene documentado con el cono: *"la hoja esta a
0 grados de su propio centro"* es cierto por construccion y no comprueba nada.

El unico control que vale es **dejar una fuera**: ajustar con dos marcas y predecir una tercera
que no vio. Eso es `loo_p50_m`, y exige TRES marcas. Medido sobre 10 clips:

    clips con el control de deja-una-fuera:  1 de 10

**El anclaje se esta usando sin control en el 90 % de los casos.**

**4. No es un problema de DETECCION sino de CORRESPONDENCIA.** Las marcas se encuentran bien:
p50 **5,5** por clip (min 2, max 13), y **13 de 18** clips identifican dos o mas. La ventana de
pista visible es p50 **14,06 m**, o sea la pista entera. Lo que falla es decidir **cual marca es
cual**, y una asignacion equivocada da residuo cero igualmente -- el propio modulo ya lo tenia
medido para las rayas pintadas: *"la escala que dan se equivoca por factores de 1,3 a 2,6"*.

**Y un control nuevo que lo caza solo.** Si el patron identificado abarca mas metros de los que
abarca la tira, la asignacion es imposible. Medido: `AbandonedBiodegradableCoqui` identifica
`[-7, 0, +2]` -- **9 m** -- en una tira que abarca **6,06 m**. No caben. Sale en **1 de 4** clips
que llegan a tener las dos cifras, y hoy **nada lo comprueba**.

La herramienta para mirarlo es `02_validacion/herramienta/verifica_marcas_pista.py`: dibuja sobre
la tira rectificada donde el anclaje situa cada marca del reglamento, con los dos bordes
detectados en blanco para poder distinguir un fallo de rectificacion de uno de emparejamiento.

## 4c · EL FALLO EXACTO DEL EMPAREJAMIENTO, leido en el codigo

`identifica_marcas` hace RANSAC **sobre PARES**:

    para cada par (i, j) de marcas detectadas:
        ajusta A, B haciendo pasar la recta por ESAS DOS
        cuenta cuantas de las demas caen cerca de alguna marca del reglamento
        se queda con la hipotesis de MAS INLIERS

Dos consecuencias, y las dos explican lo que se ve:

**1. El residuo publicado esta contaminado por las semillas.** Las dos marcas del par tienen error
**cero por construccion**: definen la recta. Con `n_ident = 3`, dos son semillas y una es predicha,
asi que la **mediana** cae sobre una semilla y sale `0,000`. No es precision.

**2. Se elige por numero de inliers, no por poder predictivo.** Las marcas del reglamento estan en
-7, -5, -2, 0, +2, +5, +7: las separaciones son **2, 3, 2, 2, 3, 2**, o sea que se repiten. Una
escala equivocada encuentra inliers igualmente -- es exactamente el *"la escala que dan se equivoca
por factores de 1,3 a 2,6"* que el propio modulo tiene medido para las rayas.

**Cuantos anclajes admiten un control de verdad**, medido sobre 28 clips con homografia:

| marcas identificadas | clips | marcas PREDICHAS (n-2) | vale como control? |
|---:|---:|---:|---|
| 0 | 6 (21 %) | 0 | no hay |
| 3 | 9 (32 %) | **1** | una sola prediccion **no discrimina** |
| 4 | 3 (11 %) | 2 | flojo |
| >= 5 | **10 (36 %)** | >= 3 | **si** |

Y el control de coherencia nuevo -- que el patron no abarque mas metros que la tira -- caza
**1 de 8**. Hoy no se comprueba.

## 4d · Y POR ESO LA IDEA DE "ANCLAR DONDE SE PUEDA Y PROPAGAR" ES LA BUENA, pero un nivel arriba

La propuso el usuario. Medido, **dentro de un clip no tiene material**: la camara se desplaza p50
**107 px de 1920 (6 %)**, o sea ~0,5 m de pista extra. Todos los cuadros ven casi lo mismo y no hay
nada nuevo que entre en escena.

**Entre clips de la misma sede si**, y mucho: misma pista, misma posicion de camara, cientos de
clips (`anaheimfoil2018` tiene 772 anotados). Con un matiz que cambia el diseno: **el zoom varia**
entre clips de la misma sede -- desviacion del ancho de la pista de **9 a 41 px sobre ~131**, o sea
del 7 % al 31 %. Asi que **lo que se puede compartir no es la escala: es la CORRESPONDENCIA**, que
es justo lo que falla.

El metodo que sale de juntar las dos cosas:

  1. por sede, quedarse con los clips de **>= 5 marcas** (el 36 %), que son los unicos con control;
  2. de esos, tomar el mejor por **error de prediccion sobre las marcas NO semilla**;
  3. fijar ahi que marca visible es cual, y **propagar esa asignacion** al resto de la sede,
     reajustando solo la escala clip a clip;
  4. validar con `strip_location`, que da verdad en 3.964 clips.

Eso convierte 5.856 decisiones ambiguas en **~150 decisiones por sede**, cada una tomada donde hay
mas informacion. Y las tres correcciones al emparejamiento se pueden hacer a la vez:

    residuo sobre las marcas NO SEMILLA          hoy incluye las dos que definen la recta
    desempatar por error de prediccion           hoy se desempata por numero de inliers
    rechazar si el patron no cabe en la tira     hoy no se comprueba, y falla en 1 de 8

## 4e · EL EMPAREJAMIENTO NUEVO, construido y careado

`01_extraccion/codigo/ancla_v2.py`, con las tres correcciones. La que de verdad cambia el
resultado es la segunda:

**A. El residuo se mide solo sobre las marcas NO semilla.** Las dos del par definen la recta y su
error es cero por construccion; incluirlas en la mediana es lo que producia el `0,000 m`.

**B. Se puntua por COMPLETITUD en vez de por numero de inliers.** Si la hipotesis dice que el
tramo visible va de -7 a -1, entonces -7, -5 y -2 **deberian estar las tres**. La puntuacion es
la fraccion de marcas del reglamento **dentro del tramo** que estan explicadas. Eso penaliza las
escalas infladas, que es justo lo que el conteo de inliers no podia distinguir: una escala 2x mete
mas marcas del reglamento en el tramo, y si no aparecen, se hunde.

**C. Se rechaza lo que no cabe** en la tira.

### Lo medido, careando los dos sobre EXACTAMENTE los mismos datos

Se sustituye `pista.identifica_marcas` por un espia que llama a los dos con el mismo `sp`, las
mismas lineas y los mismos bordes, asi que cualquier diferencia es del emparejamiento.
`02_validacion/herramienta/verifica_ancla_v2.py`. Sobre 60 clips:

| | viejo | nuevo |
|---|---:|---:|
| clips que contestan | 30 | **29** |
| **patrones con posiciones REPETIDAS** (imposible) | **8 de 30 (27 %)** | **0 de 29** |
| completitud (marcas esperadas que explica) | — | p50 **1,00**, media **0,90** |

**El 27 % de los anclajes de hoy asignan dos rayas distintas a la MISMA marca del reglamento**, y
salen adelante porque la comprobacion existente solo exige que queden `MIN_IDENT` posiciones
*distintas*: un patron `[-5, -5, -2, -2, 0]` tiene tres distintas y pasa. Ejemplos reales del
careo:

    viejo  [-5.0, -5.0, -2.0, -2.0, ...]      nuevo  [-2.0, 0.0, 2.0, 5.0]
    viejo  [-7.0, -5.0, -5.0, -5.0, ...]      nuevo  [-7.0, -5.0, -2.0]
    viejo  [-7.0, -2.0,  2.0]                 nuevo  [-5.0, -2.0, 0.0]

Y **no compra el acierto negandose a contestar**: cubre 29 de los 30 que cubria el viejo. En 7 de
13 clips los dos identifican patrones distintos, o sea que el cambio mueve mas de la mitad de los
anclajes.

### Lo que esto todavia NO demuestra

Que la POSICION resultante sea correcta. Eliminar lo imposible es condicion necesaria y no
suficiente: un patron limpio puede seguir estando desplazado en bloque. Para eso hay que rehacer
la etapa 7 con el emparejamiento nuevo y volver a medir contra `strip_location` con el criterio ya
declarado -- batir el **0,430** y subir la pendiente de **0,07** a **> 0,70**.

**Y por que no se ha cambiado `pista.py` todavia:** la extraccion esta corriendo. El anclaje solo
afecta a la etapa 7, que cuesta **13 s/clip en CPU**; las etapas 1 y 4 -- las caras, 42 s/clip de
GPU -- no dependen de el. Asi que lo correcto es dejar correr lo caro, y **rehacer solo la etapa 7
al final**: son ~21 h de CPU que van en paralelo, en vez de tirar 48 h de GPU.

## 5 · Como se arregla el anclaje, por orden de potencia

**El problema geometrico, dicho con precision.** Los dos bordes de la pista son dos rectas
paralelas de separacion conocida (1,50 m, t.18.1). Eso fija la escala **transversal** y el punto
de fuga longitudinal. Lo que **no** dan es ni el origen ni la escala **a lo largo**: por eso tres
vias fallaron y hay una razon, no mala suerte.

Lo que si lo daria:

**1. Las marcas TRANSVERSALES, que son siete y estan a distancias conocidas.** Linea central (0),
las dos de puesta en guardia (+-2 m), las dos fronteras de color de t.19 (+-5 m) y los dos limites
(+-7 m). **Detectar dos cualesquiera fija el origen Y la escala longitudinal de golpe.** Hoy solo
se buscan las fronteras de color, y aparecen en 13 de 22 clips. Las lineas de puesta en guardia
son lineas pintadas de alto contraste y **no se estan usando**.

**2. La razon doble con el punto de fuga.** Con el punto de fuga longitudinal -- que ya se tiene,
es la interseccion de los dos bordes -- y **una sola** marca conocida, la razon doble (invariante
proyectiva) propaga distancias a lo largo de toda la pista. Es metrologia de vista unica estandar
y con una marca basta, no dos.

**3. Los modelos de calibracion de campo** (PnLCalib, TVCalib, KaliCalib). No inventan escala: lo
que aportan es **detectar las marcas con mas recall**, que es exactamente el cuello -- 13 de 22
con ancla, 5 con control. El titular honesto es *"mejor deteccion de marcas"*, no *"pista
resuelta"*.

**Y ahora, por primera vez, hay con que validar cualquiera de las tres**: los 3.964 clips con
`strip_location`. El criterio declarado antes de intentarlo:

    se adopta si   el acierto por zonas bate el 0,430 de contestar siempre `middle`
                   y la pendiente del ajuste sube de 0,07 a > 0,70
    y el techo es  0,619, el acuerdo entre dos humanos

## 6 · Lo que NO hay que hacer

**No calibrar la escala con la estatura de los tiradores.** Ya esta medido en este proyecto que
eso introduce un sesgo sistematico por atleta del ~4 % en cada distancia, en la misma direccion,
en todos sus clips.

**No mover un umbral para que el acierto suba.** El 0,430 de la clase mayoritaria es la linea que
hay que batir con informacion, no con convenios.

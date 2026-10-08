# La PISTA: qué hay en esta carpeta y qué se puede creer de cada cosa

Lo que el usuario pidió, literal:

> *"Lo de la pista no veo ninguna gráfica que mida cómo se desplaza a lo largo de la longitud de la
> pista y el ancho, como una línea... estas gráficas que es una línea y te pone como un trazado de un
> coche de carreras por la pista. Pues quiero algo igual pero por la pista de esgrima, de los dos
> tiradores, dónde empiezan y cómo se va completando EN VÍDEO... Y luego lo de sacar cada keypoint:
> inicio de marcha, final de marcha, y así medimos su longitud de cada marcha... todo trackeado, para
> por ejemplo poder saber la tercera marcha del tirador de la derecha."*

## EL ENTREGABLE, tres cosas

### 1 · `video/trazado_<clip>.mp4` — el vídeo del trazado

Cuatro zonas sincronizadas: el clip con los dos tiradores marcados (verde el de la izquierda, naranja
el de la derecha), el rótulo con cuadro / segundo / posición de cada uno / distancia entre los dos,
el **TRAZADO** (posición a lo largo de la pista contra tiempo, dibujándose) y el **MAPA de la pista**
visto desde arriba con la traza creciendo y los eventos apareciendo conforme ocurren.

Hay **5 clips**, elegidos a propósito para cubrir los tres grados de credibilidad y no para quedar
bien:

| clip | qué ejemplifica | anclaje |
|---|---|---:|
| `AliveIgnorantBarebirdbat` | el mejor caso: pasa la puerta con **los dos** tiradores en 1,60-2,00 m (2 de 78 clips) | 1,13 m |
| `GentleQuerulousElephantseal` | pasa la puerta agrupada, falla con los dos (izq 2,23 m) | **0,74 m** |
| `UglyNiceAplomadofalcon` | ídem (der 1,48 m) | 2,12 m |
| `ForcefulAntiqueAlbacoretuna` | pasa la puerta con **un solo tirador medido**, y los controles físicos independientes dicen que la escala está ~2,4x inflada | 8,83 m |
| `TestyWatchfulBarebirdbat` | **NO pasa** la puerta física: el rótulo lo dice en pantalla | 1,62 m |

Generar: `python 02_validacion/herramienta/video_pista.py <clip> [<clip>...]`

### 2 · `serie_eventos.csv` y `serie_eventos_<clip>.csv` — los eventos como serie consultable

**Una fila por evento**, no por cuadro. **284 filas de 11 clips** en el agregado. La consulta que
pidió el usuario es una línea:

```python
ev = pd.read_csv("serie_eventos.csv")            # 284 filas, 11 clips
ev.query("lado == 1 and tipo == 'marcha' and indice_del_tipo == 3")
# -> 4 filas, una por cada clip que tiene 3 o mas marchas del tirador derecho.
#    En GentleQuerulousElephantseal: c204-207, t 6,80-6,90 s,
#    x 1,323 -> 1,227 m, magnitud 0,124 m, duracion 0,10 s, 1,245 m/s
# Anadiendo  clip == '<nombre>'  se vuelve a una fila. Sigue siendo una linea.
```

**Los agregados no se escriben nunca con lo que se acaba de calcular**: `reune()` los reconstruye
leyendo del disco todos los `*_<clip>.csv` y lo comprueba con dos asertos (filas del agregado igual a
la suma de los por-clip, y clips distintos igual al número de ficheros). Hizo falta porque una pasada
de un clip suelto sobrescribió los tres agregados de 11 clips con uno solo, y el criterio de
aceptación seguía pasando **por la razón equivocada** (§26.12). Ahora una pasada de un clip es
inofensiva por construcción.

`lado` es 0 el de la izquierda de la imagen y 1 el de la derecha. `indice_del_tipo` va 1, 2, 3... por
(lado, tipo) en orden de `cuadro_inicio`, y **es el mismo número que sale en pantalla en el vídeo**.

Columnas de procedencia, que son las que dicen si una fila vale: `pose_fiable` (la pose del rango del
evento está MEDIDA, no interpolada), `origen_max`, `hueco_max`, `hueco_sin_medir` (hueco > 24, donde
el error de la pose inventada no está ni medido), `solapa_fondo`, `tipo_establecido`,
`transversal_creible`, `publicable`, `puerta_dos_lados`, `ancla_fiable`.

**`unidad_magnitud` es una columna y no un comentario**, porque las magnitudes NO están todas en la
misma unidad: `fondo`, `marcha`, `romper`, `alto` y `accion` en metros a lo largo de la pista;
`bote` en metros verticales con la escala de la caja; `brazo` en fracción de torso.

Y los eventos **se solapan entre tipos a propósito**: un `alto` empieza donde acaba una `marcha`.
No son una segmentación exclusiva del asalto, son detecciones independientes con su rango de cuadros.

### 3 · `pista_<clip>.png` — la figura estática de 5 paneles

`a` el mapa con la traza y el tiempo en color · `a2` la transversal medida, aparte y en gris ·
`b` posición contra tiempo con los eventos · `c` distancia entre los dos · `d` separación de pies.

Generar: `python 02_validacion/herramienta/grafica_pista.py <clip> [<clip>...]`

## LOS SIETE TIPOS DE EVENTO, y qué se puede creer de cada uno

| tipo | criterio | magnitud p50 | estado |
|---|---|---:|---|
| `fondo` | `d_pies` sube ≥ **0,50 x la guardia del clip** en 8 cuadros **y** el pie de delante avanza ≥ 0,36 x la guardia | **0,602 m** | árbitro contra `marcha`, ver abajo |
| `marcha` / `romper` | un pie se mueve a > 1,25 m/s con el otro por debajo de la mitad, **sobre la longitudinal suavizada**, y se empareja con la zancada del otro pie | **0,397 m**, duración 0,37 s | criterio nuevo (§26.4), y el **41 % solapa un fondo**: filtrar `solapa_fondo == False` |
| `alto` | recorre ≥ 0,25 m en 8 cuadros y luego ≤ el umbral calibrado con el ruido del propio tirador | 0,30-0,66 m | umbral **adaptativo**, mirar `umbral_quieto_m`; significa "baja de 0,3-0,7 m/s", no "inmóvil" |
| `bote` | las dos plantas se separan del suelo a la vez, con control BALISTICO de tiempo de vuelo | — | **NO ESTABLECIDO**: **0 eventos**, y hay un aserto que lo garantiza |
| `brazo` | el eje `hombro -> raíz de la mano` pasa de 1,15 veces su mediana; `completa` si llega a 0,90 del p98 del clip | 0,29-0,50 x torso | eje **corregido** respecto a §23.8 |
| `accion` | inicio y fin en la misma fila (`cuadro_inicio`, `cuadro_fin`) | — | — |

**El `bote` no está establecido y no se debe usar.** El suelo de ruido de la señal sale de su cola
negativa (un pie no puede estar por debajo del suelo, así que todo lo negativo es ruido puro): vale
`|p05|` = **0,0198** del alto de la caja, o sea **3,5 cm** en un tirador de 1,75 m. El umbral a 3x son
**10,4 cm de despegue simultáneo de las dos plantas**, y una balestra o un appel de florete despegan
**5-8 cm**: por debajo. Sólo **2 cuadros de 2.782** pasan el umbral. Bajarlo sería llamar bote al
temblor del keypoint del pie.

## LAS TRES COSAS QUE NO SE PUEDEN CREER, y salen en pantalla

1. **La coordenada transversal NO es creíble.** Recorrido p50 **0,87 m** contra el control declarado
   de < 0,60 m, y falla en **8 de 10** clips (§23.7). Es la dirección de profundidad de la cámara y un
   pie en el aire se proyecta desplazado. Por eso en el mapa cada tirador va en un **carril fijo**,
   que es dibujo y va rotulado como dibujo, y la transversal medida va aparte en gris. Todas las
   filas del CSV llevan `transversal_creible = False`.

2. **La posición absoluta en la pista no está identificada** (§23.9). El origen longitudinal es una
   marca transversal sin identificar. El mapa se ancla con una **convención declarada** (el punto
   medio de los dos tiradores, mediana del clip, se toma como centro de la pista) y se da la
   **deriva medida** de ese anclaje: 0,74 · 1,13 · 1,62 · 2,12 · **8,83** m en los 5 clips del vídeo.
   Con deriva por encima de 1,00 m el rótulo dice ANCLAJE NO FIABLE. Los DESPLAZAMIENTOS están en
   metros; "en qué punto de la pista" no.

3. **La puerta de la homografía es más débil de lo que su propia cabecera declara** (§26.1). Dice "la
   estatura reproyectada de los DOS tiradores cae en 1,60-2,00 m" y el código compara la mediana
   AGRUPADA de los dos. Aplicada por tirador, de los 78 clips pasan **2 (2,6 %)** y no 10 (12,8 %).
   Las dos cifras están en `homografia.csv` (`publicable` y la columna derivada
   `puerta_dos_lados` de `serie_eventos.csv`).

## `diagnostico/` — NO es entregable

`homog_*.png` y `censo_*.png` son **diagnóstico interno**: llevan la máscara de rótulos en magenta
encima del marcador del clip, la región de pista en verde y los residuos escritos. Sirven para saber
POR QUÉ una homografía sale mal, que es imprescindible para trabajar, y no sirven para enseñar nada.
El usuario ya lo dijo: *"mira los output de imagen, no tiene sentido, es poco, pones borroso el
marcador o el fondo un poco extraño"*. Tenía razón.

## `verifica/` — la verificación a ojo

`apoyo_<clip>.png` (los pies detectados y la retícula métrica reproyectada), `ev_<tipo>_*.png` (los
detectados) y `fn_<tipo>_*.png` (los **candidatos no detectados**). Un tipo sin candidatos no
detectados mirados no está verificado: mirando sólo los aciertos, un detector que dispara una vez y
acierta parece perfecto.

Generar: `python 02_validacion/herramienta/verifica_pista.py <clip> [<clip>...]`

## Intérprete

`<scratchpad>/fz/.venv/Scripts/python.exe`. Es el único con cv2 + numpy + pandas + **matplotlib** +
scipy a la vez. `<scratchpad>/cuda/.venv` **no tiene matplotlib** (comprobado: `ModuleNotFoundError`),
así que ni la figura ni el vídeo salen con él.

## `alto` NO es una parada de esgrima, y la parada NO está implementada

Se llamaba `parada` y el nombre engañaba. En **esgrima** una parada es la acción defensiva de desviar
la hoja del atacante con la propia, y transfiere la prioridad a la **respuesta**: es un evento de
**arma**. Lo que este detector emite es que el tirador **deja de desplazarse**, que es un evento de
**pies** y significa «baja de 0,3-0,7 m/s», no «inmóvil».

**La parada en el sentido de esgrima no está implementada y no se puede con este material**: el paso
de prioridad se cerró con 0 paradas anotadas y midió que el contacto de hojas no es verificable a ojo
en este corpus. Quien lea «71 altos» no debe entender que hay paradas-respuesta detectadas.

## El RECALL, medido: 90 % de las zancadas

El usuario dijo «muy pocas marchas, unas 10 por tirador». Contando **pasos** (`marcha + romper`,
porque un paso hacia atrás también es un paso) salen **9,1 por tirador y clip** sobre los 11 clips.
La cifra de «5,7» de antes contaba sólo `marcha`.

Y se contaron a ojo TODAS las zancadas de tres tiradores, con el convenio escrito antes en
`CONVENIO_CONTEO_PASOS.md` y sobre fotogramas ESTABILIZADOS (sin usar la serie métrica, que sería
circular): **49 zancadas contadas, 44 cubiertas por una detección = 90 % de recall**. El 10 % que se
pierde son traslados LENTOS, que es lo que cabe esperar de un umbral de velocidad.

**Desfase: mediana +0,5 cuadros (+17 ms).** No hay retardo sistemático que corregir; lo grueso es el
agrupamiento, porque una detección suele cubrir 2-3 zancadas.

## La transversal: la FORMA se pinta, la ESCALA no es creíble

La versión anterior pintaba cada tirador en un carril fijo, o sea una línea recta perfecta, y eso es
tan falso como pintar ruido. Medido en los 22 tiradores-clip: la **excursión** transversal es
**9 a 77 veces** el ruido por cuadro (p50 19,8), así que **la forma es señal en 22 de 22**. Se pinta
filtrada con mediana móvil de 9 cuadros, que conserva el 97,8 % de la excursión, con una banda de
±3 veces el ruido medido, y las dos curvas (cruda y filtrada) van en el panel `a2`.

Lo que sigue mal es la **escala**: el control de recorrido <= 0,60 m lo pasan **11 de 22**, y quitar
la dependencia lineal de `x` lo lleva a 14 de 22. Quedan recorridos de hasta **1,87 m en una pista de
1,50 m**, que es imposible. El rótulo del vídeo lo dice con su cifra.

## El eje del mapa NO está invertido, y ahora está declarado

Convenio, con aserto en `posicion.apoyos`: **la x crece hacia la derecha de la imagen** y el `lado 0`
es el tirador de la izquierda. Comprobado: `lado 0` a la izquierda en la imagen en 11 de 11 clips, con
la x métrica menor en 11 de 11, y **0 cruces en 2.896 cuadros** (dos floretistas no pueden cruzarse,
así que es un control físico). Columna `cruces_imposibles`.

## Los vídeos: contador y cámara lenta

**11 vídeos** a velocidad normal y **4 en cámara lenta** a un cuarto (7,5 fps), con el más activo
(`ForcefulAntiqueAlbacoretuna`, 1,75 pasos/s/tirador) y el menos (`UglyNiceAplomadofalcon`, 0,56)
puestos a propósito uno al lado del otro.

El **contador** por tirador y tipo salta en el cuadro de `inicio` del evento y el número es **el
mismo `indice_del_tipo` del CSV**. Hay destello con el nombre y la magnitud, y las marcas de `marcha`
y `romper` van numeradas en el mapa.

Generar los lentos: `python video_pista.py <clip> --lenta`

## Por qué `bote` da 0 y hay un aserto que lo garantiza

El umbral es adaptativo (`max(0,05 ; 3 x ruido de ese tirador)`), así que en un clip con poco ruido
puede bajar a 8,5 cm. Lo que cierra el asunto es la **física**: si las dos plantas se separan `h` del
suelo, el tiempo de vuelo está fijado por `t = 2 sqrt(2h/g)` — 5 cm son 6,1 cuadros a 30 fps y 8,5 cm
son 7,9. Se exige que la duración llegue a la mitad de ese vuelo, y un **aserto** en
`posicion.eventos` aborta si una fila de `bote` lo incumple. Una balestra de florete despega 5-8 cm,
que es del orden del ruido de 3,5 cm, así que este evento no se puede afirmar con esta resolución.

## `marcha` y `fondo` son EXCLUYENTES: hay un árbitro y un aserto

El usuario lo vio en cámara lenta: *«pasa a marcha, a fondo y luego marcha otra vez. Detecta ahí un
fondo que no es fondo»*. Había **73 solapes** de tipos distintos del mismo tirador, y
`ForcefulAntiqueAlbacoretuna` lado 0 daba `MFMFMFMFMMFMRFMRRMRMFMFMFMFMFMFM`.

**Dos causas y dos arreglos:**

1. **El umbral del fondo estaba en metros absolutos**, así que en un clip con la escala inflada lo
   cruzaba un paso normal. Correlación medida entre el factor de escala del clip y los fondos por
   segundo: **r = 0,83**. Ahora se normaliza por la **guardia del propio clip**, con el coeficiente
   elegido para reproducir los números declarados (0,50 x 0,70 = 0,35 m). Es **simétrico**: aprieta
   donde la escala está inflada y afloja donde está comprimida, y hereda la calidad de la medida de
   la guardia: de los 17 lados con guardia en el CSV, **10 caen en los 0,50-0,90 m** de un floretista
   y 7 no (0,265 a 1,371 m). Columna `guardia_plausible`.

2. **El discriminador es EL PIE DE ATRÁS, no la magnitud** (que se pisa: 0,335 contra 0,602 m). En
   una marcha el pie de atrás SIGUE; en un fondo se queda CLAVADO. Dos rasgos, medidos en la **fase
   de extensión** (hasta el pico de separación, no hasta el final de la ventana, que incluye la
   recuperación y corrompe la medida): `razon_pie_atras` (frontera 0,45) y `pico_dpies_guardia`
   (frontera 1,25).

Cuando dos eventos de tipos distintos del mismo tirador se solapan, el árbitro **decide uno**. Si
decide fondo, conserva los pasos que no solapan al fondo. Resultado: **73 solapes → 0**, con un
**ASERTO** que aborta si vuelve a ocurrir. `Forceful` lado 0 queda `MMMMMMMRFRRMRMMMMMMM`.

**Matriz de confusión contra 6 gestos anotados a ojo: 5 de 6 = 83 %, con `marcha -> fondo` = 0.** Ya
no llama fondo a ninguna marcha. El único fallo es un fondo perdido en la frontera (`razon` 0,48
contra 0,45).

Columnas para contradecirlo: `arbitrado`, `razon_pie_atras`, `pico_dpies_guardia`,
`margen_arbitraje` (p50 0,286; sólo **2 de 48** por debajo de 0,05, o sea en la frontera).

**Con umbrales de 0,55 y 1,30 el acierto sube a 6 de 6, y NO se adoptan**: es ajustar a seis
ejemplos. Se usa el declarado y se publican los dos números.

## Clips SIN homografía: sólo `accion` y `brazo`, y marcados

Tener la pose de 308 puntos **no basta**: `marcha`, `romper`, `fondo` y `alto` tienen umbrales en
metros y necesitan además la homografía. De 10 clips nuevos con pose, **6 no la tienen**.

Lo que sí se puede medir sin ella:

 - **`accion`** en **altos de caja por segundo** (umbral 0,29, que es `0,50 m/s / 1,75 m`, o sea el
   mismo criterio en otra unidad). Columna `via_accion` = `metros` o `alto_de_caja`.
 - **`brazo`** en fracción de torso, que **nunca necesitó la homografía**: sólo la usaba para escribir
   `pos_m`.

En esos clips `pasos_por_s_util_y_tirador`, `cruces_imposibles` y `ancla_deriva_m` van a **`None` y no
a cero**, con la columna `sin_homografia` para distinguir «no medido» de «medido y salió cero». Esa
distinción es la que este proyecto lleva pagando todo el día.

## Los dos ÁRBITROS entregados como tirador

`PowerfulAllLarva` lado 0 e `IdolizedFancyChickadee` lado 1 son árbitros, verificados a ojo. Son 2 de
33 clips anotados (6,1 %) y es el defecto sin resolver más recurrente.

**Sus eventos SE EMITEN todos**, con la marca en las columnas `lado_valido` y `lado_no_valido_por`.
Nunca en la ausencia de fila: si se filtraran, su cero sería indistinguible de «no medido», y lo que
se quiere saber del árbitro es precisamente **lo que no tiene**.

**Y el criterio `fondo + brazo` por segundo NO los separa** (ver §31): 0,100 y **0,524**, y el 0,524
está por encima del p10 de los tiradores reales. Además `fondo` es 0 por construcción en esos clips
porque no tienen homografía, así que sólo se ejecuta la mitad del criterio.

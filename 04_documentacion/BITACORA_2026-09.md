> **2026-10-08 — este es el UNICO proyecto.** El pipeline paralelo de `Desktop\FENCINGLY.AI`
> quedo archivado en `99_historico/` y lo util se trajo aqui (etapa 11, scouting). El estado de hoy,
> con todo lo medido a escala, esta en `04_documentacion/informes/ESTADO_2026-10-08.md`.

# FENCINGLY — reextracción limpia con Sapiens 2

**Plan del 2026-09-10.** Un solo objetivo: **poses correctas en todos los fotogramas, con cero
intrusos**. La web y el entrenamiento quedan aparcados por decisión tuya.

---

## 1. La corrección que cambia el orden del plan

Propones Sapiens 2 para arreglar el problema del árbitro. **Sapiens 2 no lo arregla**, y conviene
saberlo antes de gastar una semana. Su propia tarjeta de modelo lo dice:

> *Pose is top-down — it requires bounding boxes from a person detector.*

O sea que **Sapiens 2 consume cajas, no las elige**. Si le pasas la caja del árbitro, devuelve un
esqueleto de 308 puntos precioso **del árbitro**. Mejora la mitad del problema —la calidad de los
puntos— y **no toca la otra mitad**, que es la selección.

El orden es: **primero la selección, después Sapiens 2.** Al revés produciría datos peores, porque
un intruso con 308 puntos parece más fiable que un intruso con 17.

### RETRACTADO el 2026-09-10: mis dos pruebas de contaminación eran un error mío

Este documento afirmaba antes que había medido esqueletos sobre **paredes vacías** y **oficiales de
traje** seleccionados como tiradores. **Las dos eran mi propio error de índice y las retiro.**

**El error:** el NPZ no contiene el clip entero, sino una **ventana** que empieza en `meta_json["f0"]`.
La fila `k` del NPZ le corresponde el cuadro **`f0 + k`** del vídeo. Yo pintaba la fila `k` sobre el
cuadro `k`, y con `f0 = 175` eso son casi **6 segundos de desfase**.

Verificado pintando las dos versiones lado a lado:

| clip | `f0` | con mi desfase | con el desfase corregido |
|---|---:|---|---|
| `SoreEasygoingEmperorshrimp` | 175 | esqueletos sobre pared vacía | **los dos tiradores, en guardia** |
| `AltruisticIllfatedDobermanpinscher` | 197 | dos señores de traje en la grada | **los dos tiradores, cajas ajustadas** |

**Las poses estaban bien.** Lo que estaba mal era mi manera de mirarlas.

**Lo que cae con ello:**

- el «38,25 % de mis clips limpios con blancura < 0,05»: la medida de blancura leía el fotograma
  equivocado en todo clip con `f0 > 0`. **Inválida.**
- la validación de la etapa A1 con `seg` de esta mañana: sus «negativos» eran fotogramas
  equivocados. **Inválida** (los positivos, con `f0 = 0`, sí valen).
- el diagnóstico de «solape 0,000 entre YOLO y el algoritmo viejo»: mismo error. **Inválido.**

**Lo que se mantiene**, porque no toca el vídeo y trabaja solo con el NPZ:

- la auditoría de contaminación geométrica: 88,36 % limpios, razón de alturas 0,99 %
- la auditoría de calidad de series: cambios de identidad **0,43 %**, teletransportes **0,25 % de
  los cuadros**, y **fracción útil mediana 1,000** — que ahora encaja mucho mejor: dice que en el
  clip típico están los dos tiradores en todos los cuadros
- todas las medidas de Sapiens 2, que usaron recortes directos o clips con `f0 = 0`

**Y lo que queda en pie sin demostrar:** el usuario dice haber encontrado árbitros y tiradores de
otras pistas revisando las poses, y no hay motivo para dudarlo. Pero **yo no lo he demostrado**, y
las pruebas que creía tener no valían. Encontrar casos reales es ahora la tarea del validador, con el
desfase `f0` aplicado.

## 2. Lo bueno de Sapiens 2, verificado hoy

Está **abierto**, sin lista de autorización, y trae cinco familias:

| familia | tamaños | para qué en esgrima |
|---|---|---|
| **pose** | 0.4b · 0.8b · 1b · 5b | 308 puntos con la **cadena del pulgar completa** |
| **seg** | 0.4b–5b | partes del cuerpo: «hay una persona aquí» pasa a ser medida |
| **pointmap** | 0.4b–5b | **coordenadas 3D por píxel**, no profundidad relativa |
| normal | 0.4b–5b | orientación de superficies |
| matting | 1b | recorte fino de silueta |

### El reparto real de los 308 puntos

Sacado del `keypoints308.py` oficial, no del resumen del README:

| grupo | puntos | decisión |
|---|---:|---|
| **manos** | **21 por mano** | root + pulgar(4) + índice(4) + medio(4) + anular(4) + meñique(4) |
| **pies** | 8 | tobillo + dedo gordo + dedo pequeño + talón, por lado |
| cuerpo | hombros, codos, muñecas, caderas, rodillas… | se conserva |
| **cara** | ~210 | **se descarta: los tiradores llevan máscara** |

Descartar la cara **ahorra el 61 % del almacenamiento** y no pierde nada utilizable.

Los índices que tu idea del cono necesita, verificados:

```
right_wrist = 10    right_hand_root = 112    right_thumb1..4 = 113, 114, 115, 116
left_wrist  =  9    left_hand_root  =  91    left_thumb1..4  =  92,  93,  94,  95
```

### RETRACTADO el 2026-09-10 (segunda vez): esos seis índices son FALSOS

**Los puntos 91-95 y 112-116 caen sobre la careta, no sobre la mano.** Este bloque decía
«verificados» y no lo estaban: no hay ningún `keypoints308.py` en la máquina y el mapa se dio por
bueno sin comprobarlo. Lo que lo delató fueron dos medidas imposibles:

| medida con los índices del plan | valor | por qué es imposible |
|---|---:|---|
| largo de la «cadena del pulgar» `91→95` | **1,4-2,8 px** | una mano no mide 2 px |
| distancia `hand_root`-`wrist` | **97-212 px = 47 % del alto de caja** | la muñeca no está a medio cuerpo de la mano |

Y confirmado pintándolos: los dos grupos aterrizan **encima de la máscara**.

**El mapa medido**, sacado del `flip_pairs` del propio `config.json` del modelo (que trae
`[21,42], [22,43], … [41,62]`: **21 pares con desplazamiento fijo de 21**, o sea una mano contra la
otra) y comprobado por constancia de hueso y a ojo:

```
 0 nariz   1-2 ojos   3-4 orejas   5-6 hombros   7-8 codos   9-10 CADERAS
11-12 rodillas   13-14 tobillos   15-17 pie A   18-20 pie B
21-41 MANO A  (root 21, pulgar 22-25, luego índice/medio/anular/meñique de 4 en 4)
42-62 MANO B  (root 42, pulgar 43-46, igual)
63 en adelante: cara
```

Comprobaciones: anchura de hombros `5-6` = 18,7-21,5 % del alto contra `9-10` = 9,5-11,7 %, o sea que
9 y 10 son **caderas**; el hueso `codo 8 → mano 21` tiene cv 0,087-0,200 y el cruce `7→21` llega a cv
0,714; y el grupo `root+1..root+4` es el más separado angularmente de los otros cuatro dedos en el
**78-93 %** de los cuadros, que es lo que se espera del pulgar.

**Y ojo con la trampa que hizo que esto pasara desapercibido:** los puntos de la careta tienen **más**
confianza que los de la mano (fracción con `sc ≥ 0,3` de **0,998-1,000** contra 0,648-0,987). El
«confianza 0,640 al 100 % de los cuadros» que figuraba como prueba de la cadena del pulgar estaba
midiendo la cara. Verificado en `01_extraccion/codigo/e0_mapa_indices.py`.

**No hay muñeca en el bloque de cuerpo**: el bloque 0-14 es COCO-17 **sin** las dos muñecas, porque la
muñeca *es* el root de la mano. Así que `wrist → hand_root` no existe como segmento y su sustituto
legítimo es `codo → root`.

## 2a. Cuatro trampas de Sapiens 2, medidas. Léelas antes de tocar el código

**1. El convenio de caja es `xywh`, NO `xyxy`.** Es la trampa más cara porque **el modelo parece roto
y no lo está**:

| invocación | confianza en tronco y brazos |
|---|---:|
| `xyxy` | **0,104** |
| **`xywh`** | **0,883** |
| cuadro completo sin caja | 0,051 |

Factor **8,5**. Con `xyxy` salen el 3 % de los puntos con confianza y parece que Sapiens no sirve
para esgrima. Con `xywh` salen el **87 %**.

**2. Es top-down de verdad.** Pasarle el cuadro completo da 0,051: hay que recortar. Eso significa
una pasada **por persona y por cuadro**, y es lo que multiplica el coste por dos en un asalto.

**3. No fuerces `attn_implementation="sdpa"`: es 7,5× más lento** (588,9 contra 78,4 ms). El que trae
por defecto es el bueno.

**4. `torch.compile` no funciona en esta máquina**: Windows sin `triton`. No perder tiempo ahí.

Y una que no es trampa sino aviso: la puntuación **no está acotada a [0,1]** — se vieron valores de
hasta **1,304**. Cualquier umbral que asuma un máximo de 1 está mal puesto.

## 2b. Las otras cuatro cabezas, leída la documentación oficial

### `seg` — segmentación por partes del cuerpo, **29 clases**

Y esto **resuelve la etapa A mejor que cualquier heurística mía**. La lista de clases incluye:

```
 0 Background      6 Left_Hand        15 Right_Hand      22 Torso
 1 Apparel         7 Left_Lower_Arm   16 Right_Lower_Arm 23 Upper_Clothing
 5 Left_Foot      11 Left_Upper_Arm   20 Right_Upper_Arm  9 Left_Shoe
```

Con eso, **«¿hay alguien en esta caja?» pasa de ser una inferencia a ser una medida**: si la máscara
no devuelve píxeles de torso ni de brazo, no hay nadie. Ayer inventé una heurística de *blancura del
torso* en HSV que funcionaba (0,000 sobre pared vacía, 0,72–0,96 sobre tirador) pero era un apaño;
esto es lo correcto.

Y da además `Left_Hand` y `Right_Hand` **como región**, que es justo lo que el cono necesita para
anclarse: la cazoleta está pegada a la máscara de la mano.

### `pointmap` — 3D por píxel

*«Para cada píxel, el modelo predice su posición (x, y, z) en el sistema de coordenadas de la cámara.
En conjunto forman una nube de puntos 3D densa alineada con la imagen.»*

Es lo que buscabas para ubicar a la gente en la pista, y **no es lo mismo que la profundidad
monocular** que ya cerramos con 46–51 % de error para escala: eso daba orden, esto da estructura 3D.
Aun así hay que medirlo por separado antes de creerlo, porque «coordenadas de cámara» no es «metros
sobre la pista».

Nota práctica: la visualización necesita `open3d`, que no viene instalado.

### `normal` — normales de superficie

Vectores unitarios de 3 canales. Y su documentación trae un consejo que aplica a todo: *«para mejores
resultados, ejecuta primero la segmentación por partes y pasa la máscara de primer plano para filtrar
los píxeles de fondo»*. O sea que **`seg` va antes que `normal` y que `pointmap`**, no en paralelo.

### `matting` — silueta fina

Solo hay variante de 1b. Recorte de contorno; útil para el corredor de interacción, no para poses.

---

## 3. La cifra que manda, ya MEDIDA en la 3060

Medido el 2026-09-10 en la propia máquina (RTX 3060, `cuda:0`, fp16), sobre un clip de 720p con dos
tiradores. No es una estimación:

| modelo | ms / cuadro de vídeo | puntos | proyección a 12 M cuadros |
|---|---:|---:|---|
| **YOLO11m-pose** (el actual, imgsz 960) | **24,6** | 17 | **3,4 días** |
| **Sapiens2-pose-0.4b** (1024×768) | **624** | 308 | **87 días** |

**Sapiens 2 es 25× más lento que el modelo actual.** Y el 0.4b, no el 1b.

### Dos correcciones a lo que yo mismo había supuesto

**Los lotes casi no ayudan.** Yo proyecté 410 h suponiendo que agrupar escalaría; es falso:

| lote | ms/pasada | ms/cuadro |
|---:|---:|---:|
| 1 | 340,0 | 680,1 |
| 2 | 317,4 | 634,9 |
| **4** | **312,1** | **624,2** |
| 8 | 313,5 | 627,0 |

Del lote 1 al 4 se gana un **8 %** y ahí se acaba: la GPU ya se satura con **una sola** pasada de
1024×768. No hay palanca de rendimiento por ese lado.

**Y la VRAM no es el límite:** pico de **1,52 GB de 12**. Cabría el 1b (6 GB de pesos) de sobra —
pero sería aún más lento, así que el límite es el cómputo, no la memoria.

Dato de contexto de la documentación oficial: su propio pipeline usa `JOBS_PER_GPU=3` sobre
`GPU_IDS 0–7`, o sea que **está diseñado para 8 GPUs**. Una 3060 está muy lejos de ese punto.

### La optimización que cambia el plan: **7,8× bajando la resolución**

Los recortes de tirador miden **144×155 px de mediana** y le estábamos dando **1024×768**, o sea
sobremuestrear ×6,6. Medido bajando la entrada, y juzgado por **desplazamiento de los puntos** contra
la referencia de 1024, no por confianza:

| resolución | ms/pasada | acelera | err. tronco | err. pulgar | % del alto de caja |
|---|---:|---:|---:|---:|---:|
| 1024×768 | 325,1 | 1,0× | referencia | — | — |
| 768×576 | 170,3 | 1,9× | 0,67 px | 1,16 px | 0,22 / 0,36 % |
| 640×480 | 113,3 | 2,9× | 1,05 px | 1,70 px | 0,36 / 0,56 % |
| 512×384 | 70,8 | 4,6× | 1,66 px | 1,99 px | 0,55 / 0,66 % |
| **384×288** | **41,9** | **7,8×** | **2,26 px** | **3,74 px** | 0,88 / 1,07 % |
| 256×192 | 28,5 | 11,4× | 7,45 px | 5,90 px | **2,64 %** ✗ |

Límite declarado antes de medir: error mediano por debajo del **2 % del alto de caja**. **384×288 lo
cumple con holgura** y es 7,8× más rápido.

**Y por qué esto se midió por posición y no por confianza:** primero use el criterio «fracción de
puntos con confianza ≥ 0,3» y **todas las resoluciones lo pasaban al 100 %**, incluida 256×192 donde
el tronco cae de 0,884 a 0,410. Un criterio que aprueba todo no mide nada. Peor: **la confianza baja
un 32 % en 384×288 mientras la posición se mueve 2 px**, así que juzgar por confianza habría hecho
descartar la opción buena. Confianza y exactitud divergen.

Salvedad honesta: es **acuerdo con la mejor referencia disponible**, no verdad-terreno. Si 1024 se
equivoca, 384 hereda el error sin que esta medida lo vea.

### Las otras tres palancas, medidas

| palanca | resultado |
|---|---|
| **`sdpa` forzado** | **588,9 ms contra 78,4: 7,5× MÁS LENTO.** No forzarlo nunca |
| `eager` | 124,1 ms: también peor que el que viene por defecto |
| `bf16` en vez de `fp16` | 84,9 contra 78,4: sin diferencia útil |
| **`torch.compile`** | **no se puede en Windows**: falta `triton` |
| lotes | del 1 al 4 se gana un **8 %** y ahí acaba: la GPU se satura con una sola pasada |

### La comparación que decide

| modelo | ms/cuadro | puntos | mano | `seg`+3D | 12 M cuadros | vs. actual |
|---|---:|---:|---|---|---|---|
| YOLO11m-pose (actual) | 24,6 | 17 | no | no | 3,4 días | 1,0× |
| Sapiens 1024×768 | 650 | 308 | sí | sí | 90 días | 26× |
| **Sapiens 384×288** | **84** | **308** | **sí** | **sí** | **11,6 días** | **3,4×** |

La optimización lleva Sapiens de **26× a 3,4×** el coste actual. Buscar alternativas de 133 puntos
(RTMW, DWPose) tenía sentido con 26×; con 3,4× la ganancia sería como mucho otro 2× **a cambio de
perder `seg` y `pointmap`**. No medidas aquí: `onnxruntime`, `mmpose` y `mediapipe` no están
instalados.

### La consecuencia sobre el diseño

Los dos modelos hacen cosas distintas y por eso conviven:

- **YOLO es de una pasada por cuadro** y detecta a **todas** las personas. Medido: **8 personas por
  cuadro** de media. Eso *es* el problema de la etapa A cuantificado — hay que elegir 2 de 8.
- **Sapiens 2 es top-down**: hay que darle la caja, y cuesta una pasada por persona.

Así que **YOLO en todos los cuadros** (detección y selección) y **Sapiens 2 solo donde sus puntos de
más se ganen el coste**:

| alcance de Sapiens 2 | cuadros por clip | coste |
|---|---:|---|
| todos | 342 | **87 días** |
| ±1 s alrededor del tocado | ~60 | **15 días** |
| ±0,5 s alrededor del tocado | ~30 | **7,6 días** |
| solo el instante del tocado ±3 cuadros | 7 | **1,8 días** |

Esa elección es tuya, y depende de si la mano hace falta en todo el asalto o solo en la acción.

---

## 4. Las seis etapas, en orden de dependencia

### Etapa A · Selección con cero intrusos ← **el problema de verdad**

Nada más empieza hasta que esto funcione, porque todo lo demás hereda sus errores.

Lo que ya sé que **no** basta: nueve detectores de coherencia geométrica. Un esqueleto sobre una
pared es perfectamente coherente — no salta, no cambia de identidad, sus huesos no varían, está
dentro de la banda. **Medían que la serie es estable, no que hubiera alguien.**

Lo que sí discrimina, con su cifra:

| señal | medido |
|---|---|
| **blancura del torso** (el uniforme es blanco) | 0,72–0,96 con tirador real · **0,000 sobre pared vacía** |
| razón de alturas entre los dos elegidos | preciso, 0,99 % del corpus, **2 de 2 inspeccionados eran contaminación real** |
| **`seg` de Sapiens 2** | por probar: da máscara de partes del cuerpo |

La etapa A entrega **una puerta que rechaza**, no columnas informativas, y su criterio se declara
antes de mirar.

### Etapa B · Los puntos de Sapiens 2

Solo sobre cajas que pasaron A. Empieza por **0.4b** (1,7 GB) para medir rendimiento; se sube a 1b
(6 GB) solo si la cifra lo permite.

### Etapa C · Interpolación: siempre dos poses completas

Tu idea, correcta **para huecos reales**, con tres condiciones que salen de lo ya medido:

1. **Solo entre dos extremos buenos.** Un hueco al principio o al final no tiene dos extremos: se
   declara, no se rellena.
2. **Solo si los dos extremos pasan la puerta A.** Interpolar entre dos poses falsas produce más
   poses falsas, y encima suaves y creíbles — sería peor que el hueco.
3. **Todo cuadro interpolado va marcado.** Sin la marca, nadie puede volver a distinguir lo medido
   de lo inventado.

### Etapa D · La herramienta de validación visual ← **en paralelo con A, no después**

Por una razón concreta: **ayer mis nueve métricas dijeron «limpio» y una sola imagen lo desmintió.**
Tiene que:

- pintar los dos esqueletos sobre el fotograma con los números medidos al lado
- llevar siempre una carpeta de **clips que el algoritmo declara limpios**, porque si solo se enseña
  lo que marca, nadie puede encontrar los **falsos negativos**
- mezclar **controles ciegos** en cualquier muestra, para medir si quien verifica es fiable

### Etapa E · El cono del florete

Con Sapiens 2 pasa a ser construible: eje del pulgar → línea de la hoja → cazoleta perpendicular →
cono 3D con vértice en el pulgar → mapa de calor dentro del cono → intersección de los dos conos =
probable parada.

Lo ya medido que condiciona el diseño:

- el fuerte mide **0,64 px a 720p** y 0,96 a 1080p: por debajo de un píxel. **No es indetectable**
  —el contraste deja una cresta— pero **no hay región que segmentar**.
- la cazoleta sí: **19 px a 720p, 29 a 1080p**.
- SAM3 con `the hilt of a sword` coloca a menos de un diámetro en **0,33** de los recortes contra
  **0,04** del señuelo verbal. Sirve para **proponer**, no para medir.
- **la confianza no sirve para filtrar**: aciertos 0,238, fallos 0,253.

Así que el cono se construye **desde la geometría de la mano** y la imagen solo **confirma** dentro
del cono. Nunca al revés.

### Etapa E, MEDIDA el 2026-09-10. Y el orden se da la vuelta

Código en `01_extraccion/codigo/e0`…`e8`, imágenes y vídeo en `02_validacion/muestras/cono/`.

**1. El eje del pulgar NO sirve.** No por la idea, que es buena, sino por la escala: el segmento
`root → pulgar4` mide **2,4-4,8 px**, o sea que **1 px de error son 12-23°**. Salta una mediana de
**11,4-19,8° entre cuadros consecutivos** y **pasa de 40° en el 10-26 % de los cuadros**. Y dos
estimadores de esa misma recta (dos puntos contra el centroide del pulgar) discrepan una **mediana de
67-134°**: la recta del pulgar no tiene dirección consistente. La razón lag2/lag1 vale **1,2-1,4**,
o sea ruido, mientras el antebrazo da 1,5-1,8, o sea movimiento real.

**El sustituto que sí sirve es `codo → root`**: 49-94 px de palanca, salto mediano **0,9-2,1°**, p95
de 5,1-8,2°, y pasa de 40° en el **0-0,3 %**.

**2. La apertura, acotada por los dos lados.** Suelo (temblor del eje, p95) **5,1-8,2°**; techo de
utilidad (que el cono sea más estrecho que los hombros del rival a 90 cm) **10,2-12,2°**. Queda hueco,
pero es de **1,26 a 2,02×**, no más.

**3. `matting` SÍ capta la hoja, y la pista de mirar el alfa bajo es del usuario.** Es el primer
método que la ve después de que fallaran nueve clásicos, SAM2 y el rastreo de vídeo. Dos correcciones
a la pista, las dos medidas: el alfa del fondo **no es cero** (mediana 3,1e-4 a 5,2e-4, y **cambia el
doble de un recorte a otro**, así que un umbral fijo no vale), y la hoja **no está en el alfa
bajísimo** sino entre **0,02 y 0,5** — por eso un test con alfa ≥ 0,5 daba cero. Umbral bueno:
**percentil 99,5 del alfa fuera del cuerpo, del propio recorte**.

| | 1080p | 720p |
|---|---:|---:|
| recortes con recta pegada a la mano | **42,6 %** (26/61) | **1,6 %** (1/64) |
| de ellas, delante de la mano (no es el brazo) | **92,3 %** | 100 % (n=1) |
| \|ángulo hoja − eje antebrazo\| mediana | **10,4°** | — |
| control del azar declarado | 45° | 45° |
| control del cono girado: px dentro, sitio / girado | **24,0×** | — |

**A 720p la hoja no se detecta** (1,6 %), y encaja con su ancho sub-píxel de 0,64 px. **A 1080p sí.**

**4. Y esto invierte el diseño.** El ángulo medido entre la hoja y el eje pide una semiapertura de
**24,8-29,7°** (p90), que es **2,4-2,8× el techo de utilidad**. O sea: **el cono como prior no
localiza la punta**. Pero ya no hace falta que la localice, porque la hoja **se mide**. El orden
correcto pasa a ser *detectar la hoja en el alfa* y usar el cono solo como **ventana de búsqueda**.

**5. La intersección es medible y reproducible**, y se deja como columna cruda **sin evaluarla contra
la etiqueta de tocado**, que sería circular. Repitiendo todo con poses a 512×384 en vez de 384×288:
el booleano «se cruzan» coincide en el **97,3-97,4 %**, `peso_max` da pearson **0,942-0,962** y
`d_ejes` pearson **0,997-1,000**. Parpadea en el 3,5-4,9 % de los pares.

**6. Un aviso sobre la mano armada.** Se decide con un solo criterio geométrico (la mano con x más
cerca del rival), acertó en los **4 de 4** tiradores comprobados a ojo, y **los dos criterios
independientes que probé para corroborarlo fallaron**, cada uno en un tirador distinto: «la más
extendida» falla porque en guardia el brazo de atrás también va extendido, y «la que más apunta al
rival» falla porque en pleno fondo los dos brazos se alinean con el ataque. Cuatro tiradores no
validan un criterio.

### Etapa F · Prioridad y las luces fantasma

Tu razonamiento es el más prometedor del documento y encaja con lo medido: el techo del **73,6 %**
viene de que el modelo aprende el **atajo del orden de luces** (acierta el 63,51 % por sí solo, y en
los 960 clips donde gana el que encendió primero acierta el **45,21 %**, por debajo del azar).

Tu propuesta: de los clips de **una sola luz**, separar los **contraataques** —acciones donde no
tienes prioridad en el momento del tocado pero solo se enciende tu luz— y usar el resto **como si
tuvieran dos luces**. Eso multiplica el corpus.

Va al final porque **necesita la prioridad instante a instante**, que necesita el cono, que necesita
las manos, que necesitan Sapiens 2, que necesita la etapa A.

---

## 5. La cadencia: 1 → 2 → 5 → 10 → 100

**No se pasa al siguiente número sin la cifra del anterior.**

| paso | qué se mide | puerta para avanzar |
|---|---|---|
| **1 clip** | pasadas/s reales, VRAM, puntos devueltos | corre sin agotar memoria |
| **2 clips** | reproducibilidad | los dos dan 2 poses en el 100 % de cuadros |
| **5 clips** | primera validación visual tuya | 0 intrusos en las 5 |
| **10 clips** | tasa de interpolación y su longitud | ningún hueco interpolado > límite declarado |
| **100 clips** | proyección de coste al corpus | la extrapolación cabe en el tiempo disponible |

## 6. Lo que NO vamos a hacer, y por qué

- **No borrar vídeos ni poses.** Los 40 mp4 del escritorio están verificados en `D:` y el repo está
  en GitHub sin cambios pendientes.
- **No usar profundidad monocular para escala**: **46–51 % de error** sobre un vano conocido de
  4,00 m. El `pointmap` de Sapiens 2 es otra cosa y se mide por separado.
- **No dibujar la hoja donde no se ha visto.** Un cuaderno anterior trazaba 95 px desde la muñeca y
  metía esa hoja inventada en la etiqueta de «tocado». Eso no vuelve a entrar.
- **No fiarse de la consistencia temporal para validar un rastreo**: sembrar SAM2-vídeo en el sitio
  correcto pasa el criterio en 11 de 12 casos, y sembrarlo **en un señuelo dentro del cuerpo pasa en
  10 de 12**.

## 7. Dónde está todo

```
01_extraccion/      el código nuevo: aquí empieza la etapa A
02_validacion/      la herramienta y las muestras a verificar
03_datos/           punteros a vídeos (D:) y poses existentes
04_documentacion/   informes y tus ideas
05_aparcado/        el paquete de Colab y el entrenamiento
99_historico/       notas y capturas antiguas
```

---

## 8. La etapa A sigue abierta, y ahora sin pruebas a favor ni en contra

La validación de `seg` que estaba escrita aqui **se retira**: sus dos negativos eran los fotogramas
equivocados por el desfase `f0`. Lo unico que sobrevive de aquella medida son los **positivos**, que
tenian `f0 = 0`:

| caso con `f0 = 0`, valido | fraccion de persona segun `seg` |
|---|---:|
| tirador visible 720p | 0,580 / 0,415 |
| tirador visible 1080p | 0,335 / 0,455 |

O sea que `seg` **detecta personas donde hay personas**, y eso es todo lo que esta demostrado. **No
esta demostrado que separe** a un tirador de un arbitro, porque el negativo con el que lo comprobe no
valia.

### Lo que hay que hacer, en este orden

1. **Encontrar casos de contaminacion de verdad**, con el desfase `f0` aplicado. Sin un solo caso
   verificado no se puede construir ni medir ninguna puerta. Es la tarea del validador.
2. **Solo entonces** decidir si la puerta necesita `seg`, o si con la geometria basta, o si el
   problema es mas raro de lo que parecia.
3. Y ojo con una cifra que ahora encaja distinto: la auditoria de series dio **fraccion util mediana
   1,000**, o sea que en el clip tipico estan los dos tiradores en **todos** los cuadros. Con mi bug
   eso parecia contradictorio; sin el bug, es coherente con que las poses esten bastante bien.

### La leccion, que vale mas que la puerta

Mis nueve detectores geometricos no encontraron el problema porque **no habia problema donde yo
miraba**: el fallo estaba en mi codigo de visualizacion, no en los datos. Y lo que lo descubrio no
fue una tabla ni una metrica, sino **un `IndexError`** al pedir el cuadro 200 de un array de 131.

Por eso el validador tiene que llevar el desfase escrito y avisar cuando `f0 > 0`.

---

## 9. La interpolación, medida: correcta pero recupera el 0,82 %

La idea del usuario es válida y el límite ya no se inventa: **se midió escondiendo fotogramas que sí
existen**, interpolando desde los dos extremos y comparando con lo real. Sin anotación humana.

### El límite sale de la curva

| hueco | muestras | error mediano | veredicto |
|---:|---:|---:|---|
| 1 | 69.252 | 0,678 % | vale |
| 2 | 68.671 | 0,836 % | vale |
| 3 | 54.472 | 0,994 % | vale |
| **5** | 38.520 | **1,386 %** | **vale** |
| 8 | 26.376 | 2,018 % | pasa del límite |
| 12 | 18.827 | 2,897 % | |
| 30 | 7.899 | 5,489 % | |

Límite declarado antes de medir: error mediano < **2 % del alto de caja**. **Límite medido: 5
cuadros.**

### Pero el reparto de los huecos dice que eso apenas sirve

| tipo de hueco | huecos | cuadros | % del total | interpolable |
|---|---:|---:|---:|---|
| interior | 375 | 2.199 | 3,67 % | sí |
| **borde** | 155 | **4.182** | **6,98 %** | **no: falta un extremo** |

- **53,1 %** de los clips tienen hueco de borde (mediana **0,50 s**, p90 2,12 s, máximo **4,31 s**)
- 68,2 % tienen hueco interior

| vía | cuadros recuperados | % |
|---|---:|---:|
| **interpolar interiores ≤ 5 cuadros** | 493 | **0,82 %** |
| interpolar todos los interiores | 2.199 | 3,67 % |
| **redetectar los huecos de borde** | **4.182** | **6,98 %** |

**Los huecos de borde son 8,5× más grandes que lo que la interpolación arregla.** Y aplicado sobre
300 clips, interpolar dio **+120 poses de 68.989: +0,17 %**.

### Y la interpolación necesita una comprobación por hueco, no solo un límite de longitud

Con hueco de **un solo cuadro**, la mediana del error es 0,678 % pero hay cola:

| error mediano por encima de | muestras | % |
|---|---:|---:|
| 2 % del alto de caja | 6.539 | **9,44 %** |
| 10 % | 943 | 1,36 % |
| 25 % | 118 | 0,17 % |
| 50 % | 15 | 0,02 % |

Los peores llegan al **99 %**, o sea un error del tamaño de la caja entera, interpolando **un**
cuadro. Así que el límite de longitud no basta: hace falta rechazar el hueco cuando los dos extremos
están demasiado separados en posición.

### La consecuencia para el plan

**La interpolación entra, con límite 5 y con marca, pero no es la solución al problema de los
huecos.** Lo que hay que hacer es **redetectar**, que recupera datos reales en vez de inventarlos, y
el proyecto ya tiene `redetect_roi`, que recorta y amplía la región.

Y una predicción que conviene tener presente: **con extracción a duración completa los huecos de
borde van a CRECER**, porque el clip trae entradilla y cola donde los tiradores no están en pista.
Interpolar importará aún menos, y redetectar aún más.

---

## 10. Las cinco capas, medidas y vistas en vídeo

Mosaico en `02_validacion/muestras/mosaico/` y herramienta en
`02_validacion/herramienta/mosaico_video.py`. Se hizo **en vídeo y no en imágenes** porque una imagen
fija no enseña si una capa **parpadea**, y una máscara que salta entre cuadros no sirve para medir
velocidad ni para seguir la hoja.

| capa | ms/cuadro | veredicto |
|---|---:|---|
| `pose` (384×288) | 108 | **entra**: 87 % de los 308 puntos, pulgar al 100 % |
| `seg` | 245 | **entra**: segmenta a los dos tiradores por partes, limpio |
| `normal` | 314 | **dato crudo, solo detrás de `seg`**: sin máscara da relieve de todo el pabellón |
| `pointmap` | 480 | **NO entra** |
| `matting` | 767 | **entra, con una condición dura** |
| **las cinco** | **1.914** | |

### `pointmap` no sirve, y era la que más ilusión hacía

Contestando directamente a la pregunta sobre la malla 3D de profundidad: **medido, no da conocimiento
del espacio.**

| | tirador | fondo vacío del mismo tamaño |
|---|---:|---:|
| residuo de plano | 6,5 % | **6,7 %** (suelo vacío) |
| residuo de plano | 25,1 % | **23,8 %** (pared impresa lisa) |

La mediana de los negativos (12,19 %) es **mayor** que la de los tiradores (9,34 %). Falla el control
en **7 de 7** con máscara. Y hay tres cosas peores:

- con la máscara de `seg`, el cuerpo del tirador **ajusta un plano al 0,95 %**: devuelve un recorte
  plano, no volumen
- **coloca al tirador detrás de su propio fondo en 3 de 6 casos**
- **la z depende del recorte, no de la escena**: ampliando el recorte un 30 % la z mediana cambia un
  **30 %, hasta el 694 %**
- las unidades **no son métricas**: el mismo cuerpo mide 0,073 / 0,613 / 0,176 u, o sea ×24 / ×2,9 /
  ×10 si fueran metros

Y **está bloqueado a 1024×768**: su `config.json` declara `scale_final_input_size: 6144`, un tamaño
fijo, así que la palanca de 7,8× de la pose no le aplica. 480 ms por cuadro sin forma de abaratar.

Su tarjeta de modelo **no menciona escala ni ambigüedad de escala** en ninguna parte: había que
medirlo.

### `matting` es el mejor discriminador del proyecto, pero exige UN tirador por recorte

Separación medida: POS 0,2999–0,5218 contra NEG 0,0053–0,0067 = **44,6× sin solape**, en 3 sedes y
7 negativos adversarios. Aguanta a 384×288 con 23,5× por 125 ms.

**Pero la condición es dura y la encontré mirando el mosaico:**

| entrada | alfa media |
|---|---:|
| cuadro completo 1280×720 | 0,0202 |
| **un tirador solo, con margen** | **0,2160 / 0,2088** |
| **los dos tiradores en un recorte** | **0,0053** |

Con dos personas devuelve **0,0053**, que es exactamente el valor de los fondos vacíos. Así que lo
que discrimina **no es «persona contra nada»**: es **«una persona centrada contra cualquier otra
cosa»**. Como puerta hay que llamarla **una vez por candidato**, nunca sobre el par ni sobre el
cuadro.

Y **no es redundante con la blancura HSV**, que solapa: el suelo gris da 0,6965, por encima de 4 de 6
tiradores. Así que el «0,000 sobre pared vacía» que figuraba en este documento **no generaliza**: se
midió sobre una sola pared oscura.

### Dos correcciones a la puerta A1 con `seg`

1. **A 384×288 se rompe**: una pared impresa da `frac_persona` **0,6867**, por encima de todos los
   tiradores (máximo 0,5805). Hay solape.
2. **A 1024×768 no solapa, pero el margen es 1,35×**, no el «factor 4 a 8» que decía este documento.
   Los positivos se reproducen exactos; la diferencia son negativos elegidos adversarialmente.

**Con `matting` a 44,6×, la puerta A1 debe apoyarse en `matting`, no en `seg`.**

---

## 11. La tubería completa, funcionando en 1 clip

`01_extraccion/codigo/pipeline_1_clip.py`. Vídeo de verificación en
`02_validacion/muestras/video_final/`.

### Arquitectura: selección por TRAYECTORIA, no por cuadro

Elegir «los dos mejores de este cuadro» no tiene memoria, y por eso el algoritmo viejo podía cambiar
de persona a mitad de clip. Ahora se construyen trayectorias, se puntúan enteras y se eligen **dos
para todo el clip**. Con eso el cambio de identidad deja de ser posible **por construcción**, no por
umbral.

### La puerta que echa al árbitro: la y del pie, no `pointmap`

Medido en un cuadro con árbitro visible:

| | z de `pointmap` | y del pie |
|---|---:|---:|
| tirador izquierdo | 1,0391 | 0,856 |
| tirador derecho | 1,0391 | 0,852 |
| **árbitro** | **0,8550** | **0,756** |
| fondo (otra pista, público) | 0,777–0,820 | 0,63–0,65 |

**Las dos separan.** Se usa la y del pie porque cuesta **0 ms** en vez de 480: la perspectiva ya
ordena la profundidad sin modelo. `pointmap` queda como comprobación cruzada.

**El ancla de la puerta importa, y la primera versión estaba mal.** Anclando en la trayectoria *más
adelantada* el nivel lo fijaba una persona pegada a la cámara (pie 0,998), el umbral quedaba en 0,940
y **echaba a los propios tiradores** (pie 0,85): solo 2 de 34 trayectorias pasaban y los dos elegidos
salían del mismo lado. Anclando en la **mejor puntuada** —que por duración, talla y banda ya es un
tirador— pasan 9 de 34 y la elección es correcta.

### Resultado en el primer clip

| | |
|---|---:|
| trayectorias construidas | 34 |
| pasan la puerta de primer plano | 9 |
| lado 0 presente | **91,6 %** |
| lado 1 presente | **98,4 %** |
| los dos a la vez | **91,6 %** |
| cuadros interpolados y marcados | 4 |
| elegidos con alfa de `matting` < 0,05 | **0 de 479** |

Que **ninguno** de los 479 elegidos baje de 0,05 de alfa significa que la puerta no dejó pasar nada
que no fuera una persona: `matting` da 0,21 sobre tirador y 0,0053 sobre fondo vacío.

### El discriminador de cara queda DESCARTADO

La idea era que el tirador lleva máscara y el árbitro no, así que `Face_Neck` + `Hair` los separaría.
**Medido: no separa.** Los 11 candidatos de un cuadro dan cara+pelo entre 0,024 y 0,42, sin hueco.

Y el motivo se ve en la imagen: **`seg` etiqueta la máscara de esgrima como `Face_Neck`**. Los
tiradores dan 2-8 % de cara+pelo en vez de 0.

### Coste, con la GPU compartida

| capa | ms/cuadro |
|---|---:|
| pose (384×288) | 72–397 |
| `seg` sobre el recorte | 341–349 |
| `matting`, **dos llamadas** | 802–1490 |

El rango es amplio porque la GPU estaba compartida con otro proceso. `matting` cuesta el doble que
las demás porque hay que llamarlo **una vez por tirador**: con los dos en un recorte devuelve 0,0053,
el valor del fondo vacío.

---

## 12. La selección de la pareja, reescrita y medida en 12 clips (2026-09-10, tarde)

El usuario dijo *"centrate en limpiar la deteccion de los floretistas"*, y al mirar sus fotos había
tres defectos: la repetición del principio, cuadros con una sola persona, y cuadros con ninguna.
No eran tres cosas: eran **cuatro defectos distintos, todos medidos**, y ninguno era el que yo creía.

### Resultado, cuadros con LOS DOS tiradores presentes

| clip | antes | ahora | verificado a ojo |
|---|---:|---:|---|
| `HarshExemplaryGermanspaniel` | 100,0 | **100,0** | sí |
| `defeatedHarlequinRabbitSkinkLizard` | 99,7 | **99,7** | sí, c318 |
| `AdvancedDefiniteAvocet` | fallback | **100,0** | sí, EBERT contra NISHIOKA |
| `MellowHideousAngora` | 84,7 | **100,0** | sí, MARTON contra VECCHI |
| `DeficientNippyGar` | fallback | **100,0** | sí, RUSSO contra VECCHI |
| `GrimyRipeIbizanhound` | 100,0 | **100,0** | no |
| `BigheartedNegligibleFeline` | 100,0 | **100,0** | no |
| `VainAchingGonolek` | 93,3 | **96,1** | sí, c0 |
| `RichDefensiveBushbaby` | 32,8 | **93,8** | sí, SHIKINE contra IIMURA en c10 y c290 |
| `GentleQuerulousElephantseal` | fallback | **89,0** | sí, TOLDO contra GIACON |
| `BlissfulTiredAiredale` | 84,4 | **84,4** | sí, FOCONI contra YUNES |
| `ZanyMeagerIndianrockpython` | 40,4 | **80,4** | sí, FREILICH contra FICHERA en c99, c199, c240 |

Mediana **99,8 %**, mínimo 80,4 %, **cero clips** en la vía de reserva, y 9 de 12 por encima del 90 %.
Coste 1,7 s por clip con las detecciones cacheadas.

### Defecto 1 · La restricción de lados respecto al centro de imagen

Los dos tiradores filmados de `ZanyMeager` estaban **los dos en la mitad derecha** del cuadro, porque
la cámara encuadra su pista a la derecha. La regla "uno a cada lado de 0,5" los prohibía por
construcción, y premiaba en su lugar una pareja **cruzada entre dos pistas distintas**, con los dos
a **5,58 alturas de cuerpo** de separación. Nadie se bate a 5,58 alturas.
Izquierda y derecha son ahora RELATIVAS entre los dos.

### Defecto 2 · La señal que ordena bien no era ninguna de las que usaba

`ZanyMeager` tiene cuatro pistas a la vez, así que `1-9` **también es una pareja real de tiradores**,
no basura: es otro asalto. Lo que distingue a la pareja FILMADA es que la cámara la sigue, así que
comparten plano casi todo el clip: **81 % contra 22 %**. La talla apenas separaba (1,17×).

### Defecto 3 · La fusión rechazaba el empalme más obvio del corpus

En `RichDefensive` el tirador de la izquierda estaba partido en 0-120 y 141-366. En la costura,
**IoU 0,86 y diferencia de pie 0,004**: es él, parado 21 cuadros sin que el detector lo viera. Mi vía
de predicción lo rechazaba con residuo 3,03 porque **multiplica por 21 una velocidad estimada con 5
cuadros**. Extrapolar lejos es peor que suponer que no se ha movido. Ese solo empalme valía **141 de
los 151 cuadros que le faltaban al clip**.

### Defecto 4 · La puerta de primeros planos, y CUATRO formas de plantearla mal

| planteamiento | qué le pasó |
|---|---|
| histograma HSV entre cuadros | no ve el corte; su pico máximo era un cambio de fondo dentro del mismo primer plano |
| talla normalizada por la mediana del clip | la repetición contamina el normalizador |
| **talla absoluta > 0,60** | tiró los cuadros 354-368 de `RichDefensive`, que son el final del asalto con los dos a la vista: **exactamente los cuadros donde el usuario vio "no detecta a nadie"** |
| ¿tiene compañera la caja más grande? | marcó dos tramos en vivo donde el ÁRBITRO era la caja mayor y estaba solo, con los tiradores detrás |
| ¿hay alguna pareja plausible en el cuadro? | tiró 152 cuadros de `RichDefensive` y 168 de `defeatedHarlequin`, casi todo el asalto |

Lo que sí separa es **la conjunción de las dos últimas condiciones**, y hace falta medir las dos:

| tramo | p50 del alto de la caja mayor | máx |
|---|---:|---:|
| `Harsh` c0-108, la repetición | 0,860 | 0,883 |
| `Harsh` c109+, en vivo | 0,320 | 0,351 |
| `RichDefensive`, cámara fija y ancha | 0,551 | **0,713** |
| `defeatedHarlequin`, árbitro delante | 0,495 | 0,570 |

El mínimo de la repetición es 0,789 y el máximo en vivo 0,713: **la talla sola da 1,11× de margen**,
que es poco, y por eso NO decide sola. La segunda condición es la que da el margen de verdad: en la
repetición el 73 % de los cuadros tienen UNA persona detectada, y en los cinco tramos en vivo, 0 %.
Y el tramo se **alarga mientras la caja enorme siga ahí**, porque el corte de plano está donde ella
desaparece (escalón de 2,7× en c108→c109), no donde se apaga la condición de pareja.

Con esto marca **exactamente** los 109 cuadros de la repetición de `Harsh` y **cero** en los otros once.

### Las dos puertas de geometría se ensancharon, con margen calculado

Con `sep ≤ 2,60` y `dpie ≤ 0,10`, **3 de 12 clips** caían en la vía de reserva, que coge las dos
mejores sueltas sin comprobar nada. Al mirarlos, en los tres la pareja verdadera estaba ahí y era la
que las puertas rechazaban, por planos anchos y en ángulo:

| clip | sep en alturas | dpie | qué puerta fallaba |
|---|---:|---:|---|
| `GentleQuerulousElephantseal` | **2,84** | 0,023 | separación |
| `DeficientNippyGar` | **2,82** | **0,102** | las dos |
| `AdvancedDefiniteAvocet` | 1,75 | **0,106** | pie |

Nuevos límites **3,20 alturas** y **`dpie` 0,14**, con el margen medido contra lo que hay que
rechazar: la pareja cruzada entre pistas está a 5,58 alturas (**1,74×**) y la pareja mala con el
`dpie` más bajo, tirador con espectador de fondo, a 0,180 (**1,29×**).

### La interpolación de poses baja a último recurso: la caja se estima, la pose se MIDE

Interpolar la pose es inventar datos. Pero hay un paso intermedio que no inventa: **se interpola la
CAJA**, que es una posición y un tamaño, y sobre esos píxeles reales **se mide la pose**. El detector
de cuadro completo pierde al tirador cuando es pequeño o está cortado por el borde; en un recorte
alrededor de donde debería estar, lo ve.

Medido: recupera **23 % de lo que faltaba, con un control negativo del 3 %** (la misma caja subida a
las gradas), o sea **7,7× de separación**. Verificado a ojo en `ZanyMeager` c212, que es un cuerpo a
cuerpo, y en `VainAching` c0.

### Lo que queda sin resolver, dicho claro

Quedan **181 cuadros de 7.660 (2,4 %)** donde falta uno de los dos, y en la mayoría **no está en el
plano**: son el principio de `ZanyMeager` y los once primeros de `VainAching`, donde la cámara aún no
los ha encuadrado. Ahí no hay nada que detectar y las dos únicas salidas honradas son marcarlos como
estimados o recortar el clip. `tipo` lleva esa marca (0 medido, 1 interpolado, 2 sostenido,
3 redetectado) y **no se debe perder al guardar**.

---

## 13. Siempre las dos poses, y cuanto cuesta inventarlas (2026-09-10, noche)

El usuario lo pidio asi: *"hay frames que no detecta a uno de los dos y por eso no saca el cono...
inventarlo en base a la ultima pose que conociamos y a la siguiente que conozco, que haga como las
poses intermedias, porque si no hay momentos que no vemos a los dos y perdemos el foco"*.

Hecho, y en este orden, que importa porque cada paso deja menos que inventar al siguiente:

| paso | que hace | lo que cuesta inventar |
|---|---|---|
| 1 puerta de planos | quita repeticiones | nada, son cuadros que no se usan |
| 2 fusion | une trozos de la misma persona | nada, es dato medido |
| 3 redeteccion en recorte | **la "doble pasada"** que pedia el usuario | nada, es dato medido |
| 4 caja estimada + pose medida | la ventana se estima, la pose se mide | nada de la pose |
| 5 pose interpolada | ultimo recurso, entre las dos anclas | esto si se inventa |
| 6 pose sostenida | en los bordes, sin ancla a un lado | esto si se inventa |

**Resultado: 100,0 % de los cuadros utilizables con LAS DOS poses**, en los cinco clips revisados en
video. Lo inventado es 49 poses de 500 en `ZanyMeager`, 23 de 738 en `RichDefensive`, 63 de 808 en
`BlissfulTiredAiredale`, 7 de 358 en `VainAching` y **cero** en `HarshExemplary`.

### El error de inventar, medido escondiendo poses que si se conocen

Error mediano de los 63 puntos utiles, en fraccion del largo del torso, que es lo que permite
comparar entre clips con encuadres distintos:

| largo del hueco | `ZanyMeager` | `RichDefensive` | `BlissfulTired` |
|---:|---:|---:|---:|
| 1 cuadro | 2,8 % | 2,8 % | 2,8 % |
| 3 cuadros | 4,8 % | 4,4 % | 4,3 % |
| 5 cuadros | 7,3 % | 6,0 % | 6,0 % |
| 8 cuadros | 11,4 % | 9,0 % | 8,2 % |
| 12 cuadros | 15,9 % | 14,6 % | 12,1 % |
| 24 cuadros | 31,1 % | 25,4 % | 20,7 % |

Con 400-740 casos por celda. **Interpolar gana a sostener en todos los largos**, incluido 24, asi
que sostener queda solo para los bordes, donde no hay dos anclas y no hay alternativa.

Hasta 8 cuadros el error se queda por debajo del 11 % del torso, que es del orden de lo que ya se
acepta por bajar la resolucion de la pose (2,26 px). De ahi para arriba crece rapido y la pose
inventada hay que tratarla como lo que es.

### LO QUE ESTA MEDIDA NO DICE, y es importante

El error se mide tapando cuadros **donde el tirador SI estaba y SI se detecto**. Los huecos que de
verdad se rellenan son en buena parte cuadros donde el tirador **no esta en el plano**, y para esos
no existe verdad con la que comparar: su error es, por construccion, no verificable. La cifra de
arriba es el error de interpolar un movimiento continuo, no el de adivinar a alguien que no se ve.

Por eso `origen` (0 medido, 1 interpolado, 2 sostenido) va guardado junto a los puntos y **no se
debe perder**: el entrenamiento tiene que poder pesar distinto un dato medido y uno inventado, y esa
decision no es de la extraccion.

### Videos de revision

`02_validacion/videos_revision/*.mp4`, con verde el de la izquierda, naranja el de la derecha,
**violeta y rotulado las poses inventadas**, y banda roja en los cuadros que la puerta de planos
descarta. La herramienta es `02_validacion/herramienta/video_revision.py`, en dos fases porque los
interpretes no coexisten.

---

## 14. Cuatro averias nuevas, encontradas revisando el vídeo y buscándolas a propósito

El usuario revisó `VainAchingGonolek.mp4` y dijo que **las poses inventadas estaban mal**. Lo
estaban, y la causa no era la interpolación. Buscando el motivo aparecieron cuatro averías más, y
tres de ellas **no se veían en ninguna cifra agregada**: el clip daba 96,1 % de presencia sobre
poses mezcladas de dos planos distintos.

### 14.1 · Fundido encadenado: dos planos superpuestos en el mismo fotograma

Los cuadros 7-18 de `VainAching` son un fundido: el marcador y el rótulo del fondo salen
**duplicados** y YOLO detecta tiradores de los dos planos. La pareja elegida sale mezclada y no hay
nada correcto que interpolar. La puerta de primeros planos no lo ve porque en un fundido ninguna
caja es enorme.

La señal: un fundido es `f[k] = a*A + b*B`, así que se reconstruye mucho mejor con dos cuadros que
con uno. **Y hay un confundido que apareció al verificar: un ZOOM da la misma firma**, porque el
cuadro de verdad es intermedio entre sus vecinos. Lo que los separa es el residuo absoluto, y sale
al revés de lo que esperaba:

| tramo de `VainAching` | razón r1/r2 | r2/contraste | qué es |
|---|---:|---:|---|
| c7-c18 | 3,23 | **0,219** | fundido, verificado a ojo |
| c42-c48 | 2,79 | **0,062** | zoom lento, verificado en c45 |

Y **lo intenté al revés y fue un desastre**: dejando que el relevo de trayectorias disparara, marcó
el 97,5 % de `BlissfulTiredAiredale`, el 54,8 % de `ZanyMeager` y el 65,7 % de `DeficientNippyGar`,
todos falsos. Los píxeles disparan porque son raros; las trayectorias solo afinan los límites.

Verificado en **3 fundidos** ya: `VainAching` c7-c18, `UnfortunateColossalLamb` c82-c89 (pico 3,72,
se ven dos pistas superpuestas) y `IncompleteThriftyFunnelweaverspider` c79-c80.

### 14.2 · Sostener una pose en los bordes está mal, y se sustituye por RECORTAR

Las 7 poses malas que vio el usuario eran todas de tipo `sostenida`: la pose del cuadro 7 estampada
hacia atrás sobre los cuadros 0-6, donde ese tirador **no está en el plano**. Interpolar entre dos
anclas sí está medido; sostener no tiene ancla al otro lado y no se puede validar de ninguna forma.

Ahora el clip **se recorta** al tramo donde los dos están medidos. `VainAching` pasa de 7 poses
inventadas a **cero**, con 155 cuadros utilizables de 179.

### 14.3 · Las dos cajas sobre la MISMA persona

`DearestThirdImago` c202: las dos cajas solapan **0,91**, una encima de otra sobre el mismo tirador,
porque el otro no está en el plano. La puerta de separación no lo veía porque trabaja con la MEDIANA
sobre los cuadros compartidos. Se decide cuál es la intrusa por continuidad con su propio pasado: la
que sigue en su sitio es la buena, la que aparece de la nada es la intrusa.

### 14.4 · Dos personas de pie detrás de la valla, elegidas como pareja

`WhisperedTenderDeinonychus`: la pareja elegida eran **dos espectadores al borde de arriba del
cuadro**, con la y del pie en 0,185, mientras PASZTOR y DUBROVICH ocupaban el centro. Ganaron porque
comparten el 77 % de los cuadros contra el 56 % de la pareja buena, y `comparte` pesa 0,45. Un
término de puntuación no arregla esto: hace falta puerta dura, `frente >= 0,40`. Las parejas
verificadas por el rótulo del vídeo dan 0,565 a 0,795, así que el margen es **3,05x**.

### 14.5 · Y al arreglar eso salió que el límite de `dpie` era irreparable

Con la puerta de banda puesta, ese clip dejó de coger a los espectadores pero **tampoco encontraba a
los tiradores**: su pareja verdadera tiene `dpie` **0,220**, más alto que la peor pareja intrusa
medida en `ZanyMeager`, que es 0,180. Los dos grupos se solapan y **ningún umbral absoluto los
separa**. La cámara mira a lo largo de la pista, así que los dos tiradores están de verdad a
distancias distintas.

Dividir por la talla lo arregla, porque la perspectiva mueve las dos cosas a la vez: si un tirador
está más lejos, su caja se encoge Y sus pies suben, en proporción.

| grupo | `dpie` / talla |
|---|---|
| parejas verificadas a ojo por el rótulo | 0,027 a **0,451** |
| parejas intrusas verificadas en `ZanyMeager` | **0,688** a 0,997 |

Margen **1,53x** donde el absoluto no tenía ninguno.

### 14.6 · El marco de muestreo estaba mal, y esto invalida parte de lo anterior

Muestrear todos los `.mp4` del árbol de vídeo da 42.326 nombres, pero ese árbol junta las descargas
de **varios estudios distintos**, no el corpus etiquetado. De los 12 clips que medí así, **solo 2
estaban en el corpus**, y en la muestra aparecieron un clip de **ESPADA** (`DearestThirdImago`,
"Men's Epee - T64") y uno de **SABLE** (`FickleJovialCentipede`, "Men's Sabre Grand Prix").

El corpus etiquetado son 10.840 nombres, de los que **8.559 están en disco (79,0 %)**. Esa es la
lista que hay que muestrear, y está en `scratchpad/corpus_etiquetado.txt`.

**Lo que NO resultó ser un problema**, y lo comprobé antes de alarmar: 1.482 clips del corpus
(17,3 %) vienen de tres sedes cuyo nombre no dice "foil" (`worldchamps2022`, `eurochamps2022`,
`tokyoolympics2020`), que son campeonatos de las tres armas. Mirando tres de ellos, **son florete**:
lamé solo en el torso, mangas blancas y careta de florete. n=3, así que no es una garantía, pero no
hay indicio de contaminación dentro del corpus etiquetado.

### 14.7 · El buscador automático de errores

`02_validacion/herramienta/busca_errores.py` aplica los síntomas de todas las averías conocidas
(`salto`, `solapan`, `fundido`, `recorta`, `sep`, `dpie`, `talla`, `faltan`, `pocos`), ordena los
clips por número de avisos y **guarda la imagen del cuadro señalado** de cada aviso. No decide: un
aviso no es un error hasta que se ve.

Sobre **25 clips del corpus etiquetado**, después de los cuatro arreglos:

| | antes de los arreglos | después |
|---|---:|---:|
| los dos presentes, mediana | 100,0 % | **100,0 %** |
| p10 | 96,0 % | **96,5 %** |
| mínimo | 90,2 % | **89,2 %** |
| clips sin ningún aviso | 12 de 25 | **14 de 25** |
| avisos `solapan` | 2 | **0** |

Y el cambio que más importa no está en la tabla: `WhisperedTenderDeinonychus` daba **96,5 % sobre la
pareja equivocada** y ahora, o acierta, o falla a gritos y el buscador lo señala. Cambiar un fallo
silencioso por uno ruidoso vale más que un punto de mediana.

---

## 15. El cambio de identidad entraba por una puerta que no habia mirado

El usuario pidió arreglar el recorte de `BabyishIdioticCockerspaniel` (215 de 363 cuadros) y los
saltos de caja. Resultaron ser **la misma avería**, y la más grave de todas las de hoy.

### Lo que se midió primero, porque sin esto no se puede poner ningún límite

Desplazamiento real del centro de un tirador entre cuadros consecutivos, sobre **6.243 transiciones**
de las parejas verificadas a ojo, en anchos de caja:

| p50 | p95 | p99 | p99,9 | máx |
|---:|---:|---:|---:|---:|
| **0,016** | 0,102 | **0,168** | 0,460 | 2,938 |

Los saltos que avisaba el buscador eran de **1,22, 1,41 y 1,67 anchos**, o sea **8 a 10 veces el
p99**. Eso no es un tirador moviéndose.

### Tres arreglos, y por qué hicieron falta los tres

**1 · El límite de asociación estaba en fracción del ANCHO DE IMAGEN.** `DIST_MAX_FRAC = 0,10` de
1280 px son 128 px, que para una caja de 90 px de ancho son **1,4 anchos de caja**: holgura de sobra
para saltar a otra persona. Ahora el límite va en anchos de la caja y crece con el hueco. Bajó el
recorte de `Babyish` de 215 a 124 cuadros, **pero los saltos siguieron**.

**2 · Garantía de salida.** Los saltos podían entrar por otras tres vías: la de IoU del rastreador,
la fusión por predicción (que mide el residuo contra la posición extrapolada, no contra la última
conocida) y la redetección. Perseguir cada vía deja siempre la siguiente abierta, así que ahora hay
un filtro al final: **ningún cuadro entregado tiene un salto por encima de 0,60 anchos**, que corta
el 0,08 % de las transiciones. Y hay **dos pasadas**, porque la primera corre antes de la
redetección y una caja redetectada colaba un salto después del filtro.

**3 · Trayectorias GEMELAS, que es la causa de verdad.** En `Babyish` dos trayectorias se sentaron
encima del MISMO tirador en c250-c253, con centros idénticos (cx 0,937 y 0,937, IoU 0,68), y al
separarse **cada una se fue con una persona distinta**: en c266 una estaba en cx 0,600 y la otra en
0,854. Eso es un **cambio de identidad**, el fallo que corrompe la etiqueta de lado.

No se puede cazar con el filtro de continuidad: reparte 1,8 anchos en 8 cuadros, o sea 0,23 por
cuadro, dentro de lo normal. Hay que impedirlo en el origen, y ahora se impide: si dos trayectorias
vivas comparten cuadro con IoU por encima de 0,60, se mata la más corta en el acto y sus detecciones
futuras se asocian a la que sobrevive.

### Resultado sobre 25 clips del corpus etiquetado

| | antes de esta sección | después |
|---|---:|---:|
| los dos presentes, mediana | 100,0 % | **100,0 %** |
| p10 | 96,7 % | **98,1 %** |
| **mínimo** | 92,5 % | **97,8 %** |
| clips sin ningún aviso | 14 de 25 | **15 de 25** |
| clips con aviso de `salto` | 3 | **0** |

Y sobre los 12 verificados a ojo, sin regresión: mediana 100,0 %, p10 98,3 %, mínimo 94,6 %, y
**29 cuadros que faltan de 7.660 (0,4 %)**.

### Y el recorte de `Babyish` era CORRECTO, no un defecto

Mirando el cuadro 40: solo JURKIEWICZ está en el encuadre; FILIPPI **no está en el plano**, la
cámara está tan a la izquierda que se queda fuera. Los 124 cuadros que se recortan son cuadros donde
el segundo tirador no existe en la imagen. El aviso `recorta` es verdadero, pero la decisión de la
tubería es la buena: ese clip solo tiene a los dos en 239 de sus 363 cuadros, y eso es una propiedad
del vídeo, no un error del algoritmo.

---

## 16. Tres arreglos mas, una idea mia que salio peor, y las capas densas limpias

### 16.1 · Estado sobre tres lotes distintos

| lote | mediana | p10 | mínimo | sin ningún aviso |
|---|---:|---:|---:|---:|
| 12 verificados a ojo | 100,0 % | 98,3 % | **94,6 %** | — |
| 25 del corpus | 100,0 % | 98,0 % | **97,8 %** | 15 de 25 |
| 40 del corpus, nuevos | 100,0 % | 83,2 % | 75,9 % | 22 de 40 |

En los 12 verificados quedan **29 cuadros que faltan de 7.660 (0,4 %)** y los 12 pasan del 90 %.

### 16.2 · La recuperación se enganchaba al rival

Los avisos de "las dos cajas solapan 0,91" no venían del selector: venían de la **recuperación**
poniendo la caja del que falta encima del que sí estaba. Solo aparecían en tiradas CON redetección, y
eso es lo que lo delató.

El arreglo va **sin umbral**: la caja recuperada tiene que parecerse más a donde debería estar el que
falta que a la del rival. Un umbral absoluto no sirve, y está medido: con 0,50, `ZanyMeager` bajaba
de 99,5 % a 81,6 % porque rechazaba cuerpos a cuerpo legítimos, y su c212 está verificado a ojo.
Los avisos de solape pasaron de **3 a 0** en el lote de 40.

### 16.3 · Una idea mía que sonaba gratis y salió PEOR

Propuse mirar si el que falta ya estaba entre las detecciones del cuadro completo antes de gastar GPU
redetectando en un recorte. Medido por separado:

| clip | solo recorte | solo vía gratis | las dos |
|---|---:|---:|---:|
| `ZanyMeager` | **99,5 %** | 79,6 % | 81,6 % |
| `RichDefensive` | **94,6 %** | 93,5 % | 93,2 % |
| `GentleQuerulous` | 98,1 % | 98,1 % | 98,1 % |

El mecanismo: la vía gratis coge una caja, **sobrescribe la estimación** de dónde debería estar el
tirador, y si el filtro de continuidad la tira, la redetección se queda buscando alrededor de una
estimación envenenada. No es que aporte poco: estropea la vía buena. Apagada.

Y el "25-49 % de recuperaciones disponibles gratis" con que la justifiqué **estaba mal medido**: lo
conté en una tirada SIN redetección, así que contaba como oportunidad lo que la redetección ya
encontraba sola.

### 16.4 · Y esa idea me hizo diagnosticar mal el arreglo siguiente

Metí la vía gratis y la fusión de hueco largo en la misma tirada. Los 12 verificados empeoraron
(mínimo de 94,6 % a 81,6 %), lo atribuí al umbral de la fusión, lo subí de 0,20 a 0,30 y me quedé
sin el arreglo. Separando los dos cambios, la culpa era **entera de la vía gratis**: con ella
apagada, el umbral a 0,20 sube `PowerfulAllLarva` de **51,1 % a 75,9 %** y no mueve ni un cuadro en
los otros seis clips que probé.

**Dos cambios en la misma tirada no se pueden atribuir.** Ahora hay un interruptor
(`extrae_pareja.USA_CRUDAS`) para poder apagar una vía y medirla sola.

### 16.5 · La fusión de hueco largo

`PowerfulAllLarva` tenía a los dos tiradores partidos en un hueco de 43-44 cuadros, y sus mitades
encajaban sin duda: `dpie` 0,033 y 0,046, razón de talla 0,87 y 1,02, IoU 0,32 y 0,22. Pero con hueco
de 43 la única vía abierta pedía IoU 0,50 y el tirador se había movido un poco. Ahora hay una vía
para huecos largos que pide IoU 0,20 y sigue exigiendo `dpie` y talla, que son los que de verdad
distinguen a la misma persona de otra: la fusión peligrosa que hay que rechazar, tirador con
espectador que pasa por delante, salta 0,27 de `dpie` contra el 0,05 que se admite.

### 16.6 · Las capas densas, limpias: `seg` y `normal` valen, `pointmap` no

Lo midió un agente en paralelo. **Lo que fallaba:** recortar la caja del tirador y correr la capa
dentro **no borra al árbitro**. `matting` se queda con todas las personas del recorte (fuga 0,251) y
a veces **elige a la persona equivocada** — en un cuadro recortó al árbitro y dejó al tirador fuera.
`seg` cubre bien al tirador (0,993) pero es el que más árbitro cuela (0,383).

**La corrección:** reparto por esqueleto más cercano (`distanceTransform` sobre la pose de los dos y
de todo ajeno, sin umbral) más guarda de caja.

| máscara | recall del tirador | fuga de árbitro | residuo |
|---|---:|---:|---:|
| la caja sola, línea base tonta | 0,997 | 0,0000 | 0,657 |
| `seg` | 0,993 | 0,3828 | 0,289 |
| `matting` | 0,905 | 0,2505 | 0,249 |
| **unión + esqueleto + guarda** | **0,994** | **0,0000** | **0,263** |

El esqueleto baja la fuga 2,9x sin coste de recall y la guarda la lleva a cero costando 0,001. Y
bate a la caja sola: mismo recall, misma fuga cero, **2,5x menos residuo**. Control negativo con la
máscara desplazada media anchura: recall 0,982 → **0,068**.

**Veredictos:** `seg` es la base y `normal` vale, las dos a resolución nativa. `matting` **solo en
unión con `seg`**, nunca sola. **`pointmap` NO vale como profundidad**: falla el control físico,
separa a los dos tiradores entre sí más que a la pareja del árbitro, y la `y del pie`, que es
gratis, acierta **28 de 28**.

**Lo que NO está verificado:** queda árbitro donde su caja pisa la del tirador (33 % de los pares del
primer clip, entre **0,6 % y 5,3 %** de la máscara final), el proxy de color se cae con tiradores del
1,9 % del cuadro, y son 3 clips.

### 16.7 · Lo que queda abierto

Cuatro clips del lote de 40 entre el 75,9 % y el 83,6 %, todos por la misma causa: uno de los dos
falta durante 44-81 cuadros seguidos dentro del tramo utilizable. No es un cambio de identidad ni un
intruso; es que el segundo tirador no está tracked ahí, y hay que mirar clip por clip si está en el
plano o no.

---

## 17. Por qué la profundidad falla y `normal` no, y qué pasa con la cabeza `depth` de Sapiens 1

Preguntó el usuario: *"y por qué falló la de Sapiens 2, no lo puede intentar sacar, porque si
funcionan las de las mallas no entiendo"*. Es la pregunta correcta y tiene respuesta concreta.

### Las cinco cabezas están, y el mapa de calor es una de ellas

Comprobado en el índice público, no de memoria: Sapiens 2 publica **cinco** familias de tarea —
`pose`, `seg`, `normal`, `pointmap`, `matting` — más los `pretrain`, que son solo tronco sin cabeza.
**Las cinco se extraen.** El mapa de calor que se ve en el mosaico **es el `pointmap`** pintado en
color; no es una capa aparte, y su canal z *es* la profundidad.

### Por qué `normal` sirve y `pointmap` no, siendo los dos "mallas"

Miden cosas de naturaleza distinta:

  `normal` da la DIRECCIÓN a la que mira la superficie en cada píxel. Es **local y relativa**: para
  usarla basta comparar un píxel con su vecino, dentro de una sola persona, y nunca se le pide que
  compare a dos personas entre sí.

  `pointmap` da una POSICIÓN 3D por píxel. Usarla como profundidad obliga a que sus valores sean
  comparables **entre personas distintas del mismo cuadro**, que es una exigencia mucho más fuerte.

Y ahí es donde se rompe, porque la profundidad monocular es **ambigua de escala**: una persona
pequeña y cerca y una grande y lejos proyectan lo mismo. El modelo lo resuelve con conocimiento
previo del tamaño humano, y estos modelos están entrenados con **una persona encuadrada de cerca**
— el propio nombre del fichero lo dice, `render_people`, que es un dataset de humanos escaneados de
uno en uno. Nuestros cuadros son lo contrario: un pabellón entero con 4 a 15 personas a distancias
muy distintas y de 100 a 500 px de alto.

Medido: dentro de un recorte la z **no ordena nada** (53 %, moneda al aire) y de cuadro completo
mejora pero sigue poniendo a los dos tiradores más lejos entre sí que del árbitro en buena parte de
los cuadros.

### Y sí, probé la cabeza `depth` de Sapiens 1, que Sapiens 2 ya no tiene

Existe en variante torchscript, así que se carga sin montar mmengine. El mapa sale **correcto de
vista**: el árbitro en primer plano cálido, los tiradores medios, el fondo azul. Pero no pasa el
control, y la línea base gratis lo bate en los tres clips:

| clip | ordena bien, `depth` de Sapiens 1 | ordena bien, `y del pie` |
|---|---:|---:|
| `defeatedHarlequin` | 48 % | **74 %** |
| `ZanyMeager` | 38 % | **44 %** |
| `RichDefensive` | 32 % | **100 %** |

Primero lo medí metiendo el cuadro 16:9 **estirado** en una entrada 3:4, que deforma la geometría y
deja franjas de ruido en los bordes. Rehecho con relleno, mejora algo (44→48 %, 25→38 %) y **el
veredicto no cambia**. Se deja anotado porque un veredicto sacado del preproceso malo no habría
valido.

### La razón de fondo, y por qué la `y del pie` gana

La `y del pie` funciona por un hecho **geométrico exacto**, no por un prior aprendido: con la cámara
por encima de la pista, lo que está más cerca tiene los pies más abajo en el encuadre. No hay
ambigüedad de escala que resolver. Por eso una resta de dos números gana a una red de 1,3 GB.

`pointmap` **se sigue extrayendo y está en el mosaico**, porque el usuario quiere verla y porque
puede servir para otra cosa. Lo que no hace es decidir quién está en la pista.

Sin verificar: son 3 clips y 16-28 cuadros por clip, y la razón depende de que haya un ajeno en el
plano.

---

## 18. El árbitro elegido como tirador, y por qué la blancura no puede ser una puerta

Esta sección sale de convertir cada garantía de la tubería en un **check ejecutable** en vez de mirar
la nota media. La herramienta es `02_validacion/herramienta/tabla_checks.py` y la tabla vive en
`02_validacion/CHECKS.md`.

La diferencia con `busca_errores.py` importa: aquel busca *síntomas* y ordena clips por sospecha, así
que un aviso suyo no es un error hasta que se ve. Esto son *garantías que el propio código dice
cumplir*, y ahí un fallo sí es un error. El primer pase sobre 60 clips del corpus levantó un check en
rojo, y detrás había **dos errores reales que ninguna cifra agregada veía**.

### 18.1 · El «tirador izquierdo» era el árbitro

En `PowerfulAllLarva` la caja del lado izquierdo está puesta sobre el **árbitro**: chaqueta oscura,
caminando en primer plano por delante del marcador. Verificado a ojo en c160 y c224, imágenes en
`02_validacion/muestras/cruce_lados/`.

Y pasó las tres puertas geométricas de `pareja_v2` **sin rozarlas**:

| puerta | límite | lo que dio |
|---|---|---:|
| separación | 0,15 a 3,20 alturas | 1,93 |
| dpie relativo | ≤ 0,55 | 0,46 |
| pies en la banda | ≥ 0,40 | 0,894 |

Era previsible en cuanto se dice en voz alta: **el árbitro camina junto a la pista**, así que tiene
los pies igual de bajos en el encuadre y una talla parecida. La geometría no puede distinguirlo, y
perseguirlo moviendo esos tres umbrales sólo habría tirado tiradores buenos.

El síntoma que sí lo delató fue otro: los dos lados **se cruzan** en 29 de 193 cuadros compartidos, o
sea que la caja «izquierda» acaba a la derecha de la «derecha».

### 18.2 · El otro error: un trozo de espectador en el borde

`LavishDistantDikkops` entregaba como tirador izquierdo un **fragmento de espectador pegado al borde
izquierdo del cuadro**, con blancura 0,000, mientras el tirador real estaba en el plano y sin caja.
El clip daba **27,4 % de presencia** y ningún check lo cazaba. En particular pasó la puerta de tiras
(`extrae_pareja.PROP_MIN`) porque su proporción era 0,33, por encima del 0,25.

### 18.3 · La blancura: por qué es un aviso y no una puerta

En esgrima la chaqueta del tirador **tiene que ser blanca**: es regla del deporte, no un umbral
ajustado a estos datos. Y ya estaba medido en el proyecto que la blancura **absoluta** no sirve de
puerta (dos tiradores reales en un pabellón oscuro dan 0,069 y 0,000), pero que el **contraste entre
los dos lados del mismo cuadro** sí significa algo, porque los dos comparten alumbrado.

Medido sobre 60 clips con `01_extraccion/codigo/blancura_pareja.py`, y con **siete casos verificados
a ojo uno por uno** (`02_validacion/muestras/limite_intruso/`):

| regla candidata | marca | caza de los 2 errores | se lleva de más (verificado correcto) |
|---|---:|---:|---|
| contraste ≥ 0,25 | 10 de 60 | 2 de 2 | `proudDungBeetle`, `ClearLeafyBats`, `RawReckless` |
| cruce > 2 % | 3 de 60 | 1 de 2 | `SimilarOffbeat` |
| contraste ≥ 0,25 **y** blancura mínima ≤ 0,20 | 4 de 60 | 2 de 2 | `ClearLeafyBats`, `RawReckless` |
| techo de blancura ≤ 0,20 **y** diferencia ≥ 0,25 | 3 de 60 | 1 de 2 | ninguno |

Ninguna separa limpio, y **los falsos positivos explican por qué**, que es lo que hace que esto no se
pueda arreglar apretando el número:

- `proudDungBeetleBoglenTerrier`: el contraste lo produce el **lamé dorado**, que es reglamentario en
  florete. Un tirador con lamé y otro de blanco dan 0,477 de contraste siendo los dos correctos.
- `ClearLeafyBats` y `RawRecklessBluegill`: los dos son tiradores; la blancura baja en algunos
  cuadros por sombra.
- `SimilarOffbeatIberianlynx`: los dos tiradores **se cruzan de verdad**, en el 9,3 % de los cuadros,
  al pasar uno junto al otro. Verificado a ojo.

Ese último dato es el que impide usar el cruce como puerta: el caso con árbitro daba 15,0 % y una
pareja correcta 9,3 %, o sea **1,6 veces de margen con un solo positivo verificado**. Poner un
umbral ahí sería inventárselo.

Así que la blancura entra como **aviso `blancura` en `busca_errores.py`**, con 6,7 % de carga de
revisión sobre 60 clips, y no como filtro que borra datos. Tirar a un tirador bueno es peor que
revisar un clip de más.

### 18.4 · Lo que sí se convirtió en puerta, porque es definicional

Una trayectoria cuya caja está **cortada por el borde lateral en el 100 % de sus cuadros** nunca se
vio entera, así que no existe ninguna observación con la que comprobar que es un tirador. No es que
sea poco probable: **es que no hay dato**. Está en `pareja_v2.BORDE_SIEMPRE`.

Y por qué el valor es 1,00 y no un umbral ajustado, medido en los mismos 60 clips: esa fracción tiene
p50 0,067, p90 0,432, p95 0,592 y **p99 0,936**, o sea que pasar mucho rato en el borde es normal. El
segundo valor más alto es 0,892 en `WastefulYawningAmericanquarterhorse`, y ahí el tirador de la
derecha está de verdad medio fuera de cuadro en un plano ancho: **verificado a ojo como correcto**.
Sólo un clip de los 60 llega a 1,00 y es justo el roto.

Resultado del arreglo:

| clip | antes | después |
|---|---:|---:|
| `LavishDistantDikkops` | 27,4 % | **100,0 %** |
| los otros 59 | sin cambio | sin cambio |

Y la tirada completa de 60 clips pasa de mínimo 27,4 % a **mínimo 84,3 %**, con 59 de 60 clips por
encima del 90 %.

### 18.5 · Un check que estaba mal formulado, y era mío

La primera versión de la tabla tenía un check «los dos tiradores nunca cambian de lado». Salió en
rojo, y al mirarlo resultó que **el check estaba mal, no la tubería**: los tiradores sí se cruzan
cuando uno pasa junto al otro. Está reformulado como medida con su margen al lado, no como garantía
binaria. Queda escrito porque es el tipo de error que se repite: una garantía inventada desde el
escritorio puede ser más falsa que el código que juzga.

### Lo que esta sección NO dice

Sólo hay **dos casos malos verificados a ojo**, así que cualquier umbral que los separe está
débilmente determinado y las cifras de precisión de la tabla se apoyan en 7 verificaciones, no en
cientos. Lo que sí queda establecido, y no depende del tamaño de muestra, es *por qué* cada rasgo
falla: el lamé dispara el contraste y los tiradores reales se cruzan. Eso son hechos del deporte.

Y queda un error **sin arreglar**: `PowerfulAllLarva` sigue entregando al árbitro. Está marcado por
el aviso `blancura` para revisión, pero la tubería no lo corrige sola.

---

## 19. La PRIORIDAD instante a instante: el reglamento sí, el detector todavía no

El usuario lo pidió así: *"necesito que saques lo de la PRIORIDAD, para ver si sacas cuando tiene cada
uno la prioridad, para ponerlo. Y cuando la pierde. Y revisa la FIE para ver cuando"*, y después
*"cuando la pierde, cuando la gana el otro, con el intervalo que habra, un intervalo de gracia, porque
si no estaria cambiando infinitamente todo el rato"* y *"que detectamos bien las paradas"*.

Mapa regla → señal en `04_documentacion/prioridad_fie.md`, código en `01_extraccion/codigo/prioridad.py`,
evidencia y vídeos en `02_validacion/prioridad/`.

### 19.1 · Primera corrección, y es al encargo: los artículos NO son t.56-t.60

El reglamento está en el disco: **Reglamento Técnico de la FIE, diciembre de 2025**, en español, en
`Downloads/190675-tecnico esp.pdf`, 66 páginas. Verificado contra el texto, no de memoria. Lo que hay
en `data/fie/` del otro repo son **atletas y rankings**, no reglas.

| se buscaba en | lo que hay ahí de verdad |
|---|---|
| t.56-t.58 | anulación de tocado por **fallo del material**: chaquetillas agujereadas, hoja rota |
| t.59 | lo general de las tres armas: *«el árbitro analiza brevemente la última frase de armas»* |
| t.60-t.63 | **vídeo-arbitraje** |
| **t.76-t.89** | **FLORETE, CONVENCIONES DEL COMBATE**, y la prioridad es **t.82-t.89** |
| t.8-t.15 | las definiciones sin las que no se mide nada: tiempo de esgrima, ataque, parada, punta en línea |

Buscar la prioridad en t.56 habría dado cero. El índice del propio reglamento lo confirma:
*«Prioridad del tocado: t.59, t.82ss, t.100ss»*.

### 19.2 · La regla que hace esto medible sin ver la hoja

**t.83.2.a**: *«El ataque simple es ejecutado correctamente cuando al extender el brazo, la punta que
amenaza la superficie válida, **precede el inicio del a fondo** o de la flecha.»*

O sea que la corrección de un ataque es **el signo de una resta de dos instantes**. Dos instantes
sobre una serie de poses se miden. Por eso hay detector y no solo documento.

### 19.3 · Las variables, medidas en 21 clips de FLORETE con poses de 308 puntos

| criterio declarado ANTES de medir | resultado | veredicto |
|---|---|---|
| `ext_hr` (hombro→mano / torso) con recorrido p95/p05 ≥ 1,5 | mediana **2,21**, 40 de 42 clip×lado | **pasa** |
| `ext_cr` (codo→mano / torso) más estable que `ext_hr` | cv 0,211 contra 0,241, solo 27 de 42 | **no pasa limpio** |
| algún arranque de brazo en cada clip×lado | 234 tramos, 0 de 42 sin ninguno | **pasa** |

**La variable que pedía el encargo no sirve, y está medido por qué:** `codo → raíz de la mano` es un
**hueso**, no una extensión. Su largo aparente varía un **21,1 %** por escorzo y ruido de pose, y no
crece cuando el brazo se extiende. El control que lo separa es medir los mismos largos en píxeles: el
torso normaliza bien (cv **0,079**) y la variación está en el punto de la mano. La que sirve es
`hombro → raíz de la mano`.

### 19.4 · El tiempo de esgrima de t.8, y la cota de alcance

El reglamento define el tiempo de esgrima y **no lo pone en segundos**, así que hay que medirlo:

| p25 | **p50** | p75 | p90 | máx |
|---:|---:|---:|---:|---:|
| 3 | **5 cuadros** | 8 | 10 | 22 |

**167 ms a 30 fps**, y de ahí salen el umbral de simultaneidad de t.89.2 y el intervalo de gracia.

Y una cota que es mitad reglamento y mitad medida: la hoja de florete mide **0,90 m**, así que dos
puntas no se encuentran con las manos a más de **1,80 m**; pero el ataque no empieza a distancia de
tocado, **la cierra**, y cuanto cierra está medido: p90 de **1,48 m** sobre 96 subidas de brazo. Cota
para conceder prioridad: **3,28 m** entre manos.

**Lo puse mal primero y la medida lo delató**: con la cota sola de 1,80 m el acierto en cuadros con
dueño se hundió de 25,6 % a **3,3 %**. La cota era correcta para el instante del tocado y yo la
aplicaba al del arranque.

### 19.5 · El acierto: bate a la línea base de cajas y empata con no contestar

Contra **21 ventanas anotadas a mano** de 10 clips, más **2 ventanas de control** de clips que no se
miraron al arreglar nada.

| medida | detector | base «quien avanza primero» | trivial «siempre ninguna» |
|---|---:|---:|---:|
| 21 ventanas, por cuadro (n=462) | **69,7 %** | 34,6 % | **73,8 %** |
| solo cuadros CON dueño (n=121) | 25,6 % | **41,3 %** | 0 % |
| solo cuadros SIN dueño (n=341) | 85,3 % | 32,3 % | **100 %** |
| **cuando los dos nombran dueño, el LADO** (n=36) | **86,1 %** | — | — |
| falsa alarma | 14,7 % | — | 0 % |
| instante del cambio, ±3 cuadros | **2 de 10** | — | — |
| **2 ventanas de CONTROL (n=44)** | **9,1 %** | — | 0 % |

**El detector no funciona todavía.** Bate 2,0× a la línea base de cajas, empata con no contestar
nunca, y en el lote de control no contesta. Cuando se moja acierta el lado en el **86,1 %**, pero se
moja en 36 de 121 cuadros: **el lado es bueno y la cobertura es mala**.

### 19.6 · Los dos motivos de que no generalice, localizados con cifras

**A · `pies` no separa la marcha del a fondo.** La separación de tobillos crece en cualquier paso, así
que el a fondo sale antes que el brazo y t.83.2.a anula el ataque:

| acción verificada a ojo | pico de `pies` |
|---|---:|
| tres a fondo verificados | 3,22 · 3,41 · 3,58 |
| **flecha corriendo** de `AliveIgnorantBarebirdbat` | **2,54** |
| p90 agrupado, el umbral en uso | **2,31** |

El umbral cae por debajo de la banda del que corre. **No se sube a 2,9**, que separaría los cuatro
casos, porque sería elegir el número mirando la anotación: es el mismo error que no se cometió al no
bajar el umbral de fundidos de 1,60 a 1,55.

**B · los umbrales absolutos no viajan entre clips.** `EXT_ALTO` es el p75 agrupado (1,091) y hay
tiradores que nunca lo pasan: `HarshExemplaryGermanspaniel` lado 1 tiene **máximo 1,00 en 301
cuadros**. **7 de 42 clip × lado (16,7 %) no registran ni un arranque de brazo en todo el clip**, que
es físicamente imposible. El siguiente paso es normalizar por tirador.

### 19.7 · El intervalo de gracia: el diagnóstico del usuario era correcto y el remedio era otro

| variante | cambios/s | × los reales |
|---|---:|---:|
| **completo** | **0,55** | **0,8** |
| sin intervalo de gracia | 0,55 | 0,8 |
| sin histéresis (un solo umbral) | 0,38 | 0,6 |
| **sin filtro de amplitud y altura** | **0,84** | **1,3** |
| sin nada de lo anterior | 0,89 | 1,4 |
| **anotado a mano** | **0,65** | 1,0 |

La oscilación existía —1,4× la tasa real— pero **`PERMANENCIA` no cambia ni un cuadro**: lo que la
frena es el filtro de amplitud y altura. Se queda en el código con esta tabla al lado, como
`USA_CRUDAS = False`. La tasa final, **0,55 contra 0,65 cambios por segundo**, es la única medida en
que el detector se parece a la realidad.

### 19.8 · Las paradas: siete candidatos y CERO cifra de acierto

| | |
|---|---:|
| candidatos del criterio de t.86 | 8 en 10 clips |
| **paradas anotadas a mano en 15,4 s de acción** | **0** |
| acierto de paradas | **no se puede calcular** |

No vi ni una parada-respuesta limpia en las 21 ventanas: hay ataques simples, contraataques y
acciones simultáneas. Con n = 0 no hay cifra, y decir «detecta paradas» sería inventarlo. Y la razón
de fondo es dura: **la parada es contacto de hojas por definición**, y la hoja se detecta en el
**1,6 %** de los recortes a 720p contra el 42,6 % a 1080p. Veinte de los veintiún clips de florete son 720p.

Lo que haría falta, concreto: anotar paradas a propósito partiendo de las 8 candidatas (daría
precisión, no cobertura) y clips a 1080p. Lo que **no** se va a hacer es prolongar 90 cm de hoja desde
la mano: está prohibido por escrito en este documento y medido que apunta a donde no está.

### 19.9 · Cinco averías encontradas buscándolas, cada una con su imagen

| avería | dónde | qué se ve |
|---|---|---|
| **prioridad sin adversario** | `Alive` c213 | «DERECHA con confianza 1,00» con el izquierdo **fuera del plano** y el derecho volviendo andando. Motivó la cota `ALCANCE_MAX` |
| **intruso elegido como tirador** | `PowerfulAllLarva` c147 | la caja «izquierda» está sobre una **persona de calle que cruza en primer plano**. Fallo del selector de pareja, no de este módulo, pero la prioridad de ese clip es basura. Encontrado por separado y es **el mismo clip y la misma causa que la §18**, que lo diagnosticó a fondo |
| **clip de SABLE** | `MellowHideousAngora` | cartel «SABRE FEMININ» al fondo. Colado por el muestreo viejo ya declarado mal en §14.6. Excluido |
| **cajas que faltan en plena acción** | `PotablePossible` c227 | sin las dos poses no hay decisión y la salida correcta es `ninguna` por falta de dato |
| **mi propia anotación equivocada** | `Gentle` c270-c288 | había escrito «derecha» y el zoom por tirador enseña al **izquierdo** extendiendo primero. **La corrigió el detector al discrepar** |

La última es la más útil de la tabla porque va al revés de lo normal: **el algoritmo contradijo al
humano y tenía razón.** Por eso la anotación lleva una columna `revisado_zoom`: 9 de las 21 ventanas
se volvieron a verificar con el zoom por tirador y una cambió de lado.

### 19.10 · LO QUE ESTA MEDIDA NO DICE

- **No dice que se pueda arbitrar con esto.** Cinco de las doce reglas de concesión de prioridad
  (t.84.1, t.84.2, t.85, t.86, t.87) exigen **ver la hoja** y no están implementadas. Un clip cuya
  frase se decide por una parada está **fuera del alcance** de lo medible hoy.
- **No dice que la anotación sea la verdad.** La hizo un agente mirando láminas, no un árbitro FIE, y
  se demostró equivocada una vez. La resolución del instante es **±3 cuadros** porque las láminas van
  de 3 en 3 y los de en medio no se han visto: dar el instante exacto sería fingir precisión.
- **No dice que el 69,7 % sea reproducible.** Los arreglos de definición se encontraron mirando la
  señal en 5 de las 21 ventanas, así que ese número está **inflado**. El honesto es el del lote de
  control, **9,1 %**, y solo son 44 cuadros de 2 ventanas.
- **No dice que la punta en línea se detecte.** t.15 y t.84.1 no se implementan, y no por falta de
  ganas: el control de `apunta` como sustituto ya está **fallado** en este proyecto con −2,9° de
  diferencia pareada y 17 de 36 con el signo esperado.
- **No dice nada del tocado.** Ni la etiqueta de tocado ni el orden de las luces entran en ninguna
  medida de esta sección, a propósito: el orden de las luces acierta el 63,51 % por sí solo y ya
  hundió el techo de un modelo al 73,6 %.

---

## 20. El techo lo pone la resolución del vídeo, no el modelo

Esta sección existe porque **a 720p la hoja del florete no se puede usar**, y eso cambia la
planificación de todo el trabajo de hoja y de cono.

> **MATIZ IMPORTANTE SOBRE ESAS CIFRAS, porque yo las cité mal de entrada.** Escribí «la hoja se
> detecta en el 1,6 % de los recortes a 720p contra el 42,6 % a 1080p» como si fuera una constante
> del proyecto, y no lo es: son medidas de cosas distintas y hay tres números en circulación.
>
> | medida | qué cuenta | 720p | 1080p |
> |---|---|---:|---:|
> | `e5` | recortes repartidos por todo el clip | 1,6 % | 42,6 % |
> | `e13` | rectas pegadas a la mano | 12,9 % | 6,5 % |
> | `e10` | conos **en la ventana de acción** | — | **3,0 %** |
>
> La diferencia entre el 42,6 % y el 3,0 % es el **desenfoque de movimiento**: la acción es justamente
> donde el arma va rápida. Y el 12,9 % de `e13` no contradice nada, porque el control que lo juzga no
> es la tasa sino la **alineación** con el eje del brazo: **60,9° a 720p**, peor que los 45° del azar,
> o sea que **eso que se detecta a 720p no es la hoja**; y **23,1° a 1080p**, que sí es señal.
>
> Lo que hay que citar, entonces, no es una tasa sino dos hechos: a 720p lo detectado no es la hoja, y
> a 1080p la hoja sólo se ve en el **3,0 %** de los conos de la ventana de acción. La conclusión de
> esta sección aguanta y se refuerza.
>
> Moraleja operativa: **una tasa de detección sin un control de que lo detectado es lo que se busca no
> mide nada.** Es el mismo error que ya está documentado en la sección 17 con el criterio de confianza
> de la pose, que aprobaba el 100 % de las resoluciones.

Así que la pregunta obligada es cuánta de nuestra materia prima es 720p. Medido sobre 400 clips del
corpus muestreados con semilla fija, leyendo la cabecera del vídeo:

| resolución | clips de la muestra | fracción | equivalente en el corpus de 8.559 |
|---|---:|---:|---:|
| 1280×720 | 325 | **81,2 %** | ~6.950 |
| 1920×1080 | 72 | 18,0 % | **~1.540** |
| 854×480 | 3 | 0,8 % | ~70 |

`fps`: mediana 29,99, mínimo 17,96, máximo 59,82.

### Qué implica

La resolución mayoritaria del corpus es la que **no** sirve para la hoja. Es decir:

- Las **cinco de doce reglas de prioridad FIE que exigen ver la hoja** (t.84.1, t.84.2, t.85, t.86,
  t.87) no son medibles en la mayor parte del corpus, y no por falta de modelo.
- Y ni siquiera a 1080p alcanza para trabajar por cuadro: en la ventana de acción la hoja se ve en el
  **3,0 %** de los conos, así que **el 97 % de los conos llevan la dirección supuesta desde el brazo,
  no medida desde la hoja**. Eso está marcado dentro del NPZ con `origen` (0 hoja vista, 1 supuesta),
  para que no se puedan confundir.
- El **cono en 3D apoyado en la hoja** sólo se puede construir en un subconjunto, y hay que decir
  siempre sobre cuántos clips vale cada cifra.
- Los ~1.540 clips de 1080p son el subconjunto donde este trabajo tiene sentido, y conviene tratarlos
  como una cohorte aparte en vez de promediarlos con el resto.

### De paso: el corpus NO está contaminado, el muestreo sí lo estaba

Apareció un clip de SABLE (`MellowHideousAngora`, rótulo «SABRE FEMININ») en una tirada, y la primera
conclusión fue que la lista del corpus tenía otras armas dentro. **Es falso, comprobado**: ese clip no
está en `corpus_etiquetado.txt`, vive en `D:/FENCINGLY_VIDEOS_RAW/scratchpad/height_study/mp4/`, que
es la descarga de otro estudio dentro del mismo árbol. Entró por muestrear el **árbol** en vez de la
**lista**, que es la trampa que ya estaba documentada y que sigue costando tiradas.

Lo que sí es cierto, y queda como riesgo no observado: la lista **no tiene columna de arma**. El
82,7 % de los clips vienen de sedes cuyo nombre dice florete (`anaheimfoil2018`, `qatarfoil2021`…) y
el **17,3 % (1.482 clips)** vienen de sedes de arma mixta, donde el nombre no lo garantiza:

| sede mixta | clips |
|---|---:|
| `worldchamps2022` | 931 |
| `eurochamps2022` | 388 |
| `tokyoolympics2020` | 163 |

Muestreé 6 de esas sedes y verifiqué 2 a ojo (`ThankfulMixedGerbil` de Cairo 2022 y
`WillingShamelessBrownbutterfly` de Tokio): **los dos son florete**, con lamé sólo en el torso y
careta de malla, no metálica, que es lo que los distingue de sable. Así que **no hay contaminación
observada**; el riesgo es teórico y se apoya sólo en el nombre de la sede. Por si hace falta, hay una
lista restringida a las sedes de florete en
`scratchpad/corpus_florete_seguro.txt` (7.077 clips), pero **no se ha hecho obligatoria**, porque
descartar 1.482 clips buenos por un riesgo no observado es peor que el riesgo.

### Lo que esta sección NO dice

La resolución se midió en 400 de 8.559 clips, así que las fracciones traen su error de muestreo (para
el 18 % de 1080p, el intervalo al 95 % es de unos ±3,8 puntos). Y las tasas de detección de hoja por
resolución vienen de la medida de otro agente sobre 21 clips: son suficientes para decidir dónde
trabajar, no para publicarlas como constantes del proyecto.

---

## 21. Cuatro averías en las POSES entregadas, y las tres que se arreglaron sin tocar la GPU

La tabla de checks (`02_validacion/CHECKS.md`) se extendió al nivel de pose, y levantó cuatro cosas.
Tres eran arreglables **sin volver a pasar Sapiens 2**, y esa parte es la que merece explicación.

### 21.1 · 10.140 keypoints entregados caían sobre otra persona

El proyecto ya tenía medido que el estimador de pose **pone keypoints encima del árbitro** cuando
éste tapa al tirador, y ya tenía el remedio escrito: `capas_limpias.limpia_keypoints`, que pone a
cero la confianza de un punto que cae fuera de la caja de detección de su propia persona. Es una
corrección de construcción, no un umbral: la caja *es* lo que delimita a esa persona.

Lo que nadie había comprobado es que ese filtro estaba **sólo en la ruta de las capas densas**, no en
`video_revision.poses_del_clip`, que es la ruta que produce y guarda las poses que se entregan. El
check P4 lo cuantificó: **10.140 keypoints entregados fuera de su caja**, sobre 22 clips.

Está en `video_revision._fuera_de_caja` y ahora corre en la ruta de entrega. Es exactamente el fallo
que el usuario había pedido buscar: *«se te cuele una persona de detrás en alguna pose»*.

### 21.2 · Ocho poses SOSTENIDAS que no debían existir, por un problema de orden

`extrae_pareja._recorta` garantiza que el primer y el último cuadro entregado tienen las dos **cajas**
medidas. Pero corre antes de que exista la pose. Después, `tira_poses_vacias` puede quitar
precisamente la pose del cuadro del borde por venir vacía, y ese extremo se queda con una pose
**sostenida**: la última conocida, congelada, sin ancla al otro lado con la que compararla.

Medido en la galería antes del arreglo: **8 de 45 casos eran sostenidos**. Y una pose sostenida es
justo lo que el usuario había señalado como mala al revisar `VainAchingGonolek`.

`video_revision.recorta_por_poses` vuelve a recortar los extremos **después** de filtrar las poses
vacías. Cuesta 80 cuadros en 22 clips y deja **0 sostenidas**: las 606 poses inventadas que quedan son
todas interpoladas con ancla a los dos lados, o sea todas validables.

### 21.3 · Los tres arreglos se aplicaron a los NPZ existentes SIN GPU, y por qué se pudo

Volver a pasar Sapiens 2 por 22 clips son horas de GPU. No hizo falta, y la pieza que lo permitió es
`origen`: como cada pose lleva escrito si es medida (0) o inventada (1, 2), se puede

1. quedarse sólo con las **medidas**, tirando lo inventado,
2. aplicarles el filtro de keypoints fuera de caja, que sólo necesita `KP`, `SC` y `B`,
3. aplicarles el filtro de poses vacías y volver a recortar los extremos,
4. **volver a inventar** las que faltan, desde anclas ya limpias.

El resultado es idéntico al que produciría la tubería nueva. Está en
`02_validacion/herramienta/repara_npz_poses.py`, y los originales quedan como `poses_*.npz.previo`.

**Esta es la razón práctica por la que `origen` y `tipo` no se deben perder**, y ya no es un argumento
de principio: sin esa marca, este arreglo habría sido imposible.

Lo que el arreglo NO puede hacer: no cambia las cajas. `PowerfulAllLarva` sigue teniendo al árbitro,
porque eso se corrige en `pareja_v2` y exige recalcular la pose.

### 21.4 · La cuarta: más de la mitad de lo inventado está fuera del rango medido

Al mirar la galería apareció el caso peor, `FirsthandSoreGnat` c116, con **66 cuadros de hueco**: la
pose inventada queda **puesta donde no hay nadie**, porque el tirador no se movió en línea recta
durante más de dos segundos. La imagen está en `02_validacion/poses_inventadas/013_...hueco66.png`.

El error de interpolar está medido **hasta 24 cuadros** (25,7 % del torso). Por encima no hay medida.
Y contado sobre los 22 clips:

| lo entregado | poses | fracción |
|---|---:|---:|
| medidas por el modelo | 12.340 | **95,3 %** |
| interpoladas con hueco ≤ 24 cuadros (rango con error medido) | 251 | 1,9 % |
| interpoladas con hueco > 24 cuadros (**sin medida**) | 355 | **2,7 %** |

Es decir: de las 606 poses inventadas, el **58,6 %** viene de huecos más largos que cualquier cosa
que se haya validado.

**Y no se ha puesto un tope**, a propósito. Se guarda `hueco`, el largo del hueco del que sale cada
pose inventada, junto a los puntos. La razón es la misma por la que se guarda `origen` en vez de
filtrar: quien entrene tiene que poder pesar distinto una interpolación de 1 cuadro y una de 66, y
para eso necesita el número, no un booleano que ya eligió alguien por él. Decidir el tope no es de la
extracción.

En la galería, esos casos llevan el rótulo en **rojo** y dicen `hueco FUERA del rango medido: error
NO verificable`, en vez de mostrar una cifra recortada a 25,7 % que engañaría por omisión.

### 21.5 · Un check mal formulado, otra vez mío

La primera versión de P1 daba en rojo por seis NPZ escritos **antes** de que existiera el filtro de
poses vacías: tenían poses medidas de 0 a 8 puntos útiles dentro. El check acusaba a la tubería de un
fallo que ya no comete.

Arreglado de raíz y no en el check: el NPZ ahora lleva `filtro_vacias`, `puntos_min` y
`recortados_por_pose`, o sea **el dato dice cómo se hizo**. Un fichero sin esa marca no se puede
juzgar, y la tabla lo informa aparte como pendiente de regenerar en vez de contarlo como fallo.

### Estado de los checks

Los **doce** salen APTO: C1-C8 en 60 clips de cajas y P1-P5 en 22 clips de poses. La tabla, con el
umbral declarado de cada uno, lo medido, el sitio del código que lo cumple y dónde está la evidencia,
está en `02_validacion/CHECKS.md` y subida al repo privado.

### Lo que esta sección NO dice

Los checks de pose se miden en 22 clips, no en el corpus. Y `PowerfulAllLarva` sigue entregando al
árbitro: está marcado por el aviso `blancura` para revisión, pero la tubería no lo corrige sola.

---
## 22. El cono en 3D: la z de `pointmap` no es profundidad, y el 3D que sí se puede tener es gratis

El usuario lo pidió así: *"intenta pulir cada vez más lo de la detección del florete con los conos y
eso, para añadirlo como dato el cono, y hay que ver que sea 3D, porque si no no tiene sentido en 2D.
Así que a ver cómo lo haces con la profundidad, igual te ayuda"*.

**Su razonamiento es correcto y esta sección no lo contradice**: dos floretes que se cruzan en la
imagen pueden estar a un metro en profundidad, así que una intersección medida en 2D puede ser falsa.
Lo que se mide aquí es de dónde puede salir esa profundidad. Y la respuesta es que **no puede salir
de `pointmap`**, pero sí se puede acotar con geometría que cuesta 0 ms.

Código en `01_extraccion/codigo/cono3d_lib.py` y `e9`…`e13`. Figuras en
`02_validacion/muestras/cono3d/`.

### 22.0 · `e9_eje_3d.py` no podía ejecutarse, y por eso el agente anterior se paró

Importaba `ultralytics` **y** `capas_limpias` (que importa `transformers`) en el mismo proceso, y en
esta máquina **los dos intérpretes no coexisten**:

| intérprete | tiene | le falta |
|---|---|---|
| `fz/.venv-gpu` | `ultralytics` 8.4.142 | `transformers` |
| `cuda/.venv` | `transformers` 5.16.1, torch 2.14, cv2 5.0 | **`ultralytics`** |

Comprobado con `-c "import X"` en los dos. Reescrito para trabajar sobre el caché `det_<CLIP>.pkl`
(107 clips, 0 s) con la **redetección apagada**, que es lo que baja la presencia y va dicho por clip
en vez de escondido.

### 22.1 · Las cifras de la FIE, verificadas en fuente primaria porque estaban de memoria

FIE *Material Rules* (Book 3), edición de agosto de 2026,
`static.fie.org/uploads/40/204157-book%20material%20August%202026%20ang.pdf`, cotejada con la de
diciembre de 2025 (idénticas en florete):

| magnitud | valor | artículo |
|---|---|---|
| **hoja de florete** | **90 cm** máximo | m.8.4 |
| arma montada | 110 cm máximo | m.7 |
| mango | 20 cm (líneas B-E), 18 cm (B-D) | m.4.1 |
| peso | < 500 g | m.6 |

**Y una cifra falsa que circula: los 88 cm de hoja son de SABLE** (m.23), no de florete. Anotado con
la URL en `cono_lib.HOJA_M`, junto con la salvedad de que `L = 90 cm` se mide **desde la mano**, y la
mano agarra por detrás de la cazoleta: `L` es por tanto una cota **por lo bajo** del alcance, o sea
conservadora para decir «amenaza».

De paso, y para lo que hace falta más abajo, la pista (*Technical Rules* Book 1, ago 2026 — y **no es
t.1, que es el saludo**, sino t.16-t.19): anchura **1,50 m** exactos (t.18.1, y el rango «1,5 a 2 m»
que circula es de ediciones ≤2017), longitud **14 m** ±4 cm (t.18.2), líneas de guardia a **2 m** del
centro (t.19.2), líneas de fondo a **7 m** (t.19.3), y los **últimos 2 m de otro color** (t.19.4).

### 22.2 · La muestra, y por qué es el MEJOR CASO y no una muestra representativa

Cinco clips de `corpus_etiquetado.txt`, **todos a 1080p a propósito**, verificados a ojo por el lamé
(que en florete cubre solo el torso): NISTA-PAUTY, TOFALIDES-LEFORT, BIBARD-OISHI, MERTINE-NAGANO.

**Dos avisos sobre la muestra, los dos encontrados mirando las imágenes:**

- `WelloffForcefulIndri` e `IdealFrigidJerboa` son **el mismo asalto** (BIBARD T contra OISHI R,
  mismo marcador y misma sede). Son **4 asaltos independientes, no 5**.
- El corpus es **81,2 % de 720p, 18,0 % de 1080p** y 0,8 % de 480p. Mi censo propio sobre los 78
  clips del corpus con detección cacheada da 82,1 % / 17,9 %, que concuerda. Como la hoja se ve en el
  42,6 % de los recortes a 1080p y en el **1,6 %** a 720p, **el cono con hoja VISTA solo es
  construible en unos 1.540 clips de 8.559**. Elegir 1080p era obligado para poder medir, y a la vez
  hace que estas cifras sean la cota superior de lo alcanzable, no la media del corpus.

### 22.3 · CORRECCIÓN: el eje del brazo es `hombro -> raíz de la mano`, no `codo -> raíz`

El encargo de este paso decía `codo -> raíz`. Está mal, y **los datos de esta sección lo corroboran
por su cuenta**, sin usar la medida de otro agente: el coeficiente de variación del largo
**proyectado**, sobre 10 celdas clip×lado, vale

| segmento | cv mediano | rango | qué es |
|---|---:|---|---|
| `codo -> raíz` | **0,108** | 0,065-0,589 | un **HUESO**: no crece al extender |
| `hombro -> raíz` | **0,226** | 0,062-0,420 | el que lleva la **EXTENSIÓN** |

Razón mediana **1,73×**. Un hueso rígido tiene cv pequeño; un segmento que se estira lo tiene grande.
Así que `hombro -> raíz` es la variable operativa y `codo -> raíz` sirve solo como patrón rígido —
que es justamente para lo que se usa en el criterio C3 de abajo.

Cambiado en `cono3d_lib.EJE_POR_DEFECTO = "hombro"`. **Pero cuál de los dos predice la DIRECCIÓN de
la hoja no se decide por argumento**: en los cuadros donde la hoja se ve hay verdad-terreno, y `e14`
carea los tres candidatos contra ella con el azar declarado en 45° y un margen declarado de 5°.

**Y ese careo NO puede decidir, con su cifra** (`e14`, agrupando los tres clips con hoja detectada):

| eje | n | mediana \|hoja − eje\| | p25-p75 | ≤20° |
|---|---:|---:|---|---:|
| `hombro -> raíz` | 6 | **23,1°** | 15,3-48,8 | 50 % |
| `raíz -> pulgar` | 6 | 24,3° | 19,5-36,6 | 33 % |
| `codo -> raíz` | 6 | 25,8° | 21,3-31,6 | 17 % |
| azar declarado | — | 45,0° | — | — |

Los tres están por debajo del azar, o sea que los tres llevan algo. Pero **se separan 1,2-2,7°, por
debajo del margen declarado de 5°: EMPATAN**, y con este `n` no se puede elegir entre ellos.

Y esto lleva un escarmiento propio: con los dos primeros clips (n=5) salía `hombro` ganando por
**7,9°** y lo di por bueno en un informe intermedio. **Al añadir el tercer clip se convirtió en
empate.** Un margen declarado sirve precisamente para esto; sin él habría publicado un ganador que
no existe. El eje que se usa sigue siendo `hombro -> raíz`, pero **descansa en `ext_hr` y en la tabla
de cv de arriba, no en el careo con la hoja**, que no da para tanto.

### 22.4 · Los cinco criterios sobre la z de `pointmap`, declarados antes de mirar

Balance: **C3, C4 y el test del escorzo FALLAN**, C1 queda al borde del límite, C2 se retira porque
estaba mal diseñado y C5 aprueba en mediana pero no en la cola. Ninguno de los que aprueban lo hace
por buenas razones, y los tres que fallan son los que tienen patrón externo.

Ventana de 100 cuadros **consecutivos** por clip (los criterios miden saltos entre cuadros contiguos,
y una muestra salteada no puede contestarlos). Se midió en **dos ventanas por clip** y hacen falta las
dos, porque las preguntas no viven en el mismo sitio:

- **ventana QUIETA**, el tramo consecutivo más largo con los dos presentes. Es la buena para el ruido.
- **ventana ACCIÓN**, la que **maximiza los cuadros con el rival dentro de los 90 cm**. Hizo falta
  porque en la quieta del primer clip la distancia mano-torso da mediana **200 cm** y el 2D no dice
  «amenaza» en **0 de 200** conos: ahí la pregunta «¿discrepan 2D y 3D?» no se puede ni plantear,
  porque las dos respuestas son «no». La ventana se elige por un criterio **medido** (la distancia) y
  no por la etiqueta de tocado, que metería circularidad.

Y lo primero que sale de la ventana de acción es cuántos cuadros tienen la pregunta viva:

| clip | cuadros con el rival a ≤ 90 cm | del clip | dist. mediana en la ventana |
|---|---:|---:|---:|
| `AromaticShockedGermanspitz` | 52 | 17,3 % | 89 cm |
| `SeparateHoarseFeline` | 41 | 13,7 % | 107 cm |
| `WelloffForcefulIndri` | 37 | 12,3 % | 161 cm |
| `ImpeccablePertinentBasilisk` | 8 | 3,2 % | 159 cm |
| **`IdealFrigidJerboa`** | **0** | **0,0 %** | 200 cm |

**En `IdealFrigidJerboa` el rival no entra en los 90 cm en ningún cuadro del clip**, así que ahí el
cono no puede decidir nada y no hay comparación 2D/3D posible. Es un clip de 5 y va dicho.

| criterio | límite declarado | medido | veredicto |
|---|---|---|---|
| **C1** estabilidad de la dirección 3D | ≤ 3× el salto 2D | 2D **2,28°** contra 3D **7,76°** = **2,8×** | pasa 6 de 10 (quieta) y 8 de 10 (acción) |
| **C2** la elevación lleva señal | temblor/recorrido < 0,5 | **0,08** | pasa 10 de 10, **y no vale: ver abajo** |
| **C3** RIGIDEZ: el hueso hace de patrón | cv(3D) < cv(2D) | pasa **5 de 20** (quieta) y **9 de 20** (acción) = **14 de 40** | **FALLA** |
| **C4** coherencia de cámara | dispersión de `f` < 0,25 **y** píxel cuadrado | **0,50 / 1,11 / 1,42** · y `s_col/s_fila` = **1,21-1,45** | **FALLA** |
| **C5** ruido de la z con el brazo quieto | mediana < ±17 cm | mediana **6,8 cm**, p90 **20-70 cm** | mediana pasa 9 de 10, **la cola no** |
| escorzo, con control del azar | spearman ≥ 0,30 | mediana **−0,086**, **0 de 10** celdas | **FALLA** |

**C2 estaba mal diseñado y se retira como criterio.** Su razón temblor/recorrido sale 0,08 y pasa en
las 10 celdas, pero pasa **porque el recorrido es enorme**: la elevación del brazo barre 34-137° en
un clip. Un brazo no sale 137° del plano de la imagen; eso es la amplitud del **ruido**, no de la
señal. Un criterio que aprueba precisamente cuando el ruido es grande no discrimina nada, y por eso
hacía falta declarar varios: el que sí discrimina es **C3**, que usa un patrón externo (la rigidez
del hueso) en vez de comparar la señal consigo misma.

**C4 falla también por un lado que no esperaba:** `s_col/s_fila`, o sea el tamaño de un píxel medido
por columnas contra por filas, sale **1,21-1,45**. En una cámara real el píxel es cuadrado y esa razón
vale 1. Que salga un 21-45 % de anisotropía significa que el mapa (X, Y, Z) de `pointmap` **no es la
nube de una cámara**, ni con una focal mal estimada: está deformado.

**Y C5 tuvo un bug que vale como escarmiento.** La primera versión medía la z de la mano *relativa al
vértice del cono*, y el vértice **es** la mano: la resta valía cero por construcción. C5 dio
**0,0 cm exactos en las 10 celdas** y «aprobó» 10 de 10 sin medir nada. Un criterio que sale
exactamente 0,000 en todas las celdas no es un éxito, es un aviso de que no está midiendo. Rehecho
con la Z absoluta —sin volver a pasar la GPU, porque el NPZ la guarda— el ruido real de la z de la
mano **con el brazo quieto** (menos de 1° de giro del eje 2D entre cuadros) es:

| clip | lado 0 | lado 1 | p90 | máximo |
|---|---:|---:|---:|---:|
| `AromaticShockedGermanspitz` | 5,6 cm | 4,8 cm | 20-35 cm | 81 cm |
| `ImpeccablePertinentBasilisk` | **22,2 cm** | 16,9 cm | 47-54 cm | 88 cm |
| `WelloffForcefulIndri` | 6,0 cm | 5,2 cm | 36-70 cm | **183 cm** |
| `SeparateHoarseFeline` | 14,2 cm | 16,5 cm | 46-57 cm | 93 cm |
| `IdealFrigidJerboa` | 7,6 cm | 4,4 cm | 44-61 cm | **377 cm** |

O sea: **la mediana (6,8 cm) sería utilizable, pero el p90 va de 20 a 70 cm en todas las celdas y el
máximo llega a 3,8 m, con el brazo quieto.** Un p90 de 44 cm sobre un alcance total de 90 cm no
permite decidir si la punta llega al torso. La mediana aprueba; la cola, que es la que estropea
cuadros concretos, no. Y es la cola la que importa, porque el cono se evalúa cuadro a cuadro.

**C3 es el que decide, y por eso se declaró antes.** El antebrazo es un hueso: su largo 3D es
constante y su proyección se acorta cuando apunta a la cámara. Si la z fuera profundidad de verdad,
añadirla tendría que **estabilizar** ese largo, siempre. Estabiliza en 5 de 20 celdas y lo
**desestabiliza hasta 3,2×** en el resto. Y el test del escorzo lo dice por otra vía: la correlación
entre el acortamiento proyectado y la diferencia de z mano-hombro tiene mediana −0,086 contra un
control con la z permutada de −0,003. **No hay escorzo en la z.**

### 22.5 · Y tres controles físicos gratis que lo confirman

**La focal no es identificable.** `pointmap` devuelve (X, Y, Z) por píxel; si eso fuera una nube de
cámara de proyección central, los tres estimadores de la focal tendrían que coincidir:

| estimador | focal | dispersión (p75−p25)/p50 |
|---|---:|---:|
| desde X, `(col − W/2)·Z/X` | 1841 px | 0,50 |
| desde Y, `(fila − H/2)·Z/Y` | 458 px | 1,11 |
| desde la escala local, `Z/s` | 439 px | 1,42 |

Discrepan **4×** entre ellos y ninguno baja del límite declarado de 0,25.

**La Z sale negativa** en el 11,4 % de las lecturas de `AromaticShockedGermanspitz`. Un punto delante
de la cámara no puede tener profundidad negativa.

**El antebrazo implica una cámara imposible.** El antebrazo mide ~0,25 m, así que la diferencia de
profundidad codo-mano no puede pasar de eso. De la fracción medida sale la distancia de cámara que
haría falta, y de ahí el campo de visión:

| clip | dif. relativa de z codo-mano | Z<0 | distancia implicada | **FOV implicado** |
|---|---:|---:|---:|---:|
| `AromaticShockedGermanspitz` | 0,204 | 11,4 % | 1,2 m | **144°** |
| `ImpeccablePertinentBasilisk` | 0,009 | 0 % | 27,2 m | **21°** |
| `SeparateHoarseFeline` | 0,045 | 0 % | 5,5 m | **70°** |
| `WelloffForcefulIndri` | 0,059 | 0 % | 4,2 m | **93°** |

Una cámara de televisión de esgrima usa 8-30°. **Sólo 1 de 4 clips es físicamente plausible.**

**Y el careo directo, que es la prueba más limpia.** La razón entre la profundidad del rival y la
propia es adimensional, así que se puede comparar con la que da la geometría del suelo:

| clip | dispersión `pointmap` | dispersión gratis | spearman | pendiente (debería ser 1) |
|---|---:|---:|---:|---:|
| `AromaticShockedGermanspitz` | 3,417 | **0,047** | +0,61 | **18,75** |
| `ImpeccablePertinentBasilisk` | 0,113 | 0,106 | +0,81 | 0,87 |
| `WelloffForcefulIndri` | 0,132 | 0,086 | −0,11 | **−0,13** |

En el primer clip `pointmap` es **73× más disperso** y amplifica la separación ×19. Pasa 1 de 3. Y ese
único que pasa tiene un confundido que hay que decir: allí la referencia «gratis» era el alto de caja,
y **un modelo de profundidad monocular infiere la profundidad del tamaño aparente igual que el alto de
caja**, así que su acuerdo no es una confirmación independiente.

### 22.6 · Lo que SÍ se puede tener, y es gratis: dos coordenadas exactas y una cota

Con la estatura nominal de 1,75 m y el alto aparente, la escala métrica en el propio tirador es
`S/h` m/px — que es exactamente lo que ya calcula `cono_lib.escala_px_m`. Y de ahí salen **dos de las
tres coordenadas sin ninguna incógnita**, en metros y en el sistema de la cámara. La tercera necesita
la focal, que es la incógnita.

**El plano del suelo, ajustado, hace robusta esa escala.** El alto aparente falla en un esgrimista
porque a fondo baja el centro de gravedad y su alto cae un 20-30 % sin moverse en profundidad —
medido: el alto de caja y la `y del pie` sólo concuerdan en el orden de profundidad el 82 % de las
veces. El suelo no se agacha, y «alto aparente contra fila del pie» es una recta cuya pendiente da la
altura de cámara y cuyo cero da el horizonte:

| clip | R² del ajuste | horizonte | **altura de cámara** | m/px suelo contra alto |
|---|---:|---:|---:|---:|
| `AromaticShockedGermanspitz` | 0,367 | fila −956 | **6,98 m** | discrepan 2,1-2,5 % |
| `WelloffForcefulIndri` | 0,316 | fila 456 | **1,59 m** | discrepan 4,3-4,4 % |
| `ImpeccablePertinentBasilisk` | 0,003 | −4105 | 25,9 m → **implausible** | el ajuste no vale |

Funciona en 2 de 3, y donde funciona la escala del suelo y la del alto coinciden al 2-4 %, lo que
respalda el `escala_px_m` que el proyecto ya usaba.

**Y la cota de elevación del arma con los 90 cm de la FIE**, que es la medida nueva y enteramente
gratis: si en la imagen se ve un tramo de hoja que llega a `a` cm de la mano, entonces el arma está
escorzada como mínimo en `a/90`, luego su elevación cumple `|elevación| ≤ arccos(min(1, a/90))`. Es
una **cota superior**, no una estimación, y por eso vale aunque la detección dé sólo un tramo parcial
de la hoja. Es el sustituto honrado de la elevación que `pointmap` no da.

### 22.7 · La consecuencia geométrica que sí justifica al usuario, cuantificada

La distancia **lateral** (medida, en metros) es una **cota inferior exacta** de la distancia 3D,
porque la profundidad sólo puede sumar. De ahí sale una propiedad estructural, no una medida:

> **El cono 2D nunca pierde una amenaza real; sólo puede sobrarle amenazas. Y el 3D, en el alcance,
> sólo puede quitar amenazas, nunca añadirlas.**

Medido sobre el **clip entero** (250-300 cuadros, sin `pointmap` y sin GPU), no sobre la ventana:

| clip | conos evaluables | con el torso a ≤ 90 cm | lateral mediana de esos |
|---|---:|---:|---:|
| `AromaticShockedGermanspitz` | 585 | **73 (12,5 %)** | 70 cm |
| `ImpeccablePertinentBasilisk` | 386 | 10 (2,6 %) | 81 cm |
| `WelloffForcefulIndri` | 540 | 29 (5,4 %) | 59 cm |
| `SeparateHoarseFeline` | 396 | **49 (12,4 %)** | 45 cm |

Sólo en esos 2,6-12,5 % de cuadros la pregunta del cono está viva. Y barriendo la única incógnita
(la distancia de cámara en estaturas), cuántos de los vivos saca el 3D de los 90 cm:

| distancia | cámara | `Aromatic` | `Impeccable` | `Welloff` | `Separate` |
|---:|---:|---:|---:|---:|---:|
| 3,0 estaturas | 5,2 m | 5,5 % | 40 % | 24 % | 82 % |
| **10,2 estaturas** | **17,9 m** | **53 %** | **100 %** | **83 %** | **100 %** |
| 23,2 estaturas | 40,7 m | 81 % | 100 % | 83 % | 100 % |

**O sea que la profundidad no es un detalle: a una distancia de cámara realista cambiaría el veredicto
en la mitad o más de los cuadros donde el 2D dice que amenaza.** Lo que falta es un solo número, y la
geometría monocular de una cámara **no puede darlo sin una longitud conocida en la escena**. Eso no es
un defecto del método: es un hecho.

Hay además una **cota física gratis**: estos clips salen del corpus etiquetado, luego cada uno
**tiene un tocado**, luego algún cuadro tiene que quedar dentro de los 90 cm en 3D. Cualquier
distancia que deje cero cuadros alcanzables es incompatible con que el clip tenga tocado. Sale un
techo de 35 / 6,8 / 35 / 4,1 estaturas según el clip — aprieta en dos y no aprieta en dos.

### 22.8 · El cono como DATO, con cuatro columnas de procedencia y no una

En `cono3d_completo_<clip>.npz`, junto a las poses, con la leyenda **dentro del propio NPZ**. Cuatro
procedencias distintas, porque son cuatro cosas distintas y confundirlas envenena el entrenamiento:

| columna | qué procedencia declara |
|---|---|
| `tipo` | la **CAJA**: 0 medida · 1 interpolada · 2 sostenida · 3 redetectada |
| `origen` | la **DIRECCIÓN** del eje: 0 hoja **VISTA** en el alfa · 1 **SUPUESTA** del brazo |
| `apert_origen` | la **APERTURA**: 0 temblor medido de la hoja · 1 regla de `e6`, mucho más ancha |
| `am2` / `am3` | la amenaza en 2D y en 3D, separadas y nunca fusionadas |

`origen` y `apert_origen` van separadas porque pueden diferir: si un clip no tiene bastantes pares de
cuadros con hoja detectada, su temblor propio no se puede medir y la apertura cae a la regla de `e6`
aunque la dirección sí se haya visto. **La primera versión resolvía eso devolviendo `None`, y era un
error grave y silencioso: tiraba justamente los conos de hoja vista, y sólo en los clips con pocas
detecciones, así que el sesgo dependía del clip y no aparecía en ninguna cifra.**

### 22.9 · Y no se prolonga la hoja desde la mano

Está medido y prohibido: el brazo apunta entre **24,8 y 29,7°** fuera de la dirección real de la hoja.
Lo que se hace cuando la hoja no se ve es declarar una **región** de 90 cm de radio con una apertura
que **cubre ese error medido**, no trazar una hoja. La diferencia importa: una hoja dibujada sería un
dato inventado indistinguible de uno medido, y ya hubo un cuaderno en este proyecto que trazaba 95 px
desde la muñeca y metía esa hoja inventada en la etiqueta de tocado. Las figuras de `e11` sólo pintan
el segmento de hoja **cuando se ha detectado**.

### 22.10 · LA COMPARACIÓN 2D CONTRA 3D, y el control que la decide

Montada para que **sólo cambie una cosa**: mismo vértice, mismo eje proyectado, misma apertura, mismo
L, mismo torso y la misma forma de peso `1 − ángulo/apertura`. Lo único que cambia es si la tercera
coordenada entra. Y la conversión degrada limpiamente: si la z fuera constante, el 3D coincidiría
**exactamente** con el 2D y no habría ni una discrepancia. Comprobado además que la proyección del
eje 3D devuelve la dirección medida con un residuo de **1,1e-16** (límite declarado 1e-9).

Ventana de acción, 50 cuadros por clip donde se buscó la hoja:

| clip | conos | hoja **VISTA** | amenaza 2D | amenaza 3D | **discrepan** | **control z permutada** |
|---|---:|---:|---:|---:|---:|---:|
| `AromaticShockedGermanspitz` | 63 | 2 (3,2 %) | 9 (14,3 %) | **0** | **9** (8 alcance, 1 dirección) | **9** → razón **1,00×** |
| `ImpeccablePertinentBasilisk` | 43 | 3 (7,0 %) | 4 (9,3 %) | 4 | **0** | **4** → razón **0,00×** |
| `WelloffForcefulIndri` | 92 | 1 (1,1 %) | 2 (2,2 %) | **0** | **2** (2 alcance) | **2** → razón **1,00×** |
| **total** | **198** | **6 (3,0 %)** | 15 | 4 | **11** (10 alcance, 1 dirección) | límite declarado **2×** |

**EL CONTROL ES EL RESULTADO DE TODA LA SECCIÓN.** Se rehace la comparación con las profundidades
**permutadas al azar entre cuadros**, dejando (x, y) y todo lo demás intacto, o sea cambiando
exactamente la variable que se juzga. Límite declarado antes de mirar: el 3D tiene señal si las
discrepancias con la z real son **al menos el doble** que con la z barajada. Sale:

- `Aromatic`: 9 con la z real contra **9** con la z barajada → **1,00×**
- `Welloff`: 2 contra **2** → **1,00×**
- `Impeccable`: 0 con la z real contra **5** con la barajada → la z barajada **inventa** discrepancias

**En ningún clip la z real hace algo que la z barajada no haga.** Así que las 11 discrepancias
existen, pero **no las produce la profundidad de la escena: las produce el ruido de `pointmap`.**

Y se ve en los números de las propias discrepancias, que son físicamente imposibles:

| caso | dist. 2D | dist. 3D | ángulo 2D | ángulo 3D | elevación |
|---|---:|---:|---:|---:|---:|
| `Aromatic` c229 l0 | 75 cm | **511 cm** | 4,4° | 34,5° | +42,6° |
| `Aromatic` c228 l0 | 76 cm | **485 cm** | 8,3° | 57,3° | +19,0° |
| `Welloff` c246 l0 | 84 cm | 109 cm | **0,1°** | 71,6° | −37,2° |

Dos esgrimistas a distancia de tocarse **no están a 5,11 m**. Y un eje que en la imagen apunta al
torso con 0,1° de desvío no está a 71,6° en la realidad. La elevación mediana del eje 3D sale entre
**−51,9° y +1,3°** según el clip, cuando un brazo armado apenas sale del plano de la imagen.

### 22.11 · La hoja por resolución, medida con el criterio correcto (y el mío estaba mal)

Declaré que 720p sería utilizable con una tasa de detección por encima del 10 %. **Ese criterio está
mal y los datos lo demuestran:**

| resolución | rectas pegadas a la mano | de ellas delante | **\|hoja − eje hombro-mano\|** | veredicto |
|---|---:|---:|---:|---|
| **1080p** (3 clips, agrupado) | 12 de 186 = **6,5 %** | 6 (50 %) | **23,1°** | por debajo del azar → **es la hoja** |
| **720p** (1 clip) | 8 de 62 = **12,9 %** | 7 (87,5 %) | **60,9°** | **por encima del azar** → no es la hoja |

O sea que a 720p el detector encuentra rectas a un ritmo **mayor**, y aprueba mi umbral del 10 %,
pero esas rectas **no son el arma**: su ángulo contra el eje del brazo es peor que el azar de 45°.
**Una tasa de detección se puede aprobar con ruido; lo único que distingue una hoja de una raya es la
ALINEACIÓN.** El criterio operativo pasa a ser el ángulo, y la tasa queda como dato descriptivo.

**Y mi hipótesis para explicarlo también falla, y va dicha.** Propuse que la variable que manda no
es la resolución nominal sino los PÍXELES POR METRO, porque la hoja tiene sección de 4-5 mm y su
anchura aparente es `0,005 × px/m`. **Los datos no lo sostienen:** el clip de 720p da **221 px/m**
(hoja de **1,10 px**) y los de 1080p van de **183 a 261 px/m** (hoja de **0,9 a 1,3 px**). O sea que
el ancho aparente de la hoja es **el mismo en los dos**, y aun así a 1080p las rectas se alinean
(23,1°) y a 720p no (60,9°).

Así que **la causa de la diferencia queda sin identificar.** Candidatos que no he medido: la
compresión del vídeo por píxel de contenido, el desenfoque de movimiento, o simplemente que el
720p es **un solo clip** y el 60,9° sea propiedad de ese clip y no de la resolución. Lo que sí queda
establecido es el hecho, no su mecanismo.

**Y en la ventana de acción la hoja casi no se ve, incluso a 1080p: 3,0 % de los conos** (6 de 198),
contra el 42,6 % que `e5` midió sobre cuadros repartidos por el clip. La diferencia es el desenfoque
de movimiento: la acción es justamente donde el arma va rápida. **Así que el cono con hoja vista no
está disponible donde más falta hace.**

Consecuencia sobre el corpus, con el censo propio (82,1 % de 720p, 17,9 % de 1080p, proyectado a
~7.023 y ~1.536 clips de 8.559): el cono con la dirección **medida** es construible en una fracción
pequeña de una sexta parte del corpus. **En el 97 % de los conos de esta sección la dirección se
supuso desde el brazo**, y en un clip entero (`WelloffForcefulIndri`) los 92 conos usaron incluso la
apertura de reserva traída de otros clips.

### 22.12 · Las figuras

`02_validacion/muestras/cono3d/`, 9 imágenes, cada una con dos vistas:

- **una por clip** (`cono3d_<clip>_c####_l#.png`): el fotograma con el cono 2D, el cuadrilátero del
  torso del rival, el brazo armado pintado —para poder comprobar a ojo que el cono arranca en la mano
  correcta— y el segmento de hoja **sólo si se detectó**. Al lado, la vista de **planta en
  centímetros** con los círculos de 30/60/90/120 cm, donde el de 90 cm es el alcance del arma.
- **seis de discrepancias** (`DISCREPA_solo2D_<motivo>_...png`), elegidas por ser las más extremas y
  repartidas entre clips, porque una discrepancia de 1 cm no la puede juzgar nadie a ojo y una de
  4 m sí.

El pie de las figuras de discrepancia **avisa de que el control de la z permutada dice que son
ruido**. Un rótulo que diera la razón al 3D sería peor que ninguno: el usuario mira la imagen
precisamente para poder contradecir el número.

### 22.13 · El coste, y por qué no sirve para proyectar

| paso | qué gasta | ms medidos |
|---|---|---:|
| `pose` 384×288, por tirador | GPU | 193-197 |
| `pointmap` de cuadro completo, por cuadro | GPU, bloqueado a 1024×768 | ~3.500-4.000 por cuadro y 2 tiradores |
| `matting` + `seg` por recorte | GPU | **1.362 a 4.931** |
| toda la geometría 3D, cono e intersección | CPU | despreciable |

**Estos tiempos NO son de referencia.** La GPU estuvo compartida con otros procesos todo el rato
—medida llena a **11,9 de 12,3 GB**— y eso hizo que `matting` costara hasta 4,9 s por recorte contra
los 767 ms que el proyecto mide con la GPU libre. Dos pasos murieron por falta de memoria sin dejar
traza. Por eso `e10` lleva un modo `CONO3D_SOLO_CACHE=1` que completa la geometría y las figuras en
CPU con los clips ya medidos, **diciendo por su nombre los que se salta**: `SeparateHoarseFeline` e
`IdealFrigidJerboa` quedaron sin fase A y **no entran en la tabla de 21.10**.

### 22.14 · El veredicto, en una frase

**El razonamiento del usuario es correcto —la intersección hay que decidirla en 3D— y `pointmap` no
puede darlo: sus discrepancias con el 2D son indistinguibles de barajar la profundidad al azar
(1,00×, límite declarado 2×).** Lo que queda en pie y es utilizable hoy es el cono 2D con la
distancia lateral **en centímetros**, que es cota inferior exacta de la 3D y por tanto nunca pierde
una amenaza real. Y lo que abriría el 3D de verdad no es un modelo más grande: es **un número por
clip**, la escala métrica, que la homografía de la pista puede dar con cifras de fuente primaria
(14,00 × 1,50 m, t.18). Eso está identificado y **no hecho**.

### LO QUE ESTA MEDIDA NO DICE

- **No dice que `pointmap` sea malo en general**, sólo que su z **no es profundidad métrica ni
  escorzo** en estas escenas: pabellón entero, 4-15 personas, tiradores de 200-300 px. Su tarjeta de
  modelo no menciona ambigüedad de escala y su dataset (`render_people`) es de humanos escaneados de
  uno en uno.
- **No dice que el 3D no sirva.** Dice lo contrario: el apartado 22.7 muestra que a una cámara
  realista el 3D cambiaría el veredicto en la mitad o más de los cuadros vivos. Lo que falta es **un
  solo número por clip**, y la vía para conseguirlo está identificada y con cifras de fuente
  primaria: la homografía de la pista sobre un rectángulo conocido de **14,00 × 1,50 m**, usando como
  puntos las marcas de los últimos 2 m (t.19.4), que son un **bloque de color** y no una línea fina y
  por eso sobreviven a la compresión y al desenfoque. **Eso no se ha hecho, así que no se afirma nada
  de su exactitud.**
- **No dice en cuántos clips se puede construir el cono con hoja vista más allá del censo.** Los
  cinco clips son 1080p, o sea el 18 % del corpus, y son **4 asaltos independientes** porque dos son
  del mismo. Cinco clips y cuatro asaltos no validan nada para 8.559 clips.
- **La tabla de 22.10 es de 3 clips, no de 5.** `SeparateHoarseFeline` e `IdealFrigidJerboa` se
  quedaron sin fase A porque la GPU estaba llena (11,9 de 12,3 GB con otros procesos) y sus procesos
  murieron sin traza. Van nombrados en `cono_e10_cono3d_accion.json` bajo la clave `saltados`, no
  escondidos. Y de `IdealFrigidJerboa` ya se sabía que no aportaría comparación: tiene **0 cuadros**
  con el rival dentro de los 90 cm.
- **El 720p es UN clip.** El segundo murió por la misma falta de memoria. Así que «a 720p las rectas
  son ruido» descansa en 62 recortes de un solo clip, y el mecanismo no está identificado.
- **El control de la z permutada es lo más fuerte de la sección, y también tiene su límite:** dice
  que la z de `pointmap` no aporta **por encima del azar en la decisión de amenaza**. No dice que la
  z sea inútil para todo, ni que otra forma de leerla (otra resolución de entrada, otro modelo, o
  `pointmap` sobre un recorte estabilizado) fracasaría igual. Se midió la vía de cuadro completo, que
  es la única que puede dar z comparable entre los dos tiradores.
- **No dice nada del tocado.** Ni la etiqueta de tocado ni el orden de las luces entran en ninguna
  medida de esta sección, salvo en un sitio y declarado: la cota física de la distancia usa el hecho
  de que el clip *tiene* un tocado, no *cuándo* ni *de quién*.
- **La razón de profundidades gratis no está validada contra verdad-terreno.** Su p10-p90 es ±5-9 %
  de la distancia de cámara, y que eso sea separación real o error de medida **no está establecido**:
  no hay referencia independiente. Es lo mismo que se le reprocha a `pointmap`, y vale igual aquí.
- **El test de ORDEN de profundidad no mide lo que parece.** Compara «quién está más cerca» entre
  tres señales, y ese booleano es casi constante dentro de un clip (uno de los dos va delante casi
  todo el asalto). Así que un acuerdo del 1 % no significa «se equivoca en el 99 % de los cuadros»,
  significa «tiene el signo contrario en este clip». Los valores que salen —alto contra pie 0,30 a
  0,96, `pointmap` contra pie 0,01 a 0,68— dicen que **las tres señales no concuerdan entre sí de
  forma estable**, y sin verdad-terreno **no se puede decidir cuál acierta**. Sólo vale como aviso de
  que ninguna es de fiar todavía, no para elegir entre ellas.
- **El ajuste del plano del suelo es frágil.** El mismo clip daba R² 0,367 y cámara a 6,98 m
  ajustando sobre la ventana quieta y **ningún ajuste válido** sobre la de acción. Se corrigió
  ajustando sobre **todo el clip**, que es donde la geometría de cámara es una propiedad real, pero
  que fuera sensible al tramo es en sí un aviso: no es robusto, es el mejor disponible sin calibrar
  la pista.
- **Los tiempos de esta sección no son de referencia.** La GPU estuvo compartida con otros procesos
  todo el rato (hasta 10 a la vez), así que los ms/cuadro que imprimen `e9`-`e13` son indicativos y
  no sirven para proyectar coste.

---

## 23. La PISTA en metros: los bordes sí, la escala en el 13 % de los clips

El usuario lo pidió así: *"la pista también hay que detectarla... para tener trackeada la posición
que hacen sobre la pista en directo... como una línea en 2D por donde va, en el eje de la LONGITUD
de la pista y la PROFUNDIDAD de la pista... y cada vez que hace un fondo ponerlo; cada vez que
estira el brazo marcarlo como un keypoint"*.

Código en `01_extraccion/codigo/p1_censo_pista.py`, `pista.py` y `posicion.py`; figuras, CSV y
recortes de verificación en `02_validacion/pista/`. Herramientas de figura y de verificación a ojo
en `02_validacion/herramienta/grafica_pista.py` y `verifica_pista.py`.

### 23.0 Esto ya se había cerrado en 2026, y la cifra de entonces se reproduce

El informe histórico (`99_historico/.../paper/main.tex`, `sec:abl-geometry`) medía **3,8 %** de
cuadros calibrados sobre 500 cuadros de puesta en guardia de 19 sedes, con un techo de humano
asistido de **4,3 %** sobre las mismas tiras rectificadas, y lo cerró como problema de DATOS. El
censo de hoy lo reproduce: de 13 clips, **1 (8 %)** tiene dos o más LÍNEAS transversales completas
detectables con un detector de segmentos genérico.

Lo que esta sección añade es que el veredicto de entonces era correcto **para la vía que midió** y
no para todas: los BORDES longitudinales de la pista sí se detectan, y casi siempre.

### 23.1 La geometría FIE, y dos ediciones del reglamento que NO dicen lo mismo

Comprobado línea a línea en los dos ficheros que hay en `scratchpad/`, no de memoria:

| edición | t.18.1 |
|---|---|
| `fie_tech.txt`, **diciembre 2017** | *"The piste is from 1.50 metres to 2 metres wide."* |
| `fie_2021.txt`, **diciembre 2021** | *"The combat area of the piste is 1.50 meters wide."* |

**La que aplica a este corpus es la de 2021: el ancho es 1,50 m, no un rango.** La diferencia no es
un cambio de medida: la de 2021 añade en t.18.3 que los *conductive safety borders* **no forman
parte de la pista**. O sea que la banda que se ve en el vídeo puede ser más ancha que la pista.

Y **el reglamento no define ninguna "línea de aviso" a 5 m**: t.19 dice que los últimos 2 m se
distingan *"if possible by a different colour of piste"*. La marca a 5 m del centro existe, pero
como **frontera de color**, no como raya pintada. Un detector de rectas no la encuentra nunca.

### 23.2 Dos de mis propias métricas del censo estaban sesgadas, y las dos daban CERO

Se cuentan porque el cero que fabricaban encajaba sospechosamente bien con el antecedente histórico.

| métrica | primera versión | por qué era falsa | corregida |
|---|---:|---|---:|
| líneas transversales | **0 de 13** | pedía 0,10 del ancho de IMAGEN; una transversal mide 0,05-0,09 | **1 de 13** |
| separación de color pista/suelo | 0,44 en `UglyNice` | bandas por FILAS de imagen con el eje a -13,5° | **1,69** |

### 23.3 Lo que SÍ se detecta, sobre 78 clips del corpus etiquetado

| evidencia | clips | cifra |
|---|---:|---|
| los DOS bordes longitudinales | **78 de 78 (100 %)** | residuo de recta p50 **5,45 px**, y 1,69 px en los publicables |
| al menos una marca transversal medida | **74 de 78 (95 %)** | p50 **4 marcas** por clip |
| zona de otro color detectada | 50 de 78 (64 %) | pero ver 23.6: casi nunca es la del reglamento |
| homografía plausible como cámara | **43 de 78 (55 %)** | altura de cámara en 1,2-15 m |
| **pasa la puerta declarada** | **10 de 78 (13 %)** | 720p 9/64 (14 %), 1080p 1/14 (7 %) |

La puerta se declaró ANTES de correr: la homografía se publica sólo si la **estatura reproyectada**
de los dos tiradores cae en **1,60-2,00 m**.

### 23.4 El error de la homografía, en metros

El control C2 es el único independiente que hay: reproyecta las separaciones entre TODAS las marcas
detectadas y las compara con las separaciones posibles del reglamento (2, 3, 4, 5, 7, 9, 12, 14 m).

| | valor |
|---|---:|
| error de C2, mediana de los 10 publicables | **0,29 m** (p10 0,14, p90 0,43) |
| lo mismo en % de la separación de guardia de 4,00 m | **7,3 %** |
| ancho deducido de la estatura | p50 **1,42 m** (p10 1,34, p90 1,60) contra 1,50 reglamentario |
| sesgo sistemático de escala que implica | **+5,7 %** |
| largo de pista VISIBLE en el cuadro | p50 **5,53 m** de los 14 |
| desplazamiento de cámara dentro del clip | p50 **161 px**, p90 **479 px** |

**El desplazamiento de cámara obliga a corregir cuadro a cuadro**, y se hace: cada punto pasa por la
similitud de estabilización de su cuadro antes de la homografía. Con 479 px de paneo, una sola
homografía por clip no vale.

### 23.5 Y el control cruzado entre dos vías independientes NO sale

Cuando dos vías (rayas del suelo y geometría de cámara) están disponibles a la vez, se mide el mismo
tramo de 4,00 m con las dos. Sobre los 3 clips donde ocurre, la razón entre las dos escalas sale
**0,28 · 0,76 · 2,57**. No es que difieran un poco: difieren un factor 3.

Ésa es la medida más dura de esta sección y hay que leerla así: **la escala longitudinal no está
establecida**. Los 10 clips publicables lo son porque su estatura reproyectada cae en rango, y la
estatura es la PUERTA, no un control independiente. El único control independiente es C2, con
0,29 m de error.

### 23.6 Tres vías probadas, y por qué las dos "buenas sobre el papel" fallan

| vía | qué necesita | resultado medido |
|---|---|---|
| **A · rayas identificadas** + `findHomography` | 4 marcas identificadas | con 3 marcas la identificación es azar: 126 hipótesis con 0,20 de acierto por azar cada una. `GentleQuerulous` identificaba `[-7,-5,0]` y `UglyNice` `[-7,-2,2]`, las dos con error 0,00 m **por construcción**, y las dos daban residuo DLT de 206 px y estaturas de 0,93 m |
| **B · geometría de cámara** | punto de fuga vertical de la sala | la que gana en 8 de los 10 publicables |
| **C · zona de color de t.19** | segmentar los últimos 2 m | se detecta en 50 de 78, pero implica cámaras a **28,9 · 86,1 · 89,9 m** de altura: lo que el detector encuentra es una sombra, una moqueta o una valla, no la zona del reglamento |

La vía B tiene una particularidad que costó tres intentos entender: **con la cámara casi
perpendicular al eje de la pista el sistema está un grado de libertad corto**, y se demuestra
contando. La pista da tres datos (fila del borde de lejos, fila del borde de cerca, fila del
horizonte) y hay cuatro incógnitas (focal, inclinación, altura de cámara y distancia a la pista); el
ancho fija la escala, no la focal. Y la sensibilidad no es pequeña: resolviendo a mano con f = 1200
y con f = 1600 px sobre las mismas tres filas, **la escala longitudinal cambia un 24 %**. El grado
que falta lo aportan las VERTICALES fijas de la sala (postes, sillas, paneles), que dan el punto de
fuga vertical con residuo angular de 0,10-0,40° y a 11-16 px de la columna del punto principal.

### 23.7 La posición de los tiradores en metros: el TEMBLOR es pequeño, la profundidad no vale

El apoyo se toma de los tobillos y pies de la pose de 308 puntos (índices 13-14 y 15-20) y el borde
inferior de la caja como respaldo.

| medida | valor | criterio declarado antes | veredicto |
|---|---:|---|---|
| temblor p95 a lo LARGO de la pista | **0,025 m** (0,010-0,081) | < 0,13 m, que es lo que un fondo real recorre por cuadro | **pasa con 5x de margen** |
| temblor p95 a lo ANCHO | **0,043 m** (0,015-0,067) | idem | pasa |
| salto bruto p95 entre cuadros | 0,134 m | incluye movimiento real, no es ruido | — |
| saltos imposibles (> 0,30 m/cuadro) | **1,1 %** (0-4,5 %) | — | — |
| recorrido TRANSVERSAL de un tirador | p50 **0,87 m** (0,37-1,96) | < 0,60 m: un floretista no cruza la pista a lo ancho | **falla en 8 de 10** |
| separación de pies, mediana del clip | p50 **0,82 m** (0,58-1,48) | 0,50-0,90 m, la guardia de un floretista | pasa en 2 de 4 medibles |
| apoyos fuera de la pista | p50 **0,1 %** | < 5 % | pasa en 8 de 10 |
| **los tres controles a la vez** | | | **1 de 10** |

**Conclusión de este paso, y es la que hay que retener: la coordenada A LO LARGO de la pista es
creíble y la TRANSVERSAL no.** Tiene explicación física: la transversal es la dirección de
profundidad de la cámara, donde un error de una fila de píxeles se convierte en decímetros, y donde
un pie EN EL AIRE se proyecta desplazado hacia el fondo por
`altura_del_pie x distancia / altura_de_camara`.

### 23.8 Los eventos, con su magnitud en metros

114 eventos sobre los 10 clips publicables. Los de pies y brazo necesitan la pose de 308 puntos: en
los 6 clips sin pose sólo sale `accion`.

| evento | n | magnitud p50 | criterio declarado |
|---|---:|---:|---|
| **fondo** | 30 | **0,64 m** | separación de pies +0,35 m en 8 cuadros Y pie de delante +0,25 m |
| **marcha** | 11 | 0,12 m | un pie a > 0,9 m/s con el otro a < 0,45, ≥ 3 cuadros, hacia el rival |
| **romper** | 22 | 0,15 m | ídem alejándose |
| **brazo** | 30 | 1,15x torso | eje `codo -> raíz de la mano` por encima de 1,15 veces su mediana |
| **acción** (inicio y fin) | 21 | 0,80 m | centro a > 0,5 m/s hasta 10 cuadros por debajo |

**Verificado a ojo, y con n pequeño:** de 3 eventos marcados que inspeccioné en recorte, los 3 eran
correctos (`GentleQuerulous` c287 fondo completo con 0,49 m, c205 marcha del lado derecho). Y de 2
candidatos a falso negativo, **1 era un fondo real perdido** (c239, fondo completo con los pies a
1,48 m) y 1 no era un fondo (guardia ancha).

**Ese falso negativo era un fallo mío contra mi propio criterio declarado**: la cabecera decía "el
pie de delante avanza 0,25 m" y el código medía el avance del CENTRO de los dos pies, que en un
fondo avanza la mitad. Corregido, `GentleQuerulous` pasa de **3 a 6 fondos** y sus candidatos a
falso negativo de **9 a 4**. Sólo se vio sacando el recorte del falso negativo; ninguna cifra
agregada lo delataba.

Candidatos a falso negativo después de fusionar los consecutivos: **4 a 8 fondos** y **6 a 50 pasos**
por clip. Son una cota superior, no falsos negativos medidos.

### 23.9 LO QUE ESTA SECCIÓN NO DICE

- **No dice que la pista esté calibrada.** Dice que en el 13 % de los clips la homografía pasa una
  puerta de plausibilidad física. El control independiente que le queda (C2) da 0,29 m de error, y
  el control cruzado entre dos vías independientes **no sale** (razones de 0,28 a 2,57).
- **No dice dónde está la línea central.** El origen longitudinal es la marca que se usó para el
  ajuste, y esa marca **no está identificada**. Así que los DESPLAZAMIENTOS y las DISTANCIAS están
  en metros, pero la posición ABSOLUTA respecto al centro de la pista no. Todo lo que el usuario
  pidió medir ("cuánto se ha desplazado", "cuánto ha medido el fondo", la distancia entre los dos)
  son desplazamientos; "dónde de la pista" no está resuelto.
- **No dice que las pistas del corpus estén fuera de medida.** El ancho deducido sale 1,42 m de
  mediana contra 1,50 reglamentario, y eso es indistinguible de un sesgo del 5,7 % en la escala.
  Para separar las dos cosas haría falta un clip con dos marcas identificadas, y no hay ninguno.
- **El largo total de la pista no es observable.** El cuadro abarca 5,53 m de mediana de los 14, así
  que la pregunta del usuario sobre pistas con plancha extra **no se puede contestar con un clip**.
- **Los 78 clips no son 78 asaltos.** `WelloffForcefulIndri` e `IdealFrigidJerboa` son el mismo
  asalto (BIBARD-OISHI), así que el n efectivo es 77 y puede haber más duplicados sin detectar.
- **La verificación a ojo de los eventos es n=5.** Tres marcados y dos candidatos. No es una tasa de
  acierto: es lo que se pudo mirar.
- **El resultado depende de la muestra de la placa.** Bajando los cuadros de la mediana temporal de
  48 a 28 por coste, la estatura de `GentleQuerulous` pasó de 1,91 a 1,82 m. Un cambio de
  implementación que mueve el resultado un 5 % es un aviso de que el ajuste no es robusto.
- **La zona de color de t.19 sigue siendo la vía más prometedora y no está agotada.** Lo medido es
  que MI segmentador de color encuentra otra cosa en la mayoría de los clips, no que la zona no
  sirva. Con `seg` de Sapiens 2 para borrar personas y un segmentador de región en vez de un umbral
  de color robusto, merece otro intento.

### 22.15 · CORRECCIÓN DE FONDO: el cono va contra el OTRO CONO, no contra el torso

**El planteamiento de 22.7 a 22.14 medía el cono contra el TORSO del rival, y eso venía de un encargo
equivocado.** Lo corrigió el usuario: *"lo de los conos no es con el torso, es con el OTRO CONO,
¿sabes? Pero bueno, me sirve. E intenta mejorarlo, que no está demasiado bien"*.

Y no es un matiz, porque las dos cosas miden sucesos distintos:

| intersección | qué mide |
|---|---|
| cono contra **torso** | dónde puede haber **amenaza a la superficie válida** |
| cono contra **cono** | dónde puede haber **CONTACTO ENTRE LAS DOS HOJAS** |

Lo segundo es la señal que le falta al proyecto entero: el detector de prioridad se cerró con **0
paradas anotadas** y la parada es la bisagra que pasa la prioridad a la respuesta (t.85 y ss.), así
que cinco de las doce reglas se quedan sin medir. Lo medido en 22.7-22.14 **no se retira** —contesta
la pregunta de la amenaza, que también hace falta— pero no contestaba ésta.

Código en `e15_cono_contra_cono.py`, `e16`…`e19`. **Todo sin GPU**: los conos salen de los NPZ de
poses. Y `cono_lib.cruce` ya calculaba la intersección de dos conos desde `e3`, así que se llama, no
se reimplementa; lo único que cambia respecto a `e3` es el eje (`hombro -> raíz` en vez de
`codo -> raíz`).

### 22.16 · La primera pregunta: ¿informa de algo? Sí, y eso refuta la hipótesis del cono demasiado ancho

Si los dos conos se solaparan casi siempre, «se solapan» no distinguiría un cuadro de otro. Criterio
declarado antes de medir: **informa si la frecuencia está entre el 5 % y el 80 %**.

| apertura | solape | | apertura | solape |
|---:|---:|---|---:|---:|
| 4° | 13,9 % | | 18° | 32,2 % |
| 8° | 20,5 % | | **23,1°** | **34,5 %** |
| 12° | 26,7 % | | 30° | 37,2 % |

A la apertura operativa de 23,1°, el solape es **34,5 %** agrupado (8,4 a 57,6 % por clip, 7 clips,
1.531 cuadros). **Está dentro de la banda, así que el cono NO está saturado y «demasiado ancho» no es
el defecto principal.** Y el barrido dice algo más: de 4° a 30°, o sea multiplicando la apertura por
7,5, el solape sólo pasa de 13,9 % a 37,2 %. **La apertura apenas controla el solape**; lo controla la
geometría de los ejes.

**Pero la línea base gratis muerde.** `spearman(peso_max, −distancia entre las dos manos)` sale de
**+0,55 a +0,96** (mediana ~0,73), y contra la distancia entre tiradores de +0,13 a +0,96. O sea que
el solape de conos está en buena parte explicado por lo cerca que están las manos, que cuesta 0 ms y
son dos puntos de la pose.

### 22.17 · EL DEFECTO DE FONDO: el candidato de contacto cae en el SUELO el 63,6 % de las veces

Esto es lo que estaba mal de verdad, y se ve en las figuras. El eje es `hombro -> raíz de la mano`, y
**en guardia la mano va por debajo del hombro**, así que el eje apunta hacia abajo. Es correcto como
dirección del BRAZO y falso como dirección de la HOJA, porque la hoja sale de la muñeca hacia
delante, no del hombro hacia la mano.

Medido, sin necesidad de anotación, porque es una imposibilidad física (un contacto de hojas no ocurre
por debajo de las dos manos, ahí sólo hay piernas y suelo):

| clip | candidatos | por debajo de las dos manos | ángulo del eje |
|---|---:|---:|---:|
| `AliveIgnorantBarebirdbat` | 24 | **95,8 %** | +84° |
| `BlissfulTiredAiredale` | 64 | **93,8 %** | +28° |
| `ImpeccablePertinentBasilisk` | 13 | 84,6 % | +34° |
| `IdealFrigidJerboa` | 71 | 59,2 % | +18° |
| `WelloffForcefulIndri` | 144 | 56,9 % | +2° |
| `SeparateHoarseFeline` | 79 | 55,7 % | +10° |
| `AromaticShockedGermanspitz` | 133 | 55,6 % | +37° |
| **total** | **528** | **63,6 %** | mediana **+20°** |

El **32,5 %** de los ejes apuntan francamente hacia abajo (entre 20° y 160°). En
`conocono_AromaticShockedGermanspitz_c0239.png` se ve el caso puro: `peso_max` = **1,00**, o sea el
máximo posible, y la cruz del candidato cae **entre las rodillas de los dos tiradores**.

**Consecuencia: `peso_max` alto NO significa contacto de hojas.** Es la explicación de por qué la
línea base de distancia entre manos lo explica tan bien: el solape mide proximidad, no dirección.

### 22.18 · Los otros cuatro defectos, cada uno con su cifra

`e17`, sobre 3.418 tiradores-cuadro de 7 clips. Todas son contradicciones internas o imposibilidades
físicas, así que **ninguna necesita verdad-terreno**:

| | defecto | medido |
|---|---|---:|
| **D1** | el vértice está pegado al tronco: no es una mano extendida | **2,2 %** |
| **D2** | la mano de un tirador cae DENTRO de la caja del otro | **3,0 %** |
| **D3** | la otra mano está más cerca del rival que la elegida como armada | **8,7 %** |
| **D4** | el eje se va con el escorzo: salto del eje entre cuadros | **3,06°** con brazo corto contra **0,93°** con brazo largo = **2,22×** |

D3 es el más frecuente y era esperable: `cono_lib.mano_armada` ya declara que descansa en **un solo
criterio geométrico validado en 4 tiradores**. Visto en `L10`, donde el tirador se lanza a fondo y la
marca cae en su brazo TRASERO. D1 y D2 se ven en `L01` y `L06`, donde **las dos marcas caen en el
mismo tirador** y una de ellas en la cadera.

### 22.19 · El arreglo de D4: la apertura que crece con el escorzo. Funciona, y es pequeño

Si el eje tiembla más cuando el brazo está escorzado, la apertura constante promete una precisión que
el eje no tiene justo ahí. Arreglo declarado: sumar en cuadratura los dos errores independientes,

    apertura(k) = raíz( 23,1² + temblor_p90(largo_proyectado(k))² )

con `temblor_p90` medido **por quintiles en el propio clip**, así que no entra ninguna constante nueva.
Juzgado sin verdad-terreno por el **parpadeo** del booleano de solape entre cuadros consecutivos (un
contacto real dura varios cuadros, un parpadeo es ruido):

| | solape | parpadeo |
|---|---:|---:|
| constante 23,1° | 35,9 % | **3,5 %** |
| **adaptativa** (media 24,2-24,7°) | 35,9 % | **2,9 %** |
| control: constante subida al mismo valor medio | 35,9 % | 3,5 % |

Baja el parpadeo **0,6 puntos**, sigue informando (35,9 %, dentro del 5-80 %) y **bate al control**,
o sea que lo que ayuda es «apertura donde hace falta» y no «más apertura». Reparto por clip: **3
mejoran, 3 empatan, 1 empeora**.

**Y es un arreglo pequeño, con su razón:** el temblor del eje es de 4-8°, que en cuadratura con los
23,1° de base sólo mueve la apertura de 23,1 a 24,2. **El cuello de botella no es el temblor del eje,
es el error de 23,1° entre el brazo y la hoja**, y ése sólo baja viendo la hoja — que es lo que 22.11
dice que no se puede en el 82 % del corpus. Referencia previa: `e3` midió este mismo parpadeo con el
eje del codo en el 3,5-4,9 %.

### 22.20 · La validación contra verdad-terreno NO se pudo hacer, y por qué

No hay paradas anotadas en el proyecto, así que hice la anotación yo (`e16`), **a ciegas**: láminas
sin conos, sin nombre de clip y sin `peso_max`, en orden aleatorio con semilla fija, y la clave en un
fichero aparte. Muestreo estratificado, 11 con `peso_max` > 0,20 y 11 con `peso_max` = 0, para poder
medir acierto **y** falsos negativos.

**Resultado: de 16 láminas inspeccionadas, 0 decidibles como «sí hay contacto».** 6 fueron `no`
(tiradores claramente lejos) y **10 `nose`**. La anotación está en
`02_validacion/muestras/contacto_hojas/anotacion.csv`, con las 6 no inspeccionadas marcadas como
tales y no como `no`. Las causas del `nose`, que son el hallazgo:

- **el monitor del marcador está en primer plano entre la cámara y la pista**, tapando exactamente la
  región donde se cruzarían las hojas (`L00`, `L07`, `L09`, `L17`)
- **desenfoque de movimiento** justo en los cuadros de manos cercanas, que son los que importan
  (`L02`, `L04`)
- la hoja **no se resuelve a ojo** ni con vista limpia y sin oclusión (`L11`)
- y en varias, el defecto D1/D2/D3: las marcas no estaban en las dos manos correctas

**Sin un solo positivo no se puede medir ni precisión ni falsos negativos del proxy.** Y hay que
notar de dónde vienen los `nose`: el muestreo era mitad y mitad, así que **10 de los `nose` salieron
en buena parte del estrato de `peso_max` alto**, que es justo donde el proxy afirma contacto. O sea
que lo indecidible no es el caso raro, es el caso que había que comprobar. Eso deja el
solape cono-cono como **columna cruda sin validar**, igual que `e3` dejó la intersección: reproducible
y medible, pero sin saber si predice lo que dice predecir.

Dos avisos sobre la anotación, que la hice yo: **no soy árbitro FIE**, y **inspeccioné 16 de las 22
láminas**; paré al ver que no aparecía ningún positivo decidible y redirigí el esfuerzo a medir los
defectos, que sí se pueden medir sin anotación. Las 6 restantes van marcadas como no inspeccionadas
en el CSV, para que nadie las cuente como negativos. Las 22 láminas y la plantilla quedan en
`02_validacion/muestras/contacto_hojas/` para que las mire alguien que sepa más.

### 22.21 · La asimetría que sí se puede afirmar sin medir profundidad

Dos conos que **no** se cruzan en la proyección tampoco se cruzan en el espacio. Luego:

> **El solape 2D es COTA SUPERIOR del solape 3D.** Un «no se solapan» en 2D es un «no hay contacto»
> firme; un «sí se solapan» es sólo un candidato.

Es la simétrica de lo que ya se dijo de la amenaza: la distancia lateral es cota **inferior** de la
3D, así que el cono 2D nunca pierde una amenaza real. Las dos afirmaciones son geometría de la
proyección y **no necesitan que `pointmap` funcione**, así que sobreviven al veredicto de 22.4.

### 22.22 · Figuras del cono contra cono

`02_validacion/muestras/cono_cono/`, 32 imágenes con los dos conos, la **región de solape sombreada** y
la cruz del candidato de contacto:

- **7 por clip**, el cuadro de `peso_max` más alto de cada asalto
- **25 de PARPADEO**, pares de cuadros consecutivos donde el booleano de solape cambia, lado a lado

No hay figuras de «mi anotación dice contacto y el solape dice que no» porque **no hay ningún positivo
anotado** (22.20). Va dicho en el docstring de `e19` en vez de sustituirlo por otra cosa parecida.

### LO QUE ESTA PARTE NO DICE

- **No dice que el solape cono-cono no sirva.** Dice que **tal como está construido no localiza el
  contacto**: el 63,6 % de sus candidatos caen bajo las dos manos. El arreglo que apunta la medida es
  cambiar el eje, y la dirección correcta sale de la muñeca o de la hoja, no de `hombro -> raíz`.
  Eso no se ha probado.
- **No dice que el proxy acierte ni que falle contra el contacto real**, porque no hay verdad-terreno:
  0 positivos decidibles en 15 láminas.
- **7 clips y 4 asaltos independientes.** Dos de los clips de 1080p son el mismo asalto, y dos de los
  7 son 720p. No es una muestra del corpus.
- **El parpadeo no es una medida de acierto**, es de estabilidad. Un proxy puede ser perfectamente
  estable y perfectamente equivocado — y de hecho el de 22.17 lo es.

---

## 24. Los tres defectos del cono que el usuario vio: dos eran uno, y el tercero estaba al revés

El usuario lo dijo así: *"lo de la mano dominante le pones un florete pero no es el cono que dijimos,
es recto, un palo. Revísalo. Además es muy corto, no mide como debería medir un florete. Y luego el
cono tiene que ser más estrecho, que es demasiado gordo ahora creo. Hazlo mejor"*.

**Los tres son ciertos y ninguno era lo que parecía.** Código en `e20_estado_del_cono.py` (la medida,
antes de tocar nada) y `e21_pinta_cono_arreglado.py` (el arreglo y las figuras). Figuras en
`02_validacion/muestras/cono_arreglado/`. **Todo de los NPZ y de los JSON de hoja ya cacheados: cero
GPU, cero modelos, y 6 fotogramas decodificados en total** — el PC estaba saturado y esto no necesita
más.

### 24.0 Los tres criterios, declarados antes de ejecutar

| | criterio | número que decide |
|---|---|---|
| **C-EJE** | el eje cambia a `codo -> raíz` si baja la fracción de candidatos de contacto que caen bajo las dos manos | base dada **63,6 %** · mejora declarada **≤ 31,8 %** (la mitad) |
| **C-APER** | la apertura baja si baja el error medido contra la hoja de verdad | margen declarado de `e14`: **5°**. Por debajo, empatan y no se toca |
| **C-LARGO** | no hay criterio que inventar: `L = largo * px/m` es aritmética. Lo que se mide es **de qué depende `px/m`** | — |

### 24.1 «Es un palo»: eran DOS palos y sólo uno era el cono

Leído el código antes de medir: **ninguno de los tres scripts de dibujo pintaba el cono relleno**, los
tres lo pintaban en **contorno** (`cv2.polylines`). Pero el objeto que de verdad parecía un florete
era otro: `e19` pintaba además el **brazo** (`hombro -> raíz`) con **la misma color del cono y grosor
3**. Un segmento recto y grueso que sale de la mano en el color del arma *es* un florete dibujado.

Arreglado en `e21`: el cono va **relleno semitransparente** y el brazo va en **línea de trazos, fina y
gris**, que sigue permitiendo comprobar a ojo de qué antebrazo sale el eje y no se puede confundir con
un arma.

Y el relleno **no es un tono plano: es la función de peso que usa la medida**, `1 - ángulo/apertura`,
la misma de `cono_lib.peso_sector` y de `cruce`. Eso tiene dos consecuencias y las dos importan: la
figura y el número no pueden discrepar porque son la misma función, y el cono **se ve estrecho sin
recortarlo**, porque se apaga solo hacia el borde. Es la única forma de dar a la vez «más estrecho» y
«no perder amenazas».

### 24.2 «Es muy corto»: no venía de un largo fijo, venía de medir la hoja SOLA

Comprobado leyendo el código: los tres scripts ya usaban la escala, `L = HOJA_M * pxm`. **El largo
corto no venía de una constante en píxeles.** Venía de que `HOJA_M` son los **90 cm de la hoja**
(m.8.4), que se miden **desde la cara delantera de la guarda**, y el cono sale de la **raíz de la
mano**, que es la muñeca y está por detrás de la guarda, con la empuñadura en medio.

La cota sale de dos artículos y no hay que estimar nada:

| | |
|---|---|
| m.8.4 hoja ≤ 90 cm | la muñeca **no** puede estar delante de la guarda |
| m.7 arma montada ≤ 110 cm | la muñeca **no** puede estar detrás del pomo |
| **luego alcance muñeca-punta ∈ [90, 110] cm** | los dos extremos son reglamentarios |

Se toma el **extremo alto**, `cono3d_lib.ALCANCE_M = 1,10`, y la razón no es cosmética: el cono es una
región que tiene que **contener** la amenaza. Una cota por lo bajo del alcance rompe exactamente eso —
con `L` corto la punta llega más lejos que el cono y **la amenaza se pierde en silencio**. La nota de
`CL.HOJA_M` decía que 90 cm era «conservador para decir amenaza»: es cierto para *no inventar* una
amenaza y es lo contrario de conservador para *no perderla*. Son dos errores opuestos y el cono sólo
puede protegerse de uno; se elige no perder, que es lo que 22.7 ya había declarado.

`CL.HOJA_M` **no se toca** (lo usan `e3`-`e19`) y el arco de 90 cm se sigue pintando dentro, a trazos
blancos, para no perder la referencia.

| clip | res | px/m (l0/l1) | L a 0,90 m | **L a 1,10 m** | `p95/p50` del alto de caja |
|---|---|---|---:|---:|---:|
| `AliveIgnorantBarebirdbat` | 720 | 220,9 / 220,8 | 199 px | **243 px** | 1,112 / **1,306** |
| `AromaticShockedGermanspitz` | 1080 | 260,8 / 247,6 | 229 px | **280 px** | 1,143 / 1,129 |
| `BlissfulTiredAiredale` | 720 | 205,2 / 192,6 | 179 px | **219 px** | 1,144 / 1,151 |
| `IdealFrigidJerboa` | 1080 | 233,3 / 221,9 | 205 px | **250 px** | 1,157 / 1,162 |
| `ImpeccablePertinentBasilisk` | 1080 | 183,4 / 195,8 | 171 px | **209 px** | 1,159 / 1,164 |
| `SeparateHoarseFeline` | 1080 | 258,4 / 235,2 | 222 px | **271 px** | 1,150 / 1,086 |
| `WelloffForcefulIndri` | 1080 | 230,3 / 207,0 | 197 px | **241 px** | 1,102 / 1,078 |

**Y queda un segundo motivo de acortamiento, medido y NO corregido**, que va dicho porque es el que
más puede pesar: `escala_px_m` divide el alto de la caja por una estatura de 1,75 m, y **un esgrimista
en guardia no mide su estatura**. La razón `p95/p50` del alto de caja dentro del clip vale **1,08 a
1,31, mediana 1,15**: en el cuadro típico la referencia de escala está un **15 % encogida**, luego
`px/m` sale un 15 % corto y `L` con él. Concuerda con lo que 22.6 ya midió por otra vía (el alto de
caja cae 20-30 % a fondo sin moverse en profundidad). **No se corrige aquí** porque `escala_px_m`
alimenta toda cifra en centímetros ya publicada en 22 y cambiarla la movería en silencio. Queda
identificado, cuantificado y sin hacer.

### 24.3 El eje: de `hombro -> raíz` a `codo -> raíz`, con dos medidas independientes

Y detrás hay una regla general que conviene que quede escrita: **el largo de un segmento y su
dirección contestan preguntas distintas, y el mejor segmento para una puede ser el peor para la
otra.**

| pregunta | segmento bueno | por qué |
|---|---|---|
| **CUÁNTO** se extiende el brazo | `hombro -> raíz` | crece al extenderse: cv del largo proyectado **0,226** contra 0,108 |
| **HACIA DÓNDE** apunta la hoja | `codo -> raíz` (antebrazo) | **la hoja continúa el antebrazo**, no el brazo entero |

`cono3d_lib.EJE_POR_DEFECTO` ponía `hombro` razonando con la tabla de cv, o sea **decidiendo una
pregunta de dirección con una medida de magnitud**. El razonamiento del cv sigue siendo correcto para
lo que medía; lo que estaba mal era aplicarlo al eje de orientación.

**Medida 1, el candidato de contacto** (el `peso_max` de `cruce`, la cruz de las figuras de `e19`),
7 clips, apertura 23,1° en los dos casos para que sólo cambie el eje:

| clip | candidatos bajo las dos manos | punta bajo el tobillo propio | elevación mediana del eje |
|---|---|---|---|
| | hombro → **codo** | hombro → **codo** | hombro → **codo** |
| `AliveIgnorantBarebirdbat` | 95,8 % → **52,2 %** | 15,4 → **5,7 %** | −38,3 → **−9,8°** |
| `AromaticShockedGermanspitz` | 55,6 % → **18,8 %** | 21,2 → **11,0 %** | −18,1 → **+9,1°** |
| `BlissfulTiredAiredale` | 93,8 % → **0,0 %** | 10,2 → **3,7 %** | −23,6 → **+11,5°** |
| `IdealFrigidJerboa` | 59,2 % → **13,8 %** | 18,5 → **9,3 %** | −16,5 → **+5,3°** |
| `ImpeccablePertinentBasilisk` | 84,6 % → **71,4 %** | 43,3 → **32,9 %** | −57,5 → **−30,2°** |
| `SeparateHoarseFeline` | 55,7 % → **6,7 %** | 14,7 → **11,8 %** | −6,8 → **+30,6°** |
| `WelloffForcefulIndri` | 56,9 % → **9,3 %** | 12,2 → **6,5 %** | −8,0 → **+13,3°** |
| **TOTAL** | **63,6 % → 15,7 %** (336/528 → 62/394) | **19,4 → 11,6 %** | **−21,3 → +4,8°** |

El 63,6 % **reproduce exacto** el número que 22.22 dejó escrito, sobre los mismos 7 clips: es la misma
medida, no una parecida. Criterio declarado ≤ 31,8 %; medido **15,7 %**: **mejora**, con margen de 2×
sobre la meta. Y el signo de la elevación cambia: el eje deja de apuntar al suelo y pasa a apuntar
ligeramente hacia arriba, que es lo que hace un brazo armado.

`ImpeccablePertinentBasilisk` es el único que sigue mal (71,4 %), y va nombrado: es el clip con la
elevación más negativa con los dos ejes y el que 22.4 ya señalaba con el ruido de z más alto.

**Medida 2, el ángulo contra la hoja de verdad.** `e14` había carado los ejes contra las **6** rectas
de hoja de la ventana de acción y salió **empate** (23,1 contra 25,8, separación 2,7° < margen 5°).
`e20` rehace el careo con **todas** las rectas que hay en caché: las 6 de `e13` más **25 de `e5`** que
`e14` no había usado, recalculando los ángulos del NPZ (control de reproducibilidad contra los
guardados: **0,0000°** sobre 51 comparaciones, límite declarado 0,5).

| selección | n | `codo` mediana / p90 | `hombro` mediana / p90 | separación | gana |
|---|---:|---:|---:|---:|---|
| todas, sin filtro | 39 | **19,7** / 65,5 | 44,6 / 66,8 | 24,9° | codo |
| delante según el codo | 31 | **14,4** / 33,0 | 44,8 / 66,3 | 30,4° | codo |
| delante según el hombro | 29 | **14,4** / 28,5 | 44,8 / 67,8 | 30,4° | codo |
| azar declarado | — | 45,0 | 45,0 | — | — |

**El hombro sale exactamente en el azar**: con ese eje la dirección de la hoja no se predice en
absoluto. Y los tres filtros dan el mismo ganador, así que no lo decide el filtro — que era el riesgo,
porque el filtro «delante de la mano» se calcula *con* un eje.

**Y por qué hacían falta las dos medidas y no basta una:** el ángulo contra la hoja se mide **módulo
180°**, porque un segmento detectado no tiene sentido, sólo dirección. Así que un eje que apunte al
suelo en vez de al rival da **el mismo número** que su opuesto: el defecto de «el cono se va al suelo»
es **invisible** para el careo con la hoja, por construcción.

### 24.4 «Más estrecho»: el cono no estaba gordo, estaba demasiado ESTRECHO para su propio eje

> **TODAS LAS APERTURAS DE ESTA SECCIÓN SON SEMIÁNGULOS**, medidos desde el eje. El cono completo abre
> el doble: la envolvente de **±33,0° son 66,0° de cono**, y el núcleo de ±14,4° son 28,8°. Comprobado
> en código en 24.8, no supuesto. Y con eso, **la impresión del usuario era correcta sobre lo que
> VE**: lo que hay en la figura son 66°, y por eso se ve gordo, aunque la cifra que decide sea 33.
> Las dos cosas son verdad y la confusión venía del convenio sin declarar.

Ésta es la vuelta de tuerca y es lo contrario de lo que parece. Los **23,1°** que `e15`, `e18` y `e19`
venían usando **no eran el p90 de nada**: eran la **mediana** de la muestra de n=6 de `e14`, con el eje
del hombro. Medido contra las hojas de verdad:

| cono | contiene | pierde | qué es ese número |
|---|---:|---:|---|
| **hombro 23,1°** (lo que se pintaba) | **22,6 %** | **77,4 %** | mediana de n=6 de `e14` |
| hombro 66,3° | 87,1 % | 12,9 % | el p90 honrado de ese eje |
| codo 14,4° | 51,6 % | 48,4 % | p50 medido: el **núcleo** |
| codo 23,1° | 74,2 % | 25,8 % | la apertura de hoy con el eje nuevo |
| **codo 33,0°** | **90,3 %** | **9,7 %** | p90 medido: la **envolvente** |

O sea: **el cono que el usuario ve gordo perdía tres de cada cuatro hojas reales.** Y las tres vías
posibles se separan limpiamente:

- **mismo eje, más apertura** → no arregla nada, sólo tapa
- **mejor eje, misma apertura** → de contener el 22,6 % a contener el **74,2 %** con los mismos 23,1°
- **mejor eje, p90 honrado** → la apertura necesaria baja de **66,3° a 33,0°**: **2,0× más estrecha**

Así que la respuesta a *«hazlo más estrecho»* tiene dos mitades y las dos son verdad:

1. **Sí baja, y a la mitad**, en lo que importa: la región que decide la amenaza pasa de necesitar
   66,3° a necesitar 33,0°, y eso es consecuencia directa de que el error del eje bajó. Es exactamente
   la cadena correcta: primero el eje, y la apertura baja porque la medida baja.
2. **No baja por debajo de 23,1°** para la decisión de amenaza, y decirlo es obligado: bajar de ahí
   con el eje nuevo dejaría fuera el 25,8 % de las hojas medidas. Lo que se pinta **sólido** es el
   **núcleo de 14,4°** (p50), que es **1,6× más estrecho que los 23,1° de antes** y hace que el cono
   *se vea* estrecho; y la **envolvente de 33,0°** va a trazos y es la que cuenta. Las dos a la vez,
   las dos con su cifra, y ninguna amenaza perdida por el dibujo.

### 24.5 Los dos controles del cambio, sin verdad-terreno

El cono nuevo (antebrazo, 33,0°, 1,10 m) se comprueba con los dos criterios que `e18` ya había
declarado, que no necesitan anotación:

| eje / apertura / alcance | frecuencia de solape | parpadeo |
|---|---:|---:|
| hombro · 23,1° · 0,90 m *(lo de hoy)* | 35,9 % | 3,5 % |
| hombro · 33,0° · 1,10 m | 45,5 % | 2,1 % |
| codo · 23,1° · 0,90 m | 20,6 % | 5,2 % |
| **codo · 33,0° · 1,10 m** *(el nuevo)* | **39,1 %** | **3,2 %** |

**Criterio 2 de `e18`** (el solape sigue informando si está en 5-80 %): 39,1 %, **dentro**. Y el
parpadeo baja de 3,5 a 3,2 %, dentro de la banda de 3,5-4,9 % que `e3` midió — no es una mejora que se
pueda defender, es un empate, y va dicho así.

Un detalle honesto: **el eje del antebrazo con la apertura vieja EMPEORA el parpadeo** (5,2 % contra
3,5 %). Tiene sentido y es coherente con todo lo anterior: un cono de 23,1° es demasiado estrecho para
el error del eje, y un cono demasiado estrecho entra y sale del solape. Al poner la apertura que la
medida pide, el parpadeo vuelve a 3,2 %.

### 24.6 La verificación en fuente primaria que NO se pudo hacer, y lo que el fichero sí dice

El encargo pedía verificar los 90 cm en `C:\Users\Alejandro\Desktop\Downloads\190675-tecnico esp.pdf`.
**Esa ruta no existe**; el fichero está en `C:\Users\Alejandro\Downloads\190675-tecnico esp.pdf`. Y
localizado y extraído su texto (sin `pypdf` ni `poppler` en la máquina: se descomprimieron los streams
con `zlib`, 92 de 104), resulta que **ese PDF no contiene la cifra**:

- es el **«Reglamento técnico», FIE, diciembre 2025, en español** — o sea el libro de artículos `t.`
- tiene **tres** cifras en cm en todo el documento: **4 cm**, 2 cm y 10 cm. No hay ningún «90 cm»
- **cita** `m.8.6`, `m.16.2` y `m.23.4` (la curvatura de la hoja) **sin transcribirlos**: las medidas
  del arma están en el libro de **material**, que no está en disco

Lo que ese fichero **sí** aporta, y es útil: confirma la pista en **14 m con 4 cm de tolerancia** y las
líneas de guardia a **2 m**, igual que 22.1; y su cita conjunta de `m.8.6 / m.16.2 / m.23.4` para las
tres armas en orden **corrobora que `m.8` es florete y `m.23` es sable**, que es lo que hace falta para
saber que **los 88 cm que circulan son de sable** y no de florete.

**Así que los 90 cm y los 110 cm siguen descansando en la verificación de 22.1** (FIE *Material Rules*
Book 3, agosto 2026, cotejada con diciembre 2025), no en una segunda comprobación que no se ha podido
hacer. Va dicho en vez de afirmar una verificación que no ocurrió.

### 24.7 Las figuras

`02_validacion/muestras/cono_arreglado/`, **6 imágenes**, una por asalto, cada una con **antes y
después del mismo fotograma**:

| clip | cuadro | por qué ese cuadro | `origen` l0/l1 | L: 0,90 → 1,10 m |
|---|---:|---|---|---|
| `AdorableFittingAsiansmallclawedotter` | 64 | hoja detectada | **0** / 1 | 236 → 288 px |
| `AromaticShockedGermanspitz` | 211 | hoja detectada | 1 / **0** | 229 → 280 px |
| `ImpeccablePertinentBasilisk` | 175 | hoja detectada | 1 / **0** | 171 → 209 px |
| `WelloffForcefulIndri` | 199 | hoja detectada | 1 / **0** | 197 → 241 px |
| `SeparateHoarseFeline` | 238 | manos a 62 cm | 1 / 1 | 222 → 271 px |
| `AliveIgnorantBarebirdbat` | 302 | manos a 59 cm | 1 / 1 | 199 → 243 px |

**En 4 de las 6 la hoja está detectada de verdad** y se pinta en verde con el rótulo `HOJA VISTA`, con
`origen = 0` en el pie; en las otras dos `origen = 1` y **no se dibuja ninguna hoja**. La columna
`origen` no se pierde: va en el rótulo de cada figura, con la leyenda al lado.

**Y el criterio de elección del cuadro se cambió, con su motivo.** `e19` elegía el cuadro de `peso_max`
más alto, y **ese criterio está sesgado para esto**: `peso_max` se acerca a 1 cuando los dos ejes casi
coinciden, y eso ocurre sobre todo cuando **las dos manos casi coinciden en la imagen**, o sea en
cuadros con los dos tiradores solapados y ocluidos. Se vio en la primera versión de la figura de
`SeparateHoarseFeline`, que salió con los dos vértices a unos píxeles y no se podía juzgar a ojo. El
criterio de reserva es ahora la **distancia medida entre las dos manos**, la más cercana a 60 cm, que
es el centro del tramo donde la pregunta del cono está viva (22.7). No usa la etiqueta de tocado.

### LO QUE ESTA MEDIDA NO DICE

- **El careo con la hoja descansa en UN asalto.** De las 31 rectas con el filtro de delante, **24 son
  de `AdorableFittingAsiansmallclawedotter`**. Los demás clips dan n=1, 2 y 3. Y el desglose lo dice
  todo: sólo con las 6 rectas de `e13` sale **empate** (2,7°, reproduciendo `e14` exacto), y sólo con
  las 3 de `ImpeccablePertinentBasilisk` **gana el hombro** por 5,3°. O sea que «el antebrazo predice
  la hoja» está **medido en un asalto y no replicado**. Lo que sí es robusto es C-EJE, que son 7 clips
  y 922 candidatos y no necesita ver la hoja.
- **Y ese asalto NO está en `corpus_etiquetado.txt`.** `AdorableFitting...` y
  `SentimentalLargeBedlingtonterrier` son los clips de `e1`-`e8` y no están en la lista. Comprobado a
  ojo que **sí es florete** (VOLPI ITA contra KONDRICZ HUN, Tokio 2020, lamé de torso solamente), así
  que no es contaminación de espada ni de sable; pero no es una muestra del corpus etiquetado y va
  dicho.
- **No dice que 33,0° sea la apertura correcta del corpus.** Es el p90 de 31 detecciones a 1080p, y
  1080p es el 18 % del corpus. En el 97 % de los conos la dirección se supone (22.11), así que la
  apertura que se aplica casi siempre es una regla traída de otros clips.
- **No dice que el cono localice el contacto.** Sigue en pie lo de 22.22: no hay verdad-terreno del
  contacto de hojas (0 positivos decidibles en 15 láminas), y el parpadeo mide estabilidad, no
  acierto. Un proxy puede ser estable y estar equivocado.
- **No arregla `mano_armada`, y se ve en las figuras.** En `AliveIgnorantBarebirdbat` y en
  `AromaticShockedGermanspitz` hay un vértice que no cae en una mano armada, que son los defectos D1 y
  D3 que `e17` midió. El cono nuevo hereda ese error entero: si el vértice está mal, el cono está mal
  por muy bien orientado que esté. Esa pieza la está reescribiendo otro agente con la señal del
  guante.
- **El 15,7 % no es cero.** Con el antebrazo siguen cayendo al suelo 62 de 394 candidatos, y el 71,4 %
  de `ImpeccablePertinentBasilisk` dice que en algunos clips el problema no era sólo el eje.
- **El alcance de 1,10 m es una cota, no una medida.** Es el máximo reglamentario del arma montada. No
  se ha medido dónde agarra cada tirador ni el largo real de su arma, y la cazoleta —que es lo que
  permitiría medirlo, 19-29 px según la etapa E— no se ha detectado en ningún cuadro de esta sección.
- **El encogimiento de escala del 15 % está medido y NO corregido.** Así que el cono nuevo, aun con
  1,10 m, sigue quedándose corto en los cuadros a fondo, que son justo los de la acción. Lo que haría
  falta es una escala que no dependa del alto de caja, y la vía identificada sigue siendo la de 22.14:
  la homografía de la pista sobre el rectángulo conocido de 14,00 × 1,50 m.
- **El parpadeo de 3,2 % contra 3,5 % no es una mejora demostrada**, es un empate dentro de la banda
  de 3,5-4,9 % que `e3` midió. Se da porque se declaró antes, no porque salga a favor.
- **No se ha vuelto a detectar ni una hoja.** Todas las rectas salen de los JSON que `e5` y `e13`
  dejaron en caché, a propósito: repetir la detección necesita `matting` en GPU y además podría dar
  otro resultado, y entonces la figura no sería del dato que se publicó.
- **Cambiar `EJE_POR_DEFECTO` a `codo` mueve las cifras de `e10`, `e11`, `e15`, `e18` y `e19` si se
  vuelven a correr.** No se han vuelto a correr en esta sección, así que las tablas de 22 siguen
  siendo las de `hombro` y hay que leerlas así. `eje_antebrazo` y `eje_brazo_2d(cual="hombro")` siguen
  existiendo las dos para poder reproducir cualquier medida vieja tal como se hizo.

### 24.8 El convenio de la apertura, comprobado en código: es SEMIÁNGULO

Código en `e22_control_al_rival_y_escala.py`, control A, con asertos que **fallan** si alguien cambia
el convenio. Hacía falta porque mirando una figura la envolvente se ve de 66° y la constante dice 33,
y este proyecto ya tiene documentado un día con **cinco cifras contradictorias, todas por un convenio
sin declarar**.

| punto a … del eje | peso |
|---|---:|
| 32,0° | 0,0303 |
| **33,0°** | **0,0000** |
| 34,0° | 0,0000 |
| 65,0° | 0,0000 |

O sea que `peso_sector` corta a **33° desde el eje** y el arco que dibuja `sector()` va de −33 a +33:
**66,0° de cono completo**. Las cuatro piezas de la cadena usan el mismo convenio, comprobado:

- `cono_lib.peso_sector` filtra con `a <= apert`, y `a` es el ángulo desde el eje
- `sector()` y `dibuja_cono()` barren `linspace(-apert, +apert)`
- `e6` calcula su techo de utilidad con **medio** ancho de hombros: `arctan(0,5·ancho/L)`
- el p90 se midió sobre `|ángulo hoja − ángulo eje|`, que es también un ángulo desde el eje

Así que **el dibujo y la medida eran coherentes y correctos**; lo que faltaba era decirlo. Ahora va en
tres sitios: en el **nombre** de la constante (`APERT_ENVOLVENTE_SEMI_GR`, `APERT_NUCLEO_SEMI_GR`,
y la nota de semiángulo añadida a `APERT_E6_POOL`), en el **rótulo de cada figura** (`±33,0 gr (66,0
totales)`) y en el docstring, con la frase de que el p90 se midió sobre esa misma convención.

### 24.9 El control POSITIVO que le faltaba: ¿el cono apunta al rival? Y uno de los criterios era circular

El criterio C-EJE de 24.3 es **negativo**: dice que el cono ya no se cae al suelo, no que apunte al
sitio correcto. Un cono que apuntara al techo lo pasaría igual de bien. El control positivo usa la
**malla del torso del rival** (hombros 5-6 y caderas 9-10, la misma que usa `construye_cono`), por
cuadro y sin ver la hoja. Criterios declarados antes: **B1** el antebrazo gana si su ángulo al torso
es menor por más de 5° (el margen del proyecto); **B2** en los cuadros con el torso en alcance, la
envolvente tiene que contenerlo en **más del 50 %**.

| | `hombro -> raíz` | `codo -> raíz` |
|---|---:|---:|
| ángulo al centro del torso rival, mediana (n≈3.350) | **23,7°** | 34,5° |
| p25-p75 | 8,4-62,6 | 20,5-60,3 |
| cuadros con el torso en alcance | 11,5 % | 11,4 % |
| de esos, **DENTRO de la envolvente** | 38,3 % | **49,5 %** |
| de esos, dentro del núcleo | 28,1 % | 30,2 % |
| dentro sólo en DIRECCIÓN (sin alcance) | 62,6 % | 57,9 % |

**B1 falla, y falla CONTRA el eje que elegí**: el hombro apunta al torso mejor por 10,8°, que pasa el
margen. Va dicho antes de explicarlo, porque el orden importa.

**Y B1 se retira como criterio, por una razón estructural y no por el resultado.** La enseña el careo
B3, que pone las dos preguntas una al lado de la otra:

| eje | \|eje − **hoja medida**\| | \|eje − **dónde está el rival**\| |
|---|---:|---:|
| `hombro -> raíz` | **44,8°** (= el azar) | **23,7°** |
| `codo -> raíz` | **14,4°** | 34,5° |
| azar declarado | 45,0° | — |

El hombro se parece mucho más a *«dónde está el rival»* que a *«dónde está la hoja»*, donde da
exactamente el azar. Y eso es esperable por construcción: **el hombro está detrás de la mano y el
rival delante**, así que `hombro -> raíz` prolongado es aproximadamente la dirección al adversario.
Luego B1 premia al eje que mejor aproxima «dónde está el rival», que es **circular** para elegir el
eje de un cono: un cono construido sobre él afirmaría *«la amenaza apunta a donde está el rival»*, que
no puede equivocarse nunca y por eso no informa de nada. **Si la hoja apuntara siempre al torso, el
cono no haría falta.** El eje se queda en el antebrazo, y quien decide es la hoja medida, que es la
única referencia no circular que hay.

**B2 tampoco pasa, y además no decide.** Sale **49,5 %** con la escala `mediana` y **50,9 %** con la
`p95`, con la raya declarada en 50: el veredicto cambia de lado según un estimador de escala que
tampoco pasó su propia puerta (24.10). Lo único robusto de B2 es la **razón entre ejes, 1,23-1,29× a
favor del antebrazo**, que va en el mismo sentido que todo lo demás.

**Y el número absoluto es el resultado útil, aunque sea incómodo: el cono pierde alrededor de la mitad
de las situaciones en alcance.** Eso no lo arregla el eje —ya está en su mejor versión— y apunta a las
dos causas que quedan identificadas: los vértices que no caen en la mano armada (defectos D1 y D3 de
`e17`, que otro agente está arreglando con la señal del guante) y una apertura calibrada sobre un solo
asalto.

### 24.10 La escala: intentada y RETIRADA con su medida, y lo que eso enseña del método entero

24.2 dejó anotado que `escala_px_m` toma el alto de caja por la estatura y que en guardia eso encoge
la referencia un 15 %. Se implementó el arreglo —`cono3d_lib.escala_px_m_de_pie`, que usa el **p95**
del alto del clip, con interruptor `modo="mediana"` que devuelve el valor viejo bit a bit— y se le
puso una puerta antes de medir: **R1**, que el p95 sea reproducible dentro del clip al **5 %**,
comparando la primera mitad con la segunda. Si no es estable, no es una propiedad del tirador y no
sirve de referencia de estatura.

| estimador | discrepancia mediana entre mitades | máximo | veredicto |
|---|---:|---:|---|
| **p95 del alto de caja** | **10,9 %** | 14,9 % | **NO PASA** (límite 5 %) |
| mediana del alto de caja | **10,7 %** | 24,6 % | tampoco pasaría |

**Y ahí está lo que de verdad enseña: R1 no separa los dos estimadores, falla para los dos casi
igual.** Eso no es un resultado sobre el percentil, es un resultado sobre el método: **el alto de caja
no es una referencia de estatura reproducible al 5 % en ningún percentil.** Cambiar la mediana por el
p95 sería cambiar un estimador no validado por otro no validado, y de paso mover toda cifra en
centímetros de la sección 22.

Así que **`ESCALA_POR_DEFECTO` se queda en `"mediana"`** y el cambio se retira con su medida, como
`USA_CRUDAS = False`. La función queda calculable con el interruptor, y **R3 publica exactamente lo
que se movería**, separando los dos cambios para saber cuál pesa:

| clip | 0,90 m · mediana *(reproduce 22.7)* | 1,10 m · mediana | 1,10 m · p95 |
|---|---:|---:|---:|
| `AliveIgnorantBarebirdbat` | 12,4 % | 16,0 % | 18,5 % |
| `AromaticShockedGermanspitz` | 13,7 % | 15,7 % | 17,6 % |
| `BlissfulTiredAiredale` | 1,7 % | 2,8 % | 4,1 % |
| `IdealFrigidJerboa` | 0,7 % | 4,3 % | 6,9 % |
| `ImpeccablePertinentBasilisk` | 3,9 % | 9,1 % | 11,7 % |
| `SeparateHoarseFeline` | 15,9 % | 20,5 % | 21,2 % |
| `WelloffForcefulIndri` | 8,5 % | 11,9 % | 13,7 % |
| **mediana de clips** | **8,5 %** | **11,9 %** | **13,7 %** |

O sea: los cuadros con amenaza posible pasan de una mediana del **8,5 % al 11,9 %** sólo por el
alcance reglamentario, y llegarían al **13,7 %** si además se cambiara la escala. **Se aplica el
primero y no el segundo**, y queda escrito cuánto valdría el segundo para que nadie tenga que
adivinarlo.

**Y la queja del usuario queda atendida por la parte que sí tiene fuente primaria:** el largo sube un
**+22 %** (0,90 → 1,10 m, m.8.4 y m.7), que es más que el 15 % de la escala y no depende de ningún
percentil. El arreglo de verdad de la escala sigue siendo el de 22.14: la homografía de la pista sobre
el rectángulo conocido de 14,00 × 1,50 m. R2, descriptivo: la razón `px/m` entre los dos tiradores del
mismo asalto discrepa un 6,3 % con la mediana y un 6,7 % con el p95 — no es una puerta, porque dos
personas no miden lo mismo.

### LO QUE TAMPOCO DICEN LOS CONTROLES DE 24.8-24.10

- **El eje del antebrazo NO está validado por el control de apuntar al rival.** Lo pierde (34,5°
  contra 23,7°). Se mantiene porque ese control es circular y porque contra la hoja medida el hombro
  está en el azar, pero conviene tenerlo presente: **hay un control declarado que va en contra**, y
  descansa en que la hoja medida sea mejor referencia que la posición del rival. Eso es un argumento,
  no una medida.
- **B2 no aprueba nada.** 49,5 % contra una raya de 50 % no es «casi pasa»: es un resultado dentro de
  su propia incertidumbre, porque cambiar de estimador de escala lo cruza. Lo único que se puede
  afirmar es el sentido de la comparación entre ejes.
- **El cono sigue perdiendo la mitad de las situaciones en alcance**, con el mejor eje disponible. No
  está dicho por qué exactamente: las dos causas candidatas (vértice mal puesto, apertura de un solo
  asalto) están identificadas y no separadas.
- **R1 no dice que la escala del proyecto esté mal en valor**, dice que no es reproducible dentro del
  clip al 5 %. La comparación de 22.6 contra el plano del suelo daba acuerdo del 2-4 %, y no es
  independiente porque el ajuste del suelo usa los mismos altos de caja.
- **Los asertos del control A protegen el convenio, no el valor.** Si alguien cambia 33,0 por otro
  número, A sigue pasando: sólo falla si la apertura deja de ser un semiángulo.
- **Nada de esto toca `mano_armada`**, que la está reescribiendo otro agente con la señal del guante.
  Los defectos de vértice que se ven en las figuras son suyos, y se da por hecho que mejoran por ese
  lado; cuando lo hagan, **B2 hay que volver a medirlo**, porque es el control que más depende de
  dónde esté el vértice.

---

## 25. La mano armada: el guante NO separa, y lo que sí hay es una cifra y un aviso de firmeza

El usuario lo pidió así: *"**La mano dominante muchas veces la pones mal y detectas la otra.**
Revísalo porque esto sí es importante. A ver cómo lo hace, porque muchas veces no es la que está más
adelantada. [...] **La mano dominante siempre tiene el guante y la otra va sin guante**, igual eso
ayuda, pero no sé, revisa a ver qué puedes hacer."*

**Tenía razón en el diagnóstico y la pista del guante no funciona.** Las dos cosas están medidas.
Código en `01_extraccion/codigo/mano_guante.py`, herramienta en
`02_validacion/herramienta/verifica_mano.py`, figuras y anotación en
`02_validacion/muestras/mano_dominante/`, careo reproducible en
`01_extraccion/pruebas/mano_guante_careo.txt`.

### 25.1 · El reglamento dice lo que dice, y el artículo del encargo no está en ese PDF

Verificado en fuente primaria el 2026-09-11 sobre `C:\Users\Alejandro\Downloads\190675-tecnico esp.pdf`
(FIE, *Reglamento Técnico*, edición de diciembre de 2025, en español):

- **t.115.1** — «el esgrimista se presentará en la pista completamente listo para combatir: uniforme
  reglamentario, chaqueta cerrada, **guante y arma agarradas con la mano armada**, el pasante
  conectado a la ficha del interior de la cazoleta. Única excepción para la careta, que debe ser
  agarrada con **la mano no armada**.»
- **t.1** — en el saludo, «colocan la careta bajo el brazo desarmado / levantan la cazoleta hasta la
  barbilla».
- **t.20.2** — guante obligatorio para dar y recibir la lección, y **remite al m.25**.
- **t.27, t.28, t.29.3, t.30, t.48** — «mano armada» y «mano no armada» son términos definidos.

**Dos correcciones al encargo, dichas y no escondidas.** (1) **El m.25 no está en ese PDF**: ese PDF
es el Reglamento *Técnico* (artículos `t.`) y sólo **cita** el m.25, que vive en el Reglamento del
*Material*, que no está en disco en español. Lo verificado es t.115.1. (2) En el Reglamento Técnico
**no hay ningún artículo que prohíba un guante en la mano no armada**; que la otra vaya desnuda es la
práctica normal del deporte, no una regla citable. Por eso el guante se midió como **rasgo** y se
careó contra una anotación, en vez de darse por bueno porque lo diga el reglamento.

### 25.2 · La verdad de referencia: la he hecho yo mirando recortes, no un árbitro

No existía verdad etiquetada de la mano armada, así que se construyó y **está en fichero**:
`02_validacion/muestras/mano_dominante/verdad_mano.csv`. 17 clips de `corpus_etiquetado.txt` (los
únicos con poses del formato nuevo), los dos lados de cada uno = **34 tiradores**.

Los mosaicos de anotación son **ciegos**: sólo llevan las letras `A` (raíz 21) y `B` (raíz 42), que
son identidades de índice, nunca el veredicto de un criterio. Se decidió mirando, por este orden, la
**hoja**, la **cazoleta / puño oscuro / pasante**, y en último lugar «cuál va delante»; **ninguna fila
se decidió por el guante**, justo para no medir el guante contra sí mismo.

**Dos pasadas, y las dos están en el fichero.** La segunda, con 6 cuadros en vez de 3, se hizo sobre
**todos** los tiradores que en la primera quedaron con `certeza=media` —o sea, el criterio para
volver a mirar fue mi propia duda, no lo que dijera ningún algoritmo— más los dos casos de
`certeza=alta` con contradicción de más de 60 px de margen. **17 de 34 se volvieron a mirar; 3
cambiaron de letra y 2 pasaron a `?`**, y los cambios salieron **en las dos direcciones**:
`KindEnchantingKid l0` a favor del criterio (a) y `WhiteMarriedGuineafowl l0` y `l1` en contra. Con la
verdad de la primera pasada el criterio (a) daba 0,724 y con la de la segunda da 0,741: la revisión
mueve la cifra **1,7 puntos**, no la fabrica.

Quedan **6 tiradores sin verdad** y el motivo de cada uno es un dato:

| tirador | por qué no hay verdad |
|---|---|
| `PowerfulAllLarva l0` | **no es un tirador, es el árbitro**. Es el error que la sección 18 dejó sin arreglar |
| `BabyishIdioticCockerspaniel l1` | **las dos raíces de mano colapsadas** en el 100 % de los cuadros: `sep` 3-6 px con `r` 42-57 px |
| `LavishDistantDikkops l0` | tirador **fuera de cuadro** (el clip del 100 % de borde de 18.4) y además colapsado |
| `AliveIgnorantBarebirdbat l1` | ambiguo con 6 cuadros: el brazo `A` está en el aire en 5 de 6 y no se ve arma en ninguno |
| `UnfortunateColossalLamb l0` y `l1` | mis dos pasadas se contradicen y no hay arma visible |

### 25.3 · La cifra que sustituye al «4 de 4»: el criterio actual acierta 3 de cada 4

Sobre los **27 tiradores** que tienen verdad **y** medida (falta `LavishDistantDikkops l1`, cuyo clip
sólo deja 7 cuadros utilizables):

| criterio | acierto | sobre |
|---|---:|---:|
| **(a) «la más cerca del rival», el operativo de hoy** | **0,741** | 20 de 27 |
| (a) por **voto por cuadro** (`mano_armada_voto`) | **0,815** | 22 de 27 |
| (a) por voto con distancia euclídea en vez de sólo x | 0,815 | 22 de 27 |
| (b') «la que más apunta al rival» | 0,667 | 27 |
| «la más extendida» (ya rechazada en la etapa E) | 0,519 | 27 |
| **guante blanco alrededor de la raíz** | **0,444** | 27 |
| guante blanco, sólo cuando su voto es SÓLIDO | **0,167** | 1 de 6 |
| cazoleta oscura alrededor de la raíz | 0,500 | 20 |
| la combinación declarada (guante manda si es sólido) | **0,556** | 27 |
| contestar siempre `A` / siempre `B` | 0,519 / 0,481 | 27 |
| `A` si es el tirador izquierdo, `B` si es el derecho | 0,333 | 27 |

Y por subconjuntos: en los **20 tiradores de `certeza=alta`** (a) sube a **0,800** y el voto **no
mejora** (0,800 también); en los 22 con evidencia fuerte (hoja o arma visible) (a) da 0,727 y el voto
0,773; a **720p** (25 de 27) (a) da 0,760 y el voto 0,840. **A 1080p sólo hay 2 tiradores** en todo el
corpus con poses, así que la partición por resolución **no se puede hacer**: se da la de 720p y se
dice que la otra no existe.

**Lo que esto cambia respecto a la etapa E.** El «(a) acierta en los 4 de 4 tiradores comprobados a
ojo» no era falso: era una muestra de 4. Con 27 el criterio **falla 1 de cada 4**, que es exactamente
lo que reportaba el usuario. El `4 de 4` está sustituido en el docstring de `cono_lib.mano_armada`.

### 25.4 · Por qué falla, con las siete fallas una por una

| tirador | (a) | a ojo | margen de (a) | qué pasa |
|---|---|---|---:|---|
| `WhiteMarriedGuineafowl l0` | A | **B** (arma) | **145,4 px** | de espaldas y con el arma atrás |
| `UglyNiceAplomadofalcon l1` | A | **B** (hoja) | 65,3 px | en reposo, arma hacia fuera |
| `WhiteMarriedGuineafowl l1` | A | **B** (arma) | 56,3 px | ídem, guante oscuro |
| `TestyWatchfulBarebirdbat l1` | A | **B** (arma) | 35,7 px | andando entre acciones |
| `PowerfulAllLarva l1` | A | **B** (arma) | 7,6 px | en reposo, árbitro cruzando |
| `GentleQuerulousElephantseal l1` | B | **A** (hoja) | 1,9 px | a cara o cruz |
| `WhisperedTenderDeinonychus l0` | B | **A** (acción) | 1,5 px | a cara o cruz |

**Cinco de las siete son cuadros en los que el tirador no está encarado**, entre acción y acción, con
el arma atrás o abajo; ahí «la mano más cerca del rival» **es la desarmada por construcción**. Y dos
de ellas fallan con 145 px y 65 px de margen, o sea que **no son casos apretados: el criterio está
seguro y equivocado**. Las otras dos fallan con 1,5 y 1,9 px, que es empate.

### 25.5 · El guante: el rasgo, y por qué no separa

Criterio declarado **antes** de ejecutar, en el docstring de `mano_guante.py`: parche cuadrado de
semilado `r = 0,42·|raíz − codo|` centrado `0,35·r` hacia los dedos, y `f_blanco` = fracción de
píxeles con **`V ≥ 150` y `S ≤ 70`**, **importada** de `blancura_pareja` y no redefinida. Decisión
pareada (las dos manos del mismo tirador, mismo cuadro, misma luz), voto por cuadro con abstención si
`|Δ| < 0,05`, una decisión por tirador y por clip.

Distribución sobre **855 cuadros** con las dos manos legibles y separadas:

| rasgo | armada p25/p50/p75 | la otra p25/p50/p75 | Δ mediano | signo correcto |
|---|---|---|---:|---:|
| **`f_blanco`** | 0,050 / **0,144** / 0,411 | 0,037 / **0,191** / 0,469 | **−0,002** | **0,455** |
| `a_blob` (mancha clara, la idea del guantelete) | 0,005 / 0,082 / 0,320 | 0,000 / 0,099 / 0,398 | +0,000 | 0,464 |
| `f_oscuro` (`V ≤ 60`, la cazoleta) | 0,000 / 0,000 / 0,022 | 0,000 / 0,000 / 0,009 | +0,000 | 0,419 |
| `f_punio` (saturado, puño de color) | 0,000 / 0,048 / 0,406 | 0,000 / 0,005 / 0,269 | +0,000 | 0,411 |

**La separación es −0,002 con el signo correcto en el 0,455 de los cuadros.** Eso no es «separa
poco»: es que **el signo está al revés** y la mano armada sale, si acaso, algo **menos** blanca. A
720p 0,455 sobre 797 cuadros y a 1080p 0,448 sobre 58: no cambia con la resolución.

**Los dos confundidos que el usuario avisó, medidos y no supuestos:**

- **la manga.** `f_manga`, el mismo parche movido al medio del antebrazo, correlaciona con `f_blanco`
  con **pearson 0,681 sobre 1908 parches**. El rasgo no mide guante: mide **cuánta chaqueta blanca
  hay detrás de la mano**. La figura que lo enseña es
  `discrepa_UglyNiceAplomadofalcon_l0.png`: la mano **desarmada** cae sobre el hombro blanco y da
  `f_blanco` 0,22-0,45, y la **armada**, que está sobre el suelo azul de la pista, da 0,02-0,03.
- **el puño de color.** El **32,6 %** de los parches tienen `f_punio ≥ 0,15`. Y hay guantes
  directamente **oscuros**: `UnevenChiefHippopotamus l0` y `WhiteMarriedGuineafowl l1`. En el segundo,
  con lupa a 6 aumentos, la mano armada mide **H 75-80, S 89-112, V 88-126**, o sea una masa oscura,
  no piel ni blanco.
- y un tercero que no estaba previsto: el **8,1 %** de los parches dan `f_blanco` exactamente 0.

**La cazoleta tampoco vale, y el umbral estaba mal puesto.** Se declaró un segundo criterio —la mano
armada es la más **oscura**, porque la cazoleta es un cuenco metálico de 19-29 px que **tapa** el
guante— con `V ≤ 60`. Acierto **0,500 sobre 20** y signo correcto 0,419. Y la lupa explica por qué:
la cazoleta mide **V 88-126** en estos vídeos, así que `V ≤ 60` casi nunca se alcanza (p75 de
`f_oscuro` = 0,022). **Un barrido post-hoc** del umbral —que no da acierto, sólo separación, porque
elegir después de ver no es medir— da 0,419 con `V ≤ 60`, **0,505 con 90, 0,510 con 110** y 0,475 con
130: incluso el mejor se queda en la moneda.

**Y un barrido post-hoc de la geometría del parche**, con el mismo aviso, da el mejor resultado con el
parche **más pequeño** (`r = 0,15·|raíz−codo|`, sin desplazar): Δ +0,043 y signo **0,598**; y el peor
con el más grande (`r = 0,60`): 0,414. O sea que la dirección del efecto confirma que el culpable es
la manga, pero **ni el mejor de 7 parches elegido a posteriori separa**.

**Retirado con su medida al lado**, al estilo de `USA_CRUDAS = False`: `mano_guante.USA_GUANTE =
False`. La combinación declarada (el guante manda si su voto es sólido) **quitaba 5 aciertos**
(0,556 contra 0,741), y lo peor es que **cuanto más firme era el voto del guante, peor acertaba**:
0,167 (1 de 6) en los sólidos.

### 25.6 · Las discrepancias, que son los casos que valen

Geometría (a) y guante discrepan en **14 de 27** tiradores. Al abrir el recorte de cada uno:
**la geometría tiene razón en 11 y el guante en 3**. Las tres del guante son
`GentleQuerulousElephantseal l1`, `PowerfulAllLarva l1` y `TestyWatchfulBarebirdbat l1`, y en las tres
el voto del guante era **flojo** (0,50, 0,70 y 0,57), o sea que acierta donde menos se cree a sí
mismo. Figuras `discrepa_*.png` y `falla_a_*.png` en `02_validacion/muestras/mano_dominante/`.

### 25.7 · Lo que SÍ queda: decidir por clip y publicar la firmeza

Un tirador no cambia de mano en medio de un asalto, así que la decisión es **una por tirador y por
clip**. `cono_lib.mano_armada_voto` (función nueva al final del módulo) vota **por cuadro** en vez de
resumir por medianas, sólo con cuadros de `origen == 0`, y devuelve `frac_a_voto`, la fracción de
cuadros que votan a la ganadora. Reparto sobre 32 tiradores: **p10 0,60, p50 0,93, p90 1,00**, y
`frac ≥ 0,80` en **22 de 32**.

Y esa fracción **predice el acierto**, que es lo que la hace útil:

| `frac_a_voto` | acierto | sobre |
|---|---:|---:|
| ≥ 0,85 | **0,895** | 17 de 19 |
| 0,70 – 0,85 | 0,500 | 1 de 2 |
| < 0,70 | 0,667 | 4 de 6 |

**El operativo NO cambia**: sigue siendo (a). Pasar de 0,741 a 0,815 son **+2 casos sobre 27**, y en
los 20 tiradores de certeza alta el voto **empata** con las medianas (0,800 los dos), así que la
mejora no está demostrada y no se vende. Lo que se añade a `mano_armada` son columnas: `a_voto`,
`frac_a_voto`, `corrobora_voto`, `firme` (`frac_a_voto ≥ 0,85`) y `corrobora_guante` cuando alguien le
pase el voto del guante. **La llamada de 6 argumentos sigue funcionando igual** y devuelve lo mismo
que antes en `mano` y en todas las claves viejas.

### 25.8 · Una avería que apareció al construir la herramienta: las dos raíces de mano colapsadas

Haciendo los mosaicos salió que **las dos raíces de mano caen a veces sobre el mismo punto**.
Medido sobre los 34 tiradores: `sep/r` tiene **mediana 4,78**, o sea que normalmente están bien
separadas, pero la fracción de cuadros con `sep < r` va de 0,00 a 1,00 con **p50 0,20 y p90 0,47**, y
**2 de 34 tiradores están colapsados en más de la mitad de sus cuadros**
(`BabyishIdioticCockerspaniel l1` en el 100 %, `LavishDistantDikkops l0` en el 86 %).

Cuando eso pasa **no es que el criterio se equivoque: es que la pregunta no tiene respuesta**, porque
los dos índices apuntan a la misma mano y los dos parches leen los mismos píxeles. Por eso
`mano_guante` tiene una puerta declarada `sep ≥ 2·r` —la condición exacta de que dos parches de
semilado `r` no se toquen— y por eso el elector de cuadros de `verifica_mano.py` la exige también. Y
hay un consuelo: **en los casos colapsados la elección es indiferente para el cono**, porque el
vértice sale en el mismo píxel con cualquiera de las dos manos.

### 25.9 · Coste

Todo CPU, sin GPU y sin volver a correr poses. 17 clips × 40 cuadros × 2 tiradores × 2 manos,
decodificando sólo los cuadros usados con `cap.set(CAP_PROP_POS_FRAMES, k)`: **~75 s** la tabla
completa. La anotación a ojo es lo caro: 34 mosaicos de 3 cuadros más 18 de 6 cuadros.

### Lo que esta medida NO dice

- **No dice que la mano armada esté resuelta.** Dice que el criterio de hoy acierta **0,741 sobre 27
  tiradores anotados por un agente**, y que 1 de cada 4 sigue mal. Que el usuario lo notara «muchas
  veces» encaja con esa cifra.
- **No dice que el guante no sirva para nada, dice que ESTE rasgo no separa en ESTE corpus.** Lo
  medido es `f_blanco` con `V ≥ 150`, `S ≤ 70` en un parche de 26-114 px de lado a 720p. Un detector
  de cazoleta de verdad, o el `matting` que ya se sabe que ve la hoja a 1080p, son otras vías y no se
  han probado aquí. Lo que sí queda establecido, y no depende del tamaño de muestra, es **por qué**
  falla: la chaqueta es blanca y el parche la lee (pearson 0,681), y hay guantes oscuros y con puño de
  color en el 32,6 % de los parches. Eso son hechos del deporte y del vídeo.
- **La verdad de referencia la he hecho yo mirando recortes, no un árbitro.** 20 de las 27 filas
  tienen `certeza=alta` y 22 se decidieron viendo hoja o arma; las otras 5 se decidieron por «cuál va
  delante», que es **la lectura más parecida al criterio que se está juzgando**, así que el 0,741
  podría estar algo inflado por ahí. Con sólo las 22 de evidencia fuerte da 0,727.
- **La segunda pasada de anotación la hice sabiendo lo que decía el algoritmo.** El remedio elegido
  fue revisar por un motivo independiente (mi propia `certeza=media`) y dejar en el CSV la letra de
  las dos pasadas y las dos cifras (0,724 y 0,741). No es lo mismo que una anotación ciega hecha dos
  veces por dos personas.
- **27 tiradores no validan un criterio para 8.558 clips.** Y **25 de los 27 son 720p**: a 1080p hay
  2 tiradores, así que la partición por resolución que pedía el encargo **no se puede hacer** y lo
  único honesto es decir que las cifras son de 720p.
- **`a_voto` no es una mejora demostrada.** +2 casos sobre 27, y empata en el subconjunto de certeza
  alta. Se publica porque se declaró antes de medir, no porque saliera a favor.
- **El aviso de firmeza tiene 19 casos en el tramo bueno.** El 0,895 son 17 de 19, y el tramo malo son
  6 casos. El corte 0,85 se declaró antes, pero con 27 tiradores cualquier corte está débilmente
  determinado.
- **Los barridos de parche y de umbral son post-hoc y no tienen acierto asociado.** Se publican como
  diagnóstico de por qué falla el rasgo y para que el siguiente agente no repita `V ≤ 60`. Si alguien
  quiere usar `r = 0,15` o `V ≤ 110`, tiene que declararlo y medirlo contra una anotación **nueva**.
- **No arregla el vértice del cono en los casos de 25.4.** Siete tiradores de 27 siguen con el cono
  colgado de la mano equivocada, y ahí el cono está mal por muy bien orientado que esté. Lo que la
  sección 24 pedía a esta pieza —«otro agente con la señal del guante»— **no se ha podido entregar**,
  y la cifra de por qué está en 25.5.

## 26. El vídeo del trazado, la serie de eventos consultable, y la puerta que no hacía lo que su cabecera decía

El usuario dijo que lo entregado de la pista no era lo que quería:

> *"Lo de la pista no veo ninguna gráfica que mida cómo se desplaza a lo largo de la longitud de la
> pista y el ancho, como una línea... estas gráficas que es una línea y te pone como un trazado de un
> coche de carreras por la pista. Pues quiero algo igual pero por la pista de esgrima, de los dos
> tiradores, dónde empiezan y cómo se va completando EN VÍDEO para saber en cada momento dónde están.
> Y luego lo de sacar cada keypoint: inicio de marcha, final de marcha, y así medimos su longitud de
> cada marcha... todo trackeado, para por ejemplo poder saber la tercera marcha del tirador de la
> derecha... Así que revísalo, porque lo que hay de la pista, mira los output de imagen, no tiene
> sentido, es poco, pones borroso el marcador o el fondo un poco extraño, no estás haciendo lo que
> quiero."*

**Y tenía razón en la crítica concreta:** los `homog_*.png` que había en `02_validacion/pista/` son
diagnóstico interno de la homografía (máscara de rótulos en magenta encima del marcador del clip,
región de pista en verde, residuos escritos), no un entregable. Están ahora en
`02_validacion/pista/diagnostico/` junto con los `censo_*.png`, y `pista.py` y `p1_censo_pista.py`
ya escriben ahí. No existía ningún vídeo.

Código: `02_validacion/herramienta/video_pista.py` (nuevo), `grafica_pista.py` y `verifica_pista.py`
reescritos en sus paneles, `01_extraccion/codigo/posicion.py` con tres tipos de evento nuevos y dos
criterios corregidos. Entregables y qué se puede creer de cada uno: `02_validacion/pista/LEEME.md`.

### 26.0 El intérprete de la tarea no servía, y eso se comprueba antes y no después

El encargo daba `scratchpad/cuda/.venv` como el intérprete con matplotlib. **No lo tiene**:
`-c "import matplotlib"` da `ModuleNotFoundError`. El único con cv2 + numpy + pandas + matplotlib +
scipy a la vez es **`scratchpad/fz/.venv`** (cv2 5.0.0, numpy 2.5.2, pandas 3.0.5, matplotlib 3.11.1).
`fz/.venv-gpu` también los tiene. Comprobado con `import` en los tres antes de escribir nada.

### 26.1 LA PUERTA DE PUBLICACIÓN NO HACÍA LO QUE SU PROPIA CABECERA DECÍA: 12,8 % contra 2,6 %

Es el hallazgo más duro de esta sección y no salió buscándolo: salió al elegir qué clips llevar al
vídeo. `control_estatura` declara en su docstring *"la estatura reproyectada de los DOS tiradores
cae en 1,60-2,00 m"*, en plural, y el código compara la **mediana AGRUPADA** de las alturas de los
dos. Con un tirador reproyectado a 2,23 m y el otro a 1,79 m, la mediana agrupada sale 1,82 m y
**pasa**, cuando ninguno de los dos está bien medido.

| puerta | clips de 78 | % |
|---|---:|---:|
| agrupada, la de `publicable` (sección 23.3) | **10** | 12,8 % |
| **por tirador, la que dice la cabecera** | **2** | **2,6 %** |

Los dos que pasan la estricta son `AliveIgnorantBarebirdbat` y
`VibrantJoyousAsiaticgreaterfreshwaterclam`. De los 10 publicables, **8 tienen a un tirador fuera de
1,60-2,00 m** y **1 pasa con un solo tirador medido**.

Y ese último caso es el peor: `ForcefulAntiqueAlbacoretuna` tiene `h0 = 1,93 m` y `h1` sin ninguna
caja utilizable, así que la puerta se decidió con un tirador. Sus controles físicos independientes
dicen que su escala está inflada unas **2,4 veces**:

| control independiente | valor | lo esperado |
|---|---:|---|
| separación de pies, mediana | **1,48 m** | 0,50-0,90 m, la guardia de un floretista |
| largo de pista visible en el cuadro | **12,65 m** | p50 del corpus 5,53 m |
| apoyos fuera de la pista | **8,6 %** | < 5 % |
| deriva del punto medio de los dos (26.2) | **8,83 m** | imposible en una ventana de 5,5 m |

Nótese que el clip **pasa la puerta de estatura con 1,93 m** a pesar de eso: la estatura reproyectada
de UN tirador no es un control suficiente, y estos cuatro sí lo detectan.

**No se ha cambiado `publicable`**, para no mover por debajo las cifras de la sección 23 que alguien
pueda haber citado. Se han añadido a `homografia.csv` las columnas `C1_dos_lados`,
`C1_dos_lados_pasa` y `publicable_estricto`, que son función pura de `h0_p50` y `h1_p50` y por tanto
se calcularon **sin volver a correr los 78 clips**. `pista.analiza` ya las devuelve.

### 26.2 Dónde está el tirador EN la pista: una convención declarada, y se mide lo que vale

La sección 23.9 dice que el origen longitudinal es una marca transversal **sin identificar**, así que
la x absoluta no significa nada. Pero para pintar la pista de 14 m hace falta un origen. Convención,
declarada antes de medir: **el punto medio de los dos tiradores, mediana del clip, se toma como
centro de la pista**. En un asalto de florete la acción se desarrolla alrededor del centro, así que
es la mejor apuesta posible; **es una apuesta y no una medida**.

Y se mide lo que vale: `deriva_m` es el p95 menos el p05 del punto medio dentro del clip. Criterio
declarado para llamarla utilizable: **< 1,00 m**, que es un cuarto de los 4,00 m entre líneas de
guardia.

| clip | deriva del anclaje | utilizable |
|---|---:|---|
| `WastefulYawningAmericanquarterhorse` | **0,55 m** | sí |
| `GentleQuerulousElephantseal` | **0,74 m** | sí |
| `HairyCourteousEarwig` | 1,09 m | no |
| `AliveIgnorantBarebirdbat` | 1,13 m | no |
| `IlliterateDisloyalFulmar` | 1,57 m | no |
| `TestyWatchfulBarebirdbat` | 1,62 m | no |
| `VibrantJoyousAsiaticgreaterfreshwaterclam` | 1,73 m | no |
| `UglyNiceAplomadofalcon` | 2,12 m | no |
| `SeparateHoarseFeline` | 2,99 m | no |
| `AdorableVeneratedDuiker` | 3,43 m | no |
| `ForcefulAntiqueAlbacoretuna` | **8,83 m** | no |

**Utilizable en 2 de 11.** Así que la posición sobre el mapa es ilustrativa en 9 de 11 clips y el
rótulo del vídeo lo dice en pantalla, cuadro a cuadro. Los DESPLAZAMIENTOS siguen estando en metros:
lo que no se sabe es en qué punto de los 14 m caen.

De paso, la deriva es un control físico nuevo y gratis: un punto medio que se pasea 8,83 m en una
ventana de cámara que abarca 5,53 m **no es deriva, es una escala mal puesta**, y por eso delata a
`ForcefulAntiqueAlbacoretuna` sin necesidad de ninguna verdad-terreno.

### 26.3 La coordenada transversal: cómo se pinta algo que no es creíble

La sección 23.7 midió que el recorrido transversal sale p50 **0,87 m** contra el control declarado de
< 0,60 m y **falla en 8 de 10 clips**. Y la combinación concreta es la peor posible para un dibujo:
el temblor transversal cuadro a cuadro es **pequeño** (p95 0,043 m) y la deriva es **grande**. O sea
que pintarla da una línea suave, bonita y falsa.

La figura anterior la pintaba. En `pista_GentleQuerulousElephantseal.png` la traza verde **se salía
del panel por arriba**, con valores de varios metros en una pista de 1,50 m, y el eje decía
"0 = línea central", que es justo lo que 23.9 dice que no se sabe.

Lo que se hace ahora, y es una decisión de diseño con su razón medida al lado:

 - en el mapa cada tirador va en un **carril fijo** (0,25 y 0,75 del ancho), rotulado en el propio
   mapa como DIBUJO y no medida;
 - la transversal medida tiene **su propio panel** (`a2`), en gris, con su cifra de recorrido y la
   comparación con el control declarado;
 - todas las filas del CSV llevan `transversal_creible = False`;
 - en el vídeo, el punto de posición actual lleva una **barra vertical de incertidumbre** del tamaño
   del recorrido transversal medido del clip.

La sensación de "trazado de coche de carreras" la da el panel del **TRAZADO**: posición a lo largo de
la pista en el eje horizontal y **tiempo** en el vertical. Las dos coordenadas de ese panel están
medidas, la x por homografía y el tiempo por el número de cuadro. Es la única forma de dar al usuario
la línea que pidió sin inventar un eje.

Y de paso se corrigió una tercera cosa de la figura anterior: **rotulaba una "línea de aviso" a ±5 m
que no existe.** La sección 23.1 ya lo había verificado en fuente: `fie_2021.txt` t.19 dice que los
últimos 2 m se distingan *"if possible by a different colour of piste"*, o sea una **frontera de
color**, no una raya pintada. Ahora se pinta como cambio de color y se rotula como lo que es.

### 26.4 El detector de paso: tres criterios míos fallaron antes de que saliera el cuarto

El usuario pidió *"inicio de marcha, final de marcha, y así medimos su longitud de cada marcha"*. Lo
que había daba magnitud p50 **0,12 m** (sección 23.8) y una marcha de florete son 0,30-0,50 m.

Criterio declarado antes de tocar nada: **la magnitud p50 tiene que caer en 0,25-0,60 m y la duración
en 0,20-0,60 s**. Si no, el detector no está midiendo un paso, se llame como se llame.

| versión | qué hacía | resultado medido | veredicto |
|---|---|---|---|
| **v1**, la de 23.8 | módulo en 2D de la diferencia de un cuadro sobre la posición CRUDA del pie, umbral 0,90 m/s | p50 **0,12 m** | — |
| **v2** | v1 fusionando tramos consecutivos del mismo sentido a ≤ 6 cuadros | **n 11 → 11, p50 0,122 → 0,122 m** | **no cambia nada** |
| **v3** | zancada = tramo MONÓTONO de la longitudinal suavizada con cambio ≥ 0,15 m | eventos de **4,4 s y 6,8 m** | descartado |
| **v4**, la que se usa | "un pie se mueve y el otro no" sobre la velocidad LONGITUDINAL de la posición SUAVIZADA, y un paso = zancada de un pie emparejada con la del otro | p50 **0,397 m**, duración **0,37 s** | **pasa** |

Y las tres razones de los tres fallos, que valen más que el resultado:

**v1 no detectaba pasos: detectaba saltos de keypoint.** Sacando los tramos uno a uno, duran 3-5
cuadros y los picos de velocidad del pie llegan a **12,33 m/s**, que es imposible: un pie de
esgrimista no pasa de 4-6 m/s. Lo que disparaba era el cambio de cuál de los cuatro puntos del pie
es el más bajo en `_pie_pose`, más el ruido de la coordenada transversal, que entra por tomar el
módulo en 2D. **La propia cabecera de `posicion.py` ya avisaba de esto para `d_pies`** ("meterla en
la separación mete su ruido en la señal") y el mismo argumento no se había aplicado a `_vel`.

**v2 partía de una hipótesis razonable y falsa.** "Un paso son dos tramos, primero un pie y luego el
otro, separados por el doble apoyo". Los huecos medidos entre tramos son de **2 a 266 cuadros**, no
de 6, así que no había nada que fusionar.

**v3 falló por geometría, no por umbral.** La posición de un pie a lo largo del clip es una
**escalera**, y una escalera es monótona completa: un criterio de monotonía no puede segmentar una
escalera. Salían eventos de un asalto entero.

El umbral de la v4 está **recalibrado sobre la señal nueva y no heredado**, con la regla del proyecto
de 3 veces el suelo de ruido. El ruido se mide sobre el RESIDUO `x - suavizada`, que no contiene
movimiento del tirador por construcción: p95 de **0,418 m/s de mediana** (0,063 a 2,263) en las 16
series pie-tirador de los 4 clips, así que el umbral son **1,25 m/s**.

En los 3 clips con escala sana (fuera `ForcefulAntiqueAlbacoretuna`, ver 26.1): **v1 n 26 con p50
0,139 m → v4 n 45 con p50 0,313 m y duración 0,30 s**.

**El número SUBE de 26 a 45, que es lo contrario de lo que yo había predicho al declarar la v2.** La
razón es que v1 no contaba pasos de menos por fusionar mal: contaba artefactos y se perdía pasos de
verdad. Lo que decide no es esta cifra: es la verificación a ojo de 26.7.

### 26.5 Un fondo dispara el detector de paso, y eso no se arregla con umbrales

**Encontrado mirando los recortes, no en ninguna cifra agregada.** En `AliveIgnorantBarebirdbat` el
evento `marcha l0 c49-55, 0,23 m, un pie` es el MISMO gesto que el `fondo l0 c46-57, 0,59 m`: se ve
en la tira de tres cuadros, con la separación de pies pasando de 0,70 a 1,23 m.

Y no es un fallo de implementación: **el criterio del paso describe exactamente la primera mitad de un
fondo**. Un fondo es el pie de delante saliendo disparado con el de detrás clavado. Las dos cosas son
el mismo suceso medido con la misma señal, así que ningún umbral las separa.

Lo que sí se puede hacer es decirlo. Cada paso lleva `solapa_fondo` en el CSV, y la contaminación está
medida:

| | pasos | solapan un fondo | con un solo pie |
|---|---:|---:|---:|
| los 4 clips | 87 | **36 (41 %)** | 36 (41 %) |
| `AliveIgnorantBarebirdbat` | 21 | 5 (24 %) | 10 (48 %) |
| `ForcefulAntiqueAlbacoretuna` | 42 | 24 (57 %) | 17 (40 %) |
| `GentleQuerulousElephantseal` | 14 | 6 (43 %) | 7 (50 %) |
| `UglyNiceAplomadofalcon` | 10 | 1 (10 %) | 2 (20 %) |

Filtrando `solapa_fondo == False` en los 3 clips de escala sana quedan **33 pasos con p50 0,313 m y
duración p50 0,30 s**, que sigue dentro del criterio declarado. Y los que mueven LOS DOS pies, que es
la definición de esgrima de un paso completo, son 19 con p50 **0,721 m**: el doble, como debe ser.

### 26.6 EL BOTE NO SE PUEDE MEDIR, y la cifra dice por qué

El usuario pidió *"aquí bota"*. Un bote (salto, balestra o appel) son las dos plantas separándose del
suelo a la vez. **La homografía de la pista no puede medirlo**: mapea el plano del suelo, y un pie a
10 cm se proyecta desplazado `altura x distancia / altura_de_cámara`, que es el mismo defecto que
hace que la transversal no sea creíble. Medir un salto con la homografía del suelo es medirlo con la
regla que ese salto rompe. Así que se mide en la IMAGEN, normalizado por el alto de la caja del
propio tirador, que sí es una reglilla válida para lo vertical.

**El suelo de ruido sale de la cola NEGATIVA de la propia señal, y eso es lo que hace que la medida
valga:** la señal es `nivel_del_suelo - planta`, y un pie **no puede estar por debajo del suelo**, así
que todo lo negativo es ruido de medida puro, sin ninguna hipótesis.

| medida, 8 series tirador-clip, 2.782 cuadros | valor |
|---|---:|
| valor absoluto del p05 de la señal, o sea el ruido | **0,0198** del alto de la caja = **3,5 cm** en un tirador de 1,75 m |
| umbral a 3x el ruido | **0,0595** = **10,4 cm** de despegue SIMULTÁNEO de las dos plantas |
| cuadros que lo pasan | **2 de 2.782**; el pico de cada serie se queda en 0,3 a 4,3x |
| botes emitidos con el umbral calibrado | **0** |

**Y eso no se arregla bajando el umbral:** una balestra o un appel de florete despegan del orden de
**5-8 cm**, que cae POR DEBAJO de los 10,4 cm que hacen falta para salir del ruido. Bajarlo a 5 cm
sería ponerlo dentro del ruido y llamar bote al temblor del keypoint del pie.

El tipo se deja en el código con umbral adaptativo (`umbral_bote`) y la columna `tipo_establecido`, y
va marcado **NO ESTABLECIDO** en el LEEME. La verificación a ojo lo confirma: en
`TestyWatchfulBarebirdbat` c299 hay lo que parece una flecha con los dos pies en el aire, la señal
llega a 0,043 (7,5 cm) y el umbral calibrado de ese tirador es **0,158, o sea 27 cm**, porque el
ruido de ese clip es enorme. El evento está ahí y el instrumento no lo puede ver.

### 26.7 La verificación a ojo, con acierto y falsos negativos por separado

**Los tipos de movimiento ya no se verifican con un fotograma, y eso era un defecto del método.**
Mirando un solo cuadro de una marcha no se sabe si el tirador avanzó, retrocedió o estaba quieto. Se
ha añadido `verifica_pista.tira`, que saca los cuadros de **inicio, pico y fin** del mismo evento con
los pies marcados y la posición en metros escrita en cada uno.

Recortes en `02_validacion/pista/verifica/`: 16 `ev_marcha`, 16 `ev_romper`, 13 `ev_fondo`, 11
`ev_parada`, 16 `ev_brazo`, y de candidatos no detectados 16 `fn_paso`, 15 `fn_fondo`, 8 `fn_parada`,
14 `fn_bote`. Candidatos propuestos por clip: 3-8 de fondo, 6-21 de paso, 0-7 de parada, 3-5 de bote.

Lo que MIRÉ, que es n = 8 y no una tasa:

| recorte | veredicto |
|---|---|
| `marcha` Gentle c189-200 l0, 0,31 m, 2 pies | **correcto**, avanza y la pista se desplaza detrás |
| `marcha` Alive c49-55 l0, 0,23 m, 1 pie | **es un FONDO**, el mismo que `fondo c46-57`. Origen de 26.5 |
| `romper` Ugly c172-202 l0, 1,23 m | movimiento **correcto**, pero es el tirador **volviendo andando** después del toque, no una rotura táctica |
| `parada` Gentle c190-206 l0, 0,26 m | **correcta**: avanza 0,26 m y luego 0,10 m |
| `fondo` Alive c46-57 l0, 0,59 m | **correcto**, guardia 0,70 → fondo 1,28 m |
| `brazo` Gentle c203-211 l0, 0,21 x torso | **correcto**, brazo extendido hacia el rival |
| `fn_bote` Alive c84 l1, señal 0,015 | **no es un bote**, es una recuperación de fondo. Bien rechazado |
| `fn_parada` Alive c105-117 l0 | **no es una parada**, sigue rompiendo 0,33 m. Bien rechazado |

O sea: **5 de 6 eventos marcados que miré son correctos**, el sexto es el solape fondo/paso de 26.5;
y de los 2 candidatos no detectados que miré, **los 2 están bien rechazados**. Más el caso de
`fn_bote` de 26.6, que es un evento real que el instrumento no puede ver.

**Un tipo con contaminación conocida no es un tipo verificado.** `marcha` y `romper` tienen el 41 %
de solape con fondos medido y la columna para filtrarlo; eso es lo que hay, no una tasa de acierto.

### 26.8 El eje del brazo se corrige, y con menos margen del que decía 22.3

`posicion._brazos` usaba `codo -> raíz de la mano`. La sección 22.3 ya había medido que el eje que
CRECE al extenderse es `hombro -> raíz`, porque `codo -> raíz` es un hueso y un hueso no se alarga; el
cambio no se había propagado. Medido otra vez aquí, coeficiente de variación del largo del eje
normalizado por el torso, 8 tiradores de los 4 clips:

| eje | p50 del cv | rango | gana en |
|---|---:|---|---:|
| `hombro -> raíz de la mano` | **0,298** | 0,141 a 0,420 | **5 de 8** |
| `codo -> raíz de la mano` | 0,223 | 0,096 a 0,508 | 3 de 8 |

Se adopta `hombro -> raíz`, que es la dirección que midió 22.3 (0,226 contra 0,108), **pero con mucho
menos margen y hay que decirlo entero: en 3 de los 8 tiradores el eje del codo varía más.** El
criterio declarado era "en la mayoría", y 5 de 8 lo cumple y no más.

Se añade la columna `completa`: el pico del tramo ≥ 0,90 del p98 del propio clip y ≥ 1,25 su mediana.
**"Completa" es relativa al clip y se dice**, porque no hay ninguna medida absoluta del largo del
brazo de un tirador. En `GentleQuerulousElephantseal` salen 4 completas de 9 extensiones.

### 26.9 La parada: dos umbrales fijos fallaron su propia prueba de margen, y el tercero es adaptativo

Regla del proyecto: el umbral tiene que estar 3 veces por encima del suelo de ruido de la señal con
la que se mide, y el ruido se mide sobre el residuo de la suavizada.

| intento | señal | ruido p95 medido | margen del umbral que había puesto |
|---|---|---|---|
| 1 | velocidad por diferencia centrada de ±2 cuadros | **0,06 a 1,08 m/s**, p50 0,35 | 0,60 m/s → **1,7x**, y en un tirador el ruido era MAYOR que el umbral |
| 2 | desplazamiento en ventana de 8 cuadros | **0,008 a 0,132 m**, p50 0,048 | 0,25 m para "venía" → 5,2x, **pasa**; 0,08 m para "está quieto" → **1,7x** sobre la mediana y **0,6x** sobre el peor |
| 3, el que se usa | ídem, umbral de quieto = máximo entre 0,08 m y 3 veces el ruido p95 de ese lado | — | 3x por construcción, y si pasa de 0,20 m el lado se declara **NO MEDIBLE** |

Consecuencia que hay que leer: **una "parada" de este detector significa que el tirador baja de
0,3-0,7 m/s, NO que esté inmóvil.** El umbral exacto de cada clip y tirador va en la columna
`umbral_quieto_m` del CSV, que es lo único honrado cuando el umbral es adaptativo. En
`TestyWatchfulBarebirdbat` los dos lados salen NO MEDIBLES y el clip entrega **0 paradas**, que es
el comportamiento correcto.

### 26.10 La serie temporal consultable

`02_validacion/pista/serie_eventos.csv` y `serie_eventos_<clip>.csv`: **una fila por evento**, no por
cuadro. `indice_del_tipo` va 1, 2, 3... por (lado, tipo) en orden de `cuadro_inicio`, **y es el mismo
número que sale en pantalla en el vídeo**: si no coincidieran, el vídeo y el CSV dirían cosas
distintas del mismo evento.

El criterio de que está bien hecho se declaró antes: la consulta del usuario tiene que ser una línea y
devolver una fila. Comprobado:

```python
ev = pd.read_csv("serie_eventos_GentleQuerulousElephantseal.csv")
ev.query("lado == 1 and tipo == 'marcha' and indice_del_tipo == 3")
# -> 1 fila: c204-207, t 6,798-6,898 s, x 1,323 -> 1,227 m,
#    magnitud 0,124 m, duracion 0,10 s, 1,245 m/s, 1 pie, pose_fiable True, hueco_max 0
```

Columnas de procedencia, que son las que dicen si una fila vale: `pose_fiable` (la pose del rango del
evento está MEDIDA, no interpolada), `origen_max`, `hueco_max`, `hueco_sin_medir` (hueco > 24, donde
el error de la pose inventada **no está ni medido**, sección 21.4), `solapa_fondo`,
`tipo_establecido`, `publicable`, `puerta_dos_lados`, `ancla_fiable`, `transversal_creible`.

**`unidad_magnitud` es una columna y no un comentario**, porque las magnitudes NO están todas en la
misma unidad: `fondo`, `marcha`, `romper`, `parada` y `accion` en metros a lo largo de la pista;
`bote` en metros verticales con la escala de la caja; `brazo` en fracción de torso. Meterlas todas en
una columna llamada `magnitud_m` sin decirlo sería el mismo error del convenio sin declarar que este
proyecto ya ha pagado dos veces.

284 filas en 11 clips. Y el orden de las columnas está escrito a mano: la versión anterior escribía
`sorted(todas las claves)` y salía un CSV con `tipo` en la duodécima columna. Un CSV que hay que leer
con el diccionario delante no es una serie consultable.

### 26.11 Los cinco vídeos, elegidos para cubrir los tres grados de credibilidad

`02_validacion/pista/video/trazado_<clip>.mp4`, 1280x720. Cuatro zonas sincronizadas: el clip con los
dos tiradores marcados y sus dos apoyos, el rótulo con cuadro / segundo / posición de cada uno /
distancia y **el estado de la medida**, el TRAZADO dibujándose, y el MAPA de la pista visto desde
arriba con la estela y los eventos apareciendo con su índice.

| clip | qué ejemplifica | cuadros | MB |
|---|---|---:|---:|
| `AliveIgnorantBarebirdbat` | el mejor caso: pasa la puerta **por tirador** (2 de 78) | 406 | 9,9 |
| `GentleQuerulousElephantseal` | pasa la agrupada, falla por tirador (izq 2,23 m); **único de los cinco con anclaje utilizable** | 310 | 6,0 |
| `UglyNiceAplomadofalcon` | ídem (der 1,48 m) | 270 | 6,9 |
| `ForcefulAntiqueAlbacoretuna` | pasa con **un solo tirador medido** y escala ~2,4x inflada | 359 | 9,3 |
| `TestyWatchfulBarebirdbat` | **NO pasa** la puerta: el rótulo lo dice en pantalla | 224 | 5,8 |

Cinco y no cincuenta porque el usuario ha avisado de que tiene el ordenador saturado y anoche se le
apagó. El fondo de cada panel se construye **una vez** con matplotlib y de ahí se saca la
transformación metros → píxel; todo lo que se mueve se pinta con `cv2` encima. Una figura de
matplotlib por fotograma serían 1.569 figuras.

**Un defecto de dibujo que sólo se vio recortando y ampliando un fotograma del MP4:** `cv2.putText`
con las fuentes Hershey **cambia el ancho de avance de cada glifo con el grosor**, así que dibujar el
contorno con `grosor + 2` y el relleno con `grosor` en la misma posición pinta dos textos de anchos
distintos superpuestos, y el resultado se lee como letras duplicadas y borrosas — que es literalmente
la queja que el usuario tenía de las salidas anteriores. El contorno se hace ahora desplazando el
MISMO grosor. Y las etiquetas de evento se pisaban entre ellas y tapaban el punto de posición, así
que se han movido a una banda reservada arriba del mapa.

### 26.12 Un criterio de aceptación que se cumplía por la razón equivocada

Lo cazó el coordinador **ejecutando**, no leyendo: `serie_eventos.csv` tenía **40 filas de un solo
clip** en vez de las 284 de 11. Los once ficheros por-clip sí sumaban 284, así que no se había
perdido ningún dato calculado: lo que estaba roto era el **agregado**.

El mecanismo, y es un fallo de diseño mío y no un despiste puntual: el `__main__` de
`grafica_pista.py` escribía los tres agregados (`serie_eventos.csv`, `eventos.csv`,
`posicion_y_eventos.csv`) **con los clips que se le pasaran en la línea de órdenes**. Al final de la
sesión volví a correr UN clip suelto para comprobar el arreglo del panel b, y esa pasada sobrescribió
los tres agregados de 11 clips con uno solo. Y `eventos.csv` y `posicion_y_eventos.csv` sólo existían
como agregado, así que no había de dónde recuperarlos sin recalcular.

**Lo que hay que retener es por qué mi propia verificación no lo detectó.** El criterio de aceptación
declarado era que `ev.query("lado==1 and tipo=='marcha' and indice_del_tipo==3")` devolviera **una
fila**. Lo corrí y devolvía una fila. Pero la devolvía **porque el fichero sólo tenía un clip**: sobre
el agregado de verdad tiene que devolver una fila **por clip** que tenga tres marchas de ese lado.
Un criterio que se cumple por la razón equivocada no es un criterio, y éste lo era porque no
comprobaba el **tamaño del conjunto** sobre el que se consultaba.

Arreglo, en el código y no en los datos:

 - los tres agregados **ya no se escriben con lo que se acaba de calcular**. Se escriben con
   `reune()`, que lee del disco TODOS los por-clip y los concatena, así que una pasada de un clip
   suelto es inofensiva **por construcción** y no por acordarse de pasar la lista entera.
 - `eventos_<clip>.csv` y `posicion_clip_<clip>.csv` se escriben por clip, como ya se hacía con
   `serie_eventos_<clip>.csv`. Ningún agregado es ya la única copia de nada.
 - dos **asertos** dentro de `reune()`: las filas del agregado igualan la suma de los por-clip, y los
   clips distintos del agregado igualan el número de ficheros por-clip.
 - y el agregado se excluye del glob explícitamente, porque `posicion_*.csv` casaba con
   `posicion_y_eventos.csv`, que es el propio agregado.

**Comprobado reproduciendo el fallo a propósito**: se volvió a correr un clip suelto y los agregados
se quedaron en 284 filas de 11 clips.

Estado de los seis agregados, con su número de clips distintos comprobado con aserto:

| agregado | filas | clips |
|---|---:|---:|
| `serie_eventos.csv` | 284 | 11 |
| `eventos.csv` | 284 | 11 |
| `posicion_y_eventos.csv` | 11 | 11 |
| `homografia.csv` | 78 | 78 |
| `censo_pista.csv` | 13 | 13 |
| `video/videos.csv` | 5 | 5 |

Los tres últimos no estaban afectados: los escriben otros `__main__` a los que siempre se les pasó la
lista completa. Que se salvaran fue **suerte**, no diseño, y por eso llevan ahora su aserto también.

Y el criterio de aceptación, corrido sobre el agregado bueno y con el aserto que le faltaba:

```python
ev = pd.read_csv("serie_eventos.csv")            # 284 filas, 11 clips
ev.query("lado == 1 and tipo == 'marcha' and indice_del_tipo == 3")
# -> 4 filas, de 4 clips distintos, que son EXACTAMENTE los 4 clips
#    que tienen 3 o mas marchas del tirador derecho
```

| clip | cuadros | t inicio | x inicio → fin | magnitud | duración | velocidad |
|---|---|---:|---|---:|---:|---:|
| `AliveIgnorantBarebirdbat` | 80-83 | 2,666 s | +1,415 → +1,194 m | 0,222 m | 0,10 s | 2,22 m/s |
| `ForcefulAntiqueAlbacoretuna` | 16-18 | 0,534 s | −1,815 → −2,016 m | 0,101 m | 0,07 s | 1,51 m/s |
| `GentleQuerulousElephantseal` | 204-207 | 6,798 s | +1,323 → +1,227 m | 0,124 m | 0,10 s | 1,25 m/s |
| `TestyWatchfulBarebirdbat` | 269-281 | 8,967 s | +1,513 → +0,848 m | 0,665 m | 0,40 s | 1,66 m/s |

Añadiendo `clip == '<nombre>'` a la misma consulta se vuelve a una fila. Sigue siendo una línea.

### 26.13 LO QUE ESTA MEDIDA NO DICE

- **No dice que la posición sobre la pista esté resuelta.** El anclaje del mapa es una **convención**
  y su deriva medida sólo baja de 1,00 m en **2 de 11 clips**. En los otros 9 la posición sobre el
  mapa es ilustrativa, y el rótulo del vídeo lo dice cuadro a cuadro. Lo que está en metros son los
  desplazamientos.
- **No dice que la escala longitudinal esté establecida.** Sigue en pie lo de 23.5: el control
  cruzado entre dos vías independientes da razones de 0,28 a 2,57. Esta sección añade un control
  físico nuevo (la deriva del punto medio) que **descarta un clip publicable**, no que valide los
  demás.
- **No dice que `marcha` y `romper` sean pasos de esgrima.** El 41 % de los detectados solapan un
  fondo, y esa contaminación es **irreducible con esta señal** porque los dos sucesos son el mismo
  movimiento. Y en el único romper que miré con calma, el tirador estaba volviendo andando después
  del toque: el detector no distingue una rotura táctica de andar hacia atrás.
- **El `bote` no está medido, está descartado con su cifra.** Hacen falta 10,4 cm de despegue
  simultáneo para salir del ruido y una balestra despega 5-8 cm. Cero eventos emitidos. Si alguien
  quiere este evento, el camino no es un umbral: es más resolución o una señal distinta.
- **La verificación a ojo es n = 8.** Seis marcados y dos candidatos. No es una tasa de acierto: es
  lo que se pudo mirar, y encontró un defecto de fondo (26.5), lo cual dice más de lo que dice el
  5 de 6.
- **La magnitud de la marcha cae en el rango declarado y eso no prueba que sea correcta.** Prueba que
  es compatible con la física de un paso de florete. Sin anotación humana de dónde empieza y acaba
  cada marcha no hay error en metros, y este corpus no la tiene.
- **El eje del brazo gana en 5 de 8 tiradores, no en 8 de 8.** El cambio va en la dirección que midió
  22.3 y con mucho menos margen. En 3 tiradores el eje del codo varía más.
- **No se ha vuelto a correr la homografía de los 78 clips.** Las tres columnas nuevas de
  `homografia.csv` son función pura de dos columnas que ya estaban. Si alguien cambia `pista.py` en
  lo que toca a la homografía, esas columnas hay que recalcularlas corriendo de verdad.
- **Los `homog_*.png` no se han borrado, se han movido.** Están en `diagnostico/` y siguen siendo la
  única forma de ver POR QUÉ una homografía sale mal. Lo que se afirma es que no son un entregable.
- **Los eventos se solapan entre tipos a propósito.** Una `parada` empieza donde acaba una `marcha`,
  y un `brazo` puede coincidir con un `fondo`. No son una segmentación exclusiva del asalto: son
  detecciones independientes, cada una con su rango de cuadros, y quien las use tiene que decidir
  qué hacer con los solapes.

### 25.10 · La premisa de la que partimos NO es reglamentaria, y la medida la desmiente

Esto conviene dejarlo escrito porque es el punto de partida del encargo. El usuario dijo: *«la mano
dominante siempre tiene el guante y la otra va sin guante»*.

- **La primera mitad sí es reglamentaria**: t.115.1 dice que guante y arma van en la mano armada.
- **La segunda mitad no lo es.** En el Reglamento Técnico de diciembre de 2025 **no hay ningún
  artículo que prohíba un guante en la mano no armada**, y el artículo que el encargo citaba (m.25)
  **no está en ese PDF**: pertenece al Reglamento del *Material*, que no está en disco en español.
  Lo verificado en fuente primaria es t.115.1, no m.25.
- **Y la medida la desmiente en los datos.** No sólo es que el rasgo no separe (0,444 de acierto,
  separación −0,002): hay tiradores con **guante oscuro** en la mano armada
  (`UnevenChiefHippopotamus l0`, `WhiteMarriedGuineafowl l1`, este último medido con lupa en
  H 75-80 / S 89-112 / V 88-126), y el **32,6 %** de los parches tienen puño de color. Así que «la
  armada tiene guante claro y la otra piel» no es un hecho del deporte: es una costumbre frecuente.

Esto no le quita mérito a la pista: era razonable, era gratis de probar, y **el motivo por el que
falla es lo que queda establecido**, no una cifra de muestra pequeña.

### 25.11 · Las siete puertas de «cuadro que sabe la respuesta»: la hipótesis era buena y la variable estaba mal elegida

El diagnóstico de 25.4 —cinco de las siete fallas son cuadros con el tirador no encarado— contiene
una vía de arreglo evidente: si la decisión es una por clip, **no todos los cuadros tienen que
votar**. Se declararon seis puertas y dos combinaciones **antes** de medirlas (punto 10 del docstring
de `mano_guante.py`), con el número que decide: **mejora si acierta ≥ 0,90 sobre los tiradores que
decide y deja ≤ 4 de 27 sin decisión**, con `VOTA_MIN = 10` cuadros mínimos.

| puerta | acierto | decide | sin decidir | acierto POR CUADRO dentro / fuera |
|---|---:|---:|---:|---|
| sin puerta | **0,815** | 27 | 0 | 0,805 / — |
| `alcance` (manos de los dos a ≤ 2 × 0,90 m) | 0,778 | 27 | 0 | 0,819 / 0,797 **(+0,022)** |
| `extendido` (algún brazo por encima de su mediana) | 0,741 | 27 | 0 | 0,786 / 0,825 **(−0,040)** |
| `perfil` (hombros proyectados por debajo de su mediana) | 0,741 | 27 | 0 | 0,762 / 0,848 **(−0,086)** |
| `ev_accion` (eventos del agente de la pista) | 1,000 | **2** | 25 | 0,939 / 0,791 (+0,148) |
| `ev_brazo` (eventos del agente de la pista) | 1,000 | **2** | 25 | 0,907 / 0,802 (+0,104) |
| **`encarado`** (signo: nariz del lado del rival) | **0,852** | 27 | 0 | **0,860 / 0,582 (+0,277)** |
| `alcance+extendido` | 0,727 | 22 | 5 | 0,772 / 0,814 |
| `alcance+extendido+perfil` | 0,769 | 13 | 14 | 0,772 / 0,809 |
| `alcance+encarado` | 0,778 | 27 | 0 | 0,852 / 0,783 |

**Ninguna llega a 0,90, así que ninguna se adopta.** Pero la tabla enseña tres cosas que valen más
que si hubiera funcionado:

**1. La hipótesis era correcta y mis cuatro primeras variables estaban mal elegidas: todas miden una
MAGNITUD y ninguna mide un SIGNO.** Una distancia, una extensión, una anchura y una pertenencia a un
intervalo son iguales para un tirador encarado y para uno de espaldas. La quinta variable, declarada
después de ese diagnóstico y antes de medirla —la nariz (índice 0, que en careta tiene **más**
confianza que las manos) tiene que caer del lado del rival respecto al punto medio de los hombros—
separa **0,860 dentro contra 0,582 fuera, +0,277 sobre 3.479 cuadros**. Y dos de las de magnitud van
**al revés**: los cuadros con el brazo más extendido aciertan **0,040 menos** y los de perfil **0,086
menos**. O sea que «brazo extendido» y «de perfil» no son señales de estar encarado.

**2. Y aun con +0,277 por cuadro, la decisión por clip sólo pasa de 0,815 a 0,852, un caso.** La
razón está medida y es la respuesta de fondo: **el fallo no es de unos cuantos cuadros, es del
tirador entero.** El acierto por cuadro del criterio (a), tirador a tirador, es **bimodal**:

| grupo | acierto por cuadro, p50 | extremo |
|---|---:|---|
| los 20 tiradores que (a) acierta | **0,960** | mínimo 0,324 |
| los 7 tiradores que (a) falla | **0,226** | y cuatro de ellos en 0,000 / 0,000 / 0,053 / 0,226 |

Cuando un tirador está mal en el **100 %** de sus cuadros **no hay ningún cuadro bueno que salvar**, y
por eso quitar cuadros no arregla nada. Los tres casos que **ninguna** puerta arregla son justo esos.

**3. Los eventos de la pista no se pueden usar como puerta hoy, y además se movieron debajo.** Sólo
**2 de los 27** tiradores tienen eventos `accion` o `brazo` en `02_validacion/pista/eventos.csv`. Y un
aviso de reproducibilidad: ese fichero lo está reescribiendo el agente de la pista, y entre dos
ejecuciones mías la cobertura pasó de **9 tiradores a 2**. Por eso `mano_guante.json` guarda
`eventos_csv_mtime_ns`; la tabla de arriba es la del `mtime` 1789114860959670600. Cuando ese agente
cierre su tirada, `ev_accion` y `ev_brazo` merecen volver a medirse: son las dos puertas con mejor
acierto por cuadro dentro (0,939 y 0,907) y las únicas que **no he calculado yo**.

**La marca por tirador tampoco paga**, medida con umbral declarado en 0,50 (la mayor parte del clip):
`en alcance < 0,50` marca 19 de 27 y los 8 no marcados aciertan **0,750**; `encarado < 0,50` marca 4
de 27, caza 2 de los 7 fallos, y los 23 no marcados aciertan **0,783**. Las dos son peores que el
0,815 de no marcar nada.

### 25.12 · Los dos casos «seguro y equivocado», vistos de cerca

Son los dos más valiosos porque rompen la idea de que un margen grande signifique algo. Figura nueva:
`02_validacion/muestras/mano_dominante/lupa_seguro_y_equivocado.png`.

- **`WhiteMarriedGuineafowl l0`** (AVOLA, margen 145,4 px, acierto por cuadro **0,000 sobre 84**).
  Ampliando el tirador entero se ve sin discusión: está **de espaldas a la cámara y parado en su
  línea** entre dos acciones, y **la hoja sale como una recta oscura hacia arriba-izquierda de la mano
  `B`**, que es la del lado **opuesto** al rival (el rival está en x 712-861, fuera del recorte). La
  otra mano, `A`, está vacía. El criterio (a) elige `A` en los 84 cuadros.
- **`UglyNiceAplomadofalcon l1`** (BUTRUILLE, margen 65,3 px, acierto por cuadro **0,000 sobre 37**).
  Con lupa, la mano `B` agarra el arma con **la hoja oscura cruzando la pista azul** en k=227, 241 y
  257, mientras `A` es el torso con un guante claro y sin arma. Y el rival está a x 333, 242 y 100,
  o sea a la **izquierda**, con `B` a la derecha: el arma está en la mano del lado contrario durante
  todo el clip.

**Ninguna puerta caza a estos dos**, y `encarado` tampoco: da 0,57 y 0,16 de cuadros encarados, o sea
que en `WhiteMarried` la nariz dice que sí está encarado en el 57 % de los cuadros y se equivoca
—con la careta vista desde atrás el punto de la nariz no es fiable—. Eso acota lo que el test de
signo puede hacer y es, otra vez, un límite del dato y no del criterio.

### 25.13 · Lo que la tubería debe usar, y su punto ciego

Para el agente del cono, que hereda el vértice: **tratar `frac_a_voto < 0,85` como `mano dudosa` y
excluir ese tirador**. Sobre esta muestra:

| decisión | tiradores | acierto |
|---|---:|---:|
| todos | 27 | 0,815 |
| **sólo `firme` (`frac_a_voto ≥ 0,85`)** | **19 (0,70)** | **0,895** |
| los apartados (`frac_a_voto < 0,85`) | 8 | 0,625 |

Y el **punto ciego, medido**: los 2 fallos que se cuelan con `firme == True` tienen `frac_a_voto` de
**1,00** y son exactamente los dos de 25.12. Es por construcción: **la unanimidad del voto mide
consistencia, no verdad**, y un error sistemático es perfectamente consistente. Así que `firme` sirve
para no heredar decisiones flojas y **no** para detectar decisiones seguras y equivocadas; para esas
haría falta ver el arma, que es lo que 25.5 midió que hoy no se puede a 720p.

### Lo que 25.10-25.13 NO dicen

- **No dicen que las puertas no sirvan, dicen que estas siete no llegan a 0,90 en 27 tiradores.** Y
  una de ellas, `encarado`, sí separa por cuadro con +0,277, así que la idea de pesar cuadros está
  viva; lo que está medido es que **con la decisión por clip ya hecha por mayoría, pesar cuadros
  apenas mueve el resultado**, porque el fallo es del tirador entero.
- **El `1,000` de `ev_accion` y `ev_brazo` son 2 tiradores.** No es una cifra, es una anécdota con
  cobertura del 7 %. Y el fichero de eventos cambió entre dos ejecuciones mías, así que ni siquiera
  es estable todavía.
- **La bimodalidad (0,960 contra 0,226) se apoya en 20 y 7 tiradores.** Es un reparto muy limpio,
  pero con 7 casos en el modo malo no se puede decir qué fracción del corpus cae ahí.
- **El umbral 0,85 de `firme` se declaró antes, pero con 27 tiradores está débilmente determinado**, y
  el reparto por tramos tiene 19 y 8 casos. Lo que no depende de la muestra es el punto ciego: la
  unanimidad no puede detectar un error sistemático, y eso es un argumento, no una medida.
- **La nariz como test de signo falla con la careta vista desde atrás**, medido en el caso que más
  importaba (0,57 de cuadros «encarados» en un tirador que está de espaldas todo el clip). Un test de
  signo mejor —el orden de los pies a lo largo de la pista, por ejemplo, que sí tiene sentido físico
  en esgrima— está sin probar.

### 25.14 · El hecho de esgrima que nadie había usado: el pie adelantado y la mano armada son del mismo lado

En guardia **el pie adelantado es el del mismo lado que el brazo armado**. Un diestro adelanta el pie
derecho y lleva el florete en la mano derecha. Es la posición básica del deporte, no una correlación
de estos datos. Y lo que la hace valiosa aquí es que es una correspondencia **dentro del cuerpo**, así
que no se rompe en los dos casos donde todo lo demás se rompe: los tiradores con el arma en la mano
del lado **opuesto** al rival durante todo el clip.

**Antes de nada, el mapa de los pies, MEDIDO.** El precedente de los índices 91-95 —que «estaban
verificados» y caían sobre la careta— obliga. Tres pruebas sobre los 17 clips, 32 tiradores:

| prueba | resultado |
|---|---|
| **T1** el talón es el punto del grupo más cercano al tobillo (en anchuras de hombro) | 15 → 0,505 · 16 → 0,460 · **17 → 0,209** ; 18 → 0,498 · 19 → 0,463 · **20 → 0,196**. Así que **17 y 20 son los talones** |
| **T2** cv de la longitud del hueso, menor en el emparejamiento correcto | cadera-rodilla (10,12) 0,079 contra (10,11) 0,113 ; rodilla-tobillo (12,14) 0,109 contra (12,13) 0,242 ; tobillo-talón (14,20) 0,146 contra (14,17) 0,423 |
| **T2, el eslabón que NO resuelve** | hombro-cadera: (6,10) **0,064** contra (6,9) **0,069**, un 8 % de diferencia, o sea nada. Y es justo el eslabón que une el lado de la mano con el del pie |
| **T3** coherencia lateral: fracción de cuadros en que el signo del desvío lateral coincide con el del hombro 6 | **cadera 10 → 0,989** contra cadera 9 → 0,011 ; rodilla 12 → 0,934 ; **tobillo 14 → 0,837** contra tobillo 13 → 0,163 ; talón 20 → 0,837. De control: codo 8 → 0,989 y mano A → 0,949, que reproducen el mapa que ya estaba |

**Cadena de lado verificada**, en `mano_guante.PIES`:

```
mano A (raíz 21) - codo 8 - hombro 6 - cadera 10 - rodilla 12 - tobillo 14 - pie 18,19,20
mano B (raíz 42) - codo 7 - hombro  5 - cadera  9 - rodilla 11 - tobillo 13 - pie 15,16,17
```

**El punto delicado era definir «adelantado» sin volver a usar al rival**, porque «el pie más cerca
del rival» es el criterio que ya falla. Dos vías declaradas antes de medirlas:

- **`pie_x`** — el eje de cada pie es `talón → punto medio de los dos dedos`. En guardia el pie de
  delante apunta a lo largo de la pista y el de atrás queda casi perpendicular. Gana el pie cuyo eje
  está más alineado con la horizontal, y la mano armada es la del **mismo lado**. El rival no aparece.
- **`pie_cadera`** — igual pero el eje de referencia es el de las caderas, así no se supone que la
  pista sea horizontal.
- **`nariz_dir`** — el criterio (a) con el rival sustituido por la orientación del propio tirador:
  `dir = sign(x_nariz − x_medio_hombros)`, y gana la mano más adelantada en esa dirección. La nariz
  tiene **más** confianza que las manos (0,998-1,000 contra 0,648-0,987, sección 2).

**Y la tercera vía que se propuso, «el pie que queda por delante del plano del tronco», NO SIRVE, y
no hace falta medirlo para saberlo:** proyectar los tobillos sobre el eje de las caderas devuelve
**el lado, no el delante**. El tobillo del lado A siempre proyecta positivo, estén los pies como
estén, porque el eje de las caderas es el eje **lateral** del cuerpo. Para sacar «delante» de ahí hace
falta saber hacia dónde mira el tirador, o sea el rival o la nariz: sería el mismo problema con otro
nombre. Queda escrito.

### 25.15 · Las cifras: `nariz_dir` se queda a un caso de la puerta, y la firmeza del pie cierra el punto ciego

Sobre los mismos 27 tiradores, con la puerta de adopción **sin mover** (≥ 0,90 con ≤ 4 sin decidir):

| criterio | acierto | sin decidir | mira al rival |
|---|---:|---:|---|
| (a) «la más cerca del rival», el operativo | 0,741 | 0 | sí |
| `a_voto` | 0,815 | 0 | sí |
| `pie_cadera` | 0,778 | 0 | **no** |
| **`pie_x`** | **0,852** | 0 | **no** |
| **`nariz_dir`** | **0,889** | 0 | **no** |
| `cuerpo_2de2` (contesta sólo si pie y nariz coinciden) | 0,885 | 1 | **no** |
| `mayoria_3` (`a_voto` + `pie_x` + `nariz_dir`) | 0,889 | 0 | mezcla |

**Ninguno llega a 0,90, así que ninguno se adopta como operativo y la puerta no se mueve.** Pero
`nariz_dir` se queda a **un caso** (24 de 27; 0,90 pediría 25) y **bate al criterio actual en cuatro
casos, de 0,741 a 0,889, sin mirar al rival ni una vez**. La vía del cuerpo es la mejor que se ha
medido en todo este trabajo.

Y lo que sale mejor de todo es el cruce con la firmeza, **con el corte 0,85 que ya estaba declarado
para `frac_a_voto` y que aquí se reutiliza sin volver a elegirlo**:

| criterio | `frac ≥ 0,85` | `frac < 0,85` |
|---|---|---|
| `a_voto` | 0,895 (17 de 19) | 0,625 (5 de 8) |
| **`pie_x`** | **1,000 (13 de 13)** | 0,714 (10 de 14) |
| **`nariz_dir`** | **1,000 (11 de 11)** | 0,812 (13 de 16) |

**Y eso cierra en parte el punto ciego de `frac_a_voto`.** Los tres tiradores que ningún criterio
acierta tienen `frac_pie_x` de **0,73, 0,77 y 0,72**, o sea **por debajo de 0,85**, mientras que
`frac_a_voto` les da **1,00, 0,83 y 0,68**. Dicho de otra forma: **la firmeza del pie sí los marca
como dudosos y la del voto contra el rival no.** El precio es cobertura: `pie_x` firme decide 13 de
27 (0,48) y no falla ninguno.

### 25.16 · Qué pasa con los casos «seguro y equivocado»: uno cae, tres sobreviven, y la razón es de esgrima

| tirador | a ojo | (a) | `a_voto` | `pie_x` | `nariz_dir` |
|---|---|---|---|---|---|
| `UglyNiceAplomadofalcon l1` (arma en el lado opuesto al rival todo el clip) | B | A ✗ | A ✗ | **B ✓** | **B ✓** |
| `WhiteMarriedGuineafowl l0` (margen 145 px, de espaldas) | B | A ✗ | A ✗ | A ✗ | A ✗ |
| `WhiteMarriedGuineafowl l1` | B | A ✗ | A ✗ | A ✗ | A ✗ |
| `PowerfulAllLarva l1` | B | A ✗ | A ✗ | A ✗ | A ✗ |

**El hecho de esgrima se confirma exactamente en el caso para el que se propuso**: el tirador con el
arma en la mano del lado opuesto al rival durante el 100 % de los cuadros lo aciertan **las dos**
señales del cuerpo, y lo falla **todo** lo que mira al rival.

**Y los otros tres sobreviven por una razón que también es de esgrima**: esos tiradores **no están en
guardia** en los cuadros utilizables. Están parados en su línea o andando entre acciones, con el arma
atrás. La correspondencia pie-adelantado ↔ mano-armada es una propiedad de la **guardia**, no de una
persona de pie, así que fuera de guardia el criterio no es que se equivoque: es que no aplica. Y su
propia firmeza lo dice (0,73, 0,77 y 0,72).

**La concordancia de las dos señales de signo no cierra el caso**, y era la esperanza: por clip
coinciden en **26 de 27**, pero por cuadro sólo en **0,77 de mediana** (p10 0,51, p90 0,96). O sea que
a nivel de clip **no son tan independientes como parecían**, y `cuerpo_2de2` no añade nada (0,885
sobre 26). La única discrepancia por clip es `WhisperedTenderDeinonychus l0`, donde la nariz acierta y
el pie falla; la figura `discrepa_pie_nariz_WhisperedTenderDeinonychus_l0.png` pinta los ejes
`talón → dedos` de los dos pies y se ve el motivo: el tirador **está andando**, y en dos de los tres
cuadros el pie alineado con la pista es el A y en el tercero el B.

### Lo que 25.14-25.16 NO dicen

- **No dicen que el pie resuelva la mano armada.** Dicen 0,852 sobre 27 tiradores, y 1,000 sobre los
  13 en los que su propio voto es firme. **13 casos sin fallo no es una tasa del 100 %**: con 13
  observaciones el intervalo de confianza inferior anda por 0,75-0,80, así que lo honesto es «no ha
  fallado ninguno de 13», no «acierta siempre».
- **`nariz_dir` no se adopta porque se queda en 0,889 y la puerta declarada era 0,90.** Se podría
  haber movido la puerta a 0,88 y cantar victoria; no se mueve. Lo que se hace es dejar las dos
  cifras y el motivo.
- **El corte 0,85 se reutiliza, no se ajusta** —venía declarado para `frac_a_voto`—, pero eso no lo
  convierte en óptimo para `pie_x`: nadie ha barrido ese umbral para el pie, y con 27 tiradores no se
  debe.
- **El mapa de los pies está medido, pero el eslabón hombro-cadera lo resuelve T3 y no el cv.** T3 es
  una prueba de coherencia del propio modelo: si el modelo asignase los lados al revés de forma
  sistemática en algún tipo de plano, T3 lo daría por bueno igual. Lo que T3 garantiza es que la
  cadena es **autoconsistente**, no que «10» sea la cadera derecha anatómica. Para lo que se usa aquí
  —emparejar mano con pie— la autoconsistencia basta, y eso es lo que se afirma.
- **`pie_cadera` sale peor que `pie_x` (0,778 contra 0,852)** y es el más intrínseco de los dos. O sea
  que apoyarse en que la pista es horizontal **ayuda**, y por tanto `pie_x` heredaría el problema en
  cualquier clip con la cámara girada. No hay ningún clip así en esta muestra, así que eso no está
  medido: está supuesto.
- **Las filas `ev_accion` y `ev_brazo` de 25.11 siguen sin ser finales.** `eventos.csv` ha cambiado
  **tres veces** debajo de mis ejecuciones y la cobertura de mis 27 tiradores ha ido **9 → 2 → 9**
  (`mtime_ns` 1789114323744883600, 1789114860959670600, 1789115239901689800, guardados en
  `mano_guante.json`). Se remiden cuando el agente de la pista cierre, no antes.
- **Y no cambia la conclusión de fondo**, que sigue midiéndose igual: el fallo es **del tirador
  entero** (acierto por cuadro bimodal, p50 0,960 contra 0,226). Lo que 25.14-25.16 añaden es que un
  criterio decidido **una vez por clip y con información del propio cuerpo** sí ataca eso, mientras
  que filtrar cuadros no podía — y que aun así se queda a un caso de la puerta.

### 25.17 · ADOPTADO: el criterio del pie con firmeza, y el resto se marca `mano dudosa`

Decisión del usuario. `cono_lib.mano_armada` devuelve desde ahora **la mano del criterio del pie
cuando su voto es firme** (`frac_pie >= 0,85`), y cuando no lo es devuelve la del criterio (a) **con
`dudosa = True`**. La tubería —el cono, sobre todo— tiene que mirar `dudosa` y excluir a ese tirador,
no fiarse de `mano` a secas.

| política | decide | acierto |
|---|---:|---|
| **pie firme (lo adoptado)** | **13 de 27 (0,48)** | **no ha fallado ninguno de 13** |
| (a) sobre todos (lo viejo) | 27 de 27 | 0,741, siete fallos |
| lo adoptado ignorando `dudosa` | 27 de 27 | 0,815 |

**«No ha fallado ninguno de 13» no es «acierta siempre».** Con 13 observaciones el intervalo de
confianza inferior anda por 0,75-0,80, y esa frase va pegada a la cifra en el docstring de
`cono_lib.mano_armada`, en `cono_lib.USA_PIE_FIRME` y aquí.

**Es reversible y comparable**: `cono_lib.USA_PIE_FIRME = False` devuelve el comportamiento anterior
sin tocar nada más, y las dos columnas (`mano_a` del criterio viejo y `pie`/`frac_pie` del nuevo) se
calculan siempre, así que la comparación no exige re-ejecutar nada. Precedente del interruptor:
`USA_CRUDAS = False` en `extrae_pareja.py`.

**Lo que la función publica ahora**, todo aditivo y con la llamada de 6 argumentos funcionando igual:
`mano`, `mano_a`, `pie`, `frac_pie`, `firme_pie`, `hombro`, `frac_hombro`, `firme_hombro`, `dudosa`,
`por`, más las columnas de firmeza y guante de 25.7 y 25.5.

**El radio de impacto para los otros agentes, medido, y es pequeño.** Sobre los **44 tiradores** que
hoy tienen poses:

| | tiradores | |
|---|---:|---|
| con decisión firme (`dudosa = False`) | **18 (0,41)** | la tubería puede usarlos |
| de esos, los que **cambian** de mano respecto de (a) | **2** | `GentleQuerulousElephantseal l1` (B→A) y `UglyNiceAplomadofalcon l1` (A→B) |
| marcados `dudosa` | 26 (0,59) | la tubería debe excluirlos |

O sea que la adopción **mueve dos vértices de 44**, y los dos **de mal a bien** según la anotación a
ojo: eran dos de los siete fallos de (a). Todo lo demás sigue devolviendo lo mismo que antes; lo que
cambia de verdad es que ahora hay una marca para los 26 que no se deben usar.

**Y una avería que me hice yo al adoptarlo, dicha porque es del tipo que no avisa.** Al pasar
`mano_armada` a devolver el criterio nuevo, la columna de la **línea base** en mi propio careo dejó de
ser el criterio (a) y pasó a ser el adoptado: el 0,741 subió a 0,815 solo, sin que nada fallara. Está
arreglado tomando explícitamente `det["mano_a"]` y está escrito en el código, porque una línea base
contaminada convierte cualquier mejora en una tautología.

**Una nota de implementación que también es una medida.** El criterio del pie estaba dos veces: en
`mano_guante.criterios_cuerpo` (para medirlo) y en `cono_lib.mano_pie` (para usarlo). Las dos daban
**la misma mano** pero **distinta `frac_pie`**, porque `cono_lib` no exigía que las dos manos fueran
legibles. Se unificó exigiéndolo —la pregunta es cuál de las dos manos, así que media pregunta no
vale, y el vértice del cono *es* la raíz de la mano— y ahora **coinciden en 46 de 46 tiradores**. El
careo lo comprueba en cada ejecución.

### 25.18 · La idea del hombro: empata con el pie, amplía la cobertura, y el cruce de brazo es pequeño

El usuario: *«el brazo armado POR REGLA siempre tiene que estar POR DELANTE del brazo no armado, tipo
se mide por el hombro»*. Es cierto y es de esgrima. Y tiene tres ventajas sobre la mano: los hombros
son el **18,7-21,5 % del alto de la caja** mientras las raíces de mano **colapsan en el mismo píxel**
en 2 de 34 tiradores; mira el tren **superior**, así que es otra cosa que el pie; y aguanta el caso
que preocupaba al usuario, «se pasa el brazo por detrás», porque **la mano cruza el cuerpo y el hombro
no**.

El punto delicado era el mismo que con las caderas —«delante» necesita una dirección, y si es «el
hombro más cerca del rival» hemos vuelto al criterio que falla—, así que se declararon tres vías antes
de medirlas, ninguna con el rival dentro:

| criterio | qué dirección usa | acierto | con su `frac ≥ 0,85` |
|---|---|---:|---|
| `hombro_pie` | el eje `talón → dedos` del pie que gana en ese cuadro | 0,852 | **1,000 sobre 14** |
| `hombro_nariz` | el signo de la nariz respecto al medio de los hombros | **0,889** | **1,000 sobre 11** |
| `hombro_conf` | especulativa: el hombro más visible sería el de delante | **0,741** | 1,000 sobre 2 |

**1. El hombro empata con el pie y con la nariz; no los bate.** `hombro_pie` da exactamente lo mismo
que `pie_x` (0,852) y `hombro_nariz` exactamente lo mismo que `nariz_dir` (0,889), caso por caso. Y
eso tiene sentido y es el resultado que importa: **lo que decide es la DIRECCIÓN (el pie o la nariz),
no el punto que se proyecta sobre ella.** Cambiar la mano por el hombro no cambia la respuesta.

**2. Pero sí amplía la cobertura de lo adoptado, y gratis.** La combinación declarada
`pie_o_hombro` —si el pie es firme, el pie; si no y el hombro es firme, el hombro; si no, sin
decisión— da **1,000 sobre 15 tiradores, o sea 0,56 de cobertura contra 0,48 del pie solo**, sin
ningún fallo nuevo. Está implementado en `cono_lib` con el interruptor
**`USA_HOMBRO_SI_PIE_FLOJO`, apagado por defecto**, porque la adopción la decidió el usuario y lo que
decidió fue el pie; encenderlo es una línea y está medido lo que compra. Verificado que el interruptor
hace lo que dice: apagado decide 13 con 1,000, encendido decide 15 con 1,000.

**3. `hombro_conf` era especulativo y se retira con su cifra: 0,741**, que es clavado a la línea base.
**El hombro más visible no es el de delante.** Se midió antes de creerlo, que era la instrucción.

**4. La preocupación del usuario, medida, y resulta PEQUEÑA.** `frac_cruza_brazo` es la fracción de
cuadros en que el orden en x de las dos **manos** es el contrario del de los dos **hombros**, o sea
la proxy de «se pasa el brazo por detrás» o «invierte la guardia»:

| | p10 | p50 | p90 | máximo |
|---|---:|---:|---:|---:|
| sobre los 27 tiradores | 0,00 | **0,04** | 0,14 | 0,32 |
| en los 20 que (a) acierta | | 0,03 | | |
| en los 7 que (a) falla | | 0,07 | | |

El cruce **existe** —hasta el 32 % de los cuadros en el peor tirador— pero **no es la causa de los
fallos**: sube de 0,03 a 0,07 entre los que se aciertan y los que se fallan, que no explica nada. La
causa sigue siendo la que ya estaba medida: **el tirador no está en guardia**.

**5. Y los tres irreducibles siguen irreducibles, pero la política los aparta.**
`WhiteMarriedGuineafowl l0` y `l1` fallan con las **cuatro** señales del cuerpo, y
`PowerfulAllLarva l1` lo acierta sólo `hombro_nariz`. Lo que importa es que **los tres tienen el pie
NO firme** (0,73, 0,77 y 0,72), así que **la política adoptada los marca `dudosa` y los excluye**, que
es exactamente lo que se le pedía.

### Lo que 25.17-25.18 NO dicen

- **No dicen que la mano armada esté resuelta: dicen que ahora se sabe cuándo callarse.** Lo adoptado
  contesta en el 48 % de los tiradores y en ese 48 % no ha fallado ninguno de 13. El otro 52 % no se
  pierde: se marca. Si la tubería no mira `dudosa`, el acierto es 0,815 y no 1,000.
- **13 y 15 casos son pocos.** Un 1,000 sobre 13 y un 1,000 sobre 15 tienen el intervalo inferior
  alrededor de 0,75-0,80. Lo honesto es «no ha fallado ninguno», y la muestra tiene que crecer.
- **Y ya se puede crecer**: hay **23 clips con poses en la lista del corpus, 44 tiradores medidos**,
  contra los 17 clips y 27 tiradores que tienen verdad a ojo. Sobre los 44, el pie firme decide **18
  (0,41)**, que es la estimación de cobertura con la muestra más grande disponible; el acierto sobre
  los nuevos **no se puede dar** porque no están anotados. Anotarlos es el siguiente paso obvio y es
  trabajo de ojo, no de CPU.
- **El corte 0,85 sigue siendo el de `frac_a_voto`, reutilizado y no ajustado** para el pie ni para el
  hombro. Nadie lo ha barrido para ellos y con 27 tiradores no se debe.
- **`pie_x` supone que la pista es horizontal en la imagen.** La variante totalmente intrínseca
  (`pie_cadera`) sale **peor** (0,778 contra 0,852), así que apoyarse en la horizontal ayuda y con la
  cámara girada esto se degradaría. No hay ningún clip así en la muestra: está supuesto, no medido.
- **El empate entre el hombro y la mano es un resultado de ESTA muestra.** El argumento de que el
  hombro debería aguantar mejor el brazo cruzado sigue siendo correcto; lo que dice la medida es que
  en estos 27 tiradores el cruce pesa 0,04 de mediana y por tanto **no había nada que ganar ahí**. En
  un corpus con más combate cercano podría pesar más.
- **Las filas `ev_accion` y `ev_brazo` de 25.11 siguen pendientes** de que el agente de la pista cierre
  `eventos.csv`. No se han vuelto a medir a propósito: el fichero ya cambió tres veces debajo y va a
  cambiar una cuarta. Se remiden cuando me avisen.

---

## 27. El cono como CAMPO DE PROBABILIDAD: la idea es correcta y la muestra no llega, con las dos cifras

El usuario lo pidió así: *"el cono 3D puede estar por dentro tipo como con puntos del 0 al 1: cuanto
más cerca de 1, más probabilidad de ser florete dentro del volumen del cono. Tipo, si hemos trackeado
en algún píxel y visto que es florete, lo marcamos como 1, si no menos, y así como de la probabilidad
que hemos visto por los vídeos. Igual ayuda, porque igual en muchos se ve el florete claramente"*.

**Y ataca el punto más débil de todo lo que hay**, que además ya estaba confesado en el código: el
relleno del cono es `peso_sector = 1 - ángulo/apertura`, y el docstring de `cono_lib` dice literalmente
*«es una FORMA ELEGIDA, no una probabilidad medida: nadie ha medido dónde está la hoja dentro del
cono»*. Esto la mide. Código en `e23_campo_de_probabilidad.py` (la distribución) y
`e24_campo_dato_y_figura.py` (el dato y las figuras). Todo de los NPZ y de las rectas ya cacheadas:
**cero GPU, 4 fotogramas decodificados.**

### 27.0 El criterio que decide, declarado antes de medir

El campo empírico sustituye al sector lineal **si y sólo si necesita menos área para contener el mismo
90 % de las hojas observadas**. Con el mismo radio, el área de un sector es proporcional a su anchura
angular, así que la razón de áreas es la razón de anchuras:

| | |
|---|---|
| `W_geo` | 2 × p90(\|desvío\|), el sector **simétrico** que contiene el 90 % |
| `W_emp` | la ventana de anchura **mínima** que contiene el 90 % de los desvíos **con signo** |
| **se adopta si** | `W_emp / W_geo ≤ 0,90` (margen del 10 %, declarado) |

Se mide con el eje ya arreglado (`codo -> raíz`) y con el desvío **con signo**, no en valor absoluto:
si la hoja se desviara siempre al mismo lado eso es un **sesgo**, y promediarlo a cero lo borraría.
Para que el signo signifique lo mismo en los dos tiradores se **espeja** al que mira a la izquierda,
porque espejar la imagen invierte el sentido de giro; en el marco común, positivo es **hacia el
suelo**.

### 27.1 El desvío con signo, y por qué el sesgo que parece haber NO existe

| muestra | n | p5 | p25 | **p50** | p75 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| **todas** | 31 | −30,8 | −20,9 | **−4,4** | +8,8 | +30,2 |
| `AdorableFittingAsiansmallclawedotter` | 24 | −25,6 | −21,6 | **−4,4** | 0,0 | +19,2 |
| `ImpeccablePertinentBasilisk` | 3 | — | +18,5 | **+24,3** | +25,8 | — |
| **todas menos la dominante** | 7 | −54,1 | −17,4 | **+12,8** | +25,8 | +31,3 |

El IC 95 % del p50 por remuestreo es **[−13,8, −0,2]**, o sea que **no contiene el cero** y en la tabla
agregada «hay sesgo». **Y es un artefacto.** Al quitar el asalto dominante el signo **se invierte**:
−4,4° con él, **+12,8° sin él**, y el asalto que más se separa da +24,3°. Una asimetría cuyo **signo
cambia** al quitar un asalto no es una propiedad de la esgrima, es la de ese asalto — y 24 de las 31
detecciones son del mismo tirador. **No hay sesgo establecido**, y el IC global que decía lo contrario
es la razón por la que el desglose por asalto tenía que hacerse antes de agregar.

### 27.2 El criterio decide: pasa por un 1 %, y el remuestreo dice que no está establecido

| | |
|---|---:|
| sector simétrico de hoy | ±33,0° → **66,0°** de anchura |
| ventana empírica más estrecha | [−25,8, +33,0] → **58,8°** |
| **razón de áreas** | **0,891** |
| límite declarado | ≤ 0,90 |
| **IC 95 % de la razón, por remuestreo** | **[0,698, 0,998]** |

Formalmente **pasa la puerta**, por 0,009. Y el IC **contiene el límite de 0,90**, o sea que la
ganancia del 10,9 % de área no se distingue de cero con esta n. Súmese que la asimetría de la que
vendría esa ganancia **invierte el signo entre asaltos** (27.1), y el veredicto honesto es:

> **El campo empírico ASIMÉTRICO no se adopta.** Pasa su puerta por un margen de 1 % que el
> remuestreo no sostiene, y el rasgo que le daría la ventaja no es reproducible entre asaltos.

La envolvente operativa sigue siendo **simétrica, ±33,0°**.

### 27.3 La FORMA sí es distinta de la lineal, y tampoco está resuelta

Esto es el corazón de la idea del usuario: ¿decae la probabilidad como `1 - ángulo/apertura`?

| \|desvío\| (°) | 3 | 9 | 15 | 21 | 27 | 33 |
|---|---:|---:|---:|---:|---:|---:|
| hojas medidas (n) | **9** | 4 | 4 | 6 | 4 | 2 |
| densidad / máx | **1,00** | 0,44 | 0,44 | 0,67 | 0,44 | 0,22 |
| `1 - ángulo/33` (lo de hoy) | 0,91 | 0,73 | 0,55 | 0,36 | 0,18 | 0,00 |

**Error medio absoluto entre las dos formas: 0,21 sobre una escala de 0 a 1.** No son la misma forma:
la medida tiene un **pico pegado al eje** y luego es casi **plana**, mientras la de hoy es una rampa.
Si esto se confirmara, diría algo con contenido: **la hoja no está preferentemente cerca del eje
dentro del cono**, y una rampa lineal está regalando peso al centro y quitándoselo al borde.

**Pero con n=31 los tramos llevan de 2 a 9 cuentas**, o sea un ruido de Poisson de ±1,4 a ±3, y **el
único rasgo que sobrevive es el pico del primer tramo** (9 contra una media de 4 en los otros cinco).
La planicie de los tramos 2 a 6 es indistinguible de una rampa dentro del ruido. Así que la forma
medida **se guarda como dato y se pinta, y no es operativa**: la decisión de amenaza sigue con
`peso_sector` hasta tener la n que hace falta (27.6).

### 27.4 «Si lo hemos visto, lo marcamos como 1»: el 1 medido NO es un punto

> **RETRACTADO el 2026-09-11 en 29.1. Este apartado está mal y su conclusión es la contraria.** Los
> 14,0° de abajo salen de **8 pares de un solo asalto**; con **70 pares de 12 asaltos** el temblor de
> una hoja vista es **4,2°**, o sea **7,9× más estrecho** que la dirección supuesta y no 2,36×. **Ver
> la hoja SÍ estrecha el campo**, y el planteamiento del usuario era más correcto que mi medida. Se
> deja el texto porque la cifra vieja está citada en `e23` y en el NPZ regenerado, pero **no se debe
> usar**.

El requisito del encargo es que un 1 medido y un 1 supuesto no se confundan, y para eso ya existe la
columna `origen`. Lo que no estaba medido es **cuánto más estrecho es el 1 medido**. Se mide con los
pares de cuadros consecutivos donde la hoja se ve en los dos:

| | |
|---|---:|
| pares consecutivos con hoja vista en ambos | **8** |
| \|cambio del ángulo de la hoja\| p50 / p68 / máx | 10,1° / 19,8° / 22,4° |
| **dispersión de un cuadro con hoja VISTA** | **14,0°** |
| dispersión de la dirección SUPUESTA (envolvente) | 33,0° |
| **razón** | **2,36×** |

Es decir: **ver la hoja estrecha el campo sólo 2,36×, y 14,0° es prácticamente la mediana del desvío
de la dirección supuesta (14,4°).** La expectativa del encargo —que un cuadro con hoja vista tuviera
«una distribución casi puntual»— **no se cumple**. Y tiene una explicación que ya es conocida en este
proyecto: el trozo de hoja detectado mide 0,28 m (27.5), o sea poco brazo de palanca, que es el mismo
mecanismo que tumbó el eje del pulgar.

**Lo que sí aporta ver la hoja es el CENTRO del campo, no su estrechez.** El campo se centra en el
ángulo medido en vez de en el del antebrazo, y eso es información real por cuadro. Va escrito así en
la leyenda del NPZ para que nadie lea la columna al revés.

### 27.5 El eje radial NO se puede estimar, y hay que decirlo antes de dar el número

| | |
|---|---:|
| hasta dónde llega el trozo detectado (`t_max`) | p50 **0,29 m**, p90 0,77, máx 0,87 |
| dónde empieza (`t_min`) | p50 0,06 m |
| **largo del TROZO** | p50 **0,28 m**, p90 0,70 |
| eso es el … del alcance de 1,10 m | **25 %** |

`HoughLinesP` devuelve **fragmentos**, no la hoja entera. Luego `t_max` es una **cota inferior** de
dónde está la punta y **esta distribución no puede usarse para acortar L**. Sólo sirve para confirmar
que los trozos caen delante de la mano (79 % con `t_max > 0`). **La forma radial del campo se declara
uniforme y queda sin medir**, y eso va en la leyenda del NPZ.

### 27.6 Tres denominadores distintos, y cuántas hojas harían falta

Confundirlos daría tres cifras contradictorias para «cuánto se ve el florete», que es el error que
este proyecto ya ha pagado:

| denominador | cifra | qué es |
|---|---:|---|
| hoja vista / **conos del dato** | **6 de 3.706 = 0,2 %** | lo que hay hoy en el NPZ |
| hoja vista / **recortes intentados** | **31 de 311 = 10,0 %** | la tasa del detector |
| hoja vista / conos de la **ventana de acción** | **3,0 %** (6 de 198, `e10`) | donde más falta hace |

El 0,2 % **no es la tasa del detector**: su denominador incluye los conos donde la detección **no se
intentó nunca** (sólo se pasó `matting` en 311 recortes de 3.706 conos). La cifra honrada para «cuánto
se ve el florete» es el 10,0 %; la del NPZ es «cuántos conos llevan dirección medida», y es la que
limita el campo hoy.

**Y la respuesta útil, que es cuántas harían falta:** el IC 95 % del p90 del \|desvío\| mide **13,5°**
de ancho con n=31, y el objetivo declarado es el margen del proyecto, **5°**. El ancho de un IC de
percentil baja como 1/√n, luego hacen falta

> **n ≈ 226 hojas vistas (×7,3 de las que hay)**, que a la tasa ya medida del **42,6 %** por recorte de
> 1080p **fuera** de la ventana de acción (`e5`, 26 de 61) son **~531 recortes**, o sea del orden de
> **9 clips** a dos tiradores por 30 cuadros, de los ~1.540 clips de 1080p del corpus.

Es alcanzable, es `matting` en GPU sin anotar nada a mano, y **no está hecho**. Ésa es la tarea que
convierte la idea del usuario en algo operativo.

### 27.7 Y lo que hay que contestarle al usuario sobre «en muchos se ve claramente»

Dos cosas, y la segunda es una corrección de lo que yo mismo iba a decirle.

**Primera: en muchos clips se ve, pero en pocos CUADROS.** 10,0 % de los recortes intentados, y **3,0 %
en la ventana de acción**, que es justo donde la pregunta del cono está viva.

**Segunda: NO he podido confirmar que se vea donde hay menos movimiento.** Lo medí comparando la
velocidad de la mano en los recortes con detección contra los sin detección, del mismo clip:

| | n | velocidad p50 (altos de caja por cuadro) |
|---|---:|---:|
| CON hoja detectada | 36 | **0,0260** |
| SIN hoja detectada | 254 | **0,0250** |
| **razón SIN/CON** | | **0,96×** |

**No hay diferencia.** Así que la explicación por desenfoque de movimiento que 22.11 dejaba como
candidata **no queda apoyada por esta medida**, y no se la puede vender al usuario como si lo
estuviera. La salvedad de mi propia medida va con ella: **la velocidad de la MANO es un mal
sustituto de la velocidad de la PUNTA**, que es la que borra la hoja, y esa no se puede medir sin ver
la hoja — que es el problema de partida. El mecanismo sigue **sin identificar**, igual que en 22.11.

### 27.8 El dato y las figuras

**El dato**, que es lo que le sirve a un modelo y no una imagen: `cono_campo_<clip>.npz`, 7 clips, con
la leyenda **dentro del propio fichero** al lado de las cuatro procedencias que ya existían:

| columna | qué es |
|---|---|
| `campo_mu_gr` (T,2) | dirección central del campo, en grados de imagen. **El ángulo MEDIDO de la hoja si se vio; el del antebrazo si no** |
| `campo_sigma_gr` (T,2) | 14,0° si `origen = 0`, 33,0° si `origen = 1`. Las dos **medidas**, no elegidas |
| `campo_origen` (T,2) | **0 hoja VISTA · 1 dirección SUPUESTA**. Nunca se fusiona con `mu` |
| `campo_L_px` (T,2) | alcance, 1,10 m a escala |
| `perfil_bordes` / `perfil_valor` | la forma medida de 27.3, marcada como **no operativa** |

La hoja vista se **orienta hacia delante** sobre el eje antes de tomar su ángulo: el segmento detectado
no tiene sentido propio y un giro de 180° invertiría el campo.

**Las figuras**, en `02_validacion/muestras/cono_arreglado/`, 4 imágenes `campo_<clip>_c####.png`, cada
una con dos paneles del **mismo cuadro**, mismo eje, mismo L y misma envolvente, para que **lo único
que cambie sea la forma del relleno**: a la izquierda `1 - ángulo/apertura`, a la derecha la forma
medida. Y en las cuatro la hoja está **vista de verdad**, con el campo centrado en ella y el rótulo
diciéndolo. El pie avisa de que la forma medida no es la operativa y por qué — un rótulo que la
vendiera como establecida sería peor que ninguno.

### 27.9 Una nota de coherencia con 26.8, que por fuera parece contradecirla

La sección **26.8** adopta **`hombro -> raíz`** para `posicion._brazos`, y **24.3** cambió el eje del
cono a **`codo -> raíz`**. **No se contradicen, y el motivo es la regla general de 24.3:** son dos
preguntas distintas y cada eje contesta una.

| sección | qué mide | eje que gana | con qué cifra |
|---|---|---|---|
| **26.8** | **CUÁNTO** se extiende el brazo (evento de extensión) | `hombro -> raíz` | cv del largo 0,298 contra 0,223, gana en 5 de 8 |
| **24.3** | **HACIA DÓNDE** apunta la hoja (eje del cono) | `codo -> raíz` | \|hoja − eje\| 14,4° contra 44,8°, y 44,8° es el azar |

Un eje bueno para una pregunta puede ser el peor para la otra, y aquí lo es: el hombro acierta la
magnitud y da **el azar** en la dirección. Así que las dos adopciones son correctas **en su sitio**, y
lo que sería un error es propagar cualquiera de las dos a la otra pregunta.

### LO QUE ESTA MEDIDA NO DICE

- **No dice que la idea del usuario esté mal. Dice que todavía no se puede estimar.** Las dos piezas
  que sí quedan probadas son que la forma de hoy **no es la forma medida** (error 0,21 de 0 a 1) y que
  centrar el campo en la hoja vista es información real por cuadro. Lo que falta es n, y está
  cuantificado: 226 hojas, ~9 clips de 1080p.
- **El campo asimétrico se retira con su medida**: razón de áreas 0,891 con IC [0,698, 0,998] que
  contiene el límite de 0,90, y signo de la asimetría invertido entre asaltos.
- **La forma medida descansa en 31 hojas y 24 son de un asalto** que además no está en
  `corpus_etiquetado.txt`. Los seis tramos llevan de 2 a 9 cuentas. Cualquier lectura de la forma más
  allá de «hay un pico junto al eje» está leyendo ruido.
- **El 14,0° de la hoja vista sale de 8 pares consecutivos**, todos del asalto dominante. Es la cifra
  más flaca de esta sección y es la que gobierna la distinción entre `origen = 0` y `origen = 1`.
- **La forma radial no está medida en absoluto** y se declara uniforme. Los trozos detectados son el
  25 % del alcance, así que ni siquiera se puede acotar por arriba dónde está la punta.
- **No hay ninguna validación contra el tocado ni contra el contacto de hojas.** El campo se estima de
  las hojas vistas y se juzga por área y por reproducibilidad, nunca contra la etiqueta, que sería
  circular. Sigue sin haber verdad-terreno del contacto (0 positivos decidibles en 15 láminas, 22.20).
- **El campo no arregla nada de lo que 24.9 dejó abierto**: el cono sigue conteniendo el torso del
  rival en ~la mitad de los cuadros en alcance, y eso depende del vértice, que es `mano_armada` y lo
  está reescribiendo otro agente. Cuando cambie, **27.1, 27.3 y 25.4 hay que volver a medirlos**,
  porque todos se apoyan en el eje que sale de ese vértice.
- **`peso_sector` sigue siendo el operativo y no se ha tocado.** El campo nuevo vive en
  `cono3d_lib.campo_hoja`, se guarda como dato y se pinta; no decide ninguna amenaza. Si algún día
  sustituye a `peso_sector`, eso moverá cifras publicadas y habrá que declararlo entonces.

### 25.19 · La verdad ampliada a 38 tiradores en tanda ciega, y la puerta se pasa por primera vez

Se anotaron los **6 clips nuevos** del corpus (12 tiradores) con el método reforzado: mosaicos
`ciego_NN.png` **sin nombre de clip ni de lado**, orden barajado con `random.Random(20260911)`, clave
en `clave_ciego.csv` **no mirada hasta después de anotar**, y segunda pasada de 6 cuadros sobre los 10
que quedaron en `certeza=media`. **Ninguna letra cambió entre las dos pasadas** y 6 subieron de media
a alta. Un tirador pasó a `?` en una tercera mirada (abajo). La verdad pasa de 27 a **38 tiradores con
verdad y medida**.

| criterio | 27 tiradores | **38 tiradores** |
|---|---:|---:|
| (a), la línea base | 0,741 | 0,763 |
| `a_voto` | 0,815 | 0,842 |
| `pie_x` | 0,852 | **0,895** |
| `nariz_dir` | 0,889 | **0,895** |
| `hombro_pie` | 0,852 | **0,895** |
| `hombro_nariz` | 0,889 | **0,895** |
| `hombro_conf` | 0,741 | 0,763 |
| `guante` | 0,444 | 0,447 |
| `cuerpo_2de2` | 0,885 (26) | 0,917 (36) |
| **`mayoria_3`** | 0,889 | **0,921** |
| **pie firme (ADOPTADO)** | 13 de 13 | **17 de 17 sin fallo** |
| `pie_o_hombro` | 15 de 15 | 22 de 23 = 0,957 |

**1. La puerta declarada se pasa por primera vez.** Era «acierto ≥ 0,90 sobre los tiradores que decide
y ≤ 4 sin decisión», declarada antes de medir la primera puerta y **nunca movida**. La pasa
**`mayoria_3`** —la mayoría simple entre `a_voto`, `pie_x` y `nariz_dir`, declarada antes de medirse—
con **0,921 sobre 38 y cobertura total**. `cuerpo_2de2` también la pasa (0,917 dejando 2 sin decidir).

**2. Y encaja con lo adoptado en vez de competir con él.** Los tres fallos de `mayoria_3` son **los
tres tiradores irreducibles**, y **los tres ya salen marcados `dudosa`**. En los 17 firmes `mayoria_3`
acierta 1,000; en los 21 dudosos acierta **0,857** contra **0,667** del criterio (a). De ahí el
interruptor `cono_lib.FALLBACK_MAYORIA`, que **sólo cambia lo que se devuelve en la parte que la
tubería debe excluir**:

| configuración | firmes | acierto firmes | todos (38) | acierto todos |
|---|---:|---:|---:|---:|
| adoptado (hoy) | 17 | **1,000** | 38 | 0,816 |
| `+ USA_HOMBRO_SI_PIE_FLOJO` | 23 | 0,957 | 38 | 0,842 |
| **`+ FALLBACK_MAYORIA`** | 17 | **1,000** | 38 | **0,921** |

La tercera fila es estrictamente la mejor de las tres: **no toca el subconjunto firme** y sube el
acierto global de 0,816 a 0,921. Está implementada y **apagada**, esperando aprobación.

**3. Lo adoptado aguanta y su enunciado mejora solo**, que es lo que se pedía de la muestra grande: de
«no ha fallado ninguno de 13» a **«no ha fallado ninguno de 17»**, con la cobertura en 17 de 38
(0,45). Con 17 observaciones el intervalo de confianza inferior sube a ~0,81, todavía ancho.

**4. La ampliación con el hombro ha dejado de ser gratis.** Con 27 tiradores daba 15 de 15; con 38 da
**22 de 23 = 0,957**. El fallo es `HairyCourteousEarwig l0`, donde el pie dice B con `frac` 0,69 (no
firme) y el hombro dice A con 0,90 (firme) y la verdad es B. Es exactamente para lo que sirve más
muestra, y por eso no se encendió antes de tenerla.

### 25.20 · Dos modos de fallo nuevos, y ninguna causa nueva entre los irreducibles

Lo más valioso de la tanda. Los nuevos traen **dos modos de fallo que no estaban** en los 27:

- **El tirador SIN CARETA, sosteniéndola en una mano** (`AdorableVeneratedDuiker l0`). Entre asaltos
  se quita la careta y la lleva en la mano; el criterio (a) elige **la mano de la careta** y falla con
  39 px de margen. Y tiene un remedio reglamentario: por **t.115.1 la careta va en la mano NO
  armada**, así que la careta es **evidencia de la mano contraria**. Se anotó con un valor de `cue`
  nuevo, `careta`, y la careta es un objeto grande y oscuro, mucho más visible que el guante que no
  separaba. **El pie y el hombro lo aciertan los dos.**
- **Un tercer «seguro y equivocado»** (`HairyCourteousEarwig l1`, 111 px de margen, `cue=arma alta`),
  que también aciertan el pie y el hombro.

**Y lo que NO ha aparecido:** entre los **tres irreducibles** —los que fallan con todos los
criterios— no hay ninguna causa nueva. Siguen siendo los mismos tres de antes
(`WhiteMarriedGuineafowl l0` y `l1`, `PowerfulAllLarva l1`) y siguen siendo todos **«no en guardia»**.
La conclusión de 25.11 aguanta con 38 tiradores.

**El caso que pasó a `?`, y por qué se pone `?` y no lo que dicen los criterios.**
`IlliterateDisloyalFulmar l0`: los **cuatro** criterios dicen B y mis dos pasadas ciegas dijeron A con
el `cue` más débil. Al mirarlo por tercera vez con lupa resulta que **en los cuadros de marcha la mano
del lado del rival es A y en los de guardia la mano a la altura del pecho es B**, y no se ve el arma
en ninguno de los 6. Antes de rendirme se comprobó que **no es el defecto 21.1**: la raíz de mano cae
dentro de la caja del rival en **menos del 0,03** de los cuadros (p50 0,000 y p90 0,037 sobre los 46
tiradores), o sea que la pose no está rota, es la postura la que cambia. Se pone `?` **a propósito**:
corregir la verdad hacia lo que dice el algoritmo, en el único caso donde los cuatro coinciden contra
mí, sería fabricar el acierto.

### Lo que 25.19-25.20 NO dicen

- **`mayoria_3` pasa la puerta, pero no está adoptado.** La adopción la decide el usuario; lo que hay
  es el interruptor, la cifra y el hecho de que no toca el subconjunto firme. Y pasar la puerta con
  0,921 sobre 38 sigue siendo **3 fallos**, no cero.
- **17 casos sin fallo siguen siendo pocos.** El intervalo inferior anda por 0,81. La frase correcta
  es «no ha fallado ninguno de 17».
- **El subconjunto `certeza=alta` da 0,929 para `pie_x`, `hombro_pie`, `hombro_nariz` y `mayoria_3`**,
  y es la mejor estimación libre de error de anotación, pero **no es un subconjunto aleatorio**: es
  donde se ve el arma, que correlaciona con que el tirador esté en postura clara. No se traslada al
  corpus.
- **La partición por resolución sigue sin poder hacerse**: 34 tiradores a 720p y **4 a 1080p** (un
  clip nuevo, `SeparateHoarseFeline`, es 1080p). Con 4 casos cualquier cifra a 1080p es anecdótica, y
  la de 720p (`pie_x` 0,912, `mayoria_3` 0,941) es la que vale.
- **`WastefulYawningAmericanquarterhorse` es el clip más corto** (300 cuadros, **48 utilizables** con
  el filtro de dos tiradores medidos y cuatro manos con confianza) y aun así sale con `frac_pie` de
  0,90 y 1,00 y los dos lados firmes y correctos. Eso **no** demuestra que la firmeza aguante con
  pocos cuadros: con 48 cuadros una `frac` alta es más fácil de obtener por azar que con 250, y aquí
  ha coincidido con el acierto en dos casos. Habría que medirlo aparte con más clips cortos.
- **La tanda ciega elimina el sesgo de recordar el clip, no el de ser yo el anotador.** Sigue siendo
  un agente mirando recortes, no un árbitro, y las 10 filas con `cue=accion` se decidieron por «cuál
  va delante», que es la lectura más parecida a los criterios que se juzgan.
- **`ev_accion` y `ev_brazo` siguen sin remedirse**, esperando aviso. Y ahora además el agente de la
  pista va a renombrar su evento `parada` a `alto` y a corregir un eje invertido, así que al remedir
  pueden cambiar **los nombres de columna** y no sólo los valores.

## 28. «Muy pocas marchas»: el recall medido por primera vez, y cuatro cosas que el usuario vio y yo no

El usuario miró las cifras y dijo: *«muy pocas marchas. Igual por clip habrá varias, unas 10 marchas
por tirador, varios fondos y eso, así que lo está detectando mal, revísalo»*. Después vio los vídeos
del trazado, le gustaron (*«tiene buena pinta, me gusta tener toda esta info»*) y encontró tres cosas
más: el nombre `parada`, el eje del mapa y la transversal pintada plana.

**La sospecha era razonable y la respuesta es que el detector no perdía la mitad: lo que había eran
tres cosas de contabilidad y una de nombre.** Todo lo de esta sección se mide sobre **11 clips con
pose de 308 puntos**, que es lo que hay desde que se calcularon las seis que faltaban.

### 28.1 La cifra de «5,7 marchas por tirador» contaba sólo `marcha` y excluía `romper`

Un paso hacia atrás también es un paso. Contando `marcha + romper`, que es lo que en castellano
llano son «pasos»:

| | valor |
|---|---:|
| pasos totales en los 11 clips | **201** |
| **por tirador y clip** | **9,1** |
| por segundo útil y tirador | 0,98 |
| extrapolado a un asalto de 20 s | **19,6 por tirador** |

O sea que la expectativa del usuario (**~10 por tirador**) y lo que el detector entrega (**9,1**)
coinciden. La discrepancia era de unidad, no de detección.

Y había una segunda dilución: **6 de los 11 clips daban 0 marchas, 0 fondos y 0 romper** porque no
tenían `poses_*.npz`, y los eventos de pies EXIGEN la pose de 308 puntos. Los 284 eventos de entonces
se repartían como si fueran 11 clips cuando sólo 5 eran medibles. Ya están los once, y el CSV lleva
columna **`tiene_pose_308`** para que una tasa no se pueda promediar con ceros estructurales. Estado
actual: **11 de 11 con pose, 0 clips con cero eventos**.

Tercera cosa de contabilidad, y es un fallo mío: usé la columna `n` de `posicion_y_eventos.csv` como
duración, y `n` **no es una duración**: es el número de saltos sobre LOS DOS tiradores, o sea el doble
de los cuadros. La tasa salía la mitad (0,49 pasos por segundo y tirador en vez de 0,98). Ahora hay
columna **`s_utiles`**, calculada con el tramo de cuadros, y **`pasos_por_s_util_y_tirador`**, que es
la comparación buena entre clips porque `WastefulYawningAmericanquarterhorse` sólo tiene 4,0 s útiles
y su cuenta baja es de duración, no de fallo.

### 28.2 EL RECALL, medido por primera vez: 90 % de las zancadas

Lo que estaba medido antes era **precisión** («de 3 eventos marcados, 3 correctos») y una cota de
falsos negativos de «6 a 50 pasos por clip», un rango tan ancho que admitía que el usuario tuviera
razón. **Nadie había contado nunca cuántos pasos hay de verdad en un clip.**

El convenio de qué cuenta como paso se escribió ANTES de contar, en
`02_validacion/pista/CONVENIO_CONTEO_PASOS.md`, porque ahí estaba media discusión. Resumen: la unidad
que se cuenta es la **ZANCADA** (un apoyo que se traslada al menos el largo de su propio pie, unos
25 cm), porque es observable sin ambigüedad; y una **MARCHA** es una pareja de zancadas consecutivas
del mismo sentido y de pies distintos, con la que se queda sin pareja contada aparte como medio paso.

La verdad-terreno se construye en **espacio de imagen**, con `verifica_pista.mosaico_pasos` y
`cuenta_ojo_pasos.py`: fotogramas ESTABILIZADOS por la similitud `Mk` (el suelo se queda quieto, así
que un pie que se traslada se ve contra un fondo fijo), umbral de zancada en **0,14 del alto de la
caja** del propio tirador, que es el largo del pie de una persona. **No hay ningún umbral de
velocidad, que es justo lo que se sospechaba del detector.** Y se confirmó en el mosaico que las
cruces de los pies caen sobre los zapatos en prácticamente todas las casillas.

| tirador | s útiles | zancadas contadas | cubiertas por una detección | recall |
|---|---:|---:|---:|---:|
| `UglyNiceAplomadofalcon` lado 0 | 9,0 | 11 | 11 | **100 %** |
| `GentleQuerulousElephantseal` lado 0 | 10,3 | 15 | 12 | **80 %** |
| `AliveIgnorantBarebirdbat` lado 1 | 13,5 | 23 | 21 | **91 %** |
| **agregado** | 32,8 | **49** | **44** | **90 %** |

**El detector no pierde la mitad: cubre el 90 % de los traslados de pie.** Y la cuenta a ojo da
**1,22 a 1,70 zancadas por segundo**, o sea 24-34 por tirador en un asalto de 20 s, que agrupadas en
parejas son 12-17 marchas. La cifra del detector (0,98 pasos por segundo) queda entre las parejas
contadas (0,49-0,74/s) y las zancadas (1,22-1,70/s), más cerca de las parejas, que es lo que su
convenio de emparejamiento dice que hace.

**Las 5 zancadas perdidas dicen algo:** una de ellas es `AliveIgnorantBarebirdbat` pie1 c383-414,
**61 px en 1,03 s**, o sea un traslado LENTO. Ése es exactamente el fallo que cabía esperar de un
umbral de velocidad, y confirma que el 10 % que se pierde son los pasos de preparación lentos.

**Y encontré un fallo mío contando:** la primera versión del segmentador de la verdad-terreno usaba
tramos MONÓTONOS maximales, que es **el mismo error que yo había diagnosticado en la v3 del detector**
(§26.4): la posición de un pie es una ESCALERA y una escalera es monótona completa. Daba **0 zancadas
en `UglyNice`** y una «zancada» de −406 px en 2,03 s en `Alive`. Haber escrito el diagnóstico no
impidió repetirlo; lo cazó mirar la salida. La versión que vale segmenta por **MESETAS** (rachas de
cuadros en que el pie no se mueve más que 3 veces el ruido medido), que es lo único que segmenta una
escalera.

### 28.3 EL RETARDO: mediana +0,5 cuadros. No hay nada que corregir

El usuario quería ver *«si hay delay»*. Medido contra las zancadas contadas a ojo, sobre 62
emparejamientos:

| | valor |
|---|---:|
| desfase mediano | **+0,5 cuadros = +17 ms** |
| p10 / p90 | −14 / +13 cuadros |
| `|desfase| <= 3` cuadros | 32 % |

**No hay retardo sistemático que restar.** La dispersión no es retardo: sale de que una detección
suele cubrir 2-3 zancadas, así que llega «tarde» a la primera y «temprano» a la última. Lo que es
grueso es el AGRUPAMIENTO, no el instante.

### 28.4 `parada` estaba MAL LLAMADO y ahora es `alto`

El usuario: *«`parada` no sé si se refiere a parada-respuesta o que se para»*. Tenía razón y era un
choque de nombres serio:

 - en esgrima una **parada** es la acción defensiva de desviar la hoja del atacante con la propia, y
   es la bisagra del reglamento porque transfiere la prioridad a la **respuesta**. Es un evento de
   **arma**.
 - lo que este detector emite es que el tirador **deja de desplazarse**. Es un evento de **pies**, y
   su propio criterio lo dice: «baja de 0,3-0,7 m/s», no «inmóvil».

Renombrado a **`alto`** en el CSV, en el rótulo del vídeo, en el contador, en el `LEEME.md` y en los
recortes de verificación. **La parada en el sentido de esgrima NO está implementada y no se puede con
este material**: el paso de prioridad se cerró con 0 paradas anotadas y midió que el contacto de
hojas no es verificable a ojo en este corpus. Quien lea «71 altos» no debe entender que hay
paradas-respuesta detectadas, porque no hay ninguna.

La lista de tipos que sí hay queda: **`marcha`, `romper`, `fondo`, `alto`, `brazo`, `accion`**, más
`bote` declarado no medible con 1 evento en 11 clips.

### 28.5 El eje del mapa: NO está invertido, y ahora está declarado y rotulado

El usuario: *«está invertido en el eje, el gráfico de la pista de abajo, porque el verde lo pones que
está más pegado a la derecha y está a la izquierda»*. **Medido, no lo está**, y se comprobó por tres
vías antes de tocar nada, que es lo contrario de girar el dibujo hasta que cuadre:

| comprobación | resultado |
|---|---|
| `lado 0` a la izquierda EN LA IMAGEN (centro de caja en píxeles) | **11 de 11 clips** |
| `lado 0` con la x MÉTRICA menor | **11 de 11 clips** |
| píxeles verdes a la izquierda de los naranjas en el MAPA del MP4 | **5 de 5 vídeos** que vio |
| **CRUCES** (`x_lado0 > x_lado1`), que son físicamente imposibles | **0 de 2.896 cuadros**, en metros y en píxeles |

El de los cruces es un control físico nuevo y gratis: **dos floretistas no pueden cruzarse**, así que
un solo cuadro con la izquierda a la derecha de la derecha sería o una posición mal medida o la
pareja mal asignada. Sale cero, y va como columna `cruces_imposibles`.

Lo que **sí faltaba** es que el convenio estuviera declarado y que el mapa dijera cuál es cuál: el
signo que sale de la homografía lo decide cuál de los dos bordes salió como A en el barrido, que es
un detalle interno. Ahora `posicion.apoyos` lo fija con un **aserto** («x crece hacia la derecha de la
imagen; `lado 0` = izquierda») y el mapa del vídeo lleva las etiquetas `IZQUIERDA (verde)` /
`DERECHA (naranja)` en sus carriles y una flecha con el sentido del eje.

### 28.6 La transversal: pintarla PLANA era igual de falso que pintar ruido

El usuario: *«en la pista se mueven perfectamente recto, y eso en el video no es verdad: van un poco
moviéndose en el eje y... ahora salen como que no hay cambio en el eje desde que empiezan, y sí suele
haber cambio»*.

Tiene razón en el hecho, y **la causa era una decisión de diseño mía**: como 23.7 decía que la
transversal no es creíble, la dejé fuera y pinté cada tirador en un **carril fijo**, o sea la
transversal puesta a una constante. Comprobado en el código: sí, era una constante
(`CARRIL[lado] * ancho`), deliberada y documentada, no un bug. Pero una línea recta perfecta es una
afirmación tan falsa como el ruido.

**Lo que faltaba era separar dos cosas que se trataban como una**, y el criterio se declaró antes de
medir (`excursion / ruido_por_cuadro >= 5`):

| medida, 22 tiradores-clip | valor |
|---|---:|
| excursión transversal | 0,124 a **2,131 m** |
| ruido p95 por cuadro | 0,009 a 0,071 m |
| **razón excursión / ruido** | **p50 19,8, mínimo 8,7** |
| pasan el criterio de 5x | **22 de 22** |

Así que **la FORMA es señal en todos, sin excepción**, y pintarla plana escondía una señal que está 9
a 77 veces por encima del ruido. El filtro es una mediana móvil de **9 cuadros** (0,30 s), elegida con
la tabla y no a ojo: conserva el **97,8 %** de la excursión (con 5 el 98,9 %, con 45 el 88,6 %), o sea
que no aplana la forma. Se publican **las dos curvas, cruda y filtrada**, en el panel `a2` de
`pista_*.png`, para que se vea que el filtro no se ha inventado la forma.

**Una salvedad de método, porque la otra mitad de mi criterio no valía:** yo había declarado también
«ruido residual por debajo de 0,020 m», y lo medí como el residuo de la serie ya filtrada contra su
propia mediana móvil de 5. Eso es **circular** y da casi cero para cualquier ventana >= 5, así que esa
mitad no discrimina y no se usa.

**Y LO QUE SIGUE MAL ES LA ESCALA.** El control físico declarado es recorrido <= 0,60 m y lo pasan
**11 de 22**. Se probó la hipótesis de que el exceso fuera un error de perspectiva que crece con la
posición longitudinal, quitando de `y` su dependencia lineal de `x`: pasa de **11 a 14 de 22**.
Mejora y no lo arregla. Quedan 8 tiradores con recorridos de hasta **1,87 m en una pista de 1,50 m de
ancho**, que es imposible.

Conclusión, y es la que decide el dibujo: **la FORMA de la transversal se pinta, con su banda de
±3 veces el ruido medido; su ESCALA no es creíble en 8 de 22 y eso va en el rótulo con su cifra.** No
es lo mismo decir «no se sabe cómo se mueve de lado» que decir «se sabe la forma y no el tamaño».

### 28.7 El contador en pantalla y la cámara lenta

*«Me lo puedes enseñar visualmente en un vídeo, a ver si las va contando bien? Pasa varios ejemplos y
que se vea el CONTADOR... estaría bien verlo igual, alguno de esos en cámara lenta, para ver si hay
delay.»*

 - **Contador** por tirador y por tipo (`marcha`, `romper`, `fondo`, `alto`, `brazo`), que **salta en
   el cuadro de `inicio` del evento** y no en el de pico, que es lo que permite juzgar si cuenta bien.
   El número que se ve es **el mismo `indice_del_tipo` del CSV**, tomado como el máximo índice de ese
   (lado, tipo) entre los eventos ya empezados: un contador propio podría desincronizarse del CSV y
   entonces el vídeo no verificaría el CSV, que es para lo que existe.
 - **Destello** con el nombre del evento en grande, su número y su magnitud («MARCHA 3 0,31 m»)
   durante 18 cuadros.
 - **Marcas numeradas en el mapa** para `marcha` y `romper`, las 6 últimas, con su índice al lado.
 - **11 vídeos** a velocidad normal y **4 en cámara lenta** a un cuarto (7,5 fps), eligiendo a
   propósito el más activo (`ForcefulAntiqueAlbacoretuna`, 1,75 pasos/s/tirador) y el menos
   (`UglyNiceAplomadofalcon`, 0,56), que puestos uno al lado del otro dicen más que cinco parecidos.

### 28.8 El exceso de `ForcefulAntiqueAlbacoretuna` es en parte la escala mala

Es el clip que más detecta (25 marchas, 18 fondos) y es justo el que tiene la escala inflada ~2,4x y
pasa la puerta con un solo tirador (§26.1). Test: se reescala su coordenada métrica por el factor que
implica su propio control de separación de pies (`dpies_p50 / 0,70`) y se vuelve a correr el detector.

| clip | factor | pasos antes | pasos después | pasos/s/tirador antes → después |
|---|---:|---:|---:|---|
| `ForcefulAntiqueAlbacoretuna` | **2,14** | 42 | **31** | 1,75 → **1,29** |
| `AliveIgnorantBarebirdbat` | 0,83 | 21 | 24 | 0,78 → 0,89 |
| `GentleQuerulousElephantseal` | 1,34 | 14 | 10 | 0,68 → 0,48 |
| `TestyWatchfulBarebirdbat` | 0,94 | 12 | 13 | 0,80 → 0,87 |
| `UglyNiceAplomadofalcon` | 0,92 | 10 | 10 | 0,56 → 0,56 |

**Un cuarto del exceso es la escala y el resto es un asalto de verdad más activo**: corregida, Forceful
sigue 45 % por encima del siguiente. Y la tasa de los clips buenos NO baja al corregir (media de los
cuatro: 0,71 → 0,70), así que la cifra de 9,1 pasos por tirador no depende de esto.

### 28.9 `solapa_fondo` sólo MARCA, no filtra

Comprobado contando y no leyendo el código: de los 99 pasos del CSV de entonces, 39 tienen
`solapa_fondo = True` y 60 `False`, y **los 99 están en el fichero**. Ninguno se descarta. Quien
quiera pasos limpios filtra `solapa_fondo == False`.

### 28.10 LO QUE ESTA MEDIDA NO DICE

- **El recall es de 3 tiradores, no del corpus.** 49 zancadas contadas en 32,8 s de tres tiradores de
  tres clips. El 90 % es de esos tres, y en uno solo (Gentle, 80 %) ya se ve la variación.
- **La verdad-terreno NO es independiente de los keypoints de la pose.** Es independiente de la
  homografía métrica y del criterio del detector, que era lo que había que aislar, pero si la pose
  perdiera un pie entero, el conteo también lo perdería. Lo que acota eso es que se miró el mosaico y
  las cruces caen sobre los zapatos, y que la procedencia da 267 de 270 poses MEDIDAS en `UglyNice` y
  271 de 291 en `Gentle`.
- **No dice que 9,1 pasos por tirador sea correcto, dice que coincide con lo que espera un
  floretista.** Sin anotación humana de cada paso en cada clip no hay error, y este corpus no la
  tiene: lo que hay es el 90 % de recall en tres tiradores.
- **La escala transversal sigue sin estar establecida.** La forma sí, la escala no en 8 de 22, con
  recorridos imposibles de hasta 1,87 m. Quitar la dependencia de `x` mejora de 11/22 a 14/22 y no lo
  cierra: la causa no es sólo la perspectiva longitudinal.
- **El desfase mediano de +0,5 cuadros no dice que cada evento esté en su cuadro.** Dice que no hay
  sesgo. La dispersión (p10 −14, p90 +13) está dominada por el agrupamiento y no se ha separado del
  error de instante.
- **El eje no está invertido en ESTOS once clips.** Los otros 67 no se han comprobado, y el aserto
  nuevo de `apoyos` abortará si alguno lo incumple, que es lo que se quiere.
- **`bote` sigue no establecido, y ahora con un control físico que lo cierra.** Al pasar de 4 a 11
  clips se colase **1 evento** (`WastefulYawningAmericanquarterhorse` lado 1, c194-195, 8,5 cm,
  0,033 s), y lo encontró el coordinador leyendo el CSV. No había cambiado el criterio: el umbral es
  ADAPTATIVO y en ese clip el ruido es el más bajo de la muestra (0,0143 del alto de caja contra
  0,0198 de mediana), así que 3x cae por debajo del suelo fijo de 0,05 y manda el suelo. Los 10,4 cm
  publicados eran el caso MEDIANO, no el umbral de ese clip; la frase «0 eventos» era un resultado
  observado en 4 clips y con 11 dejó de valer.

  **La incoherencia de verdad era de física y es la que da el arreglo:** si las dos plantas se
  separan `h` del suelo, el cuerpo está en caída libre y el tiempo de vuelo está FIJADO por
  `t = 2 sqrt(2h/g)`. No es un umbral, es una igualdad. Despegue de 5,0 cm → **6,1 cuadros** de vuelo
  a 30 fps; 8,5 cm → **7,9 cuadros**; 15 cm → 10,5 cuadros. El evento que se colé duraba **1 cuadro**
  diciendo 8,5 cm: **8 veces menos vuelo del que la gravedad permite**. Se exige ahora que la duración
  llegue a la mitad del vuelo implícito, y hay un **ASERTO** en `eventos()` que aborta si una fila de
  `bote` lo incumple, el mismo patrón que arregló los agregados. Resultado: **0 botes en los 11
  clips**, ahora por una razón física y no por no haberlo mirado.

### 25.21 · `FALLBACK_MAYORIA` encendido, y el hombro retirado porque más muestra lo tumbó

**Encendido `cono_lib.FALLBACK_MAYORIA = True`.** Ahora, cuando el pie no es firme, lo que se devuelve
no es el criterio (a) sino la **mayoría de `a_voto`, `pie` y `nariz`**, y el tirador sigue marcándose
`dudosa` igual. Estado operativo verificado:

| | firmes | acierto firmes | todos | acierto todos |
|---|---:|---:|---:|---:|
| antes | 17 | **1,000** | 38 | 0,816 |
| **ahora** | 17 | **1,000** | 38 | **0,921** |

Entra por el criterio del proyecto y no por gusto: la puerta **0,90 con ≤ 4 sin decidir** estaba
declarada de antemano y `mayoria_3` la pasa con **0,921 sobre 38 y cobertura total**; **no toca el
subconjunto que el usuario adoptó**, que sigue siendo el pie y sigue en 17 de 17 sin fallo; sus tres
fallos son los tres irreducibles y **los tres ya salían marcados `dudosa`**; y es reversible en una
línea.

**Y `USA_HOMBRO_SI_PIE_FLOJO` se queda apagado porque mi propia medida le quitó el respaldo**, que es
el ejemplo más limpio de este hilo de por qué no se adopta nada con 15 casos:

| muestra | `pie_o_hombro` |
|---|---|
| 27 tiradores | **15 de 15 sin fallo** |
| 38 tiradores | **22 de 23 = 0,957** |

El fallo que aparece al ampliar es `HairyCourteousEarwig l0`: el pie dice B con `frac` 0,69 (no firme)
y el hombro dice A con 0,90 (firme), y la verdad es B. Con 15 casos la regla parecía gratis; con 23 no
lo es. Y `FALLBACK_MAYORIA` cubre el mismo hueco mejor y **sin tocar los firmes**, así que el hombro
no hace falta.

### 25.22 · La careta: la pista era buena, el rasgo no la ve, y se retira con las cifras

La careta tenía el mejor punto de partida de todas las señales de apariencia probadas: **t.115.1 la
pone en la mano NO armada**, así que es evidencia de la contraria; es **grande, compacta y oscura
sobre un tirador de blanco**, o sea que el contraste va en el sentido contrario a la chaqueta que
hundió al guante; y el caso que la destapó (`AdorableVeneratedDuiker l0`) es un fallo real de (a) con
39 px de margen.

Criterio declarado antes de medir: mismo parche del guante, `oscuro = V < 150` —la **inversión de la
mitad en V** de la definición de blanco del proyecto, sin la cláusula de saturación para no echar
fuera una careta plateada o de color—, `a_osc` = área de la mancha oscura conexa mayor que toca el
disco central sobre el área del parche, y `comp` = compacidad de esa mancha.

**Primera medida, el confundido, que era la que decidía: la cazoleta del arma también es oscura y
compacta y está en la mano ARMADA.** Sobre **1.180 cuadros** de los 38 tiradores anotados:

| rasgo | armada p25/p50/p75 | la otra p25/p50/p75 | Δ mediano | signo correcto |
|---|---|---|---:|---:|
| `a_osc` | 0,181 / **0,495** / 0,797 | 0,196 / **0,533** / 0,814 | **−0,011** | **0,470** |
| `comp` | 0,546 / 0,708 / 0,870 | 0,569 / 0,703 / 0,855 | −0,005 | 0,482 |

**No separa nada, y además el diagnóstico está en el propio p50: `a_osc` vale 0,50, o sea que medio
parche sale «oscuro».** Con `V < 150` lo que se segmenta no es un objeto compacto: es todo lo que no
está bien iluminado —la piel, el lamé, el suelo de la pista, la sombra—, y la compacidad de 0,71 en
las dos manos es la de una región amorfa, no la de una careta. Así que el rasgo no ve ni la careta ni
la cazoleta.

**Segunda medida: con el umbral estricto la careta SÍ se ve, pero el detector dispara en todas
partes.** En el único tirador con careta en mano verificada a ojo, `a_osc` con `V < 60` da **0,105 a
0,347** en la mano de la careta contra **0,000 a 0,006** en los cuadros de control del mismo tirador
sin careta en mano. Parece una señal. Pero la condición necesaria para ese disparo, medida en todo el
corpus:

| | |
|---|---:|
| parches que disparan (`f_oscuro ≥ 0,10`, V ≤ 60) | **469 de 3.386 = 0,139** |
| tiradores con al menos un parche que dispara | **35 de 44** |
| tiradores con careta en mano verificados a ojo | **1 de 39 = 0,026** |

O sea que dispara en **35 de 44 tiradores** y sólo **1** tiene careta en mano: es un detector de
oscuridad, no de careta, y su precisión a nivel de tirador es del orden de **1/35**. Los que más
disparan son `PotablePossibleIcterinewarbler l0` (60 parches, `cue=hoja`), `PowerfulAllLarva l0` (que
es el árbitro de traje oscuro) y `WastefulYawningAmericanquarterhorse l1`.

**Retirada.** Con `n = 1` tirador con careta no se puede establecer separación de nada, y lo que sí se
puede medir con `n = 38` —el signo del confundido y la tasa de disparo— dice que no hay señal. Lo que
queda escrito, y no depende del tamaño de muestra, es **por qué**: a 720p, con `V` como único eje, la
careta, la cazoleta, el lamé en sombra y el suelo de la pista **caen en el mismo cubo**. Una careta de
verdad haría falta detectarla por forma y tamaño, no por brillo, y eso es otro trabajo.

### 25.23 · Cuántos cuadros hacen falta para fiarse de `frac`: el corte 0,85 no llega a 0,10 de falsa firmeza a ningún tamaño

El matiz sobre `WastefulYawningAmericanquarterhorse` —48 cuadros utilizables y `frac_pie` de 0,90 y
1,00— ya tiene número. Remuestreo declarado antes de medir: de los **36 tiradores con ≥ 120 cuadros
que votan** (18 firmes y 18 no firmes en el clip completo), 400 submuestras con reposición por tamaño.

| cuadros `n` | `sd(frac)` | **falsa firmeza** | firmeza perdida |
|---:|---:|---:|---:|
| 10 | 0,101 | **0,416** | 0,083 |
| 20 | 0,073 | **0,293** | 0,070 |
| 30 | 0,061 | **0,248** | 0,060 |
| 50 | 0,048 | **0,188** | 0,050 |
| 100 | 0,034 | **0,130** | 0,035 |

**El número declarado era «falsa firmeza ≤ 0,10» y no se alcanza a ningún tamaño probado**, ni con
100 cuadros. «Falsa firmeza» es la probabilidad de declarar firme a un tirador que en el clip completo
no lo es, o sea el error que mete decisiones malas en la parte que la tubería usa. Y como el
remuestreo es con reposición **sobre el mismo clip**, cuyos cuadros están correlacionados en el
tiempo, esa columna es una **cota por lo bajo**.

Consecuencia concreta: **la firmeza de un tirador con 50 cuadros votando tiene del orden de 1 de cada
5 posibilidades de ser espuria**, y `WastefulYawningAmericanquarterhorse` vota con 54 y 98. Así que
ese clip **no demuestra** que la firmeza aguante con pocos cuadros, exactamente como se dijo.

Qué costaría blindarlo, medido, con el mando `cono_lib.N_PIE_MIN` que queda **en 0, apagado**:

| política | decide de 38 | acierto | cobertura |
|---|---:|---:|---:|
| `frac ≥ 0,85` (lo de hoy) | 17 | 1,000 | 0,45 |
| `frac ≥ 0,85` y `n ≥ 120` | 15 | 1,000 | 0,39 |
| `frac ≥ 0,85` y `n ≥ 200` | 5 | 1,000 | 0,13 |
| `frac ≥ 0,90` y `n ≥ 120` | 10 | 1,000 | 0,26 |

Sólo **2 de los 17 firmes** votan con menos de 120 cuadros (el mínimo es 54), así que exigirlo cuesta
2 tiradores y **no gana ninguno**: con esta muestra no hay ningún error que arreglar en el subconjunto
firme. Se deja apagado y documentado para cuando la muestra crezca.

### Lo que 25.21-25.23 NO dicen

- **El 0,921 de lo adoptado incluye a los `dudosa`.** El reparto es 17 tiradores firmes sin ningún
  fallo y 21 dudosos con 0,857; la tubería debe seguir excluyendo los dudosos, y si los usa, lo que
  tiene es 0,921 y no 1,000.
- **`mayoria_3` pasa la puerta con 3 fallos, no con cero.** Y los tres son los irreducibles, que ya
  estaban identificados; no se ha arreglado ninguno.
- **La careta se retira como RASGO DE BRILLO, no como idea.** El respaldo reglamentario (t.115.1)
  sigue en pie y el caso sigue siendo un fallo real de (a). Lo que está medido es que `V` sola no la
  distingue de la cazoleta ni del lamé en sombra, y que el disparo estricto tiene precisión ~1/35.
  Detectarla por forma y tamaño está sin probar.
- **La tasa de disparo de 0,139 se apoya en que sólo hay 1 careta verificada.** Si hubiera más caretas
  sin anotar entre esos 35 tiradores, la precisión sería mejor de lo que digo. No se han buscado una
  por una: son 35 tiradores × 40 cuadros.
- **La falsa firmeza es una cota por lo bajo** y se mide sobre los 36 tiradores largos, que son
  precisamente los que tienen cuadros de sobra. Extrapolar su comportamiento a clips cortos de verdad
  es justo lo que no se puede hacer con este remuestreo.
- **Y no se ha vuelto a tocar `eventos.csv`.** Sigue pendiente el aviso, y ahora con tres arreglos en
  marcha del agente de la pista (renombrado de `parada` a `alto`, eje del mapa invertido y coordenada
  transversal), así que al remedir pueden cambiar nombres de columna y valores a la vez.

---

## 29. Las hojas que faltaban: 31 → 153 y 5 → 12 asaltos, y con eso se caen DOS de mis conclusiones

La sección 27 midió el campo de probabilidad del cono con 31 hojas y dijo que no llegaba. También dio
la vía: **n ≈ 226 hojas**, unos 531 recortes de 1080p, del orden de 9 clips. Esto es esa recogida.
Código en `e25_recoge_hojas.py` (GPU, `matting` + `seg`), `e26_por_que_se_ve.py` y
`e27_agrega_hojas.py` (los dos CPU).

**Y el resultado más importante de la sección es que dos cosas que yo había escrito son falsas.** Van
las dos primero.

### 29.1 RETRACTADO: «ver la hoja no estrecha el cono, sólo lo centra». Sí lo estrecha, y 7,9×

`27.4` midió el temblor de una hoja vista en **14,0°** con **8 pares** de cuadros consecutivos, todos
del mismo asalto, y de ahí concluyó que un `origen = 0` no es casi puntual y que «lo que aporta ver la
hoja es el centro, no la estrechez». **Con 70 pares de 12 asaltos sale 4,2°:**

| | `e23`, 8 pares, 1 asalto | **`e27`, 70 pares, 12 asaltos** |
|---|---:|---:|
| \|cambio\| p50 entre cuadros | 10,1° | **4,0°** |
| \|cambio\| p68 | 19,8° | **6,0°** |
| **σ de una hoja VISTA** | 14,0° | **4,2°** |
| razón contra la supuesta (33,0°) | 2,36× | **7,9×** |

**El planteamiento del usuario era más correcto que mi medida.** Su «si lo hemos visto, lo marcamos
como 1» describe bien lo que pasa: un cuadro con hoja vista es casi puntual comparado con uno
supuesto. Ocho pares no medían el temblor de la hoja, medían el de un asalto. Corregido en
`e24.SIGMA_VISTA` y **regenerados los 11 NPZ del campo**, que antes guardaban una anchura 3,3× mayor
de la real para los cuadros buenos.

### 29.2 El criterio declarado, con la muestra grande: FALLA, y ahora sin ambigüedad

`27.2` daba razón de áreas **0,891** contra un límite de 0,90 — pasaba por 0,009 y el remuestreo no lo
sostenía. Con la muestra grande, **el mismo criterio sin mover el límite**:

| muestra | n | razón de áreas | IC 95 % | veredicto |
|---|---:|---:|---|---|
| `e23`, 31 hojas | 31 | 0,891 | [0,698, 0,998] | pasaba, no establecido |
| **`e27`, todas** | **153** | **0,972** | **[0,782, 0,992]** | **NO pasa** |
| `e27` sin el asalto dominante | 110 | 0,857 | [0,750, 0,986] | no establecido |

**El campo empírico asimétrico queda descartado con cifra.** No haber adoptado el 0,891 fue lo
correcto: con cinco veces más muestra la ganancia de área desaparece.

### 29.3 El sesgo del desvío no existe, y ahora se ve POR QUÉ el IC agrupado engañaba

El p50 con signo agrupado sale **−3,6° con IC [−6,7, −0,2]**, que sigue excluyendo el cero. Y sigue
siendo un artefacto, pero ahora la razón es medible y no una sospecha:

| | grados |
|---|---:|
| ancho del IC 95 % **agrupado** del p50 | **6,5** |
| **desviación típica del p50 ENTRE ASALTOS** (7 asaltos con n ≥ 10) | **8,6** |
| rango de los p50 por asalto (todos) | **−45,1 a +24,3** |

**La dispersión entre asaltos es MAYOR que el intervalo de confianza agrupado.** Eso es la firma de
que las observaciones no son independientes: 153 hojas de 12 asaltos no valen como 153 observaciones
independientes, y el IC agrupado, que supone que sí, sale demasiado estrecho. De los 7 asaltos con
n ≥ 10, **4 desvían hacia arriba y 3 hacia el suelo**. No hay sesgo.

### 29.4 La envolvente de ±33,0° QUEDA CONFIRMADA, y eso es lo que sí se gana

| | n | asaltos | p90 de \|desvío\| | IC 95 % |
|---|---:|---:|---:|---|
| `e20`, la que fijó la constante | 31 | 5 (24 de uno) | 33,0° | [24,3, 37,8] |
| **`e27`** | **153** | **12** | **34,3°** | **[24,3, 40,9]** |

La constante operativa `APERT_ENVOLVENTE_SEMI_GR = 33,0` (semiángulo, 66,0° de cono completo) estaba
puesta con 31 hojas de un asalto y **se sostiene con 153 de 12**. No cambia el número; cambia que
ahora descansa en algo.

### 29.5 «¿Cuántas hojas faltan?» era la pregunta equivocada: la apertura es POR CLIP

El IC del p90 no se estrechó al quintuplicar la muestra: pasó de **13,5° a 16,6°**, porque el primer
IC estaba sub-disperso (un asalto). Aplicando la regla 1/√n saldría n ≈ 1.683, siete veces más de lo
que 27.6 estimaba. **Pero esa cuenta no vale, y la razón es la que importa:**

| | grados |
|---|---:|
| objetivo declarado de precisión | **5,0** |
| **sd del p90 ENTRE ASALTOS** (7 asaltos con n ≥ 10) | **6,3** |
| rango del p90 por asalto | **21,1 a 40,9** |

**La variación entre asaltos ya supera el objetivo, así que ningún número de hojas permite que un p90
GLOBAL llegue a 5°.** No es un problema de n: es que la apertura no es una constante del florete, es
una propiedad del clip. Y eso apunta exactamente a donde `e18` ya había ido con la apertura
adaptativa. **Recoger más hojas para afinar un número global es trabajo perdido**; lo que tendría
sentido es estimar la apertura por clip donde haya hojas y usar la agrupada donde no.

### 29.6 EL MECANISMO: es la ESCALA APARENTE, y queda desconfundida del encuadre

`27.7` dejó el mecanismo sin identificar tras descartar el desenfoque de movimiento (0,96×). Cuatro
candidatos, los cuatro con su criterio declarado antes:

| | candidato | estadístico declarado | medido | veredicto |
|---|---|---|---|---|
| C1 | **contraste local** | razón CON/SIN ≥ 1,15× | **0,98×**, y por quintiles 0,68× | **descartado** |
| C2 | **escorzo del antebrazo** | razón quintil alto/bajo ≥ 1,15× | **1,63×** (13 % → 22 %) | **apoya** |
| C3 | **firmeza de la mano** | razón firme/dudoso ≥ 1,15× | 0,40× | **no se puede contestar**, ver 29.7 |
| C4 | **px/m** | spearman ≥ 0,50 | **+0,79** | **APOYA** |

| clip | px/m (cuadros muestreados) | tasa |
|---|---:|---:|
| `BabyishIdioticCockerspaniel` | 333 | 21,2 % |
| `GentleBlueIndianspinyloach` | 312 | **38,4 %** |
| `uptightSlothEveningBat` | 305 | 11,2 % |
| `proudDungBeetleBoglenTerrier` | 289 | 28,3 % |
| `LiveDecimalLadybird` | 252 | 8,8 % |
| `MessySnappyArmyworm` | 195 | 10,3 % |
| `SeveralSkinnyGenet` | 189 | **5,2 %** |

Y el razonamiento físico cierra: con sección de 4-5 mm el ancho aparente de la hoja es `0,005 × px/m`,
o sea **0,95 px a 189 px/m y 1,67 px a 333**. Es justo el salto de sub-píxel a línea detectable, que
es lo que `HoughLinesP` necesita.

**LA LECCIÓN DE MÉTODO, que vale más que el hallazgo: una variable descartada en un rango estrecho no
está descartada.** `22.11` probó px/m y concluyó que no explicaba nada — pero midió entre **183 y 261
px/m**, una franja donde la variable casi no se mueve. Con **189 a 333** sí se mueve y sí explica.

**Y el confundido que había que descartar, descartado con medida:** el primer plano da más px/m por
construcción, así que px/m podía estar midiendo el encuadre. Medido: de los 633 recortes,
**el 0,0 % está por encima de 0,75 del alto del cuadro**. Y no es suerte, es estructural: las ráfagas
exigen **los dos tiradores presentes**, y en un primer plano de uno el otro no cabe. Así que el rango
de 189 a 333 px/m es variación de **escala de cámara a plano constante**, y la atribución a px/m es
limpia. De paso, eso deja sin efecto el aviso de sesgo de selección por primer plano: **no hay ni una
hoja de ese régimen en la muestra**, y el estrato «primer plano» sale con n=0.

**Lo que esto permite, y es gratis:** px/m sale del alto de la caja, sin GPU. Se puede decidir **antes
de correr `matting`** en qué clips merece la pena buscar la hoja. Con el límite en 250 px/m, los 4
clips por encima dan 11-38 % (mediana 25 %) y los 3 por debajo dan 5-10 % (mediana 10 %).
**Pero px/m no lo determina**: `uptightSlothEveningBat` tiene 305 px/m y sólo 11,2 %, así que es una
tendencia para priorizar, no un umbral para decidir.

### 29.7 C3, la firmeza de la mano, está CONFUNDIDA con el clip y no se puede contestar

La razón firme/dudoso sale **0,40×**, o sea al revés de lo esperado: los tiradores de mano firme dan
**9,8 %** de detección y los dudosos **24,7 %**. No es un resultado sobre la firmeza:

- la firmeza es **una decisión por clip y lado**, constante en todo el clip, así que comparar firmes
  contra dudosos **compara clips**;
- y los clips firmes son justo los de px/m bajo (`LiveDecimal` 252, `MessySnappy` 195) mientras los
  dudosos son los de px/m alto (`Babyish` 333, `GentleBlue` 312, `proudDungBeetle` 289). O sea que
  0,40× es C4 mirado del revés.

El test limpio es **dentro del clip**, donde px/m es constante, y sólo **2 de 7 clips** tienen un lado
firme y otro dudoso: `SeveralSkinnyGenet` da 4 de 4 detecciones en el lado firme (n pequeñísima) y
`uptightSlothEveningBat` da 10,9 % contra 11,5 %, o sea nada. **Inconcluyente**, y la pregunta sigue
abierta: hace falta una muestra con los dos regímenes dentro del mismo clip.

### 29.8 La recogida, y la auditoría de la muestra que cambió dos clips

Criterio de elección declarado antes de correr: de `corpus_etiquetado.txt`, a 1080p, con detección ya
cacheada en `det_<clip>.pkl` (para no volver a pasar YOLO, que vive en el otro intérprete), y de
**asaltos distintos**. Cuadros en **12 ráfagas de 5 consecutivos**, que da a la vez reparto por el
clip (la tasa fuera de la acción es 42,6 % y dentro 3,0 %) y **pares contiguos**, que es lo único que
mide el temblor. Parámetros de detección **idénticos a los de `e5`**, sin tocar ni uno: si cambiaran,
las hojas nuevas no serían comparables con las viejas.

| | |
|---|---:|
| clips elegidos | 9 |
| clips que dieron datos | **6** |
| recortes | **633** |
| rectas nuevas | **111 (17,5 %)** |
| total agregado con recta | 168 |
| **total delante de la mano** | **153** |
| **asaltos distintos** | **12** |

**Dos clips se cayeron y los dos por su cuenta:** `DemandingMindlessDeviltasmanian` sin mano armada
decidible, y `PowerfulAllLarva` con **0 cuadros con los dos tiradores presentes** en ráfagas.

**Y la auditoría a ojo de los 9, leyendo el marcador, cambió la muestra en dos sitios:**

- **`LiveDecimalLadybird` y `PowerfulAllLarva` son EL MISMO ASALTO** (CHEUNG Ka Long contra CHOI
  Nicholas Edward, R1 2-2 y R3 14-10). Es el mismo error que 22.2 con `Welloff`/`IdealFrigid`. Están
  agrupados en la tabla `e23.ASALTO`; sin ella el desglose «por asalto» habría mentido justo en lo
  que tiene que medir.
- **`PowerfulAllLarva` tiene además el ÁRBITRO elegido como tirador** en el lado 0 (sección 18,
  verificado a ojo y sin arreglar). Un árbitro no lleva florete, así que cualquier «hoja» de ahí sería
  otra cosa. Queda excluido explícitamente en `e27.EXCLUIR_CLIP`, y **aportó 0 detecciones** porque ya
  se había caído solo. Comprobado además que `LiveDecimalLadybird`, que es el mismo encuentro, **NO
  arrastra el problema**: pareja correcta pintada sobre el fotograma, y los dos lados `pie_firme` con
  `frac_pie` 0,92 y 1,00.

### LO QUE ESTA MEDIDA NO DICE

- **No dice que la muestra sea independiente.** `GentleBlueIndianspinyloach` es KOROBEYNIKOVA contra
  **A. VOLPI**, y el asalto que dominaba la muestra vieja (`AdorableFitting...`) es **VOLPI** contra
  KONDRICZ. Son asaltos distintos pero **la misma tiradora está en los dos**, y son el 1.º y el 2.º
  en número de hojas (43 y 24 de 153). Si la forma del desvío fuera propiedad de la técnica de cada
  tirador, esos dos no serían independientes. No se puede descartar con esta n.
- **No verifica el arma.** Muestreé de `corpus_etiquetado.txt`, que es la garantía de florete del
  proyecto, pero **mi comprobación visual del lamé fue inconcluyente** al tamaño de recorte que da
  1080p a esa distancia. No lo convierto en «verificado».
- **La tasa del 17,5 % no es comparable con el 42,6 % de `e5`.** `e5` la midió en un solo asalto con
  px/m alto; ahora se sabe que la tasa depende de px/m (29.6), así que el 42,6 % era el extremo bueno
  del rango y no una tasa del corpus.
- **C2 (escorzo) apoya por su test declarado y no por el otro.** La razón por quintiles da 1,63× y
  pasa, pero la comparación de medianas CON/SIN da 1,01×, o sea nada. El estadístico declarado para
  C2 era el de quintiles y se respeta, pero un candidato que sólo se ve en una de las dos formas no
  está establecido.
- **C4 son 7 clips.** Un spearman de +0,79 sobre 7 puntos es una tendencia, no una ley, y tiene un
  contraejemplo dentro de la propia tabla (`uptightSloth`, 305 px/m y 11,2 %).
- **El 4,2° del temblor sale de 70 pares, pero los pares no son independientes entre sí**: vienen de
  12 ráfagas de 5 cuadros, así que hay como mucho 12 tramos independientes. La cifra es mucho más
  firme que los 14,0° de 8 pares, pero no es un IC de 70 observaciones libres.
- **Nada de esto valida el cono contra el tocado ni contra el contacto de hojas.** Sigue sin haber
  verdad-terreno del contacto (0 positivos decidibles en 15 láminas, 22.20).
- **No arregla que el cono pierda la mitad de las situaciones en alcance** (24.9). Eso depende del
  vértice, o sea de `mano_armada`, y C3 no ha podido medir si la marca `dudosa` ayuda.
- **`peso_sector` sigue siendo el operativo.** El campo medido se guarda y se pinta; no decide ninguna
  amenaza. Lo único que cambia en el dato es `campo_sigma_gr` de los cuadros con `origen = 0`, que
  pasa de 14,0 a 4,2.

### 25.24 · Los eventos de la pista, remedidos con el fichero final: `ev_accion` vive, `ev_brazo` se cae

Medido contra la versión **final** de `02_validacion/pista/eventos.csv`, `mtime_ns`
**1789119066880271000**, 478 filas y 11 clips, guardado en `mano_guante.json`. Las versiones
anteriores con las que medí daban cobertura 9 y 2 tiradores; ésta cubre **20 de mis 38** con eventos
de `accion` y **19** con eventos de `brazo`. Los cambios del fichero no me muerden: uso `clip`,
`lado`, `tipo`, `inicio` y `fin`, y **no uso la columna `n`**, que es la que estaba contando saltos de
los dos tiradores y no una duración. Y no uso `parada`, que ahora se llama `alto`.

| puerta | acierto | decide | sin decidir | por cuadro dentro / fuera |
|---|---:|---:|---:|---|
| sin puerta | 0,842 | 38 | 0 | 0,826 / — |
| `ev_accion` | 0,850 | 20 | 18 | 0,868 / 0,784 **(+0,084)** |
| `ev_brazo` | 0,824 | 17 | 21 | 0,808 / 0,829 **(−0,021)** |
| `encarado` | 0,895 | 38 | 0 | 0,879 / 0,629 **(+0,250)** |
| **`encarado+ev_accion`** | **0,950** | 20 | 18 | **0,955 / 0,748 (+0,207)** |

**1. Ninguna pasa la puerta, y no la muevo.** `ev_accion` da 0,850 y `ev_brazo` 0,824, las dos por
debajo de 0,90, y las dos dejan 18 y 21 tiradores sin decidir, muy por encima del ≤ 4 declarado.
`encarado+ev_accion` sí llega a **0,950** de acierto pero deja 18 sin decidir, así que es **«gana con
coste»** y no adopción, igual que se ha hecho toda la sección.

**2. `ev_accion` queda confirmado como señal real.** Con la versión vieja daba +0,148 sobre 330
cuadros; con la buena da **+0,084 sobre 4.588 cuadros** (2.285 dentro). El efecto se encoge al
medirlo bien, que es lo normal, pero sigue siendo positivo y es **la única señal positiva que no
calculo yo**: la calcula el agente de la pista a partir de la marcha de los pies, y tiene recall
medido (49 zancadas contadas a ojo en 3 tiradores, 90 % de cobertura), no supuesto.

**3. `ev_brazo` se cae al medirlo bien, y es el tercer caso de este tipo en la sección.** Con la
versión vieja parecía +0,104 sobre 107 cuadros; con la final da **−0,021 sobre 609**. **Cambia de
signo.** Los otros dos casos fueron la ampliación del hombro (15/15 → 22/23) y los `1,000 sobre 2
tiradores` de las versiones viejas de este mismo fichero. Tres veces la misma lección: **una cifra
buena con poca cobertura no es una cifra buena**.

**4. Y las dos señales positivas se corroboran.** `encarado` (la nariz, mía) y `ev_accion` (la marcha,
del otro agente) son independientes, y su intersección da el mejor acierto por cuadro de todo el
trabajo: **0,955 dentro contra 0,748 fuera**. Es la mejor prueba de que la hipótesis de fondo era
correcta: los cuadros que no saben la respuesta son los de fuera de la acción y con el tirador no
encarado.

**5. Lo que el coordinador esperaba sobre los tres irreducibles: no los arregla, pero los APARTA.**
`WhiteMarriedGuineafowl l0` y `l1` y `PowerfulAllLarva l1` tienen **cero cuadros de evento `accion`**,
así que la puerta los deja **sin decisión** en vez de contestar mal. El único fallo de
`encarado+ev_accion` es `UglyNiceAplomadofalcon l1`, que es justo el tirador que **el criterio del pie
acierta**. O sea que las dos vías se complementan en el caso que importa.

**6. La combinación que sale de ahí, y que es POST-HOC.** Si manda el pie firme y, cuando no lo es,
decide `encarado+ev_accion`, el resultado sobre los 38 es **28 de 28 sin fallo, con cobertura 0,74**
frente a 17 de 17 con cobertura 0,45 de lo adoptado. El orden importa: al revés (puerta primero) baja
a 0,964, porque el fallo de la puerta es el acierto del pie.

**Pero esa unión la he elegido DESPUÉS de ver qué conjuntos se solapaban, así que su 28 de 28 no es
una medida declarada de antemano y no se adopta.** El precedente está en esta misma sección: la
ampliación con el hombro también daba 15 de 15 elegida así, y al subir la muestra se quedó en 22 de
23. Hace falta validarla en una muestra nueva.

**Y hay una razón de ingeniería para no implementarla todavía**, aparte de la estadística: haría que
`cono_lib` dependiera de `eventos.csv`, que es un fichero de otro agente y que ha cambiado **cuatro
veces** durante este trabajo. Un criterio operativo de la librería compartida no debería colgar de un
CSV que se reescribe. Mientras eso no se estabilice, la unión se queda medida y escrita, no cableada.

### Lo que 25.24 NO dice

- **No dice que la acción sea la puerta buena.** Dice que `ev_accion` aporta +0,084 por cuadro y que
  junto con `encarado` llega a +0,207, y que **ninguna de las dos pasa la puerta declarada** por
  cobertura. Deciden 20 de 38.
- **El 0,950 de `encarado+ev_accion` son 19 de 20**, y el que falla es uno de los dos «seguro y
  equivocado» del principio. No es una señal sin errores, es una señal con un error conocido.
- **El 28 de 28 es post-hoc** y está dicho arriba. Con la muestra de hoy no se puede distinguir de la
  suerte de haber elegido bien el orden.
- **Los tres irreducibles siguen sin arreglarse.** Lo que cambia es que ahora hay una señal que los
  deja fuera por no tener eventos de acción, en vez de contestar mal. Eso es útil para la tubería y no
  es lo mismo que resolverlos.
- **Las cifras de `ev_*` valen para esta versión del fichero**, `mtime_ns` 1789119066880271000. Está
  guardado en `mano_guante.json` precisamente porque ya cambió cuatro veces; si vuelve a cambiar, hay
  que remedir y comparar los dos `mtime`.
- **Sólo 11 de mis 23 clips tienen eventos.** La cobertura de 20 de 38 tiradores es de ahí, no de que
  la puerta rechace cuadros: en 18 tiradores no hay eventos que consultar.

## 30. «Detecta ahí un fondo que no es fondo»: el árbitro marcha/fondo, y la escala que inflaba los umbrales

El usuario, viendo el vídeo a cámara lenta: *«he visto que en los mismos, que es lo que dura en cámara
lenta, pasa a marcha, a fondo y luego marcha otra vez. Detecta ahí un fondo que no es fondo. Debería
ser muy diferente a la marcha y no lo debería detectar ahí»*. Y después: *«no es preciso»*.

Tenía razón, y lo más incómodo es que **yo ya tenía el defecto medido en la dirección contraria**: la
sección 26.5 dice «el 41 % de los pasos solapan un fondo» y «el criterio del paso describe la primera
mitad de un fondo». Lo que él vio es el mismo solape visto del otro lado. **Marcarlo con
`solapa_fondo` no era suficiente: había que decidir.**

La secuencia de `ForcefulAntiqueAlbacoretuna` lado 0 era literalmente
`MFMFMFMFMMFMRFMRRMRMFMFMFMFMFMFM` — dieciséis alternancias sobre los mismos gestos. En los 11 clips
había **73 solapes de tipos distintos del mismo tirador**.

### 30.1 Primera causa, y es de escala: el umbral del fondo estaba en metros absolutos

La hipótesis la apuntó el coordinador y es la económica: `FONDO_SALTO_M` = 0,35 m de crecimiento de la
separación de pies y `FONDO_AVANCE_M` = 0,25 m de avance son **metros absolutos**, así que en un clip
con la escala inflada **los cruza un paso normal**.

| clip | factor de escala (`dpies_p50` / 0,70) | fondos por segundo |
|---|---:|---:|
| `ForcefulAntiqueAlbacoretuna` | **2,14** | **1,50** |
| `AdorableVeneratedDuiker` | 1,62 | 0,98 |
| `SeparateHoarseFeline` | 1,51 | 1,10 |
| `GentleQuerulousElephantseal` | 1,34 | 0,58 |
| `AliveIgnorantBarebirdbat` | 0,83 | 0,37 |
| `UglyNiceAplomadofalcon` | **0,92** | **0,11** |

**Correlación medida: r = 0,83.** La hipótesis se confirma. El clip con la escala más inflada es el
que más fondos da, y el que la tiene bien es el que menos.

El arreglo es normalizar por la **guardia del propio clip**, que es una referencia libre de escala. Y
el coeficiente se elige **para reproducir los números declarados**, no para ajustar nada: con la
guardia nominal de 0,70 m, `0,50 x 0,70 = 0,35 m` y `0,36 x 0,70 = 0,25 m`, o sea exactamente los
umbrales de la sección 23.8. Es el mismo criterio hecho inmune a la escala.

Efecto: **68 fondos → 52 (−24 %)**, y donde más baja es donde peor está la escala
(`Forceful` 18 → 7, `Adorable` 8 → 4).

**Y LA NORMALIZACIÓN ES SIMÉTRICA, que es la parte que no se puede omitir:** aprieta donde la escala
está inflada y **afloja donde está comprimida**. `AliveIgnorantBarebirdbat` tiene factor 0,83, así
que su umbral BAJA a 0,29 m y sus fondos **suben de 5 a 9**, con sus marchas cayendo de 11 a 4. Eso
es lo correcto **si** la guardia refleja la escala, y es lo contrario si refleja un error de pose. Las
dos causas no se distinguen con lo que hay, así que la columna `guardia_plausible` lo marca: de los **17
lados** que aparecen en el CSV con guardia, **10 caen en los 0,50-0,90 m** de un floretista y **7
no**, con valores de **0,265 a 1,371 m**. La normalización no es gratis: cambia
los recuentos en los dos sentidos y hereda la calidad de la medida de la guardia.

**Y NO SE RECORTA LA GUARDIA AL RANGO FÍSICO 0,50-0,90 m, deliberadamente:** normalizar por la
guardia MEDIDA es lo que cancela un error de escala, que es el defecto que se arregla. Si la guardia
sale mal por un error de POSE, normalizar por ella empeora — y las dos causas no se distinguen con lo
que hay. Así que se usa la medida y se marca: columna **`guardia_plausible`**, y las guardias usadas
van de **0,27 a 1,37 m**, o sea que varias están fuera del rango de un floretista y quien no se las
crea puede filtrar.

### 30.2 Segunda causa: el discriminador no puede ser la magnitud, es EL PIE DE ATRÁS

La magnitud es justo donde se pisan (fondo 0,60 m contra marcha 0,34 m de mediana, con colas
solapadas), así que por ahí no se separan nunca. La diferencia es **cualitativa** y es la definición
de esgrima:

 - **MARCHA**: el pie de delante avanza y **el de atrás LE SIGUE**. La separación crece y vuelve.
 - **A FONDO**: el de delante sale y **el de atrás SE QUEDA CLAVADO**, que es lo que da la extensión.
   La separación crece y se queda grande hasta el desarme.

Dos rasgos, ninguno la magnitud:

| rasgo | marcha | fondo | frontera declarada |
|---|---|---|---:|
| `razon_pie_atras` = desplazamiento del pie de atrás / del de delante | cerca de 1 | cerca de 0 | **0,45** |
| `pico_dpies_guardia` = pico de la separación / guardia del clip | cerca de 1 | > 1 | **1,25** |

Medido sobre los 48 eventos que pasaron por el árbitro, los rasgos separan limpio:

| decisión | n | `razon_pie_atras` p50 | `pico/guardia` p50 |
|---|---:|---:|---:|
| **paso** | 21 | **0,63** | 1,15 |
| **fondo** | 27 | **0,06** | 1,87 |

### 30.3 Una corrección medida: los rasgos van en la fase de EXTENSIÓN, no en toda la ventana

Midiéndolos de extremo a extremo de la ventana del evento, el acierto salía **4 de 6**. La causa es
geométrica: cuando la ventana incluye la **recuperación**, el pie de delante VUELVE, su
desplazamiento neto se anula y la razón del pie de atrás se dispara. En `AliveIgnorantBarebirdbat` l1
c273-285 la razón pasaba de **0,97** (que dice marcha) a **0,15** (que dice fondo, y a ojo lo es) sólo
con medirla hasta el cuadro del pico de separación. **La física del fondo está en la extensión, no en
el ciclo completo.**

### 30.4 El arbitraje, y qué conserva

Los eventos de movimiento del mismo tirador se agrupan por solape (componentes conexas en el tiempo),
se miden los rasgos en la ventana unión y se decide el tipo del grupo:

 - grupo **FONDO** → se emite el fondo de mayor magnitud y se descartan los pasos;
 - grupo **PASO** → se descartan los fondos y **se conservan TODOS los pasos**.

Esa última regla es la que arregla el caso del usuario sin perder información: el grupo
`marcha c25-37 + fondo c36-47 + marcha c39-52` se resuelve como paso, se cae el fondo y **quedan las
dos marchas**. Emitir un solo evento por grupo habría perdido un paso de verdad.

Resultado, y es el que contesta al usuario:

| | antes | después |
|---|---:|---:|
| **solapes de tipos distintos** | **73** | **0** |
| `Forceful` lado 0 | `MFMFMFMFMMFMRFMRRMRMFMFMFMFMFMFM` | `MMMMMMMRFRRMRMMMMMMM` |
| fondos | 68 | **33** |
| marchas | 116 | 88 |
| romper | 85 | 84 |
| pasos por tirador y clip | 9,1 | **7,8** |
| magnitud p50, marcha / fondo | 0,343 / — | **0,335 / 0,602 m** |

**La alternancia desapareció** y hay un **ASERTO** que aborta si dos eventos de movimiento de tipos
distintos del mismo tirador vuelven a solaparse. Es el mismo patrón que arregló los agregados y el
bote.

Cada evento arbitrado lleva `arbitrado`, los dos rasgos y **`margen_arbitraje`**, que es la distancia
al umbral del rasgo que decidió: p50 **0,286**, y sólo **2 de 48** quedan por debajo de 0,05, o sea en
la frontera. Esos dos son los contradecibles y se pueden localizar en el CSV.

### 30.5 LA MATRIZ DE CONFUSIÓN, que es la cifra que pedía el usuario

Se anotaron a ojo los 6 gestos solapados de los tres tiradores ya contados, con tiras de cinco
cuadros y siguiendo el pie de atrás contra las rayas pintadas de la pista:

| clip | cuadros | a ojo | lo que emite ahora | |
|---|---|---|---|---|
| `GentleQuerulousElephantseal` l0 | 189-202 | marcha | marcha c189-200 | ok |
| `GentleQuerulousElephantseal` l0 | 275-287 | **fondo** | marcha c275-285 | **falla** |
| `AliveIgnorantBarebirdbat` l1 | 44-55 | fondo | fondo c43-55 | ok |
| `AliveIgnorantBarebirdbat` l1 | 171-206 | marcha | marcha c171-203 | ok |
| `AliveIgnorantBarebirdbat` l1 | 273-285 | fondo | fondo c273-280 | ok |
| `UglyNiceAplomadofalcon` l0 | 151-167 | marcha | marcha c157-167 | ok |

```
  a ojo MARCHA -> emite paso 3, emite fondo 0
  a ojo FONDO  -> emite paso 1, emite fondo 2
  acierto 5 de 6 = 83 %
```

**Y la dirección del error es la buena**: `marcha -> fondo` es **0**. Ya no llama fondo a ninguna
marcha, que es exactamente lo que el usuario estaba viendo. El único fallo es un fondo perdido
(`Gentle` c275-287, con `razon_pie_atras` 0,48 contra la frontera de 0,45: un caso de frontera donde
el pie de atrás se movió casi la mitad que el de delante).

Antes del arbitraje el detector emitía **los dos tipos en los 6 casos**, o sea 6 ambigüedades.

### 30.6 Lo que NO se hizo, y es lo que más importa del método

Con umbrales de **0,55 y 1,30** en vez de los declarados 0,45 y 1,25, el acierto sube a **6 de 6 =
100 %** en esos mismos seis casos. **No se adoptan.** Elegir el umbral mirando el resultado sobre seis
ejemplos es ajustar al conjunto de prueba, y con n=6 la diferencia entre 83 % y 100 % es un caso. Se
publican los dos números y se usa el declarado.

Y sobre *«igual tienes que entrenar algo»*: **con 6 gestos y 49 zancadas no se entrena, se evalúa.**
El cuello de botella son las etiquetas, no el algoritmo, y una regla con la física bien puesta (el pie
de atrás) bate a un clasificador con esa n. Lo que haría posible entrenar es
`CONVENIO_CONTEO_PASOS.md`, que permite etiquetar a escala sin que cada pasada cuente cosas distintas.

### 30.7 LO QUE ESTA MEDIDA NO DICE

- **El 83 % es sobre 6 gestos.** No es una tasa de clasificación: es lo que se pudo anotar a ojo. Lo
  que sí es sólido es la dirección del error (`marcha -> fondo` = 0) y que los 73 solapes son 0.
- **La normalización por la guardia quita fondos de verdad, no sólo falsos.** En
  `GentleQuerulousElephantseal` los fondos pasan de 6 a 0, y uno de ellos (c275-287) está verificado a
  ojo como fondo real. O sea que el −51 % de fondos incluye al menos un falso negativo, y el precio
  de quitar los espurios no es cero.
- **La guardia como referencia de escala supone que la separación de pies está bien medida.** Las
  guardias usadas van de 0,27 a 1,37 m y varias están fuera de lo que puede ser un floretista; en esos
  clips el umbral del fondo hereda el error. Columna `guardia_plausible`.
- **No dice que el pie de atrás sea suficiente.** La altura de la cadera, que el coordinador propuso
  como tercer rasgo independiente, **no se ha medido**: con dos rasgos ya se llegaba al criterio
  declarado y añadir un tercero sin necesidad habría sido ajustar. Queda como la vía si con más
  etiquetas el 83 % no sube.
- **El árbitro no arregla el recall de los pasos**, que sigue medido en 90 % de zancadas (§28.2).
  Arregla la PRECISIÓN de la clasificación entre los dos tipos, que es otra cosa.

### 25.25 · El conjunto reservado: lo adoptado aguanta fuera de muestra, y la combinación post-hoc NO SE PUEDE JUZGAR

10 clips nuevos, 20 tiradores, elegidos con **semilla 20260911** (distinta de la 20260910 de todo lo
demás), así que no se solapan con los 23 clips de entrenamiento y **no participaron en elegir ninguna
regla**. Anotados a ciegas con el mismo método: mosaicos sin nombre de clip ni de lado, orden barajado
con semilla, clave aparte no mirada hasta después de anotar, segunda pasada de 6 cuadros sobre los 8
que quedaron en `certeza=media`. **Ninguna letra cambió entre las dos pasadas** y 5 subieron de media
a alta. Uno de los 20 es el **árbitro** entregado como tirador, así que queda en `?`: **19 con verdad**.

**La regla que se juzgaba estaba declarada antes de anotar y antes de medir**, en el punto 18 del
docstring de `mano_guante.py`: *pie firme manda; si no es firme, decide `encarado + ev_accion`; si no,
sin decisión*. Con criterio de aceptación fijado también antes: (i) acierto ≥ 0,90, (ii) cobertura
≥ 0,50, (iii) más cobertura que `pie firme` solo sin bajar su acierto.

#### El veredicto: ni pasa ni se cae. No se puede juzgar, y el motivo no lo había previsto

**Ninguno de los 19 tiradores del reservado tiene eventos de `accion`.** `eventos.csv` cubre 11 clips
y **los once están en el entrenamiento**. Así que en el reservado la segunda mitad de la regla **no se
ejecuta ni una vez** y `REGLA_18` se reduce exactamente a `pie firme`: decide los mismos 12 tiradores.

| condición declarada | resultado |
|---|---|
| (i) acierto ≥ 0,90 | **1,000** ✓ |
| (ii) cobertura ≥ 0,50 | **0,63** (12 de 19) ✓ |
| (iii) más cobertura que `pie firme` solo | **añade CERO** ✗ |

**Luego la regla no se adopta.** No porque haya fallado, sino porque **este reservado no puede
probarla**. Para probarla hacen falta eventos de los 10 clips reservados, y eso es trabajo del agente
de la pista.

#### Lo que sí valida el reservado, y es justo la mitad que el usuario adoptó

**`pie firme` da 12 de 12 sin fallo, cobertura 0,63, en clips que no vio.** Y su marca funciona fuera
de muestra: el criterio del pie se equivoca en 2 de los 19 —`InsecurePettyCreature l1` y
`MediumShoddyAmphiuma l0`— y **los dos tienen `frac_pie` = 0,58**, o sea que los dos salen marcados
`dudosa` y **ninguno se cuela**. Eso es la validación out-of-sample de la firmeza, que hasta ahora
sólo tenía 17 casos de entrenamiento.

#### Y la cifra que se pedía, con la lectura que de verdad tiene

| criterio | entrenamiento (38) | **reservado (19)** |
|---|---:|---:|
| **(a), la línea base** | 0,763 | **0,947** |
| `a_voto` | 0,842 | 1,000 |
| `pie_x` | 0,895 | 0,895 |
| `nariz_dir` | 0,895 | 1,000 |
| `hombro_nariz` | 0,895 | 1,000 |
| `mayoria_3` | 0,921 | 1,000 |
| `hombro_conf` | 0,763 | 0,737 |
| `guante` | 0,447 | **0,263** |
| pie firme | 17 de 17 (0,45) | 12 de 12 (0,63) |

**La diferencia entre las dos columnas NO se puede leer como engaño post-hoc, y ése es el resultado
más útil de esta tanda.** La prueba es que **la propia línea base sube de 0,763 a 0,947**: si el salto
fuera sobreajuste mío, la línea base no se movería, porque el criterio (a) no lo he elegido yo ni lo he
tocado.

#### CORRECCIÓN de la atribución, del 2026-09-11, porque la primera versión de este apartado la tenía al revés

Escribí que el reservado era «más fácil porque estaba filtrado por calidad». **Eso era falso y cambia
el remedio.** Los diez clips reservados se eligieron **al azar** del corpus con semilla 20260911, sin
mirar ninguna métrica de calidad; que salieran con las dos poses en el 100 % de los cuadros
utilizables fue el **resultado** de calcularlas, no el criterio de selección.

La causa real es la contraria y es más interesante: **el conjunto de ENTRENAMIENTO no es aleatorio**.
Se fue formando a lo largo del día con los clips que se estaban investigando **porque daban
problemas** —el árbitro entregado como tirador, el trozo de espectador del borde, el tirador medio
fuera de cuadro, las dos raíces colapsadas, los clips que sacó el buscador de errores—, así que está
**enriquecido en casos difíciles por construcción**. Consecuencia, y es la que importa:

- las cifras del **entrenamiento** son **cota por lo BAJO** del rendimiento en el corpus, porque la
  muestra está sesgada hacia lo difícil;
- las cifras del **reservado** son la **mejor estimación disponible a nivel de corpus**, porque es la
  única muestra aleatoria que hay;
- y por tanto **no hace falta un tercer conjunto**. Lo que hace falta es leer las dos muestras como lo
  que son: una mide el rendimiento **en los casos duros** y la otra el rendimiento **típico**.

Lo único que se puede afirmar limpiamente: **en 19 tiradores que ninguna regla vio, el pie firme no ha
fallado ninguno de los 12 que decide, y su marca ha cazado los 2 errores del pie.** Eso sube el
enunciado de lo adoptado de «no ha fallado ninguno de 17» a **«no ha fallado ninguno de 29»**, 17 en
entrenamiento y 12 en reservado, que se cuentan por separado a propósito. Y que **la marca funcione
fuera de muestra** es el resultado más sólido de toda la sección 25.

Y **el guante baja de 0,447 a 0,263**, que es la cuarta confirmación de que ahí no había señal.

### Lo que 25.25 NO dice

- **No dice que la combinación post-hoc sea buena.** Dice que **no se ha podido probar**, porque el
  reservado no tiene eventos. Sigue siendo post-hoc y sigue sin adoptarse, y la objeción de ingeniería
  —que `cono_lib` no debe leer `eventos.csv`— sigue en pie: si algún día se cablea, los tramos de
  acción tienen que **entrar como argumento** a `mano_armada` (una máscara booleana `(T,)` como la
  `vota` de `mano_armada_voto`) o **precalcularse a un array dentro del NPZ de poses**, que es donde
  vive el resto de la procedencia.
- **El 0,947 del criterio (a) es el rendimiento TÍPICO, y el 0,763 el de los casos duros.** Las dos
  cifras son buenas y miden cosas distintas; ninguna de las dos es «la» cifra del corpus por sí sola,
  aunque la del reservado es la única que viene de una muestra aleatoria.
- **12 casos sin fallo siguen siendo 12.** Sumados a los 17 son 29 sin fallo, y el intervalo inferior
  de 29 de 29 anda por 0,88. Sigue sin ser «acierta siempre».
- **Y no se han agregado las dos muestras en un total**, a propósito. Agregarlas borraría justo la
  información que el reservado aporta.

### 25.26 · EL ÁRBITRO ENTREGADO COMO TIRADOR: 2 de 33 clips, y ninguna métrica de calidad de pose lo va a cazar

Ésta es la cifra que sale de las **66 filas de verdad anotadas a ojo sobre 33 clips**, y la doy porque
cambia el estatus del defecto: ya no es una anécdota de un clip, es un **modo de fallo recurrente**.

| clip | qué se entrega como tirador | cómo se detectó |
|---|---|---|
| `PowerfulAllLarva` l0 | **el árbitro**: traje oscuro, sin careta, caminando junto a la pista | ya documentado en 18.1, reencontrado al anotar |
| `IdolizedFancyChickadee` l1 | **el árbitro**: hombre de traje con barba, sin careta ni uniforme | **nuevo**, y aparece en una muestra ALEATORIA |
| ~~`LavishDistantDikkops` l0~~ | ~~un trozo de espectador pegado al borde izquierdo~~ | **RESUELTO: era un NPZ OBSOLETO**, ver abajo |

**2 de 33 clips = 6,1 % de los clips entregan un no-tirador, y los dos son EL ÁRBITRO.** Son
**2 de 66 tiradores = 3,0 %**. La primera versión de este apartado decía 3 de 33 = 9,1 % contando el
trozo de espectador, y **esa cifra ya no vale**: el caso del espectador sólo existía en un fichero
viejo.

#### El caso del espectador era un NPZ anterior al arreglo, y la medida es la que lo identificó

La duda que dejé planteada —«NPZ anterior al arreglo» contra «la métrica de la tubería no coincide con
la de la caja entregada»— tiene respuesta: **era la primera**. Regenerado
`poses_LavishDistantDikkops.npz`:

| | antes | ahora |
|---|---:|---:|
| caja del lado 0 pegada al borde lateral | **1,000** de 273 cuadros | **0,003** de 290 |
| cuadros con los dos tiradores entregados | 79 | **290 de 290** |
| cuadros utilizables para la mano armada | **7** | **100** |

Así que **`pareja_v2.BORDE_SIEMPRE = 1.0` funciona en la tubería** y el defecto sólo sobrevivía en el
fichero. Lo que lo identificó fue exactamente la medida de que era **el único de los 66 tiradores que
llegaba a 1,000** con un p50 de 0,022 y un segundo valor de 0,992: sin esa separación no se habría
sabido cuál regenerar.

Y de paso el clip **entra en la medida**: la verdad pasa de 38 a **40 tiradores de entrenamiento**
(`LavishDistantDikkops` l0 y l1, que antes no se podían medir por tener 7 cuadros útiles). La fila de
l0 tiene ahora **tres versiones anotadas en el CSV**, y las tres quedan escritas: «fuera de cuadro»
(mal), «trozo de espectador» (correcto para el NPZ viejo) y **`A` con cue `accion`** sobre el NPZ nuevo,
donde en k=210 y k=213, en pleno fondo, `A` es la mano extendida al rival y `B` el brazo trasero.

#### Búsqueda de más NPZ obsoletos, con el mismo método: no hay ninguno

Se aplicaron a los **33 NPZ** tres pruebas que un arreglo documentado del proyecto debería haber
dejado a cero:

| prueba | de dónde sale | resultado |
|---|---|---|
| `origen == 2`, poses SOSTENIDAS | 21.2 dice que no deben existir | **0 en los 33** |
| caja pegada al borde lateral en el 100 % de los cuadros presentes | puerta `BORDE_SIEMPRE` de 18.4 | **0 en los 33** tras regenerar; el máximo ahora es 0,992 |
| las dos cajas idénticas, misma persona en los dos lados | defecto 14.3 | **0 en los 33** |

**No hay ningún otro NPZ obsoleto detectable así.** El aviso obligado: estas tres pruebas sólo cazan
defectos que un arreglo conocido debería haber puesto a cero, así que un fichero viejo cuya única
diferencia sea algo que no se prueba aquí **no aparecería**.

Dos avisos sobre esa cifra. Es una **cota por lo bajo**: sólo cuenta lo que noté mirando mosaicos de
3 a 6 cuadros por tirador, y un árbitro que pasara por delante en parte del clip podría habérseme
escapado. Y el 9,1 % se apoya en **3 casos**, así que el intervalo es ancho; lo que no depende del
tamaño de muestra es que **uno de los tres apareció en una muestra aleatoria de 10 clips**, que es lo
que descarta que sea una rareza del subconjunto que estábamos investigando.

#### Por qué es peligroso: la pose del árbitro es perfectamente buena

Los diez clips reservados salieron con **las dos poses en el 100 % de los cuadros utilizables** —o sea
calidad de pose impecable— **y uno de ellos entrega al árbitro**. Eso no es una casualidad: el árbitro
es una persona de pie, entera, bien iluminada y bien seguida, así que **su pose es mejor que la de
muchos tiradores**. Ninguna métrica de calidad de pose —fracción de cuadros con pose, confianza media,
poses inventadas, huecos— lo va a detectar **nunca**, porque no hay nada malo en la pose: lo que está
mal es **a quién pertenece**.

Y la sección 18 ya lo había razonado para la geometría: el árbitro camina junto a la pista, así que
tiene los pies igual de bajos en el encuadre y una talla parecida, y pasó las tres puertas de
`pareja_v2` **sin rozarlas**.

#### Una cosa que sí es medible y que apunta a una regresión, para quien lleve `pareja_v2`

La puerta que debía cazar el caso del espectador existe y tiene el umbral correcto:
`pareja_v2.BORDE_SIEMPRE = 1.0`, que descarta una trayectoria cuya caja esté **cortada por el borde
lateral en el 100 % de sus cuadros** (18.4). Medido sobre los 66 tiradores de los NPZ entregados:

| | |
|---|---:|
| `LavishDistantDikkops` l0, fracción de cuadros presentes con la caja pegada a un borde lateral | **1,000** sobre 273 cuadros |
| tiradores de los 66 que llegan a 1,000 | **1**, exactamente ése |
| reparto del resto | p50 0,022 · p90 0,445 |
| el segundo más alto, `BabyishIdioticCockerspaniel` l1 | 0,992, justo por debajo de la puerta |

O sea que **la puerta separa el caso perfectamente y el caso sigue en los NPZ entregados**. Puede ser
que el NPZ se generara antes del arreglo, o que la `borde` que calcula la tubería no sea la misma que
se mide sobre la caja entregada. No lo toco porque `pareja_v2` y `extrae_pareja` son de otro agente,
pero queda la comprobación hecha y el número puesto.

Y un detalle que sale de la misma tabla: el segundo más alto, con 0,992, es justo el tirador de las
**dos raíces de mano colapsadas** (`BabyishIdioticCockerspaniel` l1). Un tirador medio fuera de cuadro
todo el clip y con las manos colapsadas es un candidato a marcar, aunque **sí** sea un tirador.

#### Qué haría falta para atacarlo, dicho como hipótesis y no como resultado

Lo que distingue al árbitro de un tirador **no es geométrico ni de calidad de pose**: es que **no
lleva careta, no lleva uniforme blanco y no lleva arma**. La sección 18.3 ya midió que la blancura
absoluta no vale de puerta y que el contraste entre los dos lados tampoco separa limpio, por el lamé
dorado y por las sombras. Pero el árbitro acumula **tres** ausencias a la vez, y ninguna de las tres
se ha probado en conjunto. En particular, la **careta** es el objeto más constante de un tirador —la
lleva todo el asalto, es grande y está en un sitio fijo del cuerpo— y en este trabajo sólo se midió
al revés, como objeto suelto en la mano (25.22). Eso queda como vía, sin cifra, y por tanto sin
adoptar nada.

### Lo que 25.26 NO dice

- **El 6,1 % son 2 casos de 33 clips**, y es una **cota por lo bajo** porque sólo cuenta lo que noté
  mirando de 3 a 6 cuadros por tirador. No es una tasa medida sobre los 8.558 clips del corpus.
- **La primera versión de este apartado dijo 9,1 % y estaba mal**, porque contaba como tercer caso el
  trozo de espectador de `LavishDistantDikkops`, que era un **NPZ obsoleto** y ya está regenerado. La
  cifra buena es 2 de 33, y los dos son el árbitro. Se deja escrita la corrección porque el camino
  —medir, encontrar el outlier, regenerar, recontar— es la parte reutilizable.
- **El arreglo de 18.4 NO estaba roto**, y ésa era la duda que quedaba abierta: `BORDE_SIEMPRE = 1.0`
  funciona en la tubería y el caso sólo sobrevivía en el fichero. Lo que sí queda dicho es que **un
  NPZ viejo puede sostener un defecto ya arreglado durante horas sin que nada avise**, y que la forma
  de encontrarlo fue buscar al que se separaba de la distribución.
- **La vía de la careta para cazar al árbitro no tiene ninguna cifra** y además las tres ausencias de
  apariencia están medidas y fallan por separado. La vía que sí tiene criterio declarado es la de
  **conducta**, en 25.27, y su lado positivo sigue sin medir.
- **Y no dice que los 2 casos estén arreglados ni marcados.** Hoy salen de la tubería como tiradores
  normales. Lo único que hay es que `mano_armada` los devolverá con `dudosa` si su pie no es firme, y
  eso es accidental, no una detección del problema.

### 25.27 · El árbitro NO ESGRIME: la vía de conducta, declarada, y con el confundido ya medido

Las tres ausencias visibles del árbitro están medidas y **las tres fallan por separado**: la blancura
se la come el lamé dorado (18.3, contraste 0,477 entre dos tiradores correctos), la careta como objeto
oscuro es un detector de oscuridad con precisión ~1/35 (25.22), y la hoja es invisible en el 81 % del
corpus, que es 720p. Sumar tres señales débiles con los mismos confundidos no arregla nada.

Hay una **cuarta ausencia que no es de apariencia sino de CONDUCTA: el árbitro no esgrime.** No hace
nunca un a fondo, no extiende el brazo, anda de forma continua en vez de alternar avance y retroceso,
y no tiene la estructura de acercarse y alejarse de un asalto. Y es **inmune al lamé, a la iluminación
y a la resolución**, que son los tres confundidos que han tumbado todo lo anterior.

**Criterio declarado antes de medirlo** (punto 19 del docstring de `mano_guante.py`):

- rasgo: `tasa` = eventos de tipo **`fondo` o `brazo`** de ese lado, por **segundo útil** de ese lado.
  Se usan ésos dos y **no** `marcha` ni `romper` a propósito: andar y retroceder los hace también el
  árbitro; el a fondo y la extensión de brazo son específicos de esgrimir.
- criterio: **cero `fondo` y cero `brazo` en más de 4,0 segundos útiles → no es un tirador.** Los 4 s
  salen de la cifra ya medida por el agente de la pista, 0,56 a 1,75 pasos por segundo y tirador: a la
  tasa más baja, 4 s son más de 2 acciones esperadas. **El umbral no se barre.**
- positivos: los 2 árbitros anotados. Negativos: los 63 tiradores restantes. Y lo que hay que publicar
  no es sólo la separación sino **los falsos positivos**, porque un tirador defensivo que pase el
  asalto retrocediendo podría tener pocos a fondo.

#### Medida PARCIAL, y el confundido sale bien

`eventos.csv` está **en plena reescritura** mientras escribo esto: la versión que he consultado tiene
`mtime_ns` **1789121649481043000** (12:14:09), 574 filas y 15 clips, de los que **4 ya son del
reservado**. **Ninguno de los dos clips con árbitro tiene eventos todavía**, así que el lado positivo
del test **no se puede medir**. Lo que sí se ha medido es el lado negativo, que es el confundido:

| | |
|---|---:|
| lados con eventos consultables | 26 |
| de ellos, tiradores anotados | **24** |
| `tasa` de los tiradores: p10 / p50 / p90 | **0,493 / 0,734 / 0,963** |
| máximo | 1,239 |
| **tiradores con `tasa` = 0** | **0** |
| tirador con la tasa más baja | `ForcefulAntiqueAlbacoretuna` l0, **0,333** (4 eventos en 12,0 s) |
| **el criterio marca** | **0 de 26 lados** |
| **FALSOS POSITIVOS sobre 24 tiradores** | **0** |

O sea que **el confundido que había que batir no aparece**: ningún tirador de los 24 con eventos tiene
cero eventos de esgrima, y el más flojo tiene 4 en 12 segundos. Si un árbitro sale con cero, el
criterio separaría limpiamente en esta muestra. Pero **eso está sin medir**, y lo digo en la misma
frase que el resultado.

#### Lo que con 2 positivos se puede y no se puede hacer

**Con 2 positivos no se valida nada.** Lo que sí se puede es **descartar**: si cuando lleguen los
eventos los dos árbitros no se separan de los tiradores, la vía está muerta y nos ahorramos
perseguirla. Y si se separan limpiamente, lo que hay que dar **no es un acierto** sino cuántos clips
habría que anotar para validarlo.

Esa cifra se puede estimar ya, y es la parte accionable: si la tasa de árbitro entregado es del orden
del **6,1 % de los clips** (2 de 33), para juntar **10 positivos** —que es el mínimo con el que una
tasa de acierto empieza a tener un intervalo manejable— habría que anotar del orden de
**160 clips**. Y como anotar un clip cuesta dos mosaicos y mirarlos, eso es una tarde de trabajo de
ojo, no de CPU. La alternativa más barata es **buscar los positivos a propósito** en vez de al azar:
los clips donde los dos lados se cruzan mucho (18.1 midió 29 de 193 cuadros en `PowerfulAllLarva`)
son el sitio donde mirar primero, y eso reduce el número de clips a anotar en un factor grande a
cambio de introducir un sesgo de selección que habría que declarar.

### Lo que 25.27 NO dice

- **No dice que la vía funcione.** Dice que el **confundido no aparece** (0 falsos positivos en 24
  tiradores, tasa mínima 0,333) y que **el lado positivo está sin medir** porque los dos clips con
  árbitro no tienen eventos todavía. Cuando los tengan, se mide **una vez** con el criterio ya
  congelado.
- **Las cifras son de una versión de `eventos.csv` que está cambiando**, `mtime_ns`
  1789121649481043000. Es la quinta versión que veo hoy. La medida definitiva lleva su propio `mtime`.
- **Los 4,0 segundos y el uso de `fondo` + `brazo` son elecciones declaradas, no ajustadas.** Si al
  medir los positivos hubiera que moverlos, eso sería un ajuste y habría que decirlo y revalidar con
  otra muestra.
- **La estimación de 160 clips sale de una tasa de 2 casos.** Es un orden de magnitud para decidir si
  merece la pena, no una planificación.
- **Y la vía no arregla nada por sí sola**: detectar que un lado no es un tirador no dice quién era el
  tirador real, que es lo que hizo falta en `LavishDistantDikkops` y lo que resolvió regenerar el NPZ.

## 31. Los 10 clips reservados, la vía sin homografía, y el criterio del árbitro que NO separa

Encargo: correr la detección de eventos sobre 10 clips reservados (más `PowerfulAllLarva`) para que
otro agente pudiera juzgar una regla de prioridad que usa mi evento `accion` como puerta, y para
probar la hipótesis de que **el árbitro no esgrime**.

### 31.1 Tener la pose de 308 puntos NO basta: 6 de los 10 no tienen homografía

El encargo suponía que con la pose calculada ya estaba. No: `marcha`, `romper`, `fondo` y `alto`
tienen umbrales en **metros**, y los metros salen de la homografía. Medido sobre los 10:

| | clips |
|---|---:|
| con pose de 308 puntos, pickle y vídeo | **10 de 10** |
| con **homografía** | **4 de 10** |
| sin homografía | **6 de 10** (`MediumShoddyAmphiuma`, `NextImmaterialGoldfish`, `GloomyBitterBluegill`, `CandidBronzeIndigowingedparrot`, `NervousHospitableChick`, `IdolizedFancyChickadee`) |

Los seis daban **cero eventos**, que es lo peor posible para una prueba: el cero de «no se pudo
medir» es indistinguible del cero de «se midió y no hay».

### 31.2 Lo que SÍ se puede medir sin homografía, y por qué son exactamente dos tipos

 - **`accion`**, porque es una PUERTA DE VELOCIDAD y no una medida de longitud: se puede pasar a
   **altos de caja por segundo**, midiendo el centro de la caja en píxeles del cuadro
   **estabilizado** y normalizando por el alto de la caja del propio tirador. El umbral se elige para
   reproducir el métrico, no se inventa: `0,50 m/s / 1,75 m = 0,29` altos de caja por segundo.
 - **`brazo`**, que **nunca necesitó la homografía**: se mide como el largo del eje
   `hombro -> raíz de la mano` en fracción del largo del TORSO, que es una reglilla de imagen. Sólo
   usaba la homografía para escribir `pos_m`.

Y los cuatro que NO se pueden: sus umbrales están en metros y no se trasladan a altos de caja sin
suponer que la relación píxel-metro es la misma en todo el cuadro, y **no lo es**: la pista se aleja
de la cámara.

Columna **`via_accion`** = `metros` o `alto_de_caja`, porque las dos vías no son la misma medida.
Y en esos clips `pasos_por_s_util_y_tirador`, `cruces_imposibles` y `ancla_deriva_m` van a **`None`**
y no a cero, con columna **`sin_homografia`**.

### 31.3 EL CRITERIO DEL ÁRBITRO NO SEPARA, y hay que decirlo antes de que se mida

El criterio estaba congelado antes de medir por el otro agente: `tasa` = eventos de `fondo + brazo`
por segundo útil, y **cero en más de 4,0 s significa que no es un tirador**. El confundido estaba
batido de antemano: sobre 24 tiradores reales, `tasa` p10 **0,493** y **ninguno a cero**.

Los dos números que faltaban:

| clip | lado | es | fondo | brazo | **tasa** |
|---|---:|---|---:|---:|---:|
| `PowerfulAllLarva` | 0 | **ÁRBITRO** | 0 | **4** | **0,524** |
| `PowerfulAllLarva` | 1 | tirador | 0 | 1 | 0,131 |
| `IdolizedFancyChickadee` | 1 | **ÁRBITRO** | 0 | 1 | **0,100** |
| `IdolizedFancyChickadee` | 0 | tirador | 0 | 2 | 0,200 |

**El criterio falla en los dos, y en uno falla al revés:** el árbitro de `PowerfulAllLarva` tiene
`tasa` **0,524**, que es **más alta que el p10 de los tiradores reales (0,493)** y **cuatro veces la
de su propio rival**. Ninguno de los dos árbitros da cero, así que la regla «cero = no es tirador» no
los caza.

**Y hay un agravante estructural: `fondo` es 0 POR CONSTRUCCIÓN en los dos**, porque ninguno tiene
homografía. O sea que sólo se ejecuta la mitad `brazo` del criterio, y el 0 de `fondo` **no es
evidencia de nada**. Aunque la tasa hubiera salido cero, no se podría haber concluido.

### 31.4 Mi explicación física era falsa, y la medí en vez de afirmarla

Propuse que `brazo` dispara en el árbitro porque **levanta el brazo para señalar**, y que entonces el
ÁNGULO del eje del brazo debería separarlos: el árbitro hacia arriba, el tirador hacia el rival. Es
una hipótesis razonable y es **falsa**:

| clip | lado | es | ángulo p50 | ángulo p90 | % cuadros por encima de 30° |
|---|---:|---|---:|---:|---:|
| `PowerfulAllLarva` | 0 | **ÁRBITRO** | **−76,1°** | 4,3° | **0,0 %** |
| `PowerfulAllLarva` | 1 | tirador | −21,9° | 1,8° | 0,0 % |
| `IdolizedFancyChickadee` | 1 | **ÁRBITRO** | **−36,0°** | −31,2° | **0,0 %** |
| `IdolizedFancyChickadee` | 0 | tirador | −19,0° | 25,6° | 8,2 % |
| `MediumShoddyAmphiuma` | 1 | tirador | −62,5° | −20,7° | 0,0 % |
| `GentleQuerulousElephantseal` | 0 | tirador | −20,1° | 0,0° | 0,0 % |
| `AliveIgnorantBarebirdbat` | 1 | tirador | −33,8° | −11,4° | 3,5 % |

En los dos clips el brazo del árbitro apunta **hacia ABAJO más que el del tirador**, y el porcentaje
de cuadros con el brazo levantado es **0,0 % en los dos árbitros**. No sólo no separa en la dirección
que predije: la única separación aparente va en sentido contrario, y los tiradores reales van de −20°
a −62°, que solapa con los −36° del árbitro de `IdolizedFancyChickadee`.

Así que `brazo` dispara en el árbitro por otra razón, probablemente el braceo de andar: el criterio es
«el eje se alarga por encima de 1,15 veces **su propia mediana**», y una persona que anda alarga y
acorta ese eje sin extender nada.

### 31.5 LO QUE ESTA MEDIDA NO DICE

- **No dice que no se pueda cazar al árbitro.** Dice que **estos dos rasgos no lo hacen**: ni la tasa
  de `fondo + brazo` ni el ángulo del brazo. Con **2 árbitros** no se puede establecer ningún rasgo
  de todos modos; lo que se puede es DESCARTAR, y estos dos quedan descartados.
- **La mitad del criterio no se ha ejecutado nunca.** `fondo` necesita homografía y ninguno de los dos
  clips con árbitro la tiene. Para probar el criterio completo hace falta un clip con árbitro **y**
  homografía, y en 33 clips anotados no hay ninguno.
- **El 0,524 del árbitro es de 4 eventos en 7,6 s.** Con esa n, la cifra tiene un intervalo ancho y lo
  que se afirma no es su valor sino que **no es cero**, que es lo que el criterio necesitaba.
- **Los eventos de estos lados se emiten TODOS**, con la marca en `lado_valido` y
  `lado_no_valido_por`. Filtrarlos habría convertido el cero en «no medido», que es el fallo que esta
  sesión lleva corrigiendo en cuatro sitios distintos.
- **`accion` tampoco distingue**: el árbitro de `IdolizedFancyChickadee` está dentro de las 7 acciones
  del clip, que son de clip y no de lado, así que ese evento no tiene resolución por tirador y no
  puede servir de discriminador tal como está definido.

### 25.28 · La vía del árbitro por conducta: medida y RETIRADA, y la prueba además no era concluyente

La idea era que **el árbitro no esgrime**, y el criterio estaba congelado antes de medir: cero eventos
`fondo` y cero `brazo` en más de 4,0 s útiles → no es un tirador. Medido sobre los dos árbitros
anotados, **falla en los dos, y en uno al revés**:

| clip | lado | es | `fondo` | `brazo` | **tasa** |
|---|---:|---|---:|---:|---:|
| `PowerfulAllLarva` | 0 | **ÁRBITRO** | 0 | 4 | **0,524** |
| `PowerfulAllLarva` | 1 | tirador | 0 | 1 | 0,131 |
| `IdolizedFancyChickadee` | 1 | **ÁRBITRO** | 0 | 1 | **0,100** |
| `IdolizedFancyChickadee` | 0 | tirador | 0 | 2 | 0,200 |

El árbitro de `PowerfulAllLarva` da **0,524, por encima del p10 de los tiradores reales (0,493)** y
**cuatro veces la tasa de su propio rival**. Ninguno de los dos da cero, que era la predicción.

Lo que sí se midió bien fue el **confundido**, y salía limpio: de **24 tiradores** con eventos,
**ninguno** tenía tasa cero, con p10 0,493, p50 0,734 y el más flojo en 0,333 (4 eventos en 12,0 s).
O sea que el problema no es el falso positivo: **es que el positivo no existe**.

**Y hay un agravante que invalida la prueba aunque hubiera salido bien:** `fondo` es 0 **por
construcción** en los dos, porque **ninguno de los dos clips tiene homografía** y el criterio del a
fondo tiene el umbral en metros. Así que sólo se ejecutó la mitad `brazo`. Y en los 33 clips anotados
**no hay ninguno con árbitro Y homografía**, así que hoy la prueba **no se puede hacer bien**.

**Y la explicación alternativa también es falsa, medida.** Se pensó que el árbitro levanta el brazo
para señalizar y que el ángulo separaría. Al contrario: el brazo del árbitro apunta **hacia abajo más**
que el del tirador (p50 **−76,1°** contra **−36,0°**) y el porcentaje de cuadros con el brazo por
encima de 30° es **0,0 % en los dos árbitros**. Lo que les dispara el evento `brazo` es **el braceo de
andar**, porque el criterio del evento es «se alarga sobre su propia mediana» y andar también alarga.

**La idea era buena y se cae.** No se reabre hasta que exista un clip con árbitro **y** homografía. La
variante que se podría intentar entonces es `accion` por la vía sin homografía (columna `via_accion`),
pero eso es una medida nueva y no está hecha.

### 25.29 · La regla del punto 18 PASA en el reservado y se adopta: de 0,45 a 1,00 de cobertura sin un fallo

`accion` existe ya en **22 de 22 clips**, incluidos los 10 reservados, así que la segunda mitad de la
regla —`encarado + ev_accion`— **sí se ejecuta**. Medido **una sola vez**, contra `eventos.csv` con
`mtime_ns` **1789122455576544300** (650 filas, 22 clips), y con el criterio congelado en el punto 18 y
sin tocarlo.

| condición declarada antes de anotar los 20 | resultado en el RESERVADO |
|---|---|
| (i) acierto ≥ 0,90 | **1,000** ✓ |
| (ii) cobertura ≥ 0,50 de los 20 | **19 de 19** ✓ |
| (iii) más cobertura que `pie firme` solo, sin bajar su acierto | **12 → 19**, y 1,000 en los dos ✓ |

**Las tres se cumplen, así que la regla se adopta**, y el 28 de 28 de entrenamiento queda confirmado
en datos que no vio.

**Y lo que había que comprobar antes de mezclar: las dos vías de `accion` se comportan igual.**

| muestra | vía | n | REGLA_18 decide | acierto | `pie firme` decide | acierto |
|---|---|---:|---:|---:|---:|---:|
| RESERVADO | homografía | 8 | 8 | **1,000** | 5 | 1,000 |
| RESERVADO | **altos de caja** (sin homografía) | 11 | 11 | **1,000** | 7 | 1,000 |
| ENTRENAMIENTO | homografía | 20 | 20 | 1,000 | 9 | 1,000 |
| ENTRENAMIENTO | altos de caja | 1 | 1 | **0,000** | 0 | — |
| ENTRENAMIENTO | sin eventos | 19 | 8 | 1,000 | 8 | 1,000 |

La vía nueva, la de **altos de caja para los clips sin homografía**, da 11 de 11 en el reservado. No
degrada nada, y era lo que había que saber antes de juntar las dos columnas.

**El único fallo de la regla adoptada tiene causa conocida y no es de la regla.** Es
`PowerfulAllLarva l1`, y en ese clip **el rival del lado 1 es EL ÁRBITRO**: la puerta `encarado`
compara la nariz contra el centro de la caja del rival, así que está midiendo la orientación respecto
a un no-tirador. **Si el árbitro se detectara, ese error desaparecería**, lo que une los dos hilos de
esta sección.

#### Estado operativo, verificado en la ruta real de `mano_armada`

| muestra | no dudosos | acierto | todos | acierto | por qué decide |
|---|---:|---:|---:|---:|---|
| ENTRENAMIENTO | **29** (0,73) | 0,966 | 40 | 0,925 | 17 pie firme + 12 encarado+acción |
| **RESERVADO** | **19** (1,00) | **1,000** | 19 | **1,000** | 12 pie firme + 7 encarado+acción |

La cobertura de decisión fiable pasa de **17 a 29** en entrenamiento y de **12 a 19** en el reservado.

#### Y la dependencia sigue sin aceptarse: la acción entra como ARGUMENTO

`cono_lib` **no lee `eventos.csv`**, y no lo va a leer: ese fichero es de otro agente y ha cambiado
**seis veces en un día**. Lo que se ha hecho es lo que corresponde:

- `cono_lib.mano_armada(..., accion=None)` recibe una **máscara booleana `(T,)`** de cuadros dentro de
  una acción, igual que `vota` en `mano_armada_voto`. Sin ella la puerta no se aplica y la función
  devuelve lo de antes.
- `cono_lib.mascara_encarado(...)` calcula la otra mitad de la puerta con lo que ya tiene (`KP`, `SC`,
  `B`), sin dependencias.
- `mano_guante.mascara_accion(stem, T)` construye la máscara desde el CSV **para quien la necesite
  hoy**, y devuelve **`None`** cuando el clip no tiene eventos: `None` significa «no medido» y hace que
  la puerta no se aplique, mientras que una máscara de ceros significaría «ninguna acción», que es una
  afirmación distinta y falsa.
- y lo que de verdad toca a medio plazo: **precalcular esos tramos a un array dentro del NPZ de
  poses**, que es donde ya vive el resto de la procedencia (`origen`, `hueco`, `mask`).

#### Dos fallos de lectura, uno ajeno y uno mío, con el mismo patrón

El aviso era que `res.get(k, nan)` **no protege de nada cuando la clave existe y vale `None`**. Se
blindó `lee_eventos` **por valor y no por presencia de clave**, validando las cinco columnas que usa y
**contando** las filas que se saltan para que no desaparezcan en silencio. En esta versión del fichero
no se salta ninguna: `clip`, `lado`, `tipo`, `inicio` y `fin` vienen todas limpias, y `via_accion`
viene vacía en las 633 filas que no son `accion`, que es lo esperado.

Y de paso me cacé uno mío del mismo tipo: en `careo` tenía una variable `res` con la lista del
reservado y **tres bloques más abajo** otra `res` que se reasigna a la lista de resoluciones. Reventó
con `KeyError: '1280x720'` lejos del sitio donde se escribió. Arreglado copiando las dos listas con
nombre propio donde se definen. Es la cuarta vez hoy que alguien dice lo mismo de su propio código:
**tener el precedente escrito no impidió repetirlo.**

### Lo que 25.28-25.29 NO dicen

- **La vía del árbitro no está refutada, está sin poder probar.** Lo medido es que con el criterio
  congelado **no separa** y que la mitad `fondo` **no se ejecutó** por falta de homografía. Son dos
  cosas distintas y las dos hay que decirlas.
- **El 1,000 del reservado son 19 casos.** El intervalo inferior de 19 de 19 anda por 0,83. Y sumado a
  entrenamiento no se puede dar un total, porque las dos muestras miden cosas distintas (25.25).
- **La regla adopta una puerta que depende de un detector de otro agente.** Si `eventos.csv` cambia de
  criterio, la cobertura de la regla cambia con él, y la máscara pasa a valer otra cosa sin que nada
  avise. Por eso la dependencia es un argumento y no una lectura de fichero, y por eso lo que toca es
  meter los tramos en el NPZ.
- **La cobertura de 1,00 en el reservado es de 19 tiradores de 10 clips**, y depende de que los 22
  clips tengan `accion`. En los 19 tiradores de entrenamiento **sin** eventos la regla sigue decidiendo
  sólo 8, o sea que la cobertura real en el corpus depende de cuántos clips tengan eventos calculados.
- **Y el fallo de `PowerfulAllLarva l1` no está arreglado**, sólo explicado. Mientras el árbitro se
  entregue como tirador, la puerta `encarado` de su rival mide contra un no-tirador.

## 32. La primitiva de CONTACTOS: la fragmentación baja a 1,00 por zancada, y el hueco que queda es la FRONTERA

El usuario: *«¿se puede llegar a mejorar todavía más la precisión de lo que son marchas, romper etc?
Igual me pides que te diga dónde empieza y acaba una marcha... para que no detecte varias cosas en
medio de la misma acción»*.

Su última frase es el diagnóstico. Y el problema es **estructural, no de calibración**: todos mis
criterios son de la forma «la señal pasa de X», y **una señal con ruido cruza un umbral varias veces
dentro del mismo gesto**, así que cada cruce genera un evento. De ahí la fragmentación, y de ahí que
hubiera que añadir arbitraje, fusión y mínimos de duración: eran **parches sobre la primitiva
equivocada**.

### 32.1 LA PUERTA, medida antes de construir nada: las dos poblaciones existen

La primitiva propuesta es el CONTACTO del pie: un pie apoyado tiene velocidad ~0 en el sistema de la
pista y en vuelo no. Los eventos serían los CAMBIOS de estado, que son **instantes y no tramos**, así
que no se pueden solapar ni fragmentar por construcción.

Pero eso sólo vale si las dos poblaciones existen. Señal: módulo de la velocidad del punto más bajo
de cada pie en el cuadro **ESTABILIZADO**, normalizada por el alto de la caja. Es de imagen: no
necesita homografía (así que vale también en los 7 clips que no la tienen) ni escala métrica.

Medido sobre **32 series pie-tirador** de 8 clips, con los tres criterios declarados antes:

| criterio declarado | resultado |
|---|---|
| 1 · `eta` de Otsu >= 0,60 | **32 de 32** (p50 **0,781**, mínimo 0,668) |
| 2 · valle <= 0,50 del menor pico | **31 de 32** (p50 **0,06**: un valle profundo) |
| 3 · fracción de apoyo en 0,55-0,90 | 23 de 32 (p50 0,87) |
| **razón entre las medias de las dos poblaciones** | **11,3 x** |

**La primitiva se sostiene.** Los dos primeros criterios son de BIMODALIDAD y no dependen del umbral:
pasan casi en todas. Y hay que decir algo del tercero: **es una propiedad del CLASIFICADOR y no de la
bimodalidad**, así que funciona como calibración y no como prueba. Lo declaré como prueba y no lo es.

**Y no elegí el umbral para pasar mi propio test:** probé la alternativa de `K` veces el modo de
apoyo, y con `K` = 6 la fracción de apoyo cae en el rango declarado en **27 de 32** contra 23 de 32
con Otsu. No la adopté: elegir `K` mirando cuál cumple el criterio de aceptación es ajustar el umbral
al test. Otsu no tiene parámetro libre. La diferencia entre los dos umbrales mueve la **frontera** del
evento un par de cuadros, y eso es justo lo que la anotación del usuario va a medir.

### 32.2 LA FRAGMENTACIÓN, que es lo que el usuario pidió arreglar

Un VUELO es el tramo entre el despegue y el aterrizaje **del mismo pie**. Medido contra las 47
zancadas contadas a ojo en los tres tiradores:

| | valor |
|---|---:|
| **vuelos del mismo pie por zancada, mediana** | **1,00** |
| media | 0,96 |
| zancadas con exactamente 1 vuelo | **35 de 47** |
| zancadas partidas en 2 | 5 (**11 %**) |
| zancadas partidas en 3 o más | **0** |
| zancadas sin ningún vuelo | 7 |
| recall de zancadas | **85 %** (40 de 47) |

**La fragmentación queda en 1,00 de mediana y ninguna zancada se parte en más de dos.** Compárese con
la vía publicada, donde un mismo gesto producía marcha + fondo + marcha simultáneos y había **73
solapes** en los 11 clips.

**Y UN FALLO DE MI PROPIA MÉTRICA, que hay que contar porque casi me hace creer lo contrario:** la
primera medida daba **2,15 vuelos por zancada** y parecía que la primitiva no arreglaba nada. Estaba
comparando los vuelos de LOS DOS pies contra la zancada de UNO, y en un paso los dos pies vuelan en
secuencia con solape, así que 2 era el valor esperado por construcción. No era fragmentación del
detector: era un error de la métrica. Lo cazó preguntarme por qué el valor era tan exactamente 2.

El coste medido: **7 de 47 zancadas se quedan sin vuelo**, o sea el recall baja de 90 % a 85 %. Son
los traslados lentos, donde el pie nunca supera el umbral de apoyo. Es el mismo modo de fallo que
antes, sólo que ahora vive en el clasificador de contacto.

### 32.3 LO QUE NO SE HA HECHO, y el orden importa

**La primitiva NO se ha puesto a publicar todavía**, y es deliberado. La vía publicada sigue siendo
la del arbitraje (§30), y los vídeos y el CSV son coherentes con ella.

La razón es de método: la primitiva cambia **dónde empieza y acaba** cada evento, y el convenio de
frontera **no está fijado**. Cambiar el pipeline antes de fijarlo sería mover todos los inicios y
fines a una definición que nadie ha validado, y además haría que la anotación del usuario midiera
contra un blanco móvil. El orden correcto es: fijar el convenio, anotar, medir el error de frontera,
y **entonces** cambiar la primitiva con su cifra al lado.

### 32.4 Lo que hace falta del usuario, y ya está preparado

Hay un hueco real que ninguna de mis cifras cubre: lo medido es **detección** (recall 85-90 %) y
**clasificación** (5 de 6), pero **nadie ha medido si las FRONTERAS son correctas**. Un evento puede
existir, ser del tipo correcto, y empezar 8 cuadros tarde.

  convenio: `02_validacion/pista/CONVENIO_FRONTERAS.md`, escrito **antes** de que se anote
  tiras: `02_validacion/pista/anotar/accion_<clip>_l<lado>_c<cuadro>.png`, **21 acciones** de 3
  tiradores
  fichero: `02_validacion/pista/anotar/fronteras.csv`, con las columnas puestas y una fila por acción
  herramienta: `02_validacion/herramienta/prepara_anotacion.py`

**EL CONVENIO ADOPTADO, y hay dos defendibles:** la frontera es el **DESPEGUE** (el cuadro en que el
pie ya no toca) y el fin el **ATERRIZAJE**; no «cuando empieza a moverse», que es 2-4 cuadros antes.
Se adopta el despegue porque es un **instante observable** (se ve si la suela toca o no), porque es lo
que mide el detector — si se anotara uno y se detectara el otro, la diferencia sistemática se leería
como error — y porque es la convención de biomecánica. **Y no es gratis:** con este convenio la marcha
«empieza» después de que el tirador haya iniciado el gesto, así que para el instante de INTENCIÓN
llega tarde por construcción.

Cada tira lleva **todos los cuadros de la acción, numerados en grande**, con 4 de margen a cada lado
(sin margen no se puede juzgar que un cuadro es el despegue, porque hace falta ver el contacto de
antes), los dos pies en colores distintos, y el fotograma **estabilizado** para que el suelo no se
mueva. El candidato del detector va escrito pero **en su propia línea y con un «no te fíes,
corrígelo»**, y en el CSV va en **columnas aparte** de las que se rellenan: ponerlo en la misma
casilla sesgaría la anotación hacia lo ya detectado, que es la forma más fácil de fabricar un acuerdo
falso.

### 32.6 El entregable pasa a INGLES, y la regla de que se traduce y que no

El usuario: *«pon todo el readme en inglés por favor... y las imágenes, los textos que tiene, también
inglés, así que tendrá que volverlo a generar. Quiero que sean inglés siempre para que lo entienda
todo el mundo»*. Va camino de un repo público.

**LA REGLA, declarada: si lo ve alguien de fuera, en inglés; si es interno, se queda en español.**

| va en INGLÉS | se queda en ESPAÑOL |
|---|---|
| todo el texto **dentro de** las figuras `pista_*.png` | los **docstrings** del código |
| todo el texto dentro de los vídeos: contador, destello, rótulos del mapa, avisos en rojo | esta sección y el resto de `00_EMPIEZA_AQUI.md` |
| las 21 tiras de `anotar/`, que llevan el convenio escrito en la propia imagen | los comentarios internos |
| `README.md`, `STEP_COUNTING_CONVENTION.md`, `BOUNDARY_CONVENTION.md` | |
| los rótulos de `verifica/` (en el código; las imágenes se regeneran cuando toque) | |

**Y LO QUE NO SE TOCA NI EN INGLÉS NI EN NADA: los nombres de columna de los CSV.** Cambiar
`cuadro_inicio` por `frame_start` rompería el código de otros agentes y los scripts de subida. Lo que
sí está traducido es **la tabla del README que explica qué significa cada columna**, que es lo que un
lector de fuera necesita. El valor de `via` (`camara`, `zona`, `rayas`) tampoco se cambia en el dato:
hay un diccionario `VIA_EN` que sólo traduce su PRESENTACIÓN.

**TERMINOLOGÍA DE ESGRIMA, que es donde es fácil estropearlo:** `marcha` → **march**, `romper` →
**retreat**, `fondo` → **lunge**, `alto` → **halt**, `bote` → **balestra**, `guardia` → **on-guard**.
Y el aviso que importa: **`alto` NO se traduce como `parry`**. En inglés `parry` es la parada de
esgrima, que **no está implementada**, y es exactamente la confusión que se acababa de arreglar en
español (§28.4). Traducirlo mal habría reintroducido el mismo defecto en el idioma nuevo.

Los ficheros `.md` en español se han movido a `diagnostico/`, que ya es la carpeta de lo interno, en
vez de borrarlos. Y los MP4 pasan a llamarse `trace_<clip>.mp4` y `trace_<clip>_slow.mp4`.

El README lleva arriba **las tres cifras que un lector de fuera necesita para no malinterpretar
nada**: recall del **90 %**, matriz de confusión **5 de 6 con `march -> lunge` = 0**, y que la escala
métrica sólo pasa el control físico en el **2,6 % de los clips**. Esa tercera es la que evita que
alguien lea los metros como si fueran una medida calibrada.

### 32.5 LO QUE ESTA MEDIDA NO DICE

- **No dice que la primitiva sea mejor en todo.** Es mejor en fragmentación (1,00 contra 73 solapes) y
  **peor en recall** (85 % contra 90 %). Las dos cifras son del mismo conjunto de 47 zancadas.
- **La bimodalidad es de 32 series de 8 clips**, no del corpus. Y el criterio 3 lo pasan 23 de 32: en
  9 series la partición llama apoyo a casi todo, con fracciones de hasta 0,99.
- **La frontera no está medida, sólo preparada.** El sesgo entre el umbral de Otsu y uno robusto es de
  un par de cuadros y **no se sabe cuál de los dos está más cerca de la verdad** hasta que se anote.
- **No dice que los tipos salgan mejor con la primitiva nueva.** La decisión de tipo dentro del ciclo
  (marcha / romper / fondo por la propiedad binaria del pie de atrás) **no se ha medido todavía** con
  la primitiva: lo medido es la segmentación. La matriz de confusión vigente sigue siendo la de §30,
  5 de 6 con `marcha -> fondo` = 0.
- **Y sobre entrenar:** con 47 zancadas y 21 acciones por anotar no se entrena, se evalúa. Lo que
  haría posible entrenar algún día es que `CONVENIO_CONTEO_PASOS.md` y `CONVENIO_FRONTERAS.md`
  existan, porque son lo que permite etiquetar a escala sin que cada pasada mida cosas distintas.

---

## 33. Las LUCES en los 21 tipos de marcador: 5 familias, y el detector que ya existía es mejor que el mío

El usuario pidió: «pilla un clip por torneo, saca de todos las luces, porque el marcador cambiará y
el algoritmo tiene que funcionar en todos los tipos de marcador». Se ha hecho, y lo primero que salió
es que **las luces ya estaban extraídas**: los 10.849 NPZ de `localrun/runs_extra*` traen
`lights (T,4)`, `light_observed (T,4)` y un `meta_json` con el instante de cada canal en
milisegundos, cuál encendió primero y la diferencia. Así que el encargo cambió de «construir un
detector» a «medir si el que hay generaliza», que es mucho más barato y era la pregunta de verdad.

### 33.1 El censo, mirado: 21 torneos son 5 familias de marcador

Se sacaron dos fotogramas por clip —uno con la luz encendida en `t_touch` y uno dos segundos antes— y
se miraron los 21 rótulos en `02_validacion/muestras/luces/censo/mosaico_rotulos_*.png`.

| familia | qué es y dónde está la luz | sedes | clips |
|---|---|---:|---:|
| **A · FIE clásica** | rótulo azul en losange abajo, dos placas de nombre, reloj central con pestaña de periodo, banderas circulares. La luz es una **barra** que rellena el borde inferior de la placa, en `cy` 0,92 y `cx` 0,277 (roja) / 0,725 (verde) | **16** | 20.427 |
| **B · banner ruso plano** | banner azul oscuro, caja de reloj separada y centrada debajo. La luz es un **disco** a un lado del reloj (Kazán verde en `cx` 0,675, San Petersburgo rojo en `cx` 0,33) | 2 | 1.902 |
| **C · Tokio 2022** | placas rectangulares gris oscuro con pestaña de ronda. La luz es una **barra fina** en el borde inferior de la placa | 1 | 1.433 |
| **D · OBS olímpico** | banner claro. La luz son dos **puntos** que flanquean la caja del periodo, y además la placa del que puntúa **se tiñe de rosa** | 1 | 1.010 |
| **E · París 2022** | rótulo pequeño y centrado con «Tableau 64». La luz es **UN SOLO cuadradito** entre los dos tanteos, que se pone verde o rojo: no hay lado | 1 | 367 |

**El problema pasa de 21 casos a 5, y una sola familia cubre el 76,2 % de las sedes y el 81,2 % de
los clips.** El usuario tenía razón con «habrá muchos que se repiten».

**Los confundidos que vi, todos reales y todos dentro del rótulo o pegados a él:** la bandera de
Italia tiene una franja **verde** a 30 px de la barra verde; la de Japón un disco **rojo**; las de
Francia, Rusia, Dinamarca, Austria y Polonia tienen **rojo**. La pista es **roja** en Belgrado, Doha,
Serbia y Turín 2023, **azul** en Bonn, **turquesa** en Katowice. El reloj de Turín 2023 lleva cuñas
**rojo oscuro** decorativas. En Tokio 2020 hay un gráfico rojo que **barre** la banda. Y los
**dígitos del reloj son blancos y cambian cada segundo**, justo donde está el indicador de luz
blanca. Dos de esos confundidos me costaron un falso positivo cada uno (ver 32.4).

### 33.2 La cifra de generalización del detector que ya existía, sobre 25.139 clips

Los NPZ solo contienen `status = "ok"`, y en ellos `n_valid_lights` nunca es 0. Eso no es que el
detector acierte siempre: **los clips donde no vio ninguna luz se cayeron antes con
`status = "no_lights"`** y están en los `index.jsonl`, con el campo `tournament_id` al lado. Nadie
había leído esa cifra. Es el recall a nivel de clip y va por sede:

| sede | intentados | sin ninguna luz | % | veredicto |
|---|---:|---:|---:|---|
| **tokyofoil2022** | 1.433 | 527 | **36,78** | **ROTA** |
| parisfoil2022 | 367 | 49 | 13,35 | regular |
| stpetersburgfoil2019 | 944 | 120 | 12,71 | regular |
| kazanfoil2020 | 958 | 78 | 8,14 | regular |
| tokyoolympics2020 | 1.010 | 75 | 7,43 | regular |
| anaheimfoil2018 | 2.444 | 173 | 7,08 | regular |
| *(11 sedes entre 5,2 y 7,1 %)* | | | | regular |
| belgradefoil2022 | 1.275 | 59 | 4,63 | SANA |
| incheonfoil2022 | 1.590 | 72 | 4,53 | SANA |
| bonnfoil2022 | 1.094 | 44 | 4,04 | SANA |
| turinfoil2023 | 794 | 26 | 3,27 | SANA |
| tbilisifoil2023 | 222 | 5 | 2,25 | SANA |
| **GLOBAL** | **25.139** | **2.038** | **8,11** | 1 rota, 15 regulares, 5 sanas |

El criterio (sana < 5 %, rota > 20 %) se declaró antes de ver la tabla y no se movió.

**Y la tabla tiene una estructura, no es ruido.** Hay **dos implementaciones** en producción: `fie`
(8.505 clips, 78,4 %) y `generic` (2.339, 21,6 %). Cinco sedes usan el genérico en el **100 %** de
sus clips, o sea que el detector de overlay FIE **no se aplica nunca** ahí: son kazanfoil2020,
parisfoil2022, stpetersburgfoil2019, tokyofoil2022 y tokyoolympics2020 — exactamente las familias B,
C, D y E del censo. En esas cinco:

| | clips | sin luz | % |
|---|---:|---:|---:|
| las 5 sedes donde el FIE no se aplica | 4.702 | 849 | **18,06** |
| las otras 16 | 20.437 | 1.189 | **5,84** |

**Un factor 3,1×.** Y en `incheonfoil2022` el 97,9 % de los clips y en `worldchamps2022` el 95,0 %
llevan el aviso `cajas_por_geometria_prior`, o sea que las cajas del rótulo **no se detectaron** y se
usó la geometría canónica de reserva.

> **CORRECCIÓN (sección 35).** Aquí decía que el rótulo de `worldchamps2022` tiene la barra en `cy`
> 0,939-0,967 y que el detector FIE «está mirando 0,03 H por encima de la luz». **Eso era de UN clip
> y está mal.** Midiendo la posición de la barra en 12 clips por sede, la mediana de `worldchamps2022`
> es `fy` 0,9115 con `fh` 0,0184, que cae **entera** dentro de la franja 0,910-0,932: solape
> **1,000**. Y lo mismo en 28 de las 30 parejas sede-canal de la familia A. Así que usar la geometría
> de reserva **no es un acierto por la razón equivocada**: la geometría de reserva es correcta y lo
> que falla es solo el buscador de cajas. Ver 35.2.

### 33.3 El reparto de una luz / dos luces, que es lo que condiciona la idea del contraataque

Sobre los 25.139 clips intentados: **71,81 % una luz, 19,72 % dos luces, 8,11 % ninguna**. Y por sede
va de **5,93 %** de dos luces en tokyofoil2022 a **25,05 %** en tokyoolympics2020, un factor 4,2×,
que es *sospechosamente* el mismo orden que la tasa de fallo. Está en
`02_validacion/muestras/luces/reparto_luces_por_sede.csv`.

Esto encaja con el techo del 73,6 %: si solo 4.958 clips tienen dos luces, los **960 clips donde el
que encendió primero no gana** son el 19,4 % de ellos. La idea del usuario —separar los contraataques
de los clips de una sola luz y usar el resto como si tuvieran dos— multiplicaría por **3,6** el
corpus con orden de luces utilizable (18.053 frente a 4.958). Pero el 5,93 % de tokyofoil2022 dice
que **el reparto una/dos luces está contaminado por el detector**, así que antes de construir nada
sobre esa idea hay que arreglar ese detector, no calcularla.

### 33.4 Verdad de referencia a ojo, y el acierto de los dos detectores

No existía. He anotado yo los 21 clips mirando fotogramas, en
`02_validacion/muestras/luces/verdad_luces_ojo.csv`, con su `README.md` (en inglés, porque el
repositorio va a ser público) que dice **que lo he anotado
un agente y no un árbitro**, y que distingue las dos procedencias del cuadro de encendido: **5 sedes
con medida independiente** (tira de fotogramas leída antes de mirar ningún detector) y **16 con
verificación** (la luz está apagada en los cuadros −10 y −4 y encendida en los +2 y +8, figuras
`encendido_*.png`), que acota a 6 cuadros pero **no es independiente**.

Salen **16 clips de una luz, 4 de dos, y 1 con luz blanca** (serbiafoil2022: verde derecha + blanca
izquierda, confirmada a ojo y coincidente con el NPZ viejo).

| | ajuste (familia A, 16 sedes) | prueba (familias B-E, 5 sedes) |
|---|---:|---:|
| **detector viejo**, conjunto (canal, lado) exacto | **15/16 = 93,8 %** | **4/5 = 80,0 %** |
| **`luces.py`** (el mío), conjunto exacto | **14/16 = 87,5 %** | **5/5 = 100,0 %** |
| `luces.py`, solo canales de color | 15/16 = 93,8 % | 5/5 = 100,0 % |
| `luces.py`, cuadro de encendido en tolerancia | 15/16 | 4/5 |

Se ajustó **solo en la familia A** y se probó en **4 familias y 5 sedes**, más de las 2 que pedía el
encargo. Los cinco arreglos que llevó el detector salieron todos de sedes de la familia A
(`anaheimfoil2018`, `qatarfoil2021`, `turinfoil2023`, `serbiafoil2022`, `belgradefoil2022`). La caída
de ajuste a prueba es **−12,5 puntos** (la prueba sale **mejor**), y el criterio declarado era «no
más de 20 puntos peor»: **PASA**.

**Los dos fallos del detector viejo, que la anotación a ojo encontró:**

- **`parisfoil2022`**: dice **dos** luces (roja izquierda en el cuadro 221, verde derecha en el 231).
  Hay **una**: el cuadradito central se pone verde en el 231 y en el 221 no hay nada. La familia E no
  tiene lado, y el detector se inventó uno.
- **`turinfoil2023`**: dice **roja izquierda en el cuadro 291**. La luz real es **verde**, a la
  **derecha**, encendida del cuadro ~146 al ~230: cuando el detector la sitúa, llevaba 60 cuadros
  apagada. Color, lado e instante mal, **5,7 s tarde**, con `n_valid_lights = 1` y `light_flags`
  vacío. Un clip que el pipeline daba por bueno.

**Los cuatro fallos del mío:** dos falsos positivos (`belgradefoil2022` toma la **pista roja** vista
donde el rótulo es transparente, 3.015 px; `bonnfoil2022` toma los **dígitos blancos del reloj** al
cambiar) y dos instantes fuera de tolerancia (`belgradefoil2022` 10 cuadros, `stpetersburgfoil2019`
**41 cuadros = 1,37 s**, porque su indicador rojo **se desplaza** 0,015 W en 70 cuadros y un detector
de píxel fijo lo ve llegar tarde).

### 33.5 La muestra B, independiente, tumba el 21/21: mi detector se queda en el 71,4 %

Un acierto medido donde se ajustó no es una medida. Se sorteó un **segundo clip por sede** con otra
semilla (20260911), 21 clips que no había visto nadie. Criterio declarado antes: encontrar al menos
una luz de color en ≥ 90 % y **ninguna sede a cero**.

| | muestra A (ajuste) | muestra B (independiente) |
|---|---:|---:|
| `luces.py` encuentra alguna luz de color | 21/21 | **15/21 = 71,4 %** |
| sedes a cero | 0 | **6** |
| **veredicto** | — | **NO PASA** |

Las seis: belgradefoil2022, bonnfoil2019, katowicefoil2019, qatarfoil2021, tokyofoil2022,
tokyoolympics2020. Y **en 6 de los 8 fallos la luz no está entre las regiones candidatas con ningún
umbral**, así que el límite es el criterio de región, no el de nivel.

**Medida la causa en tres de ellos**, en la región donde el detector viejo sitúa la luz:

| clip | base | pico | salto |
|---|---:|---:|---:|
| `HardtofindBitesizedBactrian` (Belgrado) | 62 | 90 | **28** |
| `PettyCelebratedGyrfalcon` (Katowice) | 90 | 96 | **11** |
| `FalseBlissfulCaimanlizard` (Doha) | 76 | 90 | **42** |

frente a saltos de **146 a 268** en la muestra A. **El salto de la misma luz, en la misma familia de
marcador, varía un factor 10 entre clips**, porque el rótulo FIE es **semitransparente**: sobre una
pista verde o turquesa la placa ya sale verdosa de base (base 62-90) y la barra solo añade decenas de
niveles; sobre una pista roja el contraste es enorme. Ésa es la respuesta técnica a la pregunta del
usuario, y no es el diseño del rótulo: es su **transparencia**.

**El arreglo evidente no funcionó, y se retira con su medida** (`USA_UMBRAL_ABSOLUTO = True` en
`luces.py`, con el código del otro camino puesto y apagado, como `USA_CRUDAS`). Comparar el salto con
el ruido propio del píxel (`salto ≥ max(20, 6 × (p60 − p20))`, sin condición de `base`) sale **peor
en las dos muestras**: 15/21 en A y 14/21 en B, y rompe dos sedes que el absoluto acertaba.

### 33.6 La comparación honesta, y qué hay que hacer

| | cobertura medida | muestra |
|---|---:|---|
| detector viejo, encuentra alguna luz | **91,89 %** | **25.139 clips**, todas las sedes |
| `luces.py`, encuentra alguna luz de color | **71,4 %** | 21 clips, uno por sede |

**El detector que ya existe es unos 20 puntos mejor en cobertura que el mío**, y lo es sobre una
muestra 1.200 veces mayor. **Aviso de sesgo declarado:** la muestra B se sorteó entre clips **que
tienen NPZ**, o sea clips donde el detector viejo ya había acertado; su 21/21 ahí está garantizado
por construcción y **no es una medida**. La cifra limpia del viejo es su `no_lights`: 91,89 %.

Así que la recomendación es la contraria a la que yo iba a dar: **auditar y arreglar el detector que
hay, no sustituirlo.** Por este orden, y cada punto con su cifra al lado:

1. **`tokyofoil2022`**: pierde el 36,78 % de 1.433 clips y solo el 5,93 % le salen con dos luces. Es
   la familia C, una sola sede, y su luz es una barra fina en `cy` 0,871-0,891. Arreglar esta sede
   sola recupera del orden de **527 clips**.
2. **Las familias B, D y E** (4.702 clips, 18,06 % de pérdida): el genérico las cubre a medias.
   París tiene un indicador **central sin lado** y el viejo se inventa el lado: eso son falsos
   positivos de «dos luces», que es justo lo que contamina el atajo del orden.
3. **`worldchamps2022` e `incheonfoil2022`** (4.169 clips): funcionan por el respaldo de geometría,
   no por detección. Basta corregir la `y` de la franja para que dejen de depender de la suerte.

> **CORRECCIÓN (sección 35).** Los tres puntos de esta lista **se han medido y dos de los tres son
> falsos.** El punto 1 no vale: de los 527 clips que `tokyofoil2022` pierde, solo el **10,0 %**
> tiene luz en la ROI aprendida de esa misma sede, así que **no hay 527 clips recuperables, hay ~53**.
> El punto 3 tampoco: el solape de la franja con la luz es **1,000**, no hay nada que corregir. El
> punto 2 sobrevive a medias. Y el sitio con más clips recuperables no es ninguna de esas sedes, es
> **la familia A** (~476 de 1.189). Ver 35.1.

Y lo que mi `luces.py` sí aporta, medido: **encuentra dos errores del viejo en 21 clips** (París y
Turín 2023), y su criterio de **anillo** (`ring_std` y contraste de color del entorno) separa la
pista roja de una luz roja, que es el confundido que el `ring_std` solo no podía.

### 33.7 El AUDIO del zumbador: la idea es buena y falla las dos puertas

El aparato pita al encender, y ese pitido es del aparato **físico**, así que sería inmune al diseño
del rótulo. Criterio declarado antes de medir: el audio es la vía principal solo si **(a)** ≥ 90 % de
los clips tienen pista y **(b)** en ≥ 90 % de ésos hay un tono de ≥ 8 dB de prominencia que supere en
≥ 6 dB a una ventana de control 3 s antes.

| | medido | criterio | |
|---|---:|---:|---|
| **(a)** clips con pista de audio (`ffprobe` sobre el mp4) | **11/21 = 52,4 %** | ≥ 90 % | **NO PASA** |
| (a) en el corpus (`has_audio` en 10.840 NPZ) | **50,6 %**, y **10 de 21 sedes al 0 %** | | |
| **(b)** tono ≥ 8 dB con margen ≥ 6 dB sobre el control | **4/11 = 36,4 %** | ≥ 90 % | **NO PASA** |
| TNR mediana en la ventana del encendido | 19,9 dB | | |
| TNR mediana en la **ventana de control**, 3 s antes | **15,3 dB** | | |
| frecuencia del pico | **2.508 a 3.480 Hz** | | |

La fila que mata la idea es la del control: **19,9 dB contra 15,3 dB**. El tono no es específico del
instante de la luz. Y hay dos clips con **margen negativo** —`turinfoil2023` −11,7 dB y
`worldchamps2022` −10,7 dB—, o sea que el control tiene un tono **más fuerte** que el encendido: en
un pabellón hay **varias pistas a la vez** y todos los aparatos pitan. Esto coincide con lo que el
propio proyecto ya tenía medido en `fencingly_v2/audio.py` sin que nadie lo hubiera leído: el tono
sirve para **corroborar** `t_touch` con σ ≈ 0,2 s, no para fijar el cuadro, y su frecuencia depende
del aparato (3,2 / 4,0 / 4,4 kHz vistos) y es **inaudible en Bonn, Tauber y Heidenheim**.

**Lo que sí sigue en pie del razonamiento del audio**: si alguna vez se usa, sería **el audio para el
instante y el rótulo para el color**, porque el pitido es uno y no tiene lado. Pero con audio en la
mitad del corpus y sin especificidad temporal, hoy no es ni la vía principal ni la de respaldo.

### Lo que 33 NO dice

- **No dice que mi detector sirva.** Dice que en 21 clips de ajuste da 14/16 y 5/5 y que en 21 clips
  independientes **encuentra luz en 15 de 21**, seis sedes a cero, y que **NO PASA** su propio
  criterio del 90 %. La cifra buena de la muestra A es de la muestra donde se ajustó.
- **No dice que el detector viejo acierte el 91,89 %.** Ese número es **recall a nivel de clip**:
  cuántas veces ve algo donde hay algo. La **precisión** no se mide con él, y los dos fallos que
  encontré (París y Turín 2023) tenían `status = "ok"`, `light_flags` vacío y estaban mal. Con 2
  fallos en 21 clips la tasa de error de precisión queda entre el 1 % y el 30 % al 95 %: **no está
  medida**.
- **La verdad de referencia la he anotado yo, no un árbitro**, y en 16 de las 21 sedes el cuadro de
  encendido es una **verificación** de lo que dijo mi detector (apagada en −4, encendida en +2), no
  una medida independiente. Con esas 16 **no se puede** descartar un sesgo sistemático menor que 6
  cuadros. Las 5 medidas aparte son las que encontraron el único error grande.
- **El orden entre dos luces separadas por menos de 2 cuadros no es resoluble** con este muestreo
  (`PASO = 2`). En `tokyoolympics2020`, a 49,985 fps, el viejo da `dt = 40 ms` y yo doy 0: ninguno de
  los dos resuelve un `dt` por debajo de 80 ms, y el bloqueo del aparato son 300 ms. Para la idea del
  contraataque eso puede no bastar, y habría que bajar a `PASO = 1`.
- **Las 5 familias son las que hay en ESTAS 21 sedes.** El corpus tiene 21 sedes y el mundo tiene
  más; el censo no es una taxonomía de la producción televisiva de esgrima.
- **El 19,72 % de clips con dos luces no es la tasa real de dos luces**, es la que da el detector
  viejo. Su variación por sede de 5,93 % a 25,05 % es del mismo orden que su tasa de fallo, así que
  parte de esa variación es el detector y no la esgrima. Cuánta parte **no está medido**.
- **el criterio del paso 1 NO pasó y se deja escrito.** Localizar la luz con un mapa de estabilidad
  temporal más el cambio entre apagado y encendido dejaba la región candidata mayor en la mitad
  inferior del cuadro en **15 de 21 sedes**, y el criterio declarado pedía 18. La causa medida: con
  paneo de cámara o un corte de plano entre los dos fotogramas, lo que más cambia es la escena. Se
  retiró como localizador único y el censo se hizo mirando.
- **No he tocado los NPZ ni nada de `localrun/`**, que es de solo lectura. Todo lo de aquí se lee de
  ahí y se escribe en `01_extraccion/codigo/f1_`…`f10_`, `luces.py`,
  `02_validacion/herramienta/verifica_luces.py` y `02_validacion/muestras/luces/`.

---

## 34. El falso negativo que vio el usuario: es del ALFA, no del ajuste de recta

El usuario, mirando el campo de `AdorableFittingAsiansmallclawedotter` c64: *"el de la izquierda la
clava prácticamente perfecto, pero el de la derecha no la detecta, y si se ve perfectamente, un poco
doblada"*. **Tiene razón, y comprobado en el dato**: en ese cuadro el lado 0 tiene `campo_origen = 0`
y el lado 1 tiene `1`.

**Y el caso vale porque rompe la explicación que el proyecto daba al 3 % de hojas vistas.** Esa
explicación era «la hoja casi no está»: desenfoque en la ventana de acción y sub-píxel a 720p. Aquí
es **1080p**, **el usuario la ve** y describe la flexión, y **la del otro tirador SÍ se detectó en el
mismo cuadro**. Luego no es el cuadro, ni el desenfoque global, ni la compresión: **es esa hoja**. O
sea que **al menos parte del 97 % es recall del detector y no ausencia de señal**, y eso cambia el
techo del problema. Código en `e28_falso_negativo.py`, figura en
`02_validacion/muestras/cono_arreglado/falso_negativo_AdorableFittingAsiansmal_c0064.png`.

### 34.1 Los dos fallos posibles, y el criterio para separarlos

| | qué falla | cómo se arreglaría |
|---|---|---|
| **F-ALFA** | `matting` no marca la hoja | la capa densa: umbral, resolución de entrada, otro modelo |
| **F-FORMA** | el alfa la marca y la descarta el ajuste | ajustar una curva o dos segmentos en vez de una recta |

**Criterio declarado.** `t` es el p99,5 del alfa fuera del cuerpo, así que **por construcción el
0,5 % de esos píxeles pasa el umbral siempre, haya hoja o no**. Luego:

> esperados por la cola = 0,005 × (área del disco fuera del cuerpo).
> La máscara lleva señal si los observados son **≥ 3×** los esperados.

**Y el primer criterio que declaré estaba MAL ESPECIFICADO, y va dicho.** Era «¿hay ≥ 20 px sobre el
umbral en el corredor entre las dos manos?», suponiendo que en un cruce de hierros la hoja de cada
uno va hacia la mano del otro. Dio 0 px — y mirando la figura resulta que **la hoja del lado 1 no va
hacia el rival: va hacia arriba y a la izquierda, por encima de él**. La suposición era falsa, y
encaja con lo que 24.9 ya había medido: el eje del arma **no apunta al rival**. El criterio de la cola
lo sustituye porque **no supone ninguna dirección**.

### 34.2 El veredicto: F-ALFA, y con un factor 121

| | lado 0 (**detecta**) | lado 1 (**el fallo**) |
|---|---:|---:|
| disco fuera del cuerpo | 95.794 px | 78.510 px |
| esperados por la cola del p99,5 | 479 | 393 |
| **observados** | **1.014** | **153** |
| **razón observados / cola** | **2,12×** | **0,39×** |
| **alfa máximo en el disco** | **0,1453** | **0,0012** |
| distancia al cuerpo de los px marcados, p50 / máx | **40 / 109 px** | **2 / 5 px** |

**El lado 1 tiene MENOS píxeles sobre el umbral que los que da el ruido de fondo** (0,39×), y su alfa
máximo es **0,0012**, o sea el propio umbral. Los 153 píxeles están a **2 px medianos del cuerpo**:
son el fleco de la silueta, no una hoja. En el lado 0, en cambio, el alfa llega a **0,1453** —
**121× más** — y su rastro se extiende hasta **109 px del cuerpo**, que es una hoja de verdad. Se ve
de un golpe en la figura: en el alfa del lado 0 sale un trazo brillante desde la mano; en el del lado
1 no hay nada donde está la hoja.

**Veredicto: F-ALFA.** El fallo es de la capa densa y **no del ajuste de recta**.

### 34.3 La hipótesis del usuario es una buena descripción y NO es la causa

El usuario dice que la hoja está «un poco doblada», y es cierto: la flexión de una hoja de florete en
tensión es real y se ve. Pero **no puede ser la causa de este fallo, porque el fallo es anterior al
ajuste**: si el alfa no marca la hoja, da igual buscar una recta, una parábola o dos segmentos.

Y medido en el rastro que sí se marca, la curvatura tampoco limita el ajuste:

| rastro del lado 0 (1.014 px) | |
|---|---:|
| largo | 109 px |
| residuo de **recta** | 3,75 px |
| residuo de **parábola** | 3,64 px |
| **mejora de la parábola** | **1,03×** |
| flecha en el centro | 3,7 px = **3,4 % del largo** |

**El rastro es prácticamente recto.** Así que la vía «ajustar una curva o dos segmentos» **no habría
recuperado este caso ni habría alargado el que sí se detecta**, y por eso no se implementa: está
medido que no es ahí.

### 34.4 Dos explicaciones más, descartadas con su cifra

**El fondo oscuro, NO.** La hoja del lado 1 está sobre el negro de la grada, y parecía la explicación
obvia. Pero el fondo del disco es **igual de oscuro en los dos lados**:

| | gris p50 | fracción por debajo de 60 |
|---|---:|---:|
| lado 0 (detecta) | **17** | 81 % |
| lado 1 (falla) | **18** | 86 % |

**Bajar el umbral, tampoco.** Con 0,39× la máscara del lado 1 tiene **menos** píxeles sobre el umbral
que el ruido de fondo, y el alfa máximo del disco es el propio umbral. No hay señal débil que
recuperar: bajar el percentil sólo añadiría fondo.

**Así que el mecanismo de este fallo concreto queda SIN IDENTIFICAR.** Lo que sí queda establecido, y
es la mitad accionable, es **dónde está**: en el alfa, no en el ajuste. Dos hojas del mismo cuadro,
sobre el mismo fondo, con tiradores de talla parecida, y `matting` marca una y no la otra.

### 34.5 Las tasas, de la muestra buena y no de este clip

Este clip aportaba **24 de las 31 hojas** de la medida vieja y **no está en
`corpus_etiquetado.txt`**: sirve como caso de estudio y **no para dar una tasa**. Las tasas salen de
la recogida de 29.8:

| resolución | régimen | recortes | rectas | tasa |
|---|---|---:|---:|---:|
| **1080p** | repartido por el clip (`e25`, 6 asaltos del corpus) | 633 | 111 | **17,5 %** |
| 1080p | repartido por el clip (`e5`, este clip, fuera del corpus) | 61 | 26 | 42,6 % |
| **1080p** | **ventana de ACCIÓN** (`e10`) | 198 | 6 | **3,0 %** |
| 720p | cualquiera | — | — | el ángulo con el eje da 60,9°, **peor que el azar**: no es la hoja |

**No hay tasa «después»**, y la razón es el resultado: el arreglo que la hipótesis implicaba —ajustar
una curva— está medido y descartado en 34.3, así que implementarlo para publicar un «antes y después»
habría sido trabajo que ya sabía inútil.

### LO QUE ESTA MEDIDA NO DICE

- **No dice por qué falla.** Dice dónde: en el alfa. El mecanismo que distingue las dos hojas del
  mismo cuadro **no está identificado**, y las tres explicaciones que tenía a mano —desenfoque
  (29.6/`e23` M4), fondo oscuro y umbral— están descartadas cada una con su cifra.
- **Es UN cuadro y DOS hojas.** No es una tasa de F-ALFA contra F-FORMA. Para saber qué fracción del
  97 % es recall del detector habría que repetir este diagnóstico sobre una muestra de cuadros donde
  un humano marque si la hoja se ve, y **eso no está hecho**: requiere anotación a ojo, que es
  justamente lo que la recogida de 29.8 evitaba.
- **No dice que la hoja del lado 1 sea suya.** Se ve un arma que sale de la zona de su mano, y en un
  cruce la del rival también pasa por ahí. Lo que se mide es que **hay un arma visible que el alfa no
  marca**, y eso vale igual para el caso.
- **El vértice del lado 1 no cae exactamente sobre su mano** en el alfa: está unos píxeles fuera de
  la silueta. No invalida el veredicto (el disco de radio R la cubre de sobra), pero es otro
  recordatorio de que el vértice viene de `mano_armada` y arrastra su error.
- **La curvatura se midió en UN rastro.** Que ése sea recto (mejora 1,03×) no dice que todas las 153
  hojas lo sean. Medir la curvatura de todas exige volver a pasar `matting` para tener las máscaras,
  porque el JSON sólo guarda los dos extremos de la recta, y no se ha hecho.
- **No cambia ninguna cifra del cono.** `peso_sector` sigue siendo el operativo, la envolvente sigue
  en ±33,0° (29.4) y el campo sigue centrándose en la hoja sólo cuando `origen = 0`. Lo único que
  cambia es que ahora se sabe que parte del 97 % de `origen = 1` **es recuperable en principio**, y
  que la palanca está en la capa densa.

---

## 35. El arreglo dirigido de las luces: dos de los tres puntos del plan eran falsos, y el que funciona da 90,5 %

La sección 33 acabó con una lista de tres arreglos por orden de clips recuperados. Se han medido los
tres **antes de tocar código**, y **dos son falsos**. Esta sección es la corrección, y trae además el
arreglo que sí funciona.

### 35.1 Los 527 clips de Tokio no existen: la mayoría de los clips «perdidos» no tienen luz

La cadena de razonamiento que había que romper era: `tokyofoil2022` pierde el 36,78 % de sus 1.433
clips → son ~527 clips → arreglar la familia C los recupera. **La primera medida ya no encajó.**

`f11_recupera_no_lights.py` corrió mi detector sobre los clips que el viejo perdió, con un **control**
de la familia A que es lo que hace que la cifra signifique algo:

| grupo | clips perdidos muestreados | mi detector encuentra luz de color |
|---|---:|---:|
| **CONTROL familia A** | 25 | **44,0 %** |
| `parisfoil2022` (familia E) | 25 | 56,0 % |
| `stpetersburgfoil2019` (familia B) | 25 | 48,0 % |
| `tokyofoil2022` (familia C) | 25 | **28,0 %** |

Criterio declarado antes: ≥ 40 % y ≥ 15 puntos sobre el control. **Ninguno pasa**, y Tokio sale
**16 puntos por DEBAJO** del control. Recuperar igual o menos en la sede «rota» que en las sanas es
incompatible con «el problema es el diseño del marcador de esa familia».

**Segunda medida: ¿son clips sin rótulo?** No. `f12_que_son_los_no_lights.py` midió la fracción de la
banda inferior que es un gráfico quieto en los últimos 2 s: el rótulo está presente en el
**92-100 %** de los clips perdidos de los cuatro grupos. No son repeticiones ni planos de grada.
Pero al mirarlos (`figuras/no_lights_que_son.png`) **no hay ninguna luz** en el último fotograma, y
varios marcadores están **empatados a 0-0 o 4-4**: tienen pinta de acciones que acabaron en alto sin
tocado.

**Tercera medida, la que decide.** `f13_roi_aprendida.py` aprende la ROI de la luz **de los propios
clips de cada sede que sí funcionaron** (mediana de la caja en 14 clips `ok`), y luego la lee de la
forma más tonta posible —máximo temporal del canal medio en la ROI menos su percentil 20— en clips
que no se usaron para aprenderla:

| grupo | clips `ok` de control (¿la lectura funciona?) | clips `no_lights` (¿hay luz?) | `salto_roi` mediano, ok → no_lights |
|---|---:|---:|---|
| `tokyofoil2022` | **90,0 %** enciende | **10,0 %** | 131,1 → **4,0** |
| `parisfoil2022` | 85,0 % | 20,0 % | 195,8 → 5,1 |
| `stpetersburgfoil2019` | 80,0 % | 30,0 % | 155,2 → 3,6 |
| **CONTROL familia A** | 85,0 % | **40,0 %** | 163,0 → 16,6 |

La primera columna valida la lectura (criterio declarado: ≥ 80 %). La segunda es el resultado: **en
los clips que el detector viejo dio por vacíos, la ROI no se enciende**, y la separación es de 131-196
niveles contra 3,6-16,6. No es un problema de umbral: **no hay luz**.

**Las consecuencias, con las cifras:**

- **`status = "no_lights"` es en su mayoría la respuesta CORRECTA.** De los 2.038 clips perdidos,
  extrapolando con estos porcentajes, los recuperables son del orden de **575, o sea el 28 %**.
- **No hay 527 clips recuperables en `tokyofoil2022`, hay ~53.** El punto 1 de la lista de 33.6 queda
  retirado.
- **El mayor grupo de clips recuperables es la familia A: ~476 de 1.189**, repartidos entre 16 sedes.
  O sea que el orden de prioridad correcto es el inverso del que yo mismo propuse.
- Y una pregunta nueva que la medida abre: si el 36,78 % de los clips de `tokyofoil2022` no tienen
  luz, **eso no es un fallo del detector de luces sino del CORTADOR de clips** de ese torneo, que
  produce acciones sin tocado. Es un problema aguas arriba y con otro dueño.

### 35.2 El «acierto por la razón equivocada» de worldchamps: refutado, y era una cifra de un clip

En 33.2 escribí que el rótulo de `worldchamps2022` tiene la barra en `cy` 0,939-0,967 mientras el
detector FIE mira la franja `[0,910, 0,932]`, y que por tanto **estaba mirando 0,03 H por encima de
la luz**. Esa afirmación salía de **un solo clip**. Con la ROI aprendida de **12 clips por sede**:

| | `fy` mediana | `fh` mediana | solape con la franja `[0,910, 0,932]` |
|---|---:|---:|---:|
| `worldchamps2022`, roja y verde | **0,9115** | 0,0184 | **1,000** |
| familia A entera, 30 parejas sede-canal | 0,9078 a 0,9115 | 0,0184 a 0,0553 | p50 **1,000**, mínimo **0,371** |
| parejas con solape < 0,30 | | | **0 de 30** |

**Así que la franja del detector viejo cubre la luz, y la geometría canónica de reserva es correcta.**
Lo que falla en `incheonfoil2022` y `worldchamps2022` es solo el **buscador de cajas de nombre**, y
al fallar cae en una geometría que es la buena. No es un acierto por la razón equivocada, y no hay
nada que corregir ahí. El punto 3 de la lista de 33.6 queda retirado.

Y de paso sale una cifra que contesta a la pregunta original del usuario mejor que el censo visual:
**dentro de la familia FIE la posición de la luz NO cambia entre torneos.** `fy` va de 0,9078 a
0,9115 —un recorrido de 0,004 H, cuatro píxeles a 1080p— en **16 sedes de 2018 a 2023**. El marcador
cambia de aspecto; la geometría de la barra de luz, no.

### 35.3 El arreglo que SÍ funciona: ROI aprendida por sede, 90,5 % en la muestra B

Si la ROI aprendida de una sede separa 131-196 contra 3,6-16,6, la pregunta obvia es si sirve como
detector. Se probó en **la muestra B**, que es la que tumbó mi detector general, con la ROI aprendida
de 12 clips por sede **excluyendo los clips de las muestras A y B**:

| | muestra B (21 clips, uno por sede) | criterio |
|---|---:|---:|
| **ROI aprendida por sede** | **19/21 = 90,5 %** | ≥ 90 % → **PASA** |
| detector general (busca en toda la banda) | 15/21 = 71,4 % | **NO PASA** |

**+19,1 puntos**, y con el mismo criterio que el general no pasó. Falla en dos sedes:
`tokyofoil2022` (`salto_roi` 2,1) y `tokyoolympics2020` (21,0, justo por debajo del 25 declarado).

Y en el primero **se sabe la causa**: su ROI roja se aprendió de 12 clips y salió en `fy` 0,9022, pero
su ROI **verde** se aprendió de 6 y salió en `fy` 0,8414, 0,06 H más arriba. Dos canales del mismo
rótulo no pueden estar a 0,06 H de distancia: **la ROI verde de `tokyofoil2022` está mal aprendida**,
y el arreglo es aprenderla con más clips o exigir coherencia entre los dos canales de una misma sede.
Eso es un arreglo de una sede con una causa medida, no un rediseño.

**Por qué esto funciona y el detector general no.** El general tiene que decidir, en cada clip y sin
saber nada, qué región de toda la banda inferior es una luz, y ahí compite con la pista, las banderas
y los dígitos del reloj. La ROI aprendida no decide nada: **le dicen dónde mirar**, y entonces basta
restar. Es la misma lección que la etapa del cono: un *prior* fuerte y medido vale más que un
detector general.

**Y las dos ideas que se propusieron para el factor 10 del salto, medidas:**

- **«Comparar cada píxel contra su propia mediana temporal»**: eso es exactamente lo que el detector
  ya hacía (`base` = percentil 20 temporal del propio píxel). No era el problema.
- **«Normalizar el salto por el ruido del píxel»**: probado y **peor en las dos muestras** (15/21 en
  A y 14/21 en B, contra 19/21 y 15/21). Retirado con su medida en el docstring de `luces.py`.
- Lo que sí resolvió el factor 10 no fue cambiar de canal ni de normalización, sino **dejar de buscar
  la región**: con la ROI dada, un salto de 28 niveles sobre pista verde se detecta igual que uno de
  268 sobre pista roja, porque ya no hay que distinguirlo de nada.

### 35.4 El atajo de las luces es «gana el que enciende SEGUNDO», y el sustituto del contraataque falla

Medido sobre los 1.815 clips de dos luces con orden resoluble y la etiqueta real `y`:

| regla | acierto |
|---|---:|
| gana el que enciende **PRIMERO** | 35,04 % |
| gana el que enciende **SEGUNDO** | **64,96 %** |
| mayoritaria | 52,18 % |

El 64,96 % reproduce el **63,51 %** que el proyecto tenía medido, y aclara un signo que no estaba
escrito: **el atajo es que gana el SEGUNDO**. Tiene sentido de esgrima —el que contraataca suele
tocar antes y perder por prioridad—, y conviene que esté escrito porque invita a leer mal el techo
del 73,6 %.

Dos cosas más que el conjunto de dos luces no da gratis: **solo el 84,0 %** tiene orden resoluble
(el resto es `same_frame`), y de ésos el **38,4 %** tiene `|dt| ≤ 40 ms`, por debajo de un cuadro a
25 fps.

**El sustituto barato del contraataque, probado y retirado.** Se midió la velocidad de acercamiento
de cada tirador en los 0,5 s previos al tocado, en coordenadas de pista, de `boxes` y `pan` del NPZ y
sin decodificar un solo vídeo. `sospecha_contraataque` = el que está en duda no se acercaba y el rival
sí.

| medida | resultado | criterio | |
|---|---:|---:|---|
| separación entre los clips donde el atajo acierta y donde falla | **3,65 puntos** | ≥ 10 | **NO PASA** |
| regla combinada contra el atajo solo | 59,61 % vs 64,96 % | mejorar | **−5,34: EMPEORA** |
| sospecha en una luz vs dos luces | 21,90 % vs 17,74 % | ≥ 10 | **NO PASA** |

Y va al revés de lo esperado: donde el atajo **falla** la sospecha aparece en el 12,89 %, y donde
**acierta**, en el 20,36 %.

**Por qué falla, para que no se repita igual:** la velocidad del centro de la caja mide desplazamiento
del CUERPO, y un contraataque es una extensión del BRAZO que puede hacerse con el cuerpo quieto o
retrocediendo. La señal que hace falta es cuándo se extiende el brazo armado, que es justo lo que el
cono y la mano armada intentan medir. **Esta vía no tiene atajo geométrico barato.**

El primer error de este experimento también queda escrito, porque es instructivo: la primera versión
solo definía el lado en duda para los clips de una luz, y por eso daba «sospecha en el 0,00 % de los
de dos luces». No era un resultado, era un `NaN` disfrazado de cero. En un clip de dos luces el que
está en duda es **el que encendió primero**.

### 35.5 Y la cifra que el usuario tenía razón en sospechar

Predecir «gana el que enciende» en los **8.049 clips de una luz** acierta el **96,19 %**. Con una sola
luz no hay convención que aplicar, así que esos clips **no enseñan prioridad**, y quedarse con los de
dos luces es lo correcto. Todo lo demás —el conjunto de dos luces por sede, la condición del
contraataque, el blanco no válido y el diseño del aumento sintético con su columna `luz_sintetica`—
está en **`04_documentacion/luces_y_aumento.md`**, con la advertencia de que cambiar una luz de color
por blanco no cambia la etiqueta sino **la física de la acción**.

### 35.6 Todo lo que se publica, en inglés, y la figura que lo resume

El repositorio va a ser público, así que **el texto dentro de las figuras y los `.md` publicables
están en inglés**; los docstrings, los nombres de columna de los CSV y esta sección siguen en
español. La terminología usada es la de esgrima en inglés: `lights`, `scoring apparatus`, `touch`,
`valid target`, `right of way`, `counterattack`, `scoreboard family`, y sobre todo **`white light
(off-target)`** y no `white` a secas, porque en florete el blanco significa fuera de superficie
válida y `white` solo no lo dice.

La figura de cabecera es **`02_validacion/muestras/luces/figuras/scoreboard_families_EN.png`**: las
cinco familias una debajo de otra, con la luz marcada en cada una, la cobertura de cada familia y
**las dos cifras que impiden malinterpretar el resto dentro de la propia imagen**: el 8,11 % de
25.139 clips sin ninguna luz detectada y el 36,78 % de una sola sede, con la aclaración de que de
esos solo el 10 % tiene luz de verdad. Sin esas dos cifras dentro, quien lea «una familia cubre el
81 % de los clips» concluiría que el problema está resuelto al 81 %.

Reescritos en inglés: la figura de familias, los mosaicos del censo, las tiras, las figuras de
encendido, el conteo de luces (`light_count_*.png`, antes `cuenta_luces_*`), el
`02_validacion/muestras/luces/README.md` (que sustituye al `LEEME.md`) y
`04_documentacion/luces_y_aumento.md`.

### Lo que 35 NO dice

- **No dice que la ROI aprendida sea un detector válido.** Dice que enciende en 19 de 21 clips de la
  muestra B. Son 21 clips, uno por sede, y **no hay verdad de referencia**: enciende, pero no se ha
  comprobado que encienda **por la luz correcta** ni en el cuadro correcto. Lo que sí está medido es
  que en los clips sin luz no enciende (10-40 %), que es el confundido que importa.
- **No dice que los 575 clips recuperables lo sean.** Es una extrapolación desde 20 clips por grupo, y
  el intervalo es amplio. Su valor es negativo: sirve para **no** gastar trabajo en Tokio creyendo que
  hay 527.
- **No dice que `tokyofoil2022` esté bien.** Dice que su 36,78 % de clips sin luz no es un fallo del
  detector de luces, y que por tanto el dueño del problema es el cortador de clips. Que el cortador
  esté mal **no se ha medido**: podría ser que ese torneo tenga de verdad más acciones sin tocado.
- **La refutación de 33.2 se apoya en 12 clips por sede**, no en el corpus. Es doce veces más que la
  afirmación que refuta, pero sigue siendo una mediana de doce.
- **El 64,96 % del atajo está medido sobre la etiqueta `y` del cortador**, que sale del cambio de
  tanteo. Si el tanteo se lee mal en alguna sede, el atajo se mide mal ahí. `score_valid` está en el
  índice y **no lo he cruzado**.
- **El sustituto del contraataque se ha probado en UNA forma**: velocidad del centro de la caja en
  0,5 s. Que ésa falle no dice nada de otras ventanas, de la velocidad de la muñeca, ni de la
  extensión del brazo.

## 36. El modelo de pista: el origen deja de ser una convención y la transversal deja de ser un carril, y la escala a lo largo sigue sin pasar su control

Lo que dijo el usuario, en tres trozos que son tres cosas distintas:

> *«Hay un error grande, y es que tomas tú por defecto que empiezan siempre en el centro de la pista,
> pero no: muchos igual empiezan al final de la pista de la derecha... y abajo en el gráfico salen
> como en el centro.»*

> *«Sacar mejor la distancia en el eje del ancho de la pista, porque los pones siempre el de la
> derecha pegado a la derecha y el de la izquierda pegado a la izquierda, pero no estás trackeando de
> verdad su posición real.»*

> *«Esa pista en concreto tiene una plancha más de ancho, un trozo para que no se caigan, pero eso no
> es válido, y está de otro color. Y el último metro es gris y el final delimitado por amarillo, como
> invertido. Y hay otras pistas que están completas y no tienen ese extra. Tienes que detectarlo
> mejor: hay varios modelos de pista y tienes que estudiarlo para saber bien qué parte de la pista es
> cuál.»*

Y el diseño, que lo dio él y es el correcto:

> *«Eso yo creo que puedes sacarlo midiendo el ancho de la pista, sin contar en este caso el extra
> que tiene... Y con eso sabes la distancia perfectamente y lo puedes hacer, porque es fácil: como
> sabes que tiene que ser recto, da igual la perspectiva, por saber que es recto lo puedes cuadrar
> perfectamente sin error, qué punto es. Mira el VAR del fútbol.»*

Más la generalización del coordinador, que es la que desbloqueó el control: **cualquier recta
transversal a distancia conocida sirve igual**, y con tres o más hay **redundancia**, o sea que se
puede ajustar con dos y PREDECIR la tercera.

### 36.1 · El resultado geométrico que hace que esto funcione, y es exacto

Se construye una TIRA RECTIFICADA mandando los dos bordes detectados de la pista a dos filas
horizontales fijas. Con eso el punto de fuga longitudinal `V = A x B` va al corte de dos horizontales,
o sea al infinito horizontal, y eso obliga a que la primera columna de la homografía mundo → tira sea
`(h11, 0, 0)`:

```
columna = (h11 x + h12 y + h13) / (h32 y + h33)      fila = (h22 y + h23) / (h32 y + h33)
```

La **fila no depende de x**, o sea que las rectas paralelas al eje de la pista son filas; y **a fila
fija, la columna es AFÍN en x**. Comprobado numéricamente con la homografía de referencia de tres
clips: su punto de fuga longitudinal cae en las columnas **2,5e10**, **4,1e9** y **-2,5e10** de la
tira, o sea en el infinito en los tres.

Consecuencia práctica, y es exactamente la intuición del usuario: **las separaciones entre marcas se
pueden comparar con las del reglamento sin corregir perspectiva**. Las posiciones del reglamento son
`-7, -5, -2, 0, 2, 5, 7` m y sus separaciones consecutivas `2, 3, 2, 3, 2`, cuyas razones no dependen
de la escala.

### 36.2 · Dos fallos MÍOS que el censo delató, y el segundo llevaba un docstring falso

**1. Usé `co["t"]` como eje largo y es el eje corto.** `corredor` devuelve `s` = recorrido A LO LARGO
y `t` = dispersión A LO ANCHO de la nube de apoyos. Mi primera tira muestreaba
`ss = linspace(t0, t1, 420)`, o sea estiraba 30 px de ancho a 420 columnas. El síntoma era perfecto:
`largo 29 px` con `ancho 102 px` en `UglyNiceAplomadofalcon`, y las **20 tiras del censo salieron con
franjas horizontales, sin ninguna variación a lo largo**. Una pista de 14 m no puede salir más corta
que ancha, y esa comprobación de una línea habría ahorrado el censo entero.

**2. La tira era un paralelogramo, no la pista.** Las filas se trazaban a distancia constante del
origen del corredor, o sea paralelas EN LA IMAGEN, y los dos bordes de la pista no son paralelos en la
imagen: convergen. Así que la fila `u = 0` no era el borde A, y mi propio docstring decía *«los dos
bordes son dos filas fijas de la tira»*, que era FALSO salvo en la toma exactamente frontal. Con el
eje largo roto no se veía, porque en 30 px no converge nada.

**3. Y un tercero, de enchufado:** `perfil_color` devolvía `(ncol, 3)` y la segmentación lo indexaba
como `(3, ncol)`, así que el perfil que segmentaba tenía longitud 3 y salían **CERO fronteras en los
20 clips**. Cero en todos los casos casi nunca es un dato: es una ruta mal enchufada.

### 36.3 · El censo de modelos de pista, con columnas MEDIDAS

`02_validacion/pista/piste_models.csv` y `piste_models_1..4.png`, 22 clips. Lo que se mide por clip:
ancho de la pista en píxeles al principio, al medio y al final del tramo visible; número de fronteras
de color a lo largo; si la zona final es **más clara o más oscura** que el resto (el SIGNO del salto
de luminancia, no una impresión); si hay plancha lateral; y el tramo visible en metros.

| | de 22 clips |
|---|---:|
| con tira rectificada (dos bordes detectados) | **20** |
| con >= 2 fronteras de color a lo largo | **15** |
| con la zona final identificada | **13** |
| plancha lateral DETECTADA | **0** |

Y el reparto de esas 13 zonas finales: **8 más oscuras que la pista y 5 más claras**. O sea que **el
usuario tenía razón y el esquema de color NO es invariante**: 5 de 13 están «invertidas» respecto a
las otras 8. Por eso el detector busca un **CAMBIO** de color y nunca un color, y por eso un detector
de «busca gris» habría fallado en más de un tercio del corpus.

**La plancha lateral: 0 de 22 detectadas, y eso es un fallo declarado, no una ausencia.** El detector
de ancho de combate (`pista.ancho_combate`) no acota nada dentro de los bordes en **ningún** clip. Lo
que sí hay es una explicación plausible y no comprobada: `region_y_bordes` se para en el PRIMER corte
de color saliendo del corredor, así que puede estar parándose ya en el borde de la plancha; si es así,
los bordes detectados son la zona de combate y no hay nada que excluir. **No está medido**, y hasta
que lo esté, la plancha del clip que el usuario señaló sigue sin estar identificada por el código.

### 36.4 · Tres vías independientes para la escala A LO LARGO, y ninguna pasa su control

Esto es el resultado principal de la tanda y conviene leerlo entero antes de creerse ningún metro.

| vía | qué usa | resultado medido |
|---|---|---|
| **rayas pintadas** (central y guardia), por razones de separación | el patrón 2:3:2 del reglamento | encaja en **14 de 20** clips con tira, pero **AMBIGUA en 10 de esos 14**; contra la homografía de referencia de 3 clips se equivoca por factores de **1,3 a 2,6** |
| **píxel cuadrado**: ángulo entre las dos direcciones del suelo + ancho reglamentario | geometría de cámara, ninguna marca | da solución exacta (residuo de cámara 0,001-0,008) en varios clips, pero la **estatura reproyectada** de los tiradores, que es un control independiente, sale de **0,4 a 4,9 m** |
| **fronteras de zona de color** (t.19) | que los últimos 2 m se distingan, sea del color que sea | encaja en **13 de 22**, pero **solo 5 con redundancia** para controlarla; donde la hay, el error de predicción de la marca reservada es **5 cm de mediana y 8 cm el peor** |

Y hay una razón geométrica por la que esto **no es cuestión de afinar**: dos rectas paralelas con
separación conocida fijan la escala **transversal** en todo el plano y **no dicen nada de la
longitudinal**. Hace falta una referencia más — una marca a distancia conocida, el punto de fuga
transversal, o la focal — y en este corpus ninguna de las tres aguanta su propio control.

Para la vía del píxel cuadrado hizo falta además corregir la tira, porque los cortes de los extremos
no son transversales del mundo: se busca la corrección que maximiza la **coherencia vertical** (la
fracción de la energía de borde que sobrevive a promediar filas, entre 0 y 1 por construcción), que
sube de **0,34-0,74 a 0,57-0,91**. La corrección funciona y aun así la escala que sale no pasa la
estatura: lo que falla no es el enderezado, es que el problema está mal condicionado con la cámara
casi perpendicular al eje, que es justo el encuadre normal de esgrima.

### 36.5 · Lo que SÍ queda medido: el ANCLA del origen

| | de 22 clips |
|---|---:|
| con **ancla MEDIDA** del origen (fronteras de zona, t.19) | **13** |
| de esas, **no ambigua** | **11** |
| de esas, con **redundancia** (>= 3 marcas, o sea con control) | **5** |
| de esas, con los **dos tiradores dentro** de la pista (\|x\| <= 7,5 m) | **11 de 13** |
| error de predicción de la marca reservada (LOO) | **p50 5 cm, máx 8 cm** (n = 5 clips) |
| residuo máximo del encaje | **p50 0 cm, máx 4 cm** (n = 13 clips) |

El criterio declarado antes de medir era **>= 8 de 22 con ancla medida, y por debajo de 4 se declara
que la vía no llega**. Sale **13 de 22**, así que la vía llega. Los dos clips donde el ancla coloca a
un tirador FUERA de la pista (`InsecurePettyCreature` a -9,05 m y
`WastefulYawningAmericanquarterhorse` a +10,48 m) se cuentan: es un control físico independiente y
gratis, porque un tirador no puede estar fuera de la pista en la que está tirando.

**El caso que el usuario describió, verificado a ojo:** en `UglyNiceAplomadofalcon` las dos tiradoras
salen ahora en **+2,69 m y +5,00 m**, o sea las dos en la mitad derecha y la de la derecha
**exactamente en la frontera de los 5 m**, que es donde empieza la zona final clara. Antes salían las
dos en el centro, porque el centro ERA el valor por defecto de la convención. Comprobado en el
fotograma 38 del vídeo: la tiradora naranja está pisando la zona clara del final y el mapa la pinta
ahí.

### 36.6 · La transversal: ahora son metros, y su control falla MÁS que antes

La coordenada a lo ancho ya no es un carril: es la fila del pie entre los dos bordes detectados, por
los 1,50 m del reglamento (t.18.1), y **está acotada por construcción**.

**Y aquí está el defecto que mi propio arreglo creó y que pillé mirando un fotograma antes de
publicarlo.** `ancho_combate` devolvía como «zona de combate» el tramo uniforme mayor del perfil a lo
ancho, y ese tramo salía `u = -0,40 a 1,40` — o sea TODO el muestreo, suelo de fuera incluido — en
**18 de 22 clips**. Quien lo usaba para renormalizar se comía un factor **1,8**: un apoyo en el borde
de la pista salía a 0,33 m del borde en vez de a 0,00, y los dos tiradores salían a **0,50 y 0,56 m**,
o sea los dos casi en el centro. El síntoma se veía en el mapa del vídeo: **los dos puntos casi en la
misma línea**. Y lo peor: con la escala comprimida el recorrido transversal salía **0,55 m** y
**PASABA** el control de 0,60 m, o sea que el defecto se estaba tapando a sí mismo.

Con la escala correcta, y esta es la cifra honesta:

| | valor | denominador |
|---|---:|---|
| recorrido transversal <= 0,60 m | **11** | de **40 tirador-clip** |
| clips donde pasan los DOS tiradores | **3** | de **22 clips** |
| recorrido transversal p50 / p90 / máx | **0,75 / 1,63 / 3,33 m** | 40 tirador-clip |
| apoyos fuera de la pista a lo ancho | **p50 2,7 %, máx 23 %** | 20 clips |
| ruido por cuadro de la transversal | **0,9 cm** (mediana) | 40 tirador-clip |
| cruces entre tiradores (control físico del eje) | **0** | 20 clips |

El criterio declarado era **>= 18 de 22** y sale **3 de 22**: **NO se cumple, y por bastante**. Y la
causa NO es temblor: con 0,9 cm de ruido por cuadro, filtrar con la mediana móvil de 9 cuadros baja el
recorrido de 0,57 a 0,55 m en la versión comprimida, o sea nada. Es señal lenta, o los bordes mal
puestos en esos clips. Los tres peores (`HairyCourteousEarwig` 1,85 m, `TestyWatchfulBarebirdbat`
1,60 m con el 23 % de apoyos fuera, `GentleQuerulousElephantseal` 1,44 m) son el sitio por donde hay
que seguir: son los bordes o la pose de esos clips, **no** la escala.

### 36.7 · Los tres criterios declarados y su veredicto

| criterio declarado antes de medir | acepto | sale | veredicto |
|---|---|---|---|
| clips con ancla MEDIDA | >= 8 de 22 | **13 de 22** (11 no ambiguas) | **se cumple** |
| recorrido transversal <= 0,60 m | >= 18 de 22 | **3 de 22 clips** (11 de 40 tirador-clip) | **NO se cumple** |
| `UglyNiceAplomadofalcon` a ojo: las dos en los extremos | las dos fuera del centro | **+2,69 y +5,00 m** | **se cumple** |

### 36.8 · La invarianza por tipo de pista: lo que se puede afirmar y lo que no

Se declaró que la invarianza se demostraría **dejando fuera TIPOS de pista**, con una caída aceptada
de <= 15 puntos. **No se ha hecho como se declaró**, y la razón es que el tipo solo se puede etiquetar
donde el ancla funciona, así que reservar tipos sería reservar un subgrupo definido por el resultado —
el error que este proyecto ya tiene documentado.

Lo que sí se puede afirmar, con su cifra: entre los 13 clips con ancla medida hay **8 con la zona
final más oscura y 5 con la zona final más clara**, o sea que la vía funciona en los dos esquemas de
color. Eso es evidencia de que el detector no está atado a un esquema; **no** es una medida de
invarianza, porque falta saber a cuántos clips de esquema claro falla.

### 36.9 · El mapa de error, que sustituye a «precisión milimétrica»

Un solo número de error sería falso en los dos extremos de la pista. La tira entrega
`px_img_por_col`, o sea cuántos píxeles de imagen vale una columna en cada punto, y de ahí sale el
error en metros por píxel a lo largo del eje. La razón entre el extremo fino y el grueso es
**p50 1,35 y máx 3,51** sobre los 20 clips: en el peor clip un píxel vale **3,5 veces más metros** en
un extremo que en el otro. El muestreo se corta cuando el ancho de la pista baja del 30 % del ancho
central, justo por esto.

### 36.10 · El vídeo: modo público y un fallo de colocación que se vio en un fotograma

El vídeo del trazado tiene ahora un interruptor `PUBLICO` (`--publico`) que sustituye el panel del
clip por el **esqueleto de 308 puntos sobre negro**, para que todos los píxeles del MP4 sean salida de
modelo y no grabación de televisión. Los otros tres paneles ya lo eran.

Y un fallo que solo se ve mirando: en modo tira las marcas de evento se pintaban en
`pos_mapa - x00`, o sea restándoles el ancla de la **convención vieja** cuando su posición ya venía en
el eje nuevo. Con `x00 = 4,88 m` en `UglyNiceAplomadofalcon`, un evento de brazo medido en **+5,18 m**
aparecía en **+0,30 m**: un punto morado suelto junto a la línea central, con la tiradora a cinco
metros de ahí. **Las magnitudes de los eventos no se han reescalado a propósito**: siguen siendo las
del CSV de la serie, porque si el vídeo y el CSV dieran cifras distintas del mismo evento el
entregable no valdría.

### 36.11 · Lo que esta medida NO dice

- **No dice dónde está un tirador en metros a lo largo de la pista con error conocido.** El ORIGEN
  está medido en 13 de 22 clips, pero la ESCALA a lo largo no ha pasado ningún control en ninguna de
  las tres vías. Lo que se entrega es la posición normalizada al tramo visible (afín en metros con
  factor desconocido) y los metros solo cuando hay ancla, con el aviso en el propio panel.
- **No dice que la transversal sea utilizable.** Son metros y están acotados, y su recorrido pasa el
  control físico en 3 de 22 clips. Un recorrido de 1,85 m en una pista de 1,50 m no es una medida: es
  un aviso.
- **No dice que la plancha lateral se esté excluyendo.** Se detecta en 0 de 22 clips. La hipótesis de
  que el barrido de bordes ya se para en ella no está comprobada.
- **No dice nada de los otros 56 clips del corpus etiquetado.** Todas las cifras de esta sección son
  sobre los 22 clips de esta tanda.
- **No dice que las 13 anclas sean correctas.** 11 son no ambiguas y 5 tienen redundancia; las 8 sin
  redundancia están exactamente determinadas por dos fronteras y **no tienen control ninguno**. De
  ellas, 2 colocan a un tirador fuera de la pista, o sea que al menos 2 de las 13 están mal.

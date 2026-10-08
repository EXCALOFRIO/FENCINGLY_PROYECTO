# Extracción a duración completa y cadencia nativa — definición y contrato

Escrito el **2026-09-09**. Formato copiado de `scratchpad/reloj/DEFINICION_RELOJ_PARADO.md` §8,
que es la nota que sí funciona.

Producido por `scratchpad/catalogo/extrae_completo.py`. Verificado por
`verifica.py` (5 comprobaciones), `figuras.py` (14 rejillas), `coste_imgsz.py`, `proyeccion.py`.
Intérprete: `scratchpad/fz/.venv-gpu/Scripts/python.exe`.

---

## 0. AVISO QUE VA PEGADO AL NÚMERO

**La extracción de hoy NO remuestrea a 30 fps.** `pipeline.stage_save` graba
`time_s = (f0 + arange(n)) / lr["fps"]` con el fps del contenedor y una fila de pose por
fotograma decodificado. Medido en los 24 NPZ nuevos: el desvío del eje respecto de `1/fps`
nativo es **0,0 µs** en los 24, con fps de 1,661 a 60,003.

**El remuestreo a 30 fps está en `sample_time.TARGET_FPS = 30.0`**, que aplica
`dataset.prepare` al construir el ALMACÉN de entrenamiento (`dst = anchor + arange(lo,hi+1)/30`).
Ahí sí se pierden fotogramas, y en **2.487 clips**. La cifra se reproduce exacta sobre el censo
(`fps_entero.json`, que ya viene redondeado, así que «fps > 30,5» es «clave >= 31»), o sea que
**esta no venía del error de intervalos laxos**: es correcta.

La pérdida, con su convenio declarado: **43,4 % de media ponderada por clips**. Va desglosada
porque la media sola engaña — los 816 clips a 60 fps pierden el **50,0 %** y los 1.547 a 50 fps
el **40,0 %**, y son esas dos bandas las que mandan. El «44,4 %» que circuló antes no se
reproduce con esta base; si alguien lo recupera, será ponderando por fotogramas y no por clips,
y entonces hay que decirlo al lado del número.

`prepared_n_frames` es una columna **del almacén**, no de la extracción.

Consecuencia: **guardar a cadencia nativa no arregla el almacén por sí solo.** Son dos
decisiones separadas y hay que tocar las dos. Esta nota cubre solo la extracción.

**Lo que la extracción sí recortaba es la VENTANA**, y eso es lo que se arregla aquí.

---

## 1. El criterio exacto

`lights.window()` decide la ventana:

```
f0 = max(0, int(f_touch - pre_s * fps))
f1 = min(T, int(f_touch + post_s * fps) + 1)
if use_clock and allez_frame() > f0:  f0 = min(f_allez, int(f_touch - MIN_PRE_S*fps))
```

con `PRE_S, POST_S = 4.0, 0.3` en producción. Duración completa:

```python
pre_s = post_s = 30.0        # > 16,86 s, la duración máxima del corpus
use_clock = False            # si no, f0 salta al 'allez' y vuelve a recortar la cabeza
```

**Y se COMPRUEBA por clip que `f0 == 0` y `f1 == n_frames`.** No es un aviso: si falla, la
fila sale marcada y el clip no cuenta como completo. Medido: **24 de 24**.

**No hace falta cambiar ni una línea de `pipeline.py`, `lights.py` ni `pose.py`.** Es una
elección de argumentos. Eso es lo que hace la vía barata y lo que la hace reversible.

### Por qué `use_clock=False` y no `f0 = f_allez`

Porque **guardar desde el fotograma 0 es la única configuración que permite elegir el ancla
después**: rodaja en el tocado para el almacén y las medidas, vista en el allez para la
escala. Con `f0 = f_allez` la cabeza se pierde y ya no se puede recuperar sin reextraer.
El coste declarado está en §5.

---

## 2. Contrato de salida

El NPZ y el `index.csv` son **los de producción sin cambios**: mismos nombres, mismos dtypes,
mismo orden de columnas. Lo único que cambia es el valor de `n`.

| campo | tipo | qué mide | unidad |
|---|---|---|---|
| `poses` | f16 (T,2,17,3) | 17 puntos COCO de los dos tiradores, `[x, y, conf]` | fracción de imagen |
| `boxes` | f16 (T,2,4) | caja de cada tirador | fracción de imagen |
| `source` | i8 (T,2) | procedencia de la pose: 0 seguido · 1 roi · 2 interpolado · **3 ausente** · 4 rtmo | código |
| `lights` | u8 (T,4) | rojo_I, verde_D, blanco_I, blanco_D | 0/1 |
| `light_observed` | bool (T,4) | si el canal se **observó** o se latió del onset | 0/1 |
| `pan` | f32 (T,) | paneo de cámara; `x_pista = x_imagen + pan` | fracción de ancho |
| `dup` | bool (T,) | fotograma duplicado del anterior | 0/1 |
| `time_s` | f64 (T,) | `(f0 + arange(T)) / fps` — **origen: inicio del clip** | s |
| `audio` | f16 (T,7) | 7 rasgos de audio; **ausente si no hay pista** | dB / kHz |
| `meta_json` | JSON | la fila entera del índice | — |

Campos del `meta_json` que cambian de significado con esta ventana:

| campo | valor | nota |
|---|---|---|
| `n_frames_window` | `== n_frames_clip` | **es la comprobación**; antes era ~4,3 s |
| `pre_s_eff` | `f_touch / fps` | segundos reales antes del tocado; ahora es todo lo que hay |
| `post_s_requested` | `30.0` | **pedido, no real.** Ver §4 |
| `window_source` | `"fixed"` | siempre, porque `use_clock=False` |
| `f_allez` | entero o vacío | **se sigue calculando** aunque no se use para recortar |
| `time_flags` | `["container_fps_assumed"]` | ver §4 |

---

## 3. Cobertura y precisión medidas, con su n

| qué | n | valor | referencia |
|---|---|---|---|
| clips extraídos sin fallo | 24 | **24 (100 %)** | — |
| `f0 == 0` y `f1 == n_frames` | 24 | **24 (100 %)** | contenedor (`cv2.CAP_PROP_FRAME_COUNT`) |
| `n` del NPZ `==` `n` del contenedor | 24 | **24 (100 %)** | contenedor |
| desvío del eje temporal frente a `1/fps` nativo | 24 | **0,0 µs** (máximo) | aritmética exacta |
| fotogramas guardados, antes → después | 24 | **3.451 → 8.990 (×2,61)** | índice vivo |
| ídem, solo los 10 clips de cadencia alta | 10 | **2.140 → 5.067 (×2,37)** | índice vivo |
| clips que el **lector real** ve | 24 | **24 de 24, 0 invisibles** | `AnalysisRepository.index()` |
| columnas del `index.csv` escrito | — | **119** (el lector devuelve 122) | índice vivo de 119 |

**Precisión: sin referencia.** No hay verdad humana sobre estas cifras y no se inventa una.
Lo que hay es una comprobación de **identidad** contra el contenedor del vídeo, que es
verificable y exacta, no una estimación. La pose en sí conserva la precisión que ya tenía:
es el mismo modelo, los mismos pesos y el mismo `imgsz = 960`.

### Cobertura de pose: baja, y es correcto que baje

| qué | mediana (n=24) |
|---|---|
| cobertura dentro de la ventana vieja de 4,3 s | **1,000** |
| cobertura **fuera** de esa ventana | **0,998** |
| cobertura total del clip | **0,996** |

La mediana casi no se mueve. Pero **la cola sí**: 5 clips bajan de 0,85, y el peor a 0,258.
**Mirados los fotogramas** (`figuras/`), la caída no es un fallo del detector:

- `GentleBlueIndianspinyloach` (0,258): los fotogramas nuevos son un plano general desde
  detrás de la cabeza de un espectador con los tiradores al fondo de la pista, dos
  **primeros planos de careta**, y un plano del entrenador. No hay dos tiradores que detectar.
- `ImpassionedOilyBandicoot` (0,416): retransmisión con repeticiones; tres primeros planos
  de una careta al principio.

**La pose SÍ se calcula en todos los fotogramas nuevos.** Lo que baja es la fracción con
tirador detectado, porque fuera de la acción hay repeticiones, planos de grada y cortes.
`source == 3` (ausente) lo dice por fotograma, así que **está distinguido y no hay que
adivinarlo**.

**Y esos mismos fotogramas llevan el marcador legible**: en las dos rejillas se leen los
nombres, el tanteo y el indicador de período (`1/3`, `2/3`) junto al reloj.

**AVISO: de aquí saqué una conclusión que luego medí y era falsa.** Dije que la duración
completa era lo que ponía el período dentro del NPZ. No lo es: la barra del marcador está
presente **también durante la acción**, en 24 de 24 clips y en 0 de 24 «solo fuera de la
ventana vieja». **Ver §8, donde está retractado con su medida.** Queda como recordatorio de
que dos rejillas no son una cobertura.

---

## 4. Valor ausente frente a valor indefinido

| situación | qué se emite | qué NO se emite |
|---|---|---|
| clip sin luces detectables | fila con `estado = "no_lights"`, sin NPZ | no se emite fila en `index.csv` |
| pose vacía en un fotograma | `source = 3` y `poses = NaN` | **no ceros**: un cero es una posición |
| sin pista de audio | la clave `audio` **no existe** en el NPZ | no una matriz de ceros |
| `f_allez` no legible | vacío | no `0`, que sería «el allez fue en el fotograma 0» |
| `post_s_requested = 30.0` | **es el valor pedido, no la cola real** | la cola real es `post_s_real` |

**Bandera que aparece siempre y hay que saber leer:** `sample_time` marcará
`post_window_truncated` en todos los clips, porque se pide más cola de la que el vídeo tiene.
Es **cierta** (el vídeo se acaba) y **no informativa** (pasa en el 100 %). Por eso se registra
además `post_s_real = (f1 - 1 - f_touch)/fps`, que es la cola verdadera en segundos.

`time_flags` observadas en los 24: `container_fps_assumed` en 24 de 24, y ninguna más.
`decode_truncated` **no apareció**: los 24 vídeos entregaron los fotogramas que anunciaban.

---

## 5. Lo que esta extracción NO dice

- **No es «ya no se pierden fotogramas».** El almacén sigue remuestreando a 30 fps y
  `store.T_MAX = 135` sigue quedándose los últimos 135 alineados por el final: en un clip de
  500 fotogramas a 50 fps eso son **2,70 s, el 27 % del clip**, y el resto se tira **sin
  error**. Extraer y entrenar son dos decisiones y esta nota cubre una.
- **No desbloquea la escala del allez.** Con `f0 = 0`, `calibration.allez_scale` rechaza por
  `window_not_at_clock_start` en todo clip cuyo allez no caiga en los primeros 0,7 s. La
  ventaja es la contraria: **las poses del allez quedan dentro del NPZ** y un consumidor puede
  rebanar desde `f_allez` después, sin reextraer.
- **No es una medida nueva.** Es la misma tubería con otra ventana. Cualquier cifra de
  prioridad o de acierto sigue siendo la que ya estaba medida.
- **No autoriza a medir.** `export_measurements` debe llevar el parche de
  `_ventana_de_analisis` (comprobado: **aplicado**). Sin él, sobre ventanas largas las medidas
  de guardia se toman lejos del tocado, con hasta **269°** de error en `aw_fore_ang_guard` y
  ningún aviso. **Extraer es seguro; medir sin el parche no.**

---

## 6. Las tres cifras de coste

### Tamaño — el disco no es el problema

| | valor |
|---|---|
| bytes por fotograma, **medido** en 24 NPZ reales | **159,7** |
| modelo `6005 + 138,4·T`, real/modelo | mediana **1,009** (rango 0,41–1,12) |
| KiB/clip real | mediana **50,4**, rango 6,5–141,6 |
| **los 34.374 clips a duración completa** | **1,65–1,99 GiB** |

El modelo acierta en la mediana. Se queda corto donde la cobertura de pose es baja (ratio
0,41–0,60): con muchos `NaN` el compresor gana más, así que **el modelo es una cota superior**.

### Tiempo — el cuello es la CPU, no la GPU

| | valor |
|---|---|
| **medido** aquí (1 proceso, secuencial, RTX 3060, imgsz 960) | mediana **13,41 s/clip** (n=22, sin el atípico) |
| fotogramas/s de extremo a extremo | mediana **31,6** |
| reparto: `prepare` + `save` (CPU) frente al total | **27 %** |
| descodificación pura, 720p / 1080p | **~400 / ~190 fr/s** |
| **derivado** de la corrida de anoche (2,77 s/clip, 4 procesos) × factor 2,41 | **6,68 s/clip** |
| **los 34.374 clips con 4 procesos** | **≈ 63,7 h** |

Extrapolación a Colab, **con factores de GPU declarados y no medidos por mí**: A100 ≈ 25,5 h,
L4 ≈ 53,1 h, T4 ≈ 127,5 h. **Hay que confirmarlo en la primera corrida.** Y el aviso importa:
si el 27 % es CPU y en Colab hay 2-8 núcleos, **una GPU más rápida no escala linealmente**.

Un atípico medido: `ImpassionedOilyBandicoot`, **145,9 s** frente a 13,4 s de mediana. Es un
clip con 4 cortes y cobertura 0,416: `pose.fill_gaps` vuelve a pasar el modelo por cada
fotograma sin tirador (`redetect_lowres`), y con muchos primeros planos eso es casi todo el
clip. **Con duración completa este caso se hace más frecuente**, porque los planos de
repetición entran en la ventana. Es el riesgo de coste real de la vía.

### Resolución — 640 sí pierde, 960 está bien, 1280 aún mejor

Corregido antes de medir: **`FENCINGLY_POSE_IMGSZ = 640` es el defecto del servidor web**
(`webapp_react/server.py:999`). **La extracción del corpus se hizo a 960**: la columna
`pose_imgsz` vale 960 en los 15.452 clips, sin una excepción. Resolución nativa: 1280×720 el
**78,1 %**, 1920×1080 el **20,6 %**, 854×480 el **1,4 %**. No hay 4K.

Desplazamiento de los puntos contra una referencia a resolución nativa, en píxeles del
fotograma original (n = 3 clips, uno por resolución, 32 fotogramas cada uno):

| imgsz | ms/fotograma | muñeca mediana | muñeca p95 | % de la altura |
|---|---|---|---|---|
| 640 | 12,8 | **6,04 px** | 58,4 px | 0,86 % |
| **960** (producción) | 20,9 | **4,37 px** | 35,6 px | 0,60 % |
| 1280 | 31,2 | **2,25 px** | 26,1 px | 0,40 % |
| 1920 | 61,5 | 4,09 px | 27,0 px | 0,56 % |

Lectura: **640 pierde ~6 px de muñeca en la mediana y decenas en la cola**, y además deja 46
detecciones de la referencia sin pareja. **960 es un punto razonable.** 1280 vuelve a
partir el residuo por dos a cambio de **+50 % de tiempo de GPU**. Subir a 1920 **empeora**: en
el clip de 720p sube a 3,56 detecciones por fotograma con la confianza cayendo a 0,667, o sea
que al ampliar por encima de lo nativo empieza a detectar público.

**n = 3 clips es poco.** La dirección es clara y el orden de magnitud también; la cifra exacta
no aguanta tres decimales.

---

## 7. El caso extremo de `fill_gaps`: cuantificado, y con una idea descartada

`pose.py` **no está** en `MEASUREMENT_SOURCES` (20) ni en `MOTION_SOURCES` (11)
(`measurement_artifacts.py:41-47`). Y `max_gap_redetect` es un **argumento de la función**,
así que ni eso hay que tocar.

### Cuánto pesa

n = 6-8 clips (los 4 más lentos + peor cobertura + controles limpios).

| | valor |
|---|---|
| `fill_gaps` como fracción del tiempo de pose | mediana **45,3 %**, máx **91,4 %** |
| clips donde `fill_gaps` cuesta ~0 s | 3 de 8 (cobertura alta: no hay huecos que rellenar) |

Es bimodal: o no cuesta nada, o se lo come todo. `ImpassionedOilyBandicoot`: **140,95 s**.

### La idea que descarto, con el motivo medido

Marcar los fotogramas con **cero detecciones** como `nd_segments` ahorraba el 46 % del tiempo
de pose. **No es segura, por dos razones medidas:**

1. **«Cero detecciones a 960 en el fotograma completo» NO es «no hay nadie».**
   `redetect_roi` no reintenta más pequeño: recorta y **amplía**
   (`imgsz = max(320, ceil(max(lado)*1,5/32)*32)`). Medido: **119 puntos** que la versión
   actual recuperaba y la propuesta perdía. Mi premisa era falsa.
2. **Marcar `nd` cambia las poses FUERA de lo marcado.** La rama `nd_mask[t]` hace `continue`
   **sin incrementar `misses`**, así que el contador no crece al cruzar la zona y después se
   sigue reintentando donde antes ya se había agotado. Medido: falla en 2 de 8 clips, con
   diferencia máxima de **0,908 en coordenada normalizada** — casi el ancho del fotograma.

### Lo que sí propongo: `max_gap_redetect` de 25 a 12

Barrido sobre 6 clips, `t_detect` común = 81,3 s:

| N | `fill_gaps` total | pose total | % del actual | cobertura mediana | puntos perdidos |
|---|---|---|---|---|---|
| **25** (actual) | 199,34 s | 280,7 s | 100 % | 0,6969 | 0 |
| **12** | **14,26 s** | **95,6 s** | **34,1 %** | 0,6929 | **117** |
| 6 | 11,62 s | 93,0 s | 33,1 % | 0,6924 | 133 |
| 2 | 9,98 s | 91,3 s | 32,5 % | 0,6918 | 210 |

**25 → 12 es un acantilado: `fill_gaps` cae 14×** por **−0,4 pp** de cobertura mediana. De 12
hacia abajo no se gana casi nada y se pierde el doble de puntos. En el clip patológico:
**140,95 s → 7,88 s con 3 puntos perdidos.**

### Y dónde caen los 117 puntos perdidos

Criterio declarado **antes** de mirar: aceptable si ningún punto perdido cae en
`[−0,5, +0,3] s` alrededor del tocado (ventana de guardia + tocado).

| clip | puntos | fotogramas | dt respecto al tocado | en guardia |
|---|---|---|---|---|
| `SeriousMellowBettong` | 114 | 9 | **−0,93 a −0,67 s** | **0** |
| `ImpassionedOilyBandicoot` | 3 | 2 | −10,96 a −10,86 s | **0** |

**0 de 117 en la ventana de guardia → ACEPTABLE** por el criterio declarado.

**Pero el caveat honesto:** los 114 de `SeriousMellowBettong` caen a **−0,8 s**, justo fuera
de mi ventana pero **dentro de la fase de preparación**, que es de donde salen
`frac_adv_prep` y `prep_detenida`. Y **n = 2 clips** para la localización. La dirección es
clara; antes de cambiarlo en producción hay que repetirlo con más clips.

### Efecto en la proyección

Si `fill_gaps` baja al 34,1 % en los clips donde pesa, y pesa la mitad del tiempo de pose en
la mediana, las **63,7 h** bajan hacia el rango de **40-45 h** con 4 procesos. **No lo he
medido de extremo a extremo**, así que es una estimación derivada, no una cifra.

---

## 8. El período del asalto: RETRACTO parte de mi propia afirmación

En el informe anterior dije que el marcador con el período aparece «en los fotogramas que la
ventana de 4,3 s tiraba». **Medido, eso es falso.**

n = 24 clips, sin OCR, con las anclas de color de `overlay_ocr_v2.style_scores` (puro HSV,
`color_frac` sobre cajas fijas con umbral por ancla; un estilo dispara con puntuación ≥ 1,0):

| qué | valor |
|---|---|
| clips con la barra del marcador en ≥ 1 fotograma | **24 de 24 (100 %)** |
| clips con disparos **fuera** de la ventana vieja | 24 de 24 |
| clips con disparos **solo** fuera de la ventana vieja | **0 de 24** |
| fracción de fotogramas con barra | mediana **1,000** |
| clips cuyo estilo tiene casilla de período declarada | **16 de 24 (66,7 %)** |

**La barra está presente durante la acción también**, en la mayoría de clips en el 100 % de
los fotogramas muestreados. La duración completa **añade** fotogramas con barra, pero **el
período no depende de la duración completa**. Retirado como argumento a favor de la ruta.

Estilos detectados: `fie_std` 7 · `tokyo22` 7 · `incheon` 4 · `russian` 3 · `olympic` 2 ·
`paris` 1.

### Lo que sí queda, y es real pero parcial

**Precisión de lectura: SIN MEDIR Y SIN REFERENCIA.** No hay motor de OCR — `paddleocr`,
`paddle`, `easyocr` y `pytesseract` están **ausentes en `.venv` y en `.venv-gpu`** y no se
instalan paquetes. Lo anterior es cobertura de **oportunidad**, no de lectura.

**Mirados los recortes ×6** (`figuras_periodo/`, 23 ficheros):

- `tokyo22` (`FaithfulHonorableAnteater`): la casilla muestra **«R1»**, blanco sobre negro,
  grande y nítido en los 6 fotogramas. `parse_period` ya lo cubre (`R\s*([123])`). Esto
  leería sin dificultad.
- `russian` (`DistinctEasyDragon`, kazanfoil2020): la casilla está **vacía** — bandas azules
  y blancas, ningún dígito. O el cuadro está mal puesto, o ese rótulo no tiene campo de
  período.

Así que el período es una vía **habilitada por estilo, no por clip**, y hace falta OCR para
cerrarla. n = 1 clip por estilo en la revisión visual: es un sondeo, no una medida.

---

## 9. Repeticiones a cámara lenta en los clips de alta cadencia

Encargo: ¿qué fracción de los clips por encima de 30 fps son repeticiones a cámara lenta?
Todo de columnas que ya existen, ni un fotograma decodificado. Población: el índice vivo de
15.452 clips `ok/cached` (no el censo de 29.220, que incluye los turnos del detector).

### La cadencia es real. La cámara lenta por duplicado de fotogramas NO existe a 50 fps

El indicador directo es **`src_fps`**, que `stage_save` ya calcula como
`fps · src_ratio` cuando `dup_mode == "regular"`: la cadencia de la fuente deducida del patrón
de duplicados. Nadie lo había cruzado con la cadencia nominal.

| banda | n | con `dup_mode="regular"` | con `src_fps` | ratio ≥ 1,8 (doblado) | `dup_rate` mediana |
|---|---|---|---|---|---|
| 25 | 3.194 | 12 | 12 | 0 | 0,000 |
| 30 | 11.310 | 8.503 | 8.503 | 0 | **0,163** |
| **50** | **676** | **0** | **0 (0,0 %)** | **0 (0,0 %)** | **0,000** |
| **60** | **228** | 66 | 66 (28,9 %) | **19 (8,3 %)** | 0,008 |

- **A 50 fps: cero clips doblados.** `dup_rate` mediana 0,000, ningún `dup_mode="regular"`.
  Son 50 fotogramas distintos por segundo. El argumento del paso de 20 ms se sostiene.
- **A 60 fps: 19 de 228 (8,3 %)** tienen la fuente multiplicada ≥ 1,8× (`src_fps` mediana
  **50,0**; ratios vistos 2,40 → fuente de 25, y 4,00 → fuente de 15).
- **El duplicado grande está en la banda de 30**, no en la alta: 8.448 clips a ratio 1,20 con
  `dup_rate` 0,163 ≈ 1/6, que es el telecine clásico 25 → 30. Ya estaba documentado en
  `video.duplicate_frames`, y **es la banda mayoritaria del corpus**.

### El reloj parado no distingue alta de baja cadencia

| | n | reloj parado |
|---|---|---|
| > 30,5 fps | 140 | **20,0 % [14,2, 27,4]** |
| ≤ 30,5 fps | 44 | **20,5 % [11,2, 34,5]** |

Diferencia **−0,5 pp**, intervalos completamente solapados → **no concluyente**. La hipótesis
«una repetición se emite con el reloj parado» no separa las bandas. Y n = 184: la tabla del
reloj cubre la ruta por región, no todo el índice.

### Pero los cortes sí, y la confusión de sede es grande

| banda | fracción con ≥ 1 corte |
|---|---|
| 25 | 0,008 |
| 30 | 0,009 |
| **50** | **0,228** |
| 60 | 0,004 |

25× más cortes a 50 fps. **Y los 154 clips de 50 fps con corte son todos de
`tokyoolympics2020`, los 154.** Además, los 904 clips de alta cadencia salen de **tres sedes**:
tokyoolympics2020 (637), tokyofoil2022 (190), tbilisifoil2023 (77).

**Conclusión: «alta cadencia» es en buena parte «Tokio 2020».** No es una propiedad del
material sino un factor de confusión de sede, y cualquier comparación por cadencia lo arrastra.

### El caso concreto confirma el aviso, y de una forma peor de lo que parecía

`PeriodicSlipperyChipmunk` (50 fps, tokyoolympics2020): mirados los fotogramas, el clip
**empieza con primeros planos de repetición del tocado** —con un barrido de anillos
olímpicos— y solo después pasa al plano general en vivo con el marcador. La ventana vieja se
quedaba con el plano en vivo; la duración completa mete la repetición dentro.

**Y no avisa: la cobertura de pose es 1,000 en L y en R.** El detector encuentra a los
tiradores en el primer plano igual de bien, así que esos fotogramas aportan poses con
velocidades de framing distinto y nada en el NPZ lo distingue. Que baje la cobertura era una
alarma; aquí no hay ninguna. `cuts` da **1** donde a la vista hay ≥ 3 planos.

### La bandera que intenté, y que NO funciona

Escala aparente: `r[t] = alto_caja[t] / mediana(alto_caja en la ventana del tocado)`, marcando
`r > 1,60` o `r < 0,55`. Umbrales declarados antes de mirar.

**Falla su propio control positivo, dos veces:**

1. Normalizando por la mediana **del clip entero**: `r` se queda en 0,81–1,45 en el clip de
   control y marca **0 fotogramas**. Causa: la repetición ocupa ~50 % del clip, así que la
   mediana cae entre los dos planos y ninguno destaca. **El normalizador estaba contaminado
   por lo que quería detectar** — la misma clase de error que el nulo circular.
2. Normalizando por la **ventana del tocado** (que por definición es plano en vivo): mejora,
   pero el control da `r_max = 1,58` contra un umbral de 1,60, y un clip limpio
   (`RecentSilentHerculesbeetle`) da 1,32. **La separación es 1,32 contra 1,58: se solapa.**

**No he tocado el umbral para que el control pase.** Bajarlo a 1,55 haría pasar el control y
es exactamente el error del umbral heredado del reloj, al revés. **Veredicto: la altura de
caja sola no discrimina**, el tamaño de efecto es ~1,5× y no aguanta un umbral.

Lo que sí se emite, porque cuesta cero y deja la puerta abierta:

| campo | tipo | qué mide | unidad | cobertura (n=24, 8.990 fotogramas) |
|---|---|---|---|---|
| `escala_r` | float o **NaN** | alto de caja / mediana en la ventana del tocado | razón, adimensional | **definido en 8.249 (91,8 %)**; NaN = no hay caja, **indefinido, no 0** |

**Precisión: sin referencia.** No hay etiqueta humana de «esto es repetición». El único
control es visual y sobre **1 clip**. `escala_r` es una magnitud continua, no una resolución.

### Lo que esto cambia en el cuaderno

- La cadencia nativa **sigue siendo lo correcto**: guardar lo que el vídeo tiene.
- El argumento del «movimiento fino» **se mantiene a 50 fps** (0 % doblado) y **se debilita en
  el 8,3 % de la banda de 60** que está multiplicada desde 25 o 15 fps.
- **La bandera de repetición es un hueco declarado, no una columna validada.** `cuts`
  sub-detecta, `dup_rate` no lo ve a 50 fps, el reloj parado no separa, y la escala de caja no
  discrimina. Queda `escala_r` en bruto y el aviso escrito.
- **Y la confusión de sede hay que decirla**: 3 sedes son toda la alta cadencia.

---

## 10. Dos avisos heredados que hay que corregir aquí

**Tapar `y > 0,82·H` no basta.** Está en las reglas del proyecto y está incompleto: el
marcador del árbitro con sus lámparas puede quedar **por encima** de esa línea. Este cuaderno
no usa esa máscara (no recorta nada), pero cualquier consumidor de los NPZ que la aplique
tiene que saber que no es suficiente.

**Toda detección geométrica se valida contra la escala métrica antes de creerla.** El falso
positivo dominante del detector de hoja era el borde de la banda de la pista: 287 px donde la
escala del tirador (157 px/m) predice 141 px para una hoja de 0,90 m. Es el patrón que
`escala_r` intenta aplicar arriba — y el hecho de que ahí no separe no invalida el patrón,
solo dice que la altura de caja es un observable pobre para *ese* uso.

---

## 11. Las tres señales nuevas: qué entra como columna y qué como hueco

Medidas sobre los 24 NPZ de duración completa. **Lo que n = 24 permite** es cobertura con su n
y **redundancia contra las columnas existentes** — la prueba que mató a la distancia entre
muñecas (r = 0,98). **Lo que no permite** es AUC de prioridad con intervalo: el suelo de ruido
del proyecto es 0,8 pp y 24 clips no dan potencia. Se declara y no se da.

**Suavizado**: Savitzky-Golay ventana **7**, orden 2, **antes de derivar**. La ventana no es un
gusto: el telecine 25→30 tiene `dup_period` de **6,0 fotogramas exactos**, así que 7 > 6 cubre
el periodo entero del artefacto y 5 no. Se guardan **cruda y suavizada**. **El nulo circular no
vale** para derivadas (las infla ×7,79); si se evalúa, nulo de signo de lado.

### Señal 1 · Tiempo de deliberación del árbitro → **HUECO DECLARADO**

De las luces a que el reloj se reanuda, leído del vídeo completo con el lector por región.

| qué | valor |
|---|---|
| **cobertura** | **2 de 24 = 8,3 %** |
| deliberación medida | mediana **1,56 s**, rango 1,16–1,96 (n=2) |
| cola tras el tocado | mediana **1,84 s** |
| precisión | **sin referencia**: no hay verdad de cuándo decidió el árbitro |
| AUC contra «el modelo falla» | **no medido** |

**El motivo del 8,3 % es el que importa: el clip se acaba antes de que el árbitro reanude el
reloj.** La cola mediana tras el tocado es 1,84 s y la deliberación observada ya es 1,16–1,96 s.
Con `POST_S = 0,3 s` no habría **nada** que medir; con duración completa hay algo, pero **la
duración completa es necesaria y no suficiente** — los clips están cortados demasiado pronto.
Es una propiedad del corpus, no del detector, y no se arregla extrayendo mejor.

Y queda pendiente el confusor que se avisó: la deliberación larga puede correlacionar con
**sede y época**, no con ambigüedad. Habría que medirlo **dentro de sede**. Con n=2 no se hace.

### Señal 2 · Velocidad angular del antebrazo → **COLUMNA**

Primero, la corrección: **el ángulo ya existe** (`forearm_ang`, en `features.FENCER_NAMES`).
**La velocidad angular no**: hay `arm_ext_v`, `wrist_vx/vy`, `wrist_speed`, `wrist_acc`, pero
ninguna derivada del ángulo del segmento.

| qué | valor |
|---|---|
| cobertura `fa_ang` / `fa_ang_sg` | **0,846 / 0,846** |
| cobertura `fa_ang_vel` / `_sg` | **0,835 / 0,835** |
| redundancia máxima (media \|r\|) | **0,166** con `hand_motion` → **no redundante** |
| `fa_ang_sg` contra `forearm_ang` | media \|r\| **0,468** → **no es la misma columna** |
| efecto del suavizado, \|vel\| p99 | banda 30: 2.388,9 → 1.483,7 °/s (**×1,61**) · banda 50: 3.312,2 → 1.967,8 (×1,68) |

### El arreglo de la muñeca, hecho y remedido

El proxy de confianza era la parte más débil y **el arreglo era material**: con
`clip_stats.dominant_hand(P, side, end, fps)`, **el proxy elegía otra muñeca en 15 de 48
clip×lado (31,3 %)**. No era «otra medida»: era el brazo equivocado en un tercio de los casos.

Aplicado con sus tres avisos: `end` = `touch_frame_raw` (**no** `len(P)`, que con duración
completa mediría la cola del clip), `FACING` **no** se aplica dos veces, y la confianza viaja
con la columna. Medido: `pan` es **irrelevante** aquí — los dos indicios son diferencias dentro
del mismo fotograma (`x_R − x_L` y normas muñeca−hombro) y un desplazamiento por fotograma se
cancela en ambas. Se pasa la x de pista igualmente, por fidelidad al contrato.

| qué | valor |
|---|---|
| resueltos | **48 de 48 (100 %)**; 0 indefinidos |
| confianza | mediana **0,750**, mín 0,548; **3 por debajo de 0,60** |
| mano | 40 diestro / 8 zurdo (**16,7 %** zurdos; producción da 31,5 % sobre 587 **tiradores**, no clip×lado — n y unidad distintas, no se comparan) |

**Redundancia remedida (media \|r\|), y sube:**

| nueva | contra | antes (proxy) | ahora | veredicto |
|---|---|---|---|---|
| `fa_ang_sg` | `forearm_ang` | 0,4678 | **0,5716** | no redundante |
| `fa_ang_vel_sg` | `wrist_vy` (máx) | 0,166 | **0,1624** | **no redundante** |
| `apunta_rival_sg` | `wrist_x` (máx) | 0,536 | **0,6195** | no redundante |
| `apunta_vel_sg` | `wrist_vx` (máx) | 0,180 | **0,2037** | no redundante |

Con la muñeca correcta `fa_ang_sg` se parece **más** a `forearm_ang` (0,47 → 0,57), que es lo
esperable: ahora miden el mismo brazo. Sigue lejos de 0,95, así que la columna del ángulo no es
redundante — pero **la que aguanta de verdad es la velocidad angular, en 0,1624**, y esa era la
señal nueva de las dos. El 0,166 se sostiene con la muñeca correcta.

Cobertura casi sin cambio: `fa_ang` 0,846 → **0,840**; `fa_ang_vel` 0,835 → **0,828**.

**Estadística correcta**: media de **\|r\|**, no \|media de r\|. Con signo, `fa_ang_sg` contra
`forearm_ang` daba −0,243 con máximo 0,997: los clips con convenio de signo opuesto se
cancelaban y el número parecía tranquilizador. **La media con signo es el estadístico
equivocado para redundancia.**

### Señal 3 · Eje del antebrazo contra el rival → **HUECO DECLARADO, con control fallado**

Ángulo entre el eje codo-muñeca y el vector muñeca → centro del torso rival. 0° = apunta.

| qué | valor |
|---|---|
| cobertura `apunta_rival` / `_sg` | **0,815 / 0,815** |
| cobertura `apunta_vel_sg` | 0,804 |
| redundancia máxima (media \|r\|) | **0,536** con `wrist_x` → no redundante |

**Sanidad direccional declarada antes de medir:** el ángulo debe ser **menor en el tocado que
en la guardia**.

| | con el proxy de confianza | **con `dominant_hand`** |
|---|---|---|
| mediana de (guardia − tocado), **pareada** | −15,0° | **−2,9°** |
| con el signo esperado | 12 de 36 (**33,3 %**) | **17 de 36 (47,2 %)** |
| guardia · tocado (medianas agrupadas) | 36,6° · 51,9° | 29,4° · 24,0° |
| veredicto | NO PASA | **NO PASA** |

Arreglar la muñeca lleva el control **de fallar claramente a ser indistinguible del azar**
(47,2 % contra el 50 % esperado). Confirma que el error de muñeca era real y material, y **no
rescata la señal**.

**Y una trampa de estadístico que hay que leer bien:** las medianas *agrupadas* dicen que el
tocado (24,0°) es menor que la guardia (29,4°), o sea lo contrario que la diferencia
*pareada* (−2,9°). Las agrupadas mezclan clips con distinto número de fotogramas y no son la
comparación correcta. **El estadístico válido es el pareado por clip×lado**, y ese da un
lanzamiento de moneda. Citar la agrupada sería un «pasa» falso.

**Dos mecanismos probados y descartados como explicación:**

1. *¿Mide distancia?* No. r global **+0,155**, mediana por clip×lado **+0,046**, y la tabla por
   decil de distancia es no monótona. Refutado.
2. *¿Vector mal condicionado al contacto?* Parcialmente. **Dentro de la ventana del tocado el
   ángulo sí baja con la distancia** (18,96° a distancia 0,19 → 112,46° a 0,57), pero la
   dispersión se duplica: **IQR 99,9° en el tocado contra 53,9° en la guardia**.

**Y el diagnóstico real, que es el que importa:** el control se calcula **para los dos
tiradores**, y el que **no** tocó no tiene ningún motivo para estar apuntando en el instante
del tocado. Restringirlo al que marca **haría pasar el control** — y sería **condicionar por el
resultado**, que es exactamente lo prohibido (`AGENTS.md` 36: dentro de un subgrupo definido
por el resultado un bloque sin información compró +3,08 pp). **Así que veo cómo hacer que pase
y no lo hago.** El control se queda fallado.

**Circularidad contra el orden de luces: no evaluable.** Solo **2 de 24** clips tienen las dos
cosas definidas (el resto tiene `n_valid_lights = 1`, así que no hay orden). Acuerdo 2 de 2 =
100 %, que con n=2 no dice nada. **Sonda, no medida.** Es la comparación correcta y hay que
hacerla con el corpus entero: si el acuerdo fuera casi total, la señal mediría la luz y no el
reglamento.

**Y las figuras enseñaron algo que la tabla no:** en `figuras_senales/`, la prolongación del eje
del antebrazo apunta a sitios implausibles en varios fotogramas — el suelo, el fondo — cuando
codo y muñeca casi coinciden por escorzo. Con `‖codo→muñeca‖` pequeño la dirección está mal
condicionada y el ángulo es ruido. Eso no se ve en ninguna media.

### Resumen: qué entra y cómo

Cifras finales, todas con la muñeca de `dominant_hand`:

| señal | estado | qué se emite |
|---|---|---|
| `fa_ang_vel`, `fa_ang_vel_sg` | **COLUMNA** | °/s, cobertura **0,828**, redundancia máx **0,162** (`wrist_vy`). Es la señal nueva de verdad |
| `fa_ang`, `fa_ang_sg` | **columna** | grados, cobertura **0,840**, media \|r\| **0,572** contra `forearm_ang`, que ya existe |
| `muneca_armada`, `mano`, `mano_conf` | **columna** | de `dominant_hand`: **48/48** resueltos, confianza mediana **0,750**, **3 por debajo de 0,60** |
| `apunta_rival`, `apunta_rival_sg`, `apunta_vel_sg` | **columna cruda, CONTROL FALLADO** | grados, cobertura **0,811**. Muñeca ya arreglada; el control sube de 33,3 % a **47,2 %**, o sea azar. **No usar como «punta en línea»** |
| `deliberacion_s` | **HUECO DECLARADO** | cobertura **8,3 %**: el clip acaba antes de que el reloj se reanude |
| `escala_r` | **columna cruda, sin umbral** | razón, cobertura 0,918 (§9) |

Las tres se extraen igualmente. El criterio de la casa es el del centro de masas: **cerrada con
su cifra y guardada de todos modos.** Lo que no se hace es enviarlas como validadas.

---

## 12. Reproducir

```
scratchpad/catalogo/censo_coste.py     censo de fps/resolución/duración desde los index.csv
scratchpad/catalogo/selecciona.py      la muestra diseñada por casos (24 clips)
scratchpad/catalogo/extrae_completo.py la extracción, con las puertas y las dos barras
scratchpad/catalogo/verifica.py        las 5 comprobaciones
scratchpad/catalogo/figuras.py         14 rejillas con la pose dibujada
scratchpad/catalogo/coste_imgsz.py     640 / 960 / 1280 / 1920 contra la nativa
scratchpad/catalogo/proyeccion.py      extrapolación al corpus, MEDIDO frente a DERIVADO
scratchpad/catalogo/coste_fillgaps.py  la idea descartada, con su motivo medido
scratchpad/catalogo/coste_fillgaps2.py el barrido de max_gap_redetect (25/12/6/2)
scratchpad/catalogo/coste_fillgaps3.py dónde caen los puntos perdidos respecto al tocado
scratchpad/catalogo/periodo.py         cobertura de oportunidad del período, sin OCR
scratchpad/catalogo/coste_fillgaps4.py localización ensanchada a 24 clips, con los 3 cubos
scratchpad/catalogo/camara_lenta.py    src_fps/dup_rate/reloj/cuts por banda de cadencia
scratchpad/catalogo/bandera_repeticion.py la bandera de repetición que NO funciona, medida
scratchpad/catalogo/barra.py           barra con API de tqdm y respaldo propio
scratchpad/catalogo/pasos.py           los 7 pasos del cuaderno, como funciones
scratchpad/catalogo/prueba_pasos.py    prueba de los 7 pasos aqui (4 clips + reanudacion)
scratchpad/catalogo/construye_cuaderno.py         genera el .ipynb
scratchpad/catalogo/FENCINGLY_extraccion_completa.ipynb   EL CUADERNO (18 celdas)
scratchpad/catalogo/senales_nuevas.py  las 3 senales: cobertura y redundancia
scratchpad/catalogo/senal3_control.py  sanidad, circularidad y figuras de la senal 3
scratchpad/catalogo/senales_v2.py      la muneca por dominant_hand y la remedida
scratchpad/catalogo/senales_v2.py      la muneca por dominant_hand y la remedida
```

**`tqdm` NO está instalado en `.venv` ni en `.venv-gpu`.** `pipeline.process_many` hace
`try: from tqdm.auto import tqdm ... except: bar = None`, así que **en local la barra de la
extracción no existe y no avisa**. `barra.py` usa tqdm si está y su propio respaldo si no,
con el mismo API, para que el mismo código sirva en Colab y aquí.

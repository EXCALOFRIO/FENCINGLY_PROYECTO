# FENCINGLY — el paquete único de Colab

Escrito el 2026-09-09 sobre `FENCINGLY_actualizado/COLAB_RUNBOOK.md` (2026-09-06) y
`ANTES_DE_COLAB.md`. **[medido]** = lo produjo un comando de esta sesión, y dice cuál.
**[extrapolado]** = declara su base.

**Corrección de cabecera al runbook anterior.** Decía: *«No GPU was available locally, so every GPU
figure is extrapolated»*. **Eso ya es falso.** Anoche se midió en una **RTX 3060 de 12 GB** (torch
2.14.0+cu126, CUDA 12.6, confirmado en esta sesión). Lo que ahora es **[medido]** y lo que sigue
siendo **[extrapolado]** está separado en el §4. Lo único que sigue extrapolado es el coste de
**YOLO-pose por clip en T4/L4/A100**, porque de esas tres no había ninguna.

---

## 0. La regla de protocolo. Va primero porque es la que más fácil se rompe

**El número que salga de esta corrida es una LÍNEA BASE NUEVA. No se puede comparar con el 73,88 %,
ni con el 70,04 %, ni con ninguna cifra anterior del proyecto.**

Con duración completa cambian `pre_s_eff`, `n_frames_window`, cada rasgo derivado de la ventana y la
huella del productor. Es otra cadena de medida. **Restar una cifra de un protocolo a una cifra de
otro protocolo es un error de protocolo, no una comparación conservadora.** El informe del proyecto
ya lleva esa disciplina escrita («nowhere in this report is a figure from one protocol subtracted
from a figure of another») y aquí aplica igual.

Lo que **sí** se compara: duración-completa contra duración-completa, con y sin el cambio que se esté
probando, **pareado por clip**.

Esto está escrito en tres sitios del paquete y en ninguno se puede saltar: la celda 0 de markdown de
los dos cuadernos, la salida de `preflight.protocolo()` que se imprime al principio de cada
ejecución, y `run_config.json`, que graba `"protocolo": "linea_base_nueva_no_comparable_con_73_88"`
junto al resultado.

### El veredicto se reporta en dos poblaciones. Siempre

El atajo «gana el que enciende segundo» acierta **63,51 %** solo, y el modelo lo predice en el
**72,71 %** de los clips: lo sobreusa. Eso parte el corpus en dos:

| población | n | acierto |
|---|---:|---:|
| ganó el que encendió **segundo** | 1.671 | **83,00 %** |
| ganó el que encendió **primero** | 960 | **45,21 %** |
| brecha | | **+37,80 pp** |

* llevar los 960 de 45,21 % a **60 %** vale **+5,40 pp** globales; a **75 %**, **+10,95 pp**
* degradar 5 pp los 1.671 cuesta **−3,18 pp**

**El veredicto es la media global pareada** (`preflight.dos_poblaciones`). El subgrupo describe dónde
cayó la ganancia; nunca es la población de evaluación, nunca un objetivo de optimización y nunca una
puerta de enrutado: dentro de un subgrupo definido por el resultado la base cae bajo el azar y allí
un bloque **sin información compró +3,08 pp**.

### Limitación honesta, y va en el informe

Con la potencia de este corpus **un efecto real de +1,6 pp es indetectable**, y hacen falta **12 a 20
campeonatos más** para tener el 80 % de potencia sobre ese tamaño de efecto. **Una mejora real puede
no ser demostrable aunque exista.** El cuaderno no promete lo que este corpus no puede medir.

### Varas de medir, para no creerse un delta

* Cambiar **solo el backend de atención**, misma receta, misma semilla, mismo intérprete: **+0,36 pp
  con 327 de 3.047 decisiones distintas**. Cualquier mejora de red por debajo de eso es
  indistinguible de recompilar. Por eso `preflight.determinismo()` fuerza el backend `math`: medido
  en RTX 3060, da `max |p1−p2| = 0,000e+00` entre procesos distintos, sin coste de tiempo (37-39 s
  por pliegue en los tres casos), y estrecha el recorrido entre semillas de 1,35 pp a **0,66 pp**.
* **El nulo se elige por el tipo de rasgo.** El desplazamiento circular **preserva** los de nivel
  (una columna quedó a r = +1,000), **infla** los de derivada (hasta 7,79×) y es **inerte** sobre
  representaciones agrupadas (coseno 0,9980). Para nivel, nulo de **signo de lado aleatorio**.
* **Dos convenios de promediado de semillas no son la misma cantidad.** El banco de ablaciones
  promedia el *acierto por clip*; el artefacto promedia las *probabilidades* y umbraliza una vez. Un
  efecto de +1,03 con el primero fue **+0,80 con intervalo cruzando el cero** con el segundo. **Usa
  el del artefacto** — es lo que hace `dos_poblaciones`.

---

## 1. Cómo se ejecuta

### Cuaderno 01 — cinco números, y tres casi nunca se tocan

```python
PERCENT_OF_FOIL_CLIPS = 1.0          # 1.0 = ensayo representativo; 100.0 = corpus de florete

VENTANA_ALMACEN      = "completa"    # qué se GUARDA en el NPZ    "desplegada"|"amplia"|"completa"
RODAJA_ENTRENAMIENTO = "recomendada" # sobre qué se MIDE y ENTRENA "historica"|"recomendada"|"ancha"
T_MAX                = 140           # cuadros del Store; tiene que admitir la rodaja
USAR_RELOJ           = False         # apagarlo: guardia 72,78 % -> 91,42 %, a coste cero
```

**Almacenar y entrenar son dos decisiones distintas, y ahí está todo el diseño.** Guardar el clip
entero cuesta GPU y High-RAM y compra la opcionalidad que se pedía —tener el clip entero para lo que
venga, sin volver a descargar 93 GB—. Entrenar sobre una rodaja derivada de ese NPZ **no cuesta nada
más** y quita tres de los cuatro precios de la duración completa. Y lo que la duración completa
compra de verdad —re-ventanear las **256 columnas agregadas**, que es donde está el techo— se compra
igual, porque las agregadas se recalculan sobre la rodaja que se elija.

Lo demás —número de clips, perfil de pose, hilos de Torch, paralelismo de descarga, presupuesto de
ventana decodificada, intervalo de instantánea, borrado del mp4, la descarga del catálogo, la de los
pesos YOLO, el registro de particiones— **se deriva del runtime que Colab te dio**. Después,
ejecutar todas las celdas.

### Cuaderno 02 — no se pone nada

Apuntar `FENCINGLY_WORKSPACE_ZIP` al zip que produjo el 01 (o subirlo) y ejecutar todas las celdas.
Épocas, paciencia, `d_model` y tamaño de lote los deriva `training_shape` del número de muestras del
Store y de la VRAM/RAM detectadas, se imprimen antes de entrenar y quedan congelados en el
checkpoint, así que una reanudación mantiene el plan.

### Cuaderno 03 — opcional

Sin cambios. No participa en la medida.

### Lo que hay que subir a Colab junto a los cuadernos

`preflight.py`. Los cuadernos lo buscan en `cwd`, `cwd/paquete`, `/content` y `/content/paquete`. Si
no está, la primera celda de configuración falla con `ModuleNotFoundError`, que es el
comportamiento correcto: sin él no hay ninguna de las comprobaciones que impiden que el número salga
inválido.

---

## 2. Lo que hace el porcentaje (sin cambios, sigue siendo cierto)

`stratified_percent_sample` (colab_workflow.py) toma una muestra **proporcional, determinista y
estratificada por torneo**, no los primeros N. Cada torneo aporta la misma fracción de la muestra que
tiene en el corpus (reparto por mayor resto, así que los conteos por torneo suman exactamente el
total pedido); dentro de un torneo la elección va por turnos sobre los asaltos, ambos ordenados por
`sha256("semilla|clave")`; la semilla está fija, así que el mismo porcentaje devuelve siempre los
mismos clips.

Verificado contra el corpus real de 25.139 clips de florete elegibles [medido, runbook 2026-09-06]:

| pedido | clips | torneos | asaltos | tiradores | train | val | test | desvío máx. de partición vs 100 % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,1 % | 26 | 19 / 21 | 26 | 44 | 65,385 % | 30,769 % | 3,846 % | 2,379 % |
| **1 %** | **252** | **21 / 21** | 252 | 219 | 66,270 % | 28,175 % | 5,556 % | **0,216 %** |
| 5 % | 1.257 | 21 / 21 | 1.183 | 383 | 66,269 % | 28,321 % | 5,410 % | 0,137 % |
| 10 % | 2.514 | 21 / 21 | 1.337 | 397 | 66,150 % | 28,361 % | 5,489 % | 0,029 % |
| 25 % | 6.285 | 21 / 21 | 1.369 | 402 | 66,126 % | 28,385 % | 5,489 % | 0,012 % |
| 100 % | 25.139 | 21 / 21 | 1.369 | 402 | 66,132 % | 28,390 % | 5,478 % | 0,000 % |

Dos salvedades honestas: por debajo del 0,1 % hay menos clips que torneos, así que la cobertura baja
(19/21 al 0,1 %); y el balance de etiquetas se mueve un poco en muestras pequeñas (completo 0,5066,
1 % 0,5437, 5 % 0,5219). El 1 % es un subconjunto estricto del 5 % [medido: 252/252].

**Aplica igual con duración completa**, porque el muestreo es sobre el manifiesto y no depende de la
ventana. Lo que sí cambia es qué fracción de la muestra sobrevive a QC: §5.

---

## 3. La ventana: las tres opciones, con su precio medido

`lights.window` da `f0 = max(0, ft − pre·fps)` y `f1 = min(T, ft + post·fps + 1)`, así que 60/60 con
`use_clock=False` es el clip entero para todo el corpus (duración máxima medida 16,71 s, y
`max_video_seconds = 90` lo acota por arriba). **`use_clock` tiene que ir a `False` en cualquier
ventana ancha**: con el reloj encendido el inicio se recorta al «allez», y eso es justo lo que quita
los cuadros de guardia.

Barrido sobre los **15.451 clips con QC utilizable** de `runs_full` [medido 2026-09-09,
`scratchpad/colab/medidas/barrido_ventana.py`]. No re-extrae nada: reevalúa la geometría de
`lights.window` y las banderas de QC ya escritas en `analysis/qc/*.json`.

| candidato | cuadros med. | MiB med. | MiB p95 | rechaza T4 (3,85 GiB) | rechaza High-RAM (6 GiB) | QC bloq. | 2ª luz | guardia |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4.0/0.3 + reloj (**hoy**) | 129 | 680 | 1.554 | 0,00 % | 0,00 % | 5,53 % | 99,05 % | 72,78 % |
| 4.0/0.3 sin reloj | 129 | 680 | 1.554 | 0,00 % | 0,00 % | 5,56 % | 99,05 % | **91,42 %** |
| **`"amplia"` 6.0/1.0** | **210** | 1.107 | 2.515 | 0,00 % | 0,00 % | 6,52 % | **99,99 %** | **97,04 %** |
| 8.0/3.0 | 294 | 1.582 | 3.939 | 4,36 % | 0,49 % | 8,30 % | 99,99 % | 97,04 % |
| 12.0/6.0 | 314 | 1.819 | 5.078 | 10,90 % | 0,50 % | 8,65 % | 99,99 % | 97,04 % |
| **`"completa"` 60/60** | **318** | 1.846 | 5.233 | **11,31 %** | **0,50 %** | **8,69 %** | 99,99 % | 97,04 % |

Y la distribución que hay detrás [medido]: duración de clip mín. 1,00 s, p05 10,00, **mediana 10,48**,
media 11,37, p95 14,70, **máx. 16,71**; ventana guardada hoy mín. 0,33 s, mediana 4,28, media 3,93,
máx. 4,35; **factor duración/ventana mediana 2,45×, medio 2,89×**.

### Las seis razones de la decisión, contrastadas contra la medida

| razón | qué gana de verdad |
|---|---|
| 1. `POST_S = 0,3` deja la segunda luz fuera en el 3,12 % | **`\|dt_ms\|` máximo 520 ms** sobre los 3.331 clips de dos luces (p50 80, p95 267, p99 360; **100 % por debajo de 1.000 ms**) [medido]. **`post_s = 0,6` la captura entera.** La duración completa no añade nada aquí |
| 2. el bloque de 256 columnas agregadas está congelado y lleva el atajo casi entero (+39,65 pp por su cuenta) | **esto sí lo exige la duración completa.** Es la única razón de las seis que no tiene alternativa más barata |
| 3. la ventana de guardia `[t−4,0 : t−3,5]` no existe en el 22,88 % | pasa de **72,78 % a 91,42 % solo apagando el reloj**, al mismo coste exacto de decodificación. Y su techo es **97,04 %, no 100 %**: en ~3 % de los clips el tocado ocurre a menos de 4,0 s del comienzo del vídeo. **La duración completa no puede inventar cuadros que nadie filmó** |
| 4. el transformador de vídeo falló por submuestreo (16 cuadros / 1,2 s = 75 ms de paso, y el suceso decisivo es un desfase de 10-30 ms) | lo exige la duración completa, pero es una vía **cerrada por medida**: VideoMAE no llegó al suelo de 0,8 pp y predice la sede al 97 % |
| 5. tabla del reloj y detección de repeticiones | `pre_s = 6,0` los da en el 97,04 % |
| 6. la estatura gana cuadros de guardia | igual que 3 y 5 |

**Lectura honesta:** de las seis razones, **cinco se satisfacen enteras con `"amplia"` (6.0/1.0)** a
1,63× el coste de hoy y **0,00 % de clips rechazados en cualquier runtime**. La sexta —la única que
exige de verdad el clip entero— es la número 2, y es la que importa, porque el bloque agregado es
donde está el techo. La duración completa es la apuesta de «ya que hay que redescargar, mejor tener
todo», y **eso es una razón válida**; lo que hay que saber es que su precio no es cero y no está en
el número, está en los cuatro sitios del §5.

### La rodaja de entrenamiento, y la aritmética que la decide

Una rodaja `(pre_s, post_s)` a 30 fps ocupa `round(pre_s·30) + round(post_s·30) + 1` cuadros, y el
Store solo guarda `T_max`, **alineados por el final**:

| rodaja | cuadros | `T_max` necesario | cabe en 135 | guardia `[t−4,0:t−3,5]` | 2ª luz |
|---|---:|---:|:---:|:---:|---:|
| `"historica"` 4.0/0.3 | 130 | 135 | **sí** | sí | 99,05 % |
| **`"recomendada"` 4.0/0.6** | **139** | **140** | no | **sí** | **100,00 %** |
| `"ancha"` 6.0/1.0 | 211 | 211 | no | sí | 100,00 % |

**El detalle que hay que ver antes de elegir:** 6,0 + 1,0 = 7,0 s son **211 cuadros**, y
`T_MAX = 135` son 4,467 s. Con `T_max` sin subir, el Store se queda los últimos 135 de esos 211, o
sea **`[t−3,47 : t+1,0]`**, y **la ventana de guardia se queda fuera igual** — justo lo que la rodaja
venía a proteger. **Poner la rodaja no hace por sí solo que `T_MAX` deje de estorbar: hay que subir
`T_max` a la vez.** `preflight.plan_rodaja(nombre, t_max=N)` comprueba las cuatro condiciones y
levanta.

Subir `T_max` toca `store.T_MAX` y el validador de `TrainingRecipe` (`8 ≤ T_max ≤ 135`). **Los dos
ficheros están en ninguna de las cuatro listas protegidas** [medido, §5.0], y
`feature_contract.MAX_TIMESTEPS = 256` ya lo permite, así que **no se invalida ningún bundle
portable**. Lo que sí cambia es la entrada de la red, y eso exige reentrenar y reexportar — que es
lo que se va a hacer igual.

**Por qué `"recomendada"` y no `"ancha"`.** 4.0/0.6 es la rodaja **mínima** que cierra la única
brecha **medida** —`|dt_ms|` máximo 520 ms, así que `post_s = 0,6` captura la segunda luz al 100 %—
y conserva la ventana de guardia entera. `"ancha"` también funciona con `T_max = 211`, pero **cambia
lo que ve la red** (7 s en vez de 4,3) y **ese cambio no está medido**: `pre_s = 4,0` se eligió
porque 2,5 s recortaba el inicio de la acción final en el 24 % de los clips, y nadie ha comparado
6,0 contra 4,0 sobre la red. La atención es O(T²), así que 211 cuadros son ~2,5× su coste. Si se
prueba, que sea **como un brazo más, pareado** contra `"recomendada"`, y con el suelo de +0,4 pp del
nulo neuronal delante.

### El reloj es un mando aparte, y es la ganancia más barata que hay

`USAR_RELOJ = False` lleva la ventana de guardia de **72,78 % a 91,42 %** al **mismo coste exacto**
de decodificación: 129 cuadros, 680 MiB, +0,03 pp de clips bloqueados. No lo compra ninguna otra
palanca de la tabla y no cuesta nada.

> **AVISO DE INTERACCIÓN.** Hay trabajo en curso para **conectar** la tabla del reloj. Las dos
> decisiones interactúan —una cambia de dónde sale `f_allez` y la otra si se usa—, así que hay que
> medirlas **juntas, en un factorial 2×2**, no una encima de la otra. Si el otro cambio ya está
> dentro cuando se ejecute esto, hay que declarar las dos configuraciones junto al resultado.

---

## 4. Recursos: qué está medido y qué no

### La pared de la ventana decodificada, y por qué la duración completa la reabre

El planificador presupuesta `cuadros × W × H × 3 × DECODE_COPY_FACTOR`. El runbook del 2026-09-06
bajó `DECODE_COPY_FACTOR` de 3 a 2 después de medir un clip real con muestreador de memoria [medido]:

| | valor |
|---|---|
| Clip | 1920×1080 @ 49,99 fps, ventana de 265 cuadros |
| Una copia BGR de la ventana | 1,54 GiB |
| **Pico de RSS medido sobre la línea base** | **1,66 GiB** |
| Estimación de planificación antigua | 4,61 GiB |
| **Factor de sobreestimación** | **2,77×** |

El coste real es **1,08× una copia, no 3×**. Con factor 2 quedan ~85 % de margen sobre la medida.

**Y aquí está la corrección importante.** `resource_plan` fija
`max_decoded_bytes = min(6,0 GiB si cuda else 3,0 , RAM_libre × 0,35)`. Con la ventana desplegada
(129 cuadros) eso no rechaza nada en ningún runtime. **Con duración completa (318 cuadros) rechaza el
11,31 % de los clips en un T4 estándar y el 0,50 % en High-RAM** [medido, barrido del §3]. El rechazo
está **correlacionado con la resolución** (los 1920×1080@50 son los que caen), o sea que **sesga la
selección**: es exactamente el fallo que el runbook eliminó, y la duración completa lo reabre.

> **La frase del runbook anterior *«You no longer need High-RAM to avoid resolution bias»* deja de
> ser cierta con duración completa. Con `VENTANA = "completa"` hace falta un runtime High-RAM.**

El cuaderno lo comprueba y lo dice antes de descargar nada (`preflight.plan_ventana`), y ofrece las
tres salidas por orden: (a) cambiar a GPU + High-RAM, (b) `VENTANA = "amplia"`, que no rechaza nada,
(c) subir a mano el tope de 6,0 GiB, sabiendo que el pico real medido es 1,08× una copia.

### Los fallos se cuentan y se gritan, nunca se tiran en silencio (sin cambios)

`run_data` agrupa cada clip no-ok por motivo (`decoded_window_planning_budget`,
`video_duration_or_resolution`, `download_budget_exhausted`, `no_lights_detected`, `pose_failed`, …),
los devuelve en `run_report["failure_reasons"]` con ejemplos de `clip_id`, e imprime un `ATTENTION:`
de una línea que **avisa explícitamente de que un recuento de fallos correlacionado con la resolución
significa una selección sesgada**. Con duración completa eso deja de ser una precaución teórica.

### Los mandos derivados (sin cambios)

| mando de antes | ahora |
|---|---|
| `max_download_total_bytes` (2 GiB, abortó una corrida de Volpi hacia el clip 677) | `resource_plan(clips=N)` lo escala a `N × clip medio × 1,5` |
| `download_workers` / `queue_size` | de núcleos y RAM libre (1 si RAM < 3 GiB, hasta 4) |
| `cpu_clip_limit` (tope silencioso en 6) | sigue siendo guarda, pero el error nombra el tope, el número pedido, las horas de CPU estimadas y la ruta exacta del menú de Colab |
| `CHECKPOINT_EVERY` | `checkpoint_interval(n) = clamp(ceil(n/20), 5, 500)` → ~20 instantáneas por corrida a cualquier tamaño |
| `DOWNLOAD_MISSING` / `DELETE_OWN_DOWNLOADED_MP4` | los dos automáticos si no se aportan clips locales, así que el disco queda plano |
| `POSE_PROFILE`, `TORCH_THREADS`, `INSTALL_MISSING` | derivados / siempre activos |
| `N_CLIPS` | del porcentaje |
| épocas, paciencia, `d_model`, lote (nb02) | `training_shape(workspace, environment, plan)` |

Cada uno sigue respetando su variable `FENCINGLY_*`. Se añade `FENCINGLY_VENTANA`.

### La GPU, con las cifras medidas y las que siguen extrapoladas

**[medido] en RTX 3060 de 12 GB, torch 2.14.0+cu126:**

| medida | valor |
|---|---|
| VideoMAE-Base, tokens por clip | 1.152 |
| VideoMAE-Base, latencia | **18-69 ms/clip** |
| VideoMAE-Base, VRAM a lote 32 en fp16 | **1.417 MiB** |
| Flujo óptico de Farnebäck, par de 128×128 | **7,2 ms** |
| Recalcular flujo desde recortes contra ruta de vídeo | **394 ms/clip contra 1.250**, bit a bit idéntico |
| Backend `math` forzado, dos procesos, mismo pliegue | `max \|p1−p2\| = 0,000e+00`, sin coste de tiempo |
| Nulo neuronal (cambiar solo el backend) | +0,36 pp, 327 de 3.047 decisiones distintas |
| Determinista contra archivado, 3 semillas | −1,18 / −0,42 / −1,12 pp, media −0,91, **no uniforme: hay que remedir, no restar** |

**Los lotes se derivan de la VRAM detectada, no se fijan**, que es lo que ya hace `training_shape`
(`train_batch = max(1, min(64, int(vram_libre_gb × 4)))`). Colab da GPUs mayores que una 3060, así
que fijar un lote a mano dejaría VRAM sin usar. La medida de VideoMAE (1.417 MiB a lote 32 en fp16)
es la referencia para saber cuánto cabe.

**Sigue [extrapolado]:** el coste de **YOLO-pose por clip en T4, L4 y A100**. De esas tres GPU no
había ninguna; los multiplicadores son suposiciones (12-25× un hilo de CPU para T4, 40-80× para
A100). La columna de CPU sí está medida.

### Costes de etapa por clip, y qué pasa al ensanchar la ventana

Medidos sobre tres clips reales en un escritorio de 24 núcleos, YOLO11s @640 en CPU [medido, runbook
2026-09-06]. La tercera columna es lo que le hace la duración completa, con el factor de cuadros
medido **2,47×** (129 → 318 mediana):

| etapa | s/clip hoy | reparto | corre en | con duración completa |
|---|---:|---:|---|---|
| `B1` detect_window (YOLO) | 20,42 | 67,7 % | **GPU** en producción | ×2,47 |
| `A1` barrido de luces/overlay del vídeo entero | 5,58 | 18,5 % | CPU | **igual** (ya barre el vídeo entero) |
| `A2` decodificar la ventana | 1,72 | 5,7 % | CPU | 0 a ×2,47, según `TAIL_BUDGET_MB` (abajo) |
| `C1` estimación de paneo | 0,91 | 3,0 % | CPU | ×2,47 |
| `B3` ensamblado | 0,71 | 2,4 % | CPU | ×2,47 |
| `C2` detección de duplicados | 0,54 | 1,8 % | CPU | ×2,47 |
| `A3` audio | 0,12 | 0,4 % | CPU | **igual** |
| `B2` score_tracks + `B4` fill_gaps | 0,16 | 0,5 % | CPU | ×2,47 |
| QC + medidas + cola de movimiento | 1,15 | — | CPU | ~×2 |

**`TAIL_BUDGET_MB = 1024` con `TAIL_MIN_FRAC = 0.45`.** La pasada de luces ya decodifica el clip
entero y se queda con los últimos cuadros completos que caben en ese presupuesto; si la ventana de
poses cae dentro, **`A2` no decodifica otra vez** y los píxeles son idénticos bit a bit. Con la
ventana desplegada eso funciona casi siempre. Con duración completa la ventana **es** el clip entero,
así que 1024 MB solo alcanzan para 720p a 25/30 fps (~314 cuadros × 2,76 MB ≈ 867 MB, cabe) y no para
720p50 ni para nada a 1080p. **Subir `TAIL_BUDGET_MB` a ~3.500 en un runtime High-RAM elimina `A2`
entero** y no aumenta el pico, porque el clip ya estaba decodificado en esa misma pasada.

**CPU irreducible que ninguna GPU quita** [derivado de las medidas de arriba]:

| | s/clip |
|---|---:|
| hoy (ventana desplegada) | **10,89** |
| duración completa, `TAIL_BUDGET_MB` subido (`A2` gratis) | **13,73** (1,26×) |
| duración completa, `TAIL_BUDGET_MB` sin tocar (`A2` re-decodifica) | **17,98** (1,65×) |

### ¿Cabe el corpus de florete completo en 8 h? Con duración completa, no

25.139 clips en 8 h (28.800 s) exigen **1,145 s/clip**. El caudal es
`max(pose_en_gpu, segundos_cpu / trabajadores_paralelos)`.

| runtime | vCPU | CPU-bound s/clip (duración completa) | pose s/clip [extrapolado] | s/clip | **florete completo** |
|---|---:|---:|---:|---:|---:|
| Colab T4 estándar | 2 | 13,7-18,0 | 3,5-7,2 | ~13,7-18,0 | **~96-126 h**, y además rechaza el 11,31 % |
| Colab L4 High-RAM | 8 | 1,96-2,57 | 2,5-4,9 | ~2,5-4,9 | **~17-34 h** |
| Colab A100 High-RAM | 12 | 1,25-1,63 | 1,0-2,2 | ~1,25-2,2 | **~8,7-15,4 h** |
| RTX 3060 local, 24 núcleos | 24 | 0,60-0,78 | 1,7-3,5 | ~1,7-3,5 | **~11,9-24,4 h** |

**Veredicto: la duración completa sobre el corpus de florete completo NO cabe en una sesión de 8 h,
ni en A100 High-RAM.** Hay tres formas de convivir con eso, y las tres están soportadas:

1. **Dos o tres sesiones con reanudación.** Es lo que la Fase 1 ya hace bien: reanuda por clip y
   pierdes como mucho el clip en vuelo. Ver §6 para lo que sí rompe la reanudación en Colab.
2. **Bajar el porcentaje.** Un 25 % son 6.285 clips, ~2,2-3,8 h en A100 High-RAM, y reproduce las
   fracciones de partición del corpus con 0,012 pp de desvío.
3. **`VENTANA = "amplia"`**, que a 210 cuadros deja el CPU irreducible en ~12,1 s/clip
   (1,11×) y el corpus completo en ~7,7-11 h en A100.

Para calibrar: **1 % = 252 clips** ≈ 6-11 min en A100 High-RAM con duración completa.

### La palanca que más compraría, y sigue sin tocarse

`A1`, el barrido de overlay/luces del vídeo entero, es **5,58 s/clip** y con duración completa pasa a
ser el **41 % del presupuesto de CPU irreducible**. Es CPU pura, así que no se encoge pagando una GPU
mayor, y decodifica el vídeo que `A2` vuelve a decodificar. Compartir una sola pasada de decodificado
—o acotar el barrido a una búsqueda gruesa antes de una fina— sigue siendo la optimización de mayor
palanca que queda. **No se intentó:** vive en `lights_generic.py` y cambiarlo arriesga cambiar qué
cuadro se llama el tocado, lo que cambiaría en silencio cada medida de aguas abajo. Necesita su
propia tarea con una comprobación de acuerdo antes/después sobre los cuadros de tocado detectados.

---

## 5. Los cuatro precios de la duración completa que no están en el número

Los cuatro son mecánicos, están medidos y **ninguno da error**. La rodaja de entrenamiento quita
tres; el primero es el que la hace obligatoria.

### 5.0 El coste de cada arreglo, medido antes de proponerlo

Cuatro listas protegen cosas distintas y confundirlas es caro. Medido con
`preflight.coste_de_las_listas()`:

| fichero | los 20 | MOTION | huella coach | bundles | coste de cambiarlo |
|---|:---:|:---:|:---:|:---:|---|
| **`store.py`** | – | – | – | – | **ninguno** |
| **`training_recipe.py`** | – | – | – | – | **ninguno** |
| **`colab_workflow.py`** | – | – | – | – | **ninguno** |
| `clip_quality.py` | – | – | **sí** | – | re-cachear el panel de coach |
| `coach_panel.py` | **sí** | – | sí | – | **re-materializar los 23.091 artefactos** |
| `measurement_artifacts.py` | **sí** | **sí** | sí | – | **re-materializar los 23.091 artefactos** |
| `feature_contract.py` | – | – | – | **sí** | **invalida todos los bundles portables** |

`store.py` es el sitio **más barato de todo el conducto** para poner la rodaja: no está en ninguna
de las cuatro. Solo aparece en `feature_contract.source_provenance()`, que se graba **dentro** de
`store.json`, así que el único efecto es que un Store nuevo declara una procedencia nueva — que es
lo correcto, porque el protocolo es nuevo.

### 5.0.bis ROMPE LA ALINEACIÓN, y es el peor de los cuatro

`build_store` hace `t = min(T_max, length); cut = length − t; array[i, -t:] = value[-t:]`: alinea por
el **final del array** y trunca la cabeza. `feature_contract` lo declara: `"alignment": "right"`,
`"truncation": "tail"`. Con la ventana desplegada eso es correcto, porque el array **termina** 0,3 s
después del tocado. Con un NPZ de duración completa el array termina al final del **CLIP**, y el clip
sigue después del tocado:

| cola tras el tocado (15.453 clips) [medido] | s |
|---|---:|
| mín. | 0,03 |
| p05 | 0,93 |
| **mediana** | **2,33** |
| media | 2,62 |
| p95 | 5,23 |
| máx. | 14,57 |

y supera 1,0 s en **14.479 clips (93,70 %)**, y 2,0 s en 9.716 (62,87 %).

Así que con `T_MAX = 135` el Store guardaría, para el clip mediano, **`[t−2,17 : t+2,33]`** en vez de
`[t−4,2 : t+0,3]`. Consecuencias, ninguna con error:

* **el último cuadro deja de ser el tocado**, que es la convención que la red asume;
* la **ventana de guardia `[t−4,0 : t−3,5]` desaparece del Store en ~94 % de los clips** — o sea
  justo lo que la duración completa venía a garantizar;
* `store_truncated` se pone a True, y su único efecto declarado es enmascarar las etiquetas
  auxiliares (§5.3).

> **Esto es una regresión, no un empate.** Por eso la rodaja de entrenamiento **no es una
> optimización: es obligatoria**.

**Y se resuelve sin cambiar ninguna fuente.** `store.build_store` carga cada clip con
`D.load_sample(npz)` y no acepta ningún recorte, así que la celda del Store lo envuelve:

```python
STORE_RESULT = PF.construir_store_con_rodaja(
    WORK, GLOBAL_REGISTRY, rodaja_nombre=RODAJA_ENTRENAMIENTO, t_max=T_MAX, weapon=WEAPON)
```

El envoltorio se retira en el `finally` y se **verifica la identidad del objeto**, no solo el nombre.

*Por qué es seguro aquí y no en cualquier sitio:* `dataset.py` está en
`portable_contract.SEMANTIC_SCOPE`, y esa validación compara el objeto de código **en memoria** con
el compilado de la fuente, así que con el envoltorio puesto cargar o guardar un bundle portable
fallaría. **Comprobado que ni el 01 ni el 02 canónicos mencionan `portable_contract`, `load_bundle`
ni `save_bundle` ni una vez**, y el Store se construye en el 01 mientras los bundles se tocan en el
02, en otra sesión. Sobre un NPZ de ventana desplegada el recorte es un **no-op verificado**
(T = 129 antes y después).

Y la verificación va detrás: `preflight.alineacion_del_store(store_dir, post_s_esperado=P)` lo mide
**después** de construir el Store y **levanta** si el tocado no quedó al final. Usa solo campos que
`sample_meta.jsonl` ya escribe (`prepared_time_start_s`, `prepared_n_frames`, `t_touch_s`), **no**
`analysis_touch_frame`, que en los Store reales sale `None` (comprobado: 0 de 211). Verificado sobre
un Store real de ventana desplegada: **N = 14.437, mediana 0,267 s contra `post_s` 0,3,
`store_truncated=True` en 0 de 14.437** — pasa. **La puerta funciona aunque el envoltorio no se
use**, así que la corrida no puede producir un Store desalineado en silencio.

### 5.1 El recorte a la ventana de análisis pasa de inerte a NECESARIO

`coach_panel._ventana_de_analisis` recorta la muestra a `[tocado − 4,0 : tocado + 0,3]`. Todo el
camino de medidas asume que el cuadro 0 está `pre_s` antes de la luz: sobre eso se apoyan los
**veinticinco sitios** que miden la guardia como `X[:GUARD_S*fps]`, la escala por clip de
`features.fencer_features` y la de `armwork._fencer_signals`. Su propio docstring declara que con NPZ
de duración completa la suposición se rompe **en silencio**: medido sobre 20 clips con las dos
versiones, `fore_ang_guard` se desviaba **hasta 152 grados** y la escala del torso un 1-2 %, **sin
ninguna bandera**.

`NativeProcessor.__call__` llama a `export_measurements(D.load_sample(npz_path), …)` con la muestra
**cruda**. Con la ventana desplegada eso era inerte porque `poses.shape[0] == n_frames_window` en los
15.452 clips del corpus: **no había ni un NPZ de duración completa**. Con duración completa deja de
serlo.

**RESUELTO durante esta sesión, por el agente que trabajaba en esos ficheros.** Estado comprobado
programáticamente con `preflight.parche_de_recorte`:

| camino | recorta |
|---|---|
| `coach_panel.build_video_stats` (LOTE) | **sí** |
| `measurement_artifacts.export_measurements` (por CLIP, el que usa Colab) | **sí** (10:49) |

**Y cambió la huella, que es la consecuencia correcta y hay que declararla.**
`measurement_artifacts.py` es uno de los 20 de `MEASUREMENT_SOURCES`, así que el parche
**re-materializa a propósito** [medido en el momento en que entró]:

| protocolo | medidas | movimiento |
|---|---|---|
| histórico (sin recorte) | `0b87d014…3c4f` | `994c874b…2658` |
| **duración completa (con recorte)** | **`3e1dbe0a…9827`** | **`6c9fa66b…3ce7`** |

Consecuencias, las dos importantes:

* **los 23.091 artefactos del protocolo histórico dejan de ser caché válida** y hay que
  re-materializar;
* es **otra razón, además de la ventana**, por la que el número de esta corrida **no se resta del
  70,04 % ni del 73,88 %**. Son dos cadenas de medida distintas por dos motivos independientes.

`preflight.huella()` conoce las dos, dice cuál ha medido, espera la de duración completa y
**levanta** si mide la histórica —porque entonces el recorte no está dentro y la corrida no puede
ser de duración completa— o una tercera, que significaría que una fuente cambió y nadie lo declaró.

### 5.2 La duración completa cuesta 488 clips de entrenamiento (3,16 %)

`clip_quality` marca `cut_in_window` y `replay_in_window` como **bloqueantes** (peso 0,0 en
`WEIGHTS`), y `colab_workflow.build_current_store` **descarta toda fila con `qc_blocking` no vacío**.
Esas dos banderas se evalúan sobre la ventana efectiva, así que al ensanchar la ventana entran cortes
de cámara que hoy quedan fuera. Medido sobre los 15.451 clips
[`scratchpad/colab/medidas/qc_duracion_completa.py`]:

| | ventana de hoy | duración completa |
|---|---:|---:|
| clips con ≥1 corte de cámara | 828 (5,36 %) | — |
| clips con ≥1 repetición | 3 (0,02 %) | — |
| `cut_in_window` | 316 (2,05 %) | **828 (5,36 %)** |
| `replay_in_window` | 1 (0,01 %) | 3 (0,02 %) |
| otras bloqueantes (no dependen de la ventana) | 586 (3,79 %) | 586 (3,79 %) |
| **BLOQUEADO** | **853 (5,52 %)** | **1.341 (8,68 %)** |
| **entran al Store** | **14.598** | **14.110** |

**Las repeticiones no son el problema** (3 clips). Los cortes de cámara sí.

**Y la rodaja de entrenamiento NO lo arregla sola.** Es la parte del diseño «extraer completo,
entrenar sobre una rodaja» que no sale gratis, y conviene tenerlo claro: **el QC se calcula en la
EXTRACCIÓN, sobre la ventana GUARDADA, y queda congelado en `analysis/qc/<clip>.json`.** La cadena
exacta: `NativeProcessor.__call__` llama a `CQ.assess_clip` durante la extracción (línea 842-847);
`assess_clip` obtiene la ventana con `effective_window`, que la toma de `time_s` del NPZ, o de
`f0`/`f1`, o de `t_window_start_s`/`t_window_end_s` — **los tres son la ventana realmente
guardada**, y `post_s_requested` solo se usa de último recurso; `_flatten_vm` evalúa las dos
banderas sobre esa ventana; `build_current_store` descarta lo que tenga `qc_blocking` no vacío.
Elegir una rodaja **después** no cambia nada de eso.

### ARREGLADO. Y es corrección de ámbito, no optimización

El corte de plano que bloquea esos 488 clips está **fuera de la ventana de entrenamiento**: se
descartaba un clip por un defecto que **no está en los datos que el modelo va a ver**. Eso no es un
criterio de calidad, es un ámbito equivocado. Y pesa más que su 3,16 %: la extracción trajo
**+30,4 % de pares**, y perder un 3,16 % por un ámbito mal puesto es tirar parte de lo ganado.

El arreglo, en `clip_quality._flatten_vm`, acota **solo las dos banderas bloqueantes**:

```python
an0, an1 = max(w0, t_touch - ANALYSIS_PRE_S), min(w1, t_touch + ANALYSIS_POST_S)   # 4.0 / 0.3
d["cut_in_window"]    = any(an0 < c <= an1 for c in cuts)
d["replay_in_window"] = any(x["t0"] <= an1 and x["t1"] > an0 for x in reps)
```

`n_cuts_before/after`, `replay_before_touch`, `replay_after_touch`, `freeze_in_window` y `n_cuts`
**siguen describiendo el clip completo**, que es lo que se quiere para el informe de calidad. Solo
cambia el ámbito de las dos que deciden la admisión.

**Las tres condiciones, verificadas antes de tocar nada, y una no cuadraba:**

| condición | resultado |
|---|---|
| `clip_quality.py` fuera de los 20 | **sí**, y fuera de `MOTION_SOURCES` y de `SEMANTIC_SCOPE`; solo en `FINGERPRINT_SOURCES` |
| no-op sobre el corpus actual | **0 discrepancias en 15.451 clips** (`medidas/noop_clip_quality.py`) |
| batería en 1.321 / 0 / 0 / 31 | **primero dio 1.233 / 0 / 34 / 16** — la había ejecutado con `.venv-gpu`, que no tiene `fastapi`. Repetida con `.venv`: **1.321 / 0 / 0 / 31, `successful: true`** |

**Verificado después de aplicarlo:**

* **la huella NO se mueve**: `3e1dbe0a…9827` y `6c9fa66b…3ce7`, o sea que los 23.091 artefactos de
  medida siguen siendo caché válida — que es la consecuencia de estar fuera de los 20;
* la huella del panel de coach **sí** cambia (`_source_fingerprint`), que es el coste esperado y
  recuperable: se recalcula;
* no hay colisión de nombres — el `a0` que ya existía para `freeze_in_action` (anclado en el onset)
  queda intacto, y por eso las variables nuevas se llaman `an0`/`an1`. **Mi primer borrador del
  parche usaba `a0`/`a1` y habría sombreado esa variable.**

Sobre el no-op, con precisión: la ventana de análisis coincide con la guardada en el **24,08 %** de
los clips y se desvía **hasta 0,267 s** en el resto, así que **no es una identidad matemática, es un
no-op empírico**: en ese margen no cae ningún corte, medido clip a clip sobre los 15.451.

**Y lo que recupera, medido sobre los 15.451:**

| | bloqueados | entran al Store |
|---|---:|---:|
| hoy, ventana desplegada | 853 (5,52 %) | 14.598 |
| almacén ancho, **sin** el arreglo | **1.341 (8,68 %)** | 14.110 |
| almacén ancho, **con** el arreglo | **858 (5,55 %)** | **14.593** |

**Recupera 483 de los 488**, y queda **+5 clips bloqueados frente a hoy**, no 0. Los 5 son clips
donde un corte cae en `[t−4,0 : t+0,3]` contado desde el cuadro 0 pero no caía en la ventana
guardada, porque el `w0` desplegado iba recortado por el reloj o por truncamiento entero. Es el
número exacto y no lo redondeo a 488.

Nota menor de la misma familia: con `pre_s` grande, `window_truncated_start` (peso 0,8, **no**
bloqueante) se enciende en todos los clips, así que **`qc_score` baja ×0,8 en bloque y tampoco es
comparable entre protocolos**. No pierde ningún clip, porque el filtro es sobre `qc_blocking`.

### 5.3 Las etiquetas auxiliares se apagan en silencio

`store_truncated = prepared_n_frames > T_max` pasa a **True en casi todos los clips** con duración
completa, y tanto `attach_aux` (`store.py:244`) como `Store` (`store.py:417`) exigen
`not store_truncated` para admitir una etiqueta auxiliar. **Todas las etiquetas auxiliares quedarían
enmascaradas —código −1, mask 0, conf 0— sin ningún aviso.** Hoy es **0 de 14.437 muestras**
[medido], así que el problema está latente y se activaría de golpe.

**La rodaja lo arregla**, porque una rodaja que cabe en `T_max` no trunca.
`preflight.alineacion_del_store` cuenta `store_truncated` y lo dice.

### 5.4 La red no ve más contexto por la duración completa

Tres topes, encadenados:

| tope | valor | dónde | cuesta cambiarlo |
|---|---:|---|---|
| `store.T_MAX` | **135** | por defecto de `build_store`, y `colab_workflow` no pasa otro | nada |
| `TrainingRecipe.T_max` | valida **8 ≤ T_max ≤ 135** | `training_recipe.py:51` | nada |
| `feature_contract.MAX_TIMESTEPS` | **256** | `feature_contract.py:22` | **todos los bundles** |

El Store guarda `T_max` cuadros alineados por el final, así que **lo que ve la red lo decide la
rodaja, no el almacén**. Con `"recomendada"` son 139 cuadros (4,63 s) contra los 130 (4,33 s) de hoy:
casi lo mismo, y a propósito.

**Subir `MAX_TIMESTEPS` no hace falta para ninguna de las tres rodajas** (la más ancha pide 211 ≤ 256)
y conviene no tocarlo: está en `portable_contract.SEMANTIC_SCOPE` **con alcance de módulo completo**,
así que cambiarlo mueve `configuration_sha256` e **invalida todos los bundles portables**. El síntoma
está documentado: `ContractError: Portable: semantica AST/configuracion no conforme`, la app corre en
silencio sin modelo neuronal, `judge` responde `model_unavailable` y el único rastro es
`STORE.model_error`.

> **Traducido: la duración completa es una inversión en el camino de MEDIDAS (bloque agregado,
> guardia, estatura, reloj) y en no tener que volver a descargar 93 GB. No es un cambio de entrada
> del modelo secuencial, y no hay que esperar que el número suba por ella.** Que eso quede escrito
> es parte del encargo: si el número baja, no es un fallo; si sube, no es por más contexto.

---

## 5.bis La tabla del reloj: la ganancia más grande, y no cuesta huella

### La posición absoluta de pista ya está desbloqueada, y no hay que cambiar nada

`piste_abs.allez_anchor` (**`piste_abs.py:216`**) hace:

```python
clk = parse_list(_get(row, "clock_running_s")) or meta.get("clock_running_s")
```

**Lee la fila del índice antes que el meta.** Verificado leyendo la línea. Consecuencia: el reloj le
llega por `npz/index.csv` —**sin reextraer, sin NPZ nuevos y sin cambiar la huella**, porque
`piste_abs.py` no está en ninguna de las cuatro listas protegidas [medido]—.

**Y la columna ya la emite el conducto actual.** `clock_running_s` es la **columna 45 de las 119** de
`npz/index.csv` en el corpus de hoy. Así que esto **no es un cambio que hacer, es un hecho que
comprobar** por espacio de trabajo. Medido sobre los 15.452 clips ok|cached con `.npz`:

| | clips |
|---|---:|
| con `clock_running_s` legible | 9.865 (63,84 %) |
| − allez censurado (`clk[0] ≤ 0,05`) | −2.640 |
| − allez tras el tocado | −64 |
| **ancla de allez utilizable** | **7.161 (46,34 %)** |

Hoy `x_abs`, `dist_m` y `dist_allez_*_m` son **NaN en el 100 %** de esas extracciones, así que son
**7.161 clips que ganan posición absoluta de pista a coste cero**. `preflight.columna_reloj(WORK)` lo
comprueba y lo cuenta en la corrida real, en vez de suponerlo.

### Apagar la rama del reloj

`USAR_RELOJ = False` lleva la cobertura de guardia de **72,78 % a 91,42 %** al mismo coste exacto de
decodificación (§3). Es la ganancia más barata del barrido.

### Lo que NO compra, sin inflarlo

* **anclar no añade ni un clip**: lo que ancla es la lámpara;
* **marcar repeticiones no quita ninguno**: ya están fuera por política.

### El ancla de allez: lo que la duración completa **rompe** y la rodaja **recupera**

Esto es lo más importante de la sección, y corrige el argumento con el que llegó.

`calibration.allez_scale` (calibration.py:333-341) exige que la ventana **empiece** en el arranque
del reloj:

```python
t_win0 = meta["f0"] / meta["fps"]
lag    = t_win0 - clock_running_s[0]
if abs(lag) > ALLEZ_TOL_S:   # 0,7 s
    reject = "window_not_at_clock_start"
```

Y **`f0` está grabado en el NPZ**, así que ni la fila del índice ni el meta lo alcanzan. Las tres
configuraciones, **medidas sobre 6 clips reales con allez legible**:

| `f0` | lag medido | pasan |
|---|---|---:|
| `tocado − 4,0 s` (lo desplegado hoy) | −1,83 a +2,80 s | **2 de 6** |
| **`0` (DURACIÓN COMPLETA)** | −1,56 a −9,90 s | **0 de 6** |
| **`f_allez` (la vista del allez)** | **+0,50 s exacto** | **6 de 6** |

> **La duración completa por sí sola MATA `allez_scale`**, porque la ventana empieza en el cuadro 0 y
> el allez cae 1,6 a 9,9 s dentro. No es neutral: es peor que hoy.
>
> **Y `preflight.vista_allez` lo recupera entero**, con lag +0,50 s por construcción (el
> `CLOCK_BIAS_S` de `lights.allez_frame`), holgado bajo la tolerancia de 0,7. **La duración completa
> es la única configuración que permite elegir el ancla después**, así que es la única en la que se
> pueden tener las dos vistas a la vez.

Verificado en el ensayo sobre NPZ de duración completa reales: `lag_s` = 0,481 / 0,497 / 0,496, y el
primer clip da `m_per_x = 7,236` con `reject=None`. Los otros dos rechazan por
`not_at_maximum_separation_phrase_already_started`, que es la compuerta de **contenido** (los
tiradores tienen que estar a separación máxima), no la de ventana: **la ventana ya no es el problema**.

**Son dos vistas distintas y las dos son necesarias:**

* rodaja anclada en el **tocado** → el Store y el camino de medidas (los 25 sitios de guardia asumen
  que el cuadro 0 está `pre_s` antes de la luz);
* vista anclada en el **allez** → solo `allez_scale`, para recuperar la escala métrica.

Mezclarlas rompería una de las dos.

**Y cuando el allez entra, mueve de verdad** [medido por el agente del reloj]: la escala métrica
cambia una mediana del **4,00 %** en logaritmo, con el **91,7 %** moviéndose más del 1 % y el 38,9 %
más del 5 %. Un 4 % de escala propaga a **toda medida métrica de ese clip** — distancias, posición
absoluta de pista y estatura. Con `f0 = f_allez` los clips donde el allez entra en la escala pasan de
**42 a 343 de 892 (38,5 %)**, un factor **8,2**, y el motivo de que hoy no entren es
`window_not_at_clock_start` en el **88,0 %**. Descontando la compuerta de contenido en la misma
proporción, reextraer daría **~290-300 clips con escala de allez corregida**.

> **Su límite, y hay que decirlo:** es una carga medida sobre la **escala**, no sobre el acierto de
> prioridad. La pregunta de acierto sigue sin responder y esto no la responde.

### `reloj_parado_todo_el_clip`: definición recibida, implementada llamando al lector

La definición llegó (`scratchpad/reloj/DEFINICION_RELOJ_PARADO.md`, 10 secciones) y el paquete la
implementa **llamando al lector de referencia**, no reinterpretándola: `preflight.reloj_contrato()`
importa `reloj.py` y `ancla_region.py` y emite los **12 campos del contrato de la §8** con sus nombres
exactos, para que una fila producida en Colab y una producida por `trabajo/paso1.py` signifiquen lo
mismo y se unan por `clip_id`. **Si los módulos no están, levanta en vez de aproximar.**

Y por qué no se reimplementa: el documento mide dos formas de producir «un número distinto con el
mismo nombre». **Submuestrear** cambia `frac_corre` en **5 de 6** clips con paso 2, y uno **cambia de
clase**. Y un **umbral único** en vez de por perfil llevó la prevalencia de 16,4 % a **53,1 %**, con
`stpetersburgfoil2019` en 40 de 40 en vez de 7 de 40, porque el umbral estaba calibrado para una caja
mucho mayor. Copiar la lógica aquí sería darle otra oportunidad a ese error.

Del contrato, dos campos que no son opcionales: **`reloj_frac_corre` va siempre junto a la bandera**,
porque el corte en 0,05 es una decisión y alguien querrá moverlo; y **`reloj_parado_todo_el_clip` va
vacío cuando es indefinida** (clip ilegible, serie corta, o sede sin perfil validado) — escribir
`false` afirmaría que el reloj corría, y eso no se ha medido.

**Y mi cero era correcto, con su explicación:** `clock_running_s = [inicio, fin]` es el intervalo en
que el reloj **sí** corre, con `inicio = primer_tic_regular − 1 s`, y solo se construye si hay un
hueco de 0,7 a 1,4 s. Todo intervalo que existe mide **≥ 1 s por construcción**, así que
`|fin − inicio| < 0,05` no podía dar nada. Un reloj parado no da un intervalo degenerado: da **NULL**.
Y el atajo `clock_running_s IS NULL` **tampoco** vale, porque NULL mezcla tres causas: reloj parado
(lo que se quiere), cajas de nombre no localizadas (hueco estructural de 5 sedes) y recorte ilegible
(`qatar`, `incheon`, indistinguible).

**Prevalencia, contada:** 15,1 % [12,5, 18,2] con ancla FIE (n=602) contra 16,5 % [12,0, 22,3] con
ancla de región (n=200). Los intervalos **se solapan por completo**, así que es una propiedad del
**corpus entero**, no de las cinco sedes de región. Control de falsos **2,2 % [1,2, 3,9]** (Wilson,
n=500), y **solo cubre el perfil `fie`**; los otros cuatro se respaldan en revisión visual de los
dígitos ampliados.

**Y el incremento ya está contado, no proyectado: 892 clips** medidos sobre 1.537 leídos con éxito
(~1.691 al extrapolar). Las tres tasas —**83,1 %** con reloj, **58,0 %** con allez utilizable,
**17,0 %** parado— caen dentro del intervalo de la muestra previa de n=200, así que están validadas
por dos vías.

### Dos índices distintos, y leer el equivocado engaña

El `npz/index.csv` **vivo** lleva `clock_running_s` (columna 45 de 119), pero el que escribe
`extrae.py` tiene **47 columnas y no la lleva**, porque su lista fija de columnas no la incluye —
aunque `pipeline.stage_save` **sí** la mete en el `meta_json` del NPZ. Si se lee el índice
equivocado, la columna «no existe» y se concluye que el reloj falta cuando está en el meta.

### Dos avisos

* **No citar** `qatarfoil2021` 63,4 % ni `incheonfoil2022` 38,1 % de reloj parado: están **sin
  validar**, porque en esas sedes el lector FIE tampoco lee el reloj y no se distingue «parado» de
  «ilegible».
* **Interacción:** el recorte de ventana por reloj y el barrido de ventana **se pisan**. Con
  `pre_s = 20,0` la ventana de un clip de región es hoy el clip entero. Hay que medirlos **juntos,
  en factorial**, no uno encima del otro.

---

## 5.ter Los tres brazos de eliminación del atajo

Van en el cuaderno 02 como **brazos medibles con el signo declarado antes de medir**, que es lo que
salva de leer un resultado al revés. `preflight.declarar_brazo(nombre)` los imprime.

### A. Ventana simétrica en contenido

* **Qué hace:** con `post_s ≥ 1,0` las **dos luces y las dos acciones posteriores caen enteras**
  dentro de la ventana, así que **la asimetría de truncamiento desaparece**.
* **Base medida:** `|dt_ms|` máximo **520 ms** sobre 3.331 clips de dos luces.
* **Qué se sabe ya:** el ancla compartido en la primera luz **ya existe**, y anclar cada tirador en
  **su** luz se probó y **empeoró** la brecha (**+2,94 pp**). La versión de arriba **no se ha probado
  nunca**.
* **Signo esperado:** la **brecha baja sin que la media global caiga**.
* **Cómo se leería al revés:** si la brecha baja porque cae el 83,00 % de los 1.671, es una pérdida
  disfrazada — degradar 5 pp esa población cuesta **−3,18 pp**.

### B. Reponderar los 960 por 1,75×

* **Qué hace:** pesa ×1,75 los clips en los que ganó el que encendió primero.
* **Base:** legítimo **en entrenamiento**, porque ahí se conoce la etiqueta. El diagnóstico está
  medido: el modelo predice «gana el segundo» en el **72,71 %** cuando la verdad es **63,51 %**.
* **Signo esperado:** los 960 suben desde 45,21 % y **la media global pareada no baja**.
* **Cómo se leería al revés:** hay evidencia de que **la información puede no estar**: cinco familias
  ajustadas caen al **44-60 %** en ese subconjunto mientras el motor de reglas, que no ajusta nada,
  se queda plano en **53-57 %**. El saldo: +5,40 pp exige llevar los 960 a 60 %, y basta degradar
  5 pp los 1.671 para perder −3,18 pp. **Veredicto en la media global pareada, nunca en los 960.**

### C. Invertir el orden de luces como aumentación

* **La trampa:** el orden de luces **informa un 63,51 %**, así que invertirlo **manteniendo la
  etiqueta enseña algo falso** y destruye señal legítima.
* **La versión sana:** invertir luces **y etiqueta juntas**, o invertir solo en una **fracción** y dar
  la **curva** sobre esa fracción.
* **Signo esperado:** la brecha baja y la media global se mantiene, con la curva mostrando el óptimo.
* **Cómo se leería al revés:** se sacrifica **información real** — una parte del 63,51 % **no es un
  atajo sino la regla de esgrima**: quien arranca primero tiende a encender primero. Hay que declarar
  el sacrificio y medir el saldo, no solo la brecha.

**Para los tres:** el suelo es el **nulo de kernel, +0,36 pp con 327 de 3.047 decisiones distintas**
cambiando solo el backend de atención. Por debajo de eso no hay efecto que leer. Y el estimando es
**el del artefacto y solo el del artefacto**: promediar probabilidades y umbralizar una vez, porque
un efecto de +1,03 con el otro convenio fue **+0,80 con el intervalo cruzando el cero**.

---

## 6. Verificar cada fase

**Tras el 01:** `EXTRACTION_RESULT["n_ok"]`, `n_failed` y, si no es cero, leer `failure_reasons`
—con duración completa hay que mirar específicamente `decoded_window_planning_budget`, porque un
recuento no nulo ahí significa sesgo por resolución. `STORE_RESULT` no puede ser `None` y
`STORE_ERROR` tiene que ser `None`; `STORE_RESULT["splits"]["counts"]` tiene que traer las tres
particiones y cada `shared_*` a 0. Anotar la ruta y el SHA-256 de `DATA_EXPORT`.

Y las tres comprobaciones nuevas, que están en la celda de verificación:

* **el índice, contado con el lector real.** `AnalysisRepository.index()` se salta **en silencio** un
  directorio de NPZ sin `npz/index.csv`: costó **689 clips invisibles** el 2026-09-09 y antes tuvo la
  página de un tirador en 180 de 587 durante días. Se cuenta el conjunto en cada eslabón
  —`manifiesto ∩ index.csv(ok|cached) ∩ .npz en un directorio registrado`— y se dice **cuál lo tira**,
  en vez de suponerlo. Genera `index.csv` al cerrar cada shard, copiando el orden de columnas de un
  índice que ya carga, y emite una fila solo cuando su `.npz` existe.
  *Observación de esta sesión, ya cerrada:* en `localrun/` hay **cinco directorios con NPZ y sin
  `index.csv`** (`runs_extra1` 3.761, `runs_extra2` 3.902, `runs_extra3` 3.177, `prueba_larga` 35,
  `prueba_larga2` 35) [medido]. **Comprobado después: 10.933 NPZ, 10.898 en el manifiesto y solo
  1 clip realmente nuevo.** Son duplicados de `runs_full`. **No hay que indexarlos**, y el caso sirve
  como ejemplo de la regla: la respuesta no salió de calcular, salió de medir el conjunto en cada
  eslabón.
* **`valid_id` exige `[A-Za-z0-9_-]{1,200}`**, así que un `clip_id` no ASCII se cae del índice sin
  error aunque su fila esté escrita y su `.npz` exista: exactamente **13 clips** del archivo
  (`Vicuña`, `Kauai'Ō'ō`, `Vendéen`; 7 derecha, 6 izquierda, sin dobles, 0,04 %). **No relajar el
  validador.** Es una brecha conocida y aceptada; no perder tiempo persiguiéndolos.
* **el coste de QC de esta corrida**, contado sobre los `analysis/qc/*.json` que acaba de escribir,
  con la referencia del §5.2 al lado para poder decir «esto es la ventana, no un fallo de
  extracción».

**Tras el 02:** `TRAINING_RESULT["state"]["state"] == "completed"`,
`["local_bundle_reload"] == "passed_cpu"` y `model_bundle.zip` en la raíz del workspace. La tabla de
evaluación imprime train/val/test con cobertura y abstenciones. **Y el veredicto se lee con
`PF.dos_poblaciones`, no de la media global sola.**

**Tamaños.** Con la ventana desplegada [medido, corrida de 6 clips]: zip del workspace 776 KB sin
modelo, 9,13 MB con modelo y checkpoints de época; `model_bundle.zip` 1.057.414 B; checkpoints de
época 851 KB y luego ~3,16 MB cada uno (presupuesto ≈ 3,2 MB × (épocas+1)); 188.063 parámetros a
d=64/F=78.

**Tamaño de los NPZ con duración completa** [medido, n=513, `bytes = 6.005 + 138,4 × T`, R² = 0,92,
desv. residual 2.972 B]:

| T (cuadros) | KiB/clip | 25.139 clips de florete |
|---:|---:|---:|
| 129 (hoy) | 23,3 | 0,56 GiB |
| 210 (`"amplia"`) | 34,2 | 0,82 GiB |
| **318 (`"completa"`)** | **48,8** | **1,17 GiB** |

Los arrays que escalan con T son `poses (T,2,17,3) f16`, `boxes (T,2,4) f16`, `lights (T,4) u8`,
`light_observed (T,4) bool`, `pan (T,) f32`, `dup (T,) bool`, `source (T,2) i8`, `time_s (T,) f64`;
243 bytes/cuadro sin comprimir, **138,4 medidos** tras la compresión del NPZ.

**El disco no es el problema.** El corpus de NPZ pasa de ~0,35 GiB a ~1,2 GiB. Los mp4 se descargan y
se **borran tras extraer** (`DELETE_OWN_DOWNLOADED_MP4` automático), y el mp4 medio mide **2,24 MB**
[medido sobre 43.170 ficheros, ~96,9 GB en total], así que el disco queda plano en la cola de
descarga (a lo sumo `queue_size × 2,24 MB`) más el workspace. **No hay que trabajar por lotes por
disco**; hay que trabajar por sesiones por **tiempo** (§4).

Los fps del corpus [medido sobre 300 mp4, semilla 42]: **~30 el 56,0 %, ~50 el 20,7 %, ~25 el 14,7 %,
~59-60 el 6,3 %**, y en la metadata de los NPZ aparecen valores no enteros como 29,991 y 24,993.
Resolución: **1280×720 el 63,0 %, 1920×1080 el 29,3 %, 854×480 el 6,3 %, 640×360 el 1,3 %**.
**No supongas 30 fps en ningún sitio.**

---

## 7. Reanudar tras una desconexión

**Fase 1** reanuda por clip desde `checkpoints/clips/*.json`; un clip se salta solo si verifican su
hash de checkpoint, el del NPZ, el de cada evidencia y el del vídeo origen. `checkpoint_interval`
hace ~20 instantáneas por corrida a cualquier tamaño. Pierdes como mucho el clip en vuelo. Restaura
el último zip de Drive en una carpeta **nueva** y vuelve a ejecutar.

**Lo que la duración completa añade aquí, y es importante:** `ClipLedger` valida el NPZ contra
`expected_config`, y `pipeline_config` graba `pre_s`/`post_s`/`use_clock`. Así que **una corrida de
duración completa no puede reanudar dentro de un workspace extraído con la ventana desplegada**:
todos los NPZ fallarían la validación de configuración y se re-extraerían. Y `build_current_store`
levanta `Training selection changed; create a new workspace/Store version` si
`eligible_index.csv` cambia. **Conclusión operativa: workspace NUEVO, y el `tournament_splits_GLOBAL.json`
SE CONSERVA** —perderlo rompe la separación train/test, y las particiones están congeladas por torneo,
que es justo lo que hace que el porcentaje mantenga las fracciones.

Dos peligros de desconexión que siguen, los dos fallan a gritos:

* un `<registry>.lock` rancio en Drive bloquea la corrida siguiente — bórralo cuando estés seguro de
  que no corre nada;
* un NPZ huérfano (la VM murió entre escribir el NPZ y confirmar el ledger) **aborta el siguiente
  `run_data` antes de hacer nada** — borra `npz/<clip_id>.npz` y su `analysis/*/<clip_id>.json`
  parcial.

**Fase 2** reanuda en frontera de época desde `training/epochs/epoch_NNNNNN_<32hex>.epoch.zip` con
estado completo de optimizador, planificador, escalador y RNG, sin pickle. Se **niega** a reanudar si
cambiaron los datos, la selección, el lote, `T_max`, el plan de épocas, los hashes de código o **la
firma del runtime**: o sea que **un modelo de GPU distinto al reconectar rechaza la reanudación**. La
reanudación en CUDA necesita además `FENCINGLY_EXPERIMENTAL_CUDA_RESUME=1`. Mantén
`training/epochs/` en el almacenamiento local de la VM, **nunca en Drive**: la publicación usa
`os.link`.

---

## 8. Lo que puede fallar en Colab y no aquí

Lista explícita, que es lo que se pidió.

### Versiones

1. **`torch 2.14.0+cu126` puede no existir para el Python de ese runtime.** Es el bloqueante número
   uno. `preflight.entorno(exigir=True)` **levanta** y da el comando exacto. Si el índice de PyTorch
   no publica ese wheel, **no sigas**: elige un runtime con otro Python o acepta explícitamente que
   la corrida es un ensayo cuyo número se tira. Sin esas versiones el número no compara con nada.
2. **Instalar y no reiniciar el runtime es peor que no instalar. Es la trampa más peligrosa del
   paquete.** `_producer()` lee las versiones con
   `importlib.metadata.version(...)`, que consulta los **metadatos de la distribución en disco**, no
   el módulo cargado. Después de `!pip install numpy==2.5.2` sin reiniciar, numpy sigue cargado en la
   versión anterior mientras los metadatos ya dicen 2.5.2: **la huella coincide y las medidas se
   calculan con el numpy equivocado**. Es el fallo silencioso peor de todos, porque la propia
   comprobación de integridad da el visto bueno. `preflight.entorno` compara `numpy.__version__`
   (memoria) con `importlib.metadata.version("numpy")` (disco) y levanta si difieren. **Localmente no
   pasa: se elige el venv y ya.**
   **Por eso `entorno()` va ANTES de `huella()` en el cuaderno, y el orden no es cosmético:** la
   huella no lo detecta y daría falsa seguridad.
3. **Un parche de numpy invalida los 23.091 artefactos y el fallo es un desempate de caché
   silencioso, sin error.** Comprobado: numpy 2.5.2 → `0b87d014…3c4f`, numpy **2.5.3** →
   `874c1ca9…6533` desde fuentes byte a byte idénticas. `preflight.huella(exigir=True)` lo mata en la
   celda tres.
4. **La versión de Python NO afecta a la huella en este juego de fuentes** [medido esta sesión,
   `scratchpad/colab/medidas/ast_sha_por_python.py`]: `source_ast_sha256` sale **idéntico en los 20
   ficheros** con Python 3.12.8 y con 3.14.2 (agregado `afe6490d…`). Era el riesgo que quedaba abierto
   y está cerrado: Colab puede traer el Python que quiera. **No está demostrado para versiones
   futuras** —`_ast_sha` ya lleva un caso especial para `type_params`, señal de que la deriva del AST
   fue un problema real— y por eso la comprobación existe.
5. **`install_missing` instala nombres SIN fijar versión y ignora `requirements_*.txt`.** Ahora que
   corre automáticamente, un refresco de wheels de Colab puede cambiar `ultralytics` por debajo. Fija
   explícitamente si eso importa.
6. **`requirements_web.txt` omite `torch`**, pero `webapp_react/server.py:32` lo importa sin
   condición.
7. **Instalar `ultralytics` mete un paquete `tests` de primer nivel en site-packages** que sombrea
   `webapp/tests/`. Ejecuta la suite desde dentro de `webapp/tests` con `PYTHONPATH=<webapp>`.

### Disco

8. **No es el cuello de botella, y conviene saberlo para no gastar esfuerzo ahí.** NPZ ~1,2 GiB para
   el corpus completo con duración completa; mp4 borrados tras extraer, 2,24 MB de media. Lo que sí
   hay que vigilar es `min_free_disk_bytes` (512 MiB) durante la descarga: el descargador aborta con
   `Download byte/disk budget exhausted; partial file retained` y **conserva el parcial**, que hay que
   borrar a mano antes de reintentar.
9. **`training/epochs/` en Drive rompe la publicación**, que usa `os.link`. Local de la VM, siempre.

### Tiempo de sesión

10. **El corpus completo con duración completa no cabe en 8 h ni en A100 High-RAM** (§4): ~8,7-15,4 h.
    Planifica dos o tres sesiones, o baja el porcentaje, o usa `"amplia"`. Aquí no pasa porque una
    máquina local no te desconecta a las 12 h.
11. **`runtime_verdict` levanta antes de la primera descarga** si no hay GPU y pediste más que el tope
    de CPU (6 clips), o si la RAM está por debajo de 1 GiB; avisa con 2 vCPU y con RAM < 8,8 GiB. Con
    duración completa, **haz caso del aviso de 2 vCPU**: las etapas de CPU, no la GPU, son el cuello.

### Reanudación

12. **Un modelo de GPU distinto al reconectar rechaza la reanudación de la Fase 2.** En Colab te
    pueden dar T4 una vez y L4 la siguiente; localmente la GPU es la misma siempre.
13. **La reanudación en CUDA exige `FENCINGLY_EXPERIMENTAL_CUDA_RESUME=1`.**
14. **`<registry>.lock` rancio en Drive** bloquea la corrida siguiente. Solo pasa con Drive.
15. **NPZ huérfano** aborta el siguiente `run_data` antes de trabajar. Más probable en Colab, donde la
    VM muere sin avisar.
16. **Una corrida de duración completa no reanuda en un workspace de ventana desplegada** (§7):
    workspace nuevo, registro GLOBAL conservado.

### Runtime

17. **La duración completa exige High-RAM.** En T4 estándar rechaza el 11,31 % de los clips y el
    rechazo está correlacionado con la resolución, o sea que sesga la selección en silencio salvo por
    el `ATTENTION:` de `failure_reasons`. Aquí no pasa porque la 3060 va con 31,93 GB de RAM física.

### La ventana y el Store

19. **Un almacén ancho sin rodaja produce un Store desalineado, y no da error.** Es el §5.0.bis. Aquí
    tampoco pasaría —es del código, no de Colab—, pero en Colab se descubre después de 10 h de GPU en
    vez de después de una prueba de 6 clips. Por eso `alineacion_del_store` es una puerta y no un
    aviso: si levanta, es que el Store se construyó desde los NPZ sin recortar.
20. **Una rodaja que no cabe en `T_max` pierde la ventana de guardia en silencio.** `6.0/1.0` son 211
    cuadros y `T_MAX = 135` son 4,467 s: el Store se queda `[t−3,47 : t+1,0]`.
    `preflight.plan_rodaja` lo dice antes de gastar nada.

### El estimador

18. **`HistGradientBoostingClassifier(early_stopping='auto')` se enciende por encima de 10.000 filas
    de entrenamiento**, y con ella un corte de validación aleatorio que hace que las semillas
    importen. Un pliegue de validación cruzada entrena 5.672 filas y no le afecta; **el ajuste sobre
    la población completa sí**, y con los pares nuevos la población pasa a **6.465 pares = 12.930
    filas con espejo** contra el umbral de 10.000, así que **ya lo cruza**. `prio2/e1_tabular.py`,
    `prio2/e10_afina.py` y `prio2/e12_final.py` ya lo fijan. **Si el cuaderno construye su propio
    estimador, tiene que fijarlo**: usa `preflight.estimador_hgb(...)`, que pone
    `early_stopping=False` explícito y **levanta** si le pasas `'auto'`. Sin eso, el comportamiento
    del estimador depende de cuántos clips se extrajeron anoche.

---

## 9. Publicar el repo para la ruta de clon opcional (sin cambios)

Expón la webapp en **exactamente uno** de `<raíz>/FENCINGLY/webapp/`, `<raíz>/webapp/` o `<raíz>/`,
con `fencingly_v2/{colab_workflow,training_recipe,epoch_checkpoint,measurement_artifacts}.py`,
`webapp_react/server.py` y **`webapp_react/frontend/dist/index.html` comprometido** (normalmente
ignorado por git — la celda se para sin él). Después `git ls-remote <url> refs/heads/main`, revisa el
diff y fija `FENCINGLY_RELEASE_GIT`, `FENCINGLY_RELEASE_GIT_REF`, `FENCINGLY_RELEASE_COMMIT`. El pin
de commit es obligatorio; el SHA-256 de `artifact_bundle.py` se sigue comprobando contra el
`HELPER_SHA256` fijado en el cuaderno. **Añade `preflight.py` al release**, o habrá que subirlo a
mano en cada sesión.

---

## 10. Lo que la noche cerró, para no reintentarlo en Colab

Seis vías, ninguna llega al suelo de 0,8 pp: **vorticidad del flujo óptico** (y con ella
**SEA-RAFT/CUDA no está indicado**), **transformador de vídeo VideoMAE**, **latente antisimétrico**,
**centro de masas**, **tirón** y la **regla t.83**. Más la **simetrización de la ventana** con tres
geometrías de ancla.

El techo **no está** en la representación (VideoMAE predice la sede al 97 %), **ni** en el ruido de
etiqueta (refutado, y el suavizado apunta al revés), **ni** en la geometría de la ventana. Está en el
orden de las luces, sobre todo en las **256 columnas agregadas y congeladas**, y una parte de eso
**no es un atajo sino la regla de esgrima**: quien arranca primero tiende a encender primero.

**Y no cambies el suavizado.** `ls = 0,30` pasó el suelo de ruido, pasó Bonferroni sobre siete brazos,
pasó tres semillas y pasó una comprobación de degeneración, y **aun así no cuenta**: con el convenio
desplegado el intervalo cruza el cero (+0,80 [−0,16, +1,85]), a peso de mezcla 0,65 queda en +0,24 a
+0,38, y la calibración ya se estaba cobrando la ventaja (promediar tres semillas lleva el ECE de
0,1137 a 0,0603 y el Platt que ya existe al 0,0258; la pendiente de Platt se pasa de largo, de 0,612
a **1,209**). Lo que queda en pie es un resultado **sobre la red**, no un cambio de producto: `ls`
estaba congelado en 0,05, nunca se había barrido, y **está en el mínimo de su propia curva**.

La lección, que vale para cualquier candidato futuro: **una mejora hay que medirla en el estimando, en
la etapa del conducto y en la población donde se va a desplegar.** Ninguna de las cuatro pruebas que
pasó habría detectado un convenio de promediado, un peso de mezcla ni un calibrador aguas abajo.

---

## 9.bis Respaldo incremental: repositorio PRIVADO de Hugging Face

Una corrida de horas en Colab **se muere sola**: la sesión caduca, el runtime se recicla, se cae la
red. Sin persistencia incremental se pierde todo. Es requisito duro, no adorno.

### La excepción está escrita, y su alcance

`AGENTS.md` **línea 4**, decidida por el usuario el 2026-09-09 y acotada a propósito. La comprobé en
el fichero antes de implementar, porque la línea 3 prohíbe subir datasets y una autorización de
palabra no cambia las instrucciones del proyecto. Lo que la hace aceptable: la carga son **~50 GB** y
caben en los **100 GB gratuitos del plan privado**, así que nada se hace público.

> **Un repositorio público sigue prohibido.** Los recortes del corredor son **fotogramas reales de
> retransmisión**, no rasgos derivados, así que un repo público redistribuiría vídeo de terceros; y
> el manifiesto lleva **nombres de tiradores, torneos y resultados**, así que publicaría el historial
> competitivo de atletas identificables. `create_repo(..., private=True)` es explícito y la celda de
> verificación **levanta** si `repo_info().private` no es `True`.

No ampliar a un repo público, a un segundo servicio, ni a ninguna carga más allá de los artefactos de
extracción, sin una decisión nueva y explícita del usuario.

### Qué cabe, y por qué

**El flujo óptico denso no se guarda.** Regenerarlo desde los recortes de 128×128 reproduce la ruta de
vídeo **bit a bit** a 394 ms/clip contra 1.250. Así que **~45 GB de recortes sustituyen a ~550 GB de
flujo**. La regla: **guarda lo que no se puede regenerar; regenera lo que sí.** Reparto objetivo:
recortes ~45 GB, pose a cadencia nativa ~2 GB, índices y columnas de resumen <1 GB.

### El token es una credencial

Se lee de un **secreto de Colab** (`google.colab.userdata`) o de **`HF_TOKEN`**. Nunca en el cuaderno,
nunca impreso, nunca en un log ni en un informe. `preflight.token_presente()` devuelve un **booleano**,
no el token. Y `aviso_respaldo()` dice exactamente qué secreto hace falta y cómo se pone, porque eso
es lo primero que va a fallar.

### Troceado, porque los límites se rozan

La plataforma admite **< 100.000 ficheros por repo** y **< 10.000 por carpeta**. Con 34.374 clips y
varios ficheros cada uno eso se roza, así que se agrupa en **`tar` por lote**:
`lotes/<sede>/lote_<nnn>.tar` con los NPZ, las evidencias y los checkpoints de sus clips, más
`index.csv` y `estado.json` en la raíz. Con lotes de ~35 clips salen **~1.000 ficheros** para el
corpus entero, dos órdenes de magnitud por debajo del límite.

**El tamaño de lote se deriva del ritmo medido**, no se fija a ojo:
`clips_por_lote(s_por_clip, minutos_por_instantanea=10)`. Con los **17,2 s/clip medidos** en el
ensayo salen **35 clips**; con 11,9 s/clip, 50.

### El estado es el índice, no un contador

`_clips_completos()` es la fuente de verdad: **fila en el índice con `status` ok|cached ∩ su `.npz`
presente**. Prefiere `npz/index.csv` y cae a `index.jsonl`, que es lo único que hay a mitad de
corrida. Si el contador dice 1.200 y el índice tiene 1.180, la reanudación miente — y eso **me pasó
en el ensayo**: mi primera versión de la instantánea escribía el contador local de la corrida, y tras
reanudar decía `completado=3` con 6 NPZ en el destino. `estado_reanudacion` lo cazó («manda el
ledger») y lo arreglé para derivarlo del índice.

**Y una cosa que sorprende a mitad de corrida:** `npz/index.csv` **no existe todavía**. `run_data`
escribe `index.jsonl` por clip (con `flush` + `os.fsync`) y `npz/index.csv` solo lo construye
`finalize_data_index` al terminar. Así que a mitad de corrida `AnalysisRepository.index()` **no ve
nada, y eso es lo esperado**, no un fallo. No hay que «arreglarlo» escribiendo un index.csv parcial:
el ledger ya cubre la reanudación.

### Qué está probado de verdad y qué queda a la celda de verificación

| pieza | estado |
|---|---|
| empaquetado en `tar` con NPZ + evidencias + checkpoints | **probado**: 2 lotes, 15 miembros cada uno, hashes distintos |
| troceado y límites de ficheros por carpeta | **probado**: 4 ficheros en destino, límites declarados en el estado |
| fichero de estado derivado del índice | **probado**, y arreglado tras romperse |
| lógica de reanudación (pendientes = completos − empaquetados) | **probado**: 3 → 3 → 0 pendientes |
| idempotencia (re-empaquetar no duplica) | **probado**: `reutilizado: True`, 0 clips |
| aviso ruidoso sin token | **probado** |
| **subida autenticada y `private=True` comprobado** | **probado, por Alejandro, no por mí** |
| **mi ruta de lotes en `tar` con reanudación contra el remoto** | **sin probar** |

Hay que distinguir las dos, porque no son lo mismo:

* **La subida autenticada funciona y está comprobada.** Con el token real se creó el repositorio
  `EXCALORIO/fencingly-extraccion` (tipo *dataset*), se subieron **15 ficheros** y se verificó con
  `repo_info` que quedó **privado**. Eso cierra la creación con `private=True`, la subida y la
  comprobación de privacidad. **No lo probé yo**: la credencial no pasó por aquí.
* **Lo que sigue sin probar es mi ruta concreta**: empaquetar en `tar` por lote, subir esos lotes y
  **reanudar contra el remoto** leyendo su estado. Todo eso está probado **contra un directorio local
  que simula el remoto**, no contra el repositorio. La celda de verificación
  (`PF.CELDA_DE_VERIFICACION`) es la que cierra ese hueco: empaqueta un lote, lo sube, lista los
  ficheros y borra la carpeta de prueba.

### La estructura del repositorio, y a la que el empaquetador tiene que escribir

```
notebooks/     los cuadernos y preflight.py
docs/          el reloj, los contratos, qué se detecta
data/poses/    poses en tar por sede
data/crops/    recortes 128x128 en tar por sede
data/tables/   métricas en Parquet por clip_id y frame_idx
data/index/    el índice y el estado
```

`respaldo_empaquetar` escribe hoy en `lotes/<sede>/lote_<nnn>.tar` con `index.csv` y `estado.json` en
la raíz. **Para encajar con esa estructura hay que mapear** `lotes/<sede>` → `data/poses/<sede>` (y
`data/crops/<sede>` cuando se suban los recortes), y el índice y el estado → `data/index/`. Es un
cambio de rutas, no de lógica, y **no está hecho**: lo dejo dicho en vez de suponer que da igual.

---

## 10.bis Vías documentadas y no tomadas, con su cifra

Van aquí y no en ficheros aparte, que es el punto: **un paquete**.

| vía | cifra | por qué no se toma |
|---|---|---|
| **Subir `T_max` por encima de 140** | `"ancha"` 6.0/1.0 pide 211; atención O(T²) → ~2,5× | cambia lo que ve la red y **ese cambio no está medido**. Si se prueba, como brazo pareado (§3) |
| **Subir `feature_contract.MAX_TIMESTEPS`** | no hace falta: 211 ≤ 256 | está en `SEMANTIC_SCOPE` con alcance de módulo → **invalida todos los bundles portables** (§5.4) |
| **`reloj_parado_todo_el_clip`** | 15-16 % del corpus | **no se deriva de `clock_running_s`**: `\|fin−inicio\| < 0,05` da **cero** sobre las 9.865 filas legibles. Hay que recibir la definición, no inventarla (§5.bis) |
| **Anclar en el reloj / marcar repeticiones** | **0 clips** cada una | anclar no añade clips (ancla la lámpara); las repeticiones ya están fuera por política |
| **Compartir la pasada de decodificado (`A1`)** | `A1` es 5,58 s/clip y con duración completa el **41 %** del CPU irreducible; halvarlo llevaría L4 a ~8 h | vive en `lights_generic.py` y arriesga cambiar **qué cuadro se llama el tocado**, lo que cambiaría en silencio cada medida de aguas abajo. Necesita su propia tarea con acuerdo antes/después sobre los cuadros de tocado (§4) |
| **`ls = 0,30`** de suavizado | +1,03 pp con el convenio de ablación → **+0,80 con el intervalo cruzando cero** con el del artefacto; +0,24 a +0,38 en la mezcla | no cuenta. La ventaja de calibración queda en 1,15× y **se pasa de largo** (pendiente de Platt 0,612 → 1,209) (§10) |
| **SEA-RAFT / CUDA para flujo óptico** | la vorticidad no llegó al suelo de 0,8 pp | **no indicado** (§10) |
| **Anclar cada tirador en su propia luz** | **+2,94 pp de brecha, peor** | ya probado y descartado (§5.ter A) |
| **Alimentar `calibration.allez_scale` con una fila nueva del índice** | **estructuralmente inerte** | `allez_scale` lee `f0` del **meta del NPZ** (línea 333, y su firma no recibe `row`), y `f0` está grabado en el NPZ. Ninguna fila lo alcanza: el 88,0 % de los rechazos es `window_not_at_clock_start`. Solo lo arregla reextraer, o la vista del allez del §5.bis |
| **Extracción a cadencia nativa (`FPS_NATIVO`)** | el Store remuestrea a 30 fps y tira el **40 %** de los cuadros a 50 fps y el **50 %** a 60 fps; afecta al **~27 %** del corpus (20,7 % a ~50 fps + 6,3 % a ~60) | **hueco reservado, no implementado.** `TARGET_FPS` está en `sample_time.py`, que está en **los 20 de MEASUREMENT_SOURCES** *y* en **`SEMANTIC_SCOPE`**: tocarlo re-materializa los 23.091 artefactos **y** invalida todos los bundles portables. Es el cambio más caro del conducto. Hay un agente midiendo su coste; el mando entra con sus cifras |
| **Subir el respaldo a un repo PÚBLICO** | — | prohibido por `AGENTS.md` línea 4: los recortes son fotogramas reales de retransmisión y el manifiesto lleva nombres de atletas (§9.bis) |

---

## 11. Contenido del paquete

```
preflight.py                  el módulo de comprobaciones (se importa desde los dos cuadernos
                              y también se ejecuta solo: `python -B preflight.py [workspace]`)
construir_notebooks.py        deriva los cuadernos de duración completa de los canónicos, sin
                              tocarlos; idempotente, cada celda insertada va marcada
notebooks/
  01_Extraccion_y_Evidencia.ipynb                 23 celdas (canónico 18 + 5)
  02_Entrenamiento_y_Brazos.ipynb                 20 celdas (canónico 18 + 2)
  03_Shared_Web_App.ipynb                         copia sin cambios
medidas/
  qc_duracion_completa.py     el coste de QC (§5.2), sobre analysis/qc/*.json
  noop_clip_quality.py        la prueba de no-op del arreglo de ambito (§5.2)
  ensayo_extremo_a_extremo.py el ensayo A/B/C/D: extraccion, reanudacion, huerfano, verificacion
  ensayo_store.py             el Store contra datos reales y contra los NPZ de duracion completa
  barrido_ventana.py          el barrido de ventanas (§3)
  ast_sha_por_python.py       ¿es la huella estable entre versiones de Python? (§8.4)
COLAB_UNICO.md                este documento
```

**Las puertas de `preflight`, que es lo que hace el paquete seguro.** Ninguna necesita un cambio de
fuente; todas usan campos que los artefactos ya escriben:

| puerta | cuándo | qué impide |
|---|---|---|
| `entorno(exigir=True)` | antes de todo | versiones no clavadas, y el runtime no reiniciado tras instalar |
| `huella(exigir=True)` | después de `entorno` | artefactos invalidados por un parche de numpy |
| `parche_de_recorte(exigir=True)` | antes de extraer | medir la guardia sobre el clip entero (152°) |
| `plan_rodaja(nombre, t_max=N)` | antes de extraer | una rodaja que no cabe en `T_max` (pierde la guardia) |
| `plan_ventana(env, almacén)` | antes de descargar | sesgo por resolución en un runtime insuficiente |
| `alineacion_del_store(dir, post_s)` | después del Store | un Store desalineado (el precio §5.0.bis) |
| `estimador_hgb(...)` | al construir el árbol | `early_stopping='auto'` |
| `indice_real(workspace)` | tras extraer | «N clips de M» sin saber qué eslabón lo tira |

Regenerar los cuadernos tras un cambio en los canónicos:

```
python -B construir_notebooks.py <repo>/colab/notebooks ./notebooks
```

Reproducir las medidas de este documento:

```
<venv-gpu>/python -B medidas/barrido_ventana.py       <workspace>/analysis/qc
<venv-gpu>/python -B medidas/qc_duracion_completa.py  <workspace>/analysis/qc
<venv-gpu>/python -B medidas/ast_sha_por_python.py    <repo>/webapp/fencingly_v2
```

---

## 11.bis El ensayo: qué se ejecutó de verdad y qué se rompió

Ejecutado en esta máquina, **RTX 3060 de 12 GB**, un solo proceso (había una materialización con 3
procesos en marcha; carga comprometida 16,35 GB de 47,43 GB al arrancar). Reproducible con
`medidas/ensayo_extremo_a_extremo.py {a|b|c|d}`.

### Lo que corrió

| fase | resultado |
|---|---|
| **A. extracción a duración completa, 3 clips** | **3/3 ok, 0 fallos, 51,6 s = 17,2 s/clip** |
| **B. reanudación, mismo espacio, 6 clips** | **reanudados=3, procesados=3, ok=6, 0 fallos, 11,9 s/clip** |
| **C. NPZ huérfano, roto a propósito** | **levanta**: `NPZ exists without a validated checkpoint; inspect/recover explicitly or use a NEW workspace` |
| **D. índice, rodaja, vista del allez, reloj** | ver abajo |

La fase B es **la prueba de reanudación pedida**: se cortó tras 3 clips, se volvió a arrancar sobre el
mismo espacio pidiendo 6, y **saltó los 3 hechos y procesó 3** sin reprocesar ni perder ninguno. La
fase C rompe el mecanismo a propósito borrando un checkpoint y dejando su NPZ: **aborta antes de
trabajar**, que es el comportamiento documentado y el correcto.

**Los primeros NPZ de duración completa del proyecto.** `T` = 249 / 360 / 358 cuadros contra los 129
de mediana desplegados; `pre_s_eff` = 8,36 / 10,48 / 7,96 s (todo el clip antes del tocado);
`f0 = 0`; `post_s_requested = 60,0`; **42,9 a 71,8 KiB** por NPZ. El modelo de tamaño predecía 48,8
KiB a T=318, así que **la extrapolación se sostiene**.

**Y los 17,2 s/clip validan la derivación del §4**: con `process_workers=1` el coste es el CPU
irreducible completo, y mi rango derivado era **13,73-17,98 s/clip**. Cae dentro.

**La rodaja funciona:** T → 115 / 138 / 140, `f0` reescrito a 109 / 194 / 118, `pre_s_eff` de vuelta a
4,0 / 4,0 / 4,03. Y se restaura con identidad verificada.

**La vista del allez funciona:** `lag_s` = 0,481 / 0,497 / 0,496, todos bajo la tolerancia de 0,7, y
el primer clip da `m_per_x = 7,236` con `reject=None`. Los otros dos rechazan por la compuerta de
**contenido**, no por la de ventana.

**El índice y el lector real:** `merge_after_processing` escribió `npz/index.csv`,
`clock_running_s` presente al **100 %**, ancla de allez utilizable **6 de 6**, y
`AnalysisRepository.index()` devolvió **6 = 6 filas − 0 no-ASCII**. Contado con el lector, no con las
filas.

**El contrato del reloj:** 3 filas, 12 campos, 0 errores, 0 indefinidas.

### Lo que se rompió, y era el objetivo

**1. `finalize_data_index` exige `work/manifest_<arma>.csv`.** Un `run_data` suelto **no deja un
espacio de trabajo finalizable**: `merge_after_processing` levanta `FileNotFoundError`. En el cuaderno
está cubierto porque `prepare_data_manifest` lo escribe; en el ensayo tuve que escribirlo. **Arreglado
en el ensayo, y es un aviso para cualquiera que llame a `run_data` por su cuenta.**

**2. Mi propia instantánea escribía un contador, no el índice.** Tras reanudar decía
`completado=3` con 6 NPZ en el destino — exactamente la mentira que el requisito prohíbe.
`estado_reanudacion` lo cazó y **lo arreglé** para derivar `clips_completos` del índice ∩ NPZ
presentes, dejando el contador local como diagnóstico con su nombre propio.

**3. `build_current_store` levanta `KeyError('coverage_L')` cuando el elegible queda vacío.** No dice
«0 clips elegibles»: sigue filtrando sobre un DataFrame sin filas hasta que una selección de columnas
se queda sin nada. Después de horas de GPU en Colab ese mensaje no dice nada de lo que hay que
arreglar. **Arreglado con `preflight.embudo_elegibles`**, que cuenta el embudo **antes** de construir
y nombra el paso que lo vacía:

```
filas del indice               6
status ok|cached               6  (-0)
label_ok == true               0  (-6)   <- aqui
```

En mi caso la causa era el manifiesto construido a mano: **`label_ok` lo recalcula
`dataset_update.prepare_manifest`** desde las columnas de marcador, no se toma del que se le pase. La
puerta ahora lo dice con esas palabras. **El Store en sí no se pudo ensayar por eso**, y lo digo en
vez de darlo por bueno.

**4. `ultralytics` avisa 37 veces de que `'half' is deprecated, use 'quantize' instead`.** Hoy es
ruido, pero es exactamente la materialización del §8.5: `install_missing` instala **sin fijar
versión**, así que un refresco de wheels de Colab puede convertir ese aviso en un error.

**4.bis. Y la rotura más grave, que solo aparece ejecutando: envolver `dataset.load_sample`
ENVENENA EL STORE.** Mi primera implementación de la rodaja envolvía esa función en caliente durante
`build_store`, para no tocar ninguna fuente. Sobre 300 clips reales el Store salió con la
**alineación perfecta** — mediana 0,267 s, `store_truncated` 0 de 300 — y **aun así está roto**:

```
ST.Store(destino)
 -> FC.validate_runtime_compatibility(info)
 -> ContractError('Bindings runtime incompatibles: dataset.py')
```

`build_store` graba en `store.json` el `FC.runtime_compatibility()` **del momento de construir**, y
esa función hashea `_module_bindings_sha256(module)` de cada fichero de `SEMANTIC_SOURCE_FILES`,
`dataset.py` incluido. Con el envoltorio puesto el hash es otro y **queda congelado en el fichero**:
al abrirlo después con un `dataset.py` limpio no coincide y **el Store no se puede abrir nunca más**.
No es un fallo del proceso, es permanente. Comprobado en el mismo proceso: el Store del corpus
(construido sin envoltorio) abre con **N=14.437**, y el mío levanta `ContractError`.

Yo había verificado que los cuadernos no llaman a `load_bundle` ni a `save_bundle`, que era la vía
que conocía de `portable_contract.SEMANTIC_SCOPE`. **Lo que no verifiqué es que `build_store` llama a
`feature_contract.runtime_compatibility()` por su cuenta**, en la línea que escribe `store.json`. La
lección: la validación de *bindings* runtime no vive solo en el contrato portable.

**Arreglado con `PF.rodaja_a_disco`**: se escriben los NPZ recortados en un directorio nuevo y el
Store se construye desde ahí, con `load_sample` intacto de principio a fin. Cuesta ~0,6 GiB de disco
para el corpus de florete (~1,8 GiB con el almacén de duración completa). El envoltorio queda
documentado como prohibido en `PF.ENVOLTORIO_PROHIBIDO`, para que nadie lo resucite.

**Verificado tras el arreglo:** Store de 300 clips reales construido desde el directorio recortado →
**abre (N=300, T_max=135)** y alinea a 0,267 s.

**4.ter. Ejecuté la batería con el intérprete equivocado.** Con `.venv-gpu` da **1.233 / 0 fallos /
34 errores / 16 omitidos** y `successful: False`. Los 34 son todos `ModuleNotFoundError: No module
named 'fastapi'` en los módulos de API y `web_workspace`. **AGENTS.md línea 24 lo documenta
exactamente**, y mi propia memoria también: la batería solo vale con **`scratchpad/fz/.venv`**, y con
`.venv-gpu` baja a 1.233 con 34 errores de importación «y no dice por qué». Los dos intérpretes dan
la misma huella, así que `.venv-gpu` sigue siendo el correcto para materializar; **solo la batería
necesita `.venv`**. Repetida con el intérprete válido.

**5. El corpus local de mp4 desapareció a mitad del ensayo.** Los ~43.000 mp4 / ~97 GB que había bajo
la base fueron borrados por otro agente liberando disco (C: pasó de 97 a 187 GB libres). Los vídeos
están íntegros en **`D:\FENCINGLY_VIDEOS_RAW`** (45.008 mp4) y el ensayo apunta ahí. **Si algún script
del paquete busca vídeo bajo la base, no lo encontrará.**

### El Store, ensayado contra datos reales

Se ensayó dos veces, con `medidas/ensayo_store.py`, y las dos importan:

**a) A escala, con etiquetas reales.** 300 filas del `training/eligible_index.csv` del corpus —el que
`prepare_manifest` ya validó, que es lo que fallaba con un manifiesto a mano— y sus NPZ de ventana
desplegada. El embudo pasa entero (300 → 300 en los cinco pasos), la rodaja «histórica» es un
**no-op verificado** (300 NPZ copiados sin recortar, 0 recortados), el Store **abre** con N=300 y la
alineación da **mediana 0,267 s, 0 de 300 fuera de tolerancia**.

**b) La cadena nueva entera, sobre NPZ de duración completa.** Los 6 NPZ reales de duración completa
(T = 249 / 360 / 358 / 442 / 263 / 406), rodaja 4.0/0.6 a disco y `T_max = 140`:

| | con rodaja | **sin** rodaja |
|---|---:|---:|
| `store_truncated=True` | **0 de 6** | **6 de 6 (100 %)** |
| distancia tocado → último cuadro, mediana | **0,566 s** (esperada 0,6) | **1,633 s** |
| máxima | 0,600 s | **3,900 s** |
| fuera de tolerancia | **0 de 6** | **6 de 6** |
| puerta de alineación | pasa | **levanta** |

> **Ahí está el cuarto precio demostrado de punta a punta, y su arreglo.** Sin la rodaja el Store
> guarda una ventana que termina 1,6 a 3,9 s **después** del tocado y enmascara el 100 % de las
> etiquetas auxiliares, sin dar error. Con la rodaja queda alineado y sin truncar.

### Lo que no se ensayó, y por qué

* **El entrenamiento completo.** Se comprobó la receta, el determinismo y el `early_stopping`
  explícito del árbol, pero un subconjunto de 300 clips del corpus no trae necesariamente las tres
  particiones, así que un número de entrenamiento sobre él no sería representativo y no se declara
  ninguno.
* **La subida autenticada a Hugging Face**, por no tener el token. Todo lo demás del respaldo sí
  (§9.bis).
* **`pip install`:** no se necesitó ninguno. El entorno clavado ya estaba, y `dependency_report` no
  reportó nada ausente. En Colab será distinto: es el bloqueante nº 1 del §8.

---

## 12. Antes de ejecutar: lista corta

1. [ ] Runtime **GPU + High-RAM** (obligatorio con `VENTANA_ALMACEN = "completa"`).
2. [ ] `preflight.py` subido junto a los cuadernos.
3. [ ] `preflight.entorno(exigir=True)` en verde — y **runtime reiniciado** si hubo que instalar.
       Va **antes** de la huella, porque la huella no detecta el desajuste (§8.2).
4. [ ] `preflight.huella(exigir=True)` da **`3e1dbe0a…9827`** (medidas) y **`6c9fa66b…3ce7`**
       (movimiento), que es el protocolo de duración completa. Si da `0b87d014…3c4f` / `994c874b…2658`
       es el protocolo **histórico** y la puerta levanta: significa que el recorte no está dentro.
5. [ ] **`preflight.parche_de_recorte(exigir=True)` en verde.** Ya lo está desde las 10:49; la puerta
       se queda puesta para que levante en cualquier máquina que no lo tenga.
6. [ ] `preflight.plan_rodaja(RODAJA_ENTRENAMIENTO, t_max=T_MAX)` en verde: la rodaja cabe, la
       guardia queda dentro, `post_s` cubre los 520 ms y `T_max ≤ 256`.
7. [ ] `preflight.columna_reloj(WORK)` confirma `clock_running_s` en `npz/index.csv` y dice a cuántos
       clips desbloquea la posición absoluta de pista (aquí: **7.161**, 46,34 %).
8. [ ] `tournament_splits_GLOBAL.json` en Drive, conservado del protocolo anterior.
9. [ ] Workspace **nuevo** (una corrida de almacén ancho no reanuda en uno viejo).
10. [ ] `PERCENT_OF_FOIL_CLIPS` elegido sabiendo que el 100 % no cabe en una sesión (§4).
11. [ ] Decidido `USAR_RELOJ`, y **coordinado con quien está conectando la tabla del reloj**: las dos
        decisiones interactúan y se miden en un **2×2**, no una encima de la otra.
12. [ ] Decidido si se acepta la pérdida de **488 clips (3,16 %)** del QC o se acota
        `clip_quality._flatten_vm` (§10.bis). Es el único de los cuatro precios que el paquete no
        cierra.
13. [ ] Los tres brazos **declarados con su signo esperado antes de medir**
        (`PF.declarar_brazo`), y el estimando del **artefacto** (probabilidades promediadas,
        umbralizadas una vez).
14. [ ] Escrito en el informe, antes de mirar el número: **es una línea base nueva y no se resta de
        nada** —por la ventana **y** por la huella—; el veredicto son las **dos poblaciones** y la
        **media global pareada**; y **+1,6 pp es indetectable con este corpus**.

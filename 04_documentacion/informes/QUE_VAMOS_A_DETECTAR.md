# Qué se detecta hoy, qué se detectará, y cuánto mejora

Escrito el 2026-09-09. Cada fila dice **lo que hay hoy**, **lo que cambia** con la extracción de clips completos a
cadencia nativa, y **si la cifra está medida o pendiente**. Las pendientes están marcadas y no son promesas.

---

## 1. Resumen en una tabla

| señal | hoy | con la extracción nueva | mejora | estado |
|---|---|---|---|---|
| **Fotogramas por clip** | remuestreado a **30 fps** al construir el almacén | **cadencia nativa** (hasta 60) | **+40 % de fotogramas** en los **819 clips medidos** a 50-60 fps (**2.363 en el censo**) | **medido** |
| **Cadencia real de origen (`src_fps`)** | no viaja | **columna nueva** | distingue 50 fps genuinos de 60 fps reempaquetados | **medido** |
| **Telecine 25→30** | invisible | **marcado** | **8.503 clips**, un fotograma repetido **cada seis** | **medido** |
| **Duración analizada** | ventana de 4,3 s | **clip completo** (3-16 s) | 2,5× más fotogramas de media | **medido** |
| **Pose, 17 puntos** | sí, con calidad variable | igual, pero a cadencia nativa | derivadas menos aliasadas | medido lo de hoy |
| **Velocidad y aceleración** | sí, calculadas a 30 fps | a cadencia nativa | paso de 75 ms → **20 ms** a 50 fps | **medido** |
| **Jerk (3.ª derivada)** | no existe | candidato (PIDI) | sin medir | **pendiente** |
| **Luces y su orden** | 92-100 % en 11 sedes, **0-0,5 % en 5** | igual: los 5 perfiles ya existen | **cero**, la tabla está completa | **medido** |
| **Reloj y momento del asalto** | **no existe** | **83,1 %** de los clips | **nuevo** | **medido** |
| **Asalto parado / sin acción** | no existe | **17,0 %** detectado | **nuevo** | **medido** |
| **Allez (inicio de frase)** | 0 de 2.914 clips de región | **892 contados** | **nuevo** | **medido** |
| **Posición absoluta en pista** | **NaN** en 2.914 clips | disponible en ~892 | **nuevo para esos** | **medido** |
| **Píxeles por metro (escala)** | entra en **42** clips | **343** reextrayendo | **×8,2**, y mueve la escala un **4 %** | **medido** |
| **Líneas de pista** | 3,8 % contra 4,3 % humano | igual | **cero: ya está en el techo humano** | **medido** |
| **Profundidad monocular** | 46-51 % de error | igual | **no sirve para escala** | **medido** |
| **Estatura** | 104 anclas, dejar-una-fuera | mejora vía escala del allez | pendiente de cuantificar | **pendiente** |
| **Mano dominante** | 185 zurdos / 400 diestros / 2 | igual | cero, ya está | **medido** |
| **Flujo óptico y vorticidad** | 32 columnas | igual, regenerable bit a bit | cero para el acierto, útil como dato | **medido** |
| **Repeticiones de jugada** | no se detectan | **`escala_r` cruda, sin umbral** | **hueco declarado**: ninguna vía discrimina | **medido, y negativo** |
| **Cruce y oclusión** | mediana 0 cruces | igual | poco, en florete casi no se cruzan | **medido** |
| **Velocidad angular del antebrazo** | **no existe** | **4 columnas nuevas** | el ángulo ya estaba; **la derivada no** | **medido, y entra** |
| **Eje del antebrazo vs rival** | no existe | columna cruda | **control fallado**, se guarda igual | **medido, y negativo** |
| **Deliberación del árbitro** | no existe | columna con **8,3 %** de cobertura | **hueco declarado** | **medido, y negativo** |

---

## 2. Las dos ideas tuyas que la medida contradice

Te las digo primero porque son las que cambian el plan.

### La profundidad no sirve para calcular píxeles por metro

Propones usar la profundidad para calcular mejor la escala. **Se probó y da 46-51 % de error** sobre un vano
conocido de 4,00 m. El veredicto medido fue **«orden sí, escala no»**: sirve para saber quién está más cerca
de la cámara, no para medir metros.

Y el modelo son **9,11 GB de pesos** descargados. Pasarlo por 34.374 clips completos cuesta GPU real para un
dato que no es fiable como escala.

**Lo que sí da la escala fiable es la geometría de la pista**, que es como se hace hoy, y su punto débil no es
el detector: es que **el allez entra en el cálculo en solo 42 clips**. Eso es lo que sube a 343 reextrayendo, y
es donde está la ganancia real de escala.

### Las líneas de pista ya están en el techo

Propones detectarlas mejor. Medido: el detector calibra el **3,8 %** de los fotogramas contra el **4,3 %** de
una referencia **asistida por humano**. O sea que **está al nivel de lo que consigue una persona ayudando**, y
no hay margen por «detectar mejor».

Lo que sí falta no es detección, es **anclaje**: la posición absoluta es NaN en 2.914 clips porque no hay
allez, no porque las líneas no se vean.

---

## 3. Lo genuinamente nuevo, y por qué importa

### La cadencia nativa, que es la más grande

Hoy el almacén **remuestrea a 30 fps uniformes**. Medido: **215 fotogramas nativos de un clip a 50 fps se
guardan como 129.** Un 40 % perdido, y justo en los **819 clips medidos** que más cadencia tienen (2.363 en el
censo entero). Ojo con la ruta: **la extracción nunca remuestreó** — se midió una desviación de **0,0 µs**
frente a la cadencia nativa entre 1,661 y 60,003 fps. El remuestreo vive en la construcción del almacén, que
es una decisión posterior y separada.

**Y hay un hallazgo que pesa más que la cadencia alta, porque afecta a la mayoría del corpus.** La banda de
30 fps no es material a 30 fps: son **8.503 clips de 11.211 (75,8 %) con telecine 25→30**, o sea **un fotograma
repetido cada seis exactos** (`dup_rate` 0,164, período 6,0, origen 25,0 fps). Un fotograma duplicado da
**velocidad cero en un paso y doble en el siguiente**, que es precisamente el artefacto que una derivada no
perdona. Eso afecta a **once mil clips**, no a los ochocientos de cadencia alta.

Tres cifras más, para que nadie confunda dos cosas distintas:

| | clips medidos | con patrón regular | **cámara lenta real (ratio ≥ 2)** |
|---|---:|---:|---:|
| 25 fps | 2.894 | 10 | **0** |
| 30 fps | 11.211 | **8.503 (75,8 %)** | 35 |
| **50 fps** | 649 | **0** | **0** |
| 60 fps | 170 | 66 | 19 |

- **A 50 fps no hay ni un clip con duplicados**, así que son 50 fotogramas distintos de verdad y el paso de
  **20 ms es real**. Ese era el argumento importante y se sostiene.
- **La cámara lenta es residual: 54 clips, el 0,35 %.** Pero con ratios de hasta **12,0**, o sea once
  fotogramas quietos y uno doce veces más rápido.
- **El 38,8 % de los 60 fps tiene duplicados y no es lento**: es conversión 50→60 a ratio 1,20.

Por eso **`src_fps` y el ratio viajan como columnas**: la cadencia nominal no es el intervalo de muestreo.

**Un aviso sobre los totales.** `dup_mode` está medido en **15.452 de los 29.220 clips con fps (52,9 %)**, y esa
submuestra **no es aleatoria**: es lo que se llegó a procesar. Ya sabemos que el residuo está sesgado, porque la
tasa de detección correlaciona **r = −0,861** con la fracción ya extraída. Las tasas y los totales se citan cada
uno con su n y **no se multiplican entre sí**.

Y conecta con un fallo ya diagnosticado: el transformador de vídeo falló porque **16 fotogramas sobre 1,2 s
son 75 ms de paso**, y el suceso que decide un doble en florete es un desfase de **10 a 30 ms**. Estaba
mirando por debajo de la escala del suceso. **A 50 fps nativos el paso es 20 ms.**

Eso afecta directamente a lo que pides de velocidad de las poses: una derivada calculada con paso de 20 ms en
vez de 33 ms no es la misma medida.

### El reloj, que no existía

Da tres cosas que hoy no hay:

- **el momento dentro del período**, porque el reloj cuenta hacia atrás desde 3:00;
- **el allez**, o sea dónde empieza la frase de verdad;
- **el asalto parado**, 17,0 % de los clips donde no pasa nada porque el reloj estaba detenido.

Cobertura **83,1 % [81,1, 84,9]** sobre 1.537 clips medidos, con control de falsos del 2,2 % en la bandera de
parado. **Dos sedes quedan sin validar** (`qatarfoil2021`, `incheonfoil2022`) y no se citan.

### El filtrado de poses

Hoy existe: confianza media, cobertura por tirador, tasa de duplicados y cuenta de cruces. Y hay un dato que
tranquiliza: **en florete los tiradores casi no se cruzan**, mediana de 0 cruces y solape de 0,004, así que la
oclusión mutua no es el problema que sería en sable.

Lo que **no** está medido es si los **640 píxeles** a los que corre la pose pierden detalle frente a la
resolución nativa del vídeo. **Nadie lo ha comprobado nunca** y está pedido: si los pierde, es una mejora de
precisión de pose a coste de GPU, y hay que decidirlo con la cifra.

---

## 4. Lo que se guarda y lo que se regenera

| qué | tamaño | decisión |
|---|---|---|
| recortes 128×128 uint8 | ~45 GB | **se guardan** |
| pose a cadencia nativa, clip completo | ~2 GB | **se guarda** |
| índices, reloj, columnas de resumen | <1 GB | **se guarda** |
| flujo óptico denso | ~550 GB | **NO se guarda** |

El flujo **no se guarda porque se regenera exacto**: recalcularlo desde los recortes reproduce la ruta de vídeo
**bit a bit** en 394 ms por clip contra 1.250. Guardar 550 GB para no gastar 394 ms no sale.

Total ~50 GB, que **caben en los 100 GB privados gratuitos** de Hugging Face. Por eso no hace falta un repo
público, con lo que ni se redistribuye vídeo de terceros ni se publican nombres de tiradores.

---

## 5. El principio, que es tuyo

> «Aunque igual pienses que no aporta, por tener esa información. Si luego no la usamos en el entrenamiento no
> pasa nada, pero dejar las cosas guardadas ayuda mucho.»

Correcto, y hay evidencia de esta noche que lo respalda. Varias señales están **cerradas como rasgo de
prioridad** y eso **no** significa que sean falsas:

- el **centro de masas** le gana por uno a dos puntos a una copia de sí mismo con el lado destruido, o sea que
  su señal es **real**; está cerrada porque el modelo ya la tenía por otra columna (r = 0,877);
- el **flujo del corredor** sí lleva información de prioridad, 63,23 % contra el 50 % del azar; está cerrada
  porque es redundante con el esqueleto;
- el **aterrizaje del pie delantero** se resuelve en los dos tiradores en el 81,7 % de los clips: la cobertura
  nunca fue la limitación.

Todas se guardan. Con una condición: **cada señal viaja con su definición**, su unidad, su cobertura medida con
su n, su precisión o un «sin referencia» explícito, y la distinción entre **ausente** e **indefinido**.

Sin eso pasa lo que ya ha pasado dos veces hoy: **el mismo nombre con un número distinto.** El reloj dio
«53,1 % parados» con un umbral heredado y **16,4 %** con el correcto. Y el suavizado de etiqueta dio **+1,03 pp**
con un convenio de promediado y **+0,80 cruzando el cero** con el otro. Mismo nombre, veredicto opuesto.

---

## 6. Lo que sigue pendiente de medir, y no se promete

- **si los 640 px de la pose pierden detalle** frente al vídeo nativo, y qué cuesta subirlo;
- **el tamaño y el tiempo reales** de la extracción a cadencia nativa, sobre una muestra que incluya 50 y 60 fps;
- **el jerk** como señal, que nunca se ha probado y cuyo aviso es que el nulo circular **infla las derivadas
  hasta 7,79 veces**, así que necesita el nulo de signo;
- **cuánto mejora la estatura** por la vía de la escala del allez;
- y **el acierto de prioridad**, que ninguna de estas mejoras ha demostrado mover todavía.

---

## 6 bis. Las tres señales nuevas: una entra, dos se cierran

Probadas el 2026-09-09. Las tres **se extraen igualmente**, que es tu criterio.

### La que entra: velocidad angular del antebrazo

Primero una corrección al planteamiento: **el ángulo ya existía** (`forearm_ang`). Lo que no existía es **su
derivada** — hay velocidad de muñeca, aceleración y extensión de brazo, pero no la derivada del ángulo del
segmento.

Y **no es redundante**: **mean |r| máximo 0,166** contra las columnas de movimiento de mano. El suavizado hace
lo que tenía que hacer contra el telecine: el percentil 99 de la velocidad baja de **2.388,9 a 1.483,7 °/s**,
un factor **1,61** de artefacto eliminado en la banda de 30 fps.

**Y su punto débil ya está arreglado, con la medida de cuánto importaba.** Elegía la muñeca por confianza
media, un apaño; sustituido por `clip_stats.dominant_hand`, que resuelve **48 de 48 (100 %)** con confianza
mediana **0,750** (3 casos por debajo de 0,60, que viajan marcados). El apaño **elegía la muñeca equivocada en
15 de 48 clip×lado, el 31,3 %**: un tercio de los casos medía el brazo contrario. No era cosmético.

Al arreglarlo la redundancia **sube, como cabía esperar, y la columna que importa sobrevive**:

| columna nueva | contra | antes | ahora |
|---|---|---:|---:|
| `fa_ang_sg` | `forearm_ang` (que ya existía) | 0,468 | **0,572** |
| **`fa_ang_vel_sg`** | `wrist_vy` | 0,166 | **0,162** |
| `apunta_rival_sg` | `wrist_x` | 0,536 | 0,620 |

Que `fa_ang_sg` se parezca **más** a `forearm_ang` es la dirección correcta: por fin miden el mismo brazo. Y
**el 0,162 se mantiene con la muñeca correcta**, que es el que decide, porque el ángulo ya existía y la señal
nueva es **la derivada**. Cobertura 0,828.

### La que falla su control, y es instructivo cómo

El eje del antebrazo apuntando al rival, como sustituto de «punta en línea». Criterio declarado **antes** de
mirar: el ángulo tiene que ser **menor en el tocado que en guardia**.

| | con el apaño | **con la muñeca correcta** |
|---|---:|---:|
| mediana pareada (guardia − tocado) | −15,0° | **−2,9°** |
| con el signo esperado | 12/36 (33,3 %) | **17/36 (47,2 %)** |
| veredicto | **no pasa** | **no pasa** |

Arreglar la muñeca la mueve de «falla claramente» a **indistinguible del azar**. Confirma que el error de
muñeca era real y **no rescata la señal**.

**Y aquí hubo una trampa estadística que casi se reporta al revés.** Las medianas **agrupadas** dicen que el
tocado (24,0°) es menor que la guardia (29,4°), o sea lo **contrario** que la diferencia **pareada** (−2,9°).
Agrupar mezcla clips con distinto número de fotogramas, así que los clips largos mandan en el montón. La
unidad de análisis es el **clip×lado, nunca el fotograma**, y citar la agrupada habría reportado un «pasa»
falso.

**Y lo importante es por qué:** el control se calculaba **para los dos tiradores**, y el que *no* puntúa no
tiene ninguna razón para estar apuntando en el instante del tocado. **Restringirlo al que puntúa haría pasar
el control — y eso es condicionar por el resultado**, que es justo lo que compró +3,08 pp con un bloque sin
información. Así que el control se queda fallado.

Dos explicaciones alternativas descartadas antes de llegar ahí: **no es un sustituto de la distancia**
(r global +0,155, mediana por clip×lado +0,046, deciles no monótonos), y el mal condicionamiento en el
contacto es solo parcial.

**Lo que lo explicó no fue ninguna tabla, fueron las figuras:** el eje prolongado apunta al suelo y al fondo en
varios fotogramas, donde **codo y muñeca casi coinciden por escorzo**. Con esa distancia pequeña la dirección
es ruido, y el IQR lo delata: **99,9° contra 53,9°**.

### La que es un límite del corpus, no del detector

La deliberación del árbitro: **2 de 24 clips, el 8,3 %**. Mediana donde se puede medir: **1,56 s**.

**Y el motivo es el hallazgo: el clip se corta antes de que el árbitro reanude el reloj.** Con la ventana
antigua de 0,3 s tras el tocado no habría **nada** que medir; con duración completa hay algo, pero **la
duración completa es necesaria y no suficiente**. Extraer mejor no lo arregla: hacen falta clips cortados más
tarde.

### Y un tercer operador que destruye lo que mide

**`np.unwrap` es acumulativo, así que un solo `NaN` envenena todo lo posterior.** Aplicado a la serie entera
bajó la cobertura de la velocidad angular de **0,846 a 0,577 sin que faltara un solo dato**: el operador la
estaba borrando. Se arregla desenvolviendo por tramos finitos contiguos.

Van tres en la sesión, y comparten un rasgo: **una operación cuya salida parece plausible mientras su supuesto
de entrada está violado.** El nulo circular preservando las columnas de nivel, la mediana del clip contaminada
por la repetición, y este.

---

## 7. El riesgo abierto número uno: contaminación silenciosa por repeticiones

Esto es lo peor que se ha encontrado hoy, y va aquí arriba porque **no tiene arreglo todavía y ninguna
comprobación existente lo caza**.

Un clip real (`PeriodicSlipperyChipmunk`, 50 fps, Tokio) **abre con primeros planos de repetición**, con
cortinilla de anillos olímpicos, y luego corta al plano en vivo. La cobertura de pose es **1,000 en los dos
tiradores**. O sea que el detector los encuentra perfectamente, porque **están ahí de verdad**, solo que con
otro encuadre y otra escala. Esos fotogramas entran como poses válidas **sin ninguna marca**.

**La cobertura bajando era nuestra alarma. Aquí no salta ninguna.** Y `cuts` informa de 1 corte donde se ven
tres planos.

Se probaron cuatro vías y **ninguna discrimina**:

| vía | resultado |
|---|---|
| normalizar la altura de caja por la mediana del clip | **0 fotogramas marcados** |
| normalizar por la ventana del tocado | **1,58** contra **1,32** de un clip limpio: se solapan |
| contar cortes de plano | informa 1 donde hay 3 |
| cruzar con el reloj parado | 20,0 % contra 20,5 %, intervalos solapados (n = 184) |

El primer fallo es el instructivo: **la repetición es la mitad del clip, así que la mediana del clip está
contaminada por lo que se pretende detectar.** Es el mismo error de clase que el nulo circular preservando las
columnas de nivel.

**Y no se bajó el umbral de 1,60 a 1,55 para que el control positivo pasara.** Eso habría sido inventar la
medida. Lo que se entrega es **`escala_r` como columna continua cruda** — definida en el 91,8 % de los
fotogramas, `NaN` = indefinido y no cero, **precisión: sin referencia** — y la detección de repeticiones
**como hueco declarado**.

### Y el confusor de sede que hay que tener presente

Los **877 clips de cadencia alta salen de tres sedes**, y **637 (72,6 %) son `tokyoolympics2020`**. Los 154
clips con corte de plano detectado son **todos de la misma sede**. Así que **cualquier comparación entre bandas
de cadencia arrastra el confusor de sede** y no se puede leer como un efecto de los fotogramas por segundo.

Es la tercera vez que un confusor de sede muerde en este proyecto: la tasa de detección por sede ya midió
**cuánto se había rebuscado en el residuo**, no la calidad de la sede.

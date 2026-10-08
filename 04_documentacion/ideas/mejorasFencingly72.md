 
Trabajar con el material que ya existe (vídeos de retransmisión estándar a 25/30 fps, sin repeticiones a cámara lenta y con la variabilidad de los directos) es el escenario real en el 95% de los proyectos de Computer Vision aplicado al deporte. 

El informe de 2026 dejó claro por qué el modelo actual toca techo en ~73.6%: **el sistema es ciego a la interacción de las hojas y depende exclusivamente de 17 puntos COCO del cuerpo**. En florete, dos tiradores pueden hacer exactamente el mismo movimiento corporal, pero si uno desvió la hoja del otro 5 centímetros antes del impacto, la prioridad cambia por completo.

A continuación, tienes una propuesta estructurada con **las vías de vanguardia más potentes (State-of-the-Art)** que no han sido descartadas en el informe, diseñadas específicamente para exprimirse sobre vuestro dataset actual sin necesidad de mejor vídeo.

---

### Bloque 1: Las dos victorias inmediatas de ingeniería (Ganancia directa)

Antes de entrenar redes complejas, hay dos problemas que el propio informe de 2026 diagnosticó con precisión matemática pero dejó sin resolver:

1. **Parchear los 4 templates de marcador para las 5 sedes ciegas (Sección 2.2 y 5.3):**
   * **Para qué sirve:** En sedes como Tokio 2020 o París 2022, el lector de luces busca en una coordenada $y$ fija donde no está la bombilla, recurriendo a un detector genérico de color que cuesta **4.93 puntos porcentuales (pp)** de precisión. Además, estas sedes están en el conjunto de test, sesgando la evaluación.
   * **Cómo sacarlo:** No hace falta IA. Es un diccionario de configuración de 5 entradas: para cada una de las 5 sedes problemáticas, definid las coordenadas de la caja delimitadora (*bounding box*) del marcador en su posición real (cápsula, celda que cambia de color, punto junto al dígito). Recuperaréis de golpe cerca de **+5 pp en esas sedes** y el test dejará de estar penalizado artificialmente.
2. **Arreglar el bug del fotograma de guardia (*on-guard segment*, Sección 8.5):**
   * **Para qué sirve:** 12 variables cinemáticas de guardia se están calculando desde el fotograma 0 del clip en lugar de contar hacia atrás desde el tocado. Si el vídeo incluye cortes de plano al inicio, esas variables leen basura.
   * **Cómo sacarlo:** Modificar la indexación en los 6 módulos afectados (`armwork.py`, `footwork.py`, etc.) para que la ventana de guardia sea siempre relativa al instante del tocado: `[touch_frame - 4.0s : touch_frame - 3.5s]`.

---

### Bloque 2: Romper la ceguera de la hoja (Modelo Híbrido Esqueleto + RGB)

En la Sección 7.6 se probaron varios modelos alternativos (GNNs, k-NN, etc.), pero **todos usaban solo las coordenadas del esqueleto**. Se descartó el 100% de la información visual. La hoja de acero no se puede segmentar a 30 fps, pero los píxeles del vídeo **contienen los destellos metálicos, el flujo óptico del choque y el impacto en el peto eléctrico**.

#### La idea: El "Tubo Visual de Combate" (Dynamic RoI Video Transformer)
* **Para qué sirve:** Permitir que una red neuronal profunda vea la interacción entre las hojas y los brazos sin verse afectada por el fondo del pabellón ni los videomarcadores.
* **Cómo sacarlo paso a paso:**
  1. **Recorte dinámico (*Spatial RoI*):** En lugar de pasar el fotograma entero de 1080p, utiliza las coordenadas de YOLO-pose de ambos tiradores (hombros, codos y muñecas) para definir una caja delimitadora dinámica que solo contenga el **"corredor de interacción"** (el espacio que hay entre los dos pechos y las manos armadas).
  2. **Ventana temporal crítica:** Recorta solo el segundo decisivo previo al tocado ($t \in [-1.2\,\text{s}, 0\,\text{s}]$), lo que supone apenas 30–36 fotogramas.
  3. **Backbone de vídeo SOTA:** Pasa ese clip recortado por un modelo fundacional de vídeo preentrenado, como **VideoMAE V2**, **TimeSformer** o **InternVideo2** (congelando el extractor y entrenando solo capas superiores o usando adaptadores tipo **LoRA** para evitar sobreajuste).
  4. **Fusión:** La red visual generará un vector de características latentes (embedding) que describe si hubo cruce de hojas o contacto. Este vector se concatena con vuestro árbol de decisión / red temporal actual.

---

### Bloque 3: Cinemática de Choque Físico ("Micro-Jerk" y derivada de orden 3)

El informe probó la distancia entre muñecas (*wrist-gap*) y la descartó porque correlacionaba al 0.98 con variables ya existentes. Pero la distancia es estática; lo que define una parada es la **fuerza de impacto**.

#### La idea: Índice de Disrupción por Impacto (PIDI - Parry Impulse Disruption Index)
* **Para qué sirve:** Detectar paradas y batimientos a través de la reacción biomecánica involuntaria de la mano armada (Tercera Ley de Newton), sin necesidad de ver la hoja.
* **Cómo sacarlo paso a paso:**
  1. **Filtrado Savitzky-Golay:** Las coordenadas de la muñeca armada a 30 fps tienen ruido de detección. Aplica un filtro Savitzky-Golay (ventana de 5 a 7 frames, polinomio de grado 3) sobre las trayectorias de la muñeca.
  2. **Calcular el *Jerk* (Tirón cinemático):** Deriva tres veces la posición: 
     $$\vec{J}(t) = \frac{d^3 \vec{x}_{\text{muñeca}}}{dt^3}$$
  3. **Descomposición vectorial:** Descompón el vector de aceleración/jerk en dos ejes:
     * *Eje longitudinal:* Hacia el oponente (avance/extensión).
     * *Eje transversal:* Perpendicular a la pista (desviación lateral de la mano).
  4. **Firma de la Parada:** Cuando hay un choque de hojas en una parada de cuarta o de sexta:
     * El atacante sufre una **desaceleración longitudinal brusca con un pico repentino de jerk transversal involuntario**.
     * El defensor muestra una aceleración controlada en arco seguida de una inversión hacia delante (la respuesta).
  5. Crear 4 variables nuevas basadas en la correlación cruzada de los picos de jerk de ambas muñecas en los 400 ms previos al encendido de las lámparas.

---

### Bloque 4: Hitos Biomecánicos de la Regla t.83 (La Caída del Pie Delantero)

El reglamento de la FIE (artículo t.83) establece una regla de oro que los árbitros aplican rigurosamente: **el ataque finaliza en el momento exacto en que el pie delantero toca el suelo** al hacer el fondo (*lunge*) o el paso adelante. Si el tirador A estira el brazo pero su pie delantero ya ha aterrizado y el tirador B inicia después su ataque y ambos tocan, la prioridad es de B (porque el ataque de A ya había terminado).

* **Para qué sirve:** Transformar el reglamento en una variable física binaria de alta discriminación.
* **Cómo sacarlo paso a paso:**
  1. **Detección del *Heel-Strike* (Impacto del talón):** Mediante las coordenadas de los tobillos/talones de YOLO, calcula el momento exacto $t_{\text{suelo}}$ en el que la velocidad vertical del pie delantero cae a cero y su posición vertical toca el suelo de la pista.
  2. **Detección de la Extensión Máxima:** Identifica el momento $t_{\text{ext}}$ en el que la distancia hombro-muñeca armada alcanza su percentil 95 (brazo completamente estirado).
  3. **Variables de Sincronización:**
     * `delta_t_pie_luz = t_luz - t_suelo` (¿cuánto tiempo pasó entre que cayó el pie y se encendió la lámpara?).
     * `ataque_muerto = 1` si `t_suelo < t_ext` (el pie cayó antes de que el brazo terminara de extenderse: ataque abortado/preparación cortada).

---

### Bloque 5: Explotar los 29.000 vídeos no usados mediante Aprendizaje Autosupervisado (SSL)

El informe indica que el corpus tiene **34.374 vídeos**, pero para entrenar el modelo de prioridad solo se usan **4.958 toques dobles**. El resto (~29.000 clips) están acumulando polvo porque:
* O son toques de una sola luz.
* O no tienen etiquetas válidas de prioridad.

En la Sección 7.5 se intentó transfer learning supervisado desde toques de una luz y fracasó porque la tarea supervisada es diferente. Pero el **movimiento biomecánico de la esgrima es el mismo**.

* **Para qué sirve:** Preentrenar una red para que aprenda cómo se mueven los esgrimistas de élite sin necesitar etiquetas de quién ganó el punto.
* **Cómo sacarlo:**
  * Utilizad un enfoque de **Aprendizaje Autosupervisado (Self-Supervised Learning)** tipo **MotionBERT** o un autoencoder temporal enmascarado (**Masked Motion Autoencoder**) sobre las secuencias de poses de los 34.000 clips.
  * La tarea previa de la red consiste en reconstruir fotogramas o articulaciones que se enmascaran al azar (ocultar un 20% de los puntos clave y obligar a la red a predecirlos).
  * Al hacer esto con 34.000 vídeos, la red aprenderá qué es un fondo, qué es una finta y qué es una preparación con una solidez estadística inmensa. Después, solo hay que afinar (*fine-tuning*) esa red con los 4.958 toques dobles para la tarea de prioridad.

---

### Resumen del Plan de Acción para el equipo (Orden de ejecución)

| Prioridad | Tarea | Esfuerzo | Impacto esperado |
| :--- | :--- | :--- | :--- |
| **1. Inmediata** | 5 plantillas de overlay para sedes ciegas + corrección bug indexación `on-guard`. | 2–3 días | **+1.5 a +2.5 pp netos** en el corpus global (+5 pp en sedes ciegas). |
| **2. Cinemática** | Extracción de picos de *Jerk* (derivada 3) en muñecas y sincronización de caída de talón (regla t.83). | 1 semana | **+1.5 a +3.0 pp** en el modelo tabular actual. |
| **3. Visión SOTA** | Recorte dinámico del "Tubo Visual de Combate" + Feature Extractor con VideoMAE/TimeSformer congelado. | 2–3 semanas | **+4.0 a +7.0 pp** (rompe la barrera de la hoja y los brillos de contacto). |
| **4. Fundación** | Preentrenamiento autosupervisado (SSL) en los 34.000 vídeos con enmascaramiento de esqueleto. | 3 semanas | Robustez masiva frente a nuevas sedes y eliminación de sobreajuste. |

Con las dos correcciones iniciales y la inyección del tubo visual recortado sobre las acciones clave, el modelo tiene el camino despejado para **superar el 80% en LOTO estricto sin abstención**, y situarse en el **88–92% bajo la puerta conformal** (Protocolo B).

Aquí tienes **5 vías adicionales de vanguardia (State-of-the-Art)** que atacan directamente los ángulos ciegos del sistema actual, cumplen con no usar cámara lenta y aprovechan la física de la esgrima y las arquitecturas de IA más modernas:

---

### 1. Detección del *"Temps d'Arrêt"* mediante Cinemática del Centro de Masas (CoM)

En el arbitraje de florete moderno, la decisión más difícil y polémica para un árbitro humano es distinguir entre un **ataque continuo** y un **ataque sobre la preparación** (Regla t.89 de la FIE). Si el tirador que avanza frena ligeramente su cuerpo (*temps d'arrêt*) o vacila con el brazo encogido, pierde automáticamente la prioridad si el oponente inicia en ese instante.

* **Por qué el modelo actual falla aquí:** Medir solo la velocidad del pie o de la cadera da falsos positivos porque los fencers dan pasos cortos, saltos o aceleraciones escalonadas (*balestras*).
* **La solución SOTA:** Estimar el **Centro de Masas Corporal Total (CoM - Center of Mass)** mediante tablas antropométricas de Winter/de Leva aplicadas a los 17 puntos COCO:
  $$\vec{R}_{\text{CoM}}(t) = \sum_{i} w_i \cdot \vec{r}_i(t)$$
  donde cada articulación $i$ aporta su peso relativo de masa corporal (tronco ~50%, muslos ~10%, cabeza/brazos ~10%).
* **Cómo implementarlo para el equipo:**
  1. Extraer la serie temporal continua del vector velocidad del CoM en el eje longitudinal: $v_{\text{CoM}, x}(t)$ en la ventana $[-2.0\,\text{s}, 0\,\text{s}]$.
  2. Aplicar un algoritmo de detección de puntos de cambio (**CUSUM** o **Ruptures** con penalty $L_2$) para identificar si existe un punto de inflexión donde $\frac{dv_{\text{CoM}, x}}{dt} \le 0$ durante la marcha.
  3. Crear la variable booleana de regla: `preparacion_detenida = True` si hay deceleración del CoM mientras la distancia relativa entre hombro y muñeca se reduce (brazo encogido). Si esto ocurre y el rival tiene vector de aceleración positivo hacia delante, la prioridad se invierte hacia el rival.

---

### 2. Recuperación de Malla 3D Invariante a Cámara (HMR 2.0 / 4D-Humans)

El informe demostró que intentar predecir la escala métrica absoluta con profundidad monocular daba un 50% de error (Sección 7.7). Sin embargo, para juzgar la prioridad **no necesitas la métrica del pabellón en metros, necesitas la rotación 3D del cuerpo del tirador en relación al adversario**.
En florete, girar el torso hacia fuera (*se détourner*) o esquivar agachándose (*inquartata / passata sotto*) altera por completo la línea de validez y la intención de ataque.

* **Por qué YOLO-pose 2D se queda corto:** Cuando un tirador se lanza a fondo o se gira en escorzo hacia la cámara, las coordenadas en 2D se colapsan por la perspectiva (el informe midió una inclinación de cámara de 18° y pérdida de proporcionalidad).
* **La solución SOTA:** Aplicar un modelo de **Human Mesh Recovery (HMR)** temporal, como **4D-Humans (HMR 2.0)** o **SMPLer-X**. Estos modelos reconstruyen la malla 3D completa de la persona (formato SMPL) a partir de un único vídeo monocular en tiempo real.
* **Qué extraer concretamente:**
  1. **Vector normal del pecho en 3D ($\vec{n}_{\text{torso}}$):** Te da el ángulo exacto (azimut y elevación) al que apunta el esternón respecto a la línea que une a ambos tiradores.
  2. **Ángulo de ataque tridimensional:** Medir si el vector de la mano armada está alineado con el eje del torso del rival en 3D, lo que define una *"amenaza continua a blanco válido"* según el reglamento, eliminando el sesgo del ángulo oblicuo de la retransmisión televisiva.

---

### 3. Vorticidad en Flujo Óptico Denso (Capturar paradas circulares sin ver la hoja)

Aunque la hoja de 4 mm es casi transparente a 30 fps, los batimientos circulares (*parada de contra-cuarta* o *contra-sexta*) obligan a la cazoleta (la campana de metal de la empuñadura) y a la mano a describir un movimiento helicoidal rápido en el aire.

* **La solución SOTA:** Utilizar un extractor de flujo óptico neuronal rápido como **SEA-RAFT** o **GMFlow** sobre el parche recortado entre los dos tiradores.
* **El cálculo físico (Rotacional / Vorticidad):**
  A partir del campo vectorial de velocidades de píxeles $(u, v)$ en ese parche, calcula la vorticidad $\omega$ (el rotacional en 2D):
  $$\omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$$
* **Por qué es un bombazo técnico:** 
  * Un ataque directo produce un campo de flujo laminar divergente ($\omega \approx 0$).
  * Una parada envolvente o un batimiento circular genera un **pico brutal de rotacional concentrado en el centro del corredor de combate**.
  * No necesitas detectar la espada: el remolino de píxeles en el aire alrededor de la mano delata la parada circular con precisión de milisegundo.

---

### 4. Conformal Prediction Estratificada (Mondrian Conformal)

En el Protocolo B (Sección 5.2 del informe), el equipo descubrió que la puerta conformal cumplía el 90% en agregado (*pooled*), pero **por sede caía hasta el 64.8%**. Esto es un fallo típico de la calibración marginal: el modelo es demasiado optimista en sedes con mala iluminación o tiros de cámara lejanos.

* **Cómo arreglarlo:** Implementar **Mondrian Conformal Prediction (MCP)**.
* **Cómo funciona:**
  En lugar de calcular un único umbral no-conformidad $\hat{q}$ para todo el dataset, divides el conjunto de calibración en grupos o estratos basados en la calidad del vídeo:
  * Estrato 1: Ratio de confianza media de YOLO > 0.85 + resolución alta.
  * Estrato 2: Cruce frecuente de fencers / oclusión alta.
  * Estrato 3: Sedes con detector de luces fallback (las 5 sedes con overlay fallido).
* **El resultado:** El modelo calculará un umbral de abstención específico para cada grupo. En sedes malas, la puerta se endurecerá automáticamente (se abstendrá más a menudo), **impidiendo que ninguna sede individual caiga jamás por debajo del 80%** de precisión cuando el modelo decida hablar.

---

### 5. Arquitectura Siamesa con Espacio Latente Antisimétrico (Self-Supervised Skew-Symmetry)

En la Sección 4.2 del informe se describe que para garantizar que el modelo trate igual la izquierda y la derecha, hacen un promedio antisimétrico al final (a nivel de probabilidad predicha en inferencia). 
Esto es un parche posterior; la red neuronal subyacente sigue aprendiendo representaciones asimétricas internamente durante el backpropagation.

* **La solución moderna:** Diseñar una red siamesa donde la **invariancia antisimétrica esté forzada dentro de la propia función de pérdida**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(y, \hat{y}) + \lambda \cdot \| z(x) + z(m(x)) \|_2^2$$
  donde $z(x)$ es el embedding latente de la jugada, $m(x)$ es la misma jugada con espejo horizontal, y se impone que el vector de la jugada espejo sea exactamente el opuesto ortogonal:
  $$z(m(x)) = -z(x)$$
* **Por qué mejora el modelo:**
  * Obliga al espacio latente a aprender únicamente la física del movimiento relativo (quién avanza respecto al otro), cancelando matemáticamente cualquier artefacto de fondo que dependa de si el tirador zurdo está en la izquierda o en la derecha.
  * Duplica el tamaño efectivo de entrenamiento sin necesidad de pedir ni un solo vídeo nuevo.

---

### Tabla de resumen para pasar a los desarrolladores

| Propuesta | Herramienta / Algoritmo | Qué variable aporta | Problema que resuelve |
| :--- | :--- | :--- | :--- |
| **1. CoM Temps d'Arrêt** | Antropometría de Winter + CUSUM / Ruptures | Caída de aceleración del CoM + retracción de brazo | Distingue ataque directo de "ataque sobre la preparación" (t.89). |
| **2. Malla 3D (HMR)** | 4D-Humans (HMR 2.0) o SMPLer-X | Orientación 3D del esternón ($\vec{n}_{\text{torso}}$) | Corrige el escorzo de la cámara de 18° y detecta esquivas/giros ilegales. |
| **3. Vorticidad Óptica** | SEA-RAFT en la caja de conflicto | Rotacional $\omega = \partial v/\partial x - \partial u/\partial y$ | Detecta paradas y batimientos circulares en el remolino de píxeles. |
| **4. Mondrian Conformal** | Calibración conformal estratificada | Umbrales de decisión $\hat{q}_k$ por calidad de sede | Elimina el suelo del 64.8% en sedes malas; garantiza el 80%+ condicional. |
| **5. Antimetría Latente** | Loss antisimétrica siamesa en PyTorch | Proyección latente forzada $z(m(x)) = -z(x)$ | Evita que la red memorice características asimétricas de los pabellones. |
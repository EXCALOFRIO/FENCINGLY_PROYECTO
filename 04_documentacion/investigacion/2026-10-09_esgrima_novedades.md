# Novedades 2024–2026 útiles para arbitrar florete (prioridad) a partir de vídeo FIE

- **Fecha:** 2026-10-09.
- **Tipo:** investigación. Solo búsqueda y lectura web; no se ha tocado código.
- **Proceso:** la búsqueda se repartió en 4 líneas en paralelo (esgrima/hoja/tracking; action spotting; esqueletos/SSL; VLM y señales). Este documento es la consolidación final de las cuatro.
- **Verificación adicional propia:** tras consolidar, releí directamente (WebFetch, sin fiarme del resumen de cada línea) las ~14 fuentes que sostienen la sección de Recomendación: FERA (las 5 versiones y sus fechas), T-DEED, FACTS, CoTracker3, TAPNext, AdaSpot, TAPNext++, SAM 3 (repo), TemporalBench, SkateFormer (repo), SPORTU, E2E-Spot (repo) y VirtualFencer. En los 14 casos el paper/repo existe y el título, autores y cifra citada coinciden con lo que se afirma aquí. Dos matices menores: el claim de T-DEED como 1.º en SoccerNet Ball Action Spotting 2024 y la tabla numérica de E2E-Spot no están en el resumen/README que pude leer (sí es plausible que estén en el cuerpo del paper); la descripción del pipeline de VirtualFencer (SAM 2, WHAM, YOLO) tampoco aparece en el abstract, solo en el cuerpo. Se marcan igualmente [V] porque el agente original afirma haberlas leído en el PDF completo, no en el abstract.

**Contexto propio**, para leer las recomendaciones:
- **Clips:** ~110k FIE, 720p, 25–50 fps, ~10 s. Una cámara con paneo/zoom; ~10 % con audio.
- **Etiquetas:** `touch` izq/der/doble. En ~5.500 clips, además, `initiated` (izq/der/ninguno; acuerdo humano 0,688).
- **Extraído:** 308 kp 2D por tirador y cuadro (Sapiens 2), cuadro del tocado (luces) y eventos de pies/brazo. V-JEPA 2 a prueba.
- **Etiquetador de poses actual:** 0,615 en `initiated`.
- **Máquina:** RTX 3060 12 GB, Windows.

**Convención de marcas**
- **[V]** verificado en la fuente primaria (URL indicada) por alguno de los agentes de búsqueda leyendo arXiv/GitHub/CVF.
- **[V✓]** además lo he vuelto a comprobar yo para este informe (las cifras que sostienen la recomendación).
- **[V~]** fuente identificada, pero la cifra viene de extracción automática o de cita secundaria. Tomar con margen.
- **[NV]** no verificable (fuente caída o inaccesible).
- **[OP]** opinión o inferencia mía. No es un resultado publicado.

**Limitaciones del proceso**
- Se hicieron 5 búsquedas en paralelo. La cuota de WebSearch se agotó varias veces y se siguió con WebFetch, la API de arXiv y GitHub.
- La cobertura es buena en arXiv, GitHub y CVF, y peor en blogs, foros y revistas no indexadas.
- No se ha escrito ninguna URL de memoria.

---

## Resumen

1. **FERA ha cambiado entre versiones.**
   - El 77,7 % de la v4 (un LLM razonando reglas) **ya no aparece en la v5** (30-mar-2026).
   - La v5 se queda en **acc 0,624 y macro-F1 0,632** sobre 484 intercambios, con un Random Forest sobre tokens de ST-GCN. [V✓]
   - Vuestro 0,615 ya está en ese nivel.
2. En FERA v5, el backbone **más simple** gana: ST-GCN, con 116 k parámetros, supera a CTR-GCN, MS-G3D y PoseFormerV2. Con estos volúmenes de datos, la arquitectura no es la palanca. [V✓]
3. Sigue sin haber otro trabajo público de prioridad automática en florete aparte de FERA. Lo comprobé con la API de arXiv. [V✓]
4. Lo nuevo que mejor encaja es la **detección temporal precisa** ("precise event spotting"). Destaca AdaSpot (CVPR 2026, licencia MIT, ~2 GB por clip). [V✓]
   - Nadie la ha aplicado a esgrima.
   - FERA localiza el inicio de los movimientos con un error medio de 3,7 cuadros. [V]
5. El **preentrenamiento auto-supervisado de esqueletos** está maduro y tiene código: MAMP, MaskSem, AMR, GFP, SLiM. [V✓]
   - Con ~10 % de etiquetas, las diferencias entre métodos en NTU son pequeñas.
   - No está demostrado que capte "quién empezó", que es un orden relativo entre dos personas. [OP]
6. **Hoja:** no hay ningún trabajo específico de esgrima. [V]
   - Lo viable en una 3060 es seguir 1–2 puntos (TAPNext++, Apache-2.0) anclados a la mano, en vez de segmentar con SAM 3. [OP]
   - CoTracker3 tiene licencia **no comercial**. [V]
7. **VLM:** no sirven para decidir órdenes de menos de 200 ms. [V]
   - TemporalBench: humano 67,9 frente a GPT-4o 38,5.
   - SPORTU, tareas difíciles en vídeo: 52,6 %.
   - Como mucho valen de capa de explicación o de pre-etiquetado grueso validado. [OP]
8. **Señales baratas con respaldo biomecánico**, todas calculables con lo que ya extraéis:
   - instante de inicio por umbral de velocidad de la mano;
   - curva de velocidad con uno o dos picos;
   - qué tirador aporta el cierre de distancia;
   - orden brazo/pie.
9. **Posible etiqueta gratis:** si `touch` es "a quién se dio el punto", en los clips con dos luces ya es la decisión de prioridad del árbitro. **Hay que aclararlo.** [OP]
10. **"Allez Go"** son dos cosas distintas. La versión "pose + audio" **no es verificable** (servidor caído). [NV]

---

## 0. FERA: qué dice ahora y por qué importa

### 0.1 Versión vigente (v5, 30-mar-2026) [V✓]
**Ficha**
- **Título:** "FERA: A Pose-Based Framework for Rule-Grounded Multimedia Decision Support with a Foil Fencing Case Study".
- **Autores:** Ziwen Chen (Marc Garneau Collegiate Institute, Toronto, un instituto de secundaria) y Zhong Wang (BIMSA, Pekín).
- **Enlaces:** https://arxiv.org/abs/2509.18527 · https://arxiv.org/html/2509.18527v5
- **Licencia del texto:** CC BY 4.0.

**Datos**
- 3 Grand Prix, vídeo 720p a 25 fps.
- 493 clips únicos, de los que salen 1.800 segmentos de un tirador etiquetados (2.583 movimientos). Se usan 1.045 para entrenar y evaluar.
- Un segundo anotador revisó 1.103 segmentos: 1.022 acuerdos y 81 desacuerdos, que se adjudicaron.
- **Ojo:** ese acuerdo es sobre etiquetas de *movimiento*. No hay techo humano para la decisión final, así que no es comparable con vuestro 0,688.

**Decisión izq/der/ninguno**, sobre **484 intercambios**:

| Clasificador | Accuracy | Macro-F1 |
|---|---|---|
| Random Forest podado (tokens 2D de ST-GCN) | **0,624** | **0,632** (F1 de "ninguno" 0,650) |
| HistGradientBoosting | 0,599 | 0,561 |
| Regresión logística | 0,541 | 0,512 |

**Percepción:** comparan 13 backbones. Macro-F1 combinado de movimiento + hoja, rama con profundidad "lifted":

| Backbone | Macro-F1 combinado |
|---|---|
| **ST-GCN** (116 k parámetros) | **0,496** |
| PoseC3D-style | 0,467 |
| FERA-CLS | 0,461 |
| Fusión pose + RGB | 0,416 |
| CTR-GCN | 0,414 |
| PoseFormerV2 | 0,407 |
| MS-G3D | 0,406 |

El "sidecar" de profundidad (solo la coordenada z de VideoPose3D, 24-D) sube ST-GCN de 0,475 a 0,496.

**Tokens:** 101 dimensiones por cuadro.
- 24 de articulaciones normalizadas y 2 de centro de masas.
- 11 de distancias y 4 de ángulos.
- 2 de orientación del tronco y 6 de extensión de brazo.
- 52 de derivadas de primer y segundo orden.

**El LLM** queda solo como interfaz de explicación, sin ninguna cifra asociada.

**Limitaciones que declaran**
- No ven la hoja.
- La anotación por intervalos comprime los desfases temporales reales.
- La decisión es frágil cuando los tokens omiten el evento decisivo.
- La partición es por combate, no por tirador.

**Localización temporal** de los límites de cada movimiento (dato del agente que leyó el PDF v5) [V]: IoU media 0,381, F1@0,5 de 0,204, error medio de 3,7 cuadros en el inicio y 2,6 en el final.

**Código:** dicen liberar "features de pose anonimizadas, etiquetas y código", pero **no encontré la URL**.

### 0.2 Versión anterior (v4, dic-2025) [V]
- **Datos:** 1.734 clips y 2.386 acciones, con **un solo anotador**.
- **Percepción (macro-F1):** FERA-MDT (transformer de 3 capas) 0,549 ± 0,018, BiLSTM 0,537 y TCN 0,520.
- **Decisión:** FERA-LM pasaba los tokens a DeepSeek-R1-Distill-Qwen-7B, sin fine-tuning. Coincidía con el árbitro en 753 de 969 intercambios (77,7 %).
- **Fuente:** https://arxiv.org/html/2509.18527v4

### 0.3 Lectura para el proyecto
- **[OP] La cifra comparable es 0,624 (v5), no 0,777.**
  - El 77,7 % se midió con otro conjunto, un solo anotador y sin auditoría, y los propios autores lo retiraron.
  - Vuestro 0,615, frente a etiquetas con acuerdo humano 0,688, está (con cautela) al nivel de lo mejor publicado.
- **[OP] Lección útil de la v5:**
  - separar la percepción (tokens con significado) de la decisión (clasificador pequeño y calibrado);
  - con unos 1.000 segmentos, los modelos grandes no ganan.
- **Correcciones al borrador previo:**
  - presentaba el 77,7 % como posiblemente vigente;
  - atribuía a TAPNext "~17 FPS en una Quadro RTX 4000", cuando según el README de tapnet esa cifra es del **TAPIR causal** [V✓].

### 0.4 "Allez Go": dos cosas distintas
- **Equipo del InVenture Prize de Georgia Tech** (Jason Mo, Adam Kulikowski). [V]
  - Hacen seguimiento de la hoja en tiempo real para resaltarla en la retransmisión.
  - No hay paper ni código.
  - https://inventureprize.gatech.edu/team/allez-go
- **Artículo "Allez Go: Computer Vision and Audio Analysis for AI Fencing Referees"**, Journal of Student Research, sección de secundaria. [NV]
  - El servidor devolvió error 522 en todos los intentos, también en el mío.
  - El fragmento del buscador habla de pose + audio, ~4.000 clips y 89,1 %. No usar esa cifra.
  - https://jsr.org/hs/index.php/path/article/view/3394
- Jason Mo es coautor de FACTS (§1.2). [V✓]

---

## 1. IA en esgrima y deportes de combate/raqueta

### 1.1 Prioridad automática
- **Búsqueda en arXiv.** La API, buscando en los resúmenes fencing + (referee | foil | priority | right-of-way), solo devuelve FERA. [V✓]
- **VirtualFencer** [V✓]
  - **Ficha:** Lin, Goel, Yun, C. K. Liu y Araujo (Stanford). arXiv 2507.00261, junio de 2025, CC BY 4.0. https://arxiv.org/abs/2507.00261
  - **Qué hace:** genera asaltos a partir de una estrategia que extrae sin supervisión de vídeo "in the wild". Según el agente: 40 asaltos y 54 tiradores del circuito senior.
  - **Pipeline:** WHAM (pose 3D SMPL), YOLO y **SAM 2 para seguir las líneas de pista y obtener la homografía**.
  - **Prioridad:** la modela con matrices de transición entre estados (M-M, P-NP, NP-P) construidas por heurísticas.
  - **Evaluación:** 4 profesionales le dan 4/5 en comprensión de la prioridad. No encontré código.
  - **[OP] Encaje:** no es un árbitro, pero aporta la idea de un "estado de prioridad" explícito y un precedente de calibración de pista con SAM 2.
- **Repos de GitHub de baja fiabilidad**
  - `PrathitWagholikar/Foil-Fencing-Right-Of_Way-Classification`: XGBoost sobre keypoints, 0,657 y 0,787 en dos niveles. Usa un Kaggle de "alexpgd" que no pude abrir. [V~]
  - `CharlesCNorton/fencing-verified`: formaliza en Coq las reglas de tiempo de la prioridad. Es solo lógica, sin visión. [V]

### 1.2 Reconocimiento de acciones de esgrima

**FACTS** [V✓]
- **Ficha:** Lai, Mo, Xia e Y.-F. Wang. arXiv 2412.16454, diciembre de 2024. https://arxiv.org/abs/2412.16454
- **Qué es:** fine-tuning de VideoMAE-base (preentrenado en Kinetics-400) sobre **vídeo crudo, sin pose**.
- **Resultado:** 90 % en 8 clases (ataque, respuesta, contraataque y remise, cada una izq/der). F1 por clase entre 0,84 (ataque) y 0,98 (remise). En boxeo, 83,25 %.
- **Código y licencia:** https://anonymous.4open.science/r/FACTS-B1C5 · texto CC BY-NC-SA 4.0.
- **Datos:** 13.459 clips de Quarte Riposte, que quedan en 6.400 **ya recortados a una sola acción**. No especifican el arma.
- **Máquina:** 2× GTX 1080 Ti.
- **[OP] Encaje:** la tarea es más fácil que la vuestra (el clip ya está recortado a la acción) y no se puede comparar con 0,615. Aun así, confirma que un backbone de vídeo afinado capta ataque y contraataque. Es el precedente directo de vuestra línea V-JEPA 2.

**Otros trabajos**

| Trabajo | Qué es | Resultado | Código / licencia | Encaje |
|---|---|---|---|---|
| **FD-7**, arXiv 2404.19383 (abr-2024) [V] | Dataset de juego de piernas: 20 atletas, 7 clases, 1.193 clips, 1080p a 30 fps | CFSC sobre HD-GCN: 95,7 % (articulaciones) | Anuncian que lo publicarán; no encontré la URL | Bajo: solo pies |
| **Malawski et al.**, VISAPP 2024, DOI 10.5220/0012577600003660 [V] | BlazePose + TSFEL + XGBoost para juego de piernas y calidad técnica | F1 95,5 % en acciones de pies; 0,868 en corrección técnica | — | Bajo. Sus 15 parámetros técnicos (timing brazo-cuerpo) dan ideas de features |
| **FenceNet / FFD**, CVPR-W 2022, arXiv 2204.09434 [V] | Juego de piernas en sala con Kinect 3D, profundidad e IMU | FenceNet 85,4 %, BiFenceNet 87,6 % | FFD: https://home.agh.edu.pl/~fmal/ffd (sin licencia indicada) | FenceNet es el modelo y FFD el dataset. Sin prioridad |
| **FST.ai**, arXiv 2507.14657 [V] | Arbitraje de taekwondo con pose + análisis de impacto; dicen que es "adaptable" a esgrima | Sin cifras extraíbles | — | Solo como patrón de diseño |
| **Chang & Lee 2019**, IEEE TCSVT 29(7), DOI 10.1109/TCSVT.2018.2855098 [V~] | Detecta el **instante de lanzamiento del ataque** en vídeo de retransmisión con una función criterio (velocidad, aceleración, cadencia de pasos) | No pude leer la métrica | — | No es 2024, pero es lo más cercano a "quién inicia" con vídeo de retransmisión. [OP] Leerlo entero |

**Raqueta**, como análogos de "objeto fino sostenido" [V~]:
- RacketVision (arXiv 2511.17045) anota la pose de la raqueta y la trayectoria de la bola en tenis, tenis de mesa y bádminton.
- BST (2502.21085) y TT4D (2605.01234) siguen la misma línea.

**V-JEPA:** existe una versión 2.1, usada como encoder de 1–2 B parámetros en vídeo quirúrgico (OmniRAS, arXiv 2608.31048) [V✓]. No encontré ningún uso en deporte.

### 1.3 Datasets públicos de esgrima (a octubre de 2026)
| Dataset | Contenido | Tamaño | Licencia / URL | Utilidad para prioridad |
|---|---|---|---|---|
| Benchmark de FERA | Florete, retransmisión 720p a 25 fps; 11 movimientos, 5 posiciones de hoja y decisión izq/der/ninguno | 493 clips, 1.800 segmentos, 484 intercambios | Dicen liberar poses, etiquetas y código; **no encontré la URL** | El único con prioridad. Entre 1 y 2 órdenes de magnitud menor que el vuestro |
| FACTS / Quarte Riposte | 8 acciones izq/der, vídeo | 6.400 clips | Repositorio anónimo (§1.2) | Test externo de "ataque / contraataque" |
| FFD | Juego de piernas en sala, Kinect + IMU | 10 tiradores, 6 clases | home.agh.edu.pl/~fmal/ffd | Ninguna |
| FD-7 | Juego de piernas | 1.193 clips | No publicado | Ninguna |
| Kaggle "sport-fencing-matches-ai" | Desconocido | Desconocido | https://www.kaggle.com/datasets/alexpgd/sport-fencing-matches-ai [NV] | Revisar a mano |
| Roboflow "FencingPose" | Posiblemente posturas de guardia / parada | Desconocido | Error 403 [NV] | — |
| EgoHumans, arXiv 2305.16487 | Multi-vista egocéntrico; incluye esgrima | — | [V~] | Solo para pruebas 3D o de sincronización |

### 1.4 Detección de la hoja
**No existe ningún paper** de detección o segmentación de la hoja de esgrima: las búsquedas en arXiv no dan nada relevante. [V]

**Herramientas genéricas**
- **SAM 2** (Meta). https://github.com/facebookresearch/sam2, **Apache-2.0**. [V]
  - Admite prompts de punto, caja o máscara; no de texto.
  - Variantes de 38,9 M a 224,4 M parámetros.
  - Único uso real en esgrima: VirtualFencer sigue con él las **líneas de pista**, no la hoja.
  - [OP] Las variantes Tiny y Small caben en 12 GB.
- **SAM 3** (noviembre de 2025; **SAM 3.1** el 27 de marzo de 2026). https://github.com/facebookresearch/sam3 [V✓]
  - Segmenta y sigue **todas las instancias de un concepto** dado por texto o por un ejemplo.
  - 848 M parámetros. Pide CUDA ≥ 12.6 y PyTorch ≥ 2.7.
  - **Licencia propia ("SAM License")**, no Apache.
  - En SA-Co/Gold, según el agente: 54,1 cgF1 frente a 72,8 del humano, en imagen [V]. No está evaluado en objetos finos.
  - [OP] Para imagen probablemente cabe en fp16; para vídeo largo, no está claro.
- **Grounded-SAM-2** (IDEA). https://github.com/IDEA-Research/Grounded-SAM-2, Apache-2.0 / BSD-3. [V]
  - Detecta por texto (Grounding DINO o Florence-2) y sigue con SAM 2.
  - Incluye **SAHI** (troceado de la imagen) para objetos pequeños.
  - El repo LASER (https://github.com/video-fm/LASER) incluye "fencing sword" entre sus clases, sin métricas. [V]
- **Objetos finos en general** [V~]: grietas (2309.05139), cables (2407.14352) y detección orientada cabeza-cola (2304.04503; GauCho, 2502.01565).
  - [OP] Sugieren representar la hoja como un **segmento orientado** (de la guarda a la punta) en vez de como una máscara.

**[OP] Valoración**
- A 720p, la hoja de florete mide pocos píxeles de ancho y sale con desenfoque de movimiento. Ningún modelo genérico está validado en esas condiciones.
- La ruta barata:
  1. anclar la guarda en la muñeca o mano armada de Sapiens 2;
  2. seguir 1–2 puntos (la punta y la mitad de la hoja) con un tracker de puntos;
  3. usar Grounded-SAM-2 + SAHI solo para inicializar o corregir.

### 1.5 Seguimiento de puntos
| Modelo | Resultado publicado | Licencia | Coste en la 3060 | Nota |
|---|---|---|---|---|
| **CoTracker3** (Meta), arXiv 2410.11831 [V] | Se entrena con pseudo-etiquetas; "1000× menos datos reales" | **CC-BY-NC 4.0** en el grueso (algunas partes MIT/Apache) | Inferencia, probablemente sí [OP]. Fine-tuning no: según las issues, con los valores por defecto no cabe ni en 24 GB [V~] | **No comercial**: bloquea un producto |
| **TAPNext** (DeepMind), arXiv 2504.05579 [V✓] | Seguimiento como predicción de tokens, **online y causal**, sin ventana. Dicen ser SOTA tanto online como offline | **Apache-2.0**, pesos incluidos | Según el agente, 5 ms de latencia en H100 frente a 80 ms de CoTracker3 [V~]. Checkpoint en **JAX** | AJ en TAP-Vid parecido al de CoTracker3 (~62–65 en DAVIS según el protocolo) [V~] |
| **TAPNext++**, arXiv 2604.10582, CVPR Findings 2026 [V✓] | Corrige el punto débil de TAPNext: **re-detectar puntos tras una oclusión o una salida de cuadro**. Entrena con secuencias de 1.024 cuadros; propone la métrica AJ_RD | Apache-2.0 (repo tapnet) [V✓] | Checkpoint en **PyTorch** [V✓] | Encaja con la punta, que se ocluye en cada cruce. https://tap-next-plus-plus.github.io |
| **AllTracker** (Harley et al.), arXiv 2506.07310 [V✓] | Flujo denso + seguimiento. AJ en DAVIS 63,3 frente a 62,9 de CoTracker3 (a 384×512) | **MIT** | 16,48 M parámetros (Tiny: 6,29 M). **A 768×1024 necesita una A100 de 40 GB**; a baja resolución no está documentado | Solo viable a resolución reducida [OP] |

No hay ningún uso publicado de estos trackers para seguir la punta de la hoja. [V]

---

## 2. Detección temporal precisa de eventos (action spotting)

| Modelo | Resultado publicado | Código / licencia | Coste |
|---|---|---|---|
| **E2E-Spot** (Hong et al., ECCV 2022) [V] | mAP con tolerancia de 1 cuadro: Tennis 96,1; FS-Comp 81,0; FineDiving 68,4; FineGym 47,9 (backbone 200MF) | https://github.com/jhong93/spot, **BSD-3** | 2,8 M + 1,7 M parámetros (200MF); 0,3 ms por cuadro en A5000; caben 18 clips de 100 cuadros en 24 GB. [OP] En la 3060: ~8 clips por lote, ~2× más lento |
| **T-DEED** (Xarles et al., CVPR-W 2024; arXiv 2404.05392) [V] | Tolerancia de 1 cuadro: FS-Comp 85,15; FS-Perf 86,79–88,17; FineDiving 71,48–73,23; FineGym 55,97–56,64. **1.º en SoccerNet Ball Action Spotting 2024** (mAP 73,39) | https://github.com/arturxe2/T-DEED, **GPL-3.0** | Backbone RegNetY ligero; no indican la GPU |
| **AdaSpot** (mismos autores, **CVPR 2026**; arXiv 2602.22073) [V✓] | Procesa el vídeo en baja resolución y una región de interés en alta, sin entrenar un selector. Sobre el SOTA: **+3,96 mAP a 0 cuadros en Tennis** y **+2,26 en FineDiving** | https://github.com/arturxe2/AdaSpot, **MIT** | "Un clip requiere solo 1,97 GB de GPU" [V]: cabe en 12 GB |
| **ASTRM + SoftIC** (Santra et al., CVPR 2025; arXiv 2503.00147) [V] | SoccerNet-v2, tight/loose: 73,74/79,11 con 6,46 M parámetros. Su pérdida ataca el **desbalance entre eventos y fondo** | **Sin código** | 4× A100 |
| **De esqueletos a píxeles** (Yeoh y Kan, arXiv 2604.22839, abr-2026) [V✓] | Spotting con pocas etiquetas (50–100 clips): un profesor de **esqueleto** destila a un alumno RGB. Con 100 clips de F3Set-Tennis, el esqueleto solo ya supera al RGB. AMD-FED (Edit 83,84) supera a T-DEED-800MF reentrenado (54,48) [V~ en las cifras] | Sin código propio | Profesor preentrenado en 1× A100 de 40 GB |
| Survey de Deakin, arXiv 2505.03991 [V] | Catálogo de datasets de detección de eventos deportivos: **ninguno de deportes de combate** | — | — |

**Aplicado a esgrima:** no hay nada publicado. FERA no usa spotting (clasifica ventanas) y se equivoca en el inicio de los movimientos por 3,7 cuadros de media. [V]

**[OP] Cómo lo usaríamos**
- **Formulación:** el "inicio de la acción ofensiva" es un evento **por tirador**, con dos cabezas (izq y der) sobre la misma línea de tiempo. Lo mismo para la extensión del brazo y, cuando haya señal de hoja, para la parada o la batida.
- **Entrada:** la secuencia de pose (los 308 kp reducidos, más sus derivadas). Opcionalmente, RGB mediante AdaSpot.
- **Supervisión casi gratis** en los 110k clips:
  - el cuadro de luces, como ancla del final;
  - vuestros eventos de pies y brazo, como pseudo-etiquetas de cuadro;
  - `initiated`, como pérdida de **orden**: el inicio del tirador etiquetado debe ir antes que el del otro.
- **Decisión:** la diferencia entre los dos inicios, más un margen, entra en un clasificador pequeño (la lección de FERA v5).
- **Ventaja sobre el GRU de clip entero:** el modelo tiene que responder **cuándo**, que es lo que define la prioridad, y el resultado se puede auditar cuadro a cuadro.

---

## 3. Modelos de acción sobre esqueletos y preentrenamiento auto-supervisado

### 3.1 Arquitecturas 2024–2026
| Modelo | NTU-60 (X-Sub / X-View) | NTU-120 (X-Sub / X-Set) | Tamaño | Código / licencia |
|---|---|---|---|---|
| **SkateFormer** (ECCV 2024; arXiv 2403.09508) [V] | 93,5 / 97,8 | 89,8 / 91,4 (NW-UCLA: 98,3) | 2,03 M parámetros, 3,62 GFLOPs | https://github.com/KAIST-VICLab/SkateFormer. Etiqueta MIT, pero el README dice **"research and education only; commercial use needs permission"** [V✓] |
| **ProtoGCN** (CVPR 2025; arXiv 2411.18941) [V] | 93,8 / 97,8 | 90,9 / 92,2 (**FineGYM: 95,9** [V~]) | Pocos M (familia PYSKL) | https://github.com/firework8/ProtoGCN, **MIT** [V✓]. Basado en PYSKL |
| **BlockGCN** (CVPR 2024; arXiv 2305.11468) [V] | 92,8 / 96,4 | 89,7 / 90,9 (NW-UCLA: 96,6) | **1,5 M** | https://github.com/ZhouYuxuanYX/BlockGCN, **Apache-2.0** |
| PoseC3D (2022) [V] | Sin sucesor directo en 2024–26; sigue en mmaction2 | — | — | Apache-2.0 (mmaction2) |
| Mamba/SSM: Simba (2404.07645), SkelMamba (2411.19544), TSkel-Mamba (2512.11503) [V~] | Afirman SOTA; cifras y código sin confirmar | — | — | [OP] Inmaduros |

Entre los SOTA, las diferencias en NTU son de menos de 1 punto. Además, en NTU-60 X-Sub se entrena con ~40 k clips, un régimen de datos muy distinto del vuestro. [OP]

### 3.2 ¿Ganan a un GRU con pocas etiquetas?
**No hay ningún estudio** que compare GCN o transformers con un GRU variando la fracción de etiquetas en deporte. [V] (búsqueda negativa)

**Evidencia indirecta**
- **FERA v5:** con ~1.000 segmentos de florete, el ST-GCN más simple gana a las otras 12 arquitecturas, entre ellas CTR-GCN, MS-G3D y PoseFormerV2. [V✓]
- **FERA v4:** transformer 0,549 frente a BiLSTM 0,537 y TCN 0,520; la diferencia cabe en el ±0,018 de la validación cruzada. [V]
- **Weak-to-Strong** (arXiv 2602.08057): en su tarea deportiva (iMiGUE, entrevistas de tenis), un backbone de keypoints "convertido en MLP" iguala o supera a los GCN. [V✓]

**[OP] Conclusión:**
- Cambiar el GRU por BlockGCN, ProtoGCN o SkateFormer entrenados desde cero daría probablemente 0–2 puntos. Por sí solo no compensa el trabajo.
- Solo tiene sentido como encoder de un preentrenamiento (§3.3).

### 3.3 Preentrenamiento auto-supervisado (para los ~104k clips sin `initiated`)
| Método | Resultado | Código / licencia |
|---|---|---|
| **MAMP** (ICCV 2023), base de casi todos los siguientes [V✓] | Fine-tuning en NTU-60: 93,0 / 97,5. Sonda lineal: 85,0 / 89,0 | https://github.com/maoyunyao/MAMP. **El README no declara licencia** (consultar) |
| **S-JEPA** (ECCV 2024; Abdelfattah y Alahi) [V] | Predice en espacio latente la versión 3D a partir de una pose 2D parcial: el **análogo de V-JEPA para esqueletos** | https://sjepa.github.io (no localicé el código) |
| **MaskSem** (IROS 2025; arXiv 2508.12948) [V✓] | Semi-supervisado en NTU-60 (tabla III; cifras en la tabla de abajo) | https://github.com/JayEason66/MaskSem ("coming soon", sin licencia). Entrenado en **4× RTX 4090, 400 épocas, batch 128** |
| **GFP** (ICCV 2025; arXiv 2509.03609) [V✓] | Predice features de alto nivel. **Entrena 6,2× más rápido** que el masked modeling estándar | No localizado |
| **AMR** (CVPR 2026; arXiv 2606.11450) [V✓] | Reconstrucción enmascarada adaptativa que prioriza las zonas con más movimiento. Afirma más velocidad y más precisión | https://github.com/AshenOne1005/AMR |
| **SLiM** (KAIST; arXiv 2603.10648) [V✓] | Predicción de features sin decoder. **7,89× menos cómputo de inferencia** que los MAE densos | Página kaist-viclab.github.io/SLiM_site (código sin confirmar) |
| **STARS** (WACV 2026) [V~] | Enmascarado + contraste entre vecinos. Mejor que el enmascarado puro en few-shot | No localizado |
| Benchmark de IJCV 2025 (arXiv 2406.02978) [V✓] | Revisión: los métodos de un solo paradigma generalizan peor. Proponen multi-granularidad | — |

Tabla III de MaskSem, NTU-60 (X-Sub / X-View):

| Método | 1 % de etiquetas | 10 % de etiquetas |
|---|---|---|
| MaskSem | 68,4 / 72,5 | 88,4 / 92,2 |
| S-JEPA | 67,5 / 69,1 | 88,4 / 91,5 |
| MAMP | 66,0 / 68,7 | 88,0 / 91,5 |

**[OP] Lectura y coste en la 3060**
- **Formato:** el formato NTU admite dos cuerpos (M = 2), que encajan directamente con los dos tiradores.
- **Reducción de puntos:** hay que bajar de 308 a ~25–40 kp por tirador (tronco, brazos, mano armada, piernas y pies; fuera la cara y los dedos). Todos estos métodos están pensados para unas 25 articulaciones.
- **Régimen de etiquetas:** vuestras 5.500 etiquetas equivalen al régimen del 10 % de NTU, donde los métodos ya están casi empatados (88,0–88,4).
  - No tengo verificada la ganancia frente a entrenar **desde cero** con ese mismo 10 %.
- **Duda de fondo:** el SSL sobre NTU aprende la *categoría* de la acción, mientras que "quién empezó" depende del **orden relativo entre dos personas**. Nada garantiza que lo capte. Ganancia esperada: incierta y probablemente modesta.
- **Coste:** AMR o GFP, por eficiencia. Del orden de días en la 3060, con batch de 16–32, AMP y acumulación de gradiente.

---

## 4. Modelos de vídeo-lenguaje como árbitro o anotador

### 4.1 ¿Sirven para decisiones finas?
**TemporalBench** (NeurIPS 2024; arXiv 2410.10818), métrica MBA [V]:

| Evaluado | MBA |
|---|---|
| Humano | 67,9 |
| GPT-4o | 38,5 |
| Qwen2-VL-72B | 38,3 |
| LLaVA-OneVision-72B | 33,0 |
| Gemini-1.5-Pro | 26,6 |

Los modelos fallan en frecuencia, dirección y magnitud del movimiento, y "más cuadros no ayudan".

**Otros resultados**
- **SPORTU** (arXiv 2410.08474) [V✓]:
  - en preguntas de reglas *solo texto*, GPT-4o llega al 71 %;
  - en vídeo, el mejor modelo (Claude-3.5-Sonnet) se queda en **52,6 %** en las tareas difíciles de razonamiento de reglas.
- **SportMV-Bench** (arXiv 2607.11844, julio de 2026) [V✓]:
  - en deporte, el cuello de botella de los MLLM está en **"la percepción visual fina y la selección de vista, no en el razonamiento lógico ni en el conocimiento del dominio"**;
  - su agente con herramientas de percepción mejora un 15,61 % relativo.
- **SoccerRef-Agents** (arXiv 2604.23392, abril de 2026) [V✓]: GPT-4o + RAG sobre el reglamento + varios agentes, con 600 clips de faltas. Afirman que "supera claramente" a los MLLM generalistas, pero el resumen no da números.
- **SoccerNet 2025** (arXiv 2508.19182) [V~]: según el agente, las soluciones ganadoras son visión especializada, no VLM.
- **Charades-STA**, mIoU de localización temporal:
  - Qwen2.5-VL (arXiv 2502.13923) [V]: 72B 50,9; 7B 43,6; 3B 38,8; GPT-4o 35,7.
  - Qwen3-VL (arXiv 2511.21631): ~62–65 [V~].
- **Gemini 2.5:** el blog oficial sobre vídeo no muestra ningún caso de deporte ni de arbitraje. [V] https://developers.googleblog.com/en/gemini-2-5-video-understanding/

### 4.2 Pre-etiquetado
- **Weak-to-Strong** (arXiv 2602.08057) [V✓]: Gemini 2.5 Pro con CoT + Reflection genera pseudo-etiquetas en iMiGUE (entrevistas de tenis). Pasan de menos de 0,6 a más de 0,69. Ojo: la tarea es de emoción, no de reglas.
- **LEViL** (arXiv 2606.21358) [V~]: destilación genérica de pseudo-etiquetas de VLM sobre UCF, HMDB y Kinetics.
- **No encontré** ningún pre-etiquetado con VLM de prioridad, ni de eventos de reglamento, a escala y validado con humanos. [V] (búsqueda negativa)

### 4.3 Coste y uso
**Coste en local** [V]/[OP]
- Caben sin cuantizar: Qwen2.5-VL-3B, Qwen3-VL-2B/4B, InternVL3-1B/2B.
- Los modelos de 7–9B caben cuantizados a 4 bits.
- Los de 30B o más, y Gemini, GPT o Claude, solo por API.

**[OP] Uso razonable**
1. Capa de **explicación** sobre una decisión que ya ha tomado vuestro modelo (el patrón de FERA, X-VARS y SoccerRef).
2. Etiquetas **gruesas** de la frase de armas que hoy no tenéis: "hay parada-respuesta", "hay batida", "ataque simple / compuesto". Validarlas antes contra una muestra humana.
3. **No** usarlos para decidir `initiated`.

---

## 5. Señales "out of the box"

### 5.1 Cinemáticas (salen de lo que ya extraéis)
| Señal | Evidencia | Cómo sacarla de lo que ya tenemos | Coste |
|---|---|---|---|
| **Instante de inicio por umbral de velocidad** | Mulloy, Mullineaux e Irwin 2015 (ISBS) [V]: el inicio es cuando la velocidad horizontal de la espada supera 0,2 m/s. Velocidad de la espada: 3,21 ± 0,22 m/s en expertos frente a 2,63 ± 0,29 en noveles. Reacción ~0,21 s; movimiento ~0,70 s | Velocidad de la muñeca o mano armada (Savitzky-Golay), en m/s gracias a la homografía de pista. Primer cuadro que supera el umbral, por tirador. Feature: **Δt de inicio (izq − der)** | Trivial |
| **Forma de la curva de velocidad (uno o dos picos)** | Zhang, Chu y Hong (ISBS) [V~]: ataques de un pico ~0,3 s frente a dos picos ~0,5 s. El "pico falso" delata la preparación | Número de picos en la velocidad de la mano y del centro de masas, entre el inicio y la luz | Trivial |
| **Quién cierra la distancia** | La biomecánica describe el fondo como "la forma más rápida de cerrar la distancia" (Zhang et al.). Chang y Lee 2019 usan la "distancia crítica" [V~] | Distancia d(t) entre centros de masas, en metros de pista. Repartir el cierre −d′(t) por tirador proyectando la velocidad de cada uno sobre el eje de la pista. Feature: fracción del cierre que aporta cada uno entre 1 s antes de la luz y la luz | Trivial |
| **Orden brazo / pie** | Corrêa et al. 2015 (ISBS) [V~]: en expertos con blanco, el brazo armado se mueve antes que el pie delantero. La definición FIE de ataque se basa en extender el brazo amenazando | Ya tenéis los eventos "brazo" y "fondo/marcha". Feature por tirador: ¿extiende antes o después de arrancar con el pie? | Trivial |
| **Separar el ritmo de pasos de la pre-acción** | Kim et al. 2025, *Sensors* 25(13):4134 (kárate, con IMU) [V]: GWA-GLID logra 87,5 % de accuracy y F1 88,2 %, frente a 66,7 % del método convencional | Modelar el ritmo de marcha con la autocorrelación del centro de masas; lo que queda sin periodicidad es el inicio "real" | Bajo |
| **Tempo / cadencia** | Chang y Lee 2019 incluyen la cadencia de pasos en su función criterio [V~]. FERA codifica derivadas, no periodicidad [V] | Intervalos entre eventos de pie y cambio de cadencia justo antes del inicio | Trivial |
| **Orientación del tronco** (en vez de cabeza o mirada) | Incognito et al. 2024, *Front. Sports Act. Living* 6:1468209 [V]: ocultar la cabeza **no** empeoró la predicción del objetivo de la patada (68 % frente a 66 % en expertos) | Ángulo de los ejes de hombros y caderas respecto a la línea entre tiradores, y la torsión entre ambos. **No** intentar estimar la mirada (llevan careta) | Trivial |

### 5.2 Otras señales
| Señal | Evidencia | Cómo obtenerla | Coste |
|---|---|---|---|
| **Gesto del árbitro** | No hay dataset de esgrima. En fútbol: YOLOv8s + GAM, mAP@0,5 de 89,9 % con 1.200 imágenes y 6 gestos (*Front. Comput. Neurosci.* 2024) [V]. En RoboCup: gesto + silbato (arXiv 2411.17347) [V] | [OP] En la retransmisión FIE el árbitro suele estar fuera de plano. Poco retorno | Medio |
| **Audio del contacto de hojas** | Allez Go (pose + audio) [NV] | Solo en el ~10 % de clips con audio. El onset acústico de una batida o parada antes de la luz indica quién tocó hoja primero. Sirve también para validar el extractor de hoja | Bajo, pero cobertura parcial |
| **Etiqueta de decisión "gratis"** | [OP] FERA detecta las luces y el marcador en el overlay [V]. Si `touch` es "a quién se dio el punto", en los clips con **dos luces** ya es la decisión de prioridad del árbitro | Cruzar `touch` con las luces. Si no se tiene, leer el marcador con OCR tras el tocado | Bajo. **Aclarar qué es exactamente `touch`** |

**[OP] Matiz sobre la última fila**
- `initiated` y "a quién se dio el punto" no son lo mismo: en una parada-respuesta, el que inicia pierde.
- Aun así, la decisión del árbitro con dos luces es **la variable objetivo real del arbitraje**. Podría estar disponible en decenas de miles de clips, para usarla como multitarea o como preentrenamiento supervisado.

---

## 6. RECOMENDACIÓN (ordenada)

### 6.1 Protocolo común, a declarar antes de correr nada [OP]
- **Folds:** 5, **fijos y disjuntos por combate** (idealmente por tirador), sobre los ~5.500 clips. Se congelan en un fichero.
- **Test principal:** el subconjunto con **acuerdo humano**, si hay doble anotación. Se reporta también el total.
- **Métricas:** accuracy y macro-F1 en 3 clases. Además, κ de Cohen modelo-humano, calculado **con la misma métrica que dio el 0,688**.
- **Referencias:** el GRU actual (0,615) en los mismos folds. Como referencia externa, FERA v5 (0,624), aunque medida en otro conjunto.
- **Significación:** bootstrap pareado por clip (o McNemar).
- **Criterio de éxito:** Δacc ≥ +0,02 con IC 95 % por encima de 0, salvo que se declare otro umbral.
- **Desglose:** por tipo de frase cuando se pueda (una luz, dos luces, "ninguno").

### 6.2 Lo más prometedor, en orden
1. **Tokens de "inicio" explícitos + clasificador pequeño, al estilo de FERA v5.** Coste: horas.
   - **Por tirador**, las señales de §5.1:
     - instante de inicio por umbral de velocidad de la mano y primer pico de aceleración del centro de masas;
     - número de picos de la curva de velocidad;
     - aporte al cierre de distancia;
     - orden brazo/pie, cadencia previa y torsión del tronco.
   - **Entre tiradores:** las **diferencias izq − der** y la distancia en el momento del inicio.
   - **Modelos:** RF/GBM sobre esas features, y GRU + features.
   - **Criterio:** el del protocolo. Además, cada familia de features debe superar una ablación: quitarla debe costar al menos 0,01.
   - **Por qué primero:** FERA v5 muestra que, con este volumen de datos, lo que funciona es un clasificador estructurado sobre tokens. No hace falta nada nuevo.
2. **Detector de inicio por tirador, tipo spotting** (AdaSpot o T-DEED adaptado a pose). Coste: días.
   - **Modelo:** dos cabezas por cuadro (inicio izq, inicio der).
   - **Supervisión:** el cuadro de luces, los eventos de brazo y pie como pseudo-etiquetas, y `initiated` como pérdida de orden.
   - **Medición:** 2 personas anotan a mano el **cuadro de inicio** en ~200 clips; se declara su desacuerdo en cuadros.
   - **Criterio de localización:** error absoluto mediano ≤ 2 cuadros a 25 fps, y mejor que el umbral simple del punto 1.
   - **Criterio de decisión:** Δacc según el protocolo, frente al punto 1.
   - **Licencias:** AdaSpot es MIT. **T-DEED es GPL-3.0**: mejor no copiar su código.
3. **Preentrenamiento auto-supervisado de esqueleto** (AMR o MAMP/MaskSem) sobre los ~104k clips sin etiqueta, y después fine-tuning. Coste: días de GPU.
   - **Formato:** NTU con 2 cuerpos y ~25–40 kp por tirador.
   - **Medición:** curva de etiquetas al 10, 25, 50 y 100 % de las 5.500, comparando:
     - (a) el mismo encoder entrenado desde cero;
     - (b) el preentrenado;
     - (c) el GRU.
   - **Criterio:** éxito del protocolo al 100 %, **y además** que el preentrenado con el 50 % iguale o supere al entrenado desde cero con el 100 %.
   - Si no se cumple a la primera, no insistir.
4. **Uno o dos puntos de la hoja con TAPNext++ (Apache-2.0)**, anclados a la mano de Sapiens 2 e inicializados con Grounded-SAM-2 + SAHI. Coste: 1–2 días de prueba.
   - **Primero, viabilidad:** anotar a mano la punta en ~300 cuadros de ~30 clips, incluyendo fases de ataque.
   - **Criterio:** PCK@10 px ≥ 70 %, más una tasa de re-detección tras oclusión declarada de antemano.
   - **Solo si pasa:** añadir features de velocidad de la punta, distancia punta-blanco y **contacto de hojas** (batida o parada, la parte de la frase que hoy no veis). Medirlas según el protocolo.
   - **Evitar CoTracker3** en un producto (licencia NC).
5. **Etiquetas débiles a escala.** Es barato y quizá lo que más rinde por euro.
   - **(a) Decisión del árbitro:**
     - aclarar si `touch` + luces ya da la decisión de prioridad en los clips con dos luces;
     - si no, sacarla del marcador con OCR;
     - usarla como multitarea o preentrenamiento supervisado.
   - **(b) Piloto de VLM**, pequeño en local (Qwen3-VL-4B o InternVL3-2B) o por API.
     - Solo para etiquetas gruesas de frase, sobre 300 clips estratificados.
     - **Criterio:** se acepta como pre-etiquetador solo si, en el tercio con más confianza, acierta al menos el 80 % frente a la etiqueta humana. Si no, se descarta.

### 6.3 Qué descartaría (por ahora)
- **Un VLM decidiendo `initiated`:** no hay evidencia de que capten órdenes de menos de 200 ms (TemporalBench, SPORTU, SportMV-Bench).
- **Sustituir el GRU por un GCN o transformer SOTA entrenado desde cero:** ganancia esperada de 0–2 puntos según FERA v4 y v5. Solo como encoder del punto 3.
- **Mamba/SSM para esqueletos:** código inmaduro y beneficio no demostrado.
- **SAM 3 para segmentar la hoja entera como primer paso:** 848 M parámetros, licencia propia y sin validación en objetos finos.
- **AllTracker a alta resolución:** necesita 40 GB.
- **CoTracker3, si el sistema va a ser comercial** (CC-BY-NC).
- **Mirada o pose de la cabeza:** llevan careta, y la evidencia va en contra (Incognito 2024).
- **Gesto del árbitro:** no hay dataset y suele estar fuera de plano.
- **FFD, FD-7 o FenceNet como datos de entrenamiento:** solo cubren juego de piernas.
- **Perseguir el 77,7 % de FERA v4:** los propios autores lo retiraron.

---

## Fuentes principales
- **FERA:** https://arxiv.org/abs/2509.18527 · v5: https://arxiv.org/html/2509.18527v5 · v4: https://arxiv.org/html/2509.18527v4
- **Allez Go:** https://inventureprize.gatech.edu/team/allez-go · https://jsr.org/hs/index.php/path/article/view/3394 (caído)
- **FACTS:** https://arxiv.org/abs/2412.16454 · https://anonymous.4open.science/r/FACTS-B1C5
- **VirtualFencer:** https://arxiv.org/abs/2507.00261
- **FD-7:** https://arxiv.org/abs/2404.19383
- **FenceNet / FFD:** https://arxiv.org/abs/2204.09434 · https://home.agh.edu.pl/~fmal/ffd
- **FST.ai:** https://arxiv.org/abs/2507.14657
- **Malawski 2024:** https://www.scitepress.org/Papers/2024/125776/125776.pdf
- **RacketVision:** https://arxiv.org/abs/2511.17045
- **Segmentación:**
  - SAM 2: https://github.com/facebookresearch/sam2
  - SAM 3: https://github.com/facebookresearch/sam3 · https://ai.meta.com/blog/segment-anything-model-3/
  - Grounded-SAM-2: https://github.com/IDEA-Research/Grounded-SAM-2
  - LASER: https://github.com/video-fm/LASER
- **Seguimiento de puntos:**
  - CoTracker3: https://arxiv.org/abs/2410.11831 · https://github.com/facebookresearch/co-tracker
  - TAPNext: https://arxiv.org/abs/2504.05579
  - TAPNext++: https://arxiv.org/abs/2604.10582 · https://tap-next-plus-plus.github.io
  - Repo tapnet: https://github.com/google-deepmind/tapnet
  - AllTracker: https://arxiv.org/abs/2506.07310 · https://github.com/aharley/alltracker
- **Spotting:**
  - E2E-Spot: https://github.com/jhong93/spot · https://arxiv.org/abs/2207.10213
  - T-DEED: https://arxiv.org/abs/2404.05392 · https://github.com/arturxe2/T-DEED
  - AdaSpot: https://arxiv.org/abs/2602.22073 · https://github.com/arturxe2/AdaSpot
  - Santra et al.: https://arxiv.org/abs/2503.00147
  - De esqueletos a píxeles: https://arxiv.org/abs/2604.22839
  - Survey de Deakin: https://arxiv.org/abs/2505.03991
  - SoccerNet 2024: https://arxiv.org/abs/2409.10587
  - SoccerNet 2025: https://arxiv.org/abs/2508.19182
- **Esqueletos (arquitecturas):**
  - SkateFormer: https://arxiv.org/abs/2403.09508 · https://github.com/KAIST-VICLab/SkateFormer
  - ProtoGCN: https://arxiv.org/abs/2411.18941 · https://github.com/firework8/ProtoGCN
  - BlockGCN: https://arxiv.org/abs/2305.11468 · https://github.com/ZhouYuxuanYX/BlockGCN
  - Simba: https://arxiv.org/abs/2404.07645 · SkelMamba: https://arxiv.org/abs/2411.19544 · TSkel-Mamba: https://arxiv.org/abs/2512.11503
- **Esqueletos (auto-supervisado):**
  - MAMP: https://github.com/maoyunyao/MAMP
  - S-JEPA: https://sjepa.github.io
  - MaskSem: https://arxiv.org/abs/2508.12948
  - GFP: https://arxiv.org/abs/2509.03609
  - AMR: https://arxiv.org/abs/2606.11450 · https://github.com/AshenOne1005/AMR
  - SLiM: https://arxiv.org/abs/2603.10648
  - Benchmark IJCV: https://arxiv.org/abs/2406.02978
  - SkeletonX: https://arxiv.org/abs/2504.11749
- **VLM:**
  - TemporalBench: https://arxiv.org/abs/2410.10818
  - SPORTU: https://arxiv.org/abs/2410.08474
  - SportMV-Bench: https://arxiv.org/abs/2607.11844
  - SoccerRef-Agents: https://arxiv.org/abs/2604.23392
  - Qwen2.5-VL: https://arxiv.org/abs/2502.13923 · Qwen3-VL: https://arxiv.org/abs/2511.21631 · InternVL3: https://arxiv.org/abs/2504.10479
  - Weak-to-Strong: https://arxiv.org/abs/2602.08057
  - LEViL: https://arxiv.org/abs/2606.21358
  - AnyGroundBench: https://arxiv.org/abs/2607.02269
- **Señales:**
  - Chang & Lee 2019: DOI 10.1109/TCSVT.2018.2855098
  - Kim et al. 2025: DOI 10.3390/s25134134
  - Incognito et al. 2024: Front. Sports Act. Living 6:1468209
  - Gestos de árbitro de fútbol: DOI 10.3389/fncom.2024.1341234
  - RoboCup: https://arxiv.org/abs/2411.17347
  - Biomecánica ISBS (Mulloy et al. 2015; Corrêa et al. 2015; Zhang, Chu & Hong): actas ISBS
  - Bottoms et al. 2013: DOI 10.5277/abb130414
- **V-JEPA 2.1 (OmniRAS):** https://arxiv.org/abs/2608.31048

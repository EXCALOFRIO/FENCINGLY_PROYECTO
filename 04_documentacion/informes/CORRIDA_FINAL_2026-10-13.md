# La corrida final de florete: el flujo, preparado y SIN lanzar (2026-10-13)

La corrida **NO se lanza** hasta que el usuario diga que se acabaron las pruebas. Aqui esta todo lo
que hara, en que orden, cuanto cuesta y que tiene que decidir el usuario antes (seccion 6). Cada
pieza que entra tiene su cifra en su informe (`ESTADO_2026-10-10.md` §1). Lo que no paso va
apagado, con su numero (§2 del mismo).

## 1. Que sale de cada clip (106.432 clips de florete)

| etapa | donde | que | estado |
|---|---|---|---|
| 1 | GPU | YOLO11m-pose en cada cuadro, la pareja de tiradores, planos utiles | entra |
| 4 | GPU | Sapiens 2, 308 puntos por tirador y cuadro | entra |
| 7 | CPU | marcha, retroceso, fondo, brazo, mano armada, posicion; puerta fisica por evento | entra |
| 9 | CPU | las luces del tocado y su cuadro (`fencingly_v2` primero) | entra |
| 11 | CPU | la x del tocado en metros y su zona (rayas pintadas + movimiento de camara) | entra |
| 11 + | CPU | **la homografia de cada cuadro** (`FENCINGLY_GUARDA_H`, `pista_h_<clip>.npz`) | entra (decidido 13/10; encendida por defecto) |
| V-JEPA 2 | GPU | resumen de video para quien inicia y la decision del arbitro | entra: G2 en los clips con dos luces (decidido 13/10) |
| DINOv2 | GPU | rasgos cuadro a cuadro | **no entra**: D0 no pasa (13/10; contra V0 -0,010, contra A' +0,030 con el IC cruzando cero) |
| despues | CPU | `zona_fusion.py` por torneo (zona final), etiquetador entrenado y etiquetas | entra, al final |

Los interruptores ya estan en su valor adoptado por defecto (`FENCINGLY_LUCES_V2=1`,
`FENCINGLY_PIERNA_K=0.5`, `FENCINGLY_PUERTA_EVENTO=1`, `FENCINGLY_ZONA_FUSION=1`; los que no pasaron,
a 0). Los videos se quedan en `D:\FENCINGLY_VIDEOS_RAW\_descargados` (`--borra-video` solo borra la
copia de `pruebas/videos`), asi que todo lo que sale del video se puede rehacer despues sin bajarlo
otra vez.

## 2. El orden propuesto: los clips con dos luces PRIMERO

La pregunta que mas importa del proyecto es **la decision del arbitro** (`prioridad`). Con los 735
clips de entrenamiento de los anotados no se pudo medir (V-JEPA no se distingue del etiquetador). La
medida buena necesita los ~21.000 clips de florete con las dos luces, y en el orden de siempre
(barajado) estarian completos al final, a las ~7 semanas. La etapa 9 no depende de ninguna otra, asi
que se pasa primero sobre todo el florete. Con eso sale la lista de clips con dos luces, que van
delante. **La medida llega hacia los ~14-15 dias (fases `luces` a `sonda_g2`), no a las ~7 semanas, y sin
calcular nada de mas.**

## 3. Las fases: un solo comando, `corrida_florete.py` (preparado y probado; SIN lanzar)

    cd 01_extraccion
    python corrida_florete.py --seco      que haria y cuanto hay hecho (no lanza nada)
    python corrida_florete.py --lanza     EL DIA QUE EL USUARIO DIGA "LANZA": arranca el director oculto
    python corrida_florete.py --estado    cuanto lleva cada fase
    python corrida_florete.py --para      lo para todo; relanzar con --lanza sigue donde estaba

El director corre las fases en orden, **espera a que la GPU este libre antes de cada fase de GPU**
(ni una sonda ni un extractor ni otra noche corriendo), y si una fase falla escribe `FALLA_<fase>` y
se para. Marcas y registro en `01_extraccion/pruebas/corrida_florete/`; salidas de las sondas en
`02_validacion/auditoria/corrida_florete/`. Todo sin ventanas y reanudable.

| fase | que | GPU | dias (estimados) |
|---|---|---|---:|
| `luces` | etapa 9 sobre los 106.432 de florete, bajando los ~78.000 mp4 que faltan (~180 GB) | no | ~1-1,5 |
| `lista` | `03_datos/catalogo/foil_dos_luces.txt` (`lista_dos_luces.py`) | no | minutos |
| `dos_luces` | etapas 1 y 4 (GPU) y 7, 9 y 11 (CPU) sobre los ~21.000 de la lista, a 96 clips/h | si | ~9 |
| `vjepa_g2` | ViT-g 256 `rec4,mic4` sobre la lista -> `D:/FENCINGLY_VJEPA/g2` (~5,8 MB/clip) | si | ~1,6 |
| `vjepa_v0` | ViT-L 256 `rec4` (= V0) sobre la lista -> `D:/FENCINGLY_VJEPA/v0`, para comparar | si | ~0,3 |
| `sonda_v0`, `sonda_g2` | `prioridad` con ~21.000 decisiones, y G2 contra V0 (protocolo en `VJEPA2_2026-10-09.md`, "Pregunta 2 EN LA CORRIDA", declarado el 13/10) | si | ~2 |
| `resto` | etapas 1, 4, 7, 9 y 11 sobre los ~85.000 restantes | si | ~37 |
| `zona_fusion` | la zona final por torneo, con el florete entero | no | horas |
| (3B) | V-JEPA sobre el resto, G2 (~6 dias) o V0 (~1 dia): **la decide el usuario** con `compara_prioridad_G2_vs_V0_corrida.txt` | si | 1-6 |

**Prueba del director (13/10 19:00-19:12)**: `--director --prueba` sobre 3 clips anotados ya
procesados, con marcas en el scratch y sin respaldo. Hizo `luces`, `lista` y `dos_luces`: lanzo y
espero a los trabajadores (todos dijeron "TERMINADO: ... estan hechos"), los paro, y no toco ningun
fichero de esos clips (fechas iguales). `--seco` sobre el florete: 106.432 clips, 28.511 con mp4 en
disco, 4.251 con luces, D: 1.514 GB libres, C: 330 GB.

## 4. Cuanto tarda, todo junto

| fase | dias | GPU |
|---|---:|---|
| 1 luces | ~1,5 | libre |
| 2A dos luces | ~9 | etapas 1 + 4 |
| 3A V-JEPA G2 y V0 + sondas `prioridad` | ~4 | V-JEPA y sondas |
| 2B resto | ~37 | etapas 1 + 4 |
| 3B V-JEPA resto | ~1-6 | V-JEPA |
| **total** | **~53-58 (7,5-8 semanas)** | |

Si DINOv2 pasa su sonda: +15-28 dias. Disco: C: ~150 GB (poses), con 350 libres; D: ~180 GB de
videos nuevos + 120-610 GB de V-JEPA, con 1.410 libres.

## 5. Lo preparado hoy (sin cambiar ninguna cifra)

- `vjepa_variantes.py`: salida `mic4` (el micro a 4x4, lo que midio G2; la cuarta parte de disco
  que `mic8`). Se comprueba que `mic4` sea el 2x2 de `mic8` en la medicion de coste de G2 (200
  clips, tras la sonda D0).
- `zona_tocado.py`: `FENCINGLY_GUARDA_H` (encendida por defecto desde la decision del 13/10). Probado en 6 anotados sin escribir en
  `pruebas`: los campos medidos de `zona_` son identicos (la unica diferencia, `via_h`, es que esos
  CSV son de antes de existir la columna). 6-15 KB por clip, ~1,5 GB en florete, 0 s de mas. En los
  4 clips con respuesta hay homografia confiable en 225-255 cuadros de cada clip hasta el tocado.
- `lista_dos_luces.py`: la regla de `sonda_vjepa.filas_prioridad`. En los anotados: 1.119 de 5.856.

## 6. Lo decidido por el usuario (13/10 ~18:00) y lo que falta

| que | decision |
|---|---|
| orden (§2) | **las dos luces primero** |
| V-JEPA | **G2 (ViT-g, `rec4 + mic4`) en los clips con dos luces**; el resto, G2 o V0 con la `prioridad` medida en la fase 3A |
| homografia de cada cuadro | **si**: `FENCINGLY_GUARDA_H` encendida por defecto en `zona_tocado.py` |
| G4 (ViT-g 384, ojo meso) | **no se corre** (`VJEPA2_VARIANTES` cambio 4) |

**Falta**: la sonda de DINOv2 (D0, mie 14/10 de madrugada) y la medida del coste de G2 a solas (200
clips, justo despues). Si D0 pasa, el usuario decide si DINOv2 entra (+15-28 dias). **Y la orden
explicita de lanzar**: la corrida no arranca sin ella.

## 7. Con otro hardware: 2 DGX Spark (pregunta del usuario, 13/10; ESTIMADO, no medido)

Lo publicado: una RTX 4090 es **3,3-4,2 veces** mas rapida que un DGX Spark (GB10) en entrenamiento
e inferencia, 3,6 en deteccion de objetos (RT-DETRv2) (`mrdbourke/gpu-benchmarking`). Y una 4090 es
~3,5-5 veces una RTX 3060. Asi que **un Spark es ~1-1,5 veces esta 3060**. Tiene mas nucleos de
tensor (~100 TFLOPS BF16 medidos), pero menos ancho de banda de memoria (273 GB/s, contra 360 de la
3060), y lo nuestro son modelos que caben en 12 GB: sus 128 GB no ayudan. Dos Spark: **~2-3 veces**.

| que | esta 3060 | 2 Spark | 2 Spark + esta 3060 |
|---|---:|---:|---:|
| florete, el plan de arriba | ~53-58 dias | **~18-29 dias** | ~14-20 dias |
| las tres armas + campeonatos (362.950 clips), con V0 | ~165 dias | ~55-82 dias | ~41-55 dias |
| lo mismo, con G2 en todo | ~185 dias | ~62-92 dias | ~46-62 dias |

Lo que cambia esas cifras: el Spark es Linux sobre ARM, y el pipeline corre en Windows. Hay que
portarlo (las rutas ya van por variables de entorno, `rutas.py`) y comprobar en ~50 clips anotados
que sale lo mismo antes de fiarse. Y el lote grande que permitiria su memoria ya se rechazo en las
poses (movia un punto 130 px). El factor de verdad solo se sabe midiendo esos 50 clips en un Spark.
Para rendimiento puro, **una sola RTX 4090 da mas que dos Spark** (3,3-4,2 veces cada uno).

## 8. Alquilando GPU (pregunta del usuario, 13/10; ESTIMADO con precios publicos de sep-oct 2026)

**Horas de trabajo, en horas de esta 3060**: florete con el plan de arriba ~1.230-1.350 h (etapas
1 + 4: 106.432 x 37,5 s = 1.109 h; V-JEPA y sondas, el resto). Todo el corpus (362.950 clips):
~3.900 h con V0, ~4.430 h con G2 en todo. El trabajo se reparte por clips, asi que con N GPU se
tarda N veces menos y cuesta casi lo mismo.

| GPU (RunPod Secure Cloud, centro de datos) | $/h | velocidad frente a la 3060 (estimada) | florete | todo el corpus (V0 / G2) |
|---|---:|---:|---:|---:|
| **RTX 5090** (recomendada) | 0,99 | ~4-6x | **~$200-340** | ~$650-970 / ~$730-1.100 |
| RTX 4090 | 0,74 | ~3-4,5x | ~$200-330 | ~$640-960 / ~$730-1.090 |
| H100 SXM | 3,49 | ~5-8x | ~$540-940 | no compensa |

- Almacenamiento: $0,07/GB/mes (el primer TB); sin coste de transferencia. Florete ~0,5-1 TB unos
  dias: ~$10-20.
- Con 4 RTX 5090, el florete son ~2-3,5 dias; con 8, todo el corpus son ~4-6 dias.
- La 5090 frente a la 4090: ~1,5 veces mas rapida y cuesta ~1,35 veces mas. La H100 cuesta 3,5 veces
  mas para ~1,3 veces la 5090: lo suyo son los lotes grandes y los modelos de 80 GB, y aqui no los
  hay (el lote de las poses se rechazo).
- Hosts de particulares (Vast.ai, RunPod Community): ~la mitad de precio, pero los videos y las
  poses de atletas identificables quedan en la maquina de un desconocido. No recomendado.
- Antes de gastar nada: **una prueba de ~$5** (1 GPU, ~3 h) con 50 clips anotados, comparando las
  salidas con las nuestras (criterio escrito antes) y midiendo los clips/hora reales. Con eso, el
  precio de verdad. Lo que pide: Linux x86 (el camino de Colab ya existe), >=16 vCPU por GPU (las
  etapas 7, 9 y 11 son ~30 s de CPU por clip) y el token de HF en la maquina, para el respaldo.

Precios: RunPod (Secure Cloud, sep 2026), RunPod docs (almacenamiento y transferencia). Velocidades:
`mrdbourke/gpu-benchmarking` (4090 frente a GB10) y comparativas publicas 5090/4090/H100. **Ninguna
medida en nuestro pipeline.**

## 9. TODO con esta 3060 (decision del usuario, 13/10: no se alquila nada; ESTIMADO)

Ritmo medido de las etapas 1 + 4: 96 clips/h. Las de CPU (7, 9, 11) van a la vez y no suman. Lo ya
hecho (los anotados) se descuenta.

| arma | clips | ya hechos | etapas 1 + 4 | V-JEPA (V0 / G2) | total |
|---|---:|---:|---:|---:|---:|
| florete (el plan de arriba) | 106.432 | 4.270 | ~44 dias | ~5-10 dias (con las sondas) | **~51-56 dias** |
| espada | 140.639 | 1.037 | ~61 dias | ~2 / ~10 dias | ~63-71 dias |
| sable | 102.618 | 566 | ~44 dias | ~1,4 / ~7,6 dias | ~46-52 dias |
| campeonatos 2025-26 (`champs`) | 13.260 | 0 | ~6 dias | ~0,2 / ~1 dia | ~6-7 dias |
| **todo** | **362.950** | 5.873 | **~155 dias** | ~9-29 dias | **~166-186 dias (5,5-6 meses)** |

Si se lanza el jue 15/10: florete hacia el 5-10/12/2026; todo, entre finales de marzo y mediados de
abril de 2027. Si DINOv2 pasa su sonda y entra: +50-95 dias en todo.

**El disco no da para todo** (si da para el florete): faltan ~730 GB de mp4, ~490 GB de poses (en
C:, que tiene 330 libres) y 0,5-1,5 TB de V-JEPA, contra 1,84 TB libres entre C: y D:. Hace falta
un disco mas (~4 TB) o borrar cada mp4 al procesarlo (se pierde poder rehacer sin volver a bajarlo).
Se decide antes de acabar el florete, no ahora.

Luz: el PC a ~0,3 kW (estimado) 24 h al dia son ~380 kWh en el florete y ~1.300 kWh en todo, unos
60-95 EUR y 190-320 EUR a 0,15-0,25 EUR/kWh.

## 10. Acelerar la corrida (13/10, lo pidio el usuario). Criterios DECLARADOS antes de medir nada

**Campeonatos**: los 3.367 clips de florete de `champs` entran (`lista_campeonatos_florete.py`); el
director corre sobre `foil_y_campeonatos.txt` (lo escribe al lanzar). En la sonda de `prioridad`
entran como `champs` por arma, solo con la lista explicita.

**La logica del usuario: lo que solo sirve al videoarbitraje, solo en los clips con dos luces; lo de
scouting, en todos.**

| pieza | sirve para | en que clips |
|---|---|---|
| luces (etapa 9) | todo: quien toco y que clips tienen dos luces | todos |
| YOLO y pareja (etapa 1) | todo | todos |
| Sapiens 2, 308 puntos (etapa 4) | arbitraje: manos, brazo fino, el zoom de G2 | **solo dos luces** |
| eventos y posicion (etapa 7) | scouting | todos; en los de una luz, con los 17 puntos de YOLO (O5, a medir) |
| zona del tocado y homografia por cuadro (etapa 11) | scouting | todos (igual que la 7) |
| V-JEPA G2/V0 y sondas de `prioridad` | arbitraje | **solo dos luces** |
| zona final por torneo | scouting | todos |

Las candidatas, en orden. Ninguna entra sin pasar su criterio:

- **O1, perfil (solo mide)**: en que se van los ~42 s de las etapas 1 y 4 (lectura, YOLO,
  redeteccion, preproceso de Sapiens en CPU, pasada en GPU, postproceso, uso de la GPU), 8 clips
  anotados con la GPU a solas. `D:/FENCINGLY_SCRATCH/perfil/perfil_etapas.py`, ya en cola: corre
  solo al acabar D0 y la medida del coste de G2.
- **O2, mismo resultado**: etapa 4 en tuberia (leer cuadros y preprocesar en hilos de CPU mientras
  la GPU calcula). Pasa si las poses son IDENTICAS (diferencia maxima 0,0 px en todos los puntos) en
  20 clips y va >= 1,2 veces mas rapida.
- **O3, mismo resultado**: etapa 1 en tuberia (lectura en un hilo). Pasa si `det_` y `revision_` son
  identicos en 20 clips y va >= 1,1 veces mas rapida.
- **O4, mismo resultado**: hilos de CPU repartidos entre trabajadores y mas trabajadores para 7, 9 y
  11. Con la GPU mas rapida, esas etapas (~28 s de CPU por clip) pasan a ser el cuello. Pasa con
  salidas identicas.
- **O5, cambia la entrada del scouting**: en los clips de una luz, etapas 7 y 11 con los 17 puntos
  de YOLO (ya guardados en `detx_`), sin Sapiens. Se mide en los anotados con una luz, con el
  reparto de siempre. Pasa si:
  - la zona contra `strip_location` no baja mas de 0,01, con el limite inferior del IC95 por
    torneo de la diferencia >= -0,02;
  - la x del tocado difiere <= 10 cm en mediana donde las dos tienen homografia;
  - la mano armada coincide >= 95 %;
  - >= 90 % de los eventos de marcha, fondo y brazo coinciden a +-3 cuadros con los de Sapiens.

  Si no pasa, los de una luz llevan Sapiens como ahora.

  **RESULTADO (13/10 21:52): NO PASA**, ninguno de los cuatro. 396 anotados con `detx_` (los
  procesados desde el 08/10), en `D:/FENCINGLY_SCRATCH/o5`; cifras en
  `02_validacion/auditoria/corrida_florete/o5_yolo17.json`:

  | criterio | pide | sale |
  |---|---|---|
  | zona total (no contesta = `middle`) | >= -0,01, IC >= -0,02 | 0,475 contra 0,492: -0,018 [-0,046, +0,019] |
  | x del tocado, mediana de la diferencia | <= 10 cm | 18 cm (p90 2,9 m), n=203 |
  | mano armada | >= 95 % | 93,8 % (n=695) |
  | eventos a +-3 cuadros | >= 90 % | 52 % de los de Sapiens, 51 % de los de YOLO |

  Sin los pies y las manos de Sapiens, la mitad de las marchas, fondos y brazos cambian. **Sapiens se
  queda en todos los clips.** La division del usuario sigue valiendo para V-JEPA y las sondas (solo
  dos luces), y por eso la fase 3B desaparece.
- **O6 (despues, si hace falta)**: en los de dos luces, Sapiens solo en la ventana [tocado - 6 s,
  tocado + 1 s] (68 % de los cuadros utiles, medido en 5.659 anotados). Necesita su propio criterio
  sobre el etiquetador.
- **O7 (necesita instalar)**: TensorRT u ONNX para YOLO. No se instala nada en
  `D:\FENCINGLY_PY\venv`; haria falta otro entorno y el permiso del usuario.

~~Estimacion si O2, O3 y O5 pasan~~ (13/10, sin medir): **ya no vale**, O5 no paso. La vigente va en
"Resultados" abajo.

### Resultados (13-14/10), cada uno con su criterio de arriba

Cifras en `02_validacion/auditoria/corrida_florete/` (`o2_o3_identidad.json`, `o8_imgsz.json`,
`o5_yolo17.json`). Pruebas en `D:/FENCINGLY_SCRATCH/perfil`, cada pasada en su proceso y su directorio,
con la GPU a solas.

**Exactas de CPU, ADOPTADAS** (13/10, commit `13505b7`; identicas bit a bit en las salidas):
- mediana de la placa (`pista._nanmediana0` por `np.nanmedian`): 6,8x; etapa 7 ~8 s menos por clip,
  identica en 6 clips;
- fusion de lineas (`pista_lineas._merge_lines` vectorizada): 1,7x; identica en 1.898 cuadros, etapa 11
  ~2 s menos e identica en 6 clips;
- busqueda del mp4 (`rutas.video`) con el indice hecho una vez: 0 diferencias en 3.000 clips;
- escrituras atomicas (`.part` + `os.replace`) en `det_`, `detx_`, `revision_`, `poses_`, `lights_` y
  las series: un corte a mitad no deja un fichero roto que cuente como hecho.

**O1, el perfil fino: ABANDONADO.** Envolviendo `predict` y las demas llamadas, el proceso reventaba
(pila desbordada por el `deepcopy` de ultralytics, y luego un acceso a memoria). Los tiempos salen de
las pruebas de identidad, que no envuelven nada. Dejo un hallazgo: al alternar fp16 (deteccion) y fp32
(redeteccion), ultralytics reconstruia el predictor en cada cambio -> O10.

**Prueba de identidad 1** (14/10 00:04; 20 anotados de florete; pasadas A base, B tuberias + O10, C =
B + O2b + O9, A2 base otra vez):

| pasada | etapa 1 | etapa 4 | `det_`/`detx_`/`revision_` = A | `poses_` = A |
|---|---:|---:|---|---|
| A (base) | 239 s | 619 s | — | — |
| B | 139 s | 729 s | 20/20 | **0/20** |
| C | 201 s | 670 s | 20/20 | **0/20** |
| A2 (base) | 149 s | 708 s | — | 20/20 (determinista) |

- **El ruido de tiempos es grande**: A y A2 son el mismo codigo y la etapa 1 va de 239 a 149 s (la
  primera pasada pagaba algo). Con esto no se puede dar la velocidad por medida -> prueba 2, intercalada
  y con una pasada de calentamiento que no cuenta.
- **O3 (etapa 1 en tuberia) + O10: IDENTICAS** en los 20 clips. Velocidad sin decidir (contra A2,
  1,07x; contra A, 1,72x) -> prueba 2.
- **O2 (etapa 4 en tuberia), primera version: NO PASA.** Poses distintas en 20/20: en los puntos
  usables (confianza >= 0,3, 3,16 M) p50 0,011 px, p99 0,13 px, **max 936 px**, 4.866 puntos a mas de
  0,5 px. Y sin ganar tiempo (729 s contra 619-708). LA CAUSA: el codigo de siempre pasa por el modelo
  **las dos cajas del cuadro juntas** (lote 2), y la tuberia las pasaba de una en una; el lote cambia
  el nucleo de la GPU y un punto ambiguo salta (lo mismo que se midio con `LOTE_POSE`). Corregida (un
  lote por cuadro, como siempre) -> **O2'**, prueba 2.
- **C (O2b + O9)**: mismas poses que B (identicas entre si), o sea que O2b y O9 no cambian nada por su
  cuenta; heredan el fallo de B. -> prueba 2, sobre O2'.

**O8 (YOLO a menor resolucion): NO PASA** (contra A, mismos 20 clips; puerta: pareja con IoU >= 0,95 en
>= 99,5 % de los cuadros y presencia igual en >= 99,9 %):

| imgsz | IoU >= 0,95 | presencia igual | etapa 1 |
|---|---:|---:|---:|
| 960 (A2) | — | — | 149 s |
| 768 | 84,8 % | 95,4 % | 141 s |
| 640 | 80,6 % | 93,8 % | 129 s |

Cambia la pareja en 15-20 % de los cuadros y apenas gana (5-13 %): el tiempo de la etapa 1 no esta en
la red. Se queda en 960.

**Nombres con letras fuera de ASCII (14/10), DOS FALLOS REALES arreglados**: 544 ids del catalogo
(`Vicuña`, `Vendéen`, `Kauai‘Ō‘ō`), **177 de florete y 27 de campeonatos**.
1. `lotes.descarga` no los bajaba NUNCA (`urllib` con la url sin codificar: UnicodeEncodeError). Ahora
   la url va con `urllib.parse.quote`: el CDN da 200, y en las 362.406 urls ASCII no cambia nada
   (medido).
2. Con la salida a un fichero, Windows escribe en cp1252 y un `print` de `Kauai‘Ō‘ō` tumba el proceso.
   El director y `noche.py` ponen `PYTHONIOENCODING=utf-8`, que heredan todos los trabajadores; las
   series y CSV de la etapa 7 se escriben con `encoding="utf-8"` (en ASCII, los mismos bytes).
OpenCV abre esos mp4 sin problema (391 y 833 cuadros leidos). El ensayo de 100 mete 3 de esos clips.

**Prueba de identidad 2** (en cola tras el ensayo de 100; `identidad2.py`, criterios en su cabecera,
los de arriba con tiempos intercalados): etapa 1 W1 (calentamiento) A1 B1 A1 B1; etapa 4 A4 P4 Q4 A4 P4
Q4, con P = O2' y Q = O2' + O2b + O9. O9 ya solo toca la etapa 4 (en la 1 torch apenas calcula en CPU).

**Lo que decide los dias: el CAUDAL EN PARALELO** (en cola tras la prueba 2; `D:/FENCINGLY_SCRATCH/caudal/caudal.py`).
Con 3 trabajadores la GPU marca ~80 % (muestreado en el ensayo de 100) y de 1 a 3 trabajadores el caudal
solo subio 1,10x (`noche.py`): la tarjeta va casi llena de nucleos pequenios, y lo que ahorra CPU en un
proceso solo puede no notarse. 60 anotados de florete, la corrida montada como de verdad (`noche.py`,
3 GPU para 1+4 y 4 CPU para 7, 9 y 11), cada brazo desde cero y en su directorio: base1, exacta,
lote16, base2. Criterios, DECLARADOS ANTES DE MEDIR:
- **exacta** (lo exacto que haya salido identico en la prueba 2; O9 siempre): todas las salidas
  identicas a base1 en los 60 (`det_`, `detx_`, `revision_`, `poses_`, `lights_`, `zona_`, series,
  `pista_h_`), y caudal de 1+4 >= 1,03 x la media de base1 y base2 y por encima de las dos. Si pasa, se
  adopta.
- **O11, lote de 16 recortes en Sapiens** (NO exacta). El 08/10 se rechazo con la puerta de pixel (max
  < 0,5 px; dio 130 px): esa puerta no la pasa ningun cambio de aritmetica, ni otra GPU, porque un punto
  ambiguo salta. Se mide con la puerta en lo que se usa despues:
  a. poses, puntos usables: p50 < 0,02 px y p99 < 0,5 px;
  b. eventos de marcha, fondo y brazo a +-3 cuadros: >= 98 % en los dos sentidos;
  c. zona del tocado igual en >= 98 % de los clips, |dx| del tocado mediana <= 2 cm;
  d. mano armada igual en >= 99 %;
  e. caudal de 1+4 >= 1,15 x base.
  **Aunque pase, NO se adopta sin el usuario**: cambia la puerta que lo dejo fuera.
- base1 contra base2 tiene que salir identico (mismo codigo): si un tipo de fichero no lo sale, se dice
  y no juzga a `exacta`.

**Disco** (medido 14/10): ~1,5 MB por clip en `pruebas` (`poses_` 1,45): los 109.799 de florete +
campeonatos son ~166 GB en C (360 GB libres). Faltan 81.267 mp4 (~191 GB, a D, 1.510 GB libres).

### Resultados de la noche del 13 al 14/10

**Ensayo de 100** (`D:/FENCINGLY_SCRATCH/ensayo/e_1014_0011/informe.json`; el director de verdad en modo
`--ensayo`, 103 clips: 30 anotados, 50 sin procesar con mp4, 10 sin mp4, 10 de campeonatos sin mp4, 3 con
letras fuera de ASCII; 4 trabajadores de CPU; tres interrupciones con `--para`):
- **Las tres interrupciones** (en `dos_luces`, en `vjepa_g2` y en `resto`): tras cada una, 0 procesos del
  ensayo vivos y 0 `.part`; relanzado, siguio donde iba. Acabo (`FIN.txt`) con todas las fases y 0
  procesos colgados, 0 `.part` y 0 ficheros vacios.
- **Por clip**: `lights_`, `det_`/`revision_`/`poses_` legibles, `zona_` y series (o `pista_vacia_`) en
  **103 de 103**; `pista_h_` en 96,1 % (los 4 sin luz no tienen tocado que mirar); zona con respuesta en
  66 %. Los 20 sin mp4 se bajaron; **los 3 de nombre no ASCII, bajados y procesados enteros**.
- **V-JEPA: G2 en 82,2 % y V0 en 84,4 % de los 45 con dos luces. FALLO REAL, arreglado** (abajo).
- En los 30 anotados, `poses_` iguales a las de `pruebas` en 4: son los 4 con `revision_` igual; en los
  demas cambia la etapa 1 (las de `pruebas` son de codigo anterior, p. ej. la puerta de piernas del 09/10).
  Con la misma entrada, la etapa 4 da lo mismo.
- **`resto` fue 3-5x mas lento** (etapa 4 a 300-420 s/clip en sus primeros lotes, contra ~80 en
  `dos_luces`), con la CPU al 97 % y la GPU al 38 %, y dio **2 MemoryError** (6 y 12 MB; los dos clips
  salieron en la vuelta siguiente). Causa medida abajo: la memoria.

**FALLO 1, la carrera en `rutas.indice_videos`** (introducida el 13/10 con la busqueda rapida del mp4):
el indice se publicaba vacio y se llenaba despues; los hilos de `prepara` de V-JEPA que llegaban mientras
se recorria el arbol recibian `None` ("salta: sin video") para clips que SI tienen mp4: 6-8 por arranque.
Reproducido con 8 hilos: 17-18 de 24 `None`. Arreglado (indice entero y con cerrojo): 0 de 24 en tres
pasadas. Los 11 clips que faltaban se preparan bien ahora.

**FALLO 2, la memoria.** Medido por proceso (`memoria_por_etapa.json`): etapa 1 RSS hasta 4,3 GB (6,8 de
commit), etapa 4 3,4 (6,5), etapa 7 en 1080p 3,2 (4,0), la 9 y la 11 ~0,7. La maquina en reposo ya tiene
23 GB de commit (limite 63) y 22 GB disponibles. Con 3 de GPU y 4 de CPU en clips 1080p se pasa de la RAM
fisica y se roza el limite de commit. La corrida va con **2 de CPU** (`N_CPU`, el ensayo forzaba 4), que
bastan: las etapas 7 + 11 son ~25-30 s por clip y trabajador, 2 hacen ~250 clips/h contra ~116 de la GPU.
Y desde hoy hay un **guardian** (`rutas.espera_ram`): cada etapa espera a tener RAM fisica libre antes de
arrancar (`lotes.RAM_ETAPA`), y la 7 lo mira antes de cada clip. No cambia ninguna salida.

**FALLO 3, hilos colgados si un clip falla a mitad** (con las tuberias): el lector de la etapa 1 no se
soltaba si `detecta_clip` lanzaba (32 cuadros retenidos por clip), y el de la tuberia de Sapiens se
quedaba bloqueado en la cola (hasta 64). Arreglado (`try/finally` y aviso de parada). Probado con fallos
provocados en las dos etapas: 0 hilos nuestros vivos despues, y el clip siguiente identico a `id_A`.

**Prueba de identidad 2** (`o2p_o3_identidad2.json`; tiempos intercalados con calentamiento):

| | identicas | tiempos (s, 20 clips) | x |
|---|---|---|---:|
| O3 (etapa 1: lectura en hilo + O10) | `det_`/`detx_`/`revision_` 20/20 en las dos pasadas | A1 226, 204 / B1 204, 152 | **1,21** |
| O2' (etapa 4 en tuberia, lote por cuadro) | `poses_` 20/20 en las dos | A4 640, 510 / P4 501, 405 | **1,27** |
| O2' + O2b + O9 | 20/20 en las dos | Q4 466, 397 | **1,33** |

Las tres **PASAN**.

**Caudal en paralelo** (`caudal_paralelo.json`; 60 anotados, 3 GPU + 4 CPU, desde cero):

| brazo | clips/h (1+4) | GPU | CPU | todas las salidas = base1 |
|---|---:|---:|---:|---|
| base1 | 113,3 | 79 % | 73 % | — |
| exacta (O3 + O10 + O2' + O2b + O9) | **120,8** | 87 % | 89 % | **60/60** en `det_`, `detx_`, `revision_`, `poses_`, `lights_`, series, `zona_` (*) y `pista_h_` |
| lote16 (O11) | 118,8 | 77 % | 85 % | no (es la prueba) |
| base2 | 119,7 | 84 % | 68 % | 60/60 (*) |

(*) `zona_` solo cambia en la columna `s` (los segundos que tardo la etapa), tambien entre base1 y base2:
sin ella, identica en 60/60.
- **exacta: PASA** (1,037x la media de base1 y base2, y por encima de las dos), pero la ganancia en
  paralelo es pequenia: con 3 trabajadores la GPU ya va al 79-87 %, y lo que se ahorra en CPU en un
  proceso solo (1,2-1,3x) casi no se nota. **Encendida por defecto desde hoy**; `=0` la apaga.
- **O11 (lote 16 en Sapiens): NO PASA**: p50 0,008 px y p99 0,07 px (pasa a), eventos 99,3/99,1 % (pasa
  b), mano 100 % (pasa d), pero la zona solo igual en 95 % de los clips (pide 98 %) y caudal 1,02x (pide
  1,15). No se propone.
- Arranque por lote, medido: etapa 1 3,3 s, etapa 4 10,4 s, 9 2,3 s, 7 1,2 s. El director sube el lote de
  20 a **50** clips (`FENCINGLY_DIRECTOR_LOTE`): 0,69 -> 0,27 s por clip y trabajador, mismas salidas.
- Fase `luces` con **6 trabajadores** en vez de 3 (`N_LUCES`; `luces_trabajadores.json`): 3.954 clips/h
  contra 3.085 y 3.628, **1,18x**, `lights_` identicos en 120/120. Descarga medida: 0,77 s por clip en
  serie (6,7 MB/s); los mp4 que faltan pesan ~5,2 MB de media, **~420 GB** a D (1.510 libres).

### CUANTO QUEDA, con lo medido el 14/10 (florete + campeonatos, 109.799 clips)

Hecho ya: 4.270 con etapas 1+4 (los anotados). Quedan **105.529** sin 1+4, 105.548 sin `lights_` y
81.267 sin mp4. Ritmos: 1+4 en paralelo 120,8 clips/h en anotados (exacta); en el corpus los clips son
algo mas largos (mediana de cuadros 331-423 contra 300; los usados, parecidos), asi que se toma
**105-116 clips/h**. Dos luces: 19 % en florete (1.119 de 5.856 anotados), 1,2 % en campeonatos.

| fase | clips | ritmo | dias |
|---|---:|---|---:|
| `luces` (etapa 9 + bajar 81.267 mp4), 6 trabajadores | 105.548 | ~3.500/h | **~1,3** |
| `dos_luces` (1+4, con 7, 9, 11 en paralelo) | ~19.500 | 105-116/h | **7,0-7,7** |
| `vjepa_g2` (6,89 s/clip, medido en el ensayo) | ~20.300 | | **1,6** |
| `vjepa_v0` (2,28 s/clip; informa la decision de la fase 3B) | ~20.300 | | **0,5** |
| sondas de `prioridad` | | | ~0,2 |
| `resto` (1+4, con 7, 9, 11 en paralelo) | ~86.000 | 105-116/h | **30,9-34,1** |
| **TOTAL** | | | **~41-45 dias** |

Antes de esta tanda (13/10) la cifra era **~51 dias**, con el ritmo real de las noches de anotados (96
clips/h, codigo de entonces y la maquina compartida). Hoy, con todo montado como en la corrida y la
maquina a solas: base 113-120 clips/h y exacta 121. **Si la maquina se usa para otra cosa durante la
corrida, el ritmo baja hacia el de las noches de anotados.** La parte grande es `resto` (las etapas 1+4
de los ~86.000 de una luz), y ahi la GPU ya va al 80-87 %: lo que queda por ganar en exacto es poco.

**Ensayo de confirmacion con el codigo final** (14/10 18:08-18:52; `D:/FENCINGLY_SCRATCH/ensayo/e_1014_1808`;
41 clips: 15 anotados, 15 sin procesar, 4 sin mp4, 4 de campeonatos, 3 no ASCII; **2 de CPU como en la
corrida**, lote 50, `luces` con 6, tuberias por defecto; interrumpido en `vjepa_g2` y en `resto`):
- las dos interrupciones: 0 procesos vivos, 0 `.part`; relanzado, siguio. Acabo con todas las fases, 0
  procesos colgados, 0 `.part`, 0 ficheros vacios;
- 41/41 con `lights_`, `det_`/`revision_`/`poses_` legibles, `zona_`, series y `pista_h_` (100 %);
- **V-JEPA G2 y V0 en el 100 % de los 18 con dos luces** (antes de arreglar la carrera, 82-84 %);
- **0 lineas de fallo** (ni un MemoryError); en `resto` la CPU al 37 % (en el de 100, con 4 de CPU, 97 %).

Memoria con las tuberias encendidas (clip 1080p de 663 cuadros y uno de campeonatos): etapa 4 RSS 3,0-3,2
GB y 5,8 de commit (sin tuberia 3,4 y 6,5); etapa 1 igual que antes (2,7-4,3). Por debajo de lo que
reserva el guardian.

**El pipeline queda listo. La corrida NO se lanza hasta que el usuario lo diga**:
`python 01_extraccion/corrida_florete.py --lanza` (y `--estado` / `--para`).

### CALIDAD DE LO QUE DEVUELVE (14/10; scripts en `02_validacion/auditoria/corrida_florete/calidad/`)

Hasta aqui todo era "sale completo, legible e identico". Esto es "dice la verdad", contra lo que se sabe:

| que | contra que | sale |
|---|---|---|
| luces (etapa 9) | `touch` del catalogo, 3.914 de florete en `pruebas` | la luz del que toco en **90,2 %**; solo la del otro 5,3 %; ninguna de color 4,5 % |
| luces en campeonatos | igual, 14 clips de los ensayos | 85,7 % (muestra pequenia) |
| zona del tocado (etapa 11) | `strip_location` a mano, 1.255 anotados de florete | **0,495**, contra 0,399 de contestar siempre `middle` |
| posicion de la etapa 7 en el tocado | lo mismo, mismos 1.255 | **0,346** (relativa al ancla) y 0,283 (absoluta): **peor que `middle`** |
| etapa 7 con homografia | anotados en `pruebas` (5.817) | **51,8 %**; "publicable" 8,8 % |
| etapa 11 con homografia (zona `ok`) | lo mismo | 60,9 %; en el 28,6 % de los clips la 11 la tiene y la 7 no |
| `pista_h_` (homografia por cuadro de la 11) | ensayos | H en el 80,6 % de los cuadros de media (mediana 100 %) |
| desplazamiento de cada evento, etapa 7 contra etapa 11 | 1.239 eventos de los ensayos | mismo sentido 85-93 % en marcha, romper, alto y fondo; \|dif\| p50 0,14-0,22 m; corr 0,6-0,74 (fondo 0,23, brazo 0,35) |

**Lo que dice**: la etapa 7 (scouting) mide bien CUANTO se mueve cada tirador (desplazamientos), pero
**NO DONDE esta en la pista**: su x va relativa a un ancla porque no sabe donde esta el centro, y como
posicion absoluta acierta menos que no mirar. Para la posicion en la pista vale la de la etapa 11, que
ademas cubre mas clips.

**Mejora propuesta (NO cambia la corrida y NO la retrasa)**: el "scouting v2", la posicion de los dos
tiradores en metros CUADRO A CUADRO con la homografia de la etapa 11 (`pista_h_`, que la corrida ya guarda
en cada clip con luz, con los pies en pixeles). Se calcula despues, en CPU, de lo guardado. Necesita su
criterio declarado antes de medirla (contra `strip_location` y la fisica: separacion entre tiradores,
velocidades) y que el usuario la quiera.

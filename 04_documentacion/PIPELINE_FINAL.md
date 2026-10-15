# El pipeline final de FENCINGLY: florete + campeonatos (15/10/2026)

**Estado: listo, probado de punta a punta y SIN LANZAR.** No se lanza hasta que el usuario diga
que se acabaron las pruebas. Este documento es la referencia única de la corrida: qué hace, cómo
se maneja, qué se probó y con qué resultado. El detalle de cada medida está en
[`informes/CORRIDA_FINAL_2026-10-13.md`](informes/CORRIDA_FINAL_2026-10-13.md) (§10) y en los
informes que se citan.

---

## 1. Dónde vive cada cosa

| qué | dónde |
|---|---|
| código y documentación | GitHub **privado** `EXCALOFRIO/FENCINGLY_PROYECTO` (`https://github.com/EXCALOFRIO/FENCINGLY_PROYECTO`), rama `main` |
| resultados por clip | `01_extraccion/pruebas/` (npz y csv) y `02_validacion/pista/` (series de la etapa 7) |
| respaldo de los resultados | Hugging Face **privado** `EXCALORIO/fencingly-extraccion`, cada 2 h mientras corren los trabajadores de `noche.py` (`lotes.sube`; sube `poses_`, `revision_`, `detx_`, `lights_`, `zona_`, `pista_h_`, `scout_` y las series; no sube vídeos ni `det_*.pkl`) |
| vídeos | `D:\FENCINGLY_VIDEOS_RAW\_descargados` (no se borran: todo se puede rehacer sin bajarlos otra vez) |
| V-JEPA | `D:\FENCINGLY_VJEPA\g2` y `\v0` |
| intérprete | `D:\FENCINGLY_PY\venv\Scripts\python.exe` |
| marcas y registro del director | `01_extraccion/pruebas/corrida_florete/` |
| salidas de las fases y auditorías | `02_validacion/auditoria/corrida_florete/` |
| pruebas de rendimiento (scripts y datos) | `D:\FENCINGLY_SCRATCH\` (`perfil`, `caudal`, `grafo`, `ensayo`, `scout`, `luces`, `memoria*`) |

El conjunto son **109.799 clips**: los 106.432 de `foil` más los 3.367 de florete de los campeonatos
2025-26 (`03_datos/catalogo/foil_y_campeonatos.txt`, lo escribe el director la primera vez).

## 2. Cómo se maneja (un solo comando)

Desde `01_extraccion`, con el intérprete de arriba:

    python corrida_florete.py --seco      qué haría y cuánto hay hecho, SIN lanzar nada
    python corrida_florete.py --lanza     arranca el director oculto (sin ventanas) y devuelve el control
    python corrida_florete.py --estado    cuánto lleva cada fase
    python corrida_florete.py --para      lo para todo, hijos incluidos

- **Reanudable**: el estado es el disco. `--para` (o un corte de luz, o reiniciar Windows) y otra vez
  `--lanza` sigue donde iba: cada fase acabada deja `HECHA_<fase>.txt`, y dentro de cada fase cada clip
  hecho se salta. Las escrituras son atómicas (`.part` + `os.replace`): un corte a mitad no deja un
  fichero roto que cuente como hecho.
- **Si una fase falla**, escribe `FALLA_<fase>.txt` y el director se para (no sigue a ciegas).
- **Nunca dos trabajos de GPU a la vez**: antes de cada fase de GPU espera a que la GPU esté libre.
- **Un cambio de código no entra en lo que ya corre**: `--para` y `--lanza`.
- Ensayo aislado (todo en un scratch, sin respaldo): `--ensayo LISTA` con `FENCINGLY_TRABAJO` y
  `FENCINGLY_PISTA` fuera de las salidas de verdad (lo exige).

## 3. Las fases, en orden

| fase | qué | GPU | días (estimado, 3060) |
|---|---|---|---:|
| `luces` | etapa 9 sobre los 109.799, bajando los ~81.000 mp4 que faltan (6 trabajadores) | no | ~1,3 |
| `lista` | `03_datos/catalogo/foil_dos_luces.txt`: los clips con las dos luces de color | no | minutos |
| `dos_luces` | etapas 1 y 4 (GPU) y 7, 9 y 11 (CPU) sobre la lista (~19.500) | sí | ~6,8-7,5 |
| `vjepa_g2` | V-JEPA 2 ViT-g 256, `rec4,mic4`, sobre la lista | sí | ~1,6 |
| `vjepa_v0` | V-JEPA 2 ViT-L 256, `rec4`, sobre la lista (para comparar) | sí | ~0,5 |
| `sonda_v0`, `sonda_g2` | la sonda de `prioridad` (decisión del árbitro) con V0 y con G2, y G2 contra V0 | sí | ~0,2 |
| `resto` | etapas 1, 4, 7, 9 y 11 sobre el resto (~86.000) | sí | ~30,0-33,1 |
| `zona_fusion` | la zona final del tocado, un modelo por torneo | no | horas |
| `scouting_v2` | la posición de los dos tiradores en metros, cuadro a cuadro (§4) | no | ~35 min |
| **total** | | | **~40-44 días** |

Lo que el director NO hace: la fase 3B (V-JEPA sobre el resto de clips). La decide el usuario con
`compara_prioridad_G2_vs_V0_corrida.txt` al acabar `sonda_g2`.

## 4. Qué sale de cada clip

| etapa | fichero (en `pruebas/` salvo que se diga) | qué |
|---|---|---|
| 1 | `det_<clip>.pkl`, `detx_<clip>.npz`, `revision_<clip>.npz` | YOLO11m-pose (960, fp16, más redetección fp32), la pareja de tiradores, cuadros útiles |
| 4 | `poses_<clip>.npz` | Sapiens 2 0.4B, 308 puntos por tirador y cuadro |
| 7 | `02_validacion/pista/`: series y `eventos_<clip>.csv` | marcha, romper, fondo, alto, brazo, mano armada; desplazamientos |
| 9 | `lights_<clip>.csv` | las luces del tocado y su cuadro |
| 11 | `zona_<clip>.csv`, `pista_h_<clip>.npz` | la x del tocado en metros y su zona; la homografía de CADA cuadro y los pies en píxeles |
| fusión | `zona_` (`zona_final`) | la zona final, por torneo |
| scouting v2 | `scout_<clip>.npz` | x, y en metros de cada tirador por cuadro, `estado` (medido / interpolado / nada), motivo de cada hueco, velocidad suavizada, separación |
| V-JEPA | `D:\FENCINGLY_VJEPA\g2\vjv_<clip>.npz` (y `v0`) | el resumen del vídeo para la decisión del árbitro (solo dos luces) |

Disco: ~1,5 MB por clip en `pruebas` (~166 GB en C, con ~360 libres); `pista_h_` ~14 KB y `scout_`
~4 KB por clip; los mp4 que faltan ~420 GB en D (1.510 libres).

**Cómo usar las posiciones del scouting**: la posición en la pista (dónde está cada tirador) se toma
de `scout_` (scouting v2); los desplazamientos y los eventos, de la etapa 7. La x de la etapa 7 NO es
una posición en la pista (va relativa a un ancla; §8).

## 5. Optimizaciones: todas con su criterio declarado ANTES de medir

| | qué | resultado | estado |
|---|---|---|---|
| exactas de CPU | mediana de la placa (6,8x), fusión de líneas (1,7x), índice de mp4, escrituras atómicas | idénticas bit a bit | **dentro** |
| O1 | perfil fino por llamada | el proceso reventaba al envolver ultralytics | abandonado |
| O2' | etapa 4 en tubería (lectura y preproceso en hilos, un lote por cuadro) | `poses_` idénticas 20/20 dos veces; 1,27x | **dentro** |
| O2b + O9 | las dos cajas de un cuadro en una llamada de preproceso; 2 hilos de torch | idénticas 20/20; con O2', 1,33x | **dentro** |
| O3 + O10 | etapa 1 con lectura en hilo; sin reconstruir el predictor de YOLO | `det_`/`detx_`/`revision_` 20/20; 1,21x | **dentro** |
| caudal exacto | todo lo de arriba con la corrida montada (3 GPU + CPU), 60 clips | todas las salidas 60/60 idénticas; 1,037x en paralelo | **dentro** |
| O4 | más trabajadores de CPU | con 4 de CPU, MemoryError y paginación en 1080p | se quedan **2** (con guardián de RAM) |
| O5 | etapas 7 y 11 con los 17 puntos de YOLO, sin Sapiens | zona −0,018; la mitad de los eventos cambian | **no pasa**: Sapiens en todos |
| O6 | Sapiens solo en una ventana antes del tocado | ahorra 4-9 días pero pierde 7-35 % de los eventos | **lo decide el usuario** (§9) |
| O7 | TensorRT / ONNX para YOLO | haría falta instalar en otro entorno | no probado |
| O8 | YOLO a 768 o 640 | cambia la pareja en 15-20 % de los cuadros; gana 5-13 % | **no pasa** |
| O11 | lote de 16 recortes en Sapiens (no exacta) | zona igual en 95 % (pide 98); 1,02x | **no pasa** |
| O12 | la pasada de Sapiens grabada en un CUDA graph | `poses_` idénticas 20/20 tres veces y 45/45 en paralelo; 1,34x en un proceso, 1,029x en paralelo | **dentro** (encendido el 14/10; `FENCINGLY_POSE_GRAFO=0` lo apaga) |
| lote 50 | 50 clips por lote en vez de 20 (arranque del modelo por lote) | 0,69 → 0,27 s por clip | **dentro** |
| luces con 6 | 6 trabajadores en `luces` en vez de 3 | 1,18x; `lights_` idénticos 120/120 | **dentro** |
| DINOv2 (D0) | rasgos cuadro a cuadro | no pasa su sonda | fuera |

Con la GPU al 93-95 % en paralelo, lo que queda por ganar en exacto es poco. Lo que sigue dando días es
lo que cambia el resultado (O6), y eso lo decide el usuario.

## 6. Fallos reales encontrados en las pruebas, y arreglados

1. **Nombres fuera de ASCII** (544 ids del catálogo, 177 de florete y 27 de campeonatos): no se bajaban
   nunca (url sin codificar) y un `print` tumbaba el proceso en cp1252. Arreglado (`urllib.parse.quote`,
   `PYTHONIOENCODING=utf-8`, CSV en UTF-8); bajados y procesados enteros en los ensayos.
2. **Carrera en `rutas.indice_videos`**: V-JEPA recibía "sin vídeo" para 6-8 clips por arranque.
   Arreglado (índice con cerrojo y publicado entero): V-JEPA en el 100 % de los de dos luces.
3. **Memoria**: con 4 trabajadores de CPU en clips 1080p, MemoryError. Se quedan 2, y cada etapa espera a
   tener RAM libre antes de arrancar (`rutas.espera_ram`, `lotes.RAM_ETAPA`).
4. **Hilos colgados si un clip falla a mitad** (tuberías): `try/finally` y aviso de parada; probado con
   fallos provocados.
5. **O2, primera versión**: poses distintas porque partía el lote del cuadro. Corregida (O2').

## 7. Pruebas del pipeline entero

| prueba | qué | resultado |
|---|---|---|
| ensayo de 100 (14/10 madrugada) | el director de verdad en `--ensayo`, 103 clips (anotados, sin procesar, sin mp4, campeonatos, no ASCII), 3 interrupciones con `--para` | 0 procesos vivos y 0 `.part` tras cada corte; siguió donde iba; 103/103 completos. Sacó los fallos 2 y 3 de §6 |
| ensayo de confirmación (14/10 18:08-18:52) | el código final, 41 clips, 2 interrupciones | 41/41 completos, V-JEPA en 18/18 de dos luces, 0 líneas de fallo, 0 `.part` |
| identidad 1 y 2 | cada optimización exacta contra la base, 20 clips, tiempos intercalados | ver §5 |
| caudal en paralelo | 60 y 45 clips con la corrida montada (ABAB) | ver §5 |
| fase `scouting_v2` (15/10) | el director sobre el ensayo de confirmación: salta las 7 fases hechas y corre solo la nueva | 41/41 `scout_`, 0 fallas |
| corte del scouting v2 (15/10) | 1.476 clips, matado a mitad (342 escritos) y relanzado | hizo solo los 1.134 que faltaban, 0 `.part`, 0 procesos vivos; una tercera pasada, 0 pendientes; los 1.476 idénticos a los de la validación |

## 8. Calidad de lo que devuelve (contra lo que se sabe)

| qué | contra qué | sale |
|---|---|---|
| luces (etapa 9) | `touch` del catálogo, 3.914 de florete | la luz del que tocó en **90,2 %** |
| zona del tocado (etapa 11) | `strip_location` a mano, 1.255 anotados | **0,495** (contestar siempre `middle`: 0,399) |
| posición de la etapa 7 | lo mismo | 0,346: **peor que `middle`**. Sirve para desplazamientos, no para posición |
| **scouting v2** | los 4.236 anotados de florete ([`informes/SCOUTING_V2_2026-10-14.md`](informes/SCOUTING_V2_2026-10-14.md)) | posición de los dos en el 56 % de los clips (etapa 7: 52 %; juntas, 78 %); física limpia 98,7-99,94 %; zona 0,495 (etapa 11 0,492, `middle` 0,399); desplazamientos en el mismo sentido que la etapa 7 en el 88 % de 21.417 eventos |
| "publicable" del scouting | 5.839 clips | 9 %: el problema es la escala de la pista (la estatura sale fuera de 1,60-2,00 m), no las marcas ([`informes/ANALISIS_SCOUTING_Y_VENTANA_2026-10-14.md`](informes/ANALISIS_SCOUTING_Y_VENTANA_2026-10-14.md)) |

Límite conocido: donde la cámara no ve las rayas de la pista no hay homografía (36 % de los pares
cuadro-tirador); ni la etapa 7 ni el scouting v2 dan posición ahí.

## 9. Decisiones del usuario

Tomadas:
- 13/10: las dos luces primero; G2 en los de dos luces; guardar la homografía de cada cuadro; G4 fuera;
  todo en la 3060, sin alquilar GPU.
- 13/10: los 3.367 de florete de campeonatos entran.
- 15/10: el scouting v2 entra, como la última fase del director.

Pendientes:
- **La ventana de Sapiens (O6)**: sin ventana (lo de ahora: no se pierde nada), −10 s (~4,3 días menos,
  pierde el 8 % de las marchas), −8 s (~5,5 días, 12 %) o −6 s (~9 días, 25 %). Si no se decide, va sin
  ventana.
- La fase 3B (V-JEPA sobre el resto), con la comparación de `sonda_g2`.
- **La orden de lanzar**: `python 01_extraccion/corrida_florete.py --lanza`.

## 10. Las reglas del proyecto

1. Criterio y línea base declarados ANTES de medir. Si no pasa, no pasa.
2. Lo que no funciona se entrega apagado y con su cifra.
3. `None`, no cero.
4. Toda cifra con su denominador, y separada por `via`.
5. Repositorios y datasets privados: los clips enseñan atletas identificables.

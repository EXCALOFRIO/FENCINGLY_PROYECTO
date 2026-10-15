# FENCINGLY — empieza aquí

> **LA CORRIDA FINAL (florete + campeonatos, 109.799 clips) está en
> [`04_documentacion/PIPELINE_FINAL.md`](04_documentacion/PIPELINE_FINAL.md)**: qué hace, cómo se
> lanza, para y reanuda (`01_extraccion/corrida_florete.py`), todas las pruebas y optimizaciones con
> su resultado, y lo que falta por decidir. Lista y probada; **sin lanzar** hasta que el usuario lo diga.

**Este es el ÚNICO proyecto.** Extracción de datos de clips de esgrima (florete primero) con
YOLO11 + Sapiens 2 (308 puntos). El estado medido de hoy está en
[`04_documentacion/informes/ESTADO_2026-10-08.md`](04_documentacion/informes/ESTADO_2026-10-08.md).
La bitácora completa de septiembre (6.900 líneas, con todas las medidas y retractaciones) está en
[`04_documentacion/BITACORA_2026-09.md`](04_documentacion/BITACORA_2026-09.md).

---

## 1. Dónde vive cada cosa

| qué | dónde | copia |
|---|---|---|
| código y documentación | este directorio | **GitHub privado** `EXCALOFRIO/FENCINGLY_PROYECTO` |
| resultados de la extracción (`poses_`, `revision_`, `lights_`, `zona_`, CSV de pista) | `01_extraccion/pruebas/` y `02_validacion/pista/` | **HF privado** `EXCALORIO/fencingly-extraccion`, cada 2 h (`respalda.py`) |
| vídeos descargados | `D:\FENCINGLY_VIDEOS_RAW\_descargados` | el CDN de origen |
| intérprete | `D:\FENCINGLY_PY\venv` | — (se reconstruye con `requirements-colab.txt`) |
| pesos de Sapiens 2 | `D:\FENCINGLY_HF` | Hugging Face |
| pesos de YOLO | `01_extraccion\yolo11m-pose.pt` | ultralytics |
| catálogo y etiquetas humanas | `03_datos/catalogo/` | GitHub |
| tablas de scouting | `03_datos/scouting/` | GitHub (se regeneran con `codigo/scouting.py`) |

**Nada de la ruta crítica vive en `%TEMP%`.** Dos noches se perdieron por eso (22/09 y casi el
08/10); `python 01_extraccion/codigo/rutas.py` imprime cada raíz y si existe.

## 2. Correr, mirar, parar

Desde `01_extraccion`, con `D:\FENCINGLY_PY\venv\Scripts\python.exe`:

    python noche.py --prevuelo                                    ¿el intérprete puede con todo?
    python noche.py --conjunto anotados --descarga --gpu 2 --cpu 2   lanza y devuelve el control
    python noche.py --estado --conjunto anotados                   cuánto lleva (mira el DISCO)
    python noche.py --para                                         para todo, hijos incluidos

Para el corpus entero de florete (106.432 clips): `--conjunto foil`. Lo ya hecho de `anotados` se
salta solo, porque el estado es el disco.

- **Reanudable**: el estado es lo que hay en disco + lo que hay en el remoto. Matar y relanzar
  sigue exactamente donde estaba. Un reinicio de Windows no pierde nada: relanzar la misma línea.
- **Un cambio de código no entra en los trabajadores que corren** (ejecutan las etapas dentro del
  proceso y los módulos quedan en caché): `--para` y relanzar.
- Trabajadores: `gpu*` etapas 1 y 4, `cpu*` etapas 7, 9 y 11, `resp` el respaldo a HF.
- **AL TERMINAR la corrida, un paso que la noche NO hace sola** (2026-10-09): la zona final del
  tocado necesita un modelo POR TORNEO (ángulo del borde lejano contra la homografía,
  `ANGULO_FUSION_2026-10-09.md`), así que se corre cuando el torneo esté entero:
  `python zona_fusion.py --conjunto foil --escribe`. Hasta entonces `zona_` lleva
  `via_zona = pendiente_torneo` y `zona` (la de la homografía sola) sigue valiendo.
- **El florete NO es la corrida final mientras se esté midiendo** (lo pidió el usuario el
  2026-10-09): se prueba sobre los 5.837 anotados y el florete se lanza cuando se dé por cerrado.

## 3. Qué sale de cada clip

Tabla completa, con el error medido de cada campo, en
[`02_validacion/DATOS_POR_CLIP.md`](02_validacion/DATOS_POR_CLIP.md).

| etapa | fichero | qué |
|---|---|---|
| 1 | `det_`, `revision_<clip>.npz` | cajas de YOLO, pareja de tiradores, planos útiles, cadencia |
| 4 | `poses_<clip>.npz` | 308 puntos por tirador y cuadro, medido vs. rellenado |
| 7 | `serie_eventos_`, `posicion_clip_<clip>.csv` | marcha, romper, fondo, brazo, mano armada, pista |
| 9 | `lights_<clip>.csv` | cuadro del tocado por las luces (`via` dice de dónde sale) |
| 11 | `zona_<clip>.csv` | x del tocado en metros por las rayas de la pista; `zona_final` tras `zona_fusion.py` |
| 11 | `pista_h_<clip>.npz` | la homografía de cada cuadro y los pies en píxeles (desde el 13/10) |
| scouting v2 | `scout_<clip>.npz` | la posición de cada tirador en metros, cuadro a cuadro (desde el 15/10; `codigo/scouting_v2.py`) |
| 10 | `etiquetas_<clip>.csv` | quién inicia y zona (apagada hasta guardar los pesos) |

## 4. Las reglas del proyecto

1. **Criterio y línea base declarados ANTES de medir.** Si no pasa, no pasa, aunque sea por poco.
2. Lo que no funciona se entrega **apagado y con su cifra**, no se esconde.
3. **`None`, no cero**: un dato que falta no es un cero.
4. Toda cifra con su **denominador**, y separada por `via` cuando hay más de una fuente.

## 5. Mapa de carpetas

    01_extraccion/      pipeline.py (etapas), lotes.py (bucle), noche.py (corrida), respalda.py
      codigo/           módulos; los ~80 a*/e*/f*/capas_* son experimentos con cifras publicadas
    02_validacion/      verificaciones, anotaciones a mano, herramienta/ (verifica_*.py)
    03_datos/           catálogo, etiquetas, scouting, modelos pequeños
    04_documentacion/   informes fechados, bitácora
    05_aparcado/        Colab y web, aparcados por decisión
    99_historico/       lo archivado (el pipeline RTMO del 08/10, copias viejas) — fuera de git

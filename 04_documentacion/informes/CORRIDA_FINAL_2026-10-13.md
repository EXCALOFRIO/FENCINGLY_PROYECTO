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
| DINOv2 | GPU | rasgos cuadro a cuadro | pendiente de su sonda (D0, mie 14/10) |
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

# Analisis del 14/10: por que casi nada del scouting sale `publicable`, la ventana de Sapiens y otras comprobaciones

Solo lectura de salidas que ya existian (`01_extraccion/pruebas`, `02_validacion/pista`, los dos ensayos del
14/10 en `D:/FENCINGLY_SCRATCH/ensayo`) y del codigo; ninguna etapa corrida, nada escrito en las salidas de
verdad. Scripts y cifras en `02_validacion/auditoria/corrida_florete/analisis/` (`a1_*` a `a4_*`, cada uno
con su `.json`). Sigue a `CORRIDA_FINAL_2026-10-13.md` §10, "CALIDAD DE LO QUE DEVUELVE".

## 1. Por que solo ~9 % del scouting sale `publicable`

**Que es `publicable`** (`pista.py` ~2298, declarado antes de correr nada): la homografia de la etapa 7 existe
Y la estatura reproyectada de los tiradores cae en 1,60-2,00 m (`C1_pasa`, mediana AGRUPADA de los dos) Y hay
>= 2 marcas transversales afinadas Y `h_p50` es finita. Es una puerta sobre la ESCALA del clip, no sobre los
eventos. Encima, cada fila de la serie tiene su puerta fisica (`fisica_evento`, 09/10). `C1_pasa` y las
marcas no se guardan en las salidas de la corrida; se usa `puerta_dos_lados` (las dos estaturas en rango) y
el `homografia.csv` antiguo (78 clips, septiembre) para separar.

**En cascada** (5.839 clips con `posicion_clip_`; `a1_publicable.json`):

| paso | todos | florete | espada | sable |
|---|---:|---:|---:|---:|
| clips | 5.839 | 4.240 | 1.036 | 563 |
| 1. sin homografia en la etapa 7 | **2.810 (48,1 %)** | 2.030 (47,9 %) | 536 (51,7 %) | 244 (43,3 %) |
| 2. con homografia pero NO `publicable` | **2.505 (42,9 %)** | 1.845 | 422 | 238 |
| &nbsp;&nbsp;de esos, con alguna estatura fuera de 1,60-2,00 m | 2.503 | 1.845 | 420 | 238 |
| 3. `publicable` | **524 (9,0 %)** | 365 (8,6 %) | 78 (7,5 %) | 81 (14,4 %) |
| &nbsp;&nbsp;y las dos estaturas en rango (la puerta estricta) | 194 (3,3 %) | 127 | 38 | 29 |
| &nbsp;&nbsp;y pasa el control fisico del clip (`pasa_fisica`) | 97 (1,7 %) | 68 | 6 | 23 |
| 4. las dos cosas | **40 (0,7 %)** | 27 | 3 | 10 |

- **Lo que tumba mas clips es no tener homografia (48 %)**. Detras, **la escala**: de los 2.505 con homografia
  y no publicables, 2.503 tienen la estatura reproyectada de algun tirador fuera de rango. En el
  `homografia.csv` antiguo, de los 33 con homografia no publicables, 31 fallan la estatura con >= 2 marcas
  afinadas y 2 con < 2: **es la estatura, no las marcas**.
- **Por separado, en los 3.029 con homografia**: estaturas de los dos en rango 196 (6,5 %); `pasa_fisica` 306
  (10,1 %); ancla fiable 471 (15,5 %); separacion de pies plausible (`pasa_dpies`) 1.020 (33,7 %); dentro de la
  pista (`pasa_fuera`) 1.550 (51,2 %); transversal creible 1.182 (39,0 %).
- **Filas de la serie**: 126.235; publicables 17.346 (13,7 %). Rebajadas por la puerta fisica del evento
  7.375 (`fuera_de_pista` y `desplazamiento` las que mas). Por tipo, publicables: brazo 4.246, marcha 4.041,
  romper 3.827, alto 2.847, accion 1.197, fondo 1.185.
- **Coherente con lo de antes**: 2.821 de 5.839 clips (48,3 %) no tienen ni una marcha, romper, fondo ni alto;
  son los sin homografia, que solo dan `accion` (altos de caja) y `brazo` (fraccion de torso).

**Lo que arreglaria el scouting v2** (la posicion por cuadro con la homografia de la etapa 11, `pista_h_`):
- los **sin homografia** (2.810): de ellos, 1.666 (59 %) tienen zona `ok` en la etapa 11; en los dos
  ensayos (`a1b_cobertura_v2.json`), de 70 clips sin homografia en la 7, **44 (63 %) tienen la de la 11 en >= 50
  % de sus cuadros**. Cobertura de `pista_h_` con su umbral de confianza (0,35): media 62 %, mediana 79 % de
  los cuadros (el 80,6 % de antes contaba cualquier H finita, sin umbral); 24 de 144 clips con 0 cuadros;
- **la posicion absoluta**: la etapa 7 va relativa a un ancla y no sabe donde esta el centro (0,346 / 0,283
  contra `middle` 0,399); la 11 da 0,495 (`CORRIDA_FINAL` §10).

**Lo que NO arreglaria**:
- **la puerta de escala**: si se le pide a la homografia de la 11 la misma estatura en rango, no se sabe cuanto
  pasa. Y la 11 calibra usando la altura de las cajas (`alt`) como ayuda, asi que comprobarla con la estatura
  seria en parte circular: le hace falta otra prueba (separacion entre tiradores, longitud de paso, velocidad);
- **los eventos**: marcha, romper, fondo y alto los detecta la etapa 7 con su propia homografia. El v2 da
  posiciones, no eventos, salvo que se rehaga la 7 con la homografia de la 11 (CPU y video; los mp4 se quedan);
- `brazo` y `accion`, que no van en metros;
- los clips sin luz (3,6 %, sin tocado): la 11 no calibra y no hay `pista_h_`;
- 1.144 clips sin homografia en ninguna de las dos (19,6 %).

## 2. La ventana de Sapiens (O6), con su coste y lo que se perderia

Unidad de coste de la etapa 4: la **pasada** (un cuadro util con al menos una caja). Tocado = `cuadro_tocado`
de `zona_` (si no, el primer `cuadro_encendido` de `lights_`). Los clips sin tocado se suponen enteros.
5.873 clips con `revision_` (`a2_ventana.json`); 214 sin tocado (3,6 %, el 3,0 % de las pasadas).

| ventana | pasadas que QUEDAN (todas) | florete | espada | sable | por clip, mediana | ahorro estimado |
|---|---:|---:|---:|---:|---:|---:|
| [tocado - 6 s, tocado + 1 s] | **68,7 %** | 69,4 % | 66,1 % | 68,2 % | 69,7 % | **~9,0-9,5 dias** |
| [tocado - 8 s, tocado + 1 s] | 81,6 % | 82,4 % | 79,1 % | 80,1 % | 86,8 % | ~5,3-5,6 dias |
| [tocado - 10 s, tocado + 1 s] | 85,5 % | 85,8 % | 84,1 % | 85,4 % | 88,8 % | ~4,2-4,4 dias |

Dias: 105.500 clips (86.000 de `resto` + 19.500 de `dos_luces`) x 31,0 s/clip (116 clips/h en paralelo) x
(76-80 %, la parte de la etapa 4: en un proceso 22-29 s de 4 contra ~7 de 1) x (1 - lo que queda). Es un
**ESTIMADO**: supone que en paralelo el tiempo baja en proporcion a las pasadas de Sapiens, y los clips del
corpus son algo mas largos que los anotados (mediana 331-423 cuadros contra ~300), asi que el ahorro real
podria ser algo mayor.

**Lo que se perderia**: eventos de la etapa 7 que EMPIEZAN fuera de la ventana (sin pose de Sapiens no hay
pies ni manos; la via con los 17 puntos de YOLO, O5, NO PASO):

| tipo | n | fuera con -6 s | -8 s | -10 s |
|---|---:|---:|---:|---:|
| marcha | 23.031 | **25,5 %** | 12,0 % | 7,7 % |
| romper | 22.140 | **35,4 %** | 22,8 % | 18,6 % |
| fondo | 7.293 | **20,4 %** | 10,5 % | 7,3 % |
| alto | 11.262 | 27,2 % | 15,3 % | 11,3 % |
| brazo | 47.283 | 28,4 % | 16,3 % | 12,0 % |
| accion | 12.463 | 48,4 % | 22,9 % | 8,7 % |

Con -6 s se pierde entre una quinta y una tercera parte de los eventos del scouting, sobre todo la
preparacion lejos del tocado (romper, marchas largas). Con -10 s el ahorro baja a ~4 dias y se pierde el
7-19 %. **No se propone como adoptado**: cambia el resultado (el scouting de la primera parte del clip) y lo
decide el usuario. Los clips sin luz quedarian enteros (o sin pose, si se decidiera saltarlos).

## 3. Otras comprobaciones y ahorros exactos

**Calidad en `pruebas`** (`a4_calidad.json`):
- poses: en los cuadros utiles hay **las dos poses en el 100 %** (por construccion: los huecos se inventan y
  los cuadros sin las dos se recortan); medidas las dos 95,7 % de media en florete (p10 86,9 %); inventadas
  2,2 %. Cuadros utiles del clip: media 87,4 %, p10 55,6 %. Clips sin un solo cuadro util: 33 de florete (0,8
  %), 7 de espada, 2 de sable;
- **la pareja que rehace la etapa 7 no lleva la redeteccion**: `pista.pareja` llama a `extrae_pareja.procesa`
  sin YOLO, y sin YOLO se salta la redeteccion. En 1.159 de 4.261 clips de florete (27 %) hay cuadros
  redetectados (1,4 % de los cuadros utiles de media): ahi las cajas de la etapa 7 no son las de `revision_`
  ni las de las poses de la etapa 4. Es pequenio pero es una incoherencia entre etapas; leer `revision_` en vez
  de rehacer la pareja la quitaria y ahorraria dos `procesa` por clip (~0,4 s cada uno), pero CAMBIA el
  resultado de la 7 en esos cuadros: hay que medirlo antes;
- valores imposibles en las filas publicables (17.346): x absoluta (`x_inicio_m + x_origen_m`) fuera de +-8 m en
  516 (3,0 %), paso de marcha o romper > 1,5 m en 210 (1,2 %), velocidad > 9 m/s en 1. En las no publicables,
  |x| > 8 m en 15.148 de 108.889 (13,9 %): por eso no se publican.

**Ahorros EXACTOS en las etapas 7, 9 y 11** (perfiles con cProfile en dos clips, 720p y 1080p; `a3_*.json`):
- **etapa 9**: decodifica el video UNA vez (mas una apertura solo para leer la cadencia); decodificar es el
  52-58 % de su tiempo (0,8 s en 720p, 2,1 s en 1080p) y necesita todos los cuadros. El lector v2 ya salta con
  `grab()` lo que no usa. **No queda nada exacto que valga la pena.**
- **etapa 11**: 14-24 s por clip; decodificar ~1,6 s (11 %). El tiempo es `HoughLinesP` (5,8-6,5 s),
  `findHomography` dentro de `calibrate` (hasta 13.157 llamadas y 5 s en un clip) y el flujo optico. El angulo
  de la fusion re-recorre el video con `grab()`, pero cuesta 0,4-0,6 s. Cambiar `findHomography` de 4 puntos
  por `getPerspectiveTransform` seria mas rapido pero NO es la misma cuenta.
- **etapa 7**: lee TODOS los cuadros a memoria (por eso 3,2 GB en 1080p) y rehace la pareja dos veces. La
  primera (~0,4 s) solo da W, H, T y decide "sin tramo utilizable" ANTES de leer el video; quitarla es exacto
  salvo en los clips cuya primera mascara sale vacia y la segunda no (habria que contarlos antes).
- **Lo que dan en dias: ~0.** En `dos_luces` y `resto` la GPU es el cuello (la CPU al 37 % con 2 trabajadores
  de CPU en el ensayo de confirmacion), asi que acelerar las etapas 7 y 11 no acorta la corrida. Solo contaria
  en una pasada de CPU sola (p. ej. rehacer la etapa 7 con la homografia de la 11 despues de la corrida), y la
  etapa 9 (fase `luces`, CPU) no tiene margen exacto.
- `grab()` sin `retrieve()` cuesta ~5 veces menos que `read()` (720p: 0,22 contra 1,15 s el clip entero;
  1080p: 0,38 contra 1,96): es la herramienta si alguna etapa lee cuadros que luego no usa.

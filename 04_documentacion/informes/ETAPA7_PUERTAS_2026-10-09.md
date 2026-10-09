# Etapa 7: puerta física por evento y mano armada con la cascada completa (2026-10-09)

Antes de lanzar el florete completo (106.432 clips), dos defectos conocidos de la etapa 7
(`pipeline._eventos` -> `grafica_pista.procesa` -> `posicion.py`). Este informe se escribió en dos
tiempos: **§1 y §2 (criterios y líneas base) antes de medir nada**, §3 y siguientes después.

---

## 1. Puerta física POR EVENTO (`FENCINGLY_PUERTA_EVENTO`)

### El defecto

`publicable` en `serie_eventos_<clip>.csv` es del CLIP: dice si la escala métrica del clip pasa el
control físico (`pista.py`, estatura + marcas). La auditoría del 09/10 (`chk_publicable_escape.py`)
encontró **287 filas en 49 clips** con `publicable=True` y valores imposibles (x hasta 538 m,
v hasta 2.324 m/s): la escala del clip pasa, pero en algunos cuadros la homografía propagada
revienta y los eventos de esos cuadros heredan el `True` del clip.

### Lo que se añade, declarado

En `posicion.serie`, para cada fila, tres comprobaciones que solo usan magnitudes en METROS y
cuyos umbrales salen de la física, no de los datos:

| clave | condición | por qué ese umbral |
|---|---|---|
| `velocidad` | `abs(velocidad_m_s) > 8,0` | un fondo o una flecha no pasan de ~5-6 m/s de media en el evento; 8 m/s es velocidad de esprint |
| `desplazamiento` | `abs(x_fin_m - x_inicio_m) / max(duracion_s, 1/fps) > 8,0` | lo mismo para las filas sin `velocidad_m_s` (p. ej. `brazo`) |
| `fuera_de_pista` | `abs(x_inicio_m)` o `abs(x_fin_m) > 16,0` | la x va relativa al anclaje (mediana del punto medio de los dos, que está sobre la pista): 14 m de pista + 2 m de zona de retroceso es la máxima distancia posible |

Columnas nuevas en la serie: `fisica_evento` (True / False / **None** si la fila no tiene ninguna
magnitud en metros que comprobar) y `motivo_no_publicable` (las claves que fallan, unidas por `+`,
o None). `publicable_clip` guarda el valor de antes. `publicable` pasa a ser
`publicable_clip AND fisica_evento is not False`. En `posicion_clip_<clip>.csv`: `n_eventos_fisica_mal`.
El evento **no se borra**. `publicable` del CLIP (en `posicion_clip_`) no cambia.

### Criterio de adopción, declarado ANTES de medir

Línea base: lo que hay hoy en `02_validacion/pista`, sin la puerta.

- (a) **Implementación**: tras la puerta, 0 filas con `publicable=True` que violen alguna de las tres
  condiciones; comprobado sobre TODOS los clips anotados ya procesados (aplicando la función de la
  puerta a copias de sus CSV) **y** de punta a punta en una muestra recalculada con el código nuevo.
- (b) **Daño colateral**: la puerta marca como no publicables **<= 5 %** de las filas que hoy son
  `publicable=True`. Si marca más, la puerta no se enciende hasta mirar por qué (un 20 % diría que
  `velocidad_m_s` tiene un artefacto de definición, no que el 20 % sea basura).
- (c) **No toca al etiquetador**: en la muestra recalculada, las columnas que lee
  (`lado`, `tipo`, `cuadro_inicio`, `x_inicio_m`) son idénticas fila a fila.
- Se informa además, sin ser puerta: cuántas de las 287 filas de la auditoría (definidas con
  `|x| > 8`, más estricto) quedan cazadas y el perfil de las que no.

Si pasa (a), (b) y (c): defecto `FENCINGLY_PUERTA_EVENTO=1`. Si no: `0`, con su número.

---

## 2. Mano armada con la cascada completa (`FENCINGLY_MANO_ACCION`)

### El defecto

`posicion._brazos` llama a `cono_lib.mano_armada(KP, SC, B, W, H, lado)` **sin `org`, sin `mask`
y sin `accion`** (NOTA_8 de `pipeline.py`). Sin `accion` el peldaño `encarado_accion` no puede
dispararse, y sin `org`/`mask` el voto del pie usa también cuadros inventados y fuera del tramo
útil. Las cifras citadas de la cascada (1,000 en 19 reservados, 0,949 en 59) se midieron CON los
tres argumentos.

La mano importa más allá de la columna `mano`: el evento `brazo` se mide sobre el brazo de esa mano,
así que una mano equivocada mide la extensión del brazo de atrás. `marcha`, `romper`, `fondo`,
`alto` y `accion` no dependen de la mano.

### Lo que se mide, declarado

Verdad: la **mano de la licencia FIE** (`fie.manos()`, mapeo `A` = diestra como en
`verifica_mano_criterios.py`), en todos los clips con `poses_` y `det_` en disco. Por tirador
(mayoría de sus clips, empate = no contesta), que es la cifra que manda, y por clip.

- **V0** (hoy): `mano_armada(KP, SC, B, W, H, lado)`.
- **V1** (candidato): `mano_armada(..., org=origen, mask=mask, accion=máscara de los eventos
  `accion` de la propia etapa 7)`, ambas vías (metros y altos de caja).
- V0m (`org` + `mask`, sin `accion`): solo para atribuir la diferencia, no es candidato.

### Criterio de adopción, declarado ANTES de medir

Se adopta V1 (`FENCINGLY_MANO_ACCION=1` por defecto) si y solo si:

- (d) por tirador, McNemar de dos colas V1 contra V0: **gana > pierde y p < 0,05**;
- (e) por clip, `acierto*` (mudos = fallo) de V1 **>= V0**;
- (f) en la muestra recalculada de punta a punta, las filas que no son `brazo` son **idénticas**
  (no se mueven las cifras de la pista).

Si no: `FENCINGLY_MANO_ACCION=0`, con su número. Aviso de solapamiento: los 59 tiradores anotados a
ojo con los que se eligió la cascada pueden estar entre los de la FIE; se cuenta cuántos.

---

## 3. Resultados de la puerta por evento (medido después de escribir §1)

`02_validacion/herramienta/verifica_puerta_evento.py`, sobre TODAS las series ya escritas del
conjunto `anotados` (leídas sin tocarlas; las series con la puerta quedan en
`D:/FENCINGLY_SCRATCH/agente_etapa7/series_con_puerta/`). La cadencia de cada clip sale de su
`posicion_clip_` (5.566 de 5.566).

    clips con serie 5.566   filas 119.175   filas publicable=True antes 16.202

| criterio | medido | pasa |
|---|---|---|
| (a) `publicable=True` que violan la física | antes **156**, después **0** | sí |
| (b) filas rebajadas / publicables | **156 / 16.202 = 0,0096** (44 clips), criterio <= 0,05 | sí |

Por motivo: velocidad+desplazamiento+fuera 32, velocidad+desplazamiento 31, desplazamiento 29,
desplazamiento+fuera 25, fuera 22, velocidad 13, velocidad+fuera 4. Concentrado: 28 de los 44 clips
tienen 1 o 2 filas; los 5 peores (`FlamboyantBigheartedAlpinegoat`, `puzzledStoatCaneCorso`,
`HugeAmusingCornsnake`, `PlainDefinitiveDamselfly`, `terribleBeardedFirewormPuggle`) suman 73.

| tipo | rebajadas / publicables | | arma | rebajadas / publicables |
|---|---:|---|---|---:|
| fondo | 43 / 1.132 = 0,038 | | foil | 101 / 11.221 = 0,009 |
| brazo | 51 / 3.962 = 0,013 | | epee | 39 / 2.683 = 0,015 |
| marcha | 29 / 3.774 = 0,008 | | sabre | 16 / 2.298 = 0,007 |
| romper | 26 / 3.592 = 0,007 | | | |
| accion | 6 / 1.125 = 0,005 | | | |
| alto | 1 / 2.613 = 0,000 | | | |

**El fondo es el tipo más rebajado, y 12 de sus 43 lo son SOLO por `velocidad`** (8,4 a 13 m/s con x
y desplazamiento creíbles). La `velocidad_m_s` del fondo es magnitud / tiempo hasta el pico
(`posicion._fondos`), así que un pico detectado 1-2 cuadros después del arranque la infla: esas 12
filas son más probablemente un fallo de TIEMPO que de posición. Se rebajan igual (una velocidad
media de 10 m/s no es publicable), pero queda dicho que no son homografía rota.

**Contra la auditoría del 09/10** (`v > 8` o `|x| > 8`, más estricta en x): 286 filas, **cazadas
136 (0,476)**. Las 150 restantes tienen `max|x|` entre 8,01 y 15,83 m (mediana 9,54) y **ninguna
v > 8**: son posiciones lejos del anclaje con movimiento suave, no imposibles por sí mismas (el
anclaje es la mediana del punto medio, no el centro de la pista). Lo que puede estar mal en ellas es
la ESCALA del clip, y eso es la puerta del clip (`publicable` de `posicion_clip_`), no la del evento.

## 4. Resultados de la mano armada (medido después de escribir §2)

`02_validacion/herramienta/verifica_mano_accion.py`, sobre los 5.703 clips anotados con `poses_` y
`det_` (0 errores). Por clip en `D:/FENCINGLY_SCRATCH/agente_etapa7/mano_accion_por_clip.csv`.

    decisiones (clip, lado) con ficha FIE 9.751   tiradores 958   azar informado (siempre R) 0,743

| | acierto | cobertura | acierto* | por tirador: acierto | cobertura | acierto* |
|---|---:|---:|---:|---:|---:|---:|
| V0 (hoy) | 0,900 | 0,975 | 0,878 | 0,938 | 0,961 | 0,902 |
| V0m (`org`+`mask`) | 0,900 | 0,975 | 0,878 | 0,938 | 0,961 | 0,902 |
| **V1** (+`accion`) | **0,914** | 0,975 | **0,892** | 0,941 | 0,959 | 0,903 |

Por arma, acierto* por clip V0 -> V1: florete 0,875 -> 0,891 (n=6.948), espada 0,909 -> 0,915
(1.808), sable 0,844 -> 0,858 (995). Por `via_accion`: con eventos `accion` 0,881 -> 0,895 (9.523);
sin serie de eventos 0,741 -> 0,750 (228, ahí V1 = V0m).

Peldaño que decide: V0 `pie_firme` 4.857, `nariz_dir_dudosa` 4.631, `geometria_a` 23, None 240;
V1 `pie_firme` 4.886, **`encarado_accion` 4.382**, `nariz_dir_dudosa` 201, `geometria_a` 42, None 240.
Cambian de mano 301 de 9.751 decisiones. `org`+`mask` solos no cambian nada: toda la diferencia es
el peldaño `encarado_accion`.

| criterio declarado | medido | pasa |
|---|---|---|
| (d) por tirador, McNemar V1 contra V0: gana > pierde y p < 0,05 | gana **9**, pierde **6**, **p = 0,61** (n=912) | **NO** |
| (e) por clip, acierto* V1 >= V0 | 0,892 >= 0,878 | sí |
| (f) filas no-`brazo` idénticas | ver §5 | |

**VEREDICTO: NO SE ADOPTA. `FENCINGLY_MANO_ACCION=0` por defecto**, porque (d) era la cifra que
mandaba y no pasa: la mayoría de los clips de cada tirador ya acertaba, así que el voto por tirador
casi no se mueve (15 tiradores cambian de veredicto, 9 a bien y 6 a mal).

**Lo que queda dicho para decidir después, y NO es una adopción:** la etapa 7 no decide por tirador
sino POR CLIP (el `brazo` de cada clip se mide sobre la mano de ese clip), y por clip la mejora es
clara: 219 decisiones pasan a bien y 82 a mal; **+0,0140 de acierto* con IC95 [+0,0094, +0,0189]
agrupando por tirador** (bootstrap de 4.000, 113 tiradores mejoran y 52 empeoran). Florete +0,0160
[+0,0100, +0,0223]; espada +0,0066 [+0,0012, +0,0125]; sable +0,0141 [-0,0021, +0,0302]. Ese
criterio se ha formulado DESPUÉS de ver los datos, así que no vale para adoptar sobre esta misma
muestra: hay que declararlo y medirlo en clips nuevos (p. ej. los primeros clips de florete con ficha
FIE). La etapa 7 es CPU y recalculable, así que esperar no pierde nada salvo recalcularla.

Solapamiento: 49 de los 958 tiradores de la ficha están entre los anotados a ojo con los que se
eligió la cascada.

## 5. De punta a punta, en copia (`verifica_etapa7_copia.py`)

La etapa 7 recalculada con `grafica_pista.SALIDA` apuntando a `D:/FENCINGLY_SCRATCH/agente_etapa7/`,
nunca a `02_validacion/pista`. Dos pasadas: 222 clips (200 al azar, semilla 20261009, + 24 de los 44
con filas rebajadas) y 53 (40 al azar, semilla 7, + los 14 donde cambiaba la mano).

| comprobación | 1.ª pasada (222) | 2.ª pasada (53) |
|---|---|---|
| excepciones | 0 | 0 |
| base (las dos variables a 0) idéntica a disco | 222 / 222 | 51 / 53 (*) |
| (a) `publicable=True` y física mal, de punta a punta | 0 filas (93 rebajadas) | 0 |
| (c) puerta sola: columnas del etiquetador idénticas | (no separada) | **53 / 53** |
| puerta sola: resto de columnas idénticas salvo `publicable` | | 53 / 53 |
| (f) puerta + mano: filas no-`brazo` idénticas | **222 / 222** | **53 / 53** |
| puerta + mano: columnas del etiquetador idénticas | 208 / 222 | 36 / 53 |

Las 14 (y 17) discrepancias de la última fila son EXACTAMENTE los clips donde cambia la mano: el
`brazo` se mide sobre otro brazo y sale en otros cuadros. Con la mano apagada no se mueve nada.

(*) Las 2 de 53 que no reproducen el disco (`PolishedEthicalAntbear`, `BrilliantUnlinedKingbird`)
no son de esta tarea: entre las dos pasadas otro agente encendió la puerta de piernas de la pareja
(`pareja_v2.PIERNA_K = 0,5` por defecto, `pista.analiza` rehace la pareja con fotogramas), y en esos
dos clips la pareja de la etapa 7 ya no es la de disco. **Ojo: recalcular la etapa 7 de un clip ya
extraído con esa puerta encendida puede usar una pareja distinta de la de sus `poses_`.**

`01_extraccion/aplica_puerta_evento.py` pone la puerta en las series ya escritas sin recalcular la
etapa 7: en 102 ficheros de la 2.ª pasada sale **idéntico** (contenido y orden de columnas) a la
etapa 7 recalculada con la puerta, y en los 26 clips con filas rebajadas de la 1.ª, las filas
no-`brazo` también. En seco sobre todo `anotados` hoy: 5.627 series, 163 filas publicables
rebajadas, 820 clips con alguna fila con `fisica_evento = False` (la mayoría ya no publicables por
el clip).

## 6. Veredictos

| pieza | variable | defecto | por qué |
|---|---|---|---|
| puerta física por evento | `FENCINGLY_PUERTA_EVENTO` | **1 (ADOPTADA)** | (a) 156 -> 0, (b) 0,0096 <= 0,05, (c) 53/53 |
| mano con la cascada completa | `FENCINGLY_MANO_ACCION` | **0 (NO)** | (d) por tirador 9 contra 6, p = 0,61; por clip +0,014 IC95 [+0,009, +0,019], formulado después |

Otros fallos de la etapa 7 buscados (§3 del encargo): **0 `FALLA` de la etapa 7 en `cpu0`/`cpu1`
desde el último arranque de los trabajadores** (los `max() iterable argument is empty` y
`solape sin arbitrar` de los logs son anteriores, y sus clips ya están rehechos). Los 124 clips con
`poses_` y sin etapa 7 están en cola de los trabajadores de CPU, no fallando.

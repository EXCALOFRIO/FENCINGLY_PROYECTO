# Luces por fuente y el cuadro del tocado = 0 (2026-10-09)

Encargo: antes de lanzar el florete completo (106k clips), arreglar la etapa 9 (luces) donde falla
por torneo, y el `cuadro_tocado == 0` de la etapa 11. Hipotesis de partida (ESTADO_2026-10-09 §5):
los torneos de florete SIN audio son otra fuente de video, con otro rotulo, y son los mismos donde
falla la luz.

---

## 0. Criterio y linea base, DECLARADOS ANTES DE MEDIR

Escrito antes de calcular ninguna cifra de este informe (ni la linea base).

### La verdad y las metricas

La verdad es la columna `touch` del manifiesto (`left` / `right` / `double`): quien se llevo el
punto. No es la verdad de la luz (en florete y sable dos luces se resuelven por prioridad), asi que
la metrica se construye donde SI lo es:

- **A1, acuerdo de una luz.** Clips cuyo grupo del tocado tiene **exactamente una luz de color**.
  Color -> tirador por la convencion del aparato (roja = izquierda, verde = derecha). Acierto = ese
  lado coincide con `touch`. Denominador: clips con `touch` en {left, right} y una sola luz de
  color. Con un detector perfecto esto es ~0,96 (medido con el detector viejo, 8.049 clips,
  `muestras/luces/README.md`): una sola luz que no gana es casi siempre una segunda luz perdida o
  una luz equivocada.
- **COB, cobertura**: clips con al menos una luz de color / clips con `touch` en {left, right, double}.
- **ACIERTO\***: clips de una luz que aciertan / TODOS los clips con `touch` en {left, right}. Es la
  cifra que no se puede inflar contestando menos.
- **C0**: fraccion de clips con luz cuyo `cuadro_tocado` (el minimo `cuadro_encendido` del grupo)
  es 0.

Todo se da **por `via`, por arma y por torneo**, con su denominador. Las cifras de antes y despues
se calculan **con el mismo codigo de medida sobre los mismos clips**, recalculando las luces con el
codigo viejo y con el nuevo (no se compara lo que haya en disco, que tiene versiones mezcladas).

### Reparto ajuste / prueba

- Un cambio GENERICO (que afecta a todas las sedes o a una familia de rotulo) se ajusta mirando
  clips de un conjunto de torneos de AJUSTE y se mide en torneos de PRUEBA que no se miraron.
- Un parametro POR SEDE (si hace falta) se ajusta con unos clips de esa sede y se mide en OTROS
  clips de la misma sede, y se dice asi: no es generalizacion entre sedes, es generalizacion entre
  clips.
- Los clips que se miren a ojo para ajustar quedan listados y fuera de la medida.

### Puerta de adopcion de un cambio en las luces (se decide con la PRUEBA, no con el ajuste)

Se adopta (variable de entorno encendida por defecto) solo si, en la prueba:

1. **A1 sube al menos +0,05** en los torneos a los que va dirigido, con IC95 por bootstrap
   (remuestreo por clip) que no cruce cero, **y**
2. **ACIERTO\*** no baja, **y**
3. en los torneos donde el codigo vigente ya funciona (A1 >= 0,85 con n >= 30) **ninguno pierde mas
   de 0,02** de A1 ni de COB, **y**
4. en espada y sable (la etapa 9 corre en las tres armas) A1 no baja mas de 0,02.

Si no pasa, el codigo queda puesto y **APAGADO** tras su variable de entorno, con su cifra al lado.

### Puerta de adopcion del arreglo de `cuadro_tocado == 0`

Se adopta si, en los mismos clips:

1. C0 baja a la quinta parte o menos, **y**
2. en los clips que cambian, A1 no baja (IC95 de la diferencia que no quede entero por debajo de
   cero), **y**
3. en los clips cuyo encendido NO era 0 la salida es identica (el arreglo no toca nada mas), **y**
4. a ojo, en 10 clips de los que cambian (tira de fotogramas), el cuadro nuevo es el encendido de
   la luz del tocado en al menos 8.

### La hipotesis de la fuente

Se da por **confirmada** si los torneos de florete sin audio difieren de los con audio en algo
medible de la FUENTE (resolucion, cadencia, familia o posicion del rotulo) **y** su A1 es menor con
IC95 (bootstrap por torneo) que no cruce cero. Si solo se cumple lo segundo, la conclusion es "los
torneos sin audio fallan mas" pero NO "por ser otra fuente".

### Adenda, escrita ANTES de ver el resultado del recalculo (misma tarde)

El diagnostico descriptivo de lo que hay en disco (§1) encontro que el detector VIEJO del proyecto
(`fencingly_v2.lights_generic.analyze_any`, el de la extraccion `localrun` de septiembre, conservado
en `D:\FENCINGLY_respaldo_2026-09-09`) parece mucho mejor que la etapa 9 en los clips de florete que
tienen las dos lecturas. Eso es una comparacion con versiones mezcladas, asi que **no decide nada**:
se recalculan los dos detectores sobre los mismos videos (`mide.py`) y se eligen reglas declaradas
aqui, antes de ver ese recalculo:

| regla | que hace |
|---|---|
| R0 | la etapa 9 vigente, tal cual (linea base) |
| R1 | el detector viejo solo, congelado: ni un parametro tocado |
| R2 | el viejo; si no da tocado, la etapa 9 vigente |
| R3 | la etapa 9 vigente sin las luces que ya estan encendidas en la primera muestra (`ini <= 0`) |
| R4 | el viejo; si no da tocado, R3 |

Dos reglas mas, anadidas tras mirar 12 errores de R2 en clips de AJUSTE (el detector generico del
viejo se equivoca en el rotulo ruso, donde la ROI de sede acierta) y ANTES de calcular ninguna tabla
con ellas:

| regla | que hace |
|---|---|
| R5 | si el viejo encuentra el rotulo FIE (`v2_fie`), el viejo; si no, la etapa 9 vigente; si esta tampoco da nada, el generico del viejo |
| R6 | R5 con R3 en lugar de la vigente |

Reparto: los torneos se barajan con la semilla 20261009 y se asignan alternando a AJUSTE y PRUEBA
(mitad de clips cada uno, aprox.). **La regla se elige en AJUSTE** por ACIERTO\* entre las que
cumplen las puertas, **y se confirma en PRUEBA** con las puertas de arriba. El detector viejo no se
ajusta en ningun sitio: lo unico que se elige es la regla de combinacion, y por eso basta una mitad.

### Segunda adenda (despues de medir R3, antes de medir R3b)

R3 NO pasa la puerta 3 del arreglo del cuadro 0 (cambia 10 de 521 clips cuyo encendido no era 0;
§3). Se declara una variante y su medida ANTES de calcularla:

- **R3b**: la etapa 9 vigente, y SOLO si su grupo del tocado empieza en el cuadro 0, el grupo se
  rehace sin las luces que ya estaban encendidas en la primera muestra (`FENCINGLY_LUZ_SIN_CUADRO0`).
  La puerta 3 se cumple por construccion (se comprueba igualmente en 50 clips).
- Se mide en TODOS los clips del recalculo cuyo R0 da cuadro 0, y se da aparte la cifra sin los 38
  de la muestra de 600 que ya se miraron para R3. Puertas 1, 2 y 4 del arreglo, sin cambios.
- En produccion la regla seria **R4b = R2 con R3b en el respaldo**.

---

## 1. Diagnostico de lo que hay en disco (descriptivo: versiones mezcladas, no decide nada)

`D:/FENCINGLY_SCRATCH/agente_luces/diag_disco.py` sobre los 5.837 `lights_*.csv` de los anotados:

| arma | clips | A1 (n) | COB | ACIERTO\* | C0 |
|---|---:|---|---:|---:|---:|
| espada | 1.036 | 0,931 (766) | 0,897 | 0,770 | 0,026 |
| florete | 4.236 | **0,808** (2.600) | 0,810 | **0,539** | **0,081** |
| sable | 565 | 0,880 (275) | 0,936 | 0,433 | 0,024 |

Florete por `via`: `roi_sede` A1 0,738 (1.082), `general_tras_roi` 0,859 (1.474), `general` 0,795
(44). **La ROI de sede es la via que peor acierta**, y la que mas `cuadro 0` da (0,102).

Florete por audio, a nivel de clip: sin audio A1 0,801 (2.220 clips de una luz), con audio 0,840
(275). La diferencia es pequenia y los torneos con audio son casi todos de 1 a 10 clips; Kazan 2020
(sin audio) da 0,935 y Anaheim 2018/2019 (sin audio) 0,875 / 0,880. **Lo que separa los torneos
buenos de los malos no parece ser el audio**; se mide en §3 con el recalculo.

(El 0,694 de ESTADO_2026-10-09 §4 usaba otra regla -- el lado de la primera luz con 3 cuadros de
tolerancia --; aqui A1 es la de §0.)

**El hallazgo.** Los NPZ de la extraccion vieja (`localrun`, 34.637 NPZ, 23.018 clips, en
`D:/FENCINGLY_respaldo_2026-09-09`) traen las luces de OTRO detector, el de `fencingly_v2`. En los
3.013 clips de florete que tienen las dos lecturas:

|  | A1 (n) | ACIERTO\* |
|---|---|---:|
| etapa 9 (disco) | 0,836 (2.088) | 0,579 |
| detector viejo (NPZ) | **0,966** (2.348) | **0,753** |

con el viejo por encima en 17 de los 20 torneos. Pero el viejo solo guardaba NPZ de los clips en
que encontraba tocado (su cobertura aqui es 1,0 por construccion) y las versiones no son las de hoy:
por eso se recalcula todo (§2-§3) antes de decidir.

---

## 2. El recalculo: los dos detectores sobre los mismos videos

`mide.py` corre, por clip, la etapa 9 vigente tal cual (R0) y el detector viejo congelado (R1) sobre
el mismo mp4. La lista son los 5.838 anotados con mp4 barajados con la semilla 20261009; con la CPU
compartida con la noche (~22 clips/min) se paro en los **2.016 primeros**, que son una muestra al
azar: 1.486 de florete, 347 de espada, 183 de sable. R3 se calculo en los clips donde hace falta
(los 823 en que el viejo no encuentra rotulo FIE) mas 600 al azar; R3b en los 123 con cuadro 0 mas
50 de control. Reparto: 60 torneos en AJUSTE (889 clips) y 60 en PRUEBA (1.127).

**Comprobado antes de medir que el codigo es el medido:** `luces_v2/` son los tres ficheros del
respaldo copiados byte a byte (sha256 en `luces_v2/__init__.py`); con `FENCINGLY_LUCES_V2=0` la etapa
9 refactorizada (`luces.lee_clip`) escribe CSV **identicos byte a byte** a los de disco (12 de 12) y
con `=1` sus filas coinciden con las de `mide.py` (12 de 12).

### Florete

| regla | AJUSTE A1 (n) | AJUSTE ACIERTO\* | PRUEBA A1 (n) | PRUEBA ACIERTO\* | PRUEBA COB | PRUEBA C0 |
|---|---|---:|---|---:|---:|---:|
| R0 vigente | 0,804 (373) | 0,570 | 0,801 (554) | 0,525 | 0,795 | 0,094 |
| R1 viejo solo | 0,898 (383) | 0,654 | **0,972** (605) | 0,696 | 0,926 | 0,000 |
| **R2 viejo, y si no la vigente** | 0,873 (410) | **0,681** | 0,947 (641) | **0,718** | **0,969** | 0,014 |
| R4 (R2 con R3 en el respaldo) | 0,875 (407) | 0,677 | 0,950 (635) | 0,714 | 0,961 | 0,000 |
| R5 viejo-FIE, si no la vigente | 0,853 (407) | 0,660 | 0,918 (648) | 0,704 | 0,963 | 0,030 |
| R6 (R5 con R3) | 0,864 (405) | 0,665 | 0,924 (644) | 0,704 | 0,957 | 0,000 |

### Las puertas, en AJUSTE (donde se elige)

| regla | 1. A1 florete >= +0,05, IC>0 | 2. ACIERTO\* no baja | 3. torneos buenos | 4. espada y sable | elegible |
|---|---|---|---|---|---|
| R1 | +0,094 [+0,053, +0,135] | +0,084 | **NO**: Kazan 2020 pierde 0,082 de COB | si | no |
| **R2** | +0,069 [+0,031, +0,106] | +0,110 | si | si | **SI** |
| R4 | +0,070 [+0,032, +0,108] | +0,106 | **NO**: Kazan COB -0,021 | si | no |
| R5 | **+0,048** [+0,027, +0,071] | +0,089 | si | si | no |
| R6 | +0,060 [+0,035, +0,088] | +0,095 | **NO**: Kazan COB -0,021 | si | no |

**R2 es la unica que pasa las cuatro en AJUSTE, y tambien la de mayor ACIERTO\* (0,681).** Se elige.

### Confirmacion en PRUEBA (R2 contra R0, 868 clips de florete)

| puerta | resultado | |
|---|---|---|
| 1. A1 florete | 0,801 (554) -> 0,947 (641), **+0,146 IC95 [+0,115, +0,176]** | PASA |
| 2. ACIERTO\* | 0,525 -> 0,718, **+0,193 IC95 [+0,158, +0,230]** | PASA |
| 3. torneos donde R0 ya funcionaba (A1 >= 0,85, n >= 30) | Anaheim 2018 0,857 -> 0,958 (COB +0,159); Anaheim 2019 0,862 -> 0,970 (COB +0,057) | PASA |
| 4. espada y sable | espada 0,902 (112) -> 0,944 (124); sable 0,864 (44) -> 0,930 (43) | PASA |

**VEREDICTO: R2 ADOPTADA.** `FENCINGLY_LUCES_V2` queda a `1` por defecto (`luces.USA_V2`).

### Por `via` (R2, los 2.016 clips; la `via` sale en cada fila de `lights_<clip>.csv`)

| arma | via | clips | A1 (n) | COB | C0 |
|---|---|---:|---|---:|---:|
| florete | `v2_fie` | 801 | **0,989** (619) | 1,000 | 0 |
| florete | `v2_generic` | 549 | 0,867 (369) | 1,000 | 0 |
| florete | `tras_v2_general_tras_roi` | 103 | 0,571 (42) | 0,485 | 0,291 |
| florete | `tras_v2_roi_sede` | 32 | 0,400 (20) | 0,812 | 0,312 |
| espada | `v2_fie` / `v2_generic` | 275 / 57 | 0,967 (240) / 0,911 (45) | 1,000 | 0 |
| sable | `v2_fie` / `v2_generic` | 117 / 57 | 1,000 (55) / 0,917 (24) | 1,000 | 0 |

El respaldo (la etapa 9 vieja, cuando el viejo no da tocado: 8 % de los clips) acierta poco: un
consumidor que quiera la luz fiable debe filtrar por `via`.

### Florete por torneo (R0 -> R2, A1; los de >= 20 clips en la muestra)

| torneo | clips | A1 R0 (n) | A1 R2 (n) | COB R0 -> R2 |
|---|---:|---|---|---|
| anaheimfoil2018 | 283 | 0,857 (182) | 0,958 (213) | 0,809 -> 0,968 |
| qatarfoil2021 | 205 | 0,746 (130) | 0,956 (158) | 0,683 -> 0,956 |
| stpetersburgfoil2019 | 155 | 0,682 (88) | 0,820 (89) | 0,839 -> 0,923 |
| kazanfoil2020 | 146 | 0,921 (76) | 0,921 (76) | 0,966 -> 0,966 |
| tokyoolympics2020 | 128 | 0,753 (77) | 0,890 (91) | 0,812 -> 0,961 |
| katowicefoil2019 | 126 | 0,797 (74) | 0,976 (82) | 0,825 -> 0,992 |
| **parisfoil2022** | 94 | 0,678 (59) | **0,676** (68) | 0,872 -> 0,904 |
| turinfoil2020 | 91 | 0,806 (62) | 0,971 (70) | 0,736 -> 0,967 |
| anaheimfoil2019 | 88 | 0,862 (65) | 0,970 (66) | 0,909 -> 0,966 |
| worldchamps2022 | 43 | 0,962 (26) | 0,943 (35) | 0,721 -> 0,977 |
| incheonfoil2022 | 36 | 1,000 (21) | 1,000 (29) | 0,694 -> 0,972 |
| eurochamps2022 | 23 | 0,812 (16) | 1,000 (16) | 0,826 -> 0,957 |

**Lo que queda flojo:** Paris 2022 (indicador UNICO central, 480p: el rotulo no dice el lado, solo el
color) y San Petersburgo 2019 (rotulo ruso, sin cajas FIE). Son 1.834 + 1.648 = 3.482 de los
106.432 clips de florete del manifiesto (3,3 %); no se tocan aqui.

---

## 3. El cuadro del tocado = 0

Medido en los 600 clips al azar (sin sesgo): con R0, **C0 = 0,068** (38 de 559 clips con luz). A
ojo, los dos tipos: la luz del tocado ANTERIOR encendida al empezar el clip (`PinkUnfitBedbug`: rojo
y verde encendidos en el cuadro 0, la roja del tocado se enciende en el 196) y una BANDERA del rotulo
tomada por luz (`DisgustingNippyHoneybee`: "roja derecha" de 24 px en el cuadro 0 = la bandera de
EE. UU.; la luz real es la verde del 243).

| arreglo | puerta 1 C0 | puerta 2 A1 en los que cambian | puerta 3 identico si no era 0 | puerta 4 a ojo | veredicto |
|---|---|---|---|---|---|
| R3, por pixel (`FENCINGLY_LUZ_ESCALON_VISTO`) | 0,068 -> 0 | 0,500 (28) -> 0,818 (22), +0,318 [+0,048, +0,558] | **NO: cambian 10 de 521** | - | **APAGADO** |
| R3b, por luz y solo si el grupo empieza en 0 (`FENCINGLY_LUZ_SIN_CUADRO0`) | 123 de 123 -> 0 | 0,474 (78) -> 0,712 (59), +0,238 [+0,085, +0,387]; sin los 38 ya vistos +0,202 [+0,031, +0,369] | 50 de 50 | **6 de 10** (exigia 8) | **APAGADO** |

Puerta 4 de R3b, clip a clip (tiras en `D:/FENCINGLY_SCRATCH/agente_luces/tiras/ojo_*.png`):
acierta el encendido en `HalfSoreDotterel`, `UnrulyKlutzyHalcyon`, `RecentKindlyLhasaapso`,
`SatisfiedCornyAdder`, `WideApprehensiveBats`, `InfatuatedYellowBaleenwhale` (este con el canal mal);
**no hay ninguna luz** en el cuadro que da en `ObedientPalatableBlackmamba`,
`ElderlyBlushingKingbird`, `JaggedEnchantingCoqui` y `CalculatingImmaterialGaur`. O sea: quitar la
luz del cuadro 0 deja a la vista la SIGUIENTE luz falsa del detector vigente en 4 de 10.

**Lo que si arregla el cuadro 0 es R2**: el viejo exige un escalon visto (apagado -> encendido) y
nunca da 0. Con R2 el C0 de florete pasa de **0,083 a 0,018** (0,094 -> 0,014 en PRUEBA), y lo que
queda esta todo en el respaldo (`tras_v2_*`, C0 0,29-0,31). Propuesta NO medida, para quien siga:
en la etapa 11, tratar un `cuadro_tocado = 0` como "sin luz" (None, no un cuadro inventado): en los
123 clips con cuadro 0 la luz acierta 0,474, que es azar.

---

## 4. La hipotesis de la fuente

| | torneos | clips | rotulo FIE encontrado (viejo) | 720p | 25 fps |
|---|---:|---:|---:|---:|---|
| florete SIN audio | 10 | 10.756 | **0,559** | 0,70 | 4 de 10 torneos |
| florete CON audio | 11 | 12.252 | **0,855** | 0,73 | 0 de 11 |

(21 sedes de florete, metadatos de los 23.018 clips de `localrun`, `fuente_viejo_por_torneo.csv`.)
Sin audio son TODOS los rotulos no FIE grandes salvo Tokio 2022: Kazan 2020 y San Petersburgo 2019
(rotulo ruso), Tokio 2020 (OBS olimpico) y Paris 2022 (indicador central, 480p); y los 25 fps solo
salen en torneos sin audio (Anaheim 2018, Bonn 2019, Katowice 2019, Turin 2020). **La fuente si es
distinta.**

Pero la segunda condicion NO se cumple. A1 de florete en el recalculo, sin audio contra con audio:

| detector | sin audio A1 (n, torneos) | con audio A1 (n, torneos) | diferencia, IC95 bootstrap POR TORNEO |
|---|---|---|---|
| R0 vigente | 0,792 (782, 10) | 0,870 (108, 23) | -0,079 [-0,194, +0,059] |
| R1 viejo | 0,943 (824) | 0,952 (125) | -0,009 [-0,081, +0,067] |

**VEREDICTO: NO CONFIRMADA.** Los torneos sin audio son otra fuente, pero no fallan
significativamente mas: lo que fallaba era la etapa 9 en todas partes (la ROI de sede es su peor
via, 0,738), y con el detector viejo la diferencia desaparece. Lo unico que queda flojo y es de
fuente son Paris 2022 y San Petersburgo 2019 (§2).

---

## 5. Que cambio en el codigo

| fichero | que |
|---|---|
| `01_extraccion/codigo/luces_v2/` (nuevo) | el detector viejo, 3 ficheros byte a byte + `__init__.py` con su origen y sha256 |
| `codigo/luces.py` | `lee_v2`, `lee_vigente`, `lee_clip`, `_grupo`; `USA_V2` (`FENCINGLY_LUCES_V2`, **defecto 1**); `ESCALON_VISTO` y `SIN_CUADRO0` (**defecto 0**, con su cifra) |
| `pipeline.py:_luces` | llama a `luces.lee_clip`; el CSV de luces trae `via` = `v2_fie` / `v2_generic` / `tras_v2_<via vieja>` y, del viejo, `cajas_v2` y `avisos_v2`; las columnas que el viejo no mide van vacias, no a cero |
| `codigo/zona_tocado.py:cuadro_tocado` | su copia de la etapa 9 sustituida por `luces.lee_clip` (una sola regla) |
| `01_extraccion/recalcula_luces.py` (nuevo) | rehace la 9 de lo ya extraido y la 11 solo si cambia el cuadro (si no, reescribe `via_luz`); probado en 3 clips |

## 6. Que hay que recalcular

- **Etapa 9, todos los clips ya extraidos** (5.838 anotados): `lotes.hechas` da la 9 por hecha si el
  CSV existe, asi que la noche NO la rehace sola. Coste: ~1 s/clip de CPU con la maquina libre
  (5-6 s con la CPU saturada como hoy): ~25 min con 4 procesos libres.
- **Etapa 11 donde cambia el cuadro**: en la muestra, el cuadro de R2 difiere del de R0 en el **63 %**
  de los clips (donde contestan los dos: igual 37 %, a <= 2 cuadros 71 %, a > 30 cuadros 18 %). Son
  ~3.500 clips a 12-20 s: **~4 h con 4 procesos**. Los que tenian cuadro 0 pasan de 0,2 s a ~20 s.
- Con el conjunto terminado o la noche parada, NO a la vez:
  `python 01_extraccion/recalcula_luces.py --conjunto anotados --procs 4`
- Despues, **volver a medir la zona** (la cifra adoptada 0,476 contra `middle` se midio con las luces
  viejas: el cuadro del tocado cambia en la mayoria de los clips).
- El florete nuevo no necesita nada: la noche usa el defecto (`FENCINGLY_LUCES_V2=1`).

Datos y scripts: `02_validacion/auditoria/luces_por_fuente_2026-10-09/` (por clip:
`evalua_clips.csv`; crudos: `med_*.jsonl.gz`).

# La prioridad en florete: del reglamento FIE a la señal medible

**Escrito el 2026-09-10.** El usuario lo pidió así: *«necesito que saques lo de la PRIORIDAD, para ver
si sacas cuando tiene cada uno la prioridad, para ponerlo. Y cuando la pierde. Y revisa la FIE para
ver cuando.»*

Este documento es el **mapa regla → señal**, y se escribió **antes** de una sola línea de detección.
Cada regla lleva tres cosas: el artículo real, lo que dice, y **con qué variable geométrica de
nuestras poses se decidiría**. Y lleva una cuarta que es la que importa: **si con lo que tenemos hoy
esa regla es medible o no**.

---

## 0. Primera corrección, y es al propio encargo: los artículos NO son t.56-t.60

El plan de la tarea decía *«artículos t.56 a t.60 aproximadamente, VERIFÍCALO en el material real»*.
Verificado, y **no es así**.

La fuente es el **Reglamento Técnico de la FIE, edición de diciembre de 2025**, en español, que está
en el disco de esta máquina en `C:\Users\Alejandro\Downloads\190675-tecnico esp.pdf`, 66 páginas.
No es una cita de memoria: el texto entrecomillado de este documento sale de ahí.

| lo que decía el plan | lo que dice el reglamento de verdad |
|---|---|
| t.56-t.60 = prioridad en florete | **t.56-t.58 son anulación de tocado por fallo del MATERIAL** (chaquetilla, hoja rota, aparato) |
| — | **t.59** es lo general de las tres armas: *«el árbitro analiza brevemente las acciones que componen la última frase de armas»* y remite a t.82ss |
| — | **t.60-t.63** es el **vídeo-arbitraje** |
| — | **t.76-t.89 = FLORETE, CONVENCIONES DEL COMBATE**, y dentro de ese bloque **t.82-t.89 es la prioridad** |

Y el propio índice del reglamento lo confirma en su entrada alfabética: *«Prioridad del tocado: t.59,
t.82ss, t.100ss, t.172.2»*. O sea **t.82 y siguientes** para florete y t.100ss para sable.

Las definiciones que las reglas de prioridad usan y sin las que no se puede medir nada están **antes**,
en el capítulo de términos técnicos: **t.8** tiempo de esgrima, **t.9** ataque/respuesta/contra-respuesta
y parada, **t.10** simple y compuesto, **t.11** respuesta, **t.12** contraataques, **t.13** reposición,
redoble, reanudación y contratiempo, **t.14** paradas, **t.15** punta en línea.

Que el bloque de artículos estuviera mal no es anecdótico: **t.56 habla de chaquetillas agujereadas**.
Buscar ahí la prioridad habría dado cero, y este proyecto ya tiene el precedente de los índices de la
mano dados por buenos sin comprobar (los puntos 91-95 que caían en la careta).

---

## 1. La definición base, que es la que fija TODAS las variables

**t.9.1**, definición de ataque:

> *«El ataque es la acción ofensiva inicial ejecutada **extendiendo el brazo** y amenazando
> continuamente la superficie válida del adversario, **precediendo el desencadenamiento del a fondo o
> de la flecha**.»*

Y **t.83.2.a**, que es la misma cosa dicha como criterio de arbitraje:

> *«El ataque simple, directo o indirecto, es ejecutado correctamente cuando **al extender el brazo,
> la punta que amenaza la superficie válida, precede el inicio del a fondo o de la flecha**.»*

Léase despacio, porque **ahí está todo el detector**. La regla no dice «tiene prioridad el que llega
primero» ni «el que avanza primero». Dice que hay **dos sucesos con orden**:

  suceso 1: **el brazo se extiende** amenazando el blanco
  suceso 2: **empieza el a fondo** (o la flecha)

y el ataque es correcto **si y solo si 1 va antes que 2**. Eso es un **signo de una resta de dos
instantes**, y una resta de dos instantes es exactamente lo que se puede medir sobre una serie de
poses. Es la razón de que la prioridad sea abordable con esqueletos y sin ver la hoja.

**t.83.2.d** cierra la definición por el otro lado:

> *«La acción simple o compuesta, la marcha o las fintas ejecutadas **con el brazo encogido**, no son
> contadas como un ataque sino como una **preparación**.»*

O sea que la variable que decide «ataque o preparación» es **la extensión del brazo armado**, y la que
decide «correcto o incorrecto» es **su orden temporal contra el fondo**.

---

## 2. El mapa regla → señal, artículo por artículo

Convenios de las variables, definidos una vez:

| variable | definición | unidad |
|---|---|---|
| `ext` | \|codo → raíz de la mano\| de la mano armada, dividido por \|hombro → cadera\| del mismo tirador | adimensional |

> **Corrección posterior, y se deja aquí para que se vea el error:** esta definición de `ext` es la
> que se declaró antes de medir y **es la equivocada**. `codo → raíz de la mano` es un hueso, no una
> extensión. La operativa es `ext_hr` = \|**hombro** → raíz de la mano\| / torso. Las cifras que lo
> demuestran están en §5.1, y la definición vieja se conserva como `ext_cr`, columna cruda de control.

| `dext/dt` | derivada de `ext` | por cuadro |
| `t_ext` | primer cuadro de un tramo con `dext/dt` sostenida por encima del umbral | cuadro |
| `pies` | \|tobillo izq → tobillo der\| dividido por el torso | adimensional |
| `t_fondo` | primer cuadro con `dpies/dt` sostenida por encima del umbral | cuadro |
| `vx` | velocidad del centro de caderas en anchos de caja por cuadro, con signo hacia el rival | 1/cuadro |
| `sep` | distancia entre los dos tiradores (centros de cadera) dividida por el torso medio | adimensional |
| `eje` | dirección `codo → raíz de la mano` de la mano armada | grados |
| `apunta` | coseno entre `eje` y la dirección al torso del rival | [-1, 1] |

**El eje del brazo es `codo → raíz de la mano` y no la muñeca**, porque en los 308 puntos de Sapiens 2
**no hay muñeca**: la muñeca *es* la raíz de la mano. Y no es el eje del pulgar, que está medido y no
sirve (2,4-4,8 px de palanca, salta 11-20° entre cuadros).

### 2.1 · Las reglas de concesión de prioridad

| art. | lo que dice, resumido | señal en vídeo que la delata | variables | ¿medible hoy? |
|---|---|---|---|---|
| **t.9.1 / t.83.2.a** | el ataque correcto es extender el brazo amenazando **antes** de iniciar el a fondo | subida sostenida de `ext` cuyo inicio precede a la subida de `pies` | `t_ext`, `t_fondo`, signo de `t_fondo − t_ext` | **SÍ** |
| **t.83.2.b** | el ataque **compuesto** es correcto si el brazo no se **encoge** durante las fintas | `ext` que sube y luego oscila **sin bajar** por debajo del nivel de la finta | mínimo local de `ext` entre `t_ext` y el final | **PARCIAL**: la oscilación se ve, distinguir finta de amague no |
| **t.83.2.c** | ataque por paso y a fondo: la extensión precede **el final de la marcha** | igual que t.83.2.a pero el ancla es el fin del paso, no su inicio | `t_ext` contra el último cuadro de `vx > 0` del paso | **PARCIAL** |
| **t.83.2.d** | brazo encogido = **preparación**, no ataque | `ext` por debajo del umbral mientras `vx > 0` | `ext`, `vx` | **SÍ** |
| **t.84.3** | el **passe-avant** es preparación y cualquier ataque simple tiene prioridad sobre él | avance sin extensión de uno, con extensión del otro | `vx`, `ext` de los dos | **SÍ** |
| **t.86** | *«La parada da derecho a la respuesta»*, y la respuesta anula lo posterior si es **inmediata, sin indecisión ni tiempo de detención** | contacto de hojas y luego inversión del sentido de la ofensiva | contacto de hoja **+** `t_ext` del que paró | **NO** (ver §3) |
| **t.87** | en ataque compuesto, si el adversario **encuentra el hierro** en una finta, tiene derecho a la respuesta | contacto de hojas durante la finta | contacto de hoja | **NO** |
| **t.84.1** | si el adversario está en **punta en línea**, el atacante debe **apartar el arma** antes | brazo del defensor extendido y apuntando **antes** de que arranque el fondo del atacante, y luego un desvío de la hoja | `ext`, `apunta`, contacto de hoja | **PARCIAL, y con un control FALLADO** (ver §3.2) |
| **t.84.2** | si al buscar el hierro **no lo encuentra** (librado), la prioridad **pasa al adversario** | dos hojas que se cruzan sin tocarse | hoja de los dos | **NO** |
| **t.85** | choque en el **tercio débil** conserva prioridad; en el **tercio fuerte** el ataque está mal hecho | punto de contacto a lo largo de la hoja | hoja con su punto de contacto | **NO** |
| **t.88** | en ataque compuesto el arresto vale si **precede el final del ataque en un tiempo de esgrima** | el contraataque toca antes del último movimiento del ataque | instante del tocado, `t_ext` de los dos, y el **tiempo de esgrima** en cuadros | **PARCIAL** |

### 2.2 · Las reglas de pérdida de prioridad, t.89.5

t.89.5 lista los casos en que **el atacante es el único tocado**, o sea en que **pierde** la prioridad
que tenía. Es la mitad de la pregunta del usuario («y cuándo la pierde») y merece su tabla:

| art. | el atacante pierde porque… | señal en vídeo | ¿medible hoy? |
|---|---|---|---|
| t.89.5.a | atacó **sin apartar** el hierro del que estaba en punta en línea | ausencia de desvío de la hoja adversa | **NO** (hace falta la hoja) |
| t.89.5.b | buscó el hierro, **no lo encontró** y siguió | dos hojas cruzándose sin contacto | **NO** |
| t.89.5.c | el adversario **encontró el hierro** en una finta y respondió | contacto en la finta | **NO** |
| t.89.5.d | tuvo un **momento de vacilación** y el otro arrestó | `dext/dt` que cae a cero, o `ext` que **baja**, en medio del ataque | **SÍ, y es la más valiosa de este bloque** |
| t.89.5.e | fue **arrestado con un tiempo de esgrima** antes del final | orden de instantes y el tiempo de esgrima | **PARCIAL** |
| t.89.5.f | tocó por **reposición/redoble/reanudación** sobre una parada seguida de respuesta inmediata | segunda subida de `ext` después de una bajada | **PARCIAL**: la segunda subida se ve, la parada no |

Y **t.89.4**, el espejo: el **atacado** es el único tocado si hace **arresto sobre un ataque simple**
(t.89.4.a), si **intenta esquivar sin conseguirlo** (t.89.4.b), o si **después de parar se detiene** y
da al otro el derecho a retomar (t.89.4.c). El primero es medible como *«contraataque = extender el
brazo cuando el otro ya tiene la prioridad»*; el tercero necesita la parada.

### 2.3 · La acción simultánea, t.89.1 y t.89.2

> *«La acción simultánea se debe a la concepción y a la acción simultánea de ataque de los dos
> esgrimistas; en este caso, los golpes dados **son anulados** para los dos.»*

Señal: **los dos `t_ext` a menos de un tiempo de esgrima**, y los dos con su fondo después. Esta es
medible y además es **la salida `ninguna` del detector**, que no es un fallo sino una respuesta
reglamentaria. Que exista un tercer valor y no dos es una consecuencia directa del reglamento.

El **tiempo de esgrima** está definido en **t.8**: *«la duración de ejecución de una acción simple»*.
El reglamento **no lo pone en segundos**, así que es un número que hay que **medir en nuestros datos**
y no copiar de ningún sitio. Se mide en §4.

---

## 3. Lo que NO es medible hoy, y por qué. Sin adornos

### 3.1 · Todo lo que necesita ver la hoja, que es la mitad del reglamento

De las 12 reglas de concesión, **cinco** (t.86, t.87, t.84.1, t.84.2, t.85) y **tres** de las seis de
pérdida piden **la hoja**: contacto, desvío, o el tercio donde se produce el choque. Y la hoja está
medida en este proyecto:

| | 1080p | 720p |
|---|---:|---:|
| recortes con recta pegada a la mano detectada en el alfa de `matting` | **42,6 %** | **1,6 %** |
| ancho del fuerte de la hoja | 0,96 px | **0,64 px** |

**A 720p la hoja no se detecta**, y 720p es la resolución de 7 de los 9 clips con poses de 308 puntos
de este proyecto. Así que hoy **la parada no se mide**, y sin la parada no se mide la respuesta, que
es el mecanismo por el que la prioridad **cambia de manos** en la mitad de las frases de florete.

Esto hay que decirlo en la cara: **un detector de prioridad sin hoja detecta el reparto INICIAL de la
prioridad y su pérdida por fallo propio, y no detecta el traspaso por parada.** Es una limitación de
alcance, no un error de implementación.

### 3.2 · La punta en línea, y un control que ya está FALLADO

**t.15**: *«La punta en línea es una posición particular en la cual el esgrimista mantiene el brazo
armado extendido y amenaza continuamente con la punta de su arma la superficie válida.»*

Parece medible con `ext` alto y `apunta` alto. Pero este proyecto **ya probó `apunta` como sustituto
de punta en línea y el control falló**, y la cifra está en `04_documentacion/informes/QUE_VAMOS_A_DETECTAR.md`:

| criterio declarado antes de mirar | resultado |
|---|---|
| el ángulo al rival debe ser **menor** en el tocado que en guardia | mediana pareada **−2,9°**, signo esperado en **17 de 36** clip×lado |

O sea **indistinguible del azar**. Y el porqué también está medido: en escorzo **codo y raíz de la mano
casi coinciden**, y ahí la dirección es ruido — IQR de 99,9° contra 53,9°.

Conclusión: `apunta` **no entra** como condición de punta en línea, y por tanto **t.84.1 y t.89.5.a no
se implementan**. Se deja `ext` alto **sin** avanzar como indicio de «brazo extendido en espera», que
es lo que se puede sostener, y se dice que no es punta en línea reglamentaria.

### 3.3 · Y una limitación que no es del reglamento sino de nuestros datos

`tipo` y `origen` marcan cómo se obtuvo cada dato (0 medido, 1 interpolado, 2 sostenido,
3 redetectado). **Una pose interpolada no puede sostener una decisión de prioridad**: el instante
`t_ext` es precisamente el sitio donde una interpolación lineal miente más, porque interpolar entre
dos anclas **borra el codo del movimiento**, que es la señal. Así que:

  **regla de uso declarada antes de medir:** un `t_ext` cuyo cuadro de cruce tenga `origen != 0` en
  el lado que lo produce **no concede prioridad**; se marca y baja la confianza.

---

## 4. Los umbrales, DECLARADOS ANTES DE MEDIR

Esto es lo que hay que fijar antes de ejecutar, para que no se pueda ajustar contra el resultado:

| decisión | umbral declarado | de dónde sale |
|---|---|---|
| «el brazo se está extendiendo» | `dext/dt` por encima del **p90** de la distribución de `dext/dt` de todos los cuadros, sostenido **2 cuadros** | el p90 se mide en §5; no se elige a mano |
| «el brazo está extendido» | `ext` por encima de su **p75** por tirador | idem |
| «empieza el fondo» | `dpies/dt` por encima de su **p90**, sostenido 2 cuadros | idem |
| «avanza» | `vx > 0` hacia el rival durante **3 cuadros** | 3 cuadros a 25-30 fps son 100-120 ms |
| **tiempo de esgrima** | mediana de la duración `t_ext → máximo de ext` medida en nuestros clips | t.8, medido, no copiado |
| «acción simultánea» | \|`t_ext` izq − `t_ext` der\| menor que **un tiempo de esgrima** | t.89.2 |
| «vacilación» (t.89.5.d) | `ext` que **baja** más de un 10 % de su recorrido después de `t_ext` | declarado; el 10 % se justifica contra el ruido de `ext` en §5 |

**Y la línea base tonta que hay que batir, declarada antes también:** *«tiene prioridad el que empieza
a avanzar primero»*, calculada **solo con la x del centro de la caja**, sin poses. Si el detector de
308 puntos no la bate, se dice.

---

## 5. Lo que salió al medirlo, con las cifras

Medido el 2026-09-10 sobre **21 clips de FLORETE con poses de 308 puntos** (9.961 cuadros × lado válidos) y
contra **21 ventanas anotadas a mano** de 10 clips. Detalle en `02_validacion/prioridad/`.

### 5.1 · Las variables: dos pasan y una es un hueso

| criterio declarado antes de medir | resultado | veredicto |
|---|---|---|
| `ext_hr` (hombro→mano) con recorrido p95/p05 ≥ **1,5** | mediana **2,21**, y 40 de 42 clip×lado pasan | **pasa** |
| `ext_cr` (codo→mano) con menos variación que `ext_hr` | cv 0,211 contra 0,241, y solo en **27 de 42** | **no pasa limpio** |
| al menos un tramo de subida por clip×lado | 234 tramos, **0 de 42** clip×lado sin ninguno | **pasa** |

**La variable que pedía el encargo, `codo → raíz de la mano` partida por el torso, NO mide la
extensión del brazo.** Es un hueso: su largo aparente varía un **21,1 %** solo por escorzo y ruido de
pose, y no crece cuando el brazo se extiende. La que sirve es **`hombro → raíz de la mano`**, y el
control que lo separa es el mismo largo en píxeles: el torso normaliza bien (cv 0,079) y la variación
está en el punto de la mano, no en el divisor.

### 5.2 · El tiempo de esgrima de t.8, medido

| p25 | **p50** | p75 | p90 | máx |
|---:|---:|---:|---:|---:|
| 3 | **5** | 8 | 10 | 22 |

**Cinco cuadros**, que a 30 fps son **167 ms**. Es la mediana del largo del tramo de subida de
`ext_hr`, o sea «la duración de ejecución de una acción simple» tal como la define t.8. De aquí sale
el umbral de simultaneidad de t.89.2 y el intervalo de gracia.

### 5.3 · La cota de alcance, mitad reglamento y mitad medida

| | |
|---|---:|
| hoja de florete de la FIE | 0,90 m |
| dos hojas, o sea el máximo en el momento del tocado | 1,80 m |
| distancia que un ataque **cierra**, p90 de 96 subidas de brazo | **1,48 m** |
| cota usada para conceder prioridad | **3,28 m** |

Y una equivocación mía que la medida delató: primero puse la cota en 1,80 m, que es correcta **para
el instante del tocado**, y la estaba aplicando **al instante del arranque**. El acierto en cuadros
con dueño se hundió de 25,6 % a **3,3 %**, porque al arrancar un ataque las manos están a 1,87 m de
mediana y solo el 47,9 % de las subidas empiezan por debajo de 1,80 m.

### 5.4 · El acierto, y no bate a la línea base tonta

| medida | detector | base «quien avanza primero» | trivial «siempre ninguna» |
|---|---:|---:|---:|
| 21 ventanas anotadas, por cuadro (n=462) | **69,7 %** | 34,6 % | **73,8 %** |
| solo cuadros CON dueño anotado (n=121) | 25,6 % | **41,3 %** | 0 % |
| solo cuadros SIN dueño anotado (n=341) | 85,3 % | 32,3 % | **100 %** |
| **cuando los dos nombran dueño, coincide el LADO** (n=36) | **86,1 %** | — | — |
| falsa alarma: inventa dueño donde no hay | 14,7 % | — | 0 % |
| instante del cambio, tolerancia ±3 cuadros | **2 de 10** | — | — |
| **2 ventanas de CONTROL, nunca vistas (n=44)** | **9,1 %** | — | 0 % |

Léase en este orden: **bate 2,0× a la línea base de cajas, empata con no contestar nunca, y en el
lote de control no contesta.** Cuando se moja acierta el lado en el 86,1 %, pero se moja en 36 de
121 cuadros. O sea **el lado es bueno y la cobertura es mala**.

### 5.5 · Los dos motivos de que no generalice, localizados

**Motivo A · `pies` no separa la marcha del a fondo.** La separación de tobillos crece en cualquier
paso, así que el «a fondo» sale antes que la extensión del brazo y t.83.2.a declara el ataque mal
iniciado:

| acción verificada a ojo | pico de `pies` |
|---|---:|
| a fondo de `GentleQuerulousElephantseal` | 3,41 |
| a fondo de `TestyWatchfulBarebirdbat` | 3,58 |
| a fondo de `FirsthandSoreGnat` | 3,22 |
| **flecha corriendo** de `AliveIgnorantBarebirdbat` | **2,54** |
| p90 agrupado, que es el umbral en uso | **2,31** |

El umbral cae **por debajo** de la banda del que corre. Y no se sube a 2,9 —que separaría los cuatro
casos— porque sería elegir el número mirando la anotación. Queda como **hueco declarado**.

**Motivo B · los umbrales absolutos no viajan entre clips.** `EXT_ALTO` es el p75 agrupado (1,091) y
hay tiradores que nunca lo pasan:

| clip × lado | p50 de `ext_hr` | máx | tramos aceptados |
|---|---:|---:|---:|
| `HarshExemplaryGermanspaniel` lado 1 | 0,82 | **1,00** | **0** en 301 cuadros |
| `UglyNiceAplomadofalcon` lado 0 | 0,95 | 1,20 | **0** en 315 cuadros |
| `VainAchingGonolek` lado 1 | 0,96 | 1,09 | **0** |

**7 de 42 clip × lado (16,7 %) no registran ni un solo arranque de brazo en todo el clip**, que es
físicamente imposible: todo floretista extiende el brazo. El culpable es el umbral agrupado contra el
encuadre y la constitución de cada tirador. El siguiente paso obvio es **normalizar por tirador**, con
el aviso ya escrito en este proyecto de que un normalizador por clip se contamina con lo que se
quiere detectar.

### 5.6 · La histéresis y el intervalo de gracia: uno sirve y el otro no

El usuario pidió el intervalo de gracia y tenía razón en el diagnóstico —la oscilación existía— pero
**la medida dice que lo que la quita es otra cosa**:

| variante | cambios/s | × los reales |
|---|---:|---:|
| **completo** | **0,55** | **0,8** |
| sin intervalo de gracia | 0,55 | 0,8 |
| sin histéresis (un solo umbral) | 0,38 | 0,6 |
| **sin filtro de amplitud y altura** | **0,84** | **1,3** |
| sin fusión de mesetas | 0,51 | 0,8 |
| sin nada de lo anterior | 0,89 | 1,4 |
| **anotado a mano** | **0,65** | 1,0 |

- **`PERMANENCIA` no cambia ni un cuadro.** Se queda en el código con esta tabla al lado, igual que
  `USA_CRUDAS = False` en `extrae_pareja.py`. La oscilación que iba a arreglar la producían los
  errores de definición, no la falta de margen.
- **Lo que de verdad frena la oscilación es el filtro de amplitud y altura**, que baja de 1,4× a 1,0×
  la tasa real de cambios.
- La tasa final, **0,53 contra 0,65 cambios por segundo anotados**, es la única medida en la que el
  detector se parece a la realidad.

### 5.7 · Las paradas: cero cifra de acierto, y por qué

| | |
|---|---:|
| candidatos del criterio de t.86 en 10 clips | **8** |
| **paradas anotadas a mano en las 21 ventanas** | **0** |
| acierto de detección de paradas | **no se puede calcular** |

En 15,4 s de acción de florete anotada no vi **ni una** parada-respuesta limpia: lo que hay son
ataques simples, contraataques y acciones simultáneas. Con n = 0 no hay precisión ni cobertura que
dar, y decir «detecta paradas» sería inventarlo.

**Y lo que haría falta para medirlo, concreto:**

1. **Anotar paradas a propósito**, seleccionando ventanas por una señal específica de parada en vez
   de por actividad o por acercamiento. Las 8 candidatas de `paradas_candidatas.csv` son el punto de
   partida, y revisarlas a ojo daría una **precisión** (no una cobertura).
2. **Clips a 1080p.** El contacto de hojas es la definición de parada y a 720p la hoja se detecta en
   el **1,6 %** de los recortes. A 1080p sube al **42,6 %**. Sin eso, «parada» seguirá siendo un
   patrón inferido y no un contacto visto.
3. **No prolongar la hoja desde la mano.** Está prohibido por escrito en este proyecto y además
   medido: el ángulo entre la hoja real y el eje del antebrazo pide 24,8-29,7° de semiapertura, o sea
   2,4-2,8× el techo de utilidad. Una hoja dibujada sobre el eje del antebrazo apunta a donde no está.

---

## 6. Lo que este documento NO dice

- **No dice que el detector funcione. Medido, no funciona todavía**: empata con «no contestar nunca»
  en las ventanas que usé para arreglarlo y no contesta en las de control. Las cifras están en §5.4.
- **La anotación la hizo un agente mirando láminas, no un árbitro FIE**, y ya se demostró equivocada
  una vez: en `GentleQuerulousElephantseal` había escrito «derecha» donde el zoom por tirador enseña
  al de la izquierda extendiendo primero. La corrigió el propio detector al discrepar.
- **Un clip del lote de poses era SABLE** (`MellowHideousAngora`, cartel «SABRE FEMININ» al fondo),
  colado por el muestreo viejo que el documento maestro ya declaró mal en su §14.6. Se excluyó. La
  prioridad del sable es t.100 y siguientes, no t.82.
- **No dice que la prioridad se pueda arbitrar con esto.** Cinco de las doce reglas de concesión
  necesitan la hoja y no están implementadas. Un clip cuya frase se decide por una parada **está
  fuera del alcance** de lo medible hoy, y el detector debe decir `ninguna` con razón declarada en vez
  de inventar un ganador.
- **No dice que los umbrales del §4 sean los correctos.** Son percentiles de nuestra propia
  distribución, o sea que se mueven con el corpus. Lo que sí garantiza es que **no se eligieron
  mirando el resultado**.
- **No traduce el reglamento entero.** Se han leído y citado t.7-t.15 y t.76-t.89. Lo que hay entre
  medias (superficie válida t.77-t.79, anulaciones t.80-t.81) no afecta a quién tiene la prioridad,
  solo a si el tocado cuenta.

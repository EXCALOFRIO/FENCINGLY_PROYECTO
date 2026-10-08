# El catalogo de clips y la anotacion humana

Los dos ficheros publicos de `clips.fencingdatabase.com` de los que sale todo lo demas de esta
tanda. **No se leen directamente desde ningun sitio**: el unico lector es
`01_extraccion/codigo/catalogo.py`, y lo que consume el resto del proyecto es `manifiesto.csv.gz`.

## Que hay aqui

| fichero | bytes | que es |
|---|---:|---|
| `gfycats.txt.gz` | 9.398.476 | el catalogo. 362.950 clips, 172 torneos. Descomprimido son 103.444.712 B |
| `form_responses.txt` | 1.133.242 | la anotacion humana. 8.513 filas, 6.829 clips distintos |
| `manifiesto.csv.gz` | 13.565.739 | el derivado: una fila por clip, la union de los dos |

### La instantanea, apuntada

Un fichero servido por un CDN cambia sin avisar, y una cifra calculada sobre "el catalogo" sin
decir **cual** es una cifra sin denominador. Lo que se descargo, medido el 2026-09-22:

```
https://clips.fencingdatabase.com/gfycats.txt
  Content-Length  103444712
  ETag            "ad2e66d0735e81d9e9ed1639afc231bb"
  Last-Modified   Fri, 11 Sep 2026 16:05:48 GMT

https://clips.fencingdatabase.com/form_responses.txt
  Content-Length  1133242
  ETag            "cab79689db8c2c909e68d763e2f5afb4"
  Last-Modified   Fri, 11 Sep 2026 11:23:16 GMT
```

### Por que el catalogo va comprimido

103 MB pasa del **limite duro de 100 MB por fichero** de GitHub, y este repo se espeja alli. A
`gzip -9` son 9,0 MB, o sea que cabe once veces, y `catalogo.py` lo lee comprimido sin
descomprimirlo a disco. El crudo no se guarda: se reconstruye con `gunzip` y su hash tiene que dar
el ETag de arriba.

## Como se reconstruye

```
python 01_extraccion/codigo/catalogo.py --construye   # escribe manifiesto.csv.gz
python 01_extraccion/codigo/catalogo.py               # imprime los recuentos
python 02_validacion/herramienta/verifica_catalogo.py # el careo, con su tabla
```

`--construye` mira `D:/FENCINGLY_VIDEOS_RAW` (o `FENCINGLY_VIDEOS`) para rellenar `en_disco`, asi
que reconstruirlo en otra maquina cambia esa columna y **solo esa**.

## Las cuatro cosas que la medida encontro y el plan no tenia

**1. `weapon` no siempre trae un arma.** 13.260 clips traen `champs` (la familia del torneo) y uno
lo trae vacio. Los 106.432 de florete son una **cota inferior**: hay 13.261 clips mas de arma
desconocida y algunos seran florete. Se guardan tal cual, sin adivinar.

**2. 8.513 filas son 6.829 clips.** 1.184 clips estan anotados mas de una vez, uno **35 veces**.
Hay que resolverlos, y el convenio cambia la cifra:

| convenio | `initiated` | `strip_location` | `body_location` |
|---|---:|---:|---:|
| la ultima fila gana | 5.628 | 5.635 | 1.698 |
| cualquiera no vacia | 5.803 | 5.800 | 1.734 |
| **mayoria, empates fuera** (el elegido) | **5.596** | **5.544** | **1.699** |

Las cifras del plan de esta tanda (5.628 / 5.635 / 1.698) son el primer convenio, que **nadie
eligio**: es lo que sale de sobrescribir un dict mientras se lee. No estan mal, estaban sin
declarar.

**3. El techo del etiquetador, que ahora se puede medir.** Sobre los clips con dos o mas
respuestas independientes, la fraccion en que **todas** coinciden:

| cabeza | clips con >=2 respuestas | en desacuerdo | acuerdo |
|---|---:|---:|---:|
| `initiated` | 911 | 284 | **0,688** |
| `strip_location` | 908 | 346 | **0,619** |
| `body_location` | 108 | 56 | **0,481** |

Esto es lo mas valioso que trae el fichero repetido. El plan declaraba la linea base por abajo
(0,51, la clase mayoritaria) y no tenia nada por arriba. Ahora **un 0,69 en `initiated` es igualar
a los humanos**, y un 0,80 hay que mirarlo con desconfianza antes que con alegria: estaria
acertando mas que la verdad de referencia.

Por eso el manifiesto lleva dos columnas que el plan no pedia, `n_anot` y `acuerdo`.

**4. "Anotado" y "con etiqueta" no son lo mismo.** De los 6.748 clips cuya fila de anotacion casa
con el catalogo, **976 tienen las tres etiquetas vacias**: el anotador solo respondio las preguntas
extra (`key-moment`, `attack-in-prep`, `epee-action-type`...). Los clips con al menos una de las
tres son **5.856**.

## "Etiquetado" no es una cosa: son capas, y de tamanios muy distintos

La pregunta *"solo hay 5.856 clips etiquetados?"* tiene truco, porque depende de **que etiqueta**.
Sobre el florete (106.432 clips):

| etiqueta | florete | de donde | hace falta modelo? |
|---|---:|---|---|
| **`touch`** -- que lado toca | **106.432** | del catalogo | **no, es gratis** |
| marcador, nombres, torneo, `event_type` | 106.432 | del catalogo | no |
| **mano armada** | **57.025** (53,6 %) | ficha de la FIE | no, se consulta |
| `initiated` -- quien inicia | **4.069** | anotacion humana | si |
| `strip_location` -- zona de pista | **3.964** | anotacion humana | si |
| `body_location` -- blanco tocado | **255** | anotacion humana | si |

**Y el punto que se malinterpreta facil:** los 5.856 no son lo que necesita el modelo final. Son
lo que necesita el **ETIQUETADOR**. Ese es el diseno entero: se entrena con los 5.856 anotados a
mano y se aplica a los 106.432, y **eso** -- 106.432 clips con las tres etiquetas predichas mas
`touch` que es verdad -- es lo que come el modelo de videoarbitraje. Por eso la etapa 10 no cuesta
GPU: etiquetar los 106.432 es gratis comparado con extraerlos.

### De donde sale el 5.856, paso a paso

    8.513 filas  ->  6.828 clips distintos  ->  6.748 casan con el catalogo
                                                 5.931  con alguna de las tres etiquetas
                                                    10  solo con preguntas extra
                                                   807  filas COMPLETAMENTE VACIAS
    5.931 - 75 que pierden la etiqueta por EMPATE entre anotadores = 5.856

Los 807 vacios y los 80 que no casan con el catalogo **no son un fallo de lectura**: son filas que
alguien abrio y no relleno, y clips que ya no estan en el catalogo vigente.

### `additional_questions`: hay mas campos, pero son pequenios

El formulario nuevo trae un JSON con preguntas extra. Relleno en **428 filas de 8.513**:

| campo | clips | valores |
|---|---:|---|
| `attack-in-prep` | 281 | no 228, yes 92 |
| `epee-action-type` | 120 | counterattack, lunge, fleche, parry-riposte, remise |
| `epee-attack-tempo` | 114 | one, two, three |
| `epee-pull-push` | 114 | push, pull |
| `key-moment` | 113 | distance-collapse, punishment, instinctual-folly, temps-perdu |

`initiated-action` y `strip-location` del JSON duplican las columnas principales, asi que no anaden
nada. De los demas, **ninguno llega a 300 clips**: son para anotar mas, no para entrenar hoy. El
mas interesante es `attack-in-prep`, porque "ataque en preparacion" es un concepto de **prioridad**,
que es el objetivo del proyecto.

### `body_location` y las armas: mezclarlas es INCORRECTO, y el reglamento dice por que

| arma | n | reparto |
|---|---:|---|
| **florete** | 255 | torso 221 (**87 %**), hand 29, resto 5 |
| **espada** | 917 | torso 442, front_arm 241, hand 108, front_leg 51, foot 38, head 26 |
| **sable** | 527 | torso 200, front_arm 163, hand 80, head 74, piernas ~ 0 |

El reparto lo fija **el arma**: en florete el blanco valido es solo el torso (el peto metalico), en
espada vale todo el cuerpo, y en sable de la cintura para arriba. Son **tres distribuciones
distintas**, y mezclarlas sin condicionar haria predecir "torso" para un tocado en el pie de
espada. Lo que si se comparte es la **geometria**, asi que la forma correcta es tronco comun y
salida condicionada al arma -- adoptandolo solo si no baja en el mismo reservado de florete.

**Y el limite de verdad, que no lo arregla mas extraccion:** en florete la clase mayoritaria es
**0,867**. Con 255 clips el reservado son ~50, y batir 0,867 ahi es indistinguible del ruido. En
florete esta cabeza no tiene un problema de datos: **la respuesta es "torso" por reglamento casi
siempre**. La pregunta util en florete no es *donde* toco sino *si fue en valido*, y eso es la luz
blanca -- la etapa 9, no esta cabeza.

## Los denominadores de "ya esta en disco", que son tres y dan tres numeros

Medido el 2026-09-22 sobre `D:/FENCINGLY_VIDEOS_RAW` (45.008 mp4, **42.326** nombres distintos):

| conjunto | n |
|---|---:|
| clips anotados (fila que casa) que estan en el corpus de 8.555 ya procesado | **1.370** |
| clips con etiqueta que estan en ese corpus | 1.142 |
| clips anotados (fila que casa) con su mp4 en D: | 4.366 |
| **clips con etiqueta con su mp4 en D:** | **3.654** |

El plan dice "1.370 ya estan en disco" y cita el primero. Es correcto, pero **es el corpus ya
procesado, no el disco**: en D: hay 2,7 veces mas clips anotados de los que el plan supone. Lo que
cambia de verdad:

- la puerta [C] se puede correr con **3.654** clips en vez de 1.370;
- quedan por bajar **2.202**, no 5.378, o sea unos **7,7 GB** a 3,5 MB/clip, no 19 GB.

**Y la trampa que hay debajo:** "el mp4 esta en D:" no es "el clip esta extraido". De los 5.856 con
etiqueta, los que tienen `poses_<clip>.npz` hoy son **3**. El coste de la puerta [C] no es bajar
video, es **extraer**, y eso es lo que hay que medir antes de prometer una hora de reloj.

## El fichero que esto sustituye

`corpus_etiquetado.txt` (8.559 ids) vivia **fuera del repo**, en el scratchpad temporal de otra
sesion, y unos diez ficheros rederivaban esa ruta cada uno por su cuenta. Un scratchpad temporal se
borra. Comprobado contra el manifiesto: **8.555 de los 8.559 estan, y los 8.555 son `foil`**. Los
cuatro que faltan del catalogo son `PerfectFavorableGalapagoshawk`,
`UnfitPracticalAmericanpainthorse`, `MilkyTediousCaimanlizard` y
`DizzyClosedGossamerwingedbutterfly`.

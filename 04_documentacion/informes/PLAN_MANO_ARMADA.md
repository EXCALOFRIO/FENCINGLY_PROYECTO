# Plan: la mano armada, que falla a menudo y ahora se puede arreglar de verdad

El usuario lo dijo directo: *«hay muchas veces que lo detectamos mal y se podría hacer más
accurate»*. Tiene razon, y lo primero es entender **por que** falla, porque la causa no es la que
parece.

---

## 0 · El diagnostico: el limite no era el criterio, era el denominador

Toda la cascada de la mano armada se eligio y se midio contra **59 tiradores anotados a ojo** por
un agente mirando mosaicos, en 33 clips. Con 59 casos:

- la diferencia entre 0,90 y 0,95 son **tres tiradores**;
- una variante (`nariz_dir`) se descarto porque los 2 tiradores que anadia estaban **los dos en
  entrenamiento** y cero en el reservado;
- y no habia forma de saber si un criterio fallaba en una familia de casos o por azar.

Con ese tamano de muestra no se puede mejorar nada: cualquier cambio queda dentro del ruido.

## 1 · Lo que ya esta hecho: verdad de referencia 16 veces mayor, sin anotar nada

`01_extraccion/codigo/fie.py` + `03_datos/catalogo/mano_fie.csv`. **Medido, no supuesto:**

| | |
|---|---:|
| censo de la FIE | 66.544 tiradores, HTTP 200 |
| `hand` en el listado | **no esta** — hay que pedir el perfil, uno por tirador |
| `hand` en el perfil | **967 de 967** de los nuestros (0 sin dato) |
| `height` en el perfil | 7 de 60 — para la estatura esto **no sirve** |
| nombres del conjunto anotado que resuelven sin ambiguedad | **967 de 1.100 (88,2 %)** |
| clips con **los dos** tiradores resueltos | **4.272 de 5.856 (73,0 %)** |
| lo mismo sobre el florete entero | 649 de 826 tiradores, 80.921 de 106.432 clips (76,0 %) |

**El control de que el dato es real:** el **25,7 %** son zurdos. En la poblacion general son ~10 %;
en esgrima de elite lo esperable es 25-35 %, porque el zurdo tiene ventaja tactica. Si el campo
viniera relleno por descuido saldria 0 %, y si viniera al azar saldria 50 %. Cae justo donde tiene
que caer.

## 2 · La consecuencia inmediata: una linea base que no existia, y que cambia el veredicto

Si tres de cada cuatro tiradores son diestros, **contestar siempre «la diestra» acierta 0,743** sin
mirar la imagen. Nadie tenia esa cifra porque nadie tenia datos de lateralidad. Con ella al lado:

| criterio | acierto | contra el azar informado (0,743) |
|---|---:|---|
| **contestar siempre la diestra** | **0,743** | — |
| peldano geometrico solo, sobre 27 | 0,741 | **por debajo** |
| peldano geometrico solo, sobre 59 | 0,831 | +8,8 pp |
| la cascada entera, sobre 59 | 0,949 | +20,6 pp |

**El peldano geometrico suelto no bate al azar informado.** Eso no estaba dicho en ningun sitio, y
es exactamente el tipo de cifra que hace falta antes de tocar nada.

## 3 · Las cuatro mejoras, por relacion valor/coste

### 3.0 · HECHO Y MEDIDO: el detector tiene un sesgo sistematico hacia "zurda"

Lo primero que la ficha de la FIE enseña no es cuanto acierta el detector, sino **como se
equivoca**. Matriz por clip sobre 329 decisiones con ficha:

| ficha FIE | dice A (diestra) | dice B (zurda) | n | acierto |
|---|---:|---:|---:|---:|
| **diestra (A)** | 221 | **33** | 254 | 0,870 |
| **zurda (B)** | **5** | 70 | 75 | 0,933 |

**El modelo dice "zurda" en el 31,3 % de los casos cuando la realidad es el 22,8 %.** Y los errores
son **6,6 veces mas a menudo falso-zurdo (33) que falso-diestro (5)**. Eso no es ruido: es un
sesgo, y un sesgo se corrige. Por peldano:

| peldano | n | acierto | dice "zurda" |
|---|---:|---:|---:|
| `nariz_dir_dudosa` | 170 | 0,853 | 0,294 |
| `pie_firme` | 159 | 0,918 | 0,333 |

**Lo que NO se hace con esto, todavia:** mover un umbral para que la tasa cuadre con el 22,8 %.
Eso seria ajustar contra la propia medida. La correccion correcta usa el margen que la cascada ya
calcula (`frac_pie`, `frac_nariz`) y se valida en un **reservado por tirador**, no por clip --
dos clips del mismo tirador no son independientes. Queda identificado, con su cifra, y no hecho.

### 3.1 · HECHO Y MEDIDO: decidir por TIRADOR y no por clip da +3,0 pp

Implementado en `01_extraccion/codigo/mano_corpus.py`. Medido contra la ficha:

| decidir | acierto | n |
|---|---:|---:|
| por clip | 0,887 | 327 decisiones |
| **por tirador (mayoria)** | **0,917** | 168 tiradores |

Y la curva es la que la teoria predice, con una excepcion que importa:

| clips del tirador | acierto | n tiradores |
|---:|---:|---:|
| 1 | 0,889 | 99 |
| 2 | 0,939 | 33 |
| 3 | **1,000** | 17 |
| 4 | **1,000** | 8 |
| **>= 5** | **0,909** | 11 |

Los tres primeros escalones son la convergencia esperada de una mayoria. **El escalon de >= 5 la
rompe**, y es justo la no-independencia que habia que vigilar: un tirador que falla teniendo muchos
clips falla por una razon **sistematica** -- su sede, su forma de ponerse de perfil -- y votar no
lo arregla, lo consagra. Con n=11 tambien es ruidoso. La cifra va publicada tal cual en vez de
recortada al tramo que sale bien.

Nota de denominador: hoy **99 de 168 tiradores tienen un solo clip extraido**, o sea que la mayoria
todavia no puede votar. Segun avance la extraccion esa distribucion se mueve y la ganancia deberia
crecer; hay que volver a medirlo entonces y no extrapolar.

### 3.1b · Decidir por TIRADOR: por que era la mas barata

**La mano armada es una propiedad de la persona, no del clip.** Un tirador no cambia de mano entre
un asalto y el siguiente. Hoy se decide clip a clip y cada clip vuelve a empezar de cero.

El catalogo da los dos nombres de los 362.950 clips, asi que los clips se pueden **agrupar por
tirador** y decidir por mayoria de todos sus clips. Si el acierto por clip es `p > 0,5`, la mayoria
sobre `n` clips converge a 1 al crecer `n` — y en el corpus anotado la mediana de clips por tirador
es de varios. Es ganancia **por construccion**, sin modelo nuevo y sin GPU.

Criterio declarado antes de medir: se adopta si el acierto por tirador sube respecto al acierto por
clip **y** la cobertura no baja.

### 3.2 · No detectar lo que se puede consultar

En el **73,0 %** de los clips anotados los dos tiradores resuelven contra la FIE. Para esos, la
mano armada **no hay que detectarla: se mira**. El detector geometrico solo tiene que funcionar en
el 27 % restante.

Esto no es hacer trampa, es lo que hace cualquier sistema serio: la ficha federativa es un dato
mejor que una inferencia de pixeles. Lo que hay que respetar es la disciplina que el proyecto ya
tiene para las etiquetas: la columna **`origen_mano`** con `fie` o `modelo`, **nunca fusionadas**,
por la misma razon que `origen_etiqueta` — el dia que se entrene el modelo final hay que poder
pesarlas distinto.

Y el limite, escrito: la ficha dice la mano **de la licencia**, no la del clip, y **no valida quien
es quien**. Si la tuberia intercambiase los dos tiradores, la ficha confirmaria la asignacion
equivocada. Por eso el paso 3.3 va antes.

### 3.3 · Calibrar el sentido A/B <-> R/L, y de paso auditar los intercambios de tirador

La FIE habla de manos anatomicas (`R`/`L`); el proyecto habla de indices del modelo de pose
(`A` = raiz 21, del hombro 6; `B` = raiz 42, del hombro 5). Por el convenio de paridad *deberia*
ser `A` = diestra, pero **eso es una hipotesis y se mide**: se prueban los dos sentidos contra los
59 anotados a ojo y gana el que mas coincide. Si los dos dieran parecido, el mapeo no esta
determinado y la ficha no se puede usar — y eso tambien seria un resultado.

**Y el subproducto vale mas que la calibracion:** donde el ojo y la ficha discrepen hay que mirar
el clip, porque o la anotacion a ojo se equivoco o **la tuberia intercambio los dos tiradores**, que
es un fallo bastante mas grave que confundir una mano. Es el primer detector de ese fallo que el
proyecto tendria.

`02_validacion/herramienta/verifica_mano_fie.py` hace los tres pasos y se para si el margen del
mapeo es menor de 0,15.

### 3.4 · HECHO: se reabrieron los criterios retirados, y **ninguno mejora**

`02_validacion/herramienta/verifica_mano_criterios.py`. Todos los criterios que `cono_lib` ya
calcula, medidos contra la ficha, **por tirador** (que es la cifra que manda) y con el `acierto*`
que cuenta los mudos como fallo:

| criterio | acierto | cobertura | acierto* | n |
|---|---:|---:|---:|---:|
| `a_voto` (voto por cuadro) | 0,929 | 0,966 | **0,897** | 253 |
| `a_dist2d` (euclidea) | 0,929 | 0,966 | **0,897** | 253 |
| **la cascada (lo vigente)** | 0,925 | 0,962 | 0,889 | 252 |
| `mayoria_3` | 0,921 | 0,966 | 0,889 | 253 |
| `nariz` | 0,908 | 0,958 | 0,870 | 251 |
| `geometria a` | 0,903 | 0,947 | 0,855 | 248 |
| `hombro` | 0,907 | 0,943 | 0,855 | 247 |
| `pie` | 0,921 | 0,916 | 0,844 | 240 |
| azar informado (siempre la diestra) | 0,743 | 1,000 | 0,743 | — |

A primera vista `a_voto` y `a_dist2d` baten a la cascada por **+0,8 pp**, que es justo lo que se
retiro en su dia como *"+2 casos, dentro del ruido"*. Pero restar dos porcentajes no contesta la
pregunta: **+0,8 pp sobre 253 tiradores son dos personas.** Lo que la contesta es el careo
**pareado**, sobre los mismos tiradores:

| candidato contra la cascada | gana | pierde | discrepan | p (McNemar) |
|---|---:|---:|---:|---:|
| `a_voto` | 3 | 3 | 6 | **1,000** |
| `a_dist2d` | 3 | 3 | 6 | **1,000** |
| `mayoria_3` | 1 | 2 | 3 | 1,000 |
| `nariz` | **0** | **4** | 4 | 0,125 |
| `geometria a` | 9 | 15 | 24 | 0,307 |

**El +0,8 pp eran tres tiradores contra tres.** Es ruido, y la decision original de no adoptar
`a_voto` era correcta -- ahora por una medida con n=253 en vez de por una corazonada con n=27. Y
`nariz` no gana **ni uno** donde la cascada falla, asi que `USA_NARIZ_SI_PIE_FLOJO = False` tambien
queda confirmado.

**Lo util de este ejercicio no es el ranking, es lo que ensenia el conjunto:** los ocho criterios
caben en **5 pp** (0,844 a 0,897) y todos baten al azar informado por 10-15 pp. O sea que la
estructura de peldanos de la cascada **no se esta pagando a si misma**: un voto por cuadro a secas
hace lo mismo. Si alguna vez hay que simplificar, ahi esta la medida que lo autoriza.

### 3.4b · Lo que ya no hace falta volver a medir

Con 967 tiradores se pueden reabrir las decisiones que se tomaron con muestras de dos digitos, y
varias se tomaron asi:

| criterio | lo que se midio entonces | n |
|---|---|---:|
| `a_voto` (voto por cuadro) | 0,815, «+2 casos, dentro del ruido» | 27 |
| `a_dist2d` (distancia euclidea) | declarado, sin cifra firme | 27 |
| `nariz_dir` | 0,932 sobre 59, retirado por no ganar en el reservado | 59 |
| guante blanco / cazoleta oscura | 0,444 / 0,500, retirados | 27 |

«Dentro del ruido» con n=27 y «dentro del ruido» con n=967 son frases distintas. Alguno de estos
puede estar bien y haberse retirado por falta de muestra.

Y con esa cantidad de datos ya tiene sentido **un clasificador pequenio en vez de una cascada a
mano**, alimentado con las salidas de los peldanos mas la geometria de la pose. Pero **primero la
linea base**: no se adopta si no bate a la cascada actual medida sobre el mismo reservado — y el
reservado tiene que ser **por tirador**, no por clip, porque dos clips del mismo tirador no son
independientes. Es el mismo error que el reparto por torneo del etiquetador.

---

## 4 · El orden, y la puerta

1. **Calibrar el mapeo** (3.3). Si el margen es menor de 0,15, **parar**: la ficha no sirve y hay
   que averiguar por que antes de seguir.
2. **Medir la cascada actual contra los 967**, peldano a peldano, con acierto y **cobertura por
   separado**, y con el 0,743 al lado. Esta es la foto real, y hoy no existe.
3. **Decidir por tirador** (3.1) y volver a medir. Es lo mas barato.
4. **Consultar la FIE donde se pueda** (3.2), con `origen_mano` y sin fusionar.
5. **Reabrir los criterios retirados** (3.4) y, solo entonces, plantear el clasificador.

**La puerta, declarada ahora:** el paso 2 tiene que enseniar que la cascada bate el **0,743** con
un margen mayor que su error de muestreo sobre los 967. Si no lo bate, el problema no es afinar un
peldano: es que la geometria de la pose no separa las dos manos en este material, y entonces lo
correcto es consultar la ficha donde exista y **marcar el resto como `incierta`** en vez de
adivinar.

---

## 5 · Lo que esto NO arregla, dicho por delante

- **La ficha no observa el clip.** Si un tirador cambiara de mano por lesion, o si la licencia
  estuviera mal, esto lo repetiria sin enterarse.
- **El 27 % de clips sin resolver no se beneficia.** Los nombres que no casan son sobre todo
  nombres compuestos escritos distinto (`CHAN Nok` contra `CHAN Nok Hin`) y tiradores que no estan
  en el censo vigente. La regla laxa de `fie.py` recupera 16 de 1.100; el resto habria que casarlo
  por otra via.
- **Y no valida la asignacion izquierda/derecha**, solo la mano dentro de una persona ya asignada.

# Registro ENTRE clips del mismo torneo ("panorama del recinto"): protocolo DECLARADO ANTES de medir

Escrito el 2026-10-09 (noche), el PASO 1 (viabilidad, SIN etiquetas) antes de mirar ningun resultado;
el PASO 2 (protocolo contra `strip_location`) esta escrito antes de cruzar ninguna cifra de PRUEBA.
Lo que se cambie despues de ver cifras se apunta como Cambio *n*, con fecha y hora.

## Punto de partida

`ZONA_COBERTURA_2026-10-09.md`, `CAMARA_TRIPODE_2026-10-09.md`, `RAYAS_ID_2026-10-09.md`,
`ANGULO_FUSION_2026-10-09.md`: de 5.472 clips anotados con `strip_location`, la homografia de
produccion contesta en 3.363 (0,615); con la fusion del angulo del borde lejano (`zona_final`)
acierta 0,516 frente a 0,480 de la homografia sola. De los 1.954 que no contestan: 0,467 nunca ven
dos rayas transversales en el mismo cuadro (el borde CERCANO esta tapado por el rotulo o justo
debajo de `BANDA_ROTULO`), 0,323 tienen homografia propagada con confianza < 0,35, 0,143 ven rayas
sin semilla valida, 0,067 pierden la cadena. Los tres respaldos ya probados (umbral mas bajo, franja
del rotulo mas abajo, camara de tripode AJUSTADA DENTRO del propio clip) contestan lo mismo o peor
que `middle`; el diagnostico comun es que el problema es de IDENTIFICACION (que raya es cual con una
o dos a la vista), no de modelo de camara ni de umbral.

## La idea nueva: registro ENTRE clips del mismo torneo

La camara de retransmision de un torneo es un TRIPODE que solo gira y hace zoom: dos cuadros de esa
camara, aunque sean de clips DISTINTOS, estan relacionados por una homografia de imagen completa
`H(q->r) = K2 R K1^-1`, que vale para el fondo (publicidad, gradas, marcador, suelo, rayas), no solo
para la pista. Un cuadro sin homografia de pista se registra contra cuadros de referencia de OTROS
clips del MISMO torneo que SI tienen homografia de produccion confiable (`motivo=ok`): se emparejan
rasgos del fondo ESTATICO (fuera de personas y del rotulo), y

    H_pista(q) = H_pista(ref) . H(q->ref)

de donde salen la x y la zona de los pies de `q` con las MISMAS funciones de siempre
(`pista_lineas.to_m`, `zona_tocado.zona`): no se reinventa la convencion.

**Que se excluye.** Las personas (cajas de `poses_<clip>.npz`) y el rotulo FIE sobreimpreso: se
reutiliza EXACTAMENTE la mascara de fondo de produccion (`pista_lineas.PisteFeatures.process`, que ya
excluye las capsulas de las personas y la banda `zona_tocado.BANDA_ROTULO`), sin reinventarla.

**Riesgo declarado para medir primero**: algunos torneos pueden usar una camara en rail que se
traslada, o varias camaras con cortes -- entonces `H(q->r)` no vale para la pista aunque se
encuentre una homografia de imagen con buenos inliers (porque la pista no esta en el mismo plano de
fondo que las referencias de alineacion, o el centro optico cambia). El PASO 1 mide justo esto: si el
error en metros es pequenio y consistente DENTRO de un torneo, el trípode compartido se cumple alli;
si es grande o erratico, no.

## Implementacion (ficheros NUEVOS, no se toca nada de produccion)

- `02_validacion/herramienta/registro_torneo.py`: `construye_uno` (cache de referencia: reproduce la
  homografia de produccion con `zona_tocado._rasgos` + `pista_lineas.calibrate`, igual que hizo el
  agente de `CAMARA_TRIPODE_2026-10-09.md`, y guarda ademas los puntos y descriptores SIFT
  RootSIFT del cuadro usado, enmascarado); `construye_query` (cache barata de un clip SIN homografia:
  un unico `PisteFeatures.process` en el cuadro `cuadro_tocado`, sin recalibrar nada); `empareja`
  (BFMatcher + ratio test 0,8 + `cv2.findHomography(..., USAC_MAGSAC, 3.0 px)`, compone
  `H_pista(ref) @ Si_ref @ H(q->ref) @ S_q` para pasar la escala de analisis -- 640 de ancho -- a
  pixeles originales, donde opera `H_pista`); `empareja_torneo` / `cmd_empareja_query` (el
  emparejamiento dentro de un torneo, dejando fuera el propio clip cuando es su propia referencia).
- Datos y resultados: `02_validacion/auditoria/registro_torneo_2026-10-09/`.
- Cache pesada (temporal): `D:/FENCINGLY_SCRATCH/agente_registro/`.

**Descriptor global de recuperacion top-k**: en vez de DINOv2 (GPU, descarga de HF), un descriptor
barato (el gris enmascarado reducido a 48x27, centrado y normalizado) -- con pozas por torneo
pequenias (<= 50 en el PASO 1) se calcula pero NO filtra: se prueban TODAS las referencias del
torneo, dentro del presupuesto de CPU (ver coste). Queda medido como diagnostico de si un futuro
recorte top-k perderia a las referencias buenas.

## PASO 1: viabilidad SIN etiquetas, SOLO en torneos de AJUSTE

**Muestra, declarada antes de ver ningun resultado**: de los 74 torneos de AJUSTE
(`02_validacion/auditoria/zona_cobertura_2026-10-09/torneos_ajuste.txt`, el reparto de
`ZONA_COBERTURA`, sin tocarlo), los que tienen >= 8 clips con homografia de produccion (`motivo=ok`)
-- el mismo umbral `MIN_REFS` de `angulo_lejano.py` --: 20 torneos. De cada uno, hasta 50 clips
`ok` (todos si hay menos; si hay mas, una muestra aleatoria, semilla 20261009): **614 clips**
(`lista_ajuste_sample.txt`). Cada uno se registra DEJANDO FUERA su propio clip, contra las demas
referencias de su mismo torneo (hasta 49): hasta 23.490 pares dirigidos.

**Que se mide, declarado antes de ver cifras**: por cada clip, entre las referencias cuyo
emparejamiento pasa un umbral minimo de inliers (se barre la rejilla {0, 6, 8, 10, 12, 15, 20, 25,
30, 40, 50}), la MEDIANA de la x proyectada; el error es `|mediana - x_tocado_m de produccion|`
(NUNCA `strip_location` en este paso). Se da la fraccion de clips con al menos una referencia que
pasa el umbral ("registrados"), la mediana y el percentil 90 del error en metros, y lo mismo por
torneo. Esto dice si la hipotesis del tripode compartido se cumple y que umbral de inliers conviene.

### Coste medido

- Cache de una referencia (reproduce la homografia de produccion + SIFT del cuadro usado): mediana
  ~9 s de CPU (un solo hilo; `zona_tocado._rasgos` + `pista_lineas.calibrate`, el mismo coste que ya
  paga la etapa 11, **no es coste nuevo** si se reusara una cache existente -- aqui se recalcula
  porque la `H` no se persiste en `zona_<clip>.csv`).
- Cache de un clip SIN homografia (la consulta real del respaldo): ~0,3-0,5 s de CPU (decodificar
  hasta el cuadro + un `PisteFeatures.process` + SIFT). Esto es lo que costaria DE VERDAD por clip en
  produccion si la referencia ya estuviese cacheada (ver mas abajo).
- Emparejar un par (BFMatcher + RANSAC): ~0,01-0,2 s segun el numero de puntos.

### Resultados (614 clips, 20 torneos; ejecutado con procesos de 1 hilo cada uno, <= 6 GB RAM, sin GPU)

Cache de las 614 referencias: 1.730 s (614 de 614 reproducen la x de produccion). Emparejamiento
(23.490 pares dirigidos, un solo hilo): 568 s. Coste medido por pieza: ver "Coste medido" arriba.

**Sin filtrar nada** (cualquier emparejamiento con `findHomography`): mediana de inliers por par
17, p90 70, maximo 825. La distribucion del error es MUY bimodal: con el umbral de inliers MAS BAJO
posible (>=0) ya contesta el 100 % (614/614), con mediana de error 1,02 m pero **p90 = 5,00 m y solo
el 48,9 % a menos de 1 m** -- la mitad de los pares da un resultado bueno y la otra mitad un
resultado que puede estar a varios metros, muy lejos del punto real.

**Subir el umbral de inliers de un SOLO par NO arregla esto**: de inliers>=6 a inliers>=50 la
mediana se queda entre 0,57 y 1,02 m y el p90 se queda en 5,0-5,5 m SIEMPRE (la fraccion registrada
baja de 1,000 a 0,860). El numero de inliers de una pareja aislada no distingue una homografia buena
de una mala: el fondo (gradas, publicidad) tiene zonas repetitivas donde un emparejamiento con muchos
inliers encaja igual de bien en el sitio equivocado -- el mismo sintoma de ambiguedad que ya se vio
con las rayas de la pista, pero ahora en el fondo.

**El ACUERDO ENTRE REFERENCIAS si separa lo bueno de lo malo**: exigiendo que al menos una fraccion
de las referencias que pasan el umbral de inliers coincidan entre si (a <= 1,5 m de la mediana
provisional), con inliers>=15:

| exigencia de acuerdo | registrados | fraccion | mediana \|err\| | p90 \|err\| |
|---|---:|---:|---:|---:|
| (ninguna) | 597 | 0,972 | 0,784 m | 5,343 m |
| >= 0,50 | 313 | 0,510 | **0,162 m** | 4,704 m |
| >= 0,67 | 195 | 0,318 | **0,086 m** | 3,146 m |
| >= 0,80 | 132 | 0,215 | **0,071 m** | 2,949 m |

Es decir: cuando VARIAS referencias independientes coinciden, el error tipico baja de ~0,8 m a
7-16 cm -- la hipotesis del tripode compartido se cumple muy bien ahi. El coste es la cobertura: el
acuerdo fuerte (>=0,67) solo se da en 1 de cada 3 clips.

**Por torneo (inliers>=15, sin exigir acuerdo), mediana del error en metros**: de los 20 torneos de
AJUSTE, **11 (55 %)** tienen mediana < 1 m (budapestsabre2019 0,03; qatar2020 0,07; budapestsabre2021
0,04; barcelona2018 0,05; barcelona2016 0,12; cali2018 0,26; anaheimfoil2018 0,54; cancunsabre2017
0,50; qatarfoil2021 0,47; budapest2020 0,61; stpetersburgfoil2019 0,75), y **9 (45 %)** tienen
mediana >= 1 m, varios por encima de 1,5-3 m (belgradefoil2022 2,96 [n=4], budapestwch2019 2,11,
katowicefoil2019 2,02, cali2019 1,76, parisfoil2022 1,69, berneepee2021 1,62, bern2018 1,48, wuxi2018
1,48, tokyoolympics2020 1,30). Con el filtro de acuerdo (inliers>=15, acuerdo>=0,5) la fraccion de
torneos con mediana < 1 m sube a 14/20 (70 %); los que siguen mal (bern2018, katowicefoil2019,
berneepee2021, budapest2020, parisfoil2022, belgradefoil2022) tienen muestras chicas (n <= 15 tras el
filtro) o son plausiblemente los casos de camara-en-rail / corte de plano que se declaro como riesgo
a medir.

**Lectura del PASO 1**: la hipotesis del tripode compartido SI se cumple, pero NO en todos los
torneos ni en todos los pares -- se cumple bien donde el fondo tiene textura rica y no repetitiva
(gradas con gente, publicidad variada) y donde VARIAS referencias independientes coinciden. Un
umbral de inliers de una sola pareja no es buen filtro de confianza por si solo; el acuerdo entre
referencias si lo es. Esto fija los parametros para el PASO 2: se declara la via **inliers>=6 (sin
exigir acuerdo)** como la family-member de mayor cobertura que bate a `middle` en AJUSTE (ver abajo),
y se da tambien, por separado, la variante con acuerdo (menos cobertura, mucho menos error en
metros) para quien prefiera precision a cobertura.

## PASO 2: protocolo de adopcion (declarado ANTES de cruzar con `strip_location`)

Igual que `ZONA_COBERTURA`/`CAMARA_TRIPODE`/`RAYAS_ID`: el umbral de inliers (y la regla de acuerdo
entre referencias, si hace falta) se eligen en AJUSTE con la rejilla del PASO 1 (sin mirar
etiquetas), por el mejor compromiso fraccion-registrada / error en metros. Con ese umbral fijo:

### Uso 1 (respaldo): clips que HOY no contestan

Muestra: la interseccion de `lista_ajuste.txt` de `ZONA_COBERTURA` (300 sin homografia + 60 que
contestan, AJUSTE) con los 20 torneos que tienen pozas de referencia -- 334 clips
(`lista_uso1_query.txt`: 277 sin homografia + 57 que ya contestan). Se registran los 277 contra las
referencias de su torneo (las mismas 614, mas las que no entraron en la muestra si el torneo las
tiene) con el umbral elegido.

**Criterio de adopcion, declarado ahora**:
- **(a)** los clips NUEVOS que contestan (sin homografia y con respuesta del registro) baten a
  `middle` en esos mismos clips, con IC95 por clip (bootstrap, 2.000) que no cruza cero (se da
  tambien el IC por torneo);
- **(b)** los 57 clips que ya contestaban NO CAMBIAN (el registro es un respaldo: por construccion
  solo actua donde la regla de siempre no responde; se comprueba re-ejecutando los 57);
- **(c)** la salida lleva una via nueva (`via_h = registro_torneo`), para separar las cifras.

Si la seleccion en AJUSTE no llega a +0,05 sobre `middle` (la misma regla de eleccion que las tres
veces anteriores), **PRUEBA no se mira**, igual que se hizo con `CAMARA_TRIPODE` y con la candidata N
de `ZONA_COBERTURA`. Si llega, se evalua UNA vez en los 716 clips sin homografia de
`lista_prueba.txt` (mismos torneos de PRUEBA que tengan poza de referencias) con el mismo criterio.

### Uso 2 (informativo o de fusion): clips que YA contestan

En los mismos 614 clips del PASO 1 (que ya tienen `zona`/`zona_final` de produccion), se compara la
zona del registro (mediana de las referencias que pasan el umbral) contra `zona_final` (la fusion
homografia+angulo ya adoptada). Criterio propio, declarado ahora: gana si acierta `zona_final` por
>= +0,02 con IC95 por torneo que no cruza cero. Esto es MAS BARATO de medir que el Uso 1 porque reusa
exactamente los pares ya calculados en el PASO 1 (no hace falta cache ni emparejamiento nuevos).

### Techo humano (contexto, de `catalogo.techo_humano`)

`strip_location`: acuerdo entre anotadores 0,619 (908 clips con >= 2 respuestas, 346 en desacuerdo).
Ningun resultado de aqui se interpreta por encima de ese techo sin desconfianza.

### Cambio 1 (2026-10-09, noche, visto el PASO 1 y ANTES de mirar `strip_location`): la rejilla de seleccion de Uso 1

Se declara ahora la rejilla sobre la que se aplica la regla de eleccion (la de las tres veces
anteriores: entre las candidatas que baten a `middle` por >= +0,05 con IC95 por clip que no cruza
cero, la que mas clips nuevos conteste): `inliers >= m`, `m` en {0, 4, 6, 8, 10, 12, 15, 20, 25, 30,
40, 50}, cada una con y sin la exigencia de acuerdo entre referencias (`a` en {ninguna, 0,34, 0,50,
0,67, 0,80}) -- 60 combinaciones, calculadas sobre los 334 clips de `lista_uso1_query.txt` (277 sin
homografia + 57 que ya contestan).

## Resultados de Uso 1 en AJUSTE (334 clips: 277 sin homografia + 57 que ya contestan)

Cache de los 277 (+57) clips sin homografia: 49 s (334 de 334, decodificar hasta el cuadro +
`PisteFeatures.process` + SIFT). Emparejar cada uno contra las referencias de su torneo (hasta 50):
456 s, un solo hilo.

De las 60 combinaciones (`barrido_uso1.txt`), las que baten a `middle` por >= +0,05 **y** tienen IC95
por clip que no cruza cero:

| regla | nuevos contestados | registro | middle | diferencia | IC95 por clip |
|---|---:|---:|---:|---:|---|
| **inliers>=6 (sin acuerdo)** | **264 (0,953)** | 0,542 | 0,477 | **+0,064** | [+0,015, +0,114] |
| inliers>=8 (sin acuerdo) | 264 (0,953) | 0,542 | 0,477 | +0,064 | [+0,011, +0,117] |
| inliers>=10 (sin acuerdo) | 264 (0,953) | 0,545 | 0,477 | +0,068 | [+0,019, +0,117] |
| inliers>=0, acuerdo>=0,34 | 194 (0,700) | 0,557 | 0,474 | +0,082 | [+0,021, ...] |
| inliers>=8, acuerdo>=0,50 | 106 (0,383) | 0,604 | 0,472 | +0,132 | [+0,047, ...] |
| inliers>=10, acuerdo>=0,50 | 115 (0,415) | 0,583 | 0,461 | +0,122 | [+0,035, ...] |
| inliers>=12, acuerdo>=0,67 | 39 (0,141) | 0,641 | 0,487 | +0,154 | [+0,026, ...] |
| inliers>=15, acuerdo>=0,67 | 39 (0,141) | 0,641 | 0,487 | +0,154 | [+0,026, +0,282] |
| inliers>=15, acuerdo>=0,80 | 22 (0,079) | 0,727 | 0,500 | +0,227 | [+0,091, ...] |
| inliers>=20, acuerdo>=0,80 | 24 (0,087) | 0,625 | 0,333 | +0,292 | [+0,125, ...] |

(la rejilla completa, con las que NO pasan, en `barrido_uso1.txt`.) **Regla de eleccion declarada
(la que mas conteste entre las que pasan): `inliers>=6`, sin exigir acuerdo.** Contesta 264 de 277
(0,953) clips que HOY no tienen zona; en esos 264, acierta 0,542 contra 0,477 de `middle`, **+0,064**,
IC95 por clip **[+0,015, +0,114]** (no cruza cero). Por torneo (n>=5): anaheimfoil2018 (n=57) +0,053,
bern2018 (n=5) +0,200, cali2018 (n=7) +0,143, cali2019 (n=8) +0,125 aprox.; la mayoria de torneos
tienen muestra chica aqui (son los `n>=8` pero solo unos pocos aportan >=20 clips sin homografia a
esta muestra de 300).

**Esto SI cruza el +0,05 de la regla de eleccion**, a diferencia de las tres veces anteriores
(`ZONA_COBERTURA` B(0,25)+p -0,167 en PRUEBA, `CAMARA_TRIPODE` ninguna candidata en AJUSTE,
`RAYAS_ID` G(8,inf) +0,019 en PRUEBA). Por el protocolo declarado arriba, **se pasa a PRUEBA**, una
vez, con `inliers>=6`.

### (b) los 57 que ya contestaban

Diagnostico (no cambia nada: el respaldo por construccion no actua si la regla de siempre ya
contesta): de los 57, el registro (mismo umbral) coincide con la zona de produccion en la mayoria;
ver `empareja_uso1_ajuste.jsonl.gz` para el detalle exacto. Como es un respaldo que solo se activa
cuando `cand` esta vacio en `zona_tocado.analiza`, (b) se cumple SIEMPRE por construccion, sin tocar
el codigo de produccion.

## Resultados de Uso 2 (fusion informativa, en los 614 del PASO 1)

Comparacion por clip (bootstrap pareado) de si el registro acierta `zona_final` (la fusion
homografia+angulo YA adoptada) frente a si la acierta la homografia estricta sola, en los mismos
clips:

| umbral | n | homografia estricta acierta `zona_final` | registro acierta `zona_final` | diferencia | IC95 por clip | IC95 por torneo |
|---|---:|---:|---:|---:|---|---|
| inliers>=6 | 614 | 0,725 | 0,721 | -0,003 | [-0,050, +0,042] | [-0,070, +0,065] |
| inliers>=15 | 597 | 0,724 | 0,702 | -0,022 | [-0,067, +0,025] | [-0,066, +0,025] |
| inliers>=20 | 590 | 0,724 | 0,683 | -0,041 | [-0,088, +0,007] | [-0,085, +0,008] |

**Uso 2 NO PASA con ningun umbral probado**: el registro nunca mejora (y con umbrales mas altos
empeora) lo que ya da la homografia estricta en los clips donde esta YA responde. Es coherente con
el PASO 1: estos 614 clips son justo los que tienen homografia de pista buena, asi que el registro
no aporta informacion nueva alli -- el valor esta en el Uso 1 (donde la pista NO da nada), no aqui.

## Resultados en PRUEBA (UNA evaluacion, `inliers>=6`, respaldo de Uso 1)

30 torneos de PRUEBA con >= 8 clips `motivo=ok`; cache de referencias (mismo cap de 50/torneo,
semilla 20261009): 724 clips, 2.570 s (724 de 724 reproducen la x de produccion). Interseccion de
`lista_prueba.txt` (716 sin homografia + 100 que contestan) con esos 30 torneos: 696 clips (605 sin
homografia + 91 que ya contestan); cache de consulta: 246 s. Emparejamiento (696 consultas contra
hasta 50 referencias de su torneo): 1.432 s.

    sin homografia 605; con referencia que pasa inliers>=6: 583 (con etiqueta y zona valida)
    NUEVOS (583 de 605, 0,964): registro 0,506   middle 0,491   diferencia +0,015
    IC95 por clip   [-0,019, +0,050]
    IC95 por torneo [-0,048, +0,056]
    CRITERIO (a): NO PASA (el IC95 por clip cruza cero)

**(a) NO PASA.** La diferencia (+0,015) es muy inferior a la de AJUSTE (+0,064) y su IC95 cruza
cero: el respaldo registra el 96,4 % de los clips sin homografia de PRUEBA, pero lo que registra
**no bate a `middle`** de forma fiable. Por torneo la cifra es muy ruidosa (de -0,615 en
cairosabre2019, n=13, a +0,333 en doha2019, n=9): ningun torneo individual tiene muestra para
concluir nada por si solo.

**(b)** diagnostico en los 91 que ya contestaban (no se tocarian en produccion, por construccion):
el registro coincide con la zona de produccion en 64 de 91 (0,703) -- consistente con el PASO 1
(fuera de un filtro de acuerdo, el registro aislado solo coincide con la pista real ~7 de cada 10
veces).

Esto repite, por CUARTA vez, el patron de los tres intentos anteriores de respaldo
(`ZONA_COBERTURA` B(0,25)+p, `CAMARA_TRIPODE`, `RAYAS_ID` G(8,inf)): prometedor en AJUSTE (aqui
+0,064, IC95 [+0,015,+0,114]), no replica en PRUEBA (+0,015, IC95 [-0,019,+0,050]).

## Resultados

- **La hipotesis del tripode compartido SI se cumple**, medida sin etiquetas en el PASO 1: en 11 de
  20 torneos de AJUSTE (55 %) la mediana del error en metros de un registro aislado (inliers>=15) ya
  es < 1 m, y sube a 14 de 20 (70 %) exigiendo que varias referencias independientes esten de
  acuerdo. Donde coinciden >= 2 de cada 3 referencias, el error tipico es de 7-9 cm.
- **Pero el error NO es uniforme**: la distribucion es bimodal (p90 de 3 a 5,5 m SIEMPRE, con
  cualquier umbral de inliers de una sola pareja). El numero de inliers de UN emparejamiento no
  distingue una homografia de fondo buena de una mala -- hace falta el acuerdo entre VARIAS
  referencias, que si separa bien lo bueno de lo malo pero cuesta cobertura (51 % de los clips a
  acuerdo>=0,5, 32 % a acuerdo>=0,67).
- **En PRUEBA, con la regla de mayor cobertura (`inliers>=6`, sin acuerdo) el respaldo NO bate a
  `middle`**: +0,015, IC95 por clip [-0,019, +0,050] (cruza cero), aunque SI registra el 96,4 % de
  los clips sin homografia de la muestra de PRUEBA. La version con acuerdo entre referencias (que en
  AJUSTE da un error en metros mucho menor) no se evaluo en PRUEBA, por disciplina de "una sola
  mirada": queda para un protocolo nuevo, con su propia porcion de PRUEBA.
- **Uso 2 (fusion en los clips que YA contestan) tampoco aporta**: -0,003 a -0,041 segun el umbral,
  siempre por debajo de produccion o indistinguible de ella.
- **Coste medido**: ~9 s de CPU por clip de REFERENCIA (recalcular `H_pista`, porque no se guarda en
  disco; ya se paga una vez por clip si se cachea, y la etapa 11 ya paga ese coste hoy); ~0,3-0,5 s
  de CPU por clip de CONSULTA (decodificar hasta el cuadro + una mascara + SIFT, sin recalibrar
  nada); ~0,01-0,2 s por emparejamiento (BFMatcher + RANSAC). Con una cache de referencias de ~50
  clips por torneo (amortizada una vez por torneo, no por clip), registrar un clip nuevo costaria
  del orden de SEGUNDOS, no minutos -- mucho mas barato que el respaldo de la camara de tripode
  (~6-11 s/clip) y del mismo orden que el de rayas_id (~0,4 s/clip + el ajuste del torneo).
- **Techo humano para contexto**: acuerdo entre anotadores en `strip_location` 0,619 (n=908); ningun
  numero de aqui se acerca a ese techo.

## Veredicto

**NO PASA. El respaldo de registro entre clips del mismo torneo queda SIN ENCENDER** (no se crea
ninguna variable de entorno de produccion: no se toco `zona_tocado.py` ni ningun fichero existente).
La cobertura de la zona se queda en 0,615 con el acierto de siempre.

Lo que si queda medido y sirve para un intento siguiente:

1. **La hipotesis de fondo es correcta** (camara de tripode compartida dentro de un torneo, util
   para registrar clips sin homografia de pista): el error SI baja a pocos centimetros cuando varias
   referencias independientes coinciden. El problema no es la idea, es que un emparejamiento aislado
   (incluso con muchos inliers) puede estar apoyado en una zona de fondo repetitiva (gradas,
   publicidad) y dar una homografia plausible pero FALSA, y ese fallo no se nota si solo se mide UN
   par.
2. **El siguiente intento con sentido, declarado para quien retome esto**: exigir desde el principio
   el acuerdo entre >= 3-5 referencias (no solo inliers de una pareja) como UNICO criterio de
   aceptacion, aceptando la cobertura mas baja que eso implica (ronda el 30-50 % de los clips sin
   homografia, por el PASO 1); evaluar ESO en un protocolo nuevo con su propia porcion de PRUEBA (la
   de aqui ya se gasto). Tambien ayudaria: (a) excluir activamente del fondo las zonas de publicidad
   repetida (un detector de "parche que se repite en la misma imagen" descartaria puntos SIFT alli,
   no solo los de personas y rotulo); (b) usar el descriptor global barato (o uno mejor) para
   RECUPERAR primero las referencias mas parecidas en apariencia general, en vez de probarlas todas
   -- aqui no hizo falta por el tamanio de las pozas, pero en produccion (torneos con cientos de
   clips `ok`) si, y podria ademas mejorar la precision si las mas parecidas en conjunto son tambien
   las que menos se confunden con otra zona del recinto.
3. **Confirmado el riesgo declarado al principio**: no se investigo aparte que torneos usan rail o
   varias camaras, pero los torneos con peor mediana de error en el PASO 1 (bern2018, katowicefoil2019,
   berneepee2021, belgradefoil2022) son candidatos a mirar primero si se retoma esto.

## Que queda en el repositorio

- `02_validacion/herramienta/registro_torneo.py` (nuevo): toda la logica (cache de referencias y de
  consultas, emparejamiento, los comandos `cache`/`empareja`/`cache_query`/`empareja_query`). No se
  importa desde ningun fichero de produccion.
- `02_validacion/auditoria/registro_torneo_2026-10-09/`: listas (`lista_ajuste_sample.txt`,
  `torneos_ajuste_con_refs.txt`, `lista_uso1_query.txt`, `lista_prueba_sample.txt`,
  `torneos_prueba_con_refs.txt`, `lista_uso1_query_prueba.txt`), resultados
  (`empareja_ajuste.jsonl.gz`, `empareja_uso1_ajuste.jsonl.gz`, `empareja_uso1_prueba.jsonl.gz`,
  `barrido_uso1.txt`) y los scripts de analisis (`analiza_ajuste.py`, `evalua_uso1.py`,
  `evalua_uso2.py`).
- Cache pesada (temporal, se puede borrar): `D:/FENCINGLY_SCRATCH/agente_registro/` (`cache/`,
  `cache_query/`, `cache_prueba/`, `cache_query_prueba/`, `logs/`).
- **No se toco ningun fichero existente del repo.**

# V-JEPA 2: que guardar de cada clip y que preguntas nuevas contesta. Protocolo DECLARADO ANTES de medir

Escrito el 2026-10-09 a las 22:00. A esa hora no se ha extraido ninguna variante, salvo la prueba de
humo de 3 clips (solo se comprobo que el recorte es el mismo, abajo). Tampoco se ha visto ninguna
cifra del reservado con 5 semillas de `VJEPA2_2026-10-09.md`. Lo que se cambie despues de ver
resultados se apunta abajo como cambio, con su fecha y su hora.

## Por que

La corrida de florete pasa una sola vez 106.432 videos por la GPU. Lo que no se guarde entonces no
existira despues sin repetirla. Hoy se guarda una sola cosa por clip ("V0"):

- el recorte de la pareja;
- 64 cuadros de `[tocado - 4,0 s, tocado + 0,5 s]`;
- los tokens promediados 4x4 en el espacio (2 MB por clip).

El usuario ha decidido que el coste no limita (`VJEPA2_2026-10-09.md`, cambio 3): se acepta hasta
~1 mes de corrida. Por eso se mide si guardar MAS da mas. Resultado previo, que motiva una de las
variantes: la sonda lineal sobre la media de V0 no sabe la zona de la pista
(`VJEPA_ZONA_2026-10-09.md`). En los clips sin homografia saca 0,489 contra `middle` 0,492, y el
recorte deja fuera casi toda la pista.

## Lo que se extrae (`01_extraccion/codigo/vjepa_variantes.py`), sobre los mismos 5.836 clips de V0

Mismo modelo (ViT-L, 256 px), misma ancla, misma ventana y mismo espejo que `vjepa.py`. El codigo se
importa de alli, no se copia. Por clip:

| clave | que es | tokens | MB/clip |
|---|---|---:|---:|
| `rec8` | el MISMO recorte y la misma ventana que V0, promediado 8x8 en vez de 4x4 | 2.048 | 8,4 |
| `ent8` | la imagen ENTERA (16:9 estirada a 256x256, sin recortar), misma ventana | 2.048 | 8,4 |
| `pre4` | el recorte de la ventana ANTERIOR, `[tocado - 8,5 s, tocado - 4,0 s]`, misma densidad | 512 | 2,1 |

**Comprobacion obligatoria antes de medir nada.** `rec8` promediado 2x2 tiene que ser el `tok` de
V0: mismos cuadros, misma caja y diferencia de redondeo fp16. En la prueba de humo (3 clips) la
diferencia relativa maxima es 0,0007 y los cuadros y la caja son identicos. Se repite sobre 50 clips
al acabar. Si falla, no se mide nada.

## Las variantes que se comparan

Todas pasan por **la misma sonda atentiva**, la misma rejilla elegida en la particion interna, las
mismas 5 semillas, el mismo reparto por torneo y las mismas fusiones que la version que decide de
`VJEPA2_2026-10-09.md`. Solo cambian los tokens que entran (concatenados en el eje de los tokens; la
vista espejo concatena los espejos, asi que la antisimetria sigue impuesta):

- **V0** = `rec4` (512 tokens), lo de hoy. Se vuelve a correr por el mismo camino que las demas,
  guardando la prediccion de cada clip, para poder comparar por pares.
- **V1** = `rec4 + ent4` (1.024 tokens): anadir la imagen entera.
- **V2** = `rec8` (2.048 tokens): menos compresion.
- **V3** = `rec4 + pre4` (1.024 tokens): anadir lo de antes de la ventana.
- **V4**, despues y aparte: el modelo grande, ViT-g a 384 px (`vjepa2-vitg-fpc64-384`, `rec4`, 512
  tokens de 1.408). Se mide su s/clip en 20 clips antes de extraerlo entero.

`ent4` y `rec4` salen exactos de `ent8` y `rec8` promediando 2x2.

## Que decide

**Pregunta 1 (`initiated`) decide.** Se compara `C'(Vk)` contra `C'(V0)` (etiquetador + luces +
sonda), por pares en los mismos clips del reservado, con IC95 por bootstrap agrupado por torneo. Hay
dos resultados posibles:

- **Vk mejora**: `C'(Vk) - C'(V0) >= +0,02`, el IC95 no cruza cero y el espejo da 1,0000.
- **No se distingue**: en cualquier otro caso. Entonces lo mas barato se queda.

Es el mismo umbral que adopto V-JEPA. Se dan tambien `B(Vk) - B(V0)`, la sonda sola, que es mas
sensible a la representacion, y el coste medido (s/clip de GPU y MB/clip). Para la corrida, el disco
cuenta: 106.432 clips x 19 MB son ~2 TB, y en D: hay 1,7 TB libres.

**Informativo, no decide:**

- la pregunta 2 (`prioridad`) con cada variante;
- la zona con V1: la sonda lineal de `VJEPA_ZONA_2026-10-09.md` sobre la media de `ent`, mismo
  reparto y mismos usos.

## Pregunta 3 (nueva, con V0, sin extraer nada): `body_location`, el blanco tocado

- **Filas**: `etiquetador.reune(cabeza="body_location")` sobre los anotados con embedding. Hay 8
  clases (`CA.BLANCOS`), muy desbalanceadas: torso 863 de 1.699.
- **Reparto**: `particion_por_torneo` con su semilla sobre ESTAS filas.
- **Simetria**: el blanco no cambia al espejar el clip. La sonda es simetrica por construccion
  (`logits = s(x) + s(x')`) y el control del espejo pide que la prediccion sea IDENTICA con las vistas
  intercambiadas (1,0000).
- **Los modelos**:
  - A = la cabeza `cuerpo` del etiquetador, que ya es simetrica (5 semillas, media de logits);
  - B = la sonda atentiva simetrica (rejilla con 1 semilla y 5 semillas fuera);
  - C = una logistica multinomial sobre `[logits de A, logits de B]`, ajustada en la particion
    interna.
- **Lineas base**: la clase mayoritaria del entrenamiento (torso) y el techo humano: dos anotadores
  coinciden en 0,481 (`catalogo.techo_humano`), POR DEBAJO de contestar siempre torso. Con ~400
  clips de reservado, el IC sera ancho.
- **Criterio "V-JEPA aporta al blanco"**: `C - A >= +0,02`, con IC95 por torneo sin cruzar cero y
  espejo 1,0000. Se dan por arma (en florete casi todo es torso) y B contra la mayoritaria.
- **La ventana incluye el tocado**: acaba 0,5 s despues, asi que el contacto esta dentro.

## Cambio 1 (2026-10-09 22:05, ANTES de extraer nada salvo pruebas de humo de 3-8 clips sin cifras de acierto)

**Por que.** El usuario propone los "tres ojos" (macro, meso, micro) y pide usar el modelo GRANDE en
vez del pequenio. Esto sustituye a las variantes V1-V4 de arriba, que no se extraen. Ninguna de las
dos cosas se habia medido contra etiquetas.

**El modelo.** ViT-g de V-JEPA 2: 1.000 M de parametros, Apache-2.0, sin acceso restringido.
Coste medido en 3 clips, con los tres ojos x 2 vistas, por clip:

| modelo | s de GPU/clip | anotados (5.836) | florete (106.432) |
|---|---:|---:|---:|
| ViT-L 256 (V0, 1 ojo) | 1,2 | 2 h | 1,5 dias |
| ViT-g 256, 3 ojos | 10,4 | ~17 h | ~13 dias |
| ViT-g 384, 3 ojos | 34,3 | ~55 h | ~42 dias |
| ViT-g 384, solo meso | ~11,4 | ~18 h | ~14 dias |

ViT-g 384 con los tres ojos no cabe en el mes de la corrida: 42 dias, ademas de las etapas 1 + 4.
Por eso se extrae en dos tandas sobre los anotados:

- **Tanda G256**: ViT-g 256, los tres ojos.
- **Tanda G384**: ViT-g 384, solo el ojo meso. Dice si la resolucion compensa su coste.

**Los tres ojos** (`vjepa_variantes.py`):

| ojo | clave | tiempo | espacio | tokens |
|---|---|---|---|---:|
| macro | `mac4` | el clip ENTERO, 64 cuadros repartidos (~6/s) | la imagen entera, 16:9 estirada | 512 |
| meso | `rec8` | `[a - 4,0 s, a + 0,5 s]`, como V0 | el recorte de la pareja de V0 | 2.048 (8x8) |
| micro | `mic8` | `[a - 1,5 s, a + 0,5 s]`, 64 cuadros (~32/s, casi los fps del video) | ZOOM movil que contiene las dos manos adelantadas (Sapiens) | 2.048 (8x8) |

`rec4` y `mic4` salen exactos promediando 2x2.

**El zoom del micro se fijo mirando 8 clips, antes de cualquier cifra**
(`02_validacion/auditoria/vjepa_ojos_2026-10-09/`). La caja fija de las manos salia tan grande como
el recorte de la pareja; la caja de lado fijo enseniaba el hueco entre los dos. La regla final:

- **centro**: el punto medio de la mano adelantada de cada tirador;
- **tamanio**: ancho = max(1,3 |dx|, 0,9 x el alto del tirador), alto lo mismo con |dy|, con razon
  entre 1 y 2;
- **suavizado**: mediana movil de 7.

En los 8 clips, 7 tienen las dos manos en 58-64 de los 64 cuadros y 1 va por el respaldo.

**Lo que se compara** (misma sonda, rejilla, 5 semillas, reparto y fusiones que arriba; decide
`initiated` con el mismo umbral, por pares contra su referencia):

| comparacion | que mide | referencia |
|---|---|---|
| G0 = G256 `rec4` | el modelo grande, mismo ojo y misma compresion | V0 |
| G0b = G256 `rec8` | menos compresion | G0 |
| G1 = `rec4 + mac4` | anadir el macro | G0 |
| G2 = `rec4 + mic4` | anadir el micro | G0 |
| G3 = `rec4 + mac4 + mic4` | los tres ojos | G0 y V0 |
| G4 = G384 `rec4` | la resolucion | G0 |

Informativo: `mic8` solo; la zona con la media de `mac` (sonda de `VJEPA_ZONA`); `prioridad` y
`body_location` con lo que gane.

**Como se entrenan las sondas.** Los tokens van a un `memmap` en el NVMe (C:, temporal; se borra al
acabar). Con 1.536 tokens de 1.408 son ~48 GB, que no caben en 32 GB de RAM. La sonda es la misma y
lee por indices igual.

## Cambio 2 (decidido 2026-10-10 00:45, escrito aqui a las 12:24; ANTES de extraer DINOv2, salvo una prueba de humo de 3 clips sin cifras)

**DINOv2 cuadro a cuadro**, la otra idea que propuso el usuario. Un modelo de IMAGEN (DINOv2-L/14,
Apache-2.0) mira cada cuadro por separado. Eso permite mas cuadros y mas resolucion por cuadro; la
relacion en el tiempo la aprende la sonda.

- Extractor: `01_extraccion/codigo/dino_cuadros.py`.
- Mismo recorte y misma ancla que V0.
- 128 cuadros a paso uniforme en `[a - 4,0 s, a + 0,5 s]`, unos 28/s: a 30 fps, casi todos.
- 448 px por cuadro (32x32 parches de 14 px), frente a 256 de V0.
- Se guarda por cuadro el CLS y los parches promediados 4x4: 17 tokens de 1.024; con las dos
  vistas, 8,9 MB por clip.
- Coste medido en 3 clips con la GPU COMPARTIDA con G256: 23,4 s/clip. Libre sera menos; se dara el
  medido.

**Orden de la cola**: G256 (en marcha), luego DINOv2 y luego G384. DINOv2 pasa delante de G384
porque es la idea mas distinta; G384 solo cambia la resolucion del mismo ojo. La extraccion G256 en
marcha no se toco.

**Lo que se compara**:

| comparacion | referencia | papel |
|---|---|---|
| D0 = DINOv2, 128 x 17 tokens | V0 | decide, con el mismo umbral (C' por pares, >= +0,02, IC95 sin cruzar cero, espejo 1,0000) |
| D0 | G0 | informativo |

**La sonda de D0** se elige en la particion interna, como la rejilla, entre dos:

- (a) la MISMA sonda atentiva de V-JEPA, sobre los 2.176 tokens con su posicion aprendida;
- (b) una sonda TEMPORAL: atencion dentro de cada cuadro, luego 2 capas de transformer a lo largo de
  los 128 cuadros, impar por construccion igual.

Gana la de mejor acierto interno; despues se evalua UNA vez fuera.

## Cambio 3 (2026-10-12 00:05, con G0-G3 ya medidos y ANTES de extraer nada de lo que sigue)

**Por que.** Lo medido hasta aqui (ver Resultados):

- el ojo micro es lo que mejora: G2 contra G0, +0,023, MEJORA;
- el modelo grande solo no aporta: G0 contra V0, +0,001;
- el macro encima del micro no suma: G3 no se distingue ni de G0 ni de V0.

Falta lo que mas pesa en el presupuesto de la corrida: **si el micro necesita el modelo grande**.
ViT-g con meso + micro no se ha medido aparte (con los tres ojos son 10,4 s/clip); ViT-L con un ojo
son 1,2 s/clip.

**G5 = ViT-L 256, `rec4 + mic4`**: V0 mas el ojo micro, con el modelo pequenio. Se extrae con
`vjepa_variantes.py --modelo vjepa2-vitl-fpc64-256 --salidas rec8,mic8` en
`D:\FENCINGLY_SCRATCH\vjepa_l256`, sobre los mismos 5.837 clips. Misma sonda, rejilla, 5 semillas,
reparto y fusiones que las demas.

| comparacion | referencia | papel |
|---|---|---|
| G5 | V0 | **decide**, mismo umbral (C' por pares >= +0,02, IC95 sin cruzar cero, espejo 1,0000) |
| G5 | G2 | informativo: lo que se pierde con el modelo pequenio |

Antes de la sonda se comprueba que el `rec4` de esta extraccion (promedio 2x2 de `rec8`) es el `tok`
de V0 en los mismos clips. Es una comprobacion del camino, no una cifra de acierto.

**`prioridad` con lo que gane** (ya declarado arriba, informativo). Gano G2, asi que la sonda de
`prioridad` se hace con `rec4 + mic4`, en vez de con los tres ojos que la cola tenia puestos por
adelantado. `rec4` se queda.

**Orden.** La extraccion G384 (600 de 5.837) se para y sigue despues de G5: guarda clip a clip y salta
los hechos. G4 solo mide la resolucion del ojo meso, y en ese ojo el modelo grande no aporto. G5, en
cambio, decide cuanto cuesta la corrida. La sonda G0b, en marcha, no se toca. Despues de ella van G5,
`mic8` solo y `prioridad`.

## Resultados

### G0 y G1 (10/10, `initiated`, 5 semillas, reservado de 1.384 clips de 42 torneos)

Salidas: `02_validacion/auditoria/vjepa_ojos_2026-10-09/compara_*.txt` y `sonda_G256_*.txt`. Por
pares en los mismos 1.384 clips, IC95 por bootstrap agrupado por torneo, espejo 1,0000 en todas.

| comparacion | C' ref | C' var | dif (IC95) | B ref -> var | veredicto |
|---|---:|---:|---|---|---|
| **G0** (ViT-g, `rec4`) contra V0 | 0,696 | 0,697 | +0,001 [-0,008, +0,022] | 0,703 -> 0,684 | **no se distingue** |
| **G1** (`rec4 + mac4`) contra G0 | 0,697 | 0,718 | +0,021 [-0,002, +0,032] | 0,684 -> 0,704 | **no se distingue** (el IC cruza cero) |
| G1 contra V0 (**informativo**: no estaba declarada) | 0,696 | 0,718 | +0,022 [+0,009, +0,036] | 0,703 -> 0,704 | cumpliria el umbral |

Por arma, C' de V0 -> G0 -> G1: espada 0,727 -> 0,722 -> 0,761 (n=209), florete 0,683 -> 0,685
-> 0,705 (n=1.040), sable 0,748 -> 0,756 -> 0,756 (n=135).

Lectura:

- **El modelo grande solo, con el mismo ojo, no aporta**: +0,001. La sonda sola (B) incluso baja
  0,019. En la particion interna G0 parecia mejor (0,698 contra 0,667), pero eso no se sostuvo fuera:
  por eso decide el reservado y no la interna.
- **El ojo macro (el clip entero, la imagen entera) es lo primero que mueve la aguja**: +0,021 sobre
  G0, justo en el umbral, pero el IC toca cero (-0,002). No pasa.
- **G1 contra V0 cumpliria el umbral**, pero esa comparacion no estaba declarada: se miro despues de
  ver las otras dos, y con 6+ variantes alguna puede pasar por azar. No decide. La declarada que
  responde a eso es **G3 contra V0** (los tres ojos), que sale despues.

### G2: el ojo micro (11/10 ~02:00)

| comparacion | C' ref | C' var | dif (IC95) | B ref -> var | veredicto |
|---|---:|---:|---|---|---|
| **G2** (`rec4 + mic4`) contra G0 | 0,697 | 0,720 | **+0,023 [+0,015, +0,041]** | 0,684 -> 0,707 | **MEJORA** |

Espejo 1,0000. Por arma, C' de G0 -> G2: espada 0,722 -> 0,770, florete 0,685 -> 0,701, sable
0,756 -> 0,793. La sonda sola sube +0,023 [-0,003, +0,041] y la lineal +0,019 [+0,002, +0,040].

**El zoom a las manos en los 2 ultimos segundos es la primera pieza que pasa el criterio
declarado.** Lo que no dice todavia: si sumado a V0 (el modelo pequenio) tambien pasa; eso lo
contesta G3 contra V0, con los tres ojos. En florete la subida es la menor (+0,016).

Coste medido de la extraccion G256 con los tres ojos: 5.837 clips en 60.561 s (10,4 s/clip de
pared, 9,6 s de GPU de mediana).

**Memoria (10/10 tarde).** Escribir y leer los memmap de 15-60 GB dejo dos veces el sistema sin RAM
disponible (la sonda retenia 20,6 GB de paginas del fichero). Arreglos, que no cambian ninguna cifra:
`sonda_vjepa.py` vuelca el memmap cada 100 clips al escribirlo y pone un techo duro de 8 GB al
conjunto de trabajo del proceso (`--tope-gb`). A la sonda G2 en marcha se le puso el mismo techo
desde fuera: la RAM disponible paso de 0,8 a 13,7 GB.

**Velocidad (11/10 13:30).** Con el techo, la sonda G3 (1.536 tokens, 45 GB) leia el memmap por
paginas de 4 KB y tardo 6.000 s por cada 80 epocas: 3,6 veces lo de G2 con solo 1,5 veces los
tokens. Desde G0b, las variantes se leen clip a clip con lecturas de fichero (`Filas`), y el lote
siguiente se lee mientras la GPU calcula. Comprobado antes de usarlo: mismas filas que el memmap en
los 47,9 GB de G3, el lote de 64 en 0,3 s frente a 0,9 s, y logits identicos (diferencia 0,0) al
bucle anterior. G3 sigue con el codigo viejo: pararla obligaria a repetir su rejilla.

### G3: los tres ojos (11/10 20:20)

| comparacion | C' ref | C' var | dif (IC95) | B ref -> var | veredicto |
|---|---:|---:|---|---|---|
| **G3** (`rec4 + mac4 + mic4`) contra G0 | 0,697 | 0,711 | +0,014 [-0,012, +0,034] | 0,684 -> 0,711 | **no se distingue** |
| **G3** contra V0 (la declarada para "los tres ojos") | 0,696 | 0,711 | +0,015 [-0,005, +0,037] | 0,703 -> 0,711 | **no se distingue** |
| G2 contra V0 (**informativo**: no estaba declarada) | 0,696 | 0,720 | +0,025 [+0,016, +0,053] | 0,703 -> 0,707 | cumpliria el umbral |
| G3 contra G2 (**informativo**) | 0,720 | 0,711 | -0,009 [-0,037, +0,006] | 0,707 -> 0,711 | no se distingue |

Espejo 1,0000 en todas. Por arma, C' de G0 -> G2 -> G3: espada 0,722 -> 0,770 -> 0,770, florete
0,685 -> 0,701 -> 0,695, sable 0,756 -> 0,793 -> 0,741. La rejilla interna de G3 eligio 20 epocas y
wd 1e-2 (0,685). Contra el etiquetador con luces, G3 da +0,056 [+0,019, +0,099]: V-JEPA sigue
entrando. Lo que no pasa es la mejora sobre G0 y sobre V0. La sonda sola si sube sobre G0, +0,027
[+0,007, +0,054], pero la fusion C' no.

Lectura:

- **Los tres ojos juntos no pasan.** Sumar el macro al micro no ayuda: G3 queda 0,009 por debajo de
  G2.
- **Lo que queda en pie es el micro.** G2 contra G0 es la unica comparacion declarada que pasa.
  Contra V0 tambien cumpliria (+0,025), pero esa no estaba declarada. Si el micro necesita el modelo
  grande lo dice G5 (cambio 3).
- **En florete la subida es pequenia en todas las variantes**: entre +0,010 y +0,022 sobre G0 o V0
  (G1 0,705, G2 0,701, G3 0,695, frente a 0,685 de G0 y 0,683 de V0).

### Pregunta 3, `body_location` con V0 y 5 semillas (10/10 00:06)

Salida: `02_validacion/auditoria/vjepa_ojos_2026-10-09/sonda_cuerpo_5semillas.txt`. El reservado son
436 clips de 25 torneos; el entrenamiento, 1.234.

| | acierto | contra A (IC95 por torneo) |
|---|---:|---|
| clase mayoritaria (torso) | 0,523 | +0,044 [-0,002, +0,091] |
| A (cabeza `cuerpo` del etiquetador) | 0,479 | linea base que decide |
| B (sonda V-JEPA) | 0,443 | -0,037 [-0,110, +0,036] |
| **C (A + B)** | **0,532** | **+0,053 [+0,007, +0,099]** |
| techo humano (n=108) | 0,481 | |

El espejo da 1,0000 en B y en C.

**El criterio declarado PASA, pero la lectura honesta es que V-JEPA no sabe donde fue el tocado.**

- **La linea base estaba mal elegida.** La que se declaro (A) queda POR DEBAJO de contestar siempre
  torso (0,479 contra 0,523).
- **C apenas se separa de la mayoritaria**: 0,532 contra 0,523, +0,009. La matriz de confusion
  dice por que: C contesta torso en 398 de los 436 clips. Lo que mejora frente a A es decir torso
  mas a menudo, no ver el blanco.
- **B sola queda por debajo de la mayoritaria**: -0,080 [-0,134, -0,031].

El fallo es mio, al declarar el protocolo: la linea base que decide tenia que haber sido la mejor de
las dos (A y la mayoritaria). Con ella, C - max(A, mayoritaria) = +0,009, que no llega a +0,02.
**Veredicto: no aporta.** No hay nada de produccion que adoptar.

Con 1.234 clips de 8 clases desbalanceadas y un techo humano de 0,481, esta etiqueta no da para mas
de lo que da contestar torso. Si se quisiera de verdad, habria que limitarla a lo que el arbitraje
usa: en florete, valido (torso) o no valido. Eso ya lo da la luz blanca de la etapa 9.

### Calidad de la extraccion G256 (10/10 12:40, muestra de 928 de 4.637 clips ya extraidos, sin etiquetas)

- **Ancla**: `zona` en 892 clips y `final` en 36.
- **Ojo micro**:
  - 879 clips por las manos (0,947), 42 por el respaldo y 7 con la imagen entera;
  - cuadros con las dos manos: mediana 61 de 64, p10 30;
  - ancho del zoom: p10 209 px, mediana 344 px, p90 678 px. El recorte meso suele medir 700-1.280
    px, asi que el zoom de la mediana ve ~2x mas detalle.
- Ningun clip con cuadros que falten y ningun NaN.
- 9,6 s de GPU por clip, con la GPU compartida con las sondas.

# Tanda del 2026-09-22 — etiquetas humanas, etiquetador, Colab y pista

Lo que se hizo, lo que se midio, **lo que la medida contradijo del plan**, y lo que queda abierto.

---

## Resumen en una tabla

| bloque | estado | lo que decide |
|---|---|---|
| **A** ingesta del catalogo y la anotacion | **hecho y verificado**, 0 lineas mal | el manifiesto del que todo lo demas tira |
| **B** el etiquetador | **construido y con su control pasando**; sin entrenar por falta de datos extraidos | la antisimetria se verifica al 100 % |
| **C** la puerta | **no se puede decidir todavia**: faltan clips extraidos | si se bajan los que faltan |
| **D** Colab | **hecho**: rutas, capacidades, etapas 7/9/10 cableadas, `lotes.py`, requirements, cuaderno | que la extraccion en masa sea posible |
| **E** la pista | **hecho**: el clip se ve, y tres defectos de figura corregidos | — |
| **F1** la mano armada contra la FIE | **hecho y medido**: 967 tiradores con la mano declarada | la cascada baja de 0,949 a **0,881** contra verdad independiente |

---

## A · La ingesta

`03_datos/catalogo/` con `gfycats.txt.gz` (9,0 MB comprimido de 103,4 MB), `form_responses.txt`
(1,1 MB), `manifiesto.csv.gz` (13,6 MB, 362.950 filas) y `FUENTE.md` con el `ETag` y el
`Last-Modified` de los dos. El unico lector es `01_extraccion/codigo/catalogo.py`.

`02_validacion/herramienta/verifica_catalogo.py` reproduce **todas** las cifras del plan y sale con
**0 lineas mal**: 362.950 clips, 172 torneos, foil 106.432 / epee 140.639 / sabre 102.618,
`touch` 173.640 / 165.194 / 24.106, y los 8.555 del corpus dentro y todos `foil`.

### Las cuatro cosas que la medida encontro y el plan no tenia

**1. `weapon` no siempre trae un arma.** 13.260 clips traen `champs` — la familia del torneo — y uno
lo trae vacio. **Los 106.432 de florete son una cota inferior**, y hay 13.261 clips de arma
desconocida que nadie ha mirado.

**2. 8.513 filas de anotacion son 6.829 clips.** 1.184 estan anotados mas de una vez, uno **35
veces**. El convenio para resolverlos cambia la cifra:

| convenio | `initiated` | `strip_location` | `body_location` |
|---|---:|---:|---:|
| la ultima fila gana | 5.628 | 5.635 | 1.698 |
| cualquiera no vacia | 5.803 | 5.800 | 1.734 |
| **mayoria, empates fuera** (elegido) | **5.596** | **5.544** | **1.699** |

Las cifras del plan son el primer convenio, que **nadie eligio**: es lo que sale de sobrescribir un
dict mientras se lee.

**3. El techo del etiquetador, que ahora se puede medir.** Sobre los clips con dos o mas respuestas
independientes, la fraccion en que **todas** coinciden:

| cabeza | clips con >=2 respuestas | en desacuerdo | **acuerdo** |
|---|---:|---:|---:|
| `initiated` | 911 | 284 | **0,688** |
| `strip_location` | 908 | 346 | **0,619** |
| `body_location` | 108 | 56 | **0,481** |

Esto es lo mas valioso de la tanda. El plan declaraba la linea base **por abajo** y no tenia nada
por arriba. Ahora **0,69 en la cabeza 1 es igualar a los humanos**, y un 0,80 hay que mirarlo con
desconfianza antes que con alegria.

**4. La clase mayoritaria no es 0,51.** El plan la declara en 0,51 para las dos primeras cabezas.
Medida: **0,47** para `initiated` (fotl, 2.608 de 5.596) y **0,41** para `strip_location` (middle,
2.249 de 5.544). La linea base declarada estaba 4 y 10 puntos por encima de la real.

### Y el denominador de "ya esta en disco", que son tres numeros

| conjunto | n |
|---|---:|
| anotados que estan en el **corpus ya procesado** de 8.555 | **1.370** ← el numero del plan |
| con etiqueta, en ese corpus | 1.171 |
| anotados con su mp4 en `D:` | 4.366 |
| **con etiqueta, con su mp4 en `D:`** | **3.654** |

El 1.370 es correcto pero es *el corpus*, no *el disco*. En D: hay **2,7 veces mas** clips anotados
de los que el plan supone, y lo que queda por bajar son **2.202 clips (~7,5 GB)**, no 5.378 y 19 GB.

**Y la trampa que hay debajo, que es la que manda:** "el mp4 esta" no es "el clip esta extraido".
De los 5.856 con etiqueta, los que tenian `poses_<clip>.npz` al empezar eran **3**.

---

## B · El etiquetador

`01_extraccion/codigo/etiquetador.py` y `02_validacion/herramienta/verifica_etiquetador.py`.

Tres cabezas, la entrada son las series que ya extraemos (no el video), reparto **por torneo**, y
la antisimetria impuesta **por construccion**: rasgos en el marco propio de cada tirador y la
cabeza sobre la **diferencia** de los dos embeddings, con la parte par sobre `|d|`.

### El control del espejo encontro dos defectos reales en su primera ejecucion

No se comprueba intercambiando las dos ranuras del tensor — eso es cierto siempre y no dice nada.
Se construye un `poses_<clip>.npz` **espejado de verdad** (`x -> W-x`, los pares simetricos del
modelo intercambiados, y las dos personas cambiadas) y se exige que salgan los mismos rasgos con
las ranuras cambiadas.

| pasada | clips espejados | fallan | peor discrepancia relativa |
|---|---:|---:|---|
| primera | 9 | **9** | 2,29 en `v_mano_adelante` |
| tras arreglar el filtro y el marco | 12 | 3 | 0,166 en `v_adelante` |
| **tras guardar la cadencia en el NPZ** | **12** | **0** | **7,3e-06** (ruido de float32) |

Los dos defectos:

- **`_suave` rompia la equivarianza.** Rellenaba los NaN con la media y convolucionaba con
  `mode="same"`, asi que el nucleo truncado en los bordes metia el nivel de relleno con peso
  distinto a cada lado. Sobre una coordenada absoluta eso no conmuta con el espejo. Ahora la media
  movil divide por la cuenta de validos y conmuta con cualquier transformacion afin.
- **La cadencia se deducia de un CSV que puede no existir, y si no existia valia 25.** Un clip de
  30 fps leido como 25 sale con **todas** las velocidades multiplicadas por 1,2 — que es exactamente
  la discrepancia que el espejo mostraba. Ahora `fps` viaja dentro de `revision_<clip>.npz` y de
  `poses_<clip>.npz`, que es donde tiene que estar: en Colab el mp4 se borra tras procesar.

El segundo es un defecto **del pipeline**, no de la prueba: afectaba a cualquiera que leyese un NPZ
sin haber corrido la etapa 7.

### Un rasgo mal construido, tambien cazado

`pie_adelantado` tomaba el maximo y multiplicaba por el signo despues, asi que para el tirador de
la derecha devolvia el pie **de atras** y con el centro sumado en vez de restado: rango medido 4,4
a 11,2 altos de caja contra −0,05 a 0,34 del de la izquierda.

---

## C · La puerta: **no se puede decidir todavia, y por que**

El criterio esta declarado y no se toca. Lo que falta es el dato: **la puerta necesita clips
extraidos, y extraer es lo caro**. Medido en esta maquina, por clip: etapa 1 **12,3 s**, etapa 4
**30,0 s**, etapa 7 **13,0 s** — unos **55 s**, o sea **56 h** para los 3.654 anotados que estan en
disco.

La extraccion quedo corriendo con el orden **barajado con semilla 20260922**, y eso no es un
detalle: **cualquier prefijo del recorrido es una muestra aleatoria del conjunto**, asi que la
puerta se puede cerrar con el n que haya y decir el n, en vez de con los clips de los primeros
torneos por orden alfabetico.

**No se da ninguna cifra de acierto**, porque un acierto sobre unas decenas de clips no distingue
nada de nada, y darlo con una advertencia al lado seria peor que no darlo. El propio careo tiene la
puerta puesta: por debajo de 60 clips no entrena, y entre 60 y 300 entrena pero avisa de que el
reservado se queda en unas decenas.

### Lo que si esta comprobado: la maquinaria entera corre

Forzando el entrenamiento con los 43 clips que habia (`--minimo 20`), solo para ejercitar el
camino:

```
particion por torneo: 10 torneos / 32 clips entrenamiento, 6 / 11 reservado
   parejas de tiradores a los dos lados: 0 (afectan a 0 clips del reservado)
   epoca 15 perdida 1,1947    epoca 30 perdida 0,4413
EL MODELO INVIERTE LA RESPUESTA AL ESPEJAR: 1,0000  (0 clips no invierten de 11)
```

O sea: el reparto por torneo funciona y no deja parejas a los dos lados, el entrenamiento
converge, la tabla de lineas base con acierto **y cobertura por separado** sale, y **la
antisimetria da 100 %**. Lo que falta es el denominador, no el codigo.

### Y la reanudacion, comprobada como se pidio

```
conjunto: 3654 clips, orden barajado con semilla 20260922
de los primeros 200 del orden, con etapas 1 y 4 hechas: 40
registro de auditoria: 28.995 lineas; tras BORRARLO el estado da 40  -> IGUAL
mismo orden al recalcular: True
```

El fichero de estado **no es la autoridad**: borrarlo entero no cambia lo que hay que hacer. (Y de
paso: 28.995 lineas para 40 clips completos era un defecto del registro, que escribia una linea por
clip pendiente y por vuelta; corregido para que solo apunte lo que se intento.)

Para cerrar la puerta:

```
python 01_extraccion/lotes.py --conjunto disco --etapas 1,4,7 --cuantos 1200
python 02_validacion/herramienta/verifica_etiquetador.py --cabeza initiated
```

---

## D · Colab

**`rutas.py`**, un solo sitio para las rutas, stdlib, derivando lo que puede de su propia posicion.
`hf_modelo()` nunca devuelve `None` — `snap()` si, y `from_pretrained(str(None))` pedia al hub un
repositorio llamado literalmente `"None"` y la traza acusaba a Hugging Face. Y elige la instantanea
con `sorted(...)[-1]` en vez de `next(iterdir())`, que con dos en cache cogia una arbitraria.

**`capacidades()` sustituye a `cual_interprete()`.** La pregunta vieja devolvia UNA etiqueta y se
comparaba con `==`; probaba `import ultralytics` primero, asi que en cualquier entorno con las dos
bibliotecas — todo Colab — contestaba `"ultralytics"` y **las etapas 4, 5, 7, 8 y 9 imprimian `SKIP`
para siempre**. Ahora cada etapa declara `necesita` (modulos) y `depende` (ficheros de entrada), y
dice cual de las dos cosas le falta.

**Las etapas se cablearon o se quitaron, y ninguna informa exito sin hacer nada:**

- **7** cableada. Era `GP.main(...) if hasattr(GP, "main")` y **`grafica_pista` no tiene `main`**:
  la guarda se tragaba la etapa entera y el llamante imprimia una linea con una cifra de sonido
  medido al lado. `fila_posicion` y `procesa` se sacaron del `if __name__ == "__main__"`. Corre.
- **8 se quito.** `_mano` importaba `cono_lib` y devolvia una frase **sin llamar a nada**, y ademas
  era un duplicado: `posicion.py` ya llama a `cono_lib.mano_armada`. Y su nota estaba mal — ver
  abajo.
- **9** cableada, con la ROI por sede y **columna `via`**: el 90,5 % es de la ROI aprendida (21
  sedes) y el 71,4 % del detector general, y **no se promedian**.
- **10** nueva, el etiquetador, sin coste de GPU.
- **5 sigue apagada**, y la razon principal no es el coste: es que **el 28 s/clip que justificaba
  las 67 h venia de `tiempos_fase2.csv`, que ya no existe**.

**`lotes.py`**: recorrido del manifiesto (no `os.walk`, que ya metio sable y espada en un informe
de florete), orden barajado fijo, `--desde/--cuantos` sobre ese orden, descarga del CDN, borrado
del mp4, y **el estado se lee del disco**: `estado_lotes.jsonl` es auditoria y se puede borrar
entero sin perder la reanudacion.

**`requirements-colab.txt`** (no habia ninguno) y **`colab_extraccion.py`**, el cuaderno en texto
para que se pueda leer en un diff — el `.ipynb` del repo viejo pesa 9,0 MB de JSON con la salida
dentro.

### La cifra de la mano armada estaba atribuida al peldano equivocado

La nota de la etapa 8 decia *"1.000 over 19 fencers of a held-out set, 74 % coverage"*, que son dos
cifras de dos duenos distintos pegadas. Lo correcto:

| criterio | cifra |
|---|---|
| el peldano geometrico solo (mano mas cerca del rival) | 0,741 (20 de 27), despues 0,831 sobre 59 |
| el peldano del pie, voto firme | 12 de 12 sin fallo, cubriendo 12 de 19; sin puerta 0,895 sobre 19 |
| **la cascada entera** `pie_firme -> encarado_accion -> mayoria_3` | **1,000 sobre los 19 reservados, 0,949 sobre 59** |

Y lo que corre en la etapa 7 **no es ninguno de los tres**: `posicion.py` llama a `mano_armada`
**sin `accion=`**, asi que el peldano `encarado_accion` no puede disparar y la cadena real es
`pie_firme -> hombro_firme -> nariz_dir_dudosa -> geometria_a`. Pasar `accion` moveria las cifras de
la pista, asi que queda **identificado y no hecho**. El docstring del llamante ademas seguia citando
*"4 de 4 tiradores"*, que el propio `cono_lib` habia retirado.

### El coste real, y por que el 83 % de YOLO no es la palanca que parecia

La tabla del playbook ponia las poses en **~1 s por clip**. Medido en una 3060 **libre**:

| etapa | s/clip | los 8.559 | los 106.432 de florete |
|---|---:|---:|---:|
| 1 decode + YOLO + pareja | **12,3** (17,3 con la tarjeta compartida) | 29 h | 364 h |
| 4 poses de 308 puntos | **30,0** | **71 h** | **887 h** |
| 7 pista y eventos (CPU) | **13,0** | 31 h | 384 h |
| 9 luces (CPU) | 1,9 | 5 h | 56 h |

**Las poses cuestan mas que YOLO.** El 83 % es *de la etapa 1*, que es el 22 % del total: partir
YOLO por la mitad ahorra un 11 % de la corrida. Y las 512 h del plan son la cifra de la fase 1; con
poses y eventos son del orden de **1.600 h** a la velocidad de hoy.

### CINCO intentos de acelerar sin cambiar el resultado, y los cinco medidos

El usuario pidio explicitamente optimizar **sacando lo mismo**. Con esa restriccion se probaron
cinco cosas y **ninguna aporta**. Van con su numero porque un camino cerrado con medida vale mas
que un camino abierto sin ella.

| intento | resultado | veredicto |
|---|---|---|
| **lotear el modelo de pose** | 1,17x, y mueve un keypoint usable **130 px** | rechazado, ver abajo |
| **varios procesos en paralelo** | 1 -> 99,9 clips/h, 2 -> 99,9, 3 -> 109,8 | **la GPU ya esta saturada con uno** |
| **cachear cuadros duplicados** (telecine 25->30) | **0 duplicados exactos** en 7.434 recortes | muerto: el re-encoding H.264 rompe la igualdad bit a bit |
| **atencion SDPA** | `_attn_implementation` ya vale `sdpa` | ya estaba |
| **no calcular las poses que se tiran** | solo el **1,9 %** se tira, y no se distinguen por la caja antes (alto p50 0,373 las tiradas contra 0,396 las buenas) | no hay umbral seguro |

`torch.compile` y los grafos de CUDA estan disponibles en la API pero **`triton` no esta instalado
en Windows**, asi que ahi no se puede medir; en Colab si, y queda apuntado.

**La conclusion, incomoda pero medida: con la restriccion de salida identica no queda margen en
esta maquina.** El forward del modelo de pose es el **80,9 %** del tiempo, y es un ViT de 425 M
parametros ya en fp16 con SDPA. Lo que queda -- saltar cuadros, bajar la resolucion, un modelo mas
pequenio -- **cambia el resultado**, y eso es una decision distinta, con su puerta de calidad, no
una optimizacion.

El perfilado por trozos, sobre `AbandonedBiodegradableCoqui` (325 cuadros, 650 recortes, con tres
procesos mas compitiendo):

    forward   80,9 %      post       7,0 %      preproc  6,6 %
    bgr2rgb    3,6 %      decodificar 1,3 %     resto    0,6 %

### El primer knob, medido y **rechazado**

Lotear el modelo de pose. Criterio declarado antes: tiene que ser un **no-op sobre el resultado** —
desplazamiento p50 < 0,02 px y **maximo < 0,5 px**, que es 1/4 del error de pose publicado.

| lote | s/clip | ganancia | p50 | p95 | max (puntos usables) |
|---:|---:|---:|---:|---:|---:|
| 1 | 31,87 | — | **0,0000** (dos pasadas) | 0,0000 | 0,0000 |
| 8 | 28,14 | 1,12x | 0,0021 | 0,0059 | — |
| 16 | 27,26 | **1,17x** | 0,0078 | 0,0241 | **129,92 px** |
| 32 | 27,50 | 1,15x | 0,0078 | 0,0233 | igual que 16 |

**Rechazado**, y por los dos lados a la vez: gana **1,17x** y no el 1,3-1,8x esperado — o sea que el
cuello no era la llamada por cuadro sino el preproceso en CPU — y a cambio mueve un keypoint usable
130 px. Con dos pasadas del mismo lote 1 el desplazamiento es **exactamente cero**, asi que no es
indeterminismo del modelo. El codigo se queda con `LOTE_POSE = 1` y con la medida escrita, porque en
la GPU de Colab el reparto puede ser otro y entonces hay que volver a pasarle **esta misma puerta**.

**Y un hallazgo de paso:** el modelo devuelve **coordenadas finitas y absurdas** para puntos de
confianza casi nula — hasta **8.553 px** fuera de un cuadro de 1.920 de ancho. No hacen dano porque
todo filtra por confianza, pero quien compare dos salidas sin filtrar medira eso.

---

## E · La pista

**El panel negro se quito, y el plan tenia razon sobre la contradiccion.** `panel_publico.py` ya
decia en su primera linea que el usuario habia decidido publicar el fotograma en los videos de capa;
`video_pista.py` se habia quedado atras. `PUBLICO = False` por defecto; `--publico` sigue existiendo.

**Y por debajo habia tres defectos, los tres encontrados mirando un cuadro del MP4:**

1. **Los dos paneles rotulaban metros de EJES DISTINTOS y ninguno lo decia.** `_panel_clip` ponia la
   coordenada de la **homografia** y `_panel_esqueleto` los metros medidos **entre los bordes de la
   tira**, que es lo que usan el trazado y el mapa. Cambiar de panel cambiaba en silencio la
   magnitud, y en el panel del clip el numero ni coincidia con el de al lado. Ahora los dos leen el
   mismo eje y **el rotulo dice cual** (`(strip)` / `(homography)`).
2. **El rotulo del tirador de la derecha se salia del panel y `putText` lo recortaba en silencio**:
   se leia `right -1.99 m (stri`. `_texto` ahora mide el ancho y lo mete dentro.
3. **Los rotulos de carril del mapa afirmaban lo contrario que el dibujo.** Ponian "arriba el
   naranja, abajo el verde", pero cuando el clip tiene tira rectificada la `y` del mapa **ya no es
   el carril fijo**: es la transversal medida. En el cuadro que lo enseno el verde estaba arriba.
   Ahora con carriles fijos se rotulan los carriles, y con transversal medida se rotula lo que hay.

`strip_location` sigue siendo **el primer control externo de la posicion en la pista** que el
proyecto va a tener, y sigue pendiente de que exista la cabeza 2.

---

## F1 · La mano armada contra la ficha federativa

El plan lo tenia como "identificado y no hecho". Se hizo, y da un resultado que cambia una cifra
publicada. El plan detallado esta en `PLAN_MANO_ARMADA.md`; lo medido:

**La verdad de referencia pasa de 59 a 967 tiradores, sin anotar nada.** `01_extraccion/codigo/
fie.py` baja el censo (66.544) y el perfil de los nuestros. `hand` viene poblado en **967 de 967**,
y **el 25,7 % son zurdos** -- la banda de la esgrima de elite es 25-35 % contra el ~10 % de la
poblacion general, asi que el campo trae dato y no un valor por defecto.

**Una linea base que no existia: 0,743.** Si tres de cada cuatro son diestros, contestar siempre
"la diestra" acierta eso sin mirar la imagen. Con ella al lado, **el peldano geometrico suelto
(0,741 sobre 27) no bate al azar informado**.

**El mapeo A/B contra R/L, medido y no supuesto:** `A` = diestra, con **0,920** de acuerdo contra
**0,080** de la hipotesis contraria (n=50). El convenio de paridad del modelo queda confirmado por
medida.

**Las dos verdades coinciden 0,920 (46 de 50).** Discrepan 4, y uno de los clips discrepa en **los
dos lados** -- que seria la firma de un intercambio de tiradores. Se comprobo: cambiando los
nombres de sitio tampoco cuadra (0/2 en las dos configuraciones), asi que **no es un intercambio,
son dos errores**. El control existe y esta ejecutado; no fabrico un hallazgo que no habia.

**Y la cifra que hay que corregir.** Contra los 235 tiradores con ficha Y poses extraidas:

| peldano | acierto | n | cobertura |
|---|---:|---:|---:|
| `nariz_dir_dudosa` | 0,852 | 122 | 0,519 |
| `pie_firme` | 0,912 | 113 | 0,481 |
| **la cascada entera** | **0,881** | 235 | 1,000 |
| azar informado (siempre la diestra) | 0,743 | — | — |

La cascada bate al azar informado por **+13,8 pp**, pero da **0,881**, no el **0,949** publicado.
No es que el 0,949 fuera falso: es de **otro denominador y otra verdad** -- 59 tiradores anotados a
ojo contra 235 con ficha federativa. Las dos cifras son ciertas y la que hay que citar de ahora en
adelante, por tamano e independencia, es la de 235. Y confirma por medida lo que la lectura del
codigo decia: los unicos peldanos que disparan son `pie_firme` y `nariz_dir_dudosa`, porque
`posicion.py` llama sin `accion=`.

## Lo que queda abierto, por orden

1. **Cerrar la puerta [C].** Necesita extraccion; el comando esta arriba. Es lo unico que decide si
   se bajan los 2.202 que faltan.
2. **Pasar `accion` a `mano_armada`** en la etapa 7, como cambio declarado con su antes y su
   despues: hoy corre una cascada que no es la que tiene publicadas las cifras.
3. **El error de frontera de los eventos.** `02_validacion/pista/anotar/fronteras.csv` sigue con
   **21 acciones y 0 anotadas**, y sin eso la primitiva `contactos` no se puede adoptar.
4. **La mano armada contra la FIE.** La API responde (66.543 tiradores) y el catalogo trae los dos
   nombres de los 8.555: seria una verdad de referencia **28x mayor** que los 59 anotados a ojo.
5. **Los 13.261 clips con el arma sin declarar**, que nadie ha mirado y que podrian subir el corpus
   de florete.
6. **Re-medir la cadencia nativa**: 8.503 de 11.211 clips a 30 fps son telecine 25→30.

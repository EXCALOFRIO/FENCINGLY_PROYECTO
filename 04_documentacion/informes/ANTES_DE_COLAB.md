# Antes del entrenamiento final en Colab

Escrito el 2026-09-09 tras la noche de medidas. Tu marco era «no me quiero ir a Colab y hacer el
entrenamiento final sin haber probado todo esto». Esto es lo que se probó y lo que **hay que fijar antes**,
porque hay dos cosas que harían que el número final de Colab no fuera comparable con nada.

## Bloqueante 1: RESUELTO. Este es el intérprete canónico

> **`<base>/scratchpad/fz/.venv-gpu/Scripts/python.exe`**
> torch 2.14.0+cu126, numpy 2.5.2, pandas 3.0.5, scipy 1.18.1, RTX 3060, backend de atención `math` forzado.
> **Su número: 70,04 % ponderado en dejar-un-torneo-fuera** (semillas 7/11/23 = 69,74 / 70,40 / 69,97),
> sobre **3.059 pares, 19 sedes, n = 3.047**, que es lo que la red puede medir.

El motivo de que sea ése y no otro, y es lo que lo hace no arbitrario: **es el único que reproduce
`producer_sha256 = 0b87d014` y además tiene CUDA.** `fz/.venv` reproduce la huella pero es CPU y va a
5,2 s/paso, o sea 29 h por brazo. `scratchpad/cuda/.venv` tiene numpy 2.5.3 y da la huella equivocada.
Así que **el intérprete de la red y el de los artefactos son el mismo**, que es justo la propiedad que hace
comparables las dos mitades del sistema.

**Para Colab: clava esas versiones y declara el número junto al resultado.** Si el cuaderno instala otro
torch, lo que salga no se puede comparar con nada de este informe.

### Y la caída de cabecera, con su matiz

Las tres semillas, determinista contra archivado: −1,18, −0,42 y −1,12 pp, media **−0,91 pp**. Sistemática,
las tres negativas, pero **no uniforme**. Así que **no hay «un punto» que restar en bloque: hay que
remedir**. Lo que sí es un subproducto valioso: el recorrido real entre semillas es **0,66 pp**, no los
1,35 que se citaban. Las barras de error de la red se estrechan a la mitad en cuanto se fija el kernel.

## Cómo se llegó a eso (el diagnóstico, ya cerrado)

Es el hallazgo más incómodo de la noche. La misma receta, la misma semilla y el mismo código dan:

| | acierto |
|---|---|
| archivado (`torch 2.11+cu128`, kernel aproximado) | 70,92 % |
| aquí (`torch 2.14.0+cu126`, kernel aproximado) | 70,13 % |
| aquí (`torch 2.14.0+cu126`, kernel exacto `math`) | 69,74 % |

**Dos tercios de la diferencia son el entorno y un tercio el kernel de atención.** Las cifras de cabecera de
la red se produjeron con un intérprete que aquí ya no existe, así que **no son reproducibles**, y ningún
arreglo de determinismo lo cambia.

Consecuencia para Colab: si el cuaderno instala otra versión de torch, el número que salga **no se puede
comparar** con el 73,88 % ni con nada de este informe. Hay que clavar la versión en el cuaderno y
declararla junto al resultado.

## Bloqueante 2: la huella de las medidas depende de numpy, pandas y scipy

`producer_sha256` **no es el hash de los 20 ficheros de `MEASUREMENT_SOURCES`**: hashea el diccionario
entero del productor, que incluye `runtime_versions`. Comprobado en esta máquina, con código idéntico:

| intérprete | numpy | huella |
|---|---|---|
| `.venv` y `.venv-gpu` | 2.5.2 | **`3e1dbe0a…9827`** |
| `scratchpad/cuda/.venv` | 2.5.3 | `0645f2c8…8f1f` |

**Actualizado el 2026-09-09 a las 10:49.** Antes del parche de alineación de la ventana de guardia estos
valores eran `0b87d014…3c4f` y `874c1ca9…6533`. El productor de movimiento pasó de `994c874b…2658` a
`6c9fa66b…3ce7` en el mismo cambio, porque `measurement_artifacts.py` está en **las dos** listas de
fuentes. **No confíes en un hash copiado: mídelo con la línea de abajo**, porque el valor se mueve cada vez
que se toca uno de los 20 ficheros. El vigente está siempre en `AGENTS.md`.

**Un parche de numpy invalida los 23.091 artefactos y el fallo es un desempate de caché silencioso**, sin
error. Si Colab materializa medidas, tiene que llevar numpy 2.5.2, pandas 3.0.5 y scipy 1.18.1. Se
comprueba con una línea: `measurement_producer()["producer_sha256"]`.

## Ya resuelto: Colab no necesita decodificar vídeo

`scratchpad/vorticidad/recortes/` tiene **4.865 clips, 6,4 GB, cero errores**, con `frames` uint8
(T,128,128,3), caja, manos, `dt_s` y `fps_real` por clip. Recalcular el flujo desde los recortes reproduce
la ruta de vídeo **bit a bit** a 394 ms/clip contra 1.250. Está respaldado en
`D:\FENCINGLY_respaldo_2026-09-09`.

## NO cambies el suavizado. Pasó todas las pruebas estadísticas y aun así no cuenta

**Corregido a las 05:00.** Durante veinte minutos esta nota decía que `ls = 0,30` era la mejora a aplicar.
Medido en el conducto real, **no lo es**, por tres motivos que ninguna prueba estadística podía ver:

| comparación (0,30 menos desplegado) | delta | IC 95 % |
|---|---|---|
| red sola, convenio de ablación | +1,03 | [+0,29, +1,81] |
| **red sola, convenio desplegado** | **+0,80** | **[−0,16, +1,85]** ← cruza el cero |
| **mezcla al 0,65, sin calibrar** | **+0,24** | [−0,53, +1,02] |
| **mezcla al 0,65, con Platt** | **+0,38** | [−0,24, +1,06] |

1. **El estimando.** El banco de ablaciones promedia el *acierto por clip* de tres modelos y luego resta.
   El artefacto promedia las *probabilidades* y umbraliza una vez. No son la misma cantidad, y con el
   convenio desplegado el intervalo ya no excluye el cero.
2. **El peso de la mezcla.** Al 0,65 contra un árbol que no cambia, y con correlación 0,651 entre cabezas,
   +0,80 en un componente queda en **+0,24 a +0,38** en el artefacto. Bajo el suelo.
3. **La calibración ya se estaba cobrando.** El «ECE cinco veces mejor» estaba medido con una semilla y sin
   calibrar. En el conducto real, promediar tres semillas ya arregla casi todo (0,1137 → 0,0603) y el Platt
   que ya existe arregla el resto (→ 0,0258). La ventaja del 0,30 queda en **1,15x**. Y encima **se pasa de
   largo**: la pendiente de Platt va de 0,612 (sobra confianza) a **1,209** (falta). El óptimo real está
   entre 0,05 y 0,30, no en 0,30.

**La lección, que vale para cualquier candidato futuro:** una mejora hay que medirla **en el estimando, en
la etapa del conducto y en la población donde se va a desplegar**. Esta pasó el suelo de ruido, pasó
Bonferroni sobre siete brazos, pasó tres semillas y pasó una comprobación de degeneración. Ninguna de esas
cuatro habría detectado un convenio de promediado, un peso de mezcla ni un calibrador aguas abajo.

## Lo que sí queda en pie del barrido

Lo que sigue siendo cierto y es nuevo: estaba congelado en **0,05**, nunca se había barrido, y **está en el
mínimo de su propia curva de acierto**. Eso vale como resultado sobre la red. No como cambio de producto.

```
ls        acierto   delta pp        IC 95 %      p       clips dif  veredicto
0,00      70,40 %    +0,62   [-0,20, +1,49]   0,074       123    bajo suelo
0,025     70,07 %    +0,30   [-0,14, +0,71]   0,112        77    bajo suelo
0,05      69,77 %      —           —            —          —     DESPLEGADO <- el minimo
0,10      69,97 %    +0,20   [-0,61, +1,04]   0,339        98    bajo suelo
0,20      70,53 %    +0,75   [-0,23, +1,68]   0,067       181    bajo suelo
0,30      71,15 %    +1,38   [+0,11, +2,67]   0,015       234    supera
0,40      71,28 %    +1,51   [+0,19, +2,78]   0,014       282    supera
0,50      71,41 %    +1,64   [+0,53, +2,79]   0,002       320    supera
nulo kernel 70,13 %  +0,36   [-0,56, +1,27]   0,235       327    bajo suelo
```

**Con las tres semillas, `ls = 0,30` da +1,03 pp [+0,29, +1,81], p = 0,003.** Pasa las tres condiciones:
el suelo de 0,8; Bonferroni sobre 5 brazos (0,010); y Bonferroni sobre 7 (0,0071), que es el umbral honesto
porque la familia creció al añadir 0,40 y 0,50.

**Cítalo con su desglose, porque el efecto no es homogéneo:** +1,38 (semilla 7), +1,21 (11) y **+0,49**
(23). Y nótese que se encogió de +1,38 a +1,03 al pasar de una semilla a tres, que es exactamente lo que
las salvedades predecían.

### Y el resultado no es el que se buscaba

Se venía a probar trece brazos de pérdidas robustas (GCE) contra el ruido de etiqueta, que era la premisa
del documento de mejoras. Lo que ha salido es **un hiperparámetro que nadie había barrido**. Más barato y
más comprobable.

## Una vara de medir que conviene conocer antes de creerse cualquier delta

**Cambiar solo el backend de atención, con la misma receta, la misma semilla y el mismo intérprete, produce
un «delta» de +0,36 pp con 327 clips decididos distinto.** O sea que **no cambiar nada del modelo** da un
número mayor que tres de los cinco efectos del barrido de arriba.

Es el primer nulo **neuronal** del proyecto; el suelo de 0,8 pp descansaba hasta ahora en cinco nulos todos
tabulares. Úsalo como referencia: cualquier mejora de la red por debajo de +0,4 pp es indistinguible de
recompilar.

El arreglo que lo elimina es `fija()` en `prio9_perdida/determinismo.py`: da
`max |p1 − p2| = 0,000e+00` entre procesos distintos, sin coste de tiempo, y ya hay un `RuntimeWarning`
que salta si se entrena en CUDA sin él.

## Una trampa que se activa justo con los clips nuevos

`HistGradientBoostingClassifier(early_stopping='auto')` enciende la parada temprana por encima de **10.000
filas de entrenamiento**, y con ella un corte de validación aleatorio que hace que las semillas importen.

- Un pliegue de validación cruzada entrena 5.672 filas: **no le afecta**, ni con los clips nuevos.
- **El ajuste sobre la población completa sí**: 4.958 pares son 9.916 filas con espejo, un 1 % por debajo.
  Los 158 pares nuevos de dos luces lo cruzan (10.232).

Los dos sitios de población completa (`e12_final.py`, que es la ruta de exportación del artefacto, y
`e10_afina.py`) están ya fijados con `early_stopping=False`. **Si el cuaderno de Colab construye su propio
estimador, tiene que fijarlo también**, o su comportamiento cambiará según cuántos clips se extrajeron.

## CORREGIDO: el reentrenamiento tiene +30,4 % de datos, no +3,2 %

**Esta sección decía +3,2 % y estaba mal por un factor de nueve en la atribución.** La premisa era que solo
los clips con `touch=double` pueden dar pares de dos luces. **Medido, esa etiqueta no informa:**

| etiqueta del manifiesto | extraídos | dos luces | tasa |
|---|---|---|---|
| `double` | 703 | 158 | **22,5 %** |
| `left` / `right` | 5.517 | 1.349 | **24,5 %** |

Casi idénticas. La población de dos luces escala con **cuántos clips se extraen**, no con la etiqueta.

**Pares nuevos aprovechables: 1.507 → +30,4 %** sobre los 4.958 actuales. Y el reparto es lo que desmonta
la premisa: **158 (10,5 %) del lote de dobles y 1.349 (89,5 %) del lote de «resto»**, que es justo el que
yo describí como «no mueve el modelo, solo completa fichas». Si se hubiera saltado por ese razonamiento,
el reentrenamiento habría perdido el 89,5 % de sus datos nuevos.

*Nota de método: al intentar verificar esto agrupando todos los bancos de trabajo me salió 31,3 % para
`left`/`right`, pero ese número está contaminado por experimentos anteriores que escogieron clips de dos
luces a propósito (uno al 91 %, otro al 98 %). La comparación válida es solo entre los dos lotes que barren
el manifiesto en orden, y son los 22,5 % contra 24,5 % de arriba.*

**Y esto cruza el umbral de la parada temprana con margen:** 4.958 + 1.507 = **6.465 pares = 12.930 filas
con espejo**, contra el umbral de 10.000. Así que el `early_stopping=False` que se fijó en
`prio2/e1_tabular.py` y `prio2/e10_afina.py` **ya no es una precaución, es necesario**: sin él, el ajuste
sobre la población completa activaría un corte de validación aleatorio y las semillas empezarían a importar
en silencio. Si el cuaderno de Colab construye su propio estimador, tiene que fijarlo también.

## Qué esperar del reentrenamiento

Con +30,4 % de datos de entrenamiento hay motivo para esperar algo, pero **no un salto legible**: con la
potencia de este corpus un efecto real de +1,6 pp es indetectable, y hacen falta 12 a 20 campeonatos más
para tener el 80 % de potencia sobre ese tamaño de efecto.

Y las fichas, que era el otro objetivo y ese sí se movió mucho: la cobertura del archivo pasó de 68,6 % a
**86,7 %**, la mediana por tirador de 74,4 % a **88,2 %**, los tiradores al 100 % de 14 a **47**, y los que
estaban por debajo del 50 % de 56 a **18**.

## Lo que la noche cerró, para no reintentarlo en Colab

Seis vías, ninguna llega al suelo de 0,8 pp: vorticidad del flujo óptico (y con ella **SEA-RAFT/CUDA no
está indicado**), transformador de vídeo VideoMAE, latente antisimétrico, centro de masas, tirón y la regla
t.83. Más la simetrización de la ventana con tres geometrías de ancla.

El techo **no está** en la representación (VideoMAE predice la sede al 97 %), **ni** en el ruido de etiqueta
(refutado, y el suavizado apunta al revés), **ni** en la geometría de la ventana. Está en el orden de las
luces, sobre todo en las 256 columnas agregadas y congeladas, y una parte de eso **no es un atajo sino la
regla de esgrima**: quien arranca primero tiende a encender primero.

Lo único geométrico sin agotar: **reextraer con `POST_S` mayor de 0,3 s**. En el 3,12 % de los clips la
segunda luz cae fuera de la ventana guardada. Es cambio de adquisición, no de modelo.

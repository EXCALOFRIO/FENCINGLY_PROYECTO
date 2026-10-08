# La posición en la pista: dónde está de verdad el fallo

**2026-09-23.** Este informe corrige el diagnóstico de `PISTA_DIAGNOSTICO_2026-09-23.md`, escrito
hoy mismo unas horas antes. Aquel decía que el problema era el **emparejamiento de marcas**. No lo
es.

---

## 1. Lo que cierra el hilo anterior

`ancla_v2.py` eliminó el 27 % de patrones físicamente imposibles del emparejamiento (dos rayas
distintas asignadas a la misma marca del reglamento). Eso es cierto y se mantiene. Lo que había que
medir era si la **posición** mejoraba. Medido sobre los mismos 20 clips con etiqueta humana:

| | acierto por zonas | pendiente del ajuste |
|---|---:|---:|
| emparejamiento de hoy | 0,250 | −0,314 |
| `ancla_v2` | **0,250** | **−0,314** |
| contestar siempre `middle` | 0,430 | — |
| techo de acuerdo humano | 0,619 | — |

Idénticos hasta el segundo decimal, celda por celda. **El emparejamiento no era el cuello de
botella**, y arreglarlo no movió nada porque el fallo está tres pasos más arriba.

---

## 2. El fallo: un control que seleccionaba sobre su propio resultado

`pista.control_estatura` es el control C1, *"el que decide"*. Reproyecta la estatura de los dos
tiradores y comprueba que caiga en 1,60–2,00 m. La línea 1649 hace:

```python
h = altura_m(hg, pie, cab)
if h is not None and 0.3 < h < 5.0:      # <-- aquí
    alturas[lado].append(h)
```

Tira las alturas fuera de 0,3–5,0 m y **después publica la mediana de las que quedan**. Medido sobre
11 clips con homografía, con todas las observaciones y sin filtrar:

| clip | supervivencia al filtro | h_p50 **filtrada** | h_p10 real | h_p90 real |
|---|---:|---:|---:|---:|
| AbandonedAntiqueGalapagostortoise | 0,14 | 2,88 | **−11,18** | 16,27 |
| AbandonedBiodegradableCoqui | 0,28 | **1,76** ✓pasa | **−11,18** | 28,39 |
| AbsoluteGreenBluejay | 0,26 | 3,37 | −8,60 | 15,87 |
| AbandonedUnderstatedGalago | **1,00** | 1,94 | 1,66 | 2,23 |
| AcceptableArtisticCutworm | **1,00** | 1,43 | 1,31 | 1,70 |

- supervivencia al filtro: **p10 0,13 · p50 0,34 · p90 1,00**
- clips donde sobrevive menos de la mitad: **6 de 11**

`AbandonedBiodegradableCoqui` **aprueba** el control C1 con un 1,76 m obtenido tirando el 72 % de
las observaciones, cuando esa homografía reproyecta tiradores entre −11 y +28 m.

Es la disciplina que el proyecto ya tiene escrita —*no se evalúa en un subgrupo definido por el
resultado*— aplicada al revés: el filtro define el subgrupo con la misma magnitud que se va a medir.

**Consecuencia:** los «10 de 78 clips métricamente sanos» hay que volver a contarlos, y el 13 % es
un techo optimista, no pesimista.

---

## 3. Estaturas negativas: no es escala, es la estructura proyectiva

Una escala mal puesta **multiplica**; no cambia el signo. Un tirador que reproyecta a −11 m significa
que el punto de fuga vertical está del lado equivocado. Y una dispersión de −11 a +28 **dentro del
mismo clip** significa que la homografía no describe esa cámara en absoluto.

El discriminador, buscado sobre los 78 clips del censo, es uno solo. No es el número de marcas (4,5
frente a 4,0), ni las identificadas (4,0 frente a 4,0), ni el ángulo de fuga vertical (0,24 frente a
0,17), ni el ancho en píxeles (132 frente a 131). Es **la focal**:

    f_rel de los buenos   0,91          f_rel de los malos   0,71
    rango observado       0,30 a 6,00   ambos son los topes del clamp

`f_rel = 0,30` es un campo horizontal de **118°** —un ojo de pez— y `f_rel = 6,00` son **9,5°**, un
teleobjetivo con el que no caben dos tiradores separados 4 m. Ninguna de las dos existe en una
retransmisión, y la focal se está yendo a los dos topes.

**Por qué sale mal.** `_f_y_v2` elige la focal por el ángulo con el que las marcas transversales
apuntan a V2. Es correcto pero débil: las marcas son cortas, casi paralelas y ruidosas. El propio
módulo lo sabe y guarda `curva_f` *«que es la medida de lo bien determinada que está: si el residuo
no sube, la focal no está medida, está elegida»*. Y mientras tanto la medida fuerte del clip —unas
100 observaciones de una cantidad conocida— solo se miraba al final como aprobado/suspenso.

---

## 4. El arreglo: la constancia elige la forma, la mediana elige la escala

`01_extraccion/codigo/focal_fisica.py`. Un tirador mide lo mismo esté donde esté en la pista, así
que con la homografía correcta su estatura reproyectada es **constante** mientras camina de un
extremo al otro; con la focal equivocada varía con la posición.

    la DISPERSIÓN de la estatura   ->   elige la FOCAL      (forma proyectiva)
    la MEDIANA de la estatura      ->   elige la ESCALA     (ancho de pista)
    el ANCHO DEDUCIDO              ->   control independiente, 1,50-2,00 m del reglamento

Separarlas importa porque la dispersión es **insensible a la escala**: elegir la focal con ella no
es circular con el paso siguiente. Y el ancho de pista deja de imponerse por decreto —hoy vale 1,50
siempre— y pasa a ser una predicción que el reglamento FIE puede desmentir.

**El precio, y hay que decirlo.** La estatura reproyectada deja de ser un control independiente:
pasa a ser el objetivo. Comprobarla después sería el mismo error que el cono (*«la hoja está a 0
grados de su propio centro»*). Lo que queda independiente: el ancho deducido contra el rango FIE, el
patrón de marcas contra el reglamento, y `strip_location`.

**Criterio declarado antes de medir:** se adopta si sube el número de clips métricamente sanos **y**
la posición mejora contra `strip_location` (hoy 0,250 de acierto, pendiente −0,314). Lo primero sin
lo segundo no vale.

---

## 4bis. La causa real de la posición: dos detectores que no se hablan

Rastreando por qué el A/B del emparejamiento salió idéntico célula por célula, la cadena real de la
posición resultó ser otra que la que se estaba arreglando:

```
posicion_entre_bordes  ->  tira_rectificada  ->  ancla_zonas  ->  m_por_col, q_m  ->  METROS
                                                      ^
                                                      solo FRONTERAS DE COLOR
```

`posicion_entre_bordes` **no usa la homografía en absoluto** y el ancho lo tiene fijo a
`ANCHO_NOMINAL = 1,50`. Los metros a lo largo salen enteros de `ancla_zonas`. Y en `pista.py` hay
dos detectores de marcas que nunca se comunican:

| | qué detecta | plantilla | alimenta a |
|---|---|---|---|
| `ancla_zonas` | fronteras de color t.19 | (−7, −5, +5, +7) | **la posición** |
| `identifica_marcas` | rayas pintadas | (−7,−5,−2,0,+2,+5,+7) | la homografía, que la posición no usa |

O sea que **las rayas pintadas —centro y puesta en guardia— no intervienen en la posición en ningún
momento**, aunque están detectadas, afinadas y emparejadas. Eso explica los dos resultados nulos del
día: ni `ancla_v2` ni la focal podían mover la posición, porque ninguno de los dos está en su camino.

Y el docstring de `ancla_zonas` ya decía lo que eso cuesta: encaja en 13 de 20 clips, pero **en 11
de esos 13 solo con DOS fronteras**, y con dos observaciones y dos parámetros el ajuste es exacto
por aritmética, se acierte o se falle de frontera.

### `ancla_unificada.py`

Un solo ajuste afín columna → metros con las **dos** familias a la vez, contra la plantilla completa
de siete posiciones, con asignación inyectiva y tolerancias distintas por familia (35 cm para una
frontera de color, 20 cm para una raya pintada, porque no son igual de precisas). Si hay
observaciones de las dos familias, la solución **tiene que** explicar al menos una de cada.

Primeras tres comparaciones:

| clip | ancla de hoy | ancla unificada |
|---|---|---|
| AbandonedUnderstatedGalago | 3 marcas, k=0,0180 | 4 marcas (1 color + 3 raya), k=**0,0110**, loo 10,5 cm |
| AbandonedBiodegradableCoqui | 2 marcas, **sin control** | 4 marcas (3+1), loo 36,7 cm → se declara no controlada |
| AcceptableArtisticCutworm | **falla** | 3 rayas, k=0,0116, loo **7,8 cm** |

La escala de Galago cambia un **63 %**. Y por primera vez hay deja-una-fuera en clips donde antes no
existía, que es lo que permite decir «esta x no vale» en vez de publicarla igual.

**Pendiente de la única medida que decide**: la posición resultante contra `strip_location`, con el
criterio ya declarado (batir 0,430 y subir la pendiente desde −0,314).

---

## 5. Sobre las alturas oficiales de la FIE

Idea del usuario: usar la estatura publicada de cada tirador en vez de un 1,75 genérico. **La fuente
no sirve**: medido al construir la mano armada, `height` viene relleno en **7 de 60** perfiles del
censo (`hand`, en cambio, en el 100 %). Y aunque sirviera, atacaría el error equivocado: mejoraría
la escala de un ~4 % a un ~1 % cuando el error de hoy es del 400 %.

Lo que sí se ha incorporado de esa idea es la otra mitad: **las dimensiones reglamentarias de la
pista como restricción**, no como número impuesto.

---

## 6. Pendiente

1. Medir `focal_por_constancia` sobre los clips ya extraídos: cuántos pasan a ser métricamente sanos.
2. Medir la posición resultante contra `strip_location` con el criterio de arriba.
3. Si pasa: arreglar `control_estatura` (quitar el filtro, publicar dispersión y signo), cablear
   `focal_fisica` en `pista.py` y rehacer solo la etapa 7. **No antes**: la extracción está
   corriendo y cambiar el criterio a mitad dejaría el corpus con dos convenios mezclados.
4. Si no pasa: el problema está aún más arriba, en la detección de los dos bordes, y hay que
   anotarlos a mano con `tablero_marcas.py` para tener verdad por marca y no por clip.

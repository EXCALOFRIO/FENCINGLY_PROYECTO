# Rasgos explicitos de esgrima + clasificador pequeno: protocolo DECLARADO ANTES de medir

Escrito el 2026-10-09, antes de calcular un solo rasgo sobre etiquetas. Lo que cambie despues de ver
resultados se apunta abajo como cambio, con fecha, y no reemplaza a esto.

## La pregunta

`04_documentacion/investigacion/2026-10-09_esgrima_novedades.md` recomienda en primer lugar "tokens
explicitos de inicio / cierre de distancia / orden brazo-pie + clasificador pequeno" (FERA v5,
arXiv 2509.18527: 0,624 con un Random Forest sobre tokens de pose). Nuestro etiquetador (GRU sobre
64 pasos del clip entero, sin ancla temporal) da 0,615 en `initiated`. El GRU **no sabe donde esta
el tocado**: ve el clip remuestreado. La hipotesis es que rasgos ANCLADOS AL CUADRO DEL TOCADO, con
sentido de esgrima, separan mejor quien inicia y, sobre todo, a quien da el punto el arbitro.

## Las dos tareas (las mismas filas y repartos que el resto del proyecto)

1. `initiated` (fotl / fotr / neither): `etiquetador.reune(cabeza="initiated")`, reparto
   `etiquetador.particion_por_torneo` con su semilla (20260922).
2. `prioridad` (gana izquierda / derecha, florete y sable con las DOS luces de color y `touch`
   izq/der): `sonda_vjepa.filas_prioridad()`, el mismo reparto sobre esas filas.

## Los rasgos (fijados ahora)

Ancla `a` = el cuadro del tocado como en `vjepa.ancla` (`zona_` > 0, si no `lights_`, si no el
ultimo cuadro util menos 2,2 s, con su `via`). Por tirador y EN SU MARCO DE ATAQUE (adelante =
hacia el rival; espejar = intercambiar los dos tiradores), con velocidades en altos de caja por
segundo y tiempos en segundos ANTES del tocado. None (NaN) donde no se pueda medir; la cobertura de
cada rasgo se da.

- avance: velocidad media hacia delante en [a-1, a], [a-2, a-1], [a-3, a-2] s; fraccion de cuadros
  avanzando y retrocediendo en el ultimo segundo; aceleracion final (ultimos 0,3 s menos el resto
  del ultimo segundo); inicio del ultimo avance continuo antes del tocado.
- brazo: extension hacia el rival en el tocado; maximo en el ultimo segundo; cuando cruza el 80 %
  de su maximo final; pico de velocidad de extension y cuando.
- orden brazo-pie: (cuando extiende el brazo) - (cuando empieza el ultimo avance del pie delantero).
- eventos de la etapa 7: segundos antes del tocado del ultimo `marcha`, `romper`, `fondo`, `brazo`
  y cuantos de cada en los ultimos 2 s.
- distancia (de la pareja, par al espejar): distancia relativa a la talla en a-1, a-0,5 y a; y la
  PARTE del cierre de distancia del ultimo segundo que pone cada tirador (impar).
- pista: metros hasta su propia linea de fondo en el tocado (`zona_`, NaN sin homografia).
- Se excluyen a proposito las LUCES: el GRU no las ve, y meterlas mezclaria dos preguntas.

## El clasificador (fijado ahora, sin rejilla)

`HistGradientBoostingClassifier` de scikit-learn (admite NaN): `max_iter=300`,
`learning_rate=0.05`, `max_leaf_nodes=15`, `l2_regularization=1.0`, `min_samples_leaf=20`,
semilla fija, sin parada temprana. **Antisimetria impuesta**: se entrena con cada clip y su espejo
(rasgos intercambiados, etiqueta cambiada) y la prediccion es la media de `p(x)` y de `p(x')` con
izquierda/derecha cambiadas. El control del espejo tiene que dar 1,0000 (los empates exactos
izquierda = derecha se cuentan aparte).

## La comparacion y el criterio

- Base: el GRU del etiquetador, `etiquetador.entrena` sobre las MISMAS filas (en `prioridad`,
  prediccion = signo de `u`).
- **Se adopta si** rasgos - GRU >= +0,02 en acierto del reservado **y** el IC95 por bootstrap
  agrupado por torneo (2.000 remuestreos) no cruza cero **y** espejo = 1,0000. Una decision por
  tarea.
- Informativo (no decide): la fusion GRU + rasgos (logistica impar ajustada en la particion INTERNA
  del entrenamiento), por arma, y la importancia de cada rasgo por permutacion sobre la particion
  interna (las palancas).
- `prioridad` tiene ~270 clips de reservado: el IC sera ancho, y se dice.

## Resultados (2026-10-09, `02_validacion/herramienta/rasgos_explicitos.py`)

5.791 clips con rasgos (los mismos que `etiquetador.reune`, asi que el reparto es el canonico).
Cobertura: ~0,95 en velocidades, brazo y distancia; 0,53 `t_inicio_avance`, 0,45 `t_inicio_pie` y
`brazo_menos_pie`, 0,60 `m_a_su_fondo` (solo con homografia), 0,32-0,86 tiempos de eventos.
Una correccion de codigo sin efecto sobre las cifras: la fusion de `prioridad` cascaba con 2 clases
(`partes_gbm`); se arreglo y se repitio SOLO `prioridad`, con los mismos rasgos en cache.

### `initiated`: PASA

    reservado 1.384 clips / 42 torneos (entrenamiento 4.152 / 105)
    clase mayoritaria                      0,461
    quien extiende el brazo primero        0,499  (cobertura 0,936)
    GRU (etiquetador, mismos datos)        0,590
    RASGOS explicitos (GBM)                0,628   +0,038  IC95 por torneo [+0,013, +0,073]
    fusion GRU + rasgos (informativo)      0,644   +0,054  IC95 por torneo [+0,035, +0,082]
    espejo 1,0000 (25 empates exactos izq = der, 1,8 %, contados aparte como se declaro)

Por arma: espada 0,589 -> 0,646 (n=209), florete 0,583 -> 0,614 (n=1.040), sable 0,644 -> 0,704
(n=135). **El GRU con los datos completos de hoy da 0,590, no el 0,615 del 08/10** (otro conjunto,
otro reparto): la comparacion valida es la de esta tabla, en el mismo reparto.

**Palancas** (caida de acierto al permutar, particion interna n=1.208): la PARTE del cierre de
distancia que pone cada uno en el ultimo segundo (+0,030), su velocidad hacia delante en el ultimo
segundo (+0,019), la aceleracion final (+0,016), la velocidad entre -2 y -1 s (+0,012), la
extension maxima del brazo (+0,007). O sea: **quien inicia es quien pone la distancia al final**.

### `prioridad` (quien gana con dos luces): NO PASA

    reservado 341 clips / 17 torneos (entrenamiento 735 / 40)
    clase mayoritaria                      0,522
    gana el que inicio (etiqueta HUMANA)   0,644  (cobertura 0,947)
    GRU (etiquetador, mismos datos)        0,642
    RASGOS explicitos (GBM)                0,537   -0,106  IC95 [-0,153, -0,034]
    fusion (informativo)                   0,604   -0,038  IC95 [-0,079, +0,000]
    espejo 1,0000 (5 empates)

Con 735 clips de entrenamiento el GBM de 60 rasgos no generaliza; el GRU si (0,642, igual que saber
quien inicio por la etiqueta humana). Palancas de la particion interna (n=245, ruidosas): cuando
llega el brazo al 80 % de su extension, la extension maxima, el inicio del ultimo avance, el orden
brazo-pie. Es exactamente lo que dice el reglamento del florete (el brazo que se extiende primero y
amenaza tiene la prioridad), pero con estos datos no da para un clasificador.

### Veredicto

- `initiated`: **ADOPTADO** como cabeza o como fusion con el GRU (la fusion es la mejor cifra del
  proyecto en esta tarea, 0,644, pero es informativa: lo que paso el criterio es el GBM solo).
- `prioridad`: **NO**. La medida buena necesita los ~21.000 clips de florete con dos luces.
- Coste: ~0,05 s de CPU por clip a partir de `poses_` y la etapa 7; no toca la GPU ni el video.

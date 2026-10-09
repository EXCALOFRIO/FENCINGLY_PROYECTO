# Camara de tripode para la zona del tocado (etapa 11): protocolo DECLARADO ANTES de medir

Escrito el 2026-10-09 (tarde), antes de ejecutar ningun ajuste del modelo nuevo contra la etiqueta.
Lo que se cambie despues de ver cifras se apunta abajo como cambio, con su fecha.

## Punto de partida

`ZONA_COBERTURA_2026-10-09.md`: de 5.472 clips con `strip_location` y `zona_`, contestan 3.363
(0,615): zona 0,482 contra `middle` 0,357. No contestan 1.954 por "sin homografia confiable en
+-0,5 s del tocado"; en 0,467 de ellos nunca se ven dos rayas transversales en un cuadro (a ojo: el
borde CERCANO de la pista esta tapado por el rotulo y sin los dos bordes `observe` no busca rayas),
0,323 tienen homografia propagada pero con confianza < 0,35, 0,143 ven rayas sin semilla y 0,067
pierden la cadena. Los respaldos por umbral (A, B, N) contestaban lo mismo que `middle`.

## La idea (de `04_documentacion/investigacion/2026-10-09_calibracion_pista.md`, recomendaciones 1-3)

La camara de retransmision es un TRIPODE: centro optico fijo, solo gira (pan, tilt) y hace zoom. En
vez de una homografia libre (8 grados de libertad, mal condicionada en una pista de 14 x 1,5 m), la
homografia del cuadro del tocado se parametriza con la camara fisica: centro `C = (Xc, Yc, Zc)`,
pan, tilt, roll (pequenio) y focal, con punto principal en el centro y pixel cuadrado, y con priors
fisicos (camara frente a la pista, roll ~0, la estatura de los tiradores ~1,65 m). Esa camara se
ajusta **sobre una ventana del clip** (los cuadros antes del tocado, encadenados con el movimiento
de camara ya medido) contra **todas las aristas** de la imagen fuera de personas y rotulo
("render and compare": las rayas y bordes del modelo de pista proyectados deben caer sobre aristas),
sin necesitar los dos bordes ni identificar las rayas una a una. Las rayas a 0, +-2, +-5, +-7 m
(separaciones 2-3-2) desambiguan donde se mira; la busqueda recorre hipotesis de donde esta la
pareja a lo largo de la pista y se queda con la que mejor explica las aristas.

Se usa como **RESPALDO**: solo en los clips en que la regla de hoy NO contesta. Los que ya contestan
no cambian por construccion.

## Reparto (el mismo que `ZONA_COBERTURA`, para no gastar torneos nuevos)

Los 148 torneos de los 5.472 clips, barajados con semilla 20261009: la mitad AJUSTE (74 torneos,
`02_validacion/auditoria/zona_cobertura_2026-10-09/torneos_ajuste.txt`) y la mitad PRUEBA (74).
- AJUSTE: los 300 clips sin homografia de `lista_ajuste.txt` (+ los 60 que contestan) y, como
  verdad en METROS sin mirar etiquetas, clips de AJUSTE que SI contestan (la x de la homografia
  estricta): ahi se mide el error en metros del modelo nuevo arrancado SIN la homografia estricta.
  Todo lo que se elija (umbrales de confianza, margen, filtros) se elige en AJUSTE.
- PRUEBA: los 716 clips sin homografia de `lista_prueba.txt` (+ los 100 que contestan), UNA vez.

## Criterio de adopcion (en PRUEBA)

1. **(a)** los clips NUEVOS que contestan (sin homografia estricta y con respuesta del tripode)
   baten a `middle` en esos mismos clips con IC95 por clip (bootstrap, 2.000) que no cruza cero;
2. **(b)** los que ya contestaban no cambian (respaldo): se comprueba re-ejecutando los 100;
3. **(c)** la salida lleva la via nueva (`via_h = tripode`), para separar las cifras.

Si pasa: se enciende por defecto tras `FENCINGLY_CAMARA_TRIPODE`. Si no: APAGADO con su numero.
Se mide tambien el coste de CPU por clip.

## Desarrollo (AJUSTE, sin etiquetas: solo la x de la homografia estricta como referencia)

Lo que se vio al construirlo, en clips de AJUSTE que SI contestan (`lista_metros_ajuste.txt`), sin
mirar `strip_location`:

1. **El centro de la camara no se identifica desde la pista.** Ajustando la camara de tripode a las
   homografias de produccion (12 clips), los ajustes buenos (rms < 10 px) salen con focal y distancia
   absurdas (f de 10^6-10^8 px, camara a cientos de km): la pista es tan estrecha que la camara es
   practicamente AFIN y distancia y zoom se compensan. Consecuencia: el modelo se usa con el centro
   en un valor nominal (0, -15, 5) m y priors, es decir, como camara de perspectiva debil con roll,
   pan, tilt y zoom; el "tripode" que se gana es la restriccion fisica (sin roll, punto principal
   centrado, pixel cuadrado, pies sobre la linea central, estatura), no un centro medido.
2. Sin orientacion de aristas, el ajuste encontraba "rayas" en el publico y los anuncios (5-6 rayas
   en clips con 2): se paso a chamfer ORIENTADO (8 bandas). Sin un prior de pies SOBRE la linea
   central, encontraba una pista aplastada contra el borde lejano. Ambos arreglados; ademas los
   huecos de la raya central (discontinua) no penalizan.
3. Con eso, en 18 clips: |x_tripode - x_estricta| mediana 2,36 m, 6 de 18 a menos de 1 m. Los errores
   son sobre todo desplazamientos de una raya (2-3 m): dos rayas a 2 m se ven igual en cuatro sitios
   de la pista. El margen del ajuste NO separa bien lo bueno de lo malo en esos 18.

## Cambio 1 (2026-10-09, antes de mirar ninguna etiqueta): las reglas de aceptacion

La eleccion se hace en AJUSTE sobre los 300 clips sin homografia, con la misma regla que
`ZONA_COBERTURA`: **entre las candidatas cuyos clips NUEVOS baten a `middle` por al menos +0,05,
la que mas clips nuevos contesta; a igualdad, la de mayor diferencia.** Familia fijada ahora:

- `R(m, k)`: margen del ajuste >= m (m en {0, 5, 10, 20, 40}) y rayas con evidencia >= k (k en
  {1, 2, 3});
- `+c3`: ademas, entre las rayas con evidencia hay TRES consecutivas del modelo (las separaciones
  2-3-2 hacen que tres seguidas solo encajen en un sitio);
- `+p`: ademas, los dos pies, distancia entre tiradores en [0,5; 5] m y |x| <= 8 m.

Se da tambien, por informacion, el error en metros de cada regla en los 150 + 60 clips de AJUSTE que
contestan (arrancando SIN la homografia estricta). La regla elegida es la UNICA que se evalua en
PRUEBA.

## Resultados en AJUSTE (510 clips: 300 sin homografia + 210 que contestan)

La cache reproduce la produccion: los 300 de `lista_ajuste.txt` salen "sin homografia" otra vez
(300 de 300); el tripode ajusta 285 (15 sin pies o sin cadena en la ventana).

| regla (seleccion) | nuevos | zona | middle | dif | ref. en metros: n, mediana \|err\|, < 1 m |
|---|---:|---:|---:|---:|---|
| R(0,1) (todo lo que ajusta) | 264 | 0,417 | 0,481 | **-0,064** | 207, 1,81 m, 0,38 |
| R(0,1)+c3+p | 176 | 0,483 | 0,477 | +0,006 | 144, 1,52 m, 0,43 |
| R(5,3)+p | 146 | 0,507 | 0,479 | +0,027 | 130, 1,54 m, 0,42 |
| **R(10,3)+c3+p** (la mejor) | 123 | 0,528 | 0,496 | **+0,033**, IC95 [-0,049, +0,114] | 101, 1,44 m, 0,45 |
| R(20,1)+c3+p | 80 | 0,562 | 0,537 | +0,025 | 64, 0,63 m, 0,55 |
| R(40,1) | 34 | 0,441 | 0,500 | -0,059 | 31, 0,68 m, 0,55 |

(las 60 combinaciones, en `02_validacion/auditoria/camara_tripode_2026-10-09/fit_ajuste.jsonl.gz`
con `evalua.py`). **Ninguna llega al +0,05 de la regla de eleccion: el tripode NO va a PRUEBA**, como
se declaro (igual que la candidata N de `ZONA_COBERTURA`). PRUEBA queda SIN MIRAR para un intento
siguiente.

Lo que dicen las cifras:
- **En metros, en clips que SI contestan, el tripode arrancado sin la homografia estricta se equivoca
  una mediana de 1,4-1,8 m y acierta a < 1 m solo en ~4 de cada 10**; con la zona estricta coincide
  en 0,535 (R(10,3)+c3+p). Con margen alto (>= 20) baja a 0,6-0,7 m, pero entonces contesta pocos y
  en esos pocos `middle` ya acierta mucho.
- La confusion de R(10,3)+c3+p (filas verdad): de 61 `middle` acierta 49, pero de 20 `fotl_half`
  manda 11 a `middle` y de 27 `fotr_half` 14: **el tripode se encoge hacia el centro**. Es el
  sintoma de la identificacion de rayas: dos rayas a 2 m (-2/0, 0/2) son las mas comunes, y el
  ajuste prefiere la hipotesis central.
- La causa de fondo es la de `ZONA_COBERTURA`: en estos clips casi nunca se ven tres rayas seguidas en
  2 s, y con una o dos rayas iguales la pista es ambigua; la restriccion fisica de la camara no quita
  esa ambiguedad (el centro de la camara, ademas, no se puede medir desde la pista: punto 1).

## Veredicto

**NO PASA (no llega a PRUEBA). `FENCINGLY_CAMARA_TRIPODE` queda en `0` (apagado).** La cobertura de
la zona se queda en 0,615 con el acierto de siempre. **No hay que recalcular nada.**

## Que queda en el codigo

- `01_extraccion/codigo/pista_tripode.py` (nuevo, LF): la camara fisica (`camara`), el encadenado
  (`cadena`), el chamfer orientado (`Evidencia`, `puntua`), los priors (`prior`), la busqueda
  (`ajusta`) y la regla de aceptacion (`acepta`).
- `01_extraccion/codigo/zona_tocado.py` (CRLF conservado): `TRIPODE` (`FENCINGLY_CAMARA_TRIPODE`,
  defecto `0`), `TRIPODE_REGLA = (10, 3, True, True)` (la mejor de AJUSTE, POR DEBAJO del umbral),
  `TRIPODE_VENT_S = 2.0` y la funcion `_tripode`. Solo actua si la regla de siempre no contesta; sale
  con `via_h = metodo_h = tripode` y `conf_h` vacio.
- **Apagado, la salida es la de antes**: `analiza` da la misma zona, x, cuadro, confianza y metodo que
  el `zona_` de disco en 5 de 5 clips (3 que contestan, 2 que no). **Encendido reproduce el
  experimento** en 2 de 2 clips aceptados (misma x a 3 decimales y misma zona), y no toca el que la
  regla rechaza (`verifica_parche.py`).
- **Coste si se encendiera**: ~6,4 s de CPU por clip SIN homografia (mediana, p90 7,2 s, con la
  maquina compartida), encima de los ~11 s de rasgos + calibracion que ya paga la etapa 11; en
  florete (~36 % sin homografia de 106.000) serian ~68 h de CPU.

## Para un intento siguiente

1. La ambiguedad es de IDENTIFICACION, no de modelo de camara: lo que falta es una senial que diga
   que raya es cual con una o dos a la vista. Candidatas: la raya central discontinua y el cambio de
   color de la zona de advertencia (+-5 a +-7) medidos sobre la imagen en gris (la cache de este
   experimento solo guardo aristas), y una ventana mas larga que la de 2 s, que vea mas pista
   mientras la camara sigue a los tiradores.
2. Con la camara de un clip VECINO del mismo torneo (misma retransmision, mismo tripode) como prior
   de roll, tilt y zoom: el centro no se mide, pero la orientacion tipica si se podria compartir.
3. El conjunto de PRUEBA (`lista_prueba.txt`, 74 torneos) esta intacto; su cache esta a medias en
   `D:/FENCINGLY_SCRATCH/agente_tripode/cache/` (los .pkl que hay sirven, faltan ~400).

Scripts: `02_validacion/auditoria/camara_tripode_2026-10-09/` (`cache.py`, `fit.py`, `evalua.py`,
`verifica_parche.py`, y los diagnosticos `diag.py`, `tri_de_H.py`, `dibuja.py`, `prueba6.py`).

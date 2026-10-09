# Cobertura de la zona del tocado (etapa 11): protocolo DECLARADO ANTES de medir

Escrito el 2026-10-09 por la tarde, antes de mirar ningun resultado de las reglas nuevas. Lo que se
cambie despues de ver cifras se apunta abajo como cambio, con su fecha.

## Punto de partida (medido hoy, `02_validacion/herramienta/mide_zona.py`, luces nuevas)

    clips con strip_location y zona_   5.472
    contestan                          3.363 (0,615)    zona 0,482   middle 0,357   +0,125
    no contestan por "sin homografia confiable en +-0,5 s del tocado"   1.954
    (sin luz 133, sin pies 22)

La regla de hoy (`zona_tocado.analiza`): se procesan los cuadros **solo hasta el tocado + 3**, se
calibra (`pista_lineas.calibrate`: semillas por rayas, propagacion con el movimiento de camara, la
confianza cae x0,985 por cuadro propagado) y se acepta el cuadro mas cercano al tocado, dentro de
+-0,5 s, con confianza >= 0,35.

## Que se va a probar

Una regla de **RESPALDO**: solo se aplica en los clips en que la regla de hoy NO contesta. Por
construccion, **los clips que ya contestan no cambian** (criterio b abajo, que se comprueba igual
re-ejecutando una muestra). Familia de candidatas, fijada ahora:

- **A(c)**: la misma calibracion (hasta el tocado + 3) aceptando confianza >= c, c en {0,25; 0,15;
  0,05; 0}.
- **B(c)**: calibracion con los cuadros **hasta el final del clip** (las semillas posteriores al
  tocado se propagan hacia atras), confianza >= c, c en {0,35; 0,25; 0,15; 0,05; 0}.
- **B(0,35) y luego A(c)** o **B(c)**: combinaciones en ese orden.

Siempre la misma ventana (+-0,5 s) y los pies del cuadro elegido, como hoy.

## Reparto

Torneos de los 5.472 clips (148) barajados con semilla 20261009: la mitad AJUSTE (74 torneos, 1.240
clips sin homografia) y la mitad PRUEBA (74 torneos, 716 clips sin homografia). La regla se elige
en AJUSTE con una muestra al azar de 300 clips sin homografia (+ 60 que contestan, para comprobar
que el experimento reproduce la produccion). Se evalua UNA vez en los **716 de PRUEBA** completos
(+ 100 que contestan, al azar).

**Como se elige en AJUSTE**: entre las candidatas cuyos clips NUEVOS baten a `middle` por al menos
+0,05 (punto), la que mas clips nuevos contesta; a igualdad, la de mayor diferencia con `middle`.

## Criterio de adopcion (en PRUEBA)

1. **(a)** los clips NUEVOS que contestan baten a `middle` (en esos mismos clips) con IC95 por clip
   (bootstrap, 2.000) que no cruza cero;
2. **(b)** el acierto de los que ya contestaban no baja mas de 0,01 (en los 100 re-ejecutados: el
   respaldo no los toca, asi que tiene que salir identico);
3. **(c)** la salida lleva la via nueva (`via_h`), para separar las cifras.

Si pasa: se enciende por defecto tras `FENCINGLY_ZONA_RESPALDO`. Si no: APAGADO con su numero.

## Resultados en AJUSTE (300 sin homografia + 60 que contestan)

El experimento (`02_validacion/auditoria/zona_cobertura_2026-10-09/exp_zona.py`) reproduce la
produccion: la regla de hoy da la misma zona que `zona_<clip>.csv` en **360 de 360** clips.

### Por que no contestan (calibracion de produccion, n=300)

| tipo | n | fraccion |
|---|---:|---:|
| ninguna homografia hasta el tocado: **nunca se ven 2 rayas transversales** en el mismo cuadro | 140 | 0,467 |
| homografia en la ventana pero con confianza < 0,35 (propagada lejos de su semilla: cae x0,985 por cuadro) | 97 | 0,323 |
| ninguna homografia: hay rayas pero ninguna semilla pasa (k < 3 y margen <= 1) | 43 | 0,143 |
| homografia en el clip pero no en la ventana (cadena rota por un corte o por el flujo) | 20 | 0,067 |
| ... de todos ellos, con calibracion hasta el FINAL del clip contesta con confianza >= 0,35 | 38 | 0,127 |

### Las candidatas (clips NUEVOS que contestarian; zona contra `middle` en esos mismos clips)

| candidata | nuevos | zona | middle | dif | IC95 por clip |
|---|---:|---:|---:|---:|---|
| A(0,25) | 19 | 0,526 | 0,368 | +0,158 | [-0,158, +0,474] |
| A(0,15) | 52 | 0,462 | 0,423 | +0,038 | [-0,115, +0,212] |
| A(0,05) | 95 | 0,379 | 0,389 | -0,011 | [-0,126, +0,116] |
| A(0) | 96 | 0,375 | 0,396 | -0,021 | |
| B(0,35) | 38 | 0,368 | 0,447 | -0,079 | [-0,289, +0,105] |
| B(0,25) | 63 | 0,397 | 0,413 | -0,016 | |
| B(0,15) | 96 | 0,385 | 0,406 | -0,021 | |
| B(0,05) | 133 | 0,361 | 0,398 | -0,038 | [-0,143, +0,060] |
| B(0,35)+A(0,25) | 56 | 0,429 | 0,429 | +0,000 | |
| B(0,35)+A(0,05) | 124 | 0,371 | 0,403 | -0,032 | |

**Lectura**: lo que contestan las reglas de respaldo vale lo mismo que contestar `middle`. Bajar la
confianza saca homografias que estan mal (deriva de la propagacion o semilla mal identificada), y
calibrar hasta el final del clip no ayuda: las semillas de despues del tocado (muchas veces tras un
cambio de plano que el detector de cortes no ve) no traen informacion.

### Cambio 1 (2026-10-09, visto AJUSTE y antes de mirar PRUEBA): filtro de plausibilidad

Se anade a la familia la variante "+p": solo se acepta la respuesta si estan los dos pies, la
distancia entre tiradores esta en [0,5; 5] m y |x| <= 8 m para los dos. En AJUSTE:

| candidata | nuevos | zona | middle | dif | IC95 |
|---|---:|---:|---:|---:|---|
| A(0,25)+p | 14 | 0,500 | 0,214 | +0,286 | [+0,000, +0,571] |
| A(0,15)+p | 37 | 0,432 | 0,378 | +0,054 | [-0,135, +0,243] |
| B(0,25)+p | 39 | 0,436 | 0,359 | +0,077 | [-0,103, +0,256] |
| B(0,35)+A(0,25)+p | 35 | 0,457 | 0,371 | +0,086 | [-0,114, +0,286] |
| B(0,05)+p | 88 | 0,409 | 0,398 | +0,011 | |

Regla de eleccion declarada (dif >= +0,05 y, entre esas, la que mas contesta) sobre la familia
ampliada: **B(0,25)+p** (39 nuevos de 300, +0,077). Es la UNICA que decide en PRUEBA; las demas
se dan como informacion. Sobre la familia original habria salido A(0,25) (19 nuevos).

### Lo que se ve a ojo (12 clips al azar de AJUSTE, `ojo/`)

En los 6 de "nunca 2 rayas" y en 3 de los otros 6, `observe` encuentra el borde LEJANO pero no el
CERCANO, y sin los dos bordes ni siquiera busca rayas transversales, **aunque se vean** (la central y
las de puesta en guardia, claras en Cali 2018, Cancun y el torneo ruso). El borde cercano esta:
justo debajo de `BANDA_ROTULO = 0,78` (la franja que se excluye por el rotulo, ~0,79 en el ruso), o
tapado por el rotulo FIE, que en estas retransmisiones empieza hacia 0,84 de la altura, no en 0,78.

### Cambio 2 (2026-10-09, antes de medirla): candidata N, la franja del rotulo mas abajo

Respaldo N(b): en los clips sin respuesta, se rehacen los rasgos con `BANDA_ROTULO = b` (b en
{0,84; 0,88}) y se aplica la regla de hoy sin tocarla (calibracion hasta tf+3, confianza >= 0,35,
+-0,5 s). Si el borde cercano esta tapado, el borde de arriba del rotulo hace de borde cercano: la
y sale mal escalada, pero la x a lo largo de la pista la dan las rayas transversales. Se mide en
una submuestra de 150 de los 300 de AJUSTE (por tiempo de CPU); se elige con la misma regla de
eleccion, y la elegida se evalua en los 716 de PRUEBA con el mismo criterio (a)-(c). Son DOS
evaluaciones en PRUEBA (B(0,25)+p y N): se dice asi, y cada una lleva su IC.

### N en AJUSTE (139 sin homografia de la submuestra)

| candidata | nuevos | cobertura | zona | middle | dif | IC95 por clip |
|---|---:|---:|---:|---:|---:|---|
| N(0,84) | 76 | 0,547 | 0,474 | 0,592 | **-0,118** | [-0,250, +0,013] |
| N(0,84)+p | 60 | 0,432 | 0,500 | 0,550 | -0,050 | [-0,200, +0,100] |
| N(0,88) | 104 | 0,748 | 0,385 | 0,500 | -0,115 | [-0,221, -0,010] |
| N(0,88)+p | 82 | 0,590 | 0,402 | 0,512 | -0,110 | [-0,232, +0,012] |

Bajar la franja **si** devuelve la homografia (contesta en la mitad o en tres cuartos de los clips que
fallaban), pero lo que contesta es PEOR que `middle`. La matriz de confusion de N(0,84) no muestra un
sesgo de escala que se pueda corregir, sino ruido: de 45 `middle` de verdad, 7 salen en zona de
aviso. Lectura: con el borde del rotulo haciendo de borde cercano, la escala vertical sale mal y el
prior de escala de `hypothesize` (que usa la altura del tirador) identifica mal las rayas (una +-2
tomada por +-5). **Ninguna N llega al +0,05 de la regla de eleccion: N no va a PRUEBA.**

## Resultados en PRUEBA (74 torneos no vistos)

**La evaluacion de B(0,25)+p se INTERRUMPIO** a los 79 clips (de 816) por tiempo de CPU (calibrar
hasta el final del clip con la maquina al 100 % salia a mas de 2 h), y se decidio interrumpirla
**sin mirar ningun resultado**, para pasar el tiempo a N. Con los 79 que llego a hacer (todos sin
homografia en produccion; reproduccion de la regla de hoy 79 de 79):

    B(0,25)+p   6 nuevos   zona 0,333   middle 0,500   -0,167   IC95 [-0,667, +0,333]   -> (a) NO PASA
    (informativo) A(0,25) 2 nuevos; B(0,05) 19 nuevos, +0,053 [-0,158, +0,263]

## Veredicto

**NO PASA ninguna regla de respaldo. `FENCINGLY_ZONA_RESPALDO` queda en `0` (apagado).** La
cobertura de la zona se queda en 0,615 con el acierto de siempre.

Lo que si queda medido y sirve para el siguiente intento:

1. Las causas (n=300, AJUSTE): 0,467 nunca ven dos rayas transversales en un cuadro (a ojo: no se
   encuentra el borde CERCANO, debajo de `BANDA_ROTULO = 0,78` o tapado por el rotulo); 0,323 tienen
   homografia propagada pero con confianza < 0,35; 0,143 ven rayas pero ninguna semilla pasa; 0,067
   pierden la cadena por un corte o por el flujo.
2. Lo que contestan las homografias de baja confianza y las de la calibracion hasta el final del
   clip vale lo mismo que `middle`: la confianza de `calibrate` SI separa lo bueno de lo malo.
3. Encontrar el borde cercano (N) no basta: hace falta otra fuente de escala que no sea el borde del
   rotulo -- p. ej. detectar por clip donde empieza el rotulo (borde horizontal quieto en el tiempo) y
   sintetizar el borde cercano a +1,5 m del lejano con la perspectiva de los pies, o identificar las
   rayas por su patron (central a trazos, cambio de color en +-5) sin el prior de escala.

## Que queda en el codigo

- `01_extraccion/codigo/zona_tocado.py`: `RESPALDO` (`FENCINGLY_ZONA_RESPALDO`, defecto `0`) y
  `BANDA_RESPALDO = 0,84`; funcion nueva `_rasgos` (la lectura de rasgos que antes estaba dentro de
  `analiza`, sin cambios); columna nueva **`via_h`** en `zona_<clip>.csv`: `estricta` si contesta la
  regla de siempre, `rotulo_0.84` si contesto el respaldo, vacia si no contesto nadie.
- **Apagado, la salida es la de antes**: verificado en 5 clips (3 que contestan, 2 que no) que
  `analiza` da la misma zona, x, cuadro, confianza y metodo que el `zona_` de disco, y con el
  respaldo encendido reproduce el experimento en 2 de 2 clips (misma zona, x y cuadro).
- **No hay que recalcular nada.** Los `zona_` ya escritos no tienen `via_h` (se lee como vacio).
  Si algun dia se enciende, solo cuesta en los clips sin homografia: una segunda pasada de rasgos y
  calibracion, ~12 s de CPU por clip (anotados: ~1.950 clips, ~6,5 h; florete: ~36 % de 106.000,
  ~130 h).

Scripts, listas y resultados (jsonl.gz): `02_validacion/auditoria/zona_cobertura_2026-10-09/`.
Imagenes a ojo: `D:/FENCINGLY_SCRATCH/agente_zona/ojo/`.

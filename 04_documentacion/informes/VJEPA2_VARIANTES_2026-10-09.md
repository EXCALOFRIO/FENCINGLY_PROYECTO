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

## Resultados

(pendiente)

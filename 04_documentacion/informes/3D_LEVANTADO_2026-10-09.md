# 3D levantado desde las poses 2D guardadas: ¿ayuda a `initiated`? Protocolo DECLARADO ANTES de medir

Escrito el 2026-10-09, antes de calcular ningún rasgo 3D sobre los anotados y antes de entrenar nada.
Lo que cambie después de ver cifras se apunta abajo como cambio fechado; no reemplaza a esto.

## Qué se monta

- **Levantador**: MotionAGFormer-B (Apache-2.0, https://github.com/TaatiTeam/MotionAGFormer), pesos
  H36M de 243 cuadros (`motionagformer-b-h36m.pth.tr`, MPJPE 38,5 mm según el propio checkpoint),
  en `D:/FENCINGLY_SCRATCH/agente_3d/terceros/` (fuera del repo y del venv). `timm` no está en el
  venv: solo se usa `DropPath`, que en inferencia es la identidad, y se sustituye por un módulo
  mínimo en memoria. No se instala nada.
- **Entrada**: `poses_<clip>.npz` (308 puntos Sapiens 2 / Goliath) -> COCO-17 con el mapa MEDIDO del
  proyecto (`cono_lib`: muñeca = raíz de mano, 21 = mano del hombro 6, 42 = mano del hombro 5) ->
  H36M-17 con la conversión estándar del demo de MHFormer/MotionAGFormer. Coordenadas normalizadas a
  pantalla como el demo (`x/W*2-1`, `y/W*2-H/W`) y la confianza como tercer canal.
- **Ventanas**: 243 cuadros; clips más cortos se remuestrean como el demo; más largos, en ventanas
  consecutivas y la última pegada al final.
- **Volteo en la prueba (flip-TTA) como el demo**: `(f(x) + flip(f(flip(x))))/2`. Con esto la salida
  es EXACTAMENTE equivariante al espejo, que es lo que permite rasgos impares.

## Rasgos 3D (12 por tirador y cuadro, en su marco de ataque, normalizados por su talla 3D)

`brazo_ang` (codo), `brazo_ext3d` (hombro-muñeca / largo del brazo), `brazo_adel3d` (alcance hacia
el rival), `brazo_elev`, `v_ext3d`, `tronco_incl` (inclinación hacia el rival), `fondo_prof`
(separación 3D de tobillos), `rodilla_del_ang`, `rodilla_tras_ang`, `perfil` (hombros a lo largo
de la pista), `cadera_alt`, `mano_prof`. El brazo que cuenta es, en cada cuadro, el MÁS adelantado
hacia el rival (como `ext_brazo_adelante` del etiquetador), y la pierna delantera es la del tobillo
más adelantado. Se remuestrean a 64 pasos sobre los cuadros útiles igual que los rasgos del
etiquetador.

## Plausibilidad (se da, no decide)

Coeficiente de variación de 8 huesos dentro del clip (brazo, antebrazo, muslo, pierna, x2 lados) y
asimetría izquierda/derecha |L-R|/media; mediana y p90 por clip.

## La comparación que decide

- Filas: `etiquetador.reune(cabeza="initiated")` sobre los anotados terminados. Reparto:
  `etiquetador.particion_por_torneo(filas)` con su semilla (20260922), el del careo del etiquetador.
- **A** = el etiquetador tal cual; **A+3D** = el mismo modelo con los 12 rasgos 3D pegados al bloque
  propio (mismo `construye`, misma receta de `entrena`: 60 épocas, AdamW 3e-3, wd 1e-4, lote 32,
  espejo como aumento, última época). Para quitar el ruido de semilla, cada brazo se entrena con
  **5 semillas de torch** (20260922 + 0..4) con el reparto FIJO y se promedian los logits.
- **Se adopta si** `acc(A+3D) - acc(A) >= +0,02` en el reservado, **y** el IC95 de la diferencia por
  bootstrap agrupado por TORNEO (2.000 remuestreos) no cruza cero, **y** el control del espejo del
  modelo da 1,0000 en los dos, **y** la prueba de espejo de los RASGOS 3D (npz espejado de verdad,
  como `verifica_etiquetador.espeja_npz`, en >= 12 clips) da discrepancia relativa máxima <= 1e-3.
- Informativo: por arma; la semilla única 20260922; el coste por clip.

## Cambios (2026-10-09, ANTES de la medida completa; solo se había corrido la maquinaria en 150 clips)

La prueba de humo (150 clips, 1 semilla, 39 reservados: cifras de acierto SIN valor y no miradas
para decidir nada) sirvió solo para comprobar que todo corre. Lo que cambió por ella:

1. **El espejo de los rasgos daba 1,35e-3** (umbral 1e-3) en 3 de 12 clips. Causa medida en
   `AbandonedBackGnu`: un tirador salía idéntico bit a bit y el otro con 4e-4 de diferencia. El
   levantador elige vecinos temporales por similitud (`use_temporal_similarity`, una selección
   discontinua), así que un último bit distinto en la entrada se amplifica. Primero se pasó a una
   secuencia por llamada y sin TF32 (en lote, la posición en el lote cambiaba el último bit).
2. **El último bit venía de la propia prueba**: `verifica_etiquetador.espeja_npz` guarda `W - x` en
   float32, que redondea en el tirador de la izquierda. La prueba de espejo de los rasgos 3D usa ahora
   un espejo en float64 (`verifica_3d.espeja_npz64`, mismo `PARES`), y la normalización se escribe
   `(2x - W)/W` y la conversión a H36M con medias de pares, para que el espejo exacto de la entrada dé
   el espejo exacto de la salida. Resultado: **discrepancia 0,0 en 12 de 12 clips**. El umbral no se
   tocó.
3. **Coste**: con la GPU compartida, ~0,22 s por pasada del levantador y 8 pasadas en un clip de 300
   cuadros. Ahora, si `T <= 486` se usa UNA ventana de 243 cuadros muestreada sobre el clip entero y
   se interpola linealmente a todos los cuadros (los rasgos se remuestrean a 64 pasos de todos modos).
   fp16 se probó y se descartó: más lento (0,40 s) y distinto.

## Resultados (2026-10-09, `verifica_3d.py --semillas 5`, log en `D:/FENCINGLY_SCRATCH/agente_3d/`)

    filas con initiated 5.536 (de 5.791 con rasgos); reparto 105 torneos / 4.152 clips de
    entrenamiento, 42 torneos / 1.384 clips reservados; sin rasgos 3D: 0

    plausibilidad (11.072 tiradores): CV de hueso dentro del clip mediana 0,079 (p90 0,135);
    peor hueso del clip mediana 0,164 (p90 0,275); asimetria izq/der media 0,082 (p90 0,143)
    espejo de los rasgos 3D: 12 de 12 clips, discrepancia maxima 0,0

    RESERVADO, 1.384 clips, 42 torneos (logits medios de 5 semillas)
      A        0,645   (semilla unica 20260922: 0,614; las 5: 0,614 0,597 0,576 0,591 0,616)
      A+3D     0,631   (semilla unica 20260922: 0,598; las 5: 0,598 0,592 0,590 0,580 0,606)
      diferencia -0,014   IC95 por torneo [-0,045, -0,001]
      control del espejo del modelo: A 1,0000   A+3D 1,0000
      espada n=209  A 0,694  A+3D 0,641 | florete n=1.040  A 0,631  A+3D 0,623 | sable n=135  A 0,681  A+3D 0,674

**VEREDICTO: NO PASA.** Los 12 rasgos 3D de MotionAGFormer-B no ayudan a `initiated`: lo empeoran
un poco (-0,014, IC95 por torneo [-0,045, -0,001]), en las tres armas. No entran en el
etiquetador ni en la corrida. Lectura (opinion): el levantador es de Human3.6M y no de esgrima
(AthletePose3D midio 237 mm de error en movimiento atletico sin reentrenar), y el CV de hueso del 8 %
dice que el 3D tiembla; sus rasgos anaden ruido a los 2D que ya resumen lo mismo.

**Coste** (para correrlo en florete si un dia se quiere): 0,45 s/clip de mediana con la GPU
compartida (V-JEPA y su sonda), ~13 h en 106k clips; una pasada aparte, no hace falta video.
Las poses 3D de los 5.536 clips se guardaron (`D:/FENCINGLY_SCRATCH/agente_3d/salidas/l3d_<clip>.npz`,
P3 float16, relativo a la pelvis) por si sirven para otra cosa (p. ej. reentrenar un levantador).

**Hallazgo de paso (informativo, NO era la pregunta y no se declaro antes):** el etiquetador con los
logits medios de 5 semillas da **0,645** en el mismo reservado, contra 0,614 de su semilla unica y
0,597 de media de las 5 por separado. La variacion entre semillas (0,576-0,616) es del tamano de
las diferencias que se estan midiendo en el proyecto: comparar con UNA semilla es comparar ruido.
Si se quiere adoptar el conjunto de 5 semillas, hay que declararlo y medirlo aparte.

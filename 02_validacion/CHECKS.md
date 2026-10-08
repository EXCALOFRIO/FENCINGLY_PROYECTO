# Checks de la extraccion

Cada fila es una **garantia que la tuberia dice cumplir**, con su umbral declarado, el sitio del codigo que la cumple y donde esta la evidencia para verla a ojo.

Esto no es una nota media. Una nota media alta puede convivir con un dato corrupto: la averia del fundido de `VainAchingGonolek` daba **96,1 % de presencia** sobre poses mezcladas de dos planos distintos. Una garantia se cumple en todos los cuadros entregados o no se cumple.

Medido sobre **60 clips de cajas** y **38 clips de poses** del corpus de florete.

| id | garantia | umbral declarado | medido | veredicto | codigo | evidencia |
|---|---|---|---|---|---|---|
| C1 | Ninguna caja entregada se teletransporta | salto del centro <= 0,60 anchos de caja por cuadro de hueco | peor salto 0.563 anchos en 60 clips | APTO | `extrae_pareja._continuidad, dos pasadas` | `evidencia/casos_sospechosos (avisos `salto`)` |
| C2 | Las dos cajas nunca caen sobre la misma persona | solape de la pareja entregada < 0,70 | peor solape 0.670 | APTO | `extrae_pareja._quita_duplicados` | `evidencia/casos_sospechosos (avisos `solapan`)` |
| C3 | Ninguna caja entregada es una tira del borde | proporcion ancho/alto >= 0,25 si toca el borde | 0 tiras entregadas; proporcion minima en el borde 0.250 | APTO | `extrae_pareja._quita_tiras` | `evidencia/poses_inventadas (el caso que lo destapo)` |
| C4 | El clip entregado empieza y acaba con los DOS medidos | primer y ultimo cuadro de la mascara con dos presencias | 0 clips de 60 con un borde sin los dos medidos | APTO | `extrae_pareja._recorta` | `evidencia/videos_revision (ya no hay poses congeladas en los bordes)` |
| C5 | El cruce de lados se queda en lo que hacen las parejas reales | NO es 0: verificado a ojo, dos tiradores reales se cruzan al pasar uno junto al otro | 3 clips de 60 con algun cruce; el peor PowerfulAllLarva con 12.7 % de sus cuadros (referencia verificada: pareja correcta 9,3 %, arbitro elegido 15,0 %) | MEDIDO | `a3_selector.elige_dos y pareja_v2.CRUZA_MAX` | `evidencia/eleccion_de_pareja, muestras/limite_intruso` |
| C8 | Ninguna caja entregada pertenece a alguien que nunca se vio entero | trayectoria cortada por el borde lateral en el 100 % de sus cuadros: descartada | 0 clips con un lado nunca visto entero | APTO | `pareja_v2.BORDE_SIEMPRE` | `muestras/limite_intruso (LavishDistantDikkops, 27,4 % -> 100 %)` |
| C6 | Los cuadros de primer plano y repeticion no se entregan | puerta de planos: caja > 0,75 del alto Y sin pareja plausible | 17333 cuadros utilizables de 19920 entregados en total | MEDIDO | `planos.tramo_utilizable` | `evidencia/puerta_de_planos` |
| C7 | Los cuadros de fundido encadenado no se entregan | mezcla >= 2,50 Y residuo relativo al contraste >= 0,15 | ver la tirada de `busca_errores.py`, aviso `fundido` | MEDIDO | `transiciones.marca` | `evidencia/fundidos_y_zoom` |
| P1 | Ninguna pose MEDIDA que se entrega esta vacia | al menos 20 puntos utiles del cuerpo de 63 | minimo 20 puntos utiles en 38 NPZ con filtro | APTO | `video_revision.tira_poses_vacias` | `evidencia/poses_inventadas` |
| P2 | Siempre hay DOS poses en los cuadros entregados | cero cuadros entregados sin pose de alguno de los dos | 0 cuadros entregados sin pose, en 38 clips | APTO | `video_revision.inventa_poses` | `evidencia/videos_revision, evidencia/poses_inventadas` |
| P3 | Toda pose inventada va marcada como tal | `origen` presente y con valor por cada punto entregado | 650 poses inventadas marcadas, de ellas 0 sostenidas | APTO | `video_revision.inventa_poses devuelve `origen`` | `evidencia/poses_inventadas (el panel del medio es lo inventado)` |
| P4 | Ningun keypoint cae fuera de la caja de su propia persona | keypoint dentro de su caja de YOLO con 5 % de holgura | 0 keypoints fuera de su caja | APTO | `video_revision._fuera_de_caja` | `evidencia/capas_densas` |
| P5 | Se puede saber que parte de lo entregado NO esta validada | cada pose inventada lleva el largo del hueco del que sale, en `hueco` | de 21950 poses entregadas: 97.0 % medidas, 1.3 % interpoladas con hueco <= 24 cuadros (rango con error medido), 1.6 % interpoladas con hueco mayor (SIN medida); hueco maximo 66 | APTO | `video_revision.inventa_poses devuelve `hueco`` | `evidencia/poses_inventadas (el rotulo lleva el largo del hueco)` |

La evidencia esta en el repo privado `EXCALORIO/fencingly-extraccion` bajo esas rutas, y en local en `02_validacion/muestras/` y `02_validacion/videos_revision/`.

## Presencia de los dos tiradores, en la misma tirada

| medida | valor |
|---|---|
| mediana de cuadros con los dos | 100.0 % |
| p10 | 95.2 % |
| minimo | 84.3 % |
| clips por encima del 90 % | 59 de 60 |

Esa columna es **antes de inventar** las poses que faltan. Despues de inventar, el check P2 dice que no queda ningun cuadro entregado sin los dos.

## Error de la pose inventada

Medido **escondiendo poses que si se conocen** e interpolando desde sus dos vecinos: mediana 2.4 % del torso, p90 5.8 %.

**Lo que esta medida NO dice.** Los huecos que de verdad se rellenan son en buena parte cuadros donde el tirador **no esta en el plano**, y para esos no existe verdad con la que comparar: su error es, por construccion, no verificable. Por eso `origen` va guardado junto a los puntos (check P3) y no se debe perder.

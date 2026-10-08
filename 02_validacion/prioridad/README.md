# Prioridad en florete: la evidencia

Todo lo que hay en esta carpeta sale de la sesión del **2026-09-10**. El mapa regla → señal está en
`04_documentacion/prioridad_fie.md` y el código en `01_extraccion/codigo/prioridad.py`.

## El veredicto, primero

**El detector no funciona todavía**, y la cifra que lo dice es la del lote de control:

| medida | detector | línea base «quien avanza primero» | trivial «siempre ninguna» |
|---|---:|---:|---:|
| 21 ventanas anotadas, acierto por cuadro | **69,7 %** | 34,6 % | **73,8 %** |
| 2 ventanas de control, nunca vistas | **9,1 %** | — | 0 % |

En las ventanas que usé para arreglarlo empata con «no contestar nunca»; en las que no había visto,
**no contesta**. Lo que sí vale y es transferible: el mapa de reglas, las variables medidas, el
tiempo de esgrima, la cota de alcance, el catálogo de averías y las herramientas.

## Qué es cada fichero

| fichero | qué es |
|---|---|
| `anotacion_a_mano.csv` | **la verdad de referencia**, 21 ventanas de 10 clips, anotada a ojo por el agente |
| `anotacion_holdout.csv` | 2 ventanas de control, anotadas **después** de congelar el código, de clips sin tocar |
| `p1_variables.csv` | las variables geométricas por clip × lado, 42 filas (21 clips de florete) |
| `p2_por_cuadro.csv` | cuadro a cuadro: verdad, detector, línea base, confianza y **la regla FIE aplicada** |
| `p2_salida.txt` | la salida completa de las ocho medidas, tal cual |
| `paradas_candidatas.csv` | los 8 candidatos a parada del criterio de t.86 |
| `tiras/*.png` | las láminas con que se anotó; `zoom_*` son las de un tirador por columna |
| `videos/*.mp4` | 5 clips con la prioridad pintada, la regla escrita y la curva de `ext_hr` |
| `fig_*.png` | las averías y la corrección de anotación, cada una con su cuadro |

## Las cinco averías, cada una con su imagen

| figura | qué se ve |
|---|---|
| `fig_fallo_prioridad_sin_adversario_Alive_c213.png` | **PRIORIDAD: DERECHA con confianza 1,00** mientras el tirador de la izquierda no está en el plano y el de la derecha vuelve andando con el brazo colgando. Es lo que motivó la cota `ALCANCE_MAX` |
| `fig_fallo_intruso_PowerfulAllLarva_c147.png` | la caja **verde**, o sea «el tirador de la izquierda», está sobre una **persona de calle que cruza en primer plano**. Los dos floretistas están juntos a la izquierda. La prioridad de ese clip es basura por construcción, y el fallo es del selector de pareja, no de este módulo |
| `fig_fallo_sable_MellowHideousAngora_c231.png` | **es SABLE, no florete**: el cartel del fondo dice «SABRE FEMININ». Se cuela porque ese clip viene del muestreo viejo, el que el propio documento maestro declaró mal en su §14.6. La prioridad del sable es t.100 y siguientes, no t.82 |
| `fig_fallo_cajas_que_faltan_PotablePossible_c227.png` | faltan cajas en c233, c236 y c245 **en medio de la acción**: sin las dos poses no hay decisión posible y la salida correcta es `ninguna` por falta de dato |
| `fig_correccion_anotacion_Gentle_c270.png` | **mi propia anotación estaba mal**, y la corrigió el detector: yo había escrito «derecha en c279» y el zoom por tirador enseña al de la **izquierda** con el brazo ya extendido en c279 mientras el otro lo tiene encogido |
| `fig_arreglado_prioridad_sin_adversario_Alive_c213.png` | el **después** del primer fallo: el mismo cuadro con la cota `ALCANCE_MAX` puesta dice «SIN PRIORIDAD». Es el único antes-y-después de la carpeta |

Esa última fila es la más importante de la tabla, porque va en la dirección contraria a la habitual:
**el algoritmo contradijo al humano y tenía razón.** Es también la razón de que la columna
`revisado_zoom` exista en la anotación: de las 21 ventanas, 9 se volvieron a verificar con el zoom
por tirador y una cambió de lado.

## Cómo se reproduce

```
01_extraccion/codigo/p1_variables.py            las variables y sus distribuciones
01_extraccion/codigo/p2_prioridad_mide.py       las ocho medidas contra la anotacion
02_validacion/herramienta/prioridad_anota.py    genera las laminas para anotar
02_validacion/herramienta/video_prioridad.py    los videos de revision
```
Todo con el intérprete de ultralytics, que es el que tiene `cv2` y `scipy` a la vez. **No hace falta
GPU**: las poses de 308 puntos ya están en `01_extraccion/pruebas/poses_*.npz`.

## Lo que esta carpeta NO dice

- **No dice quién ganó el tocado.** No se ha usado ni la etiqueta de tocado ni el orden de las luces
  en ninguna medida, a propósito: el proyecto ya midió que el orden de las luces es un atajo que
  acierta el 63,51 % por sí solo y que hundió el techo de un modelo al 73,6 %.
- **No dice que la anotación sea correcta.** La hizo un agente mirando láminas, no un árbitro FIE, y
  ya se demostró equivocada una vez. Cuatro de las 21 ventanas llevan confianza «media» o «baja».
- **No dice nada sobre paradas.** Hay 8 candidatos y **0 paradas anotadas** en las 21 ventanas, así
  que no hay ni una sola cifra de acierto de paradas. Lo que falta para tenerla está escrito en
  `04_documentacion/prioridad_fie.md`.

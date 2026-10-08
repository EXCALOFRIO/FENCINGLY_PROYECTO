# Convenio de conteo de pasos, DECLARADO ANTES DE CONTAR

Fecha: 2026-09-11. Nada se ha contado todavía cuando se escribe este fichero.

Existe porque el usuario dice *«igual por clip habrá varias, unas 10 marchas por tirador»* y el
detector da 5,7, y **esa discrepancia no se puede resolver sin fijar qué es un paso**. El proyecto ya
ha tenido cinco cifras contradictorias en un día por un convenio sin declarar.

## La unidad que se cuenta: ZANCADA, no «marcha»

**Una ZANCADA es un apoyo que se traslada**: un pie pierde el contacto con la pista (o lo desliza) y
vuelve a apoyarse **en otro sitio a lo largo de la pista**, desplazándose al menos lo que mide su
propio pie (unos 25 cm, que a ojo es inconfundible de un ajuste de peso).

Se cuenta la zancada y no la «marcha» por una razón de método: **la zancada es observable sin
ambigüedad** (se ve el pie despegar y aterrizar) y la «marcha» no, porque depende de cómo se agrupen
las zancadas, que es justo lo que está en discusión. Contar primero la unidad inequívoca y agrupar
después permite dar las dos cifras sin volver a contar.

### Qué NO cuenta como zancada

- **Ajuste de peso sin traslado.** El pie gira, se levanta el talón, se carga y descarga, pero la
  planta vuelve al mismo sitio. Es lo más frecuente en florete y no es un paso.
- **El pie que se arrastra dentro del mismo apoyo** en un fondo: el pie de atrás NO se traslada en un
  fondo bien hecho, se queda clavado. El fondo no aporta zancadas del pie de atrás.
- **Movimiento lateral o de reajuste transversal.** Sólo cuenta el traslado A LO LARGO de la pista,
  que es el eje que este proyecto mide (la transversal no es creíble, §23.7).
- **Lo que pasa fuera del tramo utilizable** (`mask` del clip) o con el tirador fuera de encuadre.

## El agrupamiento: de zancadas a marchas

**Una MARCHA (o un ROMPER) es una pareja de zancadas consecutivas del mismo sentido, una de cada
pie**: en la marcha el pie de delante avanza y el de atrás le sigue; en el romper, el de atrás va
primero. Es la definición de esgrima y es la que da la cifra que el usuario espera.

Reglas de desempate, declaradas para que el conteo sea reproducible:

1. Las zancadas se recorren en orden de cuadro. Se emparejan **dos consecutivas del mismo sentido y
   de pies distintos**.
2. Si una zancada se queda sin pareja (porque la siguiente es del mismo pie, o del sentido contrario,
   o ya no hay más), cuenta como **MEDIO PASO** y se anota aparte. No se descarta ni se redondea.
3. Un desplazamiento continuo de N zancadas alternas del mismo sentido son **N/2 marchas**, más medio
   paso si N es impar. O sea que «avanzar seguido» no es una marcha ni son N: son N/2.
4. El **fondo** aporta la zancada del pie de delante. Si esa zancada queda emparejada con la del pie
   de atrás en la recuperación, cuenta como paso además de como fondo; si no, es medio paso. **El
   fondo se cuenta aparte en su propia columna**, porque el detector lo tiene como tipo propio.

## Cómo se cuenta, y por qué no vale mirar mi propia serie

Se cuenta **sobre la imagen**, en mosaicos de la banda de las piernas muestreados cada 2 cuadros,
**con los fotogramas ESTABILIZADOS** por la similitud `Mk` de cada cuadro. La estabilización deja el
suelo quieto, así que un pie que se traslada se ve contra un fondo fijo y no hay que adivinar cuánto
de lo que se mueve es el paneo de cámara (p50 161 px en este corpus).

**Usar la serie de posiciones en metros como verdad-terreno sería circular**, porque es la misma
señal con la que el detector dispara. La estabilización NO es circular: es un ajuste geométrico de
2D sobre puntos del fondo, independiente de la pose, de la homografía y del detector de pasos.

Los marcadores de pie se dibujan en el mosaico **de dos colores distintos, uno por pierna**, para
poder seguir cuál es cuál, pero **el conteo se hace por la posición de la planta contra el suelo, no
por el marcador**: si el marcador está mal puesto se anota como cuadro dudoso.

## Lo que se publica del conteo

Por cada (clip, tirador) contado, en `conteo_ojo_pasos.csv`:

`clip, lado, cuadro_inicio, cuadro_fin, n_zancadas_adelante, n_zancadas_atras, n_marchas,
n_romper, n_medios_pasos, n_fondos, n_dudosos, notas`

Y la cifra que zanja la discusión:

**recall de zancadas** = zancadas detectadas por el detector que caen sobre una zancada contada a
ojo, dividido por las zancadas contadas a ojo. Con su tolerancia declarada: una detección cuenta como
acertada si su rango `[inicio, fin]` **solapa** el rango de la zancada contada, o cae a **≤ 4
cuadros** de él (0,13 s, que es menos de lo que dura una zancada).

## Qué decidirá esto

- si a ojo hay del orden de **10 marchas por tirador** y el detector da 5,7, el detector **pierde la
  mitad** y hay que arreglarlo;
- si a ojo hay del orden de **6**, el detector está bien y la expectativa era alta;
- y si a ojo hay 10 zancadas pero 5 marchas, entonces **los dos tenemos razón** y lo que había era un
  convenio sin declarar, que es el caso que este fichero existe para evitar.

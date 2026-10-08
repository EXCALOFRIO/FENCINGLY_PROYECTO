# Criterios declarados ANTES de mirar ninguna salida de pointmap/normal/matting
Fecha: 2026-09-10. Escritos antes de la primera pasada de los tres modelos.

## Controles
POSITIVOS (3 clips x 2 tiradores = 6 recortes), verificados a ojo:
  SentimentalLargeBedlingtonterrier video 125, AdorableFittingAsiansmallclawedotter video 250,
  SoreEasygoingEmperorshrimp video 240 (npz 65 + f0 175).
NEGATIVOS (7 recortes) = cajas del MISMO tamano que las de tirador, trasladadas a zonas
  SIN NINGUNA PERSONA, verificadas a ojo una por una. Descartados por inspeccion:
  'suelo_alto' (salen oficiales) y 'grada_vacia' (dudosa).

## C1 matting
Discrimina si alfa_medio(NEG) < alfa_medio(POS)/3. Si alfa_medio(NEG) >= alfa_medio(POS)/3,
matting INVENTA silueta donde no hay nadie.

## C2 normal
Metrica: dispersion angular = mediana del angulo entre cada normal y la normal media del recorte.
Honesto si dispersion(NEG) < dispersion(POS)/2.
Extra sobre 'pared_lisa' (superficie plana real): dispersion < 15 grados.
Si sobre pared lisa da dispersion comparable a un cuerpo humano, INVENTA relieve.

## C3 pointmap
Metrica: se ajusta un plano por minimos cuadrados a la nube (x,y,z) del recorte y se mide
el RMS del residuo dividido por la profundidad mediana (residuo relativo).
Honesto si residuo_rel(NEG) < residuo_rel(POS)/2.
Extra sobre 'pared_lisa': residuo relativo < 2 %.
Si una pared plana da residuo comparable a un cuerpo con volumen, INVENTA estructura.

## C4 escala de pointmap (la prueba que decide si sirve para ubicar)
Mismo tirador, mismo cuadro, dos recortes: padding 0 % y padding 30 %.
Si |z_mediana(30%) - z_mediana(0%)| / z_mediana(0%) >= 10 %, entonces z depende del RECORTE
y no de la ESCENA: comparar profundidades entre dos tiradores recortados por separado
NO mide distancia sobre la pista.

## C5 coherencia entre tiradores
Distancia 3D entre centroides de los dos tiradores. Coherente si cae en [0,5 m, 8 m]
asumiendo unidades metricas. Antes hay que comprobar si las unidades son metros.

## C6 coste
Mediana de 10 pasadas tras 3 de calentamiento, a 384x288 y a resolucion nativa del
procesador. VRAM pico con torch.cuda.max_memory_allocated. GPU COMPARTIDA: los tiempos
llevan esa salvedad.

---

# RESULTADOS: declarado contra medido (2026-09-10, RTX 3060 compartida)

Todo a 1024x768 salvo donde diga otra cosa. 6 recortes POS, 7 NEG del mismo tamano.

| criterio | umbral declarado | medido | veredicto |
|---|---|---|---|
| C1 matting | alfa NEG < alfa POS/3 = 0,100 | NEG max **0,0067** | **PASA** (15x de margen; separacion minPOS/maxNEG = **44,6x**, sin solape) |
| C2 normal SIN mascara | disp NEG < POS/2 = 13,20 deg | 4 de 7 negativos por encima (hasta 31,07 deg) | **FALLA** |
| C2 normal CON mascara de seg | idem = 16,19 deg | 0 de 7 por encima; POS 16,32-38,80 / NEG 1,63-13,98 | **PASA**, margen fino 1,17x |
| C2 extra pared lisa | disp < 15 deg | **2,67 deg** | PASA |
| C3 pointmap SIN mascara | residuo NEG < POS/2 = 4,67 % | 5 de 7 por encima; NEG mediana 12,19 % > POS mediana 9,34 % | **FALLA** |
| C3 pointmap CON mascara de seg | idem = 0,48 % | 7 de 7 por encima | **FALLA** |
| C3 extra pared lisa | residuo < 2 % | **23,77 %** | **FALLA por 12x** |
| C4 escala de pointmap | cambio de z < 10 % al ampliar el recorte | relleno 30 %: mediana **30,0 %**, peor caso **694 %**; 5 de 6 lo superan | **FALLA** |
| C5 unidades metricas | separacion en [0,5 m, 8 m] asumiendo metros | las unidades NO son metros: el cuerpo mide 0,073 / 0,613 / 0,176 u (error x24 / x2,9 / x10) | **NO EVALUABLE como se declaro** |

## Hallazgos que no estaban en los criterios

1. **pointmap devuelve al tirador como un plano.** Con la mascara de seg, el residuo de un
   plano ajustado dentro del cuerpo es de **0,72-1,61 %** (mediana 0,95 %). Un cuerpo humano
   no es un plano: pointmap entrega un recorte plano, no volumen.
2. **pointmap coloca al tirador DETRAS de su propio fondo en 3 de 6 casos.** El escalon de
   profundidad persona-fondo es de 0,0015 a 0,1836 u, y el ruido dentro del cuerpo es de
   4,2 a 29,7 veces mayor que ese escalon.
3. **pointmap solo funciona a 1024x768.** Su scale_head es un Linear(6144,512) de tamano
   fijo; 896x672, 768x576, 640x480, 512x384, 384x288 y 256x192 lanzan RuntimeError.
   La palanca de 7,8x por resolucion medida para pose NO le aplica.
4. **seg y normal rellenan con bordes (do_pad=True).** Sin pasar source_sizes al
   post-proceso, un cuadro completo de 720p sale con 58 % de relleno estirado dentro de la
   medida. Corregido; cambio, por ejemplo, el residuo de pointmap de 51,7 % a 6,5 %.
5. **A 384x288 la puerta de seg se rompe.** La pared impresa da frac_persona **0,6867**,
   por encima de TODOS los tiradores (max 0,5805): minPOS/maxNEG = 0,5x, con solape.
   A 1024x768 no hay solape, pero el margen es solo **1,35x**, no el factor 4-8 del plan.
6. **La blancura HSV de coste cero NO generaliza.** Sobre mis 7 negativos: POS 0,335-0,720
   contra NEG 0,007-0,697; el suelo gris da 0,6965, por encima de 4 de 6 tiradores.
   Solapa. Asi que matting no es redundante con esa heuristica.

## Corroboracion del bloqueo de resolucion de pointmap, desde su propio config

El `config.json` de `facebook/sapiens2-pointmap-0.4b` declara **`"scale_final_input_size": 6144`**,
un tamano de entrada FIJO para la cabeza de escala. Por eso solo funciona a 1024x768: a
384x288 el mapa de caracteristicas aplanado da 1152 y el `Linear(6144,512)` no multiplica.
No es un fallo de mi invocacion, es una restriccion declarada en el modelo.

La tarjeta de modelo (README.md) NO dice nada sobre escala, unidades metricas ni ambiguedad
de escala. O sea que el error de x2,9 a x24 en la altura del cuerpo no esta advertido en la
documentacion: hay que medirlo, como se ha hecho aqui.

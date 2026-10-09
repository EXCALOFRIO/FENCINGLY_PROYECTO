# V-JEPA 2 y la ZONA del tocado: protocolo DECLARADO ANTES de medir en PRUEBA

Escrito el 2026-10-09 por la tarde/noche, antes de entrenar ninguna sonda y antes de mirar ningun
acierto en PRUEBA. Lo que se cambie despues de ver cifras se apunta abajo como "Cambio n", con su
fecha y hora. Script: `02_validacion/herramienta/sonda_vjepa_zona.py` (nuevo, no toca ningun fichero
existente). Salidas: `02_validacion/auditoria/vjepa_zona_2026-10-09/`.

## La pregunta

El resumen de vídeo de V-JEPA 2 (ventana [ancla-4s, ancla+0,5s], recorte de la pareja +10%/+15%,
256x256) ¿sabe DONDE en la pista ocurre el tocado? En el recorte se ven las rayas pintadas cerca de
los pies y, en muchas pistas, la zona de aviso (ultimos 2 m) tiene otro color. La zona humana
(`strip_location`, `CA.ZONAS` = `fotl_warning_box, fotl_half, middle, fotr_half, fotr_warning_box`)
es la verdad de referencia. La produccion de hoy (etapa 11, homografia de las rayas) solo contesta
en una fraccion de los clips. La pregunta es si V-JEPA sirve de RESPALDO donde la homografia no
contesta, y/o si se puede FUSIONAR con `zona_final` donde si contesta.

## Punto de partida (medido hoy, ANTES de tocar V-JEPA; `02_validacion/herramienta/mide_zona.py`)

Nota: estas cifras salen 2026-10-09 por la noche y difieren un poco de las citadas en la tarea
(5.472/3.363) porque la sonda de `initiated` que esta corriendo en paralelo (PID 52476) no toca
`zona_`, pero el pipeline de produccion si ha seguido escribiendo clips nuevos durante el dia; es la
misma medida, con mas clips.

    clips con strip_location          5.544
    con zona_ (csv)                   5.525  (sin zona_ 19)
    contestan (homografia, "zona")    3.398 de 5.525 (0,615)
       zona (homografia)  0,480   middle 0,357   dif +0,123  IC95 clip [+0,103,+0,143]
       zona_final (fusion angulo)  0,516  (mismos 3.398 clips)
    no contestan                      2.127  (sin homografia confiable 1.971; sin luz 134; sin pies 22)

## El conjunto que usa V-JEPA (interseccion con los embeddings)

De los 5.525 clips con `strip_location`+`zona_`, **5.524 tienen `vjepa_<clip>.npz`** (solo falta 1):
la interseccion es practicamente el conjunto entero. El reparto AJUSTE/PRUEBA es el YA FIJADO en
`02_validacion/auditoria/zona_cobertura_2026-10-09/torneos_ajuste.txt` (74 torneos = AJUSTE; el
resto = PRUEBA), sin tocarlo, igual que en `ANGULO_FUSION_2026-10-09.md`:

    AJUSTE   3.394 clips, 74 torneos   contestan 2.045   no contestan 1.349
    PRUEBA   2.130 clips, 75 torneos   contestan 1.353   no contestan   777

(Estas cifras de AJUSTE/PRUEBA coinciden, clip a clip, con las de `ANGULO_FUSION_2026-10-09.md`:
confirma que la interseccion con V-JEPA no introduce un sesgo de seleccion.)

`y` (humana) en los NO-CONTESTAN, que es el denominador del uso (1):

    AJUSTE (n=1.349): middle 646 (0,479) | fotl_half 251 | fotr_half 240 | fotl_warning_box 136 | fotr_warning_box 76
    PRUEBA (n=777):   middle 382 (0,492) | fotr_half 136 | fotl_half 132 | fotl_warning_box 81 | fotr_warning_box 46

Por arma, en los NO-CONTESTAN: AJUSTE florete 1.050 / espada 194 / sable 105; PRUEBA florete 468 /
espada 189 / sable 120. (Florete domina los que no contestan: es el arma con mas clips sin anotar
`strip_location` en general y la que mas sufre la falta de 2 rayas transversales, ver
`ZONA_COBERTURA_2026-10-09.md`.)

## El modelo: sonda lineal sobre `media`, impar por construccion

`media` es `(2,1024)` fp32 por clip: vista 0 el recorte real, vista 1 su espejo horizontal. La zona
es IMPAR: espejar el video intercambia `fotl_half<->fotr_half` y `fotl_warning_box<->fotr_warning_box`
y deja `middle` fija. Se impone por construccion, no se aprende:

    a = w . m0           b = w . m1                    (w: (1024,) float64, UNA sola proyeccion)
    u = (a - b) / 2                                     (posicion lateral odd: u(espejo) = -u)
    c1 = softplus(p1) > 0     c2 = c1 + softplus(p2) > c1     tau = softplus(p_tau) + 1e-3
    centros = [-c2, -c1, 0, +c1, +c2]                    (simetricos por construccion)
    sesgos  = [B_w, B_h, B_m, B_h, B_w]                  (3 libres, reusados en espejo)
    logit_k = -(u - centro_k)^2 / (2 tau^2) + sesgo_k

Es la MISMA idea que `zona(x)` (umbrales +-2 m / +-5 m sobre una posicion 1-D) pero con `u` una
posicion LATENTE que sale de `media` en vez de metros de homografia, y con umbrales blandos
(gaussianas) en vez de dUros. Por construccion algebraica, intercambiar `m0<->m1` (y, si se usa la
caja, cambiar de signo su rasgo) da exactamente `logit_k(espejo) = logit_{4-k}(original)`: el
control del espejo (decidir con las vistas intercambiadas y comparar con la clase espejada) tiene
que dar **1,0000** SIEMPRE, para cualquier valor de los parametros (es estructural, no aprendido);
se comprueba en PRUEBA antes de leer ningun acierto.

**Rasgo barato opcional (caja)**: posicion horizontal del recorte `caja` en el fotograma,
`cx_norm = (0,5*(caja_x1+caja_x2)/W - 0,5) * 2`, con `W` el ancho del video (de `poses_<clip>.npz`,
NUNCA del video ni de `tok`). Es impar por construccion igual que arriba (`a += wc*cx`,
`b -= wc*cx`). Dos variantes declaradas: **sin caja** (la que decide los usos 1 y 2) y **con caja**
(se da siempre al lado, informativo, en los tres usos, por si acaso la posicion del recorte sola ya
dice algo).

**Entrenamiento**: L-BFGS a convergencia (300 iter, `strong_wolfe`; no rejilla de epocas: el modelo
es lineal con 1.030 parametros, L-BFGS converge en segundos) minimizando entropia cruzada de 5
clases + L2 sobre `w` (y `wc` si aplica); perdida con peso de clase opcional (inverso a la
frecuencia) para no colapsar a predecir `middle` siempre, que es lo primero que intentaria un
acierto bruto con clases desbalanceadas.

## Hiperparametros: eleccion por particion INTERNA dentro de AJUSTE

Los 74 torneos de AJUSTE se barajan con semilla **20261009** y se parten 75/25 por torneo (sin mirar
PRUEBA): ~56 torneos entrenamiento interno, ~18 validacion interna. Rejilla, FIJADA ahora:

    L2    en {1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0}
    pesos en {no, si}  (14 combinaciones), para CADA variante (sin caja, con caja)

Se elige la combinacion que maximiza el acierto de 5 clases en la validacion interna (TODOS los
clips de AJUSTE-validacion, contesten o no la homografia); a igualdad, la de menor entropia cruzada
(NLL) en esa misma validacion. Con lo elegido, se reentrena en TODO AJUSTE (3.394 clips) para sacar
la sonda final que se evalua en PRUEBA.

**Fusion (uso 2)**: logistica multinomial estandar (`sklearn.linear_model.LogisticRegression`,
`multi_class=multinomial`) sobre `[onehot(zona_final) (5), logits de la sonda sin-caja final (5)]`,
SOLO en los clips de AJUSTE que contestan (`zona_final` existe). Rejilla de `C` (inverso de L2):
`{0,01; 0,1; 1; 10; 100}`, elegida con el mismo reparto interno 75/25 (restringido a los clips que
contestan), por acierto de 5 clases en la validacion interna. Declarado AHORA, antes de medir: los
logits de la sonda que entran en la fusion son los de la sonda final (entrenada en TODO AJUSTE, no
con validacion cruzada "fuera de pliegue"), asi que la ganancia de la fusion en AJUSTE puede estar
un poco optimista (la sonda ya vio esos clips al entrenarse) — pero el criterio que decide es el de
PRUEBA, donde la sonda y la fusion NUNCA vieron esos clips ni esos torneos.

## Usos y criterios de adopcion (en PRUEBA, UNA sola vez)

1. **RESPALDO**: en los clips de PRUEBA que la produccion NO contesta (n=777), la sonda (sin caja)
   decide la zona (argmax) y se compara con contestar siempre `middle` en esos MISMOS clips. Pasa si
   `acierto(sonda) - acierto(middle) >= +0,05` con IC95 por clip (bootstrap, 2.000) que no cruza
   cero. Se da tambien el IC95 por torneo (mas exigente, informativo si no decide).
2. **FUSION**: en los clips de PRUEBA que SI contestan (n=1.353), se compara el acierto de la
   fusion (arriba) con el acierto de `zona_final` sola. Pasa si `fusion - zona_final >= +0,02` con
   IC95 **por torneo** (bootstrap, 2.000) que no cruza cero.
3. **INFORMATIVO**: la sonda sola (sin caja) en TODOS los clips de PRUEBA (n=2.130) contra
   `middle`; no decide nada, se da con su IC95 por clip.

Se mide una vez y se escribe el resultado aunque no pase. Se da, ademas, para los tres usos: por
arma, por `via_zona` (donde exista: `homografia`, `angulo_torneo`, `homografia_sin_modelo`, o vacio
si no contesta), matriz de confusion 5x5 y "a una zona o menos" (`|indice(y) - indice(pred)| <= 1`).
La variante CON caja se da al lado en los tres usos, informativa (no decide).

## Que falta por responder (recomendaciones, no medidas aqui)

- Si la sonda ATENTIVA sobre los tokens completos (`tok`, 11,6 GB) merece la pena: se recomienda SI
  el uso 1 o el uso 3 muestran señal real (aunque no lleguen al umbral) y la caja NO explica toda la
  ganancia (si `con caja` sola ya iguala a `media`, la señal es solo "donde esta el recorte en el
  fotograma", no el contenido del video, y la atentiva no añadiria nada).
- Si un recorte que incluya MAS pista (menos ceñido a la pareja) ayudaria: se infiere de si el fallo
  se concentra en clips donde la pareja esta centrada en el recorte (sin rayas visibles) frente a
  clips donde se ve algo de pista.

---

## Nota de metodo (antes de los resultados): una corrida de humo con `--rapido`

Antes de la corrida real se ejecuto `--rapido` (rejilla de 1 solo punto, documentado en el propio
`--help` del script como "SOLO para probar", igual que en `sonda_vjepa.py`) para comprobar que el
codigo no rompia. Esa corrida SI llego a imprimir cifras de PRUEBA con una sonda degenerada (L2
1e-4, sin pesos, la unica de la rejilla reducida). No se tomo ninguna decision de hiperparametro ni
de protocolo a partir de ella: la rejilla declarada arriba no cambio, y la corrida real (sin
`--rapido`, la que cuenta) eligio sus hiperparametros SOLO con la particion interna de AJUSTE, igual
que si la de humo no hubiera existido. Se declara por transparencia, no porque cambie el criterio.

## Resultados en PRUEBA (una sola corrida, `sonda_vjepa_zona.py`, sin `--rapido`)

Hiperparametros elegidos en la particion interna de AJUSTE (56 torneos entrenamiento / 18 torneos
validacion, 2.614 / 780 clips), SIN mirar PRUEBA:

| variante | L2 | pesos de clase | acierto interno (5 clases) |
|---|---:|---:|---:|
| sonda SIN caja (decide) | 3e-1 | no | 0,437 |
| sonda CON caja (informativa) | 3e-1 | no | 0,444 |
| fusion (C de la logistica) | C=1,0 | — | 0,597 (en la particion interna, solo clips que contestan) |

**Control del espejo**: `1,0000` en AJUSTE y en PRUEBA, para las dos variantes (sin caja y con
caja). Es exacto por construccion algebraica (no depende de los parametros aprendidos), y asi sale.

### USO 1 (RESPALDO) -- PRUEBA sin respuesta de produccion, n=777

    sonda (sin caja)  0,489   middle  0,492   dif -0,003
    IC95 por clip   [-0,041, +0,036]
    IC95 por torneo [-0,050, +0,050]
    (informativo, CON caja) sonda 0,492  dif +0,000  IC95 clip [-0,041, +0,039]

**CRITERIO (dif >= +0,05 e IC95 por clip > 0): NO PASA.** La sonda NO distingue de contestar
siempre `middle` en los clips donde la homografia falla: la diferencia es practicamente cero y el
intervalo cruza ampliamente el cero por los dos lados.

Por arma (informativo, sin IC — n pequenos): espada n=189 sonda 0,466 / middle 0,397 (+0,069);
florete n=468 sonda 0,466 / middle 0,502 (-0,036); sable n=120 sonda 0,617 / middle 0,600 (+0,017).

### USO 2 (FUSION) -- PRUEBA con respuesta de produccion, n=1.353

    zona_final  0,525   fusion  0,473   dif -0,052
    IC95 por clip   [-0,079, -0,024]
    IC95 por torneo [-0,085, +0,002]

**CRITERIO (dif >= +0,02 e IC95 por TORNEO > 0): NO PASA.** La fusion no solo no gana: **acierta
MENOS que `zona_final` sola** (dif -0,052, IC95 por clip claramente negativo; por torneo el extremo
superior casi toca cero pero sigue sin cruzarlo). Anadir los logits de la sonda a `zona_final` daña
la decision en vez de ayudarla.

Por arma: espada n=314 fusion 0,379 / zona_final 0,433 (-0,054); florete n=814 fusion 0,494 /
zona_final 0,545 (-0,052); sable n=225 fusion 0,529 / zona_final 0,578 (-0,049). **Baja en las TRES
armas.** Por `via_zona`: `angulo_torneo` n=300 +0,020 (la unica que mejora); `homografia` n=907
-0,084; `homografia_sin_modelo` n=146 +0,000.

### USO 3 (INFORMATIVO) -- TODO PRUEBA, n=2.130

    sonda (sin caja)  0,450   middle  0,404   dif +0,046
    IC95 por clip   [+0,021, +0,069]
    IC95 por torneo [+0,017, +0,077]
    (informativo, CON caja) sonda 0,450  dif +0,046  IC95 clip [+0,021, +0,070]
    a una zona o menos: sin_caja 0,899   con_caja 0,898

No decide nada (es informativo), pero el intervalo NO cruza cero ni por clip ni por torneo: hay una
senal real, aunque pequena. **Esta senal viene casi entera de los clips que la produccion YA
contesta**, no de los que fallan (ver por `via_zona` abajo) — es la misma lectura que el uso 1 y el
uso 2 ya mostraban por separado.

Por arma: espada n=503 sonda 0,384 / middle 0,288 (+0,095); florete n=1.282 sonda 0,449 / middle
0,397 (+0,052); sable n=345 sonda 0,551 / middle 0,600 (**-0,049**, la sonda pierde en sable).

Por `via_zona` (incluye `sin_zona` = los que no contestan, el mismo uso 1):

    angulo_torneo         n=300  sonda 0,383  middle 0,300  dif +0,083
    homografia            n=907  sonda 0,445  middle 0,372  dif +0,074
    homografia_sin_modelo n=146  sonda 0,411  middle 0,356  dif +0,055
    sin_zona              n=777  sonda 0,489  middle 0,492  dif -0,003

Las tres categorias donde la homografia SI contesta ganan +0,055 a +0,083; donde NO contesta
(`sin_zona`, el uso 1) la ganancia es -0,003. El patron es consistente: la sonda no aporta donde
hace falta, solo donde el problema ya esta resuelto.

**Matriz de confusion (PRUEBA, sonda sin caja; filas = humano, columnas = sonda; orden
`fotl_warning_box, fotl_half, middle, fotr_half, fotr_warning_box`):**

    fotl_warning_box    0  177   81   33    0
    fotl_half           0  212  175   23    0
    middle              0  165  553  143    0
    fotr_half           0   24  188  194    0
    fotr_warning_box    0    8   46  108    0

**La sonda elegida (sin pesos de clase, porque esa combinacion gano la particion interna) NUNCA
predice `fotl_warning_box` ni `fotr_warning_box`: las dos columnas de los extremos son CERO.**
Colapsa a un clasificador de facto de 3 clases (`fotl_half`, `middle`, `fotr_half`). La variante
`pesos=si` de la rejilla SI intentaba predecir los extremos, pero perdia acierto global en la
validacion interna en TODAS las combinaciones de L2 (0,29-0,36 contra 0,42-0,44 sin pesos): no se
eligio porque el criterio declarado es maximizar el acierto de 5 clases, no el equilibrio entre
clases.

## Veredicto

**NO PASA ninguno de los dos usos que decidian** (1: respaldo: NO PASA, dif -0,003, IC95 cruza
cero; 2: fusion: NO PASA, y ademas la fusion empeora frente a `zona_final`). El uso 3
(informativo) muestra una senal real pero pequena (+0,046, IC95 sin cruzar cero) que, por arma y
por `via_zona`, se explica casi entera por los clips que la produccion ya resuelve, no por los que
fallan. **No se adopta nada de este experimento**: no hay variable de entorno que tocar porque no
se escribe en produccion.

Lo que queda medido:

1. El recorte de V-JEPA (union de las cajas de los dos tiradores, +10 %/+15 %) **no trae la
   informacion que falta** en los 777 clips donde la homografia no ve 2 rayas transversales: la
   misma causa que ciega a la homografia (el borde cercano tapado por el rotulo o justo debajo de
   `BANDA_ROTULO`, `ZONA_COBERTURA_2026-10-09.md`) probablemente ciega tambien al recorte, que
   esta centrado en la pareja y no en el borde de la pista.
2. El rasgo barato de la posicion del recorte en el fotograma (`caja`) **no anade nada** sobre la
   `media` de los tokens (0,444 contra 0,437 interno; 0,450 contra 0,450 en PRUEBA informativo):
   lo poco que sabe la sonda no es "donde esta el recorte", es algo del contenido, pero ese algo no
   basta para distinguir los extremos.
3. La sonda elegida (maximizando acierto en validacion interna, como declara el protocolo) colapsa
   a 3 clases: nunca dice "zona de aviso". Con pesos de clase lo intenta pero pierde acierto total,
   asi que la regla de eleccion declarada no lo elige.

### Recomendaciones

- **Sonda atentiva sobre `tok` (11,6 GB, la correre yo cuando acabe la medida en marcha)**: hay un
  argumento mecanico a favor de probarla, no solo de intuicion: `media` es el promedio de los 512
  tokens espacio-temporales, que BORRA justo el dato posicional ("en que columna de la imagen
  cambia el color de la raya") que haria falta para distinguir `half` de `warning_box`. La atentiva
  trabaja sobre `tok` sin promediar (rejilla 4x4 por cuadro, 32 cuadros), que SI conserva esa
  estructura. Y el rasgo barato de `caja` (que si fuera solo "donde esta el recorte" ya deberia
  ayudar) no anadio nada, lo que descarta que la ganancia sea trivial de posicion y sugiere que si
  hay señal de contenido, esta en el detalle espacial que `media` destruye. Dicho esto, el uso 1
  (el que de verdad importa, el respaldo) dio CERO senal, y si la causa de fallo de la homografia es
  que el borde cercano ni siquiera ENTRA en el recorte (ver el punto 2 de abajo), ninguna sonda,
  por atenta que sea, va a inventar un dato que no esta en los pixeles. **Recomendacion: probarla,
  pero sin esperar que resuelva el respaldo si el recorte no cambia.**
- **Un recorte de V-JEPA con mas pista**: el recorte actual es MUY ceñido a los dos tiradores
  (+10 %/+15 % sobre la union de sus cajas) para que quepan en 256x256 con resolucion util en la
  hoja y el contacto; eso probablemente deja la franja de aviso (los 2 m del borde) FUERA del
  encuadre salvo cuando un tirador ya esta pegado al borde. La lectura de `ZONA_COBERTURA_2026-10-09.md`
  coincide: el 46,7 % de los fallos de homografia son "nunca se ven 2 rayas transversales", y a ojo
  el borde cercano esta tapado por el rotulo o justo debajo de el, no dentro del area de la pareja.
  Un recorte mas ancho (o un SEGUNDO recorte fijo, por ejemplo toda la franja inferior del
  fotograma, no solo la zona de la pareja) daria a V-JEPA la misma evidencia que le falta a la
  homografia. **Recomendacion: probar un recorte ancho (o un recorte secundario de la banda inferior
  del fotograma) antes de invertir mas en sondas sobre el recorte actual.**


"""LA MANO ARMADA POR EL GUANTE: una senal INDEPENDIENTE de la geometria, con su parche declarado.

EL PROBLEMA QUE VIENE DE ANTES. `cono_lib.mano_armada` elige la mano armada con un solo criterio
geometrico, el (a) «la mano con x mediana mas cerca del rival», que es literalmente «la mas
adelantada». Su propio docstring ya confiesa que los dos criterios de corroboracion que se probaron
fallaron y que la comprobacion a ojo se hizo en CUATRO tiradores. Cuatro no validan nada, y el
usuario reporta que falla a menudo.

POR QUE EL GUANTE Y NO OTRA GEOMETRIA. El guante no se deduce de la postura: se ve o no se ve. Es
una senal de APARIENCIA, asi que puede contradecir a la geometria, que es justo lo que hace falta
para tener dos criterios de verdad independientes y no dos versiones del mismo.

FUENTE PRIMARIA, VERIFICADA EN DISCO el 2026-09-11 (y con una correccion al encargo):
`C:\\Users\\Alejandro\\Downloads\\190675-tecnico esp.pdf`, FIE *Reglamento Tecnico*, edicion de
diciembre de 2025, en espaniol.

    t.115.1  «el esgrimista se presentara en la pista completamente listo para combatir: uniforme
             reglamentario, chaqueta cerrada, GUANTE Y ARMA AGARRADAS CON LA MANO ARMADA, el pasante
             conectado a la ficha del interior de la cazoleta. Unica excepcion para la careta, que
             debe ser agarrada con LA MANO NO ARMADA.»
    t.1      en el saludo, «colocan la careta bajo el brazo desarmado / levantan la cazoleta hasta
             la barbilla», o sea arma en una mano y careta en la otra.
    t.20.2   guante obligatorio para dar y recibir la leccion, y remite al **m.25** para pantalones
             y calcetines.
    t.27, t.28, t.29.3, t.30, t.48  usan «mano armada» y «mano no armada» como terminos definidos.

    CORRECCION AL ENCARGO, dicha y no escondida: **el articulo m.25 NO esta en este PDF**. Este PDF
    es el Reglamento TECNICO (articulos `t.`) y solo CITA el m.25, que vive en el Reglamento del
    MATERIAL, que no esta en disco en espaniol. Asi que lo verificado en fuente primaria es t.115.1,
    no m.25. Y una segunda salvedad del mismo tamanio: en el Reglamento Tecnico **no hay ningun
    articulo que PROHIBA un guante en la mano no armada**; lo que dice t.115.1 es que el guante y el
    arma van en la mano armada y la careta en la otra. Que la otra mano vaya desnuda es la practica
    normal del deporte, no una regla citable. Por eso aqui el guante se mide como RASGO y se compara
    contra una anotacion a ojo, y no se da por cierto porque lo diga el reglamento.

CRITERIO DECLARADO ANTES DE EJECUTAR. Todo lo que sigue estaba escrito antes de mirar una sola
cifra, y las constantes de abajo son las que se usaron.

 1. PARCHE. Semilado `r = R_PARCHE * |raiz - codo|`, o sea la escala la pone el antebrazo del propio
    tirador en el propio cuadro, y no un numero en pixeles que a 720p y a 1080p significaria cosas
    distintas. El centro se desplaza DISTALMENTE `DESPL_DISTAL * r` a lo largo de `codo -> raiz`:
    el parche va de `raiz - 0,65 r` a `raiz + 1,35 r`, o sea hacia los dedos y lejos de la manga.
    Esto es por el primer confundido que aviso el usuario: la chaqueta tambien es blanca.

 2. BLANCO. `V >= 150` y `S <= 70`, **importado** de `blancura_pareja`, que es la misma definicion
    que usa `validador_visual.blancura` y que ya se uso en las medidas de 60 clips de la seccion 18.
    NO se define aqui otro blanco: un umbral que se mueve entre medidas no deja comparar nada.

 3. LOS TRES RASGOS.
      `f_blanco`  fraccion de pixeles blancos del parche de la mano.
      `a_blob`    area de la mancha blanca conexa mas grande que toca el disco central, sobre el
                  area del parche. Es la idea del usuario de que el guante es un GUANTELETE y tapa
                  la muniera, asi que su mancha es mas grande que la de una mano desnuda.
      `f_manga`   `f_blanco` en un parche del MISMO tamanio centrado en el medio del antebrazo. Es
                  el control del primer confundido: si `f_manga` sale igual que `f_blanco`, el
                  parche esta leyendo manga y el rasgo no vale.
      `f_punio`   fraccion de pixeles saturados (`S >= 120`, `V >= 100`) del parche de la mano. Es
                  el control del segundo confundido: guante con punio de color.

 4. LA DECISION ES PAREADA Y POR CUADRO. Las dos manos del MISMO tirador estan en el MISMO cuadro,
    asi que comparten alumbrado; eso es lo unico que la seccion 18.3 encontro que significase algo.
    Por cuadro: `d_blanco = f_blanco(A) - f_blanco(B)`, gana la mayor si `|d_blanco| >=
    MARGEN_CUADRO`, y si no el cuadro se ABSTIENE. Las abstenciones se cuentan y se dicen.

 4b. PUERTA DE SOLAPE, anadida ANTES de medir el primer `f_blanco` y con el motivo a la vista. Al
    construir los mosaicos de anotacion aparecio que las DOS raices de mano caen a veces sobre el
    MISMO punto: `BabyishIdioticCockerspaniel` lado 1 da `sep` de 3 px con `r` de 54 px en el 100 %
    de sus cuadros, y `LavishDistantDikkops` lado 0 en el 86 %. Cuando eso pasa los dos parches leen
    los MISMOS pixeles y la comparacion no es que salga mal: es que no existe. Asi que un cuadro
    vota solo si `sep >= SOLAPE_MIN * r` con `SOLAPE_MIN = 2`, que es la condicion exacta de que dos
    parches de semilado `r` no se toquen. Sobre los 34 tiradores el reparto medido de `sep/r` tiene
    mediana 4,78, y solo 2 de 34 estan colapsados en mas de la mitad de sus cuadros. Y en esos dos
    casos la eleccion es INDIFERENTE para el cono, porque el vertice sale en el mismo pixel con
    cualquiera de las dos manos.

 5. LA DECISION ES UNA POR TIRADOR Y POR CLIP, por voto mayoritario sobre los cuadros que votan. Un
    tirador no cambia de mano en medio de un asalto: es un hecho del deporte, no un suavizado. Se
    reporta `frac_voto` = votos a la ganadora / votos emitidos, y la decision se llama SOLIDA solo
    si `frac_voto >= VOTO_SOLIDO` con al menos `VOTANTES_MIN` cuadros votando.

 6. SOLO POSES CON `origen == 0`. Una pose interpolada o sostenida no puede decidir que mano es la
    armada, porque su raiz de la mano no se midio: se relleno.

 7. EL NUMERO QUE DECIDE SI ESTO SIRVE, declarado antes de verlo:
      (i)  acierto por clip-tirador contra la anotacion a ojo de
           `02_validacion/muestras/mano_dominante/verdad_mano.csv` **mayor o igual** que el del
           criterio (a) actual, que es la linea base gratis, y
      (ii) separacion `SEP_MIN >= 0,10` en la mediana de `f_blanco(armada) - f_blanco(la otra)`.
    Si no se cumplen los dos, se retira CON la cifra al lado, como `USA_CRUDAS = False` en
    `extrae_pareja.py`. No se vende.

 8. LA COMBINACION, tambien declarada antes: `mano_final` es el voto del guante cuando la decision
    del guante es SOLIDA, y el criterio (a) cuando no lo es. Y siempre se publica si los dos
    criterios CONCUERDAN, porque las discrepancias son los casos que valen.

 9. SEGUNDO CRITERIO DE APARIENCIA, DECLARADO DESPUES DE QUE EL PRIMERO FALLARA Y ANTES DE MEDIRLO.
    El punto 3 se midio y **no separa**: `f_blanco(armada) - f_blanco(la otra)` da mediana -0,008
    con el signo correcto en el 0,433 de los cuadros, o sea peor que la moneda y con el signo AL
    REVES. Y el signo invertido tiene una explicacion fisica que se puede convertir en criterio:
    **la CAZOLETA del florete es un cuenco metalico oscuro de 19 px a 720p y 29 px a 1080p que TAPA
    el guante**. O sea que el parche de la mano armada no es mas blanco: es mas OSCURO.

    Criterio declarado: `f_oscuro` = fraccion de pixeles con `V <= V_OSCURO = 60` en el mismo parche
    del punto 1, y gana la mano con `f_oscuro` MAYOR. El numero que decide, declarado antes de
    verlo: signo correcto por cuadro `>= 0,65` y acierto por tirador `>= 0,724`, que es el del
    criterio (a) sobre esta misma muestra. Si no, se retira igual que el blanco.

    V_OSCURO = 60 no se ajusta a estos datos: es el complemento del `V >= 150` de `blancura_pareja`
    con el mismo margen por debajo del medio del rango (150 esta 22 por encima de 128, y 60 esta 68
    por debajo; se elige 60 y no 106 porque el suelo de la pista y la sombra del uniforme ya andan
    por 90-120 y la cazoleta es metal negro). Queda declarado que es una eleccion, no un ajuste.

10. PUERTAS DE «CUADRO QUE SABE LA RESPUESTA», declaradas el 2026-09-11 ANTES de medirlas, y por un
    motivo que ya estaba medido: de las siete fallas del criterio (a) contra la anotacion a ojo,
    **cinco son cuadros en los que el tirador no esta encarado**, entre accion y accion, con el arma
    atras o abajo. En esos cuadros «la mano mas cerca del rival» es la desarmada POR CONSTRUCCION, y
    ahora mismo votan igual que los buenos. Si la decision es una por tirador y por clip, no todos
    los cuadros tienen que votar.

    Cinco puertas candidatas, y se MIDE cual separa en vez de elegir a ojo:

      `alcance`   los dos tiradores en medida de tocar: la menor de las cuatro distancias entre las
                  raices de mano de uno y las del otro, por debajo de **2 x HOJA_M = 1,80 m**. NO es
                  un umbral ajustado: cada hoja alcanza 0,90 m desde su mano (cota FIE m.8.4, ya
                  verificada en este modulo), asi que a 1,80 m las dos hojas pueden tocarse y a mas
                  no. La escala en metros sale del alto de caja de cada tirador con ALTURA_M.
      `extendido` `max(|hombro-raiz|_A, |hombro-raiz|_B)` por encima de su mediana del clip. Es
                  SIMETRICA entre las dos manos a proposito: mira si ALGUN brazo esta extendido, no
                  cual, asi que no puede favorecer a ninguna de las dos.
      `perfil`    anchura de hombros proyectada `|kp5 - kp6|` en metros, por debajo de su mediana del
                  clip. En guardia el tirador esta de PERFIL a la camara y ese segmento se acorta;
                  entre acciones se gira y se alarga. Rasgo barato y no estaba en ninguna tabla.
      `ev_accion` cuadro dentro de un evento `tipo=accion` de `02_validacion/pista/eventos.csv`, que
                  dejo el agente de la pista. Los eventos de `accion` son de los dos lados (`lado`
                  -1).
      `ev_brazo`  cuadro dentro de un evento `tipo=brazo` DE ESE MISMO tirador, o sea una extension
                  de brazo detectada por el otro agente. Es la mas selectiva y la mas independiente,
                  porque no la he calculado yo.
      `encarado`  **SEXTA PUERTA, anadida despues de medir las cinco primeras y antes de medirla
                  ella**, y el motivo es el diagnostico: las cinco primeras miden MAGNITUDES
                  (una distancia, una extension, una anchura, una pertenencia a un intervalo) y
                  ninguna mide el SIGNO de hacia donde mira el tirador. Un tirador de espaldas al
                  rival tiene la misma anchura de hombros proyectada que uno encarado, y puede estar
                  perfectamente en alcance. Asi que se declara una variable CON SIGNO: la nariz
                  (indice 0, que en careta tiene mas confianza que las manos) tiene que caer del
                  lado del rival respecto al punto medio de los hombros, o sea
                  `sign(x_nariz - x_hombros) == sign(x_rival - x_hombros)`. Es un test de signo, sin
                  umbral que ajustar.

    Y dos combinaciones: `alcance+extendido` y `alcance+extendido+perfil`.

    EL NUMERO QUE DECIDE, declarado antes de verlo. Una puerta cuesta tiradores sin respuesta, asi
    que hay que juzgar las dos cosas a la vez. Con `VOTA_MIN = 10` cuadros minimos tras la puerta:

        MEJORA           acierto >= 0,90 sobre los tiradores que decide  Y  <= 4 de 27 sin decision
        GANANCIA CON COSTE  acierto >= 0,90 pero mas de 4 sin decision
        NO MEJORA        acierto < 0,90

    La cifra a batir es **0,815** del voto por cuadro sin puerta y **0,741** del operativo por
    medianas. Y se publica siempre, junto al acierto, **cuantos tiradores se quedan sin decision**:
    una puerta que acierte el 100 % dejando a la mitad sin respuesta no sirve para la tuberia.

    RESULTADO, medido. **NINGUNA PUERTA LLEGA A 0,90, asi que ninguna se adopta**, y las cifras
    quedan aqui para que no se vuelvan a probar a ciegas (`careo()` las reimprime todas):

        puerta                     acierto  decide  sin decidir   acierto POR CUADRO dentro/fuera
        sin puerta                  0,815     27         0              0,805  /   -
        alcance                     0,778     27         0              0,819  / 0,797   (+0,022)
        extendido                   0,741     27         0              0,786  / 0,825   (-0,040)
        perfil                      0,741     27         0              0,762  / 0,848   (-0,086)
        ev_accion (de la pista)     1,000      2        25              0,939  / 0,791   (+0,148)
        ev_brazo  (de la pista)     1,000      2        25              0,907  / 0,802   (+0,104)
        **encarado (signo)**        0,852     27         0              **0,860 / 0,582  (+0,277)**
        alcance+extendido           0,727     22         5              0,772  / 0,814
        alcance+extendido+perfil    0,769     13        14              0,772  / 0,809
        alcance+encarado            0,778     27         0              0,852  / 0,783

    LO QUE ESO ENSENIA, y es mas util que si hubiera funcionado:

    1. **La hipotesis era buena y la variable estaba mal elegida.** `encarado`, que es la unica CON
       SIGNO, separa por cuadro **0,860 contra 0,582, +0,277** sobre 3.479 cuadros. Las de magnitud
       no separan y **dos van al REVES**: los cuadros con el brazo mas extendido aciertan 0,040
       MENOS, y los de perfil 0,086 menos. Un tirador de espaldas al rival tiene la misma anchura de
       hombros que uno encarado; la anchura no distingue mirar de estar mirando al otro lado.
    2. **Y aun asi la puerta apenas mueve la decision por clip (0,815 -> 0,852, +1 caso)**, porque el
       fallo NO es de unos cuantos cuadros: es del TIRADOR ENTERO. El acierto por cuadro del
       criterio (a) por tirador es **bimodal**: los 20 tiradores acertados tienen p50 **0,960** y los
       7 fallados p50 **0,226**, con cuatro de ellos en 0,000, 0,000, 0,053 y 0,226. Cuando un
       tirador esta mal en el 100 % de sus cuadros **no hay ningun cuadro bueno que salvar**, y esa
       es la razon de fondo de que quitar cuadros no arregle nada.
    3. Cada puerta arregla algun caso suelto pero rompe otros, y el balance sale negativo. Los tres
       que **ninguna** puerta arregla son justo los de acierto por cuadro 0,000 / 0,000 / 0,226.
    4. `ev_accion` y `ev_brazo` YA ESTAN REMEDIDAS contra el `eventos.csv` FINAL, `mtime_ns`
       **1789119066880271000** (478 filas, 11 clips), y las cifras de la tabla de arriba son de las
       versiones viejas con cobertura de 2 tiradores. Las buenas, sobre 38 tiradores:

           puerta                acierto   decide   por cuadro dentro/fuera
           `ev_accion`            0,850      20      0,868 / 0,784  (+0,084)
           `ev_brazo`             0,824      17      0,808 / 0,829  (**-0,021**)
           `encarado+ev_accion`   0,950      20      0,955 / 0,748  (+0,207)

       **`ev_brazo` CAMBIA DE SIGNO al medirlo con cobertura**: parecia +0,104 sobre 107 cuadros y da
       -0,021 sobre 609. Tercer caso en este modulo de «cifra buena con poca cobertura que se cae»,
       detras de la ampliacion del hombro (15/15 -> 22/23) y de los `1,000 sobre 2 tiradores`.
       `ev_accion` aguanta, se encoge de +0,148 a +0,084, y es **la unica senial positiva que no
       calculo yo**. Ninguna pasa la puerta: 0,850 y 0,824 por acierto, y 18 y 21 sin decidir.
       `mano_guante.json` guarda el `mtime` de cada tirada; si el fichero vuelve a cambiar, hay que
       remedir y comparar los dos.

    Y LA MARCA POR TIRADOR TAMPOCO PAGA, medida con las dos variables y umbral declarado en 0,50:
    `en alcance < 0,50` marca 19 de 27 (los 8 no marcados aciertan 0,750, PEOR que 0,815);
    `encarado < 0,50` marca 4 de 27 y caza 2 de los 7 fallos, pero los 23 no marcados aciertan
    0,783, tambien peor. La unica marca que paga sigue siendo la firmeza del voto, `frac_a_voto >=
    0,85`: **0,895 sobre 19 de 27 tiradores**. Ver el punto 11.

11. QUE DEBE USAR LA TUBERIA, y esto es para el agente del cono. Marca `mano dudosa` cuando
    `frac_a_voto < 0,85`. Sobre esta muestra eso deja **19 de 27 tiradores (0,70) con acierto
    0,895** y aparta 8 con acierto 0,625. **Y tiene un punto ciego medido**: los 2 fallos que se
    cuelan tienen `frac_a_voto` de 1,00 y son exactamente los dos tiradores con acierto por cuadro
    0,000. Es por construccion: la unanimidad del voto mide consistencia, no verdad, y un error
    sistematico es perfectamente consistente. Asi que `firme` sirve para no heredar decisiones
    flojas, **no** para detectar decisiones seguras y equivocadas.

12. CRITERIOS DEL PROPIO CUERPO, que NO miran al rival. Declarados el 2026-09-11 antes de medirlos.

    EL HECHO DE ESGRIMA, que no es una correlacion de estos datos: en guardia **el pie adelantado es
    el del MISMO LADO que el brazo armado**. Un diestro adelanta el pie derecho y lleva el arma en la
    mano derecha. Lo que lo hace valioso aqui es que es una correspondencia DENTRO DEL CUERPO, asi
    que sobrevive a los dos casos que rompen todo lo demas: los dos tiradores en los que el arma esta
    en la mano del lado OPUESTO al rival durante todo el clip, donde «la mas cerca del rival» falla
    por construccion en el 100 % de los cuadros.

    ANTES DE NADA, EL MAPA DE LOS PIES, MEDIDO Y NO SUPUESTO (precedente: los indices 91-95 del plan
    caian sobre la careta). Tres pruebas sobre los 17 clips, 32 tiradores:

      T1, el talon es el punto del grupo mas cercano al tobillo. Distancia mediana en anchuras de
          hombro: 15 -> 0,505 | 16 -> 0,460 | **17 -> 0,209**, y 18 -> 0,498 | 19 -> 0,463 |
          **20 -> 0,196**. Asi que **17 y 20 son los TALONES** y 15-16 y 18-19 los dedos.
      T2, cv de la longitud del hueso, que es menor en el emparejamiento correcto:
          cadera-rodilla (10,12)=0,079 contra (10,11)=0,113; rodilla-tobillo (12,14)=0,109 contra
          (12,13)=0,242; tobillo-talon (14,20)=0,146 contra (14,17)=0,423.
          **El eslabon hombro-cadera NO lo resuelve el cv**: (6,10)=0,064 contra (6,9)=0,069, un 8 %,
          o sea nada. Eso queda dicho porque es justo el eslabon que une el lado de la mano con el
          del pie.
      T3, coherencia lateral, que SI lo resuelve: fraccion de cuadros en que el signo del desvio
          lateral respecto al medio del par coincide con el del hombro 6. Mediana sobre 32 tiradores:
          **cadera 10 -> 0,989** contra cadera 9 -> 0,011; rodilla 12 -> 0,934; **tobillo 14 -> 0,837**
          contra tobillo 13 -> 0,163; talon 20 -> 0,837; y de control, codo 8 -> 0,989 y mano A ->
          0,949, que reproducen el mapa que ya estaba.

      CADENA DE LADO VERIFICADA:
          mano A (raiz 21) - codo 8 - hombro 6 - cadera 10 - rodilla 12 - tobillo 14 - pie 18,19,20
          mano B (raiz 42) - codo 7 - hombro 5 - cadera  9 - rodilla 11 - tobillo 13 - pie 15,16,17

    Y AHORA EL PUNTO DELICADO: definir «adelantado» SIN volver a usar al rival.

      `pie_x`      el eje de cada pie es `talon -> punto medio de los dos dedos`. En guardia el pie
                   de delante apunta al frente, o sea a lo largo de la pista, y el de atras queda
                   casi perpendicular. Gana el pie cuyo eje esta mas alineado con la HORIZONTAL de la
                   imagen (la pista es horizontal en este corpus), medido con |cos|. La mano armada
                   es la del MISMO LADO que ese pie. No aparece el rival en ninguna parte.
      `pie_cadera` igual pero el eje de referencia es el de las caderas (`cadera 9 -> cadera 10`) en
                   vez de la horizontal de la imagen. Asi no se supone que la pista sea horizontal:
                   es completamente intrinseco.
      `nariz_dir`  sustituye al rival por la propia cara: direccion de frente
                   `dir = sign(x_nariz - x_medio_hombros)`, y gana la mano con `dir * x_raiz` mayor.
                   Es el criterio (a) con el rival cambiado por la orientacion del propio tirador.
                   La nariz (indice 0) tiene MAS confianza que las manos (0,998-1,000 contra
                   0,648-0,987, seccion 2), asi que como punto es mejor dato que la mano.

      LA SEGUNDA VIA QUE SE PROPUSO, «el pie que queda por delante del plano del tronco», NO SIRVE, y
      no hace falta medirlo para saberlo: proyectar los tobillos sobre el eje de las caderas
      (`cadera 9 -> cadera 10`) devuelve **el LADO, no el delante**. El tobillo del lado A siempre
      proyecta positivo y el del lado B siempre negativo, esten como esten los pies, porque el eje de
      las caderas es el eje LATERAL del cuerpo. Para sacar «delante» de ese eje hace falta saber
      hacia donde mira el tirador, o sea el rival o la nariz: seria el mismo problema con otro
      nombre. Queda escrito, no disimulado.

    PUERTA DE ADOPCION: **la misma que se declaro en el punto 10 y no se toca**: acierto >= 0,90
    sobre los tiradores que decide y <= 4 de 27 sin decision, con `VOTA_MIN = 10` cuadros. La cifra a
    batir es 0,815 del voto por cuadro y 0,741 del operativo.

    Y se mide la CONCORDANCIA entre `pie_x` y `nariz_dir`, que son dos seniales de signo
    independientes entre si -- una mira los pies y la otra la cara -- y si discrepan se saca el
    recorte.

13. LAS DOS COMBINACIONES, declaradas antes de medirlas y despues de ver los tres aciertos sueltos
    (`pie_x` 0,852, `nariz_dir` 0,889, `pie_cadera` 0,778), que es lo que hace falta para poder
    elegir QUE combinar sin elegir COMO:

      `cuerpo_2de2`  contesta solo cuando `pie_x` y `nariz_dir` COINCIDEN, y deja al tirador sin
                     decision cuando discrepan. Es la version prudente: cambia cobertura por
                     acierto, y las dos senales son independientes (pies contra cara), asi que una
                     coincidencia significa algo.
      `mayoria_3`    mayoria simple entre `a_voto`, `pie_x` y `nariz_dir`. Siempre contesta, porque
                     con tres votos binarios no hay empate.

    La puerta de adopcion sigue siendo la del punto 10 y **no se mueve**: acierto >= 0,90 con <= 4
    de 27 sin decision.

14. EL HOMBRO, idea del usuario, declarada el 2026-09-11 antes de medirla. Sus palabras: «el brazo
    armado POR REGLA siempre tiene que estar POR DELANTE del brazo no armado, tipo se mide por el
    hombro». Es cierto y es de esgrima: la guardia es de perfil con el hombro del arma adelantado. Y
    tiene tres ventajas sobre la mano:

      - los hombros 5 y 6 son una estructura **grande y bien resuelta**: 18,7-21,5 % del alto de la
        caja (seccion 2), mientras que las dos raices de mano **colapsan en el mismo pixel** en 2 de
        34 tiradores;
      - mira el tren SUPERIOR, asi que es independiente del pie, que mira el inferior;
      - y aguanta el caso que preocupa al usuario, «si se pasa el brazo por detras»: la MANO cruza el
        cuerpo, el HOMBRO no.

    EL PUNTO DELICADO ES EL MISMO QUE CON LAS CADERAS: «delante» necesita una direccion, y si se
    define como «el hombro mas cerca del rival» se ha vuelto al criterio que falla. Tres vias:

      `hombro_pie`    proyectar los hombros sobre el eje del PIE GANADOR (`talon -> dedos` del pie
                      que gana en `pie_x`), que es una direccion intrinseca. Gana el hombro con
                      proyeccion mayor. No es independiente del pie: es su CORROBORACION, y eso ya
                      vale porque dice si el pie se equivoca.
      `hombro_nariz`  igual pero con la direccion que da la nariz. Es `nariz_dir` cambiando las
                      raices de mano por los hombros, o sea la misma senial mejor condicionada.
      `hombro_conf`   **especulativo, y se mide antes de creerlo**: de perfil un hombro tapa al otro,
                      asi que puede que el de delante sea el mas visible. Gana el hombro con `sc`
                      mediana mayor.

    Y UNA MEDIDA DEL CASO QUE PREOCUPA AL USUARIO, que no es un criterio: `frac_cruza_brazo` =
    fraccion de cuadros en que el orden en x de las dos MANOS es el contrario del orden en x de los
    dos HOMBROS. Es la proxy de «el brazo pasa por detras» o «invierte la guardia». Si es alta en los
    tiradores que fallan, el hombro deberia arreglarlos.

    Y una combinacion declarada para atacar la COBERTURA, que es el precio de lo adoptado:
      `pie_o_hombro`  si `pie_x` es firme, `pie_x`; si no y `hombro_nariz` es firme, `hombro_nariz`;
                      si no, sin decision.

    PUERTA DE ADOPCION: la misma de siempre, 0,90 con <= 4 sin decidir, **y no se mueve**.

    RESULTADO DEL PUNTO 14, sobre los mismos 27 tiradores:

        criterio        acierto   con su frac >= 0,85
        `hombro_pie`     0,852     **1,000 sobre 14**
        `hombro_nariz`   0,889     **1,000 sobre 11**
        `hombro_conf`    0,741     1,000 sobre 2   <- RETIRADO: 0,741 es exactamente la linea base
        `pie_o_hombro`   **1,000 sobre 15**, o sea 0,56 de cobertura contra 0,48 del pie solo

    1. **El hombro empata con el pie y con la nariz, no los bate.** `hombro_pie` da lo mismo que
       `pie_x` (0,852) y `hombro_nariz` lo mismo que `nariz_dir` (0,889), caso por caso. O sea que
       medir por el hombro en vez de por la mano **no cambia la respuesta**, lo cual tiene sentido:
       lo que decide es la DIRECCION (pie o nariz), no el punto que se proyecta.
    2. **Pero si amplia la cobertura de lo adoptado sin coste**: `pie_o_hombro` decide 15 de 27 en
       vez de 13 y sigue sin fallar ninguno. Esta implementado en `cono_lib` con el interruptor
       `USA_HOMBRO_SI_PIE_FLOJO`, **apagado por defecto** porque la adopcion la decidio el usuario y
       lo que decidio fue el pie.
    3. **`hombro_conf` era especulativo y se retira con su cifra**: 0,741, que es clavado a la linea
       base. El hombro mas visible no es el de delante.
    4. **La preocupacion del usuario, medida y PEQUENIA.** `frac_cruza_brazo`, la fraccion de cuadros
       en que el orden en x de las manos es el contrario del de los hombros -- o sea «se pasa el
       brazo por detras» o «invierte la guardia» -- tiene **p50 0,04, p90 0,14 y maximo 0,32**. Y en
       los siete tiradores que el criterio (a) falla sube solo de **0,03 a 0,07**. El cruce existe,
       pero **no es la causa de los fallos**: la causa sigue siendo que el tirador no esta en guardia.
    5. **Y los tres irreducibles siguen igual**, con una excepcion: `WhiteMarriedGuineafowl` lados 0
       y 1 fallan con las cuatro senales del cuerpo, y `PowerfulAllLarva` lado 1 lo acierta
       `hombro_nariz` y nadie mas. Lo importante es que **los tres tienen el pie NO firme** (0,73,
       0,77 y 0,72) y por tanto **la politica adoptada los marca `dudosa` y los excluye**, que es
       justo lo que se pedia de ella.

    RESULTADO DE LOS PUNTOS 12 Y 13, medido sobre los mismos 27 tiradores:

        criterio                    acierto   sin decidir   frac_voto p50
        (a), el operativo            0,741         0            -
        a_voto                       0,815         0          0,95
        `pie_cadera`                 0,778         0          0,81
        **`pie_x`**                  0,852         0          0,79
        **`nariz_dir`**              **0,889**     0          0,83
        `cuerpo_2de2`                0,885         1            -
        `mayoria_3`                  0,889         0            -

    NINGUNO LLEGA A 0,90 Y LA PUERTA NO SE MUEVE, asi que ninguno se adopta como operativo. Pero
    `nariz_dir` se queda a UN CASO (24 de 27; 0,90 pediria 25) y **bate al criterio (a) en cuatro
    casos**, de 0,741 a 0,889, sin mirar al rival ni una vez. Y lo que sale mejor de todo es el cruce
    con la firmeza, con el corte 0,85 que ya estaba declarado para `frac_a_voto` y que aqui se
    REUTILIZA sin volver a elegirlo:

        criterio     frac >= 0,85                frac < 0,85
        a_voto       0,895  (17 de 19)           0,625  (5 de 8)
        **pie_x**    **1,000  (13 de 13)**       0,714  (10 de 14)
        nariz_dir    **1,000  (11 de 11)**       0,812  (13 de 16)

    Y ESO CIERRA EL PUNTO CIEGO DE `frac_a_voto`, en parte. Los tres tiradores que ningun criterio
    acierta tienen `frac_pie_x` de 0,73, 0,77 y 0,72, o sea **por debajo de 0,85**, mientras que
    `frac_a_voto` les da 1,00, 0,83 y 0,68. La firmeza del PIE si los marca como dudosos y la del
    voto contra el rival no. El precio es cobertura: `pie_x` firme decide 13 de 27 (0,48).

    EL HECHO DE ESGRIMA SE CONFIRMA EN EL CASO PARA EL QUE SE PROPUSO. `UglyNiceAplomadofalcon` lado
    1, el tirador con el arma en la mano del lado OPUESTO al rival en el 100 % de los cuadros, lo
    aciertan `pie_x` Y `nariz_dir` y lo falla todo lo que mira al rival. Pero
    `WhiteMarriedGuineafowl` lados 0 y 1 y `PowerfulAllLarva` lado 1 siguen fallando con los dos, y
    la razon es de esgrima tambien: **esos tiradores no estan EN GUARDIA** en los cuadros
    utilizables, estan parados en su linea o andando entre acciones, y la correspondencia
    pie-delantado con mano-armada es una propiedad de la GUARDIA, no de una persona de pie.

    CONCORDANCIA de las dos senales de signo, que era la esperanza de cerrar el caso: por clip
    coinciden en **26 de 27**, y por cuadro solo en **0,77 de mediana** (p10 0,51, p90 0,96). O sea
    que **no son tan independientes como parecia a nivel de clip** y `cuerpo_2de2` no anade nada
    (0,885 sobre 26). La unica discrepancia por clip es `WhisperedTenderDeinonychus` lado 0, donde la
    nariz acierta y el pie falla con `frac_pie_x` baja; el recorte con los ejes de los pies pintados
    esta en `discrepa_pie_nariz_WhisperedTenderDeinonychus_l0.png` y se ve el motivo: en dos de los
    tres cuadros el pie de delante es el A y en el tercero el B, porque el tirador esta andando.

15. VERDAD AMPLIADA A 38 TIRADORES, y **todas las cifras de los puntos 10 a 14 son de la etapa de
    27**: se dejan como estaban para que se vea que se movio. La ampliacion son los 6 clips nuevos
    del corpus (12 tiradores) anotados en TANDA CIEGA -- mosaicos `ciego_NN.png` sin nombre de clip
    ni de lado, orden barajado con semilla fija, clave en `clave_ciego.csv` no mirada hasta despues
    de anotar, y segunda pasada de 6 cuadros sobre los 10 que quedaron en `certeza=media`. Ninguna
    letra cambio entre las dos pasadas y 6 subieron de media a alta; un tirador
    (`IlliterateDisloyalFulmar` lado 0) paso a `?` en una tercera mirada.

        criterio             27 tiradores      38 tiradores
        (a), la linea base       0,741            0,763
        `a_voto`                 0,815            0,842
        `pie_x`                  0,852          **0,895**
        `nariz_dir`              0,889          **0,895**
        `hombro_pie`             0,852          **0,895**
        `hombro_nariz`           0,889          **0,895**
        `hombro_conf`            0,741            0,763   <- sigue clavado a la linea base
        `guante`                 0,444            0,447   <- sigue retirado
        `cuerpo_2de2`            0,885 (26)       0,917 (36)
        **`mayoria_3`**          0,889          **0,921**  <- PASA LA PUERTA
        pie firme (ADOPTADO)     13 de 13         **17 de 17 sin fallo**
        `pie_o_hombro`           15 de 15         22 de 23 = 0,957  <- ya NO es gratis

    1. **LA PUERTA DECLARADA SE PASA POR PRIMERA VEZ.** Era «acierto >= 0,90 sobre los tiradores que
       decide y <= 4 sin decision», declarada en el punto 10 y nunca movida. `mayoria_3` -- la
       mayoria simple entre `a_voto`, `pie_x` y `nariz_dir`, declarada en el punto 13 ANTES de
       medirse -- da **0,921 sobre 38 con cobertura total**. Y `cuerpo_2de2` da 0,917 dejando 2 sin
       decidir, que tambien pasa.
    2. **Sus tres fallos son los tres tiradores irreducibles**, y **los tres ya salen marcados
       `dudosa`** por la politica adoptada. En los 17 firmes `mayoria_3` acierta 1,000, y en los 21
       dudosos acierta **0,857** contra **0,667** del criterio (a). De ahi el interruptor
       `cono_lib.FALLBACK_MAYORIA`, que solo cambia lo que se devuelve en la parte que la tuberia
       debe excluir: apagado da 0,816 sobre los 38, encendido **0,921**, y el subconjunto firme no se
       toca (17 de 17 en los dos casos).
    3. **Lo adoptado aguanta y su enunciado mejora solo**: de «no ha fallado ninguno de 13» a **«no
       ha fallado ninguno de 17»**, con la cobertura en 17 de 38 (0,45).
    4. **La ampliacion con el hombro ha dejado de ser gratis**: 15 de 15 con 27 tiradores, 22 de 23
       con 38. El fallo es `HairyCourteousEarwig` lado 0, donde el pie dice B con 0,69 (no firme) y
       el hombro dice A con 0,90 (firme) y la verdad es B. Justo para eso sirve mas muestra.
    5. **DOS MODOS DE FALLO NUEVOS** que no estaban en los 27, y es lo mas valioso de la tanda:
       - **el tirador SIN CARETA, sosteniendola en una mano** (`AdorableVeneratedDuiker` lado 0). El
         criterio (a) elige la mano de la careta y falla con 39 px de margen. Por t.115.1 la careta
         va en la mano NO armada, asi que la careta es **evidencia reglamentaria de la mano
         contraria** y se anoto con el `cue` nuevo `careta`. El pie y el hombro lo aciertan los dos.
       - **el margen grande otra vez** (`HairyCourteousEarwig` lado 1, 111 px, `cue=arma alta`): un
         tercer «seguro y equivocado», que el pie y el hombro aciertan.
       Los tres irreducibles siguen siendo los mismos tres y siguen siendo todos «no en guardia»:
       **no ha aparecido ninguna causa nueva entre los irreducibles**.
    6. Y en el subconjunto de `certeza=alta` (28 tiradores, los que tienen arma u hoja visible)
       `pie_x`, `hombro_pie`, `hombro_nariz` y `mayoria_3` dan **0,929**. Es la mejor estimacion
       libre de error de anotacion, pero **no es un subconjunto aleatorio**: es donde se ve el arma,
       que correlaciona con que el tirador este en una postura clara. No se puede trasladar al corpus.

16. LA CARETA COMO EVIDENCIA DE LA MANO CONTRARIA. Declarado el 2026-09-11 antes de medirlo.

    DE DONDE SALE: al anotar los 6 clips nuevos aparecio `AdorableVeneratedDuiker` lado 0, un tirador
    que **entre acciones se quita la careta y la lleva en una mano**. El criterio (a) elige la mano de
    la careta y falla con 39 px de margen. Y hay respaldo reglamentario, ya verificado en este modulo:
    **t.115.1 pone la careta en la mano NO ARMADA**. Asi que una careta detectada en una mano es
    evidencia de que la armada es **la otra**, que es una senial que no teniamos.

    POR QUE PODRIA FUNCIONAR DONDE EL GUANTE FALLO, y esto es un argumento, no una medida: el guante
    fallo porque el rasgo de blancura **leia la manga** (pearson 0,681 con `f_manga`) y porque hay
    guantes oscuros. La careta es **grande, compacta y OSCURA sobre un tirador vestido de blanco**, o
    sea que el contraste va en el sentido contrario y no compite con la chaqueta.

    RASGO DECLARADO. Mismo parche del punto 1 (semilado `r = 0,42 |raiz - codo|`, desplazado
    `0,35 r` hacia los dedos), y:

      `oscuro = V < 150`, que es la **inversion de la mitad en V** de la definicion de blanco del
                proyecto (`V >= 150` y `S <= 70` en `blancura_pareja`). Se invierte solo la V y **no**
                se anade la clausula de saturacion, porque una careta puede ser plateada o de color y
                la clausula la echaria fuera. El 150 no es un umbral nuevo: es el del proyecto.
      `a_osc`   area de la mancha OSCURA conexa mas grande que toca el disco central, dividida por el
                area del parche. Es adimensional y no depende de la escala del tirador.
      `comp`    compacidad de esa mancha = area / area de su caja envolvente. Una careta o una
                cazoleta son compactas (~0,7-0,9); una sombra o un trozo de fondo, no.

    EL CONFUNDIDO QUE HAY QUE MEDIR ANTES DE DAR NINGUNA CIFRA, y es el que puede tumbarlo todo: **la
    CAZOLETA del arma tambien es oscura y compacta, y esta justo en la mano ARMADA**. Si el rasgo no
    distingue careta de cazoleta, apunta al lado contrario del correcto. Asi que la primera medida no
    es el acierto, es el SIGNO:

        se mide `a_osc(mano armada) - a_osc(la otra)` sobre los 38 tiradores anotados.
        si sale POSITIVO, el rasgo esta leyendo la CAZOLETA y no la careta, y entonces «oscuro =
        mano no armada» **apunta al revés**: la senial se retira y lo que queda es, si acaso, un
        criterio de la cazoleta.
        si sale NEGATIVO, hay hueco para la careta, y entonces hay que separar los dos objetos por
        TAMANIO, porque la careta mide ~25 cm (unos 43 px a 720p) y la cazoleta 12 cm (19 px), o sea
        un factor 4-5 en area.

    Y LA COBERTURA, que es la otra pregunta: **cuantos cuadros tienen careta en mano**. Solo puede
    pasar entre acciones, con el tirador descubierto, asi que se espera POCA cobertura. Si sale alta
    precision y baja cobertura, la senial entraria como **voto extra cuando aparezca**, no como
    criterio permanente, que es lo que la arquitectura de puntos 12-15 ya permite.

    LIMITE DE MUESTRA, dicho antes de medir: **hay UN solo tirador con careta en mano verificada a
    ojo**. Con uno no se puede establecer separacion de nada. Lo que si se puede medir con n=38 es el
    SIGNO del confundido, y eso es lo que decide si la via sigue viva.

17. CUANTOS CUADROS HACEN FALTA PARA FIARSE DE `frac`. Declarado antes de medirlo.

    DE DONDE SALE: `WastefulYawningAmericanquarterhorse` es el clip mas corto, 300 cuadros y **48
    utilizables**, y sale con `frac_pie` de 0,90 y 1,00, los dos lados firmes y los dos correctos.
    Eso NO demuestra que la firmeza aguante con pocos cuadros: **con 48 cuadros una `frac` alta es mas
    facil de obtener por azar que con 250**. Convertir ese matiz en un numero es lo que se mide aqui.

    MEDIDA DECLARADA: remuestreo. Para cada tirador con al menos `N_LARGO = 120` cuadros que votan, se
    toma su `frac` del clip completo como referencia y se sacan `N_BOOT = 400` submuestras de tamanio
    `n` con reposicion, para `n` en 10, 20, 30, 50, 100. De cada tamanio se publica:

      `sd(frac)`        dispersion de la fraccion de voto, que dice cuanto se mueve el numero;
      `falsa_firmeza`   P(`frac_sub >= 0,85` | el clip completo esta POR DEBAJO de 0,85), o sea la
                        probabilidad de declarar firme a un tirador que no lo es. **Este es el que
                        importa**, porque es el error que mete decisiones malas en la parte que la
                        tuberia usa.
      `firmeza_perdida` P(`frac_sub < 0,85` | el clip completo esta EN o POR ENCIMA de 0,85), el error
                        contrario, que solo cuesta cobertura.

    EL NUMERO QUE DECIDE, declarado antes de verlo: **el minimo `n` con `falsa_firmeza <= 0,10`**. Si
    ni con 100 cuadros baja de 0,10, se dice que el corte 0,85 no es fiable a ningun tamanio de los
    probados y hay que subirlo o exigir mas cuadros.

    LIMITE DECLARADO: el remuestreo con reposicion sobre los cuadros de un clip **subestima** la
    variabilidad real, porque los cuadros de un clip estan correlacionados en el tiempo y una
    submuestra de 20 cuadros de un mismo clip no equivale a un clip de 20 cuadros. Asi que la cifra
    que sale es una **cota por lo bajo** de la falsa firmeza, y eso queda dicho.

18. EL CONJUNTO RESERVADO, y la regla que se juzga con el. **ESTE PUNTO SE ESCRIBIO ENTERO ANTES DE
    ANOTAR Y ANTES DE MEDIR NADA SOBRE ESOS 20 TIRADORES.** Es lo que le faltaba a la combinacion del
    punto 17 de 25.24, que estaba elegida DESPUES de ver que conjuntos se solapaban.

    EL RESERVADO: 10 clips, 20 tiradores, elegidos con **semilla 20260911**, distinta de la 20260910
    con la que se eligio todo lo demas, asi que **no se solapan con los 23 clips de entrenamiento** y
    **no han participado en elegir ninguna regla**. Estan en `RESERVADO`. Los diez son del corpus
    etiquetado, ocho a 720p y dos a 1080p.

    LA REGLA QUE SE JUZGA, que es la del punto 17 y no otra:

        si `pie` es firme (`frac_pie >= 0,85`)      ->  la mano del PIE
        si no, y la puerta `encarado + ev_accion` decide  ->  su mano
        si no                                        ->  sin decision, `dudosa`

    El ORDEN es parte de la regla y ya estaba medido: al revés (puerta primero) el entrenamiento baja
    de 1,000 a 0,964, porque el unico fallo de la puerta es justo el acierto del pie.

    EN ENTRENAMIENTO DA 28 DE 28, cobertura 0,74. Esa es la cifra que hay que desconfiar.

    CRITERIO DE ACEPTACION, fijado AHORA y sin haber visto los 20:

      (i)   acierto en el reservado **>= 0,90** sobre los tiradores que la regla decide. Con la
            cobertura esperada de 0,74 eso son ~15 decididos, y **0,90 sobre 15 permite 1 fallo y no
            2**. Queda dicho asi para que no haya interpretacion despues.
      (ii)  cobertura **>= 0,50** de los 20. Una regla que conteste a 6 de 20 no sirve para la
            tuberia aunque acierte.
      (iii) y que **bata a lo adoptado hoy en el propio reservado**: mas cobertura que `pie firme`
            solo, sin bajar su acierto.

      Si se cumplen los tres, la regla se adopta. Si (i) falla, **la regla se ha caido** y se dice,
      igual que se dijo del hombro y de `ev_brazo`.

    Y SE PUBLICAN LAS DOS CIFRAS POR SEPARADO, entrenamiento y reservado, **sin agregarlas**. La
    diferencia entre las dos ES la medida de cuanto me engane eligiendo post-hoc, y es el numero mas
    util que puede salir de aqui. Agregarlas borraria justo esa informacion.

    PRIMER RESULTADO, Y ESTA SUPERADO. QUEDA ESCRITO PORQUE LA RAZON DE QUE NO VALIERA IMPORTA.
    Cuando se midio por primera vez, el veredicto no fue «pasa» ni «se cae» sino **no se puede
    juzgar**:

        **NINGUNO de los 19 tiradores del reservado tenia eventos de `accion`.** `eventos.csv`
        cubria 11 clips y **los once estaban en el entrenamiento**. Asi que en el reservado la
        segunda mitad de la regla -- `encarado + ev_accion` -- **no se ejecutaba ni una vez**, y
        `REGLA_18` se reducia exactamente a `pie firme`: los mismos 12 tiradores.

        (i) acierto 1,000 se cumplia; (ii) cobertura 12 de 19 = 0,63 se cumplia; (iii) «mas
        cobertura que `pie firme` solo» **NO**, porque anadia CERO. Luego no se adoptaba, y no por
        haber fallado sino porque aquel reservado no podia probarla.

    SEGUNDO RESULTADO, Y ES EL VIGENTE (seccion 25.29 de `04_documentacion/BITACORA_2026-09.md`). `eventos.csv` crecio a
    **22 de 22 clips, los 10 reservados incluidos**, asi que la segunda mitad de la regla **si se
    ejecuta**. Medido de nuevo una sola vez y con el criterio congelado sin tocarlo:

        (i)   acierto >= 0,90 en el reservado      ->  **1,000**
        (ii)  cobertura >= 0,50 de los 20          ->  **19 de 19**
        (iii) mas cobertura que el pie firme solo  ->  **12 -> 19**, y 1,000 en los dos

    Las tres se cumplen y **la regla se adopta**: `USA_ACCION_SI_PIE_FLOJO = True`. Medido en
    `mano_guante.csv`, el reservado se reparte **12 por `pie_firme` y 8 por `encarado_accion`**.

    LO QUE NO CAMBIA CON LA ADOPCION, y es la confusion que costo una cifra publicada nueve veces:
    el **1,000 con cobertura 19 de 19 es de LA CASCADA**. La **regla del pie sola** sigue cubriendo
    **12 de esos 19** (sin fallo), y el criterio del pie sin su puerta de firmeza acierta **0,895
    sobre 19** y 0,881 sobre 59. Atribuir a la regla del pie el acierto y la cobertura de la cascada
    es lo que hacia la ficha publica, y esta corregido.

    LO QUE SI VALIDA EL RESERVADO, y era la mitad que el usuario adopto: **`pie firme` da 12 de 12
    sin fallo con cobertura 0,63** en clips que no vio. Y su marca funciona fuera de muestra: el
    criterio del pie se equivoca en 2 de los 19, `InsecurePettyCreature` lado 1 y
    `MediumShoddyAmphiuma` lado 0, y **los dos tienen `frac_pie` = 0,58**, o sea que los dos salen
    marcados `dudosa` y ninguno se cuela.

    Y LA CIFRA QUE EL COORDINADOR PEDIA -- la diferencia entrenamiento contra reservado -- NO SE PUEDE
    LEER COMO ENGANIO POST-HOC, y esto es lo mas util que sale de aqui:

        criterio          entrenamiento (38)   reservado (19)
        (a), linea base        0,763              **0,947**
        `a_voto`               0,842              1,000
        `pie_x`                0,895              0,895
        `nariz_dir`            0,895              1,000
        `mayoria_3`            0,921              1,000
        `guante`               0,447              0,263
        pie firme              17 de 17           12 de 12

    **El reservado es MAS FACIL que el entrenamiento, y la prueba es que la propia linea base sube de
    0,763 a 0,947**: si el salto fuera sobreajuste mio, el criterio (a) no se moveria, porque no lo he
    elegido ni tocado yo.

    ATRIBUCION CORREGIDA (la primera version de este bloque la tenia al reves): el reservado **SI es
    una muestra aleatoria** del corpus, elegida con semilla 20260911 sin mirar ninguna metrica de
    calidad; que saliera con las dos poses al 100 % fue el resultado, no el criterio. **Lo que no es
    aleatorio es el ENTRENAMIENTO**, que se fue formando con los clips que se estaban investigando
    PORQUE daban problemas (el arbitro entregado como tirador, el trozo de espectador del borde, el
    tirador medio fuera de cuadro, las dos raices colapsadas). Consecuencia:

        entrenamiento -> cota por lo BAJO, porque esta enriquecido en casos duros
        reservado     -> mejor estimacion a nivel de CORPUS, porque es la unica muestra aleatoria

    Asi que **no hace falta un tercer conjunto**: hacen falta las dos leidas como lo que son, una el
    rendimiento en los casos duros y la otra el rendimiento tipico.

    LA OBJECION DE INGENIERIA NO LA RESUELVE UN BUEN RESULTADO: `cono_lib` **no debe depender de**
    `eventos.csv`, que es un fichero de otro agente y ha cambiado cuatro veces. Si la regla pasa, la
    forma de cablearla es que **el consumidor pase los tramos de accion como argumento** a
    `mano_armada` -- una mascara booleana (T,) igual que `vota` en `mano_armada_voto` -- o mejor, que
    esos tramos se precalculen **a un array dentro del propio NPZ de poses**, que es donde vive el
    resto de la procedencia (`origen`, `hueco`, `mask`). Lo que no se acepta es que la libreria lea un
    CSV.

19. EL ARBITRO NO ESGRIME: deteccion por CONDUCTA, no por apariencia. **Criterio declarado el
    2026-09-11 antes de medirlo.**

    POR QUE NO POR APARIENCIA. Las tres ausencias visibles del arbitro estan medidas y **las tres
    fallan por separado**: la blancura se la come el lame dorado (seccion 18.3, contraste 0,477 entre
    dos tiradores correctos), la careta como objeto oscuro es un detector de oscuridad con precision
    ~1/35 (punto 16), y la hoja es invisible en el 81 % del corpus, que es 720p (etapa E). Sumar tres
    seniales debiles que comparten confundidos no arregla nada.

    LA CUARTA AUSENCIA, que es de CONDUCTA y no de apariencia: **el arbitro no esgrime**. No hace
    nunca un a fondo, no extiende el brazo, anda de forma continua en vez de alternar avance y
    retroceso, y no tiene la estructura de acercarse y alejarse de un asalto. Y es **inmune al lame, a
    la iluminacion y a la resolucion**, que son los tres confundidos que han tumbado todo lo anterior.

    RASGO DECLARADO: **eventos de esgrima por segundo util de ese lado**, contados sobre los eventos
    que ya calcula el agente de la pista en `eventos.csv`:

        `ev_esgrima` = numero de eventos de tipo `fondo` o `brazo` de ESE lado
        `tasa`       = `ev_esgrima` / segundos utiles del lado

    Se usan `fondo` y `brazo` y NO `marcha` ni `romper` a proposito: andar y retroceder los hace
    tambien el arbitro, mientras que el a fondo y la extension de brazo son especificos de esgrimir.

    CRITERIO DECLARADO: **un lado con CERO eventos `fondo` y CERO eventos `brazo` en mas de
    `SEG_MIN_ARB = 4,0` segundos utiles NO es un tirador.** Los 4 segundos salen de la cifra ya medida
    por el otro agente, 0,56 a 1,75 pasos por segundo y tirador: a la tasa mas baja, 4 segundos son
    mas de 2 acciones esperadas, asi que cero en 4 segundos ya es raro para un tirador. No se barre el
    umbral.

    COMO SE MIDE, y el numero que decide:
      positivos: los **2 arbitros** anotados a ojo (`PowerfulAllLarva` l0, `IdolizedFancyChickadee`
                 l1). Negativos: los **63 tiradores** restantes de las 66 filas de verdad.
      se publica la separacion de `tasa` entre las dos clases, y **sobre todo los FALSOS POSITIVOS**:
      un tirador defensivo que pase el asalto retrocediendo puede tener pocos a fondo, y si el rasgo
      lo marca, no sirve. Ese es el confundido a batir y **se mide, no se supone**.

    Y LO QUE CON 2 POSITIVOS SE PUEDE Y NO SE PUEDE HACER, dicho en la misma frase que el resultado:
    **con 2 positivos no se valida nada**. Lo que si se puede es DESCARTAR: si los arbitros no se
    separan de los tiradores ni con 2 casos, la via esta muerta y nos ahorramos perseguirla. Y si se
    separan limpiamente, lo que hay que dar no es un acierto sino **cuantos clips habria que anotar
    para validarlo**, que es una pregunta accionable.

    RESULTADO: **LA VIA NO SEPARA Y SE RETIRA.** Medida con el criterio congelado sobre los dos
    arbitros, y **falla en los dos, y en uno al reves**:

        clip                      lado  es        fondo  brazo   tasa
        PowerfulAllLarva            0   ARBITRO     0      4    **0,524**
        PowerfulAllLarva            1   tirador     0      1      0,131
        IdolizedFancyChickadee      1   ARBITRO     0      1    **0,100**
        IdolizedFancyChickadee      0   tirador     0      2      0,200

    El arbitro de `PowerfulAllLarva` da **0,524, por encima del p10 de los tiradores reales (0,493)**
    y **cuatro veces la tasa de su propio rival**. Ninguno de los dos da cero, que era la prediccion.
    Lo que si se midio bien es el confundido, y salia limpio: de 24 tiradores con eventos, **ninguno**
    tenia tasa cero y el mas flojo daba 0,333. O sea que el problema no es el falso positivo: es que
    **el positivo no existe**.

    Y HAY UN AGRAVANTE QUE INVALIDA LA PRUEBA AUNQUE HUBIERA SALIDO BIEN: `fondo` es 0 **por
    construccion** en los dos, porque **ninguno de los dos clips tiene homografia** y el criterio del
    fondo tiene el umbral en metros. Asi que solo se ejecuto la mitad `brazo`. Y en los 33 clips
    anotados **no hay ninguno que tenga arbitro Y homografia**, asi que hoy la prueba **no se puede
    hacer bien**.

    Y LA EXPLICACION ALTERNATIVA TAMBIEN ES FALSA, medida: se penso que el arbitro levanta el brazo
    para senializar y que el angulo separaria. Al contrario: el brazo del arbitro apunta **hacia abajo
    MAS** que el del tirador (p50 -76,1 grados contra -36,0) y el porcentaje de cuadros con el brazo
    por encima de 30 grados es **0,0 % en los dos arbitros**. El evento `brazo` les dispara por **el
    braceo de andar**, porque su criterio es «se alarga sobre su propia mediana» y andar tambien
    alarga.

    **NO SE REABRE** hasta que exista un clip con arbitro Y homografia. La variante que se podria
    intentar entonces es `accion` por la via sin homografia (columna `via_accion`), pero eso es una
    medida nueva y no esta hecha.

LO QUE ESTE MODULO NO HACE. No mira la hoja ni la cazoleta, no usa GPU y no vuelve a correr poses:
lee los NPZ ya calculados y decodifica solo los cuadros que va a usar.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import blancura_pareja as BP
import cono_lib as CL

PRUEBAS = CL.PRUEBAS
SCR = pathlib.Path(os.environ.get(
    "SCRATCH", r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\187cfd83-d8d9-4c4d-bed4-f255724f6b21\scratchpad"))
LISTA = SCR / "corpus_etiquetado.txt"
MUESTRAS = pathlib.Path(
    r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\mano_dominante")

V_CLARO, S_MAX = BP.V_CLARO, BP.S_MAX
"""IMPORTADOS, no redefinidos. `blancura_pareja` los fija en 150 y 70 y ahi se quedan."""

R_PARCHE = 0.42
DESPL_DISTAL = 0.35
R_DISCO = 0.45
R_MANGA = 1.20
MARGEN_CUADRO = 0.05
SOLAPE_MIN = 2.0
NCUADROS = 40
VOTO_SOLIDO = 0.80
VOTANTES_MIN = 20
SEP_MIN = 0.10
S_PUNIO, V_PUNIO = 120, 100
V_OSCURO = 60
UMBRALES_OSC = (60, 90, 110, 130)
"""Barrido POST-HOC del umbral de oscuro. El declarado es 60 y se midio con 60. Este barrido existe
porque la lupa sobre `WhiteMarriedGuineafowl` lado 1 mide la mano armada en **V 88-126**, o sea que
el 60 no la alcanzaba, y sin esta cifra el proximo agente repetiria el mismo umbral. Se publica solo
la SEPARACION, nunca un acierto: un umbral elegido despues de ver no tiene acierto honesto."""
REJILLA =((0.15, 0.0), (0.22, 0.0), (0.22, 0.5), (0.30, 0.35), (0.42, 0.0), (0.42, 0.35),
           (0.60, 0.35))
"""Pares `(R_PARCHE, DESPL_DISTAL)` del BARRIDO DE DIAGNOSTICO. No son criterios: son la prueba de
si el fallo del blanco viene del tamanio del parche o del rasgo. Un barrido de 7 geometrias con el
mejor elegido a posteriori NO es una medida de acierto, y por eso solo se publica la separacion y
con el aviso de que se ha elegido despues de ver."""
DENTRO_MIN = 0.60
"""Fraccion minima del parche que tiene que caer dentro del fotograma. Un parche medio fuera mide
el borde negro del video, no la mano."""

EVENTOS = pathlib.Path(
    r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\pista\eventos.csv")
ALCANCE_M = 2.0 * CL.HOJA_M
VOTA_MIN = 10
MEJORA_MIN = 0.90
SIN_DECISION_MAX = 4
PUERTAS = ("sin", "alcance", "extendido", "perfil", "ev_accion", "ev_brazo", "encarado",
           "alcance+extendido", "alcance+extendido+perfil", "alcance+encarado",
           "encarado+ev_accion")
"""`encarado+ev_accion` se anade el 2026-09-11 con el `eventos.csv` final, y se declara antes de
medirla: son las DOS UNICAS puertas con acierto por cuadro positivo de verdad -- `encarado` +0,250 y
`ev_accion` +0,084 -- y son independientes, porque una la calculo yo de la nariz y la otra la calcula
el agente de la pista de la marcha de los pies. La puerta de adopcion sigue siendo 0,90 con <= 4 sin
decidir y no se mueve."""
NARIZ = 0
PIES = {"A": {"talon": 20, "dedos": (18, 19), "tobillo": 14, "cadera": 10},
        "B": {"talon": 17, "dedos": (15, 16), "tobillo": 13, "cadera": 9}}
"""CADENA DE LADO VERIFICADA en T1, T2 y T3 (punto 12 del docstring), no copiada de ningun README.
`A` es el lado de la raiz 21 y `B` el de la 42."""
CUERPO = ("pie_x", "pie_cadera", "nariz_dir", "hombro_pie", "hombro_nariz", "hombro_conf")
RESERVADO = frozenset((
    "MediumShoddyAmphiuma", "WholeTallAlpineroadguidetigerbeetle", "NextImmaterialGoldfish",
    "GloomyBitterBluegill", "BountifulSilentEarwig", "CandidBronzeIndigowingedparrot",
    "InsecurePettyCreature", "CautiousBeneficialLacewing", "NervousHospitableChick",
    "IdolizedFancyChickadee"))
"""CONJUNTO RESERVADO del punto 18: 10 clips elegidos con semilla 20260911, que **no han participado
en elegir ninguna regla**. `careo()` publica entrenamiento y reservado POR SEPARADO y no los agrega:
agregarlos borraria la unica informacion que este conjunto puede dar."""


def clips():
    """Stems con NPZ de poses del formato nuevo Y presentes en la lista del corpus etiquetado.

    El filtro por la lista NO es cosmetico: el arbol de `D:/FENCINGLY_VIDEOS_RAW` tiene descargas de
    otros estudios con espada y sable, y muestrear con `os.walk` ya estropeo dos tiradas del
    proyecto. Aqui solo entra lo que esta en la lista.
    """
    corpus = set(LISTA.read_text(encoding="utf-8", errors="replace").split())
    out = []
    for f in sorted(PRUEBAS.glob("poses_*.npz")):
        stem = f.name[len("poses_"):-len(".npz")]
        if stem not in corpus:
            continue
        with np.load(f, allow_pickle=True) as z:
            if "KP" not in z.files:
                continue
        out.append(stem)
    return out


def carga(stem):
    with np.load(PRUEBAS / ("poses_%s.npz" % stem), allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def utiles(d):
    """Cuadros con LOS DOS tiradores medidos y las DOS manos y los DOS codos con confianza.

    Se exige para los dos lados a la vez porque la decision es pareada y porque asi cada fotograma
    decodificado sirve para cuatro parches, que es lo que hace esto barato.
    """
    SC, org, mask = d["SC"], d["origen"], d["mask"]
    ok = mask.copy()
    for lado in (0, 1):
        ok &= (org[:, lado] == 0)
        for M in CL.MANOS.values():
            ok &= CL.bueno(SC[:, lado], [M["root"], M["codo"]])
    return np.nonzero(ok)[0]


def lee_eventos():
    """Eventos del agente de la pista, como dict `(clip, lado, tipo) -> lista de (inicio, fin)`.

    Se lee UNA vez y se anota `mtime` en la salida, porque ese fichero lo esta escribiendo otro
    agente ahora mismo y sin la marca de tiempo la medida no seria reproducible.
    """
    import csv
    if not EVENTOS.exists():
        return {}, 0
    with open(EVENTOS, newline="", encoding="utf-8", errors="replace") as fh:
        filas = list(csv.DictReader(fh))
    out = {}
    for r in filas:
        """BLINDADO POR VALOR Y NO POR PRESENCIA DE CLAVE, el 2026-09-11. El fallo que hay que evitar
        es `r.get(k, nan)`, que **no protege de nada cuando la clave existe y vale `None`** o cadena
        vacia: la columna esta, la lectura pasa el `get`, y el `int()` revienta despues. El agente de
        la pista perdio dos clips con exactamente eso, teniendo el precedente escrito en su propio
        codigo. Aqui se valida el VALOR de las cinco columnas y la fila se salta entera si alguna no
        sirve, y se cuenta cuantas se saltan para que no desaparezcan en silencio."""
        try:
            clip, tipo = r["clip"], r["tipo"]
            lado, a, b = int(r["lado"]), int(r["inicio"]), int(r["fin"])
        except (KeyError, TypeError, ValueError):
            out.setdefault("_saltadas", []).append(1)
            continue
        if not clip or not tipo:
            out.setdefault("_saltadas", []).append(1)
            continue
        out.setdefault((clip, lado, tipo), []).append((a, b))
    n = len(out.pop("_saltadas", []))
    if n:
        print("   AVISO: %d filas de eventos.csv saltadas por valor no utilizable" % n)
    return out, EVENTOS.stat().st_mtime_ns


def mascara_accion(stem, T, ev=None):
    """Mascara booleana (T,) de cuadros dentro de un evento `accion`, para pasarsela a `mano_armada`.

    Existe para que **`cono_lib` no tenga que leer `eventos.csv`**: el consumidor construye la
    mascara aqui y la pasa como argumento. Si el clip no tiene eventos devuelve `None`, que NO es lo
    mismo que una mascara de ceros: `None` significa «no medido» y hace que `mano_armada` no aplique
    la puerta, mientras que ceros significaria «ninguna accion», que es una afirmacion distinta.
    """
    ev = ev if ev is not None else lee_eventos()[0]
    tramos = [t for l in (-1, 0, 1) for t in ev.get((stem, l, "accion"), [])]
    if not tramos:
        return None
    m = np.zeros(int(T), bool)
    for a, b in tramos:
        m[max(0, a):min(int(T), b + 1)] = True
    return m


def vias_accion():
    """`{clip: via}` de los eventos `accion`. `via_accion` vacio = via con homografia.

    Se separa porque **6 de los 10 clips reservados no tienen homografia** y su `accion` viene por la
    via de altos de caja. Si las dos vias se comportan distinto hay que saberlo ANTES de mezclarlas,
    asi que las cifras se dan separadas.
    """
    import csv
    if not EVENTOS.exists():
        return {}
    with open(EVENTOS, newline="", encoding="utf-8", errors="replace") as fh:
        filas = list(csv.DictReader(fh))
    out = {}
    for r in filas:
        if r.get("tipo") != "accion":
            continue
        v = r.get("via_accion")
        v = "homografia" if v in (None, "", "None") else str(v)
        out.setdefault(r.get("clip", ""), set()).add(v)
    return {k: ("mixta" if len(v) > 1 else next(iter(v))) for k, v in out.items()}


def mascaras_puerta(d, lado, ev=None):
    """Las cinco puertas declaradas en el punto 10, mas dos combinaciones, como mascaras (T,).

    Todo se calcula desde el NPZ, sin tocar el video. Las medianas de `extendido` y `perfil` se
    toman SOLO sobre los cuadros utilizables del clip, no sobre los T cuadros: una mediana que
    incluya cuadros sin pose no es la mediana del tirador.
    """
    KP, SC, B = d["KP"], d["SC"], d["B"]
    W, H, T = int(d["W"]), int(d["H"]), int(d["T"])
    base = utiles(d)
    ut = np.zeros(T, bool)
    ut[base] = True
    pxm = np.stack([(B[:, l, 3] - B[:, l, 1]) * H / CL.ALTURA_M for l in (0, 1)])
    with np.errstate(invalid="ignore"):
        s = np.nansum(np.where(np.isfinite(pxm), pxm, 0.0), axis=0)
        c = np.isfinite(pxm).sum(axis=0)
        esc = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    """Media de los dos tiradores hecha a mano y no con `nanmean`, porque hay cuadros en los que NI
    UNO de los dos tiene caja y `nanmean` de una columna entera de NaN avisa por stderr en cada
    llamada. El valor es el mismo."""
    """px por metro del par. Se promedian los dos tiradores porque la distancia que se mide es ENTRE
    ellos y cada uno tiene su propia caja; con uno solo la distancia cambiaria segun quien la mide."""
    dmin = np.full(T, np.inf)
    for ia in (CL.MANOS["A"]["root"], CL.MANOS["B"]["root"]):
        for ib in (CL.MANOS["A"]["root"], CL.MANOS["B"]["root"]):
            v = KP[:, 0, ia] - KP[:, 1, ib]
            dmin = np.minimum(dmin, np.linalg.norm(v, axis=1))
    d_m = dmin / np.maximum(esc, 1e-9)
    ext = np.maximum(
        np.linalg.norm(KP[:, lado, CL.MANOS["A"]["root"]] - KP[:, lado, CL.MANOS["A"]["hombro"]],
                       axis=1),
        np.linalg.norm(KP[:, lado, CL.MANOS["B"]["root"]] - KP[:, lado, CL.MANOS["B"]["hombro"]],
                       axis=1))
    hom = np.linalg.norm(KP[:, lado, CL.HOMBROS[0]] - KP[:, lado, CL.HOMBROS[1]], axis=1)
    hom_m = hom / np.maximum(pxm[lado], 1e-9)
    hb = CL.bueno(SC[:, lado], list(CL.HOMBROS))
    m = {"sin": np.ones(T, bool),
         "alcance": np.isfinite(d_m) & (d_m <= ALCANCE_M),
         "extendido": ext >= (np.nanmedian(ext[ut]) if ut.any() else np.inf),
         "perfil": hb & (hom_m <= (np.nanmedian(hom_m[ut & hb]) if (ut & hb).any() else -1))}
    ev = ev if ev is not None else lee_eventos()[0]
    stem = str(d["video"]).replace("\\", "/").rsplit("/", 1)[-1][:-4]
    for nom, tipo, lados in (("ev_accion", "accion", (-1,)), ("ev_brazo", "brazo", (lado, -1))):
        z = np.zeros(T, bool)
        hay = False
        for l in lados:
            for a, b in ev.get((stem, l, tipo), []):
                z[max(0, a):min(T, b + 1)] = True
                hay = True
        m[nom] = z if hay else np.zeros(T, bool)
        m[nom + "_existe"] = hay
    xh = 0.5 * (KP[:, lado, CL.HOMBROS[0], 0] + KP[:, lado, CL.HOMBROS[1], 0])
    xn = KP[:, lado, NARIZ, 0]
    xrv = 0.5 * (B[:, 1 - lado, 0] + B[:, 1 - lado, 2]) * W
    nb = CL.bueno(SC[:, lado], [NARIZ]) & hb
    m["encarado"] = nb & np.isfinite(xn) & np.isfinite(xh) & np.isfinite(xrv) & (
        np.sign(xn - xh) == np.sign(xrv - xh))
    m["alcance+extendido"] = m["alcance"] & m["extendido"]
    m["alcance+extendido+perfil"] = m["alcance"] & m["extendido"] & m["perfil"]
    m["alcance+encarado"] = m["alcance"] & m["encarado"]
    m["encarado+ev_accion"] = m["encarado"] & m["ev_accion"]
    m["_d_manos_m_p50"] = float(np.nanmedian(d_m[ut])) if ut.any() else float("nan")
    m["_hombros_m_p50"] = float(np.nanmedian(hom_m[ut & hb])) if (ut & hb).any() else float("nan")
    return m


def voto_con_puertas(d, lado, ev=None):
    """`a_voto` y su `frac_voto` con cada puerta puesta. `None` si no llegan `VOTA_MIN` cuadros."""
    m = mascaras_puerta(d, lado, ev)
    out = {"d_manos_m_p50": m["_d_manos_m_p50"], "hombros_m_p50": m["_hombros_m_p50"]}
    for nom in PUERTAS:
        mv, dv = CL.mano_armada_voto(d["KP"], d["SC"], d["B"], int(d["W"]), int(d["H"]), lado,
                                     d["origen"], d["mask"], None if nom == "sin" else m[nom])
        n = dv.get("n_vota", 0)
        vale = n >= VOTA_MIN
        out["p_" + nom] = dv.get("a_voto") if vale else None
        out["frac_" + nom] = dv.get("frac_a_voto", float("nan")) if vale else float("nan")
        out["n_" + nom] = n
    return out


def criterios_cuerpo(d, lado):
    """Los tres criterios del punto 12, que NO miran al rival. Voto por cuadro, mayoria por clip.

    Devuelve por criterio la mano, la fraccion de votos y cuantos cuadros votaron, mas la
    concordancia por cuadro entre `pie_x` y `nariz_dir`, que son las dos seniales de signo
    independientes.
    """
    KP, SC, mask, org = d["KP"], d["SC"], d["mask"], d["origen"]
    T = int(d["T"])
    base = mask.astype(bool) & (org[:, lado] == 0)
    for M in CL.MANOS.values():
        base &= CL.bueno(SC[:, lado], [M["root"], M["codo"]])
        base &= np.isfinite(KP[:, lado, M["root"]]).all(axis=1)
    K = KP[:, lado]
    """Eje de cada pie: talon -> punto medio de los dos dedos. Se usa el punto medio y no un dedo
    concreto para no depender de cual de 15/16 es el gordo, que T1 no distingue."""
    ejes, okp = {}, base.copy()
    for nom, P in PIES.items():
        med = 0.5 * (K[:, P["dedos"][0]] + K[:, P["dedos"][1]])
        v = med - K[:, P["talon"]]
        n = np.linalg.norm(v, axis=1)
        ejes[nom] = v / np.maximum(n[:, None], 1e-9)
        okp &= CL.bueno(SC[:, lado], [P["talon"], P["dedos"][0], P["dedos"][1]])
        okp &= np.isfinite(v).all(axis=1) & (n > 2.0)
    cad = K[:, PIES["A"]["cadera"]] - K[:, PIES["B"]["cadera"]]
    ncad = np.linalg.norm(cad, axis=1)
    okc = okp & CL.bueno(SC[:, lado], [PIES["A"]["cadera"], PIES["B"]["cadera"]]) & (ncad > 2.0)
    ucad = cad / np.maximum(ncad[:, None], 1e-9)
    hb = CL.bueno(SC[:, lado], list(CL.HOMBROS)) & np.isfinite(
        K[:, CL.HOMBROS[0]]).all(axis=1) & np.isfinite(K[:, CL.HOMBROS[1]]).all(axis=1)
    xh = 0.5 * (K[:, CL.HOMBROS[0], 0] + K[:, CL.HOMBROS[1], 0])
    okn = base & CL.bueno(SC[:, lado], [NARIZ, CL.HOMBROS[0], CL.HOMBROS[1]])
    okn &= np.isfinite(K[:, NARIZ, 0]) & np.isfinite(xh)
    dirn = np.sign(K[:, NARIZ, 0] - xh)
    okn &= dirn != 0
    out, votos = {}, {}
    for nom in CUERPO:
        if nom == "pie_x":
            pa, pb = np.abs(ejes["A"][:, 0]), np.abs(ejes["B"][:, 0])
            ok = okp
            """OJO: el criterio `pie_x` vive en `cono_lib.mano_pie`, que es el ADOPTADO. Aqui se
            recalcula para poder carearlo, y el careo comprueba que las dos implementaciones dan la
            misma mano Y la misma `frac` en los 46 tiradores; si alguna vez divergen, es un fallo."""
        elif nom == "pie_cadera":
            pa = np.abs((ejes["A"] * ucad).sum(axis=1))
            pb = np.abs((ejes["B"] * ucad).sum(axis=1))
            ok = okc
        elif nom == "nariz_dir":
            pa = dirn * K[:, CL.MANOS["A"]["root"], 0]
            pb = dirn * K[:, CL.MANOS["B"]["root"], 0]
            ok = okn
        elif nom == "hombro_pie":
            """Direccion de frente = eje del pie GANADOR en cada cuadro, no del pie que gane el
            clip: asi el criterio se puede evaluar cuadro a cuadro sin depender de su propio voto."""
            gana_pie = np.abs(ejes["A"][:, 0]) > np.abs(ejes["B"][:, 0])
            u = np.where(gana_pie[:, None], ejes["A"], ejes["B"])
            cen = 0.5 * (K[:, CL.HOMBROS[0]] + K[:, CL.HOMBROS[1]])
            pa = (u * (K[:, CL.HOMBROS[1]] - cen)).sum(axis=1)
            pb = (u * (K[:, CL.HOMBROS[0]] - cen)).sum(axis=1)
            ok = okp & hb
            """`pa > pb` equivale a que el hombro 6 (lado A) este delante. El hombro 6 es el del lado
            de la mano A por la cadena verificada en T3 (coincide en el 0,989 de los cuadros)."""
        elif nom == "hombro_nariz":
            pa = dirn * K[:, CL.HOMBROS[1], 0]
            pb = dirn * K[:, CL.HOMBROS[0], 0]
            ok = okn
        else:
            pa = SC[:, lado, CL.HOMBROS[1]].astype("float64")
            pb = SC[:, lado, CL.HOMBROS[0]].astype("float64")
            ok = base & hb
        ok = ok & np.isfinite(pa) & np.isfinite(pb) & (pa != pb)
        gana = np.where(pa > pb, 1, 0)
        votos[nom] = (ok, gana)
        na, nb = int(gana[ok].sum()), int((1 - gana[ok]).sum())
        vale = na + nb >= VOTA_MIN
        out[nom] = "A" if na >= nb else "B"
        out["c_" + nom] = out[nom] if vale else None
        out["frac_" + nom] = max(na, nb) / float(max(na + nb, 1)) if vale else float("nan")
        out["n_" + nom] = na + nb
        del out[nom]
    oa, ga = votos["pie_x"]
    ob, gb = votos["nariz_dir"]
    m = oa & ob
    out["concuerda_pie_nariz_cuadro"] = float(np.mean(ga[m] == gb[m])) if m.any() else float("nan")
    out["n_pie_nariz"] = int(m.sum())
    out["concuerdan_pie_nariz"] = (None if None in (out["c_pie_x"], out["c_nariz_dir"])
                                   else bool(out["c_pie_x"] == out["c_nariz_dir"]))
    """`frac_cruza_brazo`: cuantos cuadros tienen el orden en x de las MANOS al contrario del de los
    HOMBROS. Es la medida del caso que preocupa al usuario, «se pasa el brazo por detras» o
    «invierte la guardia». No es un criterio, es una frecuencia."""
    sm = np.sign(K[:, CL.MANOS["A"]["root"], 0] - K[:, CL.MANOS["B"]["root"], 0])
    sh = np.sign(K[:, CL.HOMBROS[1], 0] - K[:, CL.HOMBROS[0], 0])
    mc = base & hb & (sm != 0) & (sh != 0)
    out["frac_cruza_brazo"] = float(np.mean(sm[mc] != sh[mc])) if mc.any() else float("nan")
    out["n_cruza"] = int(mc.sum())
    """La combinacion declarada en el punto 14 para atacar la cobertura."""
    fp, fn = out["frac_pie_x"], out["frac_hombro_nariz"]
    if out["c_pie_x"] and np.isfinite(fp) and fp >= 0.85:
        out["c_pie_o_hombro"] = out["c_pie_x"]
    elif out["c_hombro_nariz"] and np.isfinite(fn) and fn >= 0.85:
        out["c_pie_o_hombro"] = out["c_hombro_nariz"]
    else:
        out["c_pie_o_hombro"] = None
    return out


def recorta(im, cx, cy, r):
    """Parche cuadrado de semilado `r` centrado en (cx, cy). None si se sale demasiado."""
    H, W = im.shape[:2]
    a, b = int(round(cx - r)), int(round(cx + r)) + 1
    c, e = int(round(cy - r)), int(round(cy + r)) + 1
    total = max(1, (b - a) * (e - c))
    a2, b2, c2, e2 = max(0, a), min(W, b), max(0, c), min(H, e)
    if b2 - a2 < 3 or e2 - c2 < 3:
        return None
    if (b2 - a2) * (e2 - c2) / total < DENTRO_MIN:
        return None
    return im[c2:e2, a2:b2]


def rasgos(sub):
    """`f_blanco`, `a_blob`, `f_punio` y `f_oscuro` de un parche BGR. Blanco de `blancura_pareja`."""
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    bl = (V >= V_CLARO) & (S <= S_MAX)
    f = float(np.mean(bl))
    osc = float(np.mean(V <= V_OSCURO))
    n, lab, stats, _cen = cv2.connectedComponentsWithStats(bl.astype("uint8"), 4)
    h, w = bl.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = R_DISCO * 0.5 * min(h, w)
    disco = ((xx - 0.5 * (w - 1)) ** 2 + (yy - 0.5 * (h - 1)) ** 2) <= rr * rr
    ids = set(int(v) for v in np.unique(lab[disco & bl]) if v > 0)
    a = max([int(stats[i, 4]) for i in ids], default=0) / float(h * w)
    p = float(np.mean((S >= S_PUNIO) & (V >= V_PUNIO)))
    """PUNTO 16, la careta: la mancha OSCURA conexa mas grande que toca el disco central, con
    `oscuro = V < V_CLARO`, que es la inversion de la mitad en V del blanco del proyecto."""
    osq = (V < V_CLARO).astype("uint8")
    n2, lab2, st2, _c2 = cv2.connectedComponentsWithStats(osq, 4)
    ids2 = set(int(v) for v in np.unique(lab2[disco & (osq > 0)]) if v > 0)
    if ids2:
        j = max(ids2, key=lambda i: int(st2[i, 4]))
        ao = int(st2[j, 4]) / float(h * w)
        caja = max(1, int(st2[j, 2]) * int(st2[j, 3]))
        cp = int(st2[j, 4]) / float(caja)
    else:
        ao, cp = 0.0, float("nan")
    return f, a, p, osc, ao, cp


def del_cuadro(im, KP, k, lado):
    """Los cuatro rasgos de las DOS manos del tirador `lado` en el cuadro `k`.

    Devuelve dict por mano ('A','B') o None si algun parche no se puede leer. Se exige que se puedan
    leer las dos manos, porque la decision es la comparacion entre ellas: media comparacion no vale.
    """
    out = {}
    for nom, M in CL.MANOS.items():
        raiz = KP[k, lado, M["root"]].astype("float64")
        codo = KP[k, lado, M["codo"]].astype("float64")
        v = raiz - codo
        L = float(np.linalg.norm(v))
        if not np.isfinite(L) or L < 6.0:
            return None
        u = v / L
        r = R_PARCHE * L
        cm = raiz + DESPL_DISTAL * r * u
        cg = raiz - R_MANGA * r * u
        pm = recorta(im, cm[0], cm[1], r)
        pg = recorta(im, cg[0], cg[1], r)
        if pm is None:
            return None
        f, a, p, osc, ao, cp = rasgos(pm)
        fm = rasgos(pg)[0] if pg is not None else float("nan")
        fila = {"a_osc": ao, "comp": cp,
                "f_blanco": f, "a_blob": a, "f_punio": p, "f_manga": fm, "f_oscuro": osc,
                "r": r, "antebrazo": L, "cx": float(cm[0]), "cy": float(cm[1])}
        Vm = cv2.cvtColor(pm, cv2.COLOR_BGR2HSV)[:, :, 2]
        for uu in UMBRALES_OSC:
            fila["fo_v%d" % uu] = float(np.mean(Vm <= uu))
        for fr, fd in REJILLA:
            rr = fr * L
            pp = recorta(im, *(raiz + fd * rr * u), rr)
            if pp is None:
                fila["fb_%.2f_%.2f" % (fr, fd)] = float("nan")
                fila["fo_%.2f_%.2f" % (fr, fd)] = float("nan")
            else:
                g = rasgos(pp)
                fila["fb_%.2f_%.2f" % (fr, fd)] = g[0]
                fila["fo_%.2f_%.2f" % (fr, fd)] = g[3]
        out[nom] = fila
    ra = KP[k, lado, CL.MANOS["A"]["root"]].astype("float64")
    rb = KP[k, lado, CL.MANOS["B"]["root"]].astype("float64")
    out["sep"] = float(np.linalg.norm(ra - rb))
    out["r_medio"] = 0.5 * (out["A"]["r"] + out["B"]["r"])
    out["vale"] = bool(out["sep"] >= SOLAPE_MIN * out["r_medio"])
    return out


def mide_clip(stem, ncuadros=NCUADROS):
    """Rasgos del guante en `ncuadros` cuadros repartidos, mas el voto por clip y por tirador."""
    d = carga(stem)
    W, H = int(d["W"]), int(d["H"])
    vid = str(d["video"])
    ks = utiles(d)
    if len(ks) < 10:
        return {"stem": stem, "error": "cuadros utiles %d" % len(ks)}
    sel = ks[np.unique(np.linspace(0, len(ks) - 1, min(ncuadros, len(ks))).astype(int))]
    cap = cv2.VideoCapture(vid)
    filas = []
    for k in sel:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, im = cap.read()
        if not ok:
            continue
        for lado in (0, 1):
            r = del_cuadro(im, d["KP"], int(k), lado)
            if r is None:
                continue
            filas.append({"clip": stem, "res": "%dx%d" % (W, H), "k": int(k), "lado": lado,
                          "A": r["A"], "B": r["B"], "sep": r["sep"],
                          "r_medio": r["r_medio"], "vale": r["vale"]})
    cap.release()
    return {"stem": stem, "res": "%dx%d" % (W, H), "W": W, "H": H, "n_utiles": int(len(ks)),
            "filas": filas, "votos": {lado: voto(filas, lado) for lado in (0, 1)},
            "votos_osc": {lado: voto(filas, lado, "f_oscuro") for lado in (0, 1)}}


def voto(filas, lado, campo="f_blanco"):
    """Voto mayoritario por cuadro sobre `campo`, con las dos puertas declaradas.

    Puerta de solape (4b): fuera los cuadros con `sep < SOLAPE_MIN * r`, donde los dos parches leen
    los mismos pixeles. Puerta de margen (4): abstencion si `|d| < MARGEN_CUADRO`.

    `campo` es `f_blanco` para el criterio del punto 3 y `f_oscuro` para el del punto 9. En los dos
    casos gana la mano con el valor MAYOR, que es lo declarado en cada punto.
    """
    fs = [f for f in filas if f["lado"] == lado]
    n_tot = len(fs)
    fs = [f for f in fs if f["vale"]]
    ds = np.array([f["A"][campo] - f["B"][campo] for f in fs], dtype="float64")
    base = {"n_cuadros": n_tot, "n_separados": len(fs),
            "frac_solape": 1.0 - len(fs) / float(max(n_tot, 1))}
    v = ds[np.abs(ds) >= MARGEN_CUADRO]
    na, nb = int((v > 0).sum()), int((v < 0).sum())
    if na + nb == 0:
        return {"mano": None, "frac_voto": float("nan"), "n_votantes": 0, "solido": False,
                "d_mediana": float(np.median(ds)) if len(ds) else float("nan"), **base}
    mano = "A" if na >= nb else "B"
    frac = max(na, nb) / float(na + nb)
    return {"mano": mano, "frac_voto": frac, "n_votantes": na + nb, "n_A": na, "n_B": nb,
            "d_mediana": float(np.median(ds)),
            "abstiene": 1.0 - (na + nb) / float(max(len(ds), 1)),
            "solido": bool(frac >= VOTO_SOLIDO and na + nb >= VOTANTES_MIN), **base}


USA_GUANTE = False
"""RETIRADO CON SU MEDIDA AL LADO, al estilo de `USA_CRUDAS = False` en `extrae_pareja.py`.

La regla declarada en el punto 8 era «el guante manda si su voto es SOLIDO». Se midio y sale PEOR
que no hacer nada, sobre 27 tiradores anotados a ojo:

    criterio (a) solo, la linea base       0,741
    (a) + guante con la regla del punto 8  0,556   <- la combinacion QUITA 5 aciertos
    guante solo                            0,444
    guante solo, cuando su voto es SOLIDO  0,167  (1 de 6)  <- cuando mas se cree, peor

El caso que lo explica esta en la figura `discrepa_UglyNiceAplomadofalcon_l0.png`: la mano DESARMADA
cae sobre la chaqueta blanca del hombro y da `f_blanco` 0,22-0,45, mientras la armada, que esta sobre
el suelo azul de la pista, da 0,02-0,03. El rasgo no mide guante: mide cuanto fondo blanco hay detras
de la mano. Con `USA_GUANTE = False`, `combina` devuelve siempre el criterio (a) y el guante se
publica solo como corroboracion informativa."""


def combina(geo, gua):
    """Elige la mano final. Con `USA_GUANTE = False` es siempre el criterio (a), y se dice por que.

    `geo` es la mano del criterio (a) de `cono_lib.mano_armada`; `gua` es el dict de `voto`.
    """
    conc = None if gua is None or gua.get("mano") is None else bool(gua["mano"] == geo)
    if not USA_GUANTE:
        return {"mano_final": geo, "por": "geometria_a_GUANTE_RETIRADO", "concuerdan": conc}
    if gua is not None and gua.get("solido"):
        return {"mano_final": gua["mano"], "por": "guante", "concuerdan": conc}
    return {"mano_final": geo, "por": "geometria_a_guante_flojo", "concuerdan": conc}


def tabla(stems=None, ncuadros=NCUADROS, salida=True):
    """Mide todos los clips y deja `mano_guante.csv` y `mano_guante.json` en `pruebas/`."""
    import csv
    stems = stems or clips()
    ev, ev_mt = lee_eventos()
    todo, res = [], []
    for stem in stems:
        r = mide_clip(stem, ncuadros)
        if "error" in r:
            print("  %-40s SIN DATO: %s" % (stem, r["error"]))
            continue
        d = carga(stem)
        for lado in (0, 1):
            adop, det = CL.mano_armada(d["KP"], d["SC"], d["B"], int(d["W"]), int(d["H"]), lado,
                                       d["origen"], d["mask"],
                                       accion=mascara_accion(stem, int(d["T"]), ev))
            geo = det.get("mano_a", adop)
            """`geo` tiene que ser la respuesta del criterio (a) A SECAS, que es la LINEA BASE. Desde
            que `mano_armada` devuelve lo adoptado, tomar su primer valor de retorno metia el
            criterio nuevo en la columna de la linea base y la subia de 0,741 a 0,815 sola. Queda
            escrito porque es el tipo de contaminacion que no avisa."""
            g, go = r["votos"][lado], r["votos_osc"][lado]
            _mv, dv = CL.mano_armada_voto(d["KP"], d["SC"], d["B"], int(d["W"]), int(d["H"]),
                                          lado, d["origen"], d["mask"])
            c = combina(geo, g)
            res.append({"clip": stem, "res": r["res"], "lado": lado, "geo_a": geo,
                        "adoptada": adop, "dudosa": det.get("dudosa"),
                        "firme_pie": det.get("firme_pie"), "por_adoptada": det.get("por"),
                        "a_voto": dv.get("a_voto"), "frac_a_voto": dv.get("frac_a_voto"),
                        "a_dist2d": dv.get("a_dist2d"), "frac_a_dist2d": dv.get("frac_a_dist2d"),
                        "n_vota": dv.get("n_vota"),
                        "oscuro": go["mano"], "frac_voto_osc": go["frac_voto"],
                        "n_votantes_osc": go["n_votantes"], "d_mediana_osc": go["d_mediana"],
                        "solido_osc": go["solido"],
                        "geo_b2": det.get("apuntando"), "geo_ext": det.get("extendida_RECHAZADO"),
                        "margen_x_px": det.get("margen_x_px"), "guante": g["mano"],
                        "frac_voto": g["frac_voto"], "n_votantes": g["n_votantes"],
                        "n_cuadros": g["n_cuadros"], "n_separados": g["n_separados"],
                        "frac_solape": g["frac_solape"], "d_mediana": g["d_mediana"],
                        "solido": g["solido"], "mano_final": c["mano_final"], "por": c["por"],
                        "concuerdan": c["concuerdan"], **voto_con_puertas(d, lado, ev),
                        **criterios_cuerpo(d, lado)})
            print("  %-40s l%d  geo=%s blanco=%s frac=%.2f d=%+.3f %-6s | oscuro=%s frac=%.2f "
                  "d=%+.3f" % (
                      stem, lado, geo, g["mano"], g["frac_voto"] or float("nan"), g["d_mediana"],
                      "SOLIDO" if g["solido"] else "flojo", go["mano"],
                      go["frac_voto"] or float("nan"), go["d_mediana"]))
        for f in r["filas"]:
            for nom in ("A", "B"):
                todo.append({"clip": f["clip"], "res": f["res"], "k": f["k"], "lado": f["lado"],
                             "mano": nom, "sep": f["sep"], "vale": f["vale"],
                             **{kk: vv for kk, vv in f[nom].items()}})
    if salida:
        with open(PRUEBAS / "mano_guante_cuadros.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(todo[0].keys()))
            w.writeheader()
            w.writerows(todo)
        with open(PRUEBAS / "mano_guante.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(res[0].keys()))
            w.writeheader()
            w.writerows(res)
        with open(PRUEBAS / "mano_guante.json", "w", encoding="utf-8") as fh:
            json.dump({"R_PARCHE": R_PARCHE, "DESPL_DISTAL": DESPL_DISTAL,
                       "MARGEN_CUADRO": MARGEN_CUADRO, "NCUADROS": ncuadros,
                       "VOTO_SOLIDO": VOTO_SOLIDO, "VOTANTES_MIN": VOTANTES_MIN,
                       "V_CLARO": V_CLARO, "S_MAX": S_MAX, "USA_GUANTE": USA_GUANTE,
                       "ALCANCE_M": ALCANCE_M, "VOTA_MIN": VOTA_MIN,
                       "eventos_csv_mtime_ns": ev_mt, "clips": res}, fh, indent=1)
        print("\n  %s" % (PRUEBAS / "mano_guante.csv"))
    return res, todo


N_LARGO, N_BOOT = 120, 400
TAMANOS = (10, 20, 30, 50, 100)


def votos_pie_cuadro(d, lado):
    """El voto del criterio del pie CUADRO A CUADRO, como array booleano de los cuadros que votan.

    Mismas puertas que `cono_lib.mano_pie`, para que la `frac` del clip completo coincida.
    """
    KP, SC = d["KP"], d["SC"]
    ok = d["mask"].astype(bool) & (d["origen"][:, lado] == 0)
    for M in CL.MANOS.values():
        ok &= CL.bueno(SC[:, lado], [M["root"], M["codo"]])
        ok &= np.isfinite(KP[:, lado, M["root"]]).all(axis=1)
    al = {}
    for nom, P in CL.PIES.items():
        med = 0.5 * (KP[:, lado, P["dedos"][0]] + KP[:, lado, P["dedos"][1]])
        v = med - KP[:, lado, P["talon"]]
        n = np.linalg.norm(v, axis=1)
        al[nom] = np.abs(v[:, 0]) / np.maximum(n, 1e-9)
        ok &= CL.bueno(SC[:, lado], [P["talon"], P["dedos"][0], P["dedos"][1]])
        ok &= np.isfinite(v).all(axis=1) & (n > 2.0)
    ok &= np.isfinite(al["A"]) & np.isfinite(al["B"]) & (al["A"] != al["B"])
    return (al["A"][ok] > al["B"][ok])


def estabilidad_frac():
    """PUNTO 17: cuantos cuadros hacen falta para que `frac >= 0,85` signifique algo.

    Remuestrea los votos por cuadro de los tiradores largos y mide la falsa firmeza por tamanio.
    """
    rng = np.random.default_rng(20260911)
    largos = []
    for stem in clips():
        d = carga(stem)
        for lado in (0, 1):
            v = votos_pie_cuadro(d, lado)
            if len(v) >= N_LARGO:
                largos.append((stem, lado, v, max(v.mean(), 1 - v.mean())))
    print("tiradores con >= %d cuadros que votan: %d de %d" % (
        N_LARGO, len(largos), 2 * len(clips())))
    ff = [x for x in largos if x[3] >= VOTO_SOLIDO]
    nf = [x for x in largos if x[3] < VOTO_SOLIDO]
    print("   de ellos, firmes en el clip completo: %d ; no firmes: %d" % (len(ff), len(nf)))
    print("   %-6s %-9s %-14s %-16s" % ("n", "sd(frac)", "falsa_firmeza", "firmeza_perdida"))
    for n in TAMANOS:
        sds, fal, per = [], [], []
        for _stem, _l, v, fr in largos:
            idx = rng.integers(0, len(v), size=(N_BOOT, n))
            m = v[idx].mean(axis=1)
            f = np.maximum(m, 1 - m)
            sds.append(float(np.std(f)))
            if fr < VOTO_SOLIDO:
                fal.append(float(np.mean(f >= VOTO_SOLIDO)))
            else:
                per.append(float(np.mean(f < VOTO_SOLIDO)))
        print("   %-6d %-9.3f %-14.3f %-16.3f" % (
            n, np.mean(sds), np.mean(fal) if fal else float("nan"),
            np.mean(per) if per else float("nan")))
    print("   (el corte declarado es falsa_firmeza <= 0,10; y el remuestreo con reposicion sobre un")
    print("    mismo clip SUBESTIMA la variabilidad, asi que esto es una cota por lo BAJO)")
    corto = [(s, l, len(v), fr) for s, l, v, fr in
             [(s, l, votos_pie_cuadro(carga(s), l), 0) for s in clips() for l in (0, 1)]
             if len(v) < N_LARGO]
    print("\n   tiradores CORTOS (< %d cuadros que votan), que son los que el aviso afecta:" % N_LARGO)
    for s, l, n, _f in sorted(corto, key=lambda x: x[2])[:8]:
        print("      %-42s l%d  %3d cuadros" % (s, l, n))


SEG_MIN_ARB = 4.0


def tasa_esgrima():
    """PUNTO 19: eventos de esgrima por segundo util de cada lado, y el criterio del arbitro.

    Criterio DECLARADO antes de medir: cero eventos `fondo` y cero `brazo` en mas de `SEG_MIN_ARB`
    segundos utiles -> no es un tirador. Se publica la separacion y los FALSOS POSITIVOS.
    """
    ev, mt = lee_eventos()
    V = {(r["clip"], int(r["lado"])): r for r in lee_csv(MUESTRAS / "verdad_mano.csv")}
    fps = {}
    filas = []
    for stem in clips():
        d = carga(stem)
        f = float(d["fps"]) if "fps" in d else 30.01
        """30,01 es el fps que traen los NPZ del corpus (`posicion_y_eventos.csv`), y se usa de
        respaldo cuando el NPZ no guarda `fps`. Los segundos utiles se calculan con el, asi que el
        umbral de 4 s depende de esto y queda dicho."""
        for lado in (0, 1):
            n = 0
            hay = False
            for tipo in ("fondo", "brazo"):
                for l in (lado, -1):
                    if (stem, l, tipo) in ev:
                        hay = True
                        n += len(ev[(stem, l, tipo)])
            if not hay and not any((stem, l, t) in ev for l in (0, 1, -1)
                                   for t in ("fondo", "brazo", "marcha", "romper", "accion")):
                continue
            seg = float((d["mask"].astype(bool) & (d["origen"][:, lado] == 0)).sum()) / max(f, 1)
            v = V.get((stem, lado), {})
            filas.append({"clip": stem, "lado": lado, "n_ev": n, "seg": seg,
                          "tasa": n / max(seg, 1e-9), "cue": v.get("cue", ""),
                          "mano": v.get("mano", "")})
    fps.clear()
    arb = [f for f in filas if f["cue"] == "arbitro"]
    tir = [f for f in filas if f["mano"] in ("A", "B")]
    print("PUNTO 19, eventos de esgrima (`fondo` + `brazo`) por segundo util. eventos.csv mtime_ns "
          "%d" % mt)
    print("   lados con eventos consultables: %d   de ellos arbitros anotados: %d, tiradores: %d"
          % (len(filas), len(arb), len(tir)))
    if not filas:
        print("   sin eventos que consultar: nada que medir")
        return
    if len(arb) < 2:
        print("   *** PARCIAL: solo hay %d de los 2 arbitros con eventos. La medida definitiva se"
              " hace cuando eventos.csv cubra los 10 clips reservados. ***" % len(arb))
    for nom, g in (("ARBITROS", arb), ("TIRADORES", tir)):
        if not g:
            continue
        t = np.array([x["tasa"] for x in g])
        print("   %-10s n=%2d  tasa p10 %.3f p50 %.3f p90 %.3f  max %.3f  con tasa==0: %d" % (
            nom, len(g), *np.percentile(t, [10, 50, 90]), t.max(), int((t == 0).sum())))
    marca = [f for f in filas if f["n_ev"] == 0 and f["seg"] > SEG_MIN_ARB]
    fp = [f for f in marca if f["mano"] in ("A", "B")]
    vp = [f for f in marca if f["cue"] == "arbitro"]
    print("\n   CRITERIO `cero fondo y cero brazo en > %.1f s utiles`:" % SEG_MIN_ARB)
    print("      marca %d lados de %d" % (len(marca), len(filas)))
    print("      verdaderos positivos (arbitros cazados): %d de %d" % (len(vp), len(arb)))
    print("      FALSOS POSITIVOS (tiradores marcados)  : %d de %d" % (len(fp), len(tir)))
    for f in marca:
        print("         %-42s l%d  n_ev=%d  seg=%.1f  %s" % (
            f["clip"], f["lado"], f["n_ev"], f["seg"],
            "ARBITRO" if f["cue"] == "arbitro" else ("tirador %s" % f["mano"] or "?")))
    if tir:
        t = np.array([x["tasa"] for x in tir])
        print("\n   tiradores con la tasa mas baja, que son el confundido a batir:")
        for f in sorted(tir, key=lambda x: x["tasa"])[:5]:
            print("      %-42s l%d  tasa %.3f  n_ev=%d  seg=%.1f" % (
                f["clip"], f["lado"], f["tasa"], f["n_ev"], f["seg"]))


def regla_18(r):
    """LA REGLA DEL PUNTO 18, tal como se declaro antes de ver el reservado. Cadena vacia = sin
    decision. El orden es parte de la regla: primero el pie firme, luego la puerta."""
    if r["dudosa"] == "False":
        return r["c_pie_x"]
    p = r.get("p_encarado+ev_accion", "")
    return p if p in ("A", "B") else ""


def mayoria(vs):
    """Mayoria simple entre respuestas 'A'/'B'. Cadena vacia si no hay al menos dos validas."""
    vs = [v for v in vs if v in ("A", "B")]
    if len(vs) < 2:
        return ""
    return "A" if vs.count("A") > vs.count("B") else "B"


def lee_csv(p):
    import csv
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def careo():
    """EL CAREO: acierto de cada criterio contra la anotacion a ojo, con la linea base al lado.

    Lee `02_validacion/muestras/mano_dominante/verdad_mano.csv` (anotacion a ojo, hecha por el
    agente y no por un arbitro) y los dos CSV que deja `tabla`. No calcula nada nuevo: solo cruza.

    LAS LINEAS BASE GRATIS que se imprimen, porque sin ellas un 70 % no significa nada:
      `a`          el criterio actual de `cono_lib.mano_armada`, que es «la mas cerca del rival»
      `b2`         «la que mas apunta al rival», ya rechazado en su dia
      `extendida`  «la mas extendida», ya rechazado en su dia
      `siempre_A`  contestar siempre A. `siempre_B` igual. Son el azar informado.
      `espejo`     A si el tirador es el de la izquierda y B si es el de la derecha
    """
    V = {(r["clip"], int(r["lado"])): r for r in lee_csv(MUESTRAS / "verdad_mano.csv")}
    G = {(r["clip"], int(r["lado"])): r for r in lee_csv(PRUEBAS / "mano_guante.csv")}
    F = lee_csv(PRUEBAS / "mano_guante_cuadros.csv")
    claves = [k for k in sorted(G) if k in V and V[k]["mano"] in ("A", "B")]
    falta = [k for k in sorted(V) if V[k]["mano"] in ("A", "B") and k not in G]
    print("tiradores anotados A/B: %d de %d filas de verdad" % (
        len([k for k in V if V[k]["mano"] in ("A", "B")]), len(V)))
    print("tiradores con verdad Y medida del guante: %d" % len(claves))
    if falta:
        print("con verdad pero SIN medida: %s" % ", ".join("%s l%d" % k for k in falta))

    def acierto(claves, pred, nom):
        ok = [pred(k) == V[k]["mano"] for k in claves if pred(k) in ("A", "B")]
        n = len(ok)
        return nom, (sum(ok) / n if n else float("nan")), n

    crit = [("a", lambda k: G[k]["geo_a"]),
            ("a_voto", lambda k: G[k]["a_voto"]),
            ("a_dist2d", lambda k: G[k]["a_dist2d"]),
            ("PIE_x", lambda k: G[k]["c_pie_x"]),
            ("PIE_cadera", lambda k: G[k]["c_pie_cadera"]),
            ("NARIZ_dir", lambda k: G[k]["c_nariz_dir"]),
            ("HOMBRO_pie", lambda k: G[k]["c_hombro_pie"]),
            ("HOMBRO_nariz", lambda k: G[k]["c_hombro_nariz"]),
            ("HOMBRO_conf", lambda k: G[k]["c_hombro_conf"]),
            ("pie_o_hombro", lambda k: G[k]["c_pie_o_hombro"]),
            ("ADOPTADO_no_dudosa", lambda k: (G[k]["adoptada"]
                                              if G[k]["dudosa"] == "False" else "")),
            ("ADOPTADO_todos", lambda k: G[k]["adoptada"]),
            ("REGLA_18", lambda k: regla_18(G[k])),
            ("cuerpo_2de2", lambda k: (G[k]["c_pie_x"]
                                       if G[k]["c_pie_x"] == G[k]["c_nariz_dir"] else "")),
            ("mayoria_3", lambda k: mayoria([G[k]["a_voto"], G[k]["c_pie_x"],
                                             G[k]["c_nariz_dir"]])),
            ("b2", lambda k: G[k]["geo_b2"]),
            ("extendida", lambda k: G[k]["geo_ext"]),
            ("guante", lambda k: G[k]["guante"]),
            ("guante_solido", lambda k: G[k]["guante"] if G[k]["solido"] == "True" else ""),
            ("oscuro", lambda k: G[k]["oscuro"]),
            ("oscuro_solido", lambda k: G[k]["oscuro"] if G[k]["solido_osc"] == "True" else ""),
            ("final_combinado", lambda k: G[k]["mano_final"]),
            ("siempre_A", lambda k: "A"), ("siempre_B", lambda k: "B"),
            ("espejo", lambda k: "A" if k[1] == 0 else "B")]
    ent = [k for k in claves if k[0] not in RESERVADO]
    res = [k for k in claves if k[0] in RESERVADO]
    G_ENT, G_RES = list(ent), list(res)
    """Copias con nombre propio, hechas AQUI y no donde se usan. Mas abajo hay un bloque que
    reasigna `res` a una lista de resoluciones, asi que cualquier uso posterior de `res` lee otra
    cosa. Lo cace con un `KeyError: '1280x720'` tres bloques mas abajo del sitio donde se escribio:
    es el mismo tipo de fallo silencioso que se persigue todo el dia, y por eso queda escrito."""
    grupos = [("ENTRENAMIENTO, los 23 clips con los que se eligieron las reglas", ent),
              ("RESERVADO, semilla 20260911, no vio ninguna regla", res),
              ("cue fuerte (hoja/arma), entrenamiento",
               [k for k in ent if V[k]["cue"] in ("hoja", "arma")]),
              ("certeza alta, entrenamiento", [k for k in ent if V[k]["certeza"] == "alta"]),
              ("720p, entrenamiento", [k for k in ent if G[k]["res"] == "1280x720"]),
              ("1080p, entrenamiento", [k for k in ent if G[k]["res"] == "1920x1080"])]
    """NO hay grupo agregado a proposito, punto 18: las dos cifras se publican por separado y su
    diferencia es la medida del engano post-hoc. Agregarlas la borraria."""
    for nom, cs in grupos:
        print("\n== %s  (n=%d) ==" % (nom, len(cs)))
        for c, f in crit:
            _n, a, k = acierto(cs, f, c)
            print("   %-16s acierto %s  sobre %d" % (
                c, "%.3f" % a if k else "  -  ", k))

    print("\n== EL MISMO CAREO CONTRA LA VERDAD DE LA 1a PASADA, para que se vea cuanto movio la 2a")
    c1 = [k for k in sorted(G) if k in V and V[k]["mano_p1"] in ("A", "B")]
    for c, f in crit[:6]:
        ok = [f(k) == V[k]["mano_p1"] for k in c1 if f(k) in ("A", "B")]
        print("   %-16s acierto %s  sobre %d" % (
            c, "%.3f" % (sum(ok) / len(ok)) if ok else "  -  ", len(ok)))

    print("\n== SEPARACION DEL RASGO, por cuadro, armada contra la otra ==")
    ver = {}
    for r in F:
        k = (r["clip"], int(r["lado"]))
        if k in V and V[k]["mano"] in ("A", "B") and r["vale"] == "True":
            ver.setdefault((k, r["k"]), {})[r["mano"]] = r
    for campo in ("f_blanco", "f_oscuro", "a_osc", "comp", "a_blob", "f_manga", "f_punio"):
        arm, otr, dif, res = [], [], [], []
        for (k, _kk), par in ver.items():
            if len(par) != 2:
                continue
            a = par[V[k]["mano"]]
            o = par["B" if V[k]["mano"] == "A" else "A"]
            va, vo = float(a[campo]), float(o[campo])
            if not (np.isfinite(va) and np.isfinite(vo)):
                continue
            arm.append(va)
            otr.append(vo)
            dif.append(va - vo)
            res.append(a["res"])
        arm, otr, dif = np.array(arm), np.array(otr), np.array(dif)
        print("   %-9s armada p25/p50/p75 %.3f/%.3f/%.3f | otra %.3f/%.3f/%.3f | dif p50 %+.3f"
              "  signo correcto %.3f  (n=%d cuadros)" % (
                  campo, *np.percentile(arm, [25, 50, 75]), *np.percentile(otr, [25, 50, 75]),
                  np.median(dif), float(np.mean(dif > 0)), len(dif)))
        if campo == "f_blanco":
            r720 = np.array([x == "1280x720" for x in res])
            for etq, m in (("720p", r720), ("1080p", ~r720)):
                if m.sum():
                    print("      %-6s dif p50 %+.3f  signo correcto %.3f  (n=%d)" % (
                        etq, np.median(dif[m]), float(np.mean(dif[m] > 0)), int(m.sum())))
    print("\n== BARRIDO DE GEOMETRIA DEL PARCHE (DIAGNOSTICO, no es un criterio) ==")
    print("   el mejor de 7 geometrias elegido DESPUES de ver no es una medida de acierto:")
    print("   %-14s %-8s %-8s %-8s %-8s" % ("(r, despl)", "blan.p50", "blan.sgn", "osc.p50",
                                            "osc.sgn"))
    for fr, fd in REJILLA:
        out = []
        for pre in ("fb", "fo"):
            col = "%s_%.2f_%.2f" % (pre, fr, fd)
            dif = []
            for (k, _kk), par in ver.items():
                if len(par) != 2 or col not in par["A"]:
                    continue
                a = par[V[k]["mano"]]
                o = par["B" if V[k]["mano"] == "A" else "A"]
                va, vo = float(a[col]), float(o[col])
                if np.isfinite(va) and np.isfinite(vo):
                    dif.append(va - vo)
            dif = np.array(dif)
            out += [np.median(dif) if len(dif) else float("nan"),
                    float(np.mean(dif > 0)) if len(dif) else float("nan")]
        print("   (%.2f, %.2f)   %+8.3f %8.3f %+8.3f %8.3f" % (fr, fd, *out))

    print("\n== BARRIDO POST-HOC DEL UMBRAL DE OSCURO (el declarado era V <= 60) ==")
    for uu in UMBRALES_OSC:
        col, dif = "fo_v%d" % uu, []
        for (k, _kk), par in ver.items():
            if len(par) != 2 or col not in par["A"]:
                continue
            a = par[V[k]["mano"]]
            o = par["B" if V[k]["mano"] == "A" else "A"]
            va, vo = float(a[col]), float(o[col])
            if np.isfinite(va) and np.isfinite(vo):
                dif.append(va - vo)
        dif = np.array(dif)
        print("   V <= %3d   dif p50 %+.3f  signo correcto %.3f  (n=%d)" % (
            uu, np.median(dif) if len(dif) else float("nan"),
            float(np.mean(dif > 0)) if len(dif) else float("nan"), len(dif)))

    fb = np.array([float(r["f_blanco"]) for r in F if r["vale"] == "True"])
    fm = np.array([float(r["f_manga"]) for r in F if r["vale"] == "True"])
    m = np.isfinite(fb) & np.isfinite(fm)
    print("\n   correlacion f_blanco vs f_manga: pearson %.3f sobre %d parches"
          "   <- si es alta, el parche esta leyendo MANGA" % (
              float(np.corrcoef(fb[m], fm[m])[0, 1]), int(m.sum())))
    print("   parches con f_punio >= %.2f (punio de color): %.3f" % (
        0.15, float(np.mean([float(r["f_punio"]) >= 0.15 for r in F]))))
    print("   parches con f_blanco = 0 exacto: %.3f" % float(np.mean(fb == 0)))
    print("\n== SOLAPE Y FIRMEZA ==")
    fs = np.array([float(G[k]["frac_solape"]) for k in sorted(G)])
    print("   frac de cuadros descartados por solape: p50 %.3f p90 %.3f max %.3f" % (
        *np.percentile(fs, [50, 90]), fs.max()))
    fv = np.array([float(G[k]["frac_voto"]) for k in sorted(G)
                   if G[k]["frac_voto"] not in ("", "nan")])
    print("   frac_voto del guante: p10 %.2f p50 %.2f p90 %.2f | solidos %d de %d" % (
        *np.percentile(fv, [10, 50, 90]), sum(G[k]["solido"] == "True" for k in G), len(G)))
    for col in ("frac_a_voto", "frac_a_dist2d"):
        fa = np.array([float(G[k][col]) for k in sorted(G) if G[k][col] not in ("", "None")])
        print("   %-14s p10 %.2f p50 %.2f p90 %.2f  | >= 0.80 en %d de %d tiradores" % (
            col, *np.percentile(fa, [10, 50, 90]), int((fa >= 0.80).sum()), len(fa)))
    print("\n== LA REGLA DEL PUNTO 18 EN EL RESERVADO, separada por VIA de `accion` ==")
    via = vias_accion()
    for nom, cs in (("ENTRENAMIENTO", G_ENT), ("RESERVADO", G_RES)):
        for v in ("homografia", "alto_de_caja", "mixta", "sin eventos"):
            g = [k for k in cs if via.get(k[0], "sin eventos") == v]
            if not g:
                continue
            dec = [k for k in g if regla_18(G[k]) in ("A", "B")]
            ok = [regla_18(G[k]) == V[k]["mano"] for k in dec]
            pf = [k for k in g if G[k]["firme_pie"] == "True"]
            okp = [G[k]["c_pie_x"] == V[k]["mano"] for k in pf]
            """LA COLUMNA QUE SE LEE ES `firme_pie` Y NO `dudosa`, Y ERA UN ROTULO MENTIROSO. Esta
            fila decia «pie firme decide N» leyendo `dudosa == False`, que desde que se adopto
            `USA_ACCION_SI_PIE_FLOJO` es la cobertura de **la cascada entera**, no la del pie: en el
            reservado imprimia 19 donde el pie firme decide 12. Esa fila es una de las fuentes de la
            cifra publicada que atribuia a la regla del pie el 1,000 con cobertura total."""
            cas = [k for k in g if G[k]["dudosa"] == "False"]
            okc = [G[k]["adoptada"] == V[k]["mano"] for k in cas]
            print("   %-14s via %-13s n=%2d | REGLA_18 decide %2d acierto %s | pie FIRME decide %2d"
                  " acierto %s | CASCADA decide %2d acierto %s" % (
                      nom, v, len(g), len(dec),
                      "%.3f" % (sum(ok) / len(ok)) if ok else "  -  ", len(pf),
                      "%.3f" % (sum(okp) / len(okp)) if okp else "  -  ", len(cas),
                      "%.3f" % (sum(okc) / len(okc)) if okc else "  -  "))
    print("   (la via `alto_de_caja` es la de los clips SIN homografia; sus `marcha`, `romper`,")
    print("    `fondo` y `alto` NO EXISTEN, no son cero, y eso no se trata como cero en ningun sitio)")

    print("\n== LOS CRITERIOS DEL CUERPO (punto 12): cobertura, concordancia y quien gana ==")
    for nom in CUERPO:
        cs = [k for k in claves if G[k]["c_" + nom] in ("A", "B")]
        ok = [G[k]["c_" + nom] == V[k]["mano"] for k in cs]
        fr = np.nanmedian([float(G[k]["frac_" + nom]) for k in cs]) if cs else float("nan")
        nn = np.median([float(G[k]["n_" + nom]) for k in claves])
        print("   %-12s acierto %s sobre %2d  sin decidir %2d  frac_voto p50 %.2f  cuadros p50 %.0f"
              % (nom, "%.3f" % (sum(ok) / len(ok)) if ok else "  -  ", len(cs),
                 len(claves) - len(cs), fr, nn))
    cc = np.array([float(G[k]["concuerda_pie_nariz_cuadro"]) for k in claves])
    print("   concordancia pie_x vs nariz_dir POR CUADRO: p10 %.2f p50 %.2f p90 %.2f" % tuple(
        np.nanpercentile(cc, [10, 50, 90])))
    ac = [k for k in claves if G[k]["concuerdan_pie_nariz"] == "True"]
    di = [k for k in claves if G[k]["concuerdan_pie_nariz"] == "False"]
    for nom, cs in (("CONCUERDAN por clip", ac), ("DISCREPAN por clip", di)):
        ok = [G[k]["c_pie_x"] == V[k]["mano"] for k in cs]
        ok2 = [G[k]["c_nariz_dir"] == V[k]["mano"] for k in cs]
        print("   %-20s %2d tiradores  acierto pie %s  acierto nariz %s" % (
            nom, len(cs), "%.3f" % (sum(ok) / len(ok)) if ok else "  -  ",
            "%.3f" % (sum(ok2) / len(ok2)) if ok2 else "  -  "))
    for k in di:
        print("      DISCREPAN: %-40s l%d  pie=%s nariz=%s  A OJO=%s (%s)  a=%s" % (
            k[0], k[1], G[k]["c_pie_x"], G[k]["c_nariz_dir"], V[k]["mano"], V[k]["cue"],
            G[k]["geo_a"]))
    print("\n   CRUZADOS CON SU PROPIA FIRMEZA, con el corte 0,85 que YA estaba declarado para")
    print("   `frac_a_voto` y que aqui se REUTILIZA sin volver a elegirlo:")
    for nom in ("a_voto", "pie_x", "nariz_dir", "hombro_pie", "hombro_nariz", "hombro_conf"):
        col = "frac_a_voto" if nom == "a_voto" else "frac_" + nom
        pre = "a_voto" if nom == "a_voto" else "c_" + nom
        for lo, hi, et in ((0.85, 1.01, ">= 0,85"), (0.0, 0.85, " < 0,85")):
            cs = [k for k in claves if G[k][col] not in ("", "nan")
                  and lo <= float(G[k][col]) < hi and G[k][pre] in ("A", "B")]
            ok = [G[k][pre] == V[k]["mano"] for k in cs]
            print("      %-10s frac %s  acierto %s sobre %2d" % (
                nom, et, "%.3f" % (sum(ok) / len(ok)) if ok else "  -  ", len(cs)))

    print("\n   `frac_cruza_brazo`: cuadros con el orden de las MANOS al contrario que el de los")
    print("   HOMBROS, o sea la medida de «se pasa el brazo por detras / invierte la guardia»:")
    fc = np.array([float(G[k]["frac_cruza_brazo"]) for k in claves])
    print("      sobre los 27 tiradores: p10 %.2f p50 %.2f p90 %.2f max %.2f" % (
        *np.nanpercentile(fc, [10, 50, 90]), np.nanmax(fc)))
    bienc = [float(G[k]["frac_cruza_brazo"]) for k in claves if G[k]["geo_a"] == V[k]["mano"]]
    malc = [float(G[k]["frac_cruza_brazo"]) for k in claves if G[k]["geo_a"] != V[k]["mano"]]
    print("      en los %d que (a) acierta: p50 %.2f  |  en los %d que falla: p50 %.2f" % (
        len(bienc), np.nanmedian(bienc), len(malc), np.nanmedian(malc)))

    print("\n   LOS CUATRO `SEGURO Y EQUIVOCADO`, que es para lo que se probaron estas vias:")
    for k in (("UglyNiceAplomadofalcon", 1), ("WhiteMarriedGuineafowl", 0),
              ("WhiteMarriedGuineafowl", 1), ("PowerfulAllLarva", 1)):
        if k not in G or k not in V:
            continue
        ver = V[k]["mano"]
        print("      %-38s l%d  A OJO=%s | a=%s | pie=%s(%s,%.2f) hom_pie=%s(%s) hom_nar=%s(%s)"
              " nariz=%s(%s) | dudosa=%s" % (
                  k[0], k[1], ver, G[k]["geo_a"], G[k]["c_pie_x"],
                  "BIEN" if G[k]["c_pie_x"] == ver else "MAL", float(G[k]["frac_pie_x"]),
                  G[k]["c_hombro_pie"], "BIEN" if G[k]["c_hombro_pie"] == ver else "MAL",
                  G[k]["c_hombro_nariz"], "BIEN" if G[k]["c_hombro_nariz"] == ver else "MAL",
                  G[k]["c_nariz_dir"], "BIEN" if G[k]["c_nariz_dir"] == ver else "MAL",
                  G[k]["dudosa"]))
    print("\n   COBERTURA de lo adoptado y de la ampliacion con el hombro:")
    for nom, col in (("pie firme (ADOPTADO)", "c_pie_x"), ("pie_o_hombro", "c_pie_o_hombro")):
        if nom.startswith("pie firme"):
            cs = [k for k in claves if G[k]["dudosa"] == "False"]
        else:
            cs = [k for k in claves if G[k][col] in ("A", "B")]
        ok = [(G[k]["adoptada"] if nom.startswith("pie firme") else G[k][col]) == V[k]["mano"]
              for k in cs]
        print("      %-22s decide %2d de %2d (%.2f)  acierto %s" % (
            nom, len(cs), len(claves), len(cs) / len(claves),
            "%.3f" % (sum(ok) / len(ok)) if ok else "  -  "))
    tot = [k for k in sorted(G)]
    fir = [k for k in tot if G[k]["dudosa"] == "False"]
    print("      y sobre LOS %d TIRADORES CON POSES, con verdad o sin ella: pie firme decide %d"
          " (%.2f)" % (len(tot), len(fir), len(fir) / max(len(tot), 1)))

    print("\n== LAS PUERTAS DEL PUNTO 10: acierto Y tiradores sin decision ==")
    print("   %-26s %7s %8s %7s %7s %8s %7s" % (
        "puerta", "acierto", "decide", "sin dec", "cuad p50", "frac p50", "veredicto"))
    for nom in PUERTAS:
        cs = [k for k in claves if G[k]["p_" + nom] in ("A", "B")]
        sin = len(claves) - len(cs)
        ok = [G[k]["p_" + nom] == V[k]["mano"] for k in cs]
        a = sum(ok) / len(ok) if ok else float("nan")
        nc = np.median([float(G[k]["n_" + nom]) for k in claves])
        fr = np.nanmedian([float(G[k]["frac_" + nom]) for k in cs]) if cs else float("nan")
        ver = ("NO MEJORA" if not ok or a < MEJORA_MIN else
               ("MEJORA" if sin <= SIN_DECISION_MAX else "gana con coste"))
        print("   %-26s %7s %8d %7d %7.0f %8s %s" % (
            nom, "%.3f" % a if ok else "  -  ", len(cs), sin, nc,
            "%.2f" % fr if cs == cs and np.isfinite(fr) else "  -  ", ver))
    print("   (la cifra a batir es 0,815 sin puerta y 0,741 del operativo por medianas)")
    print("\n   reparto de las dos variables de puerta que son mias, sobre los tiradores con verdad:")
    dm = np.array([float(G[k]["d_manos_m_p50"]) for k in claves])
    hm = np.array([float(G[k]["hombros_m_p50"]) for k in claves])
    print("      distancia minima entre manos de los dos, mediana por tirador: "
          "p10 %.2f p50 %.2f p90 %.2f m   (puerta en %.2f m)" % (
              *np.nanpercentile(dm, [10, 50, 90]), ALCANCE_M))
    print("      anchura de hombros proyectada, mediana por tirador: "
          "p10 %.2f p50 %.2f p90 %.2f m" % tuple(np.nanpercentile(hm, [10, 50, 90])))
    print("\n   cuantos tiradores TIENEN eventos de la pista: accion %d, brazo %d, de %d" % (
        sum(1 for k in claves if int(G[k]["n_ev_accion"]) > 0),
        sum(1 for k in claves if int(G[k]["n_ev_brazo"]) > 0), len(claves)))

    print("\n== LA PRUEBA DIRECTA DE LA HIPOTESIS: acierto POR CUADRO dentro y fuera de la puerta ==")
    print("   si los cuadros de entre accion fueran los que votan mal, DENTRO tendria que acertar")
    print("   claramente mas que FUERA. Es la unica forma de saber si la puerta mira lo que dice.")
    print("   %-26s %9s %7s %9s %7s %9s" % (
        "puerta", "dentro", "n", "fuera", "n", "diferencia"))
    ev = lee_eventos()[0]
    acu = {nom: [[], []] for nom in PUERTAS}
    por_tirador = []
    for k in claves:
        d = carga(k[0])
        lado = k[1]
        KP, B = d["KP"], d["B"]
        W, H, T = int(d["W"]), int(d["H"]), int(d["T"])
        ut = np.zeros(T, bool)
        ut[utiles(d)] = True
        xr = 0.5 * (B[:, 1 - lado, 0] + B[:, 1 - lado, 2]) * W
        pa = np.abs(KP[:, lado, CL.MANOS["A"]["root"], 0] - xr)
        pb = np.abs(KP[:, lado, CL.MANOS["B"]["root"], 0] - xr)
        bien = (pa < pb) == (V[k]["mano"] == "A")
        val = ut & np.isfinite(pa) & np.isfinite(pb)
        m = mascaras_puerta(d, lado, ev)
        for nom in PUERTAS:
            g = val & m[nom]
            f = val & ~m[nom]
            acu[nom][0].append(bien[g])
            acu[nom][1].append(bien[f])
        por_tirador.append((float(bien[val].mean()) if val.any() else float("nan"),
                            k, int(val.sum()), G[k]["geo_a"] == V[k]["mano"],
                            float((val & m["alcance"]).sum()) / max(int(val.sum()), 1),
                            float((val & m["encarado"]).sum()) / max(int(val.sum()), 1)))
    for nom in PUERTAS:
        de = np.concatenate(acu[nom][0]) if acu[nom][0] else np.array([])
        fu = np.concatenate(acu[nom][1]) if acu[nom][1] else np.array([])
        ad = float(de.mean()) if len(de) else float("nan")
        af = float(fu.mean()) if len(fu) else float("nan")
        print("   %-26s %9.3f %7d %9.3f %7d %+9.3f" % (nom, ad, len(de), af, len(fu), ad - af))

    print("\n== Y LA RAZON DE QUE NINGUNA PUERTA SIRVA: el acierto POR CUADRO por tirador ==")
    print("   si el fallo fuera de unos cuantos cuadros malos, los tiradores fallados tendrian")
    print("   acierto por cuadro alrededor de 0,5. Ordenado de peor a mejor:")
    for a, k, n, acierta, fa, fe in sorted(por_tirador):
        print("   %-40s l%d  acierto_cuadro %.3f  sobre %4d cuadros  en_alcance %.2f  "
              "encarado %.2f  por clip: %s" % (
                  k[0], k[1], a, n, fa, fe, "BIEN" if acierta else "MAL"))
    mal = [a for a, _k, _n, ac, _f, _e in por_tirador if not ac]
    bien_ = [a for a, _k, _n, ac, _f, _e in por_tirador if ac]
    print("   los %d fallados: acierto_cuadro p50 %.3f, maximo %.3f" % (
        len(mal), np.median(mal), max(mal)))
    print("   los %d acertados: acierto_cuadro p50 %.3f, minimo %.3f" % (
        len(bien_), np.median(bien_), min(bien_)))

    print("\n== MARCA POR TIRADOR `mano dudosa`, declarada antes de medirla ==")
    print("   si el fallo es del tirador entero y no de unos cuadros, la puerta no puede ser por")
    print("   cuadro sino por TIRADOR. Marca declarada: DUDOSA si la fraccion de cuadros utiles que")
    print("   pasan la puerta es < 0,50, o sea si el tirador pasa menos de la MITAD del clip en esa")
    print("   condicion. 0,50 es `la mayor parte del clip`, no un umbral barrido.")
    for nom, idx in (("en alcance", 4), ("encarado", 5)):
        caz = [(k, ac) for t in por_tirador for k, ac in [(t[1], t[3])] if t[idx] < 0.50]
        no = [(k, ac) for t in por_tirador for k, ac in [(t[1], t[3])] if t[idx] >= 0.50]
        cm = sum(1 for _k, ac in caz if not ac)
        ok2 = [ac for _k, ac in no]
        print("   por `%s < 0,50` -> marca %d de %d; de ellos %d son fallos de (a) (de %d) y %d son"
              " aciertos que se tiran" % (nom, len(caz), len(por_tirador), cm, len(mal),
                                          len(caz) - cm))
        print("        los %d NO marcados aciertan %.3f" % (
            len(ok2), sum(ok2) / len(ok2) if ok2 else float("nan")))
    print("\n   y el cruce con la firmeza `frac_a_voto >= 0,85`:")
    for nom, idx in (("en alcance", 4), ("encarado", 5)):
        fuera = [t[3] for t in por_tirador
                 if t[idx] >= 0.50 and float(G[t[1]]["frac_a_voto"]) >= 0.85]
        print("      %-11s >= 0,50 Y voto firme: %d de %d tiradores, aciertan %.3f" % (
            nom, len(fuera), len(por_tirador),
            sum(fuera) / len(fuera) if fuera else float("nan")))

    print("\n== LA MEJOR PUERTA, cruzada con la firmeza ==")
    for nom in PUERTAS:
        cs = [k for k in claves if G[k]["p_" + nom] in ("A", "B")
              and np.isfinite(float(G[k]["frac_" + nom]))]
        alto = [k for k in cs if float(G[k]["frac_" + nom]) >= 0.85]
        bajo = [k for k in cs if float(G[k]["frac_" + nom]) < 0.85]
        oa = [G[k]["p_" + nom] == V[k]["mano"] for k in alto]
        ob = [G[k]["p_" + nom] == V[k]["mano"] for k in bajo]
        print("   %-26s frac>=0,85: %s sobre %2d   |  frac<0,85: %s sobre %2d" % (
            nom, "%.3f" % (sum(oa) / len(oa)) if oa else "  -  ", len(alto),
            "%.3f" % (sum(ob) / len(ob)) if ob else "  -  ", len(bajo)))

    print("\n== ACIERTO SEGUN LA FIRMEZA DEL VOTO de `a_voto` ==")
    for lo, hi in ((0.0, 0.70), (0.70, 0.85), (0.85, 1.01)):
        cs = [k for k in claves if G[k]["frac_a_voto"] not in ("", "None")
              and lo <= float(G[k]["frac_a_voto"]) < hi]
        ok = [G[k]["a_voto"] == V[k]["mano"] for k in cs]
        print("   frac_voto en [%.2f, %.2f)  acierto %s  sobre %d" % (
            lo, hi, "%.3f" % (sum(ok) / len(ok)) if ok else "  -  ", len(ok)))
    print("\n== ERRORES DEL CRITERIO (a), uno por uno, Y SI ALGUNA PUERTA LOS CAZA ==")
    for k in claves:
        if G[k]["geo_a"] != V[k]["mano"]:
            caza = [n for n in PUERTAS[1:] if G[k]["p_" + n] == V[k]["mano"]]
            print("   %-40s l%d  a=%s  A OJO=%s (%s %s)  margen_x %5.1f px  a_voto=%s (frac %.2f)"
                  % (k[0], k[1], G[k]["geo_a"], V[k]["mano"], V[k]["cue"], V[k]["certeza"],
                     float(G[k]["margen_x_px"]), G[k]["a_voto"], float(G[k]["frac_a_voto"])))
            print("        puertas que lo arreglan: %s" % (", ".join(caza) if caza else "NINGUNA"))
    dis = [k for k in claves if G[k]["concuerdan"] == "False"]
    print("\n== DISCREPANCIAS geometria (a) contra guante: %d de %d ==" % (len(dis), len(claves)))
    for k in dis:
        print("   %-40s l%d  a=%s guante=%s(%s frac %.2f)  A OJO=%s (%s)  gana %s" % (
            k[0], k[1], G[k]["geo_a"], G[k]["guante"],
            "solido" if G[k]["solido"] == "True" else "flojo", float(G[k]["frac_voto"]),
            V[k]["mano"], V[k]["cue"],
            "guante" if G[k]["guante"] == V[k]["mano"] else "geometria"))


if __name__ == "__main__":
    if sys.argv[1:2] == ["careo"]:
        careo()
    elif sys.argv[1:2] == ["estabilidad"]:
        estabilidad_frac()
    elif sys.argv[1:2] == ["arbitro"]:
        tasa_esgrima()
    else:
        stems = sys.argv[1:] or None
        print("clips en la lista del corpus con poses del formato nuevo: %d" % len(clips()))
        tabla(stems)

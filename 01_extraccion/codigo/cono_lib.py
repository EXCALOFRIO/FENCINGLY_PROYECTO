"""ETAPA E, comun: mapa de indices MEDIDO, eleccion de mano armada, y el cono como region.

EL MAPA ES MEDIDO, NO COPIADO. Se dedujo en `e0_mapa_indices.py` a partir de `flip_pairs` del
`config.json` del modelo y comprobado por constancia de hueso y pintandolo sobre el fotograma:

   0 nariz   1-2 ojos   3-4 orejas   5-6 hombros   7-8 codos   9-10 caderas
  11-12 rodillas   13-14 tobillos   15-17 pie A   18-20 pie B
  21-41 MANO A (root 21, pulgar 22-25)     42-62 MANO B (root 42, pulgar 43-46)
  63 en adelante cara

Los indices que daba el plan (`hand_root` 91 y 112, pulgar 92-95 y 113-116) caen **sobre la careta**.
Con ellos el «pulgar» mide 1,4-2,8 px y la distancia a la muneca sale el 47 % del alto de la caja.

NO HAY MUNECA en el bloque de cuerpo: la muneca ES el root de la mano. Por eso el eje largo del brazo
que se usa aqui es `codo -> root`, y no `wrist -> hand_root`, que no existe.

EL CONO ES UNA REGION DECLARADA, NO UNA HOJA. El peso `1 - angulo/apertura` es una FORMA ELEGIDA,
no una probabilidad calibrada: nadie ha medido donde esta la hoja dentro del cono. Se llama peso y no
probabilidad a proposito.
"""
from __future__ import annotations

import pathlib

import numpy as np

import rutas as RU

PRUEBAS = RU.TRABAJO
MANOS = {"A": {"root": 21, "pulgar": [22, 23, 24, 25], "codo": 8, "hombro": 6},
         "B": {"root": 42, "pulgar": [43, 44, 45, 46], "codo": 7, "hombro": 5}}
HOMBROS = (5, 6)
PIES = {"A": {"talon": 20, "dedos": (18, 19), "tobillo": 14, "cadera": 10},
        "B": {"talon": 17, "dedos": (15, 16), "tobillo": 13, "cadera": 9}}
"""CADENA DE LADO DEL TREN INFERIOR, MEDIDA, no copiada de ningun README. `A` es el lado de la raiz
de mano 21 y `B` el de la 42. Las tres pruebas estan en el punto 12 del docstring de
`mano_guante.py`: el talon es el punto del grupo mas cercano al tobillo (0,209 y 0,196 anchuras de
hombro contra 0,46-0,51 de los dedos); el cv del hueso empareja 10-12, 12-14 y 14-20; y la coherencia
lateral empareja el hombro 6 con la cadera 10 en el 0,989 de los cuadros y con el tobillo 14 en el
0,837, contra 0,011 y 0,163 del cruce."""
USA_PIE_FIRME = True
FRAC_PIE_MIN = 0.85
N_PIE_MIN = 0
"""CUADROS MINIMOS QUE TIENEN QUE VOTAR para creerse `frac_pie`. Medido y dejado en 0, o sea APAGADO,
porque con la muestra de hoy no arregla ningun error: solo quita cobertura.

Por que existe el mando: un remuestreo de los 36 tiradores con >= 120 cuadros que votan dice que la
probabilidad de declarar FIRME a un tirador que no lo es -- `frac_sub >= 0,85` cuando el clip completo
esta por debajo -- **no baja de 0,10 con ningun tamanio probado**: 0,416 con 10 cuadros, 0,293 con 20,
0,248 con 30, 0,188 con 50 y **0,130 con 100**. Y como el remuestreo es con reposicion sobre el mismo
clip, esa es una **cota por lo BAJO**. O sea que el corte 0,85 por si solo no garantiza nada a los
tamanios cortos.

    politica                        decide de 38   acierto
    `frac >= 0,85` (lo de hoy)          17          1,000
    `frac >= 0,85` y `n >= 120`         15          1,000
    `frac >= 0,85` y `n >= 200`          5          1,000
    `frac >= 0,90` y `n >= 120`         10          1,000

Solo **2 de los 17 firmes** tienen menos de 120 cuadros votando (el minimo es 54), asi que exigirlo
cuesta 2 tiradores y no gana ninguno. Se deja en 0 y se documenta para cuando la muestra crezca."""
USA_HOMBRO_SI_PIE_FLOJO = False
"""AMPLIACION MEDIDA Y NO ADOPTADA, porque la adopcion la decide el usuario y lo que decidio fue el
pie. Ponerlo a `True` hace que, cuando el pie no es firme, decida el HOMBRO si el suyo lo es
(`mano_hombro`, la direccion de la nariz proyectada sobre los hombros en vez de sobre las manos).

    politica                  decide          acierto
    pie firme (lo adoptado)   13 de 27 (0,48)  13 de 13 sin fallo
    pie firme o hombro firme  15 de 27 (0,56)  15 de 15 sin fallo

O sea **+2 tiradores de cobertura sin ningun fallo nuevo**, y es un cambio de una linea. No se activa
solo. Y el aviso de siempre: «15 de 15 sin fallo» no es «acierta siempre».

**RETIRADO, Y LO TUMBO MAS MUESTRA.** Con 27 tiradores daba 15 de 15 sin fallo; con 38 da **22 de 23,
o sea 0,957**, y el fallo es `HairyCourteousEarwig` lado 0, donde el pie dice B con 0,69 (no firme) y
el hombro dice A con 0,90 (firme) y la verdad es B. **Se queda apagado**: `FALLBACK_MAYORIA` cubre el
mismo hueco mejor (0,921 sobre los 38) y sin tocar el subconjunto firme. Es un ejemplo limpio de por
que 15 casos no bastaban para adoptar nada."""
FALLBACK_NARIZ = True
"""LO QUE SE DEVUELVE A LOS DUDOSOS: `nariz_dir` EN VEZ DE `mayoria_3`. ADOPTADO el 2026-09-11.

DE DONDE VIENE. El usuario miro `foil_proudDungBeetleBoglenTerrier_c0163.png` y dijo: *"la mano
dominante le sale en la mano de atras... hemos dicho que sea la mano del hombro mas adelantado"*.
Tenia razon: ahi la cascada caia en `mayoria_3`, que voto B (la mano de atras) 2 a 1 con sus dos
miembros mas flojos, mientras `nariz_dir` -- que es exactamente la regla que el usuario enuncia, la
mano mas adelantada en la direccion en que mira el tirador -- decia A, la correcta.

EL EMPATE, Y POR QUE AUN ASI SE ADOPTA. Sobre los 59 tiradores con verdad, cambiar el fallback de
`mayoria_3` a `nariz_dir` da **exactamente el mismo numero**:

    criterio en el fallback      cascada sobre los 59   sobre los 11 dudosos
    `mayoria_3` (lo de antes)          0,949                  0,818
    `nariz_dir` (lo de ahora)          **0,949**              **0,818**

O sea que **no se compra acierto: se rompe un empate**. Y se rompe con tres razones, ninguna de las
cuales es «me gusta mas»:

  1. **Un caso verificado FUERA de la muestra.** `proudDungBeetleBoglenTerrier` lado 0 no esta en
     `verdad_mano.csv` -- es uno de los tres clips que se PUBLICAN y ninguno esta anotado, que es
     justo al reves de lo que conviene -- y el usuario lo verifico a ojo. Ahi `nariz_dir` acierta y
     `mayoria_3` falla.
  2. **Es el mejor criterio suelto**, con diferencia: 0,932 sobre los 59, contra 0,898 de `a_voto`,
     0,881 de `pie_x` y 0,831 de (a). `mayoria_3` es la mayoria de los tres primeros, o sea que
     **arrastra a los dos peores** y puede ganarle al mejor 2 a 1, que es literalmente lo que paso.
  3. **Es mas simple**: un criterio en vez de una mayoria de tres.

Y ES LA REGLA QUE EL USUARIO ENUNCIA. El dice «el hombro mas adelantado»; `mano_hombro` proyecta los
HOMBROS y `mano_nariz` las RAICES DE MANO en la misma direccion de frente. Se adopta la de las manos
porque esta medida por encima (0,932 contra 0,915) y porque la pregunta es cual es la MANO.

`FALLBACK_MAYORIA` se queda como interruptor y como documentacion de lo que habia. Con los dos a
`True` manda este."""
DECIDE_EN_LA_GUARDIA = False
"""IDEA DEL USUARIO, MEDIDA Y **REFUTADA CON SU CIFRA**. 2026-09-11.

Literal: *"que no cambie nunca durante el asalto, que la siga trackeando, porque siempre empezara la
mano dominante delante de la otra... y luego que no la cambie aunque se retrase"*. Son dos cosas:

**(1) «que no cambie durante el asalto»: YA SE CUMPLE, y conviene decirlo.** `mano_armada` decide
**una vez por clip** por voto mayoritario sobre todos los cuadros utiles, no por cuadro. La mano no
cambia nunca a mitad de asalto, ni aunque el brazo se retrase. Lo que el usuario vio mal no era un
cambio a media accion: era **una sola decision, y equivocada**.

**(2) «decidirla en la guardia del principio»: MEDIDO Y NO PASA.** La premisa de esgrima es correcta
-- en guardia la mano armada va delante -- pero **no se cumple en como estan cortados estos clips**,
que empiezan donde empieza la accion y no en la puesta en guardia. Votando solo el tramo inicial de
cuadros utiles, el acierto **baja monotonamente cuanto mas se restringe**:

    tramo inicial    por raiz de mano    por hombro
       10 %              0,814             0,780
       20 %              0,831             0,780
       25 %              0,831             0,797
       33 %              0,864             0,847
       50 %              0,881             0,847
    **todo el clip**   **0,932**         **0,915**

Y en los 11 tiradores que la cascada no resuelve, peor todavia: 0,636 con el 10 % contra 0,818 con el
clip entero. O sea que **el principio del clip es la PEOR ventana**, no la mejor, y cuantos mas
cuadros se miran mejor sale. Se deja el interruptor y la tabla para que la idea no haya que volver a
probarla a ciegas; para adoptarla harian falta clips cortados desde la puesta en guardia."""


USA_APUNTA_SI_DUDOSA = False
"""`apunta_frac`: MEDIDO, NUNCA FALLA DONDE SE PUEDE COMPROBAR, Y **NO ADOPTADO** POR FALTA DE
MUESTRA. 2026-09-11, a raiz de un defecto que el usuario vio en una figura publicada.

EL CASO QUE LO MOTIVA. En `proudDungBeetleBoglenTerrier` lado 0 la cascada eligio la mano de ATRAS y
el cono salio apuntando FUERA del rival. Las cuatro fracciones de ese tirador andan por **0,51-0,69**
-- monedas al aire -- y `mayoria_3` gano 2 a 1 con los dos votos mas flojos. Medido sobre las poses
de esa tirada:

    el antebrazo de la mano **A** apunta al rival en **0,941** de 426 cuadros medidos
    el de la mano **B**, en **0,660**
    y en c0163 la raiz de A esta en x=381 contra x=295 de B, con el rival en x=1736

O sea que la mano adelantada es A, que es la que el usuario senialo, y **la separacion es de 0,28**
donde ningun otro criterio separaba nada.

QUE ES: por mano, la fraccion de cuadros medidos en que el antebrazo `codo -> root` tiene componente
x **hacia** el centro de la caja del rival. Gana la fraccion mas alta, y `|fracA - fracB|` es la
separacion. **No es `(b')`**, que ya esta medido en 0,667 y rechazado: (b') compara el producto
escalar por MEDIANA del clip, o sea magnitud, y se lo come el caso del fondo donde los dos brazos se
alinean con el ataque. Esto vota el SIGNO por cuadro, que sigue separando cuando una mano queda
detras del cuerpo.

CRITERIO DE ACEPTACION DECLARADO ANTES DE MEDIR, como peldanio detras de `encarado_accion`:
  (i)   acierto >= 0,90 sobre los que decide
  (ii)  **estrictamente mejor** que `mayoria_3` en el subconjunto que la cascada resuelve hoy por
        `mayoria_3`/`geometria_a`, que es el unico sitio donde puede entrar
  (iii) no tocar el subconjunto firme

MEDIDO: **solo 6 de los 59 tiradores anotados tienen `cono_poses_*.npz`**, que es lo que hace falta
para calcularlo. Sobre esos 6 acierta **6 de 6**, y con el barrido entero de separacion minima
(0,00 / 0,05 / 0,10 / 0,15 / 0,20 / 0,25 / 0,30 / 0,40) **acierta 1,000 en todos los cortes**. Pero
en el sitio donde podria entrar hay **n = 3**, y ahi la cascada tambien hace 1,000.

VEREDICTO: **(ii) no se cumple, y no por perder sino por EMPATAR con n = 3.** Es el tercer criterio
de este modulo que se cae por lo mismo -- `REGLA_18` en su primera medida y `USA_NARIZ_SI_PIE_FLOJO`
-- y la leccion es la misma: un criterio que el conjunto de evaluacion no puede distinguir no esta
probado, por limpio que salga. Se deja apagado, medido y reversible en una linea.

QUE HARIA FALTA PARA JUZGARLO: `cono_poses` de los clips ya anotados, o anotar los clips que ya
tienen `cono_poses`. Hoy los tres clips que se PUBLICAN no estan anotados, que es justo al reves de
lo que conviene, y por eso el defecto lo encontro el usuario y no una cifra.

MIENTRAS ESTE APAGADO, el defecto de la figura se arregla en otro sitio y sin tocar ningun criterio:
`panel_publico.decide_manos` hace que **manda la mano del NPZ** cuando discrepa de la suya, porque el
NPZ es el dato que se publica y sale de `cono_poses`. Eso quita la contradiccion figura-dato; la
incertidumbre sigue marcada."""
USA_NARIZ_SI_PIE_FLOJO = False
"""AMPLIACION MEDIDA CONTRA UN CRITERIO DECLARADO ANTES DE MEDIRLA, Y **NO ADOPTADA**. 2026-09-11.

De donde venia: `nariz_dir` y `hombro_nariz` sacan **1,000 sobre los 19 del reservado con cobertura
total**, y `pie_x` suelto saca 0,895. Visto asi parece una mejora disponible y gratis, y ese fue el
planteamiento de partida. **Comparaba contra el criterio equivocado.** El que ocupa el sitio no es
`pie_x` suelto: es la CASCADA, y la cascada saca **0,949 sobre los 59** contra 0,932 de `nariz_dir` y
0,915 de `hombro_nariz`. O sea que sobre todo lo que hay con verdad, la cascada ya va por delante de
las dos. Es el mismo error de denominador que este proyecto tiene documentado cinco veces.

CRITERIO DECLARADO ANTES DE IMPRIMIR UNA CIFRA, y en dos variantes porque hay dos sitios donde
podrian entrar:

    V1  sustituir `mayoria_3` como lo que se devuelve a los dudosos, siguiendo marcado `dudosa`.
        Exigido: acertar ESTRICTAMENTE MAS que `mayoria_3` en el subconjunto que hoy sale dudoso.
    V2  peldanio firme nuevo detras de `encarado_accion`, con su propia firmeza `>= 0,85` (el corte
        que ya estaba declarado, no uno nuevo) y `dudosa = False`.
        Exigido: (i) acierto >= 0,90 en el RESERVADO sobre los que decide, (ii) **anadir cobertura
        no dudosa > 0 en el reservado**, (iii) ningun fallo nuevo entre los no dudosos.

MEDIDO (59 tiradores con verdad a ojo, 19 de ellos reservados; la cascada reproducida fila a fila
contra la columna `adoptada` del CSV, 0 discrepancias de 59, porque si no la comparacion no vale):

    variante   no dudosos          acierto no dudoso   dudosos    acierto de los 59
    hoy        48 de 59            0,979               11         0,949
    V1         48 de 59            0,979               11         **0,949, igual**
    V2         50 de 59            0,980                9         **0,949, igual**

    y en el subconjunto que hoy sale dudoso, que es el unico que V1 podria tocar (n=11):
        `mayoria_3`, lo de hoy   0,818
        `nariz_dir`              **0,818**, ni uno mas
        `hombro_nariz`           0,636

VEREDICTO: **ninguna de las dos pasa, y por motivos distintos.**

    V1 falla su unico requisito: empata con `mayoria_3` en el sitio exacto donde tendria que ganar.
    V2 falla el (ii): los 2 tiradores que anade (`BabyishIdioticCockerspaniel` l0 y
       `BlissfulTiredAiredale` l0, los dos BIEN) estan **los dos en entrenamiento**, y en el
       reservado anade **CERO**. O sea que el reservado no puede juzgarla.

Y ese es exactamente el motivo por el que `REGLA_18` tampoco se adopto: una regla que el conjunto
reservado no ejecuta ni una vez no esta probada, por buena que parezca en entrenamiento. Se deja
apagado, medido y reversible en una linea, que es lo que este modulo ya hizo con
`USA_HOMBRO_SI_PIE_FLOJO`. Para juzgarla de verdad hacen falta reservados donde el pie no sea firme
y la nariz si: hoy no hay ninguno."""
FALLBACK_MAYORIA = True
"""LO QUE SE DEVUELVE PARA LOS TIRADORES `dudosa`. **ENCENDIDO el 2026-09-11**, y la razon por la que
se enciende queda escrita: la puerta de adopcion **0,90 con <= 4 sin decidir estaba declarada de
antemano** y `mayoria_3` la pasa con 0,921 sobre 38 y cobertura total, o sea que entra por el criterio
del proyecto y no por gusto; **no toca lo que se adopto**, porque el subconjunto firme sigue siendo el
pie y sigue en 17 de 17 sin fallo; y es reversible en una linea.

Hoy, cuando el pie no es firme, se devuelve el criterio (a). Con `True` se devuelve en su lugar la
MAYORIA de `a_voto`, `pie` y `nariz`, que sigue marcandose `dudosa` igual. Sobre los 21 tiradores que
la politica declara dudosos:

    criterio devuelto para los dudosos    acierto
    (a), lo de hoy                         0,667
    mayoria_3                              **0,857**

Y sobre los 38 con verdad, `mayoria_3` acierta **0,921 con cobertura total**, que es la **primera vez
que algo pasa la puerta declarada** (>= 0,90 con <= 4 sin decidir). Sus tres fallos son los tres
tiradores irreducibles, y **los tres ya salen marcados `dudosa`**, asi que encender esto no mete
ningun error en la parte que la tuberia usa: solo mejora lo que se devuelve en la parte que la
tuberia debe excluir. No se activa solo porque la adopcion la decide el usuario."""
"""ADOPTADO el 2026-09-11 por decision del usuario. Poner `False` devuelve el comportamiento viejo
(siempre el criterio (a)) sin tocar nada mas; las dos columnas se calculan igual. Precedente del
interruptor: `USA_CRUDAS = False` en `extrae_pareja.py`.

`FRAC_PIE_MIN = 0,85` no se ajusto aqui: es el mismo corte que ya estaba declarado para
`frac_a_voto`, reutilizado. Nadie ha barrido ese umbral para el pie y con 27 tiradores no se debe."""
CONF_MIN = 0.3
ALTURA_M = 1.75
HOJA_M = 0.90
"""Longitud maxima de hoja de florete de la FIE. Es una COTA reglamentaria, no una medida del clip.

VERIFICADO EN FUENTE PRIMARIA el 2026-09-10, porque la cifra estaba puesta de memoria: FIE *Material
Rules* (Book 3), edicion de agosto de 2026,
`https://static.fie.org/uploads/40/204157-book%20material%20August%202026%20ang.pdf`

    m.8.4  «The maximum length of the blade is 90 cm»
    m.7    «The maximum total length of the foil is 110 cm»
    m.4.1  mango maximo 20 cm entre las lineas B y E, 18 cm entre B y D
    m.6    peso del florete montado, menos de 500 g

Cotejado con la edicion de diciembre de 2025: las cifras del florete son identicas. Y un aviso, que
aparecio al verificar: **circula la cifra de 88 cm para la hoja de florete y es FALSA** -- 88 cm es
la hoja de SABLE (m.23). Los 90 + 20 = 110 cm cuadran con la convencion de medida de m.3.4.

SALVEDAD SOBRE COMO SE USA AQUI: `L = HOJA_M * pxm` mide 90 cm **desde el `root` de la mano**, y la
mano agarra por detras de la cazoleta, asi que el alcance real mano-punta es algo MAYOR que 90 cm (la
hoja se mide desde la cara delantera de la guarda, linea B). Asi que `L` es una cota por lo BAJO del
alcance, o sea conservadora para decir «amenaza»: nunca inventa alcance que no existe."""


def carga(clip, suf=""):
    with np.load(PRUEBAS / ("cono_poses_%s%s.npz" % (clip, suf)), allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def bueno(sc, idx):
    """Confianza Y coordenada finita. Hay puntos con confianza alta y coordenada NaN."""
    return (sc[:, idx] >= CONF_MIN).all(axis=1)


def escala_px_m(B, lado, H):
    """px por metro a partir del alto de la caja y una estatura de 1,75 m. Por cuadro y mediana."""
    alto = (B[:, lado, 3] - B[:, lado, 1]) * H
    return alto / ALTURA_M, float(np.nanmedian(alto)) / ALTURA_M


def mano_armada(kp, sc, B, W, H, lado, org=None, mask=None, guante=None, accion=None):
    """DOS criterios declarados e independientes. Si discrepan, devuelve None y se dice.

    (a)  posicion: gana la mano con x mediana mas cerca del rival
    (b') direccion: gana el antebrazo cuya direccion `codo -> root` apunta mas al rival, medida como
         producto escalar con la direccion `hombro propio -> centro del rival`

    LOS DOS CRITERIOS SECUNDARIOS QUE PROBE FALLARON, cada uno en un tirador distinto, y **el
    operativo se queda solo en (a)**:

      «la mas extendida», |hombro-root| mayor  ->  falla en `SentimentalLarge...` lado 1: la mano
        armada da 61 px y la de atras 73 px. La razon es de esgrima: en guardia el brazo de ATRAS
        tambien va extendido, hacia atras y arriba.
      (b') «la que mas apunta al rival»        ->  falla en `AdorableFitting...` lado 1: da 0,918
        para la mano de atras contra 0,859 para la armada. En pleno fondo los DOS brazos quedan
        alineados con la direccion del ataque, asi que el criterio casi no separa.

    EL «4 DE 4» QUEDA SUSTITUIDO POR UNA CIFRA DE VERDAD (2026-09-11). El usuario reporto que esta
    eleccion falla a menudo y tenia razon. Contra una anotacion a ojo de **27 tiradores** de 16
    clips de florete, guardada en `02_validacion/muestras/mano_dominante/verdad_mano.csv` y hecha
    por un agente, no por un arbitro:

        criterio (a), el de aqui           0,741  (20 de 27)   <- LINEA BASE
        (a) por voto por cuadro            0,815  (22 de 27)   +2 casos, dentro del ruido
        (b')                               0,667
        «la mas extendida»                 0,519
        guante blanco alrededor de la raiz  0,444   <- RETIRADO, ver `mano_guante.py`
        cazoleta oscura alrededor          0,500   <- RETIRADO, ver `mano_guante.py`
        contestar siempre A / siempre B    0,519 / 0,481   <- el azar informado

    El «4 de 4» de la version anterior no era falso: era una muestra de 4. Con 27 el criterio acierta
    3 de cada 4 y falla 1 de cada 4, y las siete fallas estan en `mano_guante.careo()`.

    POR QUE FALLA, medido: cinco de las siete fallas son cuadros en los que el tirador NO esta
    encarado, entre accion y accion, con el arma atras o abajo; ahi «la mano mas cerca del rival» es
    la desarmada. Dos de esas siete fallan con margen de 145 px y 65 px, o sea que NO son casos
    apretados: el criterio esta seguro y equivocado. Las otras dos fallan con margen de 1,5 y 1,9 px,
    o sea a cara o cruz.

    LO QUE SI SE PUEDE USAR ES LA FIRMEZA. `mano_armada_voto` da `frac_a_voto`, la fraccion de
    cuadros que votan a la ganadora, y el acierto se parte con ella: **0,895 (17 de 19) cuando
    `frac_a_voto >= 0,85`** contra **0,625 (5 de 8) por debajo**. Asi que el operativo sigue siendo
    (a) -- porque +2 casos sobre 27 no es una mejora demostrada -- pero ahora la columna dice cuando
    NO hay que creerla. **Quien monte el cono deberia tratar `firme == False` como MANO DUDOSA y
    excluir ese tirador**, en vez de heredar un vertice en la mano equivocada: sobre esta muestra eso
    deja 19 de 27 tiradores con 0,895.

    PUNTO CIEGO DE `firme`, medido y no escondido: los 2 fallos que se cuelan con `firme == True`
    tienen `frac_a_voto = 1,00` y son los dos tiradores en los que el criterio (a) se equivoca en el
    **100 % de sus cuadros** (`UglyNiceAplomadofalcon` lado 1 y `WhiteMarriedGuineafowl` lado 0, los
    dos con el arma en la mano del lado OPUESTO al rival durante todo el clip). **La unanimidad mide
    consistencia, no verdad**, asi que `firme` no vale como filtro de calidad sin este aviso al lado.
    Y se probaron SIETE puertas de «cuadro que sabe la respuesta» -- en alcance, brazo extendido, de
    perfil, dentro de evento de accion, dentro de evento de brazo, encarado por signo de la nariz, y
    combinaciones -- y **ninguna llega a 0,90**; las cifras estan en el punto 10 del docstring de
    `mano_guante.py`.

    Y HAY DOS CRITERIOS MEJORES QUE ESTE, que NO miran al rival y que estan en
    `mano_guante.criterios_cuerpo` (puntos 12 y 13 de su docstring):

        `nariz_dir`  la mano mas adelantada en la direccion en la que mira el propio tirador
                     (signo de `x_nariz - x_medio_hombros`):        **0,889** sobre 27
        `pie_x`      el hecho de esgrima de que en guardia el pie adelantado es el del lado del
                     brazo armado, con el pie de delante identificado por la orientacion de su
                     propio eje `talon -> dedos`:                    **0,852** sobre 27
        y con su propia firmeza `>= 0,85`: `pie_x` da **1,000 sobre 13** y `nariz_dir` **1,000 sobre
        11**, y la firmeza del PIE si marca como dudosos los tres tiradores que nadie acierta,
        mientras que `frac_a_voto` les da 1,00 / 0,83 / 0,68.

    LO QUE DEVUELVE ESTA FUNCION A PARTIR DEL 2026-09-11, por decision del usuario: **`USA_PIE_FIRME
    = True`**, o sea que `mano` es la del criterio del PIE cuando su voto es firme
    (`frac_pie >= FRAC_PIE_MIN = 0,85`), y la del criterio (a) cuando no lo es. En los dos casos se
    publica `dudosa`, y **la tuberia tiene que mirar `dudosa` y excluir al tirador**, no fiarse de
    `mano` a secas. Se devuelve la respuesta de (a) en vez de `None` para no romper a quien ya
    llamaba a esta funcion; el que quiera calidad mira la marca.

        subconjunto                 acierto            cobertura
        pie firme (lo adoptado)     13 de 13 sin fallo  13 de 27  (0,48)
        (a) sobre todos (lo viejo)  0,741, 7 fallos     27 de 27

    **«13 de 13 sin ningun fallo» NO es «acierta siempre»**: con 13 observaciones el intervalo de
    confianza inferior anda por 0,75-0,80. La cifra que se afirma es «no ha fallado ninguno de 13».

    LAS CIFRAS VIGENTES, CON 59 TIRADORES ANOTADOS Y EL ACIERTO SEPARADO DE LA COBERTURA (esta es la
    tabla que hay que citar, y la que estuvo mal citada en nueve sitios hasta el 2026-09-11):

        criterio                                 reservado (19)   todos (59)   cobertura
        (a), la linea base                            0,947         0,831       total
        `pie_x` SIN la puerta de firmeza              0,895         0,881       total
        `pie_x` CON voto firme, lo adoptado      12 de 12 sin fallo  29 de 29   **12 de 19**
        `nariz_dir`                                   1,000         0,932       total
        `hombro_nariz`                                1,000         0,915       total
        `mayoria_3`                                   1,000         0,897       total
        **la CASCADA entera** (lo que devuelve esto)  **1,000**     **0,949**   total

    **EL 1,000 CON COBERTURA 19 DE 19 ES DE LA CASCADA, NO DE LA REGLA DEL PIE.** La ficha publica
    decia *"the front foot rule -> 1.000 accuracy over 19 of 19"*, y eso junta en una frase el
    acierto de la cascada con la cobertura de la cascada y se lo atribuye a su primer peldanio. La
    regla del pie sola responde por **12 de los 19** y no ha fallado ninguno; el criterio del pie
    sin su puerta de firmeza acierta 0,895. Las tres cifras son ciertas por separado.

    Y OJO CON PREFERIR `nariz_dir` POR ESA TABLA, que fue mi primer impulso: 1,000 en el reservado
    con cobertura total parece batir al 0,895 del pie, pero **el que ocupa el sitio es la cascada**,
    que hace 0,949 sobre los 59 contra 0,932 de `nariz_dir`. Medido y no adoptado en
    `USA_NARIZ_SI_PIE_FLOJO`, con su criterio declarado antes.

    RADIO DE IMPACTO PARA QUIEN YA LLAMABA A ESTA FUNCION, medido sobre los 44 tiradores con poses:
    18 (0,41) salen con `dudosa == False`, y de esos **solo 2 cambian de mano** respecto del criterio
    (a): `GentleQuerulousElephantseal` lado 1 (B -> A) y `UglyNiceAplomadofalcon` lado 1 (A -> B). Los
    dos cambian **de mal a bien** contra la anotacion a ojo: eran dos de los siete fallos de (a). Todo
    lo demas devuelve lo mismo que antes. Lo que cambia de verdad es que hay una marca para los 26
    (0,59) que no se deben usar.

    Para volver al comportamiento anterior basta `cono_lib.USA_PIE_FIRME = False`, y entonces `mano`
    vuelve a ser siempre la de (a); las dos columnas siguen calculandose, asi que la comparacion es
    reproducible sin tocar codigo.

    `org` y `mask` son opcionales y sirven para la firmeza y para el criterio del pie; **sin ellos se
    vota con todos los cuadros con confianza, y las cifras de arriba, que se midieron con
    `origen == 0` y `mask`, no son exactamente las que saldran**. Quien quiera reproducirlas tiene que
    pasarlos. `guante` es el dict de `mano_guante.voto` si alguien lo tiene calculado, y se publica
    como `corrobora_guante` **a titulo informativo**: su acierto propio es 0,444, o sea que una
    discrepancia con el guante no es motivo para cambiar la eleccion.
    """
    k, s = kp[:, lado], sc[:, lado]
    rival = 1 - lado
    xr = float(np.nanmedian(0.5 * (B[:, rival, 0] + B[:, rival, 2]) * W))
    yr = float(np.nanmedian((B[:, rival, 1] + 0.30 * (B[:, rival, 3] - B[:, rival, 1])) * H))
    d = {}
    for nom, M in MANOS.items():
        o = bueno(s, [M["root"], M["hombro"], M["codo"]])
        if o.sum() < 10:
            return None, {}
        ante = k[o, M["root"]] - k[o, M["codo"]]
        ante = ante / np.maximum(np.linalg.norm(ante, axis=1, keepdims=True), 1e-9)
        alr = np.array([xr, yr])[None, :] - k[o, M["hombro"]]
        alr = alr / np.maximum(np.linalg.norm(alr, axis=1, keepdims=True), 1e-9)
        d[nom] = {"x": float(np.median(k[o, M["root"], 0])),
                  "ext": float(np.median(np.linalg.norm(k[o, M["root"]] - k[o, M["hombro"]], axis=1))),
                  "apunta": float(np.median((ante * alr).sum(axis=1))),
                  "n": int(o.sum())}
    a = "A" if abs(d["A"]["x"] - xr) < abs(d["B"]["x"] - xr) else "B"
    b = "A" if d["A"]["apunta"] > d["B"]["apunta"] else "B"
    d["cerca"], d["apuntando"], d["x_rival"] = a, b, xr
    d["extendida_RECHAZADO"] = "A" if d["A"]["ext"] > d["B"]["ext"] else "B"
    d["margen_x_px"] = abs(abs(d["A"]["x"] - xr) - abs(d["B"]["x"] - xr))
    d["margen_apunta"] = abs(d["A"]["apunta"] - d["B"]["apunta"])
    d["corrobora_b2"] = bool(a == b)
    """Firmeza por voto. No cambia `a`: solo dice si creerla. Ver el docstring para las cifras."""
    mv, dv = mano_armada_voto(kp, sc, B, W, H, lado, org, mask)
    d["a_voto"], d["frac_a_voto"] = mv, dv.get("frac_a_voto", float("nan"))
    d["a_dist2d"], d["frac_a_dist2d"] = dv.get("a_dist2d"), dv.get("frac_a_dist2d", float("nan"))
    d["corrobora_voto"] = None if mv is None else bool(mv == a)
    d["firme"] = bool(d["frac_a_voto"] >= 0.85) if mv is not None else False
    if guante is not None and guante.get("mano") in ("A", "B"):
        d["guante"], d["frac_guante"] = guante["mano"], guante.get("frac_voto", float("nan"))
        d["corrobora_guante"] = bool(guante["mano"] == a)
    else:
        d["guante"], d["frac_guante"], d["corrobora_guante"] = None, float("nan"), None
    """EL CRITERIO ADOPTADO. `a` se sigue publicando entero en `d` para poder comparar."""
    mp, dp = mano_pie(kp, sc, lado, org, mask)
    d.update(dp)
    d["mano_a"] = a
    d["firme_pie"] = bool(mp is not None and dp["frac_pie"] >= FRAC_PIE_MIN
                          and dp["n_pie"] >= N_PIE_MIN)
    d["corrobora_pie"] = None if mp is None else bool(mp == a)
    mh, dh = mano_hombro(kp, sc, lado, org, mask)
    d.update(dh)
    mn, dn = mano_nariz(kp, sc, lado, org, mask)
    d.update(dn)
    d["firme_hombro"] = bool(mh is not None and dh["frac_hombro"] >= FRAC_PIE_MIN)
    d["corrobora_hombro"] = None if mh is None else bool(mh == mp)
    vs = [v for v in (d.get("a_voto"), mp, mn) if v in ("A", "B")]
    d["mayoria_3"] = (None if len(vs) < 2 else
                      ("A" if vs.count("A") > vs.count("B") else "B"))
    """La puerta de accion del punto 18, que solo existe si el consumidor pasa `accion`."""
    d["accion_dada"] = accion is not None
    d["mano_accion"], d["frac_accion"], d["n_accion"] = None, float("nan"), 0
    if accion is not None:
        enc = mascara_encarado(kp, sc, B, W, H, lado)
        ma, da = mano_armada_voto(kp, sc, B, W, H, lado, org, mask,
                                  enc & np.asarray(accion, bool))
        d["mano_accion"] = ma
        d["frac_accion"] = da.get("frac_a_voto", float("nan"))
        d["n_accion"] = da.get("n_vota", 0)
    if USA_PIE_FIRME and d["firme_pie"]:
        d["mano"], d["por"], d["dudosa"] = mp, "pie_firme", False
    elif USA_ACCION_SI_PIE_FLOJO and d["mano_accion"] in ("A", "B"):
        d["mano"], d["por"], d["dudosa"] = d["mano_accion"], "encarado_accion", False
    elif USA_HOMBRO_SI_PIE_FLOJO and d["firme_hombro"]:
        d["mano"], d["por"], d["dudosa"] = mh, "hombro_firme", False
    elif USA_NARIZ_SI_PIE_FLOJO and mn is not None and d["frac_nariz"] >= FRAC_PIE_MIN:
        d["mano"], d["por"], d["dudosa"] = mn, "nariz_firme", False
        """APAGADO. Ver `USA_NARIZ_SI_PIE_FLOJO`: no paso su criterio declarado porque los 2
        tiradores que anade estan los dos en entrenamiento y en el reservado anade cero."""
    elif FALLBACK_NARIZ and mn in ("A", "B"):
        d["mano"], d["por"], d["dudosa"] = mn, "nariz_dir_dudosa", True
        """El mejor criterio suelto (0,932 sobre 59) en vez de la mayoria de tres, que empata en la
        cifra y arrastra a los dos peores. Ver `FALLBACK_NARIZ`. Sigue marcado `dudosa`: cambia lo
        que se devuelve a quien la tuberia debe excluir, no quien se excluye."""
    elif FALLBACK_MAYORIA and d["mayoria_3"] is not None:
        d["mano"], d["por"], d["dudosa"] = d["mayoria_3"], "mayoria_3_dudosa", True
    else:
        d["mano"], d["por"], d["dudosa"] = a, "geometria_a", True
    return d["mano"], d


def eje_antebrazo(kp, sc, lado, mano):
    """Direccion unitaria `codo -> root` por cuadro, con su vertice y su validez.

    Se usa este eje y NO la recta del pulgar porque el pulgar esta medido y no sirve: su brazo de
    palanca es de 2,4 a 4,8 px, salta una mediana de 11-20 grados entre cuadros y pasa de 40 grados
    en el 10-26 % de los cuadros. El antebrazo mide 49-94 px y salta 0,9-2,1 grados.
    """
    M = MANOS[mano]
    k, s = kp[:, lado], sc[:, lado]
    v = k[:, M["root"]] - k[:, M["codo"]]
    n = np.linalg.norm(v, axis=1, keepdims=True)
    u = v / np.maximum(n, 1e-9)
    ok = bueno(s, [M["codo"], M["root"]]) & (n[:, 0] > 1.0)
    return k[:, M["root"]].astype("float64"), u.astype("float64"), ok, n[:, 0]


def angulo(u):
    return np.degrees(np.arctan2(u[:, 1], u[:, 0]))


def dif_ang(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def peso_sector(P, vert, u, L, apert):
    """Peso del cono en los puntos P (N,2). 0 fuera. `1 - angulo/apertura` dentro.

    Es una FORMA DECLARADA, no una probabilidad medida. Cerca del eje pesa 1, en el borde 0.
    """
    v = P - vert[None, :]
    r = np.linalg.norm(v, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cs = (v @ u) / np.maximum(r, 1e-9)
    a = np.degrees(np.arccos(np.clip(cs, -1, 1)))
    w = np.where((r <= L) & (a <= apert) & (r > 1e-6), 1.0 - a / max(apert, 1e-9), 0.0)
    return w


def dist_segmentos(p1, p2, q1, q2):
    """Distancia minima entre dos segmentos en 2D. Sin trigonometria, por proyeccion acotada."""
    def pt_seg(p, a, b):
        d = b - a
        dd = float(d @ d)
        t = 0.0 if dd <= 0 else float(np.clip((p - a) @ d / dd, 0, 1))
        return float(np.linalg.norm(p - (a + t * d)))
    return min(pt_seg(p1, q1, q2), pt_seg(p2, q1, q2), pt_seg(q1, p1, p2), pt_seg(q2, p1, p2))


def cruce(vert1, u1, L1, ap1, vert2, u2, L2, ap2, paso=None):
    """Interseccion de los dos conos, rasterizada sobre una rejilla comun.

    Devuelve `area_rel` (area de la interseccion sobre el area del cono menor), `peso_max` (maximo
    del producto de los dos pesos, que vale 1 solo si un punto esta en los DOS ejes a la vez) y el
    punto donde se alcanza. Rasterizar y no resolver analiticamente es a proposito: el area de la
    interseccion de dos sectores tiene demasiados casos y un error de caso se veria como senal.
    """
    lo = np.array([min(vert1[0] - L1, vert2[0] - L2), min(vert1[1] - L1, vert2[1] - L2)])
    hi = np.array([max(vert1[0] + L1, vert2[0] + L2), max(vert1[1] + L1, vert2[1] + L2)])
    if paso is None:
        paso = max(1.5, max(hi - lo) / 160.0)
    xs = np.arange(lo[0], hi[0] + paso, paso)
    ys = np.arange(lo[1], hi[1] + paso, paso)
    GX, GY = np.meshgrid(xs, ys)
    P = np.stack([GX.ravel(), GY.ravel()], axis=1)
    w1 = peso_sector(P, vert1, u1, L1, ap1)
    w2 = peso_sector(P, vert2, u2, L2, ap2)
    m1, m2 = w1 > 0, w2 > 0
    n1, n2 = int(m1.sum()), int(m2.sum())
    inter = m1 & m2
    ni = int(inter.sum())
    prod = w1 * w2
    j = int(np.argmax(prod))
    return {"area_rel": ni / max(min(n1, n2), 1), "n_inter": ni, "n1": n1, "n2": n2,
            "peso_max": float(prod[j]), "px": float(P[j, 0]) if prod[j] > 0 else float("nan"),
            "py": float(P[j, 1]) if prod[j] > 0 else float("nan"), "paso": float(paso)}


def mano_pie(kp, sc, lado, org=None, mask=None):
    """EL CRITERIO ADOPTADO: en guardia el pie adelantado es el del MISMO LADO que el brazo armado.

    Es un hecho del deporte, no una correlacion de estos datos, y lo valioso es que es una
    correspondencia DENTRO DEL CUERPO: no mira al rival ni una vez, asi que no se rompe en los
    tiradores que llevan el arma en la mano del lado opuesto al rival, que son los que hunden a
    «la mas cerca del rival».

    COMO SE IDENTIFICA EL PIE DE DELANTE SIN USAR AL RIVAL: el eje de cada pie es
    `talon -> punto medio de los dos dedos`, y en guardia el pie de delante apunta a lo largo de la
    pista mientras el de atras queda casi perpendicular. Gana el pie cuyo eje esta mas alineado con
    la HORIZONTAL de la imagen, que es la direccion de la pista en este corpus. Se usa el punto medio
    de los dos dedos y no uno concreto porque la prueba T1 distingue el talon pero no cual de 15/16
    es el dedo gordo.

    Se decide UNA VEZ POR CLIP por voto mayoritario, porque un tirador no cambia de mano en medio de
    un asalto, y se devuelve `frac_pie`, la fraccion de cuadros que votan a la ganadora. **Esa
    fraccion es lo que hace util al criterio**: con `frac_pie >= 0,85` no fallo ninguno de 13
    tiradores, y por debajo acerto 10 de 14.

    LIMITE DECLARADO: la correspondencia es una propiedad de la GUARDIA. En los tiradores que estan
    parados en su linea o andando entre acciones no aplica, y ahi `frac_pie` baja: los tres tiradores
    de la muestra que ningun criterio acierta tienen 0,73, 0,77 y 0,72. Tambien depende de que la
    pista sea horizontal en la imagen: la variante que usa el eje de las caderas como referencia, que
    seria totalmente intrinseca, sale PEOR (0,778 contra 0,852), asi que apoyarse en la horizontal
    ayuda y con una camara girada esto se degradaria. No hay ningun clip asi en la muestra: eso esta
    supuesto, no medido.
    """
    k, s = kp[:, lado], sc[:, lado]
    ok = np.ones(len(k), bool) if mask is None else mask.astype(bool).copy()
    if org is not None:
        ok &= (org[:, lado] == 0)
    for M in MANOS.values():
        ok &= bueno(s, [M["root"], M["codo"]])
        ok &= np.isfinite(k[:, M["root"]]).all(axis=1)
    """Se exigen tambien LAS DOS MANOS legibles, aunque el criterio mire los pies, por dos razones:
    la pregunta es cual de las dos manos es, asi que media pregunta no vale; y el vertice del cono es
    la raiz de la mano, asi que un cuadro sin mano no se puede usar de todas formas. Sin esta
    condicion `frac_pie` sale distinta de la que se midio en `mano_guante.careo`."""
    ejes = {}
    for nom, P in PIES.items():
        med = 0.5 * (k[:, P["dedos"][0]] + k[:, P["dedos"][1]])
        v = med - k[:, P["talon"]]
        n = np.linalg.norm(v, axis=1)
        ejes[nom] = np.abs(v[:, 0]) / np.maximum(n, 1e-9)
        ok &= bueno(s, [P["talon"], P["dedos"][0], P["dedos"][1]])
        ok &= np.isfinite(v).all(axis=1) & (n > 2.0)
    ok &= np.isfinite(ejes["A"]) & np.isfinite(ejes["B"]) & (ejes["A"] != ejes["B"])
    na = int((ejes["A"][ok] > ejes["B"][ok]).sum())
    nb = int(ok.sum()) - na
    if na + nb < 10:
        return None, {"pie": None, "frac_pie": float("nan"), "n_pie": int(na + nb)}
    mano = "A" if na >= nb else "B"
    return mano, {"pie": mano, "frac_pie": max(na, nb) / float(na + nb), "n_pie": na + nb}


USA_ACCION_SI_PIE_FLOJO = True
"""ADOPTADO el 2026-09-11 tras pasar la prueba en el conjunto RESERVADO, que no vio ninguna regla.

Cuando el pie no es firme y el consumidor pasa `accion`, decide el voto del criterio (a) restringido
a los cuadros que estan **encarados Y dentro de una accion**. Criterio de aceptacion declarado ANTES
de anotar y de medir los 20 reservados (punto 18 de `mano_guante.py`), y cumplido:

    (i)   acierto >= 0,90 en el reservado          ->  **1,000**
    (ii)  cobertura >= 0,50 de los 20              ->  **19 de 19**
    (iii) mas cobertura que el pie firme solo       ->  **12 -> 19**, y 1,000 en los dos

    y las DOS VIAS de `accion` se comportan igual: con homografia 8 de 8, y por altos de caja
    (los clips SIN homografia) 11 de 11. Eso era lo que habia que comprobar antes de mezclarlas.

EN ENTRENAMIENTO da 0,966 sobre 29, con **un solo fallo**: `PowerfulAllLarva` lado 1. Y el fallo
tiene causa conocida y no es de la regla: en ese clip **el rival del lado 1 es EL ARBITRO**, y la
puerta `encarado` compara la nariz contra el centro de la caja del rival. O sea que el unico error de
la regla adoptada esta en el clip donde la pareja entregada incluye a un no-tirador. Si el arbitro se
detectara, ese error desapareceria.

LA DEPENDENCIA NO SE ACEPTA, y por eso `accion` es un ARGUMENTO: `cono_lib` **no lee `eventos.csv`**.
Ese fichero es de otro agente y ha cambiado seis veces en un dia. El consumidor pasa la mascara, o
mejor, se precalcula a un array dentro del NPZ de poses, que es donde vive el resto de la procedencia
(`origen`, `hueco`, `mask`). `mano_guante.mascara_accion` la construye desde el CSV para quien la
necesite hoy."""


def mascara_encarado(kp, sc, B, W, H, lado):
    """Cuadros en los que el tirador MIRA hacia el rival, por el signo de la nariz.

    `sign(x_nariz - x_medio_hombros) == sign(x_rival - x_medio_hombros)`. Es la unica de las seis
    puertas probadas que separa de verdad por cuadro: 0,879 dentro contra 0,629 fuera, +0,250 sobre
    3.479 cuadros. Las de magnitud (alcance, brazo extendido, anchura de hombros) no separan y dos van
    al reves; las cifras estan en el punto 10 de `mano_guante.py`.
    """
    k, s = kp[:, lado], sc[:, lado]
    ok = bueno(s, [0, HOMBROS[0], HOMBROS[1]])
    xh = 0.5 * (k[:, HOMBROS[0], 0] + k[:, HOMBROS[1], 0])
    xn = k[:, 0, 0]
    xr = 0.5 * (B[:, 1 - lado, 0] + B[:, 1 - lado, 2]) * W
    ok &= np.isfinite(xh) & np.isfinite(xn) & np.isfinite(xr)
    return ok & (np.sign(xn - xh) == np.sign(xr - xh)) & (np.sign(xn - xh) != 0)


def mano_nariz(kp, sc, lado, org=None, mask=None):
    """El criterio (a) con el rival cambiado por la ORIENTACION DEL PROPIO TIRADOR.

    Direccion de frente `dir = sign(x_nariz - x_medio_hombros)` y gana la mano con `dir * x_raiz`
    mayor. No mira al rival. Medido: **0,895 sobre 38 tiradores**, contra 0,763 del criterio (a).

    NO es lo mismo que `mano_hombro`, aunque las dos usen la nariz: una proyecta las raices de mano
    y la otra los hombros, y **dan distinta respuesta en 3 de 44 tiradores**, asi que no se pueden
    sustituir la una por la otra. Las dos aciertan 0,895, que es lo que hizo ver que **lo que decide
    es la DIRECCION y no el punto que se proyecta**.
    """
    k, s = kp[:, lado], sc[:, lado]
    ok = np.ones(len(k), bool) if mask is None else mask.astype(bool).copy()
    if org is not None:
        ok &= (org[:, lado] == 0)
    for M in MANOS.values():
        ok &= bueno(s, [M["root"], M["codo"]])
        ok &= np.isfinite(k[:, M["root"]]).all(axis=1)
    ok &= bueno(s, [0, HOMBROS[0], HOMBROS[1]])
    xh = 0.5 * (k[:, HOMBROS[0], 0] + k[:, HOMBROS[1], 0])
    dirn = np.sign(k[:, 0, 0] - xh)
    ok &= np.isfinite(dirn) & (dirn != 0) & np.isfinite(xh)
    pa = dirn * k[:, MANOS["A"]["root"], 0]
    pb = dirn * k[:, MANOS["B"]["root"], 0]
    ok &= np.isfinite(pa) & np.isfinite(pb) & (pa != pb)
    na = int((pa[ok] > pb[ok]).sum())
    nb = int(ok.sum()) - na
    if na + nb < 10:
        return None, {"nariz": None, "frac_nariz": float("nan"), "n_nariz": int(na + nb)}
    mano = "A" if na >= nb else "B"
    return mano, {"nariz": mano, "frac_nariz": max(na, nb) / float(na + nb), "n_nariz": na + nb}


def mano_hombro(kp, sc, lado, org=None, mask=None):
    """IDEA DEL USUARIO: «el brazo armado por regla siempre va POR DELANTE del no armado, tipo se
    mide por el hombro». Es cierto y es de esgrima: la guardia es de perfil con el hombro del arma
    adelantado.

    «Delante» necesita una direccion, y **no se usa el rival**: se usa la propia cara. La direccion
    de frente es `sign(x_nariz - x_medio_hombros)` y gana el hombro con mayor coordenada en esa
    direccion; la mano armada es la de su lado. Es la misma senial que `nariz_dir` pero medida sobre
    los HOMBROS, que son el 18,7-21,5 % del alto de la caja, en vez de sobre las raices de mano, que
    colapsan en el mismo pixel en 2 de 34 tiradores. Y aguanta el caso que preocupaba al usuario, «se
    pasa el brazo por detras», porque la mano cruza el cuerpo y el hombro no.

    MEDIDO: 0,889 sobre 27 tiradores suelto, y **1,000 sobre 11** cuando su propia `frac_hombro >=
    0,85`. La frecuencia del cruce de brazo, que es lo que justificaba mirar el hombro, resulta
    PEQUENIA: `frac_cruza_brazo` tiene p50 0,04 y p90 0,14, y en los siete tiradores que el criterio
    (a) falla sube solo de 0,03 a 0,07. O sea que el cruce existe pero no es la causa de los fallos.
    """
    k, s = kp[:, lado], sc[:, lado]
    ok = np.ones(len(k), bool) if mask is None else mask.astype(bool).copy()
    if org is not None:
        ok &= (org[:, lado] == 0)
    for M in MANOS.values():
        ok &= bueno(s, [M["root"], M["codo"]])
        ok &= np.isfinite(k[:, M["root"]]).all(axis=1)
    ok &= bueno(s, [0, HOMBROS[0], HOMBROS[1]])
    xh = 0.5 * (k[:, HOMBROS[0], 0] + k[:, HOMBROS[1], 0])
    dirn = np.sign(k[:, 0, 0] - xh)
    ok &= np.isfinite(dirn) & (dirn != 0) & np.isfinite(xh)
    pa, pb = dirn * k[:, HOMBROS[1], 0], dirn * k[:, HOMBROS[0], 0]
    ok &= np.isfinite(pa) & np.isfinite(pb) & (pa != pb)
    na = int((pa[ok] > pb[ok]).sum())
    nb = int(ok.sum()) - na
    if na + nb < 10:
        return None, {"hombro": None, "frac_hombro": float("nan"), "n_hombro": int(na + nb)}
    mano = "A" if na >= nb else "B"
    return mano, {"hombro": mano, "frac_hombro": max(na, nb) / float(na + nb),
                  "n_hombro": na + nb}


def mano_armada_voto(kp, sc, B, W, H, lado, org=None, mask=None, vota=None):
    """DOS VARIANTES DE «UNA DECISION POR CLIP», declaradas antes de medirse (2026-09-11).

    `mano_armada` decide con la MEDIANA del clip de la x de cada mano. La mediana es un resumen
    fuerte, pero mezcla cuadros donde la pose es buena con cuadros donde no lo es, y no dice nunca
    lo apretada que fue la decision. Un tirador no cambia de mano en medio de un asalto, asi que la
    forma honesta de usar los cuadros es **votar y publicar la fraccion de votos**.

      `a_voto`    por cuadro gana la mano con |x_mano - x_rival| menor, con `x_rival` el centro de
                  la caja del rival EN ESE CUADRO y no su mediana del clip; por clip, mayoria.
      `a_dist2d`  igual pero con distancia EUCLIDEA al punto del rival (centro x, 30 % del alto),
                  no solo la x. Motivo declarado: en pleno fondo las dos manos quedan casi a la
                  misma x y lo que las separa es la altura.

    Cuadros que votan: `mask` y `origen == 0` si se pasan (una pose interpolada no puede decidir que
    mano es la armada, porque su raiz no se midio), y confianza de las dos raices y los dos codos.

    EL NUMERO QUE DECIDE, declarado antes de verlo: acierto por tirador contra la anotacion a ojo de
    `02_validacion/muestras/mano_dominante/verdad_mano.csv` mayor que el **0,724 sobre 29 tiradores**
    que saca la version por medianas. Si no lo bate, se queda la de medianas y se dice.

    Devuelve `(mano_a_voto, dict)` con `frac_voto` de cada variante, que es lo que permite decir si
    una decision fue solida (0,95) o floja (0,55).

    `vota` es una mascara booleana (T,) opcional de cuadros AUTORIZADOS a votar, y es el enganche
    para las puertas de `mano_guante.mascaras_puerta`: la causa medida de las fallas del criterio (a)
    es que votan tambien los cuadros de entre accion, donde el tirador no esta encarado y lleva el
    arma atras. Aqui no se decide cual es la puerta buena: se deja el enganche y se mide fuera.
    """
    k, s = kp[:, lado], sc[:, lado]
    rival = 1 - lado
    ok = np.ones(len(k), bool) if mask is None else mask.astype(bool).copy()
    if vota is not None:
        ok &= vota.astype(bool)
    if org is not None:
        ok &= (org[:, lado] == 0)
    for M in MANOS.values():
        ok &= bueno(s, [M["root"], M["codo"]])
        ok &= np.isfinite(k[:, M["root"]]).all(axis=1)
        """`bueno` solo mira la confianza, no la coordenada, y hay puntos con confianza alta y
        coordenada NaN (trampa ya documentada en el modulo). Sin este filtro un NaN vota siempre a
        la misma mano, porque `nan < x` es False."""
    xr = 0.5 * (B[:, rival, 0] + B[:, rival, 2]) * W
    yr = (B[:, rival, 1] + 0.30 * (B[:, rival, 3] - B[:, rival, 1])) * H
    ok &= np.isfinite(xr) & np.isfinite(yr)
    d = {"n_vota": int(ok.sum())}
    if d["n_vota"] < 10:
        return None, d
    out = {}
    for nom, var in (("a_voto", "x"), ("a_dist2d", "2d")):
        pun = {}
        for m, M in MANOS.items():
            dx = k[ok, M["root"], 0] - xr[ok]
            if var == "x":
                pun[m] = np.abs(dx)
            else:
                dy = k[ok, M["root"], 1] - yr[ok]
                pun[m] = np.hypot(dx, dy)
        gana = np.where(pun["A"] < pun["B"], 1, 0)
        na, nb = int(gana.sum()), int((1 - gana).sum())
        out[nom] = "A" if na >= nb else "B"
        d[nom] = out[nom]
        d["frac_" + nom] = max(na, nb) / float(max(na + nb, 1))
        d["nA_" + nom], d["nB_" + nom] = na, nb
    d["concuerdan_variantes"] = bool(out["a_voto"] == out["a_dist2d"])
    return out["a_voto"], d

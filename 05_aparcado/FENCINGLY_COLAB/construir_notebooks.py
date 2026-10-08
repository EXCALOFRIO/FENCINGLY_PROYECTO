"""Deriva los dos cuadernos del PAQUETE UNICO de los canonicos, sin tocarlos.

    python -B construir_notebooks.py <dir_canonico> <dir_salida>

`dir_canonico` es `<repo>/colab/notebooks/`. Los canonicos se leen y no se escriben nunca.

Un paquete: un cuaderno de extraccion, uno de entrenamiento, y un documento que los explica.
Nada de parches sueltos: lo que hace falta va dentro de los cuadernos, y lo que no cabe va a la
seccion de vias documentadas y no tomadas del documento, con su cifra.

Que cambia respecto del canonico, y nada mas:

  01_Extraccion_y_Evidencia
      + md   PROTOCOLO       la regla metodologica, antes de todo
      ~ cod  CONFIG          los cuatro mandos: almacen, rodaja, T_max, reloj
      + cod  PREFLIGHT       entorno (antes que la huella) + huella + recorte + rodaja cabe + costes
      + md   PORQUE          las dos decisiones, la aritmetica de la rodaja y los cuatro precios
      + cod  PLAN_VENTANA    cuadros/bytes en ESTE runtime y % de clips que rechazaria
      ~ cod  EXTRACCION      NativeProcessor recibe pre_s/post_s/use_clock; run_config los graba
      ~ cod  STORE           PF.construir_store_con_rodaja: NPZ ya recortados a disco
      + cod  VERIFICACION    columna del reloj + indice con el lector real + QC + ALINEACION

  02_Entrenamiento_y_Brazos
      + md   PROTOCOLO       la misma regla
      + cod  PREFLIGHT       entorno + determinismo (backend `math`) + huella
      + cod  BRAZOS          los tres brazos con el signo declarado ANTES, y el veredicto pareado

  03_Shared_Web_App          copia sin cambios (opcional, no participa en la medida)

El generador es idempotente: si una celda marcada ya existe, la sustituye en su sitio en vez de
duplicarla. Cada celda insertada o modificada lleva la marca
`# @@FENCINGLY_DURACION_COMPLETA:<clave>` para poder reconocerla.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

MARCA = "@@FENCINGLY_DURACION_COMPLETA"


def celda_codigo(clave: str, cuerpo: str) -> dict:
    fuente = f"# {MARCA}:{clave}\n" + cuerpo.rstrip() + "\n"
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": fuente.splitlines(keepends=True)}


def celda_markdown(clave: str, cuerpo: str) -> dict:
    fuente = f"<!-- {MARCA}:{clave} -->\n" + cuerpo.rstrip() + "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": fuente.splitlines(keepends=True)}


def indice_de(celdas, clave):
    for i, c in enumerate(celdas):
        if f"{MARCA}:{clave}" in "".join(c.get("source", [])):
            return i
    return None


def poner(celdas, clave, celda, *, tras=None, al_principio=False):
    """Sustituye la celda `clave` si ya existe; si no, la inserta."""
    i = indice_de(celdas, clave)
    if i is not None:
        celdas[i] = celda
        return i
    if al_principio:
        celdas.insert(0, celda)
        return 0
    if tras is None:
        celdas.append(celda)
        return len(celdas) - 1
    celdas.insert(tras + 1, celda)
    return tras + 1


# --------------------------------------------------------------------------------------------------
# Textos
# --------------------------------------------------------------------------------------------------

MD_PROTOCOLO = """\
# 00 - REGLA DE PROTOCOLO. Leer antes de ejecutar y antes de interpretar

**El numero que salga de esta corrida es una LINEA BASE NUEVA. No se puede comparar con el 73,88 %,
ni con el 70,04 %, ni con ninguna cifra anterior del proyecto.**

Con duracion completa cambian `pre_s_eff`, `n_frames_window`, cada rasgo derivado de la ventana y la
huella del productor. Es otra cadena de medida. Restar una cifra de un protocolo a una cifra de otro
protocolo es un **error de protocolo**, no una comparacion conservadora. El informe del proyecto ya
lleva esa disciplina escrita ("nowhere in this report is a figure from one protocol subtracted from a
figure of another") y aqui aplica igual.

Lo que **si** se puede comparar: duracion-completa contra duracion-completa, con y sin el cambio que
se este probando, **pareado por clip**.

## El veredicto se reporta en dos poblaciones, siempre

El atajo "gana el que enciende segundo" acierta **63,51 %** por si solo, y el modelo lo predice en el
**72,71 %** de los clips: lo sobreusa. Eso parte el corpus:

| poblacion | n | acierto |
|---|---:|---:|
| gano el que encendio **segundo** | 1.671 | **83,00 %** |
| gano el que encendio **primero** | 960 | **45,21 %** |
| brecha | | **+37,80 pp** |

* llevar los 960 de 45,21 % a **60 %** vale **+5,40 pp** globales; a **75 %**, **+10,95 pp**
* degradar 5 pp los 1.671 cuesta **-3,18 pp**

**El veredicto es la media global pareada.** El subgrupo describe donde cayo la ganancia; nunca es la
poblacion de evaluacion, nunca un objetivo de optimizacion y nunca una puerta de enrutado. Evaluar
dentro de un subgrupo definido por el resultado hace que la base caiga bajo el azar: alli un bloque
sin informacion compro **+3,08 pp**.

## Limitacion honesta, y va en el informe

Con la potencia de este corpus **un efecto real de +1,6 pp es indetectable**, y hacen falta **12 a 20
campeonatos mas** para tener el 80 % de potencia sobre ese tamano de efecto. **Una mejora real puede
no ser demostrable aunque exista.** Este cuaderno no promete lo que no puede medir.

## Varas de medir, para no creerse un delta

* Cambiar **solo** el backend de atencion, misma receta, misma semilla y mismo interprete, da
  **+0,36 pp con 327 de 3.047 decisiones distintas**. Cualquier mejora de red por debajo de eso es
  indistinguible de recompilar. (Por eso la celda de preflight fuerza el backend `math`.)
* El nulo se elige **por el tipo de rasgo**. El desplazamiento circular **preserva** los rasgos de
  nivel (una columna quedo identica al 100 %), **infla** los de derivada (hasta 7,79x) y es **inerte**
  sobre representaciones agrupadas (coseno 0,9980). Para nivel, nulo de **signo de lado aleatorio**.
* **Dos convenios de promediado de semillas no son la misma cantidad.** El banco de ablaciones
  promedia el *acierto por clip*; el artefacto promedia las *probabilidades* y umbraliza una vez. Un
  efecto de +1,03 con el primero fue **+0,80 con intervalo cruzando el cero** con el segundo. **Usa
  el del artefacto.**
"""

MD_PORQUE = """\
## Almacenar y entrenar son dos decisiones distintas

**La tabla de abajo NO es una lista de alternativas.** Guardar el clip entero y entrenar sobre una
rodaja son dos decisiones separadas, y tomarlas por separado quita los cuatro precios de la duracion
completa sin renunciar a nada:

| | que cuesta | que compra |
|---|---|---|
| **`VENTANA_ALMACEN = "completa"`** | tiempo de GPU y un runtime High-RAM | la opcionalidad: el clip entero para lo que venga, sin volver a descargar 93 GB |
| **`RODAJA_ENTRENAMIENTO = "recomendada"`** | nada mas | la alineacion, los 488 clips, `T_max` y las etiquetas auxiliares |

Y lo que la duracion completa compra de verdad —re-ventanear las **256 columnas agregadas**, que es
donde esta el techo— se compra igual, porque las agregadas se recalculan sobre la rodaja que se
elija.

**Y la rodaja no cuesta ningun cambio de fuente**, pero NO por envolver `dataset.load_sample`:
eso se probo y ENVENENA EL STORE (`build_store` congela en `store.json` el hash de los bindings de
`dataset.py`, y el Store queda inabrible para siempre). La via correcta es `PF.rodaja_a_disco`:
escribir los NPZ recortados en un directorio nuevo y construir desde ahi, con `load_sample` intacto
de principio a fin. Cuesta ~0,6 GiB de disco para el corpus de florete. Detalle en
la fuente, pero **comprobado que ni el 01 ni el 02 canonicos mencionan `portable_contract`,
`load_bundle` ni `save_bundle` ni una vez**, y el Store se construye en el 01 mientras los bundles se
tocan en el 02, en otra sesion. Sobre un NPZ de ventana desplegada el recorte es un **no-op
verificado** (T = 129 antes y despues).

Lo unico que la rodaja **no** arregla son los 488 clips del QC, porque el QC se congela en la
extraccion: eso queda como via documentada y no tomada, con su cifra, en el documento.

### La aritmetica que decide la rodaja, y hay que verla antes de elegir

Una rodaja `(pre_s, post_s)` a 30 fps ocupa `round(pre_s*30) + round(post_s*30) + 1` cuadros, y el
Store solo guarda `T_max`, **alineados por el final**:

| rodaja | cuadros | `T_max` necesario | cabe en 135 | guardia `[t−4,0:t−3,5]` | 2ª luz |
|---|---:|---:|:---:|:---:|---:|
| `"historica"` 4.0/0.3 | 130 | 135 | **sí** | sí | 99,05 % |
| **`"recomendada"` 4.0/0.6** | **139** | **140** | no | **sí** | **100,00 %** |
| `"ancha"` 6.0/1.0 | 211 | 211 | no | sí | 100,00 % |

**El detalle que hay que ver:** 6,0 + 1,0 = 7,0 s son **211 cuadros**, y `T_MAX = 135` son 4,467 s.
Con `T_max` sin subir, el Store se queda los ultimos 135 de esos 211, o sea **`[t−3,47 : t+1,0]`**, y
la ventana de guardia **se queda fuera igual**: justo lo que la rodaja venia a proteger. **La rodaja
no hace por si sola que `T_MAX` deje de estorbar; hay que subir `T_max` a la vez.**
`PF.plan_rodaja(...)` lo comprueba y levanta.

Subir `T_max` toca `store.T_MAX` y el validador de `TrainingRecipe` (`8 <= T_max <= 135`). Los dos
estan en **ninguna** de las cuatro listas, y `feature_contract.MAX_TIMESTEPS = 256` ya lo permite,
asi que **no se invalida ningun bundle portable**. Lo que si cambia es la entrada de la red, y eso
exige reentrenar y reexportar — que es lo que se va a hacer igual.

**Por que `"recomendada"` y no `"ancha"`:** 4.0/0.6 es la rodaja **minima** que cierra la unica
brecha **medida** —`|dt_ms|` maximo 520 ms, asi que `post_s = 0,6` captura la segunda luz al 100 %—
y conserva la ventana de guardia entera. `"ancha"` tambien funciona, pero **cambia lo que ve la red**
(7 s en vez de 4,3) y **ese cambio no esta medido**: `pre_s = 4,0` se eligio porque 2,5 s recortaba
el inicio de la accion final en el 24 % de los clips, y nadie ha comparado 6,0 contra 4,0 sobre la
red. La atencion es O(T²), asi que 211 cuadros son ~2,5× el coste de atencion de 135. Si se prueba,
que sea como un brazo mas, pareado contra `"recomendada"`, y con el suelo de +0,4 pp del nulo
neuronal delante.

### El reloj es un mando aparte, y es la ganancia mas barata que hay

`USAR_RELOJ = False` lleva la ventana de guardia de **72,78 % a 91,42 %** al **mismo coste exacto**
de decodificacion (129 cuadros, 680 MiB, +0,03 pp de clips bloqueados). No cuesta nada y no lo
compra ninguna otra palanca de la tabla.

**AVISO:** hay trabajo en curso para **conectar** la tabla del reloj. Las dos decisiones interactuan,
asi que hay que medirlas **juntas, en un factorial 2×2**, no una encima de la otra. Si el otro cambio
ya esta dentro cuando se ejecute esto, hay que declarar las dos configuraciones junto al resultado.

---

## Por que duracion completa, y cual es su precio medido

Las seis razones de la decision, y al lado lo que cada una cuesta o gana **medido**, no supuesto.
Todas las cifras salen de reevaluar los 15.451 artefactos de QC ya escritos del corpus, sin
re-extraer nada.

| razon | lo que se gana | medido |
|---|---|---|
| 1. `POST_S = 0,3` deja la segunda luz fuera en el 3,12 % de los clips | la captura entera | `\\|dt_ms\\|` maximo **520 ms** sobre los 3.331 clips de dos luces (p50 80, p95 267, p99 360; 100 % < 1.000 ms). **`post_s = 0,6` ya la captura entera.** La duracion completa no anade nada aqui |
| 2. el bloque de 256 columnas agregadas esta congelado y lleva el atajo casi entero | poder re-ventanearlo | **esto si lo exige la duracion completa.** Re-ventanear solo el bloque temporal no movio nada (media +0,30 real contra +0,27 de su nulo, exceso +0,03) |
| 3. la ventana de guardia `[t-4,0 : t-3,5]` no existe en el 22,88 % de los clips | tenerla siempre | pasa de **72,78 % a 91,42 % solo apagando el reloj**, al mismo coste exacto. Y su techo es **97,04 %, no 100 %**: en ~3 % de los clips el tocado ocurre a menos de 4,0 s del comienzo. La duracion completa **no puede inventar cuadros que nadie filmo** |
| 4. el transformador de video fallo por submuestreo (16 cuadros / 1,2 s = 75 ms de paso, y el suceso decisivo es un desfase de 10-30 ms) | muestreo denso cerca del tocado | lo exige la duracion completa, pero es una via **cerrada** por medida (VideoMAE no llego al suelo de 0,8 pp y predice la sede al 97 %) |
| 5. la tabla del reloj y la deteccion de repeticiones | cuadros anteriores a la frase | `pre_s = 6,0` los da en el 97,04 % |
| 6. la estatura gana cuadros de guardia | su escala | igual que 3 y 5 |

### El precio, en la misma unidad

| candidato | cuadros mediana | MiB p95 | rechazados T4 | rechazados High-RAM | QC bloq | 2a luz | guardia |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4.0/0.3 + reloj (**hoy**) | 129 | 1.554 | 0,00 % | 0,00 % | 5,53 % | 99,05 % | 72,78 % |
| 4.0/0.3 sin reloj | 129 | 1.554 | 0,00 % | 0,00 % | 5,56 % | 99,05 % | **91,42 %** |
| **6.0/1.0 sin reloj** | **210** | 2.515 | 0,00 % | 0,00 % | 6,52 % | **99,99 %** | **97,04 %** |
| 8.0/3.0 sin reloj | 294 | 3.939 | 4,36 % | 0,49 % | 8,30 % | 99,99 % | 97,04 % |
| **DURACION COMPLETA** | **318** | 5.233 | **11,31 %** | **0,50 %** | **8,69 %** | 99,99 % | 97,04 % |

**Los cuatro precios de la duracion completa. Los tres ultimos los quita la rodaja; el primero no.**

0. **ROMPE LA ALINEACION, y es el peor.** `build_store` alinea por el FINAL del array y
   `feature_contract` lo declara (`alignment: right`, `truncation: tail`). Con un NPZ de duracion
   completa el array termina al final del **CLIP**, y el clip sigue despues del tocado: la cola tras
   el tocado tiene **mediana 2,33 s** (p05 0,93, media 2,62, p95 5,23, max 14,57) sobre 15.453 clips
   [medido], y supera 1,0 s en el **93,70 %**. Asi que el Store guardaria `[t−2,17 : t+2,33]` en vez
   de `[t−4,2 : t+0,3]`: **el ultimo cuadro deja de ser el tocado** y la **ventana de guardia
   desaparece del Store en ~94 % de los clips** — justo lo que la duracion completa venia a
   garantizar. No salta ningun error. **Esto es una regresion, no un empate**, y es la razon de que
   la rodaja sea obligatoria y no una optimizacion. `PF.alineacion_del_store` lo mide sobre el Store
   construido y **levanta**.

1. **La duracion completa exige un runtime High-RAM.** Con el tope de `max_decoded_bytes` que
   `resource_plan` calcula (`min(6,0 GiB, RAM_libre x 0,35)`), un T4 estandar rechaza el **11,31 %**
   de los clips y el rechazo esta **correlacionado con la resolucion**, o sea que sesga la seleccion:
   es exactamente el fallo que el runbook del 2026-09-06 elimino y que la duracion completa reabre.
   En High-RAM el rechazo baja al **0,50 %**. La frase del runbook *"You no longer need High-RAM to
   avoid resolution bias"* **deja de ser cierta con duracion completa**.
2. **La duracion completa cuesta 488 clips de entrenamiento (3,16 %).** `clip_quality` marca
   `cut_in_window` y `replay_in_window` como **bloqueantes** (peso 0,0) y `build_current_store`
   descarta toda fila con `qc_blocking` no vacio. Al ensanchar la ventana entran cortes de camara que
   hoy quedan fuera: 828 clips (5,36 %) tienen al menos un corte. Bloqueados 853 (5,52 %) -> **1.341
   (8,68 %)**; entran al Store 14.598 -> **14.110**. Las repeticiones **no** son el problema (3 clips).
   **Y la rodaja NO lo arregla sola**, porque el QC se calcula en la EXTRACCION sobre la ventana
   GUARDADA y queda congelado en `analysis/qc/<clip>.json`; elegir una rodaja despues no lo cambia.
   Se recupera acotando esas dos banderas a la ventana de analisis en `clip_quality._flatten_vm`
   (via documentada y no tomada, con su cifra, en el documento). `clip_quality.py` solo esta en
   `FINGERPRINT_SOURCES`, asi que el coste es re-cachear el panel de coach, no re-materializar los
   23.091 artefactos.
3. **Las etiquetas auxiliares se apagan en silencio.** `store_truncated = prepared_n_frames > T_max`
   pasa a True en casi todos los clips, y `attach_aux` (`store.py:244`) y `Store` (`store.py:417`)
   exigen `not store_truncated` para admitir una etiqueta: quedan a codigo -1, mask 0, conf 0 **sin
   aviso**. Hoy es 0 de 14.437 muestras [medido]. **La rodaja lo arregla**, porque una rodaja que
   cabe en `T_max` no trunca. `PF.alineacion_del_store` lo cuenta.
4. **La red no ve mas contexto por la duracion completa.** El Store guarda `T_max` cuadros alineados
   por el final, asi que lo que ve la red lo decide la **rodaja**, no el almacen. Con
   `"recomendada"` son 139 cuadros (4,63 s) contra los 130 (4,33 s) de hoy: casi lo mismo, y a
   proposito. Subir `feature_contract.MAX_TIMESTEPS` (256) **no hace falta** para ninguna de las tres
   rodajas, y conviene no tocarlo: esta en `portable_contract.SEMANTIC_SCOPE` con alcance de modulo
   completo, asi que cambiarlo mueve `configuration_sha256` e **invalida todos los bundles portables**
   (sintoma: `ContractError: Portable: semantica AST/configuracion no conforme`, la app corre sin
   modelo y `judge` responde `model_unavailable`).

   Traducido: **la duracion completa es una inversion en el camino de MEDIDAS (bloque agregado,
   guardia, estatura, reloj) y en no tener que volver a descargar. No es una mejora del modelo
   secuencial, y no hay que esperar que el numero suba por ella.**
"""

COD_PREFLIGHT_01 = """\
# Comprobaciones que tienen que fallar RUIDOSAMENTE, antes de gastar horas de GPU.
# preflight.py viaja junto a este cuaderno; si no esta, subelo o clona el paquete.
import sys, os
from pathlib import Path
for _cand in (Path.cwd(), Path.cwd() / "paquete", Path("/content"), Path("/content/paquete")):
    if (_cand / "preflight.py").is_file() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
import preflight as PF

PF.protocolo()

# 1. Entorno clavado. VA PRIMERO, ANTES DE LA HUELLA, y el orden no es cosmetico: `_producer()` lee
#    las versiones con `importlib.metadata.version(...)`, o sea los METADATOS EN DISCO, no el modulo
#    cargado. Tras un `pip install numpy==2.5.2` sin reiniciar el runtime, la HUELLA COINCIDE y las
#    medidas se calculan con el numpy equivocado. La comprobacion de huella daria falsa seguridad;
#    esta compara `numpy.__version__` contra los metadatos y levanta.
ENTORNO_PINES = PF.entorno(exigir=True)

# 2. La huella del productor de medidas. producer_sha256 NO es el hash de los 20 ficheros: hashea el
#    diccionario entero, que incluye runtime_versions. Un parche de numpy invalida los 23.091
#    artefactos y el fallo seria un desempate de cache SILENCIOSO.
HUELLA = PF.huella(exigir=True)

# 3. El recorte a la ventana de analisis en el camino POR CLIP. Con la ventana desplegada es inerte
#    (poses.shape[0] == n_frames_window en los 15.452 clips: no hay ni un NPZ de duracion completa).
#    Con almacen ancho pasa de inerte a NECESARIO: sin el, las medidas de guardia se calcularian
#    sobre el clip entero, con desviaciones medidas de hasta 152 grados en fore_ang_guard.
PARCHE = PF.parche_de_recorte(exigir=(VENTANA_ALMACEN != "desplegada"))

# 4. Que la rodaja de entrenamiento CABE en T_max. Si no cabe, el Store se queda los ultimos T_max
#    cuadros de la rodaja y la ventana de guardia [t-4,0 : t-3,5] se pierde EN SILENCIO. El caso
#    6.0/1.0 con T_max=135 da [t-3,47 : t+1,0]: 211 cuadros no caben en 135.
PLAN_RODAJA = PF.plan_rodaja(RODAJA_ENTRENAMIENTO, t_max=T_MAX, exigir=True)

# 5. Y que se sepa lo que cuesta cada cambio, medido y no supuesto.
PF.coste_de_las_listas()
"""

COD_CONFIG_EXTRA = """\
# ALMACENAR Y ENTRENAR SON DOS DECISIONES DISTINTAS. Estos son los mandos, y son cuatro.
#
# 1) Que se GUARDA en el NPZ. Cuesta GPU y High-RAM, y compra no volver a descargar 93 GB.
#      "desplegada" 4.0/0.3 | "amplia" 6.0/1.0 | "completa" 60/60 (el clip entero)
VENTANA_ALMACEN = os.environ.get("FENCINGLY_VENTANA_ALMACEN", "completa")
#
# 2) La RODAJA sobre la que se mide y se entrena, derivada del NPZ guardado. No cuesta nada mas.
#      "historica" 4.0/0.3 (130 cuadros, T_max 135)
#      "recomendada" 4.0/0.6 (139 cuadros, T_max 140)  <- cierra la unica brecha MEDIDA
#      "ancha" 6.0/1.0 (211 cuadros, T_max 211)        <- cambia lo que ve la red; NO medido
RODAJA_ENTRENAMIENTO = os.environ.get("FENCINGLY_RODAJA", "recomendada")
#
# 3) T_max del Store. Tiene que admitir la rodaja o la ventana de guardia se pierde en silencio.
T_MAX = int(os.environ.get("FENCINGLY_T_MAX", "140"))
#
# 4) El reloj. Apagarlo lleva la ventana de guardia de 72,78 % a 91,42 % A COSTE CERO: es la
#    ganancia mas barata de todo el barrido. AVISO: hay trabajo en curso para CONECTAR la tabla del
#    reloj, asi que las dos decisiones INTERACTUAN y hay que medirlas juntas en un 2x2, no una
#    encima de la otra. Si el otro cambio ya esta dentro, declara las dos en el informe.
USAR_RELOJ = os.environ.get("FENCINGLY_USAR_RELOJ", "0") == "1"

PRE_S, POST_S, _use_clock_por_defecto = PF.VENTANAS[VENTANA_ALMACEN]
USE_CLOCK = USAR_RELOJ
RODAJA_PRE_S, RODAJA_POST_S = PF.RODAJAS[RODAJA_ENTRENAMIENTO]
print(f"ALMACEN     {VENTANA_ALMACEN!r}: pre_s={PRE_S} post_s={POST_S} use_clock={USE_CLOCK}")
print(f"ENTRENA     {RODAJA_ENTRENAMIENTO!r}: pre_s={RODAJA_PRE_S} post_s={RODAJA_POST_S} T_max={T_MAX}")
"""

COD_RESPALDO = """\
# RESPALDO INCREMENTAL. Una corrida de horas en Colab se muere sola, asi que esto no es un adorno.
#
# Destino: repositorio PRIVADO de Hugging Face. La excepcion esta escrita en AGENTS.md linea 4,
# decidida por el usuario el 2026-09-09 y acotada: privado SOLO, artefactos de extraccion SOLO.
# Un repo PUBLICO sigue prohibido, porque los recortes son fotogramas reales de retransmision y el
# manifiesto lleva nombres de tiradores, torneos y resultados.
print(PF.RESPALDO)

# El token es una CREDENCIAL: de un secreto de Colab o de HF_TOKEN, nunca en el cuaderno. Sin el,
# esto AVISA A GRITOS y sigue: la corrida no sera reanudable entre sesiones.
RESPALDO = PF.aviso_respaldo(exigir=False)
REPO_PRIVADO = os.environ.get("FENCINGLY_HF_REPO", "")     # p.ej. "usuario/fencingly-extraccion"
RESPALDO_DIR = str(Path(WORK).parent / "_respaldo")

# El tamano de lote se DERIVA del ritmo medido, no se fija a ojo. Medido en la RTX 3060 con
# process_workers=1 y ventana completa: 17,2 s/clip en frio. Con una instantanea cada 10 min salen
# 35 clips por lote. Y se agrupa en `tar` porque la plataforma admite < 100.000 ficheros por repo y
# < 10.000 por carpeta: con 34.374 clips y varios ficheros cada uno, sin agrupar se roza el limite.
SEGUNDOS_POR_CLIP = float(os.environ.get("FENCINGLY_S_POR_CLIP", "17.2"))
CLIPS_POR_LOTE = PF.clips_por_lote(SEGUNDOS_POR_CLIP)
print(f"lote de respaldo: {CLIPS_POR_LOTE} clips ({SEGUNDOS_POR_CLIP:g} s/clip medidos)")

# Y lo que hay hecho DE VERDAD, contado por el indice y sus NPZ presentes, no por un contador:
# si el contador dice 1.200 y el indice tiene 1.180, la reanudacion miente.
ESTADO = PF.estado_reanudacion(WORK, drive_dir=RESPALDO_DIR)
PENDIENTES_RESPALDO = PF.respaldo_pendientes(WORK, RESPALDO_DIR)
print(f"pendientes de empaquetar: {len(PENDIENTES_RESPALDO)}")
"""

COD_FPS_NATIVO = """\
# HUECO RESERVADO: extraccion a CADENCIA NATIVA. El mando no esta implementado a proposito.
print(PF.FPS_NATIVO_PENDIENTE)
# FPS_NATIVO = False   # <- aqui va el mando cuando lleguen las cifras de coste en tamano y tiempo
"""

COD_VERIFICAR_RESPALDO = """\
# EJECUTAR UNA VEZ, con el token puesto. Crea el repo PRIVADO, sube un lote, lo lee, lo borra y
# confirma. Es lo que convierte "implementado" en "probado" sin que la credencial pase por otro
# sitio. El empaquetado, el troceado, el estado y la reanudacion YA estan probados en local; lo que
# esta celda anade es la subida autenticada y la comprobacion de que el repo es privado.
print(PF.CELDA_DE_VERIFICACION)
"""

COD_PLAN_VENTANA = """\
# Que va a pasar con la ventana de ALMACEN en ESTE runtime concreto: cuadros y bytes por clip, y que
# fraccion del corpus rechazaria el planificador de decodificacion. Un rechazo correlacionado con
# la resolucion SESGA la seleccion, asi que se mira antes, no clip por clip despues de horas.
if RUN_DATA:
    print(PF.BARRIDO_MEDIDO)
    PLAN_VENTANA = PF.plan_ventana(ENVIRONMENT, VENTANA_ALMACEN)
    if PLAN_VENTANA["fraccion_rechazada_estimada"] > 0.005 and VENTANA_ALMACEN == "completa":
        print()
        print("DECISION: este runtime no es High-RAM suficiente para duracion completa sin sesgo.")
        print("Opciones, por orden de preferencia:")
        print("  a) Entorno de ejecucion -> Cambiar tipo de entorno -> GPU + High-RAM, y reejecutar")
        print("  b) VENTANA_ALMACEN = 'amplia' (6.0/1.0): no rechaza nada en ningun runtime, y la")
        print("     rodaja de entrenamiento no cambia, asi que el numero sigue siendo comparable")
        print("     entre las dos: es la MISMA cadena de medida con distinto almacen")
        print("  c) subir a mano el tope de 6,0 GiB de resource_plan, sabiendo que el pico REAL")
        print("     medido es 1,08x una copia BGR (1,66 GiB para 1,54 GiB), no 2x")
    print()
    print(PF.COLA_TRAS_TOCADO)
"""

COD_VERIFICACION = """\
# Verificacion de la corrida. Nada de esto se supone: se cuenta.
if RUN_DATA:
    # 1. El indice, contado con el LECTOR REAL. AnalysisRepository.index() se salta EN SILENCIO un
    #    directorio de NPZ sin npz/index.csv (costo 689 clips invisibles), y valid_id exige
    #    [A-Za-z0-9_-], asi que los 13 clips con id no ASCII (Vicuna, Kauai'O'o, Vendeen) se caen
    #    aunque su fila este escrita. No relajes el validador: es una brecha conocida y aceptada.
    INDICE_REAL = PF.indice_real(WORK)

    # 2. El coste de QC de esta ventana, medido en ESTA corrida y no heredado del corpus.
    import json as _json
    from pathlib import Path as _Path
    _qc = sorted((_Path(WORK) / "analysis" / "qc").glob("*.json"))
    _n = _bloq = _corte = _rep = 0
    _motivos = {}
    for _p in _qc:
        try:
            _q = _json.loads(_p.read_text(encoding="utf-8"))
        except Exception:
            continue
        _n += 1
        _b = list(_q.get("blocking") or [])
        _bloq += int(bool(_b))
        _corte += int("cut_in_window" in _b)
        _rep += int("replay_in_window" in _b)
        for _f in _b:
            _motivos[_f] = _motivos.get(_f, 0) + 1
    print(f"\\nQC de esta corrida: {_n} clips evaluados")
    if _n:
        print(f"  bloqueados (los que build_current_store descarta): {_bloq} ({100*_bloq/_n:.2f} %)")
        print(f"    de ellos por cut_in_window   : {_corte} ({100*_corte/_n:.2f} %)")
        print(f"    de ellos por replay_in_window: {_rep} ({100*_rep/_n:.2f} %)")
        print("  desglose:", dict(sorted(_motivos.items(), key=lambda kv: -kv[1])))
        print("  Referencia con la ventana desplegada sobre los 15.451 del corpus: 5,52 % bloqueados,")
        print("  cut_in_window 2,05 %. Con duracion completa: 8,68 % y 5,36 %. Si tu corrida se")
        print("  parece a la segunda columna, es la ventana, no un fallo de extraccion.")

    # 2.bis LA TABLA DEL RELOJ. Coste cero y es la ganancia mas grande de la noche:
    #       `piste_abs.allez_anchor` (piste_abs.py:216) hace
    #           clk = parse_list(_get(row, "clock_running_s")) or meta.get("clock_running_s")
    #       o sea que lee LA FILA DEL INDICE ANTES QUE EL META. Asi que el reloj le llega por
    #       npz/index.csv sin reextraer, sin NPZ nuevos y SIN CAMBIAR LA HUELLA. Y la columna ya la
    #       emite el conducto: es la 45 de las 119. Esto no es un cambio que hacer, es un hecho que
    #       COMPROBAR, y de paso cuenta a cuantos clips desbloquea x_abs / dist_m / dist_allez_*_m,
    #       que hoy son NaN en el 100 % de las extracciones.
    RELOJ_INDICE = PF.columna_reloj(WORK)

    # 3. LA PUERTA DE ALINEACION. Es la comprobacion mas importante del cuaderno.
    #
    #    `build_store` alinea por el FINAL del array y `feature_contract` lo declara
    #    (`alignment: right`, `truncation: tail`). Con un NPZ de duracion completa el array termina
    #    al final del CLIP, no del tocado, y la cola tras el tocado tiene mediana 2,33 s (p95 5,23,
    #    max 14,57) sobre 15.453 clips medidos. Sin la rodaja aplicada, el Store guardaria
    #    [t-2,17 : t+2,33] en vez de [t-4,2 : t+0,3]: el ultimo cuadro deja de ser el tocado y la
    #    ventana de guardia [t-4,0 : t-3,5] desaparece en ~94 % de los clips. SIN NINGUN ERROR.
    #
    #    Esto lo mide sobre el Store ya construido y LEVANTA si esta desalineado. Funciona sin
    #    aplicar ningun parche: usa solo campos que `sample_meta.jsonl` ya escribe. Verificado
    #    sobre un Store real de ventana desplegada: mediana 0,267 s contra post_s 0,3, y
    #    store_truncated=True en 0 de 14.437 muestras.
    if STORE_RESULT is not None:
        ALINEACION = PF.alineacion_del_store(_Path(WORK) / "store",
                                             post_s_esperado=RODAJA_POST_S, exigir=True)
        from fencingly_v2 import store as _ST, feature_contract as _FC
        from fencingly_v2 import training_recipe as _TR
        print(f"\\n  topes: store.T_MAX={_ST.T_MAX} | TrainingRecipe valida 8<=T_max<=135"
              f" | feature_contract.MAX_TIMESTEPS={_FC.MAX_TIMESTEPS}")
        print(f"  T_max pedido {T_MAX} para una rodaja de {PLAN_RODAJA['cuadros']} cuadros")
        if T_MAX > 135:
            print("  !! T_max > 135 exige subir store.T_MAX y el validador de TrainingRecipe.")
            print("     Los dos ficheros estan en NINGUNA de las cuatro listas protegidas y")
            print("     MAX_TIMESTEPS=256 ya lo permite, asi que NO se invalida ningun bundle.")
            print("     Se sube en store.T_MAX y en el validador de TrainingRecipe. Si no se")
            print("     construyo con T_max=135 y la puerta de arriba lo habria dicho.")
"""

COD_PREFLIGHT_02 = """\
# Mismas comprobaciones que en 01, mas el determinismo de la atencion. El orden importa:
# `determinismo()` va ANTES de construir el modelo.
import sys, os
from pathlib import Path
for _cand in (Path.cwd(), Path.cwd() / "paquete", Path("/content"), Path("/content/paquete")):
    if (_cand / "preflight.py").is_file() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
import preflight as PF

PF.protocolo()
ENTORNO_PINES = PF.entorno(exigir=True)

# Backend de atencion `math`. Sin esto, cambiar SOLO el backend produce un "delta" de +0,36 pp con
# 327 de 3.047 decisiones distintas: un numero mayor que tres de los cinco efectos del ultimo
# barrido. Con el, max |p1-p2| = 0,000e+00 entre procesos distintos, sin coste de tiempo medible.
DETERMINISMO = PF.determinismo()

# La huella, otra vez: el interprete que entrena y el que materializo los artefactos tienen que ser
# el mismo, y esa es justo la propiedad que hace comparables las dos mitades del sistema.
HUELLA = PF.huella(exigir=True)

# Si este cuaderno construye algun HistGradientBoosting, tiene que fijar early_stopping: 'auto' se
# enciende por encima de 10.000 filas y la poblacion nueva son 12.930 filas con espejo, asi que YA
# cruza el umbral y las semillas empezarian a importar en silencio.
#   modelo = PF.estimador_hgb(max_iter=200, random_state=7)     # early_stopping=False explicito
"""

COD_DOS_POBLACIONES = """\
# LOS TRES BRAZOS DE ELIMINACION DEL ATAJO, con el signo declarado ANTES de medir.
#
# Declararlo antes es lo que evita leer un resultado al reves. Cada brazo imprime que hace, su base
# medida, que se sabe ya de el, el signo esperado y COMO SE LEERIA AL REVES.
for _b in ("A_ventana_simetrica", "B_reponderar_960", "C_invertir_luces"):
    PF.declarar_brazo(_b)
    print()

# El veredicto, en las dos poblaciones. Rellena las tres series con las de tu evaluacion:
#   y_verdad         : etiqueta real por clip (0/1)
#   y_pred           : prediccion del artefacto. PROMEDIA PROBABILIDADES sobre semillas y umbraliza
#                      UNA vez. NO promedies el acierto por clip: no es la misma cantidad, y con el
#                      convenio del artefacto un efecto de +1,03 fue +0,80 CON EL INTERVALO
#                      CRUZANDO EL CERO. Usa el del artefacto y solo el del artefacto.
#   gano_el_segundo  : True si en ese clip gano el que encendio SEGUNDO. Es una propiedad del
#                      RESULTADO, asi que solo DESCRIBE donde cayo la ganancia: nunca es la
#                      poblacion de evaluacion, ni objetivo de optimizacion, ni puerta de enrutado.
#
# VEREDICTO = PF.dos_poblaciones(y_verdad, y_pred, gano_el_segundo)
#
# Y el resultado NO se resta del 73,88 % ni del 70,04 %: es una linea base nueva, por la ventana Y
# por la huella (el recorte de export_measurements cambio producer_sha256 a proposito).
print("Rellena las tres series y llama a PF.dos_poblaciones. Ver la celda de PROTOCOLO.")
"""


def construir_01(nb):
    celdas = nb["cells"]
    poner(celdas, "protocolo", celda_markdown("protocolo", MD_PROTOCOLO), al_principio=True)

    # La celda de configuracion canonica es la que fija PERCENT_OF_FOIL_CLIPS.
    i_cfg = next((i for i, c in enumerate(celdas)
                  if c["cell_type"] == "code" and "PERCENT_OF_FOIL_CLIPS" in "".join(c["source"])), None)
    if i_cfg is None:
        raise SystemExit("01: no encuentro la celda de PERCENT_OF_FOIL_CLIPS")

    # VENTANA tiene que existir antes del preflight (el preflight la usa), y el preflight antes de
    # que la configuracion haga cualquier trabajo. Se resuelve poniendo VENTANA arriba del propio
    # bloque de configuracion y el preflight justo detras.
    fuente = "".join(celdas[i_cfg]["source"])
    if MARCA not in fuente:
        nueva = ("# " + MARCA + ":config\n"
                 + COD_CONFIG_EXTRA.replace("print(PF.BARRIDO_MEDIDO)", "").rstrip()
                 + "\n\n" + fuente)
        # PF hace falta antes: se importa en la propia celda, de forma minima e idempotente.
        arranque = ('import sys as _sys\nfrom pathlib import Path as _P\n'
                    'for _c in (_P.cwd(), _P.cwd()/"paquete", _P("/content"), _P("/content/paquete")):\n'
                    '    if (_c/"preflight.py").is_file() and str(_c) not in _sys.path:\n'
                    '        _sys.path.insert(0, str(_c))\n'
                    'import preflight as PF\n\n')
        nueva = "# " + MARCA + ":config\n" + arranque + nueva.split("\n", 1)[1]
        celdas[i_cfg]["source"] = nueva.splitlines(keepends=True)

    poner(celdas, "preflight", celda_codigo("preflight", COD_PREFLIGHT_01), tras=i_cfg)
    i_pf = indice_de(celdas, "preflight")
    poner(celdas, "porque", celda_markdown("porque", MD_PORQUE), tras=i_pf)
    poner(celdas, "plan_ventana", celda_codigo("plan_ventana", COD_PLAN_VENTANA),
          tras=indice_de(celdas, "porque"))

    # La extraccion: pasar la ventana al NativeProcessor.
    i_ext = next((i for i, c in enumerate(celdas)
                  if c["cell_type"] == "code" and "CW.NativeProcessor(" in "".join(c["source"])), None)
    if i_ext is None:
        raise SystemExit("01: no encuentro la celda que construye NativeProcessor")
    fuente = "".join(celdas[i_ext]["source"])
    if "pre_s=PRE_S" not in fuente:
        fuente = fuente.replace(
            "CW.NativeProcessor(extractor, audio=True, qc=True, web_moments=True, measurements=True)",
            "CW.NativeProcessor(extractor, pre_s=PRE_S, post_s=POST_S, use_clock=USE_CLOCK,\n"
            "                               audio=True, qc=True, web_moments=True, measurements=True)")
        fuente = fuente.replace('"post_touch_only": True',
                                '"post_touch_only": True,\n'
                                '                 "ventana_almacen": VENTANA_ALMACEN, "pre_s": PRE_S, "post_s": POST_S,\n'
                                '                 "use_clock": USE_CLOCK, "usar_reloj": USAR_RELOJ,\n'
                                '                 "rodaja_entrenamiento": RODAJA_ENTRENAMIENTO,\n'
                                '                 "rodaja_pre_s": RODAJA_PRE_S, "rodaja_post_s": RODAJA_POST_S,\n'
                                '                 "t_max": T_MAX, "plan_rodaja": PLAN_RODAJA,\n'
                                '                 "protocolo": "linea_base_nueva_no_comparable_con_73_88",\n'
                                '                 "pines": ENTORNO_PINES["observado"], "huella": HUELLA["medidas"]')
        if "pre_s=PRE_S" not in fuente:
            raise SystemExit("01: la celda de extraccion no tiene la forma esperada; revisa a mano")
        celdas[i_ext]["source"] = ("# " + MARCA + ":extraccion\n" + fuente).splitlines(keepends=True)

    # El Store se construye DENTRO de la rodaja. `store.build_store` carga cada clip con
    # `D.load_sample` y no acepta ningun recorte, asi que en vez de parchear `store.py` se envuelve
    # esa funcion durante la construccion y se restaura en el `finally`. Cero cambios de fuente.
    i_store = next((i for i, c in enumerate(celdas)
                    if c["cell_type"] == "code" and "build_current_store" in "".join(c["source"])), None)
    if i_store is None:
        raise SystemExit("01: no encuentro la celda que llama a build_current_store")
    fuente = "".join(celdas[i_store]["source"])
    if MARCA not in fuente:
        antes = "            STORE_RESULT = CW.build_current_store(WORK, GLOBAL_REGISTRY, weapon=WEAPON)"
        despues = ("            # El Store se construye desde los NPZ YA RECORTADOS a la rodaja.\n"
                   "            #\n"
                   "            # NO se envuelve `dataset.load_sample` en caliente: se probo y ENVENENA EL\n"
                   "            # STORE. `build_store` graba en `store.json` el `runtime_compatibility()` del\n"
                   "            # momento, que hashea los bindings de `dataset.py`; con el envoltorio puesto\n"
                   "            # queda congelado un hash que ya no coincide y el Store NO SE PUEDE ABRIR\n"
                   "            # NUNCA MAS (ContractError: Bindings runtime incompatibles: dataset.py).\n"
                   "            # Medido: el Store salio con la alineacion perfecta y aun asi inabrible.\n"
                   "            # Ver PF.ENVOLTORIO_PROHIBIDO.\n"
                   "            #\n"
                   "            # `construir_store_con_rodaja` replica el embudo de `build_current_store`\n"
                   "            # (que fija npz_dir a work/npz y por eso no sirve), conserva sus dos guardas\n"
                   "            # -- validate_splits y el eligible_index que no se reescribe -- y anade\n"
                   "            # `embudo_elegibles`, que nombra el paso que vacia la seleccion en vez de\n"
                   "            # levantar KeyError('coverage_L') despues de horas de GPU.\n"
                   "            STORE_RESULT = PF.construir_store_con_rodaja(\n"
                   "                WORK, GLOBAL_REGISTRY, rodaja_nombre=RODAJA_ENTRENAMIENTO,\n"
                   "                t_max=T_MAX, weapon=WEAPON)")
        if antes not in fuente:
            raise SystemExit("01: la celda del Store no tiene la forma esperada; revisa a mano")
        fuente = fuente.replace(antes, despues)
        celdas[i_store]["source"] = ("# " + MARCA + ":store\n" + fuente).splitlines(keepends=True)

    poner(celdas, "verificacion", celda_codigo("verificacion", COD_VERIFICACION), tras=i_store)
    # El respaldo va ANTES de extraer (hay que saber si es reanudable) y su celda de verificacion
    # al final, porque solo se ejecuta una vez y con credencial.
    poner(celdas, "respaldo", celda_codigo("respaldo", COD_RESPALDO),
          tras=indice_de(celdas, "plan_ventana"))
    poner(celdas, "fps_nativo", celda_codigo("fps_nativo", COD_FPS_NATIVO))
    poner(celdas, "verificar_respaldo", celda_codigo("verificar_respaldo", COD_VERIFICAR_RESPALDO))
    return nb


def construir_02(nb):
    celdas = nb["cells"]
    poner(celdas, "protocolo", celda_markdown("protocolo", MD_PROTOCOLO), al_principio=True)
    # El preflight importa `fencingly_v2`, asi que va DESPUES de la celda que abre el release y
    # pone el paquete en sys.path. Se ancla a la celda de colab_utils, que es la ultima del
    # arranque; si no estuviera, al final del arranque de codigo.
    i_arranque = next((i for i, c in enumerate(celdas)
                       if c["cell_type"] == "code" and "colab_utils" in "".join(c["source"])), None)
    if i_arranque is None:
        i_arranque = next((i for i, c in enumerate(celdas)
                           if c["cell_type"] == "code" and "zipfile" in "".join(c["source"])), 0)
    poner(celdas, "preflight", celda_codigo("preflight", COD_PREFLIGHT_02), tras=i_arranque)
    poner(celdas, "dos_poblaciones", celda_codigo("dos_poblaciones", COD_DOS_POBLACIONES))
    return nb


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    origen, destino = Path(sys.argv[1]), Path(sys.argv[2])
    destino.mkdir(parents=True, exist_ok=True)
    hechos = []
    for nombre, constructor in (("01_Data_and_Evidence.ipynb", construir_01),
                                ("02_Training.ipynb", construir_02),
                                ("03_Shared_Web_App.ipynb", None)):
        src = origen / nombre
        if not src.is_file():
            print("falta:", src)
            continue
        salida = destino / nombre.replace("01_Data_and_Evidence", "01_Extraccion_y_Evidencia").replace("02_Training", "02_Entrenamiento_y_Brazos")
        if constructor is None:
            shutil.copyfile(src, destino / nombre)
            hechos.append(str(destino / nombre) + "  (copia sin cambios)")
            continue
        nb = json.loads(src.read_text(encoding="utf-8"))
        nb = constructor(nb)
        salida.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        hechos.append(f"{salida}  ({len(nb['cells'])} celdas)")
    print("Escrito:")
    for h in hechos:
        print("  ", h)
    print("\nLos canonicos NO se han modificado.")


if __name__ == "__main__":
    main()

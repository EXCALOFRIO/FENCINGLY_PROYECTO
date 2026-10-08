"""FENCINGLY - comprobaciones previas al entrenamiento final en Colab (corrida de DURACION COMPLETA).

Se importa desde el cuaderno 01 y desde el 02. Todo lo que puede invalidar el numero final falla
RUIDOSAMENTE aqui, antes de gastar horas de GPU, en vez de convertirse en un desempate de cache
silencioso o en una seleccion sesgada por resolucion.

Uso en el cuaderno:

    import preflight as PF
    PF.protocolo()                      # la regla metodologica, impresa donde no se puede saltar
    PF.entorno(exigir=True)             # versiones clavadas; para si Colab no las puede dar
    PF.determinismo()                   # backend de atencion `math`, antes de construir el modelo
    PF.huella(exigir=True)              # producer_sha256 == 0b87d014...3c4f, o excepcion
    PF.plan_ventana(entorno, ventana)   # cuadros/bytes por clip y % de clips que el planificador rechaza
    PF.parche_de_recorte(exigir=True)   # el recorte de ventana esta dentro de export_measurements?
    PF.indice_real(workspace)           # cuenta con el lector real, no con indice.jsonl
    PF.estimador_hgb(...)               # early_stopping explicito

Nada de esto descarga, instala ni escribe en el espacio de trabajo.
"""
from __future__ import annotations

import contextlib
import shutil
import importlib
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------------------------------
# Constantes medidas. Cada una lleva de donde sale.
# --------------------------------------------------------------------------------------------------

VERSIONES = {"torch": "2.14.0+cu126", "numpy": "2.5.2", "pandas": "3.0.5", "scipy": "1.18.1"}
"""El unico juego que reproduce `producer_sha256 = 0b87d014...3c4f` Y tiene CUDA.

`fz/.venv` reproduce la huella pero es CPU (5,2 s/paso, 29 h por brazo). `scratchpad/cuda/.venv`
tiene numpy 2.5.3 y da la huella equivocada. Con estas versiones el interprete de la red y el de los
artefactos son el mismo, que es la propiedad que hace comparables las dos mitades del sistema.
Numero de referencia de este juego: 70,04 % ponderado en dejar-un-torneo-fuera, semillas 7/11/23 =
69,74 / 70,40 / 69,97, sobre 3.059 pares y 19 sedes, n = 3.047.
"""

HUELLAS_CONOCIDAS = {
    "0b87d014e6d07456f25a2b95de96f59567b3e4d418ec2a6c979b55a3ddb33c4f": (
        "protocolo HISTORICO (sin el recorte en export_measurements). Es la huella de los 23.091 "
        "artefactos ya materializados y la del numero de referencia 70,04 %."),
    "3e1dbe0ad8a4c706f0aa56b304aa498828232d8a7b0ddb0c6221dec6d7419827": (
        "protocolo DURACION COMPLETA (con el recorte a la ventana de analisis dentro de "
        "export_measurements, aplicado el 2026-09-09 a las 10:49). Es la huella correcta para esta "
        "corrida. Los artefactos del protocolo historico NO se reutilizan: hay que re-materializar."),
}
HUELLAS_MOVIMIENTO_CONOCIDAS = {
    "994c874b491472787cece212ffe98d87346b278c8ed969c97a11ab7ca2f12658": "protocolo HISTORICO",
    "6c9fa66b2f4614c10a2d9fccf4cb4a1d450c17f5fc96808e44bf418e6d4c3ce7": "protocolo DURACION COMPLETA",
}
HUELLA_MEDIDAS = "3e1dbe0ad8a4c706f0aa56b304aa498828232d8a7b0ddb0c6221dec6d7419827"
HUELLA_MOVIMIENTO = "6c9fa66b2f4614c10a2d9fccf4cb4a1d450c17f5fc96808e44bf418e6d4c3ce7"
"""Medidas el 2026-09-09 con `.venv-gpu` (python 3.14.2, numpy 2.5.2, pandas 3.0.5, scipy 1.18.1).

DOS HUELLAS, Y NO SON INTERCAMBIABLES. El parche que mete el recorte de ventana en
`export_measurements` toca `measurement_artifacts.py`, que es uno de los 20 de
MEASUREMENT_SOURCES, asi que **cambia la huella a proposito**: eso es re-materializacion declarada,
no un fallo. Medido en el momento en que el parche entro:

    protocolo            medidas              movimiento
    historico            0b87d014...3c4f      994c874b...2658
    duracion completa    3e1dbe0a...9827      6c9fa66b...3ce7

Consecuencia operativa: **los 23.091 artefactos del protocolo historico dejan de ser cache valida** y
hay que re-materializar. Y consecuencia de protocolo: es otra razon mas, ademas de la ventana, por la
que el numero de esta corrida **no se resta del 70,04 % ni del 73,88 %**.

`huella()` acepta cualquiera de las dos como CONOCIDA y dice cual, pero solo la de duracion completa
como ESPERADA. Una tercera obliga a pararse: significa que una fuente cambio y nadie lo declaro.

`producer_sha256` NO es el hash de los 20 ficheros de MEASUREMENT_SOURCES: `_producer()` hashea el
diccionario entero, que lleva ademas `runtime_versions`, las constantes efectivas de ocho modulos
(incluida la calibracion ajustada de estatura), `reporting_version`, el catalogo de 68 unidades y
`policy`. Comprobado: numpy 2.5.2 -> 0b87d014...3c4f, numpy 2.5.3 -> 874c1ca9...6533 desde fuentes
byte a byte identicas. Un parche de numpy invalida los 23.091 artefactos y el fallo es un desempate
de cache silencioso, sin error.

Sobre la version de Python: `source_ast_sha256` se calculo con Python 3.12.8 y con 3.14.2 sobre los
20 ficheros y sale IDENTICO en los 20 (agregado afe6490d...). O sea que la huella NO depende de la
version menor de Python en este juego de fuentes, y Colab puede traer el Python que quiera siempre
que numpy/pandas/scipy esten clavados. No esta demostrado para versiones futuras: por eso existe
esta comprobacion.
"""

# Geometria de la ventana. (nombre, pre_s, post_s, use_clock).
VENTANAS = {
    "desplegada": (4.0, 0.3, True),
    "amplia": (6.0, 1.0, False),
    "completa": (60.0, 60.0, False),
}
"""`lights.window` da f0 = max(0, ft - pre*fps) y f1 = min(T, ft + post*fps + 1), asi que 60/60 con
`use_clock=False` es el clip entero para todo el corpus (duracion maxima medida 16,71 s) y
`max_video_seconds = 90` lo acota por arriba.

SON DOS DECISIONES, NO UNA. Almacenar y entrenar se eligen por separado:

    VENTANA_ALMACEN       = "completa"   lo que se guarda en el NPZ; da la opcionalidad
    VENTANA_ENTRENAMIENTO = "amplia"     la rodaja sobre la que se mide y se entrena

Guardar el clip entero cuesta GPU y High-RAM y compra no volver a descargar 93 GB. Entrenar sobre
una rodaja no cuesta nada mas y evita los cuatro precios de la duracion completa. Lo que la
duracion completa compra de verdad -- re-ventanear las 256 columnas agregadas -- se compra igual,
porque las agregadas se recalculan sobre la rodaja que se elija.

`use_clock` es un mando aparte (`USAR_RELOJ`) porque su efecto es grande y gratis: apagarlo lleva la
ventana de guardia de 72,78 % a 91,42 % al mismo coste exacto de decodificacion. AVISO: hay trabajo
en curso para CONECTAR la tabla del reloj, asi que las dos decisiones interactuan y hay que medirlas
JUNTAS, en un factorial 2x2, no una encima de la otra.
"""

COLA_TRAS_TOCADO = """\
El cuarto precio de la duracion completa, y es peor que los otros tres: ROMPE LA ALINEACION.

`build_store` hace `t = min(T_max, length); cut = length - t` y escribe `array[i, -t:]`: alinea por
el FINAL del array y trunca la cabeza. `feature_contract` lo declara: `"alignment": "right"`,
`"truncation": "tail"`. Con la ventana desplegada eso es correcto, porque el array TERMINA 0,3 s
despues del tocado. Con un NPZ de duracion completa el array termina al final del CLIP, y el clip
sigue despues del tocado:

  cola tras el tocado (15.453 clips, medido)  min 0,03 | p05 0,93 | MEDIANA 2,33 | media 2,62
                                              p95 5,23 | max 14,57  segundos
  cola > 0,3 s : 15.379 (99,52 %)
  cola > 1,0 s : 14.479 (93,70 %)
  cola > 2,0 s :  9.716 (62,87 %)

O sea que con `T_MAX = 135` (4,5 s a 30 fps) el Store guardaria, para el clip mediano, la ventana
[tocado - 2,17 s : tocado + 2,33 s] en vez de [tocado - 4,2 : tocado + 0,3]:

  - el ultimo cuadro deja de ser el tocado, que es la convencion que la red asume;
  - la ventana de guardia [t-4,0 : t-3,5] DESAPARECE del Store en ~94 % de los clips, o sea
    justo lo que la duracion completa venia a garantizar;
  - y no salta ningun error: `store_truncated` se pone a True, que es lo unico que se nota, y su
    unico efecto declarado es enmascarar las etiquetas auxiliares.

Por eso la rodaja de entrenamiento NO es una optimizacion: es OBLIGATORIA. Y por eso este modulo
trae `alineacion_del_store()`, que lo mide despues de construir el Store y FALLA si el tocado no
esta al final. Sin cambiar ni una linea de fuente, eso convierte una regresion silenciosa en un
fallo ruidoso.
"""

BARRIDO_MEDIDO = """\
Barrido sobre los 15.451 clips con QC utilizable de runs_full (medido 2026-09-09, sin re-extraer:
se reevalua la geometria de `lights.window` y las banderas de QC ya escritas en analysis/qc/*.json).

  candidato                  cuadros  MiB   MiB    rechazados por el planificador     QC      2a     ventana
                             mediana  med   p95    T4 3,85GiB  HighRAM 6GiB  12GiB  bloq   luz    guardia
  4.0/0.3 +reloj (HOY)           129   680  1554       0,00%       0,00%      0,00%  5,53%  99,05%  72,78%
  4.0/0.3 sin reloj              129   680  1554       0,00%       0,00%      0,00%  5,56%  99,05%  91,42%
  6.0/1.0 sin reloj              210  1107  2515       4,14%       0,00%      0,00%  6,52%  99,99%  97,04%
  8.0/3.0 sin reloj              294  1582  3939       4,36%       0,49%      0,00%  8,30%  99,99%  97,04%
  12.0/6.0 sin reloj             314  1819  5078      10,90%       0,50%      0,00%  8,65%  99,99%  97,04%
  DURACION COMPLETA 60/60        318  1846  5233      11,31%       0,50%      0,00%  8,69%  99,99%  97,04%

Lo que dice esta tabla, y conviene leerlo antes de elegir:

1. La ventana de guardia [t-4,0 : t-3,5] pasa de 72,78 % a 91,42 % con SOLO apagar el reloj, al mismo
   coste exacto de decodificacion. Y su techo es 97,04 %, no 100 %: en ~3 % de los clips el tocado
   ocurre a menos de 4,0 s del comienzo del video. La duracion completa no puede inventar cuadros
   que nadie filmo.
2. La segunda luz: |dt_ms| tiene maximo 520 ms sobre los 3.331 clips de dos luces (p50 80, p95 267,
   p99 360, y el 100 % por debajo de 1.000 ms). `post_s = 0,6` la captura entera; `post_s = 1,0` lleva
   el doble de margen. La duracion completa no aporta nada mas en este punto.
3. El precio de la duracion completa frente a 6.0/1.0 es 318 contra 210 cuadros (1,51x mas GPU y mas
   CPU por clip), 0,50 % de clips rechazados en High-RAM y 11,31 % en un T4 estandar, y +2,17 pp de
   clips descartados por QC. Lo que compra es la libertad de re-ventanear el bloque agregado y de
   muestrear video mas ancho DESPUES, sin volver a descargar.
"""

TOPE_QC = """\
El coste de QC de la duracion completa, medido y no estimado.

`clip_quality` marca `cut_in_window` y `replay_in_window` como BLOQUEANTES (peso 0,0), y
`colab_workflow.build_current_store` descarta toda fila con `qc_blocking` no vacio. Esas dos banderas
se evaluan sobre la ventana efectiva, asi que al ensanchar la ventana entran cortes de camara que hoy
quedan fuera:

  clips con >= 1 corte de camara          828  ( 5,36 %)
  clips con >= 1 repeticion                 3  ( 0,02 %)   <- las repeticiones no son el problema

  bandera bloqueante        ventana de hoy     duracion completa
  cut_in_window              316  (2,05 %)       828  (5,36 %)
  replay_in_window             1  (0,01 %)         3  (0,02 %)
  otras (no dependen)        586  (3,79 %)       586  (3,79 %)
  BLOQUEADO cualquiera       853  (5,52 %)      1341  (8,68 %)

  entran al Store          14.598            ->  14.110      (-488 clips, -3,16 %)

Es una perdida real. Las repeticiones NO son el problema (3 clips); los cortes de camara si.

Y LA RODAJA DE ENTRENAMIENTO NO LO ARREGLA SOLA. Es la parte del diseno "extraer completo, entrenar
sobre una rodaja" que no sale gratis: el QC se calcula en la EXTRACCION, sobre la ventana GUARDADA,
y queda congelado en `analysis/qc/<clip>.json`. La cadena: `NativeProcessor.__call__` llama a
`CQ.assess_clip` durante la extraccion (colab_workflow.py:842-847); `assess_clip` toma la ventana de
`effective_window`, que la lee de `time_s` del NPZ, o de `f0`/`f1`, o de
`t_window_start_s`/`t_window_end_s` -- los tres son la ventana REALMENTE GUARDADA, y
`post_s_requested` solo se usa de ultimo recurso; `_flatten_vm` evalua las dos banderas sobre esa
ventana; `build_current_store` descarta lo que tenga `qc_blocking` no vacio. Elegir una rodaja
DESPUES no cambia nada de eso.

Se recupera acotando `cut_in_window`/`replay_in_window` a la ventana de ANALISIS
([tocado-4,0 : tocado+0,3]) en vez de a la guardada, que es la misma idea que
`coach_panel._ventana_de_analisis`. El texto exacto esta en
la seccion de vias no tomadas del documento, y es un no-op DEMOSTRABLE con la ventana desplegada,
porque entonces a0 = max(w0, t-4,0) = w0 y a1 = min(w1, t+0,3) = w1. `clip_quality.py` NO esta en
MEASUREMENT_SOURCES (si en FINGERPRINT_SOURCES), asi que el coste es re-cachear el panel de coach,
no re-materializar los 23.091 artefactos. NO APLICADO: es una decision.
"""


class PreflightError(RuntimeError):
    pass


def _linea(x=78):
    print("-" * x)


# --------------------------------------------------------------------------------------------------
# 1. La regla metodologica. Se imprime siempre, y el cuaderno la lleva tambien en markdown.
# --------------------------------------------------------------------------------------------------

PROTOCOLO = """\
================================================================================
 REGLA DE PROTOCOLO - LEER ANTES DE INTERPRETAR CUALQUIER NUMERO DE ESTA CORRIDA
================================================================================
 El numero que salga de una corrida de DURACION COMPLETA es una LINEA BASE NUEVA.
 NO se puede comparar con el 73,88 %, ni con el 70,04 %, ni con ninguna cifra
 anterior de este proyecto.

 Cambian `pre_s_eff`, `n_frames_window`, cada rasgo derivado de la ventana y la
 huella del productor. Es OTRA CADENA DE MEDIDA. Restar una cifra de un protocolo
 a una cifra de otro protocolo es un error de protocolo, no una comparacion
 conservadora.

 Lo que SI se puede comparar: duracion-completa contra duracion-completa, con y
 sin el cambio que se este probando, pareado por clip.

 Y el veredicto se reporta SIEMPRE sobre las dos poblaciones por separado:
   gano el que encendio segundo  n = 1.671   base 83,00 %
   gano el que encendio primero  n =   960   base 45,21 %   <- aqui esta el premio
   brecha +37,80 pp; el atajo solo acierta 63,51 % y el modelo lo predice el 72,71 %

   llevar los 960 de 45,21 % a 60 %  ->  +5,40 pp globales
   llevar los 960 de 45,21 % a 75 %  -> +10,95 pp globales
   degradar 5 pp los 1.671           ->  -3,18 pp globales
 El veredicto es la MEDIA GLOBAL PAREADA. El subgrupo describe donde cayo la
 ganancia; nunca es la poblacion de evaluacion ni un objetivo de optimizacion.

 LIMITACION HONESTA, y va en el informe: con la potencia de este corpus un efecto
 real de +1,6 pp es INDETECTABLE, y hacen falta de 12 a 20 campeonatos mas para
 tener el 80 % de potencia sobre ese tamano de efecto. Una mejora real puede no
 ser DEMOSTRABLE aunque exista. No prometas lo que este corpus no puede medir.

 VARAS DE MEDIR, para no creerse un delta:
  - cambiar SOLO el backend de atencion, misma receta, misma semilla, mismo
    interprete: +0,36 pp con 327 de 3.047 decisiones distintas. Cualquier mejora
    de red por debajo de eso es indistinguible de recompilar.
  - el nulo se elige por el TIPO de rasgo: el desplazamiento circular PRESERVA los
    rasgos de nivel (una columna quedo a r = +1,000), INFLA los de derivada (hasta
    7,79x) y es INERTE sobre representaciones agrupadas (coseno 0,9980). Para
    rasgos de nivel, nulo de signo de lado aleatorio.
  - dos convenios de promediado de semillas NO son la misma cantidad. El banco de
    ablaciones promedia el acierto por clip; el artefacto promedia probabilidades y
    umbraliza una vez. Un efecto de +1,03 con el primero fue +0,80 con intervalo
    cruzando el cero con el segundo. USA EL DEL ARTEFACTO.
  - nunca evaluar dentro de un subgrupo definido por el resultado: alli la base cae
    bajo el azar y un bloque SIN informacion compro +3,08 pp.
================================================================================"""


def protocolo():
    print(PROTOCOLO, flush=True)
    return PROTOCOLO


# --------------------------------------------------------------------------------------------------
# 2. Entorno clavado
# --------------------------------------------------------------------------------------------------

def _version(nombre):
    try:
        return importlib.metadata.version(nombre)
    except importlib.metadata.PackageNotFoundError:
        return None


def entorno(exigir: bool = True, permitir_torch_distinto: bool = False) -> dict:
    """Mide lo que trae este runtime y lo compara con `VERSIONES`. Con `exigir=True` levanta.

    `permitir_torch_distinto` existe solo para una pasada de ensayo (rehearsal) cuya cifra se va a
    tirar: si se usa, el numero que salga no es comparable con nada del informe y hay que decirlo
    junto al resultado.
    """
    observado = {nombre: _version(nombre) for nombre in VERSIONES}
    try:
        import torch
        observado["torch"] = torch.__version__
        cuda = {"disponible": bool(torch.cuda.is_available()),
                "version_cuda": torch.version.cuda,
                "dispositivo": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "vram_total_gb": (round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
                                  if torch.cuda.is_available() else 0.0)}
    except Exception as exc:  # noqa: BLE001
        cuda = {"disponible": False, "error": repr(exc), "vram_total_gb": 0.0}

    difieren = {n: (VERSIONES[n], observado[n]) for n in VERSIONES if observado[n] != VERSIONES[n]}
    print("Python  :", sys.version.split()[0])
    for nombre in ("torch", "numpy", "pandas", "scipy"):
        marca = "OK " if observado[nombre] == VERSIONES[nombre] else "!! "
        print(f"{marca}{nombre:8}: {observado[nombre]}   (clavado: {VERSIONES[nombre]})")
    print("CUDA    :", cuda)

    # TRAMPA QUE SOLO PASA EN COLAB, y que la comprobacion de la huella NO detecta.
    #
    # `measurement_artifacts._producer()` lee las versiones con `importlib.metadata.version(...)`,
    # que consulta los METADATOS DE LA DISTRIBUCION EN DISCO, no el modulo cargado en memoria. Si
    # haces `!pip install numpy==2.5.2` y NO reinicias el runtime, numpy sigue cargado en la version
    # anterior mientras los metadatos ya dicen 2.5.2: la huella COINCIDE y sin embargo las medidas
    # se calculan con el numpy equivocado. Es el fallo silencioso peor de todos, porque la propia
    # comprobacion de integridad da el visto bueno. Localmente no ocurre: se elige el venv y ya.
    cargado = {}
    for nombre in ("numpy", "pandas", "scipy"):
        modulo = sys.modules.get(nombre)
        if modulo is not None:
            cargado[nombre] = getattr(modulo, "__version__", None)
    desalineados = {n: (cargado[n], observado[n]) for n in cargado
                    if cargado[n] is not None and cargado[n] != observado[n]}
    if desalineados:
        print("\n" + "=" * 78)
        print("EL RUNTIME NO SE HA REINICIADO DESPUES DE INSTALAR.")
        for n, (mem, disco) in desalineados.items():
            print(f"  {n}: cargado en memoria {mem}  |  metadatos en disco {disco}")
        print("`_producer()` lee los METADATOS, asi que la huella saldria CORRECTA y las medidas se")
        print("calcularian con la version equivocada. Entorno de ejecucion -> Reiniciar sesion, y")
        print("vuelve a ejecutar desde la primera celda.")
        print("=" * 78, flush=True)
        if exigir:
            raise PreflightError(f"version cargada != version instalada: {sorted(desalineados)}; "
                                 "reinicia el runtime antes de medir nada")

    resultado = {"observado": observado, "esperado": dict(VERSIONES), "difieren": difieren, "cuda": cuda,
                 "cargado_en_memoria": cargado, "desalineados": desalineados,
                 "python": sys.version.split()[0]}
    if not difieren:
        print("\nEntorno clavado reproducido. El interprete de la red y el de los artefactos son el mismo.")
        return resultado

    pines = " ".join(f"{n}=={VERSIONES[n]}" for n in difieren if n != "torch")
    print("\n" + "=" * 78)
    print("EL ENTORNO NO COINCIDE. Difieren:", ", ".join(f"{n} {o} -> {e}" for n, (e, o) in difieren.items()))
    print("Sin estas versiones exactas la huella cambia y el numero NO es comparable con nada.")
    if pines:
        print("\n  Para clavarlo (y REINICIAR el runtime despues, o numpy queda cargado en memoria):")
        print(f"    !pip install -q --no-input {pines}")
    if "torch" in difieren:
        print(f"\n  torch: Colab trae {observado['torch']} y hace falta {VERSIONES['torch']}.")
        print("    !pip install -q torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126")
        print("    Si el indice de PyTorch no publica ese wheel para el Python de este runtime,")
        print("    NO SIGAS: elige un runtime con otra version de Python o acepta explicitamente")
        print("    que la corrida es un ensayo cuyo numero se tira.")
    print("=" * 78, flush=True)
    if exigir and (difieren.keys() - ({"torch"} if permitir_torch_distinto else set())):
        raise PreflightError(f"Entorno no clavado: {sorted(difieren)}")
    return resultado


# --------------------------------------------------------------------------------------------------
# 3. Determinismo del backend de atencion (inline de scratchpad/prio9_perdida/determinismo.py)
# --------------------------------------------------------------------------------------------------

def determinismo(activar: bool = True) -> dict:
    """Fuerza el camino determinista de la atencion. LLAMAR ANTES de construir el modelo.

    Medido en RTX 3060, torch 2.14.0+cu126, dos entrenamientos del mismo pliegue con la misma
    semilla (worldchamps2022, n=439):
        sin arreglo                        max |p1-p2| = 7,15e-02   3 de 439 decisiones
        use_deterministic_algorithms solo  max |p1-p2| = 1,03e-01   3 de 439 decisiones
        con backend `math` (esto)          max |p1-p2| = 0,00e+00   0 de 439 decisiones
    Sin coste de tiempo medible: 37-39 s por pliegue en los tres casos.

    Por que importa: el pareado cancela la varianza de semilla y de pliegue, pero NO esta, porque
    cada brazo trae su propia realizacion no determinista. Fijarlo estrecha el recorrido entre
    semillas de 1,35 pp a 0,66 pp.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    if not activar or not torch.cuda.is_available():
        estado = {"aplicado": False, "cuda": torch.cuda.is_available()}
        print("determinismo:", estado)
        return estado
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    estado = {"aplicado": True, "cuda": True,
              "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
              "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
              "math_sdp": torch.backends.cuda.math_sdp_enabled()}
    print("determinismo:", estado)
    if estado["mem_efficient_sdp"] or estado["flash_sdp"] or not estado["math_sdp"]:
        raise PreflightError("El backend de atencion `math` no quedo forzado; el numero seria irreproducible")
    return estado


# --------------------------------------------------------------------------------------------------
# 4. La huella de las medidas
# --------------------------------------------------------------------------------------------------

def huella(exigir: bool = True, esperada: str | None = None) -> dict:
    """`measurement_producer()["producer_sha256"]` contra el valor medido. Falla ruidosamente.

    Distingue tres casos, que no son lo mismo:
      - coincide con la ESPERADA (duracion completa)  -> verde
      - coincide con otra CONOCIDA (protocolo historico) -> avisa y levanta: el protocolo no es el
        que el paquete espera, y mezclar artefactos de dos protocolos es el error que todo esto
        existe para impedir
      - no coincide con ninguna -> una fuente cambio y nadie lo declaro. Pararse.
    """
    from fencingly_v2 import measurement_artifacts as MA
    esperada = HUELLA_MEDIDAS if esperada is None else esperada
    medidas = MA.measurement_producer()
    movimiento = MA.motion_producer()
    resultado = {"medidas": medidas["producer_sha256"], "medidas_esperada": esperada,
                 "movimiento": movimiento["producer_sha256"], "movimiento_esperada": HUELLA_MOVIMIENTO,
                 "runtime_versions": medidas.get("runtime_versions"),
                 "reporting_version": medidas.get("reporting_version"),
                 "protocolo_medidas": HUELLAS_CONOCIDAS.get(medidas["producer_sha256"]),
                 "protocolo_movimiento": HUELLAS_MOVIMIENTO_CONOCIDAS.get(movimiento["producer_sha256"])}
    ok_m = resultado["medidas"] == esperada
    ok_v = resultado["movimiento"] == HUELLA_MOVIMIENTO
    print(f"{'OK ' if ok_m else '!! '}producer_sha256 medidas    : {resultado['medidas']}")
    print(f"{'OK ' if ok_v else '!! '}producer_sha256 movimiento : {resultado['movimiento']}")
    print("   runtime_versions        :", resultado["runtime_versions"])
    print("   reporting_version       :", resultado["reporting_version"])
    if resultado["protocolo_medidas"]:
        print("   protocolo reconocido    :", resultado["protocolo_medidas"])
    if ok_m and ok_v:
        print("\nHuella reproducida, protocolo DURACION COMPLETA. Los artefactos del protocolo")
        print("historico (0b87d014...3c4f) NO son cache valida: hay que re-materializar, y el numero")
        print("de esta corrida no se resta del 70,04 % ni del 73,88 %.")
        return resultado
    print("\n" + "=" * 78)
    if resultado["protocolo_medidas"]:
        print("LA HUELLA ES DE OTRO PROTOCOLO, no de este.")
        print("  medida  :", resultado["protocolo_medidas"])
        print("  esperada:", HUELLAS_CONOCIDAS.get(esperada, "(desconocida)"))
        print("Mezclar artefactos de dos protocolos es exactamente el error que esta comprobacion")
        print("existe para impedir. Si el recorte de `export_measurements` no esta dentro, esta")
        print("corrida no puede ser de duracion completa: ver preflight.parche_de_recorte().")
    else:
        print("LA HUELLA NO ES NINGUNA DE LAS CONOCIDAS. Los artefactos de medida NO se reutilizan y")
        print("el fallo seria un desempate de cache SILENCIOSO si esto no existiera. Causas:")
        print("  1. numpy/pandas/scipy no clavados  -> ejecuta preflight.entorno() y REINICIA el runtime")
        print("  2. una fuente de MEASUREMENT_SOURCES cambio -> re-materializacion a proposito;")
        print("     apunta la huella nueva junto al resultado y no la mezcles con artefactos viejos")
        print("  3. una version de Python que anade campos al AST (comprobado igual en 3.12.8 y 3.14.2)")
    print("=" * 78, flush=True)
    if exigir:
        raise PreflightError("producer_sha256 no reproduce la huella esperada para este protocolo")
    return resultado


# --------------------------------------------------------------------------------------------------
# 5. El parche del recorte de ventana: con duracion completa pasa de inerte a NECESARIO
# --------------------------------------------------------------------------------------------------

def parche_de_recorte(exigir: bool = True) -> dict:
    """Comprueba que el camino de medidas por clip recorta a la ventana de analisis.

    Hoy `coach_panel._ventana_de_analisis` existe y se aplica en `build_video_stats` (camino de
    LOTE), pero `measurement_artifacts.export_measurements` mide sobre el array COMPLETO del NPZ.
    Con la ventana desplegada eso es inerte, porque `poses.shape[0] == n_frames_window` en los
    15.452 clips del corpus: no hay ni un NPZ de duracion completa.

    Con duracion completa deja de ser inerte. Sin el recorte, las medidas se calcularian sobre el
    clip entero en vez de sobre la ventana y el docstring del propio recorte declara desviaciones de
    hasta 152 grados en `fore_ang_guard` y de 1-2 % en la escala del torso, medidas sobre 20 clips,
    SIN NINGUNA BANDERA. Los veinticinco sitios que miden la guardia como `X[:GUARD_S*fps]` asumen
    que el cuadro 0 esta `pre_s` antes de la luz.

    NO se arregla desde aqui: `measurement_artifacts.py` y `coach_panel.py` son dos de los 20
    ficheros de MEASUREMENT_SOURCES, ampliarlos/cambiarlos re-materializa artefactos y hay un agente
    trabajando en ellos. Esta funcion informa; la correccion es de ese camino.
    """
    import inspect
    from fencingly_v2 import coach_panel as CP, measurement_artifacts as MA

    existe = hasattr(CP, "_ventana_de_analisis")
    fuente_export = inspect.getsource(MA.export_measurements) if hasattr(MA, "export_measurements") else ""
    aplicado = "_ventana_de_analisis" in fuente_export
    fuente_lote = inspect.getsource(CP.build_video_stats)
    aplicado_lote = "_ventana_de_analisis" in fuente_lote

    resultado = {"recorte_existe": existe, "aplicado_en_export_measurements": aplicado,
                 "aplicado_en_build_video_stats": aplicado_lote}
    print(f"{'OK ' if existe else '!! '}coach_panel._ventana_de_analisis existe            : {existe}")
    print(f"{'OK ' if aplicado_lote else '!! '}aplicado en build_video_stats (camino de LOTE)    : {aplicado_lote}")
    print(f"{'OK ' if aplicado else '!! '}aplicado en export_measurements (camino por CLIP) : {aplicado}")
    if aplicado and aplicado_lote:
        return resultado
    print("\n" + "=" * 78)
    print("EL RECORTE DE VENTANA NO ESTA EN EL CAMINO POR CLIP.")
    print("`NativeProcessor.__call__` llama a `export_measurements(D.load_sample(npz), ...)` con la")
    print("muestra CRUDA, y `export_measurements` pasa `raw` a `CP._one_clip` sin recortar. Con NPZ")
    print("de duracion completa toda medida de guardia se calcularia sobre el clip entero.")
    print("NO EJECUTES la corrida de duracion completa hasta que el parche este dentro.")
    print("=" * 78, flush=True)
    if exigir:
        raise PreflightError("export_measurements no recorta a la ventana de analisis; con duracion "
                             "completa las medidas de guardia serian invalidas sin aviso")
    return resultado


# --------------------------------------------------------------------------------------------------
# 6. Plan de ventana: cuadros, bytes y cuantos clips rechazaria el planificador
# --------------------------------------------------------------------------------------------------

def plan_ventana(entorno_reporte, ventana: str = "completa", *, ram_libre_gb: float | None = None) -> dict:
    """Traduce la eleccion de ventana a lo que va a pasar en ESTE runtime.

    `colab_workflow.check_decode_budget` estima `cuadros * W * H * 3 * DECODE_COPY_FACTOR(2)` y
    RECHAZA el clip si supera `max_decoded_bytes`, que `resource_plan` fija en
    `min(6.0 si cuda else 3.0, ram_libre * 0.35) GiB`. El tope de 6,0 GiB esta escrito a mano y es el
    que decide si la duracion completa cabe.

    Un rechazo correlacionado con la resolucion SESGA la seleccion: es exactamente el fallo que el
    runbook de 2026-09-06 elimino bajando DECODE_COPY_FACTOR de 3 a 2, y la duracion completa lo
    reabre. Medido: 11,31 % de clips rechazados con un tope de 3,85 GiB (T4 estandar) contra 0,50 %
    con 6,0 GiB (High-RAM).
    """
    if ventana not in VENTANAS:
        raise PreflightError(f"Ventana desconocida: {ventana!r}; elige una de {sorted(VENTANAS)}")
    pre_s, post_s, use_clock = VENTANAS[ventana]
    recursos = entorno_reporte.get("resources", entorno_reporte) if isinstance(entorno_reporte, dict) else {}
    cuda = bool((recursos.get("gpu") or {}).get("available"))
    if ram_libre_gb is None:
        ram_libre_gb = float((recursos.get("ram") or {}).get("free_gb") or 0.0)
    tope_gib = min(6.0 if cuda else 3.0, ram_libre_gb * 0.35)
    tope = int(tope_gib * 1024 ** 3)

    # Composicion medida del corpus (300 mp4 muestreados, semilla 42): resolucion x fps.
    # 1280x720 63,0 % | 1920x1080 29,3 % | 854x480 6,3 % | 640x360 1,3 %
    # fps ~30 56,0 % | ~50 20,7 % | ~25 14,7 % | ~60 6,3 %
    # duracion mediana 10,0 s, media 11,0 s, p95 15,0 s, maxima 15,4 s (16,71 s en el QC del corpus).
    casos = [("1280x720@30", 1280, 720, 30.0, 0.36), ("1280x720@50", 1280, 720, 50.0, 0.13),
             ("1280x720@25", 1280, 720, 25.0, 0.09), ("1280x720@60", 1280, 720, 60.0, 0.05),
             ("1920x1080@30", 1920, 1080, 30.0, 0.16), ("1920x1080@50", 1920, 1080, 50.0, 0.08),
             ("1920x1080@25", 1920, 1080, 25.0, 0.04), ("854x480@30", 854, 480, 30.0, 0.06),
             ("640x360@30", 640, 360, 30.0, 0.03)]
    dur_med, dur_p95 = 10.48, 14.70          # medido sobre 15.451 clips (QC de runs_full)
    filas = []
    peso_fuera = 0.0
    for nombre, w, h, fps, peso in casos:
        for etiqueta, dur in (("mediana", dur_med), ("p95", dur_p95)):
            cuadros = min(int(round(dur * fps)), int(math.ceil((pre_s + post_s + 1.0) * fps)))
            bytes_ = cuadros * w * h * 3 * 2
            cabe = bytes_ <= tope
            filas.append({"caso": nombre, "duracion": etiqueta, "cuadros": cuadros,
                          "MiB": round(bytes_ / 1024 ** 2, 0), "cabe": cabe, "peso": peso})
            if etiqueta == "mediana" and not cabe:
                peso_fuera += peso

    print(f"ventana '{ventana}': pre_s={pre_s} post_s={post_s} use_clock={use_clock}")
    print(f"runtime: cuda={cuda} | RAM libre {ram_libre_gb:.1f} GiB -> max_decoded_bytes = {tope_gib:.2f} GiB")
    print(f"{'caso':16} {'dur':8} {'cuadros':>8} {'MiB':>7}  cabe")
    for f in filas:
        print(f"{f['caso']:16} {f['duracion']:8} {f['cuadros']:8d} {f['MiB']:7.0f}  {'si' if f['cabe'] else 'NO'}")
    print(f"\ncota SUPERIOR de la fraccion rechazada (producto de marginales): ~{100*peso_fuera:.0f} %")
    print("   Esta cota es pesimista: multiplica las marginales de resolucion y de fps como si fueran")
    print("   independientes, y no lo son. La cifra buena es la del barrido sobre la distribucion")
    print("   conjunta real de los 15.451 clips: 0,50 % rechazados con tope 6,0 GiB (High-RAM) y")
    print("   11,31 % con tope 3,85 GiB (T4 estandar). Usa esas dos.")
    if peso_fuera > 0.005:
        print("AVISO: el rechazo esta correlacionado con la resolucion, asi que SESGA la seleccion.")
        print("       Pide un runtime High-RAM, o sube el tope de 6,0 GiB de `resource_plan` a mano,")
        print("       o usa la ventana 'amplia' (6.0/1.0), que no rechaza nada en ningun runtime.")
        print("       El pico REAL medido es 1,08x una copia BGR (1,66 GiB para 1,54 GiB de copia),")
        print("       no 2x: el tope es aritmetica de planificacion, no memoria.")
    print("\nTamano de NPZ resultante, con el modelo medido bytes = 6.005 + 138,4*T (R2 = 0,92, n = 513):")
    for T in (129, 210, 318):
        print(f"   T = {T:4d} cuadros -> {(6005 + 138.4*T)/1024:6.1f} KiB/clip"
              f" -> {25139*(6005 + 138.4*T)/1024**3:5.2f} GiB para 25.139 clips de florete")
    print("El disco no es el problema: la duracion completa lleva el corpus de NPZ de ~0,35 GiB a")
    print("~1,2 GiB. Los mp4 se borran tras extraer (media medida 2,24 MB/clip), asi que el disco")
    print("queda plano.")
    return {"ventana": ventana, "pre_s": pre_s, "post_s": post_s, "use_clock": use_clock,
            "max_decoded_bytes": tope, "max_decoded_gib": tope_gib, "cuda": cuda,
            "fraccion_rechazada_estimada": peso_fuera, "casos": filas}


# --------------------------------------------------------------------------------------------------
# 6.bis La rodaja de entrenamiento y la puerta de alineacion
# --------------------------------------------------------------------------------------------------

RODAJAS = {
    "historica": (4.0, 0.3),
    "recomendada": (4.0, 0.6),
    "ancha": (6.0, 1.0),
}
"""Rodajas de ENTRENAMIENTO candidatas, con la aritmetica que decide si caben.

Una rodaja `(pre_s, post_s)` a `TARGET_FPS = 30` ocupa `round(pre_s*30) + round(post_s*30) + 1`
cuadros, que es lo que hace `rodaja()` (y lo que hace `lights.window`). Y el Store SOLO guarda
`T_max` cuadros, alineados por el final. Asi que:

   rodaja        cuadros   T_max necesario   cabe en 135   guardia [t-4,0:t-3,5]   2a luz (|dt| max 520 ms)
   4.0 / 0.3         130               135        SI              SI                  99,05 %
   4.0 / 0.6         139               140        NO              SI                 100,00 %
   6.0 / 1.0         211               211        NO              SI                 100,00 %

**El detalle que hay que ver antes de elegir 6.0/1.0:** 6,0 + 1,0 = 7,0 s son **211 cuadros**, y
`T_MAX = 135` son 4,467 s. Con `T_max` sin subir, el Store se queda los ultimos 135 de esos 211, o
sea `[t-3,5 : t+1,0]`, y **la ventana de guardia `[t-4,0 : t-3,5]` se queda fuera**: justo lo que la
rodaja venia a proteger. La rodaja NO hace que `T_MAX` deje de estorbar por si sola; hay que subir
`T_max` a la vez, o elegir una rodaja que quepa.

Subir `T_max` cuesta: `store.T_MAX` y el validador de `TrainingRecipe` (`8 <= T_max <= 135`).
**Los dos ficheros estan en NINGUNA de las cuatro listas protegidas** (medido:
`coste_de_las_listas()`), y `feature_contract.MAX_TIMESTEPS = 256` ya permite hasta 256, asi que
**no se invalida ningun bundle portable**. Lo que si cambia es la entrada de la red, y eso exige
reentrenar y reexportar -- que es lo que se va a hacer de todos modos.

Recomendacion, y es una correccion al diseno propuesto: **`"recomendada"` (4.0/0.6) con `T_max = 140`.**
Es la rodaja MINIMA que cierra la unica brecha MEDIDA -- `post_s = 0,3` deja fuera la segunda luz, y
0,6 la captura al 100 % porque el maximo de `|dt_ms|` es 520 ms -- y conserva la ventana de guardia
entera. `"ancha"` (6.0/1.0) tambien funciona con `T_max = 211`, pero **cambia lo que ve la red** (7 s
en vez de 4,3) y **ese cambio no esta medido**: `pre_s = 4,0` se eligio porque 2,5 s recortaba el
inicio de la accion final en el 24 % de los clips, y nadie ha comparado 6,0 contra 4,0 sobre la red.
La atencion es O(T^2), asi que 211 cuadros son ~2,5x el coste de atencion de 135. Si se prueba,
pruebese como un brazo mas, pareado, contra `"recomendada"`.
"""


def plan_rodaja(rodaja_nombre: str = "recomendada", *, t_max: int | None = None,
                target_fps: float = 30.0, exigir: bool = True) -> dict:
    """Traduce una rodaja a cuadros y comprueba que `T_max` la admite. Levanta si no."""
    if rodaja_nombre not in RODAJAS:
        raise PreflightError(f"Rodaja desconocida: {rodaja_nombre!r}; elige una de {sorted(RODAJAS)}")
    pre_s, post_s = RODAJAS[rodaja_nombre]
    cuadros = int(round(pre_s * target_fps)) + int(round(post_s * target_fps)) + 1
    if t_max is None:
        t_max = cuadros
    try:
        from fencingly_v2 import feature_contract as FC
        tope_absoluto = int(FC.MAX_TIMESTEPS)
    except Exception:  # noqa: BLE001
        tope_absoluto = 256
    resultado = {"rodaja": rodaja_nombre, "pre_s": pre_s, "post_s": post_s, "cuadros": cuadros,
                 "t_max": int(t_max), "max_timesteps": tope_absoluto,
                 "cabe": cuadros <= int(t_max), "dentro_del_contrato": int(t_max) <= tope_absoluto,
                 "guardia_dentro": pre_s >= 4.0 and cuadros <= int(t_max),
                 "segunda_luz_completa": post_s >= 0.52}
    print(f"rodaja '{rodaja_nombre}': pre_s={pre_s} post_s={post_s} -> {cuadros} cuadros a {target_fps:g} fps")
    print(f"  T_max pedido {t_max} | store.T_MAX por defecto 135 | feature_contract.MAX_TIMESTEPS {tope_absoluto}")
    print(f"  {'OK ' if resultado['cabe'] else '!! '}la rodaja cabe en T_max")
    print(f"  {'OK ' if resultado['dentro_del_contrato'] else '!! '}T_max dentro de MAX_TIMESTEPS"
          " (por encima invalidaria los bundles portables)")
    print(f"  {'OK ' if resultado['guardia_dentro'] else '!! '}la ventana de guardia [t-4,0 : t-3,5] queda dentro")
    print(f"  {'OK ' if resultado['segunda_luz_completa'] else '!! '}post_s cubre el |dt_ms| maximo medido (520 ms)")
    if not resultado["cabe"]:
        print("\n" + "=" * 78)
        print(f"LA RODAJA NO CABE: {cuadros} cuadros contra T_max = {t_max}.")
        print("El Store se quedaria los ULTIMOS T_max cuadros de la rodaja, o sea"
              f" [t-{(int(t_max)-1)/target_fps - post_s:.2f} : t+{post_s:g}],")
        print("y la ventana de guardia [t-4,0 : t-3,5] se quedaria fuera. Sube T_max a"
              f" {cuadros} (store.T_MAX y el validador de TrainingRecipe: los dos estan en")
        print("ninguna de las cuatro listas protegidas) o elige una rodaja que quepa.")
        print("=" * 78, flush=True)
        if exigir:
            raise PreflightError(f"rodaja de {cuadros} cuadros contra T_max={t_max}: la guardia se pierde")
    if not resultado["dentro_del_contrato"] and exigir:
        raise PreflightError(f"T_max={t_max} supera MAX_TIMESTEPS={tope_absoluto}: subirlo invalidaria "
                             "todos los bundles portables")
    return resultado


def coste_de_las_listas(*, imprimir: bool = True) -> dict:
    """Que cuesta cambiar cada fichero candidato. Medido, no supuesto.

    Cuatro listas protegen cosas distintas y conviene no confundirlas:
      MEASUREMENT_SOURCES (20)  contrato de PROCEDENCIA dentro de cada artefacto materializado.
                                Cambiar uno invalida los 23.091 artefactos y `export_measurements`
                                se niega a sobrescribir historia: hay que subir la version del
                                espacio de trabajo.
      MOTION_SOURCES (11)       lo mismo para los artefactos de movimiento.
      FINGERPRINT_SOURCES (34)  invalida la CACHE de medidas del panel de coach. Se recalcula.
      SEMANTIC_SCOPE            mueve `configuration_sha256` e invalida TODOS los bundles portables.
    """
    from fencingly_v2 import measurement_artifacts as MA, coach_panel as CP, portable_contract as PC
    listas = {"los_20": set(MA.MEASUREMENT_SOURCES), "motion": set(MA.MOTION_SOURCES),
              "fingerprint": set(CP.FINGERPRINT_SOURCES), "semantic_scope": set(PC.SEMANTIC_SCOPE)}
    candidatos = ("store.py", "clip_quality.py", "coach_panel.py", "measurement_artifacts.py",
                  "feature_contract.py", "training_recipe.py", "colab_workflow.py")
    fuera = {f: sorted(k for k, v in listas.items() if f in v) for f in candidatos}
    if imprimir:
        print(f"{'fichero':26} {'los 20':>8} {'motion':>8} {'huella':>8} {'bundles':>9}   coste de cambiarlo")
        coste = {"store.py": "NINGUNO. No esta en ninguna lista",
                 "clip_quality.py": "re-cachear el panel de coach",
                 "coach_panel.py": "RE-MATERIALIZAR los 23.091 artefactos",
                 "measurement_artifacts.py": "RE-MATERIALIZAR los 23.091 artefactos",
                 "feature_contract.py": "INVALIDA TODOS LOS BUNDLES PORTABLES",
                 "training_recipe.py": "ninguno (no esta en ninguna lista)",
                 "colab_workflow.py": "ninguno (no esta en ninguna lista)"}
        for f in candidatos:
            marcas = ["si" if k in fuera[f] else "-" for k in ("los_20", "motion", "fingerprint", "semantic_scope")]
            print(f"{f:26} {marcas[0]:>8} {marcas[1]:>8} {marcas[2]:>8} {marcas[3]:>9}   {coste[f]}")
        print("\n`store.py` es el sitio mas barato del conducto entero para poner la rodaja: no esta en")
        print("ninguna de las cuatro listas. Solo aparece en `feature_contract.source_provenance()`,")
        print("que se graba DENTRO de `store.json`, asi que el unico efecto es que un Store nuevo")
        print("declara una procedencia nueva. Que es lo correcto: el protocolo es nuevo.")
    return fuera


def rodaja(sample: dict, pre_s: float, post_s: float, minimo: int = 12) -> dict:
    """Recorta una muestra cruda de NPZ a `[tocado - pre_s, tocado + post_s]`.

    Es `coach_panel._ventana_de_analisis` con `pre_s`/`post_s` parametrizados, escrito aparte a
    proposito: alli esos dos valores NO son perillas ajustables (subirlos desplazaria la ventana de
    los 23.011 clips ya medidos sin que nada lo avise), y aqui hacen falta variables porque la
    rodaja de ENTRENAMIENTO es una decision distinta de la ventana de ALMACEN.

    Se recorta cualquier array cuya primera dimension sea la del clip, para que anadir un campo
    nuevo al NPZ no deje una serie temporal fuera del recorte por olvido. Si el recorte no quita
    nada, devuelve la muestra tal cual, asi que sobre un NPZ de ventana desplegada es un no-op.
    """
    import numpy as np
    poses = sample.get("poses")
    meta = sample.get("meta") or {}
    fps = meta.get("fps")
    t0, tt = meta.get("t_window_start_s"), meta.get("t_touch_s")
    if poses is None or not fps or t0 is None or tt is None:
        return sample
    total = int(np.shape(poses)[0])
    try:
        cuadro = int(round((float(tt) - float(t0)) * float(fps)))
    except (TypeError, ValueError):
        return sample
    a = max(0, int(cuadro - pre_s * float(fps)))
    b = min(total, int(cuadro + post_s * float(fps)) + 1)
    if b - a < minimo or (a == 0 and b == total):
        return sample
    salida = {}
    for clave, valor in sample.items():
        if clave == "meta":
            continue
        try:
            arr = np.asarray(valor)
        except Exception:  # noqa: BLE001
            salida[clave] = valor
            continue
        salida[clave] = arr[a:b] if (arr.ndim >= 1 and arr.shape[0] == total) else valor
    nueva = dict(meta)
    f0 = int(meta.get("f0") or 0)
    nueva["f0"] = f0 + a
    nueva["f1"] = f0 + b
    nueva["n_frames_window"] = int(b - a)
    nueva["t_window_start_s"] = float(t0) + a / float(fps)
    nueva["t_window_end_s"] = nueva["t_window_start_s"] + (b - a) / float(fps)
    nueva["pre_s_eff"] = round((cuadro - a) / float(fps), 2)
    nueva["post_s_requested"] = float(post_s)
    nueva["training_window_selected"] = [int(a), int(b)]
    nueva["training_window_pre_post_s"] = [float(pre_s), float(post_s)]
    salida["meta"] = nueva
    return salida


def vista_allez(sample: dict, *, bias_s: float = 0.5) -> dict:
    """Re-recorta la muestra para que EMPIECE en el allez, y solo para `calibration.allez_scale`.

    POR QUE HACE FALTA, medido. `allez_scale` (calibration.py:333-341) hace

        t_win0 = meta["f0"] / meta["fps"]
        lag    = t_win0 - clock_running_s[0]
        if abs(lag) > ALLEZ_TOL_S (0,7):  reject = "window_not_at_clock_start"

    o sea que exige que la ventana EMPIECE en el arranque del reloj, y `f0` esta grabado en el NPZ,
    asi que ni la fila del indice ni el meta lo alcanzan: solo re-extraer, o re-recortar.

    Las tres configuraciones, comprobadas sobre 6 clips reales con allez legible:

        f0 = tocado - 4,0 s  (lo desplegado hoy)   2 de 6 pasan   (lag -1,83 a +2,80 s)
        f0 = 0               (DURACION COMPLETA)   0 de 6 pasan   (lag = -clock0, -1,56 a -9,90 s)
        f0 = f_allez         (esta vista)          6 de 6 pasan   (lag = +0,50 s exacto)

    O sea que **la duracion completa por si sola MATA `allez_scale`**, porque la ventana empieza en
    el cuadro 0 y el allez cae 1,6 a 9,9 s dentro. Y **esta vista lo recupera entero**, con lag
    +0,50 s por construccion (el `CLOCK_BIAS_S` de `lights.allez_frame`), holgado bajo la
    tolerancia de 0,7. La duracion completa es la unica configuracion que permite elegir el ancla
    DESPUES, asi que es la unica en la que se pueden tener las dos vistas.

    Es una vista SEPARADA de la rodaja de entrenamiento, y las dos son necesarias:
      - rodaja anclada en el TOCADO -> el Store y el camino de medidas (los 25 sitios de guardia
        asumen que el cuadro 0 esta `pre_s` antes de la luz);
      - esta vista, anclada en el ALLEZ -> solo `allez_scale`, para recuperar la escala metrica.
    Mezclarlas romperia una de las dos.

    Devuelve la muestra tal cual si no hay reloj legible, si el allez esta censurado
    (`clock_running_s[0] <= 0,05`) o si no queda ventana suficiente.
    """
    import numpy as np
    meta = sample.get("meta") or {}
    poses = sample.get("poses")
    fps = meta.get("fps")
    clock = meta.get("clock_running_s")
    if poses is None or not fps or not isinstance(clock, (list, tuple)) or len(clock) < 1 or clock[0] is None:
        return sample
    clock0 = float(clock[0])
    if clock0 <= 0.05:                      # allez anterior al clip: no hay allez que anclar
        return sample
    f0_actual = int(meta.get("f0") or 0)
    f_allez = int(round((clock0 + bias_s) * float(fps)))
    total = int(np.shape(poses)[0])
    a = f_allez - f0_actual                 # indice dentro del array actual
    if a < 0 or total - a < 12:
        return sample
    salida = {}
    for clave, valor in sample.items():
        if clave == "meta":
            continue
        try:
            arr = np.asarray(valor)
        except Exception:  # noqa: BLE001
            salida[clave] = valor
            continue
        salida[clave] = arr[a:] if (arr.ndim >= 1 and arr.shape[0] == total) else valor
    nueva = dict(meta)
    nueva["f0"] = f0_actual + a             # == f_allez, que es lo que `allez_scale` mira
    nueva["f1"] = f0_actual + total
    nueva["n_frames_window"] = int(total - a)
    nueva["t_window_start_s"] = f_allez / float(fps)
    nueva["t_window_end_s"] = nueva["t_window_start_s"] + (total - a) / float(fps)
    nueva["allez_view_selected"] = [int(a), int(total)]
    nueva["allez_view_lag_s"] = round(f_allez / float(fps) - clock0, 3)
    salida["meta"] = nueva
    return salida


def escala_allez(sample: dict) -> dict:
    """`calibration.allez_scale` sobre la vista del allez, con el diagnostico de la compuerta."""
    from fencingly_v2 import calibration as CAL, features as F
    vista = vista_allez(sample)
    meta = vista.get("meta") or {}
    fps = float(meta.get("fps") or 30.0)
    poses = F.sanitize_poses(vista["poses"])
    resultado = CAL.allez_scale(vista, poses, fps)
    resultado["lag_declarado_s"] = meta.get("allez_view_lag_s")
    resultado["f0_vista"] = meta.get("f0")
    resultado["vista_aplicada"] = "allez_view_selected" in meta
    return resultado


def embudo_elegibles(index, *, min_coverage: float = 0.6, exigir: bool = True) -> dict:
    """Cuenta el embudo de `build_current_store` ANTES de llamarlo, y dice donde se vacia.

    ENCONTRADO EN EL ENSAYO. Con el conjunto elegible vacio, `build_current_store` no dice "0 clips
    elegibles": levanta `KeyError('coverage_L')`, porque sigue filtrando sobre un DataFrame sin
    filas hasta que una seleccion de columnas se queda sin nada. Despues de horas de GPU en Colab,
    ese mensaje no dice nada de lo que hay que arreglar.

    El embudo real, en el orden en que filtra `build_current_store`:
        status in (ok, cached) -> label_ok == "true" -> qc_blocking vacio
        -> coverage_L/R >= min_coverage -> y in (0, 1)
    """
    import pandas as pd
    frame = index if hasattr(index, "columns") else pd.DataFrame(index)
    pasos = [("filas del indice", len(frame))]
    actual = frame
    if "status" in actual:
        actual = actual[actual["status"].isin(("ok", "cached"))]
    pasos.append(("status ok|cached", len(actual)))
    if "label_ok" in actual:
        actual = actual[actual["label_ok"].astype(str).str.lower().eq("true")]
    pasos.append(("label_ok == true", len(actual)))
    if "qc_blocking" in actual:
        bloqueado = actual["qc_blocking"].map(
            lambda v: bool(v) if isinstance(v, list) else str(v) not in ("[]", "", "nan", "None"))
        actual = actual[~bloqueado]
    pasos.append(("qc_blocking vacio", len(actual)))
    for col in ("coverage_L", "coverage_R"):
        if col in actual:
            actual = actual[pd.to_numeric(actual[col], errors="coerce") >= min_coverage]
    pasos.append((f"coverage >= {min_coverage:g}", len(actual)))
    if "y" in actual:
        actual = actual[actual["y"].isin((0, 1))]
    pasos.append(("y in (0, 1)", len(actual)))
    print("embudo de elegibilidad del Store:")
    previo = None
    culpable = None
    for nombre, n in pasos:
        caida = "" if previo is None else f"  (-{previo - n})"
        print(f"   {nombre:24} {n:7d}{caida}")
        if previo is not None and n == 0 and previo > 0 and culpable is None:
            culpable = nombre
        previo = n
    resultado = {"pasos": pasos, "elegibles": pasos[-1][1], "culpable": culpable}
    if resultado["elegibles"] == 0:
        print("\n" + "=" * 78)
        print("NO QUEDA NINGUN CLIP ELEGIBLE" + (f", y se vacia en: {culpable}" if culpable else "."))
        print("`build_current_store` levantaria KeyError('coverage_L') aqui, que no dice esto.")
        if culpable == "label_ok == true":
            print("`label_ok` lo RECALCULA `dataset_update.prepare_manifest` desde el manifiesto, no")
            print("se toma del que se le pase: necesita las columnas de marcador coherentes. Si el")
            print("manifiesto se construyo a mano, esto es lo que falla, no la extraccion.")
        print("=" * 78, flush=True)
        if exigir:
            raise PreflightError(
                f"0 clips elegibles para el Store (se vacia en: {culpable}). "
                "No es un fallo de extraccion: es el embudo de etiquetas.")
    return resultado


def alineacion_del_store(store_dir, *, post_s_esperado: float, exigir: bool = True,
                         tolerancia_s: float = 0.25) -> dict:
    """PUERTA DURA. Mide donde queda el tocado dentro de la ventana guardada del Store.

    `build_store` alinea por el final del array. Con un NPZ de duracion completa el array termina al
    final del CLIP, no del tocado, asi que el Store queda desalineado sin dar error (ver
    `COLA_TRAS_TOCADO`). Esto lo mide con los campos que `sample_meta.jsonl` trae de verdad
    -- `prepared_time_start_s`, `prepared_n_frames`, `t_touch_s` -- y no con
    `analysis_touch_frame`, que en los Store reales sale None (comprobado: 0 de 211).

        distancia_s = (prepared_time_start_s + (prepared_n_frames - 1) / fps) - t_touch_s

    Verificado sobre un Store real de ventana desplegada: 0,267 s contra `post_s_requested` 0,3.
    """
    store_dir = Path(store_dir)
    info = json.loads((store_dir / "store.json").read_text(encoding="utf-8"))
    fps = float(info.get("fps") or 30.0)
    filas = [json.loads(l) for l in (store_dir / "sample_meta.jsonl").open(encoding="utf-8")]
    distancias, truncados, sin_datos = [], 0, 0
    for r in filas:
        t0, n, tt = r.get("prepared_time_start_s"), r.get("prepared_n_frames"), r.get("t_touch_s")
        if r.get("store_truncated") in (True, "True", "true"):
            truncados += 1
        if t0 is None or not n or tt is None:
            sin_datos += 1
            continue
        distancias.append((float(t0) + (int(n) - 1) / fps) - float(tt))
    distancias.sort()
    n = len(distancias)
    resultado = {"N": info.get("N"), "T_max": info.get("T_max"), "fps": fps, "muestras": len(filas),
                 "sin_datos": sin_datos, "store_truncated": truncados,
                 "post_s_esperado": float(post_s_esperado)}
    if not n:
        print("!! No se pudo medir la alineacion: faltan campos en sample_meta.jsonl")
        if exigir:
            raise PreflightError("alineacion del Store no verificable")
        return resultado

    def q(f):
        return distancias[int(f * (n - 1))]

    resultado.update(dist_min=distancias[0], dist_p05=q(.05), dist_p50=q(.5),
                     dist_media=sum(distancias) / n, dist_p95=q(.95), dist_max=distancias[-1])
    desviados = sum(1 for d in distancias if abs(d - float(post_s_esperado)) > tolerancia_s)
    resultado["desviados"] = desviados
    resultado["fraccion_desviada"] = desviados / n
    print(f"Store: N={resultado['N']} T_max={resultado['T_max']} fps={fps:g}")
    print(f"store_truncated=True en {truncados} de {len(filas)} muestras"
          f" ({100*truncados/max(1,len(filas)):.1f} %)")
    print(f"distancia tocado -> ultimo cuadro guardado (s), esperada {post_s_esperado:g}:")
    print(f"   min {resultado['dist_min']:.3f} | p05 {resultado['dist_p05']:.3f} |"
          f" MEDIANA {resultado['dist_p50']:.3f} | media {resultado['dist_media']:.3f} |"
          f" p95 {resultado['dist_p95']:.3f} | max {resultado['dist_max']:.3f}")
    print(f"   fuera de +-{tolerancia_s:g} s de lo esperado: {desviados} de {n}"
          f" ({100*desviados/n:.2f} %)")
    if truncados:
        print("\n!! `store_truncated=True` enmascara TODA etiqueta auxiliar de esas muestras")
        print("   (codigo -1, mask 0, conf 0) por `store.py:244` y `store.py:417`, sin aviso.")
    if resultado["fraccion_desviada"] > 0.05:
        p50 = resultado["dist_p50"]
        # Dos causas distintas, y confundirlas manda a arreglar lo que no esta roto:
        #  (a) el Store es de OTRO protocolo -> su p50 coincide con el post_s de otra rodaja
        #  (b) no se aplico la rodaja        -> el p50 se va muy por encima (la cola del clip)
        otra = next((nombre for nombre, (_, ps) in RODAJAS.items() if abs(p50 - ps) <= tolerancia_s), None)
        print("\n" + "=" * 78)
        if otra is not None:
            resultado["causa"] = f"store_de_otro_protocolo:{otra}"
            print(f"ESTE STORE ES DE OTRO PROTOCOLO. Su distancia mediana ({p50:.3f} s) es la de la")
            print(f"rodaja '{otra}' (post_s = {RODAJAS[otra][1]:g}), no la de la que se ha declarado")
            print(f"(post_s = {post_s_esperado:g}). No esta roto: es de otra cadena de medida.")
            print("Construye un Store nuevo con la rodaja declarada, o declara la rodaja que le")
            print("corresponde. Lo que NO se puede es mezclar los dos y restar sus numeros.")
        else:
            resultado["causa"] = "rodaja_no_aplicada"
            print("EL STORE ESTA DESALINEADO. El ultimo cuadro guardado no es el tocado.")
            print("`feature_contract` declara `alignment: right` y `truncation: tail`, y esa")
            print("declaracion acaba de dejar de ser cierta: la ventana de guardia [t-4,0 : t-3,5]")
            print(f"no esta dentro, y la distancia mediana es {p50:.3f} s en vez de {post_s_esperado:g}.")
            print("Causa: se extrajo con VENTANA_ALMACEN ancha y el Store se construyo desde los NPZ")
            print("SIN recortar. Hay que pasar por `PF.rodaja_a_disco(...)` y construir desde el")
            print("directorio recortado (o usar `PF.construir_store_con_rodaja`), porque si no")
            print("`build_store` alinea por el final del CLIP y no del tocado (ver COLA_TRAS_TOCADO:")
            print("la cola tras el tocado tiene mediana 2,33 s y p95 5,23 s).")
            print("Medido en el ensayo sobre 6 NPZ de duracion completa: con rodaja, mediana 0,566 s")
            print("y `store_truncated` 0 de 6; sin rodaja, mediana 1,633 s (max 3,900) y 6 de 6.")
        print("=" * 78, flush=True)
        if exigir:
            raise PreflightError(
                f"Store no conforme ({resultado['causa']}): la distancia mediana tocado->final es "
                f"{p50:.3f} s y la rodaja declarada pide {post_s_esperado:g} s. NO entrenes sobre este Store.")
    else:
        print("\nAlineacion correcta: el ultimo cuadro guardado es el tocado (+ post_s).")
    return resultado


# --------------------------------------------------------------------------------------------------
# 6.ter La rodaja SIN tocar ninguna fuente, y la columna del reloj
# --------------------------------------------------------------------------------------------------

ENVOLTORIO_PROHIBIDO = """\
NO ENVOLVER `dataset.load_sample` PARA APLICAR LA RODAJA. Se probo y ENVENENA EL STORE.

La idea era tentadora: `store.build_store` carga cada clip con `D.load_sample(npz)` y no acepta
recorte, asi que envolver esa funcion durante la construccion aplicaria la rodaja sin tocar ninguna
fuente. Se implemento, se ejecuto sobre 300 clips reales, y el Store salio con la alineacion
PERFECTA (mediana 0,267 s, `store_truncated` 0 de 300). Y aun asi esta ROTO:

    ST.Store(destino)
    -> FC.validate_runtime_compatibility(info)
    -> ContractError('Bindings runtime incompatibles: dataset.py')

`build_store` graba en `store.json` el `FC.runtime_compatibility()` DEL MOMENTO DE CONSTRUIR, y esa
funcion hashea `_module_bindings_sha256(module)` de cada fichero de `SEMANTIC_SOURCE_FILES`, entre
ellos `dataset.py`. Con el envoltorio puesto, el hash de los bindings es OTRO, y queda congelado en
`store.json`. Al abrirlo despues con un `dataset.py` limpio, el hash no coincide y **el Store no se
puede abrir nunca mas**. No es un fallo del proceso: es permanente, esta escrito en el fichero.

Comprobado en el mismo proceso: el Store del corpus (construido sin envoltorio) abre con N=14.437,
y el que construi con envoltorio levanta `ContractError`.

Yo habia comprobado que los cuadernos no llaman a `load_bundle` ni a `save_bundle`, que era la via
que conocia de `portable_contract.SEMANTIC_SCOPE`. Lo que no comprobe es que `build_store` llama a
`feature_contract.runtime_compatibility()` por su cuenta, en la linea que escribe `store.json`. La
leccion: la validacion de bindings runtime no esta solo en el contrato portable.

La via correcta es `rodaja_a_disco`: escribir los NPZ recortados en un directorio NUEVO y construir
el Store desde ahi, con `load_sample` intacto de principio a fin.
"""


def rodaja_a_disco(npz_origen, npz_destino, pre_s: float, post_s: float, *, clips=None,
                   progreso: int = 200) -> dict:
    """Escribe los NPZ recortados a la rodaja en un directorio NUEVO. `load_sample` queda intacto.

    Es la via correcta para aplicar la rodaja, y la razon esta en `ENVOLTORIO_PROHIBIDO`: parchear
    `dataset.load_sample` en caliente envenena el `compatibility` que `build_store` graba en
    `store.json` y deja el Store inabrible para siempre.

    Cuesta disco: la rodaja de 4.0/0.6 son ~139 cuadros, o sea ~25 KiB por clip con el modelo medido
    (`bytes = 6.005 + 138,4*T`), unos 0,6 GiB para 25.139 clips de florete. Sobre los ~1,2 GiB del
    almacen a duracion completa, el total queda en ~1,8 GiB. Sigue siendo irrelevante frente a los
    93 GB de mp4 que se evitan redescargar.

    Sobre un NPZ de ventana desplegada el recorte es un no-op y el NPZ se copia tal cual.
    """
    import numpy as np
    from fencingly_v2 import dataset as D, pipeline as PL
    origen, destino = Path(npz_origen), Path(npz_destino)
    destino.mkdir(parents=True, exist_ok=True)
    fuentes = sorted(origen.glob("*.npz")) if clips is None else [origen / f"{c}.npz" for c in clips]
    hechos = recortados = copiados = fallos = 0
    for i, p in enumerate(fuentes):
        if not p.is_file():
            fallos += 1
            continue
        salida = destino / p.name
        if salida.exists():
            hechos += 1
            continue
        try:
            muestra = D.load_sample(p)
            cortada = rodaja(muestra, pre_s, post_s)
            if cortada is muestra:                       # no-op: el NPZ ya cabe en la rodaja
                shutil.copyfile(p, salida)
                copiados += 1
            else:
                arrays = {}
                for clave, valor in cortada.items():
                    if clave == "meta" or valor is None:
                        continue
                    arrays[clave] = np.asarray(valor)
                arrays["meta_json"] = np.array(json.dumps(cortada["meta"], ensure_ascii=False,
                                                          allow_nan=False))
                PL.save_npz(salida, arrays, overwrite=False)
                recortados += 1
            hechos += 1
        except Exception:  # noqa: BLE001
            fallos += 1
        if progreso and (i + 1) % progreso == 0:
            print(f"   rodaja a disco {i + 1}/{len(fuentes)}", flush=True)
    resultado = {"origen": str(origen), "destino": str(destino), "pre_s": pre_s, "post_s": post_s,
                 "hechos": hechos, "recortados": recortados, "copiados_sin_recorte": copiados,
                 "fallos": fallos}
    print(f"rodaja a disco [{pre_s:g}/{post_s:g}]: {hechos} NPZ ({recortados} recortados, "
          f"{copiados} no-op copiados, {fallos} fallos) -> {destino.name}")
    return resultado


def construir_store_con_rodaja(workspace, split_registry, *, rodaja_nombre="recomendada",
                               t_max=140, min_coverage=0.6, weapon="foil", store_dir=None,
                               npz_rodaja=None) -> dict:
    """`build_current_store` pero sobre los NPZ ya recortados, con `load_sample` intacto.

    Replica el embudo de `colab_workflow.build_current_store` -- que fija el `npz_dir` a
    `work/"npz"` y por eso no sirve aqui -- conservando sus dos guardas: `SC.validate_splits` y el
    `eligible_index.csv` que no se reescribe. Se anade `embudo_elegibles`, que nombra el paso que
    vacia la seleccion en vez de levantar `KeyError('coverage_L')`.
    """
    import pandas as pd
    from fencingly_v2 import colab_workflow as CW, split_contract as SC, store as ST
    work = Path(workspace)
    pre_s, post_s = RODAJAS[rodaja_nombre]
    plan_rodaja(rodaja_nombre, t_max=t_max, exigir=True)
    indice = CW.finalize_data_index(work, split_registry, weapon=weapon)
    embudo = embudo_elegibles(indice, min_coverage=min_coverage, exigir=True)
    elegibles = indice[indice["status"].isin(("ok", "cached"))
                       & indice["label_ok"].astype(str).str.lower().eq("true")].copy()
    if "qc_blocking" in elegibles:
        bloqueado = elegibles["qc_blocking"].map(
            lambda v: bool(v) if isinstance(v, list) else str(v) not in ("[]", "", "nan", "None"))
        elegibles = elegibles[~bloqueado]
    elegibles = elegibles[(pd.to_numeric(elegibles["coverage_L"], errors="coerce") >= min_coverage)
                          & (pd.to_numeric(elegibles["coverage_R"], errors="coerce") >= min_coverage)]
    elegibles = elegibles[elegibles["y"].isin((0, 1))]
    columnas = list(dict.fromkeys(["status", *ST.META_COLS, "flag", "touch_lights", "video_sha256",
                                   "duplicate_group_id", "qc_blocking", "qc_score", "qc_valid"]))
    elegibles = elegibles[[c for c in columnas if c in elegibles]].copy()
    diagnostico = SC.validate_splits(elegibles)
    destino_csv = work / "training" / f"eligible_index_{rodaja_nombre}.csv"
    destino_csv.parent.mkdir(parents=True, exist_ok=True)
    payload = elegibles.to_csv(index=False).encode("utf-8")
    if destino_csv.exists() and destino_csv.read_bytes() != payload:
        raise PreflightError("La seleccion de entrenamiento cambio; crea un workspace/Store nuevo")
    if not destino_csv.exists():
        with destino_csv.open("xb") as salida:
            salida.write(payload)
    npz_rodaja = Path(npz_rodaja) if npz_rodaja else work / f"npz_rodaja_{rodaja_nombre}"
    corte = rodaja_a_disco(work / "npz", npz_rodaja, pre_s, post_s,
                           clips=elegibles["clip_id"].astype(str).tolist())
    store_dir = Path(store_dir) if store_dir else work / f"store_{rodaja_nombre}"
    if store_dir.exists():
        with ST.Store(store_dir) as tienda:
            return {"store": str(store_dir), "n": len(tienda), "reused": True,
                    "splits": diagnostico, "embudo": embudo, "rodaja": corte}
    meta = ST.build_store(destino_csv, npz_rodaja, store_dir, T_max=t_max,
                          min_coverage=min_coverage, progress=False)
    with ST.Store(store_dir) as tienda:          # tiene que ABRIR: es la prueba del envenenamiento
        SC.validate_splits(tienda.meta)
        n = len(tienda)
    alineacion = alineacion_del_store(store_dir, post_s_esperado=post_s, exigir=True)
    return {"store": str(store_dir), "n": n, "reused": False, "splits": diagnostico,
            "embudo": embudo, "rodaja": corte, "alineacion": alineacion, "filas": len(meta)}


RELOJ = """\
LA TABLA DEL RELOJ: que compra de verdad, medido.

1) LA POSICION ABSOLUTA DE PISTA YA ESTA DESBLOQUEADA, y no hace falta cambiar nada.

   `piste_abs.allez_anchor` (piste_abs.py:216) hace:

       clk = parse_list(_get(row, "clock_running_s")) or meta.get("clock_running_s")

   o sea que lee LA FILA DEL INDICE ANTES QUE EL META. Verificado leyendo la linea. Consecuencia:
   el reloj le llega por `npz/index.csv`, sin reextraer, sin NPZ nuevos y SIN CAMBIAR LA HUELLA
   (`piste_abs.py` no esta en ninguna de las cuatro listas protegidas: comprobado).

   Y la columna YA LA EMITE el conducto actual: `clock_running_s` es la columna 45 de las 119 de
   `npz/index.csv` en el corpus de hoy. Asi que esto no es un cambio que hacer, es un hecho que
   COMPROBAR por espacio de trabajo. Medido sobre los 15.452 clips ok|cached con .npz:

       con `clock_running_s` legible          9.865  (63,84 %)
       menos allez censurado (clk[0] <= 0,05) -2.640
       menos allez tras el tocado                -64
       ANCLA DE ALLEZ UTILIZABLE              7.161  (46,34 %)

   Hoy `x_abs`, `dist_m` y `dist_allez_*_m` son NaN en el 100 % de esas extracciones, asi que son
   7.161 clips que ganan posicion absoluta de pista a coste cero. `columna_reloj()` lo comprueba y
   lo cuenta en el espacio de trabajo que se le pase, en vez de suponerlo.

2) APAGAR LA RAMA DEL RELOJ: la cobertura de guardia pasa de 72,78 % a 91,42 %, al mismo coste
   exacto de decodificacion (129 cuadros, 680 MiB, +0,03 pp de clips bloqueados). Es la ganancia
   mas barata de todo el barrido. Mando: `USAR_RELOJ`.

3) LO QUE NO COMPRA, y conviene no inflarlo:
   - anclar NO anade ni un clip: lo que ancla es la lampara;
   - marcar repeticiones NO quita ninguno: ya estan fuera por politica.

4) `reloj_parado_todo_el_clip`: DEFINICION RECIBIDA, y se implementa llamando al lector de
   referencia, no reinterpretandola. Ver `reloj_contrato()`. Prevalencia 15,1 % [12,5, 18,2] con
   ancla FIE (n=602) contra 16,5 % [12,0, 22,3] con ancla de region (n=200): los intervalos se
   solapan por completo, asi que es una propiedad del CORPUS ENTERO y no de las cinco sedes de
   region. Control de falsos 2,2 % [1,2, 3,9] (Wilson 95 %, n=500), y SOLO cubre el perfil `fie`;
   los otros cuatro se respaldan en revision visual de los digitos ampliados.

   Y por que mi cero era correcto: `clock_running_s = [inicio, fin]` es el intervalo en que el
   reloj SI corre, con `inicio = primer_tic_regular - 1 s`, y solo se construye cuando existe un
   hueco de 0,7 a 1,4 s. Asi que TODO intervalo que existe mide >= 1 s por construccion y
   `|fin - inicio| < 0,05` no podia dar nada. Un reloj parado no da un intervalo degenerado: da
   NULL. Y el atajo `clock_running_s IS NULL` TAMPOCO vale, porque NULL mezcla tres causas:
   reloj parado (lo que se quiere), cajas de nombre no localizadas (hueco estructural de 5 sedes)
   y recorte ilegible (`qatar`, `incheon`, indistinguible).

5) NO CITAR: `qatarfoil2021` 63,4 % y `incheonfoil2022` 38,1 % de reloj parado estan SIN VALIDAR.
   En esas dos sedes el lector FIE tampoco lee el reloj (14 de 40 y 13 de 21 clips con reloj de
   referencia), y sin referencia no se distingue "el reloj estaba parado" de "el reloj no se lee en
   esta sede": las dos producen `frac_corre = 0`.

6) AVISO DE INTERACCION: el recorte de ventana por reloj y el barrido de ventana SE PISAN. Con
   `pre_s = 20,0` la ventana de un clip de region es hoy el clip entero. Hay que medirlos JUNTOS,
   en factorial, no uno encima del otro.

7) DOS INDICES DISTINTOS, y leer el equivocado hace concluir que el reloj falta cuando esta en el
   meta: el `npz/index.csv` VIVO lleva `clock_running_s` (columna 45 de 119), pero el que escribe
   `extrae.py` tiene 47 columnas y NO la lleva, porque su lista fija de columnas no la incluye
   aunque `pipeline.stage_save` si la mete en el `meta_json` del NPZ.

8) Y A QUIEN ALIMENTA UNA FILA NUEVA, que no es a todos: `piste_abs.allez_anchor` lee la FILA antes
   que el meta (linea 216), asi que una fila nueva le llega sin reextraer. `calibration.allez_scale`
   lee SOLO del meta (`meta.get("clock_running_s")`, linea 333, y su firma no recibe `row`), asi que
   a esa NO le llega. No prometer "desbloquea la escala de allez": desbloquea `piste_abs`.
"""


def columna_reloj(workspace, *, exigir: bool = False) -> dict:
    """Comprueba que `npz/index.csv` trae `clock_running_s` y cuenta a cuantos clips desbloquea.

    No escribe nada. Si la columna faltara, el arreglo es emitirla al cerrar cada shard, igual que
    el resto del indice; no hay cambio de codigo porque `piste_abs` ya la lee de la fila.
    """
    import ast as _ast
    import csv as _csv
    work = Path(workspace)
    npz_dir = work / "npz"
    indice = npz_dir / "index.csv"
    if not indice.is_file():
        print("!! no hay npz/index.csv; ver indice_real()")
        if exigir:
            raise PreflightError("sin npz/index.csv no hay reloj ni indice")
        return {"index_csv": False}
    en_disco = {p.stem for p in npz_dir.glob("*.npz")}
    ok = con = cens = despues = usable = 0
    with indice.open(encoding="utf-8", newline="") as fh:
        lector = _csv.DictReader(fh)
        presente = "clock_running_s" in (lector.fieldnames or [])
        for fila in lector:
            cid = (fila.get("clip_id") or "").strip()
            if cid not in en_disco or str(fila.get("status", "")).lower() not in ("ok", "cached"):
                continue
            ok += 1
            valor = (fila.get("clock_running_s") or "").strip()
            if not valor or valor.lower() in ("nan", "none", "[]"):
                continue
            try:
                inicio, _fin = _ast.literal_eval(valor)
            except Exception:  # noqa: BLE001
                continue
            con += 1
            if inicio is None:
                continue
            inicio = float(inicio)
            if inicio <= 0.05:
                cens += 1
                continue
            try:
                tocado = float(fila.get("t_touch_s"))
            except (TypeError, ValueError):
                tocado = None
            if tocado is not None and inicio + 0.5 > tocado - 0.4:
                despues += 1
                continue
            usable += 1
    resultado = {"index_csv": True, "columna_presente": presente, "clips_ok": ok, "con_reloj": con,
                 "allez_censurado": cens, "allez_tras_tocado": despues, "ancla_utilizable": usable}
    print(f"{'OK ' if presente else '!! '}npz/index.csv trae `clock_running_s`: {presente}")
    print(f"   clips ok|cached con .npz          : {ok}")
    print(f"   con clock_running_s legible       : {con}" + (f" ({100*con/ok:.2f} %)" if ok else ""))
    print(f"   menos allez censurado             : -{cens}")
    print(f"   menos allez tras el tocado        : -{despues}")
    print(f"   ANCLA DE ALLEZ UTILIZABLE         : {usable}" + (f" ({100*usable/ok:.2f} %)" if ok else ""))
    print("   Esos son los clips que ganan x_abs / dist_m / dist_allez_*_m, que hoy son NaN en el")
    print("   100 % de las extracciones. Coste: cero. `piste_abs:216` lee la fila antes que el meta.")
    if not presente:
        print("   Arreglo: emitir la columna al cerrar cada shard. NO hay cambio de codigo.")
        if exigir:
            raise PreflightError("falta clock_running_s en npz/index.csv")
    return resultado


CONTRATO_RELOJ = ("clip_id", "tournament_id", "perfil", "clock_running_s", "clock_source",
                  "f_allez", "t_allez_s", "reloj_frac_corre", "reloj_parado_todo_el_clip",
                  "reloj_umbral", "reloj_n_tics", "reloj_caja")
"""Los 12 campos del contrato de salida de DEFINICION_RELOJ_PARADO.md seccion 8, en su orden.

Nombres exactos, para que una fila producida en Colab y una producida por `trabajo/paso1.py`
signifiquen lo mismo y se puedan unir por `clip_id`. `clock_running_s` va como texto "[ini, fin]",
el mismo formato de `lights._clock_interval`, para ser intercambiable con las 9.865 filas que ya lo
llevan. `reloj_parado_todo_el_clip` va VACIO cuando es indefinida (seccion 2.6): escribir `false`
afirmaria que el reloj corria, y eso no se ha medido.
"""

PARADO_MAX = 0.05
"""Corte de `frac_corre` para la bandera. Va acompanado de `reloj_frac_corre` SIEMPRE, porque el
corte es una decision y alguien querra moverlo. 0,05 y no 0: es holgura de un solo hueco corto en un
clip largo, y la distribucion es bimodal con el valle ahi."""

SEDES_SIN_PERFIL_VALIDADO = ("qatarfoil2021", "incheonfoil2022")
"""Seccion 0: sin reloj de referencia no se distingue "parado" de "ilegible". Bandera INDEFINIDA."""


def reloj_contrato(video_path, clip_id, tournament_id, *, ruta_reloj=None, caja=None) -> dict:
    """Emite los 12 campos del contrato para un clip, LLAMANDO AL LECTOR DE REFERENCIA.

    No reimplementa el criterio a proposito. El documento de definicion es explicito en que dos
    cosas producen "un numero distinto con el mismo nombre": el submuestreo (con paso 2 `frac_corre`
    cambia en 5 de 6 clips y uno cambia de clase) y un umbral que no sea el del perfil (con el
    unico 0,04 heredado del lector FIE la prevalencia salia 53,1 % en vez de 16,4 %). Copiar la
    logica aqui seria darle otra oportunidad a ese error, asi que se importa
    `scratchpad/reloj/trabajo/reloj.py` y `scratchpad/detector/trabajo/ancla_region.py`, que son la
    definicion, y se emiten sus valores.

    Si los modulos no estan disponibles, LEVANTA en vez de aproximar. Para Colab hay que anadirlos
    al release; son la definicion, no un parche.
    """
    import numpy as np
    reloj_mod, ar_mod = _importar_reloj(ruta_reloj)
    perfil = ar_mod.perfil_de(tournament_id, None)
    fila = {campo: "" for campo in CONTRATO_RELOJ}
    fila.update(clip_id=str(clip_id), tournament_id=str(tournament_id), perfil=perfil,
                clock_source=f"region:{perfil}")
    if str(tournament_id) in SEDES_SIN_PERFIL_VALIDADO:
        # Seccion 0 y 2.6: bandera INDEFINIDA, no falsa.
        fila["reloj_umbral"] = float(reloj_mod.umbral_de(perfil))
        fila["_indefinida"] = "sede_sin_perfil_validado"
        return fila
    out = reloj_mod.clock_region(str(video_path), sede=str(tournament_id), metrica="frac", caja=caja)
    diag = out.get("reloj_diag") or {}
    fps = float(out.get("fps") or 0.0) or None
    clk = out.get("clock_running_s")
    fila["reloj_umbral"] = float(diag.get("umbral") if diag.get("umbral") is not None
                                 else reloj_mod.umbral_de(perfil))
    fila["reloj_n_tics"] = int(diag.get("n_tics") or 0)
    fila["reloj_caja"] = json.dumps(out.get("caja")) if out.get("caja") is not None else ""
    if clk and len(clk) == 2:
        fila["clock_running_s"] = f"[{clk[0]}, {clk[1]}]"
        f_allez = reloj_mod.allez_de(clk, fps or 30.0)
        if f_allez is not None:
            fila["f_allez"] = int(f_allez)
            fila["t_allez_s"] = round(f_allez / (fps or 30.0), 3)
    frac = diag.get("frac_corre")
    # Seccion 2.6: indefinida si el clip es ilegible, la banda tiene < 6 cuadros o la serie < 3.
    indefinida = (diag.get("error") is not None or frac is None
                  or int(diag.get("n") or 0) < 3 or int(out.get("n_frames") or 0) < 6)
    if indefinida:
        fila["_indefinida"] = diag.get("error") or "serie_insuficiente"
        return fila
    fila["reloj_frac_corre"] = float(frac)
    fila["reloj_parado_todo_el_clip"] = bool(float(frac) <= PARADO_MAX)
    return fila


def _importar_reloj(ruta_reloj=None):
    """Importa el lector de referencia. Levanta si no esta: aproximar esta prohibido."""
    candidatas = []
    if ruta_reloj:
        candidatas.append(Path(ruta_reloj))
    aqui = Path(__file__).resolve()
    for base in (aqui.parent, *aqui.parents[:6]):
        candidatas += [base / "scratchpad" / "reloj" / "trabajo", base / "reloj" / "trabajo", base]
    detector = []
    for base in (aqui.parent, *aqui.parents[:6]):
        detector += [base / "scratchpad" / "detector" / "trabajo", base / "detector" / "trabajo", base]
    for d in candidatas:
        if (d / "reloj.py").is_file() and str(d) not in sys.path:
            sys.path.insert(0, str(d))
    for d in detector:
        if (d / "ancla_region.py").is_file() and str(d) not in sys.path:
            sys.path.insert(0, str(d))
    try:
        reloj_mod = importlib.import_module("reloj")
        ar_mod = importlib.import_module("ancla_region")
    except ImportError as exc:
        raise PreflightError(
            "No encuentro el lector de referencia del reloj (`reloj.py` y `ancla_region.py`). "
            "NO se aproxima: submuestrear cambia `frac_corre` en 5 de 6 clips y un umbral unico "
            "llevo la prevalencia de 16,4 % a 53,1 %. Anade los dos modulos al release de Colab "
            f"o pasa `ruta_reloj=`. Detalle: {exc}") from exc
    return reloj_mod, ar_mod


def reloj_lote(filas, videos, *, ruta_reloj=None, salida_csv=None) -> dict:
    """Aplica `reloj_contrato` a una lista de (clip_id, tournament_id) y escribe el CSV del contrato.

    `videos` es un dict clip_id -> ruta de mp4. Coste medido por el agente del reloj: 1,1 s/clip,
    una sola descodificacion, sin GPU. Los clips sin video se omiten y se cuentan.
    """
    import csv as _csv
    hechas, sin_video, indefinidas, errores = [], 0, 0, 0
    for clip_id, torneo in filas:
        ruta = videos.get(str(clip_id))
        if not ruta or not Path(ruta).is_file():
            sin_video += 1
            continue
        try:
            fila = reloj_contrato(ruta, clip_id, torneo, ruta_reloj=ruta_reloj)
        except PreflightError:
            raise
        except Exception as exc:  # noqa: BLE001
            errores += 1
            fila = {campo: "" for campo in CONTRATO_RELOJ}
            fila.update(clip_id=str(clip_id), tournament_id=str(torneo), _indefinida=repr(exc)[:120])
        if fila.get("reloj_parado_todo_el_clip") == "":
            indefinidas += 1
        hechas.append(fila)
    parados = sum(1 for f in hechas if f.get("reloj_parado_todo_el_clip") is True)
    medidas = sum(1 for f in hechas if f.get("reloj_parado_todo_el_clip") in (True, False))
    if salida_csv is not None:
        destino = Path(salida_csv)
        destino.parent.mkdir(parents=True, exist_ok=True)
        with destino.open("w", encoding="utf-8", newline="") as fh:
            escritor = _csv.DictWriter(fh, fieldnames=list(CONTRATO_RELOJ))
            escritor.writeheader()
            for fila in hechas:
                escritor.writerow({k: fila.get(k, "") for k in CONTRATO_RELOJ})
    resultado = {"filas": len(hechas), "sin_video": sin_video, "indefinidas": indefinidas,
                 "errores": errores, "medidas": medidas, "parados": parados,
                 "prevalencia": (parados / medidas) if medidas else None,
                 "csv": str(salida_csv) if salida_csv else None}
    print(f"reloj: {len(hechas)} filas | sin video {sin_video} | indefinidas {indefinidas}"
          f" | errores {errores}")
    if medidas:
        print(f"   medidas {medidas} | parados {parados} ({100*parados/medidas:.2f} %)")
        print("   referencia: 15,1 % [12,5, 18,2] ancla FIE (n=602) y 16,5 % [12,0, 22,3] ancla")
        print("   region (n=200); intervalos solapados, o sea propiedad del corpus entero.")
    print("   `reloj_frac_corre` va SIEMPRE junto a la bandera: el corte 0,05 es una decision.")
    return resultado


# --------------------------------------------------------------------------------------------------
# 6.quater Los tres brazos de eliminacion del atajo, con el signo declarado ANTES
# --------------------------------------------------------------------------------------------------

BRAZOS = {
    "A_ventana_simetrica": {
        "que": "Ventana simetrica EN CONTENIDO: con post_s >= 1,0 las dos luces y las dos acciones "
               "posteriores caen enteras dentro de la ventana, asi que la asimetria de TRUNCAMIENTO "
               "desaparece.",
        "ya_probado": "El ancla compartido en la primera luz YA existe. Anclar cada tirador en SU "
                      "luz se probo y EMPEORO la brecha (+2,94 pp). La version de arriba no se ha "
                      "probado nunca.",
        "base": "|dt_ms| maximo medido 520 ms sobre 3.331 clips de dos luces (p50 80, p95 267, "
                "p99 360; 100 % < 1.000 ms).",
        "signo_esperado": "la BRECHA baja sin que la MEDIA GLOBAL caiga",
        "como_se_lee_al_reves": "si baja la brecha porque cae el 83,00 % de los 1.671, es una "
                                "perdida disfrazada: degradar 5 pp esa poblacion cuesta -3,18 pp.",
    },
    "B_reponderar_960": {
        "que": "Reponderar por 1,75x los clips en los que gano el que encendio primero.",
        "ya_probado": "El diagnostico esta medido: el modelo predice 'gana el segundo' en el "
                      "72,71 % cuando la verdad es 63,51 %. Lo sobreusa.",
        "base": "Legitimo en ENTRENAMIENTO, porque ahi se conoce la etiqueta. Nunca como puerta de "
                "enrutado ni como poblacion de evaluacion.",
        "signo_esperado": "los 960 suben desde 45,21 % y la MEDIA GLOBAL PAREADA no baja",
        "como_se_lee_al_reves": "hay evidencia de que la informacion puede no estar: cinco familias "
                                "ajustadas caen al 44-60 % en ese subconjunto mientras el motor de "
                                "reglas, que no ajusta nada, se queda plano en 53-57 %. Si el brazo "
                                "solo mueve los 960 a costa de los 1.671, el saldo es negativo: "
                                "+5,40 pp exige llevarlos a 60 %, y -3,18 pp basta con degradar 5 pp "
                                "los 1.671.",
    },
    "C_invertir_luces": {
        "que": "Invertir el orden de las luces como aumentacion.",
        "ya_probado": "No. Y hay una trampa conocida: el orden de luces INFORMA un 63,51 %, asi que "
                      "invertirlo MANTENIENDO la etiqueta ensena algo FALSO y destruye senal "
                      "legitima.",
        "base": "La version sana es invertir luces Y etiqueta juntas, o invertir solo en una "
                "fraccion y dar la CURVA sobre esa fraccion.",
        "signo_esperado": "la brecha baja y la media global se mantiene, con la curva sobre la "
                          "fraccion invertida mostrando donde esta el optimo",
        "como_se_lee_al_reves": "se sacrifica informacion REAL: una parte del 63,51 % no es un "
                                "atajo sino la regla de esgrima (quien arranca primero tiende a "
                                "encender primero). Hay que declarar el sacrificio y medir el saldo, "
                                "no solo la brecha.",
    },
}


def declarar_brazo(nombre: str) -> dict:
    """Imprime el brazo y su signo esperado ANTES de medirlo. Declararlo antes es lo que evita
    leer un resultado al reves."""
    if nombre not in BRAZOS:
        raise PreflightError(f"Brazo desconocido: {nombre!r}; hay {sorted(BRAZOS)}")
    b = BRAZOS[nombre]
    print("=" * 78)
    print(f"BRAZO {nombre}")
    print("=" * 78)
    for clave, etiqueta in (("que", "Que hace"), ("base", "Base medida"),
                            ("ya_probado", "Que se sabe ya"),
                            ("signo_esperado", "SIGNO ESPERADO, declarado antes de medir"),
                            ("como_se_lee_al_reves", "Como se leeria al reves")):
        print(f"{etiqueta}:")
        for linea in _envolver(b[clave], 74):
            print("   " + linea)
    print("-" * 78)
    print("Suelo de ruido: el nulo de kernel da +0,36 pp con 327 de 3.047 decisiones distintas")
    print("cambiando SOLO el backend de atencion. Por debajo de eso no hay efecto que leer.")
    print("Estimando: promedia PROBABILIDADES y umbraliza una vez (convenio del artefacto).")
    print("Veredicto: MEDIA GLOBAL PAREADA, reportando las dos poblaciones por separado.")
    print("=" * 78, flush=True)
    return dict(b)


def _envolver(texto, ancho):
    palabras, linea, salida = str(texto).split(), "", []
    for p in palabras:
        if len(linea) + len(p) + 1 > ancho:
            salida.append(linea)
            linea = p
        else:
            linea = (linea + " " + p).strip()
    if linea:
        salida.append(linea)
    return salida


# --------------------------------------------------------------------------------------------------
# 6.quinquies Persistencia incremental y reanudacion
# --------------------------------------------------------------------------------------------------

PERSISTENCIA = """\
PERSISTENCIA INCREMENTAL: a Drive, y NO a un servicio externo.

Una corrida de horas en Colab se muere sola: la sesion caduca, el runtime se recicla, se cae la red.
Sin persistencia incremental se pierde todo. Asi que es requisito duro.

Y va a Google Drive, no a un repositorio de datasets. `AGENTS.md` punto 3: "This is a private
project. Do not publish, push, upload datasets or capture its findings in a global knowledge base."
Subir los NPZ o los indices a un servicio externo es subir un dataset, y lo prohibe esa linea
aunque el repositorio sea privado. Drive es el almacenamiento propio, se monta con una linea, no
publica nada y no sale de la cuenta.

QUE SE PERSISTE, y por que esos tres y no otros:
  checkpoints/  el ledger. Es la FUENTE DE VERDAD de la reanudacion: `ClipLedger.verified` revalida
                el hash del checkpoint, el del NPZ, el de cada evidencia y el del video origen.
  index.jsonl   una linea por clip, escrita con `flush` + `os.fsync` en cuanto el clip cierra.
  npz/          el producto. `analysis/` va con ellos porque el borrado del mp4 exige las tres
                evidencias (qc, moments, measurements) presentes.

EL ESTADO NO ES UN CONTADOR. Si el estado dice "1.200 hechos" y el indice tiene 1.180 filas, la
reanudacion miente. La fuente de verdad es el ledger revalidado con su .npz presente, que es la
misma disciplina de la linea 17 de AGENTS.md.

Y UNA COSA QUE SORPRENDE A MITAD DE CORRIDA: `npz/index.csv` NO EXISTE todavia. `run_data` escribe
`index.jsonl` por clip, y `npz/index.csv` solo lo construye `finalize_data_index` al terminar. Asi
que a mitad de corrida `AnalysisRepository.index()` no ve NADA, y eso es lo esperado, no un fallo.
No hay que "arreglarlo" escribiendo un index.csv parcial: el ledger ya cubre la reanudacion.

DOS PELIGROS, los dos fallan a gritos:
  - un `<registro>.lock` rancio en Drive bloquea la corrida siguiente. Borrarlo cuando se este
    seguro de que no corre nada.
  - un NPZ HUERFANO (la VM murio entre escribir el NPZ y confirmar el ledger) aborta el siguiente
    `run_data` ANTES de hacer nada, con "NPZ exists without a validated checkpoint". Es a proposito:
    borrar `npz/<clip_id>.npz` y su `analysis/*/<clip_id>.json` parcial, o usar un espacio nuevo.

Y `training/epochs/` NUNCA en Drive: la publicacion usa `os.link`.
"""


def aviso_drive(drive_dir, *, exigir: bool = False) -> dict:
    """Drive es opcional. Si no esta, AVISA A GRITOS y sigue: la corrida no sera reanudable."""
    ruta = None if drive_dir is None else Path(drive_dir)
    montado = ruta is not None and ruta.is_dir()
    if montado:
        print(f"OK Drive montado en {ruta}: la corrida es reanudable.")
        libre = shutil.disk_usage(ruta).free / 1024 ** 3 if ruta.exists() else None
        if libre is not None:
            print(f"   espacio libre {libre:.1f} GiB (el corpus de NPZ a duracion completa pide ~1,2 GiB)")
        return {"montado": True, "ruta": str(ruta), "libre_gb": libre}
    print("\n" + "!" * 78)
    print("DRIVE NO ESTA MONTADO. LA CORRIDA NO ES REANUDABLE.")
    print("Si la sesion caduca o el runtime se recicla, se pierde TODO el trabajo y hay que")
    print("empezar de cero. En una corrida de horas eso pasa. Para montarlo:")
    print("    from google.colab import drive; drive.mount('/content/drive')")
    print("Se sigue adelante a proposito, porque un ensayo corto no lo necesita, pero queda dicho.")
    print("!" * 78 + "\n", flush=True)
    if exigir:
        raise PreflightError("Drive no montado y se pidio persistencia obligatoria")
    return {"montado": False, "ruta": None if ruta is None else str(ruta)}


def estado_reanudacion(workspace, *, drive_dir=None) -> dict:
    """Que hay hecho DE VERDAD, contado por el ledger revalidado, no por un contador.

    Compara los cuatro conjuntos que tienen que coincidir y dice cual se desvia:
        NPZ en disco | lineas de index.jsonl | checkpoints del ledger | ledger que REVALIDA
    Un NPZ sin checkpoint es el huerfano que aborta la corrida siguiente. Un checkpoint que no
    revalida es un clip que se va a reprocesar (correcto, pero conviene saberlo).
    """
    work = Path(workspace)
    npzs = {p.stem for p in (work / "npz").glob("*.npz")} if (work / "npz").is_dir() else set()
    jsonl = work / "index.jsonl"
    filas = set()
    if jsonl.is_file():
        for linea in jsonl.open(encoding="utf-8"):
            try:
                filas.add(str(json.loads(linea).get("clip_id")))
            except Exception:  # noqa: BLE001
                continue
    ledger_dir = work / "checkpoints" / "clips"
    ckpts = {p.stem for p in ledger_dir.glob("*.json")} if ledger_dir.is_dir() else set()
    revalidan = set()
    try:
        from fencingly_v2 import colab_workflow as CW
        ledger = CW.ClipLedger(work)
        for cid in sorted(ckpts):
            try:
                if ledger.verified(cid) is not None:
                    revalidan.add(cid)
            except Exception:  # noqa: BLE001
                continue
    except Exception as exc:  # noqa: BLE001
        revalidan = None
        print("   (no se pudo usar ClipLedger:", repr(exc)[:120], ")")
    huerfanos = sorted(npzs - ckpts)
    resultado = {"npz": len(npzs), "index_jsonl": len(filas), "checkpoints": len(ckpts),
                 "revalidan": None if revalidan is None else len(revalidan),
                 "huerfanos": huerfanos, "index_csv_existe": (work / "npz" / "index.csv").is_file()}
    print(f"NPZ en disco                : {len(npzs)}")
    print(f"lineas de index.jsonl       : {len(filas)}")
    print(f"checkpoints del ledger      : {len(ckpts)}")
    print(f"checkpoints que REVALIDAN   : {resultado['revalidan']}   <- lo que se va a SALTAR")
    print(f"npz/index.csv existe        : {resultado['index_csv_existe']}"
          "   <- False a mitad de corrida es LO ESPERADO")
    if huerfanos:
        print(f"!! NPZ HUERFANOS ({len(huerfanos)}): {huerfanos[:5]}")
        print("   El siguiente `run_data` va a abortar ANTES de trabajar, a proposito. Borra")
        print("   `npz/<clip_id>.npz` y su `analysis/*/<clip_id>.json` parcial, o usa otro espacio.")
    if drive_dir is not None:
        estado = Path(drive_dir) / "estado.json"
        if estado.is_file():
            try:
                d = json.loads(estado.read_text(encoding="utf-8"))
                print(f"Drive: estado.json dice completado={d.get('completado')} "
                      f"npz_en_drive={d.get('npz_en_drive')}")
                if resultado["revalidan"] is not None and d.get("npz_en_drive") not in (None, resultado["revalidan"]):
                    print("   (el contador de Drive y los que revalidan no coinciden: manda el ledger)")
            except Exception:  # noqa: BLE001
                pass
        else:
            print("Drive: sin estado.json todavia")
    return resultado


RESPALDO = """\
RESPALDO REMOTO: repositorio PRIVADO de Hugging Face. La excepcion esta escrita, y su alcance.

`AGENTS.md` linea 4, decidido por el usuario el 2026-09-09 y acotado a proposito: un repositorio
**privado** de Hugging Face puede usarse como destino de reanudacion de una extraccion larga en
Colab, para que una sesion caducada no pierda horas de trabajo. Lo que lo hace aceptable es que la
carga son ~50 GB y cabe en los 100 GB gratuitos del plan PRIVADO, asi que nada se hace publico.

**UN REPOSITORIO PUBLICO SIGUE PROHIBIDO**, y el motivo esta escrito: los recortes del corredor son
fotogramas REALES de retransmision, no rasgos derivados, asi que un repo publico redistribuiria
video de terceros; y el manifiesto lleva nombres de tiradores, torneos y resultados, asi que
publicaria el historial competitivo de atletas identificables. `respaldo_configurar` pone
`private=True` explicito y FALLA si detecta que el repo no es privado.

No ampliar la excepcion a un repo publico, a un segundo servicio, ni a ninguna carga mas alla de los
artefactos de extraccion, sin una decision nueva y explicita del usuario.

QUE CABE, y por que: el FLUJO OPTICO DENSO NO SE GUARDA. Regenerarlo desde los recortes de 128x128
reproduce la ruta de video BIT A BIT a 394 ms/clip contra 1.250. Asi que ~45 GB de recortes
sustituyen a ~550 GB de flujo. La regla es: **guarda lo que no se puede regenerar; regenera lo que
si.** Reparto objetivo: recortes ~45 GB, pose a cadencia nativa ~2 GB, indices y columnas de
resumen <1 GB.

EL TOKEN ES UNA CREDENCIAL. Se lee de un secreto de Colab (`google.colab.userdata`) o de la
variable de entorno `HF_TOKEN`. Nunca se escribe en el cuaderno, nunca se imprime, nunca aparece en
un log ni en un informe. `token_presente()` devuelve un BOOLEANO, no el token.

LIMITES DE LA PLATAFORMA, comprobados por el usuario: menos de 100.000 ficheros por repo y menos de
10.000 por carpeta. Con 34.374 clips y varios ficheros cada uno eso se roza, asi que **se agrupa en
`tar` por lote**: `lotes/<sede>/lote_<nnn>.tar` con los NPZ y las evidencias de sus clips, mas
`index.csv` y `estado.json` en la raiz. Con lotes de ~70 clips salen ~500 ficheros para el corpus
entero, dos ordenes de magnitud por debajo del limite.
"""


def token_presente() -> bool:
    """True si hay token disponible. NO devuelve el token, y no lo imprime nunca."""
    if os.environ.get("HF_TOKEN"):
        return True
    try:
        from google.colab import userdata  # type: ignore
        return bool(userdata.get("HF_TOKEN"))
    except Exception:  # noqa: BLE001
        return False


def _token():
    """Uso interno. Nunca se registra, nunca se devuelve al llamante del paquete."""
    valor = os.environ.get("HF_TOKEN")
    if valor:
        return valor
    try:
        from google.colab import userdata  # type: ignore
        return userdata.get("HF_TOKEN")
    except Exception:  # noqa: BLE001
        return None


def aviso_respaldo(*, exigir: bool = False) -> dict:
    """El respaldo NO es obligatorio. Sin token, avisa a gritos y sigue."""
    hay = token_presente()
    if hay:
        print("OK token de Hugging Face disponible (de secreto de Colab o de HF_TOKEN).")
        print("   No se imprime ni se registra en ningun sitio: es una credencial.")
        return {"token": True}
    print("\n" + "!" * 78)
    print("NO HAY TOKEN DE HUGGING FACE. LA CORRIDA NO ES REANUDABLE ENTRE SESIONES.")
    print("Si la sesion caduca o el runtime se recicla, se pierde el trabajo de esta sesion.")
    print("Para ponerlo, en Colab: panel izquierdo -> la llave (Secrets) -> nuevo secreto")
    print("   nombre : HF_TOKEN")
    print("   valor  : un token de escritura de huggingface.co/settings/tokens")
    print("   y activa 'Notebook access' para este cuaderno.")
    print("Fuera de Colab: export HF_TOKEN=...  (nunca en el cuaderno).")
    print("Se sigue adelante a proposito; el ensayo corto no lo necesita.")
    print("!" * 78 + "\n", flush=True)
    if exigir:
        raise PreflightError("respaldo remoto exigido y no hay token")
    return {"token": False}


def clips_por_lote(segundos_por_clip: float, *, minutos_por_instantanea: float = 10.0,
                   minimo: int = 5, maximo: int = 500) -> int:
    """Deriva el tamano de lote del RITMO MEDIDO, no de un numero a ojo.

    Medido en el ensayo de esta maquina (RTX 3060, `process_workers=1`, ventana completa):
    17,2 s/clip en la primera corrida y 11,9 s/clip en la segunda con 3 de 6 reanudados. Con
    17,2 s/clip y una instantanea cada 10 minutos salen 34 clips por lote.
    """
    if not math.isfinite(segundos_por_clip) or segundos_por_clip <= 0:
        raise PreflightError("el ritmo medido tiene que ser un positivo finito")
    n = int(round(minutos_por_instantanea * 60.0 / float(segundos_por_clip)))
    return max(minimo, min(maximo, n or minimo))


def _clips_completos(workspace):
    """Los clips que existen DE VERDAD: fila en el indice Y su .npz presente.

    Es la fuente de verdad de la reanudacion, no un contador. Si el contador dice 1.200 y el indice
    tiene 1.180, la reanudacion miente. Se prefiere `npz/index.csv` cuando existe (es el fusionado
    que leen el Store y la web) y se cae a `index.jsonl`, que es lo unico que hay a mitad de corrida.
    """
    import csv as _csv
    work = Path(workspace)
    npzs = {p.stem for p in (work / "npz").glob("*.npz")} if (work / "npz").is_dir() else set()
    filas, fuente = set(), None
    csv_path = work / "npz" / "index.csv"
    if csv_path.is_file():
        fuente = "npz/index.csv"
        with csv_path.open(encoding="utf-8", newline="") as fh:
            for r in _csv.DictReader(fh):
                if str(r.get("status", "")).lower() in ("ok", "cached"):
                    filas.add((r.get("clip_id") or "").strip())
    elif (work / "index.jsonl").is_file():
        fuente = "index.jsonl"
        for linea in (work / "index.jsonl").open(encoding="utf-8"):
            try:
                d = json.loads(linea)
            except Exception:  # noqa: BLE001
                continue
            if str(d.get("status", "")).lower() in ("ok", "cached"):
                filas.add(str(d.get("clip_id")))
    return (filas & npzs), {"fuente": fuente, "filas": len(filas), "npz": len(npzs),
                            "solo_fila": len(filas - npzs), "solo_npz": len(npzs - filas)}


def respaldo_empaquetar(workspace, destino, *, clips=None, sede="varias", lote=0) -> dict:
    """Empaqueta un lote en un `tar` con sus NPZ y sus evidencias, y actualiza el estado.

    `destino` es un directorio: en el ensayo es local (simula el remoto) y en Colab es el punto de
    montaje o el directorio de trabajo desde el que `huggingface_hub` sube. Empaquetar y subir son
    dos pasos separados A PROPOSITO, para poder probar el primero sin credenciales.
    """
    import tarfile
    work, dest = Path(workspace), Path(destino)
    completos, diag = _clips_completos(work)
    elegidos = sorted(completos if clips is None else (set(map(str, clips)) & completos))
    if not elegidos:
        return {"tar": None, "clips": 0, "diagnostico": diag}
    carpeta = dest / "lotes" / str(sede)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"lote_{int(lote):04d}.tar"
    if ruta.exists():
        return {"tar": str(ruta), "clips": 0, "reutilizado": True, "diagnostico": diag}
    temporal = ruta.with_suffix(".tar.parcial")
    with tarfile.open(temporal, "w") as tar:
        for cid in elegidos:
            npz = work / "npz" / f"{cid}.npz"
            if npz.is_file():
                tar.add(npz, arcname=f"npz/{cid}.npz")
            for sub in ("qc", "moments", "measurements"):
                ev = work / "analysis" / sub / f"{cid}.json"
                if ev.is_file():
                    tar.add(ev, arcname=f"analysis/{sub}/{cid}.json")
            ck = work / "checkpoints" / "clips" / f"{cid}.json"
            if ck.is_file():
                tar.add(ck, arcname=f"checkpoints/clips/{cid}.json")
    temporal.replace(ruta)
    for nombre in ("index.jsonl",):
        if (work / nombre).is_file():
            shutil.copyfile(work / nombre, dest / nombre)
    if (work / "npz" / "index.csv").is_file():
        shutil.copyfile(work / "npz" / "index.csv", dest / "index.csv")
    estado = respaldo_estado(dest, workspace=work, escribir=True)
    return {"tar": str(ruta), "clips": len(elegidos), "bytes": ruta.stat().st_size,
            "diagnostico": diag, "estado": estado}


def respaldo_estado(destino, *, workspace=None, escribir: bool = False) -> dict:
    """El estado del respaldo, derivado del INDICE y de los tar, nunca de un contador."""
    import tarfile
    dest = Path(destino)
    tars = sorted(p for p in (dest / "lotes").rglob("*.tar")) if (dest / "lotes").is_dir() else []
    subidos = set()
    for t in tars:
        try:
            with tarfile.open(t, "r") as tar:
                subidos |= {Path(n).stem for n in tar.getnames() if n.startswith("npz/")}
        except Exception:  # noqa: BLE001
            continue
    estado = {"lotes": len(tars), "clips_en_lotes": len(subidos),
              "bytes_en_lotes": sum(p.stat().st_size for p in tars),
              "ficheros_en_destino": sum(1 for _ in dest.rglob("*") if _.is_file()),
              "limite_ficheros_repo": 100000, "limite_ficheros_carpeta": 10000}
    if workspace is not None:
        completos, diag = _clips_completos(workspace)
        estado["clips_completos_en_disco"] = len(completos)
        estado["pendientes_de_empaquetar"] = len(completos - subidos)
        estado["indice"] = diag
    if escribir:
        (dest / "estado.json").write_text(json.dumps(estado, indent=1, sort_keys=True), encoding="utf-8")
    return estado


def respaldo_pendientes(workspace, destino) -> list:
    """Los clips que faltan por empaquetar: completos en disco menos los que ya estan en un tar."""
    completos, _ = _clips_completos(workspace)
    estado = respaldo_estado(destino)
    import tarfile
    dest = Path(destino)
    subidos = set()
    if (dest / "lotes").is_dir():
        for t in (dest / "lotes").rglob("*.tar"):
            try:
                with tarfile.open(t, "r") as tar:
                    subidos |= {Path(n).stem for n in tar.getnames() if n.startswith("npz/")}
            except Exception:  # noqa: BLE001
                continue
    return sorted(completos - subidos)


CELDA_DE_VERIFICACION = '''
# CELDA DE VERIFICACION DEL RESPALDO. Ejecutarla UNA VEZ, con el token puesto como secreto.
# Crea el repo PRIVADO, sube un lote pequeno, lo lee de vuelta, lo borra y confirma.
# No imprime el token. Si el repo no sale privado, LEVANTA.
import preflight as PF
from huggingface_hub import HfApi, create_repo

assert PF.token_presente(), "pon HF_TOKEN como secreto de Colab antes de ejecutar esta celda"
REPO = "TU_USUARIO/fencingly-extraccion"        # cambia el usuario
api = HfApi(token=PF._token())

create_repo(REPO, repo_type="dataset", private=True, exist_ok=True, token=PF._token())
info = api.repo_info(REPO, repo_type="dataset", token=PF._token())
assert info.private is True, "EL REPO NO ES PRIVADO. Borralo y no subas nada: AGENTS.md linea 4."
print("repo privado confirmado:", REPO)

# un lote pequeno de prueba
r = PF.respaldo_empaquetar(WORK, "/content/respaldo", clips=None, sede="prueba", lote=0)
print("empaquetado:", {k: v for k, v in r.items() if k != "estado"})
api.upload_folder(folder_path="/content/respaldo", repo_id=REPO, repo_type="dataset",
                  token=PF._token(), commit_message="prueba de respaldo")
print("subido. ficheros en el repo:", len(api.list_repo_files(REPO, repo_type="dataset", token=PF._token())))

# y se limpia, para que la prueba no deje residuo
api.delete_folder(path_in_repo="lotes/prueba", repo_id=REPO, repo_type="dataset", token=PF._token())
print("prueba borrada. El respaldo esta verificado de punta a punta.")
'''


FPS_NATIVO_PENDIENTE = """\
HUECO RESERVADO: extraccion a CADENCIA NATIVA (`FPS_NATIVO`).

Hoy el Store remuestrea a `TARGET_FPS = 30` (`sample_time.TARGET_FPS`), asi que en los clips a 50 fps
se tira el 40 % de los fotogramas y en los de 60 fps el 50 %. La composicion medida del corpus dice
cuanto pesa eso: ~50 fps el 20,7 % y ~59-60 fps el 6,3 %, o sea que afecta a algo mas de un cuarto
del corpus.

El mando NO esta implementado a proposito. Hay un agente midiendo el coste en tamano y en tiempo, y
implementarlo a ciegas es exactamente el error que este paquete existe para no repetir: `TARGET_FPS`
esta en `sample_time.py`, que esta en los 20 de MEASUREMENT_SOURCES **y** en
`portable_contract.SEMANTIC_SCOPE`, asi que tocarlo re-materializa los 23.091 artefactos **y**
invalida todos los bundles portables. Es el cambio mas caro del conducto, no el mas barato.

Cuando lleguen sus cifras: aqui va el mando, y en el documento su coste medido al lado de las otras
vias. Hasta entonces esto es el hueco, declarado.
"""


# --------------------------------------------------------------------------------------------------
# 7. El indice real: 'N clips de M' casi nunca se arregla calculando
# --------------------------------------------------------------------------------------------------

def indice_real(workspace, *, exigir: bool = False) -> dict:
    """Cuenta el conjunto en CADA eslabon y dice cual lo tira. No supone ninguno.

    `AnalysisRepository.index()` se salta EN SILENCIO un directorio de NPZ sin `npz/index.csv`, asi
    que un trabajador que escribe solo `indice.jsonl` produce salida invisible: costo 689 clips
    invisibles el 2026-09-09 y antes tuvo la pagina de un tirador en 180 de 587 durante dias.
    Ademas `valid_id` exige `[A-Za-z0-9_-]{1,200}`, asi que un `clip_id` no ASCII se cae del indice
    sin error aunque su fila este escrita y su .npz exista: exactamente 13 clips del archivo
    (`Vicuna`, `Kauai'O'o`, `Vendeen`; 7 derecha, 6 izquierda, 0,04 %). NO relajar el validador.
    """
    work = Path(workspace)
    npz_dir = work / "npz"
    indice_csv = npz_dir / "index.csv"
    en_disco = sorted(p.stem for p in npz_dir.glob("*.npz")) if npz_dir.is_dir() else []
    filas_csv, ok_csv, no_ascii = [], [], []
    if indice_csv.is_file():
        import csv
        with indice_csv.open(encoding="utf-8", newline="") as fh:
            for fila in csv.DictReader(fh):
                cid = (fila.get("clip_id") or "").strip()
                filas_csv.append(cid)
                if str(fila.get("status", "")).lower() in ("ok", "cached"):
                    ok_csv.append(cid)
                if cid and not all(c.isascii() and (c.isalnum() or c in "_-") for c in cid):
                    no_ascii.append(cid)
    jsonl = work / "index.jsonl"
    n_jsonl = sum(1 for _ in jsonl.open(encoding="utf-8")) if jsonl.is_file() else 0

    leidos_por_el_lector = None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from webapp_react.analysis_store import AnalysisRepository  # type: ignore
        repo = AnalysisRepository([npz_dir], work / "_cache_preflight")
        leidos_por_el_lector = int(len(repo.index()))
    except Exception as exc:  # noqa: BLE001
        leidos_por_el_lector = f"no se pudo usar el lector real: {exc!r}"

    conjunto = set(en_disco) & set(ok_csv)
    resultado = {"npz_en_disco": len(en_disco), "filas_index_csv": len(filas_csv),
                 "filas_ok_cached": len(ok_csv), "index_csv_existe": indice_csv.is_file(),
                 "lineas_index_jsonl": n_jsonl, "no_ascii": no_ascii,
                 "interseccion_disco_y_csv_ok": len(conjunto), "leidos_por_el_lector": leidos_por_el_lector}
    _linea()
    print(f"npz en disco                         : {len(en_disco)}")
    print(f"npz/index.csv existe                 : {indice_csv.is_file()}   <- si es False, el lector")
    print(f"                                       ignora el directorio EN SILENCIO")
    print(f"filas en index.csv                   : {len(filas_csv)}")
    print(f"  de ellas status ok|cached          : {len(ok_csv)}")
    print(f"index.jsonl (NO lo lee nadie)        : {n_jsonl}")
    print(f"clip_id no ASCII (se caen, conocido) : {len(no_ascii)} {no_ascii[:5]}")
    print(f"disco INTERSECCION csv(ok|cached)    : {len(conjunto)}")
    print(f"leidos por AnalysisRepository.index(): {leidos_por_el_lector}")
    _linea()
    if not indice_csv.is_file() and en_disco:
        print("!! Hay .npz sin index.csv: genera el index.csv al cerrar cada shard, copiando el orden")
        print("   de columnas de un indice que YA carga, y emite una fila solo cuando su .npz exista.")
        if exigir:
            raise PreflightError("directorio de NPZ sin index.csv: la salida es invisible para el lector")
    if isinstance(leidos_por_el_lector, int):
        # `AnalysisRepository.index()` devuelve TODA fila con `valid_id`, sea cual sea su `status`;
        # no filtra por ok|cached. Asi que la comparacion honesta es contra las filas del CSV menos
        # las que `valid_id` tira, no contra la interseccion.
        esperado_del_lector = len(filas_csv) - len(no_ascii)
        if leidos_por_el_lector == esperado_del_lector:
            print(f"   coherente: el lector devuelve {leidos_por_el_lector} = {len(filas_csv)} filas"
                  f" - {len(no_ascii)} tiradas por valid_id (brecha conocida y aceptada).")
            print(f"   El conjunto que puede MEDIRSE es la interseccion: {len(conjunto)}.")
        else:
            print(f"!! El lector devuelve {leidos_por_el_lector} y se esperaban"
                  f" {esperado_del_lector} ({len(filas_csv)} filas - {len(no_ascii)} no ASCII).")
            print("   Antes de proponer calcular o re-extraer nada, mira que eslabon lo tira:")
            print("   manifiesto -> index.csv(ok|cached) -> .npz en un directorio registrado.")
    return resultado


# --------------------------------------------------------------------------------------------------
# 8. early_stopping explicito
# --------------------------------------------------------------------------------------------------

def estimador_hgb(**kwargs):
    """Devuelve un `HistGradientBoostingClassifier` con `early_stopping` SIEMPRE explicito.

    `early_stopping='auto'` enciende la parada temprana por encima de 10.000 filas de entrenamiento,
    y con ella un corte de validacion aleatorio que hace que las semillas importen en silencio.
    Un pliegue de validacion cruzada entrena 5.672 filas y no le afecta; el ajuste sobre la
    POBLACION COMPLETA si. Con los pares nuevos la poblacion pasa a 6.465 pares = 12.930 filas con
    espejo, contra el umbral de 10.000, asi que YA LO CRUZA. `prio2/e1_tabular.py`,
    `prio2/e10_afina.py` y `prio2/e12_final.py` ya lo fijan; cualquier estimador que construya el
    cuaderno tiene que fijarlo tambien, o su comportamiento dependera de cuantos clips se extrajeron.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    kwargs.setdefault("early_stopping", False)
    if kwargs["early_stopping"] == "auto":
        raise PreflightError("early_stopping='auto' hace que el estimador dependa de cuantos clips "
                             "se extrajeron; fijalo a False (o a True a proposito)")
    print(f"HistGradientBoostingClassifier(early_stopping={kwargs['early_stopping']!r}) - explicito")
    return HistGradientBoostingClassifier(**kwargs)


def revisar_early_stopping(modelo) -> dict:
    valor = getattr(modelo, "early_stopping", "<ausente>")
    ok = valor is not None and valor != "auto"
    print(f"{'OK ' if ok else '!! '}early_stopping = {valor!r}")
    if not ok:
        raise PreflightError("early_stopping no explicito")
    return {"early_stopping": valor}


# --------------------------------------------------------------------------------------------------
# 9. Las dos poblaciones: el numero del que se trata todo
# --------------------------------------------------------------------------------------------------

def dos_poblaciones(y_verdad, y_pred, gano_el_segundo) -> dict:
    """Reporta acierto en las dos poblaciones por separado y la media global pareada.

    `gano_el_segundo[i]` es True si en ese clip gano el tirador que encendio SEGUNDO. Es una
    propiedad del RESULTADO, asi que NO se puede usar como region de evaluacion, ni como objetivo de
    optimizacion, ni como puerta de enrutado: dentro de un subgrupo definido por el resultado la base
    cae bajo el azar y un bloque sin informacion compro +3,08 pp. Se usa solo para DESCRIBIR donde
    cayo la ganancia. El veredicto es `global`.
    """
    import numpy as np
    y_verdad = np.asarray(y_verdad)
    y_pred = np.asarray(y_pred)
    seg = np.asarray(gano_el_segundo, bool)
    if not (len(y_verdad) == len(y_pred) == len(seg)):
        raise PreflightError("Las tres series tienen que tener la misma longitud")
    acierto = (y_verdad == y_pred)
    out = {"n": int(len(acierto)), "global": float(acierto.mean()),
           "gano_el_segundo": {"n": int(seg.sum()), "acierto": float(acierto[seg].mean()) if seg.any() else None},
           "gano_el_primero": {"n": int((~seg).sum()), "acierto": float(acierto[~seg].mean()) if (~seg).any() else None}}
    p2, p1 = out["gano_el_segundo"]["acierto"], out["gano_el_primero"]["acierto"]
    out["brecha_pp"] = None if (p1 is None or p2 is None) else round(100 * (p2 - p1), 2)
    print(f"VEREDICTO (media global pareada)   : {100*out['global']:.2f} %  n = {out['n']}")
    print(f"  gano el que encendio segundo     : {100*p2:.2f} %  n = {out['gano_el_segundo']['n']}"
          if p2 is not None else "  gano el que encendio segundo     : sin clips")
    print(f"  gano el que encendio primero     : {100*p1:.2f} %  n = {out['gano_el_primero']['n']}"
          if p1 is not None else "  gano el que encendio primero     : sin clips")
    print(f"  brecha                           : {out['brecha_pp']} pp   (referencia historica +37,80 pp)")
    print("  El subgrupo DESCRIBE donde cayo la ganancia. No es la poblacion de evaluacion.")
    print("  Recordatorio de potencia: +1,6 pp es indetectable con este corpus.")
    return out


# --------------------------------------------------------------------------------------------------
# 10. Todo de una vez
# --------------------------------------------------------------------------------------------------

def todo(workspace=None, *, almacen="completa", rodaja_nombre="recomendada", t_max=140,
         exigir=True, entorno_reporte=None, store_dir=None) -> dict:
    """Todas las puertas, en el orden en que importan.

    `entorno` va ANTES de `huella` y el orden no es cosmetico: la huella no detecta que el runtime
    no se haya reiniciado tras instalar, porque `_producer()` lee los metadatos en disco y no el
    modulo cargado. Ver `entorno()`.
    """
    protocolo()
    print()
    _linea()
    print("1. ENTORNO CLAVADO  (antes que la huella: la huella no detecta el desajuste)")
    _linea()
    resultado = {"entorno": entorno(exigir=exigir)}
    print()
    _linea()
    print("2. HUELLA DEL PRODUCTOR DE MEDIDAS")
    _linea()
    resultado["huella"] = huella(exigir=exigir)
    print()
    _linea()
    print("3. RECORTE A LA VENTANA DE ANALISIS (necesario con almacen ancho)")
    _linea()
    resultado["parche"] = parche_de_recorte(exigir=exigir and almacen != "desplegada")
    print()
    _linea()
    print("4. LA RODAJA DE ENTRENAMIENTO CABE EN T_max")
    _linea()
    resultado["rodaja"] = plan_rodaja(rodaja_nombre, t_max=t_max, exigir=exigir)
    print()
    _linea()
    print("5. QUE CUESTA CADA ARREGLO")
    _linea()
    resultado["listas"] = coste_de_las_listas()
    print()
    _linea()
    print("6. PLAN DE VENTANA DE ALMACEN EN ESTE RUNTIME")
    _linea()
    print(BARRIDO_MEDIDO)
    if entorno_reporte is not None:
        resultado["plan"] = plan_ventana(entorno_reporte, almacen)
    print()
    _linea()
    print("7. LOS PRECIOS DE LA DURACION COMPLETA")
    _linea()
    print(COLA_TRAS_TOCADO)
    print(TOPE_QC)
    if workspace is not None:
        print()
        _linea()
        print("8. EL INDICE, CONTADO CON EL LECTOR REAL")
        _linea()
        resultado["indice"] = indice_real(workspace)
    if store_dir is not None:
        print()
        _linea()
        print("9. ALINEACION DEL STORE  (la puerta que impide una regresion silenciosa)")
        _linea()
        resultado["alineacion"] = alineacion_del_store(
            store_dir, post_s_esperado=RODAJAS[rodaja_nombre][1], exigir=exigir)
    return resultado


if __name__ == "__main__":
    # `python -B preflight.py [workspace] [store_dir]`. Informa de todo y no levanta por nada, para
    # que sirva de diagnostico. En el cuaderno las puertas van con `exigir=True`.
    protocolo()
    print()
    entorno(exigir=False)
    for etiqueta, funcion in (("huella", lambda: huella(exigir=False)),
                              ("parche_de_recorte", lambda: parche_de_recorte(exigir=False)),
                              ("coste_de_las_listas", coste_de_las_listas)):
        print()
        try:
            funcion()
        except Exception as exc:  # noqa: BLE001
            print(f"{etiqueta}: no se pudo evaluar: {exc!r}")
    print()
    for nombre in RODAJAS:
        try:
            plan_rodaja(nombre, t_max=135, exigir=False)
        except Exception as exc:  # noqa: BLE001
            print(f"plan_rodaja({nombre}): {exc!r}")
        print()
    print(BARRIDO_MEDIDO)
    print(COLA_TRAS_TOCADO)
    print(TOPE_QC)
    if len(sys.argv) > 1:
        indice_real(sys.argv[1])
    if len(sys.argv) > 2:
        print()
        try:
            alineacion_del_store(sys.argv[2], post_s_esperado=0.3, exigir=False)
        except Exception as exc:  # noqa: BLE001
            print("alineacion_del_store: no se pudo evaluar:", repr(exc))

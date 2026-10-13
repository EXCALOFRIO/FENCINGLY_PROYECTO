"""Las rutas del proyecto en UN sitio, para que el mismo codigo corra aqui y en Colab.

EL PROBLEMA, medido. Hay 225 rutas absolutas repartidas por el arbol, y cinco de ellas apuntan a
directorios temporales de OTRAS sesiones que un dia se borran. En Colab ninguna existe, y el modo
en que fallan es peor que un error: `snap()` devuelve `None`, `from_pretrained(str(None))` pide al
hub un repositorio llamado literalmente `"None"`, y la traza acusa a Hugging Face. O `os.walk` de
un directorio que no existe no falla: devuelve vacio, y el lote imprime "NO VIDEO, skipped"
quinientas veces, que se lee como un problema de datos y es de configuracion.

QUE HACE ESTE FICHERO. Deriva todo lo que puede de su propia posicion y deja lo demas en variables
de entorno con un valor por defecto que es el de hoy. Solo biblioteca estandar, para poder
importarse antes que `torch`, antes que `cv2` y antes de que nadie consuma `HF_HOME`.

QUE NO HACE. No convierte los 88 ficheros del arbol. Los ~80 scripts de experimento (`a2_*`,
`e1`-`e29`, `f1`-`f17`, `capas_limpias_*`) producen las cifras publicadas en `RESULTADOS.md` y
tocarlos arriesga esas cifras a cambio de cero beneficio en Colab. Se convierten los **nueve de la
ruta critica** y se pone un guardia que comprueba que siguen limpios.

    python rutas.py     imprime cada raiz, de donde sale y si existe
"""
from __future__ import annotations

import os
import pathlib

RAIZ = pathlib.Path(__file__).resolve().parents[2]
"""El repo, derivado de donde esta este fichero. No se configura: si esto estuviera mal, todo lo
demas estaria mal de una forma visible en la primera linea."""

CODIGO = RAIZ / "01_extraccion" / "codigo"
VAL = RAIZ / "02_validacion"
HERRAMIENTA = VAL / "herramienta"
PISTA = VAL / "pista"
MUESTRAS = VAL / "muestras"
DATOS = RAIZ / "03_datos"
CATALOGO = DATOS / "catalogo"
MODELOS = RAIZ / "01_extraccion" / "modelos"
"""Donde viven los pesos del etiquetador. Son unos pocos MB y viajan DENTRO del repo, que es lo que
hace que en Colab la etapa 10 solo tenga que cargarlos."""

_FUENTE = {}


def _env(clave, defecto):
    v = os.environ.get(clave)
    _FUENTE[clave] = "env" if v else "defecto"
    return pathlib.Path(v) if v else pathlib.Path(defecto)


TRABAJO = _env("FENCINGLY_TRABAJO", RAIZ / "01_extraccion" / "pruebas")
"""Donde viven `det_`, `revision_` y `poses_` de ESTA corrida. UNA sola raiz.

Que sea una sola no es estetica: hoy la etapa 1 escribe `det_<clip>.pkl` en
`01_extraccion/pruebas` y `pista.pareja()` lo lee de un scratchpad temporal de otra sesion. Medido:
en `pruebas` hay **0** `det_*.pkl` y en el scratchpad **117**. O sea que en cuanto se cablee la
etapa 7 devolvera `sin pickle de detecciones` para todos los clips -- un fallo blando, que imprime
una linea ordenada y no escribe nada."""

PISTA = _env("FENCINGLY_PISTA", PISTA)
"""Las series de la etapa 7. Por defecto `02_validacion/pista`, como siempre. `FENCINGLY_PISTA` existe
para los ENSAYOS (2026-10-13): con `FENCINGLY_TRABAJO` y esta en un scratch, un ensayo de la corrida
no escribe nada en las salidas de verdad."""

VIDEOS = _env("FENCINGLY_VIDEOS", "D:/FENCINGLY_VIDEOS_RAW")
SCRATCH = _env("SCRATCH", "D:/FENCINGLY_SCRATCH")
HF = _env("HF_HOME", "D:/FENCINGLY_HF")
"""LA CACHE DE PESOS (Sapiens 2 pose, pointmap, profundidad), y por que en D: (2026-10-08).

Hasta hoy el defecto apuntaba al scratchpad TEMPORAL de la sesion 73a45131: el mismo error que el
22/09 se llevo los interpretes (ver `VENV` abajo e `INCIDENTE_2026-09-23_VENV.md`). Una corrida de
semanas no puede colgar de un `%TEMP%`. Se copio entera (4,8 GB, `diff -r` identico) a D:."""
YOLO_PT = _env("FENCINGLY_YOLO", RAIZ / "01_extraccion" / "yolo11m-pose.pt")
"""Antes apuntaba a un temporal que ya no existia, y `pesos_yolo()` caia al nombre pelado, que
solo funcionaba porque el directorio de trabajo era `01_extraccion` y alli hay una copia."""

os.environ.setdefault("HF_HOME", str(HF))

VENV = _env("FENCINGLY_VENV", "D:/FENCINGLY_PY/venv")
"""EL INTERPRETE DURADERO, y esta linea existe por un incidente.

La noche del 2026-09-22 la extraccion corrio **25 minutos** y se paro. Los tres interpretes vivian
en `%TEMP%\\claude\\...\\73a45131-...\\scratchpad\\`, o sea **en el directorio temporal de otra
sesion**, y a las 23:32 un borrado recursivo se los llevo: de 16 directorios de sesion quedaron 5,
en `Scripts` quedo **solo `python.exe`** de unos veinte ficheros, y lo que sobrevivio fue
exactamente lo que los procesos tenian abierto -- `torch` importaba pero `torch.cuda` ya no
existia. Seis horas de GPU perdidas por tener la capacidad de computo colgando de un temporal.

Ver `04_documentacion/informes/INCIDENTE_2026-09-23_VENV.md`."""

_VENV_TMP = pathlib.Path(
    r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop"
    r"\73a45131-1b46-4cbf-95e2-29933d543565\scratchpad")
"""Los venv viejos del scratchpad. Se conservan como ULTIMO recurso y avisando, no como defecto."""


def interprete(cual="hf"):
    """QUE DEVUELVE: el ejecutable de Python que puede correr esa familia de etapas.

    `cual` es `"yolo"` (ultralytics) o `"hf"` (transformers). **Lo normal ahora es que sea el
    mismo para los dos**: un solo entorno con todo, que es como corre Colab. La separacion en dos
    venv existia para que cada etapa supiera si podia correr, y esa pregunta ya la contesta
    `pipeline.capacidades()` mirando los imports.

    EL ORDEN IMPORTA, y el temporal va el ultimo a proposito:

      1. `FENCINGLY_PY_YOLO` / `FENCINGLY_PY_HF`, si estan puestas
      2. `FENCINGLY_VENV` o `D:/FENCINGLY_PY/venv`  <- el duradero
      3. el interprete actual (el caso de Colab)
      4. los venv del scratchpad temporal, **avisando**, porque un dia se borran -- y se borraron
    """
    import sys as _s
    v = os.environ.get("FENCINGLY_PY_YOLO" if cual == "yolo" else "FENCINGLY_PY_HF")
    if v and pathlib.Path(v).is_file():
        return v
    p = VENV / "Scripts" / "python.exe"
    if p.is_file():
        return str(p)
    p = VENV / "bin" / "python"
    if p.is_file():
        return str(p)
    if _importa(_s.executable, "torch"):
        return _s.executable
    p = (_VENV_TMP / ("fz/.venv-gpu" if cual == "yolo" else "cuda/.venv")
         / "Scripts" / "python.exe")
    if p.is_file():
        _una_vez("venv_tmp",
                 "AVISO: usando el interprete del scratchpad TEMPORAL (%s). Se borra solo: "
                 "monta el duradero en %s. Ver INCIDENTE_2026-09-23_VENV.md" % (p, VENV))
        return str(p)
    return _s.executable


def _importa(py, modulo):
    """QUE HACE: pregunta a OTRO interprete si puede importar un modulo. Sin suponerlo."""
    import subprocess
    try:
        r = subprocess.run([py, "-c", "import %s" % modulo], capture_output=True, timeout=120)
        return r.returncode == 0
    except Exception:
        return False


def comprueba_interprete(py=None, modulos=("torch", "cv2", "numpy", "ultralytics",
                                           "transformers")):
    """QUE DEVUELVE: {modulo: True/False} preguntandole al interprete de verdad.

    PARA QUE: `noche.py` lanzo cinco trabajadores contra un interprete roto y **no se entero hasta
    18 minutos despues**, cuando un trabajador dijo "una vuelta sin completar nada". Esta funcion
    convierte eso en un fallo del segundo 1.
    """
    import subprocess
    py = py or interprete()
    codigo = ("import importlib, json, sys\n"
              "d = {}\n"
              "for m in %r:\n"
              "    try:\n"
              "        importlib.import_module(m); d[m] = True\n"
              "    except Exception:\n"
              "        d[m] = False\n"
              "try:\n"
              "    import torch; d['cuda'] = bool(torch.cuda.is_available())\n"
              "except Exception:\n"
              "    d['cuda'] = False\n"
              "print(json.dumps(d))\n" % (list(modulos),))
    try:
        r = subprocess.run([py, "-c", codigo], capture_output=True, text=True, timeout=300)
        import json
        return json.loads((r.stdout or "{}").strip().splitlines()[-1])
    except Exception as ex:
        return {"__error__": "%s: %s" % (type(ex).__name__, str(ex)[:100])}

_dicho = set()


def _una_vez(clave, texto):
    """QUE HACE: imprime un aviso UNA vez por proceso. Un aviso repetido 500 veces deja de leerse
    como aviso y pasa a leerse como ruido de fondo del lote."""
    if clave not in _dicho:
        _dicho.add(clave)
        print(texto)


_indice = None
_indice_arbol = None


def indice_videos(raiz=None, recarga=False):
    """QUE DEVUELVE: dict clip_id -> ruta del mp4, recorriendo el arbol UNA vez por proceso.

    Si el directorio no existe **lo dice** y devuelve un indice vacio, en vez de devolver vacio en
    silencio. Es la diferencia entre "este corpus no tiene videos" y "FENCINGLY_VIDEOS apunta a un
    disco que en Colab no esta montado".
    """
    global _indice, _indice_arbol
    m = {}
    for d in (TRABAJO / "videos", DESCARGAS):
        """LO BAJADO DEL CDN TAMBIEN ES UN VIDEO, y esto no estaba.

        El indice solo recorria el arbol de `VIDEOS`. En Colab ese arbol **no existe**, asi que el
        indice salia vacio y todo el que buscase un clip por aqui no lo encontraba -- aunque
        acabara de bajarse. Lo cazo el ensayo `verifica_colab.py`: las etapas 1 y 4 iban bien
        porque usan `rutas.video()`, que mira las carpetas de descarga; la **etapa 7 no**, porque
        pasa por el indice, y decia `SIN HOMOGRAFIA y sin accion: sin video en disco` para todos
        los clips. Habria producido **cero series de eventos en todo el corpus de Colab**,
        informando exito por clip. Estas dos lineas son ese fallo."""
        if d.is_dir():
            for f in os.listdir(d):
                if f.lower().endswith(".mp4"):
                    m.setdefault(f[:-4], str(d / f))
    """LAS CARPETAS DE DESCARGA SE MIRAN EN CADA LLAMADA; el arbol se cachea.

    No es una inconsistencia: son dos cosas distintas. El arbol de `VIDEOS` tiene 45.008 ficheros
    y recorrerlo cuesta segundos, asi que se hace una vez. Las carpetas de descarga tienen lo que
    se acaba de bajar y **cambian durante la propia corrida**: si se cachearan, un clip bajado
    despues de la primera llamada seria invisible para siempre en ese proceso. Un `os.listdir` de
    un directorio pequenio no cuesta nada."""
    r = pathlib.Path(raiz or VIDEOS)
    if _indice_arbol is None or recarga:
        _indice_arbol = {}
        if r.is_dir():
            for d, _sub, fs in os.walk(r):
                for f in fs:
                    if f.lower().endswith(".mp4"):
                        _indice_arbol.setdefault(f[:-4], os.path.join(d, f))
        elif not m:
            _una_vez("sin_videos",
                     "AVISO: no existe %s (FENCINGLY_VIDEOS) y no hay nada bajado. Los clips "
                     "habra que bajarlos del CDN." % r)
    for k, v in _indice_arbol.items():
        m.setdefault(k, v)
    _indice = m
    return m


DESCARGAS = _env("FENCINGLY_DESCARGAS", VIDEOS / "_descargados")
"""Donde se dejan los mp4 bajados del CDN.

Por defecto **al lado del resto de videos**, no en el directorio de trabajo: en esta maquina son
2.202 clips y ~7 GB, D: tiene 1,8 TB libres y C: es el disco del sistema. En Colab no hay D:, asi
que alli se pone `FENCINGLY_DESCARGAS=/content/trabajo/videos` y se borran tras procesar -- que es
lo contrario de lo que conviene aqui, donde volver a bajarlos costaria otra vez la red."""


def video(clip, raiz=None):
    """QUE DEVUELVE: la ruta del mp4 del clip, o None.

    Mira en tres sitios y en este orden: `TRABAJO/videos` (lo que se acaba de bajar en esta
    corrida), `DESCARGAS` (lo bajado en corridas anteriores) y el indice del arbol de videos. Los
    dos primeros se comprueban por ruta directa y no por indice, porque un fichero que acaba de
    aparecer no esta en un indice que se cacheo al arrancar el proceso."""
    for d in (TRABAJO / "videos", DESCARGAS):
        p = d / ("%s.mp4" % clip)
        if p.is_file():
            return str(p)
    if raiz is None and _indice_arbol is not None:
        return _indice_arbol.get(clip)
    """(2026-10-13) Lo mismo que `indice_videos().get(clip)`, sin rehacerlo: las dos carpetas de
    descarga ya se han mirado arriba por ruta directa, asi que solo queda el arbol, que esta en cache.
    Antes, cada clip que no estaba en las descargas volvia a listar `_descargados` (~80.000 ficheros
    en la corrida) y a copiar el indice entero."""
    return indice_videos(raiz).get(clip)


def pesos_yolo():
    """QUE DEVUELVE: el .pt local si esta, y si no el nombre pelado para que ultralytics lo baje.

    Nunca devuelve None. `pathlib.Path(ULTRA).parents[2]/...` construia una ruta a partir del
    ejecutable de un venv que en Colab no existe."""
    if YOLO_PT.is_file():
        _una_vez("yolo", "yolo: pesos locales %s" % YOLO_PT)
        return str(YOLO_PT)
    _una_vez("yolo", "yolo: sin pesos locales, ultralytics bajara yolo11m-pose.pt del hub")
    return "yolo11m-pose.pt"


def hf_modelo(nombre):
    """QUE DEVUELVE: la instantanea local del modelo de Hugging Face si esta, y si no el id del
    hub. Nunca None, que es lo que hacia `snap()`.

    `sorted(...)[-1]` y no `next(iterdir())`: con dos instantaneas en cache, `next` coge una
    arbitraria y el modelo de pose puede cambiar entre dos corridas sin que nada lo diga. Esta
    funcion ademas imprime cual eligio, una vez.
    """
    d = HF / "hub" / ("models--facebook--%s" % nombre) / "snapshots"
    if d.is_dir():
        snaps = sorted(p for p in d.iterdir() if p.is_dir())
        if snaps:
            _una_vez("hf_" + nombre, "hf: %s -> %s%s"
                     % (nombre, snaps[-1].name,
                        "  (habia %d instantaneas, se coge la ultima)" % len(snaps)
                        if len(snaps) > 1 else ""))
            return str(snaps[-1])
    _una_vez("hf_" + nombre, "hf: %s -> del hub (no hay cache local)" % nombre)
    return "facebook/%s" % nombre


_roi = None


def roi_sede(torneo):
    """QUE DEVUELVE: la ROI aprendida de esa sede, o None.

    Importa quien pregunte: con ROI por sede las luces se leen al **90,5 %**, sin ella al
    **71,4 %**, y `roi_por_sede.csv` cubre **21 sedes**. Un clip de una sede que no este en la
    tabla NO tiene el 90,5 %, y por eso la salida lleva `via` y las dos tasas se dan por separado
    en vez de promediadas.
    """
    global _roi
    if _roi is None:
        import csv
        _roi = {}
        p = MUESTRAS / "luces" / "roi_por_sede.csv"
        if p.is_file():
            with open(p, encoding="utf-8") as fh:
                for f in csv.DictReader(fh):
                    s = f.get("sede") or f.get("torneo")
                    try:
                        y0, y1 = float(f["fy"]), float(f["fy"]) + float(f["fh"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    d = _roi.setdefault(s, {"sede": s, "y0": y0, "y1": y1, "canales": []})
                    d["y0"] = min(d["y0"], y0)
                    d["y1"] = max(d["y1"], y1)
                    d["canales"].append(f.get("canal"))
        for d in _roi.values():
            d["banda"] = (max(0.0, d["y0"] - 0.01), min(1.0, d["y1"] + 0.01))
            """LA UNION DE LOS CANALES, no el ultimo que se leyo.

            `roi_por_sede.csv` trae **una fila por (sede, canal)**: la roja, la verde y la blanca
            de una misma sede estan en tres filas distintas. Un dict indexado por sede se queda
            con la ultima, y entonces el detector mira la banda de un solo canal y no encuentra
            los otros dos. La banda es la union de las tres, con un margen de 0,01 del alto."""
    return _roi.get(torneo)


def informe():
    """QUE HACE: imprime cada raiz, si sale de una variable de entorno o del defecto, y si existe.
    Es la primera celda del cuaderno de Colab: si algo esta mal, se ve aqui y no tres horas
    despues en forma de 'NO VIDEO, skipped'."""
    filas = [
        ("RAIZ", RAIZ, "derivada"),
        ("CODIGO", CODIGO, "derivada"),
        ("TRABAJO", TRABAJO, _FUENTE.get("FENCINGLY_TRABAJO")),
        ("VIDEOS", VIDEOS, _FUENTE.get("FENCINGLY_VIDEOS")),
        ("SCRATCH", SCRATCH, _FUENTE.get("SCRATCH")),
        ("HF_HOME", HF, _FUENTE.get("HF_HOME")),
        ("YOLO_PT", YOLO_PT, _FUENTE.get("FENCINGLY_YOLO")),
        ("CATALOGO", CATALOGO, "derivada"),
        ("MODELOS", MODELOS, "derivada"),
        ("PISTA", PISTA, "derivada"),
    ]
    print("%-10s %-8s %-6s %s" % ("raiz", "de", "existe", "valor"))
    print("-" * 100)
    for nom, val, de in filas:
        print("%-10s %-8s %-6s %s"
              % (nom, de, "si" if pathlib.Path(val).exists() else "NO", val))
    print()
    print("yolo   : %s" % pesos_yolo())
    print("sapiens: %s" % hf_modelo("sapiens2-pose-0.4b"))


if __name__ == "__main__":
    informe()

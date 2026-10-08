"""Recorre el manifiesto por lotes, extrae, y sobrevive a que te echen de la sesion.

QUE RESUELVE. Hasta ahora el corpus se recorria con `os.walk` sobre un disco, y eso ya metio clips
de sable y de espada en un informe de florete. Aqui el recorrido es **del manifiesto**, que lleva
el arma escrita, y el orden es **fijo y declarado**, no el que devuelva el sistema de ficheros.

## LA AUTORIDAD DEL ESTADO NO ES UN CONTADOR NI UN FICHERO

Un clip esta hecho si **sus ficheros de salida existen**, no si un contador lo dice.
`estado_lotes.jsonl` es un registro de AUDITORIA -- cuanto tardo cada etapa, que fallo y por que --
y se puede borrar entero sin perder la capacidad de reanudar. Esto no es teorico: en un ensayo de
este proyecto un contador dijo "completado = 3" habiendo 6 ficheros.

La prueba de que esta bien hecho, y hay que correrla: matar el proceso a mitad de lote, relanzar
(no debe repetir nada), **borrar `estado_lotes.jsonl`** y relanzar otra vez (tampoco).

## EL ORDEN ES UNA BARAJA CON SEMILLA, Y ESO NO ES UN DETALLE

El orden se fija UNA vez barajando los clips del conjunto con `SEMILLA`. Consecuencia que importa:
**cualquier prefijo del recorrido es una muestra aleatoria del conjunto**. O sea que si la corrida
se corta a la mitad, lo extraido sigue siendo una muestra sin sesgo y se puede medir sobre ello
diciendo el n -- en vez de tener los clips de los tres primeros torneos por orden alfabetico, que
no valdrian para medir nada.

`--desde/--cuantos` indexan ese orden, no el directorio. Por eso `--desde 500` significa lo mismo
despues de una caida, en otra maquina, y despues de que alguien anada filas al manifiesto.

## LOS DOS INTERPRETES YA NO SON DOS, Y ESO SE COMPROBO

El proyecto llevaba desde el principio afirmando que en Windows `ultralytics` y `transformers`
**no coexisten**, y de ahi salian los dos venv, las "tres pasadas" del playbook y la mitad de la
complicacion de este fichero. Al reconstruir el entorno el 2026-09-23 se instalaron los dos en el
mismo venv y **conviven sin problema**: `ultralytics 8.4.160` y `transformers 5.16.1` importan en
el mismo proceso, con `torch 2.14.0+cu126` y CUDA disponible.

O sea que aquello no era "la realidad instalada": era un accidente de instalacion que nadie volvio
a comprobar. Ahora Windows y Colab son **el mismo camino**: un interprete con todo.

`rutas.interprete()` sigue aceptando dos por si alguien los quiere separados, y si los dos apuntan
al mismo ejecutable **no se lanza ningun subproceso**. Y esto es lo que sustituye a
`cual_interprete()`, que devolvia UNA etiqueta y se comparaba con `==`: en un entorno con las dos
bibliotecas esa pregunta no tiene respuesta correcta, contestaba `"ultralytics"` y las etapas 4,
7, 8 y 9 imprimian `SKIP` para siempre.

    python lotes.py --estado
    python lotes.py --conjunto anotados --cuantos 200 --lote 25 --etapas 1,4,7
    python lotes.py --conjunto foil --desde 0 --cuantos 5000 --descarga --borra-video
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import random
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "codigo"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                       / "02_validacion" / "herramienta"))
import catalogo as CA  # noqa: E402
import rutas as RU  # noqa: E402

REPO = os.environ.get("FENCINGLY_REPO", "EXCALORIO/fencingly-extraccion")
"""El dataset PRIVADO de Hugging Face. Privado y comprobado antes de cada escritura: los clips
ensenian atletas identificables."""

SEMILLA = 20260922
"""La semilla de la baraja del recorrido. Cambiarla invalida cualquier `--desde` anterior, asi que
si se cambia hay que decirlo y volver a contar."""

ESTADO = RU.TRABAJO / "estado_lotes.jsonl"

PY_YOLO = RU.interprete("yolo")
PY_HF = RU.interprete("hf")
"""Se DESCUBREN en `rutas.interprete`, no hay que exportarlos antes de cada lanzamiento. Olvidar
una variable de entorno no daba un error: daba una corrida que saltaba media tuberia con un
mensaje ordenado, que es la peor forma de fallar."""

SALIDAS = {
    1: ["det_%s.pkl", "revision_%s.npz"],
    4: ["poses_%s.npz"],
    7: [],   # escribe en 02_validacion/pista, no en TRABAJO; se comprueba aparte
    9: ["lights_%s.csv"],
    10: ["etiquetas_%s.csv"],
    11: ["zona_%s.csv"],
}
"""QUE FICHERO PRUEBA QUE UNA ETAPA ESTA HECHA. Una sola lista, la misma que usan `hechas()` y el
informe: dos listas es como aparecieron los dos directorios de `det_`."""


DEPENDE = {1: (), 4: (1,), 7: (1, 4), 9: (), 10: (1, 4, 7), 11: (1, 4, 9)}
NECESITAN_VIDEO = (1, 7, 9, 11)
"""Las etapas que leen el mp4. La 7 tambien: la pista sale de los fotogramas."""
"""QUE ETAPA NECESITA QUE OTRA HAYA CORRIDO ANTES.

Sin esto, una corrida de `--etapas 7` sobre clips a los que les falta la etapa 4 los intenta uno a
uno, falla uno a uno y **vuelve a intentarlos en la siguiente pasada**, porque nada quedo escrito.
El coste no es el fallo, es el bucle: en un lote de 3.654 clips son 3.654 intentos inutiles por
vuelta. Una etapa que no puede correr lo dice y se salta, en vez de fallar en silencio."""


def _sal_pista(clip):
    return [RU.PISTA / ("serie_eventos_%s.csv" % clip),
            RU.PISTA / ("posicion_clip_%s.csv" % clip)]


def _pista_vacia(clip):
    return RU.TRABAJO / ("pista_vacia_%s.txt" % clip)
"""LA ETAPA 7 TAMBIEN PUEDE CONTESTAR "NADA", y eso es una respuesta, no un fallo. Cuando
`GP.procesa` devuelve None con el video en disco (sin homografia y sin accion, menos de 12 cuadros
utilizables...) no escribe ninguno de sus dos CSV, y `hechas()` lo daba por pendiente para siempre:
medido el 2026-10-08, los trabajadores de CPU reintentaban esos clips en cada vuelta. Esta marca
dice "se hizo y no habia nada"; se escribe SOLO si el video estaba, para que un "sin video en
disco" se siga reintentando cuando llegue la descarga."""


def conjunto(nombre, arma=None):
    """QUE DEVUELVE: la lista de clips del conjunto pedido, ya en el ORDEN barajado y fijo.

    `anotados`    los que tienen alguna de las tres etiquetas humanas
    `initiated`   los que tienen la etiqueta de quien inicia (el denominador de la cabeza 1)
    `disco`       los anotados cuyo mp4 ya esta, que es lo que la puerta [C] puede usar sin bajar
    `foil`        todo el florete declarado
    un fichero    una lista de ids, uno por linea
    """
    if os.path.exists(nombre):
        ids = [l.strip() for l in open(nombre, encoding="utf-8") if l.strip()]
    else:
        df = CA.manifiesto()
        if nombre == "anotados":
            d = CA.anotados(arma=arma)
        elif nombre == "initiated":
            d = CA.anotados(arma=arma, etiqueta="initiated")
        elif nombre == "disco":
            d = CA.anotados(arma=arma, solo_en_disco=True)
        elif nombre == "foil":
            d = df.loc[df["weapon"] == "foil"]
        elif nombre == "todo":
            d = df if arma is None else df.loc[df["weapon"] == arma]
        else:
            raise SystemExit("conjunto desconocido: %s" % nombre)
        ids = sorted(d["clip_id"])
    r = random.Random(SEMILLA)
    r.shuffle(ids)
    return ids


def hechas(clip, etapas, remoto=None):
    """QUE DEVUELVE: el conjunto de etapas ya hechas de ese clip.

    EL ESTADO ES LA UNION DE DOS SITIOS QUE EXISTEN, nunca un contador:

      **el disco**    los ficheros de salida estan y no estan vacios. Un fichero de 0 bytes es un
                      fallo a mitad de escritura y cuenta como NO hecho, porque si contara como
                      hecho la reanudacion lo saltaria para siempre.
      **el remoto**   el clip aparece en un manifiesto de lote ya subido a Hugging Face.

    POR QUE HACE FALTA EL SEGUNDO, y no es un adorno: **en Colab el disco se va con la sesion**.
    Sin mirar el remoto, cada sesion nueva empieza de cero y vuelve a extraer lo que la anterior
    ya subio -- que con 5.856 clips a ~100 clips/h es tirar sesiones enteras. En esta maquina el
    disco sobrevive y `remoto` puede ir a None; en Colab NO.

    `remoto` es {clip_id: set(etapas)}, como lo devuelve `estado_remoto()`.
    """
    out = set()
    for n in etapas:
        if n == 7:
            ps = _sal_pista(clip)
            if _pista_vacia(clip).is_file():
                out.add(n)
                continue
        else:
            ps = [RU.TRABAJO / (p % clip) for p in SALIDAS.get(n, [])]
        if ps and all(p.is_file() and p.stat().st_size > 0 for p in ps):
            out.add(n)
    if remoto:
        out |= (remoto.get(clip) or set()) & set(etapas)
    return out


_REMOTO = None


def estado_remoto(repo=None, recarga=False, verboso=True):
    """QUE DEVUELVE: {clip_id: set(etapas)} de lo que ya esta subido, leyendo los MANIFIESTOS.

    Se leen los manifiestos por clip (`data/index/lotes/*.csv`) y no la lista de tars, por la
    misma razon por la que `sube()` no da un lote por subido sin su manifiesto: **un tar sin
    manifiesto no se puede descontar clip a clip**. Si el remoto no se puede leer, se devuelve
    vacio y se DICE -- tratar "no pude leer" como "no hay nada" es correcto aqui (se reprocesa, se
    pierde tiempo) y lo contrario seria saltarse clips en silencio, que es mucho peor.
    """
    global _REMOTO
    if _REMOTO is not None and not recarga:
        return _REMOTO
    out = {}
    try:
        import respaldo as RSP
        ses = RSP.sesion(exigir=True)
        api, repo = ses["api"], (repo or REPO)
        ficheros = api.list_repo_files(repo_id=repo, repo_type="dataset")
        mans = [f for f in ficheros if f.startswith("data/index/lotes/") and f.endswith(".csv")]
        from huggingface_hub import hf_hub_download
        tok = RSP.token()
        """EL TOKEN HAY QUE PASARLO AQUI TAMBIEN, y esto fue un fallo real.

        `api.list_repo_files` lo lleva dentro porque el `HfApi` se construyo con el; pero
        `hf_hub_download` es una funcion suelta y **sin `token=` da 401 en un repo privado**. Con
        el `except: continue` que habia debajo, ese 401 se convertia en "0 clips en el remoto" sin
        una sola linea de aviso -- o sea que en Colab cada sesion habria reprocesado el corpus
        entero creyendo que no habia nada subido. Lo cazo la primera prueba de subida de verdad."""
        malos = []
        for m in mans:
            try:
                local = hf_hub_download(repo_id=repo, filename=m, repo_type="dataset",
                                        token=tok)
                with open(local, encoding="utf-8") as fh:
                    for f in csv.DictReader(fh):
                        cid = f.get("clip_id")
                        if not cid:
                            continue
                        et = {int(x) for x in (f.get("etapas") or "").split(",") if x.strip()}
                        out.setdefault(cid, set()).update(et)
            except Exception as ex:
                malos.append((m, type(ex).__name__))
        if verboso:
            print("remoto %s: %d manifiestos, %d clips ya subidos" % (repo, len(mans), len(out)))
        if malos:
            """UN MANIFIESTO QUE NO SE PUEDE LEER SE DICE, NO SE SALTA. Si se saltara, sus clips
            se reprocesarian; es recuperable, pero hay que saberlo para no achacarlo a otra cosa."""
            print("  AVISO: %d manifiesto(s) no se pudieron leer, y sus clips se reprocesaran: %s"
                  % (len(malos), ", ".join("%s (%s)" % t for t in malos[:3])))
    except Exception as ex:
        if verboso:
            print("no pude leer el remoto (%s): se trata como VACIO y se reprocesa. Se pierde"
                  " tiempo, pero NO se salta ningun clip en silencio." % type(ex).__name__)
    _REMOTO = out
    return out


def apunta(reg):
    """QUE HACE: una linea de auditoria. Si esto falla, la corrida sigue: el estado no depende de
    este fichero y perder una linea de registro no puede costar un clip."""
    try:
        ESTADO.parent.mkdir(parents=True, exist_ok=True)
        with open(ESTADO, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(reg, ensure_ascii=False) + "\n")
    except Exception:
        pass


def descarga(clip, torneo, destino):
    """QUE HACE: baja el mp4 del CDN. QUE DEVUELVE: (ruta, estado).

    No lanza excepcion: devuelve el motivo. Un 404 en el clip 300 de 5.000 no puede tumbar el lote.
    Un cuerpo de menos de 10 KB no es un video, es una pagina de error, y se borra en vez de
    dejarse en disco para que la etapa 1 falle mas tarde y mas lejos.
    """
    import urllib.request
    destino.mkdir(parents=True, exist_ok=True)
    p = destino / ("%s.mp4" % clip)
    if p.is_file() and p.stat().st_size > 10240:
        return str(p), "ya_estaba"
    url = CA.url(clip, torneo)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(p, "wb") as fh:
            fh.write(r.read())
    except Exception as ex:
        return None, "%s: %s" % (type(ex).__name__, str(ex)[:60])
    if p.stat().st_size <= 10240:
        n = p.stat().st_size
        p.unlink()
        return None, "cuerpo de %d B, no es un video" % n
    return str(p), "bajado"


def _lanza(py, codigo, clips, etiqueta):
    """QUE HACE: corre un trozo de codigo en OTRO interprete, o en este si es el mismo.

    Cuando `py == sys.executable` no se lanza subproceso: en Colab hay un solo entorno y pagar un
    arranque de Python por etapa y por lote seria tirar minutos.
    """
    t = time.perf_counter()
    if py == sys.executable:
        g = {"CLIPS": clips, "__name__": "__lote__"}
        try:
            exec(compile(codigo, "<%s>" % etiqueta, "exec"), g)
            return time.perf_counter() - t, ""
        except Exception as ex:
            return time.perf_counter() - t, "%s: %s" % (type(ex).__name__, str(ex)[:120])
    pre = ("import sys, json\n"
           "sys.path.insert(0, %r)\nsys.path.insert(0, %r)\n"
           "CLIPS = json.loads(%r)\n"
           % (str(RU.CODIGO), str(RU.HERRAMIENTA), json.dumps(clips)))
    r = subprocess.run([py, "-c", pre + codigo], capture_output=True, text=True,
                       creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
                       encoding="utf-8", errors="replace")
    d = time.perf_counter() - t
    if r.returncode != 0:
        return d, (r.stderr or "")[-300:]
    sys.stdout.write(r.stdout)
    return d, ""


CODIGO_1 = """
import video_revision as VR
VR.fase1(CLIPS)
"""

CODIGO_4 = """
import video_revision as VR
VR.fase2(CLIPS, video=False)
"""

CODIGO_7 = """
import grafica_pista as GP
import rutas as RU
for c in CLIPS:
    try:
        f = GP.procesa(c, con_figura=False)
        if f is None and RU.video(c):
            (RU.TRABAJO / ("pista_vacia_%s.txt" % c)).write_text("sin homografia y sin accion",
                                                                encoding="utf-8")
    except Exception as ex:
        print("%-38s ETAPA 7 FALLA %s: %s" % (c, type(ex).__name__, str(ex)[:80]))
"""

CODIGO_9 = """
import sys
import catalogo as CA
import rutas as RU
sys.path.insert(0, str(RU.CODIGO.parent))
import pipeline as PP
df = CA.manifiesto().set_index("clip_id")
for c in CLIPS:
    try:
        PP._luces(c, {"video": RU.video(c), "torneo": df.loc[c, "torneo"] if c in df.index else None})
    except Exception as ex:
        print("%-38s ETAPA 9 FALLA %s: %s" % (c, type(ex).__name__, str(ex)[:80]))
"""

CODIGO_11 = """
import catalogo as CA
import rutas as RU
import zona_tocado as ZT
df = CA.manifiesto().set_index("clip_id")
for c in CLIPS:
    try:
        ZT.corre(c, {"video": RU.video(c), "torneo": df.loc[c, "torneo"] if c in df.index else None})
    except Exception as ex:
        print("%-38s ETAPA 11 FALLA %s: %s" % (c, type(ex).__name__, str(ex)[:80]))
"""

CODIGO_10 = """
import etiquetador as ET
ET.etiqueta_clips(CLIPS)
"""

CODIGOS = {1: (CODIGO_1, "PY_YOLO"), 4: (CODIGO_4, "PY_HF"),
           7: (CODIGO_7, "PY_HF"), 9: (CODIGO_9, "PY_HF"), 10: (CODIGO_10, "PY_HF"),
           11: (CODIGO_11, "PY_HF")}


def corre(clips, etapas, descargar=False, borrar=False, lote=25, verboso=True,
          remoto=None):
    """QUE HACE: el bucle. Por lote: bajar si toca -> etapas en orden -> borrar el mp4 -> apuntar.

    Las etapas se aplican **al lote entero**, no clip a clip, y esa es la razon de que el lote
    exista: cargar YOLO cuesta ~2 s y cargar Sapiens ~5 s, asi que hacerlo por clip multiplicaria
    el coste de un corpus de 100.000 clips por un factor que no tiene nada que ver con el trabajo.
    """
    df = CA.manifiesto().set_index("clip_id")
    hecho = fallo = 0
    t0 = time.perf_counter()
    for i in range(0, len(clips), lote):
        trozo = clips[i:i + lote]
        pend = {n: [c for c in trozo if n not in hechas(c, [n], remoto)]
                for n in etapas}
        corridos = set()
        if descargar:
            vivos = []
            for c in trozo:
                if not any(c in pend[n] for n in etapas):
                    continue
                if RU.video(c):
                    vivos.append(c)
                    continue
                tor = df.loc[c, "torneo"] if c in df.index else None
                if tor is None:
                    apunta({"clip": c, "estado": "sin torneo en el manifiesto"})
                    continue
                _r, est = descarga(c, tor, RU.DESCARGAS)
                if _r:
                    vivos.append(c)
                else:
                    apunta({"clip": c, "estado": "descarga: %s" % est})
            if verboso:
                print("lote %d-%d: %d con video" % (i, i + len(trozo) - 1, len(vivos)))
        for n in etapas:
            cs = [c for c in pend[n]
                  if (RU.video(c) or n not in NECESITAN_VIDEO)
                  and hechas(c, DEPENDE[n], remoto) == set(DEPENDE[n])]
            saltados = len(pend[n]) - len(cs)
            if saltados and verboso:
                print("   etapa %-2d  %3d clips saltados: les falta %s"
                      % (n, saltados, "el video" if n in NECESITAN_VIDEO and n == 1 else "la entrada o el video"))
            if not cs:
                continue
            corridos.update(cs)
            codigo, cual = CODIGOS[n]
            d, err = _lanza(PY_YOLO if cual == "PY_YOLO" else PY_HF, codigo, cs, "etapa%d" % n)
            if verboso:
                print("   etapa %-2d  %3d clips  %6.1f s  %5.2f s/clip%s"
                      % (n, len(cs), d, d / max(len(cs), 1), ("  FALLO: " + err) if err else ""))
            apunta({"lote": i, "etapa": n, "n": len(cs), "s": round(d, 1), "error": err})
        """`corridos` son los clips a los que una etapa se les ejecuto DE VERDAD en esta vuelta,
        no los que estaban pendientes. Es la diferencia entre un registro y un diario de
        pendientes: con "pendientes" el fichero crecia a **609.199 lineas y 46 MB** en unas horas,
        porque cada vuelta reapuntaba los miles de clips que aun no tocaban. Un registro que crece
        mas rapido que el trabajo esconde las lineas que importan."""
        for c in trozo:
            h = hechas(c, etapas, remoto)
            if h == set(etapas):
                hecho += 1
            else:
                fallo += 1
                if c in corridos:
                    apunta({"clip": c, "estado": "incompleto", "hechas": sorted(h)})
                """SOLO SE APUNTA LO QUE SE INTENTO EN ESTA PASADA.

                Sin esta condicion, cada vuelta del bucle escribia una linea por cada clip que aun
                no estaba hecho -- incluidos los que ni siquiera se intentaron porque les faltaba
                la entrada. Medido: el registro llego a **28.995 lineas** habiendo 40 clips
                completos. Un registro que crece mas rapido que el trabajo deja de ser un registro
                y pasa a ser un problema de disco, y ademas esconde las lineas que importan."""
            if borrar:
                p = RU.TRABAJO / "videos" / ("%s.mp4" % c)
                if p.is_file():
                    p.unlink()
        if verboso:
            el = time.perf_counter() - t0
            n = i + len(trozo)
            print("   -> %d/%d completos, %d incompletos, %.0f s, ritmo %.1f s/clip"
                  % (hecho, n, fallo, el, el / max(n, 1)))
    return {"hechos": hecho, "incompletos": fallo, "s": time.perf_counter() - t0}


def sube(clips, etapas, repo=None, etiqueta=None, verboso=True):
    """QUE HACE: empaqueta las salidas de los clips HECHOS y las sube al repo privado de HF.

    POR QUE ESTO EXISTE. En Colab el disco se va con la sesion. Sin una vuelta de los resultados,
    una corrida de Colab produce datos que **no salen de alli**, que es lo mismo que no producir
    nada. Es el hueco que quedaba para que la extraccion en masa sirviese de algo.

    QUE SUBE Y QUE NO. Sube `revision_`, `poses_`, `etiquetas_` y los tres CSV de la pista. **No
    sube los mp4**: son emision de television con atletas identificables, pesan lo que pesan y ya
    estan en el CDN de origen. Y **no sube `det_*.pkl`**: es cache de detecciones, se regenera.

    DOS COSAS QUE NO SE NEGOCIAN, y las dos vienen de `respaldo.py`:
      - el repositorio tiene que ser **privado**, y se comprueba antes de escribir;
      - un lote no se da por subido si falta su **manifiesto**, porque un tar sin manifiesto no se
        puede descontar clip a clip al reanudar.
    """
    import respaldo as RSP
    ses = RSP.sesion(exigir=True)
    api, repo = ses["api"], (repo or REPO)
    RSP.verifica_repo(api, repo, privado_esperado=True)
    hechos = [c for c in clips if hechas(c, etapas) == set(etapas)]
    """OJO: aqui se mira SOLO el disco, sin `remoto`. Subir lo que ya esta subido seria duplicar;
    pero lo que hay que subir es lo que esta EN ESTE DISCO, y si un clip ya estuviera en el remoto
    no estaria pendiente de subirse porque `corre()` no lo habria reprocesado."""
    if not hechos:
        if verboso:
            print("no hay ningun clip completo que subir")
        return {"clips": 0}
    etiqueta = etiqueta or time.strftime("%Y%m%d-%H%M")
    import tempfile
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="fz_sube_"))
    rutas_ = []
    for c in hechos:
        for p in ("revision_%s.npz", "poses_%s.npz", "etiquetas_%s.csv", "lights_%s.csv",
                  "zona_%s.csv", "pista_vacia_%s.txt"):
            """lights_, zona_ y pista_vacia_ se anadieron el 2026-10-08 con las etapas 9 y 11:
            sin ellos el remoto decia "etapas 9,11 hechas" sin llevar su salida."""
            f = RU.TRABAJO / (p % c)
            if f.is_file():
                rutas_.append(f)
        for f in _sal_pista(c) + [RU.PISTA / ("eventos_%s.csv" % c)]:
            if f.is_file():
                rutas_.append(f)
    tar = tmp / ("lote_%s.tar.gz" % etiqueta)
    inf = RSP.empaqueta(rutas_, tar)
    man = tmp / ("lote_%s.csv" % etiqueta)
    with open(man, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "etapas"])
        for c in hechos:
            w.writerow([c, ",".join(str(n) for n in sorted(etapas))])
    r1 = RSP.sube_lote(api, repo, tar, "data/lotes/%s" % tar.name,
                       "lote %s: %d clips" % (etiqueta, len(hechos)))
    r2 = RSP.sube_lote(api, repo, man, "data/index/lotes/%s" % man.name,
                       "manifiesto del lote %s" % etiqueta)
    ok = bool(r1.get("ok")) and bool(r2.get("ok"))
    if verboso:
        print("lote %s: %d clips, %d ficheros, %.1f MB -> %s"
              % (etiqueta, len(hechos), inf.get("n", len(rutas_)),
                 tar.stat().st_size / 1e6, repo))
        print("   tar %s, manifiesto %s -> %s"
              % ("ok" if r1.get("ok") else "FALLO", "ok" if r2.get("ok") else "FALLO",
                 "SUBIDO" if ok else "NO se da por subido"))
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return {"clips": len(hechos), "ok": ok, "etiqueta": etiqueta}


def informe(clips, etapas, remoto=None):
    """QUE HACE: cuenta lo hecho LEYENDO EL DISCO, sin abrir `estado_lotes.jsonl`."""
    por = {n: 0 for n in etapas}
    todo = 0
    for c in clips:
        h = hechas(c, etapas, remoto)
        for n in h:
            por[n] += 1
        if h == set(etapas):
            todo += 1
    print("conjunto: %d clips" % len(clips))
    for n in etapas:
        print("   etapa %-2d hecha en %6d  (%.1f %%)" % (n, por[n], 100.0 * por[n] / max(len(clips), 1)))
    print("   las %s a la vez: %d  (%.1f %%)"
          % (etapas, todo, 100.0 * todo / max(len(clips), 1)))
    print()
    print("registro de auditoria: %s  (%s)"
          % (ESTADO, "%d lineas" % sum(1 for _ in open(ESTADO, encoding="utf-8"))
             if ESTADO.is_file() else "no existe, y no hace falta"))
    return todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", default="disco")
    ap.add_argument("--arma", default=None)
    ap.add_argument("--desde", type=int, default=0)
    ap.add_argument("--cuantos", type=int, default=None)
    ap.add_argument("--lote", type=int, default=25)
    ap.add_argument("--etapas", default="1,4,7")
    ap.add_argument("--descarga", action="store_true")
    ap.add_argument("--borra-video", action="store_true", dest="borra")
    ap.add_argument("--estado", action="store_true")
    ap.add_argument("--sube", action="store_true",
                    help="empaqueta y sube a Hugging Face lo que este completo")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--remoto", action="store_true",
                    help="cuenta como hecho lo que ya este subido a HF. Imprescindible en Colab, "
                         "donde el disco se va con la sesion; innecesario aqui.")
    a = ap.parse_args()
    etapas = [int(x) for x in a.etapas.split(",") if x.strip()]
    ids = conjunto(a.conjunto, a.arma)
    trozo = ids[a.desde:] if a.cuantos is None else ids[a.desde:a.desde + a.cuantos]
    print("conjunto %s%s: %d clips; se toman %d desde el %d (orden barajado con semilla %d)"
          % (a.conjunto, (" arma=%s" % a.arma) if a.arma else "", len(ids), len(trozo),
             a.desde, SEMILLA))
    rem = estado_remoto(a.repo) if a.remoto else None
    if a.estado:
        informe(trozo, etapas, rem)
        return 0
    if a.sube:
        sube(trozo, etapas, repo=a.repo)
        return 0
    r = corre(trozo, etapas, descargar=a.descarga, borrar=a.borra, lote=a.lote,
              remoto=rem)
    print()
    print("completos %d, incompletos %d, %.0f s" % (r["hechos"], r["incompletos"], r["s"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Ensaya el camino de Colab AQUI, antes de gastar una sesion de Colab en descubrir que no va.

## POR QUE ESTO EXISTE

El cuaderno de Colab estaba escrito y **nunca se habia ejecutado**. Un cuaderno sin correr es una
intencion, no una herramienta: la unica forma de saber si funciona era abrir Colab, y alli cada
fallo cuesta una sesion, la descarga del modelo y el enfado.

Lo que se puede ensayar aqui es casi todo, porque **Colab y esta maquina ya son el mismo camino**
desde que hay un solo interprete. Lo que se simula es el entorno:

    FENCINGLY_TRABAJO   a un directorio vacio            (en Colab, /content/trabajo)
    FENCINGLY_VIDEOS    a un directorio que NO EXISTE    (en Colab no hay D:)
    FENCINGLY_DESCARGAS al mismo directorio de trabajo   (en Colab se bajan y se borran)

Ese segundo punto es el importante. En Colab **no hay arbol de videos**, y la forma en que eso
fallaba antes no era un error: `os.walk` de un directorio inexistente devuelve vacio, asi que el
lote imprimia "NO VIDEO, skipped" quinientas veces y se leia como un problema de corpus. Aqui se
comprueba que **dice que falta el directorio, una vez**, y que el clip se baja del CDN igualmente.

## LO QUE SE COMPRUEBA, Y EL CRITERIO DE CADA COSA

    1  prevuelo                el interprete importa lo que hace falta
    2  rutas                   las ocho raices resuelven, y VIDEOS inexistente AVISA
    3  catalogo                el manifiesto se lee desde cero en el entorno simulado
    4  descarga                el mp4 baja del CDN a un directorio vacio
    5  etapas 1,4,7 en UN proceso   ninguna dice "no esta instalado"
    6  etapa 10                escribe `etiquetas_<clip>.csv`, o dice que no hay pesos
    7  borrado                 tras `borrar=True` no queda ningun mp4
    8  disco                   el delta es el de las SALIDAS, no el de los videos

    python verifica_colab.py [--n 2]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

RAIZ = pathlib.Path(__file__).resolve().parents[2]

GUION = r'''
import glob, os, pathlib, shutil, sys, time
sys.path.insert(0, r"{raiz}/01_extraccion")
sys.path.insert(0, r"{raiz}/01_extraccion/codigo")
sys.path.insert(0, r"{raiz}/02_validacion/herramienta")

fallos = []
def comprueba(n, ok, detalle=""):
    print("  %-2s %-46s %s%s" % (n, ok[1], "OK" if ok[0] else "FALLA",
                                 ("  " + detalle) if detalle else ""))
    if not ok[0]:
        fallos.append(ok[1])

print("1. PREVUELO")
import noche as NO
vale, inf = NO.prevuelo(etapas=(1, 4, 7))
for l in inf.splitlines():
    print("   " + l)
comprueba("1", (vale, "el interprete importa lo que hace falta"))

print()
print("2. RUTAS EN EL ENTORNO DE COLAB")
import rutas as RU
print("   TRABAJO   %s" % RU.TRABAJO)
print("   VIDEOS    %s  (existe: %s)" % (RU.VIDEOS, RU.VIDEOS.exists()))
print("   DESCARGAS %s" % RU.DESCARGAS)
comprueba("2a", (not RU.VIDEOS.exists(), "VIDEOS NO existe, como en Colab"))
idx = RU.indice_videos()
comprueba("2b", (idx == {{}}, "el indice sale vacio y AVISA una sola vez"),
          "(el aviso tiene que verse justo arriba)")

print()
print("3. CATALOGO DESDE CERO")
import catalogo as CA
df = CA.manifiesto()
comprueba("3", (len(df) == 362950, "el manifiesto se lee entero"), "%d filas" % len(df))

print()
print("4. DESCARGA DEL CDN A UN DIRECTORIO VACIO")
import lotes as LO
ids = LO.conjunto("anotados")[:{n}]
sub = df.set_index("clip_id")
bajados = []
for c in ids:
    r, est = LO.descarga(c, sub.loc[c, "torneo"], RU.DESCARGAS)
    print("   %-40s %-12s %s" % (c[:40], est,
          ("%.1f MB" % (os.path.getsize(r)/1e6)) if r else ""))
    if r:
        bajados.append(c)
comprueba("4", (len(bajados) == len(ids), "se bajan los %d clips" % len(ids)))

print()
print("5. LAS ETAPAS EN UN SOLO PROCESO")
import pipeline as PL
t0 = time.perf_counter()
filas = PL.corre(bajados, etapas=[1, 4, 7, 10], verboso=True)
t = time.perf_counter() - t0
sin_lib = [f for f in filas if f["estado"] == "skip" and "no esta instalado" in f["mensaje"]]
comprueba("5a", (not sin_lib, "ninguna etapa se salta por falta de biblioteca"),
          "%d se saltan" % len(sin_lib))
fallan = [f for f in filas if f["estado"] == "fallo"]
comprueba("5b", (not fallan, "ninguna etapa falla"),
          "; ".join("et%d: %s" % (f["etapa"], f["mensaje"][:60]) for f in fallan[:3]))
print("   %.0f s para %d clips -> %.1f s/clip" % (t, len(bajados), t / max(len(bajados), 1)))

print()
print("6. LA ETAPA 10")
e10 = [f for f in filas if f["etapa"] == 10]
hay_pesos = (RU.MODELOS / "etiquetador_initiated.pt").is_file()
msg = e10[0]["mensaje"] if e10 else "(no corrio)"
if hay_pesos:
    n_csv = len(glob.glob(str(RU.TRABAJO / "etiquetas_*.csv")))
    comprueba("6", (n_csv > 0, "escribe etiquetas_<clip>.csv"), "%d ficheros" % n_csv)
else:
    comprueba("6", ("no hay pesos" in msg or "no weights" in msg,
                    "sin pesos, lo DICE en vez de fallar"), msg[:60])

print()
print("7. EL BORRADO DEL MP4")
r = LO.corre(bajados, [1], borrar=True, lote=len(bajados), verboso=False)
quedan = glob.glob(str(RU.DESCARGAS / "*.mp4")) + glob.glob(str(RU.TRABAJO / "videos" / "*.mp4"))
comprueba("7", (not quedan, "no queda ningun mp4 tras borrar=True"), "%d quedan" % len(quedan))

print()
print("8. QUE QUEDA EN DISCO")
tot = 0
for pat in ("det_*.pkl", "revision_*.npz", "poses_*.npz", "etiquetas_*.csv"):
    fs = glob.glob(str(RU.TRABAJO / pat))
    mb = sum(os.path.getsize(f) for f in fs) / 1e6
    tot += mb
    print("   %-20s %3d ficheros  %6.2f MB" % (pat, len(fs), mb))
print("   %-20s %18.2f MB  (%.2f MB por clip)" % ("TOTAL", tot, tot / max(len(bajados), 1)))

print()
print("=" * 70)
if fallos:
    print("FALLA en %d comprobaciones:" % len(fallos))
    for f in fallos:
        print("   - %s" % f)
    sys.exit(1)
print("EL CAMINO DE COLAB FUNCIONA de punta a punta en un entorno simulado.")
sys.exit(0)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--conserva", action="store_true",
                    help="no borra el directorio simulado, para poder mirarlo")
    a = ap.parse_args()

    sys.path.insert(0, str(RAIZ / "01_extraccion" / "codigo"))
    import rutas as RU

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="fz_colab_"))
    trabajo = tmp / "trabajo"
    trabajo.mkdir(parents=True)
    env = dict(os.environ)
    env["FENCINGLY_TRABAJO"] = str(trabajo)
    env["FENCINGLY_VIDEOS"] = str(tmp / "no_existe_como_en_colab")
    env["FENCINGLY_DESCARGAS"] = str(trabajo / "videos")
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("FENCINGLY_PY_YOLO", None)
    env.pop("FENCINGLY_PY_HF", None)

    print("ENSAYO DEL CAMINO DE COLAB, EN LOCAL")
    print("=" * 70)
    print("trabajo simulado : %s" % trabajo)
    print("VIDEOS apunta a  : %s  (a proposito, NO existe)" % env["FENCINGLY_VIDEOS"])
    print("interprete       : %s" % RU.interprete())
    print()
    guion = GUION.format(raiz=str(RAIZ).replace("\\", "/"), n=a.n)
    r = subprocess.run([RU.interprete(), "-u", "-c", guion], env=env,
                       text=True, encoding="utf-8", errors="replace")
    if not a.conserva:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print()
        print("se conserva %s" % tmp)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())

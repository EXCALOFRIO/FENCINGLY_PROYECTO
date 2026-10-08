"""EL CUADERNO DE COLAB, escrito como fichero .py para que se pueda leer en un diff.

Un `.ipynb` es JSON con la salida incrustada: no se revisa, no se compara entre versiones y crece
sin que nadie lo mire -- el que hay en el repo viejo pesa **9,0 MB**. Esto son las mismas celdas en
texto. En Colab se pega cada bloque `# %% CELDA n` en una celda, o se ejecuta entero con
`%run colab_extraccion.py`.

UNA COSA QUE CAMBIO Y SIMPLIFICA ESTE CUADERNO. Este fichero se escribio dando por hecho que en
Windows hacian falta dos venv y que Colab era el caso raro donde todo cabia junto. **Es al reves:
el caso raro eran los dos venv.** Al reconstruir el entorno el 2026-09-23 se comprobo que
`ultralytics 8.4.160` y `transformers 5.16.1` conviven en el mismo proceso sin problema. Asi que
Colab y Windows corren **el mismo camino**, y lo que se prueba aqui se prueba tambien alli.

LO QUE ESTE CUADERNO TIENE QUE DEMOSTRAR, declarado antes de correrlo:

  1  que las diez etapas corren sobre UN clip en UN proceso -- lo que la pregunta "en que
     interprete estoy" impedia, porque contestaba `"ultralytics"` en cuanto ese import funcionaba
     y dejaba las etapas 4, 5, 7, 8 y 9 en `SKIP` para siempre;
  2  que 5 clips se bajan del CDN, se procesan y **se borra el mp4**, con el pico de disco medido;
  3  que `p10` y mediana del tiempo por clip **convergen**. En la 3060 compartida no convergen
     (razon 1,73x solo en la fase 1) y esa dispersion es contencion, no trabajo. Si tampoco
     convergen en Colab, la medida no vale y hay que ver por que **antes** de optimizar nada;
  4  la cifra de clips/hora **medida alli**, no extrapolada de la 3060;
  5  que la etapa 10 escribe las tres etiquetas con `origen_etiqueta = modelo`.

Y UNA COSA QUE NO SE HACE: exigir `preflight.VERSIONES`. Esos candados existen para la huella de
determinismo del ENTRENAMIENTO -- numpy 2.5.2 contra 2.5.3 cambia un hash e invalida 23.091
artefactos -- y la extraccion no depende de eso. Colab no va a dar python 3.14.2. Se llama con
`exigir=False`, se apuntan las versiones observadas en el manifiesto del lote, y se cierra la
puerta solo sobre `torch.cuda.is_available()`.
"""

# %% CELDA 1 - entorno, clon e informe de rutas
CELDA_1 = r"""
!git clone --depth 1 https://github.com/EXCALOFRIO/FENCINGLY.AI /content/fz 2>/dev/null || true
!pip -q install -r /content/fz/requirements-colab.txt

import os, sys
os.environ["FENCINGLY_TRABAJO"] = "/content/trabajo"
os.environ["FENCINGLY_VIDEOS"]  = "/content/trabajo/videos"
os.environ["HF_HOME"]           = "/content/hf"
# EL CACHE DE HF NO VA EN DRIVE. Son miles de ficheros pequenios y la capa FUSE de Drive los
# sirve a paso de tortuga: volver a bajar sapiens2-pose-0.4b una vez por sesion es mas rapido
# que una tormenta de aciertos de cache contra Drive.
os.makedirs("/content/trabajo/videos", exist_ok=True)

sys.path.insert(0, "/content/fz/01_extraccion")
sys.path.insert(0, "/content/fz/01_extraccion/codigo")
sys.path.insert(0, "/content/fz/02_validacion/herramienta")

import rutas as RU
RU.informe()

# EL PREVUELO ANTES QUE NADA. Pregunta al interprete que puede importar de verdad, en vez de
# suponerlo. En Windows se perdio una noche entera de GPU por lanzar contra un entorno roto y no
# enterarse hasta 18 minutos despues; aqui cuesta cinco segundos.
import noche as NO
vale, inf = NO.prevuelo()
print(inf)
assert vale, "el entorno no tiene lo que hace falta; no sigas"

import torch
print()
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
import pipeline as PL
print("puedo importar:", ", ".join(sorted(PL.capacidades())))
"""

# %% CELDA 2 - el clip de control y las diez etapas en UN proceso
CELDA_2 = r"""
import catalogo as CA, lotes as LO, pipeline as PL

CONTROL = "SeparateHoarseFeline"
# EL CLIP DE CONTROL ES EL INDICE DE VELOCIDAD DE LA SESION. 300 cuadros, 1920x1080, y esta en
# los 50 con los que se midio la linea base. Se corre al principio, a la mitad y al final: dos
# sesiones solo son comparables despues de dividir por su indice, y una sesion cuyo indice cambie
# mas de un 15 % entre el principio y el final se publica como RANGO y no como mediana, porque la
# tarjeta cambio debajo.

df = CA.manifiesto().set_index("clip_id")
tor = df.loc[CONTROL, "torneo"]
LO.descarga(CONTROL, tor, RU.TRABAJO / "videos")
filas = PL.corre([CONTROL], etapas=[1, 4, 7, 9, 10])
for f in filas:
    print(f)
# Criterio 1: ninguna etapa puede decir "no esta instalado". Si lo dice, falta un paquete.
malas = [f for f in filas if f["estado"] == "skip" and "no esta instalado" in f["mensaje"]]
print()
print("etapas que se saltan por falta de biblioteca:", len(malas), "  (tiene que ser 0)")
"""

# %% CELDA 3 - los 5 clips de punta a punta, con descarga y BORRADO
CELDA_3 = r"""
import shutil, time
import lotes as LO

def disco_mb(p="/content"):
    t, u, f = shutil.disk_usage(p)
    return u / 1e6

ids = LO.conjunto("anotados")[:5]
print("los 5 del manifiesto, en el orden barajado con semilla %d:" % LO.SEMILLA)
for i, c in enumerate(ids):
    print("  %d %s" % (i, c))

antes = disco_mb()
t0 = time.perf_counter()
r = LO.corre(ids, [1, 4, 7, 10], descargar=True, borrar=True, lote=5)
# En Colab SI se borra el mp4 (`borrar=True`): el disco se va con la sesion. En Windows no, que es
# lo contrario y esta puesto con FENCINGLY_DESCARGAS -> D:, porque volver a bajarlos cuesta red.
t = time.perf_counter() - t0
print()
print("completos %d de 5 en %.0f s -> %.1f s/clip" % (r["hechos"], t, t / 5))
print("disco usado antes %.0f MB, despues %.0f MB, delta %.0f MB" % (antes, disco_mb(),
                                                                     disco_mb() - antes))
# Criterio 2: el delta tiene que ser el de las SALIDAS, no el de los videos. Medido en Windows,
# por clip: mp4 0,9-1,4 MB (se borra), det_ 17-45 KB, revision_ 18-29 KB, poses_ 0,78-2,2 MB.
import glob
print("mp4 que quedan en /content/trabajo/videos:",
      len(glob.glob("/content/trabajo/videos/*.mp4")), " (tiene que ser 0)")
"""

# %% CELDA 4 - p10 contra mediana, y clips/hora MEDIDO aqui
CELDA_4 = r"""
import json, numpy as np
import rutas as RU

reg = [json.loads(l) for l in open(RU.TRABAJO / "estado_lotes.jsonl", encoding="utf-8")
       if l.strip()]
por = {}
for d in reg:
    if "etapa" in d and d.get("n"):
        por.setdefault(d["etapa"], []).append(d["s"] / d["n"])
print("%-8s %8s %8s %8s %8s" % ("etapa", "n lotes", "p10", "p50", "razon"))
for e in sorted(por):
    v = np.array(por[e])
    p10, p50 = np.percentile(v, 10), np.percentile(v, 50)
    print("%-8d %8d %8.2f %8.2f %8.2f%s"
          % (e, len(v), p10, p50, p50 / max(p10, 1e-9),
             "   NO CONVERGEN" if p50 / max(p10, 1e-9) > 1.25 else ""))
print()
print("En la 3060 COMPARTIDA la razon de la fase 1 es 1,73x. En una GPU dedicada p10 y mediana")
print("tienen que acercarse. Si aqui tampoco convergen, la dispersion no es contencion y hay que")
print("mirar que es ANTES de tocar ningun knob.")
print()
print("Y el s/clip de arriba NO es el ritmo. El ritmo es reloj de pared entre clips completados,")
print("incluyendo descarga, fallos, subidas, carga de modelo y la desconexion que no planeaste.")
print("Se publican los dos y la diferencia, que es el coste operativo y es lo que decide si el")
print("corpus es una sesion o tres.")
"""

# %% CELDA 4b - devolver los resultados, que si no la corrida no sirve de nada
CELDA_4B = r"""
import lotes as LO
# SUBE lo completo al repo PRIVADO. `respaldo.py` comprueba que es privado antes de escribir (los
# clips ensenian atletas identificables) y no da un lote por subido si falta su manifiesto: un tar
# sin manifiesto no se puede descontar clip a clip al reanudar.
# El token sale del secreto de Colab `HUGGINFACE`, de HF_TOKEN o de un .env, y NO se imprime nunca.
LO.sube(LO.conjunto("anotados")[:5], [1, 4, 7])
"""

# %% CELDA 5 - que la etapa 10 escribio de verdad
CELDA_5 = r"""
import csv, glob
for p in sorted(glob.glob(str(RU.TRABAJO / "etiquetas_*.csv")))[:5]:
    with open(p, encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            print("%-34s %-16s %-18s %-8s %s"
                  % (f["clip"][:34], f["cabeza"], f["valor"], f["confianza"],
                     f["origen_etiqueta"]))
# Criterio 5: cada clip SIN anotacion humana trae tres filas con origen_etiqueta = modelo, y
# ningun clip CON anotacion humana la trae sobrescrita. Las dos cosas se comprueban aqui y no se
# suponen: es la regla que impide que una prediccion entre en un reservado como si fuera verdad.
"""

CELDAS = [CELDA_1, CELDA_2, CELDA_3, CELDA_4, CELDA_4B, CELDA_5]

if __name__ == "__main__":
    import sys
    if "--ipynb" in sys.argv:
        import json
        nb = {"cells": [{"cell_type": "code", "metadata": {}, "source": c.strip().splitlines(True),
                         "outputs": [], "execution_count": None} for c in CELDAS],
              "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
              "nbformat": 4, "nbformat_minor": 5}
        print(json.dumps(nb, indent=1))
    else:
        for i, c in enumerate(CELDAS, 1):
            print("# ==================== CELDA %d ====================" % i)
            print(c.strip())
            print()

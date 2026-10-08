"""La prueba de que la corrida se puede cortar por donde sea y no repite ni pierde nada.

## LAS CUATRO PREGUNTAS, Y POR QUE ESAS

  1  **Si se corta a la mitad y se relanza, reprocesa lo ya hecho?**  Tiene que ser NO. Si lo
     reprocesara, una corrida de dos dias con tres cortes tardaria cinco.
  2  **Si se BORRA el registro de auditoria, sigue reanudando?**  Tiene que ser SI. El registro es
     un diario, no la verdad. Un contador de este proyecto dijo una vez "completado = 3" habiendo
     6 ficheros, y desde entonces la autoridad es lo que EXISTE en disco.
  3  **Reescribe lo que ya estaba?**  Tiene que ser NO, y se comprueba por el `mtime`: si un
     fichero cambia de fecha en la segunda pasada, es que se volvio a calcular.
  4  **Salta algo en silencio?**  Al final, todo clip del trozo tiene que estar hecho o tener un
     motivo. Un clip que desaparece sin mensaje es el fallo mas caro de todos, porque no se nota.

## COMO SE PRUEBA SIN TOCAR LA CORRIDA DE VERDAD

Todo va contra un `FENCINGLY_TRABAJO` temporal y un punado de clips. La corrida larga puede estar
en marcha: no se comparten ficheros. Al acabar se borra el temporal.

    python verifica_reanuda.py [--n 4]
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
import glob, json, os, pathlib, sys, time
sys.path.insert(0, r"{raiz}/01_extraccion")
sys.path.insert(0, r"{raiz}/01_extraccion/codigo")
sys.path.insert(0, r"{raiz}/02_validacion/herramienta")
import lotes as LO, rutas as RU

fallos = []
def comprueba(n, ok, texto, detalle=""):
    print("  %-3s %-52s %s%s" % (n, texto, "OK" if ok else "FALLA",
                                 ("  " + detalle) if detalle else ""))
    if not ok:
        fallos.append(texto)

ids = {ids}
ET = [1, 4]

def firmas():
    """QUE DEVUELVE: {{fichero: (tamanio, mtime)}} de todo lo producido. El mtime es lo que delata
    un recalculo: un fichero con el mismo contenido pero otra fecha se volvio a escribir."""
    d = {{}}
    for pat in ("det_*.pkl", "revision_*.npz", "poses_*.npz"):
        for f in glob.glob(str(RU.TRABAJO / pat)):
            st = os.stat(f)
            d[os.path.basename(f)] = (st.st_size, st.st_mtime_ns)
    return d

print("PRIMERA PASADA: %d clips, etapas %s" % (len(ids), ET))
LO.corre(ids[:2], ET, lote=2, verboso=False)
f1 = firmas()
hechos1 = [c for c in ids if LO.hechas(c, ET) == set(ET)]
print("   hechos tras la primera pasada: %d de %d, %d ficheros"
      % (len(hechos1), len(ids), len(f1)))
comprueba("0", len(hechos1) >= 1, "la primera pasada produce algo")

print()
print("SEGUNDA PASADA sobre el conjunto ENTERO (simula el relanzamiento tras un corte)")
t0 = time.perf_counter()
LO.corre(ids, ET, lote=2, verboso=False)
t = time.perf_counter() - t0
f2 = firmas()
hechos2 = [c for c in ids if LO.hechas(c, ET) == set(ET)]

recalc = [k for k in f1 if k in f2 and f2[k][1] != f1[k][1]]
comprueba("1", not recalc, "no reprocesa lo que ya estaba hecho",
          "%d ficheros con mtime nuevo: %s" % (len(recalc), recalc[:3]))
comprueba("3", not recalc, "no reescribe ningun fichero anterior")
nuevos = [k for k in f2 if k not in f1]
print("   %d ficheros nuevos, %d intactos, %.0f s" % (len(nuevos), len(f1) - len(recalc), t))

print()
print("TERCERA PASADA tras BORRAR el registro de auditoria")
reg = RU.TRABAJO / "estado_lotes.jsonl"
n_reg = sum(1 for _ in open(reg, encoding="utf-8")) if reg.is_file() else 0
if reg.is_file():
    reg.unlink()
print("   se borran %d lineas de %s" % (n_reg, reg.name))
LO.corre(ids, ET, lote=2, verboso=False)
f3 = firmas()
recalc3 = [k for k in f2 if k in f3 and f3[k][1] != f2[k][1]]
comprueba("2", not recalc3, "sin el registro, sigue reanudando igual",
          "%d recalculados" % len(recalc3))

print()
print("AL FINAL: que no se salte nada en silencio")
hechos3 = [c for c in ids if LO.hechas(c, ET) == set(ET)]
sin = [c for c in ids if c not in hechos3]
print("   hechos %d de %d" % (len(hechos3), len(ids)))
for c in sin:
    print("      pendiente: %-40s video=%s" % (c[:40], bool(RU.video(c))))
comprueba("4", len(hechos3) == len(ids) or all(not RU.video(c) for c in sin),
          "todo clip con video acaba hecho, o dice por que no")

print()
print("Y LA UNION CON EL REMOTO (lo que hace que Colab reanude entre sesiones)")
falso = {{ids[0]: {{1, 4}}}}
h = LO.hechas("__clip_que_no_existe__", ET, {{"__clip_que_no_existe__": {{1, 4}}}})
comprueba("5", h == {{1, 4}}, "un clip solo en el remoto cuenta como hecho", str(sorted(h)))
h2 = LO.hechas("__clip_que_no_existe__", ET, None)
comprueba("6", h2 == set(), "sin remoto, ese mismo clip cuenta como pendiente", str(sorted(h2)))

print()
print("=" * 72)
if fallos:
    print("FALLA en %d:" % len(fallos))
    for f in fallos:
        print("   - %s" % f)
    sys.exit(1)
print("LA REANUDACION ES CORRECTA: no repite, no reescribe, no depende del registro,")
print("no salta nada en silencio, y en Colab cuenta lo que ya esta subido.")
sys.exit(0)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    sys.path.insert(0, str(RAIZ / "01_extraccion"))
    sys.path.insert(0, str(RAIZ / "01_extraccion" / "codigo"))
    import lotes as LO
    import rutas as RU

    ids = [c for c in LO.conjunto("disco") if RU.video(c)][:a.n]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="fz_reanuda_"))
    env = dict(os.environ)
    env["FENCINGLY_TRABAJO"] = str(tmp)
    env["PYTHONIOENCODING"] = "utf-8"
    print("PRUEBA DE REANUDACION")
    print("=" * 72)
    print("trabajo temporal: %s" % tmp)
    print("clips: %s" % ", ".join(ids))
    print()
    guion = GUION.format(raiz=str(RAIZ).replace("\\", "/"), ids=repr(ids))
    r = subprocess.run([RU.interprete(), "-u", "-c", guion], env=env,
                       text=True, encoding="utf-8", errors="replace")
    shutil.rmtree(tmp, ignore_errors=True)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())

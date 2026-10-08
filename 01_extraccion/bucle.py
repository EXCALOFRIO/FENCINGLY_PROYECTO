"""Un trabajador: da vueltas a su trozo del manifiesto hasta que no quede nada suyo por hacer.

Es lo que `noche.py` lanza N veces sobre trozos disjuntos. Vive aparte porque tiene que poder
lanzarse desatendido y sobrevivir a que se cierre la consola, y para eso hace falta un fichero.

QUE HACE DISTINTO A LLAMAR A `lotes.py` EN UN BUCLE DE `bash`:

  - **para solo** cuando una vuelta entera no completa ni un clip nuevo, en vez de dar vueltas
    para siempre gastando CPU;
  - **aguanta una excepcion** y sigue en la vuelta siguiente, porque un clip corrupto a las 3 de
    la manana no puede costar el resto de la noche;
  - **escribe una linea de ritmo por vuelta**, que es lo que luego permite decir clips/hora
    medidos y no extrapolados.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "codigo"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                       / "02_validacion" / "herramienta"))
import lotes as LO  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", default="disco")
    ap.add_argument("--etapas", default="1,4")
    ap.add_argument("--desde", type=int, default=0)
    ap.add_argument("--cuantos", type=int, default=None)
    ap.add_argument("--lote", type=int, default=20)
    ap.add_argument("--espera", type=int, default=0,
                    help="segundos entre vueltas. La etapa 7 espera, porque depende de que la")
    ap.add_argument("--vueltas", type=int, default=5000)
    ap.add_argument("--descarga", action="store_true")
    ap.add_argument("--borra-video", action="store_true", dest="borra")
    ap.add_argument("--vacias-max", type=int, default=1, dest="vacias",
                    help="cuantas vueltas SIN completar nada se aguantan antes de parar. "
                         "La etapa 7 necesita mas de 1: su entrada la va produciendo la GPU, "
                         "asi que una vuelta vacia significa 'todavia no', no 'ya esta'.")
    a = ap.parse_args()
    etapas = [int(x) for x in a.etapas.split(",") if x.strip()]
    ids = LO.conjunto(a.conjunto)
    mio = ids[a.desde:] if a.cuantos is None else ids[a.desde:a.desde + a.cuantos]
    print("trabajador: etapas %s, %d clips (del %d), lote %d, semilla %d"
          % (etapas, len(mio), a.desde, a.lote, LO.SEMILLA), flush=True)
    t0 = time.perf_counter()
    vacias = 0
    hechos0 = sum(1 for c in mio if LO.hechas(c, etapas) == set(etapas))
    print("ya hechos al empezar: %d de %d" % (hechos0, len(mio)), flush=True)
    for v in range(a.vueltas):
        antes = sum(1 for c in mio if LO.hechas(c, etapas) == set(etapas))
        if antes >= len(mio):
            print("TERMINADO: los %d clips del trozo estan hechos" % len(mio), flush=True)
            break
        try:
            LO.corre(mio, etapas, lote=a.lote, verboso=True,
                     descargar=a.descarga, borrar=a.borra)
        except KeyboardInterrupt:
            raise
        except Exception as ex:
            print("vuelta %d: EXCEPCION %s: %s -- se sigue en la siguiente"
                  % (v, type(ex).__name__, str(ex)[:140]), flush=True)
        ahora = sum(1 for c in mio if LO.hechas(c, etapas) == set(etapas))
        el = time.perf_counter() - t0
        nuevos = ahora - hechos0
        print("VUELTA %d  hechos %d/%d (+%d en esta vuelta)  %.0f s  ritmo %.1f s/clip  "
              "%.1f clips/h"
              % (v, ahora, len(mio), ahora - antes, el,
                 el / max(nuevos, 1), 3600.0 * nuevos / max(el, 1.0)), flush=True)
        if ahora == antes:
            vacias += 1
            if vacias >= a.vacias:
                print("%d vuelta(s) seguidas sin completar nada: se para para no gastar CPU."
                      % vacias, flush=True)
                break
            print("vuelta vacia %d de %d permitidas: se espera y se reintenta"
                  % (vacias, a.vacias), flush=True)
        else:
            vacias = 0
        if a.espera:
            time.sleep(a.espera)
    return 0


if __name__ == "__main__":
    sys.exit(main())

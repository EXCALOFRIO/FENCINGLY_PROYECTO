"""Cuantos trabajadores conviene lanzar: medido, no supuesto.

## POR QUE HACE FALTA MEDIRLO

Con UN proceso `nvidia-smi` marca la GPU al **63 %**, y de ahi se deduce facilmente -- y mal --
que sobra tarjeta y que basta con lanzar mas procesos. Ese porcentaje dice "habia un nucleo
activo", no "la tarjeta esta al 63 % de su capacidad". El perfilado por trozos dice otra cosa:
con cuatro procesos a la vez, el **forward es el 80,9 %** del tiempo y pasa de ~37 ms por recorte
a **139 ms**, o sea que los procesos se estan esperando unos a otros en la GPU.

Asi que la unica cifra que decide es el **caudal agregado**: clips completos por hora con N
trabajadores. Esta herramienta lo mide lanzando N y contando ficheros nuevos en disco durante una
ventana, que es la misma autoridad que usa la reanudacion.

## EL CRITERIO, DECLARADO ANTES

Se adopta el N mas pequeno que este a menos del 5 % del mejor caudal medido. Mas procesos de los
necesarios no son gratis: cada uno ocupa VRAM y RAM, y si la maquina empieza a paginar la noche
entera se va al traste.

    python mide_paralelo.py --enes 1,2,3 --minutos 6
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "01_extraccion"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "01_extraccion" / "codigo"))
import lotes as LO  # noqa: E402
import noche as NO  # noqa: E402
import rutas as RU  # noqa: E402


def cuenta(ids, etapas):
    return sum(1 for c in ids if LO.hechas(c, etapas) == set(etapas))


def una(n, ids, minutos, lote):
    """QUE HACE: lanza n trabajadores sobre trozos disjuntos, mide `minutos` y los mata."""
    trozo = (len(ids) + n - 1) // n
    procs = []
    banderas = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if sys.platform == "win32" \
        else 0
    logs = RU.TRABAJO / "logs_medida"
    logs.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cmd = [sys.executable, "-u",
               str(pathlib.Path(__file__).resolve().parents[2] / "01_extraccion" / "bucle.py"),
               "--conjunto", "disco", "--etapas", "1,4",
               "--desde", str(i * trozo), "--cuantos", str(trozo), "--lote", str(lote)]
        fh = open(logs / ("n%d_w%d.log" % (n, i)), "wb")
        procs.append((subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                       creationflags=banderas), fh))
    time.sleep(45)
    """45 s de calentamiento antes de empezar a contar: cargar YOLO y Sapiens cuesta unos
    segundos por proceso y meterlo en la ventana penalizaria a los N grandes por una razon que no
    es la que se quiere medir -- en una corrida de toda la noche ese arranque se paga una vez."""
    a = cuenta(ids, [1, 4])
    t0 = time.perf_counter()
    time.sleep(minutos * 60)
    hechos = cuenta(ids, [1, 4]) - a
    el = time.perf_counter() - t0
    for p, fh in procs:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
        else:
            p.kill()
        fh.close()
    time.sleep(8)
    return hechos, el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enes", default="1,2,3")
    ap.add_argument("--minutos", type=float, default=6.0)
    ap.add_argument("--lote", type=int, default=10)
    a = ap.parse_args()
    ids = LO.conjunto("disco")
    print("CUANTOS TRABAJADORES: caudal agregado medido, %g min por punto" % a.minutos)
    print("=" * 66)
    print("%-4s %8s %8s %12s" % ("N", "clips", "s", "clips/h"))
    res = []
    for n in [int(x) for x in a.enes.split(",")]:
        h, el = una(n, ids, a.minutos, a.lote)
        ch = 3600.0 * h / el
        res.append((n, ch, h))
        print("%-4d %8d %8.0f %12.1f" % (n, h, el, ch))
    if res:
        mejor = max(r[1] for r in res)
        eleg = min(n for n, ch, _h in res if ch >= 0.95 * mejor)
        print()
        print("mejor caudal %.1f clips/h; el N mas pequeno a menos del 5 %% es **%d**"
              % (mejor, eleg))
        n_tot = len(ids)
        for n, ch, _h in res:
            print("   N=%d -> los %d clips en %.1f h" % (n, n_tot, n_tot / max(ch, 1e-9)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

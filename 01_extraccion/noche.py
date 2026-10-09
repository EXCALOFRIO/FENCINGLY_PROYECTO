"""Deja la extraccion corriendo toda la noche, en paralelo, y que sobreviva a que cierres todo.

## CUANTOS TRABAJADORES: MEDIDO, Y EL RESULTADO FUE QUE CASI DA IGUAL

La hipotesis era que el cuello estaba en la CPU: con un proceso `nvidia-smi` marcaba la GPU al
**63 %** y 1,7 GB de 12, y ademas **lotear el modelo de pose solo dio 1,17x**, que es lo que se
espera cuando la tarjeta no es el limite. Parecia claro que varios procesos escalarian.

**No escalan.** Caudal agregado medido con `mide_paralelo.py`, seis minutos por punto:

    N = 1    99,9 clips/h          N = 2    99,9 clips/h          N = 3   109,8 clips/h

O sea **1,10x de 1 a 3**, y con 10-11 clips por punto eso esta dentro del ruido de la propia
medida: no se puede afirmar que 3 sea mejor que 1. Con un solo proceso la GPU ya llega al 99 %.
El 63 % de antes era enganioso -- ese porcentaje dice "habia un nucleo activo", no "la tarjeta
esta al 63 % de su capacidad".

Se lanzan **3 de todas formas** porque no cuesta nada (3 GB de VRAM de 12, y la RAM ya no es
problema, ver abajo) y porque reparten el trozo, asi que si uno muere los otros siguen. Pero **no
se espere que la noche rinda el triple**: rinde ~100 clips/h.

Lo que NO acelera, todo medido: subir el lote del modelo (1,17x y **rechazado** por mover un
keypoint usable 130 px), cachear cuadros duplicados (**0 duplicados exactos** en 7.434 recortes:
el telecine no sobrevive al re-encoding), la atencion SDPA (**ya estaba puesta**) y saltarse las
poses que se tiran (solo el **1,9 %**, y no se distinguen por la caja antes de calcularlas).

## LA MEMORIA ERA UN LIMITE, Y SE QUITO IGUALMENTE

Cada proceso de la fase 2 ocupaba **5,1 GB** porque `poses_del_clip` acumulaba TODOS los
fotogramas del clip -- ~2 GB en un 1920x1080 de 325 cuadros -- y el unico consumidor era el mp4 de
revision, que en esta ruta esta apagado. Con `guarda_frames=False` eso desaparece. No hace ganar
velocidad, pero es lo que permite tener tres procesos sin que la maquina empiece a paginar, que si
la habria hecho perder.

## COMO SE REPARTE

El conjunto va en el orden barajado con `lotes.SEMILLA`, y cada trabajador se queda un trozo
contiguo de ese orden. Consecuencia buena: **el prefijo de cada trozo sigue siendo una muestra
aleatoria**, asi que si la noche se corta a la mitad lo extraido vale para medir igual.

La etapa 7 va aparte, en sus propios procesos, porque es **CPU pura** y no tiene por que esperar a
la GPU. Se queda con los clips a los que las etapas 1 y 4 ya les dieron salida.

    python noche.py --gpu 3 --cpu 2            lanza y devuelve el control
    python noche.py --estado                   cuanto lleva hecho
    python noche.py --para                     mata los trabajadores
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "codigo"))
import lotes as LO  # noqa: E402
import rutas as RU  # noqa: E402

LOGS = RU.TRABAJO / "logs_noche"
MARCA = LOGS / "pids.txt"


def _pids():
    """QUE DEVUELVE: [(nombre, pid)] de la marca, o lista vacia."""
    if not MARCA.is_file():
        return []
    out = []
    for ln in MARCA.read_text(encoding="utf-8").splitlines():
        t = ln.split()
        if len(t) >= 2:
            out.append((t[0], int(t[1])))
    return out


def _vive(pid):
    """QUE HACE: pregunta si un proceso sigue vivo. SIN MATARLO.

    **`os.kill(pid, 0)` NO sirve en Windows.** En POSIX la senial 0 es la forma canonica de
    preguntar; en Windows `os.kill` solo entiende `CTRL_C_EVENT` y `CTRL_BREAK_EVENT`, y con
    cualquier otro valor llama a `TerminateProcess`. O sea que la "comprobacion" **mata el
    proceso**. Me paso: lance cinco trabajadores, ejecute `--estado` para ver si vivian, y los
    mate a los cinco -- y el informe dijo, con toda la razon, "trabajadores vivos: 0".

    La forma correcta es abrir un asa con `OpenProcess` y mirar el codigo de salida.
    """
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    cod = ctypes.c_ulong()
    ok = k.GetExitCodeProcess(h, ctypes.byref(cod))
    k.CloseHandle(h)
    return bool(ok) and cod.value == 259     # STILL_ACTIVE


def prevuelo(etapas=(1, 4, 7)):
    """QUE COMPRUEBA, ANTES DE LANZAR NADA: que el interprete importa lo que hace falta.

    POR QUE EXISTE. La noche del 2026-09-22 se lanzaron cinco trabajadores contra un interprete
    que se habia quedado sin la mitad de sus paquetes, y **el sistema no se entero hasta 18
    minutos despues**, cuando un trabajador dijo "una vuelta sin completar nada". Seis horas de
    GPU perdidas por no preguntar al arrancar.

    QUE DEVUELVE: (vale, informe). `vale` es False si falta algo que las etapas pedidas necesitan.
    """
    py = RU.interprete()
    d = RU.comprueba_interprete(py)
    necesita = set()
    if 1 in etapas:
        necesita |= {"ultralytics", "cv2", "torch"}
    if 4 in etapas:
        necesita |= {"transformers", "torch", "cv2"}
    if 7 in etapas:
        necesita |= {"numpy"}
    falta = sorted(m for m in necesita if not d.get(m))
    lineas = ["interprete: %s" % py]
    if "__error__" in d:
        lineas.append("  NO RESPONDE: %s" % d["__error__"])
        return False, "\n".join(lineas)
    for m in sorted(d):
        if m != "__error__":
            lineas.append("  %-14s %s" % (m, "si" if d[m] else "NO"))
    if not d.get("cuda"):
        lineas.append("  AVISO: torch.cuda.is_available() es False. Las etapas 1 y 4 iran por CPU")
        lineas.append("         y eso es ~20 veces mas lento; probablemente no es lo que quieres.")
    if falta:
        lineas.append("  FALTA para las etapas %s: %s" % (list(etapas), ", ".join(falta)))
    return (not falta), "\n".join(lineas)


def lanza(n_gpu, n_cpu, conjunto="disco", lote=20, etapas_gpu="1,4", descarga=False,
          exigir=True, respaldo=120, respaldo_de=None):
    """QUE HACE: lanza los trabajadores DESATENDIDOS y devuelve el control.

    `CREATE_NEW_PROCESS_GROUP` + `CREATE_NO_WINDOW` para que sigan vivos, sin ventana, cuando se cierre la
    consola que los lanzo. Si murieran con la consola, "dejarlo toda la noche" seria una promesa
    que el codigo no cumple.
    """
    et_gpu = [int(x) for x in etapas_gpu.split(",")]
    vale, inf = prevuelo(tuple(et_gpu) + (7, 9, 11))
    print(inf)
    print()
    if not vale:
        if exigir:
            print("NO SE LANZA NADA. Arregla el interprete antes -- lanzar contra uno roto es")
            print("exactamente lo que hizo perder la noche del 22 al 23 de septiembre.")
            return []
        print("se lanza igualmente por --sin-exigir, bajo tu responsabilidad")
    ids = LO.conjunto(conjunto)
    LOGS.mkdir(parents=True, exist_ok=True)
    trozo = (len(ids) + n_gpu - 1) // n_gpu
    pids = []
    banderas = 0
    if os.name == "nt":
        banderas = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        """SIN VENTANA, y no `DETACHED_PROCESS` (2026-10-08, lo pidio el usuario). Un proceso
        DESACOPLADO no tiene consola, asi que cada programa de consola que lanza (el `python.exe`
        del venv es un lanzador que arranca otro) le abre a Windows una ventana NUEVA en pantalla.
        `CREATE_NO_WINDOW` le da una consola oculta que heredan los hijos: ninguna ventana, y siguen
        vivos al cerrar la terminal que los lanzo. Todo lo que dicen va a `logs_noche/*.log`."""
    for i in range(n_gpu):
        desde = i * trozo
        log = LOGS / ("gpu%d.log" % i)
        cmd = [sys.executable, "-u", str(pathlib.Path(__file__).parent / "bucle.py"),
               "--conjunto", conjunto, "--etapas", etapas_gpu,
               "--desde", str(desde), "--cuantos", str(trozo), "--lote", str(lote),
               "--espera", "60", "--vacias-max", "3"]
        if descarga:
            cmd += ["--descarga", "--borra-video"]
        with open(log, "ab") as fh:
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                 creationflags=banderas, close_fds=True)
        pids.append(("gpu%d" % i, p.pid, desde, trozo))
        print("gpu%d  pid %-7d clips %d..%d  -> %s" % (i, p.pid, desde, desde + trozo - 1, log))
    for i in range(n_cpu):
        desde = i * ((len(ids) + n_cpu - 1) // n_cpu)
        cuantos = (len(ids) + n_cpu - 1) // n_cpu
        log = LOGS / ("cpu%d.log" % i)
        cmd = [sys.executable, "-u", str(pathlib.Path(__file__).parent / "bucle.py"),
               "--conjunto", conjunto, "--etapas", "7,9,11",
               "--desde", str(desde), "--cuantos", str(cuantos), "--lote", "40",
               "--espera", "180", "--vacias-max", "400"]
        """LA ETAPA 7 AGUANTA VUELTAS VACIAS, la 1 y la 4 no. Su entrada la van produciendo los
        trabajadores de GPU durante toda la noche, asi que una vuelta sin nada que hacer significa
        "todavia no" y no "ya esta". Con el valor por defecto (1) los dos trabajadores de CPU se
        pararon a los pocos minutos de lanzarlos y la etapa 7 se quedo atras."""
        with open(log, "ab") as fh:
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                 creationflags=banderas, close_fds=True)
        pids.append(("cpu%d" % i, p.pid, desde, cuantos))
        print("cpu%d  pid %-7d clips %d..%d  -> %s" % (i, p.pid, desde, desde + cuantos - 1, log))
    if respaldo:
        """EL RESPALDO VA CON LA NOCHE, no aparte (2026-10-08). Hasta hoy `lotes.sube` solo lo
        llamaba Colab y el remoto tenia 0 clips con ~5.000 extraidos. Ver `respalda.py`."""
        log = LOGS / "resp.log"
        cmd = [sys.executable, "-u", str(pathlib.Path(__file__).parent / "respalda.py"),
               "--conjunto", respaldo_de or conjunto, "--cada", str(respaldo)]
        with open(log, "ab") as fh:
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                 creationflags=banderas, close_fds=True)
        pids.append(("resp", p.pid, 0, 0))
        print("resp  pid %-7d respaldo a HF cada %s min -> %s" % (p.pid, respaldo, log))
    MARCA.write_text("\n".join("%s %d %d %d" % t for t in pids), encoding="utf-8")
    print()
    print("lanzados %d trabajadores. Se puede cerrar todo: siguen." % len(pids))
    print("estado:  python 01_extraccion/noche.py --estado")
    print("parar :  python 01_extraccion/noche.py --para")
    return pids


def estado(conjunto="disco"):
    ids = LO.conjunto(conjunto)
    TODAS = (1, 4, 7, 9, 11)
    h = [LO.hechas(c, list(TODAS)) for c in ids]
    cuenta = [("etapa %d" % n, sum(1 for x in h if n in x)) for n in TODAS]
    n147 = sum(1 for x in h if x == set(TODAS))
    print("conjunto %s: %d clips" % (conjunto, len(ids)))
    for nom, n in cuenta + [("todas", n147)]:
        print("   %-10s %6d  (%.1f %%)" % (nom, n, 100.0 * n / max(len(ids), 1)))
    if MARCA.is_file():
        vivos = [n for n, p in _pids() if _vive(p)]
        print("   trabajadores vivos: %d  %s" % (len(vivos), " ".join(vivos)))
    print()
    print("El recuento sale de MIRAR EL DISCO, no de ningun contador: si se mata todo y se")
    print("relanza, se reanuda exactamente aqui.")
    return n147


def para():
    """QUE HACE: mata los trabajadores Y sus hijos. Aqui `os.kill` si es lo que se quiere.

    Mata tambien a los hijos porque cada trabajador lanza subprocesos con el otro interprete
    (`lotes._lanza`): matar solo al padre dejaria un proceso de Sapiens ocupando 1,5 GB de VRAM
    sin nadie que lo espere."""
    ps = _pids()
    if not ps:
        print("no hay marca de trabajadores")
        return
    for nom, pid in ps:
        if not _vive(pid):
            print("%-6s pid %-7d ya no estaba" % (nom, pid))
            continue
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True)
        else:
            os.kill(pid, 9)
        print("%-6s pid %-7d parado" % (nom, pid))
    MARCA.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3,
                    help="procesos para las etapas 1 y 4. 3 con 32 GB; con 16 GB, 2.")
    ap.add_argument("--cpu", type=int, default=2, help="procesos para la etapa 7, que es CPU pura")
    ap.add_argument("--conjunto", default="disco")
    ap.add_argument("--lote", type=int, default=20)
    ap.add_argument("--etapas-gpu", default="1,4", dest="etgpu")
    ap.add_argument("--descarga", action="store_true",
                    help="baja del CDN los clips que no esten en disco, a RU.DESCARGAS, donde SE "
                         "QUEDAN (`--borra-video` solo borra TRABAJO/videos, la ruta de Colab)")
    ap.add_argument("--sin-exigir", action="store_false", dest="exigir",
                    help="lanza aunque el prevuelo falle")
    ap.add_argument("--respaldo", type=float, default=120,
                    help="minutos entre subidas al HF privado; 0 lo apaga")
    ap.add_argument("--respaldo-de", default=None, dest="respaldo_de",
                    help="conjuntos que respalda, separados por coma (por defecto, --conjunto)")
    ap.add_argument("--prevuelo", action="store_true", dest="solo_prevuelo")
    ap.add_argument("--estado", action="store_true")
    ap.add_argument("--para", action="store_true")
    a = ap.parse_args()
    if a.solo_prevuelo:
        vale, inf = prevuelo()
        print(inf)
        print()
        print("prevuelo: %s" % ("OK" if vale else "FALLA"))
        return 0 if vale else 1
    if a.estado:
        estado(a.conjunto)
    elif a.para:
        para()
    else:
        lanza(a.gpu, a.cpu, a.conjunto, a.lote, etapas_gpu=a.etgpu,
              descarga=a.descarga, exigir=a.exigir, respaldo=a.respaldo,
              respaldo_de=a.respaldo_de)
    return 0


if __name__ == "__main__":
    sys.exit(main())

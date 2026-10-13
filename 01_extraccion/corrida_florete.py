"""EL DIRECTOR de la corrida de florete (`04_documentacion/informes/CORRIDA_FINAL_2026-10-13.md`).

Encadena las fases en el orden que decidio el usuario el 2026-10-13 y **nunca pone dos trabajos de
GPU a la vez** (`gpu-one-job-at-a-time`: una sonda y un extractor juntos en la 3060 hicieron la sonda
~100 veces mas lenta). Cada fase es reanudable por su cuenta, porque el estado es el disco. El
director tambien lo es: una fase acabada deja `HECHA_<fase>.txt` en `MARCAS`, y al relanzar se la
salta. Si una fase falla, escribe `FALLA_<fase>.txt` y **se para**: no sigue a ciegas con la
siguiente.

    python corrida_florete.py --seco      que haria y cuanto hay hecho, SIN lanzar nada
    python corrida_florete.py --lanza     arranca el director oculto y devuelve el control
    python corrida_florete.py --estado    lo mismo que --seco, mas corto
    python corrida_florete.py --para      para el director, sus trabajadores y lo que este corriendo

LAS FASES, en orden:

  luces        etapa 9 sobre todo el florete, bajando los mp4 que falten (CPU y red)
  lista        `foil_dos_luces.txt`: dos luces de color y `touch` izq/der (`lista_dos_luces.py`)
  dos_luces    etapas 1 y 4 (GPU) y 7, 9 y 11 (CPU) sobre la lista
  vjepa_g2     ViT-g 256 `rec4,mic4` sobre la lista (la receta decidida)
  vjepa_v0     ViT-L 256 `rec4` (= el `tok` de V0) sobre la lista, para comparar en la sonda
  sonda_v0     `prioridad` con V0 sobre la lista (protocolo declarado en `VJEPA2_2026-10-09.md`)
  sonda_g2     `prioridad` con G2 sobre la lista, y G2 contra V0 por pares
  resto        etapas 1, 4, 7, 9 y 11 sobre el resto del florete (salta lo hecho)
  zona_fusion  la zona final por torneo, con el florete entero

Lo que NO hace: la fase 3B (V-JEPA sobre el resto, G2 o V0). La decide el usuario con la
comparacion de `sonda_g2`.

**NO SE LANZA hasta que el usuario diga que se acabaron las pruebas.**
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

AQUI = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI / "codigo"))
import lotes as LO  # noqa: E402
import noche as NO  # noqa: E402
import rutas as RU  # noqa: E402

RAIZ = AQUI.parent
PY = sys.executable
FL = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
"""Sin ventanas, nunca (`launch-without-windows`); y vivo aunque se cierre la consola que lo lanzo."""

CAMP = RAIZ / "03_datos" / "catalogo" / "champs_florete_por_nombres.txt"
CONJUNTO = RAIZ / "03_datos" / "catalogo" / "foil_y_campeonatos.txt"
"""Los 106.432 de `foil` MAS los de florete de los campeonatos 2025-26 (`lista_campeonatos_florete.py`,
3.367 por los nombres de los dos tiradores; lo decidio el usuario el 13/10). Lo escribe
`prepara_conjunto` la primera vez."""
LISTA = RAIZ / "03_datos" / "catalogo" / "foil_dos_luces.txt"
VJ = pathlib.Path(os.environ.get("FENCINGLY_VJEPA_CORRIDA", "D:/FENCINGLY_VJEPA"))
VJ_G2, VJ_V0 = VJ / "g2", VJ / "v0"
MEMMAP = pathlib.Path(os.environ.get("FENCINGLY_MEMMAP", str(pathlib.Path.home() / "FENCINGLY_TMP" / "sonda_memmap")))
CACHE_ETQ = RU.SCRATCH / "sonda_cache_etq_corrida"
SAL = RAIZ / "02_validacion" / "auditoria" / "corrida_florete"
MARCAS = RU.TRABAJO / "corrida_florete"
N_CPU = int(os.environ.get("FENCINGLY_CPU_TRABAJADORES", "2"))
"""Trabajadores de CPU (etapas 7, 9, 11) en las fases con GPU. 2 como en los anotados; O4 de §10 mide
si mas trabajadores alcanzan mejor a la GPU."""
PAUSA = int(os.environ.get("FENCINGLY_DIRECTOR_PAUSA", "300"))
"""Segundos entre miradas del director a sus trabajadores (300 en la corrida; los ensayos la bajan)."""
RESPALDO_MIN = 120
"""El respaldo a HF privado va con cada fase de `noche.py`, cada 2 h, como en los anotados."""

FASES = ("luces", "lista", "dos_luces", "vjepa_g2", "vjepa_v0", "sonda_v0", "sonda_g2", "resto",
         "zona_fusion")
GPU = {"dos_luces", "vjepa_g2", "vjepa_v0", "sonda_v0", "sonda_g2", "resto"}
DISCO_MIN_GB = {"D:": 400, "C:": 200}
"""Medido el 13/10: faltan ~180 GB de mp4 en D: y 120-610 GB de V-JEPA; las poses (~150 GB) y los
memmap de las sondas (hasta ~121 GB, de uno en uno) van en C:."""
GPU_SCRIPTS = ("sonda_vjepa.py", "vjepa_variantes.py", "vjepa.py", "dino_cuadros.py", "bucle.py",
               "pipeline.py")


def log(msg):
    print("%s  %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)


def marca(nombre, texto):
    MARCAS.mkdir(parents=True, exist_ok=True)
    (MARCAS / nombre).write_text("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M"), texto), encoding="utf-8")


def hecha(fase):
    return (MARCAS / ("HECHA_%s.txt" % fase)).is_file()


def lineas_python():
    """Las lineas de orden de los python vivos. `wmic` ya no existe en este Windows: se pregunta a CIM."""
    if os.name != "nt":
        return []
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                        "ForEach-Object { $_.CommandLine }"],
                       capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    return [l for l in r.stdout.splitlines() if l.strip()]


def otros_gpu():
    return [l for l in lineas_python() if any(s in l for s in GPU_SCRIPTS)]


def espera_gpu_libre():
    """Antes de cada fase de GPU: que no corra NADA que use la GPU (una sonda, un extractor, una noche).
    Las fases del director acaban antes de empezar la siguiente, asi que lo que se vea aqui es ajeno."""
    avisado = False
    while True:
        o = otros_gpu()
        if not o:
            return
        if not avisado:
            log("GPU ocupada, se espera: %s" % " | ".join(x[:120] for x in o[:3]))
            avisado = True
        time.sleep(PAUSA)


def libre_gb(unidad):
    return shutil.disk_usage(unidad + "\\").free / 1e9


def comprueba_disco():
    falta = ["%s %.0f GB libres (pide %d)" % (u, libre_gb(u), m) for u, m in DISCO_MIN_GB.items()
             if libre_gb(u) < m]
    if falta:
        raise RuntimeError("poco disco: " + "; ".join(falta))


def ids(conjunto):
    return LO.conjunto(str(conjunto))


def prepara_conjunto(cfg):
    """`foil` + los campeonatos de florete, un id por linea. Si ya existe no se toca (el orden de la
    corrida sale de la baraja de `lotes.conjunto`, no del fichero)."""
    if cfg["prueba"] or pathlib.Path(cfg["conjunto"]).is_file():
        return
    camp = [l.strip() for l in open(CAMP, encoding="utf-8") if l.strip()] if CAMP.is_file() else []
    todos = sorted(set(LO.conjunto("foil")) | set(camp))
    pathlib.Path(cfg["conjunto"]).write_text("\n".join(todos) + "\n", encoding="utf-8")
    log("conjunto: %d de foil + %d de campeonatos = %d -> %s" % (len(todos) - len(camp), len(camp),
                                                                 len(todos), cfg["conjunto"]))


def cuenta(clips, etapas):
    e = set(etapas)
    return sum(1 for c in clips if LO.hechas(c, sorted(e)) == e)


def espera_noche(pids, clips):
    """Espera a los trabajadores de GPU de `noche.lanza`; luego a que la CPU (7, 9, 11) alcance a lo
    que la GPU dejo hecho, y para todo (los de CPU aguantan 400 vueltas vacias por diseno: sin esto
    se quedarian ~20 h dando vueltas)."""
    gpu = [p for n, p, *_ in pids if n.startswith("gpu")]
    cpu = [p for n, p, *_ in pids if n.startswith("cpu")]
    while any(NO._vive(p) for p in gpu):
        time.sleep(PAUSA)
    log("trabajadores de GPU terminados")
    antes, quietas = None, 0
    while cpu and any(NO._vive(p) for p in cpu):
        pend = sum(1 for c in clips if LO.hechas(c, [1, 4]) == {1, 4}
                   and LO.hechas(c, [7, 9, 11]) != {7, 9, 11})
        if pend == 0:
            break
        quietas = quietas + 1 if pend == antes else 0
        if quietas >= 6:
            log("la CPU lleva %d s sin avanzar con %d pendientes: se para (se reintentan al relanzar)"
                % (12 * PAUSA, pend))
            break
        antes = pend
        time.sleep(2 * PAUSA)
    NO.para()


def lanza_noche(conjunto, n_gpu, n_cpu, etapas_gpu, respaldo):
    pids = NO.lanza(n_gpu, n_cpu, str(conjunto), etapas_gpu=etapas_gpu, descarga=True,
                    respaldo=respaldo, lote=20)
    if not pids:
        raise RuntimeError("noche.lanza no lanzo nada (prevuelo)")
    return pids


def corre(cmd, salida, cwd):
    salida.parent.mkdir(parents=True, exist_ok=True)
    log("corre: %s  -> %s" % (" ".join(str(c) for c in cmd[1:]), salida))
    with open(salida, "ab") as fh:
        r = subprocess.run([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT, cwd=str(cwd),
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    return r.returncode


def acabo_extractor(log_):
    """Lo que hizo la ULTIMA pasada del extractor (el registro se anade): `hechos N en` o, si ya no
    quedaba nada, `M ya hechos, 0 pendientes` (0). None si la ultima pasada no acabo."""
    t = log_.read_text(encoding="utf-8", errors="replace") if log_.is_file() else ""
    m = [(x.start(), int(x.group(1))) for x in re.finditer(r"^hechos (\d+) en \d+ s", t, re.M)]
    m += [(x.start(), 0) for x in re.finditer(r"^\d+ clips, \d+ ya hechos, 0 pendientes", t, re.M)]
    ini = [x.start() for x in re.finditer(r"^\d+ clips, \d+ ya hechos, \d+ pendientes", t, re.M)]
    if not m or (ini and max(ini) > max(m)[0] and not re.match(r"^\d+ clips, \d+ ya hechos, 0 pendientes",
                                                                t[max(ini):])):
        return None
    return max(m)[1]


# ---------------------------------------------------------------------------------------------
# LAS FASES. Cada una devuelve una frase con sus cifras (va a HECHA_<fase>.txt) o lanza excepcion.
# ---------------------------------------------------------------------------------------------

def f_luces(cfg):
    clips = ids(cfg["conjunto"])
    espera_noche(lanza_noche(cfg["conjunto"], 3, 0, "9", cfg["respaldo"]), [])
    n = cuenta(clips, [9])
    return "lights_ en %d de %d clips" % (n, len(clips))


def f_lista(cfg):
    import lista_dos_luces as LD
    import catalogo as CA
    df = CA.manifiesto().set_index("clip_id")
    clips = ids(cfg["conjunto"])
    dos = [c for c in clips if LD.lados_de_color(c) == {"L", "R"} and c in df.index
           and df.loc[c, "touch"] in ("left", "right")]
    if not dos:
        raise RuntimeError("lista vacia")
    cfg["lista"].write_text("\n".join(dos) + "\n", encoding="utf-8")
    return "%d clips con las dos luces y touch izq/der, de %d -> %s" % (len(dos), len(clips), cfg["lista"])


def f_dos_luces(cfg):
    clips = ids(cfg["lista"])
    if cfg["prueba"]:
        assert cuenta(clips, [1, 4]) == len(clips), "en --prueba solo clips con las etapas 1 y 4 hechas"
    else:
        espera_gpu_libre()
    espera_noche(lanza_noche(cfg["lista"], 3, N_CPU, "1,4", cfg["respaldo"]), clips)
    return "etapas 1+4 en %d, 7+9+11 en %d, de %d" % (cuenta(clips, [1, 4]), cuenta(clips, [7, 9, 11]),
                                                       len(clips))


def _extrae(cfg, modelo, salidas, destino, nombre):
    espera_gpu_libre()
    lg = SAL / ("extrae_%s.txt" % nombre)
    cod = corre([PY, "-u", "vjepa_variantes.py", "--todos", "--conjunto", cfg["lista"], "--modelo", modelo,
                 "--salidas", salidas, "--destino", destino], lg, AQUI / "codigo")
    n = acabo_extractor(lg)
    if cod != 0 or n is None:
        raise RuntimeError("el extractor salio con codigo %s (mira %s)" % (cod, lg))
    return "%s: hechos %d en esta pasada; ficheros %d en %s" % (
        nombre, n, sum(1 for _ in pathlib.Path(destino).glob("vjv_*.npz")), destino)


def f_vjepa_g2(cfg):
    return _extrae(cfg, "vjepa2-vitg-fpc64-256", "rec4,mic4", VJ_G2, "g2")


def f_vjepa_v0(cfg):
    return _extrae(cfg, "vjepa2-vitl-fpc64-256", "rec4", VJ_V0, "v0")


def _sonda(cfg, d, variante, nombre, gb):
    espera_gpu_libre()
    if libre_gb(MEMMAP.drive or "C:") < gb + 30:
        raise RuntimeError("no caben los %d GB del memmap en %s" % (gb, MEMMAP))
    pred = SAL / ("pred_%s_prioridad_corrida.npz" % nombre)
    if not pred.is_file():
        corre([PY, "-u", "sonda_vjepa.py", "--tarea", "prioridad", "--semillas", "5", "--dir", d,
               "--variante", variante, "--clips-de", cfg["lista"], "--memmap", MEMMAP,
               "--cache-etq", CACHE_ETQ, "--guarda-pred", pred],
              SAL / ("sonda_%s_prioridad_corrida.txt" % nombre), RAIZ / "02_validacion" / "herramienta")
    for f in MEMMAP.glob("%s_%s_*" % (pathlib.Path(d).name, variante)):
        f.unlink()
    """El memmap es de ESTA sonda (`carga_variante` le pone este prefijo): se borra al acabar."""
    if not pred.is_file():
        raise RuntimeError("la sonda %s no dejo prediccion" % nombre)
    return pred


def f_sonda_v0(cfg):
    p = _sonda(cfg, VJ_V0, "rec4", "V0", 50)
    return "prediccion en %s" % p


def f_sonda_g2(cfg):
    p = _sonda(cfg, VJ_G2, "rec4+mic4", "G2", 130)
    ref = SAL / "pred_V0_prioridad_corrida.npz"
    sal = SAL / "compara_prioridad_G2_vs_V0_corrida.txt"
    corre([PY, "-u", "compara_variantes.py", ref, p], sal, RAIZ / "02_validacion" / "herramienta")
    ult = [l for l in sal.read_text(encoding="utf-8", errors="replace").splitlines() if l.startswith("CRITERIO")]
    return "G2 contra V0: %s" % (ult[-1] if ult else "sin linea de criterio (mira %s)" % sal)


def f_resto(cfg):
    clips = ids(cfg["conjunto"])
    espera_gpu_libre()
    espera_noche(lanza_noche(cfg["conjunto"], 3, N_CPU, "1,4", cfg["respaldo"]), clips)
    return "etapas 1+4 en %d, 7+9+11 en %d, de %d" % (cuenta(clips, [1, 4]), cuenta(clips, [7, 9, 11]),
                                                       len(clips))


def f_zona_fusion(cfg):
    cod = corre([PY, "-u", "zona_fusion.py", "--conjunto", cfg["conjunto"], "--escribe"],
                SAL / "zona_fusion.txt", AQUI)
    if cod != 0:
        raise RuntimeError("zona_fusion salio con codigo %s" % cod)
    return "zona final escrita (mira %s)" % (SAL / "zona_fusion.txt")


FUNC = {f: globals()["f_" + f] for f in FASES}


def director(cfg):
    log("DIRECTOR de la corrida: conjunto %s, lista %s, fases %s%s"
        % (cfg["conjunto"], cfg["lista"], ",".join(cfg["fases"]), "  [PRUEBA]" if cfg["prueba"] else ""))
    vale, inf = NO.prevuelo((1, 4, 7, 9, 11))
    print(inf, flush=True)
    if not vale:
        marca("FALLA_prevuelo.txt", "el interprete no puede con las etapas")
        return 1
    if not cfg["prueba"]:
        comprueba_disco()
    prepara_conjunto(cfg)
    for f in cfg["fases"]:
        if hecha(f):
            log("fase %s ya hecha: %s" % (f, (MARCAS / ("HECHA_%s.txt" % f)).read_text(encoding="utf-8").strip()))
            continue
        log("FASE %s" % f)
        t0 = time.time()
        try:
            r = FUNC[f](cfg)
        except Exception as ex:
            marca("FALLA_%s.txt" % f, "%s: %s" % (type(ex).__name__, ex))
            log("FALLA en %s: %s: %s. El director SE PARA." % (f, type(ex).__name__, ex))
            return 1
        marca("HECHA_%s.txt" % f, "%s  (%.1f h)" % (r, (time.time() - t0) / 3600))
        log("fase %s hecha en %.1f h: %s" % (f, (time.time() - t0) / 3600, r))
    if cfg["fases"][-1] == FASES[-1]:
        marca("FIN.txt", "falta la fase 3B (V-JEPA en el resto, G2 o V0): la decide el usuario con "
                         "compara_prioridad_G2_vs_V0_corrida.txt")
    log("DIRECTOR terminado")
    return 0


def pid_director():
    p = MARCAS / "director_pid.txt"
    if not p.is_file():
        return None
    pid = int(p.read_text(encoding="utf-8").split()[0])
    return pid if NO._vive(pid) else None


def lanza_director(args):
    if pid_director():
        print("ya hay un director vivo (pid %d): --estado o --para" % pid_director())
        return 1
    MARCAS.mkdir(parents=True, exist_ok=True)
    lg = MARCAS / "director.log"
    with open(lg, "ab") as fh:
        p = subprocess.Popen([PY, "-u", str(pathlib.Path(__file__).resolve()), "--director"] + args,
                             stdout=fh, stderr=subprocess.STDOUT, creationflags=FL, close_fds=True)
    (MARCAS / "director_pid.txt").write_text("%d\n" % p.pid, encoding="utf-8")
    print("director lanzado, pid %d, sin ventana. Registro: %s" % (p.pid, lg))
    print("estado: python corrida_florete.py --estado    parar: python corrida_florete.py --para")
    return 0


def para_director():
    pid = pid_director()
    if pid:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        print("director pid %d parado, con todo lo que habia lanzado" % pid)
    else:
        print("no hay director vivo")
    NO.para()
    """Por si algun trabajador de `noche.py` sobrevivio al arbol (cada uno lleva su grupo)."""


def informe(cfg, detallado):
    print("CORRIDA DE FLORETE (CORRIDA_FINAL_2026-10-13.md). Marcas en %s" % MARCAS)
    d = pid_director()
    print("director: %s" % ("VIVO, pid %d" % d if d else "no esta corriendo"))
    for u in DISCO_MIN_GB:
        print("disco %s: %.0f GB libres (pide %d)" % (u, libre_gb(u), DISCO_MIN_GB[u]))
    o = otros_gpu()
    print("GPU: %s" % ("libre" if not o else "OCUPADA por %d proceso(s): %s" % (len(o), o[0][:100])))
    clips = ids(cfg["conjunto"]) if pathlib.Path(cfg["conjunto"]).is_file() else ids("foil")
    if not pathlib.Path(cfg["conjunto"]).is_file():
        print("(el conjunto %s se escribe al lanzar: aqui se cuenta solo `foil`)" % cfg["conjunto"])
    lista = ids(cfg["lista"]) if cfg["lista"].is_file() else []
    vids = RU.indice_videos()
    """El indice UNA vez: `RU.video` por clip relee las carpetas de descarga en cada fallo, y con
    106.432 clips eso son horas."""
    print("conjunto %s: %d clips; con mp4 en disco %d; con luces %d"
          % (cfg["conjunto"], len(clips), sum(1 for c in clips if c in vids), cuenta(clips, [9])))
    print("lista de dos luces: %s" % ("%d clips, etapas 1+4 en %d, V-JEPA G2 en %d, V0 en %d"
                                      % (len(lista), cuenta(lista, [1, 4]),
                                         sum(1 for c in lista if (VJ_G2 / ("vjv_%s.npz" % c)).is_file()),
                                         sum(1 for c in lista if (VJ_V0 / ("vjv_%s.npz" % c)).is_file()))
                                      if lista else "todavia no existe"))
    print()
    for f in FASES:
        h = MARCAS / ("HECHA_%s.txt" % f)
        m = MARCAS / ("FALLA_%s.txt" % f)
        est = ("HECHA  " + h.read_text(encoding="utf-8").strip()) if h.is_file() else (
            ("FALLA  " + m.read_text(encoding="utf-8").strip()) if m.is_file() else "pendiente")
        print("   %-12s %-4s %s" % (f, "GPU" if f in GPU else "CPU", est))
    if detallado:
        print()
        print("SIN LANZAR NADA. Lo que haria, en orden (cada fase de GPU espera a que la GPU este libre):")
        print("   luces       noche.lanza(3 trabajadores de etapa 9, --descarga) sobre %s" % cfg["conjunto"])
        print("   lista       -> %s" % cfg["lista"])
        print("   dos_luces   noche.lanza(3 GPU etapas 1,4 + 2 CPU 7,9,11, --descarga) sobre la lista")
        print("   vjepa_g2    vjepa_variantes.py --todos --conjunto LISTA --modelo vjepa2-vitg-fpc64-256 "
              "--salidas rec4,mic4 --destino %s" % VJ_G2)
        print("   vjepa_v0    vjepa_variantes.py --todos --conjunto LISTA --modelo vjepa2-vitl-fpc64-256 "
              "--salidas rec4 --destino %s" % VJ_V0)
        print("   sonda_v0    sonda_vjepa.py --tarea prioridad --semillas 5 --dir %s --variante rec4 --clips-de LISTA" % VJ_V0)
        print("   sonda_g2    sonda_vjepa.py --tarea prioridad --semillas 5 --dir %s --variante rec4+mic4 "
              "--clips-de LISTA; compara_variantes V0 G2" % VJ_G2)
        print("   resto       noche.lanza(3 GPU + 2 CPU, --descarga) sobre %s" % cfg["conjunto"])
        print("   zona_fusion zona_fusion.py --conjunto %s --escribe" % cfg["conjunto"])
        print("   (3B, V-JEPA sobre el resto: NO, la decide el usuario)")


def main():
    global MARCAS, SAL
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seco", action="store_true")
    g.add_argument("--estado", action="store_true")
    g.add_argument("--lanza", action="store_true")
    g.add_argument("--para", action="store_true")
    g.add_argument("--director", action="store_true", help="(interno) el proceso que corre las fases")
    ap.add_argument("--ensayo", default=None,
                    help="fichero de ids: la corrida ENTERA (sin las sondas) sobre ellos, con "
                         "FENCINGLY_TRABAJO y FENCINGLY_PISTA en un scratch (obligatorio) y sin respaldo")
    ap.add_argument("--prueba", default=None,
                    help="fichero de ids YA PROCESADOS: corre luces, lista y dos_luces sobre ellos, con "
                         "marcas y salidas en el scratch y sin respaldo. No usa la GPU: lo comprueba")
    a = ap.parse_args()
    cfg = {"conjunto": CONJUNTO, "lista": LISTA, "fases": list(FASES), "respaldo": RESPALDO_MIN,
           "prueba": False}
    if a.prueba:
        MARCAS = RU.SCRATCH / "corrida_florete_prueba"
        SAL = MARCAS / "salidas"
        cfg.update(conjunto=pathlib.Path(a.prueba).resolve(), lista=MARCAS / "lista_prueba.txt",
                   fases=["luces", "lista", "dos_luces"], respaldo=0, prueba=True)
    if a.ensayo:
        defecto = RAIZ / "01_extraccion" / "pruebas"
        if RU.TRABAJO.resolve() == defecto.resolve() or RU.PISTA.resolve() == (RAIZ / "02_validacion" / "pista").resolve():
            raise SystemExit("--ensayo pide FENCINGLY_TRABAJO y FENCINGLY_PISTA fuera de las salidas de verdad")
        SAL = RU.TRABAJO / "salidas"
        cfg.update(conjunto=pathlib.Path(a.ensayo).resolve(), lista=RU.TRABAJO / "lista_ensayo.txt",
                   fases=[f for f in FASES if not f.startswith("sonda_")], respaldo=0)
    extra = (["--prueba", str(pathlib.Path(a.prueba).resolve())] if a.prueba else
             ["--ensayo", str(pathlib.Path(a.ensayo).resolve())] if a.ensayo else [])
    if a.seco or a.estado:
        informe(cfg, a.seco)
        return 0
    if a.lanza:
        return lanza_director(extra)
    if a.para:
        para_director()
        return 0
    return director(cfg)


if __name__ == "__main__":
    sys.exit(main())

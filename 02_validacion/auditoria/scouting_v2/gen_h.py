"""Genera `pista_h_` (y `zona_`) para los anotados de florete de `pruebas` corriendo la etapa 11 como en
`lotes.CODIGO_11`, en un TRABAJO de scratch con las entradas enlazadas (hardlink, mismo volumen C).
No escribe nada en `pruebas` ni en `02_validacion/pista`.

    python gen_h.py --prueba 20      solo 20 clips y compara su `zona_` con la de `pruebas`
    python gen_h.py                  todos, con 4 procesos, reanudable (salta los que tienen `zona_`)
"""
import csv, json, os, pathlib, random, subprocess, sys, time
RAIZ = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO")
PY = r"D:\FENCINGLY_PY\venv\Scripts\python.exe"
FL = subprocess.CREATE_NO_WINDOW
S = pathlib.Path(r"C:\Users\Alejandro\FENCINGLY_TMP\scout_h")
T, P = S / "trabajo", S / "pista"
AQUI = pathlib.Path(r"D:\FENCINGLY_SCRATCH\scout")
sys.path.insert(0, str(RAIZ / "01_extraccion" / "codigo"))
import catalogo as CA, rutas as RU  # noqa: E402  (las rutas de verdad: pruebas)
REAL = RU.TRABAJO
assert REAL != T
df = CA.manifiesto().set_index("clip_id")
fs = set(os.listdir(REAL))
clips = sorted(c for c in df.index[df.weapon == "foil"] if ("poses_%s.npz" % c) in fs and ("lights_%s.csv" % c) in fs
               and ("zona_%s.csv" % c) in fs)
random.Random(1014).shuffle(clips)
T.mkdir(parents=True, exist_ok=True); P.mkdir(parents=True, exist_ok=True)
for c in clips:
    for n in ("poses_%s.npz", "lights_%s.csv", "revision_%s.npz", "det_%s.pkl"):
        src, dst = REAL / (n % c), T / (n % c)
        if src.is_file() and not dst.exists():
            os.link(src, dst)
prueba = int(sys.argv[sys.argv.index("--prueba") + 1]) if "--prueba" in sys.argv else None
todo = clips[:prueba] if prueba else clips
(AQUI / "clips_h.txt").write_text("\n".join(todo) + "\n", encoding="utf-8")
N = int(os.environ.get("SCOUT_N", "4"))
COD = r'''
import sys, json
sys.path.insert(0, r"%s"); sys.path.insert(0, r"%s")
import rutas as RU, catalogo as CA, zona_tocado as ZT
df = CA.manifiesto().set_index("clip_id")
for c in [l.strip() for l in open(r"%s", encoding="utf-8") if l.strip()]:
    if (RU.TRABAJO / ("zona_%%s.csv" %% c)).is_file():
        continue
    RU.espera_ram(1.5)
    try:
        print(c, ZT.corre(c, {"video": RU.video(c), "torneo": df.loc[c, "torneo"] if c in df.index else None}), flush=True)
    except Exception as ex:
        print("%%s ETAPA 11 FALLA %%s: %%s" %% (c, type(ex).__name__, str(ex)[:100]), flush=True)
'''
env = dict(os.environ, FENCINGLY_TRABAJO=str(T), FENCINGLY_PISTA=str(P), PYTHONIOENCODING="utf-8")
procs = []
t0 = time.time()
for i in range(N):
    f = AQUI / ("trozo_h_%d.txt" % i)
    f.write_text("\n".join(todo[i::N]) + "\n", encoding="utf-8")
    lg = open(AQUI / ("trozo_h_%d.log" % i), "a", encoding="utf-8")
    code = COD % (RAIZ / "01_extraccion" / "codigo", RAIZ / "02_validacion" / "herramienta", f)
    procs.append((subprocess.Popen([PY, "-W", "ignore", "-c", code], env=env, stdout=lg, stderr=subprocess.STDOUT,
                                   cwd=str(AQUI), creationflags=FL), lg))
for p, lg in procs:
    p.wait(); lg.close()
print("%d clips en %.0f s con %d procesos" % (len(todo), time.time() - t0, N), flush=True)
if prueba:
    COLS = ["motivo", "cuadro_tocado", "via_luz", "cuadro_usado", "x_izq_m", "x_der_m", "x_tocado_m", "zona", "conf_h",
            "metodo_h", "frac_cuadros_con_h", "ang_lejano", "desp_pies_m", "dos_pies"]
    igual = 0; difs = {}
    for c in todo:
        a = list(csv.DictReader(open(REAL / ("zona_%s.csv" % c), encoding="utf-8")))[0]
        b = list(csv.DictReader(open(T / ("zona_%s.csv" % c), encoding="utf-8")))[0]
        d = [k for k in COLS if k in a and (a.get(k) or "") != (b.get(k) or "")]
        igual += not d
        for k in d:
            difs.setdefault(k, []).append((c[:20], a.get(k), b.get(k)))
    print("zona_ igual a pruebas (columnas de la etapa 11): %d de %d" % (igual, len(todo)))
    for k, v in difs.items():
        print(" ", k, len(v), v[:3])
    print("pista_h_ escritos:", sum((T / ("pista_h_%s.npz" % c)).is_file() for c in todo))

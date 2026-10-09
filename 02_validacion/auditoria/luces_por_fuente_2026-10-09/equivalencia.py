"""(a) con FENCINGLY_LUCES_V2=0 la etapa 9 refactorizada da EL MISMO CSV que la de disco (hecho hoy);
(b) con =1 las filas salen del detector viejo y coinciden con lo medido en mide.py (backup)."""
import os, sys, json, pathlib, filecmp, time
modo = sys.argv[1]
os.environ["FENCINGLY_TRABAJO"] = r"D:\FENCINGLY_SCRATCH\agente_luces\trabajo_eq" + modo
os.environ["FENCINGLY_LUCES_V2"] = modo
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion")
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import pipeline as PI, rutas as RU, catalogo as CA
REAL = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
df = CA.manifiesto().set_index("clip_id")
clips = [l.strip() for l in open(sys.argv[2]) if l.strip()]
med = {}
for l in open("med_anotados.jsonl"):
    r = json.loads(l); med[r["clip"]] = r
ok = mal = 0
for c in clips:
    ctx = {"video": RU.video(c), "clip": c, "torneo": df.loc[c, "torneo"]}
    print(c, PI._luces(c, ctx))
    nuevo = RU.TRABAJO / ("lights_%s.csv" % c)
    if modo == "0":
        igual = filecmp.cmp(nuevo, REAL / nuevo.name, shallow=False)
        ok += igual; mal += not igual
        if not igual: print("   DISTINTO de disco (mtime disco %s)" % time.ctime((REAL / nuevo.name).stat().st_mtime))
    else:
        import csv
        filas = list(csv.DictReader(open(nuevo, encoding="utf-8")))
        got = sorted((f["canal"], f["lado"], int(f["cuadro_encendido"])) for f in filas if f.get("cuadro_encendido"))
        v = med.get(c, {}).get("viejo") or {}
        m = {"red_L": ("roja", "L"), "green_R": ("verde", "R"), "white_L": ("blanca", "L"), "white_R": ("blanca", "R")}
        exp = sorted(m[k] + (x,) for k, x in (v.get("cuadros") or {}).items() if x is not None)
        if v.get("f_touch") is None:
            exp = None
        igual = (exp is None and filas and filas[0]["via"].startswith("tras_v2")) or got == exp
        ok += bool(igual); mal += not igual
        if not igual: print("   DISTINTO", got, exp)
print("iguales %d  distintos %d" % (ok, mal))

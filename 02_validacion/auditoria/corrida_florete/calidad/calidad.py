import csv, glob, os, pathlib, sys, collections
import numpy as np
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import catalogo as CA, rutas as RU
df = CA.manifiesto().set_index("clip_id")
def fila(p):
    try:
        with open(p, encoding="utf-8") as fh:
            r = list(csv.DictReader(fh))
        return r
    except Exception:
        return None
def mira(nombre, T, P, clips):
    n = sh = pub = zok = zh = 0
    ambos = []
    for c in clips:
        pc = fila(P / ("posicion_clip_%s.csv" % c)); z = fila(T / ("zona_%s.csv" % c))
        if pc is None or z is None:
            continue
        n += 1
        sh += pc[0].get("sin_homografia") == "True"
        pub += pc[0].get("publicable") == "True"
        zok += z[0].get("motivo") == "ok"
        zh += z[0].get("motivo") == "ok" and pc[0].get("sin_homografia") == "True"
    print("%-28s n=%5d | etapa 7 SIN homografia %5.1f %% | publicable %5.1f %% | zona 11 ok %5.1f %% | zona ok pero 7 sin H %5.1f %%"
          % (nombre, n, 100 * sh / n, 100 * pub / n, 100 * zok / n, 100 * zh / n))
for e in ("e_1014_0011", "e_1014_1808"):
    E = pathlib.Path(r"D:\FENCINGLY_SCRATCH\ensayo") / e
    cl = open(E / "clips.txt", encoding="utf-8").read().split()
    mira("ensayo " + e, E / "trabajo", E / "pista", cl)
T, P = RU.TRABAJO, RU.PISTA
cl = [p.name[5:-4] for p in T.glob("zona_*.csv")]
mira("pruebas (anotados)", T, P, cl)
for w in ("foil", "epee", "sabre"):
    mira("pruebas " + w, T, P, [c for c in cl if c in df.index and df.loc[c, "weapon"] == w])

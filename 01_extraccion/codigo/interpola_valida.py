"""Mide el error de la interpolacion sobre el corpus y FIJA el limite de hueco con la curva.

CRITERIO DECLARADO ANTES DE MIRAR: el limite de hueco sera el mayor largo cuyo error mediano se
quede por debajo del **2 % del alto de caja**. Es el mismo umbral que se uso para elegir la
resolucion de Sapiens, y por el mismo motivo: en una caja de 310 px son unos 6 px, que es el orden
del ruido del propio detector.

TAMBIEN se censan los huecos reales, separando los INTERIORES (interpolables) de los de BORDE (no
interpolables porque no tienen dos extremos). Esa distincion no la hacia ninguna medida anterior y
decide cuanto se puede arreglar de verdad.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import interpola as IP

BASE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565")
DIRS = [BASE / "localrun/runs_extra1/data_20260907T005507157743Z/npz",
        BASE / "localrun/runs_extra2/data_20260907T005507370214Z/npz",
        BASE / "localrun/runs_extra3/data_20260907T010246490954Z/npz"]
IDX = BASE / "localrun/runs_full/data_20260909T092005337087Z/npz/index.csv"
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
SALIDA.mkdir(parents=True, exist_ok=True)
N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
LIMITE_FRAC = 0.02

m = pd.read_csv(IDX, usecols=["clip_id", "fps", "tournament_id"], low_memory=False)
fps_de = dict(zip(m.clip_id.astype(str), m.fps))

ficheros = []
for d in DIRS:
    if d.is_dir():
        ficheros += sorted(d.glob("*.npz"))
ficheros = ficheros[:N_MAX]
print("npz a procesar: %d" % len(ficheros))

cen, val = [], []
for f in ficheros:
    fps = fps_de.get(f.stem)
    if not fps:
        continue
    try:
        with np.load(f, allow_pickle=True) as z:
            if "poses" not in z.files or "boxes" not in z.files:
                continue
            P = np.asarray(z["poses"], dtype="float64")
            B = np.asarray(z["boxes"], dtype="float64")
            meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    except Exception:
        continue
    if P.ndim != 4 or P.shape[1] != 2 or P.shape[0] < 6:
        continue
    u = IP.utiles(P)
    for lado in (0, 1):
        for h in IP.huecos(u[:, lado]):
            cen.append({"clip": f.stem, "lado": lado, "largo": h["largo"], "tipo": h["tipo"],
                        "seg": h["largo"] / float(fps), "T": P.shape[0],
                        "f0": int(meta.get("f0", 0))})
    val += [{**r, "clip": f.stem} for r in IP.valida_escondiendo(P, B)]

dc = pd.DataFrame(cen)
dv = pd.DataFrame(val)
dc.to_csv(SALIDA / "censo_huecos.csv", index=False)
dv.to_csv(SALIDA / "validacion_interpolacion.csv", index=False)

print()
print("=" * 88)
print("1) CENSO DE HUECOS")
print("=" * 88)
if len(dc):
    tot_cuadros = int((dc.groupby("clip")["T"].first().sum()) * 2)
    for tipo in ("interior", "borde"):
        g = dc[dc.tipo == tipo]
        print("  %-9s huecos %6d   cuadros %7d   largo mediano %4.0f   p90 %5.0f   max %5.0f" % (
            tipo, len(g), int(g.largo.sum()) if len(g) else 0,
            g.largo.median() if len(g) else 0, g.largo.quantile(0.9) if len(g) else 0,
            g.largo.max() if len(g) else 0))
    ci = dc[dc.tipo == "interior"]
    cb = dc[dc.tipo == "borde"]
    print()
    print("  cuadros x tirador en total ....... %d" % tot_cuadros)
    print("  en hueco INTERIOR (arreglable) ... %d  (%.2f %%)" % (
        int(ci.largo.sum()), 100.0 * ci.largo.sum() / max(tot_cuadros, 1)))
    print("  en hueco de BORDE (no arreglable)  %d  (%.2f %%)" % (
        int(cb.largo.sum()), 100.0 * cb.largo.sum() / max(tot_cuadros, 1)))

print()
print("=" * 88)
print("2) ERROR DE LA INTERPOLACION, escondiendo cuadros que SI existen")
print("=" * 88)
if len(dv):
    print("  %6s %8s %12s %12s %12s %10s" % (
        "hueco", "muestras", "err mediano", "err p90", "err maximo", "veredicto"))
    limite = None
    for L, g in dv.groupby("largo"):
        med = float(g.err_med.median())
        p90 = float(g.err_med.quantile(0.9))
        mx = float(g.err_max.max())
        ok = med < LIMITE_FRAC
        if ok:
            limite = int(L)
        print("  %6d %8d %11.3f %% %11.3f %% %11.3f %% %10s" % (
            L, len(g), 100 * med, 100 * p90, 100 * mx, "vale" if ok else "PASA DEL LIMITE"))
    print()
    print("  limite declarado: error mediano < %.0f %% del alto de caja" % (100 * LIMITE_FRAC))
    if limite:
        print("  LIMITE DE HUECO MEDIDO: **%d cuadros**" % limite)
        if len(dc):
            ci = dc[dc.tipo == "interior"]
            arregla = ci[ci.largo <= limite]
            print("  con ese limite se arreglan %d de %d huecos interiores (%.1f %%)" % (
                len(arregla), len(ci), 100.0 * len(arregla) / max(len(ci), 1)))
            print("  o sea %d cuadros de los %d que estan en hueco interior (%.1f %%)" % (
                int(arregla.largo.sum()), int(ci.largo.sum()),
                100.0 * arregla.largo.sum() / max(ci.largo.sum(), 1)))
    else:
        print("  NINGUN largo baja del limite: la interpolacion lineal no sirve ni para 1 cuadro,")
        print("  y eso habria que decirlo en vez de bajar el limite.")

print()
print("=" * 88)
print("3) EFECTO DE APLICARLO: cuantos cuadros pasan a tener las dos poses")
print("=" * 88)
if len(dv) and limite:
    antes_t, despues_t, nclip = 0, 0, 0
    for f in ficheros[:300]:
        try:
            with np.load(f, allow_pickle=True) as z:
                if "poses" not in z.files:
                    continue
                P = np.asarray(z["poses"], dtype="float64")
        except Exception:
            continue
        if P.ndim != 4 or P.shape[1] != 2 or P.shape[0] < 6:
            continue
        r = IP.rellena(P, limite)
        antes_t += r["utiles_antes"]
        despues_t += r["utiles_despues"]
        nclip += 1
    print("  sobre %d clips:" % nclip)
    print("    poses utiles antes ..... %d" % antes_t)
    print("    poses utiles despues ... %d   (+%d, +%.2f %%)" % (
        despues_t, despues_t - antes_t, 100.0 * (despues_t - antes_t) / max(antes_t, 1)))
print()
print("guardado en %s" % SALIDA)

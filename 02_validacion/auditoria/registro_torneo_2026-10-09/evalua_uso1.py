"""USO 1 (respaldo): clips de AJUSTE que HOY no contestan, registrados contra las referencias de su
torneo, contra `strip_location` (criterio (a) de `REGISTRO_TORNEO_2026-10-09.md`).

Reutiliza `mide_zona.boot` (el mismo bootstrap pareado por clip/por torneo de la adopcion del
08/10), sin reinventarlo.

    python evalua_uso1.py empareja_uso1_ajuste.jsonl.gz 15
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys

import numpy as np

_RAIZ = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_RAIZ / "02_validacion" / "herramienta"))
sys.path.insert(0, str(_RAIZ / "01_extraccion" / "codigo"))

import catalogo as CA  # noqa: E402
import zona_tocado as ZT  # noqa: E402
from mide_zona import boot  # noqa: E402


def carga(path):
    abre = gzip.open if path.endswith(".gz") else open
    with abre(path, "rt", encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh]


def registra(fila, min_inliers, min_matches=0, min_acuerdo=None):
    cand = [r for r in fila["resultados"]
            if r["x"] is not None and r["ninliers"] >= min_inliers and r["nmatches"] >= min_matches]
    if not cand:
        return None, 0
    xs = np.array([r["x"] for r in cand])
    x0 = float(np.median(xs))
    if min_acuerdo is not None:
        de_acuerdo = np.abs(xs - x0) <= 1.5
        if de_acuerdo.mean() < min_acuerdo:
            return None, len(cand)
        xs = xs[de_acuerdo]
        x0 = float(np.median(xs))
    return x0, len(cand)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "empareja_uso1_ajuste.jsonl.gz"
    min_inliers = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    min_acuerdo = float(sys.argv[3]) if len(sys.argv) > 3 else None
    filas = carga(path)
    df = CA.manifiesto().set_index("clip_id")
    ya = [f for f in filas if f["motivo"] == "ok"]
    sin = [f for f in filas if f["motivo"] != "ok"]
    print("filas: %d (ya contestaban: %d, sin homografia: %d)" % (len(filas), len(ya), len(sin)))

    nuevos = []
    n_sin_etiqueta = 0
    n_sin_registro = 0
    for f in sin:
        x, nref = registra(f, min_inliers, min_acuerdo=min_acuerdo)
        if x is None:
            n_sin_registro += 1
            continue
        clip = f["clip"]
        if clip not in df.index:
            continue
        y = df.loc[clip, "strip_location"]
        if not y or y not in ZT.ZONAS:
            n_sin_etiqueta += 1
            continue
        z = ZT.ZONAS[ZT.zona(x)]
        nuevos.append({"clip": clip, "torneo": f["torneo"], "y": y, "z": z, "x": x, "n_ref": nref})

    print("sin referencia que pase el umbral (inliers>=%d): %d" % (min_inliers, n_sin_registro))
    print("registrados sin etiqueta util: %d" % n_sin_etiqueta)
    print("NUEVOS con etiqueta y registro: %d de %d sin homografia (%.3f)"
          % (len(nuevos), len(sin), len(nuevos) / max(len(sin), 1)))
    if not nuevos:
        print("(nada que evaluar)")
        return
    y = np.array([n["y"] for n in nuevos])
    z = np.array([n["z"] for n in nuevos])
    tor = np.array([n["torneo"] for n in nuevos])
    ok_z = z == y
    ok_m = y == "middle"
    lo, hi = boot(ok_m, ok_z)
    lo_t, hi_t = boot(ok_m, ok_z, tor)
    print()
    print("(a) NUEVOS: registro %.3f   middle %.3f   diferencia %+.3f" % (ok_z.mean(), ok_m.mean(), ok_z.mean() - ok_m.mean()))
    print("    IC95 por clip   [%+.3f, %+.3f]" % (lo, hi))
    print("    IC95 por torneo [%+.3f, %+.3f]" % (lo_t, hi_t))
    print("    CRITERIO (bate a middle, IC95 por clip sin cruzar cero): %s" % ("PASA" if lo > 0 else "NO PASA"))
    print()
    print("por torneo:")
    for t in sorted(set(tor)):
        m = tor == t
        if m.sum() < 5:
            continue
        print("   %-26s n=%3d  registro %.3f  middle %.3f  %+.3f" % (t, m.sum(), ok_z[m].mean(), ok_m[m].mean(), ok_z[m].mean() - ok_m[m].mean()))

    # (b) los que ya contestaban: no se tocan POR CONSTRUCCION (el respaldo solo actua si `cand`
    # esta vacio en produccion); aqui solo se informa de cuantos de los 57 "ok" tendrian una
    # respuesta del registro, para ver si coincide con la de produccion (diagnostico, no cambia nada).
    coincide = 0
    difiere = 0
    for f in ya:
        x, nref = registra(f, min_inliers, min_acuerdo=min_acuerdo)
        if x is None or f.get("zona_prod") is None:
            continue
        z = ZT.ZONAS[ZT.zona(x)]
        if z == f["zona_prod"]:
            coincide += 1
        else:
            difiere += 1
    print()
    print("(b) diagnostico en los %d que YA contestaban (el respaldo no los tocaria en produccion):"
          % len(ya))
    print("    registro coincide con la zona de produccion: %d; difiere: %d (de %d con registro)"
          % (coincide, difiere, coincide + difiere))


if __name__ == "__main__":
    main()

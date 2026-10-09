"""USO 2 (informativo/fusion): en los MISMOS 614 clips del PASO 1 (que ya contestan), compara la
zona del registro (mediana de las referencias que pasan el umbral de inliers) contra `zona_final`
(la fusion homografia+angulo ya adoptada, `ANGULO_FUSION_2026-10-09.md`). No hace falta cache ni
emparejamiento nuevos: reusa `empareja_ajuste.jsonl.gz` del PASO 1.

    python evalua_uso2.py empareja_ajuste.jsonl.gz 15
"""
from __future__ import annotations

import csv
import gzip
import json
import pathlib
import sys

import numpy as np

_RAIZ = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_RAIZ / "02_validacion" / "herramienta"))
sys.path.insert(0, str(_RAIZ / "01_extraccion" / "codigo"))

import rutas as RU  # noqa: E402
import zona_tocado as ZT  # noqa: E402
from mide_zona import boot  # noqa: E402


def carga(path):
    abre = gzip.open if path.endswith(".gz") else open
    with abre(path, "rt", encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh]


def registra(fila, min_inliers):
    cand = [r for r in fila["resultados"] if r["x"] is not None and r["ninliers"] >= min_inliers]
    if not cand:
        return None
    return float(np.median([r["x"] for r in cand]))


def zona_final_de(clip):
    p = RU.TRABAJO / ("zona_%s.csv" % clip)
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as fh:
        r = next(csv.DictReader(fh), None) or {}
    return r.get("zona_final") or None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "empareja_ajuste.jsonl.gz"
    min_inliers = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    filas = carga(path)
    datos = []
    for f in filas:
        x = registra(f, min_inliers)
        if x is None or f.get("zona_prod") is None:
            continue
        zf = zona_final_de(f["clip"])
        if not zf or zf not in ZT.ZONAS:
            continue
        z = ZT.ZONAS[ZT.zona(x)]
        datos.append({"clip": f["clip"], "torneo": f["torneo"], "zf": zf, "z": z,
                      "zh": f["zona_prod"]})
    print("clips con registro, homografia estricta Y zona_final: %d de %d" % (len(datos), len(filas)))
    if not datos:
        return
    zf = np.array([d["zf"] for d in datos])
    z = np.array([d["z"] for d in datos])
    zh = np.array([d["zh"] for d in datos])
    tor = np.array([d["torneo"] for d in datos])
    ok_h = zh == zf
    ok_reg = z == zf
    lo, hi = boot(ok_h, ok_reg)
    lo_t, hi_t = boot(ok_h, ok_reg, tor)
    print()
    print("acierta zona_final:  homografia estricta %.3f   registro %.3f   diferencia %+.3f"
          % (ok_h.mean(), ok_reg.mean(), ok_reg.mean() - ok_h.mean()))
    print("IC95 por clip   [%+.3f, %+.3f]" % (lo, hi))
    print("IC95 por torneo [%+.3f, %+.3f]" % (lo_t, hi_t))
    print("CRITERIO (>= +0.02, IC95 por torneo sin cruzar cero): %s"
          % ("PASA" if lo_t > 0 and (ok_reg.mean() - ok_h.mean()) >= 0.02 else "NO PASA"))
    print()
    for t in sorted(set(tor)):
        m = tor == t
        if m.sum() < 5:
            continue
        print("   %-26s n=%3d  homografia %.3f  registro %.3f  %+.3f"
              % (t, m.sum(), ok_h[m].mean(), ok_reg[m].mean(), ok_reg[m].mean() - ok_h[m].mean()))


if __name__ == "__main__":
    main()

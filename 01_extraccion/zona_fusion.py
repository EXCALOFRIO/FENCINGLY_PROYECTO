"""La zona FINAL del tocado: la homografia de la etapa 11 desempatada con el angulo del borde lejano.

POR QUE EXISTE (2026-10-09, `04_documentacion/informes/ANGULO_FUSION_2026-10-09.md`). La regla D(2)
paso su criterio en PRUEBA: acierto TOTAL de zona 0,3089 -> 0,3338 (+0,0249, IC97,5 por clip
[+0,0113, +0,0380], n = 2.130), sin bajar en ningun arma. Pero el modelo del angulo es POR TORNEO
(se ajusta con los clips del mismo torneo que tienen homografia), asi que no cabe en la etapa 11,
que es por clip: es un paso DESPUES de ella, sobre todos los `zona_` del conjunto. Correrlo dos veces
no cambia nada; correrlo con mas clips del torneo puede mover la x de algunos (el modelo mejora), y
por eso no se escribe un fichero que no cambia.

QUE ESCRIBE en `zona_<clip>.csv`, ademas de lo de la etapa 11:
    ang_lejano, desp_pies_m, dos_pies   los rasgos del angulo (`angulo_lejano.rasgos`), si faltan
    x_angulo_m                          la x del modelo del torneo SIN el clip (None sin modelo)
    x_final_m, zona_final, via_zona     la zona que hay que usar y de donde sale:
                                        `homografia`, `angulo_torneo` (D(2)) o
                                        `homografia_sin_modelo` (torneo con < 8 referencias)
La `zona` y la `x_tocado_m` de la homografia NO se tocan: son la procedencia.

`FENCINGLY_ZONA_FUSION=0` deja `zona_final = zona` (via `homografia`) en todos.

    python zona_fusion.py --conjunto anotados                      dice que haria, no escribe
    python zona_fusion.py --conjunto anotados --escribe            escribe en RU.TRABAJO
    python zona_fusion.py --conjunto anotados --destino D:/prueba  escribe la copia en otro sitio
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "codigo"))
import rutas as RU  # noqa: E402

FUSION = os.environ.get("FENCINGLY_ZONA_FUSION", "1") == "1"
NUEVAS = ["ang_lejano", "desp_pies_m", "dos_pies", "x_angulo_m", "x_final_m", "zona_final", "via_zona"]


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def _lee(p):
    with open(p, encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        fila = next(rd, None)
        return (fila or {}), list(rd.fieldnames or [])


def _rasgos_uno(clip, fila):
    import angulo_lejano as AL
    try:
        r = AL.rasgos(clip, fila)
        return clip, r, None
    except Exception as ex:
        return clip, None, "%s: %s" % (type(ex).__name__, str(ex)[:120])


def main():
    import catalogo as CA
    import lotes as LO
    import zona_tocado as ZT
    import angulo_lejano as AL
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", default="anotados")
    ap.add_argument("--escribe", action="store_true")
    ap.add_argument("--destino", default=None)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--cache", default=None,
                    help="jsonl(.gz) de `rasgos.py` del agente H (RAYAS_ID): toma de ahi los rasgos del "
                         "angulo en vez de decodificar; comprobado identico a `angulo_lejano.rasgos`")
    a = ap.parse_args()
    df = CA.manifiesto().set_index("clip_id")
    filas = {}
    for c in LO.conjunto(a.conjunto):
        p = RU.TRABAJO / ("zona_%s.csv" % c)
        if p.is_file() and p.stat().st_size:
            filas[c] = _lee(p)[0]
    if a.cache:
        import gzip
        import json
        abre = gzip.open if a.cache.endswith(".gz") else open
        n_cache = 0
        with abre(a.cache, "rt", encoding="utf-8") as fh:
            for linea in fh:
                r = json.loads(linea)
                c = r.get("clip")
                if c in filas and r.get("fr") and filas[c].get("ang_lejano") in (None, ""):
                    v = AL.resume(r["fr"], r["W"], r["H"])
                    filas[c]["ang_lejano"] = "" if v["ang"] is None else "%.6f" % v["ang"]
                    filas[c]["desp_pies_m"] = "" if v["desp"] is None else "%.6f" % v["desp"]
                    filas[c]["dos_pies"] = "" if v["dos_pies"] is None else str(bool(v["dos_pies"]))
                    n_cache += 1
        print("rasgos del angulo tomados de la cache: %d" % n_cache, flush=True)
    falta = [c for c, f in filas.items() if (f.get("ang_lejano") in (None, "")
             and f.get("dos_pies") in (None, ""))
             and ((f.get("motivo") or "") == "ok" or (f.get("motivo") or "").startswith("sin homografia"))]
    print("%s: %d zona_; rasgos del angulo por calcular %d" % (a.conjunto, len(filas), len(falta)), flush=True)
    errores = 0
    if falta:
        with ProcessPoolExecutor(a.procs) as ex:
            fut = [ex.submit(_rasgos_uno, c, filas[c]) for c in falta]
            for i, f in enumerate(as_completed(fut), 1):
                c, r, err = f.result()
                if err:
                    errores += 1
                    continue
                filas[c]["ang_lejano"] = "" if r["ang"] is None else "%.6f" % r["ang"]
                filas[c]["desp_pies_m"] = "" if r["desp"] is None else "%.6f" % r["desp"]
                filas[c]["dos_pies"] = "" if r["dos_pies"] is None else str(bool(r["dos_pies"]))
                if i % 500 == 0:
                    print("   rasgos %d / %d" % (i, len(falta)), flush=True)
    por = collections.defaultdict(list)
    for c, f in filas.items():
        f["_t"] = df.loc[c, "torneo"] if c in df.index else None
        f["_x"] = {"clip": c, "ang": _f(f.get("ang_lejano")), "desp": _f(f.get("desp_pies_m")),
                   "x_h": _f(f.get("x_tocado_m")) if f.get("motivo") == "ok" else None}
        if f["_x"]["x_h"] is not None and f["_x"]["ang"] is not None and f["_x"]["desp"] is not None:
            por[f["_t"]].append(f["_x"])
    cuenta = collections.Counter()
    for c, f in filas.items():
        x_h = f["_x"]["x_h"]
        x_a = AL.x_angulo(por.get(f["_t"], []), f["_x"]) if (FUSION and x_h is not None) else None
        if x_h is None:
            xf, via = None, None
        elif not FUSION:
            xf, via = x_h, "homografia"
        else:
            xf, via = AL.decide(x_h, x_a)
            if x_a is None:
                via = "homografia_sin_modelo"
        f["x_angulo_m"] = "" if x_a is None else "%.3f" % x_a
        f["x_final_m"] = "" if xf is None else "%.3f" % xf
        f["zona_final"] = "" if xf is None else ZT.ZONAS[ZT.zona(xf)]
        f["via_zona"] = via or ""
        cuenta[via] += 1
        cuenta["cambia de zona"] += bool(xf is not None and f["zona_final"] != f.get("zona"))
    print("via_zona:", dict(cuenta), " errores de rasgos:", errores, flush=True)
    if not (a.escribe or a.destino):
        print("(sin --escribe ni --destino: no se escribe nada)")
        return 0
    dest = pathlib.Path(a.destino) if a.destino else RU.TRABAJO
    dest.mkdir(parents=True, exist_ok=True)
    escritos = 0
    for c, f in filas.items():
        p = RU.TRABAJO / ("zona_%s.csv" % c)
        vieja, cols = _lee(p)
        cols = [k for k in cols if k not in NUEVAS] + NUEVAS
        nueva = {k: ("" if f.get(k) is None else str(f.get(k))) for k in cols}
        if a.destino is None and all(str(vieja.get(k) or "") == nueva[k] for k in cols) \
                and set(NUEVAS) <= set(vieja):
            continue
        q = dest / p.name
        tmp = q.with_name(q.name + ".part")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerow(nueva)
        os.replace(tmp, q)
        escritos += 1
    print("escritos %d en %s" % (escritos, dest), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

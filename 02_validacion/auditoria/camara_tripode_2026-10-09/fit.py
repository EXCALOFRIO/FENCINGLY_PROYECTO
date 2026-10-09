"""Ajusta la camara de tripode sobre la cache de un clip (sin la homografia de produccion).

    python fit.py CACHE_DIR LISTA.txt SALIDA.jsonl [procs]
"""
import json
import os
import pathlib
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")


def evidencia_de(r):
    import numpy as np
    import pista_tripode as PT
    ah, aw = r["ash"]
    ar = {t: np.unpackbits(v)[: ah * aw].reshape(ah, aw) for t, v in r["aristas"].items()}
    va = {t: np.unpackbits(v)[: ah * aw].reshape(ah, aw) for t, v in r["validas"].items()}
    M = PT.cadena(r["G"], r["tq"], r["ventana"])
    return PT.Evidencia(ar, va, M, r["s"], (ah, aw))


def uno(cache_dir, clip):
    import numpy as np
    import pista_tripode as PT
    import zona_tocado as ZT
    f = pathlib.Path(cache_dir) / ("%s.pkl" % clip)
    if not f.is_file():
        return {"clip": clip, "estado": "sin cache"}
    with open(f, "rb") as fh:
        r = pickle.load(fh)
    out = {"clip": clip, "torneo": r.get("torneo"), "tf": r.get("tf")}
    if "tq" not in r:
        out["estado"] = r.get("motivo", "sin tq")
        return out
    prod = r["prod"]
    out["prod_ok"] = prod.get("ok", False)
    if prod.get("ok"):
        out.update(prod_x=prod["x"], prod_zona=prod["zona"], prod_xs=prod["xs"], prod_t=prod["t"])
    t0 = time.perf_counter()
    ev = evidencia_de(r)
    tq = r["tq"]
    res = PT.ajusta(ev, r["pies"][tq], r["alt"][tq], r["W"], r["H"])
    out["s_fit"] = round(time.perf_counter() - t0, 1)
    out["n_ventana"] = len(r["ventana"])
    out["n_cadena"] = len(ev.ts)
    out["s_cache"] = r.get("seg")
    if res is None:
        out["estado"] = "sin ajuste"
        return out
    out["estado"] = "ok"
    out.update(t_x=res["x"], t_xs=res["xs"], t_zona=ZT.ZONAS[ZT.zona(res["x"])], t_obj=res["obj"],
               t_rec=res["rec"], t_margen=res["margen"], t_rayas=res["rayas"],
               t_p=[float(v) for v in res["p"]], t_npts=res["n_puntos"])
    return out


def main():
    cache_dir, lista, salida = sys.argv[1], sys.argv[2], sys.argv[3]
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    clips = [l.strip() for l in open(lista, encoding="utf-8") if l.strip()]
    hechos = set()
    if os.path.exists(salida):
        hechos = {json.loads(l)["clip"] for l in open(salida, encoding="utf-8") if l.strip()}
    clips = [c for c in clips if c not in hechos]
    t0 = time.time()
    with ProcessPoolExecutor(procs) as ex, open(salida, "a", encoding="utf-8") as fo:
        fut = {ex.submit(uno, cache_dir, c): c for c in clips}
        for i, fu in enumerate(as_completed(fut), 1):
            try:
                o = fu.result()
            except Exception as e:
                o = {"clip": fut[fu], "estado": "FALLO %s: %s" % (type(e).__name__, str(e)[:200])}
            fo.write(json.dumps(o) + "\n")
            fo.flush()
            if i % 25 == 0:
                print("%d/%d %.0f s" % (i, len(clips), time.time() - t0), flush=True)
    print("hecho %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

"""R3: la etapa 9 vigente con FENCINGLY_LUZ_ESCALON_VISTO=1. Mismo formato que mide.py (clave 'r3')."""
import json, os, pathlib, sys, time
os.environ["FENCINGLY_LUZ_ESCALON_VISTO"] = "1"
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
sys.path.insert(0, r"D:\FENCINGLY_SCRATCH\agente_luces")
import mide as M

def uno(args):
    clip, torneo, arma = args
    import luces as LU, rutas as RU
    assert LU.ESCALON_VISTO
    v = RU.video(clip)
    out = dict(clip=clip)
    t0 = time.perf_counter()
    try:
        out["r3"] = M.vigente(v, torneo)
    except Exception as ex:
        out["r3"] = {"error": "%s: %s" % (type(ex).__name__, str(ex)[:200])}
    out["s_r3"] = round(time.perf_counter() - t0, 2)
    return out

if __name__ == "__main__":
    import catalogo as CA
    lista = [l.strip() for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    df = CA.manifiesto().set_index("clip_id")
    dst = pathlib.Path(r"D:\FENCINGLY_SCRATCH\agente_luces\med_r3.jsonl")
    hechos = set()
    if dst.is_file():
        hechos = {json.loads(l)["clip"] for l in open(dst, encoding="utf-8") if l.strip()}
    tareas = [(c, df.loc[c, "torneo"], df.loc[c, "weapon"]) for c in lista if c in df.index and c not in hechos]
    print(len(tareas), "por hacer", flush=True)
    with ProcessPoolExecutor(int(os.environ.get("PROCS", "6"))) as ex, open(dst, "a", encoding="utf-8") as fh:
        for i, fu in enumerate(as_completed([ex.submit(uno, t) for t in tareas])):
            fh.write(json.dumps(fu.result(), default=str) + "\n"); fh.flush()
            if i % 500 == 0: print(i, flush=True)
    print("fin", flush=True)

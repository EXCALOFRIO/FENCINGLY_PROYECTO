"""R3b: la etapa 9 vigente con FENCINGLY_LUZ_SIN_CUADRO0=1 (clave 'r3b')."""
import json, os, pathlib, sys, time
os.environ["FENCINGLY_LUZ_SIN_CUADRO0"] = "1"
os.environ["FENCINGLY_LUCES_V2"] = "0"
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")

def uno(args):
    clip, torneo = args
    import luces as LU, rutas as RU
    assert LU.SIN_CUADRO0 and not LU.ESCALON_VISTO and not LU.USA_V2
    t0 = time.perf_counter()
    filas, via, fps = LU.lee_clip(RU.video(clip), RU.roi_sede(torneo))
    return dict(clip=clip, r3b=dict(via=via, fps=fps, luces=[{k: f.get(k) for k in ("canal", "lado", "cuadro_encendido")} for f in filas]),
                s_r3b=round(time.perf_counter() - t0, 2))

if __name__ == "__main__":
    import catalogo as CA
    lista = [l.strip() for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    df = CA.manifiesto().set_index("clip_id")
    dst = pathlib.Path(r"D:\FENCINGLY_SCRATCH\agente_luces\med_r3b.jsonl")
    with ProcessPoolExecutor(6) as ex, open(dst, "a", encoding="utf-8") as fh:
        for fu in as_completed([ex.submit(uno, (c, df.loc[c, "torneo"])) for c in lista]):
            fh.write(json.dumps(fu.result(), default=str) + "\n"); fh.flush()
    print("fin", flush=True)

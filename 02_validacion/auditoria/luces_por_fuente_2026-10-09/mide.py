"""Recalcula las luces de cada clip con el detector VIGENTE (etapa 9 tal cual) y con el VIEJO
(fencingly_v2.lights_generic.analyze_any, congelado, sin tocar un parametro), sobre los mismos videos.
Escribe una linea JSON por clip en med_<tag>.jsonl. Reanudable. Solo CPU, 6 procesos."""
import json
import os
import pathlib
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, r"D:\FENCINGLY_respaldo_2026-09-09\scratchpad\fz\codigo\FENCINGLY\webapp")
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")

OUT = pathlib.Path(r"D:\FENCINGLY_SCRATCH\agente_luces")


def vigente(video, torneo):
    import luces as LU
    import rutas as RU
    roi = RU.roi_sede(torneo)
    det = LU.detecta(video, banda=roi["banda"]) if roi else LU.detecta(video)
    via = "roi_sede" if roi else "general"
    fps = det.get("fps") or 25.0
    gr = LU.grupo_del_tocado(LU.agrupa(det["luces"], fps), fps)
    if roi and not gr:
        det2 = LU.detecta(video)
        gr = LU.grupo_del_tocado(LU.agrupa(det2["luces"], fps), fps)
        via = "general_tras_roi"
    filas = LU.resumen_clip(gr) or []
    return dict(via=via, fps=fps, n=det.get("n"),
                luces=[{k: f.get(k) for k in ("canal", "lado", "cuadro_encendido", "cuadro_apagado",
                                               "confianza", "area", "cx", "cy", "discordante")}
                       for f in filas])


def viejo(video):
    from fencingly_v2 import lights_generic as LG
    r = LG.analyze_any(video, clock=True)
    fps = r.get("fps")
    tl = r.get("touch_lights") or {}
    return dict(source=r.get("source"), fps=fps, f_touch=r.get("f_touch"),
                touch_lights={k: tl.get(k) for k in ("red_L", "green_R", "white_L", "white_R")},
                cuadros={k: (int(round(v * fps)) if v is not None and fps else None)
                         for k, v in tl.items()},
                flags=r.get("quality_flags"), n_frames=r.get("n_frames"), size=r.get("size"),
                box_method=r.get("box_method"), error=r.get("error") or r.get("fie_error"))


def uno(args):
    clip, torneo, arma = args
    import rutas as RU
    v = RU.video(clip)
    out = dict(clip=clip, torneo=torneo, arma=arma)
    if not v:
        out["error"] = "sin video"
        return out
    for nombre, f in (("vigente", lambda: vigente(v, torneo)), ("viejo", lambda: viejo(v))):
        t0 = time.perf_counter()
        try:
            out[nombre] = f()
        except Exception as ex:
            out[nombre] = {"error": "%s: %s" % (type(ex).__name__, str(ex)[:200]),
                           "tb": traceback.format_exc()[-600:]}
        out["s_" + nombre] = round(time.perf_counter() - t0, 2)
    return out


def main():
    import catalogo as CA
    tag = sys.argv[1]
    lista = [l.strip() for l in open(sys.argv[2], encoding="utf-8") if l.strip()]
    df = CA.manifiesto().set_index("clip_id")
    dst = OUT / ("med_%s.jsonl" % tag)
    hechos = set()
    if dst.is_file():
        for l in open(dst, encoding="utf-8"):
            try:
                hechos.add(json.loads(l)["clip"])
            except Exception:
                pass
    tareas = [(c, df.loc[c, "torneo"], df.loc[c, "weapon"]) for c in lista
              if c in df.index and c not in hechos]
    print("%d clips, %d ya hechos, %d por hacer" % (len(lista), len(hechos), len(tareas)), flush=True)
    n = 0
    t0 = time.time()
    with ProcessPoolExecutor(int(os.environ.get("PROCS", "6"))) as ex, open(dst, "a", encoding="utf-8") as fh:
        futs = [ex.submit(uno, t) for t in tareas]
        for fu in as_completed(futs):
            try:
                r = fu.result()
            except Exception as e:
                print("FALLO", e, flush=True)
                continue
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            n += 1
            if n % 200 == 0:
                print("%d / %d  %.0f s" % (n, len(tareas), time.time() - t0), flush=True)
    print("fin", n, flush=True)


if __name__ == "__main__":
    main()

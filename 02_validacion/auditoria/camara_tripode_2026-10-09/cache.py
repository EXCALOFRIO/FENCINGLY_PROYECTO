"""Cache por clip para el experimento de la camara de tripode: los rasgos de produccion (hasta el
tocado + 3), la calibracion de produccion y su respuesta, y, en una ventana de 2 s antes del tocado,
las aristas (Canny) fuera de personas y rotulo y la mascara valida, a la resolucion de analisis.

    python cache.py LISTA.txt SALIDA_DIR [procs]
"""
import os
import pathlib
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")

VENT_S = 2.0


def uno(clip, torneo, salida):
    import cv2
    import numpy as np
    import pista_lineas as PL
    import rutas as RU
    import zona_tocado as ZT
    f = pathlib.Path(salida) / ("%s.pkl" % clip)
    if f.is_file():
        return clip, "ya"
    t0 = time.perf_counter()
    r = {"clip": clip, "torneo": torneo}
    video = RU.video(clip)
    z = np.load(RU.TRABAJO / ("poses_%s.npz" % clip), allow_pickle=True)
    KP, SC, B, origen = z["KP"], z["SC"], z["B"], z["origen"]
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    fps = float(z["fps"]) if "fps" in z.files else 30.0
    tf, via = ZT.cuadro_tocado(clip, video, torneo)
    r.update(tf=tf, via_luz=via, T=T, fps=fps, W=W, H=H)
    if tf is None or tf >= T:
        r["motivo"] = "sin luz" if tf is None else "fuera"
        with open(f, "wb") as fh:
            pickle.dump(r, fh)
        return clip, r["motivo"]
    k26 = ZT.a_halpe(KP, SC)
    Bpx = B * np.array([W, H, W, H])
    pf = ZT._rasgos(video, W, H, T, tf, Bpx, k26, ZT.BANDA_ROTULO)
    Tn = len(pf.feats)
    pies = np.full((Tn, 2, 2), np.nan, np.float32)
    alt = np.full((Tn, 2), np.nan, np.float32)
    for t in range(Tn):
        for s in (0, 1):
            if origen[t, s] in (0, 1) and np.isfinite(Bpx[t, s]).all():
                pies[t, s] = ZT.apoyo(k26[t, s], Bpx[t, s])
                alt[t, s] = ZT.largo_cuerpo(k26[t, s])
    t1 = time.perf_counter()
    cal = PL.calibrate(pf.feats, pies, alt, ZT.CFG_PISTA, (W, H))
    t2 = time.perf_counter()
    Hs, conf = cal["H"], cal["conf"]
    tq = min(tf, Tn - 1)
    w = int(ZT.VENTANA_S * fps)
    cand = [t for t in range(max(0, tq - w), min(Tn, tq + 2))
            if np.isfinite(Hs[t]).all() and conf[t] >= ZT.CFG_PISTA["min_conf"]]
    prod = {"ok": False}
    if cand:
        tt = min(cand, key=lambda t: abs(t - tq))
        xs = [float(PL.to_m(Hs[tt], pies[tt, s][None])[0, 0]) if np.isfinite(pies[tt, s]).all()
              else None for s in (0, 1)]
        vx = [x for x in xs if x is not None]
        if vx:
            x = float(np.mean(vx))
            prod = {"ok": True, "t": tt, "x": x, "zona": ZT.ZONAS[ZT.zona(x)], "xs": xs,
                    "H": Hs[tt].copy(), "conf": float(conf[tt])}
    paso = max(1, int(round(fps / 15.0)))
    ini = max(0, tq - int(round(VENT_S * fps)))
    ventana = list(range(tq, ini - 1, -paso))[::-1]
    if ventana[-1] != tq:
        ventana.append(tq)
    aristas, validas = {}, {}
    for t in ventana:
        ft = pf.feats[t]
        e = cv2.Canny(cv2.GaussianBlur(ft.gray, (3, 3), 0), 30, 90)
        e[ft.valid == 0] = 0
        aristas[t] = np.packbits(e > 0)
        validas[t] = np.packbits(ft.valid > 0)
    r.update(Tn=Tn, tq=tq, s=pf.feats[0].s, ash=pf.feats[0].gray.shape, pies=pies, alt=alt,
             G=[ft.G for ft in pf.feats], cut=[ft.cut for ft in pf.feats], ventana=ventana,
             aristas=aristas, validas=validas, prod=prod,
             cal_conf=conf.astype("float32"), cal_metodo=list(cal["method"]),
             cal_H={int(t): Hs[t].copy() for t in range(Tn) if np.isfinite(Hs[t]).all()},
             far={t: cal["obs"][t].far for t in ventana}, near={t: cal["obs"][t].near for t in ventana},
             seg={"rasgos": round(t1 - t0, 1), "cal": round(t2 - t1, 1)})
    with open(f, "wb") as fh:
        pickle.dump(r, fh)
    return clip, "ok"


def main():
    lista, salida = sys.argv[1], sys.argv[2]
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    pathlib.Path(salida).mkdir(parents=True, exist_ok=True)
    import catalogo as CA
    df = CA.manifiesto().set_index("clip_id")
    clips = [l.strip() for l in open(lista, encoding="utf-8") if l.strip()]
    tareas = [(c, df.loc[c, "torneo"] if c in df.index else None) for c in clips]
    t0 = time.time()
    with ProcessPoolExecutor(procs) as ex:
        fut = {ex.submit(uno, c, t, salida): c for c, t in tareas}
        for i, fu in enumerate(as_completed(fut), 1):
            try:
                fu.result()
            except Exception as e:
                (pathlib.Path(salida) / ("%s.err" % fut[fu])).write_text(
                    "%s: %s" % (type(e).__name__, e), encoding="utf-8")
            if i % 50 == 0:
                print("%d/%d %.0f s" % (i, len(tareas), time.time() - t0), flush=True)
    print("hecho %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    main()

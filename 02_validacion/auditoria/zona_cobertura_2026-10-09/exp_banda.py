"""Candidata N: rasgos con la franja del rotulo en `b` en vez de 0,78, regla de produccion intacta.

    python exp_banda.py LISTA SALIDA procs b1 [b2 ...]
"""
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
sys.path.insert(0, r"D:\FENCINGLY_SCRATCH\agente_zona")


def uno(clip, torneo, salida, bandas):
    import cv2
    import numpy as np
    import pista_lineas as PL
    import rutas as RU
    import zona_tocado as ZT
    from exp_zona import resumen_ventana
    f = pathlib.Path(salida) / ("%s.json" % clip)
    if f.is_file():
        return clip
    r = {"clip": clip}
    video = RU.video(clip)
    z = np.load(RU.TRABAJO / ("poses_%s.npz" % clip), allow_pickle=True)
    KP, SC, B, origen = z["KP"], z["SC"], z["B"], z["origen"]
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    fps = float(z["fps"]) if "fps" in z.files else 30.0
    tf, via = ZT.cuadro_tocado(clip, video, torneo)
    r.update(tf=tf, T=T)
    if tf is None or tf >= T:
        f.write_text(json.dumps(r), encoding="utf-8")
        return clip
    k26 = ZT.a_halpe(KP, SC)
    Bpx = B * np.array([W, H, W, H])
    Ta = min(T, tf + 3)
    pfs = {b: PL.PisteFeatures(ZT.CFG_VIDEO, (W, H)) for b in bandas}
    cap = cv2.VideoCapture(video)
    for t in range(Ta):
        ok, img = cap.read()
        if not ok:
            break
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H))
        okb = np.isfinite(Bpx[t]).all(1)
        for b, pf in pfs.items():
            pf.process(img, Bpx[t][okb].astype(np.float32), b * H, k26[t][okb])
    cap.release()
    Tn = min(len(pf.feats) for pf in pfs.values())
    pies = np.full((Tn, 2, 2), np.nan, np.float32)
    alt = np.full((Tn, 2), np.nan, np.float32)
    for t in range(Tn):
        for s in (0, 1):
            if origen[t, s] in (0, 1) and np.isfinite(Bpx[t, s]).all():
                pies[t, s] = ZT.apoyo(k26[t, s], Bpx[t, s])
                alt[t, s] = ZT.largo_cuerpo(k26[t, s])
    w = int(ZT.VENTANA_S * fps)
    tq = min(tf, Tn - 1)
    for b, pf in pfs.items():
        cal = PL.calibrate(pf.feats[:Tn], pies, alt, ZT.CFG_PISTA, (W, H))
        r["N%.2f" % b] = resumen_ventana(cal, pies, tq, w, Tn)
    f.write_text(json.dumps(r), encoding="utf-8")
    return clip


def main():
    lista, salida, procs = sys.argv[1], sys.argv[2], int(sys.argv[3])
    bandas = [float(x) for x in sys.argv[4:]]
    pathlib.Path(salida).mkdir(parents=True, exist_ok=True)
    import catalogo as CA
    df = CA.manifiesto().set_index("clip_id")
    clips = [l.strip() for l in open(lista, encoding="utf-8") if l.strip()]
    t0 = time.time()
    with ProcessPoolExecutor(procs) as ex:
        fut = {ex.submit(uno, c, df.loc[c, "torneo"], salida, bandas): c for c in clips}
        for i, fu in enumerate(as_completed(fut), 1):
            try:
                fu.result()
            except Exception as e:
                (pathlib.Path(salida) / ("%s.err" % fut[fu])).write_text(
                    "%s: %s" % (type(e).__name__, e), encoding="utf-8")
            if i % 25 == 0:
                print("%d/%d %.0f s" % (i, len(clips), time.time() - t0), flush=True)
    print("hecho %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    main()

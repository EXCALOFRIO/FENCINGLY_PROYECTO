"""Experimento de cobertura de la etapa 11: una pasada de rasgos hasta el FINAL del clip, dos
calibraciones (hasta tf+3, como la produccion, y hasta el final), y varias reglas de eleccion del
cuadro, todo guardado por clip en JSON para elegir y evaluar sin recalcular.

    python exp_zona.py LISTA.txt SALIDA_DIR [procs]
"""
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")

CMINS = (0.35, 0.25, 0.15, 0.05, 0.0)


def resumen_ventana(cal, pies, tq, w, Tn):
    import numpy as np
    import pista_lineas as PL
    import zona_tocado as ZT
    Hs, conf, meth = cal["H"], cal["conf"], cal["method"]
    fin = np.isfinite(Hs).all(axis=(1, 2))
    lo, hi = max(0, tq - w), min(Tn, tq + 2)
    out = {"fin_en_ventana": int(fin[lo:hi].sum()),
           "conf_max_ventana": float(conf[lo:hi].max()) if hi > lo else 0.0,
           "fin_tq": bool(fin[min(tq, Tn - 1)]),
           "conf_tq": float(conf[min(tq, Tn - 1)]),
           "n_fin": int(fin.sum()),
           "n_lines": int(sum(1 for m in meth if m == "lines")),
           "anclas": len(cal.get("anchors") or [])}
    idx = np.nonzero(fin)[0]
    out["dist_fin_mas_cerca"] = int(np.abs(idx - tq).min()) if len(idx) else None
    reglas = {}
    for cmin in CMINS:
        cand = [t for t in range(lo, hi) if fin[t] and conf[t] >= cmin]
        if not cand:
            reglas[str(cmin)] = None
            continue
        tt = min(cand, key=lambda t: abs(t - tq))
        xs = [float(PL.to_m(Hs[tt], pies[tt, s][None])[0, 0]) if np.isfinite(pies[tt, s]).all()
              else None for s in (0, 1)]
        vx = [x for x in xs if x is not None]
        if not vx:
            reglas[str(cmin)] = {"sin_pies": True}
            continue
        x = float(np.mean(vx))
        reglas[str(cmin)] = {"t": int(tt), "x": x, "zona": ZT.ZONAS[ZT.zona(x)],
                             "conf": float(conf[tt]), "metodo": str(meth[tt]),
                             "x_izq": xs[0], "x_der": xs[1]}
    out["reglas"] = reglas
    return out


def uno(clip, torneo, salida):
    import cv2
    import numpy as np
    import pista_lineas as PL
    import rutas as RU
    import zona_tocado as ZT
    f = pathlib.Path(salida) / ("%s.json" % clip)
    if f.is_file():
        return clip, "ya"
    t0 = time.perf_counter()
    r = {"clip": clip}
    video = RU.video(clip)
    z = np.load(RU.TRABAJO / ("poses_%s.npz" % clip), allow_pickle=True)
    KP, SC, B, origen = z["KP"], z["SC"], z["B"], z["origen"]
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    fps = float(z["fps"]) if "fps" in z.files else 30.0
    tf, via = ZT.cuadro_tocado(clip, video, torneo)
    r.update(tf=tf, via_luz=via, T=T, fps=fps, W=W, H=H)
    if tf is None or tf >= T:
        r["motivo"] = "sin luz" if tf is None else "fuera"
        f.write_text(json.dumps(r), encoding="utf-8")
        return clip, r["motivo"]
    k26 = ZT.a_halpe(KP, SC)
    Bpx = B * np.array([W, H, W, H])
    cap = cv2.VideoCapture(video)
    pf = PL.PisteFeatures(ZT.CFG_VIDEO, (W, H))
    for t in range(T):
        ok, img = cap.read()
        if not ok:
            break
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H))
        okb = np.isfinite(Bpx[t]).all(1)
        pf.process(img, Bpx[t][okb].astype(np.float32), ZT.BANDA_ROTULO * H, k26[t][okb])
    cap.release()
    t1 = time.perf_counter()
    Tall = len(pf.feats)
    pies = np.full((Tall, 2, 2), np.nan, np.float32)
    alt = np.full((Tall, 2), np.nan, np.float32)
    for t in range(Tall):
        for s in (0, 1):
            if origen[t, s] in (0, 1) and np.isfinite(Bpx[t, s]).all():
                pies[t, s] = ZT.apoyo(k26[t, s], Bpx[t, s])
                alt[t, s] = ZT.largo_cuerpo(k26[t, s])
    Ta = min(Tall, tf + 3)
    w = int(ZT.VENTANA_S * fps)
    tq_a = min(tf, Ta - 1)
    cal_a = PL.calibrate(pf.feats[:Ta], pies[:Ta], alt[:Ta], ZT.CFG_PISTA, (W, H))
    t2 = time.perf_counter()
    r["a"] = resumen_ventana(cal_a, pies, tq_a, w, Ta)
    cal_b = PL.calibrate(pf.feats, pies, alt, ZT.CFG_PISTA, (W, H))
    t3 = time.perf_counter()
    tq_b = min(tf, Tall - 1)
    r["b"] = resumen_ventana(cal_b, pies, tq_b, w, Tall)
    # caracterizacion barata alrededor del tocado
    lo, hi = max(0, tq_a - w), min(Ta, tq_a + 2)
    obs = cal_a["obs"]
    r["car"] = {
        "cortes_en_ventana": int(sum(pf.feats[t].cut for t in range(lo, hi))),
        "G_none_en_ventana": int(sum(pf.feats[t].G is None for t in range(max(1, lo), hi))),
        "bordes_ambos_ventana": int(sum(obs[t].far is not None and obs[t].near is not None
                                        for t in range(lo, hi))),
        "trans2_ventana": int(sum(bool(obs[t].trans) and len(obs[t].trans) >= 2
                                  for t in range(lo, hi))),
        "bordes_ambos_clip": int(sum(o.far is not None and o.near is not None for o in obs)),
        "trans2_clip": int(sum(bool(o.trans) and len(o.trans) >= 2 for o in obs)),
        "pies_tq": int(np.isfinite(pies[tq_a]).all(1).sum()),
        "n_ventana": hi - lo,
        "cortes_clip_hasta_tf": int(sum(ff.cut for ff in pf.feats[:Ta])),
        "ultimo_corte_antes_tf": max([t for t in range(Ta) if pf.feats[t].cut], default=None),
    }
    r["s"] = {"rasgos": round(t1 - t0, 1), "cal_a": round(t2 - t1, 1), "cal_b": round(t3 - t2, 1)}
    f.write_text(json.dumps(r), encoding="utf-8")
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

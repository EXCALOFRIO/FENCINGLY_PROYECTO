"""Rasgos baratos por clip para la regla del angulo del borde lejano (RAYAS_ID_2026-10-09).

Por clip con strip_location y zona_ (ok o sin homografia): cuadro t (cuadro_usado si ok; si no,
min(cuadro_tocado, T-1)), decodifica t-2..t, rasgos de pista_lineas y observe con los pies.
    python rasgos.py SALIDA.jsonl [procs]
"""
import csv, json, os, pathlib, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")


def _ang(l):
    import pista_lineas as PL
    if l is None:
        return None
    a = float(PL._ang(l))
    return a if a < 90 else a - 180


def uno(clip, torneo):
    import cv2
    import numpy as np
    import pista_lineas as PL
    import rutas as RU
    import zona_tocado as ZT
    r = {"clip": clip, "torneo": torneo}
    zr = next(csv.DictReader(open(RU.TRABAJO / ("zona_%s.csv" % clip), encoding="utf-8")), {})
    mot = zr.get("motivo") or ""
    r["motivo"] = "ok" if mot == "ok" else ("sinH" if mot.startswith("sin homografia") else mot)
    if r["motivo"] not in ("ok", "sinH"):
        return r
    r["x_prod"] = float(zr["x_tocado_m"]) if r["motivo"] == "ok" else None
    r["zona_prod"] = zr.get("zona") or None
    z = np.load(RU.TRABAJO / ("poses_%s.npz" % clip), allow_pickle=True)
    KP, SC, B, origen = z["KP"], z["SC"], z["B"], z["origen"]
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    t = int(float(zr["cuadro_usado"])) if r["motivo"] == "ok" else min(int(float(zr["cuadro_tocado"])), T - 1)
    r.update(W=W, H=H, t=t)
    k26 = ZT.a_halpe(KP, SC)
    Bpx = B * np.array([W, H, W, H])
    video = RU.video(clip)
    cap = cv2.VideoCapture(video)
    pf = PL.PisteFeatures(ZT.CFG_VIDEO, (W, H))
    fr = []
    k = 0
    while k <= t:
        if k < t - 2:
            ok = cap.grab()
            if not ok:
                break
            k += 1
            continue
        ok, img = cap.read()
        if not ok:
            break
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H))
        okb = np.isfinite(Bpx[k]).all(1)
        feat = pf.process(img, Bpx[k][okb].astype(np.float32), ZT.BANDA_ROTULO * H, k26[k][okb])
        pies = np.full((2, 2), np.nan, np.float32)
        alt = np.full(2, np.nan, np.float32)
        for s in (0, 1):
            if origen[k, s] in (0, 1) and np.isfinite(Bpx[k, s]).all():
                pies[s] = ZT.apoyo(k26[k, s], Bpx[k, s])
                alt[s] = ZT.largo_cuerpo(k26[k, s])
        ob = PL.observe(feat, pies, alt, (W, H))
        far_y = None
        if ob.far is not None:
            far_y = float(PL._y_at(ob.far, W / 2.0)) / H
        fr.append({"k": k, "far": _ang(ob.far), "near": _ang(ob.near), "far_y": far_y,
                   "ntrans": len(ob.trans or []),
                   "trans_u": [float((a[0] + b[0]) / 2) / W for a, b in (ob.trans or [])],
                   "dashed": [bool(d) for d in (ob.dashed or [])],
                   "zch": [float(v) for v in (ob.zchange or [])],
                   "pies_u": [None if not np.isfinite(p).all() else float(p[0]) / W for p in pies],
                   "pies_v": [None if not np.isfinite(p).all() else float(p[1]) / H for p in pies],
                   "alt": [None if not np.isfinite(a) else float(a) / H for a in alt]})
        k += 1
    cap.release()
    r["fr"] = fr
    return r


def main():
    sal = sys.argv[1]
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    import catalogo as CA
    import rutas as RU
    df = CA.manifiesto()
    df = df[df["strip_location"].isin(CA.ZONAS)]
    hechos = set()
    if os.path.isfile(sal):
        for l in open(sal, encoding="utf-8"):
            try:
                hechos.add(json.loads(l)["clip"])
            except Exception:
                pass
    tareas = [(m["clip_id"], m["torneo"]) for _, m in df.iterrows()
              if (RU.TRABAJO / ("zona_%s.csv" % m["clip_id"])).is_file() and m["clip_id"] not in hechos]
    print("%d clips por hacer" % len(tareas), flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(procs) as ex, open(sal, "a", encoding="utf-8") as fh:
        fut = {ex.submit(uno, c, t): c for c, t in tareas}
        for i, f in enumerate(as_completed(fut), 1):
            try:
                r = f.result()
            except Exception as e:
                r = {"clip": fut[f], "error": "%s: %s" % (type(e).__name__, str(e)[:200])}
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if i % 200 == 0:
                print("%d/%d %.0f s" % (i, len(tareas), time.time() - t0), flush=True)
    print("hecho %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

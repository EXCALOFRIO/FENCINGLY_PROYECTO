"""Que tan fiables son las seniales de aspecto de cada raya (discontinua -> central; salto de color ->
+-5/+-7), en las referencias de AJUSTE con homografia de produccion (cache del agente F)."""
import pickle, sys, json, numpy as np, collections, csv
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import cv2, pista_lineas as PL, zona_tocado as ZT, rutas as RU
aj = [l.strip() for l in open(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\auditoria\camara_tripode_2026-10-09\lista_cache_ajuste.txt", encoding="utf-8") if l.strip()]
filas = []
for c in aj:
    try: r = pickle.load(open("D:/FENCINGLY_SCRATCH/agente_tripode/cache/%s.pkl" % c, "rb"))
    except Exception: continue
    p = r.get("prod", {})
    if not p.get("ok"): continue
    t = p["t"]; Hh = p["H"]
    z = np.load(RU.TRABAJO / ("poses_%s.npz" % c), allow_pickle=True)
    KP, SC, B, origen = z["KP"], z["SC"], z["B"], z["origen"]
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    k26 = ZT.a_halpe(KP, SC); Bpx = B * np.array([W, H, W, H])
    cap = cv2.VideoCapture(RU.video(c)); pf = PL.PisteFeatures(ZT.CFG_VIDEO, (W, H))
    for k in range(t): cap.grab()
    ok, img = cap.read(); cap.release()
    if not ok: continue
    okb = np.isfinite(Bpx[t]).all(1)
    feat = pf.process(img, Bpx[t][okb].astype(np.float32), ZT.BANDA_ROTULO * H, k26[t][okb])
    pies = np.full((2, 2), np.nan, np.float32); alt = np.full(2, np.nan, np.float32)
    for s in (0, 1):
        if origen[t, s] in (0, 1) and np.isfinite(Bpx[t, s]).all():
            pies[s] = ZT.apoyo(k26[t, s], Bpx[t, s]); alt[s] = ZT.largo_cuerpo(k26[t, s])
    ob = PL.observe(feat, pies, alt, (W, H))
    for (a, b), d, zc in zip(ob.trans or [], ob.dashed or [], ob.zchange or []):
        mid = PL.to_m(Hh, ((a + b) / 2)[None])[0]
        known = np.array(ZT.CFG_PISTA["lines_m"])
        j = int(np.argmin(abs(known - mid[0])))
        filas.append({"clip": c, "x": float(mid[0]), "linea": float(known[j]), "dist": float(abs(known[j] - mid[0])), "dashed": bool(d), "zch": float(zc)})
json.dump(filas, open("fiabilidad_rayas.json", "w"))
F = [f for f in filas if f["dist"] < 0.5]
print("rayas observadas e identificadas por la H de produccion (a <0,5 m de una raya):", len(F), "de", len(filas))
cnt = collections.Counter((f["linea"], f["dashed"]) for f in F)
for L in sorted(set(f["linea"] for f in F)):
    n = sum(1 for f in F if f["linea"] == L); d = cnt[(L, True)]
    zs = [f["zch"] for f in F if f["linea"] == L]
    print("raya %+5.1f  n=%3d  discontinua %.2f   salto de color mediana %.1f  (>25: %.2f)" % (L, n, d / max(n, 1), np.median(zs), np.mean(np.array(zs) > 25)))

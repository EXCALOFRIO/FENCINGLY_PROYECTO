import csv, pathlib, sys, collections
import numpy as np
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import pista_lineas as PL
por_tipo = collections.defaultdict(list)
for e in ("e_1014_0011", "e_1014_1808"):
    E = pathlib.Path(r"D:\FENCINGLY_SCRATCH\ensayo") / e
    for c in open(E / "clips.txt", encoding="utf-8").read().split():
        ph = E / "trabajo" / ("pista_h_%s.npz" % c); se = E / "pista" / ("serie_eventos_%s.csv" % c)
        if not (ph.is_file() and se.is_file()):
            continue
        z = np.load(ph, allow_pickle=True); H, conf, pies, mc = z["H"], z["conf"], z["pies"], float(z["min_conf"])
        def x11(t, s):
            if t >= len(H) or not (np.isfinite(H[t]).all() and conf[t] >= mc and np.isfinite(pies[t, s]).all()):
                return None
            return float(PL.to_m(H[t], pies[t, s][None])[0, 0])
        for r in csv.DictReader(open(se, encoding="utf-8")):
            if r["lado"] not in ("0", "1") or not r.get("x_inicio_m") or not r.get("x_fin_m"):
                continue
            s = int(r["lado"]); a, b = int(r["cuadro_inicio"]), int(r["cuadro_fin"])
            u, v = x11(a, s), x11(b, s)
            if u is None or v is None:
                continue
            por_tipo[r["tipo"]].append((float(r["x_fin_m"]) - float(r["x_inicio_m"]), v - u, r.get("publicable") == "True"))
for t, L in sorted(por_tipo.items(), key=lambda x: -len(x[1])):
    a = np.array([(x, y) for x, y, _ in L])
    d = np.abs(a[:, 0] - a[:, 1])
    sg = np.mean(np.sign(a[:, 0]) == np.sign(a[:, 1]))
    print("%-8s n=%4d | desplazamiento etapa 7 p50 %.2f m, etapa 11 p50 %.2f m | |diferencia| p50 %.2f p90 %.2f m | mismo sentido %.0f %% | corr %.2f"
          % (t, len(a), np.median(np.abs(a[:, 0])), np.median(np.abs(a[:, 1])), np.median(d), np.percentile(d, 90), 100 * sg,
             np.corrcoef(a[:, 0], a[:, 1])[0, 1] if len(a) > 2 else float("nan")))

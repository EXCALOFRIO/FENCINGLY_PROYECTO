import csv, pathlib, sys
import numpy as np
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import pista_lineas as PL
d, por_clip, n_ev_sin7 = [], [], 0
pares = []
for e in ("e_1014_0011", "e_1014_1808"):
    E = pathlib.Path(r"D:\FENCINGLY_SCRATCH\ensayo") / e
    for c in open(E / "clips.txt", encoding="utf-8").read().split():
        ph = E / "trabajo" / ("pista_h_%s.npz" % c); se = E / "pista" / ("serie_eventos_%s.csv" % c)
        pc = E / "pista" / ("posicion_clip_%s.csv" % c)
        if not (ph.is_file() and se.is_file() and pc.is_file()):
            continue
        sin7 = list(csv.DictReader(open(pc, encoding="utf-8")))[0]["sin_homografia"] == "True"
        z = np.load(ph, allow_pickle=True); H, conf, pies, mc = z["H"], z["conf"], z["pies"], float(z["min_conf"])
        dc = []
        for r in csv.DictReader(open(se, encoding="utf-8")):
            if r["lado"] not in ("0", "1") or not r.get("x_inicio_m"):
                continue
            t, s = int(r["cuadro_inicio"]), int(r["lado"])
            if t >= len(H) or not (np.isfinite(H[t]).all() and conf[t] >= mc and np.isfinite(pies[t, s]).all()):
                continue
            x11 = float(PL.to_m(H[t], pies[t, s][None])[0, 0]); x7 = float(r["x_inicio_m"])
            pares.append((x7, x11)); dc.append(abs(x7 - x11))
        if dc:
            por_clip.append((c, len(dc), float(np.median(dc))))
p = np.array(pares)
print("eventos con x en las dos:", len(p), "en", len(por_clip), "clips")
dif = np.abs(p[:, 0] - p[:, 1]); difs = np.abs(p[:, 0] + p[:, 1])
print("|x7 - x11|: p50 %.2f m  p90 %.2f m   | con el signo cambiado |x7 + x11|: p50 %.2f" % (np.median(dif), np.percentile(dif, 90), np.median(difs)))
print("correlacion", np.corrcoef(p[:, 0], p[:, 1])[0, 1])
print("por clip (mediana de |dx|):", sorted(round(m, 2) for _, _, m in por_clip))
print()
for nom, v in (("etapa 7", p[:, 0]), ("etapa 11", p[:, 1])):
    print("%-9s fuera de [-7,5, 7,5] m: %5.1f %%  | p5 %.1f  p50 %.1f  p95 %.1f" % (nom, 100 * np.mean(np.abs(v) > 7.5), *np.percentile(v, [5, 50, 95])))
# coherencia interna: distancia entre los dos tiradores en el mismo cuadro (en guardia ~4 m, nunca > 14)
for e in ("e_1014_0011",):
    E = pathlib.Path(r"D:\FENCINGLY_SCRATCH\ensayo") / e
    dist = []
    for ph in (E / "trabajo").glob("pista_h_*.npz"):
        z = np.load(ph, allow_pickle=True); H, conf, pies, mc = z["H"], z["conf"], z["pies"], float(z["min_conf"])
        for t in range(0, len(H), 5):
            if np.isfinite(H[t]).all() and conf[t] >= mc and np.isfinite(pies[t]).all():
                x = PL.to_m(H[t], pies[t])[:, 0]
                dist.append(x[1] - x[0])
    dist = np.array(dist)
    print("etapa 11: separacion der - izq (m), %d cuadros: p5 %.2f p50 %.2f p95 %.2f; negativa %.1f %%; > 14 m %.1f %%"
          % (len(dist), *np.percentile(dist, [5, 50, 95]), 100 * np.mean(dist < 0), 100 * np.mean(np.abs(dist) > 14)))

"""Re-anclaje: la x de la etapa 7 (relativa a su ancla) contra la x de v2 (absoluta) en los mismos cuadros
(inicio y fin de cada evento con dato en las dos). Por clip, recta robusta x_v2 = a * x7 + b (Theil-Sen)
y el residuo. Si a ~ 1 y el residuo es pequenio, basta con mover el origen de la 7 con v2."""
import csv, pathlib, sys
import numpy as np
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import scouting_v2 as SV
S = pathlib.Path(r"C:\Users\Alejandro\FENCINGLY_TMP\scout_h\trabajo")
REAL_P = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\pista")
orden = open(r"D:\FENCINGLY_SCRATCH\scout\clips_h.txt", encoding="utf-8").read().split()
pend, resid, nclips, offs = [], [], 0, []
for c in orden:
    if not (S / ("pista_h_%s.npz" % c)).is_file():
        continue
    pc = REAL_P / ("posicion_clip_%s.csv" % c); se = REAL_P / ("serie_eventos_%s.csv" % c)
    if not (pc.is_file() and se.is_file()) or list(csv.DictReader(open(pc, encoding="utf-8")))[0]["sin_homografia"] == "True":
        continue
    d = SV.posiciones(c, S)
    if d is None:
        continue
    pares = []
    for r in csv.DictReader(open(se, encoding="utf-8")):
        if r["lado"] not in ("0", "1") or not r.get("x_inicio_m") or not r.get("x_fin_m"):
            continue
        s = int(r["lado"])
        for k, x in ((int(r["cuadro_inicio"]), float(r["x_inicio_m"])), (int(r["cuadro_fin"]), float(r["x_fin_m"]))):
            if k < len(d["x"]) and d["estado"][k, s] >= 0:
                pares.append((x, float(d["x"][k, s])))
    if len(pares) < 8:
        continue
    P = np.array(pares)
    i, j = np.triu_indices(len(P), 1)
    dx = P[j, 0] - P[i, 0]; ok = np.abs(dx) > 0.2
    if ok.sum() < 5:
        continue
    a = float(np.median((P[j, 1] - P[i, 1])[ok] / dx[ok])); b = float(np.median(P[:, 1] - a * P[:, 0]))
    nclips += 1; pend.append(a); resid.append(float(np.median(np.abs(P[:, 1] - (a * P[:, 0] + b)))))
    offs.append(float(np.median(np.abs(P[:, 1] - P[:, 0] - np.median(P[:, 1] - P[:, 0])))))
pend, resid, offs = map(np.array, (pend, resid, offs))
print("clips con la 7 con H y >= 8 pares:", nclips)
print("pendiente a: p10 %.2f p25 %.2f p50 %.2f p75 %.2f p90 %.2f | en [0,8, 1,25]: %.0f %% | negativa: %.0f %%" % (
    *np.percentile(pend, [10, 25, 50, 75, 90]), 100 * np.mean((pend >= 0.8) & (pend <= 1.25)), 100 * np.mean(pend < 0)))
print("residuo de la recta |x_v2 - (a x7 + b)| mediana por clip: p50 %.2f p90 %.2f m" % tuple(np.percentile(resid, [50, 90])))
print("solo desplazar (a = 1): residuo p50 %.2f p90 %.2f m; clips con residuo <= 0,25 m: %.0f %%" % (
    *np.percentile(offs, [50, 90]), 100 * np.mean(offs <= 0.25)))

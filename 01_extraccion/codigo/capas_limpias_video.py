"""Video mosaico por clip para ver A OJO si en las capas densas queda arbitro, publico o fondo.

Seis celdas de cuadro completo: original con las cajas, `matting` limpio, `seg` limpio, `normal`
limpio, `pointmap` limpio y el recorte final en color. Todo lo que este fuera de las dos mascaras
sale NEGRO, asi que cualquier resto de arbitro o de suelo se ve al instante.

`pointmap` se corre de CUADRO COMPLETO, no de recorte: dentro de un recorte su z no ordena
profundidad (el control del suelo salio 53 %, moneda al aire) y de cuadro completo si (7 de 7). El
motivo esta en el docstring de `capas_limpias.py`.

Al terminar imprime la tabla del criterio para los cuadros procesados, con el control negativo.
"""
from __future__ import annotations

import json
import os
import pathlib
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import capas_limpias as CL

import cv2

SCR = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\187cfd83-d8d9-4c4d-bed4-f255724f6b21\scratchpad")
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\capas")
SALIDA.mkdir(parents=True, exist_ok=True)
CLIP = sys.argv[1] if len(sys.argv) > 1 else "defeatedHarlequinRabbitSkinkLizard"
PASO = int(sys.argv[2]) if len(sys.argv) > 2 else 4
CELDA = 640

mapa = {}
for raiz, _d, fs in os.walk("D:/FENCINGLY_VIDEOS_RAW"):
    for f in fs:
        if f.lower().endswith(".mp4"):
            mapa.setdefault(pathlib.Path(f).stem, os.path.join(raiz, f))

with open(SCR / ("det_%s.pkl" % CLIP), "rb") as fh:
    cajas, cuadros, W, H, T = pickle.load(fh)
"""La puerta de FUNDIDOS de `extrae_pareja.procesa` necesita los cuadros en gris a 160x90, asi que
se decodifica el clip entero una vez en pequeno antes de nada. Sin `gris` no se aplica esa puerta y
se procesarian cuadros de transicion, que es justo donde `matting` y `seg` fallan."""
gris = []
_v = cv2.VideoCapture(mapa[CLIP])
while True:
    _ok, _im = _v.read()
    if not _ok:
        break
    gris.append(cv2.cvtColor(cv2.resize(_im, (160, 90)), cv2.COLOR_BGR2GRAY))
_v.release()
print("decodificados %d cuadros en gris para la puerta de fundidos" % len(gris))
sel = CL.elige_pareja_del_clip(cajas, cuadros, W, H, T, gris=gris)
B, pres, msk = sel["B"], sel["pres"], sel["mask"]
KS = [k for k in range(0, T, PASO) if msk[k] and pres[k].all()]
print("cuadros utilizables tras la cadena entera: %d de %d" % (int(msk.sum()), T))
print("clip %s  %dx%d T=%d   cuadros a procesar %d (paso %d)" % (CLIP, W, H, T, len(KS), PASO))

cap = CL.Capas(["pose", "matting", "seg", "normal", "pointmap"])
print("modelos cargados")

vid = cv2.VideoCapture(mapa[CLIP])
esc = CELDA / float(W)
hc = int(round(H * esc))
mos = cv2.VideoWriter(str(SALIDA / ("mosaico_%s.mp4" % CLIP)),
                      cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (CELDA * 3, hc * 2))
filas = []
cob_cuadro = []
t0 = time.perf_counter()
for i, k in enumerate(KS):
    vid.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, im = vid.read()
    if not ok:
        continue
    rgb = np.ascontiguousarray(im[:, :, ::-1])
    aj = CL.ajenos_del_cuadro(sel, k, H)
    cbs = [B[k, 0] * np.array([W, H, W, H]), B[k, 1] * np.array([W, H, W, H])] + aj
    poses = cap.pose(rgb, cbs)
    kp_todos = {("t", 0): poses[0], ("t", 1): poses[1]}
    for j in range(len(aj)):
        kp_todos[("a", j)] = poses[2 + j]

    """`pointmap` una vez, de cuadro completo: asi los dos tiradores comparten escala."""
    pm = cap.pointmap(rgb)
    Z = pm[:, :, 2] if pm.ndim == 3 else np.zeros((H, W), "float32")

    M_full = np.zeros((H, W), bool)
    cel_mat = np.zeros((H, W), "float32")
    cel_seg = np.zeros((H, W), "uint8")
    cel_nor = np.zeros((H, W, 3), "float32")
    for lado in (0, 1):
        r = CL.mascara_tirador(cap, rgb, sel, k, lado, W, H, kp_todos, con_matting=True)
        if r is None:
            continue
        c, mk = r["c"], r["mask"]
        M_full[c[1]:c[3], c[0]:c[2]] |= mk
        cel_mat[c[1]:c[3], c[0]:c[2]][mk] = r["alfa"][mk]
        cel_seg[c[1]:c[3], c[0]:c[2]][mk] = r["seg"][mk]
        n = cap.normal(r["sub"])
        if n.ndim == 3 and n.shape[:2] == mk.shape:
            cel_nor[c[1]:c[3], c[0]:c[2]][mk] = n[mk]
        filas.append(CL.mide(r, sel, k, lado, W, H, kp_todos[("t", lado)], bgr=im))
    cob_cuadro.append(float(M_full.mean()))

    c1 = im.copy()
    for lado in (0, 1):
        b = (B[k, lado] * np.array([W, H, W, H])).astype(int)
        cv2.rectangle(c1, (b[0], b[1]), (b[2], b[3]), (80, 255, 80), 2)
    for o in aj:
        o = np.asarray(o).astype(int)
        cv2.rectangle(c1, (o[0], o[1]), (o[2], o[3]), (70, 70, 255), 2)
    c2 = cv2.cvtColor((np.clip(cel_mat, 0, 1) * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
    c3 = CL.pinta_seg(cel_seg)
    c4 = CL.pinta_normal(cel_nor)
    c4[~M_full] = 0
    c5 = CL.pinta_z(Z, M_full)
    c6 = np.zeros_like(im)
    c6[M_full] = im[M_full]
    celdas, etqs = [c1, c2, c3, c4, c5, c6], [
        "original + cajas (verde = los dos, rojo = no elegidos)", "matting limpio", "seg limpio",
        "normal limpio", "pointmap limpio (z de cuadro completo)", "recorte final"]
    vis = []
    for cc, e in zip(celdas, etqs):
        cc = cv2.resize(cc, (CELDA, hc))
        cv2.rectangle(cc, (0, 0), (CELDA, 22), (14, 14, 18), -1)
        cv2.putText(cc, e, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1)
        cv2.putText(cc, "c%04d" % k, (CELDA - 62, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (140, 220, 255), 1)
        vis.append(cc)
    mos.write(np.vstack([np.hstack(vis[:3]), np.hstack(vis[3:])]))
    if i % 20 == 0:
        print("  c%04d  %d/%d  %.1f s/cuadro" % (k, i + 1, len(KS),
                                                 (time.perf_counter() - t0) / (i + 1)))
mos.release()
vid.release()

import pandas as pd
d = pd.DataFrame(filas)
"""BANDERA DE VALIDEZ. El fallo que se vio a ojo en c144 lado0 (recall 0,099) no es fuga, es PERDIDA:
`seg` y `matting` no encuentran al tirador de blanco corriendo sobre suelo claro.

PRIMERA VERSION, MEDIDA Y DESCARTADA: marcar por AREA, residuo < 0,35 x la mediana del propio lado.
Con paso 4 parecia funcionar (rec_kp 0,991 en los validos contra 0,368 en los marcados), pero con
paso 5 sobre el mismo clip MARCO COMO VALIDO el peor caso de todos, c145 lado0 con rec_kp 0,108 y
residuo 0,104. El area no distingue "no ha encontrado a nadie" de "ha encontrado a la persona
equivocada", y el segundo es justo el fallo de `matting`. Una bandera que depende del muestreo de
cuadros no es una bandera.

VERSION QUE SE USA: marcar por `rec_kp`, el acuerdo entre la mascara y la pose. Son dos modelos
independientes, no hace falta ninguna verdad externa, y la pose ya se calcula. Umbral 0,70 declarado
antes de mirar."""
REC_MIN_VALIDO = 0.70
d["valido"] = d["rec_kp"] >= REC_MIN_VALIDO
d.to_csv(SALIDA / ("medidas_%s.csv" % CLIP), index=False)
con = d[d["px_banda_arb"] > 2000]
print()
print("=" * 96)
print("CRITERIO sobre %d cuadros x 2 lados de %s" % (d["k"].nunique(), CLIP))
print("=" * 96)
print("  rec_kp      %.3f   (bueno >= 0,90)" % d["rec_kp"].mean())
print("  rec_desp    %.3f   CONTROL NEGATIVO: la misma mascara media anchura a la derecha" % d["rec_desp"].mean())
print("  fuga_arb    %.4f  (bueno <= 0,05)   sobre %d lados con ajeno en la banda" % (
    d["fuga_arb"].mean(), int(d["fuga_arb"].notna().sum())))
print("  fuga_fondo  %.4f" % d["fuga_fondo"].mean())
print("  fuga_solape %.4f  (ritmo con que conserva el oscuro del solape de cajas, %d lados);"
      "  nivel de referencia con el oscuro que SI es suyo %.4f" % (
          d["fuga_solape"].mean(), int(d["fuga_solape"].notna().sum()),
          d["ret_oscuro_propio"].mean()))
print("  resto_arb   %.4f  fraccion de la mascara final que es oscuro dentro de la caja de un ajeno"
      % d["resto_arb_frac"].mean())
print("  residuo     %.3f   contra 0,657 de la caja sola" % d["residuo"].mean())
print("  cobertura del cuadro por las dos mascaras: %.2f %%  -> guardar cuadro completo gasta %.0fx"
      % (100 * np.mean(cob_cuadro), 1.0 / max(np.mean(cob_cuadro), 1e-9)))
if len(con):
    print()
    print("  SOLO los %d lados con un ajeno de mas de 2000 px en la banda de margen:" % len(con))
    print("    rec_kp %.3f   rec_desp %.3f   fuga_arb %.4f   fuga_fondo %.4f   residuo %.3f" % (
        con["rec_kp"].mean(), con["rec_desp"].mean(), con["fuga_arb"].mean(),
        con["fuga_fondo"].mean(), con["residuo"].mean()))
v, nv = d[d["valido"]], d[~d["valido"]]
print()
print("  BANDERA DE VALIDEZ (rec_kp < %.2f): marca %d de %d lados" % (
    REC_MIN_VALIDO, len(nv), len(d)))
print("    rec_kp en los validos   %.3f" % v["rec_kp"].mean())
print("    rec_kp en los marcados  %s" % ("%.3f" % nv["rec_kp"].mean() if len(nv) else "n/a"))
print("    residuo en los validos %.3f   en los marcados %s" % (
    v["residuo"].mean(), "%.3f" % nv["residuo"].mean() if len(nv) else "n/a"))
print("    fuga_arb en los validos %.4f   (la bandera no debe estar tapando fuga)" % v["fuga_arb"].mean())
peor = d.nsmallest(5, "rec_kp")[["k", "lado", "rec_kp", "fuga_arb", "residuo", "valido"]]
print()
print("  los 5 peores por recall (para ir a mirarlos):")
print(peor.to_string(index=False))
with open(SALIDA / ("resumen_%s.json" % CLIP), "w") as fh:
    json.dump({"clip": CLIP, "n_cuadros": int(d["k"].nunique()),
               "rec_kp": float(d["rec_kp"].mean()), "rec_desp": float(d["rec_desp"].mean()),
               "fuga_arb": float(d["fuga_arb"].mean()), "fuga_fondo": float(d["fuga_fondo"].mean()),
               "residuo": float(d["residuo"].mean()),
               "fuga_solape": float(d["fuga_solape"].mean()),
               "ret_oscuro_propio": float(d["ret_oscuro_propio"].mean()),
               "resto_arb_frac": float(d["resto_arb_frac"].mean()),
               "cobertura_cuadro": float(np.mean(cob_cuadro)),
               "n_marcados_no_validos": int(len(nv)), "n_lados": int(len(d)),
               "rec_kp_validos": float(v["rec_kp"].mean()),
               "rec_kp_marcados": float(nv["rec_kp"].mean()) if len(nv) else None}, fh, indent=1)
print()
print("video en %s" % (SALIDA / ("mosaico_%s.mp4" % CLIP)))

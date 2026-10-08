"""Prueba el selector por trayectorias en 1 clip y PINTA todos los candidatos con su puntuacion.

QUE HAY QUE PODER VER EN LA IMAGEN: no solo a quien eligio, sino **a quien descarto y por que**. Un
selector que solo ensena su respuesta no se puede contradecir. Por eso se pintan TODAS las
trayectorias con sus rasgos, las elegidas en verde y las descartadas en gris.

CRITERIO DE EXITO, declarado antes de mirar:
  1. presencia: los dos elegidos tienen caja en >= 95 % de los cuadros
  2. ningun cambio de identidad: cada elegido es UNA trayectoria, no un pegado de varias
  3. y a ojo: los dos recuadros verdes estan sobre los dos tiradores en los cuadros que se pinten
"""
from __future__ import annotations

import os
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import a3_selector as SEL

PRUEBAS = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\selector")
SALIDA.mkdir(parents=True, exist_ok=True)

CLIP = sys.argv[1] if len(sys.argv) > 1 else "SoreEasygoingEmperorshrimp"
N_PINTA = int(sys.argv[2]) if len(sys.argv) > 2 else 6

f = PRUEBAS / ("candidatos_%s.npz" % CLIP)
if not f.is_file():
    raise SystemExit("falta %s: corre antes a3_paso1_candidatos.py %s" % (f.name, CLIP))
with np.load(f, allow_pickle=True) as z:
    cajas = z["cajas"]
    confs = z["confs"]
    cuadros = z["cuadros"]
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    video = str(z["video"])

print("=" * 92)
print("SELECTOR POR TRAYECTORIAS  ·  %s" % CLIP)
print("=" * 92)
print("  %d detecciones en %d cuadros de %dx%d" % (len(cajas), T, W, H))

tr = SEL.construye_trayectorias(cajas, cuadros, W, H, T)
print("  trayectorias construidas ... %d" % len(tr))
res = SEL.elige_dos(tr, W, H, T)
print("  motivo de la eleccion ...... %s" % res["motivo"])
print()
print("  %-4s %8s %7s %7s %8s %7s %8s %7s %7s %s" % (
    "id", "puntos", "dur", "cobert", "en_banda", "mov", "alto", "cx", "n", "estado"))
elegidas_ids = {id(r["t"]) for r in res["elegidas"]}
for i, r in enumerate(res["candidatos"][:12]):
    est = "ELEGIDA" if id(r["t"]) in elegidas_ids else ""
    print("  %-4d %8.3f %7.3f %7.3f %8.3f %7.4f %8.3f %7.3f %7d %s" % (
        i, r["puntos"], r["dur"], r["cobertura"], r["en_banda"], r["mov"],
        r["alto_med"], r["cx_med"], r["n"], est))

B, presente = SEL.a_matrices(res["elegidas"], cajas, T, W, H)
print()
print("  PRESENCIA de los elegidos")
for lado in (0, 1):
    p = float(presente[:, lado].mean())
    print("    lado %d ... %6.2f %% de los %d cuadros" % (lado, 100 * p, T))
amb = float((presente.sum(axis=1) == 2).mean())
print("    los DOS a la vez ... %6.2f %%" % (100 * amb))
print("    cuadros con 0 o 1 .. %d" % int((presente.sum(axis=1) < 2).sum()))

crit1 = all(presente[:, l].mean() >= 0.95 for l in (0, 1))
crit2 = len(res["elegidas"]) == 2
print()
print("  CRITERIO 1 presencia >= 95 %% en los dos ... %s" % ("SI" if crit1 else "NO"))
print("  CRITERIO 2 una trayectoria por tirador .... %s" % ("SI" if crit2 else "NO"))

cap = cv2.VideoCapture(video)
ks = np.linspace(0, T - 1, N_PINTA).astype(int)
for k in ks:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
    ok, im = cap.read()
    if not ok:
        continue
    for y in BANDA if False else SEL.BANDA:
        cv2.line(im, (0, int(y * H)), (W, int(y * H)), (90, 90, 90), 1)
    """Primero las descartadas en gris, para que las elegidas queden encima."""
    for r in res["candidatos"]:
        if id(r["t"]) in elegidas_ids:
            continue
        for kk, c in zip(r["t"]["cuadros"], r["t"]["cajas"]):
            if kk == k:
                cv2.rectangle(im, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (130, 130, 130), 2)
                cv2.putText(im, "descartada %.2f" % r["puntos"], (int(c[0]), max(14, int(c[1]) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1)
    for lado, r in enumerate(res["elegidas"][:2]):
        col = (0, 255, 0) if lado == 0 else (0, 220, 255)
        for kk, c in zip(r["t"]["cuadros"], r["t"]["cajas"]):
            if kk == k:
                cv2.rectangle(im, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), col, 3)
                cv2.putText(im, "lado %d  %.2f" % (lado, r["puntos"]),
                            (int(c[0]), max(16, int(c[1]) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)
    cv2.rectangle(im, (0, 0), (W, 40), (12, 12, 18), -1)
    cv2.putText(im, "%s cuadro %d   VERDE/AMARILLO elegidas   GRIS descartadas" % (CLIP[:34], k),
                (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 2)
    cv2.imwrite(str(SALIDA / ("%s_c%04d_selector.png" % (CLIP[:34], k))),
                cv2.resize(im, (1200, int(1200 * H / W))))
cap.release()

np.savez_compressed(PRUEBAS / ("seleccion_%s.npz" % CLIP),
                    boxes=B, presente=presente, W=W, H=H, T=T, video=video)
print()
print("  imagenes en %s" % SALIDA)
print("  seleccion guardada en %s" % (PRUEBAS / ("seleccion_%s.npz" % CLIP)))

"""ETAPA E, paso 19: LAS FIGURAS DEL CONO CONTRA CONO, con la region de solape marcada.

QUE SE PINTA. Los DOS conos, cada uno desde la mano de su tirador con el eje `hombro -> raiz`, y la
REGION DE SOLAPE rasterizada encima, que es donde podria estar el contacto de hojas. Mas el punto de
`peso_max`, que es donde el producto de los dos pesos es maximo, o sea el candidato mas fuerte.

DOS JUEGOS, y el segundo es el que de verdad ensena algo:

  1. **Por clip**, el cuadro de `peso_max` mas alto: el mejor candidato a contacto de cada asalto.
  2. **Los cuadros donde el booleano de solape PARPADEA**, o sea donde cambia de un cuadro al
     siguiente. Un contacto real dura varios cuadros, asi que un parpadeo es un defecto visible y es
     lo que hay que poder juzgar a ojo. Se pinta el par de cuadros consecutivos lado a lado.

LO QUE NO SE PUEDE PINTAR, y hay que decirlo en vez de disimularlo: el encargo pedia las figuras de
los casos donde la anotacion a mano dice contacto y el solape dice que no, y al contrario. **No hay
ninguna**, porque la anotacion a ciegas del paso 16 dio **cero positivos decidibles**: en las laminas
donde los dos tiradores estaban cerca la hoja no se resuelve a ojo (desenfoque de movimiento) o el
monitor del marcador tapa la region del cruce. Sin positivos no hay ni falsos negativos que pintar.
Asi que en su lugar va el juego 2, que si se puede juzgar sin verdad-terreno.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cono3d_lib as C3
import cono_lib as CL
import e7_pinta_cono as E7

import cv2

PRUEBAS = CL.PRUEBAS
SAL = pathlib.Path(
    r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\cono_cono")
SAL.mkdir(parents=True, exist_ok=True)
APERT = 23.1
COL = {0: (255, 170, 40), 1: (60, 210, 255)}
SOLAPE = (90, 90, 255)
N_PARPADEO = 4


def prepara(clip):
    with np.load(PRUEBAS / ("cono_poses_%s.npz" % clip), allow_pickle=True) as z:
        kp, sc, B = z["kp"], z["sc"], z["B"]
        W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
        video = str(z["video"])
    g = {"W": W, "H": H, "T": T, "video": video, "lados": {}}
    for lado in (0, 1):
        m, _d = CL.mano_armada(kp, sc, B, W, H, lado)
        if m is None:
            return None
        vert, u, ok, largo = C3.eje_brazo_2d(kp, sc, lado, m, "hombro")
        _pk, pxm = CL.escala_px_m(B, lado, H)
        g["lados"][lado] = {"vert": vert, "u": u, "ok": ok, "pxm": pxm, "L": CL.HOJA_M * pxm,
                            "hombro": kp[:, lado, CL.MANOS[m]["hombro"]]}
    return g


def pinta(im, g, k, tit, pie):
    """Los dos conos, la region de solape y el punto de `peso_max`."""
    v = im.copy()
    d0, d1 = g["lados"][0], g["lados"][1]
    """LA REGION DE SOLAPE se rasteriza con la MISMA funcion de peso que usa la medida
    (`cono_lib.peso_sector`), no con una aproximacion de dibujo. Si el dibujo usara otra formula, la
    figura y el numero podrian discrepar sin que nadie lo notara."""
    lo = np.array([min(d0["vert"][k][0] - d0["L"], d1["vert"][k][0] - d1["L"]),
                   min(d0["vert"][k][1] - d0["L"], d1["vert"][k][1] - d1["L"])])
    hi = np.array([max(d0["vert"][k][0] + d0["L"], d1["vert"][k][0] + d1["L"]),
                   max(d0["vert"][k][1] + d0["L"], d1["vert"][k][1] + d1["L"])])
    paso = max(2.0, float(max(hi - lo)) / 220.0)
    xs = np.arange(lo[0], hi[0] + paso, paso)
    ys = np.arange(lo[1], hi[1] + paso, paso)
    GX, GY = np.meshgrid(xs, ys)
    P = np.stack([GX.ravel(), GY.ravel()], axis=1)
    w0 = CL.peso_sector(P, d0["vert"][k], d0["u"][k], d0["L"], APERT)
    w1 = CL.peso_sector(P, d1["vert"][k], d1["u"][k], d1["L"], APERT)
    inter = (w0 > 0) & (w1 > 0)
    cap = np.zeros_like(v)
    for x, y in P[inter]:
        xi, yi = int(x), int(y)
        if 0 <= xi < v.shape[1] and 0 <= yi < v.shape[0]:
            cv2.rectangle(cap, (xi, yi), (xi + int(paso), yi + int(paso)), SOLAPE, -1)
    v = cv2.addWeighted(v, 1.0, cap, 0.45, 0)
    for lado, d in ((0, d0), (1, d1)):
        E7.dibuja_cono(v, d["vert"][k], d["u"][k], d["L"], APERT, COL[lado], 2)
        h = d["hombro"][k]
        if np.isfinite(h).all():
            cv2.line(v, np.int32(h), np.int32(d["vert"][k]), COL[lado], 3, cv2.LINE_AA)
        cv2.circle(v, np.int32(d["vert"][k]), 6, (255, 255, 255), -1, cv2.LINE_AA)
    prod = w0 * w1
    if prod.max() > 0:
        j = int(np.argmax(prod))
        cv2.drawMarker(v, (int(P[j, 0]), int(P[j, 1])), (40, 255, 255), cv2.MARKER_CROSS, 26, 3)
    pts = np.array([d0["vert"][k], d1["vert"][k],
                    d0["vert"][k] + d0["L"] * d0["u"][k], d1["vert"][k] + d1["L"] * d1["u"][k]])
    x0 = int(max(0, pts[:, 0].min() - 90))
    x1 = int(min(g["W"], pts[:, 0].max() + 90))
    y0 = int(max(0, pts[:, 1].min() - 90))
    y1 = int(min(g["H"], pts[:, 1].max() + 90))
    if x1 - x0 < 80 or y1 - y0 < 80:
        x0, y0, x1, y1 = 0, 0, g["W"], g["H"]
    return E7.celda(v[y0:y1, x0:x1], tit, pie, 900, 620)


def col_csv(clip):
    f = PRUEBAS / ("cono_cono_%s.csv" % clip)
    out = {}
    with f.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["valido"] == "1":
                out[int(r["cuadro"])] = {
                    "se_solapan": r["se_solapan"] == "1",
                    "peso_max": float(r["peso_max"]) if r["peso_max"] else 0.0,
                    "d_manos_cm": float(r["d_manos_cm"]) if r["d_manos_cm"] else float("nan")}
    return out


hechas = []
print("=" * 100)
print("FIGURAS DEL CONO CONTRA CONO")
print("=" * 100)
for f in sorted(PRUEBAS.glob("cono_cono_*.csv")):
    clip = f.name[len("cono_cono_"):-4]
    if not (PRUEBAS / ("cono_poses_%s.npz" % clip)).is_file():
        continue
    g = prepara(clip)
    if g is None:
        continue
    col = col_csv(clip)
    if not col:
        continue
    cap = cv2.VideoCapture(g["video"])
    kb = max(col, key=lambda k: col[k]["peso_max"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, kb)
    okf, im = cap.read()
    if okf:
        fig = pinta(im, g, kb, "CONO contra CONO  ·  %s  c%d" % (clip, kb),
                    "peso_max %.2f  ·  manos a %.0f cm  ·  la cruz es el candidato de contacto" % (
                        col[kb]["peso_max"], col[kb]["d_manos_cm"]))
        p = SAL / ("conocono_%s_c%04d.png" % (clip, kb))
        cv2.imwrite(str(p), fig)
        hechas.append(p.name)
        print("  %-32s mejor candidato c%-5d peso_max %.2f  manos %.0f cm  ->  %s" % (
            clip[:32], kb, col[kb]["peso_max"], col[kb]["d_manos_cm"], p.name))
    ks = sorted(col)
    flip = [(a, b) for a, b in zip(ks, ks[1:])
            if b == a + 1 and col[a]["se_solapan"] != col[b]["se_solapan"]]
    for a, b in flip[:N_PARPADEO]:
        cel = []
        for k in (a, b):
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            okf, im = cap.read()
            if not okf:
                break
            cel.append(pinta(im, g, k, "c%d  ·  solapan: %s" % (
                k, "SI" if col[k]["se_solapan"] else "NO"),
                "peso_max %.2f  ·  manos %.0f cm" % (col[k]["peso_max"], col[k]["d_manos_cm"])))
        if len(cel) != 2:
            continue
        bl = np.zeros((30, sum(c.shape[1] for c in cel), 3), np.uint8)
        bl[:] = (14, 14, 20)
        cv2.putText(bl, "PARPADEO en %s: el solape cambia de un cuadro al siguiente, y un contacto "
                        "real dura varios cuadros" % clip, (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 150), 1)
        p = SAL / ("PARPADEO_%s_c%04d_%04d.png" % (clip, a, b))
        cv2.imwrite(str(p), np.vstack([np.hstack(cel), bl]))
        hechas.append(p.name)
        print("     parpadeo c%d->%d  ->  %s" % (a, b, p.name))
    cap.release()

print()
print("%d figuras en %s" % (len(hechas), SAL))
(PRUEBAS / "cono_e19_figuras.json").write_text(
    json.dumps({"apert": APERT, "figuras": hechas}, indent=1, ensure_ascii=False),
    encoding="utf-8")

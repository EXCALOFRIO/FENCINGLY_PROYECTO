"""Se puede ARREGLAR la profundidad, o el fallo era de la medida y no del modelo?

DOS SOSPECHAS SOBRE MI PROPIA MEDIDA ANTERIOR, y las dos son mias, no del modelo:

  1. EL CRITERIO ERA MAS DURO DE LO NECESARIO. Medi la RAZON de distancias: |z1-z2| dividido por la
     distancia al ajeno mas cercano, y exigi que fuera menor que 1. Eso pide que los dos tiradores
     esten mas JUNTOS en z de lo que estan del arbitro, o sea una propiedad metrica. Pero para lo que
     hace falta de verdad basta el ORDEN: que al ordenar a todo el mundo por z, ningun ajeno se cuele
     ENTRE los dos tiradores. Un modelo puede acertar el orden y fallar las distancias, y la
     profundidad monocular es exactamente asi.

  2. LEIA LA z EN UNA VENTANA DE 9x9 SOBRE EL TORSO. Con la caja de un tirador en guardia, esa
     ventana puede caer en el hueco entre el brazo y el cuerpo, en la hoja, o en el suelo de detras.
     Se prueba a leerla en la zona de la PELVIS Y EL MUSLO, centrada y estrecha, que es la parte de
     la caja que casi siempre es cuerpo.

CRITERIOS DECLARADOS ANTES DE EJECUTAR:
  `orden`  fraccion de cuadros donde NINGUN ajeno cae entre los dos tiradores al ordenar por z
  `razon`  la medida vieja, para poder comparar
  Y las dos se calculan tambien con la `y del pie`, que es gratis y hay que batir.

  La profundidad SIRVE si `orden` sube claramente por encima de lo que da la `y del pie`. Si empata
  o pierde, se queda como capa que se extrae y se ensena, pero no decide nada.

Se prueban las DOS cabezas: `pointmap` de Sapiens 2, que es la que hay en la tuberia, y `depth` de
Sapiens 1, que es la que Sapiens 2 ya no trae.
"""
from __future__ import annotations

import os
import pathlib
import pickle
import sys

import numpy as np

CACHE_HF = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565\scratchpad\hoja_seg\hf")
os.environ.setdefault("HF_HOME", str(CACHE_HF))

import cv2
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import depth_sapiens1 as D1
import extrae_pareja as EP

SCR = pathlib.Path(os.environ.get(
    "SCRATCH", r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\187cfd83-d8d9-4c4d-bed4-f255724f6b21\scratchpad"))
VIDEOS = "D:/FENCINGLY_VIDEOS_RAW"
PASO = 12


def z_pelvis(Z, caja):
    """QUE HACE: z mediana en la pelvis y el muslo, centrada y estrecha.

    Esa banda de la caja es cuerpo casi siempre, tambien en guardia y en fondo, mientras que la
    altura de los hombros se llena de brazos separados del tronco y de fondo entre ellos.
    """
    x0, y0, x1, y1 = [float(v) for v in caja]
    w, h = x1 - x0, y1 - y0
    cx = 0.5 * (x0 + x1)
    a = int(max(0, cx - 0.15 * w))
    b = int(min(Z.shape[1], cx + 0.15 * w))
    c = int(max(0, y0 + 0.45 * h))
    d = int(min(Z.shape[0], y0 + 0.75 * h))
    if b - a < 3 or d - c < 3:
        return float("nan")
    v = Z[c:d, a:b]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def z_torso_viejo(Z, caja):
    """La lectura de antes, ventana de 9x9 en el centro de la caja, para poder comparar."""
    return D1.z_del_torso(Z, caja)


def _orden_ok(za, zb, zj):
    """QUE HACE: se cuela algun ajeno ENTRE los dos tiradores al ordenar por profundidad?"""
    lo, hi = min(za, zb), max(za, zb)
    return not any(lo < v < hi for v in zj)


def carga_pointmap():
    from transformers import Sapiens2ForPointmapEstimation, Sapiens2ImageProcessor
    snap = next((CACHE_HF / "hub/models--facebook--sapiens2-pointmap-0.4b/snapshots").iterdir())
    proc = Sapiens2ImageProcessor.from_pretrained(str(snap))
    mod = Sapiens2ForPointmapEstimation.from_pretrained(
        str(snap), dtype=torch.float16).to("cuda").eval()
    return proc, mod


def z_pointmap(proc, mod, bgr):
    """QUE HACE: canal z del `pointmap` del cuadro completo.

    `source_sizes` es OBLIGATORIO: `pointmap` tiene `do_pad=True` y sin el, un cuadro 720p entra con
    el 58 % de relleno y el mapa sale desplazado. Es una trampa ya medida en el proyecto.
    """
    H, W = bgr.shape[:2]
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    with torch.inference_mode():
        e = proc(images=rgb, return_tensors="pt")
        e = {k: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda"))
             for k, v in e.items()}
        o = mod(**e)
        m = proc.post_process_pointmap_estimation(
            o, source_sizes=[(H, W)], target_sizes=[(H, W)])[0]
    pm = np.asarray((m["pointmap"] if isinstance(m, dict) else m).cpu(), dtype="float32")
    if pm.ndim == 3 and pm.shape[0] in (3, 4):
        pm = np.transpose(pm, (1, 2, 0))
    return np.nan_to_num(pm[:, :, 2])


def main():
    clips = sys.argv[1:] or ["defeatedHarlequinRabbitSkinkLizard",
                             "ZanyMeagerIndianrockpython", "RichDefensiveBushbaby"]
    mapa = {}
    for raiz, _d, fs in os.walk(VIDEOS):
        for f in fs:
            if f.lower().endswith(".mp4"):
                mapa.setdefault(pathlib.Path(f).stem, os.path.join(raiz, f))
    d1 = D1.carga()
    pproc, pmod = carga_pointmap()
    print("cargados: `depth` de Sapiens 1 y `pointmap` de Sapiens 2")
    print()
    cab = ("clip", "n", "pm viejo", "pm pelvis", "d1 pelvis", "pie")
    print("ORDEN: cuadros donde ningun ajeno se cuela entre los dos tiradores")
    print("%-30s %4s %10s %10s %10s %8s" % cab)
    print("-" * 78)
    filas = []
    for cid in clips:
        ca = SCR / ("det_%s.pkl" % cid)
        if not ca.exists():
            continue
        with open(ca, "rb") as fh:
            cajas, cuadros, W, H, T = pickle.load(fh)
        r = EP.procesa(cajas, cuadros, W, H, T)
        B, pres, mask = r["B"], r["pres"], r["mask"]
        cap = cv2.VideoCapture(mapa[cid])
        acu = {k: [] for k in ("pm_v", "pm_p", "d1_p", "pie", "raz_pm", "raz_pie")}
        k = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            if mask[k] and pres[k, 0] and pres[k, 1] and k % PASO == 0:
                prop = [B[k, l] * np.array([W, H, W, H]) for l in (0, 1)]
                aj = []
                for i in np.nonzero(cuadros == k)[0]:
                    b = np.asarray(cajas[i], dtype="float64")
                    if max(EP.RD._iou(b, prop[0]), EP.RD._iou(b, prop[1])) > 0.30:
                        continue
                    aj.append(b)
                if aj:
                    Zp = z_pointmap(pproc, pmod, im)
                    Zd = D1.profundidad(d1, im)
                    for etq, Z, lee in (("pm_v", Zp, z_torso_viejo), ("pm_p", Zp, z_pelvis),
                                        ("d1_p", Zd, z_pelvis)):
                        za, zb = lee(Z, prop[0]), lee(Z, prop[1])
                        zj = [lee(Z, b) for b in aj]
                        zj = [v for v in zj if np.isfinite(v)]
                        if np.isfinite(za) and np.isfinite(zb) and zj:
                            acu[etq].append(_orden_ok(za, zb, zj))
                            if etq == "pm_p":
                                f = min(min(abs(za - v), abs(zb - v)) for v in zj)
                                if f > 1e-9:
                                    acu["raz_pm"].append(abs(za - zb) / f)
                    pa, pb = prop[0][3] / H, prop[1][3] / H
                    pj = [b[3] / H for b in aj]
                    acu["pie"].append(_orden_ok(pa, pb, pj))
                    f = min(min(abs(pa - v), abs(pb - v)) for v in pj)
                    if f > 1e-9:
                        acu["raz_pie"].append(abs(pa - pb) / f)
            k += 1
        cap.release()
        n = len(acu["pie"])
        if not n:
            continue
        f = (cid[:30], n) + tuple(
            100.0 * np.mean(acu[x]) if acu[x] else float("nan")
            for x in ("pm_v", "pm_p", "d1_p", "pie"))
        filas.append((f, acu))
        print("%-30s %4d %9.0f%% %9.0f%% %9.0f%% %7.0f%%" % f)
    print()
    print("RAZON de distancias, la medida vieja y mas dura (mejor por debajo de 1):")
    for f, acu in filas:
        print("   %-30s pointmap %5.2f   y del pie %5.2f" % (
            f[0], np.median(acu["raz_pm"]) if acu["raz_pm"] else float("nan"),
            np.median(acu["raz_pie"]) if acu["raz_pie"] else float("nan")))
    if filas:
        print()
        tot = {x: [v for f, a in filas for v in a[x]] for x in ("pm_v", "pm_p", "d1_p", "pie")}
        print("AGREGADO sobre %d cuadros:" % len(tot["pie"]))
        for x, et in (("pm_v", "pointmap, lectura vieja de torso"),
                      ("pm_p", "pointmap, lectura de pelvis"),
                      ("d1_p", "depth de Sapiens 1, pelvis"),
                      ("pie", "y del pie, gratis")):
            if tot[x]:
                print("   %-38s %5.1f %%" % (et, 100.0 * np.mean(tot[x])))


if __name__ == "__main__":
    main()

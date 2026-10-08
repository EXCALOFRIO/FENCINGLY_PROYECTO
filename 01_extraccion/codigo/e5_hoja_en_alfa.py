"""ETAPA E, paso 5: LA HOJA ESTA EN EL ALFA DE `matting`. Medida, con control negativo.

LA PISTA ES DEL USUARIO y es la primera que funciona despues de que fallaran nueve metodos clasicos,
SAM2 y el rastreo de video: bajando el umbral del alfa se ve el filo. Mirado antes de medir nada, en
el cuadro 100 del clip de 1080p, **la hoja aparece como una linea limpia** en el mapa de alfa.

DOS CORRECCIONES A LA PISTA, las dos medidas antes de escribir el criterio:

  1. **El alfa del fondo NO es cero**: su mediana va de 3,1e-4 a 5,2e-4 y su p90 de 4,9e-4 a 7,3e-4
     **segun el recorte**. Asi que el umbral de 0,0006 cae justo en el p90 del fondo y selecciona del
     5 al 15 % del recorte: eso no es un filo, es medio fotograma. Y como el nivel de fondo cambia el
     doble de un recorte a otro, **un umbral absoluto no vale**: se usa un percentil del propio
     recorte, medido fuera del cuerpo.

  2. **El filo no esta en el alfa bajisimo, esta en la cola alta del fondo**: aparece por encima del
     p99 (0,018-0,071) y sigue viendose en el p99,9 (0,38-0,63). O sea que la hoja da alfa **mucho
     menor que el cuerpo (que da 1,0) pero mucho mayor que el fondo**. Por eso el test anterior con
     alfa >= 0,5 salia casi todo a cero: el umbral estaba justo por encima de la hoja.

  Las dos cosas siguen encajando con el ancho sub-pixel del fuerte (0,64 px a 720p): un objeto que
  ocupa una fraccion del pixel da una fraccion del alfa, no cero y no uno.

EL CRITERIO, DECLARADO ANTES DE MIRAR LOS ANGULOS:

  region .......... DISCO alrededor del root de la mano, radio R, menos el cuerpo de `seg` DILATADO.
                    Disco y no cono, y esto es lo importante: **un cono es de por si alargado, asi
                    que ajustar una recta dentro de un cono devuelve la direccion del cono aunque
                    dentro solo haya ruido.** El disco es isotropo y no puede inventar direccion.
  umbral .......... alfa > p99 del alfa FUERA del cuerpo, en ese mismo recorte
  el cuerpo ....... se dilata 8 px porque el borde de la silueta tambien da alfa intermedio y un
                    borde mal puesto se veria como hoja
  deteccion ....... `HoughLinesP`; se queda el segmento mas largo cuya recta pase a menos de 0,12 R
                    del root de la mano. Es una condicion de POSICION, no de direccion, para no
                    contaminar la medida que interesa
  medida .......... angulo entre el segmento detectado y el eje del antebrazo, y contra el pulgar

  CONTROL NEGATIVO 1, el que pidio el usuario: fraccion de los pixeles detectados que caen dentro del
    cono, con el cono girado de 15 en 15 grados. Si el cono en su sitio no captura mas que los
    girados, la deteccion no tiene nada que ver con el eje.
  CONTROL NEGATIVO 2, el del azar: si la recta detectada no tuviera relacion con el eje, el valor
    absoluto de la diferencia angular seria uniforme en [0, 90] y su mediana valdria **45 grados**.
    Ese 45 es el numero contra el que se compara, y esta declarado aqui antes de mirar.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

CACHE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565\scratchpad\hoja_seg\hf")
os.environ["HF_HOME"] = str(CACHE)

import cv2
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cono_lib as CL

HUB = CACHE / "hub"
PRUEBAS = CL.PRUEBAS
MUESTRAS = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\cono")
MUESTRAS.mkdir(parents=True, exist_ok=True)
CLIPS = {"SentimentalLargeBedlingtonterrier": 217, "AdorableFittingAsiansmallclawedotter": 317}

PCTL = 99.0
PCTL2 = 99.5
DILATA = 8
ROTS = np.arange(0, 360, 15)
CERCA_ROOT = 0.12
MIN_LARGO = 0.25
GUARDIA_ALFA = 0.05
AZAR = 45.0
"""Mediana esperada de |diferencia angular| si la recta detectada no tuviera relacion con el eje."""


def snap(n):
    d = HUB / ("models--facebook--%s" % n) / "snapshots"
    return next(d.iterdir())


from transformers import (Sapiens2ForImageMatting, Sapiens2ForSemanticSegmentation,
                          Sapiens2ImageProcessor)

p_mat = Sapiens2ImageProcessor.from_pretrained(str(snap("sapiens2-matting-1b")))
p_mat.size = {"height": 1024, "width": 768}
m_mat = Sapiens2ForImageMatting.from_pretrained(
    str(snap("sapiens2-matting-1b")), dtype=torch.float16).to("cuda").eval()
p_seg = Sapiens2ImageProcessor.from_pretrained(str(snap("sapiens2-seg-0.4b")))
p_seg.size = {"height": 1024, "width": 768}
m_seg = Sapiens2ForSemanticSegmentation.from_pretrained(
    str(snap("sapiens2-seg-0.4b")), dtype=torch.float16).to("cuda").eval()


def infiere(sub):
    with torch.inference_mode():
        e = p_mat(images=sub, return_tensors="pt")
        e = {k: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda")) for k, v in e.items()}
        r = p_mat.post_process_image_matting(m_mat(**e), target_sizes=[sub.shape[:2]])[0]
        a = np.squeeze(np.nan_to_num(np.asarray(
            (r["alpha"] if isinstance(r, dict) else r).cpu(), dtype="float32")))
        e = p_seg(images=sub, return_tensors="pt")
        e = {k: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda")) for k, v in e.items()}
        g = np.asarray(p_seg.post_process_semantic_segmentation(
            m_seg(**e), target_sizes=[sub.shape[:2]])[0].cpu()).astype("uint8")
    return a, g


def recorte(b, br, vert, u, L, W, H):
    """Recorte de UN tirador, alargado a lo largo del eje hasta L, y cortado antes del rival.

    El corte antes del rival no es un detalle: con dos tiradores dentro, `matting` devuelve 0,0053,
    el valor del fondo vacio. Se devuelve tambien R para poder decir cuanto del cono cabe.
    """
    x0, y0, x1, y1 = b[0] * W, b[1] * H, b[2] * W, b[3] * H
    mx, my = 0.18 * (x1 - x0), 0.15 * (y1 - y0)
    punta = vert + L * u
    cx0 = min(x0 - mx, vert[0] - 0.25 * L, punta[0] - 0.10 * L)
    cx1 = max(x1 + mx, vert[0] + 0.25 * L, punta[0] + 0.10 * L)
    cy0 = min(y0 - my, punta[1] - 0.10 * L)
    cy1 = max(y1 + my, punta[1] + 0.10 * L)
    if np.isfinite(br).all():
        rx0, rx1 = br[0] * W, br[2] * W
        pad = 0.12 * (rx1 - rx0)
        if 0.5 * (rx0 + rx1) > 0.5 * (x0 + x1):
            cx1 = min(cx1, rx0 - pad)
        else:
            cx0 = max(cx0, rx1 + pad)
    cx0, cy0 = int(max(0, cx0)), int(max(0, cy0))
    cx1, cy1 = int(min(W, cx1)), int(min(H, cy1))
    R = 0.0
    if cx1 > cx0 and cy1 > cy0:
        R = min(L, max(0.0, min(vert[0] - cx0, cx1 - vert[0], vert[1] - cy0, cy1 - vert[1])))
    return (cx0, cy0, cx1, cy1), float(R)


def dif90(a, b):
    """Diferencia angular entre dos RECTAS (no vectores): en [0, 90]."""
    d = abs((a - b + 90.0) % 180.0 - 90.0)
    return d


todo = {}
for clip, kt in CLIPS.items():
    z = CL.carga(clip)
    kp, sc, B = z["kp"], z["sc"], z["B"]
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    print("=" * 104)
    print("%s   %dx%d   tocado en %d" % (clip, W, H, kt))
    print("=" * 104)
    geo = {}
    for lado in (0, 1):
        mano, _d = CL.mano_armada(kp, sc, B, W, H, lado)
        vert, u, ok, _l = CL.eje_antebrazo(kp, sc, lado, mano)
        _pk, pxm = CL.escala_px_m(B, lado, H)
        M = CL.MANOS[mano]
        cent = kp[:, lado, M["pulgar"]].mean(axis=1) - kp[:, lado, M["root"]]
        geo[lado] = {"mano": mano, "vert": vert, "u": u, "ok": ok, "pxm": pxm,
                     "L": CL.HOJA_M * pxm, "u_pulgar": cent.astype("float64")}

    ks = sorted(set([k for k in range(kt - 24, kt + 8) if 0 <= k < T]
                    + [int(round(x)) for x in np.linspace(0.06 * T, 0.94 * T, 14)]))
    cap = cv2.VideoCapture(str(z["video"]))
    filas = []
    t0 = time.perf_counter()
    for k in ks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        okf, im = cap.read()
        if not okf:
            continue
        for lado in (0, 1):
            g = geo[lado]
            if not g["ok"][k] or not np.isfinite(B[k, lado]).all():
                continue
            vert, u, L = g["vert"][k], g["u"][k], g["L"]
            (cx0, cy0, cx1, cy1), R = recorte(B[k, lado], B[k, 1 - lado], vert, u, L, W, H)
            if cx1 - cx0 < 64 or cy1 - cy0 < 64 or R < 24:
                continue
            sub = np.ascontiguousarray(im[cy0:cy1, cx0:cx1, ::-1])
            alfa, seg = infiere(sub)
            am = float(alfa.mean())
            cuerpo = cv2.dilate((seg != 0).astype("uint8"),
                                np.ones((2 * DILATA + 1,) * 2, np.uint8)) > 0
            fuera = alfa[~cuerpo]
            if fuera.size < 500:
                continue
            hh, ww = alfa.shape[:2]
            GY, GX = np.mgrid[0:hh, 0:ww]
            dr = np.hypot(GX + cx0 - vert[0], GY + cy0 - vert[1])
            disco = dr <= R
            fila = {"clip": clip, "cuadro": k, "lado": lado, "R": R, "R_frac": R / L,
                    "alfa_recorte": am, "contaminado": bool(am < GUARDIA_ALFA),
                    "recorte": [cx0, cy0, cx1, cy1],
                    "ang_eje": float(np.degrees(np.arctan2(u[1], u[0]))),
                    "fondo_med": float(np.median(fuera)), "det": {}}
            for pc in (PCTL, PCTL2):
                t = float(np.percentile(fuera, pc))
                m = disco & (~cuerpo) & (alfa > t)
                n = int(m.sum())
                d = {"umbral": t, "n_px": n, "n_disco": int(disco.sum())}
                seg_best = None
                if n >= 20:
                    bw = (m * 255).astype("uint8")
                    lines = cv2.HoughLinesP(bw, 1, np.pi / 360.0, 12,
                                            minLineLength=int(max(8, MIN_LARGO * R)),
                                            maxLineGap=int(max(2, 0.05 * R)))
                    if lines is not None:
                        """OpenCV 5 devuelve (N,4); las versiones anteriores (N,1,4). Se acepta
                        cualquiera de las dos formas en vez de suponer una."""
                        lines = np.asarray(lines).reshape(-1, 4)
                        cands = []
                        for x1_, y1_, x2_, y2_ in lines:
                            p1 = np.array([x1_ + cx0, y1_ + cy0], dtype="float64")
                            p2 = np.array([x2_ + cx0, y2_ + cy0], dtype="float64")
                            v = p2 - p1
                            lg = float(np.linalg.norm(v))
                            if lg < 1:
                                continue
                            nv = np.array([-v[1], v[0]]) / lg
                            dist = abs(float(nv @ (vert - p1)))
                            if dist > CERCA_ROOT * R:
                                continue
                            cands.append((lg, p1, p2, dist))
                        if cands:
                            cands.sort(key=lambda c: -c[0])
                            lg, p1, p2, dist = cands[0]
                            v = p2 - p1
                            aa = float(np.degrees(np.arctan2(v[1], v[0])))
                            seg_best = {"largo": lg, "largo_frac_R": lg / R,
                                        "dist_root": dist, "ang": aa,
                                        "p1": p1.tolist(), "p2": p2.tolist(),
                                        "n_lineas": int(len(lines)), "n_cerca": len(cands),
                                        "dif_eje": float(dif90(aa, fila["ang_eje"]))}
                            up = g["u_pulgar"][k]
                            if np.isfinite(up).all() and np.linalg.norm(up) > 0.5:
                                seg_best["dif_pulgar"] = float(dif90(
                                    aa, float(np.degrees(np.arctan2(up[1], up[0])))))
                    """CONTROL 1: cuantos de los pixeles detectados caen en el cono, girandolo."""
                    P = np.stack([GX[m] + cx0, GY[m] + cy0], axis=1).astype("float64")
                    perfil = {}
                    for rot in ROTS:
                        th = np.radians(fila["ang_eje"] + rot)
                        ur = np.array([np.cos(th), np.sin(th)])
                        w = CL.peso_sector(P, vert, ur, R, 8.0)
                        perfil[int(rot)] = float((w > 0).mean())
                    d["perfil_rot"] = perfil
                d["linea"] = seg_best
                fila["det"]["p%.1f" % pc] = d
            filas.append(fila)
    cap.release()
    print("  %d recortes en %.1f s (GPU COMPARTIDA: tiempo indicativo)" % (
        len(filas), time.perf_counter() - t0))

    val = [f for f in filas if not f["contaminado"]]
    print("  GUARDIA un tirador por recorte: alfa medio mediana %.4f  ·  contaminados %d de %d"
          % (float(np.median([f["alfa_recorte"] for f in filas])), len(filas) - len(val),
             len(filas)))
    print("  ALFA DEL FONDO por recorte: mediana %.2e  min %.2e  max %.2e  ->  por eso el umbral "
          "es un PERCENTIL del recorte y no un valor fijo" % (
              float(np.median([f["fondo_med"] for f in val])),
              float(np.min([f["fondo_med"] for f in val])),
              float(np.max([f["fondo_med"] for f in val]))))
    rf = np.array([f["R_frac"] for f in val])
    print("  R/hoja: mediana %.2f  p10 %.2f  max %.2f  ->  se examinan los primeros %.0f cm de 90"
          % (np.median(rf), np.percentile(rf, 10), rf.max(), 90 * np.median(rf)))

    for pc in (PCTL, PCTL2):
        key = "p%.1f" % pc
        print()
        print("  --- umbral = p%.1f del alfa fuera del cuerpo" % pc)
        print("  %-7s %-5s %8s %9s %8s %8s %9s %9s  %s" % (
            "cuadro", "lado", "umbral", "n_px", "largo/R", "d_root", "dif_eje", "dif_pulg", "nota"))
        difs, difp, con = [], [], 0
        for f in val:
            d = f["det"][key]
            ln = d["linea"]
            if ln is None:
                print("  %-7d %-5d %8.4f %9d %8s %8s %9s %9s  sin recta pegada a la mano%s" % (
                    f["cuadro"], f["lado"], d["umbral"], d["n_px"], "-", "-", "-", "-",
                    "   <-- tocado" if f["cuadro"] == kt else ""))
                continue
            con += 1
            difs.append(ln["dif_eje"])
            if "dif_pulgar" in ln:
                difp.append(ln["dif_pulgar"])
            print("  %-7d %-5d %8.4f %9d %8.2f %8.1f %9.1f %9s  %s%s" % (
                f["cuadro"], f["lado"], d["umbral"], d["n_px"], ln["largo_frac_R"],
                ln["dist_root"], ln["dif_eje"],
                ("%.1f" % ln["dif_pulgar"]) if "dif_pulgar" in ln else "-",
                "%d rectas" % ln["n_lineas"], "   <-- tocado" if f["cuadro"] == kt else ""))
        print("  RECTA PEGADA A LA MANO en %d de %d recortes (%.1f %%)" % (
            con, len(val), 100.0 * con / max(len(val), 1)))
        if difs:
            dd = np.array(difs)
            print("  |dif| CONTRA EL EJE DEL ANTEBRAZO: mediana %.1f  p25 %.1f  p75 %.1f  "
                  "(azar declarado = %.0f)" % (
                      np.median(dd), np.percentile(dd, 25), np.percentile(dd, 75), AZAR))
            print("     dentro de 10 grados: %.1f %%  ·  de 20: %.1f %%  ·  de 30: %.1f %%" % (
                100 * (dd <= 10).mean(), 100 * (dd <= 20).mean(), 100 * (dd <= 30).mean()))
            print("     VEREDICTO: %s" % (
                "la recta del alfa APUNTA como el antebrazo: mediana %.1f contra %.0f de azar"
                % (np.median(dd), AZAR) if np.median(dd) < 0.5 * AZAR else
                "la recta del alfa NO se alinea con el antebrazo: mediana %.1f contra %.0f de azar"
                % (np.median(dd), AZAR)))
        if difp:
            dp = np.array(difp)
            print("  |dif| CONTRA EL EJE DEL PULGAR: mediana %.1f  p25 %.1f  p75 %.1f" % (
                np.median(dp), np.percentile(dp, 25), np.percentile(dp, 75)))
        pr = [f["det"][key].get("perfil_rot") for f in val if f["det"][key].get("perfil_rot")]
        if pr:
            M = np.array([[p[str(r)] if str(r) in p else p[r] for r in ROTS] for p in pr])
            print("  CONTROL 1, cono girado: fraccion de los pixeles detectados dentro del cono")
            print("     rot 0 -> %.4f   mediana de las rotaciones -> %.4f   razon %.2f" % (
                np.median(M[:, 0]), np.median(np.median(M[:, 1:], axis=1)),
                np.median(M[:, 0]) / max(np.median(np.median(M[:, 1:], axis=1)), 1e-9)))
            rank = 1 + (M[:, 1:] > M[:, [0]]).sum(axis=1)
            print("     el cono en su sitio queda el 1o de 24 en %.1f %% de los recortes  ·  "
                  "rank mediano %.0f" % (100 * (rank == 1).mean(), np.median(rank)))
        todo.setdefault(clip, {})[key] = {
            "con_recta": con, "n": len(val),
            "dif_eje_mediana": float(np.median(difs)) if difs else None,
            "dif_pulgar_mediana": float(np.median(difp)) if difp else None}

    (PRUEBAS / ("cono_hoja_%s.json" % clip)).write_text(
        json.dumps(filas, indent=1, default=float), encoding="utf-8")
    print()

(PRUEBAS / "cono_e5_hoja.json").write_text(
    json.dumps(todo, indent=1, ensure_ascii=False), encoding="utf-8")
print("guardado %s" % (PRUEBAS / "cono_e5_hoja.json"))

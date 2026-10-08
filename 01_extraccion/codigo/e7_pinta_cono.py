"""ETAPA E, paso 7: imagenes y VIDEO para que el usuario lo juzgue a ojo.

QUE SE PINTA Y QUE NO, y esto es una regla del proyecto y no una preferencia:

  SE PINTA el **eje** (medido: `codo -> root`), el **cono** como region con su apertura, el **alfa
  dentro del cono** y la **recta detectada en el alfa**.
  NO SE PINTA ninguna hoja inventada. Cuando no hay recta detectada, no se dibuja recta: se escribe
  «sin recta». Un cuaderno anterior de este proyecto trazaba 95 px desde la muneca y metia esa hoja
  inventada en la etiqueta de tocado; eso no vuelve a entrar.

POR QUE VIDEO Y NO SOLO IMAGENES: una imagen fija no ensena si una capa PARPADEA, y la deteccion de
la hoja **es intermitente** (42,6 % de los recortes a 1080p, 1,6 % a 720p). Eso solo se ve en
movimiento, y es justo lo que el usuario tiene que juzgar.

EL CONO SE PINTA CON DOS APERTURAS a la vez, porque las dos estan medidas y dicen cosas distintas:
  fina  = p95 del temblor del eje entre cuadros (5,1-8,2 grados): lo que el eje puede apuntar
  ancha = p90 del angulo medido entre la hoja detectada y el eje (25,4 grados a 1080p): lo que hace
          falta para que la hoja quepa de verdad
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

CACHE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565\scratchpad\hoja_seg\hf")
os.environ["HF_HOME"] = str(CACHE)

import cv2
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cono_lib as CL

HUB = CACHE / "hub"
PRUEBAS = CL.PRUEBAS
SAL = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\cono")
SAL.mkdir(parents=True, exist_ok=True)
CLIPS = {"SentimentalLargeBedlingtonterrier": 217, "AdorableFittingAsiansmallclawedotter": 317}
VENT = 30
PCTL = 99.5
DILATA = 8
CERCA_ROOT = 0.12
MIN_LARGO = 0.25
COL = {0: (255, 170, 40), 1: (60, 210, 255)}


def snap(n):
    return next((HUB / ("models--facebook--%s" % n) / "snapshots").iterdir())


p_mat = m_mat = p_seg = m_seg = None


def carga_modelos():
    """LOS PESOS SE CARGAN AL PRIMER USO, no al importar el modulo, y esto NO es cosmetica.

    `matting-1b` en fp16 ocupa GPU de verdad. Quien quiera reusar de aqui solo `recorte`, `detecta`,
    `dibuja_cono` o `celda` (que son numpy y cv2, sin modelo) se comia una SEGUNDA copia del modelo en
    la GPU con el simple `import`, encima de la que ya tenga cargada por su cuenta. Con la carga
    perezosa `infiere` se comporta igual que antes y el import no reserva nada.
    """
    global p_mat, m_mat, p_seg, m_seg
    if m_mat is not None:
        return
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
    carga_modelos()
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
    R = min(L, max(0.0, min(vert[0] - cx0, cx1 - vert[0], vert[1] - cy0, cy1 - vert[1]))) \
        if (cx1 > cx0 and cy1 > cy0) else 0.0
    return (cx0, cy0, cx1, cy1), float(R)


def detecta(alfa, seg, vert, cx0, cy0, R):
    cuerpo = cv2.dilate((seg != 0).astype("uint8"), np.ones((2 * DILATA + 1,) * 2, np.uint8)) > 0
    fuera = alfa[~cuerpo]
    if fuera.size < 500:
        return None, None
    t = float(np.percentile(fuera, PCTL))
    hh, ww = alfa.shape[:2]
    GY, GX = np.mgrid[0:hh, 0:ww]
    disco = np.hypot(GX + cx0 - vert[0], GY + cy0 - vert[1]) <= R
    m = disco & (~cuerpo) & (alfa > t)
    if m.sum() < 20:
        return m, None
    lines = cv2.HoughLinesP((m * 255).astype("uint8"), 1, np.pi / 360.0, 12,
                            minLineLength=int(max(8, MIN_LARGO * R)),
                            maxLineGap=int(max(2, 0.05 * R)))
    if lines is None:
        return m, None
    cands = []
    for x1_, y1_, x2_, y2_ in np.asarray(lines).reshape(-1, 4):
        p1 = np.array([x1_ + cx0, y1_ + cy0], dtype="float64")
        p2 = np.array([x2_ + cx0, y2_ + cy0], dtype="float64")
        v = p2 - p1
        lg = float(np.linalg.norm(v))
        if lg < 1:
            continue
        nv = np.array([-v[1], v[0]]) / lg
        if abs(float(nv @ (vert - p1))) > CERCA_ROOT * R:
            continue
        cands.append((lg, p1, p2))
    if not cands:
        return m, None
    cands.sort(key=lambda c: -c[0])
    return m, cands[0]


def dibuja_cono(img, vert, u, L, apert, col, grosor=1):
    a0 = np.degrees(np.arctan2(u[1], u[0]))
    pts = [vert]
    for t in np.linspace(-apert, apert, 25):
        th = np.radians(a0 + t)
        pts.append(vert + L * np.array([np.cos(th), np.sin(th)]))
    cv2.polylines(img, [np.int32(pts)], True, col, grosor, cv2.LINE_AA)


def celda(img, tit, pie="", ancho=620, alto=460):
    ih, iw = img.shape[:2]
    e = min(ancho / max(iw, 1), alto / max(ih, 1))
    nw, nh = max(1, int(iw * e)), max(1, int(ih * e))
    v = np.zeros((alto, ancho, 3), np.uint8)
    ox, oy = (ancho - nw) // 2, (alto - nh) // 2
    v[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh))
    cv2.rectangle(v, (0, 0), (ancho, 24), (14, 14, 20), -1)
    cv2.putText(v, tit, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1)
    if pie:
        cv2.rectangle(v, (0, alto - 22), (ancho, alto), (14, 14, 20), -1)
        cv2.putText(v, pie, (6, alto - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 235, 255), 1)
    return v


def main():
    """El bucle va en una funcion para que `e8_evidencia_hoja.py` pueda importar los
    ayudantes (`recorte`, `infiere`, `detecta`) sin ejecutar el video entero."""
    for clip, kt in CLIPS.items():
        z = CL.carga(clip)
        kp, sc, B = z["kp"], z["sc"], z["B"]
        W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
        FPS = float(z["fps"])
        res = json.loads((PRUEBAS / "cono_e6_apertura.json").read_text(encoding="utf-8"))
        ap_ancha = res.get(clip, {}).get("p99.5", {}).get("dif_eje_p90") or 25.4
        geo = {}
        for lado in (0, 1):
            mano, _d = CL.mano_armada(kp, sc, B, W, H, lado)
            vert, u, ok, _l = CL.eje_antebrazo(kp, sc, lado, mano)
            _pk, pxm = CL.escala_px_m(B, lado, H)
            a = CL.angulo(u)
            p = ok[:-1] & ok[1:]
            d1 = np.abs(CL.dif_ang(a[1:][p], a[:-1][p]))
            d1 = d1[np.isfinite(d1)]
            geo[lado] = {"mano": mano, "vert": vert, "u": u, "ok": ok, "pxm": pxm,
                         "L": CL.HOJA_M * pxm,
                         "fina": float(np.percentile(d1, 95)) if d1.size >= 10 else 8.0}
        print("%s  ·  apertura fina %.1f / %.1f  ·  apertura ancha (hoja medida) %.1f grados" % (
            clip, geo[0]["fina"], geo[1]["fina"], ap_ancha))

        cap = cv2.VideoCapture(str(z["video"]))
        ks = [k for k in range(max(0, kt - VENT), min(T, kt + VENT // 2)) ]
        escritor, forma = None, None
        out = SAL / ("cono_%s.mp4" % clip)
        n_det = 0
        for k in ks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            okf, im = cap.read()
            if not okf:
                continue
            vis = im.copy()
            paneles = []
            pie_g = []
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
                m, ln = detecta(alfa, seg, vert, cx0, cy0, R)
                dibuja_cono(vis, vert, u, L, ap_ancha, COL[lado], 1)
                dibuja_cono(vis, vert, u, L, g["fina"], COL[lado], 2)
                cv2.line(vis, tuple(np.int32(vert - 0.35 * L * u)), tuple(np.int32(vert)),
                         (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(vis, tuple(np.int32(vert)), 4, (255, 255, 255), -1)
                if ln is not None:
                    n_det += 1
                    _lg, p1, p2 = ln
                    cv2.line(vis, tuple(np.int32(p1)), tuple(np.int32(p2)), (60, 255, 60), 2, cv2.LINE_AA)
                    dif = abs((np.degrees(np.arctan2((p2 - p1)[1], (p2 - p1)[0]))
                               - np.degrees(np.arctan2(u[1], u[0])) + 90) % 180 - 90)
                    pie_g.append("l%d recta %.0f cm, %.0f gr del eje" % (
                        lado, 100 * _lg / g["pxm"], dif))
                else:
                    pie_g.append("l%d sin recta" % lado)
                if m is not None:
                    la = np.log10(np.clip(alfa, 1e-6, 1.0))
                    hm = np.clip((la + 4.0) / 4.0, 0, 1)
                    p = cv2.applyColorMap((hm * 255).astype("uint8"), cv2.COLORMAP_TURBO)
                    ov = np.zeros_like(p)
                    ov[m] = (255, 255, 255)
                    p = cv2.addWeighted(p, 0.75, ov, 0.25, 0)
                    hh, ww = alfa.shape[:2]
                    GY, GX = np.mgrid[0:hh, 0:ww]
                    P = np.stack([GX.ravel() + cx0, GY.ravel() + cy0], axis=1).astype("float64")
                    wc = (CL.peso_sector(P, vert, u, R, ap_ancha) > 0).reshape(hh, ww)
                    cnt, _h = cv2.findContours(wc.astype("uint8"), cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(p, cnt, -1, COL[lado], 1)
                    if ln is not None:
                        _lg, p1, p2 = ln
                        cv2.line(p, tuple(np.int32(p1 - [cx0, cy0])), tuple(np.int32(p2 - [cx0, cy0])),
                                 (60, 255, 60), 2)
                    paneles.append(celda(p, "alfa (log) + cono + pixeles detectados  ·  lado %d" % lado,
                                         "umbral p%.1f = %.4f  ·  %d px" % (
                                             PCTL, float(np.percentile(alfa[seg == 0], PCTL))
                                             if (seg == 0).any() else 0.0, int(m.sum()))))
            p1 = celda(vis, "cono fino (temblor del eje) + cono ancho (hoja medida) + recta del alfa",
                       "  ·  ".join(pie_g) if pie_g else "sin tirador valido", 900, 460)
            while len(paneles) < 2:
                paneles.append(celda(np.full((240, 320, 3), 22, np.uint8), "sin recorte", ""))
            fila = np.hstack([p1] + paneles[:2])
            cv2.rectangle(fila, (0, fila.shape[0] - 26), (fila.shape[1], fila.shape[0]), (14, 14, 20), -1)
            cv2.putText(fila, "%s   cuadro %d   tocado en %d   NO SE DIBUJA HOJA SIN MEDIRLA" % (
                clip, k, kt), (8, fila.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1)
            if escritor is None:
                forma = fila.shape
                escritor = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                           max(FPS / 3, 6), (fila.shape[1], fila.shape[0]))
            if fila.shape == forma:
                escritor.write(fila)
            if k in (kt, kt - 12, kt - 20):
                f = SAL / ("cono_%s_%d.png" % (clip[:18], k))
                cv2.imwrite(str(f), fila)
                print("    imagen %s" % f)
        cap.release()
        if escritor is not None:
            escritor.release()
        print("    VIDEO %s  ·  %d cuadros  ·  recta detectada en %d cuadros-tirador" % (
            out, len(ks), n_det))


if __name__ == "__main__":
    main()

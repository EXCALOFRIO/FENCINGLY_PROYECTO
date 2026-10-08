"""ETAPA E, paso 4: CAPTA `matting` LA HOJA DENTRO DEL CONO? Con control negativo obligatorio.

SUPERADO. NO LEER EL VEREDICTO QUE IMPRIME ESTE FICHERO. Se conserva porque sus medidas crudas
valen y porque documenta dos errores mios que conviene no repetir:

  1. **el umbral estaba justo por encima de la hoja.** Aqui se pregunta por alfa >= 0,5 y la hoja da
     de 0,02 a 0,5, asi que salia casi todo a cero. El umbral bueno es un PERCENTIL del propio
     recorte, y la medida esta en `e5_hoja_en_alfa.py`.
  2. **el criterio aprobaba la nada**: `razon = frac(0)/mediana` con los dos a cero da `inf` y
     contaba como acierto. Arreglado en `e4b_veredicto_matting.py`.

  El resultado de verdad esta en `e5_hoja_en_alfa.py` y `e6_apertura_desde_la_hoja.py`: la hoja SI
  esta en el alfa, y la pista de mirar el alfa bajo es del usuario.


LA PREGUNTA: `matting` de Sapiens devuelve un alfa de silueta fina. En el mosaico se vio **una linea
diagonal dentro de la silueta** que parecia la hoja, y nadie lo habia medido. Si el alfa alto cae
**dentro del cono y fuera del cuerpo**, `matting` estaria viendo la hoja donde la segmentacion no
llega. Si no, no.

EL CONTROL NEGATIVO NO ES UN CONO ROTADO, SON VEINTICUATRO. Rotar 90 o 180 grados da un solo numero
de comparacion y una sola comparacion se puede tener suerte. Se gira el cono alrededor del vertice de
15 en 15 grados hasta dar la vuelta completa y se saca el **perfil** de alfa contra rotacion. La hoja,
si esta, tiene que producir un PICO en 0 grados.

CRITERIO DECLARADO ANTES DE MIRAR, y con dos condiciones porque una sola se cumple por azar:
  1. el 0 de rotacion tiene que estar en el **10 % superior** del perfil (o sea entre los 3 mejores
     de 24), y
  2. tiene que superar la **mediana del perfil por 2 veces o mas**,
  ... en la **mayoria** de los cuadros probados. Si no se cumple, `matting` NO discrimina la hoja y
  se dice que no.

LA CONDICION DURA DE `matting`, YA MEDIDA, QUE AQUI SE RESPETA: con DOS tiradores en el recorte
devuelve 0,0053, el valor del fondo vacio. Asi que el recorte se limita para que no entre el rival, y
eso **acorta el cono**: se mide y se dice que fraccion de los 90 cm cabe en el recorte. Ademas se
comprueba el alfa medio del recorte entero como guardia: si baja de 0,05, el recorte esta contaminado
y ese cuadro no cuenta.

EL CUERPO SE QUITA CON `seg`, no con el propio alfa, que seria circular. Y se mide con el cuerpo
dilatado y sin dilatar, porque el borde del cuerpo tiene alfa intermedio y un borde mal puesto se
veria como hoja.
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

ROTS = np.arange(0, 360, 15)
ALFA_SOLIDO = 0.5
"""Declarado: un pixel cuenta como primer plano solido si su alfa pasa de 0,5. Se informa tambien el
alfa medio, para que el veredicto no dependa de un solo umbral."""
DILATA = 3
GUARDIA_ALFA = 0.05
"""Si el alfa medio del recorte baja de esto, hay mas de una persona o ninguna: el cuadro no cuenta.
Medido antes en el proyecto: un tirador solo da 0,21 y dos tiradores o fondo vacio dan 0,0053."""
MIN_AREA_REL = 0.30
"""Una rotacion cuya region valida sea menor que el 30 % de la del 0 no es comparable y se excluye."""


def snap(n):
    d = HUB / ("models--facebook--%s" % n) / "snapshots"
    return next(d.iterdir()) if d.is_dir() else None


def cuadros_prueba(kt, T, n_vent=16, n_disp=8):
    """Una ventana pegada al tocado mas un reparto por todo el clip: si solo se mirara el tocado,
    no se sabria si lo que se mide es la hoja o el instante."""
    v = [k for k in range(kt - n_vent + 3, kt + 4) if 0 <= k < T]
    d = [int(round(x)) for x in np.linspace(0.08 * T, 0.92 * T, n_disp)]
    return sorted(set(v) | set(d))


from transformers import (Sapiens2ForImageMatting, Sapiens2ForSemanticSegmentation,
                          Sapiens2ImageProcessor)

s_mat = snap("sapiens2-matting-1b")
p_mat = Sapiens2ImageProcessor.from_pretrained(str(s_mat))
p_mat.size = {"height": 1024, "width": 768}
m_mat = Sapiens2ForImageMatting.from_pretrained(str(s_mat), dtype=torch.float16).to("cuda").eval()
s_seg = snap("sapiens2-seg-0.4b")
p_seg = Sapiens2ImageProcessor.from_pretrained(str(s_seg))
p_seg.size = {"height": 1024, "width": 768}
m_seg = Sapiens2ForSemanticSegmentation.from_pretrained(
    str(s_seg), dtype=torch.float16).to("cuda").eval()


def infiere(sub):
    with torch.inference_mode():
        e = p_mat(images=sub, return_tensors="pt")
        e = {k: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda")) for k, v in e.items()}
        o = m_mat(**e)
        r = p_mat.post_process_image_matting(o, target_sizes=[sub.shape[:2]])[0]
        a = np.squeeze(np.nan_to_num(np.asarray(
            (r["alpha"] if isinstance(r, dict) else r).cpu(), dtype="float32")))
        e = p_seg(images=sub, return_tensors="pt")
        e = {k: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda")) for k, v in e.items()}
        o = m_seg(**e)
        g = np.asarray(p_seg.post_process_semantic_segmentation(
            o, target_sizes=[sub.shape[:2]])[0].cpu()).astype("uint8")
    return a, g


todo = {}
for clip, kt in CLIPS.items():
    z = CL.carga(clip)
    kp, sc, B = z["kp"], z["sc"], z["B"]
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    print("=" * 100)
    print("%s   %dx%d   tocado en %d" % (clip, W, H, kt))
    print("=" * 100)
    geo = {}
    for lado in (0, 1):
        mano, det = CL.mano_armada(kp, sc, B, W, H, lado)
        if mano is None:
            geo[lado] = None
            continue
        vert, u, ok, _l = CL.eje_antebrazo(kp, sc, lado, mano)
        _pk, pxm = CL.escala_px_m(B, lado, H)
        a = CL.angulo(u)
        p = ok[:-1] & ok[1:]
        d1 = np.abs(CL.dif_ang(a[1:][p], a[:-1][p]))
        d1 = d1[np.isfinite(d1)]
        geo[lado] = {"mano": mano, "vert": vert, "u": u, "ok": ok, "pxm": pxm,
                     "L": CL.HOJA_M * pxm,
                     "apert": float(np.percentile(d1, 95)) if d1.size >= 10 else 8.0}

    cap = cv2.VideoCapture(str(z["video"]))
    filas = []
    guardar = {}
    t0 = time.perf_counter()
    for k in cuadros_prueba(kt, T):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        okf, im = cap.read()
        if not okf:
            continue
        for lado in (0, 1):
            g = geo[lado]
            if g is None or not g["ok"][k]:
                continue
            b, br = B[k, lado], B[k, 1 - lado]
            if not np.isfinite(b).all():
                continue
            vert = g["vert"][k]

            """R: hasta donde llega el cono SIN meter al rival en el recorte. Es el precio de la
            condicion de un tirador por recorte y se mide, no se esconde."""
            R = g["L"]
            if np.isfinite(br).all():
                rx0, ry0, rx1, ry1 = br[0] * W, br[1] * H, br[2] * W, br[3] * H
                dx = max(rx0 - vert[0], vert[0] - rx1, 0.0)
                dy = max(ry0 - vert[1], vert[1] - ry1, 0.0)
                margen = 0.10 * (rx1 - rx0)
                R = min(R, max(20.0, float(np.hypot(dx, dy)) - margen))

            x0, y0, x1, y1 = b[0] * W, b[1] * H, b[2] * W, b[3] * H
            mx, my = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
            cx0 = int(max(0, min(x0 - mx, vert[0] - R)))
            cy0 = int(max(0, min(y0 - my, vert[1] - R)))
            cx1 = int(min(W, max(x1 + mx, vert[0] + R)))
            cy1 = int(min(H, max(y1 + my, vert[1] + R)))
            if np.isfinite(br).all():
                if br[0] * W > b[2] * W:
                    cx1 = int(min(cx1, br[0] * W - 0.10 * (br[2] - br[0]) * W))
                else:
                    cx0 = int(max(cx0, br[2] * W + 0.10 * (br[2] - br[0]) * W))
            if cx1 - cx0 < 48 or cy1 - cy0 < 48:
                continue
            sub = np.ascontiguousarray(im[cy0:cy1, cx0:cx1, ::-1])
            alfa, seg = infiere(sub)
            alfa_recorte = float(alfa.mean())

            cuerpo = seg != 0
            cuerpo_d = cv2.dilate(cuerpo.astype("uint8"),
                                  np.ones((2 * DILATA + 1, 2 * DILATA + 1), np.uint8)) > 0
            hh, ww = alfa.shape[:2]
            GY, GX = np.mgrid[0:hh, 0:ww]
            P = np.stack([GX.ravel() + cx0, GY.ravel() + cy0], axis=1).astype("float64")
            ang0 = np.degrees(np.arctan2(g["u"][k][1], g["u"][k][0]))
            fila = {"clip": clip, "cuadro": k, "lado": lado, "R_px": float(R),
                    "R_frac_hoja": float(R / g["L"]), "alfa_recorte": alfa_recorte,
                    "apert": g["apert"], "recorte": [cx0, cy0, cx1, cy1],
                    "contaminado": bool(alfa_recorte < GUARDIA_ALFA), "perfil": {}}
            for rot in ROTS:
                th = np.radians(ang0 + rot)
                ur = np.array([np.cos(th), np.sin(th)])
                w = CL.peso_sector(P, vert, ur, R, g["apert"]).reshape(hh, ww)
                for nom, cu in (("d0", cuerpo), ("d%d" % DILATA, cuerpo_d)):
                    m = (w > 0) & (~cu)
                    n = int(m.sum())
                    fila["perfil"].setdefault(nom, {})[int(rot)] = {
                        "n": n,
                        "frac_solido": float((alfa[m] >= ALFA_SOLIDO).mean()) if n else float("nan"),
                        "alfa_medio": float(alfa[m].mean()) if n else float("nan")}
            filas.append(fila)
            if k in (kt, kt - 4, kt - 8):
                guardar[(k, lado)] = {"alfa": alfa, "seg": seg,
                                      "recorte": (cx0, cy0, cx1, cy1)}
    cap.release()
    dt = time.perf_counter() - t0
    print("  %d recortes medidos en %.1f s (GPU COMPARTIDA: el tiempo es indicativo)"
          % (len(filas), dt))

    val = [f for f in filas if not f["contaminado"]]
    print("  GUARDIA de un tirador por recorte: alfa medio del recorte mediana %.4f  ·  "
          "descartados por alfa < %.2f ... %d de %d" % (
              float(np.median([f["alfa_recorte"] for f in filas])) if filas else float("nan"),
              GUARDIA_ALFA, len(filas) - len(val), len(filas)))
    print("  EL CONO CABE SOLO EN PARTE: R/hoja mediana %.2f  min %.2f  ->  se prueba de %.0f a "
          "%.0f cm de los 90" % (
              float(np.median([f["R_frac_hoja"] for f in val])) if val else float("nan"),
              float(np.min([f["R_frac_hoja"] for f in val])) if val else float("nan"),
              90 * float(np.min([f["R_frac_hoja"] for f in val])) if val else float("nan"),
              90 * float(np.median([f["R_frac_hoja"] for f in val])) if val else float("nan")))

    for dil in ("d0", "d%d" % DILATA):
        print()
        print("  --- cuerpo %s   (%s)" % (
            dil, "sin dilatar" if dil == "d0" else "dilatado %d px" % DILATA))
        print("  %-6s %-4s %8s %8s %8s %8s %8s %6s %s" % (
            "cuadro", "lado", "n_0", "sol_0", "sol_med", "sol_max", "razon", "rank", "cumple"))
        cumplen, n_eval = 0, 0
        for f in val:
            p = f["perfil"][dil]
            n0 = p[0]["n"]
            if n0 < 20:
                continue
            usar = [r for r in ROTS if p[int(r)]["n"] >= MIN_AREA_REL * n0]
            if len(usar) < 8:
                continue
            v = np.array([p[int(r)]["frac_solido"] for r in usar])
            if not np.isfinite(v).all():
                continue
            v0 = p[0]["frac_solido"]
            med = float(np.median(v))
            razon = v0 / med if med > 0 else float("inf")
            rank = int(1 + (v > v0).sum())
            ok1 = rank <= max(1, int(np.ceil(0.10 * len(usar))))
            ok2 = razon >= 2.0
            n_eval += 1
            cumplen += int(ok1 and ok2)
            print("  %-6d %-4d %8d %8.4f %8.4f %8.4f %8.2f %3d/%-2d %s" % (
                f["cuadro"], f["lado"], n0, v0, med, float(v.max()), razon, rank, len(usar),
                "SI" if (ok1 and ok2) else ("no" if not ok1 else "no (razon)")))
        print("  CUMPLEN EL CRITERIO DECLARADO ... %d de %d cuadros-tirador evaluados (%.1f %%)" % (
            cumplen, n_eval, 100.0 * cumplen / max(n_eval, 1)))
        print("  VEREDICTO: %s" % (
            "matting SI capta algo alineado con el eje" if cumplen > 0.5 * max(n_eval, 1) else
            "matting NO discrimina la hoja dentro del cono: el cono en su sitio no se distingue "
            "de los rotados"))
        todo.setdefault(clip, {})[dil] = {"cumplen": cumplen, "n_eval": n_eval}

    np.savez_compressed(PRUEBAS / ("cono_matting_%s.npz" % clip),
                        filas=np.asarray(json.dumps(filas, default=float)),
                        **{("g_%d_%d_%s" % (k, l, n)): v[n]
                           for (k, l), v in guardar.items() for n in ("alfa", "seg")},
                        **{("r_%d_%d" % (k, l)): np.asarray(v["recorte"])
                           for (k, l), v in guardar.items()})
    print()

(PRUEBAS / "cono_e4_matting.json").write_text(
    json.dumps(todo, indent=1, ensure_ascii=False), encoding="utf-8")
print("guardado %s" % (PRUEBAS / "cono_e4_matting.json"))

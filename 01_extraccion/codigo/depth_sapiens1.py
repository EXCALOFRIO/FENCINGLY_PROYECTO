"""La cabeza `depth` de Sapiens 1 ordena la profundidad mejor que el `pointmap` de Sapiens 2?

POR QUE SE PRUEBA. Sapiens 2 tiene cinco cabezas (`pose`, `seg`, `normal`, `pointmap`, `matting`) y
**ninguna es `depth`**: comprobado en el indice publico, Sapiens 1 si tenia una cabeza `depth`
dedicada y en Sapiens 2 la sustituyo `pointmap`. Y nuestro `pointmap` **fallo su control fisico**:
separa a los dos tiradores entre si mas que a la pareja del arbitro (razon 0,61 y 1,08, pasando de 1
en 7 de 17 y 6 de 10 cuadros), mientras que la `y del pie`, que es gratis, acierta 28 de 28.

Asi que la pregunta es concreta: la cabeza vieja pasa el control que la nueva no pasa?

CRITERIO DECLARADO ANTES DE EJECUTAR. Por cada cuadro con los dos tiradores Y al menos una persona
ajena en el plano:

  `dentro` = |z del tirador izquierdo - z del tirador derecho|
  `fuera`  = distancia en z al ajeno mas cercano de los dos
  `razon`  = dentro / fuera

  La cabeza SIRVE si `razon` se queda por debajo de 1 en la gran mayoria de los cuadros: significa
  que los dos tiradores estan mas cerca entre si que de cualquier otra persona, que es lo que hace
  falta para poder decir "estos dos estan en la misma pista y aquel no".
  Si `razon` pasa de 1 con la misma frecuencia que el `pointmap`, no aporta nada y se descarta.

SE COMPARA CONTRA LA LINEA BASE GRATIS, la `y del pie`, con la misma formula. Una cabeza de 1,3 GB
que no bate a una resta de dos numeros no merece estar en la tuberia.

CONTROL DE QUE EL PREPROCESO ES CORRECTO, porque si la entrada esta mal el mapa sale a ruido y la
medida no valdria nada: se comprueba que la z dentro de la caja del tirador se separa de la z del
fondo. Si no se separa, se dice y no se sigue.
"""
from __future__ import annotations

import os
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import cv2
import torch

import extrae_pareja as EP

SCR = pathlib.Path(os.environ.get(
    "SCRATCH", r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\187cfd83-d8d9-4c4d-bed4-f255724f6b21\scratchpad"))
PESOS = SCR / "sapiens1_depth" / "sapiens_0.3b_render_people_epoch_100_torchscript.pt2"
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\depth1")
VIDEOS = "D:/FENCINGLY_VIDEOS_RAW"
ENTRADA = (1024, 768)
"""Alto x ancho que espera Sapiens 1. No es negociable: el modelo es torchscript y la forma esta
fijada en el grafo."""
MEDIA = np.array([123.675, 116.28, 103.53], dtype="float32")
DESV = np.array([58.395, 57.12, 57.375], dtype="float32")
L_HOM, R_HOM, L_CAD, R_CAD = 5, 6, 11, 12
"""Indices del esqueleto de 17 puntos de YOLO, que es lo que hay cacheado. Ojo: NO son los de
Sapiens 2, que tiene 308 y otro reparto."""


def carga():
    m = torch.jit.load(str(PESOS))
    return m.to("cuda").eval()


RELLENO = True
"""Meter el cuadro con RELLENO en vez de estirado. Con estirado, un cuadro 16:9 metido en una
entrada 3:4 sale deformado y se ven franjas de ruido en los bordes laterales del mapa. Es un defecto
de mi preproceso, no del modelo, y un veredicto sacado de ahi estaria confundido."""


def _mete(rgb):
    """QUE HACE: coloca el cuadro en la entrada del modelo conservando la proporcion."""
    H, W = rgb.shape[:2]
    hh, ww = ENTRADA
    if not RELLENO:
        return cv2.resize(rgb, (ww, hh), interpolation=cv2.INTER_LINEAR), (0, 0, 1.0)
    esc = min(ww / float(W), hh / float(H))
    nw, nh = int(round(W * esc)), int(round(H * esc))
    peq = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    lienzo = np.zeros((hh, ww, 3), dtype="float32")
    ox, oy = (ww - nw) // 2, (hh - nh) // 2
    lienzo[oy:oy + nh, ox:ox + nw] = peq
    return lienzo, (ox, oy, esc)


def profundidad(mod, bgr):
    """QUE HACE: mapa de profundidad del cuadro completo, devuelto al tamanio original.

    Se corre de CUADRO COMPLETO y no por recorte a proposito: lo que se quiere es ORDENAR a la gente
    entre si, y para eso todos tienen que compartir la misma escala desconocida. Medido en el
    `pointmap`, dentro de un recorte la z no ordena nada (el control del suelo salio al 53 %, moneda
    al aire) y de cuadro completo si.
    """
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype("float32")
    ent, (ox, oy, esc) = _mete(rgb)
    ent = (ent - MEDIA) / DESV
    t = torch.from_numpy(ent.transpose(2, 0, 1)[None]).to("cuda")
    with torch.inference_mode():
        o = mod(t)
    z = o[0, 0].float().cpu().numpy() if o.ndim == 4 else np.asarray(o.float().cpu())
    if RELLENO:
        """Se recorta el relleno ANTES de devolver al tamanio original: si no, se estaria
        interpolando el borde negro dentro de la imagen."""
        nh, nw = int(round(H * esc)), int(round(W * esc))
        z = z[oy:oy + nh, ox:ox + nw]
    return cv2.resize(z, (W, H), interpolation=cv2.INTER_LINEAR)


def z_del_torso(Z, caja, kp=None, H=None, W=None):
    """QUE HACE: z mediana del torso, con una ventana por punto y no un pixel.

    Un pixel de una z ruidosa no dice nada. Si no hay puntos de torso fiables se cae al centro de la
    caja, que es peor pero es algo.
    """
    pts = []
    if kp is not None:
        for j in (L_HOM, R_HOM, L_CAD, R_CAD):
            if kp[j, 2] >= 0.3 and np.isfinite(kp[j, 0]):
                pts.append((int(np.clip(kp[j, 1], 0, H - 1)), int(np.clip(kp[j, 0], 0, W - 1))))
    if len(pts) < 2:
        cy = int(np.clip(0.5 * (caja[1] + caja[3]), 0, Z.shape[0] - 1))
        cx = int(np.clip(0.5 * (caja[0] + caja[2]), 0, Z.shape[1] - 1))
        pts = [(cy, cx)]
    vals = []
    for (yy, xx) in pts:
        y0, y1 = max(0, yy - 4), min(Z.shape[0], yy + 5)
        x0, x1 = max(0, xx - 4), min(Z.shape[1], xx + 5)
        v = Z[y0:y1, x0:x1]
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(float(np.median(v)))
    return float(np.median(vals)) if vals else float("nan")


def main():
    clips = sys.argv[1:] or ["defeatedHarlequinRabbitSkinkLizard"]
    SALIDA.mkdir(parents=True, exist_ok=True)
    mapa = {}
    for raiz, _d, fs in os.walk(VIDEOS):
        for f in fs:
            if f.lower().endswith(".mp4"):
                mapa.setdefault(pathlib.Path(f).stem, os.path.join(raiz, f))
    mod = carga()
    print("modelo cargado en %s" % next(mod.parameters()).device)
    print()
    print("%-34s %6s %9s %9s %9s %9s" % (
        "clip", "casos", "z<1 dep", "z<1 pie", "razon dep", "razon pie"))
    print("-" * 82)
    for cid in clips:
        ca = SCR / ("det_%s.pkl" % cid)
        if not ca.exists():
            print("%-34s sin cache de detecciones" % cid[:34])
            continue
        with open(ca, "rb") as fh:
            cajas, cuadros, W, H, T = pickle.load(fh)
        r = EP.procesa(cajas, cuadros, W, H, T)
        B, pres, mask = r["B"], r["pres"], r["mask"]
        cap = cv2.VideoCapture(mapa[cid])
        rz_d, rz_p, sep_ok = [], [], []
        k = 0
        primero = True
        while True:
            ok, im = cap.read()
            if not ok:
                break
            usar = mask[k] and pres[k, 0] and pres[k, 1] and (k % 12 == 0)
            if usar:
                en = np.nonzero(cuadros == k)[0]
                prop = []
                for lado in (0, 1):
                    b = B[k, lado] * np.array([W, H, W, H])
                    prop.append(b)
                ajenos = []
                for i in en:
                    b = np.asarray(cajas[i], dtype="float64")
                    """Ajeno = deteccion que no es ninguno de los dos elegidos."""
                    if max(EP.RD._iou(b, prop[0]), EP.RD._iou(b, prop[1])) > 0.30:
                        continue
                    ajenos.append(b)
                if ajenos:
                    Z = profundidad(mod, im)
                    if primero:
                        """Control de preproceso: la z del tirador se separa del fondo?"""
                        b = prop[0]
                        y0, y1 = int(max(0, b[1])), int(min(H, b[3]))
                        x0, x1 = int(max(0, b[0])), int(min(W, b[2]))
                        dentro = Z[y0:y1, x0:x1]
                        fuera = np.concatenate([Z[:20].ravel(), Z[-20:].ravel()])
                        dentro = dentro[np.isfinite(dentro)]
                        fuera = fuera[np.isfinite(fuera)]
                        print("   control de preproceso en %s c%d: z del tirador %.3f, "
                              "z del borde de imagen %.3f, separacion %.3f" % (
                                  cid[:20], k, np.median(dentro), np.median(fuera),
                                  abs(np.median(dentro) - np.median(fuera))))
                        vis = cv2.applyColorMap(
                            (255 * (np.clip((Z - np.percentile(Z, 2))
                                            / max(np.percentile(Z, 98) - np.percentile(Z, 2), 1e-6),
                                            0, 1))).astype("uint8"), cv2.COLORMAP_JET)
                        cv2.imwrite(str(SALIDA / ("%s_c%04d_depth1.png" % (cid[:26], k))),
                                    cv2.resize(vis, (1100, int(1100 * H / W))))
                        primero = False
                    za = z_del_torso(Z, prop[0])
                    zb = z_del_torso(Z, prop[1])
                    zj = [z_del_torso(Z, b) for b in ajenos]
                    zj = [v for v in zj if np.isfinite(v)]
                    if np.isfinite(za) and np.isfinite(zb) and zj:
                        dentro = abs(za - zb)
                        fuera = min(min(abs(za - v), abs(zb - v)) for v in zj)
                        if fuera > 1e-9:
                            rz_d.append(dentro / fuera)
                    pa, pb = prop[0][3] / H, prop[1][3] / H
                    pj = [b[3] / H for b in ajenos]
                    d2 = abs(pa - pb)
                    f2 = min(min(abs(pa - v), abs(pb - v)) for v in pj)
                    if f2 > 1e-9:
                        rz_p.append(d2 / f2)
            k += 1
        cap.release()
        if rz_d:
            rz_d = np.array(rz_d)
            rz_p = np.array(rz_p[:len(rz_d)])
            print("%-34s %6d %8.0f%% %8.0f%% %9.2f %9.2f" % (
                cid[:34], len(rz_d), 100 * (rz_d < 1).mean(), 100 * (rz_p < 1).mean(),
                np.median(rz_d), np.median(rz_p)))
        else:
            print("%-34s sin cuadros utilizables con ajeno en el plano" % cid[:34])
    print()
    print("  `z<1` es la fraccion de cuadros donde los dos tiradores quedan mas cerca entre si que")
    print("  de cualquier otra persona. Cuanto mas alto, mejor ordena.")
    print("  imagenes en %s" % SALIDA)


if __name__ == "__main__":
    main()

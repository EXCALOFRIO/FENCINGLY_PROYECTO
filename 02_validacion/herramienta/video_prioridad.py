"""Video de revision de la PRIORIDAD: quien la tiene, por que regla, y con que confianza.

PARA QUE SIRVE: para poder contradecir al algoritmo. Una tabla con un 66,5 % no dice en que cuadro
se equivoco ni por que; esto si, porque cada cuadro lleva escrita **la regla FIE que decidio**.

QUE SE VE, y todo esta ahi para poder discutirlo:
  esqueleto VERDE       el tirador de la izquierda (lado 0), codigo de color del proyecto
  esqueleto NARANJA     el de la derecha (lado 1)
  brazo ARMADO grueso   el segmento `codo -> raiz de la mano`, que es el eje del que sale todo
  caja gruesa           la lleva el que TIENE la prioridad; el otro va con caja fina
  banda de arriba       quien la tiene y **el articulo que la concede**, en su color
  barra de confianza    a la derecha de la banda; corta = decision apretada
  VIOLETA y rotulo      pose inventada: no puede sostener una decision y se dice en el cuadro
  grafico de abajo      `ext_hr` de los dos tiradores con el cursor en el cuadro actual, y una marca
                        vertical en cada cambio de prioridad. Es lo que permite ver si el cambio
                        cae donde la curva se mueve o donde no se mueve nada.
  banda roja            cuadro no utilizable (primer plano, repeticion o fundido)

NO NECESITA GPU: las poses ya estan en `01_extraccion/pruebas/poses_*.npz`. Corre con el interprete
de ultralytics, que es el que tiene `cv2` y `scipy` a la vez.

  python video_prioridad.py CLIP [CLIP...]
"""
from __future__ import annotations

import os
import pathlib
import sys

import cv2
import numpy as np

CODIGO = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
sys.path.insert(0, str(CODIGO))
import p1_variables as P1
import prioridad as PR

PRUEBAS = P1.PRUEBAS
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\prioridad\videos")
COL = {0: (60, 240, 60), 1: (60, 160, 255), -1: (170, 170, 170)}
NOM = {0: "IZQUIERDA", 1: "DERECHA", -1: "SIN PRIORIDAD"}
CONF_PINTA = 0.3
HUESOS = [(5, 6), (9, 10), (5, 9), (6, 10), (6, 8), (8, 21), (5, 7), (7, 42),
          (9, 11), (11, 13), (10, 12), (12, 14)]
CARA_DESDE = 63


def _pinta(im, kp, sc, col, armado=None, grueso=2):
    def val(i):
        return sc[i] >= CONF_PINTA and np.isfinite(kp[i, 0]) and np.isfinite(kp[i, 1])

    for a, b in HUESOS:
        if val(a) and val(b):
            cv2.line(im, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[b, 0]), int(kp[b, 1])),
                     col, grueso)
    for i in range(0, CARA_DESDE):
        if val(i):
            cv2.circle(im, (int(kp[i, 0]), int(kp[i, 1])), 2, col, -1)
    if armado is not None:
        c, r = armado
        if val(c) and val(r):
            cv2.line(im, (int(kp[c, 0]), int(kp[c, 1])), (int(kp[r, 0]), int(kp[r, 1])),
                     (60, 240, 255), grueso + 3)
            cv2.circle(im, (int(kp[r, 0]), int(kp[r, 1])), 5, (60, 240, 255), 2)


def _grafico(s, prio, k, W, alto=96):
    """La curva de `ext_hr` de los dos, con el cursor y los cambios de prioridad marcados.

    Se pinta la serie ENTERA y no una ventana movil a proposito: asi se ve de un golpe si el clip
    tiene una accion o veinte, y donde caen los cambios respecto a las subidas del brazo.
    """
    g = np.full((alto, W, 3), 18, dtype="uint8")
    T = s["T"]
    x = np.linspace(0, W - 1, T).astype(int)
    lo, hi = 0.3, 2.0
    for lado in (0, 1):
        y = s["ext_hr"][:, lado]
        v = s["val"][:, lado]
        pts = []
        for j in range(T):
            if not (np.isfinite(y[j]) and v[j]):
                pts.append(None)
                continue
            yy = int(alto - 4 - (np.clip(y[j], lo, hi) - lo) / (hi - lo) * (alto - 10))
            pts.append((x[j], yy))
        for j in range(1, T):
            if pts[j] and pts[j - 1]:
                cv2.line(g, pts[j - 1], pts[j], COL[lado], 1)
    for j in range(1, T):
        if prio[j] != prio[j - 1]:
            cv2.line(g, (x[j], 0), (x[j], alto), COL[int(prio[j])], 1)
    cv2.line(g, (x[k], 0), (x[k], alto), (255, 255, 255), 1)
    yy = int(alto - 4 - (PR.EXT_ALTO - lo) / (hi - lo) * (alto - 10))
    cv2.line(g, (0, yy), (W, yy), (90, 90, 90), 1)
    cv2.putText(g, "ext_hr (hombro->mano / torso); la raya gris es EXT_ALTO=%.2f" % PR.EXT_ALTO,
                (6, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    return g


def uno(cid):
    with np.load(PRUEBAS / ("poses_%s.npz" % cid), allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    s = PR.senales(d)
    W, H, T = int(d["W"]), int(d["H"]), int(d["T"])
    mask, origen = d["mask"], d["origen"]
    KP, SC, B, Be = d["KP"], d["SC"], d["B"], d["Be"]
    video = str(d["video"])
    if not os.path.exists(video):
        print("SIN VIDEO %s" % cid)
        return
    r = PR.decide(s, mask, T)
    par = {c["cuadro"]: c for c in PR.paradas(s, mask, T, r)}
    prio, regla, conf = r["prio"], r["regla"], r["conf"]
    armado = {l: (PR.CL.MANOS[s["mano"][l]]["codo"], PR.CL.MANOS[s["mano"][l]]["root"])
              for l in (0, 1)}
    SALIDA.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out = cv2.VideoWriter(str(SALIDA / ("%s.mp4" % cid)), cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (W, H + 96 + 40))
    k = 0
    while True:
        ok, im = cap.read()
        if not ok or k >= T:
            break
        vis = im.copy()
        p = int(prio[k])
        if mask[k]:
            for lado in (0, 1):
                if origen[k, lado] < 0:
                    continue
                inv = origen[k, lado] > 0
                col = VR.VIOLETA if inv else COL[lado]
                _pinta(vis, KP[k, lado], SC[k, lado], col, armado[lado],
                       grueso=3 if lado == p else 1)
                b = Be[k, lado] if inv else B[k, lado]
                if np.isfinite(b).all():
                    cv2.rectangle(vis, (int(b[0] * W), int(b[1] * H)),
                                  (int(b[2] * W), int(b[3] * H)), col,
                                  3 if lado == p else 1)
                    et = ["IZQUIERDA", "DERECHA"][lado]
                    if inv:
                        et += " POSE INVENTADA: no decide"
                    cv2.putText(vis, et, (int(b[0] * W), max(14, int(b[1] * H) - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        else:
            cv2.rectangle(vis, (0, 40), (W, 76), (0, 0, 160), -1)
            cv2.putText(vis, "FUERA: primer plano, repeticion o fundido; este cuadro no se usa",
                        (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if k in par:
            c = par[k]
            cv2.rectangle(vis, (0, H - 40), (W, H), (40, 40, 140), -1)
            cv2.putText(vis, "CANDIDATO A PARADA (t.86): para el %s, atacaba el %s. Sin ver la hoja."
                        % (NOM[c["paraba"]], NOM[c["atacaba"]]), (10, H - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
        cab = np.full((40, W, 3), 14, dtype="uint8")
        cv2.rectangle(cab, (0, 0), (14, 40), COL[p], -1)
        cv2.putText(cab, "%s  c%d/%d   PRIORIDAD: %s" % (cid[:22], k, T - 1, NOM[p]),
                    (22, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL[p], 1)
        cv2.putText(cab, regla[k][:104], (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (225, 225, 225), 1)
        ancho = int(120 * float(conf[k]))
        cv2.rectangle(cab, (W - 130, 8), (W - 130 + ancho, 22), COL[p], -1)
        cv2.rectangle(cab, (W - 130, 8), (W - 10, 22), (110, 110, 110), 1)
        cv2.putText(cab, "confianza %.2f" % conf[k], (W - 130, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        out.write(np.vstack([cab, vis, _grafico(s, prio, k, W)]))
        k += 1
    cap.release()
    out.release()
    ev = [(e[0], NOM[e[1]], e[2][:44]) for e in r["eventos"]]
    print("%-38s %d cuadros, %d eventos de prioridad, %d candidatos a parada" % (
        cid[:38], k, len(r["eventos"]), len(par)))
    for e in ev:
        print("    c%-4d -> %-14s %s" % e)
    print("    video -> %s" % (SALIDA / ("%s.mp4" % cid)))


if __name__ == "__main__":
    for c in sys.argv[1:]:
        uno(c)

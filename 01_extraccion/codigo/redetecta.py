"""Recupera al tirador que falta REDETECTANDOLO en un recorte, en vez de inventarle la pose.

LA IDEA, y por que es mejor que interpolar puntos. El usuario pidio que no falte nunca ninguno de
los dos, inventando el que falte entre la ultima pose conocida y la siguiente. Interpolar la POSE es
inventar datos. Pero hay un paso intermedio que no inventa nada:

  la CAJA se interpola, que es barato y robusto porque es una sola posicion y un solo tamanio
  y la POSE se MIDE sobre los pixeles reales de esa caja

Asi la pose sigue siendo un dato medido y solo la ventana donde mirar es estimada. La interpolacion
de puntos queda como ultimo recurso, para cuando el tirador de verdad no esta en el cuadro.

Y hay motivo para esperar que funcione: el detector de cuadro completo pierde al tirador cuando es
pequenio o esta cortado por el borde del encuadre. En un recorte alrededor de donde deberia estar, el
mismo detector lo ve con mucha mas resolucion efectiva.

QUE FALTA TODAVIA, medido en 5 clips despues de arreglar la puerta de planos y la fusion:

  | clip                  | faltan | de borde | interior |
  |-----------------------|-------:|---------:|---------:|
  | `RichDefensive`       |     30 |        2 |       28 |
  | `ZanyMeager`          |     54 |       48 |        6 |
  | `VainAching`          |     19 |       19 |        0 |
  | `defeatedHarlequin`   |      1 |        0 |        1 |
  | `HarshExemplary`      |      0 |        0 |        0 |

Los de BORDE son los del principio y el final del clip, donde el tirador aun no ha entrado en plano
o ya ha salido, y ahi no hay dos anclas entre las que interpolar: solo se puede sostener la ultima
caja conocida y comprobar si hay alguien. Los INTERIORES si tienen anclas a los dos lados.
"""
from __future__ import annotations

import numpy as np

MARGEN = 0.45
"""Cuanto se agranda la caja estimada para hacer el recorte. Generoso a proposito: si la caja
estimada esta algo desviada, un recorte justo dejaria al tirador fuera y la redeteccion fallaria por
el encuadre, no por el detector."""
IOU_ACEPTA = 0.30
"""Solape minimo entre lo redetectado y la caja estimada. Por debajo de esto lo encontrado esta en
otro sitio y probablemente es otra persona."""
DPIE_ACEPTA = 0.06
"""La redeteccion tiene que estar a la misma profundidad que el tirador que falta. Es la misma
comprobacion que en la fusion y por la misma medida: dentro de la pareja el pie coincide en 0,004 y
con el arbitro se va a 0,10."""
TALLA_ACEPTA = (0.60, 1.70)


def _iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def estima_cajas(B, pres, T) -> tuple:
    """QUE HACE: rellena la caja de los cuadros que faltan, y dice de que tipo es cada relleno.

    INTERIOR se interpola entre las dos anclas. BORDE sostiene la ancla mas cercana, porque no hay
    nada al otro lado que interpolar y sostener es la unica estimacion que no inventa direccion.

    QUE DEVUELVE: `Be` con las cajas estimadas y `tipo` con 0 medido, 1 interpolado, 2 sostenido.
    """
    Be = np.array(B, dtype="float64", copy=True)
    tipo = np.where(pres, 0, -1).astype("int8")
    for lado in (0, 1):
        hay = np.nonzero(pres[:, lado])[0]
        if not len(hay):
            continue
        for k in range(T):
            if pres[k, lado]:
                continue
            ant = hay[hay < k]
            pos = hay[hay > k]
            if len(ant) and len(pos):
                a, b = int(ant[-1]), int(pos[0])
                t = (k - a) / float(b - a)
                Be[k, lado] = (1 - t) * B[a, lado] + t * B[b, lado]
                tipo[k, lado] = 1
            else:
                anc = int(ant[-1]) if len(ant) else int(pos[0])
                Be[k, lado] = B[anc, lado]
                tipo[k, lado] = 2
    return Be, tipo


def redetecta_cuadro(yolo, frame, caja_est, W, H, imgsz=640):
    """QUE HACE: busca a una persona en el recorte alrededor de la caja estimada.

    QUE DEVUELVE: `(caja, kps, iou)` en coordenadas del cuadro completo, o `None` si no hay nadie
    que cumpla las tres comprobaciones. Devolver `None` es un resultado, no un fallo: significa que
    ahi no esta.
    """
    x0, y0, x1, y1 = [float(v) for v in caja_est]
    aw, ah = (x1 - x0) * W, (y1 - y0) * H
    cx0 = int(max(0, x0 * W - MARGEN * aw))
    cy0 = int(max(0, y0 * H - MARGEN * ah))
    cx1 = int(min(W, x1 * W + MARGEN * aw))
    cy1 = int(min(H, y1 * H + MARGEN * ah))
    if cx1 - cx0 < 24 or cy1 - cy0 < 24:
        return None
    sub = np.ascontiguousarray(frame[cy0:cy1, cx0:cx1])
    r = yolo.predict(sub, imgsz=imgsz, conf=0.05, verbose=False, device=0)[0]
    if r.boxes is None or not len(r.boxes):
        return None
    cajas = r.boxes.xyxy.cpu().numpy().astype("float64")
    kps = (r.keypoints.data.cpu().numpy().astype("float64")
           if r.keypoints is not None else None)
    est = np.array([x0 * W, y0 * H, x1 * W, y1 * H], dtype="float64")
    mejor, mejor_iou, mejor_i = None, -1.0, -1
    for i in range(len(cajas)):
        b = cajas[i].copy()
        b[0] += cx0
        b[2] += cx0
        b[1] += cy0
        b[3] += cy0
        alt_est = max(est[3] - est[1], 1.0)
        razon = (b[3] - b[1]) / alt_est
        if not (TALLA_ACEPTA[0] <= razon <= TALLA_ACEPTA[1]):
            continue
        if abs(b[3] - est[3]) / float(H) > DPIE_ACEPTA:
            continue
        v = _iou(b, est)
        if v > mejor_iou:
            mejor, mejor_iou, mejor_i = b, v, i
    if mejor is None or mejor_iou < IOU_ACEPTA:
        return None
    kp = None
    if kps is not None and mejor_i >= 0:
        kp = kps[mejor_i].copy()
        kp[:, 0] += cx0
        kp[:, 1] += cy0
    return mejor, kp, float(mejor_iou)

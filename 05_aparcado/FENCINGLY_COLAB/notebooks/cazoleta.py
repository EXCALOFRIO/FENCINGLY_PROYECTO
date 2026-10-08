"""Cazoleta: propone candidatos con SAM3 sobre recorte ampliado, filtrando con el prior geometrico.

ES UN PROPONENTE, NO UNA MEDIDA. Sale como columnas con precision SIN VERIFICAR, y su uso previsto es
sembrar un conjunto de anotacion, no alimentar el modelo de prioridad.

LO MEDIDO EL 2026-09-09, y condiciona como se usa:

  QUE FUNCIONA, con el denominador honesto: 120 recortes por palabra, contando como fallo el
  recorte donde no se detecta nada.
    palabra                 dispara   COLOCA a <= 1 diametro
    the hilt of a sword       0,62         0,33
    a banana (senuelo)        0,05         0,04
    a hand (control)          0,72         0,61
    MARGEN sobre el senuelo: +0,29. El senuelo coloca solo el 4 %, asi que el prior geometrico NO
    esta haciendo el trabajo: la senal es del prompt.

  CIFRA RETIRADA: una version anterior de esta nota decia "71 % a 1080p, 53 % a 720p, 25 % a 480p".
  Estaba calculada sobre las cajas DEVUELTAS, con n de 4 a 20 por celda, no sobre los recortes. Con
  n=120 sale 0,20 / 0,55 / 0,23 para 480 / 720 / 1080, o sea que **no hay tendencia limpia por
  resolucion** y la cifra global es 0,33. El defecto era el denominador, como tantas veces aqui.

  QUE NO FUNCIONA, Y HAY QUE SABERLO ANTES DE CONSTRUIR ENCIMA
    1. SIN RECORTAR NO HAY NADA. En el cuadro completo de 1920 todas las palabras del florete
       devuelven ~0 instancias a 0,03-0,13, mientras `a person` devuelve 2-23 a 0,96-0,97. El fallo
       es de ESCALA, no de vocabulario.
    2. LA PUNTUACION NO SIRVE PARA FILTRAR: los aciertos promedian 0,238 y los fallos 0,253. Los
       fallos puntuan MAS. Se filtra con el prior geometrico, nunca con la confianza.
    3. LA FORMULACION MANDA MAS QUE EL MODELO. `a bell guard`, que es el termino tecnico correcto,
       devuelve EXACTAMENTE CERO en todas las sedes. Nunca concluir de una sola palabra.
    4. Y no se puede validar con consistencia temporal: sembrar SAM2-video en el sitio correcto pasa
       un criterio de consistencia en 11 de 12 casos, y sembrarlo en un SENUELO dentro del cuerpo
       pasa en 10 de 12. La propagacion rastrea suavemente lo que sea, asi que un rastro suave no es
       prueba de haber encontrado nada.

LA HOJA ES OTRA COSA. El fuerte son ~4 mm, o sea 0,64 px a 720p y 0,96 a 1080p: por debajo de un
pixel. Eso NO la hace indetectable —el contraste deja una cresta de intensidad y a ojo se ve como una
raya fina contra fondo oscuro— pero si hace que no haya region que segmentar. Este modulo no intenta
la hoja: intenta la cazoleta, que son 19-29 px de metal.
"""
from __future__ import annotations

import numpy as np

DIAM_M = 0.12
ALTURA_M = 1.75
FRAC_CAJA = 0.388
L_WRI, R_WRI, L_ELB, R_ELB = 9, 10, 7, 8
CONF_KP_MIN = 0.25
AMPLIA = 8.0
PALABRA = "the hilt of a sword"
PALABRA_SENUELO = "a banana"
TOL_DIAM = 1.0
UMBRAL = 0.15

PRECISION_MEDIDA = {1080: 0.71, 720: 0.53, 480: 0.25}
"""Fraccion de cajas que caen a <= 1 diametro del punto esperado, por alto de imagen. Sin verificacion
humana: el punto esperado es un prior, no verdad-terreno."""


def punto_esperado(poses_t, lado: int, W: int, H: int) -> tuple | None:
    """QUE HACE: el punto donde deberia estar la cazoleta, desde la pose.

    Muneca armada = la mas adelantada hacia el rival, mas 0,6 diametros a lo largo del antebrazo.
    Las figuras confirman que este prior queda bien centrado sobre la cazoleta, y es lo que se usa
    para filtrar, porque la confianza del modelo no discrimina.

    AVISO: elegir la muneca "por la mas adelantada" es un apano. `clip_stats.dominant_hand` lo
    resuelve mejor con dos indicios y devuelve confianza; midiendolo, el apano elige la muneca
    EQUIVOCADA en el 31,3 % de los casos. Si tienes `dominant_hand`, pasa su muneca en `muneca_idx`.
    """
    P = np.asarray(poses_t, dtype="float64")
    wl, wr, el, er = P[L_WRI], P[R_WRI], P[L_ELB], P[R_ELB]
    if not (np.isfinite(wl).all() and np.isfinite(wr).all()):
        return None
    sgn = 1.0 if lado == 0 else -1.0
    der = (sgn * (wr[0] - wl[0])) > 0
    wri, elb = (wr, er) if der else (wl, el)
    if not (np.isfinite(wri).all() and np.isfinite(elb).all()):
        return None
    if wri[2] < CONF_KP_MIN or elb[2] < CONF_KP_MIN:
        return None
    wx, wy = wri[0] * W, wri[1] * H
    v = np.array([wx - elb[0] * W, wy - elb[1] * H])
    nv = float(np.linalg.norm(v))
    u = v / nv if nv > 1e-6 else np.array([sgn, 0.0])
    diam = DIAM_M * (H * FRAC_CAJA) / ALTURA_M
    return (wx + u[0] * diam * 0.6, wy + u[1] * diam * 0.6, diam, float(u[0]), float(u[1]))


def propone(imagen, poses_t, lado, proc, modelo, palabra: str = PALABRA,
            umbral: float = UMBRAL) -> dict:
    """QUE HACE: recorta alrededor del punto esperado, amplia, pide `palabra` a SAM3 y devuelve la
    caja mas cercana al prior.

    QUE DEVUELVE: `guard_x`, `guard_y` en pixeles de la imagen original, `guard_score` (que NO sirve
    para filtrar, ver el encabezado), `guard_dist_diam` = distancia al prior en diametros, y
    `guard_ok` = si esa distancia baja de `TOL_DIAM`. Todo `NaN` si no se pudo medir.
    """
    import cv2
    import torch

    vacio = {"guard_x": float("nan"), "guard_y": float("nan"), "guard_score": float("nan"),
             "guard_dist_diam": float("nan"), "guard_ok": False, "guard_n": 0}
    im = np.asarray(imagen)
    H, W = im.shape[:2]
    pe = punto_esperado(poses_t, lado, W, H)
    if pe is None:
        return vacio
    cx, cy, diam, _ux, _uy = pe
    rad = max(int(diam * 2.2), 26)
    x0, y0 = int(max(0, cx - rad)), int(max(0, cy - rad))
    x1, y1 = int(min(W, cx + rad)), int(min(H, cy + rad))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return vacio
    gr = cv2.resize(im[y0:y1, x0:x1], None, fx=AMPLIA, fy=AMPLIA, interpolation=cv2.INTER_CUBIC)
    rgb = np.ascontiguousarray(gr[:, :, ::-1]) if gr.shape[-1] == 3 else gr

    with torch.inference_mode():
        ent = proc(images=rgb, text=palabra, return_tensors="pt").to(
            next(modelo.parameters()).device)
        out = modelo(**ent)
        res = proc.post_process_instance_segmentation(
            out, threshold=umbral, mask_threshold=0.5,
            target_sizes=[(gr.shape[0], gr.shape[1])])[0]
    cajas, punt = res.get("boxes"), res.get("scores")
    cajas = cajas.cpu().numpy() if cajas is not None and len(cajas) else np.zeros((0, 4))
    punt = punt.cpu().numpy() if punt is not None and len(punt) else np.zeros((0,))
    if not len(cajas):
        return dict(vacio, guard_n=0)

    pex, pey = (cx - x0) * AMPLIA, (cy - y0) * AMPLIA
    mejor, dmin = None, 1e18
    for b, s in zip(cajas, punt):
        bx, by = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        dd = float(np.hypot(bx - pex, by - pey)) / (diam * AMPLIA)
        if dd < dmin:
            dmin, mejor = dd, (bx, by, float(s))
    bx, by, s = mejor
    return {"guard_x": round(bx / AMPLIA + x0, 2), "guard_y": round(by / AMPLIA + y0, 2),
            "guard_score": round(s, 3), "guard_dist_diam": round(dmin, 3),
            "guard_ok": bool(dmin <= TOL_DIAM), "guard_n": int(len(cajas))}


def carga_sam3(ruta_pesos):
    """QUE HACE: carga SAM3. El repo es `facebook/sam3` y esta CERRADO con lista de autorizacion:
    hay que pedir acceso en su pagina con la cuenta que tenga el token, o la descarga da 403."""
    import torch
    from transformers import Sam3Model, Sam3Processor

    proc = Sam3Processor.from_pretrained(str(ruta_pesos))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = Sam3Model.from_pretrained(str(ruta_pesos), dtype=torch.float32).to(dev).eval()
    print("  SAM3 cargado en %s" % dev)
    return proc, modelo


def control(imagen, poses_t, lado, proc, modelo) -> dict:
    """QUE HACE: el senuelo verbal en el MISMO recorte. Sin esto una tasa de disparo no dice nada,
    porque cualquier palabra puede aterrizar donde esta la accion."""
    r = propone(imagen, poses_t, lado, proc, modelo, palabra=PALABRA_SENUELO)
    return {"senuelo_n": r["guard_n"], "senuelo_ok": r["guard_ok"]}

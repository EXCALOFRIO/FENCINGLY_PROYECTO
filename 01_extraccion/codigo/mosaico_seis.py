"""Mosaico de SEIS celdas por clip: las cuatro capas densas, el esqueleto y la deteccion del florete.

PARA QUE, con las palabras del usuario: "puedes ensenarme ejemplos de las 4 capas que hay tipo la del
esqueleto, la de profundidad, la de segmentacion y las otras, y lo del florete tambien, como lo
detecta, a ver si es verdad que esta limpio". O sea: esto NO calcula nada nuevo, es la superficie de
comprobacion de lo que ya esta medido. Si en una celda queda arbitro, publico o suelo, se tiene que
ver a ojo sin que nadie lo explique.

LAS SEIS CELDAS
===============
  1  original con las CAJAS y los DOS ESQUELETOS: verde el tirador de la izquierda, naranja el de la
     derecha, y las detecciones NO elegidas en rojo. Las manos son 21 puntos cada una y a esta escala
     se pierden, asi que la celda lleva dos RECORTES DE LA MANO ARMADA a resolucion nativa pegados
     abajo. Sin ellos la celda no ensena lo que el usuario ha pedido ver.
  2  `seg` limpio          3  `normal` limpio          4  `matting` limpio
  5  `pointmap` de CUADRO COMPLETO, con el rotulo de que NO VALE COMO PROFUNDIDAD puesto en la propia
     celda: fallo su control fisico (separa a los dos tiradores entre si mas que a la pareja del
     arbitro, razon 0,61 y 1,08 donde deberia ser >> 1) y la y del pie, que cuesta cero, acierta
     28 de 28. Se ensena porque el usuario ha pedido verla, no porque se use.
  6  EL FLORETE: el alfa de `matting` en escala log, los pixeles por encima del p99,5 del alfa fuera
     del cuerpo, la recta detectada y el cono desde el eje del brazo, con el CONO GIRADO 90 GRADOS
     dibujado en gris al lado y su cifra en el pie. El control girado es lo que separa "detecta la
     hoja" de "detecta cualquier cosa": si el cono en su sitio no captura mas pixeles que el girado,
     la deteccion no tiene nada que ver con el brazo.

En las celdas 2 a 5 TODO lo que quede fuera de las dos mascaras sale NEGRO. Eso es lo que hace que un
resto de arbitro o de suelo salte a la vista en vez de disimularse contra el fondo real.

NADA DE ESTO SE REIMPLEMENTA AQUI, y es a proposito
===================================================
  `capas_limpias`        la mascara limpia (matting | seg) & esqueleto mas cercano & guarda de caja,
                         y sus metricas con control negativo. Fuga de arbitro medida 0,0000.
  `extrae_pareja`        via `capas_limpias.elige_pareja_del_clip`: la seleccion de la pareja y la
                         mascara de cuadros utilizables.
  `cono_lib`             el mapa de indices MEDIDO (la cara empieza en el 63 y no se pinta) y la
                         eleccion de mano armada.
  `e7_pinta_cono`        `recorte`, `detecta` (el umbral de la hoja), `dibuja_cono` y `celda`.
  `video_revision`       `HUESOS`, `PULGARES`, `MANOS` y el criterio de pintado del esqueleto.

LOS CUADROS DESCARTADOS SE PINTAN, NO SE SALTAN
===============================================
`extrae_pareja` devuelve `mask` con los cuadros que la tuberia NO usa (repeticiones a camara lenta,
primeros planos, fundidos). Si el mosaico se limitara a los cuadros buenos, el video daria la
impresion de que la tuberia procesa todo el clip. Asi que un cuadro con `mask=False` se pinta con su
banda roja y las otras cinco celdas en negro. En `HarshExemplaryGermanspaniel` eso son sus primeros
cuadros, que son una repeticion en primer plano.

USO:  python mosaico_seis.py CLIP [PASO]      con el interprete de transformers + torch + CUDA
"""
from __future__ import annotations

import json
import os
import pathlib
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import capas_limpias as CL
"""Importar `capas_limpias` es lo que fija HF_HOME, y tiene que pasar antes de transformers."""
import cono_lib as CO
import e7_pinta_cono as E7

HERRAM = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\herramienta")
sys.path.insert(0, str(HERRAM))
import video_revision as VR

import cv2

SCR = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\187cfd83-d8d9-4c4d-bed4-f255724f6b21\scratchpad")
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\mosaicos")
FIJOS = SALIDA / "fijos"
"""Los cuadros fijos van DENTRO del proyecto y no en el scratchpad: son la evidencia que se mira
cuando el video se ve borroso al reducirlo, y el scratchpad es de esta sesion."""
SALIDA.mkdir(parents=True, exist_ok=True)
FIJOS.mkdir(parents=True, exist_ok=True)
VIDEOS = "D:/FENCINGLY_VIDEOS_RAW"

CLIP = sys.argv[1]
PASO = int(sys.argv[2]) if len(sys.argv) > 2 else 5
CELDA = 960
"""960 y no 640: la celda 1 lleva un cuadro de 1920 dentro, o sea una reduccion de 2x. A 640 la
reduccion es de 3x y el esqueleto de la mano desaparece, que es justo lo que hay que ver."""
AP = 25.4
"""Semiapertura del cono, en grados. Es el p90 MEDIDO de |angulo de la hoja detectada - eje del
antebrazo| a 1080p (`e6_apertura_desde_la_hoja.py`), no un valor elegido para que salga bien."""
COL = {0: (60, 240, 60), 1: (60, 160, 255)}
COL_AJENO = (70, 70, 255)
GRIS_CONTROL = (170, 170, 170)
GUARDIA_ALFA = 0.05
"""Si el alfa medio del recorte del florete baja de aqui, `matting` ha devuelto fondo vacio porque el
recorte tiene mas de una persona. Es la guardia de `e5_hoja_en_alfa.py`, no un descarte nuevo."""


def pinta(im, kp, sc, col, g=3, r=5, rm=2):
    """Esqueleto: es `video_revision._pinta` con el GROSOR como parametro y las manos del color del
    tirador.

    Los dos cambios tienen motivo. El grosor, porque la celda reduce 2x y a 2 px las lineas se
    quedan en 1 px y el video parece vacio. El color de la mano, porque en `video_revision` las dos
    manos van en cian y aqui hay que poder decir de QUIEN es la mano que se esta viendo. El criterio
    de validez de un punto no se toca: confianza >= 0,3 Y coordenada finita, porque hay puntos con
    confianza alta y coordenada NaN y filtrar solo por confianza dibuja basura.

    La cara no se pinta: son 245 de los 308 puntos, empiezan en el 63 y no se usan en el proyecto.
    """
    def val(i):
        return sc[i] >= VR.CONF_PINTA and np.isfinite(kp[i, 0]) and np.isfinite(kp[i, 1])

    def pt(i):
        return (int(kp[i, 0]), int(kp[i, 1]))

    for a, b in VR.HUESOS:
        if val(a) and val(b):
            cv2.line(im, pt(a), pt(b), col, g, cv2.LINE_AA)
    for a, b in VR.PULGARES:
        if val(a) and val(b):
            cv2.line(im, pt(a), pt(b), (60, 240, 255), g, cv2.LINE_AA)
    for i in VR.MANOS:
        if val(i):
            cv2.circle(im, pt(i), rm, col, -1)
    for i in range(0, 21):
        if val(i):
            cv2.circle(im, pt(i), r, col, -1)


def rotula(cc, tit, pie=None, k=None):
    """Cada celda con su rotulo arriba y sus cifras abajo. `pie` puede ser varias lineas."""
    h, w = cc.shape[:2]
    cv2.rectangle(cc, (0, 0), (w, 26), (14, 14, 18), -1)
    cv2.putText(cc, tit, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    if k is not None:
        cv2.putText(cc, "c%04d" % k, (w - 78, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (140, 220, 255), 1, cv2.LINE_AA)
    if pie:
        ls = [pie] if isinstance(pie, str) else list(pie)
        alto = 8 + 19 * len(ls)
        cv2.rectangle(cc, (0, h - alto), (w, h), (14, 14, 18), -1)
        for j, s in enumerate(ls):
            cv2.putText(cc, s, (7, h - alto + 16 + 19 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        (120, 235, 255), 1, cv2.LINE_AA)
    return cc


def negra(tit, pie, w, h, k):
    return rotula(np.zeros((h, w, 3), "uint8"), tit, pie, k)


def inset_mano(im_fino, kp, sc, mano, alto_caja, lado_px, nunca_reduce=False):
    """Recorte de la mano armada a resolucion NATIVA, para que se vean los 21 puntos.

    El recorte se toma del cuadro pintado con lineas FINAS: las gruesas de la celda 1 taparian la
    mano entera a este zoom.

    `nunca_reduce=True` impide que el recuadro AMPLIE menos de 1x. A 1080p la caja del tirador es
    alta, asi que `0,34 * alto_caja` pedia un recorte mas grande que el hueco donde se pega, y el
    resultado salia **reducido** con el rotulo `zoom x0.8`: un zoom que reduce no es un zoom, es una
    miniatura, y encima el rotulo afirmaba lo contrario. Con el interruptor el recorte se cinie al
    hueco disponible, o sea que se ve MENOS mano pero a resolucion nativa o mejor, que es para lo que
    existe el recuadro.
    """
    M = CO.MANOS[mano]
    root = kp[M["root"]]
    if not (np.isfinite(root).all() and sc[M["root"]] >= VR.CONF_PINTA):
        return None, 0
    s = int(max(80.0, 0.34 * alto_caja))
    if nunca_reduce:
        s = min(s, int(lado_px))
    H_, W_ = im_fino.shape[:2]
    x0 = int(np.clip(root[0] - s // 2, 0, max(0, W_ - s)))
    y0 = int(np.clip(root[1] - s // 2, 0, max(0, H_ - s)))
    cr = im_fino[y0:y0 + s, x0:x0 + s]
    if cr.shape[0] < 8 or cr.shape[1] < 8:
        return None, 0
    return cv2.resize(cr, (lado_px, lado_px), interpolation=cv2.INTER_NEAREST), s


def gira(u, grados):
    th = np.radians(grados)
    return np.array([u[0] * np.cos(th) - u[1] * np.sin(th), u[0] * np.sin(th) + u[1] * np.cos(th)])


IDIOMA = "es"
"""Idioma de los rotulos que se PINTAN, no de los docstrings.

POR QUE EXISTE: este mosaico es la superficie de comprobacion interna y sus rotulos estan en espaniol
porque quien los mira es el usuario. Pero `panel_publico` reutiliza `panel_florete` tal cual para el
repo publico, y ahi todo el texto tiene que ir en ingles. Antes de esto, la celda del florete salia en
espaniol dentro de una figura por lo demas inglesa.

La alternativa era duplicar `panel_florete` en el fichero publico, y eso crearia una segunda version
del umbral de la hoja y del control girado: entonces las cifras del pie podrian dejar de corresponder
con las medidas. Se cambia el texto, nunca el criterio.

TODO LO PINTADO ES ASCII a proposito: `cv2.putText` solo dibuja fuentes Hershey, asi que un grado, un
mas-menos o un punto medio saldrian como basura. De ahi `deg` y `|` en vez de los simbolos."""

TXT = {
    "es": {"tit": "6  florete  |  lado %d", "tit_mano": "6  florete  |  lado %d  |  mano %s",
           "sin_mano": "sin mano armada decidida",
           "sin_eje": "sin eje: falta codo o mano con confianza",
           "poco": "recorte insuficiente: R = %.0f px", "forma": "capas fuera de forma",
           "recta": "RECTA %.0f cm a %.0f deg del eje  |  %d px sobre el umbral",
           "sin_recta": "SIN RECTA en este cuadro  |  %d px sobre el umbral",
           "control": "cono en su sitio %.0f %%  |  CONTROL girado 90 deg %.0f %%  |  R = %.0f cm",
           "guardia": "GUARDIA: alfa medio %.4f, mas de una persona en el recorte"},
    "en": {"tit": "6  FOIL  |  side %d", "tit_mano": "6  FOIL - blade and threat cone  |  side %d  "
           "|  weapon hand %s",
           "sin_mano": "no weapon hand decided for this fencer",
           "sin_eje": "no arm axis: elbow or hand below confidence",
           "poco": "crop too small: R = %.0f px", "forma": "layers came back with the wrong shape",
           "recta": "BLADE FOUND: %.0f cm at %.0f deg from the arm axis  |  %d px over threshold",
           "sin_recta": "no blade line in this frame  |  %d px over threshold",
           "control": "cone in place %.0f %%  |  CONTROL, cone turned 90 deg %.0f %%  |  "
           "R = %.0f cm",
           "guardia": "GUARD TRIPPED: mean alpha %.4f, more than one person in the crop"},
}


def _t(clave):
    return TXT[IDIOMA][clave]


def panel_florete(cap, rgb, k, lado, g, B, W, H, ancho, alto):
    """Celda 6, un panel por tirador. Devuelve (panel, fila_de_medidas).

    Todo lo que decide (recorte alargado hasta la punta de la hoja y cortado antes del rival, umbral
    p99,5 del alfa fuera del cuerpo dilatado, Hough y recta mas larga que pase a menos de 0,12 R de
    la mano) sale de `e7_pinta_cono`. Aqui solo se pinta y se cuenta el control girado.
    """
    d = {"k": k, "lado": lado, "eje": False, "n_px": 0, "recta": False,
         "dentro_0": np.nan, "dentro_90": np.nan, "contaminado": None, "largo_cm": np.nan,
         "dif_eje": np.nan, "R": np.nan, "alfa_medio": np.nan}
    tit = _t("tit") % lado
    if g is None:
        return negra(tit, _t("sin_mano"), ancho, alto, None), d
    if not g["ok"][k] or not np.isfinite(B[k, lado]).all():
        return negra(tit, _t("sin_eje"), ancho, alto, None), d
    vert, u, L = g["vert"][k], g["u"][k], g["L"]
    (cx0, cy0, cx1, cy1), R = E7.recorte(B[k, lado], B[k, 1 - lado], vert, u, L, W, H)
    d["R"], d["eje"] = R, True
    if cx1 - cx0 < 64 or cy1 - cy0 < 64 or R < 24:
        return negra(tit, _t("poco") % R, ancho, alto, None), d
    sub = np.ascontiguousarray(rgb[cy0:cy1, cx0:cx1])
    alfa = cap.matting(sub)
    seg = cap.seg(sub)
    if alfa.shape[:2] != sub.shape[:2] or seg.shape[:2] != sub.shape[:2]:
        return negra(tit, _t("forma"), ancho, alto, None), d
    am = float(alfa.mean())
    d["alfa_medio"] = am
    d["contaminado"] = bool(am < GUARDIA_ALFA)
    m, ln = E7.detecta(alfa, seg, vert, cx0, cy0, R)
    hh, ww = alfa.shape[:2]
    la = np.log10(np.clip(alfa, 1e-6, 1.0))
    vis = cv2.applyColorMap((np.clip((la + 4.0) / 4.0, 0, 1) * 255).astype("uint8"),
                            cv2.COLORMAP_TURBO)
    off = np.array([cx0, cy0], dtype="float64")
    pies = []
    if m is not None:
        d["n_px"] = int(m.sum())
        vis[m] = (255, 255, 255)
        if d["n_px"] > 0:
            GY, GX = np.mgrid[0:hh, 0:ww]
            P = np.stack([GX[m] + cx0, GY[m] + cy0], axis=1).astype("float64")
            d["dentro_0"] = float((CO.peso_sector(P, vert, u, R, AP) > 0).mean())
            d["dentro_90"] = float((CO.peso_sector(P, vert, gira(u, 90.0), R, AP) > 0).mean())
    E7.dibuja_cono(vis, vert - off, gira(u, 90.0), R, AP, GRIS_CONTROL, 1)
    E7.dibuja_cono(vis, vert - off, u, R, AP, COL[lado], 2)
    cv2.line(vis, tuple(np.int32(vert - off - 0.30 * R * u)), tuple(np.int32(vert - off)),
             (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(vis, tuple(np.int32(vert - off)), 5, (255, 255, 255), -1)
    if ln is not None:
        lg, p1, p2 = ln
        cv2.line(vis, tuple(np.int32(p1 - off)), tuple(np.int32(p2 - off)), (60, 255, 60), 3,
                 cv2.LINE_AA)
        v = p2 - p1
        dif = abs((np.degrees(np.arctan2(v[1], v[0]))
                   - np.degrees(np.arctan2(u[1], u[0])) + 90.0) % 180.0 - 90.0)
        d["recta"] = True
        d["largo_cm"] = 100.0 * lg / g["pxm"]
        d["dif_eje"] = float(dif)
        pies.append(_t("recta") % (d["largo_cm"], dif, d["n_px"]))
    else:
        pies.append(_t("sin_recta") % d["n_px"])
    if np.isfinite(d["dentro_0"]):
        pies.append(_t("control") % (100 * d["dentro_0"], 100 * d["dentro_90"],
                                     100.0 * R / g["pxm"]))
    if d["contaminado"]:
        pies.append(_t("guardia") % am)
    cel = E7.celda(vis, "", "", ancho, alto)
    return rotula(cel, _t("tit_mano") % (lado, g["mano"]), pies, None), d


def main():
    mapa = {}
    for raiz, _dd, fs in os.walk(VIDEOS):
        for f in fs:
            if f.lower().endswith(".mp4"):
                mapa.setdefault(pathlib.Path(f).stem, os.path.join(raiz, f))
    with open(SCR / ("det_%s.pkl" % CLIP), "rb") as fh:
        cajas, cuadros, W, H, T = pickle.load(fh)
    """La puerta de FUNDIDOS de `extrae_pareja` necesita el clip en gris a 160x90. Sin `gris` no se
    aplica y se procesarian cuadros de transicion, que es donde `matting` y `seg` fallan."""
    gris = []
    v = cv2.VideoCapture(mapa[CLIP])
    fps = v.get(cv2.CAP_PROP_FPS) or 25.0
    while True:
        ok, im = v.read()
        if not ok:
            break
        gris.append(cv2.cvtColor(cv2.resize(im, (160, 90)), cv2.COLOR_BGR2GRAY))
    v.release()
    sel = CL.elige_pareja_del_clip(cajas, cuadros, W, H, T, gris=gris)
    B, pres, msk, tipo = sel["B"], sel["pres"], sel["mask"], sel["tipo"]
    KS = list(range(0, T, PASO))
    proc = [k for k in KS if msk[k] and pres[k].all()]
    print("clip %s  %dx%d  T=%d  fps %.1f" % (CLIP, W, H, T, fps))
    print("  utilizables tras la cadena entera: %d de %d cuadros" % (int(msk.sum()), T))
    print("  muestreo cada %d: %d cuadros, se procesan %d y se pintan como descartados %d"
          % (PASO, len(KS), len(proc), len(KS) - len(proc)))

    cap = CL.Capas(["pose", "matting", "seg", "normal", "pointmap"])
    print("  modelos cargados")

    """PRIMERA PASADA, solo pose. Hace falta ANTES del mosaico porque la mano armada se decide por
    MEDIANAS del clip (`cono_lib.mano_armada`), no cuadro a cuadro, y porque el reparto por esqueleto
    de la mascara necesita tambien las poses de los AJENOS del cuadro. Las poses se guardan y no se
    vuelven a calcular en la segunda pasada."""
    N = 308
    KP = np.full((T, 2, N, 2), np.nan, "float32")
    SC = np.zeros((T, 2, N), "float32")
    guardadas, ajenos_k = {}, {}
    vid = cv2.VideoCapture(mapa[CLIP])
    t0 = time.perf_counter()
    for i, k in enumerate(proc):
        vid.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, im = vid.read()
        if not ok:
            continue
        rgb = np.ascontiguousarray(im[:, :, ::-1])
        aj = CL.ajenos_del_cuadro(sel, k, H)
        cbs = [B[k, 0] * np.array([W, H, W, H]), B[k, 1] * np.array([W, H, W, H])] + aj
        poses = cap.pose(rgb, cbs)
        kt = {("t", 0): poses[0], ("t", 1): poses[1]}
        for j in range(len(aj)):
            kt[("a", j)] = poses[2 + j]
        guardadas[k], ajenos_k[k] = kt, aj
        for lado in (0, 1):
            kp, sc = poses[lado]
            n = min(len(kp), N)
            KP[k, lado, :n], SC[k, lado, :n] = kp[:n], sc[:n]
        if i % 25 == 0:
            print("    pose c%04d  %d/%d  %.2f s/cuadro"
                  % (k, i + 1, len(proc), (time.perf_counter() - t0) / (i + 1)))
    vid.release()

    geo = {}
    for lado in (0, 1):
        mano, det = CO.mano_armada(KP, SC, B, W, H, lado)
        if mano is None:
            geo[lado] = None
            print("  lado %d: SIN mano armada, menos de 10 cuadros con codo, hombro y mano" % lado)
            continue
        vert, u, ok, _n = CO.eje_antebrazo(KP, SC, lado, mano)
        _pk, pxm = CO.escala_px_m(B, lado, H)
        geo[lado] = {"mano": mano, "vert": vert, "u": u, "ok": ok, "pxm": pxm,
                     "L": CO.HOJA_M * pxm}
        print("  lado %d: mano armada %s por posicion (margen %.0f px), el criterio de direccion "
              "%s  ·  %.1f px/m  ·  hoja %.0f px  ·  eje valido en %d cuadros"
              % (lado, mano, det.get("margen_x_px", 0.0),
                 "corrobora" if det.get("corrobora_b2") else "NO corrobora", pxm,
                 CO.HOJA_M * pxm, int(ok[proc].sum()) if proc else 0))

    ms_pose_1 = dict(cap.tabla_tiempos())
    """Foto de los contadores al acabar la PRIMERA pasada: asi el coste de la pose queda separado del
    de las capas densas de la segunda, que es lo que hace falta para el informe de tiempos."""

    hc = int(round(H * CELDA / float(W)))
    mos = cv2.VideoWriter(str(SALIDA / ("mosaico_seis_%s.mp4" % CLIP)),
                          cv2.VideoWriter_fourcc(*"mp4v"), max(4.0, fps / PASO),
                          (CELDA * 3, hc * 2))
    fijos = set(proc[int(round(x))] for x in np.linspace(0, len(proc) - 1, 4)) if proc else set()
    filas, flor, cob, con_recta = [], [], [], []
    ms_flor = {"matting": 0.0, "seg": 0.0}
    """El florete llama a `matting` y a `seg` sobre SU recorte, que es otro y mas grande que el de la
    caja. Sin separar estos ms del resto, el coste de `matting` del mosaico saldria el doble del de la
    limpieza y no se sabria por que."""
    vid = cv2.VideoCapture(mapa[CLIP])
    t0 = time.perf_counter()
    for i, k in enumerate(KS):
        vid.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, im = vid.read()
        if not ok:
            continue
        if k not in guardadas:
            """CUADRO DESCARTADO: se pinta, no se salta. Ver el docstring del modulo."""
            c1 = im.copy()
            cv2.rectangle(c1, (0, 40), (W, 104), (0, 0, 160), -1)
            razon = ("FUERA: repeticion, primer plano o fundido, este cuadro NO se procesa"
                     if not msk[k] else "FUERA: falta uno de los dos tiradores en este cuadro")
            cv2.putText(c1, razon, (14, 86), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 3,
                        cv2.LINE_AA)
            c1 = rotula(cv2.resize(c1, (CELDA, hc)), "1  original + esqueleto + cajas",
                        "cuadro descartado por la tuberia", k)
            vis = [c1] + [negra(t, "cuadro descartado", CELDA, hc, k) for t in
                          ("2  seg limpio", "3  normal limpio", "4  matting limpio",
                           "5  pointmap (NO vale como profundidad)", "6  florete")]
            mos.write(np.vstack([np.hstack(vis[:3]), np.hstack(vis[3:])]))
            continue
        rgb = np.ascontiguousarray(im[:, :, ::-1])
        kt, aj = guardadas[k], ajenos_k[k]

        pm = cap.pointmap(rgb)
        Z = pm[:, :, 2] if pm.ndim == 3 else np.zeros((H, W), "float32")
        M_full = np.zeros((H, W), bool)
        cel_mat = np.zeros((H, W), "float32")
        cel_seg = np.zeros((H, W), "uint8")
        cel_nor = np.zeros((H, W, 3), "float32")
        for lado in (0, 1):
            r = CL.mascara_tirador(cap, rgb, sel, k, lado, W, H, kt, con_matting=True)
            if r is None:
                continue
            c, mk = r["c"], r["mask"]
            M_full[c[1]:c[3], c[0]:c[2]] |= mk
            cel_mat[c[1]:c[3], c[0]:c[2]][mk] = r["alfa"][mk]
            cel_seg[c[1]:c[3], c[0]:c[2]][mk] = r["seg"][mk]
            n = cap.normal(r["sub"])
            if n.ndim == 3 and n.shape[:2] == mk.shape:
                cel_nor[c[1]:c[3], c[0]:c[2]][mk] = n[mk]
            filas.append(CL.mide(r, sel, k, lado, W, H, kt[("t", lado)], bgr=im))
        cob.append(float(M_full.mean()))

        c1 = im.copy()
        c1f = im.copy()
        for lado in (0, 1):
            kp, sc = kt[("t", lado)]
            pinta(c1, kp, sc, COL[lado], g=3, r=5, rm=2)
            pinta(c1f, kp, sc, COL[lado], g=1, r=3, rm=1)
            b = (B[k, lado] * np.array([W, H, W, H])).astype(int)
            cv2.rectangle(c1, (b[0], b[1]), (b[2], b[3]), COL[lado], 3)
            et = ["IZQUIERDA", "DERECHA"][lado]
            if tipo[k, lado] == 1:
                et += " interpolada"
            elif tipo[k, lado] == 2:
                et += " sostenida"
            elif tipo[k, lado] == 3:
                et += " redetectada"
            cv2.putText(c1, et, (b[0], max(30, b[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        COL[lado], 3, cv2.LINE_AA)
        for o in aj:
            o = np.asarray(o).astype(int)
            cv2.rectangle(c1, (o[0], o[1]), (o[2], o[3]), COL_AJENO, 3)
            cv2.putText(c1, "NO elegido", (o[0], max(30, o[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, COL_AJENO, 2, cv2.LINE_AA)
        c1 = cv2.resize(c1, (CELDA, hc))
        lado_px = int(0.34 * hc)
        for lado in (0, 1):
            g = geo[lado]
            if g is None:
                continue
            kp, sc = kt[("t", lado)]
            alto_caja = (B[k, lado, 3] - B[k, lado, 1]) * H
            ins, s = inset_mano(c1f, kp, sc, g["mano"], alto_caja, lado_px)
            if ins is None:
                continue
            x0 = 6 if lado == 0 else CELDA - lado_px - 6
            y0 = hc - lado_px - 32
            c1[y0:y0 + lado_px, x0:x0 + lado_px] = ins
            cv2.rectangle(c1, (x0, y0), (x0 + lado_px, y0 + lado_px), COL[lado], 2)
            cv2.putText(c1, "mano %s  zoom x%.1f" % (["izq", "der"][lado], lado_px / float(s)),
                        (x0 + 5, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL[lado], 1,
                        cv2.LINE_AA)

        c2 = CL.pinta_seg(cel_seg)
        c3 = CL.pinta_normal(cel_nor)
        c3[~M_full] = 0
        c4 = cv2.cvtColor((np.clip(cel_mat, 0, 1) * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
        c5 = CL.pinta_z(Z, M_full)
        p6, hay_recta = [], False
        antes = {"matting": cap.ms["matting"], "seg": cap.ms["seg"]}
        for lado in (0, 1):
            pnl, dd = panel_florete(cap, rgb, k, lado, geo[lado], B, W, H, CELDA // 2, hc)
            p6.append(pnl)
            flor.append(dd)
            hay_recta = hay_recta or bool(dd["recta"])
        for q in ("matting", "seg"):
            ms_flor[q] += cap.ms[q] - antes[q]
        c6 = np.hstack(p6)
        if hay_recta and len(con_recta) < 4:
            """Los cuatro cuadros fijos se eligen repartidos por el clip, y la deteccion de la hoja
            es INTERMITENTE (42,6 % de los recortes a 1080p), asi que los cuatro pueden caer todos
            en cuadros sin recta y entonces no hay nada que mirar. Estos son los cuatro primeros
            cuadros CON recta, que son los que permiten juzgar la deteccion."""
            con_recta.append(k)
            cv2.imwrite(str(FIJOS / ("recta_%s_c%04d.png" % (CLIP, k))), c6)

        vis = [rotula(c1, "1  original + esqueleto + cajas (verde izq, naranja der, rojo NO elegido)",
                      "las manos son 21 puntos cada una: recortes a resolucion nativa abajo", k),
               rotula(cv2.resize(c2, (CELDA, hc)), "2  seg limpio (base de la mascara)",
                      "fuera de las dos mascaras, NEGRO", k),
               rotula(cv2.resize(c3, (CELDA, hc)), "3  normal limpio",
                      "cada eje a un canal de color", k),
               rotula(cv2.resize(c4, (CELDA, hc)), "4  matting limpio",
                      "alfa dentro de la mascara", k),
               rotula(cv2.resize(c5, (CELDA, hc)), "5  pointmap de cuadro completo",
                      ["NO VALE COMO PROFUNDIDAD: fallo su control fisico, separa a los dos tiradores",
                       "entre si mas que a la pareja del arbitro. La y del pie acierta 28 de 28.",
                       "Se ensena porque se ha pedido verla, no porque se use."], k),
               c6]
        fila = np.vstack([np.hstack(vis[:3]), np.hstack(vis[3:])])
        mos.write(fila)
        if k in fijos:
            cv2.imwrite(str(FIJOS / ("mosaico_%s_c%04d.png" % (CLIP, k))), fila)
            cv2.imwrite(str(FIJOS / ("florete_%s_c%04d.png" % (CLIP, k))), c6)
            cv2.imwrite(str(FIJOS / ("celda1_%s_c%04d.png" % (CLIP, k))), vis[0])
            cv2.imwrite(str(FIJOS / ("capas_%s_c%04d.png" % (CLIP, k))),
                        np.hstack([vis[1], vis[3]]))
        if i % 10 == 0:
            print("    c%04d  %d/%d  %.1f s/cuadro"
                  % (k, i + 1, len(KS), (time.perf_counter() - t0) / (i + 1)))
    mos.release()
    vid.release()

    import pandas as pd
    d = pd.DataFrame(filas)
    f = pd.DataFrame(flor)
    d["valido"] = d["rec_kp"] >= 0.70
    """La bandera es la de `capas_limpias_video`: `rec_kp < 0,70`, el acuerdo entre la mascara y la
    pose, que son dos modelos independientes. Marcar por AREA esta medido y descartado."""
    con_eje = f[f["eje"]]
    con_px = con_eje[con_eje["n_px"] > 0]
    hay = f[f["recta"]]

    """COSTE POR CABEZA. Se da en dos columnas porque las dos hacen falta y dicen cosas distintas:
    `ms/llamada` es lo que cuesta una pasada del modelo, y `ms/cuadro` es lo que cuesta el cuadro
    entero, que depende de CUANTAS veces hay que llamarla. `matting` cuesta cuatro llamadas por
    cuadro (dos tiradores para la mascara y dos recortes para el florete) y `pointmap` una sola,
    porque va de cuadro completo, asi que ordenarlas por ms/llamada engana."""
    tt = cap.tabla_tiempos()
    nproc = max(len(proc), 1)
    tiempos = {}
    for q, v in tt.items():
        prop = 0.0
        if q == "pose":
            prop = ms_pose_1.get(q, {}).get("ms_total", 0.0)
        tiempos[q] = {"llamadas": v["llamadas"],
                      "llamadas_por_cuadro": v["llamadas"] / float(nproc),
                      "ms_por_llamada": v["ms_por_llamada"],
                      "ms_por_cuadro": v["ms_total"] / float(nproc),
                      "ms_por_cuadro_primera_pasada": prop / float(nproc),
                      "ms_por_cuadro_del_florete": ms_flor.get(q, 0.0) / float(nproc)}
    tiempos["TODAS"] = {"ms_por_cuadro": sum(t["ms_por_cuadro"] for t in tiempos.values())}

    res = {"clip": CLIP, "W": W, "H": H, "T": T, "paso": PASO, "tiempos": tiempos,
           "utilizables": int(msk.sum()), "muestreados": len(KS), "procesados": len(proc),
           "descartados_pintados": len(KS) - len(proc),
           "rec_kp": float(d["rec_kp"].mean()), "rec_desp": float(d["rec_desp"].mean()),
           "fuga_arb": float(d["fuga_arb"].mean()), "fuga_fondo": float(d["fuga_fondo"].mean()),
           "resto_arb_frac": float(d["resto_arb_frac"].mean()),
           "residuo": float(d["residuo"].mean()),
           "fuga_solape": float(d["fuga_solape"].mean()),
           "ret_oscuro_propio": float(d["ret_oscuro_propio"].mean()),
           "cobertura_cuadro": float(np.mean(cob)) if cob else None,
           "no_validos": int((~d["valido"]).sum()), "lados": int(len(d)),
           "florete_lados_con_eje": int(len(con_eje)),
           "florete_lados_con_recta": int(f["recta"].sum()),
           "florete_cuadros_con_recta": int(hay["k"].nunique()) if len(hay) else 0,
           "florete_dif_eje_mediana": float(hay["dif_eje"].median()) if len(hay) else None,
           "florete_dentro_0": float(con_px["dentro_0"].median()) if len(con_px) else None,
           "florete_dentro_90": float(con_px["dentro_90"].median()) if len(con_px) else None,
           "florete_contaminados": int(con_eje["contaminado"].fillna(False).sum()),
           "mano": dict((str(l), (geo[l]["mano"] if geo[l] else None)) for l in (0, 1))}
    d.to_csv(SALIDA / ("medidas_%s.csv" % CLIP), index=False)
    f.to_csv(SALIDA / ("florete_%s.csv" % CLIP), index=False)
    (SALIDA / ("resumen_%s.json" % CLIP)).write_text(json.dumps(res, indent=1), encoding="utf-8")

    print()
    print("=" * 100)
    print("%s  ·  %d cuadros procesados, %d pintados como descartados"
          % (CLIP, len(proc), len(KS) - len(proc)))
    print("=" * 100)
    print("  LIMPIEZA DE LAS CAPAS")
    print("    rec_kp      %.3f   (bueno >= 0,90)   CONTROL NEGATIVO rec_desp %.3f"
          % (d["rec_kp"].mean(), d["rec_desp"].mean()))
    print("    fuga_arb    %.4f  (bueno <= 0,05)   fuga_fondo %.4f"
          % (d["fuga_arb"].mean(), d["fuga_fondo"].mean()))
    print("    resto_arb   %.4f  fraccion de la mascara que es oscuro dentro de la caja de un ajeno"
          % d["resto_arb_frac"].mean())
    print("    fuga_solape %.4f  contra nivel de referencia %.4f (el oscuro que SI es del tirador)"
          % (d["fuga_solape"].mean(), d["ret_oscuro_propio"].mean()))
    print("    residuo     %.3f   contra 0,657 de la caja sola  ·  cobertura del cuadro %.2f %%"
          % (d["residuo"].mean(), 100 * np.mean(cob) if cob else float("nan")))
    print("    marcados NO validos por rec_kp < 0,70: %d de %d lados"
          % (int((~d["valido"]).sum()), len(d)))
    print(d.nsmallest(4, "rec_kp")[["k", "lado", "rec_kp", "fuga_arb", "resto_arb_frac",
                                    "residuo"]].to_string(index=False))
    print("  FLORETE")
    print("    lados con eje del brazo valido: %d de %d" % (len(con_eje), len(f)))
    print("    RECTA DETECTADA en %d lados y en %d cuadros de %d procesados (%.1f %% de cuadros)"
          % (int(f["recta"].sum()), res["florete_cuadros_con_recta"], len(proc),
             100.0 * res["florete_cuadros_con_recta"] / max(len(proc), 1)))
    if res["florete_dif_eje_mediana"] is not None:
        print("    |angulo recta - eje| mediana %.1f gr   (azar declarado 45 gr)"
              % res["florete_dif_eje_mediana"])
    if res["florete_dentro_0"] is not None:
        """NO se imprime la razon dentro_0 / dentro_90. La mediana de `dentro_90` sale 0,0 exacto,
        asi que la razon daba 10^8, un numero que no significa nada y que ademas parece una medida
        espectacular. Se dan las dos cifras y que se comparen; y aparte la mediana SOLO sobre los
        lados con recta detectada, que es donde la comparacion es interpretable."""
        gr = f[f["recta"] == True]
        print("    CONTROL DEL CONO, mediana sobre los %d lados con pixeles detectados: en su sitio "
              "%.1f %%  ·  girado 90 gr %.1f %%" % (len(con_px), 100 * res["florete_dentro_0"],
                                                    100 * res["florete_dentro_90"]))
        if len(gr):
            print("      y SOLO sobre los %d lados con recta detectada: en su sitio %.1f %%  ·  "
                  "girado 90 gr %.1f %%" % (len(gr), 100 * gr["dentro_0"].median(),
                                            100 * gr["dentro_90"].median()))
    print("    guardia de una persona por recorte: %d lados contaminados"
          % res["florete_contaminados"])
    print("  COSTE POR CABEZA, %d cuadros de %dx%d en esta misma tirada" % (nproc, W, H))
    print("    %-10s %9s %11s %11s %11s" % ("cabeza", "llam/cua", "ms/llamada", "ms/cuadro",
                                            "de eso, el florete"))
    for q in ("pose", "seg", "normal", "matting", "pointmap"):
        if q not in tiempos:
            continue
        t = tiempos[q]
        print("    %-10s %9.1f %11.0f %11.0f %11.0f"
              % (q, t["llamadas_por_cuadro"], t["ms_por_llamada"], t["ms_por_cuadro"],
                 t["ms_por_cuadro_del_florete"]))
    print("    %-10s %9s %11s %11.0f" % ("LAS CINCO", "", "", tiempos["TODAS"]["ms_por_cuadro"]))
    print()
    print("  video  -> %s" % (SALIDA / ("mosaico_seis_%s.mp4" % CLIP)))
    print("  fijos  -> %s" % FIJOS)


if __name__ == "__main__":
    main()

"""Galeria de LA CASCADA DE TRES RONDAS: el mismo cuadro resuelto por cada ronda, uno al lado de otro.

QUE CAMBIA RESPECTO DE `galeria_inventadas`. Aquella galeria pinta **ancla / inventada / ancla**, o
sea el mismo caso en tres cuadros distintos, y sirve para juzgar si la pose inventada es una
continuacion creible. Esta pinta **el MISMO cuadro tres veces**, una por ronda, y sirve para juzgar
otra cosa: cual de las tres vias resuelve ese cuadro y si la que gana es mejor que la que se entrega.

    PANEL 1   RONDA 1 - `NO DETECTADA`. Es el estado del que se parte: YOLO fallo en el cuadro y
              fallo en el recorte. Se pinta la caja ESTIMADA en violeta y **ninguna pose**, porque
              ninguna hubo: dibujar algo aqui seria ensenar un resultado que no existe.
    PANEL 2   RONDA 2 - la cabeza `pose` de Sapiens 2 **top-down** sobre esa caja estimada, en una
              color NUEVA porque no es ni medida en la ronda 1 ni inventada. Lleva si **pasa el
              filtro** del proyecto y con cuantos puntos, tambien cuando FALLA: una galeria que solo
              ensenara los aciertos no dejaria juzgar el modo de fallo.
    PANEL 3   RONDA 3 - la pose INTERPOLADA, en violeta, que es **lo que la tuberia entrega hoy**,
              con el error esperado a ese largo de hueco.

SOLO HUECO <= 24, y el motivo es medido y no de gusto: con hueco >= 25 el filtro pasa 0,6 % con un
tirador delante y **12,7 % sobre fondo vacio**, o sea que ahi no discrimina y la ronda 2 no se
intenta. Esta en `e29_tres_rondas` y en `rescate_decision.txt`.

LO QUE NO SE REIMPLEMENTA: el panel es `galeria_inventadas.panel` tal cual, el esqueleto es
`video_revision._pinta`, las colores son `video_revision.VIOLETA` y `.RESCATADA` -- un solo sitio,
que antes eran cuatro copias del violeta -- y el error por hueco es
`galeria_inventadas.error_esperado`.

USO:  python galeria_tres_rondas.py [CUANTAS]     CPU, corre despues de `e29_tres_rondas.py`
"""
from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

HERRAM = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERRAM))
CODIGO = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
sys.path.insert(0, str(CODIGO))

import galeria_inventadas as GI
import video_revision as VR

PRUEBAS = GI.PRUEBAS
SALIDA = pathlib.Path(
    r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\poses_tres_rondas")
CUANTAS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
FUERA = GI.FUERA
"""LOS MISMOS CLIPS EXCLUIDOS QUE EN LA OTRA GALERIA, y por las mismas razones escritas alli:
`MellowHideousAngora` es de SABLE y `PowerfulAllLarva` tiene al ARBITRO elegido como tirador. Se
importa la lista en vez de copiarla para que no puedan divergir."""


FP_GATE = [(1, 1, 0.0, 28), (2, 4, 3.8, 53), (5, 12, 0.0, 115), (13, 24, 0.0, 99)]
"""EL FALSO POSITIVO DEL FILTRO, POR TRAMO DE HUECO, medido en `rescate_casos.csv`: el MISMO filtro
sobre una caja del mismo tamanio puesta en **fondo vacio** (solape cero con los dos tiradores) en el
mismo cuadro. Es la columna `neg_ok`, y va en el rotulo de cada imagen por una razon concreta.

Un modelo top-down devuelve puntos SIEMPRE, tambien donde no hay nadie. Asi que «pasa el filtro» no
significa «hay un tirador ahi» hasta que se sabe cuanto pasa el filtro cuando seguro que no hay
nadie. El primer rotulo que escribi decia *"ROUND 2 wins"*, que es un VEREDICTO, y mirando una
imagen del tramo 5-12 con la pista casi vacia se ve que no puedo afirmarlo desde una figura. Ahora
dice *"pasa el filtro"* y al lado va esta cifra, que es lo que ese «pasa» vale.

    tramo    pasa con tirador   pasa sobre fondo vacio
    1            13 de 28 (46,4 %)    0 de  28 (0,0 %)
    2-4          12 de 53 (22,6 %)    2 de  53 (3,8 %)
    5-12         12 de 115 (10,4 %)   0 de 115 (0,0 %)
    13-24        13 de 99 (13,1 %)    0 de  99 (0,0 %)
    >= 25         2 de 355 (0,6 %)   45 de 355 (**12,7 %**)   <- por eso ese tramo no se intenta"""


def fp_del_hueco(h):
    for a, b, fp, n in FP_GATE:
        if a <= h <= b:
            return fp, n
    return float("nan"), 0


def casos():
    """Los casos con ronda 2 intentada, ordenados por hueco descendente dentro de cada clip."""
    out = []
    for f in sorted(PRUEBAS.glob("rescate_poses_*.npz")):
        clip = f.name[len("rescate_poses_"):-len(".npz")]
        if clip in FUERA:
            continue
        z = np.load(f, allow_pickle=True)
        hue, pasa, nut = z["hueco_caso"], z["pasa_gate"], z["n_util"]
        for lado in (0, 1):
            for k in np.nonzero(hue[:, lado] >= 0)[0]:
                out.append({"clip": clip, "lado": lado, "k": int(k),
                            "hueco": int(hue[k, lado]), "pasa": bool(pasa[k, lado]),
                            "n_util": int(nut[k, lado]), "npz": f})
    return out


ASPECTO_MIN = 0.45
"""Ancho minimo del recorte en fraccion de su alto, y NO es un ajuste de gusto: sin el, la galeria
sale ilegible. La caja de la ronda 1 es una caja **estimada por interpolacion**, y cuando el hueco es
largo puede degenerar: medido en esta tirada, hay casos con la caja de **46 px de ancho y 1000 de
alto**. `panel` escala al ancho fijo de 430 px conservando la proporcion, asi que esos casos daban
paneles de 430 x 4000 px -- una tira vertical donde no se distingue nada. Un tirador de pie ocupa
como 1:2,5, asi que 0,45 deja pasar cualquier caja sana y solo corrige las degeneradas.

Y la caja degenerada no se esconde por ensancharla: se **dice en el rotulo**, porque una caja
estimada de 46 px de ancho es en si misma la explicacion de por que la ronda 2 falla ahi."""


def recorte(Be, B, k, lado, W, H):
    """El recorte comun a los tres paneles. TIENE QUE SER EL MISMO o la comparacion no vale.

    Devuelve `(rec, degenerada)`: `degenerada` es True si hubo que ensanchar por proporcion, o sea
    si la caja estimada era una astilla.
    """
    cajas = [b for b in (Be[k, lado], B[k, lado]) if np.isfinite(b).all()]
    if not cajas:
        return None, False
    xs = [v for b in cajas for v in (b[0] * W, b[2] * W)]
    ys = [v for b in cajas for v in (b[1] * H, b[3] * H)]
    mx, my = 0.30 * (max(xs) - min(xs)), 0.18 * (max(ys) - min(ys))
    x0, y0 = min(xs) - mx, min(ys) - my
    x1, y1 = max(xs) + mx, max(ys) + my
    deg = (x1 - x0) < ASPECTO_MIN * (y1 - y0)
    if deg:
        cx, falta = 0.5 * (x0 + x1), ASPECTO_MIN * (y1 - y0)
        x0, x1 = cx - falta / 2.0, cx + falta / 2.0
    r = (int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1)))
    if r[2] - r[0] < 30 or r[3] - r[1] < 30:
        return None, deg
    return r, deg


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    for p in SALIDA.glob("*.png"):
        p.unlink()
    cs = casos()
    if not cs:
        raise SystemExit("no hay `rescate_poses_*.npz`: corre 01_extraccion/codigo/e29_tres_rondas.py")
    n_pasa = sum(1 for c in cs if c["pasa"])
    print("casos con ronda 2 intentada: %d en %d clips  ·  la ronda 2 pasa el filtro en %d (%.1f %%)"
          % (len(cs), len(set(c["clip"] for c in cs)), n_pasa, 100.0 * n_pasa / len(cs)))
    """SE REPARTE ENTRE CLIPS Y SE PREFIEREN LOS QUE LA RONDA 2 RESUELVE, pero no solo esos: se
    intercalan los que FALLAN, porque una galeria de solo aciertos no deja juzgar nada. El orden
    dentro de cada clip es por hueco descendente, que son los casos donde interpolar es peor."""
    porclip = {}
    for c in cs:
        porclip.setdefault(c["clip"], []).append(c)
    for v in porclip.values():
        v.sort(key=lambda c: (not c["pasa"], -c["hueco"]))
    eleg, i = [], 0
    while len(eleg) < CUANTAS and any(i < len(v) for v in porclip.values()):
        for v in porclip.values():
            if i < len(v) and len(eleg) < CUANTAS:
                eleg.append(v[i])
        i += 1

    porclip2 = {}
    for c in eleg:
        porclip2.setdefault(c["clip"], []).append(c)
    n = 0
    for clip, cc in sorted(porclip2.items()):
        zp = np.load(PRUEBAS / ("poses_%s.npz" % clip), allow_pickle=True)
        zr = np.load(PRUEBAS / ("rescate_poses_%s.npz" % clip), allow_pickle=True)
        KP, SC, B, Be = zp["KP"], zp["SC"], zp["B"], zp["Be"]
        rKP, rSC = zr["KP_rescate"], zr["SC_rescate"]
        W, H, T = int(zp["W"]), int(zp["H"]), int(zp["T"])
        quiere = set(c["k"] for c in cc)
        frames = {}
        cap = cv2.VideoCapture(str(zp["video"]))
        for k in range(T):
            ok, im = cap.read()
            if not ok:
                break
            if k in quiere:
                frames[k] = im
        cap.release()
        for c in cc:
            k, lado = c["k"], c["lado"]
            if k not in frames:
                continue
            rec, deg = recorte(Be, B, k, lado, W, H)
            if rec is None:
                continue
            fr = frames[k]
            hay2 = np.isfinite(rKP[k, lado, :, 0]).any()
            p1 = GI.panel(fr, None, None, Be[k, lado], W, H, VR.VIOLETA,
                          "1) ROUND 1 - YOLO: NOT DETECTED (estimated box%s)"
                          % (", DEGENERATE" if deg else ""), rec)
            et2 = ("2) ROUND 2 - Sapiens top-down: PASSES the filter, %d/63 pts" % c["n_util"]
                   if c["pasa"] else
                   "2) ROUND 2 - Sapiens top-down: FAILS the filter, %d/63 pts (needs 20)"
                   % c["n_util"] if hay2 else "2) ROUND 2 - no pose returned")
            p2 = GI.panel(fr, rKP[k, lado] if hay2 else None, rSC[k, lado] if hay2 else None,
                          Be[k, lado], W, H,
                          VR.RESCATADA if c["pasa"] else (110, 110, 130), et2, rec)
            er = GI.error_esperado(c["hueco"])
            p3 = GI.panel(fr, KP[k, lado], SC[k, lado], Be[k, lado], W, H, VR.VIOLETA,
                          "3) ROUND 3 - INTERPOLATED, what ships today (err ~%.1f %% of torso)" % er
                          if np.isfinite(er) else
                          "3) ROUND 3 - INTERPOLATED, what ships today", rec)
            paneles = [p1, p2, p3]
            alto = max(p.shape[0] for p in paneles)
            tira = np.zeros((alto + 30, GI.ANCHO * 3 + 16, 3), "uint8")
            x = 0
            for p in paneles:
                tira[30:30 + p.shape[0], x:x + GI.ANCHO] = p
                x += GI.ANCHO + 8
            fp, nfp = fp_del_hueco(c["hueco"])
            gana = ("ROUND 2 PASSES the filter -> a pose MEASURED on the estimated box (origen 3). "
                    "Same filter on EMPTY background at this gap: %.1f %% of %d"
                    % (fp, nfp) if c["pasa"] else
                    "ROUND 2 fails the filter -> the interpolated pose ships, and stays marked")
            if deg:
                gana += " | the estimated box DEGENERATED into a sliver"
            fondo = (24, 46, 24) if c["pasa"] else (30, 20, 42)
            cv2.rectangle(tira, (0, 0), (tira.shape[1], 28), fondo, -1)
            cv2.putText(tira, "%s  %s  c%04d  gap %d frame%s  |  %s" % (
                clip[:26], ["LEFT", "RIGHT"][lado], k, c["hueco"],
                "s" if c["hueco"] != 1 else "", gana),
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (235, 232, 245), 1, cv2.LINE_AA)
            nom = "%03d_%s_%s_%s_c%04d_hueco%02d.png" % (
                n + 1, "R2" if c["pasa"] else "R3", clip[:20], ["izq", "der"][lado], k, c["hueco"])
            cv2.imwrite(str(SALIDA / nom), tira)
            n += 1
    print("escritas %d imagenes en %s" % (n, SALIDA))
    el = eleg[:n]
    if el:
        print("de las %d: la ronda 2 gana en %d, la ronda 3 en %d  ·  hueco mediana %d, maximo %d"
              % (n, sum(1 for c in el if c["pasa"]), sum(1 for c in el if not c["pasa"]),
                 int(np.median([c["hueco"] for c in el])), max(c["hueco"] for c in el)))


if __name__ == "__main__":
    main()

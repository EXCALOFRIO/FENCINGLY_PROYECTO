"""LA CASCADA DE DETECCION DE LA POSE, en una figura con sus cifras, y con ejemplos de verdad.

LO QUE PIDIO EL USUARIO, literal: *"falta que pongas un ejemplo de alguna pose que no detectaba,
como con la redeteccion ya funciona, y me pongas el porcentaje que no detectaba y ya luego con la
redeteccion cuantas ha sacado para saberlo, y luego si no pues con el ultimo metodo que es que las
inventes viendo las otras"*.

O sea: las CUATRO rondas, cada una con **cuantas recupera**, y un caso que se pueda mirar.

LO QUE DISTINGUE LAS RONDAS ES **DE DONDE SALE LA CAJA**, NO EL MODELO DE POSE. Esto lo corrigio el
usuario -- *"si en la primera ronda estamos usando Sapiens 2, por que usas ahora..."* -- y tenia
razon: los **308 puntos los saca siempre la cabeza `pose` de Sapiens 2**
(`capas_limpias.Capas.pose`), y YOLO (`yolo11m-pose`) **solo aporta la caja**. Lo dice el propio
docstring de `redetecta`: «la pose sigue siendo un dato medido y solo la ventana donde mirar es
estimada».

    RONDA 0   caja de YOLO sobre el cuadro entero        -> pose de Sapiens 2
    RONDA 1   caja de YOLO sobre un RECORTE               -> pose de Sapiens 2   <- la REDETECCION
    RONDA 2   caja INTERPOLADA, sin detector ninguno      -> pose de Sapiens 2   <- solo hueco <= 24
    RONDA 3   ni caja ni pose: se INTERPOLA la pose entre las dos medidas de alrededor

La ronda 2 existe porque la cabeza de pose es **top-down**: no necesita que un detector confirme a la
persona, le basta una caja. Por eso puede intentarlo donde YOLO ya fallo dos veces. Y por eso la
pregunta «pues usa otra vez Sapiens 2 a ver si lo hace» **es exactamente la ronda 2**, y ya esta
medida: saca 50 de 295 con hueco <= 24, y con hueco >= 25 saca 2 de 355 mientras que sobre **fondo
vacio** saca 45 de 355, o sea que alli no distingue a un tirador de un trozo de suelo.

DOS SALIDAS:

  `cascada_deteccion.png`   el resumen con las cifras, dibujado **sobre negro**: ni un pixel de
                            emision, asi que puede ir al repo publico.
  `redeteccion_<n>.png`     casos REALES de la ronda 1, en dos paneles: a la izquierda el cuadro
                            entero con la caja donde YOLO **no** encontro a nadie, y a la derecha el
                            recorte con la pose que si salio. Es lo que el usuario pidio poder mirar.

LAS CIFRAS NO SE ESCRIBEN A MANO: se leen de `cascada_deteccion.json`, que las cuenta sobre los
`poses_*.npz` y `revision_*.npz`. Una cifra a mano en una figura publica envejece sin avisar, y este
proyecto ya tiene el caso documentado.

USO:  python figura_cascada_deteccion.py [CUANTOS]      CPU
"""
from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np

HERRAM = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERRAM))
CODIGO = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
sys.path.insert(0, str(CODIGO))

import video_revision as VR

PRUEBAS = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
SALIDA = pathlib.Path(
    r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\cascada_deteccion")
CUANTOS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
FUERA = {"MellowHideousAngora", "PowerfulAllLarva"}
"""Sable y el clip del arbitro, las mismas exclusiones que las otras galerias."""
COL_R1 = (90, 235, 120)


def cifras():
    f = PRUEBAS / "cascada_deteccion.json"
    if not f.is_file():
        raise SystemExit("falta %s: cuentalo primero" % f)
    return json.loads(f.read_text(encoding="utf-8"))


def _txt(im, t, x, y, col, esc=0.62, gr=1):
    cv2.putText(im, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, esc, col, gr, cv2.LINE_AA)


def resumen(c):
    """El resumen de las cuatro rondas SOBRE NEGRO, con una barra por ronda a escala real."""
    d = c["dentro"]
    W, H = 1620, 730
    im = np.zeros((H, W, 3), "uint8")
    im[:] = (12, 12, 18)
    _txt(im, "FENCINGLY - how a pose gets found, round by round", 24, 44, (245, 245, 250), 0.92, 2)
    _txt(im, "%s fencer-frames inside `mask`, %d clips. Every number below is counted from the "
             "delivered NPZ, not typed in." % ("{:,}".format(d), c["clips"]),
         24, 76, (160, 175, 200), 0.52)
    _txt(im, "The POSE is Sapiens 2 in every round. What changes is WHERE THE BOX COMES FROM - and "
             "round 3, where there is no box and no pose at all.", 24, 102, (225, 200, 120), 0.5)
    """LO QUE DISTINGUE LAS RONDAS ES LA CAJA, NO EL MODELO DE POSE, y decirlo al reves era un error
    de esta figura que corrigio el usuario: *"si en la primera ronda estamos usando Sapiens 2, por
    que usas ahora..."*. Tenia razon. Los 308 puntos los saca **siempre** la cabeza `pose` de
    Sapiens 2 (`capas_limpias.Capas.pose`); YOLO `yolo11m-pose` solo aporta **la caja**. Lo dice el
    propio docstring de `redetecta`: «la pose sigue siendo un dato medido y solo la ventana donde
    mirar es estimada». Rotularlo como «ronda 0 YOLO / ronda 2 Sapiens» sugeria que en las primeras
    rondas la pose la hacia YOLO, que es falso."""
    filas = [
        ("ROUND 0", "box from YOLO on the whole frame  ->  pose by Sapiens 2", c["t0"],
         (80, 210, 90), "the ordinary path"),
        ("ROUND 1", "box from YOLO on a CROP  ->  pose by Sapiens 2", c["t3"], COL_R1,
         "THE REDETECTION: same pose model, second look for the box"),
        ("ROUND 2", "box INTERPOLATED, no detector at all  ->  pose by Sapiens 2", c["resc"],
         (70, 230, 255),
         "top-down needs no detector, only a box: %d of %d tried, gap <= 24" % (c["resc"],
                                                                               c["intent"])),
        ("ROUND 3", "no box and no pose: INTERPOLATED from the ones around it", c["inv"] - c["resc"],
         (200, 80, 220), "nothing measured left - and it is marked as such"),
    ]
    y = 146
    x0, ancho = 560, 700
    for nom, que, n, col, nota in filas:
        pct = 100.0 * n / d
        _txt(im, nom, 24, y + 26, col, 0.72, 2)
        _txt(im, que, 24, y + 52, (215, 220, 235), 0.5)
        _txt(im, nota, 24, y + 74, (140, 155, 180), 0.44)
        """LA BARRA VA EN RAIZ CUADRADA Y SE DICE. Con escala lineal la ronda 0 (96,9 %) aplasta a
        las otras tres a menos de un pixel y la figura no ensena nada. La raiz deja ver las cuatro
        conservando el orden; el numero exacto va al lado, que es lo que se cita."""
        w = int(ancho * (pct / 100.0) ** 0.5)
        cv2.rectangle(im, (x0, y + 14), (x0 + max(w, 3), y + 52), col, -1)
        _txt(im, "%s   %5.2f %%" % ("{:,}".format(n).rjust(7), pct), x0 + max(w, 3) + 14, y + 44,
             (245, 245, 250), 0.6, 2)
        y += 118
    _txt(im, "bar length = sqrt of the percentage, so all four stay visible; the number is the truth",
         x0, 138, (120, 132, 155), 0.42)
    ymed = H - 86
    cv2.line(im, (24, ymed - 16), (W - 24, ymed - 16), (60, 64, 80), 1)
    _txt(im, "Without round 1 the pose would be missing in %.2f %% of fencer-frames. "
             "Round 1 recovers %s of them (%.2f pp)," % (100.0 * (d - c["t0"]) / d,
                                                         "{:,}".format(c["t3"]), 100.0 * c["t3"] / d),
         24, ymed + 8, (215, 225, 245), 0.52)
    _txt(im, "leaving %.2f %% with no measured pose. Round 2 recovers %d more, and %.2f %% ends up "
             "INTERPOLATED - never silently, always flagged."
             % (100.0 * c["inv"] / d, c["resc"], 100.0 * (c["inv"] - c["resc"]) / d),
         24, ymed + 32, (215, 225, 245), 0.52)
    _txt(im, "GENERATED FIGURE: drawn on black, no broadcast frame.", 24, H - 14,
         (205, 220, 245), 0.42)
    p = SALIDA / "cascada_deteccion.png"
    cv2.imwrite(str(p), im)
    print("  %s  %dx%d" % (p.name, W, H))
    return p


ALTO_PANEL = 620


def _a_alto(im, alto):
    e = alto / float(max(im.shape[0], 1))
    return cv2.resize(im, (max(1, int(round(im.shape[1] * e))), alto),
                      interpolation=cv2.INTER_AREA if e < 1 else cv2.INTER_LINEAR)


def casos_redeteccion(n_quiere):
    """Casos REALES de la ronda 1: YOLO fallo en el cuadro y la pose salio en el recorte."""
    cands = []
    for f in sorted(PRUEBAS.glob("revision_*.npz")):
        clip = f.name[len("revision_"):-len(".npz")]
        if clip in FUERA or not (PRUEBAS / ("poses_%s.npz" % clip)).is_file():
            continue
        z = np.load(f, allow_pickle=True)
        tipo, mask = z["tipo"], z["mask"].astype(bool)
        for lado in (0, 1):
            ks = np.nonzero((tipo[:, lado] == 3) & mask)[0]
            if len(ks):
                cands.append((clip, lado, int(ks[len(ks) // 2]), len(ks)))
    cands.sort(key=lambda c: -c[3])
    print("  clips con redeteccion: %d  ·  se pintan %d" % (len(cands), min(n_quiere, len(cands))))
    hechas = []
    for i, (clip, lado, k, n) in enumerate(cands[:n_quiere]):
        zp = np.load(PRUEBAS / ("poses_%s.npz" % clip), allow_pickle=True)
        zr = np.load(PRUEBAS / ("revision_%s.npz" % clip), allow_pickle=True)
        KP, SC = zp["KP"], zp["SC"]
        Be, W, H = zr["Be"], int(zr["W"]), int(zr["H"])
        cap = cv2.VideoCapture(str(zp["video"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, fr = cap.read()
        cap.release()
        if not ok:
            continue
        b = Be[k, lado]
        if not np.isfinite(b).all():
            continue
        x0, y0, x1, y1 = (int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H))
        mx, my = int(0.45 * (x1 - x0)), int(0.45 * (y1 - y0))
        rec = (max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my))
        izq = (fr.astype("float32") * 0.45).astype("uint8")
        cv2.rectangle(izq, (x0, y0), (x1, y1), (80, 80, 255), 3)
        _txt(izq, "ROUND 0: the detector found NO BOX here", max(8, x0 - 10), max(26, y0 - 12),
             (110, 110, 255), 0.8, 2)
        """LOS DOS PANELES SE ESCALAN A LA MISMA ALTURA, no al mismo ancho. El cuadro entero es
        16:9 y el recorte de un tirador es casi 1:2, asi que igualando el ancho uno salia de 427 px
        de alto y el otro de 1500 y la figura era medio lienzo negro. Igualando la ALTURA los dos
        ocupan lo suyo y se comparan de un vistazo, que es para lo que esta la figura."""
        izq = _a_alto(izq, ALTO_PANEL)
        sub = fr[rec[1]:rec[3], rec[0]:rec[2]].copy()
        k2 = KP[k, lado].copy()
        k2[:, 0] -= rec[0]
        k2[:, 1] -= rec[1]
        VR._pinta(sub, k2, SC[k, lado], COL_R1)
        cv2.rectangle(sub, (x0 - rec[0], y0 - rec[1]), (x1 - rec[0], y1 - rec[1]), COL_R1, 2)
        sub = _a_alto(sub, ALTO_PANEL)
        _txt(sub, "ROUND 1: box found on the crop, pose by Sapiens 2", 10, 28, COL_R1,
             0.62, 2)
        anc = izq.shape[1] + sub.shape[1] + 10
        tira = np.zeros((ALTO_PANEL + 66, anc, 3), "uint8")
        tira[:] = (12, 12, 18)
        tira[56:56 + ALTO_PANEL, :izq.shape[1]] = izq
        tira[56:56 + ALTO_PANEL, izq.shape[1] + 10:anc] = sub
        _txt(tira, "%s   %s   frame %d   -   same pose model, second look for the BOX"
             % (clip[:34], ["LEFT", "RIGHT"][lado], k), 12, 34, (240, 240, 248), 0.62, 1)
        _txt(tira, "this clip and side needed the crop in %d frames" % n, 12, ALTO_PANEL + 58,
             (150, 165, 195), 0.46)
        p = SALIDA / ("redeteccion_%02d_%s_%s_c%04d.png" % (i + 1, clip[:22],
                                                            ["izq", "der"][lado], k))
        cv2.imwrite(str(p), tira)
        hechas.append(p)
        print("     %s" % p.name)
    return hechas


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    c = cifras()
    print("=" * 96)
    print("LA CASCADA DE DETECCION  ·  cuatro rondas con su cifra  ·  sin GPU")
    print("=" * 96)
    resumen(c)
    casos_redeteccion(CUANTOS)
    print("\n  salida en %s" % SALIDA)


if __name__ == "__main__":
    main()

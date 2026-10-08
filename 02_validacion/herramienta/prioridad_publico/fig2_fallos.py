"""FIGURA 2: DONDE se rompe. Los cuatro fallos concretos, cada uno sobre su fotograma.

Son los mismos casos de `fig_fallo_*.png`, rehechos en ingles, con las cajas y las cifras que los
delatan y con la causa dicha: dos de los cuatro NO son de este modulo.
"""
from __future__ import annotations

import cv2
import numpy as np

from comun import (AMARILLO, BLANCO, COL, FD, LADO, ROJO, SinPisarse, VERDE, VIOLETA, apilar,
                   banda, cabecera, caja, carga, cuadro_de, en, geo, guarda, marco, pinta_cono,
                   recorta_escala, recuadro, texto, ventana, CL, PR, VP)

ANCHO_CELDA, ALTO_IMG = 960, 540

CASOS = [
    dict(cid="PowerfulAllLarva", k=162, rot="A", entero=True,
         tit="A BYSTANDER PICKED AS A FENCER",
         que=[("The GREEN box, i.e. what the pipeline calls \"the LEFT fencer\", is on a person in "
               "dark street clothes walking across the", VERDE),
              ("foreground. The two real foilists are the pair on the LEFT and only ONE of them "
               "got a box. And this pose is origen = 0, MEASURED,", VERDE),
              ("so nothing stops it from deciding: the hands come out {alcance} m apart, which is "
               "why the t.9.1 reach bound is the only thing saving it.", AMARILLO),
              ("Right of way for this whole clip is garbage BY CONSTRUCTION - and the bug is in "
               "the PAIR SELECTOR, not in this module.", AMARILLO),
              ("NOTE ON THE FRAME NUMBER: the existing sheet is named "
               "fig_fallo_intruso_PowerfulAllLarva_c147.png, but c147 is where the STRIP starts.",
               (150, 205, 255)),
              ("The intruder box does not exist yet at c147; it runs c157-c199. Same fault, "
               "different frame.", (150, 205, 255))]),
    dict(cid="MellowHideousAngora", k=231, rot="B", entero=True,
         zoom=(8, 8, 312, 148),
         tit="IT IS SABRE, NOT FOIL",
         que=[("Read the inset: the broadcast overlay says \"Women's Sabre WC\" and the poster "
               "behind says SABRE FEMININ. Sabre right of way", (196, 196, 202)),
              ("is FIE t.100 and following, NOT t.82-t.89,", (196, 196, 202)),
              ("so every rule this detector applies is the wrong rule book for this clip. "
               "Nothing about the poses looks wrong; the SPORT is wrong.", (196, 196, 202)),
              ("It leaked in through the old sampling that the project's own master document had "
               "already declared broken. EXCLUDED from every", AMARILLO),
              ("priority measurement (p1_variables.py keeps it in FUERA). 1 clip of 21; the other "
               "20 are foil.", AMARILLO)]),
    dict(cid="PotablePossibleIcterinewarbler", k=233, rot="C", entero=False,
         tit="BOXES MISSING IN THE MIDDLE OF THE ACTION",
         que=[("The LEFT fencer has NO measured box at c233, c236 and c245 - the three frames are "
               "inside the action, not before or after it.", (196, 196, 202)),
              ("With only one of the two poses there is no decision to make: the correct output is "
               "\"nobody\", for LACK OF DATA, and saying", (196, 196, 202)),
              ("anything else would be inventing it. The violet box is the interpolated fallback "
               "(Be), drawn so the gap is visible.", (210, 170, 235)),
              ("Low-quality video, and the annotator graded this window confidence = LOW. "
               "The two fencers attack almost together.", AMARILLO)]),
    dict(cid="AliveIgnorantBarebirdbat", k=213, rot="D", entero=False,
         tit="RIGHT OF WAY AWARDED WITH NO OPPONENT IN FRAME",
         que=[("The published bug: \"RIGHT, confidence 1.00\" while the left fencer is a 86-pixel "
               "sliver at the edge of the frame and the", ROJO),
              ("right one is walking back with his arm hanging down. Confidence 1.00 for a "
               "decision that had nothing to decide about.", ROJO),
              ("THE FIX, and it is a rule, not a tuned threshold: a foil blade is {hoja} m "
               "(FIE m.8.4), so two points cannot meet with the", AMARILLO),
              ("hands more than 2 blades apart, plus the measured p90 closing of 1.48 m => bound "
               "{cota} m. Here the hands are {alcance} m apart.", AMARILLO),
              ("WHAT YOU SEE ABOVE IS THE OUTPUT AFTER THE FIX: NOBODY. The before/after pair is "
               "fig_fallo_ / fig_arreglado_prioridad_sin_adversario_Alive_c213.png.", VERDE),
              ("And the bound was WRONG the first time, which the measurement caught: with the "
               "1.80 m bound alone, accuracy on frames with an owner fell 25.6 % -> 3.3 %.",
               VERDE)]),
]


def _n(x):
    return "n/a" if not np.isfinite(x) else "%.2f" % float(x)


def _inset(img, orig, roi, W, H, ancho=430):
    """Amplia una region del fotograma ORIGINAL y la pega en la esquina del panel, con su rotulo.

    Se recorta del original y no del panel ya escalado: si se ampliara el panel, se ampliaria el
    remuestreo y el cartel del fondo, que es la prueba, saldria ilegible.
    """
    x0, y0, x1, y1 = roi
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    sub = orig[y0:y1, x0:x1]
    if sub.size == 0:
        return None
    alto = int(round(ancho * (y1 - y0) / float(x1 - x0)))
    z = cv2.resize(sub, (ancho, alto), interpolation=cv2.INTER_CUBIC)
    px, py = img.shape[1] - ancho - 12, 46
    img[py:py + alto, px:px + ancho] = z
    cv2.rectangle(img, (px - 2, py - 2), (px + ancho + 1, py + alto + 24), AMARILLO, 2)
    cv2.rectangle(img, (px - 1, py + alto), (px + ancho, py + alto + 23), (16, 16, 18), -1)
    texto(img, "ZOOM of the same frame, top-left corner", (px + 6, py + alto + 17), 0.45,
          AMARILLO, 1, sombra=False)
    return (px - 2, py - 2, ancho + 4, alto + 26)


def un_panel(c):
    cid, k = c["cid"], c["k"]
    d = carga(cid)
    s = PR.senales(d)
    W, H, T = int(d["W"]), int(d["H"]), int(d["T"])
    r = PR.decide(s, d["mask"], T)
    g = geo(d, s)
    det = int(r["prio"][k])

    im = cuadro_de(str(d["video"]), k)
    assert im is not None, "no se pudo leer %s c%d" % (cid, k)
    vis = im.copy()
    for lado in (0, 1):
        if int(d["origen"][k, lado]) == 0:
            pinta_cono(vis, g, lado, k, COL[lado])
    cajas = {}
    for lado in (0, 1):
        inv = int(d["origen"][k, lado]) > 0
        col = VIOLETA if inv else COL[lado]
        if not inv:
            VP._pinta(vis, d["KP"][k, lado], d["SC"][k, lado], col, g[lado]["armado"], grueso=2)
        b, inv = caja(d, k, lado)
        if b is not None:
            recuadro(vis, b, W, H, col, 4 if inv else 3)
        cajas[lado] = (b, inv)

    vent = (0, 0, W, H) if c["entero"] else ventana(d, k, W, H, margen=0.22,
                                                    ar=ANCHO_CELDA / float(ALTO_IMG))
    alto = int(round(ANCHO_CELDA * (vent[3] - vent[1]) / float(vent[2] - vent[0])))
    alto = min(alto, ALTO_IMG)
    img, tr = recorta_escala(vis, vent, ANCHO_CELDA, alto)
    if alto < ALTO_IMG:
        pad = (ALTO_IMG - alto) // 2
        img = np.pad(img, ((pad, ALTO_IMG - alto - pad), (0, 0), (0, 0)), constant_values=18)

        def tr2(x, y, _t=tr, _p=pad):
            a, b2 = _t(x, y)
            return a, b2 + _p
    else:
        tr2 = tr

    zona = _inset(img, im, c["zoom"], W, H) if c.get("zoom") else None
    sp = SinPisarse(ALTO_IMG)
    if zona is not None:
        """El recuadro ampliado se SIEMBRA en `SinPisarse` antes de colocar un solo rotulo, porque
        si no el rotulo de la caja de la derecha cae justo encima de la ampliacion: la clase solo
        sabe esquivar lo que ya esta en su lista."""
        zx, zy, zw, zh = zona
        for yy in range(zy, zy + zh + 1, 12):
            for xx in range(zx - 200, zx + zw + 1, 100):
                sp.puestos.append((xx, yy))
    for lado in (0, 1):
        b, inv = cajas[lado]
        if b is None:
            continue
        x, y = tr2(b[0] * W, b[1] * H)
        et = "%s (side %d)" % (LADO[lado], lado)
        if inv:
            et = "%s (side %d): INVENTED POSE, cannot decide" % (LADO[lado], lado)
        x = max(6, min(x, ANCHO_CELDA - 340))
        yy = sp.sitio(x, max(22, y - 10))
        texto(img, et, (x, yy), 0.55, VIOLETA if inv else COL[lado], 2)
    cv2.rectangle(img, (0, 0), (ANCHO_CELDA, 34), (16, 16, 18), -1)
    texto(img, "%s   %s  c%d   (%dx%d)" % (c["rot"], cid, k, W, H), (12, 24), 0.55, BLANCO, 1, FD,
          sombra=False)
    texto(img, "detector: %s" % LADO[det], (ANCHO_CELDA - 250, 24), 0.6, COL[det], 2, FD,
          sombra=False)
    marco(img)

    val = {"alcance": _n(s["alcance"][k]), "hoja": "%.2f" % CL.HOJA_M,
           "cota": "%.2f" % PR.ALCANCE_MAX}
    lineas = [(c["tit"], ROJO, 0.62)]
    for q, col in c["que"]:
        lineas.append((q.format(**val), col or (196, 196, 202), 0.45))
    lineas.append(("rule the detector applied at this frame: %s" % en(r["regla"][k]),
                   (150, 205, 255), 0.45))
    pie = banda(lineas, ANCHO_CELDA, alto_linea=23, margen=10, fondo=(24, 24, 27))
    marco(pie, (70, 70, 74), 1)
    return np.vstack([img, pie])


def main():
    cel = [un_panel(c) for c in CASOS]
    alto = max(x.shape[0] for x in cel)
    cel = [np.pad(x, ((0, alto - x.shape[0]), (0, 0), (0, 0)), constant_values=24) for x in cel]
    grid = np.vstack([np.hstack([cel[0], cel[1]]), np.hstack([cel[2], cel[3]])])
    W = grid.shape[1]

    cab = cabecera(W, "WHERE THE RIGHT-OF-WAY DETECTOR BREAKS: four faults, each on its frame",
                   ["Found by looking for them, on purpose. Two of the four are NOT this module's "
                    "fault, and saying which is the point of the figure.",
                    "None of these four is fixed by tuning a threshold."], col_tit=(120, 170, 255))

    leg = banda([
        ("WHAT THESE FOUR FAULTS COST, AND WHO OWNS THEM", BLANCO, 0.58),
        ("A - bystander picked as a fencer: owner is the PAIR SELECTOR upstream. Same clip and "
         "same cause as the pair-selection section of the master document.", (196, 196, 202), 0.45),
        ("B - sabre clip in a foil corpus: owner is the SAMPLING. Excluded, and the exclusion is "
         "in the code (FUERA = {\"MellowHideousAngora\"}), not only in prose.",
         (196, 196, 202), 0.45),
        ("C - missing boxes mid-action: owner is the DETECTOR/TRACKER upstream. The right-of-way "
         "module can only report \"no data\", and it does.", (196, 196, 202), 0.45),
        ("D - right of way with no opponent: owner is THIS MODULE, and it is the only one of the "
         "four it owns. Fixed by a reach bound taken from the rule book.", AMARILLO, 0.45),
        ("AND ONE FAULT THAT WAS THE HUMAN'S, kept because it points the other way: in "
         "GentleQuerulousElephantseal c270-c288 the annotation said the RIGHT fencer and the "
         "detector said the LEFT one.", VERDE, 0.45),
        ("   The per-fencer zoom sheets showed the LEFT fencer with his arm already extended at "
         "c279 while the right one still had his bent. THE ALGORITHM CONTRADICTED THE HUMAN AND "
         "WAS RIGHT.", VERDE, 0.45),
        ("   That is why the annotation carries a revisado_zoom column: 9 of the 21 windows were "
         "re-checked with per-fencer zoom and exactly 1 changed side. See "
         "fig_correccion_anotacion_Gentle_c270.png.", VERDE, 0.45),
        ("WHAT IS NOT ON THIS FIGURE, because there is no measurement for it: PARRIES. A parry is "
         "blade-on-blade contact by definition, the blade is seen in 1.6 % of 720p crops, and "
         "there are 8 candidates", ROJO, 0.45),
        ("   from the t.86 criterion against ZERO parries in the 21 annotated windows. With n = 0 "
         "there is no accuracy figure, and claiming \"it detects parries\" would be making it up.",
         ROJO, 0.45),
        ("ALSO NOT IMPLEMENTED: 5 of the 12 FIE rules that award right of way (t.84.1, t.84.2, "
         "t.85, t.86, t.87) require SEEING THE BLADE. A phrase decided by a parry is out of scope "
         "today.", ROJO, 0.45),
    ], W, alto_linea=23, margen=14)

    guarda("fig2_where_it_breaks.png", apilar(cab, grid, leg))


if __name__ == "__main__":
    main()

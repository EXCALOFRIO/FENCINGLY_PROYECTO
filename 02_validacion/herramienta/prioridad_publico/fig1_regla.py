"""FIGURA 1: COMO se decide la prioridad, sobre fotogramas reales.

Cuatro momentos de dos frases de armas de verdad: nadie ataca, ataque simple, contraataque y accion
simultanea. Cada panel lleva las dos cajas, los dos conos, el esqueleto, quien tiene el paso, el
articulo FIE que lo concede y LOS NUMEROS con que se decidio.
"""
from __future__ import annotations

import csv

import cv2
import numpy as np

from comun import (ALCANCE_M, AMARILLO, BLANCO, COL, ENVOL_SEMI, FD, LADO, NUCLEO_SEMI, PRIO,
                   ROJO, SinPisarse, VERDE, VIOLETA, apilar, banda, cabecera, caja, carga,
                   cuadro_de, en, geo, guarda, marco, pinta_cono, recorta_escala, recuadro,
                   texto, ventana, CL, PR, VP)

ANCHO_CELDA, ALTO_IMG = 960, 486

CASOS = [
    dict(cid="KindEnchantingKid", k=220, rot="A",
         tit="NOBODY has it: no attack has started",
         que=["Both arms are bent and neither point threatens the valid target.",
              "t.82: an attack must be *initiated* before it can be given right of way."]),
    dict(cid="KindEnchantingKid", k=228, rot="B",
         tit="SIMPLE ATTACK: the RIGHT fencer takes it",
         que=["His arm extension starts BEFORE his lunge, and the left fencer was",
              "still stepping in with a bent arm, so it lands on the preparation."]),
    dict(cid="KindEnchantingKid", k=233, rot="C",
         tit="COUNTER-ATTACK: the RIGHT fencer KEEPS it",
         que=["The left fencer now extends too, but extending into an attack that",
              "already has right of way does not take it away. It is a counter-attack."]),
    dict(cid="AliveIgnorantBarebirdbat", k=283, rot="D",
         tit="SIMULTANEOUS ACTION: NOBODY has it",
         que=["Both arms start inside one fencing time (measured: 5 frames = 167 ms), so the",
              "rules give right of way to neither. Answering \"nobody\" here is the RULE, not a bug",
              "- but the human annotator read this one as an attack by the RIGHT fencer.",
              "Article label as published; a re-run today emits \"nobody has initiated a correct",
              "attack\" at this frame, same answer by another route. See fig3, panel 5."]),
]


def publicado():
    """La verdad anotada y la salida del detector TAL COMO SE PUBLICARON, leidas del CSV.

    No se recalculan a proposito: una repeticion de hoy da 22 de 462 decisiones distintas (esta
    medido en el panel 5 de la fig3), y una figura que ensene un numero que no esta en
    `p2_salida.txt` seria un numero mas para la misma cosa.
    """
    out = {}
    with open(PRIO / "p2_por_cuadro.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.setdefault((r["clip"], int(r["cuadro"])), r)
    return out


def un_panel(c, pub):
    cid, k = c["cid"], c["k"]
    d = carga(cid)
    s = PR.senales(d)
    W, H, T = int(d["W"]), int(d["H"]), int(d["T"])
    r = PR.decide(s, d["mask"], T)
    g = geo(d, s)
    fila = pub.get((cid, k))
    verdad = int(fila["verdad"]) if fila else None
    det = int(fila["detector"]) if fila else int(r["prio"][k])
    regla = fila["regla"] if fila else r["regla"][k]
    conf = float(fila["conf"]) if fila else float(r["conf"][k])

    im = cuadro_de(str(d["video"]), k)
    assert im is not None, "no se pudo leer %s c%d" % (cid, k)
    vis = im.copy()
    conos = {}
    for lado in (0, 1):
        conos[lado] = pinta_cono(vis, g, lado, k, COL[lado])
    cajas = {}
    for lado in (0, 1):
        inv = int(d["origen"][k, lado]) > 0
        col = VIOLETA if inv else COL[lado]
        VP._pinta(vis, d["KP"][k, lado], d["SC"][k, lado], col, g[lado]["armado"],
                  grueso=4 if lado == det else 2)
        b, inv = caja(d, k, lado)
        if b is not None:
            recuadro(vis, b, W, H, col, 4 if lado == det else 2)
        cajas[lado] = (b, inv)

    vent = ventana(d, k, W, H, ar=ANCHO_CELDA / float(ALTO_IMG))
    img, tr = recorta_escala(vis, vent, ANCHO_CELDA, ALTO_IMG)
    marco(img)

    sp = SinPisarse(ALTO_IMG)
    for lado in (0, 1):
        b, inv = cajas[lado]
        if b is None:
            continue
        x, y = tr(b[0] * W, b[1] * H)
        et = "%s (side %d)" % (LADO[lado], lado)
        if lado == det:
            et += "  <== RIGHT OF WAY"
        if inv:
            et = "%s (side %d)  INVENTED POSE: cannot decide" % (LADO[lado], lado)
        yy = sp.sitio(max(6, min(x, ANCHO_CELDA - 300)), max(20, y - 10))
        texto(img, et, (max(6, min(x, ANCHO_CELDA - 300)), yy), 0.56,
              VIOLETA if inv else COL[lado], 2)
    cv2.rectangle(img, (0, 0), (ANCHO_CELDA, 34), (16, 16, 18), -1)
    texto(img, "%s   %s  c%d" % (c["rot"], cid, k), (12, 24), 0.58, BLANCO, 1, FD, sombra=False)
    texto(img, "RIGHT OF WAY: %s" % LADO[det], (ANCHO_CELDA - 330, 24), 0.62, COL[det], 2, FD,
          sombra=False)
    marco(img)

    ex = s["ext_hr"][k]
    pi = s["pies"][k]
    mn = s["manos"][k]
    al = s["alcance"][k]
    lineas = [(c["tit"], COL[det], 0.62)]
    for q in c["que"]:
        lineas.append((q, (196, 196, 202), 0.45))
    lineas.append(("rule the detector applied: %s" % en(regla), AMARILLO, 0.45))
    lineas.append((
        "measured here:  ext_hr L=%s R=%s   feet-spread L=%s R=%s   hands %s torsos = %s m"
        % (_n(ex[0]), _n(ex[1]), _n(pi[0]), _n(pi[1]), _n(mn), _n(al)),
        (150, 205, 255), 0.45))
    lineas.append((
        "hand annotation says %s  |  detector says %s  |  detector confidence %.2f  |  %s"
        % (LADO[verdad] if verdad is not None else "no ground truth", LADO[det], conf,
           "AGREE" if verdad == det else "DISAGREE"),
        VERDE if verdad == det else ROJO, 0.45))
    pie = banda(lineas, ANCHO_CELDA, alto_linea=23, margen=10, fondo=(24, 24, 27))
    marco(pie, (70, 70, 74), 1)
    return np.vstack([img, pie])


def _n(x):
    return "n/a" if not np.isfinite(x) else "%.2f" % float(x)


def main():
    pub = publicado()
    cel = [un_panel(c, pub) for c in CASOS]
    alto = max(x.shape[0] for x in cel)
    cel = [np.pad(x, ((0, alto - x.shape[0]), (0, 0), (0, 0)), constant_values=24) for x in cel]
    grid = np.vstack([np.hstack([cel[0], cel[1]]), np.hstack([cel[2], cel[3]])])
    W = grid.shape[1]

    cab = cabecera(W, "FOIL RIGHT OF WAY (priority): how the decision is made",
                   ["FIE Technical Rules t.82-t.89 (foil conventions). Real frames, real poses, "
                    "real detector output - nothing is staged.",
                    "Every number on this figure is dimensionless and normalised by the fencer's "
                    "own torso length, except the hand distance in metres."])

    leg = banda([
        ("WHAT YOU ARE LOOKING AT", BLANCO, 0.58),
        ("GREEN skeleton and box = the LEFT fencer (side 0).   ORANGE = the RIGHT fencer "
         "(side 1).   VIOLET = an invented (interpolated) pose, which is never allowed to "
         "decide.", (196, 196, 202), 0.45),
        ("THICK box and thick skeleton = the fencer the detector says holds right of way. "
         "The yellow segment is the armed forearm, elbow -> wrist, and it is the axis "
         "everything else is built on.", (196, 196, 202), 0.45),
        ("THE THREAT CONE, drawn with the same functions and the same apertures the measurement "
         "uses (e21_pinta_cono_arreglado.rellena_peso / arco, cono3d_lib apertures):",
         (196, 196, 202), 0.45),
        ("   solid inner arc = CORE, semi-angle +-%.1f deg, i.e. %.1f deg of full cone."
         "   dashed outer arc = ENVELOPE, semi-angle +-%.1f deg, i.e. %.1f deg of full cone."
         % (NUCLEO_SEMI, 2 * NUCLEO_SEMI, ENVOL_SEMI, 2 * ENVOL_SEMI), AMARILLO, 0.45),
        ("   length %.2f m from the WRIST to the point. The blade alone is %.2f m (FIE m.8.4), "
         "the extra is the grip and guard in front of the wrist."
         % (ALCANCE_M, CL.HOJA_M), (196, 196, 202), 0.45),
        ("   the fill opacity IS the weight function the measurement scores with "
         "(1 - angle/aperture), not a flat tint, so the picture and the number cannot disagree.",
         (196, 196, 202), 0.45),
        ("THE BLADE IS NEVER DRAWN. It is detected in 1.6 % of 720p crops against 42.6 % at "
         "1080p, and 20 of the 21 foil clips are 720p. Drawing 90 cm of steel from the hand is "
         "forbidden in this project.", (150, 205, 255), 0.45),
        ("THE VARIABLES.  ext_hr = |shoulder -> wrist| / torso: this is the one that grows when "
         "the arm extends, and it is the signal for t.83.2.a and t.83.2.d.", (196, 196, 202), 0.45),
        ("   feet-spread = |ankle -> ankle| / torso, the lunge indicator.   hands = distance "
         "between the two armed wrists.   elbow-to-wrist was measured and REJECTED: it is a bone, "
         "it does not lengthen.", (196, 196, 202), 0.45),
        ("THE RULE THAT MAKES THIS MEASURABLE WITHOUT SEEING THE BLADE - t.83.2.a: a simple attack "
         "is correctly executed when the arm extension, point threatening the valid target, "
         "PRECEDES the start of the lunge.", AMARILLO, 0.45),
        ("   So the correctness of an attack is the SIGN OF A DIFFERENCE OF TWO INSTANTS, and two "
         "instants on a pose series can be measured. That is why there is a detector and not only "
         "a document.", (196, 196, 202), 0.45),
        ("FENCING TIME (t.8) is not given in seconds by the rules, so it was measured: median 5 "
         "frames = 167 ms at 30 fps, over 21 foil clips. That sets the simultaneity window of "
         "t.89.2 and the grace interval.", (196, 196, 202), 0.45),
        ("AND THE HONEST PART: this figure shows the decision, not a working system. The same "
         "detector scores 69.7 % against 73.8 % for answering \"nobody\" every single time. "
         "See fig3 and fig4.", ROJO, 0.47),
    ], W, alto_linea=23, margen=14)

    guarda("fig1_how_right_of_way_is_decided.png", apilar(cab, grid, leg))


if __name__ == "__main__":
    main()

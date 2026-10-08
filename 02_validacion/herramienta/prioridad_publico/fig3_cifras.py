"""FIGURA 3: la comparacion honesta. 69,7 % contra 73,8 %, y el atajo de las luces APARTE.

LA DECISION DE DISENO QUE MAS IMPORTA AQUI: el 64,96 % del atajo de las luces **no va en el mismo
eje** que el 69,7 % del detector, y no es estetica. Son preguntas distintas con denominadores
distintos:

    69,7 %  = "quien tiene la prioridad EN ESTE CUADRO", sobre  462 CUADROS anotados de 21 ventanas
    64,96 % = "quien gano el tocado EN ESTE CLIP",       sobre 1.815 CLIPS de dos luces

Ponerlos en la misma barra invita a restarlos, y la resta no significa nada. Van en dos paneles, con
su n escrito cada uno y el aviso puesto entre ellos.
"""
from __future__ import annotations

import numpy as np

from barras import (AQUA, AVISO, AZUL, BUENO, CRITICO, NARANJA, REJILLA, SUP, TINTA, TINTA2,
                    TINTA3, nota, tile)
from comun import apilar, banda, cabecera, guarda

W = 1920
MED = 948


def main():
    cab = cabecera(W, "IS THE RIGHT-OF-WAY DETECTOR ANY GOOD? No - and this is the figure that "
                      "says so",
                   ["Every panel carries its own n. Nothing here is averaged with anything on "
                    "another panel, because the panels do not share a denominator.",
                    "Source: 02_validacion/prioridad/p2_salida.txt and p2_por_cuadro.csv, "
                    "session 2026-09-10. Neither the hit label nor the light order is used in "
                    "panels 1-5."])

    p1 = tile(
        W, "1 - FRAME-LEVEL ACCURACY: the detector loses to answering \"nobody\" every time",
        "Question: who holds right of way in THIS frame. Ground truth annotated by hand from "
        "contact sheets.",
        "n = 462 frames, 21 windows, 10 foil clips",
        [dict(et=["detector (prioridad.py)"], v=69.7, col=CRITICO, destaca=True),
         dict(et=["trivial: answer \"NOBODY\" always",
                  "costs nothing, knows nothing"], v=73.8, col=AVISO),
         dict(et=["baseline: \"whoever advances first\"",
                  "box motion only, no pose"], v=34.6, col=TINTA3)],
        ref=(73.8, AVISO, ["FREE BASELINE 73.8 %",
                           "the detector is 4.1 points BELOW it"]),
        pies=[("WHY THE FREE BASELINE IS SO HIGH: 12 of the 21 annotated windows are \"nobody\" "
               "from start to finish, which is 73.8 % of the frames.", TINTA2),
              ("A detector that never fires scores that without knowing anything - and one that "
               "does fire pays for it in those windows. So 69.7 % is not a partial success; it "
               "is a loss.", TINTA2)])

    p2 = tile(
        MED, "2 - THE SAME 462 FRAMES, SPLIT IN TWO", None,
        "n = 121 frames WITH an owner  /  341 WITHOUT",
        [dict(et=["detector"], v=25.6, col=AZUL),
         dict(et=[""], v=85.3, col=AQUA),
         dict(et=["baseline \"advances first\""], v=41.3, col=AZUL, hueco=18),
         dict(et=[""], v=32.3, col=AQUA),
         dict(et=["trivial \"nobody\""], v=0.0, col=AZUL, hueco=18),
         dict(et=[""], v=100.0, col=AQUA)],
        etq=300, mar_der=130, alto_fila=40, grosor=22,
        leyenda=[("frames WITH an owner (n = 121)", AZUL),
                 ("WITHOUT an owner (n = 341)", AQUA)],
        pies=[("THE REAL SHAPE OF THE FAILURE. When it must name an owner it gets 25.6 %, "
               "WORSE than the crude box baseline at 41.3 %.", TINTA2),
              ("When there is no owner it gets 85.3 %, worse than saying nothing at 100 %. "
               "It is beaten on both halves of the job,", TINTA2),
              ("which is why no threshold sweep fixes it: there is no operating point where it "
               "wins.", AVISO)])

    p3 = tile(
        MED, "3 - WHEN IT DOES COMMIT THE SIDE IS RIGHT. THE COVERAGE IS NOT.", None,
        "three questions, three different denominators",
        [dict(et=["side agreement when BOTH", "name an owner   (n = 36)"], v=86.1, col=BUENO,
              destaca=True),
         dict(et=["how often it commits at all", "36 of 121 frames  (n = 121)"], v=29.8,
              col=TINTA3),
         dict(et=["false alarm: invents an owner", "where there is none  (n = 341)"], v=14.7,
              col=CRITICO),
         dict(et=["instant of the change, +-3 frames", "2 of 10 changes  (n = 10)"], v=20.0,
              col=CRITICO)],
        etq=330, mar_der=130, alto_fila=52, grosor=22,
        pies=[("This is the one encouraging number on the page, and it has to be stated "
               "precisely: THE SIDE IS GOOD, THE COVERAGE IS BAD.", TINTA2),
              ("All four denominators are written out, because 36, 121, 341 and 10 are not the "
               "same 100 %. This project has already been bitten once by", TINTA2),
              ("the same \"8\" meaning \"8 of 10 clips\" in one place and \"8 of 22 fencers\" in "
               "another.", AVISO)])

    p4 = nota(
        MED, "4 - THE ONLY NUMBER THAT IS NOT INFLATED",
        [("0.0 % on the held-out frames that DO have an owner: on unseen clips it does not "
          "answer at all.", TINTA2),
         ("The definition fixes were found by staring at the signal in 5 of the 21 tuning "
          "windows, so 69.7 % is inflated by construction.", TINTA2),
         ("9.1 % is the honest one - and it is only 44 frames, which is said out loud instead of "
          "hidden.", AVISO)],
        CRITICO, (26, 22, 30),
        hero=("9.1 %", CRITICO,
              [("on the HELD-OUT batch: 2 windows, n = 44 frames,", TINTA),
               ("annotated AFTER the code was frozen, on clips never looked at.", TINTA),
               ("WhisperedTenderDeinonychus c268-c289 and UglyNiceAplomadofalcon c145-c166.",
                TINTA2)]))

    p5 = nota(
        MED, "5 - AND THE 69.7 % DOES NOT FULLY REPRODUCE",
        [("Re-running prioridad.decide() today over the same 462 frames gives 70.35 %, not "
          "69.70 %. 22 of the 462 frame decisions changed.", TINTA),
         ("Two clips moved most:  ForcefulAntiqueAlbacoretuna 68.2 -> 100.0 % (n = 22)   and   "
          "TestyWatchfulBarebirdbat 84.1 -> 70.5 % (n = 44).", TINTA2),
         ("The verdict does not move: 69.70 % and 70.35 % both lose to the free 73.8 %. "
          "The rule labels also shifted on 14 frames of one clip", TINTA2),
         ("(AliveIgnorantBarebirdbat c279-c292, published as t.89.2, re-run as \"no correct "
          "attack\" - same answer, different article).", TINTA2),
         ("THE FIGURES ON THIS PAGE KEEP THE PUBLISHED 69.7 %, because that is what "
          "p2_salida.txt says. Two measurements, two numbers,", AVISO),
         ("reported as two numbers - not averaged into a third that nobody ever measured.",
          AVISO)],
        AVISO, (24, 26, 34), alto_min=p4.shape[0])

    p6 = tile(
        W, "6 - A DIFFERENT QUESTION ON A DIFFERENT SAMPLE: the light-order shortcut",
        "Question: who WON the touch in this CLIP. Clip-level, not frame-level. Read the red box "
        "before comparing anything here with panel 1.",
        "n = 1,815 two-light clips with a resolvable light order",
        [dict(et=["\"the fencer who lights SECOND wins\""], v=64.96, col=NARANJA, destaca=True,
              txt="64.96 %"),
         dict(et=["majority baseline (always the same side)"], v=52.18, col=TINTA3,
              txt="52.18 %"),
         dict(et=["\"the fencer who lights FIRST wins\""], v=35.04, col=TINTA3, txt="35.04 %")],
        etq=420, mar_der=760,
        pies=[("Kept on the page because it is the cheapest thing in the project that beats "
               "chance, so the right-of-way work has to be judged next to it: the light order is "
               "a SHORTCUT, not a solution.", TINTA2),
              ("It reproduces the 63.51 % the project already had, and it settles a sign that "
               "was never written down: the shortcut is that the SECOND light wins. That has "
               "fencing sense - the", TINTA2),
              ("counter-attacker usually touches first and loses on right of way. And the "
               "sample is not free either: only 84.0 % of two-light clips have a resolvable "
               "order, and of those 38.4 %", TINTA2),
              ("have |dt| <= 40 ms, below one frame at 25 fps. The shortcut already sank a "
               "model's ceiling to 73.6 % once, which is why no measurement in panels 1-5 "
               "touches it.", AVISO)])

    _aviso_en(p6)

    pie = banda([
        ("EVERY DENOMINATOR ON THIS PAGE, IN ONE PLACE", TINTA, 0.58),
        ("  462 frames   21 annotated windows, 10 foil clips           ->  panels 1 and 2, and "
         "the 69.7 % headline", TINTA2, 0.455),
        ("               and 462 is OBSERVATIONS, not distinct frames: two annotated windows of "
         "GentleQuerulousElephantseal overlap on c282-c288, so 7 frames are counted twice, over "
         "455 distinct frames.", TINTA3, 0.455),
        ("               Both copies carry the same ground truth and the same detector output, so "
         "the double count does not bias the accuracy - but 462 is not 455 and the difference is "
         "written down here.", TINTA3, 0.455),
        ("  121 / 341    the same 462 frames split by whether the annotation names an owner  ->  "
         "panel 2", TINTA2, 0.455),
        ("   36 frames   the subset of those 121 where the detector ALSO names an owner        ->"
         "  panel 3, the 86.1 % side agreement", TINTA2, 0.455),
        ("   10 changes  right-of-way changes annotated by hand, in 15.4 s of action           ->"
         "  panel 3, the 2-of-10 instant accuracy", TINTA2, 0.455),
        ("   44 frames   2 HELD-OUT windows, annotated after the code was frozen              ->"
         "  panel 4, the 9.1 %", TINTA2, 0.455),
        ("1,815 clips    two-light clips with a resolvable order - A DIFFERENT UNIT ENTIRELY   ->"
         "  panel 6 only", NARANJA, 0.455),
        ("    0 parries  annotated in the 21 windows, against 8 candidates from the t.86 "
         "criterion  ->  no parry accuracy exists, anywhere", CRITICO, 0.455),
        ("WHAT THIS PAGE DOES NOT SAY: that the annotation is the truth. It was made by an agent "
         "looking at contact sheets, not by an FIE referee, and it was proved wrong once "
         "(see fig2).", TINTA3, 0.455),
        ("4 of the 21 windows carry confidence \"medium\" or \"low\". The instant resolution is "
         "+-3 frames because the sheets step by 3 and the frames in between were never seen.",
         TINTA3, 0.455),
    ], W, alto_linea=23, margen=14, fondo=SUP)

    guarda("fig3_honest_scoreboard.png",
           apilar(cab, p1, np.hstack([p2, _pad(p3, p2.shape[0])]),
                  np.hstack([p4, _pad(p5, p4.shape[0])]), p6, pie))


def _pad(a, alto):
    if a.shape[0] >= alto:
        return a[:alto]
    return np.vstack([a, np.full((alto - a.shape[0], a.shape[1], 3), SUP, dtype="uint8")])


def _aviso_en(p6):
    """La caja roja que impide leer el 64,96 % contra el 69,7 %. Va DENTRO del panel 6, en el
    hueco que `mar_der=760` deja reservado a proposito: no se pinta sobre ninguna barra."""
    import cv2
    from comun import texto, FD, F
    x, y, an, al = 1180, 96, 700, 148
    cv2.rectangle(p6, (x, y), (x + an, y + al), (26, 22, 30), -1)
    cv2.rectangle(p6, (x, y), (x + an, y + al), CRITICO, 2)
    texto(p6, "WHY THIS IS NOT ON PANEL 1's AXIS", (x + 14, y + 26), 0.54, CRITICO, 1, FD,
          sombra=False)
    for i, s in enumerate([
            "69.7  %  = who holds right of way in this FRAME, over   462 FRAMES",
            "64.96 %  = who WON the touch in this CLIP,       over 1,815 CLIPS"]):
        texto(p6, s, (x + 14, y + 52 + i * 22), 0.44, TINTA, 1, F, sombra=False)
    for i, s in enumerate([
            "Different question, different unit, different denominator.",
            "Subtracting them is meaningless. Averaging them would invent",
            "a number nobody measured."]):
        texto(p6, s, (x + 14, y + 102 + i * 18), 0.42, TINTA2, 1, F, sombra=False)


if __name__ == "__main__":
    main()

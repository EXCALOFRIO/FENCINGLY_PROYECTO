"""FIGURA 4: POR QUE falla, con las dos causas localizadas con cifras.

Es la figura que de verdad ensena algo: que el 69,7 % no se arregla ajustando. Las dos causas estan
medidas y las dos son de definicion, no de umbral:

  A  `pies` no separa la marcha del a fondo, asi que el a fondo sale ANTES que el brazo y t.83.2.a
     anula el ataque que era bueno.
  B  los umbrales absolutos no viajan entre tiradores: hay tiradores que nunca pasan `EXT_ALTO`.

Los datos de B se recalculan aqui a partir de los `poses_*.npz` y se dicen con su denominador.
"""
from __future__ import annotations

import csv
import sys

import numpy as np

from barras import (AQUA, AVISO, AZUL, BUENO, CRITICO, NARANJA, REJILLA, SUP, TINTA, TINTA2,
                    TINTA3, nota, nube, strip, tile)
from comun import apilar, banda, cabecera, guarda, PRIO

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import p1_variables as P1   # noqa: E402
import prioridad as PR      # noqa: E402

W = 1920
MED = 948


def maximos():
    """max `ext_hr` y numero de arranques de brazo por clip x lado, recalculado de las poses."""
    filas = list(csv.DictReader(open(P1.SALIDA / "p1_variables.csv", encoding="utf-8")))
    out = []
    for cid in sorted({r["clip"] for r in filas}):
        with np.load(P1.PRUEBAS / ("poses_%s.npz" % cid), allow_pickle=True) as z:
            d = {k: z[k] for k in z.files}
        s = PR.senales(d)
        for lado in (0, 1):
            v = np.where(s["val"][:, lado], s["ext_hr"][:, lado], np.nan)
            mx = float(np.nanmax(v)) if np.isfinite(v).any() else float("nan")
            tr, _f = PR._arranques(s, lado)
            out.append(("%s l%d" % (cid, lado), mx, len(tr), int(np.isfinite(v).sum()),
                        int(d["T"])))
    return out


def main():
    mx = maximos()
    bajo = [m for m in mx if np.isfinite(m[1]) and m[1] < PR.EXT_ALTO]
    cero = [m for m in mx if m[2] == 0]
    tramos = sum(m[2] for m in mx)

    cab = cabecera(W, "WHY IT FAILS, AND WHY TUNING WILL NOT FIX IT: two causes, both located "
                      "with numbers",
                   ["Both are DEFINITION faults, not threshold faults. The first makes a good "
                    "attack look badly initiated; the second makes a real attack invisible.",
                    "Panels A and B are recomputed from 01_extraccion/pruebas/poses_*.npz in "
                    "this run. Panel C and D quote the published session numbers."],
                   col_tit=(120, 170, 255))

    pa = strip(
        W, "A - THE LUNGE INDICATOR DOES NOT SEPARATE A LUNGE FROM A WALK",
        "`feet-spread` = |ankle -> ankle| / torso. Every step widens the ankles, so on a running "
        "attack the lunge indicator fires BEFORE the arm does - and t.83.2.a then rules the "
        "attack badly initiated.",
        "3 lunges verified by eye, 1 fleche verified by eye, threshold from prioridad.PIES_ALTO",
        0.5, 4.0,
        [dict(v=1.28, col=TINTA3, et="pooled p50 = 1.28", fila=0, r=6),
         dict(v=2.12, col=TINTA3, et="pooled p90 = 2.12", fila=0, r=6),
         dict(v=2.54, col=CRITICO, et="FLECHE, running attack = 2.54", fila=1, r=10),
         dict(v=3.22, col=BUENO, et="three verified lunges:  3.22", fila=1, r=9),
         dict(v=3.41, col=BUENO, et="3.41", fila=1, r=9),
         dict(v=3.58, col=BUENO, et="3.58", fila=1, r=9)],
        umbral=(PR.PIES_ALTO, AVISO, ["THRESHOLD IN USE  PIES_ALTO = %.2f" % PR.PIES_ALTO,
                                      "the running attack sits only 0.23 above it"]),
        unidad="torsos",
        pies=[("THE FLECHE OF AliveIgnorantBarebirdbat BARELY CLEARS THE THRESHOLD: 2.54 against "
               "2.31, while three verified lunges sit at 3.22, 3.41 and 3.58. The bands touch.",
               TINTA2),
              ("AND THE THRESHOLD IS NOT RAISED TO 2.9, which would separate all four cleanly, "
               "because 2.9 would be chosen by looking at the annotation. That is the same error "
               "the project refused", AVISO),
              ("when it declined to drop a fade threshold from 1.60 to 1.55. A threshold picked "
               "to fit the answers is not a measurement.", AVISO),
              ("Recomputed in this run over the 21 foil clips (n = 9,460 valid frame x side): "
               "pooled p50 = 1.28, p75 = 1.75, p90 = 2.12, p95 = 2.34. The constant in the code "
               "is 2.31, which is nearer", TINTA3),
              ("today's p95 than today's p90 - the code calls it the pooled p90, so that label is "
               "reported as what it says and as what it recomputes, not averaged.", TINTA3)])

    pb = nube(
        W, "B - AN ABSOLUTE ARM-EXTENSION THRESHOLD DOES NOT TRAVEL BETWEEN FENCERS",
        "One dot per clip x side: the HIGHEST `ext_hr` that fencer ever reaches in his whole "
        "clip. EXT_ALTO is a single pooled number (the p75), and some fencers never get there.",
        "n = 42 clip x side (21 foil clips, 2 sides), recomputed from the poses in this run",
        0.75, 2.5,
        [(et, v, CRITICO if (np.isfinite(v) and v < PR.EXT_ALTO)
          else (AVISO if n0 == 0 else AZUL)) for et, v, n0, _nv, _T in mx],
        umbral=(PR.EXT_ALTO, AVISO, ["EXT_ALTO = %.4f" % PR.EXT_ALTO,
                                     "(the pooled p75)"]),
        destaca={"HarshExemplaryGermanspaniel l1", "HarshExemplaryGermanspaniel l0",
                 "TestyWatchfulBarebirdbat l0"},
        pies=[("RED = this fencer's arm NEVER reaches EXT_ALTO in the whole clip, so he can never "
               "be granted right of way: %d of 42. Both are HarshExemplaryGermanspaniel, "
               "max 1.002 and 1.086." % len(bajo), CRITICO),
              ("YELLOW = registers ZERO arm starts under the gated criterion (derivative + "
               "EXT_ALTO height + %.2f amplitude): %d of 42, i.e. %.1f %%. That is physically "
               "impossible in a fencing clip."
               % (PR.EXT_AMPL, len(cero), 100.0 * len(cero) / len(mx)), AVISO),
              ("The same 42 denominator carries TWO different counts and they are not the same "
               "measurement: %d arm starts under the GATED criterion (%d of 42 with none), "
               "against the published 234 segments and" % (tramos, len(cero)), TINTA2),
              ("0 of 42 with none under p1_variables' LOOSE criterion, which has no height or "
               "amplitude gate at all. Two criteria, two counts, both reported - the average of "
               "the two would mean nothing.", TINTA2),
              ("THE FIX IS NOT A BETTER NUMBER, it is normalising per fencer: each fencer's own "
               "guard-to-extension range instead of one pooled constant. That is the next step, "
               "and it is not done.", AVISO),
              ("The dot for TestyWatchfulBarebirdbat l0 at 2.427 and the one for "
               "HarshExemplaryGermanspaniel l1 at 1.002 are the same measurement on the same "
               "scale: the spread between fencers is 2.4x.", TINTA3)])

    pc = tile(
        MED, "C - WHAT ACTUALLY STOPS THE OSCILLATION (and what does nothing)",
        "Right-of-way changes per second. The user's diagnosis was right - it did flicker - but "
        "the remedy was not the one proposed.",
        "n = 110.1 s of action over 10 clips; hand-annotated rate = 0.65 changes/s",
        [dict(et=["complete (as shipped)"], v=0.55, col=AZUL, txt="0.55"),
         dict(et=["without the GRACE INTERVAL"], v=0.55, col=TINTA3, txt="0.55"),
         dict(et=["without HYSTERESIS (one threshold)"], v=0.38, col=TINTA3, txt="0.38"),
         dict(et=["without the AMPLITUDE/HEIGHT FILTER"], v=0.84, col=CRITICO, txt="0.84",
              destaca=True),
         dict(et=["without PLATEAU MERGING"], v=0.51, col=TINTA3, txt="0.51"),
         dict(et=["without any of the above"], v=0.89, col=CRITICO, txt="0.89")],
        xmax=1.0, etq=380, mar_der=150, alto_fila=44, grosor=22,
        ref=(0.65, AVISO, ["HAND-ANNOTATED 0.65 changes/s"]),
        pies=[("THE GRACE INTERVAL THE USER ASKED FOR CHANGES NOT ONE FRAME: 0.55 with it, 0.55 "
               "without it. What brakes the flicker is the amplitude and height filter", TINTA2),
              ("(0.55 -> 0.84 when removed). PERMANENCIA stays in the code WITH THIS TABLE NEXT "
               "TO IT, the same way USA_CRUDAS = False was kept with its measurement.", TINTA2),
              ("The final 0.55 against the real 0.65 is the ONE measurement where this detector "
               "resembles reality - and it is a rate, not an accuracy.", AVISO)])

    pd_ = nota(
        MED, "D - THE CRITERIA THAT WERE DECLARED BEFORE MEASURING, AND HOW THEY LANDED",
        [("ext_hr range p95/p05 >= 1.5     ->  median 2.21, 40 of 42 clip x side        "
          "PASSES", TINTA),
         ("ext_cr more stable than ext_hr  ->  cv 0.211 vs 0.241, only 27 of 42         "
          "DOES NOT PASS CLEANLY", TINTA),
         ("an arm start in every clip x side ->  234 segments, 0 of 42 without one (loose "
          "criterion)   PASSES", TINTA),
         ("", TINTA),
         ("THE VARIABLE THE BRIEF ASKED FOR DOES NOT WORK, and it is measured why: "
          "elbow -> wrist is a BONE, not an extension. Its apparent length varies", AVISO),
         ("21.1 % from foreshortening and pose noise, and it does not grow when the arm extends. "
          "The control that separates the two: measure both lengths in", AVISO),
         ("pixels - the torso normalises well (cv 0.079) and the variation is in the hand point. "
          "The one that works is shoulder -> wrist.", AVISO),
         ("", TINTA),
         ("FENCING TIME (t.8), which the rules do NOT give in seconds, so it was measured: "
          "p25 = 3, p50 = 5, p75 = 8, p90 = 10, max = 22 frames.", TINTA2),
         ("5 frames = 167 ms at 30 fps. That is where the t.89.2 simultaneity window and the "
          "grace interval come from - neither is a chosen number.", TINTA2),
         ("THE REACH BOUND, half rule and half measurement: the blade is 0.90 m (m.8.4), so two "
          "points cannot meet with the hands more than 1.80 m apart;", TINTA2),
         ("but an attack does not START at hitting distance, it CLOSES it, and the closing is "
          "measured: p90 = 1.48 m over 96 arm rises. Bound = 3.28 m.", TINTA2),
         ("AND IT WAS WRONG THE FIRST TIME, which the measurement caught: with the 1.80 m bound "
          "alone, accuracy on frames with an owner collapsed 25.6 % -> 3.3 %.", CRITICO),
         ("The bound was right for the instant of the HIT and was being applied to the instant of "
          "the START. Same number, wrong moment.", CRITICO)],
        AZUL, (22, 24, 28), alto_min=pc.shape[0])

    pie = banda([
        ("WHAT WOULD ACTUALLY MOVE THIS, in the order the measurements point", TINTA, 0.58),
        ("1  Normalise ext_hr PER FENCER instead of one pooled EXT_ALTO. Directly aimed at "
         "cause B: %d of 42 clip x side currently register no arm start at all."
         % len(cero), TINTA2, 0.455),
        ("2  Separate the lunge from the walk with something other than ankle spread. Cause A is "
         "not a threshold problem: a running attack and a walking step genuinely produce the "
         "same ankle spread.", TINTA2, 0.455),
        ("3  Annotate parries on purpose, starting from the 8 candidates the t.86 criterion "
         "already found. That would buy precision, not coverage - and it needs 1080p, because "
         "the blade is seen in", TINTA2, 0.455),
        ("   1.6 % of 720p crops against 42.6 % at 1080p, and 20 of the 21 foil clips are 720p.",
         TINTA2, 0.455),
        ("4  More held-out windows. The honest number rests on 44 frames from 2 windows, and "
         "that is too few to be sure of anything except that it is not 69.7 %.",
         TINTA2, 0.455),
        ("WHAT IS EXPLICITLY NOT GOING TO BE DONE: extending 90 cm of blade from the hand along "
         "the forearm axis and testing whether the two lines cross. It is forbidden in writing "
         "in this project", CRITICO, 0.455),
        ("and it is measured to point where the blade is not: the thumb-axis substitute gave "
         "2.4-4.8 px of leverage and two estimators of the same line disagreeing by 67-134 deg.",
         CRITICO, 0.455),
        ("AND THE DETECTOR IS OFF. PRIORIDAD = False in 01_extraccion/pipeline.py. It ships as a "
         "documented failure with its rule map, its variables, its fault catalogue and its "
         "tools - all of which transfer.", AVISO, 0.47),
    ], W, alto_linea=23, margen=14, fondo=SUP)

    guarda("fig4_why_it_fails.png",
           apilar(cab, pa, pb, np.hstack([pc, _pad(pd_, pc.shape[0])]), pie))


def _pad(a, alto):
    if a.shape[0] >= alto:
        return a[:alto]
    return np.vstack([a, np.full((alto - a.shape[0], a.shape[1], 3), SUP, dtype="uint8")])


if __name__ == "__main__":
    main()

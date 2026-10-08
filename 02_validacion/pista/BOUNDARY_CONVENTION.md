# BOUNDARY convention, DECLARED BEFORE ANNOTATING

Date: 2026-09-11. Nothing had been annotated when this file was written.

The user offered: *"maybe you'll ask me to tell you where a march, or a retreat, or a lunge starts and
ends, so you learn it better"*. **Yes, it is needed**, and there is a concrete gap: what has been
measured so far is **detection** (90 % recall) and **classification** (5 of 6), but **nobody has
measured whether the BOUNDARIES are right**. An event can exist, be the right type, and start 8 frames
late.

## The question to fix first: when does a march START?

There are two defensible answers and they **differ by 2-4 frames**:

  **(A) when the foot TAKES OFF from the floor** — the instant contact is lost.
  **(B) when the foot STARTS MOVING** — which is earlier: the heel lifts and the foot pivots on the
  toe without having left the floor yet.

**(A) IS ADOPTED: the boundary is TAKE-OFF, and the end is LANDING.** Three reasons:

1. It is a **discrete, observable instant**: you can see whether the sole touches or not. (B) requires
   judging when a movement "begins", which is a threshold and brings back the very problem this change
   of primitive removes.
2. It is what the new detector measures (`posicion.contactos`), so the annotation and the detection
   measure **the same thing**. If (B) were annotated and (A) detected, the systematic 2-4 frame
   difference would read as detector error, and it is not.
3. In biomechanics it is the standard convention (*toe-off* / *foot-strike*).

**AND IT IS NOT FREE, which has to be said:** with (A) the march "starts" slightly after the fencer
has initiated the gesture. If what you want is the instant of INTENTION, (A) is late by construction.
(A) is adopted because it is measurable, not because it is the only truth.

## What exactly gets annotated

Per action in the strip, three numbers and a type:

| field | what it is |
|---|---|
| `ANOTA_cuadro_inicio` | the first frame in which **the foot no longer touches** the floor |
| `ANOTA_cuadro_fin` | the first frame in which **it touches again** |
| `ANOTA_tipo` | `march`, `retreat`, `lunge` or `halt` |
| `ANOTA_pie` | `front` or `rear`, whichever moves (if both move, the one that starts) |

Tie-break rules, so the annotation is reproducible:

- **If a frame is doubtful, write the LOWER one** (the earlier). That way the annotation's bias has a
  known sign instead of being random.
- **If the foot cannot be seen** (occluded, out of frame, motion blur), write `?` in that field and
  that action does not enter the boundary-error figure. **A `?` is worth more than an invented
  number.**
- **`lunge`**: annotate the take-off and landing of the FRONT foot. The rear one does not take off,
  and if it does it is not a lunge.
- **`halt`**: both feet in support. `inicio` is the landing of the last foot that moved and `fin` the
  take-off of the first one to move afterwards.

## What will be computed from this

  **start boundary error** = detected `cuadro_inicio` − annotated, in frames and in ms
  **end boundary error** = same
  **bias** (the median, which says whether the detector is systematically late or early) and
  **spread** (p10-p90, which says how much a single event can be trusted)

If the bias is systematic **it is corrected by subtracting it**, and the before and after are
published. If it is zero, that is stated — as already happened with the detection offset
(+0.5 frames).

## Where the strips and the file are

  numbered strips: `02_validacion/pista/anotar/accion_<clip>_l<side>_c<frame>.png`
  file to fill in: `02_validacion/pista/anotar/fronteras.csv` (columns already there, one row per
  action, only the numbers need writing)

Each strip carries **every frame of the action, numbered in large type**, with 4 frames of margin on
each side — without margin you cannot judge that a frame is the take-off, because you need to see the
contact before it — the two feet in different colours (cyan / magenta), and the frame **STABILISED**
so the floor does not move.

**The detector's candidate is written on the strip but on its own line, with "do not trust it, correct
it"**, and in the CSV it sits in **separate columns** from the ones to fill in. Putting it in the same
cell would bias the annotation towards what was already detected, which is the easiest way to
manufacture false agreement.

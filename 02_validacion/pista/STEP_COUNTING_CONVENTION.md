# Step counting convention, DECLARED BEFORE COUNTING

Date: 2026-09-11. Nothing had been counted when this file was written.

It exists because the user said *"there must be several per clip, maybe about 10 marches per fencer"*
and the detector reported 5.7, and **that disagreement cannot be settled without fixing what a step
is**. This project has already produced five contradictory figures in one day because of an
undeclared convention.

## The unit that gets counted: the STRIDE, not the "march"

**A STRIDE is one support that translates**: a foot loses contact with the piste (or slides it) and
lands again **somewhere else along the piste**, moving at least the length of its own foot (about
25 cm, which by eye is unmistakable from a weight shift).

The stride is counted rather than the "march" for a reason of method: **a stride is observable without
ambiguity** (you see the foot take off and land) and a "march" is not, because it depends on how
strides are grouped, which is exactly what is in dispute. Counting the unambiguous unit first and
grouping afterwards gives both figures without counting twice.

### What does NOT count as a stride

- **A weight shift with no translation.** The foot pivots, the heel lifts, load goes on and off, but
  the sole returns to the same place. It is the most frequent thing in foil and it is not a step.
- **The foot that drags inside the same support** during a lunge: in a well-executed lunge the rear
  foot does NOT translate, it stays planted. A lunge contributes no rear-foot stride.
- **Lateral or cross-piste adjustment.** Only translation ALONG the piste counts, which is the axis
  this project measures (the lateral one is not credible, see the README).
- **Anything outside the usable stretch** (the clip's `mask`) or with the fencer out of frame.

## Grouping: from strides to marches

**A MARCH (or a RETREAT) is a pair of consecutive strides in the same direction, one per foot**: in a
march the front foot advances and the rear one follows; in a retreat the rear one goes first. That is
the fencing definition and it is the one that gives the figure the user expects.

Tie-break rules, declared so the count is reproducible:

1. Strides are walked in frame order. Two consecutive ones **in the same direction and from different
   feet** are paired.
2. If a stride is left unpaired (because the next is from the same foot, or the opposite direction, or
   there are no more), it counts as a **HALF STEP** and is recorded separately. It is neither
   discarded nor rounded.
3. A continuous advance of N alternating strides in the same direction is **N/2 marches**, plus a half
   step if N is odd. So "advancing continuously" is neither one march nor N: it is N/2.
4. The **lunge** contributes the front foot's stride. If that stride ends up paired with the rear
   foot's during the recovery, it counts as a step as well as a lunge; if not, it is a half step.
   **The lunge is counted in its own column**, because the detector has it as its own type.

## How it is counted, and why looking at my own series would not do

Counting is done **on the image**, in mosaics of the leg band sampled every 2 frames, **with the
frames STABILISED** by each frame's `Mk` similarity. Stabilisation freezes the floor, so a foot that
translates is seen against a fixed background and there is no need to guess how much of the motion is
camera pan (median 161 px in this corpus).

**Using the metric position series as ground truth would be circular**, because it is the same signal
the detector fires on. Stabilisation is NOT circular: it is a 2D geometric fit on background points,
independent of the pose, of the homography and of the step detector.

The foot markers are drawn in the mosaic **in two different colours, one per leg**, so each can be
followed, but **the count is made from the position of the sole against the floor, not from the
marker**: if a marker is misplaced, the frame is recorded as doubtful.

## What is published from the count

Per (clip, fencer) counted, in `conteo_ojo_pasos.csv`:

`clip, lado, cuadro_inicio, cuadro_fin, n_zancadas, n_pasos_convenio, n_marchas_convenio,
n_romper_convenio, n_medios_pasos, n_fondos_det, n_detectados, n_zancadas_cubiertas,
recall_zancadas`

And the figure that settles the disagreement:

**stride recall** = strides detected by the detector that fall on a stride counted by eye, divided by
the strides counted by eye. With its declared tolerance: a detection counts as a hit if its
`[start, end]` range **overlaps** the counted stride's range, or falls within **<= 4 frames** of it
(0.13 s, less than a stride lasts).

## What this decides

- if by eye there are about **10 marches per fencer** and the detector gives 5.7, the detector
  **misses half** of them and must be fixed;
- if by eye there are about **6**, the detector is fine and the expectation was high;
- and if by eye there are 10 strides but 5 marches, then **both are right** and what existed was an
  undeclared convention — which is the case this file exists to prevent.

## Result

**Both were right.** Counting steps (`march + retreat`, because a step backwards is also a step) gives
**9.1 per fencer per clip**, which matches the ~10 expected; the "5.7" counted only `march` and
excluded `retreat`. And the measured stride recall is **90 %** (44 of 49 strides in 3 fencers), so the
detector was not missing half of them.

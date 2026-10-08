# Lights, the two-light subset, counterattacks and the fake off-target light: numbers and design

*Written in English because this is headed for a public repository. Column names in the CSVs stay in
Spanish: they are what the code reads.*

This document answers the three things the user asked for, with a number next to every claim, and it
says explicitly what cannot be claimed. The author is an agent; **no annotation here is a refereeing
annotation.**

Everything comes from the NPZ files of the previous extraction (10,840 clips with a venue) and from
the `index.jsonl` files (25,139 clips attempted, 23,011 with a label). No video decoding except where
stated.

**Terminology.** In foil the **scoring apparatus** shows a **red light** and a **green light** for a
**valid touch**, one colour per fencer, and a **white light (off-target)** when the point lands
outside the **valid target** — arm, leg, mask or the piste. **Right of way** is the convention that
decides who gets the point when both lights come on. A **counterattack** is an attack made into the
opponent's attack without first deflecting it; the counterattacker does not have right of way.

---

## 0. Before the three things: the warning that conditions all of them

The one-light / two-light split **is produced by the light detector**, and that detector is not
equally healthy across venues. Measured: the share of two-light clips ranges from **5.93 %** in
`tokyofoil2022` to **25.05 %** in `tokyoolympics2020`, a factor of 4.2, which is the same order as
the variation in its failure rate (2.25 % to 36.78 %). **Part of that variation is fencing and part
of it is the detector, and how much is which has not been measured.**

What *was* measured today, and it changes the repair plan:

| | measured | where |
|---|---|---|
| clips the detector loses (`status = no_lights`) | **2,038 of 25,139 = 8.11 %** | `no_lights_por_sede.csv` |
| of the clips lost in `tokyofoil2022`, how many really contain a light | **10.0 %** (2 of 20) | `roi_aprendida.csv` |
| same, `parisfoil2022` | 20.0 % | idem |
| same, `stpetersburgfoil2019` | 30.0 % | idem |
| same, the FIE family | **40.0 %** (8 of 20) | idem |
| control: the same reading on `ok` clips of the same venues | **80-90 % fires** | idem |

In other words: **most of the 2,038 "lost" clips have no light to find.** Extrapolating with those
rates, the recoverable ones are of the order of **575 out of 2,038 (28 %)**, and the largest group is
not `tokyofoil2022` (~53 clips) but **the FIE family (~476)**. Repairing Tokyo first was the wrong
priority, and the figure that decides it is that 10.0 %.

**The repair that does work, measured:** a **per-venue learned ROI** — the median bounding box of the
light over 12 of that venue's own clips — read in the dumbest possible way (temporal maximum of the
mean channel inside the ROI, minus its 20th percentile) fires on **90.5 % of sample B** (19 of 21),
against **71.4 %** for the general detector that searches the whole bottom band. It fails only on
`tokyofoil2022` and `tokyoolympics2020`, and for the first one the cause is known: its red ROI was
learned from 12 clips at `fy` 0.9022 while its **green** ROI was learned from 6 clips at `fy` 0.8414,
0.06 H higher up. Two channels of the same overlay cannot sit 0.06 H apart, so **the green ROI of
`tokyofoil2022` is badly learned**. That is a one-venue fix with a measured cause, not a redesign.

---

## 1. Keeping only the two-light clips

**The user's reasoning is correct and now has a number.** With a single light the label is trivial:
predicting "the fencer whose light is on wins" is right **96.19 % on 8,049 one-light clips**. There is
no convention to apply, so those clips teach nothing about right of way. With two lights a decision is
required, and there the light-order shortcut only reaches **64.96 %**.

### The two-light subset, per venue

Of the 10,840 clips with an NPZ and a venue: **2,292 two-light (21.14 %)** and **8,548 one-light
(78.86 %)**.

| venue | clips | one light | two lights | % two |
|---|---:|---:|---:|---:|
| tokyofoil2022 | 445 | 426 | **19** | **4.27** |
| tbilisifoil2023 | 81 | 74 | 7 | 8.64 |
| bonnfoil2019 | 435 | 372 | 63 | 14.48 |
| stpetersburgfoil2019 | 421 | 356 | 65 | 15.44 |
| parisfoil2022 | 173 | 144 | 29 | 16.76 |
| turinfoil2020 | 321 | 264 | 57 | 17.76 |
| anaheimfoil2018 | 1,141 | 929 | 212 | 18.58 |
| serbiafoil2022 | 803 | 653 | 150 | 18.68 |
| turinfoil2023 | 359 | 289 | 70 | 19.50 |
| belgradefoil2022 | 507 | 406 | 101 | 19.92 |
| eurochamps2022 | 487 | 390 | 97 | 19.92 |
| qatarfoil2021 | 1,137 | 897 | 240 | 21.11 |
| anaheimfoil2019 | 510 | 401 | 109 | 21.37 |
| eurosfoil2023 | 310 | 234 | 76 | 24.52 |
| incheonfoil2022 | 652 | 484 | 168 | 25.77 |
| katowicefoil2019 | 375 | 275 | 100 | 26.67 |
| kazanfoil2020 | 467 | 342 | 125 | 26.77 |
| worldchamps2022 | 1,131 | 828 | 303 | 26.79 |
| bonnfoil2022 | 549 | 401 | 148 | 26.96 |
| tokyoolympics2020 | 370 | 269 | 101 | 27.30 |
| tauberbischofsheimfoil2022 | 166 | 114 | **52** | **31.33** |

**4.27 % in `tokyofoil2022` against 31.33 % in `tauberbischofsheimfoil2022`** is a factor of 7.3
between two tournaments of the same era. That is not fencing: that is the detector.

### And one thing the two-light subset does not give for free

Of the 2,162 two-light clips with both fencers tracked, **only 84.0 % (1,815) have a resolvable
order**: in the remaining 16.0 % `first_light` is `same_frame`. And of those that do, **38.4 %** have
`|dt| <= 40 ms`, which is **below one frame at 25 fps**. So "the order of the lights" is not an
available variable in 1 of every 6 two-light clips, and is fragile in another third.

### The shortcut baseline, reproduced, with a surprise of sign

On those 1,815 clips against the real label (who got the point, derived from the score change):

| rule | accuracy |
|---|---:|
| the fencer who lights **FIRST** wins | 35.04 % |
| the fencer who lights **SECOND** wins | **64.96 %** |
| always predict the same side (majority) | 52.18 % |

The **64.96 %** reproduces the 63.51 % the project had already measured. What is worth stating out
loud, because it was not written down: **the shortcut is that the fencer who lights SECOND wins**, not
the first. And it makes fencing sense: when both lights come on, the typical case is an attack that
lands and a counterattack or remise that also lands; the counterattacker usually **hits first**
because he extends earlier, and **loses on right of way**.

---

## 2. One-light clips that are not counterattacks, used as if they had two lights

**The idea is correct in its reasoning**: if only one light comes on and it was not a counterattack,
then the fencer who touched had right of way, and the right-of-way label is known just as well as
with two lights. The corpus would go from **2,162 to 8,448 clips, a factor of 3.91**.

**The problem is that separating counterattacks cannot be done today**, and this is not an opinion:

- the project's right-of-way detector **does not work** (69.7 % against 73.8 % for never answering),
  so it cannot be used to decide whether a counterattack happened;
- and the **cheap proxy I tested and measured DOES NOT WORK.**

### The proxy that was tested, and its two measurements against it

Definition, declared before measuring: over the 0.5 s before the touch, the closing velocity of each
fencer in piste coordinates (`x_piste = x_image + pan`, from the NPZ's `boxes` and `pan`, no video
decoding). **`counterattack_suspicion` = the fencer in question was NOT closing and the opponent
was.** In a one-light clip the fencer in question is the one whose light is on; in a two-light clip it
is the one who lit **first**.

| measurement | result | criterion | |
|---|---:|---:|---|
| separation between clips where the shortcut is right and where it is wrong | **3.65 points** (25.79 % vs 22.14 %) | >= 10 | **FAILS** |
| combined rule (if the first to light was closing, he wins; otherwise the shortcut) | **59.61 %** against 64.96 % for the shortcut alone | improve | **−5.34 points: WORSE** |
| suspicion rate, one light vs two lights | 21.90 % vs 17.74 %, margin **4.16 points** | >= 10 | **FAILS** |

And one measurement points the wrong way entirely: where the shortcut **fails**, counterattack
suspicion appears in **12.89 %** of clips, and where the shortcut **succeeds**, in **20.36 %**. The
proxy flags *fewer* counterattacks exactly where it should flag more.

**Conclusion: the ×3.91 cannot be built today.** What can be left in place is the mechanism: the
column is computed and stored for 10,211 clips in
`02_validacion/muestras/luces/una_luz_y_contraataque.csv`, with `v_duda`, `v_otro`, `sospecha_ca` and
`avanzaba`, so that when right of way works the criterion can be swapped without recomputing
anything.

**Why I believe the proxy fails, so that it is not retried the same way.** The velocity of the
bounding-box centre over 0.5 s measures *body displacement*, and a counterattack is an *arm
extension* that can happen with the body still or even retreating. The signal needed is not where the
body is but **when the weapon arm extends**, which is exactly what the cone and armed-hand stages are
trying to measure. In other words: **this path has no cheap geometric shortcut, and the road is the
one already in the plan.**

---

## 3. The fake white light (off-target)

### First, how much "almost never" actually is

| | measured |
|---|---|
| clips with any **white light (off-target)** | **622 of 10,840 = 5.74 %** |
| over the venues where the white channel is looked at (the 16 FIE-family ones) | 622 of 8,964 = **6.94 %** |
| clips whose **only** light is white | **0 of 10,840** |
| white left / white right | 305 / 317 |
| venues with **0.00 %** white | 5: `kazanfoil2020`, `parisfoil2022`, `stpetersburgfoil2019`, `tokyofoil2022`, `tokyoolympics2020` (1,876 clips) |
| range in the rest | 4.82 % (`tauberbischofsheimfoil2022`) to 16.05 % (`tbilisifoil2023`) |

The two figures that matter are these:

1. **There are 0 clips whose only light is white.** The corpus is cut around touches that scored a
   point, so a purely off-target touch — which does not score — never enters. The model has **never
   seen** a case whose correct answer is "white, no touch awarded". The user is right about the
   diagnosis.
2. **The 0.00 % in five venues does not mean there is no white there**: it means the generic detector
   **has no white channel**. That is a zero of the detector, not of the fencing. So the global 5.74 %
   is a lower bound.

### The warning, which goes attached to the proposal

> **Swapping a colour light for white does not only change the label: it changes the PHYSICS of the
> action.** In foil, white means the point landed **outside the valid target** — arm, leg, mask or
> piste. If you take a valid touch to the torso and relabel it white, you are asserting that that same
> blade trajectory, the one that ended on the torso, ended on an arm. The video still shows a touch to
> the torso. **The model would learn to associate valid-touch geometry with the invalid label**, which
> is the opposite of the goal. And because the corpus contains 0 clips with white alone, there would
> be no real case against which to check whether the fake helps or poisons.

### What can be done, in order of coherence

1. **Use the 622 clips that DO have white**, which is cheap and invents nothing. There, white appears
   **alongside** a colour light, which is the regulatory case of a double hit with one part
   off-target: the white light does not annul the other fencer's point. With 622 clips and a 305/317
   split by side, one can measure whether the model treats a clip with white differently. **The figure
   needed before deciding: model accuracy on the 622 clips with white against the 10,218 without.**
   Not measured.
2. **Look for the real white-only cases**, which exist in the source videos but not in the corpus,
   because the clip cutter anchors on touches that scored. Recovering them is a change to the
   **cutter**, not to the light detector, and it is the only path that yields real data. How many
   there are **is not measured**, and it is the question I would ask first.
3. **If the fake is still wanted, it has to be marked in the data.** Minimum design:

| new column | type | meaning |
|---|---|---|
| `luz_sintetica` | uint8 (T, 4) | 1 on the frames and channels whose light was **not observed** and has been injected |
| `receta_sintetica` | str in `meta_json` | which transformation was applied, e.g. `color_a_blanco_v1` |
| `clip_origen` | str in `meta_json` | the `clip_id` it was derived from, so copies can be grouped when splitting train/val |

And three rules that are not optional:

- **Never mix synthetic and observed in the same split** unless the synthetic clip inherits the split
  of its `clip_origen`. If the original is in train and the copy in val, the model passes by copying.
- **Train with and without, and publish both figures.** An augmentation that is not compared against
  its own absence cannot be judged. And compare **on the 622 clips with real white**, which is the
  only place where the question has an answer.
- **The geometry has to be consistent with the label.** A defensible fake would inject white into
  clips where the touch **actually** landed off-target, or where the point ended on the arm according
  to the blade measurement. Injecting it into a torso touch is not defensible.

This is exactly the discipline the project already applies with `origen` and `hueco` on the poses: a
fabricated datum that cannot be told apart from a measured one is the failure mode this project has
been avoiding all day.

---

## 4. What this document does NOT say

- **It does not say the two-light subset is 2,292 clips.** It says that is what the current detector
  counts, and that its two-light rate varies by a factor of 7.3 between venues of the same era. First
  the detector, then the subset.
- **It does not say counterattacks cannot be separated.** It says **the closing-velocity proxy does
  not separate them** (3.65 points, and it makes the shortcut 5.34 points worse), and that the
  right-of-way detector it would need does not work. Another signal might work; this one does not.
- **The 96.19 % for "the fencer whose light is on wins" on one-light clips is not a ceiling on
  anything**: it is confirmation that those clips teach nothing about right of way. The remaining
  3.81 % would be worth looking at, because they are either annulled touches or label noise, and
  which is unknown.
- **The 5.74 % white figure is a lower bound** and is measured on what the old detector declares, not
  on what is in the video. In five venues the white channel is simply not looked at.
- **The ~575 recoverable clips out of 2,038 is an estimate** made from 20 clips per group. The
  interval is wide, and the value of the figure is negative: it says **not** to spend work on Tokyo
  believing there are 527 clips there.
- **Nothing here has been tested by training a model.** Every figure is a count, or arithmetic over
  labels that already existed.

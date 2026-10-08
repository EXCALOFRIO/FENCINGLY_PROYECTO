# Which hand holds the weapon?

![summary](resumen_mano_armada.png)

In foil, almost everything downstream depends on knowing **which of a fencer's two hands holds the
weapon**: the weapon hand is where the blade starts, so it is the origin of any reach, threat or
priority estimate. This folder holds the evidence for how that question is answered, and for how the
previous answer was wrong.

## The numbers, with accuracy and coverage kept apart

| | held-out (19) | all with truth (59) | coverage |
|---|---:|---:|---|
| **Old rule** — *the hand whose median x is closest to the opponent* | 0.947 | **0.831** | full |
| **Front foot**, ungated | 0.895 | 0.881 | full |
| **Front foot with a firm vote** — the adopted rung | **12 of 12, no failure** | 29 of 29, no failure | **12 of 19** |
| **The whole cascade** — front foot, then the on-guard-and-in-action gate | **1.000** | 0.949 | full |

**The 1.000 over 19 of 19 belongs to the CASCADE, not to the front-foot rule**, and an earlier
version of this page — and of eight other places, including text burned into the summary PNG — said
"New rule — *the front foot decides* — 1.000 on 19 of 19" and so credited the rule with the
cascade's accuracy *and* the cascade's coverage at once. Both figures are real; the sentence that
joined them was not. The front-foot rule on its own answers for 12 of the 19 and has not failed one
of them.

Note also that the held-out set is **easier** than the training set: the old rule scores 0.947 there
against 0.763 in training. That is why the 0.831 over all 59 is the figure to quote for it.

The old rule is not a strawman: it was the operational criterion of this project, and its previous
evidence was "correct on the 4 fencers checked by eye". Four is not a sample.

## The rule that was adopted

**In the on-guard stance the front foot is on the same side of the body as the weapon arm.** A
right-hander leads with the right foot and holds the foil in the right hand. This is the basic stance
of the sport, not a correlation found in this data.

What makes it valuable is that it is a **correspondence inside the body**, so it does not depend on
where the opponent is — and that is exactly what breaks the old rule. There are fencers who hold the
weapon in the hand on the **far** side from the opponent for a whole clip (between actions, standing
at their line, walking back), and there the old rule is wrong in *100 %* of the frames while being
maximally confident.

The front foot is identified without ever looking at the opponent: each foot has an axis
`heel -> toes`, and on guard the front foot points along the piste while the rear foot sits almost
perpendicular. The foot whose axis is more aligned with the piste is the front one.

Two more pieces:

- **One decision per fencer per clip.** A fencer does not change hands mid-bout, so the per-frame
  votes are pooled and the **vote fraction** is published. It is the vote fraction that makes the
  criterion usable.
- **The system is allowed to say "I don't know".** When the vote fraction is below 0.85 the fencer is
  flagged **`uncertain`** and the rest of the pipeline should exclude it rather than inherit a wrong
  weapon hand. On the held-out set the two cases the foot criterion got wrong were **both** flagged.

## What did not work, with its numbers

Kept here because the reasons are more reusable than the successes.

| idea | result |
|---|---|
| **the white glove** — the weapon hand wears a glove, the other is bare | **0.447**. No separation: the paired difference is **−0.002** with the sign right in only **0.455** of frames. The feature was reading the **white jacket sleeve** (pearson **0.681** against a sleeve patch), and some gloves are dark. |
| **the dark bell guard** | **0.500**. With the project's inverted brightness threshold, half of every patch comes out "dark": the guard, the lamé in shadow and the piste all land in the same bucket. |
| **the mask as evidence of the other hand** (FIE t.115.1 puts the mask in the non-weapon hand) | the detector fires on **35 of 44** fencers while only **1** actually holds a mask. Precision ~1/35. |
| **the most extended arm** | **0.525**. On guard the *rear* arm is extended too, backwards and up. |
| **gating frames by "in reach", "arm extended" or "in profile"** | none reaches the adoption bar, and two go the **wrong way**: frames with a more extended arm are **0.047** *worse*. They all measure a magnitude; the problem needed a **sign**. |

## The files

| pattern | what it is |
|---|---|
| `resumen_mano_armada.png` | the summary figure above |
| `anota_<clip>_l<side>.png` | annotation mosaics: context frame plus zooms with **both** hand roots marked `A` and `B` |
| `ciego_NN.png` | the same, **blind**: no clip name, no side, shuffled order, key kept apart in `clave_ciego.csv` |
| `elegida_<clip>_l<side>.png` | the adopted decision drawn on the frame, with the glove features printed next to each hand |
| `discrepa_*.png` | cases where two independent criteria disagree — the ones worth opening |
| `falla_a_*.png` | cases the old rule gets wrong |
| `lupa_seguro_y_equivocado.png` | the two "confident and wrong" cases, zoomed |
| `verdad_mano.csv` | the ground truth, one row per fencer, with the evidence and the certainty for each |
| `clave_ciego.csv` | the blind-batch key, mapping `NN` to clip and side |

`A` and `B` are **keypoint indices, not left and right**: `A` is hand root 21 and `B` is hand root 42
of the 308-point pose. No left/right assumption is made anywhere, and the pose model does not assign
them consistently between mirrored fencers.

CSV column names are in Spanish on purpose: other code reads them, and renaming them would break it.
The columns that matter: `mano` is the annotated weapon hand, `cue` is the evidence used
(`hoja` = blade visible, `arma` = weapon or guard visible, `accion` = leading hand only,
`careta` = mask in the other hand, `arbitro` = not a fencer), `certeza` is `alta`/`media`, and
`mano_p1` is what the first blind pass said before the second one.

## How the ground truth was built, and its limits

**It was annotated by an agent looking at image crops, not by a referee.** Every accuracy number here
inherits that.

- **Blind mosaics.** They carry only the letters `A` and `B`, never any criterion's verdict, so the
  annotation cannot be contaminated by what is being measured. In the last batches the clip name and
  the side were removed from the image too, and the order was shuffled with a fixed seed.
- **Two passes.** Every fencer left at medium certainty was looked at again with twice as many frames.
  No letter changed between passes in the last two batches; several went from medium to high.
- **`?` when undecidable, even against the algorithm.** One fencer is left undecided although all four
  criteria agree on an answer, because the two blind passes disagreed with each other and no weapon is
  visible in any frame. Correcting the ground truth towards the algorithm in the one case where they
  all agree against you is how you manufacture accuracy.
- **59 fencers over 33 clips**, of which **19 are the held-out set** chosen with a different random
  seed and never used to pick any rule. The two samples are reported **separately and never pooled**:
  the training clips were collected *because* they were giving trouble, so they are enriched in hard
  cases, while the held-out set is a random draw. The first is a lower bound, the second is the better
  estimate of typical performance.

Honest wording matters here: **"has not failed any of 19"** is not "always right". With 19
observations the lower confidence bound is around 0.83.

## One defect this work found and did not fix

**2 of the 33 annotated clips (6.1 %) deliver the referee as one of the two fencers.** One of them
turned up in the random held-out set, so it is a recurring failure mode rather than an anecdote.

And it will not be caught by any pose-quality metric: **the referee's pose is perfectly good** — a
person standing, whole, well lit, well tracked. There is nothing wrong with the pose; what is wrong is
*whose* it is. The ten held-out clips all came out with both poses present in 100 % of usable frames,
and one of them is the referee.

A behavioural detector was tried on the idea that *the referee does not fence* — no lunges, no arm
extensions. It **failed**: one referee scores **0.524** fencing events per second, above the 10th
percentile of real fencers (0.493) and four times his own opponent's rate. The event that fires for
him is arm extension, and what triggers it is the **arm swing of walking**.

---

Fencing rule cited: **FIE, Technical Rules, December 2025, t.115.1** — *"glove and weapon held in the
weapon hand [...] the mask held in the non-weapon hand"*. Note that the rules do **not** forbid a
glove on the non-weapon hand; that the other hand is bare is common practice, not a citable rule, and
the measurements above contradict it as a reliable signal.

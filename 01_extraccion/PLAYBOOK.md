# FENCINGLY - extraction playbook

How to take a corpus of fencing clips and turn each one into a data bundle, in one pass per clip, and
ship the results in batches of 500 to the public repo.

This is the **private** side: the code and how to run it. The public side holds only the results and
the specification.

---

## 0 · The two facts that shape everything

**~~The two interpreters do not coexist.~~ They do. That claim was wrong, and it was load-bearing.**

This playbook opened for months with *"`ultralytics` and `transformers` live in separate
environments — that is the installed reality, not a preference"*. From it came the two venvs, the
three passes, and half the complexity of `lotes.py`.

On **2026-09-23** the environment had to be rebuilt from scratch (the venvs lived in `%TEMP%` and
a cleanup deleted them — see `INCIDENTE_2026-09-23_VENV.md`). Both were installed into one venv and
**they coexist without trouble**: `ultralytics 8.4.160` and `transformers 5.16.1` import in the
same process, with `torch 2.14.0+cu126` and CUDA available. It was an installation accident nobody
re-tested.

So: **one interpreter, `D:\FENCINGLY_PY\venv`, and Windows and Colab are now the same path.** One
pass per clip. `pipeline.capacidades()` still asks what it can import, because that is the right
question in any environment — but now the answer is "everything".

Run `python 01_extraccion/noche.py --prevuelo` before any long run: it asks the interpreter what
it can actually import. The night that was lost was lost because nobody asked.

**Cost is dominated by two stages, not one, and the second one was mis-costed by a factor of 30.**

The famous figure — 17.3 s per clip, **83 % of it YOLO** — is *pass 1 only*. It is correct and it
still holds: everything else in pass 1 (shot gate, cross-fade gate, trajectories, pair selection,
all the cleanup) adds up to under 3 s. But this table used to put the 308-point poses at **~1 s**,
and that is wrong. Measured on three clips on an **idle** RTX 3060, 2026-09-22:

| | one clip | 8,559 clips | the 106,432 foil clips |
|---|---:|---:|---:|
| stage 1 — decode + YOLO + pair | **12.3 s** (17.3 s when the card is shared) | 29 h | 364 h |
| stage 4 — 308-point poses | **30.0 s** | **71 h** | **887 h** |
| stage 7 — piste, events, hand (CPU) | **13.0 s** | 31 h | 384 h |
| stage 9 — lights (CPU) | 1.9 s | 5 h | 56 h |
| stage 10 — the three labels | milliseconds | — | — |
| stage 5 — four dense layers | not reproducible, see below | | |

So a clip through stages 1, 4 and 7 is about **55 s**, not 17.3, and the poses cost **more than
YOLO**. Two consequences worth stating plainly:

- **The 512 h figure for the foil corpus is the pass-1 number.** With poses and events it is closer
  to **1,600 h** at today's speed. That is what makes the throughput work in §D of the plan
  load-bearing rather than nice to have.
- **Optimising YOLO alone cannot fix it.** YOLO is 83 % *of stage 1*, which is 22 % of the total.
  Halving YOLO saves ~11 % of the run. The pose pass is the bigger prize — and the first knob tried
  there, batching, was **measured and rejected**: 1.17× for a 130 px shift on usable keypoints. See
  `LOTE_POSE` in `video_revision.py`.

Stage 5, the dense layers, is off. The headline reason is not the 67 h: it is that the **28 s per
clip that justified the 67 h came from `tiempos_fase2.csv`, which no longer exists**. A number that
cannot be reproduced does not belong in a cost table.

---

## 1 · The run, per clip

```
# one interpreter, one pass, every stage
D:\FENCINGLY_PY\venv\Scripts\python.exe pipeline.py CLIP

# a corpus, resumable, downloading what is missing from the CDN
python 01_extraccion/noche.py --prevuelo                       # ask before you launch
python 01_extraccion/noche.py --conjunto anotados --descarga   # launch, detached
python 01_extraccion/noche.py --estado                         # how far it got
python 01_extraccion/noche.py --para                           # stop it
```

The three passes are gone with the two-venv premise. `pipeline.py CLIP --etapas 1,4,7` still
selects stages when you want a subset.

`python pipeline.py --lista` prints the ten stages, **which modules each needs installed**, which
files it eats, what it leaves on disk and its measured cost.

The pipeline used to ask "which interpreter am I", get back **one label**, and compare it with `==`.
That question has no correct answer in an environment where both libraries are installed — which is
every Colab — so it answered `"ultralytics"` and stages 4, 5, 7, 8 and 9 printed `SKIP` for ever.
Now it asks **what can I import**, which is a set, and a stage that cannot run says *which module*
or *which file* is missing. That change is what makes one interpreter possible -- and one
interpreter is what makes Windows and Colab the same path.

### What comes out

| file | where | contents |
|---|---|---|
| `det_<clip>.pkl` | `01_extraccion/pruebas` | raw YOLO detections, cached so re-running the selector costs nothing |
| `revision_<clip>.npz` | `01_extraccion/pruebas` | the two chosen boxes, presence, provenance, usable-frame mask, **fps** |
| `poses_<clip>.npz` | `01_extraccion/pruebas` | the 308 keypoints, with `origen` and `hueco` |
| `serie_eventos_<clip>.csv` | `02_validacion/pista` | the queryable event series, one row per event and side |
| `eventos_<clip>.csv` | `02_validacion/pista` | the same events in the flat layout of the aggregate |
| `posicion_clip_<clip>.csv` | `02_validacion/pista` | one row per clip: counts, rates, anchor drift, gates |
| `lights_<clip>.csv` | `01_extraccion/pruebas` | the scoring lights with their frame, order and `via` |
| `etiquetas_<clip>.csv` | `01_extraccion/pruebas` | the three labels with `origen_etiqueta` |

**This table used to promise `events_<clip>.csv` and that file has never existed.** The code writes
the three real names above, into `02_validacion/pista`. Renaming the code to match the promise would
break `grafica_pista.reune()`'s glob patterns *and* the deliberate exclusion at `grafica_pista.py`
that keeps the per-clip `posicion_clip_*.csv` from being swallowed by the aggregate
`posicion_y_eventos.csv`. So the playbook is what was wrong, and the playbook is what changed.

Every field is specified in the public README, with its unit, whether it is verified and **its
measured error**.

---

## 2 · The three stages that are OFF, each with its number

A stage that does not work ships **off, with the measurement next to the switch**. Not quietly
included.

| stage | switch | why |
|---|---|---|
| **right of way** | `PRIORIDAD = False` | 69.7 % accuracy against **73.8 % for never answering**, and 9.1 % on control clips. It is the goal of the project and it does not work yet |
| **3D cone** | `CONO_3D = False` | `pointmap` gives no usable depth: negative Z in 11.4 % of one clip, implied field of view 21-144 deg where a fencing camera uses 8-30, and the 11 discrepancies between 2D and 3D are **noise** - permuting the depths at random reproduces them exactly |
| **parry** | `PARADA = False` | no ground truth: 0 parries annotated, and blade contact turned out **not verifiable by eye** in this material (16 blind plates, zero decidable) |

The 2D cone stays on, and it is honest about what it is: the lateral distance in 2D is a **lower
bound** of the 3D one, so the 2D cone **never misses a real threat, it only has spares**.

---

## 3 · Batches of 500

Clips are processed and shipped in batches of 500. Rationale: a batch is about **2.4 h of pass 1** on
one GPU, so a batch is a session, not a week; and if something is wrong with a stage you lose one
batch, not the corpus.

```
python lotes.py --desde 0 --cuantos 500      # process
python lotes.py --sube 0                      # append that batch's results to the public repo
```

Two rules for the shipping step, and both exist because of mistakes already made in this project:

**Aggregates are rebuilt by reading the disk, never written from what was just computed.** A batch
that processes one clip must not be able to overwrite the aggregate with one clip. That happened: an
aggregate CSV ended up holding 40 rows of one clip instead of 284 rows of eleven, and the acceptance
test passed *because* the file only had one clip. The fix is structural, not a reminder.

**`None` is not zero.** When a clip has no piste homography - and **6 in 10 do not** - the metric
fields come out `None`, meaning *not measured*. Writing zero there would silently claim the fencer was
at the centre line.

---

## 4 · What to check before trusting a batch

Run the guarantees. They are twelve, they execute, and a red one is a real defect because the
threshold is what the code itself claims to respect.

```
python ../02_validacion/herramienta/tabla_checks.py cajas 60
python ../02_validacion/herramienta/tabla_checks.py poses
python ../02_validacion/herramienta/tabla_checks.py tabla
```

And run the symptom hunter, which is the complement: it does not check guarantees, it ranks clips by
suspicion and writes out the frames it flagged, so they can be looked at.

```
python ../02_validacion/herramienta/busca_errores.py 30 0
```

**The difference between the two matters.** Twelve green guarantees do not mean the product works;
they mean the delivered data does not contradict itself. The capability table in the public README is
what says whether something works, and several things do not.

---

## 5 · The defect that is still open

In about **6 % of clips the pipeline delivers the REFEREE as one of the two fencers.** Four routes
have been tried and **all four failed, each with a measurement**:

| route | why it failed |
|---|---|
| uniform whiteness | the gold **lamé** fires it on correct pairs; paired separation came out at **-0.002**, i.e. backwards |
| mask held in hand | with a loose threshold it becomes a darkness detector: fires on 469 of 3,386 patches while only 1 of 39 fencers holds a mask |
| geometry | the referee passed all three geometric gates **without touching them** (separation 1.93 of 3.20, foot delta 0.46 of 0.55, feet-in-band 0.894 of 0.40) - and it was predictable, because a referee walks beside the piste |
| movement behaviour | the referee scored **0.524** events per second, **above the p10 of real fencers (0.493)** and four times its own opponent's rate: walking arm-swing fires the arm-extension event |

Rows carry a `side_valid` flag and it is a **lower bound**. And the reason this is hard is worth
stating: **the referee's pose is perfectly good**. There is nothing wrong with the pose; what is wrong
is whose it is. No pose-quality metric will ever catch it.

---

## 6 · Piste: pending rework

The piste stage is being rebuilt and this section will be replaced when it lands. What is known now:

- the two **longitudinal edges** are detected in **100 %** of clips, which is the most solid
  measurement in that thread
- the **metric scale** passes its physical control in only **2.6 %** of clips, and **6 in 10 clips
  have no homography at all**
- the origin along the piste was a **declared convention**, not a measurement, which is why fencers
  appeared to start at the centre when they do not
- the fix under way rests only on what the **regulation** fixes and so is meant to be invariant to
  piste model: two parallel edges, a known width **excluding any added side plate**, and up to seven
  transverse landmarks at known distances (centre 0 m, on-guard +-2 m, colour-zone edge +-5 m, rear
  limits +-7 m). Their **spacing pattern 2:3:2** identifies which line is which without knowing the
  scale, and three or more landmarks give **redundancy**: fit with two, predict the third, and publish
  the error in centimetres. That is the independent control the thread never had.

Error will be reported **as a function of position along the piste**, not as one number: the
homography is exact but a pixel covers far more ground at the far end than near the camera.

---

## 7 · Things that were tried and do not work

Kept here so nobody spends a week rediscovering them.

| idea | measurement that killed it |
|---|---|
| depth from `pointmap` to order the fencers | orders correctly in 76 % of frames; the free **foot y** orders correctly in **100 %** |
| depth from Sapiens 1 | 73.2 %, below the free baseline, and it is an older model |
| the glove to identify the weapon hand | paired separation **-0.002**, correct sign 0.455: the whiteness feature reads the **sleeve** (pearson 0.681), and dark gloves exist |
| a free "recovery" of missing boxes from existing detections | poisons the position estimate the crop re-detection needs: one clip fell from **99.5 % to 81.6 %** |
| the thumb axis as the blade direction | lever arm of 2.4-4.8 px, jumps over 40 deg in 10-26 % of frames; the forearm jumps 0.9-2.1 deg |
| the audio beep as a light detector | **50.6 %** of the corpus has no audio track at all, and where it does, several pistes beep in the same hall |
| a cheap counterattack proxy from box velocity | fails both criteria and goes backwards: it measures the **body**, and a counterattack is an **arm** extension |

---

## 8 · The rule this project runs on

Declare the criterion and the number that decides **before** measuring, and do not move it afterwards.
Say the free baseline you have to beat. Break the sample down by bout or by venue **before**
aggregating. And when something fails, retire it with its measurement next to it rather than deleting
it.

It is not bureaucracy. In one day it caught: a criterion of mine that was **circular** (an axis that
scored well on "points at the opponent" and at **chance** against the real blade), three results that
**fell apart when the sample grew** (15/15 becoming 22/23, a feature changing sign, a dispersion going
from 14.0 deg to 4.2 deg), and an acceptance test that **passed because the file it queried held only
one clip**.

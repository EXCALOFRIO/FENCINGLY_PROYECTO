# The PISTE: what is in this folder, and what you can believe of each thing

What the user asked for:

> *"I don't see any chart measuring how they move along the length of the piste and the width, like a
> line — those charts that draw a line, like a racing car's trace around a track. I want the same but
> on a fencing piste, for both fencers, where they start and how it fills in ON VIDEO... And then
> getting each keypoint out: start of march, end of march, so we measure the length of each march...
> all tracked, so that for example I can ask for the third march of the right-hand fencer."*

## READ THIS FIRST: the three numbers that stop you misreading everything else

| | value |
|---|---:|
| **Step detection recall**, counted by eye against 47 strides in 3 fencers | **90 %** (see `boundary` caveat below) |
| **march / lunge confusion matrix**, 6 gestures annotated by eye | **5 of 6 correct, with `march -> lunge` = 0** |
| **Metric scale passing the physical gate** (both fencers' reprojected height in 1.60-2.00 m) | **2.6 % of clips (2 of 78)** |

The third one is the important one: **this is not a calibrated measuring instrument.** Displacements
are in metres; *where on the piste* is a declared convention, not a measurement. Every figure and
every video states its own status on screen, in red when it fails.

## THE DELIVERABLE

### 1 · `video/trace_<clip>.mp4` — the trace video

Four synchronised zones: the clip with both fencers marked (**green = left, orange = right**) and
their two foot contacts; the caption with frame / second / each fencer's position / distance between
them **and the status of the measurement**; **THE TRACE** (position along the piste against time,
drawing itself); and the **PISTE MAP** seen from above with the trail growing and events appearing as
they happen.

There is a **counter** per fencer and per type (`march`, `retreat`, `lunge`, `halt`, `arm`) that
**jumps on the event's start frame**, and the number shown is **the same `indice_del_tipo` as in the
CSV**, so you can go from one to the other. Events also flash on screen with their magnitude
("MARCH 3 0.31 m").

**15 clips at normal speed and 4 at quarter speed** (`_slow`, 7.5 fps), on purpose including the
most active (`ForcefulAntiqueAlbacoretuna`) and the least (`UglyNiceAplomadofalcon`).

`_public` versions replace the clip panel with the **308-point skeleton over black**, so that every
pixel of the MP4 is model output and none of it is broadcast footage. The other three panels (the
trace, the piste map, the counter) were already fully generated.

Generate: `python 02_validacion/herramienta/video_pista.py <clip> [--lenta] [--publico]`
(the flags are the command-line switches; the output files are named `_slow` and `_public`)

**Where the fencers are placed on the map is now MEASURED, not drawn.** Along the piste it comes from
the rectified strip with the origin anchored on the t.19 colour frontiers; across the piste it is
metres between the two detected edges. Both statuses are written on the map caption, and the events'
magnitudes are deliberately **not** rescaled: they stay the ones in `serie_eventos.csv`, because a
video and a CSV disagreeing about the same event would make the deliverable worthless.

### 2 · `serie_eventos.csv` — the events as a queryable time series

**One row per event**, not per frame. 650 rows over 22 clips. The query the user asked for is one
line:

```python
ev = pd.read_csv("serie_eventos.csv")
ev.query("lado == 1 and tipo == 'marcha' and indice_del_tipo == 3")
# -> one row per clip that has 3 or more marches on that side
```

**Column names are in Spanish and stay that way**: renaming them would break other agents' code.
Here is what they mean.

| column | meaning |
|---|---|
| `clip`, `lado` | clip name; `lado` 0 = left fencer in the image, 1 = right |
| `tipo` | `marcha` = **march**, `romper` = **retreat**, `fondo` = **lunge**, `alto` = **halt**, `brazo` = **arm extension**, `accion` = **action**, `bote` = **balestra** |
| `indice_del_tipo` | 1, 2, 3... per (side, type) ordered by start frame. Same number shown in the video |
| `cuadro_inicio`, `cuadro_fin`, `cuadro_pico` | start / end / peak **frame** |
| `t_inicio_s`, `t_fin_s`, `duracion_s` | start / end / duration in **seconds** |
| `x_inicio_m`, `x_fin_m` | position **along** the piste, in metres, at start and end |
| `y_inicio_m`, `y_fin_m` | **lateral** position. Always flagged `transversal_creible = False` |
| `magnitud_m`, `unidad_magnitud` | magnitude **and its unit**, because they are NOT all the same unit |
| `velocidad_m_s` | metres per second |
| `pies_movidos` | 1 or 2. **1 means a half step** |
| `solapa_fondo` | this step overlaps a lunge. **Only flags, never filters** |
| `arbitrado` | what the march/lunge arbiter decided: `paso` (step) or `fondo` (lunge) |
| `razon_pie_atras`, `pico_dpies_guardia` | the two features it decided with |
| `margen_arbitraje` | distance to the threshold. Small = contestable |
| `guardia_m`, `guardia_plausible` | the clip's own on-guard separation, used as a scale-free reference |
| `completa` | arm extension reaching 0.90 of the clip's p98 |
| `origen_max`, `hueco_max`, `pose_fiable`, `hueco_sin_medir` | pose provenance: was it **measured** or interpolated |
| `publicable`, `puerta_dos_lados`, `ancla_fiable` | credibility flags of the metric scale |
| `via_accion` | `metros` or `alto_de_caja` (see "clips without homography") |
| `lado_valido`, `lado_no_valido_por` | **`False` means this side is the REFEREE, not a fencer** |
| `sin_homografia` | no metric scale in this clip |

**`unidad_magnitud` is a column and not a footnote**, because the magnitudes are not in the same
unit: `fondo`, `marcha`, `romper`, `alto` and `accion` in metres along the piste; `bote` in vertical
metres on the fencer's box scale; `brazo` as a fraction of torso length.

Aggregates are **never written from what was just computed**: `reune()` rebuilds them by reading every
`*_<clip>.csv` from disk and checks it with two assertions. That is there because a single-clip run
once overwrote three 11-clip aggregates with one clip, and the acceptance criterion still passed — for
the wrong reason.

### 3 · `pista_<clip>.png` — the five-panel static figure

**Note on which coordinate each deliverable uses.** The five-panel figure still draws the
**homography-based** position (the one whose gate only 2 of 78 clips pass) and the lateral
coordinate re-centred on a lane. The **video** is the one that uses the rectified strip: measured
metres across the piste and a measured origin along it. They differ on purpose and each one says so
on its own panel; the section "There is more than one PISTE MODEL" below explains why.

`a` the map with the 2D trace and time in colour · `a2` the lateral coordinate, raw **and** filtered,
with its noise band · `b` position against time with events · `c` distance between the fencers ·
`d` feet separation.

Generate: `python 02_validacion/herramienta/grafica_pista.py <clip> [<clip>...]`

## THE SIX EVENT TYPES, and what you can believe of each

| type | criterion | median magnitude | status |
|---|---|---:|---|
| `fondo` **lunge** | feet separation grows >= **0.50 x the clip's on-guard** in 8 frames **and** the front foot advances >= 0.36 x on-guard | **0.565 m** | arbitrated against `march`, see below |
| `marcha` **march** / `romper` **retreat** | one foot moves above 1.25 m/s with the other below half, **on the smoothed along-piste coordinate**, paired with the other foot's stride | **0.346 m** | 41 % overlap a lunge: filter `solapa_fondo == False` |
| `alto` **halt** | travels >= 0.25 m in 8 frames then <= a threshold calibrated on that fencer's own noise | 0.30-0.66 m | **adaptive** threshold, see `umbral_quieto_m`. Means "drops below 0.3-0.7 m/s", not "motionless" |
| `brazo` **arm extension** | the `shoulder -> hand root` axis exceeds 1.15 x its own median; `completa` if it reaches 0.90 of the clip p98 | 0.29-0.50 x torso | axis **corrected**: the forearm is a bone and does not lengthen |
| `accion` **action** | start and end in the same row | — | — |
| `bote` **balestra** | both soles leave the floor together | — | **NOT ESTABLISHED: 0 events**, and an assertion guarantees it |

## `alto` IS NOT A FENCING PARRY, and the parry is NOT implemented

It used to be called `parada` and the name was misleading. In **fencing**, a parry is the defensive
action of deflecting the opponent's blade with your own, and it transfers priority to the riposte: it
is a **weapon** event. What this detector emits is that the fencer **stops travelling**, which is a
**footwork** event.

**The parry in the fencing sense is not implemented and cannot be with this material**: the priority
stage closed with 0 annotated parries and measured that blade contact is not verifiable by eye in this
corpus. Reading "95 halts" must not be understood as parry-ripostes detected.

## `march` and `lunge` are MUTUALLY EXCLUSIVE: there is an arbiter and an assertion

The user saw it in slow motion: *"it goes to march, to lunge, and then march again. It detects a lunge
there that is not a lunge."* There were **73 overlaps** of different types on the same fencer.

**Two causes, two fixes:**

1. **The lunge threshold was in absolute metres**, so in a clip with inflated scale an ordinary step
   crossed it. Measured correlation between a clip's scale factor and its lunges per second:
   **r = 0.83**. It is now normalised by the clip's **own on-guard separation**, with the coefficient
   chosen to reproduce the declared numbers (0.50 x 0.70 = 0.35 m). It is **symmetric**: it tightens
   where the scale is inflated and loosens where it is compressed, and it inherits the quality of the
   on-guard measurement — of the 17 sides with an on-guard value, **10 fall in 0.50-0.90 m** and 7 do
   not (0.265 to 1.371 m).

2. **The discriminator is THE REAR FOOT, not the magnitude** (which overlaps: 0.346 vs 0.565 m). In a
   march the rear foot FOLLOWS; in a lunge it stays PLANTED. Two features, measured in the
   **extension phase** (up to the peak of feet separation, not to the end of the window, which
   includes the recovery and corrupts the measurement): `razon_pie_atras` (threshold 0.45) and
   `pico_dpies_guardia` (threshold 1.25).

Result: **73 overlaps -> 0**, with an **assertion** that aborts if it happens again.
`ForcefulAntiqueAlbacoretuna` left side went from `MFMFMFMFMMFMRFMRRMRMFMFMFMFMFMFM` to
`MMMMMMMRFRRMRMMMMMMM`.

**With thresholds of 0.55 and 1.30 the accuracy rises to 6 of 6, and they were NOT adopted**: that is
fitting the threshold to six examples. Both numbers are published and the declared one is used.

## THE THREE THINGS YOU CANNOT BELIEVE, and they are on screen

1. **Across the piste: now MEASURED metres, and the physical control FAILS.** The lateral coordinate
   is no longer a drawn lane: it is the row of the foot between the **two detected piste edges**,
   scaled by the regulation 1.50 m (t.18.1), so it is **bounded by construction**. The number:
   the range is <= 0.60 m in **11 of 40 fencer-clips** (p50 **0.75 m**, p90 1.63 m, max **3.33 m**),
   and **both** fencers pass in only **3 of 22 clips**. **2.7 % of foot contacts (median per clip,
   max 23 %) fall outside the piste.** It is not jitter: the per-frame noise is **0.9 cm**, so
   filtering changes the range by ~2 cm. A 1.85 m range on a 1.50 m piste is not a measurement, it
   is a warning, and it is written on the panel.

2. **Along the piste: the ORIGIN is measured in 13 of 22 clips, the SCALE is not measured at all.**
   The origin comes from the **t.19 colour-zone frontiers**, whose positions are ±5 m and ±7 m by
   regulation: **13 of 22** clips, **11** unambiguous, **5** with enough redundancy to be controlled,
   and where there is redundancy the **held-out mark is predicted with 5 cm median / 8 cm worst
   error**. That is what fixed the defect the user saw ("*you take by default that they always start
   in the centre*"): in `UglyNiceAplomadofalcon` the two fencers now come out at **+2.69 m and
   +5.00 m**, the right one exactly on the 5 m frontier where the light final zone starts.

   The SCALE is a different matter. **Three independent routes were measured and none passes its own
   control**: painted lines by separation ratios (fits 14 of 20, **ambiguous in 10 of those 14**,
   and off by factors of 1.3-2.6 against the reference homography); square pixels from the two ground
   vanishing directions (exact camera solution, but the reprojected fencer heights come out at
   **0.4-4.9 m**); and the colour zones themselves (13 of 22, only 5 with any control). And there is
   a geometric reason why this is not a matter of tuning: **two parallel lines with known separation
   fix the ACROSS scale everywhere and say nothing about the ALONG scale.**

3. **The publication gate is weaker than its own docstring says.** It reads "the reprojected height
   of BOTH fencers falls in 1.60-2.00 m" and the code compared the **pooled** median. Applied per
   fencer, **2 of 78 clips pass (2.6 %)**, not 10 (12.8 %). Both figures are in `homografia.csv`
   (`publicable` and the derived `publicable_estricto`).

## There is more than one PISTE MODEL, and the census counts them

`piste_models.csv` + `piste_models_1..4.png`: the person-free plate of each clip **rectified into a
strip** where the rows run across the piste and the columns along it, with the two detected edges
marked.

> **The four PNGs are NOT publishable and must stay out of the public repos.** They are resampled
> broadcast pixels (the piste plate, and in two clips the scoreboard with athlete names). The
> publishable part of this census is the **table below and `piste_models.csv`**, which are measured
> numbers only.

Measured columns only, no impressions: piste width in pixels at the start / middle / end of the
visible run, number of colour frontiers along the axis, whether the final zone is **lighter or
darker** than the rest (the *sign* of the luminance step), whether a lateral plank is found, and the
visible run in metres.

| | of 22 clips |
|---|---:|
| rectified strip (both edges detected) | **20** |
| >= 2 colour frontiers along the axis | **15** |
| final zone identified | **13** |
| lateral plank DETECTED | **0** |

Of those 13 final zones, **8 are darker than the piste and 5 are lighter**. So the user was right and
**the colour scheme is not invariant**: 5 of 13 are "inverted" with respect to the other 8. That is
why the detector looks for a **CHANGE** of colour and never for a colour — a "look for grey" detector
would have failed on more than a third of this corpus.

**The lateral plank: 0 of 22 detected, and that is a declared failure, not an absence.** The plausible
explanation is that the edge sweep already stops at the plank boundary, so the detected edges are the
combat zone and there is nothing to exclude — but **that is not measured**, and until it is, the plank
in the clip the user pointed at is not identified by the code.

### Why a rectified strip makes the regulation usable

Map the two detected edges onto two horizontal lines and the longitudinal vanishing point goes to
horizontal infinity; that forces **rows to be independent of the along-piste coordinate, and columns
to be AFFINE in it**. Verified numerically on three clips with a reference homography: their
longitudinal vanishing point lands on strip columns 2.5e10, 4.1e9 and -2.5e10, i.e. at infinity in
all three. Consequence, and it is exactly the user's intuition ("*since you know it is straight, the
perspective does not matter*"): **the spacings between marks can be compared with the regulation's
2 : 3 : 2 without correcting perspective at all.**

### The error map, which replaces "millimetric precision"

A single error figure would be false at both ends of the piste. The strip reports `px_img_por_col` —
how many image pixels one strip column is worth at each point — and from it the error in metres per
pixel along the axis. The ratio between the fine end and the coarse end is **1.35 median, 3.51 worst**
over the 20 clips: in the worst clip one pixel is worth **3.5 times more metres** at one end than at
the other. Sampling is cut where the piste width drops below 30 % of its central value, for this
reason.

## The axis convention, with an assertion

**x increases towards the right of the image**, and `lado 0` is the left-hand fencer. Checked:
`lado 0` on the left in the image in 11 of 11 clips, with the smaller metric x in 11 of 11, and
**0 crossings in 2,896 frames** — two foilists cannot walk through each other, so that is a physical
control. Column `cruces_imposibles`.

## Clips WITHOUT homography: only `accion` and `brazo`

Having the 308-point pose is **not enough**: `marcha`, `romper`, `fondo` and `alto` have thresholds in
metres and also need the homography. Of 22 clips, **7 do not have one**.

What can still be measured: **`accion`** in **box-heights per second** (threshold 0.29, which is
`0.50 m/s / 1.75 m`, the same criterion in another unit), and **`brazo`** as a fraction of torso,
which **never needed the homography**. In those clips `pasos_por_s_util_y_tirador`,
`cruces_imposibles` and `ancla_deriva_m` are **`None`, not zero**, with a `sin_homografia` column: the
difference between "not measured" and "measured and came out zero" is what this project keeps paying
for.

## The two REFEREES delivered as fencers

`PowerfulAllLarva` side 0 and `IdolizedFancyChickadee` side 1 are referees, verified by eye. They are
2 of 33 annotated clips (6.1 %) and it is the most recurrent unsolved defect.

**Their events are all emitted**, flagged in `lado_valido` and `lado_no_valido_por` — never by the
absence of a row: if they were filtered out their zero would be indistinguishable from "not measured",
and what you want to know about a referee is precisely **what he does not have**.

**And the `lunge + arm` per second criterion does NOT separate them**: 0.100 and **0.524**, and the
0.524 is above the p10 of real fencers (0.493). Also `fondo` is zero by construction in those clips
because they have no homography, so only half the criterion runs.

## `verifica/` — verification by eye

`apoyo_<clip>.png` (detected feet and the reprojected metric grid), `ev_<type>_*.png` (detected) and
`fn_<type>_*.png` (**candidates NOT detected**). A type with no non-detected candidates looked at is
not verified: looking only at hits, a detector that fires once and is right looks perfect.

## `anotar/` — boundary annotation, pending

21 numbered frame strips from 3 fencers plus `fronteras.csv` with the columns ready. What is measured
so far is **detection** (90 % recall) and **classification** (5 of 6), but **nobody has measured
whether the BOUNDARIES are right**: an event can exist, be the right type, and start 8 frames late.

The boundary convention is declared **before** annotating, in `CONVENIO_FRONTERAS.md`, and written on
the images themselves: **the boundary is the TAKE-OFF** (the frame where the foot no longer touches),
not "when it starts moving", which is 2-4 frames earlier.

## `diagnostico/` — NOT a deliverable

`homog_*.png` and `censo_*.png` are **internal diagnostics**: magenta overlays, piste regions in
green, residuals written on top. They tell you WHY a homography failed, which is essential to work,
and they are not meant to be shown.

## Interpreter

`<scratchpad>/fz/.venv/Scripts/python.exe` — the only one with cv2 + numpy + pandas + **matplotlib** +
scipy at once. `<scratchpad>/cuda/.venv` **has no matplotlib**.

---

Note on language: **code docstrings and the internal report `04_documentacion/BITACORA_2026-09.md` are in Spanish on
purpose.** The project was worked in Spanish and translating them now would only introduce errors.
Everything that is *seen* — figures, videos, this file and the two convention files — is in English.

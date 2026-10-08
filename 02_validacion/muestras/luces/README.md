# Foil scoring lights: ground truth, figures and measurements

In foil the **scoring apparatus** shows a **red light** and a **green light** for a **valid touch**
(one colour per fencer), and a **white light (off-target)** for a touch landing outside the **valid
target** — arm, leg, mask or the piste. On television these lights are drawn inside the broadcast
**scoreboard overlay**, and that overlay changes with every tournament and every broadcaster. A
detector tuned on one tournament fails on the others; that is what this folder measures.

Start with **`figuras/scoreboard_families_EN.png`**: 21 tournaments, 5 scoreboard families, with the
coverage figures inside the image.

---

## The ground truth was annotated by an AGENT, not by a referee

`verdad_luces_ojo.csv` was annotated by **me, the agent**, by looking at the frames in `figuras/`
and `tiras/`. **It is not a refereeing annotation and has no regulatory value.** All it distinguishes
is "the green bar on the right turns on at frame 204" from "it does not", which is the only thing
needed to measure a light detector.

What this annotation does **not** say: whether the touch was valid, who had **right of way**, whether
the referee awarded it, or who got the point.

### How each column was obtained, and how precise it is

| column | how | precision |
|---|---|---|
| `n_luces_ojo` | counting the lights in the **full** scoreboard band 12 frames after onset (`figuras/light_count_*.png`) | exact: both name plates are visible at once |
| `luces_ojo` | channel and side of each light, read from the same figure | exact |
| `cuadro_ojo_principal` | see below | `tolerancia_cuadros` |

**The onset frame has two different provenances and it matters which.**

1. **Five venues with an INDEPENDENT measurement** (`kazanfoil2020`, `parisfoil2022`,
   `stpetersburgfoil2019`, `tokyoolympics2020`, `turinfoil2023`): the frame comes from a filmstrip of
   consecutive frames covering the clip, read **before** looking at what any detector said.
   Precision = the filmstrip step (2 to 6 frames).
2. **Sixteen venues with VERIFICATION, not measurement**: the frame is the one my detector produced,
   and what I checked by eye is that the light is **OFF** at frames −10 and −4 and **ON** at +2 and
   +8 (`figuras/encendido_*.png`). That brackets the onset to the interval `(det−4, det+2]`, i.e. 6
   frames, but **it is not independent of the detector** and cannot rule out a systematic bias
   smaller than that window.

The distinction matters: with those 16 it is **not** possible to claim the detector has no 2-frame
bias. With the 5 independent ones it is, and that is where the only large error showed up
(`stpetersburgfoil2019`, 41 frames late).

---

## The two cases where eye annotation overruled both detectors

- **`parisfoil2022`**: there is **ONE** light, green, the small central square of the overlay, turning
  on at frame 231. The project's existing detector reported **two** lights (red left at frame 221,
  green right at 231). In the filmstrip, frame 221 shows nothing. This family has a single central
  indicator, so it carries **no left/right information**, and the detector invented a side.
- **`turinfoil2023`**: there is **ONE** light, **green**, on the **right**, on from frame ~146 to
  ~230. The existing detector reported **red, left, frame 291**, by which time the light had been off
  for 60 frames. Wrong colour, wrong side, wrong instant by 5.7 s — with `status = "ok"` and an empty
  flag list.

---

## Files

| file | what it contains |
|---|---|
| `verdad_luces_ojo.csv` | the eye-annotated ground truth for 21 clips, one per venue |
| `luces_detectadas.csv` | every light found by `luces.py` on those 21 clips, with its confidence broken into factors |
| `generalizacion_por_sede.csv`, `generalizacion_por_familia.csv` | fit set (FIE family) vs test set (4 other families), reported separately |
| `no_lights_por_sede.csv` | how many clips the existing detector loses per venue, over 25,139 clips |
| `reparto_luces_por_sede.csv`, `canales_por_sede.csv` | one-light / two-light / white-light split per venue |
| `muestra_b.csv` | an independent second clip per venue, drawn with a different seed |
| `recupera_no_lights.csv`, `no_lights_que_son.csv`, `roi_aprendida.csv`, `roi_por_sede.csv` | what the lost clips actually are, and the per-venue learned ROI that recovers them |
| `una_luz_y_contraataque.csv` | approach velocity of both fencers before the touch, and the counterattack proxy (which **fails**) |
| `audio_zumbador.csv` | the buzzer tone of the scoring apparatus, measured against a control window |
| `censo/`, `figuras/`, `tiras/` | the census of the 21 overlays, the verification figures, the filmstrips |

Column names are kept in Spanish on purpose: they are the ones the code reads.

---

## The headline numbers

| | measured | sample |
|---|---:|---|
| scoreboard families among 21 venues | **5**, one covering 81 % of clips | census, looked at one by one |
| clips with **no light detected** by the existing detector | **8.11 %** (2,038) | 25,139 clips |
| worst venue | **36.78 %** (`tokyofoil2022`) | 1,433 clips |
| ...but of those lost clips, how many actually contain a light | **10 %** | 20 clips, learned ROI |
| clips with **two** lights | 21.14 % | 10,840 clips |
| clips with a **white light (off-target)** | 5.74 %, and **0** clips where white is the only light | 10,840 clips |
| "the fencer who lights **second** wins" | **64.96 %** (majority baseline 52.18 %) | 1,815 two-light clips |
| "the fencer who lights wins", on one-light clips | **96.19 %** | 8,049 clips |
| per-venue learned ROI, on the independent sample | **90.5 %** | 21 clips |
| general detector, same sample and same criterion | 71.4 % — **fails** | 21 clips |

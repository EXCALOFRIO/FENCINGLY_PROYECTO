"""FENCINGLY - one pipeline, one clip in, one data bundle out.

WHAT THE USER ASKED FOR, in their words: "un repo privado donde esta el como extraerlo, que sera un
Colab o algo asi, los archivos de codigo, que tendra que organizarlo en uno solo que saque todo etc,
ya limpio y perfecto en un solo pipeline bonito y perfecto".

So this is the single entry point. It does **not** reimplement anything: every stage calls the module
that was measured and validated for it. Reimplementing would fork the code and the fork would fall
behind the moment someone fixes a stage, which is exactly the failure this file exists to prevent.

    python pipeline.py CLIP [CLIP ...]        run every stage whose inputs and libraries are there
    python pipeline.py --lista                show the stages, what they need and what they leave
    python pipeline.py --clip CLIP --hasta 4  stop after stage 4
    python pipeline.py CLIP --etapas 1,4,7    run exactly these

## "WHICH INTERPRETER AM I" WAS THE WRONG QUESTION, AND IT COST NINE STAGES

This file used to ask `cual_interprete()`, which returned **one** label and was compared with `==`.
It tried `import ultralytics` first, so in any environment where that import works -- which is every
Colab, where everything is installed -- it answered `"ultralytics"` and stages 4, 5, 7, 8 and 9
printed `SKIP: needs transformers` **for ever**. The stages were never broken. They were never asked.

"What can I import" is a **set**, not a label, and that is what `capacidades()` returns now. Each
stage declares:

    necesita   which modules have to import for it to run at all
    depende    which files have to exist already, because the stage eats them

A stage that cannot run says **which module or which file is missing**, and that is a different
sentence from "wrong interpreter". The two-venv Windows workflow keeps working unchanged -- not
because the pipeline was told a label, but because the two venvs genuinely differ: in the
`ultralytics` one, `transformers` is absent, so stage 4 skips itself and says so.

## WHAT COMES OUT

`revision_<clip>.npz` (boxes, presence, shot gate), `poses_<clip>.npz` (308 points with provenance),
the three per-clip CSVs of the piste and the events, `lights_<clip>.csv`, and
`etiquetas_<clip>.csv`. Every field is documented in `02_validacion/DATOS_POR_CLIP.md`, with its
unit, whether it is verified and **its measured error**.

## WHAT IS DISABLED ON PURPOSE, AND WHY

Four stages exist and are **off**, each with its measurement written where the switch is:

  right of way: 69.7 % accuracy against 73.8 % for never answering -> `PRIORIDAD = False`
  3D cone from `pointmap`: every physical control fails, and the 2D/3D discrepancies are noise
  parry: 0 annotated, and blade contact is not verifiable by eye in this material
  dense layers (stage 5): its cost figure was deleted -> `CAPAS_DENSAS = False`

A stage that does not work is off with its number next to it, not quietly included.
"""
from __future__ import annotations

import pathlib
import sys
import time

RAIZ = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "codigo"))
sys.path.insert(0, str(RAIZ.parent / "02_validacion" / "herramienta"))

import rutas as RU  # noqa: E402

CODIGO = RU.CODIGO
PRUEBAS = RU.TRABAJO
VAL = RU.VAL
HERRAMIENTA = RU.HERRAMIENTA

PRIORIDAD = False
"""Right-of-way detection is OFF. Measured: **69.7 %** accuracy against **73.8 % for never
answering**, and 9.1 % on control clips. It is the goal of the project and it does not work yet, so it
does not ship. Section 19 of the master document has the rule-to-signal map, which is the part that
does keep its value."""

CONO_3D = False
"""The 3D cone is OFF. `pointmap` does not give usable depth: negative Z in 11.4 % of one clip, an
implied field of view of 21-144 degrees where a fencing camera uses 8-30, and the 11 discrepancies
between 2D and 3D are **noise** - permuting the depths at random reproduces them exactly (1.00x
against a declared limit of 2x). The 2D cone stays, and it is a lower bound: it never misses a real
threat, it only has spares."""

PARADA = False
"""Parry detection is OFF. Not because the code is missing: because there is **no ground truth**. Zero
parries annotated in 15.4 s of reviewed action, and blade contact turned out **not verifiable by eye**
in this material - of 16 blind plates, zero were decidable. The regulation makes the parry the hinge
that transfers right of way (FIE t.85), so this is the most valuable thing still missing."""

CAPAS_DENSAS = False
"""Stage 5, the dense layers, is OFF, and the headline reason is **not** the cost.

The cost is what was quoted to justify it -- 28 s per clip, 67 h over the corpus -- and
**`tiempos_fase2.csv` no longer exists**, so that 28 s is not reproducible today. A number that
cannot be reproduced must not sit in a cost table as if it had been measured. Three more reasons,
in order: `pointmap` is already retired as unusable for depth and `CONO_3D = False`, so one of the
four heads has no consumer; `capas_limpias.Capas.__init__` has a 6-attempt VRAM retry loop with a
600 s wait, which on a shared Colab GPU can stall a batch for ten minutes on one clip; and it is
1.6x the cost of the entire campaign. Turning it on means first re-measuring its cost in Colab."""


class Etapa:
    """A stage: what it needs installed, what it needs on disk, and what it leaves behind."""

    def __init__(self, n, nombre, necesita, depende, salida, corre, nota=""):
        self.n, self.nombre = n, nombre
        self.necesita = tuple(necesita)
        self.depende = tuple(depende)
        self.salida, self.corre, self.nota = salida, corre, nota


def _detecta_y_pareja(clip, ctx):
    """Stages 1-3: decode, run YOLO on every frame, and pick the two fencers.

    The pair is chosen **per trajectory and once per clip**, never frame by frame: that is what makes
    an identity swap impossible by construction rather than by threshold.
    """
    import video_revision as VR
    VR.fase1([clip], videos={clip: ctx["video"]} if ctx.get("video") else None)
    import numpy as np
    z = np.load(PRUEBAS / ("revision_%s.npz" % clip), allow_pickle=True)
    return ("both fencers present in %.1f %% of usable frames, over %d frames"
            % (100 * float(z["ambos"]), int(z["T"])))


def _poses(clip, ctx):
    """Stage 4: the 308-point poses, the empty-pose filter, the re-trim and the invention."""
    import video_revision as VR
    VR.fase2([clip], video=False)
    return "poses written with `origen` and `hueco`; the review mp4 is OFF in the pipeline path"


def _eventos(clip, ctx):
    """Stage 7: the piste homography if the clip allows it, the position in metres and the events.

    THIS IS THE STAGE THAT REPORTED SUCCESS WITHOUT DOING ANYTHING. It used to be
    `GP.main([clip]) if hasattr(GP, "main") else None`, and `grafica_pista` has no `main`: the
    guard swallowed the whole stage and the caller printed a tidy line with a measured-sounding
    number next to it. The real writer lived inside `if __name__ == "__main__"`, so the only way to
    obtain the three CSVs was to launch the file as a program.

    `GP.procesa` is that block, lifted into a function. `con_figura=False` because five matplotlib
    panels per clip is a tool for looking at one clip, not for crossing a corpus. And it does
    **not** call `GP.reune()`: `reune` re-reads the whole directory, so calling it per clip is
    O(n^2). The batch calls it once, from `lotes.py`.
    """
    import grafica_pista as GP
    f = GP.procesa(clip, con_figura=False)
    if f is None:
        return "no homography and no action: nothing written, and that IS the answer"
    return ("3 CSVs in %s; march %s, lunge %s, arm %s; no homography: %s"
            % (RU.PISTA.name, f.get("n_marcha"), f.get("n_fondo"), f.get("n_brazo"),
               f.get("sin_homografia")))


def _luces(clip, ctx):
    """Stage 9: the scoring lights, with the per-venue ROI when the venue is one of the 21.

    THE TWO RATES DO NOT GET AVERAGED. The **90.5 %** belongs to the learned per-venue ROI, and
    `roi_por_sede.csv` covers **21 venues**. A clip from any other venue gets the general detector,
    which is **71.4 %**. So every row carries `via`, and the rate is reported broken down by `via`.
    Averaging them would produce a number that is true of no clip.
    """
    import csv
    import luces as LU
    roi = RU.roi_sede(ctx.get("torneo"))
    filas, via, _fps = LU.lee_clip(ctx["video"], roi)
    """QUE DETECTOR, EN QUE ORDEN: `luces.lee_clip`, el mismo que usa la etapa 11 cuando no hay CSV.
    Con `FENCINGLY_LUCES_V2` primero el detector viejo (`luces_v2`, `via = v2_fie | v2_generic`) y,
    si no da tocado, el de abajo con `via = tras_v2_<via>` (`LUCES_POR_FUENTE_2026-10-09.md`).

    EL RESPALDO DE LA ROI, medido el 2026-10-08. La ROI de sede no encontraba NINGUNA luz en el 40 %
    de 582 clips anotados (anaheimfoil2018, qatarfoil2021 y katowicefoil2019 sobre todo), con la
    barra roja bien visible en el video: el detector general la ve con confianza 1,0. Como respaldo
    recupera **63 de 80** de esos clips. Y su cuadro es MENOS fiable que el de la ROI: donde
    contestan los dos, coinciden a +-3 cuadros en el **64 %** de 70 clips. Por eso no sustituye a la
    ROI, solo entra cuando la ROI no da nada, y lleva `via = general_tras_roi`."""
    for f in filas:
        f["clip"], f["via"] = clip, via
    p = PRUEBAS / ("lights_%s.csv" % clip)
    cols = (["clip", "via"] + sorted({k for f in filas for k in f} - {"clip", "via"})) \
        if filas else ["clip", "via"]
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas or [{"clip": clip, "via": via}])
    return "%d lights, via=%s (see LUCES_POR_FUENTE_2026-10-09.md for the rate by via)" % (len(filas), via)


def _zona(clip, ctx):
    """Stage 11: the touch position in metres, from the painted lines propagated with the camera
    motion to the frame where the lights come on. Reuses `lights_<clip>.csv` from stage 9."""
    import zona_tocado as ZT
    return ZT.corre(clip, ctx)


def _etiqueta(clip, ctx):
    """Stage 10: the three labels from the trained labeller, with `origen_etiqueta = modelo`.

    It costs **no GPU**: it eats the series that stages 4 and 7 just wrote, not the video. That is
    what makes labelling the whole corpus free next to extracting it.
    """
    import etiquetador as ET
    r = ET.etiqueta_clips([clip])
    return r[0] if r else "no weights in 01_extraccion/modelos: nothing written"


ETAPAS = [
    Etapa(1, "decode + detect (YOLO11m-pose, every frame)",
          ("ultralytics", "cv2"), (),
          "det_<clip>.pkl + revision_<clip>.npz", _detecta_y_pareja,
          "12.3 s/clip measured on an idle RTX 3060 (17.3 s when the card is shared)"),
    Etapa(2, "shot gate: close-ups, replays, cross-fades",
          ("ultralytics", "cv2"), (), "(inside revision_<clip>.npz)", None,
          "a cross-fade is told from a zoom by the residual relative to contrast, 0.219 vs 0.062"),
    Etapa(3, "pick the pair, per trajectory",
          ("ultralytics", "cv2"), (), "(inside revision_<clip>.npz)", None,
          "identity swap impossible by construction; a track never seen whole is discarded"),
    Etapa(4, "308-point poses (Sapiens 2)",
          ("transformers", "torch", "cv2"), ("revision_%s.npz",),
          "poses_<clip>.npz", _poses,
          "2.26 px torso error at 384x288. 30.0 s/clip: MORE than stage 1, see LOTE_POSE"),
    Etapa(5, "dense layers, cleaned (seg, normal, pointmap, matting)",
          ("transformers", "torch"), ("revision_%s.npz",), "(off, see CAPAS_DENSAS)", None,
          "OFF: its 28 s/clip came from tiempos_fase2.csv, which no longer exists"),
    Etapa(6, "piste homography and position in metres",
          (), ("det_%s.pkl",), "(inside stage 7)", None,
          "6 of 10 clips have no homography at all; those fields come out None, not zero"),
    Etapa(7, "movement events: march, retreat, lunge, halt, arm",
          (), ("det_%s.pkl", "poses_%s.npz"),
          "serie_eventos_<clip>.csv, eventos_<clip>.csv, posicion_clip_<clip>.csv "
          "(in 02_validacion/pista)", _eventos,
          "90 % recall, `march -> lunge` confusion = 0. 13 s/clip, CPU only"),
    Etapa(8, "weapon hand", (), ("poses_%s.npz",),
          "(the `mano` column of serie_eventos_<clip>.csv)", None,
          "runs INSIDE stage 7, without the `encarado_accion` rung: see the note below"),
    Etapa(9, "scoring lights", ("cv2",), (), "lights_<clip>.csv", _luces,
          "old fencingly_v2 detector first (LUCES_POR_FUENTE_2026-10-09): one-light agreement with "
          "`touch` in foil 0.801 -> 0.947 (held-out half, n=641), frame 0 0.094 -> 0.014. By `via`"),
    Etapa(10, "the three labels, predicted", ("torch",),
          ("poses_%s.npz",), "etiquetas_<clip>.csv", _etiqueta,
          "no GPU: eats the series, not the video. `origen_etiqueta = modelo`, never merged"),
    Etapa(11, "where the touch was: painted lines + camera motion, at the light",
          ("cv2",), ("poses_%s.npz",), "zona_<clip>.csv", _zona,
          "zone ADOPTED: 0.476 vs middle 0.353 (+0.123, CI95 [+0.090, +0.155], n=1313); vs stage 7 +0.141 CI95 [+0.094, +0.187]"),
]

NOTA_8 = """STAGE 8 HAS NO `corre` ON PURPOSE, AND ITS NUMBER WAS THE WRONG ONE.

`pipeline._mano` used to import `cono_lib` and return a sentence **without calling anything**. It
was not merely a no-op: it was a duplicate, because `posicion.py` already calls
`cono_lib.mano_armada` and the answer is column `mano` of `serie_eventos_<clip>.csv`.

And the note said "1.000 over 19 fencers of a held-out set, 74 % coverage", which is two figures
from two different owners glued together. The correct attribution:

    the geometric rung alone (nearest hand)     0.741 (20 of 27), later 0.831 over 59
    the front-foot rung, firm vote              12 of 12, covering 12 of 19; ungated 0.895 over 19
    the FULL cascade, pie_firme -> encarado_accion -> mayoria_3
                                                1.000 on the 19 held out, 0.949 over all 59

And what stage 7 actually runs is **none of the three**: `posicion.py` calls `mano_armada` without
`accion=`, so the `encarado_accion` rung cannot fire and the chain is
`pie_firme -> hombro_firme -> nariz_dir_dudosa -> geometria_a`. Passing `accion` would move the
piste numbers, so it is identified and NOT done here.

MEASURED 2026-10-09 (`ETAPA7_PUERTAS_2026-10-09.md` §4), behind `FENCINGLY_MANO_ACCION`: against
the FIE licence hand, 9,751 decisions / 958 fencers, the full cascade lifts per-clip accuracy* from
0.878 to 0.892, but the declared criterion was PER FENCER and there it is 9 wins vs 6 losses,
p = 0.61. It did not pass, so it ships OFF (`0`)."""


def capacidades():
    """QUE DEVUELVE: el conjunto de modulos que de verdad se pueden importar.

    Un CONJUNTO, no una etiqueta. La pregunta "en que interprete estoy" no tiene respuesta correcta
    en un entorno con las dos bibliotecas, que es el caso de Colab, y contestarla igualmente fue lo
    que dejo cinco etapas en `SKIP` para siempre.
    """
    tiene = set()
    for m in ("ultralytics", "transformers", "torch", "cv2", "numpy", "pandas", "matplotlib"):
        try:
            __import__(m)
            tiene.add(m)
        except Exception:
            pass
    return frozenset(tiene)


def puede(e, cap, clip, trabajo=None):
    """QUE DEVUELVE: (si puede correr, el motivo de que no).

    Dos motivos distintos y se dicen distintos: falta una biblioteca, o falta un fichero de
    entrada. "Wrong interpreter" no distinguia ninguno de los dos.
    """
    trabajo = pathlib.Path(trabajo or PRUEBAS)
    falta = [m for m in e.necesita if m not in cap]
    if falta:
        return False, "no esta instalado: %s" % ", ".join(falta)
    sin = [p % clip for p in e.depende if not (trabajo / (p % clip)).is_file()]
    if sin:
        return False, "falta la entrada: %s" % ", ".join(sin)
    return True, ""


def lista():
    cap = capacidades()
    print("FENCINGLY pipeline - stages, what they need, and what they leave")
    print()
    print("what this interpreter can import: %s" % ", ".join(sorted(cap)))
    print()
    print("%-3s %-48s %-26s %s" % ("#", "stage", "needs installed", "leaves on disk"))
    print("-" * 118)
    for e in ETAPAS:
        print("%-3d %-48s %-26s %s"
              % (e.n, e.nombre, ", ".join(e.necesita) or "(nothing)", e.salida))
        if e.depende:
            print("    eats: %s" % ", ".join(e.depende))
        if e.nota:
            print("    %s" % e.nota)
    print()
    print(NOTA_8)
    print()
    print("OFF ON PURPOSE, each with its number:")
    for nom, var, val, por in (
            ("right of way (the goal)", "PRIORIDAD", PRIORIDAD,
             "69.7 % against 73.8 % for never answering"),
            ("3D cone from pointmap", "CONO_3D", CONO_3D,
             "every physical control fails; the 2D/3D discrepancies are noise"),
            ("parry", "PARADA", PARADA,
             "0 annotated; blade contact not verifiable by eye in this material"),
            ("dense layers (stage 5)", "CAPAS_DENSAS", CAPAS_DENSAS,
             "its 28 s/clip came from a file that no longer exists")):
        print("   %-26s %-14s = %-6s  %s" % (nom, var, val, por))
    print()
    print("Every output field is documented in 02_validacion/DATOS_POR_CLIP.md, with its unit,")
    print("whether it is verified and its MEASURED ERROR.")


def corre(clips, hasta=99, etapas=None, videos=None, trabajo=None, verboso=True):
    """QUE DEVUELVE: una fila por (clip, etapa) con n, nombre, estado, segundos y mensaje.

    EL ESTADO DE UN CLIP NO SE DEDUCE DE ESTA LISTA. Se deduce de los ficheros que existen; esta
    lista es **registro**. Un contador ya dijo en este proyecto "completado = 3" habiendo 6
    ficheros, y desde entonces el registro y la verdad son cosas distintas.
    """
    cap = capacidades()
    trabajo = pathlib.Path(trabajo or PRUEBAS)
    sel = set(etapas) if etapas else None
    if verboso:
        print("can import: %s" % ", ".join(sorted(cap)))
        print()
    filas = []
    for clip in clips:
        ruta = (videos or {}).get(clip) or RU.video(clip)
        ctx = {"video": ruta, "clip": clip}
        try:
            import catalogo as CA
            df = CA.manifiesto()
            f = df.loc[df["clip_id"] == clip]
            if not f.empty:
                ctx["torneo"] = f.iloc[0]["torneo"]
        except Exception:
            pass
        if verboso:
            print("== %s%s" % (clip, "" if ruta else "   (no mp4: only the stages that do not"
                                                     " need one)"))
        for e in ETAPAS:
            if e.n > hasta or e.corre is None or (sel is not None and e.n not in sel):
                continue
            ok, por = puede(e, cap, clip, trabajo)
            if ok and not ruta and e.n in (1, 9, 11):
                ok, por = False, "falta el mp4"
            if not ok:
                if verboso:
                    print("   %-2d %-44s SKIP: %s" % (e.n, e.nombre[:44], por))
                filas.append({"clip": clip, "etapa": e.n, "estado": "skip", "mensaje": por})
                continue
            t = time.perf_counter()
            try:
                msg = e.corre(clip, ctx)
                est = "ok"
            except Exception as ex:
                msg, est = "%s: %s" % (type(ex).__name__, str(ex)[:90]), "fallo"
            d = time.perf_counter() - t
            if verboso:
                print("   %-2d %-44s %6.1f s  %s%s"
                      % (e.n, e.nombre[:44], d, "FALLO " if est == "fallo" else "", (msg or "")[:70]))
            filas.append({"clip": clip, "etapa": e.n, "estado": est,
                          "s": round(d, 2), "mensaje": msg})
        if verboso:
            print()
    return filas


CON_VALOR = ("--hasta", "--etapas", "--clip", "--trabajo")
"""Las opciones que se comen el argumento siguiente. Sin esta lista, `--etapas 7,9,10` dejaba
`7,9,10` en la lista de clips y el pipeline intentaba procesar un clip llamado "7,9,10", que
naturalmente daba `falta la entrada: poses_7,9,10.npz`. Un mensaje impecable sobre una pregunta
absurda."""


def _argumentos(argv):
    clips, i = [], 0
    while i < len(argv):
        x = argv[i]
        if x in CON_VALOR:
            i += 2
            continue
        if x.startswith("--"):
            i += 1
            continue
        clips.append(x)
        i += 1
    return clips


if __name__ == "__main__":
    a = _argumentos(sys.argv[1:])
    if "--clip" in sys.argv:
        a.append(sys.argv[sys.argv.index("--clip") + 1])
    if "--lista" in sys.argv or not a:
        lista()
    else:
        h = 99
        if "--hasta" in sys.argv:
            h = int(sys.argv[sys.argv.index("--hasta") + 1])
        et = None
        if "--etapas" in sys.argv:
            et = [int(x) for x in sys.argv[sys.argv.index("--etapas") + 1].split(",")]
        corre(a, h, etapas=et)

"""Deja el repo PUBLICO con dos cosas: la ficha que lo explica todo, y los videos que lo ensenian.

LO QUE PIDIO EL USUARIO, acumulado:

  *"solo quiero tener en el publico un readme explicando lo que hemos sacado de los clips. Ademas pon
  que son de FencingDatabase y das las gracias a su equipo por ceder los clips. Y cuentas todo lo que
  hemos hecho, las poses, los canales, con cada cosa que cuente dos videos que se puedan ver. Y luego
  todo: florete, los conos, la pista, absolutamente todo, la prioridad. Y una tabla con todos los
  datos que recopilamos, el formato, la forma y su precision. Y esos datos por clip son lo que le
  pasaremos al modelo para que aprenda."*

  *"sale un Dataset Viewer que no lo queria para nada"* -> `viewer: false` **y** ningun CSV.

  *"en vez de verlo en un grid, mete de una en una"* -> una seccion y un video **por capa**.

  *"mejor mete/incrusta video que sera mejor que un gif"* -> `<video>`, no GIF.

## LAS CUATRO DECISIONES DE ESTE FICHERO

**1. `viewer: false` y ningun CSV en el repo.** El visor de Hugging Face aparece solo: basta con que
haya un CSV para que la ficha se llene de una tabla interactiva que nadie pidio. Se apaga en la
cabecera Y se quita el CSV, porque apagar el visor dejando el fichero invita a que vuelva en cuanto
alguien suba otro. La tabla de campos va escrita en la ficha, que es lo que se pidio.

**2. `<video>` con URL ABSOLUTA DE HUGGING FACE, y la misma en las dos paginas.** Comprobado con
`curl`: una URL `resolve/main` de un repo publico responde `content-type: video/mp4`,
`content-disposition: inline` y `accept-ranges: bytes`, o sea que el navegador la reproduce y puede
empezar sin descargarla entera. Y sirve **tambien desde el README de GitHub**, porque el navegador va
a buscarla a huggingface.co. Con `raw.githubusercontent.com` no funcionaria: los `.mp4` de ahi salen
como `application/octet-stream` y el navegador los descarga en vez de reproducirlos. Debajo de cada
video va un enlace markdown de respaldo, por si la pagina filtra la etiqueta.

**3. GIF *Y* `<video>`, y no es un duplicado: son DOS RENDERIZADORES.** Esta decision estaba al
reves y la corrigio el usuario mirando GitHub: *"faltan los GIF... no hay GIFs largos, metelos"*.
El `<video src="https://huggingface.co/...">` se reproduce en la ficha de Hugging Face, pero **el
README de GitHub no renderiza un `<video>` con origen externo**, asi que la pagina espejo se veia
**sin una sola imagen en movimiento**. Ahora cada capa y el video de pista llevan el GIF DELANTE del
`<video>`: GitHub pinta el GIF, HF pinta los dos, y el MP4 completo sigue enlazado debajo para verlo
entero y con calidad. Los GIF los genera `previas.py` **de estos mismos MP4**, no aparte, para que no
puedan discrepar.

**4. Cada capacidad con su numero al lado, incluidas las que NO funcionan.** La prioridad es el
objetivo del proyecto y se publica como que no funciona, con la linea base gratis que no bate. Una
ficha que solo lista lo que sale bien no es una ficha, es un folleto.
"""
from __future__ import annotations

import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sube_evidencia as SE

PUB = "EXCALORIO/fencingly-datos"
TIPO = "dataset"
VAL = SE.VAL
PROY = VAL.parent
GEN = VAL / "publico_generado"
URL = "https://huggingface.co/datasets/%s" % PUB
CRUDO = "https://huggingface.co/datasets/%s/resolve/main/media" % PUB
GITHUB = "EXCALOFRIO/FENCINGLY.AI"
LOTE = 500
MEDIOS = GEN / "medios"
PRIO = VAL / "prioridad" / "publico"

SE_QUEDAN = ("README.md", ".gitattributes")

GIF_CAPA = {q: "capa_%s.gif" % q for q in ("pose", "seg", "normal", "matting", "pointmap", "foil")}
GIF_PISTA = ("pista_completa.gif", "pista_trazado.gif", "pista_mapa.gif")
GIFS_PUBLICOS = tuple(GIF_CAPA.values()) + GIF_PISTA


def gif(nombre, pie=None):
    """Un GIF incrustado con markdown normal, que **se ve en los dos renderizadores**.

    Markdown `![]()` lo pintan tanto Hugging Face como GitHub; `<video src="https://...">` solo lo
    pinta HF. Por eso el GIF va SIEMPRE delante del `<video>` y no en su lugar: quien abra la ficha
    en HF vera el video completo debajo, y quien la abra en GitHub vera al menos el GIF.
    """
    if not (MEDIOS / nombre).exists():
        return ""
    l = ["![%s](media/%s)" % (pathlib.Path(nombre).stem, nombre)]
    if pie:
        l += ["", "*%s*" % pie]
    return "\n".join(l)


CAPAS = [
    ("pose", "1 - POSE: 308 keypoints per fencer",
     "Two fencers, 308 points each, every frame. Green is side 0 (image left), orange is side 1. A "
     "red box is a person the detector found and we **did not** select. The insets at the bottom are "
     "the weapon hand at native resolution, because each hand is 21 of those 308 points and at full "
     "frame they are a blur.",
     ["**2.26 px** torso error from running the pose at 384x288 instead of 1024, which is **7.8x** "
      "cheaper",
      "**96.0 %** of delivered poses are measured; the rest are interpolated and **say so**",
      "**0 held poses**: a held pose is the same pose repeated, and it would look like a fencer "
      "standing still",
      "**10,140** keypoints were landing outside their own person's box - on the other fencer - and "
      "are now zeroed",
      "there is **no wrist index** in this map: the wrist **is** the hand root. The map was measured, "
      "not read off a datasheet",
      "**the invented poses can be partly replaced by measured ones, and this was just measured.** "
      "Of the **650** invented poses in 38 clips (2.96 % of delivered fencer-frames, all of them "
      "interpolated - `held` is **0** everywhere), re-running the pose head alone on that exact frame "
      "recovers a valid measured pose in **50 of the 295** whose gap is <= 24 frames. In a control "
      "where fully measured frames were hidden and re-measured, the rescue costs **1.94 %** of torso "
      "length against **21.02 %** for interpolating, and it is closer to the truth in **98.2 %** of "
      "626 cases",
      "**and for gaps of 25 frames or more it does not work, which is the useful half of that "
      "result**: only **2 of 355** pass the quality gate, while an **empty patch of background** in "
      "the same frame passes **45 of 355**. The rescue finds a fencer in the estimated box *less "
      "often than in a piece of floor*, because **the fencer is not in those frames at all**. So "
      "*beyond 24 frames there is no measurement* is not a tuning problem, and those 355 - **54.6 % "
      "of everything invented** - stay unverifiable"]),
    ("seg", "2 - SEG: body-part segmentation",
     "The cleaned mask, blended **only inside itself**. Outside it the frame is just darkened, which "
     "is deliberate: that is how you can check the mask landed on the right person instead of on the "
     "referee walking past.",
     ["referee leakage into the mask measured **0.0000**",
      "the mask is `(matting >= 0.5 OR seg != background) AND nearest-skeleton AND box guard`, and "
      "the loss is attributed to the three of them separately - a low recall does not say what to fix",
      "stored **per crop**, not per frame: the two masks cover **4.6-4.8 %** of the frame, so "
      "full-frame storage would waste 21x"]),
    ("normal", "3 - NORMAL: surface normals",
     "One axis per colour channel. This is the dense layer that **does** hold up for local geometry.",
     ["outside the mask the value is **NaN and not 0**, because the zero vector is not a unit normal "
      "but it would average silently inside a convolution"]),
    ("matting", "4 - MATTING: alpha, the soft silhouette",
     "The soft silhouette. At very low alpha the **edge of the blade** appears, and that is what the "
     "foil detector works on.",
     ["the footer prints the **mean alpha per side**. A near-zero one is not a bug in the video: "
      "`matting` returns background when a second person is inside the crop, and then that fencer's "
      "silhouette is empty. A null has to say it is a null and why"]),
    ("pointmap", "5 - POINTMAP: the depth-like channel we do NOT use",
     "Shown because it was asked for, not because it is used. It is kept as a channel and the "
     "pipeline ignores it.",
     ["it orders the two fencers in depth correctly in **76 %** of frames - and the **y coordinate of "
      "the free foot**, which costs nothing, gets it right **100 %** (28 of 28)",
      "Sapiens 1 depth scored **73.2 %**, below the free baseline, on a newer model's predecessor",
      "the 3D cone is off for the same reason: negative Z in 11.4 % of one clip, implied field of "
      "view 21-144 deg where a fencing camera uses 8-30, and the 11 places where 2D and 3D disagreed "
      "were **noise** - permuting the depths at random reproduced them exactly"]),
]


def medios_de(patron):
    return sorted(MEDIOS.glob(patron))


def video(nombre, alto=None):
    """Un `<video>` con URL absoluta de Hugging Face, mas un enlace markdown de respaldo."""
    p = MEDIOS / nombre
    if not p.exists():
        return ""
    return ("<video src=\"%s/%s\" controls loop muted playsinline width=\"100%%\"></video>\n\n"
            "[%s - MP4, %.1f MB](media/%s)\n" % (CRUDO, nombre, nombre, p.stat().st_size / 1e6,
                                                 nombre))


INCRUSTADOS = 2
"""Cuantos videos se INCRUSTAN por seccion. Los demas se enlazan.

Hay tres clips por capa y cada video pesa 5-12 MB. Incrustar los tres son 30 MB de `<video>` en la
misma seccion, y una pagina que tarda en abrir no cumple su funcion. Pero **enlazarlos tambien hace
falta**: si un fichero se sube y no se referencia desde ninguna parte, esta ahi ocupando sitio y nadie
lo encuentra, que es peor que no subirlo."""


def videos_de_capa(q):
    fs = medios_de("%s_*.mp4" % q)
    l = [video(p.name) for p in fs[:INCRUSTADOS]]
    resto = fs[INCRUSTADOS:]
    if resto:
        l.append("More clips, same layer: " + " | ".join(
            "[%s](media/%s)" % (p.stem.replace(q + "_", ""), p.name) for p in resto) + "\n")
    return "\n".join(l)


def fijos_de_capa(q, cuantos=2):
    """Incrusta `cuantos` fijos y ENLAZA el resto.

    POR QUE SE ENLAZA EL RESTO. Se suben 4-5 fijos por capa y por clip y solo se incrustan dos, asi
    que la primera version dejo **65 PNG subidos a los que no apuntaba nada**. Un fichero publicado y
    no referenciado ocupa sitio, no lo encuentra nadie y encima no se puede saber si sobra o si el
    enlace se rompio. Se veia contando: 83 subidos, 18 referenciados.
    """
    fs = medios_de("%s_*_c*.png" % q)
    if not fs:
        return ""
    paso = max(1, len(fs) // cuantos)
    puestos = fs[::paso][:cuantos]
    l = ["![%s](media/%s)" % (p.stem, p.name) for p in puestos]
    resto = [p for p in fs if p not in puestos]
    if resto:
        l.append("\nMore stills, same layer: " + " | ".join(
            "[%s](media/%s)" % (p.stem.replace(q + "_", ""), p.name) for p in resto))
    return "\n".join(l) + "\n"


def seccion_portada():
    """LA PORTADA: el video de pista ARRIBA DEL TODO, porque es lo que se quiere ver primero.

    El usuario lo pidio dos veces y la segunda fue *"lo de la pista no lo veo en el readme"*. Y no lo
    veia por dos razones medidas, las dos mias:

      1. **estaba al 51 % del documento**, detras de seis GIF de capa de 5-6 MB. Para llegar habia
         que bajar media pagina cargando 33 MB antes.
      2. **los GIF pesaban 5,2-8,2 MB** y GitHub sirve las imagenes del README por su CDN: a ese
         tamanio tardan tanto que parecen no estar. Ahora ninguno pasa de 3 MB.

    Asi que el mosaico de la pista sube a la portada, que es donde ensena de un golpe las cuatro
    cosas a la vez: las poses arriba a la izquierda, el evento de cada tirador escrito debajo, el
    trazado a la derecha y el recorrido por la pista abajo.
    """
    g = gif("pista_completa.gif")
    if not g:
        return ""
    return "\n".join([
        "## The whole thing in one frame", "",
        "Top left, the two skeletons on the clip with each fencer's position in metres. Under them, "
        "the label with **what each fencer is doing right now** - march, retreat, lunge, halt. On "
        "the right, the **trace**: position along the piste against time. Along the bottom, the "
        "**piste from above**, to scale, with each fencer's trail behind them and the events "
        "appearing as they happen.", "",
        g, "",
        "*10 s, generated over the clip. Everything below is one layer at a time, with what it is "
        "worth next to it.*", "",
        "---", ""])


def seccion_deteccion():
    """COMO SE ENCUENTRA UNA POSE, RONDA A RONDA, con la cifra de cada una.

    Lo pidio el usuario: *"un ejemplo de alguna pose que no detectaba, como con la redeteccion ya
    funciona, y el porcentaje que no detectaba y luego con la redeteccion cuantas ha sacado"*.

    Las cifras salen de `cascada_deteccion.json`, que las cuenta sobre los NPZ entregados. Aqui no se
    escribe ni una a mano.
    """
    import json
    f = PROY / "01_extraccion" / "pruebas" / "cascada_deteccion.json"
    if not f.exists():
        return ""
    c = json.loads(f.read_text(encoding="utf-8"))
    d = c["dentro"]
    pct = lambda n: 100.0 * n / d
    l = ["### How a pose gets found: four rounds, and what each one recovers", "",
         "A pose is not simply *detected or not*. There are four rounds, and **each one is counted "
         "separately** so you can see what would be missing without it.", "",
         "**The pose is Sapiens 2 in every round.** What changes between rounds is **where the box "
         "comes from**: YOLO gives only the box, and the 308 points are always measured by the same "
         "pose head. Round 2 exists because that head is **top-down** - it needs no detector to "
         "confirm the person, only a box - so it can try exactly where the detector already failed "
         "twice. Round 3 is the only one with no measurement left in it.", ""]
    if (MEDIOS / "cascada_deteccion.png").exists():
        l += ["![cascada_deteccion](media/cascada_deteccion.png)", ""]
    l += ["| round | where the **box** comes from (the pose is always Sapiens 2) | recovers | "
          "of %s fencer-frames |" % "{:,}".format(d),
          "|---|---|---:|---:|",
          "| **0** | YOLO on the whole frame | %s | %.2f %% |" % ("{:,}".format(c["t0"]),
                                                                 pct(c["t0"])),
          "| **1** | **YOLO again, on a crop** - the redetection | %s | %.2f %% |"
          % ("{:,}".format(c["t3"]), pct(c["t3"])),
          "| **2** | **interpolated box, no detector at all** | %d | %.2f %% |"
          % (c["resc"], pct(c["resc"])),
          "| **3** | no box and no pose: **the pose itself is interpolated** | %d | %.2f %% |"
          % (c["inv"] - c["resc"], pct(c["inv"] - c["resc"])),
          "",
          "Read it as a cascade: **without round 1 the pose would be missing in %.2f %%** of "
          "fencer-frames. Round 1 - the redetection - recovers **%s of them**, leaving **%.2f %%** "
          "with no measured pose. Round 2 recovers **%d more**, and what is left, **%.2f %%**, is "
          "interpolated and **carries a flag saying so**." % (
              pct(d - c["t0"]), "{:,}".format(c["t3"]), pct(c["inv"]), c["resc"],
              pct(c["inv"] - c["resc"])),
          "",
          "Round 2 is only attempted where the gap is **24 frames or shorter** (%d of %d cases). "
          "Beyond that it is not a tuning problem: the quality gate passes **0.6 %%** of the time "
          "with a fencer in the box and **12.7 %%** on an empty patch of background, so it does not "
          "discriminate at all - the fencer is simply not in those frames."
          % (c["resc"], c["intent"]), ""]
    ej = sorted(MEDIOS.glob("redeteccion_*.png"))
    if ej:
        l += ["**A redetection you can look at.** Left: the full frame, with the box where the "
              "detector found nobody. Right: the same box as a crop, where the pose came out.", ""]
        l += ["![%s](media/%s)" % (q.stem, q.name) for q in ej]
        l += [""]
    return "\n".join(l)


def seccion_capas():
    """LAS CINCO CAPAS, UNA POR SECCION Y CON SU VIDEO. El florete y la pista van aparte.

    Antes esto era una sola rejilla de seis celdas en 2880 px de ancho, o sea 960 por celda, y el
    usuario dijo *"en vez de verlo en un grid, mete de una en una"*. Tenia razon por una razon
    medible: a 960 px los rotulos de cada celda no se leen.
    """
    l = []
    for q, tit, texto, puntos in CAPAS:
        l += ["### %s" % tit, "", texto, ""]
        g = gif(GIF_CAPA.get(q, ""), "8 s of the layer, 720 px. The MP4 below is the whole clip "
                "at full resolution - the GIF is here because GitHub does not play the video.")
        if g:
            l += [g, ""]
        v = videos_de_capa(q)
        if v:
            l += [v, ""]
        f = fijos_de_capa(q)
        if f:
            l += ["Full-resolution stills:", "", f, ""]
        l += ["\n".join("- %s" % x for x in puntos), ""]
    return "\n".join(l)


def seccion_florete():
    l = ["### 6 - FOIL: the threat cone, as a graded probability field", "",
         "The cone is what turns *where the arm points* into *who can be hit*. It is **not** a flat "
         "wedge: the fill **is the weight function the measurement itself uses**, so the figure and "
         "the number cannot disagree. This project already lost a day to a visualisation that did not "
         "match its own data, so the drawing and the metric are now the same code.", "",
         "The yellow cross is where **the two cones cross each other**. That is the test - cone "
         "against cone, not cone against torso.", ""]
    g = gif(GIF_CAPA["foil"], "8 s of the cone. The MP4 below is the whole clip at full "
            "resolution - the GIF is here because GitHub does not play the video.")
    if g:
        l += [g, ""]
        """EL FOIL LLEVA SU GIF AQUI Y NO EN `seccion_capas`, porque el cono tiene seccion propia y
        no esta en la lista `CAPAS`. Sin esta linea `capa_foil.gif` se subia al repo y **no lo
        referenciaba nadie**, que es justo lo que `main` avisa como fichero huerfano."""
    v = videos_de_capa("foil")
    if v:
        l += [v, ""]
    f = fijos_de_capa("foil", 3)
    if f:
        l += ["Full-resolution stills:", "", f, ""]
    l += ["- reach **1.10 m** wrist-to-tip. A foil blade is **0.90 m** on its own, and the first "
          "version used exactly that - which is wrong for a *threat* radius, because the threat "
          "starts at the wrist and the guard and the hand add the rest. It looked too short and it "
          "was",
          "- the axis is the **forearm**, `elbow -> hand root`. Using the hand axis instead put the "
          "cone more than 40 deg off in **63.6 %** of frames; the forearm brings that to **15.7 %**",
          "- the arcs are **measured, not chosen**: solid = p50 core **+-14.4 deg half-angle "
          "(28.8 deg full)**, dashed = p90 envelope **+-33.0 deg half-angle (66.0 deg full)**, both "
          "from |detected blade - forearm axis| over 31 blades. Every angle here is written as *half "
          "and full*, on purpose: that undeclared convention already cost this project five "
          "contradictory figures in one day. Those two arcs are the pair for an **assumed** "
          "direction; where the blade was measured there is only one arc, at +-8.6 deg, because "
          "there is no measured p50 for that provenance and inventing one would be a third figure",
          "- when a blade is actually **detected** the field is **centred on the measured blade "
          "itself** (`origen 0`) and narrows to **+-8.6 deg**, which is **3.8x** narrower than the "
          "+-33.0 deg it uses when the direction is only assumed. Centre and width are **one "
          "decision**, not two: the 33.0 is by definition the p90 of |blade - forearm|, so the "
          "moment the centre moves onto the blade that percentile no longer applies, and 8.6 is the "
          "p90 of what is left over",
          "- **two figures on this page were wrong and are retired with their measurements**, "
          "because that is the only way a correction is worth anything. The **layer videos used to "
          "centre the cone on the forearm while the caption said it was centred on the measured "
          "blade** - in the frame a reader pointed at, the blade sat **21.4 deg** off the axis, "
          "outside the solid p50 core, and it was not a rare frame: **26 of 60**. And the width for "
          "a seen blade was published as **4.2 deg**, which is the frame-to-frame tremor of the "
          "blade's *direction* (70 pairs over 12 bouts, itself a correction of an earlier 14.0 from "
          "8 pairs of one bout) - a real number, but **not** the width of a wedge whose apex is the "
          "hand: the detected blade line misses the hand root by 7 px median at only 0.24 of the "
          "reach, and a few pixels that close to the apex subtend many degrees. At 4.2 deg the cone "
          "**did not contain the blade it was centred on** in 21 of 60 frames. `temblor_hoja_gr` "
          "still ships, as its own field, because it measures something real - just not this",
          "- **honest limit, broken down instead of pooled**: a blade line is found in **42.6 %** of "
          "crops at 1920x1080 and in **1.6 %** at 1280x720 - and **81 %** of the corpus is 720p. So "
          "most of the time the cone is a **geometric prior, not a detection**, and a single pooled "
          "percentage would hide which clips it is true for",
          "- the thumb axis was tried as the blade direction and **failed**: lever arm of 2.4-4.8 px, "
          "jumps over 40 deg in 10-26 % of frames, against 0.9-2.1 deg for the forearm", ""]
    return "\n".join(l)


def seccion_pista():
    vs = [p for p in medios_de("trace_*.mp4")
          if "_slow" not in p.stem and "_public" not in p.stem]
    lentos = [p for p in medios_de("trace_*.mp4")
              if "_slow" in p.stem and "_public" not in p.stem]
    """EL FILTRO BUSCABA `_public` Y ESOS YA NO SE SUBEN, asi que las dos listas salian VACIAS y
    la ficha no incrustaba ni enlazaba **un solo video de pista**: solo quedaba el GIF de 12 s.
    El usuario lo vio antes que ninguna comprobacion -- "sube un par de ejemplos completos, no
    solo el gif". Es el tercer sitio con el mismo criterio obsoleto, despues de `previas.PISTA_PUB`
    y `junta_medios`: cuando se revierte una regla transversal hay que buscar TODOS los sitios que
    la implementan, y aqui eran cuatro."""
    l = ["### The piste: where each fencer is, in metres", "",
         "This was rebuilt from scratch, because the first version had a defect worth stating "
         "plainly: **the origin was a convention, not a measurement**, so fencers appeared to start "
         "near the centre line when they do not. It now rests only on what the **rulebook** fixes, "
         "which is what makes it independent of the piste model.", "",
         "Left: the 308-point skeleton with each fencer's position in metres. Right: the trace - "
         "position along the piste against time, which is the racing-line view. Bottom: the piste "
         "from above, to scale, with the regulation lines and the events appearing as they happen. "
         "The counter is there so the count can be judged frame by frame rather than taken on trust.",
         ""]
    g = gif("pista_completa.gif",
            "12 s of the whole four-zone frame: skeletons and metres top-left, the label with the "
            "event underneath, the trace on the right, the piste from above along the bottom.")
    if g:
        l += [g, ""]
    l += ["The same thing zoomed into the two panels that carry the measurement, because at full "
          "frame their axis labels are too small to read:", ""]
    for nom, pie in (("pista_trazado.gif",
                      "**The trace.** Position along the piste on the horizontal axis, **time** "
                      "downwards. Green is the left fencer, orange the right. Both coordinates in "
                      "this panel are measured: x by homography, time exactly."),
                     ("pista_mapa.gif",
                      "**The piste from above, to scale**, with the regulation lines. Each fencer "
                      "is a thick dot, the trail behind is where they have been, and the events - "
                      "march, retreat, lunge, halt - appear labelled as they happen.")):
        gg = gif(nom, pie)
        if gg:
            l += [gg, ""]
    for p in vs:
        l += [video(p.name), ""]
    if lentos:
        l += ["At quarter speed, for judging the detection lag frame by frame: " + " | ".join(
            "[%s](media/%s)" % (p.stem.replace("trace_", "").replace("_slow", ""), p.name)
            for p in lentos), ""]
    l += ["**What is measured now**, each with its own denominator, never pooled:", "",
          "| | value | out of |",
          "|---|---:|---|",
          "| rectified strip obtained (both edges) | **20** | 22 clips |",
          "| **origin MEASURED** from the t.19 colour frontiers | **13** | 22 clips |",
          "| of those, unambiguous | **11** | 22 clips |",
          "| of those, with redundancy, i.e. **controlled** | **5** | 22 clips |",
          "| both fencers land on the piste (\\|x\\| <= 7.5 m) | **11** | the 13 anchored clips |",
          "| **held-out landmark prediction error** | **p50 5 cm, max 8 cm** | 5 clips |",
          "| lateral travel <= 0.60 m (the physical control) | **11** | 40 fencer-clips |",
          "| clips where **both** fencers pass that control | **3** | 22 clips |",
          "| lateral travel p50 / p90 / max | **0.75 / 1.63 / 3.33 m** | 40 fencer-clips |",
          "| foot contacts falling outside the piste | **p50 2.7 %, max 23 %** | 20 clips |",
          "| fencers crossing each other (physical control) | **0** | 20 clips |",
          "| extra lateral plank detected | **0** | 22 clips |",
          "",
          "The **5 cm** row is the one that matters: fit the anchor with two landmarks, predict a "
          "**third one it never saw**, and measure the miss. That is the independent control this "
          "part never had.", "",
          "- **the origin is measured, the along-piste scale is not.** Three independent routes were "
          "tried and **none passes its own control**: painted lines by separation ratios (fits 14 of "
          "20 clips, ambiguous in 10 of those 14, off by 1.3-2.6x against the reference), square "
          "pixels from the two ground vanishing directions (an exact camera solution that "
          "re-projects fencer heights of **0.4 to 4.9 m**), and colour zones (13 of 22, only 5 "
          "controlled). There is a **geometric** reason it is not a matter of tuning: two parallel "
          "lines with a known separation fix the scale **across** the piste everywhere and say "
          "**nothing** about the scale **along** it. So the map is labelled *origin MEASURED, scale "
          "NOT verified*",
          "- **across** the piste is now metres between the two **detected** edges, bounded by "
          "construction. The earlier version assigned each fencer a fixed lane, so the one on the "
          "right always appeared glued to the right edge; that is gone",
          "- the lateral coordinate is still **shape without scale**: travel of p50 **0.75 m** "
          "against a control of **< 0.60 m**, and in only **3 of 22 clips** do both fencers pass it. "
          "There is a physical reason: across the piste is the camera's **depth** direction, and a "
          "foot in the air projects displaced towards the background",
          "- **there is more than one piste model, and here is the number**: of the 13 clips where "
          "the final zone is identified, **8 are darker** than the piste and **5 are lighter**. A "
          "detector that looked for grey would fail on more than a third of the corpus, so "
          "everything here detects a **colour change** and never a colour",
          "- where there is no scale, the metric fields come out **None, not zero**. Writing zero "
          "would claim the fencer was standing on the centre line", ""]
    return "\n".join(l)


def seccion_eventos():
    return "\n".join([
        "### The movement events", "",
        "March, retreat, lunge, halt, arm extension - one row per event, not per frame. This is the "
        "low-dimensional series the model is meant to learn from.", "",
        "- **90 % recall**, and the confusion that mattered, **march mistaken for lunge, is 0**",
        "- the discriminator is the **rear foot**: in a lunge the front foot flies and the rear foot "
        "**stays planted**. That binary property, not the distance, is what separates the two, and "
        "the ratio is measured **in the extension phase** and not over the whole window",
        "- overlaps between different event types went from **73 to 0**, enforced by an assertion",
        "- detection lag **+0.5 frames (+17 ms)** over 62 pairings, with no bias",
        "- `balestra` and `jump` ship with **0 events**. The balestra was killed by ballistics: the "
        "8.5 cm of rise it needs would require **7.9 frames of flight**, and the candidate events "
        "lasted one frame", ""])


def seccion_mano():
    l = ["### Which hand holds the weapon", "",
         "What decides is a **cascade**, and its first rung is the rule that carries it: **in the "
         "on-guard stance the front foot is on the same side as the weapon arm** (FIE t.115.1). It "
         "is counter-intuitive and it is what works. If the front-foot vote is not firm, an "
         "on-guard-and-in-action gate takes over; what neither settles is flagged `uncertain`.", ""]
    if (MEDIOS / "weapon_hand.png").exists():
        l += ["![weapon hand](media/weapon_hand.png)", ""]
    l += ["- **the cascade: 1.000 accuracy over 19 of 19 fencers** of a held-out set that played no "
          "part in choosing the rule, and **0.949 over all 59** fencers with ground truth",
          "- **the front-foot rule on its own**, with a firm vote: **12 of 12 with no failure, "
          "covering 12 of those 19**. Ungated it scores **0.895 over 19** and 0.881 over 59. The "
          "1.000 belongs to the cascade and the coverage to the rule, and they are not the same "
          "claim: an earlier version of this page merged them into one sentence",
          "- the previous rule, *the hand closest to the opponent*, scored **0.831 over 59** - wrong "
          "about one time in six, and 0.741 on the first 27 annotated",
          "- the glove was tried and **failed**: the whiteness feature reads the **sleeve** (pearson "
          "0.681), and dark gloves exist. Paired separation **-0.002**, i.e. backwards",
          "- what cannot be decided is flagged `uncertain` rather than guessed", ""]
    return "\n".join(l)


def seccion_prioridad():
    """LA SECCION DE LA PRIORIDAD. **YA NO SE INCLUYE EN LA FICHA PUBLICA**, y se conserva entera.

    QUITADA POR DECISION DEL USUARIO: *"quita esa seccion que no me gusta que salga"*. Lo que se
    quita es el ESCAPARATE -- la seccion larga y sus cinco medios incrustados -- no la medida.

    POR QUE NO SE BORRA EL CODIGO. Las cifras de aqui son el resultado mas caro de este proyecto:
    69,7 % contra un 73,8 % que cuesta **no contestar**, el desglose que ensena que pierde en las dos
    mitades del trabajo, el 86,1 % de acierto de lado sobre las 36 de 121 veces que se atreve, y los
    dos avisos de denominador (462 observaciones contra 455 cuadros distintos, y 70,35 % al repetir).
    Borrarlas para que no salgan en una pagina seria perderlas. Siguen aqui, se pueden volver a
    meter en la ficha con una linea, y **el defecto sigue declarado en la tabla de capacidades** con
    su cifra al lado.

    Lo que se arregla de los problemas que esta seccion listaba entra por otro lado: la mano armada
    era uno de ellos y ya esta corregida. Fuera del escaparate, dentro del trabajo.
    """
    l = ["### Right of way - the goal of the whole project, and it does not work yet", "",
         "Foil is a weapon of **convention**: the valid touch is not simply the first one to land. "
         "When both lights come on, a referee decides who had **right of way** (FIE t.82-t.89). That "
         "is the label we are trying to learn, and this is the honest state of it.", ""]
    fs = sorted(MEDIOS.glob("prio_*"))
    for p in fs:
        if p.suffix == ".mp4":
            l += [video(p.name), ""]
        else:
            l += ["![%s](media/%s)" % (p.stem, p.name), ""]
    l += ["**Who holds right of way in this frame** - our detector, against the baseline that costs "
          "nothing:", "",
          "| | value | measured over |",
          "|---|---:|---|",
          "| our right-of-way detector | **69.7 %** | 462 observations |",
          "| **never answering at all** - the free baseline it has to beat | **73.8 %** | the same 462 |",
          "| the same detector on held-out control clips | **9.1 %** | 44 observations |",
          "",
          "It ships **off**, with `PRIORIDAD = False` and the number next to the switch. **A detector "
          "that is beaten by declining to answer is not a detector.**", "",
          "The free baseline is 73.8 % for a plain reason: **12 of the 21 annotated windows are "
          "*nobody* from start to finish**. Answering nothing scores that without knowing anything, "
          "and anything that does fire pays for it there.", "",
          "Split in two, the failure has a shape - and it is the shape that says **no threshold sweep "
          "will fix this**:", "",
          "| | detector | the crude box-motion baseline | answering nothing |",
          "|---|---:|---:|---:|",
          "| frames that **do** have an owner (n = 121) | **25.6 %** | 41.3 % | 0 % |",
          "| frames with **no** owner (n = 341) | **85.3 %** | 32.3 % | **100 %** |",
          "",
          "It is beaten on **both halves of the job** - worse than crude box motion when it has to "
          "name someone, worse than silence when there is nobody to name. There is no operating point "
          "where it wins.", "",
          "The one encouraging number, stated precisely: when the detector **does** commit and the "
          "annotation also names an owner, **the side it picks is right 86.1 %** of the time (n = 36). "
          "But it only commits on **36 of those 121 frames**. *The side is good; the coverage is "
          "bad.* And on the held-out clips it does not answer at all: **0.0 %** of the held-out frames "
          "that have an owner get one.", "",
          "Two caveats on those figures, because they are the kind that get quietly rounded away. The "
          "462 are **observations, not distinct frames**: two annotation windows overlap on 7 frames, "
          "so there are 455 distinct ones, and both copies carry identical truth and output so "
          "nothing is biased - but 462 is not 455. And a re-run today gave **70.35 %** on the same "
          "462, with 22 decisions differing and two clips moving in opposite directions. The verdict "
          "does not move: **both numbers lose to 73.8 %**.", "",
          "### Why the scoring lights are not the answer either", "",
          "| | value | measured over | unit |",
          "|---|---:|---|---|",
          "| *whoever lights up second wins* | 64.96 % | 1,815 clips | **who won the touch** |",
          "| in one-light clips, *whoever lights up wins* | 96.19 % | one-light clips | **who won the touch** |",
          "",
          "**These are a different question on a different unit and they do not compare to the 69.7 % "
          "above.** That row answers *who won the touch*, per clip; the detector answers *who holds "
          "right of way*, per frame. Putting them in one table - which an earlier version of this "
          "page did - invents a comparison that was never measured.",
          "",
          "What they do show is that the lights are a **shortcut, not a solution**: in the clips where "
          "*second light wins* is false, a model leaning on it falls **below chance**, and one-light "
          "clips teach nothing about right of way at all. The audio beep was tried as a light "
          "detector and **failed**: **50.6 %** of the corpus has no audio track, and where it does, "
          "several pistes beep in the same hall.", ""]
    return "\n".join(l)


def seccion_arbitro():
    return "\n".join([
        "### The open defect: sometimes one of the two fencers is the referee", "",
        "In about **6 % of clips the pipeline delivers the REFEREE as one of the two fencers.** Four "
        "detection routes were tried and **all four failed, each with a number**:", "",
        "| route | why it failed |",
        "|---|---|",
        "| uniform whiteness | the gold **lame** fires it on correct pairs; paired separation came "
        "out at **-0.002**, i.e. backwards |",
        "| mask held in hand | with a loose threshold it becomes a darkness detector: fires on 469 "
        "of 3,386 patches while only 1 of 39 fencers holds a mask |",
        "| geometry | the referee passed all three geometric gates **without touching them** "
        "(separation 1.93 of 3.20, foot delta 0.46 of 0.55, feet-in-band 0.894 of 0.40) - and it was "
        "predictable, because a referee walks beside the piste |",
        "| movement behaviour | the referee scored **0.524** events per second, **above the p10 of "
        "real fencers (0.493)** and four times its own opponent's rate: walking arm-swing fires the "
        "arm-extension event |",
        "",
        "Every row carries a `side_valid` flag and it is a **lower bound**. The reason this is hard "
        "is worth stating: **the referee's pose is perfectly good.** There is nothing wrong with the "
        "pose; what is wrong is *whose* it is. No pose-quality metric will ever catch it.", ""])


def tabla_campos():
    """QUE HACE: la tabla de campos, con su forma REAL leida del disco y su error medido.

    La forma se lee de TODOS los NPZ y no de una lista escrita a mano: asi, si se anade o se quita un
    campo, la ficha lo dice en vez de quedarse mintiendo. Con un solo fichero, dos campos salian con
    forma `-` porque se aniadieron despues y los NPZ viejos no los llevan; ahora se cuenta la
    presencia y se escribe al lado.
    """
    import csv
    import numpy as np
    fp = []
    for p in sorted((PROY / "01_extraccion" / "pruebas").glob("poses_*.npz")):
        with np.load(p, allow_pickle=True) as zz:
            if {"KP", "SC", "origen", "B", "mask"} <= set(zz.files):
                fp.append(p)
    """SE COMPRUEBA EL ESQUEMA, NO SE CUENTAN FICHEROS POR EL NOMBRE.

    Hay **39** ficheros que encajan con `poses_*.npz` y solo **38** son poses entregadas:
    `poses_SentimentalLargeBedlingtonterrier.npz` lleva el esquema viejo `(cuadro, kp, lado, sc)` y no
    tiene ni `origen` ni `mask`. Contar por el patron del nombre publicaba **39 clips con poses**, que
    es uno de mas, y encima el primero de la lista podia ser justo ese y dejar la tabla de formas
    entera equivocada. El patron de un nombre no es un contrato; las claves si."""
    n_ev, n_cl = 0, 0
    pev = VAL / "pista" / "serie_eventos.csv"
    if pev.exists():
        with open(pev, newline="", encoding="utf-8", errors="replace") as fh:
            filas = list(csv.DictReader(fh))
        n_ev = len(filas)
        n_cl = len(set(r.get("clip") for r in filas))
    """SE CUENTA CON `csv` Y NO CON PANDAS: este script corre con el interprete que tiene
    `huggingface_hub`, y ese es el unico que NO tiene pandas. Envuelto en un `try`, el recuento se
    caia al `except` y la ficha publicaba **0 eventos sobre 0 clips**, una cifra falsa en un documento
    publico. Contar filas no necesita pandas."""

    POSES = [
        ("KP", "the 308 body keypoints of both fencers, in pixels", "px, float32",
         "**2.26 px** torso error at 384x288 instead of 1024 (7.8x cheaper). **96.0 %** of delivered "
         "poses are MEASURED, not filled in"),
        ("SC", "per-keypoint confidence", "0-1, float32",
         "used with a 0.3 threshold. Points that fell outside their own person's box are zeroed: "
         "there were **10,140** of them and they sat on the other fencer"),
        ("origen", "how the pose was obtained: 0 measured, 1 interpolated, 2 held, -1 none",
         "code, int8",
         "**0 held poses** in what is delivered. Without this field the data cannot be weighted, and "
         "it **cannot be recovered later**"),
        ("hueco", "length in frames of the gap an invented pose comes from", "frames, int16",
         "**this row used to quote 2.2 % of torso length for any gap <= 24 frames, and that was "
         "wrong**: 2.2 % is the cost of a **1-frame** gap. Re-measured over the gaps that actually "
         "get filled, the whole <= 24 band costs **6.33 %** (n = 271), and a 24-frame gap costs "
         "**25.7 %**. Beyond 24 frames **there is no measurement at all**"),
        ("B", "each fencer's box, normalised 0-1, xyxy", "fraction of frame, float32",
         "no delivered frame jumps more than **0.60 box widths** (worst measured 0.563). No box is a "
         "border sliver, and none comes from a track that was never seen whole"),
        ("Be", "ESTIMATED box: where the fencer should be in a frame the detector missed",
         "fraction of frame, float32",
         "used only to aim a crop re-detection; its own error is not measured separately"),
        ("pres", "whether that fencer's pose in that frame is MEASURED", "bool",
         "**0** delivered frames without both fencers, across 28 clips"),
        ("mask", "usable frames: close-ups, replays and cross-fades removed", "bool",
         "a cross-fade is told from a zoom by the residual relative to contrast: **0.219 vs 0.062**"),
        ("filtro_vacias", "measured poses dropped for coming out empty", "count, int32",
         "a pose with fewer than **20 useful body points out of 63** cannot serve as an anchor"),
        ("filtro_fuera_de_caja", "keypoints zeroed for falling outside their own box",
         "count, int32", "**10,140** across 28 clips"),
        ("recortados_por_pose", "frames trimmed off the ends for not having both poses",
         "count, int32", "**80** across 22 clips, which is what leaves **0 held poses**"),
        ("W", "video width", "px, int", "**81 %** of the corpus is 1280x720"),
        ("H", "video height", "px, int", "18 % is 1920x1080, 0.8 % is 854x480"),
        ("T", "frames in the clip", "frames, int",
         "median **307**. The previous pipeline stored a **129-frame window**, not the whole clip"),
    ]
    EVENTOS = [
        ("clip / side", "which clip and which fencer (0 = image left, 1 = image right)",
         "str / int", "side 0/1 is an image convention, not left/right-handedness"),
        ("type", "march, retreat, lunge, halt, arm extension, action, jump", "category",
         "confusion matrix against gestures annotated by eye: **5 of 6**, with **`march -> lunge` = "
         "0**. `jump` ships with **0 events**: not detectable here, and a ballistic assertion "
         "enforces it"),
        ("index_of_type", "the event's number within its own type and side", "int",
         "this is what makes *the third march of the right-hand fencer* a one-line query, and it is "
         "**the same number the video counter shows**"),
        ("frame_start / frame_end", "when the event starts and ends", "frame, int",
         "detection lag measured at **+0.5 frames (+17 ms)** over 62 pairings, with no bias. The "
         "**boundary error is still unmeasured**: that is what is being annotated now"),
        ("magnitude_m", "how far the fencer moved during the event", "m, float",
         "march p50 **0.346 m**, lunge p50 **0.565 m**. Depends on the piste scale, which is "
         "**not verified** along the piste"),
        ("x_start_m / x_end_m", "position along the piste", "m from the centre line, float",
         "the **origin is measured** from the t.19 colour frontiers in 13 of 22 clips, and a held-out "
         "landmark is predicted to **p50 5 cm**. The **scale** along the piste is **not verified**. "
         "Per-frame tremor p95 **0.025 m**, against 0.13-0.20 m/frame for a real lunge. Convention: "
         "**x grows towards image right**"),
        ("y_start_m", "position across the width of the piste", "m, float",
         "metres between the two **detected** edges, bounded by construction, so no fencer is pinned "
         "to an edge. But the physical control (travel <= 0.60 m) passes for **11 of 40 "
         "fencer-clips**, and in only **3 of 22 clips** do both fencers pass it"),
        ("no_homography", "whether this clip has no metric scale at all", "bool",
         "**6 in 10 clips** have none. When true the metric fields come out **None, not zero**"),
        ("razon_pie_atras", "rear-foot displacement over front-foot displacement", "ratio, float",
         "**the** march/lunge discriminator: boundary 0.45, measured in the extension phase and not "
         "over the whole window"),
        ("worst_pose_provenance", "the worst pose provenance any frame of the event rests on",
         "code, int", "an event resting on an interpolated pose is weaker, and here you can see it"),
        ("gap_beyond_measured", "whether the event uses a pose from a gap longer than 24 frames",
         "bool", "flags data whose error is **not** verifiable"),
        ("side_valid", "whether that side really is a fencer", "bool",
         "in **6.1 %** of clips the delivered side is the **REFEREE**, and **all four** detection "
         "routes tried have failed. This flag is a **lower bound**"),
        ("weapon_hand", "which hand holds the weapon: A (root 21) or B (root 42)", "category",
         "the **cascade** scores **1.000 over 19 of 19 fencers** of a held-out set that played no "
         "part in choosing the rule, and 0.949 over all 59 with ground truth. Its front-foot rung "
         "alone covers **12 of those 19, with no failure**. What cannot be decided is flagged "
         "`uncertain` instead of guessed"),
        ("campo_mu_gr / campo_sigma_gr / campo_origen",
         "the blade probability field: centre direction, width, and where it came from",
         "deg, deg, code",
         "`origen 0` = the blade was **SEEN**, so the centre is the **measured** blade angle and the "
         "half-angle is **8.6 deg**; `origen 1` = the centre is the **assumed** forearm direction "
         "and the half-angle is **33.0 deg**. Both are the p90 of a measured residual, so seeing the "
         "blade narrows the field **3.8x** and centre and width are one decision. `temblor_hoja_gr` "
         "(4.2 deg) is a different quantity and ships separately: the frame-to-frame uncertainty of "
         "the blade's **direction**, not the width of the field. They never merge: **a measured 1 and "
         "an assumed 1 are not worth the same**"),
    ]
    CANALES = [
        ("pose", "308 keypoints per fencer", "(T, 2, 308, 2) float32 + (T, 2, 308) confidence",
         "the index map is MEASURED: 5-6 shoulders, 7-8 elbows, 9-10 hips, 11-12 knees, 13-14 "
         "ankles, 15-20 feet, 21-41 hand A, 42-62 hand B, 63+ face. **245 of the 308 are face** and "
         "the project discards them, because fencers wear a mask"),
        ("seg", "body-part segmentation of each fencer", "crop-sized uint8 class map",
         "referee leakage into the mask measured **0.0000**. In the layer videos the channel is "
         "blended **inside the mask only** and the frame outside it is merely darkened, so you can "
         "check by eye that the mask landed on the right person and that nothing around it moved"),
        ("normal", "surface normals", "crop-sized float16, 3 channels, NaN outside the mask",
         "one axis per colour channel; the dense layer that **does** hold up for local geometry"),
        ("matting", "alpha, the soft silhouette", "crop-sized float16, 0 outside the mask",
         "at very low alpha the **edge of the blade** appears, which is what the foil detector "
         "uses. A near-zero mean alpha means matting returned background for that crop"),
        ("pointmap", "the model's 3D point map, z channel", "crop-sized float16, NaN outside",
         "**kept but not used**: it orders the two fencers correctly in **76 %** of frames, while "
         "the y coordinate of the free foot, which costs nothing, gets it right **28 of 28**. It "
         "ships because it was asked for, not because it works"),
        ("foil / campo", "the threat cone as a graded probability field", "see `campo_*` above",
         "the fill **is** the weight function the measurement itself uses, so the figure and the "
         "number cannot disagree. The grey dashes are the forearm and are drawn even when the field "
         "is centred on the blade, so you can see how far the field moved off the axis. A dashed "
         "core arc plus `UNCERTAIN` means the weapon-hand cascade could not settle that fencer, and "
         "the front-foot vote is printed next to it"),
    ]
    """LA TABLA DE CANALES SE LLEVO EL DETALLE QUE ESTABA EN EL PIE DE LOS VIDEOS. Esos pies tenian
    hasta cuatro lineas de 115-121 caracteres quemadas encima de la imagen, que es el peor sitio
    posible para leer una cifra; ahora cada capa lleva UNA linea mirando-el-cuadro y el contexto vive
    aqui, que es donde se puede leer y citar. El movimiento fue del pie al README, no una supresion:
    si algo de aqui desaparece, la cifra se pierde de los dos sitios."""
    formas, presencia = {}, {}
    for p in fp:
        with np.load(p, allow_pickle=True) as zz:
            for k in zz.files:
                presencia[k] = presencia.get(k, 0) + 1
                formas.setdefault(k, str(getattr(zz[k], "shape", "()")) or "scalar")
    for k in list(formas):
        if presencia[k] < len(fp):
            formas[k] = "%s, in %d of %d files" % (formas[k], presencia[k], len(fp))

    l = ["### 1 - What comes out per clip, and in what shape", "",
         "`T` is the number of frames in the clip; the `2` is the two fencers, side 0 on the image "
         "left and side 1 on the image right.", "",
         "| field | shape on disk | what it is | format | measured precision |",
         "|---|---|---|---|---|"]
    for k, q, u, e in POSES:
        l.append("| `%s` | `%s` | %s | %s | %s |" % (k, formas.get(k, "-"), q, u, e))
    l += ["", "### 2 - The five model channels kept per fencer", "",
          "| channel | what it is | format | why it is there, measured |", "|---|---|---|---|"]
    for k, q, u, e in CANALES:
        l.append("| `%s` | %s | %s | %s |" % (k, q, u, e))
    l += ["", "### 3 - The movement events: one row per event, not per frame", "",
          "**%d events over %d clips** so far. Not published here as a file - the schema below is the "
          "contract, and the rows ship in batches of %d." % (n_ev, n_cl, LOTE), "",
          "| column | what it is | format | measured precision |", "|---|---|---|---|"]
    for k, q, u, e in EVENTOS:
        l.append("| `%s` | %s | %s | %s |" % (k, q, u, e))
    l.append("")
    return "\n".join(l), n_ev, n_cl, len(fp)


PLANTILLA = pathlib.Path(__file__).parent / "FICHA_PUBLICA.md"


def ficha(tabla, n_ev, n_clips, n_poses):
    """QUE HACE: rellena `FICHA_PUBLICA.md`. La prosa vive ALLI, no aqui.

    POR QUE SE PARTIO EN DOS. El usuario dijo: *"veo una perdida de tiempo que cada cambio tenga que
    ejecutar un .py en vez de modificar el readme"*. Tiene razon para el texto: una frase no deberia
    obligar a editar una cadena de Python entre `%s`.

    Y no la tiene para las cifras, que es la otra mitad. La forma de cada campo se lee de los 39 NPZ,
    el recuento de eventos del CSV y la lista de videos de la carpeta que se sube. Escritas a mano, se
    quedan mintiendo en silencio en cuanto cambia un fichero; este proyecto ya publico «0 eventos
    sobre 0 clips» porque un recuento se cayo a un `except`.

    Asi que la plantilla lleva marcas `{{...}}` y esta funcion las sustituye. Si alguien aniade una
    marca nueva y se olvida de darle valor, **se queda visible en la pagina** en vez de desaparecer,
    que es el fallo que se prefiere: un `{{N_POSES}}` a la vista se arregla en un minuto y un numero
    viejo puede durar meses.
    """
    if not PLANTILLA.exists():
        raise SystemExit("falta la plantilla %s" % PLANTILLA)
    txt = PLANTILLA.read_text(encoding="utf-8")
    corte = txt.find("-->")
    if corte >= 0:
        txt = txt[corte + 3:].lstrip("\n")
        """El comentario HTML de cabecera es para quien edite el fichero, no para quien lo lea
        publicado, asi que se quita. Y se quita ANTES de la cabecera YAML de Hugging Face: si el
        fichero no empieza exactamente por `---`, Hugging Face no interpreta la cabecera y
        `viewer: false` deja de tener efecto."""
    valores = {
        "N_VIDEOS": str(len(medios_de("*.mp4"))),
        "N_POSES": str(n_poses),
        "N_EVENTOS": "{:,}".format(n_ev),
        "N_CLIPS_EV": str(n_clips),
        "LOTE": str(LOTE),
        "REPO_PRIV": SE.REPO,
        "GITHUB": GITHUB,
        "PORTADA": seccion_portada(),
        "DETECCION": seccion_deteccion(),
        "CAPAS": seccion_capas(),
        "FLORETE": seccion_florete(),
        "PISTA": seccion_pista(),
        "EVENTOS": seccion_eventos(),
        "MANO": seccion_mano(),
        "ARBITRO": seccion_arbitro(),
        "TABLA": tabla,
    }
    for k, v in valores.items():
        txt = txt.replace("{{%s}}" % k, v)
    quedan = [m for m in ("{{",) if m in txt]
    if quedan:
        import re
        print("AVISO: marcas sin rellenar en la ficha: %s"
              % sorted(set(re.findall(r"\{\{[A-Z_]+\}\}", txt))))
    return txt



def junta_medios():
    """QUE HACE: rellena `medios/` con lo que se va a subir, y devuelve la lista.

    Se COPIA a una carpeta aparte en vez de subir desde donde esta cada cosa porque asi la lista de
    lo que se publica es una carpeta que se puede mirar. Un `upload_folder` sobre el arbol de trabajo
    es como se sube sin querer algo que no toca.
    """
    MEDIOS.mkdir(parents=True, exist_ok=True)
    fs = []

    def mete(p, nombre=None):
        d = MEDIOS / (nombre or p.name)
        shutil.copy2(p, d)
        fs.append(d)

    for p in sorted(GEN.glob("*.mp4")):
        mete(p)
    for p in sorted((GEN / "fijos").glob("*.png")):
        mete(p)
    mano = VAL / "muestras" / "mano_dominante" / "resumen_mano_armada_generado.png"
    if mano.exists():
        mete(mano, "weapon_hand.png")
    casc = VAL / "muestras" / "cascada_deteccion"
    if (casc / "cascada_deteccion.png").exists():
        mete(casc / "cascada_deteccion.png")
    for q in sorted(casc.glob("redeteccion_*.png"))[:2]:
        mete(q)
    """LA FIGURA DE LA CASCADA Y DOS EJEMPLOS DE REDETECCION. El resumen esta dibujado **sobre
    negro** y no lleva un pixel de emision; los dos ejemplos si llevan el fotograma, como el resto de
    los videos de capa de esta ficha."""
    for nom in GIFS_PUBLICOS:
        g = VAL / "previas" / nom
        if g.exists():
            mete(g)
        else:
            print("   AVISO: falta el GIF %s, la ficha lo referenciara y no estara" % nom)
    """LOS GIF TAMBIEN SE SUBEN, Y NO SON UN DUPLICADO DEL MP4.

    El `<video>` con URL de Hugging Face se reproduce en la ficha de HF, pero **el README de GitHub
    no renderiza `<video src>` de un origen externo**: alli la pagina se veia sin una sola imagen en
    movimiento, y de eso se quejo el usuario. El GIF va delante del `<video>` para que cada
    renderizador ensene lo suyo, y el MP4 sigue enlazado para verlo entero y con calidad.

    Salen de `previas.py`, que los saca de estos MISMOS MP4, asi que no pueden discrepar."""

    for p in sorted(q for q in (VAL / "pista" / "video").glob("trace_*.mp4")
                    if "_public" not in q.name):
        """LOS QUE ENSENIAN EL CLIP, Y ANTES ERA AL REVES (2026-09-23).

        Esto subia **solo los `_public`**, que son los que `video_pista.py --publico` dibuja sobre
        negro, porque la regla era que al publico no saliera ni un fotograma de la emision. **El
        usuario reverso esa regla** y `panel_publico.py` lo dice en su primera linea. Las capas ya
        ensenaban el clip y el video de la pista se habia quedado atras: la ficha publica se
        contradecia a si misma, y era lo que se veia **negro arriba a la izquierda** del README.

        Los `_public` se siguen generando y siguen sirviendo -- el esqueleto sobre negro ensena
        mejor lo que se mide -- pero ya no son lo que se publica."""
        mete(p)
    """LOS MEDIOS DE PRIORIDAD **NO SE SUBEN AL REPO PUBLICO**. Antes esto metia los 4 PNG y el MP4
    de `prioridad/publico/` como `prio_*`, y la seccion de right of way los incrustaba.

    Decision del usuario: *"quita esa seccion que no me gusta que salga"*. Asi que se va la seccion y
    se van sus medios: dejar los ficheros subidos sin nada que los referencie es justo lo que este
    script comprueba y avisa en `main`, y ademas seria material publicado que nadie explica.

    **El defecto no se esconde:** sigue en la tabla de capacidades de la ficha con su cifra (69,7 %
    contra 73,8 % de no contestar), y toda la evidencia -- las figuras, el video y las dos
    anotaciones -- sigue en el repo PRIVADO via `sube_evidencia`/`curado_hf`, donde ademas ahora se
    ve como GIF. Fuera del escaparate, dentro del trabajo."""
    return fs


def main():
    from huggingface_hub import HfApi
    api = HfApi(token=SE._token())
    info = api.repo_info(repo_id=PUB, repo_type=TIPO)
    if info.private:
        raise SystemExit("%s es PRIVADO; este script es para el publico" % PUB)
    print("repo %s, publico" % PUB)

    if MEDIOS.exists():
        shutil.rmtree(MEDIOS, ignore_errors=True)
    fs = junta_medios()
    tot = sum(f.stat().st_size for f in fs)
    print("medios reunidos: %d ficheros, %.1f MB  (%d mp4, %d png)"
          % (len(fs), tot / 1e6, len(medios_de("*.mp4")), len(medios_de("*.png"))))

    tabla, n_ev, n_clips, n_poses = tabla_campos()
    texto = ficha(tabla, n_ev, n_clips, n_poses)
    (GEN / "README.md").write_text(texto, encoding="utf-8")
    print("ficha: %d caracteres, %d videos incrustados"
          % (len(texto), texto.count("<video")))

    api.upload_folder(folder_path=str(MEDIOS), path_in_repo="media", repo_id=PUB, repo_type=TIPO,
                      commit_message="videos por capa en h264, cono nuevo, pista y prioridad")
    api.upload_file(path_or_fileobj=texto.encode("utf-8"), path_in_repo="README.md",
                    repo_id=PUB, repo_type=TIPO,
                    commit_message="ficha: que se extrae, con que precision, y los videos")

    todos = api.list_repo_files(repo_id=PUB, repo_type=TIPO)
    sobran = [f for f in todos if not (f in SE_QUEDAN or f.startswith("media/"))]
    vivos = set("media/" + f.name for f in fs) | set(SE_QUEDAN)
    viejos = [f for f in todos if f.startswith("media/") and f not in vivos]
    """LOS MEDIOS VIEJOS TAMBIEN SE BORRAN, y hace falta decirlo porque `upload_folder` **no borra**:
    solo aniade y sobreescribe. Sin esto, los `extracted_layers_*.mp4` de la tanda anterior -que van
    en `mp4v` y no se reproducen en un navegador, y que pintan el cono viejo- se quedarian en el repo
    al lado de los nuevos, y un enlace roto o un video que no arranca es peor que no tenerlo."""
    fuera = sobran + viejos
    print()
    print("se quitan %d ficheros (%d ajenos a la ficha, %d medios de tandas anteriores):"
          % (len(fuera), len(sobran), len(viejos)))
    for s in fuera:
        print("   %s" % s)
    if fuera:
        api.delete_files(repo_id=PUB, repo_type=TIPO, delete_patterns=fuera,
                         commit_message="dejar solo la ficha y los medios de esta tanda")
    print()
    print("queda %d ficheros" % len(api.list_repo_files(repo_id=PUB, repo_type=TIPO)))
    print("   %s" % URL)
    print()
    print("`viewer: false` y sin ningun CSV: el visor de datos no puede aparecer.")


if __name__ == "__main__":
    main()

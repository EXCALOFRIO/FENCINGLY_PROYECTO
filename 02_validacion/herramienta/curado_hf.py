"""Deja en Hugging Face SOLO los entregables finales, y escribe el README con enlaces directos.

LO QUE PIDIO EL USUARIO, literal: "en Hugging Face hay demasiadas evidencias, quiero solo con las
finales reales que pido yo, porque hay demasiadas. Y ponlas en el README para verlas desde ahi bien".

Tenia razon: el repo llego a **902 ficheros**, y la mayoria eran material de trabajo (diagnosticos de
homografia, laminas de anotacion, recortes de casos al limite, fotogramas sueltos de los mosaicos).
Todo eso sirvio para llegar a las conclusiones, pero no es lo que se revisa.

## QUE SE QUEDA Y POR QUE

Se queda **lo que el usuario pidio ver**, una carpeta por cosa, y nada mas:

  las poses limpias sin intrusos, en video
  los ejemplos de cuando se inventa una pose
  las cuatro capas densas antes y despues de limpiarlas
  el cono del florete antes y despues
  el trazado sobre la pista en video, con los eventos
  la prioridad en video
  la mano dominante
  el mosaico de seis celdas
  y los tres documentos con la tabla de checks, el inventario y los resultados

## LO QUE SE BORRA NO SE PIERDE

Todo lo que se quita del repo **sigue en disco** en `02_validacion/muestras/` y
`02_validacion/pista/diagnostico/`, y se puede volver a subir con `sube_evidencia.py` en cualquier
momento. Lo que se borra son figuras DERIVADAS, reproducibles corriendo el script que las genero. No
se borra ningun dato de origen: ni videos, ni NPZ, ni pickles.

Y de los casos al limite que destaparon las dos averias de intruso se conserva una **seleccion** en
`evidencia/errores_encontrados/`, porque esa evidencia es la que sostiene una afirmacion del informe y
no se puede quedar sin respaldo.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sube_evidencia as SE

VAL = SE.VAL
URL = "https://huggingface.co/datasets/%s" % SE.REPO

CARPETAS_FINALES = [
    "documentacion",
    "evidencia/videos_revision",
    "evidencia/poses_inventadas",
    "evidencia/poses_tres_rondas",
    "evidencia/cascada_deteccion",
    "evidencia/capas_limpias",
    "evidencia/cono_arreglado",
    "evidencia/mano_dominante",
    "evidencia/errores_encontrados",
]
"""Carpetas que se conservan enteras. Las de `pista`, `prioridad` y `mosaicos` se conservan
PARCIALMENTE y se filtran por patron mas abajo, porque tienen dentro el entregable y el material de
trabajo mezclados."""

PARCIALES = {
    "evidencia/pista": lambda f: (f.endswith(".mp4") or f.endswith("LEEME.md")
                                  or "/pista_" in f or f.endswith("serie_eventos.csv")
                                  or f.endswith("posicion_y_eventos.csv")
                                  or "/anotar/" in f or f.endswith("CONVENIO_FRONTERAS.md")
                                  or f.endswith("CONVENIO_CONTEO_PASOS.md")),
    "evidencia/prioridad": lambda f: (f.endswith(".mp4") or f.endswith(".md")
                                      or f.endswith("anotacion_a_mano.csv")
                                      or f.endswith("anotacion_holdout.csv")),
    "evidencia/mosaicos": lambda f: f.endswith(".mp4"),
}

VIDEOS_QUE_SE_QUEDAN = [
    "evidencia/pista/video/trace_ForcefulAntiqueAlbacoretuna_slow.mp4",
    "evidencia/pista/video/trace_UglyNiceAplomadofalcon_slow.mp4",
    "evidencia/videos_revision/ZanyMeagerIndianrockpython.mp4",
    "evidencia/videos_revision/VainAchingGonolek.mp4",
    "evidencia/prioridad/videos/GentleQuerulousElephantseal.mp4",
    "evidencia/mosaicos/mosaico_seis_AromaticShockedGermanspitz.mp4",
]
"""Los SEIS videos que se quedan, de 34 que habia. El usuario pidio borrar los demas para que no
consuman espacio, y con las vistas previas en el README el video completo solo hace falta para
mirar un caso a fondo, no para saber que hay.

POR QUE ESTOS SEIS, y no los primeros de la lista:

  los dos `_lento` de la pista son los que llevan el **contador** y la camara lenta, y son un par
  deliberado: `Forceful` es el clip que mas eventos detecta y `UglyNice` el que menos, que es la
  comparacion que pidio el usuario para juzgar si cuenta bien
  `ZanyMeager` y `VainAching` son los dos clips de poses **verificados a ojo** en el proyecto, y
  `VainAching` ademas es el del fundido encadenado de la seccion 14
  el de prioridad y el del mosaico son uno de cada cosa, para ver el formato

Los 28 que se quitan **siguen en disco** y se regeneran con su script."""

SELECCION_ERRORES = [
    (VAL / "muestras" / "cruce_lados" / "cruce_c0160.png",
     "evidencia/errores_encontrados/01_el_arbitro_elegido_como_tirador_c160.png"),
    (VAL / "muestras" / "cruce_lados" / "cruce_c0224.png",
     "evidencia/errores_encontrados/02_el_arbitro_elegido_como_tirador_c224.png"),
    (VAL / "muestras" / "limite_intruso" / "LavishDistantDikkops_c0163.png",
     "evidencia/errores_encontrados/03_trozo_de_espectador_como_tirador_ARREGLADO.png"),
    (VAL / "muestras" / "limite_intruso" / "proudDungBeetleBoglenTerri_c0227.png",
     "evidencia/errores_encontrados/04_correcto_el_contraste_lo_da_el_lame.png"),
    (VAL / "muestras" / "limite_intruso" / "SimilarOffbeatIberianlynx_c0381.png",
     "evidencia/errores_encontrados/05_correcto_dos_tiradores_que_se_cruzan_de_verdad.png"),
    (VAL / "muestras" / "limite_intruso" / "WastefulYawningAmericanqua_c0239.png",
     "evidencia/errores_encontrados/06_correcto_tirador_medio_fuera_de_cuadro.png"),
]
"""Seis imagenes, no cien. Las dos primeras son el arbitro elegido como tirador, la tercera el trozo
de espectador ya arreglado, y **las tres ultimas son casos CORRECTOS** que cada regla candidata se
llevaba de mas. Estan aqui a proposito: son las que impidieron convertir la blancura en una puerta
automatica, y una afirmacion como esa no puede quedarse sin su contraejemplo."""


PREVIA = {
    "evidencia/pista": ("pista_trazado.gif", "pista_mapa.gif"),
    "evidencia/videos_revision": "poses.gif",
    "evidencia/mosaicos": "mosaico_seis.gif",
    "evidencia/poses_inventadas": "poses_inventadas.png",
    "evidencia/capas_limpias": "capas_limpias.png",
    "evidencia/cono_arreglado": "cono_arreglado.png",
    "evidencia/errores_encontrados": "error_arbitro.png",
    "evidencia/mano_dominante": "mano_armada.png",
    "evidencia/prioridad": "prioridad.gif",
    "evidencia/poses_tres_rondas": "tres_rondas.png",
    "evidencia/cascada_deteccion": "cascada_deteccion.png",
}
"""Que previa incrustar en cada entregable. Las genera `previas.py` y se suben a `previa/` en el
repo.

`evidencia/prioridad` pasa de una FIGURA a un GIF del video: los 8,6 s de
`video_right_of_way_*.mp4` son pixeles generados sobre negro, asi que se pueden ensenar, y un cuadro
fijo de un detector que decide por cuadro no deja ver lo unico interesante, que es **cuando cambia de
dueno y cuando se calla**. Esto es el repo PRIVADO de evidencia; la seccion de right of way se ha
quitado de la ficha publica por decision del usuario, y las dos cosas no se contradicen: aqui es
donde el defecto se documenta, alla es el escaparate."""
PREVIAS_LOCAL = VAL / "previas"


def readme(inv, ficheros):
    """QUE HACE: escribe la ficha del repo, EN INGLES.

    El usuario lo pidio dos veces: «pon todo el readme en ingles porfa» y «quiero que sean ingles
    siempre para que lo entienda todo el mundo». Asi que todo lo que se LEE desde fuera esta en
    ingles. Los docstrings y el documento maestro se quedan en espaniol: son internos, y traducirlos
    ahora solo meteria errores.
    """
    def enl(c):
        return "%s/tree/main/%s" % (URL, c)

    l = ["---", "license: other", "pretty_name: FENCINGLY extraction evidence", "---", "",
         "# FENCINGLY - extraction evidence", "",
         "**Private repo.** It contains frames and clips from competition showing **identifiable "
         "athletes** with visible faces, so it is not made public and is not shared outside the "
         "project.", "",
         "This holds **only the final deliverables**. The working material (homography diagnostics, "
         "annotation sheets, crops of borderline cases, single frames) is what led to the "
         "conclusions, but it is not what you review, and it stays on disk.", "",
         "## Where to start", "",
         "| document | what it answers |", "|---|---|",
         "| [`RESULTADOS.md`](%s/blob/main/documentacion/RESULTADOS.md) | **what works and what does "
         "not**, per capability, with its figure and its source. Start here. |" % URL,
         "| [`DATOS_POR_CLIP.md`](%s/blob/main/documentacion/DATOS_POR_CLIP.md) | **everything we "
         "extract from each clip**, field by field, with its real shape, its unit, whether it is "
         "verified and **its measured error**. This is the spec of what will be used for "
         "training. |" % URL,
         "| [`CHECKS.md`](%s/blob/main/documentacion/CHECKS.md) | the **guarantees**: what the "
         "pipeline claims to satisfy, each with a pre-declared threshold and a verdict. |" % URL,
         "| [`DONDE_ESTA_TODO.md`](%s/blob/main/documentacion/DONDE_ESTA_TODO.md) | the full "
         "inventory, including what only exists on disk. |" % URL,
         "| [`00_EMPIEZA_AQUI.md`](%s/blob/main/documentacion/00_EMPIEZA_AQUI.md) | the master "
         "document: every measurement and every defect. It is long, and in Spanish. |" % URL, "",
         "**Twelve green checks do not mean the product works.** They mean that what is delivered "
         "does not contradict itself. What works and what does not is in `RESULTADOS.md` - and "
         "several things do not.", "",
         "## Fencing terms used throughout", "",
         "| term | meaning |", "|---|---|",
         "| **march** / **retreat** | a step forward / a step backward |",
         "| **lunge** | the explosive attack: the front foot flies and **the rear foot stays "
         "planted**. That binary property, not the distance, is what separates it from a march |",
         "| **halt** | the fencer stops moving. **This is NOT a parry.** The fencing parry "
         "(deflecting the opponent blade) is **not implemented** |",
         "| **right of way** | which fencer has priority for the touch. Foil is a weapon of "
         "convention: the valid touch is not simply the first to land |",
         "| **valid target** | in foil, the torso. A **white light** means the tip landed "
         "**off-target** |",
         "| **on-guard** | the basic stance. The **front foot is on the same side as the weapon "
         "arm**, which is what the weapon-hand detector exploits |", "",
         "## What to look at, one folder per thing", ""]
    filas = [
        ("evidencia/pista", "video",
         "**The trace over the piste**, like a racing line, with both fencers position in metres and "
         "the events appearing as they happen: march, retreat, lunge, halt, arm extension. A counter "
         "per fencer ticks up **on the exact frame** of each event, and the number is the same index "
         "as in the CSV, so the video verifies the data. If the caption is red, the metric scale of "
         "that clip is **not reliable** and it says so. **Jumps are reported as 0 on purpose**: "
         "measured as not detectable, and a ballistic assertion enforces it."),
        ("evidencia/videos_revision", "video",
         "**Both 308-point poses drawn over the clip.** Green is the left fencer, orange the right, "
         "**violet and labelled** are the invented poses, and a red band marks discarded frames. This "
         "is what you use to check whether an intruder slipped in."),
        ("evidencia/poses_inventadas", "imagen",
         "**When a pose is invented, in three panels**: measured anchor before, the invented pose in "
         "violet over the real pixels, measured anchor after. The caption gives the gap length; the "
         "ones in **red** are beyond the range where the error was measured, so their error is **not "
         "verifiable**."),
        ("evidencia/capas_limpias", "imagen",
         "**The four Sapiens 2 dense layers before and after cleaning**, in pairs: segmentation, "
         "normals, pointmap and matting. The referee and the crowd disappear (referee leakage "
         "0.0000). The `pointmap` is shown because it was asked for, but it carries the caption that "
         "it **is not usable as depth**."),
        ("evidencia/cono_arreglado", "imagen",
         "**The foil threat cone, before and after**, on the same frame. What looked like a straight "
         "foil **was the arm**, drawn in the cone colour. Length is **1.10 m = wrist-to-tip reach** "
         "(FIE m.8.4 + m.7), not the 0.90 m of the blade alone. The aperture is a **half-angle**: "
         "+-33 deg on the assumed forearm direction, i.e. 66 deg of full cone, and **+-8.6 deg "
         "centred on the blade itself where the blade was measured**. And the blade is actually "
         "**seen in only 3 % of cones**, so 97 % carry an assumed direction - marked in the data."),
        ("evidencia/prioridad", "video",
         "**Right of way frame by frame, with the FIE rule that grants it.** The detector **does not "
         "work yet**: 69.7 % accuracy against **73.8 % for never answering**. The videos are here to "
         "see WHERE it fails."),
        ("evidencia/mano_dominante", "imagen",
         "**The weapon hand.** The old criterion failed **1 in 4**. The new one is a cascade whose "
         "first rung is the **front foot**, a within-body correspondence that does not depend on "
         "where the opponent is. On a **held-out set** of 19 fencers that took no part in choosing "
         "the rule: the **cascade** scores 1.000 over 19 of 19, and the **front-foot rung alone** "
         "covers 12 of those 19 with no failure. What it cannot decide is flagged `uncertain`."),
        ("evidencia/mosaicos", "video",
         "**Six cells at once, synchronised**: original with skeleton, matting, segmentation, "
         "normals, pointmap and blade detection. For seeing at a glance whether one layer drifts."),
        ("evidencia/errores_encontrados", "imagen",
         "**Six cases verified by eye, one by one.** The first three are the real defects found; "
         "**the last three are CORRECT** and are here on purpose, because they are the ones that "
         "stopped us turning a whiteness test into an automatic filter."),
    ]
    for c, tipo, que in filas:
        n = inv.get(c, 0)
        if not n:
            continue
        l.append("### %s  -  %d files" % (c.split("/")[-1].replace("_", " "), n))
        l.append("")
        l.append(que)
        l.append("")
        """EL MP4 VA ENLAZADO APARTE, a calidad original. La ficha de Hugging Face renderiza markdown
        y **no reproduce un `.mp4` en linea**, asi que el GIF es lo unico que se ve sin salir de la
        pagina y el enlace es lo unico que da la calidad de verdad. Van las dos cosas."""
        mp4 = [f for f in ficheros if f.startswith(c + "/") and f.endswith(".mp4")]
        if mp4:
            l.append("**Full videos, original quality:** " + " | ".join(
                "[`%s`](%s/resolve/main/%s)" % (m.split("/")[-1], URL, m) for m in sorted(mp4)[:4]))
            l.append("")
        pv = PREVIA.get(c)
        if pv:
            """LA RUTA VA RELATIVA, y no es un detalle de estilo. Con la URL absoluta la imagen **no
            se ve en un repo PRIVADO**: el renderizador pide el fichero sin la sesion del usuario,
            recibe un 401 y muestra el enlace roto. Con ruta relativa lo resuelve dentro del repo y
            lo sirve autenticado."""
            nombres = [pv] if isinstance(pv, str) else list(pv)
            """`PREVIA` admite UN nombre o VARIOS. Hace falta para la pista: de ese video salen ahora
            un GIF del TRAZADO y otro del MAPA por separado, y con un solo nombre uno de los dos no
            se veria en ningun sitio, que es como se quedan huerfanos los ficheros subidos."""
            for pvn in nombres:
                for suf in ("", "_2", "_3"):
                    nom = pvn.replace(".png", "%s.png" % suf) if suf else pvn
                    if (PREVIAS_LOCAL / nom).exists():
                        l.append("![%s](previa/%s)" % (c.split("/")[-1], nom))
                        l.append("")
            if any(n.endswith(".gif") for n in nombres):
                l.append("*Preview only, and a GIF loses quality. Open the full videos in "
                         "[`%s`](%s).*" % (c.split("/")[-1], enl(c)))
            else:
                l.append("*Examples. All %d in [`%s`](%s).*" % (n, c.split("/")[-1], enl(c)))
        else:
            l.append("Open [`%s`](%s)." % (c.split("/")[-1], enl(c)))
        l.append("")
    l.append("## What this evidence does NOT say")
    l.append("")
    l.append("The figures come from **19 to 60 clips** depending on the measurement, **not** from "
             "the 8,559-clip corpus. The ground truth for the weapon hand, for the steps and for "
             "blade contact **was built by eye inside this project**, not by an FIE referee, and it "
             "is kept in files precisely so that it can be contradicted.")
    l.append("")
    l.append("On invented poses: the error is measured **by hiding poses that are actually known** "
             "and interpolating from their neighbours, which gives 2.2 % of torso length. But the "
             "gaps that really get filled are in good part frames where the fencer **is not in "
             "shot**, and for those no ground truth can exist: their error is, by construction, not "
             "verifiable. That is why every point carries its provenance mark and its gap length, "
             "and **those marks must not be dropped**.")
    l.append("")
    l.append("One defect is still open and recurrent: in about **6 % of clips the pipeline delivers "
             "the REFEREE as one of the two fencers**. Four routes to catch it were tried and **all "
             "four failed, each with its measurement** (whiteness, mask in hand, geometry, "
             "behaviour). The referee pose is perfectly good, so no pose-quality metric will ever "
             "flag it.")
    l.append("")
    return "\n".join(l)


def main():
    import collections
    from huggingface_hub import HfApi
    api = HfApi(token=SE._token())
    seco = "--seco" in sys.argv

    for loc, dest in SELECCION_ERRORES:
        if not loc.exists():
            print("   AVISO: no esta en disco, no se sube: %s" % loc.name)
            continue
        if not seco:
            api.upload_file(path_or_fileobj=str(loc), path_in_repo=dest, repo_id=SE.REPO,
                            repo_type=SE.TIPO, commit_message="seleccion de errores verificados")
    print("seleccion de errores: %d imagenes" % sum(1 for p, _d in SELECCION_ERRORES if p.exists()))

    n_pv = 0
    for f in sorted(PREVIAS_LOCAL.glob("*")) if PREVIAS_LOCAL.exists() else []:
        if not f.is_file():
            continue
        if not seco:
            api.upload_file(path_or_fileobj=str(f), path_in_repo="previa/%s" % f.name,
                            repo_id=SE.REPO, repo_type=SE.TIPO,
                            commit_message="vistas previas para el README")
        n_pv += 1
    print("vistas previas subidas: %d" % n_pv)

    fs = api.list_repo_files(repo_id=SE.REPO, repo_type=SE.TIPO)
    borrar = []
    for f in fs:
        if not f.startswith("evidencia/"):
            continue
        if f.endswith(".mp4") and f not in VIDEOS_QUE_SE_QUEDAN:
            borrar.append(f)
            continue
        car = "/".join(f.split("/")[:2])
        if car in CARPETAS_FINALES:
            continue
        if car in PARCIALES:
            if not PARCIALES[car](f):
                borrar.append(f)
            continue
        borrar.append(f)

    c = collections.Counter("/".join(f.split("/")[:2]) for f in borrar)
    print()
    print("se quitan %d ficheros de la ruta de revision:" % len(borrar))
    for k, v in sorted(c.items(), key=lambda z: -z[1]):
        print("   %-40s %4d" % (k, v))
    if borrar and not seco:
        for i in range(0, len(borrar), 200):
            api.delete_files(repo_id=SE.REPO, repo_type=SE.TIPO,
                             delete_patterns=borrar[i:i + 200],
                             commit_message="dejar solo los entregables finales")

    fs = api.list_repo_files(repo_id=SE.REPO, repo_type=SE.TIPO)
    inv = collections.Counter("/".join(f.split("/")[:2]) for f in fs if f.startswith("evidencia/"))
    if not seco:
        api.upload_file(path_or_fileobj=readme(inv, fs).encode("utf-8"), path_in_repo="README.md",
                        repo_id=SE.REPO, repo_type=SE.TIPO,
                        commit_message="README con los entregables finales y sus enlaces")
    print()
    print("quedan %d ficheros bajo `evidencia/`:" % sum(inv.values()))
    for k, v in sorted(inv.items(), key=lambda z: -z[1]):
        print("   %-40s %4d" % (k, v))
    print()
    print("   %s" % URL)


if __name__ == "__main__":
    main()

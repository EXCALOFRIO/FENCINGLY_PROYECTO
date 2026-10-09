"""PASO 5 · La grafica que pidio el usuario: la pista a escala con la trayectoria y los eventos.

*"Tipo tener como una pista en una grafica, y como una linea en 2D por donde va, en el eje de la
LONGITUD de la pista y la PROFUNDIDAD de la pista, porque igual se va a la derecha y luego a la
izquierda, y hay que trackearlo. Y cada vez que hace un fondo ponerlo; cada vez que estira el brazo
marcarlo como un keypoint en la pista; cada marca donde empieza y acaba."*

Y el video del trazado, que es el entregable principal, esta en `video_pista.py`. Esta figura es el
resumen estatico del mismo clip.

CINCO PANELES, y cada uno responde a una parte de la peticion:

  a  el MAPA de la pista a escala con sus lineas del reglamento y la traza de los dos tiradores, con
     el tiempo codificado en color y una barra de color para leerlo. Verde el de la izquierda y
     naranja el de la derecha, que es el codigo de color del proyecto.
  a2 la coordenada TRANSVERSAL, cruda y filtrada, con su banda de ruido. La FORMA es senial
     (excursion 9 a 77 veces el ruido) y la ESCALA no es creible en 8 de 22 tiradores-clip.
  b  la posicion a lo largo de la pista contra el tiempo, con los eventos marcados. Es el panel
     donde se ve "donde empieza y acaba" cada cosa.
  c  la DISTANCIA entre los dos tiradores en metros contra el tiempo, que es la variable que de
     verdad importa en esgrima.
  d  la separacion entre los pies de cada tirador, que es la senal con la que se detecta el fondo,
     para poder comprobar a ojo si el evento esta donde debe.

=======================================================================================
TRES COSAS QUE ESTA VERSION CORRIGE DE LA ANTERIOR, y las tres eran mias
=======================================================================================

  1. EL PANEL a PINTABA LA TRANSVERSAL COMO SI FUERA BUENA. En `GentleQuerulousElephantseal` la
     traza verde se salia del panel por arriba, porque la transversal medida llega a valores de
     varios metros en una pista de 1,50 m. La seccion 23.7 ya habia medido que ese eje no es
     creible (recorrido p50 0,87 m contra < 0,60 m declarado, falla en 8 de 10 clips) y la figura
     lo pintaba igual. Ahora cada tirador va en un CARRIL FIJO rotulado como dibujo, y la
     transversal medida tiene su propio panel en gris con su cifra.

  2. EL EJE DECIA "0 = LINEA CENTRAL" Y ESO NO SE SABE. La seccion 23.9 dice que el origen
     longitudinal es una marca sin identificar. Ahora el mapa se ancla con la convencion de
     `posicion.ancla_origen` y el titulo da la DERIVA medida de ese anclaje.

  3. ROTULABA UNA "LINEA DE AVISO" A +-5 m QUE NO EXISTE. La seccion 23.1 lo habia verificado en
     fuente: `fie_2021.txt` t.19 dice que los ultimos 2 m se distingan *"if possible by a different
     colour of piste"*, o sea una FRONTERA DE COLOR, no una raya. Ahora se pinta como cambio de
     color y se rotula como lo que es.

LO QUE LA FIGURA DICE DE SI MISMA. En el titulo va la estatura reproyectada de los dos tiradores y el
ancho deducido de la pista, que son los dos controles del paso 2, y si la puerta se pasa con LOS DOS
tiradores en rango o solo con la mediana agrupada, que no es lo mismo (ver seccion 26). Las zonas de
la pista que quedan FUERA del encuadre se pintan en gris, porque el cuadro de television abarca una
mediana de 5,53 m de los 14 y no tenerlo delante lleva a leer una trayectoria corta como si fuera un
tirador que no se mueve.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "01_extraccion" / "codigo"))
import rutas as RU

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import pista as PI
import posicion as PO

SALIDA = RU.PISTA
VERDE = "#1a9850"
NARANJA = "#f46d43"
COL = (VERDE, NARANJA)
MARCA_EV = {"fondo": ("v", "#d73027", 90), "marcha": ("^", "#4575b4", 42),
            "romper": ("v", "#91bfdb", 42), "brazo": ("*", "#762a83", 110),
            "alto": ("s", "#252525", 30), "bote": ("P", "#ff9a00", 55),
            "accion": ("|", "#333333", 60)}
CARRIL = (0.25, 0.75)
"""Donde va cada tirador A LO ANCHO del mapa, en fraccion del ancho de la pista. ES DIBUJO Y NO
MEDIDA: ver el punto 1 de la cabecera del modulo."""


VIA_EN = {"camara": "camera geometry", "zona": "colour zone", "rayas": "painted lines"}
"""Traduccion SOLO PARA MOSTRAR de la columna `via`. El valor del CSV no se toca: cambiarlo
romperia el codigo de otros agentes y los scripts de subida."""


def dibuja_pista(ax, ancho, x0, x1):
    """QUE HACE: pinta la pista del reglamento a escala.

    LAS LINEAS SON LAS DE LA EDICION QUE APLICA, verificada en fuente (seccion 23.1): `fie_2021.txt`
    t.18.1 *"The combat area of the piste is 1.50 meters wide"*, largo 14 m, puesta en guardia a
    +-2 m y limites posteriores a +-7 m. **La marca de los 5 m no se pinta como raya** porque t.19
    la define como frontera de COLOR y no como linea; se pinta como cambio de color.

    Se pinta SIEMPRE los 14 m completos y se sombrea lo que no entra en el encuadre, para que la
    trayectoria se lea en su sitio real y no reescalada al trozo visible.
    """
    ax.add_patch(plt.Rectangle((-7, 0), 14, ancho, fc="#f0efe9", ec="#555555", lw=1.2, zorder=1))
    for xa, xb in ((-7.0, -5.0), (5.0, 7.0)):
        ax.add_patch(plt.Rectangle((xa, 0), xb - xa, ancho, fc="#e0d8c0", ec="none", zorder=1.1))
    ax.plot([0, 0], [0, ancho], color="#333333", lw=1.4, ls=(0, (4, 3)), zorder=2)
    for x in (-2.0, 2.0):
        ax.plot([x, x], [0, ancho], color="#333333", lw=1.1, zorder=2)
    for x in (-7.0, 7.0):
        ax.plot([x, x], [0, ancho], color="#000000", lw=1.6, zorder=2)
    if x0 > -7.0:
        ax.add_patch(plt.Rectangle((-7, 0), x0 + 7, ancho, fc="#9e9e9e", alpha=0.55, zorder=3))
    if x1 < 7.0:
        ax.add_patch(plt.Rectangle((x1, 0), 7 - x1, ancho, fc="#9e9e9e", alpha=0.55, zorder=3))
    ax.set_xlim(-7.6, 7.6)
    ax.set_ylim(-0.35, ancho + 0.35)
    ax.set_aspect("equal")
    ax.set_yticks([])
    for x, t in ((0, "centre"), (-2, "on-guard"), (2, "on-guard"),
                 (-7, "limit"), (7, "limit")):
        ax.text(x, ancho + 0.10, t, ha="center", va="bottom", fontsize=6.0, color="#555555")
    for x in (-6.0, 6.0):
        ax.text(x, 0.5 * ancho, "t.19: colour,\nnot a line", ha="center", va="center",
                fontsize=5.4, color="#7a6a3a", zorder=4)


def _traza(ax, C, ks, lado, ancho, fps, zorder=5, trv=None):
    """QUE HACE: la traza de un tirador en el mapa, como LINEA CONTINUA con el tiempo codificado en
    color, y devuelve la coleccion para poder ponerle barra de color.

    LA TRAZA VA EN UN CARRIL FIJO y no en la transversal medida: ver el punto 1 de la cabecera. Lo
    que se dibuja es exactamente lo que esta medido, que es la coordenada A LO LARGO de la pista, con
    el tiempo como segunda variable.
    """
    x = C[ks, lado, 0]
    m = np.isfinite(x)
    if m.sum() < 2:
        return None
    xx = x[m]
    tt = (ks[m] - ks[0]) / fps
    """LA TRAZA LLEVA LA FORMA MEDIDA DE LA TRANSVERSAL, recentrada en el carril del tirador. Antes
    iba en un carril FIJO y eso pintaba una linea recta perfecta, que es una afirmacion tan falsa
    como pintar el ruido: la excursion transversal medida es 9 a 77 veces el ruido por cuadro en los
    22 tiradores-clip. Ver `posicion.transversal`."""
    if trv is not None and trv["forma_publicable"]:
        yv = trv["y"][ks][m]
        ymed = float(np.nanmedian(trv["y"]))
        yy = CARRIL[lado] * ancho + np.where(np.isfinite(yv), yv - ymed, 0.0)
    else:
        yy = np.full(len(xx), CARRIL[lado] * ancho)
    pts = np.column_stack([xx, yy]).reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap="viridis", linewidths=2.6, zorder=zorder)
    lc.set_array(tt[:-1])
    ax.add_collection(lc)
    ax.plot(xx[0], yy[0], "o", ms=8.0, mfc="none", mec=COL[lado], mew=2.0, zorder=zorder + 1)
    ax.plot(xx[-1], yy[-1], "s", ms=7.0, color=COL[lado], zorder=zorder + 1)
    ax.text(xx[0], yy[0] - 0.16 * ancho, "start", ha="center", va="top", fontsize=5.6,
            color=COL[lado])
    ax.text(-7.4, CARRIL[lado] * ancho, "LEFT (green)" if lado == 0 else "RIGHT (orange)",
            ha="right",
            va="center", fontsize=6.4, color=COL[lado])
    return lc


def figura(clip, res, ap, ev, fps, ruido, ancla=None):
    """QUE HACE: la figura de cinco paneles de un clip."""
    r = res["_r"]
    ks = np.nonzero(r["mask"])[0]
    C, dist, dpies = ap["C"], ap["dist"], ap["dpies"]
    anc = res["_hg"]["ancho_m"]
    if ancla is None:
        ancla = PO.ancla_origen(ap, r["mask"])
    x00 = ancla["x0_m"]
    C = C.copy()
    C[:, :, 0] -= x00
    """SE RESTA EL ANCLAJE UNA SOLA VEZ Y AQUI, para que los cinco paneles y el CSV hablen del mismo
    origen. Es la convencion de `posicion.ancla_origen` y su deriva medida va en el titulo."""
    xs = C[ks][:, :, 0]
    xs = xs[np.isfinite(xs)]
    x0, x1 = (float(xs.min()), float(xs.max())) if xs.size else (-7.0, 7.0)
    fig = plt.figure(figsize=(11.2, 9.6))
    gs = fig.add_gridspec(5, 1, height_ratios=[2.3, 0.8, 1.5, 1.0, 1.0], hspace=0.62)

    ax = fig.add_subplot(gs[0])
    dibuja_pista(ax, anc, x0, x1)
    lc = None
    trv_map = {l: PO.transversal(ap, r["mask"], l) for l in (0, 1)}
    for lado in (0, 1):
        lc = _traza(ax, C, ks, lado, anc, fps, trv=trv_map[lado]) or lc
    for e in ev:
        if e["tipo"] not in MARCA_EV or e.get("pos_m") is None:
            continue
        px = e["pos_m"][0]
        if px is None:
            continue
        lado = int(e.get("lado", -1))
        py = CARRIL[lado if lado >= 0 else 0] * anc
        py += 0.17 * anc if lado == 1 else -0.17 * anc
        mk, col, sz = MARCA_EV[e["tipo"]]
        ax.scatter([px - x00], [py], marker=mk, s=sz, c=col, zorder=8, linewidths=0.7,
                   edgecolors="white")
    if lc is not None:
        cb = fig.colorbar(lc, ax=ax, orientation="vertical", pad=0.012, fraction=0.055,
                          aspect=9)
        cb.set_label("time (s)", fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
    ax.set_xlabel("position along the piste (m)  ·  origin by CONVENTION (midpoint of the two "
                  "fencers, clip median): measured drift %.2f m, %s"
                  % (ancla.get("deriva_m", float("nan")),
                     "usable" if ancla.get("fiable") else "NOT usable as absolute position"),
                  fontsize=7.5)
    ax.set_title("a . THE PISTE MAP and the 2D trace of both fencers  ·  colour = time, "
                 "circle = start, square = end\n"
                 "across the piste we draw the MEASURED SHAPE of the lateral coordinate, "
                 "re-centred on its lane; its SCALE is uncertain (see a2)",
                 fontsize=8.2, loc="left")

    axt = fig.add_subplot(gs[1])
    tt0 = (ks - ks[0]) / fps
    """SE PUBLICAN LAS DOS CURVAS, CRUDA Y FILTRADA, y no solo la que se usa. Sin la cruda al lado no
    hay forma de comprobar que el filtro no se ha inventado la forma, y ese es justo el riesgo de
    filtrar. La cruda en gris finito, la filtrada en el color del tirador."""
    trv = {lado: PO.transversal(ap, r["mask"], lado) for lado in (0, 1)}
    for lado in (0, 1):
        axt.plot(tt0, trv[lado]["y_cruda"][ks], color="#bbbbbb", lw=0.7, zorder=2,
                 label="raw" if lado == 0 else None)
        axt.plot(tt0, trv[lado]["y"][ks], color=COL[lado], lw=1.5, zorder=3,
                 label=("filtered L" if lado == 0 else "filtered R"))
        ru = trv[lado]["ruido_m"]
        if np.isfinite(ru):
            axt.fill_between(tt0, trv[lado]["y"][ks] - 3 * ru, trv[lado]["y"][ks] + 3 * ru,
                             color=COL[lado], alpha=0.16, lw=0, zorder=1)
    axt.axhspan(0.0, anc, color="#dddddd", alpha=0.45, zorder=0)
    axt.axhline(0.0, color="#999999", lw=0.7)
    axt.axhline(anc, color="#999999", lw=0.7)
    raz = min(trv[l]["razon"] for l in (0, 1))
    ex = max(trv[l]["excursion_m"] for l in (0, 1))
    esc = all(trv[l]["escala_creible"] for l in (0, 1))
    axt.set_ylabel("lateral\n(m)", fontsize=7)
    axt.set_xlabel("time (s)", fontsize=7)
    axt.legend(fontsize=5.6, loc="upper right", frameon=False, ncol=3)
    axt.set_title("a2 . the lateral coordinate: the SHAPE is signal (range %.0fx the per-frame "
                  "noise, window %d) and the SCALE %s (range %.2f m, control < 0.60 m)  ·  band = "
                  "+-3x the measured noise  ·  grey = piste width"
                  % (raz, PO.TRANSV_VENTANA, "is credible" if esc else "is NOT", ex),
                  fontsize=8.0, loc="left", color="#30508a" if esc else "#8a3030")

    ax2 = fig.add_subplot(gs[2])
    tt = (ks - ks[0]) / fps
    for lado in (0, 1):
        ax2.plot(tt, C[ks, lado, 0], color=COL[lado], lw=1.5,
                 label="%s fencer" % ("left" if lado == 0 else "right"))
    for x, ls in ((0.0, "--"), (-2.0, "-"), (2.0, "-")):
        ax2.axhline(x, color="#999999", lw=0.7, ls=ls, zorder=0)
    for e in ev:
        if e["tipo"] == "accion":
            ax2.axvspan((e["inicio"] - ks[0]) / fps, (e["fin"] - ks[0]) / fps,
                        color="#000000", alpha=0.06, zorder=0)
            continue
        if e["tipo"] not in MARCA_EV or e.get("lado", -1) < 0:
            continue
        px, _py = e["pos_m"]
        if px is None:
            continue
        mk, col, sz = MARCA_EV[e["tipo"]]
        """SE RESTA EL ANCLAJE TAMBIEN AQUI, y faltaba: `pos_m` viene en el sistema CRUDO de la
        homografia y el panel esta en el sistema anclado. Sin restarlo los eventos salian flotando a
        5-6 m por encima de unas trazas que van de -2,5 a 3,5 m, y se veia mirando la figura: un
        evento que no cae sobre la traza del tirador al que pertenece es un error de dibujo, no un
        dato raro."""
        ax2.scatter([(e["cuadro"] - ks[0]) / fps], [px - x00], marker=mk, s=sz * 0.6, c=col,
                    zorder=6, linewidths=0.6, edgecolors="white")
    ax2.set_ylabel("position along\nthe piste (m)")
    ax2.set_xlabel("time (s)")
    ax2.legend(fontsize=6.5, loc="upper right", frameon=False, ncol=2)
    ax2.set_title("b . where each fencer is and when each thing happens  ·  grey band = action",
                  fontsize=8.5, loc="left")

    ax3 = fig.add_subplot(gs[3])
    ax3.plot(tt, dist[ks], color="#333333", lw=1.6)
    ax3.plot(tt, ap["dist_2d"][ks], color="#999999", lw=0.9, ls=":")
    ax3.set_ylabel("distance between\nfencers (m)")
    ax3.set_xlabel("time (s)")
    ax3.axhline(2.0, color="#d73027", lw=0.8, ls="--")
    ax3.text(tt[-1], 2.0, " 2 m", fontsize=6, color="#d73027", va="center")
    ax3.set_title("c . distance between the two along the piste (black) and in 2D (dotted), "
                  "the variable that decides in fencing", fontsize=8.5, loc="left")

    ax4 = fig.add_subplot(gs[4])
    for lado in (0, 1):
        ax4.plot(tt, dpies[ks, lado], color=COL[lado], lw=1.3)
    ax4.axhspan(0.50, 0.90, color="#1a9850", alpha=0.07, zorder=0)
    for e in ev:
        if e["tipo"] != "fondo":
            continue
        ax4.axvline((e["cuadro"] - ks[0]) / fps, color="#d73027", lw=0.9, alpha=0.8)
    ax4.set_ylabel("feet\nseparation (m)")
    ax4.set_xlabel("time (s)")
    ax4.set_title("d . the signal the lunge is detected from, with lunges marked  ·  green band = "
                  "0.50-0.90 m, a foilist on guard", fontsize=8.5, loc="left")

    n = {t: sum(1 for e in ev if e["tipo"] == t) for t in MARCA_EV}
    h0, h1 = res.get("h0_p50"), res.get("h1_p50")

    def _rg(v):
        return v is not None and np.isfinite(v) and 1.60 <= float(v) <= 2.00
    if not res.get("publicable"):
        puerta = "FAILS the physical gate: the metres are ILLUSTRATIVE ONLY"
    elif _rg(h0) and _rg(h1):
        puerta = "passes with BOTH fencers in 1.60-2.00 m"
    else:
        puerta = "passes the POOLED gate but NOT with both fencers in range: the scale may be wrong"

    def _n(x):
        """NINGUN `None` PUEDE LLEGAR A UN FORMATO NUMERICO, y este es el MISMO fallo que el proyecto
        ya documento en `pista.py`: *"el lote de 78 clips se corto a mitad por esto"*. Lo repeti aqui
        y me costo dos clips de la tirada de 21. Hay que blindar el VALOR, no la ausencia de clave."""
        try:
            return float("nan") if x is None else float(x)
        except (TypeError, ValueError):
            return float("nan")

    fig.suptitle("%s  ·  %d usable frames at %.1f fps  ·  reprojected height %.2f m "
                 "(L %.2f / R %.2f)  ·  deduced width %.2f m  ·  along-piste jitter p95 %.3f m\n"
                 "%s  ·  route %s  ·  camera pan %.0f px  ·  physical controls %s\n"
                 "events: %d lunges, %d marches, %d retreats, %d halts, %d balestras, "
                 "%d arm extensions, %d actions"
                 % (clip, len(ks), _n(fps), _n(res.get("h_p50")), _n(h0), _n(h1),
                    _n(res.get("ancho_deducido_m")),
                    _n(ruido.get("temblor_largo_p95_m")),
                    puerta, VIA_EN.get(res.get("via"), res.get("via", "?")), _n(res.get("despl_camara_px")),
                    "PASS" if ruido.get("pasa_fisica") else "FAIL",
                    n["fondo"], n["marcha"], n["romper"], n["alto"], n["bote"], n["brazo"],
                    n["accion"]),
                 fontsize=8.0, y=0.995)
    SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA / ("pista_%s.png" % clip), dpi=135, bbox_inches="tight")
    plt.close(fig)


COLUMNAS_SERIE = ["clip", "lado", "lado_valido", "tipo", "indice_del_tipo", "cuadro_inicio", "cuadro_fin",
                  "cuadro_pico", "t_inicio_s", "t_fin_s", "duracion_s", "x_inicio_m", "x_fin_m",
                  "magnitud_m", "unidad_magnitud", "velocidad_m_s", "y_inicio_m", "y_fin_m",
                  "transversal_creible", "pies_movidos", "completa", "ext_torso",
                  "solapa_fondo", "arbitrado", "razon_pie_atras", "pico_dpies_guardia",
                  "margen_arbitraje", "guardia_m", "guardia_plausible", "lado_no_valido_por", "via_accion", "magnitud_altos_caja", "levanta_frac_caja", "sobre_ruido", "tipo_establecido",
                  "sep_pies_m",
                  "umbral_quieto_m", "mano", "eje", "origen_max", "pose_fiable", "hueco_max",
                  "hueco_sin_medir", "publicable", "puerta_dos_lados", "ancla_deriva_m",
                  "ancla_fiable", "x_origen_m"]
"""EL ORDEN DE LAS COLUMNAS DEL CSV DE LA SERIE, y esta escrito a mano a proposito.

La version anterior escribia `sorted(todas las claves)` y el resultado era un CSV donde la primera
columna era `clip` y la segunda `cuadro`, con `tipo` en la duodecima. Un CSV que hay que leer con el
diccionario delante no es una serie consultable. Aqui el orden es el de la pregunta: quien, que,
cual, cuando, donde, cuanto, y luego la procedencia."""


def una(clip, con_figura=True):
    """QUE HACE: todo el encadenado de un clip: homografia, posiciones, eventos, figura y la SERIE
    de eventos consultable."""
    res = PI.analiza(clip)
    if not res.get("ok"):
        """SIN HOMOGRAFIA NO HAY METROS, pero SI hay la accion medida en altos de caja. Ver
        `posicion.acciones_sin_homografia`: es la unica de los seis tipos que se puede trasladar a
        una unidad de imagen sin suponer nada sobre la escala, y es justo la que hace falta como
        puerta de la regla de prioridad de otro agente. De los 10 clips de su conjunto reservado, 6
        no tienen homografia y sin esto daban cero."""
        fps = PO._fps(PI.indice_videos().get(clip) or "")
        ev = PO.eventos_sin_homografia(res, fps, PO.carga_poses(clip))
        if not ev:
            print("%-38s SIN HOMOGRAFIA y sin accion: %s" % (clip, res.get("motivo")))
            return None
        serie = PO.serie(ev, {"C": np.full((res["_r"]["T"], 2, 2), np.nan)}, clip, fps, None,
                         extra={"publicable": False, "puerta_dos_lados": False,
                                "ancla_fiable": False, "sin_homografia": True})
        print("%-38s SIN HOMOGRAFIA: accion %d (altos de caja) + brazo %d (fraccion de torso)"
              % (clip, sum(1 for e in ev if e["tipo"] == "accion"),
                 sum(1 for e in ev if e["tipo"] == "brazo")))
        return {"clip": clip, "res": res, "ap": None, "ev": ev, "fps": fps, "ruido": {},
                "ancla": {"fiable": False}, "serie": serie, "publicable": False,
                "puerta_dos_lados": False}
    video = PI.indice_videos()[clip]
    fps = PO._fps(video)
    poses = PO.carga_poses(clip)
    ap = PO.apoyos(res, poses)
    rui = PO.ruido(ap["C"], res["_r"]["mask"], fps)
    rui.update(PO.controles_fisicos(ap, res["_r"]["mask"], res["_hg"]["ancho_m"]))
    ev = PO.eventos(ap, res["_r"], fps, poses, res=res)
    ancla = PO.ancla_origen(ap, res["_r"]["mask"])
    h0, h1 = res.get("h0_p50"), res.get("h1_p50")

    def _rg(v):
        return v is not None and np.isfinite(v) and 1.60 <= float(v) <= 2.00
    extra = {"publicable": bool(res.get("publicable")),
             "puerta_dos_lados": bool(_rg(h0) and _rg(h1)),
             "ancla_deriva_m": round(ancla.get("deriva_m", float("nan")), 3)
             if np.isfinite(ancla.get("deriva_m", float("nan"))) else None,
             "ancla_fiable": bool(ancla.get("fiable")),
             "x_origen_m": round(float(ancla["x0_m"]), 3)}
    """LA x DE LA SERIE SE DA RELATIVA AL ANCLAJE Y SE GUARDA EL ANCLAJE, las dos cosas. Asi el CSV
    es directamente comparable con lo que sale en el video y en la figura, y quien quiera volver al
    sistema crudo de la homografia solo tiene que sumar `x_origen_m`."""
    apr = dict(ap)
    apr["C"] = ap["C"].copy()
    apr["C"][:, :, 0] -= float(ancla["x0_m"])
    serie = PO.serie(ev, apr, clip, fps, poses, extra=extra)
    if con_figura:
        figura(clip, res, ap, ev, fps, rui, ancla=ancla)
    n = {t: sum(1 for e in ev if e["tipo"] == t) for t in MARCA_EV}

    def _n(x):
        """El MISMO blindaje que en `figura`: `res.get(k, nan)` NO protege de nada cuando la clave
        existe y su valor es `None`, que es justo lo que devuelve `control_estatura` o
        `controles_fisicos` cuando un control no se ha podido medir. Hay que blindar el VALOR, no la
        ausencia de la clave. Es lo que corto la tirada de 21 clips en dos sitios distintos."""
        try:
            return float("nan") if x is None else float(x)
        except (TypeError, ValueError):
            return float("nan")

    print("%-38s pose %s h %.2f (%.2f/%.2f) dos-lados %s | temblor largo %.3f m | transv %.2f m "
          "dpies %.2f m | ancla %.2f m %s | fon %d mar %d rom %d par %d bot %d bra %d acc %d"
          % (clip, "SI" if poses else "no", _n(res.get("h_p50")), _n(h0), _n(h1),
             "SI" if extra["puerta_dos_lados"] else "no",
             _n(rui.get("temblor_largo_p95_m")),
             _n(rui.get("transv_rango_m")), _n(rui.get("dpies_p50_m")),
             _n(ancla.get("deriva_m")), "ok" if ancla.get("fiable") else "NO",
             n["fondo"], n["marcha"], n["romper"], n["alto"], n["bote"], n["brazo"],
             n["accion"]))
    return {"clip": clip, "res": res, "ap": ap, "ev": ev, "fps": fps, "ruido": rui,
            "ancla": ancla, "serie": serie, "publicable": res.get("publicable"),
            "puerta_dos_lados": extra["puerta_dos_lados"]}


def escribe_serie(filas, nombre="serie_eventos.csv"):
    """QUE HACE: el CSV de la serie de eventos, con las columnas en orden de pregunta.

    EL CRITERIO DE QUE ESTA BIEN HECHO, declarado antes: la consulta que pidio el usuario --
    *"la tercera marcha del tirador de la derecha"* -- tiene que ser UNA linea de pandas y devolver
    UNA fila:

        ev.query("lado == 1 and tipo == 'marcha' and indice_del_tipo == 3")
    """
    import csv
    SALIDA.mkdir(parents=True, exist_ok=True)
    cols = COLUMNAS_SERIE + sorted({k for f in filas for k in f} - set(COLUMNAS_SERIE))
    with open(SALIDA / nombre, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    return SALIDA / nombre


def _escribe(filas, nombre, orden=None):
    """QUE HACE: un CSV con las columnas en el orden dado y el resto detras, alfabetico."""
    import csv
    SALIDA.mkdir(parents=True, exist_ok=True)
    todas = {k for f in filas for k in f}
    cols = (list(orden) + sorted(todas - set(orden))) if orden else sorted(todas)
    with open(SALIDA / nombre, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    return SALIDA / nombre


def reune(patron, nombre, orden=None) -> dict:
    """QUE HACE: reconstruye un CSV AGREGADO concatenando TODOS los por-clip que haya en disco, y
    comprueba con asertos que no se ha perdido nada.

    POR QUE EXISTE, y es un fallo mio de diseno que ya se materializo: el `__main__` escribia los
    agregados (`serie_eventos.csv`, `eventos.csv`, `posicion_y_eventos.csv`) con los clips que se le
    pasaran en la linea de ordenes. Asi que al final de la sesion volvi a correr UN clip suelto para
    comprobar un arreglo del panel b, y eso **sobrescribio los tres agregados de 11 clips con uno
    solo**: `serie_eventos.csv` paso de 284 filas y 11 clips a 40 filas y 1 clip.

    Y lo peor es que mi criterio de aceptacion seguia pasando: la consulta
    `lado==1 and tipo=='marcha' and indice_del_tipo==3` devolvia UNA fila, pero la devolvia porque el
    fichero solo tenia un clip. Un criterio que se cumple por la razon equivocada no es un criterio:
    de ahi los dos asertos de abajo, que son la parte que impide que vuelva.

    El agregado NUNCA se escribe con lo que se acaba de calcular: se escribe leyendo del disco todos
    los por-clip. Asi una pasada de un clip suelto es inofensiva por construccion, y no por acordarse
    de pasar la lista entera.
    """
    import pandas as pd
    fich = [p for p in sorted(SALIDA.glob(patron)) if p.name != nombre]
    """EL AGREGADO SE EXCLUYE DEL GLOB EXPLICITAMENTE, y no es paranoia: `posicion_*.csv` casa con
    `posicion_y_eventos.csv`, que es el propio agregado, asi que sin este filtro cada pasada
    duplicaria todas las filas y el aserto de cuentas lo cazaria pero con un mensaje confuso. Los
    por-clip de posicion se llaman `posicion_clip_<clip>.csv` por la misma razon."""
    if not fich:
        return {"n": 0, "clips": 0, "ficheros": 0}
    trozos, suma = [], 0
    for p in fich:
        t = pd.read_csv(p)
        suma += len(t)
        trozos.append(t)
    todo = pd.concat(trozos, ignore_index=True)
    assert len(todo) == suma, \
        "el agregado %s tiene %d filas y los %d por-clip suman %d" % (nombre, len(todo),
                                                                      len(fich), suma)
    assert todo["clip"].nunique() == len(fich), \
        "el agregado %s tiene %d clips distintos y hay %d ficheros por-clip" % (
            nombre, todo["clip"].nunique(), len(fich))
    if orden:
        cols = [c for c in orden if c in todo.columns]
        todo = todo[cols + [c for c in todo.columns if c not in cols]]
    todo.to_csv(SALIDA / nombre, index=False)
    return {"n": len(todo), "clips": int(todo["clip"].nunique()), "ficheros": len(fich)}


COLUMNAS_POS = ["clip", "tiene_pose_308", "sin_homografia", "s_utiles", "fps", "publicable", "puerta_dos_lados",
                "pasos_por_s_util_y_tirador", "n_marcha", "n_romper", "n_fondo", "n_alto",
                "n_brazo", "n_bote", "n_accion", "cruces_imposibles", "ancla_deriva_m",
                "ancla_fiable", "pasa_fisica", "n"]


def fila_posicion(d, c):
    """QUE DEVUELVE: la fila de `posicion_clip_<clip>.csv` para un clip ya encadenado por `una()`.

    POR QUE ES UNA FUNCION Y NO EL CUERPO DE UN BUCLE. Esto vivia dentro del `if __name__ ==
    "__main__"`, asi que la unica manera de obtener la fila era lanzar el fichero como programa. La
    etapa 7 del pipeline la llamaba con `GP.main(...) if hasattr(GP, "main") else None`, y
    `grafica_pista` **no tiene `main`**: la etapa informaba exito y no escribia nada. Sacarla aqui
    es lo que hace que la etapa 7 se pueda cablear de verdad.
    """
    ks = np.nonzero(d["res"]["_r"]["mask"])[0]
    util = float(ks[-1] - ks[0] + 1) / d["fps"]
    npasos = sum(1 for e in d["ev"] if e["tipo"] in ("marcha", "romper"))
    """UN CLIP SIN HOMOGRAFIA NO TIENE `ap`, asi que todo lo que se derive de la posicion metrica
    va a `None` y NO a cero: `cruces_imposibles`, la deriva del anclaje y las tasas de pasos no se
    han podido medir, y ponerles un cero las haria indistinguibles de un clip donde se midieron y
    salieron cero. La columna `sin_homografia` dice cual es el caso."""
    sinh = d["ap"] is None
    f = {"clip": c, "fps": round(d["fps"], 2), "publicable": d["publicable"],
         "puerta_dos_lados": d["puerta_dos_lados"],
         "tiene_pose_308": bool(PO.carga_poses(c) is not None),
         "sin_homografia": sinh,
         "s_utiles": round(util, 2),
         "pasos_por_s_util_y_tirador": (None if sinh
                                        else round(npasos / (2.0 * util), 3)),
         "cruces_imposibles": None if sinh else d["ap"].get("cruces"),
         "ancla_deriva_m": (None if sinh or not np.isfinite(
             d["ancla"].get("deriva_m", float("nan")))
             else round(d["ancla"]["deriva_m"], 3)),
         "ancla_fiable": d["ancla"].get("fiable")}
    """`s_utiles` Y `tiene_pose_308` SON NUEVAS Y HACEN FALTA LAS DOS.

    `s_utiles` porque la columna `n` NO es una duracion: es el numero de saltos sobre los DOS
    tiradores, o sea el doble de los cuadros, y usarla como duracion daba la mitad de la tasa
    real (0,49 pasos por segundo y tirador en vez de 0,70). `s_utiles` se calcula con el TRAMO de
    cuadros.

    `tiene_pose_308` porque los eventos de pies y de brazo EXIGEN la pose de 308 puntos, y un
    clip sin ella da cero por construccion. Promediar una tasa con ceros estructurales no es una
    tasa: hay que poder filtrar."""
    f.update({k: round(v, 4) if isinstance(v, float) else v for k, v in d["ruido"].items()})
    for t in MARCA_EV:
        f["n_" + t] = sum(1 for e in d["ev"] if e["tipo"] == t)
    if PO.PUERTA_EVENTO:
        """Cuantas filas de la serie no pasan la puerta fisica POR EVENTO (`posicion.fisica_evento`).
        `publicable` de esta fila sigue siendo el del CLIP: esto dice cuanto hay que creerselo."""
        f["n_eventos_fisica_mal"] = sum(1 for s in d["serie"] if s.get("fisica_evento") is False)
    for t in ("fondo", "marcha", "romper", "alto"):
        m = [e["magnitud_m"] for e in d["ev"]
             if e["tipo"] == t and np.isfinite(e.get("magnitud_m", np.nan))]
        f["mag_p50_" + t] = round(float(np.median(m)), 3) if m else None
    evs = []
    for e in d["ev"]:
        rr = {"clip": c}
        rr.update({k: v for k, v in e.items()})
        rr["pos_x_m"] = e["pos_m"][0] if e.get("pos_m") else None
        rr["pos_y_m"] = e["pos_m"][1] if e.get("pos_m") else None
        rr.pop("pos_m", None)
        evs.append(rr)
    """LOS TRES SE ESCRIBEN POR CLIP Y DESPUES SE REUNEN. Antes `eventos.csv` y
    `posicion_y_eventos.csv` solo existian como agregado, asi que una pasada de un clip borraba
    los otros diez sin dejar de donde recuperarlos. Ahora cada clip deja su fichero."""
    escribe_serie(d["serie"], "serie_eventos_%s.csv" % c)
    _escribe(evs, "eventos_%s.csv" % c, ["clip", "tipo", "lado", "cuadro", "inicio", "fin"])
    _escribe([f], "posicion_clip_%s.csv" % c, COLUMNAS_POS)
    return f


def procesa(clip, con_figura=False):
    """QUE HACE: un clip de punta a punta y sus TRES CSV. Es lo que llama la etapa 7 del pipeline.

    `con_figura=False` a proposito, y no es el defecto de `una()`: la figura son cinco paneles de
    matplotlib por clip, que en una corrida de 100.000 clips es tiempo y disco a cambio de nada
    -- la figura existe para mirar un clip, no para recorrer un corpus.

    QUE **NO** HACE: llamar a `reune()`. `reune` recorre el directorio entero y reconcatena, asi
    que por clip es O(n^2) y ademas rompe la regla de que los agregados se reconstruyen leyendo el
    disco UNA vez. `reune` se llama una vez por lote, desde `lotes.py`.

    QUE DEVUELVE: la fila de posicion, o None si el clip no da ni homografia ni accion -- y en ese
    caso **no se escribe ningun fichero**, en vez de escribir uno vacio e informar exito.
    """
    d = una(clip, con_figura=con_figura)
    if d is None:
        return None
    return fila_posicion(d, clip)


if __name__ == "__main__":
    hechos = []
    for c in [a for a in sys.argv[1:] if not a.startswith("-")]:
        try:
            if procesa(c, con_figura="--sin-figura" not in sys.argv) is not None:
                hechos.append(c)
        except Exception as exc:
            import traceback
            print("%-38s EXCEPCION %s" % (c, exc))
            traceback.print_exc()
    print()
    for pat, nom, orden in (("serie_eventos_*.csv", "serie_eventos.csv", COLUMNAS_SERIE),
                            ("eventos_*.csv", "eventos.csv", None),
                            ("posicion_clip_*.csv", "posicion_y_eventos.csv", COLUMNAS_POS)):
        r = reune(pat, nom, orden)
        print("%-24s %4d filas de %2d clips (%d ficheros por-clip)"
              % (nom, r["n"], r["clips"], r["ficheros"]))
    print("\nrecalculados en esta pasada: %d clips (%s)"
          % (len(hechos), ", ".join(hechos) if len(hechos) <= 3 else "..."))

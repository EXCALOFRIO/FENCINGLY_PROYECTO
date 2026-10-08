"""ETAPA D - herramienta de validacion visual de poses. Cinco modos, una sola idea: que se VEA.

POR QUE EXISTE, y la razon ha resultado ser todavia mejor de lo que se creia: el 2026-09-09 nueve
detectores de coherencia geometrica declararon LIMPIO un clip donde los dos esqueletos aparecian
dibujados sobre pared vacia, y una sola imagen lo desmintio. El 2026-09-10, al pintar ese mismo
cuadro con la ruta de lectura correcta, resulto que **los esqueletos estaban bien y lo que estaba mal
era la imagen**: se pintaba el fotograma equivocado. O sea que la imagen desmintio primero a las
tablas y despues a si misma. Las dos veces gano mirar, y ninguna de las dos gano calcular.

Consecuencia que hay que tener presente al usar esto: a fecha de hoy **no queda ningun caso
verificado de contaminacion real** en el corpus. Los dos que habia eran este fallo. Encontrar el
primero de verdad es para lo que sirve esta herramienta.

LO PRIMERO QUE ARREGLA, Y ES LO MAS IMPORTANTE DE TODO EL FICHERO
-----------------------------------------------------------------
Los NPZ no cubren el clip entero: cubren una VENTANA del clip. Su metadato lo dice:

    f0=175  f1=306  n_frames_window=131  n_frames_clip=329  t_window_start_s=5.8298

O sea que `poses[65]` NO es el fotograma 65 del mp4, es el fotograma **f0 + 65 = 240**. Medido sobre
una muestra aleatoria de 300 NPZ: **292 (97,3 %) tienen `f0` distinto de cero**. Todas las
herramientas anteriores (`criticos.py`, `ver_sospechosos.py`, `etapa_a_seg_exacto.py`) hacian
`cap.set(CAP_PROP_POS_FRAMES, k)` con el indice de pose, o sea que pintaban el esqueleto correcto
sobre **otro fotograma del video**. Un esqueleto correcto sobre el fotograma equivocado cae donde no
hay nadie: sobre una pared, sobre el publico, sobre un oficial.

Por eso cada imagen que sale de aqui lleva ESCRITA la ruta de lectura que uso:

    poses[65] -> video[240]   f0=175  t=7.996 s  ventana 175..306 de 329

Sin esa linea, un fallo de alineacion se puede confundir otra vez con un intruso. Y existe la opcion
`--ignora-f0` con un unico fin: **reproducir el artefacto viejo al lado del bueno**, para poder
comparar las dos imagenes en vez de discutirlo.

LOS CINCO MODOS
---------------
    cuadros    una imagen por fotograma: dos esqueletos en colores distintos, sus cajas, RECORTE
               AMPLIADO de cada caja y los numeros medidos de ese cuadro. Numerada por cuadro.
    tira       rejilla de N cuadros seguidos en una sola imagen, con los recortes ampliados, para
               revisar un clip de un vistazo.
    criticos   solo los fotogramas sospechosos, con el criterio declarado abajo.
    control    los CUATRO casos verificados a ojo, con la ruta buena y la ruta vieja al lado.
    limpios    muestra de clips que MI algoritmo declara LIMPIOS, con controles ciegos mezclados.

POR QUE EL MODO `limpios` NO ES OPCIONAL: si solo se ensena lo que el algoritmo marca, quien mire
solo puede confirmar o negar mis positivos, y nunca encontrar lo que NO detecto. Los falsos negativos
son justo lo que preocupa. La carpeta `limpios` es la unica que los puede sacar.

EL CRITERIO DE "CRITICO", DECLARADO ANTES DE MIRAR, Y DE DONDE SALE CADA NUMERO
-------------------------------------------------------------------------------
Nada de esto se ha ajustado despues de ver los resultados. Motivos DUROS (uno solo ya marca):

    razon de alturas de caja >= 1.60   ya medido: 0,99 % del corpus. Se traia fama de PRECISO (2 de 2
                                       inspeccionados) pero esa fama se cae en cuanto se mira con la
                                       ruta buena: en `ImpartialClosedCollardlizard` c116-c121 marca
                                       2,30 y la tira de contacto ensena que es el MISMO tirador
                                       haciendo una fleche, borroso y agachado, con la caja
                                       achatada. O sea que marca bien lo raro y NO demuestra
                                       contaminacion: por eso esto es una herramienta de mirar y no
                                       una puerta de borrado
    separacion en x < 0.04             ya medido: la misma persona contada dos veces, 14 clips de
                                       10.618
    menos de 3 de los 4 puntos de torso utiles   no es una sospecha, es que el cuadro NO ES MEDIBLE:
                                       sin torso no hay blancura ni centro que comparar

Motivos de AVISO (hacen falta DOS para marcar, porque ninguno por si solo es una puerta):

    pies fuera de la banda (0.45, 0.97)   ya medido: la banda NO es puerta dura sino penalizacion
                                          (1,99 % del corpus la incumple y hay perspectiva legitima)
    salto del torso > 0.20 alturas de caja entre cuadros consecutivos
                                          medido AQUI hoy: p99,75 = 0,205 sobre 68.002 saltos de 300
                                          clips, que es exactamente la tasa de teletransportes que
                                          ya estaba medida (0,25 % de los CUADROS)
    alto de caja < 0.625 de la mediana de su propio lado en el clip
                                          es el mismo factor 1,6 de la razon de alturas, aplicado
                                          dentro de un lado en vez de entre los dos
    blancura del torso < 0.02 cuando la mediana del clip es >= 0.10
                                          es un AVISO y nunca una puerta, y aqui esta la cifra que lo
                                          obliga. La distribucion de ayer estaba medida con la ruta
                                          MALA. Remedida hoy sobre los mismos 200 clips y los mismos
                                          cuadros relativos, con las dos rutas:

                                              cuantil        ruta vieja   ruta buena
                                              p50               0,092        0,164
                                              p90               0,397        0,635
                                              por debajo de 0,02   20,5 %      4,5 %
                                              por debajo de 0,05   37,5 %     13,5 %

                                          O sea que tres cuartas partes de los clips "que no visten
                                          de blanco" eran el desfase, no la ropa. Y el 4,5 % que
                                          queda tampoco es una puerta: se ha mirado un caso a ojo
                                          (ImperturbableHeartyAustralianshelduck c120) y son DOS
                                          TIRADORES REALES en un pabellon con luz pobre, con 0,069 y
                                          0,000. Por eso solo cuenta el contraste dentro del clip.

TRAMPAS DE DATOS YA MEDIDAS QUE ESTE FICHERO RESPETA
----------------------------------------------------
    hay puntos con confianza >= 0,3 y coordenada NaN: se filtra por confianza Y por np.isfinite, y
        ademas se ESCRIBE en el panel cuantos habia, para que la trampa no se pueda olvidar
    `d.clip` en pandas es un metodo, no la columna: se usa siempre `d["clip"]`
    `box_coordinate_space` es `image_normalized`: las cajas se pintan sin tocar el `pan`, que esta en
        espacio de pista (`x_piste = x_image + pan`) y aqui no se aplica

INTERPRETE (el del PATH no tiene cv2):
    <BASE>/scratchpad/fz/.venv/Scripts/python.exe validador_visual.py <modo> ...

EJEMPLOS
    python validador_visual.py control
    python validador_visual.py cuadros --clip SoreEasygoingEmperorshrimp --desde 55 --hasta 75
    python validador_visual.py tira --clip SentimentalLargeBedlingtonterrier --n 24
    python validador_visual.py criticos --clip AltruisticIllfatedDobermanpinscher
    python validador_visual.py limpios --n 12 --ciegos 3
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import tempfile
import time

import cv2
import numpy as np

BASE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565")
NPZ_DIRS = [BASE / "localrun/runs_extra1/data_20260907T005507157743Z/npz",
            BASE / "localrun/runs_extra2/data_20260907T005507370214Z/npz",
            BASE / "localrun/runs_extra3/data_20260907T010246490954Z/npz",
            BASE / "scratchpad/catalogo/ws_completo/npz"]
RAIZ_VIDEOS = pathlib.Path("D:/FENCINGLY_VIDEOS_RAW")
PROYECTO = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO")
SALIDA_DEF = PROYECTO / "02_validacion" / "muestras" / "validacion_visual"
QC_CSV = [PROYECTO / "03_datos" / "poses_qc.csv",
          pathlib.Path(r"C:\Users\Alejandro\.claude\jobs\187cfd83\tmp\poses_qc.csv")]
CONT_CSV = [PROYECTO / "03_datos" / "contaminacion.csv",
            pathlib.Path(r"C:\Users\Alejandro\.claude\jobs\187cfd83\tmp\contaminacion.csv")]
CACHE = pathlib.Path(tempfile.gettempdir()) / "fencingly_validador_catalogo.json"

CONF_MIN = 0.3
L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12
TORSO = [L_SHO, R_SHO, L_HIP, R_HIP]
L_ANK, R_ANK = 15, 16
HUESOS = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
          (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
COLOR = ((60, 170, 255), (90, 230, 120))
NOMBRE_COLOR = ("naranja", "verde")
BANDA = (0.45, 0.97)

U_RAZON_ALTURAS = 1.60
U_SEP_X = 0.04
U_TORSO_MIN = 3
U_SALTO = 0.20
U_ALTO_REL = 0.625
U_BLANCURA = 0.02
U_BLANCURA_CLIP = 0.10

FUENTE = cv2.FONT_HERSHEY_SIMPLEX
GRIS = (150, 150, 150)
BLANCO = (240, 240, 240)
ROJO = (70, 70, 245)
AMBAR = (60, 200, 245)
FONDO = (24, 22, 28)


def catalogo(refresca=False):
    """Mapea clip_id a su mp4 y a su npz. Se cachea porque `D:` tiene mas de 42.000 mp4."""
    if CACHE.is_file() and not refresca:
        try:
            c = json.loads(CACHE.read_text(encoding="utf-8"))
            if c.get("npz") and c.get("mp4"):
                return c["npz"], c["mp4"]
        except Exception:
            pass
    npz_de, mp4_de = {}, {}
    for d in NPZ_DIRS:
        if d.is_dir():
            for f in sorted(d.glob("*.npz")):
                npz_de.setdefault(f.stem, str(f))
    for raiz, _sub, ficheros in os.walk(str(RAIZ_VIDEOS)):
        for f in ficheros:
            if f.lower().endswith(".mp4"):
                mp4_de.setdefault(pathlib.Path(f).stem, os.path.join(raiz, f))
    try:
        CACHE.write_text(json.dumps({"npz": npz_de, "mp4": mp4_de}), encoding="utf-8")
    except Exception:
        pass
    return npz_de, mp4_de


def carga(cid, npz_de):
    """Devuelve poses, cajas, tiempos y metadato. Nada de esto se recalcula: se lee tal cual."""
    ruta = npz_de.get(cid)
    if not ruta:
        return None
    with np.load(ruta, allow_pickle=True) as z:
        claves = list(z.keys())
        P = np.asarray(z["poses"], dtype="float64")
        B = np.asarray(z["boxes"], dtype="float64")
        t = np.asarray(z["time_s"], dtype="float64") if "time_s" in claves else None
        try:
            meta = json.loads(str(z["meta_json"])) if "meta_json" in claves else {}
        except Exception:
            meta = {}
    if P.ndim != 4 or B.ndim != 3:
        return None
    return {"clip": cid, "poses": P, "boxes": B, "time_s": t, "meta": meta, "npz": ruta}


class Lector:
    """Lee fotogramas del mp4 SIEMPRE traduciendo el indice de pose al fotograma real del video.

    La traduccion es `f0 + k` y se comprueba contra `time_s[k] * fps`, que es un camino
    independiente. Si los dos no coinciden en menos de 2 fotogramas se avisa por consola, porque
    eso significaria que la ventana no es contigua y esta clase estaria mintiendo.
    """

    def __init__(self, mp4, meta, time_s, ignora_f0=False):
        self.cap = cv2.VideoCapture(str(mp4))
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(meta.get("fps") or self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.f0 = 0 if ignora_f0 else int(meta.get("f0") or 0)
        self.f1 = int(meta.get("f1") or self.n)
        self.time_s = time_s
        self.ignora_f0 = ignora_f0
        self.pos = None
        self.aviso = ""
        self.meta = meta

    def cuadro_video(self, k):
        return int(self.f0 + k)

    def coherencia(self, k):
        """Segundo camino para el mismo numero: el tiempo absoluto del NPZ."""
        if self.time_s is None or k >= len(self.time_s) or not np.isfinite(self.time_s[k]):
            return None
        return int(round(float(self.time_s[k]) * self.fps))

    def lee(self, k):
        f = self.cuadro_video(k)
        alt = self.coherencia(k)
        if alt is not None and not self.ignora_f0 and abs(alt - f) > 2:
            self.aviso = "f0+k=%d pero time_s*fps=%d: la ventana NO es contigua" % (f, alt)
        if f < 0 or (self.n > 0 and f >= self.n):
            return None
        if self.pos is not None and 0 <= f - self.pos <= 12:
            while self.pos < f:
                if not self.cap.grab():
                    return None
                self.pos += 1
            ok, im = self.cap.retrieve()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, im = self.cap.read()
            self.pos = f
        if not ok or im is None:
            return None
        self.pos = f
        return im

    def cierra(self):
        self.cap.release()


def utiles(kp):
    """Confianza Y finitud. Filtrar solo por confianza pinta basura: hay conf>=0,3 con NaN."""
    return (kp[:, 2] >= CONF_MIN) & np.isfinite(kp[:, 0]) & np.isfinite(kp[:, 1])


def conf_con_nan(kp):
    return int(((kp[:, 2] >= CONF_MIN) & ~(np.isfinite(kp[:, 0]) & np.isfinite(kp[:, 1]))).sum())


def caja_torso(kp, W, H):
    """Rectangulo del torso en pixeles, o None si no hay 3 de los 4 puntos."""
    m = utiles(kp)[TORSO]
    if int(m.sum()) < U_TORSO_MIN:
        return None
    xs = kp[TORSO, 0][m] * W
    ys = kp[TORSO, 1][m] * H
    x0, x1 = int(max(0, xs.min())), int(min(W, xs.max()))
    y0, y1 = int(max(0, ys.min())), int(min(H, ys.max()))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return x0, y0, x1, y1


def blancura(im, kp):
    """Fraccion de pixeles CLAROS y POCO SATURADOS en el torso: el uniforme de florete es blanco.

    Mismo criterio que `pose._whiteness` y que la medida de ayer, sin tocar ni un umbral: claro no
    basta, tiene que ser poco saturado, porque una camiseta amarilla de arbitro tambien es clara.
    """
    H, W = im.shape[:2]
    r = caja_torso(kp, W, H)
    if r is None:
        return None
    x0, y0, x1, y1 = r
    tor = im[y0:y1, x0:x1]
    if tor.size == 0:
        return None
    hsv = cv2.cvtColor(tor, cv2.COLOR_BGR2HSV)
    return float(np.mean((hsv[:, :, 2] >= 150) & (hsv[:, :, 1] <= 70)))


def centro_torso(kp):
    m = utiles(kp)[TORSO]
    if int(m.sum()) < U_TORSO_MIN:
        return None
    return float(np.mean(kp[TORSO, 0][m])), float(np.mean(kp[TORSO, 1][m]))


def contexto(dat):
    """Medidas del clip entero que hacen falta para juzgar un cuadro: medianas por lado."""
    B = dat["boxes"]
    alto = B[:, :, 3] - B[:, :, 1]
    with np.errstate(invalid="ignore"):
        med = [float(np.nanmedian(np.where(alto[:, s] > 0, alto[:, s], np.nan))) for s in (0, 1)]
    return {"alto_med": med, "blancura_clip": None}


def segmento(meta, k):
    """En que segmento del clip cae el cuadro. `nd` significa que ahi NO HAY DATO por declaracion.

    Sin esto, los cuadros posteriores a un corte parecen un fallo de extraccion y no lo son: el
    propio metadato dice `cut_segments: [[0, 414, 'main'], [414, 500, 'nd']]`.
    """
    for tramo in (meta.get("cut_segments") or []):
        try:
            a, b, etiqueta = int(tramo[0]), int(tramo[1]), str(tramo[2])
        except Exception:
            continue
        if a <= k < b:
            return etiqueta
    return "main" if not meta.get("cut_segments") else "?"


def medidas(dat, k, im, ctx):
    """Los numeros de ESE cuadro. Ni uno viene de otro cuadro salvo el salto, que es por definicion."""
    P, B = dat["poses"], dat["boxes"]
    H, W = im.shape[:2]
    m = {"k": k, "lados": []}
    altos = []
    for s in (0, 1):
        kp, b = P[k, s], B[k, s]
        u = utiles(kp)
        alto = float(b[3] - b[1]) if np.isfinite(b).all() else float("nan")
        ancho = float(b[2] - b[0]) if np.isfinite(b).all() else float("nan")
        altos.append(alto if np.isfinite(alto) and alto > 0 else float("nan"))
        ay = kp[[L_ANK, R_ANK], 1][u[[L_ANK, R_ANK]]]
        pies = float(np.max(ay)) if ay.size else float("nan")
        salto = float("nan")
        if k > 0:
            c0, c1 = centro_torso(P[k - 1, s]), centro_torso(P[k, s])
            if c0 and c1 and np.isfinite(alto) and alto > 0:
                salto = float(np.hypot(c1[0] - c0[0], c1[1] - c0[1]) / alto)
        m["lados"].append({
            "n_util": int(u.sum()), "n_torso": int(u[TORSO].sum()),
            "conf_nan": conf_con_nan(kp),
            "alto": alto, "ancho": ancho,
            "alto_px": alto * H if np.isfinite(alto) else float("nan"),
            "ancho_px": ancho * W if np.isfinite(ancho) else float("nan"),
            "alto_rel": alto / ctx["alto_med"][s] if np.isfinite(alto) and ctx["alto_med"][s] else float("nan"),
            "blancura": blancura(im, kp), "pies_y": pies, "salto": salto,
            "en_banda": bool(BANDA[0] <= pies <= BANDA[1]) if np.isfinite(pies) else None})
    with np.errstate(invalid="ignore"):
        m["razon_alturas"] = float(np.nanmax(altos) / max(np.nanmin(altos), 1e-9)) if np.isfinite(altos).any() else float("nan")
    cx = 0.5 * (B[k, :, 0] + B[k, :, 2])
    m["sep_x"] = float(abs(cx[1] - cx[0])) if np.isfinite(cx).all() else float("nan")
    return m


def veredicto(m, blancura_clip):
    """Aplica el criterio declarado en la cabecera. Devuelve (duros, avisos), ya con su cifra."""
    duros, avisos = [], []
    if np.isfinite(m["razon_alturas"]) and m["razon_alturas"] >= U_RAZON_ALTURAS:
        duros.append("razon alturas %.2f >= %.2f" % (m["razon_alturas"], U_RAZON_ALTURAS))
    if np.isfinite(m["sep_x"]) and m["sep_x"] < U_SEP_X:
        duros.append("sep_x %.3f < %.2f: puede ser la misma persona dos veces" % (m["sep_x"], U_SEP_X))
    for s, L in enumerate(m["lados"]):
        if L["n_torso"] < U_TORSO_MIN:
            duros.append("lado %d sin torso medible (%d de 4)" % (s, L["n_torso"]))
        if L["en_banda"] is False:
            avisos.append("lado %d pies en %.3f, fuera de (%.2f, %.2f)" % (s, L["pies_y"], *BANDA))
        if np.isfinite(L["salto"]) and L["salto"] > U_SALTO:
            avisos.append("lado %d salto de torso %.2f alturas de caja" % (s, L["salto"]))
        if np.isfinite(L["alto_rel"]) and L["alto_rel"] < U_ALTO_REL:
            avisos.append("lado %d caja al %.0f %% de su mediana del clip" % (s, 100 * L["alto_rel"]))
        if (L["blancura"] is not None and L["blancura"] < U_BLANCURA
                and blancura_clip is not None and blancura_clip >= U_BLANCURA_CLIP):
            avisos.append("lado %d blancura %.3f con mediana del clip %.3f" % (s, L["blancura"], blancura_clip))
    return duros, avisos


def es_critico(duros, avisos):
    return bool(duros) or len(avisos) >= 2


def pinta_esqueleto(lienzo, kp, col, W, H, ox=0, oy=0, esc=1.0, grosor=2, radio=3):
    u = utiles(kp)
    """El int() se hace SOLO sobre puntos utiles: hay conf>=0,3 con coordenada NaN y `int(NaN)` casca."""
    px = [(int((kp[j, 0] * W - ox) * esc), int((kp[j, 1] * H - oy) * esc)) if u[j] else (0, 0)
          for j in range(kp.shape[0])]
    for a, b in HUESOS:
        if u[a] and u[b]:
            cv2.line(lienzo, px[a], px[b], col, grosor, cv2.LINE_AA)
    for j in range(kp.shape[0]):
        if u[j]:
            cv2.circle(lienzo, px[j], radio, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(lienzo, px[j], radio, col, 1, cv2.LINE_AA)


def recorte(im, kp, b, col, ancho, alto, margen=0.30):
    """El recorte AMPLIADO de la caja. Es lo que hace evidente si hay alguien o hay pared.

    Un tirador mide 144x155 px de mediana: en un cuadro reducido a 960 de ancho eso son 100 px y no
    se distingue una pared de una chaqueta. Ampliado a 260 px si.
    """
    H, W = im.shape[:2]
    cel = np.full((alto, ancho, 3), (14, 13, 17), dtype="uint8")
    if not (np.isfinite(b).all() and b[2] > b[0] and b[3] > b[1]):
        cv2.putText(cel, "SIN CAJA", (10, alto // 2), FUENTE, 0.6, ROJO, 2)
        return cel
    bw, bh = (b[2] - b[0]) * W, (b[3] - b[1]) * H
    x0 = int(max(0, b[0] * W - margen * bw))
    y0 = int(max(0, b[1] * H - margen * bh))
    x1 = int(min(W, b[2] * W + margen * bw))
    y1 = int(min(H, b[3] * H + margen * bh))
    if x1 - x0 < 6 or y1 - y0 < 6:
        cv2.putText(cel, "CAJA DIMINUTA", (10, alto // 2), FUENTE, 0.6, ROJO, 2)
        return cel
    sub = im[y0:y1, x0:x1].copy()
    esc = min(ancho / float(x1 - x0), alto / float(y1 - y0))
    dib = cv2.resize(sub, (int((x1 - x0) * esc), int((y1 - y0) * esc)),
                     interpolation=cv2.INTER_CUBIC if esc > 1 else cv2.INTER_AREA)
    lienzo = dib
    tor = caja_torso(kp, W, H)
    if tor is not None:
        cv2.rectangle(lienzo, (int((tor[0] - x0) * esc), int((tor[1] - y0) * esc)),
                      (int((tor[2] - x0) * esc), int((tor[3] - y0) * esc)), (255, 255, 255), 1)
    cv2.rectangle(lienzo, (int((b[0] * W - x0) * esc), int((b[1] * H - y0) * esc)),
                  (int((b[2] * W - x0) * esc), int((b[3] * H - y0) * esc)), col, 2)
    pinta_esqueleto(lienzo, kp, col, W, H, ox=x0, oy=y0, esc=esc, grosor=2, radio=3)
    yy = (alto - lienzo.shape[0]) // 2
    xx = (ancho - lienzo.shape[1]) // 2
    cel[yy:yy + lienzo.shape[0], xx:xx + lienzo.shape[1]] = lienzo
    return cel


def marco(im, dat, k, med, ancho=960):
    """El cuadro completo con los dos esqueletos, sus cajas y la banda de pista."""
    H, W = im.shape[:2]
    v = im.copy()
    for y in BANDA:
        cv2.line(v, (0, int(y * H)), (W, int(y * H)), (95, 95, 95), 1)
        cv2.putText(v, "banda %.2f" % y, (W - 150, int(y * H) - 8), FUENTE, 0.5, (95, 95, 95), 1)
    for s in (0, 1):
        b = dat["boxes"][k, s]
        col = COLOR[s]
        if np.isfinite(b).all() and b[2] > b[0]:
            cv2.rectangle(v, (int(b[0] * W), int(b[1] * H)), (int(b[2] * W), int(b[3] * H)), col, 2)
            cv2.putText(v, "lado %d" % s, (int(b[0] * W), max(14, int(b[1] * H) - 6)),
                        FUENTE, 0.6, col, 2)
        pinta_esqueleto(v, dat["poses"][k, s], col, W, H, grosor=2, radio=3)
    return cv2.resize(v, (ancho, int(round(ancho * H / float(W)))))


def texto(lienzo, x, y, lineas, esc=0.5, salto=20):
    for t, col in lineas:
        cv2.putText(lienzo, t, (x, y), FUENTE, esc, col, 1, cv2.LINE_AA)
        y += salto
    return y


def panel(im, dat, k, med, lec, blancura_clip, nota="", oculta_ruta=False):
    """La imagen de un fotograma: marco a la izquierda, recortes ampliados y numeros a la derecha."""
    duros, avisos = veredicto(med, blancura_clip)
    crit = es_critico(duros, avisos)
    FW, MG, CW = 960, 12, 600
    izq = marco(im, dat, k, med, ancho=FW)
    FH = izq.shape[0]
    CAB = 116
    PIE = 26 + 18 * max(1, len(duros) + len(avisos))
    W = MG + FW + MG + CW + MG
    H = CAB + FH + MG + PIE
    L = np.full((H, W, 3), FONDO, dtype="uint8")
    L[CAB:CAB + FH, MG:MG + FW] = izq
    borde = ROJO if crit else (70, 130, 70)
    cv2.rectangle(L, (MG - 2, CAB - 2), (MG + FW + 1, CAB + FH + 1), borde, 2)

    cv2.putText(L, "CUADRO %d" % k, (MG, 40), FUENTE, 1.05, BLANCO, 2, cv2.LINE_AA)
    cv2.putText(L, "%s   [%s]" % (dat["clip"][:44], "poses %d..%d" % (0, dat["poses"].shape[0] - 1)),
                (MG + 250, 34), FUENTE, 0.62, BLANCO, 1, cv2.LINE_AA)
    et = "CRITICO" if crit else "sin marca"
    cv2.putText(L, et, (W - 260, 40), FUENTE, 0.95, ROJO if crit else (120, 200, 120), 2, cv2.LINE_AA)
    if oculta_ruta:
        """En un control ciego no se escribe la ventana ni el tiempo: delatarian el desalineado."""
        ruta = "poses[%d] -> video[%d]" % (k, lec.cuadro_video(k))
    else:
        ruta = "poses[%d] -> video[%d]   f0=%d   t=%.3f s   ventana %d..%d de %d   fps %.3f" % (
            k, lec.cuadro_video(k), lec.f0,
            float(lec.time_s[k]) if lec.time_s is not None and k < len(lec.time_s) else float("nan"),
            lec.f0, lec.f1, lec.n, lec.fps)
    cv2.putText(L, "RUTA DE LECTURA  " + ruta, (MG, 66), FUENTE, 0.52,
                AMBAR if (lec.ignora_f0 and not oculta_ruta) else (170, 220, 170), 1, cv2.LINE_AA)
    if lec.ignora_f0 and not oculta_ruta:
        cv2.putText(L, "!! RUTA VIEJA: se ignora f0 a proposito, para reproducir el artefacto",
                    (MG, 88), FUENTE, 0.55, ROJO, 2, cv2.LINE_AA)
    seg = segmento(dat["meta"], k)
    if not (lec.ignora_f0 and not oculta_ruta):
        cv2.putText(L, "razon alturas %.2f   sep_x %.3f   segmento %s   %s" % (
            med["razon_alturas"], med["sep_x"], seg, nota),
            (MG, 88), FUENTE, 0.55, AMBAR if seg != "main" else GRIS, 1, cv2.LINE_AA)
    if lec.aviso:
        cv2.putText(L, "!! " + lec.aviso, (MG + 600, 88), FUENTE, 0.5, ROJO, 1, cv2.LINE_AA)

    x0 = MG + FW + MG
    for s in (0, 1):
        L2 = med["lados"][s]
        cy = CAB + s * (FH // 2 + 6)
        alto_cel = FH // 2 - 6
        cel = recorte(im, dat["poses"][k, s], dat["boxes"][k, s], COLOR[s], 258, alto_cel)
        L[cy:cy + alto_cel, x0:x0 + 258] = cel
        cv2.rectangle(L, (x0 - 1, cy - 1), (x0 + 258, cy + alto_cel), COLOR[s], 1)
        bl = "n/d" if L2["blancura"] is None else "%.3f" % L2["blancura"]
        lineas = [("LADO %d  (%s)" % (s, NOMBRE_COLOR[s]), COLOR[s]),
                  ("puntos utiles      %d de 17" % L2["n_util"], BLANCO),
                  ("torso utiles       %d de 4" % L2["n_torso"],
                   ROJO if L2["n_torso"] < U_TORSO_MIN else BLANCO),
                  ("conf>=0.3 con NaN  %d" % L2["conf_nan"],
                   AMBAR if L2["conf_nan"] else GRIS),
                  ("caja               %.0f x %.0f px" % (L2["ancho_px"], L2["alto_px"]), BLANCO),
                  ("alto / mediana     %.2f" % L2["alto_rel"],
                   AMBAR if np.isfinite(L2["alto_rel"]) and L2["alto_rel"] < U_ALTO_REL else BLANCO),
                  ("blancura torso     %s" % bl, BLANCO),
                  ("pies y             %s" % (
                      "sin tobillos utiles" if L2["en_banda"] is None else
                      "%.3f  %s" % (L2["pies_y"], "EN BANDA" if L2["en_banda"] else "FUERA")),
                   GRIS if L2["en_banda"] is None else (BLANCO if L2["en_banda"] else AMBAR)),
                  ("salto torso        %.3f alturas" % L2["salto"],
                   AMBAR if np.isfinite(L2["salto"]) and L2["salto"] > U_SALTO else BLANCO)]
        texto(L, x0 + 270, cy + 22, lineas, esc=0.48, salto=21)

    y = CAB + FH + MG + 18
    if duros or avisos:
        y = texto(L, MG, y, [("%s %s" % ("DURO  ", t), ROJO) for t in duros] +
                  [("%s %s" % ("aviso ", t), AMBAR) for t in avisos], esc=0.5, salto=18)
    else:
        texto(L, MG, y, [("ningun motivo salta en este cuadro. Si aqui hay un intruso, es un FALSO "
                          "NEGATIVO mio y es lo mas valioso que puedes encontrar", (150, 210, 150))],
              esc=0.5, salto=18)
    return L, crit, duros, avisos


def blancura_del_clip(dat, lec, n=7):
    """Mediana de la blancura del clip: sin ella, el umbral absoluto no significa nada."""
    T = dat["poses"].shape[0]
    ks = np.unique(np.linspace(0, T - 1, min(n, T)).astype(int))
    vals = []
    for k in ks:
        im = lec.lee(int(k))
        if im is None:
            continue
        for s in (0, 1):
            b = blancura(im, dat["poses"][k, s])
            if b is not None:
                vals.append(b)
    lec.pos = None
    return float(np.median(vals)) if vals else None


def prepara(cid, npz_de, mp4_de, ignora_f0=False, calla=False):
    dat = carga(cid, npz_de)
    if dat is None:
        return None, None, None
    mp4 = mp4_de.get(cid)
    if not mp4:
        return None, None, None
    lec = Lector(mp4, dat["meta"], dat["time_s"], ignora_f0=ignora_f0)
    if not calla:
        """El desfase se DICE siempre, tambien por consola. Es la trampa que ya costo un dia."""
        f0real = int(dat["meta"].get("f0") or 0)
        aviso = "" if f0real == 0 else "  <-- DESFASE de %d fotogramas" % f0real
        print("  ventana de %s: poses 0..%d -> video %d..%d de %d%s" % (
            cid, dat["poses"].shape[0] - 1, f0real, int(dat["meta"].get("f1") or lec.n), lec.n,
            aviso))
    return dat, lec, contexto(dat)


def modo_cuadros(a, npz_de, mp4_de, sal):
    dat, lec, ctx = prepara(a.clip, npz_de, mp4_de, a.ignora_f0)
    if dat is None:
        print("  falta material para %s" % a.clip)
        return
    T = dat["poses"].shape[0]
    d0 = max(0, a.desde)
    d1 = T if a.hasta is None else min(T, a.hasta + 1)
    bl = blancura_del_clip(dat, lec)
    carpeta = sal / "cuadros" / (a.clip + ("__RUTA_VIEJA" if a.ignora_f0 else ""))
    carpeta.mkdir(parents=True, exist_ok=True)
    filas, n, ncrit = [], 0, 0
    for k in range(d0, d1, max(1, a.paso)):
        im = lec.lee(k)
        if im is None:
            continue
        med = medidas(dat, k, im, ctx)
        L, crit, duros, avisos = panel(im, dat, k, med, lec, bl)
        cv2.imwrite(str(carpeta / ("c%04d%s.png" % (k, "_CRITICO" if crit else ""))), L)
        n += 1
        ncrit += int(crit)
        filas.append(fila_csv(a.clip, k, lec, med, crit, duros, avisos))
    lec.cierra()
    escribe_csv(carpeta / "medidas.csv", filas)
    print("  %s: %d imagenes en %s  (criticas %d)" % (a.clip, n, carpeta, ncrit))


def modo_tira(a, npz_de, mp4_de, sal):
    dat, lec, ctx = prepara(a.clip, npz_de, mp4_de, a.ignora_f0)
    if dat is None:
        print("  falta material para %s" % a.clip)
        return
    T = dat["poses"].shape[0]
    d0 = max(0, a.desde)
    d1 = T - 1 if a.hasta is None else min(T - 1, a.hasta)
    ks = np.unique(np.linspace(d0, d1, min(a.n, d1 - d0 + 1)).astype(int))
    bl = blancura_del_clip(dat, lec)
    CW, CH, COLS = 300, 232, 5
    filas_img, cel_actual, filas = [], [], []
    for k in ks:
        im = lec.lee(int(k))
        if im is None:
            continue
        med = medidas(dat, int(k), im, ctx)
        duros, avisos = veredicto(med, bl)
        crit = es_critico(duros, avisos)
        cel = np.full((CH, CW, 3), FONDO, dtype="uint8")
        if a.marco:
            v = marco(im, dat, int(k), med, ancho=CW - 8)
            v = v[:min(v.shape[0], CH - 30)]
            cel[26:26 + v.shape[0], 4:4 + v.shape[1]] = v
        else:
            for s in (0, 1):
                r = recorte(im, dat["poses"][int(k), s], dat["boxes"][int(k), s], COLOR[s],
                            CW // 2 - 6, CH - 32)
                cel[26:26 + r.shape[0], 4 + s * (CW // 2 - 2):4 + s * (CW // 2 - 2) + r.shape[1]] = r
        cv2.putText(cel, "c%d -> v%d" % (k, lec.cuadro_video(int(k))), (6, 18), FUENTE, 0.5,
                    ROJO if crit else BLANCO, 1, cv2.LINE_AA)
        if crit:
            cv2.rectangle(cel, (1, 1), (CW - 2, CH - 2), ROJO, 2)
            cv2.putText(cel, "CRIT", (CW - 54, 18), FUENTE, 0.5, ROJO, 1, cv2.LINE_AA)
        cel_actual.append(cel)
        filas.append(fila_csv(a.clip, int(k), lec, med, crit, duros, avisos))
        if len(cel_actual) == COLS:
            filas_img.append(np.hstack(cel_actual))
            cel_actual = []
    lec.cierra()
    if cel_actual:
        while len(cel_actual) < COLS:
            cel_actual.append(np.full((CH, CW, 3), FONDO, dtype="uint8"))
        filas_img.append(np.hstack(cel_actual))
    if not filas_img:
        print("  no se pudo leer ningun cuadro de %s" % a.clip)
        return
    cuerpo = np.vstack(filas_img)
    cab = np.full((72, cuerpo.shape[1], 3), FONDO, dtype="uint8")
    cv2.putText(cab, "TIRA DE CONTACTO   %s" % dat["clip"][:44], (12, 30), FUENTE, 0.8, BLANCO, 2)
    cv2.putText(cab, "cuadros %d..%d de %d   f0=%d   marco rojo = critico   c<pose> -> v<video>" % (
        d0, d1, T, lec.f0), (12, 56), FUENTE, 0.55, GRIS, 1)
    out = sal / "tiras"
    out.mkdir(parents=True, exist_ok=True)
    ruta = out / ("%s__c%04d_%04d%s.png" % (a.clip, d0, d1, "__RUTA_VIEJA" if a.ignora_f0 else ""))
    cv2.imwrite(str(ruta), np.vstack([cab, cuerpo]))
    escribe_csv(out / ("%s__medidas.csv" % a.clip), filas)
    print("  %s: tira con %d celdas -> %s" % (a.clip, len(filas), ruta))


def modo_criticos(a, npz_de, mp4_de, sal):
    clips = [a.clip] if a.clip else []
    if a.clips:
        clips += [c.strip() for c in a.clips.split(",") if c.strip()]
    for cid in clips:
        dat, lec, ctx = prepara(cid, npz_de, mp4_de, a.ignora_f0)
        if dat is None:
            print("  falta material para %s" % cid)
            continue
        T = dat["poses"].shape[0]
        bl = blancura_del_clip(dat, lec)
        carpeta = sal / "criticos" / cid
        carpeta.mkdir(parents=True, exist_ok=True)
        filas, n, ultimo, ult_motivos = [], 0, -10 ** 6, None
        for k in range(0, T, max(1, a.paso)):
            im = lec.lee(k)
            if im is None:
                continue
            med = medidas(dat, k, im, ctx)
            duros, avisos = veredicto(med, bl)
            if not es_critico(duros, avisos):
                continue
            filas.append(fila_csv(cid, k, lec, med, True, duros, avisos))
            """Una racha del MISMO motivo no necesita 40 imagenes iguales: se pinta la primera y
            luego una cada `--separacion` cuadros. La racha entera queda en el CSV de todos modos."""
            motivos = tuple(sorted(duros + avisos))
            if motivos == ult_motivos and k - ultimo < a.separacion:
                continue
            ultimo, ult_motivos = k, motivos
            L, crit, duros, avisos = panel(im, dat, k, med, lec, bl)
            cv2.imwrite(str(carpeta / ("c%04d.png" % k)), L)
            n += 1
            if n >= a.tope:
                break
        lec.cierra()
        escribe_csv(carpeta / "medidas.csv", filas)
        if filas:
            print("     motivos, por cuadros afectados:")
            cuenta = {}
            for f in filas:
                for m in (f["motivos_duros"] + " | " + f["avisos"]).split(" | "):
                    m = m.strip()
                    if m:
                        cuenta[m] = cuenta.get(m, 0) + 1
            for m, c in sorted(cuenta.items(), key=lambda t: -t[1])[:8]:
                print("       %4d  %s" % (c, m))
        if n == 0:
            (carpeta / "NINGUNO.txt").write_text(
                "Ningun cuadro de %s cumple el criterio de critico declarado en la cabecera de\n"
                "validador_visual.py. Eso NO significa que el clip este bien: significa que mi\n"
                "criterio no marca nada. Para buscar mis falsos negativos usa el modo `cuadros` o\n"
                "el modo `tira` sobre este mismo clip.\n" % cid, encoding="utf-8")
        print("  %s: %d cuadros criticos de %d, %d imagenes pintadas" % (cid, len(filas), T, n))


CONTROL = [("SoreEasygoingEmperorshrimp", 65, "ERA_MI_BUG",
            "se creia pared vacia: con f0=175 los dos esqueletos caen sobre los tiradores"),
           ("AltruisticIllfatedDobermanpinscher", 54, "ERA_MI_BUG",
            "se creian dos oficiales de traje: con f0=197 son los dos tiradores"),
           ("SentimentalLargeBedlingtonterrier", 125, "POSITIVO", "tirador visible 720p, f0=0"),
           ("AdorableFittingAsiansmallclawedotter", 250, "POSITIVO", "tirador visible 1080p, f0=0")]


def modo_control(a, npz_de, mp4_de, sal):
    """Los cuatro casos verificados a ojo. Es la prueba de que la herramienta sirve o no sirve.

    ESTADO A 2026-09-10, y es una correccion, no un matiz: los dos casos que se daban por NEGATIVOS
    no lo eran. Los dos tenian `f0` distinto de cero (175 y 197) y quien los miro los pinto en el
    cuadro `k` del video en vez de en `f0 + k`. Con el desfase aplicado los esqueletos caen sobre los
    dos tiradores. O sea que ahora mismo NO HAY ningun caso verificado de contaminacion real, y esta
    herramienta es lo unico que puede encontrar uno.

    Cada caso con `f0` distinto de cero se pinta DOS VECES: ruta buena y ruta vieja. Ese par es la
    prueba que hay que pasar: el mismo indice de pose tiene que verse BIEN con el desfase y MAL sin
    el. Si la herramienta no distingue ese par a simple vista, no vale para nada mas.
    """
    carpeta = sal / "control"
    carpeta.mkdir(parents=True, exist_ok=True)
    filas = []
    for cid, k, clase, motivo in CONTROL:
        for viejo in (False, True):
            dat, lec, ctx = prepara(cid, npz_de, mp4_de, ignora_f0=viejo)
            if dat is None:
                print("  falta material para %s" % cid)
                break
            if viejo and int(dat["meta"].get("f0") or 0) == 0:
                lec.cierra()
                continue
            if k >= dat["poses"].shape[0]:
                print("  %s no tiene cuadro %d (T=%d)" % (cid, k, dat["poses"].shape[0]))
                lec.cierra()
                break
            bl = blancura_del_clip(dat, lec)
            im = lec.lee(k)
            if im is None:
                print("  %s cuadro %d no decodifica" % (cid, k))
                lec.cierra()
                continue
            med = medidas(dat, k, im, ctx)
            L, crit, duros, avisos = panel(im, dat, k, med, lec, bl,
                                           nota="verificado a ojo: %s (%s)" % (clase, motivo))
            cv2.rectangle(L, (0, 0), (L.shape[1] - 1, L.shape[0] - 1),
                          ROJO if viejo else (90, 200, 90), 6)
            nom = "%s__%s__c%04d%s.png" % (clase, cid[:36], k, "__RUTA_VIEJA" if viejo else "")
            cv2.imwrite(str(carpeta / nom), L)
            filas.append(fila_csv(cid, k, lec, med, crit, duros, avisos,
                                  extra={"clase_verificada": clase, "motivo": motivo,
                                         "ruta": "vieja" if viejo else "buena",
                                         "blancura_clip": bl}))
            print("  %-9s %-38s c%-4d -> video %-4d %s  %s" % (
                clase, cid[:38], k, lec.cuadro_video(k), "RUTA VIEJA" if viejo else "ruta buena",
                "CRITICO: " + "; ".join(duros + avisos) if crit else "sin marca"))
            lec.cierra()
    escribe_csv(carpeta / "control.csv", filas)
    print()
    print("  imagenes de control en %s" % carpeta)


def limpios_declarados():
    """La MISMA definicion de limpio de ayer, para que la carpeta busque MIS falsos negativos."""
    import pandas as pd
    qc = cont = None
    for p in QC_CSV:
        if p.is_file():
            qc = pd.read_csv(p)
            break
    for p in CONT_CSV:
        if p.is_file():
            cont = pd.read_csv(p)
            break
    if qc is None:
        return None
    if cont is not None:
        cont = cont.rename(columns={"clip": "clip_id"})
        qc = qc.merge(cont[["clip_id", "desnivel_pies", "razon_alturas", "sep_x", "fuera_banda_max"]],
                      on="clip_id", how="left")
    sel = qc[(~qc["grave"].astype(bool))]
    if "razon_alturas" in sel.columns:
        sel = sel[(sel["razon_alturas"] < 1.3) & (sel["desnivel_pies"] < 0.06)]
    return sel


def modo_limpios(a, npz_de, mp4_de, sal):
    """Clips que MI algoritmo declara limpios, con controles ciegos mezclados.

    DE DONDE SALEN LOS CIEGOS, y hay que decirlo porque el 2026-09-10 se cayeron los dos que habia:
    no queda ningun caso verificado de contaminacion real, asi que un ciego no se puede sacar del
    corpus. Se FABRICA: se coge un clip cualquiera con `f0` distinto de cero y se pinta con la ruta
    vieja, o sea desalineado a proposito. El resultado es una pose sobre quien no le toca, que es
    exactamente el fallo que hay que saber ver. Se declara asi y no se disfraza de hallazgo.

    Para que sirve: si los ciegos pasan por buenos, la revision no es fiable y hay que repetirla.
    """
    sel = limpios_declarados()
    if sel is None:
        print("  no encuentro poses_qc.csv, no puedo usar la definicion de limpio de ayer")
        return
    sel = sel[sel["clip_id"].isin(npz_de) & sel["clip_id"].isin(mp4_de)]
    print("  clips que declaro limpios y tienen material: %d" % len(sel))
    muestra = sel.sample(n=min(a.n, len(sel)), random_state=a.semilla)
    carpeta = sal / "limpios"
    carpeta.mkdir(parents=True, exist_ok=True)
    tareas = [(str(r["clip_id"]), None, "declarado_limpio") for _, r in muestra.iterrows()]
    ciegos = sel.sample(n=min(a.ciegos * 4, len(sel)), random_state=a.semilla + 1)
    puestos = 0
    for _, r in ciegos.iterrows():
        if puestos >= a.ciegos:
            break
        dat = carga(str(r["clip_id"]), npz_de)
        if dat is None or int(dat["meta"].get("f0") or 0) == 0:
            continue
        tareas.append((str(r["clip_id"]), None, "ciego_desalineado_a_proposito"))
        puestos += 1
    rnd = random.Random(a.semilla)
    rnd.shuffle(tareas)
    filas, clave, n = [], [], 0
    for i, (cid, k, tipo) in enumerate(tareas):
        viejo = tipo.startswith("ciego")
        """Aqui la consola CALLA la ventana: si la imprimiera, delataria a los ciegos."""
        dat, lec, ctx = prepara(cid, npz_de, mp4_de, ignora_f0=viejo, calla=True)
        if dat is None:
            continue
        T = dat["poses"].shape[0]
        if k is None:
            ft = dat["meta"].get("f_touch")
            f0 = int(dat["meta"].get("f0") or 0)
            k = int(ft) - f0 if ft is not None and 0 <= int(ft) - f0 < T else T // 2
        if k >= T:
            lec.cierra()
            continue
        bl = blancura_del_clip(dat, lec)
        im = lec.lee(int(k))
        if im is None:
            lec.cierra()
            continue
        med = medidas(dat, int(k), im, ctx)
        L, crit, duros, avisos = panel(im, dat, int(k), med, lec, bl,
                                       nota="revisa: hay DOS tiradores de verdad en las cajas?",
                                       oculta_ruta=True)
        nom = "caso_%02d.png" % (i + 1)
        cv2.imwrite(str(carpeta / nom), L)
        filas.append(fila_csv(cid, int(k), lec, med, crit, duros, avisos,
                              extra={"imagen": nom, "tu_veredicto": "", "comentario": ""}))
        clave.append({"imagen": nom, "clip_id": cid, "cuadro": k, "tipo": tipo})
        n += 1
        lec.cierra()
    escribe_csv(carpeta / "TUS_VEREDICTOS.csv", filas)
    escribe_csv(sal / "NO_ABRIR_clave_de_los_ciegos.csv", clave)
    (carpeta / "QUE_MIRAR.txt").write_text(
        "Estos casos son clips que MI algoritmo declara LIMPIOS. Si encuentras aqui un arbitro, un\n"
        "espectador, alguien de otra pista, un oficial de traje o un esqueleto sobre una pared, es un\n"
        "FALSO NEGATIVO mio, y eso vale mas que confirmar cualquiera de mis positivos.\n\n"
        "Mira sobre todo los DOS RECORTES AMPLIADOS de la derecha: ahi se ve si en la caja hay un\n"
        "tirador o hay pared. El cuadro completo de la izquierda solo sirve para situarlos.\n\n"
        "Hay %d CONTROLES CIEGOS mezclados: casos que ya se sabe que estan MAL. Si los das por buenos,\n"
        "la revision no es fiable y hay que repetirla. La clave esta fuera de esta carpeta, en\n"
        "NO_ABRIR_clave_de_los_ciegos.csv, y no se abre hasta haber rellenado TUS_VEREDICTOS.csv.\n\n"
        "En TUS_VEREDICTOS.csv rellena `tu_veredicto` con: ok / intruso / pose_sobre_nada / otro.\n"
        % a.ciegos, encoding="utf-8")
    print("  %d casos (%d ciegos) en %s" % (n, a.ciegos, carpeta))


def fila_csv(cid, k, lec, med, crit, duros, avisos, extra=None):
    f = {"clip_id": cid, "cuadro_pose": k, "cuadro_video": lec.cuadro_video(k), "f0": lec.f0,
         "razon_alturas": round(med["razon_alturas"], 4) if np.isfinite(med["razon_alturas"]) else "",
         "sep_x": round(med["sep_x"], 4) if np.isfinite(med["sep_x"]) else "",
         "critico": int(crit), "motivos_duros": " | ".join(duros), "avisos": " | ".join(avisos)}
    for s in (0, 1):
        L = med["lados"][s]
        f["l%d_n_util" % s] = L["n_util"]
        f["l%d_n_torso" % s] = L["n_torso"]
        f["l%d_conf_nan" % s] = L["conf_nan"]
        f["l%d_alto_px" % s] = round(L["alto_px"], 1) if np.isfinite(L["alto_px"]) else ""
        f["l%d_alto_rel" % s] = round(L["alto_rel"], 3) if np.isfinite(L["alto_rel"]) else ""
        f["l%d_blancura" % s] = "" if L["blancura"] is None else round(L["blancura"], 4)
        f["l%d_pies_y" % s] = round(L["pies_y"], 4) if np.isfinite(L["pies_y"]) else ""
        f["l%d_en_banda" % s] = "" if L["en_banda"] is None else int(L["en_banda"])
        f["l%d_salto" % s] = round(L["salto"], 4) if np.isfinite(L["salto"]) else ""
    if extra:
        f.update(extra)
    return f


def escribe_csv(ruta, filas):
    import pandas as pd
    if not filas:
        return
    pd.DataFrame(filas).to_csv(ruta, index=False, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Validacion visual de poses de FENCINGLY")
    p.add_argument("modo", choices=["cuadros", "tira", "criticos", "control", "limpios"])
    p.add_argument("--clip", default=None)
    p.add_argument("--clips", default=None, help="lista separada por comas, solo para criticos")
    p.add_argument("--desde", type=int, default=0)
    p.add_argument("--hasta", type=int, default=None)
    p.add_argument("--paso", type=int, default=1)
    p.add_argument("--n", type=int, default=25, help="celdas de la tira, o casos de limpios")
    p.add_argument("--tope", type=int, default=40, help="maximo de imagenes en modo criticos")
    p.add_argument("--separacion", type=int, default=15,
                   help="cuadros minimos entre dos imagenes del MISMO motivo, en modo criticos")
    p.add_argument("--ciegos", type=int, default=2)
    p.add_argument("--semilla", type=int, default=20260910)
    p.add_argument("--marco", action="store_true", help="tira con el cuadro completo en vez de recortes")
    p.add_argument("--ignora-f0", dest="ignora_f0", action="store_true",
                   help="reproduce la ruta de lectura VIEJA, la que pinta el cuadro equivocado")
    p.add_argument("--salida", default=str(SALIDA_DEF))
    p.add_argument("--refresca-catalogo", dest="refresca", action="store_true")
    a = p.parse_args()
    if a.modo in ("cuadros", "tira") and not a.clip:
        p.error("el modo %s necesita --clip" % a.modo)
    if a.modo == "criticos" and not (a.clip or a.clips):
        p.error("el modo criticos necesita --clip o --clips")
    t0 = time.time()
    npz_de, mp4_de = catalogo(a.refresca)
    sal = pathlib.Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    print("catalogo: %d npz, %d mp4  (%.1f s)" % (len(npz_de), len(mp4_de), time.time() - t0))
    {"cuadros": modo_cuadros, "tira": modo_tira, "criticos": modo_criticos,
     "control": modo_control, "limpios": modo_limpios}[a.modo](a, npz_de, mp4_de, sal)
    print("total %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()

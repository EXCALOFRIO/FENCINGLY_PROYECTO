"""Detector GENERICO de luces (independiente del overlay) + fallback `analyze_any` sobre `fencingly_v2.lights.analyze`.

Idea: una luz de tocado es una region ROJA (izquierda) o VERDE (derecha) saturada, compacta y ESTATICA que se ENCIENDE
de golpe (apagada en las 2 muestras anteriores, encendida >= 0.5 s seguidos) en la banda inferior del frame (zona del
overlay, fuera de la zona central de los tiradores). El tocado es el ULTIMO encendido (y sus companeros en los 0.5 s
anteriores), como en lights.analyze (P5). Se muestrea 1 de cada 3 frames a media resolucion (1080p) y luego se refina
el frame exacto del encendido releyendo los 3 frames anteriores a la muestra.

Filtros anti falsos positivos:
  * pista roja/verde, publicidad, flags del overlay: presentes todo el clip -> nunca hay 'encendido'.
  * cortes de plano / aparicion del overlay entero (replay -> directo): se descartan los encendidos en muestras donde
    cambia > 45 % de la banda, o cuyo ENTORNO (caja dilatada) cambia > 35 % (la luz se enciende sola; el overlay
    aparece con todo lo que le rodea).
  * ropa / zapatillas / tarjetas: se mueven (no cumplen >= 0.5 s encendidas en los MISMOS pixeles) y no son puras
    (S >= 130, V >= 170); ademas la zona central del frame (tiradores, arbitro) no se explora.
  * tamano: area entre ~25 px (punto de Tokio a 960 px de ancho) y 6 % de la banda (glow de Incheon ~3 %); relleno de
    la caja >= 0.2 (barras y circulos ~0.8-1.0, glow-contorno ~0.3).
  * lado: rojo con centro x < 0.5 W, verde con x > 0.5 W (convencion universal del aparato).
Campos devueltos: los de `lights.analyze` (f_touch, t_touch_s, n_valid_lights, dt_ms_green_minus_red, touch_lights,
boxes, events, quality_flags) con box_method='generic'.
"""
from __future__ import annotations
import sys, json, math
from collections import deque
from itertools import islice
from pathlib import Path
import numpy as np
import cv2

from . import lights as L                      # noqa: E402  (read-only, no se modifica)
from .video import Clip                        # noqa: E402

STEP = 3            # 1 de cada 3 frames
BAND_Y0 = 0.74      # banda inferior (Incheon: cajas en 0.80-0.84H con glow; FIE: 0.852H; Kazan: circulos en 0.93-0.97H)
TOP_Y1 = 0.20       # banda superior (solo como ultimo recurso, con flag)
S_MIN, V_MIN = 140, 110   # Tokio (OBS): punto rojo S~243 V~157, punto verde S~222 V~130 (colores oscuros pero puros)
MAX_AREA_FRAC = 0.06
MIN_FILL = 0.08           # glow-contorno de Incheon: anillo de ~4 px alrededor de una caja de 320x40 (media res) -> ~0.13
CUT_FRAC = 0.45     # fraccion de la banda que cambia -> corte de plano
RING_FRAC = 0.35    # fraccion del entorno de la luz que cambia -> aparece el overlay entero, no una luz
RING_STD = 18.0     # std temporal media (gris, 0-255) del entorno durante el encendido: overlay estatico ~3-8, cuerpo/suelo descubierto > 20
MIN_ON_S = 0.5
# [v2] Tokio 2020 (grafico OBS): las luces son dos PUNTOS de ~0.018W junto a la caja del periodo (rojo x~0.463W, verde
# x~0.537W, y~0.93H, V~130-160). Las cajas de nombre NO se colorean (lo 'rojo' es la pista ROSA de Tokio vista a traves
# de las cajas semitransparentes y bajo el overlay) y la franja fina de color sobre las cajas no sigue al tocado.
OBS_DOT_X = {"red": (0.43, 0.49), "green": (0.51, 0.57)}   # centro x del punto (fraccion de W)
OBS_DOT_Y = (0.905, 0.955)                                  # centro y (fraccion de H; medido 0.928)
OBS_DOT_W = (0.012, 0.025)                                  # lado del punto (fraccion de W; medido 0.0177)
OBS_DOT_VMAX = 205                                          # los puntos son oscuros; una barra FIE (V>180 y ancha) no es un punto
OBS_OFF_VMAX = 140                                          # punto APAGADO = oscuro (V medio de la caja antes del encendido; el panel
                                                            # es semitransparente y deja ver la pista rosa: no llega a negro puro)
CENTER_Y = (0.20, BAND_Y0)                                  # zona central (aparato fisico) solo con allow_center
CENTER_MAX_AREA_FRAC = 0.004                                # LEDs del aparato: blobs pequenos
# perf (DOBLE DECODIFICACION): la pasada de luces decodifica el clip entero y, acto seguido, pipeline.stage_prepare
# vuelve a decodificar [f0, f1) para las poses. `tail_budget_mb` reserva ese presupuesto de memoria por clip para
# quedarse con los ULTIMOS frames completos de la misma pasada; si la ventana cae dentro, no hay segunda pasada.
# Los pixeles son identicos (mismo decodificador, lectura secuencial == lectura tras seek, verificado sobre el corpus)
# y, si la ventana NO cabe en el presupuesto, el llamante decodifica como siempre. 0 = comportamiento anterior.
TAIL_BUDGET_MB = 1024
# La ventana llega hasta [tocado - 4 s, tocado + 0.3 s] y detras suele quedar cola de clip: medido sobre 253 clips
# con tocado, (n_frames - f0) es el 42 % del clip en el percentil 10 y el 58 % en la mediana. Si en el presupuesto no
# cabe al menos esta fraccion del clip, guardar frames es memoria tirada (nunca cubriria la ventana), asi que no se
# guarda ninguno. Solo afecta a memoria/tiempo: el resultado es el mismo por cualquiera de los dos caminos.
TAIL_MIN_FRAC = 0.45


def _masks(frames: np.ndarray):
    """perf: cv2.inRange (una pasada SIMD sobre el HSV contiguo) en vez de 4-5 comparaciones numpy sobre vistas con
    paso 3 (h/s/v). Bits identicos: inRange es exactamente el AND de los intervalos cerrados por canal, y
    (h<=2)|(h>=172) es la union de dos inRange.
    rojo de LUZ = rojo puro/azulado (H<=2 o H>=172: barra FIE 173-178, punto Tokio 175-178, circulo Kazan 179/0,
    glow Incheon 177-179); las PISTAS rojas son naranja-rojo (Qatar H5-6, Turin 3-6, Katowice 6-7) -> fuera.
    verde <= 80: el suelo TURQUESA de Belgrado/Antalya tiene H~85-90 (barra ~60-66)."""
    if frames.size == 0:
        hsv = cv2.cvtColor(frames.reshape(-1, frames.shape[2], 3), cv2.COLOR_BGR2HSV).reshape(frames.shape)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        sat = (s >= S_MIN) & (v >= V_MIN)
        return sat & ((h <= 2) | (h >= 172)), sat & (h >= 40) & (h <= 80)
    hsv = cv2.cvtColor(np.ascontiguousarray(frames).reshape(-1, frames.shape[-2], 3), cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(cv2.inRange(hsv, (0, S_MIN, V_MIN), (2, 255, 255)),
                         cv2.inRange(hsv, (172, S_MIN, V_MIN), (255, 255, 255)))
    green = cv2.inRange(hsv, (40, S_MIN, V_MIN), (80, 255, 255))
    shape = frames.shape[:-1]
    return (red > 0).reshape(shape), (green > 0).reshape(shape)


class _Grow:
    """Acumulador de frames en un array preasignado (perf: sin lista + np.stack, sin pico de 2x memoria)."""
    def __init__(self, n_hint: int):
        self.n_hint = max(1, n_hint); self.a = None; self.n = 0

    def add(self, b: np.ndarray):
        if self.a is None:
            self.a = np.empty((self.n_hint,) + b.shape, np.uint8)
        elif self.n >= len(self.a):
            self.a = np.concatenate([self.a, np.empty((max(8, self.n // 2),) + b.shape, np.uint8)])
        self.a[self.n] = b; self.n += 1

    def array(self) -> np.ndarray:
        if self.a is None:
            return np.zeros((0, 1, 1, 3), np.uint8)
        if self.n < len(self.a):
            self.a.resize((self.n,) + self.a.shape[1:], refcheck=False)
        return self.a


def _read_band(clip: Clip, y0f: float, y1f: float, step: int = STEP):
    """Frames [y0f*H, y1f*H) muestreados cada `step` frames, reducidos a escala 0.5 si H > 720."""
    H, W = clip.h, clip.w
    y0, y1 = int(y0f * H), int(y1f * H)
    sc = 1.0 if H <= 720 else 0.5
    idx, out = [], _Grow((clip.n + step - 1) // step)
    for i, fr in clip.frames(step=step):
        b = fr[y0:y1]
        if sc != 1.0:
            b = cv2.resize(b, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        idx.append(i); out.add(b)
    return np.array(idx), out.array(), sc, y0


def _tail_capacity(clip: Clip, budget_mb: float) -> int:
    """Cuantos frames completos caben en `budget_mb` MB (0 = no guardar ninguno: ni cabe la ventana ni hay presupuesto).

    Nota medida (no cambiar sin volver a medirlo): lo que hay que retener no es la ventana [f0, f1) sino el SUFIJO
    [f0, n), porque el clip sigue despues del tocado; sobre 40 clips del banco la cola posterior a la ventana es de
    2,0 s de mediana y 7,2 s en el percentil 90. Por eso dimensionar el anillo con los segundos de la ventana no
    funciona: se probo y la ventana dejaba de estar cubierta en 35 de 40 clips en vez de en 14, y la segunda
    decodificacion subia del 8,9 % al 14,9 % de la CPU. Cubrir el sufijo cuesta 593 MB de mediana (1112 MB en 1080p,
    1940 MB como maximo), de ahi que en 1080p el presupuesto por defecto no llegue y se decodifique dos veces.
    """
    if not budget_mb or budget_mb <= 0 or clip.n <= 0 or clip.h <= 0 or clip.w <= 0:
        return 0
    keep = min(int(clip.n), int(float(budget_mb) * (1 << 20)) // (clip.h * clip.w * 3))
    return keep if keep >= TAIL_MIN_FRAC * clip.n else 0


def window_frames(tail: dict | None, f0: int, f1: int) -> np.ndarray | None:
    """Frames BGR [f0, f1) guardados por decode_bands_once, o None si no estan completos (el llamante decodifica).
    Bytes identicos a Clip.read_all(f0, f1): mismo decodificador, misma lectura secuencial y una unica copia al
    array contiguo (la que read_all tambien hace)."""
    if not tail:
        return None
    buf = tail.get("frames"); start = tail.get("start", 0); stop = tail.get("stop", 0)
    if buf is None or f1 <= f0 or f0 < start or f1 > stop:
        return None
    return np.stack(list(islice(buf, f0 - start, f1 - start)))


def decode_bands_once(clip: Clip, want_top: bool = True, tail: dict | None = None,
                      tail_budget_mb: float = 0.0, on_frame=None) -> dict:
    """perf: UNA sola pasada de decodificacion del clip que produce a la vez (1) la banda del overlay FIE que usa
    lights.analyze (read_band: [0.78H, H), escala 0.5 si H > 720), (2) la banda inferior del detector generico para TODOS
    los frames ([0.74H, H), misma escala; el detector usa 1 de cada STEP y _refine_onset los 2 anteriores a la muestra:
    con todos los frames no hace falta releer el video) y (3) la banda superior (1 de cada STEP) del ultimo recurso.
    Cuando la escala es 1.0 la banda FIE es una VISTA de la generica (misma memoria); a 0.5 se guardan las dos (las
    parejas 2x2 de INTER_AREA no coinciden). Los pixeles son identicos a los de read_band / _read_band.
    perf (`on_frame`): se llama con (indice, frame BGR completo) para cada frame decodificado, de modo que otro
    consumidor que necesite el video entero pueda servirse de ESTA pasada en vez de hacer la suya. Lo usa
    pipeline.stage_prepare para construir la muestra de clip_quality (clip_quality.FrameSampler), que si no obliga a
    una tercera pasada por el fichero. El frame que recibe es el mismo objeto que se usa aqui, sin copiar."""
    H = clip.h
    y0_f, sc_f = L.overlay_band_y0(H), L.band_scale(H)
    y0_g, y1_g, sc_g = int(BAND_Y0 * H), int(1.0 * H), (1.0 if H <= 720 else 0.5)
    y1_t = int(TOP_Y1 * H)
    share = sc_f == 1.0 and sc_g == 1.0 and y0_f >= y0_g and y1_g == H
    gf, gg, gt = _Grow(clip.n), _Grow(clip.n), _Grow((clip.n + STEP - 1) // STEP)
    idx_t = []
    n = 0
    keep = _tail_capacity(clip, tail_budget_mb) if tail is not None else 0
    # los frames utiles para la ventana de poses son los del FINAL (la luz esta cerca del final del clip): un deque
    # con maxlen se queda con los ultimos `keep`. cap.read() devuelve un buffer NUEVO y propio en cada llamada, asi
    # que guardar la referencia no copia nada; los que se salen del deque se liberan.
    ring = deque(maxlen=keep) if keep else None
    for i, fr in clip.frames():
        if ring is not None:
            ring.append(fr)
        if on_frame is not None:
            on_frame(i, fr)
        bg = fr[y0_g:y1_g]
        if sc_g != 1.0:
            bg = cv2.resize(bg, None, fx=sc_g, fy=sc_g, interpolation=cv2.INTER_AREA)
        gg.add(bg)
        if not share:
            bf = fr[y0_f:]
            if sc_f != 1.0:
                bf = cv2.resize(bf, None, fx=sc_f, fy=sc_f, interpolation=cv2.INTER_AREA)
            gf.add(bf)
        if want_top and i % STEP == 0:
            bt = fr[0:y1_t]
            if sc_g != 1.0:
                bt = cv2.resize(bt, None, fx=sc_g, fy=sc_g, interpolation=cv2.INTER_AREA)
            gt.add(bt); idx_t.append(i)
        n += 1
    band_g = gg.array()
    band_f = band_g[:, y0_f - y0_g:] if share else gf.array()
    if n == 0:
        band_f = np.zeros((0, 1, 1, 3), np.uint8)
    out = {"fie": (band_f, sc_f),
           "bottom": (np.arange(0, n, STEP), band_g[::STEP], sc_g, y0_g, band_g)}
    if want_top:
        out["top"] = (np.array(idx_t), gt.array(), sc_g, 0)
    if ring is not None:
        tail.update({"start": n - len(ring), "stop": n, "frames": ring})
    return out


def _sustained(mask: np.ndarray, k: int) -> np.ndarray:
    """run[t] = mask[t] & mask[t+1] & ... & mask[t+k-1] (False donde no cabe la ventana)."""
    T = len(mask)
    run = mask.copy()
    for j in range(1, k):
        run[:T - j] &= mask[j:]
    if k > 1:
        run[T - k + 1:] = False
    return run


def _candidates(idx, band, fps, side_w, tag, max_area_frac: float = MAX_AREA_FRAC):
    """Encendidos candidatos en la banda: lista de dicts (t_sample, frame, color, bbox(band px), area, cx, cy)."""
    T, bh, bw = band.shape[:3]
    if T < 6:
        return [], {}
    red, green = _masks(band)
    k = max(3, math.ceil(MIN_ON_S * fps / STEP))
    # perf: `gray` (float64: 8 B/px, ~200 MB por clip a 720p) ya NO se materializa entero. Solo hacen falta (a) `dif`
    # (booleano) y (b) `gray[t:t+k, caja]` de los pocos blobs candidatos, que se recalcula desde `band` -- la media
    # por pixel del ultimo eje conmuta con el recorte, asi que son los MISMOS float64. Evita ~0.5 GB de trafico.
    dif = np.empty((T - 1, bh, bw), bool)                        # (T-1, bh, bw): cambio fuerte entre muestras
    chg = np.empty(T, np.float64); chg[0] = 1.0                  # chg[t] = fraccion cambiada entre t-1 y t
    prev = band[0].mean(-1); d = np.empty_like(prev)
    for t in range(1, T):
        cur = band[t].mean(-1)
        np.subtract(cur, prev, out=d); np.abs(d, out=d); np.greater(d, 40, out=dif[t - 1])
        chg[t] = dif[t - 1].mean()
        prev = cur
    del prev, d
    area_band = bh * bw
    amin = max(12, 25 * (bw / 960.0) ** 2); amax = max_area_frac * area_band   # [v2] area maxima parametrizable (zona central)
    kern = np.ones((3, 3), np.uint8)
    cands = []
    for color, m in (("red", red), ("green", green)):
        run = _sustained(m, k)                                    # array propio: se reutiliza como `onset`
        notm = ~m                                                 # perf: una negacion en vez de ~m dos veces + ones_like
        onset = run
        onset[1:] &= notm[:-1]; onset[2:] &= notm[:-2]            # == run & off_before (off_before[0] siempre True)
        onset[:2] = False
        del notm
        counts = onset.reshape(T, -1).sum(1)
        for t in np.where(counts >= amin)[0]:
            if chg[t] > CUT_FRAC:
                continue                                          # corte de plano / cambio global
            # sin apertura morfologica: el glow de Incheon es un anillo de ~4 px que la apertura 3x3 destruiria; el
            # ruido de compresion lo filtra el area minima (amin)
            n, lab, stats, cents = cv2.connectedComponentsWithStats(onset[t].astype(np.uint8))
            for i in range(1, n):
                x, y, w, h, a = (int(v) for v in stats[i])
                if a < amin or a > amax or a < MIN_FILL * w * h:
                    continue
                if min(w, h) < 3:
                    continue                                      # astilla de 1-2 px: borde de pista en un paneo lento
                if x < 0.03 * bw or x + w > 0.97 * bw:
                    continue                                      # pegado al borde lateral: pared/publicidad/uniforme en un plano corto
                cx, cy = float(cents[i][0]), float(cents[i][1])
                if (color == "red" and cx > 0.5 * bw) or (color == "green" and cx < 0.5 * bw):
                    continue
                if cx < 0.13 * bw or cx > 0.87 * bw:
                    continue                                      # BANDERAS del overlay (x~0.03-0.11 / 0.89-0.97; la franja verde
                                                                  # de la bandera ITA derecha tiene cx~0.887): parpadean al
                                                                  # redibujarse el marcador. Ninguna luz tiene el centro ahi
                                                                  # (barra FIE ~0.28/0.72, Kazan 0.33/0.67, Tokio 0.46/0.54,
                                                                  # glow Incheon ~0.27/0.73)
                # OBJETO EN MOVIMIENTO (linea roja de la pista en un paneo, zapatilla, tarjeta): su borde de avance
                # 'se enciende' pero su borde de salida se APAGA a la vez. Una luz sobre pista roja / suelo turquesa
                # estaticos no apaga nada -> se compara con los pixeles del color que se apagan en el entorno.
                P0, Q0, P1, Q1 = max(0, x - 4), max(0, y - 4), min(bw, x + w + 4), min(bh, y + h + 4)
                off_cnt = int((m[t - 1, Q0:Q1, P0:P1] & ~m[t, Q0:Q1, P0:P1]).sum())
                if off_cnt > 0.5 * a:
                    continue
                # RE-ENCENDIDO de una luz que YA estaba encendida (trozo de barra ocultado por una mano/pierna que se
                # destapa, o cambio del digito del marcador junto a la barra): el entorno inmediato ya tiene mas
                # pixeles de ese color encendidos que el propio blob -> no es un encendido nuevo.
                P0, Q0, P1, Q1 = max(0, x - 6), max(0, y - 6), min(bw, x + w + 6), min(bh, y + h + 6)
                prev_on = int(m[t - 1, Q0:Q1, P0:P1].sum())
                if prev_on >= a:
                    continue
                # ENTORNO (caja dilatada sin la propia luz): (a) no cambia de golpe en t (si cambia, es que aparece el
                # overlay entero -> flags/banderas, no una luz); (b) es ESTATICO mientras la luz esta encendida
                # (std temporal baja): descarta manchas rojas/verdes sobre tiradores/arbitro/suelo descubierto.
                d = max(6, min(w, h) // 2 + 6)
                X0, Y0, X1, Y1 = max(0, x - d), max(0, y - d), min(bw, x + w + d), min(bh, y + h + d)
                blob = (lab[Y0:Y1, X0:X1] == i)
                if not (~blob).any():
                    continue
                ring_frac = float(dif[t - 1, Y0:Y1, X0:X1][~blob].mean())
                if ring_frac > RING_FRAC:
                    continue
                ring_std = float(band[t:t + k, Y0:Y1, X0:X1].mean(-1).std(0)[~blob].mean())
                if ring_std > RING_STD:
                    continue
                # [v2] brillo medio (V) del blob al encenderse: distingue los puntos oscuros de OBS (V~130-160) de las barras
                v_mean = float(cv2.cvtColor(band[t, y:y + h, x:x + w], cv2.COLOR_BGR2HSV)[..., 2]
                               [lab[y:y + h, x:x + w] == i].mean())   # perf: solo la caja del blob (lab==i vive ahi)
                cands.append({"t": int(t), "frame": int(idx[t]), "color": color, "bbox": [x, y, x + w, y + h],
                              "area": a, "cx": cx / bw, "cy": cy / bh, "ring_chg": round(ring_frac, 3),
                              "ring_std": round(ring_std, 1), "band": tag, "v_mean": round(v_mean, 1)})
    info = {"k_samples": k, "n_samples": T, "cut_samples": int((chg > CUT_FRAC).sum())}
    return cands, info


def _refine_onset(clip: Clip, cand: dict, y0: int, sc: float, band_all: np.ndarray | None = None) -> int:
    """Frame exacto del encendido: relee los STEP frames anteriores a la muestra y busca el primero con la luz.
    perf: si `band_all` (banda [y0, H) de TODOS los frames, ya escalada) esta disponible se usa en vez de releer el
    video (seek + decodificacion); mismos pixeles."""
    f = cand["frame"]; x0, y0b, x1, y1b = cand["bbox"]
    start = max(0, f - STEP + 1)
    best = f
    fr_on = None
    vals = []
    if band_all is not None:
        it = ((i, None) for i in range(start, min(f + 1, len(band_all))))
    else:
        it = clip.frames(start=start, stop=f + 1)
    for i, fr in it:
        if fr is None:
            b = band_all[i]
        else:
            b = fr[y0:]
            if sc != 1.0:
                b = cv2.resize(b, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        sub = b[y0b:y1b, x0:x1]
        if sub.size == 0:
            continue
        r, g = _masks(sub[None])
        vals.append((i, float((r if cand["color"] == "red" else g).mean())))
    if not vals:
        return f
    ref = vals[-1][1]
    for i, v in vals:
        if v >= 0.5 * ref and ref > 0:
            best = i; break
    return best


def _is_obs_dot(c: dict, bw: int, bh: int, y0: int, sc: float, H: int) -> bool:
    """[v2] Candidato con la geometria del PUNTO de Tokio 2020 (OBS): cuadrado pequeno, relleno, oscuro, junto al periodo."""
    x0, ya, x1, yb = c["bbox"]; w, h = x1 - x0, yb - ya
    lo, hi = OBS_DOT_X[c["color"]]
    cyf = (y0 + c["cy"] * bh / sc) / H
    return (OBS_DOT_W[0] <= w / bw <= OBS_DOT_W[1] and 0.6 <= w / max(h, 1) <= 1.6 and c["area"] >= 0.5 * w * h
            and lo <= c["cx"] <= hi and OBS_DOT_Y[0] <= cyf <= OBS_DOT_Y[1] and c.get("v_mean", 255) <= OBS_DOT_VMAX)


def _box_onsets(idx, band, fps, color, bbox, tag) -> list:
    """[v2] Onsets del color en una CAJA FIJA (serie de fraccion de pixeles, como lights._events): fraccion >= 0.5*max con
    max >= 0.3, < 0.25*max en las 2 muestras previas y sostenida k muestras. Sin filtros de entorno: la caja ya es la luz
    (el fondo en movimiento tras el panel semitransparente de OBS hacia fallar el filtro ring_std del blob)."""
    x0, ya, x1, yb = bbox
    sub = band[:, ya:yb, x0:x1]
    if sub.size == 0:
        return []
    r, g = _masks(sub)
    sig = (r if color == "red" else g).reshape(len(band), -1).mean(1)
    mx = float(sig.max()); T = len(sig)
    if mx < 0.3:
        return []
    k = max(3, math.ceil(MIN_ON_S * fps / STEP))
    on = sig >= 0.5 * mx
    out = []
    for t in range(2, T - k + 1):
        if on[t] and sig[t - 1] < 0.25 * mx and sig[t - 2] < 0.25 * mx and on[t:t + k].all():
            out.append({"t": int(t), "frame": int(idx[t]), "color": color, "bbox": [x0, ya, x1, yb],
                        "area": int((x1 - x0) * (yb - ya) * mx), "cx": (x0 + x1) / 2 / band.shape[2],
                        "cy": (ya + yb) / 2 / band.shape[1], "ring_chg": None, "ring_std": None, "band": tag,
                        "obs_dot": True, "frac_max": round(mx, 3)})
    return out


def _obs_mode(cands, idx, band, fps, y0, sc, H):
    """[v2] Si algun candidato es un punto OBS: solo cuentan los puntos (se descartan franjas, pista rosa, etc.) y se
    re-escanea por serie la caja de cada punto y la SIMETRICA (x -> bw - x) del companero que falte."""
    bh, bw = band.shape[1], band.shape[2]

    def off_dark(c):   # [v2] la caja del punto es casi NEGRA en las 2 muestras previas al encendido (V = max(B,G,R))
        x0, ya, x1, yb = c["bbox"]; t = c["t"]
        sub = band[max(0, t - 2):t, ya:yb, x0:x1]
        return sub.size > 0 and float(sub.max(-1).mean()) <= OBS_OFF_VMAX
    dots = [c for c in cands if _is_obs_dot(c, bw, bh, y0, sc, H) and off_dark(c)]
    if not dots:
        return cands, False
    boxes = {}
    for color in ("red", "green"):
        cs = [c for c in dots if c["color"] == color]
        if cs:
            boxes[color] = max(cs, key=lambda c: c["area"])["bbox"]
    for color, other in (("red", "green"), ("green", "red")):
        if color not in boxes and other in boxes:
            x0, ya, x1, yb = boxes[other]
            boxes[color] = [bw - x1, ya, bw - x0, yb]
    new = []
    for color, bb in boxes.items():
        ons = _box_onsets(idx, band, fps, color, bb, "bottom")
        new += ons if ons else [c for c in dots if c["color"] == color]
    return new, True


def analyze_generic(path: str | Path, allow_top: bool = True, allow_center: bool = False, _bands: dict | None = None,
                    _clip: Clip | None = None) -> dict:
    """`_bands` (perf): salida de decode_bands_once (bandas decodificadas en UNA pasada compartida con lights.analyze);
    sin ella se lee el video como antes. Mismos pixeles -> mismos resultados.
    `_clip` (perf): Clip ya construido (evita reabrir el contenedor); mismos fps/n/w/h."""
    clip = _clip if _clip is not None else Clip(path)
    fps = clip.fps or 30.0
    res = {"path": str(path), "fps": round(fps, 3), "n_frames": clip.n, "size": [clip.w, clip.h], "box_method": "generic"}
    flags = []
    band_all = None
    if _bands is not None:
        idx, band, sc, y0, band_all = _bands["bottom"]
    else:
        idx, band, sc, y0 = _read_band(clip, BAND_Y0, 1.0)
    res["band_y0"] = y0; res["band_scale"] = sc
    cands, info = _candidates(idx, band, fps, clip.w, "bottom")
    res["generic_info"] = info
    # [v2] modo OBS (Tokio 2020): puntos junto al periodo -> solo puntos + re-escaneo del companero
    cands, obs = _obs_mode(cands, idx, band, fps, y0, sc, clip.h)
    if obs:
        flags.append("overlay_obs_puntos")
    if not cands and allow_top:
        if _bands is not None and "top" in _bands:
            idx2, band2, sc2, y02 = _bands["top"]
        else:
            idx2, band2, sc2, y02 = _read_band(clip, 0.0, TOP_Y1)
        cands, info2 = _candidates(idx2, band2, fps, clip.w, "top")
        if cands:
            flags.append("luces_en_banda_superior"); y0 = y02; sc = sc2; band_all = None
    if not cands and allow_center:
        # [v2] clips SIN overlay (Turin 2020 final 720p): LEDs del aparato fisico en la zona central; solo blobs pequenos
        idx3, band3, sc3, y03 = _read_band(clip, CENTER_Y[0], CENTER_Y[1])
        cands, info3 = _candidates(idx3, band3, fps, clip.w, "center", max_area_frac=CENTER_MAX_AREA_FRAC)
        if cands:
            flags.append("luces_en_zona_central"); y0 = y03; sc = sc3; band_all = None
    res["candidates"] = cands
    if not cands:
        res.update({"error": "generico: sin encendidos rojo/verde", "t_touch_s": None, "f_touch": None,
                    "n_valid_lights": 0, "dt_ms_green_minus_red": None, "quality_flags": flags + ["sin_luces_generico"]})
        return res
    # candidatos 'enanos': blobs con area < 30 % del mayor candidato del mismo color (el resalte rojo del digito del
    # marcador al actualizarse la puntuacion segundos DESPUES del tocado, ~200 px, frente a la barra ~1700 px) no
    # pueden ser el tocado; se conservan en `candidates` pero no compiten
    big = {}
    for c in cands:
        big[c["color"]] = max(big.get(c["color"], 0), c["area"])
    cands = [c for c in cands if c["area"] >= 0.3 * big[c["color"]]]
    # tocado = ULTIMO encendido y sus companeros en los 0.5 s anteriores (P5)
    frames_on = sorted(c["frame"] for c in cands)
    f_last = frames_on[-1]
    group = [c for c in cands if c["frame"] >= f_last - int(0.5 * fps)]
    lights = {}
    for color in ("red", "green"):
        cs = [c for c in group if c["color"] == color]
        if cs:
            c = max(cs, key=lambda c: c["area"])           # el blob mayor de ese color en el grupo
            first = min(cs, key=lambda c: c["frame"])
            c = dict(c); c["frame"] = first["frame"]
            c["frame_refined"] = _refine_onset(clip, c, y0, sc, band_all)
            lights[color] = c
    f_touch = min(c["frame_refined"] for c in lights.values())

    def to_full(b):
        return [int(b[0] / sc), int(b[1] / sc) + y0, int(b[2] / sc), int(b[3] / sc) + y0]
    def fmt(f): return None if f is None else round(f / fps, 3)
    fr_r = lights["red"]["frame_refined"] if "red" in lights else None
    fr_g = lights["green"]["frame_refined"] if "green" in lights else None
    dt = None if (fr_r is None or fr_g is None) else round(1000 * (fr_g - fr_r) / fps)
    n_valid = len(lights)
    res.update({
        "boxes": {"L": to_full(lights["red"]["bbox"]) if "red" in lights else None,
                  "R": to_full(lights["green"]["bbox"]) if "green" in lights else None},
        "events": {"red_L": {"onsets_s": [fmt(c["frame"]) for c in cands if c["color"] == "red"]},
                   "green_R": {"onsets_s": [fmt(c["frame"]) for c in cands if c["color"] == "green"]}},
        "t_touch_s": fmt(f_touch), "f_touch": int(f_touch),
        "touch_lights": {"red_L": fmt(fr_r), "green_R": fmt(fr_g), "white_L": None, "white_R": None},
        "n_valid_lights": n_valid, "dt_ms_green_minus_red": dt,
        "first_light": (None if dt is None else ("left" if dt > 0 else "right" if dt < 0 else "same_frame")),
    })
    if dt is not None and abs(dt) > L.MAX_DT_MS:
        flags.append(f"dt_{abs(dt)}ms_supera_bloqueo")
    if f_touch < int(1.0 * fps):
        flags.append("tocado_en_el_primer_segundo")
    n_other = len([c for c in cands if c["frame"] < f_last - int(0.5 * fps)])
    if n_other:
        flags.append(f"otros_encendidos_previos_{n_other}")
    if info.get("cut_samples", 0) > 0:
        flags.append("cortes_de_plano")
    res["quality_flags"] = flags + ["luces_por_detector_generico"]
    return res


def analyze_any(path: str | Path, clock: bool = False, shared_decode: bool = True,
                tail: dict | None = None, tail_budget_mb: float = 0.0, on_frame=None) -> dict:
    """lights.analyze y, si falla o no encuentra tocado, el detector generico (independiente del overlay).
    perf (`shared_decode`): el video se decodifica UNA vez (decode_bands_once) y las bandas se pasan a lights.analyze y a
    analyze_generic; antes el generico releia todo el clip (+1-2 seeks en _refine_onset). Mismos pixeles/resultados.
    perf (`tail`): dict vacio en el que se devuelven los frames COMPLETOS del final del clip decodificados en esa misma
    pasada (ver decode_bands_once / window_frames); con el, pipeline.stage_prepare obtiene la ventana de poses sin
    decodificar el video por segunda vez.
    perf (`on_frame`): callback por frame decodificado, para que otro consumidor del video completo (la muestra de
    clip_quality) se sirva de esta pasada en vez de hacer otra."""
    bands = None
    clip = None
    if shared_decode:
        try:
            clip = Clip(path)                                # perf: un solo Clip para toda la cadena (abre el
            bands = decode_bands_once(clip, tail=tail, tail_budget_mb=tail_budget_mb,   # contenedor una vez, no tres)
                                      on_frame=on_frame)
        except Exception:                                    # pragma: no cover
            bands = None; clip = None
            if tail is not None:
                tail.clear()
    if tail is not None and clip is not None:
        tail["clip"] = clip                                  # el llamante lo reutiliza para la ventana de poses
    try:
        res = L.analyze(path, clock=clock, _band=bands["fie"] if bands else None, _clip=clip) if clock else L.analyze(path, _band=bands["fie"] if bands else None, _clip=clip)
    except Exception as e:                                   # pragma: no cover
        res = {"path": str(path), "error": repr(e), "t_touch_s": None}
    if res.get("f_touch") is not None:
        res["source"] = "fie"
        return res
    gen = analyze_generic(path, _bands=bands, _clip=clip)
    gen["fie_error"] = res.get("error"); gen["fie_flags"] = res.get("quality_flags", [])
    gen["source"] = "generic" if gen.get("f_touch") is not None else "none"
    return gen


def to_json(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = analyze_generic(p)
        print(json.dumps({k: v for k, v in to_json(r).items() if k != "candidates"}, ensure_ascii=False, indent=1))
        for c in r.get("candidates", []):
            print("  ", c)

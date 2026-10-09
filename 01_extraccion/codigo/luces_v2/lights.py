"""Detector de luces v2 sobre el overlay FIE -- COPIA PARCHEADA (agents/lights36). NO sustituye al repo.

Cambios respecto a fencingly_v2/lights.py (ver REPORT.md):
 P1  _find_name_boxes: (a) descarta componentes 'gigantes' (pista azul estatica, El Cairo 2022); (b) parte por el
     centro el componente unico L+R (Anaheim 2018/2019: la banda azul continua bajo el reloj); (c) 'aprieta' la caja
     al perfil de filas/columnas (sin la pestana '1/3' ni el reloj); (d) valida contra la geometria canonica del
     overlay FIE (identica en los 8 torneos) y, si falla, usa esa geometria como fallback verificado por color.
 P2  Ventanas temporales para el mapa estatico: clip entero -> primer 45 % -> ultimo 45 % (clips con replay sin
     overlay: FatherlyWeeklyAcouchi) + compuerta por frame 'overlay presente' (fraccion azul en las cajas).
 P3  ROI de la luz = FRANJA bajo la caja [y1, y1+0.022H] (la barra se dibuja debajo del nombre, no dentro), sin el
     extremo diagonal (ancho = altura de la caja); mascaras rojo/verde con V>180 (la pista roja de Belgrado tiene
     V~145-160, la barra V~230-255).
 P4  _events: onset = escalon (salto >= 35 % del maximo en <= 3 frames) y no simple cruce de umbral (deriva lenta de
     la pista = falso onset, AbandonedSaneCrow).
 P5  Tocado = ULTIMO onset de cualquier luz valida y sus companeros en los 0.5 s anteriores (antes: min de los
     ultimos onsets de cada canal -> mezclaba la luz del tocado anterior con la del tocado, adorableMudiLobster).
 P6  Luz blanca: el cuadradito blanco (tocado NO valido) aparece bajo el marcador de cada tirador, en x~0.38W (L) y
     x~0.62W (R), filas [y1+0.027H, y1+0.058H]; la zona antigua (entre las cajas, bajo el reloj) no lo cubria nunca.
     Deteccion por blob del tamano del indicador -> white_L / white_R sin falsos positivos por lineas/calcetines/logo.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import cv2
from .video import Clip, overlay_band_y0

MAX_DT_MS = 350   # bloqueo del florete = 300 ms: un dt mayor entre dos luces 'validas' es un falso onset

BAND_FRAC = 0.78
# geometria canonica del overlay FIE (fraccion de W, H del frame): cajas de nombre L y R (medida en 8 torneos 2018-23)
PRIOR_L = (0.120, 0.852, 0.470, 0.910)
PRIOR_R = (0.530, 0.852, 0.880, 0.910)
V_LIGHT = 180          # brillo minimo de la barra roja/verde (pista roja de Belgrado: V<=165)
STRIP_H = 0.022        # altura de la franja de la barra bajo la caja (fraccion de H)
WHITE_L = (0.335, 0.425)   # ventana x (fraccion de W) del cuadradito blanco izquierdo (cuadrado en ~0.365-0.395 W)
WHITE_R = (0.575, 0.665)   # idem derecho (~0.61-0.64 W)
GREEN_FLOOR = 0.25     # [v2] fraccion verde minima de la franja para aceptar onsets (suelo turquesa de Antalya: 0.155)


def _inrange(bgr: np.ndarray, lo, hi) -> np.ndarray:
    """perf: mascara booleana (lo <= HSV <= hi, por canal) con UNA pasada SIMD de cv2.inRange en vez de 3-4
    comparaciones numpy sobre vistas con paso 3. Bits identicos: inRange es exactamente el AND de los intervalos
    cerrados por canal. `bgr` puede ser (..., h, w, 3): se aplana a (N*h, w, 3) (cvtColor es por pixel)."""
    flat = np.ascontiguousarray(bgr).reshape(-1, bgr.shape[-2], 3)
    m = cv2.inRange(cv2.cvtColor(flat, cv2.COLOR_BGR2HSV), lo, hi)
    return (m > 0).reshape(bgr.shape[:-1])


def _white_square(roi: np.ndarray, W: int, Hs: float) -> np.ndarray:
    """(T,) 0/1: hay un blob blanco puro con la forma del indicador: rectangulo redondeado de ~0.029W x 0.033H
    (720p: 37x24 px; 1080p/2: 28x19), relleno >= 0.85, w/h 1.2-2.0, V >= 236, sin tocar el borde lateral de la ventana.
    (Suelo claro: V~217, relleno 0.7, ratio < 1 -> se descarta)."""
    if roi.size == 0:                       # caso degenerado: se conserva el comportamiento (y la excepcion) original
        hsv = cv2.cvtColor(roi.reshape(-1, roi.shape[2], 3), cv2.COLOR_BGR2HSV).reshape(roi.shape)
        wm = (hsv[..., 1] < 40) & (hsv[..., 2] > 235)
    else:
        wm = _inrange(roi, (0, 0, 236), (255, 39, 255))
    T = len(roi); out = np.zeros(T, np.float32); rw = roi.shape[2]
    wmin, wmax = 0.018 * W, 0.045 * W; hmin, hmax = 0.02 * Hs, 0.055 * Hs
    # perf: un blob valido necesita >= 0.85*wmin*hmin pixeles blancos; los frames que no llegan a ese total NO pueden
    # tener ninguno (mismo resultado, sin etiquetar componentes). En la practica casi ningun frame tiene el cuadrado.
    need = 0.85 * wmin * hmin
    counts = wm.reshape(T, -1).sum(1)
    for t in range(T):
        if counts[t] < need:
            continue
        n, _, stats, _ = cv2.connectedComponentsWithStats(wm[t].astype(np.uint8))
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if (wmin <= w <= wmax and hmin <= h <= hmax and a >= 0.85 * w * h and 1.2 <= w / h <= 2.0
                    and x > 0 and x + w < rw):
                out[t] = 1.0; break
    return out


def _hsv_masks(frames_bgr: np.ndarray, v_min: int = V_LIGHT):
    """perf: una sola conversion HSV del bloque (T,h,w,3) + cv2.inRange por mascara, en vez de un bucle Python con
    np.stack y 3-4 comparaciones numpy por mascara. Mismos bits (cvtColor es por pixel; inRange = AND de intervalos)."""
    if frames_bgr.size == 0:
        hsv = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in frames_bgr])
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        return (((h < 10) | (h > 170)) & (s > 120) & (v > v_min), (h > 40) & (h < 90) & (s > 100) & (v > v_min),
                (s < 40) & (v > 215))
    shape = frames_bgr.shape[:-1]
    flat = np.ascontiguousarray(frames_bgr).reshape(-1, frames_bgr.shape[-2], 3)
    hsv = cv2.cvtColor(flat, cv2.COLOR_BGR2HSV)
    vlo = min(int(v_min) + 1, 255)
    red = cv2.bitwise_or(cv2.inRange(hsv, (0, 121, vlo), (9, 255, 255)),
                         cv2.inRange(hsv, (171, 121, vlo), (255, 255, 255)))
    green = cv2.inRange(hsv, (41, 101, vlo), (89, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 216), (255, 39, 255))
    return (red > 0).reshape(shape), (green > 0).reshape(shape), (white > 0).reshape(shape)


BLUE_H0 = 80           # [v2] tono minimo de las cajas: 95 -> 80 para las cajas TURQUESA de Antalya/eurochamps2022 (H~85-92)


BLUE_LO, BLUE_HI = (BLUE_H0 + 1, 61, 41), (129, 255, 255)   # perf: mismos umbrales, en forma de cv2.inRange


def _blue_mask(bgr: np.ndarray) -> np.ndarray:
    return _inrange(bgr, BLUE_LO, BLUE_HI)


def _static_stats(band: np.ndarray, f0: int = 0, f1: int | None = None):
    """Media y std temporal (por pixel) de band[f0:f1:3] con acumuladores (memoria = 2 frames)."""
    sub = band[f0:f1:3]
    n = len(sub); s1 = np.zeros(sub.shape[1:], np.float32); s2 = np.zeros(sub.shape[1:], np.float32)
    if sub.size:
        for f in sub:                                     # perf: cv2.accumulate / accumulateSquare hacen
            cv2.accumulate(f, s1)                         # dst += (float)src y dst += (float)src*(float)src en una
            cv2.accumulateSquare(f, s2)                   # pasada SIMD, sin materializar el frame en float32:
                                                          # mismo orden y mismas operaciones float32 -> mismos bits.
    mean = s1 / max(n, 1)
    std = np.sqrt(np.maximum(s2 / max(n, 1) - mean * mean, 0)).mean(-1)
    return np.clip(mean, 0, 255).astype(np.uint8), std


def _tighten(mask: np.ndarray, box, frac: float = 0.5):
    """Recorta la caja al nucleo del componente: filas con >= frac*max pixeles, luego columnas idem."""
    x0, y0, x1, y1 = box
    sub = mask[y0:y1, x0:x1]
    if sub.size == 0 or sub.max() == 0:
        return box
    rows = sub.sum(1); keep = np.where(rows >= frac * rows.max())[0]
    ry0, ry1 = int(keep.min()), int(keep.max()) + 1
    cols = sub[ry0:ry1].sum(0); keepc = np.where(cols >= frac * cols.max())[0]
    cx0, cx1 = int(keepc.min()), int(keepc.max()) + 1
    return [x0 + cx0, y0 + ry0, x0 + cx1, y0 + ry1]


def _prior_boxes(W: int, Hs: float, y0s: float):
    """Cajas canonicas en coordenadas de la banda reducida (W = ancho banda, Hs = alto frame reducido, y0s = inicio banda)."""
    def bx(p): return [int(p[0] * W), int(p[1] * Hs - y0s), int(p[2] * W), int(p[3] * Hs - y0s)]
    return bx(PRIOR_L), bx(PRIOR_R)


def _iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _find_name_boxes(band: np.ndarray, W: int, Hs: float, y0s: float, f0: int = 0, f1: int | None = None):
    """Cajas de nombre = componentes estaticas (std temporal baja) y azules, anchas, una a cada lado del centro.
    Devuelve (bl, br, method)."""
    med, std = _static_stats(band, f0, f1)
    bh = band.shape[1]
    blue = _blue_mask(med)
    m = cv2.morphologyEx(((std < 12) & blue).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 15), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m)
    pl, pr = _prior_boxes(W, Hs, y0s)
    boxes = []
    method = "detected"
    for i in range(1, n):
        s = stats[i]
        x0, y0, w, h = int(s[0]), int(s[1]), int(s[2]), int(s[3])
        if s[cv2.CC_STAT_AREA] < 800 * (W / 1280) ** 2 or w < W * 0.12:
            continue
        if h > 0.5 * bh or w > 0.85 * W:          # P1a: pista azul estatica (o pista + overlay fundidos): no es una caja
            continue
        b = [x0, y0, x0 + w, y0 + h]
        if x0 < 0.45 * W and x0 + w > 0.55 * W:   # P1b: L+R fundidas por el centro -> partir en la columna menos densa
            cols = (lab[y0:y0 + h] == i).sum(0)
            c0, c1 = int(0.42 * W), int(0.58 * W)
            split = c0 + int(np.argmin(cols[c0:c1]))
            boxes.append(_tighten(m > 0, [x0, y0, split, y0 + h])); boxes.append(_tighten(m > 0, [split, y0, x0 + w, y0 + h]))
            method = "split"
        else:
            boxes.append(_tighten(m > 0, b))     # P1c: sin pestana '1/3' ni reloj
    left = [b for b in boxes if (b[0] + b[2]) / 2 < W / 2]; right = [b for b in boxes if (b[0] + b[2]) / 2 >= W / 2]
    # P1d: validar contra la geometria canonica (misma en los 8 torneos): IoU con la caja prior
    left = [b for b in left if _iou(b, pl) > 0.3]; right = [b for b in right if _iou(b, pr) > 0.3]
    if left and right:
        pick = lambda cs: max(cs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        return pick(left), pick(right), method
    # fallback: cajas canonicas si su interior es azul y estatico (>= 35 % de pixeles)
    ok = []
    for p in (pl, pr):
        sub = ((std < 12) & blue)[p[1]:p[3], p[0]:p[2]]
        ok.append(sub.size > 0 and sub.mean() > 0.35)
    if all(ok):
        return pl, pr, "prior"
    return None, None, "none"


def _events(sig: np.ndarray, thr_rel: float = 0.5, min_on: int = 5, floor: float = 0.02, step_rel: float = 0.35):
    T = len(sig)
    if T == 0 or sig.max() < floor:
        return {"on_at_start": False, "onsets": [], "offsets": [], "on": np.zeros(T, bool)}
    mx = sig.max()
    on = sig > thr_rel * mx
    onsets = []
    for t in range(1, T):
        if on[t] and not on[t - 1] and on[t:t + min_on].all():
            jump = sig[min(T - 1, t + 1)] - sig[max(0, t - 3)]      # P4: la barra se enciende de golpe (escalon)
            if jump >= step_rel * mx:
                onsets.append(t)
    offsets = [t for t in range(1, T) if (not on[t]) and on[t - 1] and (~on[t:t + min_on]).all()]
    return {"on_at_start": bool(on[:3].all()), "onsets": onsets, "offsets": offsets, "on": on}


def band_scale(h: int) -> float:
    return 1.0 if h <= 720 else 0.5


def read_band(clip: Clip, scale: float | None = None) -> tuple[np.ndarray, float]:
    """perf: array preasignado (sin lista + np.stack: -1 copia, sin pico de 2x memoria). Misma salida."""
    y0 = overlay_band_y0(clip.h)
    scale = scale or band_scale(clip.h)
    out = None; n = 0
    for _, fr in clip.frames():
        b = fr[y0:]
        b = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else b
        if out is None:
            out = np.empty((max(1, clip.n),) + b.shape, np.uint8)
        if n >= len(out):                                        # mas frames que CAP_PROP_FRAME_COUNT
            out = np.concatenate([out, np.empty((max(8, n // 2),) + b.shape, np.uint8)])
        out[n] = b; n += 1
    if out is None:
        return np.zeros((0, 1, 1, 3), np.uint8), scale
    if n < len(out):
        out.resize((n,) + out.shape[1:], refcheck=False)
    return out, scale


def analyze(path: str | Path, clock: bool = True, _band: tuple[np.ndarray, float] | None = None,
            _clip: Clip | None = None) -> dict:
    """`_band` (perf): banda ya decodificada (band, scale) tal como la devuelve read_band; evita decodificar el video
    otra vez cuando lights_generic.analyze_any comparte una unica pasada de decodificacion.
    `_clip` (perf): Clip ya construido; Clip.__init__ abre el contenedor (~10-30 ms) y en la cadena
    analyze_any -> analyze -> analyze_generic se abria 3 veces el mismo fichero. Mismos fps/n/w/h."""
    clip = _clip if _clip is not None else Clip(path)
    y0 = overlay_band_y0(clip.h)
    band, sc = _band if _band is not None else read_band(clip)
    T, W = len(band), band.shape[2]
    Hs = clip.h * sc; y0s = y0 * sc
    res = {"path": str(path), "fps": round(clip.fps, 3), "n_frames": T, "size": [clip.w, clip.h], "band_y0": y0, "band_scale": sc}
    # P2: mapa estatico sobre el clip entero; si no hay cajas, sobre el primer / ultimo 45 % (replay sin overlay)
    bl = br = None; method = "none"; win = None
    for (f0, f1, name) in ((0, None, "all"), (0, int(0.45 * T), "first45"), (int(0.55 * T), None, "last45")):
        if T and f1 is not None and f1 - f0 < 10:
            continue
        bl, br, method = _find_name_boxes(band, W, Hs, y0s, f0, f1)
        if bl is not None:
            win = name; break
    res["box_method"] = method; res["static_window"] = win
    if bl is None:
        res.update({"error": "no se localizan las cajas del overlay", "t_touch_s": None})
        return res

    # compuerta por frame: overlay presente si el interior de ambas cajas es mayoritariamente azul
    def blue_frac(bx):
        x0, yy0, x1, yy1 = bx
        sub = band[:, yy0:yy1, x0:x1]                                   # (T,h,w,3) -> una sola conversion HSV
        m = _inrange(sub, BLUE_LO, BLUE_HI)                             # [v2] misma mascara (turquesa), via inRange
        return m.reshape(T, -1).mean(1)
    present = (blue_frac(bl) > 0.3) & (blue_frac(br) > 0.3)
    res["overlay_present_frac"] = round(float(present.mean()), 3)

    def strip(bx, side):
        x0, yy0, x1, yy1 = bx; h = yy1 - yy0
        xs = (x0, x1 - h) if side == "L" else (x0 + h, x1)          # P3: sin el extremo diagonal (junto al reloj)
        ys = (yy1, min(band.shape[1], yy1 + max(4, int(STRIP_H * Hs))))
        return band[:, ys[0]:ys[1], xs[0]:xs[1]]

    r, _, _ = _hsv_masks(strip(bl, "L")); red = r.reshape(T, -1).mean(1) * present
    _, g, _ = _hsv_masks(strip(br, "R")); green = g.reshape(T, -1).mean(1) * present
    # P6: luz BLANCA = cuadradito blanco (~0.03W x 0.03H) bajo el marcador de cada tirador, centrado en x~0.38W (L) /
    # x~0.62W (R) y en filas [y1+0.027H, y1+0.058H] (identico en los 8 torneos). Deteccion por blob (tamano y
    # relleno), no por fraccion de pixeles: inmune a lineas de pista, calcetines y al logo Tissot (Anaheim).
    cy0 = max(bl[3], br[3]) + int(0.01 * Hs); cy1 = min(band.shape[1], cy0 + int(0.075 * Hs))
    wl = _white_square(band[:, cy0:cy1, int(WHITE_L[0] * W):int(WHITE_L[1] * W)], W, Hs) * present
    wr = _white_square(band[:, cy0:cy1, int(WHITE_R[0] * W):int(WHITE_R[1] * W)], W, Hs) * present
    wx0, wx1 = int(WHITE_L[0] * W), int(WHITE_R[1] * W)
    # [v2] floor del verde 0.15 -> GREEN_FLOOR: el suelo TURQUESA de Antalya bajo la caja derecha llega a 0.155 y generaba
    # un onset verde espurio (TenseSerpentineBobcat); una barra verde real llena >= 0.3 de la franja
    ev = {"red_L": _events(red, floor=0.15), "green_R": _events(green, floor=GREEN_FLOOR),
          "white_L": _events(wl, floor=0.5), "white_R": _events(wr, floor=0.5)}

    fps = clip.fps
    def fmt(t): return None if t is None else round(t / fps, 3)
    # P5: el tocado es el ULTIMO encendido de una luz valida; sus companeros son los onsets en los 0.5 s anteriores
    all_on = sorted(o for k in ("red_L", "green_R") for o in ev[k]["onsets"])
    f_touch = None
    if all_on:
        f_last = all_on[-1]
        f_touch = min(o for o in all_on if o >= f_last - int(0.5 * fps))
    lights_on = {}
    if f_touch is not None:
        lo, hi = f_touch - int(0.5 * fps), f_touch + int(0.5 * fps)
        for k in ev:
            near = [o for o in ev[k]["onsets"] if lo <= o <= hi]
            lights_on[k] = min(near) if near else None
    n_valid = sum(1 for k in ("red_L", "green_R") if lights_on.get(k) is not None)
    dt = None
    if lights_on.get("red_L") is not None and lights_on.get("green_R") is not None:
        dt = round(1000 * (lights_on["green_R"] - lights_on["red_L"]) / fps)

    def to_full(b):
        return [int(b[0] / sc), int(b[1] / sc) + y0, int(b[2] / sc), int(b[3] / sc) + y0]
    out = {"boxes": {"L": to_full(bl), "R": to_full(br)}, "boxes_band": {"L": bl, "R": br},
           "white_roi": to_full([wx0, cy0, wx1, cy1]),
           "events": {k: {"on_at_start": v["on_at_start"], "onsets_s": [fmt(t) for t in v["onsets"]],
                          "offsets_s": [fmt(t) for t in v["offsets"]]} for k, v in ev.items()},
           "t_touch_s": fmt(f_touch), "f_touch": f_touch,
           "touch_lights": {k: fmt(v) for k, v in lights_on.items()},
           "n_valid_lights": n_valid, "dt_ms_green_minus_red": dt,
           "first_light": (None if dt is None else ("left" if dt > 0 else "right" if dt < 0 else "same_frame")),
           "series_max": {"red_L": round(float(red.max()), 3), "green_R": round(float(green.max()), 3),
                          "white_L": round(float(wl.max()), 3), "white_R": round(float(wr.max()), 3)}}
    res.update(out)
    res["_on"] = {k: v["on"] for k, v in ev.items()}
    res["_series"] = {"red_L": red, "green_R": green, "white_L": wl, "white_R": wr, "present": present}
    if clock:
        cx0, cx1 = bl[2], br[0]; ccy0 = min(bl[3], br[3])
        res["clock_running_s"] = _clock_interval(band, cx0, cx1, ccy0, Hs, fps)
    # avisos de calidad (no bloquean; el consumidor decide): dt imposible, cajas por geometria a priori, replay parcial
    flags = []
    if dt is not None and abs(dt) > MAX_DT_MS:
        flags.append(f"dt_{abs(dt)}ms_supera_bloqueo")
    if method == "prior":
        flags.append("cajas_por_geometria_prior")
    if res["overlay_present_frac"] < 0.8:
        flags.append("overlay_parcial_replay")
    if f_touch is None and clock and res.get("clock_running_s") is None:
        flags.append("sin_luces_ni_reloj_revision_video")
    if f_touch is not None and f_touch < int(1.0 * fps):
        flags.append("tocado_en_el_primer_segundo")
    res["quality_flags"] = flags
    return res


def _clock_interval(band, cx0, cx1, cy0, H, fps):
    clk = band[:, max(0, cy0 - int(0.08 * H)):cy0, cx0:cx1, :].astype(np.float32).mean(-1)
    d = np.abs(np.diff(clk, axis=0)).mean((1, 2))
    if len(d) < 3:
        return None
    changes = [t + 1 for t in range(len(d)) if d[t] > max(4.0, 6 * np.median(d))]
    if len(changes) < 2:
        return None
    gaps = np.diff(changes) / fps
    ok = (gaps > 0.7) & (gaps < 1.4)
    if not ok.any():
        return None
    i0 = int(np.argmax(ok)); i1 = len(ok) - int(np.argmax(ok[::-1]))
    return [round(max(0, changes[i0] - fps) / fps, 2), round(changes[i1] / fps, 2)]


def channels(res: dict, n_frames: int | None = None) -> np.ndarray:
    keys = ("red_L", "green_R", "white_L", "white_R")
    T = n_frames or res["n_frames"]
    if "_on" not in res:
        return np.zeros((T, 4), np.float32)
    return np.stack([res["_on"][k][:T].astype(np.float32) for k in keys], axis=1)


PRE_S, POST_S = 4.0, 0.3        # ventana por defecto (agents/window_study: 2.5 s recortaba el inicio de la accion final en el 24 % de los clips)
CLOCK_BIAS_S = 0.5              # clock_running_s[0] = primer tick - 1 s (cota temprana); el 'allez' real esta ~0.5 s despues
MIN_PRE_S = 1.5                 # nunca dejar menos de 1.5 s antes de la luz aunque el reloj diga que el allez fue mas tarde


def allez_frame(res: dict, clock_bias_s: float = CLOCK_BIAS_S) -> int | None:
    """Frame aproximado del 'allez' = inicio del reloj corriendo + sesgo. None si no hay reloj legible o si el reloj ya
    corria al empezar el clip (clock_running_s[0] ~ 0: allez censurado)."""
    clk = res.get("clock_running_s")
    if not clk or len(clk) != 2 or clk[0] is None or float(clk[0]) <= 0.05:
        return None
    return int(round((float(clk[0]) + clock_bias_s) * res["fps"]))


def window(res: dict, pre_s: float = PRE_S, post_s: float = POST_S, use_clock: bool = True,
           min_pre_s: float = MIN_PRE_S) -> tuple[int, int] | None:
    """[max(allez, tocado - pre_s), tocado + post_s]. Con `use_clock` el inicio se recorta al 'allez' del reloj (si es
    legible y no censurado) sin bajar de `min_pre_s` antes de la luz; sin reloj, ventana fija de pre_s. Los npz antiguos
    de 2.8 s siguen siendo validos: el store y pad_batch alinean por el final y enmascaran la izquierda."""
    if res.get("f_touch") is None:
        return None
    fps, T, ft = res["fps"], res["n_frames"], res["f_touch"]
    f0 = max(0, int(ft - pre_s * fps)); f1 = min(T, int(ft + post_s * fps) + 1)
    res["window_source"] = "fixed"
    if use_clock:
        fa = allez_frame(res)
        if fa is not None and fa > f0:
            f0 = max(0, min(fa, int(ft - min_pre_s * fps))); res["window_source"] = "clock"
    res["f_allez"] = allez_frame(res)
    return f0, f1


def to_json(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


def analyze_many(paths, out_json: str | Path | None = None) -> dict:
    out = {}
    for p in paths:
        try:
            out[Path(p).name] = to_json(analyze(p))
        except Exception as e:
            out[Path(p).name] = {"error": repr(e), "t_touch_s": None}
    if out_json:
        Path(out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out

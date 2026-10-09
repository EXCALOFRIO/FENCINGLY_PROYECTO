"""Utilidades de video: descarga (con cabeceras de navegador), lectura de frames y recorte anti-fuga."""
from __future__ import annotations
import os, time, urllib.request
from pathlib import Path
import numpy as np
import cv2

HEADERS = {"Referer": "https://fencingdatabase.com/",
           "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128 Safari/537.36",
           "Accept": "video/mp4,video/*;q=0.9,*/*;q=0.8"}


def download_clip(url: str, dest_dir: str | Path, retries: int = 2, pause: float = 1.0) -> Path | None:
    """Descarga un clip. NOTA: clips.fencingdatabase.com bloquea la descarga masiva (403) -> pedir acceso al autor
    (Scott Dubinsky) o usar clips propios en Drive. Devuelve la ruta o None si falla."""
    dest_dir = Path(dest_dir); dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / url.rsplit("/", 1)[-1]
    if out.is_file() and out.stat().st_size > 10_000:
        return out
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as r, open(out, "wb") as f:
                f.write(r.read())
            return out
        except Exception:
            time.sleep(pause)
    return None


class Clip:
    """Lector de frames con cache opcional en memoria (los clips son cortos: 10-13 s)."""
    def __init__(self, path: str | Path):
        self.path = str(path)
        cap = cv2.VideoCapture(self.path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    def frames(self, start: int = 0, stop: int | None = None, step: int = 1):
        """Genera (idx, frame BGR) secuencialmente (mas rapido y fiable que seek por frame).
        perf: los frames saltados (step > 1) se decodifican con grab() sin retrieve(): se ahorra la conversion YUV->BGR
        del frame entero (misma salida: los frames devueltos son identicos)."""
        cap = cv2.VideoCapture(self.path)
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        i = start
        stop = self.n if stop is None else min(stop, self.n)
        while i < stop:
            if (i - start) % step == 0:
                ok, fr = cap.read()
                if not ok:
                    break
                yield i, fr
            elif not cap.grab():
                break
            i += 1
        cap.release()

    def read_all(self, start=0, stop=None, step=1) -> tuple[list[int], np.ndarray]:
        """perf: array preasignado (T,H,W,3) en vez de lista + np.stack (evita una copia y el pico de 2x memoria)."""
        stop_ = self.n if stop is None else min(stop, self.n)
        n_max = max(0, (stop_ - start + step - 1) // step)
        out = np.empty((n_max, self.h, self.w, 3), np.uint8) if n_max else None
        idx = []
        for i, fr in self.frames(start, stop, step):
            out[len(idx)] = fr; idx.append(i)             # (tamano distinto al anunciado -> ValueError, como np.stack)
        if not idx:
            return idx, np.zeros((0, self.h, self.w, 3), np.uint8)
        if len(idx) < n_max:
            out.resize((len(idx), self.h, self.w, 3), refcheck=False)   # menos frames de los anunciados: encoger in situ
        return idx, out


def estimate_pan(frames: np.ndarray, band=(0.05, 0.40), scale: float = 0.5) -> np.ndarray:
    """Paneo horizontal de la camara por frame (px, acumulado) via correlacion de fase sobre la banda SUPERIOR del
    frame (fondo: publico, marcadores; los tiradores estan abajo). Convencion (verificada midiendo el residuo del
    fondo estabilizado): x_pista = x_imagen + pan. Es decir, SUMAR pan a las x absolutas fija el fondo."""
    T, H, W = frames.shape[:3]
    y0, y1 = int(H * band[0]), int(H * band[1])
    prev = None; dx = np.zeros(T, np.float64)
    win = None
    for t in range(T):
        g = cv2.cvtColor(frames[t, y0:y1], cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=scale, fy=scale).astype(np.float32)
        if win is None:
            win = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
        if prev is not None:
            (sx, sy), resp = cv2.phaseCorrelate(prev, g, win)
            dx[t] = dx[t - 1] + (-sx / scale if resp > 0.05 else 0.0)
        prev = g
    return dx.astype(np.float32)


def duplicate_frames(frames: np.ndarray, thr: float = 0.35) -> np.ndarray:
    """Marca frames DUPLICADOS (identicos al anterior). Muchos clips vienen de broadcast a 25 fps convertido a 30 fps:
    1 de cada 6 frames esta repetido (Anaheim 2018, Tbilisi 2023: ~17 %). Un duplicado da velocidad cero y luego un
    salto doble: hay que tratarlo como frame ausente (se interpola) y no como movimiento."""
    T = len(frames)
    dup = np.zeros(T, bool)
    prev = None
    for t in range(T):
        g = cv2.cvtColor(cv2.resize(frames[t], (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.int16)
        if prev is not None and np.abs(g - prev).mean() < thr:
            dup[t] = True
        prev = g
    return dup


def detect_cuts(frames: np.ndarray, size=(160, 90), hist_thr: float = 0.30, hist_soft: float = 0.18,
                motion_soft: float = 0.5, resp_min: float = 0.03, hist_resp: float = 0.12, motion_resp: float = 0.35,
                overlay_frac: float = 0.78, return_series: bool = False):
    """pose_qc2: CORTES de camara dentro de una ventana de frames (T,H,W,3). Mismo criterio que clip_quality
    (agents/clip_quality: 31 clips con cortes, 8.8 %), pero frame a frame: en gris `size` (160x90) sin la banda del
    overlay, un frame t es corte si (a) la distancia de histogramas (L1/2, 32 bins) con t-1 > hist_thr, o (b) > hist_soft
    y ademas cambia > motion_soft de los pixeles tras compensar el paneo (correlacion de fase), o (c) la correlacion de
    fase FALLA (respuesta < resp_min: no hay desplazamiento que alinee el fondo) con histograma > hist_resp y > motion_resp
    de pixeles cambiados. Un paneo rapido con desenfoque alinea (resp alta, pocos pixeles cambiados tras compensar) y NO
    cuenta; un plano corto tras el tocado o un plano de la grada si. Devuelve un array int con los frames t (t = PRIMER
    frame del plano nuevo); frames consecutivos marcados (fundido de 2-3 frames) se agrupan en el primero. Con
    return_series=True devuelve ademas dict(hist, resp, changed)."""
    T = len(frames)
    if T < 2:
        return (np.zeros(0, np.int64), {}) if return_series else np.zeros(0, np.int64)
    top = int(size[1] * overlay_frac)
    win = cv2.createHanningWindow((size[0], top), cv2.CV_32F)
    hist_d = np.zeros(T, np.float32); resp_s = np.ones(T, np.float32); changed = np.zeros(T, np.float32)
    prev = None; prev_h = None
    for t in range(T):
        g = cv2.cvtColor(cv2.resize(frames[t], size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)[:top]
        h = cv2.calcHist([g], [0], None, [32], [0, 256]).ravel(); h /= max(1.0, h.sum())
        if prev is not None:
            hist_d[t] = 0.5 * float(np.abs(h - prev_h).sum())
            (sx, sy), resp = cv2.phaseCorrelate(prev.astype(np.float32), g.astype(np.float32), win)
            resp_s[t] = resp
            if resp < resp_min or abs(sx) > 0.5 * size[0]:
                sx, sy = 0.0, 0.0
            dx, dy = int(round(sx)), int(round(sy))
            Hh, Ww = g.shape
            xs0, xs1 = max(0, dx), min(Ww, Ww + dx); ys0, ys1 = max(0, dy), min(Hh, Hh + dy)
            if xs1 > xs0 and ys1 > ys0:
                d = np.abs(g[ys0:ys1, xs0:xs1].astype(np.int16) - prev[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx].astype(np.int16))
                changed[t] = float((d > 25).mean())
        prev, prev_h = g, h
    cut = (hist_d > hist_thr) | ((hist_d > hist_soft) & (changed > motion_soft)) | \
          ((resp_s < resp_min) & (hist_d > hist_resp) & (changed > motion_resp))
    cut[0] = False
    idx = np.where(cut)[0]
    cuts = [int(t) for i, t in enumerate(idx) if i == 0 or t - idx[i - 1] > 1]     # agrupar frames consecutivos
    out = np.asarray(cuts, np.int64)
    return (out, {"hist": hist_d, "resp": resp_s, "changed": changed}) if return_series else out


def overlay_band_y0(h: int) -> int:
    """Inicio (fila) de la banda del overlay FIE: ~78 % de la altura. Todo lo que este por debajo se recorta."""
    return int(h * 0.78)


def strip_overlay(frame: np.ndarray) -> np.ndarray:
    """ANTI-FUGA: elimina la banda del overlay (marcador, luces, nombres) antes de dar el frame a un modelo de video."""
    return frame[: overlay_band_y0(frame.shape[0])]


def crop_roi(frame: np.ndarray, boxes: list[tuple[float, float, float, float]], margin: float = 0.15,
             out_size: tuple[int, int] = (384, 224)) -> np.ndarray:
    """Recorte alrededor de los dos tiradores (union de cajas + margen) y reescalado. Excluye la banda del overlay."""
    frame = strip_overlay(frame)
    H, W = frame.shape[:2]
    if not boxes:
        return cv2.resize(frame, out_size)
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    mw, mh = (x1 - x0) * margin, (y1 - y0) * margin
    x0, y0 = int(max(0, x0 - mw)), int(max(0, y0 - mh)); x1, y1 = int(min(W, x1 + mw)), int(min(H, y1 + mh))
    return cv2.resize(frame[y0:y1, x0:x1], out_size)

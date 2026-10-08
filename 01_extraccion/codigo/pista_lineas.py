"""PISTA POR RAYAS + MOVIMIENTO DE CAMARA, frame a frame. Traido de FENCINGLY.AI (`fencingly/piste.py`)
el 2026-10-08 al unificar los dos proyectos, sin cambios de logica.

QUE APORTA que `pista.py` no tiene: la homografia de un fotograma se PROPAGA a los demas con el
movimiento de camara medido (flujo optico LK + RANSAC sobre el suelo), y se re-identifican las rayas
con la homografia predicha. Asi el origen y la escala de un fotograma bueno llegan al fotograma del
tocado aunque en el no se vea ninguna raya. Lo consume `zona_tocado.py` (etapa 11) y su cifra
contra `strip_location` esta alli.

"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import numpy as np


# ------------------------------------------------------------------------------------------
# Fase streaming
# ------------------------------------------------------------------------------------------
_LIMBS = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

@dataclass
class FrameFeat:
    segs: np.ndarray          # (N,4) x1,y1,x2,y2 en píxeles de la imagen ORIGINAL
    G: np.ndarray | None      # homografía imagen_{t-1} -> imagen_t (píxeles originales)
    cut: bool
    n_flow: int
    gray: np.ndarray | None = None   # frame en gris a resolución de análisis
    valid: np.ndarray | None = None  # máscara (análisis) sin personas ni overlay
    s: float = 1.0                   # escala análisis / original


class PisteFeatures:
    def __init__(self, cfg_video: dict, frame_wh: tuple[int, int]):
        w, h = frame_wh
        self.s = min(1.0, cfg_video["analysis_width"] / w)
        self.aw, self.ah = int(round(w * self.s)), int(round(h * self.s))
        self.cut_thr = cfg_video["cut_threshold"]
        self.prev_gray = None
        self.prev_hist = None
        self.feats: list[FrameFeat] = []
        S = np.diag([self.s, self.s, 1.0])
        self._S, self._Si = S, np.linalg.inv(S)

    def process(self, img: np.ndarray, person_boxes: np.ndarray, overlay_y0: float | None,
                person_kps: np.ndarray | None = None) -> FrameFeat:
        small = cv2.resize(img, (self.aw, self.ah), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
        cut = False
        if self.prev_hist is not None:
            cut = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > self.cut_thr

        # Máscara de "fondo útil": fuera de personas y del overlay
        mask = np.full(gray.shape, 255, np.uint8)
        for j, b in enumerate(person_boxes):
            x1, y1, x2, y2 = (b[:4] * self.s).astype(int)
            k = person_kps[j] if person_kps is not None and j < len(person_kps) else None
            if k is None or (k[:, 2] > 0.3).sum() < 8:
                pad = int(0.08 * (y2 - y1))
                cv2.rectangle(mask, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), 0, -1)
                continue
            # "cápsulas" a lo largo del esqueleto: deja libre el suelo entre las piernas
            th = max(3, int(0.09 * (y2 - y1)))
            for a, c in _LIMBS:
                if k[a, 2] > 0.3 and k[c, 2] > 0.3:
                    cv2.line(mask, tuple((k[a, :2] * self.s).astype(int)), tuple((k[c, :2] * self.s).astype(int)), 0, th)
            cv2.rectangle(mask, (x1, y1), (x2, int(y1 + 0.3 * (y2 - y1))), 0, -1)  # cabeza/hombros
            for f in (15, 16):  # pies
                if k[f, 2] > 0.3:
                    cv2.circle(mask, tuple((k[f, :2] * self.s).astype(int)), int(1.3 * th), 0, -1)
        if overlay_y0 is not None:
            mask[int(overlay_y0 * self.s) - 4:] = 0
        mask[: int(0.12 * self.ah), : int(0.45 * self.aw)] = 0  # rótulo de torneo arriba a la izq.

        # Segmentos de línea
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 40, 120)
        edges[mask == 0] = 0
        # Solo hacen falta segmentos largos y poco inclinados (bordes de la pista): las líneas
        # transversales se buscan después en la banda rectificada
        lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=40,
                                minLineLength=int(0.06 * self.aw), maxLineGap=int(0.012 * self.aw))
        segs = np.zeros((0, 4), np.float32)
        if lines is not None:
            sg = lines.reshape(-1, 4).astype(np.float32)
            ang = np.degrees(np.arctan2(sg[:, 3] - sg[:, 1], sg[:, 2] - sg[:, 0])) % 180
            sg = sg[(ang < 40) | (ang > 140)]
            segs = sg / self.s

        # Movimiento de cámara
        G, n_flow = None, 0
        if self.prev_gray is not None and not cut:
            pts = cv2.goodFeaturesToTrack(self.prev_gray, 200, 0.01, 10, mask=self._prev_mask)
            if pts is not None and len(pts) >= 12:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
                good = st.reshape(-1) == 1
                if good.sum() >= 12:
                    Hs, inl = cv2.findHomography(pts[good], nxt[good], cv2.RANSAC, 2.0)
                    if Hs is not None and inl is not None and inl.sum() >= 10:
                        G = self._Si @ Hs @ self._S
                        n_flow = int(inl.sum())
        self.prev_gray, self.prev_hist, self._prev_mask = gray, hist, mask
        f = FrameFeat(segs, G, bool(cut), n_flow, gray, mask, self.s)
        self.feats.append(f)
        return f


# ------------------------------------------------------------------------------------------
# Geometría auxiliar
# ------------------------------------------------------------------------------------------
def _line_from_seg(s):
    p = np.array([s[0], s[1], 1.0])
    q = np.array([s[2], s[3], 1.0])
    l = np.cross(p, q)
    return l / (np.linalg.norm(l[:2]) + 1e-9)


def _y_at(l, x):
    return -(l[0] * x + l[2]) / (l[1] if abs(l[1]) > 1e-9 else 1e-9)


def _intersect(l1, l2):
    p = np.cross(l1, l2)
    return p[:2] / p[2] if abs(p[2]) > 1e-9 else None


def _merge_lines(segs: np.ndarray, ang_tol=3.0, dist_tol=6.0):
    """Agrupa segmentos colineales. Devuelve [{'l','len','ang','ends'}] (recta homogénea normalizada)."""
    if len(segs) == 0:
        return []
    d = segs[:, 2:] - segs[:, :2]
    lens = np.hypot(d[:, 0], d[:, 1])
    angs = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180
    P = np.c_[segs[:, :2], np.ones(len(segs))]
    Q = np.c_[segs[:, 2:], np.ones(len(segs))]
    L = np.cross(P, Q)
    L /= np.linalg.norm(L[:, :2], axis=1, keepdims=True) + 1e-9
    mids = np.c_[(segs[:, :2] + segs[:, 2:]) / 2, np.ones(len(segs))]
    group = -np.ones(len(segs), int)
    reps = []  # (índice representante)
    for i in np.argsort(-lens):
        if reps:
            r = np.array(reps)
            dang = np.abs(angs[i] - angs[r])
            dang = np.minimum(dang, 180 - dang)
            dist = np.abs((L[r] * mids[i]).sum(1))
            ok = np.flatnonzero((dang < ang_tol) & (dist < dist_tol))
            if len(ok):
                group[i] = ok[0]
                continue
        group[i] = len(reps)
        reps.append(i)
    out = []
    for g, rep in enumerate(reps):
        idx = np.flatnonzero(group == g)
        pts = np.r_[segs[idx, :2], segs[idx, 2:]].astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
        l = np.cross([x0, y0, 1.0], [x0 + vx, y0 + vy, 1.0])
        l = l / (np.linalg.norm(l[:2]) + 1e-9)
        proj = (pts - [x0, y0]) @ np.array([vx, vy])
        ends = np.array([x0, y0]) + np.outer([proj.min(), proj.max()], [vx, vy])
        out.append({"l": l, "len": float(lens[idx].sum()), "ang": float(angs[rep]), "ends": ends})
    return out


# ------------------------------------------------------------------------------------------
# Calibración
# ------------------------------------------------------------------------------------------
@dataclass
class FrameObs:
    far: np.ndarray | None = None      # recta borde lejano
    near: np.ndarray | None = None     # recta borde cercano
    trans: list | None = None          # [(p_far, p_near)] ordenadas de izq. a dcha.
    dashed: list | None = None         # por línea transversal: ¿es discontinua? (típico de la central)
    zchange: list | None = None        # por línea: salto de intensidad entre ambos lados (cambio de zona)
    model_pts: np.ndarray | None = None  # keypoints del modelo aprendido (14,3) img


def piste_keypoints_m(cfg) -> np.ndarray:
    """Los 14 keypoints canónicos: cada línea transversal × (borde lejano, borde cercano)."""
    W = cfg["width_m"]
    return np.array([[x, y] for x in cfg["lines_m"] for y in (-W / 2, W / 2)], np.float32)


def observe(feat: FrameFeat, feet: np.ndarray, heights: np.ndarray, frame_wh) -> FrameObs:
    """Bordes y líneas transversales de un frame usando los pies de los tiradores como ancla.

    feet: (2,2) puntos de apoyo (nan si falta); heights: (2,) altura de caja en px.
    """
    w, h = frame_wh
    obs = FrameObs()
    lines = _merge_lines(feat.segs, dist_tol=max(4.0, 0.006 * w))
    if not lines:
        return obs
    good_feet = feet[~np.isnan(feet).any(1)]
    if len(good_feet) == 0:
        return obs
    fh = float(np.nanmax(heights)) if np.isfinite(heights).any() else 0.3 * h

    longi = [L for L in lines if (L["ang"] < 35 or L["ang"] > 145) and L["len"] > 0.10 * w]
    best_far = best_near = None
    for L in longi:
        ys = np.array([_y_at(L["l"], fx) for fx, _ in good_feet])
        gaps = good_feet[:, 1] - ys  # >0: la línea está por encima de los pies
        if (gaps > 0.03 * fh).all() and (gaps < 0.7 * fh).all():
            score = L["len"] / (1 + 4 * gaps.mean() / fh)
            if best_far is None or score > best_far[0]:
                best_far = (score, L)
        elif (gaps < -0.01 * fh).all() and (gaps > -0.9 * fh).all():
            score = L["len"] / (1 + 4 * -gaps.mean() / fh)
            if best_near is None or score > best_near[0]:
                best_near = (score, L)
    if best_far is None or best_near is None:
        obs.far = best_far[1]["l"] if best_far else None
        obs.near = best_near[1]["l"] if best_near else None
        return obs
    far, near = best_far[1]["l"], best_near[1]["l"]
    obs.far, obs.near = far, near
    if abs(far[1]) < 1e-6 or abs(near[1]) < 1e-6:
        return obs
    # bordes casi paralelos en la imagen (como mucho convergen por la perspectiva)
    if min(abs(_ang(far) - _ang(near)), 180 - abs(_ang(far) - _ang(near))) > 20:
        obs.near = None
        return obs
    found = transverse_in_band(feat, far, near, w)
    obs.trans = [(a, b) for a, b, _, _ in found]
    obs.dashed = [d for _, _, d, _ in found]
    obs.zchange = [z for _, _, _, z in found]
    return obs


def _ang(l):
    return np.degrees(np.arctan2(-l[0], l[1])) % 180


BAND_W, BAND_H = 1024, 64


def transverse_in_band(feat: FrameFeat, far, near, w) -> list:
    """Líneas transversales dentro de la franja entre bordes, rectificada a un rectángulo.

    En la banda rectificada las líneas transversales quedan casi verticales; se cierran los
    huecos en vertical (líneas discontinuas) y se buscan con Hough. Devuelve [(p_far, p_near)]
    en píxeles originales, de izquierda a derecha.
    """
    if feat.gray is None:
        return []
    s = feat.s
    src = np.float32([[0, _y_at(far, 0)], [w, _y_at(far, w)], [w, _y_at(near, w)], [0, _y_at(near, 0)]]) * s
    dst = np.float32([[0, 0], [BAND_W, 0], [BAND_W, BAND_H], [0, BAND_H]])
    try:
        Hp = cv2.getPerspectiveTransform(src, dst)
        Hpi = np.linalg.inv(Hp)
    except (cv2.error, np.linalg.LinAlgError):
        return []
    band = cv2.warpPerspective(feat.gray, Hp, (BAND_W, BAND_H), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(feat.valid, Hp, (BAND_W, BAND_H), flags=cv2.INTER_NEAREST)
    # erosión amplia: el borde de la imagen y las cajas de personas generan aristas falsas
    valid = cv2.erode(valid, np.ones((5, 21), np.uint8))
    edges = cv2.Canny(cv2.GaussianBlur(band, (3, 3), 0), 30, 90)
    edges[valid == 0] = 0
    raw = edges.copy()
    # cierre para unir líneas discontinuas (la central suele ser a trazos)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((13, 3), np.uint8))
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=int(0.3 * BAND_H),
                           minLineLength=int(0.45 * BAND_H), maxLineGap=int(0.25 * BAND_H))
    if segs is None:
        return []
    cands = []
    for x1, y1, x2, y2 in segs.reshape(-1, 4).astype(np.float32):
        # en la banda las transversales no son verticales (perspectiva), pero sí cruzan de
        # arriba abajo: se descartan las casi horizontales (bordes, juntas longitudinales)
        if abs(x2 - x1) > 3.0 * abs(y2 - y1):
            continue
        # x en la mitad de la banda
        t = (BAND_H / 2 - y1) / (y2 - y1) if y2 != y1 else 0.5
        xm = x1 + t * (x2 - x1)
        cands.append((xm, x1, y1, x2, y2))
    cands.sort()
    groups = []
    for c in cands:
        if groups and c[0] - groups[-1][-1][0] < 12:
            groups[-1].append(c)
        else:
            groups.append([c])
    out = []
    for g in groups:
        P = np.float32([[x1, y1] for _, x1, y1, _, _ in g] + [[x2, y2] for _, _, _, x2, y2 in g])
        ys = P[:, 1]
        if ys.max() - ys.min() < 0.5 * BAND_H:
            continue
        vx, vy, x0, y0 = cv2.fitLine(P, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
        if abs(vy) < 1e-6:
            continue
        xt = x0 + (0 - y0) / vy * vx
        xb = x0 + (BAND_H - y0) / vy * vx
        pts = cv2.perspectiveTransform(np.float32([[[xt, 0]], [[xb, BAND_H]]]), Hpi).reshape(2, 2) / s
        out.append((pts[0], pts[1], _is_dashed(raw, valid, xt, xb), _zone_change(band, valid, xt, xb)))
    return out


def _zone_change(band, valid, xt, xb, off=6, width=24) -> float:
    """Diferencia de intensidad media entre las franjas a la izquierda y a la derecha de la línea.

    Las líneas de 0 y ±2 m separan suelo del mismo color; en ±5 (inicio de la zona de
    advertencia, a menudo de otro color) y ±7 (final de pista) suele haber un salto.
    """
    rows = np.arange(BAND_H)
    xs = xt + (xb - xt) * rows / BAND_H
    left, right = [], []
    for y, x in zip(rows, xs):
        a0, a1 = int(x - off - width), int(x - off)
        b0, b1 = int(x + off), int(x + off + width)
        if a0 >= 0 and b1 < band.shape[1]:
            lv, rv = valid[y, a0:a1] > 0, valid[y, b0:b1] > 0
            if lv.sum() > width // 3 and rv.sum() > width // 3:
                left.append(band[y, a0:a1][lv].mean())
                right.append(band[y, b0:b1][rv].mean())
    if len(left) < BAND_H // 3:
        return 0.0
    return float(abs(np.median(left) - np.median(right)))


def _is_dashed(raw: np.ndarray, valid: np.ndarray, xt: float, xb: float) -> bool:
    """Línea a trazos: a lo largo de la línea hay tramos con arista y tramos sin ella."""
    rows = np.arange(BAND_H)
    xs = (xt + (xb - xt) * rows / BAND_H).astype(int)
    hit, seen = [], []
    for y, x in zip(rows, xs):
        if 2 <= x < raw.shape[1] - 2 and valid[y, x]:
            seen.append(y)
            hit.append(raw[y, x - 2:x + 3].any())
    if len(seen) < 0.5 * BAND_H:
        return False
    hit = np.array(hit)
    cov = hit.mean()
    changes = np.abs(np.diff(hit.astype(np.int8))).sum()
    return bool(0.25 < cov < 0.8 and changes >= 4)


def _H_from(lines_px: list, xs_m: list, W: float):
    src, dst = [], []
    for (pf, pn), x in zip(lines_px, xs_m):
        src += [pf, pn]
        dst += [[x, -W / 2], [x, W / 2]]
    src, dst = np.float32(src), np.float32(dst)
    if len(src) < 4:
        return None, np.inf
    if len(src) == 4:
        H = cv2.getPerspectiveTransform(src, dst)
    else:
        H, _ = cv2.findHomography(src, dst, 0)
    if H is None or not np.isfinite(H).all():
        return None, np.inf
    return H, reproj_px(H, src, dst)


def reproj_px(H, src_px, dst_m) -> float:
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.inf
    back = cv2.perspectiveTransform(np.float32(dst_m).reshape(-1, 1, 2), Hi).reshape(-1, 2)
    return float(np.linalg.norm(back - np.float32(src_px), axis=1).mean())


def to_m(H, pts_px: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_px, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def to_px(H, pts_m: np.ndarray) -> np.ndarray:
    return to_m(np.linalg.inv(H), pts_m)


def _scale_prior(H, feet, heights, cfg) -> float:
    """log-verosimilitud del prior de escala y de que los pies estén sobre la pista."""
    ll = 0.0
    W = cfg["width_m"]
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return -1e9
    xs = []
    for f, hh in zip(feet, heights):
        if np.isnan(f).any() or not np.isfinite(hh):
            continue
        m = to_m(H, f[None])[0]
        if not np.isfinite(m).all():
            return -1e9
        xs.append(m[0])
        a = cv2.perspectiveTransform(np.float32([[[m[0] - 0.5, m[1]]], [[m[0] + 0.5, m[1]]]]), Hi).reshape(2, 2)
        ppm = np.linalg.norm(a[1] - a[0])
        h_m = hh / max(ppm, 1e-6)
        ll += -0.5 * ((h_m - cfg["fencer_height_m"]) / 0.25) ** 2
        ll += -0.5 * (m[0] / 5.0) ** 2  # los tiradores suelen estar más cerca del centro
        ll += -0.5 * (max(0.0, abs(m[1]) - (W / 2 + 0.4)) / 0.2) ** 2
        ll += -0.5 * (max(0.0, abs(m[0]) - 9.0) / 0.5) ** 2
    if len(xs) == 2:
        ll += -0.5 * (max(0.0, xs[0] - xs[1] + 0.3) / 0.3) ** 2  # izquierdo a la izquierda
    return ll


def hypothesize(obs: FrameObs, feet, heights, cfg, max_hyp=3):
    """heights: longitud corporal (px) de cada tirador, ver tracking.body_length_px."""
    """Hipótesis de identificación de líneas para un frame aislado (sin historia)."""
    if not obs.trans or len(obs.trans) < 2:
        return []
    known = cfg["lines_m"]
    W = cfg["width_m"]
    tr = obs.trans[:6]
    dashed = (obs.dashed or [False] * len(obs.trans))[:6]
    zch = (obs.zchange or [0.0] * len(obs.trans))[:6]
    hyps = []
    for k in range(min(len(tr), 5), 1, -1):
        # subconjuntos contiguos de las líneas detectadas (ordenadas de izq. a dcha.)
        for st in range(len(tr) - k + 1):
            lines_px = tr[st: st + k]
            dsh = dashed[st: st + k]
            zc = zch[st: st + k]
            for xs in itertools.combinations(known, k):
                H, err = _H_from(lines_px, list(xs), W)
                if H is None or err > 6.0:
                    continue
                ll = _scale_prior(H, feet, heights, cfg) - 0.5 * (err / 2.0) ** 2 + 0.8 * k
                # la línea central suele ser discontinua: premia ponerla en 0 y penaliza lo contrario
                for d, z, x in zip(dsh, zc, xs):
                    if d:
                        ll += 1.0 if x == 0.0 else -0.5
                    if z > 25:  # cambio de color de zona: inicio de advertencia (±5) o final (±7)
                        ll += 2.0 if abs(x) >= 5 else -2.0
                hyps.append((ll, H, err, k, xs))
        if hyps and k >= 3:
            break
    hyps.sort(key=lambda z: -z[0])
    return hyps[:max_hyp]


def identify(H_pred, obs: FrameObs, cfg):
    """Identifica las líneas observadas usando una homografía predicha."""
    if not obs.trans:
        return [], []
    tol = cfg["line_match_tol_m"] * 2
    lines_px, xs = [], []
    known = np.array(cfg["lines_m"])
    for pf, pn in obs.trans:
        mid = to_m(H_pred, ((pf + pn) / 2)[None])[0]
        if not np.isfinite(mid).all():
            continue
        j = np.argmin(abs(known - mid[0]))
        if abs(known[j] - mid[0]) < tol and known[j] not in xs:
            lines_px.append((pf, pn))
            xs.append(float(known[j]))
    return lines_px, xs


def calibrate(feats: list[FrameFeat], feet: np.ndarray, heights: np.ndarray, cfg: dict, frame_wh,
              model_obs: list | None = None, model_priority: bool = False) -> dict:
    """Homografía imagen->pista por frame.

    feet: (T,2,2) puntos de apoyo de izq./dcha. (nan si falta); heights: (T,2) altura de caja px.
    model_obs: opcional, por frame (14,3) keypoints del modelo aprendido (x,y,score) o None.
    model_priority: si True el modelo manda sobre las líneas cuando ambos son coherentes con la
        predicción (se activa cuando la evaluación manual dice que el modelo es mejor).
    Devuelve dict con H (T,3,3), conf (T,), reproj (T,), n_lines (T,), method (T,) y obs.
    """
    T = len(feats)
    W = cfg["width_m"]
    kpm = piste_keypoints_m(cfg)
    obs = [observe(f, feet[t], heights[t], frame_wh) for t, f in enumerate(feats)]

    def model_H(t):
        if model_obs is None or model_obs[t] is None:
            return None, np.inf, 0
        k = model_obs[t]
        ok = k[:, 2] > 0.35
        if ok.sum() < 4:
            return None, np.inf, 0
        xs_ok = np.unique(kpm[ok, 0])
        if len(xs_ok) < 2:
            return None, np.inf, 0
        H, inl = cv2.findHomography(k[ok, :2], kpm[ok], cv2.RANSAC, 0.25)
        if H is None:
            return None, np.inf, 0
        return H, reproj_px(H, k[ok, :2], kpm[ok]), int(ok.sum())

    # Segmentos sin cortes
    seg_start = sorted(set(t for t in range(T) if t == 0 or feats[t].cut))
    bounds = list(zip(seg_start, seg_start[1:] + [T]))

    anchors = []
    Hs = np.full((T, 3, 3), np.nan, np.float64)
    conf = np.zeros(T, np.float32)
    rep = np.full(T, np.nan, np.float32)
    nl = np.zeros(T, np.int16)
    method = np.array([""] * T, dtype=object)

    for a, b in bounds:
        # 1) semillas: frames con hipótesis fuertes (k>=3 o prior muy bueno) o modelo
        seeds = []
        step = 1 if b - a < 30 else 3
        for t in range(a, b, step):
            Hm, em, nm = model_H(t)
            if Hm is not None and em < 4.0 and nm >= 6:
                seeds.append((2.0 + 0.1 * nm, t, Hm, em, nm, "model", None))
            hy = hypothesize(obs[t], feet[t], heights[t], cfg)
            if hy:
                ll, H, err, k, xs = hy[0]
                margin = ll - hy[1][0] if len(hy) > 1 else 3.0
                if k >= 3 or margin > 1.0:
                    seeds.append((ll + min(margin, 3.0), t, H, err, k, "lines", xs))
        if not seeds:
            continue
        n_seeds = len(seeds)
        seeds, sup = _consensus(seeds, feats, a, b, kpm, frame_wh)
        anchors.append({"segment": [a, b], "n_seeds": n_seeds, "support": sup, "frame": seeds[0][1],
                        "lines_m": seeds[0][6], "n_agree": len(seeds)})
        # 2) propagación desde las mejores semillas (hacia delante y hacia atrás)
        seg_H = np.full((b - a, 3, 3), np.nan)
        seg_c = np.zeros(b - a, np.float32)
        seg_r = np.full(b - a, np.nan, np.float32)
        seg_n = np.zeros(b - a, np.int16)
        seg_m = np.array([""] * (b - a), dtype=object)
        for sc, t0, H0, e0, k0, mth, _xs in seeds[:4]:
            c0 = float(np.clip(0.4 + 0.12 * k0 - 0.03 * e0, 0.2, 0.95))
            for direction in (1, -1):
                H, c = H0.copy(), c0
                t = t0
                while a <= t < b:
                    i = t - a
                    if t != t0:
                        # predicción con el movimiento de cámara
                        if direction == 1:
                            G = feats[t].G
                            Hp = H @ np.linalg.inv(G) if G is not None else None
                        else:
                            G = feats[t + 1].G
                            Hp = H @ G if G is not None else None
                        if Hp is None:
                            break
                        H, c, e, k, m = Hp, c * 0.985, np.nan, 0, "propagated"
                        cand_model = cand_lines = None
                        Hm, em, nm = model_H(t)
                        if Hm is not None and em < 4.0 and _H_disagreement(Hm, Hp, kpm, frame_wh) < 0.6:
                            cand_model = (Hm, min(0.95, 0.5 + 0.05 * nm), em, nm // 2, "model")
                        lp, xs = identify(Hp, obs[t], cfg)
                        if len(lp) >= 2:
                            Ho, eo = _H_from(lp, xs, W)
                            if Ho is not None and eo < 5.0 and _H_disagreement(Ho, Hp, kpm, frame_wh) < 0.8:
                                cand_lines = (Ho, min(0.95, 0.45 + 0.12 * len(lp) - 0.03 * eo), eo, len(lp), "lines")
                        order = (cand_model, cand_lines) if model_priority else (cand_lines, cand_model)
                        pick = next((z for z in order if z is not None), None)
                        if pick is not None:
                            H, c, e, k, m = pick
                    else:
                        e, k, m = e0, k0, mth
                        c = c0
                    if c > seg_c[i]:
                        seg_H[i], seg_c[i], seg_r[i], seg_n[i], seg_m[i] = H, c, e, k, m
                    if c < 0.05:
                        break
                    t += direction
        Hs[a:b], conf[a:b], rep[a:b], nl[a:b], method[a:b] = seg_H, seg_c, seg_r, seg_n, seg_m

    return {"H": Hs, "conf": conf, "reproj_px": rep, "n_lines": nl, "method": method, "obs": obs, "anchors": anchors}


def _cumulative(feats, a, b):
    """C[t]: homografía imagen_a -> imagen_t acumulando G (None donde se rompe la cadena)."""
    C = [None] * (b - a)
    C[0] = np.eye(3)
    for t in range(a + 1, b):
        G = feats[t].G
        C[t - a] = G @ C[t - a - 1] if (G is not None and C[t - a - 1] is not None) else None
    return C


def _consensus(seeds, feats, a, b, kpm, frame_wh, max_seeds=40, tol_m=0.7):
    """Ordena las semillas por apoyo: cuántas otras semillas (en otros frames) son coherentes con
    ella una vez propagada con el movimiento de cámara. Las identificaciones aisladas o
    simétricas-erróneas pierden frente a la que se repite a lo largo del clip."""
    seeds = sorted(seeds, key=lambda z: -z[0])[:max_seeds]
    C = _cumulative(feats, a, b)
    support = []
    for i, (sc, t0, H0, *_r) in enumerate(seeds):
        sup = 0.0
        for j, (sc2, t1, H1, *_r2) in enumerate(seeds):
            if i == j:
                sup += 1.0
                continue
            c0, c1 = C[t0 - a], C[t1 - a]
            if c0 is None or c1 is None:
                continue
            try:
                Hp = H0 @ c0 @ np.linalg.inv(c1)  # semilla i llevada al frame t1
            except np.linalg.LinAlgError:
                continue
            if _H_disagreement(H1, Hp, kpm, frame_wh) < tol_m:
                sup += 1.0
        support.append(sup + 0.05 * sc)
    order = np.argsort(-np.array(support))
    best = order[0]
    best_sup = float(support[best])
    # devuelve la mejor y las coherentes con ella (para re-anclar la propagación)
    out = [seeds[best]]
    t0, H0 = seeds[best][1], seeds[best][2]
    for j in order[1:]:
        sc2, t1, H1, *_ = seeds[j]
        c0, c1 = C[t0 - a], C[t1 - a]
        if c0 is not None and c1 is not None:
            if _H_disagreement(H1, H0 @ c0 @ np.linalg.inv(c1), kpm, frame_wh) < tol_m:
                out.append(seeds[j])
    return out, round(best_sup, 2)


def _H_disagreement(H1, H2, kpm, frame_wh) -> float:
    """Discrepancia (m) entre dos homografías sobre puntos de la pista visibles."""
    w, h = frame_wh
    try:
        px = to_px(H1, kpm)
    except np.linalg.LinAlgError:
        return np.inf
    vis = (px[:, 0] > -0.1 * w) & (px[:, 0] < 1.1 * w) & (px[:, 1] > 0) & (px[:, 1] < h)
    if vis.sum() < 2:
        return np.inf
    m2 = to_m(H2, px[vis])
    return float(np.linalg.norm(m2 - kpm[vis], axis=1).mean())


def zone_of(x_m: float) -> str:
    ax = abs(x_m)
    if ax <= 2.0:
        return "centro"            # entre líneas de en garde
    if ax <= 5.0:
        return "media"             # entre en garde y advertencia
    if ax <= 7.0:
        return "advertencia"       # últimos 2 m
    return "fuera"                 # más allá de la línea final

"""LA CAMARA DE TRIPODE para la etapa 11: la homografia del tocado parametrizada con la camara fisica.

POR QUE (2026-10-09, `04_documentacion/informes/CAMARA_TRIPODE_2026-10-09.md`). En ~4 de cada 10 clips
`pista_lineas.calibrate` no da homografia confiable cerca del tocado: la pista mide 14 x 1,5 m, se ven
pocas rayas, casi paralelas, y muchas veces el borde CERCANO esta tapado por el rotulo, con lo que
`observe` ni siquiera busca rayas. Una homografia libre tiene 8 grados de libertad y con esa geometria
esta mal condicionada. La camara de retransmision es un TRIPODE: un centro optico fijo que gira y hace
zoom. Con la camara fisica (centro `C`, pan, tilt, roll pequenio, focal; punto principal en el centro y
pixel cuadrado) y priors fisicos (camara frente a la pista, roll ~0, tiradores de ~1,65 m) el problema
queda bien planteado.

COMO. "Render and compare": las rayas y los bordes del modelo de pista, proyectados con la camara,
tienen que caer sobre aristas de la imagen (Canny fuera de personas y rotulo). La camara del cuadro
del tocado se lleva a los cuadros de una ventana anterior con el movimiento de camara ya medido
(`FrameFeat.G`), asi que las rayas vistas en CUALQUIER cuadro de la ventana cuentan. Las rayas son
iguales entre si, pero sus separaciones no (2-3-2 m): la busqueda recorre hipotesis de donde esta la
pareja a lo largo de la pista y se queda con la que mejor explica las aristas; el MARGEN contra la
mejor hipotesis de otra posicion es la confianza.

Solo geometria y numpy/OpenCV, en CPU.
"""
from __future__ import annotations

import numpy as np

LINEAS_M = (-7.0, -5.0, -2.0, 0.0, 2.0, 5.0, 7.0)
ANCHO_M = 1.5
TAU = 4.0
"""Distancia (px de ANALISIS, 640 de ancho) a la que una arista cuenta como la raya del modelo."""
LAMBDA = 0.35
"""Penalizacion por punto del modelo que cae en zona valida SIN arista: sin ella, la hipotesis que
pone una raya donde no la hay saldria gratis, y eso es justo lo que distingue +-2 de +-5."""
PESO_TRANS = 3.0
"""Las rayas transversales son cortas (1,5 m) y son las que dicen DONDE se mira: pesan mas por punto
que los bordes, que son largos y solo fijan la inclinacion."""
ESTATURA_M, ESTATURA_SD = 1.65, 0.25
"""Los mismos del prior de escala de `pista_lineas` (`largo_cuerpo` ~ 0,93 x estatura)."""
PESO_PRIOR = 10.0
PIE_Y_SD = 0.35
"""Desviacion de los pies respecto a la linea central de la pista (m)."""


def puntos_modelo(ancho=ANCHO_M, paso=0.1):
    """Puntos de pista (x, y) con y = -ancho/2 el borde LEJANO (la convencion de `pista_lineas`), y
    los mismos desplazados un poco A LO LARGO de su raya (para la orientacion en la imagen)."""
    xs = np.arange(-7.0, 7.0 + 1e-6, paso)
    lejos = np.c_[xs, np.full_like(xs, -ancho / 2)]
    cerca = np.c_[xs, np.full_like(xs, ancho / 2)]
    ys = np.linspace(-ancho / 2, ancho / 2, 21)
    trans = np.concatenate([np.c_[np.full_like(ys, x), ys] for x in LINEAS_M])
    tipo = np.r_[np.zeros(len(lejos)), np.ones(len(cerca)), np.full(len(trans), 2)]
    linea = np.r_[np.full(len(lejos) + len(cerca), -1),
                  np.repeat(np.arange(len(LINEAS_M)), len(ys))]
    P = np.r_[lejos, cerca, trans]
    dlt = np.where(tipo[:, None] == 2, [0.0, 0.05], [0.05, 0.0])
    P2 = P + dlt
    h = lambda Q: np.c_[Q, np.ones(len(Q))].T.astype(np.float64)
    return h(P), tipo, linea, h(P2)


def camara(p, w, h):
    """p = (Xc, Yc, Zc, pan, tilt, roll, log f). QUE DEVUELVE: (H pista->imagen en px originales,
    vector de mirada d, centro C). Mundo: x a lo largo de la pista (derecha de la imagen = +x), Y_w
    hacia el fondo (la camara en Y_w < 0 mirando a +Y_w), Z arriba; la pista de `pista_lineas` tiene
    y = -Y_w (el borde lejano en y = -ancho/2)."""
    Xc, Yc, Zc, pan, tilt, roll, lf = p
    f = float(np.exp(lf))
    cp, sp, ct, st = np.cos(pan), np.sin(pan), np.cos(tilt), np.sin(tilt)
    d = np.array([sp * ct, cp * ct, -st])
    r = np.array([cp, -sp, 0.0])
    a = np.cross(d, r)
    cr, sr = np.cos(roll), np.sin(roll)
    r2 = cr * r + sr * a
    a2 = -sr * r + cr * a
    R = np.stack([r2, a2, d])
    C = np.array([Xc, Yc, Zc])
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
    Hw = K @ np.c_[R[:, 0], R[:, 1], -R @ C]
    return Hw @ np.diag([1.0, -1.0, 1.0]), d, C


def cadena(G, tq, cuadros):
    """QUE DEVUELVE: {t: M} con M la homografia imagen_tq -> imagen_t, solo para los cuadros a los
    que se llega sin un G roto (corte o flujo fallido). `G[t]` es imagen_{t-1} -> imagen_t."""
    out = {tq: np.eye(3)}
    M = np.eye(3)
    for t in range(tq - 1, min(cuadros) - 1, -1):
        g = G[t + 1]
        if g is None:
            break
        try:
            M = M @ np.linalg.inv(g)
        except np.linalg.LinAlgError:
            break
        out[t] = M.copy()
    M = np.eye(3)
    for t in range(tq + 1, max(cuadros) + 1):
        g = G[t] if t < len(G) else None
        if g is None:
            break
        M = g @ M
        out[t] = M.copy()
    return {t: out[t] for t in cuadros if t in out}


N_BINS = 8
"""Bandas de orientacion de las aristas (22,5 grados cada una). Un punto del modelo solo se empareja
con aristas de su misma orientacion (+-1 banda): sin esto el publico, las vallas y los anuncios dan
aristas en todas partes y cualquier hipotesis encuentra "rayas" (medido el 09/10: el primer ajuste,
sin orientacion, veia 5-6 rayas en clips con 2)."""


def _dt_orientadas(e):
    """De un mapa de aristas binario: (N_BINS, h, w) distancias a la arista mas cercana de cada
    orientacion, truncadas a TAU, en float16."""
    import cv2
    eb = cv2.GaussianBlur(e.astype(np.float32), (5, 5), 0)
    gx = cv2.Sobel(eb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(eb, cv2.CV_32F, 0, 1, ksize=3)
    ang = (np.degrees(np.arctan2(gy, gx)) + 90.0) % 180.0
    b = np.minimum((ang / (180.0 / N_BINS)).astype(np.int32), N_BINS - 1)
    out = np.empty((N_BINS,) + e.shape, np.float16)
    for k in range(N_BINS):
        ek = e & (b == k)
        dt = cv2.distanceTransform((~ek).astype(np.uint8), cv2.DIST_L2, 3)
        out[k] = np.minimum(dt, TAU)
    return out


class Evidencia:
    """Las aristas de los cuadros de la ventana, ya como mapas de distancia por orientacion, y su
    encadenado al cuadro del tocado."""

    def __init__(self, aristas, validas, M, s, tam, max_cuadros=10):
        self.s = float(s)
        self.ah, self.aw = tam
        ts = sorted(t for t in aristas if t in M)
        if len(ts) > max_cuadros:
            sel = np.unique(np.round(np.linspace(0, len(ts) - 1, max_cuadros)).astype(int))
            ts = [ts[i] for i in sel]
        self.ts = ts
        self.DT = np.stack([_dt_orientadas(aristas[t].astype(bool)) for t in ts]) if ts else None
        self.V = np.stack([validas[t].astype(bool) for t in ts]) if ts else None
        self.M = np.stack([M[t] for t in ts]) if ts else None


def _proyecta(H, P):
    q = H @ P
    z = q[2]
    ok = z > 1e-9
    return q[0] / np.where(ok, z, 1.0), q[1] / np.where(ok, z, 1.0), ok


def puntua(p, ev, P, tipo, w, h, sub=None, detalle=False, P2=None):
    """La recompensa de las aristas (sin priors), MEDIA por cuadro de la ventana. `P2` son los mismos
    puntos desplazados a lo largo de su raya, para saber la orientacion de la raya en la imagen."""
    Hp, d, C = camara(p, w, h)
    Pw = np.c_[P[0], -P[1], np.zeros(P.shape[1])]
    delante = (Pw - C) @ d > 0.5
    peso = np.where(tipo == 2, PESO_TRANS, 1.0)
    lam = np.where((tipo == 2) & (np.abs(P[0]) < 1e-6), 0.0, LAMBDA)
    """La raya CENTRAL suele ser discontinua: sus huecos no penalizan (pero sus trazos si cuentan)."""
    idx = list(range(len(ev.ts))) if sub is None else list(sub)
    tot = 0.0
    n_ok = 0
    por_linea = []
    for i in idx:
        Ht = ev.M[i] @ Hp
        u, v, ok = _proyecta(Ht, P)
        u2, v2, ok2 = _proyecta(Ht, P2)
        ang = (np.degrees(np.arctan2(v2 - v, u2 - u))) % 180.0
        u = u * ev.s
        v = v * ev.s
        ok &= ok2 & delante & (u >= 0) & (u < ev.aw - 1) & (v >= 0) & (v < ev.ah - 1)
        ui, vi = u[ok].astype(int), v[ok].astype(int)
        val = ev.V[i][vi, ui]
        ui, vi = ui[val], vi[val]
        b = np.minimum((ang[ok][val] / (180.0 / N_BINS)).astype(int), N_BINS - 1)
        D = ev.DT[i]
        dd = np.minimum(np.minimum(D[b, vi, ui], D[(b + 1) % N_BINS, vi, ui]),
                        D[(b - 1) % N_BINS, vi, ui]).astype(np.float64)
        pw = peso[ok][val]
        lw = lam[ok][val]
        tot += float((pw * (np.maximum(0.0, 1.0 - dd / TAU) - lw * (dd >= TAU))).sum())
        n_ok += int(val.sum())
        if detalle:
            por_linea.append((np.flatnonzero(ok)[val], dd))
    n = max(len(idx), 1)
    if detalle:
        return tot / n, n_ok, por_linea
    return tot / n, n_ok


def prior(p, pies, alt, w, h):
    """Penalizacion (en unidades de recompensa) de lo fisicamente raro: roll, camara lejos del centro,
    distancia y altura, estatura de los tiradores, pies fuera de la pista, orden izquierda-derecha."""
    Xc, Yc, Zc, pan, tilt, roll, lf = p
    pen = 0.5 * (roll / 0.035) ** 2 + 0.5 * (Xc / 4.0) ** 2
    D = -Yc
    pen += 0.5 * (max(0.0, 5.0 - D) / 1.0) ** 2 + 0.5 * (max(0.0, D - 45.0) / 5.0) ** 2
    alfa = np.degrees(np.arctan2(Zc, max(D, 1e-3)))
    pen += 0.5 * (max(0.0, 3.0 - alfa) / 1.5) ** 2 + 0.5 * (max(0.0, alfa - 35.0) / 3.0) ** 2
    Hp, _d, _C = camara(p, w, h)
    try:
        Hi = np.linalg.inv(Hp)
    except np.linalg.LinAlgError:
        return 1e6, [None, None]
    xs = []
    for s in (0, 1):
        f = pies[s]
        if not np.isfinite(f).all():
            xs.append(None)
            continue
        q = Hi @ np.array([f[0], f[1], 1.0])
        if abs(q[2]) < 1e-12:
            return 1e6, [None, None]
        m = q[:2] / q[2]
        xs.append(float(m[0]))
        pen += 0.5 * (m[1] / PIE_Y_SD) ** 2
        """LOS PIES VAN SOBRE LA LINEA CENTRAL de la pista, no solo dentro: sin esto el ajuste
        encontraba una pista APLASTADA sobre el borde lejano (visto el 09/10 en
        AromaticUnfortunateElephant), con las rayas cortas apoyadas en la parte alta de las de
        verdad y los pies por debajo."""
        pen += 0.5 * (max(0.0, abs(m[0]) - 9.0) / 0.5) ** 2
        if np.isfinite(alt[s]):
            a = Hp @ np.array([[m[0] - 0.5, m[0] + 0.5], [m[1], m[1]], [1.0, 1.0]])
            a = a[:2] / a[2]
            ppm = float(np.linalg.norm(a[:, 1] - a[:, 0]))
            hm = alt[s] / max(ppm, 1e-6)
            pen += 0.5 * ((hm - ESTATURA_M) / ESTATURA_SD) ** 2
    if xs[0] is not None and xs[1] is not None:
        pen += 0.5 * (max(0.0, xs[0] - xs[1] + 0.3) / 0.3) ** 2
    return PESO_PRIOR * pen, xs


def objetivo(p, ev, P, tipo, pies, alt, w, h, sub=None, P2=None):
    r, _n = puntua(p, ev, P, tipo, w, h, sub, P2=P2)
    pen, _xs = prior(p, pies, alt, w, h)
    return -(r - pen)


def inicial(x_mid, pies, alt, w, h, D=15.0, Zc=5.0):
    """Camara en (0, -D, Zc) mirando de modo que el punto medio de los pies caiga en (x_mid, 0) y
    con la focal que da a los tiradores su estatura."""
    ok = np.isfinite(pies).all(1)
    if not ok.any():
        return None
    u, v = pies[ok].mean(0)
    am = np.nanmean(alt) if np.isfinite(alt).any() else 0.3 * h
    C = np.array([0.0, -D, Zc])
    P = np.array([x_mid, 0.0, 0.0])
    dist = float(np.linalg.norm(P - C))
    f = max(200.0, (am / ESTATURA_M) * dist)
    pan = np.arctan2(P[0] - C[0], P[1] - C[1]) - np.arctan((u - w / 2.0) / f)
    tilt = np.arctan2(C[2] - P[2], np.hypot(P[0] - C[0], P[1] - C[1])) - np.arctan((v - h / 2.0) / f)
    return np.array([0.0, -D, Zc, pan, tilt, 0.0, np.log(f)])


def _nm(fun, x0, pasos, iters):
    from scipy.optimize import minimize
    simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[i] * pasos[i] for i in range(len(x0))])
    r = minimize(fun, x0, method="Nelder-Mead",
                 options={"initial_simplex": simplex, "maxiter": iters, "xatol": 1e-4, "fatol": 1e-3})
    return r.x, float(r.fun)


def ajusta(ev, pies, alt, w, h, x_grid=np.arange(-7.0, 7.01, 0.75), top=4, otra_m=1.5):
    """QUE DEVUELVE: dict con la camara elegida, su H imagen->pista, la x de cada pie, la recompensa,
    el MARGEN contra la mejor hipotesis cuya x media difiere en >= `otra_m`, y cuantas rayas
    transversales distintas tienen evidencia. None si no hay con que ajustar."""
    if ev.DT is None or not np.isfinite(pies).all(1).any():
        return None
    P, tipo, linea, P2 = puntos_modelo()
    n = len(ev.ts)
    sub = list(range(0, n, max(1, n // 5)))
    if (n - 1) not in sub:
        sub.append(n - 1)
    cands = []
    for xm in x_grid:
        p0 = inicial(float(xm), pies, alt, w, h)
        if p0 is None:
            continue
        libres = [3, 4, 6]

        def f3(q, p0=p0):
            p = p0.copy()
            p[libres] = q
            return objetivo(p, ev, P, tipo, pies, alt, w, h, sub, P2)

        q, fv = _nm(f3, p0[libres], [0.02, 0.01, 0.05], 50)
        p = p0.copy()
        p[libres] = q
        cands.append((fv, float(xm), p))
    if not cands:
        return None
    cands.sort(key=lambda z: z[0])
    elegidos = []
    for fv, xm, p in cands:
        if all(abs(xm - e[1]) >= 1.0 for e in elegidos):
            elegidos.append((fv, xm, p))
        if len(elegidos) >= top:
            break
    finos = []
    for fv, xm, p in elegidos:
        def f7(q):
            return objetivo(q, ev, P, tipo, pies, alt, w, h, None, P2)
        q, fq = _nm(f7, p, [1.0, 2.0, 1.0, 0.02, 0.01, 0.01, 0.05], 300)
        _pen, xs = prior(q, pies, alt, w, h)
        vx = [x for x in xs if x is not None]
        if not vx:
            continue
        finos.append((fq, q, xs, float(np.mean(vx))))
    if not finos:
        return None
    finos.sort(key=lambda z: z[0])
    fb, pb, xsb, xmb = finos[0]
    otras = [z[0] for z in finos[1:] if abs(z[3] - xmb) >= otra_m]
    margen = (min(otras) - fb) if otras else None
    rec, n_ok, det = puntua(pb, ev, P, tipo, w, h, detalle=True, P2=P2)
    vistas = set()
    for idx, dd in det:
        li = linea[idx]
        for j in range(len(LINEAS_M)):
            m = li == j
            if m.sum() >= 6 and (dd[m] < TAU).mean() >= 0.5:
                vistas.add(j)
    Hp, _d, _C = camara(pb, w, h)
    return {"p": pb, "H": np.linalg.inv(Hp), "xs": xsb, "x": xmb, "obj": fb, "rec": rec,
            "margen": margen, "rayas": sorted(LINEAS_M[j] for j in vistas), "n_cuadros": n,
            "n_puntos": n_ok}


def _c3(rayas):
    idx = sorted(LINEAS_M.index(r) for r in rayas)
    return any(idx[i + 1] - idx[i] == 1 and idx[i + 2] - idx[i + 1] == 1 for i in range(len(idx) - 2))


def acepta(res, margen_min, rayas_min, tres_seguidas, plausible):
    """La regla de aceptacion (`CAMARA_TRIPODE_2026-10-09.md`, familia R(m, k) +c3 +p): la misma que
    `02_validacion/auditoria/camara_tripode_2026-10-09/evalua.py:pasa`."""
    if res is None:
        return False
    if margen_min > 0 and (res["margen"] is None or res["margen"] < margen_min):
        return False
    if len(res["rayas"]) < rayas_min:
        return False
    if tres_seguidas and not _c3(res["rayas"]):
        return False
    if plausible:
        xs = res["xs"]
        if xs[0] is None or xs[1] is None:
            return False
        if not (0.5 <= xs[1] - xs[0] <= 5.0) or abs(xs[0]) > 8 or abs(xs[1]) > 8:
            return False
    return True

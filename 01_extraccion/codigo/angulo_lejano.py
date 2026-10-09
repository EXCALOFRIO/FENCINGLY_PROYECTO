"""El angulo del borde LEJANO de la pista en el cuadro del tocado, y la x que da con un modelo por torneo.

POR QUE EXISTE (2026-10-09, `04_documentacion/informes/ANGULO_FUSION_2026-10-09.md`). Una camara que
gira para seguir a los tiradores ve la pista cada vez mas inclinada cuanto mas lejos del centro
mira: el angulo del borde lejano dice hacia que x apunta, SIN identificar ninguna raya
(`RAYAS_ID_2026-10-09.md`). Con un modelo por torneo `x = a + b tan(ang) + c desp` (Huber), ajustado
sobre los clips del mismo torneo que SI tienen homografia, da la x con mediana ~0,7 m. Usado como
DESEMPATE de la homografia (regla D(2): si la homografia y el angulo discrepan mas de 2 m, manda el
angulo), el acierto TOTAL de zona en PRUEBA sube de 0,3089 a 0,3338 (+0,0249, IC97,5 por clip
[+0,0113, +0,0380], n = 2.130; espada +0,006, florete +0,019, sable +0,075).

ESTE MODULO ES LA MEDIDA, MOVIDA: `rasgos()` es `rasgos.py:uno` y `resume()` es `modelo.py:carga`
del agente H (`02_validacion/auditoria/rayas_id_2026-10-09/`), sin cambiar nada, porque lo que paso
el criterio es exactamente eso. Se comprueba contra su `rasgos.jsonl`.

    rasgos(clip, fila_zona)    -> {'ang', 'desp', 'dos_pies', 't'} (None donde no se puede medir)
    x_angulo(refs, fila)       -> x del modelo del torneo SIN el propio clip, o None
"""
from __future__ import annotations

import numpy as np

MIN_REFS = 8
"""Referencias del torneo SIN contar el propio clip. Con menos, no hay modelo."""
UMBRAL_D = 2.0
"""La regla D(2): manda el angulo si `|x_homografia - x_angulo| > 2 m`."""
ESCALA_ESTATURA_M = 0.93 * 1.65
"""`desp` en metros: desplazamiento horizontal de los pies en estaturas por 0,93 x 1,65 m, como en
`modelo.py` (la "longitud corporal" de `largo_cuerpo` es ~0,93 de la estatura)."""


def _ang(l):
    import pista_lineas as PL
    if l is None:
        return None
    a = float(PL._ang(l))
    return a if a < 90 else a - 180


def rasgos(clip, zr, video=None):
    """`rasgos.py:uno` sin las etiquetas. `zr` es la fila de `zona_<clip>.csv`."""
    import cv2
    import pista_lineas as PL
    import rutas as RU
    import zona_tocado as ZT
    mot = zr.get("motivo") or ""
    if not (mot == "ok" or mot.startswith("sin homografia")):
        return {"ang": None, "desp": None, "dos_pies": None, "t": None}
    z = np.load(RU.TRABAJO / ("poses_%s.npz" % clip), allow_pickle=True)
    KP, SC, B, origen = z["KP"], z["SC"], z["B"], z["origen"]
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    t = (int(float(zr["cuadro_usado"])) if mot == "ok"
         else min(int(float(zr["cuadro_tocado"])), T - 1))
    k26 = ZT.a_halpe(KP, SC)
    Bpx = B * np.array([W, H, W, H])
    cap = cv2.VideoCapture(video or RU.video(clip))
    pf = PL.PisteFeatures(ZT.CFG_VIDEO, (W, H))
    fr = []
    k = 0
    while k <= t:
        if k < t - 2:
            if not cap.grab():
                break
            k += 1
            continue
        ok, img = cap.read()
        if not ok:
            break
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H))
        okb = np.isfinite(Bpx[k]).all(1)
        feat = pf.process(img, Bpx[k][okb].astype(np.float32), ZT.BANDA_ROTULO * H, k26[k][okb])
        pies = np.full((2, 2), np.nan, np.float32)
        alt = np.full(2, np.nan, np.float32)
        for s in (0, 1):
            if origen[k, s] in (0, 1) and np.isfinite(Bpx[k, s]).all():
                pies[s] = ZT.apoyo(k26[k, s], Bpx[k, s])
                alt[s] = ZT.largo_cuerpo(k26[k, s])
        ob = PL.observe(feat, pies, alt, (W, H))
        fr.append({"far": _ang(ob.far),
                   "pies_u": [None if not np.isfinite(p).all() else float(p[0]) / W for p in pies],
                   "alt": [None if not np.isfinite(a) else float(a) / H for a in alt]})
        k += 1
    cap.release()
    return dict(resume(fr, W, H), t=t)


def resume(fr, W, H):
    """`modelo.py:carga`: angulo = mediana del borde lejano en los cuadros; `desp` del ultimo."""
    if not fr:
        return {"ang": None, "desp": None, "dos_pies": None}
    angs = [f["far"] for f in fr if f["far"] is not None]
    ang = float(np.median(angs)) if angs else None
    f = fr[-1]
    us = [u for u in f["pies_u"] if u is not None]
    al = [a for a in f["alt"] if a is not None]
    u = float(np.mean(us)) - 0.5 if us else None
    alt_m = float(np.mean(al)) if al else None
    desp = (u * W) / (alt_m * H) * ESCALA_ESTATURA_M if (u is not None and alt_m) else None
    return {"ang": ang, "desp": desp, "dos_pies": len(us) == 2}


def _X(filas):
    return np.array([[np.tan(np.radians(f["ang"])), f["desp"]] for f in filas])


def ajusta(refs):
    from sklearn.linear_model import HuberRegressor
    m = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=500)
    m.fit(_X(refs), np.array([f["x_h"] for f in refs]))
    return m


def x_angulo(refs, fila):
    """La x del angulo para `fila` con el modelo del torneo ajustado SIN ella. `refs` son las filas
    del torneo con homografia, angulo y desplazamiento (pueden incluir la propia fila)."""
    if fila.get("ang") is None or fila.get("desp") is None:
        return None
    otras = [r for r in refs if r["clip"] != fila["clip"]]
    if len(otras) < MIN_REFS:
        return None
    try:
        return float(ajusta(otras).predict(_X([fila]))[0])
    except Exception:
        return None


def decide(x_h, x_a):
    """(x final, via). D(2): el angulo manda si discrepa de la homografia mas de `UMBRAL_D`."""
    if x_h is None:
        return None, None
    if x_a is None:
        return x_h, "homografia"
    if abs(x_h - x_a) > UMBRAL_D:
        return x_a, "angulo_torneo"
    return x_h, "homografia"

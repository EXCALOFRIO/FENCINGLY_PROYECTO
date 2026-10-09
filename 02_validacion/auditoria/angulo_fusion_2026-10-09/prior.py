"""Idea 2 de ANGULO_FUSION_2026-10-09: el angulo del borde lejano como PRIOR de la x en
`pista_lineas.hypothesize`. Reproduce `zona_tocado.analiza` con los rasgos decodificados UNA vez y
`calibrate` con el prior de siempre (control) y con P(s).
    python prior.py AJUSTE 600 salida.jsonl [procs]
"""
import sys, csv, json, os, random, time, collections
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\auditoria\rayas_id_2026-10-09")
S_VALORES = (1.5, 3.0)
_MOD = {}   # en el proceso hijo: el modelo del torneo (sin el clip), W, H y la s activa


def _x_ang(obs, feet, heights):
    import pista_lineas as PL
    M = _MOD.get("m")
    if M is None or obs.far is None:
        return None
    a = float(PL._ang(obs.far))
    a = a if a < 90 else a - 180
    W, H = _MOD["W"], _MOD["H"]
    ok = ~np.isnan(feet).any(1)
    if not ok.any():
        return None
    u = float(np.mean(feet[ok, 0])) / W - 0.5
    hh = heights[np.isfinite(heights)]
    if not len(hh):
        return None
    alt = float(np.mean(hh)) / H
    desp = (u * W) / (alt * H) * (0.93 * 1.65)
    return float(M.predict(np.array([[np.tan(np.radians(a)), desp]]))[0])


def _instala():
    import pista_lineas as PL
    if getattr(PL, "_PARCHE_ANGULO", False):
        return
    hyp0, prior0 = PL.hypothesize, PL._scale_prior

    def scale_prior(H, feet, heights, cfg):
        px = cfg.get("prior_x")
        ll = prior0(H, feet, heights, cfg)
        if px is None or ll <= -1e8:
            return ll
        mu, s = px
        xs = []
        for f, hh in zip(feet, heights):
            if np.isnan(f).any() or not np.isfinite(hh):
                continue
            m = PL.to_m(H, f[None])[0]
            if not np.isfinite(m).all():
                return -1e9
            xs.append(m[0])
            ll -= -0.5 * (m[0] / 5.0) ** 2      # quita el "cerca del centro" de siempre
        if xs:
            ll += -0.5 * ((float(np.mean(xs)) - mu) / s) ** 2
        return ll

    def hypothesize(obs, feet, heights, cfg, max_hyp=3):
        s = _MOD.get("s")
        if s is not None:
            xa = _x_ang(obs, feet, heights)
            if xa is not None:
                cfg = dict(cfg, prior_x=(xa, s))
        return hyp0(obs, feet, heights, cfg, max_hyp)

    PL._scale_prior = scale_prior
    PL.hypothesize = hypothesize
    PL._PARCHE_ANGULO = True


def analiza_varios(clip, torneo, modelo):
    """Como `zona_tocado.analiza` (sin respaldos, que estan apagados), decodificando una vez y
    calibrando con el prior de siempre y con cada P(s)."""
    import pista_lineas as PL
    import zona_tocado as ZT
    import rutas as RU
    _instala()
    out = {"clip": clip}
    video = RU.video(clip)
    z = np.load(RU.TRABAJO / ("poses_%s.npz" % clip), allow_pickle=True)
    KP, SC, B, origen = z["KP"], z["SC"], z["B"], z["origen"]
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    fps = float(z["fps"]) if "fps" in z.files else 30.0
    tf, via = ZT.cuadro_tocado(clip, video, torneo)
    if tf is None or tf >= T:
        out["res"] = {}
        return out
    k26 = ZT.a_halpe(KP, SC)
    Bpx = B * np.array([W, H, W, H])
    pf = ZT._rasgos(video, W, H, T, tf, Bpx, k26, ZT.BANDA_ROTULO)
    Tn = len(pf.feats)
    pies = np.full((Tn, 2, 2), np.nan, np.float32)
    alt = np.full((Tn, 2), np.nan, np.float32)
    for t in range(Tn):
        for s_ in (0, 1):
            if origen[t, s_] in (0, 1) and np.isfinite(Bpx[t, s_]).all():
                pies[t, s_] = ZT.apoyo(k26[t, s_], Bpx[t, s_])
                alt[t, s_] = ZT.largo_cuerpo(k26[t, s_])
    _MOD.update(m=modelo, W=W, H=H)
    res = {}
    for s in (None,) + S_VALORES:
        _MOD["s"] = s
        cal = PL.calibrate(pf.feats, pies, alt, ZT.CFG_PISTA, (W, H))
        Hs, conf = cal["H"], cal["conf"]
        tq = min(tf, Tn - 1)
        w = int(ZT.VENTANA_S * fps)
        cand = [t for t in range(max(0, tq - w), min(Tn, tq + 2))
                if np.isfinite(Hs[t]).all() and conf[t] >= ZT.CFG_PISTA["min_conf"]]
        clave = "base" if s is None else "P(%g)" % s
        if not cand:
            res[clave] = None
            continue
        tt = min(cand, key=lambda t: abs(t - tq))
        xs = [float(PL.to_m(Hs[tt], pies[tt, k][None])[0, 0]) if np.isfinite(pies[tt, k]).all() else None
              for k in (0, 1)]
        vx = [x for x in xs if x is not None]
        if not vx:
            res[clave] = None
            continue
        x = float(np.mean(vx))
        res[clave] = {"x": round(x, 3), "zona": ZT.ZONAS[ZT.zona(x)]}
    _MOD["s"] = None
    out["res"] = res
    return out


def main():
    import catalogo as CA
    import rutas as RU
    import modelo as MO
    mitad, n, sal = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    rs = [r for r in MO.carga("rasgos.jsonl") if r["mitad"] == mitad]
    por = collections.defaultdict(list)
    for r in rs:
        if r["motivo"] == "ok" and r["ang"] is not None and r["x_prod"] is not None and r["desp"] is not None:
            por[r["torneo"]].append(r)
    df = CA.manifiesto()
    df = df[df["strip_location"].isin(CA.ZONAS)]
    U = [(m["clip_id"], m["torneo"]) for _, m in df.iterrows()
         if ((m["torneo"] in MO.TORN_AJ) == (mitad == "AJUSTE"))
         and (RU.TRABAJO / ("zona_%s.csv" % m["clip_id"])).is_file()]
    U.sort()
    random.Random(20261009).shuffle(U)
    U = U[:n] if n > 0 else U
    hechos = set()
    if os.path.isfile(sal):
        hechos = {json.loads(l)["clip"] for l in open(sal, encoding="utf-8")}
    tareas, sin_modelo = [], []
    for c, t in U:
        if c in hechos:
            continue
        refs = [r for r in por.get(t, []) if r["clip"] != c]
        if len(refs) < 8:
            sin_modelo.append(c)
            continue
        tareas.append((c, t, MO.ajusta(refs, True, True)))
    with open(sal, "a", encoding="utf-8") as fh:
        for c in sin_modelo:
            fh.write(json.dumps({"clip": c, "sin_modelo": True}) + "\n")
    print("%s: muestra %d; sin modelo de torneo %d (no cambian); por calcular %d"
          % (mitad, len(U), len(sin_modelo), len(tareas)), flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(procs) as ex, open(sal, "a", encoding="utf-8") as fh:
        fut = {ex.submit(analiza_varios, c, t, m): c for c, t, m in tareas}
        for i, f in enumerate(as_completed(fut), 1):
            try:
                r = f.result()
            except Exception as e:
                r = {"clip": fut[f], "error": "%s: %s" % (type(e).__name__, str(e)[:200])}
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if i % 50 == 0:
                print("  %d/%d %.0f s" % (i, len(tareas), time.time() - t0), flush=True)
    print("hecho %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

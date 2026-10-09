"""Rasgos EXPLICITOS de esgrima anclados al cuadro del tocado + un clasificador pequeno, contra el GRU.

El protocolo y el criterio estan en `04_documentacion/informes/RASGOS_EXPLICITOS_2026-10-09.md`, y
se escribieron antes que este fichero. Aqui no se decide nada que no este alli.

POR QUE. El etiquetador (GRU) ve el clip entero remuestreado a 64 pasos y NO sabe donde esta el
tocado. Estos rasgos si: todo se mide en segundos ANTES del cuadro del tocado, por tirador y en su
marco de ataque (adelante = hacia el rival), asi que espejar el clip es exactamente intercambiar los
dos bloques de rasgos. Sobre eso, la antisimetria se IMPONE: se entrena con cada clip y su espejo, y
se predice promediando el clip y su espejo con izquierda/derecha cambiadas.

    python rasgos_explicitos.py                 rasgos (en cache) + las dos tareas
    python rasgos_explicitos.py --solo-rasgos   solo calcula y guarda los rasgos
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "01_extraccion" / "codigo"))
sys.path.insert(0, str(RAIZ / "01_extraccion"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import etiquetador as ET  # noqa: E402
import rutas as RU  # noqa: E402

CACHE = RU.SCRATCH / "agente_rasgos" / "rasgos_explicitos.npz"
UMBRAL_V = 0.25
"""Altos de caja por segundo para llamar "avanzar"/"retroceder". Una marcha de 0,5-1 m/s con un
tirador de ~1,7 m de caja es ~0,3-0,6 altos/s; 0,25 deja fuera el temblor de la caja."""
EVENTOS = ("marcha", "romper", "fondo", "brazo")

POR_TIRADOR = (
    ["v_ult1s", "v_1a2s", "v_2a3s", "frac_avanza_1s", "frac_retrocede_1s", "acel_final",
     "avanza_en_tocado", "t_inicio_avance", "ext_tocado", "ext_max_1s", "t_ext80",
     "vext_pico", "t_vext_pico", "t_inicio_pie", "brazo_menos_pie", "parte_cierre_1s",
     "m_a_su_fondo"]
    + ["t_ult_%s" % e for e in EVENTOS] + ["n_%s_2s" % e for e in EVENTOS])
COMPARTIDOS = ["dist_menos1s", "dist_menos05s", "dist_tocado", "t_ult_accion", "via_ancla",
               "fps"]


# ---------------------------------------------------------------------------------------------
# RASGOS
# ---------------------------------------------------------------------------------------------

def _media(a):
    a = np.asarray(a, dtype="float64")
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def _maxi(a):
    a = np.asarray(a, dtype="float64")
    a = a[np.isfinite(a)]
    return float(a.max()) if len(a) else float("nan")


def _inicio_racha(v, a, fps, umbral):
    """Segundos antes de `a` en que empieza la racha de `v > umbral` que llega hasta `a` (se
    perdona un hueco de hasta 2 cuadros sin medir). NaN si en `a` no se esta por encima."""
    cola = v[max(0, a - 2):a + 1]
    if not np.isfinite(cola).any() or _media(cola) <= umbral:
        return float("nan")
    k, huecos = a, 0
    while k > 0:
        x = v[k - 1]
        if not np.isfinite(x):
            huecos += 1
            if huecos > 2:
                break
        elif x <= umbral * 0.5:
            break
        else:
            huecos = 0
        k -= 1
    return (a - k) / fps


def _eventos(clip):
    p = RU.PISTA / ("serie_eventos_%s.csv" % clip)
    if not p.is_file():
        return None
    ev = []
    with open(p, encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            try:
                ev.append((int(f["lado"]), f["tipo"], int(float(f["cuadro_inicio"]))))
            except (ValueError, TypeError, KeyError):
                continue
    return ev


def _zona(clip):
    p = RU.TRABAJO / ("zona_%s.csv" % clip)
    if not p.is_file():
        return float("nan"), float("nan")
    with open(p, encoding="utf-8") as fh:
        r = next(csv.DictReader(fh), None) or {}
    out = []
    for k in ("x_izq_m", "x_der_m"):
        try:
            v = float(r.get(k))
        except (TypeError, ValueError):
            v = float("nan")
        out.append(v if math.isfinite(v) else float("nan"))
    return out[0], out[1]


def rasgos_de(clip):
    """QUE DEVUELVE: (clip, rasgos (2, n_por_tirador), compartidos (n_compartidos)) o (clip, None, motivo)."""
    import vjepa as VJ
    p = RU.TRABAJO / ("poses_%s.npz" % clip)
    if not p.is_file():
        return clip, None, "sin poses"
    try:
        with np.load(p, allow_pickle=True) as z:
            KP, SC, B, pres, mask = z["KP"], z["SC"], z["B"], z["pres"], z["mask"]
            W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
            fps = float(z["fps"]) if "fps" in z.files else 0.0
    except Exception as e:
        return clip, None, "poses ilegibles %s" % type(e).__name__
    fps = fps or 25.0
    if mask.sum() < 8:
        return clip, None, "menos de 8 cuadros utiles"
    a, via = VJ.ancla(clip, mask, fps)
    a = int(min(max(a, 0), T - 1))
    n1 = int(round(fps))
    valido = mask[:, None] & pres.astype(bool)
    cx = 0.5 * (B[:, :, 0] + B[:, :, 2]) * W
    bh = (B[:, :, 3] - B[:, :, 1]) * H
    cx = np.where(valido, cx, np.nan)
    bh = np.where(valido & (bh > 1), bh, np.nan)
    hmed = np.array([np.nanmedian(bh[:, j]) if np.isfinite(bh[:, j]).any() else np.nan
                     for j in (0, 1)])
    hpar = float(np.nanmean(hmed))
    dist = np.abs(cx[:, 1] - cx[:, 0]) / hpar if math.isfinite(hpar) else np.full(T, np.nan)
    ev = _eventos(clip)
    xz = _zona(clip)
    signo = (1.0, -1.0)
    filas = []
    vads = []
    for j in (0, 1):
        s, h = signo[j], hmed[j]
        kp = KP[:, j].astype("float64")
        sc = SC[:, j]

        def pt(i):
            xy = kp[:, i, :].copy()
            xy[(sc[:, i] < 0.3) | ~valido[:, j]] = np.nan
            return xy

        ho = [pt(i) for i in ET.HOMBROS]
        ma = [pt(i) for i in ET.MANOS_RAIZ]
        to = [pt(i) for i in ET.TOBILLOS]
        extx = ET._nanmax(np.stack([s * (ma[0][:, 0] - ho[1][:, 0]) / h,
                                    s * (ma[1][:, 0] - ho[0][:, 0]) / h]))
        pie = ET._nanmax(np.stack([s * to[0][:, 0], s * to[1][:, 0]])) / h
        vad = s * np.gradient(ET._suave(cx[:, j])) * fps / h
        vpie = np.gradient(ET._suave(pie)) * fps
        vext = np.gradient(ET._suave(extx)) * fps
        vads.append(vad)
        u1 = vad[max(0, a - n1):a + 1]
        fin = vad[max(0, a - int(round(0.3 * fps))):a + 1]
        resto = vad[max(0, a - n1):max(0, a - int(round(0.3 * fps)))]
        ok1 = np.isfinite(u1)
        e15 = extx[max(0, a - int(round(1.5 * fps))):a + 1]
        mx15 = _maxi(e15)
        if math.isfinite(mx15) and mx15 > 0:
            cruza = np.nonzero(np.nan_to_num(e15, nan=-1e9) >= 0.8 * mx15)[0]
            t_ext = (len(e15) - 1 - cruza[0]) / fps if len(cruza) else float("nan")
        else:
            t_ext = float("nan")
        ve15 = vext[max(0, a - int(round(1.5 * fps))):a + 1]
        if np.isfinite(ve15).any():
            kpk = int(np.nanargmax(ve15))
            vpico, t_vpico = float(ve15[kpk]), (len(ve15) - 1 - kpk) / fps
        else:
            vpico, t_vpico = float("nan"), float("nan")
        t_av = _inicio_racha(vad, a, fps, UMBRAL_V)
        t_pie = _inicio_racha(vpie, a, fps, UMBRAL_V)
        avanza = float("nan") if not np.isfinite(vad[max(0, a - 2):a + 1]).any() else \
            float(_media(vad[max(0, a - 2):a + 1]) > UMBRAL_V)
        if ev is None:
            t_ult = [float("nan")] * len(EVENTOS)
            n2 = [float("nan")] * len(EVENTOS)
        else:
            t_ult, n2 = [], []
            for e in EVENTOS:
                ks = [k for (l, t, k) in ev if l == j and t == e and k <= a]
                t_ult.append((a - max(ks)) / fps if ks else float("nan"))
                n2.append(float(sum(1 for k in ks if k >= a - 2 * n1)))
        x = xz[j]
        fila = [
            _media(u1), _media(vad[max(0, a - 2 * n1):max(0, a - n1)]),
            _media(vad[max(0, a - 3 * n1):max(0, a - 2 * n1)]),
            float((u1[ok1] > UMBRAL_V).mean()) if ok1.any() else float("nan"),
            float((u1[ok1] < -UMBRAL_V).mean()) if ok1.any() else float("nan"),
            _media(fin) - _media(resto), avanza, t_av,
            _media(extx[max(0, a - 2):a + 1]), _maxi(extx[max(0, a - n1):a + 1]), t_ext,
            vpico, t_vpico, t_pie,
            (t_ext - t_pie) if (math.isfinite(t_ext) and math.isfinite(t_pie)) else float("nan"),
            float("nan"),
            (x + 7.0) if j == 0 else (7.0 - x),
        ] + t_ult + n2
        filas.append(fila)
    c = [np.nansum(np.clip(v[max(0, a - n1):a + 1], 0, None)) / fps
         if np.isfinite(v[max(0, a - n1):a + 1]).any() else float("nan") for v in vads]
    tot = c[0] + c[1] if all(math.isfinite(x) for x in c) else float("nan")
    for j in (0, 1):
        filas[j][POR_TIRADOR.index("parte_cierre_1s")] = (
            c[j] / tot if math.isfinite(tot) and tot > 0.05 else float("nan"))

    def dist_en(k):
        return _media(dist[max(0, k - 3):k + 4])

    acc = [k for (l, t, k) in (ev or []) if l == -1 and t == "accion" and k <= a]
    comp = [dist_en(a - n1), dist_en(a - int(round(0.5 * fps))), dist_en(a),
            (a - max(acc)) / fps if acc else float("nan"),
            {"zona": 0.0, "luces": 1.0, "final": 2.0}[via], fps / 30.0]
    return clip, np.array(filas, dtype="float32"), np.array(comp, dtype="float32")


def calcula_rasgos(clips, procs=4):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    hechos = {}
    if CACHE.is_file():
        with np.load(CACHE, allow_pickle=True) as z:
            for c, r, s in zip(z["clips"], z["prop"], z["comp"]):
                hechos[str(c)] = (r, s)
    pend = [c for c in clips if c not in hechos]
    print("rasgos: %d en cache, %d por calcular" % (len(hechos), len(pend)), flush=True)
    malos = {}
    t0 = time.time()
    if pend:
        with ProcessPoolExecutor(procs) as ex:
            for i, (c, r, s) in enumerate(ex.map(rasgos_de, pend, chunksize=16), 1):
                if r is None:
                    malos[c] = s
                else:
                    hechos[c] = (r, s)
                if i % 500 == 0:
                    print("   %d / %d  %.0f s" % (i, len(pend), time.time() - t0), flush=True)
        cs = sorted(hechos)
        tmp = CACHE.with_name(CACHE.stem + ".part.npz")
        np.savez(tmp, clips=np.array(cs), prop=np.stack([hechos[c][0] for c in cs]),
                 comp=np.stack([hechos[c][1] for c in cs]))
        tmp.replace(CACHE)
    if malos:
        print("   sin rasgos: %d (%s)" % (len(malos), sorted(set(malos.values()))[:4]))
    return hechos


# ---------------------------------------------------------------------------------------------
# CLASIFICADOR ANTISIMETRICO
# ---------------------------------------------------------------------------------------------

def matriz(filas, idx, R):
    """x = [rasgos izq, rasgos der, compartidos]; x' = [der, izq, compartidos]."""
    P = np.stack([R[filas[i]["clip"]][0] for i in idx])
    S = np.stack([R[filas[i]["clip"]][1] for i in idx])
    x = np.concatenate([P[:, 0], P[:, 1], S], axis=1)
    xe = np.concatenate([P[:, 1], P[:, 0], S], axis=1)
    return x.astype("float64"), xe.astype("float64")


def espejo_y(y):
    return np.where(y == 0, 1, np.where(y == 1, 0, y))


def gbm():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, min_samples_leaf=20,
                                          early_stopping=False, random_state=ET.SEMILLA)


def entrena_gbm(x, xe, y):
    m = gbm()
    m.fit(np.concatenate([x, xe]), np.concatenate([y, espejo_y(y)]))
    return m


def proba(m, x, xe, n_clases):
    """La media de p(x) y de p(x') con izquierda/derecha cambiadas: antisimetrica por construccion."""
    def full(p):
        out = np.zeros((len(p), n_clases))
        out[:, m.classes_] = p
        return out
    p = full(m.predict_proba(x))
    pe = full(m.predict_proba(xe))
    q = p.copy()
    q[:, 0] = 0.5 * (p[:, 0] + pe[:, 1])
    q[:, 1] = 0.5 * (p[:, 1] + pe[:, 0])
    if n_clases > 2:
        q[:, 2] = 0.5 * (p[:, 2] + pe[:, 2])
    return q


def decide(q, tarea):
    return q[:, :2].argmax(1) if tarea == "prioridad" else q.argmax(1)


def control_espejo(m, x, xe, n_clases, tarea):
    a = decide(proba(m, x, xe, n_clases), tarea)
    b = decide(proba(m, xe, x, n_clases), tarea)
    q = proba(m, x, xe, n_clases)
    empate = np.isclose(q[:, 0], q[:, 1])
    ok = espejo_y(a) == b
    return float(ok[~empate].mean()) if (~empate).any() else float("nan"), int(empate.sum())


# ---------------------------------------------------------------------------------------------
# LA MEDIDA
# ---------------------------------------------------------------------------------------------

def logits_gru(mod, filas, idx):
    import torch
    P, _A, S, _y = ET._tensores(filas, idx, "initiated", torch)
    with torch.no_grad():
        return mod.inicia(P, S).numpy().astype("float64")


def partes_gbm(q):
    eps = 1e-6
    imp = 0.5 * np.log((q[:, 0] + eps) / (q[:, 1] + eps))
    if q.shape[1] < 3:
        """En `prioridad` no hay `neither`: el GBM no aporta parte par."""
        return imp[:, None], np.zeros((len(q), 0))
    par = np.log(q[:, 2] + eps) - 0.5 * np.log((q[:, 0] + eps) * (q[:, 1] + eps))
    return imp[:, None], par[:, None]


def importancia(m, x, xe, y, n_clases, tarea, nombres, reps=5, semilla=ET.SEMILLA):
    """Por permutacion, PERMUTANDO A LA VEZ la columna del izquierdo y la del derecho (con la
    misma permutacion), para no romper la antisimetria. Sobre la particion INTERNA."""
    rng = np.random.default_rng(semilla)
    base = float((decide(proba(m, x, xe, n_clases), tarea) == y).mean())
    nP = len(POR_TIRADOR)
    out = []
    for k, nom in enumerate(nombres):
        cols = [k, k + nP] if k < nP else [2 * nP + (k - nP)]
        cols_e = [k + nP, k] if k < nP else cols
        caida = []
        for _ in range(reps):
            per = rng.permutation(len(y))
            xp, xpe = x.copy(), xe.copy()
            for c, ce in zip(cols, cols_e):
                xp[:, c] = x[per, c]
                xpe[:, ce] = xe[per, ce]
            caida.append(base - float((decide(proba(m, xp, xpe, n_clases), tarea) == y).mean()))
        out.append((nom, float(np.mean(caida))))
    return sorted(out, key=lambda t: -t[1])


def mide(tarea, filas, R, verboso=True):
    import sonda_vjepa as SV
    n_cl = 3 if tarea == "initiated" else 2
    filas = [f for f in filas if f["clip"] in R]
    y = np.array([f["y_initiated"] for f in filas])
    torneos = np.array([f["torneo"] for f in filas])
    armas = np.array([f["weapon"] for f in filas])
    print()
    print("=" * 80)
    print("TAREA %s: %d filas con rasgos" % (tarea, len(filas)))
    mod, ev = ET.entrena(filas, "initiated", verboso=False)
    tr, re = np.array(ev["itr"]), np.array(ev["ire"])
    print("reparto por torneo: entrenamiento %d clips / %d torneos, reservado %d / %d"
          % (len(tr), ev["diag"]["torneos_tr"], len(re), ev["diag"]["torneos_re"]))
    Lg = logits_gru(mod, filas, re)
    pred_g = decide(Lg, tarea)
    x_tr, xe_tr = matriz(filas, tr, R)
    x_re, xe_re = matriz(filas, re, R)
    m = entrena_gbm(x_tr, xe_tr, y[tr])
    q = proba(m, x_re, xe_re, n_cl)
    pred_r = decide(q, tarea)
    esp, emp = control_espejo(m, x_re, xe_re, n_cl, tarea)

    # fusion: pesos en la particion INTERNA del entrenamiento
    filas_tr = [filas[i] for i in tr]
    mod_i, ev_i = ET.entrena(filas_tr, "initiated", verboso=False)
    itr_i, ire_i = np.array(ev_i["itr"]), np.array(ev_i["ire"])
    Lg_i = logits_gru(mod_i, filas_tr, ire_i)
    xi_tr, xie_tr = matriz(filas_tr, itr_i, R)
    xi_va, xie_va = matriz(filas_tr, ire_i, R)
    m_i = entrena_gbm(xi_tr, xie_tr, y[tr][itr_i])
    q_i = proba(m_i, xi_va, xie_va, n_cl)
    yv = y[tr][ire_i]
    fus = SV.ajusta_fusion(*SV.rasgos_fusion([SV.parte_logits(Lg_i), partes_gbm(q_i)]), yv)
    pred_f = decide(SV.aplica_fusion(fus, *SV.rasgos_fusion([SV.parte_logits(Lg),
                                                             partes_gbm(q)])), tarea)
    yr, tor = y[re], torneos[re]
    may = float((yr == np.bincount(y[tr]).argmax()).mean())
    acc_g, acc_r, acc_f = (float((p == yr).mean()) for p in (pred_g, pred_r, pred_f))
    lo, hi = SV.boot_torneo(yr, pred_g, pred_r, tor)
    lo_f, hi_f = SV.boot_torneo(yr, pred_g, pred_f, tor)
    print("%-40s %8s" % ("", "acierto"))
    print("%-40s %8.3f" % ("clase mayoritaria del entrenamiento", may))
    if tarea == "initiated":
        a_b, cob, a_b2 = ET.base_brazo_primero(filas, re)
        print("%-40s %8.3f  (cobertura %.3f; %.3f contando mudos)"
              % ("quien extiende el brazo primero", a_b, cob, a_b2))
    else:
        yh = np.array([filas[i]["y_humano"] for i in re])
        mm = yh <= 1
        print("%-40s %8.3f  (cobertura %.3f; etiqueta HUMANA)"
              % ("gana el que inicio", float((yh[mm] == yr[mm]).mean()), mm.mean()))
    print("%-40s %8.3f" % ("GRU (etiquetador)", acc_g))
    print("%-40s %8.3f   %+.3f  IC95 por torneo [%+.3f, %+.3f]"
          % ("RASGOS explicitos (GBM)", acc_r, acc_r - acc_g, lo, hi))
    print("%-40s %8.3f   %+.3f  IC95 por torneo [%+.3f, %+.3f]  (informativo)"
          % ("fusion GRU + rasgos", acc_f, acc_f - acc_g, lo_f, hi_f))
    print("control del espejo de los rasgos: %.4f  (empates exactos izq = der: %d)" % (esp, emp))
    pasa = (acc_r - acc_g >= 0.02) and lo > 0 and esp == 1.0
    print("CRITERIO: rasgos - GRU = %+.3f >= +0,02, IC95 [%+.3f, %+.3f] sin cruzar cero, espejo "
          "%.4f  ->  %s" % (acc_r - acc_g, lo, hi, esp, "PASA" if pasa else "NO PASA"))
    print()
    print("por arma (informativo):")
    for g in np.unique(armas[re]):
        mm = armas[re] == g
        print("   %-6s n=%4d  GRU %.3f  rasgos %.3f  fusion %.3f"
              % (g, mm.sum(), (pred_g[mm] == yr[mm]).mean(), (pred_r[mm] == yr[mm]).mean(),
                 (pred_f[mm] == yr[mm]).mean()))
    print()
    print("palancas: caida de acierto al permutar cada rasgo (particion interna, n=%d):" % len(yv))
    imp = importancia(m_i, xi_va, xie_va, yv, n_cl, tarea, POR_TIRADOR + COMPARTIDOS)
    for nom, c in imp[:12]:
        print("   %-22s %+.4f" % (nom, c))
    return {"tarea": tarea, "n_re": len(re), "gru": acc_g, "rasgos": acc_r, "fusion": acc_f,
            "ic": (lo, hi), "ic_f": (lo_f, hi_f), "espejo": esp, "pasa": pasa, "imp": imp}


def filas_prioridad_de(todas):
    """La misma seleccion que `sonda_vjepa.filas_prioridad`, sobre COPIAS de las filas."""
    out = []
    for f in todas:
        if f["weapon"] not in ("foil", "sabre") or f["touch"] not in ("left", "right"):
            continue
        p = RU.TRABAJO / ("lights_%s.csv" % f["clip"])
        if not p.is_file():
            continue
        with open(p, encoding="utf-8") as fh:
            lados = {r.get("lado") for r in csv.DictReader(fh) if r.get("canal") in ("roja", "verde")}
        if lados != {"L", "R"}:
            continue
        g = dict(f)
        g["y_humano"] = f["y_initiated"]
        g["y_initiated"] = 0 if f["touch"] == "left" else 1
        out.append(g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-rasgos", action="store_true")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--tarea", choices=("ambas", "initiated", "prioridad"), default="ambas")
    a = ap.parse_args()
    print("RASGOS EXPLICITOS (protocolo RASGOS_EXPLICITOS_2026-10-09)")
    todas = ET.reune(cabeza=None)
    R = calcula_rasgos([f["clip"] for f in todas], a.procs)
    allP = np.stack([R[c][0] for c in R])
    allS = np.stack([R[c][1] for c in R])
    print("cobertura por rasgo (fraccion no NaN, sobre %d clips x 2 tiradores):" % len(R))
    for k, nom in enumerate(POR_TIRADOR):
        print("   %-22s %.3f" % (nom, float(np.isfinite(allP[:, :, k]).mean())))
    for k, nom in enumerate(COMPARTIDOS):
        print("   %-22s %.3f" % (nom, float(np.isfinite(allS[:, k]).mean())))
    if a.solo_rasgos:
        return 0
    ini = [f for f in todas if f["y_initiated"] >= 0]
    pri = filas_prioridad_de(todas)
    print("filas: initiated %d, prioridad %d (florete %d, sable %d)"
          % (len(ini), len(pri), sum(f["weapon"] == "foil" for f in pri),
             sum(f["weapon"] == "sabre" for f in pri)))
    if a.tarea in ("ambas", "initiated"):
        mide("initiated", ini, R)
    if a.tarea in ("ambas", "prioridad"):
        mide("prioridad", pri, R)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""La medida de V-JEPA 2 contra el etiquetador en `initiated`, con el protocolo DECLARADO de antes.

El protocolo, el criterio y su cambio 1 estan en `04_documentacion/informes/VJEPA2_2026-10-09.md`
y se escribieron antes de que existiera este fichero. Aqui no se decide nada que no este alli.

QUE SE COMPARA, en los clips del RESERVADO por torneo del etiquetador (misma semilla):
  A    el etiquetador de poses, reentrenado con los datos de hoy
  A'   A + las luces de la etapa 9 (fusion logistica impar)            <- linea base que decide
  B    sonda ATENTIVA sobre los tokens de V-JEPA 2 (impar por construccion)
  Blin sonda lineal sobre la media de los tokens
  C    A + B (fusion)
  C'   A + luces + B (fusion)                                           <- lo que se adopta o no

COMO SE ELIGE LO QUE SE ELIGE, sin mirar el reservado: la rejilla de B (epocas x weight_decay) y los
pesos de las fusiones se ajustan en una particion INTERNA por torneo del entrenamiento (la que hace
`etiquetador.entrena` sobre las filas de entrenamiento). Despues se reentrena en todo el
entrenamiento con lo elegido y se evalua UNA vez en el reservado.

LA ANTISIMETRIA, impuesta en todo: con `x` el clip y `x'` su espejo, el logit de `fotl` es
`g(x) - g(x')`, el de `fotr` su opuesto y el de `neither` `s(x) + s(x')`. En las fusiones, los
terminos impares (logit del etiquetador, luz izquierda menos derecha) solo entran en `fotl - fotr`
y los pares (`neither` del etiquetador, las dos luces) solo en `neither`. El control del espejo se
comprueba igual que en el careo del etiquetador.

    python sonda_vjepa.py [--dir D:/FENCINGLY_SCRATCH/vjepa] [--rapido]
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "01_extraccion" / "codigo"))
import catalogo as CA  # noqa: E402
import etiquetador as ET  # noqa: E402
import rutas as RU  # noqa: E402

REJILLA_EPOCAS = (20, 40, 80)
REJILLA_WD = (1e-3, 1e-2, 5e-2)
B_BOOT = 2000


# ---------------------------------------------------------------------------------------------
# DATOS
# ---------------------------------------------------------------------------------------------

def luces_clip(clip):
    """(L de color, R de color, L blanca, R blanca) del grupo del tocado; None si no hay CSV."""
    p = RU.TRABAJO / ("lights_%s.csv" % clip)
    if not p.is_file():
        return None
    v = np.zeros(4, dtype="float32")
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            lado, canal = r.get("lado"), r.get("canal")
            if lado not in ("L", "R"):
                continue
            j = (0 if lado == "L" else 1) + (2 if canal == "blanca" else 0)
            v[j] = 1.0
    return v


def carga_tokens(clips, d):
    """QUE DEVUELVE: (tok (n, 2, 512, 1024) fp16, media (n, 2, 1024) fp32, via_ancla (n,)), en el
    orden de `clips`. Un clip sin fichero es un error aqui: se filtra antes."""
    n = len(clips)
    tok = np.empty((n, 2, 512, 1024), dtype="float16")
    med = np.empty((n, 2, 1024), dtype="float32")
    via = []
    for i, c in enumerate(clips):
        with np.load(d / ("vjepa_%s.npz" % c)) as z:
            tok[i] = z["tok"].reshape(2, 512, 1024)
            med[i] = z["media"]
            via.append(str(z["via_ancla"]))
    return tok, med, np.array(via)


# ---------------------------------------------------------------------------------------------
# SONDAS, impares por construccion
# ---------------------------------------------------------------------------------------------

def _torch():
    import torch
    return torch


def sonda_atentiva(d_in=1024, d=256, cabezas=4, n_tok=512):
    torch = _torch()
    nn = torch.nn

    class GS(nn.Module):
        """`g` y `s` de UNA vista: proyeccion, posicion aprendida, una consulta que atiende."""

        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(d_in)
            self.proj = nn.Linear(d_in, d)
            self.pos = nn.Parameter(torch.zeros(1, n_tok, d))
            self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.att = nn.MultiheadAttention(d, cabezas, batch_first=True, dropout=0.1)
            self.sal = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(),
                                     nn.Dropout(0.1), nn.Linear(128, 2))

        def forward(self, x):
            h = self.proj(self.ln(x)) + self.pos
            o, _ = self.att(self.q.expand(len(h), -1, -1), h, h, need_weights=False)
            return self.sal(o[:, 0])

    class Sonda(nn.Module):
        def __init__(self):
            super().__init__()
            self.gs = GS()

        def forward(self, x):
            """x: (n, 2, 512, 1024). Devuelve los logits [fotl, fotr, neither]."""
            a = self.gs(x[:, 0])
            b = self.gs(x[:, 1])
            u = a[:, 0] - b[:, 0]
            s = a[:, 1] + b[:, 1]
            return torch.stack([u, -u, s], dim=1)

    return Sonda()


def sonda_lineal(d_in=1024):
    torch = _torch()
    nn = torch.nn

    class Lin(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(d_in, 1, bias=False)
            self.v = nn.Linear(d_in, 1)

        def forward(self, m):
            """m: (n, 2, 1024) medias de las dos vistas."""
            u = self.w(m[:, 0] - m[:, 1])[:, 0]
            s = self.v(m[:, 0] + m[:, 1])[:, 0]
            return torch.stack([u, -u, s], dim=1)

    return Lin()


TAREA = "initiated"
"""`initiated` (pregunta 1) o `prioridad` (pregunta 2: quien gana con las dos luces). Se fija en
`main`; las funciones de abajo la leen para decidir entre 3 clases o solo izquierda/derecha."""


def decide(L):
    """La clase de unos logits. En `prioridad` no existe `neither`: gana el de la izquierda o el
    de la derecha, y decide el signo de `u` (las dos primeras columnas)."""
    L = np.asarray(L)
    return L[:, :2].argmax(1) if TAREA == "prioridad" else L.argmax(1)


def _espejo_y(y):
    return np.where(y == 0, 1, np.where(y == 1, 0, y))


def entrena_sonda(X, idx, y, epocas, wd, lineal=False, semilla=ET.SEMILLA, lote=64, lr=1e-3):
    """X: tokens (N,2,512,1024) fp16 en CPU, o medias (N,2,1024) si `lineal`; `idx` las filas de X
    con las que se entrena e `y` sus etiquetas. Por indices y no por copia: `X[idx]` copiaria ~8 GB."""
    idx = np.asarray(idx)
    torch = _torch()
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    mod = (sonda_lineal() if lineal else sonda_atentiva()).to("cuda")
    opt = torch.optim.AdamW(mod.parameters(), lr=lr, weight_decay=wd)
    yt = torch.as_tensor(y, dtype=torch.long)
    n = len(y)
    pasos = epocas * ((n + lote - 1) // lote)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=pasos, pct_start=0.1)
    for ep in range(epocas):
        mod.train()
        per = torch.randperm(n)
        for i in range(0, n, lote):
            j = per[i:i + lote]
            jj = j.numpy()
            orden = np.argsort(idx[jj])
            xb = torch.as_tensor(X[idx[jj][orden]]).to("cuda").float()
            yj = yt[jj[orden]]
            lg = mod(xb)
            p = torch.nn.functional.cross_entropy(lg, yj.to("cuda"))
            opt.zero_grad()
            p.backward()
            torch.nn.utils.clip_grad_norm_(mod.parameters(), 2.0)
            opt.step()
            sch.step()
    mod.eval()
    return mod


def logits_sonda(mod, X, idx, espejo=False, lote=128):
    torch = _torch()
    idx = np.asarray(idx)
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), lote):
            xb = torch.as_tensor(X[idx[i:i + lote]]).to("cuda").float()
            if espejo:
                xb = xb[:, [1, 0]]
            out.append(mod(xb).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------------------------
# FUSION, impar por construccion
# ---------------------------------------------------------------------------------------------

def rasgos_fusion(partes):
    """partes: lista de (impares (n,k1), pares (n,k2)). Devuelve (impares, pares) concatenados."""
    imp = np.concatenate([p[0] for p in partes], axis=1)
    par = np.concatenate([p[1] for p in partes], axis=1)
    return imp.astype("float64"), par.astype("float64")


def parte_logits(L):
    """De unos logits [u, -u, s]: impar `u`, par `s`."""
    return L[:, :1], L[:, 2:3]


def parte_luces(V):
    """Luces (Lc, Rc, Lw, Rw): impares Lc-Rc y Lw-Rw; pares Lc*Rc, Lw+Rw y si hay alguna."""
    imp = np.stack([V[:, 0] - V[:, 1], V[:, 2] - V[:, 3]], axis=1)
    par = np.stack([V[:, 0] * V[:, 1], V[:, 2] + V[:, 3], (V.sum(1) > 0).astype("float32")],
                   axis=1)
    return imp, par


def ajusta_fusion(imp, par, y, l2=1e-2):
    """Logistica multinomial con la estructura impar: [w.imp, -w.imp, b0 + v.par]."""
    torch = _torch()
    I = torch.as_tensor(imp)
    P = torch.as_tensor(par)
    Y = torch.as_tensor(y, dtype=torch.long)
    w = torch.zeros(I.shape[1], dtype=torch.float64, requires_grad=True)
    v = torch.zeros(P.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, v, b], max_iter=500, line_search_fn="strong_wolfe")

    def cierre():
        opt.zero_grad()
        u = I @ w
        lg = torch.stack([u, -u, b + P @ v], dim=1)
        p = torch.nn.functional.cross_entropy(lg, Y) + l2 * (w.pow(2).sum() + v.pow(2).sum())
        p.backward()
        return p

    opt.step(cierre)
    return w.detach().numpy(), v.detach().numpy(), float(b.detach())


def aplica_fusion(f, imp, par):
    w, v, b = f
    u = imp @ w
    return np.stack([u, -u, b + par @ v], axis=1)


# ---------------------------------------------------------------------------------------------
# ESTADISTICA
# ---------------------------------------------------------------------------------------------

def boot_torneo(y, pa, pb, torneos, n=B_BOOT, semilla=ET.SEMILLA):
    """IC95 de acc(pb) - acc(pa) remuestreando TORNEOS con reemplazo."""
    rng = np.random.default_rng(semilla)
    ts = np.unique(torneos)
    idx = {t: np.nonzero(torneos == t)[0] for t in ts}
    ok_a = (pa == y).astype("float64")
    ok_b = (pb == y).astype("float64")
    ds = []
    for _ in range(n):
        sel = np.concatenate([idx[t] for t in rng.choice(ts, len(ts), replace=True)])
        ds.append(ok_b[sel].mean() - ok_a[sel].mean())
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def control_espejo(pred, pred_esp):
    return float((_espejo_y(pred) == pred_esp).mean())


# ---------------------------------------------------------------------------------------------
# LA MEDIDA
# ---------------------------------------------------------------------------------------------

def logits_etiquetador(mod, filas, idx, espejo=False):
    torch = _torch()
    P, _A, S, _y = ET._tensores(filas, idx, "initiated", torch)
    if espejo:
        P = P[:, [1, 0]]
    with torch.no_grad():
        return mod.inicia(P, S).numpy().astype("float64")


def elige_y_fusiona(filas, tr, X, M, V, y, verboso=True, rapido=False):
    """Particion INTERNA sobre las filas `tr` (SOLO el entrenamiento externo). Devuelve la config
    elegida de B y Blin, y los pesos de las fusiones, todo sin mirar el reservado externo."""
    tr = np.asarray(tr)
    filas_tr = [filas[i] for i in tr]
    mod_i, ev_i = ET.entrena(filas_tr, "initiated", verboso=False)
    i_tr, i_va = tr[ev_i["itr"]], tr[ev_i["ire"]]
    """`ev_i` indexa `filas_tr`; `tr[...]` lo lleva a indices de `filas` y de X."""
    La = logits_etiquetador(mod_i, filas_tr, ev_i["ire"])
    yv = y[i_va]
    rej = [(e, w) for e in REJILLA_EPOCAS for w in REJILLA_WD]
    if rapido:
        rej = rej[:1]
    mejor, Lb_va = None, None
    for e, w in rej:
        t0 = time.time()
        m = entrena_sonda(X, i_tr, y[i_tr], e, w)
        L = logits_sonda(m, X, i_va)
        acc = float((decide(L) == yv).mean())
        ll = float(_nll(L, yv))
        if verboso:
            print("   interna B  epocas %3d  wd %.0e  acierto %.3f  nll %.4f  (%.0f s)"
                  % (e, w, acc, ll, time.time() - t0), flush=True)
        if mejor is None or (acc, -ll) > (mejor[2], -mejor[3]):
            mejor, Lb_va = (e, w, acc, ll), L
    mejor_l, Ll_va = None, None
    for e, w in rej:
        m = entrena_sonda(M, i_tr, y[i_tr], e, w, lineal=True)
        L = logits_sonda(m, M, i_va)
        acc = float((decide(L) == yv).mean())
        ll = float(_nll(L, yv))
        if mejor_l is None or (acc, -ll) > (mejor_l[2], -mejor_l[3]):
            mejor_l, Ll_va = (e, w, acc, ll), L
    if verboso:
        print("   interna: B elegida epocas %d wd %.0e (%.3f); Blin epocas %d wd %.0e (%.3f); "
              "etiquetador %.3f  n=%d"
              % (mejor[0], mejor[1], mejor[2], mejor_l[0], mejor_l[1], mejor_l[2],
                 float((decide(La) == yv).mean()), len(yv)), flush=True)
    Vv = V[i_va]
    fus = {
        "A'": ajusta_fusion(*rasgos_fusion([parte_logits(La), parte_luces(Vv)]), yv),
        "C": ajusta_fusion(*rasgos_fusion([parte_logits(La), parte_logits(Lb_va)]), yv),
        "C'": ajusta_fusion(*rasgos_fusion([parte_logits(La), parte_luces(Vv),
                                            parte_logits(Lb_va)]), yv),
    }
    return mejor, mejor_l, fus


def _nll(L, y):
    L = L - L.max(1, keepdims=True)
    return float(-(L[np.arange(len(y)), y] - np.log(np.exp(L).sum(1))).mean())


def filas_prioridad():
    """Las filas de la pregunta 2: florete y sable, las DOS luces de color, `touch` izq/der. La
    etiqueta va en `y_initiated` (0 = gana la izquierda, 1 = la derecha) para usar la misma cabeza
    antisimetrica y el mismo reparto; la etiqueta humana de quien inicio queda en `y_humano`."""
    filas = []
    for f in ET.reune(cabeza=None):
        if f["weapon"] not in ("foil", "sabre") or f["touch"] not in ("left", "right"):
            continue
        p = RU.TRABAJO / ("lights_%s.csv" % f["clip"])
        if not p.is_file():
            continue
        with open(p, encoding="utf-8") as fh:
            lados = {r.get("lado") for r in csv.DictReader(fh) if r.get("canal") in ("roja", "verde")}
        if lados != {"L", "R"}:
            continue
        f["y_humano"] = f["y_initiated"]
        f["y_initiated"] = 0 if f["touch"] == "left" else 1
        filas.append(f)
    print("prioridad: %d clips con dos luces y touch izq/der (florete %d, sable %d)"
          % (len(filas), sum(f["weapon"] == "foil" for f in filas),
             sum(f["weapon"] == "sabre" for f in filas)))
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(RU.SCRATCH / "vjepa"))
    ap.add_argument("--rapido", action="store_true", help="un punto de rejilla: SOLO para probar")
    ap.add_argument("--tarea", choices=("initiated", "prioridad"), default="initiated")
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    global TAREA
    TAREA = a.tarea
    print("SONDA V-JEPA 2 CONTRA EL ETIQUETADOR (protocolo VJEPA2_2026-10-09), tarea %s" % TAREA)
    print("=" * 80)
    filas_todas = ET.reune(cabeza="initiated") if TAREA == "initiated" else filas_prioridad()
    con = [f for f in filas_todas if (d / ("vjepa_%s.npz" % f["clip"])).is_file()
           and luces_clip(f["clip"]) is not None]
    print("filas con initiated %d; con embedding y luces %d; sin embedding %d; sin lights_ %d"
          % (len(filas_todas), len(con),
             sum(not (d / ("vjepa_%s.npz" % f["clip"])).is_file() for f in filas_todas),
             sum(luces_clip(f["clip"]) is None for f in filas_todas)))
    itr_t, ire_t, diag = ET.particion_por_torneo(filas_todas)
    res_t = {filas_todas[i]["torneo"] for i in ire_t}
    """EL RESERVADO SE FIJA SOBRE TODAS LAS FILAS, como en el careo del etiquetador, y luego se
    queda con las que tienen embedding: asi un clip sin embedding no puede mover un torneo de un
    lado al otro."""
    filas = con
    tr = [i for i, f in enumerate(filas) if f["torneo"] not in res_t]
    re = [i for i, f in enumerate(filas) if f["torneo"] in res_t]
    print("reservado: %d torneos, %d clips (de %d en el reparto completo); entrenamiento %d"
          % (len(res_t), len(re), len(ire_t), len(tr)))
    clips = [f["clip"] for f in filas]
    y = np.array([f["y_initiated"] for f in filas])
    torneos = np.array([f["torneo"] for f in filas])
    armas = np.array([f["weapon"] for f in filas])
    t0 = time.time()
    X, M, via = carga_tokens(clips, d)
    V = np.stack([luces_clip(c) for c in clips])
    print("tokens cargados: %.1f GB en %.0f s" % (X.nbytes / 1e9, time.time() - t0), flush=True)

    mejor, mejor_l, fus = elige_y_fusiona(filas, tr, X, M, V, y, rapido=a.rapido)

    mod_a = entrena_etiquetador_fijo(filas, tr)
    """El etiquetador externo se entrena SOLO con `tr`, no con `ET.entrena(filas)`, porque su
    particion cogeria otros torneos: aqui el reservado ya esta fijado arriba."""
    La = logits_etiquetador(mod_a, filas, re)
    La_e = logits_etiquetador(mod_a, filas, re, espejo=True)
    mb = entrena_sonda(X, tr, y[tr], mejor[0], mejor[1])
    Lb = logits_sonda(mb, X, re)
    Lb_e = logits_sonda(mb, X, re, espejo=True)
    ml = entrena_sonda(M, tr, y[tr], mejor_l[0], mejor_l[1], lineal=True)
    Ll = logits_sonda(ml, M, re)
    Vr = V[re]
    Vr_e = Vr[:, [1, 0, 3, 2]]
    pred = {
        "A": decide(La),
        "A'": decide(aplica_fusion(fus["A'"], *rasgos_fusion([parte_logits(La), parte_luces(Vr)]))),
        "B": decide(Lb),
        "Blin": decide(Ll),
        "C": decide(aplica_fusion(fus["C"], *rasgos_fusion([parte_logits(La), parte_logits(Lb)]))),
        "C'": decide(aplica_fusion(fus["C'"], *rasgos_fusion([parte_logits(La), parte_luces(Vr),
                                                       parte_logits(Lb)]))),
    }
    pred_e = {
        "B": decide(Lb_e),
        "C'": decide(aplica_fusion(fus["C'"], *rasgos_fusion([parte_logits(La_e), parte_luces(Vr_e),
                                                       parte_logits(Lb_e)]))),
    }
    yr, tor_r, arm_r, via_r = y[re], torneos[re], armas[re], via[re]
    techo = CA.techo_humano()["initiated"]
    may = float((yr == np.bincount(y[tr]).argmax()).mean())
    print()
    print("RESERVADO: %d clips, %d torneos" % (len(yr), len(np.unique(tor_r))))
    print("%-44s %8s   %s" % ("", "acierto", "contra A' (IC95 por torneo)"))
    print("-" * 80)
    print("%-44s %8.3f" % ("clase mayoritaria del entrenamiento", may))
    if TAREA == "prioridad":
        yh = np.array([filas[i]["y_humano"] for i in re])
        m = yh <= 1
        print("%-44s %8.3f   (cobertura %.3f; etiqueta HUMANA, no un modelo)"
              % ("gana el que inicio", float((yh[m] == yr[m]).mean()) if m.any() else float("nan"),
                 m.mean()))
    for k in ("A", "A'", "B", "Blin", "C", "C'"):
        acc = float((pred[k] == yr).mean())
        if k == "A'":
            print("%-44s %8.3f   (linea base que decide)" % (k, acc))
            continue
        lo, hi = boot_torneo(yr, pred["A'"], pred[k], tor_r)
        print("%-44s %8.3f   %+.3f  [%+.3f, %+.3f]"
              % (k, acc, acc - float((pred["A'"] == yr).mean()), lo, hi))
    if TAREA == "initiated":
        print("%-44s %8.3f" % ("techo humano (n=%d)" % techo["n"], techo["acuerdo"]))
    print()
    for k in ("B", "C'"):
        print("control del espejo %-4s %.4f" % (k, control_espejo(pred[k], pred_e[k])))
    lo, hi = boot_torneo(yr, pred["A"], pred["B"], tor_r)
    print("B contra A (informativo): %+.3f [%+.3f, %+.3f]"
          % (float((pred["B"] == yr).mean() - (pred["A"] == yr).mean()), lo, hi))
    print()
    print("por arma y por via del ancla (informativo):")
    for nombre, grupo in (("arma", arm_r), ("via_ancla", via_r)):
        for g in np.unique(grupo):
            m = grupo == g
            print("   %-10s %-8s n=%4d   A' %.3f   B %.3f   C' %.3f"
                  % (nombre, g, m.sum(), (pred["A'"][m] == yr[m]).mean(),
                     (pred["B"][m] == yr[m]).mean(), (pred["C'"][m] == yr[m]).mean()))
    if TAREA == "prioridad":
        print()
        print("PREGUNTA 2 (no decide sola; ver el protocolo):")
        for k in ("B", "C"):
            lo, hi = boot_torneo(yr, pred["A"], pred[k], tor_r)
            d = float((pred[k] == yr).mean() - (pred["A"] == yr).mean())
            print("   %s - A = %+.3f  IC95 por torneo [%+.3f, %+.3f]  -> %s"
                  % (k, d, lo, hi, "bate a A" if lo > 0 else "no se distingue de A"))
        return 0
    acc_a2 = float((pred["A'"] == yr).mean())
    acc_c2 = float((pred["C'"] == yr).mean())
    lo, hi = boot_torneo(yr, pred["A'"], pred["C'"], tor_r)
    esp = control_espejo(pred["C'"], pred_e["C'"])
    pasa = (acc_c2 - acc_a2 >= 0.02) and lo > 0 and esp == 1.0
    print()
    print("CRITERIO (1-3): C' - A' = %+.3f >= +0,02, IC95 [%+.3f, %+.3f] sin cruzar cero, "
          "espejo %.4f  ->  %s" % (acc_c2 - acc_a2, lo, hi, esp, "PASA" if pasa else "NO PASA"))
    print("(el 4, el coste en GPU, se lee del extractor: `s_gpu` en cada vjepa_<clip>.npz)")
    return 0


def entrena_etiquetador_fijo(filas, tr, epocas=60, lote=32, lr=3e-3, semilla=ET.SEMILLA):
    """`ET.entrena` con el reparto ya dado: mismo modelo, misma perdida con el espejo como aumento,
    mismas epocas, y la ultima epoca, no la mejor."""
    torch = _torch()
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    mod = ET.construye()
    opt = torch.optim.AdamW(mod.parameters(), lr=lr, weight_decay=1e-4)
    P, _A, S, y = ET._tensores(filas, tr, "initiated", torch)
    n = len(tr)
    for _ in range(epocas):
        mod.train()
        per = torch.randperm(n)
        for i in range(0, n, lote):
            j = per[i:i + lote]
            opt.zero_grad()
            lg = mod.inicia(P[j], S[j])
            p = torch.nn.functional.cross_entropy(lg, y[j])
            lge = mod.inicia(P[j][:, [1, 0]], S[j])
            ye = torch.where(y[j] == 0, torch.ones_like(y[j]),
                             torch.where(y[j] == 1, torch.zeros_like(y[j]), y[j]))
            p = p + torch.nn.functional.cross_entropy(lge, ye)
            p.backward()
            torch.nn.utils.clip_grad_norm_(mod.parameters(), 2.0)
            opt.step()
    mod.eval()
    return mod


if __name__ == "__main__":
    sys.exit(main())

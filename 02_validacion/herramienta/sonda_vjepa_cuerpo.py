"""Pregunta 3 de `VJEPA2_VARIANTES_2026-10-09.md`: el BLANCO tocado (`body_location`) con V-JEPA 2.

El protocolo, el criterio y las lineas base estan alli, escritos antes de que existiera este fichero.

QUE SE COMPARA, en el reservado por torneo (`etiquetador.particion_por_torneo`, su semilla) de las
filas con `body_location` y embedding:
  A   la cabeza `cuerpo` del etiquetador de poses (simetrica), media de logits de N semillas
  B   sonda atentiva SIMETRICA sobre los tokens de V-JEPA 2 (V0), media de N semillas
  C   logistica multinomial sobre [logits de A, logits de B]          <- lo que decide, contra A

LA SIMETRIA: el blanco tocado no cambia al espejar el clip. La sonda da `s(x) + s(x')` con `x'` el
espejo, asi que intercambiar las vistas no puede cambiar nada; la cabeza `cuerpo` solo ve la suma y
el valor absoluto de la diferencia de las dos ranuras. El control pide prediccion IDENTICA.

Reutiliza de `sonda_vjepa.py` (sin tocarlo): la carga de tokens, el bucle de entrenamiento por
indices, el bootstrap por torneo y la rejilla. La sonda que ese bucle construye se sustituye en ESTE
proceso por la simetrica.

    python sonda_vjepa_cuerpo.py [--semillas 5] [--rapido]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sonda_vjepa as SV  # noqa: E402

CA, ET, RU = SV.CA, SV.ET, SV.RU
K = len(CA.BLANCOS)
CAB = "body_location"


def sonda_simetrica(d_in=1024, d=256, cabezas=4, n_tok=512):
    torch = SV._torch()
    nn = torch.nn

    class GS(nn.Module):
        """La misma arquitectura que `sonda_vjepa.sonda_atentiva`, con K salidas en vez de 2."""

        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(d_in)
            self.proj = nn.Linear(d_in, d)
            self.pos = nn.Parameter(torch.zeros(1, n_tok, d))
            self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.att = nn.MultiheadAttention(d, cabezas, batch_first=True, dropout=0.1)
            self.sal = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(),
                                     nn.Dropout(0.1), nn.Linear(128, K))

        def forward(self, x):
            h = self.proj(self.ln(x)) + self.pos
            o, _ = self.att(self.q.expand(len(h), -1, -1), h, h, need_weights=False)
            return self.sal(o[:, 0])

    class Sonda(nn.Module):
        def __init__(self):
            super().__init__()
            self.gs = GS()

        def forward(self, x):
            return self.gs(x[:, 0]) + self.gs(x[:, 1])

    return Sonda()


def entrena_cuerpo_fijo(filas, tr, epocas=60, lote=32, lr=3e-3, semilla=ET.SEMILLA):
    """`ET.entrena(..., "body_location")` con el reparto ya dado: mismo modelo, misma perdida, mismas
    epocas, y la ultima epoca."""
    torch = SV._torch()
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    mod = ET.construye()
    opt = torch.optim.AdamW(mod.parameters(), lr=lr, weight_decay=1e-4)
    P, _A, S, y = ET._tensores(filas, tr, CAB, torch)
    n = len(tr)
    for _ in range(epocas):
        mod.train()
        per = torch.randperm(n)
        for i in range(0, n, lote):
            j = per[i:i + lote]
            opt.zero_grad()
            p = torch.nn.functional.cross_entropy(mod.cuerpo(P[j], S[j]), y[j])
            p.backward()
            torch.nn.utils.clip_grad_norm_(mod.parameters(), 2.0)
            opt.step()
    mod.eval()
    return mod


def logits_cuerpo(mod, filas, idx, espejo=False):
    torch = SV._torch()
    P, _A, S, _y = ET._tensores(filas, idx, CAB, torch)
    if espejo:
        P = P[:, [1, 0]]
    with torch.no_grad():
        return mod.cuerpo(P, S).numpy().astype("float64")


def ajusta_logistica(F, y, l2=1e-2):
    torch = SV._torch()
    X = torch.as_tensor(F, dtype=torch.float64)
    Y = torch.as_tensor(y, dtype=torch.long)
    W = torch.zeros(X.shape[1], K, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(K, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=500, line_search_fn="strong_wolfe")

    def cierre():
        opt.zero_grad()
        p = torch.nn.functional.cross_entropy(X @ W + b, Y) + l2 * W.pow(2).sum()
        p.backward()
        return p

    opt.step(cierre)
    return W.detach().numpy(), b.detach().numpy()


def aplica(f, F):
    return F @ f[0] + f[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(RU.SCRATCH / "vjepa"))
    ap.add_argument("--semillas", type=int, default=5)
    ap.add_argument("--rapido", action="store_true",
                    help="un punto de rejilla y para antes del reservado: SOLO para probar")
    ap.add_argument("--hilos", type=int, default=6)
    a = ap.parse_args()
    torch = SV._torch()
    torch.set_num_threads(a.hilos)
    SV.sonda_atentiva = sonda_simetrica
    """`SV.entrena_sonda` construye `sonda_atentiva()` por su nombre global: en ESTE proceso pasa a
    ser la simetrica. El fichero `sonda_vjepa.py` no se toca."""
    d = pathlib.Path(a.dir)
    S = range(a.semillas)
    print("SONDA V-JEPA 2: EL BLANCO TOCADO (pregunta 3 de VJEPA2_VARIANTES_2026-10-09), %d semilla(s)"
          % a.semillas)
    print("=" * 80)
    filas_todas = ET.reune(cabeza=CAB)
    con = [f for f in filas_todas if (d / ("vjepa_%s.npz" % f["clip"])).is_file()]
    print("filas con %s %d; con embedding %d" % (CAB, len(filas_todas), len(con)))
    itr_t, ire_t, diag = ET.particion_por_torneo(filas_todas)
    res_t = {filas_todas[i]["torneo"] for i in ire_t}
    filas = con
    tr = np.array([i for i, f in enumerate(filas) if f["torneo"] not in res_t])
    re = np.array([i for i, f in enumerate(filas) if f["torneo"] in res_t])
    print("reservado: %d torneos, %d clips; entrenamiento %d" % (len(res_t), len(re), len(tr)))
    clips = [f["clip"] for f in filas]
    y = np.array([f["y_" + CAB] for f in filas])
    torneos = np.array([f["torneo"] for f in filas])
    armas = np.array([f["weapon"] for f in filas])
    t0 = time.time()
    X, _M, _via = SV.carga_tokens(clips, d)
    print("tokens cargados: %.1f GB en %.0f s" % (X.nbytes / 1e9, time.time() - t0), flush=True)

    """PARTICION INTERNA del entrenamiento, por torneo: elige la rejilla de B y ajusta C."""
    filas_tr = [filas[i] for i in tr]
    a_tr, a_va, _ = ET.particion_por_torneo(filas_tr)
    i_tr, i_va = tr[a_tr], tr[a_va]
    yv = y[i_va]
    La_v = np.mean([logits_cuerpo(entrena_cuerpo_fijo(filas, i_tr, semilla=ET.SEMILLA + k),
                                  filas, i_va) for k in S], axis=0)
    rej = [(e, w) for e in SV.REJILLA_EPOCAS for w in SV.REJILLA_WD]
    if a.rapido:
        rej = rej[:1]
    mejor = None
    for e, w in rej:
        t1 = time.time()
        L = SV.logits_sonda(SV.entrena_sonda(X, i_tr, y[i_tr], e, w), X, i_va)
        acc, ll = float((L.argmax(1) == yv).mean()), SV._nll(L, yv)
        print("   interna B  epocas %3d  wd %.0e  acierto %.3f  nll %.4f  (%.0f s)"
              % (e, w, acc, ll, time.time() - t1), flush=True)
        if mejor is None or (acc, -ll) > (mejor[2], -mejor[3]):
            mejor = (e, w, acc, ll)
    Lb_v = np.mean([SV.logits_sonda(SV.entrena_sonda(X, i_tr, y[i_tr], mejor[0], mejor[1],
                                                     semilla=ET.SEMILLA + k), X, i_va)
                    for k in S], axis=0)
    fus = ajusta_logistica(np.concatenate([La_v, Lb_v], 1), yv)
    print("   interna (n=%d): B elegida epocas %d wd %.0e; con %d semillas A %.3f  B %.3f  C %.3f"
          % (len(yv), mejor[0], mejor[1], a.semillas, (La_v.argmax(1) == yv).mean(),
             (Lb_v.argmax(1) == yv).mean(),
             (aplica(fus, np.concatenate([La_v, Lb_v], 1)).argmax(1) == yv).mean()), flush=True)
    if a.rapido:
        print("--rapido: prueba de humo, se para ANTES del reservado (no se mira)")
        return 0

    """FUERA: todo el entrenamiento, evaluado UNA vez en el reservado."""
    mods_a = [entrena_cuerpo_fijo(filas, tr, semilla=ET.SEMILLA + k) for k in S]
    La = np.mean([logits_cuerpo(m, filas, re) for m in mods_a], axis=0)
    La_e = np.mean([logits_cuerpo(m, filas, re, espejo=True) for m in mods_a], axis=0)
    mbs = [SV.entrena_sonda(X, tr, y[tr], mejor[0], mejor[1], semilla=ET.SEMILLA + k) for k in S]
    Lb = np.mean([SV.logits_sonda(m, X, re) for m in mbs], axis=0)
    Lb_e = np.mean([SV.logits_sonda(m, X, re, espejo=True) for m in mbs], axis=0)
    pred = {"A": La.argmax(1), "B": Lb.argmax(1),
            "C": aplica(fus, np.concatenate([La, Lb], 1)).argmax(1)}
    pred_e = {"B": Lb_e.argmax(1), "C": aplica(fus, np.concatenate([La_e, Lb_e], 1)).argmax(1)}
    yr, tor_r, arm_r = y[re], torneos[re], armas[re]
    may_c = np.bincount(y[tr], minlength=K).argmax()
    may = np.full(len(yr), may_c)
    techo = CA.techo_humano().get(CAB, {})
    print()
    print("RESERVADO: %d clips, %d torneos" % (len(yr), len(np.unique(tor_r))))
    print("%-44s %8s   %s" % ("", "acierto", "contra A (IC95 por torneo)"))
    print("-" * 80)
    print("%-44s %8.3f   %s" % ("clase mayoritaria del entrenamiento (%s)" % CA.BLANCOS[may_c],
                               (may == yr).mean(), "%+.3f [%+.3f, %+.3f]" % (
                                   (may == yr).mean() - (pred["A"] == yr).mean(),
                                   *SV.boot_torneo(yr, pred["A"], may, tor_r))))
    print("%-44s %8.3f   (linea base que decide)" % ("A (etiquetador, cabeza cuerpo)",
                                                     (pred["A"] == yr).mean()))
    for k in ("B", "C"):
        lo, hi = SV.boot_torneo(yr, pred["A"], pred[k], tor_r)
        print("%-44s %8.3f   %+.3f  [%+.3f, %+.3f]" % (k, (pred[k] == yr).mean(),
                                                     (pred[k] == yr).mean() - (pred["A"] == yr).mean(),
                                                     lo, hi))
    if techo:
        print("%-44s %8.3f" % ("techo humano (n=%d)" % techo.get("n", 0), techo.get("acuerdo", np.nan)))
    lo, hi = SV.boot_torneo(yr, may, pred["B"], tor_r)
    print("B contra la mayoritaria (informativo): %+.3f [%+.3f, %+.3f]"
          % ((pred["B"] == yr).mean() - (may == yr).mean(), lo, hi))
    print()
    esp = {k: float((pred[k] == pred_e[k]).mean()) for k in ("B", "C")}
    print("control del espejo (prediccion identica)  B %.4f  C %.4f" % (esp["B"], esp["C"]))
    print()
    print("por arma (informativo):")
    for g in np.unique(arm_r):
        m = arm_r == g
        print("   %-6s n=%4d   mayoritaria %.3f   A %.3f   B %.3f   C %.3f"
              % (g, m.sum(), (may[m] == yr[m]).mean(), (pred["A"][m] == yr[m]).mean(),
                 (pred["B"][m] == yr[m]).mean(), (pred["C"][m] == yr[m]).mean()))
    print()
    print("matriz de confusion de C (filas = humano, columnas = C), clases %s" % CA.BLANCOS)
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(yr, pred["C"]):
        cm[t, p] += 1
    for k in range(K):
        print("   %-10s %s" % (CA.BLANCOS[k], " ".join("%4d" % v for v in cm[k])))
    acc_a, acc_c = float((pred["A"] == yr).mean()), float((pred["C"] == yr).mean())
    lo, hi = SV.boot_torneo(yr, pred["A"], pred["C"], tor_r)
    pasa = (acc_c - acc_a >= 0.02) and lo > 0 and esp["C"] == 1.0
    print()
    print("CRITERIO: C - A = %+.3f >= +0,02, IC95 [%+.3f, %+.3f] sin cruzar cero, espejo %.4f  ->  %s"
          % (acc_c - acc_a, lo, hi, esp["C"], "PASA" if pasa else "NO PASA"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

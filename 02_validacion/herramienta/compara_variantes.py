"""Compara POR PARES dos variantes de V-JEPA (`VJEPA2_VARIANTES_2026-10-09.md`, cambio 1).

Lee dos `--guarda-pred` de `sonda_vjepa.py`, la variante y su referencia, en los MISMOS clips del
reservado, y da la diferencia de acierto de C' (lo que decide) y de B, con IC95 por bootstrap agrupado
por torneo, y el control del espejo. El criterio es el del protocolo: `C'(Vk) - C'(ref) >= +0,02`,
IC95 sin cruzar cero y espejo 1,0000.

    python compara_variantes.py REF.npz VAR.npz [--tarea initiated|prioridad]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

B_BOOT = 2000
SEMILLA = 20260922


def boot(ok_a, ok_b, tor, n=B_BOOT, semilla=SEMILLA):
    rng = np.random.default_rng(semilla)
    ts = np.unique(tor)
    idx = {t: np.nonzero(tor == t)[0] for t in ts}
    d = []
    for _ in range(n):
        sel = np.concatenate([idx[t] for t in rng.choice(ts, len(ts), replace=True)])
        d.append(ok_b[sel].mean() - ok_a[sel].mean())
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def espejo_y(y):
    return np.where(y == 0, 1, np.where(y == 1, 0, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("var")
    a = ap.parse_args()
    R, V = np.load(a.ref, allow_pickle=True), np.load(a.var, allow_pickle=True)
    cr, cv = list(R["clips"]), list(V["clips"])
    comunes = sorted(set(cr) & set(cv))
    ir = np.array([cr.index(c) for c in comunes])
    iv = np.array([cv.index(c) for c in comunes])
    print("%s (%s) contra %s (%s): %d clips comunes (ref %d, var %d)"
          % (str(V["variante"]), a.var, str(R["variante"]), a.ref, len(comunes), len(cr), len(cv)))
    y = R["y"][ir]
    assert (y == V["y"][iv]).all(), "las etiquetas no coinciden: no es el mismo reparto"
    tor = R["torneo"][ir]
    arm = R["arma"][ir]
    pasa = None
    for k in ("A", "Ap", "B", "Blin", "C", "Cp"):
        pr, pv = R["pred_" + k][ir], V["pred_" + k][iv]
        okr, okv = (pr == y).astype(float), (pv == y).astype(float)
        lo, hi = boot(okr, okv, tor)
        d = okv.mean() - okr.mean()
        print("   %-5s ref %.3f  var %.3f  dif %+.3f  IC95 por torneo [%+.3f, %+.3f]"
              % (k.replace("p", "'"), okr.mean(), okv.mean(), d, lo, hi))
        if k == "Cp":
            esp = float((espejo_y(V["pred_Cp"][iv]) == V["esp_Cp"][iv]).mean())
            pasa = d >= 0.02 and lo > 0 and esp == 1.0
            print("   espejo C' de la variante %.4f" % esp)
    print("   por arma, C':")
    for g in np.unique(arm):
        m = arm == g
        print("      %-6s n=%4d  ref %.3f  var %.3f"
              % (g, m.sum(), (R["pred_Cp"][ir][m] == y[m]).mean(), (V["pred_Cp"][iv][m] == y[m]).mean()))
    print("CRITERIO: C'(var) - C'(ref) >= +0,02, IC95 sin cruzar cero, espejo 1,0000  ->  %s"
          % ("MEJORA" if pasa else "no se distingue"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

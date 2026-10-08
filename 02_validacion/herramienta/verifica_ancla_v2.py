"""Careo del emparejamiento viejo contra el nuevo, sobre EXACTAMENTE los mismos datos.

## COMO SE GARANTIZA QUE ES EL MISMO DATO

No se reimplementa la deteccion de marcas ni la geometria: se **sustituye** `identifica_marcas` de
`pista` por una version que llama a las dos y guarda las dos respuestas. Asi los dos reciben el
mismo `sp`, las mismas lineas y los mismos bordes, y cualquier diferencia es del emparejamiento y
de nada mas. Comparar dos tuberias enteras mediria tambien sus diferencias de deteccion.

## EL CRITERIO, DECLARADO ANTES DE MEDIR

Contra `strip_location`, que da verdad humana en 3.964 clips de florete:

    se adopta si   el acierto por zonas bate el **0,430** de contestar siempre `middle`
                   y la pendiente del ajuste sube del **0,07** de hoy a **> 0,70**
    el techo es    **0,619**, que es lo que coinciden dos personas mirando el mismo clip

Y dos cifras mas que hay que dar aunque no sean el criterio: cuantos anclajes quedan
**controlados** (>= 3 marcas no semilla) y cuantos clips se pierden por ser mas exigente. Un
emparejamiento que acierte mas contestando en la mitad de los clips no es mejor sin decir las dos
cosas.

    python verifica_ancla_v2.py [--n 40]
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "01_extraccion" / "codigo"))
import ancla_v2 as AV  # noqa: E402
import catalogo as CA  # noqa: E402
import pista as PI  # noqa: E402
import rutas as RU  # noqa: E402

CAPTURA = {}
_ORIG = PI.identifica_marcas


def _espia(lineas, bordes, pp):
    """QUE HACE: llama al de hoy, reconstruye el mismo `sp` y llama tambien al nuevo.

    El `sp` se recalcula aqui con la misma geometria en vez de tocar `pista.py`, porque la
    extraccion esta corriendo y cambiar ese fichero a mitad dejaria el corpus con dos criterios
    mezclados."""
    viejo = _ORIG(lineas, bordes, pp)
    CAPTURA["viejo"] = viejo
    CAPTURA["nuevo"] = None
    try:
        sp = _reconstruye_sp(lineas, bordes, pp)
        if sp is not None:
            CAPTURA["nuevo"] = AV.identifica(sp, span_tira_m=CAPTURA.get("span_tira_m"))
    except Exception as ex:
        CAPTURA["nuevo"] = {"ok": False, "motivo": "%s" % type(ex).__name__}
    return viejo


def _reconstruye_sp(lineas, bordes, pp):
    """La misma construccion que `pista.identifica_marcas`: corte de cada marca con los dos bordes,
    coordenada a lo largo del borde A, y la proyectiva `s = 1/(t - t_V1)`."""
    LAc = PI._a_centrada(bordes["recta_A"], pp)
    LBc = PI._a_centrada(bordes["recta_B"], pp)
    V1 = np.cross(LAc, LBc)
    dirA = np.array([-LAc[1], LAc[0]], "float64")
    dirA = dirA / np.linalg.norm(dirA)
    pA0 = -LAc[2] * np.array([LAc[0], LAc[1]], "float64")
    ts = []
    for li in lineas:
        Lc = PI._a_centrada(li["L"], pp)
        q = np.cross(Lc, LAc)
        qb = np.cross(Lc, LBc)
        if abs(q[2]) < 1e-12 or abs(qb[2]) < 1e-12:
            continue
        qa = q[:2] / q[2]
        ts.append(float((qa - pA0) @ dirA))
    if len(ts) < AV.MIN_IDENT:
        return None
    ts = np.asarray(ts)
    if abs(V1[2]) > 1e-12:
        tv1 = float((V1[:2] / V1[2] - pA0) @ dirA)
        with np.errstate(all="ignore"):
            sp = 1.0 / (ts - tv1)
    else:
        sp = ts.copy()
    return sp if np.isfinite(sp).all() else None


def zona(x, m=2.0):
    return 0 if x < -5 else 4 if x > 5 else 1 if x < -m else 3 if x > m else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    PI.identifica_marcas = _espia
    df = CA.manifiesto().set_index("clip_id")
    clips = [os.path.basename(p)[6:-4]
             for p in sorted(glob.glob(str(RU.TRABAJO / "poses_*.npz")))]
    clips = [c for c in clips if c in df.index
             and df.loc[c, "strip_location"] in CA.ZONAS][:a.n]
    print("CAREO DEL EMPAREJAMIENTO: el de hoy contra el nuevo")
    print("=" * 96)
    print("clips con etiqueta humana de zona: %d" % len(clips))
    print()
    print("%-30s %-22s %-22s" % ("clip", "VIEJO", "NUEVO"))
    print("-" * 96)
    res = []
    for c in clips:
        CAPTURA.clear()
        try:
            r = PI.analiza(c)
        except Exception:
            continue
        v, nv = CAPTURA.get("viejo"), CAPTURA.get("nuevo")
        if not v and not nv:
            continue
        sv = ("ident %d err %.3f" % (v.get("n", 0), v.get("err_p50_m", float("nan")))) \
            if (v and v.get("ok")) else (str((v or {}).get("motivo", "-"))[:20])
        sn = ("ident %d comp %.2f pred %s" % (
            nv.get("n_ident", 0), nv.get("completitud", 0),
            ("%.3f" % nv["err_pred_p50_m"]) if nv.get("err_pred_p50_m") is not None else "-")) \
            if (nv and nv.get("ok")) else (str((nv or {}).get("motivo", "-"))[:22])
        print("%-30s %-22s %-22s" % (c[:30], sv[:22], sn[:22]))
        res.append((c, CA.ZONAS.index(df.loc[c, "strip_location"]), v, nv, r))
    print()
    ok_v = sum(1 for _c, _y, v, _n, _r in res if v and v.get("ok"))
    ok_n = sum(1 for _c, _y, _v, n, _r in res if n and n.get("ok"))
    ctrl = sum(1 for _c, _y, _v, n, _r in res if n and n.get("controlada"))
    print("contestan:  viejo %d   nuevo %d   de %d" % (ok_v, ok_n, len(res)))
    print("del nuevo, CONTROLADOS (>=3 marcas no semilla y error de prediccion bajo): %d" % ctrl)
    comp = [n["completitud"] for _c, _y, _v, n, _r in res if n and n.get("ok")]
    if comp:
        print("completitud del nuevo: p10 %.2f  p50 %.2f  p90 %.2f"
              % (np.percentile(comp, 10), np.median(comp), np.percentile(comp, 90)))
    dif = [(c, v.get("patron"), n.get("patron"))
           for c, _y, v, n, _r in res
           if v and n and v.get("ok") and n.get("ok") and v.get("patron") != n.get("patron")]
    print()
    print("clips donde los dos identifican patrones DISTINTOS: %d" % len(dif))
    for c, pv, pn in dif[:8]:
        print("   %-30s viejo %-24s nuevo %s" % (c[:30], str(pv)[:24], pn))
    print()
    print("Que difieran no dice cual acierta. Lo dice `strip_location`, y para eso hace falta")
    print("rehacer la posicion con cada uno, que es el paso siguiente y cuesta una pasada de")
    print("la etapa 7. Este careo solo establece que el cambio MUEVE algo y cuanto.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

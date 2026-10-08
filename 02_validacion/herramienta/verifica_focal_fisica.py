"""La geometria de la pista, clip a clip, con el criterio de CONSTANCIA. Escribe un CSV reusable.

## QUE PREGUNTA CONTESTA

Una sola: **en cuantos clips la geometria de la pista es utilizable**, con un criterio que no se
aprueba a si mismo. El anterior si lo hacia -- `control_estatura` filtraba `0,3 < h < 5,0` antes de
publicar su mediana, y con eso aprobaba clips que reproyectan tiradores entre -11 y +28 m.

## EL CRITERIO, DECLARADO

Un tirador mide lo mismo este donde este. Con la homografia correcta su estatura reproyectada es
constante mientras camina de un extremo al otro de la pista. Asi que, barriendo la focal:

    frac_positivas >= 0,95    si el 5 % de los tiradores mide menos de cero, la vertical esta del
                              lado equivocado y no hay nada que salvar
    dispersion <= 0,35        (p90 - p10) / p50 de la estatura reproyectada
    ancho deducido 1,40-2,10  el rango FIE con margen; es el UNICO control independiente que queda
                              una vez la estatura se usa para ajustar

Los tres a la vez. `SANA` significa los tres.

## POR QUE EL CSV

Cada `analiza` cuesta del orden de un minuto con la extraccion corriendo. El CSV deja la geometria
medida para poder cruzarla despues con `strip_location` sin repetir el calculo, que es el paso que
de verdad decide si esto se adopta.

    python verifica_focal_fisica.py --n 120
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "01_extraccion" / "codigo"))
import focal_fisica as FF  # noqa: E402
import pista as PI  # noqa: E402
import rutas as RU  # noqa: E402

SALIDA = RU.VAL / "pista" / "geometria_constancia.csv"
CAMPOS = ["clip", "ok", "motivo", "f_rel_antes", "f_rel", "dispersion", "frac_positivas",
          "ancho_deducido_m", "hc_m", "ancho_en_rango_fie", "estable", "sana",
          "h_p10", "h_p50", "h_p90", "n_apoyos", "x_p50_m"]


def una(clip):
    fila = {"clip": clip, "ok": 0, "motivo": ""}
    r = PI.analiza(clip)
    if not r or not r.get("ok"):
        fila["motivo"] = str((r or {}).get("motivo", "sin homografia"))[:70]
        return fila
    hn, i = FF.focal_por_constancia(r)
    if hn is None:
        fila["motivo"] = str(i.get("motivo", "?"))[:70]
        return fila
    v = []
    for pie, cab in FF._pares(r, nmax=200):
        try:
            h = PI.altura_m(hn, pie, cab)
        except Exception:
            h = None
        if h is not None and np.isfinite(h):
            v.append(float(h))
    v = np.asarray(v) if v else np.array([np.nan])
    sana = bool(i["estable"] and i["ancho_en_rango_fie"])
    """LA X DEL CLIP CON LA GEOMETRIA NUEVA. Se guarda aqui aunque no sea el objetivo de este
    fichero porque es lo que despues se cruza con `strip_location`, y repetir `analiza` para
    sacarla costaria otra pasada entera."""
    x = np.nan
    try:
        import posicion as PO
        r2 = dict(r)
        r2["_hg"] = hn
        pb = PO.posicion_entre_bordes(r2, PO.carga_poses(clip))
        if pb.get("ok") and pb.get("metros_a_lo_largo"):
            X = pb["X"]
            x = float(np.nanmedian(0.5 * (X[:, 0] + X[:, 1])))
    except Exception:
        pass
    fila.update({"ok": 1, "f_rel_antes": round(i["f_rel_antes"], 3),
                 "f_rel": round(i["f_rel"], 3), "dispersion": round(i["dispersion"], 3),
                 "frac_positivas": round(i["frac_positivas"], 3),
                 "ancho_deducido_m": round(i["ancho_deducido_m"], 3),
                 "hc_m": round(i["hc_m"], 2),
                 "ancho_en_rango_fie": int(i["ancho_en_rango_fie"]),
                 "estable": int(i["estable"]), "sana": int(sana),
                 "h_p10": round(float(np.nanpercentile(v, 10)), 3),
                 "h_p50": round(float(np.nanmedian(v)), 3),
                 "h_p90": round(float(np.nanpercentile(v, 90)), 3),
                 "n_apoyos": i["n_apoyos"], "x_p50_m": (round(x, 3) if np.isfinite(x) else "")})
    return fila


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--n", type=int, default=120)
    a = ap.parse_args()
    clips = a.clips or [os.path.basename(p)[6:-4] for p in
                        sorted(glob.glob(str(RU.TRABAJO / "poses_*.npz")))][:a.n]
    hechos = {}
    if SALIDA.is_file():
        with open(SALIDA, encoding="utf-8", newline="") as fh:
            for f in csv.DictReader(fh):
                hechos[f["clip"]] = f
    """SE REANUDA. El disco es la autoridad, no un contador: si el proceso muere a mitad, la
    siguiente pasada continua donde estaba en vez de repetir una hora de calculo."""
    print("GEOMETRIA DE LA PISTA POR CONSTANCIA DE LA ESTATURA")
    print("=" * 96)
    print("%-32s %7s %7s %6s %8s %8s  %s"
          % ("clip", "f_ant", "f_new", "disp", "anchoM", "h_p50", ""))
    print("-" * 96)
    filas = []
    for c in clips:
        if c in hechos:
            filas.append(hechos[c])
            continue
        try:
            f = una(c)
        except Exception as ex:
            f = {"clip": c, "ok": 0, "motivo": "EXC %s" % type(ex).__name__}
        filas.append(f)
        SALIDA.parent.mkdir(parents=True, exist_ok=True)
        with open(SALIDA, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CAMPOS, extrasaction="ignore")
            w.writeheader()
            for g in filas:
                w.writerow(g)
        if not int(f.get("ok") or 0):
            print("%-32s %s" % (c[:32], str(f.get("motivo"))[:56]))
        else:
            print("%-32s %7s %7s %6s %8s %8s  %s"
                  % (c[:32], f["f_rel_antes"], f["f_rel"], f["dispersion"],
                     f["ancho_deducido_m"], f["h_p50"],
                     "SANA" if int(f["sana"]) else ""))
    print()
    n = len(filas)
    con = [f for f in filas if int(f.get("ok") or 0)]
    sana = [f for f in con if int(f.get("sana") or 0)]
    est = [f for f in con if int(f.get("estable") or 0)]
    fie = [f for f in con if int(f.get("ancho_en_rango_fie") or 0)]
    print("clips mirados                          %d" % n)
    print("con homografia y focal positiva        %d  (%.0f %%)" % (len(con), 100.0 * len(con) / n))
    print("   ... y estatura CONSTANTE (<= %.2f)   %d  (%.0f %% del total)"
          % (FF.DISP_MAX, len(est), 100.0 * len(est) / n))
    print("   ... y ancho FIE 1,40-2,10 m         %d  (%.0f %% del total)"
          % (len(fie), 100.0 * len(fie) / n))
    print("GEOMETRIA SANA (las tres a la vez)     %d  (%.0f %% del total)"
          % (len(sana), 100.0 * len(sana) / n))
    if con:
        d = np.array([float(f["dispersion"]) for f in con])
        print()
        print("dispersion alcanzable: p10 %.2f  p50 %.2f  p90 %.2f"
              % (np.percentile(d, 10), np.median(d), np.percentile(d, 90)))
        print()
        print("UNA DISPERSION ALTA CON LA MEJOR FOCAL SIGNIFICA QUE NO ES LA FOCAL. La focal se")
        print("barre entera; si aun asi la estatura varia con la posicion, lo que esta mal es la")
        print("homografia del SUELO, o sea los dos bordes detectados, un paso mas arriba.")
    print()
    print("CSV en %s" % SALIDA)
    return 0


if __name__ == "__main__":
    sys.exit(main())

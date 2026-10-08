"""MIDE las variables de prioridad y sus distribuciones, ANTES de escribir ninguna decision.

POR QUE ESTE SCRIPT VA ANTES DEL DETECTOR: los umbrales del detector estan declarados en
`04_documentacion/prioridad_fie.md` como PERCENTILES de la distribucion real («p90 de dext/dt»), no
como numeros elegidos a mano. Este script es el que los mide. Hacerlo al reves —elegir un umbral y
luego mirar si va bien— es lo que este proyecto ya se prohibio por escrito cuando no se bajo el
umbral de fundidos de 1,60 a 1,55 para que el control positivo pasara.

CRITERIO DECLARADO ANTES DE EJECUTAR, y son tres cosas distintas:

  1. Que `ext_hr` (hombro -> raiz de la mano) tenga RECORRIDO: su p95 partido por su p05, por
     tirador, tiene que estar por encima de **1,5**. Si el brazo extendido no mide vez y media el
     brazo encogido, esta variable no distingue ataque de preparacion y t.83.2.d no es medible.
  2. Que `ext_cr` (codo -> raiz) NO tenga recorrido: es un hueso. Su coeficiente de variacion tiene
     que ser **menor** que el de `ext_hr`. Si sale al contrario, el mapa de indices esta mal.
  3. Que el instante `t_ext` exista: en cada clip x lado tiene que haber al menos **un** tramo de
     subida sostenida de `ext_hr`. Si no lo hay, no hay ataque que fechar y el detector no tiene
     nada que decidir.

LA UNIDAD DE ANALISIS ES EL CLIP x LADO, NO EL CUADRO. Esta escrito en
`04_documentacion/informes/QUE_VAMOS_A_DETECTAR.md` con su escarmiento: agrupar cuadros dio el signo
CONTRARIO a la diferencia pareada en el control de punta en linea, porque los clips largos mandan en
el monton. Aqui se dan las dos cosas y se dice cual es cual.
"""
from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import prioridad as PR

PRUEBAS = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\prioridad")


FUERA = {"MellowHideousAngora"}
"""Clips que NO son de florete y hay que sacar de cualquier medida de prioridad.

`MellowHideousAngora` es **SABLE**: el cartel del fondo dice «SABRE FEMININ» y el marcador pone
MARTON (HUN) contra VECCHI (ITA). Se cuela porque ese clip viene del muestreo viejo, el que este
proyecto ya declaro mal en la seccion 14.6 del documento maestro; no esta en `corpus_etiquetado.txt`.

Y no es un detalle de limpieza: **la prioridad del sable es t.100 y siguientes, no t.82**. Medirlo con
las convenciones del florete daria un numero sin significado. La imagen esta en
`02_validacion/prioridad/fig_fallo_sable_MellowHideousAngora_c231.png`."""


def clips_con_poses():
    out = []
    for f in sorted(PRUEBAS.glob("poses_*.npz")):
        cid = f.stem[6:]
        if cid in FUERA:
            continue
        try:
            with np.load(f, allow_pickle=True) as z:
                if "W" in z.files and "origen" in z.files:
                    out.append(cid)
        except Exception:
            pass
    return out


def pct(x, qs=(5, 25, 50, 75, 95, 99)):
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    if not len(x):
        return {("p%d" % q): float("nan") for q in qs} | {"n": 0}
    d = {("p%d" % q): float(np.percentile(x, q)) for q in qs}
    d["n"] = int(len(x))
    return d


def tramos_subida(d, val, disp, sigue, minimo=2):
    """QUE HACE: devuelve los tramos donde la derivada arranca por encima de `disp` y sigue por
    encima de `sigue`, con al menos `minimo` cuadros por encima del disparo.

    ES EL PATRON DE HISTERESIS DE `transiciones.py`, con sus dos umbrales `UMBRAL_DISPARA` y
    `UMBRAL_SIGUE`. Se copia a proposito y no se reinventa: ahi esta validado, y el motivo por el que
    hacen falta dos umbrales es el mismo aqui que alli. Con un umbral solo, una senal que roza el
    limite entra y sale del tramo cuadro a cuadro y produce un tramo picado en vez de uno.
    """
    T = len(d)
    dd = np.where(val, d, np.nan)
    tramos = []
    k = 0
    while k < T:
        if not (np.isfinite(dd[k]) and dd[k] >= disp):
            k += 1
            continue
        cuenta = 0
        j = k
        while j < T and np.isfinite(dd[j]) and dd[j] >= disp:
            cuenta += 1
            j += 1
        if cuenta < minimo:
            k = j + 1
            continue
        a = k
        while a > 0 and np.isfinite(dd[a - 1]) and dd[a - 1] >= sigue:
            a -= 1
        b = j - 1
        while b < T - 1 and np.isfinite(dd[b + 1]) and dd[b + 1] >= sigue:
            b += 1
        tramos.append((a, b, float(np.nanmax(dd[a:b + 1]))))
        k = b + 1
    return tramos


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    clips = clips_con_poses()
    filas = []
    pool = {k: [] for k in ("ext_hr", "ext_cr", "codo", "pies", "vx", "dext_hr", "dpies",
                            "sep", "manos", "apunta")}
    print("clips con poses de 308 puntos: %d" % len(clips))
    print()
    print("%-38s %4s %5s %5s  %s" % ("clip", "T", "util", "val%", "mano armada / recorrido ext_hr"))
    for cid in clips:
        with np.load(PRUEBAS / ("poses_%s.npz" % cid), allow_pickle=True) as z:
            d = {k: z[k] for k in z.files}
        s = PR.senales(d)
        T = s["T"]
        mask = d["mask"]
        val = s["val"]
        fv = float(val[mask].mean()) if mask.any() else float("nan")
        rec = []
        for lado in (0, 1):
            v = val[:, lado]
            e = s["ext_hr"][v, lado]
            e = e[np.isfinite(e)]
            r = float(np.percentile(e, 95) / max(np.percentile(e, 5), 1e-9)) if len(e) > 10 else float("nan")
            rec.append(r)
            cv_hr = float(np.nanstd(s["ext_hr"][v, lado]) / max(abs(np.nanmean(s["ext_hr"][v, lado])), 1e-9))
            cv_cr = float(np.nanstd(s["ext_cr"][v, lado]) / max(abs(np.nanmean(s["ext_cr"][v, lado])), 1e-9))
            t = s["torso"][v, lado]
            hrpx = s["ext_hr"][v, lado] * t
            crpx = s["ext_cr"][v, lado] * t
            def _cv(a):
                return float(np.nanstd(a) / max(abs(np.nanmean(a)), 1e-9))
            filas.append({"clip": cid, "lado": lado, "mano": s["mano"][lado],
                          "T": T, "util": int(mask.sum()), "val": int(v.sum()),
                          "recorrido_ext_hr": r, "cv_ext_hr": cv_hr, "cv_ext_cr": cv_cr,
                          "cv_hr_px": _cv(hrpx), "cv_cr_px": _cv(crpx), "cv_torso_px": _cv(t),
                          "ext_hr_p50": float(np.nanmedian(s["ext_hr"][v, lado])),
                          "ext_cr_p50": float(np.nanmedian(s["ext_cr"][v, lado])),
                          "codo_p50": float(np.nanmedian(s["codo"][v, lado])),
                          "codo_p95": float(np.nanpercentile(s["codo"][v, lado], 95)) if v.sum() > 10 else float("nan"),
                          "pies_p50": float(np.nanmedian(s["pies"][v, lado])),
                          "pies_p95": float(np.nanpercentile(s["pies"][v, lado], 95)) if v.sum() > 10 else float("nan"),
                          "sep_p50": float(np.nanmedian(s["sep"][mask])),
                          "manos_p50": float(np.nanmedian(s["manos"][mask])),
                          "corrobora_mano": s["detalle_mano"][lado].get("corrobora_b2", None)})
            for nom, arr in (("ext_hr", s["ext_hr"]), ("ext_cr", s["ext_cr"]), ("codo", s["codo"]),
                             ("pies", s["pies"]), ("vx", s["vx"]), ("dext_hr", s["dext_hr"]),
                             ("dpies", s["dpies"]), ("apunta", s["apunta"])):
                pool[nom].append(arr[v, lado])
        pool["sep"].append(s["sep"][mask])
        pool["manos"].append(s["manos"][mask])
        print("%-38s %4d %5d %4.0f%%  %s/%s  %.2f / %.2f" % (
            cid[:38], T, int(mask.sum()), 100 * fv, s["mano"][0], s["mano"][1], rec[0], rec[1]))
    print()
    print("=== DISTRIBUCIONES AGRUPANDO CUADROS (informativas, NO son la unidad de analisis) ===")
    print("%-10s %8s %8s %8s %8s %8s %8s %8s" % ("variable", "n", "p5", "p25", "p50", "p75", "p95", "p99"))
    res = {}
    for nom in pool:
        x = np.concatenate([np.asarray(a, dtype="float64") for a in pool[nom]]) if pool[nom] else np.array([])
        p = pct(x)
        res[nom] = p
        print("%-10s %8d %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f" % (
            nom, p["n"], p["p5"], p["p25"], p["p50"], p["p75"], p["p95"], p["p99"]))
    print()
    print("=== EL RECORRIDO DE ext_hr, por clip x lado, que es la unidad de analisis ===")
    r = np.array([f["recorrido_ext_hr"] for f in filas], dtype="float64")
    r = r[np.isfinite(r)]
    print("clip x lado con recorrido: %d, mediana %.2f, minimo %.2f, por encima de 1,5: %d de %d" % (
        len(r), float(np.median(r)), float(r.min()), int((r >= 1.5).sum()), len(r)))
    cvh = np.array([f["cv_ext_hr"] for f in filas], dtype="float64")
    cvc = np.array([f["cv_ext_cr"] for f in filas], dtype="float64")
    ok = np.isfinite(cvh) & np.isfinite(cvc)
    print("coef. de variacion: ext_hr mediana %.3f, ext_cr mediana %.3f, ext_cr MENOR en %d de %d" % (
        float(np.median(cvh[ok])), float(np.median(cvc[ok])), int((cvc[ok] < cvh[ok]).sum()), int(ok.sum())))
    corr = [f["corrobora_mano"] for f in filas if f["corrobora_mano"] is not None]
    print("mano armada: los dos criterios de `cono_lib` coinciden en %d de %d tirador x clip" % (
        sum(1 for c in corr if c), len(corr)))
    print()
    print("=== CONTROL QUE AISLA EL NORMALIZADOR: los mismos largos en PIXELES ===")
    print("Si `ext_cr` varia tanto como `ext_hr` con el torso dividiendo, puede ser que el hueso se")
    print("mueva o que el torso mienta. En pixeles el torso ya no entra, asi que separa las dos cosas.")
    cvx = {"cr_px": [], "hr_px": [], "torso_px": []}
    for f in filas:
        for k in ("cr_px", "hr_px", "torso_px"):
            cvx[k].append(f["cv_" + k])
    for k in ("cr_px", "hr_px", "torso_px"):
        a = np.array(cvx[k], dtype="float64")
        a = a[np.isfinite(a)]
        print("  cv de %-9s mediana %.3f   p90 %.3f" % (k, float(np.median(a)), float(np.percentile(a, 90))))
    m = np.array(cvx["cr_px"]) < np.array(cvx["hr_px"])
    print("  |codo-raiz| varia MENOS que |hombro-raiz| en %d de %d clip x lado" % (int(m.sum()), len(m)))
    print()
    print("=== LOS UMBRALES QUE SALEN DE ESTO, y son los que usara el detector ===")
    dh = np.concatenate(pool["dext_hr"])
    dp = np.concatenate(pool["dpies"])
    print("dext_hr  p90 = %.4f  (DISPARA)   p75 = %.4f  (SIGUE)" % (
        float(np.nanpercentile(dh, 90)), float(np.nanpercentile(dh, 75))))
    print("dpies    p90 = %.4f  (DISPARA)   p75 = %.4f  (SIGUE)" % (
        float(np.nanpercentile(dp, 90)), float(np.nanpercentile(dp, 75))))
    print("ext_hr   p75 = %.4f  (brazo extendido)" % res["ext_hr"]["p75"])
    print("codo     p75 = %.1f grados" % res["codo"]["p75"])
    print()
    print("=== CRITERIO 3 Y EL TIEMPO DE ESGRIMA (t.8), medido y no copiado ===")
    disp = float(np.nanpercentile(dh, 90))
    sig = float(np.nanpercentile(dh, 75))
    largos, cuenta, sinnada = [], [], 0
    for cid in clips:
        with np.load(PRUEBAS / ("poses_%s.npz" % cid), allow_pickle=True) as z:
            d = {k: z[k] for k in z.files}
        s = PR.senales(d)
        for lado in (0, 1):
            tr = tramos_subida(s["dext_hr"][:, lado], s["val"][:, lado], disp, sig)
            cuenta.append(len(tr))
            if not tr:
                sinnada += 1
                print("  SIN NINGUN TRAMO DE SUBIDA: %s lado %d" % (cid, lado))
            largos += [b - a + 1 for a, b, _p in tr]
    largos = np.array(largos, dtype="float64")
    print("  tramos de subida de ext_hr: %d en %d clip x lado, mediana %.1f por tirador" % (
        int(sum(cuenta)), len(cuenta), float(np.median(cuenta))))
    print("  clip x lado SIN ningun tramo: %d de %d" % (sinnada, len(cuenta)))
    print("  largo del tramo en cuadros: p25 %.0f  p50 %.0f  p75 %.0f  p90 %.0f  max %.0f" % (
        float(np.percentile(largos, 25)), float(np.median(largos)),
        float(np.percentile(largos, 75)), float(np.percentile(largos, 90)), float(largos.max())))
    print("  TIEMPO DE ESGRIMA estimado = mediana del tramo = %.0f cuadros" % float(np.median(largos)))
    with open(SALIDA / "p1_variables.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print()
    print("tabla por clip x lado -> %s" % (SALIDA / "p1_variables.csv"))


if __name__ == "__main__":
    main()

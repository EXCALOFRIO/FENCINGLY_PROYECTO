"""ETAPA E, paso 4b: el VEREDICTO sobre `matting`, con el criterio arreglado.

POR QUE HACE FALTA ARREGLARLO, dicho antes de dar el numero: el criterio de `e4` calculaba
`razon = frac_solido(0) / mediana(perfil)` y ponia `inf` cuando el denominador era cero. Pero en la
mayoria de los cuadros el numerador **tambien** es cero, y `0/0 = inf` hacia que un cuadro donde no
hay NADA en ninguna rotacion contase como que `matting` acierta. Aprobaba el vacio.

Es el mismo fallo que ya esta documentado en el plan con la resolucion de la pose: *«primero use el
criterio fracion de puntos con confianza >= 0,3 y todas las resoluciones lo pasaban al 100 %. Un
criterio que aprueba todo no mide nada»*. Aqui aprobaba la nada.

EL CRITERIO ARREGLADO, con tres salidas en vez de dos, porque «no hay senal» no es «acierta»:

  SIN SENAL  el perfil entero es cero: ni en el eje ni rotado hay alfa alto. No cuenta ni a favor
             ni en contra. Se informa cuantos son, porque si son la mayoria el test no tiene poder.
  FALLA      hay senal en el perfil pero el 0 no destaca: rank fuera del 10 % superior, o razon < 2.
  ACIERTA    el 0 tiene senal (>0), esta en el 10 % superior del perfil y supera 2 veces la mediana.

Y se mide con DOS estadisticos del mismo perfil, no con uno:
  `frac_solido`  fraccion de pixeles con alfa >= 0,5. Es el declarado en `e4`, y es muy disperso.
  `alfa_medio`   alfa medio de la region. Continuo, sin umbral, y por eso no puede quedarse en cero.

Si los dos estadisticos dan veredictos distintos, se dice y no se elige el que convenga.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cono_lib as CL

PRUEBAS = CL.PRUEBAS
CLIPS = ["SentimentalLargeBedlingtonterrier", "AdorableFittingAsiansmallclawedotter"]
TOCADO = {"SentimentalLargeBedlingtonterrier": 217, "AdorableFittingAsiansmallclawedotter": 317}
MIN_AREA_REL = 0.30
MIN_N0 = 20
RAZON_MIN = 2.0
TOP_FRAC = 0.10


def juzga(perfil, est):
    """Devuelve (estado, v0, mediana, maximo, rank, n_usadas, razon)."""
    p = {int(k): v for k, v in perfil.items()}
    n0 = p[0]["n"]
    if n0 < MIN_N0:
        return None
    usar = sorted(r for r in p if p[r]["n"] >= MIN_AREA_REL * n0)
    if len(usar) < 8:
        return None
    v = np.array([p[r][est] for r in usar], dtype="float64")
    if not np.isfinite(v).all():
        return None
    v0 = float(p[0][est])
    med = float(np.median(v))
    mx = float(v.max())
    rank = int(1 + (v > v0).sum())
    if mx <= 0:
        return ("SIN SENAL", v0, med, mx, rank, len(usar), float("nan"))
    if v0 <= 0:
        return ("FALLA", v0, med, mx, rank, len(usar), 0.0)
    razon = v0 / med if med > 0 else float("inf")
    ok = (rank <= max(1, int(np.ceil(TOP_FRAC * len(usar))))) and razon >= RAZON_MIN
    return ("ACIERTA" if ok else "FALLA", v0, med, mx, rank, len(usar), razon)


resumen = {}
for clip in CLIPS:
    f = PRUEBAS / ("cono_matting_%s.npz" % clip)
    if not f.is_file():
        print("falta %s" % f)
        continue
    with np.load(f, allow_pickle=True) as z:
        filas = json.loads(str(z["filas"]))
    print("=" * 104)
    print("%s   tocado en %d   ·  %d recortes medidos" % (clip, TOCADO[clip], len(filas)))
    print("=" * 104)
    val = [r for r in filas if not r["contaminado"]]
    print("  GUARDIA un tirador por recorte: alfa medio del recorte mediana %.4f  ·  "
          "contaminados (alfa < 0,05) %d de %d" % (
              float(np.median([r["alfa_recorte"] for r in filas])), len(filas) - len(val),
              len(filas)))
    rf = np.array([r["R_frac_hoja"] for r in val])
    print("  CUANTO DEL CONO SE PUEDE PROBAR: R/hoja mediana %.2f (p10 %.2f, max %.2f)  ->  solo "
          "los primeros %.0f cm de los 90. Lo demas no cabe sin meter al rival en el recorte." % (
              float(np.median(rf)), float(np.percentile(rf, 10)), float(rf.max()),
              90 * float(np.median(rf))))
    resumen[clip] = {"R_frac_mediana": float(np.median(rf)), "n": len(val)}

    for dil in ("d0", "d3"):
        for est in ("frac_solido", "alfa_medio"):
            print()
            print("  --- cuerpo %s   estadistico %s" % (
                "sin dilatar" if dil == "d0" else "dilatado 3 px", est))
            print("  %-7s %-5s %8s %10s %10s %10s %8s %7s  %s" % (
                "cuadro", "lado", "n_0", "en 0", "mediana", "maximo", "razon", "rank", "estado"))
            cuenta = {"ACIERTA": 0, "FALLA": 0, "SIN SENAL": 0}
            for r in val:
                j = juzga(r["perfil"][dil], est)
                if j is None:
                    continue
                est_, v0, med, mx, rank, nu, razon = j
                cuenta[est_] += 1
                print("  %-7d %-5d %8d %10.5f %10.5f %10.5f %8s %4d/%-3d %s%s" % (
                    r["cuadro"], r["lado"], r["perfil"][dil]["0"]["n"] if "0" in r["perfil"][dil]
                    else r["perfil"][dil][0]["n"], v0, med, mx,
                    ("%.2f" % razon) if np.isfinite(razon) else "inf", rank, nu, est_,
                    "   <-- tocado" if r["cuadro"] == TOCADO[clip] else ""))
            n = sum(cuenta.values())
            con_senal = cuenta["ACIERTA"] + cuenta["FALLA"]
            print("  RECUENTO: acierta %d · falla %d · SIN SENAL %d   (de %d)" % (
                cuenta["ACIERTA"], cuenta["FALLA"], cuenta["SIN SENAL"], n))
            if con_senal == 0:
                v = ("NO SE PUEDE JUZGAR: no hay senal en ninguna rotacion de ningun cuadro. "
                     "`matting` no pone alfa alto en el cono, ni en su sitio ni girado.")
            elif cuenta["ACIERTA"] > 0.5 * con_senal:
                v = ("matting SI destaca el eje: acierta en %d de %d cuadros CON senal" % (
                    cuenta["ACIERTA"], con_senal))
            else:
                v = ("matting NO discrimina: con senal acierta %d de %d, o sea el cono en su sitio "
                     "no se distingue de los girados" % (cuenta["ACIERTA"], con_senal))
            print("  VEREDICTO (%s, %s): %s" % (dil, est, v))
            resumen[clip]["%s_%s" % (dil, est)] = cuenta
    print()

(PRUEBAS / "cono_e4b_veredicto.json").write_text(
    json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")
print("guardado %s" % (PRUEBAS / "cono_e4b_veredicto.json"))

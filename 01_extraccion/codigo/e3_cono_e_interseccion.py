"""ETAPA E, paso 3: LA APERTURA DEL CONO, sacada de dos medidas, y LA INTERSECCION como columna cruda.

LA APERTURA NO SE ELIGE A OJO. Se acota por los dos lados y se dice si queda hueco entre ellos:

  **SUELO (precision del eje).** Un eje que ya se mueve X grados entre dos cuadros consecutivos no
  puede apuntar mejor que X grados. Se toma el p95 del giro de `codo -> root` entre cuadros.
  Salvedad honesta declarada aqui: ese giro mezcla ruido con movimiento real del brazo, asi que es
  una cota SUPERIOR del ruido y a la vez el paso real del brazo. Como suelo de apertura vale igual:
  por debajo de eso el cono no puede ser mas fino que su propio temblor.

  **TECHO (utilidad).** Un cono solo informa si a la distancia de la hoja es MAS ESTRECHO que el
  blanco. Se toma la apertura para la que el ancho del cono a la distancia L iguala la anchura de
  hombros del rival (medida, indices 5-6). Por encima de ese techo, decir «la punta esta en el cono»
  equivale a decir «la punta esta en algun sitio del rival», que no es informacion.

  Si SUELO >= TECHO, el cono no sirve y se dice. Si SUELO < TECHO, la apertura se fija en el SUELO.

LO QUE NO SE HA MEDIDO Y SE DECLARA: **la flexion real de la hoja y el angulo muneca-hoja**. La hoja
no se ve (0,64 px de ancho a 720p), asi que nadie ha medido cuanto se aparta de la linea del brazo.
Todo lo que sale aqui cubre la incertidumbre DEL EJE, no la de la hoja respecto al eje. Se imprime la
conversion grados -> cm en la punta para que se vea lo que costaria anadirla.

LA INTERSECCION, y la trampa que se respeta: **NO se evalua contra la etiqueta de tocado**. Las
paradas correlacionan con quien tiene prioridad, y la prioridad es lo que el modelo quiere predecir:
medir la interseccion contra el resultado seria circular. Aqui solo se mide si la interseccion es
**estable y reproducible**, y se deja como columna cruda.

  estabilidad ..... cuanto cambia de un cuadro al siguiente, y cuantas veces parpadea el booleano
  reproducibilidad. se repite TODO con poses sacadas a 512x384 en vez de 384x288 y se comparan las
                    dos series. Si el numero cambia al cambiar la resolucion de entrada, no mide la
                    escena sino el ajuste.
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
PCTL_SUELO = 95
"""Declarado antes de medir: el suelo de apertura es el p95 del giro del eje entre cuadros."""


def geometria(z, apert=None):
    """Eje, longitud y apertura por cuadro para los dos tiradores. Devuelve tambien las medidas."""
    kp, sc, B = z["kp"], z["sc"], z["B"]
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    out = {"W": W, "H": H, "T": T, "lados": {}}
    for lado in (0, 1):
        mano, det = CL.mano_armada(kp, sc, B, W, H, lado)
        if mano is None:
            out["lados"][lado] = None
            continue
        vert, u, ok, largo = CL.eje_antebrazo(kp, sc, lado, mano)
        pxm_k, pxm = CL.escala_px_m(B, lado, H)
        a = CL.angulo(u)
        p = ok[:-1] & ok[1:]
        d1 = np.abs(CL.dif_ang(a[1:][p], a[:-1][p]))
        d1 = d1[np.isfinite(d1)]
        suelo = float(np.percentile(d1, PCTL_SUELO)) if d1.size >= 10 else float("nan")
        rival = 1 - lado
        o_h = CL.bueno(sc[:, rival], list(CL.HOMBROS))
        anch = float(np.median(np.linalg.norm(
            kp[o_h, rival, CL.HOMBROS[1]] - kp[o_h, rival, CL.HOMBROS[0]], axis=1))) \
            if o_h.sum() >= 10 else float("nan")
        L = CL.HOJA_M * pxm
        techo = float(np.degrees(np.arctan(0.5 * anch / max(L, 1e-9))))
        out["lados"][lado] = {
            "mano": mano, "det": det, "vert": vert, "u": u, "ok": ok, "largo_antebrazo": largo,
            "pxm": pxm, "pxm_k": pxm_k, "L": L, "suelo": suelo, "techo": techo,
            "anch_hombros_rival": anch,
            "apert": float(apert) if apert is not None else suelo,
            "salto": {"n": int(d1.size), "med": float(np.median(d1)),
                      "p90": float(np.percentile(d1, 90)), "p95": suelo,
                      "max": float(d1.max())} if d1.size else None}
    return out


resumen = {}
for clip in CLIPS:
    z = CL.carga(clip)
    g = geometria(z)
    W, H, T = g["W"], g["H"], g["T"]
    print("=" * 100)
    print("%s   %dx%d   %d cuadros   tocado en %d" % (clip, W, H, T, TOCADO[clip]))
    print("=" * 100)
    print("  LA APERTURA, acotada por los dos lados")
    for lado in (0, 1):
        d = g["lados"][lado]
        if d is None:
            print("    lado %d ... sin mano armada decidible" % lado)
            continue
        dd = d["det"]
        print("    lado %d  mano armada %s por (a), margen %.0f px  ·  (b') apuntando -> %s "
              "(margen %.3f) %s  ·  [rechazado: mas extendida -> %s]" % (
                  lado, d["mano"], dd["margen_x_px"], dd["apuntando"], dd["margen_apunta"],
                  "corrobora" if dd["corrobora_b2"] else "NO CORROBORA",
                  dd["extendida_RECHAZADO"]))
        print("      antebrazo %.0f px  ·  escala %.1f px/m  ·  L (hoja 90 cm) %.0f px" % (
            np.nanmedian(d["largo_antebrazo"]), d["pxm"], d["L"]))
        s = d["salto"]
        print("      SUELO: giro de codo->root entre cuadros  n=%d  mediana %.2f  p90 %.2f  "
              "p95 %.2f  max %.1f grados  ->  suelo = %.2f" % (
                  s["n"], s["med"], s["p90"], s["p95"], s["max"], d["suelo"]))
        print("      TECHO: hombros del rival %.0f px a distancia L %.0f px  ->  techo = %.2f grados"
              % (d["anch_hombros_rival"], d["L"], d["techo"]))
        hueco = d["techo"] / d["suelo"] if d["suelo"] > 0 else float("nan")
        print("      %s  ·  techo/suelo = %.2f" % (
            "HAY HUECO: la apertura se fija en el suelo, %.1f grados" % d["suelo"]
            if d["suelo"] < d["techo"] else
            "NO HAY HUECO: el temblor del eje ya es mas ancho que el rival. El cono NO informa",
            hueco))
        lat = d["L"] * np.tan(np.radians(d["apert"]))
        print("      a %.1f grados de semiapertura, la punta cae en +-%.0f px = +-%.0f cm "
              "(la hoja entera mide 90 cm)" % (d["apert"], lat, 100 * lat / d["pxm"]))

    print()
    print("  DESCRIPTIVO CON n=1 POR TIRADOR, no calibra nada: en el cuadro del tocado, angulo entre")
    print("  el eje del antebrazo y la direccion mano -> centro del torso del rival")
    kt = TOCADO[clip]
    kp, B = z["kp"], z["B"]
    for lado in (0, 1):
        d = g["lados"][lado]
        if d is None or not d["ok"][kt]:
            continue
        rival = 1 - lado
        b = B[kt, rival]
        if not np.isfinite(b).all():
            continue
        c = np.array([0.5 * (b[0] + b[2]) * W, (b[1] + 0.30 * (b[3] - b[1])) * H])
        v = c - d["vert"][kt]
        r = float(np.linalg.norm(v))
        a = float(np.degrees(np.arccos(np.clip(v @ d["u"][kt] / max(r, 1e-9), -1, 1))))
        print("    lado %d: distancia mano->torso rival %.0f px = %.0f cm (hoja 90 cm)  ·  "
              "desvio del eje %.1f grados" % (lado, r, 100 * r / d["pxm"], a))

    print()
    print("  LA INTERSECCION como columna cruda, con su estabilidad y su reproducibilidad")
    series = {}
    for suf, etiq in (("", "384x288"), ("_512x384", "512x384")):
        try:
            zz = CL.carga(clip, suf)
        except FileNotFoundError:
            continue
        gg = geometria(zz, apert=None)
        d0, d1 = gg["lados"][0], gg["lados"][1]
        if d0 is None or d1 is None:
            continue
        ap0, ap1 = d0["apert"], d1["apert"]
        n = min(gg["T"], T)
        col = {k: np.full(n, np.nan) for k in ("area_rel", "peso_max", "d_ejes_px", "d_ejes_cm",
                                               "ang_ejes", "px", "py")}
        cru = np.zeros(n, dtype=bool)
        val = np.zeros(n, dtype=bool)
        for k in range(n):
            if not (d0["ok"][k] and d1["ok"][k]):
                continue
            val[k] = True
            r = CL.cruce(d0["vert"][k], d0["u"][k], d0["L"], ap0,
                         d1["vert"][k], d1["u"][k], d1["L"], ap1)
            col["area_rel"][k] = r["area_rel"]
            col["peso_max"][k] = r["peso_max"]
            col["px"][k], col["py"][k] = r["px"], r["py"]
            cru[k] = r["n_inter"] > 0
            dd = CL.dist_segmentos(d0["vert"][k], d0["vert"][k] + d0["L"] * d0["u"][k],
                                   d1["vert"][k], d1["vert"][k] + d1["L"] * d1["u"][k])
            col["d_ejes_px"][k] = dd
            col["d_ejes_cm"][k] = 100.0 * dd / (0.5 * (d0["pxm"] + d1["pxm"]))
            col["ang_ejes"][k] = abs(CL.dif_ang(CL.angulo(d0["u"][k:k + 1])[0],
                                                CL.angulo(d1["u"][k:k + 1])[0]))
        series[etiq] = {"col": col, "cruce": cru, "val": val, "ap": (ap0, ap1)}
        print("    %s  apertura %.1f / %.1f grados  ·  cuadros validos %d de %d  ·  "
              "se cruzan en %.1f %% de los validos" % (
                  etiq, ap0, ap1, int(val.sum()), n,
                  100 * cru[val].mean() if val.any() else float("nan")))
        pm = col["peso_max"][val]
        de = col["d_ejes_cm"][val]
        print("       peso_max: mediana %.3f  p90 %.3f  max %.3f  ·  d_ejes: mediana %.0f cm  "
              "min %.0f cm" % (np.nanmedian(pm), np.nanpercentile(pm, 90), np.nanmax(pm),
                               np.nanmedian(de), np.nanmin(de)))
        p = val[:-1] & val[1:]
        dp = np.abs(col["peso_max"][1:][p] - col["peso_max"][:-1][p])
        flip = float((cru[1:][p] != cru[:-1][p]).mean()) if p.any() else float("nan")
        print("       ESTABILIDAD: |d peso_max| entre cuadros mediana %.3f  p90 %.3f  ·  el "
              "booleano parpadea en %.1f %% de los pares" % (
                   np.nanmedian(dp), np.nanpercentile(dp, 90), 100 * flip))

    if len(series) == 2:
        a, b = series["384x288"], series["512x384"]
        v = a["val"] & b["val"]
        print("    REPRODUCIBILIDAD 384x288 contra 512x384 (n=%d cuadros validos en las dos)"
              % int(v.sum()))
        ac = float((a["cruce"][v] == b["cruce"][v]).mean())
        print("       el booleano «se cruzan» coincide en %.1f %% de los cuadros" % (100 * ac))
        for nom in ("peso_max", "area_rel", "d_ejes_cm", "ang_ejes"):
            x, y = a["col"][nom][v], b["col"][nom][v]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 10:
                continue
            r = float(np.corrcoef(x[m], y[m])[0, 1])
            rx = np.argsort(np.argsort(x[m])).astype("float64")
            ry = np.argsort(np.argsort(y[m])).astype("float64")
            rs = float(np.corrcoef(rx, ry)[0, 1])
            print("       %-10s pearson %6.3f  spearman %6.3f  ·  |diferencia| mediana %.3f  "
                  "p90 %.3f" % (nom, r, rs, float(np.nanmedian(np.abs(x[m] - y[m]))),
                                float(np.nanpercentile(np.abs(x[m] - y[m]), 90))))

    if "384x288" in series:
        a = series["384x288"]
        f = PRUEBAS / ("cono_interseccion_%s.csv" % clip)
        with f.open("w", encoding="utf-8") as fh:
            fh.write("cuadro,valido,se_cruzan,area_rel,peso_max,d_ejes_px,d_ejes_cm,ang_ejes,px,py\n")
            for k in range(len(a["val"])):
                fh.write("%d,%d,%d,%s\n" % (
                    k, int(a["val"][k]), int(a["cruce"][k]),
                    ",".join(("" if not np.isfinite(a["col"][n][k]) else "%.4f" % a["col"][n][k])
                             for n in ("area_rel", "peso_max", "d_ejes_px", "d_ejes_cm",
                                       "ang_ejes", "px", "py"))))
        print("    columna cruda -> %s   (SIN evaluar contra la etiqueta de tocado)" % f)
        resumen[clip] = {
            "apertura": a["ap"],
            "suelo": [g["lados"][l]["suelo"] if g["lados"][l] else None for l in (0, 1)],
            "techo": [g["lados"][l]["techo"] if g["lados"][l] else None for l in (0, 1)],
            "L_px": [g["lados"][l]["L"] if g["lados"][l] else None for l in (0, 1)],
            "pxm": [g["lados"][l]["pxm"] if g["lados"][l] else None for l in (0, 1)]}
    print()

(PRUEBAS / "cono_e3_resumen.json").write_text(
    json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")
print("guardado %s" % (PRUEBAS / "cono_e3_resumen.json"))

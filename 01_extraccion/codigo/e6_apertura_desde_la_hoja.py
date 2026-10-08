"""ETAPA E, paso 6: quitar el confuso del BRAZO y calibrar la apertura con la hoja MEDIDA.

EL CONFUSO QUE HAY QUE DESCARTAR ANTES DE CREERSE NADA: en `e5` la recta detectada se alinea con el
antebrazo con una mediana de 10,4 grados contra 45 de azar. Pero **el antebrazo tambien es una recta
alineada con el antebrazo**. Si lo que `HoughLinesP` encuentra fuera el borde del brazo que la
dilatacion de 8 px no llego a tapar, saldria exactamente el mismo numero y la conclusion seria falsa.

COMO SE SEPARA, y es geometrico y no discutible: la hoja sale de la mano **hacia delante**, alejandose
del codo; el brazo esta **detras** de la mano. Se proyecta el segmento detectado sobre el eje
`codo -> root` tomando la mano como origen:

    proyeccion > 0  ->  delante de la mano: lado de la HOJA
    proyeccion < 0  ->  detras de la mano: lado del BRAZO, y ese hallazgo no cuenta

Se exige que el punto MAS LEJANO del segmento este delante (`t_max > 0`) y se informa aparte el
reparto, porque si la mayoria cayera detras, el resultado de `e5` seria el borde del brazo.

Y LO QUE ESTO PERMITE, QUE NO SE PODIA ANTES: **medir el angulo entre el brazo y la hoja de verdad.**
Hasta ahora ese angulo era lo unico que no estaba medido y por eso la apertura del cono se fijaba con
el temblor del eje, que es otra cosa. Con la hoja detectada, la apertura se calibra con la
distribucion medida de ese angulo:

    semiapertura = p90 de |angulo hoja - eje|   sobre las detecciones del lado de la hoja

y se compara con el TECHO de utilidad (la anchura de hombros del rival a la distancia de la hoja) que
ya se midio en `e3`. Si la apertura necesaria pasa del techo, el cono no localiza y se dice.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cono_lib as CL

PRUEBAS = CL.PRUEBAS
CLIPS = {"SentimentalLargeBedlingtonterrier": 217, "AdorableFittingAsiansmallclawedotter": 317}
AZAR = 45.0


def dif90(a, b):
    return abs((a - b + 90.0) % 180.0 - 90.0)


geo_cache = {}


def geo(clip):
    if clip in geo_cache:
        return geo_cache[clip]
    z = CL.carga(clip)
    kp, sc, B = z["kp"], z["sc"], z["B"]
    W, H = int(z["W"]), int(z["H"])
    d = {}
    for lado in (0, 1):
        mano, _x = CL.mano_armada(kp, sc, B, W, H, lado)
        vert, u, ok, _l = CL.eje_antebrazo(kp, sc, lado, mano)
        _pk, pxm = CL.escala_px_m(B, lado, H)
        rival = 1 - lado
        oh = CL.bueno(sc[:, rival], list(CL.HOMBROS))
        anch = float(np.median(np.linalg.norm(
            kp[oh, rival, CL.HOMBROS[1]] - kp[oh, rival, CL.HOMBROS[0]], axis=1)))
        d[lado] = {"vert": vert, "u": u, "pxm": pxm, "L": CL.HOJA_M * pxm, "anch": anch}
    geo_cache[clip] = (d, W, H)
    return geo_cache[clip]


resumen = {}
for clip, kt in CLIPS.items():
    f = PRUEBAS / ("cono_hoja_%s.json" % clip)
    if not f.is_file():
        print("falta %s" % f)
        continue
    filas = json.loads(f.read_text(encoding="utf-8"))
    g, W, H = geo(clip)
    print("=" * 100)
    print("%s   %dx%d   tocado en %d   ·  %d recortes" % (clip, W, H, kt, len(filas)))
    print("=" * 100)
    resumen[clip] = {}
    for key in ("p99.0", "p99.5"):
        det = []
        for r in filas:
            if r["contaminado"]:
                continue
            ln = r["det"].get(key, {}).get("linea")
            if not ln:
                continue
            lado, k = r["lado"], r["cuadro"]
            vert, u = g[lado]["vert"][k], g[lado]["u"][k]
            p1 = np.asarray(ln["p1"], dtype="float64")
            p2 = np.asarray(ln["p2"], dtype="float64")
            t1 = float((p1 - vert) @ u)
            t2 = float((p2 - vert) @ u)
            det.append({"clip": clip, "cuadro": k, "lado": lado, "largo": ln["largo"],
                        "dif_eje": ln["dif_eje"], "dif_pulgar": ln.get("dif_pulgar"),
                        "t_min": min(t1, t2), "t_max": max(t1, t2),
                        "t_max_cm": 100.0 * max(t1, t2) / g[lado]["pxm"],
                        "largo_cm": 100.0 * ln["largo"] / g[lado]["pxm"],
                        "R": r["R"]})
        if not det:
            print("  %s: sin detecciones" % key)
            continue
        delante = [d for d in det if d["t_max"] > 0]
        detras = [d for d in det if d["t_max"] <= 0]
        print()
        print("  --- umbral %s   ·  %d rectas detectadas" % (key, len(det)))
        print("  DESCARTE DEL CONFUSO DEL BRAZO: delante de la mano %d (%.1f %%)  ·  detras %d "
              "(%.1f %%)" % (len(delante), 100.0 * len(delante) / len(det), len(detras),
                             100.0 * len(detras) / len(det)))
        if not delante:
            print("  TODAS DETRAS: lo que detecta `e5` es el borde del brazo, no la hoja.")
            continue
        dd = np.array([d["dif_eje"] for d in delante])
        lc = np.array([d["largo_cm"] for d in delante])
        tc = np.array([d["t_max_cm"] for d in delante])
        print("  SOLO LAS DE DELANTE (n=%d):" % len(delante))
        print("    |angulo hoja - eje del antebrazo|  mediana %.1f  p75 %.1f  p90 %.1f  "
              "(azar declarado %.0f)" % (
                  np.median(dd), np.percentile(dd, 75), np.percentile(dd, 90), AZAR))
        print("    dentro de 10 grados %.1f %%  ·  20 %.1f %%  ·  30 %.1f %%" % (
            100 * (dd <= 10).mean(), 100 * (dd <= 20).mean(), 100 * (dd <= 30).mean()))
        print("    largo del segmento: mediana %.0f cm  p90 %.0f cm   ·  llega hasta %.0f cm "
              "de la mano (mediana)  ·  la hoja mide 90 cm" % (
                  np.median(lc), np.percentile(lc, 90), np.median(tc)))
        dp = np.array([d["dif_pulgar"] for d in delante if d["dif_pulgar"] is not None])
        if dp.size:
            print("    |angulo hoja - eje del PULGAR|     mediana %.1f  p75 %.1f  p90 %.1f" % (
                np.median(dp), np.percentile(dp, 75), np.percentile(dp, 90)))
            print("    -> el %s gana como predictor de la direccion de la hoja" % (
                "ANTEBRAZO" if np.median(dd) < np.median(dp) else "PULGAR"))
        if detras:
            db = np.array([d["dif_eje"] for d in detras])
            print("    (las de detras, para comparar: |dif| mediana %.1f -- si fueran el brazo, "
                  "estarian MUY alineadas)" % np.median(db))

        print()
        print("  CALIBRACION DE LA APERTURA CON LA HOJA MEDIDA")
        for lado in (0, 1):
            dl = np.array([d["dif_eje"] for d in delante if d["lado"] == lado])
            if dl.size < 5:
                print("    lado %d: solo %d detecciones, no calibra" % (lado, dl.size))
                continue
            ap = float(np.percentile(dl, 90))
            techo = float(np.degrees(np.arctan(0.5 * g[lado]["anch"] / g[lado]["L"])))
            lat = g[lado]["L"] * np.tan(np.radians(ap))
            print("    lado %d (n=%d): semiapertura necesaria = p90 = %.1f grados  ·  techo de "
                  "utilidad = %.1f grados  ->  %s" % (
                      lado, dl.size, ap, techo,
                      "CABE" if ap <= techo else "NO CABE: %.1fx por encima del techo" % (ap / techo)))
            print("             a %.1f grados la punta cae en +-%.0f cm de los 90 de hoja" % (
                ap, 100 * lat / g[lado]["pxm"]))
        resumen[clip][key] = {
            "n_det": len(det), "n_delante": len(delante),
            "frac_delante": len(delante) / len(det),
            "dif_eje_mediana": float(np.median(dd)), "dif_eje_p90": float(np.percentile(dd, 90)),
            "dif_pulgar_mediana": float(np.median(dp)) if dp.size else None,
            "largo_cm_mediana": float(np.median(lc))}
    print()

(PRUEBAS / "cono_e6_apertura.json").write_text(
    json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")
print("guardado %s" % (PRUEBAS / "cono_e6_apertura.json"))

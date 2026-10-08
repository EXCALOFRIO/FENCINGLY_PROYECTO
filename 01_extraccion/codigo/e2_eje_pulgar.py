"""ETAPA E, paso 2: EL EJE DEL PULGAR. Con los indices MEDIDOS en `e0_mapa_indices.py`.

LA IDEA ES DEL USUARIO: el florete sale de la mano prolongando la linea del pulgar, asi que la recta
del pulgar deberia dar la direccion de la hoja aunque la hoja no se vea.

CORRECCION QUE OBLIGA A REESCRIBIR ESTO: los indices que daba el plan (91-95 y 112-116) caen **sobre
la careta**, no sobre la mano. Los buenos son **21-41 y 42-62**, con root en 21 y 42 y el pulgar en
los cuatro siguientes. Vease `e0_mapa_indices.py`.

Y UNA CONSECUENCIA DE ESE MAPA: en el bloque de cuerpo (0-14) **no hay muneca**, porque la muneca ES
el root de la mano. Asi que la comparacion que pedia la tarea, `wrist -> hand_root`, no existe como
segmento. Su sustituto legitimo es `codo -> root`, y es el que se mide.

TODO EL CRITERIO QUEDA DECLARADO ANTES DE MIRAR LOS NUMEROS:

  1. **Cual mano lleva el arma.** Dos criterios independientes, los dos declarados aqui:
       (a) el brazo armado apunta al rival: gana la mano con x mas cerca del rival
       (b) el brazo armado esta mas extendido: gana la mano con |hombro-root| mayor
     Si los dos coinciden, la eleccion es firme. Si discrepan, se dice y no se elige.

  2. **Seis ejes candidatos**, porque «el eje del pulgar sirve» solo significa algo comparado:
       A  root -> pulgar4          el que propone la idea, literal
       A' root -> centroide pulgar robusto al orden interno de los cuatro puntos del pulgar
       B  codo -> root             el segmento largo del mismo brazo (sustituto de wrist->root)
       C  PCA de root + pulgar     cinco puntos en vez de dos
       D  hombro -> codo           NO es el eje de la hoja: es la REFERENCIA de cuanto gira de
                                   verdad un segmento largo entre cuadros a esta cadencia
       E  codo -> pulgar4          apuntado del pulgar con brazo de palanca largo

  3. **Estabilidad temporal**: giro absoluto del eje entre cuadros consecutivos, con colas.

  4. **Cuanto de ese giro es RUIDO y cuanto movimiento real.** Criterio declarado antes de medir: con
     ruido blanco el salto a lag 2 vale lo mismo que a lag 1 (razon ~1); con movimiento suave vale el
     doble (razon ~2). Se mide `mediana|d lag2| / mediana|d lag1|` y se lee en esa escala.

  5. **La precision de apuntado**: dos estimadores independientes de la MISMA recta fisica que
     discrepan X grados acotan la precision en X grados. Eso, y no el ojo, fija la apertura del cono.

  6. **Que significan esos grados en la punta**: con la escala de la imagen
     (px/m = alto_caja / 1,75) y el florete de 90 cm de la FIE, un error angular de theta pone la
     punta a `L * tan(theta)` px de donde se cree. Si eso es del tamano del rival, el cono no localiza.

LO QUE AQUI NO SE HACE: dibujar hoja. El eje es una direccion medida; la hoja no se ha visto.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

PRUEBAS = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
CLIPS = ["SentimentalLargeBedlingtonterrier", "AdorableFittingAsiansmallclawedotter"]
TOCADO = {"SentimentalLargeBedlingtonterrier": 217, "AdorableFittingAsiansmallclawedotter": 317}

MANOS = {"A": {"root": 21, "pulgar": [22, 23, 24, 25], "codo": 8, "hombro": 6},
         "B": {"root": 42, "pulgar": [43, 44, 45, 46], "codo": 7, "hombro": 5}}
"""Emparejamiento codo-mano MEDIDO por constancia de hueso: 8-21 da cv 0,087-0,200 y 7-42 cv
0,122-0,210, mientras los cruces dan longitudes del 51-60 % del alto de caja y cv hasta 0,714."""
CONF_MIN = 0.3
ALTURA_M = 1.75
HOJA_M = 0.90
"""Longitud maxima de hoja de florete segun la FIE. Se usa como COTA, no como medida de este clip."""


def carga(clip):
    with np.load(PRUEBAS / ("cono_poses_%s.npz" % clip), allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def bueno(sc, idx):
    return (sc[:, idx] >= CONF_MIN).all(axis=1)


def ang(v):
    return np.degrees(np.arctan2(v[:, 1], v[:, 0]))


def dif(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def pca_dir(kp, idx, root, punta):
    T = kp.shape[0]
    V = np.full((T, 2), np.nan)
    for k in range(T):
        P = kp[k, idx, :].astype("float64")
        if not np.isfinite(P).all():
            continue
        Q = P - P.mean(axis=0)
        _u, _s, vt = np.linalg.svd(Q, full_matrices=False)
        d = vt[0]
        r = kp[k, punta] - kp[k, root]
        if np.dot(d, r) < 0:
            d = -d
        V[k] = d
    return V


def perfil(a, ok):
    T = len(a)
    p1 = ok[:-1] & ok[1:]
    d1 = np.abs(dif(a[1:][p1], a[:-1][p1]))
    p2 = ok[:-2] & ok[2:]
    d2 = np.abs(dif(a[2:][p2], a[:-2][p2]))
    d1 = d1[np.isfinite(d1)]
    d2 = d2[np.isfinite(d2)]
    if d1.size < 10:
        return None
    return {"n": int(d1.size), "med": float(np.median(d1)),
            "p75": float(np.percentile(d1, 75)), "p90": float(np.percentile(d1, 90)),
            "p95": float(np.percentile(d1, 95)), "max": float(d1.max()),
            "f10": float((d1 > 10).mean()), "f20": float((d1 > 20).mean()),
            "f40": float((d1 > 40).mean()),
            "lag2": float(np.median(d2) / np.median(d1)) if d2.size and np.median(d1) > 0 else float("nan")}


salida = {}
for clip in CLIPS:
    z = carga(clip)
    kp, sc, B = z["kp"], z["sc"], z["B"]
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    print("=" * 100)
    print("%s   %dx%d   %d cuadros   tocado en %d" % (clip, W, H, T, TOCADO[clip]))
    print("=" * 100)

    print("  QUE MANO LLEVA EL ARMA  (dos criterios declarados, independientes)")
    armada, escala = {}, {}
    for lado in (0, 1):
        k, s = kp[:, lado], sc[:, lado]
        alto = float(np.nanmedian((B[:, lado, 3] - B[:, lado, 1]) * H))
        escala[lado] = alto / ALTURA_M
        fila = {}
        for nom, M in MANOS.items():
            o = bueno(s, [M["root"], M["hombro"]])
            if o.sum() < 10:
                fila[nom] = None
                continue
            fila[nom] = {"x": float(np.median(k[o, M["root"], 0])),
                         "ext": float(np.median(np.linalg.norm(
                             k[o, M["root"]] - k[o, M["hombro"]], axis=1)))}
        if not (fila.get("A") and fila.get("B")):
            armada[lado] = None
            print("    lado %d ... sin datos de las dos manos" % lado)
            continue
        rival = 1 - lado
        xr = float(np.nanmedian(0.5 * (B[:, rival, 0] + B[:, rival, 2]) * W))
        a_cerca = "A" if abs(fila["A"]["x"] - xr) < abs(fila["B"]["x"] - xr) else "B"
        a_ext = "A" if fila["A"]["ext"] > fila["B"]["ext"] else "B"
        armada[lado] = a_cerca if a_cerca == a_ext else None
        print("    lado %d (rival en x=%.0f): mano A root x=%.0f ext=%.0f  ·  mano B root x=%.0f "
              "ext=%.0f" % (lado, xr, fila["A"]["x"], fila["A"]["ext"],
                            fila["B"]["x"], fila["B"]["ext"]))
        print("      (a) mas cerca del rival -> %s   (b) mas extendida -> %s   ==> %s" % (
            a_cerca, a_ext, ("armada = mano %s" % armada[lado]) if armada[lado]
            else "DISCREPAN: no se elige"))
        print("      escala: alto de caja %.0f px -> %.1f px/m -> hoja de 90 cm = %.0f px" % (
            alto, escala[lado], HOJA_M * escala[lado]))

    print()
    print("  LONGITUD DEL BRAZO DE PALANCA (1 px de error en la punta del segmento = X grados)")
    for lado in (0, 1):
        if not armada[lado]:
            continue
        M = MANOS[armada[lado]]
        k, s = kp[:, lado], sc[:, lado]
        for nom, a, b in (("root->pulgar4", M["root"], M["pulgar"][3]),
                          ("codo->root", M["codo"], M["root"]),
                          ("hombro->codo", M["hombro"], M["codo"]),
                          ("codo->pulgar4", M["codo"], M["pulgar"][3])):
            o = bueno(s, [a, b])
            if o.sum() < 10:
                continue
            d = np.linalg.norm(k[o, b] - k[o, a], axis=1)
            L = float(np.median(d))
            print("    lado %d %-14s mediana %7.2f px  ·  1 px = %6.2f grados" % (
                lado, nom, L, np.degrees(np.arctan(1.0 / max(L, 1e-6)))))
        o = bueno(s, [M["root"]] + M["pulgar"])
        c = k[:, M["pulgar"]].mean(axis=1) - k[:, M["root"]]
        d = np.linalg.norm(c, axis=1)[o]
        print("    lado %d %-14s mediana %7.2f px  ·  1 px = %6.2f grados" % (
            lado, "root->centr.p", np.median(d),
            np.degrees(np.arctan(1.0 / max(np.median(d), 1e-6)))))

    print()
    print("  ESTABILIDAD TEMPORAL DEL EJE   (grados de giro entre cuadros consecutivos)")
    print("  %-4s %-18s %6s %7s %7s %7s %7s %7s %6s %6s %6s %6s" % (
        "lado", "eje", "n", "med", "p75", "p90", "p95", "max", ">10", ">20", ">40", "lag2"))
    filas, ANG = [], {}
    for lado in (0, 1):
        if not armada[lado]:
            continue
        M = MANOS[armada[lado]]
        k, s = kp[:, lado], sc[:, lado]
        cad = [M["root"]] + M["pulgar"]
        cent = k[:, M["pulgar"]].mean(axis=1)
        variantes = [
            ("A  root->pulgar4", k[:, M["pulgar"][3]] - k[:, M["root"]], [M["root"], M["pulgar"][3]]),
            ("A' root->centr.p", cent - k[:, M["root"]], cad),
            ("B  codo->root", k[:, M["root"]] - k[:, M["codo"]], [M["codo"], M["root"]]),
            ("C  PCA pulgar", pca_dir(k, cad, M["root"], M["pulgar"][3]), cad),
            ("D  hombro->codo", k[:, M["codo"]] - k[:, M["hombro"]], [M["hombro"], M["codo"]]),
            ("E  codo->pulgar4", k[:, M["pulgar"][3]] - k[:, M["codo"]], [M["codo"], M["pulgar"][3]]),
        ]
        for nom, v, idx in variantes:
            o = bueno(s, idx)
            a = ang(np.asarray(v, dtype="float64"))
            ANG[(lado, nom)] = (a, o)
            r = perfil(a, o)
            if r is None:
                continue
            r.update({"clip": clip, "lado": lado, "eje": nom})
            filas.append(r)
            print("  %-4d %-18s %6d %7.2f %7.2f %7.2f %7.2f %7.1f %6.3f %6.3f %6.3f %6.2f" % (
                lado, nom, r["n"], r["med"], r["p75"], r["p90"], r["p95"], r["max"],
                r["f10"], r["f20"], r["f40"], r["lag2"]))

    print()
    print("  PRECISION DE APUNTADO: discrepancia entre estimadores de la MISMA recta del pulgar")
    acot = []
    for lado in (0, 1):
        if not armada[lado]:
            continue
        for n1, n2 in (("A  root->pulgar4", "A' root->centr.p"),
                       ("A  root->pulgar4", "C  PCA pulgar"),
                       ("A' root->centr.p", "C  PCA pulgar"),
                       ("A  root->pulgar4", "B  codo->root")):
            if (lado, n1) not in ANG or (lado, n2) not in ANG:
                continue
            a1, o1 = ANG[(lado, n1)]
            a2, o2 = ANG[(lado, n2)]
            o = o1 & o2 & np.isfinite(a1) & np.isfinite(a2)
            d = np.abs(dif(a1[o], a2[o]))
            if d.size < 10:
                continue
            r = {"clip": clip, "lado": lado, "par": "%s vs %s" % (n1.strip(), n2.strip()),
                 "n": int(d.size), "med": float(np.median(d)),
                 "p90": float(np.percentile(d, 90)), "p95": float(np.percentile(d, 95)),
                 "max": float(d.max())}
            acot.append(r)
            L = HOJA_M * escala[lado]
            lat = L * np.tan(np.radians(min(r["p90"], 89.0)))
            print("    lado %d  %-38s n=%3d  mediana %6.2f  p90 %6.2f  p95 %6.2f  max %6.1f "
                  "grados  ·  p90 en la punta = %.0f px = %.0f cm" % (
                      lado, r["par"], r["n"], r["med"], r["p90"], r["p95"], r["max"],
                      lat, 100 * lat / escala[lado]))

    salida[clip] = {"armada": armada, "escala_px_m": escala, "estabilidad": filas,
                    "acuerdo": acot, "W": W, "H": H, "T": T}
    print()

f = PRUEBAS / "cono_e2_eje.json"
f.write_text(json.dumps(salida, indent=1, ensure_ascii=False), encoding="utf-8")
print("guardado %s" % f)

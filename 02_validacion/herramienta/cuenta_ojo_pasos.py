"""VERDAD-TERRENO DE ZANCADAS en espacio de IMAGEN, y el recall del detector contra ella.

QUE ES Y QUE NO ES, porque de eso depende que la cifra valga:

  SI es independiente de la homografia metrica: todo va en pixeles del cuadro ESTABILIZADO, y el
     umbral de zancada sale del alto de la caja del propio tirador, que es una medida de imagen.
  SI es independiente del criterio del detector: aqui no hay ningun umbral de VELOCIDAD, que es
     justo lo que se sospecha que falla. Se segmenta por DESPLAZAMIENTO, que es lo que define una
     zancada, y sin limite de duracion.
  NO es independiente de los keypoints de la pose. Eso se acota mirando el mosaico: en los dos clips
     contados las cruces caen sobre los zapatos practicamente en todas las casillas, y la procedencia
     dice 267 de 270 poses MEDIDAS en UglyNice y 271 de 291 en Gentle.

EL UMBRAL, declarado: una zancada es un traslado de al menos `Z_FRAC` = 0,14 del alto de la caja del
tirador. 0,14 de la estatura es el largo del pie de una persona, que es la regla que el convenio
escrito en `CONVENIO_CONTEO_PASOS.md` fija como "inconfundible de un ajuste de peso". En un tirador
de 1,75 m son 25 cm.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import pista as PI
import posicion as PO

Z_FRAC = 0.14
REVERSO_MAX = 2
TOL_CUADROS = 4
"""Tolerancia para decir que una deteccion "cae sobre" una zancada contada, en cuadros. 4 cuadros son
0,13 s, menos de lo que dura una zancada, asi que no puede emparejar cosas distintas."""


def pies_estabilizados(res, poses, lado):
    """QUE HACE: la x de cada pie en pixeles del cuadro estabilizado, y el alto de la caja."""
    r, Mk = res["_r"], res["_Mk"]
    T = r["T"]
    X = np.full((T, 2), np.nan)
    alto = np.full(T, np.nan)
    for k in np.nonzero(r["mask"])[0]:
        if not r["pres"][k, lado] or k >= poses["KP"].shape[0]:
            continue
        b = r["B"][k, lado]
        if np.isfinite(b).all():
            alto[k] = (b[3] - b[1]) * r["H"]
        for j, p in enumerate(PO._pie_pose(poses["KP"], poses["SC"], k, lado)):
            if p is None:
                continue
            q = Mk[k] @ np.array([p[0], p[1], 1.0])
            X[k, j] = q[0] / q[2]
    return X, alto


def zancadas(X, alto, mask):
    """QUE HACE: los traslados de cada pie, segmentados por MESETAS.

    Y ES LA SEGUNDA VERSION, porque la primera repetia el fallo que yo mismo habia diagnosticado en
    la v3 del detector: segmentaba por tramos MONOTONOS maximales, y la posicion de un pie a lo largo
    del clip es una ESCALERA, que es monotona completa. Resultado medido de esa version: **0 zancadas
    en `UglyNiceAplomadofalcon` y una sola "zancada" de -406 px en 2,03 s en
    `AliveIgnorantBarebirdbat`**, que es media docena de zancadas metidas en una. Haber escrito el
    diagnostico de la v3 no impidio repetirlo; lo que lo caza es mirar la salida.

    LO QUE SI SEGMENTA UNA ESCALERA SON LAS MESETAS. Un pie alterna entre PLANTADO (su posicion no
    cambia) y en el aire. Asi que:

      1. un cuadro esta PLANTADO si el recorrido de la posicion suavizada en +-2 cuadros no pasa de
         `PLANTADO`, que es 3 veces el ruido MEDIDO de la propia serie;
      2. las mesetas son las rachas de cuadros plantados de al menos 3 cuadros;
      3. una ZANCADA es el salto entre dos mesetas consecutivas cuando la diferencia de sus posiciones
         medianas pasa de `Z_FRAC` del alto de la caja.

    No hay ningun umbral de velocidad, ni limite de duracion, ni escala metrica: una zancada lenta de
    un segundo se cuenta igual que una rapida de un decimo. Esa es toda la razon de ser de esta
    medida, porque lo que se sospecha del detector es justo un umbral de velocidad.
    """
    Z = Z_FRAC * float(np.nanmedian(alto))
    ks = np.nonzero(mask)[0]
    out = []
    diag = {}
    for j in (0, 1):
        x = PO._suave(X[:, j], 5)
        rr = np.abs(X[:, j] - x)
        rr = rr[np.isfinite(rr)]
        ruido = float(np.percentile(rr, 95)) if rr.size > 20 else 1.0
        plant_umb = max(3.0 * ruido, 0.02 * float(np.nanmedian(alto)))
        diag["ruido_pie%d" % j] = round(ruido, 2)
        diag["plantado_pie%d" % j] = round(plant_umb, 2)
        pl = np.zeros(len(x), bool)
        for k in range(ks[0], ks[-1] + 1):
            v = x[max(0, k - 2):k + 3]
            v = v[np.isfinite(v)]
            pl[k] = v.size >= 3 and (v.max() - v.min()) <= plant_umb
        mesetas = []
        k = ks[0]
        while k <= ks[-1]:
            if not pl[k]:
                k += 1
                continue
            e = k
            while e + 1 <= ks[-1] and pl[e + 1]:
                e += 1
            if e - k + 1 >= 3:
                v = x[k:e + 1]
                mesetas.append((int(k), int(e), float(np.nanmedian(v))))
            k = e + 1
        diag["mesetas_pie%d" % j] = len(mesetas)
        for a, b in zip(mesetas, mesetas[1:]):
            salto = b[2] - a[2]
            if abs(salto) >= Z:
                out.append({"pie": j, "inicio": a[1], "fin": b[0], "px": float(salto),
                            "sentido": int(np.sign(salto))})
    out.sort(key=lambda z: z["inicio"])
    return out, Z, diag


def agrupa(zs, hacia):
    """CONVENIO DE AGRUPAMIENTO, el escrito en `CONVENIO_CONTEO_PASOS.md`: una marcha es una PAREJA de
    zancadas consecutivas del mismo sentido y de pies distintos. La que se queda sin pareja es MEDIO
    PASO y se cuenta aparte."""
    usado = [False] * len(zs)
    pasos, medios = [], []
    for a in range(len(zs)):
        if usado[a]:
            continue
        usado[a] = True
        par = None
        for b in range(a + 1, len(zs)):
            if usado[b]:
                continue
            if zs[b]["pie"] != zs[a]["pie"] and zs[b]["sentido"] == zs[a]["sentido"]:
                par = b
            break
        if par is None:
            medios.append(zs[a])
            continue
        usado[par] = True
        pasos.append({"inicio": zs[a]["inicio"], "fin": zs[par]["fin"],
                      "sentido": zs[a]["sentido"],
                      "tipo": "marcha" if zs[a]["sentido"] * hacia > 0 else "romper"})
    return pasos, medios


def una(clip, lado):
    res = PI.analiza(clip)
    poses = PO.carga_poses(clip)
    ap = PO.apoyos(res, poses)
    fps = PO._fps(PI.indice_videos()[clip])
    ev = PO.eventos(ap, res["_r"], fps, poses, res=res)
    X, alto = pies_estabilizados(res, poses, lado)
    zs, Z, diag = zancadas(X, alto, res["_r"]["mask"])
    C = ap["C"]
    hacia = float(np.sign(np.nanmedian(C[:, 1 - lado, 0]) - np.nanmedian(C[:, lado, 0])) or 1.0)
    """EL SENTIDO EN PIXELES tiene que casar con el sentido en metros, y el signo de la x metrica se
    da la vuelta en `apoyos` segun que borde salio como A. Se resuelve midiendo la correlacion entre
    las dos series en vez de suponerla."""
    xm = PO._suave(C[:, lado, 0], 5)
    xp = PO._suave(X[:, 0], 5)
    m = np.isfinite(xm) & np.isfinite(xp) & res["_r"]["mask"]
    signo = float(np.sign(np.corrcoef(xm[m], xp[m])[0, 1])) or 1.0
    pasos, medios = agrupa(zs, hacia * signo)
    det = [e for e in ev if e["tipo"] in ("marcha", "romper") and e.get("lado") == lado]
    aciertos = 0
    for z in zs:
        if any(not (e["fin"] + TOL_CUADROS < z["inicio"] or
                    e["inicio"] - TOL_CUADROS > z["fin"]) for e in det):
            aciertos += 1
    nfondos = sum(1 for e in ev if e["tipo"] == "fondo" and e.get("lado") == lado)
    ks = np.nonzero(res["_r"]["mask"])[0]
    util = (ks[-1] - ks[0] + 1) / fps
    print("\n=== %s  lado %d  ===" % (clip, lado))
    print("  tramo utilizable: c%d-%d, %.1f s utiles | umbral de zancada %.1f px (%.3f del alto)"
          % (ks[0], ks[-1], util, Z, Z_FRAC))
    print("  ruido y mesetas: %s" % diag)
    print("  ZANCADAS CONTADAS: %d  (pie0 %d, pie1 %d)"
          % (len(zs), sum(1 for z in zs if z["pie"] == 0), sum(1 for z in zs if z["pie"] == 1)))
    for z in zs:
        cae = [e for e in det if not (e["fin"] + TOL_CUADROS < z["inicio"] or
                                      e["inicio"] - TOL_CUADROS > z["fin"])]
        print("    pie%d c%3d-%3d  %+6.1f px  %5.2f s  ->  %s"
              % (z["pie"], z["inicio"], z["fin"], z["px"], (z["fin"] - z["inicio"]) / fps,
                 ("DETECTADO: " + ", ".join("%s c%d-%d %.2f m" % (e["tipo"], e["inicio"], e["fin"],
                                                                  e["magnitud_m"]) for e in cae))
                 if cae else "PERDIDO"))
    print("  agrupadas con el convenio: %d pasos (%d marcha, %d romper) + %d medios pasos"
          % (len(pasos), sum(1 for p in pasos if p["tipo"] == "marcha"),
             sum(1 for p in pasos if p["tipo"] == "romper"), len(medios)))
    print("  DETECTADOS por el detector: %d (%d marcha, %d romper) | fondos %d"
          % (len(det), sum(1 for e in det if e["tipo"] == "marcha"),
             sum(1 for e in det if e["tipo"] == "romper"), nfondos))
    print("  RECALL de zancadas: %d de %d = %.0f %%"
          % (aciertos, len(zs), 100.0 * aciertos / max(len(zs), 1)))
    return {"clip": clip, "lado": lado, "cuadro_inicio": int(ks[0]), "cuadro_fin": int(ks[-1]),
            "s_utiles": round(util, 1), "umbral_px": round(Z, 1),
            "n_zancadas": len(zs), "n_pasos_convenio": len(pasos),
            "n_marchas_convenio": sum(1 for p in pasos if p["tipo"] == "marcha"),
            "n_romper_convenio": sum(1 for p in pasos if p["tipo"] == "romper"),
            "n_medios_pasos": len(medios), "n_fondos_det": nfondos,
            "n_detectados": len(det), "n_zancadas_cubiertas": aciertos,
            "recall_zancadas": round(aciertos / max(len(zs), 1), 3)}


if __name__ == "__main__":
    import csv
    filas = [una(c, int(l)) for c, l in (a.split(":") for a in sys.argv[1:])]
    p = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\pista"
                     r"\conteo_ojo_pasos.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    z = sum(f["n_zancadas"] for f in filas)
    cu = sum(f["n_zancadas_cubiertas"] for f in filas)
    print("\n=== AGREGADO de %d tiradores ===" % len(filas))
    print("  zancadas contadas %d, cubiertas por una deteccion %d  ->  RECALL %.0f %%"
          % (z, cu, 100.0 * cu / max(z, 1)))
    print("  pasos con el convenio: %d | medios pasos %d | detectados %d"
          % (sum(f["n_pasos_convenio"] for f in filas), sum(f["n_medios_pasos"] for f in filas),
             sum(f["n_detectados"] for f in filas)))

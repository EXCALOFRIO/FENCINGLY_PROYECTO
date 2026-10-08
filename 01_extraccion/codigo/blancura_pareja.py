"""El contraste de blancura entre los dos lados caza al ARBITRO elegido como tirador?

## LA AVERIA QUE HAY QUE CAZAR, encontrada por la tabla de checks

En `PowerfulAllLarva` el "tirador izquierdo" que entrega la tuberia **es el arbitro**: chaqueta
oscura, andando en primer plano por delante del marcador. Verificado a ojo en c160 y c224
(`02_validacion/muestras/cruce_lados/`).

Y paso las TRES puertas geometricas de `pareja_v2` sin rozarlas:

  | puerta | limite | lo que dio |
  |---|---|---|
  | separacion | 0,15 a 3,20 alturas | 1,93 |
  | dpie relativo | <= 0,55 | 0,46 |
  | pies en la banda | >= 0,40 | 0,894 |

O sea que la geometria NO puede distinguirlo, y era previsible: el arbitro camina junto a la pista,
asi que tiene los pies igual de bajos en el cuadro y una talla parecida. Hace falta un rasgo que no
sea geometrico.

El sintoma que si lo delato fue otro: los dos lados **se cruzan** en 29 de 193 cuadros, o sea que la
caja "izquierda" acaba a la derecha de la "derecha". Dos tiradores en una pista no se intercambian
las puntas; un arbitro que pasea por delante, si cruza.

## EL RASGO QUE SE PRUEBA, Y POR QUE ESTE

En esgrima la chaqueta del tirador **tiene que ser blanca**: es regla del deporte, no un umbral
ajustado a estos datos. El arbitro va de traje oscuro. Asi que la blancura del torso separa a uno de
otro por construccion.

PERO la blancura ABSOLUTA ya se midio en este proyecto y NO vale como puerta: dos tiradores reales en
un pabellon oscuro dieron 0,069 y 0,000. Lo que si vale es el **CONTRASTE entre los dos lados del
mismo cuadro**, porque los dos comparten iluminacion: si uno viste de blanco y el otro no, la
diferencia significa algo. Ese razonamiento ya esta escrito en `scratchpad/busca_intrusos.py` y aqui
se reutiliza tal cual.

La definicion de blancura NO se toca: fraccion de pixeles con `V >= 150` y `S <= 70`, que es la misma
que usa `validador_visual.blancura`. Un umbral que se mueve entre medidas no deja comparar nada.

## CRITERIO DECLARADO ANTES DE EJECUTAR

Se mide en N clips del corpus el contraste mediano de blancura entre los dos lados elegidos.

  El rasgo SIRVE como puerta si `PowerfulAllLarva`, que tiene el arbitro elegido y esta verificado a
  ojo, queda entre los mas altos de la distribucion, Y existe un corte que lo deja fuera dejando
  dentro a los clips que pasan todos los demas checks. Si el arbitro se mezcla con los buenos, el
  rasgo no vale de puerta y se dice.

Y se mide al lado el rasgo GRATIS que hay que batir: **la fraccion de cuadros compartidos en que los
dos lados se cruzan**. Si el cruce solo ya separa igual de bien, no hace falta decodificar el video
para leer pixeles, y entonces la puerta es el cruce y no la blancura.
"""
from __future__ import annotations

import os
import pathlib
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import extrae_pareja as EP

SCR = pathlib.Path(os.environ.get(
    "SCRATCH", r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\187cfd83-d8d9-4c4d-bed4-f255724f6b21\scratchpad"))
LISTA = SCR / "corpus_etiquetado.txt"
VIDEOS = "D:/FENCINGLY_VIDEOS_RAW"
PRUEBAS = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
V_CLARO, S_MAX = 150, 70
"""Misma definicion de blanco que `validador_visual.blancura`. Claro no basta: tiene que ser POCO
SATURADO, porque una camiseta amarilla tambien es clara."""
BANDA_TORSO = (0.15, 0.55)
"""Franja vertical de la caja donde se lee el torso, en fraccion del alto. Se deja fuera la careta
(que es plateada en los dos) y las piernas (que se mezclan con el suelo de la pista)."""
BANDA_ANCHO = 0.50
"""Franja horizontal centrada. Los bordes de la caja son fondo con frecuencia."""
NCUADROS = 14
"""Cuadros muestreados por clip. Con 7 no bastaba: `ClearLeafyBats` daba blancura MEDIANA 0,074 en un
lado y a ojo los dos son tiradores correctos, porque en algunos cuadros la caja pilla sombra. La
pregunta buena no es cuanto blanco tiene de mediana, sino **si ese lado llega a parecer blanco
alguna vez**, y para eso hacen falta mas cuadros."""


def blancura(im, caja, W, H):
    """QUE HACE: fraccion de pixeles claros y poco saturados en el torso de la caja."""
    if not np.isfinite(caja).all():
        return float("nan")
    x0, y0, x1, y1 = caja[0] * W, caja[1] * H, caja[2] * W, caja[3] * H
    w, h = x1 - x0, y1 - y0
    if w < 6 or h < 12:
        return float("nan")
    cx = 0.5 * (x0 + x1)
    a = int(max(0, cx - 0.5 * BANDA_ANCHO * w))
    b = int(min(W, cx + 0.5 * BANDA_ANCHO * w))
    c = int(max(0, y0 + BANDA_TORSO[0] * h))
    d = int(min(H, y0 + BANDA_TORSO[1] * h))
    if b - a < 3 or d - c < 3:
        return float("nan")
    hsv = cv2.cvtColor(im[c:d, a:b], cv2.COLOR_BGR2HSV)
    return float(np.mean((hsv[:, :, 2] >= V_CLARO) & (hsv[:, :, 1] <= S_MAX)))


def del_clip(cid, ruta):
    ca = SCR / ("det_%s.pkl" % cid)
    if not ca.exists():
        return None
    with open(ca, "rb") as fh:
        cajas, cuadros, W, H, T = pickle.load(fh)
    if T < 20 or not len(cajas):
        return None
    r = EP.procesa(cajas, cuadros, W, H, T)
    B, pres, mask = r["B"], r["pres"], r["mask"]
    amb = np.nonzero((pres.sum(axis=1) == 2) & mask)[0]
    if len(amb) < 5:
        return None
    cx0 = 0.5 * (B[amb, 0, 0] + B[amb, 0, 2])
    cx1 = 0.5 * (B[amb, 1, 0] + B[amb, 1, 2])
    cruce = float(np.mean(cx0 > cx1))
    ks = amb[np.unique(np.linspace(0, len(amb) - 1, min(NCUADROS, len(amb))).astype(int))]
    cap = cv2.VideoCapture(ruta)
    bl = {0: [], 1: []}
    for k in ks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, im = cap.read()
        if not ok:
            continue
        for lado in (0, 1):
            v = blancura(im, B[k, lado], W, H)
            if np.isfinite(v):
                bl[lado].append(v)
    cap.release()
    if len(bl[0]) < 3 or len(bl[1]) < 3:
        return None
    b0, b1 = float(np.median(bl[0])), float(np.median(bl[1]))
    """`techo` es el percentil 90 de la blancura de ese lado a lo largo del clip: **hasta que punto
    ese lado llega a parecer blanco en su mejor cuadro**. Un tirador viste de blanco o de lame, asi
    que su techo es alto aunque su mediana baje por sombras. Un arbitro de traje oscuro, o un trozo
    de espectador en el borde, no tiene ningun cuadro blanco y su techo se queda abajo."""
    t0, t1 = float(np.quantile(bl[0], 0.90)), float(np.quantile(bl[1], 0.90))
    return {"clip": cid, "T": int(T), "ambos": float(r["ambos"]), "cruce": cruce,
            "b_izq": b0, "b_der": b1, "b_min": min(b0, b1), "b_max": max(b0, b1),
            "contraste": abs(b0 - b1),
            "techo_izq": t0, "techo_der": t1,
            "techo_min": min(t0, t1), "techo_dif": abs(t0 - t1),
            "razon": max(b0, b1) / max(min(b0, b1), 1e-4),
            "motivo": r["motivo"]}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    mapa = {}
    for raiz, _d, fs in os.walk(VIDEOS):
        for f in fs:
            if f.lower().endswith(".mp4"):
                mapa.setdefault(pathlib.Path(f).stem, os.path.join(raiz, f))
    ids = [l.strip() for l in LISTA.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = np.random.default_rng(20260910)
    rng.shuffle(ids)
    """`PowerfulAllLarva` se mete a la fuerza: es el caso VERIFICADO A OJO con el arbitro elegido, y
    sin el en la muestra no hay nada contra lo que juzgar la puerta."""
    VERIF = ["PowerfulAllLarva", "LavishDistantDikkops", "ClearLeafyBats",
             "proudDungBeetleBoglenTerrier", "SimilarOffbeatIberianlynx"]
    """Los cinco casos ya VERIFICADOS A OJO, que son la unica verdad que hay aqui. Los dos primeros
    tienen intruso confirmado (arbitro y trozo de espectador en el borde). Los tres siguientes son
    CORRECTOS y estan aqui a proposito, porque una regla se juzga por los falsos positivos: el
    contraste alto de `proudDungBeetle` es un lame dorado, no un intruso."""
    orden = VERIF + [c for c in ids if c not in VERIF]
    filas = []
    for cid in orden:
        if len(filas) >= n:
            break
        if cid not in mapa:
            continue
        f = del_clip(cid, mapa[cid])
        if f is None:
            continue
        filas.append(f)
        print("%-38s cruce %5.1f%%  mediana %.3f/%.3f  techo %.3f/%.3f  techo_min %.3f" % (
            cid[:38], 100 * f["cruce"], f["b_izq"], f["b_der"],
            f["techo_izq"], f["techo_der"], f["techo_min"]))
    if not filas:
        raise SystemExit("sin datos")
    import pandas as pd
    d = pd.DataFrame(filas)
    d.to_csv(PRUEBAS / "blancura_pareja.csv", index=False)

    MALOS = {"PowerfulAllLarva": "el arbitro elegido como tirador izquierdo",
             "LavishDistantDikkops": "un trozo de espectador en el borde izquierdo"}
    BUENOS = {"ClearLeafyBats": "los dos son tiradores, blancura baja por sombra",
              "proudDungBeetleBoglenTerrier": "los dos son tiradores, contraste por el lame dorado",
              "SimilarOffbeatIberianlynx": "los dos son tiradores y se cruzan de verdad"}
    print()
    print("=" * 96)
    print("DISTRIBUCION sobre %d clips" % len(d))
    print("=" * 96)
    for col in ("contraste", "cruce", "b_min", "techo_min", "techo_dif"):
        print("  %-11s p50 %7.3f  p10 %7.3f  p90 %7.3f  min %7.3f  max %7.3f" % (
            col, d[col].median(), d[col].quantile(0.10), d[col].quantile(0.90),
            d[col].min(), d[col].max()))
    print()
    print("  LOS CINCO CASOS VERIFICADOS A OJO")
    print("  %-38s %9s %9s %9s %9s  %s" % (
        "clip", "contraste", "techo_min", "techo_dif", "cruce", "veredicto a ojo"))
    for cid, que in list(MALOS.items()) + list(BUENOS.items()):
        q = d[d["clip"] == cid]
        if not len(q):
            continue
        q = q.iloc[0]
        print("  %-38s %9.3f %9.3f %9.3f %8.1f%%  %s: %s" % (
            cid[:38], q["contraste"], q["techo_min"], q["techo_dif"], 100 * q["cruce"],
            "MAL" if cid in MALOS else "BIEN", que))
    print()
    print("  CADA REGLA CANDIDATA, juzgada por lo que caza y por lo que se lleva de mas")
    reglas = [
        ("contraste >= 0,25", lambda z: z["contraste"] >= 0.25),
        ("cruce > 2 %", lambda z: z["cruce"] > 0.02),
        ("contraste >= 0,25 Y mediana minima <= 0,20",
         lambda z: (z["contraste"] >= 0.25) & (z["b_min"] <= 0.20)),
        ("techo minimo <= 0,20 Y diferencia de techos >= 0,25",
         lambda z: (z["techo_min"] <= 0.20) & (z["techo_dif"] >= 0.25)),
        ("techo minimo <= 0,25 Y diferencia de techos >= 0,20",
         lambda z: (z["techo_min"] <= 0.25) & (z["techo_dif"] >= 0.20))]
    for et, fn in reglas:
        m = fn(d)
        marcados = list(d[m]["clip"])
        caza = [c for c in marcados if c in MALOS]
        falla = [c for c in marcados if c in BUENOS]
        print("     %-52s marca %2d de %d   caza %d de %d verificados malos   falsos "
              "positivos verificados %d" % (
                  et, int(m.sum()), len(d), len(caza), len(MALOS), len(falla)))
        if falla:
            print("        se lleva de mas: %s" % ", ".join(falla))
        otros = [c for c in marcados if c not in MALOS and c not in BUENOS]
        if otros:
            print("        sin verificar aun, hay que mirarlos: %s" % ", ".join(otros))
    print()
    print("  csv en %s" % (PRUEBAS / "blancura_pareja.csv"))
    print()
    print("  AVISO SOBRE ESTAS CIFRAS: solo hay DOS casos malos verificados a ojo, asi que cualquier")
    print("  umbral que los separe esta debilmente determinado. Lo que si esta establecido es que el")
    print("  contraste solo NO vale (el lame dorado lo dispara) y que el cruce solo NO vale (dos")
    print("  tiradores reales se cruzan al pasar uno junto al otro).")


if __name__ == "__main__":
    main()

"""Busca automaticamente clips donde la extraccion probablemente se equivoca, y los ordena.

POR QUE EXISTE. Mirar clip a clip no escala, y las averias que han aparecido hoy no se veian en
ninguna cifra agregada: el fundido encadenado de `VainAchingGonolek` daba 96,1 % de presencia, o sea
una cifra buenisima sobre poses mezcladas de dos planos distintos. Hace falta buscar los SINTOMAS de
cada averia conocida, no la nota media.

LOS AVISOS, y cada uno viene de una averia que ya se ha visto de verdad:

  `salto`        el centro de una caja se teletransporta de un cuadro al siguiente. Es el sintoma de
                 cambio de identidad, que es el fallo que corrompe la etiqueta de lado
  `solapan`      las dos cajas elegidas se solapan mucho. Puede ser un cuerpo a cuerpo legitimo, o
                 las dos cajas puestas sobre la MISMA persona
  `fundido`      la puerta de fundidos ha marcado tramo, asi que habia dos planos superpuestos
  `recorta`      se ha tenido que recortar mucho el clip, o sea que uno de los dos no estaba
  `sep`          la pareja esta al limite de la geometria admitida, arriba o abajo
  `dpie`         la diferencia de profundidad esta al limite
  `talla`        uno de los dos es mucho mas grande que el otro, que es sintoma de estar cogiendo a
                 alguien a otra distancia
  `faltan`       quedan cuadros sin uno de los dos dentro del tramo utilizable
  `pocos`        el tramo utilizable se ha quedado en poca cosa
  `blancura`     un lado no viste de blanco y el otro si, con el mismo alumbrado. En esgrima la
                 chaqueta del tirador TIENE que ser blanca, es regla del deporte, asi que un lado
                 oscuro al lado de uno blanco es sintoma de arbitro o de espectador elegido
  `borde`        una de las dos cajas vive pegada al borde lateral del cuadro casi siempre, o sea
                 que casi nunca se ha visto a esa persona entera

QUE NO HACE: decidir. Ordena por numero de avisos y guarda imagenes de los cuadros senialados para
que se puedan mirar. Un aviso no es un error hasta que se ve.

## POR QUE `blancura` ES UN AVISO Y NO UNA PUERTA, y esto esta medido

Sobre 60 clips, el criterio `contraste >= 0,25` Y `blancura minima <= 0,20` marca 4 clips (6,7 %).
De los verificados a ojo, **caza 2 errores reales que ningun otro check veia**:

  `PowerfulAllLarva`      el ARBITRO elegido como tirador izquierdo, chaqueta oscura
  `LavishDistantDikkops`  un trozo de ESPECTADOR en el borde, blancura 0,000

y se lleva de mas 2 casos CORRECTOS:

  `proudDungBeetleBoglenTerrier`  el contraste lo da el LAME dorado, que es reglamentario en florete
  `ClearLeafyBats`               blancura baja por sombra, los dos son tiradores

Con esa precision no se puede borrar dato automaticamente: tirar a un tirador bueno es peor que
revisar un clip de mas. Y tampoco vale el rasgo solo del cruce de lados: dos tiradores reales se
cruzan de verdad al pasar uno junto al otro (`SimilarOffbeatIberianlynx`, 9,3 % de los cuadros).

Lo que SI se convirtio en puerta, porque es definicional y no un umbral, es el caso extremo: una
trayectoria cortada por el borde en el **100 %** de sus cuadros nunca se vio entera, asi que se
descarta como candidata. Esta en `pareja_v2.BORDE_SIEMPRE` con su medida al lado.
"""
from __future__ import annotations

import os
import pathlib
import pickle
import sys
import time

import cv2
import numpy as np

CODIGO = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
sys.path.insert(0, str(CODIGO))
import blancura_pareja as BP
import extrae_pareja as EP
import pareja_v2 as PAR

SCR = pathlib.Path(os.environ.get(
    "SCRATCH", r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\187cfd83-d8d9-4c4d-bed4-f255724f6b21\scratchpad"))
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\sospechosos")
PRUEBAS = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
VIDEOS = "D:/FENCINGLY_VIDEOS_RAW"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
DESDE = int(sys.argv[2]) if len(sys.argv) > 2 else 0
"""`DESDE` permite coger clips NUEVOS sin repetir los ya revisados, manteniendo la misma semilla."""

SALTO_MAX = 1.20
"""Desplazamiento del centro entre cuadros consecutivos, en anchos de caja. Un fondo de esgrima es
rapido pero continuo; por encima de esto la caja ha cambiado de persona."""
SOLAPA_MAX = 0.55
TALLA_MAX = 1.75
POCOS = 80
CONTRASTE_MIN = 0.25
BLANCURA_SUELO = 0.20
"""Los dos a la vez, nunca por separado. Ver la cabecera: por separado cada uno se dispara con casos
correctos (el lame dorado, o un pabellon oscuro donde los dos tiradores dan blancura casi cero)."""
NBLANCURA = 14
BORDE_AVISO = 0.95
"""Fraccion de cuadros con la caja cortada por el borde lateral que ya merece una mirada. El 1,00 lo
descarta `pareja_v2.BORDE_SIEMPRE`; entre 0,95 y 1,00 se avisa, porque el p99 medido en 60 clips es
0,936 y un caso correcto llego a 0,892."""


def avisos_de(r, W, H, T, frames=None):
    """QUE HACE: aplica todos los sintomas conocidos y devuelve la lista de avisos con su cuadro.

    `frames` es opcional y solo lo necesita el sintoma de blancura, que es el unico que mira pixeles.
    """
    B, pres, mask = r["B"], r["pres"], r["mask"]
    av = []
    for lado, nom in ((0, "izquierda"), (1, "derecha")):
        ks = np.nonzero(pres[:, lado] & mask)[0]
        if not len(ks):
            continue
        b = B[ks, lado]
        f = float(np.mean((b[:, 0] <= 0.004) | (b[:, 2] >= 0.996)))
        if f >= BORDE_AVISO:
            av.append(("borde", int(ks[len(ks) // 2]),
                       "la caja de la %s toca el borde en el %.0f %% de los cuadros" % (nom, 100 * f)))
    if frames is not None:
        amb0 = np.nonzero((pres.sum(axis=1) == 2) & mask)[0]
        if len(amb0) >= 5:
            ks = amb0[np.unique(np.linspace(0, len(amb0) - 1, min(NBLANCURA, len(amb0))).astype(int))]
            bl = {0: [], 1: []}
            for k in ks:
                if k >= len(frames):
                    continue
                for lado in (0, 1):
                    v = BP.blancura(frames[k], B[k, lado], W, H)
                    if np.isfinite(v):
                        bl[lado].append(v)
            if len(bl[0]) >= 3 and len(bl[1]) >= 3:
                b0, b1 = float(np.median(bl[0])), float(np.median(bl[1]))
                if abs(b0 - b1) >= CONTRASTE_MIN and min(b0, b1) <= BLANCURA_SUELO:
                    av.append(("blancura", int(ks[len(ks) // 2]),
                               "blancura izq %.3f der %.3f: un lado no viste de blanco" % (b0, b1)))
    peor_salto, peor_k = 0.0, -1
    for lado in (0, 1):
        ks = np.nonzero(pres[:, lado] & mask)[0]
        for i in range(1, len(ks)):
            a, b = int(ks[i - 1]), int(ks[i])
            if b - a > 2:
                continue
            ca = 0.5 * (B[a, lado, 0] + B[a, lado, 2]) * W
            cb = 0.5 * (B[b, lado, 0] + B[b, lado, 2]) * W
            cya = 0.5 * (B[a, lado, 1] + B[a, lado, 3]) * H
            cyb = 0.5 * (B[b, lado, 1] + B[b, lado, 3]) * H
            anc = max((B[a, lado, 2] - B[a, lado, 0]) * W, 1.0)
            d = float(np.hypot(cb - ca, cyb - cya)) / anc
            if d > peor_salto:
                peor_salto, peor_k = d, b
    if peor_salto > SALTO_MAX:
        av.append(("salto", peor_k, "centro salta %.2f anchos de caja" % peor_salto))

    amb = np.nonzero((pres.sum(axis=1) == 2) & mask)[0]
    if len(amb):
        ious, tallas = [], []
        for k in amb:
            a, b = B[k, 0], B[k, 1]
            xa, ya = max(a[0], b[0]), max(a[1], b[1])
            xb, yb = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0.0, xb - xa) * max(0.0, yb - ya)
            u = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
            ious.append(inter / u if u > 0 else 0.0)
            ha, hb = a[3] - a[1], b[3] - b[1]
            tallas.append(max(ha / max(hb, 1e-6), hb / max(ha, 1e-6)))
        ious = np.array(ious)
        tallas = np.array(tallas)
        if ious.max() > SOLAPA_MAX:
            av.append(("solapan", int(amb[int(np.argmax(ious))]),
                       "las dos cajas solapan %.2f" % ious.max()))
        if float(np.median(tallas)) > TALLA_MAX:
            av.append(("talla", int(amb[len(amb) // 2]),
                       "razon de talla mediana %.2f" % np.median(tallas)))
    if r["fundidos"]:
        a, b, p, _f = r["fundidos"][0]
        av.append(("fundido", int((a + b) // 2), "tramo c%d-c%d pico %.2f" % (a, b, p)))
    if r["recortados"] > 0.25 * T:
        av.append(("recorta", int(r["ini"]), "recortados %d de %d" % (r["recortados"], T)))
    if r["faltan"] > 0:
        falta = np.nonzero((~pres).any(axis=1) & mask)[0]
        av.append(("faltan", int(falta[0]) if len(falta) else 0,
                   "%d cuadros sin uno de los dos" % r["faltan"]))
    if int(mask.sum()) < POCOS:
        av.append(("pocos", int(T // 2), "solo %d cuadros utilizables" % int(mask.sum())))
    mot = r["motivo"]
    if "sep " in mot:
        try:
            sep = float(mot.split("sep ")[1].split(" ")[0])
            if sep > 0.85 * PAR.SEP_ALTURAS[1] or sep < 1.6 * PAR.SEP_ALTURAS[0]:
                av.append(("sep", int(T // 2), "separacion %.2f alturas, al limite" % sep))
        except (IndexError, ValueError):
            pass
    if "(rel " in mot:
        try:
            dr = float(mot.split("(rel ")[1].split(")")[0])
            if dr > 0.85 * PAR.DPIE_REL_MAX:
                av.append(("dpie", int(T // 2), "dpie relativo %.2f, al limite" % dr))
        except (IndexError, ValueError):
            pass
    if "AVISO" in mot:
        av.append(("sin_pareja", int(T // 2), mot[:60]))
    return av


mapa = {}
for raiz, _d, fs in os.walk(VIDEOS):
    for f in fs:
        if f.lower().endswith(".mp4"):
            mapa.setdefault(pathlib.Path(f).stem, os.path.join(raiz, f))
"""EL MARCO DE MUESTREO ERA MALO Y ESTO LO ARREGLA. Muestrear todos los `.mp4` del arbol de video da
42.326 nombres, pero ese arbol junta las descargas de varios estudios distintos, no el corpus
etiquetado. De los 12 clips que medi asi, **solo 2 estaban en el corpus**, y en la muestra aparecio
un clip de ESPADA (`DearestThirdImago`, "Men's Epee - T64") y uno de SABLE (`FickleJovialCentipede`,
"Men's Sabre Grand Prix"). El corpus etiquetado son 10.840 nombres de los que **8.559 estan en
disco**, y esa es la lista que hay que muestrear."""
LISTA = SCR / "corpus_etiquetado.txt"
if LISTA.exists():
    corpus = [c for c in LISTA.read_text(encoding="utf-8").splitlines() if c and c in mapa]
    print("corpus etiquetado presente en disco: %d clips" % len(corpus))
else:
    corpus = sorted(mapa)
    print("AVISO: sin lista de corpus, se muestrea todo el arbol y puede colarse otra arma")
rng = np.random.default_rng(20260910)
clips = list(corpus)
rng.shuffle(clips)
clips = clips[DESDE:DESDE + N]
SALIDA.mkdir(parents=True, exist_ok=True)
print("clips %d a %d de la misma baraja  (%d disponibles)" % (DESDE, DESDE + N, len(mapa)))

from ultralytics import YOLO
yolo = YOLO(str(pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565") / "scratchpad/smoke/weights/yolo11m-pose.pt"))
yolo.to("cuda")
print()

filas = []
for n, cid in enumerate(clips, 1):
    t0 = time.perf_counter()
    ca = SCR / ("det_%s.pkl" % cid)
    cap = cv2.VideoCapture(mapa[cid])
    if int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) < 2:
        cap.release()
        continue
    if ca.exists():
        with open(ca, "rb") as fh:
            cajas, cuadros, W, H, T = pickle.load(fh)
        frames = []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            frames.append(im)
    else:
        cajas, cuadros, W, H, T, frames = EP.detecta_clip(yolo, cap, cv2)
        with open(ca, "wb") as fh:
            pickle.dump((cajas, cuadros, W, H, T), fh)
    cap.release()
    if T < 20 or not len(cajas):
        continue
    gris = [cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype("float32")
            for f in frames]
    r = EP.procesa(cajas, cuadros, W, H, T, yolo=yolo, frames=frames, gris=gris)
    av = avisos_de(r, W, H, T, frames=frames)
    filas.append({"clip": cid, "T": T, "ambos": r["ambos"], "util": int(r["mask"].sum()),
                  "avisos": ",".join(a for a, _k, _d in av) or "-", "n_avisos": len(av),
                  "detalle": " | ".join(d for _a, _k, d in av), "motivo": r["motivo"],
                  "seg": time.perf_counter() - t0})
    print("%3d/%d %-38s ambos %5.1f%%  util %4d  %s" % (
        n, len(clips), cid[:38], 100 * r["ambos"], int(r["mask"].sum()),
        ",".join(a for a, _k, _d in av) or "sin avisos"))

    for etq, k, det in av:
        if not (0 <= k < len(frames)):
            continue
        vis = frames[k].copy()
        for lado, col, nom in ((0, (60, 240, 60), "IZQ"), (1, (60, 160, 255), "DER")):
            b = r["B"][k, lado] if r["pres"][k, lado] else r["Be"][k, lado]
            if not np.isfinite(b).all():
                continue
            gr = 3 if r["pres"][k, lado] else 1
            cv2.rectangle(vis, (int(b[0] * W), int(b[1] * H)),
                          (int(b[2] * W), int(b[3] * H)), col, gr)
            cv2.putText(vis, nom if r["pres"][k, lado] else nom + " estimada",
                        (int(b[0] * W), max(14, int(b[1] * H) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        cv2.rectangle(vis, (0, 0), (W, 34), (12, 12, 18), -1)
        cv2.putText(vis, "%s c%d  AVISO %s: %s" % (cid[:22], k, etq, det[:52]),
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1)
        cv2.imwrite(str(SALIDA / ("%s_%s_c%04d.png" % (etq, cid[:24], k))),
                    cv2.resize(vis, (1100, int(1100 * H / W))))

import pandas as pd
d = pd.DataFrame(filas)
d = d.sort_values(["n_avisos", "ambos"], ascending=[False, True])
d.to_csv(PRUEBAS / ("sospechosos_%d_%d.csv" % (DESDE, DESDE + N)), index=False)
print()
print("=" * 92)
print("RESUMEN sobre %d clips" % len(d))
print("=" * 92)
print("  los dos presentes: mediana %5.1f %%   p10 %5.1f %%   min %5.1f %%" % (
    100 * d["ambos"].median(), 100 * d["ambos"].quantile(0.1), 100 * d["ambos"].min()))
print("  clips sin ningun aviso: %d de %d (%.0f %%)" % (
    int((d["n_avisos"] == 0).sum()), len(d), 100.0 * (d["n_avisos"] == 0).mean()))
print()
cuenta = {}
for v in d["avisos"]:
    for a in v.split(","):
        if a != "-":
            cuenta[a] = cuenta.get(a, 0) + 1
print("  AVISOS por tipo, de mas a menos frecuente")
for a, c in sorted(cuenta.items(), key=lambda z: -z[1]):
    print("    %-12s %3d clips  (%.0f %%)" % (a, c, 100.0 * c / len(d)))
print()
print("  LOS MAS SOSPECHOSOS")
for _, q in d.head(12).iterrows():
    print("    %-38s ambos %5.1f%%  %s" % (q["clip"][:38], 100 * q["ambos"], q["detalle"][:70]))
print()
print("  imagenes en %s" % SALIDA)

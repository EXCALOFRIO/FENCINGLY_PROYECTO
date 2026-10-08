"""Cuanto se degrada una pose inventada segun lo LARGO que sea el hueco que rellena?

POR QUE HACE FALTA ESTO. La validacion que va dentro de `video_revision` esconde UN cuadro conocido
entre sus dos vecinos, asi que mide la interpolacion sobre un hueco de un cuadro. Pero los huecos
que de verdad se rellenan son mas largos, y dar el numero del hueco de 1 como si valiera para todos
seria decir que algo esta medido cuando no lo esta.

QUE SE HACE: se tapan N cuadros CONSECUTIVOS cuya pose esta medida, se interpola entre los dos
extremos conocidos y se compara con la verdad, para N de 1 a 24. Y se compara con la otra via, la de
SOSTENER la ultima pose conocida, que es lo que se hace en los huecos de borde donde no hay dos
anclas. Sin esa comparacion no se sabe si interpolar aporta algo sobre no hacer nada.

Se excluye la cara (indices 63 en adelante): son 245 de los 308 puntos, no se usan en el proyecto, y
promediarlos taparia el error de lo que si importa.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

PRUEBAS = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
CONF = 0.3
CARA_DESDE = 63
LARGOS = [1, 2, 3, 5, 8, 12, 16, 24]


def error(KP, SC, pres, mask, T, n, sostener=False):
    """QUE HACE: error mediano de rellenar un hueco de `n` cuadros, en px y en fraccion de torso."""
    px, rel = [], []
    for lado in (0, 1):
        hay = np.nonzero(pres[:, lado] & mask)[0]
        if len(hay) < n + 2:
            continue
        for i in range(0, len(hay) - n - 1, max(n, 1)):
            a = int(hay[i])
            b = int(hay[i + n + 1])
            if b - a != n + 1:
                """Se exige que los n cuadros tapados sean consecutivos y medidos, para que la
                verdad exista en todos ellos."""
                continue
            for j in range(1, n + 1):
                k = a + j
                if not (pres[k, lado] and mask[k]):
                    continue
                if sostener:
                    est = KP[a, lado]
                else:
                    t = j / float(n + 1)
                    est = (1 - t) * KP[a, lado] + t * KP[b, lado]
                ok = ((SC[k, lado] >= CONF) & np.isfinite(KP[k, lado, :, 0])
                      & np.isfinite(est[:, 0]))
                ok[CARA_DESDE:] = False
                if ok.sum() < 8:
                    continue
                d = float(np.median(np.linalg.norm(est[ok] - KP[k, lado][ok], axis=1)))
                px.append(d)
                if SC[k, lado, 5] >= CONF and SC[k, lado, 9] >= CONF:
                    torso = float(np.linalg.norm(KP[k, lado, 5] - KP[k, lado, 9]))
                    if torso > 1:
                        rel.append(d / torso)
    if not px:
        return None
    return {"n": len(px), "px": float(np.median(px)), "px90": float(np.percentile(px, 90)),
            "rel": float(np.median(rel)) if rel else float("nan")}


clips = sys.argv[1:] or [p.name[6:-4] for p in PRUEBAS.glob("poses_*.npz")]
for cid in clips:
    f = PRUEBAS / ("poses_%s.npz" % cid)
    if not f.exists():
        print("falta %s" % f.name)
        continue
    z = np.load(f, allow_pickle=True)
    KP, SC, pres, mask = z["KP"], z["SC"], z["pres"], z["mask"]
    T = int(z["T"])
    print()
    print("%s   %d cuadros utilizables" % (cid, int(mask.sum())))
    print("%6s %8s %9s %9s %10s %9s   %s" % (
        "hueco", "casos", "interp px", "% torso", "interp p90", "sostener", "veredicto"))
    print("-" * 82)
    for n in LARGOS:
        a = error(KP, SC, pres, mask, T, n, sostener=False)
        b = error(KP, SC, pres, mask, T, n, sostener=True)
        if a is None:
            continue
        gana = (b is not None and a["px"] < b["px"])
        print("%6d %8d %9.2f %8.1f%% %10.2f %9s   %s" % (
            n, a["n"], a["px"], 100 * a["rel"], a["px90"],
            ("%.2f" % b["px"]) if b else "-",
            "interpolar gana" if gana else "sostener iguala o gana"))

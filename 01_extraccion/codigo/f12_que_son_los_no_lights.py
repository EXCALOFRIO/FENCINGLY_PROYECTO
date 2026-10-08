"""Que SON en realidad los clips que el detector viejo pierde, mirados y contados.

POR QUE HAY QUE MIRARLOS ANTES DE ARREGLAR NADA. El plan era arreglar la familia C porque
`tokyofoil2022` pierde el 36,78 % de sus clips. Pero `f11_recupera_no_lights.py` midio que mi
detector recupera el **44,0 %** de los clips perdidos de la familia A -- el CONTROL -- y solo el
**28,0 %** de los de `tokyofoil2022`. O sea que recupera IGUAL o MENOS en la sede supuestamente rota
que en las sanas. Eso es incompatible con "el problema es el diseno del marcador de esa familia" y
compatible con "el problema es otra cosa que le pasa a esos clips".

Antes de tocar un umbral hay que saber que son. Este script:

  1. mira si el ROTULO ESTA en el clip, midiendo que fraccion de los ultimos 2 segundos tiene un
     grafico estatico en la banda inferior;
  2. mide si hay CORTE DE PLANO cerca del final;
  3. y saca un mosaico con la banda del marcador de 8 de ellos para poder MIRARLOS.

COMO SE DETECTA "EL ROTULO ESTA", declarado antes de medir. El rotulo es un grafico sobrepuesto
QUIETO: en la banda `y` de 0,84 a 1,00, la fraccion de pixeles cuya desviacion tipica temporal del
gris es < 4,0 niveles a lo largo de una ventana de 2 s. Si el rotulo esta, esa fraccion es alta
aunque la camara se mueva, porque el rotulo no se mueve con ella. Si el clip es una repeticion sin
rotulo, o un plano de la grada, es baja.

  se declara ROTULO PRESENTE si esa fraccion es >= 0,35

El 0,35 sale de la geometria, no del ajuste: el rotulo FIE ocupa del orden del 40-55 % de la banda
0,84-1,00 en los clips de la muestra A, y se deja margen a la mitad de eso.

CRITERIO QUE DECIDE EL DIAGNOSTICO. Si en los clips perdidos de `tokyofoil2022` el rotulo esta
presente en >= 80 % de los casos, el fallo es del detector y el arreglo dirigido tiene sentido. Si
esta presente en < 50 %, el fallo NO es del detector: son clips sin rotulo que mirar, y arreglar la
familia C no recuperaria nada.
"""
from __future__ import annotations

import os
import random
import sys

import cv2
import numpy as np
import pandas as pd

import f4_censo_visual as F4
import f11_recupera_no_lights as F11

SALIDA = os.path.join(F4.PROY, r"02_validacion\muestras\luces\figuras")
BANDA = (0.84, 1.00)
STD_QUIETO = 4.0
FRAC_ROTULO = 0.35
VENTANA_S = 2.0
N_MUESTRAS = 12
SEMILLA = 20260913


def mide(ruta: str) -> dict:
    """Fraccion de banda quieta en los ultimos 2 s, y si hay corte de plano en esa ventana."""
    cap = cv2.VideoCapture(ruta)
    if not cap.isOpened():
        return {}
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    k1 = n - 1
    k0 = max(0, k1 - int(round(VENTANA_S * fps)))
    pila = []
    for k in range(k0, k1 + 1, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, img = cap.read()
        if not ok:
            continue
        H, W = img.shape[:2]
        g = cv2.cvtColor(cv2.resize(img, (320, int(round(320 * H / W)))), cv2.COLOR_BGR2GRAY)
        pila.append(g[int(BANDA[0] * g.shape[0]):int(BANDA[1] * g.shape[0])].astype(np.float32))
    cap.release()
    if len(pila) < 4:
        return {}
    P = np.stack(pila, 0)
    quieta = float((P.std(axis=0) < STD_QUIETO).mean())
    difs = [float(np.abs(P[i] - P[i - 1]).mean()) for i in range(1, len(P))]
    return dict(n=n, fps=round(fps, 3), frac_quieta=round(quieta, 4),
                rotulo=quieta >= FRAC_ROTULO, dif_max=round(max(difs), 1),
                corte=max(difs) > 25.0)


def main() -> int:
    os.makedirs(SALIDA, exist_ok=True)
    nl = F11.lee_no_lights()
    mapa = F11.indice_mp4()
    rnd = random.Random(SEMILLA)
    grupos = {
        "tokyofoil2022": nl.get("tokyofoil2022", []),
        "parisfoil2022": nl.get("parisfoil2022", []),
        "stpetersburgfoil2019": nl.get("stpetersburgfoil2019", []),
        "CONTROL_familia_A": [c for s in F11.FAMILIA_A for c in nl.get(s, [])],
    }
    filas = []
    for etiqueta, clips in grupos.items():
        clips = [c for c in clips if c in mapa]
        elegidos = rnd.sample(clips, min(N_MUESTRAS, len(clips)))
        for clip in elegidos:
            m = mide(mapa[clip])
            if m:
                filas.append(dict(grupo=etiqueta, clip=clip, **m))
    d = pd.DataFrame(filas)
    pd.set_option("display.width", 200)
    print("--- el rotulo ESTA en los clips que el viejo perdio? ---")
    g = d.groupby("grupo")
    tab = pd.DataFrame({
        "clips": g.size(),
        "frac_quieta_p50": g["frac_quieta"].median().round(3),
        "pct_con_rotulo": (100.0 * g["rotulo"].mean()).round(1),
        "pct_con_corte": (100.0 * g["corte"].mean()).round(1),
    })
    print(tab.to_string())
    print("\n--- por clip ---")
    print(d.sort_values(["grupo", "frac_quieta"]).to_string(index=False))
    print("\ncriterio: >= 80 % con rotulo -> el fallo es del detector;  < 50 % -> son clips sin rotulo")
    for etq in grupos:
        s = d[d["grupo"] == etq]
        if not len(s):
            continue
        p = 100.0 * s["rotulo"].mean()
        ver = ("FALLO DEL DETECTOR" if p >= 80 else
               ("CLIPS SIN ROTULO" if p < 50 else "MEZCLA, no concluyente"))
        print(f"  {etq:24s} {p:5.1f} % con rotulo -> {ver}")

    mosaico = []
    for etq in ("tokyofoil2022", "CONTROL_familia_A"):
        for clip in d[d["grupo"] == etq]["clip"].tolist()[:6]:
            cap = cv2.VideoCapture(mapa[clip])
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 8))
            ok, img = cap.read()
            cap.release()
            if not ok:
                continue
            H, W = img.shape[:2]
            tira = cv2.resize(img[int(0.82 * H):H], (900, int(round(0.18 * H * 900 / W))))
            fq = float(d[d["clip"] == clip]["frac_quieta"].iloc[0])
            cv2.putText(tira, f"{etq} {clip[:26]} quieta={fq:.2f}", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            mosaico += [tira, np.full((2, 900, 3), 150, np.uint8)]
    if mosaico:
        p = os.path.join(SALIDA, "no_lights_que_son.png")
        cv2.imwrite(p, np.vstack(mosaico))
        print(f"\nmosaico: {p}")
    d.to_csv(os.path.join(F4.PROY, r"02_validacion\muestras\luces\no_lights_que_son.csv"),
             index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

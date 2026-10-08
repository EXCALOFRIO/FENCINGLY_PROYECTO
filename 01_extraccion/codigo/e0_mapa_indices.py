"""ETAPA E, paso 0: EL MAPA DE INDICES DE SAPIENS 2, MEDIDO. No copiado de ningun resumen.

POR QUE EXISTE ESTE FICHERO: el plan daba estos indices como verificados

    right_wrist = 10   right_hand_root = 112   right_thumb1..4 = 113..116
    left_wrist  =  9   left_hand_root  =  91   left_thumb1..4  =  92..95

y **son falsos**. Con ellos, la cadena del pulgar mide 1,4-2,8 px de largo y la distancia
`hand_root`-`wrist` sale de 97 a 212 px, o sea el 47 % del alto de la caja. Ninguna mano humana tiene
esa forma. Pintados sobre el fotograma, los puntos 91-95 y 112-116 caen **sobre la careta**.

De donde sale el mapa de verdad: del propio `config.json` del modelo. Su `flip_pairs` contiene
`[21,42], [22,43], ... [41,62]`, o sea **21 pares con desplazamiento fijo de 21**: eso es una mano
de 21 puntos contra la otra. Y medido sobre el clip, esos indices caen encima de las manos.

QUE COMPRUEBA ESTE FICHERO, cada cosa con su medida:
  1. constancia de hueso: un par de indices que sea un hueso real tiene longitud casi constante
  2. anchura de hombros contra anchura de caderas, para saber si 9 y 10 son munecas o caderas
  3. a que codo se engancha cada bloque de mano
  4. cual de los cinco grupos de cuatro dedos es el PULGAR, por separacion angular de los otros
     cuatro (los cuatro dedos van casi paralelos; el pulgar sale de lado). Criterio declarado antes
     de medir: gana el grupo cuya direccion desde el root esta mas lejos de las otras cuatro.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

PRUEBAS = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
CLIPS = ["SentimentalLargeBedlingtonterrier", "AdorableFittingAsiansmallclawedotter"]
MANO_A, MANO_B = 21, 42
CONF_MIN = 0.3


def ok(sc, idx):
    return (sc[:, idx] >= CONF_MIN).all(axis=1)


def largo(kp, a, b):
    return np.linalg.norm(kp[:, b] - kp[:, a], axis=1)


res = {}
for clip in CLIPS:
    with np.load(PRUEBAS / ("cono_poses_%s.npz" % clip), allow_pickle=True) as z:
        kp, sc, B = z["kp"], z["sc"], z["B"]
        W, H = int(z["W"]), int(z["H"])
    print("=" * 96)
    print(clip)
    print("=" * 96)
    for lado in (0, 1):
        k, s = kp[:, lado], sc[:, lado]
        alto = np.nanmedian((B[:, lado, 3] - B[:, lado, 1]) * H)
        print("  --- lado %d   alto de caja %.0f px" % (lado, alto))

        print("    CONSTANCIA DE HUESO (un hueso real tiene coef. de variacion bajo)")
        pares = [("hombro-codo I  5-7", 5, 7), ("hombro-codo D  6-8", 6, 8),
                 ("codo-mano21    8-21", 8, MANO_A), ("codo-mano42    7-42", 7, MANO_B),
                 ("codo-mano21    7-21", 7, MANO_A), ("codo-mano42    8-42", 8, MANO_B),
                 ("cadera-rodilla 9-11", 9, 11), ("cadera-rodilla 10-12", 10, 12),
                 ("rodilla-tobillo 11-13", 11, 13), ("rodilla-tobillo 12-14", 12, 14),
                 ("PLAN: 9-91", 9, 91), ("PLAN: 10-112", 10, 112)]
        for nom, a, b in pares:
            o = ok(s, [a, b])
            if o.sum() < 10:
                continue
            d = largo(k, a, b)[o]
            print("      %-22s mediana %7.1f px = %5.1f %% alto   cv %.3f" % (
                nom, np.median(d), 100 * np.median(d) / alto, np.std(d) / max(np.mean(d), 1e-6)))

        for nom, a, b in (("anchura hombros 5-6", 5, 6), ("anchura 9-10", 9, 10)):
            o = ok(s, [a, b])
            if o.sum() < 10:
                continue
            d = largo(k, a, b)[o]
            print("      %-22s mediana %7.1f px = %5.1f %% alto" % (
                nom, np.median(d), 100 * np.median(d) / alto))

        print("    QUE GRUPO DE CUATRO ES EL PULGAR (separacion angular de los otros cuatro)")
        for base in (MANO_A, MANO_B):
            grupos = [list(range(base + 1 + 4 * g, base + 5 + 4 * g)) for g in range(5)]
            o = ok(s, list(range(base, base + 21)))
            if o.sum() < 10:
                continue
            dirs = np.full((int(o.sum()), 5, 2), np.nan)
            span = np.zeros((int(o.sum()), 5))
            kk = k[o]
            for g, idx in enumerate(grupos):
                c = kk[:, idx].mean(axis=1) - kk[:, base]
                n = np.linalg.norm(c, axis=1, keepdims=True)
                dirs[:, g] = c / np.maximum(n, 1e-6)
                span[:, g] = n[:, 0]
            sep = np.zeros((len(kk), 5))
            for g in range(5):
                a = np.degrees(np.arccos(np.clip(
                    (dirs[:, g][:, None, :] * dirs).sum(axis=2), -1, 1)))
                a[:, g] = np.nan
                sep[:, g] = np.nanmedian(a, axis=1)
            gana = np.bincount(np.argmax(sep, axis=1), minlength=5)
            print("      mano %d: separacion angular mediana por grupo %s" % (
                base, np.round(np.median(sep, axis=0), 1)))
            print("               |root-centroide| mediana por grupo   %s px" % (
                np.round(np.median(span, axis=0), 1)))
            print("               grupo mas separado gana en %s de %d cuadros  (g0 = %d-%d)" % (
                gana, len(kk), base + 1, base + 4))
    print()

print("=" * 96)
print("MAPA QUE SE DEDUCE, y que usa el resto de la etapa E")
print("""
   0 nariz   1-2 ojos   3-4 orejas   5-6 hombros   7-8 codos   9-10 caderas
  11-12 rodillas   13-14 tobillos   15-17 pie A   18-20 pie B
  21-41 MANO A  (root 21, grupos de 4 desde 22)
  42-62 MANO B  (root 42, grupos de 4 desde 43)
  63 en adelante: cara y resto, no se usa aqui

  NO HAY MUNECA SEPARADA en 0-14: el bloque de cuerpo es COCO-17 SIN las dos munecas, porque la
  muneca ES el root de la mano (21 y 42). Por eso la comparacion que pedia la tarea, `wrist ->
  hand_root`, NO EXISTE como segmento, y su sustituto legitimo es `codo -> root`.
""")

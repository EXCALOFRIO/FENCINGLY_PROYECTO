"""Los fotogramas etiquetados con la zona de la pista: el conjunto para el modelo directo.

## POR QUE UN MODELO DIRECTO SOBRE EL FOTOGRAMA

Toda la via geometrica -- homografia, bordes, marcas, ancla -- da hoy una pendiente de **0,100**
contra la etiqueta humana cuando la ideal es 1,0, y un acierto de 0,323 contra el **0,430** de
contestar siempre `middle`. Y esta acotada por arriba: la geometria del suelo es sana en el **21 %**
de los clips.

La via directa no necesita nada de eso. Un humano mira el fotograma y dice en que parte de la pista
ocurre, y las pistas que usa son pocas y concretas: **que extremo de la pista se ve, donde cae la
zona de aviso de t.19, la perspectiva, y donde esta el marcador**. Comprobado a mano sobre tres
laminas ciegas: acerte 2 de 3, y el fallo fue una zona ADYACENTE en un caso de frontera que yo mismo
marque como dudoso antes de ver la respuesta.

## EL CONJUNTO, medido

    clips con etiqueta humana de zona                 5.544
    ... y ademas con el video en disco                **4.889**
    torneos distintos                                   135   el mayor 14,4 %, los 5 mayores 49,2 %
    armas                            florete 3.741, espada 749, sable 399

    reparto      middle 40,9 %   fotl_half 20,2 %   fotr_half 18,9 %
                 fotl_warning_box 12,8 %   fotr_warning_box 7,2 %

## TRES DECISIONES, declaradas antes de entrenar

**1. Fotogramas en 16:9 y no cuadrados.** La pista mide 14 m de largo y 1,5 de ancho, y la pregunta
es **donde a lo largo**. Aplastar el cuadro a 224x224 destruye justo el eje que importa. Se guarda
a 320x180, que conserva la proporcion y el reparto horizontal.

**2. Varios instantes por clip.** La etiqueta es del clip entero, asi que cada fotograma hereda la
etiqueta y al predecir se promedia por clip. Se toman cinco instantes repartidos: un clip donde los
tiradores cruzan media pista tiene mas de una respuesta posible, y promediar es el convenio que
mejor casa con "donde ocurrio la accion", que es un juicio sobre el clip.

**3. La particion es POR TORNEO.** Los clips del mismo torneo comparten pabellon, pista y camara:
partir por clip filtraria el fondo del reservado al entrenamiento y daria un numero inflado. El
proyecto ya tiene un caso documentado de comprar +3,08 pp sin informacion por evaluar dentro de un
subgrupo definido por el resultado.

## LO QUE ESTE FICHERO NO HACE

No entrena. Solo extrae y deja el indice escrito, para que el entrenamiento sea reproducible y no
dependa de volver a decodificar 4.889 videos.

    python zona_datos.py --n 4889
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import catalogo as CA  # noqa: E402
import rutas as RU  # noqa: E402

RAIZ = pathlib.Path("D:/FENCINGLY_ZONA")
IMG = RAIZ / "frames"
INDICE = RAIZ / "indice.csv"

ANCHO, ALTO = 320, 180
INSTANTES = (0.12, 0.31, 0.50, 0.69, 0.88)
"""Cinco instantes repartidos, sin llegar a los extremos: el primer y el ultimo cuadro de un clip de
`clips.fencingdatabase.com` suelen traer fundido o el marcador a pantalla completa."""


def extrae(clip, ruta_video, zona, torneo, arma):
    """QUE DEVUELVE: lista de filas `(fichero, clip, zona, torneo, arma, instante)` escritas."""
    cap = cv2.VideoCapture(ruta_video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n < 8:
        cap.release()
        return []
    filas = []
    d = IMG / clip[:2].lower()
    d.mkdir(parents=True, exist_ok=True)
    for j, t in enumerate(INSTANTES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * t))
        ok, im = cap.read()
        if not ok or im is None:
            continue
        im = cv2.resize(im, (ANCHO, ALTO), interpolation=cv2.INTER_AREA)
        p = d / ("%s_%d.jpg" % (clip, j))
        cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        filas.append((str(p.relative_to(RAIZ)).replace("\\", "/"), clip, zona, torneo, arma, t))
    cap.release()
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    a = ap.parse_args()
    df = CA.manifiesto().set_index("clip_id")
    idx = RU.indice_videos()
    et = df[df["strip_location"].isin(CA.ZONAS)]
    clips = [c for c in et.index if c in idx][:a.n]
    hechos = set()
    if INDICE.is_file():
        with open(INDICE, encoding="utf-8", newline="") as fh:
            for f in csv.DictReader(fh):
                hechos.add(f["clip"])
    """SE REANUDA POR EL DISCO, no por un contador: decodificar 4.889 videos son horas y el proceso
    puede morir. Lo que ya esta escrito no se repite."""
    nuevo = not INDICE.is_file()
    INDICE.parent.mkdir(parents=True, exist_ok=True)
    fh = open(INDICE, "a", encoding="utf-8", newline="")
    w = csv.writer(fh)
    if nuevo:
        w.writerow(["fichero", "clip", "zona", "torneo", "arma", "instante"])
    n_ok = n_fallo = 0
    for i, c in enumerate(clips):
        if c in hechos:
            continue
        try:
            filas = extrae(c, idx[c], et.loc[c, "strip_location"],
                           str(et.loc[c, "torneo"]), str(et.loc[c, "weapon"]))
        except Exception:
            filas = []
        if filas:
            w.writerows(filas)
            n_ok += 1
        else:
            n_fallo += 1
        if (i + 1) % 100 == 0:
            fh.flush()
            print("  %d/%d  ok %d  sin fotogramas %d" % (i + 1, len(clips), n_ok, n_fallo),
                  flush=True)
    fh.close()
    print()
    print("clips con fotogramas extraidos: %d   sin: %d" % (n_ok, n_fallo))
    print("indice en %s" % INDICE)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Los clips de FLORETE dentro de `champs` (13.260 clips de los campeonatos 2025-26 sin arma en el
catalogo; `catalogo.py`, "EL DEFECTO DE LA COLUMNA `weapon`").

LA REGLA (2026-10-13, decidida antes de usar la lista): un clip de `champs` es de florete si **los dos
tiradores** aparecen en clips de florete del catalogo y **ninguno** en clips de espada o de sable. No
es una medida del arma: es una inferencia por nombres, y por eso la lista va aparte y el catalogo no
se toca (`weapon` sigue diciendo `champs`).

LA COMPROBACION, sin mirar ningun video: la tasa de `touch = double` (en espada puntuan los dos):
florete del catalogo 0,018, espada 0,135, sable 0,021. Los de esta lista: 0,015; los que la misma
regla da por espada: 0,191. Distingue la espada; florete y sable se separan solo por los nombres
(nadie sale en los dos).

    python lista_campeonatos_florete.py --salida ../03_datos/catalogo/champs_florete_por_nombres.txt
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "codigo"))
import catalogo as CA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    df = CA.manifiesto()
    nom = {w: set(df.loc[df.weapon == w, "fotl_name"]) | set(df.loc[df.weapon == w, "fotr_name"])
           for w in ("foil", "epee", "sabre")}
    ch = df[df.weapon == "champs"]
    otros = nom["epee"] | nom["sabre"]
    ok = ch[[a in nom["foil"] and b in nom["foil"] and a not in otros and b not in otros
             for a, b in zip(ch.fotl_name, ch.fotr_name)]]
    ids = sorted(ok["clip_id"])
    pathlib.Path(a.salida).write_text("\n".join(ids) + "\n", encoding="utf-8")
    print("champs %d clips; de florete por los dos nombres %d (touch double %.3f); por torneo %s -> %s"
          % (len(ch), len(ids), float((ok.touch == "double").mean()),
             dict(ok["torneo"].value_counts()), a.salida))


if __name__ == "__main__":
    main()

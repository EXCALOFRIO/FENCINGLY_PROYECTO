"""La lista de clips con las DOS luces de color, para procesarlos primero en la corrida de florete.

POR QUE (2026-10-13, `CORRIDA_FINAL_2026-10-13.md`). La decision del arbitro (`prioridad`, la pregunta
2 de `VJEPA2_2026-10-09.md`) se mide con los clips de florete con las dos luces y `touch` izq/der:
son ~20 % del florete. Si salen primero, la medida con ~21.000 decisiones llega a los ~10 dias de
corrida y no al final. La etapa 9 (luces) no depende de ninguna otra (`lotes.DEPENDE[9] = ()`), asi
que se puede pasar antes que las demas.

La regla es la MISMA que `sonda_vjepa.filas_prioridad`: canales `roja` y `verde`, lados {L, R}, y
`touch` en {left, right}. Escribe un id por linea, que `lotes.conjunto` acepta como conjunto.

    python lista_dos_luces.py --conjunto foil --salida ../03_datos/catalogo/foil_dos_luces.txt
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "codigo"))
import catalogo as CA  # noqa: E402
import lotes as LO  # noqa: E402
import rutas as RU  # noqa: E402


def lados_de_color(clip):
    """QUE DEVUELVE: el conjunto de lados con luz de color, o None si no hay `lights_<clip>.csv`."""
    p = RU.TRABAJO / ("lights_%s.csv" % clip)
    if not p.is_file() or p.stat().st_size == 0:
        return None
    with open(p, encoding="utf-8") as fh:
        return {r.get("lado") for r in csv.DictReader(fh) if r.get("canal") in ("roja", "verde")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", default="foil")
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    df = CA.manifiesto().set_index("clip_id")
    ids = LO.conjunto(a.conjunto)
    sin_luces, dos, otros = 0, [], 0
    for c in ids:
        lados = lados_de_color(c)
        if lados is None:
            sin_luces += 1
        elif lados == {"L", "R"} and c in df.index and df.loc[c, "touch"] in ("left", "right"):
            dos.append(c)
        else:
            otros += 1
    pathlib.Path(a.salida).write_text("\n".join(dos) + "\n", encoding="utf-8")
    print("conjunto %s: %d clips; sin lights_ todavia %d; con las dos luces y touch izq/der %d; "
          "el resto %d -> %s" % (a.conjunto, len(ids), sin_luces, len(dos), otros, a.salida))


if __name__ == "__main__":
    main()

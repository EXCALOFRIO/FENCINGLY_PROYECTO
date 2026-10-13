"""Pone la puerta fisica POR EVENTO en las series YA ESCRITAS, sin volver a correr la etapa 7.

POR QUE SE PUEDE: la puerta (`posicion.aplica_puerta_evento`) solo mira columnas que ya estan en
`serie_eventos_<clip>.csv` (`velocidad_m_s`, `x_inicio_m`, `x_fin_m`, `duracion_s`) y la cadencia
de `posicion_clip_<clip>.csv`, y la etapa 7 recalculada da esas columnas IDENTICAS a las de disco
(medido en 222 clips, `ETAPA7_PUERTAS_2026-10-09.md` §5). Asi que esto da lo mismo que recalcular
la etapa 7 con `FENCINGLY_PUERTA_EVENTO=1` -- salvo el orden de columnas, que se copia del de
`grafica_pista` -- en segundos en vez de ~13 s de CPU por clip.

QUE ESCRIBE, por clip: la serie con `publicable_clip`, `fisica_evento`, `motivo_no_publicable` y
`publicable` rebajado donde falla, y `n_eventos_fisica_mal` en `posicion_clip_`. Escritura atomica
(`.part` + `os.replace`). Una serie que ya trae `fisica_evento` se salta: correrlo dos veces no
hace nada.

AVISO: cambia la fecha de modificacion de esos CSV, asi que `respalda.py` volvera a subir esos clips
en su siguiente vuelta (sube TODOS los ficheros del clip, poses incluidas).

    python aplica_puerta_evento.py [--conjunto anotados]            dice que haria, no escribe
    python aplica_puerta_evento.py [--conjunto anotados] --escribe  escribe en 02_validacion/pista
    python aplica_puerta_evento.py --escribe --destino D:/prueba    escribe la copia en otro sitio
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "codigo"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "02_validacion" / "herramienta"))
import lotes as LO  # noqa: E402
import posicion as PO  # noqa: E402
import rutas as RU  # noqa: E402


def _escribe(p, filas, orden):
    todas = {k for f in filas for k in f}
    cols = list(orden) + sorted(todas - set(orden))
    tmp = p.with_name(p.name + ".part")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", default="anotados")
    ap.add_argument("--escribe", action="store_true")
    ap.add_argument("--destino", default=None, help="escribir en otra carpeta (para verificar)")
    ap.add_argument("--clips", default=None, help="fichero con los clips (por defecto, el conjunto)")
    a = ap.parse_args()
    dest = pathlib.Path(a.destino) if a.destino else RU.PISTA
    dest.mkdir(parents=True, exist_ok=True)
    ids = ([l.strip() for l in open(a.clips, encoding="utf-8") if l.strip()] if a.clips
           else LO.conjunto(a.conjunto))
    import grafica_pista as GP
    n = ya = rebajadas = clips_mal = 0
    for c in ids:
        ps = RU.PISTA / ("serie_eventos_%s.csv" % c)
        pp = RU.PISTA / ("posicion_clip_%s.csv" % c)
        if not ps.is_file() or ps.stat().st_size == 0 or not pp.is_file():
            continue
        with open(ps, encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            if "fisica_evento" in (rd.fieldnames or []):
                ya += 1
                continue
            filas = list(rd)
        with open(pp, encoding="utf-8") as fh:
            pos = list(csv.DictReader(fh))
        try:
            fps = float(pos[0]["fps"])
        except (IndexError, KeyError, TypeError, ValueError):
            fps = 30.0
        mal = 0
        for f in filas:
            antes = f.get("publicable")
            PO.aplica_puerta_evento(f, fps)
            f["publicable_clip"] = antes
            if f["fisica_evento"] is False:
                mal += 1
                if antes == "True":
                    rebajadas += 1
        n += 1
        clips_mal += int(mal > 0)
        if a.escribe:
            for p in pos:
                p["n_eventos_fisica_mal"] = mal
            _escribe(dest / ps.name, filas, GP.COLUMNAS_SERIE)
            _escribe(dest / pp.name, pos, GP.COLUMNAS_POS)
    print("%s: %d series %s, %d ya la tenian; filas publicables rebajadas %d, clips con alguna "
          "fila mal %d" % (a.conjunto, n, "ESCRITAS" if a.escribe else "por escribir (sin --escribe)",
                           ya, rebajadas, clips_mal))
    return 0


if __name__ == "__main__":
    sys.exit(main())

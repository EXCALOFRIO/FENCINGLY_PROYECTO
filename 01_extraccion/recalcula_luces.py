"""Rehace la etapa 9 (luces) y, donde haga falta, la 11 (zona) de clips YA extraidos, con el codigo de hoy.

POR QUE EXISTE (2026-10-09, `04_documentacion/informes/LUCES_POR_FUENTE_2026-10-09.md`). La etapa 9
cambio de detector y `lotes.hechas` da una etapa por hecha si su fichero existe, asi que la noche NO
la rehace sola. Las dos etapas son de CPU y no necesitan GPU: solo el mp4 y `poses_<clip>.npz`.

QUE HACE, por clip:
  1. etapa 9 con `pipeline._luces` (escribe `lights_<clip>.csv`);
  2. el cuadro del tocado nuevo, leido como lo lee la etapa 11 (`zona_tocado.cuadro_tocado`);
  3. etapa 11: si el cuadro NO cambia y ya habia `zona_<clip>.csv`, la zona es la misma (la etapa 11
     solo depende del cuadro, las poses y el video) y solo se reescribe `via_luz`; si cambia, se
     rehace entera. `--etapa11 siempre` la rehace siempre.

Escribe un registro `recalcula_luces_<fecha>.csv` en `RU.TRABAJO` con el antes y el despues.

NO CORRER A LA VEZ QUE LA NOCHE SOBRE LOS MISMOS CLIPS: los dos escribirian los mismos CSV. Con el
conjunto ya terminado, o con `noche.py --para` antes.

    python recalcula_luces.py --conjunto anotados --procs 4
    python recalcula_luces.py --lista clips.txt --etapa11 siempre
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "codigo"))
import rutas as RU  # noqa: E402


def _zona_previa(clip):
    p = RU.TRABAJO / ("zona_%s.csv" % clip)
    if not p.is_file() or p.stat().st_size == 0:
        return None
    with open(p, encoding="utf-8") as fh:
        return next(csv.DictReader(fh), None)


def _entero(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def uno(clip, torneo, etapa11):
    import pipeline as PI
    import zona_tocado as ZT
    video = RU.video(clip)
    fila = {"clip": clip}
    if not video:
        fila["estado"] = "sin mp4"
        return fila
    previa = _zona_previa(clip)
    fila["tf_antes"] = _entero(previa.get("cuadro_tocado")) if previa else None
    fila["via_antes"] = previa.get("via_luz") if previa else None
    ctx = {"video": video, "clip": clip, "torneo": torneo}
    PI._luces(clip, ctx)
    tf, via = ZT.cuadro_tocado(clip, video, torneo)
    fila.update(tf_nuevo=tf, via_nueva=via)
    if not (RU.TRABAJO / ("poses_%s.npz" % clip)).is_file():
        fila["estado"] = "luces; sin poses para la 11"
        return fila
    if etapa11 == "si_cambia" and previa is not None and fila["tf_antes"] == tf:
        r = {k: previa.get(k) for k in ZT.COLUMNAS}
        r["via_luz"] = via
        ZT.escribe(r)
        fila["estado"] = "luces; 11 igual (solo via_luz)"
        return fila
    ZT.corre(clip, ctx)
    fila["estado"] = "luces; 11 rehecha"
    return fila


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", default=None)
    ap.add_argument("--lista", default=None)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--etapa11", choices=("si_cambia", "siempre"), default="si_cambia")
    a = ap.parse_args()
    import catalogo as CA
    if a.lista:
        clips = [l.strip() for l in open(a.lista, encoding="utf-8") if l.strip()]
    else:
        import lotes as LO
        clips = LO.conjunto(a.conjunto or "anotados")
    df = CA.manifiesto().set_index("clip_id")
    tareas = [(c, df.loc[c, "torneo"] if c in df.index else None) for c in clips
              if (RU.TRABAJO / ("lights_%s.csv" % c)).is_file()
              or (RU.TRABAJO / ("poses_%s.npz" % c)).is_file()]
    """Solo los clips que ya pasaron por la extraccion: el resto los hara la noche con el codigo nuevo."""
    reg = RU.TRABAJO / ("recalcula_luces_%s.csv" % time.strftime("%Y%m%d-%H%M%S"))
    cols = ["clip", "estado", "via_antes", "via_nueva", "tf_antes", "tf_nuevo", "error"]
    print("%d clips -> %s" % (len(tareas), reg), flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.procs) as ex, open(reg, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        fut = {ex.submit(uno, c, t, a.etapa11): c for c, t in tareas}
        for i, f in enumerate(as_completed(fut), 1):
            try:
                fila = f.result()
            except Exception as e:
                fila = {"clip": fut[f], "estado": "FALLO", "error": "%s: %s" % (type(e).__name__, e)}
            w.writerow(fila)
            fh.flush()
            if i % 100 == 0:
                print("   %d / %d  %.0f s" % (i, len(tareas), time.time() - t0), flush=True)
    print("hecho: %s" % reg, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    main()

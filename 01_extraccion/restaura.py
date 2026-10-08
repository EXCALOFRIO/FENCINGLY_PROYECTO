"""Trae de vuelta lo respaldado en HF: la otra mitad de `respalda.py`, sin la cual el respaldo no vale.

POR QUE EXISTE (2026-10-08). `lotes.sube` subia y `lotes.hechas` contaba lo subido como hecho, pero
no habia NADA que lo bajara. En una maquina nueva eso era peor que no tener respaldo: la reanudacion
daba por hechas las etapas 1 y 4, y la 7 caia por falta de `poses_` que solo estaban en el remoto.

QUE HACE: baja los tars de `data/lotes/` y deja cada fichero donde lo espera el pipeline --
`serie_eventos_`, `posicion_clip_` y `eventos_` en `RU.PISTA`, el resto en `RU.TRABAJO` --
**sin sobrescribir** lo que ya haya en disco. Solo acepta nombres planos de los prefijos conocidos:
un tar con rutas o nombres raros se rechaza miembro a miembro, no se extrae a ciegas.

    python restaura.py                       todo el remoto a su sitio
    python restaura.py --destino D:/prueba   a otra raiz (pruebas/ y pista/ dentro), para verificar
    python restaura.py --lote lote_X.tar.gz  solo ese lote
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys
import tarfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "codigo"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import lotes as LO  # noqa: E402
import respaldo as RSP  # noqa: E402
import rutas as RU  # noqa: E402

A_PISTA = ("serie_eventos_", "posicion_clip_", "eventos_")
A_TRABAJO = ("revision_", "poses_", "etiquetas_", "lights_", "zona_", "pista_vacia_")
NOMBRE = re.compile(r"^[A-Za-z0-9_\-]+\.(npz|csv|txt)$")


def destino_de(nombre, trabajo, pista):
    if not NOMBRE.match(nombre):
        return None
    if nombre.startswith(A_PISTA):
        return pista / nombre
    if nombre.startswith(A_TRABAJO):
        return trabajo / nombre
    return None


def restaura_tar(tar_local, trabajo, pista):
    n = ya = raros = 0
    with tarfile.open(tar_local, "r:*") as tar:
        for m in tar:
            if not m.isfile():
                continue
            d = destino_de(m.name, trabajo, pista)
            if d is None:
                raros += 1
                continue
            if d.exists():
                ya += 1
                continue
            d.parent.mkdir(parents=True, exist_ok=True)
            tmp = d.with_name(d.name + ".part")
            with tar.extractfile(m) as src, open(tmp, "wb") as fh:
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
            os.replace(tmp, d)
            n += 1
    return n, ya, raros


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", default=None)
    ap.add_argument("--lote", default=None)
    a = ap.parse_args()
    if a.destino:
        trabajo = pathlib.Path(a.destino) / "pruebas"
        pista = pathlib.Path(a.destino) / "pista"
    else:
        trabajo, pista = RU.TRABAJO, RU.PISTA
    ses = RSP.sesion(exigir=True)
    api = ses["api"]
    from huggingface_hub import hf_hub_download
    fich = api.list_repo_files(repo_id=LO.REPO, repo_type="dataset")
    tars = sorted(f for f in fich if f.startswith("data/lotes/") and f.endswith(".tar.gz"))
    if a.lote:
        tars = [t for t in tars if t.endswith(a.lote)]
    print("%d tars en %s -> %s , %s" % (len(tars), LO.REPO, trabajo, pista), flush=True)
    for t in tars:
        local = hf_hub_download(repo_id=LO.REPO, filename=t, repo_type="dataset",
                                token=RSP.token())
        n, ya, raros = restaura_tar(local, trabajo, pista)
        print("   %-48s nuevos %5d  ya estaban %5d  rechazados %d" % (t[-48:], n, ya, raros),
              flush=True)


if __name__ == "__main__":
    main()

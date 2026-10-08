"""Copia FUERA DE ESTA MAQUINA lo extraido, cada cierto tiempo, mientras la noche corre.

POR QUE EXISTE (2026-10-08). `lotes.sube()` llevaba hecho desde el 23/09, pero solo lo llamaba la
ruta de Colab. En local nadie lo llamaba, y medido hoy: **0 manifiestos, 0 clips en el remoto**,
con 7,1 GB y ~5.000 clips extraidos en un solo disco. Una corrida de semanas no puede tener todo
su producto colgando de un disco y de un `rm` equivocado.

QUE HACE, en cada vuelta:
  1. lee el remoto (`lotes.estado_remoto`, que lee MANIFIESTOS, no tars);
  2. pendientes = clips con TODAS las etapas en disco y que en el remoto no las tienen todas;
  3. los sube en tars de `--max` clips (`lotes.sube`, que exige repo PRIVADO y no da un lote por
     subido sin su manifiesto);
  4. duerme `--cada` minutos y repite.

Si una subida falla, el lote no se da por subido y entra en la vuelta siguiente: lo peor que pasa
es subir dos veces lo mismo, nunca saltarse un clip. La reanudacion de `lotes.hechas` ya junta disco
y remoto, asi que un clip subido dos veces no se cuenta doble.

    python respalda.py --conjunto anotados            vuelta cada 120 min, sin fin
    python respalda.py --conjunto anotados --una      una vuelta y sale
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "codigo"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
"""Sin esto cada subida escribe miles de lineas de barra en `resp.log`, semanas seguidas."""
import lotes as LO  # noqa: E402

ETAPAS = (1, 4, 7, 9, 11)


def vuelta(conjunto, maximo=500, etapas=ETAPAS):
    ids = LO.conjunto(conjunto)
    remoto = LO.estado_remoto(recarga=True)
    pend = [c for c in ids
            if LO.hechas(c, etapas) == set(etapas)
            and not set(etapas) <= (remoto.get(c) or set())]
    print("%s  %d clips completos en disco sin respaldo" % (time.strftime("%Y-%m-%d %H:%M"),
                                                          len(pend)), flush=True)
    ok = 0
    for i in range(0, len(pend), maximo):
        trozo = pend[i:i + maximo]
        r = LO.sube(trozo, list(etapas), etiqueta=time.strftime("%Y%m%d-%H%M%S"))
        ok += r.get("clips", 0) if r.get("ok") else 0
    print("   subidos en esta vuelta: %d" % ok, flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", default="anotados")
    ap.add_argument("--cada", type=float, default=120, help="minutos entre vueltas")
    ap.add_argument("--max", type=int, default=500, help="clips por tar")
    ap.add_argument("--una", action="store_true")
    a = ap.parse_args()
    while True:
        try:
            vuelta(a.conjunto, a.max)
        except Exception as ex:
            """Un fallo de red o de HF NO puede matar el respaldo de una corrida de semanas: se
            dice y se reintenta en la vuelta siguiente."""
            print("   VUELTA FALLIDA %s: %s" % (type(ex).__name__, str(ex)[:200]), flush=True)
        if a.una:
            break
        time.sleep(a.cada * 60)


if __name__ == "__main__":
    main()

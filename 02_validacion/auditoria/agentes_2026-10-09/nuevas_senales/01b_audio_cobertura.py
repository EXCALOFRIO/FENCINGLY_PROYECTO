"""Cuanto audio hay, con un denominador grande: ffprobe (sin decodificar) sobre TODOS los clips
de floret con 'initiated' y video en disco (4.069), agrupado por torneo.

Mucho mas barato que extraer el wav: solo lee las cabeceras del contenedor.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import pandas as pd
import pathlib

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import rutas as RU  # noqa: E402

OUT = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop-FENCINGLY-AI"
                    r"\2c75119e-09b9-43eb-a841-faa6869534b2\scratchpad\nuevas_senales\out")


def tiene_audio(ruta: str) -> bool:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
           "stream=codec_name", "-of", "json", ruta]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    try:
        d = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        return False
    return bool(d.get("streams"))


def main():
    df = pd.read_csv(RU.CATALOGO / "manifiesto.csv.gz")
    foil = df[(df.weapon == "foil") & df.initiated.notna()].copy()
    filas = []
    t0 = time.perf_counter()
    for i, cid in enumerate(foil.clip_id):
        v = RU.video(cid)
        if v is None:
            continue
        filas.append(dict(clip=cid, torneo=foil.iloc[i]["torneo"], tiene_audio=tiene_audio(v)))
        if (i + 1) % 500 == 0:
            print(f"... {i + 1}/{len(foil)}  ({time.perf_counter() - t0:.0f}s)")
    d = pd.DataFrame(filas)
    d.to_csv(OUT / "audio_cobertura_global.csv", index=False)
    print(f"\ntotal comprobados: {len(d)}   tiempo: {time.perf_counter() - t0:.0f}s"
          f"  ({(time.perf_counter() - t0) / max(1, len(d)):.3f} s/clip)")
    print(f"con audio: {d.tiene_audio.sum()}/{len(d)} = {d.tiene_audio.mean():.3f}")
    print("\npor torneo (ordenado por cobertura, torneos con >=10 clips):")
    g = d.groupby("torneo")["tiene_audio"].agg(["mean", "count"]).query("count >= 10")
    print(g.sort_values("mean", ascending=False).to_string())


if __name__ == "__main__":
    main()

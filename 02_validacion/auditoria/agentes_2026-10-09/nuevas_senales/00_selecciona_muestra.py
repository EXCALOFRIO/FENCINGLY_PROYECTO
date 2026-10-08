"""Selecciona una muestra de ~150-200 clips de floret anotados, con video, det_ y lights_ en disco.

NO escribe ni toca nada del proyecto: solo lee. La muestra se guarda en el scratchpad.
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import rutas as RU  # noqa: E402

import pandas as pd  # noqa: E402
import pathlib  # noqa: E402

OUT = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop-FENCINGLY-AI"
                    r"\2c75119e-09b9-43eb-a841-faa6869534b2\scratchpad\nuevas_senales\out")
OUT.mkdir(parents=True, exist_ok=True)

MANIFIESTO = RU.CATALOGO / "manifiesto.csv.gz"


def main():
    df = pd.read_csv(MANIFIESTO)
    foil = df[(df.weapon == "foil") & df.initiated.notna()].copy()
    print(f"floret con 'initiated': {len(foil)}")

    tiene_video, tiene_lights, tiene_det, tiene_poses = [], [], [], []
    for cid in foil.clip_id:
        tiene_video.append(RU.video(cid) is not None)
        tiene_lights.append((RU.TRABAJO / f"lights_{cid}.csv").is_file())
        tiene_det.append((RU.TRABAJO / f"det_{cid}.pkl").is_file())
        tiene_poses.append((RU.TRABAJO / f"poses_{cid}.npz").is_file())
    foil["tiene_video"] = tiene_video
    foil["tiene_lights"] = tiene_lights
    foil["tiene_det"] = tiene_det
    foil["tiene_poses"] = tiene_poses

    print("con video:", foil.tiene_video.sum())
    print("con lights_:", foil.tiene_lights.sum())
    print("con det_:", foil.tiene_det.sum())
    print("con poses_:", foil.tiene_poses.sum())
    listos = foil[foil.tiene_video & foil.tiene_lights & foil.tiene_det & foil.tiene_poses]
    print("con las 4 cosas:", len(listos))
    print(listos.initiated.value_counts())

    # muestra estratificada por 'initiated', semilla fija
    n_obj = 180
    frac = min(1.0, n_obj / max(1, len(listos)))
    partes = []
    for _, g in listos.groupby("initiated"):
        partes.append(g.sample(frac=frac, random_state=20261009))
    muestra = pd.concat(partes, ignore_index=True)
    if len(muestra) > n_obj:
        muestra = muestra.sample(n=n_obj, random_state=20261009).reset_index(drop=True)
    muestra = muestra.sort_values("clip_id").reset_index(drop=True)
    print("\nmuestra final:", len(muestra))
    print(muestra["initiated"].value_counts())

    cols = ["clip_id", "torneo", "left_score", "right_score", "touch", "initiated",
            "strip_location"]
    muestra[cols].to_csv(OUT / "muestra_clips.csv", index=False)
    print("\nguardado:", OUT / "muestra_clips.csv")


if __name__ == "__main__":
    main()

"""Reusa 01_audio.procesa_uno sobre una muestra mas grande de clips que SI tienen audio (segun el
barrido global de 01b), para medir el zumbador y los transitorios con mas N que los 7-8 clips de
la muestra principal (que solo tenia 180 clips y de ahi 8 con audio)."""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import time

import pandas as pd

HERE = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("au", str(HERE / "01_audio.py"))
au = importlib.util.module_from_spec(spec)
spec.loader.exec_module(au)

OUT = au.OUT


def main():
    cov = pd.read_csv(OUT / "audio_cobertura_global.csv")
    con_audio = cov[cov.tiene_audio == True]
    print("con audio en el barrido global:", len(con_audio))
    sub = con_audio.sample(n=min(120, len(con_audio)), random_state=5).reset_index(drop=True)

    # trae la etiqueta 'initiated' del manifiesto para cruzar
    man = pd.read_csv(au.RU.CATALOGO / "manifiesto.csv.gz", usecols=["clip_id", "initiated"])
    sub = sub.merge(man, left_on="clip", right_on="clip_id", how="left")

    filas = []
    t0 = time.perf_counter()
    for i, r in sub.iterrows():
        f = au.procesa_uno(r["clip"], r["torneo"])
        f["initiated"] = r["initiated"]
        filas.append(f)
        if (i + 1) % 30 == 0:
            print(f"... {i + 1}/{len(sub)} ({time.perf_counter() - t0:.0f}s)")
    d = pd.DataFrame(filas)
    d.to_csv(OUT / "audio_resultado_grande.csv", index=False)
    dt = time.perf_counter() - t0
    print(f"\ntiempo total {dt:.1f}s ({dt / len(sub):.2f} s/clip)")
    print(d["estado"].value_counts())
    ok = d[d.estado == "ok"]
    zz = ok[ok.zumbador_pasa.notna()]
    print(f"\nzumbador contestan (t0 valido): {len(zz)}/{len(ok)}")
    if len(zz):
        print(f"zumbador 'pasa': {zz.zumbador_pasa.sum()}/{len(zz)} = {zz.zumbador_pasa.mean():.3f}")
        print(f"TNR ventana tocado p50={zz.tnr_db.median():.1f}  control p50={zz.tnr_control_db.median():.1f}")
        print(f"margen p50={zz.margen_db.median():.1f}")

    print("\ntransitorios pre-tocado (2s) vs control (2s, 3-5s antes), por initiated:")
    tt = ok[ok.n_transitorios_pre2s.notna()]
    print(tt.groupby("initiated")[["n_transitorios_pre2s", "n_transitorios_control2s"]]
          .agg(["mean", "count"]))


if __name__ == "__main__":
    main()

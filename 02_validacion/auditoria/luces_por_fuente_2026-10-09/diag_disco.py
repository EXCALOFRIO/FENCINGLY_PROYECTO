"""Diagnostico de lo que hay en disco: A1, COB, ACIERTO*, C0 por arma, via y torneo, mas rasgos de
fuente (resolucion, fps, audio). Solo lectura."""
import csv
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import catalogo as CA  # noqa: E402
import rutas as RU  # noqa: E402

OUT = pathlib.Path(r"D:\FENCINGLY_SCRATCH\agente_luces")
AUD = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\auditoria"
                   r"\agentes_2026-10-09\nuevas_senales\audio_cobertura_global.csv")


def lee_luces(p):
    with open(p, encoding="utf-8") as fh:
        rr = csv.DictReader(fh)
        filas = list(rr)
    via = filas[0].get("via") if filas else None
    col = [f for f in filas if f.get("canal") in ("roja", "verde") and f.get("cuadro_encendido")]
    bla = [f for f in filas if f.get("canal") == "blanca" and f.get("cuadro_encendido")]
    todas = [f for f in filas if f.get("cuadro_encendido")]
    canales = sorted({f["canal"] for f in col})
    if canales == ["roja"]:
        pred = "left"
    elif canales == ["verde"]:
        pred = "right"
    elif canales == ["roja", "verde"]:
        pred = "dos"
    else:
        pred = "sin"
    lados = sorted({f["lado"] for f in col})
    pred_lado = {("L",): "left", ("R",): "right"}.get(tuple(lados), "dos" if len(lados) == 2 else "sin")
    c0 = min((int(float(f["cuadro_encendido"])) for f in todas), default=None)
    disc = any(f.get("discordante") == "True" for f in col)
    return dict(via=via, pred=pred, pred_lado=pred_lado, n_blanca=len(bla), cuadro=c0, disc=disc)


def main():
    df = CA.manifiesto().set_index("clip_id")
    aud = pd.read_csv(AUD).set_index("clip")["tiene_audio"].to_dict()
    filas = []
    for p in sorted(RU.TRABAJO.glob("lights_*.csv")):
        c = p.stem[len("lights_"):]
        if c not in df.index:
            continue
        m = df.loc[c]
        r = lee_luces(p)
        r.update(clip=c, torneo=m["torneo"], arma=m["weapon"], touch=m["touch"],
                 audio=aud.get(c))
        pz = RU.TRABAJO / ("poses_%s.npz" % c)
        if pz.is_file():
            try:
                with np.load(pz, allow_pickle=True) as z:
                    r["W"], r["H"], r["T"] = int(z["W"]), int(z["H"]), int(z["T"])
                    r["fps"] = float(z["fps"]) if "fps" in z.files else None
            except Exception:
                pass
        filas.append(r)
    d = pd.DataFrame(filas)
    d.to_csv(OUT / "diag_disco.csv", index=False)
    lr = d["touch"].isin(["left", "right"])
    una = d["pred"].isin(["left", "right"])
    d["ok1"] = np.where(lr & una, d["pred"] == d["touch"], np.nan)
    d["ok1_lado"] = np.where(lr & d["pred_lado"].isin(["left", "right"]),
                            d["pred_lado"] == d["touch"], np.nan)

    def resumen(g):
        lr_ = g["touch"].isin(["left", "right"])
        tod = g["touch"].isin(["left", "right", "double"])
        una_ = g["pred"].isin(["left", "right"]) & lr_
        return pd.Series(dict(
            n=len(g),
            n_lr=int(lr_.sum()),
            COB=round(float((g.loc[tod, "pred"] != "sin").mean()), 3) if tod.any() else None,
            frac_dos=round(float((g.loc[tod, "pred"] == "dos").mean()), 3) if tod.any() else None,
            n_A1=int(una_.sum()),
            A1=round(float((g.loc[una_, "pred"] == g.loc[una_, "touch"]).mean()), 3) if una_.any() else None,
            A1_lado=round(float(g["ok1_lado"].mean()), 3) if g["ok1_lado"].notna().any() else None,
            ACIERTOx=round(float(((g["pred"] == g["touch"]) & lr_).sum() / max(1, lr_.sum())), 3),
            C0=round(float((g["cuadro"] == 0).sum() / max(1, g["cuadro"].notna().sum())), 3),
            audio=round(float(g["audio"].astype(float).mean()), 2) if g["audio"].notna().any() else None,
            res=g["H"].mode().iloc[0] if g["H"].notna().any() else None,
            fps=round(float(g["fps"].median()), 1) if g["fps"].notna().any() else None,
            via_mod=g["via"].mode().iloc[0] if g["via"].notna().any() else None,
        ))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print("POR ARMA")
    print(d.groupby("arma").apply(resumen, include_groups=False))
    print("\nPOR ARMA Y VIA")
    print(d.groupby(["arma", "via"]).apply(resumen, include_groups=False))
    f = d[d["arma"] == "foil"]
    print("\nFLORETE POR TORNEO")
    t = f.groupby("torneo").apply(resumen, include_groups=False).sort_values("A1")
    print(t)
    t.to_csv(OUT / "diag_florete_por_torneo.csv")
    print("\nFLORETE POR AUDIO")
    print(f.groupby(f["audio"].astype(str)).apply(resumen, include_groups=False))


if __name__ == "__main__":
    main()

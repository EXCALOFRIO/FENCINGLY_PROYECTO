import sys, numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "diag_disco.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 30)
def resumen(g):
    lr_ = g["touch"].isin(["left", "right"]); tod = g["touch"].isin(["left", "right", "double"])
    una_ = g["pred"].isin(["left", "right"]) & lr_
    return pd.Series(dict(n=len(g), n_lr=int(lr_.sum()),
        COB=round(float((g.loc[tod, "pred"] != "sin").mean()), 3) if tod.any() else np.nan,
        dos=round(float((g.loc[tod, "pred"] == "dos").mean()), 3) if tod.any() else np.nan,
        n_A1=int(una_.sum()),
        A1=round(float((g.loc[una_, "pred"] == g.loc[una_, "touch"]).mean()), 3) if una_.any() else np.nan,
        ACIERTOx=round(float(((g["pred"] == g["touch"]) & lr_).sum() / max(1, lr_.sum())), 3),
        C0=round(float((g["cuadro"] == 0).sum() / max(1, g["cuadro"].notna().sum())), 3),
        audio=round(float(g["audio"].astype(float).mean()), 2) if "audio" in g and g["audio"].notna().any() else np.nan,
        H=g["H"].mode().iloc[0] if "H" in g and g["H"].notna().any() else np.nan,
        fps=round(float(g["fps"].median()), 1) if "fps" in g and g["fps"].notna().any() else np.nan))
print("POR ARMA"); print(d.groupby("arma").apply(resumen, include_groups=False))
print("\nPOR ARMA Y VIA"); print(d.groupby(["arma", "via"]).apply(resumen, include_groups=False))
f = d[d["arma"] == "foil"]
print("\nFLORETE POR TORNEO"); print(f.groupby("torneo").apply(resumen, include_groups=False).sort_values("A1"))
if "audio" in f: print("\nFLORETE POR AUDIO"); print(f.groupby(f["audio"].astype(str)).apply(resumen, include_groups=False))

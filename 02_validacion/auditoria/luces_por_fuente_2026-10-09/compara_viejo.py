import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 30)
d = pd.read_csv("diag_disco.csv")
v = pd.read_csv("viejo_meta.csv")
def predv(r):
    a, b = pd.notna(r["red_L"]), pd.notna(r["green_R"])
    return "dos" if a and b else "left" if a else "right" if b else "sin"
v["pred_v"] = v.apply(predv, axis=1)
v = v.sort_values("detector").drop_duplicates("clip")   # 'fie' antes que 'generic'
m = d.merge(v[["clip", "pred_v", "detector", "red_L", "green_R", "fps", "t_touch_s"]], on="clip", how="inner", suffixes=("", "_v"))
print("clips comunes:", len(m), m["arma"].value_counts().to_dict())
def A1(g, col):
    lr = g["touch"].isin(["left", "right"]); u = g[col].isin(["left", "right"]) & lr
    return pd.Series({"n_A1": int(u.sum()), "A1": round(float((g.loc[u, col] == g.loc[u, "touch"]).mean()), 3) if u.any() else np.nan,
                      "COB": round(float((g.loc[g["touch"].isin(["left","right","double"]), col] != "sin").mean()), 3),
                      "dos": round(float((g.loc[g["touch"].isin(["left","right","double"]), col] == "dos").mean()), 3),
                      "ACx": round(float(((g[col] == g["touch"]) & lr).sum() / max(1, lr.sum())), 3)})
def ambos(g):
    a = A1(g, "pred"); b = A1(g, "pred_v")
    return pd.concat([a.add_suffix("_nuevo"), b.add_suffix("_viejo")])
print(m.groupby("arma").apply(ambos, include_groups=False))
f = m[m["arma"] == "foil"]
print(f.groupby("torneo").apply(ambos, include_groups=False).sort_values("A1_nuevo"))
print(f.groupby("via").apply(ambos, include_groups=False))
# en los mismos clips donde LOS DOS dan una sola luz
both = f[f["pred"].isin(["left","right"]) & f["pred_v"].isin(["left","right"]) & f["touch"].isin(["left","right"])]
print("ambos una luz:", len(both), "nuevo", round((both.pred==both.touch).mean(),3), "viejo", round((both.pred_v==both.touch).mean(),3), "coinciden", round((both.pred==both.pred_v).mean(),3))
# cruce
print(pd.crosstab(f["pred"], f["pred_v"]))
m.to_csv("comun_viejo_nuevo.csv", index=False)

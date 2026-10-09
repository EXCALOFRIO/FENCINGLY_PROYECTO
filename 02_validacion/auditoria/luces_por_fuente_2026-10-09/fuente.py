"""La hipotesis de la fuente: torneos de florete sin audio vs con audio, rasgos de FUENTE y A1."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import evalua as E
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 30)
d = E.tabla_clips()
f = d[d.arma == "foil"].copy()
def a1(g, R):
    u = g[R].isin(["left", "right"]) & g.touch.isin(["left", "right"])
    return (g.loc[u, R] == g.loc[u, "touch"]).mean() if u.any() else np.nan, int(u.sum())
filas = []
for t, g in f.groupby("torneo"):
    r = dict(torneo=t, n=len(g), audio=g.audio.astype(float).mean(), H=g.H.mode().iloc[0] if g.H.notna().any() else None,
             fps=round(g.fps.median(), 1) if g.fps.notna().any() else None,
             v2_fie=(g.v2_fuente == "fie").mean(), v2_gen=(g.v2_fuente == "generic").mean(), v2_none=(g.v2_fuente == "none").mean(),
             cajas_det=(g.v2_cajas.isin(["detected", "split"])).mean(), cajas_prior=(g.v2_cajas == "prior").mean())
    for R in ("R0", "R1", "R2"):
        r["A1_" + R], r["nA1_" + R] = a1(g, R)
    filas.append(r)
t = pd.DataFrame(filas).sort_values("n", ascending=False)
print(t.round(3).to_string(index=False))
t.to_csv("fuente_por_torneo.csv", index=False)
# clip-level: sin audio vs con audio
f["sin_audio"] = f.audio.map({True: False, False: True})
for R in ("R0", "R1", "R2"):
    for k, g in f[f.sin_audio.notna()].groupby("sin_audio"):
        a, n = a1(g, R); print(R, "sin_audio=%s" % k, "A1=%.3f n=%d" % (a, n), "clips=%d torneos=%d" % (len(g), g.torneo.nunique()))
# bootstrap por TORNEO de la diferencia A1(sin audio) - A1(con audio), R0
rng = np.random.default_rng(1)
def dif(gs, R):
    s = gs[gs.sin_audio == True]; c = gs[gs.sin_audio == False]
    return a1(s, R)[0] - a1(c, R)[0]
tors = f[f.sin_audio.notna()].torneo.unique()
for R in ("R0", "R1"):
    base = dif(f, R); ds = []
    for _ in range(2000):
        pick = rng.choice(tors, len(tors))
        gs = pd.concat([f[f.torneo == x] for x in pick])
        ds.append(dif(gs, R))
    print(R, "A1 sin audio - con audio: %+.3f IC95 [%+.3f, %+.3f] (bootstrap por torneo)" % (base, *np.nanpercentile(ds, [2.5, 97.5])))

"""Puerta del arreglo de cuadro 0 (R3 contra R0), en los 600 clips al azar (sin sesgo de seleccion)."""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import evalua as E
d = E.tabla_clips()
az = set(l.strip() for l in open("lista_r3_azar600.txt") if l.strip())
g = d[d["clip"].isin(az) & d.R3.notna()].copy()
print("clips al azar con R0 y R3:", len(g), g.arma.value_counts().to_dict())
con = g.R0_c.notna()
print("C0  R0 %.3f (%d de %d con luz)   R3 %.3f (%d de %d)" % ((g.R0_c == 0).sum() / con.sum(), (g.R0_c == 0).sum(), con.sum(),
      (g.R3_c == 0).sum() / max(1, g.R3_c.notna().sum()), (g.R3_c == 0).sum(), g.R3_c.notna().sum()))
# salida identica? comparo prediccion y cuadro
med = E.lee(E.D / "med_anotados.jsonl"); r3 = E.lee(E.D / "med_r3.jsonl")
def firma(v):
    return sorted((l["canal"], l["lado"], l["cuadro_encendido"]) for l in (v or {}).get("luces", []))
g["igual"] = [firma(med[c]["vigente"]) == firma(r3[c]["r3"]) and med[c]["vigente"].get("via") == r3[c]["r3"].get("via") for c in g["clip"]]
no0 = g.R0_c.notna() & (g.R0_c != 0)
print("clips con encendido R0 != 0: %d, salida identica en %d (%.3f)" % (no0.sum(), g.loc[no0, "igual"].sum(), g.loc[no0, "igual"].mean()))
print("   los distintos:"); print(g.loc[no0 & ~g.igual, ["clip", "torneo", "touch", "R0", "R0_c", "R3", "R3_c"]].to_string())
cam = ~g.igual
print("clips que cambian: %d" % cam.sum())
gg = g[cam]
b, lo, hi = E.boot_dif(gg, "R0", "R3")
m0, m3 = E.metricas(gg, "R0"), E.metricas(gg, "R3")
print("A1 en los que cambian: R0 %.3f (n=%d)  R3 %.3f (n=%d)  dif %+.3f IC95 [%+.3f, %+.3f]" % (m0["A1"], m0["n_A1"], m3["A1"], m3["n_A1"], b, lo, hi))
print("ACx en los que cambian: R0 %.3f  R3 %.3f" % (m0["ACx"], m3["ACx"]))
print(gg[["clip", "torneo", "touch", "R0", "R0_c", "R3", "R3_c", "R1", "R1_c"]].to_string())
gg[["clip", "torneo", "touch", "R0", "R0_c", "R3", "R3_c", "R1", "R1_c"]].to_csv("c0_cambian.csv", index=False)

import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import evalua as E
d = E.tabla_clips().set_index("clip")
r = E.lee(E.D / "med_r3b.jsonl"); med = E.lee(E.D / "med_anotados.jsonl")
rows = []
for c, x in r.items():
    p, cc, via = E.de_vigente(x["r3b"])
    rows.append(dict(clip=c, R3b=p, R3b_c=cc, R3b_via=via))
b = pd.DataFrame(rows).set_index("clip")
g = d.join(b, how="inner").reset_index()
ctrl = set(l.strip() for l in open("lista_r3b_ctrl50.txt"))
az = set(l.strip() for l in open("lista_r3_azar600.txt"))
def firma(v): return sorted((l["canal"], l["lado"], l["cuadro_encendido"]) for l in v.get("luces", []))
g["igual"] = [firma(med[c]["vigente"]) == firma(r[c]["r3b"]) and med[c]["vigente"]["via"] == r[c]["r3b"]["via"] for c in g["clip"]]
k = g[g["clip"].isin(ctrl)]
print("PUERTA 3 (control, 50 clips con cuadro != 0): identicos %d de %d" % (k.igual.sum(), len(k)))
z = g[~g["clip"].isin(ctrl)]
print("clips con R0 en cuadro 0: %d (%s)" % (len(z), z.arma.value_counts().to_dict()))
print("PUERTA 1: C0 R0 %.3f -> R3b %.3f (de los que contestan: %d de %d)" % (1.0, (z.R3b_c == 0).mean() if z.R3b_c.notna().any() else 0, (z.R3b_c == 0).sum(), z.R3b_c.notna().sum()))
for nombre, gg in (("todos", z), ("sin los 38 de la muestra de 600", z[~z["clip"].isin(az)]), ("solo florete", z[z.arma == "foil"])):
    m0, m3 = E.metricas(gg.rename(columns={}), "R0"), E.metricas(gg, "R3b")
    bb, lo, hi = E.boot_dif(gg, "R0", "R3b"); b2, lo2, hi2 = E.boot_dif(gg, "R0", "R3b", campo="ACx")
    print("PUERTA 2 [%s] n=%d: A1 R0 %.3f (n=%d) -> R3b %.3f (n=%d), dif %+.3f IC95 [%+.3f, %+.3f]; ACx %.3f -> %.3f (%+.3f [%+.3f, %+.3f]); COB %.3f -> %.3f"
          % (nombre, len(gg), m0["A1"], m0["n_A1"], m3["A1"], m3["n_A1"], bb, lo, hi, m0["ACx"], m3["ACx"], b2, lo2, hi2, m0["COB"], m3["COB"]))
print(z.R3b_via.value_counts().to_dict())
z[["clip", "torneo", "arma", "touch", "R0", "R0_c", "R3b", "R3b_c", "R1", "R1_c"]].to_csv("c0b_clips.csv", index=False)
print(z[z.R3b_c.notna()].sample(12, random_state=3)[["clip", "torneo", "touch", "R0", "R3b", "R3b_c", "R1", "R1_c"]].to_string())

import sys, numpy as np, collections
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import catalogo as CA, modelo as MO, evalua as EV
for mitad in ("AJUSTE", "PRUEBA"):
    rs = [r for r in MO.carga() if r["mitad"] == mitad]
    df = CA.manifiesto().set_index("clip_id")
    mods = EV.modelos(rs); pred = EV.predice(rs, mods)
    por = collections.defaultdict(list)
    for c, (x, M, r) in pred.items():
        if M["amin"] - 2 <= r["ang"] <= M["amax"] + 2:
            y = df.loc[c, "strip_location"]
            por[r["torneo"]].append((CA.ZONAS[MO.zona(x)] == y, y == "middle", M["n"], M["e"]))
    print(mitad)
    for t, L in sorted(por.items(), key=lambda kv: -len(kv[1])):
        a = np.array(L)
        print("   %-28s nuevos %4d  zona %.2f  middle %.2f  dif %+.2f   refs %3d  err_loo %.2f m" % (t[:28], len(L), a[:, 0].mean(), a[:, 1].mean(), a[:, 0].mean() - a[:, 1].mean(), L[0][2], L[0][3]))

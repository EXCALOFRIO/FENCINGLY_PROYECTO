import sys, numpy as np, collections
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import catalogo as CA, modelo as MO, evalua as EV
rs = [r for r in MO.carga() if r["mitad"] == "AJUSTE"]
df = CA.manifiesto().set_index("clip_id")
mods = EV.modelos(rs); pred = EV.predice(rs, mods)
refs = [r for r in rs if r["motivo"] == "ok" and r["alt_m"]]
sin = [r for r in rs if r["motivo"] == "sinH" and r["alt_m"]]
for nom, L in (("referencias", refs), ("sin homografia", sin)):
    al = np.array([r["alt_m"] for r in L]); an = np.array([abs(r["ang"]) for r in L if r["ang"] is not None])
    nt = np.array([np.median([f["ntrans"] for f in r["fr"]]) for r in L])
    print("%-15s n=%4d  estatura/alto imagen mediana %.2f (p75 %.2f)  |ang| mediana %.1f  rayas vistas mediana %.1f" % (nom, len(L), np.median(al), np.percentile(al, 75), np.median(an), np.median(nt)))
M = np.zeros((5, 5), int)
for c, (x, Mo, r) in pred.items():
    if Mo["amin"] - 2 <= r["ang"] <= Mo["amax"] + 2:
        M[CA.ZONAS.index(df.loc[c, "strip_location"]), MO.zona(x)] += 1
print("confusion (filas verdad, columnas prediccion):", CA.ZONAS)
print(M)

import sys, csv, json, os
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import zona_tocado as ZT, rutas as RU, catalogo as CA
df = CA.manifiesto().set_index("clip_id")
print("RESPALDO", ZT.RESPALDO, "BANDA", ZT.BANDA_RESPALDO)
clips = sys.argv[1:]
for c in clips:
    r = ZT.analiza(c, RU.video(c), df.loc[c, "torneo"])
    prod = next(csv.DictReader(open(RU.TRABAJO / ("zona_%s.csv" % c), encoding="utf-8")), {})
    claves = ["motivo", "cuadro_tocado", "cuadro_usado", "x_tocado_m", "zona", "conf_h", "metodo_h", "frac_cuadros_con_h"]
    igual = all(str(r.get(k)) == str(prod.get(k)) or (r.get(k) is None and prod.get(k) in ("", None)) for k in claves)
    print(c, "IGUAL" if igual else "DISTINTO", r.get("via_h"), r.get("zona"), prod.get("zona"), r.get("motivo")[:30])
    if not igual:
        print("   ", {k: (r.get(k), prod.get(k)) for k in claves if str(r.get(k)) != str(prod.get(k))})

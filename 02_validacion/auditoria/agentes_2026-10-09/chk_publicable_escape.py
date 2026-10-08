"""Check 7b: filas de serie_eventos_<clip>.csv con velocidad_m_s>8 o |x|>8 que NO estan marcadas
publicable==False ni sin_homografia==True, es decir que pasaron la puerta fisica y aun asi
tienen un valor imposible. Lectura SOLO de 02_validacion/pista y el manifiesto."""
import os, csv, collections
import pandas as pd

d = r"C:/Users/Alejandro/Desktop/FENCINGLY_PROYECTO/02_validacion/pista"
OUT = r"C:/Users/Alejandro/AppData/Local/Temp/claude/C--Users-Alejandro-Desktop-FENCINGLY-AI/2c75119e-09b9-43eb-a841-faa6869534b2/scratchpad/calidad"
man = pd.read_csv(r"C:/Users/Alejandro/Desktop/FENCINGLY_PROYECTO/03_datos/catalogo/manifiesto.csv.gz",
                   dtype=str, keep_default_na=False, na_values=[]).set_index("clip_id")

rows = []
n_bad_total = 0
for f in os.listdir(d):
    if not (f.startswith("serie_eventos_") and f.endswith(".csv")):
        continue
    p = os.path.join(d, f)
    if os.path.getsize(p) == 0:
        continue
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                v = float(r["velocidad_m_s"]) if r.get("velocidad_m_s") else None
            except ValueError:
                v = None
            xs = []
            for campo in ("x_inicio_m", "x_fin_m"):
                try:
                    x = float(r[campo]) if r.get(campo) else None
                except ValueError:
                    x = None
                if x is not None:
                    xs.append(x)
            bad = (v is not None and v > 8) or any(abs(x) > 8 for x in xs)
            if not bad:
                continue
            n_bad_total += 1
            pub, sinh = r.get("publicable"), r.get("sin_homografia")
            if pub == "False" or sinh == "True":
                continue
            cid = r.get("clip")
            torneo = man.loc[cid, "torneo"] if cid in man.index else "?"
            worst = max([v or 0] + [abs(x) for x in xs])
            rows.append((cid, torneo, r.get("tipo"), v, xs, pub, sinh, worst))

print("filas imposibles (v>8 o |x|>8) en total:", n_bad_total)
print("de esas, SIN marcar no-fiable (publicable=='True' o vacio, sin_homografia no 'True'):", len(rows))
c = collections.Counter(t for _, t, *_ in rows)
print("\npor torneo:")
for t, n in c.most_common(25):
    print("  %-28s %d" % (t, n))
rows.sort(key=lambda x: -x[-1])
print("\npeores 15 por magnitud:")
for r in rows[:15]:
    print(r)
print("\nclips distintos afectados:", len(set(r[0] for r in rows)))

pd.DataFrame(rows, columns=["clip", "torneo", "tipo", "velocidad_m_s", "x_m", "publicable",
                             "sin_homografia", "peor_valor"]).to_csv(
    os.path.join(OUT, "publicable_escape.csv"), index=False)

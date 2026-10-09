"""El control 'base' de prior.py tiene que reproducir el zona_ de disco (2 clips de AJUSTE)."""
import sys, os, csv, collections
os.chdir(r"D:\FENCINGLY_SCRATCH\agente_angulo")
sys.path.insert(0, r"D:\FENCINGLY_SCRATCH\agente_angulo")
import prior as PR
import modelo as MO
import rutas as RU

rs = [r for r in MO.carga("rasgos.jsonl") if r["mitad"] == "AJUSTE"]
por = collections.defaultdict(list)
for r in rs:
    if r["motivo"] == "ok" and r["ang"] is not None and r["x_prod"] is not None and r["desp"] is not None:
        por[r["torneo"]].append(r)
t = max(por, key=lambda k: len(por[k]))
cl = [por[t][0]["clip"], por[t][1]["clip"]] + [r["clip"] for r in rs if r["torneo"] == t and r["motivo"] == "sinH"][:1]
for c in cl:
    m = MO.ajusta([r for r in por[t] if r["clip"] != c], True, True)
    out = PR.analiza_varios(c, t, m)
    z = next(csv.DictReader(open(RU.TRABAJO / ("zona_%s.csv" % c), encoding="utf-8")))
    print(c, t, "disco:", z["motivo"][:24], z.get("x_tocado_m"), z.get("zona"), "|", out["res"], flush=True)

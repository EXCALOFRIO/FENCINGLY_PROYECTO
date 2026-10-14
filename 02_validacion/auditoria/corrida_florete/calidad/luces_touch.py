import csv, json, pathlib, sys, collections
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import catalogo as CA, rutas as RU
df = CA.manifiesto().set_index("clip_id")
def lados(p):
    if not p.is_file():
        return None
    return {r.get("lado") for r in csv.DictReader(open(p, encoding="utf-8")) if r.get("canal") in ("roja", "verde")}
cuenta = collections.defaultdict(lambda: collections.Counter())
def mira(g, p, c):
    t = df.loc[c, "touch"] if c in df.index else None
    l = lados(p)
    if l is None or t not in ("left", "right"):
        cuenta[g]["sin dato"] += 1; return
    lado = "L" if t == "left" else "R"
    cuenta[g]["n"] += 1
    if not l: cuenta[g]["ninguna luz de color"] += 1
    elif lado in l: cuenta[g]["luz del que toco"] += 1
    else: cuenta[g]["SOLO la del otro"] += 1
    if l == {"L", "R"}: cuenta[g]["dos luces"] += 1
for e in ("e_1014_0011", "e_1014_1808"):
    E = pathlib.Path(r"D:\FENCINGLY_SCRATCH\ensayo") / e
    g = json.load(open(E / "informe.json", encoding="utf-8"))["clips"]
    for c, grupo in g.items():
        mira("ensayo " + grupo, E / "trabajo" / ("lights_%s.csv" % c), c)
for p in RU.TRABAJO.glob("lights_*.csv"):
    c = p.name[7:-4]
    if c in df.index and df.loc[c, "weapon"] == "foil":
        mira("pruebas foil", p, c)
for g, k in sorted(cuenta.items()):
    n = k["n"] or 1
    print("%-24s n=%5d | luz del que toco %5.1f %% | solo la del otro %4.1f %% | ninguna %4.1f %% | dos luces %4.1f %%"
          % (g, k["n"], 100 * k["luz del que toco"] / n, 100 * k["SOLO la del otro"] / n, 100 * k["ninguna luz de color"] / n, 100 * k["dos luces"] / n))

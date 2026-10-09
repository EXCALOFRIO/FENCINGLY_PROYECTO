"""Las puertas declaradas (LUCES_POR_FUENTE_2026-10-09.md §0) para una regla, en una mitad."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import evalua as E
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)

def a1(g, R):
    u = g[R].isin(["left", "right"]) & g.touch.isin(["left", "right"])
    return (g.loc[u, R] == g.loc[u, "touch"]).mean() if u.any() else np.nan, int(u.sum())
def cob(g, R):
    t = g.touch.isin(["left", "right", "double"])
    return (g.loc[t, R] != "sin").mean() if t.any() else np.nan

def puertas(d, R, mitad):
    g0 = d if mitad == "TODO" else d[d.mitad == mitad]
    g0 = g0[g0[R].notna()]
    out = {}
    f = g0[g0.arma == "foil"]
    b, lo, hi = E.boot_dif(f, "R0", R)
    b2, lo2, hi2 = E.boot_dif(f, "R0", R, campo="ACx")
    out["1 A1 florete >= +0.05, IC>0"] = ("%+.3f [%+.3f,%+.3f]" % (b, lo, hi), b >= 0.05 and lo > 0)
    out["2 ACx no baja"] = ("%+.3f [%+.3f,%+.3f]" % (b2, lo2, hi2), b2 >= 0)
    malos = []
    for t, g in g0.groupby("torneo"):
        a0, n0 = a1(g, "R0")
        if n0 >= 30 and a0 >= 0.85:
            ar, nr = a1(g, R)
            dc = cob(g, R) - cob(g, "R0")
            malos.append((t, round(a0, 3), n0, round(ar, 3), nr, round(dc, 3), (ar - a0 >= -0.02) and dc >= -0.02))
    out["3 torneos buenos sin perder >0.02"] = (malos, all(m[-1] for m in malos))
    res4 = []
    for arma in ("epee", "sabre"):
        g = g0[g0.arma == arma]
        a0, n0 = a1(g, "R0"); ar, nr = a1(g, R)
        res4.append((arma, round(a0, 3), n0, round(ar, 3), nr, ar - a0 >= -0.02))
    out["4 espada/sable A1 no baja >0.02"] = (res4, all(r[-1] for r in res4))
    return out

if __name__ == "__main__":
    d = E.tabla_clips(); d["mitad"] = E.mitades(d)
    for mitad in sys.argv[1].split(","):
        for R in sys.argv[2].split(","):
            print("== %s %s" % (mitad, R))
            for k, (v, ok) in puertas(d, R, mitad).items():
                print("   %-36s %s  %s" % (k, "PASA" if ok else "NO PASA", v))

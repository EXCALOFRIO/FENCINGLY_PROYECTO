import glob, os, sys, numpy as np
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo"); sys.path.insert(0, ".")
import fit
from concurrent.futures import ProcessPoolExecutor
metros = {l.strip() for l in open("lista_metros_ajuste.txt", encoding="utf-8") if l.strip()}
fs = [os.path.basename(f)[:-4] for f in sorted(glob.glob(r"D:\FENCINGLY_SCRATCH\agente_tripode\cache\*.pkl"))]
cs = [c for c in fs if c in metros][: int(sys.argv[1]) if len(sys.argv) > 1 else 12]
if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(fit.uno, [r"D:\FENCINGLY_SCRATCH\agente_tripode\cache"] * len(cs), cs))
    errs = []
    for o in res:
        if o.get("estado") != "ok" or not o.get("prod_ok"):
            print(o["clip"][:28], o.get("estado"), o.get("prod_ok")); continue
        e = o["t_x"] - o["prod_x"]; errs.append(e)
        p = o["t_p"]
        print("%-28s prod %6.2f %-17s trip %6.2f %-17s err %6.2f marg %s rayas %s s %.1f C=(%.1f,%.1f,%.1f) roll %.1f f %.0f" % (
            o["clip"][:28], o["prod_x"], o["prod_zona"], o["t_x"], o["t_zona"], e,
            None if o["t_margen"] is None else round(o["t_margen"], 1), o["t_rayas"], o["s_fit"], p[0], p[1], p[2], np.degrees(p[5]), np.exp(p[6])))
    errs = np.abs(errs)
    print("n", len(errs), "mediana |err|", np.median(errs), "frac<1m", (errs < 1).mean())

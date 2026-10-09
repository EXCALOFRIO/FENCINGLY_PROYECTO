"""Evalua la candidata N(b) (franja del rotulo en b) sobre los clips sin homografia en produccion.

    python evalua_banda.py res_dir
"""
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
sys.path.insert(0, r"D:\FENCINGLY_SCRATCH\agente_zona")
import catalogo as CA  # noqa: E402
import rutas as RU  # noqa: E402
from evalua import boot  # noqa: E402


def plaus(x):
    if x.get("x_izq") is None or x.get("x_der") is None:
        return False
    d = x["x_der"] - x["x_izq"]
    return 0.5 <= d <= 5.0 and abs(x["x_izq"]) <= 8 and abs(x["x_der"]) <= 8


def main():
    res = pathlib.Path(sys.argv[1])
    df = CA.manifiesto().set_index("clip_id")
    falla = []
    for f in sorted(res.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        p = RU.TRABAJO / ("zona_%s.csv" % r["clip"])
        prod = next(csv.DictReader(open(p, encoding="utf-8")), {})
        if (prod.get("motivo") or "").startswith("sin homografia"):
            falla.append((r, df.loc[r["clip"], "strip_location"]))
    print("sin homografia en produccion:", len(falla))
    claves = sorted({k for r, _ in falla for k in r if k.startswith("N")})
    print("%-18s %6s %6s %7s %7s %7s  %s" % ("candidata", "nuevos", "cob", "zona", "middle", "dif", "IC95 por clip"))
    for k in claves:
        for c in ("0.35", "0.25"):
            for pl in (False, True):
                zs, ys = [], []
                for r, y in falla:
                    x = (r.get(k) or {}).get("reglas", {}).get(c)
                    if x and "zona" in x and (not pl or plaus(x)):
                        zs.append(x["zona"])
                        ys.append(y)
                zs, ys = np.array(zs), np.array(ys)
                nombre = "%s(%s)%s" % (k, c, "+p" if pl else "")
                if len(zs) < 5:
                    print("%-18s %6d" % (nombre, len(zs)))
                    continue
                okz, okm = zs == ys, ys == "middle"
                lo, hi = boot(okm, okz)
                print("%-18s %6d %6.3f %7.3f %7.3f %+7.3f  [%+.3f, %+.3f]"
                      % (nombre, len(zs), len(zs) / len(falla), okz.mean(), okm.mean(),
                         okz.mean() - okm.mean(), lo, hi))


if __name__ == "__main__":
    main()

"""Modelo por torneo: x_tiradores = a + b*tan(ang) (+ c*desplazamiento). Error en metros contra las
referencias (x de produccion) dejando uno fuera. Solo AJUSTE en el desarrollo."""
import json, sys, numpy as np, collections
from sklearn.linear_model import HuberRegressor, LinearRegression

TORN_AJ = set(l.strip() for l in open(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\auditoria\zona_cobertura_2026-10-09\torneos_ajuste.txt", encoding="utf-8") if l.strip())


def carga(path="rasgos.jsonl"):
    out = []
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        if r.get("motivo") not in ("ok", "sinH") or not r.get("fr"):
            continue
        angs = [f["far"] for f in r["fr"] if f["far"] is not None]
        r["ang"] = float(np.median(angs)) if angs else None
        f = r["fr"][-1]
        us = [u for u in f["pies_u"] if u is not None]
        al = [a for a in f["alt"] if a is not None]
        r["u"] = float(np.mean(us)) - 0.5 if us else None
        r["alt_m"] = float(np.mean(al)) if al else None
        r["dos_pies"] = len(us) == 2
        # desplazamiento horizontal de los pies en "estaturas": (u*W)/(alt*H), metros ~ * 0.93*1.65
        r["desp"] = (r["u"] * r["W"]) / (r["alt_m"] * r["H"]) * (0.93 * 1.65) if (r["u"] is not None and r["alt_m"]) else None
        r["mitad"] = "AJUSTE" if r["torneo"] in TORN_AJ else "PRUEBA"
        out.append(r)
    return out


def X_de(rs, con_desp):
    a = np.array([np.tan(np.radians(r["ang"])) for r in rs])
    if con_desp:
        d = np.array([r["desp"] if r["desp"] is not None else 0.0 for r in rs])
        return np.c_[a, d]
    return a[:, None]


def ajusta(refs, con_desp, robusto):
    X = X_de(refs, con_desp)
    y = np.array([r["x_prod"] for r in refs])
    m = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=500) if robusto else LinearRegression()
    m.fit(X, y)
    return m


def loo(refs, con_desp, robusto):
    errs = []
    for i in range(len(refs)):
        tr = refs[:i] + refs[i + 1:]
        try:
            m = ajusta(tr, con_desp, robusto)
        except Exception:
            errs.append(np.nan); continue
        errs.append(float(m.predict(X_de([refs[i]], con_desp))[0] - refs[i]["x_prod"]))
    return np.array(errs)


def zona(x, m=2.0):
    return 0 if x < -5 else 4 if x > 5 else 1 if x < -m else 3 if x > m else 2


if __name__ == "__main__":
    rs = carga()
    mitad = sys.argv[1] if len(sys.argv) > 1 else "AJUSTE"
    rs = [r for r in rs if r["mitad"] == mitad]
    por = collections.defaultdict(list)
    for r in rs:
        if r["motivo"] == "ok" and r["ang"] is not None and r["x_prod"] is not None:
            if not (r["desp"] is None):
                por[r["torneo"]].append(r)
    print("%s: torneos con referencias %d, referencias %d" % (mitad, len(por), sum(len(v) for v in por.values())))
    for con_desp in (False, True):
        for robusto in (False, True):
            E, Z = [], []
            for t, refs in por.items():
                if len(refs) < 8:
                    continue
                e = loo(refs, con_desp, robusto)
                E += list(e)
                Z += [zona(r["x_prod"] + ei) == zona(r["x_prod"]) for r, ei in zip(refs, e) if np.isfinite(ei)]
            E = np.abs(np.array(E)); E = E[np.isfinite(E)]
            print("desp=%-5s robusto=%-5s n=%5d  |err| mediana %.2f m  p75 %.2f  <1m %.2f  zona=zona_prod %.3f"
                  % (con_desp, robusto, len(E), np.median(E), np.percentile(E, 75), (E < 1).mean(), np.mean(Z)))

"""Familia G(n, e)(+p) sobre los clips SIN homografia de una mitad, contra strip_location y middle."""
import sys, json, numpy as np, collections
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import catalogo as CA
import modelo as MO

ZONAS = CA.ZONAS


def boot(ok_a, ok_b, n=2000, semilla=20261009):
    rng = np.random.default_rng(semilla)
    d = ok_b.astype(float) - ok_a.astype(float)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def modelos(rs):
    por = collections.defaultdict(list)
    for r in rs:
        if r["motivo"] == "ok" and r["ang"] is not None and r["x_prod"] is not None and r["desp"] is not None:
            por[r["torneo"]].append(r)
    out = {}
    for t, refs in por.items():
        if len(refs) < 8:
            continue
        e = MO.loo(refs, True, True)
        m = MO.ajusta(refs, True, True)
        angs = [r["ang"] for r in refs]
        out[t] = {"n": len(refs), "e": float(np.nanmedian(np.abs(e))), "m": m,
                  "amin": min(angs), "amax": max(angs)}
    return out


def predice(rs, mods):
    pred = {}
    for r in rs:
        if r["motivo"] != "sinH" or r["ang"] is None or r["desp"] is None:
            continue
        M = mods.get(r["torneo"])
        if M is None:
            continue
        x = float(M["m"].predict(MO.X_de([r], True))[0])
        pred[r["clip"]] = (x, M, r)
    return pred


def evalua(pred, lab, n, e, p, verboso=False):
    ys, zs = [], []
    for c, (x, M, r) in pred.items():
        if c not in lab:
            continue
        if M["n"] < n or M["e"] > e:
            continue
        if not (M["amin"] - 2 <= r["ang"] <= M["amax"] + 2):
            continue
        if p and (not r["dos_pies"] or abs(x) > 8):
            continue
        ys.append(lab[c]); zs.append(ZONAS[MO.zona(x)])
    ys, zs = np.array(ys), np.array(zs)
    if not len(ys):
        return None
    okz = zs == ys; okm = ys == "middle"
    lo, hi = boot(okm, okz)
    return {"n": len(ys), "zona": okz.mean(), "middle": okm.mean(), "dif": okz.mean() - okm.mean(), "lo": lo, "hi": hi}


if __name__ == "__main__":
    mitad = sys.argv[1]
    rs = [r for r in MO.carga() if r["mitad"] == mitad]
    df = CA.manifiesto().set_index("clip_id")
    lab = {r["clip"]: df.loc[r["clip"], "strip_location"] for r in rs}
    sinH = [r for r in rs if r["motivo"] == "sinH"]
    print("%s: sin homografia %d; con angulo %d; con angulo y desplazamiento %d" % (
        mitad, len(sinH), sum(r["ang"] is not None for r in sinH), sum(r["ang"] is not None and r["desp"] is not None for r in sinH)))
    mods = modelos(rs)
    pred = predice(rs, mods)
    print("torneos con modelo %d; clips sin homografia con prediccion %d" % (len(mods), len(pred)))
    reglas = [(n, e, p) for n in (8, 15, 30) for e in (1.0, 1.5, 2.0, np.inf) for p in (False, True)]
    if len(sys.argv) > 2:
        n, e, p = json.loads(sys.argv[2]); reglas = [(n, e if e is not None else np.inf, p)]
    for n, e, p in reglas:
        R = evalua(pred, lab, n, e, p)
        if R is None:
            print("G(%d,%s)%s  sin clips" % (n, e, "+p" if p else "")); continue
        print("G(%2d,%4s)%-2s nuevos %4d  zona %.3f  middle %.3f  dif %+.3f  IC95 [%+.3f, %+.3f]" % (
            n, e, "+p" if p else "", R["n"], R["zona"], R["middle"], R["dif"], R["lo"], R["hi"]))

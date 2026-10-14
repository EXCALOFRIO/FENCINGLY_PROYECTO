"""Cobertura del scouting v2 (homografia por cuadro de la etapa 11, `pista_h_`) en los 144 clips de los dos
ensayos del 14/10, separada por si la etapa 7 tenia homografia. Solo lee."""
import csv, json, pathlib
import numpy as np
E0 = pathlib.Path(r"D:\FENCINGLY_SCRATCH\ensayo")
out = {"clips": 0, "con_pista_h": 0, "frac_h": [], "sin_h7": {"n": 0, "frac_h": []}, "con_h7": {"n": 0, "frac_h": []}}
for e in ("e_1014_0011", "e_1014_1808"):
    E = E0 / e
    for c in open(E / "clips.txt", encoding="utf-8").read().split():
        pc = E / "pista" / ("posicion_clip_%s.csv" % c)
        if not pc.is_file():
            continue
        out["clips"] += 1
        sinh = list(csv.DictReader(open(pc, encoding="utf-8")))[0]["sin_homografia"] == "True"
        ph = E / "trabajo" / ("pista_h_%s.npz" % c)
        if ph.is_file():
            z = np.load(ph, allow_pickle=True)
            ok = np.isfinite(z["H"].reshape(len(z["H"]), -1)).all(1) & (z["conf"] >= float(z["min_conf"]))
            f = float(ok.mean()) if len(ok) else 0.0
            out["con_pista_h"] += 1
        else:
            f = 0.0
        out["frac_h"].append(f)
        out["sin_h7" if sinh else "con_h7"]["n"] += 1
        out["sin_h7" if sinh else "con_h7"]["frac_h"].append(f)
def res(v):
    v = np.array(v)
    return {"n": int(len(v)), "media": round(float(v.mean()), 3) if len(v) else None,
            "mediana": round(float(np.median(v)), 3) if len(v) else None,
            "clips_con_>=50%_cuadros": int((v >= 0.5).sum()), "clips_con_0": int((v == 0).sum())}
r = {"clips": out["clips"], "con_pista_h": out["con_pista_h"], "todos": res(out["frac_h"]),
     "sin_h_en_la_7": res(out["sin_h7"]["frac_h"]), "con_h_en_la_7": res(out["con_h7"]["frac_h"])}
pathlib.Path(__file__).with_suffix(".json").write_text(json.dumps(r, indent=1), encoding="utf-8")
print(json.dumps(r, indent=1))

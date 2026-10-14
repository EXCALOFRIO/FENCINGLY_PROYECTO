"""Validacion del scouting v2 con los criterios declarados en `SCOUTING_V2_2026-10-14.md` (14/10 21:20).
Valida los anotados de florete que ya tienen su `zona_` regenerada en el scratch (orden al azar)."""
import csv, json, pathlib, sys
import numpy as np
RAIZ = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO")
sys.path.insert(0, str(RAIZ / "01_extraccion" / "codigo"))
import catalogo as CA, rutas as RU, zona_tocado as ZT, scouting_v2 as SV  # noqa: E402
S = pathlib.Path(r"C:\Users\Alejandro\FENCINGLY_TMP\scout_h\trabajo")
AQUI = pathlib.Path(r"D:\FENCINGLY_SCRATCH\scout")
REAL_P = RU.PISTA
df = CA.manifiesto().set_index("clip_id")
orden = open(AQUI / "clips_h.txt", encoding="utf-8").read().split()
clips = [c for c in orden if (S / ("zona_%s.csv" % c)).is_file()]


def filas(p):
    try:
        return list(csv.DictReader(open(p, encoding="utf-8")))
    except Exception:
        return None


def x7_tocado(c, tf):
    se = filas(REAL_P / ("serie_eventos_%s.csv" % c))
    if not se or tf is None:
        return None
    xs = []
    for lado in ("0", "1"):
        best = None
        for r in se:
            if r["lado"] != lado or not r.get("x_inicio_m") or not r.get("x_fin_m"):
                continue
            a, b = int(r["cuadro_inicio"]), int(r["cuadro_fin"])
            dist = 0 if a <= tf <= b else min(abs(tf - a), abs(tf - b))
            x = (float(r["x_fin_m"]) if tf >= b else float(r["x_inicio_m"]) if tf <= a
                 else (float(r["x_inicio_m"]) + float(r["x_fin_m"])) / 2)
            if best is None or dist < best[0]:
                best = (dist, x)
        if best is not None and best[0] <= 15:
            xs.append(best[1])
    return float(np.mean(xs)) if xs else None


fil, fis_x, fis_sep, fis_v, fis_dx, desp = [], [], [], [], [], []
for c in clips:
    pc = filas(REAL_P / ("posicion_clip_%s.csv" % c))
    h7 = bool(pc) and pc[0].get("sin_homografia") == "False"
    z11 = filas(S / ("zona_%s.csv" % c))[0]
    d = SV.posiciones(c, S)
    f = {"clip": c, "torneo": df.loc[c, "torneo"], "verdad": df.loc[c, "strip_location"] or None, "h7": h7,
         "z11": z11["zona"] if z11.get("motivo") == "ok" else None, "v2": None, "v2_vent": None, "cubierto": False,
         "con_pista_h": d is not None}
    tf11 = int(z11["cuadro_usado"] or z11["cuadro_tocado"]) if z11.get("cuadro_tocado") else None
    x7 = x7_tocado(c, tf11)
    f["z7"] = ZT.ZONAS[ZT.zona(x7)] if x7 is not None else None
    if d is not None:
        SV.escribe(d, S)
        r = SV.resumen(d)
        f.update(v2=r["zona"], v2_vent=r["zona_vent5"], cubierto=bool(r["frac_ambos"] is not None and r["frac_ambos"] >= 0.5),
                 frac_ambos=r["frac_ambos"])
        u = d["util"]; e = d["estado"]; ok = (e >= 0) & u[:, None]
        fis_x.append(np.abs(d["x"][ok]))
        both = ok[:, 0] & ok[:, 1]
        fis_sep.append(d["sep"][both])
        v = d["vel"][np.isfinite(d["vel"])]
        fis_v.append(np.abs(v))
        for s in (0, 1):
            i = np.flatnonzero((e[:, s] == 0) & u); j = i[1:][np.diff(i) == 1]
            fis_dx.append(np.abs(d["x"][j, s] - d["x"][j - 1, s]))
        se = filas(REAL_P / ("serie_eventos_%s.csv" % c)) or []
        for ev in se:
            if ev["tipo"] not in ("marcha", "romper", "alto", "fondo") or ev["lado"] not in ("0", "1"):
                continue
            if not ev.get("x_inicio_m") or not ev.get("x_fin_m"):
                continue
            s, a, b = int(ev["lado"]), int(ev["cuadro_inicio"]), int(ev["cuadro_fin"])
            if b >= len(e) or e[a, s] < 0 or e[b, s] < 0:
                continue
            desp.append((ev["tipo"], float(ev["x_fin_m"]) - float(ev["x_inicio_m"]), float(d["x"][b, s] - d["x"][a, s])))
    fil.append(f)

n = len(fil)
res = {"validados": n, "de": len(orden), "con_pista_h": sum(f["con_pista_h"] for f in fil)}
cub = sum(f["cubierto"] for f in fil); h7 = sum(f["h7"] for f in fil); uni = sum(f["cubierto"] or f["h7"] for f in fil)
res["A"] = {"cubiertos_v2": cub / n, "con_h_etapa7": h7 / n, "union": uni / n, "pasa": cub / n >= h7 / n}
X, SEP, V, DX = (np.concatenate(a) if a else np.zeros(0) for a in (fis_x, fis_sep, fis_v, fis_dx))
res["B"] = {"B1_x_en_pista": float((X <= 7.5).mean()), "B2_sep_ok": float(((SEP >= 0.5) & (SEP <= 14)).mean()),
            "B3_v_ok": float((V <= 6).mean()), "B4_dx_ok": float((DX <= 0.3).mean()),
            "n": {"posiciones": int(X.size), "cuadros_con_dos": int(SEP.size), "velocidades": int(V.size), "pares_seguidos": int(DX.size)}}
res["B"]["pasa"] = bool(res["B"]["B1_x_en_pista"] >= 0.97 and res["B"]["B2_sep_ok"] >= 0.98 and res["B"]["B3_v_ok"] >= 0.99
                        and res["B"]["B4_dx_ok"] >= 0.995)
V_ = [f for f in fil if f["verdad"]]
tor = np.array([f["torneo"] for f in V_])
def acierto(k):
    return np.array([(f[k] or "middle") == f["verdad"] for f in V_], float)
a2, a2v, a11, a7 = acierto("v2"), acierto("v2_vent"), acierto("z11"), acierto("z7")
amid = np.array([f["verdad"] == "middle" for f in V_], float)
u = np.unique(tor); rng = np.random.default_rng(0); B = {"v2-11": [], "v2-mid": [], "v2-7": []}
grupos = {t: np.flatnonzero(tor == t) for t in u}
for _ in range(2000):
    m = np.concatenate([grupos[t] for t in rng.choice(u, len(u))])
    B["v2-11"].append(a2[m].mean() - a11[m].mean()); B["v2-mid"].append(a2[m].mean() - amid[m].mean())
    B["v2-7"].append(a2[m].mean() - a7[m].mean())
ic = {k: [float(x) for x in np.percentile(v, [2.5, 97.5])] for k, v in B.items()}
res["C"] = {"n_con_verdad": len(V_), "torneos": len(u), "acierto_v2": a2.mean(), "acierto_v2_vent5": a2v.mean(),
            "acierto_etapa11": a11.mean(), "acierto_etapa7": a7.mean(), "acierto_middle": amid.mean(),
            "contestan": {"v2": sum(1 for f in V_ if f["v2"]), "etapa11": sum(1 for f in V_ if f["z11"]),
                          "etapa7": sum(1 for f in V_ if f["z7"])}, "ic95": ic,
            "C1": bool(a2.mean() >= a11.mean() - 0.01), "C2": bool(ic["v2-mid"][0] > 0), "C3": bool(ic["v2-7"][0] > 0),
            "C4": bool(a2v.mean() >= a2.mean() - 0.02)}
res["C"]["pasa"] = all(res["C"][k] for k in ("C1", "C2", "C3", "C4"))
D = {}
for t in ("marcha", "romper", "alto", "fondo", "todos"):
    L = np.array([(x, y) for tt, x, y in desp if t == "todos" or tt == t])
    if not len(L):
        continue
    D[t] = {"n": len(L), "mismo_sentido": float(np.mean(np.sign(L[:, 0]) == np.sign(L[:, 1]))),
            "dif_p50": float(np.median(np.abs(L[:, 0] - L[:, 1]))), "corr": float(np.corrcoef(L[:, 0], L[:, 1])[0, 1]),
            "spearman": float(np.corrcoef(np.argsort(np.argsort(L[:, 0])), np.argsort(np.argsort(L[:, 1])))[0, 1]),
            "frac_dif_mas_1m": float(np.mean(np.abs(L[:, 0] - L[:, 1]) > 1.0)),
            "frac_7_mas_3m": float(np.mean(np.abs(L[:, 0]) > 3.0)), "frac_v2_mas_3m": float(np.mean(np.abs(L[:, 1]) > 3.0))}
res["D"] = D
res["D_pasa"] = bool(D.get("todos") and D["todos"]["mismo_sentido"] >= 0.85 and D["todos"]["dif_p50"] <= 0.25)
res["PASA"] = bool(res["A"]["pasa"] and res["B"]["pasa"] and res["C"]["pasa"] and res["D_pasa"])
(AQUI / "valida.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
with open(AQUI / "valida_por_clip.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=sorted({k for f in fil for k in f})); w.writeheader(); w.writerows(fil)
print(json.dumps(res, indent=1, ensure_ascii=False))

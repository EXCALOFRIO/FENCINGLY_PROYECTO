"""Pregunta 2 (14/10): O6, la etapa 4 solo en [tocado - A s, tocado + 1 s], con A = 6, 8 y 10. Solo lee
`revision_` (mask, pres, fps, T), `zona_` / `lights_` (cuadro del tocado) y `serie_eventos_` (cuadro de
inicio de cada evento). La unidad de coste de la etapa 4 es la PASADA (un cuadro util con al menos una
caja: una pasada de Sapiens con 1 o 2 recortes); se da tambien en recortes."""
import csv, collections, json, pathlib, sys
import numpy as np
RAIZ = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO")
sys.path.insert(0, str(RAIZ / "01_extraccion" / "codigo"))
import catalogo as CA, rutas as RU  # noqa: E402
df = CA.manifiesto().set_index("clip_id")
T, P = RU.TRABAJO, RU.PISTA
ANTES = (6, 8, 10)
DESPUES = 1.0


def fila(p):
    try:
        with open(p, encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return None


def tocado(c):
    z = fila(T / ("zona_%s.csv" % c))
    if z and z[0].get("cuadro_tocado") not in (None, ""):
        return int(float(z[0]["cuadro_tocado"])), "zona"
    l = fila(T / ("lights_%s.csv" % c))
    v = [int(float(r["cuadro_encendido"])) for r in (l or []) if r.get("cuadro_encendido") not in (None, "")]
    return (min(v), "lights") if v else (None, None)


por = collections.defaultdict(lambda: {"clips": 0, "sin_tocado": 0, "pas_tot": 0, "rec_tot": 0,
                                        **{"pas_%d" % a: 0 for a in ANTES}, **{"rec_%d" % a: 0 for a in ANTES},
                                        "frac": {a: [] for a in ANTES}, "pas_sin_tocado": 0})
ev = collections.defaultdict(lambda: collections.Counter())
for p in T.glob("revision_*.npz"):
    c = p.name[9:-4]
    if c not in df.index:
        continue
    w = df.loc[c, "weapon"]
    try:
        z = np.load(p, allow_pickle=True)
        mask, pres = z["mask"].astype(bool), z["pres"].astype(bool)
        fps = float(z["fps"]) if "fps" in z.files and float(z["fps"]) > 0 else None
        TT = int(z["T"])
    except Exception:
        continue
    if fps is None:
        continue
    pasada = mask & pres.any(axis=1)
    recortes = (mask[:, None] & pres).sum(axis=1)
    for g in (w, "todas"):
        por[g]["clips"] += 1
        por[g]["pas_tot"] += int(pasada.sum())
        por[g]["rec_tot"] += int(recortes.sum())
    tf, _ = tocado(c)
    if tf is None:
        for g in (w, "todas"):
            por[g]["sin_tocado"] += 1
            por[g]["pas_sin_tocado"] += int(pasada.sum())
        continue
    t = np.arange(len(mask))
    for a in ANTES:
        dentro = (t >= tf - a * fps) & (t <= tf + DESPUES * fps)
        pk, rk = int((pasada & dentro).sum()), int((recortes * dentro).sum())
        for g in (w, "todas"):
            por[g]["pas_%d" % a] += pk
            por[g]["rec_%d" % a] += rk
            if pasada.sum():
                por[g]["frac"][a].append(pk / pasada.sum())
    se = fila(P / ("serie_eventos_%s.csv" % c))
    for r in se or []:
        if r.get("cuadro_inicio") in (None, ""):
            continue
        ci = int(r["cuadro_inicio"])
        for a in ANTES:
            fuera = not (tf - a * fps <= ci <= tf + DESPUES * fps)
            ev[(w, r["tipo"], a)]["fuera" if fuera else "dentro"] += 1
            ev[("todas", r["tipo"], a)]["fuera" if fuera else "dentro"] += 1
res = {}
for g, d in por.items():
    o = {"clips": d["clips"], "sin_tocado": d["sin_tocado"],
         "pasadas_en_clips_sin_tocado_frac": round(d["pas_sin_tocado"] / d["pas_tot"], 4) if d["pas_tot"] else None}
    for a in ANTES:
        # se ahorra en los clips CON tocado; los sin tocado se procesan enteros (supuesto)
        o["ventana_-%ds" % a] = {
            "pasadas_que_quedan_frac": round((d["pas_%d" % a] + d["pas_sin_tocado"]) / d["pas_tot"], 4),
            "recortes_que_quedan_frac": round(d["rec_%d" % a] / max(d["rec_tot"], 1) + d["pas_sin_tocado"] / max(d["pas_tot"], 1), 4),
            "por_clip_mediana_que_queda": round(float(np.median(d["frac"][a])), 3) if d["frac"][a] else None,
            "por_clip_media_que_queda": round(float(np.mean(d["frac"][a])), 3) if d["frac"][a] else None}
    res[g] = o
evs = {}
for (g, tipo, a), cnt in ev.items():
    n = cnt["dentro"] + cnt["fuera"]
    evs.setdefault(g, {}).setdefault(tipo, {})["-%ds" % a] = {"n": n, "fuera_frac": round(cnt["fuera"] / n, 3) if n else None}
res["eventos_que_empiezan_fuera"] = evs
# a dias: 1+4 en paralelo ~116 clips/h = 31,0 s/clip; la etapa 4 es ~76-80 % de 1+4 (7 s contra 22-29 s)
CLIPS = 86000 + 19500
s_clip = 3600 / 116
dias = {}
for a in ANTES:
    q = res["todas"]["ventana_-%ds" % a]["pasadas_que_quedan_frac"]
    dias["-%ds" % a] = {"etapa4_queda": q,
                         "ahorro_dias_si_4_es_76%": round(CLIPS * s_clip * 0.76 * (1 - q) / 86400, 1),
                         "ahorro_dias_si_4_es_80%": round(CLIPS * s_clip * 0.80 * (1 - q) / 86400, 1)}
res["dias"] = dias
pathlib.Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "eventos_que_empiezan_fuera"}, indent=1, ensure_ascii=False))
for tipo, d in sorted(res["eventos_que_empiezan_fuera"].get("todas", {}).items()):
    print("%-8s" % tipo, {k: v["fuera_frac"] for k, v in sorted(d.items())}, "n", d["-6s"]["n"])

"""Auditoria de calidad (lectura SOLO, nada se escribe/borra/mueve en el proyecto).

Checks 1,2,3,6,7 del encargo: convenio lado/fotl vs 'initiated', lights vs 'touch' del
manifiesto, cuadro del tocado vs fin de clip, duplicados, valores imposibles en
serie_eventos. Usa unicamente ficheros ya en disco (lights_, zona_, poses_ en TRABAJO;
serie_eventos_ en PISTA; manifiesto.csv.gz). No toca GPU, no mueve/borra nada.
"""
import sys, os, csv, collections, json
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import rutas as RU

TRABAJO = RU.TRABAJO
PISTA = RU.PISTA
MANIFEST = RU.CATALOGO / "manifiesto.csv.gz"
OUT = r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop-FENCINGLY-AI\2c75119e-09b9-43eb-a841-faa6869534b2\scratchpad\calidad"

print("TRABAJO=", TRABAJO, "existe", TRABAJO.exists())
print("PISTA=", PISTA, "existe", PISTA.exists())

man = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False, na_values=[])
man = man.set_index("clip_id")
print("manifiesto:", len(man), "clips")

poses_files = sorted(f for f in os.listdir(TRABAJO) if f.startswith("poses_") and f.endswith(".npz"))
clip_ids = [f[len("poses_"):-4] for f in poses_files]
print("clips con poses_:", len(clip_ids))

# ---------------------------------------------------------------
# 6a. duplicados: mismo clip_id dos veces en el manifiesto; misma url duplicada
# ---------------------------------------------------------------
dup_ids = man.index[man.index.duplicated(keep=False)]
print("\n=== 6. duplicados ===")
print("clip_id duplicado en manifiesto:", dup_ids.nunique())
dup_url = man.loc[man.index.isin(clip_ids)]
url_counts = dup_url["url"].value_counts()
dup_urls = url_counts[url_counts > 1]
print("urls duplicadas entre los %d clips extraidos: %d" % (len(dup_url), len(dup_urls)))
if len(dup_urls):
    for u, n in dup_urls.head(10).items():
        ids = dup_url.index[dup_url["url"] == u].tolist()
        print("   ", u, "->", ids)

# duplicados por ruta de video real (campo 'video' dentro de poses_<clip>.npz), y T,W,H para 3/4/5
video_to_clips = collections.defaultdict(list)
clip_info = {}
for cid, fn in zip(clip_ids, poses_files):
    p = TRABAJO / fn
    try:
        z = np.load(p, allow_pickle=True)
        T = int(z["T"]); W = int(z["W"]); H = int(z["H"])
        video = str(z["video"])
    except Exception as ex:
        print("ERROR leyendo", fn, ex)
        continue
    clip_info[cid] = {"T": T, "W": W, "H": H, "video": video}
    video_to_clips[video].append(cid)

dup_video = {v: cs for v, cs in video_to_clips.items() if len(cs) > 1 and v}
print("mismo fichero de video fisico bajo >1 clip_id:", len(dup_video))
for v, cs in list(dup_video.items())[:10]:
    print("   ", v, "->", cs)

with open(os.path.join(OUT, "dup_video.json"), "w", encoding="utf-8") as fh:
    json.dump(dup_video, fh, ensure_ascii=False, indent=1)

# ---------------------------------------------------------------
# resolucion por clip, para el check 5 (lo mide chk_poses.py con KP/SC; aqui solo guardamos W,H)
# ---------------------------------------------------------------
res_count = collections.Counter((v["W"], v["H"]) for v in clip_info.values())
print("\nresoluciones (W,H) en los clips extraidos:")
for (w, h), n in res_count.most_common(10):
    print("   %dx%d : %d" % (w, h, n))

# ---------------------------------------------------------------
# 3. cuadro del tocado vs fin del clip (zona_<clip>.csv, cuadro_tocado, contra T de poses_)
# ---------------------------------------------------------------
print("\n=== 3. cuadro_tocado vs T (fin del clip) ===")
fracs = []
sospechosos = []
n_zona = 0
n_motivo_ok = 0
for cid in clip_ids:
    p = TRABAJO / ("zona_%s.csv" % cid)
    if not p.is_file() or p.stat().st_size == 0:
        continue
    with open(p, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        continue
    n_zona += 1
    r = rows[0]
    if r.get("motivo") != "ok":
        continue
    ct = r.get("cuadro_tocado")
    if not ct:
        continue
    n_motivo_ok += 1
    T = clip_info.get(cid, {}).get("T")
    if not T:
        continue
    frac = float(ct) / T
    fracs.append(frac)
    if frac < 0.2 or frac > 1.0:
        sospechosos.append((cid, float(ct), T, frac))

fracs = np.array(fracs)
print("clips con zona_ csv no vacio: %d; con motivo=='ok' y cuadro_tocado: %d" % (n_zona, n_motivo_ok))
if len(fracs):
    for p in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        print("   p%-3d frac=%.3f" % (p, np.percentile(fracs, p)))
    print("   media", fracs.mean(), "n", len(fracs))
    print("   frac<0.2 (tocado en la primera quinta parte del clip): %d (%.4f)"
          % ((fracs < 0.2).sum(), (fracs < 0.2).mean()))
    print("   frac<0.5: %d (%.4f)" % ((fracs < 0.5).sum(), (fracs < 0.5).mean()))
    print("   frac>1.0 (tocado fuera del clip, deberia ser imposible): %d" % (fracs > 1.0).sum())
print("peores ejemplos (frac mas bajo o >1):")
sospechosos.sort(key=lambda x: x[3])
for cid, ct, T, frac in sospechosos[:15]:
    print("   %-40s cuadro_tocado=%6.0f  T=%5d  frac=%.3f" % (cid, ct, T, frac))

# ---------------------------------------------------------------
# 2. lights_<clip>.csv lado vs manifiesto['touch'], por torneo y por via
# ---------------------------------------------------------------
print("\n=== 2. lights (lado L/R) vs manifiesto['touch'] ===")
TOL = 3  # cuadros de tolerancia para agrupar "el mismo encendido"
resultado = []  # cid, torneo, via, pred, touch, match
for cid in clip_ids:
    p = TRABAJO / ("lights_%s.csv" % cid)
    if not p.is_file() or p.stat().st_size == 0:
        continue
    with open(p, encoding="utf-8") as fh:
        rr = csv.DictReader(fh)
        if "canal" not in (rr.fieldnames or []):
            continue
        rows = list(rr)
    on = [r for r in rows if r["canal"] in ("roja", "verde") and r.get("cuadro_encendido")]
    if not on:
        continue
    ce = [float(r["cuadro_encendido"]) for r in on]
    cmin = min(ce)
    lados = sorted(set(r["lado"] for r, c in zip(on, ce) if c <= cmin + TOL))
    if lados == ["L"]:
        pred = "left"
    elif lados == ["R"]:
        pred = "right"
    elif lados == ["L", "R"]:
        pred = "double"
    else:
        continue
    if cid not in man.index:
        continue
    touch = man.loc[cid, "touch"]
    if touch not in ("left", "right", "double"):
        continue
    torneo = man.loc[cid, "torneo"]
    via = on[0]["via"]
    resultado.append((cid, torneo, via, pred, touch, pred == touch))

res_df = pd.DataFrame(resultado, columns=["clip", "torneo", "via", "pred", "touch", "match"])
print("clips comparables (luz on-target + touch left/right/double en manifiesto): %d" % len(res_df))
if len(res_df):
    print("acierto global: %.4f (n=%d)" % (res_df["match"].mean(), len(res_df)))
    print("matriz pred x touch:")
    print(pd.crosstab(res_df["pred"], res_df["touch"]))
    print("\nacierto por via:")
    print(res_df.groupby("via")["match"].agg(["mean", "count"]))
    print("\nacierto por torneo (torneos con >=15 clips comparables):")
    g = res_df.groupby("torneo")["match"].agg(["mean", "count"]).sort_values("mean")
    g = g[g["count"] >= 15]
    print(g.to_string())
    res_df.to_csv(os.path.join(OUT, "lights_vs_touch.csv"), index=False)

# ---------------------------------------------------------------
# 1. lado (0=izq=fotl) vs 'initiated' (fotl/fotr), usando el primer evento de ataque
# ---------------------------------------------------------------
print("\n=== 1. lado 0/1 vs 'initiated' (fotl/fotr) usando el primer evento de ataque ===")
ATAQUE = {"marcha", "brazo", "fondo"}
res1 = []
for cid in clip_ids:
    if cid not in man.index:
        continue
    ini = man.loc[cid, "initiated"]
    if ini not in ("fotl", "fotr"):
        continue
    p = PISTA / ("serie_eventos_%s.csv" % cid)
    if not p.is_file() or p.stat().st_size == 0:
        continue
    with open(p, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    primero = {0: None, 1: None}
    for r in rows:
        if r.get("tipo") not in ATAQUE:
            continue
        if r.get("lado_valido") == "False":
            continue
        try:
            lado = int(r["lado"])
        except (ValueError, TypeError):
            continue
        if lado not in (0, 1):
            continue
        try:
            ci = float(r["cuadro_inicio"])
        except (ValueError, TypeError):
            continue
        if primero[lado] is None or ci < primero[lado]:
            primero[lado] = ci
    if primero[0] is None or primero[1] is None:
        continue
    pred = "fotl" if primero[0] < primero[1] else ("fotr" if primero[1] < primero[0] else None)
    if pred is None:
        continue
    torneo = man.loc[cid, "torneo"]
    res1.append((cid, torneo, pred, ini, pred == ini))

res1_df = pd.DataFrame(res1, columns=["clip", "torneo", "pred", "initiated", "match"])
print("clips comparables: %d" % len(res1_df))
if len(res1_df):
    print("acierto global (predicho por orden de eventos vs 'initiated' humano): %.4f (n=%d)"
          % (res1_df["match"].mean(), len(res1_df)))
    print("(referencia: techo humano publicado para 'initiated' = 0.688; acierto~0.5 es azar)")
    print("\nacierto por torneo (torneos con >=15 clips comparables):")
    g = res1_df.groupby("torneo")["match"].agg(["mean", "count"]).sort_values("mean")
    g = g[g["count"] >= 15]
    print(g.to_string())
    res1_df.to_csv(os.path.join(OUT, "lado_vs_initiated.csv"), index=False)

# ---------------------------------------------------------------
# 7. valores imposibles en serie_eventos_<clip>.csv, por torneo
# ---------------------------------------------------------------
print("\n=== 7. valores imposibles en serie_eventos (duracion<0, v>8 m/s, |x|>8 m) ===")
bad_dur = []
bad_vel = []
bad_x = []
n_rows_total = 0
n_files = 0
for cid in clip_ids:
    p = PISTA / ("serie_eventos_%s.csv" % cid)
    if not p.is_file() or p.stat().st_size == 0:
        continue
    torneo = man.loc[cid, "torneo"] if cid in man.index else ""
    n_files += 1
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            n_rows_total += 1
            try:
                d = float(r["duracion_s"]) if r.get("duracion_s") else None
            except ValueError:
                d = None
            if d is not None and d < 0:
                bad_dur.append((cid, torneo, r["tipo"], d))
            try:
                v = float(r["velocidad_m_s"]) if r.get("velocidad_m_s") else None
            except ValueError:
                v = None
            if v is not None and v > 8:
                bad_vel.append((cid, torneo, r["tipo"], v))
            for campo in ("x_inicio_m", "x_fin_m"):
                try:
                    x = float(r[campo]) if r.get(campo) else None
                except ValueError:
                    x = None
                if x is not None and abs(x) > 8:
                    bad_x.append((cid, torneo, r["tipo"], campo, x))

print("ficheros serie_eventos leidos: %d, filas totales: %d" % (n_files, n_rows_total))
print("duracion_s < 0: %d" % len(bad_dur))
for row in sorted(bad_dur, key=lambda x: x[3])[:10]:
    print("   ", row)
print("velocidad_m_s > 8: %d" % len(bad_vel))
for row in sorted(bad_vel, key=lambda x: -x[3])[:10]:
    print("   ", row)
print("|x_inicio_m| o |x_fin_m| > 8: %d" % len(bad_x))
for row in sorted(bad_x, key=lambda x: -abs(x[4]))[:10]:
    print("   ", row)

print("\npor torneo (duracion<0 + vel>8 + x>8 juntos), torneos con >=2 incidencias:")
c = collections.Counter(t for _, t, *_ in bad_dur + bad_vel + bad_x)
tot_por_torneo = man.loc[man.index.isin(clip_ids), "torneo"].value_counts()
for t, n in c.most_common(20):
    print("   %-28s incidencias=%-4d clips_de_ese_torneo=%d" % (t, n, tot_por_torneo.get(t, 0)))

print("\nFIN chk_main.py")

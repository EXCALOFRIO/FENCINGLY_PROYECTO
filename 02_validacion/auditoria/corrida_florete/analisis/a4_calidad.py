"""Pregunta 3 (14/10): comprobaciones baratas de lo que ya hay en `pruebas` y `pista`. Solo lee (de cada
npz solo los arrays que hacen falta)."""
import csv, collections, json, pathlib, sys
import numpy as np
RAIZ = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO")
sys.path.insert(0, str(RAIZ / "01_extraccion" / "codigo"))
import catalogo as CA, rutas as RU  # noqa: E402
df = CA.manifiesto().set_index("clip_id")
T, P = RU.TRABAJO, RU.PISTA
res = {}
por = collections.defaultdict(lambda: collections.defaultdict(list))
for p in T.glob("poses_*.npz"):
    c = p.name[6:-4]
    if c not in df.index:
        continue
    w = df.loc[c, "weapon"]
    try:
        with np.load(p, allow_pickle=True) as z:
            org, mask = z["origen"], z["mask"].astype(bool)
    except Exception:
        por[w]["ilegible"].append(1)
        continue
    m = mask.sum()
    if not m:
        por[w]["sin_cuadros_utiles"].append(1)
        continue
    o = org[mask]
    por[w]["util_frac"].append(m / len(mask))
    por[w]["dos_poses"].append(float(((o >= 0).sum(1) == 2).mean()))
    por[w]["dos_medidas"].append(float(((o == 0).sum(1) == 2).mean()))
    por[w]["inventadas"].append(float(np.isin(o, (1, 2)).mean()))
for p in T.glob("revision_*.npz"):
    c = p.name[9:-4]
    if c not in df.index:
        continue
    w = df.loc[c, "weapon"]
    try:
        with np.load(p, allow_pickle=True) as z:
            tipo, mask, pres = z["tipo"], z["mask"].astype(bool), z["pres"].astype(bool)
    except Exception:
        continue
    if mask.sum():
        por[w]["redetectado"].append(float(((tipo[mask] == 3) & pres[mask]).any(1).mean()))
for w, d in por.items():
    o = {}
    for k, v in d.items():
        v = np.array(v, float)
        if k in ("ilegible", "sin_cuadros_utiles"):
            o[k] = int(len(v)); continue
        o[k] = {"n": int(len(v)), "media": round(float(v.mean()), 3), "p10": round(float(np.percentile(v, 10)), 3),
                "mediana": round(float(np.median(v)), 3)}
    if "dos_poses" in d:
        o["clips_con_menos_de_50%_dos_poses"] = int((np.array(d["dos_poses"]) < 0.5).sum())
    if "redetectado" in d:
        o["clips_con_algun_cuadro_redetectado"] = int((np.array(d["redetectado"]) > 0).sum())
    res[w] = o
# series: clips sin eventos de movimiento, valores imposibles
ser = collections.Counter(); imposibles = collections.Counter(); filas_pub = 0
for p in P.glob("serie_eventos_*.csv"):
    c = p.name[len("serie_eventos_"):-4]
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    tipos = {r["tipo"] for r in rows}
    ser["clips"] += 1
    if not rows:
        ser["0 filas"] += 1
    elif not (tipos & {"marcha", "romper", "fondo", "alto"}):
        ser["sin marcha/romper/fondo/alto"] += 1
    for r in rows:
        pub = r.get("publicable") == "True" and (r.get("fisica_evento") in (None, "", "True"))
        filas_pub += pub
        try:
            v = float(r["velocidad_m_s"]) if r.get("velocidad_m_s") not in (None, "") else None
            x0 = float(r["x_inicio_m"]) + float(r.get("x_origen_m") or 0) if r.get("x_inicio_m") not in (None, "") else None
            dur = float(r["duracion_s"]) if r.get("duracion_s") not in (None, "") else None
            mag = float(r["magnitud_m"]) if r.get("magnitud_m") not in (None, "") and r.get("unidad_magnitud", "").startswith("m") else None
        except ValueError:
            continue
        clave = "publicable" if pub else "no_publicable"
        imposibles[clave + "_filas"] += 1
        if v is not None and abs(v) > 9:
            imposibles[clave + "_v>9m/s"] += 1
        if x0 is not None and abs(x0) > 8:
            imposibles[clave + "_|x|>8m"] += 1
        if dur is not None and dur <= 0:
            imposibles[clave + "_dur<=0"] += 1
        if mag is not None and r["tipo"] in ("marcha", "romper") and abs(mag) > 1.5:
            imposibles[clave + "_paso>1,5m"] += 1
res["series"] = dict(ser)
res["series_filas_publicables_y_fisica_ok"] = filas_pub
res["series_imposibles"] = dict(imposibles)
pathlib.Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False))

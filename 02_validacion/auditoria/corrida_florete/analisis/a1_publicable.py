"""Pregunta 1 (14/10): por que solo ~9 % del scouting sale `publicable`. Solo lee salidas existentes:
`02_validacion/pista/posicion_clip_*.csv`, `serie_eventos_*.csv` y `01_extraccion/pruebas/zona_*.csv`.

`publicable` (pista.py ~2298) = homografia ok Y C1_pasa (mediana AGRUPADA de la estatura reproyectada
de los dos en 1,60-2,00 m) Y n_marcas_afinadas >= 2 Y h_p50 finita. C1 y n_marcas no se guardan en las
salidas de la corrida: se aproxima con `puerta_dos_lados` (las dos estaturas en rango) y se da aparte el
desglose del `homografia.csv` antiguo (78 clips, septiembre)."""
import csv, collections, json, pathlib, sys
RAIZ = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO")
sys.path.insert(0, str(RAIZ / "01_extraccion" / "codigo"))
import catalogo as CA, rutas as RU  # noqa: E402
df = CA.manifiesto().set_index("clip_id")
P, T = RU.PISTA, RU.TRABAJO


def fila(p):
    try:
        with open(p, encoding="utf-8") as fh:
            r = list(csv.DictReader(fh))
        return r
    except Exception:
        return None


B = lambda v: v == "True"
res = {}
filas = []
for p in P.glob("posicion_clip_*.csv"):
    c = p.name[len("posicion_clip_"):-4]
    r = fila(p)
    if not r or c not in df.index:
        continue
    z = fila(T / ("zona_%s.csv" % c))
    r = r[0]
    r["_arma"] = df.loc[c, "weapon"]
    r["_zona_ok"] = bool(z and z[0].get("motivo") == "ok")
    r["_frac_h11"] = float(z[0]["frac_cuadros_con_h"]) if z and z[0].get("frac_cuadros_con_h") not in (None, "") else None
    r["_clip"] = c
    filas.append(r)
vacias = sum(1 for p in T.glob("pista_vacia_*.txt"))


def cascada(F):
    n = len(F)
    out = collections.OrderedDict()
    out["n"] = n
    sinh = [r for r in F if B(r.get("sin_homografia"))]
    conh = [r for r in F if not B(r.get("sin_homografia"))]
    out["1_sin_homografia"] = len(sinh)
    pub = [r for r in conh if B(r.get("publicable"))]
    nopub = [r for r in conh if not B(r.get("publicable"))]
    out["2_con_h_no_publicable"] = len(nopub)
    out["2a_de_esos_dos_estaturas_en_rango"] = sum(1 for r in nopub if B(r.get("puerta_dos_lados")))
    out["2b_de_esos_alguna_estatura_fuera"] = sum(1 for r in nopub if not B(r.get("puerta_dos_lados")))
    out["3_publicable"] = len(pub)
    out["3a_publicable_y_dos_estaturas_en_rango"] = sum(1 for r in pub if B(r.get("puerta_dos_lados")))
    out["3b_publicable_y_pasa_fisica"] = sum(1 for r in pub if B(r.get("pasa_fisica")))
    out["3c_publicable_y_ancla_fiable"] = sum(1 for r in pub if B(r.get("ancla_fiable")))
    out["4_publicable_y_dos_lados_y_fisica"] = sum(1 for r in pub if B(r.get("puerta_dos_lados")) and B(r.get("pasa_fisica")))
    # por separado, sobre los que tienen homografia
    sep = collections.OrderedDict()
    for k in ("puerta_dos_lados", "pasa_fisica", "ancla_fiable", "pasa_dpies", "pasa_fuera", "pasa_transv", "fisica_completa"):
        v = [r.get(k) for r in conh]
        sep[k] = {"True": sum(1 for x in v if x == "True"), "False": sum(1 for x in v if x == "False"),
                  "vacio": sum(1 for x in v if x in (None, ""))}
    out["por_separado_en_los_con_h"] = sep
    # que arreglaria el v2: sin homografia en la 7 pero la 11 si (zona ok)
    out["v2_sin_h7_con_zona11"] = sum(1 for r in sinh if r["_zona_ok"])
    out["v2_sin_h7_sin_zona11"] = sum(1 for r in sinh if not r["_zona_ok"])
    out["v2_con_h7_no_publicable_con_zona11"] = sum(1 for r in nopub if r["_zona_ok"])
    fr = [r["_frac_h11"] for r in sinh if r["_frac_h11"] is not None]
    out["v2_frac_cuadros_h11_en_sin_h7_mediana"] = sorted(fr)[len(fr) // 2] if fr else None
    return out


res["todos"] = cascada(filas)
for w in ("foil", "epee", "sabre"):
    res[w] = cascada([r for r in filas if r["_arma"] == w])
res["pista_vacia_txt"] = vacias

# filas de la serie: motivo_no_publicable y lado_no_valido_por
mot, ladov, tipos_pub = collections.Counter(), collections.Counter(), collections.Counter()
n_filas = 0
for p in P.glob("serie_eventos_*.csv"):
    for r in fila(p) or []:
        n_filas += 1
        mot[r.get("motivo_no_publicable") or "(vacio)"] += 1
        if r.get("lado_valido") == "False":
            ladov[r.get("lado_no_valido_por") or "(vacio)"] += 1
        tipos_pub[(r.get("tipo"), r.get("publicable"))] += 1
res["serie_filas"] = n_filas
res["serie_motivo_no_publicable"] = dict(mot.most_common(15))
res["serie_lado_no_valido_por"] = dict(ladov.most_common(10))
res["serie_tipo_publicable"] = {"%s|%s" % k: v for k, v in sorted(tipos_pub.items(), key=lambda x: -x[1])[:30]}

# el homografia.csv antiguo (78 clips): por que no publicable
old = fila(P / "homografia.csv") or []
oc = collections.Counter()
for r in old:
    if r.get("ok") != "True":
        oc["sin homografia"] += 1
    elif r.get("publicable") == "True":
        oc["publicable"] += 1
    else:
        c1 = r.get("C1_pasa") == "True"
        nm = r.get("n_marcas_afinadas")
        nm = int(float(nm)) if nm not in (None, "") else None
        oc["no publicable: C1 %s, marcas afinadas %s" % ("pasa" if c1 else "NO", ">=2" if (nm or 0) >= 2 else "<2")] += 1
res["homografia_csv_antiguo_78"] = dict(oc)
sal = pathlib.Path(__file__).with_suffix(".json")
sal.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False))

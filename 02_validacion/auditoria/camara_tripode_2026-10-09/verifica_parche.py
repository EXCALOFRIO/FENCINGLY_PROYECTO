"""Verifica el respaldo del tripode en produccion: APAGADO da lo mismo que el zona_ de disco; ENCENDIDO
reproduce el experimento (`fit.py`) en clips que la regla acepta.

    python verifica_parche.py FIT.jsonl apagado|encendido
"""
import csv
import json
import os
import sys

modo = sys.argv[2]
if modo == "encendido":
    os.environ["FENCINGLY_CAMARA_TRIPODE"] = "1"
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import catalogo as CA  # noqa: E402
import rutas as RU  # noqa: E402
import zona_tocado as ZT  # noqa: E402
from evalua import pasa  # noqa: E402

res = {}
for l in open(sys.argv[1], encoding="utf-8"):
    o = json.loads(l)
    res[o["clip"]] = o
df = CA.manifiesto().set_index("clip_id")
regla = ZT.TRIPODE_REGLA
acep = sorted(c for c, o in res.items() if not o.get("prod_ok") and pasa(o, *regla))
noacep = sorted(c for c, o in res.items() if not o.get("prod_ok") and o.get("estado") == "ok" and not pasa(o, *regla))
cont = sorted(c for c, o in res.items() if o.get("prod_ok"))
if modo == "apagado":
    clips = cont[:3] + noacep[:1] + acep[:1]
else:
    clips = acep[:2] + noacep[:1]
print("TRIPODE", ZT.TRIPODE, "regla", regla)
for c in clips:
    r = ZT.analiza(c, RU.video(c), df.loc[c, "torneo"])
    d = next(csv.DictReader(open(RU.TRABAJO / ("zona_%s.csv" % c), encoding="utf-8")))
    campos = ("motivo", "zona", "x_tocado_m", "cuadro_usado", "conf_h", "metodo_h")
    disco = {k: d.get(k) for k in campos}
    nuevo = {k: ("" if r.get(k) is None else str(r.get(k))) for k in campos}
    igual = all((disco[k] or "") == nuevo[k] for k in campos)
    extra = ""
    if r.get("via_h") == "tripode":
        extra = "  experimento x=%.3f zona=%s" % (res[c]["t_x"], res[c]["t_zona"])
    print("%-36s igual_a_disco=%s  via_h=%s  nuevo=%s%s" % (c[:36], igual, r.get("via_h"), nuevo, extra))
    if not igual and modo == "apagado":
        print("      disco=%s" % disco)

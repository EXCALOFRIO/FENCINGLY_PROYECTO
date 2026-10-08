"""ETAPA E, paso 14: QUE EJE PREDICE LA DIRECCION DE LA HOJA. El careo, agrupando todos los clips.

POR QUE HACE FALTA AGRUPAR, Y POR QUE ESO NO ES HACER TRAMPA. `e10` intenta el careo clip por clip y
NO PUEDE: la hoja se detecta en **1 a 3 recortes por lado**, y una mediana de 1 o 3 muestras no es
una medida. Bajar el umbral de n para que «pase» seria exactamente lo que este proyecto tiene
prohibido por escrito. Lo legitimo es lo contrario: **subir el n agrupando**, que es lo que se hace
aqui, y decir cuantas muestras tiene el numero final.

LA PREGUNTA. Cuando la hoja NO se ve —que es el caso normal, el 93-97 % de los cuadros incluso a
1080p— el eje del cono hay que suponerlo desde el cuerpo. Hay tres candidatos y no da igual cual:

    `hombro -> raiz de la mano`     la variable operativa medida del proyecto (`ext_hr`)
    `codo -> raiz de la mano`       lo que decia el encargo, y que es un HUESO
    `raiz -> centroide del pulgar`  lo que se propone como sustituto donde no hay hoja

Y en un mismo cuadro pueden apuntar muy distinto: medido en `ImpeccablePertinentBasilisk` c175
lado 0, `codo -> raiz` da 21,0 grados y `hombro -> raiz` da 51,4. **Treinta grados de diferencia**,
o sea tres veces el techo de utilidad del cono. Elegir mal el eje no es un detalle de estilo.

LA VERDAD-TERRENO. Los cuadros donde la hoja SI se ve en el alfa de `matting` son los unicos con
direccion del arma medida. Contra ella se compara cada candidato con
`dif90` (diferencia entre RECTAS, en [0, 90]).

CRITERIOS DECLARADOS ANTES DE MIRAR:

  1. AZAR. Si un candidato no tiene relacion con la hoja, |diferencia| es uniforme en [0, 90] y su
     mediana vale **45 grados**. Ese es el numero contra el que se compara, y cualquier candidato con
     mediana por encima de ~45 hay que retirarlo.
  2. GANADOR. Gana el de mediana mas baja, y solo se declara ganador si su mediana es **al menos 5
     grados** mejor que la del siguiente. Con menos de 5 grados de separacion y este n, la diferencia
     no se sostiene y se dice que **empatan**.
  3. SOLO LAS DE DELANTE. Se usan unicamente las rectas cuyo punto mas lejano cae DELANTE de la mano
     (`t_max > 0`), que es el filtro de `e6`: una recta detras de la mano es el borde del brazo y
     compararla con el eje del brazo daria un acuerdo artificialmente bueno. Ese descarte es el que
     evita el confuso, y se informa cuantas se caen por el.

LO QUE ESTE PASO NO PUEDE CONTESTAR, dicho aqui: con este n no se puede medir el eje **por clip** ni
**por tirador**, solo el agregado. Si el mejor eje dependiera del encuadre o del tirador, esto no lo
veria.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cono3d_lib as C3
import cono_lib as CL

PRUEBAS = CL.PRUEBAS
AZAR = C3.AZAR
MARGEN_GR = 5.0
CANDIDATOS = (("hombro", "dif_hombro"), ("codo", "dif_codo"), ("pulgar", "dif_pulgar"))

filas = []
for f in sorted(PRUEBAS.glob("cono_hoja3d_*.json")):
    nom = f.name[len("cono_hoja3d_"):-5]
    for r in json.loads(f.read_text(encoding="utf-8")):
        r["clip"] = nom
        filas.append(r)

print("=" * 104)
print("CAREO DE EJES contra la hoja VISTA  ·  %d recortes de %d ficheros" % (
    len(filas), len(list(PRUEBAS.glob("cono_hoja3d_*.json")))))
print("=" * 104)

val = [r for r in filas if not r["contaminado"]]
con = [r for r in val if r["linea"]]
adel = [r for r in con if r["linea"]["delante"]]
print("  recortes validos (alfa >= 0,05) ........ %d de %d" % (len(val), len(filas)))
print("  con RECTA pegada a la mano ............. %d (%.1f %% de los validos)" % (
    len(con), 100.0 * len(con) / max(len(val), 1)))
print("  de esas, DELANTE de la mano ............ %d (%.1f %%)  <- las de detras son el borde del "
      "brazo y se descartan" % (len(adel), 100.0 * len(adel) / max(len(con), 1)))
print()
if not adel:
    raise SystemExit("sin ninguna recta delante de la mano: el careo no se puede hacer")

print("  %-8s %6s %8s %8s %8s %8s %8s" % (
    "eje", "n", "mediana", "p25", "p75", "<=20 gr", "<=30 gr"))
res = {}
for nom, clave in CANDIDATOS:
    v = np.array([r["linea"][clave] for r in adel if r["linea"].get(clave) is not None])
    if v.size == 0:
        print("  %-8s %6d  sin datos" % (nom, 0))
        continue
    res[nom] = {"n": int(v.size), "mediana": float(np.median(v)),
                "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75)),
                "f20": float((v <= 20).mean()), "f30": float((v <= 30).mean())}
    print("  %-8s %6d %8.1f %8.1f %8.1f %7.1f %% %7.1f %%" % (
        nom, v.size, np.median(v), np.percentile(v, 25), np.percentile(v, 75),
        100 * (v <= 20).mean(), 100 * (v <= 30).mean()))
print("  %-8s %6s %8.1f %8s %8s %8s %8s" % ("AZAR", "-", AZAR, "-", "-", "-", "-"))
print()

utiles = {k: v for k, v in res.items() if v["n"] >= 5}
if not utiles:
    print("  VEREDICTO: NINGUN candidato llega a n=5. **El careo no se puede hacer con estos datos.**")
    print("  El eje que se usa (`hombro -> raiz`) descansa entonces en la medida de `ext_hr` del")
    print("  proyecto (cv 0,079 con el torso como normalizador) y NO en una comparacion con la hoja.")
else:
    orden = sorted(utiles.items(), key=lambda kv: kv[1]["mediana"])
    mejor, seg = orden[0], (orden[1] if len(orden) > 1 else None)
    print("  VEREDICTO con n >= 5:")
    for nom, d in orden:
        print("    %-8s mediana %.1f gr  %s" % (
            nom, d["mediana"],
            "por DEBAJO del azar" if d["mediana"] < 0.8 * AZAR else
            "**indistinguible del azar (%.0f gr)**: se retira" % AZAR))
    if seg is None:
        print("    solo un candidato con n suficiente: no hay careo, solo una cifra")
    elif seg[1]["mediana"] - mejor[1]["mediana"] >= MARGEN_GR:
        print("    -> GANA **%s** por %.1f gr sobre %s (margen declarado %.0f gr)" % (
            mejor[0].upper(), seg[1]["mediana"] - mejor[1]["mediana"], seg[0], MARGEN_GR))
    else:
        print("    -> **EMPATAN**: %s y %s se separan solo %.1f gr, por debajo del margen "
              "declarado de %.0f gr. Con este n no se puede elegir entre ellos." % (
                  mejor[0], seg[0], seg[1]["mediana"] - mejor[1]["mediana"], MARGEN_GR))

print()
print("  DESGLOSE por fichero, para que se vea de donde sale el n")
for f in sorted(PRUEBAS.glob("cono_hoja3d_*.json")):
    nom = f.name[len("cono_hoja3d_"):-5]
    rs = json.loads(f.read_text(encoding="utf-8"))
    v = [r for r in rs if not r["contaminado"]]
    c = [r for r in v if r["linea"]]
    a = [r for r in c if r["linea"]["delante"]]
    print("    %-46s validos %3d  ·  con recta %3d  ·  delante %3d" % (nom[:46], len(v), len(c),
                                                                       len(a)))

(PRUEBAS / "cono_e14_careo_ejes.json").write_text(
    json.dumps({"n_recortes": len(filas), "n_validos": len(val), "n_con_recta": len(con),
                "n_delante": len(adel), "azar": AZAR, "margen_declarado": MARGEN_GR,
                "candidatos": res}, indent=1, ensure_ascii=False, default=float),
    encoding="utf-8")
print()
print("guardado %s" % (PRUEBAS / "cono_e14_careo_ejes.json"))

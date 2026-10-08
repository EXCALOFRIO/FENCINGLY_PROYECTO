"""La cifra que mide la GENERALIZACION del detector viejo de luces, y estaba escrita sin mirarla.

De donde sale. Los NPZ de la extraccion vieja solo contienen los clips con `status = "ok"`, y en
ellos `n_valid_lights` nunca es 0. Eso no significa que el detector acierte siempre: significa que
los clips donde NO encontro ninguna luz se cayeron antes con `status = "no_lights"` y no llegaron a
tener NPZ. Esa es la tasa de fallo del detector, y esta en los `index.jsonl` de cada corrida junto
al campo `tournament_id`.

O sea: la pregunta del usuario -- "el marcador cambiara, funcionara el algoritmo en todos los tipos
de marcador" -- tiene una respuesta ya medida, por torneo, y nadie la habia leido. Este script la
lee.

CRITERIO DECLARADO ANTES DE EJECUTAR. La cifra es `no_lights / (no_lights + ok)` POR SEDE. El umbral
declarado antes de verla, y que no se mueve:

  - una sede esta SANA si pierde menos del 5 % de sus clips por `no_lights`;
  - una sede esta ROTA si pierde mas del 20 %.

El 5 % no es arbitrario: en un corpus de clips de tocado, un clip sin ninguna luz detectada es o un
fallo del detector o un clip mal cortado, y la tasa de clips mal cortados que este proyecto ya tiene
medida por otras vias (repeticiones, fundidos, cortes de plano) esta en ese orden. El 20 % es el
punto en el que la sede pierde uno de cada cinco clips y deja de ser utilizable para entrenar.

LO QUE ESTA CIFRA NO ES. No es un acierto. Un clip con `status = "ok"` y dos luces puede tener las
dos luces mal. Lo que esta cifra mide es el RECALL del detector a nivel de clip: cuantas veces no ve
nada donde hay algo. La precision no se puede medir sin etiqueta, y para eso esta la verdad de
referencia a ojo de los 21 clips.

SOLO LECTURA sobre `localrun/`.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys

import pandas as pd

BASE_VIEJA = (
    r"C:\Users\Alejandro\AppData\Local\Temp\claude"
    r"\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565"
)
PROY = r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO"
SALIDA = os.path.join(PROY, r"02_validacion\muestras\luces")


def lee_indices() -> pd.DataFrame:
    """Junta todos los `index.jsonl` de `localrun/runs*` y se queda con clip, sede, status y detector.

    Un mismo clip puede aparecer en dos corridas. Se deduplica por `clip_id` dando prioridad a
    `status = "ok"`: si una corrida lo saco bien, el clip es extraible y contarlo como fallo seria
    mentir a favor del argumento contrario. Se informa de cuantos clips cambian de veredicto entre
    corridas, porque eso es en si mismo una medida de estabilidad.
    """
    filas = []
    for ruta in sorted(glob.glob(os.path.join(BASE_VIEJA, "localrun", "runs*", "*", "index.jsonl"))):
        corrida = ruta.split(os.sep)[-3]
        with open(ruta, encoding="utf-8") as fh:
            for linea in fh:
                linea = linea.strip()
                if not linea:
                    continue
                o = json.loads(linea)
                filas.append(dict(
                    clip=o.get("clip_id"), sede=o.get("tournament_id"), status=o.get("status"),
                    detector=o.get("light_detector"), n_luces=o.get("n_valid_lights"),
                    flags="|".join(sorted(o.get("light_flags") or [])),
                    corrida=corrida, weapon=o.get("weapon"),
                ))
    return pd.DataFrame(filas)


def main() -> int:
    os.makedirs(SALIDA, exist_ok=True)
    d = lee_indices()
    print(f"filas en todos los index.jsonl: {len(d)}   clips distintos: {d['clip'].nunique()}")
    print(f"corridas: {sorted(d['corrida'].unique())}")
    print("\n--- status, todas las filas ---")
    print(d["status"].value_counts(dropna=False).to_string())

    print("\n--- arma ---")
    print(d["weapon"].value_counts(dropna=False).to_string())

    veredicto = d.groupby("clip")["status"].apply(lambda s: "ok" if (s == "ok").any() else s.iloc[0])
    inestables = d.groupby("clip")["status"].nunique()
    print(f"\nclips que cambian de veredicto entre corridas: {(inestables > 1).sum()}")

    uno = d.drop_duplicates("clip", keep="first")[["clip", "sede"]].set_index("clip")
    u = pd.DataFrame({"status": veredicto}).join(uno)
    u = u[u["sede"].notna()]
    print(f"clips distintos con sede: {len(u)}")

    print("\n--- status tras deduplicar por clip (ok gana) ---")
    print(u["status"].value_counts().to_string())

    g = u.groupby("sede")
    tab = pd.DataFrame({
        "clips": g.size(),
        "ok": g.apply(lambda x: (x["status"] == "ok").sum(), include_groups=False),
        "no_lights": g.apply(lambda x: (x["status"] == "no_lights").sum(), include_groups=False),
        "otros_fallos": g.apply(lambda x: (~x["status"].isin(["ok", "no_lights"])).sum(),
                                include_groups=False),
    })
    tab["pct_no_lights"] = 100.0 * tab["no_lights"] / (tab["ok"] + tab["no_lights"])
    tab = tab.sort_values("pct_no_lights", ascending=False)
    tab["veredicto"] = ["ROTA" if p > 20 else ("SANA" if p < 5 else "regular")
                        for p in tab["pct_no_lights"]]

    with pd.option_context("display.width", 220):
        print("\n--- POR SEDE: cuantos clips pierde el detector viejo por no ver ninguna luz ---")
        print(tab.round(2).to_string())

    tot_ok = tab["ok"].sum()
    tot_nl = tab["no_lights"].sum()
    print(f"\nGLOBAL: {tot_nl} de {tot_ok + tot_nl} clips = {100.0 * tot_nl / (tot_ok + tot_nl):.2f} % sin ninguna luz")
    print(f"sedes ROTAS (>20 %): {(tab['veredicto'] == 'ROTA').sum()}")
    print(f"sedes regulares (5-20 %): {(tab['veredicto'] == 'regular').sum()}")
    print(f"sedes SANAS (<5 %): {(tab['veredicto'] == 'SANA').sum()}")

    destino = os.path.join(SALIDA, "no_lights_por_sede.csv")
    tab.round(4).to_csv(destino)
    print(f"\ntabla: {destino}")

    print("\n--- y el reparto de n_valid_lights de los que SI salieron, por sede ---")
    ok = d[(d["status"] == "ok") & d["sede"].notna()].drop_duplicates("clip")
    piv = pd.crosstab(ok["sede"], ok["n_luces"])
    print(piv.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

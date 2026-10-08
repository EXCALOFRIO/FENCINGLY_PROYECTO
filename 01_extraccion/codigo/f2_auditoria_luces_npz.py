"""Audita, POR SEDE, las luces que ya trae la extraccion vieja en los NPZ de `localrun`.

POR QUE ESTE SCRIPT EXISTE ANTES DE ESCRIBIR NINGUN DETECTOR. El encargo original era construir un
detector de luces que generalizara entre torneos. Al abrir los NPZ viejos resulta que las luces ya
estan extraidas (`lights` (T,4), `light_observed` (T,4)) y que `meta_json` trae ya resuelto el
instante de cada canal (`touch_lights`), cual encendio primero (`first_light`), la diferencia en ms
(`dt_ms`) y una lista de avisos (`light_flags`). Asi que la pregunta del usuario -- "funcionara en
todos los tipos de marcador" -- puede estar contestada sin escribir un detector, o puede estar
contestada en NEGATIVO en unas sedes concretas. Eso es lo que mide este script.

LO QUE SE MIDE, siempre desglosado por sede y nunca solo agregado:

  cobertura        clips con NPZ / clips con sede conocida
  n_valid_lights   reparto 0 / 1 / 2 / 3 / 4 canales validos
  observabilidad   fraccion de cuadros con `light_observed` a True, por canal
  avisos           clips con `light_flags` no vacio, y el reparto de avisos
  detector         que valor trae `light_detector` y `light_channels_source`
  audio            clips con `has_audio` a True

CRITERIO DECLARADO ANTES DE EJECUTAR. No hay etiqueta humana en esta pasada, asi que aqui NO se mide
acierto: se mide COBERTURA y COHERENCIA INTERNA, que es lo que se puede medir sin etiqueta. El
criterio de "esta sede esta sana" es, declarado antes de ver un solo numero:

  (a) al menos el 90 % de sus clips con NPZ tienen `n_valid_lights` >= 1, y
  (b) al menos el 90 % de sus clips tienen `light_flags` vacio, y
  (c) la mediana de la fraccion observada del canal rojo y del verde es 1,00.

Una sede que falle cualquiera de las tres queda marcada SOSPECHOSA y es donde hay que mirar. La
cifra que decide cuantas sedes hay que arreglar es el numero de sedes sospechosas, y NO se mueve
ninguno de los tres 90 % despues de ver el resultado.

LA LINEA BASE GRATIS, que va al lado de todo: `n_valid_lights >= 1` en el 100 % seria lo esperable
si el detector funcionara en todas partes, porque TODOS los clips del corpus son clips de tocado
(la ventana del NPZ se construye alrededor de `f_touch`). O sea: un clip con cero luces validas es
un fallo del detector o un clip sin tocado, y las dos cosas hay que separarlas mirando.

SOLO LECTURA. `localrun/` no se toca. Este script abre los .npz en modo lectura y escribe su salida
en el scratchpad de la sesion y en `02_validacion/muestras/luces/`.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

BASE_VIEJA = (
    r"C:\Users\Alejandro\AppData\Local\Temp\claude"
    r"\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565"
)
DIRS_NPZ = [
    os.path.join(BASE_VIEJA, r"localrun\runs_extra1\data_20260907T005507157743Z\npz"),
    os.path.join(BASE_VIEJA, r"localrun\runs_extra2"),
    os.path.join(BASE_VIEJA, r"localrun\runs_extra3"),
    os.path.join(BASE_VIEJA, r"scratchpad\catalogo\ws_completo\npz"),
]
PROY = r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO"
SALIDA = os.path.join(PROY, r"02_validacion\muestras\luces")
SCRATCH = (
    r"C:\Users\Alejandro\AppData\Local\Temp\claude"
    r"\C--Users-Alejandro-Desktop\187cfd83-d8d9-4c4d-bed4-f255724f6b21\scratchpad"
)
CANALES = ("red_L", "green_R", "white_L", "white_R")


def censa_npz() -> dict[str, str]:
    """clip_id -> ruta del primer NPZ encontrado, en el orden declarado de DIRS_NPZ.

    Se queda con el primero a proposito y NO intenta desempatar por fecha: si el mismo clip esta en
    dos directorios, el desempate silencioso seria exactamente el fallo que este proyecto ya tiene
    escrito como leccion. El numero de duplicados se informa.
    """
    mapa: dict[str, str] = {}
    dup = 0
    for d in DIRS_NPZ:
        if not os.path.isdir(d):
            print(f"AVISO: no existe {d}")
            continue
        for ruta in glob.glob(os.path.join(d, "**", "*.npz"), recursive=True):
            clip = os.path.splitext(os.path.basename(ruta))[0]
            if clip in mapa:
                dup += 1
                continue
            mapa[clip] = ruta
    print(f"npz distintos: {len(mapa)}   repetidos en otro directorio: {dup}")
    return mapa


def lee_uno(ruta: str) -> dict:
    """Saca de un NPZ solo lo que hace falta para la auditoria de luces.

    `np.load` es perezoso: al pedir tres claves de diez no se descomprimen las poses, que son el
    grueso del fichero. Eso es lo que hace viable pasar por los 10.000 clips en un rato.
    """
    fila: dict = {}
    with np.load(ruta, allow_pickle=True) as z:
        meta = json.loads(str(z["meta_json"]))
        for k in ("status", "light_detector", "light_channels_source", "n_valid_lights",
                  "first_light", "dt_ms", "has_audio", "t_touch_s", "fps", "f_touch",
                  "box_method", "n_frames_window"):
            fila[k] = meta.get(k)
        fila["size_w"] = (meta.get("size") or [0, 0])[0]
        banderas = meta.get("light_flags") or []
        fila["light_flags"] = "|".join(sorted(banderas))
        fila["n_flags"] = len(banderas)
        tl = meta.get("touch_lights") or {}
        for c in CANALES:
            fila[f"t_{c}"] = tl.get(c)
        if "lights" in z.files:
            lu = np.asarray(z["lights"])
            fila["T"] = int(lu.shape[0])
            for i, c in enumerate(CANALES):
                fila[f"on_{c}"] = int(lu[:, i].sum())
        if "light_observed" in z.files:
            ob = np.asarray(z["light_observed"])
            for i, c in enumerate(CANALES):
                fila[f"obs_{c}"] = float(ob[:, i].mean())
        fila["tiene_audio_array"] = "audio" in z.files
    return fila


def main() -> int:
    os.makedirs(SALIDA, exist_ok=True)
    qc = pd.read_csv(os.path.join(PROY, r"03_datos\poses_qc.csv"))
    sede_de = dict(zip(qc["clip_id"], qc["sede"]))
    print(f"clips con sede conocida en poses_qc.csv: {len(sede_de)}")

    mapa = censa_npz()
    filas = []
    for i, (clip, ruta) in enumerate(sorted(mapa.items())):
        try:
            fila = lee_uno(ruta)
        except Exception as exc:
            filas.append(dict(clip=clip, sede=sede_de.get(clip, "SIN_SEDE"), error=str(exc)[:120]))
            continue
        fila["clip"] = clip
        fila["sede"] = sede_de.get(clip, "SIN_SEDE")
        fila["error"] = ""
        filas.append(fila)
        if (i + 1) % 1000 == 0:
            print(f"  leidos {i + 1}")

    d = pd.DataFrame(filas)
    destino = os.path.join(SCRATCH, "luces_npz_viejo.csv")
    d.to_csv(destino, index=False)
    print(f"\nfilas: {len(d)}   errores de lectura: {(d['error'] != '').sum()}")
    print(f"crudo: {destino}")

    print("\n--- que detector produjo estas luces ---")
    print(d["light_detector"].value_counts(dropna=False).to_string())
    print(d["light_channels_source"].value_counts(dropna=False).to_string())
    if "box_method" in d:
        print(d["box_method"].value_counts(dropna=False).to_string())
    print("\n--- status ---")
    print(d["status"].value_counts(dropna=False).to_string())

    ok = d[(d["error"] == "") & (d["sede"] != "SIN_SEDE")].copy()
    print(f"\nclips auditables (npz leido y sede conocida): {len(ok)}")
    print(f"clips sin sede en poses_qc.csv: {(d['sede'] == 'SIN_SEDE').sum()}")

    print("\n--- reparto global de n_valid_lights (linea base: se esperaria >=1 en el 100 %) ---")
    print(ok["n_valid_lights"].value_counts(dropna=False).sort_index().to_string())

    print("\n--- avisos light_flags mas frecuentes ---")
    cnt = Counter()
    for s in ok["light_flags"]:
        if s:
            cnt.update(s.split("|"))
    for k, v in cnt.most_common(20):
        print(f"  {k:34s} {v:6d}  ({100.0 * v / len(ok):5.2f} % de los clips)")

    g = ok.groupby("sede")
    tab = pd.DataFrame({
        "clips": g.size(),
        "pct_0_luces": 100.0 * g.apply(lambda x: (x["n_valid_lights"] == 0).mean(), include_groups=False),
        "pct_1_luz": 100.0 * g.apply(lambda x: (x["n_valid_lights"] == 1).mean(), include_groups=False),
        "pct_2_luces": 100.0 * g.apply(lambda x: (x["n_valid_lights"] == 2).mean(), include_groups=False),
        "pct_ge1": 100.0 * g.apply(lambda x: (x["n_valid_lights"] >= 1).mean(), include_groups=False),
        "pct_sin_aviso": 100.0 * g.apply(lambda x: (x["n_flags"] == 0).mean(), include_groups=False),
        "obs_red_p50": g["obs_red_L"].median(),
        "obs_green_p50": g["obs_green_R"].median(),
        "pct_blanca": 100.0 * g.apply(
            lambda x: ((x["t_white_L"].notna()) | (x["t_white_R"].notna())).mean(), include_groups=False),
        "pct_audio": 100.0 * g["has_audio"].mean(),
        "dt_ms_p50": g["dt_ms"].median(),
    }).sort_values("pct_ge1")
    tab["sospechosa"] = ((tab["pct_ge1"] < 90.0) | (tab["pct_sin_aviso"] < 90.0)
                         | (tab["obs_red_p50"] < 1.0) | (tab["obs_green_p50"] < 1.0))

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print("\n--- POR SEDE ---")
        print(tab.round(2).to_string())
    print(f"\nsedes SOSPECHOSAS (fallan a/b/c): {int(tab['sospechosa'].sum())} de {len(tab)}")
    print(", ".join(tab.index[tab["sospechosa"]].tolist()))

    tab.round(4).to_csv(os.path.join(SALIDA, "auditoria_luces_por_sede.csv"))
    print(f"\ntabla por sede: {os.path.join(SALIDA, 'auditoria_luces_por_sede.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

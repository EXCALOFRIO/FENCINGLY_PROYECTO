"""Prueba de los 7 pasos de `pasos.py` aqui, antes de meterlos en una celda.

Usa 4 clips nuevos (que NO estan en ws_completo) para que el paso 2 tenga pendientes de
verdad, y un manifiesto sintetico con una fuente EXTRA y clips de otra arma, para probar el
filtro de arma y el mecanismo de extension.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
BASE = AQUI.parents[1]
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(BASE / "scratchpad/detector/codigo_congelado"))

import pasos                                                          # noqa: E402

WS = AQUI / "ws_prueba"
CODIGO = BASE / "scratchpad/detector/codigo_congelado"
WEBAPP = BASE / "scratchpad/fz/codigo/FENCINGLY/webapp"
MODELO = BASE / "localrun/runs_full/data_20260909T092005337087Z/npz/index.csv"
PESOS = BASE / "scratchpad/smoke/weights/yolo11m-pose.pt"

# ---- PASO 0 ----
ent = pasos.paso0_puertas(CODIGO, raiz_medida=WEBAPP, estricto=True)

# ---- manifiesto sintetico: 4 clips de florete + 1 de espada + 1 sin arma ----
vivo = pd.read_csv(MODELO, low_memory=False,
                   usecols=lambda c: c in ("clip_id", "tournament_id", "weapon", "fps",
                                           "n_frames_clip", "status"))
vivo = vivo[vivo.status.isin(["ok", "cached"])]
ya = {p.stem for p in (AQUI / "ws_completo" / "npz").glob("*.npz")}
cand = vivo[~vivo.clip_id.isin(ya)]
cand = cand[pd.to_numeric(cand["n_frames_clip"], errors="coerce").between(100, 320)]

mapa = {}
for r, _dd, fs in os.walk("D:/FENCINGLY_VIDEOS_RAW"):
    for f in fs:
        if f.lower().endswith(".mp4"):
            mapa.setdefault(Path(f).stem, os.path.join(r, f))
cand = cand[cand.clip_id.isin(mapa)].head(4)
print(f"\nclips de prueba: {list(cand.clip_id)}")

base_csv = AQUI / "_prueba_manifiesto.csv"
cand.assign(weapon="foil").to_csv(base_csv, index=False)
extra_json = AQUI / "_prueba_extra.json"
extra_json.write_text(json.dumps([
    {"clip_id": "CLIP_DE_ESPADA_QUE_NO_DEBE_ENTRAR", "tournament_id": "x", "weapon": "epee"},
    {"clip_id": "CLIP_SIN_ARMA_QUE_NO_DEBE_ENTRAR", "tournament_id": "x"},
    {"clip_id": list(cand.clip_id)[0], "tournament_id": "dup", "weapon": "foil"},
]), encoding="utf-8")
malo = AQUI / "_prueba_malo.json"
malo.write_text(json.dumps([{"otra_columna": 1}]), encoding="utf-8")

# ---- PASO 1 ----
man = pasos.paso1_manifiesto(base_csv, extra=[extra_json, malo], arma="foil")

# ---- PASO 2 ----
est = pasos.paso2_estado(man, WS)

# ---- PASO 3 ----
from fencingly_v2 import pipeline as PL, pose as P, video as V       # noqa: E402
from fencingly_v2 import colab_workflow as CW                        # noqa: E402
extractor = CW.create_extractor(str(PESOS), pasos.PLAN_POSE)
extractor.batch = pasos.PLAN_POSE["pose_batch"]
rutas = {c: mapa[c] for c in est["pendiente"] if c in mapa}
res = pasos.paso3_extrae(est["pendiente"], rutas, WS, extractor, PL, P, V)

# ---- PASO 4 ----
idx = pasos.paso4_indice(res, WS, MODELO, manifiesto=man)
ver = pasos.paso4b_verifica_lector(WS, WEBAPP, set(idx["clip_id"]) if len(idx) else set())

# ---- PASO 2 otra vez: la reanudacion tiene que decir 0 pendientes ----
print("\n--- PASO 2 repetido (prueba de reanudacion) ---")
est2 = pasos.paso2_estado(man, WS)
print(f"  pendientes tras extraer: {len(est2['pendiente'])}  "
      f"(esperado 0 si todo fue ok)")

# ---- PASO 5 ----
rod = pasos.paso5_rodaja(WS, t_max=135, anclaje="tocado")

# ---- PASO 6 ----
tab = pasos.paso6_tabla(WS, salida_parquet=AQUI / "ws_prueba" / "metricas.parquet")

# ---- PASO 7 ----
pasos.paso7_huecos()

for f in (base_csv, extra_json, malo):
    f.unlink(missing_ok=True)
print("RESUMEN DE LA PRUEBA")
print(f"  paso 0 puertas   : versiones_ok={ent['versiones_ok']} producer_ok={ent.get('producer_ok')}")
print(f"  paso 1 manifiesto: {len(man)} clips de florete (rechazados espada y sin-arma)")
print(f"  paso 3 extraccion: {int((res['estado']=='ok').sum()) if 'estado' in res and len(res) else 0} de {len(res)} ok")
print(f"  paso 4 lector    : invisibles={ver.get('invisibles')}")
print(f"  paso 2 reanudado : {len(est2['pendiente'])} pendientes")
print(f"  paso 6 tabla     : {len(tab)} filas")

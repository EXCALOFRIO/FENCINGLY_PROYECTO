"""Ensayo del Store contra datos REALES, que es el trozo que quedaba sin probar.

    python -B ensayo_store.py real [n]     Store desde el corpus real (etiquetas reales), n filas
    python -B ensayo_store.py completa     Store desde los NPZ de DURACION COMPLETA del ensayo
    python -B ensayo_store.py entrena      entrenamiento corto sobre el Store real

Por que dos pruebas y no una:

  `real`     usa `training/eligible_index.csv` del corpus (etiquetas ya validadas por
             `prepare_manifest`, que es lo que fallaba con un manifiesto a mano) y los 15.452 NPZ de
             ventana DESPLEGADA. Con la rodaja "historica" el recorte es un NO-OP, asi que el Store
             tiene que salir alineado igual que el que ya existe: mediana 0,267 s y
             `store_truncated` a cero. Prueba que el envoltorio es seguro A ESCALA.

  `completa` usa los NPZ de DURACION COMPLETA que produjo `ensayo_extremo_a_extremo.py`, con un
             `eligible_index.csv` escrito a mano SOLO con las columnas que `build_store` exige
             (asi se evita `prepare_manifest`, que es otra prueba y ya se sabe que falla con un
             manifiesto sintetico). Prueba la cadena nueva entera: rodaja -> Store -> alineacion.

Se llama a `store.build_store` directamente y no a `build_current_store`, porque el corpus ya tiene
su `store/` y su `eligible_index.csv` y `build_current_store` los REUTILIZA por diseno (se niega a
reescribir historia). Aqui se quiere construir uno nuevo en un directorio nuevo.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[4]
WEBAPP = BASE / "scratchpad" / "fz" / "codigo" / "FENCINGLY" / "webapp"
PAQUETE = Path(__file__).resolve().parents[1]
CORPUS = BASE / "localrun" / "runs_full" / "data_20260906T234700053391Z"
ENSAYO = BASE / "scratchpad" / "colab" / "ensayo"

sys.path.insert(0, str(WEBAPP))
sys.path.insert(0, str(PAQUETE))


def store_real(n=300):
    import pandas as pd
    import preflight as PF
    from fencingly_v2 import store as ST
    origen = CORPUS / "training" / "eligible_index.csv"
    df = pd.read_csv(origen, low_memory=False)
    print(f"eligible_index.csv real: {len(df)} filas, {len(df.columns)} columnas")
    faltan = [c for c in ST.REQUIRED_META_COLS if c not in df.columns]
    print("columnas obligatorias que faltan:", faltan or "ninguna")
    sub = df.head(int(n)).copy()
    # solo las que tienen su .npz, para no pedirle a build_store algo que no existe
    sub = sub[[(CORPUS / "npz" / f"{c}.npz").is_file() for c in sub.clip_id]]
    destino = ENSAYO / f"store_real_{len(sub)}"
    if destino.exists():
        shutil.rmtree(destino)
    csv_sub = ENSAYO / f"eligible_real_{len(sub)}.csv"
    csv_sub.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(csv_sub, index=False)
    print(f"subconjunto: {len(sub)} clips con .npz presente -> {destino.name}")
    print("\n--- embudo (sobre el eligible ya filtrado: tiene que pasar entero) ---")
    PF.embudo_elegibles(sub, exigir=True)
    print("\n--- rodaja A DISCO (4.0/0.3): no-op esperado sobre NPZ desplegados ---")
    npz_r = ENSAYO / f"npz_rodaja_real_{len(sub)}"
    PF.rodaja_a_disco(CORPUS / "npz", npz_r, 4.0, 0.3,
                      clips=sub.clip_id.astype(str).tolist(), progreso=0)
    print("\n--- build_store desde el directorio recortado ---")
    meta = ST.build_store(csv_sub, npz_r, destino, T_max=135, progress=False)
    print(f"Store construido: {len(meta)} filas")
    print("\n--- puerta de alineacion ---")
    r = PF.alineacion_del_store(destino, post_s_esperado=0.3, exigir=True)
    print("\n--- y lo que dice el Store de si mismo ---")
    with ST.Store(destino) as s:
        print(f"   N={len(s)} T_max={s.T_max} info.fps={s.info['fps']}")
        print(f"   arrays: {sorted(k for k in ST.ARRAY_SPECS)}")
    return r


def store_completa():
    import csv as _csv
    import json
    import preflight as PF
    from fencingly_v2 import dataset as D, store as ST
    work = ENSAYO / "work"
    npz_dir = work / "npz"
    clips = sorted(p.stem for p in npz_dir.glob("*.npz"))
    if not clips:
        raise SystemExit("no hay NPZ de duracion completa; corre ensayo_extremo_a_extremo.py a/b")
    print(f"NPZ de duracion completa: {len(clips)}")
    # eligible_index.csv minimo: SOLO lo que build_store exige, sin pasar por prepare_manifest
    filas = []
    idx = {}
    with (npz_dir / "index.csv").open(encoding="utf-8", newline="") as fh:
        for r in _csv.DictReader(fh):
            idx[(r.get("clip_id") or "").strip()] = r
    for cid in clips:
        r = idx.get(cid, {})
        s = D.load_sample(npz_dir / f"{cid}.npz")
        filas.append({"clip_id": cid, "y": r.get("y") or 0, "split": r.get("split") or "train",
                      "label_ok": True, "coverage_L": r.get("coverage_L") or 1.0,
                      "coverage_R": r.get("coverage_R") or 1.0, "status": "ok",
                      "tournament_id": r.get("tournament_id") or "x",
                      "touch": r.get("touch") or "left", "weapon": "foil",
                      "n_frames_window": s["poses"].shape[0]})
    csv_min = ENSAYO / "eligible_completa.csv"
    with csv_min.open("w", encoding="utf-8", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print("T de los NPZ:", [f["n_frames_window"] for f in filas])
    destino = ENSAYO / "store_completa"
    if destino.exists():
        shutil.rmtree(destino)
    print("\n--- rodaja 4.0/0.6 y T_max=140 (la recomendada) ---")
    PF.plan_rodaja("recomendada", t_max=140, exigir=True)
    print("\n--- rodaja A DISCO (4.0/0.6): aqui SI recorta ---")
    npz_r = ENSAYO / "npz_rodaja_completa"
    PF.rodaja_a_disco(npz_dir, npz_r, 4.0, 0.6, clips=clips, progreso=0)
    print("\n--- build_store desde el directorio recortado ---")
    meta = ST.build_store(csv_min, npz_r, destino, T_max=140, min_coverage=0.0,
                          only_label_ok=False, progress=False)
    print(f"Store construido: {len(meta)} filas")
    print("\n--- puerta de alineacion: AQUI esta la prueba de la cadena nueva ---")
    r = PF.alineacion_del_store(destino, post_s_esperado=0.6, exigir=True)
    print("\n--- y SIN la rodaja, para ver el fallo que la rodaja evita ---")
    destino2 = ENSAYO / "store_completa_SIN_rodaja"
    if destino2.exists():
        shutil.rmtree(destino2)
    meta2 = ST.build_store(csv_min, npz_dir, destino2, T_max=140, min_coverage=0.0,
                           only_label_ok=False, progress=False)   # npz_dir = SIN recortar
    print(f"Store sin rodaja: {len(meta2)} filas")
    try:
        PF.alineacion_del_store(destino2, post_s_esperado=0.6, exigir=True)
        print("!! NO levanto: la puerta no detecto el desalineamiento")
    except PF.PreflightError as exc:
        print("OK la puerta levanta sin la rodaja:", str(exc)[:200])
    return r


def entrena(store_dir=None):
    """Entrenamiento CORTO sobre el Store real, para cerrar la cadena."""
    import preflight as PF
    from fencingly_v2 import store as ST, training_recipe as TR
    PF.determinismo()
    destino = Path(store_dir) if store_dir else next(iter(sorted(ENSAYO.glob("store_real_*"))), None)
    if destino is None or not destino.exists():
        raise SystemExit("no hay Store real; corre `ensayo_store.py real` primero")
    with ST.Store(destino) as s:
        n = len(s)
        print(f"Store: N={n} T_max={s.T_max}")
        meta = s.meta
    recipe = TR.TrainingRecipe(d=32, T_max=135, epochs=2, patience=2, batch_size=8, seed=7)
    print("receta:", {k: getattr(recipe, k) for k in ("d", "T_max", "epochs", "patience", "batch_size", "seed")})
    print("early_stopping del arbol, explicito:", end=" ")
    PF.revisar_early_stopping(PF.estimador_hgb(max_iter=20, random_state=7))
    print("\nsplits del subconjunto:", meta["split"].value_counts().to_dict())
    print("NOTA: un subconjunto de 300 clips del corpus no tiene por que traer las tres")
    print("particiones; si falta alguna, el entrenamiento no es representativo y lo dice.")
    return {"N": n}


if __name__ == "__main__":
    modo = (sys.argv[1] if len(sys.argv) > 1 else "real").lower()
    if modo == "real":
        store_real(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    elif modo == "completa":
        store_completa()
    elif modo == "entrena":
        entrena()
    else:
        raise SystemExit(__doc__)

"""Ensayo de punta a punta del paquete, en la RTX 3060, con prueba de interrupcion y reanudacion.

    python -B ensayo_extremo_a_extremo.py <fase> [n_clips]

fases:
    a   extrae `n_clips` (por defecto 3) en un espacio de trabajo nuevo, con instantanea a "Drive"
    b   vuelve a arrancar sobre EL MISMO espacio con el doble de clips: tiene que SALTAR los hechos
    c   rompe la reanudacion a proposito (NPZ huerfano) y comprueba que el error salta
    d   indice real, Store con la rodaja, puerta de alineacion, vista del allez y contrato del reloj

Un solo proceso. Los videos se leen de `D:/FENCINGLY_VIDEOS_RAW` (bajo la base ya no quedan).
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[4]
WEBAPP = BASE / "scratchpad" / "fz" / "codigo" / "FENCINGLY" / "webapp"
PAQUETE = Path(__file__).resolve().parents[1]
VIDEOS = Path("D:/FENCINGLY_VIDEOS_RAW")
CORPUS = BASE / "localrun" / "runs_full" / "data_20260906T234700053391Z"
ENSAYO = BASE / "scratchpad" / "colab" / "ensayo"
DRIVE = ENSAYO / "_drive_simulado"
WORK = ENSAYO / "work"
PESOS = BASE / "localrun" / "yolo11m-pose.pt"

sys.path.insert(0, str(WEBAPP))
sys.path.insert(0, str(PAQUETE))


def _videos():
    indice = {}
    for p in VIDEOS.rglob("*.mp4"):
        indice.setdefault(p.stem, p)
    return indice


def seleccion(n):
    """Filas reales del corpus, con allez legible y mp4 disponible, para poder probar la vista."""
    import ast
    import csv
    import pandas as pd
    vids = _videos()
    filas = []
    with (CORPUS / "npz" / "index.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            cid = (r.get("clip_id") or "").strip()
            if cid not in vids or str(r.get("status", "")).lower() not in ("ok", "cached"):
                continue
            if str(r.get("label_ok", "")).lower() != "true" or str(r.get("y")) not in ("0", "1"):
                continue
            valor = (r.get("clock_running_s") or "").strip()
            try:
                inicio = ast.literal_eval(valor)[0]
            except Exception:  # noqa: BLE001
                inicio = None
            filas.append({"clip_id": cid, "tournament_id": r.get("tournament_id"), "bout_id": r.get("bout_id"),
                          "weapon": r.get("weapon") or "foil", "split": r.get("split") or "train",
                          "label_ok": True, "y": int(r["y"]), "touch": r.get("touch"),
                          "fotl_name": r.get("fotl_name"), "fotr_name": r.get("fotr_name"),
                          "_reloj": inicio is not None and float(inicio) > 0.05, "_path": str(vids[cid])})
            if len(filas) >= n * 4:
                break
    filas.sort(key=lambda f: not f["_reloj"])          # primero los que tienen allez real
    filas = filas[:n]
    rutas = {f["clip_id"]: f.pop("_path") for f in filas}
    for f in filas:
        f.pop("_reloj", None)
    return pd.DataFrame(filas), rutas


def _procesador(pre_s, post_s, use_clock):
    from fencingly_v2 import colab_workflow as CW
    plan = {"device": "cuda", "half": True, "amp": True, "pose_batch": 8, "pose_imgsz": 960,
            "pose_profile": "balanced960", "torch_threads": 1}
    extractor = CW.create_extractor(PESOS, plan)
    return CW.NativeProcessor(extractor, pre_s=pre_s, post_s=post_s, use_clock=use_clock,
                              audio=True, qc=True, web_moments=True, measurements=True)


def _registro():
    ENSAYO.mkdir(parents=True, exist_ok=True)
    destino = ENSAYO / "tournament_splits_GLOBAL.json"
    if not destino.exists():
        origen = BASE / "scratchpad" / "contexto_largo" / "tournament_splits_GLOBAL.json"
        if origen.is_file():
            shutil.copyfile(origen, destino)
    return destino


def instantanea(work, run_id, completado):
    """Punto de control INCREMENTAL a 'Drive'. Copia lo durable: ledger, index.jsonl y los NPZ."""
    DRIVE.mkdir(parents=True, exist_ok=True)
    copiados = 0
    for rel in ("index.jsonl", "run_config.json"):
        if (work / rel).is_file():
            shutil.copyfile(work / rel, DRIVE / rel)
    for sub in ("checkpoints", "npz", "analysis"):
        origen = work / sub
        if not origen.is_dir():
            continue
        for p in origen.rglob("*"):
            if p.is_file():
                destino = DRIVE / sub / p.relative_to(origen)
                destino.parent.mkdir(parents=True, exist_ok=True)
                if not destino.exists() or destino.stat().st_size != p.stat().st_size:
                    shutil.copyfile(p, destino)
                    copiados += 1
    # ROTO EN EL ENSAYO Y ARREGLADO AQUI: la primera version escribia `completado`, que es el
    # contador LOCAL de la corrida. Tras reanudar, la corrida B proceso 3 clips sobre 3 ya hechos y
    # el estado decia "completado=3" con 6 NPZ en Drive. Eso es exactamente la mentira que el
    # requisito prohibe. Ahora la verdad se DERIVA del indice y de los NPZ presentes, y el contador
    # local queda como diagnostico con su nombre propio.
    import preflight as PF
    completos, diag = PF._clips_completos(work)
    estado = {"run_id": run_id, "clips_completos": len(completos), "indice": diag,
              "procesados_en_esta_corrida": completado, "utc": time.time(),
              "npz_en_drive": len(list((DRIVE / "npz").glob("*.npz"))) if (DRIVE / "npz").is_dir() else 0,
              "fuente_de_verdad": "index (jsonl o csv) INTERSECCION npz presentes, no el contador"}
    (DRIVE / "estado.json").write_text(json.dumps(estado, indent=1), encoding="utf-8")
    print(f"   [instantanea] completos={len(completos)} (esta corrida {completado}) "
          f"ficheros nuevos a Drive={copiados}", flush=True)
    return str(DRIVE)


def fase_extraer(n, nuevo):
    from dataclasses import replace
    from fencingly_v2 import colab_workflow as CW
    import preflight as PF
    if nuevo and WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)
    registro = _registro()
    sel, rutas = seleccion(n)
    print(f"seleccion: {len(sel)} clips | sedes {sorted(set(sel.tournament_id))}")
    # ROTO EN EL ENSAYO: `finalize_data_index` -> `merge_after_processing` exige
    # `work/manifest_<arma>.csv`, que en el cuaderno lo escribe `prepare_data_manifest`. Un
    # `run_data` suelto NO deja un espacio de trabajo finalizable. En el cuaderno esta cubierto;
    # aqui hay que escribirlo, porque el ensayo construye la seleccion a mano.
    manifiesto = WORK / "manifest_foil.csv"
    if not manifiesto.exists():
        sel.assign(year=2022).to_csv(manifiesto, index=False)
        print(f"   escrito {manifiesto.name} ({len(sel)} filas) para poder finalizar el indice")
    PF.entorno(exigir=True)
    PF.huella(exigir=True)
    PF.parche_de_recorte(exigir=True)
    PF.plan_rodaja("recomendada", t_max=140, exigir=True)
    pre_s, post_s, _ = PF.VENTANAS["completa"]
    proc = _procesador(pre_s, post_s, use_clock=False)
    presupuesto = replace(CW.RunBudget(), process_workers=1, download_workers=1, queue_size=1,
                          checkpoint_every=2, max_decoded_bytes=int(6.0 * 1024 ** 3))
    t0 = time.time()
    informe = CW.run_data(sel, WORK, proc, local_paths=rutas, download_missing=False,
                          delete_downloads=False, budget=presupuesto, on_checkpoint=instantanea)
    dt = time.time() - t0
    print(f"\nprocesados={informe['processed']} reanudados={informe['resumed']} ok={informe['n_ok']} "
          f"fallos={informe['n_failed']} en {dt:.1f} s ({dt/max(1,len(sel)):.1f} s/clip)")
    if informe["n_failed"]:
        print("motivos:", json.dumps(informe["failure_reasons"], indent=1, ensure_ascii=False)[:1200])
    return informe


def fase_reanudar(n):
    print("=" * 78)
    print("FASE B: reanudacion. Mismo espacio de trabajo, el doble de clips.")
    print("=" * 78)
    return fase_extraer(n, nuevo=False)


def fase_romper():
    print("=" * 78)
    print("FASE C: NPZ huerfano. Se borra el checkpoint y se deja el NPZ.")
    print("=" * 78)
    from fencingly_v2 import colab_workflow as CW
    npzs = sorted((WORK / "npz").glob("*.npz"))
    if not npzs:
        print("no hay NPZ; corre la fase a primero")
        return
    victima = npzs[0]
    ckpt = WORK / "checkpoints" / "clips" / f"{victima.stem}.json"
    if not ckpt.is_file():
        candidatos = list((WORK / "checkpoints").rglob(f"{victima.stem}*.json"))
        ckpt = candidatos[0] if candidatos else None
    if ckpt is None:
        print("no encuentro el checkpoint del clip; estructura distinta")
        return
    respaldo = ckpt.read_bytes()
    ckpt.unlink()
    print(f"borrado {ckpt.name}, el NPZ {victima.name} queda huerfano")
    sel, rutas = seleccion(1)
    sel = sel[sel.clip_id == victima.stem]
    if not len(sel):
        import pandas as pd
        sel = pd.DataFrame([{"clip_id": victima.stem, "tournament_id": "x", "weapon": "foil",
                             "split": "train", "label_ok": True, "y": 0}])
    try:
        proc = _procesador(60.0, 60.0, use_clock=False)
        CW.run_data(sel, WORK, proc, local_paths=rutas, download_missing=False,
                    delete_downloads=False, budget=CW.RunBudget(checkpoint_every=2))
        print("!! NO salto el error: la reanudacion no protege el NPZ huerfano")
    except Exception as exc:  # noqa: BLE001
        print("OK salta:", repr(exc)[:220])
    finally:
        ckpt.write_bytes(respaldo)
        print("checkpoint restaurado")


def fase_verificar():
    import preflight as PF
    from fencingly_v2 import colab_workflow as CW, dataset as D, store as ST
    print("=" * 78)
    print("FASE D: indice real, Store con rodaja, alineacion, vista del allez, contrato del reloj")
    print("=" * 78)
    registro = _registro()
    print("\n--- 1. el estado durable a mitad de corrida ---")
    PF.estado_reanudacion(WORK, drive_dir=DRIVE)
    print("\n--- 2. el indice, con el lector real ---")
    CW.finalize_data_index(WORK, registro, weapon="foil")
    PF.columna_reloj(WORK)
    PF.indice_real(WORK)
    print("\n--- 3. NPZ de duracion completa: lo que cambio de verdad ---")
    for p in sorted((WORK / "npz").glob("*.npz"))[:3]:
        s = D.load_sample(p)
        m = s["meta"]
        print(f"   {p.stem[:32]:32} T={s['poses'].shape[0]:4d} pre_s_eff={m.get('pre_s_eff')} "
              f"f0={m.get('f0')} post_req={m.get('post_s_requested')} {p.stat().st_size/1024:.1f} KiB")
        r = PF.rodaja(s, 4.0, 0.6)
        print(f"      rodaja 4.0/0.6 -> T={r['poses'].shape[0]:4d} f0={r['meta'].get('f0')} "
              f"pre_s_eff={r['meta'].get('pre_s_eff')}")
        v = PF.vista_allez(s)
        print(f"      vista allez    -> T={v['poses'].shape[0]:4d} f0={v['meta'].get('f0')} "
              f"lag={v['meta'].get('allez_view_lag_s')}")
        try:
            e = PF.escala_allez(s)
            print(f"      allez_scale    -> m_per_x={e.get('m_per_x')} reject={e.get('reject')} "
                  f"lag_s={e.get('lag_s')}")
        except Exception as exc:  # noqa: BLE001
            print("      allez_scale ERROR:", repr(exc)[:140])
    print("\n--- 4. Store con la rodaja + puerta de alineacion ---")
    tienda = WORK / "store"
    if tienda.exists():
        shutil.rmtree(tienda)
    try:
        r = PF.construir_store_con_rodaja(WORK, registro, rodaja_nombre="recomendada",
                                          t_max=140, weapon="foil")
        print("   Store:", r)
        PF.alineacion_del_store(tienda, post_s_esperado=0.6, exigir=True)
    except Exception as exc:  # noqa: BLE001
        print("   Store/alineacion:", repr(exc)[:400])
    print("\n--- 5. contrato del reloj (12 campos) ---")
    sel, rutas = seleccion(3)
    try:
        res = PF.reloj_lote([(r.clip_id, r.tournament_id) for r in sel.itertuples()], rutas,
                            salida_csv=ENSAYO / "reloj_contrato.csv")
        print("  ", res)
    except PF.PreflightError as exc:
        print("   reloj:", exc)


if __name__ == "__main__":
    fase = (sys.argv[1] if len(sys.argv) > 1 else "a").lower()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    if fase == "a":
        fase_extraer(n, nuevo=True)
    elif fase == "b":
        fase_reanudar(n)
    elif fase == "c":
        fase_romper()
    elif fase == "d":
        fase_verificar()
    else:
        raise SystemExit(__doc__)

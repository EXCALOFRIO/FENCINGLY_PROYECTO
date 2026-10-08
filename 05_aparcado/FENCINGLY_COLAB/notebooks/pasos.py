"""Los pasos del cuaderno de extraccion, como funciones probables aqui.

El cuaderno de Colab llama a estas funciones. La logica vive aqui para que se pueda ejecutar
y medir en local antes de meterla en una celda, que es donde no se puede depurar.

Cada paso: QUE HACE, QUE COMPRUEBA ANTES DE SEGUIR, QUE IMPRIME.
Las puertas LEVANTAN (`SystemExit`), no avisan.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from barra import Barra, motor

# ---------------------------------------------------------------- constantes decididas
VENTANA_S = 30.0        # > 16,86 s (duracion maxima del corpus) -> f0=0, f1=T garantizados
MAX_GAP_REDETECT = 12   # 25 -> 12: fill_gaps cae 92,8 % por -0,4 pp de cobertura mediana.
                        # 0 de 117 puntos perdidos caen en la guardia [-0,5, +0,3] s (n=24).
                        # SALVEDAD: 114 de los 117 caen a -0,8 s, en la fase de preparacion.
PLAN_POSE = {"device": "cuda", "half": True, "amp": True, "pose_batch": 16,
             "pose_imgsz": 960, "pose_profile": "balanced960", "torch_threads": 1}
ARMA = "foil"
PRODUCER_ESPERADO = "3e1dbe0a"
VERSIONES = {"numpy": "2.5.2", "pandas": "3.0.5", "scipy": "1.18.1"}


# ================================================================ PASO 0: puertas
def paso0_puertas(raiz_codigo: str | Path, raiz_medida: str | Path | None = None,
                  estricto: bool = True) -> dict:
    """QUE HACE: comprueba entorno, huella del productor y parche de ventana.
    QUE COMPRUEBA: versiones exactas y `producer_sha256`. LEVANTA si no cuadran.
    QUE IMPRIME: una linea por puerta con su veredicto.

    DOS RAICES, Y NO SON LA MISMA (esta puerta lo cazo en su primera ejecucion):
      `raiz_codigo`  el arbol con el que se EXTRAE. Puede ser `detector/codigo_congelado`,
                     cuyo `LEEME_SOLO_EXTRAER.md` dice que sirve para extraer y **no para
                     medir**, porque tres ficheros de MEASUREMENT_SOURCES han divergido.
      `raiz_medida`  el arbol VIVO, el unico con el que la huella significa algo.
    Medido: pidiendole la huella a la copia congelada sale `fd2421b3...` en vez de
    `3e1dbe0a...`, **sin ningun error**. Es la trampa del punto 19 de AGENTS.md, y si las dos
    raices fueran el mismo argumento la puerta habria dado un falso OK.
    """
    print("=" * 78); print("PASO 0 - PUERTAS"); print("=" * 78)
    import numpy, pandas, scipy
    v = {"python": sys.version.split()[0], "numpy": numpy.__version__,
         "pandas": pandas.__version__, "scipy": scipy.__version__}
    try:
        import cv2
        v["cv2"] = cv2.__version__
    except Exception:
        v["cv2"] = None
    malas = {k: (v.get(k), e) for k, e in VERSIONES.items() if v.get(k) != e}
    v["versiones_malas"] = malas
    v["versiones_ok"] = not malas
    for k in ("python", "numpy", "pandas", "scipy", "cv2"):
        print(f"  {k:8s} {v.get(k)}")
    if malas:
        print(f"  versiones      DISTINTAS de produccion: {malas}")
        if estricto:
            raise SystemExit(
                "PUERTA: una version de parche invalida toda medida cacheada y el fallo es "
                "un acierto de cache EN SILENCIO, no un error (AGENTS.md 19). Aborta.")
    else:
        print("  versiones      OK")

    v["producer_sha256"] = None
    raiz_codigo = str(raiz_codigo)
    raiz_medida = str(raiz_medida) if raiz_medida else raiz_codigo
    v["raiz_extraccion"] = raiz_codigo
    v["raiz_medida"] = raiz_medida
    print(f"  raiz extraccion {raiz_codigo}")
    print(f"  raiz medida     {raiz_medida}"
          + ("   (la MISMA: si es una copia congelada, la huella no vale)"
             if raiz_medida == raiz_codigo else ""))
    # la huella se pide al arbol de MEDIDA, en un subproceso para no fijar el import
    import subprocess
    try:
        cod = ("import sys; sys.path.insert(0, r'''%s''');"
               "from fencingly_v2.measurement_artifacts import measurement_producer;"
               "print(measurement_producer()['producer_sha256'])" % raiz_medida)
        h = subprocess.run([sys.executable, "-c", cod], capture_output=True, text=True,
                           timeout=300).stdout.strip().splitlines()[-1]
        v["producer_sha256"] = h
        v["producer_ok"] = str(h).startswith(PRODUCER_ESPERADO)
        print(f"  producer_sha256 {h}")
        print(f"                  esperado {PRODUCER_ESPERADO}...  ok={v['producer_ok']}")
        if not v["producer_ok"] and estricto:
            raise SystemExit(
                "PUERTA: la huella del productor no es la de produccion. Si has apuntado "
                "`raiz_medida` a una copia congelada, esto es el punto 19 de AGENTS.md: "
                "medir ahi da otra huella y NINGUN error. Apunta al arbol vivo.")
    except SystemExit:
        raise
    except Exception as exc:
        v["producer_error"] = repr(exc)
        v["producer_ok"] = None
        print(f"  producer_sha256 NO SE PUDO CALCULAR: {exc!r}")

    # parche de ventana: con NPZ largos pasa de no-op a ACTIVO. Se comprueba en la raiz
    # de MEDIDA, que es la que va a medir.
    f = Path(raiz_medida) / "fencingly_v2" / "measurement_artifacts.py"
    aplicado = None
    try:
        aplicado = "_ventana_de_analisis" in f.read_text(encoding="utf-8", errors="replace")
    except Exception:
        pass
    v["parche_ventana"] = aplicado
    print(f"  parche ventana  aplicado={aplicado}")
    if not aplicado:
        print("    AVISO: extraer es seguro; MEDIR no. Sin el parche, sobre ventanas largas")
        print("    las medidas de guardia salen del principio del clip, con hasta 269 grados")
        print("    de error en aw_fore_ang_guard y ningun aviso.")

    v["barra"] = motor()
    print(f"  barra           motor={v['barra']}"
          + ("" if v["barra"] == "tqdm" else "   (respaldo local; tqdm no esta aqui)"))
    try:
        import torch
        v["cuda"] = bool(torch.cuda.is_available())
        v["gpu"] = torch.cuda.get_device_name(0) if v["cuda"] else None
        print(f"  gpu             {v['gpu']}  cuda={v['cuda']}")
    except Exception:
        v["cuda"] = None
    return v


# ================================================================ PASO 1: manifiesto
ESQUEMA_MIN = ("clip_id",)
ESQUEMA_REC = ("clip_id", "tournament_id", "weapon")


def paso1_manifiesto(manifiesto: str | Path, extra: str | list | None = None,
                     arma: str = ARMA) -> pd.DataFrame:
    """QUE HACE: carga el manifiesto y le une fuentes EXTRA (fichero JSON/CSV o URL de JSON).
    QUE COMPRUEBA: esquema minimo, `clip_id` unico, y **filtra por arma explicitamente**.
    QUE IMPRIME: cuantos entran, cuantos se rechazan y por que.
    """
    print("=" * 78); print("PASO 1 - MANIFIESTO"); print("=" * 78)
    base = _lee_tabla(manifiesto)
    print(f"  manifiesto base : {len(base)} filas <- {manifiesto}")
    trozos = [base.assign(_fuente="base")]
    for i, e in enumerate(_como_lista(extra)):
        try:
            t = _lee_tabla(e)
        except Exception as exc:
            print(f"  extra[{i}] RECHAZADO ({e}): {exc!r}")
            continue
        falta = [c for c in ESQUEMA_MIN if c not in t.columns]
        if falta:
            print(f"  extra[{i}] RECHAZADO ({e}): falta {falta}; esquema minimo {ESQUEMA_MIN}")
            continue
        for c in ESQUEMA_REC:
            if c not in t.columns:
                t[c] = None
                print(f"  extra[{i}] aviso: sin columna '{c}', se rellena vacia")
        print(f"  extra[{i}] OK   : {len(t)} filas <- {e}")
        trozos.append(t.assign(_fuente=f"extra{i}"))

    d = pd.concat(trozos, ignore_index=True)
    n0 = len(d)
    d = d[d["clip_id"].notna()].copy()
    d["clip_id"] = d["clip_id"].astype(str)
    dup = int(d["clip_id"].duplicated().sum())
    d = d.drop_duplicates(subset="clip_id", keep="first")
    print(f"  filas unidas    : {n0}  ->  {len(d)} con clip_id unico ({dup} duplicados, "
          "se queda la PRIMERA, igual que AnalysisRepository con --npz-dir)")

    # --- FILTRO DE ARMA, explicito y comprobado; nunca por el nombre del fichero ---
    if "weapon" in d.columns:
        vis = d["weapon"].astype(str).str.lower().str.strip()
        ok = vis == arma
        nulo = d["weapon"].isna()
        print(f"  arma            : {arma}")
        print(f"    coinciden      : {int(ok.sum())}")
        print(f"    otras armas    : {int((~ok & ~nulo).sum())}  "
              f"{sorted(set(vis[~ok & ~nulo]))[:6]}")
        print(f"    sin arma (NULO): {int(nulo.sum())}  -> SE RECHAZAN, no se asumen "
              f"{arma} (el fichero se llame como se llame)")
        d = d[ok].copy()
    else:
        raise SystemExit("PUERTA: el manifiesto no tiene columna 'weapon'. No se infiere el "
                         "arma del nombre del fichero.")
    print(f"  manifiesto final: {len(d)} clips de {arma}")
    return d


def _como_lista(x):
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _lee_tabla(origen) -> pd.DataFrame:
    """CSV, JSON o JSONL, de disco o de una URL de JSON."""
    s = str(origen)
    if s.startswith(("http://", "https://")):
        with urllib.request.urlopen(s, timeout=60) as r:
            txt = r.read().decode("utf-8")
        return pd.DataFrame(json.loads(txt))
    p = Path(s)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, low_memory=False)
    if p.suffix.lower() == ".jsonl":
        return pd.DataFrame([json.loads(l) for l in p.open(encoding="utf-8") if l.strip()])
    return pd.DataFrame(json.loads(p.read_text(encoding="utf-8")))


# ================================================================ PASO 2: estado
def paso2_estado(manifiesto: pd.DataFrame, ws: str | Path) -> dict:
    """QUE HACE: calcula QUE ESTA HECHO. Es la base de datos de la reanudacion.
    QUE COMPRUEBA: el estado es **indice INTERSECCION .npz en disco**, nunca un contador.
    QUE IMPRIME: el conteo en cada eslabon, para ver cual deja caer clips.

    Por que asi: `AnalysisRepository.index()` **ignora en silencio** un directorio sin
    `index.csv`, y eso dejo 689 clips invisibles. Y `valid_id` exige `[A-Za-z0-9_-]{1,200}`,
    asi que un `clip_id` no-ASCII se cae del indice sin error (13 clips conocidos).
    """
    print("=" * 78); print("PASO 2 - ESTADO (la base de datos de que esta hecho)"); print("=" * 78)
    ws = Path(ws)
    dir_npz = ws / "npz"
    dir_npz.mkdir(parents=True, exist_ok=True)
    idx = dir_npz / "index.csv"

    en_manif = set(manifiesto["clip_id"].astype(str))
    en_disco = {p.stem for p in dir_npz.glob("*.npz")}
    en_indice: set[str] = set()
    if idx.is_file():
        try:
            en_indice = set(pd.read_csv(idx, usecols=["clip_id"], low_memory=False)["clip_id"].astype(str))
        except Exception as exc:
            print(f"  index.csv ILEGIBLE: {exc!r}")
    import re
    valido = re.compile(r"^[A-Za-z0-9_-]{1,200}$")
    no_ascii = {c for c in en_manif if not valido.match(c)}

    hecho = en_indice & en_disco
    pendiente = en_manif - hecho - no_ascii
    huerfano_npz = en_disco - en_indice          # .npz sin fila -> INVISIBLE para el lector
    huerfano_fila = en_indice - en_disco         # fila sin .npz -> el lector fallara al abrirlo

    print(f"  en el manifiesto            : {len(en_manif)}")
    print(f"  .npz en disco               : {len(en_disco)}")
    print(f"  filas en index.csv          : {len(en_indice)}")
    print(f"  HECHO (indice AND disco)    : {len(hecho)}")
    print(f"  pendiente                   : {len(pendiente)}")
    print(f"  .npz sin fila en el indice  : {len(huerfano_npz)}   <- INVISIBLES para "
          "AnalysisRepository; hay que reindexar, no reextraer")
    print(f"  filas sin .npz              : {len(huerfano_fila)}  <- el lector fallara; "
          "no emitir fila sin fichero")
    print(f"  clip_id no-ASCII descartados: {len(no_ascii)}  <- hueco conocido y aceptado; "
          "NO relajar el validador")
    return {"manifiesto": en_manif, "disco": en_disco, "indice": en_indice, "hecho": hecho,
            "pendiente": sorted(pendiente), "huerfano_npz": sorted(huerfano_npz),
            "huerfano_fila": sorted(huerfano_fila), "no_ascii": sorted(no_ascii)}


# ================================================================ PASO 3: extraccion
def _etapa_gpu(PL, P, prep, extractor, max_gap: int = MAX_GAP_REDETECT):
    """Copia de `pipeline.stage_gpu` con UN solo cambio: pasa `max_gap_redetect`.

    `stage_gpu` llama a `P.fill_gaps(out, extractor, frames)` sin ese argumento, y bajarlo de
    25 a 12 recorta el 92,8 % del tiempo de relleno. Se replica en vez de tocar `pipeline.py`
    para no cambiar un fichero de produccion; `pose.py` NO esta en MEASUREMENT_SOURCES ni en
    MOTION_SOURCES, asi que el valor no mueve ninguna huella.
    """
    clip, frames, f0, f1 = prep["clip"], prep["frames"], prep["f0"], prep["f1"]
    if not len(frames):
        return {"meta": {"error": "empty decoded window"}}
    if f1 - f0 != len(frames):
        return {"meta": {"error": "decoded window length does not match f0/f1"}}
    dets = extractor.detect_window(clip, f0, f1, frames=frames)
    if len(dets) != len(frames):
        return {"meta": {"error": "detection count does not match decoded frames"}}
    scores = P.score_tracks(clip, dets, frames_iter=frames)
    out = P.assemble(clip, dets, scores, frames=frames, cuts=prep.get("cuts"))
    if "error" not in out["meta"]:
        out = P.fill_gaps(out, extractor, frames, max_gap_redetect=max_gap)
    out["meta"].update({"f0": f0, "f1": f1, "fps": clip.fps, "size": [clip.w, clip.h]})
    return out


def paso3_extrae(pendientes: list[str], rutas: dict, ws: str | Path, extractor,
                 PL, P, V, max_gap: int = MAX_GAP_REDETECT,
                 ram_max_gib: float = 6.0) -> pd.DataFrame:
    """QUE HACE: extrae a DURACION COMPLETA y CADENCIA NATIVA.
    QUE COMPRUEBA por clip: `f0 == 0` y `f1 == n_frames`. Si falla, la fila queda marcada.
    QUE IMPRIME: dos barras (corrida y clip en curso) con ritmo y tiempo restante.
    """
    print("=" * 78); print("PASO 3 - EXTRACCION"); print("=" * 78)
    print(f"  ventana: pre_s = post_s = {VENTANA_S} s, use_clock=False -> f0=0, f1=T")
    print(f"  max_gap_redetect = {max_gap}")
    ws = Path(ws); dir_npz = ws / "npz"; dir_npz.mkdir(parents=True, exist_ok=True)
    # Camino de 0 pendientes: pasa CADA VEZ que se relanza el cuaderno ya terminado.
    # Se devuelve un DataFrame con las columnas esperadas, no uno vacio sin columnas, para
    # que los pasos 4 y siguientes no rompan con KeyError('estado').
    COLS = ["clip_id", "estado", "error", "completo", "_fila_indice", "n_frames_window",
            "n_frames_clip", "src_fps", "dup_rate", "dup_mode", "coverage_L", "coverage_R",
            "cont_fps", "cont_n_frames", "t_total_s"]
    if not pendientes:
        print("  0 pendientes: nada que extraer (reanudacion completa).")
        return pd.DataFrame({c: pd.Series(dtype="object") for c in COLS})
    filas = []
    ext = Barra(total=len(pendientes), desc="corrida", unit="clip")
    t0g = time.time()
    for cid in pendientes:
        ruta = rutas.get(cid)
        f = {"clip_id": cid}
        if not ruta or not Path(ruta).is_file():
            f["estado"] = "sin_mp4"
            filas.append(f); ext.update(1); continue
        t0 = time.perf_counter()
        try:
            clip = V.Clip(ruta)
            proy = clip.n * clip.w * clip.h * 3 / 2**30
            f.update({"cont_fps": round(clip.fps, 4), "cont_n_frames": clip.n,
                      "cont_w": clip.w, "cont_h": clip.h,
                      "ram_gib_proyectada": round(proy, 2)})
            if proy > ram_max_gib:
                f["estado"] = "ram_excedida"
                f["error"] = f"{proy:.2f} GiB > {ram_max_gib} GiB"
                filas.append(f); ext.update(1); continue
            b = Barra(total=max(clip.n, 1), desc=f"  {cid[:20]:20s} {clip.w}x{clip.h} "
                      f"{clip.fps:.0f}fps", unit="fr", leave=False)
            prep = PL.stage_prepare(Path(ruta), VENTANA_S, VENTANA_S, None,
                                    use_clock=False)
            b.update(clip.n)
            b.close()
            f["estado"] = prep.get("status")
            if prep.get("status") != "prepared":
                f["error"] = prep.get("error")
                filas.append(f); ext.update(1); continue
            lr = prep["lights"]
            f0_, f1_ = prep["f0"], prep["f1"]
            f.update({"f0": f0_, "f1": f1_, "fps": lr["fps"], "n_frames_lr": lr["n_frames"],
                      "completo": bool(f0_ == 0 and f1_ == lr["n_frames"]),
                      "frames_perdidos_cabeza": int(f0_),
                      "frames_perdidos_cola": int(lr["n_frames"] - f1_)})
            ft = lr.get("f_touch")
            if ft is not None:
                f["pre_s_real"] = round((ft - f0_) / lr["fps"], 3)
                f["post_s_real"] = round((f1_ - 1 - ft) / lr["fps"], 3)
            out = _etapa_gpu(PL, P, prep, extractor, max_gap)
            if "error" in out["meta"]:
                f["estado"] = "pose_fail"; f["error"] = out["meta"]["error"]
                filas.append(f); ext.update(1); continue
            npz = dir_npz / f"{cid}.npz"
            if npz.exists():
                npz.unlink()
            prep.pop("clip", None)
            fila = PL.stage_save(Path(ruta), prep, out, npz, time.time(), overwrite=True)
            f["estado"] = fila.get("status")
            f["_fila_indice"] = fila
            f["bytes_npz"] = npz.stat().st_size if npz.is_file() else None
            for k in ("n_frames_window", "n_frames_clip", "src_fps", "dup_rate", "dup_mode",
                      "pre_s_eff", "window_source", "coverage_L", "coverage_R"):
                if k in fila:
                    f[k] = fila[k]
            del prep, out
        except Exception as exc:
            f["estado"] = "error"; f["error"] = repr(exc)
            f["traza"] = traceback.format_exc(limit=3)
        f["t_total_s"] = round(time.perf_counter() - t0, 2)
        filas.append(f)
        ext.update(1)
        n = len(filas); ok = sum(1 for x in filas if x.get("estado") == "ok")
        ext.set_postfix(ok=ok, fallo=n - ok, s_clip=f"{(time.time()-t0g)/max(1,n):.1f}")
    ext.close()
    d = pd.DataFrame(filas)
    if len(d):
        print(f"\n  ok {int((d['estado'] == 'ok').sum())} de {len(d)}")
        if "completo" in d:
            print(f"  completos (f0==0 y f1==T): {int(d['completo'].fillna(False).sum())}")
        for e, k in d["estado"].value_counts().items():
            print(f"    {e:16s} {k}")
    return d


# ================================================================ PASO 4: indice
def paso4_indice(resultado: pd.DataFrame, ws: str | Path, modelo_index: str | Path,
                 manifiesto: pd.DataFrame | None = None) -> pd.DataFrame:
    """QUE HACE: escribe `npz/index.csv` con el ORDEN DE COLUMNAS de un indice que ya carga.
    QUE COMPRUEBA: emite fila **solo** si su `.npz` esta en disco, y luego cuenta con el
    LECTOR REAL, no con las filas.
    QUE IMPRIME: filas, columnas, y cuantos clips ve el lector.
    """
    print("=" * 78); print("PASO 4 - INDICE, CONTADO POR EL LECTOR REAL"); print("=" * 78)
    ws = Path(ws); dir_npz = ws / "npz"
    fok = [r["_fila_indice"] for _, r in resultado.iterrows()
           if r.get("estado") == "ok" and isinstance(r.get("_fila_indice"), dict)]
    if not fok:
        print("  sin filas ok: no se escribe index.csv"); return pd.DataFrame()
    cols = list(pd.read_csv(modelo_index, nrows=1, low_memory=False).columns)
    idx = pd.DataFrame(fok)
    if manifiesto is not None:
        m = manifiesto.set_index("clip_id")
        for c in m.columns:
            if c in cols and c not in idx.columns:
                idx[c] = idx["clip_id"].map(m[c])
    for c in cols:
        if c not in idx:
            idx[c] = None
    extra = [c for c in idx.columns if c not in cols]
    idx = idx[cols + extra]
    tiene = idx["clip_id"].map(lambda c: (dir_npz / f"{c}.npz").is_file())
    if int((~tiene).sum()):
        print(f"  {int((~tiene).sum())} filas SIN .npz -> no se emiten")
    idx = idx[tiene]
    tmp = dir_npz / ".index.csv.tmp"
    idx.to_csv(tmp, index=False)
    os.replace(tmp, dir_npz / "index.csv")
    print(f"  escrito: {len(idx)} filas x {len(idx.columns)} columnas "
          f"({len(cols)} del modelo + {len(extra)} extra)")
    return idx


def paso4b_verifica_lector(ws: str | Path, raiz_webapp: str | Path, esperados: set) -> dict:
    """QUE COMPRUEBA: `AnalysisRepository.index()` ve lo que creemos. Es LA comprobacion."""
    ws = Path(ws)
    p = str(Path(raiz_webapp) / "webapp_react")
    if p not in sys.path:
        sys.path.insert(0, p)
    try:
        from analysis_store import AnalysisRepository
    except Exception as exc:
        print(f"  lector real NO disponible: {exc!r}")
        return {"ok": None}
    cache = ws / "_cache_lector"; cache.mkdir(parents=True, exist_ok=True)
    rep = AnalysisRepository([str(ws / "npz")], str(cache))
    di = rep.index()
    vistos = set(di["clip_id"].astype(str)) if "clip_id" in di else set()
    inv = esperados - vistos
    print(f"  LECTOR REAL: {len(di)} filas, {len(di.columns)} columnas")
    print(f"  clips vistos: {len(vistos)} de {len(esperados)} esperados")
    print(f"  INVISIBLES  : {len(inv)}" + (f"  {sorted(inv)[:10]}" if inv else "  ninguno"))
    return {"ok": not inv, "n": len(di), "vistos": vistos, "invisibles": sorted(inv)}


# ================================================================ PASO 5: rodaja
def paso5_rodaja(ws: str | Path, t_max: int = 135, anclaje: str = "tocado",
                 pre_s: float = 4.0, post_s: float = 0.3, aplicar: bool = False) -> pd.DataFrame:
    """QUE HACE: **mando SEPARADO** de la rodaja de entrenamiento. Extraer y entrenar son dos
    decisiones y esta celda no toca los NPZ: solo dice que se perderia.

    POR QUE ESTA SEPARADO: `store.T_MAX = 135` y el almacen **alinea por la derecha**. Un clip
    de 500 fotogramas a 50 fps se queda con los ultimos 135 = **2,70 s, el 27 % del clip**, y
    tira el resto **sin dar error**. Y `dataset.prepare` rehace el eje a
    `sample_time.TARGET_FPS = 30,0`, que en los clips de mas de 30 fps borra el 44,4 % de los
    fotogramas. Ninguna de las dos cosas avisa.

    QUE IMPRIME: por clip, cuanto sobrevive con este anclaje y este `t_max`.
    """
    print("=" * 78); print("PASO 5 - RODAJA DE ENTRENAMIENTO (mando aparte)"); print("=" * 78)
    print(f"  t_max={t_max}  anclaje={anclaje}  pre_s={pre_s} post_s={post_s}  "
          f"aplicar={aplicar}")
    ws = Path(ws)
    filas = []
    for p in sorted((ws / "npz").glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(z["meta_json"].item())
            T = len(z["poses"])
        fps = meta["fps"]; ft = meta.get("f_touch")
        if anclaje == "tocado" and ft is not None:
            a = max(0, int(ft - pre_s * fps)); b = min(T, int(ft + post_s * fps) + 1)
        elif anclaje == "allez" and meta.get("f_allez") is not None:
            a = int(meta["f_allez"]); b = T
        else:
            a, b = 0, T
        n_rod = b - a
        n_30 = int(round(n_rod / fps * 30.0))
        n_store = min(n_30, t_max)
        filas.append({"clip_id": p.stem, "fps": round(fps, 2), "T_nativo": T,
                      "T_rodaja": n_rod, "T_a_30fps": n_30, "T_en_almacen": n_store,
                      "s_en_almacen": round(n_store / 30.0, 2),
                      "frac_del_clip": round(n_store / 30.0 * fps / max(T, 1), 3),
                      "truncado_por_T_MAX": bool(n_30 > t_max),
                      "perdido_por_30fps": max(0, n_rod - n_30)})
    d = pd.DataFrame(filas)
    if len(d):
        print(f"\n  clips                        : {len(d)}")
        print(f"  fotogramas nativos            : {int(d['T_nativo'].sum())}")
        print(f"  tras la rodaja                : {int(d['T_rodaja'].sum())}")
        print(f"  tras el remuestreo a 30 fps   : {int(d['T_a_30fps'].sum())}")
        print(f"  tras T_MAX={t_max}                : {int(d['T_en_almacen'].sum())}")
        print(f"  clips truncados por T_MAX     : {int(d['truncado_por_T_MAX'].sum())} de {len(d)}")
        print(f"  fraccion del clip que entrena : mediana {d['frac_del_clip'].median():.3f}")
        print(f"  fotogramas perdidos por 30 fps: {int(d['perdido_por_30fps'].sum())}")
        peor = d.nsmallest(3, "frac_del_clip")
        print("\n  los 3 que menos conservan:")
        print(peor[["clip_id", "fps", "T_nativo", "T_en_almacen", "s_en_almacen",
                    "frac_del_clip"]].to_string(index=False))
    if not aplicar:
        print("\n  aplicar=False: NO se ha tocado ningun NPZ. Esto es un informe, no una accion.")
    return d


# ================================================================ PASO 6: tabla y huecos
def paso6_tabla(ws: str | Path, salida_parquet: str | Path | None = None) -> pd.DataFrame:
    """QUE HACE: tabla por (clip_id, frame_idx) con lo derivable del NPZ, en Parquet.
    QUE COMPRUEBA: que `escala_r` distinga NaN (indefinido) de valor.
    QUE IMPRIME: filas, columnas, tamano y cobertura de cada columna.
    """
    print("=" * 78); print("PASO 6 - TABLA POR FOTOGRAMA (Parquet)"); print("=" * 78)
    ws = Path(ws)
    trozos = []
    ficheros = sorted((ws / "npz").glob("*.npz"))
    b = Barra(total=len(ficheros), desc="tabla", unit="clip")
    for p in ficheros:
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(z["meta_json"].item())
            poses = z["poses"].astype(np.float32)
            boxes = z["boxes"].astype(np.float32)
            src = z["source"]; luces = z["lights"]; t_s = z["time_s"]
            pan = z["pan"]; dup = z["dup"]
        T = len(poses); fps = meta["fps"]; ft = meta.get("f_touch")
        alto = boxes[..., 3] - boxes[..., 1]
        alto[~np.isfinite(alto)] = np.nan
        alto[alto <= 0] = np.nan
        a0 = max(0, int(ft - 4.0 * fps)) if ft is not None else 0
        a1 = min(T, int(ft + 0.3 * fps) + 1) if ft is not None else T
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(alto[a0:a1], axis=0)
            med = np.where(np.isfinite(med), med, np.nanmedian(alto, axis=0))
            escala_r = np.nanmax(alto / med[None, :], axis=1)
        trozos.append(pd.DataFrame({
            "clip_id": p.stem, "frame_idx": np.arange(T, dtype=np.int32),
            "t_s": t_s, "dt_toque_s": (np.arange(T) - (ft if ft is not None else 0)) / fps,
            "fps_real": np.float32(fps), "src_fps": np.float32(meta.get("src_fps") or np.nan),
            "src_ratio": np.float32((fps / meta["src_fps"]) if meta.get("src_fps") else np.nan),
            "dup": dup, "pan": pan,
            "src_L": src[:, 0].astype(np.int8), "src_R": src[:, 1].astype(np.int8),
            "alto_L": alto[:, 0], "alto_R": alto[:, 1], "escala_r": escala_r,
            "luz_rojo_L": luces[:, 0], "luz_verde_R": luces[:, 1],
            "luz_blanco_L": luces[:, 2], "luz_blanco_R": luces[:, 3],
        }))
        b.update(1)
    b.close()
    if not trozos:
        print("  sin NPZ"); return pd.DataFrame()
    d = pd.concat(trozos, ignore_index=True)
    print(f"\n  filas {len(d)}  columnas {len(d.columns)}")
    print("\n  cobertura por columna (fraccion no nula):")
    for c in d.columns:
        if c in ("clip_id",):
            continue
        cob = float(pd.notna(d[c]).mean())
        print(f"    {c:16s} {cob:.3f}" + ("   <- NaN = INDEFINIDO, no 0" if cob < 1 else ""))
    if salida_parquet:
        sp = Path(salida_parquet)
        try:
            d.to_parquet(sp, compression="snappy", index=False)
            print(f"\n  Parquet: {sp}  {sp.stat().st_size/2**20:.2f} MiB "
                  f"({sp.stat().st_size/len(d):.1f} bytes/fila)")
        except Exception as exc:
            # `pyarrow`/`fastparquet` no estan en `.venv` ni en `.venv-gpu`; en Colab si.
            # No se instala nada: se cae a CSV comprimido, que siempre funciona.
            alt = sp.with_suffix(".csv.gz")
            d.to_csv(alt, index=False, compression="gzip")
            print(f"\n  Parquet NO disponible ({type(exc).__name__}): "
                  "falta pyarrow/fastparquet y no se instalan paquetes.")
            print(f"  Respaldo CSV.gz: {alt}  {alt.stat().st_size/2**20:.2f} MiB "
                  f"({alt.stat().st_size/len(d):.1f} bytes/fila)")
            print("  En Colab pyarrow SI esta, asi que alli saldra Parquet.")
    return d


HUECOS_DECLARADOS = """
================================================================================
HUECOS DECLARADOS: lo que este cuaderno NO resuelve, con su motivo medido
================================================================================

1. PERIODO DEL ASALTO (1/3, 2/3, 3/3)
   Estado: HUECO. La barra del marcador esta en 24 de 24 clips, y en 0 de 24 "solo
   fuera de la ventana vieja": esta tambien durante la accion, asi que **el periodo
   NO depende de la duracion completa** (retractado de un informe anterior).
   16 de 24 clips tienen un estilo con casilla de periodo declarada (66,7 %).
   Visto a x6: `tokyo22` muestra "R1" nitido y `parse_period` ya lo cubre;
   `russian` (kazan) tiene la casilla VACIA. n = 1 clip por estilo: es una sonda.
   PRECISION DE LECTURA: SIN MEDIR Y SIN REFERENCIA. No hay motor de OCR en
   `.venv` ni en `.venv-gpu` (`paddleocr`, `paddle`, `easyocr`, `pytesseract`
   ausentes). En Colab si lo hay: el cuaderno puede INTENTARLO y medir su propia
   precision. Lo que no se hace es enviar la columna como validada.

2. REPETICION / CAMARA LENTA
   Estado: HUECO, y es el RIESGO ABIERTO NUMERO UNO de la ruta de duracion completa.
   `PeriodicSlipperyChipmunk` (50 fps, tokyoolympics2020) abre con primeros planos
   de repeticion del tocado, con barrido de anillos olimpicos, y solo despues pasa
   al plano en vivo. Y **la cobertura de pose es 1,000 en L y en R**: el detector
   encuentra a los dos tiradores porque ESTAN AHI, en otro encuadre y a otra escala,
   y ninguna comprobacion existente lo marca. Que la cobertura baje era una alarma;
   aqui no hay ninguna. `cuts` reporta 1 donde a la vista hay tres planos.
   Las CUATRO vias probadas y por que ninguna caza:
     - `dup_rate`: ciego a 50 fps (mediana 0,000; cero `dup_mode="regular"`).
     - `cuts`: sub-detecta (1 contra >=3 planos visibles).
     - `reloj_parado_todo_el_clip`: >30,5 fps 20,0 % [14,2, 27,4] contra <=30,5 fps
       20,5 % [11,2, 34,5]. Intervalos solapados, -0,5 pp: no separa.
     - escala aparente de caja: el control da r_max 1,58 contra un umbral de 1,60 y
       un clip limpio da 1,32. **1,32 contra 1,58 se solapa.** NO se bajo el umbral
       a 1,55 para que el control pasara: un umbral elegido para que pase el control
       es un umbral que mide el control.
   Se emite `escala_r` CRUDA y continua (definida en 91,8 % de los fotogramas,
   NaN = indefinido, no 0) para poder reumbralizarla cuando haya etiqueta.

3. TELECINE 25->30: EL ARTEFACTO MAYORITARIO, y no estaba en ninguna lista
   Medido sobre los 15.452 clips del indice vivo, con `dup_mode` en el 100 % de
   ellos. Bandas por fps NOMINAL REDONDEADO (no por intervalo laxo: eso movia los
   denominadores y dio dos cifras distintas con el mismo nombre):

     nominal | clips  | patron regular | ratio >= 2
        25   |  2.894 |    10 ( 0,3 %) |     0
        30   | 11.211 | 8.503 (75,8 %) |    35
        50   |    649 |     0 ( 0,0 %) |     0
        60   |    170 |    66 (38,8 %) |    19 (11,2 % de la banda)

   a) EL PROBLEMA ES LA BANDA DE 30, que es la mayoritaria: **8.503 clips (75,8 %)**
      con telecine 25->30. `dup_rate` mediana **0,164**, `dup_period` **6,0
      fotogramas exactos**, `src_fps` mediana **25,0**, ratio **1,20**. Uno de cada
      seis fotogramas es un DUPLICADO, y un duplicado da **velocidad cero en un paso
      y doble en el siguiente**.
   b) LA CAMARA LENTA ES RESIDUAL: **54 clips de 15.452, el 0,35 %**, con ratio >= 2.
      Ratios observados 2,0 · 2,4 · 4,0 · 6,0 · **12,0**. Un ratio de 12 significa
      **once fotogramas de velocidad cero seguidos de uno con velocidad doce veces
      mayor**: eso es lo que rompe una derivada, no la cadencia nominal.
      Sedes: eurochamps2022 (19), tbilisifoil2023 (19), worldchamps2022 (8),
      serbiafoil2022 (4), anaheimfoil2019 (3), parisfoil2022 (1).
   c) A 50 fps: CERO clips con patron regular. Es lo que sostiene el argumento del
      paso de 20 ms. Y a 60 fps el 38,8 % tiene duplicados pero solo 19 de 170 estan
      multiplicados de verdad; el resto es conversion 50->60 a ratio 1,20, o sea
      material real reempaquetado, NO lento.

   CONSECUENCIAS, y son tres:
     - `src_fps` Y el ratio `fps/src_fps` VIAJAN LOS DOS COMO COLUMNAS. Sin ellos,
       "50 fps" y "60 fps con origen a 15 fps" se tratan igual y no lo son.
     - EL SUAVIZADO VA ANTES DE DERIVAR, y su ventana se justifica **contra un
       duplicado cada seis fotogramas**, no contra la camara lenta: por debajo de ~7
       fotogramas no absorbe el artefacto. Guardar cruda Y suavizada.
     - TODA MEDIDA DE VELOCIDAD SE REPORTA CON `src_fps` AL LADO, no solo con la
       nominal.

   AVISO DE DENOMINADORES, y es importante: el censo completo (`fps_entero.json`)
   tiene **29.220 clips con fps** (20.938 a 30, 4.883 a 25, 1.547 a 50, 816 a 60),
   pero `dup_mode` esta medido en **15.452 de ellos (52,9 % del censo, 100 % del
   indice vivo)**. **NO multiplicar la tasa por el censo:** la muestra medida no es
   aleatoria respecto al censo, es la que se llego a procesar, y el residuo sin
   procesar esta sesgado (la tasa de deteccion correlaciona r = -0,861 con la
   fraccion ya extraida). Citar las dos cifras con su n y no cruzarlas.

4. CONFUSOR DE SEDE EN LA CADENCIA
   Los 877 clips de alta cadencia salen de TRES sedes: tokyoolympics2020 (637,
   el 72,6 %), tokyofoil2022 (163), tbilisifoil2023 (77). Y los 154 clips de 50 fps
   con corte de plano son **todos** de tokyoolympics2020, los 154.
   (Con `fps.round() >= 50`, que es el convenio. El "904 / 190" que circulo antes
   venia de intervalos laxos y NO se reproduce con ningun umbral de esta tabla:
   `fps >= 49` da 880 / 166. La conclusion no cambia, la cifra si.)
   O sea: **"alta cadencia" es en buena parte "Tokio 2020".** Cualquier comparacion
   por cadencia arrastra el confusor de sede, y alguien lo va a leer como un efecto
   de la cadencia. Es la tercera vez en la sesion que un confusor de sede muerde.

5. ESCALA DEL ALLEZ
   Con `f0 = 0`, `calibration.allez_scale` rechaza por `window_not_at_clock_start`
   en todo clip cuyo allez no caiga en los primeros 0,7 s. La ventaja de la duracion
   completa es la contraria: las poses del allez QUEDAN DENTRO del NPZ y un consumidor
   puede rebanar desde `f_allez` despues, sin reextraer. Guardar desde el fotograma 0
   es la unica configuracion que permite elegir el ancla DESPUES.

6. MASCARA DEL MARCADOR
   Tapar `y > 0,82*H` NO BASTA: el marcador del arbitro con sus lamparas puede quedar
   POR ENCIMA de esa linea. Este cuaderno no recorta nada, pero cualquier consumidor
   de los NPZ que aplique esa mascara tiene que saber que es insuficiente.
================================================================================
"""


def paso7_huecos():
    print(HUECOS_DECLARADOS)

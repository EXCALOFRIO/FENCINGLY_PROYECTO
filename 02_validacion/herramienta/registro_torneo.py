"""Registro ENTRE clips del mismo torneo ("panorama del recinto"), para la zona del tocado (etapa 11).

POR QUE EXISTE (2026-10-09). `ZONA_COBERTURA`, `CAMARA_TRIPODE` y `RAYAS_ID` (`04_documentacion/
informes/`) miden que ~1.950 de 5.472 clips anotados no tienen homografia de pista confiable cerca
del tocado, y que la causa de fondo es de IDENTIFICACION: con una o dos rayas transversales a la
vista, la pista (14 x 1,5 m) es ambigua. Esos intentos buscaban la pista DENTRO del propio clip.

LA IDEA NUEVA: la camara de retransmision de un torneo es un TRIPODE que solo gira y hace zoom
(mismo principio que `pista_tripode.py`, pero aplicado ENTRE clips en vez de dentro de uno). Dos
cuadros de esa camara, aunque sean de clips distintos, estan relacionados por una homografia de
imagen COMPLETA `H(q->r) = K2 R K1^-1`, que vale para el fondo (publicidad, gradas, marcador, suelo,
rayas), no solo para la pista. Un cuadro sin homografia de pista se REGISTRA contra cuadros de otros
clips del MISMO torneo que si tienen homografia de produccion confiable (`motivo=ok`): se emparejan
rasgos del fondo ESTATICO (fuera de personas y del rotulo sobreimpreso), y

    H_pista(q) = H_pista(ref) . H(q->ref)

da la homografia de pista del cuadro sin respuesta, de la que salen la x y la zona de los pies de
`q` con las MISMAS funciones de siempre (`pista_lineas.to_m`, `zona_tocado.zona`).

QUE EXCLUYE: las personas (cajas de `poses_<clip>.npz`) y el rotulo FIE sobreimpreso. Se reutiliza
EXACTAMENTE la mascara de fondo de produccion (`pista_lineas.PisteFeatures.process`, que ya excluye
las capsulas de las personas y la banda `BANDA_ROTULO` de `zona_tocado`): no hace falta reinventarla,
y es la que ya corre en la etapa 11 (`ZT._rasgos` deja `pf.feats[t].valid` y `.gray`, A RESOLUCION DE
ANALISIS -- 640 de ancho -- que es donde se hace el emparejamiento SIFT; la homografia de pista
opera en pixeles ORIGINALES, asi que aqui se escala con `s` como hace `PisteFeatures` con `G`).

LA HOMOGRAFIA DE PISTA DE LA REFERENCIA NO ESTA GUARDADA EN DISCO: `zona_<clip>.csv` solo trae la x
derivada, no `H`. Se RECALCULA con `pista_lineas.calibrate` sobre `zona_tocado._rasgos`, exactamente
como hace la etapa 11 (y como hizo el agente de `CAMARA_TRIPODE_2026-10-09.md` en su `cache.py`);
`construye_uno` comprueba que reproduce la x de produccion antes de usarla como referencia.

PASO 1 (viabilidad, SIN etiquetas, solo AJUSTE): clips con homografia de produccion, registrados
DEJANDO FUERA su propio clip, contra referencias de otros clips del mismo torneo; se compara la x
registrada con `x_tocado_m` de produccion (mediana y p90 de |error| en metros, fraccion registrada,
y todo por torneo). Script conductor y resultados en
`02_validacion/auditoria/registro_torneo_2026-10-09/`.

    python registro_torneo.py cache LISTA.txt              construye la cache (features/H/pies) de cada clip
    python registro_torneo.py empareja TORNEOS.txt SALIDA.jsonl.gz    empareja dentro de cada torneo (leave-one-out)

LISTA.txt: una fila `clip,torneo` (o solo `clip`, se busca el torneo en el catalogo).
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import pathlib
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "4")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "01_extraccion" / "codigo"))

import numpy as np  # noqa: E402

CACHE = pathlib.Path(os.environ.get("FENCINGLY_REGISTRO_CACHE",
                                     "D:/FENCINGLY_SCRATCH/agente_registro/cache"))
THUMB_H, THUMB_W = 27, 48
"""El descriptor global barato para la recuperacion top-k (en vez de DINOv2/GPU): el gris enmascarado
reducido a un rectangulo pequenio, centrado y normalizado a norma 1. Aqui, con pozas por torneo
pequenias (<= 50), se guarda solo como diagnostico; el emparejamiento real prueba TODAS las
referencias del torneo (ver cabecera)."""
SIFT_KW = dict(nfeatures=4000, contrastThreshold=0.02, edgeThreshold=15)
RATIO = 0.8
RANSAC_PX = 3.0
MIN_MATCHES = 8
"""Minimo de matches (tras el ratio test) para INTENTAR `findHomography`; por debajo no se intenta
y se cuenta como "sin matches bastantes", no como homografia rechazada."""


def _cv2():
    import cv2
    cv2.setNumThreads(1)
    return cv2


def _root(desc):
    if desc is None or len(desc) == 0:
        return np.zeros((0, 128), np.float32)
    d = desc.astype(np.float32)
    d /= (np.abs(d).sum(1, keepdims=True) + 1e-7)
    return np.sqrt(d)


def _thumb(cv2, gray, valid):
    v = valid > 0
    mu = float(gray[v].mean()) if v.any() else float(gray.mean())
    g2 = np.where(v, gray.astype(np.float32), mu)
    small = cv2.resize(g2, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
    small = small - small.mean()
    n = float(np.linalg.norm(small))
    return (small / n).astype(np.float32).ravel() if n > 1e-6 else small.astype(np.float32).ravel()


def construye_uno(clip, torneo):
    """QUE DEVUELVE: (clip, motivo). Escribe `CACHE/<clip>.pkl` si se puede: la homografia de pista
    recalculada, los pies, la x de produccion reproducida, y los rasgos SIFT (RootSIFT) + el
    descriptor global del cuadro usado, fuera de personas y del rotulo."""
    f = CACHE / ("%s.pkl" % clip)
    if f.is_file():
        return clip, "ya"
    import rutas as RU
    import zona_tocado as ZT
    import pista_lineas as PL
    p = RU.TRABAJO / ("zona_%s.csv" % clip)
    if not p.is_file() or not p.stat().st_size:
        return clip, "sin zona_"
    with open(p, encoding="utf-8") as fh:
        row = next(csv.DictReader(fh), None) or {}
    if row.get("motivo") != "ok":
        return clip, "no ok"
    video = RU.video(clip)
    if not video:
        return clip, "sin video"
    ppath = RU.TRABAJO / ("poses_%s.npz" % clip)
    if not ppath.is_file():
        return clip, "sin poses"
    z = np.load(ppath, allow_pickle=True)
    if "KP" not in z.files:
        return clip, "poses con otro esquema"
    cv2 = _cv2()
    KP, SC, B, origen = z["KP"], z["SC"], z["B"], z["origen"]
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    fps = float(z["fps"]) if "fps" in z.files else 30.0
    try:
        tf = int(float(row["cuadro_tocado"]))
    except (KeyError, TypeError, ValueError):
        return clip, "sin cuadro_tocado"
    k26 = ZT.a_halpe(KP, SC)
    Bpx = B * np.array([W, H, W, H])
    pf = ZT._rasgos(video, W, H, T, tf, Bpx, k26, ZT.BANDA_ROTULO)
    Tn = len(pf.feats)
    if Tn == 0:
        return clip, "sin cuadros"
    pies = np.full((Tn, 2, 2), np.nan, np.float32)
    alt = np.full((Tn, 2), np.nan, np.float32)
    for t in range(Tn):
        for s in (0, 1):
            if origen[t, s] in (0, 1) and np.isfinite(Bpx[t, s]).all():
                pies[t, s] = ZT.apoyo(k26[t, s], Bpx[t, s])
                alt[t, s] = ZT.largo_cuerpo(k26[t, s])
    cal = PL.calibrate(pf.feats, pies, alt, ZT.CFG_PISTA, (W, H))
    Hs, conf = cal["H"], cal["conf"]
    tq = min(tf, Tn - 1)
    w = int(ZT.VENTANA_S * fps)
    cand = [t for t in range(max(0, tq - w), min(Tn, tq + 2))
            if np.isfinite(Hs[t]).all() and conf[t] >= ZT.CFG_PISTA["min_conf"]]
    if not cand:
        return clip, "no reproduce (sin candidato)"
    tt = min(cand, key=lambda t: abs(t - tq))
    xs_repro = [float(PL.to_m(Hs[tt], pies[tt, s][None])[0, 0]) if np.isfinite(pies[tt, s]).all()
                else None for s in (0, 1)]
    disco_tt = row.get("cuadro_usado")
    ok_repro = disco_tt not in (None, "") and int(float(disco_tt)) == tt
    feat = pf.feats[tt]
    sift = cv2.SIFT_create(**SIFT_KW)
    kp, desc = sift.detectAndCompute(feat.gray, feat.valid)
    pts = np.float32([p.pt for p in kp]) if kp else np.zeros((0, 2), np.float32)
    rdesc = _root(desc)
    thumb = _thumb(cv2, feat.gray, feat.valid)
    xprod = row.get("x_tocado_m")
    d = {
        "clip": clip, "torneo": torneo, "s": float(feat.s), "W": W, "H": H,
        "H_pista": np.asarray(Hs[tt], np.float64),
        "pies": np.asarray(pies[tt], np.float32),
        "x_izq_m": xs_repro[0], "x_der_m": xs_repro[1],
        "x_prod": float(xprod) if xprod not in (None, "") else None,
        "zona_prod": row.get("zona") or None,
        "conf": float(conf[tt]), "cuadro_usado": int(tt), "ok_repro": bool(ok_repro),
        "n_kp": int(len(pts)),
        "pts": pts.astype(np.float32), "desc": rdesc.astype(np.float16), "thumb": thumb,
    }
    tmp = f.with_suffix(".pkl.part")
    with open(tmp, "wb") as fh:
        pickle.dump(d, fh, protocol=4)
    tmp.replace(f)
    return clip, ("ok" if ok_repro else "ok(no repro exacta)")


def _carga(clip):
    f = CACHE / ("%s.pkl" % clip)
    if not f.is_file():
        return None
    with open(f, "rb") as fh:
        return pickle.load(fh)


CACHE_Q = pathlib.Path(os.environ.get("FENCINGLY_REGISTRO_CACHE_Q",
                                       "D:/FENCINGLY_SCRATCH/agente_registro/cache_query"))
"""La cache de los clips SIN homografia de produccion (Uso 1, respaldo): no hace falta `PL.calibrate`
(no hay `H_pista` que recuperar), asi que solo se decodifica HASTA el cuadro usado (como RAYAS_ID:
`cuadro_usado` si `motivo=ok`, si no `min(cuadro_tocado, T-1)`) y se llama UNA vez a
`PisteFeatures.process` (sin la cadena de movimiento de camara, que aqui no hace falta): mucho mas
barato que `construye_uno` (~0,3-0,5 s de decodificacion + SIFT, frente a los ~9-11 s de rasgos +
calibracion de una referencia)."""


def construye_query(clip, torneo):
    f = CACHE_Q / ("%s.pkl" % clip)
    if f.is_file():
        return clip, "ya"
    import rutas as RU
    import zona_tocado as ZT
    import pista_lineas as PL
    p = RU.TRABAJO / ("zona_%s.csv" % clip)
    if not p.is_file() or not p.stat().st_size:
        return clip, "sin zona_"
    with open(p, encoding="utf-8") as fh:
        row = next(csv.DictReader(fh), None) or {}
    mot = row.get("motivo") or ""
    if not (mot == "ok" or mot.startswith("sin homografia")):
        return clip, "motivo=%s" % mot
    video = RU.video(clip)
    if not video:
        return clip, "sin video"
    ppath = RU.TRABAJO / ("poses_%s.npz" % clip)
    if not ppath.is_file():
        return clip, "sin poses"
    z = np.load(ppath, allow_pickle=True)
    if "KP" not in z.files:
        return clip, "poses con otro esquema"
    cv2 = _cv2()
    KP, SC, B, origen = z["KP"], z["SC"], z["B"], z["origen"]
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    try:
        t = (int(float(row["cuadro_usado"])) if mot == "ok"
             else min(int(float(row["cuadro_tocado"])), T - 1))
    except (KeyError, TypeError, ValueError):
        return clip, "sin cuadro"
    k26 = ZT.a_halpe(KP, SC)
    Bpx = B * np.array([W, H, W, H])
    cap = cv2.VideoCapture(video)
    k = 0
    img = None
    while k <= t:
        if k < t:
            if not cap.grab():
                break
            k += 1
            continue
        ok, img = cap.read()
        break
    cap.release()
    if img is None:
        return clip, "no decodifica"
    if img.shape[1] != W or img.shape[0] != H:
        img = cv2.resize(img, (W, H))
    okb = np.isfinite(Bpx[t]).all(1)
    pf = PL.PisteFeatures(ZT.CFG_VIDEO, (W, H))
    feat = pf.process(img, Bpx[t][okb].astype(np.float32), ZT.BANDA_ROTULO * H, k26[t][okb])
    pies = np.full((2, 2), np.nan, np.float32)
    for s in (0, 1):
        if origen[t, s] in (0, 1) and np.isfinite(Bpx[t, s]).all():
            pies[s] = ZT.apoyo(k26[t, s], Bpx[t, s])
    sift = cv2.SIFT_create(**SIFT_KW)
    kp, desc = sift.detectAndCompute(feat.gray, feat.valid)
    pts = np.float32([p.pt for p in kp]) if kp else np.zeros((0, 2), np.float32)
    rdesc = _root(desc)
    thumb = _thumb(cv2, feat.gray, feat.valid)
    d = {
        "clip": clip, "torneo": torneo, "s": float(feat.s), "W": W, "H": H,
        "pies": np.asarray(pies, np.float32), "cuadro_usado": int(t), "motivo": mot,
        "x_prod": float(row["x_tocado_m"]) if row.get("x_tocado_m") not in (None, "") else None,
        "zona_prod": row.get("zona") or None, "zona_final_prod": row.get("zona_final") or None,
        "n_kp": int(len(pts)), "pts": pts.astype(np.float32), "desc": rdesc.astype(np.float16),
        "thumb": thumb,
    }
    tmp = f.with_suffix(".pkl.part")
    with open(tmp, "wb") as fh:
        pickle.dump(d, fh, protocol=4)
    tmp.replace(f)
    return clip, "ok"


def empareja(q, r, ratio=RATIO, ransac_px=RANSAC_PX, min_matches=MIN_MATCHES, cv2=None):
    """QUE DEVUELVE: dict `nmatches`, `ninliers`, `x` (la x de pista de los pies de `q` proyectados
    con `H_pista(ref) . H(q->ref)`), `xs` (por tirador), o None si a `q`/`r` les falta descriptor."""
    if cv2 is None:
        cv2 = _cv2()
    dq, dr = q["desc"].astype(np.float32), r["desc"].astype(np.float32)
    if len(dq) < 4 or len(dr) < 4:
        return None
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(dq, dr, k=2)
    good = [m for m, n in matches if n.distance > 1e-9 and m.distance < ratio * n.distance]
    if len(good) < min_matches:
        return {"nmatches": len(good), "ninliers": 0, "x": None, "xs": [None, None]}
    src = np.float32([q["pts"][m.queryIdx] for m in good])
    dst = np.float32([r["pts"][m.trainIdx] for m in good])
    Hm, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, ransac_px, confidence=0.999, maxIters=5000)
    if Hm is None or not np.isfinite(Hm).all():
        return {"nmatches": len(good), "ninliers": 0, "x": None, "xs": [None, None]}
    ninl = int(mask.sum()) if mask is not None else 0
    Sq = np.diag([q["s"], q["s"], 1.0])
    Sir = np.diag([1.0 / r["s"], 1.0 / r["s"], 1.0])
    H_full = Sir @ Hm @ Sq
    H_pista_q = r["H_pista"] @ H_full
    import pista_lineas as PL
    xs = []
    for foot in q["pies"]:
        if np.isfinite(foot).all():
            m = PL.to_m(H_pista_q, foot[None])[0]
            xs.append(float(m[0]) if np.isfinite(m).all() else None)
        else:
            xs.append(None)
    vx = [x for x in xs if x is not None]
    x = float(np.mean(vx)) if vx else None
    return {"nmatches": len(good), "ninliers": ninl, "x": x, "xs": xs}


def empareja_torneo(clips_cache, min_matches=MIN_MATCHES):
    cv2 = _cv2()
    out = []
    for qi, q in enumerate(clips_cache):
        resultados = []
        for r in clips_cache:
            if r["clip"] == q["clip"]:
                continue
            res = empareja(q, r, min_matches=min_matches, cv2=cv2)
            if res is not None:
                res = dict(res, ref=r["clip"], conf_ref=r["conf"], n_kp_ref=r["n_kp"])
                resultados.append(res)
        out.append({
            "clip": q["clip"], "torneo": q["torneo"], "x_prod": q["x_prod"],
            "zona_prod": q["zona_prod"], "n_kp": q["n_kp"], "resultados": resultados,
        })
    return out


def _lee_lista(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split(",")
            out.append((parts[0], parts[1] if len(parts) > 1 else None))
    return out


def cmd_cache(lista_path, procs=4):
    import catalogo as CA
    tareas = _lee_lista(lista_path)
    faltan_torneo = [c for c, t in tareas if t is None]
    df = CA.manifiesto().set_index("clip_id") if faltan_torneo else None
    tareas = [(c, t if t is not None else (df.loc[c, "torneo"] if c in df.index else None))
              for c, t in tareas]
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cuenta = {}
    with ProcessPoolExecutor(procs) as ex:
        fut = {ex.submit(construye_uno, c, t): c for c, t in tareas}
        for i, fu in enumerate(as_completed(fut), 1):
            try:
                c, motivo = fu.result()
            except Exception as ex2:
                c, motivo = fut[fu], "ERROR %s: %s" % (type(ex2).__name__, str(ex2)[:150])
            cuenta[motivo] = cuenta.get(motivo, 0) + 1
            if i % 50 == 0:
                print("%d/%d %.0f s  %s" % (i, len(tareas), time.time() - t0, cuenta), flush=True)
    print("hecho %.0f s  %s" % (time.time() - t0, cuenta), flush=True)


def cmd_empareja(torneos_path, salida_path, min_matches=MIN_MATCHES):
    torneos = [l.strip() for l in open(torneos_path, encoding="utf-8") if l.strip()]
    por_torneo = {}
    for f in CACHE.glob("*.pkl"):
        try:
            with open(f, "rb") as fh:
                d = pickle.load(fh)
        except Exception:
            continue
        if d["torneo"] in torneos:
            por_torneo.setdefault(d["torneo"], []).append(d)
    print("torneos con cache: %d de %d; clips en cache: %d"
          % (len(por_torneo), len(torneos), sum(len(v) for v in por_torneo.values())), flush=True)
    t0 = time.time()
    abre = gzip.open if salida_path.endswith(".gz") else open
    n = 0
    with abre(salida_path, "wt", encoding="utf-8") as out:
        for torneo, clips_cache in sorted(por_torneo.items()):
            if len(clips_cache) < 2:
                continue
            filas = empareja_torneo(clips_cache, min_matches=min_matches)
            for fila in filas:
                out.write(json.dumps(fila) + "\n")
                n += 1
            print("  %s: %d clips, %.0f s acumulado" % (torneo, len(clips_cache), time.time() - t0),
                  flush=True)
    print("hecho: %d filas en %.0f s -> %s" % (n, time.time() - t0, salida_path), flush=True)


def cmd_cache_query(lista_path, procs=4):
    import catalogo as CA
    tareas = _lee_lista(lista_path)
    faltan_torneo = [c for c, t in tareas if t is None]
    df = CA.manifiesto().set_index("clip_id") if faltan_torneo else None
    tareas = [(c, t if t is not None else (df.loc[c, "torneo"] if c in df.index else None))
              for c, t in tareas]
    CACHE_Q.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cuenta = {}
    with ProcessPoolExecutor(procs) as ex:
        fut = {ex.submit(construye_query, c, t): c for c, t in tareas}
        for i, fu in enumerate(as_completed(fut), 1):
            try:
                c, motivo = fu.result()
            except Exception as ex2:
                c, motivo = fut[fu], "ERROR %s: %s" % (type(ex2).__name__, str(ex2)[:150])
            cuenta[motivo] = cuenta.get(motivo, 0) + 1
            if i % 50 == 0:
                print("%d/%d %.0f s  %s" % (i, len(tareas), time.time() - t0, cuenta), flush=True)
    print("hecho %.0f s  %s" % (time.time() - t0, cuenta), flush=True)


def cmd_empareja_query(lista_path, salida_path, min_matches=MIN_MATCHES):
    """Uso 1 (respaldo): empareja cada clip de `CACHE_Q` contra TODAS las referencias (`CACHE`) de su
    MISMO torneo. No hay leave-one-out: la referencia nunca es el propio clip porque este no tiene
    `H_pista` (no esta en `CACHE`)."""
    tareas = _lee_lista(lista_path)
    clips_q = [c for c, _t in tareas]
    por_torneo_ref = {}
    for f in CACHE.glob("*.pkl"):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        por_torneo_ref.setdefault(d["torneo"], []).append(d)
    cv2 = _cv2()
    t0 = time.time()
    abre = gzip.open if salida_path.endswith(".gz") else open
    n = 0
    with abre(salida_path, "wt", encoding="utf-8") as out:
        for i, clip in enumerate(clips_q, 1):
            fq = CACHE_Q / ("%s.pkl" % clip)
            if not fq.is_file():
                continue
            with open(fq, "rb") as fh:
                q = pickle.load(fh)
            refs = por_torneo_ref.get(q["torneo"], [])
            resultados = []
            for r in refs:
                res = empareja(q, r, min_matches=min_matches, cv2=cv2)
                if res is not None:
                    res = dict(res, ref=r["clip"], conf_ref=r["conf"], n_kp_ref=r["n_kp"])
                    resultados.append(res)
            out.write(json.dumps({
                "clip": q["clip"], "torneo": q["torneo"], "motivo": q["motivo"],
                "x_prod": q["x_prod"], "zona_prod": q["zona_prod"],
                "zona_final_prod": q["zona_final_prod"], "n_refs": len(refs),
                "n_kp": q["n_kp"], "resultados": resultados,
            }) + "\n")
            n += 1
            if i % 50 == 0:
                print("%d/%d %.0f s" % (i, len(clips_q), time.time() - t0), flush=True)
    print("hecho: %d filas en %.0f s -> %s" % (n, time.time() - t0, salida_path), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    if cmd == "cache":
        cmd_cache(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
    elif cmd == "empareja":
        cmd_empareja(sys.argv[2], sys.argv[3])
    elif cmd == "cache_query":
        cmd_cache_query(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
    elif cmd == "empareja_query":
        cmd_empareja_query(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)

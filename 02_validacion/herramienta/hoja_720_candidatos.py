"""HOJA 2026-10-09: localizar la hoja de floretete (y la punta) en video de retransmision 1280x720.

FICHERO NUEVO, de un agente de investigacion. No toca nada mas del repo. Construye su propio
conjunto de evaluacion (~120 recortes, etiquetados a ojo por el agente, no por un arbitro) y mide
tres candidatos sobre el MISMO conjunto, con el criterio declarado en
`04_documentacion/informes/HOJA_2026-10-09.md` ANTES de correr nada.

Subcomandos (cada uno solo necesita el anterior):

    python hoja_720_candidatos.py seleccion     -> elige ~20 clips de florete a 1280x720, varios
                                                    torneos, con cuadro_tocado valido
    python hoja_720_candidatos.py recortes      -> extrae ~120 recortes (tocado y 0.5-1.5s antes,
                                                    por tirador), ancla en la mano armada (cono_lib)
    python hoja_720_candidatos.py contactos     -> hojas de contacto PNG para que el agente (y el
                                                    usuario) las mire
    python hoja_720_candidatos.py clasico       -> candidato (c): LSD/Hough anclado, clasico
    python hoja_720_candidatos.py owlv2         -> candidato (a): OWLv2 texto-abierto + SAHI
    python hoja_720_candidatos.py sam2          -> candidato (b): SAM2 con prompt de punto anclado
    python hoja_720_candidatos.py metricas      -> metricas declaradas, usando etiquetas.csv (que
                                                    el agente rellena a mano mirando las hojas)

Todas las salidas van a `02_validacion/auditoria/hoja_2026-10-09/`. Los temporales grandes (frames
de video) van a `D:/FENCINGLY_SCRATCH/agente_hoja/`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import random
import sys
import time

import numpy as np

CODIGO = pathlib.Path(__file__).resolve().parents[2] / "01_extraccion" / "codigo"
sys.path.insert(0, str(CODIGO))
import catalogo as CA  # noqa: E402
import cono_lib as CL  # noqa: E402
import rutas as RU  # noqa: E402

AUD = pathlib.Path(__file__).resolve().parents[2] / "02_validacion" / "auditoria" / "hoja_2026-10-09"
AUD.mkdir(parents=True, exist_ok=True)
(AUD / "crops").mkdir(exist_ok=True)
(AUD / "contactos").mkdir(exist_ok=True)
SCRATCH = pathlib.Path("D:/FENCINGLY_SCRATCH/agente_hoja")
SCRATCH.mkdir(parents=True, exist_ok=True)

N_CLIPS = 20
MAX_POR_TORNEO = 3
OFFSETS_S = (0.0, -0.75, -1.25)
"""El cuadro del tocado y 0,5-1,5 s antes (aqui 0,75 y 1,25 s, dentro del rango pedido)."""
SEMILLA = 42


def seleccion():
    df = CA.manifiesto()
    foil = df[df["weapon"] == "foil"]
    trabajo = RU.TRABAJO
    have_poses = set(p.name[6:-4] for p in trabajo.glob("poses_*.npz"))
    have_lights = set(p.name[7:-4] for p in trabajo.glob("lights_*.csv"))
    have_zona = set(p.name[5:-4] for p in trabajo.glob("zona_*.csv"))
    cand = foil[foil["clip_id"].isin(have_poses & have_lights & have_zona)]
    clips = cand["clip_id"].tolist()
    random.seed(SEMILLA)
    random.shuffle(clips)

    filas = []
    for cid in clips:
        npzp = trabajo / ("poses_%s.npz" % cid)
        try:
            with np.load(npzp, allow_pickle=True) as z:
                W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
        except Exception:
            continue
        if W != 1280 or H != 720:
            continue
        zonap = trabajo / ("zona_%s.csv" % cid)
        try:
            with open(zonap, encoding="utf-8") as fh:
                r = next(csv.DictReader(fh))
        except Exception:
            continue
        if r.get("motivo") != "ok" or not r.get("cuadro_tocado"):
            continue
        try:
            ct = int(float(r["cuadro_tocado"]))
        except Exception:
            continue
        if ct <= 0 or ct >= T:
            continue
        vid = RU.video(cid)
        if not vid or not pathlib.Path(vid).is_file():
            continue
        torneo = cand.loc[cand["clip_id"] == cid, "torneo"].iloc[0]
        filas.append({"clip_id": cid, "torneo": torneo, "W": W, "H": H, "T": T,
                      "cuadro_tocado": ct, "video": vid})

    elegidos, por_torneo = [], {}
    for f in filas:
        if len(elegidos) >= N_CLIPS:
            break
        n = por_torneo.get(f["torneo"], 0)
        if n >= MAX_POR_TORNEO:
            continue
        por_torneo[f["torneo"]] = n + 1
        elegidos.append(f)
    # si no llega a N_CLIPS por el tope por torneo, rellena sin tope
    if len(elegidos) < N_CLIPS:
        ids_ya = {f["clip_id"] for f in elegidos}
        for f in filas:
            if len(elegidos) >= N_CLIPS:
                break
            if f["clip_id"] in ids_ya:
                continue
            elegidos.append(f)

    out = AUD / "seleccion_clips.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip_id", "torneo", "W", "H", "T", "cuadro_tocado",
                                           "video"])
        w.writeheader()
        for f in elegidos:
            w.writerow(f)
    print("candidatos totales (foil, 1280x720, tocado ok, video en disco): %d" % len(filas))
    print("elegidos: %d, de %d torneos distintos" % (
        len(elegidos), len(set(f["torneo"] for f in elegidos))))
    print("guardado %s" % out)


def _carga_npz(cid):
    with np.load(RU.TRABAJO / ("poses_%s.npz" % cid), allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _frame(cap, k):
    cap.set(1, k)  # cv2.CAP_PROP_POS_FRAMES == 1
    ok, im = cap.read()
    return im if ok else None


def _recorte_lado(im, B, lado, rival, vert, u, L, W, H):
    """Caja propia UNIDA al segmento vert->vert+L*u, con margen. No corta en el rival: en un
    tocado la hoja puede llegar hasta encima del cuerpo rival, y es justo lo que interesa ver."""
    x0, y0, x1, y1 = B[lado, 0] * W, B[lado, 1] * H, B[lado, 2] * W, B[lado, 3] * H
    mx, my = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
    punta = vert + L * u
    cx0 = min(x0 - mx, vert[0], punta[0]) - 0.08 * L
    cx1 = max(x1 + mx, vert[0], punta[0]) + 0.08 * L
    cy0 = min(y0 - my, punta[1]) - 0.08 * L
    cy1 = max(y1 + my, punta[1]) + 0.08 * L
    cx0, cy0 = int(max(0, cx0)), int(max(0, cy0))
    cx1, cy1 = int(min(W, cx1)), int(min(H, cy1))
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return im[cy0:cy1, cx0:cx1].copy(), (cx0, cy0, cx1, cy1)


def recortes():
    import cv2
    sel = list(csv.DictReader(open(AUD / "seleccion_clips.csv", encoding="utf-8")))
    meta = []
    idx = 0
    for f in sel:
        cid = f["clip_id"]
        z = _carga_npz(cid)
        KP, SC, B, pres, mask, origen = z["KP"], z["SC"], z["B"], z["pres"], z["mask"], z["origen"]
        W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
        fps = float(z["fps"]) if np.isfinite(z["fps"]) and z["fps"] > 0 else 25.0
        ct = int(f["cuadro_tocado"])

        manos, pxm = {}, {}
        ok_mano = True
        info_mano = {}
        for lado in (0, 1):
            m, d = CL.mano_armada(KP, SC, B, W, H, lado)
            if m is None:
                ok_mano = False
                break
            manos[lado] = m
            _p, pxm[lado] = CL.escala_px_m(B, lado, H)
            info_mano[lado] = {"mano": m, "por": d.get("por"), "dudosa": d.get("dudosa")}
        if not ok_mano:
            print("  %s: sin mano armada decidible en algun lado, descartado" % cid)
            continue

        ejes = {}
        for lado in (0, 1):
            vert, u, okk, _n = CL.eje_antebrazo(KP, SC, lado, manos[lado])
            ejes[lado] = (vert, u, okk)

        cap = cv2.VideoCapture(f["video"])
        for dt in OFFSETS_S:
            k = int(round(ct + dt * fps))
            k = max(0, min(T - 1, k))
            # si el cuadro elegido no tiene pose buena en algun lado, busca el mas cercano (+-4)
            cand_ks = [k] + [k + s for step in range(1, 5) for s in (step, -step)]
            kk = None
            for kc in cand_ks:
                if kc < 0 or kc >= T:
                    continue
                if not mask[kc]:
                    continue
                if not (pres[kc, 0] and pres[kc, 1]):
                    continue
                if not (ejes[0][2][kc] and ejes[1][2][kc]):
                    continue
                kk = kc
                break
            if kk is None:
                continue
            im = _frame(cap, kk)
            if im is None:
                continue
            for lado in (0, 1):
                vert, u, okk = ejes[lado]
                v, uu = vert[kk], u[lado if False else kk]  # vert/u ya indexados por cuadro
                L = CL.HOJA_M * pxm[lado]
                rec = _recorte_lado(im, B[kk], lado, 1 - lado, v, uu, L, W, H)
                if rec is None:
                    continue
                sub, (cx0, cy0, cx1, cy1) = rec
                idx += 1
                cid_crop = "r%03d" % idx
                import cv2 as _cv2
                _cv2.imwrite(str(AUD / "crops" / (cid_crop + ".png")), sub)
                meta.append({
                    "id": cid_crop, "clip_id": cid, "torneo": f["torneo"], "cuadro": kk,
                    "cuadro_tocado": ct, "dt_s": dt, "lado": lado,
                    "mano_armada": manos[lado], "mano_por": info_mano[lado]["por"],
                    "mano_dudosa": info_mano[lado]["dudosa"],
                    "root_x": float(v[0]), "root_y": float(v[1]),
                    "u_x": float(uu[0]), "u_y": float(uu[1]), "L_px": float(L),
                    "crop_x0": cx0, "crop_y0": cy0, "crop_x1": cx1, "crop_y1": cy1,
                    "root_x_crop": float(v[0] - cx0), "root_y_crop": float(v[1] - cy0),
                    "pxm": float(pxm[lado]),
                })
        cap.release()
        print("  %s: ok" % cid)

    out = AUD / "recortes_meta.csv"
    campos = list(meta[0].keys()) if meta else []
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        for m in meta:
            w.writerow(m)
    print("total recortes: %d" % len(meta))
    print("guardado %s y %s" % (out, AUD / "crops"))

    # plantilla de etiquetas, para que el agente (y el usuario) la rellene
    etq = AUD / "etiquetas.csv"
    if not etq.is_file():
        with open(etq, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "visible", "punta_x_crop", "punta_y_crop",
                                               "guardia_x_crop", "guardia_y_crop", "nota"])
            w.writeheader()
            for m in meta:
                w.writerow({"id": m["id"], "visible": "", "punta_x_crop": "",
                           "punta_y_crop": "", "guardia_x_crop": "", "guardia_y_crop": "",
                           "nota": ""})
        print("plantilla de etiquetas creada en %s (sin rellenar)" % etq)


def anota():
    """Version ANOTADA de cada recorte, SOLO para que el agente etiquete a ojo con precision de
    pixel: cruz amarilla en el root de la mano armada, linea del eje del antebrazo hasta el alcance
    L (90 cm proyectados), y una rejilla cada 50 px con marcas en los bordes. Los candidatos NUNCA
    ven esta version: inferencia siempre sobre `crops/`, limpios."""
    import cv2
    out_dir = AUD / "crops_anot"
    out_dir.mkdir(exist_ok=True)
    meta = _meta()
    for m in meta:
        im = cv2.imread(str(AUD / "crops" / (m["id"] + ".png")))
        if im is None:
            continue
        im = im.copy()
        H, W = im.shape[:2]
        root = np.array([float(m["root_x_crop"]), float(m["root_y_crop"])])
        u = np.array([float(m["u_x"]), float(m["u_y"])])
        L = float(m["L_px"])
        punta_eje = root + L * u
        for x in range(0, W, 50):
            cv2.line(im, (x, 0), (x, H), (80, 80, 80), 1)
            cv2.putText(im, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 220, 220), 1,
                       cv2.LINE_AA)
        for y in range(0, H, 50):
            cv2.line(im, (0, y), (W, y), (80, 80, 80), 1)
            cv2.putText(im, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 220, 220), 1,
                       cv2.LINE_AA)
        cv2.line(im, tuple(root.astype(int)), tuple(punta_eje.astype(int)), (255, 120, 0), 1,
                cv2.LINE_AA)
        cv2.drawMarker(im, tuple(root.astype(int)), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(im, "%s  lado=%s  dt=%ss" % (m["id"], m["lado"], m["dt_s"]), (4, H - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / (m["id"] + ".png")), im)
    print("guardado %d recortes anotados en %s" % (len(meta), out_dir))


def contactos():
    import cv2
    meta = list(csv.DictReader(open(AUD / "recortes_meta.csv", encoding="utf-8")))
    POR_HOJA = 24
    COLS = 6
    CELDA = 220
    for i0 in range(0, len(meta), POR_HOJA):
        bloque = meta[i0:i0 + POR_HOJA]
        filas_n = math.ceil(len(bloque) / COLS)
        hoja = np.full((filas_n * CELDA, COLS * CELDA, 3), 40, dtype="uint8")
        for j, m in enumerate(bloque):
            im = cv2.imread(str(AUD / "crops" / (m["id"] + ".png")))
            if im is None:
                continue
            h, w = im.shape[:2]
            esc = min((CELDA - 24) / max(h, 1), (CELDA - 24) / max(w, 1), 3.0)
            im2 = cv2.resize(im, (max(1, int(w * esc)), max(1, int(h * esc))))
            r, c = divmod(j, COLS)
            y0, x0 = r * CELDA + 20, c * CELDA + 4
            h2, w2 = im2.shape[:2]
            hoja[y0:y0 + h2, x0:x0 + w2] = im2
            # marca el anclaje (root de la mano armada), por referencia geometrica, no es
            # una prediccion de hoja
            rx = x0 + int(float(m["root_x_crop"]) * esc)
            ry = y0 + int(float(m["root_y_crop"]) * esc)
            cv2.drawMarker(hoja, (rx, ry), (0, 255, 255), cv2.MARKER_CROSS, 8, 1)
            cv2.putText(hoja, m["id"], (c * CELDA + 4, r * CELDA + 14),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        out = AUD / "contactos" / ("hoja_contacto_%02d.png" % (i0 // POR_HOJA))
        cv2.imwrite(str(out), hoja)
        print("guardado %s (%d recortes)" % (out, len(bloque)))


# ----------------------------------------------------------------------------------------------
# CANDIDATOS
# ----------------------------------------------------------------------------------------------

def _meta():
    return list(csv.DictReader(open(AUD / "recortes_meta.csv", encoding="utf-8")))


def clasico():
    """Candidato (c): LSD/Hough de segmentos en el recorte anclado, direccion del antebrazo,
    quedandose con el segmento mas largo cuya recta pasa cerca del root. SIN coherencia temporal
    todavia (eso se mide aparte en el paso 4 solo si este candidato pasa el umbral)."""
    import cv2
    meta = _meta()
    out_rows = []
    t0 = time.perf_counter()
    try:
        lsd = cv2.createLineSegmentDetector(0)
    except Exception:
        lsd = None
    for m in meta:
        im = cv2.imread(str(AUD / "crops" / (m["id"] + ".png")))
        if im is None:
            continue
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        root = np.array([float(m["root_x_crop"]), float(m["root_y_crop"])])
        u = np.array([float(m["u_x"]), float(m["u_y"])])
        L = float(m["L_px"])
        segs = []
        if lsd is not None:
            ls, _w, _p, _n = lsd.detect(g)[:4] if False else (lsd.detect(g)[0], None, None, None)
            if ls is not None:
                segs = ls.reshape(-1, 4)
        if len(segs) == 0:
            edges = cv2.Canny(g, 40, 120)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 15, minLineLength=10, maxLineGap=4)
            if lines is not None:
                segs = lines.reshape(-1, 4)
        mejor = None
        for x1, y1, x2, y2 in segs:
            p1 = np.array([x1, y1], float)
            p2 = np.array([x2, y2], float)
            v = p2 - p1
            lg = float(np.linalg.norm(v))
            if lg < 6:
                continue
            nv = np.array([-v[1], v[0]]) / lg
            dist = abs(float(nv @ (root - p1)))
            if dist > 0.18 * max(L, 1.0):
                continue
            ang_seg = math.degrees(math.atan2(v[1], v[0]))
            ang_eje = math.degrees(math.atan2(u[1], u[0]))
            dif = abs((ang_seg - ang_eje + 90) % 180 - 90)
            # candidato a punta: el extremo mas lejano del root EN la direccion de u (adelante)
            d1 = float((p1 - root) @ u)
            d2 = float((p2 - root) @ u)
            punta = p1 if d1 > d2 else p2
            dadelante = max(d1, d2)
            if dadelante < 0:
                continue
            score = lg - 0.5 * dist
            if mejor is None or score > mejor[0]:
                mejor = (score, lg, dif, punta, dadelante)
        if mejor is None:
            out_rows.append({"id": m["id"], "detecta": 0, "punta_x_crop": "", "punta_y_crop": "",
                             "dif_ang_eje": "", "largo_px": ""})
        else:
            _s, lg, dif, punta, dadelante = mejor
            out_rows.append({"id": m["id"], "detecta": 1, "punta_x_crop": punta[0],
                             "punta_y_crop": punta[1], "dif_ang_eje": dif, "largo_px": lg})
    dt = time.perf_counter() - t0
    out = AUD / "pred_clasico.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "detecta", "punta_x_crop", "punta_y_crop",
                                           "dif_ang_eje", "largo_px"])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print("clasico: %d recortes, %.1f ms/recorte, guardado %s" % (
        len(out_rows), 1000 * dt / max(len(out_rows), 1), out))


def _hf_setup():
    import os
    os.environ.setdefault("HF_HOME", str(RU.HF))
    libs = SCRATCH / "libs"
    if str(libs) not in sys.path:
        sys.path.insert(0, str(libs))


PROMPTS = ["fencing sword", "foil blade", "thin metal rod", "fencing foil"]


def owlv2():
    """Candidato (a): OWLv2 (Apache-2.0, sin restriccion), texto abierto, con el recorte entero
    y una version SAHI (2x2 trozos solapados) para la hoja fina. Prompt fijo, declarado arriba."""
    _hf_setup()
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    import cv2

    nombre = "google/owlv2-base-patch16-ensemble"
    proc = Owlv2Processor.from_pretrained(nombre, cache_dir=str(RU.HF / "hub"))
    model = Owlv2ForObjectDetection.from_pretrained(
        nombre, cache_dir=str(RU.HF / "hub"), dtype=torch.float16).to("cuda").eval()

    meta = _meta()
    out_rows = []
    t0 = time.perf_counter()
    for m in meta:
        im = cv2.imread(str(AUD / "crops" / (m["id"] + ".png")))
        if im is None:
            continue
        H, W = im.shape[:2]
        root = np.array([float(m["root_x_crop"]), float(m["root_y_crop"])])
        tiles = [(0, 0, W, H)]
        if W > 120 and H > 120:
            hw, hh = W // 2 + 20, H // 2 + 20
            tiles += [(0, 0, hw, hh), (W - hw, 0, W, hh), (0, H - hh, hw, H),
                     (W - hw, H - hh, W, H)]
        mejor = None
        for x0, y0, x1, y1 in tiles:
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            sub = im[y0:y1, x0:x1]
            if sub.size == 0:
                continue
            rgb = np.ascontiguousarray(sub[:, :, ::-1])
            with torch.inference_mode():
                inputs = proc(text=[PROMPTS], images=rgb, return_tensors="pt")
                inputs = {k: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda"))
                         for k, v in inputs.items()}
                outputs = model(**inputs)
                tgt = torch.tensor([sub.shape[:2]])
                res = proc.post_process_grounded_object_detection(
                    outputs, target_sizes=tgt, threshold=0.05)[0]
            for score, box in zip(res["scores"].tolist(), res["boxes"].tolist()):
                bx0, by0, bx1, by1 = box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0
                cx, cy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
                dist_root = math.hypot(cx - root[0], cy - root[1])
                if mejor is None or score > mejor[0]:
                    mejor = (score, bx0, by0, bx1, by1, dist_root)
        if mejor is None:
            out_rows.append({"id": m["id"], "detecta": 0, "score": "", "punta_x_crop": "",
                             "punta_y_crop": ""})
        else:
            score, bx0, by0, bx1, by1, _d = mejor
            u = np.array([float(m["u_x"]), float(m["u_y"])])
            esquinas = [(bx0, by0), (bx0, by1), (bx1, by0), (bx1, by1)]
            punta = max(esquinas, key=lambda p: (np.array(p) - root) @ u)
            out_rows.append({"id": m["id"], "detecta": 1, "score": score,
                             "punta_x_crop": punta[0], "punta_y_crop": punta[1]})
    dt = time.perf_counter() - t0
    out = AUD / "pred_owlv2.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "detecta", "score", "punta_x_crop",
                                           "punta_y_crop"])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print("owlv2: %d recortes, %.0f ms/recorte (GPU), guardado %s" % (
        len(out_rows), 1000 * dt / max(len(out_rows), 1), out))


def sam2():
    """Candidato (b): SAM2 (Apache-2.0) con prompt de PUNTO anclado a medio camino de la direccion
    del antebrazo (root + 0.5*L*u), un punto positivo mas, y un punto negativo en el cuerpo del
    tirador (dentro de su caja, lejos del eje). Punta = punto de la mascara mas lejano del root en
    direccion u."""
    _hf_setup()
    import torch
    from transformers import Sam2Model, Sam2Processor
    import cv2

    nombre = "facebook/sam2.1-hiera-tiny"
    proc = Sam2Processor.from_pretrained(nombre, cache_dir=str(RU.HF / "hub"))
    model = Sam2Model.from_pretrained(
        nombre, cache_dir=str(RU.HF / "hub"), dtype=torch.float16).to("cuda").eval()

    meta = _meta()
    out_rows = []
    t0 = time.perf_counter()
    for m in meta:
        im = cv2.imread(str(AUD / "crops" / (m["id"] + ".png")))
        if im is None:
            continue
        H, W = im.shape[:2]
        root = np.array([float(m["root_x_crop"]), float(m["root_y_crop"])])
        u = np.array([float(m["u_x"]), float(m["u_y"])])
        L = float(m["L_px"])
        p1 = root + 0.35 * L * u
        p2 = root + 0.65 * L * u
        pts = [[float(np.clip(p1[0], 0, W - 1)), float(np.clip(p1[1], 0, H - 1))],
               [float(np.clip(p2[0], 0, W - 1)), float(np.clip(p2[1], 0, H - 1))]]
        labels = [1, 1]
        rgb = np.ascontiguousarray(im[:, :, ::-1])
        try:
            with torch.inference_mode():
                inputs = proc(images=rgb, input_points=[[pts]], input_labels=[[labels]],
                             return_tensors="pt")
                inputs = {k: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda"))
                         for k, v in inputs.items()}
                outputs = model(**inputs)
                masks_list = proc.post_process_masks(outputs.pred_masks.cpu(), [[H, W]])
                scores = outputs.iou_scores.cpu().numpy().reshape(-1)
        except Exception as e:
            out_rows.append({"id": m["id"], "detecta": 0, "score": "", "punta_x_crop": "",
                             "punta_y_crop": "", "error": str(e)[:80]})
            continue
        best = int(np.argmax(scores))
        # masks_list[0] shape (num_point_sets=1, num_masks, H, W)
        mask = np.asarray(masks_list[0][0, best]).astype(bool)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            out_rows.append({"id": m["id"], "detecta": 0, "score": float(scores[best]),
                             "punta_x_crop": "", "punta_y_crop": "", "error": ""})
            continue
        proy = (xs - root[0]) * u[0] + (ys - root[1]) * u[1]
        j = int(np.argmax(proy))
        out_rows.append({"id": m["id"], "detecta": 1, "score": float(scores[best]),
                         "punta_x_crop": float(xs[j]), "punta_y_crop": float(ys[j]), "error": ""})
    dt = time.perf_counter() - t0
    out = AUD / "pred_sam2.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "detecta", "score", "punta_x_crop",
                                           "punta_y_crop", "error"])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print("sam2: %d recortes, %.0f ms/recorte (GPU), guardado %s" % (
        len(out_rows), 1000 * dt / max(len(out_rows), 1), out))


def metricas():
    """Metrica DECLARADA en el informe antes de evaluar: sobre las hojas VISIBLES (si/parcial),
    fraccion con error de punta <= 15 px y <= 30 px; error angular guardia-punta contra el eje del
    antebrazo (que es el mejor sustituto que hay cuando no hay guardia anotada aparte); en las NO
    visibles, tasa de falsos positivos (detecta == 1)."""
    etq = {r["id"]: r for r in csv.DictReader(open(AUD / "etiquetas.csv", encoding="utf-8"))}
    meta = {r["id"]: r for r in _meta()}
    resumen = {}
    for nombre in ("clasico", "owlv2", "sam2"):
        p = AUD / ("pred_%s.csv" % nombre)
        if not p.is_file():
            continue
        pred = {r["id"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}
        n_vis = n_15 = n_30 = n_novis = n_fp = 0
        angs = []
        for cid, e in etq.items():
            vis = e.get("visible", "").strip().lower()
            if cid not in pred:
                continue
            pr = pred[cid]
            detecta = pr.get("detecta") == "1"
            if vis in ("si", "parcial"):
                n_vis += 1
                if detecta and e.get("punta_x_crop") and pr.get("punta_x_crop"):
                    dx = float(pr["punta_x_crop"]) - float(e["punta_x_crop"])
                    dy = float(pr["punta_y_crop"]) - float(e["punta_y_crop"])
                    err = math.hypot(dx, dy)
                    if err <= 15:
                        n_15 += 1
                    if err <= 30:
                        n_30 += 1
                    mm = meta.get(cid, {})
                    if mm and e.get("guardia_x_crop"):
                        gx, gy = float(e["guardia_x_crop"]), float(e["guardia_y_crop"])
                        ang_et = math.degrees(math.atan2(float(e["punta_y_crop"]) - gy,
                                                         float(e["punta_x_crop"]) - gx))
                        ang_pr = math.degrees(math.atan2(float(pr["punta_y_crop"]) - gy,
                                                         float(pr["punta_x_crop"]) - gx))
                        dif = abs((ang_et - ang_pr + 180) % 360 - 180)
                        angs.append(dif)
            elif vis == "no":
                n_novis += 1
                if detecta:
                    n_fp += 1
        resumen[nombre] = {
            "n_visibles": n_vis, "punta_<=15px": "%d/%d" % (n_15, n_vis) if n_vis else None,
            "frac_<=15px": (n_15 / n_vis) if n_vis else None,
            "punta_<=30px": "%d/%d" % (n_30, n_vis) if n_vis else None,
            "frac_<=30px": (n_30 / n_vis) if n_vis else None,
            "ang_mediana_gr": float(np.median(angs)) if angs else None,
            "ang_n": len(angs),
            "n_no_visibles": n_novis, "falsos_positivos": "%d/%d" % (n_fp, n_novis) if n_novis else None,
            "frac_fp": (n_fp / n_novis) if n_novis else None,
        }
    out = AUD / "metricas.json"
    out.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print("guardado %s" % out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paso", choices=["seleccion", "recortes", "contactos", "anota", "clasico",
                                     "owlv2", "sam2", "metricas"])
    a = ap.parse_args()
    {"seleccion": seleccion, "recortes": recortes, "contactos": contactos, "anota": anota,
     "clasico": clasico, "owlv2": owlv2, "sam2": sam2, "metricas": metricas}[a.paso]()

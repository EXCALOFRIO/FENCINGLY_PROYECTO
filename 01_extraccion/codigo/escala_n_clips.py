"""Escala la tuberia a N clips y mide la calidad agregada. Sin video: eso es para verificar a ojo.

LA IDEA QUE HACE ESTO VIABLE: la puerta de `matting` es una decision **por trayectoria**, no por
cuadro. Un tirador no deja de ser persona a mitad de clip. Asi que basta muestrear unos pocos
cuadros de cada trayectoria en vez de los 300, y el coste baja de ~800 ms x 300 cuadros a
~800 ms x 5. Sin eso, escalar a 100 clips seria dias.

QUE SE MIDE POR CLIP, y todo sale de algo ya medido:
  presencia de cada elegido y de los dos a la vez  (techo fisico medido: 88 % de los cuadros tienen
                                                    a los dos tiradores en el plano)
  cuadros interpolados, con el limite de 5 cuadros medido escondiendo fotogramas
  alfa de `matting` del peor elegido               (tirador 0,21 / fondo vacio 0,0053)
  aviso si las dos mejores trayectorias caen del MISMO lado, que es sintoma de mala eleccion

CADENCIA DEL PLAN: 1 -> 2 -> 5 -> 10 -> 100, y no se pasa al siguiente sin la cifra del anterior.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

CACHE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565\scratchpad\hoja_seg\hf")
os.environ["HF_HOME"] = str(CACHE)

import cv2
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import a3_pareja as PAR
import a3_selector as SEL
import interpola as IP

BASE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565")
PESOS_YOLO = BASE / "scratchpad/smoke/weights/yolo11m-pose.pt"
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
HUB = CACHE / "hub"
N_CLIPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
FASE = sys.argv[2] if len(sys.argv) > 2 else "todo"
"""`yolo` = detectar y elegir (necesita ultralytics). `puerta` = matting (necesita transformers)."""
MUESTRAS_MATTING = 4
MARGEN_PIE = 0.06
LIMITE_HUECO = 5
ALFA_MIN = 0.05


def snap(n):
    d = HUB / ("models--facebook--%s" % n) / "snapshots"
    return next(d.iterdir()) if d.is_dir() else None


mapa = {}
for raiz, _d, fs in os.walk("D:/FENCINGLY_VIDEOS_RAW"):
    for f in fs:
        if f.lower().endswith(".mp4"):
            mapa.setdefault(pathlib.Path(f).stem, os.path.join(raiz, f))
rng = np.random.default_rng(20260910)
clips = sorted(mapa)
rng.shuffle(clips)
clips = clips[:N_CLIPS]
print("clips a procesar: %d de %d disponibles" % (len(clips), len(mapa)))

yolo = pm = mm = None
if FASE in ("todo", "yolo"):
    from ultralytics import YOLO
    yolo = YOLO(str(PESOS_YOLO))
    yolo.to("cuda")
    print("YOLO cargado")
if FASE in ("todo", "puerta"):
    from transformers import Sapiens2ForImageMatting, Sapiens2ImageProcessor
    s = snap("sapiens2-matting-1b")
    pm = Sapiens2ImageProcessor.from_pretrained(str(s))
    pm.size = {"height": 384, "width": 288}
    """`matting` a 384x288 conserva 23,5x de separacion sin solape por 125 ms, medido. A resolucion
    nativa da 44,6x pero cuesta 767 ms, y para una puerta por trayectoria 23,5x sobra."""
    mm = Sapiens2ForImageMatting.from_pretrained(str(s), dtype=torch.float16).to("cuda").eval()
    print("matting cargado en %s" % next(mm.parameters()).device)
INTER = SALIDA / ("escala_inter_%d.npz" % N_CLIPS)
print()

filas, guardado = [], []
for n, cid in enumerate(clips, 1):
    t_clip = time.perf_counter()
    cap = cv2.VideoCapture(mapa[cid])
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if W < 2:
        cap.release()
        continue
    frames, cajas, cuadros = [], [], []
    k = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
        r = yolo.predict(im, imgsz=960, conf=0.10, verbose=False, device=0)[0]
        if r.boxes is not None and len(r.boxes):
            b = r.boxes.xyxy.cpu().numpy()
            for i in range(len(b)):
                cajas.append(b[i])
                cuadros.append(k)
        k += 1
    cap.release()
    if k < 10 or not cajas:
        continue
    cajas = np.asarray(cajas, "float32")
    cuadros = np.asarray(cuadros, "int32")

    tr = SEL.construye_trayectorias(cajas, cuadros, W, H, k)
    res = PAR.elige_pareja(tr, W, H, k)
    cand = res["candidatos"]
    pasan = res["elegidas"]
    B, pres = SEL.a_matrices(res["elegidas"], cajas, k, W, H)
    antes = float((pres.sum(axis=1) == 2).mean())
    nmarca = 0
    for lado in (0, 1):
        for h in IP.huecos(pres[:, lado]):
            if h["tipo"] != "interior" or h["largo"] > LIMITE_HUECO:
                continue
            a, b = h["a"], h["b"]
            t = (np.arange(1, b - a) / float(b - a))[:, None]
            B[h["ini"]:h["fin"], lado] = (1 - t) * B[a, lado][None] + t * B[b, lado][None]
            pres[h["ini"]:h["fin"], lado] = True
            nmarca += h["largo"]
    ambos = float((pres.sum(axis=1) == 2).mean())

    """Puerta de matting muestreada: unos pocos cuadros por elegido, no los 300."""
    alfas = []
    if mm is None:
        guardado.append({"clip": cid, "B": B, "pres": pres, "W": W, "H": H, "T": k,
                         "video": mapa[cid], "ambos": ambos, "antes": antes, "nmarca": nmarca,
                         "ntray": len(cand), "npasan": len(pasan), "motivo": res["motivo"]})
        print("  %2d/%d %-38s T=%4d  tray %3d->%2d  ambos %5.1f%%  interp %3d  (matting en fase 2)" % (
            n, len(clips), cid[:38], k, len(cand), len(pasan), 100 * ambos, nmarca))
        continue
    for lado in (0, 1):
        idx = np.nonzero(pres[:, lado])[0]
        if not len(idx):
            continue
        for kk in idx[np.linspace(0, len(idx) - 1, min(MUESTRAS_MATTING, len(idx))).astype(int)]:
            b = B[kk, lado]
            x0, y0 = int(max(0, b[0] * W)), int(max(0, b[1] * H))
            x1, y1 = int(min(W, b[2] * W)), int(min(H, b[3] * H))
            mx, my = int(0.22 * (x1 - x0)), int(0.22 * (y1 - y0))
            cx0, cy0 = max(0, x0 - mx), max(0, y0 - my)
            cx1, cy1 = min(W, x1 + mx), min(H, y1 + my)
            if cx1 - cx0 < 24 or cy1 - cy0 < 24:
                continue
            sub = np.ascontiguousarray(frames[kk][cy0:cy1, cx0:cx1, ::-1])
            with torch.inference_mode():
                e = pm(images=sub, return_tensors="pt")
                e = {q: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda"))
                     for q, v in e.items()}
                o = mm(**e)
                mmask = pm.post_process_image_matting(o, target_sizes=[sub.shape[:2]])[0]
            a = np.squeeze(np.nan_to_num(np.asarray(
                (mmask["alpha"] if isinstance(mmask, dict) else mmask).cpu(), "float32")))
            alfas.append((lado, float(a.mean())))
    peor = min((a for _l, a in alfas), default=float("nan"))
    fila = {"clip": cid, "T": k, "W": W, "H": H, "trayectorias": len(cand),
            "pasan_puerta": len(pasan), "motivo": res["motivo"],
            "pres0": float(pres[:, 0].mean()), "pres1": float(pres[:, 1].mean()),
            "ambos_antes": antes, "ambos": ambos, "interpolados": nmarca,
            "alfa_peor": peor, "alfa_med": float(np.mean([a for _l, a in alfas])) if alfas else float("nan"),
            "seg_clip": time.perf_counter() - t_clip}
    filas.append(fila)
    print("  %2d/%d %-38s T=%4d  tray %3d->%2d  pres %5.1f/%5.1f  ambos %5.1f%%  interp %3d  alfa peor %.3f %s" % (
        n, len(clips), cid[:38], k, len(cand), len(pasan), 100 * fila["pres0"], 100 * fila["pres1"],
        100 * ambos, nmarca, peor,
        "  <- MISMO LADO" if "MISMO" in res["motivo"] else ("  <- ALFA BAJA" if peor < ALFA_MIN else "")))

if guardado:
    np.savez_compressed(INTER, datos=np.array(guardado, dtype=object), allow_pickle=True)
    print()
    print("FASE 1 hecha: %d clips en %s" % (len(guardado), INTER.name))
    print("Ahora corre la fase 2 con el interprete de cuda:")
    print("  cuda/.venv/python escala_n_clips.py %d puerta" % N_CLIPS)
    raise SystemExit

import pandas as pd
d = pd.DataFrame(filas)
d.to_csv(SALIDA / ("escala_%dclips.csv" % len(d)), index=False)
print()
print("=" * 96)
print("AGREGADO sobre %d clips y %d cuadros" % (len(d), int(d["T"].sum())))
print("=" * 96)
if len(d):
    print("  los dos presentes: mediana %5.1f %%   p10 %5.1f %%   min %5.1f %%" % (
        100 * d["ambos"].median(), 100 * d["ambos"].quantile(0.1), 100 * d["ambos"].min()))
    print("  ganancia de la interpolacion: %+.2f pp de mediana (%d cuadros en total)" % (
        100 * (d["ambos"] - d["ambos_antes"]).median(), int(d["interpolados"].sum())))
    print("  clips con los dos en >= 90 %% ..... %d de %d  (%.0f %%)" % (
        int((d["ambos"] >= 0.90).sum()), len(d), 100.0 * (d["ambos"] >= 0.90).mean()))
    print("  clips con los dos en <  50 %% ..... %d" % int((d["ambos"] < 0.50).sum()))
    print()
    print("  PUERTA DE MATTING: alfa del peor elegido")
    print("    mediana %.3f   p10 %.3f   min %.3f" % (
        d["alfa_peor"].median(), d["alfa_peor"].quantile(0.1), d["alfa_peor"].min()))
    print("    clips con algun elegido por debajo de %.2f ... %d de %d" % (
        ALFA_MIN, int((d["alfa_peor"] < ALFA_MIN).sum()), len(d)))
    print()
    print("  AVISOS DE ELECCION")
    print("    clips donde las dos mejores caen del MISMO lado ... %d de %d" % (
        int(d["motivo"].str.contains("MISMO").sum()), len(d)))
    print()
    print("  coste: %.1f s por clip de mediana (%.0f ms/cuadro, GPU compartida)" % (
        d["seg_clip"].median(), 1000 * d["seg_clip"].median() / max(d["T"].median(), 1)))
    print()
    print("  LOS QUE HAY QUE MIRAR A OJO, por orden de sospecha")
    mal = d.sort_values(["alfa_peor", "ambos"]).head(6)
    for _, r in mal.iterrows():
        print("    %-40s ambos %5.1f %%  alfa peor %.3f  %s" % (
            r["clip"][:40], 100 * r["ambos"], r["alfa_peor"], r["motivo"]))

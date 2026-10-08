"""ETAPA A: `seg` de Sapiens 2 como puerta de "hay un tirador en esta caja?", con control conocido.

POR QUE ESTE EXPERIMENTO ES DISTINTO DE LOS NUEVE ANTERIORES: mis nueve detectores eran de coherencia
geometrica, y un esqueleto dibujado sobre una pared es perfectamente coherente. Median que la serie
es estable, no que hubiera alguien. `seg` devuelve **clases de partes del cuerpo por pixel**, asi que
"hay alguien" pasa de inferencia a medida.

EL CONJUNTO DE CONTROL, que es lo que hace valida la prueba. No es un umbral inventado: son casos
donde ya SE MIRO la imagen y se sabe la respuesta.

  NEGATIVOS VERIFICADOS A OJO (no hay nadie o no son tiradores):
    SoreEasygoingEmperorshrimp          los dos esqueletos sobre paredes vacias, el tirador real sin marcar
    AltruisticIllfatedDobermanpinscher  dos oficiales de traje oscuro seleccionados como tiradores

  POSITIVOS VERIFICADOS A OJO (hay tirador y se le ve la cazoleta):
    SentimentalLargeBedlingtonterrier   720p
    AdorableFittingAsiansmallclawedotter 1080p

  Y ADEMAS, graduado: los 10 clips de blancura MAS BAJA y los 10 de blancura MAS ALTA de la medida
  de ayer sobre 400 clips que el algoritmo geometrico declaraba limpios. La blancura ya se validó por
  los dos lados (0,000 sobre pared, 0,72-0,96 sobre tirador), asi que sirve de referencia graduada
  aunque no sea verdad-terreno.

CRITERIO DECLARADO ANTES DE MIRAR:
  medida = fraccion de pixeles del recorte que `seg` asigna a CUALQUIER clase de persona (1..28),
           o sea todo menos el fondo (clase 0)
  la puerta SIRVE si la mediana de los positivos y la de los negativos estan separadas por al menos
  0,25 y no se solapan los rangos intercuartilicos. Si se solapan, `seg` no discrimina y hay que
  decirlo en vez de buscar un umbral que salve el resultado.

CLASES DE `seg` que importan aparte, porque el uniforme de florete es blanco y cubre todo:
  22 Torso, 23 Upper_Clothing, 1 Apparel, 7/16 Lower_Arm, 6/15 Hand
"""
from __future__ import annotations

import json
import os
import pathlib

CACHE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565\scratchpad\hoja_seg\hf")
os.environ["HF_HOME"] = str(CACHE)

import cv2
import numpy as np
import pandas as pd
import torch

BASE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565")
PESOS = CACHE / "hub/models--facebook--sapiens2-seg-0.4b/snapshots"
NPZ_DIRS = [BASE / "localrun/runs_extra1/data_20260907T005507157743Z/npz",
            BASE / "localrun/runs_extra2/data_20260907T005507370214Z/npz",
            BASE / "localrun/runs_extra3/data_20260907T010246490954Z/npz",
            BASE / "scratchpad/catalogo/ws_completo/npz"]
BLANCURA = pathlib.Path(r"C:\Users\Alejandro\.claude\jobs\187cfd83\tmp\blancura.csv")
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\etapa_a_seg")
SALIDA.mkdir(parents=True, exist_ok=True)

RES = (384, 288)
N_CUADROS = 4
TORSO, ROPA_ARRIBA, APAREJO = 22, 23, 1
BRAZOS = [7, 16, 11, 20]
MANOS = [6, 15]
SEPARACION_MINIMA = 0.25

NEGATIVOS = {"SoreEasygoingEmperorshrimp": "esqueletos sobre pared vacia",
             "AltruisticIllfatedDobermanpinscher": "dos oficiales de traje"}
POSITIVOS = {"SentimentalLargeBedlingtonterrier": "tirador visible 720p",
             "AdorableFittingAsiansmallclawedotter": "tirador visible 1080p"}

snap = next(PESOS.iterdir()) if PESOS.is_dir() else None
if snap is None:
    raise SystemExit("no encuentro los pesos de seg en %s" % PESOS)
print("pesos de seg: %s" % snap.name)

mapa = {}
for raiz, _d, fs in os.walk("D:/FENCINGLY_VIDEOS_RAW"):
    for f in fs:
        if f.lower().endswith(".mp4"):
            mapa.setdefault(pathlib.Path(f).stem, os.path.join(raiz, f))
npz_de = {}
for d in NPZ_DIRS:
    if d.is_dir():
        for f in d.glob("*.npz"):
            npz_de.setdefault(f.stem, f)

casos = []
for cid, motivo in NEGATIVOS.items():
    casos.append({"clip": cid, "clase": "NEGATIVO verificado", "motivo": motivo})
for cid, motivo in POSITIVOS.items():
    casos.append({"clip": cid, "clase": "POSITIVO verificado", "motivo": motivo})
if BLANCURA.is_file():
    b = pd.read_csv(BLANCURA)
    for _, r in b.nsmallest(10, "blancura_min").iterrows():
        if r["clip_id"] not in NEGATIVOS:
            casos.append({"clip": r["clip_id"], "clase": "blancura BAJA",
                          "motivo": "blancura %.3f" % r["blancura_min"]})
    for _, r in b.nlargest(10, "blancura_min").iterrows():
        if r["clip_id"] not in POSITIVOS:
            casos.append({"clip": r["clip_id"], "clase": "blancura ALTA",
                          "motivo": "blancura %.3f" % r["blancura_min"]})
casos = [c for c in casos if c["clip"] in mapa and c["clip"] in npz_de]
print("casos con material: %d" % len(casos))
for cl in ("NEGATIVO verificado", "POSITIVO verificado", "blancura BAJA", "blancura ALTA"):
    print("   %-22s %d" % (cl, sum(1 for c in casos if c["clase"] == cl)))

from transformers import Sapiens2ForSemanticSegmentation, Sapiens2ImageProcessor

proc = Sapiens2ImageProcessor.from_pretrained(str(snap))
proc.size = {"height": RES[0], "width": RES[1]}
mod = Sapiens2ForSemanticSegmentation.from_pretrained(str(snap), dtype=torch.float16).to("cuda").eval()
print("seg cargado en %s a %dx%d" % (next(mod.parameters()).device, RES[0], RES[1]))

filas = []
for c in casos:
    cid = c["clip"]
    with np.load(npz_de[cid], allow_pickle=True) as z:
        BOX = np.asarray(z["boxes"], dtype="float64")
    cap = cv2.VideoCapture(mapa[cid])
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ks = np.linspace(0, min(T, BOX.shape[0]) - 1, N_CUADROS).astype(int)
    for k in ks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, im = cap.read()
        if not ok:
            continue
        for s in (0, 1):
            bb = BOX[k, s]
            if not (np.isfinite(bb).all() and bb[2] > bb[0]):
                continue
            x1, y1 = int(max(0, bb[0] * W)), int(max(0, bb[1] * H))
            x2, y2 = int(min(W, bb[2] * W)), int(min(H, bb[3] * H))
            if x2 - x1 < 12 or y2 - y1 < 12:
                continue
            sub = np.ascontiguousarray(im[y1:y2, x1:x2, ::-1])
            with torch.inference_mode():
                e = proc(images=sub, return_tensors="pt")
                e = {kk: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda"))
                     for kk, v in e.items()}
                o = mod(**e)
                m = proc.post_process_semantic_segmentation(
                    o, target_sizes=[(sub.shape[0], sub.shape[1])])[0]
            m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
            tot = m.size
            persona = float((m > 0).sum()) / tot
            torso = float(np.isin(m, [TORSO, ROPA_ARRIBA, APAREJO]).sum()) / tot
            brazos = float(np.isin(m, BRAZOS).sum()) / tot
            manos = float(np.isin(m, MANOS).sum()) / tot
            filas.append({"clip": cid, "clase": c["clase"], "motivo": c["motivo"],
                          "cuadro": int(k), "lado": s, "persona": round(persona, 4),
                          "torso": round(torso, 4), "brazos": round(brazos, 4),
                          "manos": round(manos, 4), "clases": int(len(np.unique(m)))})
    cap.release()

d = pd.DataFrame(filas)
d.to_csv(SALIDA / "etapa_a_seg.csv", index=False)
print()
print("medidas: %d cajas" % len(d))

print()
print("=" * 96)
print("FRACCION DE PIXELES DE PERSONA, por clase de caso")
print("=" * 96)
print("%-22s %5s %9s %9s %9s %9s %9s" % ("clase", "n", "persona", "p25", "p75", "torso", "manos"))
for cl in ("NEGATIVO verificado", "blancura BAJA", "blancura ALTA", "POSITIVO verificado"):
    g = d[d.clase == cl]
    if not len(g):
        continue
    print("%-22s %5d %9.3f %9.3f %9.3f %9.3f %9.3f" % (
        cl, len(g), g.persona.median(), g.persona.quantile(0.25), g.persona.quantile(0.75),
        g.torso.median(), g.manos.median()))

print()
print("=" * 96)
print("VEREDICTO CONTRA EL CRITERIO DECLARADO")
print("=" * 96)
neg = d[d.clase.isin(["NEGATIVO verificado", "blancura BAJA"])]
pos = d[d.clase.isin(["POSITIVO verificado", "blancura ALTA"])]
if len(neg) and len(pos):
    mn, mp = float(neg.persona.median()), float(pos.persona.median())
    sep = mp - mn
    solapan = (float(pos.persona.quantile(0.25)) <= float(neg.persona.quantile(0.75)))
    print("  negativos: mediana %.3f   rango intercuartilico [%.3f, %.3f]  n=%d" % (
        mn, neg.persona.quantile(0.25), neg.persona.quantile(0.75), len(neg)))
    print("  positivos: mediana %.3f   rango intercuartilico [%.3f, %.3f]  n=%d" % (
        mp, pos.persona.quantile(0.25), pos.persona.quantile(0.75), len(pos)))
    print("  SEPARACION ... %.3f   (limite declarado %.2f)" % (sep, SEPARACION_MINIMA))
    print("  se solapan ... %s" % ("SI" if solapan else "no"))
    print()
    if sep >= SEPARACION_MINIMA and not solapan:
        print("  `seg` DISCRIMINA. Es la puerta que faltaba para la etapa A.")
        cortes = np.linspace(mn, mp, 9)[1:-1]
        print()
        print("  %8s %14s %14s" % ("umbral", "negativos que", "positivos que"))
        print("  %8s %14s %14s" % ("", "pasarian", "se perderian"))
        for u in cortes:
            fp = float((neg.persona >= u).mean())
            fn = float((pos.persona < u).mean())
            print("  %8.3f %13.1f %% %13.1f %%" % (u, 100 * fp, 100 * fn))
    else:
        print("  `seg` NO discrimina con este criterio. No se busca un umbral que salve el resultado:")
        print("  se dice que no separa y se piensa otra medida.")

print()
print("--- los dos negativos verificados a ojo, uno por uno")
for cid, motivo in NEGATIVOS.items():
    g = d[d.clip == cid]
    if len(g):
        print("  %-38s %-32s persona %.3f  torso %.3f" % (
            cid[:38], motivo, g.persona.median(), g.torso.median()))
print("--- los dos positivos verificados a ojo")
for cid, motivo in POSITIVOS.items():
    g = d[d.clip == cid]
    if len(g):
        print("  %-38s %-32s persona %.3f  torso %.3f" % (
            cid[:38], motivo, g.persona.median(), g.torso.median()))
print()
print("guardado %s" % (SALIDA / "etapa_a_seg.csv"))

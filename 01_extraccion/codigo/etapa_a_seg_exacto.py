"""ETAPA A con `seg`, esta vez sobre LOS CUADROS EXACTOS que se verificaron a ojo.

QUE ESTABA MAL EN LA VERSION ANTERIOR, y lo digo porque el error es del tipo que mas veces me ha
mordido: aplique una etiqueta POR CLIP a una medida POR FOTOGRAMA. Muestree cuatro cuadros repartidos
por el clip, no el cuadro concreto donde vi los esqueletos sobre la pared. En los demas cuadros las
cajas si tienen gente, asi que "clip negativo" mezclaba cajas buenas y malas y la separacion se
diluia. La prueba dijo "no discrimina" cuando lo que no discriminaba era mi etiqueta.

AQUI SE USA EL CUADRO EXACTO de cada caso verificado:
  SoreEasygoingEmperorshrimp          cuadro medio: los DOS esqueletos sobre pared vacia
  AltruisticIllfatedDobermanpinscher  cuadro 54: dos oficiales de traje oscuro
  SentimentalLargeBedlingtonterrier   cuadro medio: tirador visible, 720p
  AdorableFittingAsiansmallclawedotter cuadro medio: tirador visible, 1080p

Y SE PINTA LA MASCARA de cada caja, porque una tabla no permite ver si `seg` esta segmentando a una
persona de verdad o inventando partes de cuerpo sobre una pared. Las imagenes van a
`02_validacion/muestras/etapa_a_seg/`.

CRITERIO: fraccion de pixeles de persona (clases 1..28) y de TORSO+ROPA (22, 23, 1). Con n tan
pequeno no se declara un umbral: se mira si los casos conocidos se ordenan como deben, y se ensena.
"""
from __future__ import annotations

import os
import pathlib

CACHE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565\scratchpad\hoja_seg\hf")
os.environ["HF_HOME"] = str(CACHE)

import cv2
import numpy as np
import pandas as pd
import torch

BASE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565")
PESOS_DIR = CACHE / "hub/models--facebook--sapiens2-seg-0.4b/snapshots"
NPZ_DIRS = [BASE / "localrun/runs_extra1/data_20260907T005507157743Z/npz",
            BASE / "localrun/runs_extra2/data_20260907T005507370214Z/npz",
            BASE / "localrun/runs_extra3/data_20260907T010246490954Z/npz",
            BASE / "scratchpad/catalogo/ws_completo/npz"]
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\etapa_a_seg")
SALIDA.mkdir(parents=True, exist_ok=True)

RES = (384, 288)
PERSONA_MIN = 1
TORSO_ROPA = [22, 23, 1]
BRAZOS = [7, 16, 11, 20]
MANOS = [6, 15]
NOMBRES = {0: "Fondo", 1: "Apparel", 3: "Face_Neck", 4: "Hair", 5: "L_Foot", 6: "L_Hand",
           7: "L_LowArm", 8: "L_LowLeg", 9: "L_Shoe", 10: "L_Sock", 11: "L_UpArm",
           12: "L_UpLeg", 13: "LowClothing", 14: "R_Foot", 15: "R_Hand", 16: "R_LowArm",
           17: "R_LowLeg", 18: "R_Shoe", 19: "R_Sock", 20: "R_UpArm", 21: "R_UpLeg",
           22: "Torso", 23: "UpClothing"}

CASOS = [
    ("SoreEasygoingEmperorshrimp", None, "NEGATIVO", "los dos esqueletos sobre pared vacia"),
    ("AltruisticIllfatedDobermanpinscher", 54, "NEGATIVO", "dos oficiales de traje oscuro"),
    ("SentimentalLargeBedlingtonterrier", None, "POSITIVO", "tirador visible 720p"),
    ("AdorableFittingAsiansmallclawedotter", None, "POSITIVO", "tirador visible 1080p"),
]

mapa = {}
for raiz, _d, fs in os.walk("D:/FENCINGLY_VIDEOS_RAW"):
    for f in fs:
        if f.lower().endswith(".mp4"):
            mapa.setdefault(pathlib.Path(f).stem, os.path.join(raiz, f))
npz_de = {}
for d0 in NPZ_DIRS:
    if d0.is_dir():
        for f in d0.glob("*.npz"):
            npz_de.setdefault(f.stem, f)

snap = next(PESOS_DIR.iterdir())
from transformers import Sapiens2ForSemanticSegmentation, Sapiens2ImageProcessor

proc = Sapiens2ImageProcessor.from_pretrained(str(snap))
proc.size = {"height": RES[0], "width": RES[1]}
mod = Sapiens2ForSemanticSegmentation.from_pretrained(
    str(snap), dtype=torch.float16).to("cuda").eval()
print("seg cargado en %s a %dx%d" % (next(mod.parameters()).device, RES[0], RES[1]))
print()

filas = []
for cid, cuadro, clase, motivo in CASOS:
    if cid not in mapa or cid not in npz_de:
        print("  falta material: %s" % cid)
        continue
    with np.load(npz_de[cid], allow_pickle=True) as z:
        BOX = np.asarray(z["boxes"], dtype="float64")
        P = np.asarray(z["poses"], dtype="float64")
    cap = cv2.VideoCapture(mapa[cid])
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    k = int(cuadro) if cuadro is not None else int(P.shape[0] // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, im = cap.read()
    cap.release()
    if not ok:
        print("  no decodifica %s cuadro %d" % (cid, k))
        continue

    print("%-40s cuadro %4d  %s  (%s)" % (cid[:40], k, clase, motivo))
    lienzo = im.copy()
    for s in (0, 1):
        bb = BOX[k, s]
        if not (np.isfinite(bb).all() and bb[2] > bb[0]):
            print("     lado %d: sin caja" % s)
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
        persona = float((m >= PERSONA_MIN).sum()) / tot
        torso = float(np.isin(m, TORSO_ROPA).sum()) / tot
        brazos = float(np.isin(m, BRAZOS).sum()) / tot
        manos = float(np.isin(m, MANOS).sum()) / tot
        u, cnt = np.unique(m, return_counts=True)
        top = sorted(zip(u.tolist(), cnt.tolist()), key=lambda t: -t[1])[:4]
        detalle = ", ".join("%s %.0f%%" % (NOMBRES.get(int(c), str(c)), 100.0 * n / tot)
                            for c, n in top)
        print("     lado %d  caja %3dx%-3d  persona %.3f  torso %.3f  brazos %.3f  manos %.3f" % (
            s, x2 - x1, y2 - y1, persona, torso, brazos, manos))
        print("              clases dominantes: %s" % detalle)
        filas.append({"clip": cid, "cuadro": k, "lado": s, "clase": clase, "motivo": motivo,
                      "persona": round(persona, 4), "torso": round(torso, 4),
                      "brazos": round(brazos, 4), "manos": round(manos, 4),
                      "dominantes": detalle})

        """Se pinta la mascara sobre el recorte, en el lienzo del cuadro completo."""
        col = np.zeros_like(sub)
        col[m >= PERSONA_MIN] = (0, 0, 255)
        col[np.isin(m, TORSO_ROPA)] = (0, 255, 0)
        col[np.isin(m, MANOS)] = (0, 255, 255)
        mez = (0.55 * sub + 0.45 * col).astype("uint8")
        lienzo[y1:y2, x1:x2] = mez[:, :, ::-1]
        cv2.rectangle(lienzo, (x1, y1), (x2, y2), (255, 200, 0) if s == 0 else (0, 200, 255), 2)
        cv2.putText(lienzo, "persona %.2f torso %.2f" % (persona, torso),
                    (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.rectangle(lienzo, (0, 0), (W, 46), (12, 12, 18), -1)
    cv2.putText(lienzo, "%s  cuadro %d  %s: %s" % (cid[:34], k, clase, motivo),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2)
    cv2.imwrite(str(SALIDA / ("%s__%s_c%04d.png" % (clase, cid[:36], k))),
                cv2.resize(lienzo, (1100, int(1100 * H / W))))
    print()

d = pd.DataFrame(filas)
d.to_csv(SALIDA / "etapa_a_seg_exacto.csv", index=False)
print("=" * 92)
print("ORDENACION DE LOS CASOS CONOCIDOS")
print("=" * 92)
if len(d):
    for cl in ("NEGATIVO", "POSITIVO"):
        g = d[d["clase"] == cl]
        if len(g):
            print("  %-10s n=%d   persona: mediana %.3f  rango [%.3f, %.3f]   torso: mediana %.3f" % (
                cl, len(g), g["persona"].median(), g["persona"].min(), g["persona"].max(),
                g["torso"].median()))
    neg = d[d["clase"] == "NEGATIVO"]
    pos = d[d["clase"] == "POSITIVO"]
    if len(neg) and len(pos):
        print()
        print("  el MAYOR de los negativos ... %.3f" % neg["persona"].max())
        print("  el MENOR de los positivos ... %.3f" % pos["persona"].min())
        if neg["persona"].max() < pos["persona"].min():
            print("  SE ORDENAN SIN SOLAPARSE: hay hueco para un umbral entre los dos.")
        else:
            print("  SE SOLAPAN: con estos casos `seg` no separa por si solo. Hay que mirar las")
            print("  imagenes antes de decidir, y decirlo asi en vez de buscar un umbral que salve.")
print()
print("imagenes en %s" % SALIDA)

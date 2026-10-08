"""El careo del LOTE de la pose: cuanto se gana y **cuanto se mueve el resultado**.

EL CRITERIO, DECLARADO ANTES DE MEDIR. Lotear el modelo de pose tiene que ser un **no-op sobre el
resultado**: todos los recortes se llevan a 384x288, asi que la forma no cambia y el unico camino
por el que el numero puede moverse es que la tarjeta elija otro nucleo segun el tamano del lote y
eso mueva el ultimo bit en fp16.

    se adopta si    el desplazamiento p50 de los keypoints es < 0,02 px
                    y el maximo es < 0,5 px, que es 1/4 del error de pose publicado (2,26 px)
    se rechaza si   se mueve mas, **diciendo cuanto**, en vez de adoptarse "con una advertencia"

Y la referencia de cuanto se mueve el modelo consigo mismo: se corre DOS VECES con lote 1 y se da
esa dispersion al lado. Si el efecto del lote es del orden de la repeticion, no hay efecto.

    python verifica_lote_pose.py CLIP [CLIP ...] [--lotes 1,8,16,32]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "01_extraccion" / "codigo"))
import rutas as RU  # noqa: E402
import video_revision as VR  # noqa: E402

P50_MAX = 0.02
MAX_MAX = 0.5


def una(clip, proc, mod, torch, lote):
    z = np.load(RU.TRABAJO / ("revision_%s.npz" % clip), allow_pickle=True)
    B, pres, mask = z["B"], z["pres"], z["mask"]
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    video = str(z["video"])
    t = time.perf_counter()
    KP, SC, _f = VR.poses_del_clip(proc, mod, torch, video, B, pres, mask, W, H, T, lote=lote)
    return KP, SC, time.perf_counter() - t


CONF = 0.3
"""El mismo umbral con el que el resto del proyecto decide si un keypoint se usa. Comparar por
debajo de el mide puntos que nadie consume."""


def desplaza(A, B, SA=None, SB=None, conf=None):
    """QUE DEVUELVE: (p50, p95, max, n) del desplazamiento en px de los keypoints que los DOS
    tienen.

    Solo los que los dos tienen: un punto que existe en una pasada y no en la otra no es un
    desplazamiento, es una diferencia de deteccion, y mezclarlos daria un maximo enorme por una
    causa distinta de la que se esta midiendo.

    `conf` restringe ademas a los puntos que los dos dan por USABLES. Hace falta porque el modelo
    devuelve coordenadas FINITAS y absurdas para puntos de confianza casi nula -- medido aqui,
    hasta 111.059 px fuera de un cuadro de 1.920. Esos puntos ya los descarta todo lo que consume
    el NPZ, asi que meterlos en esta comparacion mide la basura y no el efecto del lote.
    """
    m = np.isfinite(A).all(axis=-1) & np.isfinite(B).all(axis=-1)
    if conf is not None and SA is not None:
        m = m & (SA >= conf) & (SB >= conf)
    if not m.any():
        return float("nan"), float("nan"), float("nan"), 0
    d = np.linalg.norm(A[m] - B[m], axis=-1)
    return float(np.median(d)), float(np.percentile(d, 95)), float(d.max()), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--lotes", default="1,8,16,32")
    a = ap.parse_args()
    clips = a.clips
    if not clips:
        import glob
        clips = sorted(pathlib.Path(p).name[9:-4]
                       for p in glob.glob(str(RU.TRABAJO / "revision_*.npz")))[:3]
    lotes = [int(x) for x in a.lotes.split(",")]

    import torch
    from transformers import Sapiens2ForPoseEstimation, Sapiens2ImageProcessor
    snap = RU.hf_modelo("sapiens2-pose-0.4b")
    proc = Sapiens2ImageProcessor.from_pretrained(str(snap))
    proc.size = VR.RES_POSE
    mod = Sapiens2ForPoseEstimation.from_pretrained(
        str(snap), dtype=torch.float16).to("cuda").eval()

    print("CAREO DEL LOTE DE LA POSE   (criterio: p50 < %.2f px y max < %.1f px)"
          % (P50_MAX, MAX_MAX))
    print("=" * 96)
    for c in clips:
        print()
        print("%s" % c)
        ref, sref, t1 = una(c, proc, mod, torch, 1)
        ref2, sref2, t1b = una(c, proc, mod, torch, 1)
        p50, p95, mx, n = desplaza(ref, ref2, sref, sref2, CONF)
        print("   %-10s %6.2f s          repeticion con el MISMO lote 1: p50 %.4f  max %.4f px"
              % ("lote 1", t1, p50, mx))
        base = 0.5 * (t1 + t1b)
        for L in lotes:
            if L == 1:
                continue
            KP, SCv, t = una(c, proc, mod, torch, L)
            a50, a95, amx, an = desplaza(ref, KP)
            p50, p95, mx, n = desplaza(ref, KP, sref, SCv, CONF)
            ok = (p50 < P50_MAX and mx < MAX_MAX)
            print("   %-10s %6.2f s %4.2fx  usables>=%.1f: p50 %.4f p95 %.4f max %.4f px (n=%d) %s"
                  % ("lote %d" % L, t, base / max(t, 1e-9), CONF, p50, p95, mx, n,
                     "pasa" if ok else "NO PASA"))
            print("   %-10s %6s %5s  TODOS:        p50 %.4f p95 %.4f max %.1f px (n=%d)"
                  % ("", "", "", a50, a95, amx, an))
    print()
    print("La referencia de arriba es el modelo contra si mismo con el mismo lote. Si el efecto")
    print("del lote es de ese orden, no hay efecto del lote: hay fp16.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

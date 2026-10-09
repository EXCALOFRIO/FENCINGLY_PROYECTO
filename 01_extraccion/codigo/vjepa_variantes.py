"""Variantes de la extraccion de V-JEPA 2, para medir QUE guardar de cada clip en la corrida de florete.

POR QUE EXISTE (2026-10-09, `04_documentacion/informes/VJEPA2_VARIANTES_2026-10-09.md`). `vjepa.py`
guarda una sola cosa: el recorte de la pareja, 4,5 s alrededor del tocado, promediado 4x4 en el
espacio. Lo que se guarde en la corrida es lo unico que habra sin volver a pasar 106.432 videos por la
GPU, asi que antes se mide si sirve guardar mas. Este modulo solo EXTRAE; mide `sonda_vjepa.py
--variante`. Lo que hace `vjepa.py` (ancla, ventana, recorte, normalizacion) se IMPORTA de alli, no se
copia: la variante `rec` tiene que ser exactamente su recorte, y se comprueba (`--comprueba`).

QUE GUARDA, por clip, `vjv_<clip>.npz` en `DESTINO` (una clave por salida pedida):
  rec8   (2, 32, 8, 8, D) fp16   el MISMO recorte y ventana que `vjepa.py`, promediado 8x8 (no 4x4)
  ent8   (2, 32, 8, 8, D) fp16   la imagen ENTERA (sin recortar, 16:9 -> 256x256), misma ventana
  pre4   (2, 32, 4, 4, D) fp16   el recorte de la ventana ANTERIOR, [a - 8,5 s, a - 4,0 s]
  rec4   (2, 32, 4, 4, D) fp16   el recorte de `vjepa.py` a 4x4 (para el modelo grande)
  media_<salida> (2, D) fp32, y idx, idx_pre, caja, caja_pre, n_pre_reales, ancla, via_ancla, fps, T
  vista 0 = la imagen, vista 1 = su ESPEJO horizontal (la sonda impone la antisimetria con las dos)
El 4x4 de `rec8` se saca exacto promediando 2x2, asi que `rec8` contiene a `tok` de `vjepa.py`.

    python vjepa_variantes.py --conjunto anotados                     rec8, ent8, pre4 (ViT-L)
    python vjepa_variantes.py --modelo vjepa2-vitg-fpc64-384 --salidas rec4 --destino D:/...vitg
    python vjepa_variantes.py --comprueba 5      rec8 -> 4x4 contra vjepa_<clip>.npz de 5 clips
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rutas as RU  # noqa: E402
import vjepa as VJ  # noqa: E402

DESTINO = pathlib.Path(os.environ.get("FENCINGLY_VJEPA_VAR_DIR", "D:/FENCINGLY_SCRATCH/vjepa_var"))
PRE_DESDE_S = 8.5
"""La ventana anterior acaba donde empieza la de `vjepa.py` (a - 4,0 s) y dura lo mismo (4,5 s), con
la misma densidad (64 cuadros, ~14/s). Medido sobre los anotados: el tocado cae a una mediana de 7,4 s
del inicio del clip, asi que la ventana de `vjepa.py` deja fuera ~3,4 s y esta los cubre."""
SALIDAS = ("rec8", "ent8", "pre4", "rec4")


def ventana_pre(a, fps):
    """64 cuadros de [a - 8,5 s, a - 4,0 s] a la densidad de la ventana principal. Los que caen antes
    del cuadro 0 repiten el 0, y `n_pre_reales` cuenta los que no."""
    ini = a - int(round(PRE_DESDE_S * fps))
    fin = a - int(round(VJ.ANTES_S * fps))
    idx = np.round(np.linspace(ini, fin, VJ.N_CUADROS)).astype("int64")
    return np.clip(idx, 0, None).astype("int32"), int((idx >= 0).sum())


def prepara(clip, salidas, lado):
    """CPU: como `vjepa.prepara`, pero decodifica UNA vez y saca todas las vistas pedidas."""
    import cv2
    p = RU.TRABAJO / ("poses_%s.npz" % clip)
    if not p.is_file():
        return None, "sin poses"
    with np.load(p, allow_pickle=True) as z:
        B, mask = z["B"], z["mask"]
        W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
        fps = float(z["fps"]) if "fps" in z.files else 0.0
    video = RU.video(clip)
    if not video:
        return None, "sin video"
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(video)
    fps = fps or float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    a, via = VJ.ancla(clip, mask, fps)
    a = min(a, T - 1)
    idx = VJ.ventana(a, fps, T)
    caja = VJ.caja_recorte(B, idx, W, H)
    idx_pre, n_pre = ventana_pre(a, fps)
    caja_pre = VJ.caja_recorte(B, np.clip(idx_pre, 0, T - 1), W, H)
    quiere_rec = any(s.startswith("rec") for s in salidas)
    quiere_ent = "ent8" in salidas
    quiere_pre = "pre4" in salidas
    q_main = set(int(i) for i in idx)
    q_pre = set(int(i) for i in idx_pre) if quiere_pre else set()
    tope = max(int(idx.max()), max(q_pre) if q_pre else 0)

    def corta(im, c):
        x1, y1, x2, y2 = [int(round(v)) for v in c]
        r = cv2.resize(im[y1:y2, x1:x2], (lado, lado), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(r, cv2.COLOR_BGR2RGB)

    rec, ent, pre = {}, {}, {}
    k = 0
    while k <= tope:
        ok, im = cap.read()
        if not ok:
            break
        if k in q_main:
            if quiere_rec:
                rec[k] = corta(im, caja)
            if quiere_ent:
                ent[k] = cv2.cvtColor(cv2.resize(im, (lado, lado), interpolation=cv2.INTER_AREA),
                                      cv2.COLOR_BGR2RGB)
        if k in q_pre:
            pre[k] = corta(im, caja_pre)
        k += 1
    cap.release()
    principal = rec if quiere_rec else ent
    if (quiere_rec or quiere_ent) and not principal:
        return None, "video ilegible"

    def pila(cuadros, ids):
        if not cuadros:
            return None
        ultimo = cuadros[max(cuadros)]
        return np.stack([cuadros.get(int(i), ultimo) for i in ids])

    d = {"clip": clip, "idx": idx, "idx_pre": idx_pre, "caja": caja, "caja_pre": caja_pre,
         "n_pre_reales": n_pre, "ancla": a, "via": via, "fps": fps, "T": T,
         "n_faltan": int(sum(int(i) not in principal for i in idx)) if principal else None,
         "s_lee": time.perf_counter() - t0, "pilas": {}}
    if quiere_rec:
        d["pilas"]["rec"] = pila(rec, idx)
    if quiere_ent:
        d["pilas"]["ent"] = pila(ent, idx)
    if quiere_pre:
        d["pilas"]["pre"] = pila(pre, idx_pre)
    return d, ""


def _pool(h, lado, k):
    """(2, 32*s*s, D) -> (2, 32, k, k, D), promedio de bloques iguales."""
    n_t, n_s = VJ.N_CUADROS // 2, lado // 16
    h = h.reshape(2, n_t, n_s, n_s, -1)
    b = n_s // k
    return h.reshape(2, n_t, k, b, k, b, h.shape[-1]).float().mean(dim=(3, 5))


def embebe(mod, pila, lado):
    import torch
    x = torch.from_numpy(pila).to("cuda").permute(0, 3, 1, 2).float().div_(255.0)
    x = (x - torch.tensor(VJ.MEDIA, device="cuda")[:, None, None]) / torch.tensor(
        VJ.STD, device="cuda")[:, None, None]
    x = torch.stack([x, torch.flip(x, dims=[-1])]).half()
    with torch.inference_mode():
        return mod.get_vision_features(x)


def embebe_todo(mod, d, salidas, lado):
    import torch
    t0 = time.perf_counter()
    out = {}
    if "rec" in d["pilas"]:
        h = embebe(mod, d["pilas"]["rec"], lado)
        out["media_rec"] = h.float().mean(dim=1)
        if "rec8" in salidas:
            out["rec8"] = _pool(h, lado, 8)
        if "rec4" in salidas:
            out["rec4"] = _pool(h, lado, 4)
    if "ent" in d["pilas"]:
        h = embebe(mod, d["pilas"]["ent"], lado)
        out["ent8"], out["media_ent"] = _pool(h, lado, 8), h.float().mean(dim=1)
    if "pre" in d["pilas"]:
        h = embebe(mod, d["pilas"]["pre"], lado)
        out["pre4"], out["media_pre"] = _pool(h, lado, 4), h.float().mean(dim=1)
    torch.cuda.synchronize()
    res = {k: (v.cpu().numpy().astype("float32") if k.startswith("media")
               else v.half().cpu().numpy()) for k, v in out.items()}
    return res, time.perf_counter() - t0


def guarda(d, res, s_gpu, destino, modelo):
    destino.mkdir(parents=True, exist_ok=True)
    f = destino / ("vjv_%s.npz" % d["clip"])
    tmp = destino / ("vjv_%s.part.npz" % d["clip"])
    np.savez(tmp, idx=d["idx"], idx_pre=d["idx_pre"], caja=d["caja"], caja_pre=d["caja_pre"],
             n_pre_reales=np.int32(d["n_pre_reales"]), ancla=np.int32(d["ancla"]),
             via_ancla=d["via"], fps=np.float64(d["fps"]), T=np.int32(d["T"]),
             n_faltan=np.int32(-1 if d["n_faltan"] is None else d["n_faltan"]), modelo=modelo,
             s_gpu=np.float64(s_gpu), s_lee=np.float64(d["s_lee"]), **res)
    os.replace(tmp, f)


def carga_modelo(nombre):
    import torch
    from transformers import VJEPA2Model
    mod = VJEPA2Model.from_pretrained(RU.hf_modelo(nombre), dtype=torch.float16,
                                      attn_implementation="sdpa")
    return mod.to("cuda").eval()


def corre(clips, salidas, destino, modelo, lado, hilos=3):
    from concurrent.futures import ThreadPoolExecutor
    pend = [c for c in clips if not (destino / ("vjv_%s.npz" % c)).is_file()]
    print("%d clips, %d ya hechos, %d pendientes -> %s  (%s, %s, %d px)"
          % (len(clips), len(clips) - len(pend), len(pend), destino, modelo, ",".join(salidas),
             lado), flush=True)
    if not pend:
        return
    mod = carga_modelo(modelo)
    t0 = time.perf_counter()
    hechos, sg = 0, []
    with ThreadPoolExecutor(hilos) as ex:
        futs = [ex.submit(prepara, c, salidas, lado) for c in pend[:2 * hilos]]
        sig = 2 * hilos
        while futs:
            f = futs.pop(0)
            if sig < len(pend):
                futs.append(ex.submit(prepara, pend[sig], salidas, lado))
                sig += 1
            try:
                d, motivo = f.result()
                if d is None:
                    print("   salta: %s" % motivo, flush=True)
                    continue
                res, s = embebe_todo(mod, d, salidas, lado)
                guarda(d, res, s, destino, modelo)
                hechos += 1
                sg.append(s)
            except Exception as e:
                print("   FALLA %s: %s" % (type(e).__name__, str(e)[:160]), flush=True)
            if hechos and hechos % 100 == 0:
                el = time.perf_counter() - t0
                print("   %d / %d  %.2f s/clip  (GPU mediana %.2f s)"
                      % (hechos, len(pend), el / hechos, float(np.median(sg))), flush=True)
    el = time.perf_counter() - t0
    print("hechos %d en %.0f s: %.2f s/clip de pared, GPU mediana %.2f s/clip"
          % (hechos, el, el / max(hechos, 1), float(np.median(sg)) if sg else float("nan")),
          flush=True)


def comprueba(n, destino):
    """`rec8` promediado 2x2 tiene que ser el `tok` de `vjepa.py` (mismo recorte, misma ventana,
    mismo modelo): si no, la variante no mide lo que dice medir."""
    import random
    hechos = sorted(p.name[4:-4] for p in destino.glob("vjv_*.npz") if ".part" not in p.name)
    random.Random(0).shuffle(hechos)
    peor = 0.0
    for c in hechos[:n]:
        with np.load(destino / ("vjv_%s.npz" % c)) as v, np.load(VJ.DESTINO / ("vjepa_%s.npz" % c)) as o:
            r8 = v["rec8"].astype("float32")
            r4 = r8.reshape(2, 32, 4, 2, 4, 2, -1).mean(axis=(3, 5))
            dif = float(np.abs(r4 - o["tok"].astype("float32")).max())
            rel = dif / float(np.abs(o["tok"].astype("float32")).max())
            igual_idx = bool((v["idx"] == o["idx"]).all() and np.allclose(v["caja"], o["caja"]))
            print("   %s  max|4x4(rec8) - tok| %.4f (relativo %.4f)  idx y caja iguales: %s"
                  % (c, dif, rel, igual_idx))
            peor = max(peor, rel)
    print("peor diferencia relativa %.4f" % peor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", default="anotados", help="anotados: los que tienen `initiated`")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--destino", default=None)
    ap.add_argument("--modelo", default=VJ.MODELO)
    ap.add_argument("--salidas", default="rec8,ent8,pre4")
    ap.add_argument("--todos", action="store_true", help="todo el conjunto con poses")
    ap.add_argument("--comprueba", type=int, default=0)
    a = ap.parse_args()
    destino = pathlib.Path(a.destino) if a.destino else DESTINO
    if a.comprueba:
        comprueba(a.comprueba, destino)
        return
    salidas = [s for s in a.salidas.split(",") if s]
    assert all(s in SALIDAS for s in salidas), salidas
    lado = 384 if a.modelo.endswith("-384") else 256
    if a.clips:
        clips = a.clips
    elif a.todos:
        import lotes as LO
        clips = sorted(c for c in LO.conjunto(a.conjunto)
                       if (RU.TRABAJO / ("poses_%s.npz" % c)).is_file())
    else:
        clips = VJ.clips_initiated()
    corre(clips, salidas, destino, a.modelo, lado)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    main()

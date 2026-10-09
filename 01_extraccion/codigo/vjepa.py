"""Embeddings de video V-JEPA 2 (ViT-L, MIT) del tramo ANTES del tocado, para medir si ayudan a arbitrar.

POR QUE EXISTE (2026-10-09, `04_documentacion/informes/VJEPA2_2026-10-09.md`). Todo lo que el
proyecto sabe de un clip sale de poses y cajas. Un codificador de video generico, congelado, ve lo
que las poses no ven (la hoja, el contacto, el ritmo). La pregunta es si eso predice `initiated`
mejor que el etiquetador de poses; este modulo solo EXTRAE, la sonda esta en
`02_validacion/herramienta/sonda_vjepa.py`. Todo lo de abajo esta fijado en el protocolo ANTES de
medir: cambiarlo despues de ver resultados es elegir la ventana mirando el reservado.

QUE GUARDA, por clip, `vjepa_<clip>.npz` en `DESTINO`:
  tok    (2, 32, 4, 4, 1024) fp16   tokens del codificador promediados 4x4 en el espacio, por vista
  media  (2, 1024) fp32             media de todos los tokens, por vista
  vista 0 = el recorte, vista 1 = su ESPEJO horizontal (para imponer la antisimetria en la sonda)
  idx (64,) cuadros usados, caja (4,) recorte en pixeles, ancla, via_ancla, fps, T, s_gpu, s_lee

    python vjepa.py --conjunto anotados               los que tienen `initiated` y poses
    python vjepa.py --clips A B C --destino D:/prueba
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import rutas as RU  # noqa: E402

MODELO = "vjepa2-vitl-fpc64-256"
N_CUADROS = 64
LADO = 256
ANTES_S = 4.0
DESPUES_S = 0.5
FINAL_S = 2.2
"""Sin luz, el ancla es el ultimo cuadro util menos 2,2 s: la mediana MEDIDA (5.429 clips anotados)
del tocado al final del clip es 66 cuadros, ~2,2 s."""
MARGEN_X, MARGEN_Y = 0.10, 0.15
RAZON_MIN, RAZON_MAX = 1.0, 2.0
MEDIA = np.array([0.485, 0.456, 0.406], dtype="float32")
STD = np.array([0.229, 0.224, 0.225], dtype="float32")
DESTINO = pathlib.Path(os.environ.get("FENCINGLY_VJEPA_DIR", "D:/FENCINGLY_SCRATCH/vjepa"))


def _entero(x):
    try:
        v = int(float(x))
    except (TypeError, ValueError):
        return None
    return v


def ancla(clip, mask, fps):
    """QUE DEVUELVE: (cuadro, via). `zona_` si su `cuadro_tocado` > 0; si no, el primer
    `cuadro_encendido` > 0 de `lights_`; si no, el ultimo cuadro util menos `FINAL_S`."""
    z = RU.TRABAJO / ("zona_%s.csv" % clip)
    if z.is_file():
        with open(z, encoding="utf-8") as fh:
            r = next(csv.DictReader(fh), None)
        v = _entero(r.get("cuadro_tocado")) if r else None
        if v and v > 0:
            return v, "zona"
    li = RU.TRABAJO / ("lights_%s.csv" % clip)
    if li.is_file():
        with open(li, encoding="utf-8") as fh:
            vs = [_entero(r.get("cuadro_encendido")) for r in csv.DictReader(fh)]
        vs = [v for v in vs if v and v > 0]
        if vs:
            return min(vs), "luces"
    ks = np.nonzero(mask)[0]
    fin = int(ks[-1]) if len(ks) else len(mask) - 1
    return max(0, fin - int(round(FINAL_S * fps))), "final"


def ventana(a, fps, T):
    ini = max(0, a - int(round(ANTES_S * fps)))
    fin = min(T - 1, a + int(round(DESPUES_S * fps)))
    if fin <= ini:
        ini, fin = 0, T - 1
    return np.round(np.linspace(ini, fin, N_CUADROS)).astype("int32")


def caja_recorte(B, idx, W, H):
    """La union de las cajas de los dos tiradores en la ventana, con margen y razon 1..2, en pixeles."""
    b = B[idx].reshape(-1, 4).astype("float64")
    b = b[np.isfinite(b).all(axis=1)]
    if not len(b):
        return np.array([0, 0, W, H], dtype="float32")
    x1, y1 = b[:, 0].min() * W, b[:, 1].min() * H
    x2, y2 = b[:, 2].max() * W, b[:, 3].max() * H
    w, h = max(x2 - x1, 2.0), max(y2 - y1, 2.0)
    x1, x2 = x1 - MARGEN_X * w, x2 + MARGEN_X * w
    y1, y2 = y1 - MARGEN_Y * h, y2 + MARGEN_Y * h
    w, h = x2 - x1, y2 - y1
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    if w / h > RAZON_MAX:
        h = w / RAZON_MAX
    elif w / h < RAZON_MIN:
        w = h * RAZON_MIN
    w, h = min(w, W), min(h, H)
    x1 = min(max(cx - w / 2, 0.0), W - w)
    y1 = min(max(cy - h / 2, 0.0), H - h)
    return np.array([x1, y1, x1 + w, y1 + h], dtype="float32")


def prepara(clip):
    """CPU: lee poses, decide ventana y recorte, decodifica. QUE DEVUELVE: dict o (None, motivo)."""
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
    a, via = ancla(clip, mask, fps)
    idx = ventana(min(a, T - 1), fps, T)
    caja = caja_recorte(B, idx, W, H)
    x1, y1, x2, y2 = [int(round(v)) for v in caja]
    quiero = set(int(i) for i in idx)
    tope = int(idx.max())
    cuadros = {}
    k = 0
    while k <= tope:
        ok, im = cap.read()
        if not ok:
            break
        if k in quiero:
            r = cv2.resize(im[y1:y2, x1:x2], (LADO, LADO), interpolation=cv2.INTER_AREA)
            cuadros[k] = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        k += 1
    cap.release()
    if not cuadros:
        return None, "video ilegible"
    ultimo = cuadros[max(cuadros)]
    pila = np.stack([cuadros.get(int(i), ultimo) for i in idx])
    """Si el video tiene menos cuadros que `T` (pasa en algun clip de fps variable), los que faltan
    repiten el ultimo leido, y se cuenta en `n_faltan`."""
    return {"clip": clip, "pila": pila, "idx": idx, "caja": caja, "ancla": a, "via": via,
            "fps": fps, "T": T, "n_faltan": int(sum(int(i) not in cuadros for i in idx)),
            "s_lee": time.perf_counter() - t0}, ""


def carga_modelo():
    import torch
    from transformers import VJEPA2Model
    snap = RU.hf_modelo(MODELO)
    mod = VJEPA2Model.from_pretrained(snap, dtype=torch.float16, attn_implementation="sdpa")
    return mod.to("cuda").eval()


def embebe(mod, d):
    import torch
    t0 = time.perf_counter()
    x = torch.from_numpy(d["pila"]).to("cuda")
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    x = (x - torch.tensor(MEDIA, device="cuda")[:, None, None]) / torch.tensor(
        STD, device="cuda")[:, None, None]
    x = torch.stack([x, torch.flip(x, dims=[-1])]).half()
    with torch.inference_mode():
        h = mod.get_vision_features(x)
        n_t, n_s = N_CUADROS // 2, LADO // 16
        h = h.reshape(2, n_t, n_s, n_s, -1)
        tok = h.reshape(2, n_t, 4, n_s // 4, 4, n_s // 4, h.shape[-1]).float().mean(dim=(3, 5))
        media = h.float().mean(dim=(1, 2, 3))
    torch.cuda.synchronize()
    return (tok.half().cpu().numpy(), media.cpu().numpy(), time.perf_counter() - t0)


def guarda(d, tok, media, s_gpu, destino):
    destino.mkdir(parents=True, exist_ok=True)
    f = destino / ("vjepa_%s.npz" % d["clip"])
    tmp = destino / ("vjepa_%s.part.npz" % d["clip"])
    np.savez(tmp, tok=tok, media=media.astype("float32"), idx=d["idx"], caja=d["caja"],
             ancla=np.int32(d["ancla"]), via_ancla=d["via"], fps=np.float64(d["fps"]),
             T=np.int32(d["T"]), n_faltan=np.int32(d["n_faltan"]), modelo=MODELO,
             ventana_s=np.array([-ANTES_S, DESPUES_S]), s_gpu=np.float64(s_gpu),
             s_lee=np.float64(d["s_lee"]))
    os.replace(tmp, f)


def corre(clips, destino=DESTINO, hilos=3):
    from concurrent.futures import ThreadPoolExecutor
    pend = [c for c in clips if not (destino / ("vjepa_%s.npz" % c)).is_file()]
    print("%d clips, %d ya hechos, %d pendientes -> %s"
          % (len(clips), len(clips) - len(pend), len(pend), destino), flush=True)
    if not pend:
        return
    mod = carga_modelo()
    t0 = time.perf_counter()
    hechos, sg = 0, []
    with ThreadPoolExecutor(hilos) as ex:
        """La decodificacion (CPU) va por delante en hilos: OpenCV suelta el GIL al decodificar,
        asi que la GPU no espera al disco."""
        futs = [ex.submit(prepara, c) for c in pend[:2 * hilos]]
        sig = 2 * hilos
        while futs:
            f = futs.pop(0)
            if sig < len(pend):
                futs.append(ex.submit(prepara, pend[sig]))
                sig += 1
            try:
                d, motivo = f.result()
                if d is None:
                    print("   salta: %s" % motivo, flush=True)
                    continue
                tok, media, s = embebe(mod, d)
                guarda(d, tok, media, s, destino)
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


def clips_initiated():
    import catalogo as CA
    df = CA.manifiesto()
    m = df["initiated"].isin(CA.INICIA)
    return sorted(c for c in df.loc[m, "clip_id"]
                  if (RU.TRABAJO / ("poses_%s.npz" % c)).is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", default=None, help="anotados: los que tienen `initiated`")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--destino", default=None)
    ap.add_argument("--todos", action="store_true",
                    help="todo el conjunto con poses, no solo los que tienen `initiated` (pregunta 2)")
    a = ap.parse_args()
    destino = pathlib.Path(a.destino) if a.destino else DESTINO
    if a.clips:
        clips = a.clips
    elif a.conjunto and a.todos:
        import lotes as LO
        clips = sorted(c for c in LO.conjunto(a.conjunto)
                       if (RU.TRABAJO / ("poses_%s.npz" % c)).is_file())
    else:
        clips = clips_initiated()
    corre(clips, destino)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    main()

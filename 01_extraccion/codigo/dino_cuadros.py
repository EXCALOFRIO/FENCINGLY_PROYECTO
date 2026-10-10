"""DINOv2 CUADRO A CUADRO: muchos cuadros a mas resolucion, la alternativa a los 64 cuadros de V-JEPA.

POR QUE EXISTE (2026-10-10, `04_documentacion/informes/VJEPA2_VARIANTES_2026-10-09.md`). V-JEPA ve
64 cuadros a 256 (o 384) px porque mira todos a la vez y su coste crece con el cuadrado de los
tokens. Un modelo de IMAGEN mira cada cuadro por separado: se le pueden dar los ~130 cuadros de la
ventana y mas resolucion por cuadro, y la relacion en el tiempo la aprende despues una red pequenia
sobre lo guardado. DINOv2 (Meta, Apache-2.0, sin acceso restringido) es el codificador de imagen
generico de referencia. Lo que NO hace: inventar detalle que el video no tiene (a 720p la hoja mide
~1 px, `HOJA_2026-10-09.md`).

QUE GUARDA, por clip, `dino_<clip>.npz` en `DESTINO`:
  tok   (2, 128, 17, 1024) fp16   por vista y cuadro: el CLS y los parches promediados 4x4
  vista 0 = el recorte de la pareja de `vjepa.py`, vista 1 = su ESPEJO (para la antisimetria)
  idx (128,) cuadros usados: 128 a paso uniforme en [a - 4,0 s, a + 0,5 s] (~28/s; a 30 fps casi
  todos los cuadros), misma ancla y mismo recorte que `vjepa.py`; caja, ancla, via_ancla, fps, T

    python dino_cuadros.py --todos                  los anotados con poses
    python dino_cuadros.py --clips A B --destino D:/prueba
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

MODELO = os.environ.get("FENCINGLY_DINO", "dinov2-large")
N_CUADROS = 128
LADO = 448
"""448 = 32 x 14: parches de 14 px de DINOv2, 1.024 parches por cuadro."""
LOTE = 32
DESTINO = pathlib.Path(os.environ.get("FENCINGLY_DINO_DIR", "D:/FENCINGLY_SCRATCH/dino"))


def ventana(a, fps, T):
    ini = max(0, a - int(round(VJ.ANTES_S * fps)))
    fin = min(T - 1, a + int(round(VJ.DESPUES_S * fps)))
    if fin <= ini:
        ini, fin = 0, T - 1
    return np.round(np.linspace(ini, fin, N_CUADROS)).astype("int32")


def prepara(clip):
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
    caja = VJ.caja_recorte(B, VJ.ventana(a, fps, T), W, H)
    """El MISMO recorte que V-JEPA (la caja se calcula sobre su ventana de 64), para que la unica
    diferencia sea el modelo, la resolucion y el numero de cuadros."""
    idx = ventana(a, fps, T)
    x1, y1, x2, y2 = [int(round(v)) for v in caja]
    quiero = set(int(i) for i in idx)
    cuadros = {}
    k = 0
    while k <= int(idx.max()):
        ok, im = cap.read()
        if not ok:
            break
        if k in quiero:
            r = im[y1:y2, x1:x2]
            interp = cv2.INTER_AREA if r.shape[0] >= LADO else cv2.INTER_CUBIC
            cuadros[k] = cv2.cvtColor(cv2.resize(r, (LADO, LADO), interpolation=interp),
                                      cv2.COLOR_BGR2RGB)
        k += 1
    cap.release()
    if not cuadros:
        return None, "video ilegible"
    ultimo = cuadros[max(cuadros)]
    pila = np.stack([cuadros.get(int(i), ultimo) for i in idx])
    return {"clip": clip, "pila": pila, "idx": idx, "caja": caja, "ancla": a, "via": via,
            "fps": fps, "T": T, "n_faltan": int(sum(int(i) not in cuadros for i in idx)),
            "s_lee": time.perf_counter() - t0}, ""


def carga_modelo():
    import torch
    from transformers import Dinov2Model
    mod = Dinov2Model.from_pretrained(RU.hf_modelo(MODELO), dtype=torch.float16,
                                      attn_implementation="sdpa")
    return mod.to("cuda").eval()


def embebe(mod, pila):
    import torch
    t0 = time.perf_counter()
    x = torch.from_numpy(pila).to("cuda").permute(0, 3, 1, 2).float().div_(255.0)
    x = (x - torch.tensor(VJ.MEDIA, device="cuda")[:, None, None]) / torch.tensor(
        VJ.STD, device="cuda")[:, None, None]
    x = torch.cat([x, torch.flip(x, dims=[-1])]).half()
    n_s = LADO // 14
    out = []
    with torch.inference_mode():
        for i in range(0, len(x), LOTE):
            h = mod(pixel_values=x[i:i + LOTE]).last_hidden_state
            cls, par = h[:, :1], h[:, 1:]
            par = par.reshape(len(par), 4, n_s // 4, 4, n_s // 4, -1).float().mean(dim=(2, 4))
            out.append(torch.cat([cls.float(), par.reshape(len(par), 16, -1)], dim=1).half())
    tok = torch.cat(out).reshape(2, N_CUADROS, 17, -1)
    torch.cuda.synchronize()
    return tok.cpu().numpy(), time.perf_counter() - t0


def guarda(d, tok, s_gpu, destino):
    destino.mkdir(parents=True, exist_ok=True)
    f = destino / ("dino_%s.npz" % d["clip"])
    tmp = destino / ("dino_%s.part.npz" % d["clip"])
    np.savez(tmp, tok=tok, idx=d["idx"], caja=d["caja"], ancla=np.int32(d["ancla"]),
             via_ancla=d["via"], fps=np.float64(d["fps"]), T=np.int32(d["T"]),
             n_faltan=np.int32(d["n_faltan"]), modelo=MODELO, lado=np.int32(LADO),
             s_gpu=np.float64(s_gpu), s_lee=np.float64(d["s_lee"]))
    os.replace(tmp, f)


def corre(clips, destino=DESTINO, hilos=3):
    from concurrent.futures import ThreadPoolExecutor
    pend = [c for c in clips if not (destino / ("dino_%s.npz" % c)).is_file()]
    print("%d clips, %d ya hechos, %d pendientes -> %s  (%s, %d px, %d cuadros)"
          % (len(clips), len(clips) - len(pend), len(pend), destino, MODELO, LADO, N_CUADROS),
          flush=True)
    if not pend:
        return
    mod = carga_modelo()
    t0 = time.perf_counter()
    hechos, sg = 0, []
    with ThreadPoolExecutor(hilos) as ex:
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
                tok, s = embebe(mod, d["pila"])
                guarda(d, tok, s, destino)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", default="anotados")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--destino", default=None)
    ap.add_argument("--todos", action="store_true", help="todo el conjunto con poses")
    a = ap.parse_args()
    destino = pathlib.Path(a.destino) if a.destino else DESTINO
    if a.clips:
        clips = a.clips
    elif a.todos:
        import lotes as LO
        clips = sorted(c for c in LO.conjunto(a.conjunto)
                       if (RU.TRABAJO / ("poses_%s.npz" % c)).is_file())
    else:
        clips = VJ.clips_initiated()
    corre(clips, destino)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    main()

"""Se puede acelerar Sapiens 2? Cuatro palancas, medidas, y con la confianza vigilada en cada una.

LA PALANCA GRANDE, y la razon por la que existe este fichero: el modelo espera 1024x768, que con
parche 16 son 64x48 = 3.072 tokens. Pero nuestros recortes de tirador miden **unos 160x158 px**, asi
que le estamos dando seis veces mas resolucion de la que hay. Bajar la entrada reduce los tokens de
forma cuadratica, y el `config.json` trae `pos_embed_rescale: 2.0`, que sugiere que la interpolacion
del embedding de posicion esta prevista.

LAS CUATRO PALANCAS:
  1. RESOLUCION de entrada: 1024x768 -> 768x576 -> 512x384 -> 384x288
  2. IMPLEMENTACION DE ATENCION: la que venga por defecto contra `sdpa`
  3. `torch.compile`, que en un ViT suele dar entre 1,3x y 2x
  4. bf16 contra fp16

REGLA QUE HACE VALIDA LA MEDIDA: **cada variante se juzga por velocidad Y por confianza a la vez.**
Una version dos veces mas rapida que pierde el pulgar no sirve para nada, porque el pulgar es el eje
del cono. Se vigila la cadena del pulgar derecho (indices 112-116) y el tronco con brazos (5-10).
"""
from __future__ import annotations

import os
import pathlib
import time

CACHE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565\scratchpad\hoja_seg\hf")
os.environ["HF_HOME"] = str(CACHE)

import cv2
import numpy as np
import torch

BASE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565")
PESOS = CACHE / "hub/models--facebook--sapiens2-pose-0.4b/snapshots/73572e121602e8a6eb40eb841732e88f386d250b"
NPZ_DIRS = [BASE / "localrun/runs_extra1/data_20260907T005507157743Z/npz",
            BASE / "localrun/runs_extra2/data_20260907T005507370214Z/npz",
            BASE / "scratchpad/catalogo/ws_completo/npz"]
CLIP = "SentimentalLargeBedlingtonterrier"
N = 10
PULGAR_DER = [112, 113, 114, 115, 116]
TRONCO = list(range(5, 11))

mp4 = None
for raiz, _d, fs in os.walk("D:/FENCINGLY_VIDEOS_RAW"):
    if "%s.mp4" % CLIP in fs:
        mp4 = os.path.join(raiz, "%s.mp4" % CLIP)
        break
fz = next(d / ("%s.npz" % CLIP) for d in NPZ_DIRS if (d / ("%s.npz" % CLIP)).is_file())
with np.load(fz, allow_pickle=True) as z:
    BOX = np.asarray(z["boxes"], dtype="float64")

cap = cv2.VideoCapture(mp4)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ini = max(0, T // 2 - N // 2)
cap.set(cv2.CAP_PROP_POS_FRAMES, ini)
ims, cjs, lados = [], [], []
for i in range(ini, min(T, ini + N)):
    ok, im = cap.read()
    if not ok:
        break
    rgb = np.ascontiguousarray(im[:, :, ::-1])
    for s in (0, 1):
        b = BOX[i, s]
        if np.isfinite(b).all() and b[2] > b[0]:
            x, y = float(b[0] * W), float(b[1] * H)
            ims.append(rgb)
            cjs.append([[x, y, float(b[2] * W) - x, float(b[3] * H) - y]])
            lados.append(s)
cap.release()
print("GPU: %s" % torch.cuda.get_device_name(0))
print("clip %s: %d pasadas de prueba" % (CLIP, len(ims)))
anchos = [c[0][2] for c in cjs]
altos = [c[0][3] for c in cjs]
print("cajas: %.0fx%.0f px de mediana  ->  a 1024x768 se amplia x%.1f" % (
    np.median(anchos), np.median(altos), 1024.0 / max(np.median(altos), 1)))

from transformers import Sapiens2ForPoseEstimation, Sapiens2ImageProcessor

resultados = []


def mide(nombre, size=None, attn=None, compilar=False, dtype=torch.float16):
    """Carga, calienta y mide. Devuelve ms/pasada y las confianzas que deciden si vale."""
    try:
        proc = Sapiens2ImageProcessor.from_pretrained(str(PESOS))
        if size:
            proc.size = {"height": size[0], "width": size[1]}
        kw = {"dtype": dtype}
        if attn:
            kw["attn_implementation"] = attn
        mod = Sapiens2ForPoseEstimation.from_pretrained(str(PESOS), **kw).to("cuda").eval()
        if compilar:
            mod = torch.compile(mod, mode="reduce-overhead")

        with torch.inference_mode():
            for _ in range(3 if compilar else 1):
                e = proc(images=ims[:2], boxes=cjs[:2], return_tensors="pt")
                e = {k: (v.to("cuda").to(dtype) if v.is_floating_point() else v.to("cuda"))
                     for k, v in e.items()}
                _ = mod(**e)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        scs = []
        with torch.inference_mode():
            for i in range(0, len(ims), 2):
                e = proc(images=ims[i:i + 2], boxes=cjs[i:i + 2], return_tensors="pt")
                e = {k: (v.to("cuda").to(dtype) if v.is_floating_point() else v.to("cuda"))
                     for k, v in e.items()}
                o = mod(**e)
                r = proc.post_process_pose_estimation(o, boxes=cjs[i:i + 2], threshold=None)
                for j in range(len(cjs[i:i + 2])):
                    rr = r[j][0] if isinstance(r[j], list) else r[j]
                    sc = rr["scores"]
                    scs.append(sc.cpu().numpy() if hasattr(sc, "cpu") else np.asarray(sc))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        S = np.stack(scs)
        ms = 1000.0 * dt / len(ims)
        fila = {"nombre": nombre, "ms": ms,
                "tronco": float(np.median(S[:, TRONCO])),
                "pulgar": float(np.median(S[:, PULGAR_DER])),
                "pulgar_ok": 100.0 * float((S[:, PULGAR_DER] >= 0.3).mean()),
                "vram": torch.cuda.max_memory_allocated() / 1e9}
        resultados.append(fila)
        print("  %-34s %7.1f ms/pasada   tronco %.3f   pulgar %.3f (%3.0f %% >= 0,3)   VRAM %.2f GB" % (
            nombre, ms, fila["tronco"], fila["pulgar"], fila["pulgar_ok"], fila["vram"]))
    except Exception as exc:
        print("  %-34s FALLO %s: %s" % (nombre, type(exc).__name__, str(exc)[:78]))
    finally:
        for v in ("mod", "proc"):
            if v in dir():
                pass
        torch.cuda.empty_cache()


print()
print("=" * 96)
print("1) RESOLUCION DE ENTRADA  (el recorte real mide ~160 px, asi que 1024 es sobremuestrear x6)")
print("=" * 96)
mide("1024x768  (por defecto)", size=(1024, 768))
mide("768x576", size=(768, 576))
mide("512x384", size=(512, 384))
mide("384x288", size=(384, 288))
mide("256x192", size=(256, 192))

print()
print("=" * 96)
print("2) IMPLEMENTACION DE ATENCION, a la mejor resolucion que aguante la confianza")
print("=" * 96)
mide("512x384 + sdpa", size=(512, 384), attn="sdpa")
mide("512x384 + eager", size=(512, 384), attn="eager")

print()
print("=" * 96)
print("3) bf16 en vez de fp16")
print("=" * 96)
mide("512x384 + bf16", size=(512, 384), dtype=torch.bfloat16)

print()
print("=" * 96)
print("4) torch.compile (la primera pasada compila, por eso se calienta 3 veces)")
print("=" * 96)
mide("512x384 + compile", size=(512, 384), compilar=True)

print()
print("=" * 96)
print("RESUMEN, ordenado por velocidad")
print("=" * 96)
base = next((r["ms"] for r in resultados if r["nombre"].startswith("1024")), None)
print("%-34s %9s %9s %9s %8s" % ("variante", "ms/pasada", "acelera", "pulgar", "% >= 0,3"))
for r in sorted(resultados, key=lambda x: x["ms"]):
    acel = ("%.1fx" % (base / r["ms"])) if base else "-"
    aviso = "" if r["pulgar_ok"] >= 90 else "   <- PIERDE EL PULGAR"
    print("%-34s %9.1f %9s %9.3f %7.0f%%%s" % (
        r["nombre"], r["ms"], acel, r["pulgar"], r["pulgar_ok"], aviso))

buenas = [r for r in resultados if r["pulgar_ok"] >= 90]
if buenas:
    mejor = min(buenas, key=lambda x: x["ms"])
    print()
    print("LA MEJOR QUE CONSERVA EL PULGAR: %s con %.1f ms/pasada" % (mejor["nombre"], mejor["ms"]))
    ms_cuadro = mejor["ms"] * 2
    h = 12e6 * ms_cuadro / 1000.0 / 3600.0
    print("  -> %.0f ms por cuadro de video (2 tiradores)" % ms_cuadro)
    print("  -> 12 M cuadros en %.0f h = %.1f dias   (antes: 87 dias)" % (h, h / 24))

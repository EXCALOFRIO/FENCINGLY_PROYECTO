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
  rec4   (2, 32, 4, 4, D) fp16   el recorte de `vjepa.py` a 4x4
  LOS TRES OJOS (cambio 1 del protocolo, con el modelo grande):
  rec8   el ojo MESO (el de arriba)
  mac4   (2, 32, 4, 4, D) fp16   ojo MACRO: la imagen ENTERA, 64 cuadros repartidos por TODO el clip
  mic8   (2, 32, 8, 8, D) fp16   ojo MICRO: [a - 1,5 s, a + 0,5 s] (~los fps del video) con ZOOM
                                 MOVIL que sigue y contiene las manos adelantadas (Sapiens);
                                 `caja_mic` (64, 4) la caja de cada cuadro
  media_<salida> (2, D) fp32, y idx, idx_pre, caja, caja_pre, n_pre_reales, ancla, via_ancla, fps, T
  vista 0 = la imagen, vista 1 = su ESPEJO horizontal (la sonda impone la antisimetria con las dos)
El 4x4 de `rec8` se saca exacto promediando 2x2, asi que `rec8` contiene a `tok` de `vjepa.py`.

    python vjepa_variantes.py --conjunto anotados                     rec8, ent8, pre4 (ViT-L)
    python vjepa_variantes.py --modelo vjepa2-vitg-fpc64-384 --salidas rec8,mac4,mic8 --destino D:/...
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
SALIDAS = ("rec8", "ent8", "pre4", "rec4", "mac4", "mic8", "mic4")
"""`mic4` (2026-10-13, para la corrida): el ojo micro promediado 4x4, que es lo que midio la sonda de
G2 (`rec4 + mic4`, sacado de `mic8` por 2x2). Ocupa la cuarta parte que `mic8`."""
MIC_ANTES_S, MIC_DESPUES_S = 1.5, 0.5
"""El ojo micro: 2 s alrededor del tocado en 64 cuadros, ~32/s: a 30 fps casi todos los cuadros del
video (60 de 64), a 25 fps alguno repetido, a 50 fps uno de cada ~1,6."""
MIC_LADO_MIN = 160
MANOS_KP = (21, 42)
"""Las dos raices de mano de Sapiens 2 (`cono_lib.MANOS`)."""


def ventana_pre(a, fps):
    """64 cuadros de [a - 8,5 s, a - 4,0 s] a la densidad de la ventana principal. Los que caen antes
    del cuadro 0 repiten el 0, y `n_pre_reales` cuenta los que no."""
    ini = a - int(round(PRE_DESDE_S * fps))
    fin = a - int(round(VJ.ANTES_S * fps))
    idx = np.round(np.linspace(ini, fin, VJ.N_CUADROS)).astype("int64")
    return np.clip(idx, 0, None).astype("int32"), int((idx >= 0).sum())


def ventana_mic(a, fps, T):
    ini = max(0, a - int(round(MIC_ANTES_S * fps)))
    fin = min(T - 1, a + int(round(MIC_DESPUES_S * fps)))
    if fin <= ini:
        ini, fin = 0, T - 1
    return np.round(np.linspace(ini, fin, VJ.N_CUADROS)).astype("int32")


def caja_manos(KP, SC, B, idx, W, H):
    """El ZOOM del ojo micro: una caja por cuadro que SIGUE el cruce de las manos y lo contiene.

    POR QUE ASI (medido en 6 clips antes de fijarlo, `vjepa_ojos_2026-10-09/`):
    - una caja FIJA de las manos en los 2 s salia de 440-720 px, tan grande como el recorte de la
      pareja, porque en 2 s las manos recorren metros: no era un zoom;
    - una caja movil de lado fijo centrada entre las manos enseniaba el HUECO vacio entre los dos
      cuando estan lejos (la distancia entre las manos adelantadas va de 750 px a 66 px).
    Asi que la caja contiene las dos manos y se cierra sobre ellas cuando se acercan.

    Por cuadro y tirador, de sus dos raices de mano (confianza >= 0,3, dentro de la imagen y de su caja
    ampliada un 25 %), la de x mas cerca del centro del rival: el criterio (a) de `cono_lib.mano_armada`,
    por cuadro y sin decidir cual es la armada, porque aqui solo hace falta saber DONDE esta el cruce.
    Con las dos manos: centro = su punto medio; ancho = max(1,3 |dx|, 0,9 h), alto = max(1,3 |dy|,
    0,9 h), con h el alto mediano del tirador en la ventana, y razon ancho/alto entre 1 y 2 como el
    recorte de la pareja. Los cuadros sin las dos manos se interpolan; mediana movil de 7 en las cuatro
    magnitudes; nunca mas grande que la imagen.
    QUE DEVUELVE: (cajas (64, 4) en pixeles, cuadros con las dos manos, via `manos` / `respaldo`)."""
    n = len(idx)
    g = np.full((n, 4), np.nan)
    altos = []
    for i, t in enumerate(idx):
        bt = B[t]
        if not np.isfinite(bt).all():
            continue
        altos.extend(((bt[:, 3] - bt[:, 1]) * H).tolist())
        ps = []
        for j in (0, 1):
            x1, y1, x2, y2 = bt[j] * [W, H, W, H]
            ax, ay = 0.25 * (x2 - x1), 0.25 * (y2 - y1)
            rx = 0.5 * (bt[1 - j][0] + bt[1 - j][2]) * W
            mejor = None
            for k in MANOS_KP:
                x, y = KP[t, j, k]
                if not (SC[t, j, k] >= 0.3 and 0 <= x < W and 0 <= y < H
                        and x1 - ax <= x <= x2 + ax and y1 - ay <= y <= y2 + ay):
                    continue
                if mejor is None or abs(x - rx) < abs(mejor[0] - rx):
                    mejor = (float(x), float(y))
            if mejor:
                ps.append(mejor)
        if len(ps) == 2:
            (xa, ya), (xb, yb) = ps
            g[i] = [0.5 * (xa + xb), 0.5 * (ya + yb), abs(xa - xb), abs(ya - yb)]
    h = float(np.median(altos)) if altos else float(MIC_LADO_MIN)
    ok = np.isfinite(g[:, 0])
    via = "manos"
    if not ok.any():
        b = B[idx].reshape(-1, 4)
        b = b[np.isfinite(b).all(axis=1)]
        if not len(b):
            return np.tile(np.array([0, 0, W, H], dtype="float32"), (n, 1)), 0, "imagen"
        x1, x2 = b[:, 0].min() * W, b[:, 2].max() * W
        y1, y2 = b[:, 1].min() * H, b[:, 3].max() * H
        g[:] = [0.5 * (x1 + x2), y1 + 0.35 * (y2 - y1), (x2 - x1) / 1.3, 0.0]
        via = "respaldo"
    else:
        pos = np.arange(n)
        for c in range(4):
            g[:, c] = np.interp(pos, pos[ok], g[ok, c])
    pad = np.pad(g, ((3, 3), (0, 0)), mode="edge")
    g = np.stack([np.median(pad[i:i + 7], axis=0) for i in range(n)])
    an = np.maximum(1.3 * g[:, 2], 0.9 * h)
    al = np.maximum(1.3 * g[:, 3], 0.9 * h)
    al = np.where(an / al > 2.0, an / 2.0, al)
    an = np.where(an / al < 1.0, al, an)
    an, al = np.minimum(an, W), np.minimum(al, H)
    x1 = np.clip(g[:, 0] - an / 2, 0.0, W - an)
    y1 = np.clip(g[:, 1] - al / 2, 0.0, H - al)
    cajas = np.stack([x1, y1, x1 + an, y1 + al], axis=1).astype("float32")
    return cajas, int(ok.sum()), via


VISTA = {"rec8": ("rec", 8), "rec4": ("rec", 4), "ent8": ("ent", 8), "pre4": ("pre", 4),
         "mac4": ("mac", 4), "mic8": ("mic", 8), "mic4": ("mic", 4)}
"""salida -> (vista, promedio k x k). Una vista se pasa UNA vez por el modelo aunque salgan dos
salidas de ella (`rec8` y `rec4`)."""


def prepara(clip, salidas, lado):
    """CPU: como `vjepa.prepara`, pero decodifica UNA vez y saca todas las vistas pedidas. Cada vista
    es (cuadros, caja); caja None = la imagen entera."""
    import cv2
    p = RU.TRABAJO / ("poses_%s.npz" % clip)
    if not p.is_file():
        return None, "sin poses"
    with np.load(p, allow_pickle=True) as z:
        B, mask = z["B"], z["mask"]
        W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
        fps = float(z["fps"]) if "fps" in z.files else 0.0
        con_mic = any(VISTA[s][0] == "mic" for s in salidas)
        KP = z["KP"] if con_mic else None
        SC = z["SC"] if con_mic else None
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
    idx_mac = np.round(np.linspace(0, T - 1, VJ.N_CUADROS)).astype("int32")
    idx_mic = ventana_mic(a, fps, T)
    caja_mic, n_manos, via_mic = (caja_manos(KP, SC, B, idx_mic, W, H) if KP is not None
                                  else (None, 0, ""))
    caja_de_mic = ({int(t): caja_mic[i] for i, t in reversed(list(enumerate(idx_mic)))}
                   if caja_mic is not None else None)
    """Cuadro -> su caja del ojo micro; si un cuadro sale dos veces (videos de 25 fps), la primera."""
    defs = {"rec": (idx, caja), "ent": (idx, None), "pre": (idx_pre, caja_pre),
            "mac": (idx_mac, None), "mic": (idx_mic, caja_de_mic)}
    vistas = sorted({VISTA[s][0] for s in salidas})
    quiero = {v: set(int(i) for i in defs[v][0]) for v in vistas}
    tope = max(max(q) for q in quiero.values())

    def corta(im, c, cubico=False):
        if c is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in c]
            im = im[y1:y2, x1:x2]
        interp = cv2.INTER_CUBIC if (cubico and im.shape[0] < lado) else cv2.INTER_AREA
        r = cv2.resize(im, (lado, lado), interpolation=interp)
        return cv2.cvtColor(r, cv2.COLOR_BGR2RGB)

    """INTER_AREA en todo, como `vjepa.py`, para que `rec` sea exactamente su recorte. Solo el ZOOM del
    ojo micro, que casi siempre AMPLIA, va con INTER_CUBIC: INTER_AREA al ampliar es casi el vecino mas
    cercano."""
    cuadros = {v: {} for v in vistas}
    k = 0
    while k <= tope:
        ok, im = cap.read()
        if not ok:
            break
        for v in vistas:
            if k in quiero[v]:
                c = defs[v][1]
                cuadros[v][k] = corta(im, c[k] if isinstance(c, dict) else c, cubico=(v == "mic"))
        k += 1
    cap.release()
    if not any(cuadros.values()):
        return None, "video ilegible"
    d = {"clip": clip, "idx": idx, "idx_pre": idx_pre, "idx_mic": idx_mic, "caja": caja,
         "caja_pre": caja_pre, "caja_mic": caja_mic if caja_mic is not None else np.zeros((0, 4), "float32"),
         "n_manos": n_manos, "via_mic": via_mic, "n_pre_reales": n_pre, "ancla": a, "via": via,
         "fps": fps, "T": T, "s_lee": time.perf_counter() - t0, "pilas": {}, "faltan": {}}
    for v in vistas:
        c = cuadros[v]
        if not c:
            return None, "video ilegible (%s)" % v
        ultimo = c[max(c)]
        d["pilas"][v] = np.stack([c.get(int(i), ultimo) for i in defs[v][0]])
        d["faltan"][v] = int(sum(int(i) not in c for i in defs[v][0]))
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
    for v, pila in d["pilas"].items():
        h = embebe(mod, pila, lado)
        out["media_" + v] = h.float().mean(dim=1)
        for s in salidas:
            if VISTA[s][0] == v:
                out[s] = _pool(h, lado, VISTA[s][1])
        del h
    torch.cuda.synchronize()
    res = {k: (v.cpu().numpy().astype("float32") if k.startswith("media")
               else v.half().cpu().numpy()) for k, v in out.items()}
    return res, time.perf_counter() - t0


def guarda(d, res, s_gpu, destino, modelo):
    destino.mkdir(parents=True, exist_ok=True)
    f = destino / ("vjv_%s.npz" % d["clip"])
    tmp = destino / ("vjv_%s.part.npz" % d["clip"])
    faltan = {"faltan_" + v: np.int32(n) for v, n in d["faltan"].items()}
    np.savez(tmp, idx=d["idx"], idx_pre=d["idx_pre"], idx_mic=d["idx_mic"], caja=d["caja"],
             caja_pre=d["caja_pre"], caja_mic=d["caja_mic"], n_manos=np.int32(d["n_manos"]),
             via_mic=d["via_mic"], n_pre_reales=np.int32(d["n_pre_reales"]),
             ancla=np.int32(d["ancla"]), via_ancla=d["via"], fps=np.float64(d["fps"]),
             T=np.int32(d["T"]), modelo=modelo, s_gpu=np.float64(s_gpu),
             s_lee=np.float64(d["s_lee"]), **faltan, **res)
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

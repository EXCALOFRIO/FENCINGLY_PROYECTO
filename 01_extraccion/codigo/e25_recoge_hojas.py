"""ETAPA E, paso 25: RECOGER LAS HOJAS QUE FALTAN, 9 asaltos nuevos de 1080p.

`e23` midio que el campo de probabilidad del cono **no se puede estimar con 31 hojas**: el IC del p90
mide 13,5 grados contra los 5 del margen del proyecto, y **24 de las 31 venian de UN asalto** que
encima no esta en `corpus_etiquetado.txt`. La misma cuenta daba la via: **n ~ 226 hojas**, o sea
~531 recortes de 1080p, del orden de 9 clips. Esto es eso, y nada mas que eso: 9 clips y paro.

COMO SE ELIGEN LOS CLIPS, y el criterio va antes que la ejecucion:
  1. de `corpus_etiquetado.txt`, que es la lista de florete verificada. `AdorableFitting...` no
     estaba y por eso su asalto no valia como muestra del corpus.
  2. a **1080p**, porque a 720p esta medido que lo que se detecta no es la hoja (60,9 grados contra
     el eje, peor que el azar de 45).
  3. con **deteccion ya cacheada** en `det_<clip>.pkl`, para no volver a pasar YOLO (que ademas vive
     en el otro interprete y los dos no coexisten).
  4. **ASALTOS DISTINTOS**, que es la leccion de `e23`. Se excluyen los clips cuyas poses ya existen
     porque sus asaltos ya estan en la muestra vieja, y se comprueba a ojo que los 9 no repiten
     marcador ni nombres. Recordatorio de por que hace falta: `WelloffForcefulIndri` e
     `IdealFrigidJerboa` son **el mismo asalto** y en 22.2 se colaron como dos.

COMO SE ELIGEN LOS CUADROS, y aqui hay un requisito que choca con el otro:
  - **repartidos por el clip**, porque la tasa de deteccion es del 42,6 % fuera de la ventana de
    accion y del 3,0 % dentro. Muestrear la accion seria gastar GPU para no ver nada.
  - pero tambien hacen falta **CUADROS CONSECUTIVOS**, porque el temblor de una hoja vista solo se
    mide con pares contiguos, y `e23` solo tenia 8 pares.
  La solucion es **rafagas**: 12 rachas de 5 cuadros seguidos, repartidas por el clip. Da 60 cuadros
  con 12 posiciones distintas y hasta 48 pares consecutivos por clip.

LOS PARAMETROS DE DETECCION SON LOS MISMOS QUE LOS DE `e5`, sin tocar ni uno (`PCTL` 99,5, `DILATA`
8, `CERCA_ROOT` 0,12, `MIN_LARGO` 0,25, y el mismo `recorte`). Es imprescindible: si cambiaran, las
hojas nuevas no serian comparables con las 31 viejas y la muestra agregada no valdria.

Y DOS COSAS MAS QUE SE APROVECHAN, porque el recorte ya esta en memoria y cuestan 0 ms de GPU:
  - el **CONTRASTE local** en la region donde se busca la hoja, para intentar explicar por que se ve
    en unos cuadros y no en otros, ahora que el desenfoque de movimiento **no lo explica** (`e23` M4
    midio 0,96x, sin diferencia).
  - el **ESCORZO** del antebrazo, que es la otra hipotesis y es geometrica: una hoja perpendicular a
    la camara proyecta mas largo que una de frente, asi que deberia verse mejor.
  - y la marca `dudosa` de `cono_lib.mano_armada`, que el otro agente acaba de adoptar por pie
    firme, para poder decir si la tasa de deteccion **sube en los tiradores firmes**.

SALIDA: `cono_poses_<clip>.npz` con las poses SOLO de los cuadros muestreados (el resto NaN) y
`cono_hoja_<clip>.json` con el formato exacto de `e5`, para que `e23` los agregue sin cambiar nada.
"""
from __future__ import annotations

import json
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
import a3_selector as SEL
import cono3d_lib as C3
import cono_lib as CL
import e7_pinta_cono as E7

PRUEBAS = CL.PRUEBAS
HUB = CACHE / "hub"
N_RAFAGAS = 12
LARGO_RAFAGA = 5
MARGEN_PIE = 0.06
N_PTS = 308
VRAM_MIN = 4.0
CLIPS = ["BabyishIdioticCockerspaniel", "DemandingMindlessDeviltasmanian",
         "GentleBlueIndianspinyloach", "LiveDecimalLadybird", "MessySnappyArmyworm",
         "PowerfulAllLarva", "SeveralSkinnyGenet", "proudDungBeetleBoglenTerrier",
         "uptightSlothEveningBat"]
VIDEOS = pathlib.Path("D:/FENCINGLY_VIDEOS_RAW")


def espera_vram(gb=VRAM_MIN, tope=600.0):
    """Espera hasta que haya `gb` GB libres. Copiado en espiritu de `capas_limpias.espera_vram`.

    No reserva nada: solo mira. Si se agota la espera se intenta igual, porque quedarse esperando
    para siempre tampoco es un resultado, pero se dice cuanto se espero.
    """
    if not torch.cuda.is_available():
        return float("nan")
    t0 = time.perf_counter()
    libre = torch.cuda.mem_get_info()[0] / 2 ** 30
    while libre < gb and time.perf_counter() - t0 < tope:
        print("    esperando VRAM: hacen falta %.1f GB y hay %.1f libres (%.0f s)"
              % (gb, libre, time.perf_counter() - t0))
        time.sleep(15.0)
        libre = torch.cuda.mem_get_info()[0] / 2 ** 30
    print("    VRAM: %.1f GB libres de %.1f, tras esperar %.0f s"
          % (libre, torch.cuda.mem_get_info()[1] / 2 ** 30, time.perf_counter() - t0))
    return libre


def localiza_video(clip):
    for p in VIDEOS.rglob("%s.mp4" % clip):
        return str(p)
    return None


def pareja(clip):
    """Los dos tiradores, desde el cache de detecciones. Mismo camino que `e1_poses_cono`."""
    cajas, cuadros, W, H, T = C3.carga_det(clip)
    tr = SEL.construye_trayectorias(cajas, cuadros, W, H, T)
    cand = [{**SEL.puntua(t, W, H, T), "t": t} for t in tr]
    cand.sort(key=lambda r: -r["puntos"])
    if not cand:
        return None
    for r in cand:
        c = np.asarray(r["t"]["cajas"], dtype="float64")
        r["pie"] = float(np.median(c[:, 3])) / H
    ref = cand[0]["pie"]
    pasan = [r for r in cand if r["pie"] >= ref - MARGEN_PIE]
    res = SEL.elige_dos([r["t"] for r in pasan], W, H, T)
    B, presente = SEL.a_matrices(res["elegidas"], cajas, T, W, H)
    return {"B": B, "presente": presente, "W": W, "H": H, "T": T, "motivo": res["motivo"],
            "n_tray": len(cand), "n_pasan": len(pasan)}


def elige_cuadros(presente, T):
    """12 rafagas de 5 cuadros consecutivos, repartidas entre los cuadros con los DOS presentes."""
    dos = presente.sum(axis=1) == 2
    validos = [k for k in range(T - LARGO_RAFAGA)
               if dos[k:k + LARGO_RAFAGA].all()]
    if not validos:
        return []
    if len(validos) <= N_RAFAGAS:
        ini = validos
    else:
        idx = np.linspace(0, len(validos) - 1, N_RAFAGAS).round().astype(int)
        ini = sorted(set(validos[i] for i in idx))
    ks = sorted(set(k for a in ini for k in range(a, a + LARGO_RAFAGA)))
    return ks


def poses(video, ks, B, W, H, T):
    from transformers import Sapiens2ForPoseEstimation, Sapiens2ImageProcessor
    d = HUB / "models--facebook--sapiens2-pose-0.4b" / "snapshots"
    s = next(d.iterdir())
    proc = Sapiens2ImageProcessor.from_pretrained(str(s))
    proc.size = {"height": 384, "width": 288}
    mod = Sapiens2ForPoseEstimation.from_pretrained(str(s), dtype=torch.float16).to("cuda").eval()
    KP = np.full((T, 2, N_PTS, 2), np.nan, dtype="float32")
    SC = np.full((T, 2, N_PTS), np.nan, dtype="float32")
    cap = cv2.VideoCapture(video)
    ims = {}
    prev = None
    for k in ks:
        if prev is None or k != prev + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, im = cap.read()
        prev = k
        if ok:
            ims[k] = im
    cap.release()
    for k, im in ims.items():
        cjs, lados = [], []
        for lado in (0, 1):
            b = B[k, lado]
            if not (np.isfinite(b).all() and b[2] > b[0] and b[3] > b[1]):
                continue
            x, y = float(b[0] * W), float(b[1] * H)
            cjs.append([[x, y, float(b[2] * W) - x, float(b[3] * H) - y]])
            lados.append(lado)
        if not cjs:
            continue
        rgb = np.ascontiguousarray(im[:, :, ::-1])
        with torch.inference_mode():
            e = proc(images=[rgb] * len(cjs), boxes=cjs, return_tensors="pt")
            e = {kk: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda"))
                 for kk, v in e.items()}
            r = proc.post_process_pose_estimation(mod(**e), boxes=cjs, threshold=None)
        for j, lado in enumerate(lados):
            rr = r[j][0] if isinstance(r[j], list) else r[j]
            kp = np.asarray(rr["keypoints"].cpu(), dtype="float32")
            sc = np.asarray(rr["scores"].cpu(), dtype="float32")
            n = min(len(kp), N_PTS)
            KP[k, lado, :n] = kp[:n]
            SC[k, lado, :n] = sc[:n]
    del mod
    torch.cuda.empty_cache()
    return KP, SC, ims


def rasgos(sub, seg, vert, cx0, cy0, R):
    """CONTRASTE local en la region donde se busca la hoja. Cuesta 0 ms de GPU.

    Se mide sobre los mismos pixeles que ve el detector: dentro del disco de radio R, fuera del
    cuerpo dilatado. Se usa el recorrido p90-p10 del gris normalizado a 1, que es robusto a un
    reflejo aislado, en vez de la desviacion tipica.
    """
    cuerpo = cv2.dilate((seg != 0).astype("uint8"),
                        np.ones((2 * E7.DILATA + 1,) * 2, np.uint8)) > 0
    hh, ww = seg.shape[:2]
    GY, GX = np.mgrid[0:hh, 0:ww]
    disco = np.hypot(GX + cx0 - vert[0], GY + cy0 - vert[1]) <= R
    m = disco & (~cuerpo)
    if m.sum() < 200:
        return float("nan"), int(m.sum())
    g = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY).astype("float32")[m]
    return float(np.percentile(g, 90) - np.percentile(g, 10)) / 255.0, int(m.sum())


def una_clip(clip):
    print("=" * 104)
    print("RECOGIENDO HOJAS  ·  %s" % clip)
    print("=" * 104)
    video = localiza_video(clip)
    if video is None:
        print("  sin video en %s" % VIDEOS)
        return None
    pa = pareja(clip)
    if pa is None:
        print("  sin pareja decidible")
        return None
    B, pres, W, H, T = pa["B"], pa["presente"], pa["W"], pa["H"], pa["T"]
    print("  %dx%d  ·  %d cuadros  ·  trayectorias %d, pasan %d  ·  %s"
          % (W, H, T, pa["n_tray"], pa["n_pasan"], pa["motivo"][:60]))
    ks = elige_cuadros(pres, T)
    if len(ks) < 10:
        print("  solo %d cuadros con los dos presentes en rafagas: se salta" % len(ks))
        return None
    print("  %d cuadros en %d rafagas de %d" % (len(ks), len(ks) // LARGO_RAFAGA, LARGO_RAFAGA))
    espera_vram()
    t0 = time.perf_counter()
    KP, SC, ims = poses(video, ks, B, W, H, T)
    print("  pose ... %.0f ms por cuadro de video" % (1000 * (time.perf_counter() - t0) / len(ks)))
    marca = np.zeros((T, 2), dtype=bool)
    np.savez_compressed(PRUEBAS / ("cono_poses_%s.npz" % clip), res=np.asarray((384, 288)),
                        kp=KP, sc=SC, B=B, presente=pres, marca=marca, W=W, H=H, T=T,
                        fps=25.0, video=video, ms_pose=0.0, cuadros_muestreados=np.asarray(ks))

    geo, duda = {}, {}
    for lado in (0, 1):
        mano, d = CL.mano_armada(KP, SC, B, W, H, lado)
        if mano is None:
            print("  lado %d sin mano armada: se salta el clip" % lado)
            return None
        vert, u, ok, _lg = C3.eje_brazo_2d(KP, SC, lado, mano, "codo")
        _pk, pxm = C3.escala_px_m_de_pie(B, lado, H)
        M = CL.MANOS[mano]
        geo[lado] = {"vert": vert, "u": u, "ok": ok, "pxm": pxm, "mano": mano,
                     "codo": KP[:, lado, M["codo"]].astype("float64"),
                     "alto": (B[:, lado, 3] - B[:, lado, 1]) * H}
        duda[lado] = {"mano": mano, "dudosa": bool(d.get("dudosa", True)),
                      "firme_pie": bool(d.get("firme_pie", False)),
                      "frac_pie": float(d.get("frac_pie", float("nan"))), "por": d.get("por")}
        print("  lado %d: mano %s por %s  ·  dudosa %s  ·  frac_pie %.2f"
              % (lado, mano, duda[lado]["por"], duda[lado]["dudosa"], duda[lado]["frac_pie"]))

    filas, t0, n_det = [], time.perf_counter(), 0
    for k in ks:
        im = ims.get(k)
        if im is None:
            continue
        for lado in (0, 1):
            g = geo[lado]
            if not (g["ok"][k] and np.isfinite(B[k, lado]).all()):
                continue
            vert, u = g["vert"][k], g["u"][k]
            L = C3.ALCANCE_M * g["pxm"]
            (cx0, cy0, cx1, cy1), R = E7.recorte(B[k, lado], B[k, 1 - lado], vert, u, L, W, H)
            if cx1 - cx0 < 64 or cy1 - cy0 < 64 or R < 24:
                continue
            sub = np.ascontiguousarray(im[cy0:cy1, cx0:cx1, ::-1])
            alfa, seg = E7.infiere(sub)
            m, ln = E7.detecta(alfa, seg, vert, cx0, cy0, R)
            con, n_px_reg = rasgos(sub, seg, vert, cx0, cy0, R)
            """ESCORZO: largo proyectado del antebrazo sobre el alto de la caja. Corto = el brazo
            apunta a la camara, y entonces la hoja tambien y proyecta menos."""
            ante = float(np.linalg.norm(vert - g["codo"][k]))
            esc = ante / max(float(g["alto"][k]), 1.0)
            det = {}
            if ln is not None:
                n_det += 1
                lg, p1, p2 = ln
                ang = float(np.degrees(np.arctan2((p2 - p1)[1], (p2 - p1)[0])))
                a_eje = float(np.degrees(np.arctan2(u[1], u[0])))
                det = {"p99.5": {"linea": {
                    "largo": float(lg), "ang": ang, "p1": [float(p1[0]), float(p1[1])],
                    "p2": [float(p2[0]), float(p2[1])],
                    "dif_eje": float(abs((ang - a_eje + 90.0) % 180.0 - 90.0))}}}
            filas.append({"clip": clip, "cuadro": int(k), "lado": int(lado),
                          "R": float(R), "recorte": [cx0, cy0, cx1, cy1],
                          "alfa_recorte": float(np.nanmean(alfa)),
                          "contaminado": False, "det": det,
                          "contraste": con, "n_px_region": n_px_reg, "escorzo": esc,
                          "dudosa": duda[lado]["dudosa"], "firme_pie": duda[lado]["firme_pie"]})
    ms = 1000.0 * (time.perf_counter() - t0) / max(len(filas), 1)
    print("  %d recortes  ·  %d con recta = **%.1f %%**  ·  %.0f ms por recorte"
          % (len(filas), n_det, 100.0 * n_det / max(len(filas), 1), ms))
    (PRUEBAS / ("cono_hoja_%s.json" % clip)).write_text(
        json.dumps(filas, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    return {"clip": clip, "W": W, "H": H, "T": T, "n_cuadros": len(ks), "n_recortes": len(filas),
            "n_det": n_det, "tasa": n_det / max(len(filas), 1), "ms_recorte": ms,
            "lados": duda}


def main():
    corpus = set(x.strip() for x in
                 (C3.SCRATCH / "corpus_etiquetado.txt").read_text(encoding="utf-8").split("\n")
                 if x.strip())
    pedidos = sys.argv[1:] or CLIPS
    print("CLIPS PEDIDOS: %d  ·  todos en corpus_etiquetado: %s" % (
        len(pedidos), all(c in corpus for c in pedidos)))
    for c in pedidos:
        if c not in corpus:
            print("  AVISO: %s NO esta en corpus_etiquetado.txt" % c)
    out = {}
    for c in pedidos:
        try:
            r = una_clip(c)
        except Exception as e:
            print("  %s FALLO: %s: %s" % (c, type(e).__name__, e))
            r = None
        if r is not None:
            out[c] = r
        (PRUEBAS / "cono_e25_recogida.json").write_text(
            json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print()
    print("=" * 104)
    tot_r = sum(r["n_recortes"] for r in out.values())
    tot_d = sum(r["n_det"] for r in out.values())
    print("RECOGIDA: %d clips  ·  %d recortes  ·  **%d rectas = %.1f %%**" % (
        len(out), tot_r, tot_d, 100.0 * tot_d / max(tot_r, 1)))
    print("guardado %s" % (PRUEBAS / "cono_e25_recogida.json"))


if __name__ == "__main__":
    main()

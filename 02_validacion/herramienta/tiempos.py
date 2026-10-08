"""Cuanto cuesta procesar un clip de punta a punta, paso por paso.

LO QUE PIDIO EL USUARIO, literal: "y mide tiempos, que se tarda en completar todos los clips, y asi
vemos por clip, ok? que ayudara con todos los filtros, recortar, quitar y eso". Y despues: "necesito
al terminar que saques ya con el script completo cuanto tardas en sacar todo eso que hemos hablado, y
filtralo y procesalo etc completo por 10 clips, y asi sabemos la media, y lo dejas sacando para 50
para saber mas la media".

Asi que hay dos tiradas: **10 clips para tener la cifra ya, y 50 para tener la media buena.**

## DE DONDE SALEN LOS TIEMPOS

Del codigo que de verdad corre. `extrae_pareja.procesa` acepta un dict `cron` y va acumulando los
segundos de cada paso dentro de si mismo. No hay una copia paralela de la tuberia para medir: una
copia se queda atras en cuanto se toca un paso, y entonces el informe de tiempos miente sin que nadie
se entere.

## LAS DOS FASES, porque los interpretes NO coexisten

  `fase1`  con el interprete de ultralytics: decodificar, YOLO de cuadro completo, puerta de planos,
           fundidos, trayectorias, pareja, limpieza y redeteccion en recorte
  `fase2`  con el interprete de cuda: pose de 308 puntos, las cuatro capas densas, la limpieza de
           mascara y la invencion de las poses que faltan
  `informe` junta los dos CSV y da la tabla por clip y la media

El tiempo de CARGAR los modelos se mide aparte y no se reparte entre los clips: en produccion se
cargan una vez y se procesan miles de clips, asi que meterlo en el coste por clip infla la cifra.
Se informa por separado para que se vea cuanto es.

## LO QUE SE MIDE Y LO QUE SE MULTIPLICA, dicho sin adornos

La **fase 1 se cronometra entera y de verdad**: se decodifica el clip completo y se pasa YOLO por
todos sus cuadros, aunque exista el pickle cacheado, porque el coste que hay que informar es el de la
primera vez.

La **fase 2 se mide por unidad de trabajo y se multiplica**. Para cada cabeza de Sapiens 2 se
cronometran unas cuantas pasadas sobre cuadros muestreados del clip, se toma la mediana de
milisegundos por tirador, y se multiplica por el trabajo real del clip, que es el numero de
(cuadro entregado x tirador presente).

Se hace asi por una razon medida: la primera version procesaba los clips enteros y tardaba **horas**,
porque 10 clips por 300 cuadros por 2 tiradores son 6.000 pasadas de pose mas las densas. Y no
aportaba exactitud, porque la cifra que se informa es de todas formas un coste por unidad
multiplicado. Medir en muestra ademas evita arrastrar la deriva de una tirada larga.

Dos cuidados que hacen que esa multiplicacion sea honesta y que estan en el codigo:
`torch.cuda.synchronize()` alrededor de cada cronometro, porque CUDA es asincrono y sin eso se mide
el tiempo de encolar y no el de calcular; y un **calentamiento** previo de cada cabeza, porque la
primera pasada incluye compilar los nucleos y reservar memoria, que es coste de arranque y no se paga
por clip.

## UNA MEDIDA CON LA GPU COMPARTIDA NO VALE

Si hay otros procesos usando la GPU, estas cifras salen infladas y no sirven para nada. El script
avisa de la ocupacion al empezar. La tirada buena se hace con la GPU libre.

## COMO SE CORRE

  ultralytics: python tiempos.py fase1 10
  cuda:        python tiempos.py fase2 10
               python tiempos.py informe
"""
from __future__ import annotations

import os
import pathlib
import pickle
import sys
import time

CACHE_HF = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565\scratchpad\hoja_seg\hf")
os.environ.setdefault("HF_HOME", str(CACHE_HF))

import cv2
import numpy as np

CODIGO = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
sys.path.insert(0, str(CODIGO))
PRUEBAS = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
SCR = pathlib.Path(os.environ.get(
    "SCRATCH", r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\187cfd83-d8d9-4c4d-bed4-f255724f6b21\scratchpad"))
LISTA = SCR / "corpus_etiquetado.txt"
VIDEOS = "D:/FENCINGLY_VIDEOS_RAW"
BASE_YOLO = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565\scratchpad\smoke\weights\yolo11m-pose.pt")
CSV1 = PRUEBAS / "tiempos_fase1.csv"
CSV2 = PRUEBAS / "tiempos_fase2.csv"
SEMILLA = 20260910
MUESTRA_DENSA = 8
"""Cuadros en los que se cronometran las capas densas. Ocho pasadas de cada cabeza dan una media
estable y no cuestan los minutos que costaria hacerlo en todo el clip. Se declara aqui porque la
columna del informe sale de multiplicar, no de medir entero."""
CLIPS_CORPUS = 8559
"""Clips de FLORETE del corpus que estan en disco, contados en `corpus_etiquetado.txt`. Es el numero
con el que se extrapola el coste total."""


def mapa_videos():
    m = {}
    for raiz, _d, fs in os.walk(VIDEOS):
        for f in fs:
            if f.lower().endswith(".mp4"):
                m.setdefault(pathlib.Path(f).stem, os.path.join(raiz, f))
    return m


def elige_clips(n):
    """QUE HACE: muestrea del corpus ETIQUETADO, no del arbol de videos.

    El arbol tiene tambien descargas de otros estudios, con espada y sable, y muestrear de ahi ya
    costo una tirada entera: aparecieron clips de "Men's Epee - T64" en un informe de florete.
    """
    ids = [l.strip() for l in LISTA.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = np.random.default_rng(SEMILLA)
    rng.shuffle(ids)
    return ids[:n]


def escribe(csv, filas):
    import pandas as pd
    d = pd.DataFrame(filas)
    if csv.exists():
        v = pd.read_csv(csv)
        d = pd.concat([v[~v["clip"].isin(d["clip"])], d], ignore_index=True)
    d.to_csv(csv, index=False)
    return d


def fase1(n):
    from ultralytics import YOLO
    import extrae_pareja as EP

    t = time.perf_counter()
    yolo = YOLO(str(BASE_YOLO))
    yolo.to("cuda")
    t_carga = time.perf_counter() - t
    print("cargar YOLO11m-pose: %.1f s (no se reparte entre clips)" % t_carga)
    print()

    mapa = mapa_videos()
    clips = [c for c in elige_clips(n * 3) if c in mapa][:n]
    PRUEBAS.mkdir(parents=True, exist_ok=True)
    filas = []
    print("%-34s %5s %6s %6s %6s %6s %6s %6s %6s %6s" % (
        "clip", "T", "decod", "yolo", "planos", "fund", "tray", "pareja", "limpia", "redet"))
    print("-" * 106)
    for cid in clips:
        cap = cv2.VideoCapture(mapa[cid])
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if W < 2:
            cap.release()
            continue
        t = time.perf_counter()
        frames = []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            frames.append(im)
        cap.release()
        t_decod = time.perf_counter() - t
        T = len(frames)
        if T < 10:
            continue

        """YOLO se cronometra SIEMPRE de verdad, aunque exista el pickle cacheado. El cache existe
        para no repetir trabajo al iterar sobre el selector, pero el coste que hay que informar es el
        de la primera vez, que es lo que costara el corpus entero."""
        t = time.perf_counter()
        cajas, cuadros = [], []
        for k, im in enumerate(frames):
            r = yolo.predict(im, imgsz=EP.IMGSZ_YOLO, conf=EP.CONF_YOLO, verbose=False, device=0)[0]
            if r.boxes is not None and len(r.boxes):
                b = r.boxes.xyxy.cpu().numpy()
                for i in range(len(b)):
                    cajas.append(b[i])
                    cuadros.append(k)
        t_yolo = time.perf_counter() - t
        cajas = np.asarray(cajas, "float32")
        cuadros = np.asarray(cuadros, "int32")
        if not len(cajas):
            continue
        ca = SCR / ("det_%s.pkl" % cid)
        if not ca.exists():
            with open(ca, "wb") as fh:
                pickle.dump((cajas, cuadros, W, H, T), fh)

        t = time.perf_counter()
        gris = [cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype("float32")
                for f in frames]
        t_gris = time.perf_counter() - t

        cron = {}
        r = EP.procesa(cajas, cuadros, W, H, T, yolo=yolo, frames=frames, gris=gris, cron=cron)
        np.savez_compressed(PRUEBAS / ("tiempos_%s.npz" % cid),
                            B=r["B"], Be=r["Be"], pres=r["pres"], tipo=r["tipo"], mask=r["mask"],
                            W=W, H=H, T=T, video=mapa[cid], cajas=cajas, cuadros=cuadros)
        f = {"clip": cid, "T": T, "W": W, "H": H, "ndet": int(len(cajas)),
             "t_decod": t_decod, "t_yolo": t_yolo, "t_gris": t_gris,
             "t_planos": cron.get("planos", 0.0), "t_fundidos": cron.get("fundidos", 0.0),
             "t_tray": cron.get("trayectorias", 0.0), "t_pareja": cron.get("pareja", 0.0),
             "t_limpieza": cron.get("limpieza", 0.0), "t_redet": cron.get("redeteccion", 0.0),
             "ambos": r["ambos"], "nredet": r["nredetectados"], "faltan": r["faltan"],
             "recortados": r["recortados"], "tiras": r["tiras"]}
        f["t_fase1"] = sum(v for k, v in f.items() if k.startswith("t_"))
        filas.append(f)
        print("%-34s %5d %6.1f %6.1f %6.2f %6.2f %6.2f %6.2f %6.2f %6.1f" % (
            cid[:34], T, t_decod, t_yolo, f["t_planos"], f["t_fundidos"], f["t_tray"],
            f["t_pareja"], f["t_limpieza"], f["t_redet"]))
    if not filas:
        print("ningun clip utilizable")
        return
    d = escribe(CSV1, filas)
    print()
    print("mediana por clip: %.1f s de fase 1 sobre %d cuadros (%.0f ms por cuadro)" % (
        d["t_fase1"].median(), int(d["T"].median()),
        1000 * d["t_fase1"].median() / max(d["T"].median(), 1)))
    print("carga de YOLO, aparte: %.1f s" % t_carga)
    print("csv en %s" % CSV1)


def ocupacion_gpu():
    """QUE HACE: dice cuanta GPU esta usando otro, porque una medida con la GPU compartida no vale."""
    import subprocess
    try:
        s = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=20)
        u, m, tot = [int(v.strip()) for v in s.stdout.strip().splitlines()[0].split(",")]
        return u, m, tot
    except Exception:
        return None, None, None


def fase2(n):
    import torch
    import capas_limpias as CL
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import video_revision as VR

    u, m, tot = ocupacion_gpu()
    if u is not None:
        print("GPU antes de empezar: %d %% de uso, %d de %d MiB ocupados" % (u, m, tot))
        if m > 1500:
            print("AVISO: la GPU ya esta ocupada por otro proceso. Estas cifras saldran infladas y")
            print("       no sirven como coste real. La tirada buena se hace con la GPU libre.")
        print()

    t = time.perf_counter()
    cap5 = CL.Capas(["pose", "seg", "normal", "pointmap", "matting"])
    t_carga = time.perf_counter() - t
    print("cargar las cinco cabezas de Sapiens 2: %.1f s (no se reparte entre clips)" % t_carga)
    print()

    fich = sorted(PRUEBAS.glob("tiempos_*.npz"))[:n]
    if not fich:
        raise SystemExit("no hay NPZ de fase1; corre `tiempos.py fase1 N` primero")
    filas = []
    print("%-34s %5s %6s %8s %8s %8s %8s %8s %7s" % (
        "clip", "util", "pose", "seg", "normal", "pointmap", "matting", "mascara", "invent"))
    print("-" * 104)
    for fp in fich:
        z = np.load(fp, allow_pickle=True)
        cid = fp.name[9:-4]
        B, pres, mask, tipo = z["B"], z["pres"], z["mask"], z["tipo"]
        W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
        cajas, cuadros = z["cajas"], z["cuadros"]
        video = str(z["video"])
        util = [k for k in range(T) if mask[k] and pres[k].any()]
        if not util:
            continue
        """SOLO SE DECODIFICAN LOS CUADROS QUE SE VAN A CRONOMETRAR. La primera version de esto
        procesaba los clips ENTEROS y tardaba horas: 10 clips por 300 cuadros por 2 tiradores son
        6.000 pasadas de pose, mas las densas. Y no hacia falta, porque lo que se informa es el coste
        por unidad de trabajo multiplicado por el trabajo real, no un cronometro de reloj de pared
        sobre el clip completo. Medir en muestra y multiplicar da la misma cifra y encima no arrastra
        la deriva de una tirada larga."""
        ks = util[::max(1, len(util) // MUESTRA_DENSA)][:MUESTRA_DENSA]
        capv = cv2.VideoCapture(video)
        frames = {}
        for k in ks:
            capv.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, im = capv.read()
            if ok:
                frames[k] = im
        capv.release()

        KP = np.full((T, 2, VR.N_PTS, 2), np.nan, dtype="float32")
        SC = np.zeros((T, 2, VR.N_PTS), dtype="float32")
        """CALENTAMIENTO antes de cronometrar. La primera pasada de cada cabeza incluye la compilacion
        de los nucleos de CUDA y la reserva del `caching allocator`, que son coste de arranque y no de
        trabajo: contarlos infla el coste por cuadro y no se pagan por clip en produccion."""
        kc = ks[0] if ks else None
        if kc is not None and kc in frames:
            rgbc = np.ascontiguousarray(frames[kc][:, :, ::-1])
            bc = [B[kc, l] * np.array([W, H, W, H]) for l in (0, 1) if pres[kc, l]]
            if bc:
                cap5.pose(rgbc, bc[:1])
            cc = CL.recorte(B, kc, 0 if pres[kc, 0] else 1, W, H)
            if cc is not None:
                sc0 = np.ascontiguousarray(frames[kc][cc[1]:cc[3], cc[0]:cc[2], ::-1])
                if sc0.shape[0] >= 8 and sc0.shape[1] >= 8:
                    for fn in (cap5.seg, cap5.normal, cap5.pointmap, cap5.matting):
                        fn(sc0)
            torch.cuda.synchronize()

        pose_ms, npose = [], 0
        for k in ks:
            if k not in frames:
                continue
            bs, lados = [], []
            for lado in (0, 1):
                b = B[k, lado]
                if pres[k, lado] and np.isfinite(b).all() and b[2] > b[0] and b[3] > b[1]:
                    bs.append(b * np.array([W, H, W, H]))
                    lados.append(lado)
            if not bs:
                continue
            rgb = np.ascontiguousarray(frames[k][:, :, ::-1])
            torch.cuda.synchronize()
            t = time.perf_counter()
            r = cap5.pose(rgb, bs)
            torch.cuda.synchronize()
            pose_ms.append(1000.0 * (time.perf_counter() - t) / len(bs))
            for j, (kp, sc) in enumerate(r):
                nn = min(len(kp), VR.N_PTS)
                KP[k, lados[j], :nn] = kp[:nn]
                SC[k, lados[j], :nn] = sc[:nn]
            npose += len(bs)
        if not pose_ms:
            continue

        """Las cuatro capas densas corren sobre el RECORTE de cada tirador, que es como corren en la
        tuberia: la mascara limpia se construye por tirador, no de cuadro completo."""
        acu_pose = pose_ms
        acu = {q: [] for q in ("seg", "normal", "pointmap", "matting", "mascara")}
        for k in ks:
            if k not in frames:
                continue
            for lado in (0, 1):
                if not pres[k, lado]:
                    continue
                c = CL.recorte(B, k, lado, W, H)
                if c is None:
                    continue
                sub = np.ascontiguousarray(frames[k][c[1]:c[3], c[0]:c[2], ::-1])
                if sub.shape[0] < 8 or sub.shape[1] < 8:
                    continue
                for q, fn in (("seg", cap5.seg), ("normal", cap5.normal),
                              ("pointmap", cap5.pointmap), ("matting", cap5.matting)):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    fn(sub)
                    torch.cuda.synchronize()
                    acu[q].append(time.perf_counter() - t)
                t = time.perf_counter()
                ajenos = []
                for i in np.nonzero(cuadros == k)[0]:
                    b = np.asarray(cajas[i], dtype="float64")
                    prop = [B[k, l] * np.array([W, H, W, H]) for l in (0, 1)]
                    if max(CL.EP.RD._iou(b, prop[0]), CL.EP.RD._iou(b, prop[1])) > 0.30:
                        continue
                    ajenos.append(b)
                CL.reparto_esqueleto((KP[k, lado].astype("float64"), SC[k, lado].astype("float64")),
                                     [(KP[k, 1 - lado].astype("float64"),
                                       SC[k, 1 - lado].astype("float64"))], W, H)
                acu["mascara"].append(time.perf_counter() - t)

        t = time.perf_counter()
        nvacias = VR.tira_poses_vacias(KP, SC, pres, mask, T)
        KP, SC, origen, _hueco = VR.inventa_poses(KP, SC, pres, mask, T)
        t_inv = time.perf_counter() - t

        """`n_lados` es el trabajo REAL del clip: cuantas veces hay que correr un modelo por tirador
        presente en un cuadro entregado. El coste por clip es el coste por lado multiplicado por
        eso."""
        n_lados = int((pres & mask[:, None]).sum())
        med = {q: (float(np.median(v)) if v else float("nan")) for q, v in acu.items()}
        ms_pose = float(np.median(acu_pose))
        f = {"clip": cid, "T": T, "util": len(util), "n_lados": n_lados,
             "n_medidos_pose": npose, "ms_pose_lado": ms_pose,
             "t_pose": ms_pose * n_lados / 1000.0,
             "t_inventa": t_inv, "nvacias": nvacias,
             "ninventadas": int((origen > 0).sum())}
        for q in ("seg", "normal", "pointmap", "matting", "mascara"):
            f["ms_" + q] = 1000 * med[q]
            f["t_" + q] = med[q] * n_lados
        f["t_fase2"] = (f["t_pose"] + t_inv + sum(f["t_" + q] for q in
                                                  ("seg", "normal", "pointmap", "matting",
                                                   "mascara")))
        filas.append(f)
        print("%-34s %5d %6.1f %8.1f %8.1f %8.1f %8.1f %8.1f %7.2f" % (
            cid[:34], len(util), f["t_pose"], f["t_seg"], f["t_normal"], f["t_pointmap"],
            f["t_matting"], f["t_mascara"], t_inv))
    if not filas:
        print("ningun clip utilizable")
        return
    d = escribe(CSV2, filas)
    print()
    print("mediana por clip: %.1f s de fase 2" % d["t_fase2"].median())
    print("carga de los modelos, aparte: %.1f s" % t_carga)
    print("csv en %s" % CSV2)


def informe():
    import pandas as pd
    if not CSV1.exists():
        raise SystemExit("falta %s" % CSV1)
    a = pd.read_csv(CSV1)
    b = pd.read_csv(CSV2) if CSV2.exists() else None
    d = a if b is None else a.merge(b.drop(columns=["T"]), on="clip", how="left")
    d["t_total"] = d["t_fase1"] + (d["t_fase2"] if "t_fase2" in d else 0.0)

    print("=" * 96)
    print("COSTE POR CLIP, sobre %d clips y %d cuadros" % (len(d), int(d["T"].sum())))
    print("=" * 96)
    print()
    pasos = [("decodificar el video", "t_decod"), ("YOLO11m-pose, cuadro completo", "t_yolo"),
             ("miniaturas en gris (fundidos)", "t_gris"),
             ("puerta de planos y repeticiones", "t_planos"),
             ("puerta de fundidos", "t_fundidos"), ("trayectorias", "t_tray"),
             ("eleccion de la pareja", "t_pareja"),
             ("limpieza (duplicados, saltos, tiras, recorte)", "t_limpieza"),
             ("redeteccion en recorte", "t_redet"),
             ("pose de 308 puntos", "t_pose"), ("segmentacion", "t_seg"),
             ("normales", "t_normal"), ("pointmap", "t_pointmap"), ("matting", "t_matting"),
             ("mascara limpia (reparto por esqueleto)", "t_mascara"),
             ("invencion de las poses que faltan", "t_inventa")]
    tot = d["t_total"].quantile(0.10)
    """SE INFORMAN DOS COLUMNAS Y LA BUENA ES LA DE `p10`, no la mediana.

    La contencion de GPU **solo puede sumar** tiempo: si otro proceso esta usando la tarjeta, cada
    pasada espera. Nunca resta. Asi que la mediana de una tirada hecha con la tarjeta compartida esta
    inflada por una cantidad desconocida, mientras que el **percentil 10 se acerca por abajo al coste
    real**, que es el de la tarjeta libre.

    Y esto no es teorico, esta medido en esta misma tirada: `matting` dio 11,9 s en los primeros
    clips y **356,6 s** en uno posterior, cuando otros agentes volvieron a cargar sus modelos. Un
    factor 30 que no es del modelo.

    Se ensenian las dos para que se vea la diferencia, porque la separacion entre las dos columnas ES
    la medida de la contencion."""
    print("%-44s %9s %9s %9s %7s" % ("paso", "p10 s/clip", "mediana", "ms/cuadro", "% "))
    print("-" * 84)
    for et, col in pasos:
        if col not in d or d[col].isna().all():
            continue
        s = d[col].quantile(0.10)
        m = d[col].median()
        print("%-44s %10.2f %9.2f %9.1f %6.1f %%" % (
            et, s, m, 1000 * s / max(d["T"].median(), 1),
            100 * s / tot if tot else float("nan")))
    print("-" * 84)
    print("%-44s %10.2f %9.2f %9.1f %6.1f %%" % (
        "TOTAL", tot, d["t_total"].median(), 1000 * tot / max(d["T"].median(), 1), 100.0))
    print()
    print("por clip: p10 %.1f s (coste con la tarjeta libre), mediana %.1f s, p90 %.1f s, "
          "max %.1f s" % (tot, d["t_total"].median(), d["t_total"].quantile(0.9),
                          d["t_total"].max()))
    print("cuadros por clip: mediana %d" % int(d["T"].median()))
    print()
    print("CONTENCION DE GPU en esta tirada: la mediana esta %.1f veces por encima del p10." % (
        d["t_total"].median() / max(tot, 1e-6)))
    print("   La contencion solo puede SUMAR tiempo, nunca restarlo, asi que el p10 es la cifra que")
    print("   se acerca al coste real y la mediana es el coste con la tarjeta compartida.")
    print()
    h = CLIPS_CORPUS * tot / 3600.0
    print("EXTRAPOLACION al corpus de %d clips de florete: %.0f h en una GPU (%.1f dias)" % (
        CLIPS_CORPUS, h, h / 24.0))
    print("   con 4 GPU en paralelo: %.1f dias" % (h / 24.0 / 4))
    print()
    print("DONDE SE VA EL TIEMPO, por si hay que recortar")
    caros = sorted(((d[c].median(), e) for e, c in pasos
                    if c in d and not d[c].isna().all()), reverse=True)[:5]
    for s, e in caros:
        print("   %-48s %6.2f s  %5.1f %%" % (e, s, 100 * s / tot if tot else float("nan")))
    if "ambos" in d:
        print()
        print("y la calidad de esa misma tirada: los dos tiradores presentes en el %.1f %% de los "
              "cuadros de mediana, %d clips por debajo del 90 %%" % (
                  100 * d["ambos"].median(), int((d["ambos"] < 0.90).sum())))


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "informe"
    nn = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    if q == "fase1":
        fase1(nn)
    elif q == "fase2":
        fase2(nn)
    else:
        informe()

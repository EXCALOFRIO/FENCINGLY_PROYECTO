"""AUDITORIA de la limpieza de capas densas en clips que NO se usaron para ajustarla.

POR QUE EXISTE ESTE FICHERO
===========================
La limpieza de `capas_limpias.py` se ajusto y se midio en TRES clips:
`defeatedHarlequinRabbitSkinkLizard`, `RichDefensiveBushbaby` y `ZanyMeagerIndianrockpython`. Las
cifras que dio (recall 0,994, fuga de arbitro 0,0000, residuo 0,263) son cifras DE SUS PROPIOS CLIPS
DE AJUSTE, y por eso no valen como veredicto: es donde se cae todo lo que se ajusta mirando.

Este script las vuelve a medir en un lote NUEVO, elegido a proposito entre los casos duros y sin
mirar el resultado antes de fijar los limites.

CRITERIO DECLARADO ANTES DE EJECUTAR
====================================
Las tres cifras son las de `capas_limpias.mide`, que es la que ya estaba escrita, y ninguna sale de
la capa que se juzga:

  rec_kp      keypoints propios del tirador (Sapiens pose, score >= 0,3, dentro del recorte) que
              caen dentro de la mascara final.            LIMITE: media por clip >= 0,90
  fuga_arb    fraccion de (banda de margen INTERSECCION caja de una deteccion NO elegida) que la
              mascara enciende. La caja de YOLO es ajustada, asi que todo pixel de la mascara en la
              banda esta mal por construccion.            LIMITE: media por clip <= 0,05
  residuo     fraccion del recorte que la mascara enciende. Sirve para comparar con la LINEA BASE
              TONTA (la caja sola, que da 0,657): si la mascara no baja de ahi, no ha limpiado nada.
                                                          LIMITE: media por clip <= 0,45
  rec_desp    CONTROL NEGATIVO obligatorio: la MISMA mascara desplazada media anchura de recorte,
              evaluada con los mismos keypoints. Si `rec_kp` es alto porque la mascara ocupa medio
              recorte, `rec_desp` tambien sale alto y el recall no significa nada.
                                                          LIMITE: rec_kp / rec_desp >= 5

Y dos cifras mas que se leen, no se aprueban, porque no tienen limite honrado que declarar:
  fuga_fondo  lo mismo que `fuga_arb` en el resto de la banda (suelo, pared, publico sin caja).
  resto_arb   fraccion de la mascara final que es pixel OSCURO dentro de la caja de un ajeno, o sea
              el arbitro que queda DENTRO de la caja propia, donde la metrica de banda es ciega.

EL LOTE, y por que estos diez
=============================
Diez clips del corpus etiquetado de FLORETE (`scratchpad/corpus_etiquetado.txt`), NINGUNO de los tres
de ajuste, elegidos por DENSIDAD DE DETECCIONES POR CUADRO, que es el proxy medible de los casos que
el usuario pidio: sala con varias pistas, arbitro pegado a los tiradores y publico cerca. Los diez
estan entre 6,7 y 14,0 personas por cuadro; los tres de ajuste estaban entre 3,6 y 4,3. O sea que el
lote de auditoria es MAS DURO que el de ajuste, no un lote equivalente.

Cuatro de los diez son 1080p y seis 720p, a proposito: `fuga_fondo` ya se vio empeorar 26x con
tiradores pequenos, asi que la resolucion no puede quedar fuera del reparto.

MUESTREO: 20 cuadros repartidos por igual entre los cuadros UTILIZABLES con los dos tiradores
presentes. No se muestrean solo los cuadros con arbitro: eso subiria `fuga_arb` artificialmente al
concentrar los casos duros. `fuga_arb` se da tambien sobre el subconjunto con ajeno medible, aparte.

USO:  python capas_limpias_auditoria.py [N_CUADROS]    con el interprete de transformers+torch+CUDA
"""
from __future__ import annotations

import json
import pathlib
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import capas_limpias as CL

import cv2

SCR = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\187cfd83-d8d9-4c4d-bed4-f255724f6b21\scratchpad")
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
VIDEOS = "D:/FENCINGLY_VIDEOS_RAW"

AJUSTE = ("defeatedHarlequinRabbitSkinkLizard", "RichDefensiveBushbaby",
          "ZanyMeagerIndianrockpython")
"""Los tres clips con los que se ajusto la limpieza. Estan aqui escritos para que no puedan colarse
en el lote de auditoria por descuido."""

LOTE = ["WhisperedTenderDeinonychus", "SimilarCluelessEeve", "GrossRelievedDwarfrabbit",
        "LegitimateShoddyBighorn", "WildHopefulIndianpalmsquirrel",
        "UnfortunateGracefulGalapagossealion",
        "BabyishIdioticCockerspaniel", "DemandingMindlessDeviltasmanian",
        "WelloffForcefulIndri", "SeveralSkinnyGenet"]

LIM_REC = 0.90
LIM_FUGA = 0.05
LIM_RESIDUO = 0.45
LIM_RAZON_CONTROL = 5.0
BASE_CAJA = 0.657
"""Residuo de la linea base tonta (la caja sola), medido en `capas_limpias.py`."""


def mapa_videos():
    import os
    m = {}
    for raiz, _d, fs in os.walk(VIDEOS):
        for f in fs:
            if f.lower().endswith(".mp4"):
                m.setdefault(pathlib.Path(f).stem, os.path.join(raiz, f))
    return m


def gris_del_clip(ruta):
    """La puerta de FUNDIDOS de `extrae_pareja` necesita el clip en gris a 160x90. Sin `gris` no se
    aplica y entrarian cuadros de transicion, que es justo donde `seg` y `matting` fallan y donde la
    auditoria mediria un fallo que la tuberia real no comete."""
    g = []
    v = cv2.VideoCapture(ruta)
    while True:
        ok, im = v.read()
        if not ok:
            break
        g.append(cv2.cvtColor(cv2.resize(im, (160, 90)), cv2.COLOR_BGR2GRAY))
    v.release()
    return g


def audita_clip(cap, clip, ruta, n_cuadros):
    with open(SCR / ("det_%s.pkl" % clip), "rb") as fh:
        cajas, cuadros, W, H, T = pickle.load(fh)
    sel = CL.elige_pareja_del_clip(cajas, cuadros, W, H, T, gris=gris_del_clip(ruta))
    B, pres, msk = sel["B"], sel["pres"], sel["mask"]
    utiles = [k for k in range(T) if msk[k] and pres[k].all()]
    if len(utiles) < 4:
        return None, []
    idx = np.unique(np.linspace(0, len(utiles) - 1, min(n_cuadros, len(utiles))).round().astype(int))
    KS = [utiles[i] for i in idx]
    vid = cv2.VideoCapture(ruta)
    filas, t_pose, t_mask, n_aj = [], 0.0, 0.0, []
    for k in KS:
        vid.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, im = vid.read()
        if not ok:
            continue
        rgb = np.ascontiguousarray(im[:, :, ::-1])
        aj = CL.ajenos_del_cuadro(sel, k, H)
        n_aj.append(len(aj))
        cbs = [B[k, 0] * np.array([W, H, W, H]), B[k, 1] * np.array([W, H, W, H])] + aj
        t0 = time.perf_counter()
        poses = cap.pose(rgb, cbs)
        t_pose += time.perf_counter() - t0
        kt = {("t", 0): poses[0], ("t", 1): poses[1]}
        for j in range(len(aj)):
            kt[("a", j)] = poses[2 + j]
        for lado in (0, 1):
            t0 = time.perf_counter()
            r = CL.mascara_tirador(cap, rgb, sel, k, lado, W, H, kt, con_matting=True)
            t_mask += time.perf_counter() - t0
            if r is None:
                continue
            f = CL.mide(r, sel, k, lado, W, H, kt[("t", lado)], bgr=im)
            f["clip"] = clip
            f["n_ajenos"] = len(aj)
            f["alto_caja_frac"] = float(B[k, lado, 3] - B[k, lado, 1])
            filas.append(f)
    vid.release()
    import pandas as pd
    d = pd.DataFrame(filas)
    con = d[d["px_banda_arb"] > 2000]
    res = {"clip": clip, "W": W, "H": H, "T": T, "utilizables": int(msk.sum()),
           "cuadros_medidos": int(d["k"].nunique()), "lados": int(len(d)),
           "ajenos_por_cuadro": float(np.mean(n_aj)) if n_aj else 0.0,
           "rec_kp": float(d["rec_kp"].mean()), "rec_desp": float(d["rec_desp"].mean()),
           "fuga_arb": float(d["fuga_arb"].mean()), "fuga_fondo": float(d["fuga_fondo"].mean()),
           "residuo": float(d["residuo"].mean()),
           "resto_arb_frac": float(d["resto_arb_frac"].mean()),
           "perdida_capas": float(d["perdida_capas"].mean()),
           "perdida_esqueleto": float(d["perdida_esqueleto"].mean()),
           "perdida_guarda": float(d["perdida_guarda"].mean()),
           "fuga_solape": float(d["fuga_solape"].mean()),
           "ret_oscuro_propio": float(d["ret_oscuro_propio"].mean()),
           "lados_con_ajeno_medible": int(len(con)),
           "fuga_arb_con_ajeno": float(con["fuga_arb"].mean()) if len(con) else None,
           "rec_kp_con_ajeno": float(con["rec_kp"].mean()) if len(con) else None,
           "marcados_rec_bajo": int((d["rec_kp"] < 0.70).sum()),
           "alto_caja_frac": float(d["alto_caja_frac"].median()),
           "ms_pose_cuadro": 1000.0 * t_pose / max(d["k"].nunique(), 1),
           "ms_mascara_lado": 1000.0 * t_mask / max(len(d), 1)}
    return res, filas


def main():
    global LOTE
    n_cuadros = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    suf = ""
    if len(sys.argv) > 2:
        """El segundo argumento repite la auditoria SOLO en unos clips, con el MISMO muestreo (es un
        `linspace` sobre los cuadros utilizables, asi que sale determinista y los cuadros coinciden
        con los de la tirada de los diez). Sirve para volver sobre los lados marcados sin pagar los
        diez clips otra vez, NO para elegir clips hasta que salga bien la cifra: por eso las tablas
        van a fichero aparte con sufijo `_sub` y no pisan las de los diez."""
        LOTE = sys.argv[2].split(",")
        suf = "_sub"
    for c in LOTE:
        assert c not in AJUSTE, "clip de ajuste en el lote de auditoria: %s" % c
    mv = mapa_videos()
    cap = CL.Capas(["pose", "matting", "seg"])
    """`normal` y `pointmap` NO se cargan aqui a proposito: la mascara se construye con `seg` y
    `matting`, asi que cargarlas costaria GPU sin cambiar ni una cifra de esta tabla. Lo que si hay
    que comprobar es si ADMITEN la mascara, y eso va en `capas_limpias_por_capa.py`."""
    todos, res = [], []
    for i, clip in enumerate(LOTE):
        if clip not in mv:
            print("  %s: NO esta en disco, se salta" % clip)
            continue
        t0 = time.perf_counter()
        r, filas = audita_clip(cap, clip, mv[clip], n_cuadros)
        if r is None:
            print("  %s: menos de 4 cuadros utilizables, se salta" % clip)
            continue
        r["s_clip"] = time.perf_counter() - t0
        res.append(r)
        todos += filas
        print("  %-38s rec %.3f (desp %.3f)  fuga_arb %.4f  residuo %.3f  resto_arb %.4f  %.0f s"
              % (clip, r["rec_kp"], r["rec_desp"], r["fuga_arb"], r["residuo"],
                 r["resto_arb_frac"], r["s_clip"]))

    import pandas as pd
    D = pd.DataFrame(todos)
    R = pd.DataFrame(res)
    D.to_csv(SALIDA / ("auditoria_capas_lados%s.csv" % suf), index=False)
    R.to_csv(SALIDA / ("auditoria_capas_clips%s.csv" % suf), index=False)

    print()
    print("=" * 118)
    print("AUDITORIA DE LA LIMPIEZA EN %d CLIPS NUEVOS  ·  %d cuadros, %d lados"
          % (len(R), int(R["cuadros_medidos"].sum()), len(D)))
    print("=" * 118)
    print("%-38s %5s %4s %6s %6s %7s %7s %7s %7s %5s"
          % ("clip", "res", "aj/c", "rec_kp", "desp", "fuga_ar", "fuga_fo", "residuo", "resto_a",
             "n<0.7"))
    for _i, r in R.iterrows():
        print("%-38s %5s %4.1f %6.3f %6.3f %7.4f %7.4f %7.3f %7.4f %5d"
              % (r["clip"], "1080p" if r["H"] >= 1000 else "720p", r["ajenos_por_cuadro"],
                 r["rec_kp"], r["rec_desp"], r["fuga_arb"], r["fuga_fondo"], r["residuo"],
                 r["resto_arb_frac"], r["marcados_rec_bajo"]))
    print("-" * 118)
    glob = {"rec_kp": float(D["rec_kp"].mean()), "rec_desp": float(D["rec_desp"].mean()),
            "fuga_arb": float(D["fuga_arb"].mean()), "fuga_fondo": float(D["fuga_fondo"].mean()),
            "residuo": float(D["residuo"].mean()),
            "resto_arb_frac": float(D["resto_arb_frac"].mean()),
            "perdida_capas": float(D["perdida_capas"].mean()),
            "perdida_esqueleto": float(D["perdida_esqueleto"].mean()),
            "perdida_guarda": float(D["perdida_guarda"].mean())}
    print("%-38s %5s %4s %6.3f %6.3f %7.4f %7.4f %7.3f %7.4f %5d"
          % ("TODOS los lados juntos", "", "", glob["rec_kp"], glob["rec_desp"], glob["fuga_arb"],
             glob["fuga_fondo"], glob["residuo"], glob["resto_arb_frac"],
             int((D["rec_kp"] < 0.70).sum())))
    print()
    razon = glob["rec_kp"] / max(glob["rec_desp"], 1e-9)
    ver = [("rec_kp >= %.2f" % LIM_REC, glob["rec_kp"] >= LIM_REC, glob["rec_kp"]),
           ("fuga_arb <= %.2f" % LIM_FUGA, glob["fuga_arb"] <= LIM_FUGA, glob["fuga_arb"]),
           ("residuo <= %.2f (caja sola %.3f)" % (LIM_RESIDUO, BASE_CAJA),
            glob["residuo"] <= LIM_RESIDUO, glob["residuo"]),
           ("rec_kp/rec_desp >= %.0f" % LIM_RAZON_CONTROL, razon >= LIM_RAZON_CONTROL, razon)]
    print("  VEREDICTO contra los limites declarados ANTES de ejecutar:")
    for nom, ok, v in ver:
        print("    %-38s %-6s  valor %.4f" % (nom, "PASA" if ok else "NO PASA", v))
    print("  Y COMO SE LEE CADA UNO, que no es igual:")
    print("    `fuga_arb` 0,0000 es TAUTOLOGICO y no prueba nada por si solo: la guarda de caja resta\n"
          "    exactamente la region sobre la que se mide. De ese cero lo que hay que leer es que la\n"
          "    guarda NO CUESTA RECALL (%.4f de perdida) y que `resto_arb`, que mide el arbitro que\n"
          "    queda DENTRO de la caja propia, donde la guarda no llega, vale %.4f de media y %.4f\n"
          "    en el peor clip. Esa es la cifra NO tautologica de cuanto arbitro sobrevive."
          % (D["perdida_guarda"].mean(), glob["resto_arb_frac"], R["resto_arb_frac"].max()))
    peor = R.loc[R["rec_kp"].idxmin()]
    print()
    print("  el clip PEOR por recall: %s con %.3f  (%d lados marcados por rec_kp < 0,70)"
          % (peor["clip"], peor["rec_kp"], peor["marcados_rec_bajo"]))
    print("  el clip PEOR por fuga_arb: %s con %.4f"
          % (R.loc[R["fuga_arb"].idxmax(), "clip"], R["fuga_arb"].max()))
    print("  el clip PEOR por resto_arb: %s con %.4f"
          % (R.loc[R["resto_arb_frac"].idxmax(), "clip"], R["resto_arb_frac"].max()))
    print()
    print("  DE QUE SE PIERDE EL RECALL, y esto es lo que dice que arreglar. Los tres sumandos suman")
    print("  1 - rec_kp = %.4f:" % (1.0 - glob["rec_kp"]))
    print("    %.4f  `seg` y `matting` no encontraron al tirador   (no es culpa de la limpieza)"
          % D["perdida_capas"].mean())
    print("    %.4f  se lo llevo el reparto por esqueleto mas cercano" % D["perdida_esqueleto"].mean())
    print("    %.4f  se lo llevo la guarda de caja" % D["perdida_guarda"].mean())
    print("  y SOLO en los %d lados marcados por rec_kp < 0,70:" % int((D["rec_kp"] < 0.70).sum()))
    mal = D[D["rec_kp"] < 0.70]
    if len(mal):
        print("    %.4f capas   %.4f esqueleto   %.4f guarda"
              % (mal["perdida_capas"].mean(), mal["perdida_esqueleto"].mean(),
                 mal["perdida_guarda"].mean()))
    print()
    print("  los 8 peores lados por recall, para ir a mirarlos:")
    print(D.nsmallest(8, "rec_kp")[["clip", "k", "lado", "rec_kp", "rec_bruto",
                                    "perdida_capas", "perdida_esqueleto", "perdida_guarda",
                                    "residuo", "alto_caja_frac"]].to_string(index=False))
    print()
    print("  COSTE con las capas de la mascara solo (pose + seg + matting):")
    print("    pose por cuadro (2 tiradores + ajenos)  %.0f ms" % R["ms_pose_cuadro"].mean())
    print("    mascara por lado (seg + matting + reparto)  %.0f ms" % R["ms_mascara_lado"].mean())
    (SALIDA / ("auditoria_capas%s.json" % suf)).write_text(
        json.dumps({"lote": LOTE, "ajuste": list(AJUSTE), "n_cuadros_pedidos": n_cuadros,
                    "global": glob, "razon_control": razon,
                    "limites": {"rec_kp": LIM_REC, "fuga_arb": LIM_FUGA,
                                "residuo": LIM_RESIDUO, "razon_control": LIM_RAZON_CONTROL},
                    "veredicto": dict((n, bool(o)) for n, o, _v in ver),
                    "clips": res}, indent=1), encoding="utf-8")
    print()
    print("  tablas -> %s" % SALIDA)


if __name__ == "__main__":
    main()

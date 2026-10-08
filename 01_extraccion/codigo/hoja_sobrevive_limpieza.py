"""LA HOJA EN EL ALFA DEL MATTING, y si la LIMPIEZA DE MASCARA se la come.

DOS PREGUNTAS, y la segunda es la que puede estropear el trabajo de hoy
=======================================================================
1) El usuario dice: *"veo en los resultados de matting que con el alfa 0,0006 se puede ver el florete,
   el filo super fino; esto nos puede ayudar a detectarlo para lo del cono"*. Hay que confirmarlo o
   desmentirlo CON CIFRA, y por separado sus dos partes, que no son la misma:
     1a  que el filo de la hoja DEJA SENAL en el alfa bajo. Se mide por CONTRASTE: alfa sobre la
         recta detectada contra alfa en una banda paralela desplazada 8 px, del mismo recorte.
         LIMITE declarado: razon >= 2. Por debajo de 2, lo que se ve es el fondo.
     1b  que **0,0006 vale como umbral**. Se mide por lo que selecciona: en que percentil del alfa de
         FUERA DEL CUERPO cae 0,0006 y que fraccion del recorte enciende.
         LIMITE declarado: para que un umbral fijo sirva tiene que encender menos del 2 % del
         recorte, que es del orden de lo que ocupa una hoja de 90 cm y 1 px de ancho.
         Y EL LIMITE SE JUZGA POR EL p90, NO POR LA MEDIA, y esto no es un tecnicismo: la media de
         `frac_recorte` con 0,0006 sale 0,0072 y APRUEBA, mientras el maximo es 0,0931, o sea que en
         el peor recorte el umbral enciende el 9 % de la imagen. Un umbral que va bien en el recorte
         tipico y enciende el 9 % en el malo no sirve como umbral fijo, y una media que lo aprueba no
         esta midiendo lo que importa. El motivo esta medido: el alfa del fondo cambia **30 veces**
         de un recorte a otro (0,000034 a 0,001011), asi que ningun valor fijo cae en el mismo
         percentil en todos.

2) La limpieza de mascara deja solo a los dos tiradores. La mascara es
   `(alfa >= 0,5 O seg != Background) Y esqueleto mas cercano Y guarda de caja`, y **la hoja tiene
   alfa entre 0,02 y 0,5 y `seg` no la etiqueta**, asi que por construccion la hoja cae FUERA. Si el
   empaquetado guarda el alfa solo dentro de la mascara, la hoja desaparece del dato guardado y
   limpiar habria tenido un efecto secundario malo: limpiar el arbitro y perder el arma.
   Se mide: de los pixeles de hoja detectados, cuantos sobreviven en el alfa empaquetado.
   LIMITE declarado para decir que "sobrevive": >= 0,50.

Y si no sobrevive, la salida NO es dejar de limpiar ni ensanchar la mascara del cuerpo (meter la hoja
en la mascara del cuerpo subiria la fuga por construccion, porque la hoja vive en la banda de margen).
La salida es un **CANAL DE HOJA aparte**, con el mismo reparto por esqueleto mas cercano y la misma
guarda de caja que el cuerpo, y se mide su propia contaminacion:
   contaminacion = fraccion de los pixeles del canal de hoja que caen dentro de la caja de una
   deteccion NO elegida.   LIMITE declarado: <= 0,05, el mismo que se le exige a la fuga del cuerpo.

TRES UMBRALES, medidos en paralelo sobre el MISMO alfa
======================================================
  fijo del usuario   0,0006
  adaptativo         percentil 99,5 del alfa de fuera del cuerpo, del propio recorte. Es el que usa
                     el proyecto y su motivo esta medido: el alfa del fondo NO es cero (mediana
                     3,1e-4 a 5,2e-4) y **cambia el doble de un recorte a otro**, asi que un umbral
                     fijo no puede valer para los dos.
  control  0,5        el umbral "razonable" con el que un test anterior dio CERO hojas. Se mide para
                     que quede la cifra de por que no vale.

CLIPS: cuatro de 1080p del corpus etiquetado, los mismos del mosaico. **A 720p no se mide** porque ya
esta medido que la hoja no se detecta ahi (1,6 % contra 42,6 %), lo que encaja con su ancho de
0,64 px a 720p contra 0,96 a 1080p.

EL NOMBRE NO LLEVA NUMERO DE ETAPA a proposito: los huecos `e10` a `e13` los esta ocupando otro
trabajo en paralelo sobre el cono en 3D, y dos ficheros con el mismo numero y distinto contenido es
peor que un nombre sin numero. Lo que se mide aqui no es un paso de la etapa E: es la comprobacion
cruzada entre la etapa E y la limpieza de capas.

Y LO QUE ESTE FICHERO NO MIDE, porque lo mide ese otro trabajo y repetirlo gastaria la misma GPU dos
veces: la TASA de deteccion de hoja a 720p contra 1080p (`e13_hoja_720_vs_1080.py`).

USO:  python hoja_sobrevive_limpieza.py [N_CUADROS_POR_CLIP]  con el interprete de transformers+CUDA
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
import capas_limpias_auditoria as AU
import cono_lib as CO
import e7_pinta_cono as E7

import cv2

SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
CLIPS = ["AromaticShockedGermanspitz", "ImpeccablePertinentBasilisk",
         "DemandingMindlessDeviltasmanian", "WelloffForcefulIndri"]

FIJO_USUARIO = 0.0006
CONTROL_ALTO = 0.5
AP = 25.4
"""Semiapertura del cono, el p90 MEDIDO de |angulo de la hoja - eje del antebrazo| a 1080p."""
LIM_CONTRASTE = 2.0
LIM_FRAC_RECORTE = 0.02
LIM_SUPERVIVENCIA = 0.50
LIM_CONTAMINA = 0.05
BANDA = 8
"""Desplazamiento perpendicular de la banda de control, en px. Ocho porque la dilatacion del cuerpo
ya es de 8 px, asi que la banda de control queda tan fuera del cuerpo como la propia recta."""


def poses_del_muestreo(cap, ruta, sel, W, H, T, KS):
    """Pose en los cuadros muestreados. Hace falta ANTES de nada porque la mano armada se decide por
    MEDIANAS del clip (`cono_lib.mano_armada`), no cuadro a cuadro."""
    KP = np.full((T, 2, 308, 2), np.nan, "float32")
    SC = np.zeros((T, 2, 308), "float32")
    guardadas = {}
    B = sel["B"]
    vid = cv2.VideoCapture(ruta)
    for k in KS:
        vid.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, im = vid.read()
        if not ok:
            continue
        rgb = np.ascontiguousarray(im[:, :, ::-1])
        aj = CL.ajenos_del_cuadro(sel, k, H)
        cbs = [B[k, 0] * np.array([W, H, W, H]), B[k, 1] * np.array([W, H, W, H])] + aj
        poses = cap.pose(rgb, cbs)
        kt = {("t", 0): poses[0], ("t", 1): poses[1]}
        for j in range(len(aj)):
            kt[("a", j)] = poses[2 + j]
        guardadas[k] = (kt, aj)
        for lado in (0, 1):
            kp, sc = poses[lado]
            n = min(len(kp), 308)
            KP[k, lado, :n], SC[k, lado, :n] = kp[:n], sc[:n]
    vid.release()
    return KP, SC, guardadas


def banda_de_recta(forma, p1, p2, off):
    """Mascara de una banda de 3 px alrededor del segmento, desplazada `off` px en perpendicular."""
    v = p2 - p1
    n = np.linalg.norm(v)
    if n < 1:
        return np.zeros(forma, bool)
    per = np.array([-v[1], v[0]]) / n
    m = np.zeros(forma, "uint8")
    cv2.line(m, tuple(np.int32(p1 + off * per)), tuple(np.int32(p2 + off * per)), 1, 3)
    return m > 0


N_POSE = 40
"""Cuadros sobre los que se corre la pose, que son MAS que los que se miden.

QUE FALLABA: `cono_lib.mano_armada` decide por MEDIANAS del clip y exige al menos 10 cuadros con
codo, hombro y mano validos. Con 8 cuadros muestreados devolvia `None` SIEMPRE, y el script daba
"0 lados" en los cuatro clips sin decir por que. La pose cuesta 0,3-0,6 s por cuadro y `matting` 0,8
por recorte, asi que ampliar solo la pasada de pose es casi gratis y es lo que arregla la mediana."""


def mide_clip(cap, clip, ruta, n_cuadros):
    with open(AU.SCR / ("det_%s.pkl" % clip), "rb") as fh:
        cajas, cuadros, W, H, T = pickle.load(fh)
    sel = CL.elige_pareja_del_clip(cajas, cuadros, W, H, T, gris=AU.gris_del_clip(ruta))
    B, pres, msk = sel["B"], sel["pres"], sel["mask"]
    utiles = [k for k in range(T) if msk[k] and pres[k].all()]
    if len(utiles) < 12:
        print("    %s: solo %d cuadros utilizables, se salta" % (clip, len(utiles)))
        return []
    n_pose = max(N_POSE, n_cuadros)
    ip = np.unique(np.linspace(0, len(utiles) - 1, min(n_pose, len(utiles))).round().astype(int))
    K_POSE = [utiles[i] for i in ip]
    idx = np.unique(np.linspace(0, len(K_POSE) - 1, min(n_cuadros, len(K_POSE))).round().astype(int))
    KS = [K_POSE[i] for i in idx]
    """Los cuadros medidos son un SUBCONJUNTO de los de pose, no otro muestreo: asi el eje que se usa
    para el cono de un cuadro medido es el eje calculado en ese mismo cuadro."""
    KP, SC, guardadas = poses_del_muestreo(cap, ruta, sel, W, H, T, K_POSE)
    geo = {}
    for lado in (0, 1):
        mano, _d = CO.mano_armada(KP, SC, B, W, H, lado)
        if mano is None:
            geo[lado] = None
            print("    %s lado %d: SIN mano armada (menos de 10 cuadros con codo, hombro y mano de "
                  "los %d con pose)" % (clip, lado, len(K_POSE)))
            continue
        vert, u, ok, _n = CO.eje_antebrazo(KP, SC, lado, mano)
        _pk, pxm = CO.escala_px_m(B, lado, H)
        geo[lado] = {"mano": mano, "vert": vert, "u": u, "ok": ok, "pxm": pxm,
                     "L": CO.HOJA_M * pxm}
    vid = cv2.VideoCapture(ruta)
    filas = []
    for k in KS:
        if k not in guardadas:
            continue
        vid.set(cv2.CAP_PROP_POS_FRAMES, k)
        okf, im = vid.read()
        if not okf:
            continue
        rgb = np.ascontiguousarray(im[:, :, ::-1])
        kt, aj = guardadas[k]

        """La mascara limpia del cuerpo, en coordenadas de cuadro, para poder preguntar si la hoja
        sobrevive dentro de ella."""
        M_lim = np.zeros((H, W), bool)
        for lado in (0, 1):
            r = CL.mascara_tirador(cap, rgb, sel, k, lado, W, H, kt, con_matting=True)
            if r is not None:
                c = r["c"]
                M_lim[c[1]:c[3], c[0]:c[2]] |= r["mask"]

        for lado in (0, 1):
            g = geo[lado]
            if g is None or not g["ok"][k] or not np.isfinite(B[k, lado]).all():
                continue
            vert, u, L = g["vert"][k], g["u"][k], g["L"]
            (cx0, cy0, cx1, cy1), R = E7.recorte(B[k, lado], B[k, 1 - lado], vert, u, L, W, H)
            if cx1 - cx0 < 64 or cy1 - cy0 < 64 or R < 24:
                continue
            sub = np.ascontiguousarray(rgb[cy0:cy1, cx0:cx1])
            alfa = cap.matting(sub)
            seg = cap.seg(sub)
            if alfa.shape[:2] != sub.shape[:2] or seg.shape[:2] != sub.shape[:2]:
                continue
            hh, ww = alfa.shape[:2]
            cuerpo = cv2.dilate((seg != 0).astype("uint8"),
                                np.ones((2 * E7.DILATA + 1,) * 2, np.uint8)) > 0
            fuera = alfa[~cuerpo]
            if fuera.size < 500:
                continue
            GY, GX = np.mgrid[0:hh, 0:ww]
            disco = np.hypot(GX + cx0 - vert[0], GY + cy0 - vert[1]) <= R
            t_ad = float(np.percentile(fuera, E7.PCTL))
            f = {"clip": clip, "k": k, "lado": lado, "W": W, "H": H, "R_px": R,
                 "alfa_medio_recorte": float(alfa.mean()),
                 "alfa_fondo_mediana": float(np.median(fuera)),
                 "alfa_fondo_p99_5": t_ad,
                 "pctl_de_0006": float(100.0 * (fuera < FIJO_USUARIO).mean()),
                 "px_recorte": int(hh * ww)}

            """LOS TRES UMBRALES CON EL MISMO HOUGH Y LA MISMA CONDICION de pasar a menos de 0,12 R
            de la mano, que es la del detector del proyecto (`e7_pinta_cono.detecta`). Se rehace aqui
            en vez de llamarlo porque ese lleva el umbral adaptativo dentro y hay que poder cambiarlo
            sin cambiar nada mas: si cada umbral se juzgara con un detector distinto, la comparacion
            no seria de umbrales.

            `frac_recorte` es la cifra que decide si un umbral fijo puede servir: una hoja de 90 cm y
            1 px de ancho no puede ocupar el 20 % del recorte."""
            for nom, t in (("fijo", FIJO_USUARIO), ("adapt", t_ad), ("ctrl05", CONTROL_ALTO)):
                m = disco & (~cuerpo) & (alfa > t)
                f["px_" + nom] = int(m.sum())
                f["frac_recorte_" + nom] = float(m.sum() / float(hh * ww))
                ln = None
                if m.sum() >= 20:
                    lines = cv2.HoughLinesP((m * 255).astype("uint8"), 1, np.pi / 360.0, 12,
                                            minLineLength=int(max(8, E7.MIN_LARGO * R)),
                                            maxLineGap=int(max(2, 0.05 * R)))
                    cands = []
                    if lines is not None:
                        for x1_, y1_, x2_, y2_ in np.asarray(lines).reshape(-1, 4):
                            p1 = np.array([x1_ + cx0, y1_ + cy0], "float64")
                            p2 = np.array([x2_ + cx0, y2_ + cy0], "float64")
                            v = p2 - p1
                            lg = float(np.linalg.norm(v))
                            if lg < 1:
                                continue
                            nv = np.array([-v[1], v[0]]) / lg
                            if abs(float(nv @ (vert - p1))) > E7.CERCA_ROOT * R:
                                continue
                            cands.append((lg, p1, p2))
                    if cands:
                        cands.sort(key=lambda c: -c[0])
                        ln = cands[0]
                f["recta_" + nom] = ln is not None
                if ln is not None:
                    lg, p1, p2 = ln
                    v = p2 - p1
                    f["largo_cm_" + nom] = 100.0 * lg / g["pxm"]
                    f["dif_eje_" + nom] = float(abs((np.degrees(np.arctan2(v[1], v[0]))
                                                     - np.degrees(np.arctan2(u[1], u[0]))
                                                     + 90.0) % 180.0 - 90.0))
                if nom == "adapt" and ln is not None:
                    """1a CONTRASTE: alfa sobre la recta contra alfa en dos bandas paralelas
                    desplazadas 8 px. Es la medida que separa "hay senal de hoja" de "hay fondo"."""
                    lg, p1, p2 = ln
                    o = np.array([cx0, cy0], "float64")
                    b0 = banda_de_recta((hh, ww), p1 - o, p2 - o, 0.0)
                    b1 = banda_de_recta((hh, ww), p1 - o, p2 - o, float(BANDA))
                    b2 = banda_de_recta((hh, ww), p1 - o, p2 - o, -float(BANDA))
                    b0 &= ~cuerpo
                    ctrl = (b1 | b2) & ~cuerpo & ~b0
                    if b0.sum() >= 20 and ctrl.sum() >= 20:
                        a0, a1 = float(np.median(alfa[b0])), float(np.median(alfa[ctrl]))
                        f["alfa_sobre_hoja"] = a0
                        f["alfa_banda_control"] = a1
                        f["contraste"] = a0 / max(a1, 1e-9)

                    """2 SUPERVIVENCIA: de los pixeles de la banda de la hoja, cuantos estan dentro
                    de la mascara limpia del cuerpo, que es lo que el empaquetado guarda."""
                    yy, xx = np.nonzero(b0)
                    if yy.size:
                        dentro = M_lim[np.clip(yy + cy0, 0, H - 1), np.clip(xx + cx0, 0, W - 1)]
                        f["px_hoja_banda"] = int(yy.size)
                        f["sobrevive_en_mascara"] = float(dentro.mean())

            """CANAL DE HOJA aparte: los pixeles de hoja del umbral adaptativo, repartidos por
            esqueleto mas cercano y con la guarda de caja, igual que el cuerpo. Se mide su propia
            contaminacion por cajas ajenas, antes y despues del reparto."""
            m_ad = disco & (~cuerpo) & (alfa > t_ad)
            if m_ad.sum() >= 20:
                vor = CL.reparto_esqueleto(
                    kt[("t", lado)],
                    [kt[("t", 1 - lado)]] + [kt[("a", i)] for i in range(len(aj))
                                             if ("a", i) in kt], W, H)[cy0:cy1, cx0:cx1]
                gu = CL.guarda_de_caja((cx0, cy0, cx1, cy1),
                                       B[k, lado] * np.array([W, H, W, H]), aj, hh, ww)
                canal = m_ad & vor & gu
                caja_aj = np.zeros((hh, ww), bool)
                for o in aj:
                    y0, y1 = int(max(0, o[1] - cy0)), int(min(hh, o[3] - cy0))
                    x0, x1 = int(max(0, o[0] - cx0)), int(min(ww, o[2] - cx0))
                    if y1 > y0 and x1 > x0:
                        caja_aj[y0:y1, x0:x1] = True
                f["canal_px_bruto"] = int(m_ad.sum())
                f["canal_px_limpio"] = int(canal.sum())
                f["canal_conserva"] = float(canal.sum() / max(m_ad.sum(), 1))
                f["canal_contamina_bruto"] = float((m_ad & caja_aj).sum() / max(m_ad.sum(), 1))
                f["canal_contamina_limpio"] = float((canal & caja_aj).sum() / max(canal.sum(), 1)) \
                    if canal.sum() else 0.0
            filas.append(f)
    vid.release()
    return filas


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    mv = AU.mapa_videos()
    cap = CL.Capas(["pose", "matting", "seg"])
    todo = []
    for clip in CLIPS:
        if clip not in mv:
            print("  %s no esta en disco" % clip)
            continue
        t0 = time.perf_counter()
        f = mide_clip(cap, clip, mv[clip], n)
        todo += f
        print("  %-36s %3d lados en %.0f s" % (clip, len(f), time.perf_counter() - t0))

    import pandas as pd
    D = pd.DataFrame(todo)
    D.to_csv(SALIDA / "hoja_y_limpieza.csv", index=False)
    if not len(D):
        print("  SIN NI UN LADO MEDIBLE. Los motivos van impresos arriba, clip por clip: sin mano "
              "armada, sin eje valido en el cuadro o recorte insuficiente. No hay cifra que dar.")
        return
    print()
    print("=" * 112)
    print("LA HOJA EN EL ALFA  ·  %d lados de %d clips de 1080p" % (len(D), D["clip"].nunique()))
    print("=" * 112)
    print("  1a  LA OBSERVACION DEL USUARIO: hay senal de hoja en el alfa bajo?")
    ct = D[D["contraste"].notna()] if "contraste" in D else D.iloc[0:0]
    if len(ct):
        print("      alfa mediano SOBRE la recta detectada    %.5f" % ct["alfa_sobre_hoja"].median())
        print("      alfa mediano en la banda de control 8 px %.5f" % ct["alfa_banda_control"].median())
        print("      CONTRASTE mediana %.2fx  p25 %.2fx  (limite declarado %.1fx)  ·  %d de %d "
              "lados por encima del limite"
              % (ct["contraste"].median(), ct["contraste"].quantile(0.25), LIM_CONTRASTE,
                 int((ct["contraste"] >= LIM_CONTRASTE).sum()), len(ct)))
        print("      VEREDICTO 1a: %s" % ("CONFIRMADA, la hoja deja senal en el alfa"
                                          if ct["contraste"].median() >= LIM_CONTRASTE
                                          else "NO CONFIRMADA"))
    print()
    print("  1b  Y VALE 0,0006 COMO UMBRAL FIJO?")
    print("      el alfa del fondo (fuera del cuerpo) tiene mediana %.6f y p99,5 %.6f"
          % (D["alfa_fondo_mediana"].median(), D["alfa_fondo_p99_5"].median()))
    print("      0,0006 cae en el percentil %.1f del alfa de fondo, o sea que deja pasar el %.1f %% "
          "del fondo" % (D["pctl_de_0006"].median(), 100.0 - D["pctl_de_0006"].median()))
    print("      el alfa del fondo cambia %.1f veces entre el recorte mas oscuro y el mas claro, "
          "asi que ningun valor fijo cae en el mismo percentil en todos"
          % (D["alfa_fondo_mediana"].max() / max(D["alfa_fondo_mediana"].min(), 1e-12)))
    print("      percentil en que cae 0,0006: mediana %.1f, MINIMO %.2f, maximo %.2f"
          % (D["pctl_de_0006"].median(), D["pctl_de_0006"].min(), D["pctl_de_0006"].max()))
    print("      fraccion del recorte que enciende cada umbral:")
    print("        %-18s %8s %8s %8s   %s" % ("umbral", "mediana", "p90", "maximo", "recta"))
    for nom, et in (("fijo", "0,0006 fijo"), ("adapt", "p99,5 adaptativo"), ("ctrl05", "0,5 control")):
        c = D["frac_recorte_" + nom]
        print("        %-18s %8.4f %8.4f %8.4f   en %d de %d lados (%.1f %%)"
              % (et, c.median(), c.quantile(0.90), c.max(), int(D["recta_" + nom].sum()), len(D),
                 100.0 * D["recta_" + nom].mean()))
    ok_fijo = D["frac_recorte_fijo"].quantile(0.90) <= LIM_FRAC_RECORTE
    print("      VEREDICTO 1b: el umbral fijo 0,0006 %s (limite %.2f de fraccion de recorte, "
          "juzgado por el p90)"
          % ("VALE" if ok_fijo else "NO VALE COMO UMBRAL FIJO: en la cola enciende demasiado",
             LIM_FRAC_RECORTE))
    for nom, et in (("fijo", "0,0006"), ("adapt", "adaptativo")):
        c = D[D["recta_" + nom] == True]
        if len(c):
            print("        con %s: largo mediano %.0f cm, p90 %.0f cm  ·  |angulo - eje| mediana "
                  "%.1f gr (azar declarado 45 gr)"
                  % (et, c["largo_cm_" + nom].median(), c["largo_cm_" + nom].quantile(0.90),
                     c["dif_eje_" + nom].median()))
    print()
    print("  2  SOBREVIVE LA HOJA A LA LIMPIEZA DE MASCARA?")
    sv = D[D["sobrevive_en_mascara"].notna()] if "sobrevive_en_mascara" in D else D.iloc[0:0]
    if len(sv):
        print("      de los pixeles de la hoja detectada, dentro de la mascara limpia del cuerpo: "
              "%.4f de media, mediana %.4f, maximo %.4f"
              % (sv["sobrevive_en_mascara"].mean(), sv["sobrevive_en_mascara"].median(),
                 sv["sobrevive_en_mascara"].max()))
        print("      VEREDICTO 2: %s (limite %.2f)"
              % ("SOBREVIVE" if sv["sobrevive_en_mascara"].mean() >= LIM_SUPERVIVENCIA
                 else "NO SOBREVIVE: la limpieza del cuerpo se come la hoja, y era de esperar porque "
                      "la hoja tiene alfa 0,02-0,5 y `seg` no la etiqueta", LIM_SUPERVIVENCIA))
    cn = D[D["canal_px_bruto"].notna()] if "canal_px_bruto" in D else D.iloc[0:0]
    if len(cn):
        print()
        print("  3  EL CANAL DE HOJA APARTE, con el MISMO reparto por esqueleto y la MISMA guarda:")
        print("      conserva %.4f de los pixeles de hoja (%d px de %d de media por lado)"
              % (cn["canal_conserva"].mean(), int(cn["canal_px_limpio"].mean()),
                 int(cn["canal_px_bruto"].mean())))
        print("      contaminacion por cajas ajenas: %.4f antes del reparto  ->  %.4f despues "
              "(limite %.2f)" % (cn["canal_contamina_bruto"].mean(),
                                 cn["canal_contamina_limpio"].mean(), LIM_CONTAMINA))
        print("      VEREDICTO 3: el canal de hoja %s"
              % ("VALE: conserva la hoja y su contaminacion esta por debajo del limite"
                 if (cn["canal_conserva"].mean() >= LIM_SUPERVIVENCIA
                     and cn["canal_contamina_limpio"].mean() <= LIM_CONTAMINA)
                 else "NO cumple los dos limites a la vez"))
    res = {"clips": CLIPS, "n_lados": int(len(D)), "limites": {
        "contraste": LIM_CONTRASTE, "frac_recorte": LIM_FRAC_RECORTE,
        "supervivencia": LIM_SUPERVIVENCIA, "contaminacion": LIM_CONTAMINA}}
    for c in D.columns:
        if c in ("clip",):
            continue
        try:
            res["mediana_" + c] = float(D[c].median())
            res["media_" + c] = float(D[c].mean())
        except (TypeError, ValueError):
            pass
    (SALIDA / "hoja_y_limpieza.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print()
    print("  tablas -> %s" % SALIDA)


if __name__ == "__main__":
    main()

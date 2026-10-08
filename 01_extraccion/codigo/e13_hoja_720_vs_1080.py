"""ETAPA E, paso 13: EN CUANTOS CLIPS SE PUEDE VER LA HOJA. La tasa por RESOLUCION, medida aparte.

POR QUE ESTE PASO EXISTE Y POR QUE ES EL QUE ACOTA TODO LO DEMAS. Los pasos 10 y 11 construyen el
eje del arma a partir del filo visto en el alfa de `matting`, y eso solo funciona si el filo se ve.
El proyecto ya tenia medido que **se ve en el 42,6 % de los recortes a 1080p y en el 1,6 % a 720p**,
donde el fuerte mide 0,64 px. Y el corpus etiquetado es:

    720p (1280x720) ..... 81,2 %      ~6.950 clips de 8.559
    1080p (1920x1080) ... 18,0 %      ~1.540 clips
    480p (854x480) ......  0,8 %

O sea que **el cono con la hoja VISTA es construible como mucho en una sexta parte del corpus**, y
los cinco clips de los pasos 9-11 se eligieron TODOS a 1080p a proposito. Eso los hace el MEJOR CASO
y no una muestra representativa. Decir «la hoja se ve en el X %» sin decir a que resolucion seria
una cifra inflada, asi que aqui se mide la tasa en las dos y se dan separadas.

LO QUE SE MIDE, con el criterio declarado antes de ejecutar:

  1. TASA DE DETECCION por resolucion: fraccion de recortes con una recta pegada a la mano, y de
     esas, la fraccion que cae DELANTE de la mano (que es la que no puede ser el borde del brazo).
     Metodo identico al de `e5`, extraido sin cambiar un parametro en `cono3d_lib.hoja_en_alfa`.
  2. EL CENSO del corpus por resolucion, que es gratis: sale de las detecciones cacheadas.
  3. LA CONSECUENCIA en numero de clips: cuantos del corpus admiten un cono con hoja vista.
  4. LOS PIXELES POR METRO, que era MI HIPOTESIS de cual es la variable que manda, y que los datos
     NO sostienen: el clip de 720p da 221 px/m (hoja de 1,10 px) y los de 1080p 183-261 px/m (hoja de
     0,9-1,3 px), o sea el MISMO ancho aparente, y aun asi solo a 1080p se alinean las rectas. Se
     sigue midiendo e informando porque acota el problema, pero la causa queda sin identificar. La
     hoja de florete tiene seccion rectangular de unos 4-5 mm, asi que su anchura APARENTE es
     `0,005 * px_por_metro` pixeles: a 200 px/m son 1,0 px y a 100 px/m son 0,5 px. Un plano ANCHO a
     1080p puede dar los mismos px/m que un plano CERRADO a 720p, y entonces la hoja se ve igual de
     mal en los dos. Asi que «1080p» no es la condicion, es un proxy de ella, y se informan las dos
     cosas para poder separarlas.

CRITERIO DECLARADO: para que 720p sea utilizable haria falta una tasa de deteccion por encima del
**10 %** de los recortes, porque por debajo de eso un clip de 342 cuadros daria menos de 34 conos
medidos y el resto tendria que salir del brazo, que apunta 24,8-29,7 grados fuera. Si sale por
debajo del 10 %, **a 720p el cono con hoja vista no existe** y hay que decirlo asi.

Y ESE CRITERIO MIO ESTABA MAL, corregido con su cifra. A 720p la tasa sale **12,9 %**, o sea que
«pasa» el 10 %. Pero el angulo de esas rectas contra el eje del brazo tiene mediana **60,9 grados**
contra un azar de 45: **son PEORES que el azar**, o sea que no son la hoja, son bordes cualesquiera
que casualmente pasan cerca de la mano. Una tasa de deteccion se puede aprobar con ruido; lo que hay
que exigir es la **ALINEACION**, que es lo unico que distingue una hoja de una raya. Asi que el
criterio operativo pasa a ser:

    la resolucion sirve si  |angulo hoja - eje|  esta CLARAMENTE por debajo de los 45 grados de azar

y la tasa de deteccion se informa al lado como dato descriptivo, no como veredicto.

Y LA MEDIDA DE CONSUELO, que es la que de verdad importa si 720p falla: en los clips de 1080p donde
la hoja SI se ve hay verdad-terreno, y contra ella se mide el error de los sustitutos que quedan a
720p (`hombro -> raiz`, `codo -> raiz` y el pulgar). Eso convierte «a 720p no hay hoja» en un numero
de grados, que es lo que hace falta para decidir si el cono sirve de todas formas. Lo hace `e10` en
su careo de ejes; aqui solo se acota CUANTOS clips estan en cada caso.

NO SE USA `pointmap` EN ESTE PASO. La deteccion de la hoja es 2D y no necesita profundidad, asi que
el coste es `matting` mas `seg` por recorte y nada mas.
"""
from __future__ import annotations

import collections
import json
import pathlib
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import capas_limpias as CL2
import cono3d_lib as C3
import cono_lib as CL
import e7_pinta_cono as E7

import cv2

PRUEBAS = CL.PRUEBAS
N_CUADROS = 40
TOL_TASA = 0.10
N_PTS = 308
CLIPS_720 = ["AliveIgnorantBarebirdbat", "BlissfulTiredAiredale"]
CLIPS_1080 = ["SeparateHoarseFeline", "WelloffForcefulIndri"]
"""Dos clips por resolucion, los dos del CORPUS ETIQUETADO y con detecciones cacheadas. Dos por
resolucion no es una muestra: es una COMPROBACION de que la tasa ya medida en el proyecto (1,6 %
contra 42,6 %) se reproduce con este codigo. Si saliera muy distinto, seria el codigo el sospechoso.
"""


def censo():
    """Reparto por resolucion de los clips del corpus con detecciones cacheadas. Cuesta 0 s."""
    corpus = set(x.strip() for x in
                 (C3.SCRATCH / "corpus_etiquetado.txt").read_text(encoding="utf-8").split("\n")
                 if x.strip())
    c = collections.Counter()
    for f in sorted(C3.SCRATCH.glob("det_*.pkl")):
        if f.name[4:-4] not in corpus:
            continue
        with open(f, "rb") as fh:
            _a, _b, W, H, _T = pickle.load(fh)
        c[(int(W), int(H))] += 1
    return c, len(corpus)


capas = CL2.Capas(["pose"])
resumen = {}
cen, n_corpus = censo()
tot = sum(cen.values())
print("=" * 104)
print("CENSO del corpus etiquetado (%d nombres) sobre los %d clips con deteccion cacheada" % (
    n_corpus, tot))
print("=" * 104)
for (W, H), n in sorted(cen.items(), key=lambda kv: -kv[1]):
    print("  %4dx%-4d  %3d clips  %5.1f %%  ->  proyectado al corpus: ~%d clips" % (
        W, H, n, 100.0 * n / tot, round(n_corpus * n / tot)))

for etiq, clips in (("720p", CLIPS_720), ("1080p", CLIPS_1080)):
    for clip in clips:
        mp4 = C3.localiza(clip)
        if not mp4:
            print("no encuentro %s" % clip)
            continue
        cajas, cuadros, W, H, T = C3.carga_det(clip)
        cap = cv2.VideoCapture(mp4)
        frames = []
        while True:
            okf, im = cap.read()
            if not okf:
                break
            frames.append(im)
        cap.release()
        T = min(T, len(frames))
        gris = [cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY) for f in frames[:T]]
        sel = CL2.elige_pareja_del_clip(cajas, cuadros, W, H, T, gris=gris)
        B, pres, mask = sel["B"], sel["pres"], sel["mask"]
        dos = np.array([bool(mask[k] and pres[k].sum() == 2) for k in range(T)])
        print()
        print("=" * 104)
        print("%s  ·  %s  ·  %dx%d  ·  %d cuadros  ·  los dos presentes %.1f %%" % (
            etiq, clip, W, H, T, 100 * float(dos[mask].mean()) if mask.any() else float("nan")))
        print("=" * 104)

        npz = PRUEBAS / ("cono_poses_%s.npz" % clip)
        if npz.is_file():
            with np.load(npz, allow_pickle=True) as z:
                KP, SC = z["kp"], z["sc"]
            print("  poses reutilizadas de %s" % npz.name)
        else:
            KP = np.full((T, 2, N_PTS, 2), np.nan, dtype="float32")
            SC = np.full((T, 2, N_PTS), np.nan, dtype="float32")
            t0 = time.perf_counter()
            for k in range(T):
                cjs, lados = [], []
                for lado in (0, 1):
                    b = B[k, lado]
                    if np.isfinite(b).all() and b[2] > b[0] and b[3] > b[1]:
                        cjs.append(np.array([b[0] * W, b[1] * H, b[2] * W, b[3] * H]))
                        lados.append(lado)
                if not cjs:
                    continue
                out = capas.pose(np.ascontiguousarray(frames[k][:, :, ::-1]), cjs)
                for j, lado in enumerate(lados):
                    kp, sc = out[j]
                    m = min(len(kp), N_PTS)
                    KP[k, lado, :m] = kp[:m, :2]
                    SC[k, lado, :m] = sc[:m]
            np.savez_compressed(npz, res=np.asarray((384, 288)), kp=KP, sc=SC, B=B,
                               presente=pres, marca=(sel["tipo"] == 1), tipo=sel["tipo"],
                               mask=mask, W=W, H=H, T=T, fps=0.0, video=mp4,
                               ms_pose=1000.0 * (time.perf_counter() - t0) / max(T, 1))
            print("  poses calculadas y guardadas en %s" % npz.name)

        mano, pxm = {}, {}
        malo = False
        for lado in (0, 1):
            m, _d = CL.mano_armada(KP, SC, B, W, H, lado)
            if m is None:
                malo = True
                break
            mano[lado] = m
            _p, pxm[lado] = CL.escala_px_m(B, lado, H)
        if malo:
            print("  sin mano armada decidible: clip descartado")
            continue

        ks = [k for k in range(T) if dos[k]]
        if len(ks) < 10:
            print("  solo %d cuadros con los dos: no se mide" % len(ks))
            continue
        paso = max(1, len(ks) // N_CUADROS)
        ks = ks[::paso][:N_CUADROS]
        """Aqui la muestra SI puede ir salteada, al contrario que en `e9`: la tasa de deteccion es
        una proporcion sobre recortes independientes y no un salto entre cuadros contiguos."""

        cap = cv2.VideoCapture(mp4)
        n_rec, n_recta, n_delante, largos, difs = 0, 0, 0, [], []
        t0 = time.perf_counter()
        for k in ks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            okf, im = cap.read()
            if not okf:
                continue
            for lado in (0, 1):
                M = CL.MANOS[mano[lado]]
                ph, pr = KP[k, lado, M["hombro"]], KP[k, lado, M["root"]]
                if not (np.isfinite(ph).all() and np.isfinite(pr).all()
                        and SC[k, lado, M["hombro"]] >= CL.CONF_MIN
                        and SC[k, lado, M["root"]] >= CL.CONF_MIN):
                    continue
                v = pr - ph
                nv = float(np.linalg.norm(v))
                if nv < 1.0:
                    continue
                u = (v / nv).astype("float64")
                vert = pr.astype("float64")
                L = CL.HOJA_M * pxm[lado]
                (cx0, cy0, cx1, cy1), R = C3.recorte_hoja(B[k, lado], B[k, 1 - lado], vert, u, L,
                                                          W, H)
                if cx1 - cx0 < 64 or cy1 - cy0 < 64 or R < 24:
                    continue
                sub = np.ascontiguousarray(im[cy0:cy1, cx0:cx1, ::-1])
                alfa, seg = E7.infiere(sub)
                if float(alfa.mean()) < 0.05:
                    continue
                n_rec += 1
                ln = C3.hoja_en_alfa(cv2, alfa, seg, vert, u, R, (cx0, cy0))
                if ln is None:
                    continue
                n_recta += 1
                if ln["delante"]:
                    n_delante += 1
                    largos.append(float(np.linalg.norm(ln["p2"] - vert)) * 100.0 / pxm[lado])
                    difs.append(C3.dif90(ln["ang"], float(np.degrees(np.arctan2(u[1], u[0])))))
        cap.release()
        dt = time.perf_counter() - t0
        if n_rec == 0:
            print("  0 recortes validos")
            continue
        tasa = n_recta / n_rec
        print("  %d recortes en %.1f s (%.0f ms/recorte, GPU COMPARTIDA)" % (
            n_rec, dt, 1000 * dt / n_rec))
        print("  RECTA pegada a la mano: %d de %d = **%.1f %%**  ->  %s (limite declarado %.0f %%)"
              % (n_recta, n_rec, 100 * tasa,
                 "USABLE" if tasa >= TOL_TASA else "NO USABLE a esta resolucion",
                 100 * TOL_TASA))
        print("  de esas, DELANTE de la mano (no es el borde del brazo): %d (%.1f %%)" % (
            n_delante, 100.0 * n_delante / max(n_recta, 1)))
        if largos:
            print("  alcance proyectado de la hoja: mediana %.0f cm  max %.0f cm  (de 90)  ·  "
                  "|hoja - eje hombro-mano| mediana %.1f gr (azar %.0f)" % (
                      np.median(largos), np.max(largos), np.median(difs), C3.AZAR))
        pxm_med = 0.5 * (pxm[0] + pxm[1])
        print("  PIXELES POR METRO: %.0f y %.0f (mediana %.0f)  ->  la hoja de 5 mm mide **%.2f px** "
              "de ancho aparente" % (pxm[0], pxm[1], pxm_med, 0.005 * pxm_med))
        resumen.setdefault(etiq, {})[clip] = {
            "W": W, "H": H, "pxm": pxm_med, "ancho_hoja_px": 0.005 * pxm_med,
            "n_recortes": n_rec, "n_recta": n_recta, "tasa": tasa,
            "n_delante": n_delante, "ms_por_recorte": 1000 * dt / n_rec,
            "largo_mediana_cm": float(np.median(largos)) if largos else None,
            "dif_hombro_mediana": float(np.median(difs)) if difs else None}

print()
print("=" * 104)
print("RESUMEN: la tasa de deteccion de la hoja POR RESOLUCION")
print("=" * 104)
for etiq in ("720p", "1080p"):
    d = resumen.get(etiq, {})
    if not d:
        continue
    nr = sum(v["n_recta"] for v in d.values())
    nt = sum(v["n_recortes"] for v in d.values())
    px = [v["pxm"] for v in d.values()]
    print("  %-6s  %d de %d recortes con recta = **%.1f %%**  (%d clips)  ·  px/m %.0f-%.0f  ·  "
          "ancho aparente de la hoja %.2f-%.2f px" % (
              etiq, nr, nt, 100.0 * nr / max(nt, 1), len(d), min(px), max(px),
              0.005 * min(px), 0.005 * max(px)))
if "720p" in resumen and "1080p" in resumen:
    t7 = sum(v["n_recta"] for v in resumen["720p"].values()) / max(
        sum(v["n_recortes"] for v in resumen["720p"].values()), 1)
    t10 = sum(v["n_recta"] for v in resumen["1080p"].values()) / max(
        sum(v["n_recortes"] for v in resumen["1080p"].values()), 1)
    print("  razon 1080p / 720p = %.1fx" % (t10 / max(t7, 1e-9)))
    print("  CONSECUENCIA: con el corpus al %.1f %% de 1080p, el cono con hoja VISTA es "
          "construible en ~%d clips de %d" % (
              100.0 * cen.get((1920, 1080), 0) / tot,
              round(n_corpus * cen.get((1920, 1080), 0) / tot), n_corpus))

(PRUEBAS / "cono_e13_hoja_por_resolucion.json").write_text(
    json.dumps({"censo": {"%dx%d" % k: v for k, v in cen.items()}, "n_corpus": n_corpus,
                "clips": resumen}, indent=1, ensure_ascii=False, default=float),
    encoding="utf-8")
print("guardado %s" % (PRUEBAS / "cono_e13_hoja_por_resolucion.json"))

"""CAPA POR CAPA: cual admite la mascara limpia sin romperse y cual no, y con que cifra.

LA PREGUNTA, que no es la de la auditoria
=========================================
`capas_limpias_auditoria.py` mide si la MASCARA es buena. Esto mide algo distinto: si APLICAR esa
mascara a cada una de las cuatro capas densas es una operacion legitima. No es lo mismo, y el usuario
lo pidio explicitamente para la de profundidad: enmascarar una imagen de color y enmascarar un campo
de coordenadas 3D no son la misma cosa.

Enmascarar es SELECCIONAR pixeles, asi que por si mismo no puede cambiar el valor de ningun pixel que
sobreviva. El riesgo real esta en otro sitio y son dos cosas:

  A) EL RELLENO. Si el cero es un valor legal de la capa, rellenar con cero mete datos falsos que
     nadie puede distinguir de los buenos.
  B) EL RECORTE. La mascara se construye por recorte de caja, asi que la capa se ejecuta sobre el
     recorte, no sobre el cuadro. Si la salida de una capa DEPENDE del recorte, entonces "la capa
     limpia" y "la capa del cuadro enmascarada" son dos cosas distintas y hay que decir cual se
     entrega. Esto es lo que hay que medir, capa por capa.

CRITERIOS DECLARADOS ANTES DE EJECUTAR
======================================
  normal    diferencia ANGULAR mediana entre la normal calculada sobre el RECORTE y la calculada
            sobre el CUADRO COMPLETO, en los pixeles de la mascara.
            LIMITE 10 grados, del mismo orden que los 9,42 grados que ya se aceptaron al bajar la
            resolucion de esta misma capa. Por encima de 10 grados, el recorte no es un detalle de
            implementacion: cambia la capa.
            Y una segunda comprobacion: la NORMA de los vectores dentro de la mascara tiene que ser
            1 (son normales unitarias). Si no lo es, `pinta_normal` la esta normalizando y tapando
            una salida que no es unitaria.
  pointmap  diferencia RELATIVA de la z mediana dentro de la mascara entre el recorte y el cuadro
            completo. LIMITE 10 %. Ya estaba medido que ampliar el recorte un 30 % mueve la z entre
            el 30 % y el 694 %, asi que se espera que NO pase; se mide para dejarlo en cifras en
            clips nuevos y para fijar el veredicto de que esta capa solo admite la mascara como
            seleccion sobre el CUADRO COMPLETO.
  seg       fraccion de los pixeles de la mascara cuya clase es 0 (Background). Es la medida de
            coherencia entre la mascara y la propia capa: la mascara es una union con `matting`, asi
            que puede encender pixeles que `seg` llama fondo. Sin limite de aprobado: es una cifra
            que hay que conocer, porque esos pixeles se guardan con clase 0 y el consumidor tiene que
            saber que existen.
  matting   fraccion de los pixeles de la mascara con alfa < 0,5 (los que entran por la via de `seg`
            y no por la de `matting`), y alfa medio dentro y fuera de la mascara. Tambien sin limite:
            el alfa 0 ES el fondo, asi que rellenar con 0 no miente y la capa admite la mascara por
            construccion.

CONTROL FISICO DE `pointmap`, repetido en clips nuevos
======================================================
Ya estaba medido que falla, y con n pequeno. Se repite aqui porque es gratis una vez cargado el
modelo: en un asalto los dos tiradores estan a la misma profundidad dentro de un par de metros y el
arbitro esta varios metros mas cerca, asi que |z(T0) - z(T1)| tiene que ser MUCHO MENOR que
|z(pareja) - z(arbitro)|. Razon esperada >> 1. Se compara contra la misma razon calculada con la
`y del pie`, que cuesta cero.

USO:  python capas_limpias_por_capa.py [N_CUADROS_POR_CLIP]   con el interprete de transformers+CUDA
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

import cv2

SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
SCR = AU.SCR

CLIPS = ["WhisperedTenderDeinonychus", "GrossRelievedDwarfrabbit",
         "DemandingMindlessDeviltasmanian", "SeveralSkinnyGenet"]
"""Cuatro del lote de auditoria, dos de 720p y dos de 1080p. Ninguno es de los tres de ajuste."""

LIM_NORMAL_GR = 10.0
LIM_POINTMAP_REL = 0.10
ALTO_ARBITRO = 0.25
"""Para el control fisico solo vale un ajeno GRANDE: si su caja mide menos del 25 % del alto de
imagen esta lejos y no es el arbitro que tapa a los tiradores."""


def angulo_entre(a, b):
    na = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)
    nb = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9)
    return np.degrees(np.arccos(np.clip((na * nb).sum(axis=-1), -1.0, 1.0)))


def mide_clip(cap, clip, ruta, n_cuadros):
    with open(SCR / ("det_%s.pkl" % clip), "rb") as fh:
        cajas, cuadros, W, H, T = pickle.load(fh)
    sel = CL.elige_pareja_del_clip(cajas, cuadros, W, H, T, gris=AU.gris_del_clip(ruta))
    B, pres, msk = sel["B"], sel["pres"], sel["mask"]
    utiles = [k for k in range(T) if msk[k] and pres[k].all()]
    if len(utiles) < 2:
        return []
    idx = np.unique(np.linspace(0, len(utiles) - 1, min(n_cuadros, len(utiles))).round().astype(int))
    vid = cv2.VideoCapture(ruta)
    filas = []
    for k in [utiles[i] for i in idx]:
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
        n_full = cap.normal(rgb)
        pm_full = cap.pointmap(rgb)
        z_full = pm_full[:, :, 2] if pm_full.ndim == 3 else np.zeros((H, W), "float32")
        zt = {}
        for lado in (0, 1):
            r = CL.mascara_tirador(cap, rgb, sel, k, lado, W, H, kt, con_matting=True)
            if r is None:
                continue
            c, mk = r["c"], r["mask"]
            if mk.sum() < 400:
                continue
            f = {"clip": clip, "k": k, "lado": lado, "W": W, "H": H, "px_mask": int(mk.sum())}

            """seg: coherencia entre la mascara y la propia capa."""
            f["seg_frac_clase0_en_mask"] = float((r["seg"][mk] == 0).mean())
            f["seg_frac_capa_que_la_mascara_tira"] = float(
                1.0 - (mk & (r["seg"] != 0)).sum() / max((r["seg"] != 0).sum(), 1))

            """matting: cuanto de la mascara entra por la via de seg y no por la de alfa."""
            if r["alfa"] is not None:
                f["mat_frac_alfa_bajo_en_mask"] = float((r["alfa"][mk] < CL.UMB_ALFA).mean())
                f["mat_alfa_medio_dentro"] = float(r["alfa"][mk].mean())
                f["mat_alfa_medio_fuera"] = float(r["alfa"][~mk].mean()) if (~mk).any() else np.nan

            """normal: recorte contra cuadro completo, dentro de la mascara."""
            n_c = cap.normal(r["sub"])
            if n_c.ndim == 3 and n_c.shape[:2] == mk.shape:
                n_f = n_full[c[1]:c[3], c[0]:c[2]]
                d = angulo_entre(n_c[mk], n_f[mk])
                d = d[np.isfinite(d)]
                if d.size:
                    f["nor_dif_mediana_gr"] = float(np.median(d))
                    f["nor_dif_p90_gr"] = float(np.percentile(d, 90))
                nn = np.linalg.norm(n_c[mk], axis=-1)
                f["nor_norma_mediana"] = float(np.median(nn))
                f["nor_norma_p5"] = float(np.percentile(nn, 5))

            """pointmap: recorte contra cuadro completo, z mediana dentro de la mascara."""
            pm_c = cap.pointmap(r["sub"])
            zf = float(np.median(z_full[c[1]:c[3], c[0]:c[2]][mk]))
            zt[lado] = zf
            if pm_c.ndim == 3 and pm_c.shape[:2] == mk.shape:
                zc = float(np.median(pm_c[:, :, 2][mk]))
                f["pm_z_recorte"] = zc
                f["pm_z_cuadro"] = zf
                f["pm_dif_rel"] = float(abs(zc - zf) / max(abs(zf), 1e-9))
            filas.append(f)

        """CONTROL FISICO de pointmap, una vez por cuadro y solo si hay un ajeno grande."""
        grandes = [o for o in aj if (o[3] - o[1]) / H >= ALTO_ARBITRO]
        if grandes and len(zt) == 2:
            o = grandes[0]
            y0, y1 = int(max(0, o[1])), int(min(H, o[3]))
            x0, x1 = int(max(0, o[0])), int(min(W, o[2]))
            if y1 - y0 > 8 and x1 - x0 > 8:
                za = float(np.median(z_full[y0:y1, x0:x1]))
                dpar = abs(zt[0] - zt[1])
                darb = abs(0.5 * (zt[0] + zt[1]) - za)
                p0, p1 = float(B[k, 0, 3]), float(B[k, 1, 3])
                pa = o[3] / H
                filas.append({"clip": clip, "k": k, "lado": -1, "ctrl": True,
                              "z_T0": zt[0], "z_T1": zt[1], "z_arb": za,
                              "dz_pareja": dpar, "dz_arbitro": darb,
                              "razon_pointmap": (darb / dpar) if dpar > 1e-9 else np.nan,
                              "razon_pie": abs(0.5 * (p0 + p1) - pa) / max(abs(p0 - p1), 1e-9)})
    vid.release()
    return filas


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    mv = AU.mapa_videos()
    cap = CL.Capas(["pose", "matting", "seg", "normal", "pointmap"])
    todo = []
    for clip in CLIPS:
        if clip not in mv:
            print("  %s no esta en disco" % clip)
            continue
        t0 = time.perf_counter()
        f = mide_clip(cap, clip, mv[clip], n)
        todo += f
        print("  %-36s %3d filas en %.0f s" % (clip, len(f), time.perf_counter() - t0))

    import pandas as pd
    D = pd.DataFrame(todo)
    D.to_csv(SALIDA / "capas_por_capa.csv", index=False)
    L = D[D.get("lado", -1) >= 0] if "lado" in D else D
    C = D[D.get("ctrl", False) == True] if "ctrl" in D else D.iloc[0:0]

    print()
    print("=" * 104)
    print("CAPA POR CAPA: admite la mascara?   %d lados de %d clips" % (len(L), L["clip"].nunique()))
    print("=" * 104)

    print("  seg  ->  la mascara es una SELECCION de etiquetas y el relleno es la clase 0, que ES")
    print("           Background, asi que el cero no miente. ADMITE.")
    print("     pixeles de la mascara con clase 0 (entran por la via de matting): %.4f"
          % L["seg_frac_clase0_en_mask"].mean())
    print("     pixeles con seg != 0 del recorte que la limpieza TIRA (otras personas y fondo): %.4f"
          % L["seg_frac_capa_que_la_mascara_tira"].mean())

    print("  matting ->  el alfa 0 ES el fondo, asi que el relleno con 0 no miente. ADMITE.")
    print("     pixeles de la mascara con alfa < %.1f (entran por la via de seg): %.4f"
          % (CL.UMB_ALFA, L["mat_frac_alfa_bajo_en_mask"].mean()))
    print("     alfa medio DENTRO de la mascara %.4f   FUERA %.4f   razon %.1fx"
          % (L["mat_alfa_medio_dentro"].mean(), L["mat_alfa_medio_fuera"].mean(),
             L["mat_alfa_medio_dentro"].mean() / max(L["mat_alfa_medio_fuera"].mean(), 1e-9)))

    nm, np90 = L["nor_dif_mediana_gr"].median(), L["nor_dif_p90_gr"].median()
    print("  normal ->  son VECTORES: el relleno tiene que ser NaN, no cero, porque un cero se")
    print("           promedia sin avisar en cualquier convolucion y el vector cero no es unitario.")
    print("     dif angular recorte contra cuadro completo, dentro de la mascara: mediana %.2f gr, "
          "p90 %.2f gr   (limite declarado %.0f gr)" % (nm, np90, LIM_NORMAL_GR))
    print("     norma de los vectores dentro de la mascara: mediana %.3f, p5 %.3f (tiene que ser 1)"
          % (L["nor_norma_mediana"].median(), L["nor_norma_p5"].median()))
    print("     VEREDICTO: %s" % ("ADMITE la mascara y ademas el recorte no cambia la capa"
                                  if nm <= LIM_NORMAL_GR else
                                  "ADMITE la mascara PERO el recorte SI cambia la capa: "
                                  "hay que declarar que se entrega la version DE RECORTE"))

    pmd = L["pm_dif_rel"].median()
    print("  pointmap ->  son COORDENADAS: z = 0 es una profundidad legal, asi que el relleno tiene")
    print("           que ser NaN por obligacion, no por gusto.")
    print("     dif relativa de la z mediana entre recorte y cuadro completo: mediana %.1f %%, "
          "p90 %.1f %%   (limite declarado %.0f %%)"
          % (100 * pmd, 100 * L["pm_dif_rel"].quantile(0.90), 100 * LIM_POINTMAP_REL))
    print("     VEREDICTO: %s" % (
        "admite la mascara sobre recorte" if pmd <= LIM_POINTMAP_REL else
        "SOLO admite la mascara como seleccion sobre el CUADRO COMPLETO. Enmascarar un recorte de "
        "esta capa no esta mal en si, pero sus valores no son los mismos que los del cuadro."))

    if len(C):
        print()
        print("  CONTROL FISICO de pointmap en %d cuadros con arbitro grande, clips NUEVOS:" % len(C))
        print("     |z(T0)-z(T1)| mediana %.4f    |z(pareja)-z(arbitro)| mediana %.4f"
              % (C["dz_pareja"].median(), C["dz_arbitro"].median()))
        print("     razon (tiene que ser >> 1): pointmap %.2f   ·   la y del pie, gratis: %.2f"
              % (C["razon_pointmap"].median(), C["razon_pie"].median()))
        print("     cuadros en que la razon supera 1: pointmap %d de %d   ·   y del pie %d de %d"
              % (int((C["razon_pointmap"] > 1).sum()), len(C),
                 int((C["razon_pie"] > 1).sum()), len(C)))

    (SALIDA / "capas_por_capa.json").write_text(json.dumps({
        "clips": CLIPS, "n_lados": int(len(L)),
        "seg_frac_clase0_en_mask": float(L["seg_frac_clase0_en_mask"].mean()),
        "seg_frac_capa_tirada": float(L["seg_frac_capa_que_la_mascara_tira"].mean()),
        "mat_frac_alfa_bajo": float(L["mat_frac_alfa_bajo_en_mask"].mean()),
        "mat_alfa_dentro": float(L["mat_alfa_medio_dentro"].mean()),
        "mat_alfa_fuera": float(L["mat_alfa_medio_fuera"].mean()),
        "nor_dif_mediana_gr": float(nm), "nor_dif_p90_gr": float(np90),
        "nor_norma_mediana": float(L["nor_norma_mediana"].median()),
        "pm_dif_rel_mediana": float(pmd),
        "limites": {"normal_gr": LIM_NORMAL_GR, "pointmap_rel": LIM_POINTMAP_REL},
        "control_fisico": ({"n": int(len(C)),
                            "razon_pointmap": float(C["razon_pointmap"].median()),
                            "razon_pie": float(C["razon_pie"].median()),
                            "pointmap_supera_1": int((C["razon_pointmap"] > 1).sum()),
                            "pie_supera_1": int((C["razon_pie"] > 1).sum())} if len(C) else None),
    }, indent=1), encoding="utf-8")
    print()
    print("  tablas -> %s" % SALIDA)


if __name__ == "__main__":
    main()

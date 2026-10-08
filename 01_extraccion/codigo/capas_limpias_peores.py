"""LOS CUATRO LADOS QUE LA AUDITORIA MARCO: de que se pierde el recall, con la culpa repartida.

De los 320 lados de `capas_limpias_auditoria.py`, TRES quedaron por debajo del limite de 0,70 de
recall y uno justo en el limite. Un recall bajo no dice que arreglar, porque puede venir de tres
sitios distintos y cada uno se arregla de otra forma:

    1 - rec_bruto          `seg` y `matting` no encontraron al tirador. No es culpa de la limpieza:
                           limpiar no puede recuperar pixeles que la capa no encendio.
    rec_bruto - rec_vor    se lo llevo el REPARTO por esqueleto mas cercano, o sea que los pixeles
                           del tirador quedaron mas cerca del esqueleto de otro.
    rec_vor  - rec_kp      se lo llevo la GUARDA DE CAJA.

Los tres suman exactamente `1 - rec_kp`, asi que el reparto es completo y no hay resto sin explicar.

Este script mide esos tres sumandos SOLO en los lados marcados, mas un CONTROL de lados buenos del
mismo clip, porque sin el control no se puede decir si el reparto de culpas de los malos es distinto
del de los buenos. Son 8 llamadas de mascara en total: cuesta una fraccion de la auditoria entera y
contesta la unica pregunta que la auditoria deja abierta.

Guarda ademas la imagen del recorte con la mascara final superpuesta, porque el precedente del
proyecto es que una tabla de nueve metricas dijo "limpio" y una sola imagen lo desmintio.

USO:  python capas_limpias_peores.py     con el interprete de transformers + torch + CUDA
"""
from __future__ import annotations

import json
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import capas_limpias as CL
import capas_limpias_auditoria as AU

import cv2

SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\capas_limpias")
SALIDA.mkdir(parents=True, exist_ok=True)

MARCADOS = [("SeveralSkinnyGenet", 0, 1), ("SeveralSkinnyGenet", 183, 1),
            ("WhisperedTenderDeinonychus", 115, 1), ("LegitimateShoddyBighorn", 400, 0)]
"""Los cuatro peores por recall de la tabla de la auditoria, copiados de su salida tal cual."""

CONTROL = [("SeveralSkinnyGenet", 124, 1), ("WhisperedTenderDeinonychus", 230, 1),
           ("LegitimateShoddyBighorn", 200, 0), ("SeveralSkinnyGenet", 62, 0)]
"""CONTROL, elegido SIN mirar su recall: cuadros del mismo clip y el mismo lado, a media distancia de
los marcados. Sin el, el reparto de culpas de los malos no se puede comparar con nada."""


def mide(cap, mv, casos, etiqueta):
    filas = []
    porclip = {}
    for clip, k, lado in casos:
        porclip.setdefault(clip, []).append((k, lado))
    for clip, cs in porclip.items():
        with open(AU.SCR / ("det_%s.pkl" % clip), "rb") as fh:
            cajas, cuadros, W, H, T = pickle.load(fh)
        sel = CL.elige_pareja_del_clip(cajas, cuadros, W, H, T, gris=AU.gris_del_clip(mv[clip]))
        B = sel["B"]
        vid = cv2.VideoCapture(mv[clip])
        for k, lado in cs:
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
            r = CL.mascara_tirador(cap, rgb, sel, k, lado, W, H, kt, con_matting=True)
            if r is None:
                continue
            f = CL.mide(r, sel, k, lado, W, H, kt[("t", lado)], bgr=im)
            f["clip"], f["grupo"], f["ajenos"] = clip, etiqueta, len(aj)

            """BARRIDO DE LA HOLGURA DE LA GUARDA. No hace falta volver a llamar a los modelos: la
            mascara se recompone con las mismas piezas (`bruto`, `vor`) y solo cambia la guarda, que
            es geometria de cajas. Asi que el barrido cuesta 0 ms de GPU."""
            hh_, ww_ = r["mask"].shape
            bp = sel["B"][k, lado] * np.array([W, H, W, H])
            for hg in (0.00, 0.05, 0.10, 0.20):
                gu = CL.guarda_de_caja(r["c"], bp, aj, hh_, ww_, holgura=hg)
                r2 = dict(r)
                r2["mask"] = r["bruto"] & r["vor"] & gu
                m2 = CL.mide(r2, sel, k, lado, W, H, kt[("t", lado)], bgr=im)
                for c2 in ("rec_kp", "rec_63", "fuga_arb", "residuo", "resto_arb_frac"):
                    f["%s_h%02d" % (c2, int(100 * hg))] = m2[c2]
            c, mk = r["c"], r["mask"]
            sub = im[c[1]:c[3], c[0]:c[2]].copy()
            f["frac_clara_recorte"] = float(
                (cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)[:, :, 2] > 170).mean())
            """`frac_clara` esta aqui porque el modo de fallo ya conocido es el tirador de BLANCO
            sobre suelo o pared CLAROS: si los lados marcados salen con mucha mas fraccion clara que
            los del control, el fallo es de contraste y no de la limpieza."""
            f["frac_seg_encendida"] = float((r["seg"] != 0).mean())
            f["frac_alfa_encendida"] = float((r["alfa"] >= CL.UMB_ALFA).mean()) \
                if r["alfa"] is not None else np.nan
            vis = sub.copy()
            vis[mk] = (0.35 * vis[mk] + 0.65 * np.array([60, 240, 60])).astype("uint8")
            solo_bruto = r["bruto"] & ~mk
            vis[solo_bruto] = (0.35 * vis[solo_bruto]
                              + 0.65 * np.array([60, 60, 255])).astype("uint8")
            comp = np.hstack([sub, vis])
            e = 2 if comp.shape[1] < 700 else 1
            if e > 1:
                comp = cv2.resize(comp, (comp.shape[1] * e, comp.shape[0] * e),
                                  interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(comp, (0, 0), (comp.shape[1], 24), (16, 16, 22), -1)
            cv2.putText(comp, "%s c%d lado %d  ·  %s  ·  rec %.3f (bruto %.3f)  ·  VERDE mascara "
                              "final, ROJO lo que la limpieza quita"
                        % (clip[:22], k, lado, etiqueta, f["rec_kp"], f["rec_bruto"]),
                        (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.imwrite(str(SALIDA / ("peor_%s_c%04d_l%d.png" % (clip, k, lado))), comp)
            filas.append(f)
        vid.release()
    return filas


def main():
    mv = AU.mapa_videos()
    cap = CL.Capas(["pose", "matting", "seg"])
    filas = mide(cap, mv, MARCADOS, "MARCADO") + mide(cap, mv, CONTROL, "control")
    import pandas as pd
    D = pd.DataFrame(filas)
    D.to_csv(AU.SALIDA / "auditoria_peores.csv", index=False)
    print()
    print("=" * 118)
    print("DE QUE SE PIERDE EL RECALL en los lados marcados, contra un control del mismo clip")
    print("=" * 118)
    print(D[["clip", "k", "lado", "grupo", "rec_kp", "rec_63", "rec_bruto", "perdida_capas",
             "perdida_esqueleto", "perdida_guarda", "frac_clara_recorte", "frac_seg_encendida",
             "frac_alfa_encendida", "ajenos"]].to_string(index=False))
    print()
    for g in ("MARCADO", "control"):
        s = D[D["grupo"] == g]
        if not len(s):
            continue
        print("  %-8s  rec %.3f  rec_63 %.3f  ·  culpa: capas %.3f, esqueleto %.3f, guarda %.3f  ·  "
              "fraccion clara del recorte %.3f"
              % (g, s["rec_kp"].mean(), s["rec_63"].mean(), s["perdida_capas"].mean(),
                 s["perdida_esqueleto"].mean(), s["perdida_guarda"].mean(),
                 s["frac_clara_recorte"].mean()))
    print()
    print("  BARRIDO DE LA HOLGURA DE LA GUARDA DE CAJA, sobre los mismos 8 lados y sin GPU.")
    print("  El limite declarado ANTES de mirar: la holgura entra solo si `fuga_arb` se queda en")
    print("  0,05 o menos, que es el limite que ya tenia la fuga.")
    print("    %-8s %8s %8s %9s %9s %9s" % ("holgura", "rec_kp", "rec_63", "fuga_arb", "residuo",
                                            "resto_arb"))
    for hg in (0, 5, 10, 20):
        s = "_h%02d" % hg
        print("    %-8s %8.3f %8.3f %9.4f %9.3f %9.4f"
              % ("%.2f" % (hg / 100.0), D["rec_kp" + s].mean(), D["rec_63" + s].mean(),
                 D["fuga_arb" + s].mean(), D["residuo" + s].mean(),
                 D["resto_arb_frac" + s].mean()))
    mal = D[D["grupo"] == "MARCADO"]
    print("    y SOLO en los 4 lados MARCADOS:")
    for hg in (0, 5, 10, 20):
        s = "_h%02d" % hg
        print("    %-8s %8.3f %8.3f %9.4f %9.3f %9.4f"
              % ("%.2f" % (hg / 100.0), mal["rec_kp" + s].mean(), mal["rec_63" + s].mean(),
                 mal["fuga_arb" + s].mean(), mal["residuo" + s].mean(),
                 mal["resto_arb_frac" + s].mean()))
    (AU.SALIDA / "auditoria_peores.json").write_text(
        json.dumps(D.to_dict("records"), indent=1, default=float), encoding="utf-8")
    print()
    print("  imagenes -> %s" % SALIDA)


if __name__ == "__main__":
    main()

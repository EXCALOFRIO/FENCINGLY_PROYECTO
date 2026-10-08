"""LA MASCARA SALE A TIRAS EN CUERPO A CUERPO, y el recall de keypoints no lo veia.

COMO SE ENCONTRO, que importa mas que el arreglo
================================================
No lo encontro ninguna de las cifras de la auditoria. Lo encontro el coordinador MIRANDO la figura
`02_validacion/muestras/capas_limpias/capas_WhisperedTenderDeinonychus_c0026.png`: la tiradora de la
izquierda, en cuerpo a cuerpo con la otra, sale en la columna DESPUES partida en TIRAS diagonales,
mientras la de la derecha sale entera. El defecto se arrastra a `normal`, `matting` y `pointmap`,
porque las cuatro capas comparten la misma mascara.

Y la auditoria daba `recall de keypoints propios 0,995` en ese mismo cuadro. **No es que la cifra
estuviera mal calculada: es que no puede ver un agujero.** Un keypoint cae en UN pixel, y los
agujeros caen entre punto y punto, asi que una mascara con medio cuerpo comido sigue conteniendo casi
todos los keypoints. Es el mismo tipo de error que ya esta documentado con la confianza de la pose:
un criterio que aprueba todo no esta midiendo nada.

LAS TRES CIFRAS NUEVAS, declaradas antes de ejecutar
====================================================
  conserva_px   de los pixeles que `seg` o `matting` daban por persona DENTRO DE LA CAJA PROPIA,
                fraccion que la mascara limpia conserva. Es el recall de PIXELES, y es el que ve los
                agujeros.        LIMITE: >= 0,90 sin solape del rival y >= 0,80 en cuerpo a cuerpo.
  componentes   componentes conexas de la mascara con al menos el 2 % de su area. Una persona es UNA
                pieza. Se da la MEDIANA y, sobre todo, `troceadas`, la fraccion de lados con TRES o
                mas trozos, porque la mediana es 1 aunque haya casos de cinco y entonces no mide el
                defecto.         LIMITE: `troceadas` <= 0,05.
                                 Dos trozos se admiten: un brazo separado por una oclusion real lo
                                 produce y no es un defecto.

  huecos        area de los agujeros INTERIORES en fraccion de la silueta rellena.
                                 LIMITE: <= 0,05.
  asimetria     |conserva_px(lado 0) - conserva_px(lado 1)| del MISMO cuadro. No necesita verdad
                etiquetada: los dos tiradores son igual de tiradores, asi que si uno conserva el
                95 % y el otro el 55 %, algo se esta comiendo a uno.   LIMITE: <= 0,10.

Y las de siempre siguen midiendose, porque un arreglo que conserve pixeles a cambio de dejar entrar
al arbitro no es un arreglo: `fuga_arb <= 0,05` y `resto_arb`.

UNA SALVEDAD DE `conserva_px`, PARA NO LEERLA AL REVES: en cuerpo a cuerpo su DENOMINADOR incluye
pixeles DEL RIVAL, porque las dos cajas se solapan y el cuerpo del rival cae dentro de la caja
propia. Quitar esos pixeles es exactamente lo que la limpieza tiene que hacer, asi que una parte de
la caida de 0,95 a 0,80 en cuerpo a cuerpo es comportamiento CORRECTO y no un defecto. Lo que no
puede explicarse asi es el TROCEADO: que la mascara salga en cinco piezas con tiras alternas no lo
produce ninguna oclusion. Por eso el veredicto se apoya en `troceadas` y en la ASIMETRIA entre los
dos lados, y `conserva_px` se lee al lado de las dos, no sola.

LAS CUATRO VARIANTES QUE SE CAREAN
==================================
Todas cambian UNA cosa, como se siembra el reparto por esqueleto mas cercano, y nada mas:

  `puntos`             los 308 puntos SUELTOS. Es la version con la que se midio todo lo anterior.
  `puntos63`           los 63 puntos utiles sueltos, sin los 245 de la cara.
  `huesos_sin_cabeza`  el esqueleto como CURVA: los 12 segmentos de tronco, brazos y piernas mas las
                       raices de las manos.
  `huesos`             lo mismo mas la cabeza (los 5 puntos de cara del bloque de cuerpo y el
                       segmento del centro de los hombros a la nariz). Esta variante existe porque la
                       de arriba, medida, hundio `rec_kp` de 0,996 a 0,772 sin mover `conserva_px`:
                       de los 308 puntos 245 son cara, y una semilla sin cabeza regala la careta al
                       esqueleto de al lado.

El mecanismo que se quiere corregir: con semillas sueltas el `distanceTransform` da UNA CELDA POR
PUNTO. En cuerpo a cuerpo la pose del rival coloca puntos encima del cuerpo propio (y su caja tambien
lo cubre, asi que `limpia_keypoints` no los quita), los dos juegos quedan INTERCALADOS y la propiedad
alterna de celda en celda. De ahi las tiras. Un conjunto CONEXO por lado no puede producir tiras
porque su frontera es una curva unica.

EL CAREO NO CUESTA GPU. `seg` y `matting` se llaman UNA vez por lado y las tres variantes se
recomponen encima de las mismas piezas, que son geometria. Asi que las tres se miden sobre
exactamente los mismos pixeles y la comparacion no arrastra ruido de otra tirada.

LOS CUADROS: se eligen por SOLAPE DE LAS DOS CAJAS, mitad con IoU >= 0,10 (cuerpo a cuerpo, que es
donde se vio el defecto) y mitad con IoU <= 0,01 (separados, que es el control). Sin separar los dos
grupos la media global esconde el defecto, que es justo lo que paso.

USO:  python capas_limpias_agujeros.py [N_POR_GRUPO_Y_CLIP]  con el interprete de transformers+CUDA
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

SALIDA = AU.SALIDA
IMAGENES = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\capas_limpias")
IMAGENES.mkdir(parents=True, exist_ok=True)

CLIPS = ["WhisperedTenderDeinonychus", "GrossRelievedDwarfrabbit", "SimilarCluelessEeve",
         "BabyishIdioticCockerspaniel"]
"""Cuatro del lote de auditoria. El primero es el clip donde se vio el defecto."""

VARIANTES = ("puntos", "puntos63", "huesos_sin_cabeza", "huesos")
IOU_CUERPO = 0.10
IOU_SEPARADOS = 0.01
LIM_CONS_SIN = 0.90
LIM_CONS_CON = 0.80
LIM_TROCEADAS = 0.05
LIM_HUECOS = 0.05
LIM_ASIM = 0.10
LIM_FUGA = 0.05


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    return float(i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i))


def elige(sel, W, H, n):
    """Devuelve [(k, grupo)] con n cuadros de cuerpo a cuerpo y n separados, repartidos por el clip."""
    B, pres, msk = sel["B"], sel["pres"], sel["mask"]
    con, sin = [], []
    for k in range(len(B)):
        if not (msk[k] and pres[k].all()):
            continue
        v = iou(B[k, 0], B[k, 1])
        (con if v >= IOU_CUERPO else sin if v <= IOU_SEPARADOS else []).append((k, v))
    out = []
    for lista, grupo in ((con, "cuerpo_a_cuerpo"), (sin, "separados")):
        if not lista:
            continue
        idx = np.unique(np.linspace(0, len(lista) - 1, min(n, len(lista))).round().astype(int))
        out += [(lista[i][0], grupo, lista[i][1]) for i in idx]
    return out


def mide_clip(cap, clip, ruta, n):
    with open(AU.SCR / ("det_%s.pkl" % clip), "rb") as fh:
        cajas, cuadros, W, H, T = pickle.load(fh)
    sel = CL.elige_pareja_del_clip(cajas, cuadros, W, H, T, gris=AU.gris_del_clip(ruta))
    B = sel["B"]
    casos = elige(sel, W, H, n)
    vid = cv2.VideoCapture(ruta)
    filas = []
    for k, grupo, v_iou in casos:
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
        porlado = {}
        for lado in (0, 1):
            r = CL.mascara_tirador(cap, rgb, sel, k, lado, W, H, kt, con_matting=True)
            if r is None or r["bruto"].sum() < 400:
                continue
            c = r["c"]
            hh, ww = r["bruto"].shape
            ajenos_kp = [kt[("t", 1 - lado)]] + [kt[("a", i)] for i in range(len(aj))
                                                 if ("a", i) in kt]
            gu = r["guarda"]
            for md in VARIANTES:
                vor = CL.reparto_esqueleto(kt[("t", lado)], ajenos_kp, W, H,
                                           modo=md)[c[1]:c[3], c[0]:c[2]]
                r2 = dict(r)
                r2["vor"] = vor
                r2["mask"] = r["bruto"] & vor & gu
                m = CL.mide(r2, sel, k, lado, W, H, kt[("t", lado)], bgr=im)
                m.update({"clip": clip, "grupo": grupo, "iou_rival": v_iou, "variante": md,
                          "ajenos": len(aj)})
                filas.append(m)
                porlado.setdefault(md, {})[lado] = (m, r2["mask"], c)
        for md, dd in porlado.items():
            if len(dd) == 2:
                a = abs(dd[0][0]["conserva_px"] - dd[1][0]["conserva_px"])
                for lado in (0, 1):
                    dd[lado][0]["asimetria"] = a
        if grupo == "cuerpo_a_cuerpo" and porlado:
            """Una imagen por cuadro de cuerpo a cuerpo con las tres variantes lado a lado, porque el
            defecto se encontro mirando y el arreglo tiene que poder juzgarse igual."""
            tiras = []
            for md in VARIANTES:
                if md not in porlado:
                    continue
                lienzo = np.zeros_like(im)
                for lado, (_m, mk, c) in porlado[md].items():
                    reg = lienzo[c[1]:c[3], c[0]:c[2]]
                    col = np.array([60, 240, 60]) if lado == 0 else np.array([60, 160, 255])
                    reg[mk] = col
                cons = " / ".join("%.2f" % porlado[md][l][0]["conserva_px"]
                                  for l in sorted(porlado[md]))
                comp = " / ".join("%d" % porlado[md][l][0]["componentes"]
                                  for l in sorted(porlado[md]))
                cv2.rectangle(lienzo, (0, 0), (lienzo.shape[1], 40), (16, 16, 22), -1)
                cv2.putText(lienzo, "%-9s  conserva_px %s  ·  componentes %s" % (md, cons, comp),
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2,
                            cv2.LINE_AA)
                tiras.append(cv2.resize(lienzo, (900, int(900 * H / float(W)))))
            if tiras:
                fig = np.hstack([cv2.resize(im, (900, int(900 * H / float(W))))] + tiras)
                cv2.imwrite(str(IMAGENES / ("reparto_%s_c%04d.png" % (clip, k))), fig)
    vid.release()
    return filas


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
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
        print("  %-34s %3d filas en %.0f s" % (clip, len(f), time.perf_counter() - t0))

    import pandas as pd
    D = pd.DataFrame(todo)
    D.to_csv(SALIDA / "capas_agujeros.csv", index=False)
    print()
    print("=" * 122)
    print("REPARTO POR ESQUELETO: TRES VARIANTES  ·  %d lados de %d clips, %d cuadros"
          % (len(D) // len(VARIANTES), D["clip"].nunique(), D["k"].nunique()))
    print("=" * 122)
    print("%-18s %-16s %5s %9s %7s %9s %8s %8s %8s %8s %8s"
          % ("variante", "grupo", "n", "conserva", "compon", "troceadas", "huecos", "asimetr",
             "rec_kp", "fuga_arb", "residuo"))
    res = {}
    for md in VARIANTES:
        for g in ("separados", "cuerpo_a_cuerpo"):
            s = D[(D["variante"] == md) & (D["grupo"] == g)]
            if not len(s):
                continue
            fila = {"n": int(len(s)), "conserva_px": float(s["conserva_px"].mean()),
                    "componentes": float(s["componentes"].median()),
                    "troceadas": float((s["componentes"] >= 3).mean()),
                    "componentes_max": int(s["componentes"].max()),
                    "huecos": float(s["huecos"].mean()),
                    "asimetria": float(s["asimetria"].mean()) if "asimetria" in s else np.nan,
                    "rec_kp": float(s["rec_kp"].mean()), "rec_63": float(s["rec_63"].mean()),
                    "fuga_arb": float(s["fuga_arb"].mean()),
                    "residuo": float(s["residuo"].mean()),
                    "resto_arb_frac": float(s["resto_arb_frac"].mean())}
            res["%s|%s" % (md, g)] = fila
            print("%-18s %-16s %5d %9.3f %7.1f %9.3f %8.4f %8.3f %8.3f %8.4f %8.3f"
                  % (md, g, fila["n"], fila["conserva_px"], fila["componentes"], fila["troceadas"],
                     fila["huecos"], fila["asimetria"], fila["rec_kp"], fila["fuga_arb"],
                     fila["residuo"]))
    print()
    print("  VEREDICTO por variante, contra los limites declarados antes de ejecutar")
    print("  (troceadas <= %.2f, asimetria <= %.2f, huecos <= %.2f, fuga_arb <= %.2f, y conserva "
          ">= %.2f en separados; en cuerpo a cuerpo `conserva` NO es limite, ver el docstring):"
          % (LIM_TROCEADAS, LIM_ASIM, LIM_HUECOS, LIM_FUGA, LIM_CONS_SIN))
    for md in VARIANTES:
        a = res.get("%s|separados" % md)
        b = res.get("%s|cuerpo_a_cuerpo" % md)
        if a is None or b is None:
            continue
        fallos = []
        if a["conserva_px"] < LIM_CONS_SIN:
            fallos.append("conserva separados %.3f" % a["conserva_px"])
        if b["troceadas"] > LIM_TROCEADAS:
            fallos.append("troceadas %.3f (max %d trozos)" % (b["troceadas"],
                                                              b["componentes_max"]))
        if b["huecos"] > LIM_HUECOS:
            fallos.append("huecos %.4f" % b["huecos"])
        if b["asimetria"] > LIM_ASIM:
            fallos.append("asimetria %.3f" % b["asimetria"])
        if b["fuga_arb"] > LIM_FUGA:
            fallos.append("fuga_arb %.4f" % b["fuga_arb"])
        print("    %-18s %-8s %s" % (md, "PASA" if not fallos else "NO PASA",
                                     "" if not fallos else "falla: " + ", ".join(fallos)))
    print()
    print("  Y EL CAREO QUE DECIDE, sobre los mismos pixeles: `puntos` contra `huesos`")
    for g in ("separados", "cuerpo_a_cuerpo"):
        a, b = res.get("puntos|" + g), res.get("huesos|" + g)
        if a and b:
            print("    %-16s conserva %.3f -> %.3f  ·  troceadas %.3f -> %.3f  ·  asimetria "
                  "%.3f -> %.3f  ·  rec_kp %.3f -> %.3f  ·  fuga_arb %.4f -> %.4f"
                  % (g, a["conserva_px"], b["conserva_px"], a["troceadas"], b["troceadas"],
                     a["asimetria"], b["asimetria"], a["rec_kp"], b["rec_kp"], a["fuga_arb"],
                     b["fuga_arb"]))
    print()
    print("  LO QUE `rec_kp` NO VE, en cifras: en cuerpo a cuerpo con la variante `puntos` el recall")
    print("  de keypoints es %.3f y el de PIXELES %.3f. La diferencia es el agujero."
          % (res.get("puntos|cuerpo_a_cuerpo", {}).get("rec_kp", float("nan")),
             res.get("puntos|cuerpo_a_cuerpo", {}).get("conserva_px", float("nan"))))
    peor = D[(D["variante"] == "puntos")].nsmallest(6, "conserva_px")
    print()
    print(peor[["clip", "k", "lado", "grupo", "iou_rival", "conserva_px", "componentes", "huecos",
                "rec_kp"]].to_string(index=False))
    (SALIDA / "capas_agujeros.json").write_text(
        json.dumps({"variantes": list(VARIANTES), "limites": {
            "conserva_separados": LIM_CONS_SIN, "conserva_cuerpo": LIM_CONS_CON,
            "troceadas": LIM_TROCEADAS, "huecos": LIM_HUECOS, "asimetria": LIM_ASIM,
            "fuga_arb": LIM_FUGA}, "tabla": res}, indent=1), encoding="utf-8")
    print()
    print("  tabla -> %s   ·   imagenes -> %s" % (SALIDA, IMAGENES))


if __name__ == "__main__":
    main()

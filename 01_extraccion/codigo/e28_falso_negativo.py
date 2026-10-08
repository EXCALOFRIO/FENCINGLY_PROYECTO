"""ETAPA E, paso 28: EL FALSO NEGATIVO QUE VIO EL USUARIO, y de cual de los dos fallos es.

El usuario, mirando el campo de `AdorableFittingAsiansmallclawedotter` c64: *"el de la izquierda la
clava practicamente perfecto, pero el de la derecha no la detecta, y si se ve perfectamente, un poco
doblada"*. Comprobado en el dato: en ese cuadro el lado 0 tiene `campo_origen = 0` y el lado 1 tiene
`1`.

POR QUE ESTE CASO VALE Y CAMBIA EL TECHO DEL PROBLEMA. La explicacion del 3 % de hojas vistas era
«la hoja casi no esta»: desenfoque en la accion y sub-pixel a 720p. Este caso la rompe: es **1080p**,
**el usuario la ve** y ademas describe la flexion, y **la del otro tirador SI se detecto en el mismo
cuadro**. Luego no es el cuadro, ni el desenfoque global, ni la compresion: es **esa hoja**. O sea
que al menos parte del 97 % es **recall del detector** y no ausencia de senial.

LOS DOS FALLOS POSIBLES SON DISTINTOS Y SE ARREGLAN DISTINTO, y esto es lo que mide este paso:

  F-ALFA     el `matting` no marca la hoja. Entonces el arreglo es de la capa densa (umbral,
             resolucion de entrada, otro modelo) y el detector de rectas no tiene nada que hacer.
  F-FORMA    el alfa SI la marca y la descarta el ajuste. Hay tres puertas que pueden tirarla:
               `m.sum() < 20`            pocos pixeles por encima del umbral
               `HoughLinesP` no devuelve nada, o nada que llegue a `MIN_LARGO * R`
               `CERCA_ROOT`: la recta no pasa a menos de 0,12 R del vertice
             Y la hipotesis del usuario cae aqui: **si la hoja esta DOBLADA, no es una recta**, y
             `HoughLinesP` busca rectas. La flexion es maxima con tension, o sea justo cuando la
             hoja mas informa, lo que explicaria que se detecten mas fuera de la accion.

CRITERIO DECLARADO ANTES DE MIRAR, para no contar una cosa por otra: se considera que el alfa SI
marca la hoja si en el corredor de la hoja del lado 1 hay al menos **20 pixeles** por encima del
umbral `p99,5` (el mismo numero que ya usa la puerta de `e7.detecta`), y se declara `F-FORMA`. Si hay
menos, es `F-ALFA`.

EL CORREDOR DE LA HOJA DEL LADO 1 no se elige a ojo: se toma el segmento que va del vertice del lado
1 al vertice del lado 0, ensanchado 12 px a cada lado. En un cruce de hierros con las dos manos
enfrentadas, la hoja de cada uno va hacia la del otro, asi que ese corredor la contiene sin que haya
que senialarla con el raton. Va dicho porque es lo unico de este paso que no es automatico.

SOLO SE MIRA UN CUADRO Y NO SE SACA NINGUNA TASA DE AQUI. Este clip es el que aportaba 24 de las 31
hojas de la medida vieja y **no esta en `corpus_etiquetado.txt`**. Vale como caso de estudio para
entender el fallo y no vale para dar una tasa; las tasas salen de los 12 asaltos de `e27`.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

CACHE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565\scratchpad\hoja_seg\hf")
os.environ["HF_HOME"] = str(CACHE)

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cono3d_lib as C3
import cono_lib as CL
import e20_estado_del_cono as E20
import e7_pinta_cono as E7

PRUEBAS = CL.PRUEBAS
SAL = pathlib.Path(
    r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\cono_arreglado")
CLIP = "AdorableFittingAsiansmallclawedotter"
K = 64
MIN_PX = 20
ANCHO_CORREDOR = 12.0


def geometria(clip):
    d = E20.carga(clip)
    kp, sc, B, W, H = d["kp"], d["sc"], d["B"], d["W"], d["H"]
    g = {}
    for lado in (0, 1):
        mano, _x = CL.mano_armada(kp, sc, B, W, H, lado)
        vert, u, ok, _lg = C3.eje_brazo_2d(kp, sc, lado, mano, "codo")
        _pk, pxm = C3.escala_px_m_de_pie(B, lado, H)
        g[lado] = {"vert": vert, "u": u, "ok": ok, "pxm": pxm, "mano": mano}
    return g, d


def diagnostica(alfa, seg, vert, cx0, cy0, R):
    """Repite EXACTAMENTE `e7.detecta` pero contando por que se cae cada candidato."""
    cuerpo = cv2.dilate((seg != 0).astype("uint8"),
                        np.ones((2 * E7.DILATA + 1,) * 2, np.uint8)) > 0
    fuera = alfa[~cuerpo]
    inf = {"n_fuera": int(fuera.size)}
    if fuera.size < 500:
        inf["fallo"] = "menos de 500 px fuera del cuerpo: no hay fondo para el umbral"
        return None, inf
    t = float(np.percentile(fuera, E7.PCTL))
    hh, ww = alfa.shape[:2]
    GY, GX = np.mgrid[0:hh, 0:ww]
    disco = np.hypot(GX + cx0 - vert[0], GY + cy0 - vert[1]) <= R
    m = disco & (~cuerpo) & (alfa > t)
    inf.update({"umbral": t, "n_en_mascara": int(m.sum()), "R": float(R),
                "min_largo_px": float(max(8, E7.MIN_LARGO * R)),
                "cerca_root_px": float(E7.CERCA_ROOT * R)})
    if m.sum() < MIN_PX:
        inf["fallo"] = "la mascara tiene %d px, menos de %d" % (m.sum(), MIN_PX)
        return m, inf
    lines = cv2.HoughLinesP((m * 255).astype("uint8"), 1, np.pi / 360.0, 12,
                            minLineLength=int(max(8, E7.MIN_LARGO * R)),
                            maxLineGap=int(max(2, 0.05 * R)))
    if lines is None:
        inf["fallo"] = "HoughLinesP no devuelve ninguna recta de al menos %.0f px" % (
            max(8, E7.MIN_LARGO * R))
        inf["n_lineas"] = 0
        return m, inf
    cand, rech = [], []
    for x1_, y1_, x2_, y2_ in np.asarray(lines).reshape(-1, 4):
        p1 = np.array([x1_ + cx0, y1_ + cy0], dtype="float64")
        p2 = np.array([x2_ + cx0, y2_ + cy0], dtype="float64")
        v = p2 - p1
        lg = float(np.linalg.norm(v))
        if lg < 1:
            continue
        nv = np.array([-v[1], v[0]]) / lg
        d = abs(float(nv @ (vert - p1)))
        if d > E7.CERCA_ROOT * R:
            rech.append({"largo": lg, "dist_vertice": d})
            continue
        cand.append((lg, p1, p2))
    inf["n_lineas"] = len(lines)
    inf["n_rechazadas_por_cerca_root"] = len(rech)
    inf["rechazadas"] = sorted(rech, key=lambda r: -r["largo"])[:6]
    if not cand:
        inf["fallo"] = ("las %d rectas existen pero NINGUNA pasa a menos de %.1f px del vertice; "
                        "la mas cercana esta a %.1f px" % (
                            len(lines), E7.CERCA_ROOT * R,
                            min(r["dist_vertice"] for r in rech) if rech else float("nan")))
        return m, inf
    cand.sort(key=lambda c: -c[0])
    inf["fallo"] = None
    inf["elegida"] = {"largo": cand[0][0], "p1": cand[0][1].tolist(), "p2": cand[0][2].tolist()}
    return m, inf


def corredor(v0, v1, forma, ancho=ANCHO_CORREDOR):
    """Mascara del corredor entre las dos manos, que es donde tiene que estar la hoja del cruce."""
    hh, ww = forma
    GY, GX = np.mgrid[0:hh, 0:ww]
    d = v0 - v1
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return np.zeros(forma, bool)
    u = d / n
    px = GX - v1[0]
    py = GY - v1[1]
    t = px * u[0] + py * u[1]
    perp = np.abs(-px * u[1] + py * u[0])
    return (t >= 0) & (t <= n) & (perp <= ancho)


def main():
    print("=" * 108)
    print("EL FALSO NEGATIVO DE %s c%d  ·  ¿F-ALFA o F-FORMA?" % (CLIP, K))
    print("=" * 108)
    g, d = geometria(CLIP)
    W, H = d["W"], d["H"]
    cap = cv2.VideoCapture(d["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, K)
    okf, im = cap.read()
    cap.release()
    if not okf:
        raise SystemExit("no se pudo leer el cuadro")
    res, vis = {}, {}
    for lado in (0, 1):
        gl = g[lado]
        vert, u = gl["vert"][K], gl["u"][K]
        L = C3.ALCANCE_M * gl["pxm"]
        (cx0, cy0, cx1, cy1), R = E7.recorte(d["B"][K, lado], d["B"][K, 1 - lado],
                                             vert, u, L, W, H)
        sub = np.ascontiguousarray(im[cy0:cy1, cx0:cx1, ::-1])
        alfa, seg = E7.infiere(sub)
        m, inf = diagnostica(alfa, seg, vert, cx0, cy0, R)
        inf["lado"] = lado
        inf["recorte"] = [cx0, cy0, cx1, cy1]
        print()
        print("  LADO %d  ·  mano %s  ·  recorte %dx%d  ·  R %.0f px  ·  umbral p99,5 %.5f" % (
            lado, gl["mano"], cx1 - cx0, cy1 - cy0, R, inf.get("umbral", float("nan"))))
        print("    px en la mascara %d  ·  rectas de Hough %s  ·  rechazadas por cerca_root %s" % (
            inf.get("n_en_mascara", -1), inf.get("n_lineas", "-"),
            inf.get("n_rechazadas_por_cerca_root", "-")))
        print("    veredicto: %s" % (
            "DETECTA, largo %.0f px" % inf["elegida"]["largo"] if not inf.get("fallo")
            else "NO detecta: %s" % inf["fallo"]))
        for r in inf.get("rechazadas", [])[:3]:
            print("      rechazada: largo %.0f px, a %.1f px del vertice (limite %.1f)" % (
                r["largo"], r["dist_vertice"], inf["cerca_root_px"]))
        res[lado] = inf
        vis[lado] = (alfa, seg, m, (cx0, cy0, cx1, cy1), R)

    """EL CRITERIO DECLARADO SE CAMBIA, Y SE DICE POR QUE. El corredor mano1 -> mano0 daba 0 px,
    pero mirando la figura **el corredor estaba mal especificado**: la hoja del lado 1 no va hacia la
    mano del rival, va hacia ARRIBA Y A LA IZQUIERDA, por encima de el. O sea que mi suposicion de
    «en un cruce de hierros la hoja de cada uno va hacia la del otro» es falsa en este cuadro, y
    encaja con lo que 24.9 ya habia medido: el eje del arma **no apunta al rival**.

    El sustituto no supone ninguna direccion: se compara los pixeles que la mascara marca DENTRO del
    disco con los que se esperarian solo por la cola del percentil. `t` es el p99,5 del alfa fuera
    del cuerpo, asi que por construccion el 0,5 % de esos pixeles lo pasa SIEMPRE, haya hoja o no.

        esperados por la cola = 0,005 * (area del disco fuera del cuerpo)

    CRITERIO DECLARADO: la mascara lleva senial si los observados son al menos **3x** los esperados.
    Por debajo, lo que hay es la cola del ruido del fondo y el alfa NO marca la hoja."""
    print()
    print("  EL CRITERIO, sin suponer direccion: observados contra la cola del percentil")
    print("  %-6s %10s %12s %10s %9s %11s %11s" % (
        "lado", "disco", "disco s/cuerpo", "esperados", "observ.", "razon", "alfa max"))
    for lado in (0, 1):
        alfa, seg, m, (ax0, ay0, ax1, ay1), R = vis[lado]
        cuerpo = cv2.dilate((seg != 0).astype("uint8"),
                            np.ones((2 * E7.DILATA + 1,) * 2, np.uint8)) > 0
        hh, ww = alfa.shape[:2]
        GY, GX = np.mgrid[0:hh, 0:ww]
        v = g[lado]["vert"][K]
        disco = np.hypot(GX + ax0 - v[0], GY + ay0 - v[1]) <= R
        libre = disco & (~cuerpo)
        esp = 0.005 * float(libre.sum())
        obs = float(m.sum()) if m is not None else 0.0
        raz = obs / max(esp, 1e-9)
        amax = float(np.max(alfa[libre])) if libre.any() else float("nan")
        print("  %-6d %10d %12d %10.0f %9.0f %10.2fx %11.4f" % (
            lado, int(disco.sum()), int(libre.sum()), esp, obs, raz, amax))
        res[lado]["cola"] = {"esperados": esp, "observados": obs, "razon": raz, "alfa_max": amax}
    r0 = res[0]["cola"]["razon"]
    r1 = res[1]["cola"]["razon"]
    veredicto = "F-FORMA" if r1 >= 3.0 else "F-ALFA"
    print("    lado 0 (que SI se detecta) razon **%.2fx**  ·  lado 1 (el fallo) razon **%.2fx**  ·  "
          "limite declarado 3x" % (r0, r1))
    print("    -> **%s**: %s" % (
        veredicto,
        "el alfa SI marca la hoja y la descarta el ajuste de recta"
        if veredicto == "F-FORMA" else
        "el alfa NO marca la hoja: lo que pasa el umbral es la cola del fondo, y el fallo es de la "
        "capa densa"))
    res["criterio"] = {"razon_lado0": r0, "razon_lado1": r1, "limite": 3.0,
                       "veredicto": veredicto,
                       "corredor_retirado": ("el corredor mano1 -> mano0 daba 0 px pero estaba mal "
                                             "especificado: la hoja no va hacia la mano del rival")}

    """Y EL FONDO, que es la diferencia visible entre los dos lados: la hoja del lado 1 esta sobre el
    fondo NEGRO de la grada y la del lado 0 sobre la pista clara. Una hoja oscura sobre fondo oscuro
    no da contraste, y `matting` estima transparencia, no bordes."""
    print()
    print("  EL FONDO DETRAS DE LA HOJA, que es lo que distingue a los dos lados")
    for lado in (0, 1):
        alfa, seg, m, (ax0, ay0, ax1, ay1), R = vis[lado]
        sub = np.ascontiguousarray(im[ay0:ay1, ax0:ax1, ::-1])
        cuerpo = cv2.dilate((seg != 0).astype("uint8"),
                            np.ones((2 * E7.DILATA + 1,) * 2, np.uint8)) > 0
        hh, ww = alfa.shape[:2]
        GY, GX = np.mgrid[0:hh, 0:ww]
        v = g[lado]["vert"][K]
        libre = (np.hypot(GX + ax0 - v[0], GY + ay0 - v[1]) <= R) & (~cuerpo)
        gr = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY).astype("float32")[libre]
        print("  lado %d: gris del fondo en el disco  p50 **%.0f**  p10 %.0f  p90 %.0f  ·  "
              "fraccion por debajo de 60 **%.0f %%**" % (
                  lado, np.median(gr), np.percentile(gr, 10), np.percentile(gr, 90),
                  100 * float((gr < 60).mean())))
        res[lado]["fondo"] = {"gris_p50": float(np.median(gr)),
                              "frac_oscuro": float((gr < 60).mean())}

    """DONDE MARCA `matting` LA HOJA CUANDO LA MARCA. `matting` no es un detector de bordes: estima
    la transparencia de LA PERSONA. Asi que una hoja solo recibe alfa si el modelo la considera parte
    de la silueta del sujeto, y eso deberia pasar mas cerca del cuerpo que lejos."""
    print()
    print("  DONDE ESTAN LOS PIXELES QUE `matting` MARCA, en distancia al cuerpo")
    for lado in (0, 1):
        alfa, seg, m, (ax0, ay0, ax1, ay1), R = vis[lado]
        if m is None or not m.any():
            continue
        cue = (seg != 0).astype("uint8")
        dist = cv2.distanceTransform(
            1 - cv2.dilate(cue, np.ones((2 * E7.DILATA + 1,) * 2, np.uint8)), cv2.DIST_L2, 3)
        dm = dist[m]
        print("  lado %d: %4d px marcados  ·  distancia al cuerpo p50 **%.0f px**  p90 %.0f  "
              "max %.0f  ·  alfa p50 %.4f" % (
                  lado, int(m.sum()), np.median(dm), np.percentile(dm, 90), dm.max(),
                  float(np.median(alfa[m]))))
        res[lado]["dist_cuerpo"] = {"p50": float(np.median(dm)),
                                    "p90": float(np.percentile(dm, 90)),
                                    "max": float(dm.max()),
                                    "alfa_p50": float(np.median(alfa[m]))}
    print("  -> si los marcados del lado 0 se pegan al cuerpo, `matting` esta extendiendo la silueta")
    print("     y no viendo la hoja como objeto suelto; entonces una hoja lejos del cuerpo no se")
    print("     marca nunca, y eso NO se arregla con un ajuste de curva.")

    """LA CURVATURA, y por que la hipotesis del usuario NO puede ser la causa de ESTE fallo: el fallo
    es anterior al ajuste. Si el alfa no marca la hoja, da igual buscar una recta o una parabola. Se
    mide en el rastro del lado 0 igualmente, porque si el rastro marcado fuera curvo el ajuste de
    recta si estaria limitando el LARGO que se recupera de cada hoja."""
    alfa, seg, m0, (bx0, by0, bx1, by1), R0 = vis[0]
    ys, xs = np.nonzero(m0)
    if xs.size >= 12:
        P = np.stack([xs.astype("float64"), ys.astype("float64")], axis=1)
        Q = P - P.mean(axis=0)
        _u, _s, vt = np.linalg.svd(Q, full_matrices=False)
        s, r = Q @ vt[0], Q @ vt[1]
        rec = float(np.sqrt(np.mean(r ** 2)))
        A = np.stack([s ** 2, s, np.ones_like(s)], axis=1)
        coef, *_ = np.linalg.lstsq(A, r, rcond=None)
        cur = float(np.sqrt(np.mean((r - A @ coef) ** 2)))
        largo = float(s.max() - s.min())
        flecha = abs(float(coef[0])) * (largo / 2.0) ** 2
        print()
        print("  LA CURVATURA DEL RASTRO DEL LADO 0, el que si se detecta (%d px)" % xs.size)
        print("    largo %.0f px  ·  residuo de RECTA %.2f px  ·  de PARABOLA %.2f px  ·  mejora "
              "**%.2fx**  ·  flecha %.1f px (%.1f %% del largo)" % (
                  largo, rec, cur, rec / max(cur, 1e-9), flecha,
                  100 * flecha / max(largo, 1e-9)))
        print("    -> %s" % (
            "el rastro es CURVO y una recta no lo describe"
            if rec / max(cur, 1e-9) >= 1.5 else
            "el rastro es practicamente recto: la curvatura no limita el ajuste en este caso"))
        res["curvatura_lado0"] = {"n_px": int(xs.size), "largo_px": largo, "residuo_recta": rec,
                                  "residuo_parabola": cur, "mejora": rec / max(cur, 1e-9),
                                  "flecha_px": flecha}

    cel = []
    for lado in (0, 1):
        alfa, seg, m, (ax0, ay0, ax1, ay1), R = vis[lado]
        sub = np.ascontiguousarray(im[ay0:ay1, ax0:ax1])
        hm = np.clip((np.log10(np.clip(alfa, 1e-6, 1.0)) + 4.0) / 4.0, 0, 1)
        p = cv2.applyColorMap((hm * 255).astype("uint8"), cv2.COLORMAP_TURBO)
        if m is not None and m.any():
            ov = np.zeros_like(p)
            ov[m] = (255, 255, 255)
            p = cv2.addWeighted(p, 0.72, ov, 0.28, 0)
        v = g[lado]["vert"][K] - np.array([ax0, ay0])
        cv2.circle(p, np.int32(v), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(sub, np.int32(v), 7, (40, 255, 255), -1, cv2.LINE_AA)
        est = "DETECTED" if not res[lado].get("fallo") else "NOT DETECTED"
        cel.append(E7.celda(sub, "side %d | %s | the blade IS VISIBLE in the image" % (lado, est),
                            "yellow dot = wrist (cone vertex)", 620, 470))
        cel.append(E7.celda(p, "side %d | matting alpha (log scale) + px above threshold" % lado,
                            "max alpha inside the disc %.4f | observed/noise-tail %.2fx" % (
                                res[lado]["cola"]["alfa_max"], res[lado]["cola"]["razon"]),
                            620, 470))
    fig = np.vstack([np.hstack(cel[:2]), np.hstack(cel[2:])])
    bl = np.zeros((54, fig.shape[1], 3), np.uint8)
    bl[:] = (14, 14, 20)
    cv2.putText(bl, "FALSE NEGATIVE SPOTTED BY THE USER | %s | frame %d | verdict %s (the dense "
                    "layer, not the line fit)" % (
                        CLIP, K, res["criterio"]["veredicto"].replace("ALFA", "ALPHA")), (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(bl, "`matting` marks one blade and not the other in the SAME frame: alpha %.4f vs "
                    "%.4f, factor %.0f. Both blades are visible and over the same dark background."
                    % (res[0]["cola"]["alfa_max"], res[1]["cola"]["alfa_max"],
                       res[0]["cola"]["alfa_max"] / max(res[1]["cola"]["alfa_max"], 1e-9)),
                (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150, 220, 255), 1, cv2.LINE_AA)
    SAL.mkdir(parents=True, exist_ok=True)
    fp = SAL / ("falso_negativo_%s_c%04d.png" % (CLIP[:24], K))
    cv2.imwrite(str(fp), np.vstack([fig, bl]))
    print()
    print("  figura %s" % fp.name)
    (PRUEBAS / "cono_e28_falso_negativo.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("  guardado %s" % (PRUEBAS / "cono_e28_falso_negativo.json"))


if __name__ == "__main__":
    main()

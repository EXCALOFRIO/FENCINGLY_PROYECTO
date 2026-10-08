"""FIGURAS ANTES / DESPUES de la limpieza, una por cuadro, para juzgarlo a ojo sin leer nada.

QUE PIDIO EL USUARIO, con sus palabras: "hay que limpiarlo, para que por ejemplo en el de profundidad
solo detecte a los floretistas y no al arbitro, y eso, para borrarlo de lo de profundidad tambien, y
el fondo". Asi que la figura tiene que dejar ver DE UN GOLPE que el arbitro ha desaparecido de las
cuatro capas, y para eso las dos versiones van EN PAREJA en la misma fila.

LAS DIEZ CELDAS
===============
  fila 0   original con los dos esqueletos de 308 puntos y las cajas (verde izquierda, naranja
           derecha, ROJO las detecciones NO elegidas: arbitro, publico y tiradores de otras pistas)
           |  LO QUE LA LIMPIEZA BORRA, pintado en rojo encima del original
  fila 1   `seg` ANTES        |  `seg` DESPUES
  fila 2   `normal` ANTES     |  `normal` DESPUES
  fila 3   `matting` ANTES    |  `matting` DESPUES
  fila 4   `pointmap` ANTES   |  `pointmap` DESPUES

QUE ES EXACTAMENTE "ANTES", y esto importa porque hay dos cosas que se podrian llamar asi
=========================================================================================
ANTES es **la capa corrida sobre el recorte de la caja del tirador, sin mascara**: todos los pixeles
del recorte, tal cual salen del modelo. Se elige esa y no "la capa del cuadro completo" porque ES la
linea base que el proyecto probo primero y que fallo: recortar la caja del tirador y correr la capa
dentro NO borra al arbitro, y esta medido (`matting` deja fuga 0,251 y `seg` 0,383). O sea que la
columna de la izquierda no es un espantajo inventado para que la derecha luzca: es lo que habia.

La UNICA excepcion es `pointmap`, cuyo ANTES es el **cuadro completo**, porque esa capa se corre de
cuadro completo a proposito: dentro de un recorte su z no ordena profundidad (53 %, moneda al aire).
Asi que su ANTES ensena el pabellon entero y su DESPUES los dos tiradores.

LO QUE LA FIGURA NO DEMUESTRA
=============================
Que el arbitro no se vea no significa que no quede arbitro: la celda de BORRADO ensena lo que se ha
quitado, no lo que ha sobrevivido dentro de la caja propia del tirador, donde la metrica de banda es
ciega. Esa parte va en cifras (`resto_arb`) en el pie de la figura, no en la imagen.

ELECCION DE CUADROS, declarada antes de mirar: de los cuadros utilizables con los dos tiradores
presentes se cogen los TRES con mas area de (banda de margen INTERSECCION caja de un ajeno), o sea
los tres cuadros en que el arbitro o el publico se meten mas dentro de los recortes. Es una eleccion
del CASO DURO a proposito y no una muestra representativa del clip.

USO:  python capas_limpias_figuras.py [N_CUADROS_POR_CLIP]  con el interprete de transformers+CUDA
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

HERRAM = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\herramienta")
sys.path.insert(0, str(HERRAM))
import video_revision as VR

import cv2

SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\capas_limpias")
SALIDA.mkdir(parents=True, exist_ok=True)

CLIPS = ["WhisperedTenderDeinonychus", "GrossRelievedDwarfrabbit", "BabyishIdioticCockerspaniel",
         "DemandingMindlessDeviltasmanian", "SeveralSkinnyGenet"]
"""Los cinco con mas resto de arbitro del lote de auditoria, para que la figura ensene el caso duro y
para que sus cifras del pie se puedan cotejar con la tabla de la auditoria. Ninguno es de los tres
clips con los que se ajusto la limpieza."""

COL = {0: (60, 240, 60), 1: (60, 160, 255)}
COL_AJENO = (70, 70, 255)
ANCHO = 880


def esqueleto(im, kp, sc, col, g=2, r=4, rm=2):
    """Es `video_revision._pinta` con el grosor por parametro. El criterio de validez de un punto no
    se toca: confianza >= 0,3 Y coordenada finita, porque hay puntos con confianza alta y coordenada
    NaN y filtrar solo por confianza dibuja basura. La cara (del 63 en adelante) no se pinta."""
    def val(i):
        return sc[i] >= VR.CONF_PINTA and np.isfinite(kp[i, 0]) and np.isfinite(kp[i, 1])

    def pt(i):
        return (int(kp[i, 0]), int(kp[i, 1]))

    for a, b in VR.HUESOS:
        if val(a) and val(b):
            cv2.line(im, pt(a), pt(b), col, g, cv2.LINE_AA)
    for a, b in VR.PULGARES:
        if val(a) and val(b):
            cv2.line(im, pt(a), pt(b), (60, 240, 255), g, cv2.LINE_AA)
    for i in VR.MANOS:
        if val(i):
            cv2.circle(im, pt(i), rm, col, -1)
    for i in range(0, 21):
        if val(i):
            cv2.circle(im, pt(i), r, col, -1)


def celda(img, tit, pie, ancho, alto, marco=(60, 60, 66)):
    ih, iw = img.shape[:2]
    e = min(ancho / float(iw), alto / float(ih))
    nw, nh = max(1, int(iw * e)), max(1, int(ih * e))
    v = np.zeros((alto, ancho, 3), "uint8")
    oy, ox = (alto - nh) // 2, (ancho - nw) // 2
    v[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    cv2.rectangle(v, (0, 0), (ancho - 1, 27), (16, 16, 22), -1)
    cv2.putText(v, tit, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (240, 240, 240), 1, cv2.LINE_AA)
    if pie:
        ls = [pie] if isinstance(pie, str) else list(pie)
        h = 8 + 20 * len(ls)
        cv2.rectangle(v, (0, alto - h), (ancho - 1, alto - 1), (16, 16, 22), -1)
        for j, s in enumerate(ls):
            cv2.putText(v, s, (8, alto - h + 17 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        (125, 235, 255), 1, cv2.LINE_AA)
    cv2.rectangle(v, (0, 0), (ancho - 1, alto - 1), marco, 1)
    return v


ALTO_CERCANO = 0.20
"""Un ajeno tiene que medir al menos el 20 % del alto de imagen para contar en la eleccion de cuadro.

QUE FALLABA, y se vio en la PRIMERA figura que salio: sumando el area de TODOS los ajenos, el cuadro
ganador de `WhisperedTenderDeinonychus` fue el c17, con 12 ajenos que eran **publico al fondo**. Sus
cajas pisan el recorte porque el recorte es alto, pero no estan pegadas a los tiradores, asi que en
la columna de ANTES no habia nada que borrar y la figura no ensenaba lo que el usuario ha pedido ver.
El arbitro y los tiradores de la pista de al lado son GRANDES porque estan cerca; el publico del
fondo es pequeno. Asi que la talla del ajeno es lo que separa "caso duro" de "caso con muchas
cajas"."""


def cuadros_duros(sel, W, H, n):
    """Los cuadros con MAS area de banda de margen ocupada por la caja de un ajeno CERCANO. Solo
    geometria de cajas, sin GPU, asi que se puede evaluar el clip entero antes de gastar modelo."""
    B, pres, msk = sel["B"], sel["pres"], sel["mask"]
    puntos = []
    for k in range(len(B)):
        if not (msk[k] and pres[k].all()):
            continue
        aj = [o for o in CL.ajenos_del_cuadro(sel, k, H)
              if (o[3] - o[1]) / float(H) >= ALTO_CERCANO]
        if not aj:
            continue
        area = 0.0
        for lado in (0, 1):
            c = CL.recorte(B, k, lado, W, H)
            if c is None:
                continue
            bp = B[k, lado] * np.array([W, H, W, H])
            for o in aj:
                x0, y0 = max(c[0], o[0]), max(c[1], o[1])
                x1, y1 = min(c[2], o[2]), min(c[3], o[3])
                if x1 <= x0 or y1 <= y0:
                    continue
                px0, py0 = max(x0, bp[0]), max(y0, bp[1])
                px1, py1 = min(x1, bp[2]), min(y1, bp[3])
                dentro = max(0.0, px1 - px0) * max(0.0, py1 - py0)
                area += (x1 - x0) * (y1 - y0) - dentro
        if area > 0:
            puntos.append((area, k))
    puntos.sort(key=lambda t: -t[0])
    elegidos = []
    for _a, k in puntos:
        if all(abs(k - j) >= 12 for j in elegidos):
            """Doce cuadros de separacion minima: tres cuadros consecutivos son la misma imagen y no
            ensenan tres casos."""
            elegidos.append(k)
        if len(elegidos) >= n:
            break
    return elegidos


def figura(cap, clip, ruta, n_cuadros):
    with open(AU.SCR / ("det_%s.pkl" % clip), "rb") as fh:
        cajas, cuadros, W, H, T = pickle.load(fh)
    sel = CL.elige_pareja_del_clip(cajas, cuadros, W, H, T, gris=AU.gris_del_clip(ruta))
    B, tipo = sel["B"], sel["tipo"]
    KS = cuadros_duros(sel, W, H, n_cuadros)
    if not KS:
        return []
    alto = int(round(ANCHO * H / float(W)))
    vid = cv2.VideoCapture(ruta)
    hechas = []
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

        pm = cap.pointmap(rgb)
        Z = pm[:, :, 2] if pm.ndim == 3 else np.zeros((H, W), "float32")

        M_lim = np.zeros((H, W), bool)
        M_bru = np.zeros((H, W), bool)
        M_cro = np.zeros((H, W), bool)
        seg_a = np.zeros((H, W), "uint8")
        seg_d = np.zeros((H, W), "uint8")
        mat_a = np.zeros((H, W), "float32")
        mat_d = np.zeros((H, W), "float32")
        nor_a = np.zeros((H, W, 3), "float32")
        nor_d = np.zeros((H, W, 3), "float32")
        med = []
        for lado in (0, 1):
            r = CL.mascara_tirador(cap, rgb, sel, k, lado, W, H, kt, con_matting=True)
            if r is None:
                continue
            c, mk = r["c"], r["mask"]
            bruto = (r["seg"] != 0)
            if r["alfa"] is not None:
                bruto = bruto | (r["alfa"] >= CL.UMB_ALFA)
            """`bruto` es la mascara SIN limpiar: lo que `seg` o `matting` encienden dentro del
            recorte, o sea todas las personas que caen dentro. La celda de BORRADO es
            `bruto & ~limpio`, que es literalmente lo que la limpieza quita."""
            sl = (slice(c[1], c[3]), slice(c[0], c[2]))
            M_lim[sl] |= mk
            M_bru[sl] |= bruto
            M_cro[sl] = True
            seg_a[sl] = np.maximum(seg_a[sl], r["seg"])
            seg_d[sl][mk] = r["seg"][mk]
            if r["alfa"] is not None:
                mat_a[sl] = np.maximum(mat_a[sl], r["alfa"])
                mat_d[sl][mk] = r["alfa"][mk]
            n = cap.normal(r["sub"])
            if n.ndim == 3 and n.shape[:2] == mk.shape:
                nor_a[sl] = n
                nor_d[sl][mk] = n[mk]
            med.append(CL.mide(r, sel, k, lado, W, H, kt[("t", lado)], bgr=im))

        c_ori = im.copy()
        for o in aj:
            o = np.asarray(o).astype(int)
            cv2.rectangle(c_ori, (o[0], o[1]), (o[2], o[3]), COL_AJENO, 2)
            cv2.putText(c_ori, "NO elegido", (o[0], max(24, o[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_AJENO, 2, cv2.LINE_AA)
        for lado in (0, 1):
            kp, sc = kt[("t", lado)]
            esqueleto(c_ori, kp, sc, COL[lado], g=max(2, W // 640), r=max(3, W // 400))
            b = (B[k, lado] * np.array([W, H, W, H])).astype(int)
            cv2.rectangle(c_ori, (b[0], b[1]), (b[2], b[3]), COL[lado], 2)
            et = ["IZQUIERDA", "DERECHA"][lado]
            if tipo[k, lado] != 0:
                et += " " + {1: "interpolada", 2: "sostenida", 3: "redetectada"}[int(tipo[k, lado])]
            cv2.putText(c_ori, et, (b[0], max(24, b[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        COL[lado], 2, cv2.LINE_AA)

        borr = M_bru & ~M_lim
        c_bor = im.copy()
        c_bor[borr] = (0.25 * c_bor[borr] + 0.75 * np.array([40, 40, 255])).astype("uint8")
        for o in aj:
            o = np.asarray(o).astype(int)
            cv2.rectangle(c_bor, (o[0], o[1]), (o[2], o[3]), COL_AJENO, 2)

        v_seg_a = CL.pinta_seg(seg_a)
        v_seg_d = CL.pinta_seg(seg_d)
        v_nor_a = CL.pinta_normal(nor_a)
        v_nor_a[~M_cro] = 0
        v_nor_d = CL.pinta_normal(nor_d)
        v_nor_d[~M_lim] = 0
        v_mat_a = cv2.cvtColor((np.clip(mat_a, 0, 1) * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
        v_mat_d = cv2.cvtColor((np.clip(mat_d, 0, 1) * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
        v_pm_a = CL.pinta_z(Z, np.ones((H, W), bool))
        v_pm_d = CL.pinta_z(Z, M_lim)

        rec = float(np.nanmean([m["rec_kp"] for m in med])) if med else float("nan")
        fa = float(np.nanmean([m["fuga_arb"] for m in med])) if med else float("nan")
        ra = float(np.nanmean([m["resto_arb_frac"] for m in med])) if med else float("nan")
        rs = float(np.nanmean([m["residuo"] for m in med])) if med else float("nan")
        px_bor = int(borr.sum())
        px_lim = int(M_lim.sum())

        pares = [
            (celda(c_ori, "1  ORIGINAL  ·  esqueleto de 308 puntos  ·  verde izquierda, naranja "
                          "derecha, ROJO no elegidos",
                   ["%s  ·  cuadro %d  ·  %dx%d  ·  %d detecciones NO elegidas en el cuadro"
                    % (clip, k, W, H, len(aj))], ANCHO, alto),
             celda(c_bor, "2  LO QUE LA LIMPIEZA BORRA, en rojo",
                   ["%d px borrados de %d que encendian seg o matting  ·  la mascara final se queda "
                    "con %d px" % (px_bor, px_bor + px_lim, px_lim)], ANCHO, alto)),
            (celda(v_seg_a, "3  seg ANTES  ·  la capa sobre el recorte de caja, sin limpiar",
                   ["segmenta a TODO el que cae en el recorte: fuga de arbitro medida 0,3828"],
                   ANCHO, alto),
             celda(v_seg_d, "4  seg DESPUES  ·  mascara limpia",
                   ["recall de keypoints propios %.3f  ·  fuga de arbitro %.4f" % (rec, fa)],
                   ANCHO, alto)),
            (celda(v_nor_a, "5  normal ANTES  ·  la capa sobre el recorte, sin limpiar",
                   ["cada eje a un canal de color: el suelo y el arbitro tambien tienen normal"],
                   ANCHO, alto),
             celda(v_nor_d, "6  normal DESPUES  ·  mascara limpia, fuera va NaN no cero",
                   ["el vector cero no es una normal unitaria: el relleno tiene que ser NaN"],
                   ANCHO, alto)),
            (celda(v_mat_a, "7  matting ANTES  ·  alfa sobre el recorte, sin limpiar",
                   ["se queda con TODAS las personas del recorte: fuga medida 0,2505"], ANCHO, alto),
             celda(v_mat_d, "8  matting DESPUES  ·  alfa dentro de la mascara limpia",
                   ["residuo %.3f del recorte contra 0,657 de la caja sola" % rs], ANCHO, alto)),
            (celda(v_pm_a, "9  pointmap ANTES  ·  z del CUADRO COMPLETO, el pabellon entero",
                   ["esta capa se corre de cuadro completo: en un recorte su z no ordena "
                    "profundidad (53 %, moneda al aire)"], ANCHO, alto),
             celda(v_pm_d, "10  pointmap DESPUES  ·  z solo en los dos tiradores",
                   ["NO VALE COMO PROFUNDIDAD: falla el control fisico. Se ensena porque se ha "
                    "pedido verla"], ANCHO, alto)),
        ]
        fig = np.vstack([np.hstack(p) for p in pares])
        pie = np.zeros((78, fig.shape[1], 3), "uint8")
        cv2.putText(pie, "%s   cuadro %d   ·   recall %.3f   fuga de arbitro %.4f   resto de "
                         "arbitro %.4f   residuo %.3f" % (clip, k, rec, fa, ra, rs),
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(pie, "ANTES = la capa corrida sobre el recorte de la caja SIN mascara, que es la "
                         "linea base que fallo.  Que no se vea arbitro no prueba que no quede: eso "
                         "lo dice resto de arbitro.",
                    (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 200, 235), 1, cv2.LINE_AA)
        fig = np.vstack([fig, pie])
        f = SALIDA / ("capas_%s_c%04d.png" % (clip, k))
        cv2.imwrite(str(f), fig)
        hechas.append({"clip": clip, "k": k, "W": W, "H": H, "ajenos": len(aj),
                       "rec_kp": rec, "fuga_arb": fa, "resto_arb_frac": ra, "residuo": rs,
                       "px_borrados": px_bor, "px_mascara": px_lim, "fichero": str(f)})
        print("    c%04d  ajenos %d  rec %.3f  fuga %.4f  resto %.4f  borra %d px  -> %s"
              % (k, len(aj), rec, fa, ra, px_bor, f.name))
    vid.release()
    return hechas


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    clips = sys.argv[2].split(",") if len(sys.argv) > 2 else list(CLIPS)
    """El segundo argumento permite hacer los clips DE UNO EN UNO, y esto no es comodidad: la primera
    tirada de las quince figuras murio despues de la octava con codigo 1 y SIN traza de Python, o sea
    una muerte dura del proceso y no una excepcion. Con un proceso por clip, una muerte asi cuesta un
    clip y no la tirada entera. La causa de esa muerte queda sin diagnosticar y va dicho."""
    mv = AU.mapa_videos()
    cap = CL.Capas(["pose", "matting", "seg", "normal", "pointmap"])
    todo = []
    for clip in clips:
        if clip not in mv:
            print("  %s no esta en disco" % clip)
            continue
        t0 = time.perf_counter()
        print("  %s" % clip)
        todo += figura(cap, clip, mv[clip], n)
        print("    %.0f s" % (time.perf_counter() - t0))
    ind = SALIDA / "INDICE.json"
    previo = json.loads(ind.read_text(encoding="utf-8")) if ind.is_file() else []
    hechas = dict(((r["clip"], r["k"]), r) for r in previo)
    for r in todo:
        hechas[(r["clip"], r["k"])] = r
    """El indice se ACUMULA en vez de sobrescribirse, porque los clips se hacen de uno en uno en
    procesos distintos y una tirada de un clip no puede borrar el indice de los otros cuatro."""
    ind.write_text(json.dumps(sorted(hechas.values(), key=lambda r: (r["clip"], r["k"])), indent=1),
                   encoding="utf-8")
    print()
    print("  %d figuras nuevas, %d en el indice, en %s" % (len(todo), len(hechas), SALIDA))


if __name__ == "__main__":
    main()

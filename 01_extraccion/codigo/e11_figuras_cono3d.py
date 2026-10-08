"""ETAPA E, paso 11: LAS FIGURAS DEL CONO 3D, y aparte las de los cuadros donde 2D y 3D DISCREPAN.

QUE SE DIBUJA Y POR QUE DOS VISTAS. Una sola imagen no puede ensenar un cono 3D, porque la imagen ES
la proyeccion que borra la profundidad: sobre el fotograma, un cono que sale hacia la camara y otro
que va paralelo al plano se pintan casi igual. Asi que cada figura lleva dos celdas:

  IZQUIERDA  el fotograma con el cono 2D pintado, el cuadrilatero del torso del rival (que es la
             superficie valida en florete) y, si se vio, el segmento de HOJA detectado en el alfa.
  DERECHA    la VISTA DE PLANTA en centimetros, desde arriba, con los circulos de 30/60/90/120 cm.
             El de 90 cm es el alcance del arma (hoja de florete, FIE m.8.4), asi que lo que caiga
             fuera de ese circulo **no se puede tocar** aunque en la imagen parezca pegado.

LA GEOMETRIA NO SE RECONSTRUYE AQUI. Se llama a `cono3d_lib.construye_cono`, que es la misma funcion
que produjo los numeros del paso 10. Si la figura recalculara el cono con su propio codigo, una
diferencia entre el dibujo y el dato seria indetectable, y este proyecto ya perdio un dia entero por
una visualizacion que no coincidia con los datos (el desfase `f0`).

Y LOS ROTULOS LLEVAN LAS CIFRAS AL LADO, no solo el veredicto: el origen del cono (hoja VISTA o hoja
SUPUESTA), la apertura usada, las dos distancias al torso y los dos angulos. Quien mire la imagen
tiene que poder contradecir el numero.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cono3d_lib as C3
import cono_lib as CL
import e7_pinta_cono as E7

import cv2

PRUEBAS = CL.PRUEBAS
MUESTRAS = pathlib.Path(
    r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\cono3d")
MUESTRAS.mkdir(parents=True, exist_ok=True)
COL = {0: (255, 170, 40), 1: (60, 210, 255)}
VERDE = (110, 255, 170)
AMBAR = (60, 220, 255)
BLANCO = (240, 240, 240)
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
N_DISC = 6


def lee_cuadro(video, k):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
    ok, im = cap.read()
    cap.release()
    return im if ok else None


def pinta_sobre_imagen(im, g, lado, linea, tit, W, H):
    """El fotograma con el cono 2D, el torso del rival y la hoja vista. Recortado a lo que importa."""
    v = im.copy()
    q = np.int32([g["q2"][0], g["q2"][1], g["q2"][2], g["q2"][3]])
    cv2.polylines(v, [q], True, COL[1 - lado], 2, cv2.LINE_AA)
    cv2.putText(v, "torso rival = superficie valida", (int(q[:, 0].min()), int(q[:, 1].min()) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL[1 - lado], 1)
    """EL BRAZO QUE GENERA EL EJE, pintado para que se pueda comprobar a ojo que el cono arranca en
    la mano ARMADA. La eleccion de mano armada descansa en un solo criterio geometrico sin
    confirmacion independiente (medido y declarado en `cono_lib.mano_armada`), asi que tiene que ser
    contradecible mirando la imagen."""
    if g.get("p_hombro") is not None:
        cv2.line(v, np.int32(g["p_hombro"]), np.int32(g["p_codo"]), COL[lado], 2, cv2.LINE_AA)
    cv2.line(v, np.int32(g["p_codo"]), np.int32(g["vert2"]), COL[lado], 3, cv2.LINE_AA)
    cv2.putText(v, "brazo armado", (int(g["p_codo"][0]) - 30, int(g["p_codo"][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL[lado], 1)
    E7.dibuja_cono(v, g["vert2"], g["u2"], g["L"], g["apert"], VERDE, 2)
    if linea is not None:
        cv2.line(v, np.int32(linea["p1"]), np.int32(linea["p2"]), (40, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(v, "hoja vista en el alfa", (int(linea["p2"][0]) + 6, int(linea["p2"][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 255, 255), 1)
    cv2.circle(v, np.int32(g["vert2"]), 5, BLANCO, -1, cv2.LINE_AA)
    pts = np.array(list(g["q2"]) + [g["vert2"], g["vert2"] + g["L"] * g["u2"]])
    x0 = int(max(0, pts[:, 0].min() - 90))
    x1 = int(min(W, pts[:, 0].max() + 90))
    y0 = int(max(0, pts[:, 1].min() - 110))
    y1 = int(min(H, pts[:, 1].max() + 110))
    if x1 - x0 < 80 or y1 - y0 < 80:
        x0, y0, x1, y1 = 0, 0, W, H
    return E7.celda(v[y0:y1, x0:x1], tit, "", 660, 470)


def figura(clip, k, lado, g, linea, video, W, H, pxm, tit_izq, pie):
    im = lee_cuadro(video, k)
    if im is None:
        return None
    izq = pinta_sobre_imagen(im, g, lado, linea, tit_izq, W, H)
    der = C3.planta(cv2, g["V"], g["u3"], g["L"], g["apert"], g["malla3"], pxm,
                    "PLANTA (desde arriba)  ·  circulo 90 cm = alcance de la hoja", pie)
    fig = np.hstack([izq, der])
    bl = np.zeros((78, fig.shape[1], 3), np.uint8)
    bl[:] = (14, 14, 20)
    l1 = ("%s  c%d  lado %d  ·  direccion del eje: %s  ·  apertura %.1f gr (%s)" % (
        clip, k, lado, "HOJA VISTA en el alfa" if g["origen"] == 0 else "HOJA SUPUESTA del brazo",
        g["apert"], "temblor MEDIDO de la hoja" if g["apert_origen"] == 0
        else "regla e6, mas ancha"))
    l2 = ("2D: dist mano-torso %.0f cm  ang %.1f gr  ->  %s      |      3D: dist %.0f cm  "
          "ang %.1f gr  ->  %s" % (
              g["d2_cm"], g["ang2"], "AMENAZA" if g["am2"] else "no amenaza",
              g["d3_cm"], g["ang3"], "AMENAZA" if g["am3"] else "no amenaza"))
    l3 = ("elevacion del eje 3D %+.1f gr  ·  dz mano->torso %+.0f cm  ·  alcance de la hoja "
          "%.0f cm  ·  %s" % (
              g["elev"], g["dz_cm"], g["L_cm"],
              "DISCREPAN 2D y 3D" if g["am2"] != g["am3"] else "2D y 3D coinciden"))
    for j, t in enumerate((l1, l2, l3)):
        cv2.putText(bl, t, (8, 20 + 24 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (255, 140, 140) if (j == 2 and g["am2"] != g["am3"]) else (225, 225, 225), 1)
    return np.vstack([fig, bl])


def geo_de(clip, k, lado):
    """Reconstruye el cono de un (clip, cuadro, lado) con la MISMA funcion que lo midio."""
    lect, ks, mano, pxm, W, H, video = C3.carga_z(clip, SUF)
    f_med, _d, _t = C3.focal(lect, W, H)
    _r = json.loads((PRUEBAS / ("cono_e10_cono3d%s.json" % SUF)).read_text(
        encoding="utf-8"))["clips"][clip]
    cal, eje_clip = _r["cal"], _r.get("eje_usado", C3.EJE_POR_DEFECTO)
    """El eje se lee del JSON del paso 10 y NO se elige aqui: la figura tiene que dibujar el mismo
    cono que se midio, y si eligiera su propio eje la diferencia no se veria."""
    filas = json.loads((PRUEBAS / ("cono_hoja3d_%s%s.json" % (clip, SUF))).read_text(encoding="utf-8"))
    por = {(f["cuadro"], f["lado"]): f for f in filas}
    fila = por.get((k, lado))
    ln = fila["linea"] if (fila and not fila["contaminado"]) else None
    c = cal[str(lado)] if str(lado) in cal else cal[lado]
    g = C3.construye_cono(lect.get((k, lado), {}), lect.get((k, 1 - lado), {}), mano[lado],
                          W, H, f_med, pxm[lado], ln, c["apert_medida"], c["apert_supuesta"],
                          C3.N_MALLA, eje_clip)
    dib = ln if (ln and ln.get("delante")) else None
    return g, dib, video, W, H, pxm[lado], ks


res = json.loads((PRUEBAS / ("cono_e10_cono3d%s.json" % SUF)).read_text(encoding="utf-8"))
CLIPS = list(res["clips"].keys())
disc = res["discrepantes"]

print("=" * 100)
print("UNA FIGURA POR CLIP: el cono con mas elevacion medida, para que la vista de planta ensene algo")
print("=" * 100)
hechas = []
for clip in CLIPS:
    with np.load(PRUEBAS / ("cono3d_%s%s.npz" % (clip, SUF)), allow_pickle=True) as z:
        ks, orig, elev, am2, am3 = z["ks"], z["origen"], z["elev"], z["am2"], z["am3"]
    cands = [(abs(float(elev[k, l])), int(k), int(l)) for k in ks for l in (0, 1)
             if orig[k, l] == 0 and np.isfinite(elev[k, l])]
    if not cands:
        cands = [(abs(float(elev[k, l])), int(k), int(l)) for k in ks for l in (0, 1)
                 if orig[k, l] >= 0 and np.isfinite(elev[k, l])]
    if not cands:
        print("  %s: sin cono dibujable" % clip)
        continue
    cands.sort(reverse=True)
    _e, k, lado = cands[0]
    g, dib, video, W, H, pxm, _ks = geo_de(clip, k, lado)
    if g is None:
        print("  %s: el cono de c%d lado %d no se reconstruye" % (clip, k, lado))
        continue
    fig = figura(clip, k, lado, g, dib, video, W, H, pxm,
                 "FOTOGRAMA con el cono 2D  ·  %s" % clip,
                 "hoja %s" % ("VISTA" if g["origen"] == 0 else "SUPUESTA"))
    if fig is None:
        continue
    f = MUESTRAS / ("cono3d_%s_c%04d_l%d.png" % (clip, k, lado))
    cv2.imwrite(str(f), fig)
    hechas.append(f.name)
    print("  %-34s c%-5d lado %d  ·  elevacion %+.1f gr  ·  %s  ->  %s" % (
        clip, k, lado, g["elev"], "hoja VISTA" if g["origen"] == 0 else "hoja supuesta", f.name))

print()
print("=" * 100)
print("LAS DISCREPANCIAS 2D / 3D, aparte y rotuladas: %d en total" % len(disc))
print("=" * 100)
if not disc:
    print("  NINGUNA. Y esa es una respuesta valida: con lo que hay, el 3D no cambia la decision")
    print("  del cono en ningun cuadro de la muestra, y solo anade coste. Va dicho en el informe.")
else:
    """Se eligen las MAS EXTREMAS y repartidas entre clips, no las primeras: una discrepancia de
    1 cm por encima del alcance no la puede juzgar nadie a ojo, y una de 80 cm si."""
    def fuerza(d):
        return abs(d["d3_cm"] - d["d2_cm"]) if d["motivo"] == "alcance" else abs(d["elev"])
    porc = {}
    for d in sorted(disc, key=lambda d: -fuerza(d)):
        porc.setdefault(d["clip"], []).append(d)
    orden = []
    j = 0
    while len(orden) < min(N_DISC, len(disc)):
        puesto = False
        for cl in porc:
            if j < len(porc[cl]):
                orden.append(porc[cl][j])
                puesto = True
                if len(orden) >= min(N_DISC, len(disc)):
                    break
        if not puesto:
            break
        j += 1
    for d in orden:
        g, dib, video, W, H, pxm, _ks = geo_de(d["clip"], d["cuadro"], d["lado"])
        if g is None:
            print("  %s c%d lado %d: no se reconstruye" % (d["clip"], d["cuadro"], d["lado"]))
            continue
        et = "2D dice AMENAZA y 3D dice NO" if d["caso"] == "solo2D" else \
            "3D dice AMENAZA y 2D dice NO"
        """EL PIE DE LA FIGURA NO PUEDE DAR LA RAZON AL 3D, y esto es una correccion. Decia «el 3D
        manda», y **el control de la z permutada dice lo contrario**: barajando las profundidades al
        azar salen tantas discrepancias como con la z real. Asi que el pie describe el MECANISMO de
        la discrepancia y avisa de que no esta demostrado que sea real. Un rotulo que prejuzga es
        peor que ninguno, porque el usuario mira la imagen para contradecir el numero."""
        fig = figura(d["clip"], d["cuadro"], d["lado"], g, dib, video, W, H, pxm,
                     "DISCREPA: %s  (%s)" % (et, d["motivo"]),
                     "el 3D quita la amenaza por %s  ·  OJO: el control de la z permutada dice que "
                     "esto es ruido de `pointmap`" % (
                         "ALCANCE (no llega)" if d["motivo"] == "alcance"
                         else "DIRECCION (el eje sale del plano)"))
        if fig is None:
            continue
        f = MUESTRAS / ("DISCREPA_%s_%s_%s_c%04d_l%d.png" % (
            d["caso"], d["motivo"], d["clip"], d["cuadro"], d["lado"]))
        cv2.imwrite(str(f), fig)
        hechas.append(f.name)
        print("  %-9s %-10s %-30s c%-5d l%d  ·  d2 %5.0f cm  d3 %5.0f cm  ·  ang2 %5.1f  "
              "ang3 %5.1f  ·  elev %+6.1f  ->  %s" % (
                  d["caso"], d["motivo"], d["clip"], d["cuadro"], d["lado"], d["d2_cm"],
                  d["d3_cm"], d["ang2"], d["ang3"], d["elev"], f.name))

print()
print("%d figuras en %s" % (len(hechas), MUESTRAS))

"""ETAPA E, paso 21: EL CONO ARREGLADO, y el antes-y-despues en el MISMO fotograma.

El usuario dijo: *"le pones un florete pero no es el cono que dijimos, es recto, un palo. Revisalo.
Ademas es muy corto, no mide como deberia medir un florete. Y luego el cono tiene que ser mas
estrecho, que es demasiado gordo"*. Tres defectos, y `e20` los midio antes de tocar nada. Este paso
es el arreglo, y va con el antes al lado para que se juzgue de un golpe.

LOS TRES, Y COMO QUEDAN:

  1. «ES UN PALO». Dos cosas se veian como un palo y solo una era el cono. La otra era el **brazo**,
     que `e19` pintaba con la MISMA color del cono y grosor 3: un segmento recto y grueso que sale de
     la mano y parece un florete. Aqui el cono se pinta **RELLENO** (sector semitransparente) y el
     brazo, que sigue haciendo falta para comprobar a ojo de donde sale el eje, va en **linea de
     trazos, fina y en gris**, que no se puede confundir con un arma.

  2. «ES MUY CORTO». Y no venia de un largo fijo en pixeles: `e20` comprobo leyendo el codigo que los
     tres scripts de dibujo ya usaban la escala, `L = HOJA_M * pxm`. Venia de que **`HOJA_M` son los
     90 cm de la HOJA SOLA** (m.8.4), medidos desde la cara delantera de la guarda, y el cono sale de
     la RAIZ DE LA MANO, que esta por detras de la guarda. El alcance muneca-punta esta acotado por
     el reglamento en **[90, 110] cm** (m.8.4 y m.7) y se toma el extremo alto, `C3.ALCANCE_M = 1,10`,
     porque una region que tiene que CONTENER la amenaza no puede usar una cota por lo bajo del
     alcance. El arco de 90 cm se sigue pintando dentro, a trazos, para no perder la referencia.

  3. «MAS ESTRECHO». Aqui la apertura NO es una opcion de dibujo: es el error medido del eje. Y `e20`
     midio que con el eje que se estaba usando (`hombro -> raiz`) el cono de 23,1 grados **contenia
     solo el 22,6 % de las hojas detectadas**, o sea que perdia el 77 %. No estaba gordo: estaba
     demasiado estrecho para su propio eje. Cambiando al **antebrazo** (`codo -> raiz`) el error
     medido baja de 44,8 a 14,4 grados de mediana, y la apertura que contiene el 90 % baja de **66,3
     a 33,0 grados**. Eso es lo que permite dibujar mas estrecho sin perder nada:

         NUCLEO      p50 = 14,4 grados, relleno solido: donde esta la hoja la mitad de las veces
         ENVOLVENTE  p90 = 33,0 grados, relleno tenue: la region que DECIDE la amenaza

     El nucleo que se ve solido es **1,6x mas estrecho que los 23,1 grados de antes**, y la
     envolvente, que es la que cuenta, contiene el **90,3 %** de las hojas medidas contra el 22,6 %
     de antes. Las dos cosas a la vez, y las dos con su cifra.

Y LA HOJA SE PINTA SOLO SI SE HA DETECTADO, con la columna `origen` en el rotulo (0 hoja vista, 1
direccion supuesta del antebrazo). Nunca se prolonga una hoja desde la mano: esta medido y prohibido,
y ya hubo en este proyecto un cuaderno que trazaba 95 px desde la muneca y metia esa hoja inventada
en la etiqueta de tocado.

COSTE: se decodifican **6 fotogramas en total**, uno por clip, con `cap.set(CAP_PROP_POS_FRAMES, k)`.
Cero GPU y cero modelos: las detecciones de hoja salen de los JSON que `e5` y `e13` ya dejaron.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cono3d_lib as C3
import cono_lib as CL
import e20_estado_del_cono as E20

import cv2

PRUEBAS = CL.PRUEBAS
SAL = pathlib.Path(
    r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\cono_arreglado")
SAL.mkdir(parents=True, exist_ok=True)

APERT_VIEJA = 23.1
NUCLEO = C3.APERT_NUCLEO_SEMI_GR
ENVOLVENTE = C3.APERT_ENVOLVENTE_SEMI_GR
"""p50 y p90 de |hoja detectada - eje del antebrazo|, medidos en `e20` sobre 31 hojas. Se toman de
`cono3d_lib` y no se repiten aqui: una figura con su propia copia de la constante puede quedar
desincronizada de la medida sin que nadie lo note.

Y SON SEMIANGULOS, medidos desde el eje: el cono completo abre el DOBLE. El rotulo de cada figura lo
dice con las dos cifras (`+-33,0 gr` y `66,0 gr totales`) porque mirando el dibujo se ve el angulo
completo y en la constante esta el medio, y ese desajuste es exactamente la clase de convenio sin
declarar que ya le costo a este proyecto cinco cifras contradictorias en un dia."""
COL = {0: (255, 170, 40), 1: (60, 210, 255)}
SOLAPE = (90, 90, 255)
HOJA = (60, 255, 60)
EJE = (185, 185, 185)
CLIPS = ["AdorableFittingAsiansmallclawedotter", "AromaticShockedGermanspitz",
         "ImpeccablePertinentBasilisk", "WelloffForcefulIndri", "SeparateHoarseFeline",
         "AliveIgnorantBarebirdbat"]


def hojas_cacheadas():
    """Las rectas de hoja que `e5` y `e13` ya detectaron, por (clip, cuadro, lado).

    SE LEEN DE DISCO Y NO SE VUELVEN A DETECTAR. Detectar la hoja necesita `matting` en GPU y el PC
    del usuario esta saturado; y ademas repetirla podria dar otro resultado y entonces la figura no
    seria del dato que se publico.
    """
    out = {}
    for f in sorted(PRUEBAS.glob("cono_hoja3d_*_accion.json")):
        clip = f.name[len("cono_hoja3d_"):-len("_accion.json")]
        for r in json.loads(f.read_text(encoding="utf-8")):
            ln = r.get("linea")
            if ln and not r.get("contaminado") and ln.get("delante"):
                out[(clip, r["cuadro"], r["lado"])] = ln
    for f in sorted(PRUEBAS.glob("cono_hoja_*.json")):
        clip = f.name[len("cono_hoja_"):-len(".json")]
        for r in json.loads(f.read_text(encoding="utf-8")):
            ln = r["det"].get("p99.5", {}).get("linea") if not r.get("contaminado") else None
            if ln:
                out[(clip, r["cuadro"], r["lado"])] = ln
    return out


def sector(vert, u, L, apert, n=40):
    a0 = np.degrees(np.arctan2(u[1], u[0]))
    pts = [vert]
    for t in np.linspace(-apert, apert, n):
        th = np.radians(a0 + t)
        pts.append(vert + L * np.array([np.cos(th), np.sin(th)]))
    return np.int32(pts)


def rellena_peso(img, vert, u, L, apert, col, alfa_max=0.55):
    """EL CONO RELLENO, y el relleno ES LA FUNCION DE PESO QUE USA LA MEDIDA, no un tono plano.

    La opacidad de cada pixel vale `alfa_max * peso_sector(pixel)`, o sea `1 - angulo/apertura`: la
    misma forma con la que `cono_lib.peso_sector` puntua el cono y con la que `cruce` decide el
    candidato de contacto. Tiene dos consecuencias, y las dos importan:

      - la figura y el numero no pueden discrepar, porque son la MISMA funcion. Este proyecto ya
        perdio un dia por una visualizacion que no coincidia con los datos (el desfase `f0`).
      - el cono se ve estrecho SIN recortarlo: se apaga hacia el borde porque ahi el peso es casi
        cero, y a la vez la region entera sigue estando, con su contorno dibujado. Asi el usuario ve
        un cono estrecho y la medida no pierde ninguna amenaza. Es la unica forma de dar las dos
        cosas a la vez sin mentir en ninguna.

    Se pinta un tono plano por encima para el NUCLEO, que es la region del p50.
    """
    x0 = max(0, int(vert[0] - L - 2))
    x1 = min(img.shape[1], int(vert[0] + L + 2))
    y0 = max(0, int(vert[1] - L - 2))
    y1 = min(img.shape[0], int(vert[1] + L + 2))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    GY, GX = np.mgrid[y0:y1, x0:x1]
    P = np.stack([GX.ravel() + 0.5, GY.ravel() + 0.5], axis=1).astype("float64")
    w = CL.peso_sector(P, np.asarray(vert, "float64"), np.asarray(u, "float64"), L, apert)
    a = (alfa_max * w).reshape(y1 - y0, x1 - x0)[:, :, None]
    sub = img[y0:y1, x0:x1].astype("float32")
    img[y0:y1, x0:x1] = np.clip(sub * (1.0 - a) + np.asarray(col, "float32") * a,
                                0, 255).astype("uint8")


def arco(img, vert, u, L, apert, col, grosor=1, trazos=False):
    p = sector(vert, u, L, apert)
    if not trazos:
        cv2.polylines(img, [p], True, col, grosor, cv2.LINE_AA)
        return
    for j in range(1, len(p) - 1, 2):
        cv2.line(img, tuple(p[j]), tuple(p[j + 1]), col, grosor, cv2.LINE_AA)


def trazos(img, a, b, col, grosor=1, paso=9):
    a, b = np.asarray(a, "float64"), np.asarray(b, "float64")
    n = float(np.linalg.norm(b - a))
    if n < 1:
        return
    for t in np.arange(0, n, 2 * paso):
        p = a + (b - a) * (t / n)
        q = a + (b - a) * (min(t + paso, n) / n)
        cv2.line(img, tuple(np.int32(p)), tuple(np.int32(q)), col, grosor, cv2.LINE_AA)


def raster_solape(img, g, k, apert, alcance, alfa=0.40, ejes=None, aperts=None):
    """La region de solape, con la MISMA funcion de peso que usa la medida (`peso_sector`).

    Si el dibujo usara otra formula, la figura y el numero podrian discrepar sin que nadie lo note.
    Devuelve la capa para poder pintarla ENCIMA de los rellenos, porque la region de solape es lo
    que hay que poder ver: es donde podria estar el contacto de hojas.

    `ejes` y `aperts` SON OPCIONALES Y POR DEFECTO NO EXISTEN: sin ellos esto hace exactamente lo de
    siempre, el eje `d["u"][k]` y una sola apertura para los dos lados, que es lo que pintaron las
    figuras `cono_arreglado_*.png` ya aprobadas. Se anaden porque el video publico centra el campo en
    la HOJA MEDIDA cuando la hay, y entonces **cada lado tiene su centro y su anchura**: un solape
    calculado con otro centro que el dibujado ensenaria una region de contacto fuera de los conos
    pintados, o sea la misma clase de discrepancia figura-numero que este modulo existe para evitar.
    `ejes[lado]` es un vector unitario y `aperts[lado]` su semiangulo en grados.
    """
    d0, d1 = g[0], g[1]
    u0 = d0["u"][k] if ejes is None else np.asarray(ejes[0], "float64")
    u1 = d1["u"][k] if ejes is None else np.asarray(ejes[1], "float64")
    ap0 = apert if aperts is None else float(aperts[0])
    ap1 = apert if aperts is None else float(aperts[1])
    L0, L1 = alcance * d0["pxm"], alcance * d1["pxm"]
    lo = np.array([min(d0["vert"][k][0] - L0, d1["vert"][k][0] - L1),
                   min(d0["vert"][k][1] - L0, d1["vert"][k][1] - L1)])
    hi = np.array([max(d0["vert"][k][0] + L0, d1["vert"][k][0] + L1),
                   max(d0["vert"][k][1] + L0, d1["vert"][k][1] + L1)])
    paso = max(2.0, float(max(hi - lo)) / 220.0)
    xs = np.arange(lo[0], hi[0] + paso, paso)
    ys = np.arange(lo[1], hi[1] + paso, paso)
    GX, GY = np.meshgrid(xs, ys)
    P = np.stack([GX.ravel(), GY.ravel()], axis=1)
    w0 = CL.peso_sector(P, d0["vert"][k], u0, L0, ap0)
    w1 = CL.peso_sector(P, d1["vert"][k], u1, L1, ap1)
    inter = (w0 > 0) & (w1 > 0)
    cap = np.zeros_like(img)
    ip = int(paso) + 1
    for x, y in P[inter]:
        xi, yi = int(x), int(y)
        if 0 <= xi < img.shape[1] and 0 <= yi < img.shape[0]:
            cv2.rectangle(cap, (xi, yi), (xi + ip, yi + ip), SOLAPE, -1)
    prod = w0 * w1
    j = int(np.argmax(prod))
    return (P[j] if prod[j] > 0 else None), cap, alfa


def panel_antes(im, g, k):
    """EXACTAMENTE lo que `e19` pinta hoy: eje `hombro -> raiz`, 23,1 grados, L = 0,90 m, y el cono
    en CONTORNO. El brazo va en la color del cono y grosor 3, que es el «palo» que el usuario vio."""
    v = im.copy()
    cruz, cap, al = raster_solape(v, g, k, APERT_VIEJA, CL.HOJA_M)
    cv2.addWeighted(v, 1.0, cap, al, 0, dst=v)
    for lado in (0, 1):
        d = g[lado]
        L = CL.HOJA_M * d["pxm"]
        arco(v, d["vert"][k], d["u"][k], L, APERT_VIEJA, COL[lado], 2)
        h = d["ancla"][k]
        if np.isfinite(h).all():
            cv2.line(v, np.int32(h), np.int32(d["vert"][k]), COL[lado], 3, cv2.LINE_AA)
        cv2.circle(v, np.int32(d["vert"][k]), 6, (255, 255, 255), -1, cv2.LINE_AA)
    if cruz is not None:
        cv2.drawMarker(v, (int(cruz[0]), int(cruz[1])), (40, 255, 255), cv2.MARKER_CROSS, 26, 3)
    return v


def panel_despues(im, g, k, hojas, clip):
    """El cono arreglado: relleno, 1,10 m, nucleo p50 y envolvente p90, eje del antebrazo."""
    v = im.copy()
    cruz, cap, al = raster_solape(v, g, k, ENVOLVENTE, C3.ALCANCE_M)
    org, dhoja = {}, {}
    for lado in (0, 1):
        d = g[lado]
        rellena_peso(v, d["vert"][k], d["u"][k], C3.ALCANCE_M * d["pxm"], ENVOLVENTE, COL[lado])
    cv2.addWeighted(v, 1.0, cap, al, 0, dst=v)
    for lado in (0, 1):
        d = g[lado]
        L = C3.ALCANCE_M * d["pxm"]
        vert, u = d["vert"][k], d["u"][k]
        arco(v, vert, u, L, ENVOLVENTE, COL[lado], 1, trazos=True)
        arco(v, vert, u, L, NUCLEO, COL[lado], 1)
        arco(v, vert, u, CL.HOJA_M * d["pxm"], NUCLEO, (255, 255, 255), 1, trazos=True)
        """EL BRAZO EN TRAZOS FINOS Y GRISES, no en la color del cono y grosor 3. Hace falta para
        poder comprobar a ojo que el eje sale del antebrazo de la mano armada, y NO tiene que
        poderse confundir con un arma: por eso no es continuo ni tiene la color del cono."""
        if np.isfinite(d["ancla"][k]).all():
            trazos(v, d["ancla"][k], vert, EJE, 1)
        cv2.circle(v, np.int32(vert), 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(v, np.int32(vert), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(v, "hand %s s%d" % (d["mano"], lado),
                    (int(vert[0]) + 8, int(vert[1]) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        ln = hojas.get((clip, k, lado))
        org[lado] = 1 if ln is None else 0
        if ln is None:
            continue
        p1, p2 = np.int32(ln["p1"]), np.int32(ln["p2"])
        dhoja[lado] = float(np.linalg.norm(0.5 * (np.asarray(ln["p1"], "float64")
                                                  + np.asarray(ln["p2"], "float64")) - vert))
        cv2.line(v, tuple(p1), tuple(p2), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(v, tuple(p1), tuple(p2), HOJA, 3, cv2.LINE_AA)
        tx = (int(p1[0]), int(p1[1]) + 26) if p1[1] < p2[1] else (int(p2[0]), int(p2[1]) + 26)
        cv2.putText(v, "BLADE SEEN s%d" % lado, tx, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(v, "BLADE SEEN s%d" % lado, tx, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    HOJA, 1, cv2.LINE_AA)
    if cruz is not None:
        cv2.drawMarker(v, (int(cruz[0]), int(cruz[1])), (40, 255, 255), cv2.MARKER_CROSS, 26, 3)
    return v, org, dhoja


def recorta(g, k, alcance, apert, W, H):
    """EL RECORTE TIENE QUE CONTENER EL CONO ENTERO, arco incluido.

    La primera version acotaba con el vertice y la punta del eje, y **cortaba el sector por abajo**:
    un cono recortado se ve mas pequeno de lo que es, que es justo el defecto que hay que juzgar.
    """
    pts = []
    for lado in (0, 1):
        d = g[lado]
        L = alcance * d["pxm"]
        pts.extend(sector(d["vert"][k], d["u"][k], L, apert).tolist())
        pts.append((d["vert"][k] - 0.35 * L * d["u"][k]).tolist())
    p = np.array(pts, dtype="float64")
    x0 = int(max(0, p[:, 0].min() - 40))
    x1 = int(min(W, p[:, 0].max() + 40))
    y0 = int(max(0, p[:, 1].min() - 40))
    y1 = int(min(H, p[:, 1].max() + 40))
    return (0, 0, W, H) if (x1 - x0 < 120 or y1 - y0 < 120) else (x0, y0, x1, y1)


def celda(img, tit, pie, ancho=880, alto=620):
    ih, iw = img.shape[:2]
    e = min(ancho / max(iw, 1), alto / max(ih, 1))
    nw, nh = max(1, int(iw * e)), max(1, int(ih * e))
    v = np.zeros((alto, ancho, 3), np.uint8)
    ox, oy = (ancho - nw) // 2, (alto - nh) // 2
    v[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh))
    cv2.rectangle(v, (0, 0), (ancho, 26), (14, 14, 20), -1)
    cv2.putText(v, tit, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    y = alto - 8 - 18 * (len(pie) - 1)
    cv2.rectangle(v, (0, y - 16), (ancho, alto), (14, 14, 20), -1)
    for j, t in enumerate(pie):
        cv2.putText(v, t, (7, y + 18 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (120, 235, 255), 1, cv2.LINE_AA)
    return v


def geometria(clip, cual):
    d = E20.carga(clip)
    kp, sc, B, W, H = d["kp"], d["sc"], d["B"], d["W"], d["H"]
    g = {}
    for lado in (0, 1):
        mano, _x = CL.mano_armada(kp, sc, B, W, H, lado)
        if mano is None:
            return None, d
        vert, u, ok, _lg = C3.eje_brazo_2d(kp, sc, lado, mano, cual)
        _pk, pxm = CL.escala_px_m(B, lado, H)
        M = CL.MANOS[mano]
        i_a = M["hombro"] if cual == "hombro" else M["codo"]
        g[lado] = {"vert": vert, "u": u, "ok": ok, "pxm": pxm, "mano": mano,
                   "ancla": kp[:, lado, i_a].astype("float64")}
    d["dos"] = np.array([bool(d["mask"][j] and d["pres"][j].sum() == 2) for j in range(d["T"])])
    return g, d


D_OBJETIVO_CM = 60.0


def elige_cuadro(clip, gh, gc, d, hojas):
    """Se prefiere un cuadro con HOJA DETECTADA, para que la figura pueda ensenar `origen = 0`.

    Y SI NO HAY, NO SE USA EL CUADRO DE `peso_max` MAS ALTO, que es lo que hacia `e19`. Ese criterio
    esta sesgado para esto: `peso_max` se acerca a 1 cuando los dos ejes casi coinciden, y eso pasa
    sobre todo cuando **las dos manos casi coinciden en la imagen**, o sea en cuadros con los dos
    tiradores solapados y ocluidos. Comprobado mirando la figura de `SeparateHoarseFeline`, que salio
    con los dos vertices a unos pocos pixeles y no se puede juzgar a ojo.

    El criterio de reserva es la DISTANCIA MEDIDA entre las dos manos, y se busca la mas cercana a
    60 cm, que es el centro del tramo donde la pregunta del cono esta viva (`22.7` midio que el
    torso rival esta a <= 90 cm solo en el 2,6-12,5 % de los cuadros, con lateral mediana 45-81 cm).
    No usa la etiqueta de tocado, asi que no mete circularidad.
    """
    cand = sorted([kk for (cl, kk, _l) in hojas if cl == clip])
    for k in cand:
        if (k < d["T"] and d["dos"][k] and gh[0]["ok"][k] and gh[1]["ok"][k]
                and gc[0]["ok"][k] and gc[1]["ok"][k]):
            return k, "blade detected in this frame"
    mejor, mejor_e, mejor_d = None, 1e9, float("nan")
    pxm = 0.5 * (gc[0]["pxm"] + gc[1]["pxm"])
    for k in range(d["T"]):
        if not (d["dos"][k] and gh[0]["ok"][k] and gh[1]["ok"][k]
                and gc[0]["ok"][k] and gc[1]["ok"][k]):
            continue
        dm = 100.0 * float(np.linalg.norm(gc[0]["vert"][k] - gc[1]["vert"][k])) / pxm
        if not np.isfinite(dm) or abs(dm - D_OBJETIVO_CM) >= mejor_e:
            continue
        mejor, mejor_e, mejor_d = k, abs(dm - D_OBJETIVO_CM), dm
    if mejor is not None:
        return mejor, "hands %.0f cm apart (closest to %.0f)" % (mejor_d, D_OBJETIVO_CM)
    return None, "no frame with both fencers valid"


def main():
    hojas = hojas_cacheadas()
    corpus = E20.corpus()
    print("=" * 108)
    print("EL CONO ARREGLADO  ·  antes y despues en el mismo fotograma  ·  6 fotogramas decodificados")
    print("=" * 108)
    print("  hojas detectadas en cache: %d  ·  clips con alguna: %s" % (
        len(hojas), ", ".join(sorted(set(c for c, _k, _l in hojas)))))
    print()
    hechas, meta = [], {}
    for clip in CLIPS:
        gh, d = geometria(clip, "hombro")
        gc, _d = geometria(clip, "codo")
        if gh is None or gc is None:
            print("  %-38s sin mano armada decidible" % clip[:38])
            continue
        k, por_que = elige_cuadro(clip, gh, gc, d, hojas)
        if k is None:
            print("  %-38s %s" % (clip[:38], por_que))
            continue
        cap = cv2.VideoCapture(d["video"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        okf, im = cap.read()
        cap.release()
        if not okf:
            print("  %-38s no se pudo leer el cuadro %d" % (clip[:38], k))
            continue
        a = panel_antes(im, gh, k)
        b, org, dhoja = panel_despues(im, gc, k, hojas, clip)
        rx = recorta(gc, k, C3.ALCANCE_M, ENVOLVENTE, d["W"], d["H"])
        a = a[rx[1]:rx[3], rx[0]:rx[2]]
        b = b[rx[1]:rx[3], rx[0]:rx[2]]
        pxm = 0.5 * (gc[0]["pxm"] + gc[1]["pxm"])
        Lv = CL.HOJA_M * pxm
        Ln = C3.ALCANCE_M * pxm
        ca = celda(a, "BEFORE | %s | frame %d" % (clip, k), [
            "axis `shoulder -> wrist` | L = 0.90 m = %.0f px (BLADE ONLY, measured from the bell "
            "guard) | half-angle +-23.1 deg (46.2 deg full cone), OUTLINE only" % Lv,
            "the thick segment in the cone colour is the ARM: that is the 'stick' the user saw"])
        cb = celda(b, "AFTER | %s | frame %d" % (clip, k), [
            "axis `elbow -> wrist` (FOREARM) | L = %.2f m = %.0f px at %.0f px/m = WRIST-TO-TIP "
            "REACH (FIE m.8.4 + m.7) | dashed white arc = 0.90 m blade only" % (
                C3.ALCANCE_M, Ln, pxm),
            "core half-angle +-%.1f deg (%.1f deg full cone, p50) | envelope +-%.1f deg (%.1f deg "
            "full cone, p90) <- THIS one decides | fill = weight 1 - ang/half-angle" % (
                NUCLEO, 2 * NUCLEO, ENVOLVENTE, 2 * ENVOLVENTE),
            "origen side 0 = %d | side 1 = %d   (0 = blade SEEN, 1 = direction ASSUMED; ASSUMED in "
            "97 %% of all cones)%s" % (
                org[0], org[1],
                "".join(" | blade s%d at %.0f px from the hand" % (l, dv)
                        for l, dv in sorted(dhoja.items())))])
        fig = np.hstack([ca, cb])
        bl = np.zeros((56, fig.shape[1], 3), np.uint8)
        bl[:] = (14, 14, 20)
        cv2.putText(bl, "%s | frame %d chosen by: %s | %s | %s" % (
            clip, k, por_que, "1080p" if d["H"] >= 1000 else "720p",
            "in corpus_etiquetado" if clip in corpus else "NOT in corpus_etiquetado"),
            (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(bl, "the +-23.1 deg cone on the shoulder axis contained 22.6 % of the measured "
                        "blades; the +-33.0 deg envelope on the forearm axis contains 90.3 %",
                    (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150, 220, 255), 1, cv2.LINE_AA)
        p = SAL / ("cono_arreglado_%s_c%04d.png" % (clip, k))
        cv2.imwrite(str(p), np.vstack([fig, bl]))
        hechas.append(p.name)
        meta[clip] = {"cuadro": k, "motivo": por_que, "pxm": pxm, "L_vieja_px": Lv,
                      "L_nueva_px": Ln, "origen": org, "res": d["H"],
                      "en_corpus": clip in corpus, "mano": {l: gc[l]["mano"] for l in (0, 1)}}
        print("  %-38s c%-5d %-34s pxm %5.1f  L %3.0f -> %3.0f px  origen %d/%d  -> %s" % (
            clip[:38], k, por_que[:34], pxm, Lv, Ln, org[0], org[1], p.name))
    print()
    print("  %d figuras en %s" % (len(hechas), SAL))
    (PRUEBAS / "cono_e21_figuras.json").write_text(json.dumps(
        {"nucleo": NUCLEO, "envolvente": ENVOLVENTE, "apert_vieja": APERT_VIEJA,
         "alcance_m": C3.ALCANCE_M, "hoja_m": CL.HOJA_M, "figuras": hechas, "clips": meta},
        indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("guardado %s" % (PRUEBAS / "cono_e21_figuras.json"))


if __name__ == "__main__":
    main()

"""VERIFICACION A OJO de la pista, del punto de apoyo y de los eventos. Obligatoria, no opcional.

El usuario lo pidio asi: *"haz pruebas, no me digas que funciona y luego mire y se te cuele una
persona de detras en alguna pose, o algun frame sin pose, o que no marcas las marchas bien"*.

TRES SALIDAS, y la tercera es la que casi nadie hace y es la que importa:

  `apoyo_<clip>.png`   cuatro fotogramas con el punto de apoyo de cada pie dibujado y la reticula
                       metrica de la pista reproyectada encima. Si la reticula no cae sobre las
                       rayas del suelo, la homografia esta mal y se ve al instante.
  `ev_<tipo>_<clip>_c<k>.png`   el recorte del instante en que se detecto cada evento.
  `fn_<tipo>_<clip>_c<k>.png`   los FALSOS NEGATIVOS: instantes que la senal dice que deberian
                       haberse detectado y no se detectaron. Un evento sin falsos negativos medidos
                       no esta medido: solo mirando los aciertos, un detector que dispara una vez y
                       acierta parece perfecto.

COMO SE PROPONEN LOS FALSOS NEGATIVOS SIN ANOTACION HUMANA. Para el fondo, se cogen los maximos
locales de la separacion de pies que NO fueron marcados y que estan por encima del percentil 90 del
clip: si la separacion de pies llega al p90 y no se marco un fondo, o es un fondo perdido o es una
senal que no es un fondo, y las dos cosas hay que verlas. Para la marcha y el romper, los tramos en
que el centro del cuerpo se desplaza mas de 0,25 m sin que se haya marcado ningun paso. Para la
ALTO, los cuadros en que el tirador venia recorriendo mas de `ALTO_ANTES_M` y se queda por
debajo del umbral sin que se marcara nada. Para el BOTE, los maximos locales de la senal de
`levanta_pies` por encima del p98 del clip sin bote marcado, que es la unica forma de ver si el
umbral calibrado con el ruido se esta dejando botes de verdad fuera.
"""
from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import pista as PI
import posicion as PO

SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\pista\verifica")
COL = ((0, 220, 0), (0, 140, 255))
TIPOS_TIRA = ("marcha", "romper", "alto", "bote", "fondo")
"""Tipos que se sacan como TIRA de tres cuadros y no como recorte suelto: los de movimiento. Ver
`tira` para por que un fotograma no verifica una marcha."""


def _reticula(im, hg, M):
    """QUE HACE: dibuja la pista del reglamento sobre el fotograma, pasando por la similitud de
    estabilizacion de ese cuadro. Es LA comprobacion visual de la homografia."""
    Mi = np.linalg.inv(M)

    def px(pm):
        q = PI.a_imagen(hg, np.asarray(pm, "float64"))
        if not np.isfinite(q).all():
            return None
        v = Mi @ np.array([q[0], q[1], 1.0])
        if abs(v[2]) < 1e-9:
            return None
        return (int(round(v[0] / v[2])), int(round(v[1] / v[2])))

    w = hg["ancho_m"]
    for xm, col, gr in ((0.0, (0, 255, 255), 2), (-2.0, (255, 255, 0), 1), (2.0, (255, 255, 0), 1),
                        (-5.0, (255, 0, 255), 1), (5.0, (255, 0, 255), 1),
                        (-7.0, (255, 255, 255), 2), (7.0, (255, 255, 255), 2)):
        a, b = px([xm, 0.0]), px([xm, w])
        if a and b:
            cv2.line(im, a, b, col, gr)
    for ym, col in ((0.0, (0, 0, 255)), (w, (255, 0, 0))):
        pts = [px([x, ym]) for x in np.linspace(-7.0, 7.0, 15)]
        for i in range(len(pts) - 1):
            if pts[i] and pts[i + 1]:
                cv2.line(im, pts[i], pts[i + 1], col, 1)


def panel_apoyo(clip, res, ap, poses, fps):
    """QUE HACE: cuatro fotogramas con los pies detectados y la reticula metrica."""
    r, hg, Mk = res["_r"], res["_hg"], res["_Mk"]
    frames = res["_frames"]
    ks = np.nonzero(r["mask"])[0]
    sel = [int(ks[int(f * (len(ks) - 1))]) for f in (0.10, 0.40, 0.65, 0.90)]
    tiras = []
    for k in sel:
        if k >= len(frames):
            continue
        im = frames[k].copy()
        _reticula(im, hg, Mk[k])
        KP = poses["KP"] if poses else None
        SC = poses["SC"] if poses else None
        for lado in (0, 1):
            if not r["pres"][k, lado]:
                continue
            b = r["B"][k, lado]
            if np.isfinite(b).all():
                cv2.rectangle(im, (int(b[0] * r["W"]), int(b[1] * r["H"])),
                              (int(b[2] * r["W"]), int(b[3] * r["H"])), COL[lado], 1)
            if KP is not None and k < KP.shape[0]:
                for p in PO._pie_pose(KP, SC, k, lado):
                    if p is not None:
                        cv2.drawMarker(im, (int(p[0]), int(p[1])), COL[lado],
                                       cv2.MARKER_TILTED_CROSS, 16, 2)
            q = ap["C"][k, lado]
            if np.isfinite(q).all():
                cv2.putText(im, "%.2f, %.2f m" % (q[0], q[1]),
                            (int(0.5 * (b[0] + b[2]) * r["W"]) - 40,
                             int(b[3] * r["H"]) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            COL[lado], 1)
        cv2.putText(im, "f%d  t=%.2f s  feet %.2f / %.2f m  dist %.2f m"
                    % (k, (k - ks[0]) / fps, ap["dpies"][k, 0], ap["dpies"][k, 1],
                       ap["dist"][k]), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255), 1)
        tiras.append(im)
    if not tiras:
        return
    h = min(t.shape[0] for t in tiras)
    tiras = [t[:h] for t in tiras]
    a = np.vstack([np.hstack(tiras[:2]), np.hstack(tiras[2:4])]) if len(tiras) == 4 \
        else np.vstack(tiras)
    esc = 1500.0 / a.shape[1]
    a = cv2.resize(a, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
    SALIDA.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SALIDA / ("apoyo_%s.png" % clip)), a)


def recorte(clip, res, ap, k, lado, etiqueta, txt):
    """QUE HACE: un recorte alrededor de un tirador en un cuadro, para juzgar un evento a ojo."""
    r, frames = res["_r"], res["_frames"]
    if k >= len(frames):
        return
    W, H = r["W"], r["H"]
    b = r["B"][k, lado] if lado >= 0 and r["pres"][k, lado] else None
    if b is None or not np.isfinite(b).all():
        x0, y0, x1, y1 = 0, 0, W, H
    else:
        cx = 0.5 * (b[0] + b[2]) * W
        cy = 0.5 * (b[1] + b[3]) * H
        alt = (b[3] - b[1]) * H
        x0 = int(max(0, cx - 1.5 * alt))
        x1 = int(min(W, cx + 1.5 * alt))
        y0 = int(max(0, cy - 0.9 * alt))
        y1 = int(min(H, cy + 0.9 * alt))
    im = frames[k].copy()
    _reticula(im, res["_hg"], res["_Mk"][k])
    if b is not None and np.isfinite(b).all():
        cv2.rectangle(im, (int(b[0] * W), int(b[1] * H)), (int(b[2] * W), int(b[3] * H)),
                      COL[lado], 1)
    c = im[y0:y1, x0:x1]
    if c.size == 0:
        return
    esc = 620.0 / max(c.shape[1], 1)
    c = cv2.resize(c, None, fx=esc, fy=esc, interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(c, (0, 0), (c.shape[1] - 1, 22), (0, 0, 0), -1)
    cv2.putText(c, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
    SALIDA.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SALIDA / ("%s_%s_c%04d_l%d.png" % (etiqueta, clip, k, max(lado, 0)))), c)


def tira(clip, res, ap, k0, kp, k1, lado, etiqueta, txt, poses=None):
    """QUE HACE: tres recortes del MISMO evento, en el cuadro de inicio, el de pico y el de fin,
    pegados en una tira.

    HACE FALTA Y NO ES UN LUJO: un evento de movimiento no se puede juzgar en un fotograma. Mirando
    un solo cuadro de una marcha no se sabe si el tirador avanzo, retrocedio o estaba quieto, asi que
    "verificado a ojo" sobre un cuadro suelto no verifica nada de los tipos `marcha`, `romper`,
    `alto` y `bote`. Con los tres cuadros y los pies marcados se ve el gesto.
    """
    r, frames, hg = res["_r"], res["_frames"], res["_hg"]
    W, H = r["W"], r["H"]
    trozos = []
    for k, nom in ((k0, "start"), (kp, "peak"), (k1, "end")):
        if k >= len(frames):
            continue
        b = r["B"][k, lado] if lado >= 0 and r["pres"][k, lado] else None
        im = frames[k].copy()
        _reticula(im, hg, res["_Mk"][k])
        if b is not None and np.isfinite(b).all():
            cv2.rectangle(im, (int(b[0] * W), int(b[1] * H)), (int(b[2] * W), int(b[3] * H)),
                          COL[lado], 2)
            cx, cy = 0.5 * (b[0] + b[2]) * W, 0.5 * (b[1] + b[3]) * H
            alt = (b[3] - b[1]) * H
            x0 = int(max(0, cx - 1.3 * alt))
            x1 = int(min(W, cx + 1.3 * alt))
            y0 = int(max(0, cy - 0.85 * alt))
            y1 = int(min(H, cy + 0.85 * alt))
        else:
            x0, y0, x1, y1 = 0, 0, W, H
        if poses is not None and k < poses["KP"].shape[0] and lado >= 0:
            for p in PO._pie_pose(poses["KP"], poses["SC"], k, lado):
                if p is not None:
                    cv2.drawMarker(im, (int(p[0]), int(p[1])), COL[lado],
                                   cv2.MARKER_TILTED_CROSS, 18, 2)
        c = im[y0:y1, x0:x1]
        if c.size == 0:
            continue
        e = 420.0 / max(c.shape[1], 1)
        c = cv2.resize(c, None, fx=e, fy=e, interpolation=cv2.INTER_CUBIC)
        q = ap["C"][k, lado] if lado >= 0 else np.array([np.nan, np.nan])
        cv2.rectangle(c, (0, 0), (c.shape[1] - 1, 20), (0, 0, 0), -1)
        cv2.putText(c, "%s f%d  x %s m  feet %s m"
                    % (nom, k, ("%+.2f" % q[0]) if np.isfinite(q[0]) else "-",
                       ("%.2f" % ap["dpies"][k, lado]) if lado >= 0
                       and np.isfinite(ap["dpies"][k, lado]) else "-"),
                    (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        trozos.append(c)
    if not trozos:
        return
    h = min(t.shape[0] for t in trozos)
    a = np.hstack([t[:h] for t in trozos])
    cab = np.zeros((24, a.shape[1], 3), "uint8")
    cv2.putText(cab, txt, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    a = np.vstack([cab, a])
    SALIDA.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SALIDA / ("%s_%s_c%04d_l%d.png" % (etiqueta, clip, kp, max(lado, 0)))), a)


PIE_COL = ((255, 220, 0), (255, 0, 220))
"""Un color por PIERNA en el mosaico de conteo: cian para el pie 0 (tobillo 13) y magenta para el
pie 1 (tobillo 14). Hace falta para poder seguir cual es cual a lo largo de la tira."""
REJILLA_PX = 40
"""Separacion de las verticales de la rejilla de suelo, en pixeles del sistema ESTABILIZADO. Es la
referencia contra la que se juzga si una planta se ha trasladado o solo ha girado. 40 px son del
orden de 0,35-0,45 m en este corpus, o sea algo menor que una zancada, que es lo que hace falta para
distinguirlas."""


def mosaico_pasos(clip, res, ap, poses, lado, cada=2, por_fila=8, filas=5):
    """QUE HACE: los mosaicos con los que se cuentan A OJO todas las zancadas de un tirador.

    ES LA HERRAMIENTA DE LA VERDAD-TERRENO DEL RECALL, y el proyecto no la tenia: lo que habia medido
    era PRECISION (de 3 eventos marcados, 3 correctos) y una cota de falsos negativos de "6 a 50 pasos
    por clip", que es un rango tan ancho que no contesta nada. Nadie habia contado nunca cuantos pasos
    hay DE VERDAD en un clip.

    DOS DECISIONES QUE HACEN QUE EL CONTEO VALGA:

      1. LOS FOTOGRAMAS VAN ESTABILIZADOS por la similitud `Mk` de su cuadro, asi que **el suelo se
         queda quieto** y un pie que se traslada se ve contra un fondo fijo. Sin esto hay que adivinar
         cuanto de lo que se mueve es el paneo de camara, que en este corpus es p50 161 px y p90
         479 px, o sea del orden del propio paso.

      2. NO SE USA LA SERIE EN METROS PARA NADA. Usarla como verdad-terreno seria circular: es la
         misma senal con la que el detector dispara. La estabilizacion no es circular, es un ajuste
         geometrico 2D sobre puntos del FONDO, independiente de la pose, de la homografia y del
         detector de pasos.

    Se recorta la banda de las PIERNAS, que es donde esta la informacion, y no el cuerpo entero: a
    190 px de ancho por recorte, gastar la mitad en el torso y la careta es tirar resolucion donde no
    hay nada que contar.

    El convenio de que cuenta como zancada esta en `02_validacion/pista/STEP_COUNTING_CONVENTION.md`,
    escrito ANTES de contar.
    """
    r, frames, Mk = res["_r"], res["_frames"], res["_Mk"]
    W, H = r["W"], r["H"]
    ks = [int(k) for k in np.nonzero(r["mask"])[0]
          if k < len(frames) and r["pres"][k, lado] and np.isfinite(r["B"][k, lado]).all()]
    if len(ks) < 20:
        print("%-38s lado %d: menos de 20 cuadros utilizables" % (clip, lado))
        return []
    sel = ks[::cada]
    caj = []
    for k in sel:
        b = r["B"][k, lado]
        p0 = Mk[k] @ np.array([b[0] * W, b[1] * H, 1.0])
        p1 = Mk[k] @ np.array([b[2] * W, b[3] * H, 1.0])
        caj.append([p0[0] / p0[2], p0[1] / p0[2], p1[0] / p1[2], p1[1] / p1[2]])
    caj = np.asarray(caj)
    alt = float(np.median(caj[:, 3] - caj[:, 1]))
    """LA VENTANA SIGUE AL TIRADOR Y LA REFERENCIA DEL SUELO LA DA UNA REJILLA, y esta es la segunda
    version: la primera fijaba la ventana en el sistema estabilizado, que sobre el papel es lo
    correcto porque el suelo se queda quieto, y en la practica NO SE PODIA CONTAR. Un tirador que
    recorre 2 m obliga a una ventana de 490 x 180 px, que reducida a un mosaico deja las piernas en
    unos 40 px: a esa resolucion no se ve un pie despegar. Se vio generando la hoja y mirandola.

    Asi que la ventana se centra en el tirador, para gastar los pixeles en las piernas, y la
    referencia de suelo la da una REJILLA de verticales cada `REJILLA_PX` pixeles **del sistema
    estabilizado**, dibujada antes de recortar. En el mosaico la rejilla se mueve respecto al tirador
    exactamente lo que el tirador se mueve respecto al suelo, asi que una planta apoyada se queda en
    la misma linea y una que se traslada cambia de linea. Y las rayas pintadas de la pista, que
    tambien salen en el recorte, son la misma referencia de forma independiente."""
    ancho = 1.90 * alt
    """LA VENTANA SE FIJA POR FILA, y es la TERCERA version del mosaico. Las dos anteriores no
    servian para contar y las dos se descartaron mirando la hoja generada:

      ventana fija para todo el clip. Sobre el papel es lo correcto, porque el suelo se queda quieto
      en todos los cuadros. En la practica un tirador que recorre 2 m obliga a una ventana de
      490 x 180 px, que en el mosaico deja las piernas en unos 40 px: a esa resolucion no se ve un
      pie despegar.

      ventana centrada en el tirador, con rejilla de suelo. Resolucion buena, pero la casilla SIGUE
      al cuerpo, asi que para saber si un pie se ha trasladado hay que restar a ojo el movimiento de
      la rejilla del movimiento del pie, y eso es justo la cuenta que no se puede hacer de fiar.

    La que se usa: la ventana se calcula UNA VEZ POR FILA, con la caja mediana de esa fila. Dentro de
    una fila -- `por_fila` muestras, o sea 0,5-0,6 s -- el suelo NO se mueve, asi que un zapato que
    cambia de sitio se ve directamente, sin restar nada. Entre filas la ventana salta, y el salto se
    lee en la rejilla, que sigue dibujada."""
    ven = {}
    for i0 in range(0, len(sel), por_fila):
        g = slice(i0, min(i0 + por_fila, len(sel)))
        cx = float(np.median(0.5 * (caj[g, 0] + caj[g, 2])))
        by0 = float(np.median(caj[g, 1]))
        by1 = float(np.max(caj[g, 3]))
        for i in range(i0, min(i0 + por_fila, len(sel))):
            ven[i] = (cx, by0, by1)
    trozos = []
    for idx, k in enumerate(sel):
        im = cv2.warpAffine(frames[k], Mk[k][:2], (W, H), flags=cv2.INTER_LINEAR,
                            borderValue=(20, 20, 20))
        for gx in range(0, W + REJILLA_PX, REJILLA_PX):
            col = (0, 255, 255) if (gx // REJILLA_PX) % 2 == 0 else (120, 200, 200)
            cv2.line(im, (gx, 0), (gx, H), col, 1)
        pies = PO._pie_pose(poses["KP"], poses["SC"], k, lado) \
            if poses is not None and k < poses["KP"].shape[0] else [None, None]
        for j, p in enumerate(pies):
            if p is None:
                continue
            q = Mk[k] @ np.array([p[0], p[1], 1.0])
            cv2.drawMarker(im, (int(q[0] / q[2]), int(q[1] / q[2])), PIE_COL[j],
                           cv2.MARKER_TILTED_CROSS, 15, 2)
        cx, by0, by1 = ven[idx]
        x0, x1 = int(cx - 0.5 * ancho), int(cx + 0.5 * ancho)
        y0, y1 = int(by0 + 0.40 * alt), int(by1 + 0.16 * alt)
        pad = np.full((max(y1 - y0, 10), max(x1 - x0, 10), 3), 20, "uint8")
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(W, x1), min(H, y1)
        if sx1 > sx0 and sy1 > sy0:
            pad[sy0 - y0:sy0 - y0 + (sy1 - sy0), sx0 - x0:sx0 - x0 + (sx1 - sx0)] = \
                im[sy0:sy1, sx0:sx1]
        c = cv2.resize(pad, (int(ancho), int(0.86 * alt)), interpolation=cv2.INTER_AREA)
        esc = 330.0 / c.shape[1]
        c = cv2.resize(c, None, fx=esc, fy=esc, interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(c, (0, 0), (c.shape[1] - 1, 15), (0, 0, 0), -1)
        cv2.putText(c, "c%d" % k, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
        cv2.rectangle(c, (0, 0), (c.shape[1] - 1, c.shape[0] - 1), (90, 90, 90), 1)
        trozos.append(c)
    SALIDA.mkdir(parents=True, exist_ok=True)
    hs = min(t.shape[0] for t in trozos)
    ws = min(t.shape[1] for t in trozos)
    trozos = [t[:hs, :ws] for t in trozos]
    por_hoja = por_fila * filas
    salidas = []
    for h in range(0, len(trozos), por_hoja):
        hoja = trozos[h:h + por_hoja]
        fil = []
        for i in range(0, len(hoja), por_fila):
            fila = hoja[i:i + por_fila]
            while len(fila) < por_fila:
                fila.append(np.full_like(fila[0], 20))
            fil.append(np.hstack(fila))
        a = np.vstack(fil)
        cab = np.zeros((26, a.shape[1], 3), "uint8")
        cv2.putText(cab, "%s  side %d  frames %d-%d every %d  (sheet %d)"
                    % (clip, lado, sel[h], sel[min(h + por_hoja - 1, len(sel) - 1)], cada,
                       h // por_hoja + 1), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255), 1)
        p = SALIDA / ("conteo_%s_l%d_h%d.png" % (clip, lado, h // por_hoja + 1))
        cv2.imwrite(str(p), np.vstack([cab, a]))
        salidas.append(p)
    print("%-38s lado %d: %d cuadros muestreados de %d, %d hojas"
          % (clip, lado, len(trozos), len(ks), len(salidas)))
    return salidas


def falsos_negativos(ap, r, ev, fps, poses=None) -> list:
    """QUE HACE: propone instantes que DEBERIAN haberse detectado y no lo fueron.

    Ver la cabecera del modulo para el criterio. No son verdad-terreno: son candidatos que hay que
    MIRAR, y su utilidad es que si el detector se pierde la mitad de los fondos, aqui aparecen.
    """
    out = []
    ks = np.nonzero(r["mask"])[0]
    marcados = {(e["tipo"], e["lado"]): [] for e in ev}
    for e in ev:
        marcados.setdefault((e["tipo"], e["lado"]), []).append(e["cuadro"])
    for lado in (0, 1):
        dp = PO._suave(ap["dpies"][:, lado])
        v = dp[ks]
        v = v[np.isfinite(v)]
        if v.size < 20:
            continue
        umb = float(np.percentile(v, 90))
        ya = marcados.get(("fondo", lado), [])
        for i in range(ks[0] + 2, ks[-1] - 2):
            if not np.isfinite(dp[i]) or dp[i] < umb:
                continue
            if not (dp[i] >= dp[i - 1] and dp[i] >= dp[i + 1]):
                continue
            if any(abs(i - m) <= PO.FONDO_VENTANA for m in ya):
                continue
            if out and out[-1]["tipo"] == "fondo" and out[-1]["lado"] == lado and                     i - out[-1]["cuadro"] <= PO.FONDO_VENTANA:
                """SE FUSIONAN LOS CANDIDATOS SEGUIDOS, y hace falta para no inflar la cifra: un
                fondo que se MANTIENE deja la separacion de pies alta durante veinte cuadros, y cada
                maximo local de ese tramo se contaba como un falso negativo distinto. En
                `GentleQuerulousElephantseal` los cuadros 230, 231, 232 y 239 son el MISMO fondo."""
                continue
            out.append({"tipo": "fondo", "lado": lado, "cuadro": int(i),
                        "por_que": "separacion de pies %.2f m, p90 del clip %.2f m, sin fondo "
                                   "marcado" % (dp[i], umb)})
        pas = sorted(marcados.get(("marcha", lado), []) + marcados.get(("romper", lado), []))
        x = PO._suave(ap["C"][:, lado, 0])
        i = ks[0]
        while i + 10 <= ks[-1]:
            d = abs(x[i + 10] - x[i]) if np.isfinite(x[i]) and np.isfinite(x[i + 10]) else 0.0
            if d > 0.25 and not any(i <= m <= i + 10 for m in pas):
                out.append({"tipo": "paso", "lado": lado, "cuadro": int(i + 5),
                            "por_que": "el centro se desplaza %.2f m en 10 cuadros sin paso "
                                       "marcado" % d})
                i += 12
                continue
            i += 4
    out += _fn_altos(ap, r, marcados, ks)
    if poses is not None:
        out += _fn_botes(ap, r, marcados, ks, poses)
    return out


def _fn_altos(ap, r, marcados, ks):
    """CANDIDATOS A ALTO NO DETECTADA: el tirador venia recorriendo mas de `ALTO_ANTES_M` en la
    ventana y despues se queda por debajo del umbral de quieto SIN MARGEN, o sea entre el umbral
    calibrado y el doble. Son los casos frontera del umbral adaptativo, que es donde un umbral
    calibrado con el ruido se puede estar dejando fuera un alto de verdad.
    """
    V = PO.ALTO_VENTANA
    out = []
    for lado in (0, 1):
        umb, _rp, medible = PO.umbral_quieto(ap["C"], lado, r["mask"])
        if not medible:
            out.append({"tipo": "alto", "lado": lado, "cuadro": int(ks[len(ks) // 2]),
                        "por_que": "ALTO NO MEDIBLE en este lado: el umbral calibrado con el "
                                   "ruido sale %.3f m, por encima del tope de %.2f m"
                                   % (umb, PO.ALTO_TOPE_M)})
            continue
        xs = PO._suave(ap["C"][:, lado, 0], 5)
        ya = marcados.get(("alto", lado), [])
        k = ks[0] + V
        while k + V <= ks[-1]:
            a = xs[k] - xs[k - V]
            b = xs[k + V] - xs[k]
            if np.isfinite(a) and np.isfinite(b) and abs(a) >= PO.ALTO_ANTES_M \
                    and umb < abs(b) <= 2.0 * umb and not any(abs(k - m) <= V for m in ya):
                out.append({"tipo": "alto", "lado": lado, "cuadro": int(k),
                            "por_que": "venia %.2f m en %d cuadros y luego %.3f m, entre el umbral "
                                       "%.3f y su doble: frontera" % (abs(a), V, abs(b), umb)})
                k += V
                continue
            k += 1
    return out


def _fn_botes(ap, r, marcados, ks, poses):
    """CANDIDATOS A BOTE NO DETECTADO: los maximos locales de la senal de `levanta_pies` por encima
    del p98 del clip que no llegaron al umbral calibrado.

    ES EL CANDIDATO QUE MAS IMPORTA DE LOS CINCO, porque el bote es el tipo que la seccion 26 declara
    NO ESTABLECIDO: el umbral que hace falta para salir del ruido son 10,4 cm de despegue simultaneo
    y una balestra de florete despega 5-8 cm. Si al mirar estos candidatos se ven botes de verdad,
    queda confirmado que el detector no puede verlos; si no se ve ninguno, es que tampoco hay botes
    en la muestra. Las dos respuestas son informacion y hay que sacar los recortes para saber cual.
    """
    s = PO.levanta_pies(poses, r)
    out = []
    for lado in (0, 1):
        v = PO._suave(s[:, lado], 3)
        umb, _rp, _m = PO.umbral_bote(s, lado, r["mask"])
        fin = v[np.isfinite(v)]
        if fin.size < 40:
            continue
        p98 = float(np.percentile(fin, 98))
        ya = marcados.get(("bote", lado), [])
        for i in range(max(ks[0] + 1, 1), min(ks[-1], len(v) - 1)):
            if not np.isfinite(v[i]) or v[i] < p98 or v[i] >= umb:
                continue
            if not (v[i] >= v[i - 1] and v[i] >= v[i + 1]):
                continue
            if any(abs(i - m) <= 6 for m in ya) or \
                    (out and out[-1]["lado"] == lado and i - out[-1]["cuadro"] <= 6):
                continue
            out.append({"tipo": "bote", "lado": lado, "cuadro": int(i),
                        "por_que": "las dos plantas se levantan %.3f del alto de la caja, p98 del "
                                   "clip %.3f, umbral calibrado %.3f: por debajo del umbral"
                                   % (v[i], p98, umb)})
    return out


def una(clip, max_ev=4):
    """QUE HACE: la verificacion completa de un clip."""
    res = PI.analiza(clip)
    if not res.get("ok"):
        print("%-38s SIN HOMOGRAFIA: %s" % (clip, res.get("motivo")))
        return None
    fps = PO._fps(PI.indice_videos()[clip])
    poses = PO.carga_poses(clip)
    ap = PO.apoyos(res, poses)
    ev = PO.eventos(ap, res["_r"], fps, poses, res=res)
    panel_apoyo(clip, res, ap, poses, fps)
    n = {}
    for e in ev:
        t = e["tipo"]
        n[t] = n.get(t, 0) + 1
        if n[t] > max_ev or t == "accion":
            continue
        txt = "%s l%d c%d-%d" % (t, e.get("lado", -1), e["inicio"], e["fin"])
        if np.isfinite(e.get("magnitud_m", np.nan)):
            txt += "  mag %.2f %s" % (e["magnitud_m"],
                                      "xt" if t == "brazo" else "m")
        if e.get("pies_movidos"):
            txt += "  %d foot(s)" % e["pies_movidos"]
        if e.get("completa"):
            txt += "  FULL"
        if e.get("sobre_ruido"):
            txt += "  %.1f x the noise" % e["sobre_ruido"]
        if t in TIPOS_TIRA:
            tira(clip, res, ap, e["inicio"], e["cuadro"], e["fin"], e.get("lado", 0),
                 "ev_" + t, txt, poses)
        else:
            recorte(clip, res, ap, e["cuadro"], e.get("lado", 0), "ev_" + t, txt)
    fn = falsos_negativos(ap, res["_r"], ev, fps, poses)
    m = {}
    for e in fn:
        t = e["tipo"]
        m[t] = m.get(t, 0) + 1
        if m[t] > max_ev:
            continue
        if t in TIPOS_TIRA:
            tira(clip, res, ap, max(0, e["cuadro"] - 6), e["cuadro"], e["cuadro"] + 6,
                 e["lado"], "fn_" + t, "CANDIDATE NOT DETECTED: " + e["por_que"], poses)
        else:
            recorte(clip, res, ap, e["cuadro"], e["lado"], "fn_" + e["tipo"],
                    "CANDIDATE NOT DETECTED: " + e["por_que"])
    print("%-38s detectados %s | candidatos no detectados %s" % (clip, n, m))
    return {"clip": clip, "n": n, "fn": m, "ev": ev, "fn_lista": fn}


if __name__ == "__main__":
    import csv
    filas = []
    for c in sys.argv[1:]:
        try:
            d = una(c)
        except Exception as exc:
            print("%-38s EXCEPCION %s" % (c, exc))
            continue
        if d is None:
            continue
        f = {"clip": c}
        for t in ("fondo", "marcha", "romper", "alto", "bote", "brazo", "accion"):
            f["det_" + t] = d["n"].get(t, 0)
        for t in ("fondo", "paso", "alto", "bote"):
            f["cand_no_det_" + t] = d["fn"].get(t, 0)
        filas.append(f)
    if filas:
        SALIDA.mkdir(parents=True, exist_ok=True)
        with open(SALIDA.parent / "verificacion_eventos.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for f in filas for k in f}))
            w.writeheader()
            w.writerows(filas)
        print("\n%d clips verificados, imagenes en %s" % (len(filas), SALIDA))

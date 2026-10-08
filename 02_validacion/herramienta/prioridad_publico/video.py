"""VIDEO: una frase de armas con la PRIORIDAD anotada cuadro a cuadro, en ingles.

Que se ve, de arriba abajo, y todo APILADO con `vstack`, nunca pintado encima:
  banda 1  clip, cuadro, quien tiene el paso segun el detector, el articulo FIE y la confianza
  el fotograma  con las dos cajas, los dos conos, el esqueleto y el antebrazo armado
  banda 2  la verdad anotada a mano contra el detector, y el acumulado de acuerdo
  banda 3  la curva de `ext_hr` de los dos con el cursor y los cambios de prioridad marcados

LA TRAMPA DEL TAMANO, y esta escrita porque cuesta una hora encontrarla: el `VideoWriter` tiene que
abrirse con el alto TOTAL, o sea el del fotograma MAS el de las tres bandas. Si no coincide, OpenCV
**descarta cada cuadro por tamano y el MP4 sale de 0 bytes sin un solo error**. Aqui se comprueba en
cada cuadro con un `assert` y al final se comprueba el tamano del fichero.
"""
from __future__ import annotations

import csv

import cv2
import numpy as np

from comun import (ALCANCE_M, AMARILLO, BLANCO, COL, ENVOL_SEMI, F, FD, GRIS, LADO, NUCLEO_SEMI,
                   PRIO, ROJO, SALIDA, SinPisarse, VERDE, VIOLETA, banda, caja, carga, en, geo,
                   guarda, marco, pinta_cono, recuadro, texto, CL, PR, VP)

CLIP = "KindEnchantingKid"
INI, FIN = 205, 262
FPS = 10.0
"""10 cuadros por segundo sobre un origen de 30: el video dura el triple que la accion real. Es a
proposito, porque el encargo es que se pueda leer CUADRO A CUADRO, y a 30 fps ni el articulo ni la
cifra de confianza se ven. El numero de cuadro real va escrito en cada uno, asi que la camara lenta
no puede confundirse con la velocidad de la accion."""
HITOS = {226: ("RIGHT OF WAY IS GRANTED HERE", 14),
         232: ("THE LEFT FENCER COUNTER-ATTACKS - AND GAINS NOTHING", 14)}
"""Cuadros que se repiten unas decimas para que se puedan leer. El cuadro no cambia: se repite el
MISMO cuadro compuesto, asi que no se inventa ni un fotograma."""
ALTO_CAB, ALTO_VER, ALTO_GRA = 64, 58, 104


def publicado():
    out = {}
    with open(PRIO / "p2_por_cuadro.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["clip"] == CLIP:
                out.setdefault(int(r["cuadro"]), r)
    return out


def grafico(s, prio, k, W, alto=ALTO_GRA):
    """`ext_hr` de los dos tiradores, entera, con el cursor y cada cambio de prioridad marcado.

    Se pinta la serie COMPLETA y no una ventana movil, igual que en `video_prioridad._grafico`: asi
    se ve de un golpe si el cambio cae donde la curva se mueve o donde no se mueve nada.
    """
    g = np.full((alto, W, 3), (18, 18, 18), dtype="uint8")
    T = int(s["T"])
    x = np.linspace(0, W - 1, T).astype(int)
    lo, hi = 0.3, 2.0
    for lado in (0, 1):
        y, v = s["ext_hr"][:, lado], s["val"][:, lado]
        pts = []
        for j in range(T):
            if not (np.isfinite(y[j]) and v[j]):
                pts.append(None)
                continue
            pts.append((x[j], int(alto - 6 - (np.clip(y[j], lo, hi) - lo) / (hi - lo)
                                  * (alto - 30))))
        for j in range(1, T):
            if pts[j] and pts[j - 1]:
                cv2.line(g, pts[j - 1], pts[j], COL[lado], 1, cv2.LINE_AA)
    for j in range(1, T):
        if prio[j] != prio[j - 1]:
            cv2.line(g, (x[j], 16), (x[j], alto), COL[int(prio[j])], 1)
    yy = int(alto - 6 - (PR.EXT_ALTO - lo) / (hi - lo) * (alto - 30))
    cv2.line(g, (0, yy), (W, yy), (96, 96, 96), 1)
    cv2.line(g, (x[k], 14), (x[k], alto), BLANCO, 2)
    texto(g, "ext_hr = |shoulder -> wrist| / torso, whole clip. GREEN = left, ORANGE = right. "
             "Grey line = EXT_ALTO %.3f. White cursor = this frame. "
             "Vertical ticks = right-of-way changes." % PR.EXT_ALTO,
          (8, 12), 0.38, (196, 196, 202), 1, sombra=False)
    texto(g, "c%d" % k, (min(x[k] + 6, W - 52), 26), 0.42, BLANCO, 1, sombra=True)
    return g


def main():
    pub = publicado()
    d = carga(CLIP)
    s = PR.senales(d)
    W, H, T = int(d["W"]), int(d["H"]), int(d["T"])
    r = PR.decide(s, d["mask"], T)
    g = geo(d, s)
    prio, regla, conf = r["prio"], r["regla"], r["conf"]

    difieren = [k for k, f in pub.items()
                if k < T and int(f["detector"]) != int(prio[k])]
    if difieren:
        print("  AVISO: el detector recalculado difiere del publicado en %d cuadros: %s"
              % (len(difieren), difieren[:12]))
    else:
        print("  el detector recalculado coincide con p2_por_cuadro.csv en los %d cuadros "
              "publicados de %s" % (len(pub), CLIP))

    alto_total = ALTO_CAB + H + ALTO_VER + ALTO_GRA
    SALIDA.mkdir(parents=True, exist_ok=True)
    nombre = "video_right_of_way_%s_c%d_c%d.mp4" % (CLIP, INI, FIN)
    ruta = SALIDA / nombre
    out = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, alto_total))
    assert out.isOpened(), "no se pudo abrir el VideoWriter"

    cap = cv2.VideoCapture(str(d["video"]))
    ok_n = tot_n = escritos = 0
    for k in range(FIN + 1):
        ok, im = cap.read()
        if not ok:
            break
        if k < INI or k >= T:
            continue
        fila = pub.get(k)
        verdad = int(fila["verdad"]) if fila else None
        p = int(prio[k])

        vis = im.copy()
        for lado in (0, 1):
            if int(d["origen"][k, lado]) == 0:
                pinta_cono(vis, g, lado, k, COL[lado])
        cajas = {}
        for lado in (0, 1):
            inv = int(d["origen"][k, lado]) > 0
            col = VIOLETA if inv else COL[lado]
            VP._pinta(vis, d["KP"][k, lado], d["SC"][k, lado], col, g[lado]["armado"],
                      grueso=4 if lado == p else 2)
            b, inv = caja(d, k, lado)
            if b is not None:
                recuadro(vis, b, W, H, col, 4 if lado == p else 2)
            cajas[lado] = (b, inv)
        sp = SinPisarse(H)
        for lado in (0, 1):
            b, inv = cajas[lado]
            if b is None:
                continue
            x, y = int(b[0] * W), int(b[1] * H)
            et = "%s (side %d)" % (LADO[lado], lado)
            if inv:
                et += "  INVENTED POSE: cannot decide"
            elif lado == p:
                et += "  <== RIGHT OF WAY"
            x = max(6, min(x, W - 360))
            texto(vis, et, (x, sp.sitio(x, max(22, y - 10))), 0.56,
                  VIOLETA if inv else COL[lado], 2)
        if not d["mask"][k]:
            texto(vis, "FRAME NOT USABLE: close-up, replay or fade. It cannot decide.",
                  (12, H - 18), 0.6, ROJO, 2)

        cab = np.full((ALTO_CAB, W, 3), (16, 16, 18), dtype="uint8")
        cv2.rectangle(cab, (0, 0), (13, ALTO_CAB), COL[p], -1)
        texto(cab, "%s   frame c%d   (source 30 fps, shown at %d fps)" % (CLIP, k, int(FPS)),
              (24, 20), 0.5, (196, 196, 202), 1, FD, sombra=False)
        texto(cab, "RIGHT OF WAY: %s" % LADO[p], (24, 46), 0.72, COL[p], 2, FD, sombra=False)
        texto(cab, en(regla[k])[:96], (320, 46), 0.46, AMARILLO, 1, F, sombra=False)
        an = int(160 * float(conf[k]))
        cv2.rectangle(cab, (W - 180, 14), (W - 180 + an, 30), COL[p], -1)
        cv2.rectangle(cab, (W - 180, 14), (W - 20, 30), (110, 110, 110), 1)
        texto(cab, "detector confidence %.2f" % conf[k], (W - 180, 48), 0.44, (196, 196, 202), 1,
              sombra=False)

        ver = np.full((ALTO_VER, W, 3), (22, 22, 24), dtype="uint8")
        if verdad is None:
            texto(ver, "GROUND TRUTH: none - this frame is outside the two annotated windows "
                       "(c215-c236 and c239-c260)", (24, 22), 0.5, GRIS, 1, FD, sombra=False)
            texto(ver, "The detector still runs; there is simply nothing to score it against here.",
                  (24, 46), 0.44, (140, 140, 140), 1, sombra=False)
        else:
            tot_n += 1
            bien = verdad == p
            ok_n += int(bien)
            texto(ver, "HAND ANNOTATION: %s" % LADO[verdad], (24, 22), 0.52, COL[verdad], 1, FD,
                  sombra=False)
            texto(ver, "DETECTOR: %s" % LADO[p], (330, 22), 0.52, COL[p], 1, FD, sombra=False)
            texto(ver, "AGREE" if bien else "DISAGREE", (600, 22), 0.58,
                  VERDE if bien else ROJO, 2, FD, sombra=False)
            texto(ver, "running agreement over the annotated frames shown so far: %d / %d = "
                       "%.0f %%   |   published figure for the whole clip: 45.5 %% (n = 44)   |   "
                       "whole corpus: 69.7 %% against 73.8 %% for \"nobody\" (n = 462)"
                  % (ok_n, tot_n, 100.0 * ok_n / tot_n), (24, 46), 0.42, (196, 196, 202), 1,
                  sombra=False)

        cuadro = np.vstack([cab, vis, ver, grafico(s, prio, k, W)])
        assert cuadro.shape[:2] == (alto_total, W), \
            "tamano %s distinto del declarado al VideoWriter %s" % (cuadro.shape[:2],
                                                                    (alto_total, W))
        veces = 1
        if k in HITOS:
            et, veces = HITOS[k]
            c2 = cuadro.copy()
            cv2.rectangle(c2, (24, ALTO_CAB + 12), (24 + 14 * len(et), ALTO_CAB + 52),
                          (16, 16, 18), -1)
            cv2.rectangle(c2, (24, ALTO_CAB + 12), (24 + 14 * len(et), ALTO_CAB + 52),
                          AMARILLO, 2)
            texto(c2, et, (36, ALTO_CAB + 42), 0.72, AMARILLO, 2, FD, sombra=False)
            for _ in range(veces):
                out.write(c2)
                escritos += 1
        for _ in range(1 if k not in HITOS else 1):
            out.write(cuadro)
            escritos += 1
    cap.release()
    out.release()

    assert ruta.exists() and ruta.stat().st_size > 10000, \
        "el MP4 salio vacio o diminuto (%d bytes): revisa el tamano del VideoWriter" % (
            ruta.stat().st_size if ruta.exists() else -1)
    v = cv2.VideoCapture(str(ruta))
    leidos = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    ww, hh = int(v.get(cv2.CAP_PROP_FRAME_WIDTH)), int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
    v.release()
    print("  %-52s %4d x %-4d %7.1f KB  %d cuadros escritos, %d leidos, %.1f s a %d fps"
          % (nombre, ww, hh, ruta.stat().st_size / 1024.0, escritos, leidos,
             escritos / FPS, int(FPS)))
    assert (ww, hh) == (W, alto_total) and leidos > 0


if __name__ == "__main__":
    main()

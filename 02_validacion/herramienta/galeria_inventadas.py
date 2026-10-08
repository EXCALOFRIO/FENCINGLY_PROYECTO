"""Galeria de las poses INVENTADAS, una imagen por caso, para que el usuario las revise a ojo.

LO QUE PIDIO, literal: "cuando nos inventamos las poses que saques la foto para revisarla
visualmente y metes varias, unas 50, en una carpeta, y luego yo tambien las reviso, de las poses que
te las inventes".

QUE SE VE EN CADA IMAGEN, tres paneles del mismo recorte:
  IZQUIERDA   el cuadro ANTERIOR con su pose MEDIDA, que es el ancla de donde se parte
  CENTRO      el cuadro inventado, con la pose inventada en VIOLETA sobre los pixeles reales
  DERECHA     el cuadro SIGUIENTE con su pose MEDIDA, que es el otro ancla

Asi se puede juzgar lo unico que importa: si la pose del medio es una continuacion creible de las
dos de los lados, o si esta puesta en un sitio donde no hay nadie.

En el rotulo va el largo del hueco y el error esperado a ese largo, medido escondiendo poses que si
se conocen: 1 cuadro 2,8 % del torso, 3 cuadros 4,4 %, 5 cuadros 6,0 %, 8 cuadros 8,2-11,4 %,
12 cuadros 12-16 %, 24 cuadros 21-31 %.

Y va tambien el TIPO: `interpolada` cuando hay ancla a los dos lados, que es lo medido, o
`sostenida` cuando solo hay a un lado. Las sostenidas ya no se generan en la tuberia nueva, porque
los bordes del clip se recortan, pero si aparece alguna quiero verla marcada.
"""
from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import video_revision as VR

PRUEBAS = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\poses_inventadas")
CUANTAS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
ANCHO = 430
"""Ancho de cada panel. Tres paneles por imagen."""
FUERA = {"MellowHideousAngora", "PowerfulAllLarva"}
"""Clips que NO deben salir en la galeria, y por que cada uno:

`MellowHideousAngora` es de SABLE, no de florete. Se colo en una tirada por muestrear el arbol de
video en vez de la lista del corpus.

`PowerfulAllLarva` tiene el ARBITRO elegido como tirador izquierdo, verificado a ojo. Sus poses
inventadas interpolan al arbitro, asi que ensenarlas para juzgar la calidad de la interpolacion
mezclaria dos problemas distintos. El caso esta documentado aparte, con sus imagenes, en
`evidencia/cruce_de_lados`."""
ERROR_POR_HUECO = {1: 2.8, 2: 3.7, 3: 4.4, 5: 6.0, 8: 9.5, 12: 14.0, 16: 16.5, 24: 25.7}
"""Error mediano en fraccion del torso, medido escondiendo poses conocidas. Interpolado entre
valores para los largos que no estan en la tabla."""


HUECO_MEDIDO = 24
"""Hueco mas largo para el que EXISTE medida del error de interpolar. Por encima, el rotulo dice `sin
medida` en vez de dar una cifra: dar 25,7 % para un hueco de 66 cuadros seria mentir por omision, y
en `FirsthandSoreGnat` c116 se ve a ojo que la pose inventada acaba puesta donde no hay nadie."""


def error_esperado(largo):
    if largo > HUECO_MEDIDO:
        return float("nan")
    ks = sorted(ERROR_POR_HUECO)
    if largo <= ks[0]:
        return ERROR_POR_HUECO[ks[0]]
    if largo >= ks[-1]:
        return ERROR_POR_HUECO[ks[-1]]
    for a, b in zip(ks, ks[1:]):
        if a <= largo <= b:
            t = (largo - a) / float(b - a)
            return (1 - t) * ERROR_POR_HUECO[a] + t * ERROR_POR_HUECO[b]
    return float("nan")


def panel(frame, kp, sc, b, W, H, col, etq, rec):
    """QUE HACE: un panel recortado alrededor de la caja, con la pose pintada encima."""
    x0, y0, x1, y1 = rec
    sub = frame[y0:y1, x0:x1].copy()
    if kp is not None:
        k2 = kp.copy()
        k2[:, 0] -= x0
        k2[:, 1] -= y0
        VR._pinta(sub, k2, sc, col)
    if b is not None and np.isfinite(b).all():
        cv2.rectangle(sub, (int(b[0] * W) - x0, int(b[1] * H) - y0),
                      (int(b[2] * W) - x0, int(b[3] * H) - y0), col, 2)
    esc = ANCHO / float(max(sub.shape[1], 1))
    sub = cv2.resize(sub, (ANCHO, max(1, int(sub.shape[0] * esc))))
    cv2.rectangle(sub, (0, 0), (sub.shape[1], 22), (12, 12, 18), -1)
    cv2.putText(sub, etq, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1)
    return sub


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    for p in SALIDA.glob("*.png"):
        p.unlink()
    fich = sorted(PRUEBAS.glob("poses_*.npz"))
    if not fich:
        raise SystemExit("no hay NPZ de poses; corre video_revision.py fase1 y fase2 primero")
    fich = [f for f in fich if f.name[6:-4] not in FUERA]
    casos = []
    for f in fich:
        z = np.load(f, allow_pickle=True)
        if "origen" not in z.files:
            """Hay NPZ de poses escritos por otras herramientas que no llevan `origen`. Sin esa
            marca no se sabe cual pose es inventada, asi que ese fichero no sirve aqui."""
            print("   se salta %s: no lleva la marca `origen`" % f.name)
            continue
        origen, pres, mask = z["origen"], z["pres"], z["mask"]
        T = int(z["T"])
        for lado in (0, 1):
            hay = np.nonzero(pres[:, lado] & mask)[0]
            k = 0
            while k < T:
                if origen[k, lado] <= 0:
                    k += 1
                    continue
                ini = k
                while k < T and origen[k, lado] > 0:
                    k += 1
                fin = k - 1
                ant = hay[hay < ini]
                pos = hay[hay > fin]
                casos.append({"npz": f, "clip": f.name[6:-4], "lado": lado,
                              "ini": ini, "fin": fin, "largo": fin - ini + 1,
                              "ant": int(ant[-1]) if len(ant) else None,
                              "pos": int(pos[0]) if len(pos) else None,
                              "tipo": int(origen[ini, lado])})
    if not casos:
        print("no hay ninguna pose inventada en los NPZ disponibles")
        return
    print("casos de pose inventada encontrados: %d, en %d clips" % (
        len(casos), len(set(c["clip"] for c in casos))))
    """Se reparten los sitios entre clips para que la galeria no sea todo el mismo video, y dentro de
    cada clip se cogen los huecos MAS LARGOS primero, que son los que mas hay que mirar."""
    porclip = {}
    for c in casos:
        porclip.setdefault(c["clip"], []).append(c)
    for v in porclip.values():
        v.sort(key=lambda c: -c["largo"])
    elegidos, i = [], 0
    while len(elegidos) < CUANTAS and any(i < len(v) for v in porclip.values()):
        for v in porclip.values():
            if i < len(v) and len(elegidos) < CUANTAS:
                elegidos.append(v[i])
        i += 1

    porfich = {}
    for c in elegidos:
        porfich.setdefault(str(c["npz"]), []).append(c)
    n = 0
    for ruta, cs in porfich.items():
        z = np.load(ruta, allow_pickle=True)
        KP, SC, B, mask = z["KP"], z["SC"], z["B"], z["mask"]
        Be = z["Be"] if "Be" in z.files else B
        W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
        video = str(z["video"])
        quiere = set()
        for c in cs:
            for kk in (c["ant"], c["pos"]):
                if kk is not None:
                    quiere.add(kk)
            quiere.update(range(c["ini"], c["fin"] + 1))
        cap = cv2.VideoCapture(video)
        frames = {}
        for k in range(T):
            ok, im = cap.read()
            if not ok:
                break
            if k in quiere:
                frames[k] = im
        cap.release()
        for c in cs:
            lado = c["lado"]
            kmed = (c["ini"] + c["fin"]) // 2
            if kmed not in frames:
                continue
            cajas = [Be[k, lado] for k in (c["ant"], kmed, c["pos"]) if k is not None]
            cajas = [b for b in cajas if np.isfinite(b).all()]
            if not cajas:
                continue
            xs = [v for b in cajas for v in (b[0] * W, b[2] * W)]
            ys = [v for b in cajas for v in (b[1] * H, b[3] * H)]
            mx = 0.30 * (max(xs) - min(xs))
            my = 0.18 * (max(ys) - min(ys))
            rec = (int(max(0, min(xs) - mx)), int(max(0, min(ys) - my)),
                   int(min(W, max(xs) + mx)), int(min(H, max(ys) + my)))
            if rec[2] - rec[0] < 30 or rec[3] - rec[1] < 30:
                continue
            col = VR.COL_LADO[lado]
            paneles = []
            if c["ant"] is not None and c["ant"] in frames:
                paneles.append(panel(frames[c["ant"]], KP[c["ant"], lado], SC[c["ant"], lado],
                                     B[c["ant"], lado], W, H, col,
                                     "c%d MEDIDA (ancla antes)" % c["ant"], rec))
            paneles.append(panel(frames[kmed], KP[kmed, lado], SC[kmed, lado], Be[kmed, lado],
                                 W, H, VR.VIOLETA,
                                 "c%d INVENTADA %s" % (kmed, "interpolada" if c["tipo"] == 1
                                                       else "SOSTENIDA"), rec))
            if c["pos"] is not None and c["pos"] in frames:
                paneles.append(panel(frames[c["pos"]], KP[c["pos"], lado], SC[c["pos"], lado],
                                     B[c["pos"], lado], W, H, col,
                                     "c%d MEDIDA (ancla despues)" % c["pos"], rec))
            alto = max(p.shape[0] for p in paneles)
            tira = np.zeros((alto + 30, ANCHO * len(paneles) + 8 * (len(paneles) - 1), 3), "uint8")
            x = 0
            for p in paneles:
                tira[30:30 + p.shape[0], x:x + ANCHO] = p
                x += ANCHO + 8
            er = error_esperado(c["largo"])
            if np.isfinite(er):
                fin = "error esperado %.1f %% del torso" % er
                col_rot = (235, 225, 250)
                fondo = (28, 20, 40)
            else:
                """Por encima de 24 cuadros no hay medida, y se dice en el rotulo y con el color, para
                que estas no se puedan confundir con las validadas al pasar las imagenes."""
                fin = "hueco FUERA del rango medido (>%d): error NO verificable" % HUECO_MEDIDO
                col_rot = (255, 235, 235)
                fondo = (20, 20, 130)
            cv2.rectangle(tira, (0, 0), (tira.shape[1], 28), fondo, -1)
            cv2.putText(tira, "%s  %s  hueco de %d cuadro%s  %s" % (
                c["clip"][:30], ["IZQUIERDA", "DERECHA"][lado], c["largo"],
                "s" if c["largo"] != 1 else "", fin),
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_rot, 1)
            nom = "%03d_%s_%s_c%04d_hueco%02d.png" % (
                n + 1, c["clip"][:22], ["izq", "der"][lado], kmed, c["largo"])
            cv2.imwrite(str(SALIDA / nom), tira)
            n += 1
    print("escritas %d imagenes en %s" % (n, SALIDA))
    largos = [c["largo"] for c in elegidos[:n]]
    if largos:
        print("largo del hueco en la galeria: mediana %d, maximo %d" % (
            int(np.median(largos)), max(largos)))
        sost = sum(1 for c in elegidos[:n] if c["tipo"] == 2)
        print("de las %d, sostenidas (sin ancla a un lado): %d" % (n, sost))


if __name__ == "__main__":
    main()

"""HERRAMIENTA PARA CONSTRUIR LA VERDAD DE REFERENCIA DE LA MANO ARMADA, Y LUEGO LAS FIGURAS.

NO HAY VERDAD ETIQUETADA de la mano armada en este proyecto, asi que hay que construirla. La hace
UN AGENTE MIRANDO LOS RECORTES, no un arbitro de esgrima, y eso queda dicho en el propio fichero de
anotacion y en cada cifra que se derive de el.

MODO `anota` — LOS MOSAICOS SON CIEGOS A PROPOSITO. Cada mosaico se etiqueta solo con `A` y `B`, que
son las dos manos por INDICE del mapa de 308 puntos (`A` = raiz 21, `B` = raiz 42) y no el veredicto
de ningun criterio. Si el mosaico dijera «esta es la que elige el criterio (a)», la anotacion ya no
seria independiente de lo que se quiere medir, y la cifra de acierto no valdria nada.

QUE SE MIRA PARA DECIDIR A OJO, en este orden:
 1. la CAZOLETA, que son 19 px a 720p y 29 a 1080p, o sea un disco visible, y el `pasante` negro que
    sale de ella;
 2. la HOJA, cuando el clip es de 1080p y se ve la recta saliendo de la mano;
 3. la accion: el brazo que se extiende al fondo y lleva la punta al rival;
 4. y en ultimo lugar el GUANTE. Si un tirador solo se puede decidir por el guante, se anota
    `cue=guante` y **ese tirador no cuenta** en el acierto del criterio del guante, porque seria
    medirse contra si mismo. Es el riesgo de circularidad de esta medida y esta es la defensa.

MODO `figuras` — pinta la mano elegida con el rasgo del guante escrito al lado, y aparte los casos
donde la geometria y el guante DISCREPAN, que son los que valen.

MODO `ciego` — LA VERSION FUERTE DEL MISMO METODO, anadida el 2026-09-11 para ampliar la verdad de
27 a 44 tiradores. La primera tanda ya era ciega respecto al criterio, pero el NOMBRE DEL CLIP iba
escrito en la imagen y en el nombre del fichero, asi que el anotador podia acordarse de lo que habia
puesto antes o de lo que decia una tabla. En `ciego`:

  - los mosaicos se llaman `ciego_NN.png` y **no llevan ni el clip ni el lado** escritos;
  - el orden se baraja con `random.Random(SEMILLA_CIEGO)`, semilla fija y declarada, para que la
    tanda se pueda reproducir;
  - la correspondencia `NN -> (clip, lado)` se escribe en `clave_ciego.csv` **aparte**, y no se mira
    hasta que la anotacion esta escrita;
  - la segunda pasada se pide por `NN`, no por clip, asi que sigue siendo ciega.

Uso:
    python verifica_mano.py anota            todos los clips de la lista del corpus
    python verifica_mano.py anota  CLIP...   solo esos
    python verifica_mano.py ciego  CLIP...   mosaicos ciegos numerados + clave aparte
    python verifica_mano.py ciego  6 NN...   segunda pasada de esos NN con 6 cuadros
    python verifica_mano.py figuras
"""
from __future__ import annotations

import csv
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import cono_lib as CL
import mano_guante as MG

SAL = MG.MUESTRAS
VERDAD = SAL / "verdad_mano.csv"
NFRAMES = 3
ALTO_DET = 380
ANCHO_CTX = 760
COL = {"A": (255, 0, 255), "B": (255, 255, 0)}
"""Magenta para la mano A (raiz 21) y cian para la mano B (raiz 42). Fijo en todas las figuras."""
EN_CUE = {"hoja": "blade visible", "arma": "weapon/guard visible",
          "accion": "leading hand only", "careta": "mask in other hand (FIE t.115.1)",
          "piel": "bare skin on other hand", "arbitro": "NOT A FENCER: referee",
          "noTirador": "NOT A FENCER", "": "undecidable"}
EN_POR = {"pie_firme": "front foot, firm vote", "encarado_accion": "on-guard + in-action gate",
          "hombro_firme": "shoulder, firm vote", "mayoria_3_dudosa": "3-way majority (UNCERTAIN)",
          "geometria_a": "old rule (UNCERTAIN)",
          "geometria_a_GUANTE_RETIRADO": "old rule (glove withdrawn)"}
"""TODO EL TEXTO QUE SE VE VA EN INGLES, porque las figuras van a un repo publico y las tiene que
entender cualquiera. Los docstrings y los nombres de columna de los CSV se quedan como estan: son
internos, y traducir los nombres de columna romperia el codigo de otros agentes."""


def elige_cuadros(d, lado, n=NFRAMES):
    """`n` cuadros repartidos en el tiempo, y dentro de cada tercio el de caja mas ALTA.

    TRES FILTROS, y los tres salieron de mirar el primer mosaico, que fue inservible:

      1. la caja tiene que estar ENTERA de lado a lado (`x0 > 0,01` y `x1 < 0,99`). En el primer
         intento salio `AliveIgnorantBarebirdbat` con la caja en `x` de 0 a 81 px: el tirador estaba
         medio fuera y la raiz de la mano cayo en `x = 0`.
      2. la caja tiene que estar por encima de la mediana de ALTO del clip: caja mas alta = tirador
         mas cerca de la camara = mas pixeles en la mano. A 720p eso es la diferencia entre poder
         mirar y no poder.
      3. las DOS RAICES DE MANO tienen que estar separadas al menos `2r` (`MG.SOLAPE_MIN`). Este
         filtro tambien salio de mirar: los tres cuadros elegidos de `BlissfulTiredAiredale` lado 0
         daban `sep` de 1-2 px con `r` de 19-30 px, y ahi no hay nada que anotar porque los dos
         marcadores caen sobre el mismo punto. El clip entero tiene `sep` mediana de 144 px, o sea
         que el problema era el cuadro elegido, no el clip: el filtro 4 buscaba maximo enganche y el
         maximo enganche es justo donde las manos se juntan.
      4. de los que quedan, el de SEPARACION MINIMA entre los dos tiradores en cada tercio del
         clip. Entre accion y accion los tiradores andan, con el brazo bajo y el arma al suelo, y
         ahi no se distingue nada; en el momento de maximo enganche el brazo esta extendido y la
         cazoleta se ve.

    Esto es eleccion de QUE MIRAR para anotar, no un filtro de la medida: `mano_guante` muestrea
    uniformemente sobre todos los cuadros utiles y no usa esta funcion.
    """
    ks = MG.utiles(d)
    if len(ks) == 0:
        return []
    B = d["B"]
    KP = d["KP"]
    ra, rb = KP[ks, lado, 21].astype("float64"), KP[ks, lado, 42].astype("float64")
    ca, cb = KP[ks, lado, 8].astype("float64"), KP[ks, lado, 7].astype("float64")
    sman = np.linalg.norm(ra - rb, axis=1)
    r = MG.R_PARCHE * 0.5 * (np.linalg.norm(ra - ca, axis=1) + np.linalg.norm(rb - cb, axis=1))
    dentro = (B[ks, lado, 0] > 0.01) & (B[ks, lado, 2] < 0.99)
    alto = B[ks, lado, 3] - B[ks, lado, 1]
    sep = np.abs(0.5 * (B[ks, 1, 0] + B[ks, 1, 2]) - 0.5 * (B[ks, 0, 0] + B[ks, 0, 2]))
    keep = dentro & (alto >= np.nanmedian(alto)) & (sman >= MG.SOLAPE_MIN * r)
    if keep.sum() < n:
        keep = dentro & (sman >= MG.SOLAPE_MIN * r)
    if keep.sum() < n:
        keep = dentro if dentro.sum() >= n else np.ones(len(ks), bool)
    cand, sc = ks[keep], sep[keep]
    out = []
    for g in np.array_split(np.arange(len(cand)), n):
        if len(g) == 0:
            continue
        out.append(int(cand[g[int(np.nanargmin(sc[g]))]]))
    return out


def marca(im, KP, k, lado, etiquetas=True):
    """Dibuja `codo -> raiz` y un circulo en la raiz de las dos manos, con su letra."""
    for nom, M in CL.MANOS.items():
        raiz = KP[k, lado, M["root"]]
        codo = KP[k, lado, M["codo"]]
        if not (np.isfinite(raiz).all() and np.isfinite(codo).all()):
            continue
        p, q = tuple(np.round(codo).astype(int)), tuple(np.round(raiz).astype(int))
        cv2.line(im, p, q, COL[nom], 2, cv2.LINE_AA)
        cv2.circle(im, q, 9, COL[nom], 2, cv2.LINE_AA)
        if etiquetas:
            cv2.putText(im, nom, (q[0] + 11, q[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        COL[nom], 2, cv2.LINE_AA)
    return im


def recorte_caja(im, B, k, lado, ancho_rel=1.6):
    x0, y0, x1, y1 = B[k, lado] * np.array([im.shape[1], im.shape[0], im.shape[1], im.shape[0]])
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    w, h = ancho_rel * (x1 - x0), ancho_rel * (y1 - y0)
    a, b = int(max(0, cx - 0.5 * w)), int(min(im.shape[1], cx + 0.5 * w))
    c, e = int(max(0, cy - 0.5 * h)), int(min(im.shape[0], cy + 0.5 * h))
    return im[c:e, a:b]


def zoom_manos(im, KP, k, lado, lado_px=ALTO_DET, rasgo=None):
    """UN SOLO recorte con LAS DOS raices marcadas dentro, en vez de dos recortes separados.

    EL PRIMER DISENIO ERA MALO Y SE CAMBIO CON EL MOTIVO MEDIDO: con un recorte por mano de semilado
    `2r`, cuando las dos raices estan a menos de `2r` una de otra los dos recortes ENSENIAN CASI LO
    MISMO y no se puede decir cual marcador esta sobre la mano del arma. Paso en
    `BabyishIdioticCockerspaniel` lado 1 y en `BlissfulTiredAiredale` lado 0. Con un recorte comun y
    los dos marcadores pintados encima, la pregunta «cual de los dos puntos cae sobre el guante y el
    arma» se contesta mirando, y ademas se VE cuando las dos manos estan pegadas.
    """
    ps, cs, Ls = {}, {}, []
    for nom, M in CL.MANOS.items():
        ps[nom] = KP[k, lado, M["root"]].astype("float64")
        cs[nom] = KP[k, lado, M["codo"]].astype("float64")
        Ls.append(float(np.linalg.norm(ps[nom] - cs[nom])))
    r = MG.R_PARCHE * float(np.mean(Ls))
    cen = 0.5 * (ps["A"] + ps["B"])
    sep = float(np.linalg.norm(ps["A"] - ps["B"]))
    half = max(12.0, 2.2 * r, 0.6 * sep + 1.7 * r)
    a, b = int(round(cen[0] - half)), int(round(cen[0] + half))
    c, e = int(round(cen[1] - half)), int(round(cen[1] + half))
    lienzo = np.zeros((max(e - c, 3), max(b - a, 3), 3), "uint8")
    a2, b2, c2, e2 = max(0, a), min(im.shape[1], b), max(0, c), min(im.shape[0], e)
    if b2 > a2 and e2 > c2:
        lienzo[c2 - c:e2 - c, a2 - a:b2 - a] = im[c2:e2, a2:b2]
    z = cv2.resize(lienzo, (lado_px, lado_px), interpolation=cv2.INTER_LANCZOS4)
    s = lado_px / float(lienzo.shape[1])
    for nom in ("A", "B"):
        q = (int(round((ps[nom][0] - a) * s)), int(round((ps[nom][1] - c) * s)))
        cv2.circle(z, q, int(max(4, MG.R_PARCHE * Ls[0] * s * 0.25)), COL[nom], 2, cv2.LINE_AA)
        cv2.line(z, (q[0] - 14, q[1]), (q[0] + 14, q[1]), COL[nom], 1, cv2.LINE_AA)
        cv2.line(z, (q[0], q[1] - 14), (q[0], q[1] + 14), COL[nom], 1, cv2.LINE_AA)
        tx = q[0] + 16 if q[0] < 0.60 * lado_px else q[0] - 150
        cv2.putText(z, nom, (tx, q[1] + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    COL[nom], 2, cv2.LINE_AA)
    """Los rasgos van A PIE DE PANEL y no junto a cada marcador: cuando las dos manos quedan a la
    misma altura los dos rotulos se pisaban, y eso salio en `elegida_BountifulSilentEarwig_l0`. A
    pie de panel no se pueden solapar nunca, y ademas se leen los dos de un golpe."""
    if rasgo is not None:
        for i, nom in enumerate(("A", "B")):
            if nom in rasgo:
                cv2.putText(z, "%s: white=%.2f blob=%.2f dark=%.2f" % (
                    nom, rasgo[nom]["f_blanco"], rasgo[nom]["a_blob"], rasgo[nom]["f_oscuro"]),
                    (6, lado_px - 30 + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL[nom], 1,
                    cv2.LINE_AA)
    """Arriba a la derecha cuando hay rasgos abajo, para no pisarlos. La version anterior los ponia
    los dos abajo y se cruzaban en horizontal."""
    t = "hand gap=%.0fpx  patch r=%.0fpx" % (sep, r)
    an = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    cv2.putText(z, t, (6, lado_px - 8) if rasgo is None else (lado_px - an - 6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return z


def pega(filas, ancho):
    """Apila filas de alturas distintas rellenando con negro hasta `ancho`."""
    out = []
    for f in filas:
        if f.shape[1] < ancho:
            f = np.pad(f, ((0, 0), (0, ancho - f.shape[1]), (0, 0)))
        out.append(f[:, :ancho])
    return np.vstack(out)


def mosaico(stem, lado, n_cuadros=NFRAMES, etiqueta=None):
    """`etiqueta` sustituye al nombre del clip en la cabecera. Con `etiqueta='CIEGO 07'` el mosaico
    no dice de que clip ni de que lado es, que es lo que hace la anotacion ciega de verdad."""
    d = MG.carga(stem)
    W, H = int(d["W"]), int(d["H"])
    ks = elige_cuadros(d, lado, n_cuadros)
    if not ks:
        return None
    cap = cv2.VideoCapture(str(d["video"]))
    filas = []
    for i, k in enumerate(ks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, im = cap.read()
        if not ok:
            continue
        pin = marca(im.copy(), d["KP"], k, lado)
        if i == 0:
            ctx = pin.copy()
            x0, y0, x1, y1 = d["B"][k, lado] * np.array([W, H, W, H])
            cv2.rectangle(ctx, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), 2)
            ctx = cv2.resize(ctx, (ANCHO_CTX, int(ANCHO_CTX * H / W)))
            cab = etiqueta if etiqueta else "%s   side %d   (context)" % (stem[:34], lado)
            """Banda negra detras del titulo: los videos traen su propio rotulo de retransmision
            arriba a la izquierda y los dos textos se pisaban, que en una figura publica se ve."""
            an = cv2.getTextSize(cab, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
            cv2.rectangle(ctx, (0, 0), (min(ctx.shape[1], an + 18), 36), (0, 0, 0), -1)
            cv2.putText(ctx, cab, (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            filas.append(ctx)
        rc = recorte_caja(pin, d["B"], k, lado)
        if rc.size == 0:
            continue
        rc = cv2.resize(rc, (max(60, int(ALTO_DET * rc.shape[1] / rc.shape[0])), ALTO_DET))
        fila = np.hstack([rc, zoom_manos(im, d["KP"], k, lado)])
        cv2.putText(fila, "k=%d" % k, (4, ALTO_DET - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        filas.append(fila)
    cap.release()
    if len(filas) < 2:
        return None
    ancho = max(f.shape[1] for f in filas)
    return pega(filas, ancho)


def anota(stems=None, n_cuadros=NFRAMES, lados=(0, 1)):
    """Si el primer argumento de la linea de ordenes es un entero, se usa como numero de cuadros.

    Sirve para volver sobre un tirador que quedo AMBIGUO con tres cuadros, sin rehacer los otros.
    """
    SAL.mkdir(parents=True, exist_ok=True)
    stems = stems or MG.clips()
    n = 0
    for stem in stems:
        for lado in lados:
            m = mosaico(stem, lado, n_cuadros)
            if m is None:
                print("  %-40s l%d SIN CUADROS UTILES" % (stem, lado))
                continue
            p = SAL / ("anota_%s_l%d.png" % (stem, lado))
            cv2.imwrite(str(p), m)
            n += 1
            print("  %s" % p.name)
    print("\n%d mosaicos en %s" % (n, SAL))


SEMILLA_CIEGO = 20260911
CLAVE = SAL / "clave_ciego.csv"


def ciego(stems, n_cuadros=NFRAMES, solo=None):
    """Mosaicos numerados y barajados, sin el nombre del clip, y la clave en un fichero aparte.

    `solo` es una lista de `NN` para la segunda pasada: reusa la clave ya escrita, asi que el mismo
    tirador conserva su numero y el anotador sigue sin saber cual es.
    """
    import random
    SAL.mkdir(parents=True, exist_ok=True)
    if solo and CLAVE.exists():
        with open(CLAVE, newline="", encoding="utf-8") as fh:
            pares = [(r["nn"], r["clip"], int(r["lado"]))
                     for r in csv.DictReader(l for l in fh if not l.startswith("#"))]
        pares = [p for p in pares if p[0] in set(solo)]
    else:
        pares = [(stem, lado) for stem in stems for lado in (0, 1)]
        random.Random(SEMILLA_CIEGO).shuffle(pares)
        pares = [("%02d" % (i + 1), s, l) for i, (s, l) in enumerate(pares)]
        with open(CLAVE, "w", newline="", encoding="utf-8") as fh:
            fh.write("# CLAVE de la anotacion ciega. NO se mira hasta que la anotacion esta\n"
                     "# escrita. Orden barajado con random.Random(%d) sobre los clips en el orden\n"
                     "# en que se pasaron. Generada el 2026-09-11.\n" % SEMILLA_CIEGO)
            w = csv.DictWriter(fh, fieldnames=["nn", "clip", "lado"])
            w.writeheader()
            for nn, s, l in pares:
                w.writerow({"nn": nn, "clip": s, "lado": l})
    n = 0
    for nn, stem, lado in pares:
        m = mosaico(stem, lado, n_cuadros, etiqueta="BLIND %s   which hand holds the weapon?" % nn)
        if m is None:
            print("  ciego_%s SIN CUADROS UTILES" % nn)
            continue
        cv2.imwrite(str(SAL / ("ciego_%s.png" % nn)), m)
        n += 1
        print("  ciego_%s.png" % nn)
    print("\n%d mosaicos ciegos en %s ; clave en %s" % (n, SAL, CLAVE.name))


RESUMEN = [("BountifulSilentEarwig", 0, "clean case"),
           ("WholeTallAlpineroadguidetigerbeetle", 1, "clean case"),
           ("UglyNiceAplomadofalcon", 1, "old rule fails here"),
           ("WhiteMarriedGuineafowl", 0, "flagged UNCERTAIN")]
"""Los cuatro paneles de `resumen_mano_armada.png`, elegidos con un motivo cada uno: dos casos
limpios del conjunto RESERVADO con la hoja visible y voto del pie de 0,95 y 0,99; el caso donde el
criterio viejo falla porque el arma esta en la mano del lado CONTRARIO al rival todo el clip; y un
caso que el sistema marca `uncertain` y por tanto no usa."""


def cuadro_mas_legible(d, lado):
    """Para la figura de sintesis hace falta el cuadro mas LEGIBLE, que no es el mismo que el mas
    informativo para anotar.

    `elige_cuadros` busca el maximo enganche, o sea el instante en que los dos tiradores estan mas
    cerca, y ahi **se solapan** y el panel sale confuso: pasó en los dos primeros paneles del primer
    intento. Aqui se pide lo contrario: caja entera de lado a lado, los dos tiradores SEPARADOS por
    encima de su mediana, las dos manos separadas, y de todo eso la caja mas ALTA.
    """
    ks = MG.utiles(d)
    if len(ks) == 0:
        return [0]
    B, KP = d["B"], d["KP"]
    dentro = (B[ks, lado, 0] > 0.02) & (B[ks, lado, 2] < 0.98)
    alto = B[ks, lado, 3] - B[ks, lado, 1]
    sep = np.abs(0.5 * (B[ks, 1, 0] + B[ks, 1, 2]) - 0.5 * (B[ks, 0, 0] + B[ks, 0, 2]))
    ra, rb = KP[ks, lado, 21].astype("float64"), KP[ks, lado, 42].astype("float64")
    ca, cb = KP[ks, lado, 8].astype("float64"), KP[ks, lado, 7].astype("float64")
    sman = np.linalg.norm(ra - rb, axis=1)
    r = MG.R_PARCHE * 0.5 * (np.linalg.norm(ra - ca, axis=1) + np.linalg.norm(rb - cb, axis=1))
    keep = dentro & (sep >= np.nanmedian(sep)) & (sman >= MG.SOLAPE_MIN * r)
    if keep.sum() == 0:
        keep = dentro if dentro.sum() else np.ones(len(ks), bool)
    cand = ks[keep]
    return [int(cand[int(np.nanargmax(B[cand, lado, 3] - B[cand, lado, 1]))])]


COL_ESQ = {0: (150, 210, 150), 1: (150, 190, 230)}
"""Colores del esqueleto en el modo PUBLICO, deliberadamente APAGADOS.

En el modo normal el esqueleto no se pinta porque el fondo es el fotograma y la flecha verde ya se ve.
Sobre negro el esqueleto es lo unico que hay, y si se pintara en verde saturado competiria con la
flecha del pie adelantado, que es lo que la figura tiene que dejar claro. Asi que el cuerpo va en tono
pastel y la anotacion se queda con el verde y el ambar."""


def panel_resumen(stem, lado, nota, alto=430, publico=False):
    """Un panel de la figura de sintesis: mano armada elegida + PIE ADELANTADO, y que se vea que son
    del mismo lado, que es el criterio adoptado y el que resulta contraintuitivo.

    `publico=True` dibuja el esqueleto sobre NEGRO en vez de sobre el fotograma. Es la version que
    puede salir del proyecto: el material de origen es grabacion de television con atletas
    identificables, y un esqueleto de 308 puntos es una representacion derivada, no la imagen. El
    resto del panel -el recorte, las flechas, el circulo de la mano y los pies de texto con las
    cifras- es identico en los dos modos **a proposito**: si la version publica se dibujara con otro
    codigo, podria acabar ensenando algo que no es lo que se midio.
    """
    import csv as _csv
    with open(MG.PRUEBAS / "mano_guante.csv", newline="", encoding="utf-8") as fh:
        G = {(r["clip"], int(r["lado"])): r for r in _csv.DictReader(fh)}
    r = G[(stem, lado)]
    d = MG.carga(stem)
    W, H = int(d["W"]), int(d["H"])
    ks = cuadro_mas_legible(d, lado)
    k = int(ks[0])
    if publico:
        sys.path.insert(0, str(pathlib.Path(__file__).parent))
        import video_revision as VR
        im = np.zeros((H, W, 3), "uint8")
        for l2 in (0, 1):
            VR._pinta(im, d["KP"][k, l2], d["SC"][k, l2], COL_ESQ[l2])
    else:
        cap = cv2.VideoCapture(str(d["video"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, im = cap.read()
        cap.release()
        if not ok:
            return None
    mano = r["adoptada"]
    dudosa = r["dudosa"] == "True"
    verde, ambar = (0, 235, 0), (0, 200, 255)
    col = ambar if dudosa else verde
    """El pie de delante es el que gana en `pie_x`: eje `talon -> dedos` mas alineado con la pista."""
    ejes = {}
    for nom, P in CL.PIES.items():
        med = 0.5 * (d["KP"][k, lado, P["dedos"][0]] + d["KP"][k, lado, P["dedos"][1]])
        v = med - d["KP"][k, lado, P["talon"]]
        n = np.linalg.norm(v)
        ejes[nom] = (abs(v[0]) / max(n, 1e-9), d["KP"][k, lado, P["talon"]], med)
    pie = "A" if ejes["A"][0] > ejes["B"][0] else "B"
    x0, y0, x1, y1 = d["B"][k, lado] * np.array([W, H, W, H])
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    w, h = 1.95 * (x1 - x0), 1.30 * (y1 - y0)
    a, b = int(max(0, cx - 0.5 * w)), int(min(W, cx + 0.5 * w))
    c, e = int(max(0, cy - 0.5 * h)), int(min(H, cy + 0.5 * h))
    z = im[c:e, a:b].copy()
    s = alto / float(max(z.shape[0], 1))
    z = cv2.resize(z, (max(60, int(z.shape[1] * s)), alto), interpolation=cv2.INTER_LANCZOS4)

    def P(p):
        return (int(round((p[0] - a) * s)), int(round((p[1] - c) * s)))

    qh = P(d["KP"][k, lado, CL.MANOS[mano]["root"]])
    qt, qd = P(ejes[pie][1]), P(ejes[pie][2])
    cv2.arrowedLine(z, qt, qd, col, 3, cv2.LINE_AA, tipLength=0.45)
    cv2.line(z, qd, qh, col, 1, cv2.LINE_AA)
    cv2.circle(z, qh, 16, col, 3, cv2.LINE_AA)
    dx = -150 if qh[0] > 0.55 * z.shape[1] else 22
    cv2.putText(z, "UNCERTAIN" if dudosa else "WEAPON HAND", (max(2, qh[0] + dx), qh[1] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    dxf = -130 if qd[0] > 0.55 * z.shape[1] else 14
    cv2.putText(z, "FRONT FOOT", (max(2, qd[0] + dxf), min(alto - 6, qd[1] + 26)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
    if nota == "old rule fails here":
        xr = 0.5 * (d["B"][k, 1 - lado, 0] + d["B"][k, 1 - lado, 2]) * W
        ly = int(0.12 * alto)
        p1 = (z.shape[1] - 8, ly) if xr > cx else (8, ly)
        p2 = (z.shape[1] - 90, ly) if xr > cx else (90, ly)
        cv2.arrowedLine(z, p2, p1, (60, 60, 255), 3, cv2.LINE_AA, tipLength=0.35)
        cv2.putText(z, "opponent", (p2[0] - (86 if xr > cx else 0), ly - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 2, cv2.LINE_AA)
    txt = ["%s  side %d" % (stem[:26], lado),
           "front foot %s = hand %s side" % (pie, pie),
           ("UNCERTAIN: foot vote %.2f < 0.85" % float(r["frac_pie_x"]) if dudosa else
            "weapon hand %s, foot vote %.2f" % (mano, float(r["frac_pie_x"])))]
    if nota == "old rule fails here":
        txt.append("old rule said %s: the FAR hand" % r["geo_a"])
    elif nota == "clean case":
        txt.append("held-out clip, unseen by the rule")
    else:
        txt.append("declines instead of guessing")
    """La escala del texto se BAJA hasta que la linea mas larga entra en el panel. En el primer
    intento estaba fija en 0,45 y se cortaban tres pies de panel («...the r», «...the oppone»): un
    rotulo cortado en una figura publica es un error de los que se ven, no de los que se miden."""
    esc = 0.45
    while esc > 0.26 and max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, esc, 1)[0][0]
                             for t in txt) > z.shape[1] - 12:
        esc -= 0.01
    dy = int(round(20 * esc / 0.45)) + 4
    pie_txt = np.zeros((dy * len(txt) + 10, z.shape[1], 3), "uint8")
    for i, t in enumerate(txt):
        cv2.putText(pie_txt, t, (6, dy - 4 + dy * i), cv2.FONT_HERSHEY_SIMPLEX, esc,
                    (255, 255, 255) if i == 0 else (col if i == 2 else (200, 200, 200)),
                    1, cv2.LINE_AA)
    return np.vstack([z, pie_txt])


def resumen(publico=False):
    """`resumen_mano_armada.png`: la figura de sintesis para el README, toda en ingles.

    Con `publico=True` escribe `resumen_mano_armada_generado.png`, que es la misma figura con el
    esqueleto sobre negro en lugar del fotograma, y con una linea mas en el pie que lo declara.
    """
    SAL.mkdir(parents=True, exist_ok=True)
    ps = [p for p in (panel_resumen(s, l, n, publico=publico) for s, l, n in RESUMEN)
          if p is not None]
    hh = max(p.shape[0] for p in ps)
    ps = [np.pad(p, ((0, hh - p.shape[0]), (0, 12), (0, 0))) for p in ps]
    cuerpo = np.hstack(ps)
    Wf = cuerpo.shape[1]
    cab = np.zeros((92, Wf, 3), "uint8")
    cv2.putText(cab, "FENCINGLY - which hand holds the weapon?", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(cab, "First rung of the cascade: in the on-guard stance the FRONT FOOT is on the"
                " same side as the weapon arm. Green = chosen, amber = uncertain.", (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)
    cv2.putText(cab, "Hands are labelled A and B by keypoint index only: A = hand root 21,"
                " B = hand root 42. No left/right assumption.", (10, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
    pie = np.zeros((100 + (20 if publico else 0), Wf, 3), "uint8")
    cv2.putText(pie, "Old rule, closest hand to the opponent: 0.831 accuracy over 59 - wrong about"
                " 1 in 6 (0.741 on the first 27 annotated).",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (140, 140, 255), 2, cv2.LINE_AA)
    cv2.putText(pie, "The CASCADE: 1.000 on 19 of 19 fencers of a HELD-OUT set that played no part"
                " in choosing it, and 0.949 over all 59.", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (140, 255, 140), 2, cv2.LINE_AA)
    cv2.putText(pie, "The FRONT-FOOT rung on its own, firm vote: 12 of 12 with no failure, covering"
                " 12 of those 19. Ungated, 0.895 over 19.", (10, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (140, 255, 200), 1, cv2.LINE_AA)
    """LAS DOS CIFRAS POR SEPARADO Y EN LA MISMA IMAGEN. Este PNG llevaba QUEMADO *"New rule: 1.000
    accuracy on 19 of 19"* debajo de un titulo que decia que la regla era el pie: o sea que atribuia
    a la regla del pie el acierto Y la cobertura de la cascada entera. El 1,000 con cobertura total
    es de la cascada; la regla del pie sola cubre 12 de 19. Las dos son ciertas y juntarlas no."""
    cv2.putText(pie, "Ground truth annotated by eye on blind mosaics, not by a referee. 59 fencers,"
                " 33 clips. Fencing rule: FIE t.115.1.", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
    if publico:
        cv2.putText(pie, "GENERATED LAYERS ONLY: the body is our 308-keypoint skeleton drawn on"
                    " black. No broadcast frame is shown.", (10, 114),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (205, 220, 245), 1, cv2.LINE_AA)
    img = np.vstack([cab, cuerpo, pie])
    p = SAL / ("resumen_mano_armada_generado.png" if publico else "resumen_mano_armada.png")
    cv2.imwrite(str(p), img)
    print("%s  %dx%d" % (p.name, img.shape[1], img.shape[0]))


def extras():
    """Las tres figuras que se hicieron con scripts sueltos, ahora reproducibles y EN INGLES.

    `lupa_seguro_y_equivocado`: los dos casos en que el criterio viejo esta seguro y equivocado.
    `discrepa_pie_nariz_*`: el unico tirador donde el pie y la nariz discrepan, con los ejes de los
    pies pintados. `lupa_IlliterateDisloyalFulmar_l0`: el caso que se dejo en `?` a proposito.
    """
    SAL.mkdir(parents=True, exist_ok=True)
    tiras = []
    for stem, lado, ks in (("UglyNiceAplomadofalcon", 1, (227, 241, 257)),
                           ("WhiteMarriedGuineafowl", 0, (280, 295, 309))):
        d = MG.carga(stem)
        W = int(d["W"])
        cap = cv2.VideoCapture(str(d["video"]))
        for k in ks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, im = cap.read()
            if not ok:
                continue
            fila = []
            for nom in ("A", "B"):
                p = d["KP"][k, lado, CL.MANOS[nom]["root"]].astype("float64")
                L = float(np.linalg.norm(p - d["KP"][k, lado, CL.MANOS[nom]["codo"]]))
                r = int(max(22, 0.7 * L))
                a, b = int(p[0]) - r, int(p[0]) + r
                c, e = int(p[1]) - r, int(p[1]) + r
                lz = np.zeros((2 * r, 2 * r, 3), "uint8")
                a2, b2 = max(0, a), min(im.shape[1], b)
                c2, e2 = max(0, c), min(im.shape[0], e)
                lz[c2 - c:e2 - c, a2 - a:b2 - a] = im[c2:e2, a2:b2]
                z = cv2.resize(lz, (280, 280), interpolation=cv2.INTER_LANCZOS4)
                cv2.drawMarker(z, (140, 140), (0, 255, 255), cv2.MARKER_CROSS, 26, 2)
                cv2.putText(z, "hand %s   %s  side %d  frame %d" % (nom, stem[:11], lado, k),
                            (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                            cv2.LINE_AA)
                fila.append(z)
            xr = 0.5 * (d["B"][k, 1 - lado, 0] + d["B"][k, 1 - lado, 2]) * W
            cv2.putText(fila[0], "opponent at x=%.0f px" % xr, (5, 272),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            tiras.append(np.hstack(fila))
        cap.release()
    cab = np.zeros((60, tiras[0].shape[1], 3), "uint8")
    cv2.putText(cab, "Confident and wrong: the weapon is in the hand FAR from the opponent",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(cab, "all clip long, so the old rule is wrong in 100 % of the frames",
                (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(SAL / "lupa_seguro_y_equivocado.png"), np.vstack([cab] + tiras))
    print("lupa_seguro_y_equivocado.png")

    for stem, lado, ks, nom, tit in (
            ("WhisperedTenderDeinonychus", 0, (31, 45, 72),
             "discrepa_pie_nariz_WhisperedTenderDeinonychus_l0.png",
             "Foot and nose disagree: the fencer is WALKING, not on guard"),
            ("IlliterateDisloyalFulmar", 0, None, "lupa_IlliterateDisloyalFulmar_l0.png",
             "Left as UNDECIDABLE on purpose: all four criteria say B, my two blind passes said A")):
        d = MG.carga(stem)
        W, H = int(d["W"]), int(d["H"])
        if ks is None:
            u = MG.utiles(d)
            ks = tuple(int(u[int(x)]) for x in np.linspace(0, len(u) - 1, 3))
        cap = cv2.VideoCapture(str(d["video"]))
        fs = []
        for k in ks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, im = cap.read()
            if not ok:
                continue
            x0, y0, x1, y1 = d["B"][k, lado] * np.array([W, H, W, H])
            cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            w, h = 1.9 * (x1 - x0), 1.3 * (y1 - y0)
            a, b = int(max(0, cx - 0.5 * w)), int(min(W, cx + 0.5 * w))
            c, e = int(max(0, cy - 0.5 * h)), int(min(H, cy + 0.5 * h))
            z = im[c:e, a:b].copy()
            for nm, cl2 in (("A", COL["A"]), ("B", COL["B"])):
                q = np.round(d["KP"][k, lado, CL.MANOS[nm]["root"]] - [a, c]).astype(int)
                cv2.circle(z, tuple(q), 7, cl2, 2, cv2.LINE_AA)
                cv2.putText(z, nm, (q[0] + 9, q[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cl2, 2)
                Pp = CL.PIES[nm]
                t = d["KP"][k, lado, Pp["talon"]] - [a, c]
                m = 0.5 * (d["KP"][k, lado, Pp["dedos"][0]]
                           + d["KP"][k, lado, Pp["dedos"][1]]) - [a, c]
                cv2.arrowedLine(z, tuple(np.round(t).astype(int)), tuple(np.round(m).astype(int)),
                                cl2, 2, cv2.LINE_AA, tipLength=0.3)
            nz = d["KP"][k, lado, 0] - [a, c]
            cv2.circle(z, tuple(np.round(nz).astype(int)), 6, (0, 255, 0), 2)
            cv2.putText(z, "nose", (int(nz[0]) + 8, int(nz[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            s = 520.0 / z.shape[0]
            z = cv2.resize(z, (int(z.shape[1] * s), 520), interpolation=cv2.INTER_LANCZOS4)
            cv2.putText(z, "frame %d   opponent at x=%.0f px  (arrows = foot axis heel->toes)" % (
                k, 0.5 * (d["B"][k, 1 - lado, 0] + d["B"][k, 1 - lado, 2]) * W), (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2, cv2.LINE_AA)
            fs.append(z)
        cap.release()
        wm = max(f.shape[1] for f in fs)
        cab = np.zeros((40, wm, 3), "uint8")
        cv2.putText(cab, tit, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.imwrite(str(SAL / nom), np.vstack(
            [cab] + [np.pad(f, ((0, 0), (0, wm - f.shape[1]), (0, 0))) for f in fs]))
        print(nom)


def lee_verdad():
    if not VERDAD.exists():
        return {}
    out = {}
    with open(VERDAD, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith("#")):
            out[(r["clip"], int(r["lado"]))] = r
    return out


def lee_guante():
    p = MG.PRUEBAS / "mano_guante.csv"
    if not p.exists():
        return {}
    with open(p, newline="", encoding="utf-8") as fh:
        return {(r["clip"], int(r["lado"])): r for r in csv.DictReader(fh)}


def figuras():
    """Mano elegida marcada con el rasgo del guante al lado, y aparte las discrepancias."""
    SAL.mkdir(parents=True, exist_ok=True)
    ver, gua = lee_verdad(), lee_guante()
    if not gua:
        print("falta mano_guante.csv: corre antes mano_guante.py")
        return
    n_ok = n_dis = 0
    for (stem, lado), g in sorted(gua.items()):
        d = MG.carga(stem)
        ks = elige_cuadros(d, lado, 2)
        if not ks:
            continue
        cap = cv2.VideoCapture(str(d["video"]))
        filas = []
        for k in ks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, im = cap.read()
            if not ok:
                continue
            r = MG.del_cuadro(im, d["KP"], int(k), lado)
            pin = marca(im.copy(), d["KP"], k, lado, etiquetas=False)
            elegida = g["adoptada"]
            dud = g["dudosa"] == "True"
            cl2 = (0, 200, 255) if dud else (0, 235, 0)
            M = CL.MANOS.get(elegida)
            if M is not None:
                q = tuple(np.round(d["KP"][k, lado, M["root"]]).astype(int))
                cv2.circle(pin, q, 16, cl2, 3, cv2.LINE_AA)
                et = ("UNCERTAIN " if dud else "WEAPON ") + elegida
                """El rotulo se pone a la IZQUIERDA del marcador cuando el marcador esta en la mitad
                derecha del fotograma, porque si no se sale del recorte de la caja. En la primera
                tirada en ingles salia `WEAPON` y `WEAPE` cortados."""
                anc = cv2.getTextSize(et, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
                tx = q[0] + 18 if q[0] < 0.5 * pin.shape[1] else max(2, q[0] - 18 - anc)
                cv2.putText(pin, et, (tx, max(16, q[1] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            cl2, 2, cv2.LINE_AA)
            rc = recorte_caja(pin, d["B"], k, lado)
            if rc.size == 0:
                continue
            rc = cv2.resize(rc, (max(60, int(ALTO_DET * rc.shape[1] / rc.shape[0])), ALTO_DET))
            filas.append(np.hstack([rc, zoom_manos(im, d["KP"], k, lado, rasgo=r)]))
        cap.release()
        if not filas:
            continue
        cab = np.zeros((78, max(f.shape[1] for f in filas), 3), "uint8")
        v = ver.get((stem, lado), {})
        cv2.putText(cab, "%s   side %d   %s" % (stem[:36], lado, g["res"]), (6, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        """La cabecera ensenia lo ADOPTADO, no la combinacion del guante, que esta retirada. En la
        primera tirada en ingles ponia `final=A by old rule (glove withdrawn)` en un caso donde el
        sistema adoptado dice B: correcto de dato y enganioso de lectura."""
        cv2.putText(cab, "ADOPTED: %s by %s%s   |   old rule=%s   glove=%s (vote %.2f)" % (
            g["adoptada"], EN_POR.get(g["por_adoptada"], g["por_adoptada"]),
            "  [UNCERTAIN]" if g["dudosa"] == "True" else "", g["geo_a"], g["guante"],
            float(g["frac_voto"] or "nan")), (6, 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(cab, "ground truth by eye=%s (%s)   frame vote=%s (%s)   glove agrees=%s" % (
            v.get("mano", "?"), EN_CUE.get(v.get("cue", ""), v.get("cue", "?")), g["a_voto"],
            g["frac_a_voto"][:4], g["concuerdan"]), (6, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 220, 255), 1, cv2.LINE_AA)
        img = pega([cab] + filas, max(f.shape[1] for f in filas))
        dis = g["concuerdan"] == "False"
        falla = v.get("mano") in ("A", "B") and g["geo_a"] != v["mano"]
        pre = "falla_a" if falla else ("discrepa" if dis else "elegida")
        cv2.imwrite(str(SAL / ("%s_%s_l%d.png" % (pre, stem, lado))), img)
        n_dis += dis
        n_ok += 1
    print("%d figuras, %d con discrepancia geometria/guante, en %s" % (n_ok, n_dis, SAL))


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "anota"
    if modo == "anota":
        rest = sys.argv[2:]
        nn = NFRAMES
        if rest and rest[0].isdigit():
            nn, rest = int(rest[0]), rest[1:]
        lad = (0, 1)
        if rest and rest[-1] in ("l0", "l1"):
            lad, rest = (int(rest[-1][1]),), rest[:-1]
        anota(rest or None, nn, lad)
    elif modo == "ciego":
        rest = sys.argv[2:]
        nn = NFRAMES
        if rest and rest[0].isdigit() and len(rest[0]) == 1:
            nn, rest = int(rest[0]), rest[1:]
        num = [x for x in rest if x.isdigit()]
        ciego([x for x in rest if not x.isdigit()], nn, num or None)
    elif modo == "resumen":
        resumen(publico="publico" in sys.argv[2:])
    elif modo == "extras":
        extras()
    elif modo == "figuras":
        figuras()
    else:
        print(__doc__)

"""Piezas comunes de las figuras publicas de PRIORIDAD. TODO EL TEXTO QUE SALE ES ASCII.

No reimplementa nada: el cono sale de `e21_pinta_cono_arreglado` (`rellena_peso`, `arco`), las
aperturas de `cono3d_lib`, el esqueleto de `video_revision`/`video_prioridad._pinta`, las cajas y la
pareja de `capas_limpias`, y la decision de `prioridad.decide`.
"""
from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

COD = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
HER = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\herramienta")
for p in (str(COD), str(HER)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cono3d_lib as C3            # noqa: E402  aperturas y eje 2d del brazo
import cono_lib as CL              # noqa: E402  MANOS, HOJA_M, escala_px_m, mano_armada
import e21_pinta_cono_arreglado as E21   # noqa: E402  rellena_peso, arco, trazos
import prioridad as PR             # noqa: E402  senales, decide, paradas
import video_prioridad as VP       # noqa: E402  _pinta (esqueleto), COL, NOM
from panel_publico import SinPisarse   # noqa: E402  rotulos que no se pisan

PRUEBAS = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
PRIO = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\prioridad")
SALIDA = PRIO / "publico"

# Convenio de color del proyecto, el mismo de `video_prioridad`: BGR.
VERDE = (60, 240, 60)      # lado 0 = LEFT
NARANJA = (60, 160, 255)   # lado 1 = RIGHT
GRIS = (170, 170, 170)     # sin prioridad
VIOLETA = (200, 80, 220)   # pose inventada
"""La MISMA que `video_revision.VIOLETA`, que es el sitio de donde sale. Aqui se repite el
literal porque este paquete se copia suelto al repo publico y no puede importar la
herramienta; si se cambia alla, hay que cambiarla aqui, y por eso queda dicho."""
ROJO = (70, 70, 235)
AMARILLO = (60, 230, 250)
BLANCO = (245, 245, 245)
FONDO = (20, 20, 22)
COL = {0: VERDE, 1: NARANJA, -1: GRIS}
LADO = {0: "LEFT", 1: "RIGHT", -1: "NOBODY"}

NUCLEO_SEMI = C3.APERT_NUCLEO_SEMI_GR          # 14.4
ENVOL_SEMI = C3.APERT_ENVOLVENTE_SEMI_GR       # 33.0
ALCANCE_M = C3.ALCANCE_M                       # 1.10 m de muneca a punta
F = cv2.FONT_HERSHEY_SIMPLEX
FD = cv2.FONT_HERSHEY_DUPLEX


def A(s):
    """PUERTA ASCII. `cv2.putText` solo dibuja fuentes Hershey: un grado, un mas-menos o un punto
    medio salen como basura en la imagen y nadie lo ve hasta que la figura esta publicada. Asi que
    cualquier cadena que vaya a un `putText` pasa por aqui y revienta si no es ASCII."""
    s = str(s)
    try:
        s.encode("ascii")
    except UnicodeEncodeError as e:
        raise AssertionError("TEXTO NO ASCII en una figura publica: %r (%s)" % (s, e))
    return s


def texto(img, s, xy, esc=0.5, col=BLANCO, grosor=1, fuente=F, sombra=True):
    """Texto con contorno negro para que se lea sobre el fotograma.

    EL CONTORNO SE DIBUJA DESPLAZANDO, NO ENGORDANDO, y eso es un ARREGLO, no un gusto. En OpenCV
    5.0 el AVANCE de los glifos Hershey crece con el grosor del trazo: la misma cadena a 0,44 mide
    171 px con grosor 1 y 183 px con grosor 3 (`getTextSize` lo dice). Asi que el truco clasico de
    pintar la cadena en negro con `grosor + 2` y encima en color con `grosor` deja **12 px de cola
    negra asomando por la derecha**: en una figura de esta pagina se leia «= 2.54» y detras un «54»
    fantasma en negro, y parecia un rotulo pisado por otro cuando era el propio contorno.

    Desplazar en las ocho direcciones con EL MISMO grosor mantiene el avance identico, asi que el
    contorno y el relleno cubren exactamente los mismos pixeles.
    """
    s = A(s)
    if sombra:
        x, y = xy
        for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
            cv2.putText(img, s, (x + dx, y + dy), fuente, esc, (0, 0, 0), grosor, cv2.LINE_AA)
    cv2.putText(img, s, xy, fuente, esc, col, grosor, cv2.LINE_AA)


def banda(lineas, W, esc=0.46, alto_linea=22, margen=14, fondo=FONDO, col=BLANCO, fuente=F):
    """Una banda de texto para APILAR CON `vstack`, nunca para pintar encima de un panel ya hecho.

    POR QUE ESTA REGLA. Pintar una banda sobre un panel compuesto se come lo que hubiera debajo, y
    lo que se come es justamente la ultima linea del rotulo: la que dice por que algo no se usa.
    Aqui la banda se DEVUELVE con su alto exacto y quien la use la apila; y si va a un video, hay
    que sumar su alto al tamano del `VideoWriter`, porque si no coincide el escritor descarta cada
    cuadro por tamano y el MP4 sale de 0 bytes sin un solo error.
    """
    alto = 2 * margen + alto_linea * len(lineas)
    b = np.full((alto, W, 3), fondo, dtype="uint8")
    for i, (s, c, e) in enumerate(_norm(lineas, col, esc)):
        cv2.putText(b, A(s), (margen, margen + alto_linea * (i + 1) - 6), fuente, e, c, 1,
                    cv2.LINE_AA)
    return b


def _norm(lineas, col, esc):
    out = []
    for l in lineas:
        if isinstance(l, str):
            out.append((l, col, esc))
        elif len(l) == 2:
            out.append((l[0], l[1], esc))
        else:
            out.append(tuple(l))
    return out


EN = [
    ("ninguna: nadie ha iniciado un ataque correcto",
     "none: nobody has initiated a correct attack"),
    ("t.83.2.a ataque MAL iniciado: el fondo precede a la extension",
     "t.83.2.a attack BADLY initiated: the lunge precedes the arm extension"),
    ("t.83.2.a extension del brazo, sin fondo todavia",
     "t.83.2.a arm extension, no lunge yet"),
    ("t.83.2.a extension del brazo antes del inicio del fondo",
     "t.83.2.a arm extension before the start of the lunge"),
    ("t.84.3 ataque sobre la preparacion del adversario",
     "t.84.3 attack on the opponent's preparation"),
    ("t.89.2 accion simultanea, no hay prioridad",
     "t.89.2 simultaneous action, no right of way for either"),
    ("t.89.4.a contraataque: no quita la prioridad al atacante",
     "t.89.4.a counter-attack: it does not take right of way from the attacker"),
    ("t.89.5.d el ataque se acorta o se detiene",
     "t.89.5.d the attack falls short or stops"),
    ("t.9.1 sin dato de alcance: no se puede afirmar que haya amenaza",
     "t.9.1 no reach measurement: a threat cannot be asserted"),
    ("t.84.3 ataque sobre la marcha del izquierdo",
     "t.84.3 attack on the left fencer's step-in"),
    ("t.83.2.a extension antes del fondo con el derecho retrocediendo",
     "t.83.2.a extension before the lunge, right fencer retreating"),
    ("t.83.2.a extension antes del fondo",
     "t.83.2.a extension before the lunge"),
    ("t.89.4.a el izquierdo contraataca y no gana prioridad",
     "t.89.4.a the left fencer counter-attacks and gains no right of way"),
    ("t.89.4.a el izquierdo contraataca", "t.89.4.a the left fencer counter-attacks"),
    ("t.89.4.a el derecho contraataca", "t.89.4.a the right fencer counter-attacks"),
    ("fin de la frase de armas tras el tocado",
     "end of the phrase, after the hit"),
    ("el derecho encoge el brazo y deja de amenazar",
     "the right fencer bends his arm back and stops threatening"),
    ("(concedida al expirar el intervalo de gracia)",
     "(granted when the grace interval expired)"),
    ("intervalo de gracia: arranque correcto en espera",
     "grace interval: a correct arm start is on hold"),
    ("t.9.1 no hay amenaza posible:", "t.9.1 no threat is possible:"),
    ("m entre manos y la hoja mide", "m between the hands and the blade is"),
]


def en(s):
    """Traduce al INGLES las cadenas de regla que el codigo y la anotacion escriben en espanol.

    SE TRADUCE, NO SE CAMBIA EL ORIGEN: `prioridad.py` y los CSV no se tocan (los consumen otras
    medidas ya publicadas), asi que el idioma se arregla en la capa que publica. Si aparece una
    cadena que no esta en la tabla, se ve: queda con acento o con una palabra espanola y la puerta
    ASCII o el propio ojo la delatan.
    """
    s = str(s)
    for a, b in EN:
        s = s.replace(a, b)
    resto = [w for w in ("ninguna", "ataque", "accion", "prioridad", "brazo", "fondo", "izquierd",
                         "derech", "contraataque", "extension del", "manos y", "alcance",
                         "cuadro", "tirador", "hoja", "parada")
             if w in s.lower()]
    assert not resto, "cadena sin traducir en una figura publica: %r (sobra %s)" % (s, resto)
    return s


def carga(cid):
    with np.load(PRUEBAS / ("poses_%s.npz" % cid), allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def cuadro_de(video, k):
    """LEE EL CUADRO k LEYENDO DESDE EL PRINCIPIO, no con `CAP_PROP_POS_FRAMES`.

    El indice de `poses_*.npz` es el indice de decodificacion desde 0 (asi lo consume
    `video_prioridad`, que lee `cap.read()` en bucle y usa `KP[k]`). Un `set(POS_FRAMES)` en MP4 con
    B-frames puede caer en el fotograma clave anterior y devolver OTRO cuadro sin avisar, y entonces
    la figura ensena una pose sobre un cuerpo que no es el suyo. Leer en secuencia cuesta unos
    cientos de cuadros y elimina la duda.
    """
    cap = cv2.VideoCapture(str(video))
    im = None
    for j in range(k + 1):
        ok, fr = cap.read()
        if not ok:
            break
        if j == k:
            im = fr
    cap.release()
    return im


def geo(d, s):
    """Vertice, direccion, validez y escala px/m de los dos lados, para pintar el cono."""
    KP, SC, B, H = d["KP"], d["SC"], d["B"], int(d["H"])
    g = {}
    for lado in (0, 1):
        mano = s["mano"][lado]
        vert, u, ok, _lg = C3.eje_brazo_2d(KP, SC, lado, mano, C3.EJE_POR_DEFECTO)
        _pk, pxm = CL.escala_px_m(B, lado, H)
        M = CL.MANOS[mano]
        g[lado] = {"vert": vert, "u": u, "ok": ok, "pxm": pxm, "mano": mano,
                   "codo": KP[:, lado, M["codo"]].astype("float64"),
                   "armado": (M["codo"], M["root"])}
    return g


def pinta_cono(vis, g, lado, k, col):
    """El cono de amenaza con las MISMAS funciones y las MISMAS aperturas que la medida."""
    d = g[lado]
    if not bool(d["ok"][k]):
        return False
    vert, u = d["vert"][k], d["u"][k]
    if not (np.isfinite(vert).all() and np.isfinite(u).all()):
        return False
    L = ALCANCE_M * d["pxm"]
    E21.rellena_peso(vis, vert, u, L, ENVOL_SEMI, col)
    E21.arco(vis, vert, u, L, ENVOL_SEMI, col, 1, trazos=True)
    E21.arco(vis, vert, u, L, NUCLEO_SEMI, col, 1)
    if np.isfinite(d["codo"][k]).all():
        E21.trazos(vis, d["codo"][k], vert, (185, 185, 185), 1)
    cv2.circle(vis, np.int32(vert), 5, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(vis, np.int32(vert), 4, BLANCO, -1, cv2.LINE_AA)
    return True


def caja(d, k, lado):
    """La caja del lado, y SI es la de una pose inventada se devuelve la envolvente `Be`.

    Es la distincion que hace `video_prioridad`: `origen == 0` es pose MEDIDA y solo esa puede
    sostener una decision; `origen > 0` es inventada y su caja vive en `Be`.
    """
    inv = int(d["origen"][k, lado]) > 0
    b = d["Be"][k, lado] if inv else d["B"][k, lado]
    return (None, inv) if not np.isfinite(b).all() else (np.asarray(b, "float64"), inv)


def recuadro(vis, b, W, H, col, grosor):
    x0, y0 = int(b[0] * W), int(b[1] * H)
    x1, y1 = int(b[2] * W), int(b[3] * H)
    cv2.rectangle(vis, (x0, y0), (x1, y1), col, grosor)
    return x0, y0, x1, y1


def ventana(d, k, W, H, margen=0.12, ar=16.0 / 9.0):
    """Recorte que contiene las dos cajas presentes, con margen y la razon de aspecto pedida."""
    xs, ys = [], []
    for lado in (0, 1):
        b, _inv = caja(d, k, lado)
        if b is None:
            continue
        xs += [b[0] * W, b[2] * W]
        ys += [b[1] * H, b[3] * H]
    if not xs:
        return 0, 0, W, H
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    mx, my = margen * (x1 - x0) + 24, margen * (y1 - y0) + 24
    x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    an, al = max(x1 - x0, 160.0), max(y1 - y0, 90.0)
    if an / al < ar:
        an = al * ar
    else:
        al = an / ar
    x0, x1 = cx - an / 2, cx + an / 2
    y0, y1 = cy - al / 2, cy + al / 2
    if x0 < 0:
        x1, x0 = x1 - x0, 0.0
    if y0 < 0:
        y1, y0 = y1 - y0, 0.0
    if x1 > W:
        x0, x1 = max(0.0, x0 - (x1 - W)), float(W)
    if y1 > H:
        y0, y1 = max(0.0, y0 - (y1 - H)), float(H)
    return int(x0), int(y0), int(max(x0 + 16, x1)), int(max(y0 + 9, y1))


def recorta_escala(vis, vent, ancho, alto):
    x0, y0, x1, y1 = vent
    sub = vis[y0:y1, x0:x1]
    if sub.size == 0:
        sub = vis
        x0 = y0 = 0
        x1, y1 = vis.shape[1], vis.shape[0]
    out = cv2.resize(sub, (ancho, alto), interpolation=cv2.INTER_LANCZOS4)
    sx, sy = ancho / float(x1 - x0), alto / float(y1 - y0)

    def tr(x, y):
        return int(round((x - x0) * sx)), int(round((y - y0) * sy))

    return out, tr


def marco(img, col=(70, 70, 74), grosor=2):
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), col, grosor)
    return img


def cabecera(W, titulo, sub, alto=104, col_tit=BLANCO):
    b = np.full((alto, W, 3), (26, 26, 30), dtype="uint8")
    cv2.rectangle(b, (0, alto - 3), (W, alto), (90, 90, 96), -1)
    texto(b, titulo, (22, 44), 0.95, col_tit, 2, FD, sombra=False)
    y = 72
    for s in sub:
        texto(b, s, (24, y), 0.48, (176, 176, 184), 1, F, sombra=False)
        y += 22
    return b


def apilar(*trozos):
    W = max(t.shape[1] for t in trozos)
    out = []
    for t in trozos:
        if t.shape[1] != W:
            t = np.pad(t, ((0, 0), (0, W - t.shape[1]), (0, 0)), constant_values=20)
        out.append(t)
    return np.vstack(out)


def guarda(nombre, img):
    SALIDA.mkdir(parents=True, exist_ok=True)
    p = SALIDA / nombre
    ok = cv2.imwrite(str(p), img, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
    assert ok and p.exists() and p.stat().st_size > 0, "no se escribio %s" % p
    print("  %-52s %5d x %-5d %7.1f KB" % (nombre, img.shape[1], img.shape[0],
                                           p.stat().st_size / 1024.0))
    return p

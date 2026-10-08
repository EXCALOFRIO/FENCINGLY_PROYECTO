"""PASO 1 · Censo de evidencia de la PISTA en la imagen. Mide si el problema tiene solucion ANTES
de escribir la homografia.

POR QUE ESTE PASO EXISTE Y NO SE SALTA. El proyecto ya intento detectar las lineas de la pista en su
etapa historica y lo CERRO con una cifra: `99_historico/FENCINGLY/notas/scratchpad/informe/paper/
main.tex`, seccion `sec:abl-geometry`, dice que sobre 500 cuadros de puesta en guardia de 19 sedes la
deteccion devolvia calibracion en 19 (3,8 %) y que el techo de un humano asistido sobre las MISMAS
tiras rectificadas era 1 de 23 (4,3 %). O sea: no era un problema de algoritmo, era falta de
evidencia. La causa que dio: el cuadro de television abarca una mediana de 8,3 m de pista (p05 5,8,
p95 11,2, n=348) y NINGUNO de los 348 abarca los 14 m.

Ese antecedente NO cierra este paso, porque medía otra cosa: buscaba las MARCAS TRANSVERSALES (la via
`a`). Este censo mide por separado tres evidencias, y una de ellas no estaba en aquel estudio:

  `n_trans`  segmentos rectos transversales al eje de la pista  -> via (a), la ya cerrada
  `n_long`   segmentos rectos LONGITUDINALES, o sea los bordes  -> via (b), sin medir todavia
  `sep_col`  separacion de color entre la pista y el suelo      -> via (b) por region en vez de recta

GEOMETRIA FIE, comprobada en el material del proyecto y no de memoria. Fichero
`scratchpad/fie_tech.txt`, articulos t.18 y t.19:

  t.18.1  la pista mide de 1,50 a 2 metros de ancho
  t.18.2  la pista mide 14 metros de largo
  t.19.2  una linea CENTRAL, discontinua, de lado a lado del ancho
  t.19.4  dos lineas de PUESTA EN GUARDIA a 2 metros de cada lado del centro -> 4,00 m entre ellas
  t.19    dos lineas de LIMITE POSTERIOR a 7 metros del centro
  t.19    los ultimos 2 metros antes del limite posterior deben distinguirse, si se puede por un
          COLOR distinto de pista

UNA CORRECCION AL ENUNCIADO DE LA TAREA: el reglamento NO define una "linea de aviso" a 5 m del
centro. Lo que define es que la ZONA de los ultimos 2 m se distinga, normalmente por color. Asi que
la marca a 5 m del centro existe, pero como FRONTERA DE COLOR, no como linea pintada. Eso importa
porque un detector de rectas no la va a encontrar y un detector de color si.

CRITERIO DECLARADO ANTES DE MIRAR NINGUN NUMERO:

  via (a), lineas transversales + distancias FIE:  viable solo si `n_trans >= 2` en >= 50 % de los
           clips, porque con una sola transversal no hay punto de fuga transversal y sin el no hay
           rectificacion metrica.
  via (b), bordes longitudinales + punto de fuga:  viable solo si `n_long >= 2` en >= 50 % de los
           clips.
  via (b'), la pista como REGION de color:          viable solo si `sep_col >= 2,0` (dos desviaciones
           combinadas entre el color de dentro y el de fuera) en >= 50 % de los clips.

Si ninguna llega al 50 %, se dice que la pista NO se puede detectar en estos videos y se pasa a la
linea base, que es lo honrado.

COMO SE LIMPIA EL SUELO, que es la parte que el usuario pidio en otro hilo. No se usa Sapiens `seg`
porque hay algo mas barato y mas completo: las detecciones de YOLO del cuadro completo ya cacheadas
cubren a TODAS las personas del pabellon (mediana medida en el proyecto: 8 por cuadro), no solo a los
dos tiradores. Se dilatan sus cajas, se tapan, y sobre los cuadros restantes se toma la MEDIANA
temporal. Una persona que se mueve desaparece de la mediana; una linea pintada no. `seg` costaria
245 ms por cuadro y solo cubriria las cajas que se le den.

QUE SE GUARDA: un CSV con las cifras y un PNG por clip con tres paneles (cuadro, suelo limpio,
segmentos detectados) para poder mirar si las cifras dicen la verdad.
"""
from __future__ import annotations

import json
import pathlib
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import extrae_pareja as EP

SCRATCH_VIEJO = pathlib.Path(
    r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop"
    r"\187cfd83-d8d9-4c4d-bed4-f255724f6b21\scratchpad")
SCRATCH = pathlib.Path(
    r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop-FENCINGLY-AI"
    r"\187cfd83-d8d9-4c4d-bed4-f255724f6b21\scratchpad")
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\pista")

N_PLACA = 48
"""Cuadros que entran en la mediana temporal. Con menos de unas decenas la mediana conserva rastro de
las personas; con muchos mas no mejora y cuesta memoria."""
DILATA = 0.06
"""Margen con el que se agranda cada caja de persona antes de taparla, en fraccion del ancho de la
caja. Hace falta porque la caja de YOLO corta el pie y la sombra, y la sombra es lo que mas ensucia
el suelo."""
LARGO_TRANS_CAJA = 0.25
"""Largo minimo de un segmento para contar como transversal, en fraccion del ALTO DE CAJA mediano de
los tiradores del clip.

POR QUE NO EN FRACCION DEL ANCHO DE IMAGEN, Y ES UN ERROR MIO QUE ME PILLE MIRANDO LAS IMAGENES. La
primera version pedia 0,10 del ancho de imagen y dio `n_trans = 0` en los 13 clips, que encajaba
sospechosamente bien con el antecedente historico. Mirando `censo_GentleQuerulousElephantseal.png` se
ve que las marcas transversales ESTAN ahi, blancas y nitidas, y que miden del orden de 120 px de 1280
= 0,094 del ancho: justo por debajo de mi umbral. O sea que el cero lo fabricaba el umbral.

La unidad correcta es el alto de caja del tirador, porque escala con la perspectiva igual que la
pista: si el plano es mas ancho, la caja se encoge Y la marca transversal se encoge en proporcion.
Una transversal cruza de 1,50 a 2 m de pista y un tirador mide del orden de 1,75 m, pero la marca
esta TUMBADA en el suelo y la caja esta DE PIE, asi que el escorzo reduce mucho la marca: 0,25 del
alto de caja es un umbral flojo a proposito, para no volver a fabricar un cero."""
LARGO_LONG = 0.20
"""Largo minimo para contar como longitudinal. Un borde de pista recorre el cuadro entero, asi que
aqui se puede pedir mas."""
ANG_LONG = 20.0
"""Tolerancia en grados respecto al eje de la pista para llamar longitudinal a un segmento."""
ANG_TRANS = 35.0
"""Desviacion minima en grados respecto al eje para llamar transversal a un segmento."""


def indice_videos() -> dict:
    """QUE HACE: el mapa `stem` -> ruta del mp4, cacheado en el scratchpad de la sesion."""
    p = SCRATCH / "indice_videos.json"
    if p.exists():
        return json.loads(p.read_text())
    import os
    idx = {}
    for r, _d, fs in os.walk("D:/FENCINGLY_VIDEOS_RAW"):
        for f in fs:
            if f.lower().endswith(".mp4"):
                idx.setdefault(pathlib.Path(f).stem, os.path.join(r, f))
    p.write_text(json.dumps(idx))
    return idx


def pareja(clip: str) -> dict | None:
    """QUE HACE: los dos tiradores del clip desde el pickle de detecciones ya cacheado.

    No vuelve a pasar YOLO. Si no hay pickle devuelve None y el clip se salta, porque el coste de
    redetectar un clip entero no se justifica para un censo.
    """
    pk = SCRATCH_VIEJO / ("det_%s.pkl" % clip)
    if not pk.exists():
        return None
    with open(pk, "rb") as fh:
        cajas, cuadros, W, H, T = pickle.load(fh)
    r = EP.procesa(cajas, cuadros, W, H, T)
    r["cajas"], r["cuadros"], r["W"], r["H"], r["T"] = cajas, cuadros, W, H, T
    return r


def pies(r: dict) -> np.ndarray:
    """QUE HACE: el punto de contacto con el suelo de cada tirador, en pixeles.

    Es el centro del borde INFERIOR de la caja, que es la linea base gratis declarada en la tarea.
    Devuelve (T, 2, 2) con NaN donde no hay caja.
    """
    B, pres, W, H, T = r["B"], r["pres"], r["W"], r["H"], r["T"]
    P = np.full((T, 2, 2), np.nan, "float64")
    for k in range(T):
        for lado in (0, 1):
            if not pres[k, lado]:
                continue
            b = B[k, lado]
            if not np.isfinite(b).all():
                continue
            P[k, lado] = (0.5 * (b[0] + b[2]) * W, b[3] * H)
    return P


def eje_por_los_pies(P: np.ndarray, mask: np.ndarray) -> float:
    """QUE HACE: la direccion aproximada del eje de la pista, en grados, ajustando una recta a los
    puntos de apoyo de los dos tiradores a lo largo del clip.

    ESTO NO ES LA HOMOGRAFIA. Es solo la direccion que se usa para CLASIFICAR los segmentos
    detectados en longitudinales y transversales, o sea un instrumento de medida del censo. La
    homografia del paso 2 se ajusta con la pista y los tiradores quedan como control, que es el
    requisito duro de la tarea. Se usa aqui porque los tiradores solo pueden andar a lo largo de la
    pista, asi que la nube de sus apoyos es el eje por construccion.
    """
    q = P[mask].reshape(-1, 2)
    q = q[np.isfinite(q).all(axis=1)]
    if len(q) < 8:
        return 0.0
    q = q - q.mean(axis=0)
    _u, _s, vt = np.linalg.svd(q, full_matrices=False)
    return float(np.degrees(np.arctan2(vt[0, 1], vt[0, 0])))


def placa_limpia(video: str, r: dict) -> tuple:
    """QUE HACE: el suelo sin personas, por mediana temporal con las cajas de TODAS las personas
    tapadas.

    Devuelve la placa en BGR, el numero de cuadros que entraron y el mapa de cobertura, que dice
    cuantas muestras validas ha tenido cada pixel. Un pixel con pocas muestras no es suelo medido:
    es suelo adivinado, y hay que poder descontarlo.
    """
    mask, T, W, H = r["mask"], r["T"], r["W"], r["H"]
    cajas, cuadros = r["cajas"], r["cuadros"]
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        idx = np.arange(T)
    sel = idx[np.linspace(0, len(idx) - 1, min(N_PLACA, len(idx))).astype(int)]
    sel_s = set(int(v) for v in sel)
    cap = cv2.VideoCapture(video)
    pila, tapas = [], []
    k = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if k in sel_s:
            t = np.ones((H, W), "uint8")
            for i in np.nonzero(cuadros == k)[0]:
                x0, y0, x1, y1 = [float(v) for v in cajas[i]]
                m = DILATA * max(x1 - x0, 1.0)
                cv2.rectangle(t, (int(x0 - m), int(y0 - m)), (int(x1 + m), int(y1 + m)), 0, -1)
            pila.append(im)
            tapas.append(t)
        k += 1
    cap.release()
    if not pila:
        return None, 0, None
    V = np.stack(pila).astype("float32")
    M = np.stack(tapas).astype(bool)
    V[~M] = np.nan
    with np.errstate(all="ignore"):
        placa = np.nanmedian(V, axis=0)
    cobertura = M.sum(axis=0)
    placa = np.where(np.isfinite(placa), placa, 0.0).astype("uint8")
    return placa, len(pila), cobertura


def banda_suelo(P: np.ndarray, mask: np.ndarray, H: int) -> tuple:
    """QUE HACE: las dos filas de imagen entre las que se busca la pista.

    La pista es una franja alrededor del corredor por el que andan los pies. Se toma el rango de
    filas de apoyo del clip y se ensancha, porque el borde de lejos esta por encima del apoyo y el de
    cerca por debajo. El ensanchamiento va en fraccion del ALTO DE CAJA tipico y no en fraccion de la
    imagen, para que valga igual en un plano corto y en uno ancho.
    """
    v = P[mask][:, :, 1]
    v = v[np.isfinite(v)]
    if len(v) < 8:
        return int(0.45 * H), H - 1
    lo, hi = float(np.percentile(v, 2)), float(np.percentile(v, 98))
    ancho = max(hi - lo, 0.02 * H)
    return int(max(0, lo - 1.6 * ancho)), int(min(H - 1, hi + 1.6 * ancho))


def segmentos(placa, cobertura, v0, v1, eje_deg, alto_caja_px) -> dict:
    """QUE HACE: los segmentos rectos del suelo limpio, clasificados en longitudinales y
    transversales respecto al eje.

    Se usa `createLineSegmentDetector`, que es lo que ya trae el cv2 del proyecto; `cv2.ximgproc` no
    esta instalado en ninguno de los dos interpretes, comprobado con `import`.

    Ademas de contar segmentos se cuentan LINEAS: LSD parte una misma marca en varios trozos, asi
    que contar segmentos sobreestima la evidencia. Dos segmentos transversales son la misma linea si
    su recta pasa a menos de un umbral el uno del centro del otro. La cifra que decide la via (a) es
    el numero de LINEAS, no de segmentos.
    """
    g = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    val = np.zeros((H, W), "uint8")
    val[v0:v1 + 1, :] = 1
    if cobertura is not None:
        val[cobertura < max(3, int(0.15 * cobertura.max()))] = 0
    g = cv2.GaussianBlur(g, (3, 3), 0)
    lsd = cv2.createLineSegmentDetector()
    li = lsd.detect(g)[0]
    out = {"long": [], "trans": [], "otros": [], "largos_trans": []}
    if li is None:
        out["n_lineas_trans"] = 0
        out["n_lineas_long"] = 0
        return out
    lmin_t = LARGO_TRANS_CAJA * max(alto_caja_px, 8.0)
    for s in li.reshape(-1, 4):
        x0, y0, x1, y1 = [float(v) for v in s]
        cx, cy = int(0.5 * (x0 + x1)), int(0.5 * (y0 + y1))
        if not (0 <= cx < W and 0 <= cy < H) or val[cy, cx] == 0:
            continue
        Lpx = float(np.hypot(x1 - x0, y1 - y0))
        L = Lpx / W
        a = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
        d = abs((a - eje_deg + 90.0) % 180.0 - 90.0)
        if d <= ANG_LONG and L >= LARGO_LONG:
            out["long"].append((x0, y0, x1, y1, L, d))
        elif d >= ANG_TRANS:
            out["largos_trans"].append(Lpx / max(alto_caja_px, 8.0))
            if Lpx >= lmin_t:
                out["trans"].append((x0, y0, x1, y1, L, d))
        else:
            out["otros"].append((x0, y0, x1, y1, L, d))
    out["n_lineas_trans"] = _agrupa(out["trans"], 0.35 * max(alto_caja_px, 8.0))
    out["n_lineas_long"] = _agrupa(out["long"], 0.20 * max(alto_caja_px, 8.0))
    return out


def _agrupa(segs, tol_px: float) -> int:
    """QUE HACE: cuenta cuantas RECTAS distintas hay entre unos segmentos, uniendo los colineales.

    Existe porque LSD devuelve una marca pintada partida en dos o tres trozos y contar trozos
    sobreestima la evidencia disponible para la homografia. Dos segmentos son la misma recta si cada
    centro queda a menos de `tol_px` de la recta del otro.
    """
    if not segs:
        return 0
    rectas = []
    for x0, y0, x1, y1, _L, _d in segs:
        p = np.array([x0, y0]);  q = np.array([x1, y1])
        c = 0.5 * (p + q)
        v = q - p
        n = np.array([-v[1], v[0]])
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn
        rectas.append((c, n))
    grupos = []
    for c, n in rectas:
        puesto = False
        for g in grupos:
            c2, n2 = g[0]
            if abs(float(n2 @ (c - c2))) <= tol_px and abs(float(n @ (c2 - c))) <= tol_px:
                g.append((c, n))
                puesto = True
                break
        if not puesto:
            grupos.append([(c, n)])
    return len(grupos)


def separacion_color(placa, cobertura, P, mask) -> float:
    """QUE HACE: cuanto se distingue la pista del suelo de alrededor por color, en desviaciones.

    COMO SE DEFINE "DENTRO" SIN CONOCER LA PISTA, que es el problema del huevo y la gallina: se toma
    la banda estrecha por la que pasan los apoyos de los tiradores, que esta dentro de la pista por
    construccion, contra dos bandas del mismo grosor a un lado y al otro, fuera del corredor. Si la
    pista se distingue, esas tres bandas tienen colores distintos.

    LAS BANDAS VAN A LO LARGO DEL EJE, NO A LO LARGO DE LAS FILAS DE IMAGEN, y esto tambien es un
    error mio que se vio en las imagenes. La primera version usaba bandas de filas y dio 0,44 en
    `UglyNiceAplomadofalcon`, un clip donde la pista es de madera clara sobre un suelo AZUL y se
    distingue de sobra a ojo. La causa: en ese clip el eje va a -13,5 grados, asi que una banda de
    filas mezcla pista y suelo en la misma muestra y la separacion se autodestruye. Ahora la banda se
    define por la distancia PERPENDICULAR al eje ajustado.

    La cifra es |mediana_dentro - mediana_fuera| dividido por la desviacion combinada, en el canal
    que mas separe de los tres de LAB. Es una d de Cohen, asi que 2,0 quiere decir dos desviaciones,
    que es una separacion que se ve a ojo.
    """
    q = P[mask].reshape(-1, 2)
    q = q[np.isfinite(q).all(axis=1)]
    if len(q) < 8:
        return float("nan")
    c0 = q.mean(axis=0)
    _u, _s, vt = np.linalg.svd(q - c0, full_matrices=False)
    n = np.array([-vt[0, 1], vt[0, 0]], "float64")
    t = (q - c0) @ n
    lo, hi = float(np.percentile(t, 10)), float(np.percentile(t, 90))
    gr = max(hi - lo, 6.0)
    lab = cv2.cvtColor(placa, cv2.COLOR_BGR2LAB).astype("float32")
    H, W = lab.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    D = (xx - c0[0]) * n[0] + (yy - c0[1]) * n[1]
    ok = np.ones((H, W), bool)
    if cobertura is not None:
        ok = cobertura >= max(3, int(0.15 * cobertura.max()))

    def banda(a, b):
        m = (D >= a) & (D < b) & ok
        return lab[m] if m.sum() > 200 else None

    dentro = banda(lo, hi)
    fuera_a = banda(lo - 2.2 * gr, lo - 0.6 * gr)
    fuera_b = banda(hi + 0.6 * gr, hi + 2.2 * gr)
    if dentro is None:
        return float("nan")
    mejor = 0.0
    for fuera in (fuera_a, fuera_b):
        if fuera is None:
            continue
        for c in range(3):
            a, b = dentro[:, c], fuera[:, c]
            s = np.sqrt(0.5 * (a.std() ** 2 + b.std() ** 2)) + 1e-6
            mejor = max(mejor, abs(float(np.median(a) - np.median(b))) / s)
    return mejor


def pinta(clip, im, placa, seg, v0, v1, P, mask, eje):
    """QUE HACE: el panel de tres imagenes para poder MIRAR si las cifras dicen la verdad."""
    H, W = placa.shape[:2]
    a = im.copy()
    for k in np.nonzero(mask)[0][::4]:
        for lado, col in ((0, (0, 220, 0)), (1, (0, 140, 255))):
            p = P[k, lado]
            if np.isfinite(p).all():
                cv2.circle(a, (int(p[0]), int(p[1])), 1, col, -1)
    cv2.line(a, (0, v0), (W, v0), (255, 0, 255), 1)
    cv2.line(a, (0, v1), (W, v1), (255, 0, 255), 1)
    b = placa.copy()
    c = placa.copy()
    for x0, y0, x1, y1, _L, _d in seg["long"]:
        cv2.line(c, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
    for x0, y0, x1, y1, _L, _d in seg["trans"]:
        cv2.line(c, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 2)
    cv2.putText(a, "%s  eje %.1f deg" % (clip[:26], eje), (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(b, "suelo limpio", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(c, "long %d (rojo)  trans %d (amarillo)" % (len(seg["long"]), len(seg["trans"])),
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    tira = np.vstack([a, b, c])
    esc = 900.0 / tira.shape[1] if tira.shape[1] > 900 else 1.0
    if esc != 1.0:
        tira = cv2.resize(tira, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
    SALIDA.mkdir(parents=True, exist_ok=True)
    """LOS `censo_*.png` SON DIAGNOSTICO INTERNO y van a la subcarpeta, no al entregable: llevan la
    region de pista en verde y las mascaras de rotulo encima del marcador del clip. Sirven para saber
    por que el barrido encuentra o no encuentra los bordes, no para ensenarlos."""
    (SALIDA / "diagnostico").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SALIDA / "diagnostico" / ("censo_%s.png" % clip)), tira)


def main(clips):
    idx = indice_videos()
    filas = []
    for clip in clips:
        if clip not in idx:
            print("%-36s sin video" % clip)
            continue
        r = pareja(clip)
        if r is None or not r["mask"].any():
            print("%-36s sin pickle o sin tramo utilizable" % clip)
            continue
        P = pies(r)
        eje = eje_por_los_pies(P, r["mask"])
        placa, n, cob = placa_limpia(idx[clip], r)
        if placa is None:
            print("%-36s no se pudo leer" % clip)
            continue
        v0, v1 = banda_suelo(P, r["mask"], r["H"])
        alt = (r["B"][:, :, 3] - r["B"][:, :, 1])[r["pres"]]
        alt_px = float(np.nanmedian(alt)) * r["H"] if len(alt) else 0.0
        seg = segmentos(placa, cob, v0, v1, eje, alt_px)
        sc = separacion_color(placa, cob, P, r["mask"])
        cap = cv2.VideoCapture(idx[clip])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(np.nonzero(r["mask"])[0][len(np.nonzero(r["mask"])[0]) // 2]))
        _ok, im = cap.read()
        cap.release()
        if im is None:
            im = placa.copy()
        pinta(clip, im, placa, seg, v0, v1, P, r["mask"], eje)
        lt = seg.get("largos_trans") or [0.0]
        fila = {"clip": clip, "W": r["W"], "H": r["H"], "T": r["T"],
                "utiles": int(r["mask"].sum()), "placa": n, "eje_deg": round(eje, 2),
                "alto_caja_px": round(alt_px, 1),
                "n_long": len(seg["long"]), "lin_long": seg["n_lineas_long"],
                "n_trans": len(seg["trans"]), "lin_trans": seg["n_lineas_trans"],
                "p90_largo_trans_caja": round(float(np.percentile(lt, 90)), 3),
                "n_otros": len(seg["otros"]), "sep_col": round(float(sc), 3),
                "v0": v0, "v1": v1}
        filas.append(fila)
        print("%-36s %dx%d util%4d eje %6.1f  long %3d/%2d  trans %3d/%2d  p90L %.2f  sep_col %5.2f"
              % (clip, r["W"], r["H"], fila["utiles"], eje, fila["n_long"], fila["lin_long"],
                 fila["n_trans"], fila["lin_trans"], fila["p90_largo_trans_caja"],
                 fila["sep_col"]))
    if not filas:
        return
    import csv
    SALIDA.mkdir(parents=True, exist_ok=True)
    with open(SALIDA / "censo_pista.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    nl = sum(1 for f in filas if f["lin_long"] >= 2)
    nt = sum(1 for f in filas if f["lin_trans"] >= 2)
    nc = sum(1 for f in filas if f["sep_col"] >= 2.0)
    print("\nCLIPS %d" % len(filas))
    print("  via (a) LINEAS transversales >= 2 : %d (%.0f %%)" % (nt, 100.0 * nt / len(filas)))
    print("  via (b) LINEAS longitudinales >= 2: %d (%.0f %%)" % (nl, 100.0 * nl / len(filas)))
    print("  via (b') color >= 2,0 sd          : %d (%.0f %%)" % (nc, 100.0 * nc / len(filas)))


if __name__ == "__main__":
    lista = sys.argv[1:]
    if not lista:
        lista = json.loads((SCRATCH / "clips_censo.json").read_text())
    main(lista)

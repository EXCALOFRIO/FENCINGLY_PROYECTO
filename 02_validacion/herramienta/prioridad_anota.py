"""Monta las tiras de fotogramas con las que se ANOTA A MANO quien tiene la prioridad.

POR QUE EXISTE: no hay verdad de referencia etiquetada de prioridad en este proyecto, asi que hay
que construirla. Y hay que construirla ANTES de escribir la decision del detector, porque si se
anota despues de ver lo que el detector dice, la anotacion deja de ser independiente y la cifra de
acierto no vale nada.

QUE SE VE Y QUE NO SE VE, y esto es lo importante del diseno:
  SE VE   el recorte del video con los dos tiradores, ampliado, con el numero de cuadro
  SE VE   una caja fina VERDE en el de la izquierda (lado 0) y NARANJA en el de la derecha (lado 1),
          que es identidad, no prioridad: hace falta para poder escribir «izquierda» en la anotacion
  NO SE VE  ninguna pose, ningun angulo, ninguna variable y por supuesto ninguna decision del
          detector. Si la tira llevara la salida del algoritmo, anotar seria copiar.

COMO SE ELIGEN LAS VENTANAS: por ACTIVIDAD, no por prioridad. Se buscan los cuadros donde alguno de
los dos mueve el brazo o los pies, que es donde hay una accion que arbitrar. Elegir la ventana con
una senal de actividad no filtra a favor de ninguno de los dos lados, asi que no sesga la anotacion.

DOS FORMAS DE ELEGIR LA VENTANA, y las dos hacen falta. La razon esta en el docstring de cada una:
  `act`     actividad de brazo y pies partida por la separacion
  `cierre`  velocidad a la que los dos se acercan, que es la que no se confunde con el post-tocado

USO:
  python prioridad_anota.py act CLIP [CLIP...]       laminas por actividad
  python prioridad_anota.py cierre CLIP [CLIP...]    laminas por acercamiento
  python prioridad_anota.py lista                    lista los clips de florete con poses
"""
from __future__ import annotations

import os
import pathlib
import sys

import cv2
import numpy as np

CODIGO = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
sys.path.insert(0, str(CODIGO))
import p1_variables as P1
import prioridad as PR

PRUEBAS = P1.PRUEBAS
SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\prioridad\tiras")
PASO = 3
"""Cuadros de salto entre vinetas. Con 8 vinetas cubre 24 cuadros, que a 30 fps son 0,80 s. El
tiempo de esgrima medido en este proyecto es de **5 cuadros** (mediana del tramo de subida de
`ext_hr`), asi que la ventana contiene casi cinco tiempos de esgrima: el arranque, la accion y su respuesta.

Y EL PASO FIJA LA RESOLUCION DE LA ANOTACION: el cuadro de cambio se anota con una tolerancia de
**+-3 cuadros**, no mejor, porque los cuadros de en medio no se han visto. Eso hay que decirlo al dar
el acierto, y es la razon de que el acierto del instante se mida con tolerancia y no exacto."""
N_VIN = 8
COLS = 2
ALTO_VIN = 460
"""EL PRESUPUESTO DE PIXELES ES LO QUE MANDA AQUI, y me costo dos intentos.

Quien mira el PNG tiene un limite de unos 2.000 px de lado, o sea del orden de 4 millones de pixeles
para toda la lamina. Ese presupuesto se reparte entre las vinetas, asi que **mas vinetas es menos
resolucion por vineta**, y no es un detalle esteticо:

  | version | vinetas | px por vineta | se podia anotar |
  |---|---:|---:|---|
  | 12 vinetas en 3 columnas | 12 | ~660 | **no**: brazo extendido y encogido no se distinguen |
  | 10 vinetas en 2 columnas | 10 | ~460 | **no**: la lamina sale apaisada al reves y se reduce |
  | **8 vinetas en 2 columnas** | 8 | **~830** | si, el tirador mide ~250 px de alto |

Con 8 vinetas y paso 3 la ventana cubre 24 cuadros, que a 30 fps son 0,8 s y casi **cinco tiempos de
esgrima** de los 5 cuadros medidos. Cabe la carrera de aproximacion, la accion y su respuesta, y eso
importa: la prioridad se decide en el ARRANQUE, asi que la lamina tiene que contener el arranque.

Y CUANDO ESO NO BASTA hay una segunda herramienta, `zoom_*`: un tirador por columna, a doble
resolucion, sin la pista de en medio. Es la que resolvio la unica discrepancia real entre mi
anotacion y el detector, y la que demostro que **el equivocado era yo**."""


def ventanas(s, mask, T, cuantas=3):
    """QUE HACE: elige los tramos con mas actividad de brazo o de pies Y con los dos tiradores CERCA.

    La puntuacion es la suma de |dext_hr| y |dpies| de LOS DOS lados, partida por la separacion
    mediana de la ventana. Al sumar los dos lados la eleccion no puede favorecer a uno, asi que no
    sesga la anotacion.

    POR QUE DIVIDIR POR LA SEPARACION, y lo aprendi mirando la primera tanda: la actividad sola
    elige **juego de piernas a cinco metros**, donde no hay ninguna accion que arbitrar y la unica
    anotacion honrada es «ninguna». La distancia de ataque es una condicion previa de que exista
    prioridad: sin ella, ninguno de los dos amenaza nada. Dividir por `sep` no mira quien ataca, solo
    si hay a quien atacar.
    """
    act = np.zeros(T)
    for lado in (0, 1):
        for k in ("dext_hr", "dpies"):
            a = np.abs(np.nan_to_num(s[k][:, lado], nan=0.0))
            act += np.where(s["val"][:, lado], a, 0.0)
    sep = np.nan_to_num(s["sep"], nan=np.nanmax(s["sep"]) if np.isfinite(s["sep"]).any() else 1.0)
    pun = act / np.maximum(sep, 0.5)
    largo = PASO * N_VIN
    atras = PASO * (N_VIN - 2)
    """EL PICO VA AL FINAL DE LA VENTANA, NO EN MEDIO, y esto lo aprendi mirando la segunda tanda.

    Centrando la ventana en el maximo de actividad, la lamina de `AliveIgnorantBarebirdbat` empezaba
    en c288 con los dos tiradores YA en pleno fondo y cuerpo a cuerpo: el tocado estaba dado en c294
    con las dos luces encendidas. Con eso **no se puede anotar la prioridad**, porque la prioridad se
    decide en el ARRANQUE de la accion y el arranque quedaba fuera de la lamina.

    Asi que la ventana se coloca hacia atras: el pico cae en la penultima vineta y las seis primeras
    son la carrera de aproximacion, que es donde estan los dos instantes que decide t.83.2.a."""
    mejor = []
    usado = np.zeros(T, dtype=bool)
    for _ in range(cuantas):
        p = pun.copy()
        p[usado] = -1.0
        for k in range(T):
            a = k - atras
            if a < 0 or a + largo > T or not mask[a:a + largo].all():
                p[k] = -1.0
        if p.max() <= 0:
            break
        k = int(np.argmax(p))
        mejor.append(k - atras)
        usado[max(0, k - largo):min(T, k + largo)] = True
    return sorted(mejor)


def ventanas_cierre(s, mask, T, cuantas=2):
    """QUE HACE: elige las ventanas donde los dos tiradores se ACERCAN mas rapido.

    POR QUE HACE FALTA UNA SEGUNDA FORMA DE ELEGIR, y lo aprendi mirando la tercera tanda: la
    puntuacion `actividad / separacion` de `ventanas` premia estar cerca, y estar cerca **tambien
    pasa despues del tocado**, cuando los dos se cruzan y se van andando. Dos laminas de las diez
    salieron asi: `FirsthandSoreGnat` c351-c372 son los dos de pie con la hoja mirando al suelo, y
    `BlissfulTiredAiredale` c372-c393 es uno de ellos ya fuera de la pista.

    La velocidad de acercamiento no tiene ese problema: solo es grande **mientras** alguien va a por
    el otro, y se apaga en cuanto se cruzan. Es una senal de que hay una accion en marcha, y sigue
    sin mirar de quien es.
    """
    sep = s["sep"]
    d = -PR.suaviza(sep, deriv=1)
    d = np.nan_to_num(d, nan=-1.0)
    largo = PASO * N_VIN
    atras = PASO * (N_VIN - 3)
    mejor, usado = [], np.zeros(T, dtype=bool)
    for _ in range(cuantas):
        p = d.copy()
        p[usado] = -1.0
        for k in range(T):
            a = k - atras
            if a < 0 or a + largo > T or not mask[a:a + largo].all():
                p[k] = -1.0
        if p.max() <= 0:
            break
        k = int(np.argmax(p))
        mejor.append(k - atras)
        usado[max(0, k - largo):min(T, k + largo)] = True
    return sorted(mejor)


def tira(cid, modo="act"):
    with np.load(PRUEBAS / ("poses_%s.npz" % cid), allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    s = PR.senales(d)
    B, pres, mask = d["B"], d["pres"], d["mask"]
    W, H, T = int(d["W"]), int(d["H"]), int(d["T"])
    video = str(d["video"])
    if not os.path.exists(video):
        print("SIN VIDEO %s" % cid)
        return []
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = {}
    ini = ventanas_cierre(s, mask, T) if modo == "cierre" else ventanas(s, mask, T)
    quiere = set()
    for a in ini:
        quiere.update(range(a, a + PASO * N_VIN))
    k = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if k in quiere:
            frames[k] = im
        k += 1
    cap.release()
    SALIDA.mkdir(parents=True, exist_ok=True)
    hechas = []
    for a in ini:
        ks = [a + i * PASO for i in range(N_VIN)]
        x0, y0, x1, y1 = 1e9, 1e9, -1e9, -1e9
        for k in ks:
            for lado in (0, 1):
                if not pres[k, lado]:
                    continue
                b = B[k, lado]
                if not np.isfinite(b).all():
                    continue
                x0, y0 = min(x0, b[0] * W), min(y0, b[1] * H)
                x1, y1 = max(x1, b[2] * W), max(y1, b[3] * H)
        if x1 <= x0:
            continue
        mx, my = 0.10 * (x1 - x0) + 12, 0.14 * (y1 - y0) + 12
        x0, y0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
        x1, y1 = min(W, int(x1 + mx)), min(H, int(y1 + my))
        vin = []
        for k in ks:
            im = frames.get(k)
            if im is None:
                continue
            v = im[y0:y1, x0:x1].copy()
            for lado in (0, 1):
                if not pres[k, lado]:
                    continue
                b = B[k, lado]
                if not np.isfinite(b).all():
                    continue
                col = (60, 240, 60) if lado == 0 else (60, 160, 255)
                cv2.rectangle(v, (int(b[0] * W) - x0, int(b[1] * H) - y0),
                              (int(b[2] * W) - x0, int(b[3] * H) - y0), col, 1)
            esc = ALTO_VIN / max(v.shape[0], 1)
            v = cv2.resize(v, (int(v.shape[1] * esc), ALTO_VIN), interpolation=cv2.INTER_CUBIC)
            cv2.rectangle(v, (0, 0), (78, 22), (16, 16, 20), -1)
            cv2.putText(v, "c%d" % k, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            vin.append(v)
        if not vin:
            continue
        anc = max(v.shape[1] for v in vin)
        filas = []
        for i in range(0, len(vin), COLS):
            grupo = vin[i:i + COLS]
            grupo = [cv2.copyMakeBorder(v, 2, 2, 2, 2 + anc - v.shape[1], cv2.BORDER_CONSTANT,
                                        value=(30, 30, 30)) for v in grupo]
            while len(grupo) < COLS:
                grupo.append(np.full_like(grupo[0], 30))
            filas.append(np.hstack(grupo))
        img = np.vstack(filas)
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (12, 12, 18), -1)
        cv2.putText(img, "%s  c%d-c%d  paso %d  %.0f fps   VERDE=izquierda(0)  NARANJA=derecha(1)"
                    % (cid, ks[0], ks[-1], PASO, fps), (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)
        p = SALIDA / ("%s_%s_c%04d.png" % (cid, modo, a))
        cv2.imwrite(str(p), img)
        hechas.append((str(p), a, ks[-1], fps))
        print("%s  c%d-c%d  %.2f fps -> %s" % (cid, ks[0], ks[-1], fps, p.name))
    return hechas


if __name__ == "__main__":
    if sys.argv[1] == "lista":
        for c in P1.clips_con_poses():
            print(c)
    else:
        for cid in sys.argv[2:]:
            tira(cid, modo=sys.argv[1])

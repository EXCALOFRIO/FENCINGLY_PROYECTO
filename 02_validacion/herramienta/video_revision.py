"""Video de revision: las DOS poses de Sapiens 2 pintadas sobre el clip, para verificar a ojo.

PARA QUE SIRVE, con las palabras del usuario: "necesito verlo para ver como se ven las poses en un
video y ver que no detecta intrusos y lo verifico yo". Una cifra de presencia del 99 % no dice si la
pareja elegida es la correcta; esto si.

Y LO QUE PIDIO DESPUES, tambien literal: "hay frames que no detecta a uno de los dos y por eso no
saca el cono; los frames que no detecte hay que ver por que, y si no se crea igual ni con una doble
pasada, pues inventarlo en base a la ultima pose que conociamos y a la siguiente que conozco, que
haga como las poses intermedias, porque si no hay momentos que no vemos a los dos y perdemos el
foco". Eso es lo que hace `inventa_poses`, y `valida_inventadas` le pone un numero.

QUE SE VE, y cada cosa esta ahi para poder contradecir al algoritmo:
  caja VERDE     el tirador de la izquierda        caja NARANJA  el de la derecha
  caja VIOLETA   pose INVENTADA, con la etiqueta puesta: interpolada o sostenida
  trazo fino y `redetectada` cuando el cuadro se recupero en un recorte
  banda roja     `FUERA: primer plano o repeticion`, el cuadro no se usa
  rotulo con el motivo de la eleccion de pareja y sus cifras

DOS FASES, porque los interpretes NO coexisten y esto ya ha costado tiempo antes:
  `python video_revision.py fase1 CLIP [CLIP...]`   con el interprete de ultralytics
  `python video_revision.py fase2 CLIP [CLIP...]`   con el interprete de cuda y transformers

EL MAPA DE INDICES ES EL MEDIDO en `cono_lib`, no el que traia el plan. Los indices del plan
(`hand_root` 91 y 112) caen sobre la careta, y con ellos el pulgar medía 1,4-2,8 px. Aqui:
  5-6 hombros  7-8 codos  9-10 caderas  11-12 rodillas  13-14 tobillos  15-20 pies
  21-41 mano A (root 21)  42-62 mano B (root 42)   63 en adelante cara, que NO se pinta

LA CAJA VA EN `xywh`, no en `xyxy`. Es la trampa mas cara del proyecto: con `xyxy` la confianza de
la pose cae de 0,883 a 0,104 y el modelo parece roto.
"""
from __future__ import annotations

import os
import pathlib
import pickle
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "01_extraccion" / "codigo"))
import rutas as RU  # noqa: E402  -- antes que cv2 y que torch, para fijar HF_HOME

CACHE_HF = RU.HF
os.environ.setdefault("HF_HOME", str(CACHE_HF))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

CODIGO = RU.CODIGO
PRUEBAS = RU.TRABAJO
SALIDA = RU.VAL / "videos_revision"
SCR = RU.SCRATCH
VIDEOS = str(RU.VIDEOS)
RES_POSE = {"height": 384, "width": 288}
"""384x288 da 2,26 px de error de torso contra la referencia de 1024 y cuesta 7,8x menos. Medido en
`resolucion_vs_exactitud.py`, juzgando la POSICION de los puntos y no su confianza: un criterio de
confianza a 0,3 aprobaba el 100 % de las resoluciones, o sea no medía nada."""
CONF_PINTA = 0.3
N_PTS = 308
COL_LADO = {0: (60, 240, 60), 1: (60, 160, 255)}
VIOLETA = (200, 80, 220)
RESCATADA = (70, 230, 255)
"""LAS COLORES DE LA PROCEDENCIA DE UNA POSE, EN UN SOLO SITIO Y POR UNA RAZON.

    `COL_LADO`   pose MEDIDA: verde el lado 0, ambar el lado 1
    `VIOLETA`    pose INVENTADA (interpolada), la que el usuario llama «las rosas»
    `RESCATADA`  pose MEDIDA EN UNA SEGUNDA PASADA sobre la caja estimada (`origen = 3`), que no es
                 ni medida en la primera pasada ni inventada, y por eso no puede llevar ninguna de
                 las dos colores de arriba

El violeta estaba escrito **en linea en tres ficheros** (`galeria_inventadas`, `video_prioridad`,
aqui) mas un cuarto que lo tenia con nombre pero sin que nadie lo importara
(`prioridad_publico/comun.py`). Cuatro copias de la color que significa «esto no es dato»: cambiar
una y no las otras deja dos figuras del mismo hecho con distinto codigo de color, y este proyecto ya
tiene documentado lo que cuesta una figura que no coincide con otra."""

HUESOS = [(5, 6), (9, 10), (5, 9), (6, 10),
          (6, 8), (8, 21), (5, 7), (7, 42),
          (9, 11), (11, 13), (10, 12), (12, 14)]
"""Torso, brazos y piernas con el emparejamiento por paridad del `flip_pairs` del modelo: los pares
son (5,6) (7,8) (9,10) (11,12) (13,14), y el lado impar va con el impar."""
PULGARES = [(21, 22), (22, 23), (23, 24), (24, 25), (42, 43), (43, 44), (44, 45), (45, 46)]
MANOS = list(range(21, 63))
CARA_DESDE = 63


def _savez_atomico(p, **arrays):
    """`np.savez_compressed` a `<p>.part` y `os.replace`. (2026-10-13) A un `.part` y luego `os.replace`: un corte a mitad no deja un fichero roto que cuente como hecho. Mismos arrays, misma
    compresion."""
    tmp = p.with_name(p.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, p)


def mapa_videos():
    return RU.indice_videos()


def fase1(clips, videos=None):
    """QUE HACE: detecta, elige la pareja y redetecta, y guarda las cajas en un NPZ por clip.

    `videos` es un dict clip -> ruta para el caso de Colab, donde el mp4 lo acaba de bajar
    `lotes.py` y no esta en ningun arbol que recorrer. Sin el, se busca con `rutas.video()`.
    """
    from ultralytics import YOLO
    import extrae_pareja as EP
    import torch
    """Sin `_hilos_torch` aqui (14/10): en la etapa 1 torch apenas hace cuentas en CPU (el recorte es de
    cv2 y el NMS va en la GPU), y en la prueba C la etapa 1 con 2 hilos fue 201 s contra 139 de B. O9
    es para el preproceso de Sapiens, en `fase2`."""
    yolo = YOLO(RU.pesos_yolo())
    yolo.to("cuda")
    yolo_redet = yolo
    if DOS_YOLO:
        yolo_redet = YOLO(RU.pesos_yolo())
        yolo_redet.to("cuda")
    videos = videos or {}
    PRUEBAS.mkdir(parents=True, exist_ok=True)
    for cid in clips:
        try:
            _fase1_uno(cid, yolo, EP, videos, yolo_redet)
        except Exception as ex:
            """UN CLIP QUE CASCA NO SE LLEVA EL LOTE (2026-10-09). Antes la excepcion salia
            del bucle y los clips que quedaban del lote no se procesaban NUNCA: en cada vuelta
            volvia a cascar el mismo. Medido en la noche de anotados: 3 clips sin una sola
            deteccion (`planos.marca_primer_plano` con un array 1-D) dejaron **9 clips buenos**
            sin etapa 1, y el trabajador se paro creyendo que ya no habia nada que hacer."""
            print("%-40s ETAPA 1 FALLA %s: %s" % (cid[:40], type(ex).__name__, str(ex)[:120]),
                  flush=True)


DOS_YOLO = os.environ.get("FENCINGLY_YOLO_DOS", "0") == "1"
"""O10 (2026-10-13): un SEGUNDO YOLO solo para la redeteccion en recorte. La deteccion va en fp16
(`quantize=16`) y la redeteccion no lo pasa (fp32); con UN objeto, ultralytics reconstruye el predictor
(copia, fusiona y calienta el modelo) en cada cambio. Medido: deteccion + redeteccion alternadas 476 ms
por par, con dos objetos 172 ms, y la redeteccion identica. Entra solo con la prueba de identidad."""
TUBERIA_YOLO = os.environ.get("FENCINGLY_YOLO_TUBERIA", "0") == "1"
"""O3 de `CORRIDA_FINAL_2026-10-13.md` §10 (2026-10-13): la lectura de cuadros de la etapa 1 en un hilo
aparte (`LectorHilo`), para que la decodificacion no espere a YOLO ni YOLO a ella. Mismos cuadros, en
el mismo orden: `det_` y `revision_` tienen que salir IDENTICOS. Apagada por defecto hasta medirlo."""


class LectorHilo:
    """Un `cv2.VideoCapture` que lee por delante en un hilo (cola de 32 cuadros). `read` y `get` como
    el original; `release` para el hilo."""

    def __init__(self, ruta):
        import queue
        import threading
        self.cap = cv2.VideoCapture(ruta)
        self.props = {p: self.cap.get(p) for p in (cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH,
                                                     cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_COUNT)}
        self.cola = queue.Queue(maxsize=32)
        self.parar = threading.Event()
        self.fin = False
        self.th = threading.Thread(target=self._lee, daemon=True)
        self.th.start()

    def _lee(self):
        while not self.parar.is_set():
            ok, im = self.cap.read()
            self.cola.put((ok, im))
            if not ok:
                return

    def get(self, p):
        return self.props[p] if p in self.props else self.cap.get(p)

    def read(self):
        if self.fin:
            return False, None
        ok, im = self.cola.get()
        if not ok:
            self.fin = True
        return ok, im

    def release(self):
        self.parar.set()
        while self.th.is_alive():
            try:
                self.cola.get_nowait()
            except Exception:
                pass
            self.th.join(timeout=0.05)
        self.cap.release()


def _fase1_uno(cid, yolo, EP, videos, yolo_redet=None):
    ruta = videos.get(cid) or RU.video(cid)
    if not ruta:
        print("%-40s SIN VIDEO, no se procesa" % cid[:40])
        return
    ca = PRUEBAS / ("det_%s.pkl" % cid)
    cap = LectorHilo(ruta) if TUBERIA_YOLO else cv2.VideoCapture(ruta)
    fps_clip = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    """LA CADENCIA SE GUARDA EN EL NPZ, y no se vuelve a deducir mas tarde.

    Antes el unico sitio donde quedaba escrita era `posicion_clip_<clip>.csv`, que lo escribe
    la etapa 7. Quien leyera el NPZ sin haber corrido la etapa 7 se quedaba con un 25 por
    defecto, y **una cadencia equivocada escala TODAS las velocidades**: con un clip de 30 fps
    leido como 25, cada velocidad sale multiplicada por 1,2. Lo cazo el control del espejo del
    etiquetador, donde el clip espejo no tenia CSV y sus velocidades salian exactamente 1,2
    veces las del original. En Colab el mp4 se borra tras procesar, asi que si la cadencia no
    viaja en el NPZ no hay de donde recuperarla."""
    if ca.exists():
        with open(ca, "rb") as fh:
            cajas, cuadros, W, H, T = pickle.load(fh)
        frames = []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            frames.append(im)
    else:
        ex = {}
        cajas, cuadros, W, H, T, frames = EP.detecta_clip(yolo, cap, cv2, extra=ex)
        if not T:
            """UN VIDEO DE 0 CUADROS ES UN VIDEO ROTO, no un clip vacio (2026-10-09). Escribir su
            `revision_` lo daba por hecho y nadie lo volvia a bajar. Sin `det_` ni `revision_`,
            la vuelta siguiente lo ve pendiente y `lotes.legible` lo manda bajar otra vez."""
            cap.release()
            print("%-40s VIDEO ILEGIBLE (0 cuadros): no se escribe nada" % cid[:40], flush=True)
            return
        with open(ca.with_name(ca.name + ".part"), "wb") as fh:
            pickle.dump((cajas, cuadros, W, H, T), fh)
        os.replace(ca.with_name(ca.name + ".part"), ca)
        """(2026-10-13) A un `.part` y luego `os.replace`: un corte a mitad no deja un fichero roto que cuente como hecho."""
        _savez_atomico(
            PRUEBAS / ("detx_%s.npz" % cid),
            conf=(np.concatenate(ex["conf"]) if ex.get("conf") else np.zeros(0)).astype("float16"),
            kp=(np.concatenate(ex["kp"]) if ex.get("kp") else np.zeros((0, 17, 3))).astype("float16"))
        """LO QUE YOLO VE Y ANTES SE TIRABA (2026-10-08): la confianza y los 17 puntos de CADA
        persona detectada, arbitro y publico incluidos, en el orden de `cajas`. Con esto la
        pareja se puede volver a elegir sin GPU con algo mas que cajas; sin esto, cambiar la
        seleccion obligaba a repetir la etapa 1. ~50 KB por clip."""
    cap.release()
    gris = [cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype("float32")
            for f in frames]
    r = EP.procesa(cajas, cuadros, W, H, T, yolo=yolo_redet or yolo, frames=frames, gris=gris)
    _savez_atomico(PRUEBAS / ("revision_%s.npz" % cid),
                        B=r["B"], Be=r["Be"], pres=r["pres"], tipo=r["tipo"], mask=r["mask"],
                        W=W, H=H, T=T, motivo=r["motivo"], video=ruta,
                        tramos=np.array(r["tramos"], dtype="int32").reshape(-1, 2),
                        ambos=r["ambos"], fps=np.float64(fps_clip))
    print("%-40s ambos %5.1f %%  redet %3d  planos fuera %3d  fundidos %s  recortados %3d" % (
        cid[:40], 100 * r["ambos"], r["nredetectados"], r["fuera"],
        ("c%d-c%d" % (r["fundidos"][0][0], r["fundidos"][0][1])) if r["fundidos"] else "-",
        r["recortados"]))


def _pinta(im, kp, sc, col):
    """QUE HACE: pinta un esqueleto, saltando los puntos sin confianza o con coordenada no finita.

    Hay puntos con confianza por encima de 0,3 y coordenada NaN: filtrar solo por confianza casca o
    dibuja basura. Es un fallo ya visto en este proyecto.
    """
    def val(i):
        return sc[i] >= CONF_PINTA and np.isfinite(kp[i, 0]) and np.isfinite(kp[i, 1])

    for a, b in HUESOS:
        if val(a) and val(b):
            cv2.line(im, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[b, 0]), int(kp[b, 1])), col, 2)
    for a, b in PULGARES:
        if val(a) and val(b):
            cv2.line(im, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[b, 0]), int(kp[b, 1])),
                     (60, 240, 255), 2)
    for i in MANOS:
        if val(i):
            cv2.circle(im, (int(kp[i, 0]), int(kp[i, 1])), 1, (60, 240, 255), -1)
    for i in range(0, 21):
        if val(i):
            cv2.circle(im, (int(kp[i, 0]), int(kp[i, 1])), 3, col, -1)


HOLGURA_KP = 0.05
"""Holgura sobre la caja de YOLO para aceptar un keypoint como propio. La caja de YOLO es ajustada,
asi que se perdona un 5 % del ancho y del alto por el borde, y nada mas."""


def _fuera_de_caja(kp, sc, caja_xywh):
    """QUE HACE: pone a cero la confianza de los keypoints que caen fuera de SU propia caja.

    QUE AVERIA PARA, y estaba ya medida en `capas_limpias.limpia_keypoints`: el estimador de pose
    coloca keypoints ENCIMA DEL ARBITRO cuando este tapa al tirador. Un punto asi tiene confianza
    alta y es de otra persona, asi que envenena todo lo que se construya despues: el reparto de la
    mascara densa lo usa como semilla, el eje del brazo se va con el, y la pose inventada lo hereda
    por interpolacion.

    LA CORRECCION ES DE CONSTRUCCION, NO DE UMBRAL: un keypoint fuera de la caja de deteccion de su
    propia persona es espurio por definicion, porque la caja es lo que delimita a esa persona.

    POR QUE ESTABA AQUI SIN APLICAR: la funcion existia en `capas_limpias`, que es la ruta de las
    capas densas, pero NO en esta, que es la ruta que produce las poses que se entregan y se guardan
    en el NPZ. La tabla de checks lo cazo: **10.140 keypoints entregados caian fuera de su caja**.
    """
    x, y, w, h = [float(v) for v in caja_xywh]
    hx, hy = HOLGURA_KP * w, HOLGURA_KP * h
    dentro = ((kp[:, 0] >= x - hx) & (kp[:, 0] <= x + w + hx)
              & (kp[:, 1] >= y - hy) & (kp[:, 1] <= y + h + hy))
    return np.where(dentro, sc, 0.0)


PREPROC_GPU = os.environ.get("FENCINGLY_PREPROC_GPU", "0") == "1"
"""EL PREPROCESO DE SAPIENS (recorte con `grid_sample`, redimension, normalizado) EN LA GPU. El
procesador de HF es torch puro y sigue el `device` de la entrada; sin `device=` lo hace en CPU,
~13,7 ms por recorte medido en CPU (2026-10-08), y luego pasa el resultado a la GPU.

**MEDIDO Y RECHAZADO (2026-10-08)**, con la puerta de siempre (p50 < 0,02 px, max < 0,5 px), en 2
clips y 367.524 puntos usables: p50 0,009 px pero **max 166,5 px**, y velocidad **1,03x** (GPU
compartida con la corrida). Es el mismo fallo que el lote: un cambio minimo de aritmetica en un
punto ambiguo lo manda a otro sitio. Se queda apagado; el interruptor queda para la medida."""

LOTE_POSE = int(os.environ.get("FENCINGLY_LOTE_POSE", "1"))
"""CUANTOS RECORTES POR PASADA DEL MODELO DE POSE. **Vale 1: el knob se midio y se RECHAZO.**

EL PROBLEMA QUE PRETENDIA ARREGLAR, medido. Esta funcion hace **una pasada por cuadro**, con una o
dos cajas dentro: en un clip de 325 cuadros son 325 llamadas de una o dos imagenes de 384x288.
Medido en tres clips en una RTX 3060 sin compartir: la fase 2 cuesta **29,99 s por clip** contra
**12,35 s** de la fase 1 entera. O sea que **la pose cuesta mas que YOLO**, al reves de lo que
dice la tabla del plan, que solo habia medido la fase 1 y de ahi salia el "83 % es YOLO".

EL CRITERIO, DECLARADO ANTES DE MEDIR: lotear tiene que ser un **no-op sobre el resultado** --
p50 del desplazamiento < 0,02 px y **maximo < 0,5 px**, que es 1/4 del error de pose publicado
(2,26 px). Todos los recortes se llevan a 384x288, asi que la forma no cambia y el unico camino
para que el numero se mueva es que la tarjeta elija otro nucleo segun el tamano del lote.

LO MEDIDO, en `AbandonedBiodegradableCoqui` (325 cuadros, 1920x1080):

    lote  1  31,87 s  ---     dos pasadas identicas: p50 0,0000  max 0,0000 px
    lote  8  28,14 s  1,12x   p50 0,0021  p95 0,0059
    lote 16  27,26 s  1,17x   p50 0,0078  p95 0,0241  **max 129,92 px** (puntos usables)
    lote 32  27,50 s  1,15x   igual que 16

**Se rechaza.** No por el maximo solamente: por los dos juntos. Gana **1,17x**, no el 1,3-1,8x que
el plan esperaba -- o sea que el cuello no era la llamada por cuadro sino el preproceso en CPU y la
decodificacion -- y a cambio mueve un keypoint usable 130 px. Con dos pasadas del mismo lote 1 el
desplazamiento es **exactamente cero**, asi que no es indeterminismo del modelo: es el lote.

Y UN HALLAZGO DE PASO, que no es sobre el lote: el modelo devuelve **coordenadas finitas y
absurdas** para puntos de confianza casi nula, hasta **8.553 px** fuera de un cuadro de 1.920 de
ancho. No hacen dano porque todo lo que consume el NPZ filtra por confianza, pero quien compare
dos salidas sin filtrar por confianza medira eso y creera que ha medido otra cosa.

Poner `FENCINGLY_LOTE_POSE=16` reactiva el lote para volver a medirlo; el codigo se queda porque
en la GPU de Colab, con otro preproceso y otro nucleo, el reparto puede ser distinto y entonces
hay que volver a pasarle esta misma puerta -- no darlo por bueno."""


TUBERIA_POSE = os.environ.get("FENCINGLY_POSE_TUBERIA", "0") == "1"
HILOS_PREPROC = int(os.environ.get("FENCINGLY_POSE_HILOS", "3"))
POR_CUADRO = os.environ.get("FENCINGLY_POSE_POR_CUADRO", "0") == "1"
"""O2b (2026-10-13): con la tuberia, las DOS cajas de un cuadro se preprocesan en UNA llamada (la imagen
se pasa a float una vez y no dos) y luego van al modelo DE UNA EN UNA, como siempre: la entrada del
modelo es la misma. Medido en CPU en 159 cuadros reales con dos cajas: `pixel_values` identicos, 172 ->
100 ms por cuadro. Entra solo con la prueba de identidad de las poses."""
TORCH_HILOS = int(os.environ.get("FENCINGLY_TORCH_HILOS", "0"))
"""O9 (2026-10-13): hilos de torch en CPU de este proceso (0 = lo de siempre, todos los nucleos). El
preproceso de Sapiens va a 38,7 ms por recorte con 12 hilos y a 14,3 con 1, porque los hilos se pisan;
`pixel_values` identicos con 12, 1, 2 y 4 hilos en 122 cuadros. Entra solo con la prueba de identidad."""


def _hilos_torch(torch):
    if TORCH_HILOS > 0:
        torch.set_num_threads(TORCH_HILOS)
"""O2 de `CORRIDA_FINAL_2026-10-13.md` §10 (2026-10-13): LA MISMA CUENTA, EN TUBERIA. Un hilo lee los
cuadros y `HILOS_PREPROC` hilos hacen el preproceso de Sapiens (CPU) mientras la GPU pasa el recorte
anterior. Cada recorte se preprocesa con la MISMA llamada (`proc(images=[rgb], boxes=[caja])`), la
pasada y el postproceso van en el mismo orden y de uno en uno, como con `LOTE_POSE = 1`: el
resultado tiene que salir IDENTICO bit a bit. Solo entra si lo sale (criterio declarado antes de
medir: diferencia maxima 0,0 px en 20 clips, y >= 1,2 veces mas rapida). Apagada por defecto."""


def _poses_tuberia(proc, mod, torch, video, B, pres, mask, W, H, T, guarda_frames):
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor
    KP = np.full((T, 2, N_PTS, 2), np.nan, dtype="float32")
    SC = np.zeros((T, 2, N_PTS), dtype="float32")
    frames = []
    cola = queue.Queue(maxsize=64)
    FIN = object()
    error = []

    def prepara_cuadro(rgb, cajas):
        if POR_CUADRO:
            return proc(images=[rgb], boxes=[[c[0] for c in cajas]], return_tensors="pt")
        return proc(images=[rgb] * len(cajas), boxes=cajas, return_tensors="pt")
    """UNA PASADA POR CUADRO, CON SUS UNA O DOS CAJAS, como `vacia()` con `LOTE_POSE = 1` (que vacia
    al acabar cada cuadro, no cada caja). La primera version (13/10) pasaba las cajas de una en una:
    lote 1 en vez de 2 cambia el nucleo de la GPU, y la prueba de identidad dio p50 0,011 px y max
    936 px en puntos usables, 0 clips de 20 identicos (14/10). Por eso no se puede partir el cuadro."""

    def lector(ex):
        try:
            cap = cv2.VideoCapture(video)
            for k in range(T):
                ok, im = cap.read()
                if not ok:
                    break
                if guarda_frames:
                    frames.append(im)
                if not mask[k]:
                    continue
                rgb = None
                cajas, lados = [], []
                for lado in (0, 1):
                    if not pres[k, lado]:
                        continue
                    b = B[k, lado]
                    if not (np.isfinite(b).all() and b[2] > b[0] and b[3] > b[1]):
                        continue
                    if rgb is None:
                        rgb = np.ascontiguousarray(im[:, :, ::-1])
                    x, y = float(b[0] * W), float(b[1] * H)
                    cajas.append([[x, y, float(b[2] * W) - x, float(b[3] * H) - y]])
                    lados.append(lado)
                if cajas:
                    cola.put((k, lados, cajas, ex.submit(prepara_cuadro, rgb, cajas)))
            cap.release()
        except BaseException as e:
            error.append(e)
        finally:
            cola.put(FIN)

    with ThreadPoolExecutor(HILOS_PREPROC) as ex:
        th = threading.Thread(target=lector, args=(ex,), daemon=True)
        th.start()
        while True:
            it = cola.get()
            if it is FIN:
                break
            k, lados, cajas, fut = it
            e = fut.result()
            with torch.inference_mode():
                e = {q: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda"))
                     for q, v in e.items()}
                o = mod(**e)
                r = proc.post_process_pose_estimation(o, boxes=cajas, threshold=None)
            for j, lado in enumerate(lados):
                rr = r[j][0] if isinstance(r[j], list) else r[j]
                kp = np.asarray(rr["keypoints"].cpu(), dtype="float32")
                sc = np.asarray(rr["scores"].cpu(), dtype="float32")
                n = min(len(kp), N_PTS)
                KP[k, lado, :n] = kp[:n]
                SC[k, lado, :n] = _fuera_de_caja(kp[:n], sc[:n], cajas[j][0])
        th.join()
    if error:
        raise error[0]
    return KP, SC, frames


def poses_del_clip(proc, mod, torch, video, B, pres, mask, W, H, T, lote=None,
                   guarda_frames=True):
    """QUE HACE: la pose de Sapiens 2 en las cajas presentes, y devuelve tambien los fotogramas.

    Se separa de la pasada de dibujar porque para INVENTAR las poses que faltan hacen falta la
    anterior y la siguiente, y eso no se puede hacer cuadro a cuadro mientras se pinta.

    LOTEA POR RECORTE Y NO POR CUADRO, y eso importa: los cuadros traen una o dos cajas segun si
    los dos tiradores estan presentes, asi que agrupar "de N en N cuadros" daria lotes de tamano
    variable entre N y 2N. Se agrupan los RECORTES, que es la unidad de trabajo real.

    UNA IMAGEN POR TIRADOR Y UNA CAJA DENTRO DE CADA UNA, tambien dentro del lote. No es cosmetica:
    `matting` se confunde con dos personas en el mismo recorte, y el post-proceso devuelve una
    lista por imagen, asi que meter las dos cajas en una sola imagen da lista vacia.
    """
    lote = int(lote or LOTE_POSE)
    if TUBERIA_POSE and lote == 1 and not PREPROC_GPU:
        return _poses_tuberia(proc, mod, torch, video, B, pres, mask, W, H, T, guarda_frames)
    KP = np.full((T, 2, N_PTS, 2), np.nan, dtype="float32")
    SC = np.zeros((T, 2, N_PTS), dtype="float32")
    cap = cv2.VideoCapture(video)
    frames = []
    pend_img, pend_caja, pend_dest = [], [], []

    def vacia():
        if not pend_img:
            return
        with torch.inference_mode():
            e = (proc(images=pend_img, boxes=pend_caja, return_tensors="pt", device="cuda")
                 if PREPROC_GPU else proc(images=pend_img, boxes=pend_caja, return_tensors="pt"))
            e = {q: (v.to("cuda").half() if v.is_floating_point() else v.to("cuda"))
                 for q, v in e.items()}
            o = mod(**e)
            r = proc.post_process_pose_estimation(o, boxes=pend_caja, threshold=None)
        for j, (k, lado) in enumerate(pend_dest):
            rr = r[j][0] if isinstance(r[j], list) else r[j]
            kp = np.asarray(rr["keypoints"].cpu(), dtype="float32")
            sc = np.asarray(rr["scores"].cpu(), dtype="float32")
            n = min(len(kp), N_PTS)
            KP[k, lado, :n] = kp[:n]
            SC[k, lado, :n] = _fuera_de_caja(kp[:n], sc[:n], pend_caja[j][0])
        pend_img.clear()
        pend_caja.clear()
        pend_dest.clear()

    for k in range(T):
        ok, im = cap.read()
        if not ok:
            break
        if guarda_frames:
            frames.append(im)
        """LOS FOTOGRAMAS SOLO SE GUARDAN SI ALGUIEN LOS VA A PINTAR.

        Se acumulaban SIEMPRE, y son la mayor reserva de memoria del proceso: un clip de 325
        cuadros a 1920x1080 son **~2 GB** de `uint8` vivos hasta el final de la funcion, y el
        unico consumidor es el mp4 de revision, que en la ruta del pipeline esta apagado. Medido:
        el proceso de la fase 2 ocupaba **5,1 GB** de RAM. Sin guardarlos, en 32 GB caben varios
        procesos de extraccion en paralelo, que es lo que de verdad acelera la corrida -- la GPU
        esta al 63 %, o sea que el cuello es el preproceso en CPU y se ataca con paralelismo, no
        con un lote mayor (ver `LOTE_POSE`, medido y rechazado)."""
        if not mask[k]:
            continue
        rgb = None
        for lado in (0, 1):
            if not pres[k, lado]:
                continue
            b = B[k, lado]
            if not (np.isfinite(b).all() and b[2] > b[0] and b[3] > b[1]):
                continue
            if rgb is None:
                rgb = np.ascontiguousarray(im[:, :, ::-1])
            x, y = float(b[0] * W), float(b[1] * H)
            pend_img.append(rgb)
            pend_caja.append([[x, y, float(b[2] * W) - x, float(b[3] * H) - y]])
            pend_dest.append((k, lado))
        if len(pend_img) >= lote:
            vacia()
    vacia()
    cap.release()
    return KP, SC, frames


PUNTOS_MIN = 20
"""Puntos utiles del cuerpo, sin cara, por debajo de los cuales la pose no vale. La mediana global
medida sobre 4.892 poses es 63 de 63, y las poses que salen de una tira en el borde del cuadro dan
mediana 20. Vease `extrae_pareja.PROP_MIN`."""


def tira_poses_vacias(KP, SC, pres, mask, T):
    """QUE HACE: marca como NO medida la pose de la que apenas sale ningun punto del cuerpo.

    Es la comprobacion DIRECTA de lo que la puerta de proporcion solo puede sospechar antes de correr
    el modelo. Sin esto, una pose vacia se convierte en el ancla de una interpolacion, y entonces lo
    inventado tambien sale vacio: eso es lo que el usuario vio en `RichDefensiveBushbaby` c130.
    """
    n = 0
    for k in range(T):
        if not mask[k]:
            continue
        for lado in (0, 1):
            if not pres[k, lado]:
                continue
            ok = int(((SC[k, lado, :CARA_DESDE] >= CONF_PINTA)
                      & np.isfinite(KP[k, lado, :CARA_DESDE, 0])).sum())
            if ok < PUNTOS_MIN:
                pres[k, lado] = False
                KP[k, lado] = np.nan
                SC[k, lado] = 0.0
                n += 1
    return n


def recorta_por_poses(pres, mask):
    """QUE HACE: vuelve a recortar los extremos para que el tramo entregado empiece y acabe con las
    DOS poses MEDIDAS.

    QUE AVERIA PARA, y es un problema de ORDEN que se vio en la galeria: `extrae_pareja._recorta`
    garantiza que el primer y el ultimo cuadro entregado tienen las dos CAJAS medidas, pero corre
    antes de que exista la pose. Despues, `tira_poses_vacias` puede quitar precisamente la pose del
    cuadro del borde por venir vacia, y entonces ese extremo se queda con una pose SOSTENIDA, que es
    justo lo que el usuario vio mal en el video y lo que el recorte venia a evitar.

    Medido en la galeria antes de este arreglo: **8 de 45 casos eran sostenidos**.

    Una pose sostenida no se puede validar de ninguna forma, porque no hay ancla al otro lado con la
    que comparar. Interpolar entre dos anclas si esta medido; sostener no. Asi que el extremo se
    recorta, que cuesta cuadros y no inventa nada.
    """
    ambas = pres.sum(axis=1) == 2
    ok = ambas & mask
    if not ok.any():
        return np.zeros_like(mask), 0
    a = int(np.argmax(ok))
    b = int(len(ok) - 1 - np.argmax(ok[::-1]))
    nueva = mask.copy()
    nueva[:a] = False
    nueva[b + 1:] = False
    return nueva, int(mask.sum()) - int(nueva.sum())


def inventa_poses(KP, SC, pres, mask, T):
    """QUE HACE: rellena las poses que faltan, y deja dicho como se ha rellenado cada una.

    DOS CASOS, y no son lo mismo:
      INTERIOR   hay pose conocida antes y despues -> se interpola punto a punto entre las dos
      BORDE      solo hay pose a un lado           -> se SOSTIENE la mas cercana, porque no hay
                                                      nada al otro lado y inventar direccion seria
                                                      inventar movimiento que no se ha visto

    LA CONFIANZA DE UNA POSE INVENTADA ES EL MINIMO de las dos anclas, no la media. Un punto que el
    modelo no vio bien en un extremo no se vuelve fiable por promediarlo con el otro, y quien lea
    esto despues tiene que poder tirar los puntos flojos con el mismo umbral que usa en los medidos.

    QUE DEVUELVE: `KP`, `SC` rellenos, `origen` (0 medido, 1 interpolado, 2 sostenido, -1 nada) y
    `hueco`, que es la SEPARACION EN CUADROS entre las dos anclas de las que sale cada pose
    inventada, y 0 en las medidas.

    POR QUE SE GUARDA EL LARGO DEL HUECO, y no se decide aqui un limite. El error de interpolar esta
    medido hasta 24 cuadros de hueco (25,7 % del torso) y de ahi para arriba **no hay medida**. Y se
    ve a ojo que se rompe: en `FirsthandSoreGnat` c116, con 66 cuadros de hueco, la pose inventada
    queda puesta donde no hay nadie, porque el tirador no se movio en linea recta durante mas de dos
    segundos (imagen en `02_validacion/poses_inventadas/013_...hueco66.png`).

    Se podria poner un tope aqui, pero **esa decision no es de la extraccion**: es la misma razon por
    la que `origen` y `tipo` se guardan en vez de filtrar. Quien entrene tiene que poder pesar
    distinto una pose interpolada de 1 cuadro y una de 66, y para eso necesita el numero, no un
    booleano que ya escogio alguien por el.
    """
    origen = np.where(pres, 0, -1).astype("int8")
    hueco = np.zeros_like(origen, dtype="int16")
    for lado in (0, 1):
        hay = np.nonzero(pres[:, lado] & mask)[0]
        if not len(hay):
            continue
        for k in range(T):
            if pres[k, lado] or not mask[k]:
                continue
            ant = hay[hay < k]
            pos = hay[hay > k]
            if len(ant) and len(pos):
                a, b = int(ant[-1]), int(pos[0])
                t = (k - a) / float(b - a)
                KP[k, lado] = (1 - t) * KP[a, lado] + t * KP[b, lado]
                SC[k, lado] = np.minimum(SC[a, lado], SC[b, lado])
                origen[k, lado] = 1
                hueco[k, lado] = b - a - 1
            else:
                anc = int(ant[-1]) if len(ant) else int(pos[0])
                KP[k, lado] = KP[anc, lado]
                SC[k, lado] = SC[anc, lado]
                origen[k, lado] = 2
                hueco[k, lado] = abs(k - anc)
    return KP, SC, origen, hueco


def valida_inventadas(KP, SC, pres, mask, T, paso=7):
    """QUE HACE: mide el error de inventar una pose, ESCONDIENDO poses que si se conocen.

    Es la unica forma de poner un numero a una pose inventada: se tapa un cuadro cuya pose esta
    medida, se interpola desde sus dos vecinos conocidos y se compara con la verdad. El error se da
    en px y tambien normalizado por el largo del torso, que es lo que permite comparar entre clips
    con encuadres distintos.

    La cara se excluye del error a proposito: son 245 de los 308 puntos y no se usan en el proyecto,
    asi que promediarlos taparia el error de lo que si importa.
    """
    errs, errs_rel = [], []
    for lado in (0, 1):
        hay = np.nonzero(pres[:, lado] & mask)[0]
        for i in range(1, len(hay) - 1, paso):
            a, k, b = int(hay[i - 1]), int(hay[i]), int(hay[i + 1])
            if b - a < 2:
                continue
            t = (k - a) / float(b - a)
            est = (1 - t) * KP[a, lado] + t * KP[b, lado]
            ok = ((SC[k, lado] >= CONF_PINTA) & np.isfinite(KP[k, lado, :, 0])
                  & np.isfinite(est[:, 0]))
            ok[CARA_DESDE:] = False
            if ok.sum() < 8:
                continue
            d = np.linalg.norm(est[ok] - KP[k, lado][ok], axis=1)
            errs.append(float(np.median(d)))
            if SC[k, lado, 5] >= CONF_PINTA and SC[k, lado, 9] >= CONF_PINTA:
                torso = float(np.linalg.norm(KP[k, lado, 5] - KP[k, lado, 9]))
                if torso > 1:
                    errs_rel.append(float(np.median(d)) / torso)
    if not errs:
        return None
    return {"n": len(errs), "px_p50": float(np.median(errs)),
            "px_p90": float(np.percentile(errs, 90)),
            "rel_p50": float(np.median(errs_rel)) if errs_rel else float("nan"),
            "rel_p90": float(np.percentile(errs_rel, 90)) if errs_rel else float("nan")}


def fase2(clips, video=True):
    """QUE HACE: pose, invencion de las que faltan, validacion escondiendo, y el video.

    `video=False` NO escribe el mp4 de revision, y esa es la unica diferencia: los `poses_*.npz`
    salen identicos. Existe porque el mp4 es una **ayuda de verificacion, no un producto**, y en
    una corrida en masa cuesta lo que nadie quiere pagar: medido sobre los que ya hay, **9,4 a
    17,1 MB por clip**, mas el tiempo de codificar, mas tener `frames` entero en RAM hasta el
    final. Sobre 1.765 clips son ~20 GB de artefacto que se borraria despues.

    El defecto de tenerlo siempre puesto no era el tamano, era que **no habia interruptor**: la
    unica forma de no escribirlo era no llamar a la funcion.
    """
    import torch
    from transformers import Sapiens2ForPoseEstimation, Sapiens2ImageProcessor
    _hilos_torch(torch)
    snap = RU.hf_modelo("sapiens2-pose-0.4b")
    proc = Sapiens2ImageProcessor.from_pretrained(str(snap))
    proc.size = RES_POSE
    mod = Sapiens2ForPoseEstimation.from_pretrained(str(snap), dtype=torch.float16).to("cuda").eval()
    SALIDA.mkdir(parents=True, exist_ok=True)
    for cid in clips:
        try:
            _fase2_uno(cid, proc, mod, torch, video)
        except Exception as ex:
            """UN CLIP QUE CASCA NO SE LLEVA EL LOTE (2026-10-09). Antes la excepcion salia
            del bucle y los clips que quedaban del lote no se procesaban NUNCA: en cada vuelta
            volvia a cascar el mismo. Medido en la noche de anotados: 3 clips sin una sola
            deteccion (`planos.marca_primer_plano` con un array 1-D) dejaron **9 clips buenos**
            sin etapa 4, y el trabajador se paro creyendo que ya no habia nada que hacer."""
            print("%-40s ETAPA 4 FALLA %s: %s" % (cid[:40], type(ex).__name__, str(ex)[:120]),
                  flush=True)


def _fase2_uno(cid, proc, mod, torch, video):
    z = np.load(PRUEBAS / ("revision_%s.npz" % cid), allow_pickle=True)
    B, pres, tipo, mask = z["B"], z["pres"], z["tipo"], z["mask"]
    Be = z["Be"] if "Be" in z.files else B
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    fps_clip = float(z["fps"]) if "fps" in z.files else 0.0
    motivo, ruta_video = str(z["motivo"]), str(z["video"])
    KP, SC, frames = poses_del_clip(proc, mod, torch, ruta_video, B, pres, mask, W, H, T,
                                    guarda_frames=video)
    nvac = tira_poses_vacias(KP, SC, pres, mask, T)
    mask, nrec2 = recorta_por_poses(pres, mask)
    val = valida_inventadas(KP, SC, pres, mask, T)
    KP, SC, origen, hueco = inventa_poses(KP, SC, pres, mask, T)
    ninv = int(((origen == 1) | (origen == 2)).sum())
    dobles = int(((origen[mask] >= 0).sum(axis=1) == 2).sum())
    util = int(mask.sum())
    """`filtro_vacias` dice que ESTE fichero se escribio con el filtro de poses vacias puesto, y
    cuantas se tiraron. Sin esa marca no se puede juzgar un NPZ: los escritos antes de que el
    filtro existiera tienen poses medidas de 0 a 8 puntos utiles dentro, y un check que no
    distinga los dos casos acusa a la tuberia de un fallo que ya no comete. Es el mismo principio
    que `origen` y `tipo`: el dato lleva escrito como se obtuvo."""
    _savez_atomico(PRUEBAS / ("poses_%s.npz" % cid),
                        KP=KP, SC=SC, origen=origen, hueco=hueco,
                        B=B, Be=Be, pres=pres, mask=mask,
                        W=W, H=H, T=T, motivo=motivo, video=ruta_video,
                        filtro_vacias=np.array(nvac, dtype="int32"),
                        puntos_min=np.array(PUNTOS_MIN, dtype="int32"),
                        fps=np.float64(fps_clip), ambos=z["ambos"] if "ambos" in z.files
                        else np.float64(float("nan")))
    print("%-34s con LAS DOS poses %4d de %4d utilizables (%.1f %%), inventadas %d, "
          "poses vacias tiradas %d, recortados por pose %d" % (
              cid[:34], dobles, util, 100.0 * dobles / max(util, 1), ninv, nvac, nrec2))
    if val:
        print("   error de inventar, escondiendo poses conocidas: mediana %.2f px "
              "(%.1f %% del torso), p90 %.2f px, n=%d" % (
                  val["px_p50"], 100 * val["rel_p50"], val["px_p90"], val["n"]))

    if not video:
        return
    cap = cv2.VideoCapture(ruta_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    out = cv2.VideoWriter(str(SALIDA / ("%s.mp4" % cid)),
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for k in range(min(T, len(frames))):
        vis = frames[k].copy()
        if mask[k]:
            for lado in (0, 1):
                if origen[k, lado] < 0:
                    continue
                col = COL_LADO[lado]
                inv = origen[k, lado] > 0
                if inv:
                    """Las inventadas van en violeta y con la etiqueta puesta, para que nadie
                    las confunda con dato medido al mirar el video."""
                    col = VIOLETA
                _pinta(vis, KP[k, lado], SC[k, lado], col)
                b = Be[k, lado] if inv else B[k, lado]
                if np.isfinite(b).all():
                    cv2.rectangle(vis, (int(b[0] * W), int(b[1] * H)),
                                  (int(b[2] * W), int(b[3] * H)), col, 1 if inv else 2)
                et = ["IZQUIERDA", "DERECHA"][lado]
                if origen[k, lado] == 1:
                    et += " INVENTADA interpolada"
                elif origen[k, lado] == 2:
                    et += " INVENTADA sostenida"
                elif tipo[k, lado] == 3:
                    et += " redetectada"
                if np.isfinite(b).all():
                    cv2.putText(vis, et, (int(b[0] * W), max(14, int(b[1] * H) - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        else:
            cv2.rectangle(vis, (0, 36), (W, 72), (0, 0, 160), -1)
            cv2.putText(vis, "FUERA: primer plano o repeticion, este cuadro no se usa",
                        (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.rectangle(vis, (0, 0), (W, 34), (12, 12, 18), -1)
        cv2.putText(vis, "%s c%d/%d  %s" % (cid[:24], k, T - 1, motivo[:70]),
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1)
        out.write(vis)
    out.release()
    print("   video -> %s" % (SALIDA / ("%s.mp4" % cid)))


if __name__ == "__main__":
    fase = sys.argv[1]
    cl = [x for x in sys.argv[2:] if not x.startswith("--")]
    if fase == "fase1":
        fase1(cl)
    elif fase == "fase2":
        fase2(cl, video="--sin-video" not in sys.argv)
    else:
        raise SystemExit("usa fase1 o fase2 [--sin-video]")
